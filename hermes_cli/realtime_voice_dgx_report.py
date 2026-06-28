"""Compare DGX Spark KAME voice evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.realtime_voice_smoke_report import (
    load_realtime_voice_smoke_report_runs,
    summarize_realtime_voice_smoke_report_runs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize DGX Spark KAME voice evaluation artifacts")
    parser.add_argument("--artifact-dir", required=True, help="Root artifact directory from dgx_spark_gemma4_voice_eval.sh")
    parser.add_argument("--output", help="JSON report path")
    parser.add_argument("--markdown-output", help="Markdown report path")
    parser.add_argument("--oracle-probe", help="Override oracle probe JSON path")
    parser.add_argument("--cartesia-alpha", help="Override Cartesia alpha evidence directory")
    parser.add_argument("--local-speech-alpha", help="Override local speech alpha evidence directory")
    parser.add_argument(
        "--local-within-ratio",
        type=float,
        default=1.25,
        help="Prefer local speech when its latency is within this ratio of Cartesia",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_dir = Path(args.artifact_dir).expanduser()
    report = build_dgx_voice_recommendation_report(
        artifact_dir=artifact_dir,
        oracle_probe=Path(args.oracle_probe).expanduser() if args.oracle_probe else None,
        cartesia_alpha=Path(args.cartesia_alpha).expanduser() if args.cartesia_alpha else None,
        local_speech_alpha=Path(args.local_speech_alpha).expanduser() if args.local_speech_alpha else None,
        local_within_ratio=float(args.local_within_ratio or 1.25),
    )
    if args.output:
        _write_text(Path(args.output).expanduser(), json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown_output:
        _write_text(Path(args.markdown_output).expanduser(), render_dgx_voice_recommendation_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_dgx_voice_recommendation_report(
    *,
    artifact_dir: Path,
    oracle_probe: Path | None = None,
    cartesia_alpha: Path | None = None,
    local_speech_alpha: Path | None = None,
    local_within_ratio: float = 1.25,
) -> dict[str, Any]:
    oracle = summarize_oracle_probe(oracle_probe or artifact_dir / "oracle-gemma4-probe.json")
    cartesia = summarize_alpha_track("cartesia", cartesia_alpha or artifact_dir / "cartesia-alpha")
    local = summarize_alpha_track("local_speech", local_speech_alpha or artifact_dir / "local-speech-alpha")
    recommendation = choose_voice_track_recommendation(
        oracle=oracle,
        cartesia=cartesia,
        local_speech=local,
        local_within_ratio=local_within_ratio,
    )
    return {
        "kind": "dgx_spark_kame_voice_recommendation",
        "artifact_dir": str(artifact_dir),
        "recommendation": recommendation,
        "tracks": {
            "oracle": oracle,
            "cartesia": cartesia,
            "local_speech": local,
        },
    }


def summarize_oracle_probe(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "track": "oracle",
            "status": "missing",
            "path": str(path),
            "issues": ["oracle probe was not collected"],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "track": "oracle",
            "status": "invalid",
            "path": str(path),
            "issues": [f"oracle probe is not readable JSON: {exc}"],
        }
    if not isinstance(payload, Mapping):
        return {
            "track": "oracle",
            "status": "invalid",
            "path": str(path),
            "issues": ["oracle probe JSON must be an object"],
        }
    issues: list[str] = []
    if payload.get("ok") is not True:
        issues.append(str(payload.get("error") or "oracle probe did not pass"))
    elapsed_ms = _number(payload.get("elapsed_ms"))
    tokens_per_second = _number(payload.get("tokens_per_second"))
    if elapsed_ms is not None and elapsed_ms > 5000:
        issues.append(f"oracle simple response {elapsed_ms:g}ms exceeds 5000ms target")
    if tokens_per_second is not None and tokens_per_second < 20:
        issues.append(f"oracle decode {tokens_per_second:g} tok/s is below 20 tok/s target")
    return {
        "track": "oracle",
        "status": "passed" if not issues else "failed",
        "path": str(path),
        "model": payload.get("model"),
        "base_url": payload.get("base_url"),
        "elapsed_ms": elapsed_ms,
        "tokens_per_second": tokens_per_second,
        "issues": issues,
    }


def summarize_alpha_track(name: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "track": name,
            "status": "missing",
            "path": str(path),
            "issues": [f"{name} alpha evidence was not collected"],
        }
    try:
        runs = load_realtime_voice_smoke_report_runs(path)
    except Exception as exc:
        return {
            "track": name,
            "status": "invalid",
            "path": str(path),
            "issues": [f"{name} alpha evidence is not readable: {exc}"],
        }
    if not runs:
        return {
            "track": name,
            "status": "missing",
            "path": str(path),
            "issues": [f"{name} alpha evidence has no JSON runs"],
        }
    entries = [entry for _label, run_entries in runs for entry in run_entries if isinstance(entry, Mapping)]
    failed_entries = [
        str(entry.get("kind") or "entry")
        for entry in entries
        if str(entry.get("kind") or "") != "manifest" and entry.get("ok") is not True
    ]
    summary = summarize_realtime_voice_smoke_report_runs(runs)
    latency_ms = summary.get("latency_ms") if isinstance(summary.get("latency_ms"), Mapping) else {}
    first_audio_p50 = _latency_p50(
        latency_ms,
        (
            "speech_end_to_first_audio",
            "final_transcript_to_first_audio",
            "speech_end_to_local_first_audio",
        ),
    )
    partial_p50 = _latency_p50(latency_ms, ("audio_to_partial_transcript",))
    barge_in_p50 = _latency_p50(latency_ms, ("barge_in_ack", "barge_in_confirmed_to_playback_stopped"))
    issues = [f"{len(failed_entries)} failed evidence entries: {', '.join(failed_entries[:5])}"] if failed_entries else []
    if first_audio_p50 is None:
        issues.append("missing first-audio latency evidence")
    return {
        "track": name,
        "status": "passed" if not issues else "failed",
        "path": str(path),
        "runs": len(runs),
        "entries": len(entries),
        "latency_ms": {
            "first_audio_p50": first_audio_p50,
            "partial_transcript_p50": partial_p50,
            "barge_in_p50": barge_in_p50,
        },
        "kame_routes": summary.get("kame_routes"),
        "issues": issues,
    }


def choose_voice_track_recommendation(
    *,
    oracle: Mapping[str, Any],
    cartesia: Mapping[str, Any],
    local_speech: Mapping[str, Any],
    local_within_ratio: float = 1.25,
) -> dict[str, Any]:
    if oracle.get("status") != "passed":
        return {
            "decision": "fix_oracle_first",
            "reason": "Track A oracle evidence must pass before judging voice frontends.",
        }
    cartesia_ok = cartesia.get("status") == "passed"
    local_ok = local_speech.get("status") == "passed"
    cartesia_latency = _track_first_audio_latency(cartesia)
    local_latency = _track_first_audio_latency(local_speech)
    if cartesia_ok and local_ok and cartesia_latency is not None and local_latency is not None:
        if local_latency <= cartesia_latency * max(local_within_ratio, 1.0):
            return {
                "decision": "prefer_local_speech",
                "reason": (
                    f"Track C first-audio p50 {local_latency:g}ms is within "
                    f"{local_within_ratio:g}x of Track B {cartesia_latency:g}ms."
                ),
            }
        return {
            "decision": "keep_cartesia_baseline",
            "reason": (
                f"Track C first-audio p50 {local_latency:g}ms is slower than "
                f"{local_within_ratio:g}x Track B {cartesia_latency:g}ms."
            ),
        }
    if cartesia_ok:
        return {
            "decision": "keep_cartesia_baseline",
            "reason": "Track B passed and Track C is missing or not yet usable.",
        }
    if local_ok:
        return {
            "decision": "prefer_local_speech",
            "reason": "Track C passed and Track B is missing or not yet usable.",
        }
    return {
        "decision": "collect_voice_frontend_evidence",
        "reason": "Neither Track B nor Track C has passing voice frontend evidence.",
    }


def render_dgx_voice_recommendation_markdown(report: Mapping[str, Any]) -> str:
    recommendation = report.get("recommendation") if isinstance(report.get("recommendation"), Mapping) else {}
    tracks = report.get("tracks") if isinstance(report.get("tracks"), Mapping) else {}
    lines = [
        "# DGX Spark KAME Voice Recommendation",
        "",
        f"Decision: `{recommendation.get('decision', 'unknown')}`",
        "",
        str(recommendation.get("reason") or ""),
        "",
        "## Tracks",
        "",
    ]
    for key in ("oracle", "cartesia", "local_speech"):
        track = tracks.get(key) if isinstance(tracks.get(key), Mapping) else {}
        lines.append(f"- `{key}`: {track.get('status', 'missing')}")
        issues = track.get("issues")
        if isinstance(issues, Sequence) and issues:
            lines.append(f"  Issues: {'; '.join(str(issue) for issue in issues)}")
        latency = track.get("latency_ms") if isinstance(track.get("latency_ms"), Mapping) else {}
        if latency:
            lines.append(
                "  Latency: "
                f"first_audio_p50={latency.get('first_audio_p50')}ms, "
                f"partial_p50={latency.get('partial_transcript_p50')}ms, "
                f"barge_in_p50={latency.get('barge_in_p50')}ms"
            )
    return "\n".join(lines).rstrip() + "\n"


def _latency_p50(latency_ms: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        metric = latency_ms.get(name)
        if not isinstance(metric, Mapping):
            continue
        parsed = _number(metric.get("p50"))
        if parsed is not None:
            return parsed
    return None


def _track_first_audio_latency(track: Mapping[str, Any]) -> float | None:
    latency = track.get("latency_ms")
    if not isinstance(latency, Mapping):
        return None
    return _number(latency.get("first_audio_p50"))


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
