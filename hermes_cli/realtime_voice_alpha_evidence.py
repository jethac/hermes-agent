"""Run repeated realtime voice alpha evidence checks."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from agent.realtime_voice_smoke_report import (
    ALPHA_REQUIRED_AUDIO_FIXTURES,
    load_realtime_voice_smoke_report,
    summarize_realtime_voice_smoke_report_runs,
    validate_realtime_voice_alpha_report_runs,
)
from agent.realtime_voice_errors import sanitize_realtime_voice_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect repeated Hermes realtime voice private-alpha evidence reports"
    )
    parser.add_argument(
        "--output-dir",
        default="./artifacts/realtime-voice-evidence",
        help="Directory where per-run JSON reports will be written",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of independent alpha evidence runs to collect",
    )
    parser.add_argument(
        "--prefix",
        default="realtime-voice-alpha",
        help="Report filename prefix",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting numeric suffix for report filenames",
    )
    parser.add_argument(
        "--audio-codec",
        choices=("webm_opus", "opus", "pcm16"),
        default="webm_opus",
        help="Codec for the required alpha audio fixtures",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing report files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_count = max(1, int(args.runs or 1))
    start_index = max(1, int(args.start_index or 1))
    output_dir = Path(args.output_dir).expanduser()

    report_paths = [
        output_dir / f"{args.prefix}-{index:03d}.json"
        for index in range(start_index, start_index + run_count)
    ]
    existing = [path for path in report_paths if path.exists()]
    if existing and not args.overwrite:
        print(
            "Realtime voice alpha evidence failed: report file already exists "
            f"({existing[0]}); pass --overwrite or choose another --start-index",
            file=sys.stderr,
        )
        return 1

    missing_fixtures = missing_required_audio_fixtures()
    if missing_fixtures:
        print(
            "Realtime voice alpha evidence failed: missing required audio fixture(s)",
            file=sys.stderr,
        )
        for fixture in missing_fixtures:
            print(f"  - {fixture}", file=sys.stderr)
        return 1

    try:
        from hermes_cli.doctor import _realtime_voice_smoke_config, run_doctor

        _realtime_voice_smoke_config()
    except Exception as exc:
        print(
            "Realtime voice alpha evidence failed: realtime voice smoke is not configured "
            f"({sanitize_realtime_voice_error(exc)})",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with managed_realtime_voice_sidecar_for_evidence():
            live_like_issue = realtime_voice_live_like_preflight_issue()
            if live_like_issue:
                print(
                    "Realtime voice alpha evidence failed: live-like realtime voice is not ready "
                    f"({live_like_issue})",
                    file=sys.stderr,
                )
                print_realtime_voice_live_setup_hint()
                return 1
            for ordinal, report_path in enumerate(report_paths, start=1):
                print(f"Realtime voice alpha evidence run {ordinal}/{run_count}: {report_path}")
                run_doctor(
                    Namespace(
                        fix=False,
                        ack=None,
                        realtime_voice=True,
                        realtime_voice_alpha=True,
                        realtime_voice_smoke=False,
                        realtime_voice_audio_fixture=None,
                        realtime_voice_audio_codec=args.audio_codec,
                        realtime_voice_tts_smoke=None,
                        realtime_voice_barge_in_smoke=None,
                        realtime_voice_report=str(report_path),
                    )
                )
                if not report_path.exists():
                    print(
                        f"Realtime voice alpha evidence failed: {report_path} was not written",
                        file=sys.stderr,
                    )
                    return 1
    except RuntimeError:
        return 1

    runs = [(str(path), load_realtime_voice_smoke_report(path)) for path in report_paths]
    issues = validate_realtime_voice_alpha_report_runs(runs, min_runs=run_count)
    if issues:
        print(f"Realtime voice alpha evidence failed: {len(issues)} issue(s)", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue.format()}", file=sys.stderr)
        return 1

    _print_summary(runs)
    return 0


def missing_required_audio_fixtures() -> list[str]:
    return [
        fixture
        for fixture in ALPHA_REQUIRED_AUDIO_FIXTURES
        if not Path(fixture).expanduser().is_file()
    ]


@contextmanager
def managed_realtime_voice_sidecar_for_evidence() -> Iterator[None]:
    """Start the configured managed loopback sidecar for CLI evidence runs."""

    proc = None
    try:
        from hermes_cli import web_server

        realtime = web_server._realtime_voice_config_dict()
        base_url = web_server._realtime_voice_sidecar_base_url(realtime)
        if web_server._realtime_voice_should_autostart_sidecar(realtime, base_url):
            env_on_disk = web_server.load_env()
            token = web_server._realtime_voice_sidecar_token(realtime, env_on_disk)
            was_healthy = web_server._realtime_voice_sidecar_healthy(base_url, token=token)
            if not was_healthy:
                web_server._ensure_realtime_voice_sidecar(realtime)
                candidate = getattr(web_server, "_VOICE_SIDECAR_PROC", None)
                if candidate is not None and candidate.poll() is None:
                    proc = candidate
    except Exception as exc:
        print(
            "Realtime voice alpha evidence failed: managed sidecar is not ready "
            f"({sanitize_realtime_voice_error(exc)})",
            file=sys.stderr,
        )
        raise RuntimeError("managed realtime voice sidecar is not ready") from exc
    try:
        yield
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


def realtime_voice_live_like_preflight_issue() -> str:
    try:
        from hermes_cli import web_server

        status = web_server._realtime_voice_status_payload()
    except AttributeError:
        return ""
    except Exception as exc:
        return f"status unavailable: {sanitize_realtime_voice_error(exc)}"
    if not isinstance(status, dict):
        return "status unavailable"
    if status.get("enabled") is not True:
        return "disabled"
    unavailable = str(status.get("unavailable_reason") or "").strip()
    conversation_quality = status.get("conversation_quality")
    conversation_quality = conversation_quality if isinstance(conversation_quality, dict) else {}
    if status.get("available") is not True and unavailable:
        mode = str(conversation_quality.get("mode") or "unknown")
        reason = str(conversation_quality.get("reason") or "unknown")
        return f"{unavailable}; mode={mode}; reason={reason}"
    if conversation_quality.get("live_like") is not True:
        mode = str(conversation_quality.get("mode") or "unknown")
        reason = str(conversation_quality.get("reason") or "unknown")
        return (
            f"not_live_like; mode={mode}; reason={reason}; "
            "configure streaming STT/TTS or native S2S"
        )
    return ""


def print_realtime_voice_live_setup_hint() -> None:
    print("Portable live setup:", file=sys.stderr)
    print("  python -m hermes_cli.realtime_voice_profile --preset deepgram --apply", file=sys.stderr)
    print("  set DEEPGRAM_API_KEY=...", file=sys.stderr)
    print("  python -m hermes_cli.realtime_voice_deepgram_bridge --generate-token", file=sys.stderr)
    print("  python -m hermes_cli.realtime_voice_deepgram_bridge --check --strict --production-en-ja", file=sys.stderr)
    print(
        "  python -m hermes_cli.realtime_voice_deepgram_bridge --host 127.0.0.1 --port 8766 --production-en-ja",
        file=sys.stderr,
    )
    print("  python -m hermes_cli.realtime_voice_alpha_evidence --runs 3", file=sys.stderr)


def _print_summary(runs: list[tuple[str, list[dict[str, Any]]]]) -> None:
    summary = summarize_realtime_voice_smoke_report_runs(runs)
    print(
        "Realtime voice alpha evidence OK: "
        f"{summary.get('entries')} smoke result(s) across {summary.get('runs')} run(s)"
    )
    latency = summary.get("latency_ms")
    if not isinstance(latency, dict):
        return
    for label in ("audio_to_partial_transcript", "final_transcript_to_first_audio", "barge_in_ack"):
        metric = latency.get(label)
        if not isinstance(metric, dict) or not metric.get("count"):
            continue
        print(
            f"  {label}: p50={metric.get('p50')}ms "
            f"p95={metric.get('p95')}ms max={metric.get('max')}ms n={metric.get('count')}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
