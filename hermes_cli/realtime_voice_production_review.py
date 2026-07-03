"""Create and validate Hermes realtime voice production launch reviews."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from agent.realtime_voice_errors import sanitize_realtime_voice_error


REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS = {
    "human_en_ja_conversations": "Human English and Japanese conversation sessions passed",
    "noisy_room_and_headset_coverage": "Noisy-room, laptop microphone, and headset coverage passed",
    "remote_sidecar_latency_drill": "Remote sidecar latency and reconnect drill passed",
    "desktop_reconnect_recovery": (
        "Desktop websocket reconnect recovery releases microphone capture, stops playback, "
        "drops stale queued audio, and returns to realtime or fallback voice cleanly"
    ),
    "provider_failure_drill": "Streaming STT/TTS or native S2S provider failure drill passed",
    "barge_in_reliability": "Barge-in reliability passed under real playback",
    "tool_call_policy_review": "Tool-call and data-access behavior reviewed for live voice",
    "accessibility_review": "Accessibility review for captions, mute, fallback, and interruption passed",
    "security_review": "Credential, transport, and remote-sidecar security review passed",
    "operator_docs_review": "Operator setup, fallback, and incident docs reviewed",
    "kame_dgx_benchmark_evidence": (
        "DGX Spark KAME benchmark evidence passed the generated benchmark matrix validator"
    ),
    "kame_e2b_direct_audio_launch": (
        "Gemma 4 E2B direct-audio reflex launch was validated on the target DGX Spark runtime"
    ),
    "kame_oracle_asr_hypothesis_comparison": (
        "Oracle outcomes were compared with and without oracle-verbatim ASR transcript hypotheses"
    ),
    "kame_all_local_dgx_spark_smoke": (
        "All-local DGX Spark smoke passed with interface, oracle, ASR, TTS, and sidecar running together"
    ),
    "kame_live_discord_full_path_smoke": (
        "Live Discord smoke passed the full KAME path with production credentials"
    ),
}
KAME_DGX_BENCHMARK_EVIDENCE_CHECK = "kame_dgx_benchmark_evidence"
KAME_DGX_REQUIRED_BENCHMARK_COVERAGE = frozenset(
    {
        "interface_candidate_model_matrix",
        "interface_direct_audio_vs_stt_fallback",
        "interface_direct_audio_latency",
        "oracle_simple_first_audio_latency",
        "oracle_outcomes_with_and_without_asr_hypotheses",
        "oracle_verbatim_asr_latency_and_literal_accuracy",
        "local_asr_tts_benchmark_matrix",
        "model_assumptions_validated",
        "all_local_smoke",
        "cloud_fallback_smoke",
        "capability_honesty_smoke",
        "barge_in_interruption_smoke",
        "async_oracle_witness_fusion_single_bundle",
        "async_oracle_interpreter_prompt_input_order_visible",
        "async_oracle_interpreter_prompt_policy_visible",
        "async_oracle_transcript_hypotheses_unpromoted",
        "async_oracle_unpromoted_hypothesis_action_sinks_clean",
        "async_oracle_runtime_kame_action_gate_enforced",
        "async_oracle_unflagged_high_risk_tool_event_fails_closed",
        "async_oracle_kame_ack_latency_metrics_visible",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or validate a Hermes realtime voice production launch-review report"
    )
    parser.add_argument(
        "report",
        help="Path to the realtime voice production review JSON report",
    )
    parser.add_argument(
        "--write-template",
        action="store_true",
        help="Write a review template instead of validating an existing report",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing report when writing a template",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="After validation succeeds, set voice.realtime.production_review_report in config.yaml",
    )
    parser.add_argument(
        "--reviewer",
        default="",
        help="Reviewer name or team to include in a written template",
    )
    parser.add_argument(
        "--pass-check",
        action="append",
        default=[],
        choices=tuple(REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS),
        metavar="CHECK",
        help="Mark CHECK true in a written template; repeat for multiple checks",
    )
    parser.add_argument(
        "--all-passed",
        action="store_true",
        help="Mark every required check true in a written template",
    )
    parser.add_argument(
        "--evidence-note",
        action="append",
        default=[],
        metavar="CHECK=TEXT",
        help="Attach evidence notes for CHECK in a written template; repeat for multiple checks",
    )
    parser.add_argument(
        "--evidence-artifact",
        action="append",
        default=[],
        metavar="CHECK=PATH_OR_URL",
        help="Attach an evidence artifact reference for CHECK in a written template; repeat for multiple artifacts",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = Path(args.report).expanduser()

    if args.write_template:
        try:
            evidence = parse_production_review_evidence_args(
                artifact_args=args.evidence_artifact,
                note_args=args.evidence_note,
            )
            report = build_production_review_report(
                evidence=evidence,
                reviewer=args.reviewer,
                passed_checks=REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS
                if args.all_passed
                else args.pass_check,
            )
            write_production_review_report(report_path, report, overwrite=bool(args.overwrite))
        except Exception as exc:
            print(
                f"Realtime voice production review failed: {sanitize_realtime_voice_error(exc)}",
                file=sys.stderr,
            )
            return 1
        print(f"Realtime voice production review template written: {report_path}")
        issues = validate_production_review_report(report, report_path=report_path)
        if issues:
            print(f"Pending check(s): {len(issues)}")
            for issue in issues:
                print(f"  - {issue}")
            if args.apply:
                print("Config not updated until every production review check passes")
        else:
            print("Realtime voice production review OK")
            if args.apply:
                config_path = apply_production_review_report(report_path)
                print(f"Updated realtime voice production_review_report in {config_path}")
        return 0

    try:
        report = load_production_review_report(report_path)
    except Exception as exc:
        print(
            f"Realtime voice production review failed: {sanitize_realtime_voice_error(exc)}",
            file=sys.stderr,
        )
        return 1

    issues = validate_production_review_report(report, report_path=report_path)
    if not issues:
        print("Realtime voice production review OK")
        if args.apply:
            config_path = apply_production_review_report(report_path)
            print(f"Updated realtime voice production_review_report in {config_path}")
        return 0

    print(f"Realtime voice production review failed: {len(issues)} issue(s)", file=sys.stderr)
    for issue in issues:
        print(f"  - {issue}", file=sys.stderr)
    return 1


def apply_production_review_report(report_path: str | Path) -> Path:
    from hermes_cli.config import get_config_path, read_raw_config, save_config

    path = Path(report_path).expanduser()
    config = read_raw_config()
    if not isinstance(config, dict):
        config = {}
    voice = config.get("voice")
    if not isinstance(voice, dict):
        voice = {}
    realtime = voice.get("realtime")
    if not isinstance(realtime, dict):
        realtime = {}
    realtime["production_review_report"] = str(path)
    voice["realtime"] = realtime
    config["voice"] = voice
    save_config(config)
    return get_config_path()


def parse_production_review_evidence_args(
    *,
    artifact_args: list[str] | tuple[str, ...] = (),
    note_args: list[str] | tuple[str, ...] = (),
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for raw in note_args:
        key, value = _parse_check_value_arg(raw, "--evidence-note")
        entry = evidence.setdefault(key, {"notes": "", "artifacts": []})
        existing = str(entry.get("notes") or "").strip()
        entry["notes"] = f"{existing}\n{value}".strip() if existing else value
    for raw in artifact_args:
        key, value = _parse_check_value_arg(raw, "--evidence-artifact")
        entry = evidence.setdefault(key, {"notes": "", "artifacts": []})
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, list):
            artifacts = []
        artifacts.append(value)
        entry["artifacts"] = artifacts
    return evidence


def _parse_check_value_arg(raw: str, flag: str) -> tuple[str, str]:
    text = str(raw or "")
    if "=" not in text:
        raise ValueError(f"{flag} must use CHECK=VALUE")
    key, value = text.split("=", 1)
    key = key.strip()
    value = value.strip()
    if key not in REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS:
        raise ValueError(f"{flag} uses unknown check: {key}")
    if not value:
        raise ValueError(f"{flag} for {key} requires a non-empty value")
    return key, value


def build_production_review_report(
    *,
    evidence: Mapping[str, Any] | None = None,
    reviewer: str = "",
    passed_checks: Mapping[str, Any] | list[str] | tuple[str, ...] | set[str] = (),
) -> dict[str, Any]:
    if isinstance(passed_checks, Mapping):
        passed = set(passed_checks.keys())
    else:
        passed = {str(key) for key in passed_checks}
    evidence = evidence if isinstance(evidence, Mapping) else {}
    return {
        "kind": "realtime_voice_production_review",
        "reviewer": str(reviewer or ""),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "checks": {
            key: key in passed
            for key in REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS
        },
        "evidence": {
            key: _production_review_evidence_template(evidence.get(key))
            for key in REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS
        },
    }


def write_production_review_report(
    report_path: str | Path,
    report: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    path = Path(report_path).expanduser()
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite or choose another path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_production_review_report(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path).expanduser()
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError("production review report must be a JSON object")
    return report


def validate_production_review_report(
    report: Mapping[str, Any],
    *,
    report_path: str | Path | None = None,
) -> list[str]:
    issues: list[str] = []
    if str(report.get("kind") or "") != "realtime_voice_production_review":
        issues.append("invalid_kind")
    reviewed_at = str(report.get("reviewed_at") or "").strip()
    if not reviewed_at:
        issues.append("missing_reviewed_at")
    reviewer = str(report.get("reviewer") or "").strip()
    if not reviewer:
        issues.append("missing_reviewer")
    checks = report.get("checks")
    checks = checks if isinstance(checks, Mapping) else {}
    if not checks:
        issues.append("missing_checks")
    evidence = report.get("evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    if not evidence:
        issues.append("missing_evidence")
    for key in REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS:
        if checks.get(key) is not True:
            issues.append(f"review_check_missing:{key}")
        elif not _production_review_evidence_is_supported(evidence.get(key)):
            issues.append(f"review_evidence_missing:{key}")
        elif key == KAME_DGX_BENCHMARK_EVIDENCE_CHECK:
            issue = _kame_dgx_benchmark_evidence_issue(
                evidence.get(key),
                report_path=report_path,
            )
            if issue:
                issues.append(issue)
    return list(dict.fromkeys(issues))


def _production_review_evidence_template(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        artifacts = value.get("artifacts")
        artifacts = artifacts if isinstance(artifacts, list) else []
        return {
            "notes": str(value.get("notes") or ""),
            "artifacts": [str(item) for item in artifacts if str(item or "").strip()],
        }
    if isinstance(value, str) and value.strip():
        return {"notes": value.strip(), "artifacts": []}
    return {"notes": "", "artifacts": []}


def _production_review_evidence_is_supported(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if not isinstance(value, Mapping):
        return False
    notes = str(value.get("notes") or "").strip()
    if notes:
        return True
    artifacts = value.get("artifacts")
    if isinstance(artifacts, list) and any(str(item or "").strip() for item in artifacts):
        return True
    return False


def _kame_dgx_benchmark_evidence_issue(
    value: Any,
    *,
    report_path: str | Path | None = None,
) -> str:
    artifacts = _production_review_evidence_artifacts(value)
    if not artifacts:
        return "review_evidence_invalid:kame_dgx_benchmark_evidence:requires_local_validator_json"
    base_dir = _production_review_report_base_dir(report_path)
    attempted: list[str] = []
    for artifact in artifacts:
        path = _local_artifact_path(artifact, base_dir=base_dir)
        if path is None:
            attempted.append(f"{artifact}:not_local_file")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            attempted.append(f"{artifact}:{sanitize_realtime_voice_error(exc)}")
            continue
        if _kame_dgx_benchmark_validation_payload_ok(payload):
            return ""
        attempted.append(f"{artifact}:validator_not_ok")
    suffix = ",".join(attempted[:3])
    return (
        "review_evidence_invalid:kame_dgx_benchmark_evidence:"
        "requires_local_validator_json"
        f"[{suffix}]" if suffix else
        "review_evidence_invalid:kame_dgx_benchmark_evidence:requires_local_validator_json"
    )


def _production_review_evidence_artifacts(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list):
        return []
    return [str(item).strip() for item in artifacts if str(item or "").strip()]


def _production_review_report_base_dir(report_path: str | Path | None) -> Path:
    if report_path is None:
        return Path.cwd()
    path = Path(report_path).expanduser()
    if path.suffix:
        return path.parent
    return path


def _local_artifact_path(value: str, *, base_dir: Path) -> Path | None:
    if value.startswith(("http://", "https://")):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def _kame_dgx_benchmark_validation_payload_ok(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if isinstance(payload.get("benchmark_evidence"), Mapping):
        return bool(payload.get("ok") is True) and _kame_dgx_benchmark_validation_payload_ok(
            payload["benchmark_evidence"]
        )
    if payload.get("ok") is not True:
        return False
    issues = payload.get("issues")
    if isinstance(issues, list) and any(str(item or "").strip() for item in issues):
        return False
    coverage = payload.get("coverage")
    if not isinstance(coverage, Mapping):
        return False
    return all(coverage.get(key) is True for key in KAME_DGX_REQUIRED_BENCHMARK_COVERAGE)


if __name__ == "__main__":
    raise SystemExit(main())
