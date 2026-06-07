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
    "provider_failure_drill": "Streaming STT/TTS or native S2S provider failure drill passed",
    "barge_in_reliability": "Barge-in reliability passed under real playback",
    "tool_call_policy_review": "Tool-call and data-access behavior reviewed for live voice",
    "accessibility_review": "Accessibility review for captions, mute, fallback, and interruption passed",
    "security_review": "Credential, transport, and remote-sidecar security review passed",
    "operator_docs_review": "Operator setup, fallback, and incident docs reviewed",
}


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_path = Path(args.report).expanduser()

    if args.write_template:
        try:
            report = build_production_review_report(
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
        issues = validate_production_review_report(report)
        if issues:
            print(f"Pending check(s): {len(issues)}")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Realtime voice production review OK")
        return 0

    try:
        report = load_production_review_report(report_path)
    except Exception as exc:
        print(
            f"Realtime voice production review failed: {sanitize_realtime_voice_error(exc)}",
            file=sys.stderr,
        )
        return 1

    issues = validate_production_review_report(report)
    if not issues:
        print("Realtime voice production review OK")
        return 0

    print(f"Realtime voice production review failed: {len(issues)} issue(s)", file=sys.stderr)
    for issue in issues:
        print(f"  - {issue}", file=sys.stderr)
    return 1


def build_production_review_report(
    *,
    reviewer: str = "",
    passed_checks: Mapping[str, Any] | list[str] | tuple[str, ...] | set[str] = (),
) -> dict[str, Any]:
    if isinstance(passed_checks, Mapping):
        passed = set(passed_checks.keys())
    else:
        passed = {str(key) for key in passed_checks}
    return {
        "kind": "realtime_voice_production_review",
        "reviewer": str(reviewer or ""),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "checks": {
            key: key in passed
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


def validate_production_review_report(report: Mapping[str, Any]) -> list[str]:
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
    for key in REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS:
        if checks.get(key) is not True:
            issues.append(f"review_check_missing:{key}")
    return list(dict.fromkeys(issues))


if __name__ == "__main__":
    raise SystemExit(main())
