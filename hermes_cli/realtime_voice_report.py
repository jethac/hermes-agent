"""CLI helpers for realtime voice smoke reports."""

from __future__ import annotations

import argparse
import sys

from agent.realtime_voice_smoke_report import (
    load_realtime_voice_smoke_report,
    summarize_realtime_voice_smoke_report_runs,
    validate_realtime_voice_alpha_report_runs,
    validate_realtime_voice_alpha_report,
    validate_realtime_voice_smoke_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a Hermes realtime voice smoke report")
    parser.add_argument(
        "report",
        nargs="+",
        help="Path to one or more JSON reports written by hermes doctor --realtime-voice-report",
    )
    parser.add_argument(
        "--alpha",
        action="store_true",
        help="Require the documented English/Japanese private-alpha fixture and TTS set",
    )
    parser.add_argument(
        "--required-audio-fixture",
        action="append",
        default=[],
        metavar="PATH",
        help="Require a passing audio fixture result for PATH; repeat for multiple fixtures",
    )
    parser.add_argument(
        "--required-tts-text",
        action="append",
        default=[],
        metavar="TEXT",
        help="Require a passing TTS result for TEXT; repeat for multiple phrases",
    )
    parser.add_argument(
        "--no-protocol",
        action="store_true",
        help="Do not require a protocol smoke result",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=1,
        help="Require at least this many report files/runs when validating --alpha evidence",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = [(report, load_realtime_voice_smoke_report(report)) for report in args.report]
    if args.alpha:
        if len(runs) == 1 and args.min_runs <= 1:
            issues = validate_realtime_voice_alpha_report(runs[0][1])
        else:
            issues = validate_realtime_voice_alpha_report_runs(runs, min_runs=args.min_runs)
    else:
        entries = [entry for _report, report_entries in runs for entry in report_entries]
        issues = validate_realtime_voice_smoke_report(
            entries,
            required_audio_fixtures=args.required_audio_fixture,
            required_tts_texts=args.required_tts_text,
            require_protocol=not args.no_protocol,
        )
    if not issues:
        result_count = sum(len(entries) for _report, entries in runs)
        print(f"Realtime voice smoke report OK: {result_count} result(s) across {len(runs)} run(s)")
        summary = summarize_realtime_voice_smoke_report_runs(runs)
        latency = summary.get("latency_ms", {})
        for label in ("audio_to_partial_transcript", "final_transcript_to_first_audio", "barge_in_ack"):
            metric = latency.get(label) if isinstance(latency, dict) else None
            if not isinstance(metric, dict) or not metric.get("count"):
                continue
            print(
                f"  {label}: p50={metric.get('p50')}ms "
                f"p95={metric.get('p95')}ms max={metric.get('max')}ms n={metric.get('count')}"
            )
        return 0
    print(f"Realtime voice smoke report failed: {len(issues)} issue(s)", file=sys.stderr)
    for issue in issues:
        print(f"  - {issue.format()}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
