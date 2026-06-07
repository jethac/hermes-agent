"""Run repeated realtime voice alpha evidence checks."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

from agent.realtime_voice_smoke_report import (
    load_realtime_voice_smoke_report,
    summarize_realtime_voice_smoke_report_runs,
    validate_realtime_voice_alpha_report_runs,
)


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
    output_dir.mkdir(parents=True, exist_ok=True)

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

    from hermes_cli.doctor import run_doctor

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

    runs = [(str(path), load_realtime_voice_smoke_report(path)) for path in report_paths]
    issues = validate_realtime_voice_alpha_report_runs(runs, min_runs=run_count)
    if issues:
        print(f"Realtime voice alpha evidence failed: {len(issues)} issue(s)", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue.format()}", file=sys.stderr)
        return 1

    _print_summary(runs)
    return 0


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
