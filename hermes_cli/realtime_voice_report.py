"""CLI helpers for realtime voice smoke reports."""

from __future__ import annotations

import argparse
import sys

from agent.realtime_voice_smoke_report import (
    load_realtime_voice_smoke_report,
    validate_realtime_voice_alpha_report,
    validate_realtime_voice_smoke_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a Hermes realtime voice smoke report")
    parser.add_argument("report", help="Path to the JSON report written by hermes doctor --realtime-voice-report")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    entries = load_realtime_voice_smoke_report(args.report)
    if args.alpha:
        issues = validate_realtime_voice_alpha_report(entries)
    else:
        issues = validate_realtime_voice_smoke_report(
            entries,
            required_audio_fixtures=args.required_audio_fixture,
            required_tts_texts=args.required_tts_text,
            require_protocol=not args.no_protocol,
        )
    if not issues:
        print(f"Realtime voice smoke report OK: {len(entries)} result(s)")
        return 0
    print(f"Realtime voice smoke report failed: {len(issues)} issue(s)", file=sys.stderr)
    for issue in issues:
        print(f"  - {issue.format()}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
