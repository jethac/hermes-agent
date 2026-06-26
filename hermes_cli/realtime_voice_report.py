"""CLI helpers for realtime voice smoke reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
        "--discord-live-probe",
        action="store_true",
        help="Validate Discord live voice probe entries written by hermes doctor",
    )
    parser.add_argument(
        "--require-inbound",
        action="store_true",
        help="With --discord-live-probe, require observed inbound speech frames or speech-start callbacks",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=1,
        help="Require at least this many report files/runs when validating --alpha evidence",
    )
    parser.add_argument(
        "--apply-production-evidence",
        action="store_true",
        help=(
            "After --alpha validation succeeds, set "
            "voice.realtime.production_evidence_report in config.yaml"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.apply_production_evidence and not args.alpha:
        print(
            "Realtime voice smoke report failed: --apply-production-evidence requires --alpha",
            file=sys.stderr,
        )
        return 1
    runs = [(report, load_realtime_voice_smoke_report(report)) for report in args.report]
    if args.alpha:
        if len(runs) == 1 and args.min_runs <= 1 and not args.apply_production_evidence:
            issues = validate_realtime_voice_alpha_report(runs[0][1])
        else:
            issues = validate_realtime_voice_alpha_report_runs(
                runs,
                min_runs=args.min_runs,
                allow_loopback_validation=not args.apply_production_evidence,
            )
    elif args.discord_live_probe:
        entries = [entry for _report, report_entries in runs for entry in report_entries]
        issues = validate_discord_live_probe_report(
            entries,
            require_inbound=bool(args.require_inbound),
        )
    else:
        entries = [entry for _report, report_entries in runs for entry in report_entries]
        issues = validate_realtime_voice_smoke_report(
            entries,
            required_audio_fixtures=args.required_audio_fixture,
            required_tts_texts=args.required_tts_text,
            require_protocol=not args.no_protocol,
        )
    if not issues:
        result_count = sum(
            1
            for _report, entries in runs
            for entry in entries
            if str(entry.get("kind") or "") != "manifest"
        )
        print(f"Realtime voice smoke report OK: {result_count} result(s) across {len(runs)} run(s)")
        summary = summarize_realtime_voice_smoke_report_runs(runs)
        latency = summary.get("latency_ms", {})
        for label in (
            "audio_to_partial_transcript",
            "final_transcript_to_first_text",
            "final_transcript_to_first_audio",
            "barge_in_ack",
        ):
            metric = latency.get(label) if isinstance(latency, dict) else None
            if not isinstance(metric, dict) or not metric.get("count"):
                continue
            print(
                f"  {label}: p50={metric.get('p50')}ms "
                f"p95={metric.get('p95')}ms max={metric.get('max')}ms n={metric.get('count')}"
            )
        if args.apply_production_evidence:
            from hermes_cli.realtime_voice_alpha_evidence import (
                apply_realtime_voice_production_evidence_report,
            )

            try:
                evidence_path = _production_evidence_reference_path(args.report)
            except ValueError as exc:
                print(f"Realtime voice smoke report failed: {exc}", file=sys.stderr)
                return 1
            config_path = apply_realtime_voice_production_evidence_report(evidence_path)
            print(f"Updated realtime voice production_evidence_report in {config_path}")
        return 0
    print(f"Realtime voice smoke report failed: {len(issues)} issue(s)", file=sys.stderr)
    for issue in issues:
        print(f"  - {_format_report_issue(issue)}", file=sys.stderr)
    return 1


def _format_report_issue(issue: object) -> str:
    formatter = getattr(issue, "format", None)
    if callable(formatter):
        return str(formatter())
    return str(issue)


def validate_discord_live_probe_report(
    entries: list[dict],
    *,
    require_inbound: bool = False,
) -> list[str]:
    probes = [entry for entry in entries if str(entry.get("kind") or "") == "discord_live_probe"]
    issues: list[str] = []
    if not probes:
        return ["discord_live_probe: missing Discord live probe result"]
    if not any(bool(probe.get("ok")) for probe in probes):
        details = [
            str(probe.get("failure_reason") or probe.get("error") or "probe failed")
            for probe in probes
            if not bool(probe.get("ok"))
        ]
        issues.append(f"discord_live_probe: no passing probe ({'; '.join(details[:3])})")
    latest = probes[-1]
    required_bools = (
        "connect_perm",
        "speak_perm",
        "connected",
        "opus_loaded",
        "accepted_audio_source",
        "played",
        "playing_during_probe",
        "receiver_started",
        "disconnected",
    )
    for field in required_bools:
        if latest.get(field) is not True:
            issues.append(f"discord_live_probe: {field} is not true")
    if require_inbound:
        inbound = (
            latest.get("inbound_observed") is True
            or _positive_int(latest.get("receiver_frames")) > 0
            or _positive_int(latest.get("receiver_speech_start")) > 0
        )
        if not inbound:
            reason = str(latest.get("failure_reason") or latest.get("error") or "no inbound speech observed")
            issues.append(f"discord_live_probe: inbound speech not observed ({reason})")
    return issues


def _positive_int(value: object) -> int:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def _production_evidence_reference_path(reports: list[str]) -> Path:
    paths = [Path(report).expanduser() for report in reports]
    if not paths:
        raise ValueError("at least one report path is required")
    if len(paths) == 1:
        return paths[0]
    parents = {path.parent for path in paths}
    if len(parents) != 1:
        raise ValueError(
            "--apply-production-evidence with multiple reports requires all reports to share one directory"
        )
    return next(iter(parents))


if __name__ == "__main__":
    raise SystemExit(main())
