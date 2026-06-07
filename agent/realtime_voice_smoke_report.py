"""Realtime voice smoke report validation.

The doctor command can produce a JSON report for realtime voice sidecar smoke
checks. This module validates that report as an alpha/release gate without
depending on any particular machine, sidecar implementation, or accelerator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ALPHA_REQUIRED_AUDIO_FIXTURES = (
    "./fixtures/realtime-voice/en/hello.webm",
    "./fixtures/realtime-voice/en/tool-question.webm",
    "./fixtures/realtime-voice/ja/hello.webm",
    "./fixtures/realtime-voice/ja/tool-question.webm",
)

ALPHA_REQUIRED_TTS_TEXTS = (
    "Hello from Hermes.",
    "Can you hear me clearly?",
    "こんにちは、Hermesです。",
    "音声で会話できますか？",
)

ALPHA_REQUIRED_BARGE_IN_TEXTS = (
    "Hello from Hermes.",
)


@dataclass(frozen=True)
class RealtimeVoiceSmokeReportIssue:
    kind: str
    message: str
    identifier: str = ""

    def format(self) -> str:
        suffix = f" ({self.identifier})" if self.identifier else ""
        return f"{self.kind}: {self.message}{suffix}"


def load_realtime_voice_smoke_report(path: str | Path) -> list[dict[str, Any]]:
    report_path = Path(path)
    data = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("realtime voice smoke report must be a JSON array")
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"realtime voice smoke report entry {index} must be an object")
        entries.append(dict(entry))
    return entries


def validate_realtime_voice_smoke_report(
    entries: Sequence[Mapping[str, Any]],
    *,
    required_audio_fixtures: Iterable[str] = (),
    required_tts_texts: Iterable[str] = (),
    required_barge_in_texts: Iterable[str] = (),
    require_protocol: bool = True,
) -> list[RealtimeVoiceSmokeReportIssue]:
    issues: list[RealtimeVoiceSmokeReportIssue] = []
    by_kind = _entries_by_kind(entries)

    if require_protocol:
        protocol_entries = by_kind.get("protocol", [])
        if not protocol_entries:
            issues.append(RealtimeVoiceSmokeReportIssue("protocol", "missing protocol smoke result"))
        else:
            for entry in protocol_entries:
                issues.extend(_validate_protocol_entry(entry))

    audio_entries = by_kind.get("audio_fixture", [])
    tts_entries = by_kind.get("tts", [])
    barge_in_entries = by_kind.get("barge_in", [])

    issues.extend(_validate_required_entries(
        entries=audio_entries,
        required=required_audio_fixtures,
        field="fixture",
        kind="audio_fixture",
    ))
    issues.extend(_validate_required_entries(
        entries=tts_entries,
        required=required_tts_texts,
        field="text",
        kind="tts",
    ))
    issues.extend(_validate_required_entries(
        entries=barge_in_entries,
        required=required_barge_in_texts,
        field="text",
        kind="barge_in",
    ))

    for entry in audio_entries:
        issues.extend(_validate_audio_fixture_entry(entry))
    for entry in tts_entries:
        issues.extend(_validate_tts_entry(entry))
    for entry in barge_in_entries:
        issues.extend(_validate_barge_in_entry(entry))

    return issues


def validate_realtime_voice_alpha_report(entries: Sequence[Mapping[str, Any]]) -> list[RealtimeVoiceSmokeReportIssue]:
    return validate_realtime_voice_smoke_report(
        entries,
        required_audio_fixtures=ALPHA_REQUIRED_AUDIO_FIXTURES,
        required_tts_texts=ALPHA_REQUIRED_TTS_TEXTS,
        required_barge_in_texts=ALPHA_REQUIRED_BARGE_IN_TEXTS,
        require_protocol=True,
    )


def _entries_by_kind(entries: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    by_kind: dict[str, list[Mapping[str, Any]]] = {}
    for entry in entries:
        kind = str(entry.get("kind") or "").strip()
        if not kind:
            continue
        by_kind.setdefault(kind, []).append(entry)
    return by_kind


def _validate_required_entries(
    *,
    entries: Sequence[Mapping[str, Any]],
    required: Iterable[str],
    field: str,
    kind: str,
) -> list[RealtimeVoiceSmokeReportIssue]:
    available = {str(entry.get(field) or "") for entry in entries}
    issues: list[RealtimeVoiceSmokeReportIssue] = []
    for item in required:
        if item not in available:
            issues.append(RealtimeVoiceSmokeReportIssue(kind, f"missing required {field}", item))
    return issues


def _validate_protocol_entry(entry: Mapping[str, Any]) -> list[RealtimeVoiceSmokeReportIssue]:
    identifier = str(entry.get("kind") or "protocol")
    issues = _validate_common_ok(entry, kind="protocol", identifier=identifier)
    events = _events(entry)
    if "frontend.state" not in events:
        issues.append(RealtimeVoiceSmokeReportIssue("protocol", "missing frontend.state event", identifier))
    if "transcript.final" not in events:
        issues.append(RealtimeVoiceSmokeReportIssue("protocol", "missing transcript.final event", identifier))
    if _positive_int(entry.get("transcript_final_ms")) is None:
        issues.append(RealtimeVoiceSmokeReportIssue("protocol", "missing transcript_final_ms", identifier))
    return issues


def _validate_audio_fixture_entry(entry: Mapping[str, Any]) -> list[RealtimeVoiceSmokeReportIssue]:
    identifier = str(entry.get("fixture") or "audio_fixture")
    issues = _validate_common_ok(entry, kind="audio_fixture", identifier=identifier)
    events = _events(entry)
    if "transcript.partial" not in events:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_fixture", "missing transcript.partial event", identifier))
    if "transcript.final" not in events:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_fixture", "missing transcript.final event", identifier))
    if _positive_int(entry.get("audio_bytes")) is None:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_fixture", "missing audio bytes", identifier))
    partial_ms = _positive_int(entry.get("transcript_partial_ms"))
    target_ms = _positive_int(entry.get("target_ms"))
    if partial_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_fixture", "missing transcript_partial_ms", identifier))
    if target_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_fixture", "missing target_ms", identifier))
    elif partial_ms is not None and partial_ms > target_ms:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "audio_fixture",
                f"transcript_partial_ms {partial_ms} exceeds target {target_ms}",
                identifier,
            )
        )
    if _positive_int(entry.get("transcript_final_ms")) is None:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_fixture", "missing transcript_final_ms", identifier))
    return issues


def _validate_tts_entry(entry: Mapping[str, Any]) -> list[RealtimeVoiceSmokeReportIssue]:
    identifier = str(entry.get("text") or "tts")
    issues = _validate_common_ok(entry, kind="tts", identifier=identifier)
    events = _events(entry)
    if "audio.output.chunk" not in events:
        issues.append(RealtimeVoiceSmokeReportIssue("tts", "missing audio.output.chunk event", identifier))
    if _positive_int(entry.get("output_audio_bytes")) is None:
        issues.append(RealtimeVoiceSmokeReportIssue("tts", "missing output audio bytes", identifier))
    first_audio_ms = _positive_int(entry.get("first_audio_ms"))
    target_ms = _positive_int(entry.get("target_ms"))
    if first_audio_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("tts", "missing first_audio_ms", identifier))
    if target_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("tts", "missing target_ms", identifier))
    elif first_audio_ms is not None and first_audio_ms > target_ms:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "tts",
                f"first_audio_ms {first_audio_ms} exceeds target {target_ms}",
                identifier,
            )
        )
    return issues


def _validate_barge_in_entry(entry: Mapping[str, Any]) -> list[RealtimeVoiceSmokeReportIssue]:
    identifier = str(entry.get("text") or "barge_in")
    issues = _validate_common_ok(entry, kind="barge_in", identifier=identifier)
    events = _events(entry)
    if "barge_in" not in events:
        issues.append(RealtimeVoiceSmokeReportIssue("barge_in", "missing barge_in event", identifier))
    ack_ms = _nonnegative_int(entry.get("barge_in_ack_ms"))
    target_ms = _positive_int(entry.get("target_ms"))
    if ack_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("barge_in", "missing barge_in_ack_ms", identifier))
    if target_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("barge_in", "missing target_ms", identifier))
    elif ack_ms is not None and ack_ms > target_ms:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "barge_in",
                f"barge_in_ack_ms {ack_ms} exceeds target {target_ms}",
                identifier,
            )
        )
    return issues


def _validate_common_ok(
    entry: Mapping[str, Any],
    *,
    kind: str,
    identifier: str,
) -> list[RealtimeVoiceSmokeReportIssue]:
    if entry.get("ok") is True:
        return []
    error = str(entry.get("error") or "smoke result was not ok")
    return [RealtimeVoiceSmokeReportIssue(kind, error, identifier)]


def _events(entry: Mapping[str, Any]) -> set[str]:
    raw = entry.get("events")
    if not isinstance(raw, list):
        return set()
    return {str(item) for item in raw if isinstance(item, str)}


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
