"""Realtime voice smoke report validation.

The doctor command can produce a JSON report for realtime voice sidecar smoke
checks. This module validates that report as an alpha/release gate without
depending on any particular machine, sidecar implementation, or accelerator.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ALPHA_REQUIRED_AUDIO_FIXTURES = (
    "./fixtures/realtime-voice/en/hello.webm",
    "./fixtures/realtime-voice/en/tool-question.webm",
    "./fixtures/realtime-voice/ja/hello.webm",
    "./fixtures/realtime-voice/ja/tool-question.webm",
)

ALPHA_REQUIRED_AUDIO_FIXTURE_TEXTS = {
    "./fixtures/realtime-voice/en/hello.webm": "Hello from Hermes.",
    "./fixtures/realtime-voice/en/tool-question.webm": (
        "What files can Hermes see in this workspace?"
    ),
    "./fixtures/realtime-voice/ja/hello.webm": "こんにちは、Hermesです。",
    "./fixtures/realtime-voice/ja/tool-question.webm": (
        "Hermesはこのワークスペースで何を確認できますか？"
    ),
}

ALPHA_REQUIRED_TTS_TEXTS = (
    "Hello from Hermes.",
    "Can you hear me clearly?",
    "こんにちは、Hermesです。",
    "音声で会話できますか？",
)

ALPHA_REQUIRED_TTS_METADATA = {
    "Hello from Hermes.": {"language": "en", "script": "Latn"},
    "Can you hear me clearly?": {"language": "en", "script": "Latn"},
    "こんにちは、Hermesです。": {"language": "ja", "script": "Jpan"},
    "音声で会話できますか？": {"language": "ja", "script": "Jpan"},
}

ALPHA_REQUIRED_BARGE_IN_TEXTS = (
    "Hello from Hermes.",
)

ALPHA_REQUIRED_SESSION_TURN_TEXTS = (
    "Hello from Hermes.",
    "こんにちは、Hermesです。",
)

ALPHA_REQUIRED_SESSION_TURN_METADATA = {
    "Hello from Hermes.": {"language": "en", "script": "Latn"},
    "こんにちは、Hermesです。": {"language": "ja", "script": "Jpan"},
}

ALPHA_REQUIRED_QUALITY_TARGETS_MS = {
    "audio_to_partial_transcript_ms": 300,
    "final_transcript_to_first_text_ms": 500,
    "final_transcript_to_first_audio_ms": 900,
    "barge_in_ack_ms": 150,
}


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


def load_realtime_voice_smoke_report_runs(path: str | Path) -> list[tuple[str, list[dict[str, Any]]]]:
    report_path = Path(path).expanduser()
    if report_path.is_dir():
        files = sorted(item for item in report_path.glob("*.json") if item.is_file())
        return [(str(item), load_realtime_voice_smoke_report(item)) for item in files]
    return [(str(report_path), load_realtime_voice_smoke_report(report_path))]


def validate_realtime_voice_alpha_report_runs(
    runs: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    *,
    min_runs: int = 1,
) -> list[RealtimeVoiceSmokeReportIssue]:
    issues: list[RealtimeVoiceSmokeReportIssue] = []
    required_runs = max(1, int(min_runs or 1))
    if len(runs) < required_runs:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "evidence",
                f"requires at least {required_runs} run(s), found {len(runs)}",
            )
        )
    for label, entries in runs:
        for issue in validate_realtime_voice_alpha_report(entries):
            issues.append(
                RealtimeVoiceSmokeReportIssue(
                    issue.kind,
                    issue.message,
                    f"{label}: {issue.identifier}" if issue.identifier else label,
                )
            )
    fingerprints: dict[tuple[Any, ...], str] = {}
    for label, entries in runs:
        manifest = _first_entry_by_kind(entries, "manifest")
        if manifest is None:
            continue
        fingerprint = realtime_voice_alpha_manifest_fingerprint(manifest)
        if fingerprint not in fingerprints:
            fingerprints[fingerprint] = label
    if len(fingerprints) > 1:
        labels = ", ".join(sorted(fingerprints.values()))
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "evidence",
                "alpha runs used mixed realtime voice stack manifests",
                labels,
            )
        )
    return issues


def summarize_realtime_voice_smoke_report_runs(
    runs: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    entries = [
        entry
        for _label, report_entries in runs
        for entry in report_entries
        if str(entry.get("kind") or "") != "manifest"
    ]
    by_kind = _entries_by_kind(entries)
    return {
        "runs": len(runs),
        "entries": len(entries),
        "kinds": {
            kind: {
                "entries": len(kind_entries),
                "ok": sum(1 for entry in kind_entries if entry.get("ok") is True),
                "failed": sum(1 for entry in kind_entries if entry.get("ok") is not True),
            }
            for kind, kind_entries in sorted(by_kind.items())
        },
        "latency_ms": {
            "audio_to_partial_transcript": _latency_summary(
                entry.get("transcript_partial_ms")
                for entry in by_kind.get("audio_fixture", [])
            ),
            "final_transcript_to_first_text": _latency_summary(
                entry.get("first_text_ms")
                for entry in by_kind.get("session_turn", [])
            ),
            "final_transcript_to_first_audio": _latency_summary(
                entry.get("first_audio_ms")
                for entry in [*by_kind.get("session_turn", []), *by_kind.get("tts", [])]
            ),
            "barge_in_ack": _latency_summary(
                entry.get("barge_in_ack_ms")
                for entry in by_kind.get("barge_in", [])
            ),
        },
    }


def validate_realtime_voice_smoke_report(
    entries: Sequence[Mapping[str, Any]],
    *,
    required_audio_fixtures: Iterable[str] = (),
    required_tts_texts: Iterable[str] = (),
    required_barge_in_texts: Iterable[str] = (),
    required_session_turn_texts: Iterable[str] = (),
    require_protocol: bool = True,
    require_manifest: bool = False,
    require_alpha_targets: bool = False,
) -> list[RealtimeVoiceSmokeReportIssue]:
    issues: list[RealtimeVoiceSmokeReportIssue] = []
    by_kind = _entries_by_kind(entries)
    manifest_entries = by_kind.get("manifest", [])
    if not manifest_entries and require_manifest:
        issues.append(RealtimeVoiceSmokeReportIssue("manifest", "missing manifest entry"))
    elif manifest_entries:
        issues.extend(_validate_alpha_manifest_entry(
            manifest_entries[0],
            require_alpha_targets=require_alpha_targets,
        ))

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
    session_turn_entries = by_kind.get("session_turn", [])

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
    issues.extend(_validate_required_entries(
        entries=session_turn_entries,
        required=required_session_turn_texts,
        field="text",
        kind="session_turn",
    ))

    for entry in audio_entries:
        issues.extend(_validate_audio_fixture_entry(
            entry,
            max_target_ms=(
                ALPHA_REQUIRED_QUALITY_TARGETS_MS["audio_to_partial_transcript_ms"]
                if require_alpha_targets
                else None
            ),
        ))
    for entry in tts_entries:
        issues.extend(_validate_tts_entry(
            entry,
            max_target_ms=(
                ALPHA_REQUIRED_QUALITY_TARGETS_MS["final_transcript_to_first_audio_ms"]
                if require_alpha_targets
                else None
            ),
        ))
    for entry in session_turn_entries:
        issues.extend(_validate_session_turn_entry(
            entry,
            max_first_text_target_ms=(
                ALPHA_REQUIRED_QUALITY_TARGETS_MS["final_transcript_to_first_text_ms"]
                if require_alpha_targets
                else None
            ),
            max_first_audio_target_ms=(
                ALPHA_REQUIRED_QUALITY_TARGETS_MS["final_transcript_to_first_audio_ms"]
                if require_alpha_targets
                else None
            ),
        ))
    for entry in barge_in_entries:
        issues.extend(_validate_barge_in_entry(
            entry,
            max_target_ms=(
                ALPHA_REQUIRED_QUALITY_TARGETS_MS["barge_in_ack_ms"]
                if require_alpha_targets
                else None
            ),
        ))

    return issues


def validate_realtime_voice_alpha_report(entries: Sequence[Mapping[str, Any]]) -> list[RealtimeVoiceSmokeReportIssue]:
    return validate_realtime_voice_smoke_report(
        entries,
        required_audio_fixtures=ALPHA_REQUIRED_AUDIO_FIXTURES,
        required_tts_texts=ALPHA_REQUIRED_TTS_TEXTS,
        required_barge_in_texts=ALPHA_REQUIRED_BARGE_IN_TEXTS,
        required_session_turn_texts=ALPHA_REQUIRED_SESSION_TURN_TEXTS,
        require_protocol=True,
        require_manifest=True,
        require_alpha_targets=True,
    )


def _validate_alpha_manifest_entry(
    entry: Mapping[str, Any],
    *,
    require_alpha_targets: bool = False,
) -> list[RealtimeVoiceSmokeReportIssue]:
    issues: list[RealtimeVoiceSmokeReportIssue] = []
    if entry.get("ok") is not True:
        issues.append(RealtimeVoiceSmokeReportIssue("manifest", "manifest entry was not ok", "manifest"))
    if entry.get("available") is not True:
        issues.append(RealtimeVoiceSmokeReportIssue("manifest", "manifest was not realtime-available", "manifest"))
    conversation_quality = (
        entry.get("conversation_quality")
        if isinstance(entry.get("conversation_quality"), Mapping)
        else {}
    )
    if conversation_quality.get("live_like") is not True:
        issues.append(RealtimeVoiceSmokeReportIssue("manifest", "manifest was not live-like", "manifest"))
    if require_alpha_targets:
        quality_targets = (
            entry.get("quality_targets_ms")
            if isinstance(entry.get("quality_targets_ms"), Mapping)
            else {}
        )
        for key, ceiling in ALPHA_REQUIRED_QUALITY_TARGETS_MS.items():
            actual = _positive_int(quality_targets.get(key))
            if actual is None:
                issues.append(
                    RealtimeVoiceSmokeReportIssue(
                        "manifest",
                        f"missing quality target {key}",
                        "manifest",
                    )
                )
            elif actual > ceiling:
                issues.append(
                    RealtimeVoiceSmokeReportIssue(
                        "manifest",
                        f"quality target {key} {actual} exceeds alpha ceiling {ceiling}",
                        "manifest",
                    )
                )

    sidecar = entry.get("sidecar") if isinstance(entry.get("sidecar"), Mapping) else {}
    if sidecar.get("healthy") is not True:
        issues.append(RealtimeVoiceSmokeReportIssue("manifest", "manifest sidecar was not healthy", "manifest"))
    health = sidecar.get("health") if isinstance(sidecar.get("health"), Mapping) else {}
    capabilities = health.get("capabilities") if isinstance(health.get("capabilities"), Mapping) else {}
    if capabilities.get("native_s2s") is not True and not (
        capabilities.get("streaming_stt") is True and capabilities.get("tts") is True
    ):
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "manifest",
                "missing native_s2s or streaming_stt+tts sidecar capability",
                "manifest",
            )
        )

    required_tts_languages = {
        metadata["language"]
        for metadata in ALPHA_REQUIRED_TTS_METADATA.values()
        if metadata.get("language")
    }
    if not required_tts_languages:
        return issues
    frontend = health.get("frontend") if isinstance(health.get("frontend"), Mapping) else {}
    configured_languages = _primary_language_set(frontend.get("tts_model_languages", []))
    configured_languages.update(_primary_language_set(capabilities.get("output_languages", [])))
    if not configured_languages:
        return [
            RealtimeVoiceSmokeReportIssue(
                "manifest",
                "missing output_languages or tts_model_languages for EN/JA TTS routing",
                "manifest",
            )
        ]
    for language in sorted(required_tts_languages):
        if language.lower() not in configured_languages:
            issues.append(
                RealtimeVoiceSmokeReportIssue(
                    "manifest",
                    f"missing TTS model route for {language}",
                    "manifest",
                )
            )
    return issues


def _primary_language_set(values: Any) -> set[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return set()
    return {
        language
        for item in values
        if isinstance(item, str)
        for language in [item.strip().lower().split("-", 1)[0]]
        if language
    }


def _entries_by_kind(entries: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    by_kind: dict[str, list[Mapping[str, Any]]] = {}
    for entry in entries:
        kind = str(entry.get("kind") or "").strip()
        if not kind:
            continue
        by_kind.setdefault(kind, []).append(entry)
    return by_kind


def _first_entry_by_kind(entries: Sequence[Mapping[str, Any]], kind: str) -> Mapping[str, Any] | None:
    for entry in entries:
        if str(entry.get("kind") or "").strip() == kind:
            return entry
    return None


def realtime_voice_alpha_manifest_fingerprint(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the profile fields that make alpha evidence apply to one stack."""

    sidecar = entry.get("sidecar") if isinstance(entry.get("sidecar"), Mapping) else {}
    health = sidecar.get("health") if isinstance(sidecar.get("health"), Mapping) else {}
    frontend = health.get("frontend") if isinstance(health.get("frontend"), Mapping) else {}
    capabilities = health.get("capabilities") if isinstance(health.get("capabilities"), Mapping) else {}
    conversation_quality = (
        entry.get("conversation_quality")
        if isinstance(entry.get("conversation_quality"), Mapping)
        else {}
    )
    return (
        str(entry.get("engine") or ""),
        str(entry.get("frontend_provider") or ""),
        str(entry.get("frontend_model") or ""),
        str(conversation_quality.get("mode") or ""),
        str(sidecar.get("mode") or ""),
        capabilities.get("native_s2s") is True,
        capabilities.get("streaming_stt") is True,
        capabilities.get("tts") is True,
        tuple(sorted(_primary_language_set(capabilities.get("output_languages", [])))),
        tuple(sorted(_primary_language_set(frontend.get("tts_model_languages", [])))),
    )


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


def _validate_audio_fixture_entry(
    entry: Mapping[str, Any],
    *,
    max_target_ms: int | None = None,
) -> list[RealtimeVoiceSmokeReportIssue]:
    identifier = str(entry.get("fixture") or "audio_fixture")
    issues = _validate_common_ok(entry, kind="audio_fixture", identifier=identifier)
    events = _events(entry)
    if "transcript.partial" not in events:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_fixture", "missing transcript.partial event", identifier))
    if "transcript.final" not in events:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_fixture", "missing transcript.final event", identifier))
    if _positive_int(entry.get("audio_bytes")) is None:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_fixture", "missing audio bytes", identifier))
    expected_text = ALPHA_REQUIRED_AUDIO_FIXTURE_TEXTS.get(identifier)
    if expected_text:
        final_text = str(entry.get("final_text") or "").strip()
        if not final_text:
            issues.append(
                RealtimeVoiceSmokeReportIssue(
                    "audio_fixture",
                    "missing final_text for required fixture",
                    identifier,
                )
            )
        elif not _transcript_matches_expected(final_text, expected_text):
            issues.append(
                RealtimeVoiceSmokeReportIssue(
                    "audio_fixture",
                    "final_text did not match expected fixture transcript",
                    identifier,
                )
            )
    partial_ms = _positive_int(entry.get("transcript_partial_ms"))
    target_ms = _positive_int(entry.get("target_ms"))
    if partial_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_fixture", "missing transcript_partial_ms", identifier))
    if target_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_fixture", "missing target_ms", identifier))
    elif max_target_ms is not None and target_ms > max_target_ms:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "audio_fixture",
                f"target_ms {target_ms} exceeds alpha ceiling {max_target_ms}",
                identifier,
            )
        )
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


def _validate_tts_entry(
    entry: Mapping[str, Any],
    *,
    max_target_ms: int | None = None,
) -> list[RealtimeVoiceSmokeReportIssue]:
    identifier = str(entry.get("text") or "tts")
    issues = _validate_common_ok(entry, kind="tts", identifier=identifier)
    expected_metadata = ALPHA_REQUIRED_TTS_METADATA.get(str(entry.get("text") or ""))
    if expected_metadata:
        for key, expected_value in expected_metadata.items():
            actual = str(entry.get(key) or "")
            if actual != expected_value:
                issues.append(
                    RealtimeVoiceSmokeReportIssue(
                        "tts",
                        f"missing {key}={expected_value} metadata",
                        identifier,
                    )
                )
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
    elif max_target_ms is not None and target_ms > max_target_ms:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "tts",
                f"target_ms {target_ms} exceeds alpha ceiling {max_target_ms}",
                identifier,
            )
        )
    elif first_audio_ms is not None and first_audio_ms > target_ms:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "tts",
                f"first_audio_ms {first_audio_ms} exceeds target {target_ms}",
                identifier,
            )
        )
    return issues


def _validate_session_turn_entry(
    entry: Mapping[str, Any],
    *,
    max_first_audio_target_ms: int | None = None,
    max_first_text_target_ms: int | None = None,
) -> list[RealtimeVoiceSmokeReportIssue]:
    identifier = str(entry.get("text") or "session_turn")
    issues = _validate_common_ok(entry, kind="session_turn", identifier=identifier)
    expected_metadata = ALPHA_REQUIRED_SESSION_TURN_METADATA.get(str(entry.get("text") or ""))
    if expected_metadata:
        for key, expected_value in expected_metadata.items():
            actual = str(entry.get(key) or "")
            if actual != expected_value:
                issues.append(
                    RealtimeVoiceSmokeReportIssue(
                        "session_turn",
                        f"missing {key}={expected_value} metadata",
                        identifier,
                    )
                )
    events = _events(entry)
    if "transcript.final" not in events:
        issues.append(RealtimeVoiceSmokeReportIssue("session_turn", "missing transcript.final event", identifier))
    if "assistant.text.partial" not in events:
        issues.append(
            RealtimeVoiceSmokeReportIssue("session_turn", "missing assistant.text.partial event", identifier)
        )
    if "audio.output.chunk" not in events:
        issues.append(RealtimeVoiceSmokeReportIssue("session_turn", "missing audio.output.chunk event", identifier))
    if _positive_int(entry.get("output_audio_bytes")) is None:
        issues.append(RealtimeVoiceSmokeReportIssue("session_turn", "missing output audio bytes", identifier))

    first_text_ms = _nonnegative_int(entry.get("first_text_ms"))
    first_text_target_ms = _positive_int(entry.get("first_text_target_ms"))
    if first_text_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("session_turn", "missing first_text_ms", identifier))
    if first_text_target_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("session_turn", "missing first_text_target_ms", identifier))
    elif max_first_text_target_ms is not None and first_text_target_ms > max_first_text_target_ms:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "session_turn",
                f"first_text_target_ms {first_text_target_ms} exceeds alpha ceiling {max_first_text_target_ms}",
                identifier,
            )
        )
    elif first_text_ms is not None and first_text_ms > first_text_target_ms:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "session_turn",
                f"first_text_ms {first_text_ms} exceeds target {first_text_target_ms}",
                identifier,
            )
        )

    first_audio_ms = _nonnegative_int(entry.get("first_audio_ms"))
    first_audio_target_ms = _positive_int(entry.get("first_audio_target_ms"))
    if first_audio_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("session_turn", "missing first_audio_ms", identifier))
    if first_audio_target_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("session_turn", "missing first_audio_target_ms", identifier))
    elif max_first_audio_target_ms is not None and first_audio_target_ms > max_first_audio_target_ms:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "session_turn",
                f"first_audio_target_ms {first_audio_target_ms} exceeds alpha ceiling {max_first_audio_target_ms}",
                identifier,
            )
        )
    elif first_audio_ms is not None and first_audio_ms > first_audio_target_ms:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "session_turn",
                f"first_audio_ms {first_audio_ms} exceeds target {first_audio_target_ms}",
                identifier,
            )
        )
    return issues


def _validate_barge_in_entry(
    entry: Mapping[str, Any],
    *,
    max_target_ms: int | None = None,
) -> list[RealtimeVoiceSmokeReportIssue]:
    identifier = str(entry.get("text") or "barge_in")
    issues = _validate_common_ok(entry, kind="barge_in", identifier=identifier)
    events = _events(entry)
    if "barge_in" not in events:
        issues.append(RealtimeVoiceSmokeReportIssue("barge_in", "missing barge_in event", identifier))
    audio_after_barge_in_bytes = _nonnegative_int(entry.get("audio_after_barge_in_bytes"))
    if audio_after_barge_in_bytes is None:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "barge_in",
                "missing audio_after_barge_in_bytes",
                identifier,
            )
        )
    elif audio_after_barge_in_bytes > 0:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "barge_in",
                f"audio.output.chunk arrived after barge_in ({audio_after_barge_in_bytes} byte(s))",
                identifier,
            )
        )
    ack_ms = _nonnegative_int(entry.get("barge_in_ack_ms"))
    target_ms = _positive_int(entry.get("target_ms"))
    if ack_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("barge_in", "missing barge_in_ack_ms", identifier))
    if target_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("barge_in", "missing target_ms", identifier))
    elif max_target_ms is not None and target_ms > max_target_ms:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "barge_in",
                f"target_ms {target_ms} exceeds alpha ceiling {max_target_ms}",
                identifier,
            )
        )
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


def _transcript_matches_expected(actual: str, expected: str) -> bool:
    actual_norm = _normalize_transcript_text(actual)
    expected_norm = _normalize_transcript_text(expected)
    return bool(actual_norm and expected_norm and actual_norm == expected_norm)


_TRANSCRIPT_PUNCTUATION_RE = re.compile(r"[\s\.,!?;:'\"`~()\[\]{}<>/\\|_\-+=、。，．！？；：『』「」（）【】]")


def _normalize_transcript_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return _TRANSCRIPT_PUNCTUATION_RE.sub("", normalized)


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


def _latency_summary(values: Iterable[Any]) -> dict[str, Any]:
    parsed = sorted(value for value in (_nonnegative_int(item) for item in values) if value is not None)
    if not parsed:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    return {
        "count": len(parsed),
        "p50": _percentile_nearest_rank(parsed, 50),
        "p95": _percentile_nearest_rank(parsed, 95),
        "max": parsed[-1],
    }


def _percentile_nearest_rank(values: Sequence[int], percentile: int) -> int:
    if not values:
        raise ValueError("percentile requires at least one value")
    index = max(0, min(len(values) - 1, math.ceil((percentile / 100) * len(values)) - 1))
    return values[index]
