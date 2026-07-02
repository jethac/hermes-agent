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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agent.realtime_voice_kame import kame_local_reply_denies_voice_capability

OUTPUT_AUDIO_EVENT_NAMES = frozenset({"audio.output.chunk", "assistant.audio.chunk"})


ALPHA_REQUIRED_AUDIO_FIXTURES = (
    "./fixtures/realtime-voice/en/hello.webm",
    "./fixtures/realtime-voice/en/tool-question.webm",
    "./fixtures/realtime-voice/ja/hello.webm",
    "./fixtures/realtime-voice/ja/tool-question.webm",
)

ALPHA_REQUIRED_AUDIO_SESSION_FIXTURES = (
    "./fixtures/realtime-voice/en/hello.webm",
    "./fixtures/realtime-voice/ja/hello.webm",
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

KAME_LATENCY_METRICS = (
    ("first_audio_to_interface_decision", "kame_first_audio_to_interface_decision_ms"),
    ("speech_boundary_to_final_transcript", "eou_to_final_transcript_ms"),
    ("speech_boundary_to_final_intent", "kame_speech_boundary_to_final_intent_ms"),
    ("speech_end_to_interface_decision", "kame_speech_end_to_interface_decision_ms"),
    ("final_transcript_to_interface_decision", "kame_final_transcript_to_interface_decision_ms"),
    ("speech_end_to_local_first_audio", "kame_speech_end_to_local_first_audio_ms"),
    ("speech_end_to_first_audio", "kame_speech_end_to_first_audio_ms"),
    ("interface_decision_to_local_first_audio", "kame_interface_decision_to_local_first_audio_ms"),
    ("interface_decision_to_first_audio", "kame_interface_decision_to_first_audio_ms"),
    ("interface_decision_to_oracle_accepted", "kame_interface_decision_to_oracle_accepted_ms"),
    ("oracle_accepted_to_first_token", "kame_oracle_accepted_to_first_token_ms"),
    ("oracle_first_token_to_first_spoken_text", "kame_oracle_first_token_to_first_spoken_text_ms"),
    ("oracle_first_token_to_first_tts_audio", "kame_oracle_first_token_to_first_tts_audio_ms"),
    ("first_tts_audio_to_playback_start", "kame_first_tts_audio_to_playback_start_ms"),
    ("speech_end_to_playback_start", "kame_speech_end_to_playback_start_ms"),
    ("oracle_total_stream", "kame_oracle_total_stream_ms"),
    ("oracle_verbatim_asr", "oracle_verbatim_asr_ms"),
    ("barge_in_confirmed_to_playback_stopped", "barge_in_confirmed_to_playback_stopped_ms"),
)

KAME_LATENCY_REPORT_LABELS = (
    "first_audio_to_interface_decision",
    "speech_boundary_to_final_transcript",
    "speech_boundary_to_final_intent",
    "speech_end_to_interface_decision",
    "final_transcript_to_interface_decision",
    "speech_end_to_local_first_audio",
    "speech_end_to_first_audio",
    "interface_decision_to_local_first_audio",
    "interface_decision_to_first_audio",
    "interface_decision_to_oracle_accepted",
    "oracle_accepted_to_first_token",
    "oracle_first_token_to_first_spoken_text",
    "oracle_first_token_to_first_tts_audio",
    "first_tts_audio_to_playback_start",
    "speech_end_to_playback_start",
    "oracle_total_stream",
    "oracle_verbatim_asr",
    "barge_in_confirmed_to_playback_stopped",
)

KAME_ROUTE_LABELS = ("local", "defer", "oracle_direct", "reject_or_clarify")
KAME_ORACLE_AVOIDING_ROUTES = frozenset({"local", "reject_or_clarify"})

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

_EVIDENCE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")


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
    max_collected_age_days: int | None = None,
    allow_loopback_validation: bool = True,
    require_async_oracle_smoke: bool = False,
    now: datetime | None = None,
) -> list[RealtimeVoiceSmokeReportIssue]:
    issues: list[RealtimeVoiceSmokeReportIssue] = []
    required_runs = max(1, int(min_runs or 1))
    max_age_days = _positive_int(max_collected_age_days)
    current_time = now if isinstance(now, datetime) else datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    if len(runs) < required_runs:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "evidence",
                f"requires at least {required_runs} run(s), found {len(runs)}",
            )
        )
    for label, entries in runs:
        for issue in validate_realtime_voice_alpha_report(
            entries,
            require_async_oracle_smoke=require_async_oracle_smoke,
        ):
            issues.append(
                RealtimeVoiceSmokeReportIssue(
                    issue.kind,
                    issue.message,
                    f"{label}: {issue.identifier}" if issue.identifier else label,
                )
            )
        if not allow_loopback_validation and _report_is_loopback_validation(entries):
            issues.append(
                RealtimeVoiceSmokeReportIssue(
                    "evidence",
                    "loopback validation cannot satisfy production evidence",
                    label,
                )
            )
    fingerprints: dict[tuple[Any, ...], str] = {}
    run_ids: dict[str, str] = {}
    for label, entries in runs:
        manifest = _first_entry_by_kind(entries, "manifest")
        if manifest is None:
            continue
        fingerprint = realtime_voice_alpha_manifest_fingerprint(manifest)
        if fingerprint not in fingerprints:
            fingerprints[fingerprint] = label
        run_id = _manifest_run_id(manifest)
        if run_id:
            previous_label = run_ids.get(run_id)
            if previous_label is not None:
                issues.append(
                    RealtimeVoiceSmokeReportIssue(
                        "evidence",
                        "alpha runs reused evidence run_id",
                        f"{previous_label}, {label}: {run_id}",
                    )
                )
            else:
                run_ids[run_id] = label
        collected_at = _parse_manifest_timestamp(str(manifest.get("collected_at") or ""))
        if collected_at is not None and max_age_days is not None:
            age_seconds = (current_time - collected_at).total_seconds()
            max_age_seconds = max_age_days * 24 * 60 * 60
            if age_seconds > max_age_seconds:
                issues.append(
                    RealtimeVoiceSmokeReportIssue(
                        "evidence",
                        f"alpha run evidence is older than {max_age_days} day(s)",
                        f"{label}: {manifest.get('collected_at')}",
                    )
                )
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


def _report_is_loopback_validation(entries: Sequence[Mapping[str, Any]]) -> bool:
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("loopback_validation") is True:
            return True
        if str(entry.get("evidence_provider") or "").strip().lower() == "loopback":
            return True
    return False


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
        "latency_ms": _latency_summary_for_entries(entries),
        "kame_routes": _kame_route_summary(entries),
        "kame_reflex_provenance": _kame_reflex_provenance_summary(entries),
        "latency_by_stack": _latency_summary_by_stack(runs),
    }


def _latency_summary_for_entries(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_kind = _entries_by_kind(entries)
    latency = {
        "audio_to_partial_transcript": _latency_summary(
            entry.get("transcript_partial_ms")
            for entry in [*by_kind.get("audio_fixture", []), *by_kind.get("audio_session", [])]
        ),
        "final_transcript_to_first_text": _latency_summary(
            entry.get("first_text_ms")
            for entry in [*by_kind.get("session_turn", []), *by_kind.get("audio_session", [])]
        ),
        "final_transcript_to_first_audio": _latency_summary(
            entry.get("first_audio_ms")
            for entry in [
                *by_kind.get("session_turn", []),
                *by_kind.get("audio_session", []),
                *by_kind.get("tts", []),
            ]
        ),
        "barge_in_ack": _latency_summary(
            entry.get("barge_in_ack_ms")
            for entry in by_kind.get("barge_in", [])
        ),
    }
    for label, metric_key in KAME_LATENCY_METRICS:
        metric = _latency_summary(_entry_metric_values(entries, metric_key))
        if metric.get("count"):
            latency[label] = metric
    return latency


def _latency_summary_by_stack(
    runs: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for label, report_entries in runs:
        manifest = _first_entry_by_kind(report_entries, "manifest") or {}
        stack = _manifest_stack_summary(manifest)
        key = _stack_summary_key(stack)
        bucket = grouped.setdefault(
            key,
            {
                "stack": stack,
                "runs": 0,
                "entries": [],
                "report_labels": [],
            },
        )
        bucket["runs"] += 1
        bucket["report_labels"].append(str(label))
        bucket["entries"].extend(
            entry
            for entry in report_entries
            if str(entry.get("kind") or "") != "manifest"
        )
    return {
        key: {
            "stack": bucket["stack"],
            "runs": bucket["runs"],
            "report_labels": bucket["report_labels"],
            "latency_ms": _latency_summary_for_entries(bucket["entries"]),
            "kame_routes": _kame_route_summary(bucket["entries"]),
            "kame_reflex_provenance": _kame_reflex_provenance_summary(bucket["entries"]),
        }
        for key, bucket in sorted(grouped.items())
    }


def _kame_route_summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {route: 0 for route in KAME_ROUTE_LABELS}
    total = 0
    for entry in entries:
        route = str(entry.get("route") or "").strip().lower()
        if route not in counts:
            continue
        counts[route] += 1
        total += 1
    oracle_avoided = sum(counts[route] for route in KAME_ORACLE_AVOIDING_ROUTES)
    oracle_required = total - oracle_avoided
    return {
        "total": total,
        "counts": counts,
        "oracle_avoided": oracle_avoided,
        "oracle_required": oracle_required,
        "oracle_avoidance_rate": round(oracle_avoided / total, 4) if total else None,
    }


def _kame_reflex_provenance_summary(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    routed_entries = [
        entry
        for entry in entries
        if str(entry.get("route") or "").strip().lower()
    ]
    input_sources: dict[str, int] = {}
    reflex_providers: dict[str, int] = {}
    fallback_count = 0
    for entry in routed_entries:
        input_source = str(entry.get("interface_input_source") or "").strip().lower()
        reflex_provider = str(entry.get("reflex_provider") or "").strip().lower()
        if input_source:
            input_sources[input_source] = input_sources.get(input_source, 0) + 1
        if reflex_provider:
            reflex_providers[reflex_provider] = reflex_providers.get(reflex_provider, 0) + 1
        if (
            entry.get("interface_audio_input_fallback") is True
            or input_source in {"local_stt", "streaming_stt"}
        ):
            fallback_count += 1
    total = len(routed_entries)
    native_audio_count = input_sources.get("native_audio", 0)
    vllm_count = reflex_providers.get("vllm", 0)
    return {
        "total": total,
        "input_sources": dict(sorted(input_sources.items())),
        "reflex_providers": dict(sorted(reflex_providers.items())),
        "native_audio": native_audio_count,
        "vllm": vllm_count,
        "fallback": fallback_count,
        "fallback_only": bool(total and fallback_count == total),
    }


def validate_realtime_voice_smoke_report(
    entries: Sequence[Mapping[str, Any]],
    *,
    required_audio_fixtures: Iterable[str] = (),
    required_tts_texts: Iterable[str] = (),
    required_barge_in_texts: Iterable[str] = (),
    required_session_turn_texts: Iterable[str] = (),
    required_audio_session_fixtures: Iterable[str] = (),
    require_protocol: bool = True,
    require_manifest: bool = False,
    require_alpha_targets: bool = False,
    require_async_oracle_smoke: bool = False,
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

    alpha_target_ceilings_ms = (
        _alpha_quality_target_ceilings_ms(manifest_entries[0])
        if require_alpha_targets and manifest_entries
        else dict(ALPHA_REQUIRED_QUALITY_TARGETS_MS)
    )

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
    audio_session_entries = by_kind.get("audio_session", [])
    async_oracle_entries = by_kind.get("async_oracle_smoke", [])
    if require_alpha_targets and manifest_entries and _manifest_entry_is_kame_reflex(manifest_entries[0]):
        kame_route_entries = [*session_turn_entries, *audio_session_entries]
        issues.extend(_validate_kame_route_evidence(kame_route_entries))
        issues.extend(_validate_kame_reflex_provenance(kame_route_entries))
        issues.extend(_validate_kame_capability_honesty(entries))
    if require_async_oracle_smoke and not async_oracle_entries:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "async_oracle_smoke",
                "missing async oracle smoke proof",
                "async_oracle_smoke",
            )
        )

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
    issues.extend(_validate_required_entries(
        entries=audio_session_entries,
        required=required_audio_session_fixtures,
        field="fixture",
        kind="audio_session",
    ))

    for entry in audio_entries:
        issues.extend(_validate_audio_fixture_entry(
            entry,
            max_target_ms=(
                alpha_target_ceilings_ms["audio_to_partial_transcript_ms"]
                if require_alpha_targets
                else None
            ),
        ))
    for entry in tts_entries:
        issues.extend(_validate_tts_entry(
            entry,
            max_target_ms=(
                alpha_target_ceilings_ms["final_transcript_to_first_audio_ms"]
                if require_alpha_targets
                else None
            ),
        ))
    for entry in session_turn_entries:
        issues.extend(_validate_session_turn_entry(
            entry,
            max_first_text_target_ms=(
                alpha_target_ceilings_ms["final_transcript_to_first_text_ms"]
                if require_alpha_targets
                else None
            ),
            max_first_audio_target_ms=(
                alpha_target_ceilings_ms["final_transcript_to_first_audio_ms"]
                if require_alpha_targets
                else None
            ),
        ))
    for entry in audio_session_entries:
        issues.extend(_validate_audio_session_entry(
            entry,
            max_partial_target_ms=(
                alpha_target_ceilings_ms["audio_to_partial_transcript_ms"]
                if require_alpha_targets
                else None
            ),
            max_first_text_target_ms=(
                alpha_target_ceilings_ms["final_transcript_to_first_text_ms"]
                if require_alpha_targets
                else None
            ),
            max_first_audio_target_ms=(
                alpha_target_ceilings_ms["final_transcript_to_first_audio_ms"]
                if require_alpha_targets
                else None
            ),
        ))
    for entry in barge_in_entries:
        issues.extend(_validate_barge_in_entry(
            entry,
            max_target_ms=(
                alpha_target_ceilings_ms["barge_in_ack_ms"]
                if require_alpha_targets
                else None
            ),
        ))
    for entry in async_oracle_entries:
        issues.extend(_validate_async_oracle_smoke_entry(entry))

    return issues


def _validate_kame_route_evidence(entries: Sequence[Mapping[str, Any]]) -> list[RealtimeVoiceSmokeReportIssue]:
    summary = _kame_route_summary(entries)
    issues: list[RealtimeVoiceSmokeReportIssue] = []
    if summary["total"] <= 0:
        return [
            RealtimeVoiceSmokeReportIssue(
                "kame_routes",
                "missing KAME route evidence",
                "local/defer/oracle_direct/reject_or_clarify",
            )
        ]
    if summary["oracle_avoided"] <= 0:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "kame_routes",
                "missing oracle-avoiding local or clarify route evidence",
                "local/reject_or_clarify",
            )
        )
    if summary["oracle_required"] <= 0:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "kame_routes",
                "missing oracle-bound defer or direct route evidence",
                "defer/oracle_direct",
            )
        )
    return issues


def _validate_kame_reflex_provenance(entries: Sequence[Mapping[str, Any]]) -> list[RealtimeVoiceSmokeReportIssue]:
    routed_entries = [
        entry
        for entry in entries
        if str(entry.get("route") or "").strip().lower()
    ]
    if not routed_entries:
        return []
    native_audio_entries = [
        entry
        for entry in routed_entries
        if str(entry.get("interface_input_source") or "").strip().lower() == "native_audio"
    ]
    vllm_reflex_entries = [
        entry
        for entry in routed_entries
        if str(entry.get("reflex_provider") or "").strip().lower() == "vllm"
    ]
    fallback_entries = [
        entry
        for entry in routed_entries
        if entry.get("interface_audio_input_fallback") is True
        or str(entry.get("interface_input_source") or "").strip().lower() in {"local_stt", "streaming_stt"}
    ]
    malformed_entries = [
        entry
        for entry in routed_entries
        if str(entry.get("reflex_validation_error") or "").strip()
    ]
    issues: list[RealtimeVoiceSmokeReportIssue] = []
    if not native_audio_entries:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "kame_reflex_provenance",
                "missing native-audio reflex route evidence",
                "interface_input_source=native_audio",
            )
        )
    if not vllm_reflex_entries:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "kame_reflex_provenance",
                "missing vLLM reflex provider route evidence",
                "reflex_provider=vllm",
            )
        )
    if fallback_entries and len(fallback_entries) == len(routed_entries):
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "kame_reflex_provenance",
                "KAME route evidence used only fallback reflex input",
                "interface_audio_input_fallback",
            )
        )
    if malformed_entries:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "kame_reflex_provenance",
                "KAME route evidence includes malformed reflex output",
                _kame_reflex_error_summary(malformed_entries),
            )
        )
    return issues


def _kame_reflex_error_summary(entries: Sequence[Mapping[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for entry in entries:
        error = str(entry.get("reflex_validation_error") or "").strip() or "unknown"
        counts[error] = counts.get(error, 0) + 1
    return ", ".join(f"{error}={count}" for error, count in sorted(counts.items()))


def _validate_kame_capability_honesty(entries: Sequence[Mapping[str, Any]]) -> list[RealtimeVoiceSmokeReportIssue]:
    issues: list[RealtimeVoiceSmokeReportIssue] = []
    for entry in entries:
        for key, text in _assistant_output_texts(entry):
            if not kame_local_reply_denies_voice_capability(text):
                continue
            identifier = str(entry.get("kind") or "entry")
            if key:
                identifier = f"{identifier}.{key}"
            issues.append(
                RealtimeVoiceSmokeReportIssue(
                    "kame_capability_honesty",
                    "assistant output denied live voice capability",
                    identifier,
                )
            )
    return issues


_ASSISTANT_OUTPUT_TEXT_KEYS = (
    "assistant_text",
    "assistant_response",
    "assistant_final_text",
    "final_assistant_text",
    "interface_reply",
    "local_reply",
    "oracle_hint",
    "oracle_response",
    "output_text",
    "reply",
    "response_text",
    "spoken_text",
)


def _assistant_output_texts(entry: Mapping[str, Any]) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = []
    for key in _ASSISTANT_OUTPUT_TEXT_KEYS:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            texts.append((key, value.strip()))
    kind = str(entry.get("kind") or "").strip()
    if kind in {"session_turn", "barge_in"}:
        value = entry.get("final_text")
        if isinstance(value, str) and value.strip():
            texts.append(("final_text", value.strip()))
    if kind == "tts":
        value = entry.get("text")
        if isinstance(value, str) and value.strip():
            texts.append(("text", value.strip()))
    return texts


def validate_realtime_voice_alpha_report(
    entries: Sequence[Mapping[str, Any]],
    *,
    require_async_oracle_smoke: bool = False,
) -> list[RealtimeVoiceSmokeReportIssue]:
    return validate_realtime_voice_smoke_report(
        entries,
        required_audio_fixtures=ALPHA_REQUIRED_AUDIO_FIXTURES,
        required_tts_texts=ALPHA_REQUIRED_TTS_TEXTS,
        required_barge_in_texts=ALPHA_REQUIRED_BARGE_IN_TEXTS,
        required_session_turn_texts=ALPHA_REQUIRED_SESSION_TURN_TEXTS,
        required_audio_session_fixtures=ALPHA_REQUIRED_AUDIO_SESSION_FIXTURES,
        require_protocol=True,
        require_manifest=True,
        require_alpha_targets=True,
        require_async_oracle_smoke=require_async_oracle_smoke,
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
    run_id = _manifest_run_id(entry)
    if not run_id:
        issues.append(RealtimeVoiceSmokeReportIssue("manifest", "missing valid evidence run_id", "manifest"))
    collected_at = str(entry.get("collected_at") or "").strip()
    if not collected_at:
        issues.append(RealtimeVoiceSmokeReportIssue("manifest", "missing collected_at timestamp", "manifest"))
    elif not _valid_manifest_timestamp(collected_at):
        issues.append(RealtimeVoiceSmokeReportIssue("manifest", "invalid collected_at timestamp", "manifest"))
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
        target_ceilings_ms = _alpha_quality_target_ceilings_ms(entry)
        for key, ceiling in target_ceilings_ms.items():
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
    if health.get("ok") is not True:
        issues.append(RealtimeVoiceSmokeReportIssue("manifest", "manifest sidecar health was not ok", "manifest"))
    capabilities = health.get("capabilities") if isinstance(health.get("capabilities"), Mapping) else {}
    engine = str(entry.get("engine") or "")
    has_kame_reflex = (
        engine == "kame_interface_oracle"
        and capabilities.get("vllm_audio_frontend") is True
        and capabilities.get("tts") is True
    )
    if (
        capabilities.get("native_s2s") is not True
        and not (capabilities.get("streaming_stt") is True and capabilities.get("tts") is True)
        and not has_kame_reflex
    ):
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "manifest",
                "missing native_s2s, streaming_stt+tts, or kame_reflex+tts sidecar capability",
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


def _manifest_entry_is_kame_reflex(entry: Mapping[str, Any]) -> bool:
    conversation_quality = (
        entry.get("conversation_quality")
        if isinstance(entry.get("conversation_quality"), Mapping)
        else {}
    )
    return (
        str(entry.get("engine") or "") == "kame_interface_oracle"
        or str(conversation_quality.get("mode") or "") == "kame_reflex"
    )


def _alpha_quality_target_ceilings_ms(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Return per-report alpha target ceilings declared in the evidence manifest."""
    ceilings = dict(ALPHA_REQUIRED_QUALITY_TARGETS_MS)
    raw = manifest.get("quality_target_ceilings_ms")
    if not isinstance(raw, Mapping):
        return ceilings
    for key, default in ALPHA_REQUIRED_QUALITY_TARGETS_MS.items():
        value = _positive_int(raw.get(key))
        if value is not None:
            ceilings[key] = max(default, value)
    return ceilings


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


def _manifest_run_id(entry: Mapping[str, Any]) -> str:
    value = str(entry.get("run_id") or "").strip()
    return value if _EVIDENCE_RUN_ID_RE.fullmatch(value) else ""


def _valid_manifest_timestamp(value: str) -> bool:
    return _parse_manifest_timestamp(value) is not None


def _parse_manifest_timestamp(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


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
        str(entry.get("interface_audio_input") or ""),
        *_kame_interface_config_fingerprint(entry),
        str(entry.get("asr_mode") or ""),
        str(entry.get("preferred_local_oracle_model") or ""),
        str(conversation_quality.get("mode") or ""),
        str(sidecar.get("mode") or ""),
        capabilities.get("native_s2s") is True,
        capabilities.get("vllm_audio_frontend") is True,
        capabilities.get("streaming_stt") is True,
        capabilities.get("tts") is True,
        tuple(sorted(_primary_language_set(capabilities.get("output_languages", [])))),
        tuple(sorted(_primary_language_set(frontend.get("tts_model_languages", [])))),
    )


def _kame_interface_config_fingerprint(entry: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    conversation_quality = (
        entry.get("conversation_quality")
        if isinstance(entry.get("conversation_quality"), Mapping)
        else {}
    )
    is_kame = (
        str(entry.get("engine") or "") == "kame_interface_oracle"
        or str(conversation_quality.get("mode") or "") == "kame_reflex"
    )
    if not is_kame:
        return ("", "", "", "", "")
    return (
        str(entry.get("interface_base_url") or ""),
        _fingerprint_number(entry.get("interface_temperature")),
        _fingerprint_number(entry.get("interface_max_output_tokens")),
        _fingerprint_number(entry.get("interface_timeout_seconds")),
        _fingerprint_number(entry.get("interface_max_audio_seconds")),
    )


def _fingerprint_number(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{parsed:g}"


def _manifest_stack_summary(entry: Mapping[str, Any]) -> dict[str, str]:
    sidecar = entry.get("sidecar") if isinstance(entry.get("sidecar"), Mapping) else {}
    health = sidecar.get("health") if isinstance(sidecar.get("health"), Mapping) else {}
    frontend = health.get("frontend") if isinstance(health.get("frontend"), Mapping) else {}
    return {
        "engine": str(entry.get("engine") or ""),
        "frontend_provider": str(entry.get("frontend_provider") or frontend.get("provider") or ""),
        "frontend_model": str(entry.get("frontend_model") or frontend.get("model") or ""),
        "interface_audio_input": str(entry.get("interface_audio_input") or ""),
        "asr_mode": str(entry.get("asr_mode") or ""),
        "asr_provider": str(entry.get("asr_provider") or ""),
        "asr_model": str(entry.get("asr_model") or ""),
        "preferred_local_oracle_model": str(entry.get("preferred_local_oracle_model") or ""),
        "tts_provider": str(entry.get("tts_provider") or ""),
        "tts_model": str(entry.get("tts_model") or ""),
        "tts_voice": str(entry.get("tts_voice") or ""),
    }


def _stack_summary_key(stack: Mapping[str, str]) -> str:
    parts = [
        stack.get("engine") or "unknown_engine",
        stack.get("frontend_provider") or "unknown_frontend",
        stack.get("frontend_model") or "unknown_model",
        stack.get("preferred_local_oracle_model") or "unknown_oracle",
        stack.get("tts_provider") or "unknown_tts",
        stack.get("tts_model") or "unknown_tts_model",
    ]
    return "|".join(_slug_token(part) for part in parts)


def _slug_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._:-]+", "_", str(value or "").strip())
    return token.strip("_") or "unknown"


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
    if not _has_final_user_turn_event(events):
        issues.append(RealtimeVoiceSmokeReportIssue("protocol", "missing final user turn event", identifier))
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
    partial_ms = _positive_int(entry.get("transcript_partial_ms"))
    target_ms = _positive_int(entry.get("target_ms"))
    final_ms = _positive_int(entry.get("transcript_final_ms"))
    fast_final_satisfies_partial = _final_transcript_satisfies_partial_target(final_ms, target_ms)
    if "transcript.partial" not in events and not fast_final_satisfies_partial:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_fixture", "missing transcript.partial event", identifier))
    if not _has_final_user_turn_event(events):
        issues.append(RealtimeVoiceSmokeReportIssue("audio_fixture", "missing final user turn event", identifier))
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
    if partial_ms is None and not fast_final_satisfies_partial:
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
    if final_ms is None:
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
    if not _has_output_audio_event(events):
        issues.append(RealtimeVoiceSmokeReportIssue("tts", "missing output audio event", identifier))
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
    if not _has_final_user_turn_event(events):
        issues.append(RealtimeVoiceSmokeReportIssue("session_turn", "missing final user turn event", identifier))
    if "assistant.text.partial" not in events:
        issues.append(
            RealtimeVoiceSmokeReportIssue("session_turn", "missing assistant.text.partial event", identifier)
        )
    if not _has_output_audio_event(events):
        issues.append(RealtimeVoiceSmokeReportIssue("session_turn", "missing output audio event", identifier))
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


def _validate_audio_session_entry(
    entry: Mapping[str, Any],
    *,
    max_partial_target_ms: int | None = None,
    max_first_audio_target_ms: int | None = None,
    max_first_text_target_ms: int | None = None,
) -> list[RealtimeVoiceSmokeReportIssue]:
    identifier = str(entry.get("fixture") or "audio_session")
    issues = _validate_common_ok(entry, kind="audio_session", identifier=identifier)
    events = _events(entry)
    partial_ms = _nonnegative_int(entry.get("transcript_partial_ms"))
    partial_target_ms = _positive_int(entry.get("target_ms"))
    final_ms = _positive_int(entry.get("transcript_final_ms"))
    fast_final_satisfies_partial = _final_transcript_satisfies_partial_target(final_ms, partial_target_ms)
    kame_native_audio_reflex = _entry_uses_kame_native_audio_reflex(entry)
    partial_requirement_satisfied = fast_final_satisfies_partial or kame_native_audio_reflex
    if fast_final_satisfies_partial:
        issues = [
            issue for issue in issues
            if issue.message != "unknown realtime voice error"
        ]
    if "transcript.partial" not in events and not partial_requirement_satisfied:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_session", "missing transcript.partial event", identifier))
    if not _has_final_user_turn_event(events):
        issues.append(RealtimeVoiceSmokeReportIssue("audio_session", "missing final user turn event", identifier))
    if "assistant.text.partial" not in events:
        issues.append(
            RealtimeVoiceSmokeReportIssue("audio_session", "missing assistant.text.partial event", identifier)
        )
    if not _has_output_audio_event(events):
        issues.append(RealtimeVoiceSmokeReportIssue("audio_session", "missing output audio event", identifier))
    if _positive_int(entry.get("audio_bytes")) is None:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_session", "missing audio bytes", identifier))
    if _positive_int(entry.get("output_audio_bytes")) is None:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_session", "missing output audio bytes", identifier))
    if final_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_session", "missing transcript_final_ms", identifier))

    expected_text = ALPHA_REQUIRED_AUDIO_FIXTURE_TEXTS.get(identifier)
    if expected_text:
        final_text = str(entry.get("final_text") or "").strip()
        if not final_text:
            issues.append(
                RealtimeVoiceSmokeReportIssue(
                    "audio_session",
                    "missing final_text for required fixture",
                    identifier,
                )
            )
        elif not _transcript_matches_expected(final_text, expected_text):
            issues.append(
                RealtimeVoiceSmokeReportIssue(
                    "audio_session",
                    "final_text did not match expected fixture transcript",
                    identifier,
                )
            )

    if partial_ms is None and not partial_requirement_satisfied:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_session", "missing transcript_partial_ms", identifier))
    if partial_target_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_session", "missing target_ms", identifier))
    elif max_partial_target_ms is not None and partial_target_ms > max_partial_target_ms:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "audio_session",
                f"target_ms {partial_target_ms} exceeds alpha ceiling {max_partial_target_ms}",
                identifier,
            )
        )
    elif partial_ms is not None and partial_ms > partial_target_ms:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "audio_session",
                f"transcript_partial_ms {partial_ms} exceeds target {partial_target_ms}",
                identifier,
            )
        )

    first_text_ms = _nonnegative_int(entry.get("first_text_ms"))
    first_text_target_ms = _positive_int(entry.get("first_text_target_ms"))
    if first_text_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_session", "missing first_text_ms", identifier))
    if first_text_target_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_session", "missing first_text_target_ms", identifier))
    elif max_first_text_target_ms is not None and first_text_target_ms > max_first_text_target_ms:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "audio_session",
                f"first_text_target_ms {first_text_target_ms} exceeds alpha ceiling {max_first_text_target_ms}",
                identifier,
            )
        )
    elif first_text_ms is not None and first_text_ms > first_text_target_ms:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "audio_session",
                f"first_text_ms {first_text_ms} exceeds target {first_text_target_ms}",
                identifier,
            )
        )

    first_audio_ms = _nonnegative_int(entry.get("first_audio_ms"))
    first_audio_target_ms = _positive_int(entry.get("first_audio_target_ms"))
    if first_audio_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_session", "missing first_audio_ms", identifier))
    if first_audio_target_ms is None:
        issues.append(RealtimeVoiceSmokeReportIssue("audio_session", "missing first_audio_target_ms", identifier))
    elif max_first_audio_target_ms is not None and first_audio_target_ms > max_first_audio_target_ms:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "audio_session",
                f"first_audio_target_ms {first_audio_target_ms} exceeds alpha ceiling {max_first_audio_target_ms}",
                identifier,
            )
        )
    elif first_audio_ms is not None and first_audio_ms > first_audio_target_ms:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "audio_session",
                f"first_audio_ms {first_audio_ms} exceeds target {first_audio_target_ms}",
                identifier,
            )
        )
    return issues


def _entry_uses_kame_native_audio_reflex(entry: Mapping[str, Any]) -> bool:
    route = str(entry.get("route") or "").strip().lower()
    input_source = str(entry.get("interface_input_source") or "").strip().lower()
    return route in KAME_ROUTE_LABELS and input_source == "native_audio"


def _validate_barge_in_entry(
    entry: Mapping[str, Any],
    *,
    max_target_ms: int | None = None,
) -> list[RealtimeVoiceSmokeReportIssue]:
    identifier = str(entry.get("text") or "barge_in")
    issues = _validate_common_ok(entry, kind="barge_in", identifier=identifier)
    events = _events(entry)
    if not {"barge_in.detected", "barge_in"}.intersection(events):
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "barge_in",
                "missing barge_in.detected event",
                identifier,
            )
        )
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
                f"output audio event arrived after barge_in.detected ({audio_after_barge_in_bytes} byte(s))",
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


def _validate_async_oracle_smoke_entry(
    entry: Mapping[str, Any],
) -> list[RealtimeVoiceSmokeReportIssue]:
    identifier = str(entry.get("scenario") or entry.get("kind") or "async_oracle_smoke")
    issues = _validate_common_ok(entry, kind="async_oracle_smoke", identifier=identifier)
    if str(entry.get("scenario") or "") != "async_kame_oracle_jobs_fake":
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "async_oracle_smoke",
                "unexpected async oracle smoke scenario",
                identifier,
            )
        )
    required_true_fields = (
        "worker_overlap_proved",
        "worker_overlap_within_capacity",
        "local_turn_during_running_jobs_observed",
        "status_turn_committed",
        "terminal_status_committed",
        "fifth_job_queued",
        "fifth_job_started_after_capacity_freed",
        "queued_job_update_observed",
        "queued_update_latest_update_visible",
        "queued_update_started_with_priority",
        "queued_update_reached_oracle",
        "running_job_update_observed",
        "running_update_latest_update_visible",
        "running_update_reached_oracle",
        "running_update_delivery_metadata_ok",
        "spoken_cancel_control_observed",
        "spoken_priority_control_observed",
        "spoken_update_control_observed",
        "queued_cancel_smoke_ok",
        "queued_cancel_requested_observed",
        "queued_cancel_observed",
        "queued_cancel_spoken_control_observed",
        "queued_cancelled_before_start",
        "queued_cancel_not_sent_to_oracle",
        "queued_cancel_running_completed",
        "shutdown_bounded_close_observed",
        "shutdown_forced_cancel_observed",
        "shutdown_close_cancel_entered",
        "local_turn_committed",
        "completed_result_status_visible",
        "late_cancelled_output_attempted",
        "durable_cancelled_record_present",
        "approval_wait_observed",
        "approval_status_committed",
        "approval_tool_progress_observed",
        "approval_payload_redacted",
        "approval_secret_canary_checked",
        "approval_completed",
        "cancel_drain_capacity_smoke_ok",
        "cancel_drain_requested_observed",
        "cancel_drain_cancelled_observed",
        "cancel_drain_followup_queued",
        "cancel_drain_active_visible",
        "cancel_drain_followup_started_after_cancel",
        "failed_job_reported",
        "failed_job_spoken",
        "durable_failed_record_present",
        "session_survived_failed_job",
        "verbose_result_spoken_bounded",
        "verbose_result_committed_bounded",
        "verbose_result_commit_marked_truncated",
        "verbose_full_result_durable",
        "terminal_result_policy_smoke_ok",
        "terminal_result_auto_summarize_default",
        "terminal_result_suppressed",
        "terminal_result_status_available",
        "audit_scalar_smoke_ok",
        "audit_scalar_payload_redacted",
        "audit_scalar_secret_canary_checked",
        "audit_scalar_result_text_omitted",
        "audit_scalar_completed_event_seen",
        "audit_scalar_waiting_event_seen",
        "sidecar_control_smoke_ok",
        "sidecar_control_update_observed",
        "sidecar_control_update_reached_oracle",
        "sidecar_control_cancel_requested",
        "sidecar_control_cancelled",
        "sidecar_control_feedback_update_sent",
        "sidecar_control_feedback_cancel_sent",
    )
    for field in required_true_fields:
        if entry.get(field) is not True:
            issues.append(
                RealtimeVoiceSmokeReportIssue(
                    "async_oracle_smoke",
                    f"missing required async proof {field}",
                    identifier,
                )
            )
    required_false_fields = (
        "noncooperative_cancel_overlap_observed",
        "cancelled_result_spoken",
        "cancelled_result_committed",
        "cancelled_result_progress_leaked",
        "cancelled_result_durable_completed",
        "cancelled_result_durable_text",
        "approval_secret_leaked",
        "cancel_drain_misleading_running_capacity",
        "playback_stop_cancelled_jobs",
        "terminal_result_unsolicited_spoken",
        "sidecar_control_completed_after_cancel",
    )
    for field in required_false_fields:
        if entry.get(field) is not False:
            issues.append(
                RealtimeVoiceSmokeReportIssue(
                    "async_oracle_smoke",
                    f"missing required negative async proof {field}",
                    identifier,
                )
            )
    if int(entry.get("max_worker_overlap") or 0) < 4:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "async_oracle_smoke",
                "max_worker_overlap did not prove four concurrent oracle jobs",
                identifier,
            )
        )
    if int(entry.get("local_turn_active_job_count") or 0) < 1:
        issues.append(
            RealtimeVoiceSmokeReportIssue(
                "async_oracle_smoke",
                "local turn did not prove an active oracle job overlap",
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


def _has_output_audio_event(events: set[str]) -> bool:
    return bool(OUTPUT_AUDIO_EVENT_NAMES.intersection(events))


def _has_final_user_turn_event(events: set[str]) -> bool:
    return bool({"transcript.final", "interface.intent.final"}.intersection(events))


def _transcript_matches_expected(actual: str, expected: str) -> bool:
    actual_norm = _normalize_transcript_text(actual)
    expected_norm = _normalize_transcript_text(expected)
    return bool(actual_norm and expected_norm and actual_norm == expected_norm)


def _final_transcript_satisfies_partial_target(
    final_ms: int | None,
    target_ms: int | None,
) -> bool:
    return final_ms is not None and target_ms is not None and final_ms <= target_ms


_TRANSCRIPT_PUNCTUATION_RE = re.compile(r"[\s\.,!?;:'\"`~()\[\]{}<>/\\|_\-+=、。，．！？；：『』「」（）【】]")


def _normalize_transcript_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    normalized = normalized.replace("ハーメス", "hermes").replace("ハルメス", "hermes")
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


def _entry_metric_values(entries: Sequence[Mapping[str, Any]], key: str) -> Iterable[Any]:
    for entry in entries:
        metrics = entry.get("metrics")
        if isinstance(metrics, Mapping):
            yield metrics.get(key)
        yield entry.get(key)


def _latency_summary(values: Iterable[Any]) -> dict[str, Any]:
    parsed = sorted(value for value in (_nonnegative_int(item) for item in values) if value is not None)
    if not parsed:
        return {"count": 0, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "count": len(parsed),
        "p50": _percentile_nearest_rank(parsed, 50),
        "p90": _percentile_nearest_rank(parsed, 90),
        "p95": _percentile_nearest_rank(parsed, 95),
        "max": parsed[-1],
    }


def _percentile_nearest_rank(values: Sequence[int], percentile: int) -> int:
    if not values:
        raise ValueError("percentile requires at least one value")
    index = max(0, min(len(values) - 1, math.ceil((percentile / 100) * len(values)) - 1))
    return values[index]
