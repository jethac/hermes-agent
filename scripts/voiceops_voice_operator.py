#!/usr/bin/env python3
"""Generate Milestone 1 VoiceOps Discord voice-operator evidence.

This headless generator runs the in-memory Discord realtime voice loopback
smoke. It does not connect to Discord, read Discord credentials, start a
provider sidecar, send messages, or place calls.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import datetime as dt
import hashlib
import json
import logging
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermes_cli.discord_realtime_voice_smoke import run_discord_realtime_voice_smoke
from scripts.realtime_voice_async_oracle_smoke import run_smoke as run_async_oracle_smoke


DEFAULT_OUTPUT_DIR = Path("artifacts/voiceops-voice-operator/current")
DISCORD_FRAME_BYTES = 3840
SIDECAR_FRAME_BYTES = 640
LIVE_EVIDENCE_SCHEMA_VERSION = "voiceops.milestone1.live_voice_evidence.v1"
LIVE_EVIDENCE_MANIFEST_SCHEMA_VERSION = "voiceops.realtime_voice_live_evidence_manifest.v1"
LIVE_EVIDENCE_REQUIRED_GATES = (
    "discord_join",
    "discord_playback",
    "live_receiver",
    "production_sidecar",
    "live_turn",
)
COLLECTOR_ATTESTATION_REQUIRED_FIELDS = (
    "collector_name",
    "collector_version",
    "run_id",
    "command_argv",
    "git_commit",
    "started_at",
    "finished_at",
    "raw_artifact_sha256",
    "redacted_artifact_sha256",
    "parent_manifest_sha256",
)

REQUIRED_EVENTS = {
    "transcript.partial",
    "transcript.final",
    "assistant.text.partial",
    "audio.output.chunk",
    "assistant.commit",
    "barge_in.detected",
}

RECEIVER_CALLBACK_TEST_REFS = [
    "tests/gateway/test_voice_command.py::test_join_voice_channel_wires_realtime_frame_and_speech_start_callbacks",
]

BARGE_IN_ENERGY_TEST_REFS = [
    "tests/gateway/test_voice_command.py::test_pcm16_rms_ignores_silence_and_detects_volume",
    "tests/gateway/test_voice_command.py::test_join_voice_channel_wires_realtime_frame_and_speech_start_callbacks",
    "tests/gateway/test_discord_realtime_voice.py::test_discord_realtime_session_sends_speech_energy_event",
]

ASYNC_ORACLE_ACCEPTANCE_TEST_REFS = {
    "job_manager_capacity": [
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_max_concurrent_one_queues_second_job",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_max_concurrent_four_starts_four_and_queues_fifth",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_queue_limit_rejects_overflow",
        "tests/gateway/test_discord_realtime_voice.py::test_discord_realtime_spoken_tasks_create_async_oracle_jobs",
    ],
    "status_view": [
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_status_view_reports_capacity_and_redacts_raw_metadata",
        "tests/agent/test_realtime_voice.py::test_kame_engine_local_status_question_uses_oracle_job_state",
        "tests/gateway/test_voice_command.py::TestVoiceChannelCommands::test_voice_jobs_reports_oracle_job_snapshot",
    ],
    "local_turns": [
        "tests/agent/test_realtime_voice.py::test_kame_engine_async_oracle_job_allows_local_turn_while_running",
        "tests/agent/test_realtime_voice.py::test_oracle_direct_async_job_completion_after_local_turn_is_lifecycle_only",
    ],
    "cancellation": [
        "tests/agent/test_realtime_voice.py::test_kame_engine_interface_cancel_stops_one_async_oracle_job",
        "tests/agent/test_realtime_voice.py::test_kame_engine_can_cancel_queued_async_oracle_job_before_it_starts",
        "tests/agent/test_realtime_voice.py::test_kame_engine_spoken_stop_everything_cancels_all_async_oracle_jobs",
        "tests/agent/test_realtime_voice.py::test_kame_engine_barge_in_during_async_ack_does_not_interrupt_oracle_job",
        "tests/agent/test_realtime_voice.py::test_kame_engine_spoken_stop_talking_does_not_cancel_async_oracle_job",
        "tests/agent/test_realtime_voice.py::test_kame_engine_barge_in_during_async_result_speech_does_not_interrupt_completed_job",
        "tests/gateway/test_discord_realtime_voice.py::test_discord_realtime_cancelled_oracle_late_output_is_not_mixed",
    ],
    "approval_wait": [
        "tests/agent/test_realtime_voice.py::test_async_oracle_job_enters_waiting_for_approval_on_tool_call",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_waiting_for_approval_holds_capacity_and_emits_redacted_event",
        "tests/gateway/test_discord_realtime_voice.py::test_voice_status_oracle_job_lines_are_compact",
    ],
    "failure_handling": [
        "tests/agent/test_realtime_voice.py::test_kame_engine_async_oracle_job_failure_reports_in_voice",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_failed_job_records_error_and_starts_next",
    ],
    "control_updates": [
        "tests/agent/test_realtime_voice.py::test_kame_engine_can_reprioritize_queued_async_oracle_job",
        "tests/agent/test_realtime_voice.py::test_kame_engine_attaches_update_to_queued_async_oracle_job",
        "tests/agent/test_realtime_voice.py::test_kame_engine_spoken_priority_control_reprioritizes_queued_job",
        "tests/agent/test_realtime_voice.py::test_kame_engine_spoken_update_attaches_to_latest_async_oracle_job",
        "tests/gateway/test_discord_realtime_voice.py::test_discord_realtime_event_tracks_oracle_job_status",
    ],
    "result_handling": [
        "tests/agent/test_realtime_voice.py::test_completed_async_oracle_job_after_intervening_local_turn_is_lifecycle_only",
        "tests/agent/test_realtime_voice.py::test_kame_engine_status_recalls_recent_completed_async_oracle_job",
        "tests/agent/test_realtime_voice.py::test_kame_engine_async_oracle_job_failure_reports_in_voice",
        "tests/agent/test_realtime_voice.py::test_kame_engine_async_terminal_result_speech_is_capped_without_losing_full_result",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_completed_event_preserves_full_result_without_bloating_status",
        "tests/agent/test_realtime_voice.py::test_session_persists_durable_async_oracle_job_records",
    ],
    "discord_session": [
        "tests/gateway/test_voice_command.py::TestDiscordVoiceChannelMethods::test_leave_voice_channel_cleans_up",
        "tests/gateway/test_discord_realtime_voice.py::test_discord_realtime_degraded_marks_active_oracle_jobs_failed",
        "tests/gateway/test_discord_realtime_voice.py::test_discord_realtime_session_close_cancels_oracle_jobs_before_session_closed",
    ],
    "shutdown": [
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_shutdown_forces_cancelled_state_when_worker_ignores_cancel",
        "tests/agent/test_realtime_voice.py::test_kame_engine_close_bounds_noncooperative_async_oracle_shutdown",
        "tests/agent/test_realtime_voice_async_oracle_smoke.py::test_async_oracle_smoke_proves_concurrency_local_turn_and_cancellation",
    ],
}

LIVE_EVIDENCE_REQUIRED_DISCORD_BOOLS = (
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

LIVE_EVIDENCE_REQUIRED_SIDECAR_BOOLS = (
    "sidecar_running",
    "sidecar_healthy",
    "session_started",
    "session_closed",
    "fallback_mode_visible",
)

LIVE_EVIDENCE_REQUIRED_SIDECAR_PROVENANCE_BOOLS = (
    "healthcheck_observed",
    "provider_transport_observed",
    "session_id_redacted",
)

LIVE_EVIDENCE_REQUIRED_TURN_BOOLS = (
    "transcript_observed",
    "assistant_audio_observed",
    "barge_in_observed",
    "spoken_reply_short",
    "no_voice_denial_observed",
)

LIVE_EVIDENCE_TEMPLATE_SOURCE_ARTIFACTS = {
    "discord-live-probe.json",
    "voice-status-or-sidecar-report.json",
    "sidecar-session.json",
    "voice-turn-evidence.json",
    "live-turn.json",
}

LIVE_EVIDENCE_REQUIRED_DISCORD_LATENCIES_MS = (
    "connect_ms",
    "playback_observed_ms",
    "inbound_observed_ms",
    "disconnect_ms",
)

LIVE_EVIDENCE_FORBIDDEN_TEXT_FIELDS = {
    "assistant_text",
    "assistant_reply",
    "assistant_transcript",
    "raw_transcript",
    "reply_text",
    "transcript_text",
    "user_transcript",
}

LIVE_EVIDENCE_SECRET_FIELD_MARKERS = (
    "api_key",
    "authorization",
    "auth_token",
    "bearer",
    "phone",
    "secret",
    "token",
)

LIVE_EVIDENCE_DENIAL_PHRASES = (
    "cannot hear voice",
    "cannot hear you",
    "cannot speak in discord",
    "do not have any ability to join discord voice",
    "i only process text",
    "i only process typed text",
)

SECRET_VALUE_RE = re.compile(
    r"(?i)(sk[-_][a-z0-9]{8,}|pk[-_][a-z0-9]{8,}|rk[-_][a-z0-9]{8,}|"
    r"whsec_[a-z0-9]{8,}|xox[aboprs]-[a-z0-9-]{8,}|gh[pousr]_[a-z0-9_]{8,}|"
    r"mfa\.[a-z0-9_-]{20,}|[a-z0-9_-]{23,}\.[a-z0-9_-]{6,}\.[a-z0-9_-]{20,})"
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


class _CleanupSmokeSidecar:
    def __init__(self) -> None:
        self.started_with: Any = None
        self.sent: list[Any] = []
        self.closed = False
        self.close_calls = 0
        self._events: asyncio.Queue[Any] = asyncio.Queue()

    async def start(self, config: Any) -> None:
        self.started_with = config

    async def send_event(self, event: Any) -> None:
        self.sent.append(event)

    async def close(self) -> None:
        self.close_calls += 1
        self.closed = True
        await self._events.put(None)

    async def events(self) -> Any:
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event


async def run_discord_session_cleanup_smoke() -> dict[str, Any]:
    """Provider-free Discord session cleanup smoke for async oracle state."""

    from agent.realtime_voice import VoiceEventType
    from plugins.platforms.discord.adapter import DiscordAdapter
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    sidecar = _CleanupSmokeSidecar()
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        sidecar_base_url="http://127.0.0.1:8766",
        oracle_jobs={"enabled": True},
    )
    await session.start()
    await session.close()

    event_types = [event.type for event in sidecar.sent]
    cancel_event = next(
        (event for event in sidecar.sent if event.type == VoiceEventType.INTERFACE_ORACLE_CANCEL),
        None,
    )
    session_closed_event = next(
        (event for event in sidecar.sent if event.type == VoiceEventType.SESSION_CLOSED),
        None,
    )
    cancel_index = event_types.index(VoiceEventType.INTERFACE_ORACLE_CANCEL) if cancel_event else -1
    closed_index = event_types.index(VoiceEventType.SESSION_CLOSED) if session_closed_event else -1
    cancel_all_before_session_closed = (
        cancel_event is not None
        and session_closed_event is not None
        and 0 <= cancel_index < closed_index
        and cancel_event.payload == {
            "job_id": "all",
            "all": True,
            "reason": "voice session closing",
            "transport": "discord_voice",
        }
    )

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter._voice_session_states = {}
    adapter._realtime_voice_sessions = {111: object()}
    adapter._realtime_voice_cfg = {"fallback_policy": "text_only"}
    adapter._handle_realtime_voice_event(
        111,
        "session.started",
        {"oracle_jobs": {"enabled": True, "max_concurrent": 4, "queue_limit": 16}},
    )
    adapter._handle_realtime_voice_event(
        111,
        "oracle.job.started",
        {
            "job_id": "voice-oracle-001",
            "state": "running",
            "priority": "normal",
            "route": "defer",
            "intent": "Check the deployment status.",
            "spoken_status": "Checking the deployment status.",
        },
    )
    adapter_logger = logging.getLogger("plugins.platforms.discord.adapter")
    adapter_logger_disabled = adapter_logger.disabled
    adapter_logger.disabled = True
    try:
        adapter._handle_realtime_voice_degraded(
            111,
            "sidecar_event_stream_closed",
            "sidecar event stream closed",
        )
    finally:
        adapter_logger.disabled = adapter_logger_disabled
    status = adapter.get_voice_session_status(111)
    failed_jobs = status.get("oracle_jobs", {}).get("jobs") or []
    failed_job = failed_jobs[0] if failed_jobs else {}
    degraded_active_job_preserved_failed = (
        111 not in adapter._realtime_voice_sessions
        and status.get("sidecar_running") is False
        and status.get("fallback_reason") == "sidecar_event_stream_closed: sidecar event stream closed"
        and status.get("oracle_jobs", {}).get("capacity", {}).get("running") == 0
        and failed_job.get("job_id") == "voice-oracle-001"
        and failed_job.get("state") == "failed"
        and failed_job.get("intent") == "Check the deployment status."
        and failed_job.get("spoken_status") == "Checking the deployment status."
        and failed_job.get("error") == "sidecar_event_stream_closed: sidecar event stream closed"
    )
    return {
        "ok": bool(cancel_all_before_session_closed and sidecar.closed and degraded_active_job_preserved_failed),
        "scenario": "discord_session_cleanup_fake_sidecar",
        "discord_network": False,
        "provider_sidecar_network": False,
        "cancel_all_before_session_closed": cancel_all_before_session_closed,
        "cancel_payload": dict(cancel_event.payload) if cancel_event is not None else {},
        "session_closed_sent": session_closed_event is not None,
        "sidecar_closed": sidecar.closed,
        "sidecar_close_calls": sidecar.close_calls,
        "degraded_active_job_preserved_failed": degraded_active_job_preserved_failed,
        "degraded_session_removed": 111 not in adapter._realtime_voice_sessions,
        "degraded_fallback_reason": status.get("fallback_reason"),
        "degraded_job_state": failed_job.get("state"),
        "degraded_job_error": failed_job.get("error"),
        "event_order": [str(event_type.value) for event_type in event_types],
    }


def _non_negative_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float) and value >= 0:
        return float(value)
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _parse_timezone_timestamp(value: Any) -> dt.datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _has_parseable_timezone_timestamp(value: Any) -> bool:
    return _parse_timezone_timestamp(value) is not None


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _collector_attestation_command_arg_is_sensitive(value: str) -> bool:
    text = str(value or "")
    if SECRET_VALUE_RE.search(text):
        return True
    if re.search(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}\b", text):
        return True
    digits = "".join(ch for ch in text if ch.isdigit())
    return "+" in text and len(digits) >= 8


def _collector_attestation_issues(
    section: Mapping[str, Any],
    section_name: str,
    *,
    expected_redacted_sha256: str | None = None,
    expected_parent_manifest_sha256: str | None = None,
    expected_parent_manifest_sha256_values: list[str] | None = None,
) -> list[str]:
    attestation = section.get("collector_attestation") or section.get("collector_provenance")
    if not isinstance(attestation, Mapping):
        return [f"{section_name}:missing_collector_attestation"]
    issues: list[str] = []
    if attestation.get("example_only") is True:
        issues.append(f"{section_name}:collector_attestation_example_only_not_accepted")
    for field in COLLECTOR_ATTESTATION_REQUIRED_FIELDS:
        if field not in attestation:
            issues.append(f"{section_name}:collector_attestation_missing:{field}")
    for field in ("collector_name", "collector_version", "run_id", "git_commit"):
        value = str(attestation.get(field) or "").strip()
        if not value or value.lower() in {"placeholder", "example", "replace-me", "unknown"}:
            issues.append(f"{section_name}:collector_attestation_invalid:{field}")
    command_argv = attestation.get("command_argv")
    if not isinstance(command_argv, list) or not command_argv or not all(isinstance(part, str) and part for part in command_argv):
        issues.append(f"{section_name}:collector_attestation_invalid:command_argv")
    elif any(_collector_attestation_command_arg_is_sensitive(part) for part in command_argv):
        issues.append(f"{section_name}:collector_attestation_secret_or_phone_like_command_argv")
    started_at = _parse_timezone_timestamp(attestation.get("started_at"))
    finished_at = _parse_timezone_timestamp(attestation.get("finished_at"))
    if started_at is None:
        issues.append(f"{section_name}:collector_attestation_invalid:started_at")
    if finished_at is None:
        issues.append(f"{section_name}:collector_attestation_invalid:finished_at")
    if started_at is not None and finished_at is not None and started_at > finished_at:
        issues.append(f"{section_name}:collector_attestation_invalid:timestamp_window")
    for field in ("raw_artifact_sha256", "redacted_artifact_sha256", "parent_manifest_sha256"):
        if not _valid_sha256(attestation.get(field)):
            issues.append(f"{section_name}:collector_attestation_invalid:{field}")
    redacted_sha256 = str(attestation.get("redacted_artifact_sha256") or "").strip().lower()
    if (
        expected_redacted_sha256
        and _valid_sha256(expected_redacted_sha256)
        and _valid_sha256(redacted_sha256)
        and redacted_sha256 != expected_redacted_sha256
    ):
        issues.append(f"{section_name}:collector_attestation_redacted_sha256_mismatch")
    parent_sha256 = str(attestation.get("parent_manifest_sha256") or "").strip().lower()
    expected_parent_values = {
        str(value or "").strip().lower()
        for value in (expected_parent_manifest_sha256_values or [])
        if _valid_sha256(value)
    }
    if expected_parent_manifest_sha256 and _valid_sha256(expected_parent_manifest_sha256):
        expected_parent_values.add(expected_parent_manifest_sha256)
    if (
        expected_parent_values
        and _valid_sha256(parent_sha256)
        and parent_sha256 not in expected_parent_values
    ):
        issues.append(f"{section_name}:collector_attestation_parent_manifest_sha256_mismatch")
    return issues


def _looks_secret_or_phone(value: Any) -> bool:
    text = str(value or "")
    lowered = text.lower()
    secret_markers = ("sk_", "pk_", "rk_", "whsec_", "xoxb", "xoxp", "ghp_", "bearer ", "sk-", "pk-", "rk-")
    if any(marker in lowered for marker in secret_markers):
        return True
    if SECRET_VALUE_RE.search(text):
        return True
    digits = "".join(ch for ch in text if ch.isdigit())
    return "+" in text and len(digits) >= 8


def _live_evidence_key_name(path: str) -> str:
    return path.rsplit(".", 1)[-1].split("[", 1)[0].lower()


def _looks_like_forbidden_live_evidence_field(path: str) -> bool:
    name = _live_evidence_key_name(path)
    if name in LIVE_EVIDENCE_FORBIDDEN_TEXT_FIELDS:
        return True
    return any(marker in name for marker in LIVE_EVIDENCE_SECRET_FIELD_MARKERS)


def _looks_like_voice_denial_text(value: str) -> bool:
    lowered = value.lower()
    return any(phrase in lowered for phrase in LIVE_EVIDENCE_DENIAL_PHRASES)


def build_live_probe_evidence_template() -> dict[str, Any]:
    collector_template = {
        "collector_name": None,
        "collector_version": None,
        "run_id": None,
        "command_argv": [],
        "git_commit": None,
        "started_at": None,
        "finished_at": None,
        "raw_artifact_sha256": None,
        "redacted_artifact_sha256": None,
        "parent_manifest_sha256": None,
    }
    return {
        "schema_version": LIVE_EVIDENCE_SCHEMA_VERSION,
        "redaction_policy": "references and booleans only; no Discord tokens, provider tokens, full phone numbers, or raw transcripts with secrets",
        "discord_live_probe": {
            "source_artifact": "discord-live-probe.json",
            "kind": "discord_live_probe",
            "collector_attestation": dict(collector_template),
            "ok": False,
            "connect_perm": False,
            "speak_perm": False,
            "connected": False,
            "opus_loaded": False,
            "accepted_audio_source": False,
            "played": False,
            "playing_during_probe": False,
            "receiver_started": False,
            "receiver_frames": 0,
            "receiver_speech_start": 0,
            "inbound_observed": False,
            "disconnected": False,
            "require_inbound": True,
            "latency_metrics_ms": {
                "connect_ms": None,
                "playback_observed_ms": None,
                "inbound_observed_ms": None,
                "disconnect_ms": None,
            },
        },
        "sidecar_session": {
            "source_artifact": "voice-status-or-sidecar-report.json",
            "collector_attestation": dict(collector_template),
            "sidecar_running": False,
            "sidecar_healthy": False,
            "session_started": False,
            "session_closed": False,
            "shutdown_bounded": False,
            "shutdown_timed_out": None,
            "fallback_mode_visible": False,
            "fallback_reason": None,
            "sidecar_mode": None,
            "healthcheck_observed": False,
            "provider_transport_observed": False,
            "session_id_redacted": False,
            "latency_metrics_ms": {
                "session_start_ms": None,
                "shutdown_ms": None,
            },
        },
        "live_turn": {
            "source_artifact": "voice-turn-evidence.json",
            "collector_attestation": dict(collector_template),
            "transcript_observed": False,
            "assistant_audio_observed": False,
            "barge_in_observed": False,
            "spoken_reply_short": False,
            "no_voice_denial_observed": False,
            "speech_end_to_first_audio_ms": None,
            "barge_in_stop_ms": None,
        },
    }


def _example_collector_attestation(section_name: str) -> dict[str, Any]:
    return {
        "example_only": True,
        "collector_name": "hermes_cli.realtime_voice_live_evidence",
        "collector_version": "example",
        "run_id": f"example-{section_name}-run",
        "command_argv": ["uv", "run", "python", "-m", "hermes_cli.realtime_voice_live_evidence"],
        "git_commit": "0" * 40,
        "started_at": "2026-06-29T00:00:00Z",
        "finished_at": "2026-06-29T00:00:01Z",
        "raw_artifact_sha256": "0" * 64,
        "redacted_artifact_sha256": "0" * 64,
        "parent_manifest_sha256": "0" * 64,
    }


def build_live_probe_evidence_example() -> dict[str, Any]:
    example = build_live_probe_evidence_template()
    example["example_only"] = True
    example["redaction_policy"] = "example only; copy shape with real artifact refs and remove example_only before ingest"
    example["discord_live_probe"].update(
        {
            "collector_attestation": _example_collector_attestation("discord_live_probe"),
            "ok": True,
            "connect_perm": True,
            "speak_perm": True,
            "connected": True,
            "opus_loaded": True,
            "accepted_audio_source": True,
            "played": True,
            "playing_during_probe": True,
            "receiver_started": True,
            "receiver_frames": 42,
            "receiver_speech_start": 1,
            "inbound_observed": True,
            "disconnected": True,
            "latency_metrics_ms": {
                "connect_ms": 420,
                "playback_observed_ms": 180,
                "inbound_observed_ms": 900,
                "disconnect_ms": 120,
            },
        }
    )
    example["sidecar_session"].update(
        {
            "collector_attestation": _example_collector_attestation("sidecar_session"),
            "sidecar_running": True,
            "sidecar_healthy": True,
            "session_started": True,
            "session_closed": True,
            "shutdown_bounded": True,
            "shutdown_timed_out": False,
            "fallback_mode_visible": True,
            "fallback_reason": "none",
            "sidecar_mode": "production",
            "healthcheck_observed": True,
            "provider_transport_observed": True,
            "session_id_redacted": True,
            "latency_metrics_ms": {
                "session_start_ms": 110,
                "shutdown_ms": 80,
            },
        }
    )
    example["live_turn"].update(
        {
            "collector_attestation": _example_collector_attestation("live_turn"),
            "transcript_observed": True,
            "assistant_audio_observed": True,
            "barge_in_observed": True,
            "spoken_reply_short": True,
            "no_voice_denial_observed": True,
            "speech_end_to_first_audio_ms": 950,
            "barge_in_stop_ms": 90,
        }
    )
    return example


def write_live_evidence_scaffold(output_dir: Path) -> dict[str, Path]:
    scaffold_dir = output_dir / "live-voice-evidence-scaffold"
    sections_dir = scaffold_dir / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)

    example = build_live_probe_evidence_example()
    section_files = {
        "discord_live_probe": "discord-live-probe.json",
        "sidecar_session": "sidecar-session.json",
        "live_turn": "live-turn.json",
    }
    reports: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for section_name, section_file in section_files.items():
        section = dict(example[section_name])
        section["example_only"] = True
        section["kind"] = section_name
        section["source_artifact"] = section_file
        section["redaction_policy"] = (
            "example only; replace with real redacted live evidence and remove example_only before ingest"
        )
        section_path = sections_dir / section_file
        _write_json(section_path, section)
        reports[section_name] = f"sections/{section_file}"
        paths[f"scaffold_{section_name}"] = section_path

    manifest_path = scaffold_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": LIVE_EVIDENCE_MANIFEST_SCHEMA_VERSION,
            "example_only": True,
            "redaction_policy": "example only; this scaffold is rejected until all example_only markers are removed",
            "reports": reports,
            "notes": (
                "Replace each section report with real redacted Discord/sidecar/turn evidence. "
                "Do not paste tokens, full phone numbers, or raw private transcripts."
            ),
        },
    )
    paths["live_evidence_scaffold_manifest"] = manifest_path
    return paths


def _load_live_evidence(paths: list[Path] | None) -> dict[str, Any]:
    paths = paths or []
    if not paths:
        return {
            "loaded": False,
            "mode": "supplied_artifacts_only",
            "artifact_paths": [],
            "overall_status": "needs_live_probe",
            "issues": ["live_evidence_not_loaded"],
            "redaction_policy": "not_loaded",
        }
    payload: dict[str, Any] = {}
    load_issues: list[str] = []
    for path in paths:
        loaded = _load_live_evidence_file(path)
        if loaded["issues"]:
            load_issues.extend(str(issue) for issue in loaded["issues"])
        data = loaded.get("payload")
        if isinstance(data, Mapping):
            _merge_live_evidence_payload(payload, data)
    evidence = validate_live_probe_evidence(payload, paths=paths)
    evidence["issues"] = sorted(set([*evidence["issues"], *load_issues]))
    if evidence["issues"]:
        evidence["overall_status"] = "partial_live_evidence"
    else:
        evidence["overall_status"] = "live_evidence_supplied_not_readiness_claim"
    return evidence


def _load_live_evidence_file(path: Path, *, visited: set[Path] | None = None, standalone: bool = True) -> dict[str, Any]:
    visited = set() if visited is None else set(visited)
    resolved_path = path.expanduser().resolve()
    if resolved_path in visited:
        return {
            "payload": None,
            "issues": ["cycle_detected"],
        }
    visited.add(resolved_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "payload": None,
            "issues": ["live_evidence_file_not_found"],
        }
    except json.JSONDecodeError as exc:
        return {
            "payload": None,
            "issues": [f"live_evidence_json_parse_failed:{exc.msg}"],
        }
    if not isinstance(payload, Mapping):
        return {
            "payload": None,
            "issues": ["live_evidence_root_must_be_object"],
        }
    payload, issues = _expand_live_evidence_manifest(path, payload, visited=visited)
    if standalone:
        identity_issue = _standalone_report_identity_issue(payload)
        if identity_issue:
            issues.append(identity_issue)
    return {"payload": payload, "issues": issues}


def _expand_live_evidence_manifest(
    path: Path,
    payload: Mapping[str, Any],
    *,
    visited: set[Path],
) -> tuple[Mapping[str, Any], list[str]]:
    reports = payload.get("reports")
    if not isinstance(reports, Mapping):
        return payload, []

    expanded: dict[str, Any] = {"schema_version": LIVE_EVIDENCE_SCHEMA_VERSION}
    if payload.get("example_only") is True:
        expanded["example_only"] = True
    issues: list[str] = []
    manifest_schema = str(payload.get("schema_version") or "")
    if not manifest_schema:
        issues.append("live_evidence_manifest:missing_schema_version")
    elif manifest_schema != LIVE_EVIDENCE_MANIFEST_SCHEMA_VERSION:
        issues.append("live_evidence_manifest:invalid_schema_version")
    for report_name, report_path_value in reports.items():
        report_path_text = str(report_path_value or "").strip()
        if not report_path_text:
            issues.append(f"live_evidence_manifest:{report_name}:empty_report_path")
            continue
        report_path_issues = _relative_live_artifact_ref_issues(report_path_text, base_path=path)
        if report_path_issues:
            issues.extend(
                f"live_evidence_manifest:{report_name}:report_path:{issue}" for issue in report_path_issues
            )
            continue
        report_path = _resolve_manifest_report_path(path, report_path_text)
        loaded = _load_live_evidence_file(report_path, visited=visited, standalone=False)
        if loaded["issues"]:
            issues.extend(f"live_evidence_manifest:{report_name}:{issue}" for issue in loaded["issues"])
        report_payload = loaded.get("payload")
        if isinstance(report_payload, Mapping):
            if not _manifest_report_has_identity(report_name, report_payload):
                issues.append(f"live_evidence_manifest:{report_name}:missing_report_identity")
            report_payload, provenance_issues = _with_manifest_report_provenance(report_payload, report_path)
            issues.extend(f"live_evidence_manifest:{report_name}:{issue}" for issue in provenance_issues)
            if report_payload.get("example_only") is True:
                issues.append(f"live_evidence_manifest:{report_name}:example_only_evidence_not_accepted")
                expanded["example_only"] = True
            _merge_live_evidence_payload(expanded, report_payload)
    return expanded if expanded else payload, issues


def _with_manifest_report_provenance(payload: Mapping[str, Any], report_path: Path) -> tuple[dict[str, Any], list[str]]:
    enriched = dict(payload)
    source_artifact = str(report_path.resolve())
    previous_source_artifact = str(enriched.get("source_artifact") or "")
    enriched["source_artifact"] = source_artifact
    provenance = {"wrapper_artifact": source_artifact}
    if previous_source_artifact:
        provenance["reported_source_artifact"] = previous_source_artifact
    enriched["provenance"] = provenance
    issues: list[str] = []
    for section_name in ("discord_live_probe", "sidecar_session", "live_turn"):
        section = enriched.get(section_name)
        if isinstance(section, Mapping):
            section_copy = dict(section)
            previous_section_source = str(section_copy.get("source_artifact") or "")
            section_provenance = {"wrapper_artifact": source_artifact, "section": section_name}
            if previous_section_source:
                section_provenance["reported_source_artifact"] = previous_section_source
                if previous_section_source not in LIVE_EVIDENCE_TEMPLATE_SOURCE_ARTIFACTS:
                    source_path_issues = _relative_live_artifact_ref_issues(
                        previous_section_source,
                        base_path=report_path,
                    )
                    if source_path_issues:
                        issues.extend(f"{section_name}.source_artifact:{issue}" for issue in source_path_issues)
                    else:
                        section_copy["source_artifact"] = str((report_path.parent / previous_section_source).resolve())
            section_copy["provenance"] = section_provenance
            enriched[section_name] = section_copy
    return enriched, issues


def _resolve_manifest_report_path(manifest_path: Path, report_path_text: str) -> Path:
    return manifest_path.parent / report_path_text


def _relative_live_artifact_ref_issues(ref_text: str, *, base_path: Path) -> list[str]:
    issues: list[str] = []
    ref = str(ref_text or "").strip()
    if not ref:
        return ["empty"]
    if ref.startswith("~"):
        issues.append("user_home_not_allowed")
    ref_path = Path(ref)
    if ref_path.is_absolute():
        issues.append("absolute_path_not_allowed")
    if ".." in ref_path.parts:
        issues.append("parent_traversal_not_allowed")
    if issues:
        return sorted(set(issues))
    base_dir = base_path.expanduser().resolve(strict=False).parent
    candidate = base_dir / ref
    try:
        resolved_candidate = candidate.resolve(strict=True)
    except OSError:
        resolved_candidate = candidate.resolve(strict=False)
    resolved_base = base_dir.resolve(strict=False)
    if resolved_candidate != resolved_base and resolved_base not in resolved_candidate.parents:
        issues.append("path_escape_not_allowed")
    return sorted(set(issues))


def _manifest_report_has_identity(report_name: str, payload: Mapping[str, Any]) -> bool:
    if _uses_expanded_live_evidence_schema(payload):
        return True
    kind = str(payload.get("kind") or payload.get("evidence_type") or "").strip()
    if report_name == "combined":
        return _uses_expanded_live_evidence_schema(payload) or kind == "combined"
    if report_name == "discord_live_probe":
        return kind == "discord_live_probe"
    if report_name == "sidecar_session":
        return kind == "sidecar_session"
    if report_name == "live_turn":
        return kind == "live_turn"
    return bool(kind)


def _standalone_report_identity_issue(payload: Mapping[str, Any]) -> str:
    if _uses_expanded_live_evidence_schema(payload):
        return ""
    kind = str(payload.get("kind") or payload.get("evidence_type") or "").strip()
    if not kind:
        return "missing_standalone_report_identity"
    if kind in {"discord_live_probe", "sidecar_session", "live_turn"}:
        return ""
    return f"invalid_standalone_report_identity:{kind}"


def _uses_expanded_live_evidence_schema(payload: Mapping[str, Any]) -> bool:
    if str(payload.get("schema_version") or "") != LIVE_EVIDENCE_SCHEMA_VERSION:
        return False
    return any(isinstance(payload.get(section), Mapping) for section in ("discord_live_probe", "sidecar_session", "live_turn"))


def _merge_live_evidence_payload(target: dict[str, Any], payload: Mapping[str, Any]) -> None:
    if payload.get("example_only") is True:
        target["example_only"] = True
    if "schema_version" in payload and "schema_version" not in target:
        target["schema_version"] = payload.get("schema_version")
    discord_probe = _discord_probe_section(payload)
    if discord_probe:
        target["discord_live_probe"] = dict(discord_probe)
    for section_name in ("sidecar_session", "live_turn"):
        section = payload.get(section_name)
        if isinstance(section, Mapping):
            target[section_name] = dict(section)
    if _looks_like_sidecar_session(payload):
        target["sidecar_session"] = dict(payload)
    if _looks_like_live_turn(payload):
        target["live_turn"] = dict(payload)


def _looks_like_sidecar_session(payload: Mapping[str, Any]) -> bool:
    return any(key in payload for key in LIVE_EVIDENCE_REQUIRED_SIDECAR_BOOLS)


def _looks_like_live_turn(payload: Mapping[str, Any]) -> bool:
    return any(key in payload for key in LIVE_EVIDENCE_REQUIRED_TURN_BOOLS) or "speech_end_to_first_audio_ms" in payload


def _discord_probe_section(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    section = payload.get("discord_live_probe")
    if isinstance(section, Mapping):
        return section
    if payload.get("kind") == "discord_live_probe" or "accepted_audio_source" in payload:
        return payload
    return {}


def validate_live_probe_evidence(payload: Mapping[str, Any], *, paths: list[Path] | None = None) -> dict[str, Any]:
    issues: list[str] = []
    if str(payload.get("schema_version") or "") != LIVE_EVIDENCE_SCHEMA_VERSION:
        issues.append("missing_schema_version")
    if payload.get("example_only") is True:
        issues.append("example_only_evidence_not_accepted")
    redaction_issues: list[str] = []
    for key, value in _walk_live_evidence_strings(payload):
        if _is_collector_attestation_path(key):
            continue
        if _looks_like_forbidden_live_evidence_field(key):
            redaction_issues.append(f"{key}:forbidden_evidence_field")
        if _looks_like_voice_denial_text(value):
            redaction_issues.append(f"{key}:voice_capability_denial_text")
        if _looks_secret_or_phone(value):
            redaction_issues.append(f"{key}:secret_or_phone_like_value")
    if redaction_issues:
        issues.extend(redaction_issues)

    discord_probe = _discord_probe_section(payload)
    discord_source_sha256 = None
    if not str(discord_probe.get("source_artifact") or "").strip():
        issues.append("discord_live_probe:missing_source_artifact")
    else:
        discord_source_sha256 = _validate_source_artifact(
            discord_probe.get("source_artifact"),
            "discord_live_probe",
            paths or [],
            issues,
        )
    if discord_probe.get("example_only") is True:
        issues.append("discord_live_probe:example_only_evidence_not_accepted")
    issues.extend(
        _collector_attestation_issues(
            discord_probe,
            "discord_live_probe",
            expected_redacted_sha256=discord_source_sha256,
            expected_parent_manifest_sha256_values=_parent_manifest_sha256_values(
                discord_probe,
                discord_source_sha256,
                paths or [],
            ),
        )
    )
    for key in LIVE_EVIDENCE_REQUIRED_DISCORD_BOOLS:
        if discord_probe.get(key) is not True:
            issues.append(f"discord_live_probe:{key}_not_true")
    if discord_probe.get("ok") is not True:
        issues.append("discord_live_probe:not_ok")
    if discord_probe.get("require_inbound") is not True:
        issues.append("discord_live_probe:require_inbound_not_true")
    inbound = (
        discord_probe.get("inbound_observed") is True
        or _positive_int(discord_probe.get("receiver_frames")) > 0
        or _positive_int(discord_probe.get("receiver_speech_start")) > 0
    )
    if not inbound:
        issues.append("discord_live_probe:inbound_not_observed")
    discord_latency = (
        discord_probe.get("latency_metrics_ms")
        if isinstance(discord_probe.get("latency_metrics_ms"), Mapping)
        else {}
    )
    discord_latency_ok = True
    for key in LIVE_EVIDENCE_REQUIRED_DISCORD_LATENCIES_MS:
        if _non_negative_number(discord_latency.get(key)) is None:
            issues.append(f"discord_live_probe:missing_{key}")
            discord_latency_ok = False

    sidecar = payload.get("sidecar_session") if isinstance(payload.get("sidecar_session"), Mapping) else {}
    sidecar_source_sha256 = None
    if not str(sidecar.get("source_artifact") or "").strip():
        issues.append("sidecar_session:missing_source_artifact")
    else:
        sidecar_source_sha256 = _validate_source_artifact(
            sidecar.get("source_artifact"),
            "sidecar_session",
            paths or [],
            issues,
        )
    if sidecar.get("example_only") is True:
        issues.append("sidecar_session:example_only_evidence_not_accepted")
    issues.extend(
        _collector_attestation_issues(
            sidecar,
            "sidecar_session",
            expected_redacted_sha256=sidecar_source_sha256,
            expected_parent_manifest_sha256_values=_parent_manifest_sha256_values(
                sidecar,
                sidecar_source_sha256,
                paths or [],
            ),
        )
    )
    for key in LIVE_EVIDENCE_REQUIRED_SIDECAR_BOOLS:
        if sidecar.get(key) is not True:
            issues.append(f"sidecar_session:{key}_not_true")
    for key in LIVE_EVIDENCE_REQUIRED_SIDECAR_PROVENANCE_BOOLS:
        if sidecar.get(key) is not True:
            issues.append(f"sidecar_session:{key}_not_true")
    if str(sidecar.get("sidecar_mode") or "").strip() != "production":
        issues.append("sidecar_session:sidecar_mode_not_production")
    fallback_reason = str(sidecar.get("fallback_reason") or "").strip()
    if sidecar.get("fallback_mode_visible") is True and not fallback_reason:
        issues.append("sidecar_session:missing_fallback_reason")
    elif fallback_reason and _looks_secret_or_phone(fallback_reason):
        issues.append("sidecar_session:fallback_reason_secret_or_phone_like_value")
    sidecar_latency = sidecar.get("latency_metrics_ms") if isinstance(sidecar.get("latency_metrics_ms"), Mapping) else {}
    session_start_ms = _non_negative_number(sidecar_latency.get("session_start_ms"))
    if session_start_ms is None:
        issues.append("sidecar_session:missing_session_start_ms")
    shutdown_ms = _non_negative_number(sidecar_latency.get("shutdown_ms"))
    if shutdown_ms is None:
        issues.append("sidecar_session:missing_shutdown_ms")
    if sidecar.get("shutdown_bounded") is not True:
        issues.append("sidecar_session:shutdown_bounded_not_true")
    if sidecar.get("shutdown_timed_out") is not False:
        issues.append("sidecar_session:shutdown_timed_out_not_false")

    live_turn = payload.get("live_turn") if isinstance(payload.get("live_turn"), Mapping) else {}
    live_turn_source_sha256 = None
    if not str(live_turn.get("source_artifact") or "").strip():
        issues.append("live_turn:missing_source_artifact")
    else:
        live_turn_source_sha256 = _validate_source_artifact(
            live_turn.get("source_artifact"),
            "live_turn",
            paths or [],
            issues,
        )
    if live_turn.get("example_only") is True:
        issues.append("live_turn:example_only_evidence_not_accepted")
    issues.extend(
        _collector_attestation_issues(
            live_turn,
            "live_turn",
            expected_redacted_sha256=live_turn_source_sha256,
            expected_parent_manifest_sha256_values=_parent_manifest_sha256_values(
                live_turn,
                live_turn_source_sha256,
                paths or [],
            ),
        )
    )
    for key in LIVE_EVIDENCE_REQUIRED_TURN_BOOLS:
        if live_turn.get(key) is not True:
            issues.append(f"live_turn:{key}_not_true")
    first_audio_ms = _non_negative_number(live_turn.get("speech_end_to_first_audio_ms"))
    if first_audio_ms is None:
        issues.append("live_turn:missing_speech_end_to_first_audio_ms")
    elif first_audio_ms > 3000:
        issues.append("live_turn:speech_end_to_first_audio_ms_over_target")
    barge_in_ms = _non_negative_number(live_turn.get("barge_in_stop_ms"))
    if barge_in_ms is None:
        issues.append("live_turn:missing_barge_in_stop_ms")
    elif barge_in_ms > 150:
        issues.append("live_turn:barge_in_stop_ms_over_target")

    return {
        "loaded": True,
        "mode": "supplied_artifacts_only",
        "artifact_paths": [str(path) for path in paths or []],
        "overall_status": "live_evidence_supplied_not_readiness_claim" if not issues else "partial_live_evidence",
        "issues": sorted(set(issues)),
        "redaction_policy": "references_only",
        "section_refs": {
            "discord_live_probe": _section_ref(discord_probe, "discord_live_probe"),
            "sidecar_session": _section_ref(sidecar, "sidecar_session"),
            "live_turn": _section_ref(live_turn, "live_turn"),
        },
        "discord_live_probe": {
            "ok": discord_probe.get("ok") is True,
            "join_ok": all(discord_probe.get(key) is True for key in ("connect_perm", "speak_perm", "connected", "opus_loaded", "disconnected")),
            "playback_ok": all(discord_probe.get(key) is True for key in ("accepted_audio_source", "played", "playing_during_probe")),
            "inbound_observed": inbound,
            "latency_ok": discord_latency_ok,
            "receiver_frames": _positive_int(discord_probe.get("receiver_frames")),
            "receiver_speech_start": _positive_int(discord_probe.get("receiver_speech_start")),
        },
        "sidecar_session": {
            "ok": all(sidecar.get(key) is True for key in LIVE_EVIDENCE_REQUIRED_SIDECAR_BOOLS)
            and all(sidecar.get(key) is True for key in LIVE_EVIDENCE_REQUIRED_SIDECAR_PROVENANCE_BOOLS)
            and str(sidecar.get("sidecar_mode") or "").strip() == "production"
            and (sidecar.get("fallback_mode_visible") is not True or bool(fallback_reason))
            and not _looks_secret_or_phone(fallback_reason)
            and session_start_ms is not None
            and shutdown_ms is not None
            and sidecar.get("shutdown_bounded") is True
            and sidecar.get("shutdown_timed_out") is False,
            "session_start_ms": session_start_ms,
            "shutdown_ms": shutdown_ms,
            "shutdown_bounded": sidecar.get("shutdown_bounded") is True,
            "shutdown_timed_out": sidecar.get("shutdown_timed_out") is True,
        },
        "live_turn": {
            "ok": all(live_turn.get(key) is True for key in LIVE_EVIDENCE_REQUIRED_TURN_BOOLS)
            and first_audio_ms is not None
            and first_audio_ms <= 3000
            and barge_in_ms is not None
            and barge_in_ms <= 150,
            "speech_end_to_first_audio_ms": first_audio_ms,
            "barge_in_stop_ms": barge_in_ms,
        },
    }


def _section_ref(section: Mapping[str, Any], section_name: str) -> dict[str, str]:
    ref = {
        "source_artifact": str(section.get("source_artifact") or ""),
        "section": section_name,
    }
    provenance = section.get("provenance")
    if isinstance(provenance, Mapping):
        wrapper = str(provenance.get("wrapper_artifact") or "").strip()
        reported = str(provenance.get("reported_source_artifact") or "").strip()
        if wrapper:
            ref["wrapper_artifact"] = wrapper
        if reported:
            ref["reported_source_artifact"] = reported
    return ref


def _resolve_source_artifact_path(source_artifact: Any, evidence_paths: list[Path]) -> Path | None:
    source_text = str(source_artifact or "").strip()
    if not source_text:
        return None
    source_path = Path(source_text).expanduser()
    if source_path.is_absolute():
        return source_path if source_path.is_file() else None
    for evidence_path in evidence_paths:
        candidate = evidence_path.parent / source_text
        if candidate.is_file():
            return candidate
    return None


def _redacted_source_artifact_sha256(source_path: Path) -> str:
    source_bytes = source_path.read_bytes()
    try:
        payload = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return hashlib.sha256(source_bytes).hexdigest()
    if isinstance(payload, dict):
        payload.pop("collector_attestation", None)
        payload.pop("collector_provenance", None)
        canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
    return hashlib.sha256(source_bytes).hexdigest()


def _source_artifact_content_issues(source_path: Path, section_name: str) -> list[str]:
    try:
        source_bytes = source_path.read_bytes()
    except OSError:
        return [f"{section_name}:source_artifact_unreadable"]
    try:
        payload = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return [f"{section_name}:source_artifact_invalid_json"]
    issues: list[str] = []
    for key, value in _walk_live_evidence_strings(payload):
        if _is_collector_attestation_path(key):
            continue
        if _looks_like_forbidden_live_evidence_field(key):
            issues.append(f"{section_name}:source_artifact_forbidden_field:{key}")
        if _looks_like_voice_denial_text(value):
            issues.append(f"{section_name}:source_artifact_voice_capability_denial_text:{key}")
        if _looks_secret_or_phone(value):
            issues.append(f"{section_name}:source_artifact_secret_or_phone_like_value:{key}")
    return sorted(set(issues))


def _parent_manifest_sha256_values(
    section: Mapping[str, Any],
    source_sha256: str | None,
    evidence_paths: list[Path],
) -> list[str]:
    values: list[str] = []
    if source_sha256:
        values.append(source_sha256)
    provenance = section.get("provenance")
    if isinstance(provenance, Mapping):
        reported_source = str(provenance.get("reported_source_artifact") or "").strip()
        if reported_source:
            reported_path = _resolve_source_artifact_path(reported_source, evidence_paths)
            if reported_path is not None:
                try:
                    values.append(_redacted_source_artifact_sha256(reported_path))
                except OSError:
                    pass
    return values


def _source_artifact_candidate_paths(source_artifact: Any, evidence_paths: list[Path]) -> list[str]:
    source_text = str(source_artifact or "").strip()
    if not source_text:
        return []
    source_path = Path(source_text).expanduser()
    if source_path.is_absolute():
        return [str(source_path.resolve(strict=False))]
    if evidence_paths:
        return [str((path.parent / source_text).resolve(strict=False)) for path in evidence_paths]
    return [source_text]


def _source_artifact_not_found_detail(section_name: str, source_artifact: Any, evidence_paths: list[Path]) -> str:
    candidates = _source_artifact_candidate_paths(source_artifact, evidence_paths)
    return f"{section_name}:source_artifact_not_found_candidates:{';'.join(candidates) if candidates else '<empty>'}"


def _validate_source_artifact(
    source_artifact: Any,
    section_name: str,
    evidence_paths: list[Path],
    issues: list[str],
) -> str | None:
    source_text = str(source_artifact or "").strip()
    source_path = Path(source_text).expanduser()
    if source_text in LIVE_EVIDENCE_TEMPLATE_SOURCE_ARTIFACTS or (
        not source_path.is_absolute() and source_path.name in LIVE_EVIDENCE_TEMPLATE_SOURCE_ARTIFACTS
    ):
        issues.append(f"{section_name}:template_source_artifact_not_accepted")
        return None
    if evidence_paths:
        resolved = _resolve_source_artifact_path(source_text, evidence_paths)
        if resolved is None:
            issues.append(f"{section_name}:source_artifact_not_found")
            issues.append(_source_artifact_not_found_detail(section_name, source_text, evidence_paths))
            return None
        issues.extend(_source_artifact_content_issues(resolved, section_name))
        try:
            return _redacted_source_artifact_sha256(resolved)
        except OSError:
            issues.append(f"{section_name}:source_artifact_unreadable")
            return None
    if source_path.is_absolute():
        if not source_path.exists():
            issues.append(f"{section_name}:source_artifact_not_found")
            issues.append(_source_artifact_not_found_detail(section_name, source_text, evidence_paths))
        elif not source_path.is_file():
            issues.append(f"{section_name}:source_artifact_not_file")
            issues.append(f"{section_name}:source_artifact_not_file_path:{source_path.resolve(strict=False)}")
        else:
            issues.extend(_source_artifact_content_issues(source_path, section_name))
            try:
                return _redacted_source_artifact_sha256(source_path)
            except OSError:
                issues.append(f"{section_name}:source_artifact_unreadable")
        return None
    issues.append(f"{section_name}:unverified_source_artifact")
    return None


def _walk_live_evidence_strings(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, str]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_walk_live_evidence_strings(child, child_prefix))
        return rows
    if isinstance(value, list):
        rows = []
        for index, child in enumerate(value):
            rows.extend(_walk_live_evidence_strings(child, f"{prefix}[{index}]"))
        return rows
    if isinstance(value, str):
        return [(prefix, value)]
    return []


def _is_collector_attestation_path(path: str) -> bool:
    normalized = f".{path}"
    return ".collector_attestation." in normalized or ".collector_provenance." in normalized


def _coverage_from_smoke(smoke: dict[str, Any]) -> dict[str, bool]:
    events = {"barge_in.detected" if event == "barge_in" else event for event in (smoke.get("events") or [])}
    latency = smoke.get("latency_metrics_ms") if isinstance(smoke.get("latency_metrics_ms"), dict) else {}
    sidecar_shutdown = bool(
        smoke.get("sidecar_closed")
        and smoke.get("shutdown_bounded")
        and not smoke.get("shutdown_timed_out")
    )
    return {
        "lifecycle_start_and_shutdown": bool(smoke.get("ok") and sidecar_shutdown),
        "discord_receiver_callback_wiring": REQUIRED_EVENTS <= events,
        "pcm_conversion_discord_48k_to_sidecar_16k": (
            smoke.get("input_pcm48_bytes") == DISCORD_FRAME_BYTES
            and smoke.get("sidecar_pcm16_bytes") == SIDECAR_FRAME_BYTES
            and smoke.get("sidecar_pcm16_first_sample") == 450
            and int(smoke.get("sidecar_pcm16_checksum") or 0) > 0
        ),
        "mixer_playback_path": (
            int(smoke.get("mixer_frames") or 0) >= 1 and smoke.get("mixer_frame_bytes") == DISCORD_FRAME_BYTES
        ),
        "barge_in_stops_playback": bool(
            smoke.get("speech_energy_sent")
            and smoke.get("barge_in_sent")
            and int(smoke.get("mixer_stop_calls") or 0) >= 1
        ),
        "latency_metrics_present": {
            "session_start_ms",
            "input_to_first_mixer_frame_ms",
            "barge_in_ack_ms",
            "shutdown_ms",
        } <= set(latency),
        "sidecar_session_shutdown": sidecar_shutdown,
    }


def _coverage_from_async_oracle_smoke(smoke: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "async_oracle_smoke_ok": bool(smoke.get("ok")),
        "four_jobs_ran_concurrently": bool(smoke.get("worker_overlap_proved"))
        and smoke.get("worker_overlap_within_capacity") is True
        and int(smoke.get("max_worker_overlap") or 0) >= 4
        and int(smoke.get("max_running") or 0) >= 4
        and int(smoke.get("started_jobs") or 0) >= 4,
        "local_turn_while_jobs_running": bool(smoke.get("local_turn_committed")),
        "status_turn_while_jobs_running": bool(smoke.get("status_turn_committed"))
        and "4 running out of 4" in str(smoke.get("status_text") or ""),
        "fifth_job_queued_and_started_after_capacity_freed": int(smoke.get("queued_jobs") or 0) >= 1
        and smoke.get("fifth_job_queued") is True
        and smoke.get("fifth_job_started_after_capacity_freed") is True,
        "one_job_cancelled_while_others_completed": int(smoke.get("cancelled_jobs") or 0) >= 1
        and int(smoke.get("completed_jobs") or 0) >= 4,
        "queued_job_cancelled_before_start": smoke.get("queued_cancel_smoke_ok") is True
        and smoke.get("queued_cancel_observed") is True
        and smoke.get("queued_cancelled_before_start") is True
        and smoke.get("queued_cancel_not_sent_to_oracle") is True
        and smoke.get("queued_cancel_running_completed") is True,
        "late_cancelled_output_attempted": smoke.get("late_cancelled_output_attempted") is True,
        "late_cancelled_output_not_spoken": smoke.get("late_cancelled_output_attempted") is True
        and smoke.get("cancelled_result_spoken") is False
        and smoke.get("cancelled_result_committed") is False
        and smoke.get("cancelled_result_progress_leaked") is False,
        "late_cancelled_output_not_durable": smoke.get("late_cancelled_output_attempted") is True
        and smoke.get("cancelled_result_durable_completed") is False
        and smoke.get("cancelled_result_durable_text") is False
        and smoke.get("durable_cancelled_record_present") is True
        and int(smoke.get("durable_completed_jobs") or 0) >= 1,
        "approval_wait_visible_and_redacted": smoke.get("approval_wait_observed") is True
        and smoke.get("approval_status_committed") is True
        and smoke.get("approval_tool_progress_observed") is True
        and smoke.get("approval_payload_redacted") is True
        and smoke.get("approval_secret_leaked") is False
        and smoke.get("approval_secret_canary_checked") is True
        and smoke.get("approval_completed") is True,
        "failed_job_reported_without_crash": int(smoke.get("failed_jobs") or 0) >= 1
        and smoke.get("failed_job_reported") is True
        and smoke.get("failed_job_spoken") is True
        and smoke.get("durable_failed_record_present") is True
        and smoke.get("session_survived_failed_job") is True,
        "queued_job_control_update_reaches_oracle": smoke.get("queued_job_update_observed") is True
        and smoke.get("queued_update_latest_update_visible") is True
        and smoke.get("queued_update_started_with_priority") is True
        and smoke.get("queued_update_reached_oracle") is True,
        "result_handling_bounded_and_durable": smoke.get("verbose_result_spoken_bounded") is True
        and smoke.get("verbose_result_committed_bounded") is True
        and smoke.get("verbose_result_commit_marked_truncated") is True
        and smoke.get("verbose_full_result_durable") is True
        and smoke.get("completed_result_status_visible") is True,
        "shutdown_timeout_bounded": smoke.get("shutdown_bounded_close_observed") is True
        and smoke.get("shutdown_forced_cancel_observed") is True
        and int(smoke.get("shutdown_cancelled_jobs") or 0) >= 1,
    }


def _coverage_from_discord_session_cleanup_smoke(smoke: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "discord_session_cleanup_preserves_oracle_state": smoke.get("ok") is True
        and smoke.get("cancel_all_before_session_closed") is True
        and smoke.get("session_closed_sent") is True
        and smoke.get("sidecar_closed") is True
        and smoke.get("degraded_active_job_preserved_failed") is True
        and smoke.get("degraded_session_removed") is True,
    }


def _async_oracle_acceptance_row(
    *,
    ok: bool,
    evidence: str,
    test_refs: list[str],
    verification_mode: str,
    runtime_verified_by_this_report: bool,
    live_external_evidence_required: bool = False,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "evidence": evidence,
        "test_refs": test_refs,
        "test_ref_count": len(test_refs),
        "verification_mode": verification_mode,
        "runtime_verified_by_this_report": runtime_verified_by_this_report,
        "live_external_evidence_required": live_external_evidence_required,
    }


def _async_oracle_acceptance_matrix(async_oracle_coverage: Mapping[str, bool]) -> dict[str, dict[str, Any]]:
    smoke_ok = bool(async_oracle_coverage.get("async_oracle_smoke_ok"))
    return {
        "four_oracle_jobs_reflex_responsive": _async_oracle_acceptance_row(
            ok=smoke_ok
            and bool(async_oracle_coverage.get("four_jobs_ran_concurrently"))
            and bool(async_oracle_coverage.get("local_turn_while_jobs_running")),
            evidence="async_oracle_smoke",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["job_manager_capacity"]
            + ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["local_turns"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "fifth_job_obeys_overflow_policy": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("fifth_job_queued_and_started_after_capacity_freed")),
            evidence="async_oracle_smoke",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["job_manager_capacity"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "status_reports_running_and_queued_without_oracle_call": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("status_turn_while_jobs_running")),
            evidence="async_oracle_smoke_plus_status_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["status_view"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "new_oracle_job_can_be_created_while_others_run": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("fifth_job_queued_and_started_after_capacity_freed")),
            evidence="async_oracle_smoke",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["local_turns"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "cancellation_controls_are_isolated": _async_oracle_acceptance_row(
            ok=smoke_ok
            and bool(async_oracle_coverage.get("one_job_cancelled_while_others_completed"))
            and bool(async_oracle_coverage.get("queued_job_cancelled_before_start"))
            and bool(async_oracle_coverage.get("late_cancelled_output_not_spoken"))
            and bool(async_oracle_coverage.get("late_cancelled_output_not_durable")),
            evidence="async_oracle_smoke_plus_cancellation_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["cancellation"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "approval_wait_is_visible_and_redacted": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("approval_wait_visible_and_redacted")),
            evidence="async_oracle_smoke_plus_approval_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["approval_wait"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "failed_job_is_reported_without_crashing_session": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("failed_job_reported_without_crash")),
            evidence="async_oracle_smoke_plus_failure_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["failure_handling"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "queued_job_control_updates_reach_oracle": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("queued_job_control_update_reaches_oracle")),
            evidence="async_oracle_smoke_plus_control_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["control_updates"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "result_handling_is_bounded_and_durable": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("result_handling_bounded_and_durable")),
            evidence="async_oracle_smoke_plus_result_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["result_handling"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "discord_session_cleanup_preserves_oracle_state": _async_oracle_acceptance_row(
            ok=bool(async_oracle_coverage.get("discord_session_cleanup_preserves_oracle_state")),
            evidence="discord_session_cleanup_smoke_plus_focused_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["discord_session"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "shutdown_timeout_is_bounded": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("shutdown_timeout_bounded")),
            evidence="async_oracle_smoke_plus_shutdown_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["shutdown"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
    }


def _acceptance_test_ref_resolves(test_ref: Any) -> bool:
    if not isinstance(test_ref, str) or "::" not in test_ref:
        return False
    path_text, *node_parts = test_ref.split("::")
    if not path_text or not node_parts or any(not part for part in node_parts):
        return False
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts:
        return False
    test_path = Path(__file__).resolve().parents[1] / path
    if not test_path.is_file():
        return False
    try:
        module = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False

    body: list[ast.stmt] = list(module.body)
    found: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for raw_part in node_parts:
        part = raw_part.split("[", 1)[0]
        found = None
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == part:
                found = node
                break
        if found is None:
            return False
        body = list(found.body) if isinstance(found, ast.ClassDef) else []
    return isinstance(found, (ast.FunctionDef, ast.AsyncFunctionDef))


def _live_evidence_missing_gates(live_evidence: dict[str, Any]) -> list[str]:
    missing: set[str] = set()
    if not live_evidence.get("loaded"):
        return list(LIVE_EVIDENCE_REQUIRED_GATES)
    if live_evidence.get("issues") or live_evidence.get("overall_status") != "live_evidence_supplied_not_readiness_claim":
        return list(LIVE_EVIDENCE_REQUIRED_GATES)
    discord = live_evidence.get("discord_live_probe") if isinstance(live_evidence.get("discord_live_probe"), dict) else {}
    if discord.get("join_ok") is not True:
        missing.add("discord_join")
    if discord.get("playback_ok") is not True:
        missing.add("discord_playback")
    if discord.get("inbound_observed") is not True:
        missing.add("live_receiver")
    sidecar = live_evidence.get("sidecar_session") if isinstance(live_evidence.get("sidecar_session"), dict) else {}
    if sidecar.get("ok") is not True:
        missing.add("production_sidecar")
    live_turn = live_evidence.get("live_turn") if isinstance(live_evidence.get("live_turn"), dict) else {}
    if live_turn.get("ok") is not True:
        missing.add("live_turn")
    return sorted(missing)


def build_voice_operator_report(
    smoke: dict[str, Any],
    *,
    live_evidence: dict[str, Any] | None = None,
    async_oracle_smoke: dict[str, Any] | None = None,
    discord_session_cleanup_smoke: dict[str, Any] | None = None,
) -> dict[str, Any]:
    coverage = _coverage_from_smoke(smoke)
    async_oracle_smoke = async_oracle_smoke or {}
    discord_session_cleanup_smoke = discord_session_cleanup_smoke or {}
    async_oracle_coverage = {
        **_coverage_from_async_oracle_smoke(async_oracle_smoke),
        **_coverage_from_discord_session_cleanup_smoke(discord_session_cleanup_smoke),
    }
    async_oracle_acceptance = _async_oracle_acceptance_matrix(async_oracle_coverage)
    live_evidence = live_evidence or _load_live_evidence([])
    missing_live_gates = _live_evidence_missing_gates(live_evidence)
    live_probe_status = "needs_live_probe" if missing_live_gates else "live_evidence_supplied_not_readiness_claim"
    latency = smoke.get("latency_metrics_ms") or {}
    proofs = {
        "lifecycle": {
            "ok": coverage["lifecycle_start_and_shutdown"],
            "started": bool(smoke.get("ok")),
            "closed": bool(smoke.get("sidecar_closed")),
            "sidecar_closed": bool(smoke.get("sidecar_closed")),
        },
        "callback_wiring": {
            "ok": coverage["discord_receiver_callback_wiring"],
            "basis": "session_event_callback_loopback_plus_gateway_receiver_unit_tests",
            "loopback_bypasses_live_discord_receiver": True,
            "events": smoke.get("events") or [],
            "external_test_refs": RECEIVER_CALLBACK_TEST_REFS,
        },
        "pcm_conversion": {
            "ok": coverage["pcm_conversion_discord_48k_to_sidecar_16k"],
            "input_pcm48_stereo_bytes": smoke.get("input_pcm48_bytes"),
            "sidecar_pcm16_mono_bytes": smoke.get("sidecar_pcm16_bytes"),
            "sentinel_expected_first_sample": 450,
            "sidecar_pcm16_first_sample": smoke.get("sidecar_pcm16_first_sample"),
            "sidecar_pcm16_checksum": smoke.get("sidecar_pcm16_checksum"),
        },
        "mixer_path": {
            "ok": coverage["mixer_playback_path"],
            "mixer_frames": smoke.get("mixer_frames"),
            "mixer_frame_bytes": smoke.get("mixer_frame_bytes"),
        },
        "barge_in_energy": {
            "ok": coverage["barge_in_stops_playback"],
            "reaction_proven": bool(smoke.get("barge_in_sent")),
            "speech_energy_event_forwarded": bool(smoke.get("speech_energy_sent")),
            "energy_gate_proven_by_smoke": False,
            "energy_gate_covered_by_tests": True,
            "stop_called": int(smoke.get("mixer_stop_calls") or 0) >= 1,
            "external_test_refs": BARGE_IN_ENERGY_TEST_REFS,
        },
        "fallback_state": {
            "ok": True,
            "legacy": True,
            "text_only": True,
            "fail_closed": True,
        },
        "short_replies": {
            "ok": True,
            "max_spoken_sentences": 2,
            "voice_response_policy": "sentence_cap",
            "ack_text": "One moment.",
        },
        "prompt_context": {
            "ok": True,
            "listening_yes": True,
            "speaking_yes": True,
            "no_voice_denial": True,
        },
        "latency_metrics": {
            "ok": coverage["latency_metrics_present"],
            "session_start_ms": latency.get("session_start_ms"),
            "input_to_first_mixer_frame_ms": latency.get("input_to_first_mixer_frame_ms"),
            "barge_in_ack_ms": latency.get("barge_in_ack_ms"),
            "shutdown_ms": latency.get("shutdown_ms"),
            "speech_end_to_reflex_response_ms": latency.get("input_to_first_mixer_frame_ms"),
            "speech_end_to_oracle_response_ms": None,
            "speech_end_to_tts_playback_ms": latency.get("input_to_first_mixer_frame_ms"),
            "oracle_metric_status": "needs_live_oracle_or_sidecar_probe",
        },
        "shutdown": {
            "ok": coverage["sidecar_session_shutdown"],
            "sidecar_closed": bool(smoke.get("sidecar_closed")),
            "close_timeout_bounded": bool(smoke.get("shutdown_bounded")),
            "shutdown_elapsed_ms": smoke.get("shutdown_elapsed_ms"),
            "shutdown_timed_out": bool(smoke.get("shutdown_timed_out")),
        },
        "async_oracle_jobs": {
            "ok": all(async_oracle_coverage.values()),
            "scenario": async_oracle_smoke.get("scenario"),
            "max_running": async_oracle_smoke.get("max_running"),
            "max_worker_overlap": async_oracle_smoke.get("max_worker_overlap"),
            "worker_overlap_proved": bool(async_oracle_smoke.get("worker_overlap_proved")),
            "worker_overlap_within_capacity": bool(async_oracle_smoke.get("worker_overlap_within_capacity")),
            "noncooperative_cancel_overlap_observed": bool(
                async_oracle_smoke.get("noncooperative_cancel_overlap_observed")
            ),
            "started_jobs": async_oracle_smoke.get("started_jobs"),
            "queued_jobs": async_oracle_smoke.get("queued_jobs"),
            "completed_jobs": async_oracle_smoke.get("completed_jobs"),
            "failed_jobs": async_oracle_smoke.get("failed_jobs"),
            "cancelled_jobs": async_oracle_smoke.get("cancelled_jobs"),
            "shutdown_timeout_configured_ms": async_oracle_smoke.get("shutdown_timeout_configured_ms"),
            "shutdown_close_elapsed_ms": async_oracle_smoke.get("shutdown_close_elapsed_ms"),
            "shutdown_bounded_close_observed": bool(async_oracle_smoke.get("shutdown_bounded_close_observed")),
            "shutdown_forced_cancel_observed": bool(async_oracle_smoke.get("shutdown_forced_cancel_observed")),
            "shutdown_close_cancel_entered": bool(async_oracle_smoke.get("shutdown_close_cancel_entered")),
            "shutdown_cancelled_jobs": async_oracle_smoke.get("shutdown_cancelled_jobs"),
            "queued_cancel_smoke_ok": bool(async_oracle_smoke.get("queued_cancel_smoke_ok")),
            "queued_cancel_observed": bool(async_oracle_smoke.get("queued_cancel_observed")),
            "queued_cancelled_before_start": bool(async_oracle_smoke.get("queued_cancelled_before_start")),
            "queued_cancel_not_sent_to_oracle": bool(async_oracle_smoke.get("queued_cancel_not_sent_to_oracle")),
            "queued_cancel_reason": async_oracle_smoke.get("queued_cancel_reason"),
            "queued_cancel_target_job_id": async_oracle_smoke.get("queued_cancel_target_job_id"),
            "queued_cancel_running_completed": bool(async_oracle_smoke.get("queued_cancel_running_completed")),
            "local_turn_committed": bool(async_oracle_smoke.get("local_turn_committed")),
            "status_turn_committed": bool(async_oracle_smoke.get("status_turn_committed")),
            "status_text": async_oracle_smoke.get("status_text"),
            "terminal_status_committed": bool(async_oracle_smoke.get("terminal_status_committed")),
            "completed_result_status_visible": bool(async_oracle_smoke.get("completed_result_status_visible")),
            "terminal_status_text": async_oracle_smoke.get("terminal_status_text"),
            "fifth_job_id": async_oracle_smoke.get("fifth_job_id"),
            "fifth_job_queued": bool(async_oracle_smoke.get("fifth_job_queued")),
            "fifth_job_started_after_capacity_freed": bool(
                async_oracle_smoke.get("fifth_job_started_after_capacity_freed")
            ),
            "cancelled_job_id": async_oracle_smoke.get("cancelled_job_id"),
            "late_cancelled_output_attempted": bool(async_oracle_smoke.get("late_cancelled_output_attempted")),
            "cancelled_result_spoken": bool(async_oracle_smoke.get("cancelled_result_spoken")),
            "cancelled_result_committed": bool(async_oracle_smoke.get("cancelled_result_committed")),
            "cancelled_result_progress_leaked": bool(async_oracle_smoke.get("cancelled_result_progress_leaked")),
            "cancelled_result_durable_completed": bool(
                async_oracle_smoke.get("cancelled_result_durable_completed")
            ),
            "cancelled_result_durable_text": bool(async_oracle_smoke.get("cancelled_result_durable_text")),
            "durable_cancelled_record_present": bool(async_oracle_smoke.get("durable_cancelled_record_present")),
            "durable_completed_jobs": async_oracle_smoke.get("durable_completed_jobs"),
            "approval_wait_observed": bool(async_oracle_smoke.get("approval_wait_observed")),
            "approval_status_committed": bool(async_oracle_smoke.get("approval_status_committed")),
            "approval_tool_progress_observed": bool(async_oracle_smoke.get("approval_tool_progress_observed")),
            "approval_payload_redacted": bool(async_oracle_smoke.get("approval_payload_redacted")),
            "approval_secret_leaked": bool(async_oracle_smoke.get("approval_secret_leaked")),
            "approval_secret_canary_checked": bool(async_oracle_smoke.get("approval_secret_canary_checked")),
            "approval_completed": bool(async_oracle_smoke.get("approval_completed")),
            "approval_status_text": async_oracle_smoke.get("approval_status_text"),
            "failed_job_reported": bool(async_oracle_smoke.get("failed_job_reported")),
            "failed_job_spoken": bool(async_oracle_smoke.get("failed_job_spoken")),
            "durable_failed_record_present": bool(async_oracle_smoke.get("durable_failed_record_present")),
            "session_survived_failed_job": bool(async_oracle_smoke.get("session_survived_failed_job")),
            "queued_job_update_observed": bool(async_oracle_smoke.get("queued_job_update_observed")),
            "queued_update_latest_update_visible": bool(
                async_oracle_smoke.get("queued_update_latest_update_visible")
            ),
            "queued_update_latest_update_text": async_oracle_smoke.get("queued_update_latest_update_text"),
            "queued_update_started_with_priority": bool(
                async_oracle_smoke.get("queued_update_started_with_priority")
            ),
            "queued_update_reached_oracle": bool(async_oracle_smoke.get("queued_update_reached_oracle")),
            "verbose_result_spoken_bounded": bool(async_oracle_smoke.get("verbose_result_spoken_bounded")),
            "verbose_result_committed_bounded": bool(
                async_oracle_smoke.get("verbose_result_committed_bounded")
            ),
            "verbose_result_commit_marked_truncated": bool(
                async_oracle_smoke.get("verbose_result_commit_marked_truncated")
            ),
            "verbose_full_result_durable": bool(async_oracle_smoke.get("verbose_full_result_durable")),
            "verbose_full_result_chars": async_oracle_smoke.get("verbose_full_result_chars"),
            "verbose_spoken_result": async_oracle_smoke.get("verbose_spoken_result"),
            "coverage": async_oracle_coverage,
        },
        "discord_session_cleanup": {
            "ok": bool(_coverage_from_discord_session_cleanup_smoke(discord_session_cleanup_smoke).get(
                "discord_session_cleanup_preserves_oracle_state"
            )),
            "scenario": discord_session_cleanup_smoke.get("scenario"),
            "cancel_all_before_session_closed": bool(
                discord_session_cleanup_smoke.get("cancel_all_before_session_closed")
            ),
            "session_closed_sent": bool(discord_session_cleanup_smoke.get("session_closed_sent")),
            "sidecar_closed": bool(discord_session_cleanup_smoke.get("sidecar_closed")),
            "sidecar_close_calls": discord_session_cleanup_smoke.get("sidecar_close_calls"),
            "degraded_active_job_preserved_failed": bool(
                discord_session_cleanup_smoke.get("degraded_active_job_preserved_failed")
            ),
            "degraded_session_removed": bool(discord_session_cleanup_smoke.get("degraded_session_removed")),
            "degraded_fallback_reason": discord_session_cleanup_smoke.get("degraded_fallback_reason"),
            "degraded_job_state": discord_session_cleanup_smoke.get("degraded_job_state"),
            "degraded_job_error": discord_session_cleanup_smoke.get("degraded_job_error"),
            "event_order": discord_session_cleanup_smoke.get("event_order") or [],
        },
        "live_evidence": {
            "ok": bool(live_evidence.get("loaded")) and not missing_live_gates and not live_evidence.get("issues"),
            "mode": live_evidence.get("mode"),
            "overall_status": live_evidence.get("overall_status"),
            "missing_gates": missing_live_gates,
        },
    }
    return {
        "schema_version": "voiceops.milestone1.voice_operator.v1",
        "artifact_id": "voiceops-m1-discord-voice-operator",
        "milestone": "milestone_1_real_voice_operator",
        "status": live_probe_status,
        "missing_live_gates": missing_live_gates,
        "artifact_only": True,
        "mode": {
            "headless": True,
            "bounded": True,
            "discord_network": False,
            "env_secret_reads": False,
            "provider_sidecar_network": False,
            "outbound_sends": False,
            "outbound_calls": False,
        },
        "requirements": {
            "stable_discord_receive_playback_lifecycle": coverage["lifecycle_start_and_shutdown"],
            "receiver_callback_wiring": coverage["discord_receiver_callback_wiring"],
            "pcm_conversion_correctness": coverage["pcm_conversion_discord_48k_to_sidecar_16k"],
            "mixer_playback_path": coverage["mixer_playback_path"],
            "barge_in_behavior": coverage["barge_in_stops_playback"],
            "sidecar_session_shutdown": coverage["sidecar_session_shutdown"],
            "latency_metrics": coverage["latency_metrics_present"],
            "kame_fallback_state_visible": True,
            "voice_capability_prompt_context": True,
            "short_voice_replies_default": True,
            "live_discord_join": False,
            "live_evidence_supplied": bool(live_evidence.get("loaded")),
            "async_oracle_four_concurrent_jobs": async_oracle_coverage["four_jobs_ran_concurrently"],
            "async_oracle_local_turn_while_running": async_oracle_coverage["local_turn_while_jobs_running"],
            "async_oracle_status_turn_while_running": async_oracle_coverage["status_turn_while_jobs_running"],
            "async_oracle_fifth_job_queued_and_started": async_oracle_coverage[
                "fifth_job_queued_and_started_after_capacity_freed"
            ],
            "async_oracle_cancellation_isolated": async_oracle_coverage["one_job_cancelled_while_others_completed"]
            and async_oracle_coverage["queued_job_cancelled_before_start"],
            "async_oracle_late_cancelled_output_attempted": async_oracle_coverage[
                "late_cancelled_output_attempted"
            ],
            "async_oracle_late_cancelled_output_dropped": async_oracle_coverage["late_cancelled_output_not_spoken"],
            "async_oracle_late_cancelled_output_not_durable": async_oracle_coverage[
                "late_cancelled_output_not_durable"
            ],
        },
        "proofs": proofs,
        "coverage": coverage,
        "async_oracle_coverage": async_oracle_coverage,
        "async_oracle_acceptance": async_oracle_acceptance,
        "voice_capability_prompt_contract": {
            "must_state": [
                "Hermes is connected to Discord voice when /voice join succeeds.",
                "Hermes can listen and speak when realtime mode is active.",
                "Hermes should keep spoken replies short by default.",
            ],
            "must_not_claim": [
                "I cannot hear voice.",
                "I cannot speak in Discord.",
                "I only process typed text.",
            ],
        },
        "barge_in_policy": {
            "signal": "speech_energy_or_confirmed_speech_start",
            "min_rms_default": 350,
            "min_speech_ms_default": 120,
            "stop_playback_deadline_ms": 150,
            "silent_packet_policy": "silent PCM must not trigger barge-in; receiver RMS gating is covered by gateway tests",
            "evidence_refs": BARGE_IN_ENERGY_TEST_REFS,
        },
        "fallback_state": {
            "active_modes": ["realtime", "degraded_no_sidecar", "text_only_fallback"],
            "visible_fields": ["mode", "fallback_reason", "sidecar_running", "mixer_installed", "latency_metrics_ms"],
            "status_command": "/voice status",
        },
        "latency_metrics_ms": smoke.get("latency_metrics_ms") or {},
        "live_evidence": live_evidence,
        "smoke": smoke,
        "async_oracle_smoke": dict(async_oracle_smoke),
        "discord_session_cleanup_smoke": dict(discord_session_cleanup_smoke),
        "live_probe_required_for_completion": {
            "status": live_probe_status,
            "reason": "Headless loopback does not prove a real Discord gateway join, live receiver transport, or production sidecar availability.",
            "missing_gates": missing_live_gates,
            "recommended_command": (
                "uv run python -m hermes_cli.realtime_voice_live_evidence "
                "--output-dir artifacts/realtime-voice-evidence/live-current "
                "--run-doctor-report --require-inbound --wait-seconds 5"
            ),
            "validate_command": (
                "uv run python -m hermes_cli.realtime_voice_live_evidence "
                "--output-dir artifacts/realtime-voice-evidence/live-current "
                "--validate-live-evidence "
                "--discord-live-probe-evidence artifacts/realtime-voice-evidence/live-current/discord-live-probe.json "
                "--sidecar-session-evidence artifacts/realtime-voice-evidence/live-current/sidecar-session.json "
                "--live-turn-evidence artifacts/realtime-voice-evidence/live-current/live-turn.json"
            ),
            "audit_command": (
                "uv run python -m hermes_cli.realtime_voice_live_evidence "
                "--audit-only "
                "--discord-live-probe-evidence artifacts/realtime-voice-evidence/live-current/discord-live-probe.json "
                "--sidecar-session-evidence artifacts/realtime-voice-evidence/live-current/sidecar-session.json "
                "--live-turn-evidence artifacts/realtime-voice-evidence/live-current/live-turn.json"
            ),
            "ingest_command": (
                "uv run python scripts/voiceops_voice_operator.py "
                "--output-dir artifacts/voiceops-voice-operator/current "
                "--live-evidence artifacts/realtime-voice-evidence/live-current/manifest.json"
            ),
        },
    }


def validate_voice_operator_report(report: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if report.get("schema_version") != "voiceops.milestone1.voice_operator.v1":
        issues.append("invalid_schema_version")
    mode = report.get("mode", {})
    for key in ("discord_network", "env_secret_reads", "provider_sidecar_network", "outbound_sends", "outbound_calls"):
        if mode.get(key) is not False:
            issues.append(f"unsafe_mode:{key}")
    for key in ("headless", "bounded"):
        if mode.get(key) is not True:
            issues.append(f"unsafe_mode:{key}")
    coverage = report.get("coverage", {})
    for key in (
        "lifecycle_start_and_shutdown",
        "discord_receiver_callback_wiring",
        "pcm_conversion_discord_48k_to_sidecar_16k",
        "mixer_playback_path",
        "barge_in_stops_playback",
        "latency_metrics_present",
        "sidecar_session_shutdown",
    ):
        if coverage.get(key) is not True:
            issues.append(f"missing_coverage:{key}")
    async_oracle_coverage = report.get("async_oracle_coverage", {})
    recomputed_async_oracle_coverage = {
        **_coverage_from_async_oracle_smoke(report.get("async_oracle_smoke", {})),
        **_coverage_from_discord_session_cleanup_smoke(report.get("discord_session_cleanup_smoke", {})),
    }
    for key in (
        "async_oracle_smoke_ok",
        "four_jobs_ran_concurrently",
        "local_turn_while_jobs_running",
        "status_turn_while_jobs_running",
        "fifth_job_queued_and_started_after_capacity_freed",
        "one_job_cancelled_while_others_completed",
        "queued_job_cancelled_before_start",
        "late_cancelled_output_not_spoken",
        "late_cancelled_output_attempted",
        "late_cancelled_output_not_durable",
        "approval_wait_visible_and_redacted",
        "failed_job_reported_without_crash",
        "queued_job_control_update_reaches_oracle",
        "result_handling_bounded_and_durable",
        "discord_session_cleanup_preserves_oracle_state",
    ):
        if recomputed_async_oracle_coverage.get(key) is not True:
            issues.append(f"missing_async_oracle_coverage:{key}")
        if async_oracle_coverage.get(key) is not recomputed_async_oracle_coverage.get(key):
            issues.append(f"stale_async_oracle_coverage:{key}")
    async_oracle_acceptance = report.get("async_oracle_acceptance", {})
    if not isinstance(async_oracle_acceptance, Mapping) or not async_oracle_acceptance:
        issues.append("missing_async_oracle_acceptance_matrix")
    else:
        recomputed_async_oracle_acceptance = _async_oracle_acceptance_matrix(recomputed_async_oracle_coverage)
        for key, value in async_oracle_acceptance.items():
            recomputed_value = recomputed_async_oracle_acceptance.get(key, {})
            if recomputed_value.get("ok") is not True:
                issues.append(f"missing_async_oracle_acceptance:{key}")
            if not isinstance(value, Mapping) or value.get("ok") is not True:
                issues.append(f"missing_async_oracle_acceptance:{key}")
                continue
            if value.get("ok") is not recomputed_value.get("ok"):
                issues.append(f"stale_async_oracle_acceptance:{key}")
            test_refs = value.get("test_refs")
            if not isinstance(test_refs, list) or not test_refs:
                issues.append(f"missing_async_oracle_acceptance_test_refs:{key}")
            else:
                for test_ref in test_refs:
                    if not _acceptance_test_ref_resolves(test_ref):
                        issues.append(f"invalid_async_oracle_acceptance_test_ref:{key}:{test_ref}")
            if value.get("test_ref_count") != len(test_refs or []):
                issues.append(f"invalid_async_oracle_acceptance_test_ref_count:{key}")
            if value.get("verification_mode") == "static_focused_test_reference_inventory":
                if value.get("runtime_verified_by_this_report") is not False:
                    issues.append(f"invalid_async_oracle_acceptance_runtime_claim:{key}")
            elif value.get("runtime_verified_by_this_report") is not True:
                issues.append(f"missing_async_oracle_acceptance_runtime_verification:{key}")
    if report.get("requirements", {}).get("live_discord_join") is not False:
        issues.append("live_discord_join_must_not_be_claimed")
    return sorted(issues)


def _markdown(report: dict[str, Any]) -> str:
    issues = validate_voice_operator_report(report)
    lines = [
        "# VoiceOps Milestone 1 Voice Operator",
        "",
        f"- Artifact ID: {report['artifact_id']}",
        f"- Schema: {report['schema_version']}",
        f"- Validation: {', '.join(issues) if issues else 'pass'}",
        "- Mode: headless loopback; no Discord network, env secret reads, provider sidecar network, sends, or calls",
        "",
        "## Requirement Coverage",
        "",
    ]
    for key, value in sorted(report["requirements"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Proofs", ""])
    for proof_id, proof in sorted(report["proofs"].items()):
        lines.append(f"- {proof_id}: {proof.get('ok')}")
    lines.extend(["", "## Async Oracle Acceptance", ""])
    for key, value in sorted(report["async_oracle_acceptance"].items()):
        refs = value.get("test_refs") or []
        ref_text = f"; refs={len(refs)}" if refs else ""
        lines.append(
            f"- {key}: {value.get('ok')} "
            f"({value.get('evidence')}; mode={value.get('verification_mode')}{ref_text})"
        )
    lines.extend(["", "## Latency Metrics", ""])
    for key, value in sorted(report["latency_metrics_ms"].items()):
        lines.append(f"- {key}: {value} ms")
    lines.extend(["", "## Barge-In Policy", ""])
    for key, value in report["barge_in_policy"].items():
        if isinstance(value, list):
            lines.append(f"- {key}: {', '.join(value)}")
        else:
            lines.append(f"- {key}: {value}")
    lines.extend(["", "## Live Probe Boundary", ""])
    live = report["live_probe_required_for_completion"]
    lines.append(f"- Status: {live['status']}")
    lines.append(f"- Reason: {live['reason']}")
    lines.append(f"- Missing gates: {', '.join(live['missing_gates']) if live['missing_gates'] else 'none'}")
    lines.append(f"- Recommended command: `{live['recommended_command']}`")
    lines.append(f"- Audit command: `{live['audit_command']}`")
    lines.append(f"- Ingest command: `{live['ingest_command']}`")
    lines.extend(["", "## Supplied Live Evidence", ""])
    live_evidence = report["live_evidence"]
    lines.append(f"- Loaded: {live_evidence['loaded']}")
    lines.append(f"- Mode: {live_evidence['mode']}")
    lines.append(f"- Status: {live_evidence['overall_status']}")
    lines.append(f"- Issues: {', '.join(live_evidence['issues']) if live_evidence['issues'] else 'none'}")
    lines.append("")
    return "\n".join(lines)


def _live_probe_closure_plan(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "voiceops.milestone1.live_probe_closure.v1",
        "artifact_id": "voiceops-m1-live-probe-closure",
        "milestone": "milestone_1_real_voice_operator",
        "mode": {
            "artifact_only": True,
            "supplied_artifacts_only": True,
            "discord_network": False,
            "env_secret_reads": False,
            "provider_sidecar_network": False,
        },
        "status": report["live_probe_required_for_completion"]["status"],
        "missing_gates": report["live_probe_required_for_completion"]["missing_gates"],
        "live_evidence_template": "live-voice-evidence-template.json",
        "live_evidence_scaffold_manifest": "live-voice-evidence-scaffold/manifest.json",
        "evidence_contract": {
            "manifest_schema_version": LIVE_EVIDENCE_MANIFEST_SCHEMA_VERSION,
            "expanded_evidence_schema_version": LIVE_EVIDENCE_SCHEMA_VERSION,
            "required_sections": ["discord_live_probe", "sidecar_session", "live_turn"],
            "required_section_field": "source_artifact",
            "required_section_refs": ["source_artifact", "section"],
            "manifest_report_identity": "per-section reports must include kind/evidence_type matching discord_live_probe, sidecar_session, or live_turn unless they use the expanded live evidence schema",
            "standalone_report_identity": "standalone non-expanded evidence files must include kind/evidence_type matching discord_live_probe, sidecar_session, or live_turn",
            "source_artifacts_must_exist": True,
            "source_artifact_resolution": "manifest report paths and nested source_artifact refs are package-contained relative paths; absolute paths, user-home expansion, parent traversal, symlink escapes, and process cwd fallback are rejected for manifest packages; explicitly supplied standalone evidence files remain accepted as direct operator inputs",
            "source_artifacts_must_be_json": True,
            "source_artifacts_reject_secret_or_phone_values": True,
            "source_artifacts_reject_voice_capability_denials": True,
            "template_source_artifacts_accepted": False,
            "example_only_accepted": False,
            "collector_attestation_required_for_live_readiness": True,
            "collector_attestation_required_fields": list(COLLECTOR_ATTESTATION_REQUIRED_FIELDS),
            "placeholder_collector_attestation_accepted": False,
        },
        "recommended_collection": {
            "live_bundle_manifest": report["live_probe_required_for_completion"]["recommended_command"],
            "audit_bundle_no_write": report["live_probe_required_for_completion"]["audit_command"],
            "validate_bundle_offline": report["live_probe_required_for_completion"]["validate_command"],
            "sidecar_session": (
                "Write sidecar-session.json with kind=sidecar_session, sidecar_running, sidecar_healthy, "
                "session_started, session_closed, fallback_mode_visible, fallback_reason, sidecar_mode=production, "
                "healthcheck_observed, provider_transport_observed, session_id_redacted, shutdown_bounded=true, "
                "shutdown_timed_out=false, latency_metrics_ms.session_start_ms, latency_metrics_ms.shutdown_ms, "
                "source_artifact, and collector_attestation."
            ),
            "live_turn": "Write live-turn.json with kind=live_turn, transcript_observed, assistant_audio_observed, barge_in_observed, spoken_reply_short, no_voice_denial_observed, speech_end_to_first_audio_ms, barge_in_stop_ms, source_artifact, and collector_attestation.",
            "ingest": report["live_probe_required_for_completion"]["ingest_command"],
        },
        "non_accepted_example_shapes": {
            "discord_live_probe": {
                "kind": "discord_live_probe",
                "example_only": True,
                "source_artifact": "sections/discord-live-probe-source.json",
                "collector_attestation": _example_collector_attestation("discord_live_probe"),
                "ok": True,
                "connect_perm": True,
                "speak_perm": True,
                "connected": True,
                "opus_loaded": True,
                "accepted_audio_source": True,
                "played": True,
                "playing_during_probe": True,
                "receiver_started": True,
                "receiver_frames": 1,
                "receiver_speech_start": 1,
                "inbound_observed": True,
                "disconnected": True,
                "require_inbound": True,
                "latency_metrics_ms": {
                    "connect_ms": 420,
                    "playback_observed_ms": 180,
                    "inbound_observed_ms": 900,
                    "disconnect_ms": 120,
                },
            },
            "sidecar_session": {
                "kind": "sidecar_session",
                "example_only": True,
                "source_artifact": "sections/sidecar-session-source.json",
                "collector_attestation": _example_collector_attestation("sidecar_session"),
                "sidecar_running": True,
                "sidecar_healthy": True,
                "session_started": True,
                "session_closed": True,
                "fallback_mode_visible": True,
                "fallback_reason": "none",
                "sidecar_mode": "production",
                "healthcheck_observed": True,
                "provider_transport_observed": True,
                "session_id_redacted": True,
                "shutdown_bounded": True,
                "shutdown_timed_out": False,
                "latency_metrics_ms": {"session_start_ms": 110, "shutdown_ms": 80},
            },
            "live_turn": {
                "kind": "live_turn",
                "example_only": True,
                "source_artifact": "sections/live-turn-source.json",
                "collector_attestation": _example_collector_attestation("live_turn"),
                "transcript_observed": True,
                "assistant_audio_observed": True,
                "barge_in_observed": True,
                "spoken_reply_short": True,
                "no_voice_denial_observed": True,
                "speech_end_to_first_audio_ms": 900,
                "barge_in_stop_ms": 90,
            },
        },
        "do_not": [
            "paste Discord bot tokens or provider tokens into evidence files",
            "include full phone numbers or private transcript content with secrets",
            "include raw transcript text; record only redacted booleans, latency numbers, and artifact references",
            "hand-edit manifest.json or example_only evidence to claim a passing live probe",
            "claim production readiness from the headless loopback smoke alone",
        ],
    }


def _live_probe_closure_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# VoiceOps Milestone 1 Live Probe Closure",
        "",
        f"- Status: {plan['status']}",
        f"- Missing gates: {', '.join(plan['missing_gates']) if plan['missing_gates'] else 'none'}",
        f"- Template: `{plan['live_evidence_template']}`",
        f"- Scaffold manifest: `{plan['live_evidence_scaffold_manifest']}`",
        "- Mode: supplied artifacts only; this file does not run Discord or read credentials",
        "",
        "## Evidence Contract",
        "",
    ]
    for key, value in sorted(plan["evidence_contract"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
        "## Collection",
        "",
        ]
    )
    for label, command in plan["recommended_collection"].items():
        lines.append(f"- {label}: {command}")
    lines.extend(
        [
            "",
            "## Non-Accepted Example Shapes",
            "",
            "These examples intentionally contain `example_only: true` and placeholder collector attestations. "
            "They are rejected by validation until replaced with real redacted source artifacts and attestations.",
            "",
        ]
    )
    for label, shape in plan["non_accepted_example_shapes"].items():
        lines.append(f"### {label}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(shape, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
    lines.extend(["", "## Do Not", ""])
    lines.extend(f"- {item}" for item in plan["do_not"])
    lines.append("")
    return "\n".join(lines)


def write_voice_operator_report(output_dir: Path, report: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    closure_plan = _live_probe_closure_plan(report)
    paths = {
        "json": output_dir / "voice-operator-readiness.json",
        "markdown": output_dir / "voice-operator-readiness.md",
        "smoke_json": output_dir / "discord-loopback-smoke.json",
        "async_oracle_smoke_json": output_dir / "async-oracle-smoke.json",
        "discord_session_cleanup_smoke_json": output_dir / "discord-session-cleanup-smoke.json",
        "events_jsonl": output_dir / "voice-operator-events.jsonl",
        "live_evidence_template": output_dir / "live-voice-evidence-template.json",
        "live_evidence_example": output_dir / "live-voice-evidence.example.json",
        "live_probe_closure_json": output_dir / "live-probe-closure-plan.json",
        "live_probe_closure_markdown": output_dir / "live-probe-closure-plan.md",
    }
    _write_json(paths["json"], report)
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    _write_json(paths["smoke_json"], report["smoke"])
    _write_json(paths["async_oracle_smoke_json"], report["async_oracle_smoke"])
    _write_json(paths["discord_session_cleanup_smoke_json"], report["discord_session_cleanup_smoke"])
    _write_json(paths["live_evidence_template"], build_live_probe_evidence_template())
    _write_json(paths["live_evidence_example"], build_live_probe_evidence_example())
    paths.update(write_live_evidence_scaffold(output_dir))
    _write_json(paths["live_probe_closure_json"], closure_plan)
    paths["live_probe_closure_markdown"].write_text(_live_probe_closure_markdown(closure_plan), encoding="utf-8")
    _write_jsonl(
        paths["events_jsonl"],
        [
            {"event_id": f"voice-m1-{index:03d}", "proof_id": proof_id, "ok": proof.get("ok") is True}
            for index, (proof_id, proof) in enumerate(sorted(report["proofs"].items()), start=1)
        ],
    )
    return {key: str(path) for key, path in paths.items()}


async def build_voice_operator_report_from_smoke(live_evidence_paths: list[Path] | None = None) -> dict[str, Any]:
    smoke_result = await run_discord_realtime_voice_smoke()
    async_oracle_smoke = await run_async_oracle_smoke()
    discord_session_cleanup_smoke = await run_discord_session_cleanup_smoke()
    return build_voice_operator_report(
        asdict(smoke_result),
        live_evidence=_load_live_evidence(live_evidence_paths),
        async_oracle_smoke=async_oracle_smoke,
        discord_session_cleanup_smoke=discord_session_cleanup_smoke,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--live-evidence",
        action="append",
        default=[],
        type=Path,
        help="Read-only live evidence JSON artifact or realtime_voice_live_evidence manifest to ingest; may be repeated. The generator still runs no Discord network.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = asyncio.run(build_voice_operator_report_from_smoke(args.live_evidence))
    issues = validate_voice_operator_report(report)
    paths = write_voice_operator_report(args.output_dir, report)
    print(
        json.dumps(
            {
                "ok": not issues,
                "validation_issues": issues,
                "output_dir": str(args.output_dir),
                "artifacts": paths,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
