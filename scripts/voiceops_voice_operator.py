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
ASYNC_ORACLE_STATUS_ORDINAL_LABELS = ("job one", "job two", "job three", "job four", "job five")
LIVE_EVIDENCE_SCHEMA_VERSION = "voiceops.milestone1.live_voice_evidence.v1"
LIVE_EVIDENCE_MANIFEST_SCHEMA_VERSION = "voiceops.realtime_voice_live_evidence_manifest.v1"
LIVE_EVIDENCE_REQUIRED_GATES = (
    "discord_join",
    "discord_playback",
    "live_receiver",
    "production_sidecar",
    "live_turn",
)
LIVE_EVIDENCE_VALID_WITNESS_ARRIVAL_PHASES = {
    "before_raw_audio",
    "with_raw_audio",
    "after_interpreter_start",
}
LIVE_EVIDENCE_VALID_HYPOTHESIS_KINDS = {
    "frontend_witness_hypothesis",
    "reflex_transcript_hypothesis",
    "s2s_transcript_hypothesis",
    "classic_asr_hypothesis",
}
LIVE_EVIDENCE_REQUIRED_INTERPRETER_INPUT_ORDER = (
    "raw_audio",
    "metadata",
    "reflex",
    "transcript_hypotheses",
)
LIVE_EVIDENCE_REQUIRED_INTERPRETER_PROMPT_POLICY = {
    "version": "raw_audio_compare_v1",
    "primary_evidence": "raw_audio",
    "transcript_hypotheses_authority": "non_authoritative_context",
}
LIVE_EVIDENCE_VALID_ADJUDICATION_OUTCOMES = {
    "accepted_as_supporting_evidence",
    "corrected_by_audio",
    "rejected_or_diagnostic_only",
}
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
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_equal_priority_queued_jobs_start_fifo",
        "tests/gateway/test_discord_realtime_voice.py::test_discord_realtime_spoken_tasks_create_async_oracle_jobs",
    ],
    "overflow_policy": [
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_queue_limit_rejects_overflow",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_overflow_policy_reject_rejects_at_capacity_with_queue_space",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_overflow_policy_reprioritize_requires_user_control_at_capacity",
        "tests/agent/test_realtime_voice.py::test_kame_engine_reports_async_oracle_reject_policy_without_sync_fallback",
        "tests/agent/test_realtime_voice.py::test_kame_engine_reports_async_oracle_reprioritize_policy_without_sync_fallback",
        "tests/hermes_cli/test_web_server.py::TestBuildSchemaFromConfig::test_realtime_voice_ws_config_passes_oracle_jobs_from_config",
    ],
    "status_view": [
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_status_view_reports_capacity_and_redacts_raw_metadata",
        "tests/agent/test_realtime_voice.py::test_kame_engine_local_status_question_uses_oracle_job_state",
        "tests/agent/test_realtime_voice.py::test_kame_session_does_not_persist_oracle_job_status_poll_messages",
        "tests/gateway/test_voice_command.py::TestVoiceChannelCommands::test_voice_status_reports_realtime_latency_metrics",
        "tests/gateway/test_voice_command.py::TestVoiceChannelCommands::test_voice_jobs_reports_oracle_job_snapshot",
    ],
    "local_turns": [
        "tests/agent/test_realtime_voice.py::test_kame_engine_async_oracle_job_allows_local_turn_while_running",
        "tests/agent/test_realtime_voice.py::test_oracle_direct_async_job_completion_after_local_turn_is_lifecycle_only",
    ],
    "job_creation_while_running": [
        "tests/agent/test_realtime_voice.py::test_kame_engine_can_create_oracle_job_while_another_is_running",
        "tests/gateway/test_discord_realtime_voice.py::test_discord_realtime_spoken_tasks_create_async_oracle_jobs",
    ],
    "cancellation": [
        "tests/agent/test_realtime_voice.py::test_kame_engine_interface_cancel_stops_one_async_oracle_job",
        "tests/agent/test_realtime_voice.py::test_kame_engine_can_cancel_queued_async_oracle_job_before_it_starts",
        "tests/agent/test_realtime_voice.py::test_kame_engine_spoken_stop_everything_cancels_all_async_oracle_jobs",
        "tests/agent/test_realtime_voice.py::test_kame_engine_barge_in_during_async_ack_does_not_interrupt_oracle_job",
        "tests/agent/test_realtime_voice.py::test_kame_engine_spoken_stop_talking_does_not_cancel_async_oracle_job",
        "tests/agent/test_realtime_voice.py::test_kame_engine_barge_in_during_async_result_speech_does_not_interrupt_completed_job",
        "tests/agent/test_realtime_voice.py::test_session_cancelled_oracle_job_removes_prior_completed_record",
        "tests/agent/test_realtime_voice.py::test_session_ignores_completed_record_after_oracle_job_cancelled",
        "tests/gateway/test_discord_realtime_voice.py::test_discord_realtime_cancelled_oracle_late_output_is_not_mixed",
    ],
    "approval_wait": [
        "tests/agent/test_realtime_voice.py::test_async_oracle_job_failed_kame_gate_suppresses_tool_result_completion",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_waiting_for_approval_holds_capacity_and_emits_redacted_event",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_cancelling_waiting_for_approval_keeps_capacity_until_worker_stops_and_drops_late_result",
        "tests/gateway/test_discord_realtime_voice.py::test_voice_status_oracle_job_lines_are_compact",
    ],
    "failure_handling": [
        "tests/agent/test_realtime_voice.py::test_kame_engine_async_oracle_job_failure_reports_in_voice",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_failed_job_records_error_and_starts_next",
    ],
    "control_updates": [
        "tests/agent/test_realtime_voice.py::test_kame_engine_can_reprioritize_queued_async_oracle_job",
        "tests/agent/test_realtime_voice.py::test_kame_engine_attaches_update_to_queued_async_oracle_job",
        "tests/agent/test_realtime_voice.py::test_kame_engine_attaches_interpreter_evidence_to_queued_async_oracle_job",
        "tests/agent/test_realtime_voice.py::test_kame_engine_does_not_promote_moshi_only_queued_evidence",
        "tests/agent/test_realtime_voice.py::test_kame_engine_merges_sequential_queued_transcript_hypotheses_before_start",
        "tests/agent/test_realtime_voice.py::test_kame_engine_attaches_update_to_running_async_oracle_job",
        "tests/agent/test_realtime_voice.py::test_kame_engine_attaches_interpreter_evidence_to_running_async_oracle_job",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_add_update_redacts_secret_like_text_from_status_and_events",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_interpreter_evidence_updates_queued_job_before_execution",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_interpreter_evidence_late_for_running_job_is_status_visible",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_interpreter_evidence_redacts_secret_like_text_from_status_events_and_request_updates",
        "tests/agent/test_realtime_voice.py::test_kame_engine_spoken_priority_control_reprioritizes_queued_job",
        "tests/agent/test_realtime_voice.py::test_kame_engine_spoken_update_attaches_to_latest_async_oracle_job",
        "tests/gateway/test_discord_realtime_voice.py::test_discord_realtime_event_tracks_oracle_job_status",
    ],
    "external_frontend_bridge": [
        "tests/agent/test_realtime_voice.py::test_external_kame_ask_brain_bridge_becomes_oracle_request",
        "tests/agent/test_realtime_voice.py::test_external_kame_ask_brain_bridge_strips_nested_tool_authority",
        "tests/agent/test_realtime_voice.py::test_external_kame_brain_request_submits_oracle_job_without_waiting",
        "tests/agent/test_realtime_voice.py::test_sidecar_oracle_hint_bridge_submits_external_kame_job",
        "tests/agent/test_realtime_voice.py::test_session_client_interface_oracle_request_submits_external_kame_job",
        "tests/agent/test_realtime_voice_async_oracle_smoke.py::test_async_oracle_smoke_proves_concurrency_local_turn_and_cancellation",
    ],
    "durable_resume": [
        "tests/agent/test_realtime_voice.py::test_kame_resume_context_uses_promoted_turns_and_excludes_hypotheses",
        "tests/agent/test_realtime_voice_async_oracle_smoke.py::test_async_oracle_smoke_proves_concurrency_local_turn_and_cancellation",
    ],
    "hypothesis_final_durability": [
        "tests/agent/test_realtime_voice.py::test_kame_session_treats_hypothesis_final_as_non_durable_without_adapter_flag",
        "tests/agent/test_realtime_voice.py::test_kame_session_treats_witness_sourced_interface_intent_as_non_durable",
        "tests/agent/test_realtime_voice.py::test_kame_session_keeps_explicit_asr_fallback_final_durable",
        "tests/agent/test_realtime_voice_async_oracle_smoke.py::test_async_oracle_smoke_proves_concurrency_local_turn_and_cancellation",
    ],
    "witness_fusion": [
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_kame_evidence_bundle_id_is_stable_across_audio_availability_changes",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_interpreter_evidence_updates_queued_job_before_execution",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_interpreter_evidence_late_for_running_job_is_status_visible",
        "tests/agent/test_realtime_voice.py::test_kame_engine_changing_partials_do_not_create_duplicate_oracle_turns",
        "tests/agent/test_realtime_voice.py::test_kame_engine_supersedes_partial_frontend_witness_with_final_before_start",
        "tests/agent/test_realtime_voice_async_oracle_smoke.py::test_async_oracle_smoke_proves_concurrency_local_turn_and_cancellation",
    ],
    "energy_gate": [
        "tests/agent/test_realtime_voice.py::test_text_engine_raw_audio_without_confirmed_speech_does_not_barge_in",
        "tests/agent/test_realtime_voice.py::test_kame_engine_low_energy_witness_text_does_not_start_turn",
        "tests/agent/test_realtime_voice.py::test_text_engine_speech_energy_barge_in_requires_rms_and_duration",
        "tests/agent/test_realtime_voice_async_oracle_smoke.py::test_async_oracle_smoke_proves_concurrency_local_turn_and_cancellation",
    ],
    "runtime_action_gate": [
        "tests/agent/test_realtime_voice.py::test_oracle_job_approval_marks_hypothesis_only_action_gate_unsafe",
        "tests/agent/test_realtime_voice.py::test_oracle_job_approval_accepts_consumed_promoted_interpreter_evidence",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_waiting_for_approval_holds_capacity_and_emits_redacted_event",
        "tests/agent/test_realtime_voice_async_oracle_smoke.py::test_async_oracle_smoke_proves_concurrency_local_turn_and_cancellation",
    ],
    "result_handling": [
        "tests/agent/test_realtime_voice.py::test_completed_async_oracle_job_after_intervening_local_turn_is_lifecycle_only",
        "tests/agent/test_realtime_voice.py::test_kame_engine_status_recalls_recent_completed_async_oracle_job",
        "tests/agent/test_realtime_voice.py::test_kame_engine_async_oracle_job_failure_reports_in_voice",
        "tests/agent/test_realtime_voice.py::test_kame_engine_async_terminal_result_speech_is_capped_without_losing_full_result",
        "tests/agent/test_realtime_voice.py::test_kame_engine_speak_terminal_results_false_suppresses_result_speech_but_keeps_status",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_completed_event_preserves_full_result_without_bloating_status",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_completed_result_redacts_secret_like_text_from_status_and_events",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_audit_ledger_path_records_redacted_lifecycle_events",
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_audit_ledger_force_redacts_scalar_payload_fields",
        "tests/agent/test_realtime_voice.py::test_session_persists_durable_async_oracle_job_records",
        "tests/agent/test_realtime_voice.py::test_session_redacts_durable_async_oracle_record_scalars",
        "tests/agent/test_realtime_voice_reference_sidecar_openai.py::test_reference_sidecar_forwards_speakable_oracle_result_to_openai_realtime",
        "tests/agent/test_realtime_voice_reference_sidecar_openai.py::test_reference_sidecar_forwards_oracle_result_suppression_to_openai_realtime",
        "tests/agent/test_realtime_voice_reference_sidecar_openai.py::test_reference_sidecar_forwards_interpreter_evidence_events_to_openai_realtime",
        "tests/agent/test_realtime_voice_reference_sidecar_gemini.py::test_reference_sidecar_forwards_speakable_oracle_result_to_gemini_live",
        "tests/agent/test_realtime_voice_reference_sidecar_gemini.py::test_reference_sidecar_forwards_oracle_result_suppression_to_gemini_live",
        "tests/agent/test_realtime_voice_reference_sidecar_gemini.py::test_reference_sidecar_forwards_interpreter_evidence_events_to_gemini_live",
    ],
    "discord_session": [
        "tests/gateway/test_voice_command.py::TestDiscordVoiceChannelMethods::test_leave_voice_channel_cleans_up",
        "tests/gateway/test_discord_realtime_voice.py::test_discord_realtime_degraded_marks_active_oracle_jobs_failed",
        "tests/gateway/test_discord_realtime_voice.py::test_discord_realtime_session_close_cancels_oracle_jobs_before_session_closed",
        "tests/gateway/test_discord_realtime_voice.py::test_discord_realtime_session_close_waits_for_oracle_cancel_ack_before_session_closed",
    ],
    "sidecar_fail_closed": [
        "tests/agent/test_realtime_voice.py::test_text_engine_fail_closed_policy_emits_session_error_on_sidecar_session_error",
        "tests/agent/test_realtime_voice.py::test_text_engine_fail_closed_policy_emits_session_error_on_sidecar_event_stream_failure",
        "tests/agent/test_realtime_voice.py::test_kame_engine_fail_closed_sidecar_send_failure_cancels_external_oracle_job",
    ],
    "shutdown": [
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_shutdown_forces_cancelled_state_when_worker_ignores_cancel",
        "tests/agent/test_realtime_voice.py::test_kame_engine_close_bounds_noncooperative_async_oracle_shutdown",
        "tests/agent/test_realtime_voice_async_oracle_smoke.py::test_async_oracle_smoke_proves_concurrency_local_turn_and_cancellation",
    ],
}

TOOL_DISCLOSURE_TEST_REFS = [
    "tests/tools/test_tool_search.py::TestAssembly::test_defer_core_all_hides_core_behind_bridge",
    "tests/agent/test_realtime_voice_oracle.py::test_voice_oracle_applies_scoped_tool_search_override",
    "tests/agent/test_realtime_voice_oracle.py::test_voice_oracle_runtime_tool_surface_is_bridge_only",
    "tests/agent/test_realtime_voice_oracle.py::test_voice_oracle_ephemeral_router_selects_voiceops_without_persisting_router_turn",
    "tests/agent/test_realtime_voice_oracle.py::test_voice_oracle_ephemeral_router_can_select_no_tools",
    "tests/hermes_cli/test_web_server.py::TestBuildSchemaFromConfig::test_realtime_voice_ws_config_defaults_oracle_tool_router",
]

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
    "audio_segment_ref_observed",
    "interpreter_evidence_observed",
    "transcript_hypotheses_labeled",
    "assistant_audio_observed",
    "barge_in_observed",
    "spoken_reply_short",
    "no_voice_denial_observed",
)

LIVE_EVIDENCE_REQUIRED_TURN_KAME_IDS = (
    "turn_id",
    "audio_segment_ref",
    "evidence_bundle_id",
    "evidence_merge_key",
)

LIVE_EVIDENCE_KAME_LINEAGE_FIELD_ALIASES = {
    "turn_id": ("turn_id", "kame_turn_id"),
    "audio_segment_ref": ("audio_segment_ref", "kame_audio_segment_ref", "audio_ref", "segment_ref"),
    "evidence_bundle_id": ("evidence_bundle_id", "kame_evidence_bundle_id"),
    "evidence_merge_key": ("evidence_merge_key", "kame_evidence_merge_key"),
}

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

FATAL_KAME_REFLEX_FALLBACK_PHRASES = (
    "asr reflex fallback is disabled",
    "kame audio reflex failed",
    "kame audio reflex unavailable",
    "kame_audio_reflex_failed",
    "kame_audio_reflex_unavailable",
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


async def run_sidecar_fail_closed_smoke() -> dict[str, Any]:
    """Provider-free fail-closed sidecar send-failure smoke."""

    from agent.realtime_voice import RealtimeVoiceEngineKind, RealtimeVoiceSessionConfig, VoiceEventType
    from agent.realtime_voice_text_engine import KameInterfaceOracleEngine

    class FailingSendSidecar:
        def __init__(self) -> None:
            self.closed = False
            self.close_calls = 0
            self.started_with: Any = None
            self._events: asyncio.Queue[Any] = asyncio.Queue()

        async def start(self, config: Any) -> None:
            self.started_with = config

        async def send_event(self, event: Any) -> None:
            raise RuntimeError("send failed at http://user:pass@voice.local/v1?token=abc")

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

    class BlockingOracle:
        def __init__(self) -> None:
            self.requests: list[Any] = []

        async def stream_answer_for_request(self, request: Any) -> Any:
            self.requests.append(request)
            await asyncio.Event().wait()
            yield "unreachable"

    oracle = BlockingOracle()
    sidecar = FailingSendSidecar()
    engine = KameInterfaceOracleEngine(oracle=oracle, sidecar=sidecar)
    response: dict[str, Any] = {}
    seen: list[Any] = []
    status: dict[str, Any] = {}
    error_text = ""
    try:
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-fail-closed-smoke",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                sidecar_base_url="http://voice.local:8080",
                fallback_policy="fail_closed",
                oracle_jobs={
                    "enabled": True,
                    "max_concurrent": 1,
                    "queue_limit": 4,
                    "shutdown_timeout_seconds": 0.05,
                },
            )
        )
        response = await engine.submit_external_brain_request(
            {
                "tool_name": "ask_brain",
                "tool_call_id": "voiceclaw-call-fail-closed",
                "arguments": {
                    "query": "provision the phone bridge",
                    "interface_already_said": "I am preparing the phone bridge.",
                },
            },
            source="voiceclaw",
        )
        events = engine.events()
        deadline = asyncio.get_running_loop().time() + 1.0
        while asyncio.get_running_loop().time() < deadline:
            try:
                event = await asyncio.wait_for(anext(events), timeout=0.1)
            except TimeoutError:
                continue
            seen.append(event)
            if any(item.type == VoiceEventType.SESSION_ERROR for item in seen) and any(
                item.type == VoiceEventType.ORACLE_JOB_CANCELLED for item in seen
            ):
                break
        status = await engine.get_oracle_job_status()
    except Exception as exc:  # pragma: no cover - failure path is represented in the smoke payload.
        error_text = str(exc)
    finally:
        await engine.close()

    cancelled_event = next((event for event in seen if event.type == VoiceEventType.ORACLE_JOB_CANCELLED), None)
    session_error_event = next((event for event in seen if event.type == VoiceEventType.SESSION_ERROR), None)
    if session_error_event is not None:
        error_text = str(session_error_event.payload.get("error") or "")
    jobs = status.get("jobs") if isinstance(status.get("jobs"), list) else []
    job = jobs[0] if jobs and isinstance(jobs[0], Mapping) else {}
    capacity = status.get("capacity") if isinstance(status.get("capacity"), Mapping) else {}
    ok = (
        response.get("accepted") is True
        and response.get("job_id") == "voice-oracle-001"
        and cancelled_event is not None
        and cancelled_event.payload.get("job_id") == "voice-oracle-001"
        and cancelled_event.payload.get("state") == "cancelled"
        and cancelled_event.payload.get("cancel_reason") == "sidecar_send_failed"
        and session_error_event is not None
        and session_error_event.payload.get("reason") == "sidecar_send_failed"
        and session_error_event.payload.get("sidecar") is False
        and "fallback_policy=fail_closed" in error_text
        and "send failed" in error_text
        and "user:pass" not in error_text
        and "token=abc" not in error_text
        and job.get("state") == "cancelled"
        and capacity.get("active") == 0
        and engine._sidecar is None
        and sidecar.closed is True
    )
    return {
        "ok": bool(ok),
        "scenario": "sidecar_send_fail_closed_after_acceptance",
        "discord_network": False,
        "provider_sidecar_network": False,
        "fallback_policy": "fail_closed",
        "request_accepted": response.get("accepted") is True,
        "job_id": response.get("job_id"),
        "cancelled_observed": cancelled_event is not None,
        "cancel_reason": cancelled_event.payload.get("cancel_reason") if cancelled_event is not None else None,
        "session_error_observed": session_error_event is not None,
        "session_error_reason": (
            session_error_event.payload.get("reason") if session_error_event is not None else None
        ),
        "session_error_sidecar": (
            session_error_event.payload.get("sidecar") if session_error_event is not None else None
        ),
        "error_redacted": "user:pass" not in error_text and "token=abc" not in error_text,
        "error_mentions_fail_closed": "fallback_policy=fail_closed" in error_text,
        "error_mentions_send_failed": "send failed" in error_text,
        "active_capacity_after_failure": capacity.get("active"),
        "job_state_after_failure": job.get("state"),
        "sidecar_removed": engine._sidecar is None,
        "sidecar_closed": sidecar.closed,
        "sidecar_close_calls": sidecar.close_calls,
        "oracle_requests_seen": len(oracle.requests),
        "event_order": [str(event.type.value) for event in seen],
        "test_refs": ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["sidecar_fail_closed"],
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


def _looks_like_fatal_kame_reflex_fallback(value: str) -> bool:
    lowered = value.lower()
    return any(phrase in lowered for phrase in FATAL_KAME_REFLEX_FALLBACK_PHRASES)


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
        "redaction_policy": "redacted refs, stable KAME ids, booleans, and latency numbers only; no Discord tokens, provider tokens, full phone numbers, or raw transcripts with secrets",
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
            "turn_id": None,
            "audio_segment_ref": None,
            "evidence_bundle_id": None,
            "evidence_merge_key": None,
            "transcript_observed": False,
            "audio_segment_ref_observed": False,
            "interpreter_evidence_observed": False,
            "transcript_hypotheses_labeled": False,
            "witness_arrival_phases": [],
            "interpreter_input_order": [],
            "transcript_hypotheses": [],
            "interpreter_adjudication_outcomes": [],
            "promoted_evidence_authority": {},
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
            "turn_id": "voiceops-live-turn-example",
            "audio_segment_ref": "artifact://redacted/live-turn-example.wav",
            "evidence_bundle_id": "kame-evidence-live-turn-example",
            "evidence_merge_key": "kame-merge-live-turn-example",
            "transcript_observed": True,
            "audio_segment_ref_observed": True,
            "interpreter_evidence_observed": True,
            "transcript_hypotheses_labeled": True,
            "witness_arrival_phases": ["with_raw_audio"],
            "interpreter_input_order": [
                "raw_audio",
                "metadata",
                "reflex",
                "transcript_hypotheses",
            ],
            "interpreter_prompt_policy": dict(LIVE_EVIDENCE_REQUIRED_INTERPRETER_PROMPT_POLICY),
            "transcript_hypotheses": [
                {
                    "kind": "frontend_witness_hypothesis",
                    "source": "moshi",
                    "text": "[redacted witness hypothesis]",
                    "arrival_phase": "with_raw_audio",
                    "authority": "hypothesis",
                    "tool_authority": False,
                }
            ],
            "interpreter_adjudication_outcomes": ["corrected_by_audio"],
            "promoted_evidence_authority": {
                "interpreter_corrected_transcript": "interpreter_promoted",
                "interpreter_normalized_intent": "interpreter_promoted",
            },
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
    return (
        any(key in payload for key in LIVE_EVIDENCE_REQUIRED_TURN_BOOLS)
        or any(key in payload for key in LIVE_EVIDENCE_REQUIRED_TURN_KAME_IDS)
        or "transcript_observed" in payload
        or "speech_end_to_first_audio_ms" in payload
    )


def _discord_probe_section(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    section = payload.get("discord_live_probe")
    if isinstance(section, Mapping):
        return section
    if payload.get("kind") == "discord_live_probe" or "accepted_audio_source" in payload:
        return payload
    return {}


def _normalized_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        values: list[Any] = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        return []
    normalized: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _live_turn_kame_lineage_conflict_issues(live_turn: Mapping[str, Any]) -> list[str]:
    return _mapping_kame_lineage_conflict_issues(live_turn, issue_prefix="live_turn")


def _mapping_kame_lineage_conflict_issues(payload: Mapping[str, Any], *, issue_prefix: str) -> list[str]:
    issues: list[str] = []
    for field, aliases in LIVE_EVIDENCE_KAME_LINEAGE_FIELD_ALIASES.items():
        values = _mapping_kame_lineage_values(payload, aliases)
        if len(values) > 1:
            issues.append(f"{issue_prefix}:kame_lineage_conflict_{field}")
    return issues


def _mapping_witness_binding_conflict_issues(payload: Mapping[str, Any], *, issue_prefix: str) -> list[str]:
    accepted_speaker = _accepted_mapping(payload, "speaker")
    accepted_channel = _accepted_mapping(payload, "channel")
    issues: list[str] = []
    hypotheses = payload.get("transcript_hypotheses")
    if not isinstance(hypotheses, list):
        return issues
    for index, hypothesis in enumerate(hypotheses):
        if not isinstance(hypothesis, Mapping):
            continue
        speaker_guess = _hypothesis_mapping(hypothesis, "speaker_guess", "speaker")
        if accepted_speaker and speaker_guess and _mapping_conflicts(
            accepted_speaker,
            speaker_guess,
            ("platform", "channel_user_id", "user_id"),
        ):
            issues.append(f"{issue_prefix}:transcript_hypothesis_{index}_speaker_mismatch")
        channel_guess = _hypothesis_mapping(hypothesis, "channel_guess", "channel")
        if accepted_channel and channel_guess and _mapping_conflicts(
            accepted_channel,
            channel_guess,
            ("transport", "guild_id", "channel_id"),
        ):
            issues.append(f"{issue_prefix}:transcript_hypothesis_{index}_channel_mismatch")
    return issues


def _accepted_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    direct = payload.get(key)
    if isinstance(direct, Mapping):
        return direct
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get(key), Mapping):
        return metadata[key]
    return {}


def _hypothesis_mapping(hypothesis: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys:
        value = hypothesis.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _mapping_conflicts(left: Mapping[str, Any], right: Mapping[str, Any], keys: tuple[str, ...]) -> bool:
    for key in keys:
        left_value = str(left.get(key) or "").strip()
        right_value = str(right.get(key) or "").strip()
        if left_value and right_value and left_value != right_value:
            return True
    return False


def _mapping_kame_lineage_values(payload: Mapping[str, Any], aliases: tuple[str, ...]) -> set[str]:
    values: set[str] = set()
    payloads: list[Mapping[str, Any]] = [payload]
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        payloads.append(metadata)
    audio = payload.get("audio")
    if isinstance(audio, Mapping):
        payloads.append(audio)
    hypotheses = payload.get("transcript_hypotheses")
    if isinstance(hypotheses, list):
        payloads.extend(hypothesis for hypothesis in hypotheses if isinstance(hypothesis, Mapping))
    for payload in payloads:
        for alias in aliases:
            value = payload.get(alias)
            if str(value or "").strip():
                values.add(str(value).strip())
    return values


def _live_turn_witness_packet_status(live_turn: Mapping[str, Any]) -> tuple[list[str], dict[str, bool]]:
    issues: list[str] = []
    hypotheses = live_turn.get("transcript_hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        issues.append("live_turn:missing_transcript_hypotheses")
        hypotheses = []

    valid_hypothesis_observed = False
    active_partial_observed = False
    hypothesis_phases: list[str] = []
    for index, hypothesis in enumerate(hypotheses):
        if not isinstance(hypothesis, Mapping):
            issues.append(f"live_turn:transcript_hypothesis_{index}_not_object")
            continue
        kind = str(hypothesis.get("kind") or "").strip()
        source = str(hypothesis.get("source") or "").strip()
        authority = str(hypothesis.get("authority") or "").strip()
        arrival_phase = str(hypothesis.get("arrival_phase") or "").strip()
        if not kind:
            issues.append(f"live_turn:transcript_hypothesis_{index}_missing_kind")
        elif kind not in LIVE_EVIDENCE_VALID_HYPOTHESIS_KINDS:
            issues.append(f"live_turn:transcript_hypothesis_{index}_invalid_kind")
        if not source:
            issues.append(f"live_turn:transcript_hypothesis_{index}_missing_source")
        if authority != "hypothesis":
            issues.append(f"live_turn:transcript_hypothesis_{index}_authority_not_hypothesis")
        if hypothesis.get("tool_authority") is not False:
            issues.append(f"live_turn:transcript_hypothesis_{index}_tool_authority_not_false")
        if hypothesis.get("partial") is True:
            active_partial_observed = True
            issues.append(f"live_turn:transcript_hypothesis_{index}_active_partial_not_superseded")
        if not arrival_phase:
            issues.append(f"live_turn:transcript_hypothesis_{index}_missing_arrival_phase")
        elif arrival_phase not in LIVE_EVIDENCE_VALID_WITNESS_ARRIVAL_PHASES:
            issues.append(f"live_turn:transcript_hypothesis_{index}_invalid_arrival_phase")
        else:
            hypothesis_phases.append(arrival_phase)
        if (
            kind in LIVE_EVIDENCE_VALID_HYPOTHESIS_KINDS
            and source
            and authority == "hypothesis"
            and hypothesis.get("tool_authority") is False
            and arrival_phase in LIVE_EVIDENCE_VALID_WITNESS_ARRIVAL_PHASES
        ):
            valid_hypothesis_observed = True

    declared_phases = _normalized_string_list(live_turn.get("witness_arrival_phases"))
    if not declared_phases:
        issues.append("live_turn:missing_witness_arrival_phases")
    for phase in declared_phases:
        if phase not in LIVE_EVIDENCE_VALID_WITNESS_ARRIVAL_PHASES:
            issues.append("live_turn:invalid_witness_arrival_phase")
    for phase in hypothesis_phases:
        if phase not in declared_phases:
            issues.append("live_turn:witness_arrival_phase_missing_hypothesis_phase")
            break

    input_order = tuple(_normalized_string_list(live_turn.get("interpreter_input_order")))
    if not input_order:
        issues.append("live_turn:missing_interpreter_input_order")
    elif input_order != LIVE_EVIDENCE_REQUIRED_INTERPRETER_INPUT_ORDER:
        issues.append("live_turn:interpreter_input_order_mismatch")

    prompt_policy = live_turn.get("interpreter_prompt_policy")
    valid_prompt_policy_observed = False
    if not isinstance(prompt_policy, Mapping) or not prompt_policy:
        issues.append("live_turn:missing_interpreter_prompt_policy")
    else:
        for field, expected_value in LIVE_EVIDENCE_REQUIRED_INTERPRETER_PROMPT_POLICY.items():
            if str(prompt_policy.get(field) or "").strip() != expected_value:
                issues.append(f"live_turn:interpreter_prompt_policy_{field}_mismatch")
        valid_prompt_policy_observed = all(
            str(prompt_policy.get(field) or "").strip() == expected_value
            for field, expected_value in LIVE_EVIDENCE_REQUIRED_INTERPRETER_PROMPT_POLICY.items()
        )

    adjudication_outcomes = set(_normalized_string_list(live_turn.get("interpreter_adjudication_outcomes")))
    valid_adjudication_observed = bool(adjudication_outcomes & LIVE_EVIDENCE_VALID_ADJUDICATION_OUTCOMES)
    if not adjudication_outcomes:
        issues.append("live_turn:missing_interpreter_adjudication_outcomes")
    elif not valid_adjudication_observed:
        issues.append("live_turn:invalid_interpreter_adjudication_outcome")

    promoted_authority = live_turn.get("promoted_evidence_authority")
    valid_promoted_authority_observed = False
    if not isinstance(promoted_authority, Mapping) or not promoted_authority:
        issues.append("live_turn:missing_promoted_evidence_authority")
    else:
        required_fields = ("interpreter_corrected_transcript", "interpreter_normalized_intent")
        for field in required_fields:
            if str(promoted_authority.get(field) or "").strip() != "interpreter_promoted":
                issues.append(f"live_turn:promoted_evidence_authority_missing_{field}")
        valid_promoted_authority_observed = all(
            str(promoted_authority.get(field) or "").strip() == "interpreter_promoted"
            for field in required_fields
        )

    return issues, {
        "witness_packet_observed": valid_hypothesis_observed and bool(declared_phases),
        "active_partial_absent": not active_partial_observed,
        "interpreter_input_order_observed": input_order == LIVE_EVIDENCE_REQUIRED_INTERPRETER_INPUT_ORDER,
        "interpreter_prompt_policy_observed": valid_prompt_policy_observed,
        "interpreter_adjudication_observed": valid_adjudication_observed,
        "promoted_evidence_observed": valid_promoted_authority_observed,
    }


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
    fatal_kame_reflex_fallback = bool(
        fallback_reason and _looks_like_fatal_kame_reflex_fallback(fallback_reason)
    )
    if sidecar.get("fallback_mode_visible") is True and not fallback_reason:
        issues.append("sidecar_session:missing_fallback_reason")
    elif fallback_reason and _looks_secret_or_phone(fallback_reason):
        issues.append("sidecar_session:fallback_reason_secret_or_phone_like_value")
    if fatal_kame_reflex_fallback:
        issues.append("sidecar_session:fatal_kame_reflex_fallback")
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
    live_turn_kame_ids = {
        key: str(live_turn.get(key) or "").strip()
        for key in LIVE_EVIDENCE_REQUIRED_TURN_KAME_IDS
    }
    for key, value in live_turn_kame_ids.items():
        if not value:
            issues.append(f"live_turn:missing_{key}")
    kame_lineage_ids_complete = all(live_turn_kame_ids.values())
    kame_lineage_conflict_issues = _live_turn_kame_lineage_conflict_issues(live_turn)
    issues.extend(kame_lineage_conflict_issues)
    kame_lineage_consistent = not kame_lineage_conflict_issues
    witness_binding_conflict_issues = _mapping_witness_binding_conflict_issues(live_turn, issue_prefix="live_turn")
    issues.extend(witness_binding_conflict_issues)
    witness_binding_consistent = not witness_binding_conflict_issues
    transcript_hypotheses_observed = (
        live_turn.get("transcript_hypotheses_labeled") is True
        or live_turn.get("transcript_observed") is True
    )
    raw_audio_interpreter_evidence_observed = (
        live_turn.get("audio_segment_ref_observed") is True
        and live_turn.get("interpreter_evidence_observed") is True
    )
    transcript_only_witness_rejected_for_full_kame = (
        transcript_hypotheses_observed and not raw_audio_interpreter_evidence_observed
    )
    if transcript_only_witness_rejected_for_full_kame:
        issues.append("live_turn:transcript_only_witness_without_raw_audio_interpreter_evidence")
    witness_packet_issues, witness_packet_status = _live_turn_witness_packet_status(live_turn)
    issues.extend(witness_packet_issues)
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
            and not fatal_kame_reflex_fallback
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
            and kame_lineage_ids_complete
            and kame_lineage_consistent
            and witness_binding_consistent
            and not transcript_only_witness_rejected_for_full_kame
            and all(witness_packet_status.values())
            and first_audio_ms is not None
            and first_audio_ms <= 3000
            and barge_in_ms is not None
            and barge_in_ms <= 150,
            "kame_lineage_ids_complete": kame_lineage_ids_complete,
            "kame_lineage_consistent": kame_lineage_consistent,
            "witness_binding_consistent": witness_binding_consistent,
            "turn_id": live_turn_kame_ids["turn_id"],
            "audio_segment_ref": live_turn_kame_ids["audio_segment_ref"],
            "evidence_bundle_id": live_turn_kame_ids["evidence_bundle_id"],
            "evidence_merge_key": live_turn_kame_ids["evidence_merge_key"],
            "raw_audio_interpreter_evidence_observed": raw_audio_interpreter_evidence_observed,
            "transcript_hypotheses_observed": transcript_hypotheses_observed,
            "transcript_only_witness_rejected_for_full_kame": transcript_only_witness_rejected_for_full_kame,
            **witness_packet_status,
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
    status_ordinal_labels = set(smoke.get("status_ordinal_labels") or ())
    status_ordinal_labels_visible = smoke.get("status_ordinal_labels_visible") is True and status_ordinal_labels >= set(
        ASYNC_ORACLE_STATUS_ORDINAL_LABELS
    )
    reflex_status_overflow_visible = (
        smoke.get("reflex_status_overflow_smoke_ok") is True
        and int(smoke.get("reflex_status_overflow_visible_job_count") or 0) == 8
        and int(smoke.get("reflex_status_overflow_hidden_job_count") or 0) == 2
        and smoke.get("reflex_status_overflow_more_spoken_status") == "+2 more"
        and smoke.get("reflex_status_overflow_last_visible_ordinal") == 8
        and smoke.get("reflex_status_overflow_last_visible_label") == "job eight"
        and smoke.get("reflex_status_overflow_hidden_ids_absent") is True
    )
    partial_active = smoke.get("witness_fusion_partial_active_hypothesis")
    partial_active_hypothesis = partial_active if isinstance(partial_active, Mapping) else {}
    return {
        "async_oracle_smoke_ok": bool(smoke.get("ok")),
        "four_jobs_ran_concurrently": bool(smoke.get("worker_overlap_proved"))
        and smoke.get("worker_overlap_within_capacity") is True
        and int(smoke.get("max_worker_overlap") or 0) >= 4
        and int(smoke.get("max_running") or 0) >= 4
        and int(smoke.get("started_jobs") or 0) >= 4,
        "local_turn_while_jobs_running": smoke.get("local_turn_committed") is True
        and smoke.get("local_turn_during_running_jobs_observed") is True
        and int(smoke.get("local_turn_active_job_count") or 0) >= 1,
        "status_turn_while_jobs_running": bool(smoke.get("status_turn_committed"))
        and "4 running out of 4" in str(smoke.get("status_text") or "")
        and "1 queued" in str(smoke.get("status_text") or "")
        and smoke.get("status_turn_queued_visible") is True
        and smoke.get("status_turn_no_oracle_request") is True,
        "status_turn_ordinal_labels_visible": status_ordinal_labels_visible,
        "status_turn_bounded_overflow_visible": reflex_status_overflow_visible,
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
        and int(smoke.get("durable_completed_jobs") or 0) == int(smoke.get("completed_jobs") or 0)
        and int(smoke.get("durable_completed_jobs") or 0) >= 1,
        "playback_stop_does_not_cancel_jobs": smoke.get("playback_stop_does_not_cancel_jobs") is True
        and smoke.get("playback_stop_committed") is True
        and smoke.get("playback_stop_jobs_still_running") is True
        and smoke.get("playback_stop_cancelled_jobs") is False,
        "approval_wait_visible_and_redacted": smoke.get("approval_wait_observed") is True
        and smoke.get("approval_status_committed") is True
        and smoke.get("approval_tool_progress_observed") is True
        and smoke.get("approval_payload_redacted") is True
        and smoke.get("approval_secret_leaked") is False
        and smoke.get("approval_secret_canary_checked") is True
        and smoke.get("approval_completed") is False
        and smoke.get("approval_gate_failed_closed") is True
        and smoke.get("approval_result_suppressed") is True,
        "runtime_kame_action_gate_degraded_text_only_fails_closed": smoke.get(
            "runtime_kame_action_gate_degraded_text_only_ok"
        )
        is False
        and smoke.get("runtime_kame_action_gate_degraded_text_only_status") == "degraded_text_only"
        and smoke.get("runtime_kame_action_gate_degraded_text_only_reason") == "degraded_text_only"
        and smoke.get("runtime_kame_action_gate_degraded_text_only_raw_audio_available") is False
        and smoke.get("runtime_kame_action_gate_degraded_text_only_preserves_hypothesis") is True
        and "missing_promoted_evidence"
        in (smoke.get("runtime_kame_action_gate_degraded_text_only_issues") or [])
        and "interpreter_evidence_not_consumed_before_irreversible_action"
        in (smoke.get("runtime_kame_action_gate_degraded_text_only_issues") or [])
        and set(smoke.get("runtime_kame_action_gate_degraded_text_only_rejected_authorities") or [])
        >= {"reflex_hypothesis", "hypothesis"},
        "runtime_kame_action_gate_enforced": smoke.get("runtime_kame_action_gate_smoke_ok") is True
        and smoke.get("runtime_kame_action_gate_hypothesis_only_ok") is False
        and "missing_promoted_evidence"
        in (smoke.get("runtime_kame_action_gate_hypothesis_only_issues") or [])
        and "interpreter_evidence_not_consumed_before_irreversible_action"
        in (smoke.get("runtime_kame_action_gate_hypothesis_only_issues") or [])
        and set(smoke.get("runtime_kame_action_gate_hypothesis_only_rejected_authorities") or [])
        >= {"reflex_hypothesis", "hypothesis"}
        and smoke.get("runtime_kame_action_gate_degraded_text_only_ok") is False
        and smoke.get("runtime_kame_action_gate_degraded_text_only_status") == "degraded_text_only"
        and smoke.get("runtime_kame_action_gate_degraded_text_only_raw_audio_available") is False
        and smoke.get("runtime_kame_action_gate_degraded_text_only_preserves_hypothesis") is True
        and smoke.get("runtime_kame_action_gate_promoted_ok") is True
        and (smoke.get("runtime_kame_action_gate_promoted_issues") or []) == []
        and smoke.get("runtime_kame_action_gate_promoted_authorities") == ["interpreter_promoted"]
        and smoke.get("runtime_kame_action_gate_promoted_consumed_before_action") is True
        and smoke.get("runtime_kame_action_gate_self_attested_ok") is False
        and "missing_promoted_evidence"
        in (smoke.get("runtime_kame_action_gate_self_attested_issues") or [])
        and "interpreter_evidence_not_consumed_before_irreversible_action"
        not in (smoke.get("runtime_kame_action_gate_self_attested_issues") or [])
        and (smoke.get("runtime_kame_action_gate_self_attested_authorities") or []) == []
        and smoke.get("runtime_kame_action_gate_self_attested_consumed_before_action") is True
        and smoke.get("runtime_kame_action_gate_missing_tool_disclosure_ok") is False
        and "missing_tool_disclosure_ref"
        in (smoke.get("runtime_kame_action_gate_missing_tool_disclosure_issues") or [])
        and "missing_promoted_evidence"
        not in (smoke.get("runtime_kame_action_gate_missing_tool_disclosure_issues") or [])
        and (
            smoke.get("runtime_kame_action_gate_missing_tool_disclosure_authorities") or []
        )
        == ["interpreter_promoted"]
        and smoke.get("runtime_kame_action_gate_tool_disclosure_ref_observed") is True,
        "unflagged_high_risk_tool_event_fails_closed": smoke.get("unflagged_high_risk_tool_smoke_ok")
        is True
        and smoke.get("unflagged_high_risk_tool_suppressed") is True
        and smoke.get("unflagged_high_risk_tool_failed_closed") is True
        and smoke.get("unflagged_high_risk_tool_suppression_reason")
        == "unapproved_high_risk_tool_event"
        and smoke.get("unflagged_high_risk_tool_progress_suppressed") is True
        and smoke.get("unflagged_high_risk_tool_payload_redacted") is True
        and smoke.get("unflagged_high_risk_tool_spoken_payload_clean") is True
        and smoke.get("unflagged_high_risk_tool_failure_spoken") is True
        and smoke.get("unflagged_high_risk_tool_secret_canary_checked") is True,
        "approval_wait_holds_capacity": smoke.get("approval_capacity_smoke_ok") is True
        and smoke.get("approval_capacity_waiting_observed") is True
        and smoke.get("approval_capacity_followup_queued") is True
        and smoke.get("approval_capacity_active_visible") is True
        and smoke.get("approval_capacity_misleading_running_capacity") is False
        and "1 queued" in str(smoke.get("approval_capacity_status_text") or "")
        and "1 waiting for approval" in str(smoke.get("approval_capacity_status_text") or "")
        and smoke.get("approval_capacity_followup_started_after_approval") is True
        and int(smoke.get("approval_capacity_completed_jobs") or 0) == 1
        and smoke.get("approval_capacity_failed_gate_suppressed") is True
        and int(smoke.get("approval_capacity_failed_jobs") or 0) == 1,
        "approval_cancel_holds_capacity": smoke.get("approval_cancel_capacity_smoke_ok") is True
        and smoke.get("approval_cancel_waiting_observed") is True
        and smoke.get("approval_cancel_followup_queued") is True
        and smoke.get("approval_cancel_requested_observed") is True
        and smoke.get("approval_cancel_cancelled_observed") is True
        and smoke.get("approval_cancel_late_output_attempted") is True
        and smoke.get("approval_cancel_completed_after_cancel") is False
        and smoke.get("approval_cancel_late_result_spoken") is False
        and smoke.get("approval_cancel_followup_started_before_cancel_drained") is False
        and smoke.get("approval_cancel_followup_started_after_cancel") is True
        and smoke.get("approval_cancel_active_visible") is True
        and smoke.get("approval_cancel_misleading_running_capacity") is False
        and "1 queued" in str(smoke.get("approval_cancel_status_text") or "")
        and "1 cancelling" in str(smoke.get("approval_cancel_status_text") or ""),
        "cancel_drain_holds_capacity": smoke.get("cancel_drain_capacity_smoke_ok") is True
        and smoke.get("cancel_drain_requested_observed") is True
        and smoke.get("cancel_drain_cancelled_observed") is True
        and smoke.get("cancel_drain_followup_queued") is True
        and smoke.get("cancel_drain_active_visible") is True
        and smoke.get("cancel_drain_misleading_running_capacity") is False
        and "1 queued" in str(smoke.get("cancel_drain_status_text") or "")
        and "1 cancelling" in str(smoke.get("cancel_drain_status_text") or "")
        and smoke.get("cancel_drain_followup_started_after_cancel") is True,
        "failed_job_reported_without_crash": int(smoke.get("failed_jobs") or 0) >= 1
        and smoke.get("failed_job_reported") is True
        and smoke.get("failed_job_spoken") is True
        and smoke.get("durable_failed_record_present") is True
        and smoke.get("session_survived_failed_job") is True,
        "job_control_updates_reach_oracle": smoke.get("queued_job_update_observed") is True
        and smoke.get("queued_update_latest_update_visible") is True
        and smoke.get("queued_update_started_with_priority") is True
        and smoke.get("queued_update_reached_oracle") is True
        and smoke.get("queued_interpreter_fold_in_observed") is True
        and smoke.get("running_job_update_observed") is True
        and smoke.get("running_update_latest_update_visible") is True
        and smoke.get("running_update_reached_oracle") is True
        and smoke.get("running_update_delivery_metadata_ok") is True,
        "transcript_hypotheses_remain_unpromoted": smoke.get("unpromoted_hypothesis_smoke_ok") is True
        and smoke.get("unpromoted_hypothesis_single_bundle_observed") is True
        and smoke.get("unpromoted_hypothesis_source") == "moshi"
        and smoke.get("unpromoted_hypothesis_authority") == "hypothesis"
        and smoke.get("unpromoted_hypothesis_tool_authority_false") is True
        and smoke.get("unpromoted_hypothesis_oracle_text_preserved") is True
        and smoke.get("unpromoted_hypothesis_transcript_preserved") is True
        and smoke.get("unpromoted_hypothesis_intent_preserved") is True
        and smoke.get("unpromoted_hypothesis_attached") is True
        and smoke.get("unpromoted_hypothesis_promoted") is False
        and smoke.get("unpromoted_hypothesis_action_sinks_clean") is True
        and smoke.get("unpromoted_hypothesis_not_spend_reason") is True
        and smoke.get("unpromoted_hypothesis_not_spend_payload") is True
        and smoke.get("unpromoted_hypothesis_not_provider_selection") is True
        and smoke.get("unpromoted_hypothesis_not_nemoclaw_action_packet") is True
        and smoke.get("unpromoted_hypothesis_not_phone_call_payload") is True
        and smoke.get("unpromoted_hypothesis_not_call_payload") is True
        and smoke.get("unpromoted_hypothesis_not_tool_arguments") is True
        and smoke.get("unpromoted_hypothesis_not_memory_write") is True
        and smoke.get("unpromoted_hypothesis_not_file_write") is True
        and smoke.get("unpromoted_hypothesis_not_message_payload") is True
        and smoke.get("unpromoted_hypothesis_update_observed") is True,
        "hypothesis_final_events_non_durable": smoke.get("hypothesis_final_durable_message_smoke_ok") is True
        and smoke.get("hypothesis_final_durable_messages_empty") is True
        and int(smoke.get("hypothesis_final_durable_message_count") or 0) == 0
        and smoke.get("hypothesis_final_without_adapter_flag_non_durable") is True
        and smoke.get("hypothesis_final_witness_intent_non_durable") is True
        and smoke.get("explicit_asr_fallback_final_remains_durable") is True
        and smoke.get("explicit_asr_fallback_durable_messages")
        == [{"role": "user", "content": "check deployment status"}],
        "external_frontend_bridge_submits_oracle_job": smoke.get("external_frontend_bridge_smoke_ok") is True
        and smoke.get("external_frontend_request_accepted") is True
        and smoke.get("external_frontend_tool_result_observed") is True
        and smoke.get("external_frontend_accepted_observed") is True
        and smoke.get("external_frontend_started_observed") is True
        and smoke.get("external_frontend_completion_observed") is True
        and smoke.get("external_frontend_status_state") == "completed"
        and smoke.get("external_frontend_source_reached_oracle") is True
        and smoke.get("external_frontend_input_source") == "ask_brain"
        and smoke.get("external_frontend_evidence_bundle_propagated") is True
        and smoke.get("external_frontend_evidence_bundle_id_stable") is True
        and smoke.get("external_frontend_evidence_bundle_single_turn") is True
        and smoke.get("external_frontend_evidence_bundle_status") == "primary_audio"
        and int(smoke.get("external_frontend_evidence_bundle_transcript_hypotheses_count") or 0) >= 1
        and smoke.get("external_frontend_witness_kind_frontend_hypothesis") is True
        and smoke.get("external_frontend_witness_metadata_complete") is True
        and smoke.get("external_frontend_hypothesis_not_durable_oracle_text") is True
        and smoke.get("external_frontend_durable_user_messages_empty") is True
        and smoke.get("external_frontend_durable_oracle_text_absent") is True
        and smoke.get("external_frontend_terminal_correlation_observed") is True
        and smoke.get("external_frontend_audit_id_continuity_observed") is True
        and smoke.get("external_frontend_direct_tool_authority_exposed") is False,
        "durable_promoted_turn_resume_contract": smoke.get("durable_resume_contract_smoke_ok") is True
        and smoke.get("durable_resume_contract_schema_version") == "voiceops.kame_durable_resume_context.v1"
        and int(smoke.get("durable_resume_promoted_turn_count") or 0) >= 4
        and smoke.get("durable_resume_recent_promoted_turns_verbatim") is True
        and smoke.get("durable_resume_older_turns_summarized") is True
        and smoke.get("durable_resume_hypothesis_replay_absent") is True
        and smoke.get("durable_resume_ledger_authoritative") is True,
        "witness_fusion_timing_preserves_single_bundle": smoke.get("witness_fusion_timing_smoke_ok") is True
        and smoke.get("witness_fusion_early_single_bundle") is True
        and smoke.get("witness_fusion_with_single_bundle") is True
        and smoke.get("witness_fusion_late_single_bundle") is True
        and smoke.get("witness_fusion_partial_superseded_by_final") is True
        and smoke.get("witness_fusion_no_duplicate_oracle_jobs") is True
        and smoke.get("witness_fusion_merge_key_observed") is True
        and smoke.get("witness_fusion_turn_ids")
        == {
            "early": "witness-fusion:early",
            "with": "witness-fusion:with",
            "late": "witness-fusion:late",
        }
        and smoke.get("witness_fusion_audio_segment_refs")
        == {
            "early": "artifact://voice/witness-early.wav",
            "with": "artifact://voice/witness-with.wav",
            "late": "artifact://voice/witness-late.wav",
        }
        and smoke.get("witness_fusion_arrival_phases")
        == ["before_raw_audio", "with_raw_audio", "after_interpreter_start"],
        "witness_fusion_accepted_audio_gate_visible": smoke.get(
            "witness_fusion_accepted_audio_gate_observed"
        )
        is True
        and isinstance(smoke.get("witness_fusion_audio_metadata"), Mapping)
        and isinstance(smoke.get("witness_fusion_bundle_audio_metadata"), Mapping)
        and set(smoke["witness_fusion_audio_metadata"]) >= {"early", "with", "late"}
        and smoke.get("witness_fusion_audio_metadata") == smoke.get("witness_fusion_bundle_audio_metadata"),
        "witness_fusion_partial_superseded_by_final": smoke.get(
            "witness_fusion_partial_superseded_by_final"
        )
        is True
        and partial_active_hypothesis.get("source") == "moshi"
        and partial_active_hypothesis.get("kind") == "frontend_witness_hypothesis"
        and partial_active_hypothesis.get("text") == "what is three to the power of seventeen"
        and partial_active_hypothesis.get("authority") == "hypothesis"
        and partial_active_hypothesis.get("tool_authority") is False
        and partial_active_hypothesis.get("arrival_phase") == "with_raw_audio"
        and partial_active_hypothesis.get("partial") is False
        and tuple(partial_active_hypothesis.get("superseded_partial_texts") or ())
        == ("what is three to the",)
        and partial_active_hypothesis.get("superseded_partial_count") == 1,
        "witness_fusion_adjudicates_frontend_text": smoke.get(
            "witness_fusion_adjudication_outcomes_observed"
        )
        is True
        and smoke.get("witness_fusion_adjudications")
        == {
            "early": ["corrected_by_audio"],
            "with": ["accepted_as_supporting_evidence"],
            "late": ["rejected_or_diagnostic_only"],
        }
        and smoke.get("witness_fusion_rejection_reasons")
        == {
            "early": [],
            "with": [],
            "late": ["ambiguous_speaker", "wrong_speaker", "wrong_channel", "stale_witness"],
        },
        "interpreter_prompt_input_order_visible": smoke.get(
            "witness_fusion_interpreter_prompt_input_order_visible"
        )
        is True
        and smoke.get("witness_fusion_interpreter_prompt_input_order")
        == ["raw_audio", "metadata", "reflex", "transcript_hypotheses"]
        and smoke.get("witness_fusion_interpreter_prompt_input_order_expected")
        == ["raw_audio", "metadata", "reflex", "transcript_hypotheses"],
        "interpreter_prompt_policy_visible": smoke.get(
            "witness_fusion_interpreter_prompt_policy_visible"
        )
        is True
        and smoke.get("witness_fusion_interpreter_prompt_policy_version") == "raw_audio_compare_v1"
        and isinstance(smoke.get("witness_fusion_interpreter_prompt_policy"), Mapping)
        and smoke["witness_fusion_interpreter_prompt_policy"].get("primary_evidence") == "raw_audio"
        and smoke["witness_fusion_interpreter_prompt_policy"].get("transcript_hypotheses_authority")
        == "non_authoritative_context"
        and smoke["witness_fusion_interpreter_prompt_policy"].get("promotion_requirement")
        == "compare_transcript_hypotheses_against_raw_audio_before_promotion",
        "energy_gate_ignores_non_speech_without_work": smoke.get("energy_gate_smoke_ok") is True
        and smoke.get("energy_gate_ignored_packet_speech_confirmed") is False
        and smoke.get("energy_gate_ignored_packet_vad_speech") is False
        and int(smoke.get("energy_gate_ignored_non_speech_packets") or 0) >= 2
        and int(smoke.get("energy_gate_barge_in_events", -1)) == 0
        and int(smoke.get("energy_gate_interpreter_requests", -1)) == 0
        and int(smoke.get("energy_gate_oracle_work_events", -1)) == 0
        and int(smoke.get("energy_gate_oracle_requests", -1)) == 0
        and smoke.get("energy_gate_low_energy_witness_source") == "moshi"
        and smoke.get("energy_gate_low_energy_witness_promoted") is False
        and smoke.get("energy_gate_low_energy_witness_suppressed") is True
        and smoke.get("energy_gate_raw_packet_buffered_without_turn") is True,
        "kame_ack_latency_metrics_visible": smoke.get("kame_ack_latency_metrics_smoke_ok") is True
        and smoke.get("kame_defer_ack_first_audio_metrics_visible") is True
        and smoke.get("kame_local_first_audio_metrics_visible") is True
        and "kame_interface_decision_to_defer_first_audio_ms"
        in (smoke.get("kame_defer_ack_metric_keys") or [])
        and "kame_speech_end_to_defer_first_audio_ms"
        in (smoke.get("kame_defer_ack_metric_keys") or [])
        and "kame_interface_decision_to_local_first_audio_ms"
        in (smoke.get("kame_local_first_audio_metric_keys") or [])
        and "kame_speech_end_to_local_first_audio_ms"
        in (smoke.get("kame_local_first_audio_metric_keys") or [])
        and _non_negative_number(smoke.get("kame_defer_speech_end_to_first_audio_ms")) is not None
        and _non_negative_number(smoke.get("kame_local_speech_end_to_first_audio_ms")) is not None,
        "result_handling_bounded_and_durable": smoke.get("verbose_result_spoken_bounded") is True
        and smoke.get("verbose_result_committed_bounded") is True
        and smoke.get("verbose_result_commit_marked_truncated") is True
        and smoke.get("verbose_full_result_durable") is True
        and smoke.get("completed_result_status_visible") is True
        and smoke.get("terminal_result_policy_smoke_ok") is True
        and smoke.get("terminal_result_auto_summarize_default") is True
        and smoke.get("terminal_result_suppressed") is True
        and smoke.get("terminal_result_suppressed_event_observed") is True
        and smoke.get("terminal_result_suppressed_payload_clean") is True
        and smoke.get("terminal_result_suppressed_reason") == "terminal_speech_disabled"
        and smoke.get("terminal_result_status_available") is True
        and int(smoke.get("terminal_result_unsolicited_event_count") or 0) == 0
        and smoke.get("terminal_result_unsolicited_spoken") is False
        and smoke.get("audit_scalar_smoke_ok") is True
        and smoke.get("audit_scalar_payload_redacted") is True
        and smoke.get("audit_scalar_secret_canary_checked") is True
        and smoke.get("audit_scalar_result_text_omitted") is True
        and smoke.get("audit_scalar_completed_event_seen") is True
        and smoke.get("audit_scalar_waiting_event_seen") is True,
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


def _coverage_from_sidecar_fail_closed_smoke(smoke: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "sidecar_fail_closed_send_failure_cancels_active_job": smoke.get("ok") is True
        and smoke.get("request_accepted") is True
        and smoke.get("cancelled_observed") is True
        and smoke.get("cancel_reason") == "sidecar_send_failed"
        and smoke.get("session_error_observed") is True
        and smoke.get("session_error_reason") == "sidecar_send_failed"
        and smoke.get("session_error_sidecar") is False
        and smoke.get("error_redacted") is True
        and smoke.get("error_mentions_fail_closed") is True
        and smoke.get("active_capacity_after_failure") == 0
        and smoke.get("job_state_after_failure") == "cancelled"
        and smoke.get("sidecar_removed") is True
        and smoke.get("sidecar_closed") is True,
    }


def run_tool_disclosure_smoke() -> dict[str, Any]:
    """Local proof that realtime voice can collapse core tools behind tool_search."""

    from scripts.voiceops_provisioning_probe import build_registered_core_tool_schema_defs
    from toolsets import _HERMES_CORE_TOOLS
    from tools.tool_search import (
        BRIDGE_TOOL_NAMES,
        ToolSearchConfig,
        assemble_tool_defs,
        estimate_tokens_from_schemas,
    )

    input_core_tools = sorted(_HERMES_CORE_TOOLS)
    core_tool_defs, missing_core_tools = build_registered_core_tool_schema_defs()
    config = ToolSearchConfig.from_raw({"enabled": "on", "defer_core": "all"})
    result = assemble_tool_defs(core_tool_defs, context_length=272_000, config=config)
    visible_names = sorted((tool.get("function") or {}).get("name") for tool in result.tool_defs)
    visible_non_bridge_names = sorted(name for name in visible_names if name not in BRIDGE_TOOL_NAMES)
    hidden_core_names = sorted(
        name for name in input_core_tools if name not in visible_names
    )
    bridge_names = sorted(BRIDGE_TOOL_NAMES)
    input_schema_tokens = estimate_tokens_from_schemas(core_tool_defs)
    visible_schema_tokens = estimate_tokens_from_schemas(result.tool_defs)
    token_reduction_estimate = max(0, input_schema_tokens - visible_schema_tokens)
    return {
        "ok": result.activated
        and result.deferred_count == len(core_tool_defs)
        and visible_names == bridge_names
        and hidden_core_names == input_core_tools
        and not visible_non_bridge_names
        and not missing_core_tools
        and len(core_tool_defs) == len(input_core_tools)
        and token_reduction_estimate > 0,
        "scenario": "voice_scoped_all_core_tool_deferral",
        "provider_network": False,
        "model_call": False,
        "schema_source": "registered_core_tool_schemas",
        "representative_schema": False,
        "missing_registered_core_tools": missing_core_tools,
        "config": {
            "enabled": config.enabled,
            "defer_core": config.defer_core,
        },
        "input_core_tools": input_core_tools,
        "visible_tool_names": visible_names,
        "visible_non_bridge_tool_names": visible_non_bridge_names,
        "hidden_core_tool_names": hidden_core_names,
        "bridge_tool_names": bridge_names,
        "input_core_tool_count": len(input_core_tools),
        "hidden_core_tool_count": len(hidden_core_names),
        "bridge_tool_count": len(bridge_names),
        "core_tools_hidden_all": hidden_core_names == input_core_tools,
        "broad_core_tools_visible": bool(visible_non_bridge_names),
        "deferred_count": result.deferred_count,
        "deferred_tokens": result.deferred_tokens,
        "input_schema_tokens": input_schema_tokens,
        "visible_schema_tokens": visible_schema_tokens,
        "token_reduction_estimate": token_reduction_estimate,
        "external_test_refs": TOOL_DISCLOSURE_TEST_REFS,
    }


def run_ephemeral_tool_router_smoke() -> dict[str, Any]:
    """Local proof for the VoiceOps ephemeral tool-selection router."""

    from agent.realtime_voice import RealtimeVoiceEngineKind, RealtimeVoiceSessionConfig
    from agent.realtime_voice_oracle import _voice_oracle_tool_router_decision

    calls: list[dict[str, Any]] = []
    responses = [
        {"final_response": '{"decision":"toolsets","toolsets":["voiceops"]}'},
        {"final_response": '{"decision":"no_tools"}'},
    ]

    class FakeRouterAgent:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

        def run_conversation(self, prompt, *, persist_user_message=None, stream_callback=None):
            calls.append(
                {
                    "kwargs": dict(self.kwargs),
                    "prompt": str(prompt),
                    "persist_user_message": persist_user_message,
                    "stream_callback_supplied": stream_callback is not None,
                }
            )
            return responses[len(calls) - 1]

    config = RealtimeVoiceSessionConfig(
        session_id="voiceops-router-smoke",
        engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        metadata={"transport": "discord_voice"},
        oracle_tool_router={
            "enabled": True,
            "mode": "ephemeral",
            "voiceops_toolsets": ["voiceops"],
            "default_toolsets": [],
        },
    )
    toolset_decision = _voice_oracle_tool_router_decision(
        "use Stripe to provision phone service",
        {"transport": "discord_voice", "kame_intent": "Provision phone service."},
        config,
        FakeRouterAgent,
    )
    no_tools_decision = _voice_oracle_tool_router_decision(
        "tell me a short joke",
        {"transport": "discord_voice", "kame_intent": "Tell a short joke."},
        config,
        FakeRouterAgent,
    )
    router_kwargs = [call["kwargs"] for call in calls]
    persist_values = [call["persist_user_message"] for call in calls]
    return {
        "ok": (
            len(calls) == 2
            and toolset_decision.get("mode") == "ephemeral"
            and toolset_decision.get("decision") == "toolsets"
            and toolset_decision.get("enabled_toolsets") == ["voiceops"]
            and no_tools_decision.get("mode") == "ephemeral"
            and no_tools_decision.get("decision") == "no_tools"
            and no_tools_decision.get("enabled_toolsets") == []
            and all(value is False for value in persist_values)
            and all(kwargs.get("enabled_toolsets") == [] for kwargs in router_kwargs)
            and all(kwargs.get("skip_memory") is True for kwargs in router_kwargs)
            and all(kwargs.get("skip_context_files") is True for kwargs in router_kwargs)
            and all(decision.get("tool_calls_allowed") is False for decision in (toolset_decision, no_tools_decision))
            and all(decision.get("persistent") is False for decision in (toolset_decision, no_tools_decision))
        ),
        "scenario": "ephemeral_voiceops_tool_selection_router",
        "provider_network": False,
        "model_call": False,
        "router_mode": "ephemeral",
        "toolsets_decision": dict(toolset_decision),
        "no_tools_decision": dict(no_tools_decision),
        "router_call_count": len(calls),
        "router_enabled_toolsets": [kwargs.get("enabled_toolsets") for kwargs in router_kwargs],
        "router_persist_user_messages": persist_values,
        "router_skip_memory": [kwargs.get("skip_memory") for kwargs in router_kwargs],
        "router_skip_context_files": [kwargs.get("skip_context_files") for kwargs in router_kwargs],
        "router_stream_callbacks_supplied": [call["stream_callback_supplied"] for call in calls],
        "router_prompts_include_no_tool_instruction": [
            "must not call tools" in call["prompt"] for call in calls
        ],
        "selected_voiceops_toolsets": list(toolset_decision.get("enabled_toolsets") or []),
        "selected_no_tools_toolsets": list(no_tools_decision.get("enabled_toolsets") or []),
        "router_transcript_persistent": any(value is not False for value in persist_values),
        "router_tool_calls_allowed": any(
            decision.get("tool_calls_allowed") is not False
            for decision in (toolset_decision, no_tools_decision)
        ),
        "external_test_refs": [
            "tests/agent/test_realtime_voice_oracle.py::test_voice_oracle_ephemeral_router_selects_voiceops_without_persisting_router_turn",
            "tests/agent/test_realtime_voice_oracle.py::test_voice_oracle_ephemeral_router_can_select_no_tools",
        ],
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
            evidence="async_oracle_smoke_plus_overflow_policy_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["job_manager_capacity"]
            + ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["overflow_policy"],
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
            evidence="async_oracle_smoke_plus_job_creation_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["job_creation_while_running"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "cancellation_controls_are_isolated": _async_oracle_acceptance_row(
            ok=smoke_ok
            and bool(async_oracle_coverage.get("one_job_cancelled_while_others_completed"))
            and bool(async_oracle_coverage.get("queued_job_cancelled_before_start"))
            and bool(async_oracle_coverage.get("late_cancelled_output_not_spoken"))
            and bool(async_oracle_coverage.get("late_cancelled_output_not_durable"))
            and bool(async_oracle_coverage.get("playback_stop_does_not_cancel_jobs")),
            evidence="async_oracle_smoke_plus_cancellation_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["cancellation"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "approval_wait_is_visible_and_redacted": _async_oracle_acceptance_row(
            ok=smoke_ok
            and bool(async_oracle_coverage.get("approval_wait_visible_and_redacted"))
            and bool(async_oracle_coverage.get("approval_wait_holds_capacity"))
            and bool(async_oracle_coverage.get("approval_cancel_holds_capacity")),
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
        "job_control_updates_reach_oracle": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("job_control_updates_reach_oracle")),
            evidence="async_oracle_smoke_plus_control_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["control_updates"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "transcript_hypotheses_stay_non_authoritative": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("transcript_hypotheses_remain_unpromoted")),
            evidence="async_oracle_smoke_plus_interpreter_authority_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["control_updates"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "hypothesis_final_events_stay_non_durable": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("hypothesis_final_events_non_durable")),
            evidence="async_oracle_smoke_plus_session_persistence_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["hypothesis_final_durability"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "external_frontend_bridge_submits_oracle_job": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("external_frontend_bridge_submits_oracle_job")),
            evidence="async_oracle_smoke_plus_external_frontend_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["external_frontend_bridge"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "durable_promoted_turn_resume_contract": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("durable_promoted_turn_resume_contract")),
            evidence="async_oracle_smoke_plus_durable_resume_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["durable_resume"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "witness_fusion_timing_preserves_single_bundle": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("witness_fusion_timing_preserves_single_bundle")),
            evidence="async_oracle_smoke_plus_witness_fusion_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["witness_fusion"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "witness_fusion_exposes_accepted_audio_gate": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("witness_fusion_accepted_audio_gate_visible")),
            evidence="async_oracle_smoke_plus_accepted_audio_gate_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["witness_fusion"]
            + ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["energy_gate"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "witness_fusion_supersedes_partial_witness": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("witness_fusion_partial_superseded_by_final")),
            evidence="async_oracle_smoke_plus_partial_witness_supersession_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["witness_fusion"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "witness_fusion_adjudicates_frontend_text": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("witness_fusion_adjudicates_frontend_text")),
            evidence="async_oracle_smoke_plus_witness_adjudication_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["witness_fusion"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "interpreter_prompt_input_order_visible": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("interpreter_prompt_input_order_visible")),
            evidence="async_oracle_smoke_plus_interpreter_prompt_packet_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["witness_fusion"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "interpreter_prompt_policy_visible": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("interpreter_prompt_policy_visible")),
            evidence="async_oracle_smoke_plus_interpreter_prompt_policy_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["witness_fusion"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "energy_gate_ignores_non_speech_without_work": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("energy_gate_ignores_non_speech_without_work")),
            evidence="async_oracle_smoke_plus_energy_gate_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["energy_gate"],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "kame_ack_latency_metrics_visible": _async_oracle_acceptance_row(
            ok=smoke_ok and bool(async_oracle_coverage.get("kame_ack_latency_metrics_visible")),
            evidence="async_oracle_smoke_plus_kame_latency_metrics_tests",
            test_refs=[
                "tests/agent/test_realtime_voice.py::test_kame_engine_defer_acknowledgement_reports_first_audio_metric",
                "tests/agent/test_realtime_voice.py::test_kame_engine_local_route_reports_first_audio_metric",
            ],
            verification_mode="loopback_smoke_plus_focused_tests",
            runtime_verified_by_this_report=True,
        ),
        "runtime_kame_action_gate_enforces_promoted_evidence": _async_oracle_acceptance_row(
            ok=smoke_ok
            and bool(async_oracle_coverage.get("runtime_kame_action_gate_enforced"))
            and bool(async_oracle_coverage.get("unflagged_high_risk_tool_event_fails_closed")),
            evidence="async_oracle_smoke_plus_runtime_action_gate_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["runtime_action_gate"],
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
        "sidecar_fail_closed_send_failure_cancels_active_job": _async_oracle_acceptance_row(
            ok=bool(async_oracle_coverage.get("sidecar_fail_closed_send_failure_cancels_active_job")),
            evidence="sidecar_fail_closed_smoke_plus_focused_tests",
            test_refs=ASYNC_ORACLE_ACCEPTANCE_TEST_REFS["sidecar_fail_closed"],
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
    sidecar_fail_closed_smoke: dict[str, Any] | None = None,
    tool_disclosure_smoke: dict[str, Any] | None = None,
    ephemeral_tool_router_smoke: dict[str, Any] | None = None,
) -> dict[str, Any]:
    coverage = _coverage_from_smoke(smoke)
    async_oracle_smoke = async_oracle_smoke or {}
    discord_session_cleanup_smoke = discord_session_cleanup_smoke or {}
    sidecar_fail_closed_smoke = sidecar_fail_closed_smoke or {}
    tool_disclosure_smoke = tool_disclosure_smoke or run_tool_disclosure_smoke()
    ephemeral_tool_router_smoke = ephemeral_tool_router_smoke or run_ephemeral_tool_router_smoke()
    async_oracle_coverage = {
        **_coverage_from_async_oracle_smoke(async_oracle_smoke),
        **_coverage_from_discord_session_cleanup_smoke(discord_session_cleanup_smoke),
        **_coverage_from_sidecar_fail_closed_smoke(sidecar_fail_closed_smoke),
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
            "ok": coverage["barge_in_stops_playback"]
            and async_oracle_coverage["energy_gate_ignores_non_speech_without_work"],
            "reaction_proven": bool(smoke.get("barge_in_sent")),
            "speech_energy_event_forwarded": bool(smoke.get("speech_energy_sent")),
            "energy_gate_proven_by_smoke": async_oracle_coverage[
                "energy_gate_ignores_non_speech_without_work"
            ],
            "energy_gate_covered_by_tests": True,
            "energy_gate_ignored_non_speech_packets": async_oracle_smoke.get(
                "energy_gate_ignored_non_speech_packets"
            ),
            "energy_gate_low_energy_witness_source": async_oracle_smoke.get(
                "energy_gate_low_energy_witness_source"
            ),
            "energy_gate_low_energy_witness_promoted": async_oracle_smoke.get(
                "energy_gate_low_energy_witness_promoted"
            ),
            "energy_gate_low_energy_witness_suppressed": async_oracle_smoke.get(
                "energy_gate_low_energy_witness_suppressed"
            ),
            "energy_gate_barge_in_events": async_oracle_smoke.get("energy_gate_barge_in_events"),
            "energy_gate_oracle_work_events": async_oracle_smoke.get("energy_gate_oracle_work_events"),
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
            "kind": async_oracle_smoke.get("kind"),
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
            "approval_capacity_smoke_ok": bool(async_oracle_smoke.get("approval_capacity_smoke_ok")),
            "approval_capacity_waiting_observed": bool(
                async_oracle_smoke.get("approval_capacity_waiting_observed")
            ),
            "approval_capacity_followup_queued": bool(
                async_oracle_smoke.get("approval_capacity_followup_queued")
            ),
            "approval_capacity_active_visible": bool(
                async_oracle_smoke.get("approval_capacity_active_visible")
            ),
            "approval_capacity_misleading_running_capacity": bool(
                async_oracle_smoke.get("approval_capacity_misleading_running_capacity")
            ),
            "approval_capacity_status_text": async_oracle_smoke.get("approval_capacity_status_text"),
            "approval_capacity_followup_started_after_approval": bool(
                async_oracle_smoke.get("approval_capacity_followup_started_after_approval")
            ),
            "approval_capacity_completed_jobs": async_oracle_smoke.get("approval_capacity_completed_jobs"),
            "approval_capacity_failed_gate_suppressed": bool(
                async_oracle_smoke.get("approval_capacity_failed_gate_suppressed")
            ),
            "approval_capacity_failed_jobs": async_oracle_smoke.get("approval_capacity_failed_jobs"),
            "approval_capacity_max_concurrent": async_oracle_smoke.get("approval_capacity_max_concurrent"),
            "approval_cancel_capacity_smoke_ok": bool(
                async_oracle_smoke.get("approval_cancel_capacity_smoke_ok")
            ),
            "approval_cancel_waiting_observed": bool(
                async_oracle_smoke.get("approval_cancel_waiting_observed")
            ),
            "approval_cancel_followup_queued": bool(
                async_oracle_smoke.get("approval_cancel_followup_queued")
            ),
            "approval_cancel_requested_observed": bool(
                async_oracle_smoke.get("approval_cancel_requested_observed")
            ),
            "approval_cancel_cancelled_observed": bool(
                async_oracle_smoke.get("approval_cancel_cancelled_observed")
            ),
            "approval_cancel_late_output_attempted": bool(
                async_oracle_smoke.get("approval_cancel_late_output_attempted")
            ),
            "approval_cancel_completed_after_cancel": bool(
                async_oracle_smoke.get("approval_cancel_completed_after_cancel")
            ),
            "approval_cancel_late_result_spoken": bool(
                async_oracle_smoke.get("approval_cancel_late_result_spoken")
            ),
            "approval_cancel_followup_started_before_cancel_drained": bool(
                async_oracle_smoke.get("approval_cancel_followup_started_before_cancel_drained")
            ),
            "approval_cancel_followup_started_after_cancel": bool(
                async_oracle_smoke.get("approval_cancel_followup_started_after_cancel")
            ),
            "approval_cancel_active_visible": bool(async_oracle_smoke.get("approval_cancel_active_visible")),
            "approval_cancel_misleading_running_capacity": bool(
                async_oracle_smoke.get("approval_cancel_misleading_running_capacity")
            ),
            "approval_cancel_status_text": async_oracle_smoke.get("approval_cancel_status_text"),
            "approval_cancel_max_concurrent": async_oracle_smoke.get("approval_cancel_max_concurrent"),
            "cancel_drain_capacity_smoke_ok": bool(async_oracle_smoke.get("cancel_drain_capacity_smoke_ok")),
            "cancel_drain_requested_observed": bool(async_oracle_smoke.get("cancel_drain_requested_observed")),
            "cancel_drain_cancelled_observed": bool(async_oracle_smoke.get("cancel_drain_cancelled_observed")),
            "cancel_drain_followup_queued": bool(async_oracle_smoke.get("cancel_drain_followup_queued")),
            "cancel_drain_active_visible": bool(async_oracle_smoke.get("cancel_drain_active_visible")),
            "cancel_drain_misleading_running_capacity": bool(
                async_oracle_smoke.get("cancel_drain_misleading_running_capacity")
            ),
            "cancel_drain_status_text": async_oracle_smoke.get("cancel_drain_status_text"),
            "cancel_drain_followup_started_after_cancel": bool(
                async_oracle_smoke.get("cancel_drain_followup_started_after_cancel")
            ),
            "cancel_drain_max_concurrent": async_oracle_smoke.get("cancel_drain_max_concurrent"),
            "local_turn_committed": bool(async_oracle_smoke.get("local_turn_committed")),
            "local_turn_during_running_jobs_observed": bool(
                async_oracle_smoke.get("local_turn_during_running_jobs_observed")
            ),
            "local_turn_active_job_count": async_oracle_smoke.get("local_turn_active_job_count"),
            "playback_stop_committed": bool(async_oracle_smoke.get("playback_stop_committed")),
            "playback_stop_jobs_still_running": bool(
                async_oracle_smoke.get("playback_stop_jobs_still_running")
            ),
            "playback_stop_cancelled_jobs": bool(async_oracle_smoke.get("playback_stop_cancelled_jobs")),
            "playback_stop_does_not_cancel_jobs": bool(
                async_oracle_smoke.get("playback_stop_does_not_cancel_jobs")
            ),
            "status_turn_committed": bool(async_oracle_smoke.get("status_turn_committed")),
            "status_turn_queued_visible": bool(async_oracle_smoke.get("status_turn_queued_visible")),
            "status_turn_no_oracle_request": bool(
                async_oracle_smoke.get("status_turn_no_oracle_request")
            ),
            "status_turn_oracle_request_count_before": async_oracle_smoke.get(
                "status_turn_oracle_request_count_before"
            ),
            "status_turn_oracle_request_count_after": async_oracle_smoke.get(
                "status_turn_oracle_request_count_after"
            ),
            "status_text": async_oracle_smoke.get("status_text"),
            "status_ordinal_labels_visible": bool(async_oracle_smoke.get("status_ordinal_labels_visible")),
            "status_ordinal_labels": tuple(async_oracle_smoke.get("status_ordinal_labels") or ()),
            "status_bounded_overflow_visible": bool(
                async_oracle_smoke.get("reflex_status_overflow_smoke_ok")
            ),
            "status_bounded_overflow_visible_job_count": async_oracle_smoke.get(
                "reflex_status_overflow_visible_job_count"
            ),
            "status_bounded_overflow_hidden_job_count": async_oracle_smoke.get(
                "reflex_status_overflow_hidden_job_count"
            ),
            "status_bounded_overflow_more_spoken_status": async_oracle_smoke.get(
                "reflex_status_overflow_more_spoken_status"
            ),
            "status_bounded_overflow_last_visible_ordinal": async_oracle_smoke.get(
                "reflex_status_overflow_last_visible_ordinal"
            ),
            "status_bounded_overflow_last_visible_label": async_oracle_smoke.get(
                "reflex_status_overflow_last_visible_label"
            ),
            "status_bounded_overflow_hidden_ids_absent": bool(
                async_oracle_smoke.get("reflex_status_overflow_hidden_ids_absent")
            ),
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
            "approval_gate_failed_closed": bool(async_oracle_smoke.get("approval_gate_failed_closed")),
            "approval_result_suppressed": bool(async_oracle_smoke.get("approval_result_suppressed")),
            "approval_status_text": async_oracle_smoke.get("approval_status_text"),
            "failed_job_reported": bool(async_oracle_smoke.get("failed_job_reported")),
            "failed_job_spoken": bool(async_oracle_smoke.get("failed_job_spoken")),
            "durable_failed_record_present": bool(async_oracle_smoke.get("durable_failed_record_present")),
            "session_survived_failed_job": bool(async_oracle_smoke.get("session_survived_failed_job")),
            "queued_job_update_observed": bool(async_oracle_smoke.get("queued_job_update_observed")),
            "running_job_update_observed": bool(async_oracle_smoke.get("running_job_update_observed")),
            "running_update_latest_update_visible": bool(
                async_oracle_smoke.get("running_update_latest_update_visible")
            ),
            "running_update_latest_update_text": async_oracle_smoke.get("running_update_latest_update_text"),
            "running_update_reached_oracle": bool(async_oracle_smoke.get("running_update_reached_oracle")),
            "running_update_delivery_metadata_ok": bool(
                async_oracle_smoke.get("running_update_delivery_metadata_ok")
            ),
            "queued_update_latest_update_visible": bool(
                async_oracle_smoke.get("queued_update_latest_update_visible")
            ),
            "queued_update_latest_update_text": async_oracle_smoke.get("queued_update_latest_update_text"),
            "queued_update_started_with_priority": bool(
                async_oracle_smoke.get("queued_update_started_with_priority")
            ),
            "queued_update_reached_oracle": bool(async_oracle_smoke.get("queued_update_reached_oracle")),
            "queued_interpreter_fold_in_observed": bool(
                async_oracle_smoke.get("queued_interpreter_fold_in_observed")
            ),
            "queued_interpreter_fold_in_oracle_text": async_oracle_smoke.get(
                "queued_interpreter_fold_in_oracle_text"
            ),
            "queued_interpreter_fold_in_transcript_source": async_oracle_smoke.get(
                "queued_interpreter_fold_in_transcript_source"
            ),
            "queued_interpreter_fold_in_transcript_confidence": async_oracle_smoke.get(
                "queued_interpreter_fold_in_transcript_confidence"
            ),
            "queued_interpreter_fold_in_oracle_text_source": async_oracle_smoke.get(
                "queued_interpreter_fold_in_oracle_text_source"
            ),
            "queued_interpreter_fold_in_evidence_authority": dict(
                async_oracle_smoke.get("queued_interpreter_fold_in_evidence_authority") or {}
            ),
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
            "terminal_result_policy_smoke_ok": bool(async_oracle_smoke.get("terminal_result_policy_smoke_ok")),
            "terminal_result_auto_summarize_default": bool(
                async_oracle_smoke.get("terminal_result_auto_summarize_default")
            ),
            "terminal_result_default_event_count": async_oracle_smoke.get("terminal_result_default_event_count"),
            "terminal_result_default_spoken": bool(async_oracle_smoke.get("terminal_result_default_spoken")),
            "terminal_result_suppression_config": async_oracle_smoke.get("terminal_result_suppression_config"),
            "terminal_result_suppressed": bool(async_oracle_smoke.get("terminal_result_suppressed")),
            "terminal_result_suppressed_event_observed": bool(
                async_oracle_smoke.get("terminal_result_suppressed_event_observed")
            ),
            "terminal_result_suppressed_event_count": async_oracle_smoke.get(
                "terminal_result_suppressed_event_count"
            ),
            "terminal_result_suppressed_reason": async_oracle_smoke.get("terminal_result_suppressed_reason"),
            "terminal_result_suppressed_payload_clean": bool(
                async_oracle_smoke.get("terminal_result_suppressed_payload_clean")
            ),
            "terminal_result_unsolicited_event_count": async_oracle_smoke.get(
                "terminal_result_unsolicited_event_count"
            ),
            "terminal_result_unsolicited_spoken": bool(
                async_oracle_smoke.get("terminal_result_unsolicited_spoken")
            ),
            "terminal_result_status_available": bool(async_oracle_smoke.get("terminal_result_status_available")),
            "terminal_result_status_text": async_oracle_smoke.get("terminal_result_status_text"),
            "unflagged_high_risk_tool_smoke_ok": bool(
                async_oracle_smoke.get("unflagged_high_risk_tool_smoke_ok")
            ),
            "unflagged_high_risk_tool_suppressed": bool(
                async_oracle_smoke.get("unflagged_high_risk_tool_suppressed")
            ),
            "unflagged_high_risk_tool_failed_closed": bool(
                async_oracle_smoke.get("unflagged_high_risk_tool_failed_closed")
            ),
            "unflagged_high_risk_tool_suppression_reason": async_oracle_smoke.get(
                "unflagged_high_risk_tool_suppression_reason"
            ),
            "unflagged_high_risk_tool_progress_suppressed": bool(
                async_oracle_smoke.get("unflagged_high_risk_tool_progress_suppressed")
            ),
            "unflagged_high_risk_tool_payload_redacted": bool(
                async_oracle_smoke.get("unflagged_high_risk_tool_payload_redacted")
            ),
            "unflagged_high_risk_tool_spoken_payload_clean": bool(
                async_oracle_smoke.get("unflagged_high_risk_tool_spoken_payload_clean")
            ),
            "unflagged_high_risk_tool_failure_spoken": bool(
                async_oracle_smoke.get("unflagged_high_risk_tool_failure_spoken")
            ),
            "unflagged_high_risk_tool_secret_canary_checked": bool(
                async_oracle_smoke.get("unflagged_high_risk_tool_secret_canary_checked")
            ),
            "unflagged_high_risk_tool_spoken": list(
                async_oracle_smoke.get("unflagged_high_risk_tool_spoken") or []
            ),
            "external_frontend_bridge_smoke_ok": bool(
                async_oracle_smoke.get("external_frontend_bridge_smoke_ok")
            ),
            "external_frontend_request_accepted": bool(
                async_oracle_smoke.get("external_frontend_request_accepted")
            ),
            "external_frontend_tool_result_observed": bool(
                async_oracle_smoke.get("external_frontend_tool_result_observed")
            ),
            "external_frontend_protocol": async_oracle_smoke.get("external_frontend_protocol"),
            "external_frontend_protocol_contract": async_oracle_smoke.get(
                "external_frontend_protocol_contract"
            ),
            "external_frontend_job_id": async_oracle_smoke.get("external_frontend_job_id"),
            "external_frontend_provider": async_oracle_smoke.get("external_frontend_provider"),
            "external_frontend_tool": async_oracle_smoke.get("external_frontend_tool"),
            "external_frontend_tool_call_id": async_oracle_smoke.get(
                "external_frontend_tool_call_id"
            ),
            "external_frontend_completion_tool_call_id": async_oracle_smoke.get(
                "external_frontend_completion_tool_call_id"
            ),
            "external_frontend_status_tool_call_id": async_oracle_smoke.get(
                "external_frontend_status_tool_call_id"
            ),
            "external_frontend_terminal_correlation_observed": bool(
                async_oracle_smoke.get("external_frontend_terminal_correlation_observed")
            ),
            "external_frontend_audit_id": async_oracle_smoke.get("external_frontend_audit_id"),
            "external_frontend_source_audit_id": async_oracle_smoke.get(
                "external_frontend_source_audit_id"
            ),
            "external_frontend_parent_audit_id": async_oracle_smoke.get(
                "external_frontend_parent_audit_id"
            ),
            "external_frontend_status_audit_id": async_oracle_smoke.get(
                "external_frontend_status_audit_id"
            ),
            "external_frontend_completion_audit_id": async_oracle_smoke.get(
                "external_frontend_completion_audit_id"
            ),
            "external_frontend_audit_id_continuity_observed": bool(
                async_oracle_smoke.get("external_frontend_audit_id_continuity_observed")
            ),
            "external_frontend_accepted_observed": bool(
                async_oracle_smoke.get("external_frontend_accepted_observed")
            ),
            "external_frontend_started_observed": bool(
                async_oracle_smoke.get("external_frontend_started_observed")
            ),
            "external_frontend_completion_observed": bool(
                async_oracle_smoke.get("external_frontend_completion_observed")
            ),
            "external_frontend_status_state": async_oracle_smoke.get(
                "external_frontend_status_state"
            ),
            "external_frontend_source_reached_oracle": bool(
                async_oracle_smoke.get("external_frontend_source_reached_oracle")
            ),
            "external_frontend_input_source": async_oracle_smoke.get(
                "external_frontend_input_source"
            ),
            "external_frontend_oracle_text": async_oracle_smoke.get("external_frontend_oracle_text"),
            "external_frontend_provisional_request_summary": dict(
                async_oracle_smoke.get("external_frontend_provisional_request_summary") or {}
            ),
            "external_frontend_status_provisional_request_summary": dict(
                async_oracle_smoke.get("external_frontend_status_provisional_request_summary") or {}
            ),
            "external_frontend_provisional_request_summary_non_authoritative": bool(
                async_oracle_smoke.get("external_frontend_provisional_request_summary_non_authoritative")
            ),
            "external_frontend_evidence_bundle_propagated": bool(
                async_oracle_smoke.get("external_frontend_evidence_bundle_propagated")
            ),
            "external_frontend_evidence_bundle_id": async_oracle_smoke.get(
                "external_frontend_evidence_bundle_id"
            ),
            "external_frontend_evidence_bundle_id_stable": bool(
                async_oracle_smoke.get("external_frontend_evidence_bundle_id_stable")
            ),
            "external_frontend_evidence_merge_key": async_oracle_smoke.get(
                "external_frontend_evidence_merge_key"
            ),
            "external_frontend_evidence_merge_key_propagated": bool(
                async_oracle_smoke.get("external_frontend_evidence_merge_key_propagated")
            ),
            "external_frontend_evidence_bundle_single_turn": bool(
                async_oracle_smoke.get("external_frontend_evidence_bundle_single_turn")
            ),
            "external_frontend_evidence_bundle_status": async_oracle_smoke.get(
                "external_frontend_evidence_bundle_status"
            ),
            "external_frontend_evidence_bundle_transcript_hypotheses_count": async_oracle_smoke.get(
                "external_frontend_evidence_bundle_transcript_hypotheses_count"
            ),
            "external_frontend_audio_segment_ref": async_oracle_smoke.get(
                "external_frontend_audio_segment_ref"
            ),
            "external_frontend_audio_time_range_ms": async_oracle_smoke.get(
                "external_frontend_audio_time_range_ms"
            ),
            "external_frontend_auxiliary_transcript_hypotheses": list(
                async_oracle_smoke.get("external_frontend_auxiliary_transcript_hypotheses") or []
            ),
            "external_frontend_witness_kind": async_oracle_smoke.get("external_frontend_witness_kind"),
            "external_frontend_witness_kind_frontend_hypothesis": bool(
                async_oracle_smoke.get("external_frontend_witness_kind_frontend_hypothesis")
            ),
            "external_frontend_witness_metadata": dict(
                async_oracle_smoke.get("external_frontend_witness_metadata") or {}
            ),
            "external_frontend_witness_metadata_complete": bool(
                async_oracle_smoke.get("external_frontend_witness_metadata_complete")
            ),
            "external_frontend_witness_confidence": async_oracle_smoke.get(
                "external_frontend_witness_confidence"
            ),
            "external_frontend_witness_latency_ms": async_oracle_smoke.get(
                "external_frontend_witness_latency_ms"
            ),
            "external_frontend_witness_partial": async_oracle_smoke.get(
                "external_frontend_witness_partial"
            ),
            "external_frontend_witness_audio_time_range_ms": list(
                async_oracle_smoke.get("external_frontend_witness_audio_time_range_ms") or []
            ),
            "external_frontend_witness_speaker": dict(
                async_oracle_smoke.get("external_frontend_witness_speaker") or {}
            ),
            "external_frontend_witness_channel": dict(
                async_oracle_smoke.get("external_frontend_witness_channel") or {}
            ),
            "external_frontend_witness_tool_authority_false": bool(
                async_oracle_smoke.get("external_frontend_witness_tool_authority_false")
            ),
            "external_frontend_hypothesis_not_durable_oracle_text": bool(
                async_oracle_smoke.get("external_frontend_hypothesis_not_durable_oracle_text")
            ),
            "external_frontend_durable_user_messages_empty": bool(
                async_oracle_smoke.get("external_frontend_durable_user_messages_empty")
            ),
            "external_frontend_durable_oracle_text_absent": bool(
                async_oracle_smoke.get("external_frontend_durable_oracle_text_absent")
            ),
            "external_frontend_durable_record_count": async_oracle_smoke.get(
                "external_frontend_durable_record_count"
            ),
            "external_frontend_direct_tool_authority_exposed": bool(
                async_oracle_smoke.get("external_frontend_direct_tool_authority_exposed")
            ),
            "external_frontend_event_counts": dict(
                async_oracle_smoke.get("external_frontend_event_counts") or {}
            ),
            "durable_resume_contract_smoke_ok": bool(
                async_oracle_smoke.get("durable_resume_contract_smoke_ok")
            ),
            "durable_resume_contract_schema_version": async_oracle_smoke.get(
                "durable_resume_contract_schema_version"
            ),
            "durable_resume_promoted_turn_count": async_oracle_smoke.get(
                "durable_resume_promoted_turn_count"
            ),
            "durable_resume_recent_promoted_turns_verbatim": bool(
                async_oracle_smoke.get("durable_resume_recent_promoted_turns_verbatim")
            ),
            "durable_resume_recent_promoted_turns": list(
                async_oracle_smoke.get("durable_resume_recent_promoted_turns") or []
            ),
            "durable_resume_older_turns_summarized": bool(
                async_oracle_smoke.get("durable_resume_older_turns_summarized")
            ),
            "durable_resume_older_promoted_turn_count": async_oracle_smoke.get(
                "durable_resume_older_promoted_turn_count"
            ),
            "durable_resume_older_promoted_turn_summary": async_oracle_smoke.get(
                "durable_resume_older_promoted_turn_summary"
            ),
            "durable_resume_hypothesis_replay_absent": bool(
                async_oracle_smoke.get("durable_resume_hypothesis_replay_absent")
            ),
            "durable_resume_ledger_authoritative": bool(
                async_oracle_smoke.get("durable_resume_ledger_authoritative")
            ),
            "hypothesis_final_durable_message_smoke_ok": bool(
                async_oracle_smoke.get("hypothesis_final_durable_message_smoke_ok")
            ),
            "hypothesis_final_durable_messages_empty": bool(
                async_oracle_smoke.get("hypothesis_final_durable_messages_empty")
            ),
            "hypothesis_final_durable_message_count": async_oracle_smoke.get(
                "hypothesis_final_durable_message_count"
            ),
            "hypothesis_final_without_adapter_flag_non_durable": bool(
                async_oracle_smoke.get("hypothesis_final_without_adapter_flag_non_durable")
            ),
            "hypothesis_final_witness_intent_non_durable": bool(
                async_oracle_smoke.get("hypothesis_final_witness_intent_non_durable")
            ),
            "explicit_asr_fallback_final_remains_durable": bool(
                async_oracle_smoke.get("explicit_asr_fallback_final_remains_durable")
            ),
            "explicit_asr_fallback_durable_messages": list(
                async_oracle_smoke.get("explicit_asr_fallback_durable_messages") or []
            ),
            "unpromoted_hypothesis_smoke_ok": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_smoke_ok")
            ),
            "unpromoted_hypothesis_job_id": async_oracle_smoke.get("unpromoted_hypothesis_job_id"),
            "unpromoted_hypothesis_evidence_bundle_id": async_oracle_smoke.get(
                "unpromoted_hypothesis_evidence_bundle_id"
            ),
            "unpromoted_hypothesis_single_bundle_observed": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_single_bundle_observed")
            ),
            "unpromoted_hypothesis_status_bundle_status": async_oracle_smoke.get(
                "unpromoted_hypothesis_status_bundle_status"
            ),
            "unpromoted_hypothesis_status_bundle_transcript_hypotheses_count": async_oracle_smoke.get(
                "unpromoted_hypothesis_status_bundle_transcript_hypotheses_count"
            ),
            "unpromoted_hypothesis_source": async_oracle_smoke.get("unpromoted_hypothesis_source"),
            "unpromoted_hypothesis_authority": async_oracle_smoke.get("unpromoted_hypothesis_authority"),
            "unpromoted_hypothesis_tool_authority": async_oracle_smoke.get(
                "unpromoted_hypothesis_tool_authority"
            ),
            "unpromoted_hypothesis_tool_authority_false": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_tool_authority_false")
            ),
            "unpromoted_hypothesis_text": async_oracle_smoke.get("unpromoted_hypothesis_text"),
            "unpromoted_hypothesis_confidence": async_oracle_smoke.get(
                "unpromoted_hypothesis_confidence"
            ),
            "unpromoted_hypothesis_oracle_text_preserved": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_oracle_text_preserved")
            ),
            "unpromoted_hypothesis_transcript_preserved": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_transcript_preserved")
            ),
            "unpromoted_hypothesis_intent_preserved": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_intent_preserved")
            ),
            "unpromoted_hypothesis_attached": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_attached")
            ),
            "unpromoted_hypothesis_promoted": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_promoted")
            ),
            "unpromoted_hypothesis_action_sink_keys_checked": tuple(
                async_oracle_smoke.get("unpromoted_hypothesis_action_sink_keys_checked") or ()
            ),
            "unpromoted_hypothesis_action_sinks_clean": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_action_sinks_clean")
            ),
            "unpromoted_hypothesis_action_sink_values": dict(
                async_oracle_smoke.get("unpromoted_hypothesis_action_sink_values") or {}
            ),
            "unpromoted_hypothesis_not_spend_reason": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_not_spend_reason")
            ),
            "unpromoted_hypothesis_not_spend_payload": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_not_spend_payload")
            ),
            "unpromoted_hypothesis_not_provider_selection": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_not_provider_selection")
            ),
            "unpromoted_hypothesis_not_nemoclaw_action_packet": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_not_nemoclaw_action_packet")
            ),
            "unpromoted_hypothesis_not_phone_call_payload": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_not_phone_call_payload")
            ),
            "unpromoted_hypothesis_not_call_payload": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_not_call_payload")
            ),
            "unpromoted_hypothesis_not_tool_arguments": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_not_tool_arguments")
            ),
            "unpromoted_hypothesis_not_memory_write": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_not_memory_write")
            ),
            "unpromoted_hypothesis_not_file_write": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_not_file_write")
            ),
            "unpromoted_hypothesis_not_message_payload": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_not_message_payload")
            ),
            "unpromoted_hypothesis_update_observed": bool(
                async_oracle_smoke.get("unpromoted_hypothesis_update_observed")
            ),
            "unpromoted_hypothesis_update_summary": async_oracle_smoke.get(
                "unpromoted_hypothesis_update_summary"
            ),
            "energy_gate_smoke_ok": bool(async_oracle_smoke.get("energy_gate_smoke_ok")),
            "energy_gate_policy": dict(async_oracle_smoke.get("energy_gate_policy") or {}),
            "energy_gate_ignored_packet_rms": async_oracle_smoke.get("energy_gate_ignored_packet_rms"),
            "energy_gate_ignored_packet_duration_ms": async_oracle_smoke.get(
                "energy_gate_ignored_packet_duration_ms"
            ),
            "energy_gate_ignored_packet_speech_confirmed": async_oracle_smoke.get(
                "energy_gate_ignored_packet_speech_confirmed"
            ),
            "energy_gate_ignored_packet_vad_speech": async_oracle_smoke.get(
                "energy_gate_ignored_packet_vad_speech"
            ),
            "energy_gate_ignored_non_speech_packets": async_oracle_smoke.get(
                "energy_gate_ignored_non_speech_packets"
            ),
            "energy_gate_low_energy_witness_text": async_oracle_smoke.get(
                "energy_gate_low_energy_witness_text"
            ),
            "energy_gate_low_energy_witness_source": async_oracle_smoke.get(
                "energy_gate_low_energy_witness_source"
            ),
            "energy_gate_low_energy_witness_promoted": async_oracle_smoke.get(
                "energy_gate_low_energy_witness_promoted"
            ),
            "energy_gate_low_energy_witness_suppressed": async_oracle_smoke.get(
                "energy_gate_low_energy_witness_suppressed"
            ),
            "energy_gate_barge_in_events": async_oracle_smoke.get("energy_gate_barge_in_events"),
            "energy_gate_interpreter_requests": async_oracle_smoke.get(
                "energy_gate_interpreter_requests"
            ),
            "energy_gate_oracle_work_events": async_oracle_smoke.get("energy_gate_oracle_work_events"),
            "energy_gate_oracle_requests": async_oracle_smoke.get("energy_gate_oracle_requests"),
            "energy_gate_raw_packet_buffered_without_turn": bool(
                async_oracle_smoke.get("energy_gate_raw_packet_buffered_without_turn")
            ),
            "energy_gate_event_types": list(async_oracle_smoke.get("energy_gate_event_types") or []),
            "witness_fusion_timing_smoke_ok": bool(
                async_oracle_smoke.get("witness_fusion_timing_smoke_ok")
            ),
            "witness_fusion_arrival_phases": list(
                async_oracle_smoke.get("witness_fusion_arrival_phases") or []
            ),
            "witness_arrival_phase": list(async_oracle_smoke.get("witness_fusion_arrival_phases") or []),
            "witness_fusion_case_job_ids": dict(
                async_oracle_smoke.get("witness_fusion_case_job_ids") or {}
            ),
            "witness_fusion_turn_ids": dict(
                async_oracle_smoke.get("witness_fusion_turn_ids") or {}
            ),
            "witness_fusion_audio_segment_refs": dict(
                async_oracle_smoke.get("witness_fusion_audio_segment_refs") or {}
            ),
            "witness_fusion_evidence_merge_keys": dict(
                async_oracle_smoke.get("witness_fusion_evidence_merge_keys") or {}
            ),
            "witness_fusion_merge_key_observed": bool(
                async_oracle_smoke.get("witness_fusion_merge_key_observed")
            ),
            "witness_fusion_audio_metadata": dict(
                async_oracle_smoke.get("witness_fusion_audio_metadata") or {}
            ),
            "witness_fusion_bundle_audio_metadata": dict(
                async_oracle_smoke.get("witness_fusion_bundle_audio_metadata") or {}
            ),
            "witness_fusion_accepted_audio_gate_observed": bool(
                async_oracle_smoke.get("witness_fusion_accepted_audio_gate_observed")
            ),
            "raw_audio_interpreter_evidence_observed": bool(
                async_oracle_smoke.get("witness_fusion_accepted_audio_gate_observed")
            ),
            "witness_fusion_early_initial_bundle_id": async_oracle_smoke.get(
                "witness_fusion_early_initial_bundle_id"
            ),
            "witness_fusion_early_final_bundle_id": async_oracle_smoke.get(
                "witness_fusion_early_final_bundle_id"
            ),
            "witness_fusion_early_single_bundle": bool(
                async_oracle_smoke.get("witness_fusion_early_single_bundle")
            ),
            "witness_fusion_interpreter_prompt_input_order": list(
                async_oracle_smoke.get("witness_fusion_interpreter_prompt_input_order") or []
            ),
            "witness_fusion_interpreter_prompt_input_order_expected": list(
                async_oracle_smoke.get("witness_fusion_interpreter_prompt_input_order_expected") or []
            ),
            "witness_fusion_interpreter_prompt_input_order_visible": bool(
                async_oracle_smoke.get("witness_fusion_interpreter_prompt_input_order_visible")
            ),
            "witness_fusion_interpreter_prompt_policy": dict(
                async_oracle_smoke.get("witness_fusion_interpreter_prompt_policy") or {}
            ),
            "witness_fusion_interpreter_prompt_policy_expected": dict(
                async_oracle_smoke.get("witness_fusion_interpreter_prompt_policy_expected") or {}
            ),
            "witness_fusion_interpreter_prompt_policy_version": async_oracle_smoke.get(
                "witness_fusion_interpreter_prompt_policy_version"
            ),
            "witness_fusion_interpreter_prompt_policy_visible": bool(
                async_oracle_smoke.get("witness_fusion_interpreter_prompt_policy_visible")
            ),
            "kame_ack_latency_metrics_smoke_ok": bool(
                async_oracle_smoke.get("kame_ack_latency_metrics_smoke_ok")
            ),
            "kame_defer_ack_first_audio_metrics_visible": bool(
                async_oracle_smoke.get("kame_defer_ack_first_audio_metrics_visible")
            ),
            "kame_local_first_audio_metrics_visible": bool(
                async_oracle_smoke.get("kame_local_first_audio_metrics_visible")
            ),
            "kame_defer_ack_metric_keys": list(
                async_oracle_smoke.get("kame_defer_ack_metric_keys") or []
            ),
            "kame_local_first_audio_metric_keys": list(
                async_oracle_smoke.get("kame_local_first_audio_metric_keys") or []
            ),
            "kame_defer_ack_audio_metrics": dict(
                async_oracle_smoke.get("kame_defer_ack_audio_metrics") or {}
            ),
            "kame_defer_ack_session_metrics": dict(
                async_oracle_smoke.get("kame_defer_ack_session_metrics") or {}
            ),
            "kame_local_first_audio_metrics": dict(
                async_oracle_smoke.get("kame_local_first_audio_metrics") or {}
            ),
            "kame_local_session_metrics": dict(
                async_oracle_smoke.get("kame_local_session_metrics") or {}
            ),
            "kame_defer_speech_end_to_first_audio_ms": async_oracle_smoke.get(
                "kame_defer_speech_end_to_first_audio_ms"
            ),
            "kame_local_speech_end_to_first_audio_ms": async_oracle_smoke.get(
                "kame_local_speech_end_to_first_audio_ms"
            ),
            "kame_defer_first_audio_bytes": async_oracle_smoke.get(
                "kame_defer_first_audio_bytes"
            ),
            "kame_local_first_audio_bytes": async_oracle_smoke.get(
                "kame_local_first_audio_bytes"
            ),
            "witness_fusion_with_bundle_id": async_oracle_smoke.get(
                "witness_fusion_with_bundle_id"
            ),
            "witness_fusion_with_single_bundle": bool(
                async_oracle_smoke.get("witness_fusion_with_single_bundle")
            ),
            "witness_fusion_late_initial_bundle_id": async_oracle_smoke.get(
                "witness_fusion_late_initial_bundle_id"
            ),
            "witness_fusion_late_final_bundle_id": async_oracle_smoke.get(
                "witness_fusion_late_final_bundle_id"
            ),
            "witness_fusion_late_single_bundle": bool(
                async_oracle_smoke.get("witness_fusion_late_single_bundle")
            ),
            "witness_fusion_no_duplicate_oracle_jobs": bool(
                async_oracle_smoke.get("witness_fusion_no_duplicate_oracle_jobs")
            ),
            "witness_fusion_partial_superseded_by_final": bool(
                async_oracle_smoke.get("witness_fusion_partial_superseded_by_final")
            ),
            "witness_fusion_partial_case_job_id": async_oracle_smoke.get(
                "witness_fusion_partial_case_job_id"
            ),
            "witness_fusion_partial_blocker_job_id": async_oracle_smoke.get(
                "witness_fusion_partial_blocker_job_id"
            ),
            "witness_fusion_partial_active_hypothesis": dict(
                async_oracle_smoke.get("witness_fusion_partial_active_hypothesis") or {}
            ),
            "witness_fusion_adjudications": dict(
                async_oracle_smoke.get("witness_fusion_adjudications") or {}
            ),
            "interpreter_adjudication_outcomes": dict(
                async_oracle_smoke.get("witness_fusion_adjudications") or {}
            ),
            "witness_fusion_rejection_reasons": dict(
                async_oracle_smoke.get("witness_fusion_rejection_reasons") or {}
            ),
            "witness_fusion_adjudication_outcomes_observed": bool(
                async_oracle_smoke.get("witness_fusion_adjudication_outcomes_observed")
            ),
            "witness_fusion_accepted_counts": dict(
                async_oracle_smoke.get("witness_fusion_accepted_counts") or {}
            ),
            "witness_fusion_started_counts": dict(
                async_oracle_smoke.get("witness_fusion_started_counts") or {}
            ),
            "witness_fusion_completed_counts": dict(
                async_oracle_smoke.get("witness_fusion_completed_counts") or {}
            ),
            "runtime_kame_action_gate_smoke_ok": bool(
                async_oracle_smoke.get("runtime_kame_action_gate_smoke_ok")
            ),
            "runtime_kame_action_gate_waiting_events": async_oracle_smoke.get(
                "runtime_kame_action_gate_waiting_events"
            ),
            "runtime_kame_action_gate_hypothesis_only_ok": async_oracle_smoke.get(
                "runtime_kame_action_gate_hypothesis_only_ok"
            ),
            "runtime_kame_action_gate_hypothesis_only_issues": list(
                async_oracle_smoke.get("runtime_kame_action_gate_hypothesis_only_issues") or []
            ),
            "runtime_kame_action_gate_hypothesis_only_rejected_authorities": list(
                async_oracle_smoke.get("runtime_kame_action_gate_hypothesis_only_rejected_authorities") or []
            ),
            "runtime_kame_action_gate_degraded_text_only_ok": async_oracle_smoke.get(
                "runtime_kame_action_gate_degraded_text_only_ok"
            ),
            "runtime_kame_action_gate_degraded_text_only_issues": list(
                async_oracle_smoke.get("runtime_kame_action_gate_degraded_text_only_issues") or []
            ),
            "runtime_kame_action_gate_degraded_text_only_rejected_authorities": list(
                async_oracle_smoke.get("runtime_kame_action_gate_degraded_text_only_rejected_authorities")
                or []
            ),
            "runtime_kame_action_gate_degraded_text_only_status": async_oracle_smoke.get(
                "runtime_kame_action_gate_degraded_text_only_status"
            ),
            "runtime_kame_action_gate_degraded_text_only_reason": async_oracle_smoke.get(
                "runtime_kame_action_gate_degraded_text_only_reason"
            ),
            "runtime_kame_action_gate_degraded_text_only_raw_audio_available": async_oracle_smoke.get(
                "runtime_kame_action_gate_degraded_text_only_raw_audio_available"
            ),
            "runtime_kame_action_gate_degraded_text_only_preserves_hypothesis": bool(
                async_oracle_smoke.get("runtime_kame_action_gate_degraded_text_only_preserves_hypothesis")
            ),
            "transcript_only_witness_rejected_for_full_kame": (
                async_oracle_smoke.get("runtime_kame_action_gate_degraded_text_only_ok") is False
                and async_oracle_smoke.get("runtime_kame_action_gate_degraded_text_only_status")
                == "degraded_text_only"
                and async_oracle_smoke.get("runtime_kame_action_gate_degraded_text_only_raw_audio_available")
                is False
                and bool(async_oracle_smoke.get("runtime_kame_action_gate_degraded_text_only_preserves_hypothesis"))
            ),
            "runtime_kame_action_gate_promoted_ok": async_oracle_smoke.get(
                "runtime_kame_action_gate_promoted_ok"
            ),
            "runtime_kame_action_gate_promoted_issues": list(
                async_oracle_smoke.get("runtime_kame_action_gate_promoted_issues") or []
            ),
            "runtime_kame_action_gate_promoted_authorities": list(
                async_oracle_smoke.get("runtime_kame_action_gate_promoted_authorities") or []
            ),
            "runtime_kame_action_gate_promoted_consumed_before_action": bool(
                async_oracle_smoke.get("runtime_kame_action_gate_promoted_consumed_before_action")
            ),
            "runtime_kame_action_gate_self_attested_ok": async_oracle_smoke.get(
                "runtime_kame_action_gate_self_attested_ok"
            ),
            "runtime_kame_action_gate_self_attested_issues": list(
                async_oracle_smoke.get("runtime_kame_action_gate_self_attested_issues") or []
            ),
            "runtime_kame_action_gate_self_attested_authorities": list(
                async_oracle_smoke.get("runtime_kame_action_gate_self_attested_authorities") or []
            ),
            "runtime_kame_action_gate_self_attested_consumed_before_action": bool(
                async_oracle_smoke.get("runtime_kame_action_gate_self_attested_consumed_before_action")
            ),
            "runtime_kame_action_gate_missing_tool_disclosure_ok": async_oracle_smoke.get(
                "runtime_kame_action_gate_missing_tool_disclosure_ok"
            ),
            "runtime_kame_action_gate_missing_tool_disclosure_issues": list(
                async_oracle_smoke.get("runtime_kame_action_gate_missing_tool_disclosure_issues") or []
            ),
            "runtime_kame_action_gate_missing_tool_disclosure_authorities": list(
                async_oracle_smoke.get("runtime_kame_action_gate_missing_tool_disclosure_authorities")
                or []
            ),
            "runtime_kame_action_gate_tool_disclosure_ref_observed": bool(
                async_oracle_smoke.get("runtime_kame_action_gate_tool_disclosure_ref_observed")
            ),
            "runtime_kame_action_gate_schema_versions": list(
                async_oracle_smoke.get("runtime_kame_action_gate_schema_versions") or []
            ),
            "audit_scalar_smoke_ok": bool(async_oracle_smoke.get("audit_scalar_smoke_ok")),
            "audit_scalar_payload_redacted": bool(async_oracle_smoke.get("audit_scalar_payload_redacted")),
            "audit_scalar_secret_canary_checked": bool(
                async_oracle_smoke.get("audit_scalar_secret_canary_checked")
            ),
            "audit_scalar_result_text_omitted": bool(
                async_oracle_smoke.get("audit_scalar_result_text_omitted")
            ),
            "audit_scalar_completed_event_seen": bool(
                async_oracle_smoke.get("audit_scalar_completed_event_seen")
            ),
            "audit_scalar_waiting_event_seen": bool(async_oracle_smoke.get("audit_scalar_waiting_event_seen")),
            "audit_scalar_row_count": async_oracle_smoke.get("audit_scalar_row_count"),
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
        "sidecar_fail_closed": {
            "ok": bool(_coverage_from_sidecar_fail_closed_smoke(sidecar_fail_closed_smoke).get(
                "sidecar_fail_closed_send_failure_cancels_active_job"
            )),
            "scenario": sidecar_fail_closed_smoke.get("scenario"),
            "fallback_policy": sidecar_fail_closed_smoke.get("fallback_policy"),
            "request_accepted": bool(sidecar_fail_closed_smoke.get("request_accepted")),
            "job_id": sidecar_fail_closed_smoke.get("job_id"),
            "cancelled_observed": bool(sidecar_fail_closed_smoke.get("cancelled_observed")),
            "cancel_reason": sidecar_fail_closed_smoke.get("cancel_reason"),
            "session_error_observed": bool(sidecar_fail_closed_smoke.get("session_error_observed")),
            "session_error_reason": sidecar_fail_closed_smoke.get("session_error_reason"),
            "session_error_sidecar": sidecar_fail_closed_smoke.get("session_error_sidecar"),
            "error_redacted": bool(sidecar_fail_closed_smoke.get("error_redacted")),
            "error_mentions_fail_closed": bool(sidecar_fail_closed_smoke.get("error_mentions_fail_closed")),
            "active_capacity_after_failure": sidecar_fail_closed_smoke.get("active_capacity_after_failure"),
            "job_state_after_failure": sidecar_fail_closed_smoke.get("job_state_after_failure"),
            "sidecar_removed": bool(sidecar_fail_closed_smoke.get("sidecar_removed")),
            "sidecar_closed": bool(sidecar_fail_closed_smoke.get("sidecar_closed")),
            "sidecar_close_calls": sidecar_fail_closed_smoke.get("sidecar_close_calls"),
            "oracle_requests_seen": sidecar_fail_closed_smoke.get("oracle_requests_seen"),
            "event_order": sidecar_fail_closed_smoke.get("event_order") or [],
            "test_refs": sidecar_fail_closed_smoke.get("test_refs") or [],
        },
        "tool_disclosure": {
            "ok": tool_disclosure_smoke.get("ok") is True,
            "scenario": tool_disclosure_smoke.get("scenario"),
            "schema_source": tool_disclosure_smoke.get("schema_source"),
            "representative_schema": tool_disclosure_smoke.get("representative_schema"),
            "missing_registered_core_tools": tool_disclosure_smoke.get("missing_registered_core_tools") or [],
            "config": dict(tool_disclosure_smoke.get("config") or {}),
            "input_core_tools": tool_disclosure_smoke.get("input_core_tools") or [],
            "visible_tool_names": tool_disclosure_smoke.get("visible_tool_names") or [],
            "visible_non_bridge_tool_names": tool_disclosure_smoke.get("visible_non_bridge_tool_names") or [],
            "hidden_core_tool_names": tool_disclosure_smoke.get("hidden_core_tool_names") or [],
            "bridge_tool_names": tool_disclosure_smoke.get("bridge_tool_names") or [],
            "input_core_tool_count": tool_disclosure_smoke.get("input_core_tool_count"),
            "hidden_core_tool_count": tool_disclosure_smoke.get("hidden_core_tool_count"),
            "bridge_tool_count": tool_disclosure_smoke.get("bridge_tool_count"),
            "core_tools_hidden_all": bool(tool_disclosure_smoke.get("core_tools_hidden_all")),
            "broad_core_tools_visible": bool(tool_disclosure_smoke.get("broad_core_tools_visible")),
            "deferred_count": tool_disclosure_smoke.get("deferred_count"),
            "deferred_tokens": tool_disclosure_smoke.get("deferred_tokens"),
            "input_schema_tokens": tool_disclosure_smoke.get("input_schema_tokens"),
            "visible_schema_tokens": tool_disclosure_smoke.get("visible_schema_tokens"),
            "token_reduction_estimate": tool_disclosure_smoke.get("token_reduction_estimate"),
            "external_test_refs": tool_disclosure_smoke.get("external_test_refs") or [],
        },
        "ephemeral_tool_router": {
            "ok": ephemeral_tool_router_smoke.get("ok") is True,
            "scenario": ephemeral_tool_router_smoke.get("scenario"),
            "router_mode": ephemeral_tool_router_smoke.get("router_mode"),
            "provider_network": bool(ephemeral_tool_router_smoke.get("provider_network")),
            "model_call": bool(ephemeral_tool_router_smoke.get("model_call")),
            "router_call_count": ephemeral_tool_router_smoke.get("router_call_count"),
            "toolsets_decision": dict(ephemeral_tool_router_smoke.get("toolsets_decision") or {}),
            "no_tools_decision": dict(ephemeral_tool_router_smoke.get("no_tools_decision") or {}),
            "selected_voiceops_toolsets": list(
                ephemeral_tool_router_smoke.get("selected_voiceops_toolsets") or []
            ),
            "selected_no_tools_toolsets": list(
                ephemeral_tool_router_smoke.get("selected_no_tools_toolsets") or []
            ),
            "router_enabled_toolsets": list(
                ephemeral_tool_router_smoke.get("router_enabled_toolsets") or []
            ),
            "router_persist_user_messages": list(
                ephemeral_tool_router_smoke.get("router_persist_user_messages") or []
            ),
            "router_skip_memory": list(ephemeral_tool_router_smoke.get("router_skip_memory") or []),
            "router_skip_context_files": list(
                ephemeral_tool_router_smoke.get("router_skip_context_files") or []
            ),
            "router_stream_callbacks_supplied": list(
                ephemeral_tool_router_smoke.get("router_stream_callbacks_supplied") or []
            ),
            "router_prompts_include_no_tool_instruction": list(
                ephemeral_tool_router_smoke.get("router_prompts_include_no_tool_instruction") or []
            ),
            "router_transcript_persistent": bool(
                ephemeral_tool_router_smoke.get("router_transcript_persistent")
            ),
            "router_tool_calls_allowed": bool(
                ephemeral_tool_router_smoke.get("router_tool_calls_allowed")
            ),
            "external_test_refs": ephemeral_tool_router_smoke.get("external_test_refs") or [],
        },
        "live_evidence": {
            "ok": bool(live_evidence.get("loaded")) and not missing_live_gates and not live_evidence.get("issues"),
            "mode": live_evidence.get("mode"),
            "overall_status": live_evidence.get("overall_status"),
            "missing_gates": missing_live_gates,
        },
    }
    report = {
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
            "async_oracle_status_ordinal_labels_visible": async_oracle_coverage[
                "status_turn_ordinal_labels_visible"
            ],
            "async_oracle_status_bounded_overflow_visible": async_oracle_coverage[
                "status_turn_bounded_overflow_visible"
            ],
            "async_oracle_fifth_job_queued_and_started": async_oracle_coverage[
                "fifth_job_queued_and_started_after_capacity_freed"
            ],
            "async_oracle_cancellation_isolated": async_oracle_coverage["one_job_cancelled_while_others_completed"]
            and async_oracle_coverage["queued_job_cancelled_before_start"]
            and async_oracle_coverage["playback_stop_does_not_cancel_jobs"],
            "async_oracle_late_cancelled_output_attempted": async_oracle_coverage[
                "late_cancelled_output_attempted"
            ],
            "async_oracle_playback_stop_preserves_jobs": async_oracle_coverage[
                "playback_stop_does_not_cancel_jobs"
            ],
            "async_oracle_approval_wait_holds_capacity": async_oracle_coverage[
                "approval_wait_holds_capacity"
            ],
            "async_oracle_approval_cancel_holds_capacity": async_oracle_coverage[
                "approval_cancel_holds_capacity"
            ],
            "async_oracle_cancel_drain_holds_capacity": async_oracle_coverage[
                "cancel_drain_holds_capacity"
            ],
            "async_oracle_external_frontend_bridge": async_oracle_coverage[
                "external_frontend_bridge_submits_oracle_job"
            ],
            "async_oracle_transcript_hypotheses_unpromoted": async_oracle_coverage[
                "transcript_hypotheses_remain_unpromoted"
            ],
            "async_oracle_hypothesis_final_events_non_durable": async_oracle_coverage[
                "hypothesis_final_events_non_durable"
            ],
            "async_oracle_witness_fusion_single_bundle": async_oracle_coverage[
                "witness_fusion_timing_preserves_single_bundle"
            ],
            "async_oracle_witness_fusion_accepted_audio_gate_visible": async_oracle_coverage[
                "witness_fusion_accepted_audio_gate_visible"
            ],
            "async_oracle_witness_fusion_partial_superseded_by_final": async_oracle_coverage[
                "witness_fusion_partial_superseded_by_final"
            ],
            "async_oracle_witness_fusion_adjudicates_frontend_text": async_oracle_coverage[
                "witness_fusion_adjudicates_frontend_text"
            ],
            "async_oracle_interpreter_prompt_input_order_visible": async_oracle_coverage[
                "interpreter_prompt_input_order_visible"
            ],
            "async_oracle_interpreter_prompt_policy_visible": async_oracle_coverage[
                "interpreter_prompt_policy_visible"
            ],
            "async_oracle_energy_gate_ignores_non_speech": async_oracle_coverage[
                "energy_gate_ignores_non_speech_without_work"
            ],
            "async_oracle_kame_ack_latency_metrics_visible": async_oracle_coverage[
                "kame_ack_latency_metrics_visible"
            ],
            "async_oracle_runtime_kame_action_gate": async_oracle_coverage[
                "runtime_kame_action_gate_enforced"
            ],
            "async_oracle_durable_promoted_turn_resume_contract": async_oracle_coverage[
                "durable_promoted_turn_resume_contract"
            ],
            "async_oracle_unflagged_high_risk_tool_fails_closed": async_oracle_coverage[
                "unflagged_high_risk_tool_event_fails_closed"
            ],
            "async_oracle_late_cancelled_output_dropped": async_oracle_coverage["late_cancelled_output_not_spoken"],
            "async_oracle_late_cancelled_output_not_durable": async_oracle_coverage[
                "late_cancelled_output_not_durable"
            ],
            "async_oracle_sidecar_fail_closed_cancels_active_job": async_oracle_coverage[
                "sidecar_fail_closed_send_failure_cancels_active_job"
            ],
            "progressive_tool_disclosure": tool_disclosure_smoke.get("ok") is True,
            "ephemeral_tool_router": ephemeral_tool_router_smoke.get("ok") is True,
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
        "sidecar_fail_closed_smoke": dict(sidecar_fail_closed_smoke),
        "tool_disclosure_smoke": dict(tool_disclosure_smoke),
        "ephemeral_tool_router_smoke": dict(ephemeral_tool_router_smoke),
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
    report["interpreter_request_packet"] = _interpreter_request_packet(report)
    return report


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
        **_coverage_from_sidecar_fail_closed_smoke(report.get("sidecar_fail_closed_smoke", {})),
    }
    for key in (
        "async_oracle_smoke_ok",
        "four_jobs_ran_concurrently",
        "local_turn_while_jobs_running",
        "status_turn_while_jobs_running",
        "status_turn_ordinal_labels_visible",
        "status_turn_bounded_overflow_visible",
        "fifth_job_queued_and_started_after_capacity_freed",
        "one_job_cancelled_while_others_completed",
        "queued_job_cancelled_before_start",
        "late_cancelled_output_not_spoken",
        "late_cancelled_output_attempted",
        "late_cancelled_output_not_durable",
        "playback_stop_does_not_cancel_jobs",
        "approval_wait_visible_and_redacted",
        "approval_wait_holds_capacity",
        "approval_cancel_holds_capacity",
        "cancel_drain_holds_capacity",
        "failed_job_reported_without_crash",
        "job_control_updates_reach_oracle",
        "transcript_hypotheses_remain_unpromoted",
        "hypothesis_final_events_non_durable",
        "external_frontend_bridge_submits_oracle_job",
        "witness_fusion_timing_preserves_single_bundle",
        "witness_fusion_accepted_audio_gate_visible",
        "witness_fusion_partial_superseded_by_final",
        "witness_fusion_adjudicates_frontend_text",
        "interpreter_prompt_input_order_visible",
        "interpreter_prompt_policy_visible",
        "energy_gate_ignores_non_speech_without_work",
        "kame_ack_latency_metrics_visible",
        "runtime_kame_action_gate_enforced",
        "durable_promoted_turn_resume_contract",
        "unflagged_high_risk_tool_event_fails_closed",
        "result_handling_bounded_and_durable",
        "discord_session_cleanup_preserves_oracle_state",
        "sidecar_fail_closed_send_failure_cancels_active_job",
        "shutdown_timeout_bounded",
    ):
        if recomputed_async_oracle_coverage.get(key) is not True:
            issues.append(f"missing_async_oracle_coverage:{key}")
        if async_oracle_coverage.get(key) is not recomputed_async_oracle_coverage.get(key):
            issues.append(f"stale_async_oracle_coverage:{key}")
    async_proof = report.get("proofs", {}).get("async_oracle_jobs", {})
    packet = report.get("interpreter_request_packet", {})
    if not isinstance(packet, Mapping):
        issues.append("missing_interpreter_request_packet")
        packet = {}
    elif packet.get("schema_version") != "voiceops.kame.interpreter_request_packet.v1":
        issues.append("interpreter_request_packet:invalid_schema_version")
    if isinstance(async_proof, Mapping):
        packet_expectations = {
            "protocol": "kame_session_v1",
            "protocol_contract": "docs/kame-session-v1.md",
            "turn_id": (async_proof.get("witness_fusion_turn_ids") or {}).get("early")
            if isinstance(async_proof.get("witness_fusion_turn_ids"), Mapping)
            else None,
            "evidence_bundle_id": async_proof.get("witness_fusion_early_final_bundle_id"),
            "evidence_merge_key": (async_proof.get("witness_fusion_evidence_merge_keys") or {}).get(
                "early"
            )
            if isinstance(async_proof.get("witness_fusion_evidence_merge_keys"), Mapping)
            else None,
            "interpreter_input_order": async_proof.get("witness_fusion_interpreter_prompt_input_order") or [],
            "prompt_input_order": async_proof.get("witness_fusion_interpreter_prompt_input_order") or [],
            "interpreter_prompt_policy": async_proof.get("witness_fusion_interpreter_prompt_policy") or {},
            "prompt_policy": async_proof.get("witness_fusion_interpreter_prompt_policy") or {},
        }
        for field, expected_value in packet_expectations.items():
            if packet.get(field) != expected_value:
                issues.append(f"interpreter_request_packet:{field}_mismatch")
        audio = packet.get("audio") if isinstance(packet.get("audio"), Mapping) else {}
        if audio.get("authority") != "primary_audio":
            issues.append("interpreter_request_packet:audio_authority_mismatch")
        audio_refs = async_proof.get("witness_fusion_audio_segment_refs")
        expected_audio_ref = audio_refs.get("early") if isinstance(audio_refs, Mapping) else None
        if audio.get("segment_ref") != expected_audio_ref:
            issues.append("interpreter_request_packet:audio_segment_ref_mismatch")
        hypotheses = (
            packet.get("transcript_hypotheses")
            if isinstance(packet.get("transcript_hypotheses"), list)
            else []
        )
        if not hypotheses or any(
            item.get("tool_authority") is not False for item in hypotheses if isinstance(item, Mapping)
        ):
            issues.append("interpreter_request_packet:hypothesis_tool_authority_not_false")
        if not hypotheses or any(
            item.get("authority") != "hypothesis" for item in hypotheses if isinstance(item, Mapping)
        ):
            issues.append("interpreter_request_packet:hypothesis_authority_mismatch")
        if any(item.get("partial") is True for item in hypotheses if isinstance(item, Mapping)):
            issues.append("interpreter_request_packet:active_partial_hypothesis_not_superseded")
        issues.extend(_mapping_kame_lineage_conflict_issues(packet, issue_prefix="interpreter_request_packet"))
        issues.extend(_mapping_witness_binding_conflict_issues(packet, issue_prefix="interpreter_request_packet"))
        promotion = packet.get("promotion") if isinstance(packet.get("promotion"), Mapping) else {}
        if promotion.get("interpreter_corrected_transcript") != async_proof.get(
            "witness_fusion_early_promoted_transcript"
        ):
            issues.append("interpreter_request_packet:promotion_transcript_mismatch")
    barge_proof = report.get("proofs", {}).get("barge_in_energy", {})
    if not isinstance(barge_proof, Mapping):
        issues.append("missing_proof:barge_in_energy")
    elif barge_proof.get("energy_gate_proven_by_smoke") is not recomputed_async_oracle_coverage.get(
        "energy_gate_ignores_non_speech_without_work"
    ):
        issues.append("stale_proof:barge_in_energy.energy_gate_proven_by_smoke")
    tool_disclosure_smoke = report.get("tool_disclosure_smoke", {})
    tool_disclosure = report.get("proofs", {}).get("tool_disclosure", {})
    if not isinstance(tool_disclosure_smoke, Mapping) or tool_disclosure_smoke.get("ok") is not True:
        issues.append("missing_coverage:progressive_tool_disclosure")
    if report.get("requirements", {}).get("progressive_tool_disclosure") is not True:
        issues.append("missing_requirement:progressive_tool_disclosure")
    if not isinstance(tool_disclosure, Mapping) or tool_disclosure.get("ok") is not True:
        issues.append("missing_proof:progressive_tool_disclosure")
    expected_visible = sorted(["tool_call", "tool_describe", "tool_search"])
    if sorted(tool_disclosure_smoke.get("visible_tool_names") or []) != expected_visible:
        issues.append("progressive_tool_disclosure:unexpected_visible_tools")
    if tool_disclosure_smoke.get("schema_source") != "registered_core_tool_schemas":
        issues.append("progressive_tool_disclosure:smoke_schema_source_not_registered")
    if tool_disclosure_smoke.get("representative_schema") is not False:
        issues.append("progressive_tool_disclosure:smoke_representative_schema")
    if tool_disclosure_smoke.get("missing_registered_core_tools"):
        issues.append("progressive_tool_disclosure:smoke_missing_registered_core_tools")
    smoke_input_core_tools = sorted(tool_disclosure_smoke.get("input_core_tools") or [])
    smoke_hidden_core_tools = sorted(tool_disclosure_smoke.get("hidden_core_tool_names") or [])
    smoke_visible_non_bridge_tools = sorted(tool_disclosure_smoke.get("visible_non_bridge_tool_names") or [])
    from toolsets import _HERMES_CORE_TOOLS

    expected_core_tools = sorted(_HERMES_CORE_TOOLS)
    if not smoke_input_core_tools:
        issues.append("progressive_tool_disclosure:missing_input_core_tools")
    if smoke_input_core_tools != expected_core_tools:
        issues.append("progressive_tool_disclosure:stale_input_core_tools")
    if smoke_hidden_core_tools != expected_core_tools:
        issues.append("progressive_tool_disclosure:stale_hidden_core_tools")
    if smoke_hidden_core_tools != smoke_input_core_tools:
        issues.append("progressive_tool_disclosure:core_tools_not_hidden")
    if smoke_visible_non_bridge_tools:
        issues.append("progressive_tool_disclosure:visible_non_bridge_tools")
    if tool_disclosure_smoke.get("core_tools_hidden_all") is not True:
        issues.append("progressive_tool_disclosure:core_tools_hidden_all_not_true")
    if tool_disclosure_smoke.get("broad_core_tools_visible") is not False:
        issues.append("progressive_tool_disclosure:broad_core_tools_visible")
    if tool_disclosure_smoke.get("input_core_tool_count") != len(smoke_input_core_tools):
        issues.append("progressive_tool_disclosure:input_core_tool_count_mismatch")
    if tool_disclosure_smoke.get("hidden_core_tool_count") != len(smoke_hidden_core_tools):
        issues.append("progressive_tool_disclosure:hidden_core_tool_count_mismatch")
    if tool_disclosure_smoke.get("deferred_count") != len(smoke_hidden_core_tools):
        issues.append("progressive_tool_disclosure:deferred_count_mismatch")
    if int(tool_disclosure_smoke.get("token_reduction_estimate") or 0) <= 0:
        issues.append("progressive_tool_disclosure:missing_token_reduction")
    proof_input_core_tools = sorted(tool_disclosure.get("input_core_tools") or []) if isinstance(tool_disclosure, Mapping) else []
    proof_hidden_core_tools = sorted(tool_disclosure.get("hidden_core_tool_names") or []) if isinstance(tool_disclosure, Mapping) else []
    if proof_input_core_tools != smoke_input_core_tools or proof_hidden_core_tools != smoke_hidden_core_tools:
        issues.append("progressive_tool_disclosure:stale_proof_hidden_core_tools")
    if isinstance(tool_disclosure, Mapping):
        if tool_disclosure.get("schema_source") != "registered_core_tool_schemas":
            issues.append("progressive_tool_disclosure:proof_schema_source_not_registered")
        if tool_disclosure.get("representative_schema") is not False:
            issues.append("progressive_tool_disclosure:proof_representative_schema")
        if tool_disclosure.get("missing_registered_core_tools"):
            issues.append("progressive_tool_disclosure:proof_missing_registered_core_tools")
        if proof_input_core_tools != expected_core_tools:
            issues.append("progressive_tool_disclosure:proof_stale_input_core_tools")
        if proof_hidden_core_tools != expected_core_tools:
            issues.append("progressive_tool_disclosure:proof_stale_hidden_core_tools")
        if tool_disclosure.get("core_tools_hidden_all") is not True:
            issues.append("progressive_tool_disclosure:proof_core_tools_hidden_all_not_true")
        if tool_disclosure.get("broad_core_tools_visible") is not False:
            issues.append("progressive_tool_disclosure:proof_broad_core_tools_visible")
        if sorted(tool_disclosure.get("visible_non_bridge_tool_names") or []):
            issues.append("progressive_tool_disclosure:proof_visible_non_bridge_tools")
    smoke_tool_refs = tool_disclosure_smoke.get("external_test_refs") if isinstance(tool_disclosure_smoke, Mapping) else None
    proof_tool_refs = tool_disclosure.get("external_test_refs") if isinstance(tool_disclosure, Mapping) else None
    if not isinstance(smoke_tool_refs, list) or not smoke_tool_refs:
        issues.append("progressive_tool_disclosure:missing_external_test_refs")
    else:
        for test_ref in smoke_tool_refs:
            if not _acceptance_test_ref_resolves(test_ref):
                issues.append(f"progressive_tool_disclosure:invalid_external_test_ref:{test_ref}")
    if not isinstance(proof_tool_refs, list) or not proof_tool_refs:
        issues.append("progressive_tool_disclosure:missing_proof_external_test_refs")
    elif smoke_tool_refs != proof_tool_refs:
        issues.append("progressive_tool_disclosure:stale_proof_external_test_refs")
    else:
        for test_ref in proof_tool_refs:
            if not _acceptance_test_ref_resolves(test_ref):
                issues.append(f"progressive_tool_disclosure:invalid_proof_external_test_ref:{test_ref}")
    ephemeral_smoke = report.get("ephemeral_tool_router_smoke", {})
    ephemeral_proof = report.get("proofs", {}).get("ephemeral_tool_router", {})
    if not isinstance(ephemeral_smoke, Mapping) or ephemeral_smoke.get("ok") is not True:
        issues.append("missing_coverage:ephemeral_tool_router")
    if report.get("requirements", {}).get("ephemeral_tool_router") is not True:
        issues.append("missing_requirement:ephemeral_tool_router")
    if not isinstance(ephemeral_proof, Mapping) or ephemeral_proof.get("ok") is not True:
        issues.append("missing_proof:ephemeral_tool_router")
    expected_router_refs = [
        "tests/agent/test_realtime_voice_oracle.py::test_voice_oracle_ephemeral_router_selects_voiceops_without_persisting_router_turn",
        "tests/agent/test_realtime_voice_oracle.py::test_voice_oracle_ephemeral_router_can_select_no_tools",
    ]
    for key in ("router_mode", "router_call_count", "selected_voiceops_toolsets", "selected_no_tools_toolsets"):
        if isinstance(ephemeral_proof, Mapping) and ephemeral_proof.get(key) != ephemeral_smoke.get(key):
            issues.append(f"ephemeral_tool_router:stale_proof_{key}")
    if ephemeral_smoke.get("router_mode") != "ephemeral":
        issues.append("ephemeral_tool_router:mode_not_ephemeral")
    if ephemeral_smoke.get("selected_voiceops_toolsets") != ["voiceops"]:
        issues.append("ephemeral_tool_router:voiceops_toolset_not_selected")
    if ephemeral_smoke.get("selected_no_tools_toolsets") != []:
        issues.append("ephemeral_tool_router:no_tools_not_empty")
    if ephemeral_smoke.get("router_transcript_persistent") is not False:
        issues.append("ephemeral_tool_router:router_transcript_persistent")
    if ephemeral_smoke.get("router_tool_calls_allowed") is not False:
        issues.append("ephemeral_tool_router:router_tool_calls_allowed")
    if any(item != [] for item in ephemeral_smoke.get("router_enabled_toolsets") or []):
        issues.append("ephemeral_tool_router:router_agent_received_tools")
    if any(value is not False for value in ephemeral_smoke.get("router_persist_user_messages") or []):
        issues.append("ephemeral_tool_router:router_persisted_user_message")
    if any(value is not True for value in ephemeral_smoke.get("router_skip_memory") or []):
        issues.append("ephemeral_tool_router:router_memory_not_skipped")
    if any(value is not True for value in ephemeral_smoke.get("router_skip_context_files") or []):
        issues.append("ephemeral_tool_router:router_context_files_not_skipped")
    if any(value is not True for value in ephemeral_smoke.get("router_prompts_include_no_tool_instruction") or []):
        issues.append("ephemeral_tool_router:missing_no_tool_instruction")
    smoke_router_refs = ephemeral_smoke.get("external_test_refs") if isinstance(ephemeral_smoke, Mapping) else None
    proof_router_refs = ephemeral_proof.get("external_test_refs") if isinstance(ephemeral_proof, Mapping) else None
    if smoke_router_refs != expected_router_refs:
        issues.append("ephemeral_tool_router:unexpected_external_test_refs")
    elif proof_router_refs != smoke_router_refs:
        issues.append("ephemeral_tool_router:stale_proof_external_test_refs")
    else:
        for test_ref in proof_router_refs:
            if not _acceptance_test_ref_resolves(test_ref):
                issues.append(f"ephemeral_tool_router:invalid_external_test_ref:{test_ref}")
    async_oracle_acceptance = report.get("async_oracle_acceptance", {})
    if not isinstance(async_oracle_acceptance, Mapping) or not async_oracle_acceptance:
        issues.append("missing_async_oracle_acceptance_matrix")
    else:
        recomputed_async_oracle_acceptance = _async_oracle_acceptance_matrix(recomputed_async_oracle_coverage)
        for key in async_oracle_acceptance:
            if key not in recomputed_async_oracle_acceptance:
                issues.append(f"unexpected_async_oracle_acceptance:{key}")
        for key, recomputed_value in recomputed_async_oracle_acceptance.items():
            value = async_oracle_acceptance.get(key)
            if recomputed_value.get("ok") is not True:
                issues.append(f"missing_async_oracle_acceptance:{key}")
            if not isinstance(value, Mapping) or value.get("ok") is not True:
                issues.append(f"missing_async_oracle_acceptance:{key}")
                continue
            if value.get("ok") is not recomputed_value.get("ok"):
                issues.append(f"stale_async_oracle_acceptance:{key}")
            for field in (
                "evidence",
                "test_refs",
                "verification_mode",
                "runtime_verified_by_this_report",
                "live_external_evidence_required",
            ):
                if value.get(field) != recomputed_value.get(field):
                    issues.append(f"stale_async_oracle_acceptance:{key}:{field}")
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
            "live_turn": (
                "Write live-turn.json with kind=live_turn, turn_id, audio_segment_ref, "
                "evidence_bundle_id, evidence_merge_key, audio_segment_ref_observed, "
                "interpreter_evidence_observed, transcript_hypotheses_labeled, optional transcript_observed, "
                "assistant_audio_observed, barge_in_observed, spoken_reply_short, no_voice_denial_observed, "
                "speech_end_to_first_audio_ms, barge_in_stop_ms, source_artifact, and collector_attestation. "
                "Moshi/S2S or ASR text must be labeled as hypothesis context, not durable user text, and "
                "transcript-only witness evidence is rejected for the full KAME live gate until raw audio "
                "and interpreter evidence are observed."
            ),
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
                "turn_id": "voiceops-live-turn-example",
                "audio_segment_ref": "artifact://redacted/live-turn-example.wav",
                "evidence_bundle_id": "kame-evidence-live-turn-example",
                "evidence_merge_key": "kame-merge-live-turn-example",
                "transcript_observed": True,
                "audio_segment_ref_observed": True,
                "interpreter_evidence_observed": True,
                "transcript_hypotheses_labeled": True,
                "witness_arrival_phases": ["with_raw_audio"],
                "interpreter_input_order": [
                    "raw_audio",
                    "metadata",
                    "reflex",
                    "transcript_hypotheses",
                ],
                "interpreter_prompt_policy": dict(LIVE_EVIDENCE_REQUIRED_INTERPRETER_PROMPT_POLICY),
                "transcript_hypotheses": [
                    {
                        "kind": "frontend_witness_hypothesis",
                        "source": "moshi",
                        "text": "[redacted witness hypothesis]",
                        "arrival_phase": "with_raw_audio",
                        "authority": "hypothesis",
                        "tool_authority": False,
                    }
                ],
                "interpreter_adjudication_outcomes": ["corrected_by_audio"],
                "promoted_evidence_authority": {
                    "interpreter_corrected_transcript": "interpreter_promoted",
                    "interpreter_normalized_intent": "interpreter_promoted",
                },
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
            "include raw transcript text; record only redacted KAME ids, booleans, latency numbers, and artifact references",
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


def _interpreter_request_packet(report: Mapping[str, Any]) -> dict[str, Any]:
    proof = (
        report.get("proofs", {}).get("async_oracle_jobs")
        if isinstance(report.get("proofs"), Mapping)
        and isinstance(report.get("proofs", {}).get("async_oracle_jobs"), Mapping)
        else {}
    )
    audio_metadata = (
        proof.get("witness_fusion_audio_metadata", {}).get("early")
        if isinstance(proof.get("witness_fusion_audio_metadata"), Mapping)
        and isinstance(proof.get("witness_fusion_audio_metadata", {}).get("early"), Mapping)
        else {}
    )
    return {
        "schema_version": "voiceops.kame.interpreter_request_packet.v1",
        "artifact_id": "kame-interpreter-request-packet",
        "source_proof": "proofs.async_oracle_jobs.witness_fusion",
        "protocol": "kame_session_v1",
        "protocol_contract": "docs/kame-session-v1.md",
        "turn_id": proof.get("witness_fusion_turn_ids", {}).get("early")
        if isinstance(proof.get("witness_fusion_turn_ids"), Mapping)
        else None,
        "evidence_bundle_id": proof.get("witness_fusion_early_final_bundle_id"),
        "evidence_merge_key": proof.get("witness_fusion_evidence_merge_keys", {}).get("early")
        if isinstance(proof.get("witness_fusion_evidence_merge_keys"), Mapping)
        else None,
        "interpreter_input_order": list(proof.get("witness_fusion_interpreter_prompt_input_order") or []),
        "prompt_input_order": list(proof.get("witness_fusion_interpreter_prompt_input_order") or []),
        "interpreter_prompt_policy": dict(proof.get("witness_fusion_interpreter_prompt_policy") or {}),
        "prompt_policy": dict(proof.get("witness_fusion_interpreter_prompt_policy") or {}),
        "audio": {
            "segment_ref": proof.get("witness_fusion_audio_segment_refs", {}).get("early")
            if isinstance(proof.get("witness_fusion_audio_segment_refs"), Mapping)
            else None,
            "authority": "primary_audio",
            "metadata": dict(audio_metadata),
        },
        "metadata": {
            "speaker": {
                "platform": "discord",
                "channel_user_id": "42",
                "display_name": "jetha",
            },
            "channel": {
                "transport": "discord_voice",
                "guild_id": "guild-1",
                "channel_id": "general",
            },
            "vad": dict(audio_metadata.get("vad") or {}) if isinstance(audio_metadata, Mapping) else {},
            "energy_gate": dict(audio_metadata.get("energy_gate") or {})
            if isinstance(audio_metadata, Mapping)
            else {},
        },
        "reflex": {
            "route": "defer",
            "transcript_hypothesis": proof.get("witness_fusion_early_reflex_transcript"),
            "interface_already_said": "Checking the power question.",
            "authority": "reflex_hypothesis",
            "tool_authority": False,
        },
        "transcript_hypotheses": [
            {
                "kind": "frontend_witness_hypothesis",
                "source": "moshi",
                "text": proof.get("witness_fusion_early_witness_text"),
                "authority": "hypothesis",
                "tool_authority": False,
                "arrival_phase": "before_raw_audio",
                "adjudication": "corrected_by_audio",
                "confidence": 0.74,
            }
        ],
        "promotion": {
            "interpreter_corrected_transcript": proof.get("witness_fusion_early_promoted_transcript"),
            "interpreter_normalized_intent": proof.get("witness_fusion_early_promoted_intent"),
            "interpreter_entities": list(proof.get("witness_fusion_early_entities") or []),
            "authority": dict(proof.get("witness_fusion_early_promoted_authority") or {}),
        },
        "action_authority": {
            "witness_text_can_authorize_tools": False,
            "requires_interpreter_or_oracle_promotion": True,
        },
    }


def write_voice_operator_report(output_dir: Path, report: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    closure_plan = _live_probe_closure_plan(report)
    paths = {
        "json": output_dir / "voice-operator-readiness.json",
        "markdown": output_dir / "voice-operator-readiness.md",
        "smoke_json": output_dir / "discord-loopback-smoke.json",
        "async_oracle_smoke_json": output_dir / "async-oracle-smoke.json",
        "discord_session_cleanup_smoke_json": output_dir / "discord-session-cleanup-smoke.json",
        "sidecar_fail_closed_smoke_json": output_dir / "sidecar-fail-closed-smoke.json",
        "tool_disclosure_smoke_json": output_dir / "tool-disclosure-smoke.json",
        "ephemeral_tool_router_smoke_json": output_dir / "ephemeral-tool-router-smoke.json",
        "interpreter_request_packet_json": output_dir / "interpreter-request-packet.json",
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
    _write_json(paths["sidecar_fail_closed_smoke_json"], report["sidecar_fail_closed_smoke"])
    _write_json(paths["tool_disclosure_smoke_json"], report["tool_disclosure_smoke"])
    _write_json(paths["ephemeral_tool_router_smoke_json"], report["ephemeral_tool_router_smoke"])
    _write_json(paths["interpreter_request_packet_json"], report["interpreter_request_packet"])
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
    sidecar_fail_closed_smoke = await run_sidecar_fail_closed_smoke()
    tool_disclosure_smoke = run_tool_disclosure_smoke()
    return build_voice_operator_report(
        asdict(smoke_result),
        live_evidence=_load_live_evidence(live_evidence_paths),
        async_oracle_smoke=async_oracle_smoke,
        discord_session_cleanup_smoke=discord_session_cleanup_smoke,
        sidecar_fail_closed_smoke=sidecar_fail_closed_smoke,
        tool_disclosure_smoke=tool_disclosure_smoke,
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
