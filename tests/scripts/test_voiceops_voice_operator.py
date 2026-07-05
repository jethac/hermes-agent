from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts.voiceops_voice_operator import (
    DEFAULT_OUTPUT_DIR,
    _load_live_evidence,
    build_live_probe_evidence_example,
    build_live_probe_evidence_template,
    build_voice_operator_report,
    parse_args,
    validate_live_probe_evidence,
    validate_voice_operator_report,
    write_voice_operator_report,
)
from toolsets import _HERMES_CORE_TOOLS


def _text_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _smoke_payload() -> dict:
    return {
        "ok": True,
        "mode": "discord_loopback",
        "transport": "discord_voice",
        "input_pcm48_bytes": 3840,
        "sidecar_pcm16_bytes": 640,
        "sidecar_pcm16_first_sample": 450,
        "sidecar_pcm16_checksum": 195,
        "mixer_frames": 1,
        "mixer_frame_bytes": 3840,
        "speech_energy_sent": True,
        "barge_in_sent": True,
        "mixer_stop_calls": 1,
        "sidecar_closed": True,
        "shutdown_elapsed_ms": 1,
        "shutdown_bounded": True,
        "shutdown_timed_out": False,
        "events": [
            "transcript.partial",
            "transcript.final",
            "assistant.text.partial",
            "audio.output.chunk",
            "assistant.commit",
            "barge_in",
        ],
        "evidence_context": {"git_commit": "abc", "git_branch": "branch"},
        "latency_metrics_ms": {
            "session_start_ms": 1,
            "input_to_first_mixer_frame_ms": 2,
            "barge_in_ack_ms": 3,
            "shutdown_ms": 1,
        },
        "error": "",
    }


def _async_oracle_smoke_payload() -> dict:
    return {
        "kind": "async_oracle_smoke",
        "ok": True,
        "scenario": "async_kame_oracle_jobs_fake",
        "max_running": 4,
        "max_worker_overlap": 4,
        "worker_overlap_proved": True,
        "worker_overlap_within_capacity": True,
        "noncooperative_cancel_overlap_observed": False,
        "started_jobs": 9,
        "queued_jobs": 1,
        "completed_jobs": 5,
        "failed_jobs": 2,
        "cancelled_jobs": 2,
        "queued_cancel_smoke_ok": True,
        "queued_cancel_observed": True,
        "queued_cancelled_before_start": True,
        "queued_cancel_not_sent_to_oracle": True,
        "queued_cancel_reason": "spoken request to cancel oracle job",
        "queued_cancel_target_job_id": "voice-oracle-002",
        "queued_cancel_running_completed": True,
        "approval_capacity_smoke_ok": True,
        "approval_capacity_waiting_observed": True,
        "approval_capacity_followup_queued": True,
        "approval_capacity_active_visible": True,
        "approval_capacity_misleading_running_capacity": False,
        "approval_capacity_status_text": (
            "Oracle jobs: 1 active out of 1, 0 running, 1 queued, 1 waiting for approval. "
            "waiting_for_approval: Preparing spend approval."
        ),
        "approval_capacity_followup_started_after_approval": True,
        "approval_capacity_completed_jobs": 1,
        "approval_capacity_failed_gate_suppressed": True,
        "approval_capacity_failed_jobs": 1,
        "approval_capacity_max_concurrent": 1,
        "approval_cancel_capacity_smoke_ok": True,
        "approval_cancel_waiting_observed": True,
        "approval_cancel_followup_queued": True,
        "approval_cancel_requested_observed": True,
        "approval_cancel_cancelled_observed": True,
        "approval_cancel_late_output_attempted": True,
        "approval_cancel_completed_after_cancel": False,
        "approval_cancel_late_result_spoken": False,
        "approval_cancel_followup_started_before_cancel_drained": False,
        "approval_cancel_followup_started_after_cancel": True,
        "approval_cancel_active_visible": True,
        "approval_cancel_misleading_running_capacity": False,
        "approval_cancel_status_text": (
            "Oracle jobs: 1 active out of 1, 0 running, 1 queued, 1 cancelling. "
            "cancel_requested: Preparing cancellable spend approval."
        ),
        "approval_cancel_max_concurrent": 1,
        "cancel_drain_capacity_smoke_ok": True,
        "cancel_drain_requested_observed": True,
        "cancel_drain_cancelled_observed": True,
        "cancel_drain_followup_queued": True,
        "cancel_drain_active_visible": True,
        "cancel_drain_misleading_running_capacity": False,
        "cancel_drain_status_text": (
            "Oracle jobs: 1 active out of 1, 0 running, 1 queued, 1 cancelling. "
            "cancel_requested: Starting cancellable smoke task."
        ),
        "cancel_drain_followup_started_after_cancel": True,
        "cancel_drain_max_concurrent": 1,
        "shutdown_timeout_configured_ms": 10,
        "shutdown_close_elapsed_ms": 15.0,
        "shutdown_bounded_close_observed": True,
        "shutdown_forced_cancel_observed": True,
        "shutdown_close_cancel_entered": True,
        "shutdown_cancelled_jobs": 1,
        "local_turn_committed": True,
        "local_turn_during_running_jobs_observed": True,
        "local_turn_active_job_count": 4,
        "playback_stop_committed": True,
        "playback_stop_jobs_still_running": True,
        "playback_stop_cancelled_jobs": False,
        "playback_stop_does_not_cancel_jobs": True,
        "status_turn_committed": True,
        "status_turn_queued_visible": True,
        "status_turn_no_oracle_request": True,
        "status_turn_oracle_request_count_before": 4,
        "status_turn_oracle_request_count_after": 4,
        "status_ordinal_labels_visible": True,
        "status_ordinal_labels": ("job one", "job two", "job three", "job four", "job five"),
        "reflex_status_overflow_smoke_ok": True,
        "reflex_status_overflow_visible_job_count": 8,
        "reflex_status_overflow_hidden_job_count": 2,
        "reflex_status_overflow_more_spoken_status": "+2 more",
        "reflex_status_overflow_last_visible_ordinal": 8,
        "reflex_status_overflow_last_visible_label": "job eight",
        "reflex_status_overflow_hidden_ids_absent": True,
        "status_text": (
            "Oracle jobs: 4 running out of 4, 1 queued. "
            "job one running: Starting smoke task 1. "
            "job two running: Starting smoke task 2. "
            "job three running: Starting smoke task 3. "
            "job four running: Starting smoke task 4. "
            "job five queued: Starting smoke task 5."
        ),
        "terminal_status_committed": True,
        "completed_result_status_visible": True,
        "terminal_status_text": (
            "No oracle jobs are running or queued right now. Recent: "
            "completed: First sentence. Second sentence. Third sentence."
        ),
        "fifth_job_id": "voice-oracle-005",
        "fifth_job_queued": True,
        "fifth_job_started_after_capacity_freed": True,
        "cancelled_job_id": "voice-oracle-003",
        "late_cancelled_output_attempted": True,
        "cancelled_result_spoken": False,
        "cancelled_result_committed": False,
        "cancelled_result_progress_leaked": False,
        "cancelled_result_durable_completed": False,
        "cancelled_result_durable_text": False,
        "durable_cancelled_record_present": True,
        "durable_completed_jobs": 5,
        "approval_wait_observed": True,
        "approval_status_committed": True,
        "approval_tool_progress_observed": True,
        "approval_tool_progress_kame_gate_present": True,
        "approval_tool_progress_kame_gate_schema_version": "voiceops.runtime_kame_action_gate.v1",
        "approval_tool_progress_kame_gate_failed_closed": True,
        "approval_tool_progress_kame_gate_issues": [
            "missing_promoted_evidence",
            "interpreter_evidence_not_consumed_before_irreversible_action",
        ],
        "approval_payload_redacted": True,
        "approval_secret_leaked": False,
        "approval_secret_canary_checked": True,
        "approval_completed": False,
        "approval_gate_failed_closed": True,
        "approval_result_suppressed": True,
        "approval_status_text": (
            "Oracle jobs: 1 active out of 4, 0 running, 1 waiting for approval. "
            "waiting_for_approval: Preparing spend approval."
        ),
        "failed_job_reported": True,
        "failed_job_spoken": True,
        "durable_failed_record_present": True,
        "session_survived_failed_job": True,
        "queued_job_update_observed": True,
        "running_job_update_observed": True,
        "running_update_latest_update_visible": True,
        "running_update_latest_update_text": "include running update context",
        "running_update_reached_oracle": True,
        "running_update_delivery_metadata_ok": True,
        "queued_update_latest_update_visible": True,
        "queued_update_latest_update_text": "include smoke update context",
        "queued_update_started_with_priority": True,
        "queued_update_reached_oracle": True,
        "queued_interpreter_fold_in_observed": True,
        "queued_interpreter_fold_in_oracle_text": "run corrected smoke task five",
        "queued_interpreter_fold_in_transcript_source": "gemma_interpreter",
        "queued_interpreter_fold_in_transcript_confidence": 0.88,
        "queued_interpreter_fold_in_oracle_text_source": "gemma_interpreter",
        "queued_interpreter_fold_in_evidence_authority": {
            "intent": "interpreter_promoted",
            "oracle_text": "interpreter_promoted",
            "reflex_transcript_hypothesis": "hypothesis",
            "transcript": "interpreter_promoted",
        },
        "verbose_result_spoken_bounded": True,
        "verbose_result_committed_bounded": True,
        "verbose_result_commit_marked_truncated": True,
        "verbose_full_result_durable": True,
        "verbose_full_result_chars": 48,
        "verbose_spoken_result": "First sentence.",
        "terminal_result_policy_smoke_ok": True,
        "terminal_result_auto_summarize_default": True,
        "terminal_result_default_event_count": 1,
        "terminal_result_default_spoken": True,
        "terminal_result_suppression_config": "oracle_jobs.speak_terminal_results=false",
        "terminal_result_suppressed": True,
        "terminal_result_suppressed_event_observed": True,
        "terminal_result_suppressed_event_count": 1,
        "terminal_result_suppressed_reason": "terminal_speech_disabled",
        "terminal_result_suppressed_payload_clean": True,
        "terminal_result_unsolicited_event_count": 0,
        "terminal_result_unsolicited_spoken": False,
        "terminal_result_status_available": True,
        "terminal_result_status_text": (
            "No oracle jobs are running or queued right now. Recent: "
            "completed: Finished Suppress terminal result."
        ),
        "unflagged_high_risk_tool_smoke_ok": True,
        "unflagged_high_risk_tool_cases": [
            {
                "category": category,
                "tool_name": tool_name,
                "ok": True,
                "suppressed": True,
                "failed_closed": True,
                "suppression_reason": "unapproved_high_risk_tool_event",
                "progress_suppressed": True,
                "payload_redacted": True,
                "spoken_payload_clean": True,
                "failure_spoken": True,
                "secret_canary_checked": True,
                "spoken": [
                    "Preparing the spend request.",
                    "I couldn't finish Buy service credits: KAME action gate failed; suppressed unapproved high-risk tool event",
                ],
            }
            for category, tool_name in (
                ("memory", "write_memory"),
                ("file", "write_file"),
                ("shell", "run_command"),
                ("spend", "stripe_link_purchase"),
                ("phone", "phone_call"),
                ("message", "whatsapp_send_message"),
                ("credential", "credential_write"),
                ("provisioning", "provision_service"),
                ("spend", "dispatch_action"),
            )
        ],
        "unflagged_high_risk_tool_case_count": 9,
        "unflagged_high_risk_tool_categories": [
            "memory",
            "file",
            "shell",
            "spend",
            "phone",
            "message",
            "credential",
            "provisioning",
            "spend",
        ],
        "unflagged_high_risk_tool_names": [
            "write_memory",
            "write_file",
            "run_command",
            "stripe_link_purchase",
            "phone_call",
            "whatsapp_send_message",
            "credential_write",
            "provision_service",
            "dispatch_action",
        ],
        "unflagged_high_risk_tool_all_cases_failed_closed": True,
        "unflagged_high_risk_tool_all_progress_suppressed": True,
        "unflagged_high_risk_tool_all_payloads_redacted": True,
        "unflagged_high_risk_tool_all_spoken_payloads_clean": True,
        "unflagged_high_risk_tool_suppressed": True,
        "unflagged_high_risk_tool_failed_closed": True,
        "unflagged_high_risk_tool_suppression_reason": "unapproved_high_risk_tool_event",
        "unflagged_high_risk_tool_progress_suppressed": True,
        "unflagged_high_risk_tool_payload_redacted": True,
        "unflagged_high_risk_tool_spoken_payload_clean": True,
        "unflagged_high_risk_tool_failure_spoken": True,
        "unflagged_high_risk_tool_secret_canary_checked": True,
        "unflagged_high_risk_tool_name": "write_memory",
        "unflagged_high_risk_tool_spoken": [
            "Preparing the spend request.",
            "I couldn't finish Buy service credits: KAME action gate failed; suppressed unapproved high-risk tool event",
        ],
        "external_frontend_bridge_smoke_ok": True,
        "external_frontend_request_accepted": True,
        "external_frontend_tool_result_observed": True,
        "external_frontend_protocol": "kame_session_v1",
        "external_frontend_protocol_contract": "docs/kame-session-v1.md",
        "external_frontend_mode": "witness_assisted_direct_audio",
        "external_frontend_interpreter_profile": "witness_assisted_direct_audio",
        "external_frontend_interpreter_input_order": [
            "raw_audio",
            "metadata",
            "reflex",
            "transcript_hypotheses",
        ],
        "external_frontend_witness_direct_audio_profile_ok": True,
        "external_frontend_witness_adjudications": [
            {
                "source": "moshi",
                "kind": "frontend_witness_hypothesis",
                "text_digest": _text_digest("prepare an external came hand off"),
                "adjudication": "corrected_by_audio",
            },
            {
                "source": "voiceclaw",
                "kind": "frontend_witness_hypothesis",
                "text_digest": _text_digest("prepare an external came hand off"),
                "adjudication": "corrected_by_audio",
            },
        ],
        "external_frontend_interpreter_promoted": {
            "corrected_transcript": "prepare an external KAME handoff",
            "normalized_intent": "Prepare external KAME handoff",
            "confidence": 0.91,
            "authority": "interpreter_promoted",
        },
        "external_frontend_job_id": "voice-oracle-001",
        "external_frontend_provider": "voiceclaw",
        "external_frontend_tool": "ask_brain",
        "external_frontend_tool_call_id": "voiceclaw-call-1",
        "external_frontend_completion_tool_call_id": "voiceclaw-call-1",
        "external_frontend_status_tool_call_id": "voiceclaw-call-1",
        "external_frontend_terminal_correlation_observed": True,
        "external_frontend_audit_id": "voiceclaw-audit-001",
        "external_frontend_source_audit_id": "discord-audit-voice-001",
        "external_frontend_parent_audit_id": "discord-audit-root-001",
        "external_frontend_status_audit_id": "voiceclaw-audit-001",
        "external_frontend_completion_audit_id": "voiceclaw-audit-001",
        "external_frontend_audit_id_continuity_observed": True,
        "external_frontend_accepted_observed": True,
        "external_frontend_started_observed": True,
        "external_frontend_completion_observed": True,
        "external_frontend_status_state": "completed",
        "external_frontend_source_reached_oracle": True,
        "external_frontend_input_source": "ask_brain",
        "external_frontend_oracle_text": "Prepare external KAME handoff",
        "external_frontend_promoted_request_summary": {
            "text": "Prepare external KAME handoff",
            "source": "gemma_interpreter",
            "authority": "interpreter_promoted",
            "tool_authority": False,
        },
        "external_frontend_provisional_request_summary": {
            "text": "Prepare external KAME handoff",
            "source": "reflex_audio",
            "kind": "reflex_hypothesis",
            "authority": "hypothesis",
            "tool_authority": False,
        },
        "external_frontend_status_provisional_request_summary": {
            "text": "Prepare external KAME handoff",
            "source": "reflex_audio",
            "kind": "reflex_hypothesis",
            "authority": "hypothesis",
            "tool_authority": False,
        },
        "external_frontend_provisional_request_summary_non_authoritative": True,
        "external_frontend_evidence_bundle_propagated": True,
        "external_frontend_evidence_bundle_id": "kame-evidence-abc123",
        "external_frontend_evidence_bundle_id_stable": True,
        "external_frontend_evidence_merge_key": "kame-merge-external-front-end",
        "external_frontend_evidence_merge_key_propagated": True,
        "external_frontend_evidence_bundle_single_turn": True,
        "external_frontend_evidence_bundle_status": "primary_audio",
        "external_frontend_evidence_bundle_transcript_hypotheses_count": 1,
        "external_frontend_audio_segment_ref": "artifact://voiceclaw/turn-1.wav",
        "external_frontend_audio_time_range_ms": [100, 2100],
        "external_frontend_transcript_hypotheses": [
            {
                "source": "moshi",
                "kind": "frontend_witness_hypothesis",
                "text": "prepare an external kame handoff",
                "role": "witness_context",
                "authority": "hypothesis",
                "promotion_required": "interpreter_promoted_or_oracle_promoted",
                "tool_authority": False,
                "confidence": 0.78,
                "latency_ms": 140,
                "partial": False,
                "audio_time_range_ms": [120, 2080],
                "arrival_phase": "with_raw_audio",
                "adjudication": "corrected_by_audio",
                "speaker": {
                    "platform": "discord",
                    "channel_user_id": "jetha-redacted",
                    "display_name": "jetha",
                    "is_bot": False,
                },
                "channel": {
                    "transport": "discord_voice",
                    "guild_id": "guild-redacted",
                    "channel_id": "general-redacted",
                    "surface": "desk_voice",
                },
            }
        ],
        "external_frontend_auxiliary_transcript_hypotheses": [
            {
                "source": "moshi",
                "kind": "frontend_witness_hypothesis",
                "text": "prepare an external kame handoff",
                "role": "witness_context",
                "authority": "hypothesis",
                "promotion_required": "interpreter_promoted_or_oracle_promoted",
                "tool_authority": False,
                "confidence": 0.78,
                "latency_ms": 140,
                "partial": False,
                "audio_time_range_ms": (120, 2080),
                "arrival_phase": "with_raw_audio",
                "adjudication": "corrected_by_audio",
                "speaker": {
                    "platform": "discord",
                    "channel_user_id": "jetha-redacted",
                    "display_name": "jetha",
                    "is_bot": False,
                },
                "channel": {
                    "transport": "discord_voice",
                    "guild_id": "guild-redacted",
                    "channel_id": "general-redacted",
                    "surface": "desk_voice",
                },
            }
        ],
        "external_frontend_witness_kind": "frontend_witness_hypothesis",
        "external_frontend_witness_kind_frontend_hypothesis": True,
        "external_frontend_witness_metadata": {
            "source": "moshi",
            "kind": "frontend_witness_hypothesis",
            "text": "prepare an external kame handoff",
            "role": "witness_context",
            "authority": "hypothesis",
            "promotion_required": "interpreter_promoted_or_oracle_promoted",
            "tool_authority": False,
            "confidence": 0.78,
            "latency_ms": 140,
            "partial": False,
            "audio_time_range_ms": (120, 2080),
            "arrival_phase": "with_raw_audio",
            "adjudication": "corrected_by_audio",
            "speaker": {
                "platform": "discord",
                "channel_user_id": "jetha-redacted",
                "display_name": "jetha",
                "is_bot": False,
            },
            "channel": {
                "transport": "discord_voice",
                "guild_id": "guild-redacted",
                "channel_id": "general-redacted",
                "surface": "desk_voice",
            },
        },
        "external_frontend_witness_metadata_complete": True,
        "external_frontend_witness_confidence": 0.78,
        "external_frontend_witness_latency_ms": 140,
        "external_frontend_witness_partial": False,
        "external_frontend_witness_audio_time_range_ms": [120, 2080],
        "external_frontend_witness_speaker": {
            "platform": "discord",
            "channel_user_id": "jetha-redacted",
            "display_name": "jetha",
            "is_bot": False,
        },
        "external_frontend_witness_channel": {
            "transport": "discord_voice",
            "guild_id": "guild-redacted",
            "channel_id": "general-redacted",
            "surface": "desk_voice",
        },
        "external_frontend_witness_tool_authority_false": True,
        "external_frontend_witness_role_context": True,
        "external_frontend_witness_promotion_required": True,
        "external_frontend_hypothesis_not_durable_oracle_text": True,
        "external_frontend_durable_user_messages_empty": True,
        "external_frontend_durable_oracle_text_absent": True,
        "external_frontend_durable_record_count": 2,
        "external_frontend_direct_tool_authority_exposed": False,
        "external_frontend_direct_tool_rejected": True,
        "external_frontend_direct_tool_rejected_tool": "stripe_link_purchase",
        "external_frontend_direct_tool_rejection_reason": "unsupported_external_kame_tool",
        "external_frontend_direct_tool_created_oracle_job": False,
        "external_frontend_tool_result_payload_safe": True,
        "external_frontend_reflex_status_payload_safe": True,
        "external_frontend_placeholder_payload_safe": True,
        "external_frontend_tool_result_forbidden_paths": [],
        "external_frontend_reflex_status_forbidden_paths": [],
        "external_frontend_placeholder": "Accepted job one running: I'm preparing the handoff.",
        "external_frontend_placeholder_forbidden_paths": [],
        "minimum_interpreter_packet_smoke_ok": True,
        "minimum_interpreter_packet": {
            "schema_version": "voiceops.minimum_interpreter_packet.v1",
            "mode": "witness_assisted_direct_audio",
            "turn_id": "voice-smoke-external-frontend:voiceclaw:1",
            "audio_segment_ref": "artifact://voiceclaw/turn-1.wav",
            "interpreter_input_order": [
                "raw_audio",
                "metadata",
                "reflex",
                "transcript_hypotheses",
            ],
            "metadata": {
                "evidence_bundle_id": "kame-evidence-abc123",
                "evidence_merge_key": "kame-merge-external-front-end",
                "speaker_or_actor_ref": "discord:jetha-redacted",
                "channel_or_surface_ref": "discord_voice:general-redacted",
                "vad_speech": True,
                "energy_gate": "accepted",
                "audio_time_range_ms": [100, 2100],
            },
            "reflex": {
                "acknowledgement_text": "I'm preparing the handoff.",
                "acknowledgement_source": "reflex_acknowledgement",
                "route": "defer",
                "authority": "reflex_hypothesis",
                "tool_authority": False,
            },
            "transcript_hypotheses": [
                {
                    "source": "moshi",
                    "kind": "frontend_witness_hypothesis",
                    "text_digest": _text_digest("prepare an external kame handoff"),
                    "text_redacted": True,
                    "role": "witness_context",
                    "authority": "hypothesis",
                    "promotion_required": "interpreter_promoted_or_oracle_promoted",
                    "tool_authority": False,
                    "arrival_phase": "with_raw_audio",
                    "latency_ms": 140,
                    "confidence": 0.78,
                    "partial": False,
                    "adjudication": "corrected_by_audio",
                }
            ],
        },
        "minimum_interpreter_packet_input_order": [
            "raw_audio",
            "metadata",
            "reflex",
            "transcript_hypotheses",
        ],
        "minimum_interpreter_packet_text_redacted": True,
        "minimum_interpreter_packet_witness_count": 1,
        "minimum_interpreter_packet_raw_audio_primary": True,
        "minimum_interpreter_packet_hypotheses_authority": True,
        "external_frontend_event_counts": {
            "tool.result": 2,
            "oracle.job.accepted": 1,
            "oracle.job.started": 1,
            "oracle.job.completed": 1,
        },
        "unpromoted_hypothesis_smoke_ok": True,
        "unpromoted_hypothesis_job_id": "voice-oracle-002",
        "unpromoted_hypothesis_evidence_bundle_id": "kame-evidence-def456",
        "unpromoted_hypothesis_single_bundle_observed": True,
        "unpromoted_hypothesis_status_bundle_status": "primary_audio",
        "unpromoted_hypothesis_status_bundle_transcript_hypotheses_count": 1,
        "unpromoted_hypothesis_source": "moshi",
        "unpromoted_hypothesis_authority": "hypothesis",
        "unpromoted_hypothesis_tool_authority": False,
        "unpromoted_hypothesis_tool_authority_false": True,
        "unpromoted_hypothesis_text": "spend two hundred dollars and call my phone",
        "unpromoted_hypothesis_confidence": 0.71,
        "unpromoted_hypothesis_oracle_text_preserved": True,
        "unpromoted_hypothesis_transcript_preserved": True,
        "unpromoted_hypothesis_intent_preserved": True,
        "unpromoted_hypothesis_attached": True,
        "unpromoted_hypothesis_promoted": False,
        "unpromoted_hypothesis_action_sink_keys_checked": (
            "spend_reason",
            "spend_payload",
            "provider_selection",
            "provider_choice",
            "provider_payload",
            "nemoclaw_action_packet",
            "nemoclaw_action_payload",
            "action_packet",
            "action_payload",
            "approval_payload",
            "phone_call_payload",
            "call_payload",
            "tool_arguments",
            "arguments",
            "memory_write",
            "file_write",
            "message_payload",
            "external_message",
        ),
        "unpromoted_hypothesis_action_sinks_clean": True,
        "unpromoted_hypothesis_action_sink_values": {},
        "unpromoted_hypothesis_not_spend_reason": True,
        "unpromoted_hypothesis_not_spend_payload": True,
        "unpromoted_hypothesis_not_provider_selection": True,
        "unpromoted_hypothesis_not_nemoclaw_action_packet": True,
        "unpromoted_hypothesis_not_phone_call_payload": True,
        "unpromoted_hypothesis_not_call_payload": True,
        "unpromoted_hypothesis_not_tool_arguments": True,
        "unpromoted_hypothesis_not_memory_write": True,
        "unpromoted_hypothesis_not_file_write": True,
        "unpromoted_hypothesis_not_message_payload": True,
        "unpromoted_hypothesis_update_observed": True,
        "unpromoted_hypothesis_update_summary": "interpreter evidence: auxiliary_hypotheses=1",
        "witness_fusion_timing_smoke_ok": True,
        "witness_fusion_arrival_phases": [
            "before_raw_audio",
            "with_raw_audio",
            "after_interpreter_start",
        ],
        "witness_fusion_case_job_ids": {
            "early": "voice-oracle-003",
            "with": "voice-oracle-004",
            "late": "voice-oracle-005",
        },
        "witness_fusion_turn_ids": {
            "early": "witness-fusion:early",
            "with": "witness-fusion:with",
            "late": "witness-fusion:late",
        },
        "witness_fusion_audio_segment_refs": {
            "early": "artifact://voice/witness-early.wav",
            "with": "artifact://voice/witness-with.wav",
            "late": "artifact://voice/witness-late.wav",
        },
        "witness_fusion_evidence_merge_keys": {
            "early": "kame-merge-witness-early",
            "with": "kame-merge-witness-with",
            "late": "kame-merge-witness-late",
        },
        "witness_fusion_merge_key_observed": True,
        "witness_fusion_same_turn_convergence_ok": True,
        "witness_fusion_same_turn_arrival_phases": [
            "before_raw_audio",
            "with_raw_audio",
            "after_interpreter_start",
        ],
        "witness_fusion_same_turn_lineage": {
            "session_id": "voice-smoke-witness-fusion",
            "turn_id": "witness-fusion:same-turn",
            "audio_segment_ref": "artifact://voice/witness-same-turn.wav",
            "evidence_bundle_id": "kame-evidence-witness-same-turn",
            "evidence_merge_key": "kame-merge-witness-same-turn",
            "job_id": "voice-oracle-008",
        },
        "witness_fusion_same_turn_phase_lineage": {
            "before_raw_audio": {
                "session_id": "voice-smoke-witness-fusion",
                "turn_id": "witness-fusion:same-turn",
                "audio_segment_ref": "",
                "evidence_bundle_id": "kame-evidence-witness-same-turn",
                "evidence_merge_key": "kame-merge-witness-same-turn-pending",
                "job_id": "voice-oracle-008",
            },
            "with_raw_audio": {
                "session_id": "voice-smoke-witness-fusion",
                "turn_id": "witness-fusion:same-turn",
                "audio_segment_ref": "artifact://voice/witness-same-turn.wav",
                "evidence_bundle_id": "kame-evidence-witness-same-turn",
                "evidence_merge_key": "kame-merge-witness-same-turn",
                "job_id": "voice-oracle-008",
            },
            "after_interpreter_start": {
                "session_id": "voice-smoke-witness-fusion",
                "turn_id": "witness-fusion:same-turn",
                "audio_segment_ref": "artifact://voice/witness-same-turn.wav",
                "evidence_bundle_id": "kame-evidence-witness-same-turn",
                "evidence_merge_key": "kame-merge-witness-same-turn",
                "job_id": "voice-oracle-008",
            },
        },
        "witness_fusion_same_turn_bundle_ids_by_phase": {
            "before_raw_audio": "kame-evidence-witness-same-turn",
            "with_raw_audio": "kame-evidence-witness-same-turn",
            "after_interpreter_start": "kame-evidence-witness-same-turn",
        },
        "witness_fusion_same_turn_job_ids_by_phase": {
            "before_raw_audio": "voice-oracle-008",
            "with_raw_audio": "voice-oracle-008",
            "after_interpreter_start": "voice-oracle-008",
        },
        "witness_fusion_same_turn_single_bundle": True,
        "witness_fusion_same_turn_one_oracle_job": True,
        "witness_fusion_same_turn_oracle_job_counts": {
            "accepted": 1,
            "started": 1,
            "completed": 1,
        },
        "witness_fusion_same_turn_no_duplicate_oracle_job": True,
        "witness_fusion_same_turn_expected_merge_key": "kame-merge-witness-same-turn",
        "witness_fusion_audio_metadata": {
            "early": {
                "codec": "pcm_s16le",
                "sample_rate_hz": 16000,
                "channels": 1,
                "authority": "primary_audio",
                "vad": {"speech_start_ms": 100, "speech_end_ms": 1400, "vad_speech": True},
                "energy_gate": {
                    "accepted": True,
                    "rms": 620,
                    "duration_ms": 1300,
                    "min_rms": 350,
                    "min_speech_ms": 120,
                },
            },
            "with": {
                "codec": "pcm_s16le",
                "sample_rate_hz": 16000,
                "channels": 1,
                "authority": "primary_audio",
                "time_range_ms": (200, 1500),
                "vad": {"speech_start_ms": 200, "speech_end_ms": 1500, "vad_speech": True},
                "energy_gate": {
                    "accepted": True,
                    "rms": 620,
                    "duration_ms": 1300,
                    "min_rms": 350,
                    "min_speech_ms": 120,
                },
            },
            "late": {
                "codec": "pcm_s16le",
                "sample_rate_hz": 16000,
                "channels": 1,
                "authority": "primary_audio",
                "time_range_ms": (300, 1600),
                "vad": {"speech_start_ms": 300, "speech_end_ms": 1600, "vad_speech": True},
                "energy_gate": {
                    "accepted": True,
                    "rms": 620,
                    "duration_ms": 1300,
                    "min_rms": 350,
                    "min_speech_ms": 120,
                },
            },
        },
        "witness_fusion_bundle_audio_metadata": {
            "early": {
                "codec": "pcm_s16le",
                "sample_rate_hz": 16000,
                "channels": 1,
                "authority": "primary_audio",
                "vad": {"speech_start_ms": 100, "speech_end_ms": 1400, "vad_speech": True},
                "energy_gate": {
                    "accepted": True,
                    "rms": 620,
                    "duration_ms": 1300,
                    "min_rms": 350,
                    "min_speech_ms": 120,
                },
            },
            "with": {
                "codec": "pcm_s16le",
                "sample_rate_hz": 16000,
                "channels": 1,
                "authority": "primary_audio",
                "time_range_ms": (200, 1500),
                "vad": {"speech_start_ms": 200, "speech_end_ms": 1500, "vad_speech": True},
                "energy_gate": {
                    "accepted": True,
                    "rms": 620,
                    "duration_ms": 1300,
                    "min_rms": 350,
                    "min_speech_ms": 120,
                },
            },
            "late": {
                "codec": "pcm_s16le",
                "sample_rate_hz": 16000,
                "channels": 1,
                "authority": "primary_audio",
                "time_range_ms": (300, 1600),
                "vad": {"speech_start_ms": 300, "speech_end_ms": 1600, "vad_speech": True},
                "energy_gate": {
                    "accepted": True,
                    "rms": 620,
                    "duration_ms": 1300,
                    "min_rms": 350,
                    "min_speech_ms": 120,
                },
            },
        },
        "witness_fusion_accepted_audio_gate_observed": True,
        "witness_fusion_early_initial_bundle_id": "kame-evidence-witness-early",
        "witness_fusion_early_final_bundle_id": "kame-evidence-witness-early",
        "witness_fusion_early_single_bundle": True,
        "witness_fusion_interpreter_prompt_input_order": [
            "raw_audio",
            "metadata",
            "reflex",
            "transcript_hypotheses",
        ],
        "witness_fusion_interpreter_prompt_input_order_expected": [
            "raw_audio",
            "metadata",
            "reflex",
            "transcript_hypotheses",
        ],
        "witness_fusion_interpreter_prompt_input_order_visible": True,
        "witness_fusion_interpreter_prompt_policy": {
            "version": "raw_audio_compare_v1",
            "primary_evidence": "raw_audio",
            "transcript_hypotheses_authority": "non_authoritative_context",
            "promotion_requirement": "compare_transcript_hypotheses_against_raw_audio_before_promotion",
            "forbidden_direct_uses": (
                "oracle_text",
                "durable_transcript",
                "spend_reason",
                "phone_call_payload",
                "tool_arguments",
            ),
        },
        "witness_fusion_interpreter_prompt_policy_expected": {
            "version": "raw_audio_compare_v1",
            "primary_evidence": "raw_audio",
            "transcript_hypotheses_authority": "non_authoritative_context",
            "promotion_requirement": "compare_transcript_hypotheses_against_raw_audio_before_promotion",
            "forbidden_direct_uses": (
                "oracle_text",
                "durable_transcript",
                "spend_reason",
                "phone_call_payload",
                "tool_arguments",
            ),
        },
        "witness_fusion_interpreter_prompt_policy_version": "raw_audio_compare_v1",
        "witness_fusion_interpreter_prompt_policy_visible": True,
        "energy_gate_smoke_ok": True,
        "energy_gate_policy": {"min_rms": 350, "min_speech_ms": 120},
        "energy_gate_ignored_packet_rms": 80,
        "energy_gate_ignored_packet_duration_ms": 200,
        "energy_gate_ignored_packet_speech_confirmed": False,
        "energy_gate_ignored_packet_vad_speech": False,
        "energy_gate_ignored_non_speech_packets": 3,
        "energy_gate_low_energy_witness_text": "spend money from room tone",
        "energy_gate_low_energy_witness_source": "moshi",
        "energy_gate_low_energy_witness_adjudication": "rejected_or_diagnostic_only",
        "energy_gate_low_energy_witness_rejection_reasons": ["low_energy_non_speech"],
        "energy_gate_low_energy_witness_authority": "hypothesis",
        "energy_gate_low_energy_witness_tool_authority": False,
        "energy_gate_low_energy_witness_promoted": False,
        "energy_gate_low_energy_witness_suppressed": True,
        "energy_gate_barge_in_events": 0,
        "energy_gate_interpreter_requests": 0,
        "energy_gate_oracle_work_events": 0,
        "energy_gate_oracle_requests": 0,
        "energy_gate_raw_packet_buffered_without_turn": True,
        "energy_gate_event_types": [
            "playback.started",
            "speech.start",
            "speech.energy",
            "audio.input.chunk",
        ],
        "kame_ack_latency_metrics_smoke_ok": True,
        "kame_defer_ack_first_audio_metrics_visible": True,
        "kame_local_first_audio_metrics_visible": True,
        "kame_defer_ack_metric_keys": [
            "kame_interface_decision_to_defer_first_audio_ms",
            "kame_speech_end_to_defer_first_audio_ms",
        ],
        "kame_local_first_audio_metric_keys": [
            "kame_interface_decision_to_local_first_audio_ms",
            "kame_speech_end_to_local_first_audio_ms",
        ],
        "kame_defer_ack_audio_metrics": {
            "kame_interface_decision_to_defer_first_audio_ms": 5,
            "kame_speech_end_to_defer_first_audio_ms": 46,
        },
        "kame_defer_ack_session_metrics": {
            "kame_interface_decision_to_defer_first_audio_ms": 5,
            "kame_speech_end_to_defer_first_audio_ms": 46,
        },
        "kame_local_first_audio_metrics": {
            "kame_interface_decision_to_local_first_audio_ms": 4,
            "kame_speech_end_to_local_first_audio_ms": 41,
        },
        "kame_local_session_metrics": {
            "kame_interface_decision_to_local_first_audio_ms": 4,
            "kame_speech_end_to_local_first_audio_ms": 41,
        },
        "kame_defer_speech_end_to_first_audio_ms": 46,
        "kame_local_speech_end_to_first_audio_ms": 41,
        "kame_defer_first_audio_bytes": 15,
        "kame_local_first_audio_bytes": 17,
        "kame_latency_breakdown_smoke_ok": True,
        "kame_latency_breakdown_required_segments": [
            "speech_end_to_reflex_ack_ms",
            "audio_cut_to_interpreter_submit_ms",
            "witness_arrival_ms",
            "interpreter_submit_to_promotion_ms",
            "promotion_to_oracle_start_ms",
            "oracle_start_to_first_token_ms",
            "first_token_to_tts_first_audio_ms",
            "tts_first_audio_to_playback_start_ms",
            "playback_start_to_completion_ms",
        ],
        "kame_latency_breakdown_segments_ms": {
            "speech_end_to_reflex_ack_ms": 42,
            "audio_cut_to_interpreter_submit_ms": 14,
            "witness_arrival_ms": 88,
            "interpreter_submit_to_promotion_ms": 261,
            "promotion_to_oracle_start_ms": 12,
            "oracle_start_to_first_token_ms": 178,
            "first_token_to_tts_first_audio_ms": 38,
            "tts_first_audio_to_playback_start_ms": 8,
            "playback_start_to_completion_ms": 344,
        },
        "kame_latency_breakdown_timeline_ms": {
            "speech_end": 0,
            "reflex_ack": 42,
            "audio_cut": 45,
            "interpreter_submit": 59,
            "witness_arrival": 88,
            "interpreter_promotion": 320,
            "oracle_start": 332,
            "oracle_first_token": 510,
            "tts_first_audio": 548,
            "playback_start": 556,
            "playback_completion": 900,
        },
        "kame_latency_breakdown_total_ms": 900,
        "kame_latency_breakdown_segment_total_ms": 985,
        "kame_latency_breakdown_monotonic": True,
        "reflex_ack_transcript_smoke_ok": True,
        "reflex_ack_transcript_visible": True,
        "reflex_ack_transcript_record": {
            "schema_version": "voiceops.reflex_ack_transcript.v1",
            "turn_id": "voiceops-demo-turn-budget",
            "oracle_job_id": "voice-oracle-voiceops-demo-001",
            "speaker": "assistant_reflex",
            "text": "I heard you. I am preparing the phone handoff approval.",
            "text_source": "reflex_acknowledgement",
            "authority": "reflex_hypothesis",
            "durability": "visible_transcript_and_audit",
            "visible_to_user": True,
            "spoken": True,
            "provisional": True,
            "action_authority": False,
            "tool_authority": False,
            "audit_event_id": "evt-reflex-ack-001",
        },
        "reflex_ack_transcript_audit_record": {
            "event_id": "evt-reflex-ack-001",
            "event": "reflex.ack.transcript_recorded",
            "turn_id": "voiceops-demo-turn-budget",
            "oracle_job_id": "voice-oracle-voiceops-demo-001",
            "text_source": "reflex_acknowledgement",
            "authority": "reflex_hypothesis",
            "action_authority": False,
            "tool_authority": False,
            "durability": "visible_transcript_and_audit",
        },
        "reflex_ack_text": "I heard you. I am preparing the phone handoff approval.",
        "reflex_ack_text_source": "reflex_acknowledgement",
        "reflex_ack_authority": "reflex_hypothesis",
        "reflex_ack_action_authority": False,
        "reflex_ack_tool_authority": False,
        "reflex_ack_durability": "visible_transcript_and_audit",
        "reflex_ack_turn_id": "voiceops-demo-turn-budget",
        "reflex_ack_oracle_job_id": "voice-oracle-voiceops-demo-001",
        "witness_fusion_with_bundle_id": "kame-evidence-witness-with",
        "witness_fusion_with_single_bundle": True,
        "witness_fusion_late_initial_bundle_id": "kame-evidence-witness-late",
        "witness_fusion_late_final_bundle_id": "kame-evidence-witness-late",
        "witness_fusion_late_single_bundle": True,
        "witness_fusion_no_duplicate_oracle_jobs": True,
        "witness_fusion_partial_superseded_by_final": True,
        "witness_fusion_partial_case_job_id": "voice-oracle-007",
        "witness_fusion_partial_blocker_job_id": "voice-oracle-006",
        "witness_fusion_partial_active_hypothesis": {
            "source": "moshi",
            "kind": "frontend_witness_hypothesis",
            "text": "what is three to the power of seventeen",
            "text_digest": _text_digest("what is three to the power of seventeen"),
            "role": "witness_context",
            "authority": "hypothesis",
            "promotion_required": "interpreter_promoted_or_oracle_promoted",
            "tool_authority": False,
            "confidence": 0.88,
            "arrival_phase": "with_raw_audio",
            "partial": False,
            "superseded_partial_texts": ("what is three to the",),
            "superseded_partial_count": 1,
        },
        "witness_fusion_adjudications": {
            "early": ["corrected_by_audio"],
            "with": ["accepted_as_supporting_evidence"],
            "late": ["rejected_or_diagnostic_only"],
        },
        "witness_fusion_rejection_reasons": {
            "early": [],
            "with": [],
            "late": ["ambiguous_speaker", "wrong_speaker", "wrong_channel", "stale_witness"],
        },
        "witness_fusion_adjudication_outcomes_observed": True,
        "witness_fusion_multi_speaker_witness_smoke_ok": True,
        "witness_fusion_multi_speaker_wrong_witness_rejected": True,
        "witness_fusion_multi_speaker_bound_to_second_human": True,
        "witness_fusion_multi_speaker_action_sinks_clean": True,
        "witness_fusion_multi_speaker_promoted_text": "prepare late witness handoff",
        "witness_fusion_accepted_counts": {"early": 1, "with": 1, "late": 1},
        "witness_fusion_started_counts": {"early": 1, "with": 1, "late": 1},
        "witness_fusion_completed_counts": {"early": 1, "with": 1, "late": 1},
        "runtime_kame_action_gate_smoke_ok": True,
        "runtime_kame_action_gate_waiting_events": 6,
        "runtime_kame_action_gate_hypothesis_only_ok": False,
        "runtime_kame_action_gate_hypothesis_only_issues": [
            "missing_promoted_evidence",
            "interpreter_evidence_not_consumed_before_irreversible_action",
        ],
        "runtime_kame_action_gate_hypothesis_only_rejected_authorities": [
            "hypothesis",
            "reflex_hypothesis",
        ],
        "runtime_kame_action_gate_degraded_text_only_ok": False,
        "runtime_kame_action_gate_degraded_text_only_issues": [
            "missing_promoted_evidence",
            "interpreter_evidence_not_consumed_before_irreversible_action",
            "degraded_text_only_cannot_authorize_high_risk_action",
        ],
        "runtime_kame_action_gate_degraded_text_only_rejected_authorities": [
            "hypothesis",
            "reflex_hypothesis",
        ],
        "runtime_kame_action_gate_degraded_text_only_status": "degraded_text_only",
        "runtime_kame_action_gate_degraded_text_only_reason": "degraded_text_only",
        "runtime_kame_action_gate_degraded_text_only_raw_audio_available": False,
        "runtime_kame_action_gate_degraded_text_only_preserves_hypothesis": True,
        "runtime_kame_action_gate_degraded_oracle_promoted_ok": False,
        "runtime_kame_action_gate_degraded_oracle_promoted_issues": [
            "degraded_text_only_cannot_authorize_high_risk_action",
        ],
        "runtime_kame_action_gate_degraded_oracle_promoted_authorities": ["oracle_promoted"],
        "runtime_kame_action_gate_degraded_oracle_promoted_rejected_authorities": [
            "hypothesis",
            "reflex_hypothesis",
        ],
        "runtime_kame_action_gate_degraded_oracle_promoted_status": "degraded_text_only",
        "runtime_kame_action_gate_degraded_oracle_promoted_raw_audio_available": False,
        "runtime_kame_action_gate_degraded_oracle_promoted_consumed_before_action": True,
        "runtime_kame_action_gate_promoted_ok": True,
        "runtime_kame_action_gate_promoted_issues": [],
        "runtime_kame_action_gate_promoted_authorities": ["interpreter_promoted"],
        "runtime_kame_action_gate_promoted_consumed_before_action": True,
        "runtime_kame_action_gate_self_attested_ok": False,
        "runtime_kame_action_gate_self_attested_issues": ["missing_promoted_evidence"],
        "runtime_kame_action_gate_self_attested_authorities": [],
        "runtime_kame_action_gate_self_attested_consumed_before_action": True,
        "runtime_kame_action_gate_missing_tool_disclosure_ok": False,
        "runtime_kame_action_gate_missing_tool_disclosure_issues": ["missing_tool_disclosure_ref"],
        "runtime_kame_action_gate_missing_tool_disclosure_authorities": ["interpreter_promoted"],
        "runtime_kame_action_gate_tool_disclosure_ref_observed": True,
        "runtime_kame_action_gate_schema_versions": [
            "voiceops.runtime_kame_action_gate.v1",
            "voiceops.runtime_kame_action_gate.v1",
            "voiceops.runtime_kame_action_gate.v1",
            "voiceops.runtime_kame_action_gate.v1",
            "voiceops.runtime_kame_action_gate.v1",
            "voiceops.runtime_kame_action_gate.v1",
        ],
        "witness_assisted_voiceops_action_smoke_ok": True,
        "witness_assisted_voiceops_action_gate_ok": True,
        "witness_assisted_voiceops_action_gate_authorities": ["interpreter_promoted"],
        "witness_assisted_voiceops_action_consumed_before_action": True,
        "witness_assisted_voiceops_action_single_bundle": True,
        "witness_assisted_voiceops_action_witness_text": "spend two hundred dollars and call my phone",
        "witness_assisted_voiceops_action_promoted_text": (
            "prepare Stripe approval for twenty dollars of phone credits and a phone handoff"
        ),
        "witness_assisted_voiceops_action_witness_authority": "hypothesis",
        "witness_assisted_voiceops_action_witness_role_context": True,
        "witness_assisted_voiceops_action_witness_tool_authority_false": True,
        "witness_assisted_voiceops_action_witness_adjudication": "corrected_by_audio",
        "witness_assisted_voiceops_action_promoted_authorities": ["interpreter_promoted"],
        "witness_assisted_voiceops_action_sink_keys_checked": [
            "spend_reason",
            "spend_payload",
            "provider_selection",
            "provider_choice",
            "provider_payload",
            "nemoclaw_action_packet",
            "nemoclaw_action_payload",
            "action_packet",
            "action_payload",
            "approval_payload",
            "phone_call_payload",
            "call_payload",
            "tool_arguments",
            "arguments",
            "memory_write",
            "file_write",
            "message_payload",
            "external_message",
            "durable_history",
            "durable_user_history",
            "durable_transcript",
        ],
        "witness_assisted_voiceops_action_sinks_clean": True,
        "witness_assisted_voiceops_action_sink_values": {
            "spend_reason": {
                "text": "prepare Stripe approval for twenty dollars of phone credits and a phone handoff",
                "authority": "interpreter_promoted",
            },
            "spend_payload": {
                "amount_usd": 20,
                "reason": "prepare Stripe approval for twenty dollars of phone credits and a phone handoff",
                "authority": "interpreter_promoted",
            },
            "phone_call_payload": {
                "script_summary": "Continue the promoted phone handoff context.",
                "authority": "interpreter_promoted",
            },
            "tool_arguments": {
                "reason": "prepare Stripe approval for twenty dollars of phone credits and a phone handoff",
                "authority": "interpreter_promoted",
            },
            "durable_history": {
                "summary": "prepare Stripe approval for twenty dollars of phone credits and a phone handoff",
                "authority": "interpreter_promoted",
            },
        },
        "witness_assisted_voiceops_action_raw_witness_absent": True,
        "witness_assisted_voiceops_action_promoted_text_present": True,
        "durable_resume_contract_smoke_ok": True,
        "durable_resume_contract_schema_version": "voiceops.kame_durable_resume_context.v1",
        "durable_resume_promoted_turn_count": 4,
        "durable_resume_recent_promoted_turns_verbatim": True,
        "durable_resume_recent_promoted_turns": [
            {
                "turn_id": "voice-smoke-durable-resume:3",
                "text": "promoted durable resume request 3",
                "source": "gemma_interpreter",
                "authority": "promoted",
            },
            {
                "turn_id": "voice-smoke-durable-resume:4",
                "text": "promoted durable resume request 4",
                "source": "gemma_interpreter",
                "authority": "promoted",
            },
        ],
        "durable_resume_older_turns_summarized": True,
        "durable_resume_older_promoted_turn_count": 2,
        "durable_resume_older_promoted_turn_summary": (
            "2 older promoted voice turn(s) summarized from durable oracle ledger: "
            "voice-smoke-durable-resume:1, voice-smoke-durable-resume:2."
        ),
        "durable_resume_hypothesis_replay_absent": True,
        "durable_resume_ledger_authoritative": True,
        "hypothesis_final_durable_message_smoke_ok": True,
        "hypothesis_final_durable_messages_empty": True,
        "hypothesis_final_durable_message_count": 0,
        "hypothesis_final_without_adapter_flag_non_durable": True,
        "hypothesis_final_witness_intent_non_durable": True,
        "explicit_asr_fallback_final_remains_durable": True,
        "explicit_asr_fallback_durable_messages": [
            {"role": "user", "content": "check deployment status"}
        ],
        "audit_scalar_smoke_ok": True,
        "audit_scalar_payload_redacted": True,
        "audit_scalar_secret_canary_checked": True,
        "audit_scalar_result_text_omitted": True,
        "audit_scalar_completed_event_seen": True,
        "audit_scalar_waiting_event_seen": True,
        "audit_scalar_row_count": 5,
        "spoken": [
            "Starting smoke task 1.",
            "Starting smoke task 2.",
            "Starting smoke task 3.",
            "Starting smoke task 4.",
            "Yes, I can hear you.",
            "Finished Run smoke task 1.",
            "Finished Run smoke task 2.",
            "Finished Run smoke task 4.",
            "Preparing spend approval.",
            "Oracle jobs: 1 active out of 4, 0 running, 1 waiting for approval. "
            "waiting_for_approval: Preparing spend approval.",
            "Approval smoke cleared.",
            "Testing failure handling.",
            "I couldn't finish Fail smoke task: smoke oracle failure",
            "Still listening.",
            "Working on the plan.",
            "First sentence.",
        ],
        "event_counts": {
            "oracle.job.started": 9,
            "oracle.job.completed": 6,
            "oracle.job.failed": 1,
            "oracle.job.cancelled": 1,
            "oracle.job.waiting_for_approval": 1,
            "interface.oracle.update": 1,
            "assistant.commit": 7,
        },
    }


def _discord_session_cleanup_smoke_payload() -> dict:
    return {
        "ok": True,
        "scenario": "discord_session_cleanup_fake_sidecar",
        "discord_network": False,
        "provider_sidecar_network": False,
        "cancel_all_before_session_closed": True,
        "cancel_payload": {
            "job_id": "all",
            "all": True,
            "reason": "voice session closing",
            "transport": "discord_voice",
        },
        "session_closed_sent": True,
        "sidecar_closed": True,
        "sidecar_close_calls": 1,
        "degraded_active_job_preserved_failed": True,
        "degraded_session_removed": True,
        "degraded_fallback_reason": "sidecar_event_stream_closed: sidecar event stream closed",
        "degraded_job_state": "failed",
        "degraded_job_error": "sidecar_event_stream_closed: sidecar event stream closed",
        "event_order": ["interface.oracle.cancel", "session.closed"],
    }


def _sidecar_fail_closed_smoke_payload() -> dict:
    return {
        "ok": True,
        "scenario": "sidecar_send_fail_closed_after_acceptance",
        "discord_network": False,
        "provider_sidecar_network": False,
        "fallback_policy": "fail_closed",
        "request_accepted": True,
        "job_id": "voice-oracle-001",
        "cancelled_observed": True,
        "cancel_reason": "sidecar_send_failed",
        "session_error_observed": True,
        "session_error_reason": "sidecar_send_failed",
        "session_error_sidecar": False,
        "error_redacted": True,
        "error_mentions_fail_closed": True,
        "error_mentions_send_failed": True,
        "active_capacity_after_failure": 0,
        "job_state_after_failure": "cancelled",
        "sidecar_removed": True,
        "sidecar_closed": True,
        "sidecar_close_calls": 1,
        "oracle_requests_seen": 1,
        "event_order": ["oracle.job.cancelled", "session.error"],
        "test_refs": [
            "tests/agent/test_realtime_voice.py::test_text_engine_fail_closed_policy_emits_session_error_on_sidecar_session_error",
            "tests/agent/test_realtime_voice.py::test_text_engine_fail_closed_policy_emits_session_error_on_sidecar_event_stream_failure",
            "tests/agent/test_realtime_voice.py::test_kame_engine_fail_closed_sidecar_send_failure_cancels_external_oracle_job",
        ],
    }


def _voice_operator_report(live_evidence: dict | None = None, smoke: dict | None = None) -> dict:
    return build_voice_operator_report(
        smoke or _smoke_payload(),
        live_evidence=live_evidence,
        async_oracle_smoke=_async_oracle_smoke_payload(),
        discord_session_cleanup_smoke=_discord_session_cleanup_smoke_payload(),
        sidecar_fail_closed_smoke=_sidecar_fail_closed_smoke_payload(),
    )


def _collector_attestation(section_name: str) -> dict:
    return {
        "collector_name": "pytest.voiceops_live_fixture",
        "collector_version": "voiceops-live-fixture-v1",
        "run_id": f"pytest-{section_name}",
        "command_argv": ["pytest", section_name],
        "git_commit": "abc123def456",
        "started_at": "2026-06-29T00:00:00Z",
        "finished_at": "2026-06-29T00:00:01Z",
        "raw_artifact_sha256": "a" * 64,
        "redacted_artifact_sha256": "b" * 64,
        "parent_manifest_sha256": "c" * 64,
    }


def _payload_sha256(payload: dict) -> str:
    attested_payload = dict(payload)
    attested_payload.pop("collector_attestation", None)
    attested_payload.pop("collector_provenance", None)
    raw = json.dumps(attested_payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _attest_section(section: dict, section_name: str) -> dict:
    section["collector_attestation"] = _collector_attestation(section_name)
    payload_sha256 = _payload_sha256(section)
    section["collector_attestation"]["raw_artifact_sha256"] = payload_sha256
    section["collector_attestation"]["redacted_artifact_sha256"] = payload_sha256
    section["collector_attestation"]["parent_manifest_sha256"] = payload_sha256
    return section


def _write_attested_section(path: Path, section: dict, section_name: str) -> None:
    _attest_section(section, section_name)
    path.write_text(json.dumps(section), encoding="utf-8")


def _complete_live_turn_fields(*, speech_end_to_first_audio_ms: int = 950, barge_in_stop_ms: int = 80) -> dict:
    return {
        "turn_id": "voiceops-live-turn-budget",
        "audio_segment_ref": "artifact://redacted/voiceops-live-turn-budget.wav",
        "evidence_bundle_id": "kame-evidence-live-turn-budget",
        "evidence_merge_key": "kame-merge-live-turn-budget",
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
        "interpreter_prompt_policy": {
            "version": "raw_audio_compare_v1",
            "primary_evidence": "raw_audio",
            "transcript_hypotheses_authority": "non_authoritative_context",
        },
        "transcript_hypotheses": [
            {
                "kind": "frontend_witness_hypothesis",
                "source": "moshi",
                "text": "[redacted witness hypothesis]",
                "text_digest": hashlib.sha256(b"redacted witness hypothesis").hexdigest(),
                "role": "witness_context",
                "arrival_phase": "with_raw_audio",
                "adjudication": "corrected_by_audio",
                "authority": "hypothesis",
                "promotion_required": "interpreter_promoted_or_oracle_promoted",
                "tool_authority": False,
                "latency_ms": 140,
                "confidence": 0.78,
                "speaker_or_actor_ref": "discord:user:jetha-redacted",
                "channel_or_surface_ref": "discord_voice:guild-redacted:general-redacted",
            }
        ],
        "interpreter_adjudication_outcomes": ["corrected_by_audio"],
        "promoted_evidence_authority": {
            "interpreter_corrected_transcript": "interpreter_promoted",
            "interpreter_normalized_intent": "interpreter_promoted",
        },
        "unpromoted_witness_sink_checks": {
            "spend_clean": True,
            "phone_clean": True,
            "nemoclaw_clean": True,
            "tool_clean": True,
            "memory_clean": True,
            "file_clean": True,
            "message_clean": True,
            "durable_history_clean": True,
        },
        "unpromoted_witness_sink_values": {},
        "assistant_audio_observed": True,
        "barge_in_observed": True,
        "spoken_reply_short": True,
        "no_voice_denial_observed": True,
        "speech_end_to_first_audio_ms": speech_end_to_first_audio_ms,
        "barge_in_stop_ms": barge_in_stop_ms,
    }


def _complete_live_evidence() -> dict:
    evidence = build_live_probe_evidence_template()
    evidence["discord_live_probe"].update(
        {
            "collector_attestation": _collector_attestation("discord_live_probe"),
            "ok": True,
            "connect_perm": True,
            "speak_perm": True,
            "connected": True,
            "opus_loaded": True,
            "accepted_audio_source": True,
            "played": True,
            "playing_during_probe": True,
            "receiver_started": True,
            "receiver_frames": 12,
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
        }
    )
    evidence["sidecar_session"].update(
        {
            "collector_attestation": _collector_attestation("sidecar_session"),
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
            "latency_metrics_ms": {"session_start_ms": 110, "shutdown_ms": 80},
        }
    )
    evidence["live_turn"].update(
        {
            "collector_attestation": _collector_attestation("live_turn"),
            **_complete_live_turn_fields(speech_end_to_first_audio_ms=900, barge_in_stop_ms=90),
        }
    )
    return evidence


def _complete_discord_latency_metrics() -> dict:
    return {
        "connect_ms": 420,
        "playback_observed_ms": 180,
        "inbound_observed_ms": 900,
        "disconnect_ms": 120,
    }


def _complete_sidecar_session_fields() -> dict:
    return {
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
        "latency_metrics_ms": {"session_start_ms": 110, "shutdown_ms": 80},
    }


def test_voice_operator_report_maps_loopback_smoke_to_milestone_1_contract():
    report = _voice_operator_report()

    assert report["schema_version"] == "voiceops.milestone1.voice_operator.v1"
    assert report["artifact_only"] is True
    assert report["status"] == "needs_live_probe"
    assert report["missing_live_gates"] == [
        "discord_join",
        "discord_playback",
        "live_receiver",
        "production_sidecar",
        "live_turn",
    ]
    assert report["mode"] == {
        "bounded": True,
        "discord_network": False,
        "env_secret_reads": False,
        "headless": True,
        "outbound_calls": False,
        "outbound_sends": False,
        "provider_sidecar_network": False,
    }
    assert validate_voice_operator_report(report) == []
    assert report["requirements"]["stable_discord_receive_playback_lifecycle"] is True
    assert report["requirements"]["receiver_callback_wiring"] is True
    assert report["requirements"]["pcm_conversion_correctness"] is True
    assert report["requirements"]["mixer_playback_path"] is True
    assert report["requirements"]["barge_in_behavior"] is True
    assert report["requirements"]["latency_metrics"] is True
    assert report["requirements"]["sidecar_session_shutdown"] is True
    assert report["requirements"]["async_oracle_four_concurrent_jobs"] is True
    assert report["requirements"]["async_oracle_local_turn_while_running"] is True
    assert report["requirements"]["async_oracle_status_turn_while_running"] is True
    assert report["requirements"]["async_oracle_status_ordinal_labels_visible"] is True
    assert report["requirements"]["async_oracle_status_bounded_overflow_visible"] is True
    assert report["requirements"]["async_oracle_fifth_job_queued_and_started"] is True
    assert report["requirements"]["async_oracle_cancellation_isolated"] is True
    assert report["requirements"]["async_oracle_playback_stop_preserves_jobs"] is True
    assert report["requirements"]["async_oracle_approval_wait_holds_capacity"] is True
    assert report["requirements"]["async_oracle_cancel_drain_holds_capacity"] is True
    assert report["requirements"]["async_oracle_late_cancelled_output_attempted"] is True
    assert report["requirements"]["async_oracle_late_cancelled_output_dropped"] is True
    assert report["requirements"]["async_oracle_late_cancelled_output_not_durable"] is True
    assert report["requirements"]["progressive_tool_disclosure"] is True
    assert report["requirements"]["live_discord_join"] is False
    assert report["requirements"]["live_evidence_supplied"] is False
    assert report["proofs"]["lifecycle"]["sidecar_closed"] is True
    assert report["proofs"]["callback_wiring"]["loopback_bypasses_live_discord_receiver"] is True
    assert report["proofs"]["pcm_conversion"]["sidecar_pcm16_first_sample"] == 450
    assert report["proofs"]["barge_in_energy"]["ok"] is True
    assert report["proofs"]["barge_in_energy"]["speech_energy_event_forwarded"] is True
    assert report["proofs"]["barge_in_energy"]["energy_gate_proven_by_smoke"] is True
    assert report["proofs"]["barge_in_energy"]["energy_gate_covered_by_tests"] is True
    assert report["proofs"]["barge_in_energy"]["energy_gate_ignored_non_speech_packets"] >= 2
    assert report["proofs"]["barge_in_energy"]["energy_gate_barge_in_events"] == 0
    assert report["proofs"]["barge_in_energy"]["energy_gate_oracle_work_events"] == 0
    assert report["proofs"]["shutdown"]["close_timeout_bounded"] is True
    assert report["proofs"]["async_oracle_jobs"]["ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["kind"] == "async_oracle_smoke"
    assert report["proofs"]["async_oracle_jobs"]["scenario"] == "async_kame_oracle_jobs_fake"
    assert report["proofs"]["async_oracle_jobs"]["max_running"] == 4
    assert report["proofs"]["async_oracle_jobs"]["queued_jobs"] == 1
    assert report["proofs"]["async_oracle_jobs"]["failed_jobs"] == 2
    assert report["proofs"]["async_oracle_jobs"]["queued_cancel_smoke_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["queued_cancel_observed"] is True
    assert report["proofs"]["async_oracle_jobs"]["queued_cancelled_before_start"] is True
    assert report["proofs"]["async_oracle_jobs"]["queued_cancel_not_sent_to_oracle"] is True
    assert report["proofs"]["async_oracle_jobs"]["queued_cancel_reason"] == "spoken request to cancel oracle job"
    assert report["proofs"]["async_oracle_jobs"]["queued_cancel_target_job_id"] == "voice-oracle-002"
    assert report["proofs"]["async_oracle_jobs"]["queued_cancel_running_completed"] is True
    assert report["proofs"]["async_oracle_jobs"]["approval_capacity_smoke_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["approval_capacity_waiting_observed"] is True
    assert report["proofs"]["async_oracle_jobs"]["approval_capacity_followup_queued"] is True
    assert report["proofs"]["async_oracle_jobs"]["approval_capacity_active_visible"] is True
    assert report["proofs"]["async_oracle_jobs"]["approval_capacity_misleading_running_capacity"] is False
    assert "1 active out of 1" in report["proofs"]["async_oracle_jobs"]["approval_capacity_status_text"]
    assert "0 running out of 1" not in report["proofs"]["async_oracle_jobs"]["approval_capacity_status_text"]
    assert "1 queued" in report["proofs"]["async_oracle_jobs"]["approval_capacity_status_text"]
    assert "1 waiting for approval" in report["proofs"]["async_oracle_jobs"]["approval_capacity_status_text"]
    assert report["proofs"]["async_oracle_jobs"]["approval_capacity_followup_started_after_approval"] is True
    assert report["proofs"]["async_oracle_jobs"]["approval_capacity_completed_jobs"] == 1
    assert report["proofs"]["async_oracle_jobs"]["approval_capacity_failed_gate_suppressed"] is True
    assert report["proofs"]["async_oracle_jobs"]["approval_capacity_failed_jobs"] == 1
    assert report["proofs"]["async_oracle_jobs"]["approval_capacity_max_concurrent"] == 1
    assert report["proofs"]["async_oracle_jobs"]["cancel_drain_capacity_smoke_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["cancel_drain_requested_observed"] is True
    assert report["proofs"]["async_oracle_jobs"]["cancel_drain_cancelled_observed"] is True
    assert report["proofs"]["async_oracle_jobs"]["cancel_drain_followup_queued"] is True
    assert report["proofs"]["async_oracle_jobs"]["cancel_drain_active_visible"] is True
    assert report["proofs"]["async_oracle_jobs"]["cancel_drain_misleading_running_capacity"] is False
    assert "1 active out of 1" in report["proofs"]["async_oracle_jobs"]["cancel_drain_status_text"]
    assert "0 running out of 1" not in report["proofs"]["async_oracle_jobs"]["cancel_drain_status_text"]
    assert "1 queued" in report["proofs"]["async_oracle_jobs"]["cancel_drain_status_text"]
    assert "1 cancelling" in report["proofs"]["async_oracle_jobs"]["cancel_drain_status_text"]
    assert report["proofs"]["async_oracle_jobs"]["cancel_drain_followup_started_after_cancel"] is True
    assert report["proofs"]["async_oracle_jobs"]["cancel_drain_max_concurrent"] == 1
    assert report["proofs"]["async_oracle_jobs"]["playback_stop_committed"] is True
    assert report["proofs"]["async_oracle_jobs"]["playback_stop_jobs_still_running"] is True
    assert report["proofs"]["async_oracle_jobs"]["playback_stop_cancelled_jobs"] is False
    assert report["proofs"]["async_oracle_jobs"]["playback_stop_does_not_cancel_jobs"] is True
    assert report["proofs"]["async_oracle_jobs"]["status_turn_committed"] is True
    assert report["proofs"]["async_oracle_jobs"]["terminal_status_committed"] is True
    assert report["proofs"]["async_oracle_jobs"]["completed_result_status_visible"] is True
    assert "completed: First sentence. Second sentence. Third sentence." in report["proofs"]["async_oracle_jobs"][
        "terminal_status_text"
    ]
    assert report["proofs"]["async_oracle_jobs"]["fifth_job_queued"] is True
    assert report["proofs"]["async_oracle_jobs"]["fifth_job_started_after_capacity_freed"] is True
    assert report["proofs"]["async_oracle_jobs"]["late_cancelled_output_attempted"] is True
    assert report["proofs"]["async_oracle_jobs"]["cancelled_result_spoken"] is False
    assert report["proofs"]["async_oracle_jobs"]["cancelled_result_committed"] is False
    assert report["proofs"]["async_oracle_jobs"]["cancelled_result_progress_leaked"] is False
    assert report["proofs"]["async_oracle_jobs"]["cancelled_result_durable_completed"] is False
    assert report["proofs"]["async_oracle_jobs"]["cancelled_result_durable_text"] is False
    assert report["proofs"]["async_oracle_jobs"]["durable_cancelled_record_present"] is True
    assert report["proofs"]["async_oracle_jobs"]["durable_completed_jobs"] == report["proofs"]["async_oracle_jobs"][
        "completed_jobs"
    ]
    assert report["proofs"]["async_oracle_jobs"]["approval_wait_observed"] is True
    assert report["proofs"]["async_oracle_jobs"]["approval_status_committed"] is True
    assert report["proofs"]["async_oracle_jobs"]["approval_tool_progress_observed"] is True
    assert report["proofs"]["async_oracle_jobs"]["approval_tool_progress_kame_gate_present"] is True
    assert (
        report["proofs"]["async_oracle_jobs"]["approval_tool_progress_kame_gate_schema_version"]
        == "voiceops.runtime_kame_action_gate.v1"
    )
    assert report["proofs"]["async_oracle_jobs"]["approval_tool_progress_kame_gate_failed_closed"] is True
    assert "missing_promoted_evidence" in report["proofs"]["async_oracle_jobs"][
        "approval_tool_progress_kame_gate_issues"
    ]
    assert report["proofs"]["async_oracle_jobs"]["approval_payload_redacted"] is True
    assert report["proofs"]["async_oracle_jobs"]["approval_completed"] is False
    assert report["proofs"]["async_oracle_jobs"]["approval_gate_failed_closed"] is True
    assert report["proofs"]["async_oracle_jobs"]["approval_result_suppressed"] is True
    assert "waiting_for_approval: Preparing spend approval." in report["proofs"]["async_oracle_jobs"][
        "approval_status_text"
    ]
    assert report["proofs"]["async_oracle_jobs"]["failed_job_reported"] is True
    assert report["proofs"]["async_oracle_jobs"]["failed_job_spoken"] is True
    assert report["proofs"]["async_oracle_jobs"]["durable_failed_record_present"] is True
    assert report["proofs"]["async_oracle_jobs"]["session_survived_failed_job"] is True
    assert report["proofs"]["async_oracle_jobs"]["queued_job_update_observed"] is True
    assert report["proofs"]["async_oracle_jobs"]["running_job_update_observed"] is True
    assert report["proofs"]["async_oracle_jobs"]["running_update_latest_update_visible"] is True
    assert report["proofs"]["async_oracle_jobs"]["running_update_latest_update_text"] == "include running update context"
    assert report["proofs"]["async_oracle_jobs"]["running_update_reached_oracle"] is True
    assert report["proofs"]["async_oracle_jobs"]["running_update_delivery_metadata_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["queued_update_latest_update_visible"] is True
    assert report["proofs"]["async_oracle_jobs"]["queued_update_latest_update_text"] == "include smoke update context"
    assert report["proofs"]["async_oracle_jobs"]["queued_update_started_with_priority"] is True
    assert report["proofs"]["async_oracle_jobs"]["queued_update_reached_oracle"] is True
    assert report["proofs"]["async_oracle_jobs"]["queued_interpreter_fold_in_observed"] is True
    assert (
        report["proofs"]["async_oracle_jobs"]["queued_interpreter_fold_in_oracle_text"]
        == "run corrected smoke task five"
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["queued_interpreter_fold_in_transcript_source"]
        == "gemma_interpreter"
    )
    assert report["proofs"]["async_oracle_jobs"]["queued_interpreter_fold_in_transcript_confidence"] == 0.88
    assert (
        report["proofs"]["async_oracle_jobs"]["queued_interpreter_fold_in_evidence_authority"][
            "oracle_text"
        ]
        == "interpreter_promoted"
    )
    assert report["proofs"]["async_oracle_jobs"]["verbose_result_spoken_bounded"] is True
    assert report["proofs"]["async_oracle_jobs"]["verbose_result_committed_bounded"] is True
    assert report["proofs"]["async_oracle_jobs"]["verbose_result_commit_marked_truncated"] is True
    assert report["proofs"]["async_oracle_jobs"]["verbose_full_result_durable"] is True
    assert report["proofs"]["async_oracle_jobs"]["verbose_spoken_result"] == "First sentence."
    assert report["proofs"]["async_oracle_jobs"]["terminal_result_policy_smoke_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["terminal_result_auto_summarize_default"] is True
    assert report["proofs"]["async_oracle_jobs"]["terminal_result_default_event_count"] == 1
    assert report["proofs"]["async_oracle_jobs"]["terminal_result_default_spoken"] is True
    assert (
        report["proofs"]["async_oracle_jobs"]["terminal_result_suppression_config"]
        == "oracle_jobs.speak_terminal_results=false"
    )
    assert report["proofs"]["async_oracle_jobs"]["terminal_result_suppressed"] is True
    assert report["proofs"]["async_oracle_jobs"]["terminal_result_suppressed_event_observed"] is True
    assert report["proofs"]["async_oracle_jobs"]["terminal_result_suppressed_event_count"] == 1
    assert (
        report["proofs"]["async_oracle_jobs"]["terminal_result_suppressed_reason"]
        == "terminal_speech_disabled"
    )
    assert report["proofs"]["async_oracle_jobs"]["terminal_result_suppressed_payload_clean"] is True
    assert report["proofs"]["async_oracle_jobs"]["terminal_result_unsolicited_event_count"] == 0
    assert report["proofs"]["async_oracle_jobs"]["terminal_result_unsolicited_spoken"] is False
    assert report["proofs"]["async_oracle_jobs"]["terminal_result_status_available"] is True
    assert "completed: Finished Suppress terminal result." in report["proofs"]["async_oracle_jobs"][
        "terminal_result_status_text"
    ]
    assert report["proofs"]["async_oracle_jobs"]["unflagged_high_risk_tool_smoke_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["unflagged_high_risk_tool_case_count"] == 9
    assert set(report["proofs"]["async_oracle_jobs"]["unflagged_high_risk_tool_categories"]) == {
        "credential",
        "file",
        "memory",
        "message",
        "phone",
        "provisioning",
        "shell",
        "spend",
    }
    assert report["proofs"]["async_oracle_jobs"]["unflagged_high_risk_tool_all_cases_failed_closed"] is True
    assert report["proofs"]["async_oracle_jobs"]["unflagged_high_risk_tool_all_progress_suppressed"] is True
    assert report["proofs"]["async_oracle_jobs"]["unflagged_high_risk_tool_all_payloads_redacted"] is True
    assert report["proofs"]["async_oracle_jobs"]["unflagged_high_risk_tool_all_spoken_payloads_clean"] is True
    assert all(
        case["ok"] is True
        for case in report["proofs"]["async_oracle_jobs"]["unflagged_high_risk_tool_cases"]
    )
    assert report["proofs"]["async_oracle_jobs"]["unflagged_high_risk_tool_suppressed"] is True
    assert report["proofs"]["async_oracle_jobs"]["unflagged_high_risk_tool_failed_closed"] is True
    assert (
        report["proofs"]["async_oracle_jobs"]["unflagged_high_risk_tool_suppression_reason"]
        == "unapproved_high_risk_tool_event"
    )
    assert report["proofs"]["async_oracle_jobs"]["unflagged_high_risk_tool_progress_suppressed"] is True
    assert report["proofs"]["async_oracle_jobs"]["unflagged_high_risk_tool_payload_redacted"] is True
    assert report["proofs"]["async_oracle_jobs"]["unflagged_high_risk_tool_spoken_payload_clean"] is True
    assert report["proofs"]["async_oracle_jobs"]["unflagged_high_risk_tool_failure_spoken"] is True
    assert report["proofs"]["async_oracle_jobs"]["unflagged_high_risk_tool_secret_canary_checked"] is True
    assert "dispatch_action" in report["proofs"]["async_oracle_jobs"]["unflagged_high_risk_tool_names"]
    assert report["proofs"]["async_oracle_jobs"]["unflagged_high_risk_tool_name"] == "write_memory"
    assert report["proofs"]["async_oracle_jobs"]["unflagged_high_risk_tool_spoken"][0] == (
        "Preparing the spend request."
    )
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_bridge_smoke_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_request_accepted"] is True
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_tool_result_observed"] is True
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_protocol"] == "kame_session_v1"
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_protocol_contract"]
        == "docs/kame-session-v1.md"
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_mode"]
        == "witness_assisted_direct_audio"
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_interpreter_profile"]
        == "witness_assisted_direct_audio"
    )
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_interpreter_input_order"] == [
        "raw_audio",
        "metadata",
        "reflex",
        "transcript_hypotheses",
    ]
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_witness_direct_audio_profile_ok"]
        is True
    )
    assert report["requirements"]["async_oracle_minimum_interpreter_packet_canonical"] is True
    assert report["async_oracle_coverage"]["minimum_interpreter_packet_canonical"] is True
    assert report["async_oracle_acceptance"]["minimum_interpreter_packet_is_canonical"]["ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["minimum_interpreter_packet_smoke_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["minimum_interpreter_packet_text_redacted"] is True
    assert report["proofs"]["async_oracle_jobs"]["minimum_interpreter_packet_raw_audio_primary"] is True
    assert report["proofs"]["async_oracle_jobs"]["minimum_interpreter_packet_hypotheses_authority"] is True
    assert report["proofs"]["async_oracle_jobs"]["minimum_interpreter_packet_input_order"] == [
        "raw_audio",
        "metadata",
        "reflex",
        "transcript_hypotheses",
    ]
    minimum_packet = report["proofs"]["async_oracle_jobs"]["minimum_interpreter_packet"]
    assert minimum_packet["schema_version"] == "voiceops.minimum_interpreter_packet.v1"
    assert minimum_packet["mode"] == "witness_assisted_direct_audio"
    assert minimum_packet["audio_segment_ref"] == "artifact://voiceclaw/turn-1.wav"
    assert minimum_packet["metadata"]["energy_gate"] == "accepted"
    assert minimum_packet["reflex"]["authority"] == "reflex_hypothesis"
    assert minimum_packet["reflex"]["tool_authority"] is False
    assert minimum_packet["transcript_hypotheses"][0]["text_redacted"] is True
    assert "text" not in minimum_packet["transcript_hypotheses"][0]
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_witness_adjudications"] == [
        {
            "source": "moshi",
            "kind": "frontend_witness_hypothesis",
            "text_digest": _text_digest("prepare an external came hand off"),
            "adjudication": "corrected_by_audio",
        },
        {
            "source": "voiceclaw",
            "kind": "frontend_witness_hypothesis",
            "text_digest": _text_digest("prepare an external came hand off"),
            "adjudication": "corrected_by_audio",
        },
    ]
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_interpreter_promoted"] == {
        "corrected_transcript": "prepare an external KAME handoff",
        "normalized_intent": "Prepare external KAME handoff",
        "confidence": 0.91,
        "authority": "interpreter_promoted",
    }
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_provider"] == "voiceclaw"
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_tool"] == "ask_brain"
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_completion_tool_call_id"]
        == "voiceclaw-call-1"
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_status_tool_call_id"]
        == "voiceclaw-call-1"
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_terminal_correlation_observed"]
        is True
    )
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_audit_id"] == "voiceclaw-audit-001"
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_source_audit_id"]
        == "discord-audit-voice-001"
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_parent_audit_id"]
        == "discord-audit-root-001"
    )
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_status_audit_id"] == "voiceclaw-audit-001"
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_completion_audit_id"]
        == "voiceclaw-audit-001"
    )
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_audit_id_continuity_observed"] is True
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_status_state"] == "completed"
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_source_reached_oracle"] is True
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_input_source"] == "ask_brain"
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_promoted_request_summary"] == {
        "text": "Prepare external KAME handoff",
        "source": "gemma_interpreter",
        "authority": "interpreter_promoted",
        "tool_authority": False,
    }
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_provisional_request_summary"] == {
        "text": "Prepare external KAME handoff",
        "source": "reflex_audio",
        "kind": "reflex_hypothesis",
        "authority": "hypothesis",
        "tool_authority": False,
    }
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_status_provisional_request_summary"] == {
        "text": "Prepare external KAME handoff",
        "source": "reflex_audio",
        "kind": "reflex_hypothesis",
        "authority": "hypothesis",
        "tool_authority": False,
    }
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_provisional_request_summary_non_authoritative"]
        is True
    )
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_evidence_bundle_propagated"] is True
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_evidence_merge_key"].startswith(
        "kame-merge-"
    )
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_evidence_merge_key_propagated"] is True
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_audio_segment_ref"]
        == "artifact://voiceclaw/turn-1.wav"
    )
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_audio_time_range_ms"] == [100, 2100]
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_transcript_hypotheses"][0][
            "authority"
        ]
        == "hypothesis"
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_transcript_hypotheses"][0][
            "role"
        ]
        == "witness_context"
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_transcript_hypotheses"][0][
            "promotion_required"
        ]
        == "interpreter_promoted_or_oracle_promoted"
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_transcript_hypotheses"][0][
            "kind"
        ]
        == "frontend_witness_hypothesis"
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_auxiliary_transcript_hypotheses"]
        == report["proofs"]["async_oracle_jobs"]["external_frontend_transcript_hypotheses"]
    )
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_witness_kind"] == "frontend_witness_hypothesis"
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_witness_kind_frontend_hypothesis"]
        is True
    )
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_witness_metadata_complete"] is True
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_witness_confidence"] == 0.78
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_witness_latency_ms"] == 140
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_witness_partial"] is False
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_witness_audio_time_range_ms"] == [
        120,
        2080,
    ]
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_witness_speaker"][
            "channel_user_id"
        ]
        == "jetha-redacted"
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_witness_channel"]["channel_id"]
        == "general-redacted"
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_auxiliary_transcript_hypotheses"][0][
            "tool_authority"
        ]
        is False
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_transcript_hypotheses"][0][
            "adjudication"
        ]
        == "corrected_by_audio"
    )
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_witness_tool_authority_false"] is True
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_witness_role_context"] is True
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_witness_promotion_required"] is True
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_hypothesis_not_durable_oracle_text"] is True
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_durable_user_messages_empty"] is True
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_durable_oracle_text_absent"] is True
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_durable_record_count"] == 2
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_direct_tool_authority_exposed"] is False
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_direct_tool_rejected"] is True
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_direct_tool_rejected_tool"]
        == "stripe_link_purchase"
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_direct_tool_rejection_reason"]
        == "unsupported_external_kame_tool"
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["external_frontend_direct_tool_created_oracle_job"]
        is False
    )
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_tool_result_payload_safe"] is True
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_reflex_status_payload_safe"] is True
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_placeholder_payload_safe"] is True
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_tool_result_forbidden_paths"] == []
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_reflex_status_forbidden_paths"] == []
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_placeholder_forbidden_paths"] == []
    assert report["proofs"]["async_oracle_jobs"]["external_frontend_placeholder"].startswith(
        "Accepted job one"
    )
    assert report["requirements"]["async_oracle_external_frontend_bridge"] is True
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_smoke_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_source"] == "moshi"
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_authority"] == "hypothesis"
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_tool_authority"] is False
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_tool_authority_false"] is True
    assert (
        report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_text"]
        == "spend two hundred dollars and call my phone"
    )
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_oracle_text_preserved"] is True
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_transcript_preserved"] is True
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_intent_preserved"] is True
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_attached"] is True
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_promoted"] is False
    assert set(report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_action_sink_keys_checked"]) >= {
        "spend_reason",
        "spend_payload",
        "provider_selection",
        "provider_choice",
        "provider_payload",
        "nemoclaw_action_packet",
        "nemoclaw_action_payload",
        "action_packet",
        "action_payload",
        "approval_payload",
        "phone_call_payload",
        "call_payload",
        "tool_arguments",
        "arguments",
        "memory_write",
        "file_write",
        "message_payload",
        "external_message",
    }
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_action_sinks_clean"] is True
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_action_sink_values"] == {}
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_not_spend_reason"] is True
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_not_spend_payload"] is True
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_not_provider_selection"] is True
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_not_nemoclaw_action_packet"] is True
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_not_phone_call_payload"] is True
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_not_call_payload"] is True
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_not_tool_arguments"] is True
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_not_memory_write"] is True
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_not_file_write"] is True
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_not_message_payload"] is True
    assert report["proofs"]["async_oracle_jobs"]["unpromoted_hypothesis_update_observed"] is True
    assert report["proofs"]["async_oracle_jobs"]["witness_assisted_voiceops_action_smoke_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["witness_assisted_voiceops_action_gate_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["witness_assisted_voiceops_action_gate_authorities"] == [
        "interpreter_promoted"
    ]
    assert (
        report["proofs"]["async_oracle_jobs"]["witness_assisted_voiceops_action_consumed_before_action"]
        is True
    )
    assert report["proofs"]["async_oracle_jobs"]["witness_assisted_voiceops_action_single_bundle"] is True
    assert (
        report["proofs"]["async_oracle_jobs"]["witness_assisted_voiceops_action_witness_authority"]
        == "hypothesis"
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["witness_assisted_voiceops_action_witness_role_context"]
        is True
    )
    assert (
        report["proofs"]["async_oracle_jobs"][
            "witness_assisted_voiceops_action_witness_tool_authority_false"
        ]
        is True
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["witness_assisted_voiceops_action_witness_adjudication"]
        == "corrected_by_audio"
    )
    assert report["proofs"]["async_oracle_jobs"]["witness_assisted_voiceops_action_promoted_authorities"] == [
        "interpreter_promoted"
    ]
    assert report["proofs"]["async_oracle_jobs"]["witness_assisted_voiceops_action_sinks_clean"] is True
    assert (
        report["proofs"]["async_oracle_jobs"]["witness_assisted_voiceops_action_raw_witness_absent"]
        is True
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["witness_assisted_voiceops_action_promoted_text_present"]
        is True
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["witness_assisted_voiceops_action_witness_text"]
        not in str(report["proofs"]["async_oracle_jobs"]["witness_assisted_voiceops_action_sink_values"])
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["witness_assisted_voiceops_action_promoted_text"]
        in str(report["proofs"]["async_oracle_jobs"]["witness_assisted_voiceops_action_sink_values"])
    )
    assert report["requirements"]["async_oracle_transcript_hypotheses_unpromoted"] is True
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_timing_smoke_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_arrival_phases"] == [
        "before_raw_audio",
        "with_raw_audio",
        "after_interpreter_start",
    ]
    assert report["proofs"]["async_oracle_jobs"]["witness_arrival_phase"] == [
        "before_raw_audio",
        "with_raw_audio",
        "after_interpreter_start",
    ]
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_turn_ids"] == {
        "early": "witness-fusion:early",
        "with": "witness-fusion:with",
        "late": "witness-fusion:late",
    }
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_audio_segment_refs"] == {
        "early": "artifact://voice/witness-early.wav",
        "with": "artifact://voice/witness-with.wav",
        "late": "artifact://voice/witness-late.wav",
    }
    assert all(
        value.startswith("kame-merge-")
        for value in report["proofs"]["async_oracle_jobs"]["witness_fusion_evidence_merge_keys"].values()
    )
    assert len(set(report["proofs"]["async_oracle_jobs"]["witness_fusion_evidence_merge_keys"].values())) == 3
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_merge_key_observed"] is True
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_same_turn_convergence_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_same_turn_arrival_phases"] == [
        "before_raw_audio",
        "with_raw_audio",
        "after_interpreter_start",
    ]
    same_turn_lineage = report["proofs"]["async_oracle_jobs"]["witness_fusion_same_turn_lineage"]
    assert same_turn_lineage["turn_id"] == "witness-fusion:same-turn"
    assert same_turn_lineage["audio_segment_ref"] == "artifact://voice/witness-same-turn.wav"
    assert same_turn_lineage["evidence_merge_key"] == "kame-merge-witness-same-turn"
    phase_lineage = report["proofs"]["async_oracle_jobs"]["witness_fusion_same_turn_phase_lineage"]
    assert phase_lineage["before_raw_audio"]["audio_segment_ref"] == ""
    assert phase_lineage["with_raw_audio"]["audio_segment_ref"] == "artifact://voice/witness-same-turn.wav"
    assert (
        phase_lineage["after_interpreter_start"]["audio_segment_ref"]
        == "artifact://voice/witness-same-turn.wav"
    )
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_same_turn_bundle_ids_by_phase"] == {
        phase: same_turn_lineage["evidence_bundle_id"]
        for phase in ("before_raw_audio", "with_raw_audio", "after_interpreter_start")
    }
    assert len(
        set(
            report["proofs"]["async_oracle_jobs"][
                "witness_fusion_same_turn_bundle_ids_by_phase"
            ].values()
        )
    ) == 1
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_same_turn_job_ids_by_phase"] == {
        phase: same_turn_lineage["job_id"]
        for phase in ("before_raw_audio", "with_raw_audio", "after_interpreter_start")
    }
    assert len(
        set(
            report["proofs"]["async_oracle_jobs"][
                "witness_fusion_same_turn_job_ids_by_phase"
            ].values()
        )
    ) == 1
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_same_turn_single_bundle"] is True
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_same_turn_one_oracle_job"] is True
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_same_turn_oracle_job_counts"] == {
        "accepted": 1,
        "started": 1,
        "completed": 1,
    }
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_same_turn_no_duplicate_oracle_job"] is True
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_accepted_audio_gate_observed"] is True
    assert report["proofs"]["async_oracle_jobs"]["raw_audio_interpreter_evidence_observed"] is True
    assert (
        report["proofs"]["async_oracle_jobs"]["witness_fusion_bundle_audio_metadata"]
        == report["proofs"]["async_oracle_jobs"]["witness_fusion_audio_metadata"]
    )
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_audio_metadata"]["early"]["vad"] == {
        "speech_start_ms": 100,
        "speech_end_ms": 1400,
        "vad_speech": True,
    }
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_audio_metadata"]["early"][
        "energy_gate"
    ] == {
        "accepted": True,
        "rms": 620,
        "duration_ms": 1300,
        "min_rms": 350,
        "min_speech_ms": 120,
    }
    assert report["requirements"]["async_oracle_witness_fusion_accepted_audio_gate_visible"] is True
    assert report["async_oracle_coverage"]["witness_fusion_accepted_audio_gate_visible"] is True
    accepted_gate = report["async_oracle_acceptance"]["witness_fusion_exposes_accepted_audio_gate"]
    assert accepted_gate["ok"] is True
    assert accepted_gate["evidence"] == "async_oracle_smoke_plus_accepted_audio_gate_tests"
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_early_initial_bundle_id"] == (
        report["proofs"]["async_oracle_jobs"]["witness_fusion_early_final_bundle_id"]
    )
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_early_single_bundle"] is True
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_interpreter_prompt_input_order"] == [
        "raw_audio",
        "metadata",
        "reflex",
        "transcript_hypotheses",
    ]
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_interpreter_prompt_input_order_expected"] == [
        "raw_audio",
        "metadata",
        "reflex",
        "transcript_hypotheses",
    ]
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_interpreter_prompt_input_order_visible"] is True
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_interpreter_prompt_policy"] == {
        "version": "raw_audio_compare_v1",
        "primary_evidence": "raw_audio",
        "transcript_hypotheses_authority": "non_authoritative_context",
        "promotion_requirement": "compare_transcript_hypotheses_against_raw_audio_before_promotion",
        "forbidden_direct_uses": (
            "oracle_text",
            "durable_transcript",
            "spend_reason",
            "phone_call_payload",
            "tool_arguments",
        ),
    }
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_interpreter_prompt_policy_expected"] == (
        report["proofs"]["async_oracle_jobs"]["witness_fusion_interpreter_prompt_policy"]
    )
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_interpreter_prompt_policy_version"] == (
        "raw_audio_compare_v1"
    )
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_interpreter_prompt_policy_visible"] is True
    assert report["interpreter_request_packet"]["interpreter_input_order"] == [
        "raw_audio",
        "metadata",
        "reflex",
        "transcript_hypotheses",
    ]
    assert report["interpreter_request_packet"]["prompt_input_order"] == (
        report["interpreter_request_packet"]["interpreter_input_order"]
    )
    assert report["interpreter_request_packet"]["interpreter_prompt_policy"] == {
        "version": "raw_audio_compare_v1",
        "primary_evidence": "raw_audio",
        "transcript_hypotheses_authority": "non_authoritative_context",
        "promotion_requirement": "compare_transcript_hypotheses_against_raw_audio_before_promotion",
        "forbidden_direct_uses": (
            "oracle_text",
            "durable_transcript",
            "spend_reason",
            "phone_call_payload",
            "tool_arguments",
        ),
    }
    assert report["interpreter_request_packet"]["prompt_policy"] == (
        report["interpreter_request_packet"]["interpreter_prompt_policy"]
    )
    assert report["interpreter_request_packet"]["reflex"]["kind"] == "reflex_hypothesis"
    assert report["interpreter_request_packet"]["reflex"]["authority"] == "hypothesis"
    assert report["interpreter_request_packet"]["reflex"]["tool_authority"] is False
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_with_single_bundle"] is True
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_late_initial_bundle_id"] == (
        report["proofs"]["async_oracle_jobs"]["witness_fusion_late_final_bundle_id"]
    )
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_late_single_bundle"] is True
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_no_duplicate_oracle_jobs"] is True
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_partial_superseded_by_final"] is True
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_partial_active_hypothesis"] == {
        "source": "moshi",
        "kind": "frontend_witness_hypothesis",
        "text": "what is three to the power of seventeen",
        "text_digest": _text_digest("what is three to the power of seventeen"),
        "role": "witness_context",
        "authority": "hypothesis",
        "promotion_required": "interpreter_promoted_or_oracle_promoted",
        "tool_authority": False,
        "confidence": 0.88,
        "arrival_phase": "with_raw_audio",
        "partial": False,
        "superseded_partial_texts": ("what is three to the",),
        "superseded_partial_count": 1,
    }
    assert report["requirements"]["async_oracle_witness_fusion_partial_superseded_by_final"] is True
    assert report["async_oracle_coverage"]["witness_fusion_partial_superseded_by_final"] is True
    assert report["async_oracle_acceptance"]["witness_fusion_supersedes_partial_witness"]["ok"] is True
    assert (
        "tests/agent/test_realtime_voice.py::test_kame_engine_changing_partials_do_not_create_duplicate_oracle_turns"
        in report["async_oracle_acceptance"]["witness_fusion_supersedes_partial_witness"]["test_refs"]
    )
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_adjudications"] == {
        "early": ["corrected_by_audio"],
        "with": ["accepted_as_supporting_evidence"],
        "late": ["rejected_or_diagnostic_only"],
    }
    assert report["proofs"]["async_oracle_jobs"]["interpreter_adjudication_outcomes"] == {
        "early": ["corrected_by_audio"],
        "with": ["accepted_as_supporting_evidence"],
        "late": ["rejected_or_diagnostic_only"],
    }
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_rejection_reasons"] == {
        "early": [],
        "with": [],
        "late": ["ambiguous_speaker", "wrong_speaker", "wrong_channel", "stale_witness"],
    }
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_adjudication_outcomes_observed"] is True
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_accepted_counts"] == {
        "early": 1,
        "with": 1,
        "late": 1,
    }
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_started_counts"] == {
        "early": 1,
        "with": 1,
        "late": 1,
    }
    assert report["proofs"]["async_oracle_jobs"]["witness_fusion_completed_counts"] == {
        "early": 1,
        "with": 1,
        "late": 1,
    }
    assert (
        report["proofs"]["async_oracle_jobs"]["witness_fusion_multi_speaker_promoted_text"]
        == "prepare late witness handoff"
    )
    witness_fusion = report["async_oracle_acceptance"]["witness_fusion_timing_preserves_single_bundle"]
    assert witness_fusion["ok"] is True
    assert witness_fusion["evidence"] == "async_oracle_smoke_plus_witness_fusion_tests"
    assert report["requirements"]["async_oracle_witness_fusion_single_bundle"] is True
    witness_adjudication = report["async_oracle_acceptance"]["witness_fusion_adjudicates_frontend_text"]
    assert witness_adjudication["ok"] is True
    assert witness_adjudication["evidence"] == "async_oracle_smoke_plus_witness_adjudication_tests"
    assert report["requirements"]["async_oracle_witness_fusion_adjudicates_frontend_text"] is True
    prompt_order = report["async_oracle_acceptance"]["interpreter_prompt_input_order_visible"]
    assert prompt_order["ok"] is True
    assert prompt_order["evidence"] == "async_oracle_smoke_plus_interpreter_prompt_packet_tests"
    assert report["requirements"]["async_oracle_interpreter_prompt_input_order_visible"] is True
    prompt_policy = report["async_oracle_acceptance"]["interpreter_prompt_policy_visible"]
    assert prompt_policy["ok"] is True
    assert prompt_policy["evidence"] == "async_oracle_smoke_plus_interpreter_prompt_policy_tests"
    assert report["requirements"]["async_oracle_interpreter_prompt_policy_visible"] is True
    assert report["requirements"]["async_oracle_energy_gate_ignores_non_speech"] is True
    assert report["async_oracle_coverage"]["energy_gate_ignores_non_speech_without_work"] is True
    energy_gate = report["async_oracle_acceptance"]["energy_gate_ignores_non_speech_without_work"]
    assert energy_gate["ok"] is True
    assert energy_gate["evidence"] == "async_oracle_smoke_plus_energy_gate_tests"
    assert energy_gate["verification_mode"] == "loopback_smoke_plus_focused_tests"
    assert energy_gate["runtime_verified_by_this_report"] is True
    assert (
        "tests/agent/test_realtime_voice.py::test_text_engine_raw_audio_without_confirmed_speech_does_not_barge_in"
        in energy_gate["test_refs"]
    )
    assert (
        "tests/agent/test_realtime_voice.py::test_kame_engine_low_energy_witness_text_does_not_start_turn"
        in energy_gate["test_refs"]
    )
    assert report["proofs"]["async_oracle_jobs"]["energy_gate_smoke_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["energy_gate_policy"] == {
        "min_rms": 350,
        "min_speech_ms": 120,
    }
    assert report["proofs"]["async_oracle_jobs"]["energy_gate_ignored_packet_rms"] == 80
    assert report["proofs"]["async_oracle_jobs"]["energy_gate_ignored_packet_duration_ms"] == 200
    assert report["proofs"]["async_oracle_jobs"]["energy_gate_ignored_packet_speech_confirmed"] is False
    assert report["proofs"]["async_oracle_jobs"]["energy_gate_ignored_packet_vad_speech"] is False
    assert report["proofs"]["async_oracle_jobs"]["energy_gate_ignored_non_speech_packets"] >= 2
    assert (
        report["proofs"]["async_oracle_jobs"]["energy_gate_low_energy_witness_text"]
        == "spend money from room tone"
    )
    assert report["proofs"]["async_oracle_jobs"]["energy_gate_low_energy_witness_source"] == "moshi"
    assert (
        report["proofs"]["async_oracle_jobs"]["energy_gate_low_energy_witness_adjudication"]
        == "rejected_or_diagnostic_only"
    )
    assert report["proofs"]["async_oracle_jobs"][
        "energy_gate_low_energy_witness_rejection_reasons"
    ] == ["low_energy_non_speech"]
    assert (
        report["proofs"]["async_oracle_jobs"]["energy_gate_low_energy_witness_authority"]
        == "hypothesis"
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["energy_gate_low_energy_witness_tool_authority"]
        is False
    )
    assert report["proofs"]["async_oracle_jobs"]["energy_gate_low_energy_witness_promoted"] is False
    assert report["proofs"]["async_oracle_jobs"]["energy_gate_low_energy_witness_suppressed"] is True
    assert report["proofs"]["async_oracle_jobs"]["energy_gate_barge_in_events"] == 0
    assert report["proofs"]["async_oracle_jobs"]["energy_gate_interpreter_requests"] == 0
    assert report["proofs"]["async_oracle_jobs"]["energy_gate_oracle_work_events"] == 0
    assert report["proofs"]["async_oracle_jobs"]["energy_gate_oracle_requests"] == 0
    assert report["proofs"]["async_oracle_jobs"]["energy_gate_raw_packet_buffered_without_turn"] is True
    assert "barge_in.detected" not in report["proofs"]["async_oracle_jobs"]["energy_gate_event_types"]
    assert report["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_smoke_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_waiting_events"] == 6
    assert report["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_hypothesis_only_ok"] is False
    assert "missing_promoted_evidence" in report["proofs"]["async_oracle_jobs"][
        "runtime_kame_action_gate_hypothesis_only_issues"
    ]
    assert "interpreter_evidence_not_consumed_before_irreversible_action" in report["proofs"][
        "async_oracle_jobs"
    ]["runtime_kame_action_gate_hypothesis_only_issues"]
    assert set(
        report["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_hypothesis_only_rejected_authorities"]
    ) >= {"reflex_hypothesis", "hypothesis"}
    assert report["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_degraded_text_only_ok"] is False
    assert (
        report["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_degraded_text_only_status"]
        == "degraded_text_only"
    )
    assert (
        report["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_degraded_text_only_reason"]
        == "degraded_text_only"
    )
    assert (
        report["proofs"]["async_oracle_jobs"][
            "runtime_kame_action_gate_degraded_text_only_raw_audio_available"
        ]
        is False
    )
    assert report["proofs"]["async_oracle_jobs"]["transcript_only_witness_rejected_for_full_kame"] is True
    assert (
        report["proofs"]["async_oracle_jobs"][
            "runtime_kame_action_gate_degraded_text_only_preserves_hypothesis"
        ]
        is True
    )
    assert "missing_promoted_evidence" in report["proofs"]["async_oracle_jobs"][
        "runtime_kame_action_gate_degraded_text_only_issues"
    ]
    assert "interpreter_evidence_not_consumed_before_irreversible_action" in report["proofs"][
        "async_oracle_jobs"
    ]["runtime_kame_action_gate_degraded_text_only_issues"]
    assert set(
        report["proofs"]["async_oracle_jobs"][
            "runtime_kame_action_gate_degraded_text_only_rejected_authorities"
        ]
    ) >= {"reflex_hypothesis", "hypothesis"}
    assert (
        report["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_degraded_oracle_promoted_ok"]
        is False
    )
    assert (
        report["proofs"]["async_oracle_jobs"][
            "runtime_kame_action_gate_degraded_oracle_promoted_status"
        ]
        == "degraded_text_only"
    )
    assert (
        report["proofs"]["async_oracle_jobs"][
            "runtime_kame_action_gate_degraded_oracle_promoted_raw_audio_available"
        ]
        is False
    )
    assert report["proofs"]["async_oracle_jobs"][
        "runtime_kame_action_gate_degraded_oracle_promoted_authorities"
    ] == ["oracle_promoted"]
    assert (
        report["proofs"]["async_oracle_jobs"][
            "runtime_kame_action_gate_degraded_oracle_promoted_consumed_before_action"
        ]
        is True
    )
    assert "missing_promoted_evidence" not in report["proofs"]["async_oracle_jobs"][
        "runtime_kame_action_gate_degraded_oracle_promoted_issues"
    ]
    assert "degraded_text_only_cannot_authorize_high_risk_action" in report["proofs"][
        "async_oracle_jobs"
    ]["runtime_kame_action_gate_degraded_oracle_promoted_issues"]
    assert report["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_promoted_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_promoted_issues"] == []
    assert report["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_promoted_authorities"] == [
        "interpreter_promoted"
    ]
    assert (
        report["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_promoted_consumed_before_action"]
        is True
    )
    assert report["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_self_attested_ok"] is False
    assert report["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_self_attested_issues"] == [
        "missing_promoted_evidence"
    ]
    assert report["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_self_attested_authorities"] == []
    assert (
        report["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_self_attested_consumed_before_action"]
        is True
    )
    assert report["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_missing_tool_disclosure_ok"] is False
    assert report["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_missing_tool_disclosure_issues"] == [
        "missing_tool_disclosure_ref"
    ]
    assert report["proofs"]["async_oracle_jobs"][
        "runtime_kame_action_gate_missing_tool_disclosure_authorities"
    ] == ["interpreter_promoted"]
    assert report["proofs"]["async_oracle_jobs"]["runtime_kame_action_gate_tool_disclosure_ref_observed"] is True
    runtime_action_gate = report["async_oracle_acceptance"][
        "runtime_kame_action_gate_enforces_promoted_evidence"
    ]
    assert runtime_action_gate["ok"] is True
    assert runtime_action_gate["evidence"] == "async_oracle_smoke_plus_runtime_action_gate_tests"
    assert (
        "tests/agent/test_realtime_voice.py::test_async_oracle_unflagged_high_risk_tool_call_fails_closed"
        in runtime_action_gate["test_refs"]
    )
    assert (
        "tests/agent/test_realtime_voice.py::test_async_oracle_unflagged_high_risk_tool_result_fails_closed"
        in runtime_action_gate["test_refs"]
    )
    assert report["proofs"]["async_oracle_jobs"]["durable_resume_contract_smoke_ok"] is True
    assert (
        report["proofs"]["async_oracle_jobs"]["durable_resume_contract_schema_version"]
        == "voiceops.kame_durable_resume_context.v1"
    )
    assert report["proofs"]["async_oracle_jobs"]["durable_resume_promoted_turn_count"] == 4
    assert report["proofs"]["async_oracle_jobs"]["durable_resume_recent_promoted_turns_verbatim"] is True
    assert report["proofs"]["async_oracle_jobs"]["durable_resume_recent_promoted_turns"] == [
        {
            "turn_id": "voice-smoke-durable-resume:3",
            "text": "promoted durable resume request 3",
            "source": "gemma_interpreter",
            "authority": "promoted",
        },
        {
            "turn_id": "voice-smoke-durable-resume:4",
            "text": "promoted durable resume request 4",
            "source": "gemma_interpreter",
            "authority": "promoted",
        },
    ]
    assert report["proofs"]["async_oracle_jobs"]["durable_resume_older_turns_summarized"] is True
    assert report["proofs"]["async_oracle_jobs"]["durable_resume_older_promoted_turn_count"] == 2
    assert "voice-smoke-durable-resume:1" in report["proofs"]["async_oracle_jobs"][
        "durable_resume_older_promoted_turn_summary"
    ]
    assert "voice-smoke-durable-resume:2" in report["proofs"]["async_oracle_jobs"][
        "durable_resume_older_promoted_turn_summary"
    ]
    assert report["proofs"]["async_oracle_jobs"]["durable_resume_hypothesis_replay_absent"] is True
    assert report["proofs"]["async_oracle_jobs"]["durable_resume_ledger_authoritative"] is True
    assert report["proofs"]["async_oracle_jobs"]["hypothesis_final_durable_message_smoke_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["hypothesis_final_durable_messages_empty"] is True
    assert report["proofs"]["async_oracle_jobs"]["hypothesis_final_durable_message_count"] == 0
    assert (
        report["proofs"]["async_oracle_jobs"]["hypothesis_final_without_adapter_flag_non_durable"]
        is True
    )
    assert report["proofs"]["async_oracle_jobs"]["hypothesis_final_witness_intent_non_durable"] is True
    assert report["proofs"]["async_oracle_jobs"]["explicit_asr_fallback_final_remains_durable"] is True
    assert report["proofs"]["async_oracle_jobs"]["explicit_asr_fallback_durable_messages"] == [
        {"role": "user", "content": "check deployment status"}
    ]
    assert report["requirements"]["async_oracle_hypothesis_final_events_non_durable"] is True
    hypothesis_final = report["async_oracle_acceptance"]["hypothesis_final_events_stay_non_durable"]
    assert hypothesis_final["ok"] is True
    assert hypothesis_final["evidence"] == "async_oracle_smoke_plus_session_persistence_tests"
    durable_resume = report["async_oracle_acceptance"]["durable_promoted_turn_resume_contract"]
    assert durable_resume["ok"] is True
    assert durable_resume["evidence"] == "async_oracle_smoke_plus_durable_resume_tests"
    assert (
        "tests/agent/test_realtime_voice.py::test_kame_resume_context_uses_promoted_turns_and_excludes_hypotheses"
        in durable_resume["test_refs"]
    )
    assert report["requirements"]["async_oracle_durable_promoted_turn_resume_contract"] is True
    assert report["requirements"]["async_oracle_runtime_kame_action_gate"] is True
    assert report["requirements"]["async_oracle_unflagged_high_risk_tool_fails_closed"] is True
    assert report["proofs"]["async_oracle_jobs"]["audit_scalar_smoke_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["audit_scalar_payload_redacted"] is True
    assert report["proofs"]["async_oracle_jobs"]["audit_scalar_secret_canary_checked"] is True
    assert report["proofs"]["async_oracle_jobs"]["audit_scalar_result_text_omitted"] is True
    assert report["proofs"]["async_oracle_jobs"]["audit_scalar_completed_event_seen"] is True
    assert report["proofs"]["async_oracle_jobs"]["audit_scalar_waiting_event_seen"] is True
    assert report["proofs"]["async_oracle_jobs"]["audit_scalar_row_count"] == 5
    assert report["proofs"]["async_oracle_jobs"]["shutdown_timeout_configured_ms"] == 10
    assert report["proofs"]["async_oracle_jobs"]["shutdown_bounded_close_observed"] is True
    assert report["proofs"]["async_oracle_jobs"]["shutdown_forced_cancel_observed"] is True
    assert report["proofs"]["async_oracle_jobs"]["shutdown_close_cancel_entered"] is True
    assert report["proofs"]["async_oracle_jobs"]["shutdown_cancelled_jobs"] == 1
    assert report["proofs"]["async_oracle_jobs"]["local_turn_during_running_jobs_observed"] is True
    assert report["proofs"]["async_oracle_jobs"]["local_turn_active_job_count"] == 4
    assert report["proofs"]["async_oracle_jobs"]["status_turn_queued_visible"] is True
    assert report["proofs"]["async_oracle_jobs"]["status_ordinal_labels_visible"] is True
    assert report["proofs"]["async_oracle_jobs"]["status_ordinal_labels"] == (
        "job one",
        "job two",
        "job three",
        "job four",
        "job five",
    )
    assert report["proofs"]["async_oracle_jobs"]["status_bounded_overflow_visible"] is True
    assert report["proofs"]["async_oracle_jobs"]["status_bounded_overflow_visible_job_count"] == 8
    assert report["proofs"]["async_oracle_jobs"]["status_bounded_overflow_hidden_job_count"] == 2
    assert report["proofs"]["async_oracle_jobs"]["status_bounded_overflow_more_spoken_status"] == "+2 more"
    assert report["proofs"]["async_oracle_jobs"]["status_bounded_overflow_last_visible_ordinal"] == 8
    assert report["proofs"]["async_oracle_jobs"]["status_bounded_overflow_last_visible_label"] == "job eight"
    assert report["proofs"]["async_oracle_jobs"]["status_bounded_overflow_hidden_ids_absent"] is True
    assert report["proofs"]["async_oracle_jobs"]["status_turn_no_oracle_request"] is True
    assert report["proofs"]["async_oracle_jobs"]["status_turn_oracle_request_count_before"] == 4
    assert report["proofs"]["async_oracle_jobs"]["status_turn_oracle_request_count_after"] == 4
    assert report["proofs"]["tool_disclosure"]["ok"] is True
    assert report["proofs"]["tool_disclosure"]["schema_source"] == "registered_core_tool_schemas"
    assert report["proofs"]["tool_disclosure"]["representative_schema"] is False
    assert report["proofs"]["tool_disclosure"]["missing_registered_core_tools"] == []
    assert report["proofs"]["tool_disclosure"]["visible_tool_names"] == [
        "tool_call",
        "tool_describe",
        "tool_search",
    ]
    assert report["proofs"]["tool_disclosure"]["visible_non_bridge_tool_names"] == []
    assert report["proofs"]["tool_disclosure"]["input_core_tools"] == sorted(_HERMES_CORE_TOOLS)
    assert report["proofs"]["tool_disclosure"]["hidden_core_tool_names"] == sorted(_HERMES_CORE_TOOLS)
    assert report["proofs"]["tool_disclosure"]["input_core_tool_count"] == len(_HERMES_CORE_TOOLS)
    assert report["proofs"]["tool_disclosure"]["hidden_core_tool_count"] == len(_HERMES_CORE_TOOLS)
    assert report["proofs"]["tool_disclosure"]["core_tools_hidden_all"] is True
    assert report["proofs"]["tool_disclosure"]["broad_core_tools_visible"] is False
    assert report["proofs"]["tool_disclosure"]["deferred_count"] == len(_HERMES_CORE_TOOLS)
    assert report["proofs"]["tool_disclosure"]["token_reduction_estimate"] > 0
    assert report["tool_disclosure_smoke"]["ok"] is True
    for test_ref in (
        "tests/agent/test_realtime_voice_oracle.py::test_voice_oracle_ephemeral_router_selects_voiceops_without_persisting_router_turn",
        "tests/agent/test_realtime_voice_oracle.py::test_voice_oracle_ephemeral_router_can_select_no_tools",
    ):
        assert test_ref in report["tool_disclosure_smoke"]["external_test_refs"]
        assert test_ref in report["proofs"]["tool_disclosure"]["external_test_refs"]
    router_proof = report["proofs"]["ephemeral_tool_router"]
    assert router_proof["ok"] is True
    assert report["ephemeral_tool_router_smoke"]["ok"] is True
    assert report["requirements"]["ephemeral_tool_router"] is True
    assert router_proof["router_mode"] == "ephemeral"
    assert router_proof["provider_network"] is False
    assert router_proof["model_call"] is False
    assert router_proof["router_call_count"] == 2
    assert router_proof["selected_voiceops_toolsets"] == ["voiceops"]
    assert router_proof["selected_no_tools_toolsets"] == []
    assert router_proof["router_transcript_persistent"] is False
    assert router_proof["router_tool_calls_allowed"] is False
    assert router_proof["router_enabled_toolsets"] == [[], []]
    assert router_proof["router_persist_user_messages"] == [False, False]
    assert router_proof["router_skip_memory"] == [True, True]
    assert router_proof["router_skip_context_files"] == [True, True]
    assert router_proof["router_prompts_include_no_tool_instruction"] == [True, True]
    overflow_policy = report["async_oracle_acceptance"]["fifth_job_obeys_overflow_policy"]
    assert overflow_policy["ok"] is True
    assert overflow_policy["evidence"] == "async_oracle_smoke_plus_overflow_policy_tests"
    assert (
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_overflow_policy_reject_rejects_at_capacity_with_queue_space"
        in overflow_policy["test_refs"]
    )
    assert (
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_overflow_policy_reprioritize_requires_user_control_at_capacity"
        in overflow_policy["test_refs"]
    )
    assert (
        "tests/agent/test_realtime_voice.py::test_kame_engine_reports_async_oracle_reprioritize_policy_without_sync_fallback"
        in overflow_policy["test_refs"]
    )
    assert (
        "tests/hermes_cli/test_web_server.py::TestBuildSchemaFromConfig::test_realtime_voice_ws_config_passes_oracle_jobs_from_config"
        in overflow_policy["test_refs"]
    )
    assert report["async_oracle_acceptance"]["status_reports_running_and_queued_without_oracle_call"]["ok"] is True
    job_creation = report["async_oracle_acceptance"]["new_oracle_job_can_be_created_while_others_run"]
    assert job_creation["ok"] is True
    assert job_creation["evidence"] == "async_oracle_smoke_plus_job_creation_tests"
    assert (
        "tests/agent/test_realtime_voice.py::test_kame_engine_can_create_oracle_job_while_another_is_running"
        in job_creation["test_refs"]
    )
    assert (
        "tests/gateway/test_discord_realtime_voice.py::test_discord_realtime_spoken_tasks_create_async_oracle_jobs"
        in job_creation["test_refs"]
    )
    cancellation = report["async_oracle_acceptance"]["cancellation_controls_are_isolated"]
    assert cancellation["ok"] is True
    assert (
        "tests/agent/test_realtime_voice.py::test_session_cancelled_oracle_job_removes_prior_completed_record"
        in cancellation["test_refs"]
    )
    assert (
        "tests/agent/test_realtime_voice.py::test_session_ignores_completed_record_after_oracle_job_cancelled"
        in cancellation["test_refs"]
    )
    assert report["async_oracle_acceptance"]["shutdown_timeout_is_bounded"]["ok"] is True
    assert report["async_oracle_acceptance"]["approval_wait_is_visible_and_redacted"]["ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["approval_cancel_capacity_smoke_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["approval_cancel_late_output_attempted"] is True
    assert report["proofs"]["async_oracle_jobs"]["approval_cancel_completed_after_cancel"] is False
    assert report["proofs"]["async_oracle_jobs"]["approval_cancel_late_result_spoken"] is False
    assert report["proofs"]["async_oracle_jobs"][
        "approval_cancel_followup_started_before_cancel_drained"
    ] is False
    assert report["requirements"]["async_oracle_approval_cancel_holds_capacity"] is True
    assert (
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_cancelling_waiting_for_approval_keeps_capacity_until_worker_stops_and_drops_late_result"
        in report["async_oracle_acceptance"]["approval_wait_is_visible_and_redacted"]["test_refs"]
    )
    assert report["proofs"]["async_oracle_jobs"]["approval_secret_leaked"] is False
    assert report["proofs"]["async_oracle_jobs"]["approval_secret_canary_checked"] is True
    assert (
        report["async_oracle_acceptance"]["approval_wait_is_visible_and_redacted"]["verification_mode"]
        == "loopback_smoke_plus_focused_tests"
    )
    assert report["async_oracle_acceptance"]["failed_job_is_reported_without_crashing_session"]["ok"] is True
    assert (
        report["async_oracle_acceptance"]["failed_job_is_reported_without_crashing_session"]["verification_mode"]
        == "loopback_smoke_plus_focused_tests"
    )
    assert report["async_oracle_acceptance"]["job_control_updates_reach_oracle"]["ok"] is True
    assert (
        report["async_oracle_acceptance"]["job_control_updates_reach_oracle"]["verification_mode"]
        == "loopback_smoke_plus_focused_tests"
    )
    assert "tests/gateway/test_discord_realtime_voice.py::test_discord_realtime_event_tracks_oracle_job_status" in (
        report["async_oracle_acceptance"]["job_control_updates_reach_oracle"]["test_refs"]
    )
    assert "tests/agent/test_realtime_voice.py::test_kame_engine_attaches_update_to_running_async_oracle_job" in (
        report["async_oracle_acceptance"]["job_control_updates_reach_oracle"]["test_refs"]
    )
    assert (
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_add_update_redacts_secret_like_text_from_status_and_events"
        in report["async_oracle_acceptance"]["job_control_updates_reach_oracle"]["test_refs"]
    )
    transcript_authority = report["async_oracle_acceptance"]["transcript_hypotheses_stay_non_authoritative"]
    assert transcript_authority["ok"] is True
    assert transcript_authority["evidence"] == "async_oracle_smoke_plus_interpreter_authority_tests"
    assert transcript_authority["verification_mode"] == "loopback_smoke_plus_focused_tests"
    assert (
        "tests/agent/test_realtime_voice.py::test_kame_engine_does_not_promote_moshi_only_queued_evidence"
        in transcript_authority["test_refs"]
    )
    external_frontend = report["async_oracle_acceptance"]["external_frontend_bridge_submits_oracle_job"]
    assert external_frontend["ok"] is True
    assert external_frontend["evidence"] == "async_oracle_smoke_plus_external_frontend_tests"
    assert external_frontend["verification_mode"] == "loopback_smoke_plus_focused_tests"
    assert external_frontend["runtime_verified_by_this_report"] is True
    assert (
        "tests/agent/test_realtime_voice.py::test_session_client_interface_oracle_request_submits_external_kame_job"
        in external_frontend["test_refs"]
    )
    assert (
        "tests/agent/test_realtime_voice.py::test_external_kame_ask_brain_bridge_strips_nested_tool_authority"
        in external_frontend["test_refs"]
    )
    result_handling = report["async_oracle_acceptance"]["result_handling_is_bounded_and_durable"]
    assert result_handling["ok"] is True
    assert result_handling["evidence"] == "async_oracle_smoke_plus_result_tests"
    assert result_handling["verification_mode"] == "loopback_smoke_plus_focused_tests"
    assert result_handling["runtime_verified_by_this_report"] is True
    assert result_handling["live_external_evidence_required"] is False
    assert result_handling["test_ref_count"] == len(result_handling["test_refs"])
    assert result_handling["test_ref_count"] >= 1
    assert (
        "tests/agent/test_realtime_voice_reference_sidecar_openai.py::test_reference_sidecar_forwards_speakable_oracle_result_to_openai_realtime"
        in result_handling["test_refs"]
    )
    assert (
        "tests/agent/test_realtime_voice_reference_sidecar_gemini.py::test_reference_sidecar_forwards_speakable_oracle_result_to_gemini_live"
        in result_handling["test_refs"]
    )
    assert (
        "tests/agent/test_realtime_voice_oracle_jobs.py::test_audit_ledger_path_records_redacted_lifecycle_events"
        in result_handling["test_refs"]
    )
    discord_cleanup = report["async_oracle_acceptance"]["discord_session_cleanup_preserves_oracle_state"]
    assert discord_cleanup["ok"] is True
    assert discord_cleanup["evidence"] == "discord_session_cleanup_smoke_plus_focused_tests"
    assert discord_cleanup["verification_mode"] == "loopback_smoke_plus_focused_tests"
    assert discord_cleanup["runtime_verified_by_this_report"] is True
    assert discord_cleanup["live_external_evidence_required"] is False
    assert discord_cleanup["test_ref_count"] == len(discord_cleanup["test_refs"])
    assert discord_cleanup["test_ref_count"] >= 1
    assert (
        "tests/gateway/test_discord_realtime_voice.py::test_discord_realtime_session_close_waits_for_oracle_cancel_ack_before_session_closed"
        in discord_cleanup["test_refs"]
    )
    assert report["proofs"]["discord_session_cleanup"]["ok"] is True
    assert report["proofs"]["discord_session_cleanup"]["cancel_all_before_session_closed"] is True
    assert report["proofs"]["discord_session_cleanup"]["session_closed_sent"] is True
    assert report["proofs"]["discord_session_cleanup"]["sidecar_closed"] is True
    assert report["proofs"]["discord_session_cleanup"]["degraded_active_job_preserved_failed"] is True
    assert report["proofs"]["discord_session_cleanup"]["degraded_session_removed"] is True
    assert report["proofs"]["discord_session_cleanup"]["degraded_job_state"] == "failed"
    sidecar_fail_closed = report["async_oracle_acceptance"][
        "sidecar_fail_closed_send_failure_cancels_active_job"
    ]
    assert sidecar_fail_closed["ok"] is True
    assert sidecar_fail_closed["evidence"] == "sidecar_fail_closed_smoke_plus_focused_tests"
    assert sidecar_fail_closed["verification_mode"] == "loopback_smoke_plus_focused_tests"
    assert sidecar_fail_closed["runtime_verified_by_this_report"] is True
    assert sidecar_fail_closed["live_external_evidence_required"] is False
    assert sidecar_fail_closed["test_ref_count"] == len(sidecar_fail_closed["test_refs"])
    assert (
        "tests/agent/test_realtime_voice.py::test_kame_engine_fail_closed_sidecar_send_failure_cancels_external_oracle_job"
        in sidecar_fail_closed["test_refs"]
    )
    assert report["requirements"]["async_oracle_sidecar_fail_closed_cancels_active_job"] is True
    assert report["proofs"]["sidecar_fail_closed"]["ok"] is True
    assert report["proofs"]["sidecar_fail_closed"]["request_accepted"] is True
    assert report["proofs"]["sidecar_fail_closed"]["cancel_reason"] == "sidecar_send_failed"
    assert report["proofs"]["sidecar_fail_closed"]["session_error_reason"] == "sidecar_send_failed"
    assert report["proofs"]["sidecar_fail_closed"]["error_redacted"] is True
    assert report["proofs"]["sidecar_fail_closed"]["active_capacity_after_failure"] == 0
    assert report["proofs"]["sidecar_fail_closed"]["job_state_after_failure"] == "cancelled"
    assert report["proofs"]["sidecar_fail_closed"]["sidecar_removed"] is True
    assert report["proofs"]["sidecar_fail_closed"]["sidecar_closed"] is True
    assert "tests/agent/test_realtime_voice.py::test_kame_engine_local_status_question_uses_oracle_job_state" in (
        report["async_oracle_acceptance"]["status_reports_running_and_queued_without_oracle_call"]["test_refs"]
    )
    assert report["proofs"]["async_oracle_jobs"]["max_worker_overlap"] == 4
    assert report["proofs"]["async_oracle_jobs"]["worker_overlap_proved"] is True
    assert report["proofs"]["async_oracle_jobs"]["worker_overlap_within_capacity"] is True
    assert report["proofs"]["async_oracle_jobs"]["noncooperative_cancel_overlap_observed"] is False
    assert report["requirements"]["async_oracle_kame_ack_latency_metrics_visible"] is True
    assert report["async_oracle_coverage"]["kame_ack_latency_metrics_visible"] is True
    assert report["async_oracle_acceptance"]["kame_ack_latency_metrics_visible"]["ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["kame_ack_latency_metrics_smoke_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["kame_defer_ack_first_audio_metrics_visible"] is True
    assert report["proofs"]["async_oracle_jobs"]["kame_local_first_audio_metrics_visible"] is True
    assert (
        "kame_interface_decision_to_defer_first_audio_ms"
        in report["proofs"]["async_oracle_jobs"]["kame_defer_ack_metric_keys"]
    )
    assert (
        "kame_interface_decision_to_local_first_audio_ms"
        in report["proofs"]["async_oracle_jobs"]["kame_local_first_audio_metric_keys"]
    )
    assert report["proofs"]["async_oracle_jobs"]["kame_defer_speech_end_to_first_audio_ms"] >= 41
    assert report["proofs"]["async_oracle_jobs"]["kame_local_speech_end_to_first_audio_ms"] >= 37
    assert report["requirements"]["async_oracle_reflex_ack_transcript_visible"] is True
    assert report["async_oracle_coverage"]["reflex_ack_transcript_visible"] is True
    assert report["async_oracle_acceptance"]["reflex_ack_transcript_visible"]["ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["reflex_ack_transcript_smoke_ok"] is True
    assert report["proofs"]["async_oracle_jobs"]["reflex_ack_transcript_visible"] is True
    assert report["proofs"]["async_oracle_jobs"]["reflex_ack_authority"] == "reflex_hypothesis"
    assert report["proofs"]["async_oracle_jobs"]["reflex_ack_action_authority"] is False
    assert report["proofs"]["async_oracle_jobs"]["reflex_ack_tool_authority"] is False
    assert report["proofs"]["async_oracle_jobs"]["reflex_ack_durability"] == "visible_transcript_and_audit"
    ack_record = report["proofs"]["async_oracle_jobs"]["reflex_ack_transcript_record"]
    assert ack_record["schema_version"] == "voiceops.reflex_ack_transcript.v1"
    assert ack_record["text_source"] == "reflex_acknowledgement"
    assert ack_record["authority"] == "reflex_hypothesis"
    assert ack_record["visible_to_user"] is True
    assert ack_record["spoken"] is True
    assert report["proofs"]["latency_metrics"]["oracle_metric_status"] == "needs_live_oracle_or_sidecar_probe"
    assert report["live_probe_required_for_completion"]["status"] == "needs_live_probe"
    assert report["live_probe_required_for_completion"]["missing_gates"] == [
        "discord_join",
        "discord_playback",
        "live_receiver",
        "production_sidecar",
        "live_turn",
    ]
    assert report["live_evidence"]["overall_status"] == "needs_live_probe"
    assert "I cannot hear voice." in report["voice_capability_prompt_contract"]["must_not_claim"]
    assert report["barge_in_policy"]["silent_packet_policy"].startswith("silent PCM")


def test_voice_operator_validation_rejects_conflicting_interpreter_packet_hypothesis_lineage():
    conflicts = {
        "turn_id": "stale-turn-id",
        "audio_segment_ref": "artifact://redacted/stale-or-wrong-speaker-cut.wav",
        "evidence_bundle_id": "stale-evidence-bundle",
        "evidence_merge_key": "stale-evidence-merge-key",
    }
    for field, value in conflicts.items():
        report = _voice_operator_report()
        report["interpreter_request_packet"]["transcript_hypotheses"][0][field] = value

        issues = validate_voice_operator_report(report)

        assert f"interpreter_request_packet:kame_lineage_conflict_{field}" in issues


def test_voice_operator_validation_rejects_interpreter_packet_prompt_policy_drift():
    report = _voice_operator_report()
    report["interpreter_request_packet"]["interpreter_input_order"] = [
        "transcript_hypotheses",
        "raw_audio",
    ]
    report["interpreter_request_packet"]["interpreter_prompt_policy"] = {
        "version": "legacy_transcript_first"
    }
    report["interpreter_request_packet"]["reflex"].pop("kind", None)
    report["interpreter_request_packet"]["reflex"]["authority"] = "reflex_hypothesis"
    report["interpreter_request_packet"]["reflex"]["tool_authority"] = True

    issues = validate_voice_operator_report(report)

    assert "interpreter_request_packet:interpreter_input_order_mismatch" in issues
    assert "interpreter_request_packet:interpreter_prompt_policy_mismatch" in issues
    assert "interpreter_request_packet:reflex_kind_mismatch" in issues
    assert "interpreter_request_packet:reflex_authority_mismatch" in issues
    assert "interpreter_request_packet:reflex_tool_authority_not_false" in issues


def test_voice_operator_validation_rejects_conflicting_interpreter_packet_witness_binding():
    report = _voice_operator_report()
    report["interpreter_request_packet"]["transcript_hypotheses"][0]["speaker_guess"] = {
        "platform": "discord",
        "channel_user_id": "other-human",
    }
    report["interpreter_request_packet"]["transcript_hypotheses"][0]["channel_guess"] = {
        "transport": "discord_voice",
        "channel_id": "other-channel",
    }

    issues = validate_voice_operator_report(report)

    assert "interpreter_request_packet:transcript_hypothesis_0_speaker_mismatch" in issues
    assert "interpreter_request_packet:transcript_hypothesis_0_channel_mismatch" in issues


def test_voice_operator_validation_accepts_rejected_interpreter_packet_witness_binding():
    report = _voice_operator_report()
    report["interpreter_request_packet"]["transcript_hypotheses"][0].update(
        {
            "speaker_guess": {
                "platform": "discord",
                "channel_user_id": "other-human",
            },
            "channel_guess": {
                "transport": "discord_voice",
                "channel_id": "other-channel",
            },
            "adjudication": "rejected_or_diagnostic_only",
            "rejection_reasons": ["wrong_speaker", "wrong_channel"],
        }
    )

    issues = validate_voice_operator_report(report)

    assert "interpreter_request_packet:transcript_hypothesis_0_speaker_mismatch" not in issues
    assert "interpreter_request_packet:transcript_hypothesis_0_channel_mismatch" not in issues


def test_voice_operator_validation_rejects_active_partial_interpreter_packet_hypothesis():
    report = _voice_operator_report()
    report["interpreter_request_packet"]["transcript_hypotheses"][0]["partial"] = True

    issues = validate_voice_operator_report(report)

    assert "interpreter_request_packet:active_partial_hypothesis_not_superseded" in issues


def test_voice_operator_validation_rejects_missing_core_coverage():
    smoke = _smoke_payload()
    smoke["events"] = ["audio.output.chunk"]
    smoke["sidecar_closed"] = False
    smoke["shutdown_bounded"] = False
    report = _voice_operator_report(smoke=smoke)

    issues = validate_voice_operator_report(report)
    assert "missing_coverage:discord_receiver_callback_wiring" in issues
    assert "missing_coverage:lifecycle_start_and_shutdown" in issues
    assert "missing_coverage:sidecar_session_shutdown" in issues


def test_voice_operator_validation_rejects_missing_async_oracle_smoke():
    report = build_voice_operator_report(_smoke_payload(), async_oracle_smoke={})

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_coverage:async_oracle_smoke_ok" in issues
    assert "missing_async_oracle_coverage:four_jobs_ran_concurrently" in issues
    assert "missing_async_oracle_coverage:local_turn_while_jobs_running" in issues
    assert "missing_async_oracle_coverage:status_turn_while_jobs_running" in issues
    assert "missing_async_oracle_coverage:status_turn_ordinal_labels_visible" in issues
    assert "missing_async_oracle_coverage:fifth_job_queued_and_started_after_capacity_freed" in issues
    assert "missing_async_oracle_coverage:one_job_cancelled_while_others_completed" in issues
    assert "missing_async_oracle_coverage:queued_job_cancelled_before_start" in issues
    assert "missing_async_oracle_coverage:late_cancelled_output_attempted" in issues
    assert "missing_async_oracle_coverage:late_cancelled_output_not_spoken" in issues
    assert "missing_async_oracle_coverage:late_cancelled_output_not_durable" in issues
    assert "missing_async_oracle_coverage:playback_stop_does_not_cancel_jobs" in issues
    assert "missing_async_oracle_coverage:approval_wait_visible_and_redacted" in issues
    assert "missing_async_oracle_coverage:approval_wait_holds_capacity" in issues
    assert "missing_async_oracle_coverage:cancel_drain_holds_capacity" in issues
    assert "missing_async_oracle_coverage:failed_job_reported_without_crash" in issues
    assert "missing_async_oracle_coverage:job_control_updates_reach_oracle" in issues
    assert "missing_async_oracle_coverage:transcript_hypotheses_remain_unpromoted" in issues
    assert "missing_async_oracle_coverage:hypothesis_final_events_non_durable" in issues
    assert "missing_async_oracle_coverage:witness_fusion_timing_preserves_single_bundle" in issues
    assert "missing_async_oracle_coverage:witness_fusion_accepted_audio_gate_visible" in issues
    assert "missing_async_oracle_coverage:witness_fusion_partial_superseded_by_final" in issues
    assert "missing_async_oracle_coverage:energy_gate_ignores_non_speech_without_work" in issues
    assert "missing_async_oracle_coverage:runtime_kame_action_gate_enforced" in issues
    assert "missing_async_oracle_coverage:result_handling_bounded_and_durable" in issues
    assert "missing_async_oracle_coverage:discord_session_cleanup_preserves_oracle_state" in issues
    assert "missing_async_oracle_coverage:sidecar_fail_closed_send_failure_cancels_active_job" in issues
    assert "missing_async_oracle_acceptance:four_oracle_jobs_reflex_responsive" in issues
    assert "missing_async_oracle_acceptance:fifth_job_obeys_overflow_policy" in issues
    assert "missing_async_oracle_acceptance:approval_wait_is_visible_and_redacted" in issues
    assert "missing_async_oracle_acceptance:failed_job_is_reported_without_crashing_session" in issues
    assert "missing_async_oracle_acceptance:job_control_updates_reach_oracle" in issues
    assert "missing_async_oracle_acceptance:transcript_hypotheses_stay_non_authoritative" in issues
    assert "missing_async_oracle_acceptance:hypothesis_final_events_stay_non_durable" in issues
    assert "missing_async_oracle_acceptance:witness_fusion_timing_preserves_single_bundle" in issues
    assert "missing_async_oracle_acceptance:witness_fusion_exposes_accepted_audio_gate" in issues
    assert "missing_async_oracle_acceptance:witness_fusion_supersedes_partial_witness" in issues
    assert "missing_async_oracle_acceptance:energy_gate_ignores_non_speech_without_work" in issues
    assert "missing_async_oracle_acceptance:runtime_kame_action_gate_enforces_promoted_evidence" in issues
    assert "missing_async_oracle_acceptance:discord_session_cleanup_preserves_oracle_state" in issues
    assert "missing_async_oracle_acceptance:sidecar_fail_closed_send_failure_cancels_active_job" in issues
    result_handling = report["async_oracle_acceptance"]["result_handling_is_bounded_and_durable"]
    assert result_handling["ok"] is False
    assert result_handling["verification_mode"] == "loopback_smoke_plus_focused_tests"
    assert result_handling["runtime_verified_by_this_report"] is True


def test_voice_operator_validation_recomputes_async_coverage_from_embedded_smoke():
    report = _voice_operator_report()
    report["async_oracle_smoke"]["worker_overlap_proved"] = False
    report["async_oracle_smoke"]["max_worker_overlap"] = 0

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_coverage:four_jobs_ran_concurrently" in issues
    assert "stale_async_oracle_coverage:four_jobs_ran_concurrently" in issues


def test_voice_operator_validation_rejects_partial_witness_without_arrival_phase():
    report = _voice_operator_report()
    report["async_oracle_smoke"]["witness_fusion_partial_active_hypothesis"].pop(
        "arrival_phase",
        None,
    )

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_coverage:witness_fusion_partial_superseded_by_final" in issues
    assert "stale_async_oracle_coverage:witness_fusion_partial_superseded_by_final" in issues
    assert "missing_async_oracle_acceptance:witness_fusion_supersedes_partial_witness" in issues


def test_voice_operator_validation_rejects_local_turn_without_running_job_overlap():
    report = _voice_operator_report()
    report["async_oracle_smoke"]["local_turn_during_running_jobs_observed"] = False
    report["async_oracle_smoke"]["local_turn_active_job_count"] = 0

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_coverage:local_turn_while_jobs_running" in issues
    assert "stale_async_oracle_coverage:local_turn_while_jobs_running" in issues
    assert "missing_async_oracle_acceptance:four_oracle_jobs_reflex_responsive" in issues


def test_voice_operator_validation_rejects_status_turn_without_queued_or_no_oracle_call_proof():
    report = _voice_operator_report()
    report["async_oracle_smoke"]["status_turn_queued_visible"] = False
    report["async_oracle_smoke"]["status_ordinal_labels_visible"] = False
    report["async_oracle_smoke"]["status_ordinal_labels"] = ("job one",)
    report["async_oracle_smoke"]["status_turn_no_oracle_request"] = False
    report["async_oracle_smoke"]["status_turn_oracle_request_count_after"] = 5
    report["async_oracle_smoke"]["status_text"] = "Oracle jobs: 4 running out of 4."
    report["async_oracle_smoke"]["reflex_status_overflow_hidden_job_count"] = 0
    report["async_oracle_smoke"]["reflex_status_overflow_more_spoken_status"] = ""

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_coverage:status_turn_while_jobs_running" in issues
    assert "stale_async_oracle_coverage:status_turn_while_jobs_running" in issues
    assert "missing_async_oracle_coverage:status_turn_ordinal_labels_visible" in issues
    assert "stale_async_oracle_coverage:status_turn_ordinal_labels_visible" in issues
    assert "missing_async_oracle_coverage:status_turn_bounded_overflow_visible" in issues
    assert "stale_async_oracle_coverage:status_turn_bounded_overflow_visible" in issues
    assert "missing_async_oracle_acceptance:status_reports_running_and_queued_without_oracle_call" in issues


def test_voice_operator_validation_rejects_missing_queued_cancel_proof():
    report = _voice_operator_report()
    report["async_oracle_smoke"]["queued_cancel_not_sent_to_oracle"] = False

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_coverage:queued_job_cancelled_before_start" in issues
    assert "stale_async_oracle_coverage:queued_job_cancelled_before_start" in issues
    assert "missing_async_oracle_acceptance:cancellation_controls_are_isolated" in issues


def test_voice_operator_validation_rejects_promoted_transcript_hypothesis():
    report = _voice_operator_report()
    report["async_oracle_smoke"]["unpromoted_hypothesis_promoted"] = True
    report["async_oracle_smoke"]["unpromoted_hypothesis_oracle_text_preserved"] = False

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_coverage:transcript_hypotheses_remain_unpromoted" in issues
    assert "stale_async_oracle_coverage:transcript_hypotheses_remain_unpromoted" in issues
    assert "missing_async_oracle_acceptance:transcript_hypotheses_stay_non_authoritative" in issues


def test_voice_operator_validation_rejects_energy_gate_work_from_noise():
    report = _voice_operator_report()
    report["async_oracle_smoke"]["energy_gate_barge_in_events"] = 1
    report["async_oracle_smoke"]["energy_gate_raw_packet_buffered_without_turn"] = False
    report["async_oracle_smoke"]["energy_gate_low_energy_witness_adjudication"] = (
        "accepted_as_supporting_evidence"
    )
    report["async_oracle_smoke"]["energy_gate_low_energy_witness_rejection_reasons"] = []
    report["async_oracle_smoke"]["energy_gate_low_energy_witness_authority"] = "interpreter_promoted"
    report["async_oracle_smoke"]["energy_gate_low_energy_witness_tool_authority"] = True

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_coverage:energy_gate_ignores_non_speech_without_work" in issues
    assert "stale_async_oracle_coverage:energy_gate_ignores_non_speech_without_work" in issues
    assert "missing_async_oracle_acceptance:energy_gate_ignores_non_speech_without_work" in issues


def test_voice_operator_validation_rejects_missing_accepted_audio_gate_metadata():
    report = _voice_operator_report()
    report["async_oracle_smoke"]["witness_fusion_accepted_audio_gate_observed"] = False
    report["async_oracle_smoke"]["witness_fusion_bundle_audio_metadata"] = {}

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_coverage:witness_fusion_accepted_audio_gate_visible" in issues
    assert "stale_async_oracle_coverage:witness_fusion_accepted_audio_gate_visible" in issues
    assert "missing_async_oracle_acceptance:witness_fusion_exposes_accepted_audio_gate" in issues


def test_voice_operator_validation_rejects_stale_barge_in_energy_gate_proof():
    report = _voice_operator_report()
    report["proofs"]["barge_in_energy"]["energy_gate_proven_by_smoke"] = False

    issues = validate_voice_operator_report(report)

    assert "stale_proof:barge_in_energy.energy_gate_proven_by_smoke" in issues


def test_voice_operator_validation_rejects_async_approval_secret_leak():
    report = _voice_operator_report()
    report["async_oracle_smoke"]["approval_secret_leaked"] = True

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_coverage:approval_wait_visible_and_redacted" in issues
    assert "stale_async_oracle_coverage:approval_wait_visible_and_redacted" in issues
    assert "missing_async_oracle_acceptance:approval_wait_is_visible_and_redacted" in issues


def test_voice_operator_validation_rejects_missing_async_approval_secret_canary_check():
    report = _voice_operator_report()
    report["async_oracle_smoke"]["approval_secret_canary_checked"] = False

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_coverage:approval_wait_visible_and_redacted" in issues
    assert "stale_async_oracle_coverage:approval_wait_visible_and_redacted" in issues
    assert "missing_async_oracle_acceptance:approval_wait_is_visible_and_redacted" in issues


def test_voice_operator_validation_rejects_missing_approval_cancel_capacity_proof():
    report = _voice_operator_report()
    report["async_oracle_smoke"]["approval_cancel_late_output_attempted"] = False
    report["async_oracle_smoke"]["approval_cancel_followup_started_before_cancel_drained"] = True

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_coverage:approval_cancel_holds_capacity" in issues
    assert "stale_async_oracle_coverage:approval_cancel_holds_capacity" in issues
    assert "missing_async_oracle_acceptance:approval_wait_is_visible_and_redacted" in issues


def test_voice_operator_validation_rejects_missing_visible_queued_update_status():
    report = _voice_operator_report()
    report["async_oracle_smoke"]["queued_update_latest_update_visible"] = False

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_coverage:job_control_updates_reach_oracle" in issues
    assert "stale_async_oracle_coverage:job_control_updates_reach_oracle" in issues
    assert "missing_async_oracle_acceptance:job_control_updates_reach_oracle" in issues


def test_voice_operator_validation_rejects_missing_tool_disclosure_test_refs():
    report = _voice_operator_report()
    report["tool_disclosure_smoke"]["external_test_refs"] = []
    report["proofs"]["tool_disclosure"]["external_test_refs"] = []

    issues = validate_voice_operator_report(report)

    assert "progressive_tool_disclosure:missing_external_test_refs" in issues
    assert "progressive_tool_disclosure:missing_proof_external_test_refs" in issues


def test_voice_operator_validation_rejects_stale_tool_disclosure_test_ref():
    report = _voice_operator_report()
    report["tool_disclosure_smoke"]["external_test_refs"] = [
        "tests/tools/test_tool_search.py::TestAssembly::test_missing_tool_search_contract"
    ]
    report["proofs"]["tool_disclosure"]["external_test_refs"] = report["tool_disclosure_smoke"][
        "external_test_refs"
    ]

    issues = validate_voice_operator_report(report)

    assert (
        "progressive_tool_disclosure:invalid_external_test_ref:"
        "tests/tools/test_tool_search.py::TestAssembly::test_missing_tool_search_contract"
    ) in issues
    assert (
        "progressive_tool_disclosure:invalid_proof_external_test_ref:"
        "tests/tools/test_tool_search.py::TestAssembly::test_missing_tool_search_contract"
    ) in issues


def test_voice_operator_validation_rejects_tool_disclosure_proof_ref_mismatch():
    report = _voice_operator_report()
    report["proofs"]["tool_disclosure"]["external_test_refs"] = [
        "tests/tools/test_tool_search.py::TestAssembly::test_defer_core_all_hides_core_behind_bridge"
    ]

    issues = validate_voice_operator_report(report)

    assert "progressive_tool_disclosure:stale_proof_external_test_refs" in issues


def test_voice_operator_validation_rejects_stale_tool_disclosure_core_list():
    report = _voice_operator_report()
    report["tool_disclosure_smoke"]["input_core_tools"] = ["read_file"]
    report["tool_disclosure_smoke"]["hidden_core_tool_names"] = ["read_file"]
    report["tool_disclosure_smoke"]["input_core_tool_count"] = 1
    report["tool_disclosure_smoke"]["hidden_core_tool_count"] = 1
    report["tool_disclosure_smoke"]["deferred_count"] = 1
    report["proofs"]["tool_disclosure"]["input_core_tools"] = ["read_file"]
    report["proofs"]["tool_disclosure"]["hidden_core_tool_names"] = ["read_file"]
    report["proofs"]["tool_disclosure"]["input_core_tool_count"] = 1
    report["proofs"]["tool_disclosure"]["hidden_core_tool_count"] = 1
    report["proofs"]["tool_disclosure"]["deferred_count"] = 1

    issues = validate_voice_operator_report(report)

    assert "progressive_tool_disclosure:stale_input_core_tools" in issues
    assert "progressive_tool_disclosure:stale_hidden_core_tools" in issues
    assert "progressive_tool_disclosure:proof_stale_input_core_tools" in issues
    assert "progressive_tool_disclosure:proof_stale_hidden_core_tools" in issues


def test_voice_operator_validation_rejects_representative_tool_disclosure_schema():
    report = _voice_operator_report()
    report["tool_disclosure_smoke"]["schema_source"] = "representative_core_tool_schemas"
    report["tool_disclosure_smoke"]["representative_schema"] = True
    report["tool_disclosure_smoke"]["missing_registered_core_tools"] = ["terminal"]
    report["proofs"]["tool_disclosure"]["schema_source"] = "representative_core_tool_schemas"
    report["proofs"]["tool_disclosure"]["representative_schema"] = True
    report["proofs"]["tool_disclosure"]["missing_registered_core_tools"] = ["terminal"]

    issues = validate_voice_operator_report(report)

    assert "progressive_tool_disclosure:smoke_schema_source_not_registered" in issues
    assert "progressive_tool_disclosure:smoke_representative_schema" in issues
    assert "progressive_tool_disclosure:smoke_missing_registered_core_tools" in issues
    assert "progressive_tool_disclosure:proof_schema_source_not_registered" in issues
    assert "progressive_tool_disclosure:proof_representative_schema" in issues
    assert "progressive_tool_disclosure:proof_missing_registered_core_tools" in issues


def test_voice_operator_validation_rejects_completed_result_missing_from_status_view():
    report = _voice_operator_report()
    report["async_oracle_smoke"]["completed_result_status_visible"] = False

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_coverage:result_handling_bounded_and_durable" in issues
    assert "stale_async_oracle_coverage:result_handling_bounded_and_durable" in issues
    assert "missing_async_oracle_acceptance:result_handling_is_bounded_and_durable" in issues


def test_voice_operator_validation_rejects_missing_terminal_result_suppression_policy():
    report = _voice_operator_report()
    report["async_oracle_smoke"]["terminal_result_suppressed"] = False

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_coverage:result_handling_bounded_and_durable" in issues
    assert "stale_async_oracle_coverage:result_handling_bounded_and_durable" in issues
    assert "missing_async_oracle_acceptance:result_handling_is_bounded_and_durable" in issues


def test_voice_operator_validation_rejects_missing_external_frontend_bridge_proof():
    report = _voice_operator_report()
    report["async_oracle_smoke"]["external_frontend_request_accepted"] = False
    report["async_oracle_smoke"]["external_frontend_evidence_bundle_propagated"] = False
    report["async_oracle_smoke"]["external_frontend_terminal_correlation_observed"] = False
    report["async_oracle_smoke"]["external_frontend_direct_tool_authority_exposed"] = True

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_coverage:external_frontend_bridge_submits_oracle_job" in issues
    assert "stale_async_oracle_coverage:external_frontend_bridge_submits_oracle_job" in issues
    assert "missing_async_oracle_acceptance:external_frontend_bridge_submits_oracle_job" in issues


def test_voice_operator_validation_rejects_missing_shutdown_timeout_coverage():
    report = _voice_operator_report()
    report["async_oracle_smoke"]["shutdown_bounded_close_observed"] = False

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_coverage:shutdown_timeout_bounded" in issues
    assert "stale_async_oracle_coverage:shutdown_timeout_bounded" in issues
    assert "missing_async_oracle_acceptance:shutdown_timeout_is_bounded" in issues


def test_voice_operator_validation_rejects_missing_shutdown_timeout_acceptance_row():
    report = _voice_operator_report()
    del report["async_oracle_acceptance"]["shutdown_timeout_is_bounded"]

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_acceptance:shutdown_timeout_is_bounded" in issues


def test_voice_operator_validation_rejects_unexpected_async_acceptance_row():
    report = _voice_operator_report()
    report["async_oracle_acceptance"]["unverified_magic_voice_capability"] = {
        "ok": True,
        "evidence": "invented",
        "verification_mode": "loopback_smoke_plus_focused_tests",
        "runtime_verified_by_this_report": True,
        "test_refs": [
            "tests/scripts/test_voiceops_voice_operator.py::test_voice_operator_validation_accepts_current_async_acceptance_test_refs"
        ],
        "test_ref_count": 1,
    }

    issues = validate_voice_operator_report(report)

    assert "unexpected_async_oracle_acceptance:unverified_magic_voice_capability" in issues


def test_voice_operator_validation_rejects_static_acceptance_without_test_refs():
    report = _voice_operator_report()
    row = report["async_oracle_acceptance"]["result_handling_is_bounded_and_durable"]
    row["test_refs"] = []
    row["test_ref_count"] = 0

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_acceptance_test_refs:result_handling_is_bounded_and_durable" in issues


def test_voice_operator_validation_accepts_current_async_acceptance_test_refs():
    report = _voice_operator_report()

    assert validate_voice_operator_report(report) == []
    assert (
        "tests/gateway/test_voice_command.py::TestVoiceChannelCommands::test_voice_status_reports_realtime_latency_metrics"
        in report["async_oracle_acceptance"]["status_reports_running_and_queued_without_oracle_call"]["test_refs"]
    )
    assert (
        "tests/gateway/test_voice_command.py::TestVoiceChannelCommands::test_voice_jobs_reports_oracle_job_snapshot"
        in report["async_oracle_acceptance"]["status_reports_running_and_queued_without_oracle_call"]["test_refs"]
    )
    assert (
        report["async_oracle_acceptance"]["status_reports_running_and_queued_without_oracle_call"][
            "test_ref_count"
        ]
        == len(
            report["async_oracle_acceptance"]["status_reports_running_and_queued_without_oracle_call"][
                "test_refs"
            ]
        )
    )
    assert (
        "tests/agent/test_realtime_voice.py::test_kame_engine_attaches_interpreter_evidence_to_queued_async_oracle_job"
        in report["async_oracle_acceptance"]["job_control_updates_reach_oracle"]["test_refs"]
    )
    assert (
        "tests/agent/test_realtime_voice.py::test_kame_engine_merges_sequential_queued_transcript_hypotheses_before_start"
        in report["async_oracle_acceptance"]["job_control_updates_reach_oracle"]["test_refs"]
    )
    assert (
        "tests/agent/test_realtime_voice.py::test_session_client_interface_oracle_request_submits_external_kame_job"
        in report["async_oracle_acceptance"]["external_frontend_bridge_submits_oracle_job"]["test_refs"]
    )
    assert (
        report["async_oracle_acceptance"]["external_frontend_bridge_submits_oracle_job"][
            "test_ref_count"
        ]
        == len(
            report["async_oracle_acceptance"]["external_frontend_bridge_submits_oracle_job"][
                "test_refs"
            ]
        )
    )
    assert (
        "tests/gateway/test_voice_command.py::TestDiscordVoiceChannelMethods::test_leave_voice_channel_cleans_up"
        in report["async_oracle_acceptance"]["discord_session_cleanup_preserves_oracle_state"]["test_refs"]
    )
    runtime_action_gate = report["async_oracle_acceptance"][
        "runtime_kame_action_gate_enforces_promoted_evidence"
    ]
    assert (
        "tests/agent/test_realtime_voice.py::test_async_oracle_unflagged_high_risk_tool_call_fails_closed"
        in runtime_action_gate["test_refs"]
    )
    assert (
        "tests/agent/test_realtime_voice.py::test_async_oracle_unflagged_nested_high_risk_tool_call_fails_closed"
        in runtime_action_gate["test_refs"]
    )
    assert (
        "tests/agent/test_realtime_voice.py::test_async_oracle_unflagged_high_risk_tool_result_fails_closed"
        in runtime_action_gate["test_refs"]
    )
    assert runtime_action_gate["test_ref_count"] == len(runtime_action_gate["test_refs"])


def test_voice_operator_validation_rejects_stale_async_acceptance_test_ref():
    report = _voice_operator_report()
    row = report["async_oracle_acceptance"]["status_reports_running_and_queued_without_oracle_call"]
    row["test_refs"] = ["tests/gateway/test_voice_command.py::test_voice_jobs_reports_oracle_job_snapshot"]
    row["test_ref_count"] = 1

    issues = validate_voice_operator_report(report)

    assert (
        "invalid_async_oracle_acceptance_test_ref:"
        "status_reports_running_and_queued_without_oracle_call:"
        "tests/gateway/test_voice_command.py::test_voice_jobs_reports_oracle_job_snapshot"
    ) in issues


def test_voice_operator_validation_rejects_missing_current_async_acceptance_test_ref():
    report = _voice_operator_report()
    row = report["async_oracle_acceptance"]["job_control_updates_reach_oracle"]
    row["test_refs"] = row["test_refs"][:-1]
    row["test_ref_count"] = len(row["test_refs"])

    issues = validate_voice_operator_report(report)

    assert "stale_async_oracle_acceptance:job_control_updates_reach_oracle:test_refs" in issues


def test_voice_operator_validation_rejects_static_acceptance_runtime_claim():
    report = _voice_operator_report()
    row = report["async_oracle_acceptance"]["discord_session_cleanup_preserves_oracle_state"]
    row["verification_mode"] = "static_focused_test_reference_inventory"
    row["runtime_verified_by_this_report"] = True

    issues = validate_voice_operator_report(report)

    assert "invalid_async_oracle_acceptance_runtime_claim:discord_session_cleanup_preserves_oracle_state" in issues


def test_voice_operator_validation_rejects_scheduler_only_async_concurrency():
    async_oracle_smoke = _async_oracle_smoke_payload()
    async_oracle_smoke["worker_overlap_proved"] = False
    async_oracle_smoke["max_worker_overlap"] = 0
    report = build_voice_operator_report(
        _smoke_payload(),
        async_oracle_smoke=async_oracle_smoke,
    )

    issues = validate_voice_operator_report(report)

    assert "missing_async_oracle_coverage:four_jobs_ran_concurrently" in issues
    assert "missing_async_oracle_acceptance:four_oracle_jobs_reflex_responsive" in issues


def test_write_voice_operator_report_artifacts(tmp_path):
    report = _voice_operator_report()
    paths = write_voice_operator_report(tmp_path, report)

    required_paths = {
        "async_oracle_smoke_json",
        "discord_session_cleanup_smoke_json",
        "events_jsonl",
        "interpreter_request_packet_json",
        "json",
        "live_evidence_example",
        "live_evidence_scaffold_manifest",
        "live_evidence_template",
        "live_probe_closure_json",
        "live_probe_closure_markdown",
        "markdown",
        "sidecar_fail_closed_smoke_json",
        "smoke_json",
        "tool_disclosure_smoke_json",
        "ephemeral_tool_router_smoke_json",
    }
    assert required_paths <= set(paths)
    assert {
        "scaffold_discord_live_probe",
        "scaffold_sidecar_session",
        "scaffold_live_turn",
    } <= set(paths)
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    smoke = json.loads(Path(paths["smoke_json"]).read_text(encoding="utf-8"))
    async_oracle_smoke = json.loads(Path(paths["async_oracle_smoke_json"]).read_text(encoding="utf-8"))
    discord_cleanup_smoke = json.loads(Path(paths["discord_session_cleanup_smoke_json"]).read_text(encoding="utf-8"))
    sidecar_fail_closed_smoke = json.loads(Path(paths["sidecar_fail_closed_smoke_json"]).read_text(encoding="utf-8"))
    tool_disclosure_smoke = json.loads(Path(paths["tool_disclosure_smoke_json"]).read_text(encoding="utf-8"))
    ephemeral_tool_router_smoke = json.loads(
        Path(paths["ephemeral_tool_router_smoke_json"]).read_text(encoding="utf-8")
    )
    interpreter_request_packet = json.loads(Path(paths["interpreter_request_packet_json"]).read_text(encoding="utf-8"))
    live_template = json.loads(Path(paths["live_evidence_template"]).read_text(encoding="utf-8"))
    live_example = json.loads(Path(paths["live_evidence_example"]).read_text(encoding="utf-8"))
    live_scaffold_manifest_path = Path(paths["live_evidence_scaffold_manifest"])
    live_scaffold_manifest = json.loads(live_scaffold_manifest_path.read_text(encoding="utf-8"))
    live_closure = json.loads(Path(paths["live_probe_closure_json"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    closure_markdown = Path(paths["live_probe_closure_markdown"]).read_text(encoding="utf-8")
    events = Path(paths["events_jsonl"]).read_text(encoding="utf-8").splitlines()
    assert payload["schema_version"] == "voiceops.milestone1.voice_operator.v1"
    assert payload["status"] == "needs_live_probe"
    assert payload["missing_live_gates"] == [
        "discord_join",
        "discord_playback",
        "live_receiver",
        "production_sidecar",
        "live_turn",
    ]
    assert smoke["ok"] is True
    assert tool_disclosure_smoke == payload["tool_disclosure_smoke"]
    assert ephemeral_tool_router_smoke == payload["ephemeral_tool_router_smoke"]
    assert interpreter_request_packet == payload["interpreter_request_packet"]
    assert interpreter_request_packet["prompt_input_order"] == [
        "raw_audio",
        "metadata",
        "reflex",
        "transcript_hypotheses",
    ]
    assert interpreter_request_packet["interpreter_input_order"] == interpreter_request_packet["prompt_input_order"]
    assert interpreter_request_packet["interpreter_prompt_policy"]["version"] == "raw_audio_compare_v1"
    assert interpreter_request_packet["prompt_policy"] == interpreter_request_packet["interpreter_prompt_policy"]
    assert interpreter_request_packet["audio"]["authority"] == "primary_audio"
    assert interpreter_request_packet["transcript_hypotheses"][0]["authority"] == "hypothesis"
    assert interpreter_request_packet["transcript_hypotheses"][0]["tool_authority"] is False
    assert sidecar_fail_closed_smoke["ok"] is True
    assert sidecar_fail_closed_smoke["scenario"] == "sidecar_send_fail_closed_after_acceptance"
    assert async_oracle_smoke["ok"] is True
    assert async_oracle_smoke["max_running"] == 4
    assert discord_cleanup_smoke["ok"] is True
    assert discord_cleanup_smoke["cancel_all_before_session_closed"] is True
    assert live_template["schema_version"] == "voiceops.milestone1.live_voice_evidence.v1"
    assert live_example["example_only"] is True
    assert "example_only_evidence_not_accepted" in validate_live_probe_evidence(live_example)["issues"]
    assert live_scaffold_manifest["example_only"] is True
    assert live_scaffold_manifest["reports"]["sidecar_session"] == "sections/sidecar-session.json"
    scaffold_evidence = _load_live_evidence([live_scaffold_manifest_path])
    assert scaffold_evidence["overall_status"] == "partial_live_evidence"
    assert "example_only_evidence_not_accepted" in scaffold_evidence["issues"]
    assert "live_evidence_manifest:sidecar_session:example_only_evidence_not_accepted" in scaffold_evidence["issues"]
    assert all("source_artifact_not_found" not in issue for issue in scaffold_evidence["issues"])
    assert json.loads(Path(paths["scaffold_live_turn"]).read_text(encoding="utf-8"))["kind"] == "live_turn"
    assert live_closure["schema_version"] == "voiceops.milestone1.live_probe_closure.v1"
    assert live_closure["live_evidence_scaffold_manifest"] == "live-voice-evidence-scaffold/manifest.json"
    assert live_closure["evidence_contract"]["manifest_schema_version"] == (
        "voiceops.realtime_voice_live_evidence_manifest.v1"
    )
    assert live_closure["evidence_contract"]["required_section_field"] == "source_artifact"
    assert live_closure["evidence_contract"]["source_artifacts_must_exist"] is True
    assert live_closure["evidence_contract"]["source_artifacts_must_be_json"] is True
    assert live_closure["evidence_contract"]["source_artifacts_reject_secret_or_phone_values"] is True
    assert live_closure["evidence_contract"]["source_artifacts_reject_voice_capability_denials"] is True
    assert live_closure["evidence_contract"]["template_source_artifacts_accepted"] is False
    assert set(live_closure["evidence_contract"]["required_transcript_hypothesis_fields"]) >= {
        "text_digest",
        "role",
        "promotion_required",
        "latency_ms",
        "confidence",
        "speaker_or_actor_ref",
        "channel_or_surface_ref",
    }
    assert live_closure["evidence_contract"]["transcript_hypothesis_contract"] == {
        "role": "witness_context",
        "authority": "hypothesis",
        "promotion_required": "interpreter_promoted_or_oracle_promoted",
        "tool_authority": False,
    }
    assert live_closure["evidence_contract"]["collector_attestation_required_for_live_readiness"] is True
    assert live_closure["evidence_contract"]["collector_attestation_required_fields"] == [
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
    ]
    assert live_closure["evidence_contract"]["placeholder_collector_attestation_accepted"] is False
    assert "kind/evidence_type" in live_closure["evidence_contract"]["manifest_report_identity"]
    assert "standalone non-expanded evidence files" in live_closure["evidence_contract"]["standalone_report_identity"]
    assert "evidence_shapes" not in live_closure
    example_shapes = live_closure["non_accepted_example_shapes"]
    assert example_shapes["discord_live_probe"]["kind"] == "discord_live_probe"
    assert example_shapes["discord_live_probe"]["example_only"] is True
    assert example_shapes["discord_live_probe"]["source_artifact"] == "sections/discord-live-probe-source.json"
    assert example_shapes["discord_live_probe"]["collector_attestation"]["example_only"] is True
    assert example_shapes["discord_live_probe"]["require_inbound"] is True
    assert example_shapes["sidecar_session"]["kind"] == "sidecar_session"
    assert example_shapes["sidecar_session"]["example_only"] is True
    assert example_shapes["sidecar_session"]["source_artifact"] == "sections/sidecar-session-source.json"
    assert example_shapes["sidecar_session"]["collector_attestation"]["example_only"] is True
    assert example_shapes["sidecar_session"]["shutdown_bounded"] is True
    assert example_shapes["sidecar_session"]["shutdown_timed_out"] is False
    assert example_shapes["sidecar_session"]["sidecar_mode"] == "production"
    assert example_shapes["sidecar_session"]["fallback_reason"] == "none"
    assert example_shapes["sidecar_session"]["healthcheck_observed"] is True
    assert example_shapes["sidecar_session"]["provider_transport_observed"] is True
    assert example_shapes["sidecar_session"]["session_id_redacted"] is True
    assert example_shapes["sidecar_session"]["latency_metrics_ms"]["session_start_ms"] == 110
    assert example_shapes["sidecar_session"]["latency_metrics_ms"]["shutdown_ms"] == 80
    assert example_shapes["live_turn"]["kind"] == "live_turn"
    assert example_shapes["live_turn"]["example_only"] is True
    assert example_shapes["live_turn"]["source_artifact"] == "sections/live-turn-source.json"
    assert example_shapes["live_turn"]["collector_attestation"]["example_only"] is True
    example_shape_validation = validate_live_probe_evidence(
        {
            "schema_version": "voiceops.milestone1.live_voice_evidence.v1",
            **example_shapes,
        }
    )
    assert example_shape_validation["overall_status"] == "partial_live_evidence"
    assert "discord_live_probe:example_only_evidence_not_accepted" in example_shape_validation["issues"]
    assert "sidecar_session:example_only_evidence_not_accepted" in example_shape_validation["issues"]
    assert "live_turn:example_only_evidence_not_accepted" in example_shape_validation["issues"]
    assert "hermes_cli.realtime_voice_live_evidence" in live_closure["recommended_collection"]["live_bundle_manifest"]
    assert "--run-doctor-report" in live_closure["recommended_collection"]["live_bundle_manifest"]
    assert "--require-inbound" in live_closure["recommended_collection"]["live_bundle_manifest"]
    assert "--audit-only" in live_closure["recommended_collection"]["audit_bundle_no_write"]
    assert "--discord-live-probe-evidence" in live_closure["recommended_collection"]["audit_bundle_no_write"]
    assert "--validate-live-evidence" in live_closure["recommended_collection"]["validate_bundle_offline"]
    assert "--discord-live-probe-evidence" in live_closure["recommended_collection"]["validate_bundle_offline"]
    assert "manifest.json" in live_closure["recommended_collection"]["ingest"]
    assert json.loads(events[0])["event_id"] == "voice-m1-001"
    assert "VoiceOps Milestone 1 Voice Operator" in markdown
    assert "Proofs" in markdown
    assert "Live Probe Boundary" in markdown
    assert "Supplied Live Evidence" in markdown
    assert "VoiceOps Milestone 1 Live Probe Closure" in closure_markdown
    assert "Non-Accepted Example Shapes" in closure_markdown
    assert "They are rejected by validation" in closure_markdown
    assert "live-voice-evidence-scaffold/manifest.json" in closure_markdown
    assert "voiceops.realtime_voice_live_evidence_manifest.v1" in closure_markdown
    assert "source_artifact" in closure_markdown
    assert "collector_attestation" in closure_markdown
    assert "collector_name" in closure_markdown
    assert "collector_version" in closure_markdown
    assert "command_argv" in closure_markdown
    assert "redacted_artifact_sha256" in closure_markdown
    assert "parent_manifest_sha256" in closure_markdown
    assert "text_digest" in closure_markdown
    assert "speaker_or_actor_ref" in closure_markdown
    assert "kind/evidence_type" in closure_markdown
    assert "--validate-live-evidence" in closure_markdown
    assert "artifacts/realtime-voice-evidence/live-current/sidecar-session.json" in closure_markdown
    assert "sidecar_session" in closure_markdown
    assert "live_turn" in closure_markdown
    assert "sidecar-session.json" in closure_markdown
    assert "raw transcript text" in closure_markdown
    assert "hand-edit manifest.json" in closure_markdown
    assert "example_only" in closure_markdown


def test_live_evidence_classifies_partial_discord_probe_without_inbound():
    evidence = {
        "kind": "discord_live_probe",
        "ok": True,
        "connect_perm": True,
        "speak_perm": True,
        "connected": True,
        "opus_loaded": True,
        "accepted_audio_source": True,
        "played": True,
        "playing_during_probe": True,
        "receiver_started": True,
        "receiver_frames": 0,
        "receiver_speech_start": 0,
        "inbound_observed": False,
        "disconnected": True,
        "require_inbound": True,
    }

    result = validate_live_probe_evidence(evidence)

    assert result["overall_status"] == "partial_live_evidence"
    assert result["discord_live_probe"]["join_ok"] is True
    assert result["discord_live_probe"]["playback_ok"] is True
    assert result["discord_live_probe"]["inbound_observed"] is False
    assert "missing_schema_version" in result["issues"]
    assert "discord_live_probe:missing_source_artifact" in result["issues"]
    assert "sidecar_session:missing_source_artifact" in result["issues"]
    assert "live_turn:missing_source_artifact" in result["issues"]
    assert "discord_live_probe:inbound_not_observed" in result["issues"]


def test_live_evidence_example_is_not_accepted_as_proof():
    result = validate_live_probe_evidence(build_live_probe_evidence_example())

    assert result["overall_status"] == "partial_live_evidence"
    assert "example_only_evidence_not_accepted" in result["issues"]


def test_voice_operator_accepts_complete_supplied_live_evidence_without_changing_safety_mode(tmp_path):
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["source_artifact"] = str(tmp_path / "discord-live-probe.json")
    evidence["sidecar_session"]["source_artifact"] = str(tmp_path / "sidecar-session.json")
    evidence["live_turn"]["source_artifact"] = str(tmp_path / "live-turn.json")
    _write_attested_section(tmp_path / "discord-live-probe.json", evidence["discord_live_probe"], "discord_live_probe")
    _write_attested_section(tmp_path / "sidecar-session.json", evidence["sidecar_session"], "sidecar_session")
    _write_attested_section(tmp_path / "live-turn.json", evidence["live_turn"], "live_turn")
    live_evidence = validate_live_probe_evidence(evidence)
    report = _voice_operator_report(live_evidence=live_evidence)

    assert report["mode"]["discord_network"] is False
    assert report["mode"]["env_secret_reads"] is False
    assert report["requirements"]["live_discord_join"] is False
    assert report["requirements"]["live_evidence_supplied"] is True
    assert report["live_evidence"]["overall_status"] == "live_evidence_supplied_not_readiness_claim"
    assert report["live_probe_required_for_completion"]["missing_gates"] == []
    assert report["proofs"]["live_evidence"]["ok"] is True
    assert live_evidence["live_turn"]["kame_lineage_ids_complete"] is True
    assert live_evidence["live_turn"]["kame_lineage_consistent"] is True
    assert live_evidence["live_turn"]["witness_binding_consistent"] is True
    assert live_evidence["live_turn"]["turn_id"] == "voiceops-live-turn-budget"
    assert live_evidence["live_turn"]["audio_segment_ref"] == "artifact://redacted/voiceops-live-turn-budget.wav"
    assert live_evidence["live_turn"]["evidence_bundle_id"] == "kame-evidence-live-turn-budget"
    assert live_evidence["live_turn"]["evidence_merge_key"] == "kame-merge-live-turn-budget"
    assert live_evidence["live_turn"]["raw_audio_interpreter_evidence_observed"] is True
    assert live_evidence["live_turn"]["transcript_hypotheses_observed"] is True
    assert live_evidence["live_turn"]["transcript_only_witness_rejected_for_full_kame"] is False
    assert live_evidence["live_turn"]["witness_packet_observed"] is True
    assert live_evidence["live_turn"]["active_partial_absent"] is True
    assert live_evidence["live_turn"]["transcript_hypothesis_metadata_observed"] is True
    assert live_evidence["live_turn"]["transcript_hypothesis_adjudication_observed"] is True
    assert live_evidence["live_turn"]["interpreter_input_order_observed"] is True
    assert live_evidence["live_turn"]["interpreter_prompt_policy_observed"] is True
    assert live_evidence["live_turn"]["interpreter_adjudication_observed"] is True
    assert live_evidence["live_turn"]["promoted_evidence_observed"] is True
    assert live_evidence["live_turn"]["unpromoted_witness_sinks_clean_observed"] is True
    assert not any("collector_attestation" in issue for issue in live_evidence["issues"])


def test_live_evidence_requires_concrete_kame_lineage_ids():
    evidence = _complete_live_evidence()
    evidence["live_turn"]["turn_id"] = ""
    evidence["live_turn"]["audio_segment_ref"] = None
    evidence["live_turn"].pop("evidence_bundle_id")
    evidence["live_turn"]["evidence_merge_key"] = " "

    live_evidence = validate_live_probe_evidence(evidence)

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "live_turn:missing_turn_id" in live_evidence["issues"]
    assert "live_turn:missing_audio_segment_ref" in live_evidence["issues"]
    assert "live_turn:missing_evidence_bundle_id" in live_evidence["issues"]
    assert "live_turn:missing_evidence_merge_key" in live_evidence["issues"]
    assert live_evidence["live_turn"]["kame_lineage_ids_complete"] is False
    assert live_evidence["live_turn"]["ok"] is False


def test_live_evidence_rejects_conflicting_witness_kame_lineage():
    evidence = _complete_live_evidence()
    evidence["live_turn"]["transcript_hypotheses"][0][
        "audio_segment_ref"
    ] = "artifact://redacted/stale-or-wrong-speaker-cut.wav"

    live_evidence = validate_live_probe_evidence(evidence)

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "live_turn:kame_lineage_conflict_audio_segment_ref" in live_evidence["issues"]
    assert live_evidence["live_turn"]["kame_lineage_ids_complete"] is True
    assert live_evidence["live_turn"]["kame_lineage_consistent"] is False
    assert live_evidence["live_turn"]["ok"] is False


def test_live_evidence_rejects_conflicting_witness_speaker_channel_binding():
    evidence = _complete_live_evidence()
    evidence["live_turn"]["speaker"] = {
        "platform": "discord",
        "channel_user_id": "jetha-redacted",
    }
    evidence["live_turn"]["channel"] = {
        "transport": "discord_voice",
        "channel_id": "general-redacted",
    }
    evidence["live_turn"]["transcript_hypotheses"][0]["speaker_guess"] = {
        "platform": "discord",
        "channel_user_id": "other-human",
    }
    evidence["live_turn"]["transcript_hypotheses"][0]["channel_guess"] = {
        "transport": "discord_voice",
        "channel_id": "other-channel",
    }

    live_evidence = validate_live_probe_evidence(evidence)

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "live_turn:transcript_hypothesis_0_speaker_mismatch" in live_evidence["issues"]
    assert "live_turn:transcript_hypothesis_0_channel_mismatch" in live_evidence["issues"]
    assert live_evidence["live_turn"]["witness_binding_consistent"] is False
    assert live_evidence["live_turn"]["ok"] is False


def test_live_evidence_accepts_rejected_witness_speaker_channel_mismatch_with_reasons(tmp_path):
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["source_artifact"] = str(tmp_path / "discord-live-probe.json")
    evidence["sidecar_session"]["source_artifact"] = str(tmp_path / "sidecar-session.json")
    evidence["live_turn"]["source_artifact"] = str(tmp_path / "live-turn.json")
    evidence["live_turn"]["speaker"] = {
        "platform": "discord",
        "channel_user_id": "jetha-redacted",
    }
    evidence["live_turn"]["channel"] = {
        "transport": "discord_voice",
        "channel_id": "general-redacted",
    }
    evidence["live_turn"]["transcript_hypotheses"][0].update(
        {
            "speaker_guess": {
                "platform": "discord",
                "channel_user_id": "other-human",
            },
            "channel_guess": {
                "transport": "discord_voice",
                "channel_id": "other-channel",
            },
            "adjudication": "rejected_or_diagnostic_only",
            "rejection_reasons": ["wrong_speaker", "wrong_channel"],
        }
    )
    evidence["live_turn"]["interpreter_adjudication_outcomes"] = [
        "rejected_or_diagnostic_only"
    ]
    _write_attested_section(tmp_path / "discord-live-probe.json", evidence["discord_live_probe"], "discord_live_probe")
    _write_attested_section(tmp_path / "sidecar-session.json", evidence["sidecar_session"], "sidecar_session")
    _write_attested_section(tmp_path / "live-turn.json", evidence["live_turn"], "live_turn")

    live_evidence = validate_live_probe_evidence(evidence)

    assert "live_turn:transcript_hypothesis_0_speaker_mismatch" not in live_evidence["issues"]
    assert "live_turn:transcript_hypothesis_0_channel_mismatch" not in live_evidence["issues"]
    assert live_evidence["live_turn"]["witness_binding_consistent"] is True
    assert live_evidence["live_turn"]["rejected_witness_reasons_observed"] is True
    assert live_evidence["live_turn"]["ok"] is True


def test_live_evidence_rejects_active_partial_witness_hypothesis():
    evidence = _complete_live_evidence()
    evidence["live_turn"]["transcript_hypotheses"][0]["partial"] = True

    live_evidence = validate_live_probe_evidence(evidence)

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "live_turn:transcript_hypothesis_0_active_partial_not_superseded" in live_evidence["issues"]
    assert live_evidence["live_turn"]["active_partial_absent"] is False
    assert live_evidence["live_turn"]["ok"] is False


def test_live_evidence_rejects_transcript_only_witness_for_full_kame():
    evidence = _complete_live_evidence()
    evidence["live_turn"]["audio_segment_ref_observed"] = False
    evidence["live_turn"]["interpreter_evidence_observed"] = False
    evidence["live_turn"]["transcript_hypotheses_labeled"] = True
    evidence["live_turn"]["transcript_observed"] = True

    live_evidence = validate_live_probe_evidence(evidence)

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "live_turn:audio_segment_ref_observed_not_true" in live_evidence["issues"]
    assert "live_turn:interpreter_evidence_observed_not_true" in live_evidence["issues"]
    assert (
        "live_turn:transcript_only_witness_without_raw_audio_interpreter_evidence"
        in live_evidence["issues"]
    )
    assert live_evidence["live_turn"]["raw_audio_interpreter_evidence_observed"] is False
    assert live_evidence["live_turn"]["transcript_hypotheses_observed"] is True
    assert live_evidence["live_turn"]["transcript_only_witness_rejected_for_full_kame"] is True
    assert live_evidence["live_turn"]["ok"] is False


def test_live_evidence_rejects_labeled_hypothesis_without_concrete_packet():
    evidence = _complete_live_evidence()
    evidence["live_turn"].pop("transcript_hypotheses")
    evidence["live_turn"].pop("interpreter_input_order")
    evidence["live_turn"].pop("interpreter_prompt_policy")
    evidence["live_turn"].pop("interpreter_adjudication_outcomes")
    evidence["live_turn"].pop("promoted_evidence_authority")
    evidence["live_turn"]["witness_arrival_phases"] = []

    live_evidence = validate_live_probe_evidence(evidence)

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "live_turn:missing_transcript_hypotheses" in live_evidence["issues"]
    assert "live_turn:missing_witness_arrival_phases" in live_evidence["issues"]
    assert "live_turn:missing_interpreter_input_order" in live_evidence["issues"]
    assert "live_turn:missing_interpreter_prompt_policy" in live_evidence["issues"]
    assert "live_turn:missing_interpreter_adjudication_outcomes" in live_evidence["issues"]
    assert "live_turn:missing_promoted_evidence_authority" in live_evidence["issues"]
    assert live_evidence["live_turn"]["witness_packet_observed"] is False
    assert live_evidence["live_turn"]["interpreter_input_order_observed"] is False
    assert live_evidence["live_turn"]["interpreter_prompt_policy_observed"] is False
    assert live_evidence["live_turn"]["interpreter_adjudication_observed"] is False
    assert live_evidence["live_turn"]["promoted_evidence_observed"] is False
    assert live_evidence["live_turn"]["ok"] is False


def test_live_evidence_rejects_missing_or_legacy_interpreter_prompt_policy():
    missing = _complete_live_evidence()
    missing["live_turn"].pop("interpreter_prompt_policy")

    missing_result = validate_live_probe_evidence(missing)

    assert "live_turn:missing_interpreter_prompt_policy" in missing_result["issues"]
    assert missing_result["live_turn"]["interpreter_prompt_policy_observed"] is False
    assert missing_result["live_turn"]["ok"] is False

    legacy = _complete_live_evidence()
    legacy["live_turn"]["interpreter_prompt_policy"] = {
        "version": "legacy_transcript_first",
        "primary_evidence": "transcript_hypotheses",
        "transcript_hypotheses_authority": "candidate_transcript",
    }

    legacy_result = validate_live_probe_evidence(legacy)

    assert "live_turn:interpreter_prompt_policy_version_mismatch" in legacy_result["issues"]
    assert "live_turn:interpreter_prompt_policy_primary_evidence_mismatch" in legacy_result["issues"]
    assert (
        "live_turn:interpreter_prompt_policy_transcript_hypotheses_authority_mismatch"
        in legacy_result["issues"]
    )
    assert legacy_result["live_turn"]["interpreter_prompt_policy_observed"] is False
    assert legacy_result["live_turn"]["ok"] is False


def test_live_evidence_rejects_authoritative_or_unphased_witness_hypothesis():
    evidence = _complete_live_evidence()
    evidence["live_turn"]["transcript_hypotheses"] = [
        {
            "kind": "frontend_witness_hypothesis",
            "source": "moshi",
            "text": "[redacted witness hypothesis]",
            "authority": "interpreter_promoted",
            "tool_authority": True,
        }
    ]
    evidence["live_turn"]["interpreter_input_order"] = [
        "transcript_hypotheses",
        "raw_audio",
        "metadata",
        "reflex",
    ]

    live_evidence = validate_live_probe_evidence(evidence)

    assert "live_turn:transcript_hypothesis_0_authority_not_hypothesis" in live_evidence["issues"]
    assert "live_turn:transcript_hypothesis_0_tool_authority_not_false" in live_evidence["issues"]
    assert "live_turn:transcript_hypothesis_0_missing_arrival_phase" in live_evidence["issues"]
    assert "live_turn:transcript_hypothesis_0_missing_adjudication" in live_evidence["issues"]
    assert "live_turn:interpreter_input_order_mismatch" in live_evidence["issues"]
    assert live_evidence["live_turn"]["transcript_hypothesis_adjudication_observed"] is False
    assert live_evidence["live_turn"]["ok"] is False


def test_live_evidence_rejects_incomplete_witness_hypothesis_metadata():
    evidence = _complete_live_evidence()
    hypothesis = evidence["live_turn"]["transcript_hypotheses"][0]
    hypothesis.pop("text_digest")
    hypothesis.pop("speaker_or_actor_ref")
    hypothesis["role"] = "verified_transcript"
    hypothesis["promotion_required"] = "none"
    hypothesis["latency_ms"] = -1
    hypothesis["confidence"] = 1.2

    live_evidence = validate_live_probe_evidence(evidence)

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert (
        "live_turn:transcript_hypothesis_0_missing_metadata:speaker_or_actor_ref,text_digest"
        in live_evidence["issues"]
    )
    assert "live_turn:transcript_hypothesis_0_role_not_witness_context" in live_evidence["issues"]
    assert "live_turn:transcript_hypothesis_0_promotion_required_invalid" in live_evidence["issues"]
    assert "live_turn:transcript_hypothesis_0_latency_ms_invalid" in live_evidence["issues"]
    assert "live_turn:transcript_hypothesis_0_confidence_invalid" in live_evidence["issues"]
    assert live_evidence["live_turn"]["transcript_hypothesis_metadata_observed"] is False
    assert live_evidence["live_turn"]["witness_packet_observed"] is False
    assert live_evidence["live_turn"]["ok"] is False


def test_live_evidence_rejects_invalid_witness_text_digest():
    evidence = _complete_live_evidence()
    evidence["live_turn"]["transcript_hypotheses"][0]["text_digest"] = "not-a-sha"

    live_evidence = validate_live_probe_evidence(evidence)

    assert "live_turn:transcript_hypothesis_0_invalid_text_digest" in live_evidence["issues"]
    assert live_evidence["live_turn"]["transcript_hypothesis_metadata_observed"] is False
    assert live_evidence["live_turn"]["ok"] is False


def test_live_evidence_rejects_hypothesis_without_bound_adjudication():
    missing = _complete_live_evidence()
    missing["live_turn"]["transcript_hypotheses"][0].pop("adjudication")

    missing_result = validate_live_probe_evidence(missing)

    assert "live_turn:transcript_hypothesis_0_missing_adjudication" in missing_result["issues"]
    assert missing_result["live_turn"]["transcript_hypothesis_adjudication_observed"] is False
    assert missing_result["live_turn"]["witness_packet_observed"] is False
    assert missing_result["live_turn"]["ok"] is False

    invalid = _complete_live_evidence()
    invalid["live_turn"]["transcript_hypotheses"][0]["adjudication"] = "trusted_without_audio"

    invalid_result = validate_live_probe_evidence(invalid)

    assert "live_turn:transcript_hypothesis_0_invalid_adjudication" in invalid_result["issues"]
    assert invalid_result["live_turn"]["transcript_hypothesis_adjudication_observed"] is False
    assert invalid_result["live_turn"]["witness_packet_observed"] is False
    assert invalid_result["live_turn"]["ok"] is False


def test_live_evidence_rejects_diagnostic_hypothesis_without_typed_reasons():
    missing = _complete_live_evidence()
    missing["live_turn"]["transcript_hypotheses"][0]["adjudication"] = (
        "rejected_or_diagnostic_only"
    )
    missing["live_turn"]["interpreter_adjudication_outcomes"] = ["rejected_or_diagnostic_only"]

    missing_result = validate_live_probe_evidence(missing)

    assert (
        "live_turn:transcript_hypothesis_0_missing_rejection_reasons"
        in missing_result["issues"]
    )
    assert missing_result["live_turn"]["rejected_witness_reasons_observed"] is False
    assert missing_result["live_turn"]["ok"] is False

    invalid = _complete_live_evidence()
    invalid["live_turn"]["transcript_hypotheses"][0]["adjudication"] = (
        "rejected_or_diagnostic_only"
    )
    invalid["live_turn"]["transcript_hypotheses"][0]["rejection_reasons"] = [
        "trusted_text_arrived_first"
    ]
    invalid["live_turn"]["interpreter_adjudication_outcomes"] = ["rejected_or_diagnostic_only"]

    invalid_result = validate_live_probe_evidence(invalid)

    assert "live_turn:transcript_hypothesis_0_invalid_rejection_reason" in invalid_result["issues"]
    assert invalid_result["live_turn"]["rejected_witness_reasons_observed"] is False
    assert invalid_result["live_turn"]["ok"] is False

    valid = _complete_live_evidence()
    valid["live_turn"]["transcript_hypotheses"][0]["adjudication"] = (
        "rejected_or_diagnostic_only"
    )
    valid["live_turn"]["transcript_hypotheses"][0]["rejection_reasons"] = [
        "wrong_speaker"
    ]
    valid["live_turn"]["interpreter_adjudication_outcomes"] = ["rejected_or_diagnostic_only"]

    valid_result = validate_live_probe_evidence(valid)

    assert "live_turn:transcript_hypothesis_0_missing_rejection_reasons" not in valid_result["issues"]
    assert "live_turn:transcript_hypothesis_0_invalid_rejection_reason" not in valid_result["issues"]
    assert valid_result["live_turn"]["rejected_witness_reasons_observed"] is True
    assert valid_result["live_turn"]["ok"] is True


def test_live_evidence_rejects_interpreter_adjudication_outcome_mismatch():
    evidence = _complete_live_evidence()
    evidence["live_turn"]["transcript_hypotheses"][0]["adjudication"] = (
        "rejected_or_diagnostic_only"
    )
    evidence["live_turn"]["transcript_hypotheses"][0]["rejection_reasons"] = [
        "waveform_conflict"
    ]
    evidence["live_turn"]["interpreter_adjudication_outcomes"] = [
        "accepted_as_supporting_evidence"
    ]

    live_evidence = validate_live_probe_evidence(evidence)

    assert "live_turn:interpreter_adjudication_outcomes_mismatch" in live_evidence["issues"]
    assert live_evidence["live_turn"]["interpreter_adjudication_observed"] is False
    assert live_evidence["live_turn"]["ok"] is False


def test_live_evidence_rejects_unpromoted_hypothesis_without_sink_checks():
    evidence = _complete_live_evidence()
    evidence["live_turn"].pop("unpromoted_witness_sink_checks")
    evidence["live_turn"].pop("unpromoted_witness_sink_values")

    live_evidence = validate_live_probe_evidence(evidence)

    assert "live_turn:missing_unpromoted_witness_sink_checks" in live_evidence["issues"]
    assert "live_turn:missing_unpromoted_witness_sink_values" in live_evidence["issues"]
    assert live_evidence["live_turn"]["unpromoted_witness_sinks_clean_observed"] is False
    assert live_evidence["live_turn"]["ok"] is False


def test_live_evidence_rejects_unpromoted_hypothesis_sink_contamination():
    evidence = _complete_live_evidence()
    evidence["live_turn"]["unpromoted_witness_sink_checks"]["spend_clean"] = False
    evidence["live_turn"]["unpromoted_witness_sink_values"] = {
        "spend": "[redacted witness hypothesis]"
    }

    live_evidence = validate_live_probe_evidence(evidence)

    assert "live_turn:unpromoted_witness_sink_spend_not_clean" in live_evidence["issues"]
    assert "live_turn:unpromoted_witness_sink_values_not_empty" in live_evidence["issues"]
    assert live_evidence["live_turn"]["unpromoted_witness_sinks_clean_observed"] is False
    assert live_evidence["live_turn"]["ok"] is False


def test_live_evidence_rejects_stale_source_artifact_attestation_hash(tmp_path):
    evidence = _complete_live_evidence()
    source_path = tmp_path / "discord-live-probe.json"
    evidence["discord_live_probe"]["source_artifact"] = str(source_path)
    _write_attested_section(source_path, evidence["discord_live_probe"], "discord_live_probe")
    source_path.write_text(
        json.dumps({"kind": "discord_live_probe", "redacted": True, "updated": True}),
        encoding="utf-8",
    )

    live_evidence = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:collector_attestation_redacted_sha256_mismatch" in live_evidence["issues"]


def test_live_evidence_rejects_inverted_collector_attestation_window():
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["collector_attestation"]["started_at"] = "2026-06-29T00:00:02Z"
    evidence["discord_live_probe"]["collector_attestation"]["finished_at"] = "2026-06-29T00:00:01Z"

    live_evidence = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:collector_attestation_invalid:timestamp_window" in live_evidence["issues"]


def test_live_evidence_rejects_sensitive_collector_attestation_command_argv():
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["collector_attestation"]["command_argv"] = [
        "voiceops-live-collector",
        "--notify=+15551234567",
    ]

    live_evidence = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:collector_attestation_secret_or_phone_like_command_argv" in live_evidence["issues"]


def test_live_evidence_allows_spark_matrix_test_path_in_collector_attestation_command_argv():
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["collector_attestation"]["command_argv"] = [
        "pytest",
        "tests/scripts/test_voiceops_spark_matrix.py",
    ]

    live_evidence = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:collector_attestation_secret_or_phone_like_command_argv" not in live_evidence["issues"]


def test_live_evidence_rejects_fake_parent_manifest_attestation_hash(tmp_path):
    evidence = _complete_live_evidence()
    source_path = tmp_path / "discord-live-probe.json"
    evidence["discord_live_probe"]["source_artifact"] = str(source_path)
    _write_attested_section(source_path, evidence["discord_live_probe"], "discord_live_probe")
    evidence["discord_live_probe"]["collector_attestation"]["parent_manifest_sha256"] = "d" * 64

    live_evidence = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:collector_attestation_parent_manifest_sha256_mismatch" in live_evidence["issues"]


def test_live_evidence_rejects_source_artifact_secret_and_phone_leak(tmp_path):
    evidence = _complete_live_evidence()
    source_path = tmp_path / "discord-live-probe.json"
    evidence["discord_live_probe"]["source_artifact"] = str(source_path)
    source_path.write_text(
        json.dumps(
            {
                "kind": "discord_live_probe",
                "redacted": True,
                "api_key": "sk_live_123456789abcdef",
                "operator_note": "call +15551234567 after the test",
            }
        ),
        encoding="utf-8",
    )
    source_sha256 = hashlib.sha256(
        json.dumps(
            {
                "kind": "discord_live_probe",
                "redacted": True,
                "api_key": "sk_live_123456789abcdef",
                "operator_note": "call +15551234567 after the test",
            },
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    evidence["discord_live_probe"]["collector_attestation"]["redacted_artifact_sha256"] = source_sha256
    evidence["discord_live_probe"]["collector_attestation"]["parent_manifest_sha256"] = source_sha256

    live_evidence = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:source_artifact_forbidden_field:api_key" in live_evidence["issues"]
    assert "discord_live_probe:source_artifact_secret_or_phone_like_value:api_key" in live_evidence["issues"]
    assert "discord_live_probe:source_artifact_secret_or_phone_like_value:operator_note" in live_evidence["issues"]


def test_live_evidence_rejects_source_artifact_voice_capability_denial(tmp_path):
    evidence = _complete_live_evidence()
    source_path = tmp_path / "live-turn.json"
    evidence["live_turn"]["source_artifact"] = str(source_path)
    source_path.write_text(
        json.dumps(
            {
                "kind": "live_turn",
                "redacted": True,
                "assistant_text": "I cannot hear you in Discord voice.",
            }
        ),
        encoding="utf-8",
    )
    source_sha256 = hashlib.sha256(
        json.dumps(
            {
                "kind": "live_turn",
                "redacted": True,
                "assistant_text": "I cannot hear you in Discord voice.",
            },
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    evidence["live_turn"]["collector_attestation"]["redacted_artifact_sha256"] = source_sha256
    evidence["live_turn"]["collector_attestation"]["parent_manifest_sha256"] = source_sha256

    live_evidence = validate_live_probe_evidence(evidence)

    assert "live_turn:source_artifact_forbidden_field:assistant_text" in live_evidence["issues"]
    assert "live_turn:source_artifact_voice_capability_denial_text:assistant_text" in live_evidence["issues"]


def test_live_evidence_rejects_complete_hand_authored_sections_without_collector_attestation(tmp_path):
    evidence = _complete_live_evidence()
    for section_name in ("discord_live_probe", "sidecar_session", "live_turn"):
        evidence[section_name]["source_artifact"] = str(tmp_path / f"{section_name}.json")
        evidence[section_name].pop("collector_attestation")
        (tmp_path / f"{section_name}.json").write_text(json.dumps(evidence[section_name]), encoding="utf-8")

    live_evidence = validate_live_probe_evidence(evidence)

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "discord_live_probe:missing_collector_attestation" in live_evidence["issues"]
    assert "sidecar_session:missing_collector_attestation" in live_evidence["issues"]
    assert "live_turn:missing_collector_attestation" in live_evidence["issues"]


def test_voice_operator_rejects_loaded_evidence_with_missing_source_artifact_files(tmp_path):
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["source_artifact"] = "missing-discord-live-probe.json"
    evidence["sidecar_session"]["source_artifact"] = "missing-sidecar-session.json"
    evidence["live_turn"]["source_artifact"] = "missing-live-turn.json"
    evidence_path = tmp_path / "live-evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    live_evidence = _load_live_evidence([evidence_path])

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "discord_live_probe:source_artifact_not_found" in live_evidence["issues"]
    assert "sidecar_session:source_artifact_not_found" in live_evidence["issues"]
    assert "live_turn:source_artifact_not_found" in live_evidence["issues"]
    assert (
        f"discord_live_probe:source_artifact_not_found_candidates:{tmp_path / 'missing-discord-live-probe.json'}"
        in live_evidence["issues"]
    )
    assert (
        f"sidecar_session:source_artifact_not_found_candidates:{tmp_path / 'missing-sidecar-session.json'}"
        in live_evidence["issues"]
    )
    assert (
        f"live_turn:source_artifact_not_found_candidates:{tmp_path / 'missing-live-turn.json'}"
        in live_evidence["issues"]
    )
    report = _voice_operator_report(live_evidence=live_evidence)
    assert report["status"] == "needs_live_probe"
    assert report["proofs"]["live_evidence"]["ok"] is False
    assert report["live_probe_required_for_completion"]["missing_gates"] == [
        "discord_join",
        "discord_playback",
        "live_receiver",
        "production_sidecar",
        "live_turn",
    ]


def test_live_evidence_rejects_template_source_artifact_placeholders():
    live_evidence = validate_live_probe_evidence(_complete_live_evidence())

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "discord_live_probe:template_source_artifact_not_accepted" in live_evidence["issues"]
    assert "sidecar_session:template_source_artifact_not_accepted" in live_evidence["issues"]
    assert "live_turn:template_source_artifact_not_accepted" in live_evidence["issues"]


def test_live_turn_accepts_interpreter_evidence_without_transcript_observed_flag():
    evidence = _complete_live_evidence()
    evidence["live_turn"].pop("transcript_observed", None)

    live_evidence = validate_live_probe_evidence(evidence)

    assert "live_turn:transcript_observed_not_true" not in live_evidence["issues"]
    assert live_evidence["live_turn"]["ok"] is True


def test_voice_operator_loaded_evidence_does_not_resolve_source_artifacts_from_cwd(monkeypatch, tmp_path):
    evidence_dir = tmp_path / "evidence-dir"
    cwd_dir = tmp_path / "cwd"
    evidence_dir.mkdir()
    cwd_dir.mkdir()
    for name in ("cwd-discord-live-probe.json", "cwd-sidecar-session.json", "cwd-live-turn.json"):
        (cwd_dir / name).write_text("{}", encoding="utf-8")
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["source_artifact"] = "cwd-discord-live-probe.json"
    evidence["sidecar_session"]["source_artifact"] = "cwd-sidecar-session.json"
    evidence["live_turn"]["source_artifact"] = "cwd-live-turn.json"
    evidence_path = evidence_dir / "live-evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    monkeypatch.chdir(cwd_dir)

    live_evidence = _load_live_evidence([evidence_path])

    assert "discord_live_probe:source_artifact_not_found" in live_evidence["issues"]
    assert "sidecar_session:source_artifact_not_found" in live_evidence["issues"]
    assert "live_turn:source_artifact_not_found" in live_evidence["issues"]
    assert (
        f"discord_live_probe:source_artifact_not_found_candidates:{evidence_dir / 'cwd-discord-live-probe.json'}"
        in live_evidence["issues"]
    )


def test_voice_operator_manifest_reports_do_not_resolve_from_cwd(monkeypatch, tmp_path):
    evidence_dir = tmp_path / "evidence-dir"
    cwd_dir = tmp_path / "cwd"
    evidence_dir.mkdir()
    cwd_dir.mkdir()
    for name in ("cwd-discord-live-probe.json", "cwd-sidecar-session.json", "cwd-live-turn.json"):
        (cwd_dir / name).write_text(json.dumps({"kind": name.removesuffix(".json")}), encoding="utf-8")
    manifest_path = evidence_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "reports": {
                    "discord_live_probe": "cwd-discord-live-probe.json",
                    "sidecar_session": "cwd-sidecar-session.json",
                    "live_turn": "cwd-live-turn.json",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(cwd_dir)

    live_evidence = _load_live_evidence([manifest_path])

    assert "live_evidence_manifest:discord_live_probe:live_evidence_file_not_found" in live_evidence["issues"]
    assert "live_evidence_manifest:sidecar_session:live_evidence_file_not_found" in live_evidence["issues"]
    assert "live_evidence_manifest:live_turn:live_evidence_file_not_found" in live_evidence["issues"]


def test_voice_operator_manifest_reports_do_not_fallback_to_basename(tmp_path):
    discord_probe = _complete_live_evidence()["discord_live_probe"]
    discord_probe["kind"] = "discord_live_probe"
    discord_probe["source_artifact"] = str(tmp_path / "discord-live-probe.json")
    (tmp_path / "discord-live-probe.json").write_text(json.dumps(discord_probe), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "reports": {
                    "discord_live_probe": "sections/discord-live-probe.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert "live_evidence_manifest:discord_live_probe:live_evidence_file_not_found" in live_evidence["issues"]


def test_voice_operator_manifest_rejects_absolute_report_path(tmp_path):
    discord_probe = _complete_live_evidence()["discord_live_probe"]
    discord_probe["kind"] = "discord_live_probe"
    report_path = tmp_path / "discord-live-probe.json"
    report_path.write_text(json.dumps(discord_probe), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "reports": {
                    "discord_live_probe": str(report_path),
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert "live_evidence_manifest:discord_live_probe:report_path:absolute_path_not_allowed" in live_evidence["issues"]


def test_voice_operator_manifest_rejects_parent_escape_report_path(tmp_path):
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    report_path = tmp_path / "discord-live-probe.json"
    discord_probe = _complete_live_evidence()["discord_live_probe"]
    discord_probe["kind"] = "discord_live_probe"
    report_path.write_text(json.dumps(discord_probe), encoding="utf-8")
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "reports": {
                    "discord_live_probe": "../discord-live-probe.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert (
        "live_evidence_manifest:discord_live_probe:report_path:parent_traversal_not_allowed"
        in live_evidence["issues"]
    )


def test_voice_operator_manifest_rejects_user_home_report_path(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "reports": {
                    "discord_live_probe": "~/discord-live-probe.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert "live_evidence_manifest:discord_live_probe:report_path:user_home_not_allowed" in live_evidence["issues"]


def test_voice_operator_manifest_rejects_symlink_escape_report_path(tmp_path):
    manifest_dir = tmp_path / "manifest"
    sections_dir = manifest_dir / "sections"
    sections_dir.mkdir(parents=True)
    outside_report = tmp_path / "discord-live-probe.json"
    discord_probe = _complete_live_evidence()["discord_live_probe"]
    discord_probe["kind"] = "discord_live_probe"
    outside_report.write_text(json.dumps(discord_probe), encoding="utf-8")
    (sections_dir / "discord-live-probe.json").symlink_to(outside_report)
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "reports": {
                    "discord_live_probe": "sections/discord-live-probe.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert "live_evidence_manifest:discord_live_probe:report_path:path_escape_not_allowed" in live_evidence["issues"]


def test_voice_operator_manifest_cycle_returns_validation_issue(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "reports": {
                    "combined": "manifest.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "live_evidence_manifest:combined:cycle_detected" in live_evidence["issues"]


def test_voice_operator_rejects_source_artifact_directory(tmp_path):
    evidence = _complete_live_evidence()
    for section_name in ("discord_live_probe", "sidecar_session", "live_turn"):
        artifact_dir = tmp_path / f"{section_name}.json"
        artifact_dir.mkdir()
        evidence[section_name]["source_artifact"] = str(artifact_dir)

    live_evidence = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:source_artifact_not_file" in live_evidence["issues"]
    assert "sidecar_session:source_artifact_not_file" in live_evidence["issues"]
    assert "live_turn:source_artifact_not_file" in live_evidence["issues"]


def test_voice_operator_ingests_realtime_live_evidence_manifest(tmp_path):
    discord_probe = {
        "kind": "discord_live_probe",
        "collector_attestation": _collector_attestation("discord_live_probe"),
        "ok": True,
        "connect_perm": True,
        "speak_perm": True,
        "connected": True,
        "opus_loaded": True,
        "accepted_audio_source": True,
        "played": True,
        "playing_during_probe": True,
        "receiver_started": True,
        "receiver_frames": 18,
        "receiver_speech_start": 1,
        "inbound_observed": True,
        "disconnected": True,
        "require_inbound": True,
        "latency_metrics_ms": _complete_discord_latency_metrics(),
    }
    sidecar = {
        "kind": "sidecar_session",
        "collector_attestation": _collector_attestation("sidecar_session"),
        **_complete_sidecar_session_fields(),
    }
    live_turn = {
        "kind": "live_turn",
        "collector_attestation": _collector_attestation("live_turn"),
        **_complete_live_turn_fields(),
    }
    _write_attested_section(tmp_path / "discord-live-probe.json", discord_probe, "discord_live_probe")
    _write_attested_section(tmp_path / "sidecar-session.json", sidecar, "sidecar_session")
    _write_attested_section(tmp_path / "live-turn.json", live_turn, "live_turn")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "ok": True,
                "reports": {
                    "discord_live_probe": "discord-live-probe.json",
                    "sidecar_session": "sidecar-session.json",
                    "live_turn": "live-turn.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])
    report = _voice_operator_report(live_evidence=live_evidence)

    assert live_evidence["overall_status"] == "live_evidence_supplied_not_readiness_claim"
    assert live_evidence["issues"] == []
    assert live_evidence["discord_live_probe"]["join_ok"] is True
    assert live_evidence["section_refs"] == {
        "discord_live_probe": {
            "source_artifact": str(tmp_path / "discord-live-probe.json"),
            "section": "discord_live_probe",
            "wrapper_artifact": str(tmp_path / "discord-live-probe.json"),
        },
        "sidecar_session": {
            "source_artifact": str(tmp_path / "sidecar-session.json"),
            "section": "sidecar_session",
            "wrapper_artifact": str(tmp_path / "sidecar-session.json"),
        },
        "live_turn": {
            "source_artifact": str(tmp_path / "live-turn.json"),
            "section": "live_turn",
            "wrapper_artifact": str(tmp_path / "live-turn.json"),
        },
    }
    assert report["live_probe_required_for_completion"]["missing_gates"] == []
    assert report["status"] == "live_evidence_supplied_not_readiness_claim"


def test_voice_operator_ingests_repeated_standalone_live_evidence_files(tmp_path):
    discord_path = tmp_path / "actual-discord-probe.json"
    sidecar_path = tmp_path / "actual-sidecar-session.json"
    turn_path = tmp_path / "actual-live-turn.json"
    _write_attested_section(
        discord_path,
        {
            "schema_version": "voiceops.milestone1.live_voice_evidence.v1",
            "kind": "discord_live_probe",
            "source_artifact": discord_path.name,
            "ok": True,
            "connect_perm": True,
            "speak_perm": True,
            "connected": True,
            "opus_loaded": True,
            "accepted_audio_source": True,
            "played": True,
            "playing_during_probe": True,
            "receiver_started": True,
            "receiver_frames": 18,
            "receiver_speech_start": 1,
            "inbound_observed": True,
            "disconnected": True,
            "require_inbound": True,
            "latency_metrics_ms": _complete_discord_latency_metrics(),
        },
        "discord_live_probe",
    )
    _write_attested_section(
        sidecar_path,
        {
            "schema_version": "voiceops.milestone1.live_voice_evidence.v1",
            "kind": "sidecar_session",
            "source_artifact": sidecar_path.name,
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
            "latency_metrics_ms": {"session_start_ms": 110, "shutdown_ms": 80},
        },
        "sidecar_session",
    )
    _write_attested_section(
        turn_path,
        {
            "schema_version": "voiceops.milestone1.live_voice_evidence.v1",
            "kind": "live_turn",
            "source_artifact": turn_path.name,
            **_complete_live_turn_fields(),
        },
        "live_turn",
    )

    live_evidence = _load_live_evidence([discord_path, sidecar_path, turn_path])

    assert live_evidence["overall_status"] == "live_evidence_supplied_not_readiness_claim"
    assert live_evidence["issues"] == []
    assert live_evidence["section_refs"] == {
        "discord_live_probe": {
            "source_artifact": discord_path.name,
            "section": "discord_live_probe",
        },
        "sidecar_session": {
            "source_artifact": sidecar_path.name,
            "section": "sidecar_session",
        },
        "live_turn": {
            "source_artifact": turn_path.name,
            "section": "live_turn",
        },
    }


def test_voice_operator_rejects_standalone_section_without_identity(tmp_path):
    sidecar_path = tmp_path / "actual-sidecar-session.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.milestone1.live_voice_evidence.v1",
                "source_artifact": sidecar_path.name,
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
                "latency_metrics_ms": {"session_start_ms": 110, "shutdown_ms": 80},
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([sidecar_path])

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "missing_standalone_report_identity" in live_evidence["issues"]


def test_voice_operator_accepts_standalone_section_evidence_type_identity(tmp_path):
    turn_path = tmp_path / "actual-live-turn.json"
    turn_path.write_text(
        json.dumps(
            {
                "evidence_type": "live_turn",
                "source_artifact": turn_path.name,
                "turn_id": "voiceops-live-turn-budget",
                "audio_segment_ref": "artifact://redacted/voiceops-live-turn-budget.wav",
                "evidence_bundle_id": "kame-evidence-live-turn-budget",
                "evidence_merge_key": "kame-merge-live-turn-budget",
                "transcript_observed": True,
                "audio_segment_ref_observed": True,
                "interpreter_evidence_observed": True,
                "transcript_hypotheses_labeled": True,
                "assistant_audio_observed": True,
                "barge_in_observed": True,
                "spoken_reply_short": True,
                "no_voice_denial_observed": True,
                "speech_end_to_first_audio_ms": 950,
                "barge_in_stop_ms": 80,
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([turn_path])

    assert "missing_standalone_report_identity" not in live_evidence["issues"]
    assert "invalid_standalone_report_identity" not in " ".join(live_evidence["issues"])
    assert "live_turn:missing_source_artifact" not in live_evidence["issues"]


def test_voice_operator_rejects_standalone_section_with_wrong_identity(tmp_path):
    turn_path = tmp_path / "actual-live-turn.json"
    turn_path.write_text(
        json.dumps(
            {
                "kind": "calendar_event",
                "source_artifact": turn_path.name,
                "transcript_observed": True,
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([turn_path])

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "invalid_standalone_report_identity:calendar_event" in live_evidence["issues"]


def test_voice_operator_rejects_combined_manifest_placeholder_source_artifacts(tmp_path):
    evidence = _complete_live_evidence()
    (tmp_path / "all-live-evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "ok": True,
                "reports": {
                    "combined": "all-live-evidence.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "discord_live_probe:template_source_artifact_not_accepted" in live_evidence["issues"]
    assert "sidecar_session:template_source_artifact_not_accepted" in live_evidence["issues"]
    assert "live_turn:template_source_artifact_not_accepted" in live_evidence["issues"]


def test_voice_operator_rejects_template_source_artifact_path_variants():
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["source_artifact"] = "./discord-live-probe.json"
    evidence["sidecar_session"]["source_artifact"] = "./sidecar-session.json"
    evidence["live_turn"]["source_artifact"] = "./live-turn.json"

    live_evidence = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:template_source_artifact_not_accepted" in live_evidence["issues"]
    assert "sidecar_session:template_source_artifact_not_accepted" in live_evidence["issues"]
    assert "live_turn:template_source_artifact_not_accepted" in live_evidence["issues"]


def test_voice_operator_rejects_anonymous_combined_manifest_report(tmp_path):
    evidence = _complete_live_evidence()
    evidence.pop("schema_version")
    evidence["discord_live_probe"]["source_artifact"] = str(tmp_path / "all-live-evidence.json")
    evidence["sidecar_session"]["source_artifact"] = str(tmp_path / "all-live-evidence.json")
    evidence["live_turn"]["source_artifact"] = str(tmp_path / "all-live-evidence.json")
    (tmp_path / "all-live-evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "reports": {"combined": "all-live-evidence.json"},
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "live_evidence_manifest:combined:missing_report_identity" in live_evidence["issues"]


def test_voice_operator_rejects_combined_manifest_missing_nested_source_artifact(tmp_path):
    evidence = _complete_live_evidence()
    evidence["live_turn"].pop("source_artifact")
    evidence["discord_live_probe"]["source_artifact"] = str(tmp_path / "all-live-evidence.json")
    evidence["sidecar_session"]["source_artifact"] = str(tmp_path / "all-live-evidence.json")
    (tmp_path / "all-live-evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "ok": True,
                "reports": {
                    "combined": "all-live-evidence.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "live_turn:missing_source_artifact" in live_evidence["issues"]


def test_voice_operator_manifest_nested_report_source_artifacts_resolve_relative_to_report(tmp_path):
    report_dir = tmp_path / "reports"
    raw_dir = report_dir / "raw"
    raw_dir.mkdir(parents=True)
    for name in ("discord.json", "sidecar.json", "turn.json"):
        (raw_dir / name).write_text("{}", encoding="utf-8")
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["source_artifact"] = "raw/discord.json"
    evidence["sidecar_session"]["source_artifact"] = "raw/sidecar.json"
    evidence["live_turn"]["source_artifact"] = "raw/turn.json"
    raw_payload_sha256 = hashlib.sha256(b"{}").hexdigest()
    for section_name in ("discord_live_probe", "sidecar_session", "live_turn"):
        evidence[section_name]["collector_attestation"]["raw_artifact_sha256"] = raw_payload_sha256
        evidence[section_name]["collector_attestation"]["redacted_artifact_sha256"] = raw_payload_sha256
        evidence[section_name]["collector_attestation"]["parent_manifest_sha256"] = raw_payload_sha256
    combined_path = report_dir / "combined.json"
    combined_path.write_text(json.dumps(evidence), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "ok": True,
                "reports": {
                    "combined": "reports/combined.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert live_evidence["overall_status"] == "live_evidence_supplied_not_readiness_claim"
    assert live_evidence["issues"] == []
    assert live_evidence["section_refs"]["sidecar_session"]["source_artifact"] == str(raw_dir / "sidecar.json")
    assert live_evidence["section_refs"]["sidecar_session"]["wrapper_artifact"] == str(report_dir / "combined.json")
    assert live_evidence["section_refs"]["sidecar_session"]["reported_source_artifact"] == "raw/sidecar.json"


def test_voice_operator_manifest_rejects_nested_source_artifact_parent_escape(tmp_path):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    evidence = _complete_live_evidence()
    evidence["sidecar_session"]["source_artifact"] = "../sidecar-raw.json"
    combined_path = report_dir / "combined.json"
    combined_path.write_text(json.dumps(evidence), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "ok": True,
                "reports": {
                    "combined": "reports/combined.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert (
        "live_evidence_manifest:combined:sidecar_session.source_artifact:parent_traversal_not_allowed"
        in live_evidence["issues"]
    )


def test_voice_operator_manifest_rejects_nested_source_artifact_absolute_and_home_refs(tmp_path):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["source_artifact"] = str(report_dir / "raw" / "discord.json")
    evidence["sidecar_session"]["source_artifact"] = "~/sidecar-raw.json"
    combined_path = report_dir / "combined.json"
    combined_path.write_text(json.dumps(evidence), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "ok": True,
                "reports": {
                    "combined": "reports/combined.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert (
        "live_evidence_manifest:combined:discord_live_probe.source_artifact:absolute_path_not_allowed"
        in live_evidence["issues"]
    )
    assert (
        "live_evidence_manifest:combined:sidecar_session.source_artifact:user_home_not_allowed"
        in live_evidence["issues"]
    )


def test_voice_operator_manifest_rejects_nested_source_artifact_symlink_escape(tmp_path):
    report_dir = tmp_path / "reports"
    raw_dir = report_dir / "raw"
    raw_dir.mkdir(parents=True)
    outside_source = tmp_path / "turn-raw.json"
    outside_source.write_text("{}", encoding="utf-8")
    (raw_dir / "turn.json").symlink_to(outside_source)
    evidence = _complete_live_evidence()
    evidence["live_turn"]["source_artifact"] = "raw/turn.json"
    combined_path = report_dir / "combined.json"
    combined_path.write_text(json.dumps(evidence), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "ok": True,
                "reports": {
                    "combined": "reports/combined.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert (
        "live_evidence_manifest:combined:live_turn.source_artifact:path_escape_not_allowed"
        in live_evidence["issues"]
    )


def test_live_evidence_rejects_complete_payload_without_schema_and_source_artifacts():
    evidence = build_live_probe_evidence_template()
    evidence.pop("schema_version")
    evidence["discord_live_probe"].pop("source_artifact")
    evidence["sidecar_session"].pop("source_artifact")
    evidence["live_turn"].pop("source_artifact")
    evidence["discord_live_probe"].update(
        {
            "collector_attestation": _collector_attestation("discord_live_probe"),
            "ok": True,
            "connect_perm": True,
            "speak_perm": True,
            "connected": True,
            "opus_loaded": True,
            "accepted_audio_source": True,
            "played": True,
            "playing_during_probe": True,
            "receiver_started": True,
            "receiver_frames": 12,
            "receiver_speech_start": 1,
            "inbound_observed": True,
            "disconnected": True,
            "require_inbound": True,
            "latency_metrics_ms": _complete_discord_latency_metrics(),
        }
    )
    evidence["sidecar_session"].update(
        {
            "collector_attestation": _collector_attestation("sidecar_session"),
            **_complete_sidecar_session_fields(),
        }
    )
    evidence["live_turn"].update(
        {
            "collector_attestation": _collector_attestation("live_turn"),
            **_complete_live_turn_fields(speech_end_to_first_audio_ms=900, barge_in_stop_ms=90),
        }
    )

    result = validate_live_probe_evidence(evidence)

    assert result["overall_status"] == "partial_live_evidence"
    assert result["issues"] == [
        "discord_live_probe:missing_source_artifact",
        "live_turn:missing_source_artifact",
        "missing_schema_version",
        "sidecar_session:missing_source_artifact",
    ]


def test_voice_operator_rejects_manifest_with_example_only_referenced_section(tmp_path):
    discord_probe = build_live_probe_evidence_example()["discord_live_probe"]
    discord_probe["example_only"] = True
    sidecar = build_live_probe_evidence_template()["sidecar_session"]
    sidecar.update(
        {
            **_complete_sidecar_session_fields(),
        }
    )
    live_turn = build_live_probe_evidence_template()["live_turn"]
    live_turn.update(
        {
            "turn_id": "voiceops-live-turn-budget",
            "audio_segment_ref": "artifact://redacted/voiceops-live-turn-budget.wav",
            "evidence_bundle_id": "kame-evidence-live-turn-budget",
            "evidence_merge_key": "kame-merge-live-turn-budget",
            "transcript_observed": True,
            "audio_segment_ref_observed": True,
            "interpreter_evidence_observed": True,
            "transcript_hypotheses_labeled": True,
            "assistant_audio_observed": True,
            "barge_in_observed": True,
            "spoken_reply_short": True,
            "no_voice_denial_observed": True,
            "speech_end_to_first_audio_ms": 950,
            "barge_in_stop_ms": 80,
        }
    )
    (tmp_path / "discord-live-probe.json").write_text(json.dumps(discord_probe), encoding="utf-8")
    (tmp_path / "sidecar-session.json").write_text(json.dumps(sidecar), encoding="utf-8")
    (tmp_path / "live-turn.json").write_text(json.dumps(live_turn), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                "ok": True,
                "reports": {
                    "discord_live_probe": "discord-live-probe.json",
                    "sidecar_session": "sidecar-session.json",
                    "live_turn": "live-turn.json",
                },
            }
        ),
        encoding="utf-8",
    )

    live_evidence = _load_live_evidence([manifest_path])

    assert live_evidence["overall_status"] == "partial_live_evidence"
    assert "example_only_evidence_not_accepted" in live_evidence["issues"]
    assert "live_evidence_manifest:discord_live_probe:example_only_evidence_not_accepted" in live_evidence["issues"]


def test_voice_operator_rejects_manifest_with_missing_or_invalid_schema(tmp_path):
    discord_probe = build_live_probe_evidence_example()["discord_live_probe"]
    discord_probe.pop("example_only", None)
    sidecar = build_live_probe_evidence_template()["sidecar_session"]
    sidecar.update(
        {
            **_complete_sidecar_session_fields(),
        }
    )
    live_turn = build_live_probe_evidence_template()["live_turn"]
    live_turn.update(
        {
            "turn_id": "voiceops-live-turn-budget",
            "audio_segment_ref": "artifact://redacted/voiceops-live-turn-budget.wav",
            "evidence_bundle_id": "kame-evidence-live-turn-budget",
            "evidence_merge_key": "kame-merge-live-turn-budget",
            "transcript_observed": True,
            "audio_segment_ref_observed": True,
            "interpreter_evidence_observed": True,
            "transcript_hypotheses_labeled": True,
            "assistant_audio_observed": True,
            "barge_in_observed": True,
            "spoken_reply_short": True,
            "no_voice_denial_observed": True,
            "speech_end_to_first_audio_ms": 950,
            "barge_in_stop_ms": 80,
        }
    )
    (tmp_path / "discord-live-probe.json").write_text(json.dumps(discord_probe), encoding="utf-8")
    (tmp_path / "sidecar-session.json").write_text(json.dumps(sidecar), encoding="utf-8")
    (tmp_path / "live-turn.json").write_text(json.dumps(live_turn), encoding="utf-8")

    base_manifest = {
        "ok": True,
        "reports": {
            "discord_live_probe": "discord-live-probe.json",
            "sidecar_session": "sidecar-session.json",
            "live_turn": "live-turn.json",
        },
    }
    missing_schema_path = tmp_path / "missing-schema-manifest.json"
    missing_schema_path.write_text(json.dumps(base_manifest), encoding="utf-8")
    missing_schema = _load_live_evidence([missing_schema_path])
    assert missing_schema["overall_status"] == "partial_live_evidence"
    assert "live_evidence_manifest:missing_schema_version" in missing_schema["issues"]

    invalid_schema_path = tmp_path / "invalid-schema-manifest.json"
    invalid_schema_path.write_text(json.dumps({**base_manifest, "schema_version": "wrong.schema.v1"}), encoding="utf-8")
    invalid_schema = _load_live_evidence([invalid_schema_path])
    assert invalid_schema["overall_status"] == "partial_live_evidence"
    assert "live_evidence_manifest:invalid_schema_version" in invalid_schema["issues"]


def test_live_evidence_rejects_nested_example_only_sections():
    evidence = build_live_probe_evidence_template()
    evidence["discord_live_probe"].update(
        {
            "example_only": True,
            "ok": True,
            "connect_perm": True,
            "speak_perm": True,
            "connected": True,
            "opus_loaded": True,
            "accepted_audio_source": True,
            "played": True,
            "playing_during_probe": True,
            "receiver_started": True,
            "receiver_frames": 12,
            "receiver_speech_start": 1,
            "inbound_observed": True,
            "disconnected": True,
            "require_inbound": True,
        }
    )
    evidence["sidecar_session"].update(
        {
            "example_only": True,
            **_complete_sidecar_session_fields(),
        }
    )
    evidence["live_turn"].update(
        {
            "example_only": True,
            "turn_id": "voiceops-live-turn-budget",
            "audio_segment_ref": "artifact://redacted/voiceops-live-turn-budget.wav",
            "evidence_bundle_id": "kame-evidence-live-turn-budget",
            "evidence_merge_key": "kame-merge-live-turn-budget",
            "transcript_observed": True,
            "audio_segment_ref_observed": True,
            "interpreter_evidence_observed": True,
            "transcript_hypotheses_labeled": True,
            "assistant_audio_observed": True,
            "barge_in_observed": True,
            "spoken_reply_short": True,
            "no_voice_denial_observed": True,
            "speech_end_to_first_audio_ms": 950,
            "barge_in_stop_ms": 80,
        }
    )

    result = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:example_only_evidence_not_accepted" in result["issues"]
    assert "sidecar_session:example_only_evidence_not_accepted" in result["issues"]
    assert "live_turn:example_only_evidence_not_accepted" in result["issues"]


def test_live_evidence_rejects_raw_text_secret_and_denial_fields():
    evidence = _complete_live_evidence()
    evidence["live_turn"]["assistant_text"] = "I cannot hear voice. I only process typed text."
    evidence["live_turn"]["raw_transcript"] = "call me at +15551234567"
    evidence["sidecar_session"]["api_key"] = "sk-car-exampletoken123456"

    result = validate_live_probe_evidence(evidence)

    assert "live_turn.assistant_text:forbidden_evidence_field" in result["issues"]
    assert "live_turn.assistant_text:voice_capability_denial_text" in result["issues"]
    assert "live_turn.raw_transcript:forbidden_evidence_field" in result["issues"]
    assert "live_turn.raw_transcript:secret_or_phone_like_value" in result["issues"]
    assert "sidecar_session.api_key:forbidden_evidence_field" in result["issues"]
    assert "sidecar_session.api_key:secret_or_phone_like_value" in result["issues"]


def test_live_evidence_requires_bounded_sidecar_shutdown():
    evidence = _complete_live_evidence()
    evidence["sidecar_session"].pop("shutdown_bounded")
    evidence["sidecar_session"]["shutdown_timed_out"] = True
    evidence["sidecar_session"]["latency_metrics_ms"] = {}

    result = validate_live_probe_evidence(evidence)

    assert "sidecar_session:missing_shutdown_ms" in result["issues"]
    assert "sidecar_session:shutdown_bounded_not_true" in result["issues"]
    assert "sidecar_session:shutdown_timed_out_not_false" in result["issues"]
    assert result["sidecar_session"]["ok"] is False


def test_live_evidence_requires_sidecar_production_provenance_and_fallback_reason():
    evidence = _complete_live_evidence()
    evidence["sidecar_session"].pop("sidecar_mode")
    evidence["sidecar_session"]["healthcheck_observed"] = False
    evidence["sidecar_session"]["provider_transport_observed"] = False
    evidence["sidecar_session"]["session_id_redacted"] = False
    evidence["sidecar_session"]["fallback_reason"] = ""

    result = validate_live_probe_evidence(evidence)

    assert "sidecar_session:sidecar_mode_not_production" in result["issues"]
    assert "sidecar_session:healthcheck_observed_not_true" in result["issues"]
    assert "sidecar_session:provider_transport_observed_not_true" in result["issues"]
    assert "sidecar_session:session_id_redacted_not_true" in result["issues"]
    assert "sidecar_session:missing_fallback_reason" in result["issues"]
    assert result["sidecar_session"]["ok"] is False


def test_live_evidence_rejects_fatal_kame_reflex_fallback_reason():
    evidence = _complete_live_evidence()
    evidence["sidecar_session"]["fallback_reason"] = (
        "transcription failed: KAME audio reflex failed and ASR reflex fallback is disabled: timed out"
    )

    result = validate_live_probe_evidence(evidence)

    assert result["overall_status"] == "partial_live_evidence"
    assert "sidecar_session:fatal_kame_reflex_fallback" in result["issues"]
    assert result["sidecar_session"]["ok"] is False


def test_live_evidence_requires_discord_and_sidecar_session_start_latencies():
    evidence = _complete_live_evidence()
    evidence["discord_live_probe"]["latency_metrics_ms"].pop("connect_ms")
    evidence["discord_live_probe"]["latency_metrics_ms"]["disconnect_ms"] = -1
    evidence["sidecar_session"]["latency_metrics_ms"].pop("session_start_ms")

    result = validate_live_probe_evidence(evidence)

    assert "discord_live_probe:missing_connect_ms" in result["issues"]
    assert "discord_live_probe:missing_disconnect_ms" in result["issues"]
    assert "sidecar_session:missing_session_start_ms" in result["issues"]
    assert result["discord_live_probe"]["latency_ok"] is False
    assert result["sidecar_session"]["ok"] is False


def test_live_turn_latency_boundaries_are_exact():
    evidence = _complete_live_evidence()
    evidence["live_turn"]["speech_end_to_first_audio_ms"] = 3000
    evidence["live_turn"]["barge_in_stop_ms"] = 150

    boundary = validate_live_probe_evidence(evidence)

    assert "live_turn:speech_end_to_first_audio_ms_over_target" not in boundary["issues"]
    assert "live_turn:barge_in_stop_ms_over_target" not in boundary["issues"]

    evidence["live_turn"]["speech_end_to_first_audio_ms"] = 3000.1
    evidence["live_turn"]["barge_in_stop_ms"] = 150.1
    over = validate_live_probe_evidence(evidence)

    assert "live_turn:speech_end_to_first_audio_ms_over_target" in over["issues"]
    assert "live_turn:barge_in_stop_ms_over_target" in over["issues"]

    evidence["live_turn"]["speech_end_to_first_audio_ms"] = -1
    evidence["live_turn"]["barge_in_stop_ms"] = "not-a-number"
    invalid = validate_live_probe_evidence(evidence)

    assert "live_turn:missing_speech_end_to_first_audio_ms" in invalid["issues"]
    assert "live_turn:missing_barge_in_stop_ms" in invalid["issues"]


def test_voice_operator_cli_smoke(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "voiceops_voice_operator.py"
    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["validation_issues"] == []
    assert Path(payload["artifacts"]["json"]).exists()
    assert Path(payload["artifacts"]["markdown"]).exists()
    assert Path(payload["artifacts"]["smoke_json"]).exists()
    assert Path(payload["artifacts"]["async_oracle_smoke_json"]).exists()
    assert Path(payload["artifacts"]["events_jsonl"]).exists()


def test_parse_args_defaults_to_requested_artifact_dir():
    args = parse_args([])

    assert args.output_dir == DEFAULT_OUTPUT_DIR
    assert args.live_evidence == []
