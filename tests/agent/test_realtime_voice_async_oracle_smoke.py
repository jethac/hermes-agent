import pytest

from scripts.realtime_voice_async_oracle_smoke import run_smoke


@pytest.mark.asyncio
async def test_async_oracle_smoke_proves_concurrency_local_turn_and_cancellation():
    report = await run_smoke()

    assert report["ok"] is True
    assert report["max_running"] == 4
    assert report["max_worker_overlap"] == 4
    assert report["worker_overlap_proved"] is True
    assert report["worker_overlap_within_capacity"] is True
    assert report["noncooperative_cancel_overlap_observed"] is False
    assert report["started_jobs"] == 9
    assert report["queued_jobs"] == 1
    assert report["completed_jobs"] == 5
    assert report["failed_jobs"] == 2
    assert report["cancelled_jobs"] == 2
    assert report["queued_cancel_smoke_ok"] is True
    assert report["queued_cancel_requested_observed"] is True
    assert report["queued_cancel_observed"] is True
    assert report["queued_cancel_spoken_control_observed"] is True
    assert report["queued_cancelled_before_start"] is True
    assert report["queued_cancel_not_sent_to_oracle"] is True
    assert report["queued_cancel_reason"] == "spoken request to cancel oracle job"
    assert report["queued_cancel_target_job_id"] == "voice-oracle-002"
    assert report["queued_cancel_running_completed"] is True
    assert report["shutdown_timeout_configured_ms"] == 10
    assert report["shutdown_bounded_close_observed"] is True
    assert report["shutdown_forced_cancel_observed"] is True
    assert report["shutdown_close_cancel_entered"] is True
    assert report["shutdown_cancelled_jobs"] == 1
    assert report["local_turn_committed"] is True
    assert report["status_turn_committed"] is True
    assert report["status_text"].startswith("Oracle jobs: 4 running out of 4, 1 queued.")
    assert report["terminal_status_committed"] is True
    assert report["completed_result_status_visible"] is True
    assert "completed: First sentence. Second sentence. Third sentence." in report["terminal_status_text"]
    assert report["fifth_job_queued"] is True
    assert report["fifth_job_started_after_capacity_freed"] is True
    assert report["late_cancelled_output_attempted"] is True
    assert report["cancelled_result_spoken"] is False
    assert report["cancelled_result_committed"] is False
    assert report["cancelled_result_progress_leaked"] is False
    assert report["cancelled_result_durable_completed"] is False
    assert report["cancelled_result_durable_text"] is False
    assert report["durable_cancelled_record_present"] is True
    assert report["durable_completed_jobs"] == report["completed_jobs"]
    assert report["approval_wait_observed"] is True
    assert report["approval_status_committed"] is True
    assert report["approval_tool_progress_observed"] is True
    assert report["approval_payload_redacted"] is True
    assert report["approval_secret_leaked"] is False
    assert report["approval_secret_canary_checked"] is True
    assert report["approval_completed"] is False
    assert report["approval_gate_failed_closed"] is True
    assert report["approval_result_suppressed"] is True
    assert "1 waiting for approval" in report["approval_status_text"]
    assert "waiting_for_approval: Preparing spend approval." in report["approval_status_text"]
    assert report["cancel_drain_capacity_smoke_ok"] is True
    assert report["cancel_drain_requested_observed"] is True
    assert report["cancel_drain_cancelled_observed"] is True
    assert report["cancel_drain_followup_queued"] is True
    assert report["cancel_drain_active_visible"] is True
    assert report["cancel_drain_misleading_running_capacity"] is False
    assert "1 active out of 1" in report["cancel_drain_status_text"]
    assert "0 running out of 1" not in report["cancel_drain_status_text"]
    assert "1 queued" in report["cancel_drain_status_text"]
    assert "1 cancelling" in report["cancel_drain_status_text"]
    assert report["cancel_drain_followup_started_after_cancel"] is True
    assert report["cancel_drain_max_concurrent"] == 1
    assert report["failed_job_reported"] is True
    assert report["failed_job_spoken"] is True
    assert report["durable_failed_record_present"] is True
    assert report["session_survived_failed_job"] is True
    assert report["queued_job_update_observed"] is True
    assert report["spoken_priority_control_observed"] is True
    assert report["spoken_update_control_observed"] is True
    assert report["spoken_cancel_control_observed"] is True
    assert report["running_job_update_observed"] is True
    assert report["running_update_latest_update_visible"] is True
    assert report["running_update_latest_update_text"] == "include running update context"
    assert report["running_update_reached_oracle"] is True
    assert report["running_update_delivery_metadata_ok"] is True
    assert report["queued_update_latest_update_visible"] is True
    assert report["queued_update_latest_update_text"] == "include smoke update context"
    assert report["queued_update_started_with_priority"] is True
    assert report["queued_update_reached_oracle"] is True
    assert report["verbose_result_spoken_bounded"] is True
    assert report["verbose_result_committed_bounded"] is True
    assert report["verbose_result_commit_marked_truncated"] is True
    assert report["verbose_full_result_durable"] is True
    assert report["verbose_spoken_result"] == "First sentence."
    assert report["audit_scalar_smoke_ok"] is True
    assert report["audit_scalar_payload_redacted"] is True
    assert report["audit_scalar_secret_canary_checked"] is True
    assert report["audit_scalar_result_text_omitted"] is True
    assert report["audit_scalar_completed_event_seen"] is True
    assert report["audit_scalar_waiting_event_seen"] is True
    assert report["audit_scalar_row_count"] >= 3
    assert report["terminal_result_policy_smoke_ok"] is True
    assert report["terminal_result_auto_summarize_default"] is True
    assert report["terminal_result_default_event_count"] >= 1
    assert report["terminal_result_default_spoken"] is True
    assert report["terminal_result_suppression_config"] == "oracle_jobs.speak_terminal_results=false"
    assert report["terminal_result_suppressed"] is True
    assert report["terminal_result_suppressed_event_observed"] is True
    assert report["terminal_result_suppressed_event_count"] >= 1
    assert report["terminal_result_suppressed_reason"] == "terminal_speech_disabled"
    assert report["terminal_result_suppressed_payload_clean"] is True
    assert report["terminal_result_unsolicited_event_count"] == 0
    assert report["terminal_result_unsolicited_spoken"] is False
    assert report["terminal_result_status_available"] is True
    assert "completed: Finished Suppress terminal result." in report["terminal_result_status_text"]
    assert report["unflagged_high_risk_tool_smoke_ok"] is True
    assert report["unflagged_high_risk_tool_suppressed"] is True
    assert report["unflagged_high_risk_tool_failed_closed"] is True
    assert report["unflagged_high_risk_tool_suppression_reason"] == "unapproved_high_risk_tool_event"
    assert report["unflagged_high_risk_tool_progress_suppressed"] is True
    assert report["unflagged_high_risk_tool_payload_redacted"] is True
    assert report["unflagged_high_risk_tool_spoken_payload_clean"] is True
    assert report["unflagged_high_risk_tool_failure_spoken"] is True
    assert report["unflagged_high_risk_tool_secret_canary_checked"] is True
    assert report["unflagged_high_risk_tool_spoken"][0] == "Preparing the spend request."
    assert not any(
        "This unsafe tool result should not be spoken." in text
        for text in report["unflagged_high_risk_tool_spoken"]
    )
    assert report["sidecar_control_smoke_ok"] is True
    assert report["sidecar_control_update_observed"] is True
    assert report["sidecar_control_update_reached_oracle"] is True
    assert report["sidecar_control_cancel_requested"] is True
    assert report["sidecar_control_cancelled"] is True
    assert report["sidecar_control_cancel_reason"] == "sidecar smoke cancel"
    assert report["sidecar_control_completed_after_cancel"] is False
    assert report["sidecar_control_feedback_update_sent"] is True
    assert report["sidecar_control_feedback_cancel_sent"] is True
    assert report["external_frontend_bridge_smoke_ok"] is True
    assert report["external_frontend_request_accepted"] is True
    assert report["external_frontend_tool_result_observed"] is True
    assert report["external_frontend_job_id"] == "voice-oracle-001"
    assert report["external_frontend_provider"] == "voiceclaw"
    assert report["external_frontend_tool"] == "ask_brain"
    assert report["external_frontend_tool_call_id"] == "voiceclaw-call-1"
    assert report["external_frontend_completion_tool_call_id"] == "voiceclaw-call-1"
    assert report["external_frontend_status_tool_call_id"] == "voiceclaw-call-1"
    assert report["external_frontend_terminal_correlation_observed"] is True
    assert report["external_frontend_accepted_observed"] is True
    assert report["external_frontend_started_observed"] is True
    assert report["external_frontend_completion_observed"] is True
    assert report["external_frontend_status_state"] == "completed"
    assert report["external_frontend_source_reached_oracle"] is True
    assert report["external_frontend_input_source"] == "ask_brain"
    assert report["external_frontend_oracle_text"] == "Prepare external KAME handoff"
    assert report["external_frontend_evidence_bundle_propagated"] is True
    assert report["external_frontend_evidence_bundle_id"].startswith("kame-evidence-")
    assert report["external_frontend_evidence_bundle_id_stable"] is True
    assert report["external_frontend_evidence_bundle_single_turn"] is True
    assert report["external_frontend_evidence_bundle_status"] == "primary_audio"
    assert report["external_frontend_evidence_bundle_transcript_hypotheses_count"] == 2
    assert report["external_frontend_audio_segment_ref"] == "artifact://voiceclaw/turn-1.wav"
    assert report["external_frontend_audio_time_range_ms"] == [100, 2100]
    assert report["external_frontend_auxiliary_transcript_hypotheses"][0]["source"] == "moshi"
    assert report["external_frontend_auxiliary_transcript_hypotheses"][0]["authority"] == "hypothesis"
    assert report["external_frontend_hypothesis_not_durable_oracle_text"] is True
    assert report["external_frontend_durable_user_messages_empty"] is True
    assert report["external_frontend_durable_oracle_text_absent"] is True
    assert report["external_frontend_durable_record_count"] >= 1
    assert report["external_frontend_direct_tool_authority_exposed"] is False
    assert report["external_frontend_event_counts"]["tool.result"] == 1
    assert report["external_frontend_event_counts"]["oracle.job.accepted"] == 1
    assert report["external_frontend_event_counts"]["oracle.job.started"] == 1
    assert report["external_frontend_event_counts"]["oracle.job.completed"] == 1
    assert report["unpromoted_hypothesis_smoke_ok"] is True
    assert report["unpromoted_hypothesis_evidence_bundle_id"].startswith("kame-evidence-")
    assert report["unpromoted_hypothesis_single_bundle_observed"] is True
    assert report["unpromoted_hypothesis_status_bundle_status"] == "degraded_no_raw_audio"
    assert report["unpromoted_hypothesis_status_bundle_transcript_hypotheses_count"] == 2
    assert report["unpromoted_hypothesis_source"] == "moshi"
    assert report["unpromoted_hypothesis_authority"] == "hypothesis"
    assert report["unpromoted_hypothesis_text"] == "spend two hundred dollars and call my phone"
    assert report["unpromoted_hypothesis_confidence"] == 0.71
    assert report["unpromoted_hypothesis_oracle_text_preserved"] is True
    assert report["unpromoted_hypothesis_transcript_preserved"] is True
    assert report["unpromoted_hypothesis_intent_preserved"] is True
    assert report["unpromoted_hypothesis_attached"] is True
    assert report["unpromoted_hypothesis_promoted"] is False
    assert report["unpromoted_hypothesis_update_observed"] is True
    assert report["unpromoted_hypothesis_update_summary"] == "interpreter evidence: auxiliary_hypotheses=1"
    assert report["witness_fusion_timing_smoke_ok"] is True
    assert report["witness_fusion_arrival_phases"] == [
        "before_raw_audio",
        "with_raw_audio",
        "after_interpreter_start",
    ]
    assert report["witness_fusion_early_initial_bundle_id"].startswith("kame-evidence-")
    assert report["witness_fusion_early_initial_bundle_id"] == report["witness_fusion_early_final_bundle_id"]
    assert report["witness_fusion_early_single_bundle"] is True
    assert report["witness_fusion_with_bundle_id"].startswith("kame-evidence-")
    assert report["witness_fusion_with_single_bundle"] is True
    assert report["witness_fusion_late_initial_bundle_id"].startswith("kame-evidence-")
    assert report["witness_fusion_late_initial_bundle_id"] == report["witness_fusion_late_final_bundle_id"]
    assert report["witness_fusion_late_single_bundle"] is True
    assert report["witness_fusion_no_duplicate_oracle_jobs"] is True
    assert report["witness_fusion_adjudications"] == {
        "early": ["corrected_by_audio"],
        "with": ["accepted_as_supporting_evidence"],
        "late": ["rejected_or_diagnostic_only"],
    }
    assert report["witness_fusion_adjudication_outcomes_observed"] is True
    assert report["witness_fusion_accepted_counts"] == {"early": 1, "with": 1, "late": 1}
    assert report["witness_fusion_started_counts"] == {"early": 1, "with": 1, "late": 1}
    assert report["witness_fusion_completed_counts"] == {"early": 1, "with": 1, "late": 1}
    assert report["runtime_kame_action_gate_smoke_ok"] is True
    assert report["runtime_kame_action_gate_waiting_events"] == 5
    assert report["runtime_kame_action_gate_hypothesis_only_ok"] is False
    assert "missing_promoted_evidence" in report["runtime_kame_action_gate_hypothesis_only_issues"]
    assert "interpreter_evidence_not_consumed_before_irreversible_action" in (
        report["runtime_kame_action_gate_hypothesis_only_issues"]
    )
    assert set(report["runtime_kame_action_gate_hypothesis_only_rejected_authorities"]) >= {
        "reflex_hypothesis",
        "auxiliary_hypothesis",
    }
    assert report["runtime_kame_action_gate_degraded_text_only_ok"] is False
    assert report["runtime_kame_action_gate_degraded_text_only_status"] == "degraded_text_only"
    assert report["runtime_kame_action_gate_degraded_text_only_reason"] == "degraded_text_only"
    assert report["runtime_kame_action_gate_degraded_text_only_raw_audio_available"] is False
    assert report["runtime_kame_action_gate_degraded_text_only_preserves_hypothesis"] is True
    assert "missing_promoted_evidence" in report["runtime_kame_action_gate_degraded_text_only_issues"]
    assert "interpreter_evidence_not_consumed_before_irreversible_action" in (
        report["runtime_kame_action_gate_degraded_text_only_issues"]
    )
    assert set(report["runtime_kame_action_gate_degraded_text_only_rejected_authorities"]) >= {
        "reflex_hypothesis",
        "auxiliary_hypothesis",
    }
    assert report["runtime_kame_action_gate_promoted_ok"] is True
    assert report["runtime_kame_action_gate_promoted_issues"] == []
    assert report["runtime_kame_action_gate_promoted_authorities"] == ["interpreter_promoted"]
    assert report["runtime_kame_action_gate_promoted_consumed_before_action"] is True
    assert report["runtime_kame_action_gate_self_attested_ok"] is False
    assert "missing_promoted_evidence" in report["runtime_kame_action_gate_self_attested_issues"]
    assert "interpreter_evidence_not_consumed_before_irreversible_action" not in (
        report["runtime_kame_action_gate_self_attested_issues"]
    )
    assert report["runtime_kame_action_gate_self_attested_authorities"] == []
    assert report["runtime_kame_action_gate_self_attested_consumed_before_action"] is True
    assert report["runtime_kame_action_gate_missing_tool_disclosure_ok"] is False
    assert "missing_tool_disclosure_ref" in report[
        "runtime_kame_action_gate_missing_tool_disclosure_issues"
    ]
    assert "missing_promoted_evidence" not in report[
        "runtime_kame_action_gate_missing_tool_disclosure_issues"
    ]
    assert "interpreter_evidence_not_consumed_before_irreversible_action" not in report[
        "runtime_kame_action_gate_missing_tool_disclosure_issues"
    ]
    assert report["runtime_kame_action_gate_missing_tool_disclosure_authorities"] == [
        "interpreter_promoted"
    ]
    assert report["runtime_kame_action_gate_tool_disclosure_ref_observed"] is True
    assert report["runtime_kame_action_gate_schema_versions"] == [
        "voiceops.runtime_kame_action_gate.v1",
        "voiceops.runtime_kame_action_gate.v1",
        "voiceops.runtime_kame_action_gate.v1",
        "voiceops.runtime_kame_action_gate.v1",
        "voiceops.runtime_kame_action_gate.v1",
    ]
    assert report["event_counts"]["interface.oracle.update"] >= 2
    assert report["event_counts"]["oracle.job.progress"] >= 1
    assert report["event_counts"]["oracle.job.result_suppressed"] >= 1
