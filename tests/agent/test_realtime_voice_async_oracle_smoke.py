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
    assert report["started_jobs"] == 10
    assert report["queued_jobs"] == 1
    assert report["completed_jobs"] == 6
    assert report["failed_jobs"] == 1
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
    assert report["approval_completed"] is True
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
    assert report["sidecar_control_smoke_ok"] is True
    assert report["sidecar_control_update_observed"] is True
    assert report["sidecar_control_update_reached_oracle"] is True
    assert report["sidecar_control_cancel_requested"] is True
    assert report["sidecar_control_cancelled"] is True
    assert report["sidecar_control_cancel_reason"] == "sidecar smoke cancel"
    assert report["sidecar_control_completed_after_cancel"] is False
    assert report["sidecar_control_feedback_update_sent"] is True
    assert report["sidecar_control_feedback_cancel_sent"] is True
    assert report["event_counts"]["interface.oracle.update"] >= 2
    assert report["event_counts"]["oracle.job.progress"] >= 1
    assert report["event_counts"]["oracle.job.result_suppressed"] >= 1
