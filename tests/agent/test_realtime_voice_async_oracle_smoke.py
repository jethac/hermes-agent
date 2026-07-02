import pytest

from scripts.realtime_voice_async_oracle_smoke import run_smoke


@pytest.mark.asyncio
async def test_async_oracle_smoke_proves_concurrency_local_turn_and_cancellation():
    report = await run_smoke()

    assert report["ok"] is True
    assert report["max_running"] == 4
    assert report["started_jobs"] == 5
    assert report["queued_jobs"] == 1
    assert report["completed_jobs"] == 4
    assert report["cancelled_jobs"] == 1
    assert report["local_turn_committed"] is True
    assert report["status_turn_committed"] is True
    assert report["status_text"].startswith("Oracle jobs: 4 running out of 4, 1 queued.")
    assert report["fifth_job_queued"] is True
    assert report["fifth_job_started_after_capacity_freed"] is True
    assert report["late_cancelled_output_attempted"] is True
    assert report["cancelled_result_spoken"] is False
    assert report["cancelled_result_committed"] is False
    assert report["cancelled_result_progress_leaked"] is False
    assert report["cancelled_result_durable_completed"] is False
    assert report["cancelled_result_durable_text"] is False
    assert report["durable_cancelled_record_present"] is True
    assert report["durable_completed_jobs"] == 4
