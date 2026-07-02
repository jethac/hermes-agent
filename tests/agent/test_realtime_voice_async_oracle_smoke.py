import pytest

from scripts.realtime_voice_async_oracle_smoke import run_smoke


@pytest.mark.asyncio
async def test_async_oracle_smoke_proves_concurrency_local_turn_and_cancellation():
    report = await run_smoke()

    assert report["ok"] is True
    assert report["max_running"] == 4
    assert report["started_jobs"] == 4
    assert report["completed_jobs"] == 3
    assert report["cancelled_jobs"] == 1
    assert report["local_turn_committed"] is True
    assert report["late_cancelled_output_attempted"] is True
    assert report["cancelled_result_spoken"] is False
    assert report["cancelled_result_committed"] is False
    assert report["cancelled_result_progress_leaked"] is False
