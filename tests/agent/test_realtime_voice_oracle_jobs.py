import asyncio
import json

import pytest

from agent.realtime_voice_kame import KameOracleRequest, KameRoute
from agent.realtime_voice import RealtimeVoiceSessionConfig, VoiceEventType
from agent.realtime_voice_oracle_jobs import (
    OracleJobManager,
    OracleJobQueueFullError,
    OracleJobReprioritizationRequiredError,
    OracleJobState,
)
from agent.realtime_voice_text_engine import _oracle_request_for_job


def _request(text: str, *, route: KameRoute = KameRoute.ORACLE_DIRECT) -> KameOracleRequest:
    return KameOracleRequest(
        session_id="voice-session-1",
        turn_id=f"turn:{text}",
        source="discord_voice",
        user_id="42",
        intent=text,
        route=route,
        interface_already_said=f"I'm handling {text}.",
    )


def test_oracle_job_protocol_surface_is_wire_serializable():
    assert VoiceEventType("oracle.job.accepted") == VoiceEventType.ORACLE_JOB_ACCEPTED
    assert (
        VoiceEventType("oracle.job.interpreter_evidence_attached")
        == VoiceEventType.ORACLE_JOB_INTERPRETER_EVIDENCE_ATTACHED
    )
    assert (
        VoiceEventType("oracle.job.interpreter_evidence_late")
        == VoiceEventType.ORACLE_JOB_INTERPRETER_EVIDENCE_LATE
    )
    assert (
        VoiceEventType("oracle.job.result_suppressed")
        == VoiceEventType.ORACLE_JOB_RESULT_SUPPRESSED
    )
    assert VoiceEventType("interface.oracle.update") == VoiceEventType.INTERFACE_ORACLE_UPDATE

    config = RealtimeVoiceSessionConfig(
        session_id="voice-session-1",
        oracle_jobs={
            "max_concurrent": 4,
            "queue_limit": 16,
            "default_priority": "normal",
            "overflow_policy": "queue",
        },
    )
    wire = config.to_wire()
    restored = RealtimeVoiceSessionConfig.from_wire(wire)

    assert wire["oracle_jobs"]["max_concurrent"] == 4
    assert restored.oracle_jobs == config.oracle_jobs


@pytest.mark.asyncio
async def test_submit_starts_job_with_stable_id_and_events():
    events = []

    async def runner(job):
        return {"result_summary": f"done {job.oracle_text}"}

    manager = OracleJobManager(
        max_concurrent=1,
        runner=runner,
        event_callback=lambda event: events.append(event.type.value),
    )

    job = await manager.submit(_request("check logs"))
    await manager.wait_for_idle()
    completed = await manager.get(job.job_id)

    assert job.job_id == "voice-oracle-001"
    assert completed.state == OracleJobState.COMPLETED
    assert completed.result_summary == "done check logs"
    assert events == [
        "oracle.job.accepted",
        "oracle.job.started",
        "oracle.job.completed",
    ]


@pytest.mark.asyncio
async def test_job_status_exposes_bounded_kame_evidence_contract_fields():
    release = asyncio.Event()

    async def runner(job):
        await release.wait()
        return "done"

    manager = OracleJobManager(max_concurrent=1, runner=runner)
    request = KameOracleRequest(
        session_id="voice-session-1",
        turn_id="turn-voice-1",
        source="discord_voice",
        user_id="42",
        intent="calculate a power",
        route=KameRoute.ORACLE_DIRECT,
        reflex_transcript_hypothesis="three to the power of seventeen with sk_test_abcdefghijklmnopqrstuvwxyz",
        reflex_transcript_source="reflex_audio",
        reflex_transcript_confidence=0.71,
        audio_segment_ref="segments/turn-voice-1.wav",
        audio_time_range_ms=(120, 2480),
        auxiliary_transcript_hypotheses=(
            {
                "source": "moshi",
                "text": "three to the power of seventeen",
                "confidence": 0.78,
            },
            {
                "source": "asr",
                "text": "what is three to the power of seventeen",
                "confidence": 0.92,
            },
        ),
    )

    job = await manager.submit(request)
    await manager.add_interpreter_evidence(
        job.job_id,
        corrected_transcript="what is three to the power of seventeen",
        normalized_intent="answer a math question",
        entities=[{"type": "math_expression", "value": "3^17"}],
        confidence=0.94,
        disagreements=["reflex transcript omitted request prefix"],
    )
    status = await manager.status_view()
    job_status = next(item for item in status["jobs"] if item["job_id"] == job.job_id)
    combined = json.dumps(job_status, sort_keys=True)

    assert job.audio_segment_ref == "segments/turn-voice-1.wav"
    assert job.audio_time_range_ms == (120, 2480)
    assert "sk_test" not in job.reflex_transcript_hypothesis
    assert job.reflex_transcript_source == "reflex_audio"
    assert job.auxiliary_transcript_hypotheses == (
        {"source": "moshi", "text": "three to the power of seventeen", "authority": "hypothesis", "confidence": 0.78},
        {
            "source": "asr",
            "text": "what is three to the power of seventeen",
            "authority": "hypothesis",
            "confidence": 0.92,
        },
    )
    assert job.interpreter_corrected_transcript == "what is three to the power of seventeen"
    assert job.interpreter_confidence == 0.94
    assert job.interpreter_entities == ({"type": "math_expression", "value": "3^17"},)
    assert job.interpreter_disagreements == ("reflex transcript omitted request prefix",)
    assert job_status["audio_segment_ref"] == "segments/turn-voice-1.wav"
    assert job_status["audio_time_range_ms"] == (120, 2480)
    assert job_status["evidence_authority"] == {
        "intent": "reflex_hypothesis",
        "oracle_text": "reflex_hypothesis",
        "raw_audio": "primary_audio",
        "reflex_transcript_hypothesis": "reflex_hypothesis",
        "auxiliary_transcript_hypotheses": "auxiliary_hypothesis",
        "interpreter_corrected_transcript": "interpreter_promoted",
        "interpreter_normalized_intent": "interpreter_promoted",
    }
    assert "Bearer" not in combined
    assert "sk_test" not in combined
    assert job_status["reflex_transcript_source"] == "reflex_audio"
    assert job_status["reflex_transcript_confidence"] == 0.71
    assert job_status["auxiliary_transcript_hypotheses_count"] == 2
    assert job_status["transcript_hypotheses_count"] == 3
    assert job_status["transcript_hypotheses"][0]["kind"] == "reflex_transcript_hypothesis"
    assert job_status["transcript_hypotheses"][0]["source"] == "reflex_audio"
    assert job_status["transcript_hypotheses"][0]["authority"] == "reflex_hypothesis"
    assert job_status["transcript_hypotheses"][0]["confidence"] == 0.71
    assert job_status["transcript_hypotheses"][0]["text"].startswith("three to the power of seventeen with")
    assert "sk_test" not in job_status["transcript_hypotheses"][0]["text"]
    assert job_status["transcript_hypotheses"][1:] == (
        {
            "kind": "s2s_transcript_hypothesis",
            "source": "moshi",
            "text": "three to the power of seventeen",
            "authority": "auxiliary_hypothesis",
            "confidence": 0.78,
        },
        {
            "kind": "classic_asr_hypothesis",
            "source": "asr",
            "text": "what is three to the power of seventeen",
            "authority": "auxiliary_hypothesis",
            "confidence": 0.92,
        },
    )
    assert job_status["interpreter_corrected_transcript"] == "what is three to the power of seventeen"
    assert job_status["interpreter_confidence"] == 0.94
    assert job_status["interpreter_entities"] == ({"type": "math_expression", "value": "3^17"},)
    assert job_status["interpreter_disagreements"] == ("reflex transcript omitted request prefix",)
    assert "metadata" not in job_status
    assert "oracle_text" not in job_status

    await manager.cancel(job.job_id)
    release.set()
    await manager.wait_for_idle()


@pytest.mark.asyncio
async def test_max_concurrent_one_queues_second_job():
    started = []
    release_first = asyncio.Event()

    async def runner(job):
        started.append(job.job_id)
        if job.job_id == "voice-oracle-001":
            await release_first.wait()
        return f"finished {job.job_id}"

    manager = OracleJobManager(max_concurrent=1, runner=runner)

    first = await manager.submit(_request("first"))
    second = await manager.submit(_request("second"))
    await asyncio.sleep(0)

    assert (await manager.get(first.job_id)).state == OracleJobState.RUNNING
    assert (await manager.get(second.job_id)).state == OracleJobState.QUEUED
    assert started == ["voice-oracle-001"]

    release_first.set()
    await manager.wait_for_idle()

    assert started == ["voice-oracle-001", "voice-oracle-002"]
    assert (await manager.get(second.job_id)).state == OracleJobState.COMPLETED


@pytest.mark.asyncio
async def test_queued_job_uses_runner_supplied_at_submission_time():
    calls = []
    release_first = asyncio.Event()

    async def default_runner(job):
        calls.append(("default", job.job_id))
        if job.job_id == "voice-oracle-001":
            await release_first.wait()
        return f"default {job.job_id}"

    async def override_runner(job):
        calls.append(("override", job.job_id))
        return f"override {job.job_id}"

    manager = OracleJobManager(max_concurrent=1, runner=default_runner)

    first = await manager.submit(_request("first"))
    second = await manager.submit(_request("second"), runner=override_runner)
    await asyncio.sleep(0)

    assert (await manager.get(first.job_id)).state == OracleJobState.RUNNING
    assert (await manager.get(second.job_id)).state == OracleJobState.QUEUED

    release_first.set()
    await manager.wait_for_idle()

    assert calls == [("default", first.job_id), ("override", second.job_id)]
    assert (await manager.get(first.job_id)).result_summary == f"default {first.job_id}"
    assert (await manager.get(second.job_id)).result_summary == f"override {second.job_id}"


@pytest.mark.asyncio
async def test_queued_job_with_submission_runner_starts_without_default_runner():
    calls = []
    release_first = asyncio.Event()

    async def first_runner(job):
        calls.append(("first", job.job_id))
        await release_first.wait()
        return "first done"

    async def second_runner(job):
        calls.append(("second", job.job_id))
        return "second done"

    manager = OracleJobManager(max_concurrent=1)

    first = await manager.submit(_request("first"), runner=first_runner)
    second = await manager.submit(_request("second"), runner=second_runner)
    await asyncio.sleep(0)

    assert (await manager.get(first.job_id)).state == OracleJobState.RUNNING
    assert (await manager.get(second.job_id)).state == OracleJobState.QUEUED

    release_first.set()
    await manager.wait_for_idle()

    assert calls == [("first", first.job_id), ("second", second.job_id)]
    assert (await manager.get(first.job_id)).state == OracleJobState.COMPLETED
    assert (await manager.get(second.job_id)).state == OracleJobState.COMPLETED


@pytest.mark.asyncio
async def test_high_priority_queued_job_starts_before_low_priority_job():
    started = []
    release_first = asyncio.Event()

    async def runner(job):
        started.append(job.job_id)
        if job.job_id == "voice-oracle-001":
            await release_first.wait()
        return f"finished {job.job_id}"

    manager = OracleJobManager(max_concurrent=1, runner=runner)

    await manager.submit(_request("running"))
    low = await manager.submit(_request("low"), priority="low")
    high = await manager.submit(_request("high"), priority="high")
    await asyncio.sleep(0)

    assert (await manager.get(low.job_id)).priority == "low"
    assert (await manager.get(high.job_id)).priority == "high"

    release_first.set()
    await manager.wait_for_idle()

    assert started == ["voice-oracle-001", high.job_id, low.job_id]


@pytest.mark.asyncio
async def test_equal_priority_queued_jobs_start_fifo():
    started = []
    release_first = asyncio.Event()

    async def runner(job):
        started.append(job.job_id)
        if job.job_id == "voice-oracle-001":
            await release_first.wait()
        return f"finished {job.job_id}"

    manager = OracleJobManager(max_concurrent=1, runner=runner)

    await manager.submit(_request("running"))
    second = await manager.submit(_request("second"), priority="normal")
    third = await manager.submit(_request("third"), priority="normal")
    await asyncio.sleep(0)

    release_first.set()
    await manager.wait_for_idle()

    assert started == ["voice-oracle-001", second.job_id, third.job_id]


@pytest.mark.asyncio
async def test_reprioritizing_queued_job_moves_it_ahead_before_capacity_frees():
    events = []
    started = []
    release_first = asyncio.Event()

    async def runner(job):
        started.append(job.job_id)
        if job.job_id == "voice-oracle-001":
            await release_first.wait()
        return f"finished {job.job_id}"

    manager = OracleJobManager(
        max_concurrent=1,
        runner=runner,
        event_callback=lambda event: events.append(event.to_status()),
    )

    await manager.submit(_request("running"))
    second = await manager.submit(_request("second"), priority="normal")
    third = await manager.submit(_request("third"), priority="normal")
    await asyncio.sleep(0)

    updated = await manager.update_priority(third.job_id, priority="highest")
    progress = next(
        event
        for event in events
        if event["type"] == "oracle.job.progress"
        and event["job_id"] == third.job_id
        and event["payload"].get("operation") == "priority"
    )
    assert updated.priority == "high"
    assert progress["payload"]["priority"] == "high"
    assert progress["payload"]["state"] == "queued"

    release_first.set()
    await manager.wait_for_idle()

    assert started == ["voice-oracle-001", third.job_id, second.job_id]


@pytest.mark.asyncio
async def test_reflex_status_orders_queued_jobs_by_current_priority_queue():
    release_first = asyncio.Event()

    async def runner(job):
        if job.job_id == "voice-oracle-001":
            await release_first.wait()
        return f"finished {job.job_id}"

    manager = OracleJobManager(max_concurrent=1, runner=runner)

    await manager.submit(_request("running"))
    second = await manager.submit(_request("second"), priority="normal")
    third = await manager.submit(_request("third"), priority="normal")
    await asyncio.sleep(0)

    await manager.update_priority(third.job_id, priority="highest")

    status = await manager.status_view()
    assert [
        (job["job_id"], job["state"], job["ordinal"], job["ordinal_label"], job["priority"])
        for job in status["reflex"]["jobs"][:3]
    ] == [
        ("voice-oracle-001", "running", 1, "job one", "normal"),
        (third.job_id, "queued", 2, "job two", "high"),
        (second.job_id, "queued", 3, "job three", "normal"),
    ]

    release_first.set()
    await manager.wait_for_idle()


@pytest.mark.asyncio
async def test_add_update_records_compact_status_without_running_job():
    events = []
    release = asyncio.Event()

    async def runner(job):
        await release.wait()
        return "done"

    manager = OracleJobManager(
        max_concurrent=1,
        runner=runner,
        event_callback=lambda event: events.append(event.to_status()),
    )

    first = await manager.submit(_request("running"))
    second = await manager.submit(_request("second"))
    await asyncio.sleep(0)

    updated = await manager.add_update(
        second.job_id,
        text="also check the Stripe receipt before answering",
        source="discord_voice",
        update_type="clarification",
    )
    status = await manager.status_view()
    progress = next(
        event
        for event in events
        if event["type"] == "oracle.job.progress"
        and event["job_id"] == second.job_id
        and event["payload"].get("operation") == "update"
    )

    assert updated.updates == [
        {
            "text": "also check the Stripe receipt before answering",
            "source": "discord_voice",
            "type": "clarification",
            "created_at": updated.updates[0]["created_at"],
        }
    ]
    queued_status = next(job for job in status["jobs"] if job["job_id"] == second.job_id)
    assert queued_status["update_count"] == 1
    assert queued_status["latest_update"] == "also check the Stripe receipt before answering"
    assert progress["payload"]["update_count"] == 1
    assert progress["payload"]["latest_update"] == "also check the Stripe receipt before answering"
    assert progress["payload"]["state"] == "queued"

    await manager.cancel(first.job_id)
    release.set()
    await manager.wait_for_idle()


@pytest.mark.asyncio
async def test_add_update_redacts_secret_like_text_from_status_and_events():
    events = []
    release = asyncio.Event()

    async def runner(job):
        await release.wait()
        return "done"

    manager = OracleJobManager(
        max_concurrent=1,
        runner=runner,
        event_callback=lambda event: events.append(event.to_status()),
    )

    running = await manager.submit(_request("running"))
    queued = await manager.submit(_request("queued"))
    await asyncio.sleep(0)

    updated = await manager.add_update(
        queued.job_id,
        text=(
            "use Authorization: Bearer raw-token and "
            "sk_test_abcdefghijklmnopqrstuvwxyz before answering"
        ),
        source="discord_voice",
        update_type="clarification",
    )
    status = await manager.status_view()
    progress = next(
        event
        for event in events
        if event["type"] == "oracle.job.progress"
        and event["job_id"] == queued.job_id
        and event["payload"].get("operation") == "update"
    )
    combined = json.dumps(
        {
            "stored": updated.updates,
            "status": status,
            "event": progress,
        },
        sort_keys=True,
    )

    queued_status = next(job for job in status["jobs"] if job["job_id"] == queued.job_id)
    assert "Bearer ***" in queued_status["latest_update"]
    assert "sk_tes" in queued_status["latest_update"]
    assert progress["payload"]["latest_update"] == queued_status["latest_update"]
    assert "raw-token" not in combined
    assert "sk_test_abcdefghijklmnopqrstuvwxyz" not in combined

    await manager.cancel(running.job_id)
    release.set()
    await manager.wait_for_idle()


@pytest.mark.asyncio
async def test_find_by_evidence_key_resolves_active_job_without_internal_job_id():
    release = asyncio.Event()

    async def runner(job):
        await release.wait()
        return {"result_summary": f"done {job.oracle_text}"}

    manager = OracleJobManager(max_concurrent=1, runner=runner)
    request = KameOracleRequest(
        session_id="voice-session-1",
        turn_id="turn:audio-001",
        source="discord_voice",
        user_id="42",
        intent="answer a math question",
        route=KameRoute.DEFER,
        audio_segment_ref="artifact://redacted/audio-001.wav",
        interface_already_said="I'm checking the math.",
    )
    job = await manager.submit(request)

    resolved = await manager.find_by_evidence_key(
        turn_id="turn:audio-001",
        audio_segment_ref="artifact://redacted/audio-001.wav",
    )

    assert resolved.job_id == job.job_id
    release.set()
    await manager.wait_for_idle()


@pytest.mark.asyncio
async def test_interpreter_evidence_updates_queued_job_before_execution():
    events = []
    release = asyncio.Event()

    async def runner(job):
        await release.wait()
        return "done"

    manager = OracleJobManager(
        max_concurrent=1,
        runner=runner,
        event_callback=lambda event: events.append(event.to_status()),
    )

    running = await manager.submit(_request("running"))
    queued = await manager.submit(_request("power question"))
    await asyncio.sleep(0)

    updated = await manager.add_interpreter_evidence(
        queued.job_id,
        corrected_transcript="what is three to the power of seventeen",
        normalized_intent="answer a math question",
        audio_segment_ref="artifact://redacted/oracle-002.wav",
        audio_time_range_ms=[120, 2120],
        reflex_transcript_hypothesis={
            "source": "moshi",
            "text": "three to the power of seventeen",
            "confidence": 0.71,
        },
        auxiliary_transcript_hypotheses=[
            {
                "source": "classic_asr_fallback_optional",
                "text": "what is three to the power of seventeen",
                "confidence": 0.88,
            }
        ],
        speaker_metadata={
            "platform": "discord",
            "channel_user_id": "42",
            "display_name": "jetha",
            "profile": {"token": "must-not-copy"},
        },
        channel_metadata={
            "transport": "discord_voice",
            "guild_id": "guild-1",
            "channel_id": "general",
            "webhook": {"token": "must-not-copy"},
        },
        entities=[{"type": "math_expression", "value": "3^17"}],
        confidence=0.94,
        disagreements=["reflex transcript omitted request prefix"],
        source="gemma_interpreter",
    )
    status = await manager.status_view()
    attached = next(
        event
        for event in events
        if event["type"] == "oracle.job.interpreter_evidence_attached"
        and event["job_id"] == queued.job_id
    )
    queued_status = next(job for job in status["jobs"] if job["job_id"] == queued.job_id)
    oracle_request = _oracle_request_for_job(updated, updated.request)

    assert updated.interpreter_evidence[0]["corrected_transcript"] == "what is three to the power of seventeen"
    assert updated.interpreter_evidence[0]["audio_segment_ref"] == "artifact://redacted/oracle-002.wav"
    assert updated.interpreter_evidence[0]["audio_time_range_ms"] == (120, 2120)
    assert updated.interpreter_evidence[0]["reflex_transcript_hypothesis"] == {
        "source": "moshi",
        "text": "three to the power of seventeen",
        "authority": "hypothesis",
        "confidence": 0.71,
    }
    assert updated.interpreter_evidence[0]["auxiliary_transcript_hypotheses"] == (
        {
            "source": "classic_asr_fallback_optional",
            "text": "what is three to the power of seventeen",
            "authority": "hypothesis",
            "confidence": 0.88,
        },
    )
    assert updated.interpreter_evidence[0]["evidence_authority"] == {
        "raw_audio": "primary_audio",
        "reflex_transcript_hypothesis": "reflex_hypothesis",
        "auxiliary_transcript_hypotheses": "auxiliary_hypothesis",
        "speaker_metadata": "diagnostic_only",
        "channel_metadata": "diagnostic_only",
        "interpreter_corrected_transcript": "interpreter_promoted",
        "interpreter_normalized_intent": "interpreter_promoted",
        "interpreter_entities": "interpreter_promoted",
        "interpreter_disagreements": "diagnostic_only",
    }
    assert updated.interpreter_evidence[0]["late"] is False
    assert queued_status["intent"] == "power question"
    assert queued_status["interpreter_normalized_intent"] == "answer a math question"
    assert queued_status["interpreter_intent_source"] == "gemma_interpreter"
    assert queued_status["audio_segment_ref"] == "artifact://redacted/oracle-002.wav"
    assert queued_status["audio_time_range_ms"] == (120, 2120)
    assert queued_status["speaker"] == {
        "platform": "discord",
        "channel_user_id": "42",
        "display_name": "jetha",
    }
    assert queued_status["channel"] == {
        "transport": "discord_voice",
        "guild_id": "guild-1",
        "channel_id": "general",
    }
    assert queued_status["reflex_transcript_hypothesis"] == "three to the power of seventeen"
    assert queued_status["reflex_transcript_source"] == "moshi"
    assert queued_status["auxiliary_transcript_hypotheses_count"] == 1
    assert queued_status["interpreter_evidence_count"] == 1
    assert queued_status["interpreter_evidence_late"] is False
    assert queued_status["latest_interpreter_evidence_authority"] == updated.interpreter_evidence[0]["evidence_authority"]
    assert queued_status["evidence_authority"]["interpreter_corrected_transcript"] == "interpreter_promoted"
    assert queued_status["evidence_authority"]["interpreter_normalized_intent"] == "interpreter_promoted"
    assert "transcript=what is three to the power of seventeen" in queued_status["latest_interpreter_evidence"]
    assert "audio=attached" in queued_status["latest_interpreter_evidence"]
    assert "speaker=attached" in queued_status["latest_interpreter_evidence"]
    assert "channel=attached" in queued_status["latest_interpreter_evidence"]
    assert "auxiliary_hypotheses=1" in queued_status["latest_interpreter_evidence"]
    assert attached["payload"]["operation"] == "interpreter_evidence"
    assert attached["payload"]["interpreter_evidence_late"] is False
    assert attached["payload"]["speaker"] == queued_status["speaker"]
    assert attached["payload"]["channel"] == queued_status["channel"]
    assert attached["payload"]["latest_interpreter_evidence_authority"] == updated.interpreter_evidence[0]["evidence_authority"]
    assert oracle_request.oracle_text == "what is three to the power of seventeen"
    assert oracle_request.oracle_text_source == "gemma_interpreter"
    assert oracle_request.audio_segment_ref == "artifact://redacted/oracle-002.wav"
    assert oracle_request.audio_time_range_ms == (120, 2120)
    assert oracle_request.speaker_metadata == queued_status["speaker"]
    assert oracle_request.channel_metadata == queued_status["channel"]
    assert oracle_request.auxiliary_transcript_hypotheses == (
        {
            "source": "moshi",
            "text": "three to the power of seventeen",
            "authority": "hypothesis",
            "confidence": 0.71,
        },
        {
            "source": "classic_asr_fallback_optional",
            "text": "what is three to the power of seventeen",
            "authority": "hypothesis",
            "confidence": 0.88,
        },
    )
    assert oracle_request.transcript == "what is three to the power of seventeen"
    assert oracle_request.transcript_source == "gemma_interpreter"
    assert oracle_request.transcript_confidence == 0.94
    assert oracle_request.intent == "answer a math question"
    assert oracle_request.intent_source == "gemma_interpreter"
    assert oracle_request.asr_transcript == ""
    assert any(
        "entities=math_expression=3^17" in update
        for update in oracle_request.job_updates
    )
    assert "must-not-copy" not in json.dumps(
        {
            "stored": updated.interpreter_evidence,
            "status": queued_status,
            "event": attached,
            "request_metadata": oracle_request.to_metadata(),
        },
        sort_keys=True,
    )

    await manager.cancel(running.job_id)
    release.set()
    await manager.wait_for_idle()


@pytest.mark.asyncio
async def test_interpreter_evidence_delivery_status_emits_progress_event():
    events = []
    release = asyncio.Event()

    async def runner(job):
        await release.wait()
        return "done"

    manager = OracleJobManager(
        max_concurrent=1,
        runner=runner,
        event_callback=lambda event: events.append(event.to_status()),
    )
    running = await manager.submit(_request("prepare spend approval"))
    await asyncio.sleep(0)
    await manager.add_interpreter_evidence(
        running.job_id,
        corrected_transcript="buy twenty dollars of phone credits",
        normalized_intent="prepare a Stripe approval for phone credits",
        source="gemma_interpreter",
    )

    await manager.mark_latest_interpreter_evidence_delivery(
        running.job_id,
        delivered_to_oracle=True,
        consumed_before_irreversible_action=True,
        delivery_status="included in approval packet",
    )

    delivery = next(
        event
        for event in events
        if event["type"] == "oracle.job.progress"
        and event["payload"].get("operation") == "interpreter_evidence_delivery"
    )
    status = await manager.status_view()
    running_status = next(job for job in status["jobs"] if job["job_id"] == running.job_id)

    assert delivery["job_id"] == running.job_id
    assert delivery["payload"]["interpreter_evidence_delivered_to_oracle"] is True
    assert delivery["payload"]["interpreter_evidence_consumed_before_irreversible_action"] is True
    assert delivery["payload"]["interpreter_evidence_delivery_status"] == "included in approval packet"
    assert running_status["interpreter_evidence_delivered_to_oracle"] is True
    assert running_status["interpreter_evidence_consumed_before_irreversible_action"] is True
    assert running_status["interpreter_evidence_delivery_status"] == "included in approval packet"

    release.set()
    await manager.wait_for_idle()


@pytest.mark.asyncio
async def test_interpreter_evidence_late_for_running_job_is_status_visible():
    events = []
    release = asyncio.Event()

    async def runner(job):
        await release.wait()
        return "done"

    manager = OracleJobManager(
        max_concurrent=1,
        runner=runner,
        event_callback=lambda event: events.append(event.to_status()),
    )

    running = await manager.submit(_request("running"))
    await asyncio.sleep(0)

    updated = await manager.add_interpreter_evidence(
        running.job_id,
        corrected_transcript="check the current deployment logs",
        normalized_intent="inspect deployment logs",
        audio_segment_ref="artifact://redacted/running.wav",
        audio_time_range_ms=(200, 2400),
        reflex_transcript_hypothesis={
            "source": "moshi",
            "text": "check deployment",
            "confidence": 0.66,
        },
        auxiliary_transcript_hypotheses=[
            {
                "source": "classic_asr_fallback_optional",
                "text": "check the current deployment logs",
            }
        ],
        speaker_metadata={
            "platform": "discord",
            "channel_user_id": "42",
            "display_name": "jetha",
            "profile": {"token": "must-not-copy"},
        },
        channel_metadata={
            "transport": "discord_voice",
            "guild_id": "guild-1",
            "channel_id": "general",
            "webhook": {"token": "must-not-copy"},
        },
        confidence=0.81,
        source="gemma_interpreter",
    )
    status = await manager.status_view()
    late = next(
        event
        for event in events
        if event["type"] == "oracle.job.interpreter_evidence_late"
        and event["job_id"] == running.job_id
    )
    running_status = next(job for job in status["jobs"] if job["job_id"] == running.job_id)
    oracle_request = _oracle_request_for_job(updated, updated.request)

    assert updated.interpreter_evidence[0]["late"] is True
    assert updated.interpreter_evidence[0]["evidence_authority"] == {
        "raw_audio": "primary_audio",
        "reflex_transcript_hypothesis": "reflex_hypothesis",
        "auxiliary_transcript_hypotheses": "auxiliary_hypothesis",
        "speaker_metadata": "diagnostic_only",
        "channel_metadata": "diagnostic_only",
        "interpreter_corrected_transcript": "interpreter_promoted",
        "interpreter_normalized_intent": "interpreter_promoted",
    }
    assert running_status["audio_segment_ref"] == "artifact://redacted/running.wav"
    assert running_status["audio_time_range_ms"] == (200, 2400)
    assert running_status["speaker"] == {
        "platform": "discord",
        "channel_user_id": "42",
        "display_name": "jetha",
    }
    assert running_status["channel"] == {
        "transport": "discord_voice",
        "guild_id": "guild-1",
        "channel_id": "general",
    }
    assert running_status["reflex_transcript_hypothesis"] == "check deployment"
    assert running_status["reflex_transcript_source"] == "moshi"
    assert running_status["auxiliary_transcript_hypotheses_count"] == 1
    assert running_status["interpreter_evidence_late"] is True
    assert running_status["latest_interpreter_evidence_authority"] == updated.interpreter_evidence[0]["evidence_authority"]
    assert late["payload"]["interpreter_evidence_late"] is True
    assert late["payload"]["latest_interpreter_evidence_authority"] == updated.interpreter_evidence[0]["evidence_authority"]
    assert late["payload"]["audio_segment_ref"] == "artifact://redacted/running.wav"
    assert late["payload"]["speaker"] == running_status["speaker"]
    assert late["payload"]["channel"] == running_status["channel"]
    assert "audio=attached" in late["payload"]["latest_interpreter_evidence"]
    assert "speaker=attached" in late["payload"]["latest_interpreter_evidence"]
    assert "channel=attached" in late["payload"]["latest_interpreter_evidence"]
    assert "reflex_hypothesis=attached" in late["payload"]["latest_interpreter_evidence"]
    assert oracle_request.oracle_text == "running"
    assert oracle_request.intent == "running"
    assert oracle_request.intent_source == "reflex_audio"
    assert oracle_request.speaker_metadata == running_status["speaker"]
    assert oracle_request.channel_metadata == running_status["channel"]
    assert any("intent=inspect deployment logs" in update for update in oracle_request.job_updates)
    assert "must-not-copy" not in json.dumps(
        {
            "stored": updated.interpreter_evidence,
            "status": running_status,
            "event": late,
            "request_metadata": oracle_request.to_metadata(),
        },
        sort_keys=True,
    )

    release.set()
    await manager.wait_for_idle()


@pytest.mark.asyncio
async def test_interpreter_evidence_redacts_secret_like_text_from_status_events_and_request_updates():
    events = []
    release = asyncio.Event()

    async def runner(job):
        await release.wait()
        return "done"

    manager = OracleJobManager(
        max_concurrent=1,
        runner=runner,
        event_callback=lambda event: events.append(event.to_status()),
    )

    running = await manager.submit(_request("running"))
    queued = await manager.submit(_request("queued"))
    await asyncio.sleep(0)

    updated = await manager.add_interpreter_evidence(
        queued.job_id,
        corrected_transcript=(
            "use Authorization: Bearer raw-token and "
            "sk_test_abcdefghijklmnopqrstuvwxyz before answering"
        ),
        normalized_intent="check a credential-bearing request",
        confidence=0.9,
    )
    status = await manager.status_view()
    attached = next(
        event
        for event in events
        if event["type"] == "oracle.job.interpreter_evidence_attached"
        and event["job_id"] == queued.job_id
    )
    oracle_request = _oracle_request_for_job(updated, updated.request)
    combined = json.dumps(
        {
            "stored": updated.interpreter_evidence,
            "status": status,
            "event": attached,
            "request_updates": oracle_request.job_updates,
        },
        sort_keys=True,
    )
    queued_status = next(job for job in status["jobs"] if job["job_id"] == queued.job_id)

    assert "Bearer ***" in queued_status["latest_interpreter_evidence"]
    assert "Bearer ***" in oracle_request.oracle_text
    assert "raw-token" not in combined
    assert "sk_test_abcdefghijklmnopqrstuvwxyz" not in combined

    await manager.cancel(running.job_id)
    release.set()
    await manager.wait_for_idle()


@pytest.mark.asyncio
async def test_max_concurrent_four_starts_four_and_queues_fifth():
    started = []
    release = asyncio.Event()

    async def runner(job):
        started.append(job.job_id)
        await release.wait()
        return f"finished {job.job_id}"

    manager = OracleJobManager(max_concurrent=4, runner=runner)

    jobs = [await manager.submit(_request(f"job {idx}")) for idx in range(5)]
    await asyncio.sleep(0)

    states = [(await manager.get(job.job_id)).state for job in jobs]
    status = await manager.status_view()
    assert states == [
        OracleJobState.RUNNING,
        OracleJobState.RUNNING,
        OracleJobState.RUNNING,
        OracleJobState.RUNNING,
        OracleJobState.QUEUED,
    ]
    assert started == [
        "voice-oracle-001",
        "voice-oracle-002",
        "voice-oracle-003",
        "voice-oracle-004",
    ]
    assert [
        (job["job_id"], job["state"], job["ordinal"], job["ordinal_label"])
        for job in status["reflex"]["jobs"][:5]
    ] == [
        ("voice-oracle-001", "running", 1, "job one"),
        ("voice-oracle-002", "running", 2, "job two"),
        ("voice-oracle-003", "running", 3, "job three"),
        ("voice-oracle-004", "running", 4, "job four"),
        ("voice-oracle-005", "queued", 5, "job five"),
    ]

    release.set()
    await manager.wait_for_idle()
    assert started[-1] == "voice-oracle-005"
    assert (await manager.get(jobs[-1].job_id)).state == OracleJobState.COMPLETED


@pytest.mark.asyncio
async def test_reflex_status_prioritizes_active_jobs_over_terminal_history():
    started = []
    release = asyncio.Event()

    async def runner(job):
        if job.oracle_text.startswith("old"):
            return {"result_summary": f"done {job.job_id}"}
        started.append(job.job_id)
        await release.wait()
        return {"result_summary": f"done {job.job_id}"}

    manager = OracleJobManager(max_concurrent=4, runner=runner)

    for index in range(8):
        await manager.submit(_request(f"old completed {index}"))
    await manager.wait_for_idle()

    active_jobs = [await manager.submit(_request(f"active {index}")) for index in range(5)]
    await asyncio.sleep(0)

    status = await manager.status_view()
    assert [job["state"] for job in status["jobs"][:8]] == ["completed"] * 8
    assert [
        (job["job_id"], job["state"], job["ordinal"], job["ordinal_label"])
        for job in status["reflex"]["jobs"][:5]
    ] == [
        ("voice-oracle-009", "running", 1, "job one"),
        ("voice-oracle-010", "running", 2, "job two"),
        ("voice-oracle-011", "running", 3, "job three"),
        ("voice-oracle-012", "running", 4, "job four"),
        ("voice-oracle-013", "queued", 5, "job five"),
    ]
    assert [job["job_id"] for job in status["reflex"]["jobs"][:5]] == [
        job.job_id for job in active_jobs
    ]

    release.set()
    await manager.wait_for_idle()


@pytest.mark.asyncio
async def test_queue_limit_rejects_overflow():
    release = asyncio.Event()

    async def runner(job):
        await release.wait()
        return "done"

    manager = OracleJobManager(max_concurrent=1, queue_limit=1, runner=runner)
    await manager.submit(_request("running"))
    await manager.submit(_request("queued"))

    with pytest.raises(OracleJobQueueFullError):
        await manager.submit(_request("overflow"))

    release.set()
    await manager.wait_for_idle()


@pytest.mark.asyncio
async def test_overflow_policy_reject_rejects_at_capacity_with_queue_space():
    release = asyncio.Event()

    async def runner(job):
        await release.wait()
        return "done"

    manager = OracleJobManager(
        max_concurrent=1,
        queue_limit=16,
        overflow_policy="reject",
        runner=runner,
    )
    await manager.submit(_request("running"))

    with pytest.raises(OracleJobQueueFullError):
        await manager.submit(_request("should reject"))

    status = await manager.status_view()
    assert status["capacity"]["running"] == 1
    assert status["capacity"]["queued"] == 0
    assert len(status["jobs"]) == 1

    release.set()
    await manager.wait_for_idle()


@pytest.mark.asyncio
async def test_overflow_policy_reprioritize_requires_user_control_at_capacity():
    release = asyncio.Event()

    async def runner(job):
        await release.wait()
        return "done"

    manager = OracleJobManager(
        max_concurrent=1,
        queue_limit=16,
        overflow_policy="reprioritize",
        runner=runner,
    )
    await manager.submit(_request("running"))

    with pytest.raises(OracleJobReprioritizationRequiredError):
        await manager.submit(_request("needs reprioritization"))

    status = await manager.status_view()
    assert status["capacity"]["running"] == 1
    assert status["capacity"]["queued"] == 0
    assert len(status["jobs"]) == 1

    release.set()
    await manager.wait_for_idle()


@pytest.mark.asyncio
async def test_cancelling_queued_job_prevents_execution():
    started = []
    events = []
    release = asyncio.Event()

    async def runner(job):
        started.append(job.job_id)
        if job.job_id == "voice-oracle-001":
            await release.wait()
        return f"done {job.job_id}"

    manager = OracleJobManager(
        max_concurrent=1,
        runner=runner,
        event_callback=lambda event: events.append(event.to_status()),
    )
    first = await manager.submit(_request("first"))
    second = await manager.submit(_request("second"))
    await asyncio.sleep(0)

    await manager.cancel(second.job_id, reason="user cancelled second")
    release.set()
    await manager.wait_for_idle()

    assert started == [first.job_id]
    cancelled = await manager.get(second.job_id)
    assert cancelled.state == OracleJobState.CANCELLED
    assert cancelled.cancel_reason == "user cancelled second"
    queued_cancel_lifecycle = [
        event["type"]
        for event in events
        if event["job_id"] == second.job_id
        and event["type"] in {"oracle.job.cancel_requested", "oracle.job.cancelled"}
    ]
    assert queued_cancel_lifecycle == ["oracle.job.cancel_requested", "oracle.job.cancelled"]


@pytest.mark.asyncio
async def test_cancelling_running_job_calls_interrupt_and_ignores_late_result():
    interrupts = []

    async def runner(job):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            return "late result should not survive"

    manager = OracleJobManager(
        max_concurrent=1,
        runner=runner,
        interrupt_callback=lambda job, reason: interrupts.append((job.job_id, reason)),
    )
    job = await manager.submit(_request("slow"))
    await asyncio.sleep(0)

    await manager.cancel(job.job_id, reason="stop that")
    await manager.wait_for_idle()
    cancelled = await manager.get(job.job_id)

    assert interrupts == [(job.job_id, "stop that")]
    assert cancelled.state == OracleJobState.CANCELLED
    assert cancelled.result_summary == ""


@pytest.mark.asyncio
async def test_cancelling_running_job_still_cancels_when_interrupt_callback_fails():
    cancelled = asyncio.Event()
    interrupt_calls = []

    async def runner(job):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def interrupt(job, reason):
        interrupt_calls.append((job.job_id, reason))
        raise RuntimeError("provider interrupt failed with api_key=raw-token")

    manager = OracleJobManager(
        max_concurrent=1,
        runner=runner,
        interrupt_callback=interrupt,
    )
    job = await manager.submit(_request("slow"))
    await asyncio.sleep(0)

    await manager.cancel(job.job_id, reason="user changed topic")
    await manager.wait_for_idle()
    stored = await manager.get(job.job_id)
    status = await manager.status_view()

    assert interrupt_calls == [(job.job_id, "user changed topic")]
    assert cancelled.is_set()
    assert stored.state == OracleJobState.CANCELLED
    assert stored.cancel_reason == "user changed topic"
    assert status["capacity"]["running"] == 0
    assert status["capacity"]["queued"] == 0


@pytest.mark.asyncio
async def test_cancel_requested_job_keeps_capacity_until_worker_stops():
    started = []
    cancellation_entered = asyncio.Event()
    release_cancelled_worker = asyncio.Event()

    async def runner(job):
        started.append(job.job_id)
        if job.job_id == "voice-oracle-001":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_entered.set()
                await release_cancelled_worker.wait()
                raise
        return f"done {job.job_id}"

    manager = OracleJobManager(max_concurrent=1, runner=runner)
    first = await manager.submit(_request("first"))
    second = await manager.submit(_request("second"))
    await asyncio.sleep(0)

    await manager.cancel(first.job_id, reason="stop first")
    await asyncio.wait_for(cancellation_entered.wait(), timeout=1)
    await asyncio.sleep(0)

    assert started == [first.job_id]
    assert (await manager.get(first.job_id)).state == OracleJobState.CANCEL_REQUESTED
    assert (await manager.get(second.job_id)).state == OracleJobState.QUEUED
    status = await manager.status_view()
    assert status["capacity"] == {
        "active": 1,
        "running": 0,
        "max_concurrent": 1,
        "queued": 1,
        "queue_limit": 16,
        "waiting_for_approval": 0,
        "cancel_requested": 1,
    }

    release_cancelled_worker.set()
    await manager.wait_for_idle()

    assert started == [first.job_id, second.job_id]
    assert (await manager.get(first.job_id)).state == OracleJobState.CANCELLED
    assert (await manager.get(second.job_id)).state == OracleJobState.COMPLETED


@pytest.mark.asyncio
async def test_shutdown_forces_cancelled_state_when_worker_ignores_cancel():
    started = asyncio.Event()
    cancellation_entered = asyncio.Event()
    release_worker = asyncio.Event()

    async def runner(job):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_entered.set()
            await release_worker.wait()
            return "late result"

    manager = OracleJobManager(max_concurrent=1, runner=runner)
    job = await manager.submit(_request("slow"))
    await asyncio.wait_for(started.wait(), timeout=1)

    drained = await manager.shutdown(reason="session closed", timeout_seconds=0.01)
    status = await manager.status_view()

    assert drained is False
    assert (await manager.get(job.job_id)).state == OracleJobState.CANCELLED
    assert (await manager.get(job.job_id)).cancel_reason == "session closed"
    assert status["capacity"] == {
        "active": 0,
        "running": 0,
        "max_concurrent": 1,
        "queued": 0,
        "queue_limit": 16,
        "waiting_for_approval": 0,
        "cancel_requested": 0,
    }
    assert cancellation_entered.is_set()

    release_worker.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_failed_job_records_error_and_starts_next():
    async def runner(job):
        if job.job_id == "voice-oracle-001":
            raise RuntimeError("provider exploded")
        return "second done"

    manager = OracleJobManager(max_concurrent=1, runner=runner)
    first = await manager.submit(_request("first"))
    second = await manager.submit(_request("second"))
    await manager.wait_for_idle()

    failed = await manager.get(first.job_id)
    completed = await manager.get(second.job_id)
    assert failed.state == OracleJobState.FAILED
    assert failed.error == "provider exploded"
    assert completed.state == OracleJobState.COMPLETED
    assert completed.result_summary == "second done"


@pytest.mark.asyncio
async def test_failed_job_sanitizes_error_for_status_events_and_audit(tmp_path):
    events = []

    async def runner(job):
        raise RuntimeError(
            "provider failed Bearer raw-token token=sk-secret "
            "at https://user:pass@example.invalid/v1?api_key=raw"
        )

    manager = OracleJobManager(
        max_concurrent=1,
        runner=runner,
        event_callback=events.append,
        audit_ledger_path=tmp_path / "audit.jsonl",
    )
    job = await manager.submit(_request("secret-bearing failure"))
    await manager.wait_for_idle()

    failed = await manager.get(job.job_id)
    status = await manager.status_view()
    failed_event = next(event for event in events if event.type.value == "oracle.job.failed")
    audit_lines = (tmp_path / "audit.jsonl").read_text().splitlines()
    audit_failed = next(
        json.loads(line)
        for line in audit_lines
        if json.loads(line)["event_type"] == "oracle.job.failed"
    )

    combined = json.dumps(
        {
            "stored": failed.error,
            "status": status,
            "event": failed_event.to_status(),
            "audit": audit_failed,
        },
        sort_keys=True,
    )
    assert failed.state == OracleJobState.FAILED
    assert "provider failed" in failed.error
    assert "Bearer ***" in failed.error
    assert "token=***" in failed.error
    assert "https://***@example.invalid/v1" in failed.error
    assert "raw-token" not in combined
    assert "sk-secret" not in combined
    assert "user:pass" not in combined
    assert "api_key=raw" not in combined


@pytest.mark.asyncio
async def test_status_view_reports_capacity_and_redacts_raw_metadata():
    release = asyncio.Event()

    async def runner(job):
        await release.wait()
        return {"result_summary": "safe summary", "tool_trace": "secret trace"}

    manager = OracleJobManager(max_concurrent=1, runner=runner)
    await manager.submit(_request("inspect deployment", route=KameRoute.DEFER))
    await manager.submit(_request("check stripe"))
    await asyncio.sleep(0)

    status = await manager.status_view()
    assert status["capacity"] == {
        "active": 1,
        "running": 1,
        "max_concurrent": 1,
        "queued": 1,
        "queue_limit": 16,
        "waiting_for_approval": 0,
        "cancel_requested": 0,
    }
    assert status["jobs"][0]["spoken_status"] == "I'm handling inspect deployment."
    assert "metadata" not in status["jobs"][0]
    assert "oracle_text" not in status["jobs"][0]
    assert status["reflex"]["capacity"] == status["capacity"]
    assert status["reflex"]["jobs"][0] == {
        "job_id": "voice-oracle-001",
        "state": "running",
        "ordinal": 1,
        "ordinal_label": "job one",
        "priority": "normal",
        "spoken_status": "I'm handling inspect deployment.",
    }
    assert status["reflex"]["jobs"][1] == {
        "job_id": "voice-oracle-002",
        "state": "queued",
        "ordinal": 2,
        "ordinal_label": "job two",
        "priority": "normal",
        "spoken_status": "I'm handling check stripe.",
    }
    reflex_blob = json.dumps(status["reflex"], sort_keys=True)
    assert "metadata" not in reflex_blob
    assert "oracle_text" not in reflex_blob
    assert "latest_interpreter_evidence" not in reflex_blob
    assert "transcript_hypotheses" not in reflex_blob
    assert "speaker" not in reflex_blob
    assert "channel" not in reflex_blob

    release.set()
    await manager.wait_for_idle()
    done = await manager.status_view()
    assert done["jobs"][0]["result_summary"] == "safe summary"
    assert "secret trace" not in str(done)
    assert done["reflex"]["jobs"][0]["result_summary"] == "safe summary"
    assert "result_text" not in str(done["reflex"])
    assert "secret trace" not in str(done["reflex"])


@pytest.mark.asyncio
async def test_completed_event_preserves_full_result_without_bloating_status():
    events = []
    full_text = "First sentence.\n\n" + ("detail " * 500)

    async def runner(job):
        return {
            "result_summary": "Short spoken summary.",
            "result_text": full_text,
        }

    manager = OracleJobManager(
        max_concurrent=1,
        runner=runner,
        event_callback=lambda event: events.append(event.to_status()),
    )
    job = await manager.submit(_request("write a detailed report"))
    await manager.wait_for_idle()

    completed = await manager.get(job.job_id)
    status = await manager.status_view()
    completed_event = next(event for event in events if event["type"] == "oracle.job.completed")

    assert completed.result_summary == "Short spoken summary."
    assert completed.result_text == full_text
    assert completed_event["payload"]["result_summary"] == "Short spoken summary."
    assert completed_event["payload"]["result_text"] == full_text
    assert completed_event["payload"]["result_text_chars"] == len(full_text)
    assert status["jobs"][0]["result_summary"] == "Short spoken summary."
    assert "result_text" not in status["jobs"][0]
    assert full_text not in str(status)
    assert status["reflex"]["jobs"][0]["result_summary"] == "Short spoken summary."
    assert "result_text" not in str(status["reflex"])
    assert full_text not in str(status["reflex"])


@pytest.mark.asyncio
async def test_completed_result_redacts_hidden_reasoning_from_status_and_event():
    events = []
    visible_text = "Visible details.\nSecond line."

    async def runner(job):
        return {
            "result_summary": "<think>private scratchpad</think> Visible summary.",
            "result_text": f"<think>private scratchpad</think>\n{visible_text}",
        }

    manager = OracleJobManager(
        max_concurrent=1,
        runner=runner,
        event_callback=lambda event: events.append(event.to_status()),
    )
    job = await manager.submit(_request("write a visible report"))
    await manager.wait_for_idle()

    completed = await manager.get(job.job_id)
    status = await manager.status_view()
    completed_event = next(event for event in events if event["type"] == "oracle.job.completed")

    assert completed.result_summary == "Visible summary."
    assert completed.result_text == visible_text
    assert status["jobs"][0]["result_summary"] == "Visible summary."
    assert completed_event["payload"]["result_summary"] == "Visible summary."
    assert completed_event["payload"]["result_text"] == visible_text
    assert completed_event["payload"]["result_text_chars"] == len(visible_text)
    assert "private scratchpad" not in str(status)
    assert "private scratchpad" not in str(completed_event)


@pytest.mark.asyncio
async def test_completed_result_redacts_orphan_close_reasoning_trace():
    async def runner(job):
        return {
            "result_summary": (
                'We need to reply with exactly "ready". No extra text. '
                "Let's do that.</think>\nready"
            )
        }

    manager = OracleJobManager(max_concurrent=1, runner=runner)
    job = await manager.submit(_request("say ready"))
    await manager.wait_for_idle()

    completed = await manager.get(job.job_id)
    status = await manager.status_view()

    assert completed.result_summary == "ready"
    assert status["jobs"][0]["result_summary"] == "ready"
    assert "No extra text" not in str(status)


@pytest.mark.asyncio
async def test_completed_result_redacts_secret_like_text_from_status_and_events():
    events = []
    test_secret = "sk" + "_test_" + "abcdefghijklmnopqrstuvwxyz"
    live_secret = "sk" + "_live_" + "abcdefghijklmnopqrstuvwxyz"

    async def runner(job):
        return {
            "result_summary": (
                f"Created provider credential {test_secret} "
                "with Authorization: Bearer raw-token"
            ),
            "result_text": (
                "Full provider result includes provider_token=raw-provider-token "
                f"and {live_secret} with max_tokens=8192"
            ),
        }

    manager = OracleJobManager(
        max_concurrent=1,
        runner=runner,
        event_callback=lambda event: events.append(event.to_status()),
    )
    job = await manager.submit(_request("provision voice provider"))
    await manager.wait_for_idle()

    completed = await manager.get(job.job_id)
    status = await manager.status_view()
    completed_event = next(event for event in events if event["type"] == "oracle.job.completed")
    combined = json.dumps(
        {
            "stored_summary": completed.result_summary,
            "stored_text": completed.result_text,
            "status": status,
            "event": completed_event,
        },
        sort_keys=True,
    )

    assert "Authorization: Bearer ***" in combined
    assert "sk_tes" in combined
    assert "sk_liv" in combined
    assert "max_tokens=8192" in combined
    assert "raw-token" not in combined
    assert "raw-provider-token" not in combined
    assert test_secret not in combined
    assert live_secret not in combined


@pytest.mark.asyncio
async def test_waiting_for_approval_holds_capacity_and_emits_redacted_event():
    events = []
    release = asyncio.Event()

    async def runner(job):
        await release.wait()
        return "done"

    manager = OracleJobManager(
        max_concurrent=1,
        runner=runner,
        event_callback=lambda event: events.append(event.to_status()),
    )
    first = await manager.submit(_request("buy service credits"))
    second = await manager.submit(_request("check logs"))
    await asyncio.sleep(0)

    waiting = await manager.mark_waiting_for_approval(
        first.job_id,
        reason="Stripe Link spend requires approval",
        approval={
            "approval_id": "approval-123",
            "tool_name": "stripe_link_purchase",
            "secret": "do not expose",
            "arguments": {"card": "do not expose"},
            "latest_interpreter_evidence": (
                "interpreter evidence: transcript=buy twenty dollars of phone credits; "
                "intent=prepare Stripe approval"
            ),
            "latest_interpreter_evidence_source": "gemma_interpreter",
            "interpreter_evidence_count": 1,
            "interpreter_evidence_late": True,
        },
    )
    status = await manager.status_view()

    assert waiting.state == OracleJobState.WAITING_FOR_APPROVAL
    assert (await manager.get(second.job_id)).state == OracleJobState.QUEUED
    assert status["capacity"] == {
        "active": 1,
        "running": 0,
        "max_concurrent": 1,
        "queued": 1,
        "queue_limit": 16,
        "waiting_for_approval": 1,
        "cancel_requested": 0,
    }
    assert status["jobs"][0]["approval_reason"] == "Stripe Link spend requires approval"
    assert status["jobs"][0]["approval"] == {
        "approval_id": "approval-123",
        "tool_name": "stripe_link_purchase",
        "latest_interpreter_evidence": (
            "interpreter evidence: transcript=buy twenty dollars of phone credits; "
            "intent=prepare Stripe approval"
        ),
        "latest_interpreter_evidence_source": "gemma_interpreter",
        "interpreter_evidence_count": 1,
        "interpreter_evidence_late": True,
    }
    assert status["reflex"]["jobs"][0]["approval_reason"] == "Stripe Link spend requires approval"
    assert "approval" not in status["reflex"]["jobs"][0]
    assert "do not expose" not in str(status)
    waiting_event = next(event for event in events if event["type"] == "oracle.job.waiting_for_approval")
    assert waiting_event["state"] == "waiting_for_approval"
    assert waiting_event["payload"]["approval_reason"] == "Stripe Link spend requires approval"
    assert waiting_event["payload"]["approval"] == {
        "approval_id": "approval-123",
        "tool_name": "stripe_link_purchase",
        "latest_interpreter_evidence": (
            "interpreter evidence: transcript=buy twenty dollars of phone credits; "
            "intent=prepare Stripe approval"
        ),
        "latest_interpreter_evidence_source": "gemma_interpreter",
        "interpreter_evidence_count": 1,
        "interpreter_evidence_late": True,
    }
    assert "do not expose" not in str(waiting_event)

    release.set()
    await manager.wait_for_idle()


@pytest.mark.asyncio
async def test_cancelling_waiting_for_approval_keeps_capacity_until_worker_stops_and_drops_late_result(tmp_path):
    ledger_path = tmp_path / "voiceops-oracle-jobs.jsonl"
    started = []
    cancellation_entered = asyncio.Event()
    release_cancelled_worker = asyncio.Event()

    async def runner(job):
        started.append(job.job_id)
        if job.oracle_text == "buy service credits":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_entered.set()
                await release_cancelled_worker.wait()
                return {
                    "result_summary": "late result with raw-token",
                    "result_text": "late result should not be recorded",
                }
        return f"done {job.job_id}"

    manager = OracleJobManager(
        max_concurrent=1,
        runner=runner,
        audit_ledger_path=ledger_path,
    )
    first = await manager.submit(_request("buy service credits"))
    second = await manager.submit(_request("check logs"))
    await asyncio.sleep(0)

    await manager.mark_waiting_for_approval(
        first.job_id,
        reason="Stripe Link spend requires approval",
        approval={"approval_id": "approval-123", "tool_name": "stripe_link_purchase"},
    )
    await manager.cancel(first.job_id, reason="approval denied")
    await asyncio.wait_for(cancellation_entered.wait(), timeout=1)
    await asyncio.sleep(0)

    assert started == [first.job_id]
    assert (await manager.get(first.job_id)).state == OracleJobState.CANCEL_REQUESTED
    assert (await manager.get(second.job_id)).state == OracleJobState.QUEUED
    assert (await manager.status_view())["capacity"] == {
        "active": 1,
        "running": 0,
        "max_concurrent": 1,
        "queued": 1,
        "queue_limit": 16,
        "waiting_for_approval": 0,
        "cancel_requested": 1,
    }

    release_cancelled_worker.set()
    await manager.wait_for_idle()

    first_stored = await manager.get(first.job_id)
    second_stored = await manager.get(second.job_id)
    assert started == [first.job_id, second.job_id]
    assert first_stored.state == OracleJobState.CANCELLED
    assert first_stored.result_summary == ""
    assert first_stored.result_text == ""
    assert second_stored.state == OracleJobState.COMPLETED

    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    first_events = [row["event_type"] for row in rows if row["job_id"] == first.job_id]
    assert first_events == [
        "oracle.job.accepted",
        "oracle.job.started",
        "oracle.job.waiting_for_approval",
        "oracle.job.cancel_requested",
        "oracle.job.result_suppressed",
        "oracle.job.cancelled",
    ]
    assert all(
        set(row["capacity"]) == {
            "active",
            "running",
            "max_concurrent",
            "queued",
            "queue_limit",
            "waiting_for_approval",
            "cancel_requested",
        }
        for row in rows
    )
    waiting = next(row for row in rows if row["event_type"] == "oracle.job.waiting_for_approval")
    assert waiting["capacity"] == {
        "active": 1,
        "running": 0,
        "max_concurrent": 1,
        "queued": 1,
        "queue_limit": 16,
        "waiting_for_approval": 1,
        "cancel_requested": 0,
    }
    cancel_requested = next(row for row in rows if row["event_type"] == "oracle.job.cancel_requested")
    assert cancel_requested["capacity"] == {
        "active": 1,
        "running": 0,
        "max_concurrent": 1,
        "queued": 1,
        "queue_limit": 16,
        "waiting_for_approval": 0,
        "cancel_requested": 1,
    }
    assert "oracle.job.completed" not in first_events
    suppressed = next(row for row in rows if row["event_type"] == "oracle.job.result_suppressed")
    assert suppressed["payload"]["suppression_reason"] == "cancelled_job_returned_result"
    assert suppressed["payload"]["result_suppressed"] is True
    assert suppressed["payload"]["suppressed_result_present"] is True
    assert "result_summary" not in suppressed["payload"]
    assert "result_text" not in suppressed["payload"]
    assert "late result" not in str(rows)
    assert "raw-token" not in str(rows)


@pytest.mark.asyncio
async def test_audit_ledger_path_records_redacted_lifecycle_events(tmp_path):
    ledger_path = tmp_path / "voiceops-oracle-jobs.jsonl"
    release = asyncio.Event()

    async def runner(job):
        await release.wait()
        return {
            "result_summary": "Bought service credits after approval.",
            "result_text": "Long private result " * 50,
            "tool_trace": "private trace",
        }

    manager = OracleJobManager(
        max_concurrent=1,
        runner=runner,
        audit_ledger_path=ledger_path,
    )
    job = await manager.submit(_request("buy service credits"))
    await asyncio.sleep(0)
    await manager.add_update(
        job.job_id,
        text=(
            "include redacted receipt reference with Authorization: Bearer raw-token "
            "and sk_test_abcdefghijklmnopqrstuvwxyz"
        ),
        source="discord_voice",
        update_type="clarification",
    )
    await manager.mark_waiting_for_approval(
        job.job_id,
        reason="Stripe Link spend requires approval",
        approval={
            "approval_id": "approval-123",
            "tool_name": "stripe_link_purchase",
            "secret": "do not expose",
            "arguments": {"card": "do not expose"},
        },
    )
    await manager.mark_running(job.job_id)
    release.set()
    await manager.wait_for_idle()

    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [row["event_type"] for row in rows] == [
        "oracle.job.accepted",
        "oracle.job.started",
        "oracle.job.progress",
        "oracle.job.waiting_for_approval",
        "oracle.job.started",
        "oracle.job.completed",
    ]
    assert all(row["schema_version"] == "voiceops.oracle_job_audit_event.v1" for row in rows)
    assert all(row["action"] == "oracle_job_event" for row in rows)
    waiting = next(row for row in rows if row["event_type"] == "oracle.job.waiting_for_approval")
    progress = next(row for row in rows if row["event_type"] == "oracle.job.progress")
    assert progress["payload"]["operation"] == "update"
    assert "include redacted receipt reference" in progress["payload"]["latest_update"]
    assert "Bearer ***" in progress["payload"]["latest_update"]
    assert "sk_tes" in progress["payload"]["latest_update"]
    assert waiting["payload"]["approval"] == {
        "approval_id": "approval-123",
        "tool_name": "stripe_link_purchase",
    }
    completed = rows[-1]
    assert completed["payload"]["result_summary"] == "Bought service credits after approval."
    assert completed["payload"]["result_text_chars"] == len("Long private result " * 50)
    assert "result_text" not in completed["payload"]
    assert "metadata" not in rows[0]["payload"]
    assert "oracle_text" not in rows[0]["payload"]
    assert "do not expose" not in str(rows)
    assert "Long private result" not in str(rows)
    assert "raw-token" not in str(rows)
    assert "sk_test_abcdefghijklmnopqrstuvwxyz" not in str(rows)


@pytest.mark.asyncio
async def test_audit_ledger_force_redacts_scalar_payload_fields(tmp_path):
    ledger_path = tmp_path / "voiceops-oracle-jobs.jsonl"
    release = asyncio.Event()
    test_secret = "sk_test_" + "abcdefghijklmnopqrstuvwxyz"
    approval_secret = "sk_test_" + "zyxwvutsrqponmlkjihgfedcba"
    approval_summary_secret = "sk_test_" + "qwertyuiopasdfghjklzxcvbnm"
    live_secret = "sk_live_" + "abcdefghijklmnopqrstuvwxyz"

    async def runner(job):
        await release.wait()
        return {
            "result_summary": (
                f"Created provider credential {test_secret} "
                "with Authorization: Bearer raw-token"
            ),
            "result_text": (
                "Full result includes provider_token=raw-provider-token "
                f"and {live_secret}"
            ),
        }

    manager = OracleJobManager(
        max_concurrent=1,
        runner=runner,
        audit_ledger_path=ledger_path,
    )
    job = await manager.submit(_request("provision voice provider"))
    await asyncio.sleep(0)
    await manager.mark_waiting_for_approval(
        job.job_id,
        reason=(
            "Approve Stripe spend with Authorization: Bearer approval-token "
            f"and {approval_secret}"
        ),
        approval={
            "approval_id": "approval-123",
            "tool_name": "stripe_link_purchase",
            "summary": f"Charge uses {approval_summary_secret}",
        },
    )
    await manager.mark_running(job.job_id)
    release.set()
    await manager.wait_for_idle()

    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    waiting = next(row for row in rows if row["event_type"] == "oracle.job.waiting_for_approval")
    completed = next(row for row in rows if row["event_type"] == "oracle.job.completed")
    combined = json.dumps(rows, sort_keys=True)

    assert "approval_reason" in waiting["payload"]
    assert "result_summary" in completed["payload"]
    assert "result_text_chars" in completed["payload"]
    assert "result_text" not in completed["payload"]
    assert "Authorization: Bearer ***" in combined
    assert "sk_tes" in combined
    assert "raw-token" not in combined
    assert "approval-token" not in combined
    assert "raw-provider-token" not in combined
    assert test_secret not in combined
    assert approval_secret not in combined
    assert approval_summary_secret not in combined
    assert live_secret not in combined
