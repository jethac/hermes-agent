"""Headless smoke for async KAME reflex/oracle voice jobs.

This exercises the in-process realtime voice engine with fake oracle workers.
It is intentionally provider-free: no Discord, sidecar, Spark, STT, or TTS
service is required.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable

from agent.realtime_voice import RealtimeVoiceEngineKind, RealtimeVoiceSessionConfig, VoiceEvent, VoiceEventType
from agent.realtime_voice_session import RealtimeVoiceSession
from agent.realtime_voice_text_engine import KameInterfaceOracleEngine


APPROVAL_SECRET_CANARY = "secret test value must not leak"


class SmokeOracle:
    def __init__(self) -> None:
        self.running = 0
        self.max_running = 0
        self.requests: list[Any] = []
        self.releases: dict[str, asyncio.Event] = {}
        self.late_cancelled_output_attempted = False

    async def stream_answer_for_request(self, request: Any):
        self.requests.append(request)
        key = str(request.intent or request.oracle_text)
        self.releases.setdefault(key, asyncio.Event())
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        try:
            if key == "Prepare approval spend":
                yield {
                    "event": "tool_call",
                    "tool_name": "stripe_link_purchase",
                    "tool_call_id": "call-approve-smoke",
                    "approval_required": True,
                    "approval_id": "approval-smoke-123",
                    "approval_reason": "Stripe Link spend requires approval",
                    "arguments": {"amount": 200, "card": APPROVAL_SECRET_CANARY},
                }
                await self.releases[key].wait()
                yield {
                    "event": "tool_result",
                    "tool_name": "stripe_link_purchase",
                    "tool_call_id": "call-approve-smoke",
                    "approval_id": "approval-smoke-123",
                    "result": {"approved": True},
                }
                yield "Approval smoke cleared."
                return
            if key == "Fail smoke task":
                raise RuntimeError("smoke oracle failure")
            if key == "Explain verbose plan":
                yield "First sentence. "
                yield "Second sentence. "
                yield "Third sentence."
                return
            try:
                await self.releases[key].wait()
            except asyncio.CancelledError:
                if key != "Run smoke task 3":
                    raise
                await self.releases[key].wait()
                self.late_cancelled_output_attempted = True
            yield f"Finished {key}."
        finally:
            self.running -= 1

    def release(self, intent: str) -> None:
        self.releases.setdefault(intent, asyncio.Event()).set()


class SmokeEngine(KameInterfaceOracleEngine):
    def __init__(self, *, oracle: SmokeOracle) -> None:
        super().__init__(oracle=oracle)
        self.spoken: list[str] = []

    async def _speak_chunk(self, text: str, playback_generation: int) -> None:
        self.spoken.append(text)
        await asyncio.sleep(0)


class EventRecorder:
    def __init__(self, engine: KameInterfaceOracleEngine) -> None:
        self.engine = engine
        self.events: list[VoiceEvent] = []
        self._condition = asyncio.Condition()

    async def run(self) -> None:
        async for event in self.engine.events():
            async with self._condition:
                self.events.append(event)
                self._condition.notify_all()

    async def wait_for(
        self,
        predicate: Callable[[list[VoiceEvent]], bool],
        *,
        timeout_seconds: float = 2.0,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        async with self._condition:
            while not predicate(self.events):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for realtime voice smoke event")
                await asyncio.wait_for(self._condition.wait(), timeout=remaining)


async def run_smoke() -> dict[str, Any]:
    oracle = SmokeOracle()
    engine = SmokeEngine(oracle=oracle)
    recorder = EventRecorder(engine)
    await engine.start(
        RealtimeVoiceSessionConfig(
            session_id="voice-smoke-async-oracle",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            oracle_jobs={
                "enabled": True,
                "max_concurrent": 4,
                "queue_limit": 4,
                "speak_terminal_results": True,
            },
            metadata={"transport": "smoke"},
        )
    )
    collector = asyncio.create_task(recorder.run())
    sequence = 0

    async def send(payload: dict[str, Any]) -> None:
        nonlocal sequence
        sequence += 1
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-smoke-async-oracle",
                sequence=sequence,
                payload={**payload, "end_of_utterance": True},
            )
        )

    for index in range(1, 6):
        await send(
            {
                "transcript": f"run smoke task {index}",
                "intent": f"Run smoke task {index}",
                "intent_source": "smoke_reflex",
                "route": "defer",
                "interface_already_said": f"Starting smoke task {index}.",
            }
        )

    await recorder.wait_for(
        lambda events: (
            sum(event.type == VoiceEventType.ORACLE_JOB_STARTED for event in events) == 4
            and sum(event.type == VoiceEventType.ORACLE_JOB_QUEUED for event in events) == 1
        )
    )

    await send(
        {
            "transcript": "what are you working on",
            "intent": "What are you working on?",
            "intent_source": "smoke_reflex",
            "route": "local",
            "local_reply": "Let me check.",
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ASSISTANT_COMMIT
            and str(event.payload.get("text") or "").startswith("Oracle jobs: 4 running out of 4, 1 queued.")
            for event in events
        )
    )

    await send(
        {
            "transcript": "can you hear me",
            "intent": "The user is checking whether Hermes can hear them.",
            "intent_source": "smoke_reflex",
            "route": "local",
            "local_reply": "Yes, I can hear you.",
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ASSISTANT_COMMIT and event.payload.get("local_reply")
            and event.payload.get("text") == "Yes, I can hear you."
            for event in events
        )
    )

    task_5_queued = next(
        event
        for event in recorder.events
        if event.type == VoiceEventType.ORACLE_JOB_QUEUED
        and event.payload.get("intent") == "Run smoke task 5"
    )
    sequence += 1
    await engine.receive_event(
        VoiceEvent(
            type=VoiceEventType.INTERFACE_ORACLE_UPDATE,
            session_id="voice-smoke-async-oracle",
            sequence=sequence,
            payload={
                "job_id": task_5_queued.payload["job_id"],
                "priority": "high",
                "update_text": "include smoke update context",
                "reason": "smoke queued job update",
            },
        )
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE
            and event.payload.get("job_id") == task_5_queued.payload["job_id"]
            and event.payload.get("priority") == "high"
            and event.payload.get("update_count") == 1
            for event in events
        )
    )

    task_3_started = next(
        event
        for event in recorder.events
        if event.type == VoiceEventType.ORACLE_JOB_STARTED
        and event.payload.get("intent") == "Run smoke task 3"
    )
    sequence += 1
    await engine.receive_event(
        VoiceEvent(
            type=VoiceEventType.INTERFACE_ORACLE_CANCEL,
            session_id="voice-smoke-async-oracle",
            sequence=sequence,
            payload={
                "job_id": task_3_started.payload["job_id"],
                "reason": "smoke cancellation",
            },
        )
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_CANCEL_REQUESTED
            and event.payload.get("job_id") == task_3_started.payload["job_id"]
            for event in events
        )
    )
    oracle.release("Run smoke task 3")
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_CANCELLED
            and event.payload.get("job_id") == task_3_started.payload["job_id"]
            for event in events
        )
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_STARTED
            and event.payload.get("intent") == "Run smoke task 5"
            for event in events
        )
    )

    for index in (1, 2, 4, 5):
        oracle.release(f"Run smoke task {index}")

    await recorder.wait_for(
        lambda events: sum(event.type == VoiceEventType.ORACLE_JOB_COMPLETED for event in events) == 4
    )

    await send(
        {
            "transcript": "prepare approval spend",
            "intent": "Prepare approval spend",
            "intent_source": "smoke_reflex",
            "route": "defer",
            "interface_already_said": "Preparing spend approval.",
        }
    )
    await recorder.wait_for(
        lambda events: any(event.type == VoiceEventType.ORACLE_JOB_WAITING_FOR_APPROVAL for event in events)
    )
    await send(
        {
            "transcript": "what are you waiting on",
            "intent": "What are you working on?",
            "intent_source": "smoke_reflex",
            "route": "local",
            "local_reply": "Let me check.",
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ASSISTANT_COMMIT
            and "waiting_for_approval: Preparing spend approval." in str(event.payload.get("text") or "")
            for event in events
        )
    )
    oracle.release("Prepare approval spend")
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_COMPLETED
            and event.payload.get("intent") == "Prepare approval spend"
            for event in events
        )
    )
    await send(
        {
            "transcript": "fail smoke task",
            "intent": "Fail smoke task",
            "intent_source": "smoke_reflex",
            "route": "defer",
            "interface_already_said": "Testing failure handling.",
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ASSISTANT_COMMIT
            and event.payload.get("oracle_job_failed")
            and "smoke oracle failure" in str(event.payload.get("text") or "")
            for event in events
        )
    )
    await send(
        {
            "transcript": "can you still hear me",
            "intent": "The user is checking whether Hermes is still live after a failed oracle job.",
            "intent_source": "smoke_reflex",
            "route": "local",
            "local_reply": "Still listening.",
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ASSISTANT_COMMIT
            and event.payload.get("local_reply")
            and event.payload.get("text") == "Still listening."
            for event in events
        )
    )
    await send(
        {
            "transcript": "explain verbose plan",
            "intent": "Explain verbose plan",
            "intent_source": "smoke_reflex",
            "route": "defer",
            "interface_already_said": "Working on the plan.",
            "max_spoken_sentences": 1,
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ASSISTANT_COMMIT
            and event.payload.get("oracle_job_result")
            and event.payload.get("text") == "First sentence."
            and event.payload.get("voice_response_truncated") is True
            for event in events
        )
    )
    await engine.close()
    collector.cancel()
    try:
        await collector
    except asyncio.CancelledError:
        pass

    started = [event for event in recorder.events if event.type == VoiceEventType.ORACLE_JOB_STARTED]
    queued = [event for event in recorder.events if event.type == VoiceEventType.ORACLE_JOB_QUEUED]
    completed = [event for event in recorder.events if event.type == VoiceEventType.ORACLE_JOB_COMPLETED]
    failed = [event for event in recorder.events if event.type == VoiceEventType.ORACLE_JOB_FAILED]
    cancelled = [event for event in recorder.events if event.type == VoiceEventType.ORACLE_JOB_CANCELLED]
    active_job_ids: set[str] = set()
    scheduler_max_running = 0
    for event in recorder.events:
        job_id = str(event.payload.get("job_id") or "")
        if not job_id:
            continue
        if event.type == VoiceEventType.ORACLE_JOB_STARTED:
            active_job_ids.add(job_id)
            scheduler_max_running = max(scheduler_max_running, len(active_job_ids))
        elif event.type in {
            VoiceEventType.ORACLE_JOB_COMPLETED,
            VoiceEventType.ORACLE_JOB_FAILED,
            VoiceEventType.ORACLE_JOB_CANCELLED,
        }:
            active_job_ids.discard(job_id)
    local_commits = [
        event
        for event in recorder.events
        if event.type == VoiceEventType.ASSISTANT_COMMIT and event.payload.get("local_reply")
    ]
    status_commits = [
        event
        for event in recorder.events
        if event.type == VoiceEventType.ASSISTANT_COMMIT
        and str(event.payload.get("text") or "").startswith("Oracle jobs:")
    ]
    running_status_commits = [
        event
        for event in status_commits
        if str(event.payload.get("text") or "").startswith("Oracle jobs: 4 running out of 4, 1 queued.")
    ]
    approval_waiting = [
        event for event in recorder.events if event.type == VoiceEventType.ORACLE_JOB_WAITING_FOR_APPROVAL
    ]
    approval_tool_progress = [
        event
        for event in recorder.events
        if event.type == VoiceEventType.ORACLE_JOB_PROGRESS
        and event.payload.get("phase") == "tool"
        and event.payload.get("tool_event", {}).get("approval_required") is True
    ]
    approval_status_commits = [
        event
        for event in recorder.events
        if event.type == VoiceEventType.ASSISTANT_COMMIT
        and "waiting_for_approval: Preparing spend approval." in str(event.payload.get("text") or "")
    ]
    approval_completed = any(
        event.type == VoiceEventType.ORACLE_JOB_COMPLETED
        and event.payload.get("intent") == "Prepare approval spend"
        and event.payload.get("result_summary") == "Approval smoke cleared."
        for event in recorder.events
    )
    approval_payload_redacted = bool(approval_waiting) and "secret test value" not in str(
        approval_waiting[-1].payload
    )
    failure_commits = [
        event
        for event in recorder.events
        if event.type == VoiceEventType.ASSISTANT_COMMIT
        and event.payload.get("oracle_job_failed")
        and "smoke oracle failure" in str(event.payload.get("text") or "")
    ]
    failure_spoken = any("smoke oracle failure" in text for text in engine.spoken)
    session_survived_failure = any(
        event.type == VoiceEventType.ASSISTANT_COMMIT
        and event.payload.get("local_reply")
        and event.payload.get("text") == "Still listening."
        for event in recorder.events
    )
    queued_job_update_observed = any(
        event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE
        and event.payload.get("job_id") == task_5_queued.payload["job_id"]
        and event.payload.get("priority") == "high"
        and event.payload.get("update_count") == 1
        for event in recorder.events
    )
    queued_update_started_with_priority = any(
        event.type == VoiceEventType.ORACLE_JOB_STARTED
        and event.payload.get("job_id") == task_5_queued.payload["job_id"]
        and event.payload.get("priority") == "high"
        for event in recorder.events
    )
    queued_update_reached_oracle = any(
        str(getattr(request, "intent", "")) == "Run smoke task 5"
        and "include smoke update context" in tuple(getattr(request, "job_updates", ()))
        for request in oracle.requests
    )
    cancelled_job_id = str(task_3_started.payload["job_id"])
    fifth_job_id = next(
        (
            str(event.payload.get("job_id") or "")
            for event in queued
            if event.payload.get("intent") == "Run smoke task 5"
        ),
        "",
    )
    fifth_job_started_after_capacity_freed = any(
        event.type == VoiceEventType.ORACLE_JOB_STARTED
        and event.payload.get("job_id") == fifth_job_id
        and event.payload.get("intent") == "Run smoke task 5"
        for event in recorder.events
    )
    cancelled_result_spoken = any("Finished Run smoke task 3" in text for text in engine.spoken)
    cancelled_result_committed = any(
        event.type == VoiceEventType.ASSISTANT_COMMIT
        and "Finished Run smoke task 3" in str(event.payload.get("text") or "")
        for event in recorder.events
    )
    cancelled_result_progress_leaked = any(
        event.type == VoiceEventType.ORACLE_JOB_PROGRESS
        and event.payload.get("job_id") == cancelled_job_id
        and (
            "Finished Run smoke task 3" in str(event.payload.get("delta") or "")
            or "Finished Run smoke task 3" in str(event.payload.get("text") or "")
        )
        for event in recorder.events
    )
    durable_session = RealtimeVoiceSession(
        RealtimeVoiceSessionConfig(session_id="voice-smoke-async-oracle"),
        engine=SmokeEngine(oracle=SmokeOracle()),
    )
    for event in recorder.events:
        durable_session._apply_server_event(event)
    durable_records = durable_session.durable_oracle_records()
    cancelled_result_durable_completed = any(
        record.get("type") == VoiceEventType.ORACLE_JOB_COMPLETED.value
        and record.get("payload", {}).get("job_id") == cancelled_job_id
        for record in durable_records
    )
    cancelled_result_durable_text = any(
        record.get("payload", {}).get("job_id") == cancelled_job_id
        and "Finished Run smoke task 3" in str(record.get("payload", {}))
        for record in durable_records
    )
    durable_cancelled_record_present = any(
        record.get("type") == VoiceEventType.ORACLE_JOB_CANCELLED.value
        and record.get("payload", {}).get("job_id") == cancelled_job_id
        for record in durable_records
    )
    durable_completed_jobs = sum(
        record.get("type") == VoiceEventType.ORACLE_JOB_COMPLETED.value
        for record in durable_records
    )
    durable_failed_record_present = any(
        record.get("type") == VoiceEventType.ORACLE_JOB_FAILED.value
        and "smoke oracle failure" in str(record.get("payload", {}))
        for record in durable_records
    )
    approval_secret_leaked = APPROVAL_SECRET_CANARY in json.dumps(
        {
            "spoken": engine.spoken,
            "event_payloads": [dict(event.payload) for event in recorder.events],
            "durable_records": durable_records,
        },
        sort_keys=True,
        default=str,
    )
    verbose_full_result = "First sentence. Second sentence. Third sentence."
    verbose_completed_events = [
        event
        for event in completed
        if event.payload.get("intent") == "Explain verbose plan"
        and event.payload.get("result_summary") == verbose_full_result
    ]
    verbose_job_id = str(verbose_completed_events[-1].payload.get("job_id") or "") if verbose_completed_events else ""
    verbose_result_commits = [
        event
        for event in recorder.events
        if event.type == VoiceEventType.ASSISTANT_COMMIT
        and event.payload.get("oracle_job_result")
        and event.payload.get("oracle_job_id") == verbose_job_id
    ]
    verbose_result_spoken_bounded = any(text == "First sentence." for text in engine.spoken) and not any(
        text == verbose_full_result or "Second sentence." in text or "Third sentence." in text
        for text in engine.spoken
    )
    verbose_result_committed_bounded = bool(verbose_result_commits) and any(
        event.payload.get("text") == "First sentence." for event in verbose_result_commits
    ) and all(
        "Second sentence." not in str(event.payload.get("text") or "")
        and "Third sentence." not in str(event.payload.get("text") or "")
        for event in verbose_result_commits
    )
    verbose_result_commit_marked_truncated = any(
        event.payload.get("voice_response_truncated") is True
        and event.payload.get("max_spoken_sentences") == 1
        for event in verbose_result_commits
    )
    verbose_full_result_durable = any(
        record.get("type") == VoiceEventType.ORACLE_JOB_COMPLETED.value
        and record.get("payload", {}).get("job_id") == verbose_job_id
        and record.get("payload", {}).get("result_summary") == verbose_full_result
        and record.get("payload", {}).get("result_text") == verbose_full_result
        for record in durable_records
    )
    report = {
        "ok": (
            len(started) == 9
            and len(queued) == 1
            and scheduler_max_running == 4
            and oracle.max_running >= 4
            and len(local_commits) >= 2
            and bool(running_status_commits)
            and len(completed) == 6
            and len(failed) == 1
            and len(cancelled) == 1
            and bool(fifth_job_id)
            and fifth_job_started_after_capacity_freed
            and oracle.late_cancelled_output_attempted
            and not cancelled_result_spoken
            and not cancelled_result_committed
            and not cancelled_result_progress_leaked
            and not cancelled_result_durable_completed
            and not cancelled_result_durable_text
            and durable_cancelled_record_present
            and durable_completed_jobs == 6
            and bool(approval_waiting)
            and bool(approval_tool_progress)
            and bool(approval_status_commits)
            and approval_payload_redacted
            and not approval_secret_leaked
            and approval_completed
            and bool(failure_commits)
            and failure_spoken
            and durable_failed_record_present
            and session_survived_failure
            and queued_job_update_observed
            and queued_update_started_with_priority
            and queued_update_reached_oracle
            and verbose_result_spoken_bounded
            and verbose_result_committed_bounded
            and verbose_result_commit_marked_truncated
            and verbose_full_result_durable
        ),
        "scenario": "async_kame_oracle_jobs_fake",
        "max_running": scheduler_max_running,
        "max_worker_overlap": oracle.max_running,
        "worker_overlap_proved": oracle.max_running >= 4,
        "noncooperative_cancel_overlap_observed": oracle.max_running > scheduler_max_running,
        "started_jobs": len(started),
        "queued_jobs": len(queued),
        "completed_jobs": len(completed),
        "failed_jobs": len(failed),
        "cancelled_jobs": len(cancelled),
        "local_turn_committed": bool(local_commits),
        "status_turn_committed": bool(running_status_commits),
        "status_text": str(running_status_commits[-1].payload.get("text") or "") if running_status_commits else "",
        "fifth_job_id": fifth_job_id,
        "fifth_job_queued": bool(fifth_job_id),
        "fifth_job_started_after_capacity_freed": fifth_job_started_after_capacity_freed,
        "cancelled_job_id": cancelled_job_id,
        "late_cancelled_output_attempted": oracle.late_cancelled_output_attempted,
        "cancelled_result_spoken": cancelled_result_spoken,
        "cancelled_result_committed": cancelled_result_committed,
        "cancelled_result_progress_leaked": cancelled_result_progress_leaked,
        "cancelled_result_durable_completed": cancelled_result_durable_completed,
        "cancelled_result_durable_text": cancelled_result_durable_text,
        "durable_cancelled_record_present": durable_cancelled_record_present,
        "durable_completed_jobs": durable_completed_jobs,
        "approval_wait_observed": bool(approval_waiting),
        "approval_status_committed": bool(approval_status_commits),
        "approval_tool_progress_observed": bool(approval_tool_progress),
        "approval_payload_redacted": approval_payload_redacted,
        "approval_secret_leaked": approval_secret_leaked,
        "approval_secret_canary_checked": True,
        "approval_completed": approval_completed,
        "approval_status_text": str(approval_status_commits[-1].payload.get("text") or "")
        if approval_status_commits
        else "",
        "failed_job_reported": bool(failure_commits),
        "failed_job_spoken": failure_spoken,
        "durable_failed_record_present": durable_failed_record_present,
        "session_survived_failed_job": session_survived_failure,
        "queued_job_update_observed": queued_job_update_observed,
        "queued_update_started_with_priority": queued_update_started_with_priority,
        "queued_update_reached_oracle": queued_update_reached_oracle,
        "verbose_result_spoken_bounded": verbose_result_spoken_bounded,
        "verbose_result_committed_bounded": verbose_result_committed_bounded,
        "verbose_result_commit_marked_truncated": verbose_result_commit_marked_truncated,
        "verbose_full_result_durable": verbose_full_result_durable,
        "verbose_full_result_chars": len(verbose_full_result),
        "verbose_spoken_result": "First sentence." if verbose_result_spoken_bounded else "",
        "spoken": list(engine.spoken),
        "event_counts": {
            event_type.value: sum(event.type == event_type for event in recorder.events)
            for event_type in {
                VoiceEventType.ORACLE_JOB_STARTED,
                VoiceEventType.ORACLE_JOB_QUEUED,
                VoiceEventType.ORACLE_JOB_WAITING_FOR_APPROVAL,
                VoiceEventType.ORACLE_JOB_COMPLETED,
                VoiceEventType.ORACLE_JOB_FAILED,
                VoiceEventType.ORACLE_JOB_CANCELLED,
                VoiceEventType.ASSISTANT_COMMIT,
            }
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = asyncio.run(run_smoke())
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
