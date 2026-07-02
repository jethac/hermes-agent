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
        self.updates: list[tuple[Any, str, dict[str, Any]]] = []
        self.releases: dict[str, asyncio.Event] = {}
        self.late_cancelled_output_attempted = False
        self.close_cancel_entered = asyncio.Event()
        self.close_release = asyncio.Event()

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
            if key == "Noncooperative close task":
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.close_cancel_entered.set()
                    await self.close_release.wait()
                    raise
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

    async def update_request(self, request: Any, update_text: str, metadata: dict[str, Any]) -> None:
        self.updates.append((request, update_text, dict(metadata)))


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


async def _run_queued_cancel_smoke() -> dict[str, Any]:
    oracle = SmokeOracle()
    engine = SmokeEngine(oracle=oracle)
    recorder = EventRecorder(engine)
    await engine.start(
        RealtimeVoiceSessionConfig(
            session_id="voice-smoke-queued-cancel",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            oracle_jobs={
                "enabled": True,
                "max_concurrent": 1,
                "queue_limit": 4,
                "speak_terminal_results": True,
                "shutdown_timeout_seconds": 0.01,
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
                session_id="voice-smoke-queued-cancel",
                sequence=sequence,
                payload={**payload, "end_of_utterance": True},
            )
        )

    await send(
        {
            "transcript": "queued proof running",
            "intent": "Queued proof running",
            "intent_source": "smoke_reflex",
            "route": "defer",
            "interface_already_said": "Starting queued proof running.",
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_STARTED
            and event.payload.get("intent") == "Queued proof running"
            for event in events
        )
    )
    await send(
        {
            "transcript": "queued proof target",
            "intent": "Queued proof target",
            "intent_source": "smoke_reflex",
            "route": "defer",
            "interface_already_said": "Starting queued proof target.",
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_QUEUED
            and event.payload.get("intent") == "Queued proof target"
            for event in events
        )
    )

    queued_target = next(
        event
        for event in recorder.events
        if event.type == VoiceEventType.ORACLE_JOB_QUEUED
        and event.payload.get("intent") == "Queued proof target"
    )
    await send(
        {
            "transcript": "cancel task two",
            "intent": "Cancel task two.",
            "intent_source": "smoke_reflex",
            "route": "local",
            "local_reply": "Cancelling task two.",
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_CANCELLED
            and event.payload.get("job_id") == queued_target.payload["job_id"]
            for event in events
        )
    )
    oracle.release("Queued proof running")
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_COMPLETED
            and event.payload.get("intent") == "Queued proof running"
            for event in events
        )
    )
    await engine.close()
    collector.cancel()
    try:
        await collector
    except asyncio.CancelledError:
        pass

    target_job_id = str(queued_target.payload.get("job_id") or "")
    target_cancelled = [
        event
        for event in recorder.events
        if event.type == VoiceEventType.ORACLE_JOB_CANCELLED
        and event.payload.get("job_id") == target_job_id
    ]
    target_started = any(
        event.type == VoiceEventType.ORACLE_JOB_STARTED
        and event.payload.get("job_id") == target_job_id
        for event in recorder.events
    )
    target_sent_to_oracle = any(
        str(getattr(request, "intent", "")) == "Queued proof target"
        for request in oracle.requests
    )
    running_completed = any(
        event.type == VoiceEventType.ORACLE_JOB_COMPLETED
        and event.payload.get("intent") == "Queued proof running"
        for event in recorder.events
    )
    cancelled_reason = str(target_cancelled[-1].payload.get("cancel_reason") or "") if target_cancelled else ""
    spoken_cancel_observed = any(
        event.type == VoiceEventType.INTERFACE_ORACLE_CANCEL
        and event.payload.get("job_id") == target_job_id
        and event.payload.get("spoken_control") is True
        for event in recorder.events
    )
    return {
        "ok": bool(target_cancelled)
        and not target_started
        and not target_sent_to_oracle
        and running_completed
        and cancelled_reason == "spoken request to cancel oracle job"
        and spoken_cancel_observed,
        "queued_cancel_observed": bool(target_cancelled),
        "queued_cancel_spoken_control_observed": spoken_cancel_observed,
        "queued_cancelled_before_start": not target_started,
        "queued_cancel_not_sent_to_oracle": not target_sent_to_oracle,
        "queued_cancel_reason": cancelled_reason,
        "queued_cancel_target_job_id": target_job_id,
        "queued_cancel_running_completed": running_completed,
    }


async def _run_approval_capacity_smoke() -> dict[str, Any]:
    oracle = SmokeOracle()
    engine = SmokeEngine(oracle=oracle)
    recorder = EventRecorder(engine)
    await engine.start(
        RealtimeVoiceSessionConfig(
            session_id="voice-smoke-approval-capacity",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            oracle_jobs={
                "enabled": True,
                "max_concurrent": 1,
                "queue_limit": 4,
                "speak_terminal_results": True,
                "shutdown_timeout_seconds": 0.01,
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
                session_id="voice-smoke-approval-capacity",
                sequence=sequence,
                payload={**payload, "end_of_utterance": True},
            )
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
    approval_waiting = [
        event for event in recorder.events if event.type == VoiceEventType.ORACLE_JOB_WAITING_FOR_APPROVAL
    ]
    await send(
        {
            "transcript": "run approval blocked followup",
            "intent": "Approval blocked followup",
            "intent_source": "smoke_reflex",
            "route": "defer",
            "interface_already_said": "Queueing the follow-up.",
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_QUEUED
            and event.payload.get("intent") == "Approval blocked followup"
            for event in events
        )
    )
    queued_followup = [
        event
        for event in recorder.events
        if event.type == VoiceEventType.ORACLE_JOB_QUEUED
        and event.payload.get("intent") == "Approval blocked followup"
    ]
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
            and "1 active out of 1" in str(event.payload.get("text") or "")
            and "0 running" in str(event.payload.get("text") or "")
            and "1 queued" in str(event.payload.get("text") or "")
            and "1 waiting for approval" in str(event.payload.get("text") or "")
            for event in events
        )
    )
    status_commits = [
        event
        for event in recorder.events
        if event.type == VoiceEventType.ASSISTANT_COMMIT
        and "1 waiting for approval" in str(event.payload.get("text") or "")
    ]
    oracle.release("Prepare approval spend")
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_STARTED
            and event.payload.get("intent") == "Approval blocked followup"
            for event in events
        )
    )
    oracle.release("Approval blocked followup")
    await recorder.wait_for(
        lambda events: sum(event.type == VoiceEventType.ORACLE_JOB_COMPLETED for event in events) == 2
    )
    await engine.close()
    collector.cancel()
    try:
        await collector
    except asyncio.CancelledError:
        pass

    followup_started_after_approval = any(
        event.type == VoiceEventType.ORACLE_JOB_STARTED
        and event.payload.get("intent") == "Approval blocked followup"
        for event in recorder.events
    )
    completed = [event for event in recorder.events if event.type == VoiceEventType.ORACLE_JOB_COMPLETED]
    status_text = str(status_commits[-1].payload.get("text") or "") if status_commits else ""
    return {
        "ok": bool(approval_waiting)
        and bool(queued_followup)
        and "1 active out of 1" in status_text
        and "0 running out of 1" not in status_text
        and "1 queued" in status_text
        and "1 waiting for approval" in status_text
        and followup_started_after_approval
        and len(completed) == 2,
        "approval_capacity_waiting_observed": bool(approval_waiting),
        "approval_capacity_followup_queued": bool(queued_followup),
        "approval_capacity_active_visible": "1 active out of 1" in status_text,
        "approval_capacity_misleading_running_capacity": "0 running out of 1" in status_text,
        "approval_capacity_status_text": status_text,
        "approval_capacity_followup_started_after_approval": followup_started_after_approval,
        "approval_capacity_completed_jobs": len(completed),
        "approval_capacity_max_concurrent": 1,
    }


async def _run_cancel_drain_capacity_smoke() -> dict[str, Any]:
    oracle = SmokeOracle()
    engine = SmokeEngine(oracle=oracle)
    recorder = EventRecorder(engine)
    await engine.start(
        RealtimeVoiceSessionConfig(
            session_id="voice-smoke-cancel-drain-capacity",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            oracle_jobs={
                "enabled": True,
                "max_concurrent": 1,
                "queue_limit": 4,
                "speak_terminal_results": True,
                "shutdown_timeout_seconds": 0.01,
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
                session_id="voice-smoke-cancel-drain-capacity",
                sequence=sequence,
                payload={**payload, "end_of_utterance": True},
            )
        )

    await send(
        {
            "transcript": "run smoke task 3",
            "intent": "Run smoke task 3",
            "intent_source": "smoke_reflex",
            "route": "defer",
            "interface_already_said": "Starting cancellable smoke task.",
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_STARTED
            and event.payload.get("intent") == "Run smoke task 3"
            for event in events
        )
    )
    await send(
        {
            "transcript": "run drain followup",
            "intent": "Drain followup",
            "intent_source": "smoke_reflex",
            "route": "defer",
            "interface_already_said": "Queueing the follow-up.",
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_QUEUED
            and event.payload.get("intent") == "Drain followup"
            for event in events
        )
    )
    await send(
        {
            "transcript": "cancel task one",
            "intent": "Cancel task one.",
            "intent_source": "smoke_reflex",
            "route": "local",
            "local_reply": "Cancelling task one.",
        }
    )
    await recorder.wait_for(
        lambda events: any(event.type == VoiceEventType.ORACLE_JOB_CANCEL_REQUESTED for event in events)
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
            and "1 active out of 1" in str(event.payload.get("text") or "")
            and "0 running" in str(event.payload.get("text") or "")
            and "1 queued" in str(event.payload.get("text") or "")
            and "1 cancelling" in str(event.payload.get("text") or "")
            for event in events
        )
    )
    status_commits = [
        event
        for event in recorder.events
        if event.type == VoiceEventType.ASSISTANT_COMMIT
        and "1 cancelling" in str(event.payload.get("text") or "")
    ]
    oracle.release("Run smoke task 3")
    await recorder.wait_for(
        lambda events: any(event.type == VoiceEventType.ORACLE_JOB_CANCELLED for event in events)
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_STARTED
            and event.payload.get("intent") == "Drain followup"
            for event in events
        )
    )
    oracle.release("Drain followup")
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_COMPLETED
            and event.payload.get("intent") == "Drain followup"
            for event in events
        )
    )
    await engine.close()
    collector.cancel()
    try:
        await collector
    except asyncio.CancelledError:
        pass

    cancel_requested = [
        event for event in recorder.events if event.type == VoiceEventType.ORACLE_JOB_CANCEL_REQUESTED
    ]
    cancelled = [event for event in recorder.events if event.type == VoiceEventType.ORACLE_JOB_CANCELLED]
    followup_queued = any(
        event.type == VoiceEventType.ORACLE_JOB_QUEUED
        and event.payload.get("intent") == "Drain followup"
        for event in recorder.events
    )
    followup_started = any(
        event.type == VoiceEventType.ORACLE_JOB_STARTED
        and event.payload.get("intent") == "Drain followup"
        for event in recorder.events
    )
    status_text = str(status_commits[-1].payload.get("text") or "") if status_commits else ""
    return {
        "ok": bool(cancel_requested)
        and bool(cancelled)
        and followup_queued
        and followup_started
        and "1 active out of 1" in status_text
        and "0 running out of 1" not in status_text
        and "1 queued" in status_text
        and "1 cancelling" in status_text,
        "cancel_drain_requested_observed": bool(cancel_requested),
        "cancel_drain_cancelled_observed": bool(cancelled),
        "cancel_drain_followup_queued": followup_queued,
        "cancel_drain_active_visible": "1 active out of 1" in status_text,
        "cancel_drain_misleading_running_capacity": "0 running out of 1" in status_text,
        "cancel_drain_status_text": status_text,
        "cancel_drain_followup_started_after_cancel": followup_started,
        "cancel_drain_max_concurrent": 1,
    }


async def _run_terminal_result_policy_smoke() -> dict[str, Any]:
    oracle = SmokeOracle()
    engine = SmokeEngine(oracle=oracle)
    recorder = EventRecorder(engine)
    await engine.start(
        RealtimeVoiceSessionConfig(
            session_id="voice-smoke-terminal-result-policy",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            oracle_jobs={
                "enabled": True,
                "max_concurrent": 1,
                "queue_limit": 4,
                "speak_terminal_results": False,
                "shutdown_timeout_seconds": 0.01,
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
                session_id="voice-smoke-terminal-result-policy",
                sequence=sequence,
                payload={**payload, "end_of_utterance": True},
            )
        )

    await send(
        {
            "transcript": "suppress terminal result",
            "intent": "Suppress terminal result",
            "intent_source": "smoke_reflex",
            "route": "defer",
            "interface_already_said": "Checking terminal result policy.",
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_STARTED
            and event.payload.get("intent") == "Suppress terminal result"
            for event in events
        )
    )
    oracle.release("Suppress terminal result")
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_COMPLETED
            and event.payload.get("intent") == "Suppress terminal result"
            for event in events
        )
    )
    unsolicited_result_events = [
        event
        for event in recorder.events
        if event.type in {VoiceEventType.ASSISTANT_TEXT_PARTIAL, VoiceEventType.ASSISTANT_COMMIT}
        and event.payload.get("oracle_job_result")
    ]
    unsolicited_result_spoken = any("Finished Suppress terminal result" in text for text in engine.spoken)

    await send(
        {
            "transcript": "what completed",
            "intent": "What completed?",
            "intent_source": "smoke_reflex",
            "route": "local",
            "local_reply": "Let me check.",
            "max_spoken_sentences": 5,
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ASSISTANT_COMMIT
            and str(event.payload.get("text") or "").startswith(
                "No oracle jobs are running or queued right now. Recent:"
            )
            for event in events
        )
    )
    status_commits = [
        event
        for event in recorder.events
        if event.type == VoiceEventType.ASSISTANT_COMMIT
        and str(event.payload.get("text") or "").startswith("No oracle jobs are running or queued right now. Recent:")
    ]
    await engine.close()
    collector.cancel()
    try:
        await collector
    except asyncio.CancelledError:
        pass

    status_text = str(status_commits[-1].payload.get("text") or "") if status_commits else ""
    terminal_result_status_available = "completed: Finished Suppress terminal result." in status_text
    return {
        "ok": not unsolicited_result_events
        and not unsolicited_result_spoken
        and terminal_result_status_available,
        "terminal_result_auto_summarize_default": True,
        "terminal_result_suppression_config": "oracle_jobs.speak_terminal_results=false",
        "terminal_result_suppressed": not unsolicited_result_events and not unsolicited_result_spoken,
        "terminal_result_unsolicited_event_count": len(unsolicited_result_events),
        "terminal_result_unsolicited_spoken": unsolicited_result_spoken,
        "terminal_result_status_available": terminal_result_status_available,
        "terminal_result_status_text": status_text,
    }


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
                "shutdown_timeout_seconds": 0.01,
            },
            max_spoken_sentences=5,
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
            "transcript": "stop talking",
            "intent": "Stop talking.",
            "intent_source": "smoke_reflex",
            "route": "local",
            "local_reply": "Okay.",
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ASSISTANT_COMMIT
            and event.payload.get("local_reply")
            and event.payload.get("text") == "Okay."
            for event in events
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
            "transcript": "task one also include running update context",
            "intent": "Task one also include running update context.",
            "intent_source": "smoke_reflex",
            "route": "local",
            "local_reply": "Adding that to task one.",
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE
            and event.payload.get("job_id") == "voice-oracle-001"
            and event.payload.get("update_count") == 1
            and event.payload.get("latest_update") == "include running update context"
            and event.payload.get("spoken_control") is True
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
    await send(
        {
            "transcript": "make task five high priority",
            "intent": "Make task five high priority.",
            "intent_source": "smoke_reflex",
            "route": "local",
            "local_reply": "Making task five high priority.",
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE
            and event.payload.get("job_id") == task_5_queued.payload["job_id"]
            and event.payload.get("priority") == "high"
            and event.payload.get("spoken_control") is True
            for event in events
        )
    )
    await send(
        {
            "transcript": "task five also include smoke update context",
            "intent": "Task five also include smoke update context.",
            "intent_source": "smoke_reflex",
            "route": "local",
            "local_reply": "Adding that context.",
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE
            and event.payload.get("job_id") == task_5_queued.payload["job_id"]
            and event.payload.get("priority") == "high"
            and event.payload.get("update_count") == 1
            and event.payload.get("spoken_control") is True
            for event in events
        )
    )

    task_3_started = next(
        event
        for event in recorder.events
        if event.type == VoiceEventType.ORACLE_JOB_STARTED
        and event.payload.get("intent") == "Run smoke task 3"
    )
    await send(
        {
            "transcript": "cancel task three",
            "intent": "Cancel task three.",
            "intent_source": "smoke_reflex",
            "route": "local",
            "local_reply": "Cancelling task three.",
        }
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
            and "1 waiting for approval" in str(event.payload.get("text") or "")
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
    await send(
        {
            "transcript": "what completed",
            "intent": "What completed?",
            "intent_source": "smoke_reflex",
            "route": "local",
            "local_reply": "Let me check.",
            "max_spoken_sentences": 5,
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ASSISTANT_COMMIT
            and str(event.payload.get("text") or "").startswith(
                "No oracle jobs are running or queued right now. Recent:"
            )
            and "completed: First sentence. Second sentence. Third sentence." in str(event.payload.get("text") or "")
            for event in events
        )
    )
    await send(
        {
            "transcript": "run noncooperative close task",
            "intent": "Noncooperative close task",
            "intent_source": "smoke_reflex",
            "route": "defer",
            "interface_already_said": "Starting close-time task.",
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_STARTED
            and event.payload.get("intent") == "Noncooperative close task"
            for event in events
        )
    )
    close_started_at = time.perf_counter()
    await asyncio.wait_for(engine.close(), timeout=1.0)
    close_elapsed_ms = round((time.perf_counter() - close_started_at) * 1000, 3)
    close_cancel_entered = oracle.close_cancel_entered.is_set()
    oracle.close_release.set()
    await asyncio.sleep(0)
    collector.cancel()
    try:
        await collector
    except asyncio.CancelledError:
        pass
    queued_cancel_smoke = await _run_queued_cancel_smoke()
    approval_capacity_smoke = await _run_approval_capacity_smoke()
    cancel_drain_capacity_smoke = await _run_cancel_drain_capacity_smoke()
    terminal_result_policy_smoke = await _run_terminal_result_policy_smoke()

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
    stop_talking_commits = [
        event
        for event in recorder.events
        if event.type == VoiceEventType.ASSISTANT_COMMIT
        and event.payload.get("local_reply")
        and event.payload.get("text") == "Okay."
    ]
    stop_talking_commit_index = (
        recorder.events.index(stop_talking_commits[-1])
        if stop_talking_commits
        else -1
    )
    first_running_status_index = (
        recorder.events.index(running_status_commits[0])
        if running_status_commits
        else -1
    )
    events_after_stop_until_status = (
        recorder.events[stop_talking_commit_index + 1 : first_running_status_index + 1]
        if stop_talking_commit_index >= 0 and first_running_status_index >= stop_talking_commit_index
        else []
    )
    playback_stop_cancelled_jobs = any(
        event.type in {VoiceEventType.INTERFACE_ORACLE_CANCEL, VoiceEventType.ORACLE_JOB_CANCELLED}
        for event in events_after_stop_until_status
    )
    playback_stop_jobs_still_running = bool(running_status_commits)
    playback_stop_did_not_cancel_jobs = bool(stop_talking_commits) and playback_stop_jobs_still_running and not playback_stop_cancelled_jobs
    terminal_status_commits = [
        event
        for event in recorder.events
        if event.type == VoiceEventType.ASSISTANT_COMMIT
        and str(event.payload.get("text") or "").startswith("No oracle jobs are running or queued right now. Recent:")
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
        and "1 waiting for approval" in str(event.payload.get("text") or "")
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
    spoken_priority_control_observed = any(
        event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE
        and event.payload.get("job_id") == task_5_queued.payload["job_id"]
        and event.payload.get("priority") == "high"
        and event.payload.get("spoken_control") is True
        for event in recorder.events
    )
    spoken_update_control_observed = any(
        event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE
        and event.payload.get("job_id") == task_5_queued.payload["job_id"]
        and event.payload.get("update_count") == 1
        and event.payload.get("spoken_control") is True
        for event in recorder.events
    )
    queued_update_latest_update_visible = any(
        event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE
        and event.payload.get("job_id") == task_5_queued.payload["job_id"]
        and event.payload.get("latest_update") == "include smoke update context"
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
    running_update_records = [
        (request, update_text, metadata)
        for request, update_text, metadata in oracle.updates
        if str(getattr(request, "intent", "")) == "Run smoke task 1"
        and update_text == "include running update context"
    ]
    running_job_update_observed = any(
        event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE
        and event.payload.get("job_id") == "voice-oracle-001"
        and event.payload.get("update_count") == 1
        and event.payload.get("spoken_control") is True
        for event in recorder.events
    )
    running_update_latest_update_visible = any(
        event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE
        and event.payload.get("job_id") == "voice-oracle-001"
        and event.payload.get("latest_update") == "include running update context"
        for event in recorder.events
    )
    running_update_reached_oracle = any(
        "include running update context" in tuple(getattr(request, "job_updates", ()))
        for request, _, _ in running_update_records
    )
    running_update_delivery_metadata_ok = any(
        metadata.get("job_id") == "voice-oracle-001"
        and metadata.get("state") == "running"
        and metadata.get("update_count") == 1
        and metadata.get("latest_update") == "include running update context"
        for _, _, metadata in running_update_records
    )
    cancelled_job_id = str(task_3_started.payload["job_id"])
    spoken_cancel_control_observed = any(
        event.type == VoiceEventType.INTERFACE_ORACLE_CANCEL
        and event.payload.get("job_id") == cancelled_job_id
        and event.payload.get("spoken_control") is True
        for event in recorder.events
    )
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
        text == verbose_full_result for text in engine.spoken
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
    terminal_status_text = str(terminal_status_commits[-1].payload.get("text") or "") if terminal_status_commits else ""
    completed_result_status_visible = (
        "completed: First sentence. Second sentence. Third sentence." in terminal_status_text
    )
    close_cancelled_events = [
        event
        for event in cancelled
        if event.payload.get("intent") == "Noncooperative close task"
        and event.payload.get("cancel_reason") == "session closed"
    ]
    shutdown_timeout_configured_ms = 10
    shutdown_bounded_close_observed = close_elapsed_ms < 1000
    shutdown_forced_cancel_observed = bool(close_cancelled_events) and close_cancel_entered
    report = {
        "ok": (
            len(started) == 10
            and len(queued) == 1
            and scheduler_max_running == 4
            and oracle.max_running == 4
            and len(local_commits) >= 2
            and playback_stop_did_not_cancel_jobs
            and bool(running_status_commits)
            and len(completed) == 6
            and len(failed) == 1
            and len(cancelled) == 2
            and bool(fifth_job_id)
            and fifth_job_started_after_capacity_freed
            and oracle.late_cancelled_output_attempted
            and not cancelled_result_spoken
            and not cancelled_result_committed
            and not cancelled_result_progress_leaked
            and not cancelled_result_durable_completed
            and not cancelled_result_durable_text
            and durable_cancelled_record_present
            and durable_completed_jobs == 1
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
            and spoken_priority_control_observed
            and spoken_update_control_observed
            and spoken_cancel_control_observed
            and running_job_update_observed
            and running_update_latest_update_visible
            and running_update_reached_oracle
            and running_update_delivery_metadata_ok
            and queued_update_latest_update_visible
            and queued_update_started_with_priority
            and queued_update_reached_oracle
            and verbose_result_spoken_bounded
            and verbose_result_committed_bounded
            and verbose_result_commit_marked_truncated
            and verbose_full_result_durable
            and completed_result_status_visible
            and shutdown_bounded_close_observed
            and shutdown_forced_cancel_observed
            and queued_cancel_smoke["ok"]
            and approval_capacity_smoke["ok"]
            and cancel_drain_capacity_smoke["ok"]
            and terminal_result_policy_smoke["ok"]
        ),
        "scenario": "async_kame_oracle_jobs_fake",
        "max_running": scheduler_max_running,
        "max_worker_overlap": oracle.max_running,
        "worker_overlap_proved": oracle.max_running >= 4,
        "worker_overlap_within_capacity": oracle.max_running <= scheduler_max_running,
        "noncooperative_cancel_overlap_observed": oracle.max_running > scheduler_max_running,
        "started_jobs": len(started),
        "queued_jobs": len(queued),
        "completed_jobs": len(completed),
        "failed_jobs": len(failed),
        "cancelled_jobs": len(cancelled),
        "shutdown_timeout_configured_ms": shutdown_timeout_configured_ms,
        "shutdown_close_elapsed_ms": close_elapsed_ms,
        "shutdown_bounded_close_observed": shutdown_bounded_close_observed,
        "shutdown_forced_cancel_observed": shutdown_forced_cancel_observed,
        "shutdown_close_cancel_entered": close_cancel_entered,
        "shutdown_cancelled_jobs": len(close_cancelled_events),
        "queued_cancel_smoke_ok": queued_cancel_smoke["ok"],
        "queued_cancel_observed": queued_cancel_smoke["queued_cancel_observed"],
        "queued_cancel_spoken_control_observed": queued_cancel_smoke["queued_cancel_spoken_control_observed"],
        "queued_cancelled_before_start": queued_cancel_smoke["queued_cancelled_before_start"],
        "queued_cancel_not_sent_to_oracle": queued_cancel_smoke["queued_cancel_not_sent_to_oracle"],
        "queued_cancel_reason": queued_cancel_smoke["queued_cancel_reason"],
        "queued_cancel_target_job_id": queued_cancel_smoke["queued_cancel_target_job_id"],
        "queued_cancel_running_completed": queued_cancel_smoke["queued_cancel_running_completed"],
        "approval_capacity_smoke_ok": approval_capacity_smoke["ok"],
        "approval_capacity_waiting_observed": approval_capacity_smoke["approval_capacity_waiting_observed"],
        "approval_capacity_followup_queued": approval_capacity_smoke["approval_capacity_followup_queued"],
        "approval_capacity_active_visible": approval_capacity_smoke["approval_capacity_active_visible"],
        "approval_capacity_misleading_running_capacity": approval_capacity_smoke[
            "approval_capacity_misleading_running_capacity"
        ],
        "approval_capacity_status_text": approval_capacity_smoke["approval_capacity_status_text"],
        "approval_capacity_followup_started_after_approval": approval_capacity_smoke[
            "approval_capacity_followup_started_after_approval"
        ],
        "approval_capacity_completed_jobs": approval_capacity_smoke["approval_capacity_completed_jobs"],
        "approval_capacity_max_concurrent": approval_capacity_smoke["approval_capacity_max_concurrent"],
        "cancel_drain_capacity_smoke_ok": cancel_drain_capacity_smoke["ok"],
        "cancel_drain_requested_observed": cancel_drain_capacity_smoke["cancel_drain_requested_observed"],
        "cancel_drain_cancelled_observed": cancel_drain_capacity_smoke["cancel_drain_cancelled_observed"],
        "cancel_drain_followup_queued": cancel_drain_capacity_smoke["cancel_drain_followup_queued"],
        "cancel_drain_active_visible": cancel_drain_capacity_smoke["cancel_drain_active_visible"],
        "cancel_drain_misleading_running_capacity": cancel_drain_capacity_smoke[
            "cancel_drain_misleading_running_capacity"
        ],
        "cancel_drain_status_text": cancel_drain_capacity_smoke["cancel_drain_status_text"],
        "cancel_drain_followup_started_after_cancel": cancel_drain_capacity_smoke[
            "cancel_drain_followup_started_after_cancel"
        ],
        "cancel_drain_max_concurrent": cancel_drain_capacity_smoke["cancel_drain_max_concurrent"],
        "terminal_result_policy_smoke_ok": terminal_result_policy_smoke["ok"],
        "terminal_result_auto_summarize_default": terminal_result_policy_smoke[
            "terminal_result_auto_summarize_default"
        ],
        "terminal_result_suppression_config": terminal_result_policy_smoke["terminal_result_suppression_config"],
        "terminal_result_suppressed": terminal_result_policy_smoke["terminal_result_suppressed"],
        "terminal_result_unsolicited_event_count": terminal_result_policy_smoke[
            "terminal_result_unsolicited_event_count"
        ],
        "terminal_result_unsolicited_spoken": terminal_result_policy_smoke["terminal_result_unsolicited_spoken"],
        "terminal_result_status_available": terminal_result_policy_smoke["terminal_result_status_available"],
        "terminal_result_status_text": terminal_result_policy_smoke["terminal_result_status_text"],
        "local_turn_committed": bool(local_commits),
        "playback_stop_committed": bool(stop_talking_commits),
        "playback_stop_jobs_still_running": playback_stop_jobs_still_running,
        "playback_stop_cancelled_jobs": playback_stop_cancelled_jobs,
        "playback_stop_does_not_cancel_jobs": playback_stop_did_not_cancel_jobs,
        "status_turn_committed": bool(running_status_commits),
        "status_text": str(running_status_commits[-1].payload.get("text") or "") if running_status_commits else "",
        "terminal_status_committed": bool(terminal_status_commits),
        "completed_result_status_visible": completed_result_status_visible,
        "terminal_status_text": terminal_status_text,
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
        "running_job_update_observed": running_job_update_observed,
        "running_update_latest_update_visible": running_update_latest_update_visible,
        "running_update_latest_update_text": "include running update context"
        if running_update_latest_update_visible
        else "",
        "running_update_reached_oracle": running_update_reached_oracle,
        "running_update_delivery_metadata_ok": running_update_delivery_metadata_ok,
        "spoken_priority_control_observed": spoken_priority_control_observed,
        "spoken_update_control_observed": spoken_update_control_observed,
        "spoken_cancel_control_observed": spoken_cancel_control_observed,
        "queued_update_latest_update_visible": queued_update_latest_update_visible,
        "queued_update_latest_update_text": "include smoke update context"
        if queued_update_latest_update_visible
        else "",
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
                VoiceEventType.ORACLE_JOB_PROGRESS,
                VoiceEventType.ORACLE_JOB_WAITING_FOR_APPROVAL,
                VoiceEventType.ORACLE_JOB_COMPLETED,
                VoiceEventType.ORACLE_JOB_FAILED,
                VoiceEventType.ORACLE_JOB_CANCEL_REQUESTED,
                VoiceEventType.ORACLE_JOB_CANCELLED,
                VoiceEventType.INTERFACE_ORACLE_UPDATE,
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
