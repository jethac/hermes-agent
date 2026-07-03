"""Headless smoke for async KAME reflex/oracle voice jobs.

This exercises the in-process realtime voice engine with fake oracle workers.
It is intentionally provider-free: no Discord, sidecar, Spark, STT, or TTS
service is required.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from agent.realtime_voice import RealtimeVoiceEngineKind, RealtimeVoiceSessionConfig, VoiceEvent, VoiceEventType
from agent.realtime_voice_kame import KameOracleRequest
from agent.realtime_voice_oracle_jobs import OracleJobManager
from agent.realtime_voice_session import RealtimeVoiceSession
from agent.realtime_voice_text_engine import KameInterfaceOracleEngine


APPROVAL_SECRET_CANARY = "secret test value must not leak"
STATUS_ORDINAL_LABELS = ("job one", "job two", "job three", "job four", "job five")


class SmokeOracle:
    def __init__(self) -> None:
        self.running = 0
        self.max_running = 0
        self.requests: list[Any] = []
        self.updates: list[tuple[Any, str, dict[str, Any]]] = []
        self.releases: dict[str, asyncio.Event] = {}
        self.late_cancelled_output_attempted = False
        self.approval_cancel_late_output_attempted = False
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
            if key == "Cancellable approval spend":
                yield {
                    "event": "tool_call",
                    "tool_name": "stripe_link_purchase",
                    "tool_call_id": "call-approval-cancel-smoke",
                    "approval_required": True,
                    "approval_id": "approval-cancel-smoke-123",
                    "approval_reason": "Stripe Link spend requires approval",
                    "arguments": {"amount": 200, "card": APPROVAL_SECRET_CANARY},
                }
                try:
                    await self.releases[key].wait()
                except asyncio.CancelledError:
                    self.approval_cancel_late_output_attempted = True
                    await self.releases[key].wait()
                    yield "Late approval cancellation result."
                    return
                yield "Approval cancellation smoke should not complete."
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
    def __init__(self, *, oracle: SmokeOracle, sidecar: Any = None) -> None:
        super().__init__(oracle=oracle, sidecar=sidecar)
        self.spoken: list[str] = []

    async def _speak_chunk(self, text: str, playback_generation: int) -> None:
        self.spoken.append(text)
        await asyncio.sleep(0)


class SmokeSidecar:
    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.received: list[VoiceEvent] = []
        self._events: asyncio.Queue[VoiceEvent | None] = asyncio.Queue()

    async def start(self, _config: RealtimeVoiceSessionConfig) -> None:
        self.started = True

    async def send_event(self, event: VoiceEvent) -> None:
        self.received.append(event)

    async def events(self):
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def inject(self, event: VoiceEvent) -> None:
        await self._events.put(event)

    async def close(self) -> None:
        self.closed = True
        await self._events.put(None)


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
    target_cancel_requested = [
        event
        for event in recorder.events
        if event.type == VoiceEventType.ORACLE_JOB_CANCEL_REQUESTED
        and event.payload.get("job_id") == target_job_id
    ]
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
        "ok": bool(target_cancel_requested)
        and bool(target_cancelled)
        and not target_started
        and not target_sent_to_oracle
        and running_completed
        and cancelled_reason == "spoken request to cancel oracle job"
        and spoken_cancel_observed,
        "queued_cancel_requested_observed": bool(target_cancel_requested),
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


async def _run_approval_cancel_capacity_smoke() -> dict[str, Any]:
    oracle = SmokeOracle()
    engine = SmokeEngine(oracle=oracle)
    recorder = EventRecorder(engine)
    await engine.start(
        RealtimeVoiceSessionConfig(
            session_id="voice-smoke-approval-cancel-capacity",
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
                session_id="voice-smoke-approval-cancel-capacity",
                sequence=sequence,
                payload={**payload, "end_of_utterance": True},
            )
        )

    await send(
        {
            "transcript": "prepare cancellable approval spend",
            "intent": "Cancellable approval spend",
            "intent_source": "smoke_reflex",
            "route": "defer",
            "interface_already_said": "Preparing cancellable spend approval.",
        }
    )
    await recorder.wait_for(
        lambda events: any(event.type == VoiceEventType.ORACLE_JOB_WAITING_FOR_APPROVAL for event in events)
    )
    waiting = [
        event for event in recorder.events if event.type == VoiceEventType.ORACLE_JOB_WAITING_FOR_APPROVAL
    ]
    first_job_id = str(waiting[-1].payload.get("job_id") or "") if waiting else ""
    await send(
        {
            "transcript": "run approval followup",
            "intent": "Approval followup",
            "intent_source": "smoke_reflex",
            "route": "defer",
            "interface_already_said": "Queueing the follow-up.",
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_QUEUED
            and event.payload.get("intent") == "Approval followup"
            for event in events
        )
    )
    queued_followup = [
        event
        for event in recorder.events
        if event.type == VoiceEventType.ORACLE_JOB_QUEUED
        and event.payload.get("intent") == "Approval followup"
    ]
    followup_job_id = str(queued_followup[-1].payload.get("job_id") or "") if queued_followup else ""
    await send(
        {
            "transcript": "cancel cancellable approval spend",
            "intent": "Cancel cancellable approval spend.",
            "intent_source": "smoke_reflex",
            "route": "local",
            "local_reply": "Cancelling task one.",
        }
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_CANCEL_REQUESTED
            and event.payload.get("job_id") == first_job_id
            for event in events
        )
    )
    await send(
        {
            "transcript": "what is still running",
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
    followup_started_before_cancel_drained = any(
        event.type == VoiceEventType.ORACLE_JOB_STARTED
        and event.payload.get("job_id") == followup_job_id
        for event in recorder.events
    )
    oracle.release("Cancellable approval spend")
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_CANCELLED
            and event.payload.get("job_id") == first_job_id
            for event in events
        )
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_STARTED
            and event.payload.get("job_id") == followup_job_id
            for event in events
        )
    )
    oracle.release("Approval followup")
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_COMPLETED
            and event.payload.get("job_id") == followup_job_id
            for event in events
        )
    )
    await engine.close()
    collector.cancel()
    try:
        await collector
    except asyncio.CancelledError:
        pass

    cancelled = [
        event
        for event in recorder.events
        if event.type == VoiceEventType.ORACLE_JOB_CANCELLED
        and event.payload.get("job_id") == first_job_id
    ]
    first_completed = any(
        event.type == VoiceEventType.ORACLE_JOB_COMPLETED
        and event.payload.get("job_id") == first_job_id
        for event in recorder.events
    )
    late_result_spoken = any("Late approval cancellation result." in text for text in engine.spoken)
    status_text = str(status_commits[-1].payload.get("text") or "") if status_commits else ""
    return {
        "ok": bool(waiting)
        and bool(queued_followup)
        and bool(cancelled)
        and oracle.approval_cancel_late_output_attempted
        and not first_completed
        and not late_result_spoken
        and not followup_started_before_cancel_drained
        and "1 active out of 1" in status_text
        and "0 running out of 1" not in status_text
        and "1 queued" in status_text
        and "1 cancelling" in status_text,
        "approval_cancel_waiting_observed": bool(waiting),
        "approval_cancel_followup_queued": bool(queued_followup),
        "approval_cancel_requested_observed": any(
            event.type == VoiceEventType.ORACLE_JOB_CANCEL_REQUESTED
            and event.payload.get("job_id") == first_job_id
            for event in recorder.events
        ),
        "approval_cancel_cancelled_observed": bool(cancelled),
        "approval_cancel_late_output_attempted": oracle.approval_cancel_late_output_attempted,
        "approval_cancel_completed_after_cancel": first_completed,
        "approval_cancel_late_result_spoken": late_result_spoken,
        "approval_cancel_followup_started_before_cancel_drained": followup_started_before_cancel_drained,
        "approval_cancel_followup_started_after_cancel": any(
            event.type == VoiceEventType.ORACLE_JOB_STARTED
            and event.payload.get("job_id") == followup_job_id
            for event in recorder.events
        ),
        "approval_cancel_active_visible": "1 active out of 1" in status_text,
        "approval_cancel_misleading_running_capacity": "0 running out of 1" in status_text,
        "approval_cancel_status_text": status_text,
        "approval_cancel_max_concurrent": 1,
    }


async def _run_terminal_result_policy_smoke() -> dict[str, Any]:
    default_oracle = SmokeOracle()
    default_engine = SmokeEngine(oracle=default_oracle)
    default_recorder = EventRecorder(default_engine)
    await default_engine.start(
        RealtimeVoiceSessionConfig(
            session_id="voice-smoke-terminal-result-default",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            oracle_jobs={
                "enabled": True,
                "max_concurrent": 1,
                "queue_limit": 4,
                "shutdown_timeout_seconds": 0.01,
            },
            metadata={"transport": "smoke"},
        )
    )
    default_collector = asyncio.create_task(default_recorder.run())
    await default_engine.receive_event(
        VoiceEvent(
            type=VoiceEventType.AUDIO_INPUT_CHUNK,
            session_id="voice-smoke-terminal-result-default",
            sequence=1,
            payload={
                "transcript": "default terminal result",
                "intent": "Default terminal result",
                "intent_source": "smoke_reflex",
                "route": "defer",
                "interface_already_said": "Checking default terminal result.",
                "end_of_utterance": True,
            },
        )
    )
    await default_recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_STARTED
            and event.payload.get("intent") == "Default terminal result"
            for event in events
        )
    )
    default_oracle.release("Default terminal result")
    await default_recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ASSISTANT_COMMIT
            and event.payload.get("oracle_job_result")
            and "Finished Default terminal result." in str(event.payload.get("text") or "")
            for event in events
        )
    )
    default_result_events = [
        event
        for event in default_recorder.events
        if event.type in {VoiceEventType.ASSISTANT_TEXT_PARTIAL, VoiceEventType.ASSISTANT_COMMIT}
        and event.payload.get("oracle_job_result")
    ]
    default_result_spoken = any(
        "Finished Default terminal result." in text for text in default_engine.spoken
    )
    await default_engine.close()
    default_collector.cancel()
    try:
        await default_collector
    except asyncio.CancelledError:
        pass

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
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_RESULT_SUPPRESSED
            and event.payload.get("intent") == "Suppress terminal result"
            for event in events
        )
    )
    suppressed_result_events = [
        event
        for event in recorder.events
        if event.type == VoiceEventType.ORACLE_JOB_RESULT_SUPPRESSED
        and event.payload.get("intent") == "Suppress terminal result"
    ]
    suppressed_result_payloads = [dict(event.payload) for event in suppressed_result_events]
    suppressed_result_payload_clean = bool(suppressed_result_payloads) and all(
        "result_summary" not in payload
        and "result_text" not in payload
        and "Finished Suppress terminal result" not in json.dumps(payload, sort_keys=True)
        for payload in suppressed_result_payloads
    )
    suppressed_result_reason = (
        str(suppressed_result_payloads[-1].get("suppression_reason") or "")
        if suppressed_result_payloads
        else ""
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
        "ok": bool(default_result_events)
        and default_result_spoken
        and not unsolicited_result_events
        and not unsolicited_result_spoken
        and bool(suppressed_result_events)
        and suppressed_result_payload_clean
        and terminal_result_status_available,
        "terminal_result_auto_summarize_default": bool(default_result_events),
        "terminal_result_default_event_count": len(default_result_events),
        "terminal_result_default_spoken": default_result_spoken,
        "terminal_result_suppression_config": "oracle_jobs.speak_terminal_results=false",
        "terminal_result_suppressed": not unsolicited_result_events and not unsolicited_result_spoken,
        "terminal_result_suppressed_event_observed": bool(suppressed_result_events),
        "terminal_result_suppressed_event_count": len(suppressed_result_events),
        "terminal_result_suppressed_reason": suppressed_result_reason,
        "terminal_result_suppressed_payload_clean": suppressed_result_payload_clean,
        "terminal_result_unsolicited_event_count": len(unsolicited_result_events),
        "terminal_result_unsolicited_spoken": unsolicited_result_spoken,
        "terminal_result_status_available": terminal_result_status_available,
        "terminal_result_status_text": status_text,
    }


async def _run_sidecar_control_smoke() -> dict[str, Any]:
    oracle = SmokeOracle()
    sidecar = SmokeSidecar()
    engine = SmokeEngine(oracle=oracle, sidecar=sidecar)
    recorder = EventRecorder(engine)
    await engine.start(
        RealtimeVoiceSessionConfig(
            session_id="voice-smoke-sidecar-control",
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
    await engine.receive_event(
        VoiceEvent(
            type=VoiceEventType.AUDIO_INPUT_CHUNK,
            session_id="voice-smoke-sidecar-control",
            sequence=1,
            payload={
                "transcript": "sidecar controlled task",
                "intent": "Sidecar controlled task",
                "intent_source": "smoke_reflex",
                "route": "defer",
                "interface_already_said": "Starting sidecar controlled task.",
                "end_of_utterance": True,
            },
        )
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_STARTED
            and event.payload.get("intent") == "Sidecar controlled task"
            for event in events
        )
    )
    started = next(
        event
        for event in recorder.events
        if event.type == VoiceEventType.ORACLE_JOB_STARTED
        and event.payload.get("intent") == "Sidecar controlled task"
    )
    job_id = str(started.payload.get("job_id") or "")
    await sidecar.inject(
        VoiceEvent(
            type=VoiceEventType.INTERFACE_ORACLE_UPDATE,
            session_id="voice-smoke-sidecar-control",
            sequence=2,
            payload={
                "job_id": job_id,
                "priority": "high",
                "update_text": "include sidecar update context",
                "reason": "sidecar smoke update",
                "transport": "discord_voice",
                "sidecar_control": True,
            },
        )
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE
            and event.payload.get("job_id") == job_id
            and event.payload.get("priority") == "high"
            and event.payload.get("latest_update") == "include sidecar update context"
            for event in events
        )
    )
    await sidecar.inject(
        VoiceEvent(
            type=VoiceEventType.INTERFACE_ORACLE_CANCEL,
            session_id="voice-smoke-sidecar-control",
            sequence=3,
            payload={
                "job_id": job_id,
                "reason": "sidecar smoke cancel",
                "transport": "discord_voice",
                "sidecar_control": True,
            },
        )
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_CANCELLED
            and event.payload.get("job_id") == job_id
            for event in events
        )
    )
    await engine.close()
    collector.cancel()
    try:
        await collector
    except asyncio.CancelledError:
        pass

    update_observed = any(
        event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE
        and event.payload.get("job_id") == job_id
        and event.payload.get("latest_update") == "include sidecar update context"
        for event in recorder.events
    )
    update_reached_oracle = any(
        update_text == "include sidecar update context"
        and str(getattr(request, "intent", "")) == "Sidecar controlled task"
        and metadata.get("job_id") == job_id
        for request, update_text, metadata in oracle.updates
    )
    cancel_requested = any(
        event.type == VoiceEventType.ORACLE_JOB_CANCEL_REQUESTED
        and event.payload.get("job_id") == job_id
        for event in recorder.events
    )
    cancelled = [
        event
        for event in recorder.events
        if event.type == VoiceEventType.ORACLE_JOB_CANCELLED
        and event.payload.get("job_id") == job_id
    ]
    completed = any(
        event.type == VoiceEventType.ORACLE_JOB_COMPLETED
        and event.payload.get("job_id") == job_id
        for event in recorder.events
    )
    feedback_types = {event.type for event in sidecar.received}
    return {
        "ok": update_observed
        and update_reached_oracle
        and cancel_requested
        and bool(cancelled)
        and not completed
        and VoiceEventType.INTERFACE_ORACLE_UPDATE in feedback_types
        and VoiceEventType.INTERFACE_ORACLE_CANCEL in feedback_types,
        "sidecar_control_job_id": job_id,
        "sidecar_control_update_observed": update_observed,
        "sidecar_control_update_reached_oracle": update_reached_oracle,
        "sidecar_control_cancel_requested": cancel_requested,
        "sidecar_control_cancelled": bool(cancelled),
        "sidecar_control_cancel_reason": str(cancelled[-1].payload.get("cancel_reason") or "") if cancelled else "",
        "sidecar_control_completed_after_cancel": completed,
        "sidecar_control_feedback_update_sent": VoiceEventType.INTERFACE_ORACLE_UPDATE in feedback_types,
        "sidecar_control_feedback_cancel_sent": VoiceEventType.INTERFACE_ORACLE_CANCEL in feedback_types,
    }


async def _run_external_frontend_bridge_smoke() -> dict[str, Any]:
    oracle = SmokeOracle()
    engine = SmokeEngine(oracle=oracle)
    recorder = EventRecorder(engine)
    await engine.start(
        RealtimeVoiceSessionConfig(
            session_id="voice-smoke-external-frontend",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            oracle_jobs={
                "enabled": True,
                "max_concurrent": 1,
                "queue_limit": 4,
                "speak_terminal_results": True,
                "shutdown_timeout_seconds": 0.01,
            },
            metadata={"transport": "voiceclaw"},
        )
    )
    collector = asyncio.create_task(recorder.run())

    await engine.receive_event(
        VoiceEvent(
            type=VoiceEventType.INTERFACE_ORACLE_REQUEST,
            session_id="voice-smoke-external-frontend",
            sequence=1,
            payload={
                "tool": "ask_brain",
                "tool_call_id": "voiceclaw-call-1",
                "provider": "voiceclaw",
                "turn_id": "voice-smoke-external-frontend:voiceclaw:1",
                "user_id": "jetha",
                "text": "prepare an external KAME handoff",
                "intent": "Prepare external KAME handoff",
                "transcript": "prepare an external kame handoff",
                "audio_segment_ref": "artifact://voiceclaw/turn-1.wav",
                "audio_time_range_ms": [100, 2100],
                "moshi_transcript_hypothesis": "prepare an external kame handoff",
                "auxiliary_transcript_hypotheses": [
                    {
                        "source": "moshi",
                        "text": "prepare an external kame handoff",
                        "confidence": 0.78,
                        "latency_ms": 140,
                    }
                ],
                "interface_already_said": "I'm preparing the handoff.",
                "conversation_summary": "The user is testing an external voice frontend.",
                "requested_response_style": {"spoken": True, "max_sentences": 1},
            },
        )
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.TOOL_RESULT
            and event.payload.get("accepted") is True
            and event.payload.get("tool") == "ask_brain"
            for event in events
        )
    )
    tool_result = next(
        event
        for event in recorder.events
        if event.type == VoiceEventType.TOOL_RESULT
        and event.payload.get("accepted") is True
        and event.payload.get("tool") == "ask_brain"
    )
    job_id = str(tool_result.payload.get("job_id") or "")
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_STARTED
            and event.payload.get("job_id") == job_id
            for event in events
        )
    )
    oracle.release("Prepare external KAME handoff")
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_COMPLETED
            and event.payload.get("job_id") == job_id
            for event in events
        )
    )
    status = await engine.get_oracle_job_status()
    await engine.close()
    collector.cancel()
    try:
        await collector
    except asyncio.CancelledError:
        pass

    request = oracle.requests[0] if oracle.requests else None
    metadata = request.to_metadata() if request is not None else {}
    metadata_text = json.dumps(metadata, sort_keys=True)
    status_jobs = status.get("jobs") if isinstance(status.get("jobs"), list) else []
    status_job = next((job for job in status_jobs if job.get("job_id") == job_id), {})
    accepted_observed = any(
        event.type == VoiceEventType.ORACLE_JOB_ACCEPTED
        and event.payload.get("job_id") == job_id
        for event in recorder.events
    )
    started_observed = any(
        event.type == VoiceEventType.ORACLE_JOB_STARTED
        and event.payload.get("job_id") == job_id
        for event in recorder.events
    )
    completed_observed = any(
        event.type == VoiceEventType.ORACLE_JOB_COMPLETED
        and event.payload.get("job_id") == job_id
        for event in recorder.events
    )
    completed_event = next(
        (
            event
            for event in recorder.events
            if event.type == VoiceEventType.ORACLE_JOB_COMPLETED
            and event.payload.get("job_id") == job_id
        ),
        None,
    )
    external_tool_call_id = str(tool_result.payload.get("tool_call_id") or "")
    completion_tool_call_id = (
        str(completed_event.payload.get("interface_tool_call_id") or "")
        if completed_event is not None
        else ""
    )
    status_tool_call_id = str(status_job.get("interface_tool_call_id") or "")
    terminal_correlation_observed = (
        bool(external_tool_call_id)
        and completion_tool_call_id == external_tool_call_id
        and status_tool_call_id == external_tool_call_id
    )
    direct_tool_authority_exposed = any(
        forbidden in metadata_text
        for forbidden in (
            '"tool_name"',
            '"arguments"',
            "stripe_link_purchase",
            "read_file",
        )
    )
    evidence_bundle_propagated = (
        request is not None
        and getattr(request, "audio_segment_ref", "") == "artifact://voiceclaw/turn-1.wav"
        and getattr(request, "audio_time_range_ms", ()) == (100, 2100)
        and bool(getattr(request, "auxiliary_transcript_hypotheses", ()))
        and getattr(request, "auxiliary_transcript_hypotheses", ())[0].get("source") == "moshi"
        and getattr(request, "auxiliary_transcript_hypotheses", ())[0].get("authority") == "hypothesis"
    )
    durable_session = RealtimeVoiceSession(
        RealtimeVoiceSessionConfig(session_id="voice-smoke-external-frontend"),
        engine=SmokeEngine(oracle=SmokeOracle()),
    )
    for event in recorder.events:
        durable_session._apply_server_event(event)
    durable_messages = durable_session.durable_messages()
    durable_records = durable_session.durable_oracle_records()
    durable_record_payloads = [
        record.get("payload") for record in durable_records if isinstance(record.get("payload"), dict)
    ]
    durable_oracle_text_absent = all("oracle_text" not in payload for payload in durable_record_payloads)
    durable_user_messages_empty = all(message.get("role") != "user" for message in durable_messages)
    durable_hypothesis_not_promoted = (
        durable_user_messages_empty
        and durable_oracle_text_absent
        and any(
            record.get("type") == VoiceEventType.ORACLE_JOB_COMPLETED.value
            and isinstance(record.get("payload"), dict)
            and record["payload"].get("evidence_authority", {}).get("oracle_text") == "reflex_hypothesis"
            and "result_summary" in record["payload"]
            for record in durable_records
        )
    )
    return {
        "ok": bool(job_id)
        and tool_result.payload.get("accepted") is True
        and accepted_observed
        and started_observed
        and completed_observed
        and request is not None
        and getattr(request, "source", "") == "voiceclaw"
        and getattr(request, "interface_input_source", "") == "ask_brain"
        and getattr(request, "oracle_text", "") == "Prepare external KAME handoff"
        and evidence_bundle_propagated
        and durable_hypothesis_not_promoted
        and terminal_correlation_observed
        and not direct_tool_authority_exposed
        and status_job.get("state") == "completed",
        "external_frontend_request_accepted": tool_result.payload.get("accepted") is True,
        "external_frontend_tool_result_observed": True,
        "external_frontend_job_id": job_id,
        "external_frontend_provider": str(tool_result.payload.get("provider") or ""),
        "external_frontend_tool": str(tool_result.payload.get("tool") or ""),
        "external_frontend_tool_call_id": external_tool_call_id,
        "external_frontend_completion_tool_call_id": completion_tool_call_id,
        "external_frontend_status_tool_call_id": status_tool_call_id,
        "external_frontend_terminal_correlation_observed": terminal_correlation_observed,
        "external_frontend_accepted_observed": accepted_observed,
        "external_frontend_started_observed": started_observed,
        "external_frontend_completion_observed": completed_observed,
        "external_frontend_status_state": str(status_job.get("state") or ""),
        "external_frontend_source_reached_oracle": getattr(request, "source", "") == "voiceclaw"
        if request is not None
        else False,
        "external_frontend_input_source": getattr(request, "interface_input_source", "")
        if request is not None
        else "",
        "external_frontend_oracle_text": getattr(request, "oracle_text", "") if request is not None else "",
        "external_frontend_evidence_bundle_propagated": evidence_bundle_propagated,
        "external_frontend_audio_segment_ref": getattr(request, "audio_segment_ref", "") if request is not None else "",
        "external_frontend_audio_time_range_ms": list(getattr(request, "audio_time_range_ms", ()))
        if request is not None
        else [],
        "external_frontend_auxiliary_transcript_hypotheses": [
            dict(item) for item in getattr(request, "auxiliary_transcript_hypotheses", ())
        ]
        if request is not None
        else [],
        "external_frontend_hypothesis_not_durable_oracle_text": durable_hypothesis_not_promoted,
        "external_frontend_durable_user_messages_empty": durable_user_messages_empty,
        "external_frontend_durable_oracle_text_absent": durable_oracle_text_absent,
        "external_frontend_durable_record_count": len(durable_records),
        "external_frontend_direct_tool_authority_exposed": direct_tool_authority_exposed,
        "external_frontend_metadata_keys": sorted(str(key) for key in metadata),
        "external_frontend_event_counts": {
            event_type.value: sum(event.type == event_type for event in recorder.events)
            for event_type in {
                VoiceEventType.TOOL_RESULT,
                VoiceEventType.ORACLE_JOB_ACCEPTED,
                VoiceEventType.ORACLE_JOB_STARTED,
                VoiceEventType.ORACLE_JOB_COMPLETED,
            }
        },
    }


async def _run_unpromoted_transcript_hypothesis_smoke() -> dict[str, Any]:
    oracle = SmokeOracle()
    engine = SmokeEngine(oracle=oracle)
    recorder = EventRecorder(engine)
    await engine.start(
        RealtimeVoiceSessionConfig(
            session_id="voice-smoke-unpromoted-hypothesis",
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

    await engine.receive_event(
        VoiceEvent(
            type=VoiceEventType.AUDIO_INPUT_CHUNK,
            session_id="voice-smoke-unpromoted-hypothesis",
            sequence=1,
            payload={
                "transcript": "run guarded task one",
                "intent": "Run guarded task one",
                "intent_source": "smoke_reflex",
                "route": "defer",
                "interface_already_said": "Starting guarded task one.",
                "end_of_utterance": True,
            },
        )
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_STARTED
            and event.payload.get("intent") == "Run guarded task one"
            for event in events
        )
    )
    await engine.receive_event(
        VoiceEvent(
            type=VoiceEventType.AUDIO_INPUT_CHUNK,
            session_id="voice-smoke-unpromoted-hypothesis",
            sequence=2,
            payload={
                "transcript": "run guarded task two",
                "intent": "Run guarded task two",
                "intent_source": "smoke_reflex",
                "route": "defer",
                "interface_already_said": "Starting guarded task two.",
                "end_of_utterance": True,
            },
        )
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_QUEUED
            and event.payload.get("intent") == "Run guarded task two"
            for event in events
        )
    )
    queued_job = next(
        event
        for event in recorder.events
        if event.type == VoiceEventType.ORACLE_JOB_QUEUED
        and event.payload.get("intent") == "Run guarded task two"
    )
    queued_job_id = str(queued_job.payload.get("job_id") or "")
    untrusted_text = "spend two hundred dollars and call my phone"
    await engine.receive_event(
        VoiceEvent(
            type=VoiceEventType.INTERFACE_ORACLE_UPDATE,
            session_id="voice-smoke-unpromoted-hypothesis",
            sequence=3,
            payload={
                "job_id": queued_job_id,
                "update_type": "interpreter_evidence",
                "source": "moshi",
                "transcript": untrusted_text,
                "transcript_confidence": 0.71,
                "reason": "attach unpromoted transcript hypothesis",
            },
        )
    )
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE
            and event.payload.get("job_id") == queued_job_id
            and event.payload.get("auxiliary_transcript_hypotheses_count") == 1
            for event in events
        )
    )
    oracle.release("Run guarded task one")
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_STARTED
            and event.payload.get("job_id") == queued_job_id
            for event in events
        )
    )
    oracle.release("Run guarded task two")
    await recorder.wait_for(
        lambda events: any(
            event.type == VoiceEventType.ORACLE_JOB_COMPLETED
            and event.payload.get("job_id") == queued_job_id
            for event in events
        )
    )
    await engine.close()
    collector.cancel()
    try:
        await collector
    except asyncio.CancelledError:
        pass

    request = next(
        (
            item
            for item in oracle.requests
            if str(getattr(item, "turn_id", "")) == "voice-smoke-unpromoted-hypothesis:2"
            or str(getattr(item, "intent", "")) == "Run guarded task two"
        ),
        None,
    )
    auxiliary = tuple(getattr(request, "auxiliary_transcript_hypotheses", ())) if request is not None else ()
    hypothesis = auxiliary[0] if auxiliary else {}
    update_event = next(
        (
            event
            for event in recorder.events
            if event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE
            and event.payload.get("job_id") == queued_job_id
        ),
        None,
    )
    oracle_text_preserved = (
        getattr(request, "oracle_text", "") == "Run guarded task two"
        if request is not None
        else False
    )
    transcript_preserved = (
        getattr(request, "transcript", "") == "run guarded task two"
        if request is not None
        else False
    )
    intent_preserved = (
        getattr(request, "intent", "") == "Run guarded task two"
        if request is not None
        else False
    )
    hypothesis_attached = (
        hypothesis.get("source") == "moshi"
        and hypothesis.get("text") == untrusted_text
        and hypothesis.get("authority") == "hypothesis"
        and hypothesis.get("confidence") == 0.71
    )
    promoted = (
        any(
            untrusted_text == str(getattr(request, field, "") or "")
            for field in ("oracle_text", "transcript", "intent")
        )
        if request is not None
        else False
    )
    return {
        "ok": request is not None
        and update_event is not None
        and oracle_text_preserved
        and transcript_preserved
        and intent_preserved
        and hypothesis_attached
        and not promoted,
        "unpromoted_hypothesis_smoke_ok": request is not None
        and update_event is not None
        and oracle_text_preserved
        and transcript_preserved
        and intent_preserved
        and hypothesis_attached
        and not promoted,
        "unpromoted_hypothesis_job_id": queued_job_id,
        "unpromoted_hypothesis_source": hypothesis.get("source", ""),
        "unpromoted_hypothesis_authority": hypothesis.get("authority", ""),
        "unpromoted_hypothesis_text": hypothesis.get("text", ""),
        "unpromoted_hypothesis_confidence": hypothesis.get("confidence"),
        "unpromoted_hypothesis_oracle_text_preserved": oracle_text_preserved,
        "unpromoted_hypothesis_transcript_preserved": transcript_preserved,
        "unpromoted_hypothesis_intent_preserved": intent_preserved,
        "unpromoted_hypothesis_attached": hypothesis_attached,
        "unpromoted_hypothesis_promoted": promoted,
        "unpromoted_hypothesis_update_observed": update_event is not None,
        "unpromoted_hypothesis_update_summary": str(
            (update_event.payload if update_event is not None else {}).get("latest_interpreter_evidence") or ""
        ),
    }


async def _run_audit_scalar_redaction_smoke() -> dict[str, Any]:
    """Prove oracle job JSONL audit rows redact scalar payload fields."""
    release = asyncio.Event()
    secret_prefix = "sk" + "_test_"
    live_prefix = "sk" + "_live_"
    result_secret = secret_prefix + "abcdefghijklmnopqrstuvwxyz"
    approval_secret = secret_prefix + "zyxwvutsrqponmlkjihgfedcba"
    approval_summary_secret = secret_prefix + "qwertyuiopasdfghjklzxcvbnm"
    live_secret = live_prefix + "abcdefghijklmnopqrstuvwxyz"

    async def runner(_job):
        await release.wait()
        return {
            "result_summary": (
                f"Created provider credential {result_secret} "
                "with Authorization: Bearer raw-token"
            ),
            "result_text": (
                "Full result includes provider_token=raw-provider-token "
                f"and {live_secret}"
            ),
        }

    request = KameOracleRequest(
        session_id="voice-smoke-audit",
        turn_id="turn:audit-scalar-redaction",
        source="discord_voice",
        user_id="42",
        intent="Provision voice provider",
        interface_already_said="I'm preparing the voice provider action.",
    )
    with tempfile.TemporaryDirectory(prefix="voiceops-audit-smoke-") as tmpdir:
        ledger_path = Path(tmpdir) / "voiceops-oracle-jobs.jsonl"
        manager = OracleJobManager(max_concurrent=1, runner=runner, audit_ledger_path=ledger_path)
        job = await manager.submit(request)
        await asyncio.sleep(0)
        await manager.mark_waiting_for_approval(
            job.job_id,
            reason=(
                "Approve Stripe spend with Authorization: Bearer approval-token "
                f"and {approval_secret}"
            ),
            approval={
                "approval_id": "approval-smoke-123",
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

    combined = json.dumps(rows, sort_keys=True)
    completed_rows = [row for row in rows if row.get("event_type") == VoiceEventType.ORACLE_JOB_COMPLETED.value]
    waiting_rows = [
        row for row in rows if row.get("event_type") == VoiceEventType.ORACLE_JOB_WAITING_FOR_APPROVAL.value
    ]
    raw_canaries_absent = all(
        canary not in combined
        for canary in (
            result_secret,
            approval_secret,
            approval_summary_secret,
            live_secret,
            "raw-token",
            "approval-token",
            "raw-provider-token",
        )
    )
    result_text_omitted = bool(completed_rows) and all(
        "result_text" not in (row.get("payload") or {}) for row in completed_rows
    )
    return {
        "ok": bool(rows)
        and bool(completed_rows)
        and bool(waiting_rows)
        and raw_canaries_absent
        and result_text_omitted
        and "Authorization: Bearer ***" in combined
        and "sk_tes" in combined,
        "audit_scalar_payload_redacted": raw_canaries_absent,
        "audit_scalar_secret_canary_checked": True,
        "audit_scalar_result_text_omitted": result_text_omitted,
        "audit_scalar_completed_event_seen": bool(completed_rows),
        "audit_scalar_waiting_event_seen": bool(waiting_rows),
        "audit_scalar_row_count": len(rows),
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
    status_turn_oracle_request_count_before = len(oracle.requests)
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
    status_turn_oracle_request_count_after = len(oracle.requests)

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
    approval_cancel_capacity_smoke = await _run_approval_cancel_capacity_smoke()
    terminal_result_policy_smoke = await _run_terminal_result_policy_smoke()
    sidecar_control_smoke = await _run_sidecar_control_smoke()
    external_frontend_bridge_smoke = await _run_external_frontend_bridge_smoke()
    unpromoted_hypothesis_smoke = await _run_unpromoted_transcript_hypothesis_smoke()
    audit_scalar_smoke = await _run_audit_scalar_redaction_smoke()

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

    def active_job_count_at(event_index: int) -> int:
        active: set[str] = set()
        for event in recorder.events[: event_index + 1]:
            job_id = str(event.payload.get("job_id") or "")
            if not job_id:
                continue
            if event.type == VoiceEventType.ORACLE_JOB_STARTED:
                active.add(job_id)
            elif event.type in {
                VoiceEventType.ORACLE_JOB_COMPLETED,
                VoiceEventType.ORACLE_JOB_FAILED,
                VoiceEventType.ORACLE_JOB_CANCELLED,
            }:
                active.discard(job_id)
        return len(active)

    local_commits = [
        event
        for event in recorder.events
        if event.type == VoiceEventType.ASSISTANT_COMMIT and event.payload.get("local_reply")
    ]
    can_hear_local_commits = [
        event
        for event in local_commits
        if event.payload.get("text") == "Yes, I can hear you."
    ]
    can_hear_local_commit_index = (
        recorder.events.index(can_hear_local_commits[-1])
        if can_hear_local_commits
        else -1
    )
    local_turn_active_job_count = (
        active_job_count_at(can_hear_local_commit_index)
        if can_hear_local_commit_index >= 0
        else 0
    )
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
    status_text = str(running_status_commits[-1].payload.get("text") or "") if running_status_commits else ""
    status_ordinal_labels = tuple(label for label in STATUS_ORDINAL_LABELS if label in status_text)
    status_ordinal_labels_visible = len(status_ordinal_labels) == len(STATUS_ORDINAL_LABELS)
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
        "kind": "async_oracle_smoke",
        "ok": (
            len(started) == 10
            and len(queued) == 1
            and scheduler_max_running == 4
            and oracle.max_running == 4
            and len(local_commits) >= 2
            and local_turn_active_job_count >= 1
            and playback_stop_did_not_cancel_jobs
            and bool(running_status_commits)
            and status_ordinal_labels_visible
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
            and durable_completed_jobs == len(completed)
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
            and approval_cancel_capacity_smoke["ok"]
            and terminal_result_policy_smoke["ok"]
            and sidecar_control_smoke["ok"]
            and external_frontend_bridge_smoke["ok"]
            and unpromoted_hypothesis_smoke["ok"]
            and audit_scalar_smoke["ok"]
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
        "queued_cancel_requested_observed": queued_cancel_smoke["queued_cancel_requested_observed"],
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
        "approval_cancel_capacity_smoke_ok": approval_cancel_capacity_smoke["ok"],
        "approval_cancel_waiting_observed": approval_cancel_capacity_smoke[
            "approval_cancel_waiting_observed"
        ],
        "approval_cancel_followup_queued": approval_cancel_capacity_smoke[
            "approval_cancel_followup_queued"
        ],
        "approval_cancel_requested_observed": approval_cancel_capacity_smoke[
            "approval_cancel_requested_observed"
        ],
        "approval_cancel_cancelled_observed": approval_cancel_capacity_smoke[
            "approval_cancel_cancelled_observed"
        ],
        "approval_cancel_late_output_attempted": approval_cancel_capacity_smoke[
            "approval_cancel_late_output_attempted"
        ],
        "approval_cancel_completed_after_cancel": approval_cancel_capacity_smoke[
            "approval_cancel_completed_after_cancel"
        ],
        "approval_cancel_late_result_spoken": approval_cancel_capacity_smoke[
            "approval_cancel_late_result_spoken"
        ],
        "approval_cancel_followup_started_before_cancel_drained": approval_cancel_capacity_smoke[
            "approval_cancel_followup_started_before_cancel_drained"
        ],
        "approval_cancel_followup_started_after_cancel": approval_cancel_capacity_smoke[
            "approval_cancel_followup_started_after_cancel"
        ],
        "approval_cancel_active_visible": approval_cancel_capacity_smoke[
            "approval_cancel_active_visible"
        ],
        "approval_cancel_misleading_running_capacity": approval_cancel_capacity_smoke[
            "approval_cancel_misleading_running_capacity"
        ],
        "approval_cancel_status_text": approval_cancel_capacity_smoke["approval_cancel_status_text"],
        "approval_cancel_max_concurrent": approval_cancel_capacity_smoke["approval_cancel_max_concurrent"],
        "terminal_result_policy_smoke_ok": terminal_result_policy_smoke["ok"],
        "terminal_result_auto_summarize_default": terminal_result_policy_smoke[
            "terminal_result_auto_summarize_default"
        ],
        "terminal_result_default_event_count": terminal_result_policy_smoke[
            "terminal_result_default_event_count"
        ],
        "terminal_result_default_spoken": terminal_result_policy_smoke["terminal_result_default_spoken"],
        "terminal_result_suppression_config": terminal_result_policy_smoke["terminal_result_suppression_config"],
        "terminal_result_suppressed": terminal_result_policy_smoke["terminal_result_suppressed"],
        "terminal_result_suppressed_event_observed": terminal_result_policy_smoke[
            "terminal_result_suppressed_event_observed"
        ],
        "terminal_result_suppressed_event_count": terminal_result_policy_smoke[
            "terminal_result_suppressed_event_count"
        ],
        "terminal_result_suppressed_reason": terminal_result_policy_smoke["terminal_result_suppressed_reason"],
        "terminal_result_suppressed_payload_clean": terminal_result_policy_smoke[
            "terminal_result_suppressed_payload_clean"
        ],
        "terminal_result_unsolicited_event_count": terminal_result_policy_smoke[
            "terminal_result_unsolicited_event_count"
        ],
        "terminal_result_unsolicited_spoken": terminal_result_policy_smoke["terminal_result_unsolicited_spoken"],
        "terminal_result_status_available": terminal_result_policy_smoke["terminal_result_status_available"],
        "terminal_result_status_text": terminal_result_policy_smoke["terminal_result_status_text"],
        "sidecar_control_smoke_ok": sidecar_control_smoke["ok"],
        "sidecar_control_job_id": sidecar_control_smoke["sidecar_control_job_id"],
        "sidecar_control_update_observed": sidecar_control_smoke["sidecar_control_update_observed"],
        "sidecar_control_update_reached_oracle": sidecar_control_smoke[
            "sidecar_control_update_reached_oracle"
        ],
        "sidecar_control_cancel_requested": sidecar_control_smoke["sidecar_control_cancel_requested"],
        "sidecar_control_cancelled": sidecar_control_smoke["sidecar_control_cancelled"],
        "sidecar_control_cancel_reason": sidecar_control_smoke["sidecar_control_cancel_reason"],
        "sidecar_control_completed_after_cancel": sidecar_control_smoke[
            "sidecar_control_completed_after_cancel"
        ],
        "sidecar_control_feedback_update_sent": sidecar_control_smoke[
            "sidecar_control_feedback_update_sent"
        ],
        "sidecar_control_feedback_cancel_sent": sidecar_control_smoke[
            "sidecar_control_feedback_cancel_sent"
        ],
        "external_frontend_bridge_smoke_ok": external_frontend_bridge_smoke["ok"],
        "external_frontend_request_accepted": external_frontend_bridge_smoke[
            "external_frontend_request_accepted"
        ],
        "external_frontend_tool_result_observed": external_frontend_bridge_smoke[
            "external_frontend_tool_result_observed"
        ],
        "external_frontend_job_id": external_frontend_bridge_smoke["external_frontend_job_id"],
        "external_frontend_provider": external_frontend_bridge_smoke["external_frontend_provider"],
        "external_frontend_tool": external_frontend_bridge_smoke["external_frontend_tool"],
        "external_frontend_tool_call_id": external_frontend_bridge_smoke[
            "external_frontend_tool_call_id"
        ],
        "external_frontend_completion_tool_call_id": external_frontend_bridge_smoke[
            "external_frontend_completion_tool_call_id"
        ],
        "external_frontend_status_tool_call_id": external_frontend_bridge_smoke[
            "external_frontend_status_tool_call_id"
        ],
        "external_frontend_terminal_correlation_observed": external_frontend_bridge_smoke[
            "external_frontend_terminal_correlation_observed"
        ],
        "external_frontend_accepted_observed": external_frontend_bridge_smoke[
            "external_frontend_accepted_observed"
        ],
        "external_frontend_started_observed": external_frontend_bridge_smoke[
            "external_frontend_started_observed"
        ],
        "external_frontend_completion_observed": external_frontend_bridge_smoke[
            "external_frontend_completion_observed"
        ],
        "external_frontend_status_state": external_frontend_bridge_smoke[
            "external_frontend_status_state"
        ],
        "external_frontend_source_reached_oracle": external_frontend_bridge_smoke[
            "external_frontend_source_reached_oracle"
        ],
        "external_frontend_input_source": external_frontend_bridge_smoke[
            "external_frontend_input_source"
        ],
        "external_frontend_oracle_text": external_frontend_bridge_smoke[
            "external_frontend_oracle_text"
        ],
        "external_frontend_evidence_bundle_propagated": external_frontend_bridge_smoke[
            "external_frontend_evidence_bundle_propagated"
        ],
        "external_frontend_audio_segment_ref": external_frontend_bridge_smoke[
            "external_frontend_audio_segment_ref"
        ],
        "external_frontend_audio_time_range_ms": external_frontend_bridge_smoke[
            "external_frontend_audio_time_range_ms"
        ],
        "external_frontend_auxiliary_transcript_hypotheses": external_frontend_bridge_smoke[
            "external_frontend_auxiliary_transcript_hypotheses"
        ],
        "external_frontend_hypothesis_not_durable_oracle_text": external_frontend_bridge_smoke[
            "external_frontend_hypothesis_not_durable_oracle_text"
        ],
        "external_frontend_durable_user_messages_empty": external_frontend_bridge_smoke[
            "external_frontend_durable_user_messages_empty"
        ],
        "external_frontend_durable_oracle_text_absent": external_frontend_bridge_smoke[
            "external_frontend_durable_oracle_text_absent"
        ],
        "external_frontend_durable_record_count": external_frontend_bridge_smoke[
            "external_frontend_durable_record_count"
        ],
        "external_frontend_direct_tool_authority_exposed": external_frontend_bridge_smoke[
            "external_frontend_direct_tool_authority_exposed"
        ],
        "external_frontend_event_counts": external_frontend_bridge_smoke[
            "external_frontend_event_counts"
        ],
        "unpromoted_hypothesis_smoke_ok": unpromoted_hypothesis_smoke["ok"],
        "unpromoted_hypothesis_job_id": unpromoted_hypothesis_smoke["unpromoted_hypothesis_job_id"],
        "unpromoted_hypothesis_source": unpromoted_hypothesis_smoke["unpromoted_hypothesis_source"],
        "unpromoted_hypothesis_authority": unpromoted_hypothesis_smoke["unpromoted_hypothesis_authority"],
        "unpromoted_hypothesis_text": unpromoted_hypothesis_smoke["unpromoted_hypothesis_text"],
        "unpromoted_hypothesis_confidence": unpromoted_hypothesis_smoke[
            "unpromoted_hypothesis_confidence"
        ],
        "unpromoted_hypothesis_oracle_text_preserved": unpromoted_hypothesis_smoke[
            "unpromoted_hypothesis_oracle_text_preserved"
        ],
        "unpromoted_hypothesis_transcript_preserved": unpromoted_hypothesis_smoke[
            "unpromoted_hypothesis_transcript_preserved"
        ],
        "unpromoted_hypothesis_intent_preserved": unpromoted_hypothesis_smoke[
            "unpromoted_hypothesis_intent_preserved"
        ],
        "unpromoted_hypothesis_attached": unpromoted_hypothesis_smoke[
            "unpromoted_hypothesis_attached"
        ],
        "unpromoted_hypothesis_promoted": unpromoted_hypothesis_smoke[
            "unpromoted_hypothesis_promoted"
        ],
        "unpromoted_hypothesis_update_observed": unpromoted_hypothesis_smoke[
            "unpromoted_hypothesis_update_observed"
        ],
        "unpromoted_hypothesis_update_summary": unpromoted_hypothesis_smoke[
            "unpromoted_hypothesis_update_summary"
        ],
        "audit_scalar_smoke_ok": audit_scalar_smoke["ok"],
        "audit_scalar_payload_redacted": audit_scalar_smoke["audit_scalar_payload_redacted"],
        "audit_scalar_secret_canary_checked": audit_scalar_smoke[
            "audit_scalar_secret_canary_checked"
        ],
        "audit_scalar_result_text_omitted": audit_scalar_smoke["audit_scalar_result_text_omitted"],
        "audit_scalar_completed_event_seen": audit_scalar_smoke["audit_scalar_completed_event_seen"],
        "audit_scalar_waiting_event_seen": audit_scalar_smoke["audit_scalar_waiting_event_seen"],
        "audit_scalar_row_count": audit_scalar_smoke["audit_scalar_row_count"],
        "local_turn_committed": bool(local_commits),
        "local_turn_during_running_jobs_observed": local_turn_active_job_count >= 1,
        "local_turn_active_job_count": local_turn_active_job_count,
        "playback_stop_committed": bool(stop_talking_commits),
        "playback_stop_jobs_still_running": playback_stop_jobs_still_running,
        "playback_stop_cancelled_jobs": playback_stop_cancelled_jobs,
        "playback_stop_does_not_cancel_jobs": playback_stop_did_not_cancel_jobs,
        "status_turn_committed": bool(running_status_commits),
        "status_ordinal_labels_visible": status_ordinal_labels_visible,
        "status_ordinal_labels": status_ordinal_labels,
        "status_turn_queued_visible": (
            bool(running_status_commits)
            and "1 queued" in str(running_status_commits[-1].payload.get("text") or "")
        ),
        "status_turn_no_oracle_request": (
            status_turn_oracle_request_count_before == status_turn_oracle_request_count_after
        ),
        "status_turn_oracle_request_count_before": status_turn_oracle_request_count_before,
        "status_turn_oracle_request_count_after": status_turn_oracle_request_count_after,
        "status_text": status_text,
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
                VoiceEventType.ORACLE_JOB_RESULT_SUPPRESSED,
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
