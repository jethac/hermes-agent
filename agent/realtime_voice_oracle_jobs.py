"""Async oracle job scheduling for KAME realtime voice sessions."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Mapping, Optional, Sequence

from agent.redact import redact_sensitive_text
from agent.realtime_voice_errors import sanitize_realtime_voice_error
from agent.realtime_voice_kame import KameOracleRequest
from agent.think_scrubber import StreamingThinkScrubber, strip_leading_reasoning_trace


logger = logging.getLogger(__name__)


class OracleJobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"


class OracleJobEventType(StrEnum):
    ACCEPTED = "oracle.job.accepted"
    QUEUED = "oracle.job.queued"
    STARTED = "oracle.job.started"
    INTERPRETER_EVIDENCE_ATTACHED = "oracle.job.interpreter_evidence_attached"
    INTERPRETER_EVIDENCE_LATE = "oracle.job.interpreter_evidence_late"
    PROGRESS = "oracle.job.progress"
    WAITING_FOR_APPROVAL = "oracle.job.waiting_for_approval"
    COMPLETED = "oracle.job.completed"
    FAILED = "oracle.job.failed"
    CANCEL_REQUESTED = "oracle.job.cancel_requested"
    CANCELLED = "oracle.job.cancelled"
    RESULT_SUPPRESSED = "oracle.job.result_suppressed"


TERMINAL_STATES = frozenset(
    {
        OracleJobState.COMPLETED,
        OracleJobState.FAILED,
        OracleJobState.CANCELLED,
    }
)


OracleJobRunner = Callable[["OracleJob"], Awaitable[Any]]
OracleJobEventCallback = Callable[["OracleJobEvent"], Any]
OracleJobInterruptCallback = Callable[["OracleJob", str], Any]


@dataclass
class OracleJob:
    job_id: str
    session_id: str
    created_at: float
    updated_at: float
    state: OracleJobState
    priority: str
    route: str
    oracle_text: str
    reflex_intent: str
    audio_segment_ref: str = ""
    audio_time_range_ms: tuple[int, int] | tuple[()] = field(default_factory=tuple)
    reflex_transcript_hypothesis: str = ""
    reflex_transcript_source: str = ""
    reflex_transcript_confidence: Optional[float] = None
    auxiliary_transcript_hypotheses: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    interpreter_corrected_transcript: str = ""
    interpreter_confidence: Optional[float] = None
    interpreter_entities: tuple[dict[str, str], ...] = field(default_factory=tuple)
    interpreter_disagreements: tuple[str, ...] = field(default_factory=tuple)
    interface_already_said: str = ""
    interface_tool_call_id: str = ""
    requested_response_style: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    result_text: str = ""
    error: str = ""
    cancel_reason: str = ""
    updates: list[dict[str, Any]] = field(default_factory=list)
    interpreter_evidence: list[dict[str, Any]] = field(default_factory=list)
    request: Optional[KameOracleRequest] = field(default=None, repr=False, compare=False)

    def to_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "job_id": self.job_id,
            "state": self.state.value,
            "priority": self.priority,
            "route": self.route,
            "intent": self.reflex_intent,
            "spoken_status": _spoken_status(self),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.result_summary:
            status["result_summary"] = self.result_summary
        if self.error:
            status["error"] = self.error
        if self.cancel_reason:
            status["cancel_reason"] = self.cancel_reason
        if self.interface_tool_call_id:
            status["interface_tool_call_id"] = self.interface_tool_call_id
        if self.audio_segment_ref:
            status["audio_segment_ref"] = self.audio_segment_ref
        if self.audio_time_range_ms:
            status["audio_time_range_ms"] = tuple(self.audio_time_range_ms)
        if self.reflex_transcript_hypothesis:
            status["reflex_transcript_hypothesis"] = self.reflex_transcript_hypothesis
            status["reflex_transcript_source"] = self.reflex_transcript_source or "reflex_audio"
        if self.reflex_transcript_confidence is not None:
            status["reflex_transcript_confidence"] = self.reflex_transcript_confidence
        if self.auxiliary_transcript_hypotheses:
            status["auxiliary_transcript_hypotheses_count"] = len(self.auxiliary_transcript_hypotheses)
            status["auxiliary_transcript_hypotheses"] = tuple(self.auxiliary_transcript_hypotheses)
        if self.interpreter_corrected_transcript:
            status["interpreter_corrected_transcript"] = self.interpreter_corrected_transcript
        if self.interpreter_confidence is not None:
            status["interpreter_confidence"] = self.interpreter_confidence
        if self.interpreter_entities:
            status["interpreter_entities"] = self.interpreter_entities
        if self.interpreter_disagreements:
            status["interpreter_disagreements"] = self.interpreter_disagreements
        if self.updates:
            status["update_count"] = len(self.updates)
            status["latest_update"] = str(self.updates[-1].get("text") or "")[:240]
        if self.interpreter_evidence:
            latest_evidence = self.interpreter_evidence[-1]
            status["interpreter_evidence_count"] = len(self.interpreter_evidence)
            status["latest_interpreter_evidence"] = str(latest_evidence.get("summary") or "")[:240]
            status["latest_interpreter_evidence_source"] = str(latest_evidence.get("source") or "")[:40]
            status["interpreter_evidence_late"] = bool(latest_evidence.get("late"))
            if "delivered_to_oracle" in latest_evidence:
                status["interpreter_evidence_delivered_to_oracle"] = bool(latest_evidence.get("delivered_to_oracle"))
            if "consumed_before_irreversible_action" in latest_evidence:
                status["interpreter_evidence_consumed_before_irreversible_action"] = bool(
                    latest_evidence.get("consumed_before_irreversible_action")
                )
            delivery_status = str(latest_evidence.get("delivery_status") or "").strip()
            if delivery_status:
                status["interpreter_evidence_delivery_status"] = delivery_status[:80]
        if self.request is not None:
            status["turn_id"] = self.request.turn_id
        return status


@dataclass(frozen=True)
class OracleJobEvent:
    type: OracleJobEventType
    job_id: str
    session_id: str
    state: OracleJobState
    payload: Mapping[str, Any] = field(default_factory=dict)
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_status(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "job_id": self.job_id,
            "session_id": self.session_id,
            "state": self.state.value,
            "timestamp_ms": self.timestamp_ms,
            "payload": dict(self.payload),
        }


class OracleJobQueueFullError(RuntimeError):
    """Raised when a new oracle job cannot be accepted."""


class OracleJobReprioritizationRequiredError(OracleJobQueueFullError):
    """Raised when capacity is full and the caller must reprioritize first."""


class OracleJobNotFoundError(KeyError):
    """Raised when a requested oracle job does not exist."""


class OracleJobManager:
    """Bounded async scheduler for KAME reflex-to-oracle work.

    The manager owns scheduling state only. It does not call tools directly and
    does not decide model routing. Callers provide an oracle runner that uses
    Hermes' existing oracle path.
    """

    def __init__(
        self,
        *,
        max_concurrent: int = 1,
        queue_limit: int = 16,
        default_priority: str = "normal",
        overflow_policy: str = "queue",
        runner: Optional[OracleJobRunner] = None,
        event_callback: Optional[OracleJobEventCallback] = None,
        interrupt_callback: Optional[OracleJobInterruptCallback] = None,
        audit_ledger_path: Optional[str | Path] = None,
        id_prefix: str = "voice-oracle",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.max_concurrent = max(1, int(max_concurrent or 1))
        self.queue_limit = max(0, int(queue_limit or 0))
        self.default_priority = str(default_priority or "normal")
        self.overflow_policy = str(overflow_policy or "queue").strip().lower() or "queue"
        self.runner = runner
        self.event_callback = event_callback
        self.interrupt_callback = interrupt_callback
        self.audit_ledger_path = _audit_ledger_path(audit_ledger_path)
        self.id_prefix = str(id_prefix or "voice-oracle")
        self._clock = clock
        self._jobs: dict[str, OracleJob] = {}
        self._queue: Deque[str] = deque()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._runners: dict[str, OracleJobRunner] = {}
        self._sequence = 0
        self._lock = asyncio.Lock()

    async def submit(
        self,
        request: KameOracleRequest,
        *,
        priority: Optional[str] = None,
        runner: Optional[OracleJobRunner] = None,
    ) -> OracleJob:
        if runner is None:
            runner = self.runner
        if runner is None:
            raise ValueError("OracleJobManager requires a runner to submit jobs")

        async with self._lock:
            active_count = self._active_count_locked()
            if self.overflow_policy == "reject" and active_count >= self.max_concurrent:
                raise OracleJobQueueFullError("oracle job queue is full")
            if self.overflow_policy == "reprioritize" and active_count >= self.max_concurrent:
                raise OracleJobReprioritizationRequiredError("oracle job reprioritization required")
            if self._queued_count_locked() >= self.queue_limit and active_count >= self.max_concurrent:
                raise OracleJobQueueFullError("oracle job queue is full")

            job = self._job_from_request(request, priority=priority)
            self._jobs[job.job_id] = job
            self._runners[job.job_id] = runner
            await self._emit_locked(OracleJobEventType.ACCEPTED, job)
            if active_count < self.max_concurrent:
                self._start_job_locked(job, runner)
            else:
                self._queue.append(job.job_id)
                self._sort_queue_locked()
                await self._emit_locked(OracleJobEventType.QUEUED, job)
            return job

    async def cancel(self, job_id: str, *, reason: str = "cancelled") -> OracleJob:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise OracleJobNotFoundError(job_id)
            if job.state in TERMINAL_STATES:
                return job
            reason = str(reason or "cancelled")
            job.cancel_reason = reason
            job.updated_at = self._clock()
            if job.state == OracleJobState.QUEUED:
                self._remove_queued_locked(job.job_id)
                job.state = OracleJobState.CANCEL_REQUESTED
                await self._emit_locked(OracleJobEventType.CANCEL_REQUESTED, job)
                job.updated_at = self._clock()
                job.state = OracleJobState.CANCELLED
                await self._emit_locked(OracleJobEventType.CANCELLED, job)
                self._runners.pop(job.job_id, None)
                self._start_available_locked()
                return job
            job.state = OracleJobState.CANCEL_REQUESTED
            await self._emit_locked(OracleJobEventType.CANCEL_REQUESTED, job)
            task = self._tasks.get(job.job_id)
            interrupt = self.interrupt_callback

        try:
            if interrupt is not None:
                await _maybe_await(interrupt(job, reason))
        except Exception as exc:
            logger.warning(
                "Realtime voice oracle interrupt callback failed for job %s: %s",
                job.job_id,
                sanitize_realtime_voice_error(exc),
            )
        finally:
            if task is not None:
                task.cancel()
        return job

    async def cancel_all(self, *, reason: str = "cancelled") -> list[OracleJob]:
        async with self._lock:
            job_ids = [
                job_id for job_id, job in self._jobs.items()
                if job.state not in TERMINAL_STATES
            ]
        cancelled = []
        for job_id in job_ids:
            cancelled.append(await self.cancel(job_id, reason=reason))
        return cancelled

    async def update_priority(self, job_id: str, *, priority: str) -> OracleJob:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise OracleJobNotFoundError(job_id)
            if job.state in TERMINAL_STATES:
                return job
            job.priority = _normalize_priority(priority, default=self.default_priority)
            job.updated_at = self._clock()
            if job.state == OracleJobState.QUEUED:
                self._sort_queue_locked()
            await self._emit_locked(
                OracleJobEventType.PROGRESS,
                job,
                payload={
                    **job.to_status(),
                    "operation": "priority",
                    "priority": job.priority,
                },
            )
            return job

    async def add_update(
        self,
        job_id: str,
        *,
        text: str,
        source: str = "user",
        update_type: str = "clarification",
    ) -> OracleJob:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise OracleJobNotFoundError(job_id)
            if job.state in TERMINAL_STATES:
                return job
            update_text = _compact_update_text(text)
            if not update_text:
                return job
            job.updates.append(
                {
                    "text": update_text,
                    "source": str(source or "user")[:40],
                    "type": str(update_type or "clarification")[:40],
                    "created_at": self._clock(),
                }
            )
            job.updated_at = self._clock()
            await self._emit_locked(
                OracleJobEventType.PROGRESS,
                job,
                payload={
                    **job.to_status(),
                    "operation": "update",
                    "latest_update": update_text,
                    "update_count": len(job.updates),
                },
            )
            return job

    async def add_interpreter_evidence(
        self,
        job_id: str,
        *,
        corrected_transcript: str = "",
        normalized_intent: str = "",
        audio_segment_ref: str = "",
        audio_time_range_ms: Optional[Sequence[Any]] = None,
        reflex_transcript_hypothesis: Any = None,
        auxiliary_transcript_hypotheses: Optional[Sequence[Mapping[str, Any]]] = None,
        entities: Optional[list[Mapping[str, Any]]] = None,
        confidence: Optional[float] = None,
        disagreements: Optional[list[str]] = None,
        source: str = "gemma_interpreter",
    ) -> OracleJob:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise OracleJobNotFoundError(job_id)
            if job.state in TERMINAL_STATES:
                return job

            evidence = _compact_interpreter_evidence(
                corrected_transcript=corrected_transcript,
                normalized_intent=normalized_intent,
                audio_segment_ref=audio_segment_ref,
                audio_time_range_ms=audio_time_range_ms,
                reflex_transcript_hypothesis=reflex_transcript_hypothesis,
                auxiliary_transcript_hypotheses=auxiliary_transcript_hypotheses,
                entities=entities,
                confidence=confidence,
                disagreements=disagreements,
                source=source,
                created_at=self._clock(),
                late=job.state != OracleJobState.QUEUED,
            )
            if not evidence:
                return job

            job.interpreter_evidence.append(evidence)
            _promote_interpreter_evidence(job, evidence)
            job.updated_at = self._clock()
            event_type = (
                OracleJobEventType.INTERPRETER_EVIDENCE_LATE
                if evidence.get("late")
                else OracleJobEventType.INTERPRETER_EVIDENCE_ATTACHED
            )
            await self._emit_locked(
                event_type,
                job,
                payload={
                    **job.to_status(),
                    "operation": "interpreter_evidence",
                    "interpreter_evidence_count": len(job.interpreter_evidence),
                    "latest_interpreter_evidence": evidence["summary"],
                    "interpreter_evidence_late": bool(evidence.get("late")),
                },
            )
            return job

    async def mark_latest_interpreter_evidence_delivery(
        self,
        job_id: str,
        *,
        delivered_to_oracle: bool,
        consumed_before_irreversible_action: bool = False,
        delivery_status: str = "",
    ) -> OracleJob:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise OracleJobNotFoundError(job_id)
            if not job.interpreter_evidence:
                return job
            latest = job.interpreter_evidence[-1]
            latest["delivered_to_oracle"] = bool(delivered_to_oracle)
            latest["consumed_before_irreversible_action"] = bool(consumed_before_irreversible_action)
            status = _compact_evidence_text(delivery_status, limit=80)
            if status:
                latest["delivery_status"] = status
            job.updated_at = self._clock()
            await self._emit_locked(
                OracleJobEventType.PROGRESS,
                job,
                payload={
                    **job.to_status(),
                    "operation": "interpreter_evidence_delivery",
                    "interpreter_evidence_count": len(job.interpreter_evidence),
                    "interpreter_evidence_delivered_to_oracle": bool(delivered_to_oracle),
                    "interpreter_evidence_consumed_before_irreversible_action": bool(
                        consumed_before_irreversible_action
                    ),
                    "interpreter_evidence_delivery_status": status,
                },
            )
            return job

    async def mark_waiting_for_approval(
        self,
        job_id: str,
        *,
        reason: str = "waiting for approval",
        approval: Optional[Mapping[str, Any]] = None,
    ) -> OracleJob:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise OracleJobNotFoundError(job_id)
            if job.state in TERMINAL_STATES or job.state == OracleJobState.CANCEL_REQUESTED:
                return job
            job.state = OracleJobState.WAITING_FOR_APPROVAL
            job.updated_at = self._clock()
            payload = dict(job.to_status())
            payload["approval_reason"] = str(reason or "waiting for approval")[:240]
            if approval:
                payload["approval"] = _compact_approval_payload(approval)
            await self._emit_locked(
                OracleJobEventType.WAITING_FOR_APPROVAL,
                job,
                payload=payload,
            )
            return job

    async def mark_running(self, job_id: str) -> OracleJob:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise OracleJobNotFoundError(job_id)
            if job.state in TERMINAL_STATES or job.state == OracleJobState.CANCEL_REQUESTED:
                return job
            if job.state != OracleJobState.RUNNING:
                job.state = OracleJobState.RUNNING
                job.updated_at = self._clock()
                await self._emit_locked(OracleJobEventType.STARTED, job)
            return job

    async def status_view(self) -> dict[str, Any]:
        async with self._lock:
            jobs = [job.to_status() for job in self._jobs.values()]
            active = self._active_count_locked()
            running = sum(1 for job in self._jobs.values() if job.state == OracleJobState.RUNNING)
            queued = sum(1 for job in self._jobs.values() if job.state == OracleJobState.QUEUED)
            waiting_for_approval = sum(
                1 for job in self._jobs.values() if job.state == OracleJobState.WAITING_FOR_APPROVAL
            )
            cancel_requested = sum(
                1 for job in self._jobs.values() if job.state == OracleJobState.CANCEL_REQUESTED
            )
        return {
            "capacity": {
                "active": active,
                "running": running,
                "max_concurrent": self.max_concurrent,
                "queued": queued,
                "queue_limit": self.queue_limit,
                "waiting_for_approval": waiting_for_approval,
                "cancel_requested": cancel_requested,
            },
            "jobs": jobs,
        }

    async def get(self, job_id: str) -> OracleJob:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise OracleJobNotFoundError(job_id)
            return job

    async def wait_for_idle(self) -> None:
        while True:
            async with self._lock:
                tasks = [task for task in self._tasks.values() if not task.done()]
                queued = bool(self._queue)
            if not tasks and not queued:
                return
            if tasks:
                await asyncio.wait(tasks, timeout=0.05)
            else:
                await asyncio.sleep(0.01)

    async def shutdown(self, *, reason: str = "session closed", timeout_seconds: float = 2.0) -> bool:
        """Cancel pending work without letting non-cooperative oracle jobs hang close()."""
        await self.cancel_all(reason=reason)
        try:
            await asyncio.wait_for(
                self.wait_for_idle(),
                timeout=_positive_float(timeout_seconds, default=2.0),
            )
            return True
        except TimeoutError:
            await self._force_cancel_remaining(reason=reason)
            return False

    def _job_from_request(self, request: KameOracleRequest, *, priority: Optional[str]) -> OracleJob:
        self._sequence += 1
        now = self._clock()
        job_id = f"{self.id_prefix}-{self._sequence:03d}"
        return OracleJob(
            job_id=job_id,
            session_id=request.session_id,
            created_at=now,
            updated_at=now,
            state=OracleJobState.QUEUED,
            priority=_normalize_priority(priority, default=self.default_priority),
            route=request.route.value,
            oracle_text=request.oracle_text,
            reflex_intent=request.intent,
            interface_already_said=request.interface_already_said,
            interface_tool_call_id=_compact_evidence_text(request.interface_tool_call_id, limit=160),
            audio_segment_ref=_compact_evidence_text(request.audio_segment_ref, limit=240),
            audio_time_range_ms=_audio_time_range_ms(request.audio_time_range_ms),
            reflex_transcript_hypothesis=_compact_evidence_text(
                request.reflex_transcript_hypothesis
                or (request.transcript if request.transcript_source == "reflex_audio" else ""),
                limit=500,
            ),
            reflex_transcript_source=_compact_evidence_text(
                request.reflex_transcript_source
                or ("reflex_audio" if request.reflex_transcript_hypothesis else ""),
                limit=40,
            ),
            reflex_transcript_confidence=_compact_confidence(
                request.reflex_transcript_confidence
                if request.reflex_transcript_confidence is not None
                else (request.transcript_confidence if request.transcript_source == "reflex_audio" else None)
            ),
            auxiliary_transcript_hypotheses=_compact_auxiliary_transcript_hypotheses(
                request.auxiliary_transcript_hypotheses
            ),
            requested_response_style=dict(request.requested_response_style or {}),
            metadata=request.to_metadata(),
            request=request,
        )

    def _start_job_locked(self, job: OracleJob, runner: OracleJobRunner) -> None:
        job.state = OracleJobState.RUNNING
        job.updated_at = self._clock()
        task = asyncio.create_task(self._run_job(job.job_id, runner))
        self._tasks[job.job_id] = task

    async def _run_job(self, job_id: str, runner: OracleJobRunner) -> None:
        async with self._lock:
            job = self._jobs[job_id]
            await self._emit_locked(OracleJobEventType.STARTED, job)
        try:
            result = await runner(job)
        except asyncio.CancelledError:
            async with self._lock:
                job = self._jobs[job_id]
                if job.state not in TERMINAL_STATES:
                    job.state = OracleJobState.CANCELLED
                    job.updated_at = self._clock()
                    await self._emit_locked(
                        OracleJobEventType.RESULT_SUPPRESSED,
                        job,
                        payload=_result_suppressed_payload(
                            job,
                            reason="cancelled_runner_interrupted",
                        ),
                    )
                    await self._emit_locked(OracleJobEventType.CANCELLED, job)
                self._tasks.pop(job_id, None)
                self._runners.pop(job_id, None)
                self._start_available_locked()
            return
        except Exception as exc:
            async with self._lock:
                job = self._jobs[job_id]
                if job.state not in TERMINAL_STATES:
                    job.state = OracleJobState.FAILED
                    job.error = sanitize_realtime_voice_error(exc)
                    job.updated_at = self._clock()
                    await self._emit_locked(OracleJobEventType.FAILED, job)
                self._tasks.pop(job_id, None)
                self._runners.pop(job_id, None)
                self._start_available_locked()
            return

        async with self._lock:
            job = self._jobs[job_id]
            if job.state in {OracleJobState.CANCEL_REQUESTED, OracleJobState.CANCELLED}:
                job.state = OracleJobState.CANCELLED
                job.result_summary = ""
                job.result_text = ""
                job.updated_at = self._clock()
                await self._emit_locked(
                    OracleJobEventType.RESULT_SUPPRESSED,
                    job,
                    payload=_result_suppressed_payload(
                        job,
                        reason="cancelled_job_returned_result",
                        result=result,
                    ),
                )
                await self._emit_locked(OracleJobEventType.CANCELLED, job)
            elif job.state not in TERMINAL_STATES:
                job.state = OracleJobState.COMPLETED
                job.result_summary = _result_summary(result)
                job.result_text = _result_text(result)
                job.updated_at = self._clock()
                await self._emit_locked(OracleJobEventType.COMPLETED, job, payload=_completion_payload(job))
            self._tasks.pop(job_id, None)
            self._runners.pop(job_id, None)
            self._start_available_locked()

    def _start_available_locked(self) -> None:
        while self._queue and self._active_count_locked() < self.max_concurrent:
            job_id = self._queue.popleft()
            job = self._jobs.get(job_id)
            if job is None or job.state != OracleJobState.QUEUED:
                continue
            runner = self._runners.get(job_id) or self.runner
            if runner is None:
                logger.warning("Realtime voice oracle job %s has no runner; leaving queued", job_id)
                self._queue.appendleft(job_id)
                return
            self._start_job_locked(job, runner)

    async def _emit_locked(
        self,
        event_type: OracleJobEventType,
        job: OracleJob,
        *,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> None:
        event = OracleJobEvent(
            type=event_type,
            job_id=job.job_id,
            session_id=job.session_id,
            state=job.state,
            payload=dict(payload) if payload is not None else job.to_status(),
        )
        if self.audit_ledger_path is not None:
            _append_audit_ledger_event(self.audit_ledger_path, event)
        callback = self.event_callback
        if callback is None:
            return
        await _maybe_await(callback(event))

    def _running_count_locked(self) -> int:
        return sum(1 for job in self._jobs.values() if job.state == OracleJobState.RUNNING)

    def _active_count_locked(self) -> int:
        return sum(
            1
            for job in self._jobs.values()
            if job.state
            in {
                OracleJobState.RUNNING,
                OracleJobState.WAITING_FOR_APPROVAL,
                OracleJobState.CANCEL_REQUESTED,
            }
        )

    def _queued_count_locked(self) -> int:
        return sum(1 for job in self._jobs.values() if job.state == OracleJobState.QUEUED)

    def _remove_queued_locked(self, job_id: str) -> None:
        self._queue = deque(existing for existing in self._queue if existing != job_id)

    def _sort_queue_locked(self) -> None:
        self._queue = deque(
            sorted(
                self._queue,
                key=lambda job_id: (
                    _priority_rank(self._jobs[job_id].priority),
                    self._jobs[job_id].created_at,
                    job_id,
                ),
            )
        )

    async def _force_cancel_remaining(self, *, reason: str) -> None:
        async with self._lock:
            queued_job_ids = list(self._queue)
            self._queue.clear()
            task_items = list(self._tasks.items())

            for job_id in queued_job_ids:
                job = self._jobs.get(job_id)
                if job is None or job.state in TERMINAL_STATES:
                    continue
                job.state = OracleJobState.CANCELLED
                job.cancel_reason = job.cancel_reason or reason
                job.updated_at = self._clock()
                await self._emit_locked(OracleJobEventType.CANCELLED, job)
                self._runners.pop(job_id, None)

            for job_id, task in task_items:
                job = self._jobs.get(job_id)
                if job is not None and job.state not in TERMINAL_STATES:
                    job.state = OracleJobState.CANCELLED
                    job.cancel_reason = job.cancel_reason or reason
                    job.updated_at = self._clock()
                    await self._emit_locked(OracleJobEventType.CANCELLED, job)
                if not task.done():
                    task.cancel()
                    task.add_done_callback(_consume_task_exception)
                self._tasks.pop(job_id, None)
                self._runners.pop(job_id, None)


def _spoken_status(job: OracleJob) -> str:
    if job.interface_already_said:
        return job.interface_already_said
    text = " ".join((job.reflex_intent or job.oracle_text or "").split())
    return text[:160]


def _result_summary(result: Any) -> str:
    if isinstance(result, Mapping):
        for key in ("result_summary", "summary", "text", "final_response"):
            value = result.get(key)
            if value:
                summary = _visible_result_summary(value)
                if summary:
                    return summary
        return ""
    return _visible_result_summary(result)


def _result_text(result: Any) -> str:
    if isinstance(result, Mapping):
        for key in ("result_text", "text", "final_response", "result_summary", "summary"):
            value = result.get(key)
            if value:
                text = _visible_result_text(value)
                if text.strip():
                    return text
        return ""
    text = _visible_result_text(result)
    return text if text.strip() else ""


def _visible_result_summary(value: Any) -> str:
    return " ".join(_visible_result_text(value).split())[:2000]


def _visible_result_text(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    without_leading_trace = strip_leading_reasoning_trace(text)
    text = without_leading_trace.lstrip() if without_leading_trace != text else without_leading_trace
    without_blocks = _strip_reasoning_blocks(text)
    text = without_blocks.lstrip() if without_blocks != text else without_blocks
    return redact_sensitive_text(text, force=True)


def _strip_reasoning_blocks(text: str) -> str:
    scrubber = StreamingThinkScrubber()
    return scrubber.feed(text) + scrubber.flush()


def _completion_payload(job: OracleJob) -> dict[str, Any]:
    payload = job.to_status()
    if job.result_text:
        payload["result_text"] = job.result_text
        payload["result_text_chars"] = len(job.result_text)
    return payload


def _result_suppressed_payload(
    job: OracleJob,
    *,
    reason: str,
    result: Any = None,
) -> dict[str, Any]:
    payload = job.to_status()
    payload["suppression_reason"] = str(reason or "result_suppressed")[:120]
    payload["result_suppressed"] = True
    payload["suppressed_result_present"] = result is not None
    return payload


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _consume_task_exception(task: asyncio.Task[Any]) -> None:
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass


def _positive_float(value: object, *, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return default
    return parsed


def _normalize_priority(value: object, *, default: str = "normal") -> str:
    text = str(value or default or "normal").strip().lower()
    aliases = {
        "urgent": "high",
        "highest": "high",
        "important": "high",
        "medium": "normal",
        "default": "normal",
        "background": "low",
    }
    text = aliases.get(text, text)
    if text not in {"high", "normal", "low"}:
        return _normalize_priority(default if value != default else "normal", default="normal")
    return text


def _priority_rank(priority: object) -> int:
    return {"high": 0, "normal": 1, "low": 2}.get(_normalize_priority(priority), 1)


def _compact_update_text(value: object) -> str:
    return " ".join(redact_sensitive_text(str(value or ""), force=True).split())[:500]


def _compact_interpreter_evidence(
    *,
    corrected_transcript: object,
    normalized_intent: object,
    audio_segment_ref: object,
    audio_time_range_ms: object,
    reflex_transcript_hypothesis: object,
    auxiliary_transcript_hypotheses: Optional[Sequence[Mapping[str, Any]]],
    entities: Optional[list[Mapping[str, Any]]],
    confidence: Optional[float],
    disagreements: Optional[list[str]],
    source: object,
    created_at: float,
    late: bool,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "source": _compact_evidence_text(source, limit=40) or "gemma_interpreter",
        "created_at": created_at,
        "late": bool(late),
    }
    transcript = _compact_evidence_text(corrected_transcript, limit=500)
    intent = _compact_evidence_text(normalized_intent, limit=300)
    compact_audio_ref = _compact_evidence_text(audio_segment_ref, limit=240)
    compact_audio_range = _audio_time_range_ms(audio_time_range_ms)
    compact_reflex_hypothesis = _compact_reflex_transcript_hypothesis(reflex_transcript_hypothesis)
    compact_auxiliary_hypotheses = _compact_auxiliary_transcript_hypotheses(
        auxiliary_transcript_hypotheses or []
    )
    compact_entities = _compact_interpreter_entities(entities or [])
    compact_disagreements = tuple(
        text
        for text in (
            _compact_evidence_text(disagreement, limit=180)
            for disagreement in (disagreements or [])
        )
        if text
    )[:6]

    if transcript:
        evidence["corrected_transcript"] = transcript
    if intent:
        evidence["normalized_intent"] = intent
    if compact_audio_ref:
        evidence["audio_segment_ref"] = compact_audio_ref
    if compact_audio_range:
        evidence["audio_time_range_ms"] = compact_audio_range
    if compact_reflex_hypothesis:
        evidence["reflex_transcript_hypothesis"] = compact_reflex_hypothesis
    if compact_auxiliary_hypotheses:
        evidence["auxiliary_transcript_hypotheses"] = compact_auxiliary_hypotheses
    if compact_entities:
        evidence["entities"] = compact_entities
    parsed_confidence = _compact_confidence(confidence)
    if parsed_confidence is not None:
        evidence["confidence"] = parsed_confidence
    if compact_disagreements:
        evidence["disagreements"] = compact_disagreements

    summary = _interpreter_evidence_summary(evidence)
    if not summary:
        return {}
    evidence["summary"] = summary
    return evidence


def _compact_reflex_transcript_hypothesis(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        text = _compact_evidence_text(
            value.get("text") or value.get("transcript") or value.get("hypothesis"),
            limit=500,
        )
        source = _compact_evidence_text(value.get("source") or value.get("provider"), limit=40) or "reflex_audio"
        confidence = _compact_confidence(value.get("confidence"))  # type: ignore[arg-type]
    else:
        text = _compact_evidence_text(value, limit=500)
        source = "reflex_audio"
        confidence = None
    if not text:
        return {}
    item: dict[str, Any] = {
        "source": source,
        "text": text,
        "authority": "hypothesis",
    }
    if confidence is not None:
        item["confidence"] = confidence
    return item


def _compact_evidence_text(value: object, *, limit: int) -> str:
    return " ".join(redact_sensitive_text(str(value or ""), force=True).split())[:limit]


def _compact_confidence(value: Optional[float]) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return 0.0
    if parsed > 1:
        return 1.0
    return round(parsed, 4)


def _audio_time_range_ms(value: object) -> tuple[int, int] | tuple[()]:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        return ()
    if len(value) != 2:
        return ()
    try:
        start = int(value[0])
        end = int(value[1])
    except (TypeError, ValueError):
        return ()
    if start < 0 or end < start:
        return ()
    return (start, end)


def _compact_auxiliary_transcript_hypotheses(
    values: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    compact: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for value in values[:5]:
        if not isinstance(value, Mapping):
            continue
        source = _compact_evidence_text(value.get("source") or value.get("provider"), limit=40) or "unknown"
        text = _compact_evidence_text(
            value.get("text") or value.get("transcript") or value.get("hypothesis"),
            limit=500,
        )
        if not text:
            continue
        dedupe_key = (source, text)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        item: dict[str, Any] = {
            "source": source,
            "text": text,
            "authority": "hypothesis",
        }
        confidence = _compact_confidence(value.get("confidence"))  # type: ignore[arg-type]
        if confidence is not None:
            item["confidence"] = confidence
        compact.append(item)
    return tuple(compact)


def _compact_interpreter_entities(values: list[Mapping[str, Any]]) -> tuple[dict[str, str], ...]:
    allowed = ("type", "value", "name", "role", "source")
    compact: list[dict[str, str]] = []
    for value in values[:12]:
        item: dict[str, str] = {}
        for key in allowed:
            if key not in value:
                continue
            text = _compact_evidence_text(value.get(key), limit=160)
            if text:
                item[key] = text
        if item:
            compact.append(item)
    return tuple(compact)


def _promote_interpreter_evidence(job: OracleJob, evidence: Mapping[str, Any]) -> None:
    audio_ref = _compact_evidence_text(evidence.get("audio_segment_ref"), limit=240)
    if audio_ref:
        job.audio_segment_ref = audio_ref
    audio_time_range = _audio_time_range_ms(evidence.get("audio_time_range_ms"))
    if audio_time_range:
        job.audio_time_range_ms = audio_time_range
    reflex_hypothesis = evidence.get("reflex_transcript_hypothesis")
    if isinstance(reflex_hypothesis, Mapping):
        text = _compact_evidence_text(
            reflex_hypothesis.get("text") or reflex_hypothesis.get("transcript") or reflex_hypothesis.get("hypothesis"),
            limit=500,
        )
        if text:
            job.reflex_transcript_hypothesis = text
            job.reflex_transcript_source = _compact_evidence_text(
                reflex_hypothesis.get("source") or reflex_hypothesis.get("provider"),
                limit=40,
            ) or "reflex_audio"
            job.reflex_transcript_confidence = _compact_confidence(reflex_hypothesis.get("confidence"))  # type: ignore[arg-type]
    auxiliary = evidence.get("auxiliary_transcript_hypotheses")
    if isinstance(auxiliary, tuple):
        job.auxiliary_transcript_hypotheses = _compact_auxiliary_transcript_hypotheses(
            [item for item in auxiliary if isinstance(item, Mapping)]
        )
    transcript = _compact_evidence_text(evidence.get("corrected_transcript"), limit=500)
    if transcript:
        job.interpreter_corrected_transcript = transcript
    confidence = _compact_confidence(evidence.get("confidence"))  # type: ignore[arg-type]
    if confidence is not None:
        job.interpreter_confidence = confidence
    entities = evidence.get("entities")
    if isinstance(entities, tuple):
        job.interpreter_entities = _compact_interpreter_entities(
            [entity for entity in entities if isinstance(entity, Mapping)]
        )
    disagreements = evidence.get("disagreements")
    if isinstance(disagreements, tuple):
        job.interpreter_disagreements = tuple(
            text
            for text in (
                _compact_evidence_text(disagreement, limit=180)
                for disagreement in disagreements
            )
            if text
        )[:6]


def _interpreter_evidence_summary(evidence: Mapping[str, Any]) -> str:
    parts: list[str] = []
    transcript = str(evidence.get("corrected_transcript") or "").strip()
    intent = str(evidence.get("normalized_intent") or "").strip()
    if transcript:
        parts.append(f"transcript={transcript}")
    if intent:
        parts.append(f"intent={intent}")
    if evidence.get("audio_segment_ref"):
        parts.append("audio=attached")
    if evidence.get("reflex_transcript_hypothesis"):
        parts.append("reflex_hypothesis=attached")
    auxiliary = evidence.get("auxiliary_transcript_hypotheses")
    if isinstance(auxiliary, tuple) and auxiliary:
        parts.append(f"auxiliary_hypotheses={len(auxiliary)}")
    entities = evidence.get("entities")
    if isinstance(entities, tuple) and entities:
        rendered_entities = []
        for entity in entities[:4]:
            if not isinstance(entity, Mapping):
                continue
            entity_type = str(entity.get("type") or "entity").strip()
            entity_value = str(entity.get("value") or entity.get("name") or "").strip()
            if entity_value:
                rendered_entities.append(f"{entity_type}={entity_value}")
        if rendered_entities:
            parts.append(f"entities={', '.join(rendered_entities)}")
    if evidence.get("confidence") is not None:
        parts.append(f"confidence={evidence['confidence']}")
    disagreements = evidence.get("disagreements")
    if isinstance(disagreements, tuple) and disagreements:
        parts.append(f"disagreements={'; '.join(str(item) for item in disagreements[:3])}")
    if not parts:
        return ""
    return "interpreter evidence: " + "; ".join(parts)[:500]


def _compact_approval_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "approval_id",
        "approval_kind",
        "approval_reason",
        "status",
        "tool_name",
        "tool_call_id",
        "summary",
        "message",
        "reason",
        "latest_interpreter_evidence",
        "latest_interpreter_evidence_source",
        "interpreter_evidence_count",
        "interpreter_evidence_late",
        "interpreter_evidence_delivered_to_oracle",
        "interpreter_evidence_consumed_before_irreversible_action",
        "interpreter_evidence_delivery_status",
    }
    compact: dict[str, Any] = {}
    for key, raw in value.items():
        text_key = str(key)
        if text_key not in allowed:
            continue
        if isinstance(raw, bool):
            compact[text_key] = raw
        elif isinstance(raw, (int, float)):
            compact[text_key] = raw
        elif raw is not None:
            compact[text_key] = " ".join(redact_sensitive_text(str(raw), force=True).split())[:240]
    return compact


def _audit_ledger_path(value: Optional[str | Path]) -> Optional[Path]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser()


def _append_audit_ledger_event(path: Path, event: OracleJobEvent) -> None:
    row = {
        "schema_version": "voiceops.oracle_job_audit_event.v1",
        "action": "oracle_job_event",
        "event_type": event.type.value,
        "job_id": event.job_id,
        "session_id": event.session_id,
        "state": event.state.value,
        "timestamp_ms": event.timestamp_ms,
        "payload": _compact_audit_payload(event.payload),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    except OSError:
        # The audit hook must not break the live voice loop. External readiness
        # checks validate the ledger file when it is required as evidence.
        return


def _compact_audit_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    blocked = {"metadata", "oracle_text", "result_text", "requested_response_style"}
    compact: dict[str, Any] = {}
    for key, raw in payload.items():
        text_key = str(key)
        if text_key in blocked:
            continue
        if text_key == "approval" and isinstance(raw, Mapping):
            compact[text_key] = _compact_approval_payload(raw)
        elif isinstance(raw, bool):
            compact[text_key] = raw
        elif isinstance(raw, int) and not isinstance(raw, bool):
            compact[text_key] = raw
        elif isinstance(raw, float):
            compact[text_key] = raw
        elif raw is None:
            continue
        elif isinstance(raw, Mapping):
            compact[text_key] = _compact_audit_payload(raw)
        elif isinstance(raw, list):
            compact[text_key] = [
                _compact_audit_payload(item) if isinstance(item, Mapping) else _compact_audit_scalar(item)
                for item in raw[:20]
            ]
        else:
            compact[text_key] = _compact_audit_scalar(raw)
    return compact


def _compact_audit_scalar(value: Any) -> str:
    return " ".join(redact_sensitive_text(str(value or ""), force=True).split())[:2000]
