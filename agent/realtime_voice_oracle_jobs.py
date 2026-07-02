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
from typing import Any, Awaitable, Callable, Deque, Mapping, Optional

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
    PROGRESS = "oracle.job.progress"
    WAITING_FOR_APPROVAL = "oracle.job.waiting_for_approval"
    COMPLETED = "oracle.job.completed"
    FAILED = "oracle.job.failed"
    CANCEL_REQUESTED = "oracle.job.cancel_requested"
    CANCELLED = "oracle.job.cancelled"


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
    interface_already_said: str = ""
    requested_response_style: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    result_text: str = ""
    error: str = ""
    cancel_reason: str = ""
    updates: list[dict[str, Any]] = field(default_factory=list)
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
        if self.updates:
            status["update_count"] = len(self.updates)
            status["latest_update"] = str(self.updates[-1].get("text") or "")[:240]
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
            if self._queued_count_locked() >= self.queue_limit and active_count >= self.max_concurrent:
                raise OracleJobQueueFullError("oracle job queue is full")

            job = self._job_from_request(request, priority=priority)
            self._jobs[job.job_id] = job
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
                job.state = OracleJobState.CANCELLED
                await self._emit_locked(OracleJobEventType.CANCELLED, job)
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
                    await self._emit_locked(OracleJobEventType.CANCELLED, job)
                self._tasks.pop(job_id, None)
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
                self._start_available_locked()
            return

        async with self._lock:
            job = self._jobs[job_id]
            if job.state in {OracleJobState.CANCEL_REQUESTED, OracleJobState.CANCELLED}:
                job.state = OracleJobState.CANCELLED
                job.result_summary = ""
                job.result_text = ""
                job.updated_at = self._clock()
                await self._emit_locked(OracleJobEventType.CANCELLED, job)
            elif job.state not in TERMINAL_STATES:
                job.state = OracleJobState.COMPLETED
                job.result_summary = _result_summary(result)
                job.result_text = _result_text(result)
                job.updated_at = self._clock()
                await self._emit_locked(OracleJobEventType.COMPLETED, job, payload=_completion_payload(job))
            self._tasks.pop(job_id, None)
            self._start_available_locked()

    def _start_available_locked(self) -> None:
        runner = self.runner
        if runner is None:
            return
        while self._queue and self._active_count_locked() < self.max_concurrent:
            job_id = self._queue.popleft()
            job = self._jobs.get(job_id)
            if job is None or job.state != OracleJobState.QUEUED:
                continue
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
    return text


def _strip_reasoning_blocks(text: str) -> str:
    scrubber = StreamingThinkScrubber()
    return scrubber.feed(text) + scrubber.flush()


def _completion_payload(job: OracleJob) -> dict[str, Any]:
    payload = job.to_status()
    if job.result_text:
        payload["result_text"] = job.result_text
        payload["result_text_chars"] = len(job.result_text)
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
    return " ".join(str(value or "").split())[:500]


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
            compact[text_key] = " ".join(str(raw).split())[:240]
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
    return " ".join(str(value or "").split())[:2000]
