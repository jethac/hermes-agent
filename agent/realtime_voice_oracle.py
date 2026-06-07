"""Hermes oracle adapter for realtime voice."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

from agent.realtime_voice import RealtimeVoiceSessionConfig


class HermesRealtimeOracle:
    """Call the configured Hermes agent as the authoritative voice oracle."""

    def __init__(self, config: RealtimeVoiceSessionConfig):
        self.config = config

    async def answer(self, transcript: str) -> str:
        text = (transcript or "").strip()
        if not text:
            return ""

        return await asyncio.to_thread(self._answer_sync, text)

    async def stream_answer(self, transcript: str) -> AsyncIterator[str]:
        text = (transcript or "").strip()
        if not text:
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        def run() -> None:
            try:
                self._answer_sync(
                    text,
                    stream_callback=lambda delta: loop.call_soon_threadsafe(queue.put_nowait, delta),
                )
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, f"\n\n[realtime voice oracle error: {exc}]")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        task = asyncio.create_task(asyncio.to_thread(run))
        try:
            while True:
                delta = await queue.get()
                if delta is None:
                    break
                if delta:
                    yield delta
        finally:
            await task

    def _answer_sync(self, transcript: str, stream_callback: Optional[callable] = None) -> str:
        from run_agent import AIAgent

        agent = AIAgent(
            model=self.config.oracle_model or "",
            platform="desktop_voice",
            session_id=self.config.session_id,
        )
        prompt = (
            "The user is speaking to Hermes in a realtime voice session. "
            "Answer naturally and concisely. Avoid exposing hidden reasoning, "
            "raw tool traces, JSON envelopes, or transcript metadata.\n\n"
            f"User said: {transcript}"
        )
        result = agent.run_conversation(
            prompt,
            persist_user_message=transcript,
            stream_callback=stream_callback,
        )
        return str(result.get("final_response") or "").strip()


class NullRealtimeOracle:
    """Oracle used by tests and transcript-only sessions."""

    async def answer(self, transcript: str) -> str:
        return ""

    async def stream_answer(self, transcript: str) -> AsyncIterator[str]:
        if False:
            yield transcript
