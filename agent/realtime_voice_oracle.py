"""Hermes oracle adapter for realtime voice."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import AsyncIterator, Mapping, Optional

from agent.realtime_voice import (
    TRANSCRIPT_METADATA_KEYS,
    TRANSCRIPT_METADATA_VALUE_RE,
    RealtimeVoiceSessionConfig,
)


class HermesRealtimeOracle:
    """Call the configured Hermes agent as the authoritative voice oracle."""

    def __init__(self, config: RealtimeVoiceSessionConfig):
        self.config = config
        self._active_agent = None
        self._active_lock = threading.Lock()

    async def answer(self, transcript: str) -> str:
        text = (transcript or "").strip()
        if not text:
            return ""

        return await asyncio.to_thread(self._answer_sync, text)

    async def stream_answer(self, transcript: str) -> AsyncIterator[str]:
        async for delta in self.stream_answer_with_metadata(transcript, {}):
            yield delta

    async def stream_answer_with_metadata(
        self,
        transcript: str,
        metadata: Mapping[str, object],
    ) -> AsyncIterator[str]:
        text = (transcript or "").strip()
        if not text:
            return

        loop = asyncio.get_running_loop()
        sentinel = object()
        queue: asyncio.Queue[object] = asyncio.Queue()

        def on_delta(delta: object) -> None:
            if delta:
                loop.call_soon_threadsafe(queue.put_nowait, str(delta))

        def run() -> None:
            try:
                self._answer_sync(text, stream_callback=on_delta, metadata=metadata)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, f"\n\n[realtime voice oracle error: {exc}]")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        task = asyncio.create_task(asyncio.to_thread(run))
        cancelled = False
        try:
            while True:
                delta = await queue.get()
                if delta is sentinel:
                    break
                yield str(delta)
        except asyncio.CancelledError:
            cancelled = True
            self.interrupt("Realtime voice turn interrupted")
            raise
        finally:
            if cancelled and not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def interrupt(self, message: str = "Realtime voice turn interrupted") -> None:
        with self._active_lock:
            agent = self._active_agent
        if agent is not None:
            with contextlib.suppress(Exception):
                agent.interrupt(message)

    def _answer_sync(
        self,
        transcript: str,
        stream_callback: Optional[callable] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> str:
        from run_agent import AIAgent

        agent = AIAgent(
            model=self.config.oracle_model or "",
            platform="desktop_voice",
            session_id=self.config.session_id,
        )
        prompt_metadata = dict(self.config.metadata or {})
        prompt_metadata.update(metadata or {})
        prompt = _voice_oracle_prompt(transcript, prompt_metadata)
        with self._active_lock:
            self._active_agent = agent
        try:
            result = agent.run_conversation(
                prompt,
                persist_user_message=transcript,
                stream_callback=stream_callback,
            )
            return str(result.get("final_response") or "").strip()
        finally:
            with self._active_lock:
                if self._active_agent is agent:
                    self._active_agent = None


def _voice_oracle_prompt(transcript: str, metadata: Mapping[str, object]) -> str:
    language_context = _voice_language_context(metadata)
    transport_context = _voice_transport_context(metadata)
    architecture_context = _voice_architecture_context(metadata)
    prompt = (
        "The user is speaking to Hermes in a realtime voice session. "
        "Answer naturally and concisely. Avoid exposing hidden reasoning, "
        "raw tool traces, JSON envelopes, or transcript metadata."
    )
    if transport_context:
        prompt += f"\n{transport_context}"
    if architecture_context:
        prompt += f"\n{architecture_context}"
    if language_context:
        prompt += (
            "\nPreserve the user's spoken language and script unless the user explicitly asks "
            f"for translation or a different language. Detected speech metadata: {language_context}."
        )
    return f"{prompt}\n\nUser said: {transcript}"


def _voice_language_context(metadata: Mapping[str, object]) -> str:
    parts = []
    for key in TRANSCRIPT_METADATA_KEYS:
        value = metadata.get(key)
        if not isinstance(value, str):
            continue
        token = value.strip()
        if TRANSCRIPT_METADATA_VALUE_RE.fullmatch(token):
            parts.append(f"{key}={token}")
    return ", ".join(parts)


def _voice_architecture_context(metadata: Mapping[str, object]) -> str:
    if metadata.get("voice_architecture") != "kame_frontend_oracle":
        return ""
    frontend_bits = []
    for key in ("frontend_provider", "frontend_model"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            frontend_bits.append(value.strip())
    oracle_model = metadata.get("oracle_model")
    oracle_label = oracle_model.strip() if isinstance(oracle_model, str) and oracle_model.strip() else "the configured Hermes model"
    frontend_label = f" ({' '.join(frontend_bits)})" if frontend_bits else ""
    return (
        "Voice architecture: a low-latency realtime frontend model handles live speech, "
        f"turn-taking, and spoken audio{frontend_label}. You are the Hermes backend oracle "
        f"for deeper reasoning, tools, memory, and durable agent work ({oracle_label}). "
        "Do not describe the frontend as a separate user-visible bot; answer as Hermes "
        "through the voice interface."
    )


def _voice_transport_context(metadata: Mapping[str, object]) -> str:
    transport = metadata.get("transport")
    if transport == "discord_voice":
        return (
            "You are in a live Discord voice channel. The user's speech has already been "
            "captured from that channel, and your spoken reply will be sent back to the "
            "same voice channel. Do not claim that you cannot hear, join, or speak in "
            "Discord voice unless the provided session state explicitly says voice is "
            "unavailable or degraded."
        )
    return (
        "The user's speech has already been captured from the live voice transport, "
        "and your reply may be spoken back through that transport. Do not deny live "
        "voice capability unless the session state explicitly says voice is unavailable."
    )


class NullRealtimeOracle:
    """Oracle used by tests and transcript-only sessions."""

    async def answer(self, transcript: str) -> str:
        return ""

    async def stream_answer(self, transcript: str) -> AsyncIterator[str]:
        if False:
            yield transcript

    def interrupt(self, message: str = "Realtime voice turn interrupted") -> None:
        return None
