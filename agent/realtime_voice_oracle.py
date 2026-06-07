"""Hermes oracle adapter for realtime voice."""

from __future__ import annotations

import asyncio
from typing import Optional

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

    def _answer_sync(self, transcript: str) -> str:
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
        result = agent.run_conversation(prompt, persist_user_message=transcript)
        return str(result.get("final_response") or "").strip()


class NullRealtimeOracle:
    """Oracle used by tests and transcript-only sessions."""

    async def answer(self, transcript: str) -> str:
        return ""
