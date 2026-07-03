"""Hermes oracle adapter for realtime voice."""

from __future__ import annotations

import asyncio
import contextlib
import re
import threading
from typing import Any, AsyncIterator, Mapping, Optional

from agent.realtime_voice import (
    TRANSCRIPT_METADATA_KEYS,
    TRANSCRIPT_METADATA_VALUE_RE,
    RealtimeVoiceSessionConfig,
)
from agent.realtime_voice_kame import KameOracleRequest


class HermesRealtimeOracle:
    """Call the configured Hermes agent as the authoritative voice oracle."""

    def __init__(self, config: RealtimeVoiceSessionConfig):
        self.config = config
        self._active_agent = None
        self._active_agents: dict[str, Any] = {}
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
                self._answer_sync(
                    text,
                    stream_callback=on_delta,
                    metadata=metadata,
                    active_keys=_active_agent_keys(metadata),
                )
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

    async def stream_answer_for_request(self, request: KameOracleRequest) -> AsyncIterator[Any]:
        """Stream an answer for a structured KAME reflex-to-oracle request."""

        async for delta in self.stream_answer_with_metadata(request.oracle_text, request.to_metadata()):
            yield delta

    def interrupt(self, message: str = "Realtime voice turn interrupted") -> None:
        with self._active_lock:
            agents = list({id(agent): agent for agent in self._active_agents.values()}.values())
            if not agents and self._active_agent is not None:
                agents = [self._active_agent]
        for agent in agents:
            with contextlib.suppress(Exception):
                agent.interrupt(message)

    def interrupt_request(
        self,
        request: KameOracleRequest,
        message: str = "Realtime voice turn interrupted",
    ) -> None:
        keys = _active_agent_keys(request.to_metadata())
        with self._active_lock:
            agents = [
                self._active_agents[key]
                for key in keys
                if key in self._active_agents
            ]
            agents = list({id(agent): agent for agent in agents}.values())
        if not agents:
            return self.interrupt(message)
        for agent in agents:
            with contextlib.suppress(Exception):
                agent.interrupt(message)

    def _answer_sync(
        self,
        transcript: str,
        stream_callback: Optional[callable] = None,
        metadata: Optional[Mapping[str, object]] = None,
        active_keys: Optional[tuple[str, ...]] = None,
    ) -> str:
        from run_agent import AIAgent
        from tools.tool_search import tool_search_config_override

        prompt_metadata = dict(self.config.metadata or {})
        prompt_metadata.update(metadata or {})
        enabled_toolsets = _voice_oracle_enabled_toolsets(
            transcript,
            prompt_metadata,
            self.config,
        )
        tool_search_config = _voice_oracle_tool_search_config(self.config)
        agent_kwargs: dict[str, Any] = dict(
            model="",
            platform="desktop_voice",
            session_id=self.config.session_id,
        )
        if enabled_toolsets is not None:
            agent_kwargs["enabled_toolsets"] = enabled_toolsets
        config_context = (
            tool_search_config_override(tool_search_config)
            if tool_search_config is not None
            else contextlib.nullcontext()
        )
        with config_context:
            agent = AIAgent(**agent_kwargs)
            prompt = _voice_oracle_prompt(transcript, prompt_metadata)
            active_keys = tuple(active_keys or _active_agent_keys(prompt_metadata))
            with self._active_lock:
                for key in active_keys:
                    self._active_agents[key] = agent
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
                    for key in active_keys:
                        if self._active_agents.get(key) is agent:
                            self._active_agents.pop(key, None)


def _voice_oracle_tool_search_config(config: RealtimeVoiceSessionConfig) -> Optional[Mapping[str, Any]]:
    router = config.oracle_tool_router if isinstance(config.oracle_tool_router, Mapping) else {}
    if _metadata_bool(router.get("enabled"), default=True) is False:
        return None

    raw = router.get("tool_search")
    if raw is False or raw is None:
        return {"enabled": "on", "defer_core": "all"}
    if isinstance(raw, Mapping):
        return raw
    token = str(raw).strip().lower()
    if token in {"off", "false", "0", "no", "disabled"}:
        return {"enabled": "off", "defer_core": "off"}
    return {"enabled": "on", "defer_core": "all"}


def _voice_oracle_enabled_toolsets(
    transcript: str,
    metadata: Mapping[str, object],
    config: RealtimeVoiceSessionConfig,
) -> Optional[list[str]]:
    router = config.oracle_tool_router if isinstance(config.oracle_tool_router, Mapping) else {}
    if _metadata_bool(router.get("enabled"), default=True) is False:
        return None

    mode = str(router.get("mode") or "deterministic").strip().lower()
    if mode not in {"", "deterministic"}:
        return _toolset_list_or_none(router.get("default_toolsets"))

    if _looks_like_voiceops_request(transcript, metadata):
        return _toolset_list_or_none(router.get("voiceops_toolsets")) or ["voiceops"]
    return _toolset_list_or_none(router.get("default_toolsets"))


def _looks_like_voiceops_request(transcript: str, metadata: Mapping[str, object]) -> bool:
    haystack = " ".join(
        part
        for part in (
            transcript,
            _metadata_text(metadata.get("kame_intent")),
            _metadata_text(metadata.get("kame_transcript")),
            _metadata_text(metadata.get("kame_asr_transcript")),
            _metadata_text(metadata.get("kame_oracle_text_source")),
        )
        if part
    )
    text = re.sub(r"\s+", " ", haystack.casefold()).strip()
    if not text:
        return False
    if "voiceops" in text:
        return True

    money_or_commerce = _contains_any(
        text,
        (
            "stripe",
            "spending money",
            "spending budget",
            "budget",
            "pay for",
            "payment",
            "payments",
            "purchase",
            "buy ",
            "buying",
            "spend",
            "credit card",
            "checkout",
            "subscription",
            "invoice",
        ),
    )
    provisioning = _contains_any(
        text,
        (
            "provision",
            "set up",
            "setup",
            "sign up",
            "create an account",
            "open an account",
            "saas",
            "provider account",
            "voip",
            "twilio",
            "telnyx",
            "plivo",
            "sip",
            "outbound call",
            "phone",
            "call my phone",
            "call me",
        ),
    )
    if money_or_commerce and provisioning:
        return True
    return _contains_any(text, ("provision", "set up", "setup", "sign up")) and _contains_any(
        text,
        ("voip", "phone provider", "phone service", "outbound call", "call my phone"),
    )


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _toolset_list_or_none(raw: object) -> Optional[list[str]]:
    if raw is None or raw is False:
        return None
    if isinstance(raw, str):
        items = re.split(r"[, ]+", raw)
    elif isinstance(raw, (list, tuple, set)):
        items = [str(item) for item in raw]
    else:
        return None
    toolsets = [item.strip() for item in items if item and item.strip()]
    return list(dict.fromkeys(toolsets)) or None


def _metadata_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on", "enabled"}:
            return True
        if token in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


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
    kame_context = _voice_kame_request_context(metadata)
    if kame_context:
        prompt += f"\n{kame_context}"
    return f"{prompt}\n\nUser said: {transcript}"


def _active_agent_keys(metadata: Mapping[str, object]) -> tuple[str, ...]:
    keys = []
    for key in ("kame_turn_id", "kame_cancellation_token"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            keys.append(f"{key}:{value.strip()}")
    return tuple(dict.fromkeys(keys))


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
    frontend_label = f" ({' '.join(frontend_bits)})" if frontend_bits else ""
    return (
        "Voice architecture: a low-latency realtime frontend model handles live speech, "
        f"turn-taking, and spoken audio{frontend_label}. You are the Hermes backend oracle "
        "for deeper reasoning, tools, memory, and durable agent work using the active Hermes model. "
        "Do not describe the frontend as a separate user-visible bot; answer as Hermes "
        "through the voice interface."
    )


def _voice_kame_request_context(metadata: Mapping[str, object]) -> str:
    if metadata.get("voice_architecture") != "kame_frontend_oracle":
        return ""

    intent = _metadata_text(metadata.get("kame_intent"))
    transcript = _metadata_text(metadata.get("kame_transcript"))
    transcript_source = _metadata_text(metadata.get("kame_transcript_source"))
    asr_transcript = _metadata_text(metadata.get("kame_asr_transcript"))
    asr_transcript_source = _metadata_text(metadata.get("kame_asr_transcript_source"))
    audio_segment_ref = _metadata_text(metadata.get("kame_audio_segment_ref"))
    audio_time_range_ms = _metadata_time_range(metadata.get("kame_audio_time_range_ms"))
    auxiliary_transcript_hypotheses = _metadata_transcript_hypotheses(
        metadata.get("kame_auxiliary_transcript_hypotheses")
    )
    intent_source = _metadata_text(metadata.get("kame_intent_source"))
    route = _metadata_text(metadata.get("kame_route"))
    route_confidence = _metadata_float(metadata.get("kame_route_confidence"))
    reflex_validation_error = _metadata_text(metadata.get("kame_reflex_validation_error"))
    interface_already_said = _metadata_text(metadata.get("kame_interface_already_said"))
    summary = _metadata_text(metadata.get("kame_conversation_summary"))
    job_updates = _metadata_text_sequence(metadata.get("kame_job_updates"))
    oracle_text_source = _metadata_text(metadata.get("kame_oracle_text_source"))
    evidence_authority = _metadata_evidence_authority(metadata.get("kame_evidence_authority"))
    interface_input_source = _metadata_text(metadata.get("kame_interface_input_source"))
    interface_audio_input_fallback = metadata.get("kame_interface_audio_input_fallback") is True
    response_style = _metadata_response_style(metadata.get("kame_requested_response_style"))

    parts = [
        "KAME request: the realtime reflex has already handled live turn-taking "
        "and is escalating this turn to the Hermes oracle.",
    ]
    if intent:
        parts.append(f"Reflex interpreted intent ({intent_source or 'reflex'}): {intent}")
    if route:
        if route_confidence is None:
            parts.append(f"Reflex route: {route}.")
        else:
            parts.append(f"Reflex route: {route} (confidence {route_confidence:.2f}).")
    if interface_audio_input_fallback:
        source_label = interface_input_source or "ASR fallback"
        parts.append(
            f"The audio-native reflex was unavailable; this turn used {source_label} as the interface fallback."
        )
    if reflex_validation_error:
        parts.append(f"Reflex route override: {reflex_validation_error}.")
    transcript_source_is_asr = transcript_source.lower().startswith("asr")
    if transcript and not transcript_source_is_asr:
        parts.append(f"Reflex transcript hypothesis ({transcript_source or 'reflex_audio'}): {transcript}")
    if audio_segment_ref:
        if audio_time_range_ms:
            parts.append(
                f"Raw audio evidence ref: {audio_segment_ref} ({audio_time_range_ms[0]}-{audio_time_range_ms[1]} ms). "
                "Treat the raw audio/interpreter evidence as higher authority than transcript hypotheses."
            )
        else:
            parts.append(
                f"Raw audio evidence ref: {audio_segment_ref}. "
                "Treat the raw audio/interpreter evidence as higher authority than transcript hypotheses."
            )
    for hypothesis in auxiliary_transcript_hypotheses:
        source = hypothesis.get("source") or "unknown"
        text = hypothesis.get("text") or ""
        confidence = hypothesis.get("confidence")
        confidence_text = f", confidence {confidence:.2f}" if isinstance(confidence, float) else ""
        latency = hypothesis.get("latency_ms")
        latency_text = f", latency {latency} ms" if isinstance(latency, int) else ""
        parts.append(
            f"Auxiliary transcript hypothesis ({source}{confidence_text}{latency_text}): {text}. "
            "Use it only as labeled evidence; do not treat it as durable truth unless it agrees with interpreter/oracle judgment."
        )
    if not asr_transcript and transcript and transcript_source_is_asr:
        asr_transcript = transcript
        asr_transcript_source = transcript_source
    if asr_transcript:
        source_label = asr_transcript_source or "asr"
        parts.append(
            f"Verbatim ASR evidence ({source_label}): {asr_transcript}. "
            "Compare it against the reflex intent, transcript hypotheses, and interpreter evidence for names, numbers, "
            "code identifiers, and possible tool arguments; treat it as evidence rather than ground truth."
        )
    if oracle_text_source:
        if oracle_text_source.lower().startswith("asr"):
            parts.append(
                f"The oracle-facing text was selected from {oracle_text_source} evidence; preserve the reflex intent and route as the control signal."
            )
        else:
            parts.append(f"The oracle-facing text source is {oracle_text_source}.")
    if evidence_authority:
        labels = "; ".join(
            f"{key}={value}"
            for key, value in evidence_authority.items()
        )
        parts.append(
            f"Evidence authority labels: {labels}. Treat primary_audio and interpreter_promoted fields as higher authority than reflex_hypothesis or auxiliary_hypothesis fields."
        )
    if interface_already_said:
        parts.append(f"The voice reflex already told the user: {interface_already_said}")
    if summary:
        parts.append(f"Ephemeral live voice summary: {summary}")
    if job_updates:
        parts.append(f"User added updates for this oracle job: {' | '.join(job_updates)}")
    max_sentences = response_style.get("max_sentences") or _metadata_positive_int(metadata.get("max_spoken_sentences"))
    policy = _metadata_text(response_style.get("policy")) or _metadata_text(metadata.get("voice_response_policy"))
    if max_sentences is not None:
        if policy == "full":
            parts.append("The voice response policy is full: include the complete spoken answer when needed.")
        elif policy == "brief_summary":
            parts.append(
                f"Summarize long oracle output for speech and keep the spoken summary to at most {max_sentences} sentence(s)."
            )
        else:
            parts.append(f"Keep spoken output to at most {max_sentences} sentence(s) unless the task requires more.")
    if response_style:
        spoken = "true" if response_style.get("spoken", True) else "false"
        followups = "allowed" if response_style.get("allow_followup_offer") else "avoid automatic follow-up offers"
        policy_part = f"; policy={policy}" if policy else ""
        parts.append(f"Requested response style: spoken={spoken}{policy_part}; {followups}.")
    return " ".join(parts)


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


def _metadata_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:1000]


def _metadata_text_sequence(value: object) -> list[str]:
    if isinstance(value, str):
        text = _metadata_text(value)
        return [text] if text else []
    if not isinstance(value, (list, tuple)):
        return []
    items = []
    for item in value:
        text = _metadata_text(item)
        if text:
            items.append(text)
    return items[:5]


def _metadata_time_range(value: object) -> tuple[int, int] | tuple[()]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return ()
    try:
        start = int(value[0])
        end = int(value[1])
    except (TypeError, ValueError):
        return ()
    if start < 0 or end < start:
        return ()
    return (start, end)


def _metadata_transcript_hypotheses(value: object) -> list[dict[str, object]]:
    if not isinstance(value, (list, tuple)):
        return []
    hypotheses: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        text = _metadata_text(item.get("text"))
        if not text:
            continue
        hypothesis: dict[str, object] = {
            "source": _metadata_text(item.get("source")) or "unknown",
            "text": text,
        }
        confidence = _metadata_float(item.get("confidence"))
        if confidence is not None:
            hypothesis["confidence"] = confidence
        latency = _metadata_positive_int(item.get("latency_ms"))
        if latency is not None:
            hypothesis["latency_ms"] = latency
        hypotheses.append(hypothesis)
        if len(hypotheses) >= 5:
            break
    return hypotheses


def _metadata_evidence_authority(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    authority: dict[str, str] = {}
    allowed = {
        "primary_audio",
        "reflex_hypothesis",
        "auxiliary_hypothesis",
        "interpreter_promoted",
        "oracle_promoted",
        "diagnostic_only",
    }
    for key, raw_value in value.items():
        field = _metadata_text(str(key))
        label = _metadata_text(raw_value)
        if field and label in allowed:
            authority[field] = label
        if len(authority) >= 12:
            break
    return authority


def _metadata_positive_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def _metadata_float(value: object) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _metadata_response_style(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    style: dict[str, object] = {}
    spoken = value.get("spoken")
    if isinstance(spoken, bool):
        style["spoken"] = spoken
    max_sentences = _metadata_positive_int(value.get("max_sentences"))
    if max_sentences is not None:
        style["max_sentences"] = max_sentences
    followups = value.get("allow_followup_offer")
    if isinstance(followups, bool):
        style["allow_followup_offer"] = followups
    policy = _metadata_text(value.get("policy") or value.get("voice_response_policy"))
    if policy:
        style["policy"] = policy
    return style


class NullRealtimeOracle:
    """Oracle used by tests and transcript-only sessions."""

    async def answer(self, transcript: str) -> str:
        return ""

    async def stream_answer(self, transcript: str) -> AsyncIterator[str]:
        if False:
            yield transcript

    def interrupt(self, message: str = "Realtime voice turn interrupted") -> None:
        return None
