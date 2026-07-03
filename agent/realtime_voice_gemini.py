"""Gemini Live frontend provider for Hermes realtime voice."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping, Optional
from urllib.parse import quote

from agent.realtime_voice import (
    AudioChunk,
    RealtimeVoiceSessionConfig,
    VoiceAudioCodec,
    VoiceEvent,
    VoiceEventType,
    create_realtime_voice_event_queue,
    put_realtime_voice_event,
)
from agent.realtime_voice_errors import sanitize_realtime_voice_error
from agent.realtime_voice_openai import resample_pcm16_mono


GEMINI_LIVE_INPUT_SAMPLE_RATE_HZ = 16000
GEMINI_LIVE_OUTPUT_SAMPLE_RATE_HZ = 24000
GEMINI_LIVE_DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
GEMINI_LIVE_DEFAULT_VOICE = "Puck"
GEMINI_LIVE_DEFAULT_BASE_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
GEMINI_LIVE_BASE_INSTRUCTIONS = (
    "You are Hermes' low-latency realtime voice interface. Keep spoken output short. "
    "Do not claim to be a separate bot. Hermes' backend oracle owns durable reasoning, "
    "tools, memory, and final task execution."
)
GEMINI_LIVE_CAPABILITY_HONESTY_INSTRUCTIONS = (
    "This voice session is already connected; never claim Hermes cannot hear, listen, join, "
    "or speak through the live voice interface."
)
GEMINI_LIVE_ORACLE_TOOL_INSTRUCTIONS = (
    "Use ask_hermes_oracle when a request needs the backend Hermes agent."
)
GEMINI_LIVE_ORACLE_CONTEXT_EVENT_TYPES = frozenset(
    {
        VoiceEventType.ORACLE_ACCEPTED,
        VoiceEventType.ORACLE_HINT,
        VoiceEventType.ORACLE_TOOL_CALL,
        VoiceEventType.ORACLE_TOOL_RESULT,
        VoiceEventType.ORACLE_RESPONSE_PARTIAL,
        VoiceEventType.ORACLE_RESPONSE_FINAL,
        VoiceEventType.ORACLE_ERROR,
    }
)


WebSocketConnector = Callable[[str, float], Awaitable[Any]]


@dataclass(frozen=True)
class GeminiLiveFrontendConfig:
    """Runtime config for the Gemini Live frontend provider."""

    api_key: Optional[str] = None
    base_url: str = GEMINI_LIVE_DEFAULT_BASE_URL
    model: str = GEMINI_LIVE_DEFAULT_MODEL
    voice: str = GEMINI_LIVE_DEFAULT_VOICE
    connect_timeout_seconds: float = 10.0
    enable_google_search: bool = False
    enable_oracle_tool: bool = True
    instructions: str = GEMINI_LIVE_BASE_INSTRUCTIONS


class GeminiLiveFrontendSession:
    """Bridge Hermes realtime voice events to Gemini Live WebSocket events."""

    def __init__(
        self,
        runtime: GeminiLiveFrontendConfig,
        *,
        connector: Optional[WebSocketConnector] = None,
    ):
        self.runtime = runtime
        self.config: Optional[RealtimeVoiceSessionConfig] = None
        self._connector = connector or _connect_websocket
        self._ws: Any = None
        self._events: asyncio.Queue[VoiceEvent | None] = create_realtime_voice_event_queue()
        self._reader_task: Optional[asyncio.Task[None]] = None
        self._sequence = 0
        self._closed = False
        self._active_playback_generation: Optional[int] = None
        self._playback_active = False
        self._playback_started_generation: Optional[int] = None

    async def start(self, config: RealtimeVoiceSessionConfig) -> None:
        api_key = (self.runtime.api_key or "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or HERMES_GEMINI_LIVE_API_KEY is required")
        self.config = config
        model = config.frontend_model or self.runtime.model or GEMINI_LIVE_DEFAULT_MODEL
        self._ws = await self._connector(
            _gemini_live_url(self.runtime.base_url, api_key),
            max(self.runtime.connect_timeout_seconds, 0.1),
        )
        await self._send_gemini(_setup_payload(model, self.runtime))
        self._reader_task = asyncio.create_task(self._consume_gemini_events())
        await self._emit(
            VoiceEventType.FRONTEND_STATE,
            {
                "status": "ready",
                "provider": "gemini_live",
                "model": model,
                "voice": self.runtime.voice,
                "streaming_stt": True,
                "streaming_tts": True,
                "native_s2s": True,
                "server_vad": True,
                "response_cancel": True,
                "tool_calls": self.runtime.enable_oracle_tool,
                "google_search": self.runtime.enable_google_search,
                "input_sample_rate_hz": GEMINI_LIVE_INPUT_SAMPLE_RATE_HZ,
                "output_sample_rate_hz": GEMINI_LIVE_OUTPUT_SAMPLE_RATE_HZ,
            },
        )

    async def receive_event(self, event: VoiceEvent) -> None:
        if self._closed:
            return
        if event.type == VoiceEventType.SESSION_CLOSED:
            await self.close()
            return
        if event.type == VoiceEventType.BARGE_IN:
            await self._handle_barge_in(event)
            return
        if event.type == VoiceEventType.AUDIO_INPUT_CHUNK:
            await self._handle_audio_input(event)
            return
        if event.type in GEMINI_LIVE_ORACLE_CONTEXT_EVENT_TYPES:
            await self._handle_oracle_context(event)
            return
        if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL and event.payload.get("speak") is True:
            await self._handle_speak(event)

    async def events(self) -> AsyncIterator[VoiceEvent]:
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def close(self) -> None:
        if self._closed:
            return
        await self._emit_playback_stopped(self._playback_started_generation)
        self._closed = True
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader_task
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        await self._emit(VoiceEventType.SESSION_CLOSED, {"reason": "closed"})
        await put_realtime_voice_event(self._events, None)

    async def _handle_audio_input(self, event: VoiceEvent) -> None:
        try:
            chunk = AudioChunk.from_payload(event.payload)
        except Exception:
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": "invalid audio chunk"})
            return
        if chunk.codec != VoiceAudioCodec.PCM16:
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": "Gemini Live provider requires pcm16 input"})
            return
        pcm16 = resample_pcm16_mono(
            chunk.data,
            from_rate_hz=int(event.payload.get("sample_rate_hz") or chunk.sample_rate_hz),
            to_rate_hz=GEMINI_LIVE_INPUT_SAMPLE_RATE_HZ,
            channels=int(event.payload.get("channels") or chunk.channels),
        )
        await self._send_gemini(
            {
                "realtimeInput": {
                    "audio": {
                        "data": base64.b64encode(pcm16).decode("ascii"),
                        "mimeType": f"audio/pcm;rate={GEMINI_LIVE_INPUT_SAMPLE_RATE_HZ}",
                    }
                }
            }
        )

    async def _handle_oracle_context(self, event: VoiceEvent) -> None:
        text = _oracle_context_text(event)
        if not text:
            return
        await self._send_gemini(
            {
                "clientContent": {
                    "turns": [{"role": "user", "parts": [{"text": text}]}],
                    "turnComplete": False,
                }
            }
        )

    async def _handle_speak(self, event: VoiceEvent) -> None:
        text = str(event.payload.get("text") or "").strip()
        if not text:
            return
        playback_generation = _payload_int(event.payload.get("playback_generation"))
        self._active_playback_generation = playback_generation
        await self._send_gemini(
            {
                "realtimeInput": {
                    "text": (
                        "Speak the following Hermes response verbatim and do not add extra words:\n\n"
                        f"{text}"
                    )
                }
            }
        )

    async def _handle_barge_in(self, event: VoiceEvent) -> None:
        payload = {"reason": event.payload.get("reason") or "client"}
        playback_generation = _payload_int(event.payload.get("playback_generation"))
        if playback_generation is not None:
            payload["playback_generation"] = playback_generation
        self._active_playback_generation = None
        with contextlib.suppress(Exception):
            await self._send_gemini({"realtimeInput": {"text": "[interrupt current response]"}})
        await self._emit_playback_stopped(playback_generation)
        await self._emit(VoiceEventType.BARGE_IN, payload)

    async def _consume_gemini_events(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    event = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
                except Exception:
                    await self._emit(VoiceEventType.SESSION_ERROR, {"error": "invalid Gemini Live event"})
                    continue
                await self._handle_gemini_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closed:
                await self._emit(
                    VoiceEventType.SESSION_ERROR,
                    {"error": f"Gemini Live event stream failed: {sanitize_realtime_voice_error(exc)}"},
                )

    async def _handle_gemini_event(self, event: Mapping[str, Any]) -> None:
        if "error" in event:
            error = event.get("error")
            message = error.get("message") if isinstance(error, Mapping) else "Gemini Live error"
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": sanitize_realtime_voice_error(message)})
            return
        if "toolCall" in event and isinstance(event.get("toolCall"), Mapping):
            await self._handle_tool_call(event["toolCall"])
            return
        server_content = event.get("serverContent")
        if not isinstance(server_content, Mapping):
            return
        input_transcription = server_content.get("inputTranscription")
        if isinstance(input_transcription, Mapping):
            text = str(input_transcription.get("text") or "").strip()
            if text:
                event_type = (
                    VoiceEventType.TRANSCRIPT_FINAL
                    if server_content.get("turnComplete") is True or server_content.get("generationComplete") is True
                    else VoiceEventType.TRANSCRIPT_PARTIAL
                )
                payload: dict[str, Any] = {"text": text}
                if event_type == VoiceEventType.TRANSCRIPT_PARTIAL:
                    payload["stability"] = 0.7
                await self._emit(event_type, payload)
        output_transcription = server_content.get("outputTranscription")
        if isinstance(output_transcription, Mapping):
            text = str(output_transcription.get("text") or "").strip()
            if text:
                await self._emit(VoiceEventType.ASSISTANT_TEXT_PARTIAL, {"text": text})
        model_turn = server_content.get("modelTurn")
        if isinstance(model_turn, Mapping):
            for part in _list(model_turn.get("parts")):
                if not isinstance(part, Mapping):
                    continue
                inline_data = part.get("inlineData")
                if isinstance(inline_data, Mapping):
                    data_b64 = inline_data.get("data")
                    if isinstance(data_b64, str):
                        await self._emit_audio_delta(data_b64)
                text = str(part.get("text") or "").strip()
                if text:
                    await self._emit(VoiceEventType.ASSISTANT_TEXT_PARTIAL, {"text": text})
        if server_content.get("generationComplete") is True or server_content.get("turnComplete") is True:
            await self._emit_playback_stopped(self._active_playback_generation)
            await self._emit(VoiceEventType.ASSISTANT_COMMIT, {"provider": "gemini_live"})

    async def _handle_tool_call(self, tool_call: Mapping[str, Any]) -> None:
        function_responses = []
        for call in _list(tool_call.get("functionCalls")):
            if not isinstance(call, Mapping):
                continue
            name = str(call.get("name") or "")
            call_id = str(call.get("id") or "")
            args = call.get("args") if isinstance(call.get("args"), Mapping) else {}
            await self._emit(
                VoiceEventType.TOOL_PENDING,
                {"provider": "gemini_live", "tool": name, "tool_call_id": call_id},
            )
            if name == "ask_hermes_oracle" and self.runtime.enable_oracle_tool:
                query = str(args.get("query") or args.get("text") or "").strip()
                if query:
                    await self._emit(
                        VoiceEventType.INTERFACE_ORACLE_REQUEST,
                        {"provider": "gemini_live", "tool": name, "tool_call_id": call_id, "text": query},
                    )
                    response = {"result": "queued_to_hermes_oracle"}
                else:
                    response = {"error": "ask_hermes_oracle requires query"}
            elif name == "get_voice_session_status":
                response = {
                    "result": {
                        "provider": "gemini_live",
                        "model": self.config.frontend_model if self.config else self.runtime.model,
                        "backend": "hermes_oracle",
                    }
                }
            elif name == "cancel_hermes_oracle":
                job_id = str(args.get("job_id") or "").strip()
                if job_id:
                    await self._emit(
                        VoiceEventType.INTERFACE_ORACLE_CANCEL,
                        {
                            "job_id": job_id,
                            "reason": "gemini_live_tool",
                            "tool_call_id": call_id,
                            "provider": "gemini_live",
                        },
                    )
                else:
                    await self._emit(VoiceEventType.BARGE_IN, {"reason": "gemini_live_tool", "tool_call_id": call_id})
                response = {"result": "cancel_requested"}
            else:
                response = {"error": f"tool {name or '<missing>'} is not enabled for Gemini Live"}
            await self._emit(
                VoiceEventType.TOOL_RESULT,
                {"provider": "gemini_live", "tool": name, "tool_call_id": call_id, **response},
            )
            function_responses.append({"name": name, "id": call_id, "response": response})
        if function_responses:
            await self._send_gemini({"toolResponse": {"functionResponses": function_responses}})

    async def _emit_audio_delta(self, data_b64: str) -> None:
        config = self.config
        if config is None:
            return
        try:
            pcm24 = base64.b64decode(data_b64.encode("ascii"), validate=True)
        except Exception:
            await self._emit(VoiceEventType.SESSION_ERROR, {"error": "invalid Gemini Live output audio delta"})
            return
        pcm = resample_pcm16_mono(
            pcm24,
            from_rate_hz=GEMINI_LIVE_OUTPUT_SAMPLE_RATE_HZ,
            to_rate_hz=int(config.sample_rate_hz or 16000),
            channels=1,
        )
        payload = AudioChunk(
            codec=VoiceAudioCodec.PCM16,
            data=pcm,
            sample_rate_hz=int(config.sample_rate_hz or 16000),
            channels=1,
        ).to_payload()
        if self._active_playback_generation is not None:
            payload["playback_generation"] = self._active_playback_generation
        payload["metrics"] = {"gemini_live": True}
        await self._emit_playback_started(self._active_playback_generation)
        await self._emit(VoiceEventType.AUDIO_OUTPUT_CHUNK, payload)

    async def _emit_playback_started(self, generation: Optional[int]) -> None:
        if self._playback_active and self._playback_started_generation == generation:
            return
        if self._playback_active:
            await self._emit_playback_stopped(self._playback_started_generation)
        self._playback_active = True
        self._playback_started_generation = generation
        payload = {"playback_generation": generation} if generation is not None else {}
        await self._emit(VoiceEventType.PLAYBACK_STARTED, payload)

    async def _emit_playback_stopped(self, generation: Optional[int]) -> None:
        if not self._playback_active:
            return
        payload_generation = generation if generation is not None else self._playback_started_generation
        payload = {"playback_generation": payload_generation} if payload_generation is not None else {}
        self._playback_active = False
        self._playback_started_generation = None
        await self._emit(VoiceEventType.PLAYBACK_STOPPED, payload)
        await self._emit(VoiceEventType.ASSISTANT_AUDIO_END, payload)

    async def _send_gemini(self, payload: Mapping[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("Gemini Live websocket is not connected")
        await self._ws.send(json.dumps(dict(payload)))

    async def _emit(self, event_type: VoiceEventType, payload: Mapping[str, Any]) -> None:
        if self.config is None:
            return
        if self._closed and event_type != VoiceEventType.SESSION_CLOSED:
            return
        self._sequence += 1
        await put_realtime_voice_event(
            self._events,
            VoiceEvent(
                type=event_type,
                session_id=self.config.session_id,
                sequence=self._sequence,
                payload=dict(payload),
            ),
        )


async def _connect_websocket(url: str, timeout: float) -> Any:
    import websockets

    return await websockets.connect(url, open_timeout=timeout)


def _gemini_live_url(base_url: str, api_key: str) -> str:
    base = (base_url or GEMINI_LIVE_DEFAULT_BASE_URL).strip()
    separator = "&" if "?" in base else "?"
    return f"{base}{separator}key={quote(api_key, safe='')}"


def _setup_payload(model: str, runtime: GeminiLiveFrontendConfig) -> dict[str, Any]:
    setup: dict[str, Any] = {
        "model": f"models/{_strip_model_prefix(model)}",
        "responseModalities": ["AUDIO"],
        "systemInstruction": {"parts": [{"text": _gemini_live_system_instruction(runtime)}]},
        "inputAudioTranscription": {},
        "outputAudioTranscription": {},
        "contextWindowCompression": {"slidingWindow": {}},
        "speechConfig": {
            "voiceConfig": {
                "prebuiltVoiceConfig": {
                    "voiceName": runtime.voice or GEMINI_LIVE_DEFAULT_VOICE,
                }
            }
        },
    }
    tools = []
    if runtime.enable_oracle_tool:
        tools.append({"functionDeclarations": _gemini_kame_tool_declarations()})
    if runtime.enable_google_search:
        tools.append({"googleSearch": {}})
    if tools:
        setup["tools"] = tools
    return {"setup": setup}


def _gemini_live_system_instruction(runtime: GeminiLiveFrontendConfig) -> str:
    instruction = str(runtime.instructions or GEMINI_LIVE_BASE_INSTRUCTIONS).strip()
    if not instruction:
        instruction = GEMINI_LIVE_BASE_INSTRUCTIONS
    if "already connected" not in instruction.lower():
        instruction = f"{instruction} {GEMINI_LIVE_CAPABILITY_HONESTY_INSTRUCTIONS}"
    if runtime.enable_oracle_tool and "ask_hermes_oracle" not in instruction:
        instruction = f"{instruction} {GEMINI_LIVE_ORACLE_TOOL_INSTRUCTIONS}"
    return instruction


def _gemini_kame_tool_declarations() -> list[dict[str, Any]]:
    return [
        {
            "name": "ask_hermes_oracle",
            "description": "Route a user request to the Hermes backend oracle for reasoning, memory, and tool execution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The concise user request or clarification to send to Hermes.",
                    }
                },
                "required": ["query"],
            },
        },
        {
            "name": "cancel_hermes_oracle",
            "description": "Request cancellation of a Hermes backend oracle job after user barge-in or correction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Optional Hermes oracle job id to cancel. Omit only for generic playback barge-in.",
                    }
                },
            },
        },
        {
            "name": "get_voice_session_status",
            "description": "Inspect the current KAME voice session role split and frontend provider.",
            "parameters": {"type": "object", "properties": {}},
        },
    ]


def _strip_model_prefix(model: str) -> str:
    text = str(model or GEMINI_LIVE_DEFAULT_MODEL).strip()
    return text.removeprefix("models/")


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _payload_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _oracle_context_text(event: VoiceEvent) -> str:
    payload = event.payload if isinstance(event.payload, Mapping) else {}
    parts = [
        "Hermes backend oracle context update for the realtime voice interface.",
        "Use this only to track the current oracle handoff; do not speak it by itself.",
        f"event={event.type.value}",
    ]
    for key in (
        "turn_id",
        "route",
        "text",
        "delta",
        "final",
        "accepted",
        "tool_name",
        "tool_call_id",
        "reason",
        "error",
    ):
        if key not in payload:
            continue
        value = _oracle_context_value(payload.get(key))
        if value:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _oracle_context_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        return str(int(value)) if float(value).is_integer() else str(value)
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    if len(text) > 240:
        text = text[:239].rstrip() + "..."
    return json.dumps(text, ensure_ascii=True)
