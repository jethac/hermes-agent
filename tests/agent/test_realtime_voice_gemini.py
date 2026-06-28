import asyncio
import base64
import json

import pytest

from agent.realtime_voice import AudioChunk, RealtimeVoiceSessionConfig, VoiceAudioCodec, VoiceEvent, VoiceEventType
from agent.realtime_voice_gemini import (
    GEMINI_LIVE_INPUT_SAMPLE_RATE_HZ,
    GEMINI_LIVE_OUTPUT_SAMPLE_RATE_HZ,
    GeminiLiveFrontendConfig,
    GeminiLiveFrontendSession,
)


class FakeGeminiWebSocket:
    def __init__(self, fail_send: bool = False):
        self.sent = []
        self.closed = False
        self.fail_send = fail_send
        self._incoming = asyncio.Queue()

    async def send(self, data):
        payload = json.loads(data)
        self.sent.append(payload)
        if self.fail_send:
            raise RuntimeError("send failed")

    async def close(self):
        self.closed = True
        await self._incoming.put(None)

    async def emit(self, payload):
        await self._incoming.put(json.dumps(payload))

    async def emit_raw(self, payload):
        await self._incoming.put(payload)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self._incoming.get()
        if item is None:
            raise StopAsyncIteration
        return item


@pytest.mark.asyncio
async def test_gemini_live_start_sends_setup_and_ready_state():
    fake_ws = FakeGeminiWebSocket()
    calls = []

    async def connector(url, timeout):
        calls.append((url, timeout))
        return fake_ws

    session = GeminiLiveFrontendSession(
        GeminiLiveFrontendConfig(
            api_key="gemini-secret",
            model="gemini-3.1-flash-live-preview",
            voice="Puck",
            connect_timeout_seconds=0.5,
        ),
        connector=connector,
    )

    await session.start(RealtimeVoiceSessionConfig(session_id="voice-1", frontend_model="gemini-3.1-flash-live-preview"))
    event = await asyncio.wait_for(session.events().__anext__(), timeout=1)

    assert calls == [
        (
            "wss://generativelanguage.googleapis.com/ws/"
            "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=gemini-secret",
            0.5,
        )
    ]
    setup = fake_ws.sent[0]["setup"]
    assert setup["model"] == "models/gemini-3.1-flash-live-preview"
    assert setup["responseModalities"] == ["AUDIO"]
    assert setup["inputAudioTranscription"] == {}
    assert setup["outputAudioTranscription"] == {}
    assert setup["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Puck"
    assert setup["tools"][0]["functionDeclarations"][0]["name"] == "ask_hermes_oracle"
    assert event.type == VoiceEventType.FRONTEND_STATE
    assert event.payload["provider"] == "gemini_live"
    assert event.payload["tool_calls"] is True

    await session.close()


@pytest.mark.asyncio
async def test_gemini_live_audio_input_resamples_to_16khz_pcm():
    fake_ws = FakeGeminiWebSocket()

    async def connector(_url, _timeout):
        return fake_ws

    session = GeminiLiveFrontendSession(
        GeminiLiveFrontendConfig(api_key="gemini-secret"),
        connector=connector,
    )
    await session.start(RealtimeVoiceSessionConfig(session_id="voice-1", sample_rate_hz=48000, channels=2))

    stereo_48k = b"\x00\x00\x00\x00" * 480
    await session.receive_event(
        VoiceEvent(
            type=VoiceEventType.AUDIO_INPUT_CHUNK,
            session_id="voice-1",
            sequence=1,
            payload=AudioChunk(
                codec=VoiceAudioCodec.PCM16,
                data=stereo_48k,
                sample_rate_hz=48000,
                channels=2,
            ).to_payload(),
        )
    )

    audio_message = fake_ws.sent[1]["realtimeInput"]["audio"]
    pcm = base64.b64decode(audio_message["data"])
    assert audio_message["mimeType"] == f"audio/pcm;rate={GEMINI_LIVE_INPUT_SAMPLE_RATE_HZ}"
    assert len(pcm) == 320

    await session.close()


@pytest.mark.asyncio
async def test_gemini_live_server_events_map_to_hermes_events():
    fake_ws = FakeGeminiWebSocket()

    async def connector(_url, _timeout):
        return fake_ws

    session = GeminiLiveFrontendSession(
        GeminiLiveFrontendConfig(api_key="gemini-secret"),
        connector=connector,
    )
    await session.start(RealtimeVoiceSessionConfig(session_id="voice-1", sample_rate_hz=48000))
    _ready = await asyncio.wait_for(session.events().__anext__(), timeout=1)

    pcm24 = b"\x00\x00" * 240
    await fake_ws.emit(
        {
            "serverContent": {
                "inputTranscription": {"text": "hello"},
                "outputTranscription": {"text": "hi"},
                "modelTurn": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": f"audio/pcm;rate={GEMINI_LIVE_OUTPUT_SAMPLE_RATE_HZ}",
                                "data": base64.b64encode(pcm24).decode("ascii"),
                            }
                        },
                        {"text": "there"},
                    ]
                },
                "generationComplete": True,
            }
        }
    )

    seen = [await asyncio.wait_for(session.events().__anext__(), timeout=1) for _ in range(7)]
    assert [event.type for event in seen] == [
        VoiceEventType.TRANSCRIPT_FINAL,
        VoiceEventType.ASSISTANT_TEXT_PARTIAL,
        VoiceEventType.PLAYBACK_STARTED,
        VoiceEventType.AUDIO_OUTPUT_CHUNK,
        VoiceEventType.ASSISTANT_TEXT_PARTIAL,
        VoiceEventType.PLAYBACK_STOPPED,
        VoiceEventType.ASSISTANT_COMMIT,
    ]
    assert seen[0].payload["text"] == "hello"
    assert seen[1].payload["text"] == "hi"
    assert "playback_generation" not in seen[2].payload
    audio_payload = AudioChunk.from_payload(seen[3].payload)
    assert audio_payload.sample_rate_hz == 48000
    assert seen[3].payload["metrics"] == {"gemini_live": True}
    assert "playback_generation" not in seen[5].payload

    await session.close()


@pytest.mark.asyncio
async def test_gemini_live_tool_call_routes_only_to_kame_oracle_bridge():
    fake_ws = FakeGeminiWebSocket()

    async def connector(_url, _timeout):
        return fake_ws

    session = GeminiLiveFrontendSession(
        GeminiLiveFrontendConfig(api_key="gemini-secret"),
        connector=connector,
    )
    await session.start(RealtimeVoiceSessionConfig(session_id="voice-1"))
    _ready = await asyncio.wait_for(session.events().__anext__(), timeout=1)

    await fake_ws.emit(
        {
            "toolCall": {
                "functionCalls": [
                    {"id": "call-1", "name": "ask_hermes_oracle", "args": {"query": "use memory and tools"}},
                    {"id": "call-2", "name": "delete_files", "args": {"path": "/tmp"}},
                ]
            }
        }
    )

    events = [await asyncio.wait_for(session.events().__anext__(), timeout=1) for _ in range(5)]
    assert [event.type for event in events] == [
        VoiceEventType.TOOL_PENDING,
        VoiceEventType.ORACLE_HINT,
        VoiceEventType.TRANSCRIPT_FINAL,
        VoiceEventType.TOOL_RESULT,
        VoiceEventType.TOOL_PENDING,
    ]
    assert events[1].payload["text"] == "use memory and tools"
    assert events[2].payload["source"] == "gemini_live_tool"
    result = await asyncio.wait_for(session.events().__anext__(), timeout=1)
    assert result.type == VoiceEventType.TOOL_RESULT
    assert "not enabled" in result.payload["error"]
    tool_response = fake_ws.sent[-1]["toolResponse"]["functionResponses"]
    assert tool_response[0]["response"] == {"result": "queued_to_hermes_oracle"}
    assert "not enabled" in tool_response[1]["response"]["error"]

    await session.close()


@pytest.mark.asyncio
async def test_gemini_live_requires_api_key():
    session = GeminiLiveFrontendSession(GeminiLiveFrontendConfig(api_key=""))

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        await session.start(RealtimeVoiceSessionConfig(session_id="voice-1"))
