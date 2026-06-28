import asyncio
import base64
import json

import pytest

from agent.realtime_voice import AudioChunk, RealtimeVoiceSessionConfig, VoiceAudioCodec, VoiceEvent, VoiceEventType
from agent.realtime_voice_openai import (
    OPENAI_REALTIME_SAMPLE_RATE_HZ,
    OpenAIRealtimeFrontendConfig,
    OpenAIRealtimeFrontendSession,
    resample_pcm16_mono,
)


class FakeOpenAIWebSocket:
    def __init__(self, fail_types=None):
        self.sent = []
        self.closed = False
        self.fail_types = set(fail_types or [])
        self._incoming = asyncio.Queue()

    async def send(self, data):
        payload = json.loads(data)
        self.sent.append(payload)
        if payload.get("type") in self.fail_types:
            raise RuntimeError(f"send failed for {payload['type']}")

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
async def test_openai_realtime_start_sends_session_update_and_ready_state():
    fake_ws = FakeOpenAIWebSocket()
    calls = []

    async def connector(url, headers, timeout):
        calls.append((url, headers, timeout))
        return fake_ws

    session = OpenAIRealtimeFrontendSession(
        OpenAIRealtimeFrontendConfig(
            api_key="sk-test",
            model="gpt-realtime-2",
            voice="marin",
            connect_timeout_seconds=0.5,
        ),
        connector=connector,
    )

    await session.start(RealtimeVoiceSessionConfig(session_id="voice-1", frontend_model="gpt-realtime-2"))
    event = await asyncio.wait_for(session.events().__anext__(), timeout=1)

    assert calls == [
        (
            "wss://api.openai.com/v1/realtime?model=gpt-realtime-2",
            {"Authorization": "Bearer sk-test"},
            0.5,
        )
    ]
    assert fake_ws.sent[0]["type"] == "session.update"
    assert "tools" not in fake_ws.sent[0]["session"]
    assert fake_ws.sent[0]["session"]["audio"]["input"]["format"] == {
        "type": "audio/pcm",
        "rate": OPENAI_REALTIME_SAMPLE_RATE_HZ,
    }
    assert fake_ws.sent[0]["session"]["audio"]["input"]["turn_detection"] is None
    assert event.type == VoiceEventType.FRONTEND_STATE
    assert event.payload["provider"] == "openai_realtime"
    assert event.payload["response_cancel"] is True

    await session.close()


@pytest.mark.asyncio
async def test_openai_realtime_audio_input_resamples_and_commits_on_utterance_end():
    fake_ws = FakeOpenAIWebSocket()

    async def connector(_url, _headers, _timeout):
        return fake_ws

    session = OpenAIRealtimeFrontendSession(
        OpenAIRealtimeFrontendConfig(api_key="sk-test"),
        connector=connector,
    )
    await session.start(RealtimeVoiceSessionConfig(session_id="voice-1", sample_rate_hz=16000, channels=1))

    await session.receive_event(
        VoiceEvent(
            type=VoiceEventType.AUDIO_INPUT_CHUNK,
            session_id="voice-1",
            sequence=1,
            payload={
                **AudioChunk(
                    codec=VoiceAudioCodec.PCM16,
                    data=b"\x00\x00" * 320,
                    sample_rate_hz=16000,
                    channels=1,
                ).to_payload(),
                "end_of_utterance": True,
            },
        )
    )

    append = fake_ws.sent[1]
    assert append["type"] == "input_audio_buffer.append"
    assert len(base64.b64decode(append["audio"])) == 960
    assert fake_ws.sent[2] == {"type": "input_audio_buffer.commit"}

    await session.close()


@pytest.mark.asyncio
async def test_openai_realtime_server_events_map_to_hermes_events():
    fake_ws = FakeOpenAIWebSocket()

    async def connector(_url, _headers, _timeout):
        return fake_ws

    session = OpenAIRealtimeFrontendSession(
        OpenAIRealtimeFrontendConfig(api_key="sk-test"),
        connector=connector,
    )
    await session.start(RealtimeVoiceSessionConfig(session_id="voice-1", sample_rate_hz=16000, channels=1))
    events = session.events()
    ready = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert ready.type == VoiceEventType.FRONTEND_STATE

    await session.receive_event(
        VoiceEvent(
            type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
            session_id="voice-1",
            sequence=1,
            payload={"text": "Hello.", "speak": True, "playback_generation": 7},
        )
    )
    assert fake_ws.sent[-1]["type"] == "response.create"
    assert "Hello." in fake_ws.sent[-1]["response"]["instructions"]
    assert "tools" not in fake_ws.sent[-1]["response"]

    await fake_ws.emit(
        {
            "type": "conversation.item.input_audio_transcription.delta",
            "delta": "Hel",
        }
    )
    await fake_ws.emit(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "Hello from voice",
        }
    )
    await fake_ws.emit(
        {
            "type": "response.output_audio.delta",
            "delta": base64.b64encode(b"\x00\x00" * 480).decode("ascii"),
        }
    )
    await fake_ws.emit({"type": "response.done"})

    partial = await asyncio.wait_for(events.__anext__(), timeout=1)
    final = await asyncio.wait_for(events.__anext__(), timeout=1)
    playback_started = await asyncio.wait_for(events.__anext__(), timeout=1)
    audio = await asyncio.wait_for(events.__anext__(), timeout=1)
    playback_stopped = await asyncio.wait_for(events.__anext__(), timeout=1)
    commit = await asyncio.wait_for(events.__anext__(), timeout=1)

    assert partial.type == VoiceEventType.TRANSCRIPT_PARTIAL
    assert partial.payload["text"] == "Hel"
    assert final.type == VoiceEventType.TRANSCRIPT_FINAL
    assert final.payload["text"] == "Hello from voice"
    assert playback_started.type == VoiceEventType.PLAYBACK_STARTED
    assert playback_started.payload["playback_generation"] == 7
    assert audio.type == VoiceEventType.AUDIO_OUTPUT_CHUNK
    assert audio.payload["playback_generation"] == 7
    assert len(base64.b64decode(audio.payload["data_b64"])) == 640
    assert playback_stopped.type == VoiceEventType.PLAYBACK_STOPPED
    assert playback_stopped.payload["playback_generation"] == 7
    assert commit.type == VoiceEventType.ASSISTANT_COMMIT

    await session.close()


@pytest.mark.asyncio
async def test_openai_realtime_barge_in_cancels_response_and_clears_input():
    fake_ws = FakeOpenAIWebSocket()

    async def connector(_url, _headers, _timeout):
        return fake_ws

    session = OpenAIRealtimeFrontendSession(
        OpenAIRealtimeFrontendConfig(api_key="sk-test"),
        connector=connector,
    )
    await session.start(RealtimeVoiceSessionConfig(session_id="voice-1"))
    events = session.events()
    await asyncio.wait_for(events.__anext__(), timeout=1)

    await session.receive_event(
        VoiceEvent(
            type=VoiceEventType.BARGE_IN,
            session_id="voice-1",
            sequence=1,
            payload={"reason": "speech_started", "playback_generation": 9},
        )
    )

    assert fake_ws.sent[-2] == {"type": "response.cancel"}
    assert fake_ws.sent[-1] == {"type": "input_audio_buffer.clear"}
    event = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert event.type == VoiceEventType.BARGE_IN
    assert event.payload["playback_generation"] == 9

    await session.close()


@pytest.mark.asyncio
async def test_openai_realtime_malformed_server_event_emits_session_error():
    fake_ws = FakeOpenAIWebSocket()

    async def connector(_url, _headers, _timeout):
        return fake_ws

    session = OpenAIRealtimeFrontendSession(
        OpenAIRealtimeFrontendConfig(api_key="sk-test"),
        connector=connector,
    )
    await session.start(RealtimeVoiceSessionConfig(session_id="voice-1"))
    events = session.events()
    await asyncio.wait_for(events.__anext__(), timeout=1)

    await fake_ws.emit_raw("{not-json")

    event = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert event.type == VoiceEventType.SESSION_ERROR
    assert event.payload["error"] == "invalid OpenAI realtime event"

    await session.close()


@pytest.mark.asyncio
async def test_openai_realtime_invalid_output_audio_delta_emits_session_error():
    fake_ws = FakeOpenAIWebSocket()

    async def connector(_url, _headers, _timeout):
        return fake_ws

    session = OpenAIRealtimeFrontendSession(
        OpenAIRealtimeFrontendConfig(api_key="sk-test"),
        connector=connector,
    )
    await session.start(RealtimeVoiceSessionConfig(session_id="voice-1"))
    events = session.events()
    await asyncio.wait_for(events.__anext__(), timeout=1)

    await fake_ws.emit({"type": "response.audio.delta", "delta": "not valid base64"})

    event = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert event.type == VoiceEventType.SESSION_ERROR
    assert event.payload["error"] == "invalid OpenAI output audio delta"

    await session.close()


@pytest.mark.asyncio
async def test_openai_realtime_barge_in_survives_provider_cancel_failure():
    fake_ws = FakeOpenAIWebSocket(fail_types={"response.cancel"})

    async def connector(_url, _headers, _timeout):
        return fake_ws

    session = OpenAIRealtimeFrontendSession(
        OpenAIRealtimeFrontendConfig(api_key="sk-test"),
        connector=connector,
    )
    await session.start(RealtimeVoiceSessionConfig(session_id="voice-1"))
    events = session.events()
    await asyncio.wait_for(events.__anext__(), timeout=1)

    await session.receive_event(
        VoiceEvent(
            type=VoiceEventType.BARGE_IN,
            session_id="voice-1",
            sequence=1,
            payload={"reason": "speech_started"},
        )
    )

    assert fake_ws.sent[-2] == {"type": "response.cancel"}
    assert fake_ws.sent[-1] == {"type": "input_audio_buffer.clear"}
    event = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert event.type == VoiceEventType.BARGE_IN

    await session.close()


@pytest.mark.asyncio
async def test_openai_realtime_close_cancels_response_clears_input_and_is_idempotent():
    fake_ws = FakeOpenAIWebSocket()

    async def connector(_url, _headers, _timeout):
        return fake_ws

    session = OpenAIRealtimeFrontendSession(
        OpenAIRealtimeFrontendConfig(api_key="sk-test"),
        connector=connector,
    )
    await session.start(RealtimeVoiceSessionConfig(session_id="voice-1"))

    await session.close()
    await session.close()

    assert fake_ws.sent[-2] == {"type": "response.cancel"}
    assert fake_ws.sent[-1] == {"type": "input_audio_buffer.clear"}
    assert fake_ws.closed is True
    assert [event["type"] for event in fake_ws.sent].count("response.cancel") == 1


def test_resample_pcm16_mono_converts_stereo_48k_to_mono_24k():
    stereo_48k_20ms = b"".join(
        (1000).to_bytes(2, "little", signed=True) + (-1000).to_bytes(2, "little", signed=True)
        for _ in range(960)
    )

    mono_24k = resample_pcm16_mono(stereo_48k_20ms, from_rate_hz=48000, to_rate_hz=24000, channels=2)

    assert len(mono_24k) == 960
    assert set(mono_24k) == {0}


def test_resample_pcm16_mono_keeps_mono_when_sample_rate_matches():
    pcm = b"".join(sample.to_bytes(2, "little", signed=True) for sample in (-1000, 0, 1000))

    assert resample_pcm16_mono(pcm, from_rate_hz=24000, to_rate_hz=24000, channels=1) == pcm


def test_resample_pcm16_mono_handles_short_and_odd_frames():
    one_sample = (1234).to_bytes(2, "little", signed=True)

    assert resample_pcm16_mono(b"\x01", from_rate_hz=24000, to_rate_hz=24000, channels=1) == b""
    assert resample_pcm16_mono(one_sample, from_rate_hz=24000, to_rate_hz=48000, channels=1) == (
        one_sample + one_sample
    )


def test_resample_pcm16_mono_keeps_extreme_samples_in_pcm16_range():
    pcm = b"".join(sample.to_bytes(2, "little", signed=True) for sample in (-32768, 32767))

    output = resample_pcm16_mono(pcm, from_rate_hz=24000, to_rate_hz=48000, channels=1)
    values = [
        int.from_bytes(output[index:index + 2], "little", signed=True)
        for index in range(0, len(output), 2)
    ]

    assert min(values) >= -32768
    assert max(values) <= 32767


@pytest.mark.asyncio
async def test_openai_realtime_requires_api_key():
    session = OpenAIRealtimeFrontendSession(OpenAIRealtimeFrontendConfig(api_key=""))

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await session.start(RealtimeVoiceSessionConfig(session_id="voice-1"))
