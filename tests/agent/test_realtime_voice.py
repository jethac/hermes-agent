import asyncio
import types

import pytest

from agent.realtime_voice import (
    AudioChunk,
    RealtimeVoiceEngineKind,
    RealtimeVoiceSessionConfig,
    VoiceAudioCodec,
    VoiceEvent,
    VoiceEventType,
    binary_audio_frame_from_event,
    event_from_binary_audio_frame,
    put_realtime_voice_event,
    validate_client_event,
    validate_server_event,
)
from agent.realtime_voice_planner import RealtimeSpeechPlanner
from agent.realtime_voice_reference_sidecar import (
    ReferenceRealtimeVoiceSidecarSession,
    ReferenceSidecarRuntimeConfig,
    create_reference_sidecar_app,
    reference_sidecar_health_payload,
)
from agent.realtime_voice_errors import sanitize_realtime_voice_error
from agent.realtime_voice_session import RealtimeVoiceSession, RealtimeVoiceSessionState
from agent.realtime_voice_s2s_engine import NativeS2SSidecarEngine
from agent.realtime_voice_sidecar import RealtimeVoiceSidecarClient, sidecar_ws_url, wants_realtime_sidecar
from agent.realtime_voice_text_engine import TextOracleTTSEngine


class FakeOracle:
    async def answer(self, transcript: str) -> str:
        return f"Answering: {transcript}."

    async def stream_answer(self, transcript: str):
        yield "Answering: "
        yield f"{transcript}."


class FakeSidecar:
    def __init__(self):
        self.started = False
        self.closed = False
        self.received = []
        self.spoken = []
        self._events = asyncio.Queue()

    async def start(self, config):
        self.started = True
        self.config = config

    async def send_event(self, event):
        self.received.append(event)
        if event.type == VoiceEventType.AUDIO_INPUT_CHUNK and event.payload.get("end_of_utterance") is True:
            await self._events.put(
                VoiceEvent(
                    type=VoiceEventType.TRANSCRIPT_PARTIAL,
                    session_id=event.session_id,
                    sequence=1,
                    payload={"text": "hello", "stability": 0.7},
                )
            )
            await self._events.put(
                VoiceEvent(
                    type=VoiceEventType.TRANSCRIPT_FINAL,
                    session_id=event.session_id,
                    sequence=2,
                    payload={"text": "hello hermes"},
                )
            )

    async def speak(self, event):
        self.spoken.append(event)
        payload = AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"sidecar-audio").to_payload()
        if "playback_generation" in event.payload:
            payload["playback_generation"] = event.payload["playback_generation"]
        await self._events.put(
            VoiceEvent(
                type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
                session_id=event.session_id,
                sequence=3,
                payload=payload,
            )
        )

    async def events(self):
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def close(self):
        self.closed = True
        await self._events.put(None)


def test_realtime_voice_error_sanitizer_redacts_credentials():
    sanitized = sanitize_realtime_voice_error(
        "failed Bearer secret-token at http://user:pass@voice.local:8765/v1?token=abc&api_key=def secret=raw"
    )

    assert sanitized == "failed Bearer *** at http://***@voice.local:8765/v1 secret=***"
    assert "secret-token" not in sanitized
    assert "user:pass" not in sanitized
    assert "token=abc" not in sanitized
    assert "api_key=def" not in sanitized


def test_session_config_round_trips_wire_payload():
    config = RealtimeVoiceSessionConfig(
        session_id="voice-123",
        engine=RealtimeVoiceEngineKind.TEXT_ORACLE_TTS,
        input_codec=VoiceAudioCodec.OPUS,
        output_codec=VoiceAudioCodec.OPUS,
        input_buffer_limit_bytes=4096,
        frontend_provider="gemma",
        frontend_model="gemma-4-e4b",
        oracle_model="configured-hermes-model",
        tts_provider="edge",
        sidecar_base_url="http://voice.local:8080",
        sidecar_token="secret-token",
        sidecar_connect_timeout_seconds=3.5,
        metadata={"profile": "default"},
    )

    restored = RealtimeVoiceSessionConfig.from_wire(config.to_wire())

    assert restored.to_wire() == config.to_wire()
    assert restored.effective_sidecar_base_url == "http://voice.local:8080"
    assert restored.effective_sidecar_token == "secret-token"
    assert restored.input_buffer_limit_bytes == 4096
    assert restored.sidecar_connect_timeout_seconds == 3.5


def test_session_config_accepts_legacy_spark_sidecar_wire_payload():
    restored = RealtimeVoiceSessionConfig.from_wire(
        {
            "session_id": "voice-123",
            "spark_base_url": "http://voice.local:8080",
            "spark_token": "legacy-token",
        }
    )

    assert restored.effective_sidecar_base_url == "http://voice.local:8080"
    assert restored.effective_sidecar_token == "legacy-token"


def test_audio_chunk_round_trips_base64_payload():
    chunk = AudioChunk(
        codec=VoiceAudioCodec.PCM16,
        data=b"\x00\x01\x02\x03",
        sample_rate_hz=24000,
        channels=1,
    )

    restored = AudioChunk.from_payload(chunk.to_payload())

    assert restored == chunk


def test_binary_audio_frame_round_trips_audio_event_payload():
    event = VoiceEvent(
        type=VoiceEventType.AUDIO_INPUT_CHUNK,
        session_id="voice-123",
        sequence=7,
        payload={
            **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"audio").to_payload(),
            "end_of_utterance": True,
        },
    )

    frame = binary_audio_frame_from_event(event)

    assert frame is not None
    restored = event_from_binary_audio_frame(frame, expected_type=VoiceEventType.AUDIO_INPUT_CHUNK)
    assert restored.type == VoiceEventType.AUDIO_INPUT_CHUNK
    assert restored.session_id == "voice-123"
    assert restored.sequence == 7
    assert restored.payload["end_of_utterance"] is True
    assert AudioChunk.from_payload(restored.payload).data == b"audio"


def test_realtime_voice_event_queue_drops_oldest_audio_for_control_event():
    async def run():
        queue = asyncio.Queue(maxsize=2)
        first_audio = VoiceEvent(
            type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
            session_id="voice-123",
            sequence=1,
            payload=AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"one").to_payload(),
        )
        second_audio = VoiceEvent(
            type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
            session_id="voice-123",
            sequence=2,
            payload=AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"two").to_payload(),
        )
        state = VoiceEvent(
            type=VoiceEventType.FRONTEND_STATE,
            session_id="voice-123",
            sequence=3,
            payload={"status": "degraded"},
        )

        assert await put_realtime_voice_event(queue, first_audio)
        assert await put_realtime_voice_event(queue, second_audio)
        assert await put_realtime_voice_event(queue, state)

        assert [queue.get_nowait().sequence, queue.get_nowait().sequence] == [2, 3]

    asyncio.run(run())


def test_realtime_voice_event_queue_drops_new_audio_when_control_queue_is_full():
    async def run():
        queue = asyncio.Queue(maxsize=1)
        state = VoiceEvent(
            type=VoiceEventType.FRONTEND_STATE,
            session_id="voice-123",
            sequence=1,
            payload={"status": "ok"},
        )
        audio = VoiceEvent(
            type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
            session_id="voice-123",
            sequence=2,
            payload=AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"audio").to_payload(),
        )

        assert await put_realtime_voice_event(queue, state)
        assert not await put_realtime_voice_event(queue, audio)
        assert queue.get_nowait().sequence == 1

    asyncio.run(run())


def test_event_validation_separates_client_and_server_events():
    audio_event = VoiceEvent(
        type=VoiceEventType.AUDIO_INPUT_CHUNK,
        session_id="voice-123",
        sequence=1,
        payload=AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"abc").to_payload(),
    )
    transcript_event = VoiceEvent(
        type=VoiceEventType.TRANSCRIPT_PARTIAL,
        session_id="voice-123",
        sequence=2,
        payload={"text": "hello"},
    )

    validate_client_event(audio_event)
    validate_server_event(transcript_event)

    with pytest.raises(ValueError):
        validate_client_event(transcript_event)

    with pytest.raises(ValueError):
        validate_server_event(audio_event)


def test_voice_event_round_trips_wire_payload():
    event = VoiceEvent(
        type=VoiceEventType.ORACLE_HINT,
        session_id="voice-123",
        sequence=7,
        timestamp_ms=123456,
        payload={"text": "answer from Hermes oracle", "confidence": 0.82},
    )

    restored = VoiceEvent.from_wire(event.to_wire())

    assert restored == event


def test_planner_suppresses_internal_markup_and_chunks_text():
    planner = RealtimeSpeechPlanner()

    planned = planner.plan(
        "<thinking>hidden</thinking>Here is the answer. MEDIA:/tmp/file.png "
        "[[audio_as_voice]]Second sentence!"
    )

    assert planned.committed_text == "Here is the answer. Second sentence!"
    assert planned.chunks == ["Here is the answer. Second sentence!"]


def test_text_engine_accepts_transcript_payload_and_emits_oracle_text(monkeypatch):
    async def run():
        async def fake_speak(self, text, playback_generation):
            await self._emit(
                VoiceEventType.AUDIO_OUTPUT_CHUNK,
                {
                    **AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"audio").to_payload(),
                    "playback_generation": playback_generation,
                },
            )

        monkeypatch.setattr(TextOracleTTSEngine, "_speak_chunk", fake_speak)

        config = RealtimeVoiceSessionConfig(session_id="voice-123")
        engine = TextOracleTTSEngine(oracle=FakeOracle())
        await engine.start(config)
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "hello hermes"},
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            event_types = [event.type for event in seen]
            if (
                VoiceEventType.ASSISTANT_COMMIT in event_types
                and VoiceEventType.AUDIO_OUTPUT_CHUNK in event_types
            ):
                await engine.close()
                break

        assert [event.type for event in seen] == [
            VoiceEventType.SESSION_STARTED,
            VoiceEventType.TRANSCRIPT_FINAL,
            VoiceEventType.ASSISTANT_TEXT_PARTIAL,
            VoiceEventType.AUDIO_OUTPUT_CHUNK,
            VoiceEventType.ASSISTANT_COMMIT,
        ]
        assert seen[-1].payload["text"] == "Answering: hello hermes."

    asyncio.run(run())


def test_text_engine_keeps_explicit_partial_transcript_out_of_oracle():
    class TrackingOracle(FakeOracle):
        def __init__(self):
            self.called = False

        async def stream_answer(self, transcript: str):
            self.called = True
            yield "should not happen"

    async def run():
        oracle = TrackingOracle()
        engine = TextOracleTTSEngine(oracle=oracle)
        await engine.start(RealtimeVoiceSessionConfig(session_id="voice-123"))
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "hello her", "end_of_utterance": False},
            )
        )

        events = [await anext(engine.events()), await anext(engine.events())]

        assert [event.type for event in events] == [
            VoiceEventType.SESSION_STARTED,
            VoiceEventType.TRANSCRIPT_PARTIAL,
        ]
        assert events[-1].payload["text"] == "hello her"
        assert oracle.called is False
        await engine.close()

    asyncio.run(run())


def test_text_engine_speaks_before_oracle_stream_finishes(monkeypatch):
    class StreamingOracle:
        async def stream_answer(self, transcript: str):
            yield "First sentence. "
            yield "Second sentence."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(TextOracleTTSEngine, "_speak_chunk", fake_speak)

        engine = TextOracleTTSEngine(oracle=StreamingOracle())
        await engine.start(RealtimeVoiceSessionConfig(session_id="voice-123"))
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "hello"},
            )
        )

        async for event in engine.events():
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                await engine.close()
                break

        assert spoken == ["First sentence.", "Second sentence."]

    asyncio.run(run())


def test_text_engine_streams_audio_to_sidecar_then_uses_hermes_oracle():
    async def run():
        sidecar = FakeSidecar()
        config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            frontend_provider="gemma",
            frontend_model="gemma-4-e4b",
            sidecar_base_url="http://voice.local:8080",
        )
        engine = TextOracleTTSEngine(oracle=FakeOracle(), sidecar=sidecar)
        await engine.start(config)
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"audio").to_payload(),
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            event_types = [seen_event.type for seen_event in seen]
            if (
                VoiceEventType.ASSISTANT_COMMIT in event_types
                and VoiceEventType.AUDIO_OUTPUT_CHUNK in event_types
            ):
                await engine.close()
                break

        assert sidecar.started is True
        assert sidecar.received[0].type == VoiceEventType.AUDIO_INPUT_CHUNK
        assert sidecar.spoken
        assert sidecar.spoken[0].payload["playback_generation"] == 1
        assert VoiceEventType.TRANSCRIPT_PARTIAL in [event.type for event in seen]
        assert VoiceEventType.TRANSCRIPT_FINAL in [event.type for event in seen]
        commit_events = [event for event in seen if event.type == VoiceEventType.ASSISTANT_COMMIT]
        assert commit_events[0].payload["text"] == "Answering: hello hermes."
        assert commit_events[0].payload["playback_generation"] == 1
        audio_events = [event for event in seen if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK]
        assert audio_events[0].payload["playback_generation"] == 1

    asyncio.run(run())


def test_text_engine_reports_sidecar_start_failure_as_frontend_fallback():
    class FailingStartSidecar:
        async def start(self, config):
            raise RuntimeError("sidecar down at http://user:pass@voice.local:8765/v1?token=abc")

    async def run():
        engine = TextOracleTTSEngine(oracle=FakeOracle(), sidecar=FailingStartSidecar())
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                frontend_provider="gemma4",
                sidecar_base_url="http://voice.local:8080",
            )
        )

        events = [await anext(engine.events()), await anext(engine.events())]

        assert [event.type for event in events] == [
            VoiceEventType.FRONTEND_STATE,
            VoiceEventType.SESSION_STARTED,
        ]
        assert events[0].payload["status"] == "fallback"
        assert events[0].payload["reason"] == "sidecar_unavailable"
        assert events[0].payload["sidecar"] is False
        assert "sidecar down" in events[0].payload["error"]
        assert "user:pass" not in events[0].payload["error"]
        assert "token=abc" not in events[0].payload["error"]
        assert events[1].payload["sidecar"] is False

    asyncio.run(run())


def test_text_engine_falls_back_to_local_stt_when_sidecar_send_fails(monkeypatch):
    class FailingSendSidecar(FakeSidecar):
        async def send_event(self, event):
            raise RuntimeError("send failed")

    async def run():
        async def fake_speak(self, text, playback_generation):
            return None

        monkeypatch.setattr(TextOracleTTSEngine, "_speak_chunk", fake_speak)
        monkeypatch.setattr(TextOracleTTSEngine, "_transcribe_sync", lambda self, audio, codec: "local transcript")

        engine = TextOracleTTSEngine(oracle=FakeOracle(), sidecar=FailingSendSidecar())
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                frontend_provider="gemma4",
                sidecar_base_url="http://voice.local:8080",
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"audio").to_payload(),
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                await engine.close()
                break

        fallback = next(event for event in seen if event.type == VoiceEventType.FRONTEND_STATE)
        final = next(event for event in seen if event.type == VoiceEventType.TRANSCRIPT_FINAL)

        assert fallback.payload["status"] == "fallback"
        assert fallback.payload["reason"] == "sidecar_send_failed"
        assert fallback.payload["sidecar"] is False
        assert final.payload["text"] == "local transcript"
        assert VoiceEventType.SESSION_ERROR not in [event.type for event in seen]

    asyncio.run(run())


def test_text_engine_bounds_local_audio_fallback_buffer():
    async def run():
        engine = TextOracleTTSEngine(oracle=FakeOracle())
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                input_buffer_limit_bytes=3,
            )
        )

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload=AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"ab").to_payload(),
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload=AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"cd").to_payload(),
            )
        )

        events = [await anext(engine.events()), await anext(engine.events())]
        await engine.close()

        assert events[0].type == VoiceEventType.SESSION_STARTED
        assert events[1].type == VoiceEventType.FRONTEND_STATE
        assert events[1].payload["status"] == "degraded"
        assert events[1].payload["reason"] == "input_buffer_limit_exceeded"
        assert events[1].payload["limit_bytes"] == 3
        assert engine._inbound_audio_bytes == 0
        assert VoiceEventType.SESSION_ERROR not in [event.type for event in events]

    asyncio.run(run())


def test_text_engine_barge_in_interrupts_oracle_and_sidecar():
    class InterruptibleOracle(FakeOracle):
        def __init__(self):
            self.interrupted = False

        def interrupt(self, message: str = ""):
            self.interrupted = True

    async def run():
        oracle = InterruptibleOracle()
        sidecar = FakeSidecar()
        engine = TextOracleTTSEngine(oracle=oracle, sidecar=sidecar)
        await engine.start(RealtimeVoiceSessionConfig(session_id="voice-123", sidecar_base_url="http://voice.local"))
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.BARGE_IN,
                session_id="voice-123",
                sequence=1,
                payload={"reason": "test"},
            )
        )
        event = await anext(engine.events())
        assert event.type == VoiceEventType.SESSION_STARTED
        event = await anext(engine.events())
        assert event.type == VoiceEventType.BARGE_IN
        assert oracle.interrupted is True
        assert sidecar.received[0].type == VoiceEventType.BARGE_IN
        await engine.close()

    asyncio.run(run())


def test_text_engine_suppresses_stale_cancelled_commit_when_new_turn_starts(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.release = asyncio.Event()

        async def stream_answer(self, transcript: str):
            if transcript == "first":
                yield "First answer starts. "
                await self.release.wait()
                yield "stale"
            else:
                yield "Second answer."

    async def run():
        async def fake_speak(self, text, playback_generation):
            return None

        monkeypatch.setattr(TextOracleTTSEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = TextOracleTTSEngine(oracle=oracle)
        await engine.start(RealtimeVoiceSessionConfig(session_id="voice-123"))
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "first"},
            )
        )

        seen = []
        while True:
            event = await anext(engine.events())
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL:
                break

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload={"transcript": "second"},
            )
        )

        while True:
            event = await anext(engine.events())
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        commit_events = [event for event in seen if event.type == VoiceEventType.ASSISTANT_COMMIT]
        assert commit_events == [
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_COMMIT,
                session_id="voice-123",
                sequence=6,
                payload={"text": "Second answer.", "playback_generation": 2},
                timestamp_ms=commit_events[0].timestamp_ms,
            )
        ]
        assert not any(
            event.payload.get("interrupted") is True and event.payload.get("playback_generation") == 1
            for event in seen
        )
        await engine.close()

    asyncio.run(run())


def test_sidecar_config_detection_and_url_building():
    assert wants_realtime_sidecar(
        RealtimeVoiceSessionConfig(
            session_id="voice-123",
            frontend_provider="gemma",
            sidecar_base_url="http://voice.local:8080/base",
        )
    )
    assert not wants_realtime_sidecar(RealtimeVoiceSessionConfig(session_id="voice-123", frontend_provider="gemma"))
    assert sidecar_ws_url("http://voice.local:8080/base", "/v1/realtime-text/session") == (
        "ws://voice.local:8080/base/v1/realtime-text/session"
    )


def test_sidecar_detection_does_not_depend_on_hardware_aliases():
    assert not wants_realtime_sidecar(
        RealtimeVoiceSessionConfig(
            session_id="voice-123",
            frontend_provider="pgx",
            sidecar_base_url="http://voice.local:8080",
        )
    )
    assert not wants_realtime_sidecar(
        RealtimeVoiceSessionConfig(
            session_id="voice-123",
            frontend_provider="dgx",
            sidecar_base_url="http://voice.local:8080",
        )
    )
    assert wants_realtime_sidecar(
        RealtimeVoiceSessionConfig(
            session_id="voice-123",
            frontend_model="google/gemma-4-E4B-it-qat-w4a16-ct",
            sidecar_base_url="http://voice.local:8080",
        )
    )


def test_sidecar_client_uses_configured_connect_timeout(monkeypatch):
    class FakeWs:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(payload)

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def close(self):
            return None

    async def run():
        captured = {}
        fake_ws = FakeWs()

        async def fake_connect(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return fake_ws

        async def fake_wait_for(awaitable, timeout):
            captured["timeout"] = timeout
            return await awaitable

        monkeypatch.setitem(__import__("sys").modules, "websockets", types.SimpleNamespace(connect=fake_connect))
        monkeypatch.setattr("agent.realtime_voice_sidecar.asyncio.wait_for", fake_wait_for)

        client = RealtimeVoiceSidecarClient()
        await client.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                frontend_provider="gemma4",
                sidecar_base_url="http://voice.local:8765",
                sidecar_connect_timeout_seconds=2.5,
            )
        )

        assert captured["url"] == "ws://voice.local:8765/v1/realtime-text/session"
        assert captured["timeout"] == 2.5
        assert fake_ws.sent
        assert __import__("json").loads(fake_ws.sent[0])["type"] == "session.config"
        await client.close()

    asyncio.run(run())


def test_sidecar_client_sends_audio_input_as_binary_frame(monkeypatch):
    class FakeWs:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(payload)

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def close(self):
            return None

    async def run():
        fake_ws = FakeWs()

        async def fake_connect(url, **kwargs):
            return fake_ws

        async def fake_wait_for(awaitable, timeout):
            return await awaitable

        monkeypatch.setitem(__import__("sys").modules, "websockets", types.SimpleNamespace(connect=fake_connect))
        monkeypatch.setattr("agent.realtime_voice_sidecar.asyncio.wait_for", fake_wait_for)

        client = RealtimeVoiceSidecarClient()
        await client.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                frontend_provider="gemma4",
                sidecar_base_url="http://voice.local:8765",
            )
        )

        await client.send_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload=AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"mic-audio").to_payload(),
            )
        )

        assert isinstance(fake_ws.sent[0], str)
        assert isinstance(fake_ws.sent[-1], bytes)
        restored = event_from_binary_audio_frame(fake_ws.sent[-1], expected_type=VoiceEventType.AUDIO_INPUT_CHUNK)
        assert restored.sequence == 2
        assert AudioChunk.from_payload(restored.payload).data == b"mic-audio"
        await client.close()

    asyncio.run(run())


def test_sidecar_client_accepts_binary_audio_output_frame(monkeypatch):
    class FakeWs:
        def __init__(self, items):
            self._items = list(items)
            self.sent = []

        async def send(self, payload):
            self.sent.append(payload)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

        async def close(self):
            return None

    async def run():
        output = VoiceEvent(
            type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
            session_id="sidecar-session",
            sequence=9,
            payload={
                **AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"speaker-audio").to_payload(),
                "playback_generation": 3,
            },
        )
        fake_ws = FakeWs([binary_audio_frame_from_event(output)])

        async def fake_connect(url, **kwargs):
            return fake_ws

        async def fake_wait_for(awaitable, timeout):
            return await awaitable

        monkeypatch.setitem(__import__("sys").modules, "websockets", types.SimpleNamespace(connect=fake_connect))
        monkeypatch.setattr("agent.realtime_voice_sidecar.asyncio.wait_for", fake_wait_for)

        client = RealtimeVoiceSidecarClient()
        await client.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                frontend_provider="gemma4",
                sidecar_base_url="http://voice.local:8765",
            )
        )

        event = await asyncio.wait_for(client._events.get(), timeout=1)
        assert event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK
        assert event.sequence == 9
        assert event.payload["playback_generation"] == 3
        assert AudioChunk.from_payload(event.payload).data == b"speaker-audio"
        await client.close()

    asyncio.run(run())


def test_reference_sidecar_accepts_transcript_payloads_without_gpu():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        await sidecar.start(RealtimeVoiceSessionConfig(session_id="voice-123", frontend_provider="local"))
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "hello hermes"},
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload={"transcript": "hello hermes", "end_of_utterance": True},
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if event.type == VoiceEventType.TRANSCRIPT_FINAL:
                await sidecar.close()
                break

        assert [event.type for event in seen] == [
            VoiceEventType.FRONTEND_STATE,
            VoiceEventType.TRANSCRIPT_PARTIAL,
            VoiceEventType.TRANSCRIPT_FINAL,
        ]
        assert seen[-1].payload["text"] == "hello hermes"

    asyncio.run(run())


def test_reference_sidecar_local_stt_and_tts_without_gpu(tmp_path):
    def fake_transcribe(path):
        assert path
        return {"success": True, "transcript": "local transcript"}

    def fake_synthesize(text):
        audio = tmp_path / "speech.ogg"
        audio.write_bytes(b"audio")
        return {"success": True, "file_path": str(audio)}

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(vllm_base_url=None, vllm_model=None),
            transcribe_audio_func=fake_transcribe,
            synthesize_func=fake_synthesize,
        )
        await sidecar.start(RealtimeVoiceSessionConfig(session_id="voice-123", frontend_provider="local"))
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"audio").to_payload(),
                    "end_of_utterance": True,
                },
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                session_id="voice-123",
                sequence=2,
                payload={"text": "hello back", "speak": True, "playback_generation": 7},
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            event_types = [event.type for event in seen]
            if (
                VoiceEventType.AUDIO_OUTPUT_CHUNK in event_types
                and VoiceEventType.TRANSCRIPT_FINAL in event_types
            ):
                await sidecar.close()
                break

        assert VoiceEventType.TRANSCRIPT_FINAL in [event.type for event in seen]
        assert [event.payload.get("text") for event in seen if event.type == VoiceEventType.TRANSCRIPT_FINAL] == [
            "local transcript"
        ]
        audio_events = [event for event in seen if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK]
        assert audio_events[0].payload["data_b64"]
        assert audio_events[0].payload["playback_generation"] == 7

    asyncio.run(run())


def test_reference_sidecar_vllm_audio_frontend(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"vllm transcript"}}]}'

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["body"] = __import__("json").loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("agent.realtime_voice_reference_sidecar.urllib.request.urlopen", fake_urlopen)

    sidecar = ReferenceRealtimeVoiceSidecarSession(
        ReferenceSidecarRuntimeConfig(
            vllm_base_url="http://vllm.local:8000/v1",
            vllm_model="google/gemma-4-E4B-it-qat-w4a16-ct",
            vllm_timeout_seconds=12,
        )
    )

    transcript = sidecar._transcribe_sync(b"audio", VoiceAudioCodec.WEBM_OPUS)

    assert transcript == "vllm transcript"
    assert captured["url"] == "http://vllm.local:8000/v1/chat/completions"
    assert captured["timeout"] == 12
    assert captured["body"]["model"] == "google/gemma-4-E4B-it-qat-w4a16-ct"
    assert captured["body"]["messages"][0]["content"][0]["type"] == "audio_url"


def test_reference_sidecar_health_payload_is_sanitized():
    payload = reference_sidecar_health_payload(
        ReferenceSidecarRuntimeConfig(
            vllm_base_url="http://user:secret@voice.local:8000/v1",
            vllm_model="google/gemma-4-E4B-it-qat-w4a16-ct",
            local_stt_enabled=False,
            local_tts_enabled=True,
        )
    )

    assert payload == {
        "ok": True,
        "kind": "reference",
        "frontend": {
            "provider": "vllm",
            "model": "google/gemma-4-E4B-it-qat-w4a16-ct",
        },
        "capabilities": {
            "utterance_stt": True,
            "streaming_stt": False,
            "tts": True,
            "native_s2s": False,
            "vllm_audio_frontend": True,
        },
        "local": {
            "stt": False,
            "tts": True,
        },
    }
    assert "secret" not in __import__("json").dumps(payload)


def test_reference_sidecar_health_requires_bearer_token():
    from fastapi.testclient import TestClient

    client = TestClient(
        create_reference_sidecar_app(
            ReferenceSidecarRuntimeConfig(
                vllm_base_url="http://user:secret@voice.local:8000/v1",
                vllm_model="google/gemma-4-E4B-it-qat-w4a16-ct",
                auth_token="secret-token",
            )
        )
    )

    assert client.get("/health").status_code == 401
    response = client.get("/health", headers={"Authorization": "Bearer secret-token"})

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert "secret-token" not in __import__("json").dumps(response.json())
    assert "user:secret" not in __import__("json").dumps(response.json())


def test_reference_sidecar_websocket_requires_bearer_token():
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(create_reference_sidecar_app(ReferenceSidecarRuntimeConfig(auth_token="secret-token")))

    with pytest.raises(WebSocketDisconnect) as unauthorized:
        with client.websocket_connect("/v1/realtime-text/session"):
            pass

    assert unauthorized.value.code == 1008

    with client.websocket_connect(
        "/v1/realtime-text/session",
        headers={"Authorization": "Bearer secret-token"},
    ) as ws:
        ws.send_json(
            {
                "type": "session.config",
                "payload": RealtimeVoiceSessionConfig(session_id="voice-123").to_wire(),
            }
        )
        response = ws.receive_json()

    assert response["type"] == "frontend.state"
    assert response["session_id"] == "voice-123"


def test_session_persists_only_final_and_committed_messages(monkeypatch):
    async def run():
        async def fake_speak(self, text, playback_generation):
            return None

        monkeypatch.setattr(TextOracleTTSEngine, "_speak_chunk", fake_speak)

        config = RealtimeVoiceSessionConfig(session_id="voice-123")
        session = RealtimeVoiceSession(config, engine=TextOracleTTSEngine(oracle=FakeOracle()))
        await session.start()
        await session.receive_client_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "what is kame"},
            )
        )

        async for event in session.events():
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        assert session.state == RealtimeVoiceSessionState.LISTENING
        assert session.durable_messages() == [
            {"role": "user", "content": "what is kame"},
            {"role": "assistant", "content": "Answering: what is kame."},
        ]
        await session.close()

    asyncio.run(run())


def test_session_ignores_stale_interrupted_commit_from_prior_generation():
    async def run():
        session = RealtimeVoiceSession(
            RealtimeVoiceSessionConfig(session_id="voice-123"),
            engine=TextOracleTTSEngine(oracle=FakeOracle()),
        )
        session._apply_server_event(
            VoiceEvent(
                type=VoiceEventType.TRANSCRIPT_FINAL,
                session_id="voice-123",
                sequence=1,
                payload={"text": "first", "playback_generation": 1},
            )
        )
        session._apply_server_event(
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                session_id="voice-123",
                sequence=2,
                payload={"text": "new draft", "playback_generation": 2},
            )
        )
        session._apply_server_event(
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_COMMIT,
                session_id="voice-123",
                sequence=3,
                payload={"interrupted": True, "text": "", "playback_generation": 1},
            )
        )

        assert session.transcript.assistant_draft == "new draft"
        assert session.transcript.interrupted_assistant_segments == []

    asyncio.run(run())


def test_session_adds_latency_metrics_to_realtime_events(monkeypatch):
    async def run():
        async def fake_speak(self, text, playback_generation):
            await self._emit(
                VoiceEventType.AUDIO_OUTPUT_CHUNK,
                {
                    **AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"audio").to_payload(),
                    "playback_generation": playback_generation,
                },
            )

        monkeypatch.setattr(TextOracleTTSEngine, "_speak_chunk", fake_speak)

        session = RealtimeVoiceSession(
            RealtimeVoiceSessionConfig(session_id="voice-123"),
            engine=TextOracleTTSEngine(oracle=FakeOracle()),
        )
        await session.start()
        await session.receive_client_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "hello metrics", "end_of_utterance": True},
            )
        )

        seen = []
        async for event in session.events():
            seen.append(event)
            event_types = [seen_event.type for seen_event in seen]
            if (
                VoiceEventType.ASSISTANT_COMMIT in event_types
                and VoiceEventType.AUDIO_OUTPUT_CHUNK in event_types
            ):
                break

        final = next(event for event in seen if event.type == VoiceEventType.TRANSCRIPT_FINAL)
        text = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL)
        audio = next(event for event in seen if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK)

        assert final.payload["metrics"]["session_elapsed_ms"] >= 0
        assert final.payload["metrics"]["audio_to_final_transcript_ms"] >= 0
        assert final.payload["metrics"]["eou_to_final_transcript_ms"] >= 0
        assert text.payload["metrics"]["final_transcript_to_first_text_ms"] >= 0
        assert audio.payload["metrics"]["final_transcript_to_first_audio_ms"] >= 0
        await session.close()

    asyncio.run(run())


def test_session_adds_turn_state_to_realtime_events(monkeypatch):
    async def run():
        async def fake_speak(self, text, playback_generation):
            return None

        monkeypatch.setattr(TextOracleTTSEngine, "_speak_chunk", fake_speak)

        session = RealtimeVoiceSession(
            RealtimeVoiceSessionConfig(session_id="voice-123"),
            engine=TextOracleTTSEngine(oracle=FakeOracle()),
        )
        await session.start()
        await session.receive_client_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "hello state", "end_of_utterance": True},
            )
        )

        seen = []
        async for event in session.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        states = {event.type: event.payload.get("session_state") for event in seen}

        assert states[VoiceEventType.SESSION_STARTED] == RealtimeVoiceSessionState.LISTENING.value
        assert states[VoiceEventType.TRANSCRIPT_FINAL] == RealtimeVoiceSessionState.ASSISTANT_PENDING.value
        assert states[VoiceEventType.ASSISTANT_TEXT_PARTIAL] == RealtimeVoiceSessionState.SPEAKING.value
        assert states[VoiceEventType.ASSISTANT_COMMIT] == RealtimeVoiceSessionState.LISTENING.value
        await session.close()

    asyncio.run(run())


def test_session_adds_barge_in_ack_latency_metric():
    async def run():
        session = RealtimeVoiceSession(
            RealtimeVoiceSessionConfig(session_id="voice-123"),
            engine=TextOracleTTSEngine(oracle=FakeOracle()),
        )
        await session.start()
        await session.receive_client_event(
            VoiceEvent(
                type=VoiceEventType.BARGE_IN,
                session_id="voice-123",
                sequence=1,
                payload={"reason": "test"},
            )
        )

        seen = []
        async for event in session.events():
            seen.append(event)
            if event.type == VoiceEventType.BARGE_IN:
                break

        barge_in = seen[-1]
        assert barge_in.payload["metrics"]["barge_in_ack_ms"] >= 0
        assert barge_in.payload["metrics"]["session_elapsed_ms"] >= 0
        await session.close()

    asyncio.run(run())


def test_native_s2s_engine_streams_oracle_hint_to_sidecar():
    class FakeWs:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(payload)

    async def run():
        ws = FakeWs()
        engine = NativeS2SSidecarEngine()
        engine.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE,
            sidecar_base_url="ws://voice.local",
        )
        engine._ws = ws
        engine._oracle = FakeOracle()

        await engine._send_oracle_hint("what is kame")

        events = [VoiceEvent.from_wire(__import__("json").loads(raw)) for raw in ws.sent]
        assert [event.type for event in events] == [
            VoiceEventType.ORACLE_HINT,
            VoiceEventType.ORACLE_HINT,
            VoiceEventType.ORACLE_HINT,
        ]
        assert events[0].payload == {
            "text": "Answering: ",
            "delta": "Answering: ",
            "final": False,
            "source": "hermes",
        }
        assert events[1].payload["delta"] == "what is kame."
        assert events[-1].payload == {
            "text": "Answering: what is kame.",
            "delta": "",
            "final": True,
            "source": "hermes",
        }

    asyncio.run(run())


def test_native_s2s_engine_uses_configured_connect_timeout(monkeypatch):
    class FakeWs:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(payload)

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def close(self):
            return None

    async def run():
        captured = {}
        fake_ws = FakeWs()

        async def fake_connect(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return fake_ws

        async def fake_wait_for(awaitable, timeout):
            captured["timeout"] = timeout
            return await awaitable

        monkeypatch.setitem(__import__("sys").modules, "websockets", types.SimpleNamespace(connect=fake_connect))
        monkeypatch.setattr("agent.realtime_voice_s2s_engine.asyncio.wait_for", fake_wait_for)

        engine = NativeS2SSidecarEngine()
        engine.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE,
            sidecar_base_url="ws://voice.local",
            sidecar_connect_timeout_seconds=4.0,
        )

        await engine._connect_sidecar(engine.config)

        assert captured["url"] == "ws://voice.local/v1/s2s/session"
        assert captured["timeout"] == 4.0
        assert fake_ws.sent
        await engine.close()

    asyncio.run(run())


def test_native_s2s_engine_sends_audio_input_as_binary_frame():
    class FakeWs:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(payload)

    async def run():
        ws = FakeWs()
        engine = NativeS2SSidecarEngine()
        engine.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE,
            sidecar_base_url="ws://voice.local",
        )
        engine._ws = ws

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=12,
                payload=AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"s2s-mic").to_payload(),
            )
        )

        assert isinstance(ws.sent[-1], bytes)
        event = event_from_binary_audio_frame(ws.sent[-1], expected_type=VoiceEventType.AUDIO_INPUT_CHUNK)
        assert event.sequence == 12
        assert AudioChunk.from_payload(event.payload).data == b"s2s-mic"

    asyncio.run(run())


def test_native_s2s_engine_normalizes_sidecar_generation_and_session():
    engine = NativeS2SSidecarEngine()
    engine.config = RealtimeVoiceSessionConfig(
        session_id="voice-123",
        engine=RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE,
        sidecar_base_url="ws://voice.local",
    )

    transcript = engine._normalize_sidecar_event(
        VoiceEvent(
            type=VoiceEventType.TRANSCRIPT_FINAL,
            session_id="sidecar-session",
            sequence=50,
            payload={"text": "hello"},
        ),
        new_generation=True,
    )
    audio = engine._normalize_sidecar_event(
        VoiceEvent(
            type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
            session_id="sidecar-session",
            sequence=51,
            payload=AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"audio").to_payload(),
        )
    )

    assert transcript.session_id == "voice-123"
    assert transcript.payload["playback_generation"] == 1
    assert audio.session_id == "voice-123"
    assert audio.payload["playback_generation"] == 1


def test_native_s2s_engine_accepts_binary_audio_output_frame():
    class FakeWs:
        def __init__(self, items):
            self._items = list(items)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

    async def run():
        output = VoiceEvent(
            type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
            session_id="sidecar-session",
            sequence=44,
            payload={
                **AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"s2s-speaker").to_payload(),
                "playback_generation": 6,
            },
        )
        engine = NativeS2SSidecarEngine()
        engine.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE,
            sidecar_base_url="ws://voice.local",
        )
        engine._ws = FakeWs([binary_audio_frame_from_event(output)])

        await engine._read_sidecar()

        event = await engine._events.get()
        assert event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK
        assert event.session_id == "voice-123"
        assert event.payload["playback_generation"] == 6
        assert AudioChunk.from_payload(event.payload).data == b"s2s-speaker"

    asyncio.run(run())


def test_native_s2s_engine_tags_legacy_raw_audio_with_active_generation():
    class FakeWs:
        def __init__(self):
            self._items = [b"audio"]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

    async def run():
        engine = NativeS2SSidecarEngine()
        engine.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE,
            sidecar_base_url="ws://voice.local",
        )
        engine._ws = FakeWs()
        engine._playback_generation = 3

        await engine._read_sidecar()

        event = await engine._events.get()
        assert event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK
        assert event.payload["playback_generation"] == 3

    asyncio.run(run())


def test_native_s2s_engine_barge_in_cancels_active_oracle_hint():
    class FakeWs:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(payload)

    class SlowOracle:
        def __init__(self):
            self.interrupted = []

        async def stream_answer(self, transcript):
            await asyncio.sleep(30)
            yield transcript

        def interrupt(self, message=""):
            self.interrupted.append(message)

    async def run():
        ws = FakeWs()
        oracle = SlowOracle()
        engine = NativeS2SSidecarEngine()
        engine.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE,
            sidecar_base_url="ws://voice.local",
        )
        engine._ws = ws
        engine._oracle = oracle
        engine._playback_generation = 1

        engine._start_oracle_hint("slow question", 1)
        await asyncio.sleep(0)
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.BARGE_IN,
                session_id="voice-123",
                sequence=1,
                payload={"reason": "user_speech"},
            )
        )
        await asyncio.sleep(0)

        assert oracle.interrupted
        assert engine._oracle_hint_task is None or engine._oracle_hint_task.cancelled()
        forwarded = VoiceEvent.from_wire(__import__("json").loads(ws.sent[-1]))
        assert forwarded.type == VoiceEventType.BARGE_IN
        assert forwarded.payload["playback_generation"] == 2

        ack = await engine._events.get()
        assert ack.type == VoiceEventType.BARGE_IN
        assert ack.payload["playback_generation"] == 2

    asyncio.run(run())


def test_native_s2s_engine_close_awaits_reader_task():
    class FakeWs:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    async def run():
        cancelled = {"reader": False}

        async def reader():
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled["reader"] = True
                raise

        ws = FakeWs()
        engine = NativeS2SSidecarEngine()
        engine.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE,
            sidecar_base_url="ws://voice.local",
        )
        engine._ws = ws
        engine._reader_task = asyncio.create_task(reader())
        await asyncio.sleep(0)

        await engine.close()

        assert cancelled["reader"] is True
        assert engine._reader_task.done()
        assert ws.closed is True

    asyncio.run(run())
