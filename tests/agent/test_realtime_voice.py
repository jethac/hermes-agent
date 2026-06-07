import asyncio

import pytest

from agent.realtime_voice import (
    AudioChunk,
    RealtimeVoiceEngineKind,
    RealtimeVoiceSessionConfig,
    VoiceAudioCodec,
    VoiceEvent,
    VoiceEventType,
    validate_client_event,
    validate_server_event,
)
from agent.realtime_voice_planner import RealtimeSpeechPlanner
from agent.realtime_voice_reference_sidecar import (
    ReferenceRealtimeVoiceSidecarSession,
    ReferenceSidecarRuntimeConfig,
)
from agent.realtime_voice_session import RealtimeVoiceSession, RealtimeVoiceSessionState
from agent.realtime_voice_s2s_engine import NativeS2SSidecarEngine
from agent.realtime_voice_sidecar import sidecar_ws_url, wants_realtime_sidecar
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
        await self._events.put(
            VoiceEvent(
                type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
                session_id=event.session_id,
                sequence=3,
                payload=AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"sidecar-audio").to_payload(),
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


def test_session_config_round_trips_wire_payload():
    config = RealtimeVoiceSessionConfig(
        session_id="voice-123",
        engine=RealtimeVoiceEngineKind.TEXT_ORACLE_TTS,
        input_codec=VoiceAudioCodec.OPUS,
        output_codec=VoiceAudioCodec.OPUS,
        frontend_provider="gemma",
        frontend_model="gemma-4-e4b",
        oracle_model="configured-hermes-model",
        tts_provider="edge",
        spark_base_url="http://spark.local:8080",
        metadata={"profile": "default"},
    )

    restored = RealtimeVoiceSessionConfig.from_wire(config.to_wire())

    assert restored == config


def test_audio_chunk_round_trips_base64_payload():
    chunk = AudioChunk(
        codec=VoiceAudioCodec.PCM16,
        data=b"\x00\x01\x02\x03",
        sample_rate_hz=24000,
        channels=1,
    )

    restored = AudioChunk.from_payload(chunk.to_payload())

    assert restored == chunk


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
        async def fake_speak(self, text):
            await self._emit(
                VoiceEventType.AUDIO_OUTPUT_CHUNK,
                AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"audio").to_payload(),
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
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
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


def test_text_engine_speaks_before_oracle_stream_finishes(monkeypatch):
    class StreamingOracle:
        async def stream_answer(self, transcript: str):
            yield "First sentence. "
            yield "Second sentence."

    async def run():
        spoken = []

        async def fake_speak(self, text):
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
            spark_base_url="http://spark.local:8080",
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
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                await engine.close()
                break

        assert sidecar.started is True
        assert sidecar.received[0].type == VoiceEventType.AUDIO_INPUT_CHUNK
        assert sidecar.spoken
        assert VoiceEventType.TRANSCRIPT_PARTIAL in [event.type for event in seen]
        assert VoiceEventType.TRANSCRIPT_FINAL in [event.type for event in seen]
        assert seen[-1].payload["text"] == "Answering: hello hermes."

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
        await engine.start(RealtimeVoiceSessionConfig(session_id="voice-123", spark_base_url="http://spark.local"))
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


def test_sidecar_config_detection_and_url_building():
    assert wants_realtime_sidecar(
        RealtimeVoiceSessionConfig(
            session_id="voice-123",
            frontend_provider="gemma",
            spark_base_url="http://spark.local:8080/base",
        )
    )
    assert not wants_realtime_sidecar(RealtimeVoiceSessionConfig(session_id="voice-123", frontend_provider="gemma"))
    assert sidecar_ws_url("http://spark.local:8080/base", "/v1/realtime-text/session") == (
        "ws://spark.local:8080/base/v1/realtime-text/session"
    )


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
                payload={"text": "hello back", "speak": True},
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


def test_session_persists_only_final_and_committed_messages(monkeypatch):
    async def run():
        async def fake_speak(self, text):
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


def test_native_s2s_engine_sends_oracle_hint_to_sidecar():
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
            spark_base_url="ws://spark.local",
        )
        engine._ws = ws
        engine._oracle = FakeOracle()

        await engine._send_oracle_hint("what is kame")

        assert ws.sent
        event = VoiceEvent.from_wire(__import__("json").loads(ws.sent[0]))
        assert event.type == VoiceEventType.ORACLE_HINT
        assert event.payload["text"] == "Answering: what is kame."

    asyncio.run(run())
