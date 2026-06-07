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
from agent.realtime_voice_session import RealtimeVoiceSession, RealtimeVoiceSessionState
from agent.realtime_voice_s2s_engine import NativeS2SSidecarEngine
from agent.realtime_voice_text_engine import TextOracleTTSEngine


class FakeOracle:
    async def answer(self, transcript: str) -> str:
        return f"Answering: {transcript}."

    async def stream_answer(self, transcript: str):
        yield "Answering: "
        yield f"{transcript}."


def test_session_config_round_trips_wire_payload():
    config = RealtimeVoiceSessionConfig(
        session_id="voice-123",
        engine=RealtimeVoiceEngineKind.TEXT_ORACLE_TTS,
        input_codec=VoiceAudioCodec.OPUS,
        output_codec=VoiceAudioCodec.OPUS,
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
