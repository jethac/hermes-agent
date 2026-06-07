import asyncio
import json
import types

import pytest

import agent.realtime_voice_reference_sidecar as reference_sidecar_module
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
    transcript_event_payload_from_payload,
    validate_client_event,
    validate_server_event,
)
from agent.realtime_voice_planner import RealtimeSpeechPlanner
from agent.realtime_voice_reference_sidecar import (
    ReferenceRealtimeVoiceSidecarSession,
    ReferenceSidecarRuntimeConfig,
    create_reference_sidecar_app,
    reference_sidecar_health_payload,
    runtime_config_from_env,
)
from agent.realtime_voice_errors import sanitize_realtime_voice_error
from agent.realtime_voice_oracle import _voice_oracle_prompt
from agent.realtime_voice_session import RealtimeVoiceSession, RealtimeVoiceSessionState
from agent.realtime_voice_s2s_engine import NativeS2SSidecarEngine
from agent.realtime_voice_sidecar import RealtimeVoiceSidecarClient, sidecar_ws_url, wants_realtime_sidecar
from agent.realtime_voice_text_engine import TextOracleTTSEngine, _take_speakable_chunk


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
            transcript_payload = {}
            if "input_generation" in event.payload:
                transcript_payload["input_generation"] = event.payload["input_generation"]
            await self._events.put(
                VoiceEvent(
                    type=VoiceEventType.TRANSCRIPT_PARTIAL,
                    session_id=event.session_id,
                    sequence=1,
                    payload={
                        "text": "hello",
                        "stability": 0.7,
                        "language_url": "https://voice.local/secret",
                        **transcript_payload,
                    },
                )
            )
            await self._events.put(
                VoiceEvent(
                    type=VoiceEventType.TRANSCRIPT_FINAL,
                    session_id=event.session_id,
                    sequence=2,
                    payload={
                        "text": "hello hermes",
                        "language": "ja",
                        "script": "Jpan",
                        "language_url": "https://voice.local/secret",
                        **transcript_payload,
                    },
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


def test_transcript_event_payload_keeps_sanitized_non_target_language_metadata():
    payload = transcript_event_payload_from_payload(
        {
            "text": "안녕하세요",
            "language": "ko",
            "locale": "ko-KR",
            "script": "Kore",
            "confidence": 1.5,
            "stability": -0.2,
            "input_generation": "7",
            "playback_generation": -1,
            "language_url": "https://voice.local/secret",
            "raw_metadata": {"token": "secret"},
        }
    )

    assert payload == {
        "text": "안녕하세요",
        "confidence": 1.0,
        "stability": 0.0,
        "input_generation": 7,
        "language": "ko",
        "locale": "ko-KR",
        "script": "Kore",
    }


def test_planner_suppresses_internal_markup_and_chunks_text():
    planner = RealtimeSpeechPlanner()

    planned = planner.plan(
        "<thinking>hidden</thinking>Here is the answer. MEDIA:/tmp/file.png "
        "[[audio_as_voice]]Second sentence!"
    )

    assert planned.committed_text == "Here is the answer. Second sentence!"
    assert planned.chunks == ["Here is the answer. Second sentence!"]


def test_planner_chunks_multilingual_sentence_boundaries():
    planner = RealtimeSpeechPlanner()

    assert planner.plan("これは最初の長い返答です。続きもあります").chunks == [
        "これは最初の長い返答です。",
        "続きもあります",
    ]
    assert planner.plan("هذا رد طويل بما يكفي؟وهذه متابعة").chunks == [
        "هذا رد طويل بما يكفي؟",
        "وهذه متابعة",
    ]


def test_planner_chunks_multilingual_phrase_boundaries():
    planner = RealtimeSpeechPlanner()
    text = (
        "これは多言語音声の計画で英語の空白に頼らず自然な句読点を探すための長い前置きです"
        "これは多言語音声の計画で英語の空白に頼らず自然な句読点を探すための長い前置きです、"
        "ここから先もまだかなり長く続くので句読点で先に読み上げられる必要があります"
        "ここから先もまだかなり長く続くので句読点で先に読み上げられる必要があります"
    )

    chunks = list(planner.chunk(text))

    assert chunks[0].endswith("です、")
    assert "".join(chunks) == text


def test_text_engine_takes_speakable_phrase_before_full_sentence():
    phrase = (
        "This response starts with a stable opening phrase, and then keeps going "
        "without ending the full sentence yet"
    )

    chunk, remaining = _take_speakable_chunk(phrase)

    assert chunk == "This response starts with a stable opening phrase,"
    assert remaining == "and then keeps going without ending the full sentence yet"


def test_text_engine_takes_unicode_phrase_boundary_without_spaces():
    phrase = "これはかなり長い応答の最初の安定した部分です、まだ文は終わっていませんが会話ではここで話し始められます"

    chunk, remaining = _take_speakable_chunk(phrase)

    assert chunk == "これはかなり長い応答の最初の安定した部分です、"
    assert remaining == "まだ文は終わっていませんが会話ではここで話し始められます"


def test_text_engine_takes_unicode_sentence_boundary_without_trailing_space():
    chunk, remaining = _take_speakable_chunk("これは短い文です。次の文も続きます")

    assert chunk == "これは短い文です。"
    assert remaining == "次の文も続きます"


def test_text_engine_keeps_short_unfinished_phrase_buffered():
    chunk, remaining = _take_speakable_chunk("This is still forming")

    assert chunk is None
    assert remaining == "This is still forming"


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
                payload={"transcript": "こんにちは hermes", "language": "ja", "locale": "ja-JP", "script": "Jpan"},
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
        final = next(event for event in seen if event.type == VoiceEventType.TRANSCRIPT_FINAL)
        assert final.payload["text"] == "こんにちは hermes"
        assert final.payload["language"] == "ja"
        assert final.payload["locale"] == "ja-JP"
        assert final.payload["script"] == "Jpan"
        assistant_partial = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL)
        assert assistant_partial.payload["language"] == "ja"
        assert assistant_partial.payload["locale"] == "ja-JP"
        assert assistant_partial.payload["script"] == "Jpan"
        assert seen[-1].payload["text"] == "Answering: こんにちは hermes."
        assert seen[-1].payload["language"] == "ja"
        assert seen[-1].payload["locale"] == "ja-JP"
        assert seen[-1].payload["script"] == "Jpan"

    asyncio.run(run())


def test_realtime_oracle_prompt_preserves_sanitized_speech_language_metadata():
    prompt = _voice_oracle_prompt(
        "こんにちは",
        {
            "language": "ja",
            "locale": "ja-JP",
            "script": "Jpan",
            "raw_language_url": "https://voice.local/secret",
            "language_url": "https://voice.local/secret",
        },
    )

    assert "Preserve the user's spoken language and script" in prompt
    assert "language=ja" in prompt
    assert "locale=ja-JP" in prompt
    assert "script=Jpan" in prompt
    assert "voice.local" not in prompt
    assert "language_url" not in prompt


def test_text_engine_speaks_stable_phrase_before_sentence_ends(monkeypatch):
    class PhraseOracle:
        async def stream_answer(self, transcript: str):
            yield "This response starts with a stable opening phrase, "
            yield "and then keeps going without ending the full sentence yet"

    async def run():
        spoken = []
        spoke = asyncio.Event()

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)
            spoke.set()

        monkeypatch.setattr(TextOracleTTSEngine, "_speak_chunk", fake_speak)

        engine = TextOracleTTSEngine(oracle=PhraseOracle())
        await engine.start(RealtimeVoiceSessionConfig(session_id="voice-123"))
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "tell me something"},
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL:
                break

        await asyncio.wait_for(spoke.wait(), timeout=1)
        assert spoken[0] == "This response starts with a stable opening phrase,"
        assert seen[-1].payload["text"] == "This response starts with a stable opening phrase,"
        await engine.close()

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
                payload={
                    "transcript": "こんにちは",
                    "end_of_utterance": False,
                    "language": "ja",
                    "locale": "ja-JP",
                    "script": "Jpan",
                    "raw_language_url": "https://voice.local/secret",
                    "language_url": "https://voice.local/secret",
                },
            )
        )

        events = [await anext(engine.events()), await anext(engine.events())]

        assert [event.type for event in events] == [
            VoiceEventType.SESSION_STARTED,
            VoiceEventType.TRANSCRIPT_PARTIAL,
        ]
        assert events[-1].payload["text"] == "こんにちは"
        assert events[-1].payload["language"] == "ja"
        assert events[-1].payload["locale"] == "ja-JP"
        assert events[-1].payload["script"] == "Jpan"
        assert "language_url" not in events[-1].payload
        assert oracle.called is False
        await engine.close()

    asyncio.run(run())


def test_text_engine_session_started_includes_non_secret_runtime_contract():
    async def run():
        engine = TextOracleTTSEngine(oracle=FakeOracle())
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                sidecar_token="secret-token",
                metadata={
                    "language_support": {
                        "production_languages": ["en", "ja"],
                        "production_scripts": ["Latn", "Jpan"],
                        "best_effort_languages": True,
                    },
                    "quality_targets_ms": {
                        "audio_to_partial_transcript_ms": 250,
                        "final_transcript_to_first_text_ms": 450,
                        "final_transcript_to_first_audio_ms": 850,
                        "barge_in_ack_ms": 120,
                    },
                    "conversation_quality": {
                        "mode": "streaming_text",
                        "reason": "streaming_stt_tts",
                        "live_like": True,
                    },
                    "sidecar_token": "do-not-forward",
                },
            )
        )

        event = await anext(engine.events())

        assert event.type == VoiceEventType.SESSION_STARTED
        assert event.payload["language_support"]["production_languages"] == ["en", "ja"]
        assert event.payload["quality_targets_ms"]["final_transcript_to_first_audio_ms"] == 850
        assert event.payload["conversation_quality"]["mode"] == "streaming_text"
        assert "metadata" not in event.payload
        serialized = json.dumps(event.to_wire())
        assert "secret-token" not in serialized
        assert "do-not-forward" not in serialized
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


def test_text_engine_does_not_block_oracle_stream_on_local_tts(monkeypatch):
    class StreamingOracle:
        async def stream_answer(self, transcript: str):
            yield "First sentence. "
            yield "Second sentence."

    async def run():
        spoken = []
        first_tts_started = asyncio.Event()
        release_first_tts = asyncio.Event()

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)
            if text == "First sentence.":
                first_tts_started.set()
                await release_first_tts.wait()

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

        partials = []
        async for event in engine.events():
            if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL:
                partials.append(event.payload["text"])
                if len(partials) == 1:
                    await asyncio.wait_for(first_tts_started.wait(), timeout=1)
                if len(partials) == 2:
                    break

        assert partials == ["First sentence.", "Second sentence."]
        assert spoken == ["First sentence."]

        release_first_tts.set()
        async for event in engine.events():
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                await engine.close()
                break

        assert spoken == ["First sentence.", "Second sentence."]

    asyncio.run(run())


def test_text_engine_cancels_queued_tts_when_oracle_fails(monkeypatch):
    class FailingOracle:
        def __init__(self):
            self.release_failure = asyncio.Event()

        async def stream_answer(self, transcript: str):
            yield "First sentence. "
            await self.release_failure.wait()
            raise RuntimeError("oracle failed")

    async def run():
        spoken = []
        first_tts_started = asyncio.Event()
        first_tts_cancelled = asyncio.Event()

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)
            first_tts_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                first_tts_cancelled.set()
                raise

        monkeypatch.setattr(TextOracleTTSEngine, "_speak_chunk", fake_speak)

        oracle = FailingOracle()
        engine = TextOracleTTSEngine(oracle=oracle)
        await engine.start(RealtimeVoiceSessionConfig(session_id="voice-123"))
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "hello"},
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL:
                await asyncio.wait_for(first_tts_started.wait(), timeout=1)
                oracle.release_failure.set()
            if event.type == VoiceEventType.SESSION_ERROR:
                break

        await asyncio.wait_for(first_tts_cancelled.wait(), timeout=1)
        await engine.close()

        assert [event.type for event in seen] == [
            VoiceEventType.SESSION_STARTED,
            VoiceEventType.TRANSCRIPT_FINAL,
            VoiceEventType.ASSISTANT_TEXT_PARTIAL,
            VoiceEventType.SESSION_ERROR,
        ]
        assert seen[-1].payload["error"] == "oracle/tts failed: oracle failed"
        assert spoken == ["First sentence."]

    asyncio.run(run())


def test_text_engine_degrades_to_text_when_tts_fails(monkeypatch):
    class StreamingOracle:
        async def stream_answer(self, transcript: str):
            yield "First sentence. "
            yield "Second sentence."

    async def run():
        attempted_tts = []

        async def failing_speak(self, text, playback_generation):
            attempted_tts.append(text)
            raise RuntimeError("tts failed at http://user:pass@voice.local/v1?token=abc")

        monkeypatch.setattr(TextOracleTTSEngine, "_speak_chunk", failing_speak)

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

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                await engine.close()
                break

        degraded = [event for event in seen if event.type == VoiceEventType.FRONTEND_STATE]
        assert len(degraded) == 1
        assert degraded[0].payload["status"] == "degraded"
        assert degraded[0].payload["reason"] == "tts_failed"
        assert "user:pass" not in degraded[0].payload["error"]
        assert "token=abc" not in degraded[0].payload["error"]
        commit = seen[-1]
        assert commit.type == VoiceEventType.ASSISTANT_COMMIT
        assert commit.payload["text"] == "First sentence. Second sentence."
        assert attempted_tts == ["First sentence."]
        assert VoiceEventType.SESSION_ERROR not in [event.type for event in seen]

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
        assert sidecar.received[0].payload["input_generation"] == 1
        assert sidecar.spoken
        assert sidecar.spoken[0].payload["playback_generation"] == 1
        assert sidecar.spoken[0].payload["language"] == "ja"
        assert sidecar.spoken[0].payload["script"] == "Jpan"
        assert VoiceEventType.TRANSCRIPT_PARTIAL in [event.type for event in seen]
        assert VoiceEventType.TRANSCRIPT_FINAL in [event.type for event in seen]
        partial_events = [event for event in seen if event.type == VoiceEventType.TRANSCRIPT_PARTIAL]
        assert partial_events[0].payload["stability"] == 0.7
        assert "language_url" not in partial_events[0].payload
        commit_events = [event for event in seen if event.type == VoiceEventType.ASSISTANT_COMMIT]
        assert commit_events[0].payload["text"] == "Answering: hello hermes."
        assert commit_events[0].payload["playback_generation"] == 1
        final_events = [event for event in seen if event.type == VoiceEventType.TRANSCRIPT_FINAL]
        assert final_events[0].payload["input_generation"] == 1
        assert final_events[0].payload["language"] == "ja"
        assert final_events[0].payload["script"] == "Jpan"
        assert "language_url" not in final_events[0].payload
        audio_events = [event for event in seen if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK]
        assert audio_events[0].payload["playback_generation"] == 1

    asyncio.run(run())


def test_text_engine_drops_stale_sidecar_transcript_after_new_input(monkeypatch):
    class ManualSidecar(FakeSidecar):
        async def send_event(self, event):
            self.received.append(event)

    async def run():
        async def fake_speak(self, text, playback_generation):
            return None

        monkeypatch.setattr(TextOracleTTSEngine, "_speak_chunk", fake_speak)

        sidecar = ManualSidecar()
        engine = TextOracleTTSEngine(oracle=FakeOracle(), sidecar=sidecar)
        await engine.start(RealtimeVoiceSessionConfig(session_id="voice-123", sidecar_base_url="http://voice.local"))

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"old-audio").to_payload(),
                    "end_of_utterance": True,
                },
            )
        )
        old_generation = sidecar.received[-1].payload["input_generation"]

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.BARGE_IN,
                session_id="voice-123",
                sequence=2,
                payload={"reason": "user_speech"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=3,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"new-audio").to_payload(),
                    "end_of_utterance": True,
                },
            )
        )
        new_generation = sidecar.received[-1].payload["input_generation"]

        await sidecar._events.put(
            VoiceEvent(
                type=VoiceEventType.TRANSCRIPT_FINAL,
                session_id="voice-123",
                sequence=1,
                payload={"text": "stale transcript", "input_generation": old_generation},
            )
        )
        await sidecar._events.put(
            VoiceEvent(
                type=VoiceEventType.TRANSCRIPT_FINAL,
                session_id="voice-123",
                sequence=2,
                payload={"text": "fresh transcript", "input_generation": new_generation},
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.TRANSCRIPT_FINAL:
                await engine.close()
                break

        final = next(event for event in seen if event.type == VoiceEventType.TRANSCRIPT_FINAL)
        assert old_generation < new_generation
        assert final.payload["text"] == "fresh transcript"
        assert final.payload["input_generation"] == new_generation

    asyncio.run(run())


def test_text_engine_reports_sidecar_start_failure_as_frontend_fallback():
    closed = {"value": False}

    class FailingStartSidecar:
        async def start(self, config):
            raise RuntimeError("sidecar down at http://user:pass@voice.local:8765/v1?token=abc")

        async def close(self):
            closed["value"] = True

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
        assert closed["value"] is True

    asyncio.run(run())


def test_text_engine_treats_sidecar_session_error_as_frontend_fallback():
    class ErrorSidecar(FakeSidecar):
        async def start(self, config):
            await super().start(config)
            await self._events.put(
                VoiceEvent(
                    type=VoiceEventType.SESSION_ERROR,
                    session_id=config.session_id,
                    sequence=1,
                    payload={"error": "TTS failed Bearer secret-token at http://user:pass@voice.local/v1?token=abc"},
                )
            )

    async def run():
        sidecar = ErrorSidecar()
        engine = TextOracleTTSEngine(oracle=FakeOracle(), sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                frontend_provider="sidecar",
                sidecar_base_url="http://voice.local:8080",
            )
        )

        events = [await anext(engine.events()), await anext(engine.events())]

        assert [event.type for event in events] == [
            VoiceEventType.SESSION_STARTED,
            VoiceEventType.FRONTEND_STATE,
        ]
        fallback = events[-1]
        assert fallback.payload["status"] == "fallback"
        assert fallback.payload["reason"] == "sidecar_session_error"
        assert fallback.payload["sidecar"] is False
        assert "TTS failed" in fallback.payload["error"]
        assert "secret-token" not in fallback.payload["error"]
        assert "user:pass" not in fallback.payload["error"]
        assert "token=abc" not in fallback.payload["error"]
        assert engine._sidecar is None
        assert sidecar.closed is True
        await engine.close()

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

        sidecar = FailingSendSidecar()
        engine = TextOracleTTSEngine(oracle=FakeOracle(), sidecar=sidecar)
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
        assert sidecar.closed is True
        assert VoiceEventType.SESSION_ERROR not in [event.type for event in seen]

    asyncio.run(run())


def test_text_engine_closes_sidecar_when_sidecar_tts_fails(monkeypatch, tmp_path):
    class FailingSpeakSidecar(FakeSidecar):
        async def speak(self, event):
            self.spoken.append(event)
            raise RuntimeError("tts failed at http://user:pass@voice.local/v1?token=abc")

    async def run():
        audio_file = tmp_path / "fallback.ogg"
        audio_file.write_bytes(b"fallback-audio")

        monkeypatch.setattr(TextOracleTTSEngine, "_tts_sync", lambda self, text: str(audio_file))

        sidecar = FailingSpeakSidecar()
        engine = TextOracleTTSEngine(oracle=FakeOracle(), sidecar=sidecar)
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
                payload={"transcript": "hello"},
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                await engine.close()
                break

        degraded = next(event for event in seen if event.type == VoiceEventType.FRONTEND_STATE)
        audio = next(event for event in seen if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK)

        assert degraded.payload["status"] == "degraded"
        assert degraded.payload["reason"] == "sidecar_tts_failed"
        assert degraded.payload["sidecar"] is False
        assert "user:pass" not in degraded.payload["error"]
        assert "token=abc" not in degraded.payload["error"]
        assert AudioChunk.from_payload(audio.payload).data == b"fallback-audio"
        assert sidecar.closed is True
        assert engine._sidecar is None
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
        assert sidecar.received[0].payload["playback_generation"] == 1
        await engine.close()

    asyncio.run(run())


def test_text_engine_barge_in_ack_is_not_blocked_by_slow_sidecar():
    class SlowSidecar(FakeSidecar):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def send_event(self, event):
            self.received.append(event)
            self.entered.set()
            await self.release.wait()

    async def run():
        sidecar = SlowSidecar()
        engine = TextOracleTTSEngine(oracle=FakeOracle(), sidecar=sidecar)
        await engine.start(RealtimeVoiceSessionConfig(session_id="voice-123", sidecar_base_url="http://voice.local"))
        assert (await anext(engine.events())).type == VoiceEventType.SESSION_STARTED

        receive_task = asyncio.create_task(
            engine.receive_event(
                VoiceEvent(
                    type=VoiceEventType.BARGE_IN,
                    session_id="voice-123",
                    sequence=1,
                    payload={"reason": "test"},
                )
            )
        )

        event = await asyncio.wait_for(anext(engine.events()), timeout=0.1)
        assert event.type == VoiceEventType.BARGE_IN
        assert event.payload["playback_generation"] == 1
        assert receive_task.done() is False
        sidecar.release.set()
        await asyncio.wait_for(receive_task, timeout=1)
        assert sidecar.received[0].type == VoiceEventType.BARGE_IN
        await engine.close()

    asyncio.run(run())


def test_text_engine_auto_barge_in_on_new_speech_while_answering(monkeypatch):
    class SlowInterruptibleOracle:
        def __init__(self):
            self.interrupted = False
            self.release = asyncio.Event()

        async def stream_answer(self, transcript: str):
            yield "First answer starts."
            await self.release.wait()
            yield " stale ending."

        def interrupt(self, message: str = ""):
            self.interrupted = True

    async def run():
        async def fake_speak(self, text, playback_generation):
            return None

        monkeypatch.setattr(TextOracleTTSEngine, "_speak_chunk", fake_speak)

        oracle = SlowInterruptibleOracle()
        engine = TextOracleTTSEngine(oracle=oracle)
        await engine.start(RealtimeVoiceSessionConfig(session_id="voice-123"))
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "first turn"},
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
                payload=AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"new-speech").to_payload(),
            )
        )

        barge_in = await anext(engine.events())
        await engine.close()

        assert barge_in.type == VoiceEventType.BARGE_IN
        assert barge_in.payload["reason"] == "user_speech"
        assert barge_in.payload["playback_generation"] == 2
        assert oracle.interrupted is True
        assert engine._inbound_audio == [b"new-speech"]
        assert not any(
            event.payload.get("interrupted") is True and event.payload.get("playback_generation") == 1
            for event in seen
        )

    asyncio.run(run())


def test_text_engine_close_suppresses_cancelled_turn_commit(monkeypatch):
    class SlowOracle:
        async def stream_answer(self, transcript: str):
            yield "Answer starts. "
            await asyncio.Event().wait()
            yield "stale"

    async def run():
        async def fake_speak(self, text, playback_generation):
            return None

        monkeypatch.setattr(TextOracleTTSEngine, "_speak_chunk", fake_speak)

        engine = TextOracleTTSEngine(oracle=SlowOracle())
        await engine.start(RealtimeVoiceSessionConfig(session_id="voice-123"))
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "hello"},
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL:
                break

        await engine.close()

        async for event in engine.events():
            seen.append(event)

        assert seen[-1].type == VoiceEventType.SESSION_CLOSED
        assert not any(
            event.type == VoiceEventType.ASSISTANT_COMMIT and event.payload.get("interrupted") is True
            for event in seen
        )

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
        assert len(commit_events) == 1
        assert commit_events[0].payload == {"text": "Second answer.", "playback_generation": 2}
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
        captured = {"timeouts": []}
        fake_ws = FakeWs()

        async def fake_connect(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return fake_ws

        async def fake_wait_for(awaitable, timeout):
            captured["timeouts"].append(timeout)
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
        assert captured["timeouts"][0] == 2.5
        assert captured["timeouts"][1] == 2.0
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


def test_sidecar_client_times_out_stalled_sends(monkeypatch):
    class FakeWs:
        async def send(self, payload):
            await asyncio.sleep(30)

    async def run():
        async def fake_wait_for(awaitable, timeout):
            awaitable.close()
            raise asyncio.TimeoutError

        monkeypatch.setattr("agent.realtime_voice_sidecar.asyncio.wait_for", fake_wait_for)

        client = RealtimeVoiceSidecarClient()
        client.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            frontend_provider="gemma4",
            sidecar_base_url="http://voice.local:8765",
        )
        client._ws = FakeWs()

        with pytest.raises(RuntimeError, match="sidecar send timed out"):
            await client.send_event(
                VoiceEvent(
                    type=VoiceEventType.AUDIO_INPUT_CHUNK,
                    session_id="voice-123",
                    sequence=2,
                    payload=AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"mic-audio").to_payload(),
                )
            )

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
                payload={
                    "transcript": "hello hermes",
                    "input_generation": 5,
                    "language": "ja",
                    "locale": "ja-JP",
                    "script": "Jpan",
                },
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload={
                    "transcript": "hello hermes",
                    "end_of_utterance": True,
                    "input_generation": 5,
                    "language": "ja",
                    "locale": "ja-JP",
                    "script": "Jpan",
                },
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
        assert [event.payload.get("input_generation") for event in seen[1:]] == [5, 5]
        assert [event.payload.get("language") for event in seen[1:]] == ["ja", "ja"]
        assert [event.payload.get("locale") for event in seen[1:]] == ["ja-JP", "ja-JP"]
        assert [event.payload.get("script") for event in seen[1:]] == ["Jpan", "Jpan"]

    asyncio.run(run())


def test_reference_sidecar_echoes_barge_in_generation():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        await sidecar.start(RealtimeVoiceSessionConfig(session_id="voice-123", frontend_provider="local"))
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.BARGE_IN,
                session_id="voice-123",
                sequence=1,
                payload={"reason": "user_speech", "playback_generation": 4},
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if event.type == VoiceEventType.BARGE_IN:
                await sidecar.close()
                break

        barge_in = seen[-1]
        assert barge_in.payload["reason"] == "user_speech"
        assert barge_in.payload["playback_generation"] == 4

    asyncio.run(run())


def test_reference_sidecar_barge_in_ack_is_not_blocked_by_slow_streaming_bridge(
    monkeypatch,
):
    created = []

    class SlowStreamingClient:
        def __init__(self, *, path="/v1/realtime-text/session"):
            self.path = path
            self.sent = []
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self._events = asyncio.Queue()
            created.append(self)

        async def start(self, config):
            self.config = config

        async def send_event(self, event):
            self.sent.append(event)
            self.entered.set()
            await self.release.wait()

        async def events(self):
            while True:
                event = await self._events.get()
                if event is None:
                    return
                yield event

        async def close(self):
            await self._events.put(None)

    monkeypatch.setattr(
        "agent.realtime_voice_reference_sidecar.RealtimeVoiceSidecarClient",
        SlowStreamingClient,
    )

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(
                streaming_stt_base_url="http://streaming-stt.local:9000",
            )
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(session_id="voice-123", frontend_provider="sidecar")
        )
        assert (await anext(sidecar.events())).type == VoiceEventType.FRONTEND_STATE

        receive_task = asyncio.create_task(
            sidecar.receive_event(
                VoiceEvent(
                    type=VoiceEventType.BARGE_IN,
                    session_id="voice-123",
                    sequence=1,
                    payload={"reason": "user_speech", "playback_generation": 4},
                )
            )
        )

        event = await asyncio.wait_for(anext(sidecar.events()), timeout=0.1)
        assert event.type == VoiceEventType.BARGE_IN
        assert event.payload["playback_generation"] == 4
        assert receive_task.done() is False
        created[0].release.set()
        await asyncio.wait_for(receive_task, timeout=1)
        assert created[0].sent[0].type == VoiceEventType.BARGE_IN
        await sidecar.close()

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
                    "input_generation": 9,
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
        final = next(event for event in seen if event.type == VoiceEventType.TRANSCRIPT_FINAL)
        assert final.payload["input_generation"] == 9
        audio_events = [event for event in seen if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK]
        assert audio_events[0].payload["data_b64"]
        assert audio_events[0].payload["playback_generation"] == 7

    asyncio.run(run())


def test_reference_sidecar_passes_language_metadata_to_tts_callback(tmp_path):
    captured = {}

    def fake_synthesize(text, *, metadata=None):
        captured["text"] = text
        captured["metadata"] = metadata
        audio = tmp_path / "speech.ogg"
        audio.write_bytes(b"audio")
        return {"success": True, "file_path": str(audio)}

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(vllm_base_url=None, vllm_model=None),
            synthesize_func=fake_synthesize,
        )
        await sidecar.start(RealtimeVoiceSessionConfig(session_id="voice-123", frontend_provider="local"))
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                session_id="voice-123",
                sequence=1,
                payload={
                    "text": "こんにちは、Hermesです。",
                    "speak": True,
                    "playback_generation": 7,
                    "language": "ja",
                    "locale": "ja-JP",
                    "script": "Jpan",
                    "language_url": "https://voice.local/secret",
                },
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK:
                await sidecar.close()
                break

        audio = seen[-1]
        assert captured == {
            "text": "こんにちは、Hermesです。",
            "metadata": {
                "language": "ja",
                "locale": "ja-JP",
                "script": "Jpan",
            },
        }
        assert audio.payload["language"] == "ja"
        assert audio.payload["locale"] == "ja-JP"
        assert audio.payload["script"] == "Jpan"
        assert "language_url" not in audio.payload

    asyncio.run(run())


def test_reference_sidecar_bounds_utterance_audio_buffer():
    transcribe_called = False

    def fake_transcribe(path):
        nonlocal transcribe_called
        transcribe_called = True
        return {"success": True, "transcript": "should not run"}

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(vllm_base_url=None, vllm_model=None),
            transcribe_audio_func=fake_transcribe,
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                frontend_provider="local",
                input_buffer_limit_bytes=3,
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"ab").to_payload(),
                    "input_generation": 9,
                },
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"cd").to_payload(),
                    "end_of_utterance": True,
                    "input_generation": 9,
                },
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if event.type == VoiceEventType.FRONTEND_STATE and event.payload.get("reason") == "input_buffer_limit_exceeded":
                await sidecar.close()
                break

        degraded = seen[-1]
        assert degraded.payload["status"] == "degraded"
        assert degraded.payload["reason"] == "input_buffer_limit_exceeded"
        assert degraded.payload["sidecar"] is True
        assert degraded.payload["limit_bytes"] == 3
        assert sidecar._audio == []
        assert sidecar._audio_bytes == 0
        assert sidecar._audio_input_generation is None

    asyncio.run(run())
    assert transcribe_called is False


def test_reference_sidecar_suppresses_worker_events_after_close():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        await sidecar.start(RealtimeVoiceSessionConfig(session_id="voice-123", frontend_provider="local"))

        async def worker():
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                await sidecar._emit(VoiceEventType.TRANSCRIPT_FINAL, {"text": "late transcript"})
                raise

        task = asyncio.create_task(worker())
        sidecar._track_task(task)
        await asyncio.sleep(0)

        await sidecar.close()

        seen = []
        async for event in sidecar.events():
            seen.append(event)

        assert task.done()
        assert [event.type for event in seen] == [
            VoiceEventType.FRONTEND_STATE,
            VoiceEventType.SESSION_CLOSED,
        ]
        assert all(event.payload.get("text") != "late transcript" for event in seen)

    asyncio.run(run())


def test_reference_sidecar_close_does_not_wait_forever_for_stubborn_workers(monkeypatch):
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        await sidecar.start(RealtimeVoiceSessionConfig(session_id="voice-123", frontend_provider="local"))

        async def stubborn_worker():
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                await asyncio.sleep(30)

        task = asyncio.create_task(stubborn_worker())
        sidecar._track_task(task)
        await asyncio.sleep(0)

        monkeypatch.setattr(
            "agent.realtime_voice_reference_sidecar.REFERENCE_SIDECAR_CLOSE_DRAIN_TIMEOUT_SECONDS",
            0.001,
        )

        await asyncio.wait_for(sidecar.close(), timeout=1)

        seen = []
        async for event in sidecar.events():
            seen.append(event)

        assert [event.type for event in seen] == [
            VoiceEventType.FRONTEND_STATE,
            VoiceEventType.SESSION_CLOSED,
        ]
        assert task not in sidecar._active_tasks
        assert not task.done()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())


def test_reference_sidecar_sanitizes_provider_errors():
    def failing_transcribe(path):
        raise RuntimeError("STT failed at http://user:pass@voice.local/v1?token=abc")

    def failing_synthesize(text):
        raise RuntimeError("TTS failed Bearer secret-token api_key=raw")

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(vllm_base_url=None, vllm_model=None),
            transcribe_audio_func=failing_transcribe,
            synthesize_func=failing_synthesize,
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

        errors = []
        async for event in sidecar.events():
            if event.type == VoiceEventType.SESSION_ERROR:
                errors.append(str(event.payload.get("error") or ""))
            if len(errors) == 2:
                await sidecar.close()
                break

        combined = "\n".join(errors)
        assert "http://***@voice.local/v1" in combined
        assert "Bearer ***" in combined
        assert "api_key=***" in combined
        assert "user:pass" not in combined
        assert "token=abc" not in combined
        assert "secret-token" not in combined
        assert "api_key=raw" not in combined

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
    prompt = captured["body"]["messages"][0]["content"][1]["text"]
    assert "Preserve the speaker's language and script" in prompt
    assert "do not translate" in prompt


def test_reference_sidecar_health_marks_streaming_stt_only_after_bridge_health():
    runtime = ReferenceSidecarRuntimeConfig(
        streaming_stt_base_url="http://streaming-stt.local:9000",
        streaming_stt_model="portable-streaming-asr",
        local_stt_enabled=True,
        local_tts_enabled=True,
    )

    unverified = reference_sidecar_health_payload(runtime)

    assert unverified["frontend"]["provider"] == "local"
    assert unverified["frontend"]["streaming_stt_bridge"] == {
        "configured": True,
        "healthy": False,
    }
    assert unverified["capabilities"]["streaming_stt"] is False
    assert unverified["capabilities"]["streaming_stt_bridge"] is True
    assert unverified["capabilities"]["utterance_stt"] is True

    verified = reference_sidecar_health_payload(
        runtime,
        streaming_stt_health={
            "ok": True,
            "capabilities": {
                "streaming_stt": True,
            },
        },
    )

    assert verified["frontend"]["provider"] == "streaming_stt"
    assert verified["frontend"]["model"] == "portable-streaming-asr"
    assert verified["frontend"]["streaming_stt_bridge"] == {
        "configured": True,
        "healthy": True,
    }
    assert verified["capabilities"]["streaming_stt"] is True
    assert verified["capabilities"]["utterance_stt"] is True


def test_reference_sidecar_health_marks_streaming_tts_bridge_after_health():
    runtime = ReferenceSidecarRuntimeConfig(
        streaming_tts_base_url="http://streaming-tts.local:9001",
        streaming_tts_model="portable-streaming-voice",
        local_tts_enabled=False,
    )

    unverified = reference_sidecar_health_payload(runtime)

    assert unverified["capabilities"]["tts"] is False
    assert unverified["capabilities"]["streaming_tts_bridge"] is True
    assert unverified["frontend"]["streaming_tts_bridge"] == {
        "configured": True,
        "healthy": False,
        "model": "portable-streaming-voice",
    }

    verified = reference_sidecar_health_payload(
        runtime,
        streaming_tts_health={
            "ok": True,
            "capabilities": {
                "tts": True,
                "streaming_tts": True,
            },
        },
    )

    assert verified["capabilities"]["tts"] is True
    assert verified["frontend"]["streaming_tts_bridge"] == {
        "configured": True,
        "healthy": True,
        "model": "portable-streaming-voice",
    }


def test_reference_sidecar_health_payload_is_sanitized():
    payload = reference_sidecar_health_payload(
        ReferenceSidecarRuntimeConfig(
            vllm_base_url="http://user:secret@voice.local:8000/v1",
            vllm_model="google/gemma-4-E4B-it-qat-w4a16-ct",
            local_stt_enabled=False,
            local_tts_enabled=True,
            input_languages=("ja", "en-US", "JA", "https://voice.local/secret"),
            output_languages=("ja", "ko", "token=secret"),
            scripts=("Jpan", "Latn", "bad/script"),
        )
    )

    assert payload == {
        "ok": True,
        "kind": "reference",
        "frontend": {
            "provider": "vllm",
            "model": "google/gemma-4-E4B-it-qat-w4a16-ct",
            "languages": ["ja", "en-US", "ko"],
            "scripts": ["Jpan", "Latn"],
        },
        "capabilities": {
            "utterance_stt": True,
            "streaming_stt": False,
            "tts": True,
            "native_s2s": False,
            "vllm_audio_frontend": True,
            "input_languages": ["ja", "en-US"],
            "output_languages": ["ja", "ko"],
            "scripts": ["Jpan", "Latn"],
        },
        "local": {
            "stt": False,
            "tts": True,
        },
    }
    assert "secret" not in __import__("json").dumps(payload)


def test_reference_sidecar_runtime_reads_language_metadata_from_env(monkeypatch):
    monkeypatch.setenv("HERMES_VOICE_LANGUAGES", "en,ja https://voice.local/secret")
    monkeypatch.setenv("HERMES_VOICE_INPUT_LANGUAGES", "ja en-US JA")
    monkeypatch.setenv("HERMES_VOICE_OUTPUT_LANGUAGES", "ja,ko token=secret")
    monkeypatch.setenv("HERMES_VOICE_SCRIPTS", "Jpan Latn bad/script")
    monkeypatch.setenv("HERMES_VOICE_STREAMING_STT_BASE_URL", "http://streaming-stt.local:9000")
    monkeypatch.setenv("HERMES_VOICE_STREAMING_STT_MODEL", "portable-streaming-asr")
    monkeypatch.setenv("HERMES_VOICE_STREAMING_STT_TOKEN", "secret-token")
    monkeypatch.setenv("HERMES_VOICE_STREAMING_STT_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setenv("HERMES_VOICE_STREAMING_BRIDGE_HEALTH_TIMEOUT_SECONDS", "0.25")
    monkeypatch.setenv("HERMES_VOICE_STREAMING_TTS_BASE_URL", "http://streaming-tts.local:9001")
    monkeypatch.setenv("HERMES_VOICE_STREAMING_TTS_MODEL", "portable-streaming-voice")
    monkeypatch.setenv("HERMES_VOICE_STREAMING_TTS_TOKEN", "tts-secret-token")
    monkeypatch.setenv("HERMES_VOICE_STREAMING_TTS_TIMEOUT_SECONDS", "3.5")

    runtime = runtime_config_from_env()

    assert runtime.input_languages == ("ja", "en-US")
    assert runtime.output_languages == ("ja", "ko")
    assert runtime.scripts == ("Jpan", "Latn")
    assert runtime.streaming_stt_base_url == "http://streaming-stt.local:9000"
    assert runtime.streaming_stt_model == "portable-streaming-asr"
    assert runtime.streaming_stt_token == "secret-token"
    assert runtime.streaming_stt_timeout_seconds == 2.5
    assert runtime.streaming_bridge_health_timeout_seconds == 0.25
    assert runtime.streaming_tts_base_url == "http://streaming-tts.local:9001"
    assert runtime.streaming_tts_model == "portable-streaming-voice"
    assert runtime.streaming_tts_token == "tts-secret-token"
    assert runtime.streaming_tts_timeout_seconds == 3.5


def test_reference_sidecar_health_probe_uses_short_bridge_health_timeout(monkeypatch):
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"ok": true, "capabilities": {"streaming_stt": true}}'

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return FakeResponse()

    monkeypatch.setattr(reference_sidecar_module.urllib.request, "urlopen", fake_urlopen)

    runtime = ReferenceSidecarRuntimeConfig(
        streaming_stt_base_url="http://streaming-stt.local:9000",
        streaming_stt_timeout_seconds=10.0,
        streaming_bridge_health_timeout_seconds=0.2,
    )

    health = reference_sidecar_module._probe_streaming_stt_health_sync(runtime)

    assert health == {"ok": True, "capabilities": {"streaming_stt": True}}
    assert calls == [("http://streaming-stt.local:9000/health", 0.2)]


def test_reference_sidecar_bridges_streaming_stt_events(monkeypatch):
    created = []

    class FakeStreamingSTTClient:
        def __init__(self, *, path="/v1/realtime-text/session"):
            self.path = path
            self.sent = []
            self._events = asyncio.Queue()
            created.append(self)

        async def start(self, config):
            self.config = config

        async def send_event(self, event):
            self.sent.append(event)
            if event.type == VoiceEventType.AUDIO_INPUT_CHUNK:
                await self._events.put(
                    VoiceEvent(
                        type=VoiceEventType.TRANSCRIPT_PARTIAL,
                        session_id=event.session_id,
                        sequence=1,
                        payload={
                            "text": "こん",
                            "stability": 0.4,
                            "input_generation": event.payload.get("input_generation"),
                            "language": "ja",
                            "locale": "ja-JP",
                            "script": "Jpan",
                            "language_url": "https://voice.local/secret",
                        },
                    )
                )
                await self._events.put(
                    VoiceEvent(
                        type=VoiceEventType.TRANSCRIPT_FINAL,
                        session_id=event.session_id,
                        sequence=2,
                        payload={
                            "text": "こんにちは Hermes",
                            "confidence": 0.92,
                            "input_generation": event.payload.get("input_generation"),
                            "language": "ja",
                            "locale": "ja-JP",
                            "script": "Jpan",
                        },
                    )
                )

        async def events(self):
            while True:
                event = await self._events.get()
                if event is None:
                    return
                yield event

        async def close(self):
            await self._events.put(None)

    monkeypatch.setattr(
        "agent.realtime_voice_reference_sidecar.RealtimeVoiceSidecarClient",
        FakeStreamingSTTClient,
    )

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(
                streaming_stt_base_url="http://streaming-stt.local:9000",
                streaming_stt_model="portable-streaming-asr",
                streaming_stt_token="secret-token",
                streaming_stt_timeout_seconds=2.5,
            )
        )
        await sidecar.start(RealtimeVoiceSessionConfig(session_id="voice-123", frontend_provider="sidecar"))
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"audio").to_payload(),
                    "input_generation": 12,
                },
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if event.type == VoiceEventType.TRANSCRIPT_FINAL:
                await sidecar.close()
                break

        assert created[0].path == "/v1/streaming-stt/session"
        assert created[0].config.sidecar_base_url == "http://streaming-stt.local:9000"
        assert created[0].config.sidecar_token == "secret-token"
        assert created[0].config.frontend_model == "portable-streaming-asr"
        assert created[0].config.sidecar_connect_timeout_seconds == 2.5
        assert created[0].sent[0].payload["input_generation"] == 12
        assert [event.type for event in seen] == [
            VoiceEventType.FRONTEND_STATE,
            VoiceEventType.TRANSCRIPT_PARTIAL,
            VoiceEventType.TRANSCRIPT_FINAL,
        ]
        assert seen[0].payload["provider"] == "streaming_stt"
        assert seen[0].payload["streaming_stt"] is True
        assert seen[1].payload == {
            "language": "ja",
            "locale": "ja-JP",
            "script": "Jpan",
            "text": "こん",
            "stability": 0.4,
            "input_generation": 12,
        }
        assert seen[2].payload == {
            "language": "ja",
            "locale": "ja-JP",
            "script": "Jpan",
            "text": "こんにちは Hermes",
            "confidence": 0.92,
            "input_generation": 12,
        }

    asyncio.run(run())


def test_reference_sidecar_bridges_streaming_tts_audio(monkeypatch):
    created = []

    class FakeStreamingTTSClient:
        def __init__(self, *, path="/v1/realtime-text/session"):
            self.path = path
            self.sent = []
            self._events = asyncio.Queue()
            created.append(self)

        async def start(self, config):
            self.config = config

        async def send_event(self, event):
            self.sent.append(event)
            if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL:
                payload = AudioChunk(codec=VoiceAudioCodec.PCM16, data=b"pcm-audio").to_payload()
                payload["playback_generation"] = event.payload.get("playback_generation")
                await self._events.put(
                    VoiceEvent(
                        type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
                        session_id=event.session_id,
                        sequence=1,
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
            await self._events.put(None)

    monkeypatch.setattr(
        "agent.realtime_voice_reference_sidecar.RealtimeVoiceSidecarClient",
        FakeStreamingTTSClient,
    )

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(
                streaming_tts_base_url="http://streaming-tts.local:9001",
                streaming_tts_model="portable-streaming-voice",
                streaming_tts_token="tts-secret-token",
                streaming_tts_timeout_seconds=3.5,
                local_tts_enabled=False,
            )
        )
        await sidecar.start(RealtimeVoiceSessionConfig(session_id="voice-123", frontend_provider="sidecar"))
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                session_id="voice-123",
                sequence=1,
                payload={
                    "text": "hello back",
                    "speak": True,
                    "playback_generation": 7,
                    "language": "en",
                },
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK:
                await sidecar.close()
                break

        assert created[0].path == "/v1/streaming-tts/session"
        assert created[0].config.sidecar_base_url == "http://streaming-tts.local:9001"
        assert created[0].config.sidecar_token == "tts-secret-token"
        assert created[0].config.frontend_model == "portable-streaming-voice"
        assert created[0].config.sidecar_connect_timeout_seconds == 3.5
        assert created[0].sent[0].payload == {
            "text": "hello back",
            "speak": True,
            "playback_generation": 7,
            "language": "en",
        }
        assert [event.type for event in seen] == [
            VoiceEventType.FRONTEND_STATE,
            VoiceEventType.AUDIO_OUTPUT_CHUNK,
        ]
        assert seen[0].payload["streaming_tts"] is True
        assert AudioChunk.from_payload(seen[1].payload).data == b"pcm-audio"
        assert seen[1].payload["playback_generation"] == 7

    asyncio.run(run())


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


def test_session_treats_assistant_partial_text_as_cumulative():
    session = RealtimeVoiceSession(
        RealtimeVoiceSessionConfig(session_id="voice-123"),
        engine=TextOracleTTSEngine(oracle=FakeOracle()),
    )

    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
            session_id="voice-123",
            sequence=1,
            payload={"text": "Hello"},
        )
    )
    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
            session_id="voice-123",
            sequence=2,
            payload={"text": "Hello world"},
        )
    )
    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.ASSISTANT_COMMIT,
            session_id="voice-123",
            sequence=3,
            payload={},
        )
    )

    assert session.transcript.committed_assistant_segments == ["Hello world"]


def test_session_accumulates_assistant_partial_delta():
    session = RealtimeVoiceSession(
        RealtimeVoiceSessionConfig(session_id="voice-123"),
        engine=TextOracleTTSEngine(oracle=FakeOracle()),
    )

    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
            session_id="voice-123",
            sequence=1,
            payload={"delta": "Hello "},
        )
    )
    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
            session_id="voice-123",
            sequence=2,
            payload={"delta": "world"},
        )
    )

    assert session.transcript.assistant_draft == "Hello world"


def test_session_drops_stale_generated_events_after_barge_in():
    class ScriptedEngine:
        def __init__(self):
            self.received = []
            self._events = [
                VoiceEvent(
                    type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
                    session_id="voice-123",
                    sequence=10,
                    payload={
                        **AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"stale-audio").to_payload(),
                        "playback_generation": 1,
                    },
                ),
                VoiceEvent(
                    type=VoiceEventType.ASSISTANT_COMMIT,
                    session_id="voice-123",
                    sequence=11,
                    payload={"text": "stale answer", "playback_generation": 1},
                ),
                VoiceEvent(
                    type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                    session_id="voice-123",
                    sequence=12,
                    payload={"text": "fresh answer", "playback_generation": 2},
                ),
            ]

        async def start(self, config):
            return None

        async def receive_event(self, event):
            self.received.append(event)

        async def events(self):
            for event in self._events:
                yield event

        async def close(self):
            return None

    async def run():
        engine = ScriptedEngine()
        session = RealtimeVoiceSession(RealtimeVoiceSessionConfig(session_id="voice-123"), engine=engine)
        await session.start()
        session._apply_server_event(
            VoiceEvent(
                type=VoiceEventType.TRANSCRIPT_FINAL,
                session_id="voice-123",
                sequence=1,
                payload={"text": "first", "playback_generation": 1},
            )
        )
        await session.receive_client_event(
            VoiceEvent(
                type=VoiceEventType.BARGE_IN,
                session_id="voice-123",
                sequence=1,
                payload={"reason": "user_speech"},
            )
        )

        seen = []
        async for event in session.events():
            seen.append(event)

        assert [event.sequence for event in seen] == [12]
        assert seen[0].payload["text"] == "fresh answer"
        assert session.durable_messages() == [{"role": "user", "content": "first"}]

    asyncio.run(run())


def test_session_marks_audio_only_output_as_speaking():
    class AudioOnlyEngine:
        async def start(self, config):
            return None

        async def receive_event(self, event):
            return None

        async def events(self):
            yield VoiceEvent(
                type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"native-audio").to_payload(),
                    "playback_generation": 1,
                },
            )

        async def close(self):
            return None

    async def run():
        session = RealtimeVoiceSession(RealtimeVoiceSessionConfig(session_id="voice-123"), engine=AudioOnlyEngine())
        await session.start()

        async for event in session.events():
            assert event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK
            assert event.payload["session_state"] == RealtimeVoiceSessionState.SPEAKING.value
            break

        assert session.state == RealtimeVoiceSessionState.SPEAKING
        assert session.transcript.active_playback_generation == 1

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


def test_session_marks_latency_quality_target_misses(monkeypatch):
    class PartialTranscriptEngine:
        async def start(self, config):
            return None

        async def receive_event(self, event):
            return None

        async def events(self):
            yield VoiceEvent(
                type=VoiceEventType.TRANSCRIPT_PARTIAL,
                session_id="voice-123",
                sequence=1,
                payload={"text": "hello"},
            )

        async def close(self):
            return None

    async def run():
        session = RealtimeVoiceSession(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                metadata={
                    "quality_targets_ms": {
                        "audio_to_partial_transcript_ms": 300,
                    },
                },
            ),
            engine=PartialTranscriptEngine(),
        )
        monkeypatch.setattr(
            session,
            "_event_metrics",
            lambda event: {
                "audio_to_partial_transcript_ms": 500,
                "session_elapsed_ms": 500,
            },
        )
        await session.start()

        event = await anext(session.events())

        assert event.payload["metrics"]["audio_to_partial_transcript_ms"] == 500
        assert event.payload["quality_target_misses"] == [
            {
                "metric": "audio_to_partial_transcript_ms",
                "actual_ms": 500,
                "target_ms": 300,
            }
        ]
        assert event.payload["quality_summary"] == {
            "target_miss_count": 1,
            "last_target_miss": {
                "metric": "audio_to_partial_transcript_ms",
                "actual_ms": 500,
                "target_ms": 300,
            },
        }
        await session.close()

    asyncio.run(run())


def test_session_carries_quality_summary_after_target_miss(monkeypatch):
    class TwoEventEngine:
        async def start(self, config):
            return None

        async def receive_event(self, event):
            return None

        async def events(self):
            yield VoiceEvent(
                type=VoiceEventType.TRANSCRIPT_PARTIAL,
                session_id="voice-123",
                sequence=1,
                payload={"text": "hello"},
            )
            yield VoiceEvent(
                type=VoiceEventType.TRANSCRIPT_FINAL,
                session_id="voice-123",
                sequence=2,
                payload={"text": "hello"},
            )

        async def close(self):
            return None

    async def run():
        session = RealtimeVoiceSession(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                metadata={
                    "quality_targets_ms": {
                        "audio_to_partial_transcript_ms": 300,
                    },
                },
            ),
            engine=TwoEventEngine(),
        )
        metric_payloads = iter([
            {"audio_to_partial_transcript_ms": 500},
            {"audio_to_final_transcript_ms": 550},
        ])
        monkeypatch.setattr(session, "_event_metrics", lambda event: next(metric_payloads))
        await session.start()

        first = await anext(session.events())
        second = await anext(session.events())

        assert first.payload["quality_summary"]["target_miss_count"] == 1
        assert "quality_target_misses" not in second.payload
        assert second.payload["quality_summary"] == first.payload["quality_summary"]
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


def test_session_marks_session_error_as_closing_state():
    class ErrorEngine:
        kind = RealtimeVoiceEngineKind.TEXT_ORACLE_TTS

        async def start(self, config):
            self.config = config

        async def receive_event(self, event):
            return None

        async def events(self):
            yield VoiceEvent(
                type=VoiceEventType.SESSION_ERROR,
                session_id="voice-123",
                sequence=1,
                payload={"error": "sidecar failed"},
            )

        async def close(self):
            return None

    async def run():
        session = RealtimeVoiceSession(
            RealtimeVoiceSessionConfig(session_id="voice-123"),
            engine=ErrorEngine(),
        )
        await session.start()

        event = await anext(session.events())

        assert event.type == VoiceEventType.SESSION_ERROR
        assert event.payload["session_state"] == RealtimeVoiceSessionState.CLOSING.value
        assert session.state == RealtimeVoiceSessionState.CLOSING
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


def test_native_s2s_engine_session_started_matches_realtime_contract(monkeypatch):
    async def fake_connect(self, config):
        self._ws = None

    async def run():
        monkeypatch.setattr(NativeS2SSidecarEngine, "_connect_sidecar", fake_connect)

        engine = NativeS2SSidecarEngine()
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE,
                sidecar_base_url="ws://voice.local",
                frontend_provider="native",
                frontend_model="s2s-reference",
                metadata={
                    "language_support": {
                        "production_languages": ["en", "ja"],
                        "production_scripts": ["Latn", "Jpan"],
                        "best_effort_languages": True,
                    },
                    "quality_targets_ms": {
                        "audio_to_partial_transcript_ms": 250,
                        "final_transcript_to_first_text_ms": 450,
                        "final_transcript_to_first_audio_ms": 850,
                        "barge_in_ack_ms": 120,
                    },
                    "conversation_quality": {
                        "mode": "native_s2s",
                        "reason": "native_s2s",
                        "live_like": True,
                    },
                },
            )
        )

        event = await anext(engine.events())

        assert event.type == VoiceEventType.SESSION_STARTED
        assert event.payload["engine"] == RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE.value
        assert event.payload["input_codec"] == VoiceAudioCodec.OPUS.value
        assert event.payload["output_codec"] == VoiceAudioCodec.OPUS.value
        assert event.payload["frontend_provider"] == "native"
        assert event.payload["frontend_model"] == "s2s-reference"
        assert event.payload["sidecar"] is True
        assert event.payload["language_support"]["production_languages"] == ["en", "ja"]
        assert event.payload["quality_targets_ms"]["barge_in_ack_ms"] == 120
        assert event.payload["conversation_quality"]["mode"] == "native_s2s"
        await engine.close()

    asyncio.run(run())


def test_native_s2s_engine_degrades_when_oracle_hint_fails():
    class FakeWs:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(payload)

    class FailingOracle:
        async def stream_answer(self, transcript):
            raise RuntimeError("oracle failed at http://user:pass@voice.local/v1?token=abc")
            yield transcript

    async def run():
        ws = FakeWs()
        engine = NativeS2SSidecarEngine()
        engine.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE,
            sidecar_base_url="ws://voice.local",
        )
        engine._ws = ws
        engine._oracle = FailingOracle()

        await engine._send_oracle_hint("what is kame", 3)

        event = await engine._events.get()
        assert event.type == VoiceEventType.FRONTEND_STATE
        assert event.payload["status"] == "degraded"
        assert event.payload["reason"] == "oracle_hint_failed"
        assert event.payload["sidecar"] is True
        assert "user:pass" not in event.payload["error"]
        assert "token=abc" not in event.payload["error"]
        assert ws.sent == []

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
        captured = {"timeouts": []}
        fake_ws = FakeWs()

        async def fake_connect(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return fake_ws

        async def fake_wait_for(awaitable, timeout):
            captured["timeouts"].append(timeout)
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
        assert captured["timeouts"][0] == 4.0
        assert captured["timeouts"][1] == 2.0
        assert fake_ws.sent
        await engine.close()

    asyncio.run(run())


def test_native_s2s_engine_times_out_stalled_sidecar_sends(monkeypatch):
    class FakeWs:
        async def send(self, payload):
            await asyncio.sleep(30)

    async def run():
        async def fake_wait_for(awaitable, timeout):
            awaitable.close()
            raise asyncio.TimeoutError

        monkeypatch.setattr("agent.realtime_voice_s2s_engine.asyncio.wait_for", fake_wait_for)

        engine = NativeS2SSidecarEngine()
        engine.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE,
            sidecar_base_url="ws://voice.local",
        )
        engine._ws = FakeWs()

        with pytest.raises(RuntimeError, match="native S2S sidecar send timed out"):
            await engine.receive_event(
                VoiceEvent(
                    type=VoiceEventType.AUDIO_INPUT_CHUNK,
                    session_id="voice-123",
                    sequence=12,
                    payload=AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"s2s-mic").to_payload(),
                )
            )

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
            payload={
                "text": "hello",
                "language": "ja",
                "script": "Jpan",
                "language_url": "https://voice.local/secret",
            },
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
    assert transcript.payload["language"] == "ja"
    assert transcript.payload["script"] == "Jpan"
    assert "language_url" not in transcript.payload
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


def test_native_s2s_engine_drops_stale_sidecar_transcript_before_oracle_hint():
    class FakeWs:
        def __init__(self, items):
            self._items = list(items)
            self.sent = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

        async def send(self, payload):
            self.sent.append(payload)

    class FailingOracle:
        async def stream_answer(self, transcript):
            raise AssertionError("stale transcript should not start oracle hint")
            yield transcript

    async def run():
        stale = VoiceEvent(
            type=VoiceEventType.TRANSCRIPT_FINAL,
            session_id="sidecar-session",
            sequence=10,
            payload={"text": "old turn", "playback_generation": 1},
        )
        engine = NativeS2SSidecarEngine()
        engine.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE,
            sidecar_base_url="ws://voice.local",
        )
        engine._ws = FakeWs([__import__("json").dumps(stale.to_wire())])
        engine._oracle = FailingOracle()
        engine._playback_generation = 2

        await engine._read_sidecar()

        assert engine._oracle_hint_task is None
        assert engine._events.empty()
        assert engine._ws.sent == []

    asyncio.run(run())


def test_native_s2s_engine_drops_stale_binary_audio_output_frame():
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
        stale = VoiceEvent(
            type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
            session_id="sidecar-session",
            sequence=44,
            payload={
                **AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"stale-s2s-speaker").to_payload(),
                "playback_generation": 1,
            },
        )
        engine = NativeS2SSidecarEngine()
        engine.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE,
            sidecar_base_url="ws://voice.local",
        )
        engine._ws = FakeWs([binary_audio_frame_from_event(stale)])
        engine._playback_generation = 2

        await engine._read_sidecar()

        assert engine._events.empty()

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


def test_native_s2s_engine_auto_barge_in_on_new_speech_while_output_active():
    class FakeWs:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(payload)

    class InterruptibleOracle:
        def __init__(self):
            self.interrupted = []

        def interrupt(self, message=""):
            self.interrupted.append(message)

    async def run():
        ws = FakeWs()
        oracle = InterruptibleOracle()
        engine = NativeS2SSidecarEngine()
        engine.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE,
            sidecar_base_url="ws://voice.local",
        )
        engine._ws = ws
        engine._oracle = oracle
        engine._playback_generation = 1
        engine._assistant_output_active = True

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=12,
                payload=AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"new-speech-1").to_payload(),
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=13,
                payload=AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"new-speech-2").to_payload(),
            )
        )

        forwarded_barge = VoiceEvent.from_wire(__import__("json").loads(ws.sent[0]))
        assert forwarded_barge.type == VoiceEventType.BARGE_IN
        assert forwarded_barge.payload["reason"] == "user_speech"
        assert forwarded_barge.payload["playback_generation"] == 2
        assert isinstance(ws.sent[1], bytes)
        assert isinstance(ws.sent[2], bytes)
        assert oracle.interrupted
        assert [VoiceEvent.from_wire(__import__("json").loads(item)).type for item in ws.sent if isinstance(item, str)] == [
            VoiceEventType.BARGE_IN
        ]

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


def test_native_s2s_engine_reader_error_after_close_is_terminal():
    class FailingWs:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise RuntimeError("sidecar failed during close at http://user:pass@voice.local/v1?token=abc")

    async def run():
        engine = NativeS2SSidecarEngine()
        engine.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE,
            sidecar_base_url="ws://voice.local",
        )
        engine._ws = FailingWs()
        engine._closed = True

        await engine._read_sidecar()

        assert engine._events.empty()

    asyncio.run(run())
