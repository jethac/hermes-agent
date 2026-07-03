import asyncio
import base64
import io
import json
import time
import types
import urllib.error
import wave

import pytest

import agent.realtime_voice_reference_sidecar as reference_sidecar_module
from agent.realtime_voice import (
    AudioChunk,
    RealtimeVoiceASRMode,
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
from agent.realtime_voice_kame import (
    KameOracleRequest,
    KameReflexDecision,
    KameRoute,
    kame_external_brain_request_to_oracle_request,
    kame_reflex_decision_json_schema,
    kame_reflex_schema_issues,
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
from agent.realtime_voice_gemini import GeminiLiveFrontendConfig, _setup_payload
from agent.realtime_voice_oracle import _voice_oracle_prompt
from agent.realtime_voice_session import RealtimeVoiceSession, RealtimeVoiceSessionState, create_realtime_voice_engine
from agent.realtime_voice_s2s_engine import NativeS2SSidecarEngine
from agent.realtime_voice_sidecar import RealtimeVoiceSidecarClient, sidecar_ws_url, wants_realtime_sidecar
from agent.realtime_voice_text_engine import (
    KameInterfaceOracleEngine,
    TextOracleTTSEngine,
    _kame_oracle_job_control_operation,
    _take_speakable_chunk,
)


def _write_test_wav(path, pcm: bytes = b"\x01\x00\x02\x00\x03\x00\x04\x00", *, sample_rate: int = 16000) -> bytes:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return pcm


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
    provider_secret = "sk" + "_test_" + "abcdefghijklmnopqrstuvwxyz"
    sanitized = sanitize_realtime_voice_error(
        "failed Bearer secret-token "
        "provider_token=raw-provider-token "
        f"access_token={provider_secret} "
        "max_tokens=8192 "
        "at http://user:pass@voice.local:8765/v1?token=abc&api_key=def secret=raw"
    )

    assert sanitized == (
        "failed Bearer *** "
        "provider_token=*** "
        "access_token=*** "
        "max_tokens=8192 "
        "at http://***@voice.local:8765/v1 secret=***"
    )
    assert "secret-token" not in sanitized
    assert "raw-provider-token" not in sanitized
    assert provider_secret not in sanitized
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
        interface_temperature=0.4,
        interface_max_output_tokens=128,
        interface_timeout_seconds=1.25,
        interface_max_audio_seconds=24.0,
        asr_provider="streaming_stt",
        asr_model="nemotron-speech",
        oracle_timeout_seconds=12.5,
        tts_provider="edge",
        tts_model="sonic-3.5",
        tts_voice="voice-123",
        fallback_policy="legacy_voice",
        sidecar_base_url="http://voice.local:8080",
        sidecar_token="secret-token",
        sidecar_connect_timeout_seconds=3.5,
        max_spoken_sentences=3,
        voice_response_policy="brief_summary",
        metadata={"profile": "default"},
    )

    restored = RealtimeVoiceSessionConfig.from_wire(config.to_wire())

    assert restored.to_wire() == config.to_wire()
    assert restored.effective_sidecar_base_url == "http://voice.local:8080"
    assert restored.effective_sidecar_token == "secret-token"
    assert restored.input_buffer_limit_bytes == 4096
    assert restored.interface_temperature == 0.4
    assert restored.interface_max_output_tokens == 128
    assert restored.interface_timeout_seconds == 1.25
    assert restored.interface_max_audio_seconds == 24.0
    assert restored.sidecar_connect_timeout_seconds == 3.5
    assert restored.oracle_timeout_seconds == 12.5
    assert restored.max_spoken_sentences == 3
    assert restored.voice_response_policy == "brief_summary"


def test_session_config_round_trips_kame_fields():
    config = RealtimeVoiceSessionConfig(
        session_id="voice-123",
        engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        frontend_provider="gemma4",
        frontend_model="gemma-4-E2B-it",
        interface_temperature=0.35,
        interface_max_output_tokens=96,
        interface_timeout_seconds=0.6,
        interface_max_audio_seconds=12.0,
        interface_audio_input="native_audio",
        interface_base_url="http://interface.local:8000/v1",
        asr_mode=RealtimeVoiceASRMode.SPECULATIVE,
        asr_provider="streaming_stt",
        asr_model="nemotron-speech",
        asr_base_url="http://asr.local:8767",
        preferred_local_oracle_model="gemma-4-26B-A4B-it",
        max_spoken_sentences=4,
        voice_response_policy="full",
        tts_provider="streaming_tts",
        tts_model="sonic-3.5",
        tts_voice="voice-123",
        tts_base_url="http://tts.local:8768",
        fallback_policy="fail_closed",
        turn_acknowledgement={"enabled": True, "text": "One moment."},
        routing_policy={"local_confidence_threshold": 0.82},
        metrics_policy={"enabled": True, "log_provider_spans": False},
        output_events={"caption_aliases": True},
        quality_targets_ms={"kame_speech_end_to_playback_start_ms": 2500},
        barge_in_policy={"min_rms": 350, "min_speech_ms": 120},
    )

    restored = RealtimeVoiceSessionConfig.from_wire(config.to_wire())

    assert restored.engine == RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE
    assert restored.frontend_model == "gemma-4-E2B-it"
    assert restored.interface_temperature == 0.35
    assert restored.interface_max_output_tokens == 96
    assert restored.interface_timeout_seconds == 0.6
    assert restored.interface_max_audio_seconds == 12.0
    assert restored.interface_audio_input == "native_audio"
    assert restored.interface_base_url == "http://interface.local:8000/v1"
    assert restored.asr_mode == RealtimeVoiceASRMode.SPECULATIVE
    assert restored.asr_provider == "streaming_stt"
    assert restored.asr_model == "nemotron-speech"
    assert restored.asr_base_url == "http://asr.local:8767"
    assert restored.preferred_local_oracle_model == "gemma-4-26B-A4B-it"
    assert restored.max_spoken_sentences == 4
    assert restored.voice_response_policy == "full"
    assert restored.tts_provider == "streaming_tts"
    assert restored.tts_model == "sonic-3.5"
    assert restored.tts_voice == "voice-123"
    assert restored.tts_base_url == "http://tts.local:8768"
    assert restored.fallback_policy == "fail_closed"
    assert restored.turn_acknowledgement == {"enabled": True, "text": "One moment."}
    assert restored.routing_policy == {"local_confidence_threshold": 0.82}
    assert restored.metrics_policy == {"enabled": True, "log_provider_spans": False}
    assert restored.output_events == {"caption_aliases": True}
    assert restored.quality_targets_ms == {"kame_speech_end_to_playback_start_ms": 2500}
    assert restored.barge_in_policy == {"min_rms": 350, "min_speech_ms": 120}


def test_session_config_normalizes_kame_interface_audio_input():
    restored = RealtimeVoiceSessionConfig.from_wire(
        {
            "session_id": "voice-123",
            "engine": RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE.value,
            "interface_audio_input": "native-audio",
        }
    )
    invalid = RealtimeVoiceSessionConfig(
        session_id="voice-456",
        engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        interface_audio_input="surprise-mode",
    )

    assert restored.interface_audio_input == "native_audio"
    assert invalid.to_wire()["interface_audio_input"] == "auto"


def test_session_config_normalizes_kame_voice_response_policy():
    restored = RealtimeVoiceSessionConfig.from_wire(
        {
            "session_id": "voice-123",
            "engine": RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE.value,
            "voice_response_policy": "brief-summary",
        }
    )
    invalid = RealtimeVoiceSessionConfig(
        session_id="voice-456",
        engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        voice_response_policy="read-the-whole-internet",
    )

    assert restored.voice_response_policy == "brief_summary"
    assert invalid.to_wire()["voice_response_policy"] == "sentence_cap"


def test_session_config_bounds_kame_interface_max_audio_seconds():
    too_high = RealtimeVoiceSessionConfig.from_wire(
        {
            "session_id": "voice-123",
            "engine": RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE.value,
            "interface_max_audio_seconds": 45,
        }
    )
    too_low = RealtimeVoiceSessionConfig.from_wire(
        {
            "session_id": "voice-123",
            "engine": RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE.value,
            "interface_max_audio_seconds": 0.25,
        }
    )

    assert too_high.interface_max_audio_seconds == 30.0
    assert too_low.interface_max_audio_seconds == 1.0


def test_kame_engine_factory_uses_kame_interface_oracle_engine():
    engine = create_realtime_voice_engine(
        RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        )
    )

    assert isinstance(engine, KameInterfaceOracleEngine)


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


def test_realtime_voice_event_queue_drops_oldest_assistant_audio_for_control_event():
    async def run():
        queue = asyncio.Queue(maxsize=2)
        first_audio = VoiceEvent(
            type=VoiceEventType.ASSISTANT_AUDIO_CHUNK,
            session_id="voice-123",
            sequence=1,
            payload=AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"one").to_payload(),
        )
        second_audio = VoiceEvent(
            type=VoiceEventType.ASSISTANT_AUDIO_CHUNK,
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


def test_realtime_voice_event_queue_drops_new_assistant_audio_when_control_queue_is_full():
    async def run():
        queue = asyncio.Queue(maxsize=1)
        state = VoiceEvent(
            type=VoiceEventType.FRONTEND_STATE,
            session_id="voice-123",
            sequence=1,
            payload={"status": "ok"},
        )
        audio = VoiceEvent(
            type=VoiceEventType.ASSISTANT_AUDIO_CHUNK,
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
    speech_start_event = VoiceEvent(
        type=VoiceEventType.SPEECH_START,
        session_id="voice-123",
        sequence=2,
        payload={"user_id": "42"},
    )
    speech_end_event = VoiceEvent(
        type=VoiceEventType.SPEECH_END,
        session_id="voice-123",
        sequence=3,
        payload={"user_id": "42"},
    )
    transcript_event = VoiceEvent(
        type=VoiceEventType.TRANSCRIPT_PARTIAL,
        session_id="voice-123",
        sequence=4,
        payload={"text": "hello"},
    )
    interface_event = VoiceEvent(
        type=VoiceEventType.INTERFACE_INTENT_PARTIAL,
        session_id="voice-123",
        sequence=5,
        payload={"intent": "Greeting.", "intent_source": "reflex_audio"},
    )
    oracle_event = VoiceEvent(
        type=VoiceEventType.ORACLE_RESPONSE_PARTIAL,
        session_id="voice-123",
        sequence=6,
        payload={"delta": "Hello", "playback_generation": 1},
    )
    metrics_event = VoiceEvent(
        type=VoiceEventType.SESSION_METRICS,
        session_id="voice-123",
        sequence=7,
        payload={"metrics": {"kame_oracle_called": 1}, "playback_generation": 1},
    )
    audio_alias_event = VoiceEvent(
        type=VoiceEventType.ASSISTANT_AUDIO_CHUNK,
        session_id="voice-123",
        sequence=8,
        payload={"audio_alias_for": VoiceEventType.AUDIO_OUTPUT_CHUNK.value, "playback_generation": 1},
    )
    playback_started_event = VoiceEvent(
        type=VoiceEventType.PLAYBACK_STARTED,
        session_id="voice-123",
        sequence=9,
        payload={"playback_generation": 1},
    )
    playback_stopped_event = VoiceEvent(
        type=VoiceEventType.PLAYBACK_STOPPED,
        session_id="voice-123",
        sequence=10,
        payload={"playback_generation": 1},
    )
    session_stop_event = VoiceEvent(
        type=VoiceEventType.SESSION_STOP,
        session_id="voice-123",
        sequence=11,
        payload={"reason": "client_leave"},
    )

    validate_client_event(audio_event)
    validate_client_event(speech_start_event)
    validate_client_event(speech_end_event)
    validate_client_event(playback_started_event)
    validate_client_event(playback_stopped_event)
    validate_client_event(session_stop_event)
    validate_server_event(transcript_event)
    validate_server_event(interface_event)
    validate_server_event(oracle_event)
    validate_server_event(metrics_event)
    validate_server_event(audio_alias_event)
    validate_server_event(playback_started_event)
    validate_server_event(playback_stopped_event)

    with pytest.raises(ValueError):
        validate_client_event(transcript_event)
    with pytest.raises(ValueError):
        validate_client_event(interface_event)
    with pytest.raises(ValueError):
        validate_client_event(oracle_event)
    with pytest.raises(ValueError):
        validate_client_event(metrics_event)
    with pytest.raises(ValueError):
        validate_client_event(audio_alias_event)

    with pytest.raises(ValueError):
        validate_server_event(audio_event)
    with pytest.raises(ValueError):
        validate_server_event(session_stop_event)


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


def test_barge_in_event_uses_contract_name_and_accepts_legacy_alias():
    event = VoiceEvent(
        type=VoiceEventType.BARGE_IN,
        session_id="voice-123",
        sequence=8,
        timestamp_ms=123457,
        payload={"reason": "user_speech"},
    )

    assert event.to_wire()["type"] == "barge_in.detected"

    restored = VoiceEvent.from_wire(
        {
            "type": "barge_in",
            "session_id": "voice-123",
            "sequence": 8,
            "timestamp_ms": 123457,
            "payload": {"reason": "user_speech"},
        }
    )

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


def test_text_engine_takes_paragraph_boundary_before_normalizing_whitespace():
    chunk, remaining = _take_speakable_chunk("First paragraph is ready.\n\nSecond paragraph follows.")

    assert chunk == "First paragraph is ready."
    assert remaining == "Second paragraph follows."


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


def test_text_engine_emits_opt_in_caption_alias_events(monkeypatch):
    async def run():
        async def fake_speak(self, text, playback_generation):
            return None

        monkeypatch.setattr(TextOracleTTSEngine, "_speak_chunk", fake_speak)

        engine = TextOracleTTSEngine(oracle=FakeOracle())
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                output_events={"caption_aliases": True},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "hello captions", "end_of_utterance": True},
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_CAPTION_FINAL:
                break

        await engine.close()

        partial = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL)
        caption_partial = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_CAPTION_PARTIAL)
        commit = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_COMMIT)
        caption_final = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_CAPTION_FINAL)
        assert caption_partial.payload["text"] == partial.payload["text"]
        assert caption_partial.payload["caption_alias_for"] == VoiceEventType.ASSISTANT_TEXT_PARTIAL.value
        assert caption_partial.payload["playback_generation"] == partial.payload["playback_generation"]
        assert caption_final.payload["text"] == commit.payload["text"]
        assert caption_final.payload["caption_alias_for"] == VoiceEventType.ASSISTANT_COMMIT.value
        assert caption_final.payload["playback_generation"] == commit.payload["playback_generation"]

    asyncio.run(run())


def test_text_engine_emits_opt_in_audio_alias_events(monkeypatch):
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

        engine = TextOracleTTSEngine(oracle=FakeOracle())
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                output_events={"audio_aliases": True},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "hello audio aliases", "end_of_utterance": True},
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_AUDIO_CHUNK:
                break

        await engine.close()

        audio = next(event for event in seen if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK)
        alias = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_AUDIO_CHUNK)
        assert alias.payload["audio_alias_for"] == VoiceEventType.AUDIO_OUTPUT_CHUNK.value
        assert alias.payload["playback_generation"] == audio.payload["playback_generation"]
        assert alias.payload["data_b64"] == audio.payload["data_b64"]
        assert alias.payload["codec"] == audio.payload["codec"]

    asyncio.run(run())


def test_reference_sidecar_mirrors_external_oracle_control_events():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.INTERFACE_ORACLE_CANCEL,
                session_id="voice-123",
                sequence=1,
                payload={
                    "job_id": "voice-oracle-001",
                    "reason": "user requested /voice cancel",
                    "transport": "discord_voice",
                },
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.INTERFACE_ORACLE_UPDATE,
                session_id="voice-123",
                sequence=2,
                payload={
                    "job_id": "voice-oracle-002",
                    "priority": "high",
                    "update_text": "also check Stripe",
                    "reason": "user requested /voice update",
                    "transport": "discord_voice",
                },
            )
        )

        seen = []
        for _ in range(6):
            event = await asyncio.wait_for(anext(sidecar.events()), timeout=1)
            seen.append(event)
            if event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE:
                break
        await sidecar.close()
        return seen

    seen = asyncio.run(run())

    cancel = next(event for event in seen if event.type == VoiceEventType.INTERFACE_ORACLE_CANCEL)
    update = next(event for event in seen if event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE)
    assert cancel.payload["job_id"] == "voice-oracle-001"
    assert cancel.payload["transport"] == "discord_voice"
    assert cancel.payload["sidecar_control"] is True
    assert update.payload["job_id"] == "voice-oracle-002"
    assert update.payload["priority"] == "high"
    assert update.payload["update_text"] == "also check Stripe"
    assert update.payload["sidecar_control"] is True


def test_kame_engine_applies_oracle_cancel_from_sidecar_stream():
    async def run():
        sidecar = FakeSidecar()
        engine = KameInterfaceOracleEngine(oracle=FakeOracle(), sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                oracle_jobs={
                    "enabled": True,
                    "max_concurrent": 1,
                },
            )
        )
        assert engine._oracle_job_manager is not None

        async def runner(job):
            await asyncio.sleep(10)

        job = await engine._oracle_job_manager.submit(
            KameOracleRequest(
                session_id="voice-123",
                turn_id="voice-123:1",
                source="voice",
                user_id="123",
                intent="check the invoice",
                transcript="check the invoice",
            ),
            runner=runner,
        )
        await asyncio.sleep(0)
        await sidecar._events.put(
            VoiceEvent(
                type=VoiceEventType.INTERFACE_ORACLE_CANCEL,
                session_id="voice-123",
                sequence=7,
                payload={
                    "job_id": job.job_id,
                    "reason": "user requested /voice cancel",
                    "transport": "discord_voice",
                    "sidecar_control": True,
                },
            )
        )
        await engine._oracle_job_manager.wait_for_idle()
        stored = await engine._oracle_job_manager.get(job.job_id)
        seen = []
        while not engine._events.empty():
            event = await asyncio.wait_for(anext(engine.events()), timeout=1)
            seen.append(event)
        await engine.close()
        return stored, seen

    stored, seen = asyncio.run(run())

    assert stored.state.value == "cancelled"
    assert stored.cancel_reason == "user requested /voice cancel"
    assert any(event.type == VoiceEventType.ORACLE_JOB_CANCEL_REQUESTED for event in seen)
    assert any(event.type == VoiceEventType.ORACLE_JOB_CANCELLED for event in seen)


def test_kame_engine_applies_oracle_update_from_sidecar_stream():
    async def run():
        sidecar = FakeSidecar()
        engine = KameInterfaceOracleEngine(oracle=FakeOracle(), sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                oracle_jobs={
                    "enabled": True,
                    "max_concurrent": 1,
                },
            )
        )
        assert engine._oracle_job_manager is not None

        async def runner(job):
            await asyncio.sleep(10)

        job = await engine._oracle_job_manager.submit(
            KameOracleRequest(
                session_id="voice-123",
                turn_id="voice-123:1",
                source="voice",
                user_id="123",
                intent="check the invoice",
                transcript="check the invoice",
            ),
            runner=runner,
        )
        await sidecar._events.put(
            VoiceEvent(
                type=VoiceEventType.INTERFACE_ORACLE_UPDATE,
                session_id="voice-123",
                sequence=8,
                payload={
                    "job_id": job.job_id,
                    "priority": "high",
                    "update_text": "include the Stripe receipt",
                    "reason": "user requested /voice update",
                    "transport": "discord_voice",
                    "sidecar_control": True,
                },
            )
        )
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            stored = await engine._oracle_job_manager.get(job.job_id)
            if stored.updates and stored.priority == "high":
                break
            await asyncio.sleep(0)
        stored = await engine._oracle_job_manager.get(job.job_id)
        await engine._oracle_job_manager.cancel(job.job_id, reason="test cleanup")
        await engine._oracle_job_manager.wait_for_idle()
        await engine.close()
        return stored

    stored = asyncio.run(run())

    assert stored.priority == "high"
    assert stored.updates[-1]["text"] == "include the Stripe receipt"


def test_reference_sidecar_emits_opt_in_caption_alias_events():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                output_events={"caption_aliases": True},
            )
        )
        await sidecar._emit(
            VoiceEventType.ASSISTANT_TEXT_PARTIAL,
            {"text": "Partial caption.", "playback_generation": 4},
        )
        await sidecar._emit(
            VoiceEventType.ASSISTANT_COMMIT,
            {"text": "Final caption.", "playback_generation": 4},
        )

        seen = []
        for _ in range(8):
            event = await asyncio.wait_for(anext(sidecar.events()), timeout=1)
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_CAPTION_FINAL:
                break
        await sidecar.close()
        return seen

    seen = asyncio.run(run())

    partial = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL)
    caption_partial = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_CAPTION_PARTIAL)
    commit = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_COMMIT)
    caption_final = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_CAPTION_FINAL)
    assert caption_partial.payload["text"] == partial.payload["text"]
    assert caption_partial.payload["caption_alias_for"] == VoiceEventType.ASSISTANT_TEXT_PARTIAL.value
    assert caption_partial.payload["playback_generation"] == partial.payload["playback_generation"]
    assert caption_final.payload["text"] == commit.payload["text"]
    assert caption_final.payload["caption_alias_for"] == VoiceEventType.ASSISTANT_COMMIT.value
    assert caption_final.payload["playback_generation"] == commit.payload["playback_generation"]


def test_reference_sidecar_emits_opt_in_audio_alias_events():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                output_events={"audio_aliases": True},
            )
        )
        await sidecar._emit(
            VoiceEventType.AUDIO_OUTPUT_CHUNK,
            {
                **AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"audio").to_payload(),
                "playback_generation": 4,
            },
        )

        seen = []
        for _ in range(6):
            event = await asyncio.wait_for(anext(sidecar.events()), timeout=1)
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_AUDIO_CHUNK:
                break
        await sidecar.close()
        return seen

    seen = asyncio.run(run())

    audio = next(event for event in seen if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK)
    alias = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_AUDIO_CHUNK)
    assert alias.payload["audio_alias_for"] == VoiceEventType.AUDIO_OUTPUT_CHUNK.value
    assert alias.payload["playback_generation"] == audio.payload["playback_generation"]
    assert alias.payload["data_b64"] == audio.payload["data_b64"]
    assert alias.payload["codec"] == audio.payload["codec"]


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


def test_kame_oracle_prompt_separates_reflex_intent_from_asr_evidence():
    request = KameOracleRequest(
        session_id="voice-123",
        turn_id="turn-1",
        source="discord_voice",
        user_id="42",
        intent="Find the note from yesterday's meeting.",
        route=KameRoute.ORACLE_DIRECT,
        route_confidence=0.81,
        transcript="find the note from yesterday's meeting",
        transcript_source="reflex_audio",
        transcript_confidence=0.73,
        asr_transcript="find the node from yesterday's meeting",
        asr_transcript_source="asr",
        asr_transcript_confidence=0.68,
        interface_already_said="One moment.",
        conversation_summary="The user is testing KAME voice.",
        reflex_validation_error="oracle_required_for_files",
        interface_input_source="local_stt",
        interface_audio_input_fallback=True,
        job_updates=("also check the Stripe receipt before answering",),
    )

    prompt = _voice_oracle_prompt(request.oracle_text, request.to_metadata())

    assert "KAME request" in prompt
    assert "Reflex interpreted intent (reflex_audio): Find the note" in prompt
    assert "Reflex route: oracle_direct (confidence 0.81)." in prompt
    assert "The audio-native reflex was unavailable; this turn used local_stt as the interface fallback." in prompt
    assert "Reflex route override: oracle_required_for_files." in prompt
    assert "Reflex transcript hypothesis (reflex_audio): find the note" in prompt
    assert "Verbatim ASR evidence (asr): find the node" in prompt
    assert "tool arguments" in prompt
    assert "oracle-facing text was selected from asr evidence" in prompt
    assert "preserve the reflex intent and route as the control signal" in prompt
    assert "The voice reflex already told the user: One moment." in prompt
    assert "User added updates for this oracle job: also check the Stripe receipt before answering" in prompt
    assert "Requested response style: spoken=true; policy=sentence_cap; avoid automatic follow-up offers." in prompt


def test_kame_oracle_request_accepts_transport_and_speaker_aliases():
    request = KameOracleRequest.from_turn(
        session_id="voice-123",
        turn_id="turn-1",
        source="voice",
        user_id="session-user",
        payload={
            "transport": "discord_voice",
            "speaker_id": "turn-speaker",
            "intent": "Check the repository status.",
            "text": "check the repository status",
            "route": "oracle_direct",
        },
        fallback_text="check status",
    )

    assert request.source == "discord_voice"
    assert request.user_id == "turn-speaker"


def test_kame_oracle_request_preserves_interface_audio_input_fallback_flag():
    request = KameOracleRequest.from_turn(
        session_id="voice-123",
        turn_id="turn-1",
        source="voice",
        user_id="session-user",
        payload={
            "intent": "Check deployment status.",
            "text": "check deployment status",
            "route": "oracle_direct",
            "transcript": "check deployment status",
            "transcript_source": "asr",
            "asr_transcript": "check deployment status",
            "asr_transcript_source": "asr",
            "interface_audio_input_fallback": True,
            "interface_input_source": "local_stt",
            "reflex_provider": "local_stt",
        },
        fallback_text="check status",
    )

    metadata = request.to_metadata()
    assert request.interface_audio_input_fallback is True
    assert metadata["kame_interface_audio_input_fallback"] is True
    assert metadata["kame_interface_input_source"] == "local_stt"
    assert metadata["kame_reflex_provider"] == "local_stt"


def test_kame_engine_sends_structured_request_to_oracle(monkeypatch):
    class StructuredOracle:
        def __init__(self):
            self.requests = []

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            yield "Done."

        async def stream_answer_with_metadata(self, transcript, metadata):
            raise AssertionError("KAME engine should prefer stream_answer_for_request")

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = StructuredOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
                metadata={"transport": "discord_voice", "user_id": "42"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "find the note from yesterday's meeting",
                    "intent": "Find the note from yesterday's meeting.",
                    "intent_source": "reflex_audio",
                    "route_confidence": 0.81,
                    "transcript_source": "reflex_audio",
                    "transcript_confidence": 0.73,
                    "asr_transcript": "find the node from yesterday's meeting",
                    "asr_transcript_source": "asr",
                    "asr_transcript_confidence": 0.68,
                    "interface_input_source": "native_audio",
                    "reflex_provider": "vllm",
                    "interface_already_said": "One moment.",
                    "conversation_summary": "The user is testing KAME voice.",
                    "priority": "high",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()
        assert len(oracle.requests) == 1
        request = oracle.requests[0]
        assert request.intent == "Find the note from yesterday's meeting."
        assert request.intent_source == "reflex_audio"
        assert request.route == KameRoute.ORACLE_DIRECT
        assert request.route_confidence == 0.81
        assert request.transcript == "find the note from yesterday's meeting"
        assert request.transcript_source == "reflex_audio"
        assert request.transcript_confidence == 0.73
        assert request.asr_transcript == "find the node from yesterday's meeting"
        assert request.asr_transcript_source == "asr"
        assert request.asr_transcript_confidence == 0.68
        assert request.oracle_text == "find the node from yesterday's meeting"
        assert request.oracle_text_source == "asr"
        assert request.interface_input_source == "native_audio"
        assert request.reflex_provider == "vllm"
        assert request.priority == "high"
        assert request.source == "discord_voice"
        assert request.user_id == "42"
        assert request.mode == "voice"
        assert request.urgency == "interactive"
        assert request.requested_response_style == {
            "spoken": True,
            "max_sentences": 2,
            "policy": "sentence_cap",
            "allow_followup_offer": False,
        }
        assert request.cancellation_token == "voice-123:1:cancel"
        final = next(event for event in seen if event.type == VoiceEventType.TRANSCRIPT_FINAL)
        intent = next(event for event in seen if event.type == VoiceEventType.INTERFACE_INTENT_FINAL)
        oracle_request = next(event for event in seen if event.type == VoiceEventType.INTERFACE_ORACLE_REQUEST)
        interface_commit = next(event for event in seen if event.type == VoiceEventType.INTERFACE_COMMIT)
        session_metrics = next(event for event in seen if event.type == VoiceEventType.SESSION_METRICS)
        commit = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_COMMIT)
        assert final.payload["kame_session_id"] == "voice-123"
        assert final.payload["kame_intent"] == "Find the note from yesterday's meeting."
        assert final.payload["kame_route"] == "oracle_direct"
        assert final.payload["kame_priority"] == "high"
        assert final.payload["kame_route_confidence"] == 0.81
        assert final.payload["kame_mode"] == "voice"
        assert final.payload["kame_urgency"] == "interactive"
        assert final.payload["kame_transcript"] == "find the note from yesterday's meeting"
        assert final.payload["kame_asr_transcript"] == "find the node from yesterday's meeting"
        assert final.payload["kame_oracle_text_source"] == "asr"
        assert final.payload["kame_interface_input_source"] == "native_audio"
        assert final.payload["kame_reflex_provider"] == "vllm"
        assert final.payload["kame_cancellation_token"] == "voice-123:1:cancel"
        assert intent.payload["session_id"] == "voice-123"
        assert intent.payload["route"] == "oracle_direct"
        assert intent.payload["route_confidence"] == 0.81
        assert intent.payload["intent"] == "Find the note from yesterday's meeting."
        assert intent.payload["mode"] == "voice"
        assert intent.payload["urgency"] == "interactive"
        assert intent.payload["transcript"] == "find the note from yesterday's meeting"
        assert intent.payload["asr_transcript"] == "find the node from yesterday's meeting"
        assert intent.payload["cancellation_token"] == "voice-123:1:cancel"
        assert oracle_request.payload["session_id"] == "voice-123"
        assert oracle_request.payload["turn_id"] == "voice-123:1"
        assert oracle_request.payload["route_confidence"] == 0.81
        assert oracle_request.payload["mode"] == "voice"
        assert oracle_request.payload["urgency"] == "interactive"
        assert oracle_request.payload["text"] == "find the node from yesterday's meeting"
        assert oracle_request.payload["oracle_text_source"] == "asr"
        assert oracle_request.payload["transcript"] == "find the note from yesterday's meeting"
        assert oracle_request.payload["asr_transcript"] == "find the node from yesterday's meeting"
        assert oracle_request.payload["interface_input_source"] == "native_audio"
        assert oracle_request.payload["reflex_provider"] == "vllm"
        assert oracle_request.payload["requested_response_style"] == {
            "spoken": True,
            "max_sentences": 2,
            "policy": "sentence_cap",
            "allow_followup_offer": False,
        }
        assert oracle_request.payload["cancellation_token"] == "voice-123:1:cancel"
        assert interface_commit.payload["text"] == "Done."
        assert interface_commit.payload["session_id"] == "voice-123"
        assert interface_commit.payload["mode"] == "voice"
        assert interface_commit.payload["urgency"] == "interactive"
        assert interface_commit.payload["route_confidence"] == 0.81
        assert interface_commit.payload["cancellation_token"] == "voice-123:1:cancel"
        assert session_metrics.payload["outcome"] == "oracle_commit"
        assert session_metrics.payload["oracle_called"] is True
        assert session_metrics.payload["session_id"] == "voice-123"
        assert session_metrics.payload["turn_id"] == "voice-123:1"
        assert session_metrics.payload["mode"] == "voice"
        assert session_metrics.payload["urgency"] == "interactive"
        assert session_metrics.payload["metrics"]["kame_oracle_called"] == 1
        assert session_metrics.payload["metrics"]["kame_oracle_bypassed"] == 0
        assert commit.payload["kame_session_id"] == "voice-123"
        assert commit.payload["session_id"] == "voice-123"
        assert commit.payload["kame_cancellation_token"] == "voice-123:1:cancel"
        assert commit.payload["metrics"]["kame_oracle_called"] == 1
        assert commit.payload["metrics"]["kame_oracle_bypassed"] == 0
        assert spoken == ["Done."]

    asyncio.run(run())


def test_text_engine_suppresses_oracle_thinking_from_speech_and_commit(monkeypatch):
    class ThinkingOracle:
        async def stream_answer(self, transcript: str):
            yield "<think>planning the spoken answer"
            yield "</think>"
            yield "Visible answer."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)
            await self._emit(
                VoiceEventType.AUDIO_OUTPUT_CHUNK,
                {
                    **AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"audio").to_payload(),
                    "playback_generation": playback_generation,
                },
            )

        monkeypatch.setattr(TextOracleTTSEngine, "_speak_chunk", fake_speak)

        engine = TextOracleTTSEngine(oracle=ThinkingOracle())
        await engine.start(RealtimeVoiceSessionConfig(session_id="voice-123"))
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "test", "end_of_utterance": True},
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()
        assistant_text = "\n".join(
            str(event.payload.get("text") or "")
            for event in seen
            if event.type
            in {
                VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                VoiceEventType.INTERFACE_COMMIT,
                VoiceEventType.ASSISTANT_COMMIT,
            }
        )
        assert "planning" not in assistant_text
        assert "<think>" not in assistant_text
        assert "</think>" not in assistant_text
        assert "Visible answer." in assistant_text
        assert spoken == ["Visible answer."]

    asyncio.run(run())


def test_kame_engine_drops_asr_evidence_when_asr_mode_disabled(monkeypatch):
    class StructuredOracle:
        def __init__(self):
            self.requests = []

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            yield "Done."

    async def run():
        async def fake_speak(self, text, playback_generation):
            return None

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = StructuredOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.DISABLED,
                metadata={"transport": "discord_voice", "user_id": "42"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "text": "reflex wording",
                    "intent": "Reflex intent.",
                    "intent_source": "reflex_audio",
                    "route": "oracle_direct",
                    "transcript": "reflex wording",
                    "transcript_source": "reflex_audio",
                    "asr_transcript": "literal ASR should not be used",
                    "asr_transcript_source": "asr",
                    "asr_transcript_confidence": 0.99,
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()
        assert len(oracle.requests) == 1
        request = oracle.requests[0]
        assert request.asr_transcript == ""
        assert request.asr_transcript_source == ""
        assert request.asr_transcript_confidence is None
        assert request.oracle_text == "reflex wording"
        assert request.oracle_text_source == "reflex_audio"
        final = next(event for event in seen if event.type == VoiceEventType.TRANSCRIPT_FINAL)
        intent = next(event for event in seen if event.type == VoiceEventType.INTERFACE_INTENT_FINAL)
        oracle_request = next(event for event in seen if event.type == VoiceEventType.INTERFACE_ORACLE_REQUEST)
        assert "kame_asr_transcript" not in final.payload
        assert "asr_transcript" not in intent.payload
        assert "asr_transcript" not in oracle_request.payload
        assert oracle_request.payload["text"] == "reflex wording"
        assert oracle_request.payload["oracle_text_source"] == "reflex_audio"

    asyncio.run(run())


def test_kame_engine_uses_session_source_when_transport_is_absent():
    async def run():
        engine = KameInterfaceOracleEngine(oracle=FakeOracle())
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
                metadata={"source": "desktop", "user_id": "session-user"},
            )
        )
        request = engine._kame_oracle_request(
            "open the workspace status",
            1,
            oracle_payload={
                "text": "open the workspace status",
                "intent": "Open the workspace status.",
                "intent_source": "reflex_audio",
                "route": "oracle_direct",
            },
            metadata={},
            cancellation_token="voice-123:1:cancel",
        )
        await engine.close()

        assert request is not None
        assert request.source == "desktop"
        assert request.user_id == "session-user"

    asyncio.run(run())


def test_kame_engine_defer_acknowledgement_is_reflex_context(monkeypatch):
    class StructuredOracle:
        def __init__(self):
            self.requests = []

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            yield "The deployment is healthy."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = StructuredOracle()
        sidecar = FakeSidecar()
        engine = KameInterfaceOracleEngine(oracle=oracle, sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                sidecar_base_url="http://voice.local:8765",
                metadata={
                    "transport": "discord_voice",
                    "turn_acknowledgement": {"enabled": True, "text": "One moment."},
                },
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "check the deployment status",
                    "intent": "Check the deployment status.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "transcript_source": "asr",
                    "interface_already_said": "Checking that now.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()
        assert len(oracle.requests) == 1
        request = oracle.requests[0]
        assert request.route == KameRoute.DEFER
        assert request.interface_already_said == "Checking that now."

        defer = next(event for event in seen if event.type == VoiceEventType.INTERFACE_REPLY_DEFER)
        oracle_request = next(event for event in seen if event.type == VoiceEventType.INTERFACE_ORACLE_REQUEST)
        narration = next(
            event
            for event in seen
            if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL and event.payload.get("reflex_narration")
        )
        commit = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_COMMIT)
        forwarded_interface_events = [
            event
            for event in sidecar.received
            if event.type
            in {
                VoiceEventType.INTERFACE_INTENT_FINAL,
                VoiceEventType.INTERFACE_REPLY_DEFER,
                VoiceEventType.INTERFACE_ORACLE_REQUEST,
                VoiceEventType.INTERFACE_COMMIT,
            }
        ]
        assert defer.payload["route"] == "defer"
        assert defer.payload["interface_already_said"] == "Checking that now."
        assert defer.payload["text"] == "Checking that now."
        assert defer.payload["reflex_narration_text"] == "Checking that now."
        assert defer.payload["oracle_text"] == "check the deployment status"
        assert defer.payload["oracle_text_source"] == "asr"
        assert oracle_request.payload["route"] == "defer"
        assert oracle_request.payload["turn_id"] == "voice-123:1"
        assert oracle_request.payload["interface_already_said"] == "Checking that now."
        assert oracle_request.payload["intent"] == "Check the deployment status."
        assert oracle_request.payload["text"] == "check the deployment status"
        assert oracle_request.payload["oracle_text_source"] == "asr"
        assert "reflex_narration_text" not in oracle_request.payload
        assert narration.payload["text"] == "Checking that now."
        assert narration.payload["kame_interface_already_said"] == "Checking that now."
        assert commit.payload["text"] == "The deployment is healthy."
        assert [event.type for event in forwarded_interface_events] == [
            VoiceEventType.INTERFACE_INTENT_FINAL,
            VoiceEventType.INTERFACE_REPLY_DEFER,
            VoiceEventType.INTERFACE_ORACLE_REQUEST,
            VoiceEventType.INTERFACE_COMMIT,
        ]
        assert forwarded_interface_events[1].payload == defer.payload
        assert forwarded_interface_events[2].payload == oracle_request.payload
        assert forwarded_interface_events[3].payload["text"] == "The deployment is healthy."
        assert spoken == ["Checking that now.", "The deployment is healthy."]

    asyncio.run(run())


def test_kame_engine_async_oracle_job_emits_lifecycle_without_blocking_ack(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await self.release.wait()
            yield "The deployment is healthy."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        sidecar = FakeSidecar()
        engine = KameInterfaceOracleEngine(oracle=oracle, sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                sidecar_base_url="http://voice.local:8765",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "check the deployment status",
                    "intent": "Check the deployment status.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "transcript_source": "asr",
                    "interface_already_said": "Checking that now.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_STARTED:
                break

        await asyncio.wait_for(oracle.started.wait(), timeout=1)
        assert len(oracle.requests) == 1
        assert [event.type for event in seen if event.type in {
            VoiceEventType.INTERFACE_REPLY_DEFER,
            VoiceEventType.INTERFACE_ORACLE_REQUEST,
            VoiceEventType.ORACLE_JOB_ACCEPTED,
            VoiceEventType.ORACLE_JOB_STARTED,
        }] == [
            VoiceEventType.INTERFACE_REPLY_DEFER,
            VoiceEventType.INTERFACE_ORACLE_REQUEST,
            VoiceEventType.ORACLE_JOB_ACCEPTED,
            VoiceEventType.ORACLE_JOB_STARTED,
        ]
        assert not any(event.type == VoiceEventType.ASSISTANT_COMMIT for event in seen)
        assert spoken == ["Checking that now."]

        oracle.release.set()
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT and event.payload.get("oracle_job_result"):
                break

        await engine.close()
        completed = next(event for event in seen if event.type == VoiceEventType.ORACLE_JOB_COMPLETED)
        commit = next(
            event
            for event in seen
            if event.type == VoiceEventType.ASSISTANT_COMMIT and event.payload.get("oracle_job_result")
        )
        assert completed.payload["job_id"] == "voice-oracle-001"
        assert completed.payload["state"] == "completed"
        assert completed.payload["result_summary"] == "The deployment is healthy."
        assert commit.payload["text"] == "The deployment is healthy."
        assert commit.payload["oracle_job_id"] == "voice-oracle-001"
        assert spoken == ["Checking that now.", "The deployment is healthy."]
        forwarded_job_events = [
            event
            for event in sidecar.received
            if event.type in {
                VoiceEventType.ORACLE_JOB_ACCEPTED,
                VoiceEventType.ORACLE_JOB_STARTED,
                VoiceEventType.ORACLE_JOB_COMPLETED,
            }
        ]
        assert [event.type for event in forwarded_job_events] == [
            VoiceEventType.ORACLE_JOB_ACCEPTED,
            VoiceEventType.ORACLE_JOB_STARTED,
            VoiceEventType.ORACLE_JOB_COMPLETED,
        ]

    asyncio.run(run())


def test_kame_engine_documented_oracle_jobs_config_enables_async_scheduler(monkeypatch):
    class ImmediateOracle:
        def __init__(self):
            self.requests = []

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            yield "The documented async config worked."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = ImmediateOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle, sidecar=FakeSidecar())
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                sidecar_base_url="http://voice.local:8765",
                oracle_jobs={
                    "max_concurrent": 1,
                    "queue_limit": 4,
                    "default_priority": "normal",
                    "overflow_policy": "queue",
                    "shutdown_timeout_seconds": 0.01,
                },
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "check the deployment status",
                    "intent": "Check the deployment status.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Checking that now.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        events = engine.events()
        while True:
            event = await asyncio.wait_for(events.__anext__(), timeout=1)
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_COMPLETED:
                break

        await engine.close()

        lifecycle = [
            event.type
            for event in seen
            if event.type in {
                VoiceEventType.ORACLE_JOB_ACCEPTED,
                VoiceEventType.ORACLE_JOB_STARTED,
                VoiceEventType.ORACLE_JOB_COMPLETED,
            }
        ]
        assert lifecycle == [
            VoiceEventType.ORACLE_JOB_ACCEPTED,
            VoiceEventType.ORACLE_JOB_STARTED,
            VoiceEventType.ORACLE_JOB_COMPLETED,
        ]
        assert len(oracle.requests) == 1
        assert spoken[0] == "Checking that now."

    asyncio.run(run())


def test_kame_engine_async_oracle_job_writes_configured_audit_ledger(monkeypatch, tmp_path):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await self.release.wait()
            yield "The VoIP provisioning plan is ready."

    async def run():
        async def fake_speak(self, text, playback_generation):
            return None

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        ledger_path = tmp_path / "voiceops-oracle-jobs.jsonl"
        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle, sidecar=FakeSidecar())
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                sidecar_base_url="http://voice.local:8765",
                oracle_jobs={
                    "enabled": True,
                    "max_concurrent": 1,
                    "queue_limit": 4,
                    "audit_ledger_path": str(ledger_path),
                },
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "give Hermes two hundred dollars and provision VoIP",
                    "intent": "Use a 200 dollar budget to provision VoIP safely.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "transcript_source": "asr",
                    "interface_already_said": "I'm preparing the approval packet.",
                    "end_of_utterance": True,
                },
            )
        )

        async for event in engine.events():
            if event.type == VoiceEventType.ORACLE_JOB_STARTED:
                break

        await asyncio.wait_for(oracle.started.wait(), timeout=1)
        oracle.release.set()
        async for event in engine.events():
            if event.type == VoiceEventType.ORACLE_JOB_COMPLETED:
                break

        await engine.close()
        rows = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
        ]

        assert [row["event_type"] for row in rows] == [
            "oracle.job.accepted",
            "oracle.job.started",
            "oracle.job.completed",
        ]
        assert rows[0]["payload"]["spoken_status"] == "I'm preparing the approval packet."
        assert rows[-1]["payload"]["result_summary"] == "The VoIP provisioning plan is ready."
        assert "oracle_text" not in rows[0]["payload"]
        assert "metadata" not in rows[0]["payload"]

    asyncio.run(run())


def test_kame_engine_async_oracle_job_allows_local_turn_while_running(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await self.release.wait()
            yield "The deployment is healthy."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "check the deployment status",
                    "intent": "Check the deployment status.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Checking that now.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_STARTED:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload={
                    "transcript": "can you hear me",
                    "intent": "The user is checking whether Hermes can hear them.",
                    "route": "local",
                    "local_reply": "Yes, I can hear you.",
                    "intent_source": "reflex_audio",
                    "end_of_utterance": True,
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        assert len(oracle.requests) == 1
        local_commit = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_COMMIT)
        assert local_commit.payload["text"] == "Yes, I can hear you."
        assert spoken == ["Checking that now.", "Yes, I can hear you."]

        oracle.release.set()
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_COMPLETED:
                break

        await engine.close()
        completed = next(event for event in seen if event.type == VoiceEventType.ORACLE_JOB_COMPLETED)
        assert completed.payload["result_summary"] == "The deployment is healthy."
        assert completed.payload["source_playback_generation"] == 1
        assert completed.payload["playback_generation"] == 2
        assert not any(event.payload.get("oracle_job_result") for event in seen)
        assert spoken == ["Checking that now.", "Yes, I can hear you."]

    asyncio.run(run())


def test_kame_engine_can_create_oracle_job_while_another_is_running(monkeypatch):
    class ConcurrentOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()
            self.second_started = asyncio.Event()
            self.releases: dict[str, asyncio.Event] = {}

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            release = asyncio.Event()
            self.releases[request.intent] = release
            if len(self.requests) == 1:
                self.started.set()
            if len(self.requests) == 2:
                self.second_started.set()
            await release.wait()
            yield f"Finished {request.intent}"

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = ConcurrentOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 2, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "check deployment status",
                    "intent": "Check deployment status.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Checking deployment status.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if (
                event.type == VoiceEventType.ORACLE_JOB_STARTED
                and event.payload.get("intent") == "Check deployment status."
            ):
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload={
                    "transcript": "draft vendor memo",
                    "intent": "Draft vendor memo.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Drafting the vendor memo.",
                    "end_of_utterance": True,
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            if (
                event.type == VoiceEventType.ORACLE_JOB_STARTED
                and event.payload.get("intent") == "Draft vendor memo."
            ):
                break
        await asyncio.wait_for(oracle.second_started.wait(), timeout=1)

        assert [request.intent for request in oracle.requests] == [
            "Check deployment status.",
            "Draft vendor memo.",
        ]
        assert any(
            event.type == VoiceEventType.ORACLE_JOB_STARTED
            and event.payload.get("job_id") == "voice-oracle-001"
            and event.payload.get("intent") == "Check deployment status."
            for event in seen
        )
        assert any(
            event.type == VoiceEventType.ORACLE_JOB_STARTED
            and event.payload.get("job_id") == "voice-oracle-002"
            and event.payload.get("intent") == "Draft vendor memo."
            for event in seen
        )
        assert not any(event.type == VoiceEventType.ORACLE_JOB_QUEUED for event in seen)
        assert spoken == ["Checking deployment status.", "Drafting the vendor memo."]

        oracle.releases["Check deployment status."].set()
        oracle.releases["Draft vendor memo."].set()
        async for event in engine.events():
            seen.append(event)
            completed = [
                item
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_COMPLETED
            ]
            if len(completed) == 2:
                break

        await engine.close()
        completed = [
            event
            for event in seen
            if event.type == VoiceEventType.ORACLE_JOB_COMPLETED
        ]
        assert [event.payload["job_id"] for event in completed] == [
            "voice-oracle-001",
            "voice-oracle-002",
        ]
        assert completed[0].payload["source_playback_generation"] == 1
        assert completed[1].payload["source_playback_generation"] == 2

    asyncio.run(run())


def test_oracle_direct_async_job_completion_after_local_turn_is_lifecycle_only(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await self.release.wait()
            yield "The deployment is healthy."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "check the deployment status",
                    "intent": "Check the deployment status.",
                    "intent_source": "reflex_audio",
                    "route": "oracle_direct",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_STARTED:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)

        assert len(oracle.requests) == 1
        assert oracle.requests[0].route == KameRoute.ORACLE_DIRECT
        assert not any(event.type == VoiceEventType.ASSISTANT_COMMIT for event in seen)
        assert any(event.type == VoiceEventType.INTERFACE_ORACLE_REQUEST for event in seen)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload={
                    "transcript": "can you hear me",
                    "intent": "The user is checking whether Hermes can hear them.",
                    "route": "local",
                    "local_reply": "Yes, I can hear you.",
                    "intent_source": "reflex_audio",
                    "end_of_utterance": True,
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        local_commit = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_COMMIT)
        assert local_commit.payload["text"] == "Yes, I can hear you."

        oracle.release.set()
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_COMPLETED:
                break

        await engine.close()
        completed = next(event for event in seen if event.type == VoiceEventType.ORACLE_JOB_COMPLETED)
        assert completed.payload["result_summary"] == "The deployment is healthy."
        assert completed.payload["source_playback_generation"] == 1
        assert completed.payload["playback_generation"] == 2
        assert not any(event.payload.get("oracle_job_result") for event in seen)
        assert spoken == ["Yes, I can hear you."]

    asyncio.run(run())


def test_completed_async_oracle_job_after_intervening_local_turn_is_lifecycle_only(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await self.release.wait()
            yield "The deployment is healthy."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "check the deployment status",
                    "intent": "Check the deployment status.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Checking that now.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_STARTED:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload={
                    "transcript": "can you hear me",
                    "intent": "The user is checking whether Hermes can hear them.",
                    "route": "local",
                    "local_reply": "Yes, I can hear you.",
                    "intent_source": "reflex_audio",
                    "end_of_utterance": True,
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        oracle.release.set()
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_COMPLETED:
                break

        await engine.close()
        completed = next(event for event in seen if event.type == VoiceEventType.ORACLE_JOB_COMPLETED)
        assert completed.payload["result_summary"] == "The deployment is healthy."
        assert completed.payload["source_playback_generation"] == 1
        assert completed.payload["playback_generation"] == 2
        assert not any(event.payload.get("oracle_job_result") for event in seen)
        assert spoken == ["Checking that now.", "Yes, I can hear you."]

    asyncio.run(run())


def test_kame_engine_local_status_question_uses_oracle_job_state(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await self.release.wait()
            yield "The deployment is healthy."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "check the deployment status",
                    "intent": "Check the deployment status.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Checking that now.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_STARTED:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload={
                    "transcript": "what are you working on",
                    "intent": "What are you working on?",
                    "route": "local",
                    "local_reply": "Let me check.",
                    "intent_source": "reflex_audio",
                    "end_of_utterance": True,
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()
        assert len(oracle.requests) == 1
        status_transcript = next(
            event
            for event in seen
            if event.type == VoiceEventType.TRANSCRIPT_FINAL
            and event.payload.get("oracle_job_status_poll") is True
        )
        status_intent = next(
            event
            for event in seen
            if event.type == VoiceEventType.INTERFACE_INTENT_FINAL
            and event.payload.get("oracle_job_status_poll") is True
        )
        status_reply = next(
            event
            for event in seen
            if event.type == VoiceEventType.INTERFACE_REPLY_LOCAL
            and event.payload.get("oracle_job_status_poll") is True
        )
        commit = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_COMMIT)
        assert status_transcript.payload["durable"] is False
        assert status_intent.payload["durable"] is False
        assert status_reply.payload["durable"] is False
        assert commit.payload["oracle_job_status_poll"] is True
        assert commit.payload["durable"] is False
        assert commit.payload["local_reply"] is True
        assert commit.payload["text"] == "Oracle jobs: 1 running out of 1. running: Checking that now."
        assert spoken == [
            "Checking that now.",
            "Oracle jobs: 1 running out of 1. running: Checking that now.",
        ]

    asyncio.run(run())


def test_kame_engine_fifth_async_oracle_job_queues_and_starts_after_capacity_frees(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.releases = {}
            self.request_count_changed = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.request_count_changed.set()
            event = self.releases.setdefault(request.intent, asyncio.Event())
            await event.wait()
            yield f"Finished {request.intent}."

        def release(self, intent):
            self.releases.setdefault(intent, asyncio.Event()).set()

        async def wait_for_requests(self, count):
            while len(self.requests) < count:
                self.request_count_changed.clear()
                await asyncio.wait_for(self.request_count_changed.wait(), timeout=1)

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 4, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )

        seen = []
        for index in range(1, 6):
            await engine.receive_event(
                VoiceEvent(
                    type=VoiceEventType.AUDIO_INPUT_CHUNK,
                    session_id="voice-123",
                    sequence=index,
                    payload={
                        "transcript": f"run task {index}",
                        "intent": f"Run task {index}",
                        "intent_source": "reflex_audio",
                        "route": "defer",
                        "interface_already_said": f"Starting task {index}.",
                        "end_of_utterance": True,
                    },
                )
            )

        queued = None
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_QUEUED:
                queued = event
            started_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_STARTED
            ]
            if queued is not None and started_ids == [
                "voice-oracle-001",
                "voice-oracle-002",
                "voice-oracle-003",
                "voice-oracle-004",
            ]:
                break

        started = [event for event in seen if event.type == VoiceEventType.ORACLE_JOB_STARTED]
        assert queued is not None
        assert [event.payload["job_id"] for event in started] == [
            "voice-oracle-001",
            "voice-oracle-002",
            "voice-oracle-003",
            "voice-oracle-004",
        ]
        assert queued.payload["job_id"] == "voice-oracle-005"
        assert queued.payload["state"] == "queued"
        await oracle.wait_for_requests(4)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=6,
                payload={
                    "transcript": "what are you working on",
                    "intent": "What are you working on?",
                    "route": "local",
                    "local_reply": "Let me check.",
                    "intent_source": "reflex_audio",
                    "end_of_utterance": True,
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            if (
                event.type == VoiceEventType.ASSISTANT_COMMIT
                and str(event.payload.get("text") or "").startswith("Oracle jobs:")
            ):
                break
        status_commit = next(
            event
            for event in seen
            if event.type == VoiceEventType.ASSISTANT_COMMIT
            and str(event.payload.get("text") or "").startswith("Oracle jobs:")
        )
        assert status_commit.payload["text"].startswith("Oracle jobs: 4 running out of 4, 1 queued.")

        oracle.release("Run task 1")
        async for event in engine.events():
            seen.append(event)
            started_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_STARTED
            ]
            if "voice-oracle-005" in started_ids:
                break
        await oracle.wait_for_requests(5)

        await engine.close()
        assert len(oracle.requests) == 5

    asyncio.run(run())


def test_kame_engine_can_cancel_queued_async_oracle_job_before_it_starts(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()
            self.release_first = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await self.release_first.wait()
            yield f"Finished {request.intent}."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        for index in range(1, 3):
            await engine.receive_event(
                VoiceEvent(
                    type=VoiceEventType.AUDIO_INPUT_CHUNK,
                    session_id="voice-123",
                    sequence=index,
                    payload={
                        "transcript": f"run task {index}",
                        "intent": f"Run task {index}",
                        "intent_source": "reflex_audio",
                        "route": "defer",
                        "interface_already_said": f"Starting task {index}.",
                        "end_of_utterance": True,
                    },
                )
            )

        seen = []
        queued = None
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_QUEUED:
                queued = event
            started_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_STARTED
            ]
            if queued is not None and started_ids == ["voice-oracle-001"]:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)
        assert queued is not None
        assert len(oracle.requests) == 1

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.INTERFACE_ORACLE_CANCEL,
                session_id="voice-123",
                sequence=3,
                payload={"job_id": "voice-oracle-002", "reason": "queued task no longer needed"},
            )
        )
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_CANCELLED:
                break

        cancelled = next(event for event in seen if event.type == VoiceEventType.ORACLE_JOB_CANCELLED)
        assert cancelled.payload["job_id"] == "voice-oracle-002"
        assert cancelled.payload["state"] == "cancelled"
        assert cancelled.payload["cancel_reason"] == "queued task no longer needed"
        assert len(oracle.requests) == 1

        oracle.release_first.set()
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_COMPLETED:
                break

        await engine.close()
        assert len(oracle.requests) == 1
        assert not any(
            event.type == VoiceEventType.ORACLE_JOB_STARTED and event.payload["job_id"] == "voice-oracle-002"
            for event in seen
        )

    asyncio.run(run())


def test_kame_engine_can_cancel_all_async_oracle_jobs(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await asyncio.Event().wait()
            yield f"Finished {request.intent}."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        for index in range(1, 3):
            await engine.receive_event(
                VoiceEvent(
                    type=VoiceEventType.AUDIO_INPUT_CHUNK,
                    session_id="voice-123",
                    sequence=index,
                    payload={
                        "transcript": f"run task {index}",
                        "intent": f"Run task {index}",
                        "intent_source": "reflex_audio",
                        "route": "defer",
                        "interface_already_said": f"Starting task {index}.",
                        "end_of_utterance": True,
                    },
                )
            )

        seen = []
        queued = None
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_QUEUED:
                queued = event
            started_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_STARTED
            ]
            if queued is not None and started_ids == ["voice-oracle-001"]:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.INTERFACE_ORACLE_CANCEL,
                session_id="voice-123",
                sequence=3,
                payload={"all": True, "reason": "stop everything"},
            )
        )
        async for event in engine.events():
            seen.append(event)
            cancelled_ids = {
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_CANCELLED
            }
            interface_cancel = next(
                (
                    item
                    for item in seen
                    if item.type == VoiceEventType.INTERFACE_ORACLE_CANCEL
                    and item.payload.get("all") is True
                ),
                None,
            )
            if cancelled_ids == {"voice-oracle-001", "voice-oracle-002"} and interface_cancel is not None:
                break

        await engine.close()
        interface_cancel = next(
            event
            for event in seen
            if event.type == VoiceEventType.INTERFACE_ORACLE_CANCEL and event.payload.get("all") is True
        )
        cancelled = [
            event
            for event in seen
            if event.type == VoiceEventType.ORACLE_JOB_CANCELLED
        ]
        assert interface_cancel.payload["cancelled_jobs"] == [
            "voice-oracle-001",
            "voice-oracle-002",
        ]
        assert {event.payload["job_id"] for event in cancelled} == {
            "voice-oracle-001",
            "voice-oracle-002",
        }
        assert all(event.payload["cancel_reason"] == "stop everything" for event in cancelled)
        assert len(oracle.requests) == 1
        assert not any(
            event.type == VoiceEventType.ORACLE_JOB_STARTED and event.payload["job_id"] == "voice-oracle-002"
            for event in seen
        )
        assert spoken
        assert all(not item.startswith("Finished") for item in spoken)

    asyncio.run(run())


def test_kame_engine_can_reprioritize_queued_async_oracle_job(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.releases = {}
            self.request_count_changed = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.request_count_changed.set()
            event = self.releases.setdefault(request.intent, asyncio.Event())
            await event.wait()
            yield f"Finished {request.intent}."

        def release(self, intent):
            self.releases.setdefault(intent, asyncio.Event()).set()

        async def wait_for_requests(self, count):
            while len(self.requests) < count:
                self.request_count_changed.clear()
                await asyncio.wait_for(self.request_count_changed.wait(), timeout=1)

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        for index in range(1, 4):
            await engine.receive_event(
                VoiceEvent(
                    type=VoiceEventType.AUDIO_INPUT_CHUNK,
                    session_id="voice-123",
                    sequence=index,
                    payload={
                        "transcript": f"run task {index}",
                        "intent": f"Run task {index}",
                        "intent_source": "reflex_audio",
                        "route": "defer",
                        "priority": "normal",
                        "interface_already_said": f"Starting task {index}.",
                        "end_of_utterance": True,
                    },
                )
            )

        seen = []
        async for event in engine.events():
            seen.append(event)
            queued_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_QUEUED
            ]
            started_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_STARTED
            ]
            if queued_ids == ["voice-oracle-002", "voice-oracle-003"] and started_ids == ["voice-oracle-001"]:
                break
        await oracle.wait_for_requests(1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.INTERFACE_ORACLE_UPDATE,
                session_id="voice-123",
                sequence=4,
                payload={
                    "job_id": "voice-oracle-003",
                    "priority": "highest",
                    "reason": "make task three highest priority",
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE:
                break

        oracle.release("Run task 1")
        async for event in engine.events():
            seen.append(event)
            if (
                event.type == VoiceEventType.ORACLE_JOB_STARTED
                and event.payload["job_id"] == "voice-oracle-003"
            ):
                break
        await oracle.wait_for_requests(2)

        await engine.close()
        update = next(event for event in seen if event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE)
        assert update.payload == {
            "job_id": "voice-oracle-003",
            "priority": "high",
            "state": "queued",
            "reason": "make task three highest priority",
            "update_count": 0,
        }
        assert [request.intent for request in oracle.requests] == ["Run task 1", "Run task 3"]
        assert not any(
            event.type == VoiceEventType.ORACLE_JOB_STARTED and event.payload["job_id"] == "voice-oracle-002"
            for event in seen
        )

    asyncio.run(run())


def test_kame_engine_attaches_update_to_queued_async_oracle_job(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.releases = {}
            self.request_count_changed = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.request_count_changed.set()
            event = self.releases.setdefault(request.intent, asyncio.Event())
            await event.wait()
            yield f"Finished {request.intent}."

        def release(self, intent):
            self.releases.setdefault(intent, asyncio.Event()).set()

        async def wait_for_requests(self, count):
            while len(self.requests) < count:
                self.request_count_changed.clear()
                await asyncio.wait_for(self.request_count_changed.wait(), timeout=1)

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        for index in range(1, 3):
            await engine.receive_event(
                VoiceEvent(
                    type=VoiceEventType.AUDIO_INPUT_CHUNK,
                    session_id="voice-123",
                    sequence=index,
                    payload={
                        "transcript": f"run task {index}",
                        "intent": f"Run task {index}",
                        "intent_source": "reflex_audio",
                        "route": "defer",
                        "interface_already_said": f"Starting task {index}.",
                        "end_of_utterance": True,
                    },
                )
            )

        seen = []
        async for event in engine.events():
            seen.append(event)
            queued_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_QUEUED
            ]
            started_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_STARTED
            ]
            if queued_ids == ["voice-oracle-002"] and started_ids == ["voice-oracle-001"]:
                break
        await oracle.wait_for_requests(1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.INTERFACE_ORACLE_UPDATE,
                session_id="voice-123",
                sequence=3,
                payload={
                    "job_id": "voice-oracle-002",
                    "update_text": "also check the Stripe receipt before answering",
                    "reason": "add receipt clarification",
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE:
                break

        oracle.release("Run task 1")
        async for event in engine.events():
            seen.append(event)
            if (
                event.type == VoiceEventType.ORACLE_JOB_STARTED
                and event.payload["job_id"] == "voice-oracle-002"
            ):
                break
        await oracle.wait_for_requests(2)

        await engine.close()
        update = next(event for event in seen if event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE)
        assert update.payload == {
            "job_id": "voice-oracle-002",
            "priority": "normal",
            "state": "queued",
            "reason": "add receipt clarification",
            "update_count": 1,
            "latest_update": "also check the Stripe receipt before answering",
        }
        assert oracle.requests[1].intent == "Run task 2"
        assert oracle.requests[1].job_updates == ("also check the Stripe receipt before answering",)
        assert oracle.requests[1].to_metadata()["kame_job_updates"] == (
            "also check the Stripe receipt before answering",
        )

    asyncio.run(run())


def test_kame_engine_attaches_interpreter_evidence_to_queued_async_oracle_job(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.releases = {}
            self.request_count_changed = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.request_count_changed.set()
            event = self.releases.setdefault(request.intent, asyncio.Event())
            await event.wait()
            yield f"Finished {request.intent}."

        def release(self, intent):
            self.releases.setdefault(intent, asyncio.Event()).set()

        async def wait_for_requests(self, count):
            while len(self.requests) < count:
                self.request_count_changed.clear()
                await asyncio.wait_for(self.request_count_changed.wait(), timeout=1)

    async def run():
        async def fake_speak(self, text, playback_generation):
            pass

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        for index in range(1, 3):
            await engine.receive_event(
                VoiceEvent(
                    type=VoiceEventType.AUDIO_INPUT_CHUNK,
                    session_id="voice-123",
                    sequence=index,
                    payload={
                        "transcript": f"run task {index}",
                        "intent": f"Run task {index}",
                        "intent_source": "reflex_audio",
                        "route": "defer",
                        "interface_already_said": f"Starting task {index}.",
                        "end_of_utterance": True,
                    },
                )
            )

        seen = []
        async for event in engine.events():
            seen.append(event)
            queued_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_QUEUED
            ]
            started_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_STARTED
            ]
            if queued_ids == ["voice-oracle-002"] and started_ids == ["voice-oracle-001"]:
                break
        await oracle.wait_for_requests(1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.INTERFACE_ORACLE_UPDATE,
                session_id="voice-123",
                sequence=3,
                payload={
                    "job_id": "voice-oracle-002",
                    "update_type": "interpreter_evidence",
                    "corrected_transcript": "what is three to the power of seventeen",
                    "normalized_intent": "answer a math question",
                    "entities": [{"type": "math_expression", "value": "3^17"}],
                    "confidence": 0.94,
                    "disagreements": ["reflex transcript omitted request prefix"],
                    "reason": "attach interpreter evidence",
                    "source": "gemma_interpreter",
                },
            )
        )
        for _ in range(8):
            event = await asyncio.wait_for(engine.events().__anext__(), timeout=1)
            seen.append(event)
            if event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE:
                break
        else:
            assert False, [(event.type.value, event.payload) for event in seen]

        oracle.release("Run task 1")
        for _ in range(16):
            event = await asyncio.wait_for(engine.events().__anext__(), timeout=1)
            seen.append(event)
            if (
                event.type == VoiceEventType.ORACLE_JOB_STARTED
                and event.payload["job_id"] == "voice-oracle-002"
            ):
                break
        else:
            assert False, [(event.type.value, event.payload) for event in seen]
        await oracle.wait_for_requests(2)

        await engine.close()
        evidence = next(
            event
            for event in seen
            if event.type == VoiceEventType.ORACLE_JOB_INTERPRETER_EVIDENCE_ATTACHED
        )
        update = next(event for event in seen if event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE)
        assert evidence.payload["job_id"] == "voice-oracle-002"
        assert evidence.payload["interpreter_evidence_late"] is False
        assert update.payload["job_id"] == "voice-oracle-002"
        assert update.payload["state"] == "queued"
        assert update.payload["interpreter_evidence_count"] == 1
        assert update.payload["interpreter_evidence_late"] is False
        assert "transcript=what is three to the power of seventeen" in update.payload[
            "latest_interpreter_evidence"
        ]
        assert oracle.requests[1].intent == "answer a math question"
        assert oracle.requests[1].intent_source == "gemma_interpreter"
        assert oracle.requests[1].oracle_text == "what is three to the power of seventeen"
        assert oracle.requests[1].oracle_text_source == "gemma_interpreter"
        assert oracle.requests[1].transcript == "what is three to the power of seventeen"
        assert oracle.requests[1].transcript_source == "gemma_interpreter"
        assert oracle.requests[1].transcript_confidence == 0.94
        assert any("entities=math_expression=3^17" in update for update in oracle.requests[1].job_updates)

    asyncio.run(run())


def test_kame_engine_attaches_update_to_running_async_oracle_job(monkeypatch):
    class RunningUpdateOracle:
        def __init__(self):
            self.requests = []
            self.updates = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.update_seen = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await self.release.wait()
            yield f"Finished {request.intent}."

        async def update_request(self, request, update_text, metadata):
            self.updates.append((request, update_text, metadata))
            self.update_seen.set()

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = RunningUpdateOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "run task one",
                    "intent": "Run task one",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Starting task one.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_STARTED:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.INTERFACE_ORACLE_UPDATE,
                session_id="voice-123",
                sequence=2,
                payload={
                    "job_id": "voice-oracle-001",
                    "update_text": "also check the Stripe receipt before answering",
                    "reason": "add running receipt clarification",
                },
            )
        )
        await asyncio.wait_for(oracle.update_seen.wait(), timeout=1)
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE:
                break

        oracle.release.set()
        await engine.close()

        update = next(event for event in seen if event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE)
        assert update.payload == {
            "job_id": "voice-oracle-001",
            "priority": "normal",
            "state": "running",
            "reason": "add running receipt clarification",
            "update_count": 1,
            "latest_update": "also check the Stripe receipt before answering",
        }
        updated_request, update_text, metadata = oracle.updates[0]
        assert updated_request.intent == "Run task one"
        assert updated_request.job_updates == ("also check the Stripe receipt before answering",)
        assert update_text == "also check the Stripe receipt before answering"
        assert metadata == {
            "job_id": "voice-oracle-001",
            "state": "running",
            "reason": "add running receipt clarification",
            "update_count": 1,
            "latest_update": "also check the Stripe receipt before answering",
        }
        assert "Starting task one." in spoken

    asyncio.run(run())


def test_kame_engine_attaches_interpreter_evidence_to_running_async_oracle_job(monkeypatch):
    class RunningEvidenceOracle:
        def __init__(self):
            self.requests = []
            self.updates = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.update_seen = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await self.release.wait()
            yield f"Finished {request.intent}."

        async def update_request(self, request, update_text, metadata):
            self.updates.append((request, update_text, metadata))
            self.update_seen.set()

    async def run():
        async def fake_speak(self, text, playback_generation):
            pass

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = RunningEvidenceOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "run task one",
                    "intent": "Run task one",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Starting task one.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_STARTED:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.INTERFACE_ORACLE_UPDATE,
                session_id="voice-123",
                sequence=2,
                payload={
                    "job_id": "voice-oracle-001",
                    "update_type": "interpreter_evidence",
                    "corrected_transcript": "check the current deployment logs",
                    "normalized_intent": "inspect deployment logs",
                    "confidence": 0.81,
                    "reason": "attach late interpreter evidence",
                    "source": "gemma_interpreter",
                },
            )
        )
        await asyncio.wait_for(oracle.update_seen.wait(), timeout=1)
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE:
                break

        oracle.release.set()
        await engine.close()

        evidence = next(
            event
            for event in seen
            if event.type == VoiceEventType.ORACLE_JOB_INTERPRETER_EVIDENCE_LATE
        )
        update = next(event for event in seen if event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE)
        assert evidence.payload["job_id"] == "voice-oracle-001"
        assert evidence.payload["interpreter_evidence_late"] is True
        assert update.payload["state"] == "running"
        assert update.payload["interpreter_evidence_late"] is True

        updated_request, update_text, metadata = oracle.updates[0]
        assert updated_request.intent == "Run task one"
        assert updated_request.oracle_text == "run task one"
        assert updated_request.oracle_text_source == "reflex_audio"
        assert any("intent=inspect deployment logs" in update for update in updated_request.job_updates)
        assert "transcript=check the current deployment logs" in update_text
        assert metadata["latest_interpreter_evidence"] == update_text
        assert metadata["interpreter_evidence_count"] == 1
        assert metadata["interpreter_evidence_late"] is True

    asyncio.run(run())


def test_kame_engine_spoken_stop_everything_cancels_all_async_oracle_jobs(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await asyncio.Event().wait()
            yield f"Finished {request.intent}."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        for index in range(1, 3):
            await engine.receive_event(
                VoiceEvent(
                    type=VoiceEventType.AUDIO_INPUT_CHUNK,
                    session_id="voice-123",
                    sequence=index,
                    payload={
                        "transcript": f"run task {index}",
                        "intent": f"Run task {index}",
                        "intent_source": "reflex_audio",
                        "route": "defer",
                        "interface_already_said": f"Starting task {index}.",
                        "end_of_utterance": True,
                    },
                )
            )

        seen = []
        async for event in engine.events():
            seen.append(event)
            queued_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_QUEUED
            ]
            started_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_STARTED
            ]
            if queued_ids == ["voice-oracle-002"] and started_ids == ["voice-oracle-001"]:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=3,
                payload={
                    "transcript": "stop everything",
                    "intent": "Stop everything.",
                    "intent_source": "reflex_audio",
                    "route": "local",
                    "local_reply": "Stopping everything.",
                    "end_of_utterance": True,
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            cancelled_ids = {
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_CANCELLED
            }
            control_commit = next(
                (
                    item
                    for item in seen
                    if item.type == VoiceEventType.ASSISTANT_COMMIT
                    and item.payload.get("oracle_job_control")
                ),
                None,
            )
            if cancelled_ids == {"voice-oracle-001", "voice-oracle-002"} and control_commit is not None:
                break

        await engine.close()
        interface_cancel = next(
            event
            for event in seen
            if event.type == VoiceEventType.INTERFACE_ORACLE_CANCEL and event.payload.get("all") is True
        )
        playback_stop = next(
            event
            for event in seen
            if event.type == VoiceEventType.BARGE_IN and event.payload.get("oracle_job_control") is True
        )
        assert playback_stop.payload["reason"] == "spoken request to stop everything"
        assert playback_stop.payload["frontend_cancel_requested"] is False
        assert playback_stop.payload["playback_generation"] > playback_stop.payload["cancelled_playback_generation"]
        assert interface_cancel.payload["spoken_control"] is True
        assert interface_cancel.payload["cancelled_jobs"] == [
            "voice-oracle-001",
            "voice-oracle-002",
        ]
        assert len(oracle.requests) == 1
        assert "I cancelled all current oracle jobs." in spoken

    asyncio.run(run())


def test_kame_engine_spoken_cancel_second_job_only(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await asyncio.Event().wait()
            yield f"Finished {request.intent}."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        for index in range(1, 3):
            await engine.receive_event(
                VoiceEvent(
                    type=VoiceEventType.AUDIO_INPUT_CHUNK,
                    session_id="voice-123",
                    sequence=index,
                    payload={
                        "transcript": f"run task {index}",
                        "intent": f"Run task {index}",
                        "intent_source": "reflex_audio",
                        "route": "defer",
                        "interface_already_said": f"Starting task {index}.",
                        "end_of_utterance": True,
                    },
                )
            )

        seen = []
        async for event in engine.events():
            seen.append(event)
            queued_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_QUEUED
            ]
            started_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_STARTED
            ]
            if queued_ids == ["voice-oracle-002"] and started_ids == ["voice-oracle-001"]:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=3,
                payload={
                    "transcript": "cancel the second one",
                    "intent": "Cancel the second one.",
                    "intent_source": "reflex_audio",
                    "route": "local",
                    "local_reply": "Cancelling the second one.",
                    "end_of_utterance": True,
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            control_cancel = next(
                (
                    item
                    for item in seen
                    if item.type == VoiceEventType.INTERFACE_ORACLE_CANCEL
                    and item.payload.get("spoken_control")
                ),
                None,
            )
            control_commit = next(
                (
                    item
                    for item in seen
                    if item.type == VoiceEventType.ASSISTANT_COMMIT
                    and item.payload.get("oracle_job_control")
                ),
                None,
            )
            if control_cancel is not None and control_commit is not None:
                break

        await engine.close()
        control_cancel = next(
            event
            for event in seen
            if event.type == VoiceEventType.INTERFACE_ORACLE_CANCEL and event.payload.get("spoken_control")
        )
        cancelled = [
            event
            for event in seen
            if event.type == VoiceEventType.ORACLE_JOB_CANCELLED
            and event.payload.get("job_id") == "voice-oracle-002"
        ]
        assert control_cancel.payload == {
            "job_id": "voice-oracle-002",
            "reason": "spoken request to cancel oracle job",
            "spoken_control": True,
        }
        assert len(cancelled) == 1
        assert cancelled[0].payload["cancel_reason"] == "spoken request to cancel oracle job"
        assert len(oracle.requests) == 1
        assert oracle.requests[0].intent == "Run task 1"
        assert "I cancelled Starting task 2. Run task 2." in spoken

    asyncio.run(run())


def test_kame_engine_spoken_cancel_matches_descriptive_job_intent(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await asyncio.Event().wait()
            yield f"Finished {request.intent}."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        for sequence, payload in enumerate(
            (
                {
                    "transcript": "check provisioning logs",
                    "intent": "Check provisioning logs",
                    "interface_already_said": "Starting log check.",
                },
                {
                    "transcript": "draft the procurement memo",
                    "intent": "Draft procurement memo",
                    "interface_already_said": "Starting memo draft.",
                },
            ),
            start=1,
        ):
            await engine.receive_event(
                VoiceEvent(
                    type=VoiceEventType.AUDIO_INPUT_CHUNK,
                    session_id="voice-123",
                    sequence=sequence,
                    payload={
                        **payload,
                        "intent_source": "reflex_audio",
                        "route": "defer",
                        "end_of_utterance": True,
                    },
                )
            )

        seen = []
        async for event in engine.events():
            seen.append(event)
            queued_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_QUEUED
            ]
            started_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_STARTED
            ]
            if queued_ids == ["voice-oracle-002"] and started_ids == ["voice-oracle-001"]:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=3,
                payload={
                    "transcript": "cancel that log check",
                    "intent": "Cancel that log check.",
                    "intent_source": "reflex_audio",
                    "route": "local",
                    "local_reply": "Cancelling the log check.",
                    "end_of_utterance": True,
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            control_cancel = next(
                (
                    item
                    for item in seen
                    if item.type == VoiceEventType.INTERFACE_ORACLE_CANCEL
                    and item.payload.get("spoken_control")
                ),
                None,
            )
            cancelled = next(
                (
                    item
                    for item in seen
                    if item.type == VoiceEventType.ORACLE_JOB_CANCELLED
                    and item.payload.get("job_id") == "voice-oracle-001"
                ),
                None,
            )
            if control_cancel is not None and cancelled is not None:
                break

        await engine.close()
        control_cancel = next(
            event
            for event in seen
            if event.type == VoiceEventType.INTERFACE_ORACLE_CANCEL and event.payload.get("spoken_control")
        )
        assert control_cancel.payload == {
            "job_id": "voice-oracle-001",
            "reason": "spoken request to cancel oracle job",
            "spoken_control": True,
        }
        assert not any(
            event.type == VoiceEventType.ORACLE_JOB_CANCELLED
            and event.payload.get("job_id") == "voice-oracle-002"
            for event in seen
        )
        assert len(oracle.requests) >= 1
        assert oracle.requests[0].intent == "Check provisioning logs"
        assert "I cancelled Starting log check. Check provisioning logs." in spoken

    asyncio.run(run())


def test_kame_oracle_job_control_uses_latest_job_for_pronoun_cancel_and_priority():
    status = {
        "jobs": [
            {
                "job_id": "voice-oracle-001",
                "state": "running",
                "spoken_status": "Checking provisioning logs.",
            },
            {
                "job_id": "voice-oracle-002",
                "state": "queued",
                "spoken_status": "Drafting procurement memo.",
            },
        ]
    }
    cancel_request = KameOracleRequest(
        session_id="voice-123",
        turn_id="voice-123:3",
        source="discord_voice",
        user_id="42",
        intent="Cancel it.",
        route=KameRoute.LOCAL,
        local_reply="Cancelling it.",
    )
    priority_request = KameOracleRequest(
        session_id="voice-123",
        turn_id="voice-123:4",
        source="discord_voice",
        user_id="42",
        intent="Make it highest priority.",
        route=KameRoute.LOCAL,
        local_reply="Making it highest priority.",
    )

    cancel = _kame_oracle_job_control_operation(cancel_request, status)
    priority = _kame_oracle_job_control_operation(priority_request, status)

    assert cancel == {
        "kind": "cancel",
        "job_id": "voice-oracle-002",
        "reason": "spoken request to cancel oracle job",
    }
    assert priority == {
        "kind": "priority",
        "job_id": "voice-oracle-002",
        "priority": "high",
        "reason": "spoken request to set high priority",
    }


def test_kame_engine_reports_async_oracle_reject_policy_without_sync_fallback(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await asyncio.Event().wait()
            yield f"Finished {request.intent}."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={
                    "enabled": True,
                    "max_concurrent": 1,
                    "queue_limit": 16,
                    "overflow_policy": "reject",
                },
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "run task one",
                    "intent": "Run task one.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Starting task one.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_STARTED:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload={
                    "transcript": "run task two",
                    "intent": "Run task two.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Starting task two.",
                    "end_of_utterance": True,
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            queue_error = next(
                (
                    item
                    for item in seen
                    if item.type == VoiceEventType.ORACLE_ERROR
                    and item.payload.get("reason") == "oracle_job_queue_full"
                ),
                None,
            )
            capacity_partial = next(
                (
                    item
                    for item in seen
                    if item.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL
                    and item.payload.get("oracle_job_queue_full")
                ),
                None,
            )
            if queue_error is not None and capacity_partial is not None:
                break

        await engine.close()
        queue_error = next(
            event
            for event in seen
            if event.type == VoiceEventType.ORACLE_ERROR
            and event.payload.get("reason") == "oracle_job_queue_full"
        )
        capacity_partial = next(
            event
            for event in seen
            if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL
            and event.payload.get("oracle_job_queue_full")
        )
        assert queue_error.payload["error"] == "oracle job queue is full"
        assert capacity_partial.payload["text"] == "I am at oracle job capacity right now."
        assert spoken[-1] == "I am at oracle job capacity right now."
        assert len(oracle.requests) == 1
        assert oracle.requests[0].intent == "Run task one."

    asyncio.run(run())


def test_kame_engine_reports_async_oracle_reprioritize_policy_without_sync_fallback(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await asyncio.Event().wait()
            yield f"Finished {request.intent}."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={
                    "enabled": True,
                    "max_concurrent": 1,
                    "queue_limit": 16,
                    "overflow_policy": "reprioritize",
                },
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "run task one",
                    "intent": "Run task one.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Starting task one.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_STARTED:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload={
                    "transcript": "run task two",
                    "intent": "Run task two.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Starting task two.",
                    "end_of_utterance": True,
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            reprioritize_error = next(
                (
                    item
                    for item in seen
                    if item.type == VoiceEventType.ORACLE_ERROR
                    and item.payload.get("reason") == "oracle_job_reprioritization_required"
                ),
                None,
            )
            reprioritize_partial = next(
                (
                    item
                    for item in seen
                    if item.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL
                    and item.payload.get("oracle_job_reprioritization_required")
                ),
                None,
            )
            if reprioritize_error is not None and reprioritize_partial is not None:
                break

        await engine.close()
        reprioritize_error = next(
            event
            for event in seen
            if event.type == VoiceEventType.ORACLE_ERROR
            and event.payload.get("reason") == "oracle_job_reprioritization_required"
        )
        reprioritize_partial = next(
            event
            for event in seen
            if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL
            and event.payload.get("oracle_job_reprioritization_required")
        )
        assert reprioritize_error.payload["error"] == "oracle job reprioritization required"
        assert (
            reprioritize_partial.payload["text"]
            == "I am at oracle job capacity. Tell me which job to prioritize or cancel."
        )
        assert spoken[-1] == "I am at oracle job capacity. Tell me which job to prioritize or cancel."
        assert len(oracle.requests) == 1
        assert oracle.requests[0].intent == "Run task one."

    asyncio.run(run())


def test_kame_engine_spoken_priority_control_reprioritizes_queued_job(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.releases = {}
            self.request_count_changed = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.request_count_changed.set()
            event = self.releases.setdefault(request.intent, asyncio.Event())
            await event.wait()
            yield f"Finished {request.intent}."

        def release(self, intent):
            self.releases.setdefault(intent, asyncio.Event()).set()

        async def wait_for_requests(self, count):
            while len(self.requests) < count:
                self.request_count_changed.clear()
                await asyncio.wait_for(self.request_count_changed.wait(), timeout=1)

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        for index in range(1, 4):
            await engine.receive_event(
                VoiceEvent(
                    type=VoiceEventType.AUDIO_INPUT_CHUNK,
                    session_id="voice-123",
                    sequence=index,
                    payload={
                        "transcript": f"run task {index}",
                        "intent": f"Run task {index}",
                        "intent_source": "reflex_audio",
                        "route": "defer",
                        "priority": "normal",
                        "interface_already_said": f"Starting task {index}.",
                        "end_of_utterance": True,
                    },
                )
            )

        seen = []
        async for event in engine.events():
            seen.append(event)
            queued_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_QUEUED
            ]
            started_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_STARTED
            ]
            if queued_ids == ["voice-oracle-002", "voice-oracle-003"] and started_ids == ["voice-oracle-001"]:
                break
        await oracle.wait_for_requests(1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=4,
                payload={
                    "transcript": "make task three highest priority",
                    "intent": "Make task three highest priority.",
                    "intent_source": "reflex_audio",
                    "route": "local",
                    "local_reply": "Task three is highest priority.",
                    "end_of_utterance": True,
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE and event.payload.get("spoken_control"):
                break

        oracle.release("Run task 1")
        async for event in engine.events():
            seen.append(event)
            if (
                event.type == VoiceEventType.ORACLE_JOB_STARTED
                and event.payload["job_id"] == "voice-oracle-003"
            ):
                break
        await oracle.wait_for_requests(2)

        await engine.close()
        update = next(
            event
            for event in seen
            if event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE and event.payload.get("spoken_control")
        )
        assert update.payload["job_id"] == "voice-oracle-003"
        assert update.payload["priority"] == "high"
        assert update.payload["spoken_control"] is True
        assert [request.intent for request in oracle.requests] == ["Run task 1", "Run task 3"]
        assert "I set high priority for Starting task 3. Run task 3." in spoken

    asyncio.run(run())


def test_kame_engine_spoken_update_attaches_to_latest_async_oracle_job(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.releases = {}
            self.request_count_changed = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.request_count_changed.set()
            event = self.releases.setdefault(request.intent, asyncio.Event())
            await event.wait()
            yield f"Finished {request.intent}."

        def release(self, intent):
            self.releases.setdefault(intent, asyncio.Event()).set()

        async def wait_for_requests(self, count):
            while len(self.requests) < count:
                self.request_count_changed.clear()
                await asyncio.wait_for(self.request_count_changed.wait(), timeout=1)

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        for index in range(1, 3):
            await engine.receive_event(
                VoiceEvent(
                    type=VoiceEventType.AUDIO_INPUT_CHUNK,
                    session_id="voice-123",
                    sequence=index,
                    payload={
                        "transcript": f"run task {index}",
                        "intent": f"Run task {index}",
                        "intent_source": "reflex_audio",
                        "route": "defer",
                        "interface_already_said": f"Starting task {index}.",
                        "end_of_utterance": True,
                    },
                )
            )

        seen = []
        async for event in engine.events():
            seen.append(event)
            queued_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_QUEUED
            ]
            started_ids = [
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_STARTED
            ]
            if queued_ids == ["voice-oracle-002"] and started_ids == ["voice-oracle-001"]:
                break
        await oracle.wait_for_requests(1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=3,
                payload={
                    "transcript": "also check the Stripe receipt before answering",
                    "intent": "Also check the Stripe receipt before answering.",
                    "intent_source": "reflex_audio",
                    "route": "local",
                    "local_reply": "I'll add that.",
                    "end_of_utterance": True,
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE and event.payload.get("spoken_control"):
                break

        oracle.release("Run task 1")
        async for event in engine.events():
            seen.append(event)
            if (
                event.type == VoiceEventType.ORACLE_JOB_STARTED
                and event.payload["job_id"] == "voice-oracle-002"
            ):
                break
        await oracle.wait_for_requests(2)

        await engine.close()
        update = next(
            event
            for event in seen
            if event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE and event.payload.get("spoken_control")
        )
        assert update.payload["job_id"] == "voice-oracle-002"
        assert update.payload["update_count"] == 1
        assert update.payload["latest_update"] == "check the stripe receipt before answering"
        assert oracle.requests[1].job_updates == ("check the stripe receipt before answering",)
        assert "I added that to Starting task 2. Run task 2." in spoken

    asyncio.run(run())


def test_kame_engine_barge_in_during_async_ack_does_not_interrupt_oracle_job(monkeypatch):
    class InterruptibleBlockingOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.interrupted = False

        def interrupt(self, message: str = ""):
            self.interrupted = True

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await self.release.wait()
            yield "The deployment is healthy."

    async def run():
        speak_started = asyncio.Event()
        release_speak = asyncio.Event()

        async def fake_speak(self, text, playback_generation):
            speak_started.set()
            await release_speak.wait()

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = InterruptibleBlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "check the deployment status",
                    "intent": "Check the deployment status.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Checking that now.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_STARTED:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)
        await asyncio.wait_for(speak_started.wait(), timeout=1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.BARGE_IN,
                session_id="voice-123",
                sequence=2,
                payload={"reason": "user_speech"},
            )
        )
        assert oracle.interrupted is False

        oracle.release.set()
        release_speak.set()
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_COMPLETED:
                break

        await engine.close()
        completed = next(event for event in seen if event.type == VoiceEventType.ORACLE_JOB_COMPLETED)
        assert completed.payload["result_summary"] == "The deployment is healthy."

    asyncio.run(run())


def test_kame_engine_spoken_stop_talking_does_not_cancel_async_oracle_job(monkeypatch):
    class InterruptibleBlockingOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.interrupted = False

        def interrupt(self, message: str = ""):
            self.interrupted = True

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await self.release.wait()
            yield "The deployment is healthy."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = InterruptibleBlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "check the deployment status",
                    "intent": "Check the deployment status.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Checking that now.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_STARTED:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload={
                    "transcript": "stop talking",
                    "intent": "Stop talking.",
                    "route": "local",
                    "local_reply": "Okay.",
                    "intent_source": "reflex_audio",
                    "end_of_utterance": True,
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT and event.payload.get("local_reply"):
                break

        assert oracle.interrupted is False
        assert not any(event.type == VoiceEventType.INTERFACE_ORACLE_CANCEL for event in seen)
        assert not any(event.type == VoiceEventType.ORACLE_JOB_CANCELLED for event in seen)

        oracle.release.set()
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_COMPLETED:
                break

        await engine.close()
        completed = next(event for event in seen if event.type == VoiceEventType.ORACLE_JOB_COMPLETED)
        assert completed.payload["result_summary"] == "The deployment is healthy."
        assert len(oracle.requests) == 1
        assert spoken[:2] == ["Checking that now.", "Okay."]

    asyncio.run(run())


def test_kame_engine_barge_in_during_async_result_speech_does_not_interrupt_completed_job(monkeypatch):
    class InterruptibleBlockingOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.interrupted = False

        def interrupt(self, message: str = ""):
            self.interrupted = True

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await self.release.wait()
            yield "The deployment is healthy."

    async def run():
        spoken = []
        result_speech_started = asyncio.Event()
        result_speech_cancelled = asyncio.Event()
        release_result_speech = asyncio.Event()

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)
            if text == "The deployment is healthy.":
                result_speech_started.set()
                try:
                    await release_result_speech.wait()
                except asyncio.CancelledError:
                    result_speech_cancelled.set()
                    raise

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = InterruptibleBlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "check the deployment status",
                    "intent": "Check the deployment status.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Checking that now.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_STARTED:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)

        oracle.release.set()
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL and event.payload.get("oracle_job_result"):
                break
        await asyncio.wait_for(result_speech_started.wait(), timeout=1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.BARGE_IN,
                session_id="voice-123",
                sequence=2,
                payload={"reason": "user_speech"},
            )
        )
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.BARGE_IN:
                break

        release_result_speech.set()
        await engine.close()
        completed = next(event for event in seen if event.type == VoiceEventType.ORACLE_JOB_COMPLETED)
        barge_in = next(event for event in seen if event.type == VoiceEventType.BARGE_IN)
        assert completed.payload["result_summary"] == "The deployment is healthy."
        assert barge_in.payload["backend_interrupt_requested"] is True
        assert oracle.interrupted is False
        assert result_speech_cancelled.is_set()
        assert not any(event.type == VoiceEventType.INTERFACE_ORACLE_CANCEL for event in seen)
        assert not any(event.type == VoiceEventType.ORACLE_JOB_CANCELLED for event in seen)

    asyncio.run(run())


def test_kame_engine_async_terminal_result_speech_is_capped_without_losing_full_result(monkeypatch):
    class VerboseOracle:
        def __init__(self):
            self.requests = []

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            yield "First sentence. "
            yield "Second sentence. "
            yield "Third sentence."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = VerboseOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                max_spoken_sentences=1,
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "explain the plan",
                    "intent": "Explain the plan.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Working on that.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT and event.payload.get("oracle_job_result"):
                break

        await engine.close()
        completed = next(event for event in seen if event.type == VoiceEventType.ORACLE_JOB_COMPLETED)
        result_commit = next(
            event
            for event in seen
            if event.type == VoiceEventType.ASSISTANT_COMMIT and event.payload.get("oracle_job_result")
        )
        assert completed.payload["result_summary"] == "First sentence. Second sentence. Third sentence."
        assert result_commit.payload["text"] == "First sentence."
        assert result_commit.payload["voice_response_truncated"] is True
        assert result_commit.payload["max_spoken_sentences"] == 1
        assert spoken == ["Working on that.", "First sentence."]

    asyncio.run(run())


def test_kame_engine_async_oracle_job_failure_reports_in_voice(monkeypatch):
    class FailingOracle:
        def __init__(self):
            self.requests = []

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            raise RuntimeError(
                "oracle backend unavailable Bearer raw-token token=sk-secret "
                "at https://user:pass@voice.local/v1?api_key=raw"
            )
            yield ""

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = FailingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "check the deployment status",
                    "intent": "Check the deployment status.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Checking that now.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT and event.payload.get("oracle_job_failed"):
                break

        await engine.close()
        failed = next(event for event in seen if event.type == VoiceEventType.ORACLE_JOB_FAILED)
        commit = next(
            event
            for event in seen
            if event.type == VoiceEventType.ASSISTANT_COMMIT and event.payload.get("oracle_job_failed")
        )
        assert failed.payload["job_id"] == "voice-oracle-001"
        assert failed.payload["state"] == "failed"
        assert failed.payload["error"] == (
            "oracle backend unavailable Bearer *** token=*** at https://***@voice.local/v1"
        )
        assert commit.payload["oracle_job_id"] == "voice-oracle-001"
        assert commit.payload["text"] == (
            "I couldn't finish Check the deployment status: "
            "oracle backend unavailable Bearer *** token=*** at https://***@voice.local/v1"
        )
        assert spoken == [
            "Checking that now.",
            (
                "I couldn't finish Check the deployment status: "
                "oracle backend unavailable Bearer *** token=*** at https://***@voice.local/v1"
            ),
        ]
        combined = json.dumps(
            {"failed": failed.payload, "commit": commit.payload, "spoken": spoken},
            sort_keys=True,
        )
        assert "raw-token" not in combined
        assert "sk-secret" not in combined
        assert "user:pass" not in combined
        assert "api_key=raw" not in combined

    asyncio.run(run())


def test_kame_engine_status_recalls_recent_completed_async_oracle_job(monkeypatch):
    class CompletingOracle:
        def __init__(self):
            self.requests = []

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            yield "The deployment is healthy."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = CompletingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "check the deployment status",
                    "intent": "Check the deployment status.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Checking that now.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT and event.payload.get("oracle_job_result"):
                break

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload={
                    "transcript": "what happened with the last job",
                    "intent": "What happened with the last job?",
                    "route": "local",
                    "local_reply": "Let me check.",
                    "intent_source": "reflex_audio",
                    "end_of_utterance": True,
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            if (
                event.type == VoiceEventType.ASSISTANT_COMMIT
                and str(event.payload.get("text") or "").startswith("No oracle jobs are running")
            ):
                break

        await engine.close()
        status_commit = next(
            event
            for event in seen
            if event.type == VoiceEventType.ASSISTANT_COMMIT
            and str(event.payload.get("text") or "").startswith("No oracle jobs are running")
        )
        assert status_commit.payload["local_reply"] is True
        assert status_commit.payload["text"] == (
            "No oracle jobs are running or queued right now. Recent: completed: The deployment is healthy."
        )
        assert spoken[-1] == status_commit.payload["text"]

    asyncio.run(run())


def test_kame_engine_speak_terminal_results_false_suppresses_result_speech_but_keeps_status(monkeypatch):
    class CompletingOracle:
        def __init__(self):
            self.requests = []

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            yield "The deployment is healthy."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = CompletingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={
                    "enabled": True,
                    "max_concurrent": 1,
                    "queue_limit": 4,
                    "speak_terminal_results": False,
                },
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "check the deployment status",
                    "intent": "Check the deployment status.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Checking that now.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_COMPLETED:
                break
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_RESULT_SUPPRESSED:
                break

        assert len(oracle.requests) == 1
        completed = next(event for event in seen if event.type == VoiceEventType.ORACLE_JOB_COMPLETED)
        suppressed = next(event for event in seen if event.type == VoiceEventType.ORACLE_JOB_RESULT_SUPPRESSED)
        assert completed.payload["result_summary"] == "The deployment is healthy."
        assert suppressed.payload["suppression_reason"] == "terminal_speech_disabled"
        assert suppressed.payload["result_suppressed"] is True
        assert suppressed.payload["suppressed_result_present"] is True
        assert "result_summary" not in suppressed.payload
        assert "result_text" not in suppressed.payload
        assert not any(
            event.type == VoiceEventType.ASSISTANT_COMMIT
            and event.payload.get("oracle_job_result")
            for event in seen
        )
        assert not any(
            event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL
            and event.payload.get("oracle_job_result")
            for event in seen
        )
        assert spoken == ["Checking that now."]

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload={
                    "transcript": "what happened with the last job",
                    "intent": "What happened with the last job?",
                    "route": "local",
                    "local_reply": "Let me check.",
                    "intent_source": "reflex_audio",
                    "end_of_utterance": True,
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            if (
                event.type == VoiceEventType.ASSISTANT_COMMIT
                and str(event.payload.get("text") or "").startswith("No oracle jobs are running")
            ):
                break

        await engine.close()
        status_commit = next(
            event
            for event in seen
            if event.type == VoiceEventType.ASSISTANT_COMMIT
            and str(event.payload.get("text") or "").startswith("No oracle jobs are running")
        )
        assert status_commit.payload["local_reply"] is True
        assert status_commit.payload["text"] == (
            "No oracle jobs are running or queued right now. Recent: completed: The deployment is healthy."
        )
        assert spoken == [
            "Checking that now.",
            "No oracle jobs are running or queued right now. Recent: completed: The deployment is healthy.",
        ]

    asyncio.run(run())


def test_async_oracle_job_enters_waiting_for_approval_on_tool_call(monkeypatch):
    class ApprovalOracle:
        def __init__(self):
            self.release = asyncio.Event()

        async def stream_answer_for_request(self, request):
            yield {
                "event": "tool_call",
                "tool_name": "stripe_link_purchase",
                "tool_call_id": "call-approve-1",
                "approval_required": True,
                "approval_id": "approval-123",
                "approval_reason": "Stripe Link spend requires approval",
                "arguments": {"amount": 200, "card": "secret"},
            }
            await self.release.wait()
            yield {
                "event": "tool_result",
                "tool_name": "stripe_link_purchase",
                "tool_call_id": "call-approve-1",
                "approval_id": "approval-123",
                "result": {"approved": True},
            }
            yield "The spend approval cleared."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = ApprovalOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "buy service credits",
                    "intent": "Buy service credits.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Preparing the spend request.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if (
                any(seen_event.type == VoiceEventType.ORACLE_JOB_WAITING_FOR_APPROVAL for seen_event in seen)
                and any(
                    seen_event.type == VoiceEventType.ORACLE_JOB_PROGRESS
                    and seen_event.payload.get("phase") == "tool"
                    and seen_event.payload.get("tool_event", {}).get("approval_required") is True
                    for seen_event in seen
                )
            ):
                break

        waiting = next(event for event in seen if event.type == VoiceEventType.ORACLE_JOB_WAITING_FOR_APPROVAL)
        tool_call_progress = next(
            event
            for event in seen
            if event.type == VoiceEventType.ORACLE_JOB_PROGRESS
            and event.payload.get("phase") == "tool"
            and event.payload.get("tool_event", {}).get("approval_required") is True
        )
        assert waiting.payload["job_id"] == "voice-oracle-001"
        assert waiting.payload["state"] == "waiting_for_approval"
        assert waiting.payload["approval_reason"] == "Stripe Link spend requires approval"
        assert waiting.payload["approval"]["approval_id"] == "approval-123"
        assert waiting.payload["approval"]["tool_name"] == "stripe_link_purchase"
        assert "secret" not in str(waiting.payload)
        assert tool_call_progress.payload["tool_event"]["approval_required"] is True
        assert "arguments" not in tool_call_progress.payload["tool_event"]
        assert "secret" not in str(tool_call_progress.payload)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload={
                    "transcript": "what are you working on",
                    "intent": "What are you working on?",
                    "intent_source": "reflex_audio",
                    "route": "local",
                    "local_reply": "Checking status.",
                    "end_of_utterance": True,
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break
        status_commit = next(
            event
            for event in reversed(seen)
            if event.type == VoiceEventType.ASSISTANT_COMMIT
        )
        assert "1 waiting for approval" in status_commit.payload["text"]
        assert "1 active out of 1" in status_commit.payload["text"]
        assert "0 running out of 1" not in status_commit.payload["text"]
        assert "waiting_for_approval: Preparing the spend request." in status_commit.payload["text"]

        oracle.release.set()
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_COMPLETED:
                break

        await engine.close()
        completed = next(event for event in seen if event.type == VoiceEventType.ORACLE_JOB_COMPLETED)
        assert completed.payload["result_summary"] == "The spend approval cleared."
        assert completed.payload["source_playback_generation"] == 1
        assert completed.payload["playback_generation"] == 2
        assert not any(event.payload.get("oracle_job_result") for event in seen)
        assert spoken[-1] == status_commit.payload["text"]

    asyncio.run(run())


def test_async_oracle_tool_approval_carries_late_interpreter_evidence(monkeypatch):
    class ApprovalAfterEvidenceOracle:
        def __init__(self):
            self.requests = []
            self.updates = []
            self.started = asyncio.Event()
            self.update_seen = asyncio.Event()
            self.release_tool_call = asyncio.Event()
            self.release_result = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await self.release_tool_call.wait()
            yield {
                "event": "tool_call",
                "tool_name": "stripe_link_purchase",
                "tool_call_id": "call-approve-1",
                "approval_required": True,
                "approval_id": "approval-123",
                "approval_reason": "Stripe Link spend requires approval",
                "arguments": {"amount": 20, "card": "secret-card"},
            }
            await self.release_result.wait()
            yield "The spend approval cleared."

        async def update_request(self, request, update_text, metadata):
            self.updates.append((request, update_text, metadata))
            self.update_seen.set()

    async def run():
        async def fake_speak(self, text, playback_generation):
            pass

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = ApprovalAfterEvidenceOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "buy phone credits",
                    "intent": "Buy phone credits.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Preparing the spend request.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_STARTED:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.INTERFACE_ORACLE_UPDATE,
                session_id="voice-123",
                sequence=2,
                payload={
                    "job_id": "voice-oracle-001",
                    "update_type": "interpreter_evidence",
                    "corrected_transcript": "buy twenty dollars of phone credits",
                    "normalized_intent": "prepare a Stripe approval for phone credits",
                    "confidence": 0.86,
                    "reason": "attach late interpreter evidence before spend approval",
                    "source": "gemma_interpreter",
                    "disagreements": ["reflex transcript omitted budget amount"],
                },
            )
        )
        await asyncio.wait_for(oracle.update_seen.wait(), timeout=1)

        oracle.release_tool_call.set()
        async for event in engine.events():
            seen.append(event)
            if (
                any(seen_event.type == VoiceEventType.ORACLE_JOB_WAITING_FOR_APPROVAL for seen_event in seen)
                and any(
                    seen_event.type == VoiceEventType.ORACLE_JOB_PROGRESS
                    and seen_event.payload.get("phase") == "tool"
                    and seen_event.payload.get("tool_event", {}).get("approval_required") is True
                    for seen_event in seen
                )
            ):
                break

        waiting = next(event for event in seen if event.type == VoiceEventType.ORACLE_JOB_WAITING_FOR_APPROVAL)
        tool_call_progress = next(
            event
            for event in seen
            if event.type == VoiceEventType.ORACLE_JOB_PROGRESS
            and event.payload.get("phase") == "tool"
            and event.payload.get("tool_event", {}).get("approval_required") is True
        )

        approval = waiting.payload["approval"]
        tool_event = tool_call_progress.payload["tool_event"]
        for payload in (approval, tool_event):
            assert payload["interpreter_evidence_count"] == 1
            assert payload["interpreter_evidence_late"] is True
            assert payload["latest_interpreter_evidence_source"] == "gemma_interpreter"
            assert "transcript=buy twenty dollars of phone credits" in payload["latest_interpreter_evidence"]
            assert "intent=prepare a Stripe approval for phone credits" in payload["latest_interpreter_evidence"]
        assert "arguments" not in tool_event
        assert "secret-card" not in str(waiting.payload)
        assert "secret-card" not in str(tool_call_progress.payload)

        oracle.release_result.set()
        await engine.close()

    asyncio.run(run())


def test_kame_engine_interface_cancel_stops_one_async_oracle_job(monkeypatch):
    class InterruptibleBlockingOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()
            self.interrupted = False
            self.interrupted_requests = []

        def interrupt_request(self, request, message: str = ""):
            self.interrupted_requests.append((request.turn_id, request.oracle_text, message))

        def interrupt(self, message: str = ""):
            self.interrupted = True

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await asyncio.Event().wait()
            yield "late result"

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = InterruptibleBlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "check the deployment status",
                    "intent": "Check the deployment status.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Checking that now.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_STARTED:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.INTERFACE_ORACLE_CANCEL,
                session_id="voice-123",
                sequence=2,
                payload={"job_id": "voice-oracle-001", "reason": "user requested cancellation"},
            )
        )
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_CANCELLED:
                break

        await engine.close()
        assert oracle.interrupted is False
        assert oracle.interrupted_requests == [
            (
                "voice-123:1",
                "check the deployment status",
                "Realtime voice oracle job voice-oracle-001 cancelled: user requested cancellation",
            )
        ]
        cancel_requested = next(event for event in seen if event.type == VoiceEventType.ORACLE_JOB_CANCEL_REQUESTED)
        cancelled = next(event for event in seen if event.type == VoiceEventType.ORACLE_JOB_CANCELLED)
        assert cancel_requested.payload["job_id"] == "voice-oracle-001"
        assert cancelled.payload["job_id"] == "voice-oracle-001"
        assert cancelled.payload["state"] == "cancelled"
        assert cancelled.payload["cancel_reason"] == "user requested cancellation"
        assert not any(event.payload.get("oracle_job_result") for event in seen)
        assert spoken == ["Checking that now."]

    asyncio.run(run())


def test_kame_engine_cancel_one_of_two_running_oracle_jobs_leaves_other_running(monkeypatch):
    class ConcurrentInterruptibleOracle:
        def __init__(self):
            self.requests = []
            self.second_started = asyncio.Event()
            self.releases: dict[str, asyncio.Event] = {}
            self.interrupted_requests = []

        def interrupt_request(self, request, message: str = ""):
            self.interrupted_requests.append((request.turn_id, request.oracle_text, message))

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.releases[request.intent] = asyncio.Event()
            if len(self.requests) == 2:
                self.second_started.set()
            await self.releases[request.intent].wait()
            yield f"Finished {request.intent}"

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = ConcurrentInterruptibleOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={"enabled": True, "max_concurrent": 2, "queue_limit": 4},
                metadata={"transport": "discord_voice"},
            )
        )

        for sequence, transcript, intent, ack in [
            (1, "check deployment status", "Check deployment status.", "Checking deployment status."),
            (2, "draft vendor memo", "Draft vendor memo.", "Drafting the vendor memo."),
        ]:
            await engine.receive_event(
                VoiceEvent(
                    type=VoiceEventType.AUDIO_INPUT_CHUNK,
                    session_id="voice-123",
                    sequence=sequence,
                    payload={
                        "transcript": transcript,
                        "intent": intent,
                        "intent_source": "reflex_audio",
                        "route": "defer",
                        "interface_already_said": ack,
                        "end_of_utterance": True,
                    },
                )
            )

        seen = []
        async for event in engine.events():
            seen.append(event)
            started_ids = {
                item.payload["job_id"]
                for item in seen
                if item.type == VoiceEventType.ORACLE_JOB_STARTED
            }
            if started_ids == {"voice-oracle-001", "voice-oracle-002"}:
                break
        await asyncio.wait_for(oracle.second_started.wait(), timeout=1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.INTERFACE_ORACLE_CANCEL,
                session_id="voice-123",
                sequence=3,
                payload={"job_id": "voice-oracle-001", "reason": "deployment check cancelled"},
            )
        )
        async for event in engine.events():
            seen.append(event)
            if (
                event.type == VoiceEventType.ORACLE_JOB_CANCELLED
                and event.payload.get("job_id") == "voice-oracle-001"
            ):
                break

        assert oracle.interrupted_requests == [
            (
                "voice-123:1",
                "check deployment status",
                "Realtime voice oracle job voice-oracle-001 cancelled: deployment check cancelled",
            )
        ]
        assert not any(
            event.type in {
                VoiceEventType.ORACLE_JOB_CANCEL_REQUESTED,
                VoiceEventType.ORACLE_JOB_CANCELLED,
            }
            and event.payload.get("job_id") == "voice-oracle-002"
            for event in seen
        )
        status = await engine.get_oracle_job_status()
        jobs_by_id = {job["job_id"]: job for job in status["jobs"]}
        assert jobs_by_id["voice-oracle-001"]["state"] == "cancelled"
        assert jobs_by_id["voice-oracle-002"]["state"] == "running"
        assert status["capacity"]["running"] == 1

        oracle.releases["Draft vendor memo."].set()
        async for event in engine.events():
            seen.append(event)
            if (
                event.type == VoiceEventType.ORACLE_JOB_COMPLETED
                and event.payload.get("job_id") == "voice-oracle-002"
            ):
                break

        await engine.close()
        assert any(
            event.type == VoiceEventType.ORACLE_JOB_COMPLETED
            and event.payload.get("job_id") == "voice-oracle-002"
            for event in seen
        )
        assert not any(
            event.type == VoiceEventType.ORACLE_JOB_COMPLETED
            and event.payload.get("job_id") == "voice-oracle-001"
            for event in seen
        )
        assert "Drafting the vendor memo." in spoken

    asyncio.run(run())


def test_kame_engine_close_bounds_noncooperative_async_oracle_shutdown(monkeypatch):
    class NonCooperativeOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()
            self.cancellation_entered = asyncio.Event()
            self.release_worker = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancellation_entered.set()
                await self.release_worker.wait()
                yield "late result should not be spoken"

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = NonCooperativeOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                oracle_jobs={
                    "enabled": True,
                    "max_concurrent": 1,
                    "queue_limit": 4,
                    "shutdown_timeout_seconds": 0.01,
                },
                metadata={"transport": "discord_voice"},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "check the deployment status",
                    "intent": "Check the deployment status.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "interface_already_said": "Checking that now.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_STARTED:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)

        await asyncio.wait_for(engine.close(), timeout=1)
        assert oracle.cancellation_entered.is_set()

        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.SESSION_CLOSED:
                break

        oracle.release_worker.set()
        await asyncio.sleep(0)

        cancelled = next(event for event in seen if event.type == VoiceEventType.ORACLE_JOB_CANCELLED)
        assert cancelled.payload["job_id"] == "voice-oracle-001"
        assert cancelled.payload["state"] == "cancelled"
        assert cancelled.payload["cancel_reason"] == "session closed"
        assert not any(event.payload.get("oracle_job_result") for event in seen)
        assert spoken == ["Checking that now."]

    asyncio.run(run())


def test_kame_engine_accepts_sidecar_interface_intent_final(monkeypatch):
    class StructuredOracle:
        def __init__(self):
            self.requests = []

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            yield "The deployment is healthy."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = StructuredOracle()
        sidecar = FakeSidecar()
        engine = KameInterfaceOracleEngine(oracle=oracle, sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
                sidecar_base_url="http://voice.local:8765",
                metadata={"transport": "discord_voice", "user_id": "42"},
            )
        )
        await sidecar._events.put(
            VoiceEvent(
                type=VoiceEventType.INTERFACE_INTENT_FINAL,
                session_id="voice-123",
                sequence=1,
                payload={
                    "input_generation": 3,
                    "text": "check the deployment status",
                    "intent": "Check the deployment status.",
                    "intent_source": "reflex_audio",
                    "route": "oracle_direct",
                    "route_confidence": 0.86,
                    "transcript": "check the deployment status",
                    "transcript_source": "reflex_audio",
                    "transcript_confidence": 0.74,
                    "asr_transcript": "check deployment status",
                    "asr_transcript_source": "asr",
                    "asr_transcript_confidence": 0.91,
                    "interface_input_source": "native_audio",
                    "reflex_provider": "vllm",
                    "metrics": {"kame_speech_end_to_interface_decision_ms": 42},
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()
        assert len(oracle.requests) == 1
        request = oracle.requests[0]
        assert request.intent == "Check the deployment status."
        assert request.intent_source == "reflex_audio"
        assert request.route == KameRoute.ORACLE_DIRECT
        assert request.route_confidence == 0.86
        assert request.transcript == "check the deployment status"
        assert request.transcript_source == "reflex_audio"
        assert request.asr_transcript == "check deployment status"
        assert request.asr_transcript_source == "asr"
        assert request.asr_transcript_confidence == 0.91
        assert request.oracle_text == "check deployment status"
        assert request.interface_input_source == "native_audio"
        assert request.reflex_provider == "vllm"

        final = next(event for event in seen if event.type == VoiceEventType.TRANSCRIPT_FINAL)
        intent = next(event for event in seen if event.type == VoiceEventType.INTERFACE_INTENT_FINAL)
        oracle_request = next(event for event in seen if event.type == VoiceEventType.INTERFACE_ORACLE_REQUEST)
        commit = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_COMMIT)
        assert final.payload["input_generation"] == 3
        assert final.payload["kame_intent"] == "Check the deployment status."
        assert final.payload["kame_asr_transcript"] == "check deployment status"
        assert final.payload["kame_interface_input_source"] == "native_audio"
        assert intent.payload["input_generation"] == 3
        assert intent.payload["metrics"]["kame_speech_end_to_interface_decision_ms"] == 42
        assert intent.payload["metrics"]["kame_final_transcript_to_interface_decision_ms"] >= 0
        assert oracle_request.payload["text"] == "check deployment status"
        assert oracle_request.payload["oracle_text_source"] == "asr"
        assert commit.payload["text"] == "The deployment is healthy."
        assert spoken == ["The deployment is healthy."]

    asyncio.run(run())


def test_kame_engine_defer_acknowledgement_reports_first_audio_metric(monkeypatch, tmp_path):
    class StructuredOracle:
        async def stream_answer_for_request(self, request):
            yield "The deployment is healthy."

    async def run():
        counter = 0

        def fake_tts_sync(text):
            nonlocal counter
            counter += 1
            audio_path = tmp_path / f"defer-{counter}.ogg"
            audio_path.write_bytes(text.encode("utf-8"))
            return str(audio_path)

        engine = KameInterfaceOracleEngine(oracle=StructuredOracle())
        monkeypatch.setattr(engine, "_tts_sync", fake_tts_sync)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                metadata={
                    "transport": "discord_voice",
                    "turn_acknowledgement": {"enabled": True, "text": "One moment."},
                },
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "check the deployment status",
                    "intent": "Check the deployment status.",
                    "intent_source": "reflex_audio",
                    "route": "defer",
                    "transcript_source": "reflex_audio",
                    "end_of_utterance": True,
                    "metrics": {"kame_speech_end_to_interface_decision_ms": 41},
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()

        audio = next(event for event in seen if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK)
        session_metrics = next(event for event in seen if event.type == VoiceEventType.SESSION_METRICS)
        assert AudioChunk.from_payload(audio.payload).data == b"I'm checking the deployment status."
        assert audio.payload["kame_interface_already_said"] == "I'm checking the deployment status."
        assert audio.payload["metrics"]["kame_interface_decision_to_first_audio_ms"] >= 0
        assert audio.payload["metrics"]["kame_interface_decision_to_defer_first_audio_ms"] >= 0
        assert audio.payload["metrics"]["kame_speech_end_to_first_audio_ms"] >= 41
        assert audio.payload["metrics"]["kame_speech_end_to_defer_first_audio_ms"] >= 41
        assert session_metrics.payload["metrics"]["kame_interface_decision_to_defer_first_audio_ms"] >= 0
        assert session_metrics.payload["metrics"]["kame_speech_end_to_defer_first_audio_ms"] >= 41

    asyncio.run(run())


def test_kame_engine_enforces_oracle_required_routing_for_local_payloads(monkeypatch):
    class StructuredOracle:
        def __init__(self):
            self.requests = []

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            yield "I will check the project config."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = StructuredOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                metadata={
                    "transport": "discord_voice",
                    "routing": {
                        "require_oracle_for_tools": True,
                        "require_oracle_for_memory": True,
                        "require_oracle_for_files": True,
                        "local_confidence_threshold": 0.75,
                    },
                },
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "check the project config file",
                    "intent": "Check the project config file.",
                    "intent_source": "reflex_audio",
                    "route": "local",
                    "route_confidence": 0.97,
                    "local_reply": "The config file looks fine.",
                    "transcript_source": "reflex_audio",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()

        assert len(oracle.requests) == 1
        request = oracle.requests[0]
        assert request.route == KameRoute.ORACLE_DIRECT
        assert request.local_reply == ""
        assert request.reflex_validation_error == "oracle_required_for_files"
        intent = next(event for event in seen if event.type == VoiceEventType.INTERFACE_INTENT_FINAL)
        oracle_request = next(event for event in seen if event.type == VoiceEventType.INTERFACE_ORACLE_REQUEST)
        final = next(event for event in seen if event.type == VoiceEventType.TRANSCRIPT_FINAL)
        assert not any(event.type == VoiceEventType.INTERFACE_REPLY_LOCAL for event in seen)
        assert intent.payload["route"] == "oracle_direct"
        assert intent.payload["reflex_validation_error"] == "oracle_required_for_files"
        assert "local_reply" not in intent.payload
        assert oracle_request.payload["route"] == "oracle_direct"
        assert oracle_request.payload["reflex_validation_error"] == "oracle_required_for_files"
        assert final.payload["kame_reflex_validation_error"] == "oracle_required_for_files"
        assert spoken == ["I will check the project config."]

    asyncio.run(run())


def test_kame_engine_downgrades_local_voice_capability_denial(monkeypatch):
    class StructuredOracle:
        def __init__(self):
            self.requests = []

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            yield "Voice is active; I can hear and speak here."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = StructuredOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "can you hear me",
                    "intent": "The user asks whether Hermes can hear them.",
                    "intent_source": "reflex_audio",
                    "route": "local",
                    "route_confidence": 0.98,
                    "local_reply": "I cannot hear you or speak in Discord voice.",
                    "transcript_source": "reflex_audio",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()

        assert len(oracle.requests) == 1
        request = oracle.requests[0]
        assert request.route == KameRoute.ORACLE_DIRECT
        assert request.local_reply == ""
        assert request.reflex_validation_error == "voice_capability_denial"
        intent = next(event for event in seen if event.type == VoiceEventType.INTERFACE_INTENT_FINAL)
        oracle_request = next(event for event in seen if event.type == VoiceEventType.INTERFACE_ORACLE_REQUEST)
        final = next(event for event in seen if event.type == VoiceEventType.TRANSCRIPT_FINAL)
        assert not any(event.type == VoiceEventType.INTERFACE_REPLY_LOCAL for event in seen)
        assert intent.payload["route"] == "oracle_direct"
        assert intent.payload["reflex_validation_error"] == "voice_capability_denial"
        assert "local_reply" not in intent.payload
        assert oracle_request.payload["route"] == "oracle_direct"
        assert oracle_request.payload["reflex_validation_error"] == "voice_capability_denial"
        assert final.payload["kame_reflex_validation_error"] == "voice_capability_denial"
        assert spoken == ["Voice is active; I can hear and speak here."]

    asyncio.run(run())


def test_kame_engine_enforces_disabled_local_greetings(monkeypatch):
    class StructuredOracle:
        def __init__(self):
            self.requests = []

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            yield "Voice is active."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = StructuredOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                metadata={"routing": {"allow_local_greetings": False}},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "can you hear me",
                    "intent": "The user asks whether Hermes can hear them.",
                    "intent_source": "reflex_audio",
                    "route": "local",
                    "route_confidence": 0.98,
                    "local_reply": "Yes, I can hear you.",
                    "transcript_source": "reflex_audio",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()

        assert len(oracle.requests) == 1
        assert oracle.requests[0].route == KameRoute.ORACLE_DIRECT
        assert oracle.requests[0].reflex_validation_error == "local_greetings_disabled"
        assert not any(event.type == VoiceEventType.INTERFACE_REPLY_LOCAL for event in seen)
        intent = next(event for event in seen if event.type == VoiceEventType.INTERFACE_INTENT_FINAL)
        assert intent.payload["route"] == "oracle_direct"
        assert intent.payload["reflex_validation_error"] == "local_greetings_disabled"
        assert spoken == ["Voice is active."]

    asyncio.run(run())


def test_kame_engine_barge_in_carries_cancelled_turn_token(monkeypatch):
    class SlowOracle:
        def __init__(self):
            self.release = asyncio.Event()
            self.interrupted = False
            self.requests = []

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            yield "First answer starts."
            await self.release.wait()
            yield " stale ending."

        def interrupt(self, message: str = ""):
            self.interrupted = True

    async def run():
        async def fake_speak(self, text, playback_generation):
            return None

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = SlowOracle()
        sidecar = FakeSidecar()
        engine = KameInterfaceOracleEngine(oracle=oracle, sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                sidecar_base_url="http://voice.local:8765",
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "look this up",
                    "text": "look this up",
                    "intent": "Look this up.",
                    "route": "oracle_direct",
                    "intent_source": "reflex_audio",
                    "asr_transcript": "look this up exactly",
                    "asr_transcript_source": "asr",
                    "interface_input_source": "native_audio",
                    "reflex_provider": "vllm",
                    "end_of_utterance": True,
                },
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
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"new-speech").to_payload(),
                    "speech_confirmed": True,
                },
            )
        )

        cancel = await anext(engine.events())
        barge_in = await anext(engine.events())
        oracle_error = await anext(engine.events())
        await engine.close()

        assert oracle.requests[0].cancellation_token == "voice-123:1:cancel"
        final = next(event for event in seen if event.type == VoiceEventType.TRANSCRIPT_FINAL)
        assert final.payload["kame_cancellation_token"] == "voice-123:1:cancel"
        assert cancel.type == VoiceEventType.INTERFACE_ORACLE_CANCEL
        assert cancel.payload["playback_generation"] == 2
        assert cancel.payload["cancelled_playback_generation"] == 1
        assert cancel.payload["cancellation_token"] == "voice-123:1:cancel"
        assert cancel.payload["text"] == "look this up exactly"
        assert cancel.payload["oracle_text_source"] == "asr"
        assert cancel.payload["asr_transcript"] == "look this up exactly"
        assert cancel.payload["asr_transcript_source"] == "asr"
        assert cancel.payload["interface_input_source"] == "native_audio"
        assert cancel.payload["reflex_provider"] == "vllm"
        assert barge_in.type == VoiceEventType.BARGE_IN
        assert barge_in.payload["reason"] == "user_speech"
        assert barge_in.payload["playback_generation"] == 2
        assert barge_in.payload["cancelled_playback_generation"] == 1
        assert barge_in.payload["cancellation_token"] == "voice-123:1:cancel"
        assert barge_in.payload["backend_interrupt_requested"] is True
        assert oracle_error.type == VoiceEventType.ORACLE_ERROR
        assert oracle_error.payload["reason"] == "oracle_cancelled"
        assert oracle_error.payload["cancel_reason"] == "user_speech"
        assert oracle_error.payload["playback_generation"] == 2
        assert oracle_error.payload["cancelled_playback_generation"] == 1
        assert oracle_error.payload["cancellation_token"] == "voice-123:1:cancel"
        assert oracle_error.payload["turn_id"] == "voice-123:1"
        assert oracle.interrupted is True
        forwarded_cancel = next(
            event for event in sidecar.received if event.type == VoiceEventType.INTERFACE_ORACLE_CANCEL
        )
        assert forwarded_cancel.payload == cancel.payload
        forwarded_oracle_error = next(event for event in sidecar.received if event.type == VoiceEventType.ORACLE_ERROR)
        assert forwarded_oracle_error.payload == oracle_error.payload
        assert engine._cancellation_token_by_generation == {}

    asyncio.run(run())


def test_kame_engine_streams_oracle_hints_to_sidecar(monkeypatch):
    class HintOracle:
        async def stream_answer_for_request(self, request):
            yield "Looking now"
            yield "."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        sidecar = FakeSidecar()
        engine = KameInterfaceOracleEngine(oracle=HintOracle(), sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                sidecar_base_url="http://voice.local:8765",
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "look this up",
                    "text": "look this up",
                    "intent": "Look this up.",
                    "route": "oracle_direct",
                    "intent_source": "reflex_audio",
                    "end_of_utterance": True,
                    "metrics": {"kame_speech_end_to_interface_decision_ms": 123},
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()

        hints = [event for event in seen if event.type == VoiceEventType.ORACLE_HINT]
        oracle_events = [
            event for event in seen
            if event.type in {
                VoiceEventType.ORACLE_ACCEPTED,
                VoiceEventType.ORACLE_RESPONSE_PARTIAL,
                VoiceEventType.ORACLE_RESPONSE_FINAL,
            }
        ]
        forwarded = [event for event in sidecar.received if event.type == VoiceEventType.ORACLE_HINT]
        forwarded_oracle_events = [
            event for event in sidecar.received
            if event.type in {
                VoiceEventType.ORACLE_ACCEPTED,
                VoiceEventType.ORACLE_RESPONSE_PARTIAL,
                VoiceEventType.ORACLE_RESPONSE_FINAL,
            }
        ]
        forwarded_interface_events = [
            event
            for event in sidecar.received
            if event.type
            in {
                VoiceEventType.INTERFACE_INTENT_FINAL,
                VoiceEventType.INTERFACE_ORACLE_REQUEST,
                VoiceEventType.INTERFACE_COMMIT,
            }
        ]
        assert [event.type for event in oracle_events] == [
            VoiceEventType.ORACLE_ACCEPTED,
            VoiceEventType.ORACLE_RESPONSE_PARTIAL,
            VoiceEventType.ORACLE_RESPONSE_PARTIAL,
            VoiceEventType.ORACLE_RESPONSE_FINAL,
        ]
        assert [event.payload.get("accepted") for event in oracle_events] == [True, None, None, None]
        assert [event.payload["delta"] for event in oracle_events] == ["", "Looking now", ".", ""]
        assert oracle_events[-1].payload["final"] is True
        assert oracle_events[-1].payload["text"] == "Looking now."
        assert oracle_events[-1].payload["turn_id"] == "voice-123:1"
        assert oracle_events[-1].payload["metrics"]["kame_oracle_total_stream_ms"] >= 0
        intent = next(event for event in seen if event.type == VoiceEventType.INTERFACE_INTENT_FINAL)
        assert intent.payload["metrics"]["kame_speech_end_to_interface_decision_ms"] == 123
        assert intent.payload["metrics"]["kame_final_transcript_to_interface_decision_ms"] >= 0
        assert [hint.payload.get("accepted") for hint in hints] == [True, None, None, None]
        assert [hint.payload["delta"] for hint in hints] == ["", "Looking now", ".", ""]
        assert hints[-1].payload["final"] is True
        assert hints[-1].payload["text"] == "Looking now."
        assert hints[-1].payload["metrics"]["kame_speech_end_to_interface_decision_ms"] == 123
        assert hints[-1].payload["metrics"]["kame_final_transcript_to_interface_decision_ms"] >= 0
        assert hints[-1].payload["metrics"]["kame_oracle_called"] == 1
        assert hints[-1].payload["metrics"]["kame_oracle_bypassed"] == 0
        assert hints[-1].payload["metrics"]["kame_interface_decision_to_oracle_accepted_ms"] >= 0
        assert hints[-1].payload["metrics"]["kame_oracle_accepted_to_first_token_ms"] >= 0
        assert hints[-1].payload["metrics"]["kame_oracle_first_token_to_first_spoken_text_ms"] >= 0
        assert hints[-1].payload["metrics"]["kame_oracle_total_stream_ms"] >= 0
        commit = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_COMMIT)
        assert commit.payload["metrics"]["kame_speech_end_to_interface_decision_ms"] == 123
        assert commit.payload["metrics"]["kame_final_transcript_to_interface_decision_ms"] >= 0
        assert commit.payload["metrics"]["kame_oracle_accepted_to_first_token_ms"] >= 0
        assert commit.payload["metrics"]["kame_oracle_first_token_to_first_spoken_text_ms"] >= 0
        assert [event.payload for event in forwarded] == [event.payload for event in hints]
        assert [event.payload for event in forwarded_oracle_events] == [event.payload for event in oracle_events]
        assert [event.type for event in forwarded_interface_events] == [
            VoiceEventType.INTERFACE_INTENT_FINAL,
            VoiceEventType.INTERFACE_ORACLE_REQUEST,
            VoiceEventType.INTERFACE_COMMIT,
        ]
        assert forwarded_interface_events[0].payload == intent.payload
        assert forwarded_interface_events[1].payload["turn_id"] == "voice-123:1"
        assert forwarded_interface_events[1].payload["route"] == "oracle_direct"
        assert forwarded_interface_events[2].payload["text"] == "Looking now."
        assert spoken == ["Looking now."]

    asyncio.run(run())


def test_kame_engine_ignores_raw_live_provider_transcript_final():
    class RawTranscriptSidecar:
        def __init__(self):
            self._events = asyncio.Queue()
            self.received = []

        async def start(self, config):
            self.config = config

        async def send_event(self, event):
            self.received.append(event)

        async def speak(self, event):
            self.received.append(event)

        async def events(self):
            while True:
                event = await self._events.get()
                if event is None:
                    return
                yield event

        async def close(self):
            await self._events.put(None)

    class FailingOracle:
        async def stream_answer_for_request(self, request):
            raise AssertionError("raw live-provider transcript must not drive the KAME oracle")
            yield ""

    async def run():
        sidecar = RawTranscriptSidecar()
        engine = KameInterfaceOracleEngine(oracle=FailingOracle(), sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemini_live",
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
            )
        )
        events = engine.events()
        started = await asyncio.wait_for(anext(events), timeout=1)
        assert started.type == VoiceEventType.SESSION_STARTED

        await sidecar._events.put(
            VoiceEvent(
                type=VoiceEventType.TRANSCRIPT_FINAL,
                session_id="voice-123",
                sequence=1,
                payload={"text": "raw provider transcript"},
            )
        )

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(anext(events), timeout=0.05)
        await engine.close()

    asyncio.run(run())


def test_kame_engine_allows_live_provider_tool_transcript_final(monkeypatch):
    class ToolTranscriptSidecar:
        def __init__(self):
            self._events = asyncio.Queue()
            self.received = []

        async def start(self, config):
            self.config = config

        async def send_event(self, event):
            self.received.append(event)

        async def speak(self, event):
            self.received.append(event)

        async def events(self):
            while True:
                event = await self._events.get()
                if event is None:
                    return
                yield event

        async def close(self):
            await self._events.put(None)

    class ToolOracle:
        def __init__(self):
            self.requests = []

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            yield "Tool answer."

    async def run():
        async def fake_speak(self, text, playback_generation):
            return None

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        sidecar = ToolTranscriptSidecar()
        oracle = ToolOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle, sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemini_live",
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
            )
        )
        events = engine.events()
        started = await asyncio.wait_for(anext(events), timeout=1)
        assert started.type == VoiceEventType.SESSION_STARTED

        await sidecar._events.put(
            VoiceEvent(
                type=VoiceEventType.TRANSCRIPT_FINAL,
                session_id="voice-123",
                sequence=1,
                payload={
                    "text": "use memory and tools",
                    "source": "gemini_live_tool",
                    "tool_call_id": "call-1",
                },
            )
        )

        seen = []
        async for event in events:
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break
        await engine.close()
        assert len(oracle.requests) == 1
        assert oracle.requests[0].intent == "use memory and tools"
        assert oracle.requests[0].source == "gemini_live_tool"
        assert any(event.type == VoiceEventType.INTERFACE_ORACLE_REQUEST for event in seen)

    asyncio.run(run())


def test_kame_engine_adds_first_audio_metrics_to_sidecar_tts_chunks():
    class HintOracle:
        async def stream_answer_for_request(self, request):
            yield "Looking now."

    async def run():
        sidecar = FakeSidecar()
        engine = KameInterfaceOracleEngine(oracle=HintOracle(), sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                sidecar_base_url="http://voice.local:8765",
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "look this up",
                    "text": "look this up",
                    "intent": "Look this up.",
                    "route": "oracle_direct",
                    "intent_source": "reflex_audio",
                    "end_of_utterance": True,
                    "metrics": {"kame_speech_end_to_interface_decision_ms": 25},
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK:
                await engine.close()
                break

        audio = next(event for event in seen if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK)
        assert audio.payload["voice_architecture"] == "kame_frontend_oracle"
        assert audio.payload["kame_route"] == KameRoute.ORACLE_DIRECT.value
        assert audio.payload["metrics"]["kame_speech_end_to_interface_decision_ms"] == 25
        assert audio.payload["metrics"]["kame_interface_decision_to_first_audio_ms"] >= 0
        assert audio.payload["metrics"]["kame_speech_end_to_first_audio_ms"] >= 25
        assert audio.payload["metrics"]["kame_oracle_first_token_to_first_tts_audio_ms"] >= 0
        assert sidecar.spoken[0].payload["voice_architecture"] == "kame_frontend_oracle"

    asyncio.run(run())


def test_kame_engine_streams_oracle_tool_events_to_sidecar(monkeypatch):
    class ToolEventOracle:
        async def stream_answer_for_request(self, request):
            yield {
                "type": "oracle.tool_call",
                "tool_name": "read_file",
                "tool_call_id": "call-1",
                "arguments": {"path": "pyproject.toml"},
            }
            yield {
                "event": "tool_result",
                "tool_name": "read_file",
                "tool_call_id": "call-1",
                "result": {"ok": True, "bytes": 42},
            }
            yield "I checked it."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        sidecar = FakeSidecar()
        engine = KameInterfaceOracleEngine(oracle=ToolEventOracle(), sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                sidecar_base_url="http://voice.local:8765",
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "check the project file",
                    "intent": "Check the project file.",
                    "intent_source": "reflex_audio",
                    "route": "oracle_direct",
                    "end_of_utterance": True,
                    "metrics": {"kame_speech_end_to_interface_decision_ms": 17},
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()

        tool_call = next(event for event in seen if event.type == VoiceEventType.ORACLE_TOOL_CALL)
        tool_result = next(event for event in seen if event.type == VoiceEventType.ORACLE_TOOL_RESULT)
        partial = next(event for event in seen if event.type == VoiceEventType.ORACLE_RESPONSE_PARTIAL)
        final = next(event for event in seen if event.type == VoiceEventType.ORACLE_RESPONSE_FINAL)
        forwarded_tool_call = next(event for event in sidecar.received if event.type == VoiceEventType.ORACLE_TOOL_CALL)
        forwarded_tool_result = next(event for event in sidecar.received if event.type == VoiceEventType.ORACLE_TOOL_RESULT)

        assert tool_call.payload["turn_id"] == "voice-123:1"
        assert tool_call.payload["route"] == "oracle_direct"
        assert tool_call.payload["tool_name"] == "read_file"
        assert tool_call.payload["tool_call_id"] == "call-1"
        assert tool_call.payload["arguments"] == {"path": "pyproject.toml"}
        assert tool_call.payload["metrics"]["kame_oracle_called"] == 1
        assert tool_call.payload["metrics"]["kame_speech_end_to_interface_decision_ms"] == 17
        assert tool_result.payload["tool_name"] == "read_file"
        assert tool_result.payload["tool_call_id"] == "call-1"
        assert tool_result.payload["result"] == {"ok": True, "bytes": 42}
        assert partial.payload["delta"] == "I checked it."
        assert final.payload["text"] == "I checked it."
        assert forwarded_tool_call.payload == tool_call.payload
        assert forwarded_tool_result.payload == tool_result.payload
        assert spoken == ["I checked it."]

    asyncio.run(run())


def test_kame_engine_corrects_oracle_voice_capability_denial(monkeypatch):
    class DenialOracle:
        async def stream_answer_for_request(self, request):
            yield "You're absolutely right — I cannot hear you or speak in Discord voice."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        engine = KameInterfaceOracleEngine(oracle=DenialOracle())
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                metadata={"voice_capability_correction_text": "Voice is active; I can hear and speak here."},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "can you use voice",
                    "intent": "Confirm Hermes can use live voice.",
                    "route": "oracle_direct",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()

        assistant_partials = [event.payload["text"] for event in seen if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL]
        hints = [event for event in seen if event.type == VoiceEventType.ORACLE_HINT]
        oracle_responses = [
            event for event in seen
            if event.type in {VoiceEventType.ORACLE_RESPONSE_PARTIAL, VoiceEventType.ORACLE_RESPONSE_FINAL}
        ]
        commit = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_COMMIT)

        assert spoken == ["Voice is active; I can hear and speak here."]
        assert assistant_partials == ["Voice is active; I can hear and speak here."]
        assert commit.payload["text"] == "Voice is active; I can hear and speak here."
        assert all("cannot hear" not in str(event.payload).lower() for event in hints)
        assert all("cannot hear" not in str(event.payload).lower() for event in oracle_responses)
        assert any(event.payload.get("voice_capability_corrected") is True for event in hints)
        assert any(event.payload.get("voice_capability_corrected") is True for event in oracle_responses)

    asyncio.run(run())


def test_kame_engine_caps_oracle_speech_to_configured_sentence_budget(monkeypatch):
    class VerboseOracle:
        def __init__(self):
            self.requests = []

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            yield "First sentence. "
            yield "Second sentence. "
            yield "Third sentence."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = VerboseOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                max_spoken_sentences=2,
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "explain the plan",
                    "intent": "Explain the plan.",
                    "route": "oracle_direct",
                    "intent_source": "reflex_audio",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()

        partials = [event.payload["text"] for event in seen if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL]
        commit = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_COMMIT)
        assert oracle.requests[0].max_spoken_sentences == 2
        assert oracle.requests[0].requested_response_style["max_sentences"] == 2
        assert oracle.requests[0].requested_response_style["policy"] == "sentence_cap"
        assert partials == ["First sentence.", "Second sentence."]
        assert spoken == ["First sentence.", "Second sentence."]
        assert commit.payload["text"] == "First sentence. Second sentence."
        assert commit.payload["max_spoken_sentences"] == 2
        assert commit.payload["voice_response_policy"] == "sentence_cap"
        assert commit.payload["metrics"]["kame_oracle_called"] == 1

    asyncio.run(run())


def test_kame_engine_full_voice_response_policy_disables_sentence_cap(monkeypatch):
    class VerboseOracle:
        def __init__(self):
            self.requests = []

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            yield "First sentence. "
            yield "Second sentence. "
            yield "Third sentence."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = VerboseOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                max_spoken_sentences=2,
                voice_response_policy="full",
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "explain the plan",
                    "intent": "Explain the plan.",
                    "route": "oracle_direct",
                    "intent_source": "reflex_audio",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()

        commit = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_COMMIT)
        assert oracle.requests[0].requested_response_style["policy"] == "full"
        assert spoken == ["First sentence.", "Second sentence.", "Third sentence."]
        assert commit.payload["text"] == "First sentence. Second sentence. Third sentence."
        assert commit.payload["voice_response_policy"] == "full"
        assert commit.payload["voice_response_truncated"] is False

    asyncio.run(run())


def test_kame_engine_brief_summary_policy_limits_spoken_output_to_one_sentence(monkeypatch):
    class VerboseOracle:
        def __init__(self):
            self.requests = []

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            yield "First sentence. "
            yield "Second sentence. "
            yield "Third sentence."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = VerboseOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                max_spoken_sentences=3,
                voice_response_policy="brief_summary",
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "summarize the plan",
                    "intent": "Summarize the plan.",
                    "route": "oracle_direct",
                    "intent_source": "reflex_audio",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()

        commit = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_COMMIT)
        assert oracle.requests[0].requested_response_style["policy"] == "brief_summary"
        assert oracle.requests[0].requested_response_style["max_sentences"] == 1
        assert spoken == ["First sentence."]
        assert commit.payload["text"] == "First sentence."
        assert commit.payload["voice_response_policy"] == "brief_summary"
        assert commit.payload["max_spoken_sentences"] == 1
        assert commit.payload["voice_response_truncated"] is True

    asyncio.run(run())


def test_kame_engine_sends_oracle_timing_metrics_to_tts_sidecar():
    class TimedOracle:
        async def stream_answer_for_request(self, request):
            yield "Done."

    async def run():
        sidecar = FakeSidecar()
        engine = KameInterfaceOracleEngine(oracle=TimedOracle(), sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                sidecar_base_url="http://voice.local:8765",
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "look this up",
                    "intent": "Look this up.",
                    "route": "oracle_direct",
                    "end_of_utterance": True,
                    "metrics": {"kame_speech_end_to_interface_decision_ms": 42},
                },
            )
        )

        async for event in engine.events():
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()

        assert sidecar.spoken
        metrics = sidecar.spoken[0].payload["metrics"]
        assert metrics["kame_speech_end_to_interface_decision_ms"] == 42
        assert metrics["kame_final_transcript_to_interface_decision_ms"] >= 0
        assert metrics["kame_oracle_called"] == 1
        assert metrics["kame_interface_decision_to_oracle_accepted_ms"] >= 0
        assert metrics["kame_oracle_accepted_to_first_token_ms"] >= 0
        assert metrics["kame_oracle_first_token_to_first_spoken_text_ms"] >= 0

    asyncio.run(run())


def test_kame_engine_measures_oracle_acceptance_from_interface_decision():
    engine = KameInterfaceOracleEngine(oracle=FakeOracle())
    engine._interface_decision_at_by_generation[7] = 10.0

    assert engine._interface_decision_metric_start(7, fallback=10.2) == 10.0
    assert engine._interface_decision_metric_start(8, fallback=10.2) == 10.2


def test_kame_engine_local_route_speaks_without_oracle(monkeypatch):
    class UnexpectedOracle:
        async def stream_answer_for_request(self, request):
            raise AssertionError("local KAME route must not call oracle")

        async def stream_answer_with_metadata(self, transcript, metadata):
            raise AssertionError("local KAME route must not call oracle")

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        engine = KameInterfaceOracleEngine(oracle=UnexpectedOracle())
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "can you hear me",
                    "text": "can you hear me",
                    "intent": "The user is checking whether Hermes can hear them.",
                    "route": "local",
                    "local_reply": "Yes, I can hear you.",
                    "intent_source": "reflex_audio",
                    "end_of_utterance": True,
                    "metrics": {"kame_speech_end_to_interface_decision_ms": 37},
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()

        final = next(event for event in seen if event.type == VoiceEventType.TRANSCRIPT_FINAL)
        interface_reply = next(event for event in seen if event.type == VoiceEventType.INTERFACE_REPLY_LOCAL)
        interface_commit = next(event for event in seen if event.type == VoiceEventType.INTERFACE_COMMIT)
        partial = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL)
        session_metrics = next(event for event in seen if event.type == VoiceEventType.SESSION_METRICS)
        commit = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_COMMIT)
        assert not any(event.type == VoiceEventType.ORACLE_HINT for event in seen)
        assert not any(event.type == VoiceEventType.INTERFACE_ORACLE_REQUEST for event in seen)
        assert final.payload["kame_route"] == "local"
        assert final.payload["kame_local_reply"] == "Yes, I can hear you."
        assert interface_reply.payload["route"] == "local"
        assert interface_reply.payload["text"] == "Yes, I can hear you."
        assert interface_commit.payload["local_reply"] is True
        assert interface_commit.payload["text"] == "Yes, I can hear you."
        assert partial.payload["local_reply"] is True
        assert session_metrics.payload["outcome"] == "local_commit"
        assert session_metrics.payload["oracle_called"] is False
        assert session_metrics.payload["local_reply"] is True
        assert session_metrics.payload["metrics"]["kame_oracle_called"] == 0
        assert session_metrics.payload["metrics"]["kame_oracle_bypassed"] == 1
        assert commit.payload["local_reply"] is True
        assert commit.payload["text"] == "Yes, I can hear you."
        assert commit.payload["metrics"]["kame_oracle_called"] == 0
        assert commit.payload["metrics"]["kame_oracle_bypassed"] == 1
        assert spoken == ["Yes, I can hear you."]

    asyncio.run(run())


def test_kame_engine_forwards_local_interface_events_to_sidecar(monkeypatch):
    class UnexpectedOracle:
        async def stream_answer_for_request(self, request):
            raise AssertionError("local KAME route must not call oracle")

    async def run():
        async def fake_speak(self, text, playback_generation):
            return None

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        sidecar = FakeSidecar()
        engine = KameInterfaceOracleEngine(oracle=UnexpectedOracle(), sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                sidecar_base_url="http://voice.local:8765",
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "can you hear me",
                    "intent": "The user is checking whether Hermes can hear them.",
                    "route": "local",
                    "route_confidence": 0.95,
                    "local_reply": "Yes, I can hear you.",
                    "intent_source": "reflex_audio",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()

        emitted_interface_events = [
            event
            for event in seen
            if event.type
            in {
                VoiceEventType.INTERFACE_INTENT_FINAL,
                VoiceEventType.INTERFACE_REPLY_LOCAL,
                VoiceEventType.INTERFACE_COMMIT,
            }
        ]
        forwarded_interface_events = [
            event
            for event in sidecar.received
            if event.type
            in {
                VoiceEventType.INTERFACE_INTENT_FINAL,
                VoiceEventType.INTERFACE_REPLY_LOCAL,
                VoiceEventType.INTERFACE_COMMIT,
            }
        ]

        assert [event.type for event in emitted_interface_events] == [
            VoiceEventType.INTERFACE_INTENT_FINAL,
            VoiceEventType.INTERFACE_REPLY_LOCAL,
            VoiceEventType.INTERFACE_COMMIT,
        ]
        assert [event.payload for event in forwarded_interface_events] == [
            event.payload for event in emitted_interface_events
        ]
        assert forwarded_interface_events[1].payload["text"] == "Yes, I can hear you."
        assert forwarded_interface_events[2].payload["local_reply"] is True

    asyncio.run(run())


def test_kame_engine_local_route_reports_first_audio_metric(monkeypatch, tmp_path):
    class UnexpectedOracle:
        async def stream_answer_for_request(self, request):
            raise AssertionError("local KAME route must not call oracle")

        async def stream_answer_with_metadata(self, transcript, metadata):
            raise AssertionError("local KAME route must not call oracle")

    async def run():
        audio_path = tmp_path / "local.ogg"
        audio_path.write_bytes(b"local-audio")

        engine = KameInterfaceOracleEngine(oracle=UnexpectedOracle())
        monkeypatch.setattr(engine, "_tts_sync", lambda text: str(audio_path))
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "can you hear me",
                    "intent": "The user is checking whether Hermes can hear them.",
                    "route": "local",
                    "local_reply": "Yes, I can hear you.",
                    "intent_source": "reflex_audio",
                    "end_of_utterance": True,
                    "metrics": {"kame_speech_end_to_interface_decision_ms": 37},
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()

        audio = next(event for event in seen if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK)
        session_metrics = next(event for event in seen if event.type == VoiceEventType.SESSION_METRICS)
        assert AudioChunk.from_payload(audio.payload).data == b"local-audio"
        assert audio.payload["metrics"]["kame_interface_decision_to_first_audio_ms"] >= 0
        assert audio.payload["metrics"]["kame_interface_decision_to_local_first_audio_ms"] >= 0
        assert audio.payload["metrics"]["kame_speech_end_to_first_audio_ms"] >= 37
        assert audio.payload["metrics"]["kame_speech_end_to_local_first_audio_ms"] >= 37
        assert session_metrics.payload["metrics"]["kame_interface_decision_to_first_audio_ms"] >= 0
        assert session_metrics.payload["metrics"]["kame_interface_decision_to_local_first_audio_ms"] >= 0
        assert session_metrics.payload["metrics"]["kame_speech_end_to_first_audio_ms"] >= 37
        assert session_metrics.payload["metrics"]["kame_speech_end_to_local_first_audio_ms"] >= 37

    asyncio.run(run())


def test_kame_engine_adds_committed_voice_context_to_next_oracle_request(monkeypatch):
    class StructuredOracle:
        def __init__(self):
            self.requests = []

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            yield "I found the deployment status."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = StructuredOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "can you hear me",
                    "intent": "The user is checking whether Hermes can hear them.",
                    "route": "local",
                    "local_reply": "Yes, I can hear you.",
                    "intent_source": "reflex_audio",
                    "end_of_utterance": True,
                },
            )
        )
        async for event in engine.events():
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload={
                    "transcript": "check deployment status",
                    "intent": "Check deployment status.",
                    "route": "oracle_direct",
                    "intent_source": "reflex_audio",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()

        assert len(oracle.requests) == 1
        summary = oracle.requests[0].conversation_summary
        assert summary.startswith("Recent committed voice turns:")
        assert "The user is checking whether Hermes can hear them." in summary
        assert "Yes, I can hear you." in summary
        oracle_request = next(event for event in seen if event.type == VoiceEventType.INTERFACE_ORACLE_REQUEST)
        assert oracle_request.payload["conversation_summary"] == summary
        assert spoken == ["Yes, I can hear you.", "I found the deployment status."]

    asyncio.run(run())


def test_kame_engine_respects_metrics_policy_disabled(monkeypatch):
    class UnexpectedOracle:
        async def stream_answer_for_request(self, request):
            raise AssertionError("local KAME route must not call oracle")

    async def run():
        async def fake_speak(self, text, playback_generation):
            return None

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        engine = KameInterfaceOracleEngine(oracle=UnexpectedOracle())
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                metrics_policy={"enabled": False},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "can you hear me",
                    "intent": "The user is checking whether Hermes can hear them.",
                    "route": "local",
                    "local_reply": "Yes, I can hear you.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()
        assert not any(event.type == VoiceEventType.SESSION_METRICS for event in seen)
        assert any(event.type == VoiceEventType.ASSISTANT_COMMIT for event in seen)

    asyncio.run(run())


def test_kame_engine_respects_turn_span_metrics_policy(monkeypatch):
    class UnexpectedOracle:
        async def stream_answer_for_request(self, request):
            raise AssertionError("local KAME route must not call oracle")

    async def run():
        async def fake_speak(self, text, playback_generation):
            return None

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        engine = KameInterfaceOracleEngine(oracle=UnexpectedOracle())
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                metrics_policy={"enabled": True, "log_turn_spans": False},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "can you hear me",
                    "intent": "The user is checking whether Hermes can hear them.",
                    "route": "local",
                    "local_reply": "Yes, I can hear you.",
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()
        assert not any(event.type == VoiceEventType.SESSION_METRICS for event in seen)
        assert any(event.type == VoiceEventType.ASSISTANT_COMMIT for event in seen)

    asyncio.run(run())


def test_kame_engine_respects_provider_span_metrics_policy():
    class TimedOracle:
        async def stream_answer_for_request(self, request):
            yield "Done."

    async def run():
        sidecar = FakeSidecar()
        engine = KameInterfaceOracleEngine(oracle=TimedOracle(), sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                sidecar_base_url="http://voice.local:8765",
                metrics_policy={"enabled": True, "log_provider_spans": False},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "look this up",
                    "intent": "Look this up.",
                    "route": "oracle_direct",
                    "end_of_utterance": True,
                    "metrics": {"kame_speech_end_to_interface_decision_ms": 42},
                },
            )
        )

        async for event in engine.events():
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()

        assert sidecar.spoken
        metrics = sidecar.spoken[0].payload["metrics"]
        assert metrics["kame_speech_end_to_interface_decision_ms"] == 42
        assert metrics["kame_final_transcript_to_interface_decision_ms"] >= 0
        assert "kame_interface_decision_to_oracle_accepted_ms" not in metrics
        assert "kame_oracle_accepted_to_first_token_ms" not in metrics
        assert "kame_oracle_first_token_to_first_spoken_text_ms" not in metrics
        assert "kame_oracle_first_token_to_first_tts_audio_ms" not in metrics
        assert "tts_synthesis_ms" not in metrics

    asyncio.run(run())


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


def test_text_engine_drains_large_oracle_delta_into_multiple_speech_chunks(monkeypatch):
    class ParagraphOracle:
        async def stream_answer(self, transcript: str):
            yield "First paragraph is ready.\n\nSecond paragraph follows.\n\nThird paragraph lands."

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(TextOracleTTSEngine, "_speak_chunk", fake_speak)

        engine = TextOracleTTSEngine(oracle=ParagraphOracle())
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                max_spoken_sentences=10,
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "tell me something"},
            )
        )

        partials = []
        async for event in engine.events():
            if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL:
                partials.append(event.payload["text"])
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        await engine.close()

        assert partials == [
            "First paragraph is ready.",
            "Second paragraph follows.",
            "Third paragraph lands.",
        ]
        assert spoken == partials

    asyncio.run(run())


def test_text_engine_speaks_configured_acknowledgement_before_slow_oracle(monkeypatch):
    class SlowOracle:
        def __init__(self):
            self.release = asyncio.Event()

        async def stream_answer(self, transcript: str):
            await self.release.wait()
            yield "Done."

    async def run():
        spoken = []
        spoke = asyncio.Event()

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)
            spoke.set()

        monkeypatch.setattr(TextOracleTTSEngine, "_speak_chunk", fake_speak)

        oracle = SlowOracle()
        engine = TextOracleTTSEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                turn_acknowledgement={"enabled": True, "text": "One moment."},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "run something slow"},
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL:
                break

        await asyncio.wait_for(spoke.wait(), timeout=1)
        assert seen[-1].payload["text"] == "One moment."
        assert seen[-1].payload["acknowledgement"] is True
        assert spoken == ["One moment."]

        oracle.release.set()
        async for event in engine.events():
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                assert event.payload["text"] == "Done."
                await engine.close()
                break

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


def test_kame_engine_emits_partial_interface_intent_without_oracle():
    class TrackingOracle(FakeOracle):
        def __init__(self):
            self.called = False

        async def stream_answer(self, transcript: str):
            self.called = True
            yield "should not happen"

    async def run():
        oracle = TrackingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "can you",
                    "intent": "The user is starting a hearing check.",
                    "intent_source": "reflex_audio",
                    "route": "local",
                    "end_of_utterance": False,
                    "input_generation": 4,
                },
            )
        )

        events = [await anext(engine.events()), await anext(engine.events())]

        assert [event.type for event in events] == [
            VoiceEventType.SESSION_STARTED,
            VoiceEventType.INTERFACE_INTENT_PARTIAL,
        ]
        assert events[1].payload["intent"] == "The user is starting a hearing check."
        assert events[1].payload["route"] == "local"
        assert events[1].payload["input_generation"] == 4
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(anext(engine.events()), timeout=0.01)
        assert oracle.called is False
        await engine.close()

    asyncio.run(run())


def test_kame_engine_emits_partial_transcripts_in_debug_mode():
    async def run():
        engine = KameInterfaceOracleEngine(oracle=FakeOracle())
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.DEBUG,
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "can you",
                    "intent": "The user is starting a hearing check.",
                    "intent_source": "reflex_audio",
                    "route": "local",
                    "end_of_utterance": False,
                    "input_generation": 4,
                },
            )
        )

        events = [await anext(engine.events()), await anext(engine.events()), await anext(engine.events())]
        assert [event.type for event in events] == [
            VoiceEventType.SESSION_STARTED,
            VoiceEventType.INTERFACE_INTENT_PARTIAL,
            VoiceEventType.TRANSCRIPT_PARTIAL,
        ]
        assert events[2].payload["text"] == "can you"
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


def test_text_engine_speaks_status_when_oracle_times_out(monkeypatch):
    class HangingOracle:
        async def stream_answer(self, transcript: str):
            await asyncio.Event().wait()
            yield "unreachable"

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(TextOracleTTSEngine, "_speak_chunk", fake_speak)

        engine = TextOracleTTSEngine(oracle=HangingOracle())
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                oracle_timeout_seconds=0.01,
                metadata={"oracle_timeout_status_text": "Hermes is still thinking."},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "hello", "end_of_utterance": True},
            )
        )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                await engine.close()
                break

        degraded = next(event for event in seen if event.type == VoiceEventType.FRONTEND_STATE)
        partial = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL)
        commit = seen[-1]
        assert degraded.payload["reason"] == "oracle_timeout"
        assert degraded.payload["oracle_timeout_seconds"] == 0.01
        assert partial.payload == {
            "text": "Hermes is still thinking.",
            "playback_generation": 1,
            "oracle_timeout": True,
        }
        assert commit.payload == partial.payload
        assert spoken == ["Hermes is still thinking."]

    asyncio.run(run())


def test_kame_engine_timeout_status_keeps_oracle_metrics(monkeypatch):
    class HangingStructuredOracle:
        async def stream_answer_for_request(self, request):
            await asyncio.Event().wait()
            yield "unreachable"

    async def run():
        spoken = []

        async def fake_speak(self, text, playback_generation):
            spoken.append(text)

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        engine = KameInterfaceOracleEngine(oracle=HangingStructuredOracle())
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                oracle_timeout_seconds=0.01,
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "find the note",
                    "intent": "Find the note.",
                    "route": "oracle_direct",
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

        commit = seen[-1]
        oracle_error = next(event for event in seen if event.type == VoiceEventType.ORACLE_ERROR)
        assert oracle_error.payload["reason"] == "oracle_timeout"
        assert oracle_error.payload["error"] == "oracle response timed out"
        assert oracle_error.payload["turn_id"] == "voice-123:1"
        assert commit.payload["oracle_timeout"] is True
        assert commit.payload["metrics"]["kame_oracle_called"] == 1
        assert commit.payload["metrics"]["kame_oracle_bypassed"] == 0
        assert spoken == ["Hermes is taking too long to answer. Please try that again in a moment."]

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


def test_text_engine_drops_duplicate_sidecar_final_for_completed_input_generation(monkeypatch):
    class ManualSidecar(FakeSidecar):
        async def send_event(self, event):
            self.received.append(event)

    class CountingOracle:
        def __init__(self):
            self.calls = []

        async def stream_answer(self, transcript: str):
            self.calls.append(transcript)
            yield f"Answering: {transcript}."

    async def run():
        async def fake_speak(self, text, playback_generation):
            return None

        monkeypatch.setattr(TextOracleTTSEngine, "_speak_chunk", fake_speak)

        sidecar = ManualSidecar()
        oracle = CountingOracle()
        engine = TextOracleTTSEngine(oracle=oracle, sidecar=sidecar)
        await engine.start(RealtimeVoiceSessionConfig(session_id="voice-123", sidecar_base_url="http://voice.local"))

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
        generation = sidecar.received[-1].payload["input_generation"]
        for sequence in (1, 2):
            await sidecar._events.put(
                VoiceEvent(
                    type=VoiceEventType.TRANSCRIPT_FINAL,
                    session_id="voice-123",
                    sequence=sequence,
                    payload={"text": "duplicate transcript", "input_generation": generation},
                )
            )

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                await engine.close()
                break

        final_events = [event for event in seen if event.type == VoiceEventType.TRANSCRIPT_FINAL]
        assert [event.payload["text"] for event in final_events] == ["duplicate transcript"]
        assert [event.payload["input_generation"] for event in final_events] == [generation]
        assert oracle.calls == ["duplicate transcript"]

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


def test_text_engine_fail_closed_policy_raises_on_sidecar_start_failure():
    closed = {"value": False}

    class FailingStartSidecar:
        async def start(self, config):
            raise RuntimeError("sidecar down at http://user:pass@voice.local:8765/v1?token=abc")

        async def close(self):
            closed["value"] = True

    async def run():
        engine = TextOracleTTSEngine(oracle=FakeOracle(), sidecar=FailingStartSidecar())
        with pytest.raises(RuntimeError) as exc:
            await engine.start(
                RealtimeVoiceSessionConfig(
                    session_id="voice-123",
                    frontend_provider="gemma4",
                    sidecar_base_url="http://voice.local:8080",
                    fallback_policy="fail_closed",
                )
            )

        message = str(exc.value)
        assert "fallback_policy=fail_closed" in message
        assert "sidecar down" in message
        assert "user:pass" not in message
        assert "token=abc" not in message
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


def test_text_engine_fail_closed_policy_emits_session_error_on_sidecar_session_error():
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
                fallback_policy="fail_closed",
            )
        )

        events = [await anext(engine.events()), await anext(engine.events())]

        assert [event.type for event in events] == [
            VoiceEventType.SESSION_STARTED,
            VoiceEventType.SESSION_ERROR,
        ]
        error = events[-1]
        assert error.payload["reason"] == "sidecar_session_error"
        assert error.payload["sidecar"] is False
        assert "fallback_policy=fail_closed" in error.payload["error"]
        assert "TTS failed" in error.payload["error"]
        assert "secret-token" not in error.payload["error"]
        assert "user:pass" not in error.payload["error"]
        assert "token=abc" not in error.payload["error"]
        assert engine._sidecar is None
        assert sidecar.closed is True
        await engine.close()

    asyncio.run(run())


def test_text_engine_fail_closed_policy_emits_session_error_on_sidecar_event_stream_failure():
    class BrokenEventSidecar(FakeSidecar):
        async def events(self):
            raise RuntimeError("websocket died at http://user:pass@voice.local/v1?token=abc")
            yield

    async def run():
        sidecar = BrokenEventSidecar()
        engine = TextOracleTTSEngine(oracle=FakeOracle(), sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                frontend_provider="sidecar",
                sidecar_base_url="http://voice.local:8080",
                fallback_policy="fail_closed",
            )
        )

        events = [await anext(engine.events()), await anext(engine.events())]

        assert [event.type for event in events] == [
            VoiceEventType.SESSION_STARTED,
            VoiceEventType.SESSION_ERROR,
        ]
        error = events[-1]
        assert error.payload["reason"] == "sidecar_event_stream_failed"
        assert error.payload["sidecar"] is False
        assert "fallback_policy=fail_closed" in error.payload["error"]
        assert "websocket died" in error.payload["error"]
        assert "user:pass" not in error.payload["error"]
        assert "token=abc" not in error.payload["error"]
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


def test_kame_text_engine_suppresses_blank_audio_partial_in_normal_mode(monkeypatch):
    async def run():
        async def fake_speak(self, text, playback_generation):
            return None

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)
        monkeypatch.setattr(KameInterfaceOracleEngine, "_transcribe_sync", lambda self, audio, codec: "local transcript")

        engine = KameInterfaceOracleEngine(oracle=FakeOracle())
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
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

        assert VoiceEventType.TRANSCRIPT_PARTIAL not in [event.type for event in seen]
        final = next(event for event in seen if event.type == VoiceEventType.TRANSCRIPT_FINAL)
        assert final.payload["text"] == "local transcript"

    asyncio.run(run())


def test_kame_text_engine_labels_local_stt_as_interface_fallback(monkeypatch):
    class StructuredOracle:
        def __init__(self):
            self.requests = []

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            yield "Checked."

    async def run():
        async def fake_speak(self, text, playback_generation):
            return None

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)
        monkeypatch.setattr(KameInterfaceOracleEngine, "_transcribe_sync", lambda self, audio, codec: "local transcript")

        oracle = StructuredOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
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
        return seen, oracle.requests

    seen, requests = asyncio.run(run())
    assert len(requests) == 1
    request = requests[0]
    assert request.intent == "local transcript"
    assert request.intent_source == "asr_fallback"
    assert request.route == KameRoute.ORACLE_DIRECT
    assert request.transcript == "local transcript"
    assert request.transcript_source == "asr"
    assert request.asr_transcript == "local transcript"
    assert request.asr_transcript_source == "asr"
    assert request.interface_audio_input_fallback is True
    assert request.interface_input_source == "local_stt"
    assert request.reflex_provider == "local_stt"

    state = next(event for event in seen if event.type == VoiceEventType.FRONTEND_STATE)
    final = next(event for event in seen if event.type == VoiceEventType.TRANSCRIPT_FINAL)
    intent = next(event for event in seen if event.type == VoiceEventType.INTERFACE_INTENT_FINAL)
    oracle_request = next(event for event in seen if event.type == VoiceEventType.INTERFACE_ORACLE_REQUEST)
    session_metrics = next(event for event in seen if event.type == VoiceEventType.SESSION_METRICS)

    assert state.payload["status"] == "fallback"
    assert state.payload["reason"] == "kame_audio_reflex_unavailable"
    assert state.payload["provider"] == "local_stt"
    assert state.payload["fallback_provider"] == "local_stt"
    assert state.payload["intent_source"] == "asr_fallback"
    assert state.payload["transcript_source"] == "asr"
    assert state.payload["interface_audio_input_fallback"] is True
    assert state.payload["interface_input_source"] == "local_stt"
    assert state.payload["reflex_provider"] == "local_stt"
    assert state.payload["interface_audio_input"] == "native_audio"
    assert state.payload["asr_mode"] == "on_escalation"
    assert final.payload["text"] == "local transcript"
    assert final.payload["intent_source"] == "asr_fallback"
    assert final.payload["interface_audio_input_fallback"] is True
    assert final.payload["interface_input_source"] == "local_stt"
    assert final.payload["reflex_provider"] == "local_stt"
    assert final.payload["kame_interface_audio_input_fallback"] is True
    assert final.payload["kame_interface_input_source"] == "local_stt"
    assert final.payload["kame_reflex_provider"] == "local_stt"

    for event in (intent, oracle_request, session_metrics):
        assert event.payload["intent_source"] == "asr_fallback"
        assert event.payload["interface_audio_input_fallback"] is True
        assert event.payload["interface_input_source"] == "local_stt"
        assert event.payload["reflex_provider"] == "local_stt"


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
        assert event.payload["frontend_cancel_requested"] is True
        assert event.payload["backend_interrupt_requested"] is False
        assert sidecar.received[0].type == VoiceEventType.BARGE_IN
        assert sidecar.received[0].payload["playback_generation"] == 1
        assert sidecar.received[0].payload["frontend_cancel_requested"] is True
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
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"new-speech").to_payload(),
                    "speech_confirmed": True,
                },
            )
        )

        barge_in = await anext(engine.events())
        await engine.close()

        assert barge_in.type == VoiceEventType.BARGE_IN
        assert barge_in.payload["reason"] == "user_speech"
        assert barge_in.payload["playback_generation"] == 2
        assert barge_in.payload["backend_interrupt_requested"] is True
        assert oracle.interrupted is True
        assert engine._inbound_audio == [b"new-speech"]
        assert not any(
            event.payload.get("interrupted") is True and event.payload.get("playback_generation") == 1
            for event in seen
        )

    asyncio.run(run())


def test_text_engine_raw_audio_without_confirmed_speech_does_not_barge_in(monkeypatch):
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

        while True:
            event = await anext(engine.events())
            if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL:
                break

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload=AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"packet-without-speech").to_payload(),
            )
        )

        await engine.close()

        assert oracle.interrupted is False
        assert engine._playback_generation == 1
        assert engine._inbound_audio == [b"packet-without-speech"]

    asyncio.run(run())


def test_text_engine_speech_energy_barge_in_requires_rms_and_duration(monkeypatch):
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
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                barge_in_policy={"min_rms": 350, "min_speech_ms": 120},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "first turn"},
            )
        )

        while True:
            event = await anext(engine.events())
            if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL:
                break

        for sequence, payload in [
            (2, {"user_id": "42", "rms": 120, "duration_ms": 200}),
            (3, {"user_id": "42", "rms": 512, "duration_ms": 80}),
        ]:
            await engine.receive_event(
                VoiceEvent(
                    type=VoiceEventType.SPEECH_ENERGY,
                    session_id="voice-123",
                    sequence=sequence,
                    payload=payload,
                )
            )
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(anext(engine.events()), timeout=0.01)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.SPEECH_ENERGY,
                session_id="voice-123",
                sequence=4,
                payload={"user_id": "42", "rms": 512, "duration_ms": 40},
            )
        )

        barge_in = await anext(engine.events())
        await engine.close()

        assert barge_in.type == VoiceEventType.BARGE_IN
        assert barge_in.payload["reason"] == "user_speech"
        assert barge_in.payload["playback_generation"] == 2
        assert barge_in.payload["backend_interrupt_requested"] is True
        assert oracle.interrupted is True

    asyncio.run(run())


def test_text_engine_speech_end_resets_energy_barge_in_accumulator(monkeypatch):
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
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                barge_in_policy={"min_rms": 350, "min_speech_ms": 120},
            )
        )
        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "first turn"},
            )
        )

        while True:
            event = await anext(engine.events())
            if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL:
                break

        for sequence, event_type, payload in [
            (2, VoiceEventType.SPEECH_ENERGY, {"user_id": "42", "rms": 512, "duration_ms": 80}),
            (3, VoiceEventType.SPEECH_END, {"user_id": "42"}),
            (4, VoiceEventType.SPEECH_ENERGY, {"user_id": "42", "rms": 512, "duration_ms": 40}),
        ]:
            await engine.receive_event(
                VoiceEvent(
                    type=event_type,
                    session_id="voice-123",
                    sequence=sequence,
                    payload=payload,
                )
            )

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(anext(engine.events()), timeout=0.01)

        await engine.close()

        assert oracle.interrupted is False
        assert engine._playback_generation == 1

    asyncio.run(run())


def test_text_engine_forwards_speech_lifecycle_events_to_sidecar():
    async def run():
        sidecar = FakeSidecar()
        engine = TextOracleTTSEngine(oracle=FakeOracle(), sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                sidecar_base_url="http://voice.local",
            )
        )
        assert (await anext(engine.events())).type == VoiceEventType.SESSION_STARTED

        for sequence, event_type, payload in [
            (1, VoiceEventType.SPEECH_START, {"user_id": "42"}),
            (2, VoiceEventType.SPEECH_ENERGY, {"user_id": "42", "rms": 512, "duration_ms": 20}),
            (3, VoiceEventType.SPEECH_END, {"user_id": "42"}),
        ]:
            await engine.receive_event(
                VoiceEvent(
                    type=event_type,
                    session_id="voice-123",
                    sequence=sequence,
                    payload=payload,
                )
            )

        await engine.close()

        speech_events = [
            event
            for event in sidecar.received
            if event.type in {VoiceEventType.SPEECH_START, VoiceEventType.SPEECH_ENERGY, VoiceEventType.SPEECH_END}
        ]
        assert [event.type for event in speech_events] == [
            VoiceEventType.SPEECH_START,
            VoiceEventType.SPEECH_ENERGY,
            VoiceEventType.SPEECH_END,
        ]

    asyncio.run(run())


def test_text_engine_forwards_transport_playback_lifecycle_events_to_sidecar():
    async def run():
        sidecar = FakeSidecar()
        engine = TextOracleTTSEngine(oracle=FakeOracle(), sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                sidecar_base_url="http://voice.local",
            )
        )
        assert (await anext(engine.events())).type == VoiceEventType.SESSION_STARTED

        for sequence, event_type in [
            (1, VoiceEventType.PLAYBACK_STARTED),
            (2, VoiceEventType.PLAYBACK_STOPPED),
        ]:
            await engine.receive_event(
                VoiceEvent(
                    type=event_type,
                    session_id="voice-123",
                    sequence=sequence,
                    payload={"playback_generation": 3},
                )
            )

        await engine.close()

        playback_events = [
            event for event in sidecar.received if event.type in {VoiceEventType.PLAYBACK_STARTED, VoiceEventType.PLAYBACK_STOPPED}
        ]
        assert [event.type for event in playback_events] == [
            VoiceEventType.PLAYBACK_STARTED,
            VoiceEventType.PLAYBACK_STOPPED,
        ]
        assert [event.payload["playback_generation"] for event in playback_events] == [3, 3]
        assert engine._playback_generation == 3
        assert engine._frontend_output_active is False

    asyncio.run(run())


def test_text_engine_notifies_sidecar_before_session_close():
    async def run():
        sidecar = FakeSidecar()
        engine = TextOracleTTSEngine(oracle=FakeOracle(), sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                sidecar_base_url="http://voice.local",
            )
        )
        assert (await anext(engine.events())).type == VoiceEventType.SESSION_STARTED

        await engine.close()

        assert sidecar.closed is True
        assert sidecar.received[-1].type == VoiceEventType.SESSION_CLOSED
        assert sidecar.received[-1].payload == {"reason": "closed"}
        assert (await anext(engine.events())).type == VoiceEventType.SESSION_CLOSED

    asyncio.run(run())


def test_text_engine_forwards_client_session_close_to_sidecar():
    async def run():
        sidecar = FakeSidecar()
        engine = TextOracleTTSEngine(oracle=FakeOracle(), sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                sidecar_base_url="http://voice.local",
            )
        )
        assert (await anext(engine.events())).type == VoiceEventType.SESSION_STARTED

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.SESSION_CLOSED,
                session_id="voice-123",
                sequence=9,
                payload={"reason": "client_leave"},
            )
        )

        assert sidecar.closed is True
        assert sidecar.received[-1].type == VoiceEventType.SESSION_CLOSED
        assert sidecar.received[-1].sequence == 9
        assert sidecar.received[-1].payload == {"reason": "client_leave"}
        assert (await anext(engine.events())).type == VoiceEventType.SESSION_CLOSED

    asyncio.run(run())


def test_text_engine_accepts_session_stop_and_emits_session_closed():
    async def run():
        sidecar = FakeSidecar()
        engine = TextOracleTTSEngine(oracle=FakeOracle(), sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                sidecar_base_url="http://voice.local",
            )
        )
        assert (await anext(engine.events())).type == VoiceEventType.SESSION_STARTED

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.SESSION_STOP,
                session_id="voice-123",
                sequence=9,
                payload={"reason": "client_leave"},
            )
        )

        assert sidecar.closed is True
        assert sidecar.received[-1].type == VoiceEventType.SESSION_STOP
        assert sidecar.received[-1].sequence == 9
        assert sidecar.received[-1].payload == {"reason": "client_leave"}
        closed = await anext(engine.events())
        assert closed.type == VoiceEventType.SESSION_CLOSED
        assert closed.payload == {"reason": "closed"}

    asyncio.run(run())


def test_text_engine_auto_barge_in_on_new_speech_while_frontend_output_active():
    class ManualSidecar(FakeSidecar):
        async def send_event(self, event):
            self.received.append(event)

    async def run():
        sidecar = ManualSidecar()
        engine = TextOracleTTSEngine(oracle=FakeOracle(), sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                frontend_provider="sidecar",
                sidecar_base_url="http://voice.local",
            )
        )
        assert (await anext(engine.events())).type == VoiceEventType.SESSION_STARTED

        await sidecar._events.put(
            VoiceEvent(
                type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"frontend-audio").to_payload(),
                    "playback_generation": 7,
                },
            )
        )

        audio = await anext(engine.events())
        assert audio.type == VoiceEventType.AUDIO_OUTPUT_CHUNK
        assert engine._frontend_output_active is True

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=2,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"new-speech").to_payload(),
                    "speech_confirmed": True,
                },
            )
        )

        barge_in = await anext(engine.events())
        await engine.close()

        assert barge_in.type == VoiceEventType.BARGE_IN
        assert barge_in.payload["reason"] == "user_speech"
        assert barge_in.payload["playback_generation"] == 8
        assert barge_in.payload["frontend_cancel_requested"] is True
        assert barge_in.payload["backend_interrupt_requested"] is False
        turn_events = [
            event
            for event in sidecar.received
            if event.type in {VoiceEventType.BARGE_IN, VoiceEventType.AUDIO_INPUT_CHUNK}
        ]
        assert [event.type for event in turn_events] == [
            VoiceEventType.BARGE_IN,
            VoiceEventType.AUDIO_INPUT_CHUNK,
        ]
        assert turn_events[0].payload["playback_generation"] == 8
        assert engine._frontend_output_active is False

    asyncio.run(run())


def test_text_engine_forwards_frontend_commit_and_clears_output_active():
    async def run():
        sidecar = FakeSidecar()
        engine = TextOracleTTSEngine(oracle=FakeOracle(), sidecar=sidecar)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                frontend_provider="sidecar",
                sidecar_base_url="http://voice.local",
            )
        )
        assert (await anext(engine.events())).type == VoiceEventType.SESSION_STARTED

        await sidecar._events.put(
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                session_id="voice-123",
                sequence=1,
                payload={"text": "frontend reply", "playback_generation": 3},
            )
        )
        partial = await anext(engine.events())
        assert partial.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL
        assert engine._frontend_output_active is True

        await sidecar._events.put(
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_COMMIT,
                session_id="voice-123",
                sequence=2,
                payload={"text": "frontend reply", "playback_generation": 3},
            )
        )
        commit = await anext(engine.events())
        await engine.close()

        assert commit.type == VoiceEventType.ASSISTANT_COMMIT
        assert commit.payload["text"] == "frontend reply"
        assert engine._frontend_output_active is False

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


def test_sidecar_client_accepts_binary_assistant_audio_frame(monkeypatch):
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
            type=VoiceEventType.ASSISTANT_AUDIO_CHUNK,
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
        assert event.type == VoiceEventType.ASSISTANT_AUDIO_CHUNK
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
            VoiceEventType.SESSION_STARTED,
            VoiceEventType.FRONTEND_STATE,
            VoiceEventType.TRANSCRIPT_PARTIAL,
            VoiceEventType.TRANSCRIPT_FINAL,
        ]
        assert seen[-1].payload["text"] == "hello hermes"
        assert [event.payload.get("input_generation") for event in seen[2:]] == [5, 5]
        assert [event.payload.get("language") for event in seen[2:]] == ["ja", "ja"]
        assert [event.payload.get("locale") for event in seen[2:]] == ["ja-JP", "ja-JP"]
        assert [event.payload.get("script") for event in seen[2:]] == ["Jpan", "Jpan"]

    asyncio.run(run())


def test_reference_sidecar_session_started_matches_realtime_contract():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                input_codec=VoiceAudioCodec.WEBM_OPUS,
                output_codec=VoiceAudioCodec.OPUS,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                routing_policy={"local_confidence_threshold": 0.75},
                metrics_policy={"enabled": True},
                quality_targets_ms={"kame_speech_end_to_playback_start_ms": 2500},
                barge_in_policy={"min_rms": 350, "min_speech_ms": 120},
            )
        )
        started = await asyncio.wait_for(anext(sidecar.events()), timeout=1)
        ready = await asyncio.wait_for(anext(sidecar.events()), timeout=1)
        await sidecar.close()
        return started, ready

    started, ready = asyncio.run(run())
    assert started.type == VoiceEventType.SESSION_STARTED
    assert started.payload["engine"] == "kame_interface_oracle"
    assert started.payload["input_codec"] == "webm_opus"
    assert started.payload["output_codec"] == "opus"
    assert started.payload["frontend_provider"] == "gemma4"
    assert started.payload["frontend_model"] == "gemma-4-E2B-it"
    assert started.payload["sidecar"] is True
    assert started.payload["routing"] == {"local_confidence_threshold": 0.75}
    assert started.payload["metrics"] == {"enabled": True}
    assert started.payload["quality_targets_ms"] == {"kame_speech_end_to_playback_start_ms": 2500}
    assert started.payload["barge_in"] == {"min_rms": 350, "min_speech_ms": 120}
    assert ready.type == VoiceEventType.FRONTEND_STATE


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


def test_reference_sidecar_barge_in_finalizes_active_playback_generations():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        await sidecar.start(RealtimeVoiceSessionConfig(session_id="voice-123", frontend_provider="local"))
        sidecar._active_playback_generations.add(4)

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
            if event.type == VoiceEventType.ASSISTANT_AUDIO_END:
                await sidecar.close()
                break

        barge_in = next(event for event in seen if event.type == VoiceEventType.BARGE_IN)
        stopped = next(event for event in seen if event.type == VoiceEventType.PLAYBACK_STOPPED)
        audio_end = next(event for event in seen if event.type == VoiceEventType.ASSISTANT_AUDIO_END)
        assert barge_in.payload["playback_generation"] == 4
        assert stopped.payload == {
            "reason": "barge_in",
            "interrupted": True,
            "barge_in_reason": "user_speech",
            "playback_generation": 4,
        }
        assert audio_end.payload == stopped.payload
        assert sidecar._active_playback_generations == set()

    asyncio.run(run())


def test_reference_sidecar_forwards_provider_kame_oracle_events():
    class Provider:
        def __init__(self, events):
            self._events = events

        async def events(self):
            for event in self._events:
                yield event

    async def collect_forwarded(provider_attr: str, consume_method: str):
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        await sidecar.start(RealtimeVoiceSessionConfig(session_id="voice-123", frontend_provider="local"))
        started = await asyncio.wait_for(sidecar._events.get(), timeout=1)
        assert started.type == VoiceEventType.SESSION_STARTED
        ready = await asyncio.wait_for(sidecar._events.get(), timeout=1)
        assert ready.type == VoiceEventType.FRONTEND_STATE
        source_events = [
            VoiceEvent(
                type=VoiceEventType.ORACLE_ACCEPTED,
                session_id="voice-123",
                sequence=1,
                payload={"turn_id": "turn-1", "playback_generation": 3},
            ),
            VoiceEvent(
                type=VoiceEventType.ORACLE_TOOL_CALL,
                session_id="voice-123",
                sequence=2,
                payload={
                    "turn_id": "turn-1",
                    "tool_name": "read_file",
                    "tool_call_id": "call-1",
                    "playback_generation": 3,
                },
            ),
            VoiceEvent(
                type=VoiceEventType.ORACLE_TOOL_RESULT,
                session_id="voice-123",
                sequence=3,
                payload={
                    "turn_id": "turn-1",
                    "tool_name": "read_file",
                    "tool_call_id": "call-1",
                    "result": "ok",
                    "playback_generation": 3,
                },
            ),
            VoiceEvent(
                type=VoiceEventType.ORACLE_RESPONSE_FINAL,
                session_id="voice-123",
                sequence=4,
                payload={"turn_id": "turn-1", "text": "Done.", "playback_generation": 3},
            ),
            VoiceEvent(
                type=VoiceEventType.INTERFACE_INTENT_FINAL,
                session_id="voice-123",
                sequence=5,
                payload={"turn_id": "turn-2", "intent": "Say hello.", "route": "local"},
            ),
            VoiceEvent(
                type=VoiceEventType.INTERFACE_REPLY_LOCAL,
                session_id="voice-123",
                sequence=6,
                payload={"turn_id": "turn-2", "text": "Hello.", "playback_generation": 4},
            ),
            VoiceEvent(
                type=VoiceEventType.INTERFACE_ORACLE_REQUEST,
                session_id="voice-123",
                sequence=7,
                payload={"turn_id": "turn-3", "intent": "Check the file.", "playback_generation": 5},
            ),
            VoiceEvent(
                type=VoiceEventType.ORACLE_JOB_ACCEPTED,
                session_id="voice-123",
                sequence=8,
                payload={
                    "job_id": "voice-oracle-001",
                    "state": "queued",
                    "intent": "Check the file.",
                    "playback_generation": 5,
                },
            ),
            VoiceEvent(
                type=VoiceEventType.ORACLE_JOB_STARTED,
                session_id="voice-123",
                sequence=9,
                payload={
                    "job_id": "voice-oracle-001",
                    "state": "running",
                    "intent": "Check the file.",
                    "playback_generation": 5,
                },
            ),
            VoiceEvent(
                type=VoiceEventType.ORACLE_JOB_COMPLETED,
                session_id="voice-123",
                sequence=10,
                payload={
                    "job_id": "voice-oracle-001",
                    "state": "completed",
                    "result_summary": "Done.",
                    "playback_generation": 5,
                },
            ),
            VoiceEvent(
                type=VoiceEventType.INTERFACE_COMMIT,
                session_id="voice-123",
                sequence=11,
                payload={"turn_id": "turn-2", "text": "Hello.", "local_reply": True},
            ),
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_CAPTION_FINAL,
                session_id="voice-123",
                sequence=12,
                payload={"text": "Hello.", "playback_generation": 4},
            ),
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_AUDIO_CHUNK,
                session_id="voice-123",
                sequence=13,
                payload={"codec": "opus", "data_b64": "", "playback_generation": 4},
            ),
            VoiceEvent(
                type=VoiceEventType.SESSION_METRICS,
                session_id="voice-123",
                sequence=14,
                payload={"metrics": {"kame_oracle_called": 1}},
            ),
        ]
        setattr(sidecar, provider_attr, Provider(source_events))

        await getattr(sidecar, consume_method)()

        forwarded = []
        for _ in source_events:
            forwarded.append(await asyncio.wait_for(sidecar._events.get(), timeout=1))
        assert [event.type for event in forwarded] == [event.type for event in source_events]
        assert [event.payload for event in forwarded] == [event.payload for event in source_events]

    async def run():
        await collect_forwarded("_openai_realtime", "_consume_openai_realtime_events")
        await collect_forwarded("_gemini_live", "_consume_gemini_live_events")

    asyncio.run(run())


def test_reference_sidecar_records_inbound_kame_feedback_events():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        await sidecar.start(RealtimeVoiceSessionConfig(session_id="voice-123", frontend_provider="local"))
        assert (await anext(sidecar.events())).type == VoiceEventType.SESSION_STARTED
        assert (await anext(sidecar.events())).type == VoiceEventType.FRONTEND_STATE

        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.INTERFACE_ORACLE_REQUEST,
                session_id="voice-123",
                sequence=1,
                payload={"turn_id": "voice-123:1", "route": "defer", "playback_generation": 3},
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.ORACLE_JOB_STARTED,
                session_id="voice-123",
                sequence=2,
                payload={
                    "job_id": "voice-oracle-001",
                    "state": "running",
                    "spoken_status": "Checking the file.",
                    "playback_generation": 3,
                },
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.ORACLE_RESPONSE_FINAL,
                session_id="voice-123",
                sequence=3,
                payload={"turn_id": "voice-123:1", "text": "Done.", "playback_generation": 3},
            )
        )

        assert [record["type"] for record in sidecar._kame_feedback_events] == [
            VoiceEventType.INTERFACE_ORACLE_REQUEST.value,
            VoiceEventType.ORACLE_JOB_STARTED.value,
            VoiceEventType.ORACLE_RESPONSE_FINAL.value,
        ]
        assert [record["type"] for record in sidecar._kame_feedback_events_by_generation[3]] == [
            VoiceEventType.INTERFACE_ORACLE_REQUEST.value,
            VoiceEventType.ORACLE_JOB_STARTED.value,
            VoiceEventType.ORACLE_RESPONSE_FINAL.value,
        ]
        assert sidecar._kame_last_interface_event == {
            "type": VoiceEventType.INTERFACE_ORACLE_REQUEST.value,
            "payload": {"turn_id": "voice-123:1", "route": "defer", "playback_generation": 3},
        }
        assert sidecar._kame_last_oracle_event == {
            "type": VoiceEventType.ORACLE_RESPONSE_FINAL.value,
            "payload": {"turn_id": "voice-123:1", "text": "Done.", "playback_generation": 3},
        }

        await sidecar.close()

    asyncio.run(run())


def test_reference_sidecar_forwards_oracle_feedback_to_live_frontends():
    class FakeLiveFrontend:
        def __init__(self):
            self.received = []

        async def receive_event(self, event):
            self.received.append(event)

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        sidecar.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            frontend_provider="gemma4",
        )
        openai = FakeLiveFrontend()
        gemini = FakeLiveFrontend()
        sidecar._openai_realtime = openai
        sidecar._gemini_live = gemini

        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.ORACLE_HINT,
                session_id="voice-123",
                sequence=1,
                payload={
                    "turn_id": "voice-123:3",
                    "delta": "Hermes found the deployment note.",
                    "playback_generation": 3,
                },
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.ORACLE_JOB_STARTED,
                session_id="voice-123",
                sequence=2,
                payload={
                    "job_id": "voice-oracle-001",
                    "state": "running",
                    "spoken_status": "Checking the deployment note.",
                    "playback_generation": 3,
                },
            )
        )

        assert [event.type for event in openai.received] == [
            VoiceEventType.ORACLE_HINT,
            VoiceEventType.ORACLE_JOB_STARTED,
        ]
        assert [event.type for event in gemini.received] == [
            VoiceEventType.ORACLE_HINT,
            VoiceEventType.ORACLE_JOB_STARTED,
        ]
        assert sidecar._kame_last_oracle_event == {
            "type": VoiceEventType.ORACLE_JOB_STARTED.value,
            "payload": {
                "job_id": "voice-oracle-001",
                "state": "running",
                "spoken_status": "Checking the deployment note.",
                "playback_generation": 3,
            },
        }

    asyncio.run(run())


def test_reference_sidecar_clears_kame_feedback_on_barge_in_and_close():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        await sidecar.start(RealtimeVoiceSessionConfig(session_id="voice-123", frontend_provider="local"))
        assert (await anext(sidecar.events())).type == VoiceEventType.SESSION_STARTED
        assert (await anext(sidecar.events())).type == VoiceEventType.FRONTEND_STATE

        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.ORACLE_RESPONSE_FINAL,
                session_id="voice-123",
                sequence=1,
                payload={"turn_id": "voice-123:1", "text": "Done.", "playback_generation": 3},
            )
        )
        assert sidecar._kame_feedback_events

        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.BARGE_IN,
                session_id="voice-123",
                sequence=2,
                payload={"reason": "user_speech", "playback_generation": 4},
            )
        )
        assert sidecar._kame_feedback_events == []
        assert sidecar._kame_feedback_events_by_generation == {}
        assert sidecar._kame_last_interface_event is None
        assert sidecar._kame_last_oracle_event is None

        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.INTERFACE_COMMIT,
                session_id="voice-123",
                sequence=3,
                payload={"turn_id": "voice-123:2", "text": "Fresh.", "playback_generation": 5},
            )
        )
        assert sidecar._kame_feedback_events
        await sidecar.close()
        assert sidecar._kame_feedback_events == []
        assert sidecar._kame_feedback_events_by_generation == {}
        assert sidecar._kame_last_interface_event is None
        assert sidecar._kame_last_oracle_event is None

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
        assert (await anext(sidecar.events())).type == VoiceEventType.SESSION_STARTED
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
        audio = tmp_path / "speech.wav"
        _write_test_wav(audio)
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
        audio_chunk = AudioChunk.from_payload(audio_events[0].payload)
        assert audio_chunk.codec == VoiceAudioCodec.PCM16
        assert audio_chunk.sample_rate_hz == 16000
        assert audio_chunk.channels == 1
        assert audio_events[0].payload["playback_generation"] == 7
        assert audio_events[0].payload["metrics"]["tts_synthesis_ms"] >= 0

    asyncio.run(run())


def test_reference_sidecar_reports_vllm_as_active_kame_reflex_with_asr_evidence_bridge():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(
                vllm_base_url="http://vllm.local:8000/v1",
                vllm_model="google/gemma-4-E2B-it",
                streaming_stt_base_url="http://streaming-stt.local:9000",
                streaming_stt_model="nemotron-speech",
            )
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="configured-alias",
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
            )
        )

        started = await asyncio.wait_for(anext(sidecar.events()), timeout=1)
        event = await asyncio.wait_for(anext(sidecar.events()), timeout=1)
        await sidecar.close()
        return started, event

    started, event = asyncio.run(run())
    assert started.type == VoiceEventType.SESSION_STARTED
    assert started.payload["frontend_model"] == "configured-alias"
    assert event.type == VoiceEventType.FRONTEND_STATE
    assert event.payload["status"] == "ready"
    assert event.payload["provider"] == "gemma4"
    assert event.payload["implementation_provider"] == "vllm"
    assert event.payload["model"] == "configured-alias"
    assert event.payload["streaming_stt"] is False
    assert event.payload["vllm"] is True
    assert event.payload["interface_audio_input"] == "native_audio"


def test_reference_sidecar_uses_session_interface_endpoint_for_kame_reflex():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(vllm_base_url=None, vllm_model=None)
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="google/gemma-4-E2B-it",
                interface_base_url="http://session-vllm.local:8000/v1",
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
            )
        )

        started = await asyncio.wait_for(anext(sidecar.events()), timeout=1)
        event = await asyncio.wait_for(anext(sidecar.events()), timeout=1)
        runtime = sidecar.runtime
        await sidecar.close()
        return started, event, runtime

    started, event, runtime = asyncio.run(run())
    assert started.type == VoiceEventType.SESSION_STARTED
    assert event.type == VoiceEventType.FRONTEND_STATE
    assert event.payload["status"] == "ready"
    assert event.payload["provider"] == "gemma4"
    assert event.payload["implementation_provider"] == "vllm"
    assert event.payload["model"] == "google/gemma-4-E2B-it"
    assert event.payload["vllm"] is True
    assert runtime.vllm_base_url == "http://session-vllm.local:8000/v1"
    assert runtime.vllm_model == "google/gemma-4-E2B-it"


def test_reference_sidecar_runtime_with_session_config_scopes_endpoint_fields():
    runtime = ReferenceSidecarRuntimeConfig(
        interface_provider="runtime-interface",
        vllm_base_url="http://runtime-interface.local:8000/v1",
        vllm_model="runtime-interface-model",
        streaming_stt_provider="runtime-asr",
        streaming_stt_base_url="http://runtime-asr.local:8767",
        streaming_stt_model="runtime-asr-model",
        streaming_tts_provider="runtime-tts",
        streaming_tts_base_url="http://runtime-tts.local:8768",
        streaming_tts_model="runtime-tts-model",
    )
    config = RealtimeVoiceSessionConfig(
        session_id="voice-123",
        frontend_provider="session-interface",
        interface_base_url="http://session-interface.local:8000/v1",
        frontend_model="session-interface-model",
        asr_provider="session-asr",
        asr_base_url="http://session-asr.local:8767",
        asr_model="session-asr-model",
        tts_provider="session-tts",
        tts_base_url="http://session-tts.local:8768",
        tts_model="session-tts-model",
    )

    scoped = reference_sidecar_module._runtime_with_session_config(runtime, config)

    assert scoped.interface_provider == "session-interface"
    assert scoped.vllm_base_url == "http://session-interface.local:8000/v1"
    assert scoped.vllm_model == "session-interface-model"
    assert scoped.streaming_stt_provider == "session-asr"
    assert scoped.streaming_stt_base_url == "http://session-asr.local:8767"
    assert scoped.streaming_stt_model == "session-asr-model"
    assert scoped.streaming_tts_provider == "session-tts"
    assert scoped.streaming_tts_base_url == "http://session-tts.local:8768"
    assert scoped.streaming_tts_model == "session-tts-model"
    assert runtime.interface_provider == "runtime-interface"
    assert runtime.vllm_base_url == "http://runtime-interface.local:8000/v1"


def test_reference_sidecar_reports_kame_audio_reflex_fallback_without_vllm():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(vllm_base_url=None, vllm_model=None)
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.FALLBACK,
            )
        )

        started = await asyncio.wait_for(anext(sidecar.events()), timeout=1)
        event = await asyncio.wait_for(anext(sidecar.events()), timeout=1)
        await sidecar.close()
        return started, event

    started, event = asyncio.run(run())
    assert started.type == VoiceEventType.SESSION_STARTED
    assert event.type == VoiceEventType.FRONTEND_STATE
    assert event.payload["status"] == "fallback"
    assert event.payload["reason"] == "kame_audio_reflex_unavailable"
    assert event.payload["requested_provider"] == "gemma4"
    assert event.payload["provider"] == "local_stt"
    assert event.payload["fallback_provider"] == "local_stt"
    assert event.payload["intent_source"] == "asr_fallback"
    assert event.payload["transcript_source"] == "asr"
    assert event.payload["interface_audio_input"] == "native_audio"


def test_reference_sidecar_auto_requires_native_audio_without_explicit_fallback():
    transcribe_calls = []

    def fake_transcribe(path):
        transcribe_calls.append(path)
        raise AssertionError("on_escalation ASR must not drive the KAME reflex")

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(vllm_base_url=None, vllm_model=None),
            transcribe_audio_func=fake_transcribe,
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                interface_audio_input="auto",
                asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"audio").to_payload(),
                    "end_of_utterance": True,
                    "input_generation": 11,
                },
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if event.type == VoiceEventType.SESSION_ERROR:
                await sidecar.close()
                break
        return seen

    seen = asyncio.run(run())
    state = next(event for event in seen if event.type == VoiceEventType.FRONTEND_STATE)
    error = next(event for event in seen if event.type == VoiceEventType.SESSION_ERROR)
    assert state.payload["status"] == "degraded"
    assert state.payload["reason"] == "kame_audio_reflex_unavailable"
    assert state.payload["provider"] == "unavailable"
    assert state.payload["fallback_provider"] == "unavailable"
    assert "intent_source" not in state.payload
    assert "transcript_source" not in state.payload
    assert state.payload["interface_audio_input"] == "auto"
    assert "KAME audio reflex unavailable and ASR reflex fallback is disabled" in error.payload["error"]
    assert transcribe_calls == []


def test_reference_sidecar_auto_starts_streaming_stt_with_explicit_fallback_mode(monkeypatch):
    created = []

    class FakeBridge:
        def __init__(self, *, path="/v1/realtime-text/session"):
            self.path = path
            self._events = asyncio.Queue()
            created.append(self)

        async def start(self, config):
            self.config = config

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
        FakeBridge,
    )

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(
                streaming_stt_base_url="http://streaming-stt.local:9000",
                streaming_stt_model="nemotron-speech",
                vllm_base_url=None,
                vllm_model=None,
            )
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                interface_audio_input="auto",
                asr_mode=RealtimeVoiceASRMode.FALLBACK,
            )
        )

        started = await asyncio.wait_for(anext(sidecar.events()), timeout=1)
        state = await asyncio.wait_for(anext(sidecar.events()), timeout=1)
        await sidecar.close()
        return started, state

    started, state = asyncio.run(run())
    assert started.type == VoiceEventType.SESSION_STARTED
    assert len(created) == 1
    assert created[0].path == "/v1/streaming-stt/session"
    assert created[0].config.sidecar_base_url == "http://streaming-stt.local:9000"
    assert created[0].config.frontend_model == "nemotron-speech"
    assert state.type == VoiceEventType.FRONTEND_STATE
    assert state.payload["status"] == "fallback"
    assert state.payload["reason"] == "kame_auto_text_fallback_selected"
    assert state.payload["provider"] == "streaming_stt"
    assert state.payload["fallback_provider"] == "streaming_stt"
    assert state.payload["intent_source"] == "asr_fallback"
    assert state.payload["transcript_source"] == "asr"
    assert state.payload["streaming_stt"] is True
    assert state.payload["interface_audio_input"] == "auto"


def test_reference_sidecar_reports_explicit_kame_text_fallback_even_with_vllm():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(
                vllm_base_url="http://vllm.local:8000/v1",
                vllm_model="google/gemma-4-E2B-it",
            )
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                interface_audio_input="text_fallback",
                asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
            )
        )

        started = await asyncio.wait_for(anext(sidecar.events()), timeout=1)
        event = await asyncio.wait_for(anext(sidecar.events()), timeout=1)
        await sidecar.close()
        return started, event

    started, event = asyncio.run(run())
    assert started.type == VoiceEventType.SESSION_STARTED
    assert event.type == VoiceEventType.FRONTEND_STATE
    assert event.payload["status"] == "fallback"
    assert event.payload["reason"] == "kame_text_fallback_requested"
    assert event.payload["requested_provider"] == "gemma4"
    assert event.payload["provider"] == "local_stt"
    assert event.payload["fallback_provider"] == "local_stt"
    assert event.payload["intent_source"] == "asr_fallback"
    assert event.payload["transcript_source"] == "asr"
    assert event.payload["interface_audio_input"] == "text_fallback"


def test_reference_sidecar_kame_on_escalation_reports_degraded_without_audio_reflex():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(vllm_base_url=None, vllm_model=None)
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
            )
        )

        started = await asyncio.wait_for(anext(sidecar.events()), timeout=1)
        event = await asyncio.wait_for(anext(sidecar.events()), timeout=1)
        await sidecar.close()
        return started, event

    started, event = asyncio.run(run())
    assert started.type == VoiceEventType.SESSION_STARTED
    assert event.type == VoiceEventType.FRONTEND_STATE
    assert event.payload["status"] == "degraded"
    assert event.payload["reason"] == "kame_audio_reflex_unavailable"
    assert event.payload["requested_provider"] == "gemma4"
    assert event.payload["provider"] == "unavailable"
    assert event.payload["fallback_provider"] == "unavailable"
    assert "intent_source" not in event.payload
    assert "transcript_source" not in event.payload
    assert event.payload["interface_audio_input"] == "native_audio"
    assert event.payload["asr_mode"] == "on_escalation"


def test_reference_sidecar_labels_kame_local_stt_fallback_as_asr_evidence():
    def fake_transcribe(path):
        assert path
        return {"success": True, "transcript": "check deployment status"}

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(vllm_base_url=None, vllm_model=None),
            transcribe_audio_func=fake_transcribe,
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.FALLBACK,
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"audio").to_payload(),
                    "end_of_utterance": True,
                    "input_generation": 11,
                },
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if event.type == VoiceEventType.INTERFACE_INTENT_FINAL:
                await sidecar.close()
                break
        return seen

    seen = asyncio.run(run())
    final = next(event for event in seen if event.type == VoiceEventType.INTERFACE_INTENT_FINAL)
    assert final.payload["text"] == "check deployment status"
    assert final.payload["intent"] == "check deployment status"
    assert final.payload["intent_source"] == "asr_fallback"
    assert final.payload["route"] == "oracle_direct"
    assert final.payload["transcript"] == "check deployment status"
    assert final.payload["transcript_source"] == "asr"
    assert final.payload["interface_audio_input_fallback"] is True
    assert final.payload["input_generation"] == 11


def test_reference_sidecar_falls_back_to_local_stt_when_kame_vllm_reflex_fails(monkeypatch):
    def failing_urlopen(req, timeout):
        raise OSError("vLLM connection refused")

    def fake_transcribe(path):
        assert path
        return {"success": True, "transcript": "check deployment status"}

    monkeypatch.setattr("agent.realtime_voice_reference_sidecar.urllib.request.urlopen", failing_urlopen)

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(
                vllm_base_url="http://vllm.local:8000/v1",
                vllm_model="google/gemma-4-E2B-it",
            ),
            transcribe_audio_func=fake_transcribe,
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="google/gemma-4-E2B-it",
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.FALLBACK,
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"audio").to_payload(),
                    "end_of_utterance": True,
                    "input_generation": 12,
                },
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if event.type == VoiceEventType.INTERFACE_INTENT_FINAL:
                await sidecar.close()
                break
        return seen

    seen = asyncio.run(run())
    ready = next(
        event
        for event in seen
        if event.type == VoiceEventType.FRONTEND_STATE and event.payload.get("status") == "ready"
    )
    fallback = next(
        event
        for event in seen
        if event.type == VoiceEventType.FRONTEND_STATE and event.payload.get("reason") == "kame_audio_reflex_failed"
    )
    final = next(event for event in seen if event.type == VoiceEventType.INTERFACE_INTENT_FINAL)
    assert ready.payload["status"] == "ready"
    assert ready.payload["provider"] == "gemma4"
    assert ready.payload["implementation_provider"] == "vllm"
    assert ready.payload["model"] == "google/gemma-4-E2B-it"
    assert ready.payload["vllm"] is True
    assert fallback.payload["status"] == "fallback"
    assert fallback.payload["provider"] == "local_stt"
    assert fallback.payload["requested_provider"] == "gemma4"
    assert "vLLM connection refused" in fallback.payload["error"]
    assert final.payload["text"] == "check deployment status"
    assert final.payload["intent_source"] == "asr_fallback"
    assert final.payload["route"] == "oracle_direct"
    assert final.payload["transcript_source"] == "asr"
    assert final.payload["fallback_reason"] == "kame_audio_reflex_failed"
    assert "vLLM connection refused" in final.payload["fallback_error"]
    assert final.payload["input_generation"] == 12


def test_reference_sidecar_kame_vllm_failure_respects_nonfallback_asr_mode(monkeypatch):
    transcribe_calls = []

    def failing_urlopen(req, timeout):
        raise OSError("vLLM connection refused")

    def fake_transcribe(path):
        transcribe_calls.append(path)
        raise AssertionError("local STT must not drive the reflex in on_escalation mode")

    monkeypatch.setattr("agent.realtime_voice_reference_sidecar.urllib.request.urlopen", failing_urlopen)

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(
                vllm_base_url="http://vllm.local:8000/v1",
                vllm_model="google/gemma-4-E2B-it",
            ),
            transcribe_audio_func=fake_transcribe,
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="google/gemma-4-E2B-it",
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"audio").to_payload(),
                    "end_of_utterance": True,
                    "input_generation": 12,
                },
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if event.type == VoiceEventType.SESSION_ERROR:
                await sidecar.close()
                break
        return seen

    seen = asyncio.run(run())
    error = next(event for event in seen if event.type == VoiceEventType.SESSION_ERROR)
    assert "ASR reflex fallback is disabled" in error.payload["error"]
    assert "vLLM connection refused" in error.payload["error"]
    assert transcribe_calls == []


def test_reference_sidecar_passes_language_metadata_to_tts_callback(tmp_path):
    captured = {}

    def fake_synthesize(text, *, metadata=None):
        captured["text"] = text
        captured["metadata"] = metadata
        audio = tmp_path / "speech.wav"
        _write_test_wav(audio)
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


def test_reference_sidecar_does_not_cache_acknowledgement_audio(tmp_path):
    synth_calls = []

    def fake_synthesize(text):
        synth_calls.append(text)
        audio = tmp_path / f"speech-{len(synth_calls)}.wav"
        pcm = bytes([len(synth_calls), 0, len(synth_calls) + 1, 0])
        _write_test_wav(audio, pcm)
        return {"success": True, "file_path": str(audio)}

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(vllm_base_url=None, vllm_model=None),
            synthesize_func=fake_synthesize,
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                frontend_provider="local",
                metadata={"turn_acknowledgement": {"enabled": True, "text": "One moment."}},
            )
        )
        assert synth_calls == []

        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                session_id="voice-123",
                sequence=1,
                payload={"text": "One moment.", "speak": True, "playback_generation": 7},
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                session_id="voice-123",
                sequence=2,
                payload={"text": "One moment.", "speak": True, "playback_generation": 8},
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if sum(1 for item in seen if item.type == VoiceEventType.AUDIO_OUTPUT_CHUNK) == 2:
                await sidecar.close()
                break

        audio_events = [event for event in seen if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK]
        assert synth_calls == ["One moment.", "One moment."]
        first = AudioChunk.from_payload(audio_events[0].payload)
        second = AudioChunk.from_payload(audio_events[1].payload)
        assert first.codec == VoiceAudioCodec.PCM16
        assert second.codec == VoiceAudioCodec.PCM16
        assert first.data == b"\x01\x00\x02\x00"
        assert second.data == b"\x02\x00\x03\x00"
        assert "tts_cache" not in audio_events[0].payload["metrics"]
        assert audio_events[0].payload["metrics"]["tts_synthesis_ms"] >= 0

    asyncio.run(run())


def test_reference_sidecar_kame_local_tts_reports_playback_start_metric(tmp_path):
    def fake_synthesize(text):
        audio = tmp_path / "speech.wav"
        _write_test_wav(audio)
        return {"success": True, "file_path": str(audio)}

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(vllm_base_url=None, vllm_model=None),
            synthesize_func=fake_synthesize,
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="local",
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                session_id="voice-123",
                sequence=1,
                payload={
                    "text": "One moment.",
                    "speak": True,
                    "playback_generation": 7,
                    "voice_architecture": "kame_frontend_oracle",
                    "kame_route": KameRoute.DEFER.value,
                    "metrics": {
                        "kame_speech_end_to_interface_decision_ms": 25,
                        "kame_speech_end_to_first_audio_ms": 120,
                    },
                },
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK:
                await sidecar.close()
                break

        playback = next(event for event in seen if event.type == VoiceEventType.PLAYBACK_STARTED)
        audio = next(event for event in seen if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK)
        assert playback.payload["metrics"]["kame_first_tts_audio_to_playback_start_ms"] >= 0
        assert playback.payload["metrics"]["kame_speech_end_to_playback_start_ms"] >= 120
        assert audio.payload["metrics"]["kame_first_tts_audio_to_playback_start_ms"] >= 0
        assert audio.payload["metrics"]["kame_speech_end_to_playback_start_ms"] >= 120
        assert audio.payload["metrics"]["tts_synthesis_ms"] >= 0
        chunk = AudioChunk.from_payload(audio.payload)
        assert chunk.codec == VoiceAudioCodec.PCM16
        assert chunk.sample_rate_hz == 16000
        assert chunk.channels == 1

    asyncio.run(run())


def test_reference_sidecar_kame_reflex_narration_reports_playback_start_metric(tmp_path):
    def fake_synthesize(text):
        audio = tmp_path / "speech.wav"
        _write_test_wav(audio)
        return {"success": True, "file_path": str(audio)}

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(vllm_base_url=None, vllm_model=None),
            synthesize_func=fake_synthesize,
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="local",
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                session_id="voice-123",
                sequence=1,
                payload={
                    "text": "One moment.",
                    "speak": True,
                    "playback_generation": 7,
                    "voice_architecture": "kame_frontend_oracle",
                    "kame_route": KameRoute.DEFER.value,
                    "metrics": {"kame_speech_end_to_first_audio_ms": 80},
                },
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK:
                await sidecar.close()
                break

        playback = next(event for event in seen if event.type == VoiceEventType.PLAYBACK_STARTED)
        audio = next(event for event in seen if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK)
        assert playback.payload["metrics"]["kame_first_tts_audio_to_playback_start_ms"] >= 0
        assert playback.payload["metrics"]["kame_speech_end_to_playback_start_ms"] >= 80
        assert audio.payload["metrics"]["kame_first_tts_audio_to_playback_start_ms"] >= 0
        assert audio.payload["metrics"]["kame_speech_end_to_playback_start_ms"] >= 80
        assert "tts_cache" not in audio.payload["metrics"]

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
            VoiceEventType.SESSION_STARTED,
            VoiceEventType.FRONTEND_STATE,
            VoiceEventType.SESSION_CLOSED,
        ]
        assert all(event.payload.get("text") != "late transcript" for event in seen)

    asyncio.run(run())


def test_reference_sidecar_accepts_session_stop_event():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        await sidecar.start(RealtimeVoiceSessionConfig(session_id="voice-123", frontend_provider="local"))

        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.SESSION_STOP,
                session_id="voice-123",
                sequence=1,
                payload={"reason": "client_leave"},
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)

        assert [event.type for event in seen] == [
            VoiceEventType.SESSION_STARTED,
            VoiceEventType.FRONTEND_STATE,
            VoiceEventType.SESSION_CLOSED,
        ]
        assert seen[-1].payload == {"reason": "closed"}

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
            VoiceEventType.SESSION_STARTED,
            VoiceEventType.FRONTEND_STATE,
            VoiceEventType.SESSION_CLOSED,
        ]
        assert task not in sidecar._active_tasks
        assert not task.done()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())


def test_reference_sidecar_close_clears_provider_and_session_state():
    class DummyProvider:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    async def sleeper():
        await asyncio.Event().wait()

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        await sidecar.start(RealtimeVoiceSessionConfig(session_id="voice-123", frontend_provider="local"))

        streaming_stt = DummyProvider()
        streaming_tts = DummyProvider()
        openai_realtime = DummyProvider()
        gemini_live = DummyProvider()
        streaming_stt_task = asyncio.create_task(sleeper())
        streaming_tts_task = asyncio.create_task(sleeper())
        openai_realtime_task = asyncio.create_task(sleeper())
        gemini_live_task = asyncio.create_task(sleeper())
        sidecar._streaming_stt = streaming_stt
        sidecar._streaming_tts = streaming_tts
        sidecar._openai_realtime = openai_realtime
        sidecar._gemini_live = gemini_live
        sidecar._streaming_stt_task = streaming_stt_task
        sidecar._streaming_tts_task = streaming_tts_task
        sidecar._openai_realtime_task = openai_realtime_task
        sidecar._gemini_live_task = gemini_live_task
        sidecar._audio.append(b"stale-audio")
        sidecar._audio_bytes = len(b"stale-audio")
        sidecar._audio_input_generation = 3
        sidecar._asr_hypotheses_by_generation[3] = {"asr_transcript": "stale hypothesis"}
        sidecar._active_playback_generations.add(9)
        sidecar._last_speech_lifecycle_event = {"event": "speech.energy", "input_generation": 3}
        sidecar._last_streaming_tts_failure = {"reason": "streaming_tts_send_failed", "error": "failed"}

        await sidecar.close()

        seen = []
        async for event in sidecar.events():
            seen.append(event)

        assert streaming_stt.closed is True
        assert streaming_tts.closed is True
        assert openai_realtime.closed is True
        assert gemini_live.closed is True
        assert streaming_stt_task.cancelled()
        assert streaming_tts_task.cancelled()
        assert openai_realtime_task.cancelled()
        assert gemini_live_task.cancelled()
        assert sidecar._streaming_stt is None
        assert sidecar._streaming_tts is None
        assert sidecar._openai_realtime is None
        assert sidecar._gemini_live is None
        assert sidecar._streaming_stt_task is None
        assert sidecar._streaming_tts_task is None
        assert sidecar._openai_realtime_task is None
        assert sidecar._gemini_live_task is None
        assert sidecar._audio == []
        assert sidecar._audio_bytes == 0
        assert sidecar._audio_input_generation is None
        assert sidecar._asr_hypotheses_by_generation == {}
        assert sidecar._active_playback_generations == set()
        assert sidecar._last_speech_lifecycle_event is None
        assert sidecar._last_streaming_tts_failure is None
        assert [event.type for event in seen] == [
            VoiceEventType.SESSION_STARTED,
            VoiceEventType.FRONTEND_STATE,
            VoiceEventType.PLAYBACK_STOPPED,
            VoiceEventType.ASSISTANT_AUDIO_END,
            VoiceEventType.SESSION_CLOSED,
        ]
        assert seen[2].payload == {"reason": "session_closed", "playback_generation": 9}
        assert seen[3].payload == {"reason": "session_closed", "playback_generation": 9}

    asyncio.run(run())


def test_reference_sidecar_close_bounds_stuck_provider_shutdown(monkeypatch):
    class StuckProvider:
        def __init__(self):
            self.close_started = False
            self.close_cancelled = False

        async def close(self):
            self.close_started = True
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.close_cancelled = True
                raise

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        await sidecar.start(RealtimeVoiceSessionConfig(session_id="voice-123", frontend_provider="local"))
        provider = StuckProvider()
        sidecar._streaming_tts = provider
        sidecar._active_playback_generations.add(11)

        monkeypatch.setattr(
            "agent.realtime_voice_reference_sidecar.REFERENCE_SIDECAR_PROVIDER_CLOSE_TIMEOUT_SECONDS",
            0.001,
        )

        await asyncio.wait_for(sidecar.close(), timeout=1)

        seen = []
        async for event in sidecar.events():
            seen.append(event)

        assert provider.close_started is True
        assert provider.close_cancelled is True
        assert sidecar._streaming_tts is None
        assert sidecar._active_playback_generations == set()
        assert [event.type for event in seen] == [
            VoiceEventType.SESSION_STARTED,
            VoiceEventType.FRONTEND_STATE,
            VoiceEventType.PLAYBACK_STOPPED,
            VoiceEventType.ASSISTANT_AUDIO_END,
            VoiceEventType.SESSION_CLOSED,
        ]
        assert seen[2].payload == {"reason": "session_closed", "playback_generation": 11}
        assert seen[3].payload == {"reason": "session_closed", "playback_generation": 11}

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
        captured["authorization"] = req.get_header("Authorization")
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


def test_reference_sidecar_vllm_kame_audio_reflex(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return (
                b'{"choices":[{"message":{"content":"'
                b'{\\"route\\":\\"oracle_direct\\",'
                b'\\"intent\\":\\"Find the note from yesterday.\\",'
                b'\\"text\\":\\"find the note from yesterday\\",'
                b'\\"transcript\\":\\"find the node from yesterday\\",'
                b'\\"transcript_confidence\\":0.71}'
                b'"}}]}'
            )

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["body"] = __import__("json").loads(req.data.decode("utf-8"))
        captured["authorization"] = req.get_header("Authorization")
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("agent.realtime_voice_reference_sidecar.urllib.request.urlopen", fake_urlopen)

    sidecar = ReferenceRealtimeVoiceSidecarSession(
        ReferenceSidecarRuntimeConfig(
            vllm_base_url="http://vllm.local:8000/v1",
            vllm_model="google/gemma-4-E2B-it",
            vllm_token="reflex-secret-token",
            vllm_timeout_seconds=12,
        )
    )
    sidecar.config = RealtimeVoiceSessionConfig(
        session_id="voice-123",
        engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        interface_temperature=0.35,
        interface_max_output_tokens=96,
        interface_timeout_seconds=0.6,
        interface_audio_input="native_audio",
        asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
        metadata={
            "routing": {
                "allow_local_greetings": False,
                "allow_local_clarifications": True,
                "require_oracle_for_tools": True,
                "require_oracle_for_memory": True,
                "require_oracle_for_files": True,
                "local_confidence_threshold": 0.82,
            }
        },
    )
    sidecar._track_playback_lifecycle_event(
        VoiceEventType.PLAYBACK_STARTED,
        {"playback_generation": 7},
    )
    sidecar._record_speech_lifecycle_event(
        VoiceEvent(
            type=VoiceEventType.SPEECH_ENERGY,
            session_id="voice-123",
            sequence=1,
            payload={"user_id": "42", "input_generation": 5, "rms": 512, "duration_ms": 140},
        )
    )
    sidecar._record_kame_feedback_event(
        VoiceEvent(
            type=VoiceEventType.INTERFACE_ORACLE_REQUEST,
            session_id="voice-123",
            sequence=2,
            payload={"turn_id": "voice-123:7", "route": "oracle_direct", "playback_generation": 7},
        )
    )
    sidecar._record_kame_feedback_event(
        VoiceEvent(
            type=VoiceEventType.ORACLE_HINT,
            session_id="voice-123",
            sequence=3,
            payload={
                "turn_id": "voice-123:7",
                "delta": "Hermes is checking the deployment notes.",
                "final": False,
                "playback_generation": 7,
            },
        )
    )

    payload = sidecar._understand_audio_sync(b"audio", VoiceAudioCodec.WEBM_OPUS)

    metrics = payload.pop("metrics")
    assert metrics["kame_interface_model_request_ms"] >= 0
    assert captured["authorization"] == "Bearer reflex-secret-token"
    assert payload == {
        "text": "find the note from yesterday",
        "intent": "Find the note from yesterday.",
        "intent_source": "reflex_audio",
        "route": "oracle_direct",
        "transcript_source": "reflex_audio",
        "transcript": "find the node from yesterday",
        "transcript_confidence": 0.71,
        "interface_input_source": "native_audio",
        "reflex_provider": "vllm",
    }
    assert captured["url"] == "http://vllm.local:8000/v1/chat/completions"
    assert captured["timeout"] == 0.6
    assert captured["body"]["model"] == "google/gemma-4-E2B-it"
    assert captured["body"]["temperature"] == 0.35
    assert captured["body"]["max_tokens"] == 96
    assert captured["body"]["response_format"]["type"] == "json_schema"
    response_schema = captured["body"]["response_format"]["json_schema"]
    assert response_schema["name"] == "kame_reflex_decision"
    assert response_schema["strict"] is True
    assert response_schema["schema"]["required"] == [
        "route",
        "intent",
        "text",
        "route_confidence",
        "transcript",
        "transcript_confidence",
    ]
    assert response_schema["schema"]["properties"]["route"]["enum"] == [
        "defer",
        "local",
        "oracle_direct",
        "reject_or_clarify",
    ]
    prompt = captured["body"]["messages"][0]["content"][1]["text"]
    assert "KAME reflex" in prompt
    assert "Required keys: route, intent, text, route_confidence, transcript, transcript_confidence" in prompt
    assert "JSON schema:" in prompt
    assert '"required":["route","intent","text","route_confidence","transcript","transcript_confidence"]' in prompt
    assert "transcript to an empty string" in prompt
    assert "do not invent a command" in prompt
    assert "route must be one of local, defer, oracle_direct, or reject_or_clarify" in prompt
    assert "This voice session is already connected" in prompt
    assert "never claim Hermes cannot hear" in prompt
    assert "allow_local_greetings=False" in prompt
    assert "local_confidence_threshold=0.82" in prompt
    assert "ASR evidence mode is on_escalation" in prompt
    assert "Live session context:" in prompt
    assert "playback_active=true" in prompt
    assert "active_playback_generations=7" in prompt
    assert "last_speech_event=speech.energy" in prompt
    assert "last_speech_user_id=42" in prompt
    assert "last_speech_input_generation=5" in prompt
    assert "last_speech_rms=512" in prompt
    assert "last_speech_duration_ms=140" in prompt
    assert 'last_interface_event="interface.oracle.request"' in prompt
    assert 'last_interface_turn_id="voice-123:7"' in prompt
    assert 'last_interface_route="oracle_direct"' in prompt
    assert 'last_oracle_event="oracle.hint"' in prompt
    assert 'last_oracle_delta="Hermes is checking the deployment notes."' in prompt
    assert "last_oracle_final=false" in prompt


def test_reference_sidecar_vllm_kame_audio_reflex_falls_back_when_json_schema_unsupported(monkeypatch):
    captured_bodies = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return (
                b'{"choices":[{"message":{"content":"'
                b'{\\"route\\":\\"oracle_direct\\",'
                b'\\"intent\\":\\"Check status.\\",'
                b'\\"text\\":\\"check status\\",'
                b'\\"route_confidence\\":0.9}'
                b'"}}]}'
            )

    def fake_urlopen(req, timeout):
        captured_bodies.append(json.loads(req.data.decode("utf-8")))
        if len(captured_bodies) == 1:
            raise urllib.error.HTTPError(
                req.full_url,
                400,
                "Bad Request",
                {},
                io.BytesIO(b"response_format json_schema is not supported"),
            )
        return FakeResponse()

    monkeypatch.setattr("agent.realtime_voice_reference_sidecar.urllib.request.urlopen", fake_urlopen)

    sidecar = ReferenceRealtimeVoiceSidecarSession(
        ReferenceSidecarRuntimeConfig(
            vllm_base_url="http://vllm.local:8000/v1",
            vllm_model="google/gemma-4-E2B-it",
            vllm_timeout_seconds=12,
        )
    )
    sidecar.config = RealtimeVoiceSessionConfig(
        session_id="voice-123",
        engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        interface_audio_input="native_audio",
    )

    payload = sidecar._understand_audio_sync(b"audio", VoiceAudioCodec.WEBM_OPUS)

    assert len(captured_bodies) == 2
    assert captured_bodies[0]["response_format"]["type"] == "json_schema"
    assert captured_bodies[1]["response_format"] == {"type": "json_object"}
    assert payload["text"] == "check status"
    assert payload["intent"] == "Check status."
    assert payload["route"] == "oracle_direct"
    assert payload["route_confidence"] == 0.9
    assert payload["reflex_response_format_fallback"] == "json_object"
    assert payload["metrics"]["kame_interface_model_request_ms"] >= 0
    assert payload["metrics"]["kame_interface_response_format_fallback"] == 1


def test_reference_sidecar_vllm_kame_audio_reflex_does_not_retry_unrelated_http_errors(monkeypatch):
    captured_bodies = []

    def fake_urlopen(req, timeout):
        captured_bodies.append(json.loads(req.data.decode("utf-8")))
        raise urllib.error.HTTPError(
            req.full_url,
            401,
            "Unauthorized",
            {},
            io.BytesIO(b"response_format json_schema is not supported"),
        )

    monkeypatch.setattr("agent.realtime_voice_reference_sidecar.urllib.request.urlopen", fake_urlopen)

    sidecar = ReferenceRealtimeVoiceSidecarSession(
        ReferenceSidecarRuntimeConfig(
            vllm_base_url="http://vllm.local:8000/v1",
            vllm_model="google/gemma-4-E2B-it",
            vllm_timeout_seconds=12,
        )
    )
    sidecar.config = RealtimeVoiceSessionConfig(
        session_id="voice-123",
        engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        interface_audio_input="native_audio",
    )

    with pytest.raises(RuntimeError, match="HTTP 401: Unauthorized"):
        sidecar._understand_kame_with_vllm(b"audio", VoiceAudioCodec.WEBM_OPUS)

    assert len(captured_bodies) == 1
    assert captured_bodies[0]["response_format"]["type"] == "json_schema"


def test_reference_sidecar_vllm_kame_audio_reflex_wraps_pcm16_as_wav(monkeypatch):
    captured = {}
    pcm = b"\x01\x00\xff\x7f"

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"route\\":\\"oracle_direct\\",\\"intent\\":\\"Check status.\\",\\"text\\":\\"check status\\",\\"route_confidence\\":0.9}"}}]}'

    def fake_urlopen(req, timeout):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("agent.realtime_voice_reference_sidecar.urllib.request.urlopen", fake_urlopen)
    sidecar = ReferenceRealtimeVoiceSidecarSession(
        ReferenceSidecarRuntimeConfig(
            vllm_base_url="http://vllm.local:8000/v1",
            vllm_model="google/gemma-4-E2B-it",
        )
    )
    sidecar.config = RealtimeVoiceSessionConfig(
        session_id="voice-123",
        engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        sample_rate_hz=16000,
        channels=1,
        interface_audio_input="native_audio",
    )

    payload = sidecar._understand_audio_sync(pcm, VoiceAudioCodec.PCM16)

    audio_url = captured["body"]["messages"][0]["content"][0]["audio_url"]["url"]
    prefix, encoded = audio_url.split(",", 1)
    wav = base64.b64decode(encoded)
    assert payload["reflex_provider"] == "vllm"
    assert prefix == "data:audio/wav;base64"
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert int.from_bytes(wav[22:24], "little") == 1
    assert int.from_bytes(wav[24:28], "little") == 16000
    assert wav[36:40] == b"data"
    assert int.from_bytes(wav[40:44], "little") == len(pcm)
    assert wav[44:] == pcm


def test_reference_sidecar_drops_stale_speech_context_for_new_audio_generation():
    sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
    sidecar._record_speech_lifecycle_event(
        VoiceEvent(
            type=VoiceEventType.SPEECH_ENERGY,
            session_id="voice-123",
            sequence=1,
            payload={"input_generation": 5, "rms": 512},
        )
    )

    assert "last_speech_input_generation=5" in sidecar._kame_live_session_context_text()

    sidecar._clear_stale_speech_lifecycle_event(6)

    context = sidecar._kame_live_session_context_text()
    assert "last_speech_event=none" in context
    assert "last_speech_input_generation=5" not in context


def test_reference_sidecar_vllm_kame_audio_reflex_respects_provider_metrics_policy(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"route\\":\\"oracle_direct\\",\\"intent\\":\\"Check status.\\",\\"text\\":\\"check status\\",\\"route_confidence\\":0.9}"}}]}'

    def fake_urlopen(req, timeout):
        return FakeResponse()

    monkeypatch.setattr("agent.realtime_voice_reference_sidecar.urllib.request.urlopen", fake_urlopen)
    sidecar = ReferenceRealtimeVoiceSidecarSession(
        ReferenceSidecarRuntimeConfig(
            vllm_base_url="http://vllm.local:8000/v1",
            vllm_model="google/gemma-4-E2B-it",
        )
    )
    sidecar.config = RealtimeVoiceSessionConfig(
        session_id="voice-123",
        engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        interface_audio_input="native_audio",
        metrics_policy={"enabled": True, "log_provider_spans": False},
    )

    payload = sidecar._understand_audio_sync(b"audio", VoiceAudioCodec.WEBM_OPUS)

    assert payload["reflex_provider"] == "vllm"
    assert "metrics" not in payload


def test_reference_sidecar_text_fallback_bypasses_configured_vllm_reflex(monkeypatch):
    urlopen_calls = []
    transcribe_calls = []

    def forbidden_urlopen(req, timeout):
        urlopen_calls.append((req, timeout))
        raise AssertionError("text_fallback must not call the vLLM audio reflex")

    def fake_transcribe(path):
        transcribe_calls.append(path)
        with open(path, "rb") as handle:
            wav = handle.read()
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        assert wav[36:40] == b"data"
        return {"success": True, "transcript": "check deployment status"}

    monkeypatch.setattr("agent.realtime_voice_reference_sidecar.urllib.request.urlopen", forbidden_urlopen)

    sidecar = ReferenceRealtimeVoiceSidecarSession(
        ReferenceSidecarRuntimeConfig(
            vllm_base_url="http://vllm.local:8000/v1",
            vllm_model="google/gemma-4-E2B-it",
            vllm_timeout_seconds=12,
        ),
        transcribe_audio_func=fake_transcribe,
    )
    sidecar.config = RealtimeVoiceSessionConfig(
        session_id="voice-123",
        engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        interface_audio_input="text_fallback",
        asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
    )

    payload = sidecar._understand_audio_sync(b"\x00\x00\x01\x00", VoiceAudioCodec.PCM16)

    assert urlopen_calls == []
    assert transcribe_calls
    assert payload["text"] == "check deployment status"
    assert payload["intent"] == "check deployment status"
    assert payload["intent_source"] == "asr_fallback"
    assert payload["route"] == "oracle_direct"
    assert payload["transcript"] == "check deployment status"
    assert payload["transcript_source"] == "asr"
    assert payload["asr_transcript"] == "check deployment status"
    assert payload["asr_transcript_source"] == "asr"
    assert payload["interface_audio_input_fallback"] is True
    assert payload["interface_input_source"] == "local_stt"
    assert payload["reflex_provider"] == "local_stt"


def test_reference_sidecar_kame_audio_reflex_rejects_pcm_segments_over_model_limit(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        raise AssertionError("overlong native audio must be rejected before vLLM")

    monkeypatch.setattr("agent.realtime_voice_reference_sidecar.urllib.request.urlopen", fake_urlopen)

    sidecar = ReferenceRealtimeVoiceSidecarSession(
        ReferenceSidecarRuntimeConfig(
            vllm_base_url="http://vllm.local:8000/v1",
            vllm_model="google/gemma-4-E2B-it",
            vllm_timeout_seconds=12,
        )
    )
    sidecar.config = RealtimeVoiceSessionConfig(
        session_id="voice-123",
        engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        sample_rate_hz=16000,
        channels=1,
        interface_max_audio_seconds=1.0,
        interface_audio_input="native_audio",
        asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
    )

    with pytest.raises(RuntimeError, match="interface_max_audio_seconds"):
        sidecar._understand_audio_sync(b"\x00\x00" * 16001, VoiceAudioCodec.PCM16)

    assert calls == []


def test_reference_sidecar_kame_audio_reflex_reports_vllm_http_body(monkeypatch):
    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url,
            500,
            "Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"Please install vllm[audio] for audio support"}}'),
        )

    monkeypatch.setattr("agent.realtime_voice_reference_sidecar.urllib.request.urlopen", fake_urlopen)

    sidecar = ReferenceRealtimeVoiceSidecarSession(
        ReferenceSidecarRuntimeConfig(
            vllm_base_url="http://vllm.local:8000/v1",
            vllm_model="google/gemma-4-E2B-it",
            vllm_timeout_seconds=12,
        )
    )
    sidecar.config = RealtimeVoiceSessionConfig(
        session_id="voice-123",
        engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        sample_rate_hz=16000,
        channels=1,
        interface_audio_input="native_audio",
        asr_mode=RealtimeVoiceASRMode.DISABLED,
    )

    with pytest.raises(RuntimeError, match=r"Please install vllm\[audio\]"):
        sidecar._understand_audio_sync(b"\x00\x00" * 160, VoiceAudioCodec.PCM16)


def test_reference_sidecar_reports_kame_audio_segment_limit_during_live_receive(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(req.full_url)
        raise AssertionError("overlong native audio must be rejected before vLLM")

    monkeypatch.setattr("agent.realtime_voice_reference_sidecar.urllib.request.urlopen", fake_urlopen)

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(
                vllm_base_url="http://vllm.local:8000/v1",
                vllm_model="google/gemma-4-E2B-it",
                vllm_timeout_seconds=12,
            )
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                sample_rate_hz=16000,
                channels=1,
                interface_max_audio_seconds=1.0,
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.PCM16, data=b"\x00\x00" * 16001).to_payload(),
                    "end_of_utterance": True,
                    "input_generation": 4,
                },
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if event.type == VoiceEventType.SESSION_ERROR:
                await sidecar.close()
                break
        return seen

    seen = asyncio.run(run())
    degraded = next(
        event
        for event in seen
        if event.type == VoiceEventType.FRONTEND_STATE
        and event.payload.get("reason") == "kame_audio_segment_too_long"
    )
    error = next(event for event in seen if event.type == VoiceEventType.SESSION_ERROR)
    assert degraded.payload["status"] == "degraded"
    assert degraded.payload["interface_audio_input"] == "native_audio"
    assert degraded.payload["interface_max_audio_seconds"] == 1.0
    assert "interface_max_audio_seconds" in degraded.payload["error"]
    assert "kame audio segment too long" in error.payload["error"]
    assert calls == []


def test_reference_sidecar_kame_reflex_validation_rejects_invalid_route():
    payload = reference_sidecar_module._kame_reflex_payload_from_content(
        json.dumps(
            {
                "route": "maybe_local",
                "intent": "Check whether Hermes can hear me.",
                "text": "can you hear me",
                "local_reply": "Yes, I can hear you.",
            }
        )
    )

    assert payload["route"] == "oracle_direct"
    assert payload["intent"] == "Check whether Hermes can hear me."
    assert payload["text"] == "can you hear me"
    assert payload["local_reply"] == "Yes, I can hear you."
    assert payload["reflex_validation_error"] == "invalid_route"


def test_reference_sidecar_kame_reflex_marks_malformed_model_output():
    non_json = reference_sidecar_module._kame_reflex_payload_from_content(
        "I think the user wants me to check the repository."
    )
    wrong_shape = reference_sidecar_module._kame_reflex_payload_from_content(
        json.dumps(["oracle_direct", "check the repository"])
    )

    assert non_json == {
        "text": "I think the user wants me to check the repository.",
        "intent": "I think the user wants me to check the repository.",
        "intent_source": "reflex_audio",
        "transcript_source": "none",
        "route": "oracle_direct",
        "reflex_validation_error": "invalid_json",
    }
    assert wrong_shape == {
        "text": '["oracle_direct", "check the repository"]',
        "intent": '["oracle_direct", "check the repository"]',
        "intent_source": "reflex_audio",
        "transcript_source": "none",
        "route": "oracle_direct",
        "reflex_validation_error": "invalid_json_shape",
    }


def test_gemini_live_setup_omits_oracle_tool_instruction_when_tool_disabled():
    payload = _setup_payload(
        "gemini-live-test",
        GeminiLiveFrontendConfig(api_key="secret", enable_oracle_tool=False),
    )

    setup = payload["setup"]
    instruction = setup["systemInstruction"]["parts"][0]["text"]
    assert "ask_hermes_oracle" not in instruction
    assert "tools" not in setup


def test_gemini_live_setup_advertises_oracle_tool_only_when_enabled():
    payload = _setup_payload(
        "gemini-live-test",
        GeminiLiveFrontendConfig(api_key="secret", enable_oracle_tool=True),
    )

    setup = payload["setup"]
    instruction = setup["systemInstruction"]["parts"][0]["text"]
    tool_names = [
        declaration["name"]
        for tool in setup["tools"]
        for declaration in tool.get("functionDeclarations", [])
    ]
    assert "ask_hermes_oracle" in instruction
    assert "ask_hermes_oracle" in tool_names


def test_kame_reflex_decision_validates_schema_and_exports_payload():
    schema = kame_reflex_decision_json_schema()
    schema["required"].append("mutated")

    assert kame_reflex_decision_json_schema()["required"] == [
        "route",
        "intent",
        "text",
        "route_confidence",
        "transcript",
        "transcript_confidence",
    ]
    assert kame_reflex_schema_issues(
        {
            "route": "local",
            "intent": "The user is checking whether Hermes can hear them.",
            "text": "can you hear me",
            "route_confidence": 0.93,
            "local_reply": "Yes, I can hear you.",
        }
    ) == []
    assert kame_reflex_schema_issues(
        {
            "route": "local",
            "intent": "The user is checking whether Hermes can hear them.",
            "text": "can you hear me",
            "route_confidence": 1.2,
        }
    ) == [
        "route_confidence must be between 0 and 1",
        "local_reply is required for local or reject_or_clarify",
    ]
    assert kame_reflex_schema_issues(
        {
            "route": "defer",
            "intent": "The user wants Hermes to check deployment status.",
            "text": "check deployment status",
            "route_confidence": 0.9,
        }
    ) == ["interface_already_said is required for defer"]

    decision = KameReflexDecision.from_payload(
        {
            "route": "local",
            "intent": "The user is checking whether Hermes can hear them.",
            "text": "can you hear me",
            "local_reply": "Yes, I can hear you.",
            "transcript": "can you hear me",
            "transcript_source": "asr",
            "transcript_confidence": "1.7",
            "route_confidence": "0.93",
        }
    )

    assert decision.route == KameRoute.LOCAL
    assert decision.local_reply == "Yes, I can hear you."
    assert decision.route_confidence == 0.93
    assert decision.transcript_confidence == 1.0
    assert decision.validation_errors == ()
    assert decision.to_payload() == {
        "text": "can you hear me",
        "intent": "The user is checking whether Hermes can hear them.",
        "intent_source": "reflex_audio",
        "transcript_source": "asr",
        "route": "local",
        "route_confidence": 0.93,
        "local_reply": "Yes, I can hear you.",
        "transcript": "can you hear me",
        "transcript_confidence": 1.0,
    }


def test_kame_reflex_decision_rejects_invalid_local_schema():
    missing_reply = KameReflexDecision.from_payload(
        {
            "route": "reject_or_clarify",
            "intent": "The request is ambiguous.",
            "text": "that one",
        }
    )
    denial = KameReflexDecision.from_payload(
        {
            "route": "local",
            "intent": "The user asks whether Hermes can hear them.",
            "text": "can you hear me",
            "local_reply": "I cannot hear you or speak in Discord voice.",
            "transcript_confidence": "-0.5",
        }
    )
    invalid_route = KameReflexDecision.from_payload(
        {
            "route": "maybe",
            "intent": "Do something.",
            "text": "do something",
            "transcript": "do something",
        }
    )

    assert missing_reply.route == KameRoute.ORACLE_DIRECT
    assert missing_reply.validation_errors == ("missing_local_reply",)
    assert denial.route == KameRoute.ORACLE_DIRECT
    assert denial.local_reply == ""
    assert denial.transcript_confidence == 0.0
    assert denial.validation_errors == ("voice_capability_denial",)
    assert invalid_route.route == KameRoute.ORACLE_DIRECT
    assert invalid_route.transcript_source == "reflex_audio"
    assert invalid_route.validation_errors == ("invalid_route",)


def test_kame_reflex_decision_rejects_direct_tool_authority():
    issues = kame_reflex_schema_issues(
        {
            "route": "defer",
            "intent": "The user wants Hermes to inspect a file.",
            "text": "read pyproject.toml",
            "route_confidence": 0.88,
            "interface_already_said": "I'm checking pyproject.toml.",
            "tool_calls": [{"name": "read_file", "arguments": {"path": "pyproject.toml"}}],
        }
    )
    decision = KameReflexDecision.from_payload(
        {
            "route": "defer",
            "intent": "The user wants Hermes to inspect a file.",
            "text": "read pyproject.toml",
            "route_confidence": 0.88,
            "interface_already_said": "I'm checking pyproject.toml.",
            "tool_calls": [{"name": "read_file", "arguments": {"path": "pyproject.toml"}}],
            "local_reply": "I'll read it now.",
        }
    )

    assert "unexpected key tool_calls" in issues
    assert "direct tool authority is not allowed for the reflex" in issues
    assert decision.route == KameRoute.ORACLE_DIRECT
    assert decision.local_reply == ""
    assert decision.validation_errors == ("direct_tool_authority_not_allowed",)


def test_kame_oracle_request_strips_direct_tool_authority_fields():
    request = KameOracleRequest.from_turn(
        session_id="voice-123",
        turn_id="voice-123:1",
        source="discord_voice",
        user_id="jetha",
        payload={
            "route": "local",
            "intent": "The user wants Hermes to read a project file.",
            "text": "read pyproject.toml",
            "route_confidence": 0.96,
            "local_reply": "I'll read it now.",
            "tool_name": "read_file",
            "arguments": {"path": "pyproject.toml"},
        },
        fallback_text="read pyproject.toml",
    )

    metadata = request.to_metadata()
    assert request.route == KameRoute.ORACLE_DIRECT
    assert request.local_reply == ""
    assert request.reflex_validation_error == "direct_tool_authority_not_allowed"
    assert metadata["kame_reflex_validation_error"] == "direct_tool_authority_not_allowed"
    assert "tool_name" not in metadata
    assert "arguments" not in metadata


def test_external_kame_ask_brain_bridge_becomes_oracle_request():
    request = kame_external_brain_request_to_oracle_request(
        {
            "tool_name": "ask_brain",
            "arguments": {
                "query": "use my Stripe budget to prepare a VoIP provisioning plan",
                "intent": "Prepare VoIP provisioning with a Stripe budget.",
                "reflex_transcript_hypothesis": "use my Stripe budget to prepare a VoIP provisioning plan",
                "s2s_transcript_hypothesis": "use my stripe budget to prepare a voip provisioning plan",
                "interface_already_said": "I'm preparing the provisioning plan.",
                "conversation_summary": "The user is testing Discord voice to phone handoff.",
                "frontend_provider": "voiceclaw",
                "requested_response_style": {"spoken": True, "max_sentences": 1},
            },
        },
        session_id="external-kame-1",
        turn_id="external-kame-1:7",
        source="voiceclaw",
        user_id="jetha",
    )

    metadata = request.to_metadata()
    assert request.route == KameRoute.ORACLE_DIRECT
    assert request.source == "voiceclaw"
    assert request.user_id == "jetha"
    assert request.oracle_text == "use my Stripe budget to prepare a VoIP provisioning plan"
    assert request.transcript == "use my Stripe budget to prepare a VoIP provisioning plan"
    assert request.transcript_source == "external_frontend"
    assert request.interface_already_said == "I'm preparing the provisioning plan."
    assert request.conversation_summary == "The user is testing Discord voice to phone handoff."
    assert request.interface_input_source == "ask_brain"
    assert metadata["voice_architecture"] == "kame_frontend_oracle"
    assert metadata["kame_interface_input_source"] == "ask_brain"
    assert "tool_name" not in metadata
    assert "arguments" not in metadata


def test_external_kame_ask_brain_bridge_strips_nested_tool_authority():
    request = kame_external_brain_request_to_oracle_request(
        {
            "function": {
                "name": "openclaw_agent_consult",
                "arguments": json.dumps(
                    {
                        "query": "buy service credits",
                        "intent": "Buy service credits.",
                        "tool_name": "stripe_link_purchase",
                        "arguments": {"amount": 200, "card": "secret-card"},
                        "interface_already_said": "I'm preparing the spend request.",
                    }
                ),
            },
        },
        session_id="external-kame-2",
        turn_id="external-kame-2:1",
        source="openclaw_talk",
        user_id="jetha",
    )

    metadata = request.to_metadata()
    assert request.route == KameRoute.ORACLE_DIRECT
    assert request.interface_input_source == "openclaw_agent_consult"
    assert request.reflex_validation_error == "direct_tool_authority_not_allowed"
    assert metadata["kame_reflex_validation_error"] == "direct_tool_authority_not_allowed"
    assert request.oracle_text == "Buy service credits."
    assert "stripe_link_purchase" not in str(metadata)
    assert "secret-card" not in str(metadata)


def test_external_kame_brain_request_submits_oracle_job_without_waiting(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await self.release.wait()
            yield "Provisioning plan prepared."

    async def run():
        async def fake_speak(self, text, playback_generation):
            pass

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="external-kame-session",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
                metadata={"transport": "voiceclaw"},
            )
        )

        response = await engine.submit_external_brain_request(
            {
                "tool_name": "ask_brain",
                "arguments": {
                    "query": "prepare a VoIP provisioning plan",
                    "transcript": "prepare a voip provisioning plan",
                    "interface_already_said": "I'm preparing the provisioning plan.",
                },
            },
            turn_id="external-kame-session:voiceclaw:1",
            source="voiceclaw",
            user_id="jetha",
        )

        assert response["accepted"] is True
        assert response["job_id"] == "voice-oracle-001"
        assert response["state"] in {"queued", "running"}
        assert response["capacity"]["active"] == 1

        seen = []
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_STARTED:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)
        assert not any(event.type == VoiceEventType.ORACLE_JOB_COMPLETED for event in seen)
        assert oracle.requests[0].source == "voiceclaw"
        assert oracle.requests[0].interface_input_source == "ask_brain"
        assert oracle.requests[0].oracle_text == "prepare a voip provisioning plan"

        oracle.release.set()
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_COMPLETED:
                break

        completed = next(event for event in seen if event.type == VoiceEventType.ORACLE_JOB_COMPLETED)
        assert completed.payload["result_summary"] == "Provisioning plan prepared."
        await engine.close()

    asyncio.run(run())


def test_external_kame_frontend_can_cancel_matching_oracle_job(monkeypatch):
    class BlockingOracle:
        def __init__(self):
            self.started_turns = []
            self.started = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.started_turns.append(request.turn_id)
            if len(self.started_turns) == 2:
                self.started.set()
            await asyncio.Event().wait()
            yield "late result"

    async def run():
        async def fake_speak(self, text, playback_generation):
            pass

        monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

        oracle = BlockingOracle()
        engine = KameInterfaceOracleEngine(oracle=oracle)
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="external-kame-session",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                oracle_jobs={"enabled": True, "max_concurrent": 2, "queue_limit": 4},
                metadata={"transport": "openclaw_talk"},
            )
        )
        first = await engine.submit_external_brain_request(
            {"tool_name": "ask_brain", "arguments": {"query": "check invoices"}},
            turn_id="external-kame-session:openclaw:1",
            source="openclaw_talk",
            user_id="jetha",
        )
        second = await engine.submit_external_brain_request(
            {"tool_name": "ask_brain", "arguments": {"query": "check deployment"}},
            turn_id="external-kame-session:openclaw:2",
            source="openclaw_talk",
            user_id="jetha",
        )
        assert first["job_id"] == "voice-oracle-001"
        assert second["job_id"] == "voice-oracle-002"

        seen = []
        async for event in engine.events():
            seen.append(event)
            if len([item for item in seen if item.type == VoiceEventType.ORACLE_JOB_STARTED]) == 2:
                break
        await asyncio.wait_for(oracle.started.wait(), timeout=1)

        await engine.receive_event(
            VoiceEvent(
                type=VoiceEventType.INTERFACE_ORACLE_CANCEL,
                session_id="external-kame-session",
                sequence=99,
                payload={
                    "job_id": first["job_id"],
                    "reason": "external frontend cancellation",
                    "source": "openclaw_talk",
                },
            )
        )
        async for event in engine.events():
            seen.append(event)
            if event.type == VoiceEventType.ORACLE_JOB_CANCELLED and event.payload.get("job_id") == first["job_id"]:
                break

        cancel_event = next(
            event
            for event in seen
            if event.type == VoiceEventType.INTERFACE_ORACLE_CANCEL and event.payload.get("job_id") == first["job_id"]
        )
        assert cancel_event.payload["reason"] == "external frontend cancellation"
        status = await engine.get_oracle_job_status()
        jobs = {job["job_id"]: job for job in status["jobs"]}
        assert jobs[first["job_id"]]["state"] == "cancelled"
        assert jobs[second["job_id"]]["state"] == "running"
        await engine.close()

    asyncio.run(run())


def test_reference_sidecar_kame_reflex_never_speaks_voice_denial_locally():
    payload = reference_sidecar_module._kame_reflex_payload_from_content(
        json.dumps(
            {
                "route": "local",
                "intent": "The user asks whether Hermes can hear them.",
                "text": "can you hear me",
                "local_reply": "I cannot hear you or speak in Discord voice.",
            }
        )
    )

    assert payload["route"] == "oracle_direct"
    assert "local_reply" not in payload
    assert payload["reflex_validation_error"] == "voice_capability_denial"


def test_reference_sidecar_kame_reflex_never_forwards_direct_tool_payloads():
    payload = reference_sidecar_module._kame_reflex_payload_from_content(
        json.dumps(
            {
                "route": "local",
                "intent": "The user wants Hermes to read a project file.",
                "text": "read pyproject.toml",
                "route_confidence": 0.97,
                "local_reply": "I'll read it now.",
                "tool_calls": [{"name": "read_file", "arguments": {"path": "pyproject.toml"}}],
            }
        )
    )

    assert payload["route"] == "oracle_direct"
    assert "local_reply" not in payload
    assert "tool_calls" not in payload
    assert payload["reflex_validation_error"] == "direct_tool_authority_not_allowed"


def test_reference_sidecar_kame_local_route_respects_confidence_threshold():
    payload = reference_sidecar_module._kame_reflex_payload_from_content(
        json.dumps(
            {
                "route": "local",
                "route_confidence": 0.42,
                "intent": "The user is checking whether Hermes can hear them.",
                "text": "can you hear me",
                "local_reply": "Yes, I can hear you.",
            }
        ),
        config=RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            metadata={"routing": {"local_confidence_threshold": 0.75}},
        ),
    )

    assert payload["route"] == "oracle_direct"
    assert payload["route_confidence"] == 0.42
    assert "local_reply" not in payload
    assert payload["reflex_validation_error"] == "local_confidence_below_threshold"


def test_reference_sidecar_kame_local_route_requires_oracle_for_files_by_default():
    payload = reference_sidecar_module._kame_reflex_payload_from_content(
        json.dumps(
            {
                "route": "local",
                "route_confidence": 0.91,
                "intent": "The user wants Hermes to inspect the project config file.",
                "text": "check the config file",
                "local_reply": "The config file looks fine.",
            }
        ),
        config=RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        ),
    )

    assert payload["route"] == "oracle_direct"
    assert payload["route_confidence"] == 0.91
    assert "local_reply" not in payload
    assert payload["reflex_validation_error"] == "oracle_required_for_files"


def test_reference_sidecar_kame_local_route_requires_oracle_for_memory_by_default():
    payload = reference_sidecar_module._kame_reflex_payload_from_content(
        json.dumps(
            {
                "route": "local",
                "route_confidence": 0.91,
                "intent": "The user wants Hermes to remember a preference.",
                "text": "remember this voice id",
                "local_reply": "I will remember it.",
            }
        ),
        config=RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        ),
    )

    assert payload["route"] == "oracle_direct"
    assert "local_reply" not in payload
    assert payload["reflex_validation_error"] == "oracle_required_for_memory"


def test_reference_sidecar_kame_local_route_allows_configured_file_opt_out():
    payload = reference_sidecar_module._kame_reflex_payload_from_content(
        json.dumps(
            {
                "route": "local",
                "route_confidence": 0.91,
                "intent": "The user asks about a generic config setting.",
                "text": "what is a config file",
                "local_reply": "A config file stores settings.",
            }
        ),
        config=RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            metadata={"routing": {"require_oracle_for_files": False}},
        ),
    )

    assert payload["route"] == "local"
    assert payload["route_confidence"] == 0.91
    assert payload["local_reply"] == "A config file stores settings."


def test_reference_sidecar_kame_local_route_allows_smoke_test_phrase():
    payload = reference_sidecar_module._kame_reflex_payload_from_content(
        json.dumps(
            {
                "route": "local",
                "route_confidence": 0.91,
                "intent": "The user is checking whether voice works.",
                "text": "this is a test",
                "local_reply": "Test received.",
            }
        ),
        config=RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        ),
    )

    assert payload["route"] == "local"
    assert payload["local_reply"] == "Test received."


def test_reference_sidecar_kame_local_route_respects_disabled_greetings():
    payload = reference_sidecar_module._kame_reflex_payload_from_content(
        json.dumps(
            {
                "route": "local",
                "route_confidence": 0.91,
                "intent": "The user is checking whether Hermes can hear them.",
                "text": "can you hear me",
                "local_reply": "Yes, I can hear you.",
            }
        ),
        config=RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            metadata={"routing": {"allow_local_greetings": False}},
        ),
    )

    assert payload["route"] == "oracle_direct"
    assert "local_reply" not in payload
    assert payload["reflex_validation_error"] == "local_greetings_disabled"


def test_reference_sidecar_kame_clarify_route_respects_routing_policy():
    payload = reference_sidecar_module._kame_reflex_payload_from_content(
        json.dumps(
            {
                "route": "reject_or_clarify",
                "route_confidence": 0.91,
                "intent": "The user gave an incomplete request.",
                "text": "that one from yesterday",
                "local_reply": "Which project should I check?",
            }
        ),
        config=RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            metadata={"routing": {"allow_local_clarifications": False}},
        ),
    )

    assert payload["route"] == "oracle_direct"
    assert payload["route_confidence"] == 0.91
    assert "local_reply" not in payload
    assert payload["reflex_validation_error"] == "local_clarifications_disabled"


def test_reference_sidecar_kame_clarify_route_respects_confidence_threshold():
    payload = reference_sidecar_module._kame_reflex_payload_from_content(
        json.dumps(
            {
                "route": "reject_or_clarify",
                "route_confidence": 0.5,
                "intent": "The user gave an ambiguous request.",
                "text": "fix the thing",
                "local_reply": "Which issue should I fix?",
            }
        ),
        config=RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            metadata={"routing": {"local_confidence_threshold": 0.75}},
        ),
    )

    assert payload["route"] == "oracle_direct"
    assert payload["route_confidence"] == 0.5
    assert "local_reply" not in payload
    assert payload["reflex_validation_error"] == "local_confidence_below_threshold"


def test_reference_sidecar_kame_reflex_local_route_requires_local_reply():
    payload = reference_sidecar_module._kame_reflex_payload_from_content(
        json.dumps(
            {
                "route": "local",
                "intent": "The user says hello.",
                "text": "hello",
            }
        )
    )

    assert payload["route"] == "oracle_direct"
    assert payload["reflex_validation_error"] == "missing_local_reply"


def test_reference_sidecar_kame_on_escalation_does_not_start_streaming_stt(monkeypatch):
    created = []

    class FakeBridge:
        def __init__(self, *, path="/v1/realtime-text/session"):
            created.append(path)

        async def start(self, config):
            raise AssertionError("KAME on_escalation should not start streaming STT at session start")

    monkeypatch.setattr(
        "agent.realtime_voice_reference_sidecar.RealtimeVoiceSidecarClient",
        FakeBridge,
    )

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(
                streaming_stt_base_url="http://streaming-stt.local:9000",
                vllm_base_url="http://vllm.local:8000/v1",
                vllm_model="google/gemma-4-E2B-it",
            )
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
            )
        )
        await sidecar.close()

    asyncio.run(run())
    assert created == []


def test_reference_sidecar_kame_debug_exposes_streaming_stt_partials(monkeypatch):
    class FakeBridge:
        def __init__(self, *, path="/v1/realtime-text/session"):
            self.path = path
            self._events = asyncio.Queue()

        async def start(self, config):
            self.config = config

        async def send_event(self, event):
            await self._events.put(
                VoiceEvent(
                    type=VoiceEventType.TRANSCRIPT_PARTIAL,
                    session_id=event.session_id,
                    sequence=1,
                    payload={
                        "text": "debug caption",
                        "stability": 0.5,
                        "input_generation": event.payload.get("input_generation"),
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
        FakeBridge,
    )

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(
                streaming_stt_base_url="http://streaming-stt.local:9000",
                vllm_base_url="http://vllm.local:8000/v1",
                vllm_model="google/gemma-4-E2B-it",
            )
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.DEBUG,
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"audio").to_payload(),
                    "input_generation": 14,
                },
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if event.type == VoiceEventType.TRANSCRIPT_PARTIAL:
                await sidecar.close()
                break
        return seen

    seen = asyncio.run(run())
    partial = next(event for event in seen if event.type == VoiceEventType.TRANSCRIPT_PARTIAL)
    assert partial.payload["text"] == "debug caption"
    assert partial.payload["stability"] == 0.5
    assert partial.payload["input_generation"] == 14


def test_reference_sidecar_forwards_speech_lifecycle_to_active_frontends():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        sidecar.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            frontend_provider="gemma4",
            interface_audio_input="native_audio",
            asr_mode=RealtimeVoiceASRMode.SPECULATIVE,
        )
        sidecar._streaming_stt = object()
        sidecar._openai_realtime = object()
        sidecar._gemini_live = object()
        forwarded = []

        async def fake_send_streaming_stt_event(event):
            forwarded.append(("stt", event.type, dict(event.payload)))
            return True

        async def fake_send_openai_realtime_event(event):
            forwarded.append(("openai", event.type, dict(event.payload)))
            return True

        async def fake_send_gemini_live_event(event):
            forwarded.append(("gemini", event.type, dict(event.payload)))
            return True

        sidecar._send_streaming_stt_event = fake_send_streaming_stt_event
        sidecar._send_openai_realtime_event = fake_send_openai_realtime_event
        sidecar._send_gemini_live_event = fake_send_gemini_live_event

        for sequence, event_type, payload in [
            (1, VoiceEventType.SPEECH_START, {"user_id": "42", "input_generation": 5}),
            (2, VoiceEventType.SPEECH_ENERGY, {"rms": 512, "input_generation": 5}),
            (3, VoiceEventType.SPEECH_END, {"user_id": "42", "input_generation": 5}),
        ]:
            await sidecar.receive_event(
                VoiceEvent(
                    type=event_type,
                    session_id="voice-123",
                    sequence=sequence,
                    payload=payload,
                )
            )

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(anext(sidecar.events()), timeout=0.01)
        return forwarded

    forwarded = asyncio.run(run())

    assert [(provider, event_type) for provider, event_type, _payload in forwarded] == [
        ("stt", VoiceEventType.SPEECH_START),
        ("openai", VoiceEventType.SPEECH_START),
        ("gemini", VoiceEventType.SPEECH_START),
        ("stt", VoiceEventType.SPEECH_ENERGY),
        ("openai", VoiceEventType.SPEECH_ENERGY),
        ("gemini", VoiceEventType.SPEECH_ENERGY),
        ("stt", VoiceEventType.SPEECH_END),
        ("openai", VoiceEventType.SPEECH_END),
        ("gemini", VoiceEventType.SPEECH_END),
    ]
    assert forwarded[0][2] == {"user_id": "42", "input_generation": 5}
    assert forwarded[3][2] == {"rms": 512, "input_generation": 5}


def test_reference_sidecar_tracks_and_forwards_transport_playback_lifecycle():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        sidecar.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            frontend_provider="gemma4",
            interface_audio_input="native_audio",
        )
        sidecar._openai_realtime = object()
        sidecar._gemini_live = object()
        forwarded = []

        async def fake_send_openai_realtime_event(event):
            forwarded.append(("openai", event.type, dict(event.payload)))
            return True

        async def fake_send_gemini_live_event(event):
            forwarded.append(("gemini", event.type, dict(event.payload)))
            return True

        sidecar._send_openai_realtime_event = fake_send_openai_realtime_event
        sidecar._send_gemini_live_event = fake_send_gemini_live_event

        for sequence, event_type in [
            (1, VoiceEventType.PLAYBACK_STARTED),
            (2, VoiceEventType.PLAYBACK_STOPPED),
        ]:
            await sidecar.receive_event(
                VoiceEvent(
                    type=event_type,
                    session_id="voice-123",
                    sequence=sequence,
                    payload={"playback_generation": 9},
                )
            )

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(anext(sidecar.events()), timeout=0.01)
        return forwarded, set(sidecar._active_playback_generations)

    forwarded, active_playback_generations = asyncio.run(run())

    assert [(provider, event_type) for provider, event_type, _payload in forwarded] == [
        ("openai", VoiceEventType.PLAYBACK_STARTED),
        ("gemini", VoiceEventType.PLAYBACK_STARTED),
        ("openai", VoiceEventType.PLAYBACK_STOPPED),
        ("gemini", VoiceEventType.PLAYBACK_STOPPED),
    ]
    assert forwarded[0][2] == {"playback_generation": 9}
    assert active_playback_generations == set()


def test_reference_sidecar_kame_on_escalation_attaches_one_shot_asr_evidence(monkeypatch):
    calls = []
    sent_events = []

    class FakeBridge:
        def __init__(self, *, path="/v1/realtime-text/session"):
            calls.append(("init", path))
            self.config = None

        async def start(self, config):
            self.config = config
            calls.append(("start", config.frontend_provider, config.sidecar_base_url))

        async def send_event(self, event):
            sent_events.append(event)
            calls.append(("send", event.type.value, event.payload.get("input_generation")))

        async def events(self):
            yield VoiceEvent(
                type=VoiceEventType.TRANSCRIPT_FINAL,
                session_id="voice-123",
                sequence=1,
                payload={
                    "text": "literal ASR note 123",
                    "confidence": 0.84,
                    "input_generation": 7,
                },
            )

        async def close(self):
            calls.append(("close",))

    monkeypatch.setattr(
        "agent.realtime_voice_reference_sidecar.RealtimeVoiceSidecarClient",
        FakeBridge,
    )

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(
                streaming_stt_base_url="http://streaming-stt.local:9000",
                streaming_stt_model="nemotron-speech",
                streaming_stt_timeout_seconds=1,
                vllm_base_url="http://vllm.local:8000/v1",
                vllm_model="google/gemma-4-E2B-it",
            )
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
            )
        )

        def fake_understand_audio(audio, codec):
            calls.append(("reflex", audio, codec.value))
            return {
                "text": "reflex wording",
                "intent": "Reflex intent.",
                "intent_source": "reflex_audio",
                "route": "oracle_direct",
                "transcript_source": "none",
            }

        monkeypatch.setattr(sidecar, "_understand_audio_sync", fake_understand_audio)

        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"audio").to_payload(),
                    "end_of_utterance": True,
                    "input_generation": 7,
                },
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if event.type == VoiceEventType.INTERFACE_INTENT_FINAL:
                break

        await sidecar.close()
        return seen

    seen = asyncio.run(run())
    final = seen[-1]
    assert VoiceEventType.TRANSCRIPT_PARTIAL not in [event.type for event in seen]
    assert final.type == VoiceEventType.INTERFACE_INTENT_FINAL
    assert final.payload["text"] == "reflex wording"
    assert final.payload["intent"] == "Reflex intent."
    assert final.payload["route"] == "oracle_direct"
    assert "transcript" not in final.payload
    assert final.payload["transcript_source"] == "none"
    assert final.payload["asr_transcript"] == "literal ASR note 123"
    assert final.payload["asr_transcript_source"] == "asr"
    assert final.payload["asr_transcript_confidence"] == 0.84
    assert final.payload["metrics"]["kame_speech_end_to_interface_decision_ms"] >= 0
    assert final.payload["metrics"]["kame_first_audio_to_interface_decision_ms"] >= 0
    assert final.payload["metrics"]["kame_speech_boundary_to_final_intent_ms"] >= 0
    assert final.payload["metrics"]["oracle_verbatim_asr_ms"] >= 0
    assert sent_events[0].type == VoiceEventType.AUDIO_INPUT_CHUNK
    assert sent_events[0].payload["end_of_utterance"] is True
    assert sent_events[0].payload["input_generation"] == 7
    assert calls[0] == ("reflex", b"audio", "webm_opus")
    assert ("start", "streaming_stt", "http://streaming-stt.local:9000") in calls
    assert ("close",) in calls


def test_reference_sidecar_emits_kame_partial_interface_intent():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                interface_audio_input="native_audio",
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "can you",
                    "intent": "The user is starting a hearing check.",
                    "intent_source": "reflex_audio",
                    "route": "local",
                    "end_of_utterance": False,
                    "input_generation": 8,
                },
            )
        )

        seen = [await anext(sidecar.events()), await anext(sidecar.events()), await anext(sidecar.events())]
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(anext(sidecar.events()), timeout=0.01)
        await sidecar.close()
        return seen

    seen = asyncio.run(run())
    partial_intent = next(event for event in seen if event.type == VoiceEventType.INTERFACE_INTENT_PARTIAL)
    assert partial_intent.payload["intent"] == "The user is starting a hearing check."
    assert partial_intent.payload["route"] == "local"
    assert partial_intent.payload["input_generation"] == 8


def test_reference_sidecar_emits_kame_partial_transcripts_in_debug_mode():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.DEBUG,
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    "transcript": "can you",
                    "intent": "The user is starting a hearing check.",
                    "intent_source": "reflex_audio",
                    "route": "local",
                    "end_of_utterance": False,
                    "input_generation": 8,
                },
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if event.type == VoiceEventType.TRANSCRIPT_PARTIAL:
                await sidecar.close()
                break
        return seen

    seen = asyncio.run(run())
    partial_intent = next(event for event in seen if event.type == VoiceEventType.INTERFACE_INTENT_PARTIAL)
    transcript_partial = next(event for event in seen if event.type == VoiceEventType.TRANSCRIPT_PARTIAL)
    assert partial_intent.payload["intent"] == "The user is starting a hearing check."
    assert transcript_partial.payload["text"] == "can you"


def test_reference_sidecar_kame_local_route_skips_on_escalation_asr(monkeypatch):
    created = []

    class FakeBridge:
        def __init__(self, *, path="/v1/realtime-text/session"):
            created.append(path)

    monkeypatch.setattr(
        "agent.realtime_voice_reference_sidecar.RealtimeVoiceSidecarClient",
        FakeBridge,
    )

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(
                streaming_stt_base_url="http://streaming-stt.local:9000",
                vllm_base_url="http://vllm.local:8000/v1",
                vllm_model="google/gemma-4-E2B-it",
            )
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                interface_audio_input="native_audio",
                asr_mode=RealtimeVoiceASRMode.ON_ESCALATION,
            )
        )

        def fake_understand_audio(audio, codec):
            return {
                "text": "hello",
                "intent": "Greeting.",
                "intent_source": "reflex_audio",
                "route": "local",
                "local_reply": "Hi.",
                "transcript_source": "none",
            }

        monkeypatch.setattr(sidecar, "_understand_audio_sync", fake_understand_audio)

        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"audio").to_payload(),
                    "end_of_utterance": True,
                    "input_generation": 3,
                },
            )
        )

        async for event in sidecar.events():
            if event.type == VoiceEventType.INTERFACE_INTENT_FINAL:
                await sidecar.close()
                return event
        raise AssertionError("missing interface intent final")

    final = asyncio.run(run())
    assert final.payload["route"] == "local"
    assert "transcript" not in final.payload
    assert created == []


def test_reference_sidecar_kame_speculative_asr_does_not_drive_reflex(monkeypatch):
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(
                streaming_stt_base_url="http://streaming-stt.local:9000",
                vllm_base_url="http://vllm.local:8000/v1",
                vllm_model="google/gemma-4-E2B-it",
            )
        )
        sidecar.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            interface_audio_input="native_audio",
            asr_mode=RealtimeVoiceASRMode.SPECULATIVE,
        )
        sidecar._streaming_stt = object()
        sidecar._asr_hypotheses_by_generation[1] = {
            "asr_transcript": "speculative literal transcript",
            "asr_transcript_source": "asr",
            "asr_transcript_confidence": 0.91,
        }
        sent_to_asr = []

        async def fake_send_streaming_stt_event(event):
            sent_to_asr.append(event)
            return True

        def fake_understand_audio(audio, codec):
            return {
                "text": "reflex wording",
                "intent": "Reflex intent.",
                "intent_source": "reflex_audio",
                "route": "local",
                "local_reply": "Local reply.",
                "transcript_source": "none",
            }

        sidecar._send_streaming_stt_event = fake_send_streaming_stt_event
        monkeypatch.setattr(sidecar, "_understand_audio_sync", fake_understand_audio)

        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"audio").to_payload(),
                    "end_of_utterance": True,
                    "input_generation": 1,
                },
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if event.type == VoiceEventType.INTERFACE_INTENT_FINAL:
                break

        assert sent_to_asr
        final = seen[-1]
        assert final.type == VoiceEventType.INTERFACE_INTENT_FINAL
        assert final.payload["text"] == "reflex wording"
        assert final.payload["intent"] == "Reflex intent."
        assert final.payload["route"] == "local"
        assert "asr_transcript" not in final.payload
        assert "asr_transcript_source" not in final.payload
        assert "asr_transcript_confidence" not in final.payload

    asyncio.run(run())


def test_reference_sidecar_kame_speculative_asr_attaches_to_oracle_turn(monkeypatch):
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(
                streaming_stt_base_url="http://streaming-stt.local:9000",
                vllm_base_url="http://vllm.local:8000/v1",
                vllm_model="google/gemma-4-E2B-it",
            )
        )
        sidecar.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            interface_audio_input="native_audio",
            asr_mode=RealtimeVoiceASRMode.SPECULATIVE,
        )
        sidecar._streaming_stt = object()
        sidecar._asr_hypotheses_by_generation[1] = {
            "asr_transcript": "open the exact file config dot yaml",
            "asr_transcript_source": "asr",
            "asr_transcript_confidence": 0.88,
        }
        sent_to_asr = []

        async def fake_send_streaming_stt_event(event):
            sent_to_asr.append(event)
            return True

        def fake_understand_audio(audio, codec):
            return {
                "text": "open config.yaml",
                "intent": "Open the requested config file.",
                "intent_source": "reflex_audio",
                "route": "oracle_direct",
                "transcript_source": "none",
            }

        sidecar._send_streaming_stt_event = fake_send_streaming_stt_event
        monkeypatch.setattr(sidecar, "_understand_audio_sync", fake_understand_audio)

        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"audio").to_payload(),
                    "end_of_utterance": True,
                    "input_generation": 1,
                },
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if event.type == VoiceEventType.INTERFACE_INTENT_FINAL:
                break

        assert sent_to_asr
        final = seen[-1]
        assert final.type == VoiceEventType.INTERFACE_INTENT_FINAL
        assert final.payload["text"] == "open config.yaml"
        assert final.payload["intent"] == "Open the requested config file."
        assert final.payload["route"] == "oracle_direct"
        assert final.payload["asr_transcript"] == "open the exact file config dot yaml"
        assert final.payload["asr_transcript_source"] == "asr"
        assert final.payload["asr_transcript_confidence"] == 0.88

    asyncio.run(run())


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


def test_reference_sidecar_health_rejects_unready_streaming_stt_bridge():
    runtime = ReferenceSidecarRuntimeConfig(
        streaming_stt_base_url="http://streaming-stt.local:9000",
        streaming_stt_model="portable-streaming-asr",
        local_stt_enabled=False,
    )

    payload = reference_sidecar_health_payload(
        runtime,
        streaming_stt_health={
            "ok": False,
            "capabilities": {
                "streaming_stt": True,
            },
        },
    )

    assert payload["frontend"]["provider"] == "local"
    assert payload["frontend"]["streaming_stt_bridge"] == {
        "configured": True,
        "healthy": False,
    }
    assert payload["capabilities"]["streaming_stt"] is False
    assert payload["capabilities"]["utterance_stt"] is False


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
            "frontend": {
                "tts_model_languages": ["ja", "https://voice.local/secret"],
            },
            "capabilities": {
                "tts": True,
                "streaming_tts": True,
                "output_languages": ["en", "ja", "token=secret"],
            },
        },
    )

    assert verified["capabilities"]["tts"] is True
    assert verified["capabilities"]["output_languages"] == ["en", "ja"]
    assert verified["frontend"]["tts_model_languages"] == ["ja"]
    assert verified["frontend"]["streaming_tts_bridge"] == {
        "configured": True,
        "healthy": True,
        "model": "portable-streaming-voice",
    }


def test_reference_sidecar_health_rejects_unready_streaming_tts_bridge_metadata():
    runtime = ReferenceSidecarRuntimeConfig(
        streaming_tts_base_url="http://streaming-tts.local:9001",
        streaming_tts_model="portable-streaming-voice",
        local_tts_enabled=False,
        output_languages=("en", "ja"),
    )

    payload = reference_sidecar_health_payload(
        runtime,
        streaming_tts_health={
            "ok": False,
            "frontend": {
                "tts_model_languages": ["en", "ja"],
            },
            "capabilities": {
                "tts": True,
                "streaming_tts": True,
                "output_languages": ["en", "ja"],
            },
        },
    )

    assert payload["capabilities"]["tts"] is False
    assert "output_languages" not in payload["capabilities"]
    assert "tts_model_languages" not in payload["frontend"]
    assert payload["frontend"]["streaming_tts_bridge"] == {
        "configured": True,
        "healthy": False,
        "model": "portable-streaming-voice",
    }


def test_reference_sidecar_health_uses_streaming_tts_bridge_language_evidence_over_policy():
    runtime = ReferenceSidecarRuntimeConfig(
        streaming_tts_base_url="http://streaming-tts.local:9001",
        streaming_tts_model="portable-streaming-voice",
        local_tts_enabled=False,
        output_languages=("en", "ja"),
    )

    no_route_metadata = reference_sidecar_health_payload(
        runtime,
        streaming_tts_health={
            "ok": True,
            "capabilities": {
                "tts": True,
                "streaming_tts": True,
            },
        },
    )

    assert no_route_metadata["capabilities"]["tts"] is True
    assert "output_languages" not in no_route_metadata["capabilities"]

    english_only = reference_sidecar_health_payload(
        runtime,
        streaming_tts_health={
            "ok": True,
            "capabilities": {
                "tts": True,
                "streaming_tts": True,
                "output_languages": ["en"],
            },
        },
    )

    assert english_only["capabilities"]["output_languages"] == ["en"]


def test_reference_sidecar_health_verifies_vllm_audio_frontend_after_models_health():
    runtime = ReferenceSidecarRuntimeConfig(
        vllm_base_url="http://voice.local:8000/v1",
        vllm_model="google/gemma-4-E2B-it",
        local_stt_enabled=False,
    )

    unverified = reference_sidecar_health_payload(
        runtime,
        vllm_health_checked=True,
        vllm_health={"data": [{"id": "different-model"}]},
    )
    verified = reference_sidecar_health_payload(
        runtime,
        vllm_health_checked=True,
        vllm_health={"data": [{"id": "google/gemma-4-E2B-it"}]},
    )

    assert unverified["frontend"]["provider"] == "local"
    assert unverified["frontend"]["vllm_audio_frontend"] == {
        "configured": True,
        "healthy": False,
        "model": "google/gemma-4-E2B-it",
        "token_configured": False,
    }
    assert unverified["capabilities"]["vllm_audio_frontend"] is False
    assert unverified["capabilities"]["vllm_audio_frontend_configured"] is True
    assert unverified["capabilities"]["utterance_stt"] is False
    assert verified["frontend"]["provider"] == "vllm"
    assert verified["frontend"]["vllm_audio_frontend"] == {
        "configured": True,
        "healthy": True,
        "model": "google/gemma-4-E2B-it",
        "token_configured": False,
    }
    assert verified["capabilities"]["vllm_audio_frontend"] is True
    assert verified["capabilities"]["utterance_stt"] is True


def test_reference_sidecar_health_prefers_vllm_reflex_over_stt_evidence_bridge():
    runtime = ReferenceSidecarRuntimeConfig(
        vllm_base_url="http://voice.local:8000/v1",
        vllm_model="google/gemma-4-E2B-it",
        streaming_stt_base_url="http://voice.local:8767",
        streaming_stt_model="nemotron-speech-streaming-0.6b",
        local_stt_enabled=False,
    )

    payload = reference_sidecar_health_payload(
        runtime,
        vllm_health_checked=True,
        vllm_health={"data": [{"id": "google/gemma-4-E2B-it"}]},
        streaming_stt_health={
            "ok": True,
            "capabilities": {
                "streaming_stt": True,
            },
        },
    )

    assert payload["frontend"]["provider"] == "vllm"
    assert payload["frontend"]["model"] == "google/gemma-4-E2B-it"
    assert payload["frontend"]["vllm_audio_frontend"] == {
        "configured": True,
        "healthy": True,
        "model": "google/gemma-4-E2B-it",
        "token_configured": False,
    }
    assert payload["frontend"]["streaming_stt_bridge"] == {
        "configured": True,
        "healthy": True,
    }
    assert payload["capabilities"]["vllm_audio_frontend"] is True
    assert payload["capabilities"]["streaming_stt"] is True
    assert payload["capabilities"]["streaming_stt_bridge"] is True
    assert payload["capabilities"]["utterance_stt"] is True


def test_reference_sidecar_health_reports_configured_kame_provider_labels():
    runtime = ReferenceSidecarRuntimeConfig(
        interface_provider="gemma4",
        vllm_base_url="http://voice.local:8000/v1",
        vllm_model="google/gemma-4-E2B-it",
        streaming_stt_provider="nvidia_speech",
        streaming_stt_base_url="http://voice.local:8767",
        streaming_stt_model="nemotron-speech-streaming-0.6b",
        streaming_tts_provider="cartesia",
        streaming_tts_base_url="http://voice.local:8768",
        streaming_tts_model="sonic-3.5",
        local_stt_enabled=False,
        local_tts_enabled=False,
    )

    payload = reference_sidecar_health_payload(
        runtime,
        vllm_health_checked=True,
        vllm_health={"data": [{"id": "google/gemma-4-E2B-it"}]},
        streaming_stt_health={
            "ok": True,
            "capabilities": {"streaming_stt": True},
        },
        streaming_tts_health={
            "ok": True,
            "capabilities": {"tts": True, "streaming_tts": True},
        },
    )

    assert payload["frontend"]["provider"] == "gemma4"
    assert payload["frontend"]["model"] == "google/gemma-4-E2B-it"
    assert payload["frontend"]["vllm_audio_frontend"]["provider"] == "gemma4"
    assert payload["frontend"]["vllm_audio_frontend"]["implementation_provider"] == "vllm"
    assert payload["frontend"]["streaming_stt_bridge"] == {
        "configured": True,
        "healthy": True,
        "provider": "nvidia_speech",
        "implementation_provider": "streaming_stt",
    }
    assert payload["frontend"]["streaming_tts_bridge"] == {
        "configured": True,
        "healthy": True,
        "provider": "cartesia",
        "implementation_provider": "streaming_tts",
        "model": "sonic-3.5",
    }


def test_reference_sidecar_health_payload_is_sanitized():
    payload = reference_sidecar_health_payload(
        ReferenceSidecarRuntimeConfig(
            vllm_base_url="http://user:secret@voice.local:8000/v1",
            vllm_model="google/gemma-4-E4B-it-qat-w4a16-ct",
            vllm_token="reflex-secret-token",
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
            "vllm_audio_frontend": {
                "configured": True,
                "healthy": True,
                "model": "google/gemma-4-E4B-it-qat-w4a16-ct",
                "token_configured": True,
            },
            "languages": ["ja", "en-US", "ko"],
            "scripts": ["Jpan", "Latn"],
        },
        "capabilities": {
            "utterance_stt": True,
            "streaming_stt": False,
            "tts": True,
            "native_s2s": False,
            "vllm_audio_frontend": True,
            "vllm_audio_frontend_configured": True,
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
    monkeypatch.setenv("HERMES_KAME_INTERFACE_PROVIDER", "gemma4")
    monkeypatch.setenv("HERMES_KAME_INTERFACE_BASE_URL", "http://interface.local:8000/v1")
    monkeypatch.setenv("HERMES_VOICE_VLLM_BASE_URL", "http://legacy.local:8000/v1")
    monkeypatch.setenv("HERMES_KAME_INTERFACE_MODEL", "gemma-4-E2B-it")
    monkeypatch.setenv("HERMES_VOICE_VLLM_MODEL", "legacy-reflex-model")
    monkeypatch.setenv("HERMES_KAME_INTERFACE_API_KEY", "interface-secret-token")
    monkeypatch.setenv("HERMES_DGX_SPARK_ASR_PROVIDER", "nvidia_speech")
    monkeypatch.setenv("HERMES_VOICE_STREAMING_STT_BASE_URL", "http://streaming-stt.local:9000")
    monkeypatch.setenv("HERMES_VOICE_STREAMING_STT_MODEL", "portable-streaming-asr")
    monkeypatch.setenv("HERMES_VOICE_STREAMING_STT_TOKEN", "secret-token")
    monkeypatch.setenv("HERMES_VOICE_STREAMING_STT_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setenv("HERMES_VOICE_STREAMING_BRIDGE_HEALTH_TIMEOUT_SECONDS", "0.25")
    monkeypatch.setenv("HERMES_DGX_SPARK_TTS_PROVIDER", "cartesia")
    monkeypatch.setenv("HERMES_VOICE_STREAMING_TTS_BASE_URL", "http://streaming-tts.local:9001")
    monkeypatch.setenv("HERMES_VOICE_STREAMING_TTS_MODEL", "portable-streaming-voice")
    monkeypatch.setenv("HERMES_VOICE_STREAMING_TTS_TOKEN", "tts-secret-token")
    monkeypatch.setenv("HERMES_VOICE_STREAMING_TTS_TIMEOUT_SECONDS", "3.5")
    monkeypatch.setenv("HERMES_OPENAI_REALTIME_API_KEY", "openai-secret-token")
    monkeypatch.setenv("HERMES_OPENAI_REALTIME_BASE_URL", "wss://api.openai.example.test/v1/realtime")
    monkeypatch.setenv("HERMES_OPENAI_REALTIME_MODEL", "gpt-realtime-2")
    monkeypatch.setenv("HERMES_OPENAI_REALTIME_VOICE", "cedar")
    monkeypatch.setenv("HERMES_OPENAI_REALTIME_TRANSCRIPTION_MODEL", "gpt-realtime-whisper")
    monkeypatch.setenv("HERMES_OPENAI_REALTIME_SAFETY_IDENTIFIER", "stable-user")

    runtime = runtime_config_from_env()

    assert runtime.input_languages == ("ja", "en-US")
    assert runtime.output_languages == ("ja", "ko")
    assert runtime.scripts == ("Jpan", "Latn")
    assert runtime.interface_provider == "gemma4"
    assert runtime.vllm_base_url == "http://interface.local:8000/v1"
    assert runtime.vllm_model == "gemma-4-E2B-it"
    assert runtime.vllm_token == "interface-secret-token"
    assert runtime.streaming_stt_provider == "nvidia_speech"
    assert runtime.streaming_stt_base_url == "http://streaming-stt.local:9000"
    assert runtime.streaming_stt_model == "portable-streaming-asr"
    assert runtime.streaming_stt_token == "secret-token"
    assert runtime.streaming_stt_timeout_seconds == 2.5
    assert runtime.streaming_bridge_health_timeout_seconds == 0.25
    assert runtime.streaming_tts_provider == "cartesia"
    assert runtime.streaming_tts_base_url == "http://streaming-tts.local:9001"
    assert runtime.streaming_tts_model == "portable-streaming-voice"
    assert runtime.streaming_tts_token == "tts-secret-token"
    assert runtime.streaming_tts_timeout_seconds == 3.5
    assert runtime.openai_realtime_api_key == "openai-secret-token"
    assert runtime.openai_realtime_base_url == "wss://api.openai.example.test/v1/realtime"
    assert runtime.openai_realtime_model == "gpt-realtime-2"
    assert runtime.openai_realtime_voice == "cedar"
    assert runtime.openai_realtime_transcription_model == "gpt-realtime-whisper"
    assert runtime.openai_realtime_safety_identifier == "stable-user"


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


def test_reference_sidecar_vllm_health_probe_uses_interface_token(monkeypatch):
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"data": [{"id": "google/gemma-4-E2B-it"}]}'

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, request.get_header("Authorization"), request.get_header("Accept"), timeout))
        return FakeResponse()

    monkeypatch.setattr(reference_sidecar_module.urllib.request, "urlopen", fake_urlopen)

    runtime = ReferenceSidecarRuntimeConfig(
        vllm_base_url="http://vllm.local:8000/v1",
        vllm_model="google/gemma-4-E2B-it",
        vllm_token="reflex-secret-token",
        vllm_timeout_seconds=12.0,
    )

    health = reference_sidecar_module._probe_vllm_health_sync(runtime)

    assert health == {"data": [{"id": "google/gemma-4-E2B-it"}]}
    assert calls == [
        ("http://vllm.local:8000/v1/models", "Bearer reflex-secret-token", "application/json", 2.0)
    ]


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
            VoiceEventType.SESSION_STARTED,
            VoiceEventType.FRONTEND_STATE,
            VoiceEventType.TRANSCRIPT_PARTIAL,
            VoiceEventType.TRANSCRIPT_FINAL,
        ]
        assert seen[1].payload["provider"] == "streaming_stt"
        assert seen[1].payload["streaming_stt"] is True
        assert seen[2].payload == {
            "language": "ja",
            "locale": "ja-JP",
            "script": "Jpan",
            "text": "こん",
            "stability": 0.4,
            "input_generation": 12,
        }
        assert seen[3].payload == {
            "language": "ja",
            "locale": "ja-JP",
            "script": "Jpan",
            "text": "こんにちは Hermes",
            "confidence": 0.92,
            "input_generation": 12,
        }

    asyncio.run(run())


def test_reference_sidecar_kame_streaming_stt_fallback_labels_reflex_provenance(monkeypatch):
    class FakeStreamingSTTClient:
        def __init__(self, *, path="/v1/realtime-text/session"):
            self.path = path
            self._events = asyncio.Queue()

        async def start(self, config):
            self.config = config

        async def send_event(self, event):
            await self._events.put(
                VoiceEvent(
                    type=VoiceEventType.TRANSCRIPT_FINAL,
                    session_id=event.session_id,
                    sequence=2,
                    payload={
                        "text": "check deployment status",
                        "confidence": 0.88,
                        "input_generation": event.payload.get("input_generation"),
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
            )
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                interface_audio_input="text_fallback",
                asr_mode=RealtimeVoiceASRMode.FALLBACK,
            )
        )
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

        async for event in sidecar.events():
            if event.type == VoiceEventType.TRANSCRIPT_FINAL:
                await sidecar.close()
                return event
        raise AssertionError("missing transcript final")

    final = asyncio.run(run())
    assert final.payload["text"] == "check deployment status"
    assert final.payload["intent"] == "check deployment status"
    assert final.payload["intent_source"] == "asr_fallback"
    assert final.payload["route"] == "oracle_direct"
    assert final.payload["transcript"] == "check deployment status"
    assert final.payload["transcript_source"] == "asr"
    assert final.payload["asr_transcript"] == "check deployment status"
    assert final.payload["asr_transcript_source"] == "asr"
    assert final.payload["interface_audio_input_fallback"] is True
    assert final.payload["interface_input_source"] == "streaming_stt"
    assert final.payload["reflex_provider"] == "streaming_stt"


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
        assert created[0].sent[1].type == VoiceEventType.SESSION_CLOSED
        assert created[0].sent[1].session_id == "voice-123"
        assert created[0].sent[1].payload == {"reason": "sidecar_session_closed"}
        assert [event.type for event in seen] == [
            VoiceEventType.SESSION_STARTED,
            VoiceEventType.FRONTEND_STATE,
            VoiceEventType.AUDIO_OUTPUT_CHUNK,
            VoiceEventType.PLAYBACK_STOPPED,
            VoiceEventType.ASSISTANT_AUDIO_END,
            VoiceEventType.SESSION_CLOSED,
        ]
        assert seen[1].payload["streaming_tts"] is True
        assert AudioChunk.from_payload(seen[2].payload).data == b"pcm-audio"
        assert seen[2].payload["playback_generation"] == 7
        assert seen[3].payload == {"reason": "session_closed", "playback_generation": 7}
        assert seen[4].payload == {"reason": "session_closed", "playback_generation": 7}

    asyncio.run(run())


def test_reference_sidecar_reports_tts_unavailable_when_streaming_tts_fails_without_local_fallback(monkeypatch):
    created = []

    class FailingStreamingTTSClient:
        def __init__(self, *, path="/v1/realtime-text/session"):
            self.path = path
            self.closed = False
            created.append(self)

        async def start(self, config):
            self.config = config

        async def send_event(self, event):
            raise RuntimeError("TTS failed Bearer secret-token at http://user:pass@voice.local/v1?token=abc")

        async def events(self):
            if False:
                yield None
            await asyncio.Event().wait()

        async def close(self):
            self.closed = True

    monkeypatch.setattr(
        "agent.realtime_voice_reference_sidecar.RealtimeVoiceSidecarClient",
        FailingStreamingTTSClient,
    )

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(
                streaming_tts_base_url="http://streaming-tts.local:9001",
                streaming_tts_model="portable-streaming-voice",
                local_tts_enabled=False,
            )
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                frontend_provider="sidecar",
                tts_provider="streaming_tts",
                tts_model="portable-streaming-voice",
                tts_voice="spark-voice",
            )
        )
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                session_id="voice-123",
                sequence=1,
                payload={
                    "text": "hello back",
                    "speak": True,
                    "playback_generation": 7,
                },
            )
        )

        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if event.type == VoiceEventType.SESSION_ERROR:
                await sidecar.close()
                break
        return seen

    seen = asyncio.run(run())

    assert created[0].closed is True
    assert [event.type for event in seen] == [
        VoiceEventType.SESSION_STARTED,
        VoiceEventType.FRONTEND_STATE,
        VoiceEventType.FRONTEND_STATE,
        VoiceEventType.SESSION_ERROR,
    ]
    assert seen[2].payload["reason"] == "streaming_tts_send_failed"
    assert seen[2].payload["streaming_tts"] is False
    error = seen[3]
    assert error.payload["reason"] == "tts_unavailable"
    assert error.payload["streaming_tts"] is False
    assert error.payload["local_tts"] is False
    assert error.payload["playback_generation"] == 7
    assert error.payload["tts_provider"] == "streaming_tts"
    assert error.payload["tts_model"] == "portable-streaming-voice"
    assert error.payload["tts_voice"] == "spark-voice"
    assert "streaming_tts_send_failed" in error.payload["error"]
    assert "TTS failed" in error.payload["error"]
    assert "secret-token" not in error.payload["error"]
    assert "user:pass" not in error.payload["error"]
    assert "token=abc" not in error.payload["error"]


def test_reference_sidecar_health_requires_bearer_token(monkeypatch):
    from fastapi.testclient import TestClient

    async def fake_probe_vllm_health(runtime):
        return {"data": [{"id": "google/gemma-4-E4B-it-qat-w4a16-ct"}]}

    monkeypatch.setattr(reference_sidecar_module, "_probe_vllm_health", fake_probe_vllm_health)

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
        ready = ws.receive_json()

    assert response["type"] == "session.started"
    assert response["session_id"] == "voice-123"
    assert ready["type"] == "frontend.state"
    assert ready["session_id"] == "voice-123"


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


def test_kame_session_persists_reflex_intent_not_raw_transcript():
    session = RealtimeVoiceSession(
        RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        ),
        engine=KameInterfaceOracleEngine(oracle=FakeOracle()),
    )

    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.TRANSCRIPT_PARTIAL,
            session_id="voice-123",
            sequence=1,
            payload={"text": "find the node"},
        )
    )
    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.TRANSCRIPT_FINAL,
            session_id="voice-123",
            sequence=2,
            payload={
                "text": "find the node from yesterday",
                "voice_architecture": "kame_frontend_oracle",
                "kame_intent": "Find the note from yesterday.",
                "kame_transcript": "find the node from yesterday",
                "kame_transcript_source": "reflex_audio",
                "playback_generation": 1,
            },
        )
    )

    assert session.transcript.partial_user_text == ""
    assert session.durable_messages() == [{"role": "user", "content": "Find the note from yesterday."}]


def test_session_persists_only_durable_oracle_records():
    session = RealtimeVoiceSession(
        RealtimeVoiceSessionConfig(session_id="voice-123"),
        engine=TextOracleTTSEngine(oracle=FakeOracle()),
    )

    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.INTERFACE_ORACLE_REQUEST,
            session_id="voice-123",
            sequence=1,
            payload={
                "turn_id": "voice-123:1",
                "intent": "Check deployment status.",
                "transcript": "check the deployment status",
                "asr_transcript": "check deployment status",
                "playback_generation": 1,
            },
        )
    )
    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.ORACLE_TOOL_CALL,
            session_id="voice-123",
            sequence=2,
            payload={
                "tool_name": "read_file",
                "tool_call_id": "call-1",
                "playback_generation": 1,
            },
        )
    )
    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.ORACLE_RESPONSE_PARTIAL,
            session_id="voice-123",
            sequence=3,
            payload={"text": "Checking", "playback_generation": 1},
        )
    )
    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.ORACLE_TOOL_RESULT,
            session_id="voice-123",
            sequence=4,
            payload={
                "tool_name": "read_file",
                "tool_call_id": "call-1",
                "result": "ok",
                "playback_generation": 1,
            },
        )
    )
    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.ORACLE_HINT,
            session_id="voice-123",
            sequence=5,
            payload={"text": "Use the deployment docs."},
        )
    )
    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.ORACLE_RESPONSE_FINAL,
            session_id="voice-123",
            sequence=6,
            payload={"text": "Deployment is healthy.", "playback_generation": 1},
        )
    )
    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.ORACLE_ERROR,
            session_id="voice-123",
            sequence=7,
            payload={"error": "timeout", "playback_generation": 1},
        )
    )
    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.ORACLE_ERROR,
            session_id="voice-123",
            sequence=8,
            payload={
                "reason": "oracle_cancelled",
                "error": "oracle request cancelled by realtime voice interruption",
                "playback_generation": 1,
            },
        )
    )

    assert session.durable_oracle_records() == [
        {
            "type": VoiceEventType.INTERFACE_ORACLE_REQUEST.value,
            "payload": {
                "turn_id": "voice-123:1",
                "intent": "Check deployment status.",
                "transcript": "check the deployment status",
                "asr_transcript": "check deployment status",
                "playback_generation": 1,
            },
        },
        {
            "type": VoiceEventType.ORACLE_TOOL_RESULT.value,
            "payload": {
                "tool_name": "read_file",
                "tool_call_id": "call-1",
                "result": "ok",
                "playback_generation": 1,
            },
        },
        {
            "type": VoiceEventType.ORACLE_RESPONSE_FINAL.value,
            "payload": {"text": "Deployment is healthy.", "playback_generation": 1},
        },
        {
            "type": VoiceEventType.ORACLE_ERROR.value,
            "payload": {"error": "timeout", "playback_generation": 1},
        },
    ]


def test_session_persists_durable_async_oracle_job_records():
    session = RealtimeVoiceSession(
        RealtimeVoiceSessionConfig(session_id="voice-123"),
        engine=TextOracleTTSEngine(oracle=FakeOracle()),
    )

    events = [
        VoiceEvent(
            type=VoiceEventType.ORACLE_JOB_PROGRESS,
            session_id="voice-123",
            sequence=1,
            payload={
                "job_id": "voice-oracle-001",
                "phase": "tool",
                "tool_event": {"tool_name": "stripe_link_purchase", "arguments": {"card": "secret"}},
            },
        ),
        VoiceEvent(
            type=VoiceEventType.ORACLE_JOB_WAITING_FOR_APPROVAL,
            session_id="voice-123",
            sequence=2,
            payload={
                "job_id": "voice-oracle-001",
                "state": "waiting_for_approval",
                "approval_reason": "Stripe Link spend requires approval",
                "approval": {
                    "approval_id": "approval-123",
                    "tool_name": "stripe_link_purchase",
                },
            },
        ),
        VoiceEvent(
            type=VoiceEventType.ORACLE_JOB_COMPLETED,
            session_id="voice-123",
            sequence=3,
            payload={
                "job_id": "voice-oracle-001",
                "state": "completed",
                "result_summary": "The spend approval cleared.",
                "result_text": "The spend approval cleared with a longer durable explanation.",
                "result_text_chars": 61,
            },
        ),
        VoiceEvent(
            type=VoiceEventType.ORACLE_JOB_FAILED,
            session_id="voice-123",
            sequence=4,
            payload={
                "job_id": "voice-oracle-002",
                "state": "failed",
                "error": "oracle backend unavailable",
            },
        ),
        VoiceEvent(
            type=VoiceEventType.ORACLE_JOB_CANCELLED,
            session_id="voice-123",
            sequence=5,
            payload={
                "job_id": "voice-oracle-003",
                "state": "cancelled",
                "cancel_reason": "user cancelled queued job",
            },
        ),
    ]
    for event in events:
        session._apply_server_event(event)

    assert session.durable_oracle_records() == [
        {
            "type": VoiceEventType.ORACLE_JOB_WAITING_FOR_APPROVAL.value,
            "payload": {
                "job_id": "voice-oracle-001",
                "state": "waiting_for_approval",
                "approval_reason": "Stripe Link spend requires approval",
                "approval": {
                    "approval_id": "approval-123",
                    "tool_name": "stripe_link_purchase",
                },
            },
        },
        {
            "type": VoiceEventType.ORACLE_JOB_COMPLETED.value,
            "payload": {
                "job_id": "voice-oracle-001",
                "state": "completed",
                "result_summary": "The spend approval cleared.",
                "result_text": "The spend approval cleared with a longer durable explanation.",
                "result_text_chars": 61,
            },
        },
        {
            "type": VoiceEventType.ORACLE_JOB_FAILED.value,
            "payload": {
                "job_id": "voice-oracle-002",
                "state": "failed",
                "error": "oracle backend unavailable",
            },
        },
        {
            "type": VoiceEventType.ORACLE_JOB_CANCELLED.value,
            "payload": {
                "job_id": "voice-oracle-003",
                "state": "cancelled",
                "cancel_reason": "user cancelled queued job",
            },
        },
    ]
    assert "secret" not in str(session.durable_oracle_records())


def test_session_redacts_durable_async_oracle_record_scalars():
    session = RealtimeVoiceSession(
        RealtimeVoiceSessionConfig(session_id="voice-123"),
        engine=TextOracleTTSEngine(oracle=FakeOracle()),
    )
    test_secret = "sk" + "_test_" + "abcdefghijklmnopqrstuvwxyz"
    live_secret = "sk" + "_live_" + "abcdefghijklmnopqrstuvwxyz"

    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.ORACLE_JOB_WAITING_FOR_APPROVAL,
            session_id="voice-123",
            sequence=1,
            payload={
                "job_id": "voice-oracle-001",
                "state": "waiting_for_approval",
                "approval_reason": f"Approve spend with Authorization: Bearer raw-token and {test_secret}",
                "approval": {
                    "approval_id": "approval-123",
                    "tool_name": "stripe_link_purchase",
                    "summary": f"Charge uses {live_secret}",
                    "nested": [{"token": "raw-nested-token"}],
                },
            },
        )
    )
    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.ORACLE_JOB_COMPLETED,
            session_id="voice-123",
            sequence=2,
            payload={
                "job_id": "voice-oracle-001",
                "state": "completed",
                "result_summary": f"Provider credential {test_secret} created.",
                "result_text": f"Full provider result includes Authorization: Bearer raw-result-token and {live_secret}.",
                "result_text_chars": 123,
            },
        )
    )

    records = session.durable_oracle_records()
    serialized = json.dumps(records, sort_keys=True)

    assert len(records) == 2
    assert records[0]["payload"]["approval"]["tool_name"] == "stripe_link_purchase"
    assert records[1]["payload"]["result_text_chars"] == 123
    assert "Authorization: Bearer ***" in serialized
    assert "sk_tes" in serialized
    assert "sk_liv" in serialized
    assert "raw-token" not in serialized
    assert "raw-result-token" not in serialized
    assert "raw-nested-token" not in serialized
    assert test_secret not in serialized
    assert live_secret not in serialized


def test_session_drops_stale_durable_oracle_records_after_barge_in():
    session = RealtimeVoiceSession(
        RealtimeVoiceSessionConfig(session_id="voice-123"),
        engine=TextOracleTTSEngine(oracle=FakeOracle()),
    )
    session.transcript.active_playback_generation = 2

    for sequence, event_type in enumerate(
        [
            VoiceEventType.INTERFACE_ORACLE_REQUEST,
            VoiceEventType.ORACLE_TOOL_RESULT,
            VoiceEventType.ORACLE_RESPONSE_FINAL,
            VoiceEventType.ORACLE_ERROR,
            VoiceEventType.ORACLE_JOB_COMPLETED,
            VoiceEventType.ORACLE_JOB_FAILED,
            VoiceEventType.ORACLE_JOB_CANCELLED,
        ],
        start=1,
    ):
        session._apply_server_event(
            VoiceEvent(
                type=event_type,
                session_id="voice-123",
                sequence=sequence,
                payload={"text": "stale", "playback_generation": 1},
            )
        )

    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.ORACLE_RESPONSE_FINAL,
            session_id="voice-123",
            sequence=10,
            payload={"text": "fresh", "playback_generation": 2},
        )
    )

    assert session.durable_oracle_records() == [
        {
            "type": VoiceEventType.ORACLE_RESPONSE_FINAL.value,
            "payload": {"text": "fresh", "playback_generation": 2},
        }
    ]


def test_session_keeps_stale_completed_oracle_job_record_by_source_generation():
    class SourceGenerationEngine:
        @property
        def kind(self):
            return RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE

        async def start(self, config):
            pass

        async def receive_event(self, event):
            pass

        async def events(self):
            yield VoiceEvent(
                type=VoiceEventType.ORACLE_JOB_COMPLETED,
                session_id="voice-123",
                sequence=1,
                payload={
                    "job_id": "voice-oracle-001",
                    "state": "completed",
                    "result_summary": "Old job result",
                    "source_playback_generation": 1,
                    "playback_generation": 2,
                },
            )

        async def close(self):
            pass

    async def run():
        session = RealtimeVoiceSession(
            RealtimeVoiceSessionConfig(session_id="voice-123"),
            engine=SourceGenerationEngine(),
        )
        session.transcript.active_playback_generation = 2

        events = [event async for event in session.events()]

        assert [event.type for event in events] == [VoiceEventType.ORACLE_JOB_COMPLETED]
        assert events[0].payload["result_summary"] == "Old job result"
        assert session.durable_oracle_records() == [
            {
                "type": VoiceEventType.ORACLE_JOB_COMPLETED.value,
                "payload": {
                    "job_id": "voice-oracle-001",
                    "state": "completed",
                    "result_summary": "Old job result",
                    "source_playback_generation": 1,
                    "playback_generation": 2,
                },
            }
        ]

    asyncio.run(run())


def test_session_cancelled_oracle_job_removes_prior_completed_record():
    session = RealtimeVoiceSession(
        RealtimeVoiceSessionConfig(session_id="voice-123"),
        engine=TextOracleTTSEngine(oracle=FakeOracle()),
    )
    session.transcript.active_playback_generation = 3

    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.ORACLE_JOB_COMPLETED,
            session_id="voice-123",
            sequence=1,
            payload={
                "job_id": "voice-oracle-001",
                "state": "completed",
                "result_summary": "stale result",
                "source_playback_generation": 1,
                "playback_generation": 3,
            },
        )
    )
    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.ORACLE_JOB_CANCELLED,
            session_id="voice-123",
            sequence=2,
            payload={
                "job_id": "voice-oracle-001",
                "state": "cancelled",
                "cancel_reason": "spoken request to cancel oracle job",
                "source_playback_generation": 1,
                "playback_generation": 3,
            },
        )
    )

    assert session.durable_oracle_records() == [
        {
            "type": VoiceEventType.ORACLE_JOB_CANCELLED.value,
            "payload": {
                "job_id": "voice-oracle-001",
                "state": "cancelled",
                "cancel_reason": "spoken request to cancel oracle job",
                "source_playback_generation": 1,
                "playback_generation": 3,
            },
        }
    ]


def test_session_ignores_completed_record_after_oracle_job_cancelled():
    session = RealtimeVoiceSession(
        RealtimeVoiceSessionConfig(session_id="voice-123"),
        engine=TextOracleTTSEngine(oracle=FakeOracle()),
    )
    session.transcript.active_playback_generation = 3

    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.ORACLE_JOB_CANCELLED,
            session_id="voice-123",
            sequence=1,
            payload={
                "job_id": "voice-oracle-001",
                "state": "cancelled",
                "cancel_reason": "spoken request to cancel oracle job",
                "source_playback_generation": 1,
                "playback_generation": 3,
            },
        )
    )
    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.ORACLE_JOB_COMPLETED,
            session_id="voice-123",
            sequence=2,
            payload={
                "job_id": "voice-oracle-001",
                "state": "completed",
                "result_summary": "late stale result",
                "source_playback_generation": 1,
                "playback_generation": 3,
            },
        )
    )

    assert session.durable_oracle_records() == [
        {
            "type": VoiceEventType.ORACLE_JOB_CANCELLED.value,
            "payload": {
                "job_id": "voice-oracle-001",
                "state": "cancelled",
                "cancel_reason": "spoken request to cancel oracle job",
                "source_playback_generation": 1,
                "playback_generation": 3,
            },
        }
    ]


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


def test_session_drops_stale_oracle_tool_events_after_barge_in():
    class ToolEventEngine:
        def __init__(self):
            self.received = []
            self._events = [
                VoiceEvent(
                    type=VoiceEventType.ORACLE_TOOL_CALL,
                    session_id="voice-123",
                    sequence=10,
                    payload={
                        "tool_name": "read_file",
                        "tool_call_id": "call-stale",
                        "playback_generation": 1,
                    },
                ),
                VoiceEvent(
                    type=VoiceEventType.ORACLE_TOOL_RESULT,
                    session_id="voice-123",
                    sequence=11,
                    payload={
                        "tool_name": "read_file",
                        "tool_call_id": "call-stale",
                        "result": "old",
                        "playback_generation": 1,
                    },
                ),
                VoiceEvent(
                    type=VoiceEventType.ORACLE_TOOL_RESULT,
                    session_id="voice-123",
                    sequence=12,
                    payload={
                        "tool_name": "read_file",
                        "tool_call_id": "call-fresh",
                        "result": "new",
                        "playback_generation": 2,
                    },
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
        engine = ToolEventEngine()
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
        assert seen[0].type == VoiceEventType.ORACLE_TOOL_RESULT
        assert seen[0].payload["tool_call_id"] == "call-fresh"

    asyncio.run(run())


def test_session_drops_stale_oracle_hints_after_barge_in():
    class HintEngine:
        def __init__(self):
            self.received = []
            self._events = [
                VoiceEvent(
                    type=VoiceEventType.ORACLE_HINT,
                    session_id="voice-123",
                    sequence=10,
                    payload={
                        "delta": "old",
                        "playback_generation": 1,
                    },
                ),
                VoiceEvent(
                    type=VoiceEventType.ORACLE_HINT,
                    session_id="voice-123",
                    sequence=11,
                    payload={
                        "delta": "new",
                        "playback_generation": 2,
                    },
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
        engine = HintEngine()
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

        assert [event.sequence for event in seen] == [11]
        assert seen[0].type == VoiceEventType.ORACLE_HINT
        assert seen[0].payload["delta"] == "new"

    asyncio.run(run())


def test_session_drops_stale_kame_interface_and_alias_events_after_barge_in():
    class KameAliasEngine:
        def __init__(self):
            self.received = []
            self._events = [
                VoiceEvent(
                    type=VoiceEventType.INTERFACE_REPLY_LOCAL,
                    session_id="voice-123",
                    sequence=10,
                    payload={"text": "old local", "playback_generation": 1},
                ),
                VoiceEvent(
                    type=VoiceEventType.INTERFACE_COMMIT,
                    session_id="voice-123",
                    sequence=11,
                    payload={"text": "old commit", "playback_generation": 1},
                ),
                VoiceEvent(
                    type=VoiceEventType.ASSISTANT_AUDIO_CHUNK,
                    session_id="voice-123",
                    sequence=12,
                    payload={"codec": "opus", "data_b64": "", "playback_generation": 1},
                ),
                VoiceEvent(
                    type=VoiceEventType.INTERFACE_COMMIT,
                    session_id="voice-123",
                    sequence=13,
                    payload={"text": "fresh commit", "playback_generation": 2},
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
        engine = KameAliasEngine()
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

        assert [event.sequence for event in seen] == [13]
        assert seen[0].type == VoiceEventType.INTERFACE_COMMIT
        assert seen[0].payload["text"] == "fresh commit"

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


def test_session_marks_playback_lifecycle_and_drops_stale_events_after_barge_in():
    class LifecycleEngine:
        def __init__(self):
            self.received = []
            self._events = [
                VoiceEvent(
                    type=VoiceEventType.PLAYBACK_STARTED,
                    session_id="voice-123",
                    sequence=10,
                    payload={"playback_generation": 1},
                ),
                VoiceEvent(
                    type=VoiceEventType.PLAYBACK_STOPPED,
                    session_id="voice-123",
                    sequence=11,
                    payload={"playback_generation": 1},
                ),
                VoiceEvent(
                    type=VoiceEventType.PLAYBACK_STARTED,
                    session_id="voice-123",
                    sequence=12,
                    payload={"playback_generation": 2},
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
        session = RealtimeVoiceSession(RealtimeVoiceSessionConfig(session_id="voice-123"), engine=LifecycleEngine())
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

        assert [event.type for event in seen] == [VoiceEventType.PLAYBACK_STARTED]
        assert seen[0].payload["playback_generation"] == 2
        assert seen[0].payload["session_state"] == RealtimeVoiceSessionState.SPEAKING.value
        assert session.state == RealtimeVoiceSessionState.SPEAKING

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
                type=VoiceEventType.SPEECH_START,
                session_id="voice-123",
                sequence=1,
                payload={"user_id": "42"},
            )
        )
        await session.receive_client_event(
            VoiceEvent(
                type=VoiceEventType.SPEECH_END,
                session_id="voice-123",
                sequence=2,
                payload={"user_id": "42"},
            )
        )
        await session.receive_client_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=3,
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
        assert audio.payload["metrics"]["speech_boundary_to_first_audio_ms"] >= 0
        await session.close()

    asyncio.run(run())


def test_session_treats_assistant_audio_chunk_as_first_audio():
    class NativeAssistantAudioEngine:
        def __init__(self):
            self._events = asyncio.Queue()

        async def start(self, config):
            self.config = config

        async def receive_event(self, event):
            if event.type == VoiceEventType.AUDIO_INPUT_CHUNK and event.payload.get("end_of_utterance") is True:
                await self._events.put(
                    VoiceEvent(
                        type=VoiceEventType.TRANSCRIPT_FINAL,
                        session_id=self.config.session_id,
                        sequence=1,
                        payload={"text": "hello native audio"},
                    )
                )
                await self._events.put(
                    VoiceEvent(
                        type=VoiceEventType.ASSISTANT_AUDIO_CHUNK,
                        session_id=self.config.session_id,
                        sequence=2,
                        payload={
                            **AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"assistant-audio").to_payload(),
                            "playback_generation": 1,
                        },
                    )
                )

        async def events(self):
            while True:
                yield await self._events.get()

        async def close(self):
            return None

    async def run():
        session = RealtimeVoiceSession(
            RealtimeVoiceSessionConfig(session_id="voice-123"),
            engine=NativeAssistantAudioEngine(),
        )
        await session.start()
        await session.receive_client_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={"transcript": "hello native audio", "end_of_utterance": True},
            )
        )

        final = await anext(session.events())
        audio = await anext(session.events())

        assert final.type == VoiceEventType.TRANSCRIPT_FINAL
        assert audio.type == VoiceEventType.ASSISTANT_AUDIO_CHUNK
        assert audio.payload["metrics"]["final_transcript_to_first_audio_ms"] >= 0
        assert audio.payload["metrics"]["speech_boundary_to_first_audio_ms"] >= 0
        assert session.state == RealtimeVoiceSessionState.SPEAKING
        await session.close()

    asyncio.run(run())


def test_text_engine_file_tts_audio_chunk_includes_synthesis_metric(monkeypatch, tmp_path):
    async def run():
        audio_path = tmp_path / "tts.mp3"
        audio_path.write_bytes(b"audio")
        engine = TextOracleTTSEngine(oracle=FakeOracle())
        await engine.start(RealtimeVoiceSessionConfig(session_id="voice-123"))
        monkeypatch.setattr(engine, "_tts_sync", lambda text: str(audio_path))
        engine._playback_generation = 1
        engine._assistant_metadata_by_generation[1] = {}

        await engine._speak_chunk("hello", 1)
        event = await asyncio.wait_for(engine._events.get(), timeout=1)

        assert event.type == VoiceEventType.SESSION_STARTED
        event = await asyncio.wait_for(engine._events.get(), timeout=1)
        assert event.type == VoiceEventType.PLAYBACK_STARTED
        assert event.payload["playback_generation"] == 1
        event = await asyncio.wait_for(engine._events.get(), timeout=1)
        assert event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK
        assert event.payload["metrics"]["tts_synthesis_ms"] >= 0
        event = await asyncio.wait_for(engine._events.get(), timeout=1)
        assert event.type == VoiceEventType.PLAYBACK_STOPPED
        assert event.payload["playback_generation"] == 1
        event = await asyncio.wait_for(engine._events.get(), timeout=1)
        assert event.type == VoiceEventType.ASSISTANT_AUDIO_END
        assert event.payload["playback_generation"] == 1
        await engine.close()

    asyncio.run(run())


def test_kame_file_tts_audio_chunk_includes_first_token_to_tts_metric(monkeypatch, tmp_path):
    async def run():
        audio_path = tmp_path / "tts.mp3"
        audio_path.write_bytes(b"audio")
        engine = KameInterfaceOracleEngine(oracle=FakeOracle())
        await engine.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            )
        )
        monkeypatch.setattr(engine, "_tts_sync", lambda text: str(audio_path))
        engine._playback_generation = 7
        engine._assistant_metadata_by_generation[7] = {
            "voice_architecture": "kame_frontend_oracle",
            "kame_route": KameRoute.ORACLE_DIRECT.value,
            "metrics": {"kame_speech_end_to_interface_decision_ms": 25},
        }
        engine._interface_decision_at_by_generation[7] = time.perf_counter() - 0.1
        engine._oracle_first_token_at_by_generation[7] = time.perf_counter() - 0.05

        await engine._speak_chunk("hello", 7)
        event = await asyncio.wait_for(engine._events.get(), timeout=1)
        assert event.type == VoiceEventType.SESSION_STARTED
        event = await asyncio.wait_for(engine._events.get(), timeout=1)
        assert event.type == VoiceEventType.PLAYBACK_STARTED
        event = await asyncio.wait_for(engine._events.get(), timeout=1)
        assert event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK
        assert event.payload["metrics"]["kame_interface_decision_to_first_audio_ms"] >= 0
        assert event.payload["metrics"]["kame_speech_end_to_first_audio_ms"] >= 25
        assert event.payload["metrics"]["kame_oracle_first_token_to_first_tts_audio_ms"] >= 0
        assert event.payload["metrics"]["kame_first_tts_audio_to_playback_start_ms"] >= 0
        assert (
            engine._assistant_metadata_by_generation[7]["metrics"]["kame_oracle_first_token_to_first_tts_audio_ms"]
            == event.payload["metrics"]["kame_oracle_first_token_to_first_tts_audio_ms"]
        )
        assert (
            engine._assistant_metadata_by_generation[7]["metrics"]["kame_first_tts_audio_to_playback_start_ms"]
            == event.payload["metrics"]["kame_first_tts_audio_to_playback_start_ms"]
        )
        await engine.close()

    asyncio.run(run())


def test_session_marks_assistant_audio_end_generation_state():
    class AudioEndEngine:
        async def start(self, config):
            return None

        async def receive_event(self, event):
            return None

        async def events(self):
            yield VoiceEvent(
                type=VoiceEventType.ASSISTANT_AUDIO_END,
                session_id="voice-123",
                sequence=1,
                payload={"playback_generation": 4},
            )

        async def close(self):
            return None

    async def run():
        session = RealtimeVoiceSession(RealtimeVoiceSessionConfig(session_id="voice-123"), engine=AudioEndEngine())
        await session.start()

        event = await anext(session.events())

        assert event.type == VoiceEventType.ASSISTANT_AUDIO_END
        assert event.payload["session_state"] == RealtimeVoiceSessionState.SPEAKING.value
        assert session.state == RealtimeVoiceSessionState.SPEAKING
        assert session.transcript.active_playback_generation == 4
        assert session.durable_messages() == []
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
                quality_targets_ms={
                    "audio_to_partial_transcript_ms": 300,
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


def test_session_marks_kame_quality_target_misses(monkeypatch):
    class KameMetricsEngine:
        async def start(self, config):
            return None

        async def receive_event(self, event):
            return None

        async def events(self):
            yield VoiceEvent(
                type=VoiceEventType.SESSION_METRICS,
                session_id="voice-123",
                sequence=1,
                payload={
                    "playback_generation": 1,
                    "metrics": {
                        "kame_speech_end_to_interface_decision_ms": 650,
                        "kame_final_transcript_to_interface_decision_ms": 700,
                        "kame_speech_end_to_first_audio_ms": 3500,
                        "kame_speech_end_to_playback_start_ms": 3600,
                        "barge_in_confirmed_to_playback_stopped_ms": 180,
                    },
                },
            )

        async def close(self):
            return None

    async def run():
        session = RealtimeVoiceSession(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                quality_targets_ms={
                    "kame_speech_end_to_interface_decision_ms": 500,
                    "kame_final_transcript_to_interface_decision_ms": 500,
                    "kame_speech_end_to_first_audio_ms": 3000,
                    "kame_speech_end_to_playback_start_ms": 3000,
                    "barge_in_confirmed_to_playback_stopped_ms": 150,
                },
            ),
            engine=KameMetricsEngine(),
        )
        monkeypatch.setattr(session, "_event_metrics", lambda event: {"session_elapsed_ms": 10})
        await session.start()

        event = await anext(session.events())

        assert event.payload["quality_target_misses"] == [
            {
                "metric": "barge_in_confirmed_to_playback_stopped_ms",
                "actual_ms": 180,
                "target_ms": 150,
            },
            {
                "metric": "kame_final_transcript_to_interface_decision_ms",
                "actual_ms": 700,
                "target_ms": 500,
            },
            {
                "metric": "kame_speech_end_to_first_audio_ms",
                "actual_ms": 3500,
                "target_ms": 3000,
            },
            {
                "metric": "kame_speech_end_to_interface_decision_ms",
                "actual_ms": 650,
                "target_ms": 500,
            },
            {
                "metric": "kame_speech_end_to_playback_start_ms",
                "actual_ms": 3600,
                "target_ms": 3000,
            },
        ]
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


def test_kame_session_treats_interface_intent_final_as_turn_boundary():
    class KameEventEngine:
        async def start(self, config):
            return None

        async def receive_event(self, event):
            return None

        async def events(self):
            yield VoiceEvent(
                type=VoiceEventType.INTERFACE_INTENT_FINAL,
                session_id="voice-123",
                sequence=1,
                payload={
                    "turn_id": "voice-123:1",
                    "intent": "Check deployment status.",
                    "text": "check deployment status",
                    "route": "oracle_direct",
                    "playback_generation": 1,
                },
            )
            yield VoiceEvent(
                type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                session_id="voice-123",
                sequence=2,
                payload={"text": "Checking.", "playback_generation": 1},
            )
            yield VoiceEvent(
                type=VoiceEventType.ASSISTANT_COMMIT,
                session_id="voice-123",
                sequence=3,
                payload={"text": "Checking.", "playback_generation": 1},
            )

        async def close(self):
            return None

    async def run():
        session = RealtimeVoiceSession(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            ),
            engine=KameEventEngine(),
        )
        await session.start()

        seen = []
        async for event in session.events():
            seen.append(event)
            if event.type == VoiceEventType.ASSISTANT_COMMIT:
                break

        states = {event.type: event.payload.get("session_state") for event in seen}
        assert states[VoiceEventType.INTERFACE_INTENT_FINAL] == RealtimeVoiceSessionState.ASSISTANT_PENDING.value
        assert states[VoiceEventType.ASSISTANT_TEXT_PARTIAL] == RealtimeVoiceSessionState.SPEAKING.value
        assert states[VoiceEventType.ASSISTANT_COMMIT] == RealtimeVoiceSessionState.LISTENING.value
        assert session.state == RealtimeVoiceSessionState.LISTENING
        assert session.durable_messages() == [
            {"role": "user", "content": "Check deployment status."},
            {"role": "assistant", "content": "Checking."},
        ]
        await session.close()

    asyncio.run(run())


def test_kame_session_persists_interface_commit_as_assistant_response():
    class InterfaceCommitEngine:
        async def start(self, config):
            return None

        async def receive_event(self, event):
            return None

        async def events(self):
            yield VoiceEvent(
                type=VoiceEventType.INTERFACE_INTENT_FINAL,
                session_id="voice-123",
                sequence=1,
                payload={
                    "turn_id": "voice-123:1",
                    "intent": "Check deployment status.",
                    "text": "check deployment status",
                    "route": "local",
                    "playback_generation": 1,
                },
            )
            yield VoiceEvent(
                type=VoiceEventType.INTERFACE_REPLY_LOCAL,
                session_id="voice-123",
                sequence=2,
                payload={
                    "turn_id": "voice-123:1",
                    "text": "Checking locally.",
                    "playback_generation": 1,
                },
            )
            yield VoiceEvent(
                type=VoiceEventType.INTERFACE_COMMIT,
                session_id="voice-123",
                sequence=3,
                payload={
                    "turn_id": "voice-123:1",
                    "text": "Checking locally.",
                    "playback_generation": 1,
                },
            )

        async def close(self):
            return None

    async def run():
        session = RealtimeVoiceSession(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
            ),
            engine=InterfaceCommitEngine(),
        )
        await session.start()

        seen = []
        async for event in session.events():
            seen.append(event)
            if event.type == VoiceEventType.INTERFACE_COMMIT:
                break

        states = {event.type: event.payload.get("session_state") for event in seen}
        assert states[VoiceEventType.INTERFACE_INTENT_FINAL] == RealtimeVoiceSessionState.ASSISTANT_PENDING.value
        assert states[VoiceEventType.INTERFACE_REPLY_LOCAL] == RealtimeVoiceSessionState.SPEAKING.value
        assert states[VoiceEventType.INTERFACE_COMMIT] == RealtimeVoiceSessionState.LISTENING.value
        assert session.durable_messages() == [
            {"role": "user", "content": "Check deployment status."},
            {"role": "assistant", "content": "Checking locally."},
        ]
        await session.close()

    asyncio.run(run())


def test_kame_session_deduplicates_interface_and_assistant_commit_for_same_turn():
    session = RealtimeVoiceSession(
        RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        ),
        engine=KameInterfaceOracleEngine(oracle=FakeOracle()),
    )

    session._apply_server_event(
        VoiceEvent(
            type=VoiceEventType.INTERFACE_INTENT_FINAL,
            session_id="voice-123",
            sequence=1,
            payload={
                "turn_id": "voice-123:1",
                "intent": "Check deployment status.",
                "text": "check deployment status",
                "route": "local",
                "playback_generation": 7,
            },
        )
    )
    for event_type in (VoiceEventType.INTERFACE_COMMIT, VoiceEventType.ASSISTANT_COMMIT):
        session._apply_server_event(
            VoiceEvent(
                type=event_type,
                session_id="voice-123",
                sequence=2,
                payload={
                    "turn_id": "voice-123:1",
                    "text": "Checking locally.",
                    "playback_generation": 7,
                },
            )
        )

    assert session.durable_messages() == [
        {"role": "user", "content": "Check deployment status."},
        {"role": "assistant", "content": "Checking locally."},
    ]


def test_kame_session_deduplicates_transcript_and_interface_final_for_same_turn():
    session = RealtimeVoiceSession(
        RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        ),
        engine=KameInterfaceOracleEngine(oracle=FakeOracle()),
    )

    for event_type in (VoiceEventType.TRANSCRIPT_FINAL, VoiceEventType.INTERFACE_INTENT_FINAL):
        session._apply_server_event(
            VoiceEvent(
                type=event_type,
                session_id="voice-123",
                sequence=1,
                payload={
                    "turn_id": "voice-123:1",
                    "kame_turn_id": "voice-123:1",
                    "text": "check deployment status",
                    "intent": "Check deployment status.",
                    "kame_intent": "Check deployment status.",
                    "voice_architecture": "kame_frontend_oracle",
                    "playback_generation": 1,
                },
            )
        )

    assert session.durable_messages() == [{"role": "user", "content": "Check deployment status."}]


def test_kame_session_does_not_persist_oracle_job_status_poll_messages():
    session = RealtimeVoiceSession(
        RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
        ),
        engine=KameInterfaceOracleEngine(oracle=FakeOracle()),
    )

    common = {
        "turn_id": "voice-123:1",
        "intent": "What are you working on?",
        "text": "what are you working on",
        "route": "local",
        "playback_generation": 7,
        "oracle_job_status_poll": True,
        "durable": False,
    }
    for sequence, event_type in enumerate(
        (VoiceEventType.TRANSCRIPT_FINAL, VoiceEventType.INTERFACE_INTENT_FINAL),
        start=1,
    ):
        session._apply_server_event(
            VoiceEvent(
                type=event_type,
                session_id="voice-123",
                sequence=sequence,
                payload=dict(common),
            )
        )
    for event_type in (VoiceEventType.INTERFACE_COMMIT, VoiceEventType.ASSISTANT_COMMIT):
        session._apply_server_event(
            VoiceEvent(
                type=event_type,
                session_id="voice-123",
                sequence=10,
                payload={
                    **common,
                    "text": "Oracle jobs: 1 running out of 1. running: Checking logs.",
                    "local_reply": True,
                },
            )
        )

    assert session.durable_messages() == []
    assert session.state == RealtimeVoiceSessionState.LISTENING


def test_session_treats_caption_events_as_ephemeral_state():
    class CaptionEngine:
        async def start(self, config):
            return None

        async def receive_event(self, event):
            return None

        async def events(self):
            yield VoiceEvent(
                type=VoiceEventType.ASSISTANT_CAPTION_PARTIAL,
                session_id="voice-123",
                sequence=1,
                payload={"text": "Draft caption.", "playback_generation": 2},
            )
            yield VoiceEvent(
                type=VoiceEventType.ASSISTANT_CAPTION_FINAL,
                session_id="voice-123",
                sequence=2,
                payload={"text": "Final caption.", "playback_generation": 2},
            )

        async def close(self):
            return None

    async def run():
        session = RealtimeVoiceSession(RealtimeVoiceSessionConfig(session_id="voice-123"), engine=CaptionEngine())
        await session.start()

        events = session.events()
        partial = await anext(events)
        final = await anext(events)

        assert partial.payload["session_state"] == RealtimeVoiceSessionState.SPEAKING.value
        assert final.payload["session_state"] == RealtimeVoiceSessionState.LISTENING.value
        assert session.state == RealtimeVoiceSessionState.LISTENING
        assert session.transcript.active_playback_generation == 2
        assert session.durable_messages() == []
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


def test_session_adds_barge_in_to_playback_stopped_latency_metric():
    class BargeStopEngine:
        def __init__(self):
            self.received = []

        async def start(self, config):
            return None

        async def receive_event(self, event):
            self.received.append(event)

        async def events(self):
            yield VoiceEvent(
                type=VoiceEventType.BARGE_IN,
                session_id="voice-123",
                sequence=10,
                payload={"reason": "user_speech", "playback_generation": 1},
            )
            yield VoiceEvent(
                type=VoiceEventType.PLAYBACK_STOPPED,
                session_id="voice-123",
                sequence=11,
                payload={"reason": "barge_in", "playback_generation": 1},
            )

        async def close(self):
            return None

    async def run():
        session = RealtimeVoiceSession(
            RealtimeVoiceSessionConfig(session_id="voice-123"),
            engine=BargeStopEngine(),
        )
        await session.start()
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

        barge_in = seen[0]
        stopped = seen[1]
        assert barge_in.type == VoiceEventType.BARGE_IN
        assert stopped.type == VoiceEventType.PLAYBACK_STOPPED
        assert barge_in.payload["metrics"]["barge_in_ack_ms"] >= 0
        assert stopped.payload["metrics"]["barge_in_confirmed_to_playback_stopped_ms"] >= 0
        assert "barge_in_confirmed_to_playback_stopped_ms" not in barge_in.payload["metrics"]
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
        assert event.type == VoiceEventType.PLAYBACK_STARTED
        assert event.payload["playback_generation"] == 6
        event = await engine._events.get()
        assert event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK
        assert event.session_id == "voice-123"
        assert event.payload["playback_generation"] == 6
        assert AudioChunk.from_payload(event.payload).data == b"s2s-speaker"

    asyncio.run(run())


def test_native_s2s_engine_accepts_binary_assistant_audio_frame():
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
            type=VoiceEventType.ASSISTANT_AUDIO_CHUNK,
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
        assert event.type == VoiceEventType.PLAYBACK_STARTED
        assert event.payload["playback_generation"] == 6
        event = await engine._events.get()
        assert event.type == VoiceEventType.ASSISTANT_AUDIO_CHUNK
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
        assert event.type == VoiceEventType.PLAYBACK_STARTED
        assert event.payload["playback_generation"] == 3
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


def test_native_s2s_engine_reader_error_degrades_then_fails_session():
    class FailingWs:
        def __init__(self):
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise RuntimeError("sidecar failed at http://user:pass@voice.local/v1?token=abc")

        async def close(self):
            self.closed = True

    async def run():
        ws = FailingWs()
        engine = NativeS2SSidecarEngine()
        engine.config = RealtimeVoiceSessionConfig(
            session_id="voice-123",
            engine=RealtimeVoiceEngineKind.NATIVE_S2S_ORACLE,
            sidecar_base_url="ws://voice.local",
        )
        engine._ws = ws
        engine._assistant_output_active = True
        engine._auto_barge_in_input_active = True

        await engine._read_sidecar()

        degraded = await engine._events.get()
        error = await engine._events.get()

        assert degraded.type == VoiceEventType.FRONTEND_STATE
        assert degraded.payload["status"] == "degraded"
        assert degraded.payload["reason"] == "native_s2s_sidecar_disconnected"
        assert degraded.payload["sidecar"] is False
        assert degraded.payload["native_s2s"] is False
        assert "user:pass" not in degraded.payload["error"]
        assert "token=abc" not in degraded.payload["error"]
        assert error.type == VoiceEventType.SESSION_ERROR
        assert "native S2S sidecar failed" in error.payload["error"]
        assert "user:pass" not in error.payload["error"]
        assert "token=abc" not in error.payload["error"]
        assert ws.closed is True
        assert engine._ws is None
        assert engine._assistant_output_active is False
        assert engine._auto_barge_in_input_active is False

    asyncio.run(run())
