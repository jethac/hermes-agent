import asyncio

from agent.realtime_voice import (
    AudioChunk,
    RealtimeVoiceEngineKind,
    RealtimeVoiceSessionConfig,
    VoiceAudioCodec,
    VoiceEvent,
    VoiceEventType,
)
from agent.realtime_voice_smoke import (
    RealtimeVoiceSidecarSmokeResult,
    realtime_voice_smoke_result_payload,
    realtime_voice_smoke_text_metadata,
    run_realtime_voice_session_audio_smoke,
    run_realtime_voice_session_turn_smoke,
    run_realtime_voice_sidecar_barge_in_smoke,
    run_realtime_voice_sidecar_tts_smoke,
)


def test_tts_smoke_sends_language_metadata(monkeypatch):
    sent = []

    class FakeSidecarClient:
        async def start(self, config):
            self.config = config

        async def send_event(self, event):
            sent.append(event)

        async def events(self):
            yield VoiceEvent(
                type=VoiceEventType.FRONTEND_STATE,
                session_id=self.config.session_id,
                sequence=1,
                payload={"status": "ready"},
            )
            while not sent:
                await asyncio.sleep(0)
            yield VoiceEvent(
                type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
                session_id=self.config.session_id,
                sequence=2,
                payload=AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"audio").to_payload(),
            )

        async def close(self):
            return None

    monkeypatch.setattr(
        "agent.realtime_voice_smoke.RealtimeVoiceSidecarClient",
        FakeSidecarClient,
    )

    result = asyncio.run(
        run_realtime_voice_sidecar_tts_smoke(
            RealtimeVoiceSessionConfig(
                session_id="voice-smoke",
                sidecar_base_url="http://voice.example.test:8765",
            ),
            text="こんにちは、Hermesです。",
            metadata=realtime_voice_smoke_text_metadata("こんにちは、Hermesです。"),
            timeout_seconds=1,
        )
    )

    assert result.ok is True
    assert sent[0].payload["language"] == "ja"
    assert sent[0].payload["locale"] == "ja-JP"
    assert sent[0].payload["script"] == "Jpan"


def test_tts_smoke_accepts_native_assistant_audio_chunk(monkeypatch):
    sent = []

    class FakeSidecarClient:
        async def start(self, config):
            self.config = config

        async def send_event(self, event):
            sent.append(event)

        async def events(self):
            yield VoiceEvent(
                type=VoiceEventType.FRONTEND_STATE,
                session_id=self.config.session_id,
                sequence=1,
                payload={"status": "ready"},
            )
            while not sent:
                await asyncio.sleep(0)
            yield VoiceEvent(
                type=VoiceEventType.ASSISTANT_AUDIO_CHUNK,
                session_id=self.config.session_id,
                sequence=2,
                payload=AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"audio").to_payload(),
            )

        async def close(self):
            return None

    monkeypatch.setattr(
        "agent.realtime_voice_smoke.RealtimeVoiceSidecarClient",
        FakeSidecarClient,
    )

    result = asyncio.run(
        run_realtime_voice_sidecar_tts_smoke(
            RealtimeVoiceSessionConfig(
                session_id="voice-smoke",
                sidecar_base_url="http://voice.example.test:8765",
            ),
            text="Hello from Hermes.",
            timeout_seconds=1,
        )
    )

    assert result.ok is True
    assert result.output_audio_bytes == len(b"audio")
    assert result.events == ("frontend.state", "assistant.audio.chunk")


def test_session_turn_smoke_measures_first_text_and_audio(monkeypatch):
    async def fake_speak_chunk(self, text, playback_generation):
        await self._emit(
            VoiceEventType.AUDIO_OUTPUT_CHUNK,
            {
                **AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"audio").to_payload(),
                "playback_generation": playback_generation,
                "metrics": {"tts_synthesis_ms": 123},
            },
        )

    monkeypatch.setattr(
        "agent.realtime_voice_text_engine.TextOracleTTSEngine._speak_chunk",
        fake_speak_chunk,
    )

    result = asyncio.run(
        run_realtime_voice_session_turn_smoke(
            RealtimeVoiceSessionConfig(session_id="voice-smoke"),
            answer="Hello from Hermes.",
            transcript="hello",
            timeout_seconds=1,
        )
    )

    assert result.ok is True
    assert result.first_text_ms is not None
    assert result.first_audio_ms is not None
    assert result.first_audio_metrics["tts_synthesis_ms"] == 123
    assert result.output_audio_bytes == len(b"audio")
    assert result.final_text == "Hello from Hermes."
    assert result.events == (
        "session.started",
        "transcript.final",
        "assistant.text.partial",
        "audio.output.chunk",
        "assistant.commit",
    )


def test_session_audio_smoke_uses_sidecar_stt_and_tts_in_one_session():
    sent = []

    class FakeSidecar:
        async def start(self, config):
            self.config = config

        async def send_event(self, event):
            sent.append(event)

        async def speak(self, event):
            sent.append(event)

        async def events(self):
            yield VoiceEvent(
                type=VoiceEventType.FRONTEND_STATE,
                session_id=self.config.session_id,
                sequence=1,
                payload={"status": "ready"},
            )
            while not any(event.type == VoiceEventType.AUDIO_INPUT_CHUNK for event in sent):
                await asyncio.sleep(0)
            yield VoiceEvent(
                type=VoiceEventType.TRANSCRIPT_PARTIAL,
                session_id=self.config.session_id,
                sequence=2,
                payload={"text": "Hello from", "input_generation": 1},
            )
            yield VoiceEvent(
                type=VoiceEventType.TRANSCRIPT_FINAL,
                session_id=self.config.session_id,
                sequence=3,
                payload={"text": "Hello from Hermes.", "input_generation": 1},
            )
            while not any(event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL for event in sent):
                await asyncio.sleep(0)
            yield VoiceEvent(
                type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
                session_id=self.config.session_id,
                sequence=4,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"audio").to_payload(),
                    "metrics": {"tts_synthesis_ms": 456},
                },
            )

        async def close(self):
            return None

    result = asyncio.run(
        run_realtime_voice_session_audio_smoke(
            RealtimeVoiceSessionConfig(
                session_id="voice-smoke",
                sidecar_base_url="http://voice.example.test:8765",
            ),
            audio=b"webm bytes",
            answer="Hello from Hermes.",
            timeout_seconds=1,
            sidecar=FakeSidecar(),
        )
    )

    assert result.ok is True
    assert result.audio_bytes == len(b"webm bytes")
    assert result.output_audio_bytes == len(b"audio")
    assert result.final_text == "Hello from Hermes."
    assert result.transcript_partial_ms is not None
    assert result.first_text_ms is not None
    assert result.first_audio_ms is not None
    assert result.first_audio_metrics["tts_synthesis_ms"] == 456
    assert result.events == (
        "session.started",
        "frontend.state",
        "transcript.partial",
        "transcript.final",
        "assistant.text.partial",
        "audio.output.chunk",
    )


def test_kame_audio_smoke_captures_reflex_route_without_partial_transcript(monkeypatch):
    sent = []

    async def fake_speak_chunk(self, text, playback_generation):
        await self._emit(
            VoiceEventType.AUDIO_OUTPUT_CHUNK,
            {
                **AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"kame-audio").to_payload(),
                "playback_generation": playback_generation,
                "metrics": {"tts_synthesis_ms": 12},
            },
        )

    monkeypatch.setattr(
        "agent.realtime_voice_text_engine.TextOracleTTSEngine._speak_chunk",
        fake_speak_chunk,
    )

    class FakeKameSidecar:
        async def start(self, config):
            self.config = config

        async def send_event(self, event):
            sent.append(event)

        async def speak(self, event):
            sent.append(event)

        async def events(self):
            yield VoiceEvent(
                type=VoiceEventType.FRONTEND_STATE,
                session_id=self.config.session_id,
                sequence=1,
                payload={"status": "ready"},
            )
            while not any(event.type == VoiceEventType.AUDIO_INPUT_CHUNK for event in sent):
                await asyncio.sleep(0)
            yield VoiceEvent(
                type=VoiceEventType.INTERFACE_INTENT_FINAL,
                session_id=self.config.session_id,
                sequence=2,
                payload={
                    "text": "can you hear me",
                    "intent": "The user is checking whether Hermes can hear them.",
                    "route": "local",
                    "route_confidence": 0.94,
                    "local_reply": "Yes, I can hear you.",
                    "interface_input_source": "native_audio",
                    "reflex_provider": "vllm",
                    "end_of_utterance": True,
                    "input_generation": 1,
                },
            )

        async def close(self):
            return None

    result = asyncio.run(
        run_realtime_voice_session_audio_smoke(
            RealtimeVoiceSessionConfig(
                session_id="voice-smoke",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                frontend_provider="gemma4",
                frontend_model="gemma-4-E2B-it",
                interface_audio_input="native_audio",
                sidecar_base_url="http://voice.example.test:8765",
            ),
            audio=b"webm bytes",
            answer="unused oracle answer",
            timeout_seconds=1,
            sidecar=FakeKameSidecar(),
        )
    )

    assert result.ok is True
    assert result.transcript_partial_ms is None
    assert result.route == "local"
    assert result.interface_input_source == "native_audio"
    assert result.reflex_provider == "vllm"
    assert result.final_text == "can you hear me"
    assert result.assistant_final_text == "Yes, I can hear you."
    assert result.output_audio_bytes == len(b"kame-audio")
    assert "interface.intent.final" in result.events
    payload = realtime_voice_smoke_result_payload(result, kind="audio_session")
    assert payload["route"] == "local"
    assert payload["final_text"] == "can you hear me"
    assert payload["assistant_final_text"] == "Yes, I can hear you."
    assert payload["metrics"]["tts_synthesis_ms"] == 12


def test_smoke_result_payload_preserves_kame_reflex_validation_error():
    payload = realtime_voice_smoke_result_payload(
        RealtimeVoiceSidecarSmokeResult(
            ok=True,
            route="oracle_direct",
            interface_input_source="native_audio",
            reflex_provider="vllm",
            reflex_validation_error="invalid_json",
            first_audio_metrics={"kame_speech_end_to_first_audio_ms": 250},
        ),
        kind="audio_session",
    )

    assert payload["route"] == "oracle_direct"
    assert payload["interface_input_source"] == "native_audio"
    assert payload["reflex_provider"] == "vllm"
    assert payload["reflex_validation_error"] == "invalid_json"
    assert payload["metrics"]["kame_speech_end_to_first_audio_ms"] == 250


def test_barge_in_smoke_measures_ack_latency(monkeypatch):
    sent = []

    class FakeSidecarClient:
        async def start(self, config):
            self.config = config

        async def send_event(self, event):
            sent.append(event)

        async def events(self):
            yield VoiceEvent(
                type=VoiceEventType.FRONTEND_STATE,
                session_id=self.config.session_id,
                sequence=1,
                payload={"status": "ready"},
            )
            while not any(event.type == VoiceEventType.BARGE_IN for event in sent):
                await asyncio.sleep(0)
            yield VoiceEvent(
                type=VoiceEventType.BARGE_IN,
                session_id=self.config.session_id,
                sequence=2,
                payload={"playback_generation": 2},
            )

        async def close(self):
            return None

    monkeypatch.setattr(
        "agent.realtime_voice_smoke.RealtimeVoiceSidecarClient",
        FakeSidecarClient,
    )

    result = asyncio.run(
        run_realtime_voice_sidecar_barge_in_smoke(
            RealtimeVoiceSessionConfig(
                session_id="voice-smoke",
                sidecar_base_url="http://voice.example.test:8765",
            ),
            text="Hello from Hermes.",
            timeout_seconds=1,
        )
    )

    assert result.ok is True
    assert result.barge_in_ack_ms is not None
    assert result.audio_after_barge_in_bytes == 0
    assert result.events == ("frontend.state", "barge_in.detected")
    assert sent[0].type == VoiceEventType.ASSISTANT_TEXT_PARTIAL
    assert sent[0].payload["speak"] is True
    assert sent[1].type == VoiceEventType.BARGE_IN
    assert sent[1].payload["playback_generation"] == 2


def test_barge_in_smoke_does_not_count_startup_backlog_as_ack_latency(monkeypatch):
    sent = []

    class FakeSidecarClient:
        async def start(self, config):
            self.config = config

        async def send_event(self, event):
            sent.append(event)

        async def events(self):
            for index in range(3):
                await asyncio.sleep(0.02)
                yield VoiceEvent(
                    type=VoiceEventType.FRONTEND_STATE,
                    session_id=self.config.session_id,
                    sequence=index + 1,
                    payload={"status": "degraded" if index < 2 else "ready"},
                )
            while not any(event.type == VoiceEventType.BARGE_IN for event in sent):
                await asyncio.sleep(0)
            yield VoiceEvent(
                type=VoiceEventType.BARGE_IN,
                session_id=self.config.session_id,
                sequence=4,
                payload={"playback_generation": 2},
            )

        async def close(self):
            return None

    monkeypatch.setattr(
        "agent.realtime_voice_smoke.RealtimeVoiceSidecarClient",
        FakeSidecarClient,
    )

    result = asyncio.run(
        run_realtime_voice_sidecar_barge_in_smoke(
            RealtimeVoiceSessionConfig(
                session_id="voice-smoke",
                sidecar_base_url="http://voice.example.test:8765",
            ),
            text="Hello from Hermes.",
            timeout_seconds=1,
        )
    )

    assert result.ok is True
    assert result.ready_ms is not None
    assert result.barge_in_ack_ms is not None
    assert result.barge_in_ack_ms < 20
    assert result.audio_after_barge_in_bytes == 0
    assert result.events == (
        "frontend.state",
        "frontend.state",
        "frontend.state",
        "barge_in.detected",
    )


def test_barge_in_smoke_rejects_audio_after_barge_in_ack(monkeypatch):
    sent = []

    class FakeSidecarClient:
        async def start(self, config):
            self.config = config

        async def send_event(self, event):
            sent.append(event)

        async def events(self):
            yield VoiceEvent(
                type=VoiceEventType.FRONTEND_STATE,
                session_id=self.config.session_id,
                sequence=1,
                payload={"status": "ready"},
            )
            while not any(event.type == VoiceEventType.BARGE_IN for event in sent):
                await asyncio.sleep(0)
            yield VoiceEvent(
                type=VoiceEventType.BARGE_IN,
                session_id=self.config.session_id,
                sequence=2,
                payload={"playback_generation": 2},
            )
            yield VoiceEvent(
                type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
                session_id=self.config.session_id,
                sequence=3,
                payload=AudioChunk(codec=VoiceAudioCodec.OPUS, data=b"stale").to_payload(),
            )

        async def close(self):
            return None

    monkeypatch.setattr(
        "agent.realtime_voice_smoke.RealtimeVoiceSidecarClient",
        FakeSidecarClient,
    )

    result = asyncio.run(
        run_realtime_voice_sidecar_barge_in_smoke(
            RealtimeVoiceSessionConfig(
                session_id="voice-smoke",
                sidecar_base_url="http://voice.example.test:8765",
            ),
            post_barge_in_quiet_seconds=0.1,
            text="Hello from Hermes.",
            timeout_seconds=1,
        )
    )

    assert result.ok is False
    assert result.audio_after_barge_in_bytes == len(b"stale")
    assert "output audio chunk arrived after barge_in" in result.error
    assert result.events == ("frontend.state", "barge_in.detected", "audio.output.chunk")
