import asyncio

from fastapi.testclient import TestClient

from agent.realtime_voice import (
    AudioChunk,
    RealtimeVoiceSessionConfig,
    VoiceAudioCodec,
    VoiceEvent,
    VoiceEventType,
    binary_audio_frame_from_event,
    event_from_binary_audio_frame,
)
from agent.realtime_voice_local_speech_bridge import (
    LocalSpeechProxyBridgeConfig,
    create_local_speech_proxy_bridge_app,
    local_speech_proxy_health_payload,
    local_speech_proxy_prerequisite_issues,
)


def test_local_speech_proxy_health_advertises_stt_when_upstream_is_ready():
    runtime = LocalSpeechProxyBridgeConfig(
        provider="nemotron_speech",
        role="stt",
        model="nemotron-speech-streaming-0.6b",
        upstream_base_url="http://127.0.0.1:9101",
        input_languages=("en", "ja"),
    )

    payload = local_speech_proxy_health_payload(
        runtime,
        upstream_health={"ok": True, "capabilities": {"streaming_stt": True}},
    )

    assert payload["ok"] is True
    assert payload["kind"] == "local_speech_proxy_bridge"
    assert payload["frontend"]["provider"] == "nemotron_speech"
    assert payload["frontend"]["role"] == "stt"
    assert payload["frontend"]["languages"] == ["en", "ja"]
    assert payload["capabilities"]["streaming_stt"] is True
    assert payload["capabilities"]["utterance_stt"] is True
    assert payload["capabilities"]["native_s2s"] is False


def test_local_speech_proxy_health_advertises_tts_when_upstream_is_ready():
    runtime = LocalSpeechProxyBridgeConfig(
        provider="magpie_tts",
        role="tts",
        model="magpie-local-streaming-tts",
        upstream_base_url="http://127.0.0.1:9102",
        output_languages=("en", "ja"),
    )

    payload = local_speech_proxy_health_payload(
        runtime,
        upstream_health={"ok": True, "capabilities": {"tts": True}},
    )

    assert payload["ok"] is True
    assert payload["frontend"]["provider"] == "magpie_tts"
    assert payload["frontend"]["role"] == "tts"
    assert payload["frontend"]["tts_model_languages"] == ["en", "ja"]
    assert payload["capabilities"]["tts"] is True
    assert payload["capabilities"]["streaming_tts"] is True
    assert payload["capabilities"]["native_s2s"] is False


def test_local_speech_proxy_health_fails_closed_without_ready_upstream():
    runtime = LocalSpeechProxyBridgeConfig(
        provider="nemotron_speech",
        role="stt",
        model="nemotron-speech-streaming-0.6b",
        upstream_base_url="http://127.0.0.1:9101",
    )

    missing_capability = local_speech_proxy_health_payload(
        runtime,
        upstream_health={"ok": True, "capabilities": {"tts": True}},
    )
    unhealthy = local_speech_proxy_health_payload(
        runtime,
        upstream_health={"ok": False, "error": "Bearer secret-token failed"},
    )
    unconfigured = local_speech_proxy_health_payload(
        LocalSpeechProxyBridgeConfig(
            provider="nemotron_speech",
            role="stt",
            model="nemotron-speech-streaming-0.6b",
        ),
        upstream_health={"ok": True, "capabilities": {"streaming_stt": True}},
    )

    assert missing_capability["ok"] is False
    assert missing_capability["capabilities"]["streaming_stt"] is False
    assert unhealthy["ok"] is False
    assert "secret-token" not in unhealthy["upstream"]["error"]
    assert unconfigured["ok"] is False
    assert unconfigured["frontend"]["upstream_configured"] is False


def test_local_speech_proxy_prerequisites_require_upstream_and_capability():
    runtime = LocalSpeechProxyBridgeConfig(
        provider="nemotron_speech",
        role="stt",
        model="nemotron-speech-streaming-0.6b",
    )

    issues = local_speech_proxy_prerequisite_issues(
        runtime,
        require_auth_token=True,
        upstream_health={"ok": True, "capabilities": {"tts": True}},
    )

    assert "HERMES_NEMOTRON_SPEECH_UPSTREAM_BASE_URL is required" in issues
    assert "HERMES_NEMOTRON_SPEECH_BRIDGE_TOKEN is required" in issues
    assert "upstream does not advertise required stt capability" in issues


def test_local_speech_proxy_forwards_stt_session_to_configured_upstream():
    runtime = LocalSpeechProxyBridgeConfig(
        provider="nemotron_speech",
        role="stt",
        model="nemotron-speech-streaming-0.6b",
        upstream_base_url="http://127.0.0.1:9101",
        upstream_token="upstream-token",
        auth_token="bridge-token",
    )
    client = TestClient(
        create_local_speech_proxy_bridge_app(
            runtime,
            health_probe=lambda _runtime: {"ok": True, "capabilities": {"streaming_stt": True}},
            client_factory=_FakeSidecarClient,
        )
    )

    config = RealtimeVoiceSessionConfig(session_id="voice-123")
    with client.websocket_connect(
        "/v1/streaming-stt/session",
        headers={"Authorization": "Bearer bridge-token"},
    ) as ws:
        ws.send_json({"type": "session.config", "payload": config.to_wire()})
        ready = ws.receive_json()
        assert ready["type"] == VoiceEventType.FRONTEND_STATE.value

        frame = binary_audio_frame_from_event(
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
        assert frame is not None
        ws.send_bytes(frame)
        final = ws.receive_json()

    fake = _FakeSidecarClient.instances[-1]
    assert fake.path == "/v1/streaming-stt/session"
    assert fake.config is not None
    assert fake.config.frontend_provider == "nemotron_speech"
    assert fake.config.frontend_model == "nemotron-speech-streaming-0.6b"
    assert fake.config.asr_provider == "nemotron_speech"
    assert fake.config.asr_model == "nemotron-speech-streaming-0.6b"
    assert fake.config.effective_sidecar_base_url == "http://127.0.0.1:9101"
    assert fake.config.effective_sidecar_token == "upstream-token"
    assert fake.sent[0].type == VoiceEventType.AUDIO_INPUT_CHUNK
    assert final["type"] == VoiceEventType.TRANSCRIPT_FINAL.value
    assert final["payload"]["text"] == "local proxy heard audio"
    assert fake.closed is True


def test_local_speech_proxy_forwards_tts_session_and_binary_audio():
    runtime = LocalSpeechProxyBridgeConfig(
        provider="magpie_tts",
        role="tts",
        model="magpie-local-streaming-tts",
        upstream_base_url="http://127.0.0.1:9102",
    )
    client = TestClient(
        create_local_speech_proxy_bridge_app(
            runtime,
            health_probe=lambda _runtime: {"ok": True, "capabilities": {"streaming_tts": True}},
            client_factory=_FakeSidecarClient,
        )
    )

    config = RealtimeVoiceSessionConfig(session_id="voice-456")
    with client.websocket_connect("/v1/streaming-tts/session") as ws:
        ws.send_json({"type": "session.config", "payload": config.to_wire()})
        ready = ws.receive_json()
        assert ready["type"] == VoiceEventType.FRONTEND_STATE.value
        ws.send_json(
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                session_id="voice-456",
                sequence=1,
                payload={"text": "Checking.", "speak": True, "playback_generation": 7},
            ).to_wire()
        )
        started = ws.receive_json()
        audio = event_from_binary_audio_frame(ws.receive_bytes(), expected_type=VoiceEventType.AUDIO_OUTPUT_CHUNK)

    fake = _FakeSidecarClient.instances[-1]
    assert fake.path == "/v1/streaming-tts/session"
    assert fake.config is not None
    assert fake.config.frontend_provider == "magpie_tts"
    assert fake.config.frontend_model == "magpie-local-streaming-tts"
    assert fake.config.tts_provider == "magpie_tts"
    assert fake.config.tts_model == "magpie-local-streaming-tts"
    assert fake.sent[0].type == VoiceEventType.ASSISTANT_TEXT_PARTIAL
    assert started["type"] == VoiceEventType.PLAYBACK_STARTED.value
    assert audio.payload["playback_generation"] == 7
    assert audio.payload["sample_rate_hz"] == 16000
    assert fake.closed is True


class _FakeSidecarClient:
    instances = []

    def __init__(self, *, path):
        self.path = path
        self.config = None
        self.sent = []
        self.closed = False
        self._events = asyncio.Queue()
        self.instances.append(self)

    async def start(self, config):
        self.config = config
        await self._events.put(
            VoiceEvent(
                type=VoiceEventType.FRONTEND_STATE,
                session_id=config.session_id,
                sequence=1,
                payload={"provider": config.frontend_provider, "model": config.frontend_model},
            )
        )

    async def send_event(self, event):
        self.sent.append(event)
        if event.type == VoiceEventType.AUDIO_INPUT_CHUNK:
            await self._events.put(
                VoiceEvent(
                    type=VoiceEventType.TRANSCRIPT_FINAL,
                    session_id=event.session_id,
                    sequence=2,
                    payload={"text": "local proxy heard audio", "is_final": True},
                )
            )
        if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL:
            playback_generation = event.payload.get("playback_generation")
            await self._events.put(
                VoiceEvent(
                    type=VoiceEventType.PLAYBACK_STARTED,
                    session_id=event.session_id,
                    sequence=2,
                    payload={"playback_generation": playback_generation},
                )
            )
            await self._events.put(
                VoiceEvent(
                    type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
                    session_id=event.session_id,
                    sequence=3,
                    payload={
                        **AudioChunk(codec=VoiceAudioCodec.PCM16, data=b"pcm").to_payload(),
                        "sample_rate_hz": 16000,
                        "channels": 1,
                        "playback_generation": playback_generation,
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
        self.closed = True
        await self._events.put(None)
