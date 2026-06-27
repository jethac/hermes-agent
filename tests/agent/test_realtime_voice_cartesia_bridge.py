from agent.realtime_voice import AudioChunk, RealtimeVoiceSessionConfig, VoiceAudioCodec, VoiceEventType
from agent.realtime_voice_cartesia_bridge import (
    CartesiaRealtimeBridgeConfig,
    cartesia_bridge_config_from_env,
    cartesia_bridge_prerequisite_issues,
    cartesia_stt_audio_bytes,
    cartesia_stt_message_to_transcript_payload,
    cartesia_stt_url,
    cartesia_tts_generation_message,
    cartesia_tts_url,
    create_cartesia_realtime_bridge_app,
)


def test_cartesia_stt_url_uses_manual_pcm_defaults():
    url = cartesia_stt_url(
        CartesiaRealtimeBridgeConfig(
            stt_url="wss://api.cartesia.ai/stt/websocket",
            api_version="2026-03-01",
            model="ink-2",
            stt_sample_rate_hz=16000,
        ),
        RealtimeVoiceSessionConfig(
            session_id="voice-123",
            input_codec=VoiceAudioCodec.PCM16,
            sample_rate_hz=48000,
            channels=2,
        ),
    )

    assert url.startswith("wss://api.cartesia.ai/stt/websocket?")
    assert "model=ink-2" in url
    assert "encoding=pcm_s16le" in url
    assert "sample_rate=16000" in url
    assert "cartesia_version=2026-03-01" in url


def test_cartesia_tts_url_includes_api_version():
    url = cartesia_tts_url(
        CartesiaRealtimeBridgeConfig(tts_url="wss://api.cartesia.ai/tts/websocket", api_version="2026-03-01")
    )

    assert url == "wss://api.cartesia.ai/tts/websocket?cartesia_version=2026-03-01"


def test_cartesia_tts_generation_message_uses_voice_model_language_and_pcm_output():
    message = cartesia_tts_generation_message(
        CartesiaRealtimeBridgeConfig(
            tts_model="sonic-3.5",
            voice_id="voice-123",
            language="en",
            tts_sample_rate_hz=24000,
            tts_model_by_language={"ja": "sonic-3.5"},
            tts_voice_by_language={"ja": "voice-ja"},
        ),
        {"language": "ja-JP"},
        "hello",
        "ctx-1",
    )

    assert message == {
        "model_id": "sonic-3.5",
        "transcript": "hello",
        "voice": {"mode": "id", "id": "voice-ja"},
        "language": "ja",
        "context_id": "ctx-1",
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": 24000,
        },
        "continue": False,
    }


def test_cartesia_stt_message_to_transcript_payload_maps_partial_and_final():
    partial_type, partial = cartesia_stt_message_to_transcript_payload(
        {"type": "turn.update", "text": "hello"},
        input_generation=5,
    )
    final_type, final = cartesia_stt_message_to_transcript_payload(
        {"type": "turn.end", "transcript": "hello Hermes", "language": "en-US"},
        input_generation=5,
    )

    assert partial_type == VoiceEventType.TRANSCRIPT_PARTIAL
    assert partial == {"text": "hello", "input_generation": 5}
    assert final_type == VoiceEventType.TRANSCRIPT_FINAL
    assert final == {"text": "hello Hermes", "language": "en-US", "input_generation": 5}


def test_cartesia_runtime_reads_env(monkeypatch):
    monkeypatch.setenv("CARTESIA_API_KEY", "cartesia-secret")
    monkeypatch.delenv("CARTESIA_VOICE_ID", raising=False)
    monkeypatch.setenv("HERMES_STREAMING_STT_BRIDGE_TOKEN", "bridge-token")
    monkeypatch.setenv("HERMES_CARTESIA_STT_URL", "wss://cartesia.example.test/stt")
    monkeypatch.setenv("HERMES_CARTESIA_TTS_URL", "wss://cartesia.example.test/tts")
    monkeypatch.setenv("HERMES_CARTESIA_STT_MODEL", "ink-2")
    monkeypatch.setenv("HERMES_CARTESIA_TTS_MODEL", "sonic-3.5")
    monkeypatch.setenv("HERMES_CARTESIA_VOICE_ID", "voice-123")
    monkeypatch.setenv("HERMES_CARTESIA_LANGUAGE", "en-US")
    monkeypatch.setenv("HERMES_CARTESIA_OUTPUT_LANGUAGES", "en-US,ja-JP,https://bad.example/x")
    monkeypatch.setenv("HERMES_CARTESIA_TTS_MODEL_BY_LANGUAGE", "ja:sonic-3.5")
    monkeypatch.setenv("HERMES_CARTESIA_TTS_VOICE_BY_LANGUAGE", "ja:voice-ja")
    monkeypatch.setenv("HERMES_CARTESIA_STT_SAMPLE_RATE_HZ", "16000")
    monkeypatch.setenv("HERMES_CARTESIA_TTS_SAMPLE_RATE_HZ", "24000")
    monkeypatch.setenv("HERMES_CARTESIA_CONNECT_TIMEOUT_SECONDS", "2.5")

    runtime = cartesia_bridge_config_from_env()

    assert runtime.api_key == "cartesia-secret"
    assert runtime.auth_token == "bridge-token"
    assert runtime.stt_url == "wss://cartesia.example.test/stt"
    assert runtime.tts_url == "wss://cartesia.example.test/tts"
    assert runtime.model == "ink-2"
    assert runtime.tts_model == "sonic-3.5"
    assert runtime.voice_id == "voice-123"
    assert runtime.language == "en-US"
    assert runtime.output_languages == ("en", "ja")
    assert runtime.tts_model_by_language == {"ja": "sonic-3.5"}
    assert runtime.tts_voice_by_language == {"ja": "voice-ja"}
    assert runtime.stt_sample_rate_hz == 16000
    assert runtime.tts_sample_rate_hz == 24000
    assert runtime.connect_timeout_seconds == 2.5


def test_cartesia_bridge_prerequisite_check_reports_missing_requirements():
    issues = cartesia_bridge_prerequisite_issues(
        CartesiaRealtimeBridgeConfig(api_key=None, auth_token=None, voice_id=""),
        require_auth_token=True,
        module_available=lambda name: False,
    )

    assert "CARTESIA_API_KEY or HERMES_CARTESIA_API_KEY is required" in issues
    assert "CARTESIA_VOICE_ID or HERMES_CARTESIA_VOICE_ID is required for streaming TTS" in issues
    assert any("websockets==15.0.1" in issue for issue in issues)
    assert "HERMES_STREAMING_STT_BRIDGE_TOKEN is required in strict mode" in issues


def test_cartesia_bridge_prerequisite_check_accepts_en_ja_tts_routing():
    issues = cartesia_bridge_prerequisite_issues(
        CartesiaRealtimeBridgeConfig(
            api_key="cartesia-secret",
            auth_token="bridge-token",
            voice_id="voice-123",
            language="en",
            output_languages=("en", "ja"),
        ),
        require_auth_token=True,
        required_output_languages=("en", "ja"),
        module_available=lambda name: name == "websockets",
    )

    assert issues == []


def test_cartesia_bridge_health_advertises_streaming_stt_tts():
    from fastapi.testclient import TestClient

    client = TestClient(
        create_cartesia_realtime_bridge_app(
            CartesiaRealtimeBridgeConfig(
                api_key="cartesia-secret",
                voice_id="voice-123",
                language="en",
                output_languages=("en", "ja"),
            )
        )
    )

    body = client.get("/health").json()

    assert body["ok"] is True
    assert body["frontend"]["provider"] == "cartesia"
    assert body["frontend"]["voice_id"] == "configured"
    assert body["capabilities"]["streaming_stt"] is True
    assert body["capabilities"]["streaming_tts"] is True
    assert body["capabilities"]["input_languages"] == ["en"]
    assert body["capabilities"]["output_languages"] == ["en", "ja"]


def test_cartesia_stt_audio_bytes_passes_matching_pcm16_through():
    chunk = AudioChunk(codec=VoiceAudioCodec.PCM16, data=b"pcm", sample_rate_hz=16000, channels=1)

    assert cartesia_stt_audio_bytes(chunk, target_sample_rate_hz=16000, target_channels=1) == b"pcm"


def test_cartesia_stt_audio_bytes_resamples_mismatched_pcm16(monkeypatch):
    calls = []

    def fake_convert(audio, *, input_sample_rate_hz, input_channels, output_sample_rate_hz, output_channels):
        calls.append((audio, input_sample_rate_hz, input_channels, output_sample_rate_hz, output_channels))
        return b"converted-pcm"

    monkeypatch.setattr("agent.realtime_voice_cartesia_bridge._ffmpeg_raw_pcm16le", fake_convert)
    chunk = AudioChunk(codec=VoiceAudioCodec.PCM16, data=b"pcm", sample_rate_hz=48000, channels=2)

    assert cartesia_stt_audio_bytes(chunk, target_sample_rate_hz=16000, target_channels=1) == b"converted-pcm"
    assert calls == [(b"pcm", 48000, 2, 16000, 1)]


def test_cartesia_stt_audio_bytes_converts_compressed_audio(monkeypatch):
    calls = []

    def fake_convert(audio, *, codec, sample_rate_hz, channels):
        calls.append((audio, codec, sample_rate_hz, channels))
        return b"converted-pcm"

    monkeypatch.setattr("agent.realtime_voice_cartesia_bridge._ffmpeg_to_pcm16le", fake_convert)
    chunk = AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"webm", sample_rate_hz=48000, channels=2)

    assert cartesia_stt_audio_bytes(chunk, target_sample_rate_hz=16000, target_channels=1) == b"converted-pcm"
    assert calls == [(b"webm", VoiceAudioCodec.WEBM_OPUS, 16000, 1)]


def test_cartesia_bridge_cli_check_loads_hermes_env(monkeypatch, capsys):
    from hermes_cli import realtime_voice_cartesia_bridge

    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_VOICE_ID", raising=False)
    monkeypatch.delenv("HERMES_STREAMING_STT_BRIDGE_TOKEN", raising=False)
    monkeypatch.delenv("HERMES_CARTESIA_BRIDGE_TOKEN_ENV", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.load_env",
        lambda: {
            "CARTESIA_API_KEY": "cartesia-secret",
            "CARTESIA_VOICE_ID": "voice-123",
            "HERMES_STREAMING_STT_BRIDGE_TOKEN": "bridge-token",
        },
    )

    result = realtime_voice_cartesia_bridge.main(["--check", "--strict", "--production-en-ja"])

    assert result == 0
    output = capsys.readouterr().out
    assert "Cartesia realtime voice bridge check OK" in output
    assert "cartesia-secret" not in output
    assert "bridge-token" not in output
    assert "voice-123" not in output


def test_cartesia_bridge_cli_generates_bridge_token(monkeypatch, capsys):
    from hermes_cli import realtime_voice_cartesia_bridge

    saved = {}
    monkeypatch.delenv("HERMES_STREAMING_STT_BRIDGE_TOKEN", raising=False)
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda key, value: saved.setdefault(key, value))

    result = realtime_voice_cartesia_bridge.main(["--generate-token"])

    assert result == 0
    assert "HERMES_STREAMING_STT_BRIDGE_TOKEN" in saved
    assert len(saved["HERMES_STREAMING_STT_BRIDGE_TOKEN"]) >= 32
    assert saved["HERMES_STREAMING_STT_BRIDGE_TOKEN"] not in capsys.readouterr().out
