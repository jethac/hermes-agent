import asyncio
import base64
import json
import os
import sys
import types

from agent.realtime_voice import (
    AudioChunk,
    RealtimeVoiceSessionConfig,
    VoiceAudioCodec,
    VoiceEvent,
    VoiceEventType,
)
from agent.realtime_voice_elevenlabs_bridge import (
    ElevenLabsRealtimeBridgeConfig,
    ElevenLabsStreamingSTTBridgeSession,
    ElevenLabsStreamingTTSBridgeSession,
    create_elevenlabs_realtime_bridge_app,
    elevenlabs_bridge_config_from_env,
    elevenlabs_bridge_prerequisite_issues,
    elevenlabs_stt_audio_bytes,
    elevenlabs_stt_message_to_transcript_payload,
    elevenlabs_stt_url,
    elevenlabs_tts_start_message,
    elevenlabs_tts_url,
)


def test_elevenlabs_stt_url_uses_realtime_defaults_for_pcm16():
    url = elevenlabs_stt_url(
        ElevenLabsRealtimeBridgeConfig(
            stt_url="wss://api.elevenlabs.io/v1/speech-to-text/realtime",
            model="scribe_v2_realtime",
            language="ja",
        ),
        RealtimeVoiceSessionConfig(
            session_id="voice-123",
            input_codec=VoiceAudioCodec.PCM16,
            sample_rate_hz=16000,
            channels=1,
        ),
    )

    assert url.startswith("wss://api.elevenlabs.io/v1/speech-to-text/realtime?")
    assert "model_id=scribe_v2_realtime" in url
    assert "audio_format=pcm_16000" in url
    assert "commit_strategy=manual" in url
    assert "include_language_detection=true" in url
    assert "language_code=ja" in url


def test_elevenlabs_stt_url_omits_language_code_for_auto_multilingual():
    url = elevenlabs_stt_url(
        ElevenLabsRealtimeBridgeConfig(language="auto"),
        RealtimeVoiceSessionConfig(session_id="voice-123", sample_rate_hz=24000),
    )

    assert "audio_format=pcm_24000" in url
    assert "language_code" not in url


def test_elevenlabs_tts_url_uses_voice_model_and_pcm_output_format():
    url = elevenlabs_tts_url(
        ElevenLabsRealtimeBridgeConfig(
            tts_url="wss://api.elevenlabs.io/v1/text-to-speech",
            voice_id="voice/with spaces",
            tts_model="eleven_flash_v2_5",
            output_format="pcm_24000",
        )
    )

    assert url.startswith("wss://api.elevenlabs.io/v1/text-to-speech/voice%2Fwith%20spaces/stream-input?")
    assert "model_id=eleven_flash_v2_5" in url
    assert "output_format=pcm_24000" in url


def test_elevenlabs_tts_start_message_does_not_drop_generation_settings():
    message = elevenlabs_tts_start_message(
        ElevenLabsRealtimeBridgeConfig(
            api_key="elevenlabs-secret",
            voice_settings={"stability": 0.5},
            chunk_length_schedule=(80, 120),
        )
    )

    assert message == {
        "text": " ",
        "xi_api_key": "elevenlabs-secret",
        "generation_config": {"chunk_length_schedule": [80, 120]},
        "voice_settings": {"stability": 0.5},
    }


def test_elevenlabs_stt_message_to_transcript_payload_maps_partial_and_final():
    partial_type, partial = elevenlabs_stt_message_to_transcript_payload(
        {"message_type": "partial_transcript", "text": "hello"},
        input_generation=5,
    )
    final_type, final = elevenlabs_stt_message_to_transcript_payload(
        {
            "message_type": "committed_transcript_with_timestamps",
            "text": "hello Hermes",
            "language_code": "en-US",
        },
        input_generation=5,
    )

    assert partial_type == VoiceEventType.TRANSCRIPT_PARTIAL
    assert partial == {"text": "hello", "input_generation": 5}
    assert final_type == VoiceEventType.TRANSCRIPT_FINAL
    assert final == {"text": "hello Hermes", "language": "en-US", "input_generation": 5}


def test_elevenlabs_runtime_reads_env(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "elevenlabs-secret")
    monkeypatch.setenv("HERMES_STREAMING_STT_BRIDGE_TOKEN", "bridge-token")
    monkeypatch.setenv("HERMES_ELEVENLABS_STT_URL", "wss://elevenlabs.example.test/stt")
    monkeypatch.setenv("HERMES_ELEVENLABS_TTS_URL", "wss://elevenlabs.example.test/tts")
    monkeypatch.setenv("HERMES_ELEVENLABS_STT_MODEL", "scribe_v2_realtime")
    monkeypatch.setenv("HERMES_ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5")
    monkeypatch.setenv("HERMES_ELEVENLABS_VOICE_ID", "voice-123")
    monkeypatch.setenv("HERMES_ELEVENLABS_LANGUAGE", "en-US")
    monkeypatch.setenv("HERMES_ELEVENLABS_OUTPUT_FORMAT", "pcm_16000")
    monkeypatch.setenv("HERMES_ELEVENLABS_OUTPUT_LANGUAGES", "en-US,ja-JP,https://bad.example/x")
    monkeypatch.setenv("HERMES_ELEVENLABS_VOICE_SETTINGS", '{"stability": 0.4}')
    monkeypatch.setenv("HERMES_ELEVENLABS_CHUNK_LENGTH_SCHEDULE", "60,90,bad")
    monkeypatch.setenv("HERMES_ELEVENLABS_CONNECT_TIMEOUT_SECONDS", "2.5")

    runtime = elevenlabs_bridge_config_from_env()

    assert runtime.api_key == "elevenlabs-secret"
    assert runtime.auth_token == "bridge-token"
    assert runtime.stt_url == "wss://elevenlabs.example.test/stt"
    assert runtime.tts_url == "wss://elevenlabs.example.test/tts"
    assert runtime.model == "scribe_v2_realtime"
    assert runtime.tts_model == "eleven_flash_v2_5"
    assert runtime.voice_id == "voice-123"
    assert runtime.language == "en-US"
    assert runtime.output_format == "pcm_16000"
    assert runtime.tts_sample_rate_hz == 16000
    assert runtime.output_languages == ("en", "ja")
    assert runtime.voice_settings == {"stability": 0.4}
    assert runtime.chunk_length_schedule == (60, 90)
    assert runtime.connect_timeout_seconds == 2.5


def test_elevenlabs_bridge_prerequisite_check_reports_missing_requirements():
    issues = elevenlabs_bridge_prerequisite_issues(
        ElevenLabsRealtimeBridgeConfig(api_key=None, auth_token=None, voice_id=""),
        require_auth_token=True,
        module_available=lambda name: False,
    )

    assert "ELEVENLABS_API_KEY or HERMES_ELEVENLABS_API_KEY is required" in issues
    assert "ELEVENLABS_VOICE_ID or HERMES_ELEVENLABS_VOICE_ID is required for streaming TTS" in issues
    assert any("websockets==15.0.1" in issue for issue in issues)
    assert "HERMES_STREAMING_STT_BRIDGE_TOKEN is required in strict mode" in issues


def test_elevenlabs_bridge_prerequisite_check_accepts_en_ja_defaults():
    issues = elevenlabs_bridge_prerequisite_issues(
        ElevenLabsRealtimeBridgeConfig(
            api_key="elevenlabs-secret",
            auth_token="bridge-token",
            voice_id="voice-123",
            language="auto",
            tts_model="eleven_flash_v2_5",
        ),
        require_auth_token=True,
        required_input_languages=("en", "ja"),
        required_output_languages=("en", "ja"),
        module_available=lambda name: name == "websockets",
    )

    assert issues == []


def test_elevenlabs_bridge_health_advertises_streaming_stt_tts():
    from fastapi.testclient import TestClient

    client = TestClient(
        create_elevenlabs_realtime_bridge_app(
            ElevenLabsRealtimeBridgeConfig(
                api_key="elevenlabs-secret",
                voice_id="voice-123",
                language="auto",
                tts_model="eleven_flash_v2_5",
            )
        )
    )

    body = client.get("/health").json()

    assert body["ok"] is True
    assert body["frontend"]["provider"] == "elevenlabs"
    assert body["frontend"]["voice_id"] == "configured"
    assert body["capabilities"]["streaming_stt"] is True
    assert body["capabilities"]["streaming_tts"] is True
    assert body["capabilities"]["input_languages"] == ["en", "ja"]
    assert body["capabilities"]["output_languages"] == ["en", "ja"]


def test_elevenlabs_stt_audio_bytes_passes_pcm16_through():
    chunk = AudioChunk(codec=VoiceAudioCodec.PCM16, data=b"pcm", sample_rate_hz=16000, channels=1)

    assert elevenlabs_stt_audio_bytes(chunk) == b"pcm"


def test_elevenlabs_stt_audio_bytes_converts_compressed_audio(monkeypatch):
    calls = []

    def fake_convert(audio, *, codec, sample_rate_hz, channels):
        calls.append((audio, codec, sample_rate_hz, channels))
        return b"converted-pcm"

    monkeypatch.setattr("agent.realtime_voice_elevenlabs_bridge._ffmpeg_to_pcm16le", fake_convert)
    chunk = AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"webm", sample_rate_hz=16000, channels=1)

    assert elevenlabs_stt_audio_bytes(chunk) == b"converted-pcm"
    assert calls == [(b"webm", VoiceAudioCodec.WEBM_OPUS, 16000, 1)]


def test_elevenlabs_bridge_cli_check_loads_hermes_env(monkeypatch, capsys):
    from hermes_cli import realtime_voice_elevenlabs_bridge

    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_STREAMING_STT_BRIDGE_TOKEN", raising=False)
    monkeypatch.delenv("HERMES_ELEVENLABS_BRIDGE_TOKEN_ENV", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.load_env",
        lambda: {
            "ELEVENLABS_API_KEY": "elevenlabs-secret",
            "ELEVENLABS_VOICE_ID": "voice-123",
            "HERMES_STREAMING_STT_BRIDGE_TOKEN": "bridge-token",
        },
    )

    result = realtime_voice_elevenlabs_bridge.main(["--check", "--strict", "--production-en-ja"])

    assert result == 0
    output = capsys.readouterr().out
    assert "ElevenLabs realtime voice bridge check OK" in output
    assert "elevenlabs-secret" not in output
    assert "bridge-token" not in output
    assert "voice-123" not in output


def test_elevenlabs_bridge_cli_generates_bridge_token(monkeypatch, capsys):
    from hermes_cli import realtime_voice_elevenlabs_bridge

    saved = {}
    monkeypatch.delenv("HERMES_STREAMING_STT_BRIDGE_TOKEN", raising=False)
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda key, value: saved.setdefault(key, value))

    result = realtime_voice_elevenlabs_bridge.main(["--generate-token"])

    assert result == 0
    assert "HERMES_STREAMING_STT_BRIDGE_TOKEN" in saved
    assert len(saved["HERMES_STREAMING_STT_BRIDGE_TOKEN"]) >= 32
    output = capsys.readouterr().out
    assert "stored in HERMES_STREAMING_STT_BRIDGE_TOKEN" in output
    assert saved["HERMES_STREAMING_STT_BRIDGE_TOKEN"] not in output


def test_elevenlabs_stt_session_streams_partial_and_final_events(monkeypatch):
    captured = {}

    class FakeElevenLabsSTTWebSocket:
        def __init__(self):
            self.sent = []
            self.closed = False
            self._messages = asyncio.Queue()

        async def send(self, payload):
            self.sent.append(payload)
            data = json.loads(payload)
            if data["message_type"] == "input_audio_chunk":
                await self._messages.put(json.dumps({"message_type": "partial_transcript", "text": "hello"}))
                if data["commit"] is True:
                    await self._messages.put(
                        json.dumps(
                            {
                                "message_type": "committed_transcript_with_timestamps",
                                "text": "hello Hermes",
                                "language_code": "en-US",
                            }
                        )
                    )

        async def close(self):
            self.closed = True
            await self._messages.put(None)

        def __aiter__(self):
            return self

        async def __anext__(self):
            item = await self._messages.get()
            if item is None:
                raise StopAsyncIteration
            return item

    fake_ws = FakeElevenLabsSTTWebSocket()

    async def fake_connect(url, additional_headers=None, extra_headers=None):
        captured["url"] = url
        captured["headers"] = additional_headers or extra_headers
        return fake_ws

    monkeypatch.setitem(sys.modules, "websockets", types.SimpleNamespace(connect=fake_connect))

    async def run():
        session = ElevenLabsStreamingSTTBridgeSession(
            ElevenLabsRealtimeBridgeConfig(
                api_key="elevenlabs-secret",
                model="scribe_v2_realtime",
                language="auto",
                connect_timeout_seconds=1,
            )
        )
        await session.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                input_codec=VoiceAudioCodec.PCM16,
                sample_rate_hz=16000,
            )
        )
        await session.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.PCM16, data=b"audio", sample_rate_hz=16000).to_payload(),
                    "input_generation": 9,
                    "end_of_utterance": True,
                },
            )
        )

        seen = []
        async for event in session.events():
            seen.append(event)
            if event.type == VoiceEventType.TRANSCRIPT_FINAL:
                await session.close()
                break

        sent = json.loads(fake_ws.sent[0])
        assert captured["headers"] == {"xi-api-key": "elevenlabs-secret"}
        assert captured["url"].startswith("wss://api.elevenlabs.io/v1/speech-to-text/realtime?")
        assert sent["message_type"] == "input_audio_chunk"
        assert base64.b64decode(sent["audio_base_64"]) == b"audio"
        assert sent["commit"] is True
        assert sent["sample_rate"] == 16000
        assert [event.type for event in seen] == [
            VoiceEventType.FRONTEND_STATE,
            VoiceEventType.TRANSCRIPT_PARTIAL,
            VoiceEventType.TRANSCRIPT_FINAL,
        ]
        assert seen[2].payload == {
            "text": "hello Hermes",
            "language": "en-US",
            "input_generation": 9,
        }

    asyncio.run(run())


def test_elevenlabs_tts_session_streams_audio_and_barge_in(monkeypatch):
    captured_urls = []
    sockets = []

    class FakeElevenLabsTTSWebSocket:
        def __init__(self):
            self.sent = []
            self.closed = False
            self._messages = asyncio.Queue()
            sockets.append(self)

        async def send(self, payload):
            self.sent.append(payload)
            data = json.loads(payload)
            if data.get("text") == "" and data.get("flush") is True:
                await self._messages.put(json.dumps({"audio": base64.b64encode(b"pcm-audio").decode("ascii")}))

        async def close(self):
            self.closed = True
            await self._messages.put(None)

        def __aiter__(self):
            return self

        async def __anext__(self):
            item = await self._messages.get()
            if item is None:
                raise StopAsyncIteration
            return item

    async def fake_connect(url, additional_headers=None, extra_headers=None):
        captured_urls.append(url)
        return FakeElevenLabsTTSWebSocket()

    monkeypatch.setitem(sys.modules, "websockets", types.SimpleNamespace(connect=fake_connect))

    async def run():
        session = ElevenLabsStreamingTTSBridgeSession(
            ElevenLabsRealtimeBridgeConfig(
                api_key="elevenlabs-secret",
                voice_id="voice-123",
                tts_model="eleven_flash_v2_5",
                output_format="pcm_24000",
                tts_sample_rate_hz=24000,
                connect_timeout_seconds=1,
            )
        )
        await session.start(RealtimeVoiceSessionConfig(session_id="voice-123"))
        await session.receive_event(
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                session_id="voice-123",
                sequence=1,
                payload={
                    "text": "hello from Hermes",
                    "speak": True,
                    "playback_generation": 3,
                },
            )
        )

        seen = []
        async for event in session.events():
            seen.append(event)
            if event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK:
                break

        await session.receive_event(
            VoiceEvent(
                type=VoiceEventType.BARGE_IN,
                session_id="voice-123",
                sequence=2,
                payload={"reason": "user_speech", "playback_generation": 4},
            )
        )
        async for event in session.events():
            seen.append(event)
            if event.type == VoiceEventType.BARGE_IN:
                await session.close()
                break

        assert len(sockets) == 2
        assert captured_urls[0].startswith(
            "wss://api.elevenlabs.io/v1/text-to-speech/voice-123/stream-input?"
        )
        assert json.loads(sockets[0].sent[0])["xi_api_key"] == "elevenlabs-secret"
        assert json.loads(sockets[0].sent[1]) == {
            "text": "hello from Hermes",
            "try_trigger_generation": True,
        }
        assert json.loads(sockets[0].sent[2]) == {"text": "", "flush": True}
        assert json.loads(sockets[0].sent[3]) == {"text": ""}
        assert [event.type for event in seen] == [
            VoiceEventType.FRONTEND_STATE,
            VoiceEventType.AUDIO_OUTPUT_CHUNK,
            VoiceEventType.BARGE_IN,
        ]
        audio = AudioChunk.from_payload(seen[1].payload)
        assert audio.codec == VoiceAudioCodec.PCM16
        assert audio.data == b"pcm-audio"
        assert audio.sample_rate_hz == 24000
        assert seen[1].payload["playback_generation"] == 3
        assert seen[2].payload["playback_generation"] == 4

    asyncio.run(run())
