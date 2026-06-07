import asyncio
import json
from pathlib import Path
import sys
import tomllib
import types

from agent.realtime_voice import (
    AudioChunk,
    RealtimeVoiceSessionConfig,
    VoiceAudioCodec,
    VoiceEvent,
    VoiceEventType,
)
from agent.realtime_voice_deepgram_bridge import (
    DeepgramStreamingSTTBridgeConfig,
    DeepgramStreamingSTTBridgeSession,
    DeepgramStreamingTTSBridgeSession,
    deepgram_bridge_config_from_env,
    deepgram_bridge_prerequisite_issues,
    deepgram_listen_url,
    deepgram_result_to_transcript_payload,
    deepgram_tts_url,
)


def test_deepgram_listen_url_uses_streaming_defaults_for_pcm16():
    url = deepgram_listen_url(
        DeepgramStreamingSTTBridgeConfig(
            deepgram_url="wss://api.deepgram.com/v1/listen",
            model="nova-3",
            language="ja",
            endpointing_ms=80,
        ),
        RealtimeVoiceSessionConfig(
            session_id="voice-123",
            input_codec=VoiceAudioCodec.PCM16,
            sample_rate_hz=24000,
            channels=1,
        ),
    )

    assert url.startswith("wss://api.deepgram.com/v1/listen?")
    assert "model=nova-3" in url
    assert "interim_results=true" in url
    assert "endpointing=80" in url
    assert "language=ja" in url
    assert "encoding=linear16" in url
    assert "sample_rate=24000" in url


def test_deepgram_tts_url_uses_streaming_audio_defaults():
    url = deepgram_tts_url(
        DeepgramStreamingSTTBridgeConfig(
            deepgram_tts_url="wss://api.deepgram.com/v1/speak",
            tts_model="aura-2-thalia-en",
            tts_sample_rate_hz=24000,
        )
    )

    assert url.startswith("wss://api.deepgram.com/v1/speak?")
    assert "model=aura-2-thalia-en" in url
    assert "encoding=linear16" in url
    assert "sample_rate=24000" in url


def test_deepgram_result_to_transcript_payload_sanitizes_metadata():
    payload = deepgram_result_to_transcript_payload(
        {
            "type": "Results",
            "channel": {
                "alternatives": [
                    {
                        "transcript": "こんにちは Hermes",
                        "confidence": 0.91,
                        "languages": ["ja-JP"],
                        "words": [{"language": "https://voice.local/secret"}],
                    }
                ]
            },
            "speech_final": True,
        },
        input_generation=7,
    )

    assert payload == {
        "text": "こんにちは Hermes",
        "confidence": 0.91,
        "language": "ja-JP",
        "input_generation": 7,
    }


def test_deepgram_runtime_reads_env(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "deepgram-secret")
    monkeypatch.setenv("HERMES_STREAMING_STT_BRIDGE_TOKEN", "bridge-token")
    monkeypatch.setenv("HERMES_DEEPGRAM_LISTEN_URL", "wss://deepgram.example.test/v1/listen")
    monkeypatch.setenv("HERMES_DEEPGRAM_TTS_URL", "wss://deepgram.example.test/v1/speak")
    monkeypatch.setenv("HERMES_DEEPGRAM_MODEL", "nova-3")
    monkeypatch.setenv("HERMES_DEEPGRAM_TTS_MODEL", "aura-2-thalia-en")
    monkeypatch.setenv("HERMES_DEEPGRAM_LANGUAGE", "en-US")
    monkeypatch.setenv("HERMES_DEEPGRAM_TTS_SAMPLE_RATE_HZ", "48000")
    monkeypatch.setenv("HERMES_DEEPGRAM_ENDPOINTING_MS", "120")
    monkeypatch.setenv("HERMES_DEEPGRAM_CONNECT_TIMEOUT_SECONDS", "2.5")

    runtime = deepgram_bridge_config_from_env()

    assert runtime.api_key == "deepgram-secret"
    assert runtime.auth_token == "bridge-token"
    assert runtime.deepgram_url == "wss://deepgram.example.test/v1/listen"
    assert runtime.deepgram_tts_url == "wss://deepgram.example.test/v1/speak"
    assert runtime.model == "nova-3"
    assert runtime.tts_model == "aura-2-thalia-en"
    assert runtime.language == "en-US"
    assert runtime.tts_sample_rate_hz == 48000
    assert runtime.endpointing_ms == 120
    assert runtime.connect_timeout_seconds == 2.5


def test_deepgram_bridge_prerequisite_check_reports_missing_requirements():
    issues = deepgram_bridge_prerequisite_issues(
        DeepgramStreamingSTTBridgeConfig(api_key=None, auth_token=None),
        require_auth_token=True,
        module_available=lambda name: False,
    )

    assert "DEEPGRAM_API_KEY or HERMES_DEEPGRAM_API_KEY is required" in issues
    assert any("websockets==15.0.1" in issue for issue in issues)
    assert "HERMES_STREAMING_STT_BRIDGE_TOKEN is required in strict mode" in issues


def test_deepgram_bridge_prerequisite_check_accepts_runtime_with_websockets():
    issues = deepgram_bridge_prerequisite_issues(
        DeepgramStreamingSTTBridgeConfig(
            api_key="deepgram-secret",
            auth_token="bridge-token",
        ),
        require_auth_token=True,
        module_available=lambda name: name == "websockets",
    )

    assert issues == []


def test_deepgram_bridge_cli_check_reports_missing_env(monkeypatch, capsys):
    from hermes_cli import realtime_voice_deepgram_bridge

    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_STREAMING_STT_BRIDGE_TOKEN", raising=False)
    monkeypatch.setattr(
        "agent.realtime_voice_deepgram_bridge._module_available",
        lambda name: name != "websockets",
    )

    result = realtime_voice_deepgram_bridge.main(["--check", "--strict"])

    assert result == 1
    output = capsys.readouterr().out
    assert "Deepgram realtime voice bridge check failed" in output
    assert "DEEPGRAM_API_KEY" in output
    assert "websockets" in output
    assert "HERMES_STREAMING_STT_BRIDGE_TOKEN" in output


def test_deepgram_bridge_cli_generates_bridge_token(monkeypatch, capsys):
    from hermes_cli import realtime_voice_deepgram_bridge

    saved = {}
    monkeypatch.delenv("HERMES_STREAMING_STT_BRIDGE_TOKEN", raising=False)
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda key, value: saved.setdefault(key, value))

    result = realtime_voice_deepgram_bridge.main(["--generate-token"])

    assert result == 0
    assert "HERMES_STREAMING_STT_BRIDGE_TOKEN" in saved
    assert len(saved["HERMES_STREAMING_STT_BRIDGE_TOKEN"]) >= 32
    output = capsys.readouterr().out
    assert "stored in HERMES_STREAMING_STT_BRIDGE_TOKEN" in output
    assert saved["HERMES_STREAMING_STT_BRIDGE_TOKEN"] not in output


def test_deepgram_bridge_cli_does_not_overwrite_existing_token(monkeypatch, capsys):
    from hermes_cli import realtime_voice_deepgram_bridge

    calls = []
    monkeypatch.setenv("HERMES_STREAMING_STT_BRIDGE_TOKEN", "existing-token")
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda key, value: calls.append((key, value)))

    result = realtime_voice_deepgram_bridge.main(["--generate-token"])

    assert result == 0
    assert calls == []
    assert "already configured" in capsys.readouterr().out


def test_deepgram_bridge_cli_generate_token_accepts_custom_env(monkeypatch):
    from hermes_cli import realtime_voice_deepgram_bridge

    saved = {}
    monkeypatch.delenv("CUSTOM_BRIDGE_TOKEN", raising=False)
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda key, value: saved.setdefault(key, value))

    result = realtime_voice_deepgram_bridge.main(
        ["--generate-token", "--token-env", "CUSTOM_BRIDGE_TOKEN"]
    )

    assert result == 0
    assert "CUSTOM_BRIDGE_TOKEN" in saved


def test_voice_extra_installs_websocket_client():
    pyproject = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )

    voice_extra = pyproject["project"]["optional-dependencies"]["voice"]
    assert "websockets==15.0.1" in voice_extra


def test_deepgram_session_streams_partial_and_final_events(monkeypatch):
    captured = {}

    class FakeDeepgramWebSocket:
        def __init__(self):
            self.sent = []
            self.closed = False
            self._messages = asyncio.Queue()

        async def send(self, payload):
            self.sent.append(payload)
            if isinstance(payload, bytes):
                await self._messages.put(
                    json.dumps(
                        {
                            "type": "Results",
                            "channel": {
                                "alternatives": [
                                    {
                                        "transcript": "hello",
                                        "confidence": 0.7,
                                        "languages": ["en-US"],
                                    }
                                ]
                            },
                            "is_final": False,
                            "speech_final": False,
                        }
                    )
                )
            elif json.loads(payload).get("type") == "Finalize":
                await self._messages.put(
                    json.dumps(
                        {
                            "type": "Results",
                            "channel": {
                                "alternatives": [
                                    {
                                        "transcript": "hello Hermes",
                                        "confidence": 0.94,
                                        "languages": ["en-US"],
                                    }
                                ]
                            },
                            "from_finalize": True,
                            "speech_final": True,
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

    fake_ws = FakeDeepgramWebSocket()

    async def fake_connect(url, additional_headers=None, extra_headers=None):
        captured["url"] = url
        captured["headers"] = additional_headers or extra_headers
        return fake_ws

    monkeypatch.setitem(sys.modules, "websockets", types.SimpleNamespace(connect=fake_connect))

    async def run():
        session = DeepgramStreamingSTTBridgeSession(
            DeepgramStreamingSTTBridgeConfig(
                api_key="deepgram-secret",
                model="nova-3",
                language="en-US",
                connect_timeout_seconds=1,
            )
        )
        await session.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                input_codec=VoiceAudioCodec.WEBM_OPUS,
            )
        )
        await session.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-123",
                sequence=1,
                payload={
                    **AudioChunk(codec=VoiceAudioCodec.WEBM_OPUS, data=b"audio").to_payload(),
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

        assert captured["headers"] == {"Authorization": "Token deepgram-secret"}
        assert captured["url"].startswith("wss://api.deepgram.com/v1/listen?")
        assert "interim_results=true" in captured["url"]
        assert fake_ws.sent[0] == b"audio"
        assert json.loads(fake_ws.sent[1]) == {"type": "Finalize"}
        assert [event.type for event in seen] == [
            VoiceEventType.FRONTEND_STATE,
            VoiceEventType.TRANSCRIPT_PARTIAL,
            VoiceEventType.TRANSCRIPT_FINAL,
        ]
        assert seen[1].payload == {
            "text": "hello",
            "confidence": 0.7,
            "language": "en-US",
            "input_generation": 9,
        }
        assert seen[2].payload == {
            "text": "hello Hermes",
            "confidence": 0.94,
            "language": "en-US",
            "input_generation": 9,
        }

    asyncio.run(run())


def test_deepgram_tts_session_streams_audio_and_barge_in(monkeypatch):
    captured = {}

    class FakeDeepgramTTSWebSocket:
        def __init__(self):
            self.sent = []
            self.closed = False
            self._messages = asyncio.Queue()

        async def send(self, payload):
            self.sent.append(payload)
            if json.loads(payload).get("type") == "Flush":
                await self._messages.put(b"pcm-audio")

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

    fake_ws = FakeDeepgramTTSWebSocket()

    async def fake_connect(url, additional_headers=None, extra_headers=None):
        captured["url"] = url
        captured["headers"] = additional_headers or extra_headers
        return fake_ws

    monkeypatch.setitem(sys.modules, "websockets", types.SimpleNamespace(connect=fake_connect))

    async def run():
        session = DeepgramStreamingTTSBridgeSession(
            DeepgramStreamingSTTBridgeConfig(
                api_key="deepgram-secret",
                tts_model="aura-2-thalia-en",
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

        assert captured["headers"] == {"Authorization": "Token deepgram-secret"}
        assert captured["url"].startswith("wss://api.deepgram.com/v1/speak?")
        assert json.loads(fake_ws.sent[0]) == {"type": "Speak", "text": "hello from Hermes"}
        assert json.loads(fake_ws.sent[1]) == {"type": "Flush"}
        assert json.loads(fake_ws.sent[2]) == {"type": "Clear"}
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
