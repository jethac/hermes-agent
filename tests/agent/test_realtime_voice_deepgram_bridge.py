import asyncio
import json
import sys
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
    deepgram_bridge_config_from_env,
    deepgram_listen_url,
    deepgram_result_to_transcript_payload,
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
    monkeypatch.setenv("HERMES_DEEPGRAM_MODEL", "nova-3")
    monkeypatch.setenv("HERMES_DEEPGRAM_LANGUAGE", "en-US")
    monkeypatch.setenv("HERMES_DEEPGRAM_ENDPOINTING_MS", "120")
    monkeypatch.setenv("HERMES_DEEPGRAM_CONNECT_TIMEOUT_SECONDS", "2.5")

    runtime = deepgram_bridge_config_from_env()

    assert runtime.api_key == "deepgram-secret"
    assert runtime.auth_token == "bridge-token"
    assert runtime.deepgram_url == "wss://deepgram.example.test/v1/listen"
    assert runtime.model == "nova-3"
    assert runtime.language == "en-US"
    assert runtime.endpointing_ms == 120
    assert runtime.connect_timeout_seconds == 2.5


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
