import asyncio
import base64
import importlib
import struct
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

np = pytest.importorskip("numpy")


class FakeSidecar:
    def __init__(self):
        self.started_with = None
        self.sent = []
        self._events = asyncio.Queue()
        self.closed = False
        self.close_calls = 0

    async def start(self, config):
        self.started_with = config

    async def send_event(self, event):
        self.sent.append(event)

    async def close(self):
        self.close_calls += 1
        self.closed = True
        await self._events.put(None)

    async def events(self):
        while True:
            event = await self._events.get()
            if event is None:
                return
            yield event

    async def emit(self, event):
        await self._events.put(event)


def test_discord_realtime_config_derives_reference_sidecar_and_env_token(monkeypatch):
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.delenv("HERMES_REALTIME_VOICE_SIDECAR_URL", raising=False)
    monkeypatch.delenv("HERMES_REALTIME_VOICE_SIDECAR_TOKEN", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "sidecar_host": "127.0.0.1",
                    "sidecar_port": 8877,
                    "sidecar_token_env": "CUSTOM_VOICE_TOKEN",
                    "frontend_provider": "reference",
                    "routing": {
                        "allow_local_greetings": False,
                        "local_confidence_threshold": 0.9,
                    },
                    "metrics": {
                        "enabled": True,
                        "log_turn_spans": False,
                        "log_provider_spans": True,
                    },
                },
            },
            "discord": {
                "realtime_voice": {
                    "enabled": True,
                },
            },
        },
    )
    monkeypatch.setattr("hermes_cli.config.load_env", lambda: {"CUSTOM_VOICE_TOKEN": "secret-token"})

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    cfg = adapter._load_realtime_voice_config()

    assert cfg["enabled"] is True
    assert cfg["sidecar_base_url"] == "http://127.0.0.1:8877"
    assert cfg["sidecar_token"] == "secret-token"
    assert cfg["frontend_provider"] == "reference"
    assert cfg["routing"]["allow_local_greetings"] is False
    assert cfg["routing"]["local_confidence_threshold"] == 0.9
    assert cfg["metrics"]["log_turn_spans"] is False


def test_discord_realtime_config_accepts_documented_nested_kame_shape(monkeypatch):
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.delenv("HERMES_REALTIME_VOICE_SIDECAR_URL", raising=False)
    monkeypatch.delenv("HERMES_REALTIME_VOICE_SIDECAR_TOKEN", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "engine": "kame_interface_oracle",
                    "sidecar_host": "127.0.0.1",
                    "sidecar_port": 8877,
                    "sidecar_token_env": "CUSTOM_VOICE_TOKEN",
                    "interface": {
                        "provider": "openai_compatible",
                        "base_url": "http://spark.local:8000/v1",
                        "model": "gemma-4-E2B-it",
                        "temperature": 0.3,
                        "max_output_tokens": 96,
                        "timeout_ms": 700,
                        "max_audio_seconds": 16,
                        "audio_input": "auto",
                        "asr_mode": "on_escalation",
                    },
                    "oracle": {
                        "provider": "custom",
                        "provider_name": "Spark Oracle",
                        "preferred_local_model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
                        "model": "configured-oracle",
                        "base_url": "http://spark.local:8001/v1",
                        "api_mode": "chat_completions",
                        "timeout_ms": 12000,
                        "max_spoken_sentences": 2,
                        "voice_response_policy": "brief_summary",
                    },
                    "asr": {
                        "provider": "nemotron",
                        "model": "nemotron-speech",
                        "base_url": "http://spark.local:8767",
                    },
                    "tts": {
                        "provider": "cartesia",
                        "model": "sonic-3.5",
                        "voice": "5ee9feff-1265-424a-9d7f-8e4d431a12c7",
                        "base_url": "http://spark.local:8768",
                    },
                    "barge_in": {
                        "min_rms": 410,
                        "min_speech_ms": 130,
                        "stop_playback_deadline_ms": 95,
                    },
                    "routing": {
                        "allow_local_greetings": True,
                        "allow_local_clarifications": True,
                        "require_oracle_for_tools": True,
                        "require_oracle_for_memory": True,
                        "require_oracle_for_files": True,
                        "local_confidence_threshold": 0.75,
                    },
                    "quality_targets_ms": {
                        "kame_speech_end_to_interface_decision_ms": 321,
                        "kame_speech_end_to_playback_start_ms": 2345,
                    },
                },
            },
            "discord": {
                "realtime_voice": {
                    "enabled": True,
                    "interface": {
                        "provider": "gemma4",
                        "model": "discord-reflex",
                    },
                    "routing": {
                        "allow_local_clarifications": False,
                        "local_confidence_threshold": 0.88,
                    },
                    "asr_base_url": "http://discord.local:8767",
                    "tts_base_url": "http://discord.local:8768",
                },
            },
        },
    )
    monkeypatch.setattr("hermes_cli.config.load_env", lambda: {"CUSTOM_VOICE_TOKEN": "secret-token"})

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    cfg = adapter._load_realtime_voice_config()

    assert cfg["enabled"] is True
    assert cfg["engine"] == "kame_interface_oracle"
    assert cfg["sidecar_base_url"] == "http://127.0.0.1:8877"
    assert cfg["sidecar_token"] == "secret-token"
    assert cfg["frontend_provider"] == "gemma4"
    assert cfg["frontend_model"] == "discord-reflex"
    assert cfg["interface_base_url"] == "http://spark.local:8000/v1"
    assert cfg["vllm_base_url"] == "http://spark.local:8000/v1"
    assert cfg["interface_temperature"] == 0.3
    assert cfg["interface_max_output_tokens"] == 96
    assert cfg["interface_timeout_seconds"] == 0.7
    assert cfg["interface_max_audio_seconds"] == 16
    assert cfg["interface_audio_input"] == "auto"
    assert cfg["asr_mode"] == "on_escalation"
    assert cfg["asr_provider"] == "nemotron"
    assert cfg["asr_model"] == "nemotron-speech"
    assert cfg["asr_base_url"] == "http://discord.local:8767"
    assert cfg["streaming_stt_base_url"] == "http://discord.local:8767"
    assert cfg["oracle_provider"] == "custom"
    assert cfg["oracle_provider_name"] == "Spark Oracle"
    assert cfg["preferred_local_oracle_model"] == "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
    assert "oracle_model" not in cfg
    assert cfg["oracle_base_url"] == "http://spark.local:8001/v1"
    assert cfg["oracle_api_mode"] == "chat_completions"
    assert cfg["oracle_timeout_seconds"] == 12.0
    assert cfg["voice_response_policy"] == "brief_summary"
    assert cfg["tts_provider"] == "cartesia"
    assert cfg["tts_model"] == "sonic-3.5"
    assert cfg["tts_voice"] == "5ee9feff-1265-424a-9d7f-8e4d431a12c7"
    assert cfg["tts_base_url"] == "http://discord.local:8768"
    assert cfg["streaming_tts_base_url"] == "http://discord.local:8768"
    assert cfg["barge_in_min_rms"] == 410
    assert cfg["barge_in_min_speech_ms"] == 130
    assert cfg["barge_in_stop_playback_deadline_ms"] == 95
    assert cfg["routing"] == {
        "allow_local_greetings": True,
        "allow_local_clarifications": False,
        "require_oracle_for_tools": True,
        "require_oracle_for_memory": True,
        "require_oracle_for_files": True,
        "local_confidence_threshold": 0.88,
    }
    assert cfg["quality_targets_ms"]["kame_speech_end_to_interface_decision_ms"] == 321
    assert cfg["quality_targets_ms"]["kame_speech_end_to_playback_start_ms"] == 2345


def test_discord_realtime_native_audio_infers_kame_and_disables_asr(monkeypatch):
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.delenv("HERMES_REALTIME_VOICE_SIDECAR_URL", raising=False)
    monkeypatch.delenv("HERMES_REALTIME_VOICE_SIDECAR_TOKEN", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "engine": "text_oracle_tts",
                    "frontend_provider": "gemma4",
                    "frontend_model": "gemma-4-e2b-reflex",
                    "interface_base_url": "http://pgx.local:8001/v1",
                    "interface_audio_input": "native_audio",
                    "asr_mode": "disabled",
                    "asr_base_url": "http://127.0.0.1:8769",
                    "streaming_stt_base_url": "http://127.0.0.1:8769",
                    "streaming_stt_model": "ink-2",
                    "tts_provider": "piper",
                    "tts_base_url": "http://127.0.0.1:8769",
                    "streaming_tts_base_url": "http://127.0.0.1:8769",
                    "streaming_tts_model": "sonic-3.5",
                },
            },
            "discord": {
                "realtime_voice": {
                    "enabled": True,
                },
            },
        },
    )
    monkeypatch.setattr("hermes_cli.config.load_env", lambda: {})

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    cfg = adapter._load_realtime_voice_config()

    assert cfg["engine"] == "kame_interface_oracle"
    assert cfg["interface_audio_input"] == "native_audio"
    assert cfg["asr_mode"] == "disabled"
    assert cfg["asr_provider"] == ""
    assert cfg["asr_model"] == ""
    assert cfg["asr_base_url"] == ""
    assert cfg["streaming_stt_base_url"] == ""
    assert cfg["tts_provider"] == "piper"
    assert cfg["tts_base_url"] == ""
    assert cfg["streaming_tts_base_url"] == ""
    assert cfg.get("streaming_tts_model", "") == ""


def test_discord_realtime_config_maps_gui_streaming_model_aliases(monkeypatch):
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.delenv("HERMES_REALTIME_VOICE_SIDECAR_URL", raising=False)
    monkeypatch.delenv("HERMES_REALTIME_VOICE_SIDECAR_TOKEN", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "engine": "kame_interface_oracle",
                    "sidecar_base_url": "http://127.0.0.1:8765",
                    "vllm_model": "google/gemma-4-E2B-it",
                    "streaming_stt_base_url": "http://127.0.0.1:8766",
                    "streaming_stt_model": "nemotron-speech-streaming-0.6b",
                    "streaming_tts_base_url": "http://127.0.0.1:8769",
                    "streaming_tts_model": "sonic-3.5",
                    "streaming_tts_voice": "5ee9feff-1265-424a-9d7f-8e4d431a12c7",
                },
            },
            "discord": {
                "realtime_voice": {
                    "enabled": True,
                },
            },
        },
    )
    monkeypatch.setattr("hermes_cli.config.load_env", lambda: {})

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    cfg = adapter._load_realtime_voice_config()

    assert cfg["frontend_model"] == "google/gemma-4-E2B-it"
    assert cfg["asr_base_url"] == "http://127.0.0.1:8766"
    assert cfg["asr_model"] == "nemotron-speech-streaming-0.6b"
    assert cfg["tts_base_url"] == "http://127.0.0.1:8769"
    assert cfg["tts_model"] == "sonic-3.5"
    assert cfg["tts_voice"] == "5ee9feff-1265-424a-9d7f-8e4d431a12c7"


@pytest.mark.asyncio
async def test_discord_realtime_session_streams_downsampled_pcm_to_sidecar():
    from agent.realtime_voice import VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    sidecar = FakeSidecar()
    mixer = MagicMock()
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        mixer=mixer,
        sidecar_base_url="http://127.0.0.1:8766",
        frontend_provider="elevenlabs",
        frontend_model="realtime-voice",
        interface_base_url="http://interface.local:8000/v1",
        interface_temperature=0.35,
        interface_max_output_tokens=96,
        interface_timeout_seconds=0.6,
        interface_max_audio_seconds=14,
        asr_base_url="http://asr.local:8767",
        oracle_timeout_seconds=17.5,
        max_spoken_sentences=3,
        voice_response_policy="brief_summary",
        tts_base_url="http://tts.local:8768",
        barge_in_stop_playback_deadline_ms=95,
        sidecar_connect_timeout_seconds=0.5,
        turn_acknowledgement={"enabled": True, "text": "One moment."},
        routing_policy={
            "allow_local_greetings": True,
            "allow_local_clarifications": False,
            "require_oracle_for_tools": True,
            "require_oracle_for_memory": True,
            "require_oracle_for_files": True,
            "local_confidence_threshold": 0.8,
        },
        metrics_policy={"enabled": True, "log_turn_spans": True, "log_provider_spans": False},
        output_events={"caption_aliases": True, "audio_aliases": True},
        quality_targets_ms={
            "kame_speech_end_to_interface_decision_ms": 321,
            "kame_speech_end_to_playback_start_ms": 2345,
        },
    )

    await session.start()
    # 20 ms of Discord-native 48 kHz stereo s16le silence = 3840 bytes.
    await session.handle_pcm_frame(user_id=42, pcm48_stereo=b"\x00" * 3840)

    assert sidecar.started_with is not None
    assert sidecar.started_with.session_id == "discord:111:222"
    assert sidecar.started_with.input_codec.value == "pcm16"
    assert sidecar.started_with.sample_rate_hz == 16000
    assert sidecar.started_with.channels == 1
    assert sidecar.started_with.sidecar_connect_timeout_seconds == 0.5
    assert sidecar.started_with.frontend_provider == "elevenlabs"
    assert sidecar.started_with.frontend_model == "realtime-voice"
    assert sidecar.started_with.interface_temperature == 0.35
    assert sidecar.started_with.interface_max_output_tokens == 96
    assert sidecar.started_with.interface_timeout_seconds == 0.6
    assert sidecar.started_with.interface_max_audio_seconds == 14
    assert sidecar.started_with.interface_base_url == "http://interface.local:8000/v1"
    assert sidecar.started_with.asr_base_url == "http://asr.local:8767"
    assert "oracle_model" not in sidecar.started_with.to_wire()
    assert sidecar.started_with.oracle_timeout_seconds == 17.5
    assert sidecar.started_with.max_spoken_sentences == 3
    assert sidecar.started_with.voice_response_policy == "brief_summary"
    assert sidecar.started_with.tts_base_url == "http://tts.local:8768"
    assert sidecar.started_with.turn_acknowledgement == {"enabled": True, "text": "One moment."}
    assert sidecar.started_with.routing_policy == {
        "allow_local_greetings": True,
        "allow_local_clarifications": False,
        "require_oracle_for_tools": True,
        "require_oracle_for_memory": True,
        "require_oracle_for_files": True,
        "local_confidence_threshold": 0.8,
    }
    assert sidecar.started_with.metrics_policy == {
        "enabled": True,
        "log_turn_spans": True,
        "log_provider_spans": False,
    }
    assert sidecar.started_with.output_events == {"caption_aliases": True, "audio_aliases": True}
    assert sidecar.started_with.quality_targets_ms == {
        "kame_speech_end_to_interface_decision_ms": 321,
        "kame_speech_end_to_playback_start_ms": 2345,
    }
    assert sidecar.started_with.barge_in_policy == {
        "stop_playback_deadline_ms": 95,
    }
    assert sidecar.started_with.metadata["voice_architecture"] == "kame_frontend_oracle"
    assert sidecar.started_with.metadata["frontend_role"] == "low_latency_voice_interface"
    assert sidecar.started_with.metadata["oracle_role"] == "hermes_backend_oracle"
    assert sidecar.started_with.metadata["frontend_provider"] == "elevenlabs"
    assert sidecar.started_with.metadata["frontend_model"] == "realtime-voice"
    assert sidecar.started_with.metadata["interface_base_url"] == "http://interface.local:8000/v1"
    assert sidecar.started_with.metadata["interface_temperature"] == 0.35
    assert sidecar.started_with.metadata["interface_max_output_tokens"] == 96
    assert sidecar.started_with.metadata["interface_timeout_seconds"] == 0.6
    assert sidecar.started_with.metadata["interface_max_audio_seconds"] == 14
    assert sidecar.started_with.metadata["asr_base_url"] == "http://asr.local:8767"
    assert "oracle_model" not in sidecar.started_with.metadata
    assert sidecar.started_with.metadata["oracle_timeout_seconds"] == 17.5
    assert sidecar.started_with.metadata["max_spoken_sentences"] == 3
    assert sidecar.started_with.metadata["voice_response_policy"] == "brief_summary"
    assert sidecar.started_with.metadata["tts_base_url"] == "http://tts.local:8768"
    assert sidecar.started_with.metadata["turn_acknowledgement"] == {
        "enabled": True,
        "text": "One moment.",
    }
    assert sidecar.started_with.metadata["routing"] == {
        "allow_local_greetings": True,
        "allow_local_clarifications": False,
        "require_oracle_for_tools": True,
        "require_oracle_for_memory": True,
        "require_oracle_for_files": True,
        "local_confidence_threshold": 0.8,
    }
    assert sidecar.started_with.metadata["metrics"] == {
        "enabled": True,
        "log_turn_spans": True,
        "log_provider_spans": False,
    }
    assert sidecar.started_with.metadata["output_events"] == {"caption_aliases": True, "audio_aliases": True}
    assert sidecar.started_with.metadata["quality_targets_ms"] == {
        "kame_speech_end_to_interface_decision_ms": 321,
        "kame_speech_end_to_playback_start_ms": 2345,
    }
    assert sidecar.started_with.metadata["barge_in"] == {
        "stop_playback_deadline_ms": 95,
    }
    assert sidecar.started_with.metadata["barge_in_stop_playback_deadline_ms"] == 95
    assert sidecar.sent[-1].type == VoiceEventType.AUDIO_INPUT_CHUNK
    assert sidecar.sent[-1].payload["sample_rate_hz"] == 16000
    assert sidecar.sent[-1].payload["channels"] == 1
    assert len(base64.b64decode(sidecar.sent[-1].payload["data_b64"])) == 640


@pytest.mark.asyncio
async def test_discord_realtime_session_tags_transcripts_with_last_input_user():
    from agent.realtime_voice import VoiceEvent, VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    observed = []
    sidecar = FakeSidecar()
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        sidecar_base_url="http://127.0.0.1:8766",
        event_callback=lambda event_type, payload: observed.append((event_type, payload)),
    )

    await session.start()
    await session.handle_pcm_frame(user_id=42, pcm48_stereo=b"\x00" * 3840)
    await sidecar.emit(VoiceEvent(
        type=VoiceEventType.TRANSCRIPT_FINAL,
        session_id="discord:111:222",
        sequence=1,
        payload={"text": "this is a test"},
    ))
    await asyncio.wait_for(session.wait_until_idle(), timeout=1)

    assert observed[-1][0] == VoiceEventType.TRANSCRIPT_FINAL.value
    assert observed[-1][1]["user_id"] == "42"


@pytest.mark.asyncio
async def test_discord_realtime_session_sends_end_of_utterance_marker():
    from agent.realtime_voice import VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    sidecar = FakeSidecar()
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        sidecar_base_url="http://127.0.0.1:8766",
    )

    await session.start()
    await session.handle_speech_end(user_id=42)

    assert sidecar.sent[-2].type == VoiceEventType.SPEECH_END
    assert sidecar.sent[-2].payload["user_id"] == "42"
    assert sidecar.sent[-2].payload["transport"] == "discord_voice"
    assert sidecar.sent[-1].type == VoiceEventType.AUDIO_INPUT_CHUNK
    assert sidecar.sent[-1].payload["user_id"] == "42"
    assert sidecar.sent[-1].payload["end_of_utterance"] is True
    assert sidecar.sent[-1].payload["data_b64"] == ""


@pytest.mark.asyncio
async def test_discord_realtime_session_sends_speech_energy_event():
    from agent.realtime_voice import VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    sidecar = FakeSidecar()
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        sidecar_base_url="http://127.0.0.1:8766",
    )

    await session.start()
    await session.handle_speech_energy(user_id=42, rms=512.6, duration_seconds=0.02)

    assert sidecar.sent[-1].type == VoiceEventType.SPEECH_ENERGY
    assert sidecar.sent[-1].payload == {
        "user_id": "42",
        "transport": "discord_voice",
        "rms": 512,
        "duration_ms": 20,
    }


@pytest.mark.asyncio
async def test_discord_realtime_session_routes_output_audio_to_mixer():
    from agent.realtime_voice import AudioChunk, VoiceAudioCodec, VoiceEvent, VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    sidecar = FakeSidecar()
    mixer = MagicMock()
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        mixer=mixer,
        sidecar_base_url="http://127.0.0.1:8766",
    )

    await session.start()
    pcm16_mono_20ms = b"\x00" * 640
    await sidecar.emit(VoiceEvent(
        type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
        session_id="discord:111:222",
        sequence=1,
        payload=AudioChunk(
            codec=VoiceAudioCodec.PCM16,
            data=pcm16_mono_20ms,
            sample_rate_hz=16000,
            channels=1,
        ).to_payload(),
    ))
    await asyncio.wait_for(session.wait_until_idle(), timeout=1)

    mixer.enqueue_speech_frame.assert_called_once()
    frame = mixer.enqueue_speech_frame.call_args.args[0]
    assert len(frame) == 3840


@pytest.mark.asyncio
async def test_discord_realtime_session_routes_assistant_audio_chunk_to_mixer():
    from agent.realtime_voice import AudioChunk, VoiceAudioCodec, VoiceEvent, VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    sidecar = FakeSidecar()
    mixer = MagicMock()
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        mixer=mixer,
        sidecar_base_url="http://127.0.0.1:8766",
    )

    await session.start()
    await sidecar.emit(VoiceEvent(
        type=VoiceEventType.ASSISTANT_AUDIO_CHUNK,
        session_id="discord:111:222",
        sequence=1,
        payload=AudioChunk(
            codec=VoiceAudioCodec.PCM16,
            data=b"\x00" * 640,
            sample_rate_hz=16000,
            channels=1,
        ).to_payload(),
    ))
    await asyncio.wait_for(session.wait_until_idle(), timeout=1)

    mixer.enqueue_speech_frame.assert_called_once()
    frame = mixer.enqueue_speech_frame.call_args.args[0]
    assert len(frame) == 3840


@pytest.mark.asyncio
async def test_discord_realtime_session_does_not_replay_assistant_audio_alias():
    from agent.realtime_voice import AudioChunk, VoiceAudioCodec, VoiceEvent, VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    sidecar = FakeSidecar()
    mixer = MagicMock()
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        mixer=mixer,
        sidecar_base_url="http://127.0.0.1:8766",
    )

    payload = AudioChunk(
        codec=VoiceAudioCodec.PCM16,
        data=b"\x00" * 640,
        sample_rate_hz=16000,
        channels=1,
    ).to_payload()
    payload["playback_generation"] = 4
    alias_payload = dict(payload)
    alias_payload["audio_alias_for"] = VoiceEventType.AUDIO_OUTPUT_CHUNK.value

    await session.start()
    await sidecar.emit(VoiceEvent(
        type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
        session_id="discord:111:222",
        sequence=1,
        payload=payload,
    ))
    await sidecar.emit(VoiceEvent(
        type=VoiceEventType.ASSISTANT_AUDIO_CHUNK,
        session_id="discord:111:222",
        sequence=2,
        payload=alias_payload,
    ))
    await asyncio.wait_for(session.wait_until_idle(), timeout=1)

    mixer.enqueue_speech_frame.assert_called_once()


@pytest.mark.asyncio
async def test_discord_realtime_session_reports_event_metrics_to_callback():
    from agent.realtime_voice import AudioChunk, VoiceAudioCodec, VoiceEvent, VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    observed = []
    sidecar = FakeSidecar()
    mixer = MagicMock()
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        mixer=mixer,
        sidecar_base_url="http://127.0.0.1:8766",
        event_callback=lambda event_type, payload: observed.append((event_type, payload)),
    )

    payload = AudioChunk(
        codec=VoiceAudioCodec.PCM16,
        data=b"\x00" * 640,
        sample_rate_hz=16000,
        channels=1,
    ).to_payload()
    payload["playback_generation"] = 3
    payload["metrics"] = {
        "final_transcript_to_first_audio_ms": 812,
        "kame_speech_end_to_first_audio_ms": 900,
        "kame_oracle_first_token_to_first_tts_audio_ms": 48,
    }

    await session.start()
    await sidecar.emit(VoiceEvent(
        type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
        session_id="discord:111:222",
        sequence=1,
        payload=payload,
    ))
    await asyncio.wait_for(session.wait_until_idle(), timeout=1)

    assert observed[-1][0] == VoiceEventType.AUDIO_OUTPUT_CHUNK.value
    assert observed[-1][1]["metrics"]["final_transcript_to_first_audio_ms"] == 812
    assert observed[-1][1]["metrics"]["kame_speech_end_to_first_audio_ms"] == 900
    assert observed[-1][1]["metrics"]["kame_oracle_first_token_to_first_tts_audio_ms"] == 48
    assert observed[-1][1]["metrics"]["kame_first_tts_audio_to_playback_start_ms"] >= 0
    assert (
        observed[-1][1]["metrics"]["kame_speech_end_to_playback_start_ms"]
        >= observed[-1][1]["metrics"]["kame_speech_end_to_first_audio_ms"]
    )


def test_discord_realtime_event_records_kame_reflex_provenance():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter._voice_session_states = {}

    adapter._handle_realtime_voice_event(
        111,
        "frontend.state",
        {
            "status": "ready",
            "provider": "vllm",
            "frontend_model": "gemma-4-E2B-it",
            "interface_audio_input": "native_audio",
            "vllm_audio_frontend": True,
        },
    )
    adapter._handle_realtime_voice_event(
        111,
        "transcript.final",
        {
            "text": "hello",
            "interface_input_source": "native_audio",
            "reflex_provider": "vllm",
        },
    )
    adapter._handle_realtime_voice_event(
        111,
        "interface.intent.final",
        {
            "route": "oracle_direct",
            "reflex_validation_error": "invalid_json",
        },
    )

    status = adapter.get_voice_session_status(111)

    assert status["frontend_state"] == {
        "status": "ready",
        "provider": "vllm",
        "frontend_model": "gemma-4-E2B-it",
        "interface_audio_input": "native_audio",
        "vllm_audio_frontend": True,
        "interface_input_source": "native_audio",
        "reflex_provider": "vllm",
        "route": "oracle_direct",
        "reflex_validation_error": "invalid_json",
    }


def test_discord_realtime_event_records_tts_failure_provenance():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter._voice_session_states = {}

    adapter._handle_realtime_voice_event(
        111,
        "session.error",
        {
            "reason": "tts_unavailable",
            "streaming_tts": False,
            "local_tts": False,
            "tts_provider": "cartesia",
            "tts_model": "sonic-3.5",
            "tts_voice": "5ee9feff-1265-424a-9d7f-8e4d431a12c7",
        },
    )

    status = adapter.get_voice_session_status(111)

    assert status["frontend_state"] == {
        "reason": "tts_unavailable",
        "streaming_tts": False,
        "local_tts": False,
        "tts_provider": "cartesia",
        "tts_model": "sonic-3.5",
        "tts_voice": "5ee9feff-1265-424a-9d7f-8e4d431a12c7",
    }


def test_discord_realtime_event_counts_native_assistant_audio_chunks():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter._voice_session_states = {}

    adapter._handle_realtime_voice_event(
        111,
        "assistant.audio.chunk",
        {
            "metrics": {
                "kame_speech_end_to_first_audio_ms": 820,
            },
        },
    )

    status = adapter.get_voice_session_status(111)

    assert status["last_realtime_event"] == "assistant.audio.chunk"
    assert status["latency_metrics_ms"]["audio_output_chunks"] == 1
    assert status["latency_metrics_ms"]["kame_speech_end_to_first_audio_ms"] == 820


def test_discord_realtime_event_does_not_double_count_assistant_audio_alias():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter._voice_session_states = {}

    adapter._handle_realtime_voice_event(111, "audio.output.chunk", {})
    adapter._handle_realtime_voice_event(
        111,
        "assistant.audio.chunk",
        {
            "audio_alias_for": "audio.output.chunk",
            "metrics": {
                "kame_speech_end_to_first_audio_ms": 820,
            },
        },
    )

    status = adapter.get_voice_session_status(111)

    assert status["last_realtime_event"] == "assistant.audio.chunk"
    assert status["latency_metrics_ms"]["audio_output_chunks"] == 1
    assert status["latency_metrics_ms"]["kame_speech_end_to_first_audio_ms"] == 820


@pytest.mark.asyncio
async def test_discord_realtime_session_barge_in_stops_mixer_and_notifies_sidecar():
    from agent.realtime_voice import VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    sidecar = FakeSidecar()
    mixer = SimpleNamespace(speech_active=True, stop_speech=MagicMock())
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        mixer=mixer,
        sidecar_base_url="http://127.0.0.1:8766",
    )

    await session.start()
    await session.handle_speech_start(user_id=42)

    mixer.stop_speech.assert_called_once()
    assert sidecar.sent[-2].type == VoiceEventType.SPEECH_START
    assert sidecar.sent[-2].payload["user_id"] == "42"
    assert sidecar.sent[-2].payload["transport"] == "discord_voice"
    assert sidecar.sent[-1].type == VoiceEventType.BARGE_IN
    assert sidecar.sent[-1].payload["user_id"] == "42"
    assert sidecar.sent[-1].payload["playback_active"] is True
    assert sidecar.sent[-1].payload["playback_stop_attempted"] is True
    assert sidecar.sent[-1].payload["playback_stop_deadline_ms"] == 150
    assert sidecar.sent[-1].payload["barge_in_confirmed_to_playback_stopped_ms"] >= 0
    assert (
        sidecar.sent[-1].payload["metrics"]["barge_in_confirmed_to_playback_stopped_ms"]
        == sidecar.sent[-1].payload["barge_in_confirmed_to_playback_stopped_ms"]
    )


@pytest.mark.asyncio
async def test_discord_realtime_session_barge_in_bounds_slow_mixer_stop():
    from agent.realtime_voice import VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    stop_started = asyncio.Event()

    async def slow_stop():
        stop_started.set()
        await asyncio.sleep(1)

    sidecar = FakeSidecar()
    mixer = SimpleNamespace(speech_active=True, stop_speech=slow_stop)
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        mixer=mixer,
        sidecar_base_url="http://127.0.0.1:8766",
        barge_in_stop_playback_deadline_ms=25,
    )

    await session.start()
    started = asyncio.get_running_loop().time()
    await session.handle_speech_start(user_id=42)
    elapsed_ms = (asyncio.get_running_loop().time() - started) * 1000

    assert stop_started.is_set()
    assert elapsed_ms < 200
    assert sidecar.sent[-1].type == VoiceEventType.BARGE_IN
    assert sidecar.sent[-1].payload["playback_stop_attempted"] is True
    assert sidecar.sent[-1].payload["playback_stop_timed_out"] is True
    assert sidecar.sent[-1].payload["playback_stop_deadline_ms"] == 25
    assert sidecar.sent[-1].payload["barge_in_confirmed_to_playback_stopped_ms"] >= 25
    assert (
        sidecar.sent[-1].payload["metrics"]["barge_in_confirmed_to_playback_stopped_ms"]
        == sidecar.sent[-1].payload["barge_in_confirmed_to_playback_stopped_ms"]
    )


@pytest.mark.asyncio
async def test_discord_realtime_session_barge_in_notifies_sidecar_without_active_mixer():
    from agent.realtime_voice import VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    sidecar = FakeSidecar()
    mixer = SimpleNamespace(speech_active=False, stop_speech=MagicMock())
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        mixer=mixer,
        sidecar_base_url="http://127.0.0.1:8766",
    )

    await session.start()
    await session.handle_speech_start(user_id=42)

    mixer.stop_speech.assert_not_called()
    assert sidecar.sent[-1].type == VoiceEventType.BARGE_IN
    assert sidecar.sent[-1].payload["playback_active"] is False


@pytest.mark.asyncio
async def test_discord_realtime_session_drops_stale_audio_after_barge_in():
    from agent.realtime_voice import AudioChunk, VoiceAudioCodec, VoiceEvent, VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    sidecar = FakeSidecar()
    mixer = SimpleNamespace(
        speech_active=True,
        stop_speech=MagicMock(),
        enqueue_speech_frame=MagicMock(),
    )
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        mixer=mixer,
        sidecar_base_url="http://127.0.0.1:8766",
    )

    stale_audio = AudioChunk(
        codec=VoiceAudioCodec.PCM16,
        data=b"\x00" * 640,
        sample_rate_hz=16000,
        channels=1,
    ).to_payload()
    stale_audio["playback_generation"] = 0
    fresh_audio = dict(stale_audio)
    fresh_audio["playback_generation"] = 1

    await session.start()
    await session.handle_speech_start(user_id=42)
    await sidecar.emit(VoiceEvent(
        type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
        session_id="discord:111:222",
        sequence=1,
        payload=stale_audio,
    ))
    await asyncio.wait_for(session.wait_until_idle(), timeout=1)
    mixer.enqueue_speech_frame.assert_not_called()

    await sidecar.emit(VoiceEvent(
        type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
        session_id="discord:111:222",
        sequence=2,
        payload=fresh_audio,
    ))
    await asyncio.wait_for(session.wait_until_idle(), timeout=1)

    mixer.stop_speech.assert_called_once()
    mixer.enqueue_speech_frame.assert_called_once()


def test_voice_mixer_can_enqueue_single_streaming_frame():
    from plugins.platforms.discord import voice_mixer as vm

    mixer = vm.VoiceMixer()
    frame = (np.ones(vm.SAMPLES_PER_FRAME * vm.CHANNELS) * 12000).astype(np.int16).tobytes()
    mixer.enqueue_speech_frame(frame, fade_in_ms=0)

    out = mixer.read()

    assert len(out) == vm.FRAME_SIZE
    assert int(np.max(np.abs(np.frombuffer(out, dtype=np.int16)))) == 12000
    assert mixer.speech_active is True
    mixer.finish_speech_stream()
    assert mixer.read() == vm.SILENCE_FRAME
    assert mixer.speech_active is False


def test_voice_mixer_is_discord_audio_source_when_discord_available(monkeypatch):
    from plugins.platforms.discord import voice_mixer as vm

    original_discord = sys.modules.get("discord")
    audio_source_cls = type("AudioSource", (), {})
    monkeypatch.setitem(sys.modules, "discord", SimpleNamespace(AudioSource=audio_source_cls))
    try:
        reloaded = importlib.reload(vm)
        assert isinstance(reloaded.VoiceMixer(), audio_source_cls)
    finally:
        if original_discord is None:
            sys.modules.pop("discord", None)
        else:
            sys.modules["discord"] = original_discord
        importlib.reload(vm)


def test_discord_pcm_downsample_averages_full_resample_window():
    from plugins.platforms.discord.realtime_voice import discord_pcm48_stereo_to_pcm16_mono

    pcm = struct.pack(
        "<hhhhhh",
        300,
        600,
        300,
        600,
        300,
        600,
    )

    out = discord_pcm48_stereo_to_pcm16_mono(pcm)

    assert struct.unpack("<h", out)[0] == 450


@pytest.mark.asyncio
async def test_discord_realtime_session_buffers_partial_output_until_commit():
    from agent.realtime_voice import AudioChunk, VoiceAudioCodec, VoiceEvent, VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    sidecar = FakeSidecar()
    mixer = MagicMock()
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        mixer=mixer,
        sidecar_base_url="http://127.0.0.1:8766",
    )

    await session.start()
    await sidecar.emit(VoiceEvent(
        type=VoiceEventType.AUDIO_OUTPUT_CHUNK,
        session_id="discord:111:222",
        sequence=1,
        payload=AudioChunk(
            codec=VoiceAudioCodec.PCM16,
            data=b"\x00" * 320,
            sample_rate_hz=16000,
            channels=1,
        ).to_payload(),
    ))
    await asyncio.wait_for(session.wait_until_idle(), timeout=1)
    mixer.enqueue_speech_frame.assert_not_called()

    await sidecar.emit(VoiceEvent(
        type=VoiceEventType.ASSISTANT_COMMIT,
        session_id="discord:111:222",
        sequence=2,
        payload={},
    ))
    await asyncio.wait_for(session.wait_until_idle(), timeout=1)

    mixer.enqueue_speech_frame.assert_called_once()
    assert len(mixer.enqueue_speech_frame.call_args.args[0]) == 3840
    mixer.finish_speech_stream.assert_called_once()


@pytest.mark.asyncio
async def test_discord_realtime_session_sends_session_closed_before_sidecar_close():
    from agent.realtime_voice import VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    sidecar = FakeSidecar()
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        sidecar_base_url="http://127.0.0.1:8766",
    )

    await session.start()
    await session.close()

    assert sidecar.closed is True
    assert sidecar.sent[-1].type == VoiceEventType.SESSION_CLOSED


@pytest.mark.asyncio
async def test_discord_realtime_session_close_stops_active_mixer_speech():
    from agent.realtime_voice import VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    sidecar = FakeSidecar()
    mixer = SimpleNamespace(
        speech_active=True,
        stop_speech=MagicMock(),
        finish_speech_stream=MagicMock(),
    )
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        mixer=mixer,
        sidecar_base_url="http://127.0.0.1:8766",
    )

    await session.start()
    await session.close()

    mixer.stop_speech.assert_called_once()
    mixer.finish_speech_stream.assert_called_once()
    assert sidecar.closed is True
    assert sidecar.sent[-1].type == VoiceEventType.SESSION_CLOSED


@pytest.mark.asyncio
async def test_discord_realtime_session_close_is_idempotent():
    from agent.realtime_voice import VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    sidecar = FakeSidecar()
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        sidecar_base_url="http://127.0.0.1:8766",
    )

    await session.start()
    await session.close()
    await session.close()

    assert sidecar.close_calls == 1
    assert [event.type for event in sidecar.sent].count(VoiceEventType.SESSION_CLOSED) == 1


@pytest.mark.asyncio
async def test_discord_realtime_session_reports_sidecar_session_error_as_degraded():
    from agent.realtime_voice import VoiceEvent, VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    degraded = []
    sidecar = FakeSidecar()
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        sidecar_base_url="http://127.0.0.1:8766",
        degraded_callback=lambda reason, error: degraded.append((reason, error)),
    )

    await session.start()
    await sidecar.emit(VoiceEvent(
        type=VoiceEventType.SESSION_ERROR,
        session_id="discord:111:222",
        sequence=1,
        payload={"error": "sidecar unavailable"},
    ))
    await asyncio.wait_for(session.wait_until_idle(), timeout=1)

    assert degraded == [("sidecar_session_error", "sidecar unavailable")]
