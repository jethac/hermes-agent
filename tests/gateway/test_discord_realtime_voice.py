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


class ScriptedKameSidecar:
    def __init__(self, *, oracle, utterances):
        self.oracle = oracle
        self.utterances = list(utterances)
        self.started_with = None
        self.sent = []
        self.session = None
        self.closed = False

    async def start(self, config):
        from agent.realtime_voice_session import RealtimeVoiceSession
        from agent.realtime_voice_text_engine import KameInterfaceOracleEngine

        self.started_with = config
        self.session = RealtimeVoiceSession(
            config,
            engine=KameInterfaceOracleEngine(oracle=self.oracle),
        )
        await self.session.start()

    async def send_event(self, event):
        self.sent.append(event)
        payload = dict(event.payload)
        if event.type.value == "audio.input.chunk" and payload.get("end_of_utterance") is True and self.utterances:
            payload.update(self.utterances.pop(0))
            event = type(event)(
                type=event.type,
                session_id=event.session_id,
                sequence=event.sequence,
                timestamp_ms=event.timestamp_ms,
                payload=payload,
            )
        await self.session.receive_client_event(event)

    async def events(self):
        async for event in self.session.events():
            yield event

    async def get_oracle_job_status(self):
        if self.session is None:
            return {}
        return await self.session.get_oracle_job_status()

    async def close(self):
        self.closed = True
        if self.session is not None:
            await self.session.close()


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
                    "oracle_jobs": {
                        "enabled": True,
                        "max_concurrent": 4,
                        "queue_limit": 12,
                        "default_priority": "high",
                        "overflow_policy": "reject",
                        "shutdown_timeout_seconds": 3.5,
                        "speak_terminal_results": False,
                        "audit_ledger_path": "artifacts/voiceops/oracle-jobs.jsonl",
                    },
                    "oracle_tool_router": {
                        "enabled": True,
                        "mode": "deterministic",
                        "voiceops_toolsets": ["voiceops"],
                        "default_toolsets": [],
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
    assert cfg["oracle_jobs"] == {
        "enabled": True,
        "max_concurrent": 4,
        "queue_limit": 12,
        "default_priority": "high",
        "overflow_policy": "reject",
        "shutdown_timeout_seconds": 3.5,
        "speak_terminal_results": False,
        "audit_ledger_path": "artifacts/voiceops/oracle-jobs.jsonl",
    }
    assert cfg["oracle_tool_router"] == {
        "enabled": True,
        "mode": "deterministic",
        "voiceops_toolsets": ["voiceops"],
        "default_toolsets": [],
    }


def test_discord_realtime_config_overrides_shared_oracle_jobs(monkeypatch):
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.delenv("HERMES_REALTIME_VOICE_SIDECAR_URL", raising=False)
    monkeypatch.delenv("HERMES_REALTIME_VOICE_SIDECAR_TOKEN", raising=False)
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "sidecar_base_url": "http://127.0.0.1:8877",
                    "oracle_jobs": {
                        "enabled": True,
                        "max_concurrent": 4,
                        "queue_limit": 16,
                        "default_priority": "normal",
                        "overflow_policy": "queue",
                        "shutdown_timeout_seconds": 2.0,
                        "speak_terminal_results": True,
                        "audit_ledger_path": "artifacts/shared.jsonl",
                    },
                    "oracle_tool_router": {
                        "enabled": True,
                        "mode": "deterministic",
                        "voiceops_toolsets": ["voiceops"],
                        "default_toolsets": [],
                    },
                },
            },
            "discord": {
                "realtime_voice": {
                    "oracle_jobs": {
                        "max_concurrent": 2,
                        "shutdown_timeout_seconds": 0.75,
                        "audit_ledger_path": "artifacts/discord.jsonl",
                    },
                    "oracle_tool_router": {
                        "voiceops_toolsets": ["voiceops", "discord"],
                    },
                },
            },
        },
    )
    monkeypatch.setattr("hermes_cli.config.load_env", lambda: {})

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    cfg = adapter._load_realtime_voice_config()

    assert cfg["oracle_jobs"] == {
        "enabled": True,
        "max_concurrent": 2,
        "queue_limit": 16,
        "default_priority": "normal",
        "overflow_policy": "queue",
        "shutdown_timeout_seconds": 0.75,
        "speak_terminal_results": True,
        "audit_ledger_path": "artifacts/discord.jsonl",
    }
    assert cfg["oracle_tool_router"] == {
        "enabled": True,
        "mode": "deterministic",
        "voiceops_toolsets": ["voiceops", "discord"],
        "default_toolsets": [],
    }


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
                    "input_noise_gate": {
                        "enabled": True,
                        "min_rms": 275,
                        "start_ms": 80,
                        "hangover_ms": 240,
                        "preroll_ms": 100,
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
    assert cfg["input_noise_gate_enabled"] is True
    assert cfg["input_noise_gate_min_rms"] == 275
    assert cfg["input_noise_gate_start_ms"] == 80
    assert cfg["input_noise_gate_hangover_ms"] == 240
    assert cfg["input_noise_gate_preroll_ms"] == 100
    assert cfg["routing"] == {
        "allow_local_greetings": True,
        "allow_local_clarifications": False,
        "require_oracle_for_tools": True,
        "require_oracle_for_memory": True,
        "require_oracle_for_files": True,
        "local_confidence_threshold": 0.88,
    }
    assert cfg["oracle_jobs"] == {
        "enabled": True,
        "max_concurrent": 1,
        "queue_limit": 16,
        "default_priority": "normal",
        "overflow_policy": "queue",
        "shutdown_timeout_seconds": 2.0,
        "speak_terminal_results": True,
        "audit_ledger_path": "",
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
        oracle_jobs={"enabled": True, "max_concurrent": 4, "queue_limit": 16},
        oracle_tool_router={
            "enabled": True,
            "mode": "deterministic",
            "voiceops_toolsets": ["voiceops"],
            "default_toolsets": [],
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
    assert sidecar.started_with.oracle_jobs == {"enabled": True, "max_concurrent": 4, "queue_limit": 16}
    assert sidecar.started_with.oracle_tool_router == {
        "enabled": True,
        "mode": "deterministic",
        "voiceops_toolsets": ["voiceops"],
        "default_toolsets": [],
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
    assert sidecar.started_with.metadata["oracle_jobs"] == {
        "enabled": True,
        "max_concurrent": 4,
        "queue_limit": 16,
    }
    assert sidecar.started_with.metadata["barge_in"] == {
        "stop_playback_deadline_ms": 95,
    }
    assert sidecar.started_with.metadata["barge_in_stop_playback_deadline_ms"] == 95
    assert sidecar.sent[-1].type == VoiceEventType.AUDIO_INPUT_CHUNK
    assert sidecar.sent[-1].payload["sample_rate_hz"] == 16000
    assert sidecar.sent[-1].payload["channels"] == 1
    assert len(base64.b64decode(sidecar.sent[-1].payload["data_b64"])) == 640


def test_discord_voice_receiver_noise_gate_drops_low_energy_pcm_before_sidecar():
    from plugins.platforms.discord.adapter import VoiceReceiver

    ended = []
    dropped = []
    receiver = VoiceReceiver(
        SimpleNamespace(),
        realtime_speech_end_callback=lambda user_id: ended.append(user_id),
        realtime_noise_gate_min_rms=100,
        realtime_noise_gate_start_ms=40,
        realtime_noise_gate_hangover_ms=40,
        realtime_noise_gate_preroll_ms=40,
        realtime_noise_gate_drop_callback=lambda user_id, rms, duration: dropped.append((user_id, rms, duration)),
    )

    quiet = b"\x00" * 3840
    voiced_1 = b"\x01" * 3840
    voiced_2 = b"\x02" * 3840
    assert receiver._realtime_gate_frames(1, 42, quiet, 0, 0.02) == []
    assert receiver._realtime_gate_frames(1, 42, voiced_1, 200, 0.02) == []

    opened = receiver._realtime_gate_frames(1, 42, voiced_2, 200, 0.02)
    assert opened == [voiced_1, voiced_2]
    assert dropped == [(42, 0, 0.02)]

    assert receiver._realtime_gate_frames(1, 42, quiet, 0, 0.02) == [quiet]
    assert receiver._realtime_gate_frames(1, 42, quiet, 0, 0.02) == [quiet]
    assert receiver._realtime_gate_frames(1, 42, quiet, 0, 0.02) == []
    assert ended == [42]


def test_discord_voice_receiver_noise_gate_zero_start_ignores_quiet_preroll():
    from plugins.platforms.discord.adapter import VoiceReceiver

    dropped = []
    receiver = VoiceReceiver(
        SimpleNamespace(),
        realtime_noise_gate_min_rms=100,
        realtime_noise_gate_start_ms=0,
        realtime_noise_gate_hangover_ms=40,
        realtime_noise_gate_preroll_ms=80,
        realtime_noise_gate_drop_callback=lambda user_id, rms, duration: dropped.append((user_id, rms, duration)),
    )

    quiet = b"\x00" * 3840
    voiced = b"\x02" * 3840

    assert receiver._realtime_gate_frames(1, 42, quiet, 0, 0.02) == []
    opened = receiver._realtime_gate_frames(1, 42, voiced, 200, 0.02)

    assert opened == [quiet, voiced]
    assert dropped == [(42, 0, 0.02)]


def test_discord_voice_receiver_noise_gate_can_be_disabled():
    from plugins.platforms.discord.adapter import VoiceReceiver

    receiver = VoiceReceiver(
        SimpleNamespace(),
        realtime_noise_gate_enabled=False,
        realtime_noise_gate_min_rms=1000,
    )

    quiet = b"\x00" * 3840
    assert receiver._realtime_gate_frames(1, 42, quiet, 0, 0.02) == [quiet]


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
async def test_discord_realtime_session_tags_interface_intents_with_last_input_user():
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
        type=VoiceEventType.INTERFACE_INTENT_FINAL,
        session_id="discord:111:222",
        sequence=1,
        payload={"text": "hey hermes", "route": "oracle_direct"},
    ))
    await asyncio.wait_for(session.wait_until_idle(), timeout=1)

    assert observed[-1][0] == VoiceEventType.INTERFACE_INTENT_FINAL.value
    assert observed[-1][1]["user_id"] == "42"


@pytest.mark.asyncio
async def test_discord_realtime_session_keeps_user_tag_for_intent_and_transcript_pair():
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
        event_callback=lambda event_type, payload: observed.append((event_type, dict(payload))),
    )

    await session.start()
    await session.handle_pcm_frame(user_id=42, pcm48_stereo=b"\x00" * 3840)
    await sidecar.emit(VoiceEvent(
        type=VoiceEventType.INTERFACE_INTENT_FINAL,
        session_id="discord:111:222",
        sequence=1,
        payload={"text": "hey hermes", "route": "oracle_direct"},
    ))
    await sidecar.emit(VoiceEvent(
        type=VoiceEventType.TRANSCRIPT_FINAL,
        session_id="discord:111:222",
        sequence=2,
        payload={"text": "hey hermes"},
    ))
    await asyncio.wait_for(session.wait_until_idle(), timeout=1)

    finals = [
        payload for event_type, payload in observed
        if event_type in {
            VoiceEventType.INTERFACE_INTENT_FINAL.value,
            VoiceEventType.TRANSCRIPT_FINAL.value,
        }
    ]
    assert [payload["user_id"] for payload in finals] == ["42", "42"]


@pytest.mark.asyncio
async def test_discord_realtime_session_resets_kame_user_tag_before_next_speaker():
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
        event_callback=lambda event_type, payload: observed.append((event_type, dict(payload))),
    )

    await session.start()
    await session.handle_pcm_frame(user_id=42, pcm48_stereo=b"\x00" * 3840)
    await sidecar.emit(VoiceEvent(
        type=VoiceEventType.INTERFACE_INTENT_FINAL,
        session_id="discord:111:222",
        sequence=1,
        payload={"text": "check the first task", "route": "oracle_direct"},
    ))
    await asyncio.wait_for(session.wait_until_idle(), timeout=1)

    await session.handle_pcm_frame(user_id=43, pcm48_stereo=b"\x00" * 3840)
    await sidecar.emit(VoiceEvent(
        type=VoiceEventType.INTERFACE_INTENT_FINAL,
        session_id="discord:111:222",
        sequence=2,
        payload={"text": "check the second task", "route": "oracle_direct"},
    ))
    await asyncio.wait_for(session.wait_until_idle(), timeout=1)

    finals = [
        payload for event_type, payload in observed
        if event_type == VoiceEventType.INTERFACE_INTENT_FINAL.value
    ]
    assert [payload["user_id"] for payload in finals] == ["42", "43"]


@pytest.mark.asyncio
async def test_discord_realtime_session_does_not_guess_user_for_mixed_speakers():
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
        event_callback=lambda event_type, payload: observed.append((event_type, dict(payload))),
    )

    await session.start()
    await session.handle_pcm_frame(user_id=42, pcm48_stereo=b"\x00" * 3840)
    await session.handle_pcm_frame(user_id=43, pcm48_stereo=b"\x00" * 3840)
    await sidecar.emit(VoiceEvent(
        type=VoiceEventType.TRANSCRIPT_FINAL,
        session_id="discord:111:222",
        sequence=1,
        payload={"text": "mixed speech"},
    ))
    await asyncio.wait_for(session.wait_until_idle(), timeout=1)

    assert observed[-1][0] == VoiceEventType.TRANSCRIPT_FINAL.value
    assert "user_id" not in observed[-1][1]


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


@pytest.mark.asyncio
async def test_discord_adapter_live_voice_status_overrides_stale_oracle_snapshot():
    from plugins.platforms.discord.adapter import DiscordAdapter

    class LiveSession:
        async def get_oracle_job_status(self):
            return {
                "enabled": True,
                "capacity": {
                    "running": 1,
                    "max_concurrent": 4,
                    "queued": 0,
                    "waiting_for_approval": 0,
                    "cancel_requested": 0,
                    "queue_limit": 16,
                },
                "jobs": [
                    {
                        "job_id": "voice-oracle-live",
                        "state": "running",
                        "spoken_status": "Checking the live status.",
                        "metadata": {"secret": "do-not-leak"},
                        "oracle_text": "raw oracle prompt",
                    }
                ],
            }

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter._voice_session_states = {}
    adapter._voice_mixers = {}
    adapter._voice_receivers = {}
    adapter._voice_text_channels = {}
    adapter._realtime_voice_sessions = {111: LiveSession()}

    adapter._handle_realtime_voice_event(
        111,
        "session.started",
        {"oracle_jobs": {"enabled": True, "max_concurrent": 4, "queue_limit": 16}},
    )
    adapter._handle_realtime_voice_event(
        111,
        "oracle.job.completed",
        {
            "job_id": "voice-oracle-stale",
            "state": "completed",
            "spoken_status": "Old status.",
        },
    )

    status = await adapter.get_voice_session_status_live(111)
    job = status["oracle_jobs"]["jobs"][0]

    assert status["oracle_jobs"]["capacity"]["running"] == 1
    assert job == {
        "job_id": "voice-oracle-live",
        "state": "running",
        "spoken_status": "Checking the live status.",
    }
    assert adapter.get_voice_session_status(111)["oracle_jobs"]["jobs"][0]["job_id"] == "voice-oracle-live"


def test_discord_realtime_event_tracks_redacted_kame_stack_status():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter._voice_session_states = {}

    adapter._handle_realtime_voice_event(
        111,
        "session.started",
        {
            "kame_stack": {
                "engine": "kame_interface_oracle",
                "reflex": {
                    "provider": "moshi",
                    "model": "moshi-reflex",
                    "audio_input": "native_audio",
                    "base_url_configured": True,
                    "base_url": "http://reflex.local/secret",
                },
                "interpreter": {
                    "provider": "gemma4",
                    "model": "gemma-4-e2b-it",
                    "audio_input": "native_audio",
                    "base_url_configured": True,
                    "base_url": "http://gemma.local/secret",
                    "role": "raw_audio_evidence_adjudicator",
                },
                "transcript_evidence": {
                    "mode": "speculative",
                    "provider": "nemotron-speech",
                    "model": "fastconformer",
                    "base_url_configured": True,
                    "authority": "hypothesis",
                    "schedule_oracle_from_transcript": False,
                },
                "oracle": {
                    "mode": "hermes_active_model",
                    "preferred_local_model": "nemotron-3-super",
                    "timeout_seconds": 60.0,
                },
                "tts": {
                    "provider": "magpie",
                    "model": "magpie-preview",
                    "voice_configured": True,
                    "base_url_configured": True,
                },
                "fallback_policy": "fail_closed",
            }
        },
    )

    stack = adapter.get_voice_session_status(111)["kame_stack"]

    assert stack["reflex"] == {
        "provider": "moshi",
        "model": "moshi-reflex",
        "audio_input": "native_audio",
        "base_url_configured": True,
    }
    assert stack["interpreter"] == {
        "provider": "gemma4",
        "model": "gemma-4-e2b-it",
        "audio_input": "native_audio",
        "base_url_configured": True,
        "role": "raw_audio_evidence_adjudicator",
    }
    assert stack["oracle"]["mode"] == "hermes_active_model"
    assert "reflex.local" not in repr(stack)
    assert "gemma.local" not in repr(stack)


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


def test_discord_realtime_event_tracks_oracle_job_status():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter._voice_session_states = {}

    adapter._handle_realtime_voice_event(
        111,
        "session.started",
        {"oracle_jobs": {"enabled": True, "max_concurrent": 4, "queue_limit": 16}},
    )
    adapter._handle_realtime_voice_event(
        111,
        "oracle.job.started",
        {
            "job_id": "voice-oracle-001",
            "state": "running",
            "priority": "normal",
            "route": "defer",
            "intent": "Check the deployment status.",
            "spoken_status": "Checking the deployment status.",
            "metadata": {"tool_schema": "must not leak"},
            "oracle_text": "must not leak",
        },
    )
    adapter._handle_realtime_voice_event(
        111,
        "oracle.job.completed",
        {
            "job_id": "voice-oracle-001",
            "state": "completed",
            "result_summary": "The deployment is healthy.",
        },
    )
    adapter._handle_realtime_voice_event(
        111,
        "interface.oracle.update",
        {
            "job_id": "voice-oracle-002",
            "state": "queued",
            "priority": "high",
            "update_count": 1,
            "latest_update": "also check the Stripe receipt before answering",
            "metadata": {"tool_schema": "must not leak"},
            "oracle_text": "must not leak",
        },
    )
    adapter._handle_realtime_voice_event(
        111,
        "oracle.job.waiting_for_approval",
        {
            "job_id": "voice-oracle-002",
            "state": "waiting_for_approval",
            "priority": "normal",
            "route": "oracle_direct",
            "intent": "Prepare Stripe spend.",
            "spoken_status": "Preparing the Stripe spend request.",
        },
    )
    adapter._handle_realtime_voice_event(
        111,
        "oracle.job.interpreter_evidence_late",
        {
            "job_id": "voice-oracle-002",
            "state": "waiting_for_approval",
            "latest_interpreter_evidence": "interpreter evidence: transcript=buy phone credits",
            "latest_interpreter_evidence_source": "gemma_interpreter",
            "interpreter_evidence_count": 1,
            "interpreter_evidence_late": True,
            "interpreter_evidence_delivered_to_oracle": True,
            "interpreter_evidence_consumed_before_irreversible_action": False,
            "interpreter_evidence_delivery_status": "delivered",
            "evidence_authority": {
                "raw_audio": "primary_audio",
                "reflex_transcript_hypothesis": "hypothesis",
                "auxiliary_transcript_hypotheses": "hypothesis",
                "interpreter_corrected_transcript": "interpreter_promoted",
                "unsafe_extra": "http://do-not-copy.local",
            },
            "latest_interpreter_evidence_authority": {
                "raw_audio": "primary_audio",
                "interpreter_disagreements": "diagnostic_only",
                "unsafe_extra": "http://do-not-copy.local",
            },
            "corrected_transcript": "must not leak",
        },
    )

    status = adapter.get_voice_session_status(111)

    assert status["oracle_jobs"]["capacity"] == {
        "active": 1,
        "running": 0,
        "queued": 0,
        "waiting_for_approval": 1,
        "cancel_requested": 0,
        "max_concurrent": 4,
        "queue_limit": 16,
    }
    assert status["oracle_jobs"]["jobs"] == [
        {
            "job_id": "voice-oracle-001",
            "state": "completed",
            "priority": "normal",
            "route": "defer",
            "intent": "Check the deployment status.",
            "spoken_status": "Checking the deployment status.",
            "result_summary": "The deployment is healthy.",
        },
        {
            "job_id": "voice-oracle-002",
            "state": "waiting_for_approval",
            "priority": "normal",
            "route": "oracle_direct",
            "intent": "Prepare Stripe spend.",
            "spoken_status": "Preparing the Stripe spend request.",
            "update_count": 1,
            "latest_update": "also check the Stripe receipt before answering",
            "latest_interpreter_evidence": "interpreter evidence: transcript=buy phone credits",
            "latest_interpreter_evidence_source": "gemma_interpreter",
            "interpreter_evidence_count": 1,
            "interpreter_evidence_late": True,
            "interpreter_evidence_delivered_to_oracle": True,
            "interpreter_evidence_consumed_before_irreversible_action": False,
            "interpreter_evidence_delivery_status": "delivered",
            "evidence_authority": {
                "raw_audio": "primary_audio",
                "reflex_transcript_hypothesis": "hypothesis",
                "auxiliary_transcript_hypotheses": "hypothesis",
                "interpreter_corrected_transcript": "interpreter_promoted",
            },
            "latest_interpreter_evidence_authority": {
                "raw_audio": "primary_audio",
                "interpreter_disagreements": "diagnostic_only",
            },
        },
    ]
    assert "metadata" not in status["oracle_jobs"]["jobs"][0]
    assert "oracle_text" not in status["oracle_jobs"]["jobs"][0]
    assert "corrected_transcript" not in status["oracle_jobs"]["jobs"][1]


def test_discord_realtime_degraded_marks_active_oracle_jobs_failed():
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter._voice_session_states = {}
    adapter._realtime_voice_sessions = {111: object()}
    adapter._realtime_voice_cfg = {"fallback_policy": "text_only"}

    adapter._handle_realtime_voice_event(
        111,
        "session.started",
        {"oracle_jobs": {"enabled": True, "max_concurrent": 4, "queue_limit": 16}},
    )
    adapter._handle_realtime_voice_event(
        111,
        "oracle.job.started",
        {
            "job_id": "voice-oracle-001",
            "state": "running",
            "priority": "normal",
            "route": "defer",
            "intent": "Check the deployment status.",
            "spoken_status": "Checking the deployment status.",
        },
    )

    adapter._handle_realtime_voice_degraded(
        111,
        "sidecar_event_stream_closed",
        "sidecar event stream closed",
    )
    status = adapter.get_voice_session_status(111)

    assert 111 not in adapter._realtime_voice_sessions
    assert status["sidecar_running"] is False
    assert status["fallback_reason"] == "sidecar_event_stream_closed: sidecar event stream closed"
    assert status["oracle_jobs"]["capacity"]["running"] == 0
    assert status["oracle_jobs"]["jobs"] == [
        {
            "job_id": "voice-oracle-001",
            "state": "failed",
            "priority": "normal",
            "route": "defer",
            "intent": "Check the deployment status.",
            "spoken_status": "Checking the deployment status.",
            "error": "sidecar_event_stream_closed: sidecar event stream closed",
        }
    ]


def test_voice_status_oracle_job_lines_are_compact():
    from gateway.slash_commands import _voice_status_oracle_job_lines

    lines = _voice_status_oracle_job_lines(
        {
            "enabled": True,
            "capacity": {
                "active": 3,
                "running": 1,
                "max_concurrent": 4,
                "queued": 2,
                "waiting_for_approval": 1,
                "cancel_requested": 1,
            },
            "jobs": [
                {
                    "job_id": "voice-oracle-001",
                    "state": "running",
                    "spoken_status": "Checking the deployment status.",
                    "latest_update": "include the staging region too.",
                },
                {
                    "job_id": "voice-oracle-002",
                    "state": "waiting_for_approval",
                    "spoken_status": "Waiting for Stripe spend approval.",
                    "interpreter_evidence_count": 1,
                    "interpreter_evidence_late": True,
                    "interpreter_evidence_delivery_status": "delivered",
                    "evidence_authority": {
                        "raw_audio": "primary_audio",
                        "reflex_transcript_hypothesis": "hypothesis",
                        "auxiliary_transcript_hypotheses": "hypothesis",
                        "interpreter_corrected_transcript": "interpreter_promoted",
                    },
                },
                {
                    "job_id": "voice-oracle-003",
                    "state": "cancel_requested",
                    "spoken_status": "Stopping the stale deployment check.",
                    "latest_interpreter_evidence_authority": {
                        "raw_audio": "primary_audio",
                        "interpreter_disagreements": "diagnostic_only",
                    },
                },
            ],
        }
    )

    assert lines == [
        "Oracle jobs: active=3/4, running=1, queued=2, waiting_for_approval=1, cancel_requested=1",
        "Oracle job: voice-oracle-001 running - Checking the deployment status. | update: include the staging region too.",
        "Oracle job: voice-oracle-002 waiting_for_approval - Waiting for Stripe spend approval. | evidence: delivered, late, x1, authority=audio/interpreter/hypothesis",
        "Oracle job: voice-oracle-003 cancel_requested - Stopping the stale deployment check. | evidence: latest=audio/diagnostic",
    ]


def test_voice_status_oracle_job_lines_prefer_reflex_safe_ordinals():
    from gateway.slash_commands import _voice_status_oracle_job_lines

    lines = _voice_status_oracle_job_lines(
        {
            "enabled": True,
            "capacity": {
                "active": 4,
                "running": 4,
                "max_concurrent": 4,
                "queued": 1,
            },
            "jobs": [
                {
                    "job_id": "voice-oracle-hidden",
                    "state": "running",
                    "spoken_status": "Raw job that should not drive references.",
                    "metadata": {"hidden": "raw evidence"},
                },
            ],
            "reflex": {
                "capacity": {
                    "active": 4,
                    "running": 4,
                    "max_concurrent": 4,
                    "queued": 1,
                },
                "jobs": [
                    {
                        "job_id": f"voice-oracle-{index:03d}",
                        "state": "queued" if index == 5 else "running",
                        "ordinal": index,
                        "ordinal_label": f"job {name}",
                        "spoken_status": f"Visible job {index}.",
                    }
                    for index, name in enumerate(("one", "two", "three", "four", "five"), start=1)
                ],
            },
        }
    )

    assert lines == [
        "Oracle jobs: running=4/4, queued=1",
        "Oracle job: job one (voice-oracle-001) running - Visible job 1.",
        "Oracle job: job two (voice-oracle-002) running - Visible job 2.",
        "Oracle job: job three (voice-oracle-003) running - Visible job 3.",
        "Oracle job: job four (voice-oracle-004) running - Visible job 4.",
        "Oracle job: job five (voice-oracle-005) queued - Visible job 5.",
    ]
    assert "voice-oracle-hidden" not in "\n".join(lines)
    assert "raw evidence" not in "\n".join(lines)


def test_voice_status_oracle_job_lines_prefer_terminal_outcomes():
    from gateway.slash_commands import _voice_status_oracle_job_lines

    lines = _voice_status_oracle_job_lines(
        {
            "enabled": True,
            "capacity": {
                "running": 0,
                "max_concurrent": 4,
                "queued": 0,
            },
            "jobs": [
                {
                    "job_id": "voice-oracle-completed",
                    "state": "completed",
                    "spoken_status": "Checking the deployment status.",
                    "result_summary": "The deployment is healthy.",
                },
                {
                    "job_id": "voice-oracle-failed",
                    "state": "failed",
                    "spoken_status": "Checking Stripe.",
                    "error": "Stripe preflight failed.",
                },
                {
                    "job_id": "voice-oracle-cancelled",
                    "state": "cancelled",
                    "spoken_status": "Checking logs.",
                    "cancel_reason": "User cancelled the log check.",
                },
            ],
        }
    )

    assert lines == [
        "Oracle jobs: running=0/4, queued=0",
        "Oracle job: voice-oracle-completed completed - The deployment is healthy.",
        "Oracle job: voice-oracle-failed failed - Stripe preflight failed.",
        "Oracle job: voice-oracle-cancelled cancelled - User cancelled the log check.",
    ]


@pytest.mark.asyncio
async def test_discord_realtime_session_sends_oracle_job_cancel_event():
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
    await session.cancel_oracle_job("voice-oracle-001", reason="user requested /voice cancel")
    await session.close()

    cancel = next(event for event in sidecar.sent if event.type == VoiceEventType.INTERFACE_ORACLE_CANCEL)
    assert cancel.payload == {
        "job_id": "voice-oracle-001",
        "all": False,
        "reason": "user requested /voice cancel",
        "transport": "discord_voice",
    }


@pytest.mark.asyncio
async def test_discord_realtime_session_sends_oracle_job_cancel_all_event():
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
    await session.cancel_oracle_job("all", reason="user requested /voice cancel")
    await session.close()

    cancel = next(event for event in sidecar.sent if event.type == VoiceEventType.INTERFACE_ORACLE_CANCEL)
    assert cancel.payload == {
        "job_id": "all",
        "all": True,
        "reason": "user requested /voice cancel",
        "transport": "discord_voice",
    }


@pytest.mark.asyncio
async def test_discord_realtime_session_sends_oracle_job_update_event():
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
    await session.update_oracle_job(
        "voice-oracle-002",
        priority="high",
        update_text="also check the Stripe receipt",
        reason="user requested /voice update",
    )
    await session.close()

    update = next(event for event in sidecar.sent if event.type == VoiceEventType.INTERFACE_ORACLE_UPDATE)
    assert update.payload == {
        "job_id": "voice-oracle-002",
        "reason": "user requested /voice update",
        "transport": "discord_voice",
        "priority": "high",
        "update_text": "also check the Stripe receipt",
    }


@pytest.mark.asyncio
async def test_discord_realtime_spoken_tasks_create_async_oracle_jobs():
    from agent.realtime_voice import VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    class BlockingOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            await self.release.wait()
            yield f"Finished {request.intent}."

    async def wait_for(predicate):
        deadline = asyncio.get_running_loop().time() + 1.0
        while not predicate():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("timed out waiting for Discord async oracle event")
            await asyncio.sleep(0.01)

    oracle = BlockingOracle()
    observed = []
    sidecar = ScriptedKameSidecar(
        oracle=oracle,
        utterances=[
            {
                "transcript": "check provisioning logs",
                "intent": "Check provisioning logs",
                "intent_source": "reflex_audio",
                "route": "defer",
                "interface_already_said": "Checking provisioning logs.",
            },
            {
                "transcript": "draft the vendor memo",
                "intent": "Draft vendor memo",
                "intent_source": "reflex_audio",
                "route": "defer",
                "interface_already_said": "Drafting the vendor memo.",
            },
        ],
    )
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        sidecar_base_url="http://127.0.0.1:8766",
        engine="kame_interface_oracle",
        frontend_provider="gemma4",
        frontend_model="gemma-4-E2B-it",
        interface_audio_input="native_audio",
        oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
        event_callback=lambda event_type, payload: observed.append((event_type, payload)),
    )

    await session.start()
    await session.handle_speech_end(user_id=42)
    await wait_for(
        lambda: any(
            event_type == VoiceEventType.ORACLE_JOB_STARTED.value
            and payload.get("intent") == "Check provisioning logs"
            for event_type, payload in observed
        )
    )
    await session.handle_speech_end(user_id=42)
    await wait_for(
        lambda: any(
            event_type == VoiceEventType.ORACLE_JOB_QUEUED.value
            and payload.get("intent") == "Draft vendor memo"
            for event_type, payload in observed
        )
    )

    await session.close()

    accepted = [
        payload
        for event_type, payload in observed
        if event_type == VoiceEventType.ORACLE_JOB_ACCEPTED.value
    ]
    assert [payload["intent"] for payload in accepted] == [
        "Check provisioning logs",
        "Draft vendor memo",
    ]
    assert [request.intent for request in oracle.requests] == ["Check provisioning logs"]
    assert any(
        event_type == VoiceEventType.ORACLE_JOB_STARTED.value
        and payload.get("job_id") == "voice-oracle-001"
        for event_type, payload in observed
    )
    assert any(
        event_type == VoiceEventType.ORACLE_JOB_QUEUED.value
        and payload.get("job_id") == "voice-oracle-002"
        for event_type, payload in observed
    )
    assert any(
        event_type == VoiceEventType.INTERFACE_REPLY_DEFER.value
        and payload.get("text") == "Checking provisioning logs."
        for event_type, payload in observed
    )
    assert any(
        event_type == VoiceEventType.INTERFACE_REPLY_DEFER.value
        and payload.get("text") == "Drafting the vendor memo."
        for event_type, payload in observed
    )


@pytest.mark.asyncio
async def test_discord_realtime_live_oracle_status_reflects_job_manager():
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    class BlockingOracle:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_answer_for_request(self, request):
            self.started.set()
            await self.release.wait()
            yield f"Finished {request.intent}."

    oracle = BlockingOracle()
    sidecar = ScriptedKameSidecar(
        oracle=oracle,
        utterances=[
            {
                "transcript": "check provisioning logs",
                "intent": "Check provisioning logs",
                "intent_source": "reflex_audio",
                "route": "defer",
                "interface_already_said": "Checking provisioning logs.",
            },
        ],
    )
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        sidecar_base_url="http://127.0.0.1:8766",
        engine="kame_interface_oracle",
        oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
    )

    await session.start()
    await session.handle_speech_end(user_id=42)
    await asyncio.wait_for(oracle.started.wait(), timeout=1)

    status = await session.get_oracle_job_status()

    await session.close()

    assert status["enabled"] is True
    assert status["capacity"]["running"] == 1
    assert status["capacity"]["max_concurrent"] == 1
    assert status["capacity"]["queued"] == 0
    assert status["jobs"][0]["job_id"] == "voice-oracle-001"
    assert status["jobs"][0]["state"] == "running"
    assert status["jobs"][0]["intent"] == "Check provisioning logs"
    assert "metadata" not in status["jobs"][0]
    assert "oracle_text" not in status["jobs"][0]


@pytest.mark.asyncio
async def test_discord_realtime_cancelled_oracle_late_output_is_not_mixed(monkeypatch):
    from agent.realtime_voice import AudioChunk, VoiceAudioCodec, VoiceEventType
    from agent.realtime_voice_text_engine import KameInterfaceOracleEngine
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    class LateOutputOracle:
        def __init__(self):
            self.requests = []
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.late_output_attempted = False

        async def stream_answer_for_request(self, request):
            self.requests.append(request)
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await self.release.wait()
                self.late_output_attempted = True
            yield f"Finished {request.intent}."

    async def fake_speak(self, text, playback_generation):
        if "Finished" not in text:
            return
        payload = AudioChunk(
            codec=VoiceAudioCodec.PCM16,
            data=b"\x01\x00" * 320,
            sample_rate_hz=16000,
            channels=1,
        ).to_payload()
        payload["playback_generation"] = playback_generation
        await self._emit(VoiceEventType.AUDIO_OUTPUT_CHUNK, payload)

    async def wait_for(predicate):
        deadline = asyncio.get_running_loop().time() + 1.0
        while not predicate():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("timed out waiting for Discord cancelled oracle event")
            await asyncio.sleep(0.01)

    monkeypatch.setattr(KameInterfaceOracleEngine, "_speak_chunk", fake_speak)

    oracle = LateOutputOracle()
    observed = []
    mixer = SimpleNamespace(
        speech_active=False,
        stop_speech=MagicMock(),
        enqueue_speech_frame=MagicMock(),
        finish_speech_stream=MagicMock(),
    )
    sidecar = ScriptedKameSidecar(
        oracle=oracle,
        utterances=[
            {
                "transcript": "check provisioning logs",
                "intent": "Check provisioning logs",
                "intent_source": "reflex_audio",
                "route": "defer",
                "interface_already_said": "Checking provisioning logs.",
            },
        ],
    )
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        mixer=mixer,
        sidecar_base_url="http://127.0.0.1:8766",
        engine="kame_interface_oracle",
        frontend_provider="gemma4",
        frontend_model="gemma-4-E2B-it",
        interface_audio_input="native_audio",
        oracle_jobs={"enabled": True, "max_concurrent": 1, "queue_limit": 4},
        event_callback=lambda event_type, payload: observed.append((event_type, payload)),
    )

    await session.start()
    await session.handle_speech_end(user_id=42)
    await wait_for(
        lambda: any(
            event_type == VoiceEventType.ORACLE_JOB_STARTED.value
            and payload.get("job_id") == "voice-oracle-001"
            for event_type, payload in observed
        )
    )
    await asyncio.wait_for(oracle.started.wait(), timeout=1)

    await session.cancel_oracle_job("voice-oracle-001", reason="test cancellation")
    await wait_for(
        lambda: any(
            event_type == VoiceEventType.ORACLE_JOB_CANCEL_REQUESTED.value
            and payload.get("job_id") == "voice-oracle-001"
            for event_type, payload in observed
        )
    )
    oracle.release.set()
    await wait_for(
        lambda: any(
            event_type == VoiceEventType.ORACLE_JOB_CANCELLED.value
            and payload.get("job_id") == "voice-oracle-001"
            for event_type, payload in observed
        )
    )
    await wait_for(lambda: oracle.late_output_attempted)
    await session.wait_until_idle()
    await session.close()

    mixer.enqueue_speech_frame.assert_not_called()
    assert not any(
        event_type == VoiceEventType.ORACLE_JOB_COMPLETED.value
        and payload.get("job_id") == "voice-oracle-001"
        for event_type, payload in observed
    )
    suppressed = next(
        payload
        for event_type, payload in observed
        if event_type == VoiceEventType.ORACLE_JOB_RESULT_SUPPRESSED.value
        and payload.get("job_id") == "voice-oracle-001"
    )
    assert suppressed["suppression_reason"] == "cancelled_runner_interrupted"
    assert suppressed["result_suppressed"] is True
    assert suppressed["suppressed_result_present"] is False
    assert "result_summary" not in suppressed
    assert "result_text" not in suppressed
    assert not any(
        event_type == VoiceEventType.ASSISTANT_COMMIT.value
        and payload.get("oracle_job_result")
        and payload.get("oracle_job_id") == "voice-oracle-001"
        for event_type, payload in observed
    )


@pytest.mark.asyncio
async def test_discord_adapter_cancel_voice_oracle_job_delegates_to_session():
    from plugins.platforms.discord.adapter import DiscordAdapter

    class Session:
        def __init__(self):
            self.cancelled = []

        async def cancel_oracle_job(self, job_id, *, reason):
            self.cancelled.append((job_id, reason))

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    session = Session()
    adapter._realtime_voice_sessions = {111: session}

    result = await adapter.cancel_voice_oracle_job(
        111,
        "voice-oracle-001",
        reason="user requested /voice cancel",
    )
    missing = await adapter.cancel_voice_oracle_job(222, "voice-oracle-002")

    assert result == {
        "ok": True,
        "job_id": "voice-oracle-001",
        "reason": "user requested /voice cancel",
    }
    assert session.cancelled == [("voice-oracle-001", "user requested /voice cancel")]
    assert missing == {"ok": False, "reason": "no_active_realtime_voice_session"}


@pytest.mark.asyncio
async def test_discord_adapter_cancel_voice_oracle_job_marks_degraded_on_send_failure():
    from plugins.platforms.discord.adapter import DiscordAdapter

    class Session:
        async def cancel_oracle_job(self, job_id, *, reason):
            raise RuntimeError("websocket closed")

        async def close(self):
            pass

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter._realtime_voice_sessions = {111: Session()}
    adapter._voice_session_states = {}
    adapter._realtime_voice_cfg = {"fallback_policy": "text_only"}

    result = await adapter.cancel_voice_oracle_job(
        111,
        "voice-oracle-001",
        reason="user requested /voice cancel",
    )
    await asyncio.sleep(0)

    assert result == {
        "ok": False,
        "reason": "control_send_failed",
        "error": "websocket closed",
    }
    assert 111 not in adapter._realtime_voice_sessions
    status = adapter.get_voice_session_status(111)
    assert status["mode"] == "text_only_fallback"
    assert status["session_state"] == "degraded"
    assert status["fallback_reason"] == "control_send_failed: websocket closed"


@pytest.mark.asyncio
async def test_discord_adapter_update_voice_oracle_job_delegates_to_session():
    from plugins.platforms.discord.adapter import DiscordAdapter

    class Session:
        def __init__(self):
            self.updated = []

        async def update_oracle_job(self, job_id, *, priority, update_text, reason):
            self.updated.append((job_id, priority, update_text, reason))

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    session = Session()
    adapter._realtime_voice_sessions = {111: session}

    result = await adapter.update_voice_oracle_job(
        111,
        "voice-oracle-002",
        priority="high",
        update_text="also check the Stripe receipt",
        reason="user requested /voice update",
    )
    missing = await adapter.update_voice_oracle_job(222, "voice-oracle-002", priority="high")

    assert result == {
        "ok": True,
        "job_id": "voice-oracle-002",
        "priority": "high",
        "update_text": "also check the Stripe receipt",
        "reason": "user requested /voice update",
    }
    assert session.updated == [
        ("voice-oracle-002", "high", "also check the Stripe receipt", "user requested /voice update")
    ]
    assert missing == {"ok": False, "reason": "no_active_realtime_voice_session"}


@pytest.mark.asyncio
async def test_discord_adapter_update_voice_oracle_job_marks_degraded_on_send_failure():
    from plugins.platforms.discord.adapter import DiscordAdapter

    class Session:
        async def update_oracle_job(self, job_id, *, priority, update_text, reason):
            raise RuntimeError("websocket closed")

        async def close(self):
            pass

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter._realtime_voice_sessions = {111: Session()}
    adapter._voice_session_states = {}
    adapter._realtime_voice_cfg = {"fallback_policy": "text_only"}

    result = await adapter.update_voice_oracle_job(
        111,
        "voice-oracle-002",
        priority="high",
        reason="user requested /voice priority",
    )
    await asyncio.sleep(0)

    assert result == {
        "ok": False,
        "reason": "control_send_failed",
        "error": "websocket closed",
    }
    assert 111 not in adapter._realtime_voice_sessions
    status = adapter.get_voice_session_status(111)
    assert status["mode"] == "text_only_fallback"
    assert status["session_state"] == "degraded"
    assert status["fallback_reason"] == "control_send_failed: websocket closed"


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
async def test_discord_realtime_session_close_cancels_oracle_jobs_before_session_closed():
    from agent.realtime_voice import VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    sidecar = FakeSidecar()
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        sidecar_base_url="http://127.0.0.1:8766",
        oracle_jobs={"enabled": True, "shutdown_timeout_seconds": 0},
    )

    await session.start()
    await session.close()

    event_types = [event.type for event in sidecar.sent]
    assert event_types[-2:] == [
        VoiceEventType.INTERFACE_ORACLE_CANCEL,
        VoiceEventType.SESSION_CLOSED,
    ]
    assert sidecar.sent[-2].payload == {
        "job_id": "all",
        "all": True,
        "reason": "voice session closing",
        "transport": "discord_voice",
    }
    assert sidecar.closed is True


@pytest.mark.asyncio
async def test_discord_realtime_session_close_waits_for_oracle_cancel_ack_before_session_closed():
    from agent.realtime_voice import VoiceEvent, VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    class AckingSidecar(FakeSidecar):
        def __init__(self):
            super().__init__()
            self.timeline = []

        async def send_event(self, event):
            self.sent.append(event)
            self.timeline.append(("sent", event.type.value))
            if event.type == VoiceEventType.INTERFACE_ORACLE_CANCEL:
                await self._events.put(
                    VoiceEvent(
                        type=VoiceEventType.ORACLE_JOB_CANCEL_REQUESTED,
                        session_id=event.session_id,
                        sequence=100,
                        payload={
                            "job_id": "voice-oracle-001",
                            "state": "cancel_requested",
                            "cancel_reason": event.payload.get("reason"),
                        },
                    )
                )

    sidecar = AckingSidecar()
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        sidecar_base_url="http://127.0.0.1:8766",
        oracle_jobs={"enabled": True, "shutdown_timeout_seconds": 0.25},
        event_callback=lambda event_type, payload: sidecar.timeline.append(("observed", event_type)),
    )

    await session.start()
    await session.close()

    assert sidecar.closed is True
    assert sidecar.timeline.index(("observed", VoiceEventType.ORACLE_JOB_CANCEL_REQUESTED.value)) < sidecar.timeline.index(
        ("sent", VoiceEventType.SESSION_CLOSED.value)
    )


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


@pytest.mark.asyncio
async def test_discord_realtime_session_reports_unexpected_session_closed_as_degraded():
    from agent.realtime_voice import VoiceEvent, VoiceEventType
    from plugins.platforms.discord.realtime_voice import DiscordRealtimeVoiceSession

    degraded = []
    observed = []
    sidecar = FakeSidecar()
    session = DiscordRealtimeVoiceSession(
        guild_id=111,
        voice_channel_id=222,
        text_channel_id=333,
        sidecar=sidecar,
        sidecar_base_url="http://127.0.0.1:8766",
        degraded_callback=lambda reason, error: degraded.append((reason, error)),
        event_callback=lambda event_type, payload: observed.append((event_type, payload)),
    )

    await session.start()
    await sidecar.emit(VoiceEvent(
        type=VoiceEventType.SESSION_CLOSED,
        session_id="discord:111:222",
        sequence=1,
        payload={"reason": "provider closed"},
    ))
    await asyncio.wait_for(session.wait_until_idle(), timeout=1)

    assert observed == [("session.closed", {"reason": "provider closed"})]
    assert degraded == [("sidecar_session_closed", "sidecar closed the session")]


@pytest.mark.asyncio
async def test_discord_realtime_session_reports_unexpected_sidecar_eof_as_degraded():
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
    await sidecar._events.put(None)
    await asyncio.wait_for(session.wait_until_idle(), timeout=1)

    assert degraded == [("sidecar_event_stream_closed", "sidecar event stream closed")]


@pytest.mark.asyncio
async def test_discord_realtime_session_local_close_does_not_report_sidecar_eof_degraded():
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
    await session.close()
    await asyncio.sleep(0)

    assert degraded == []
