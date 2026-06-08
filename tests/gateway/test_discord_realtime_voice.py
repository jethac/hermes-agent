import asyncio
import base64
import struct
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

    async def start(self, config):
        self.started_with = config

    async def send_event(self, event):
        self.sent.append(event)

    async def close(self):
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
        sidecar_connect_timeout_seconds=0.5,
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
    assert sidecar.sent[-1].type == VoiceEventType.AUDIO_INPUT_CHUNK
    assert sidecar.sent[-1].payload["sample_rate_hz"] == 16000
    assert sidecar.sent[-1].payload["channels"] == 1
    assert len(base64.b64decode(sidecar.sent[-1].payload["data_b64"])) == 640


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
    assert sidecar.sent[-1].type == VoiceEventType.BARGE_IN
    assert sidecar.sent[-1].payload["user_id"] == "42"


def test_voice_mixer_can_enqueue_single_streaming_frame():
    from plugins.platforms.discord import voice_mixer as vm

    mixer = vm.VoiceMixer()
    frame = (np.ones(vm.SAMPLES_PER_FRAME * vm.CHANNELS) * 12000).astype(np.int16).tobytes()
    mixer.enqueue_speech_frame(frame, fade_in_ms=0)

    out = mixer.read()

    assert len(out) == vm.FRAME_SIZE
    assert int(np.max(np.abs(np.frombuffer(out, dtype=np.int16)))) == 12000
    assert mixer.speech_active is False


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
