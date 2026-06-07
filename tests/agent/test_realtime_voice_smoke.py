import asyncio

from agent.realtime_voice import RealtimeVoiceSessionConfig, VoiceEvent, VoiceEventType
from agent.realtime_voice_smoke import run_realtime_voice_sidecar_barge_in_smoke


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
    assert result.events == ("frontend.state", "barge_in")
    assert sent[0].type == VoiceEventType.ASSISTANT_TEXT_PARTIAL
    assert sent[0].payload["speak"] is True
    assert sent[1].type == VoiceEventType.BARGE_IN
    assert sent[1].payload["playback_generation"] == 2
