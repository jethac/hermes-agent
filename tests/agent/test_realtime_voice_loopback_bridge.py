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
from agent.realtime_voice_loopback_bridge import (
    LoopbackStreamingBridgeConfig,
    create_loopback_streaming_bridge_app,
)


def test_loopback_bridge_health_advertises_streaming_stt_and_tts():
    client = TestClient(create_loopback_streaming_bridge_app(LoopbackStreamingBridgeConfig()))

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["kind"] == "streaming_loopback_bridge"
    assert body["capabilities"]["streaming_stt"] is True
    assert body["capabilities"]["streaming_tts"] is True
    assert body["capabilities"]["tts"] is True
    assert body["capabilities"]["output_languages"] == ["en", "ja"]


def test_loopback_bridge_streaming_stt_and_tts_sessions():
    client = TestClient(
        create_loopback_streaming_bridge_app(
            LoopbackStreamingBridgeConfig(
                transcript="hello loopback",
                partial_transcript="hello",
            )
        )
    )
    config = RealtimeVoiceSessionConfig(session_id="voice-123")

    with client.websocket_connect("/v1/streaming-stt/session") as ws:
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
                    "input_generation": 3,
                },
            )
        )
        assert frame is not None
        ws.send_bytes(frame)
        partial = ws.receive_json()
        final = ws.receive_json()
        assert partial["type"] == VoiceEventType.TRANSCRIPT_PARTIAL.value
        assert partial["payload"]["text"] == "hello"
        assert partial["payload"]["input_generation"] == 3
        assert final["type"] == VoiceEventType.TRANSCRIPT_FINAL.value
        assert final["payload"]["text"] == "hello loopback"
        assert final["payload"]["input_generation"] == 3

    with client.websocket_connect("/v1/streaming-tts/session") as ws:
        ws.send_json({"type": "session.config", "payload": config.to_wire()})
        ready = ws.receive_json()
        assert ready["type"] == VoiceEventType.FRONTEND_STATE.value
        ws.send_json(
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                session_id="voice-123",
                sequence=1,
                payload={"text": "One moment.", "speak": True, "playback_generation": 4},
            ).to_wire()
        )
        audio = event_from_binary_audio_frame(ws.receive_bytes(), expected_type=VoiceEventType.AUDIO_OUTPUT_CHUNK)
        assert audio.payload["playback_generation"] == 4
        assert audio.payload["sample_rate_hz"] == 16000
        assert audio.payload["channels"] == 1
        assert audio.payload["metrics"]["loopback"] is True
