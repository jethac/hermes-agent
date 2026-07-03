import asyncio

from agent.realtime_voice import AudioChunk, RealtimeVoiceSessionConfig, VoiceAudioCodec, VoiceEvent, VoiceEventType
from agent.realtime_voice_reference_sidecar import (
    ReferenceRealtimeVoiceSidecarSession,
    ReferenceSidecarRuntimeConfig,
    reference_sidecar_health_payload,
)


class FakeGeminiSession:
    started = []

    def __init__(self, runtime):
        self.runtime = runtime
        self.config = None
        self.received = []
        self.closed = False
        self._events = asyncio.Queue()

    async def start(self, config):
        self.config = config
        self.started.append(self)
        await self._events.put(
            VoiceEvent(
                type=VoiceEventType.FRONTEND_STATE,
                session_id=config.session_id,
                sequence=1,
                payload={"status": "ready", "provider": "gemini_live", "gemini_live": True},
            )
        )

    async def receive_event(self, event):
        self.received.append(event)
        if event.type == VoiceEventType.AUDIO_INPUT_CHUNK:
            await self._events.put(
                VoiceEvent(
                    type=VoiceEventType.TRANSCRIPT_FINAL,
                    session_id=event.session_id,
                    sequence=2,
                    payload={"text": "hello via gemini"},
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


def test_reference_sidecar_health_advertises_gemini_live_when_key_configured():
    payload = reference_sidecar_health_payload(
        ReferenceSidecarRuntimeConfig(
            gemini_live_api_key="gemini-secret",
            gemini_live_model="gemini-3.1-flash-live-preview",
            gemini_live_voice="Puck",
            gemini_live_google_search=True,
        )
    )

    assert payload["capabilities"]["gemini_live"] is True
    assert payload["capabilities"]["streaming_stt"] is True
    assert payload["capabilities"]["tts"] is True
    assert payload["capabilities"]["native_s2s"] is True
    assert payload["capabilities"]["response_cancel"] is True
    assert payload["capabilities"]["tool_calls"] is True
    assert payload["capabilities"]["google_search"] is True
    assert payload["frontend"]["provider"] == "gemini_live"
    assert payload["frontend"]["model"] == "gemini-3.1-flash-live-preview"
    assert payload["frontend"]["gemini_live"] == {
        "configured": True,
        "model": "gemini-3.1-flash-live-preview",
        "voice": "Puck",
        "oracle_tool": True,
        "google_search": True,
    }


def test_reference_sidecar_routes_requested_gemini_live_provider(monkeypatch):
    FakeGeminiSession.started = []
    monkeypatch.setattr(
        "agent.realtime_voice_gemini.GeminiLiveFrontendSession",
        FakeGeminiSession,
    )

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(gemini_live_api_key="gemini-secret")
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-1",
                frontend_provider="gemini_live",
                frontend_model="gemini-3.1-flash-live-preview",
            )
        )
        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if len(seen) == 3:
                break
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.AUDIO_INPUT_CHUNK,
                session_id="voice-1",
                sequence=1,
                payload={
                    **AudioChunk(
                        codec=VoiceAudioCodec.PCM16,
                        data=b"\x00\x00" * 320,
                        sample_rate_hz=16000,
                        channels=1,
                    ).to_payload(),
                    "end_of_utterance": True,
                },
            )
        )
        final = await asyncio.wait_for(sidecar.events().__anext__(), timeout=1)
        await sidecar.close()
        return seen, final, FakeGeminiSession.started[0]

    seen, final, provider = asyncio.run(run())

    assert seen[0].type == VoiceEventType.SESSION_STARTED
    assert seen[1].payload["provider"] == "gemini_live"
    assert seen[2].payload["provider"] == "gemini_live"
    assert provider.config.frontend_model == "gemini-3.1-flash-live-preview"
    assert provider.received[0].type == VoiceEventType.AUDIO_INPUT_CHUNK
    assert final.type == VoiceEventType.TRANSCRIPT_FINAL
    assert final.payload["text"] == "hello via gemini"
    assert provider.closed is True


def test_reference_sidecar_forwards_speakable_oracle_result_to_gemini_live(monkeypatch):
    FakeGeminiSession.started = []
    monkeypatch.setattr(
        "agent.realtime_voice_gemini.GeminiLiveFrontendSession",
        FakeGeminiSession,
    )

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(gemini_live_api_key="gemini-secret")
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-1",
                frontend_provider="gemini_live",
                frontend_model="gemini-3.1-flash-live-preview",
            )
        )
        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if len(seen) == 3:
                break
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.ASSISTANT_TEXT_PARTIAL,
                session_id="voice-1",
                sequence=7,
                payload={
                    "text": "The deployment is healthy.",
                    "speak": True,
                    "oracle_job_result": True,
                    "job_id": "job-1",
                },
            )
        )
        await sidecar.close()
        return FakeGeminiSession.started[0]

    provider = asyncio.run(run())

    forwarded = next(event for event in provider.received if event.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL)
    assert forwarded.type == VoiceEventType.ASSISTANT_TEXT_PARTIAL
    assert forwarded.payload == {
        "text": "The deployment is healthy.",
        "speak": True,
        "oracle_job_result": True,
        "job_id": "job-1",
    }


def test_reference_sidecar_forwards_oracle_result_suppression_to_gemini_live(monkeypatch):
    FakeGeminiSession.started = []
    monkeypatch.setattr(
        "agent.realtime_voice_gemini.GeminiLiveFrontendSession",
        FakeGeminiSession,
    )

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(gemini_live_api_key="gemini-secret")
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-1",
                frontend_provider="gemini_live",
                frontend_model="gemini-3.1-flash-live-preview",
            )
        )
        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if len(seen) == 3:
                break
        await sidecar.receive_event(
            VoiceEvent(
                type=VoiceEventType.ORACLE_JOB_RESULT_SUPPRESSED,
                session_id="voice-1",
                sequence=8,
                payload={
                    "job_id": "job-1",
                    "state": "cancelled",
                    "result_suppressed": True,
                    "suppression_reason": "cancelled_job_returned_result",
                },
            )
        )
        await sidecar.close()
        return FakeGeminiSession.started[0]

    provider = asyncio.run(run())

    forwarded = next(
        event for event in provider.received if event.type == VoiceEventType.ORACLE_JOB_RESULT_SUPPRESSED
    )
    assert forwarded.payload == {
        "job_id": "job-1",
        "state": "cancelled",
        "result_suppressed": True,
        "suppression_reason": "cancelled_job_returned_result",
    }


def test_reference_sidecar_forwards_interpreter_evidence_events_to_gemini_live(monkeypatch):
    FakeGeminiSession.started = []
    monkeypatch.setattr(
        "agent.realtime_voice_gemini.GeminiLiveFrontendSession",
        FakeGeminiSession,
    )

    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(
            ReferenceSidecarRuntimeConfig(gemini_live_api_key="gemini-secret")
        )
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-1",
                frontend_provider="gemini_live",
                frontend_model="gemini-3.1-flash-live-preview",
            )
        )
        seen = []
        async for event in sidecar.events():
            seen.append(event)
            if len(seen) == 3:
                break
        for sequence, event_type, late in (
            (9, VoiceEventType.ORACLE_JOB_INTERPRETER_EVIDENCE_ATTACHED, False),
            (10, VoiceEventType.ORACLE_JOB_INTERPRETER_EVIDENCE_LATE, True),
        ):
            await sidecar.receive_event(
                VoiceEvent(
                    type=event_type,
                    session_id="voice-1",
                    sequence=sequence,
                    payload={
                        "job_id": "job-1",
                        "state": "running" if late else "queued",
                        "latest_interpreter_evidence": "Gemma heard the invoice amount as nineteen dollars.",
                        "latest_interpreter_evidence_source": "gemma_interpreter",
                        "interpreter_evidence_count": sequence - 8,
                        "interpreter_evidence_late": late,
                    },
                )
            )
        await sidecar.close()
        return FakeGeminiSession.started[0]

    provider = asyncio.run(run())

    forwarded = [
        event
        for event in provider.received
        if event.type
        in {
            VoiceEventType.ORACLE_JOB_INTERPRETER_EVIDENCE_ATTACHED,
            VoiceEventType.ORACLE_JOB_INTERPRETER_EVIDENCE_LATE,
        }
    ]
    assert [event.type for event in forwarded] == [
        VoiceEventType.ORACLE_JOB_INTERPRETER_EVIDENCE_ATTACHED,
        VoiceEventType.ORACLE_JOB_INTERPRETER_EVIDENCE_LATE,
    ]
    assert forwarded[0].payload["latest_interpreter_evidence_source"] == "gemma_interpreter"
    assert forwarded[0].payload["interpreter_evidence_late"] is False
    assert forwarded[1].payload["interpreter_evidence_late"] is True


def test_reference_sidecar_degrades_gemini_live_without_key_and_keeps_local_path():
    async def run():
        sidecar = ReferenceRealtimeVoiceSidecarSession(ReferenceSidecarRuntimeConfig())
        await sidecar.start(
            RealtimeVoiceSessionConfig(
                session_id="voice-1",
                frontend_provider="gemini_live",
            )
        )
        events = []
        async for event in sidecar.events():
            events.append(event)
            if len(events) == 3:
                break
        await sidecar.close()
        return events

    events = asyncio.run(run())

    assert events[0].type == VoiceEventType.SESSION_STARTED
    assert events[1].type == VoiceEventType.FRONTEND_STATE
    assert events[1].payload["status"] == "degraded"
    assert events[1].payload["reason"] == "gemini_live_unavailable"
    assert events[2].type == VoiceEventType.FRONTEND_STATE
    assert events[2].payload["status"] == "ready"
    assert events[2].payload["provider"] == "local"
