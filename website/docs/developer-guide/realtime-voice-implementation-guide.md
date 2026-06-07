---
title: "Realtime Voice Implementation Guide"
description: "Implementation guide for the KAME-inspired realtime Hermes voice subsystem"
---

# Realtime Voice Implementation Guide

This guide is the working plan for adding realtime voice to Hermes. It assumes the PRD in [Realtime Voice PRD](./realtime-voice-prd.md).

## Current State

Desktop voice conversation is currently turn-based:

- `apps/desktop/src/app/chat/composer/hooks/use-mic-recorder.ts` records audio into a blob.
- `apps/desktop/src/app/chat/composer/hooks/use-voice-conversation.ts` waits for silence, uploads the blob, submits the transcript, then speaks chunks of streamed text.
- `hermes_cli/web_server.py` exposes `/api/audio/transcribe` and `/api/audio/speak`.
- `agent/transcription_provider.py` and `agent/tts_provider.py` define one-shot STT/TTS provider interfaces, with optional TTS byte streaming.

The new subsystem should not remove any of that. It adds a parallel realtime path.

## First Code Boundary

`agent/realtime_voice.py` defines the stable protocol primitives:

- `RealtimeVoiceSessionConfig`
- `RealtimeVoiceEngineKind`
- `VoiceAudioCodec`
- `VoiceEventType`
- `VoiceEvent`
- `AudioChunk`
- `RealtimeVoiceEngine`

Treat this module as the shared contract between the desktop app, FastAPI websocket endpoint, Hermes session runtime, and optional model sidecars.

## Implemented Surfaces

The realtime voice implementation now has both engine families behind the same protocol:

- `agent/realtime_voice_session.py` owns the session state machine, monotonically increasing client sequence validation, barge-in state, and the durable transcript boundary.
- `agent/realtime_voice_text_engine.py` implements the text-oracle path: audio or transcript input, STT via the existing Hermes transcription provider chain at utterance boundaries, streaming Hermes oracle deltas, speech planning, and chunked audio output via the existing TTS provider chain.
- `agent/realtime_voice_s2s_engine.py` implements the native S2S path as a websocket bridge to a DGX Spark or other model sidecar. When the sidecar emits final transcript events, Hermes calls the configured oracle model and sends `oracle.hint` events back to the sidecar.
- `hermes_cli/web_server.py` exposes `/api/voice/realtime` behind the same websocket auth and Host/Origin guards as the dashboard chat websocket.
- `apps/desktop/src/app/chat/composer/hooks/use-realtime-voice-session.ts` implements the desktop websocket client, microphone frame capture, simple VAD, playback queue, and barge-in cancellation.

The existing one-shot voice mode remains the fallback. Realtime voice is opt-in via `voice.realtime.enabled`.

Current limits:

- In-core STT still uses Hermes' existing file-based transcription providers after an utterance boundary. Fully streaming STT should be added as a provider/sidecar capability behind the same event protocol.
- The native S2S model itself is not shipped in Hermes. Hermes provides the sidecar bridge and oracle hint stream; the DGX/Spark service owns model inference.
- Audio frames are JSON/base64 for the first implementation. Binary websocket frames can replace this without changing the semantic event contract.

## Target File Layout

```text
agent/
  realtime_voice.py              # shared protocol and engine ABC
  realtime_voice_session.py      # session state machine, sequence, persistence boundary
  realtime_voice_oracle.py       # Hermes oracle adapter around AIAgent/context/tools
  realtime_voice_planner.py      # early-speech, commit, interruption policy
  realtime_voice_text_engine.py  # STT -> oracle -> TTS implementation
  realtime_voice_s2s_engine.py   # native S2S sidecar bridge + oracle stream protocol

hermes_cli/
  web_server.py                  # /api/voice/realtime websocket endpoint

apps/desktop/src/app/chat/voice/
  realtime-voice-client.ts       # websocket client
  audio-input-worklet.ts         # frame capture
  use-realtime-voice-session.ts  # websocket client, frame capture, playback, barge-in
```

## Wire Protocol

Use websocket JSON frames for control and metadata. Audio chunks can start as base64 JSON payloads for simplicity, then move to binary frames if profiling proves it necessary.

Client events:

```json
{"type":"audio.input.chunk","session_id":"...","sequence":1,"payload":{"codec":"opus","data_b64":"..."}}
{"type":"barge_in","session_id":"...","sequence":2,"payload":{"reason":"user_speech"}}
{"type":"session.closed","session_id":"...","sequence":3,"payload":{"reason":"client_closed"}}
```

Server events:

```json
{"type":"session.started","session_id":"...","sequence":1,"payload":{"engine":"text_oracle_tts"}}
{"type":"transcript.partial","session_id":"...","sequence":2,"payload":{"text":"what was that","stability":0.42}}
{"type":"transcript.final","session_id":"...","sequence":3,"payload":{"text":"what was that KAME paper about?"}}
{"type":"assistant.text.partial","session_id":"...","sequence":4,"payload":{"text":"KAME is interesting because "}}
{"type":"audio.output.chunk","session_id":"...","sequence":5,"payload":{"codec":"opus","data_b64":"..."}}
{"type":"assistant.commit","session_id":"...","sequence":6,"payload":{"text":"KAME is interesting because ..."}}
```

## Backend Endpoint

Add a FastAPI websocket endpoint in `hermes_cli/web_server.py`:

```python
@app.websocket("/api/voice/realtime")
async def realtime_voice_ws(ws: WebSocket) -> None:
    ...
```

Implementation notes:

- Reuse existing websocket host/origin/auth guards in `web_server.py`.
- Create one `RealtimeVoiceSession` per websocket.
- Validate every client event with `validate_client_event`.
- Never let model-sidecar exceptions kill the process; emit `session.error`, close the engine, then close the websocket.
- On disconnect, call `engine.close()`.

## Session State Machine

Suggested states:

```text
idle -> starting -> listening -> assistant_pending -> speaking -> listening -> closing -> closed
```

Keep these pieces of state:

- current final user segment
- partial transcript text and stability
- active assistant draft id
- committed assistant text
- interrupted assistant text
- pending tool calls
- playback generation id
- last inbound and outbound sequence numbers

The session owns persistence. Engines produce events; the session decides which events become durable Hermes messages.

## Hermes Oracle Adapter

The oracle is not Gemma by definition. It is a Hermes adapter that calls the configured Hermes model unless the session explicitly overrides it.

Responsibilities:

- Build a voice-specific prompt wrapper around current transcript state.
- Include normal Hermes system prompt, memory, context, tools, and profile state.
- Enforce tool-call policy for partial vs final transcript.
- Return incremental text guidance to the planner.
- Cancel or supersede in-flight oracle calls on barge-in.

Partial transcript policy:

- Pure answer drafting is allowed.
- Read-only tools may be allowed after a stability threshold.
- Write, shell, browser, messaging, and external side-effect tools require final transcript or confirmation.

## Text Oracle + Streaming TTS Engine

Pipeline:

```text
receive audio chunk
  -> streaming STT frontend
  -> transcript partial/final events
  -> oracle call on final transcript, optionally on stable partials
  -> planner emits text chunks
  -> streaming TTS emits audio chunks
```

Provider choices should be config-driven:

```yaml
voice:
  realtime:
    enabled: true
    engine: text_oracle_tts
    spark_base_url: "http://spark.local:8080"
    frontend_provider: gemma
    frontend_model: gemma-4-e4b
    streaming_stt_provider: local
    tts_provider: edge
```

Do not add new core dependencies for provider-specific engines. Use extras or lazy install paths.

## DGX Spark Sidecar

Treat Spark as a model sidecar, not as the Hermes authority.

Suggested sidecar API:

```text
POST /v1/audio/understand
WS   /v1/audio/stream-understand
WS   /v1/tts/stream
WS   /v1/s2s/session
GET  /health
```

Hermes sends audio and transcript state. Spark returns transcript/frontend state, local draft hints, or audio chunks. Hermes keeps the session, permissions, memory, and tool execution.

Security requirements:

- Bind to LAN-private interface only or require auth.
- Support bearer token.
- Log model names and latencies, not raw audio by default.
- Make raw audio trace capture opt-in.

## Native S2S Engine

Only start this after text-oracle mode works.

The native engine owns low-level speaking:

```text
audio in -> S2S model -> audio out
                    ^
                    |
            Hermes oracle hints
```

The Hermes oracle stream should provide:

- likely intent
- canonical facts from tools/memory/files
- answer plan
- correction hints
- wait/stop/speak directives

Do not try to make the S2S model execute Hermes tools directly. It should receive oracle guidance, not bypass the agent runtime.

## Desktop Implementation

Add a realtime client separate from the existing dictation hook.

Responsibilities:

- Capture microphone frames with WebAudio or AudioWorklet.
- Send frames over websocket with monotonically increasing sequence numbers.
- Maintain captions from transcript and assistant text events.
- Play `audio.output.chunk` through a queue.
- Cancel playback immediately on local barge-in.
- Fall back to the current MediaRecorder blob loop when realtime mode is unavailable.

## Testing Plan

Unit tests:

- protocol serialization and validation
- session state transitions
- persistence boundary: partials are not committed
- barge-in cancels active playback generation
- tool policy rejects unsafe partial-transcript tool calls

Integration tests:

- websocket opens and closes cleanly
- fake STT emits partial/final transcript
- fake oracle streams text
- fake TTS streams audio chunks
- disconnect closes engine resources

Manual checks:

- Start desktop app.
- Enable realtime voice feature flag.
- Speak a short question.
- Confirm partial transcript appears before final transcript.
- Confirm first audio starts before full response completes.
- Interrupt playback and verify it stops.
- Confirm durable transcript contains only final user text and committed assistant text.

Run focused tests with:

```bash
scripts/run_tests.sh tests/agent/test_realtime_voice.py
```

## Implementation Order

1. Land protocol primitives and docs.
2. Add fake in-process `RealtimeVoiceEngine` for websocket testing.
3. Add `/api/voice/realtime` endpoint behind `voice.realtime.enabled`. Done.
4. Add desktop websocket client and playback path. Done.
5. Add STT provider adapter. Partially done by reusing Hermes' existing provider chain at utterance boundaries; true streaming STT remains a provider/sidecar follow-up.
6. Add Hermes oracle adapter. Done.
7. Add TTS adapter. Done by reusing Hermes' existing provider chain.
8. Add barge-in and commit semantics. Done in the session and desktop playback layers.
9. Add Spark sidecar adapter. Done for native S2S websocket bridging and oracle hints.
10. Add native S2S engine. Done as a sidecar-backed engine; model-sidecar deployment is external to Hermes.

## Review Checklist

- Existing one-shot voice mode still works.
- No raw audio is persisted by default.
- No partial transcript is committed to session history.
- Tool calls from unstable transcript are gated.
- Websocket auth matches existing dashboard boundaries.
- Sidecar credentials are profile-safe.
- Provider dependencies are optional or lazy-installed.
