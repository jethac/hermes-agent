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
- `agent/realtime_voice_sidecar.py` implements the configured model-sidecar client for local, Gemma/vLLM, and remote STT/TTS frontends.
- `agent/realtime_voice_reference_sidecar.py` implements the reference sidecar server that can run on ordinary machines with configured STT/TTS providers, or call a vLLM/Gemma audio endpoint when available.
- `agent/realtime_voice_text_engine.py` implements the text-oracle path: audio or transcript input, streaming frontend events from a configured sidecar, local STT fallback via Hermes' existing transcription provider chain at utterance boundaries, streaming Hermes oracle deltas, speech planning, and chunked audio output via sidecar TTS or the existing TTS provider chain.
- `agent/realtime_voice_s2s_engine.py` implements the native S2S path as a websocket bridge to a local, remote, or cloud model sidecar. When the sidecar emits final transcript events, Hermes calls the configured oracle model and sends `oracle.hint` events back to the sidecar.
- `hermes_cli/web_server.py` exposes `/api/voice/realtime` behind the same websocket auth and Host/Origin guards as the dashboard chat websocket. For loopback `local`, `reference`, `gemma`, `gemma4`, `lmstudio`, and `vllm` frontends, it can also supervise the reference sidecar process automatically.
- `apps/desktop/src/app/chat/composer/hooks/use-realtime-voice-session.ts` implements the desktop websocket client, microphone frame capture, simple VAD, playback queue, and barge-in cancellation.

The existing one-shot voice mode remains the fallback. Realtime voice is opt-in via `voice.realtime.enabled`.

Current limits:

- In-core local STT still uses Hermes' existing file-based transcription providers after an utterance boundary. True streaming STT is available through the configured sidecar protocol, not through the local provider chain.
- The native S2S model itself is not shipped in Hermes. Hermes provides the sidecar bridge and oracle hint stream; a local, remote, or cloud sidecar owns model inference.
- Audio frames are JSON/base64 for the first implementation. Binary websocket frames can replace this without changing the semantic event contract.

## Portability Boundary

The desktop app should only know about Hermes' `/api/voice/realtime` websocket. It should not know whether speech inference is local, a supervised loopback process, a LAN GPU host, or a provider endpoint.

Hermes owns:

- websocket auth and session lifecycle
- microphone/playback event protocol
- oracle calls, tool gates, memory, files, and permissions
- durable transcript commit policy

The voice inference process owns:

- streaming STT or audio understanding
- streaming TTS or native S2S audio generation
- model-specific media dependencies and GPU scheduling

This split is why `sidecar_base_url` remains server-side configuration. The desktop cannot point Hermes at an arbitrary inference host through query params.

## Target File Layout

```text
agent/
  realtime_voice.py              # shared protocol and engine ABC
  realtime_voice_session.py      # session state machine, sequence, persistence boundary
  realtime_voice_oracle.py       # Hermes oracle adapter around AIAgent/context/tools
  realtime_voice_planner.py      # early-speech, commit, interruption policy
  realtime_voice_sidecar.py      # Gemma/STT/TTS sidecar websocket client
  realtime_voice_reference_sidecar.py # reference local/provider/vLLM sidecar server
  realtime_voice_text_engine.py  # STT -> oracle -> TTS implementation
  realtime_voice_s2s_engine.py   # native S2S sidecar bridge + oracle stream protocol

hermes_cli/
  web_server.py                  # /api/voice/realtime websocket endpoint
  realtime_voice_sidecar.py      # reference sidecar CLI

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
{"type":"assistant.text.partial","session_id":"...","sequence":4,"payload":{"text":"KAME is interesting because ","playback_generation":1}}
{"type":"audio.output.chunk","session_id":"...","sequence":5,"payload":{"codec":"opus","data_b64":"...","playback_generation":1}}
{"type":"assistant.commit","session_id":"...","sequence":6,"payload":{"text":"KAME is interesting because ...","playback_generation":1}}
```

Server events may include a `metrics` object in the payload. The session layer annotates events with monotonic timing data such as `session_elapsed_ms`, `audio_to_partial_transcript_ms`, `audio_to_final_transcript_ms`, `eou_to_final_transcript_ms`, `final_transcript_to_first_text_ms`, `final_transcript_to_first_audio_ms`, and `barge_in_ack_ms`. Engines and sidecars should preserve any existing metric fields they provide; the session appends Hermes-observed timings before forwarding the event to the desktop. The desktop hook keeps the latest valid metrics as a realtime session snapshot for diagnostics and future quality UI.

## Backend Endpoint

Add a FastAPI websocket endpoint in `hermes_cli/web_server.py`:

```python
@app.websocket("/api/voice/realtime")
async def realtime_voice_ws(ws: WebSocket) -> None:
    ...
```

Add a FastAPI status endpoint for desktop preflight and operator diagnostics:

```http
GET /api/voice/realtime/status
```

The status endpoint returns `enabled`, `available`, selected engine/codecs, frontend provider/model, native S2S requirement flags, and sanitized sidecar state:

```json
{
  "enabled": true,
  "available": true,
  "engine": "text_oracle_tts",
  "frontend_provider": "gemma4",
  "sidecar": {
    "mode": "managed_loopback",
    "base_url": "http://127.0.0.1:8765",
    "autostart": true,
    "healthy": false
  }
}
```

A managed loopback sidecar can be `available: true` while `healthy: false` because the websocket path will autostart it. An externally managed remote sidecar that is unhealthy is `available: false`, so the desktop should keep or return to the one-shot voice fallback.

Implementation notes:

- Reuse existing websocket host/origin/auth guards in `web_server.py`.
- Create one `RealtimeVoiceSession` per websocket.
- Validate every client event with `validate_client_event`.
- Never let model-sidecar exceptions kill the process; emit `session.error`, close the engine, then close the websocket.
- On disconnect, call `engine.close()`.
- Never expose sidecar bearer tokens, URL credentials, or query-string secrets through the status endpoint.

## Session State Machine

Suggested states:

```text
idle -> starting -> listening -> assistant_pending -> speaking -> listening -> closing -> closed
```

Keep these pieces of state:

- current final user segment
- partial transcript text and stability
- active assistant draft id
- active playback generation
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
    frontend_provider: gemma4
    frontend_model: google/gemma-4-E4B-it-qat-w4a16-ct
    sidecar_host: 127.0.0.1
    sidecar_port: 8765
    vllm_base_url: "http://voice-gpu.local:8000/v1"
    vllm_model: google/gemma-4-E4B-it-qat-w4a16-ct
    tts_provider: edge
```

Do not add new core dependencies for provider-specific engines. Use extras or lazy install paths.

## Reference And Remote Inference Sidecars

Treat every sidecar as a model/media process, not as the Hermes authority. Hermes keeps the session, permissions, memory, and tool execution.

Implemented sidecar websocket:

```text
WS /v1/realtime-text/session
```

Hermes sends a `session.config` frame first, then forwards `audio.input.chunk`, `barge_in`, and assistant text chunks with `{"speak": true}`. The sidecar returns `transcript.partial`, `transcript.final`, `frontend.state`, `audio.output.chunk`, or `session.error` events using the shared wire protocol.

Reference sidecar command:

```bash
python -m hermes_cli.realtime_voice_sidecar --host 127.0.0.1 --port 8765
```

Gemma/vLLM audio frontend:

```bash
python -m hermes_cli.realtime_voice_sidecar \
  --host 127.0.0.1 \
  --port 8765 \
  --vllm-base-url http://voice-gpu.local:8000/v1 \
  --vllm-model google/gemma-4-E4B-it-qat-w4a16-ct
```

The vLLM runtime must include audio dependencies. If the server returns `Invalid or unsupported audio file` and logs `Please install vllm[audio] for audio support`, install or bake `av`, `librosa`, `soundfile`, and `soxr` into the vLLM image.

Hermes config for no-special-hardware local mode:

```yaml
voice:
  realtime:
    enabled: true
    engine: text_oracle_tts
    frontend_provider: local
    sidecar_host: 127.0.0.1
    sidecar_port: 8765
    sidecar_autostart: true
```

With `sidecar_autostart: true`, Hermes checks `GET /health` on the loopback sidecar URL before accepting a realtime voice websocket. If the sidecar is absent, Hermes starts:

```bash
python -m hermes_cli.realtime_voice_sidecar --host 127.0.0.1 --port 8765
```

Hermes config for an externally managed remote inference sidecar:

```yaml
voice:
  realtime:
    enabled: true
    engine: text_oracle_tts
    frontend_provider: gemma4
    frontend_model: google/gemma-4-E4B-it-qat-w4a16-ct
    sidecar_base_url: "http://voice-inference.local:8765"
    sidecar_token_env: HERMES_VOICE_SIDECAR_TOKEN
    sidecar_autostart: false
```

For `gemma4` or `vllm` frontends, the same supervised sidecar can call a remote vLLM audio endpoint through `vllm_base_url` and `vllm_model`. If `sidecar_base_url` points at a non-loopback host, Hermes treats that as an externally managed inference host and does not spawn a local process. `spark_base_url` remains a deprecated compatibility alias for existing private profiles.

Suggested sidecar API expansion:

```text
POST /v1/audio/understand
WS   /v1/audio/stream-understand
WS   /v1/tts/stream
WS   /v1/s2s/session
GET  /health
```

Hermes sends audio and transcript state. The sidecar returns transcript/frontend state, local draft hints, or audio chunks.

Security requirements:

- Bind to LAN-private interface only or require auth.
- Support bearer token.
- Log model names and latencies, not raw audio by default.
- Make raw audio trace capture opt-in.

## Native S2S Engine

Native S2S is a first-class engine family behind the same session protocol. It does not need to wait for the text-oracle path to be "done"; the shared contract is the requirement.

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
- Keep the microphone stream and analyser alive across assistant playback; stop and recreate only the per-utterance `MediaRecorder`.
- Track `playback_generation` and drop stale audio chunks from interrupted assistant output.
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
- fake sidecar receives browser audio and assistant TTS chunks
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
python -m pytest tests/hermes_cli/test_web_server.py::TestRealtimeVoiceWebSocket -q
npm run test:ui -- src/app/chat/composer/hooks/use-realtime-voice-session.test.ts
```

## Implementation Order

1. Land protocol primitives and docs.
2. Add fake in-process `RealtimeVoiceEngine` for websocket testing.
3. Add `/api/voice/realtime` endpoint behind `voice.realtime.enabled`. Done.
4. Add desktop websocket client and playback path. Done.
5. Add STT provider adapter. Done for the sidecar streaming path; local fallback still reuses Hermes' existing provider chain at utterance boundaries.
6. Add Hermes oracle adapter. Done.
7. Add TTS adapter. Done by reusing Hermes' existing provider chain.
8. Add barge-in and commit semantics. Done in the session and desktop playback layers.
9. Add reference sidecar server. Done for local/provider STT/TTS and optional vLLM audio frontend.
10. Add managed local sidecar lifecycle. Done for loopback local/reference/Gemma/vLLM frontends; remote inference sidecars remain externally managed.
11. Add remote inference sidecar adapter. Done for text-oracle Gemma/STT/TTS streaming, native S2S websocket bridging, and oracle hints.
12. Add native S2S engine. Done as a sidecar-backed engine; model-sidecar deployment is external to Hermes and remains a first-class production track.

## Review Checklist

- Existing one-shot voice mode still works.
- No raw audio is persisted by default.
- No partial transcript is committed to session history.
- Tool calls from unstable transcript are gated.
- Websocket auth matches existing dashboard boundaries.
- Sidecar credentials are profile-safe.
- Provider dependencies are optional or lazy-installed.
