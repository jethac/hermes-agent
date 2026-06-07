---
title: "Realtime Voice PRD"
description: "Product requirements for a KAME-inspired realtime voice subsystem in Hermes"
---

# Realtime Voice PRD

## Summary

Hermes currently treats voice as a message convenience: record audio, transcribe a complete blob, submit the transcript as a normal user message, then synthesize the assistant's streamed text. That works for dictation and hands-free turns, but it is not a live conversation.

This project adds a KAME-inspired realtime voice subsystem. The subsystem must support two engine families behind one session protocol:

1. **Text oracle + streaming TTS**: a speech understanding frontend, a Hermes oracle backed by the configured Hermes model, and streaming text-to-speech output.
2. **Native S2S + Hermes oracle**: a Moshi/KAME-style speech-to-speech frontend that receives asynchronous oracle hints from Hermes.

The production architecture has three deployment tiers behind the same desktop protocol:

1. **Local/provider tier**: no special hardware; local or cloud-configured STT/TTS providers drive the text-oracle path.
2. **Gemma/vLLM audio tier**: a LAN or local sidecar hosts Gemma 4 E4B-IT audio understanding for higher-quality speech frontend behavior.
3. **Native S2S tier**: a speech-to-speech sidecar emits low-latency audio directly while receiving Hermes oracle/tool hints.

## Goals

- Let a user have a low-latency spoken conversation with Hermes in the desktop app.
- Preserve Hermes' existing data boundary: memory, files, MCP, tools, profiles, approvals, and session state remain owned by Hermes.
- Allow a local process, remote inference host, or cloud endpoint to run Gemma, streaming STT, streaming TTS, and native S2S inference.
- Make the desktop portable by separating microphone/playback/session UI from the machine or service that performs voice inference.
- Autostart the reference sidecar for loopback local/Gemma/vLLM configurations so basic realtime voice does not require a second manual process.
- Keep the desktop UI protocol stable while engines change behind it.
- Support barge-in: user speech interrupts assistant playback and updates the live session state.
- Commit only stable transcript and assistant text to durable Hermes session history.

## Non-goals

- Train a native speech-to-speech foundation model from scratch.
- Replace the normal Hermes message loop.
- Expose raw Hermes memory or tool results directly to speech vendors outside the existing permission model.
- Require special GPU hardware for basic use.
- Remove the existing one-shot voice mode.

## User Stories

- As a user, I can talk to Hermes without pressing send after every utterance.
- As a user, I can interrupt Hermes while it is speaking and have the assistant stop immediately.
- As a user, I can ask about my actual Hermes context, files, tools, and memories during a voice session.
- As a user without local GPU hardware, I can still use realtime voice through configured STT/TTS providers.
- As a user with a remote inference host, I can run low-latency speech/frontend models over the LAN.
- As a developer, I can swap the voice engine from text-oracle-TTS to native S2S without rewriting the desktop UI.

## Architecture

```text
Desktop mic stream
  -> /api/voice/realtime websocket
  -> RealtimeVoiceSession
       -> selected RealtimeVoiceEngine
       -> Hermes oracle
       -> tool and permission gates
       -> durable transcript commit logic
  -> assistant audio stream
  -> desktop playback
```

The desktop talks only to Hermes' realtime websocket. Voice inference may run in-process, in a supervised loopback sidecar, on another LAN machine, or through a provider endpoint. The backend decides which engine/sidecar to use from profile config; the desktop protocol does not change.

### Text Oracle + Streaming TTS

```text
mic audio
  -> speech understanding frontend
       Gemma audio input, streaming STT, Whisper, or provider STT
  -> frontend state
       partial transcript, final transcript, intent, confidence, barge-in
  -> Hermes oracle
       configured Hermes model
       memory, files, MCP, tools, profile config, approvals
  -> speech planner
       decides when text is stable enough to speak
  -> streaming TTS
  -> audio output
```

### Native S2S + Hermes Oracle

```text
mic audio
  -> native S2S frontend
       starts speaking in acoustic-token space
       receives oracle hints from Hermes
  -> speech audio

Hermes oracle
  -> asynchronous guidance stream
       facts, task intent, tool results, correction hints, stop/wait hints
```

## Product Requirements

### Session Lifecycle

- A voice session starts from the desktop app and receives a Hermes session id.
- The backend returns `session.started` before accepting audio frames.
- For loopback local/reference/Gemma/vLLM frontends, the backend verifies sidecar health and starts the reference sidecar when configured to do so.
- The session can close from the client, backend, or model-sidecar failure.
- Closing a session must stop playback, cancel pending model/TTS work, and release sidecar resources.

### Audio Input

- Desktop captures microphone audio as small frames, not whole blobs.
- The first implementation may use Opus or WebM/Opus frames; PCM16 is allowed for local/LAN debugging.
- The browser must keep the existing MediaRecorder voice mode as a fallback until realtime mode is stable.

### Speech Understanding

- The engine emits `transcript.partial` for unstable speech.
- The engine emits `transcript.final` when a segment is stable enough to commit to the live session state.
- Gemma can be used as a speech understanding/frontend model, but Hermes must not depend on Gemma being the backend oracle model.

### Hermes Oracle

- The oracle uses the model Hermes is configured to use unless the voice session explicitly overrides it.
- The oracle can see final transcript and selected partial transcript state.
- Tool calls from partial speech require a conservative gate. Destructive or external side-effect tools require final transcript or explicit confirmation.
- Oracle output is guidance until committed by the speech planner.

### Speech Planning

- The planner may emit short acknowledgements quickly.
- The planner must wait for more speech when the user query is incomplete.
- The planner must suppress raw tool traces, JSON, and hidden reasoning from audio output.
- The planner must support correction when partial transcript changes meaning.
- The planner must mark assistant output as interrupted when barge-in occurs before a commit.

### Output

- The text-oracle engine emits `assistant.text.partial` and `audio.output.chunk`.
- Streaming TTS should start before the full response is complete.
- The native S2S engine may emit only audio plus optional committed transcript.
- The desktop app should show captions when text is available.

### Barge-in

- User speech during playback sends a `barge_in` event.
- The backend cancels active TTS/native S2S output and marks uncommitted speech as interrupted.
- The oracle receives the interruption as part of live state.

### Persistence

- Durable session history only stores:
  - final user transcript segments
  - committed assistant text
  - tool calls/results that actually executed
  - interruption markers where relevant
- Partial transcripts, tentative assistant text, and acoustic chunks are not stored by default.

### Security and Privacy

- Raw audio endpoints must be loopback, LAN-private, or authenticated.
- Sidecar access should use a token, mTLS, SSH tunnel, loopback binding, or Tailscale.
- Audio should not be sent to a cloud provider unless the user configured that provider.
- Hermes remains the permission boundary for memory, MCP, file access, and tools.
- Remote voice inference hosts are model/media workers only; they must not receive direct Hermes tool authority.

## Success Metrics

- Median first transcript partial: under 300 ms on LAN.
- Median first assistant audio after final user speech: under 900 ms for simple responses.
- Barge-in stop latency: under 150 ms from detected speech to playback cancellation.
- No durable transcript pollution from partial ASR or abandoned assistant drafts.
- Existing one-shot voice mode continues to work.

## Rollout

1. Add inert protocol types and docs.
2. Add websocket session endpoint behind a config flag.
3. Implement desktop audio frame streaming behind a feature flag.
4. Implement text-oracle engine with streaming STT and streaming TTS.
5. Add reference sidecar adapter for local/provider STT and TTS.
6. Add managed loopback sidecar lifecycle for local/Gemma/vLLM frontends.
7. Add remote inference sidecar adapter for Gemma/vLLM audio frontend and TTS.
8. Keep native S2S as a first-class engine path behind the same protocol.

## Open Questions

- Which streaming STT should be the first supported provider?
- Should Gemma receive raw audio chunks, transcript chunks, or both in the first prototype?
- How aggressively should the planner allow early speech before the final transcript?
- Which tool classes are safe during partial transcript state?
- Should voice session traces be saved as optional debug artifacts?
