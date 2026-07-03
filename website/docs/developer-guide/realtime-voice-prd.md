---
title: "Realtime Voice PRD"
description: "Product requirements for a KAME-inspired realtime voice subsystem in Hermes"
---

# Realtime Voice PRD

## Summary

Hermes currently treats voice as a message convenience: record audio, transcribe a complete blob, submit the transcript as a normal user message, then synthesize the assistant's streamed text. That works for dictation and hands-free turns, but it is not a live conversation.

This project adds a KAME-inspired realtime voice subsystem behind one session
protocol. The intended production architecture is three-tier:

1. **Reflex / floor-control tier**: a fast realtime voice frontend handles live
   listening, barge-in, short acknowledgements, local clarification, and rough
   transcript hypotheses.
2. **Interpreter / evidence tier**: a Gemma 4 audio-multimodal model reviews
   clipped raw audio plus labeled Moshi/S2S/STT transcript hypotheses and emits
   corrected multilingual evidence, entities, confidence, and oracle request
   patches.
3. **Hermes oracle tier**: the active Hermes model selected through `/model`
   owns tools, memory, MCP, files, approvals, durable transcript promotion, and
   task execution.

Portable STT/TTS providers remain supported as fallback, diagnostics, and
bring-up baselines. They must not be treated as the primary full-KAME control
path when the reflex and interpreter are healthy.

## Production-Readiness Ladder

Realtime voice should ship in visible tiers instead of as a single "done" switch. Each tier keeps the same desktop websocket and Hermes oracle boundary, so users can move inference from a laptop to a LAN sidecar or provider endpoint without changing the desktop app.

0. **One-shot fallback**: existing MediaRecorder upload, transcript submission, and TTS playback. This remains available for machines, browsers, or profiles where realtime preflight fails.
1. **Portable desktop path**: desktop streams microphone frames to Hermes, Hermes owns session state, barge-in, permissions, durable transcript boundaries, and fallback to one-shot voice. This tier must not require local audio-model hardware.
2. **External text-oracle sidecar path**: a loopback, LAN, or provider-backed sidecar supplies STT/audio understanding plus TTS. If the sidecar reports `streaming_stt: true` and `tts: true`, Hermes can treat it as live-like for text-oracle conversation. If it only has utterance STT, it is useful but not Gemini Live-style yet.
3. **Gemma/interpreter LLM path**: Gemma 4 E2B/E4B/12B, or a similar audio-capable frontend model, can run inside the sidecar as an interpreter over clipped raw audio plus provenance-labeled transcript hypotheses. Gemma is not the Hermes oracle; the backend oracle remains whatever Hermes is configured to use for memory, files, tools, MCP, approvals, and profile behavior.
4. **Native S2S/reflex path**: a sidecar speaks directly in a speech-to-speech loop and submits asynchronous oracle jobs to Hermes through a narrow `ask_brain`/`interface.oracle.request` bridge. This is the best long-term path for prosody and interruption feel. The portable text-oracle path remains a fallback and compatibility tier, not the final KAME architecture.
5. **Gemini Live-style production quality**: requires `voice.realtime.production_evidence_report` to point at at least three verified EN/JA smoke report runs by default and `voice.realtime.production_review_report` to point at an evidence-backed launch-review JSON report. The combined gate covers repeatable latency evidence, full audio-session evidence from fixture audio through reflex/interpreter evidence, Hermes oracle text, and TTS, interruption reliability, multilingual metadata preservation, desktop reconnect recovery, graceful fallback, security review, and enough real conversation testing to prove the experience remains coherent under noise, remote sidecar latency, TTS/provider failure, and tool-using Hermes answers. Fallback STT evidence may satisfy fallback coverage only when labeled as fallback.

The ladder is intentionally hardware-neutral. A developer may use a large local inference workstation, a small desktop with cloud STT/TTS, a LAN model server, or a hosted provider, but the product contract is capabilities and evidence: preflight status, sidecar health, live-like `conversation_quality`, evidence-backed `production_readiness`, latency targets, English/Japanese fixture reports, launch-review checks, and safe fallback behavior.

## Goals

- Let a user have a low-latency spoken conversation with Hermes in the desktop app.
- Preserve Hermes' existing data boundary: memory, files, MCP, tools, profiles, approvals, and session state remain owned by Hermes.
- Allow a local process, remote inference host, or cloud endpoint to run Gemma, streaming STT, streaming TTS, and native S2S inference.
- Make the desktop portable by separating microphone/playback/session UI from the machine or service that performs voice inference.
- Autostart the reference sidecar for loopback local/Gemma/vLLM configurations so basic realtime voice does not require a second manual process.
- Keep the desktop UI protocol stable while engines change behind it.
- Preserve the user's spoken language and script by default; realtime voice must not silently translate to English or assume English punctuation/word boundaries.
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

### KAME Reflex + Interpreter + Hermes Oracle

```text
mic audio
  -> realtime reflex / floor-control frontend
       VAD, barge-in, acknowledgement, local clarification
       rough transcript hypothesis when available
  -> Gemma interpreter evidence bundle
       clipped raw audio
       reflex/Moshi transcript hypothesis
       optional STT hypothesis for fallback or literal-evidence checks
       corrected transcript candidate, entities, confidence, disagreements
  -> Hermes oracle job
       active Hermes /model
       memory, files, MCP, tools, profile config, approvals
  -> speech planner
       sentence-level chunks, reasoning/tool trace suppression, commit policy
  -> streaming TTS
  -> audio output
```

### Portable STT/TTS Fallback

```text
mic audio
  -> streaming STT or provider speech understanding
  -> Hermes oracle
       active Hermes /model
  -> streaming TTS
  -> audio output
```

This fallback exists so realtime voice can run on machines without local
audio-model capacity and so providers can be benchmarked. In full KAME mode,
STT output is hypothesis evidence for the interpreter/oracle, not the reflex
driver.

## Product Requirements

### Session Lifecycle

- A voice session starts from the desktop app and receives a Hermes session id.
- The desktop can preflight `GET /api/voice/realtime/status` before opening the microphone or websocket.
- The backend returns `session.started` before accepting audio frames.
- For loopback local/reference/Gemma/vLLM frontends, the backend verifies sidecar health and starts the reference sidecar when configured to do so.
- The session can close from the client, backend, or model-sidecar failure.
- Closing a session must stop playback, cancel pending model/TTS work, and release sidecar resources.

### Audio Input

- Desktop captures microphone audio as small frames, not whole blobs.
- The first implementation may use Opus or WebM/Opus frames; PCM16 is allowed for local/LAN debugging.
- The browser must keep the existing MediaRecorder voice mode as a fallback until realtime mode is stable.
- If the status endpoint reports realtime voice unavailable, the desktop must fall back to the one-shot MediaRecorder voice loop instead of failing after microphone capture starts.
- If a realtime websocket fails during an active session, the desktop must stop active playback, release microphone capture, invalidate queued microphone audio, and reconnect or fall back without leaking stale audio into the next session.

### Speech Understanding

- The engine may emit `transcript.partial` for unstable speech.
- The engine may emit `transcript.final` when a segment is stable enough to commit to the live session state in text-oracle or fallback modes.
- In full KAME mode, Moshi/S2S and ASR transcript strings are hypotheses attached to the same interpreter evidence bundle as the clipped raw audio. They are not committed as durable user text unless promoted by interpreter/oracle judgment.
- Gemma can be used as a speech understanding/interpreter model over raw audio plus transcript hypotheses, but Hermes must not depend on Gemma being the backend oracle model.
- STT/audio-understanding prompts must preserve source language and script unless the user explicitly asks for translation.
- Transcript events may include `language`, `locale`, `script`, and provider confidence metadata when available, but the protocol must also work when a provider cannot identify language.

### Language and Locale

- English and Japanese are the initial production quality targets. Acceptance testing must cover English and Japanese speech input, one-session raw-audio -> reflex/interpreter evidence -> Hermes oracle -> TTS evidence, assistant captions, barge-in behavior, and spoken output before realtime voice is considered ready for general release. Fallback STT/TTS runs should be tested separately and labeled as fallback evidence.
- Other languages are best-effort and provider-driven. Hermes must not reject clean language metadata, silently translate to English, or require EN/JA-only language tags just because the first rollout is optimized for English and Japanese.
- Language selection is provider/config driven, not hardcoded in the realtime protocol.
- The desktop UI language is not automatically the speech language. Use explicit voice/STT config, provider auto-detection, or per-session metadata.
- Speech chunking and planning must handle common non-ASCII sentence and phrase delimiters, and must not require whitespace-delimited words.
- TTS provider selection must allow multilingual voices. If the configured voice cannot speak the transcript/assistant language, the engine should emit a degraded frontend state or fall back to a compatible configured provider instead of forcing English.
- Native S2S sidecars should report language capability in `/health` when known; Hermes uses that for diagnostics, not for granting tool authority.

### Hermes Oracle

- The oracle uses the model Hermes is configured to use. Voice configuration must not add a separate `oracle_model`; model selection stays in the normal Hermes `/model` and provider configuration path.
- The oracle can see corrected transcript, reflex transcript hypothesis, Moshi/S2S transcript hypothesis, optional ASR hypothesis, interpreter confidence, and source provenance when a voice turn escalates.
- Moshi/S2S and ASR transcript text reaches the oracle only as labeled evidence attached to the same raw-audio/interpreter bundle. It must not create a second oracle turn, overwrite the compact reflex intent directly, or become `oracle_text` unless interpreter/oracle judgment explicitly promotes it.
- Tool calls from hypothesis-only speech require a conservative gate. Destructive or external side-effect tools require interpreter/oracle confirmation or explicit user confirmation.
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
- The desktop keeps the microphone stream and VAD/analyser active across assistant playback; only the per-utterance recorder stops at utterance boundaries.
- The backend cancels active TTS/native S2S output and marks uncommitted speech as interrupted.
- The oracle receives the interruption as part of live state.
- Assistant text, audio chunks, commits, and barge-in acknowledgements carry a `playback_generation` so the desktop can discard late audio from interrupted output.
- Acceptance evidence must prove the interrupted output goes quiet after the barge-in acknowledgement; acknowledging `barge_in` while stale assistant audio keeps arriving is not sufficient.

### Persistence

- Durable session history only stores:
  - final user transcript segments promoted by interpreter/oracle judgment
  - committed assistant text
  - tool calls/results that actually executed
  - interruption markers where relevant
- Partial transcripts, Moshi/S2S hypotheses, ASR hypotheses, tentative assistant text, and acoustic chunks are not stored by default.
- Durable oracle recovery records must follow the same rule. A queued or running oracle job may retain hypothesis evidence in the voice-session audit ledger, but durable Hermes chat history must store only promoted user wording, user-visible outcomes, executed tool/action records, and explicit interruption/cancellation markers.
- Implementations must guard both persistence paths: the `persist_user_message` path must prefer promoted interpreter/oracle intent over raw `transcript` fields, and durable oracle records must not preserve raw hypothesis fields as if they were verified user text.

### Security and Privacy

- Raw audio endpoints must be loopback, LAN-private, or authenticated.
- Sidecar access should use a token, mTLS, SSH tunnel, loopback binding, or Tailscale.
- Audio should not be sent to a cloud provider unless the user configured that provider.
- Hermes remains the permission boundary for memory, MCP, file access, and tools.
- Remote voice inference hosts are model/media workers only; they must not receive direct Hermes tool authority.

## Success Metrics

- Median first transcript or reflex hypothesis partial: under 300 ms on LAN.
- Median first assistant text after final user speech: under 500 ms for simple responses.
- Median first assistant audio after final user speech: under 900 ms for simple responses.
- Barge-in stop latency: under 150 ms from detected speech to playback cancellation.
- Realtime server events expose session latency metrics for transcript hypotheses, interpreter evidence, first-text, first-audio, and barge-in paths.
- No durable transcript pollution from partial ASR, Moshi/S2S hypotheses, or abandoned assistant drafts.
- No durable transcript pollution through indirect oracle paths: hypothesis-only transcript strings must not become persisted Hermes user messages via `oracle_text`, request replay, or job recovery.
- Existing one-shot voice mode continues to work.
- Reconnect drills prove the desktop releases the microphone and clears playback/queued audio before starting the next realtime session or fallback loop.
- Production launch-review checks include non-empty notes or artifact references for every passed manual check; booleans alone are not enough to claim Gemini Live-style quality.

## Rollout

1. Add inert protocol types and docs.
2. Add websocket session endpoint behind a config flag.
3. Implement desktop audio frame streaming behind a feature flag.
4. Implement portable STT/TTS fallback with explicit degraded-state reporting.
5. Add reference sidecar adapter for local/provider fallback STT and TTS.
6. Add managed loopback sidecar lifecycle for local/Gemma/vLLM frontends.
7. Add the KAME reflex path and typed asynchronous oracle jobs.
8. Add Gemma interpreter evidence bundles over raw audio plus labeled transcript hypotheses.
9. Keep native S2S/reflex as a first-class frontend path behind the same protocol.

## Open Questions

- Which local reflex/S2S path should provide the first low-latency transcript hypotheses?
- What is the maximum acceptable delay from speech end to Gemma interpreter evidence for escalated turns?
- How aggressively should the planner allow early speech before interpreter evidence arrives?
- Which tool classes are safe during partial transcript state?
- Should voice session traces be saved as optional debug artifacts?
