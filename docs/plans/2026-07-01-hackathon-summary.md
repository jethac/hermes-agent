# Hermes VoiceOps Business Agent Hackathon Summary

## One-Line Pitch

Hermes becomes a Spark/PGX-powered household and business operator that can be spoken to over Discord, given a spending budget through Stripe, provision services for itself, and carry the same operating context into a phone call.

## Demo Shape

The demo starts in Discord voice. The user joins a voice channel and talks naturally to Hermes. Hermes listens, acknowledges quickly, reasons with its configured oracle model, and replies by voice in the channel.

The user then gives Hermes a spending budget. Hermes uses Stripe-backed spending controls and skills to provision a service it needs, such as a VoIP provider account or phone-number-capable communications service.

Once provisioned, Hermes calls the user's phone and continues with the same context from Discord. The handoff demonstrates that Hermes is not just a chatbot in one channel; it is an operating agent that can acquire tools, pay for services, and act across real communication surfaces.

## What We Are Building

We are building a voice-first operations layer for Hermes Agent:

- **Realtime Discord voice interface** for speaking to Hermes in a live channel.
- **KAME-style voice architecture** with a low-latency reflex, auxiliary
  transcript evidence, a Gemma interpreter/evidence lane, and Hermes' normal
  oracle model. Raw audio is the interpreter's primary evidence; Moshi/S2S or
  ASR transcripts are labeled hypotheses, not the control path.
- **Local model serving on PGX/Spark-class hardware** using vLLM containers for reproducible iteration.
- **Hermes-native oracle selection** where `/model` and the existing Hermes model configuration remain authoritative.
- **Stripe-enabled spending and provisioning** so Hermes can pay for tools and services under explicit user-granted limits.
- **NemoClaw action boundary** for high-risk operations: proposed purchases,
  account provisioning, credential changes, and outbound calls should be
  packaged as auditable action packets before live execution.
- **Cross-channel continuity** from Discord voice to phone, and later WhatsApp.
- **Operational safety posture** with budgets, auditability, config scoping, and fail-closed voice behavior where appropriate.

## Current Architecture

The current local deployment has three layers.

First, Discord handles live voice I/O. The Hermes gateway joins a voice channel, receives Discord audio frames, forwards them to the realtime voice sidecar, posts live transcript messages, and plays synthesized replies back into the channel.

Second, the voice sidecar and streaming speech bridge handle speech conversion,
playback, and bring-up fallbacks. Cartesia remains useful as a cloud STT/TTS
baseline, but the intended KAME path is not Cartesia-driven: the reflex owns
live floor control, Gemma interprets clipped raw audio with labeled transcript
hypotheses, and Hermes' active oracle owns durable work.

Third, the local model server runs reproducible model containers:

- **Fast reflex model:** Moshi/PersonaPlex-class S2S or smaller floor-control
  model for barge-in, immediate acknowledgement, and rough transcript
  hypotheses.
- **Auxiliary transcript evidence:** Moshi/S2S transcript output or classic ASR
  output may be passed to Gemma and the oracle as supporting evidence, but does
  not drive the reflex turn, is not required for voice to work, and must be
  labeled as hypothesis context when attached to raw audio.
- **Interpreter/evidence model:** Gemma 4 E2B/E4B/12B-style audio-multimodal
  model for raw-audio review, multilingual correction, entity extraction, and
  oracle request patches.
- **Primary oracle model:** Nemotron 3 Super or the current Hermes active model
  selected through the normal `/model` path.
- **Optional fallback model:** a hosted or smaller local model if Super is
  unavailable or too slow for the demo window.

Hermes itself remains the source of truth for the oracle. There is no separate
`oracle_model` setting in the intended design; the oracle is whatever Hermes'
active model is. For the hackathon setup, Hermes' active model should point at
Nemotron 3 Super when that endpoint is available, or a clearly labeled fallback
through the same normal model path.

The previous Gemma 12B + Nemotron Nano mixture-of-agents path is no longer the
live demo strategy. It added too much orchestration latency to voice turns and
made response timing harder to reason about. The demo path is now tiered:
the reflex acknowledges immediately, Gemma interprets the raw audio plus the
reflex/Moshi transcript hypotheses without blocking the voice loop, and Hermes'
active model handles oracle work through the normal Hermes model path. Moshi/S2S
or ASR transcript output is supporting evidence for Gemma and the oracle, not a
replacement for raw-audio interpretation. If the Moshi-style frontend emits a
transcript, it should be attached to the same interpreter request as the clipped
audio segment so Gemma can compare what the live reflex thought it heard against
the waveform. The demo must keep those fields separate in logs and prompts: raw
audio is primary interpreter evidence, Moshi/S2S and ASR text are labeled
hypotheses, and only interpreter/oracle judgment can promote wording into a
durable user request or tool-critical argument.

The current implementation target should be described as evidence-bundle KAME,
not "Moshi STT" and not "Gemma ASR." The user-facing reflex is allowed to be an
open S2S model that responds quickly and emits transcript-looking text, but that
text remains a reflex or auxiliary hypothesis. Gemma is the post-cut
interpreter: it reads the clipped waveform, speaker/timing metadata, the reflex
route, the acknowledgement already spoken, and any Moshi/open-S2S or classic ASR
hypotheses. It promotes only the evidence that is safe to hand to Hermes'
active `/model`.

The concrete implementation packet is one evidence bundle per speech cut. Raw
audio and timing are primary. The reflex route, the acknowledgement already
spoken, and transcript-like text from Moshi, VoiceClaw/OpenClaw, or classic ASR
attach to that bundle with source and authority labels. A field called "Moshi
STT" is still stored as a hypothesis: useful context for Gemma, not the user
message of record.

That means the demo should not present Moshi as "the ASR layer." It should
present Moshi/open-S2S text as the reflex's hearing hypothesis. The stronger
story is that Hermes can keep the voice loop fast, preserve what the realtime
frontend thought it heard, and still require Gemma/interpreter or oracle
promotion before money, credentials, outbound calls, files, memory, or durable
chat history depend on that wording.

For the demo narrative, this is an advantage rather than a complication: Hermes
can acknowledge quickly from the reflex, then show the judges a safer evidence
trail before spending money or placing a call. If a transcript side channel
mishears the user, the artifact should show it as a rejected or corrected
hypothesis instead of silently baking it into the action packet.

## Why This Fits The Hackathon

The hackathon asks for agents that can earn, spend, and run real operations. This project demonstrates all three ingredients in a single workflow:

- **Usefulness:** A household/business operator can speak, search, provision services, and call across channels.
- **Viability:** The architecture runs on local NVIDIA hardware with explicit service boundaries and reproducible Dockerized model serving.
- **Presentation:** The demo has a clear visible arc: talk to Hermes, grant spending, watch it acquire a capability, then receive a real phone call with the same context.

It also aligns with the sponsor stack:

- **NVIDIA:** local accelerated inference, Nemotron-family oracle serving, Spark/PGX deployment story, and NemoClaw-shaped safety for operational actions.
- **Stripe:** spending limits, service provisioning, and auditable agent purchases.
- **Nous/Hermes:** Hermes Agent remains the operating framework and tool-using brain.

## KAME-Style Direction

The final voice architecture should separate low-latency conversational reflexes from slower full-agent reasoning:

- The **reflex** model handles immediate acknowledgement, floor control,
  rough transcript hypotheses, and concise narration of what it is asking the
  oracle to do.
- The **interpreter** model, preferably Gemma 4, reviews raw audio plus the
  reflex/Moshi hypotheses and produces corrected transcript, multilingual
  intent, entities, confidence, and oracle request patches.
- The **oracle** is Hermes' active model and handles tool use, memory, business logic, and longer reasoning.
- **Heavy requests** go directly to Hermes' active oracle model. For the hackathon target, that is Nemotron 3 Super, not an MoA wrapper.
- The reflex should produce real transcript-visible messages, not hidden filler audio.
- Voice output should be fragmented into sentence-level chunks so text and speech arrive incrementally instead of waiting for a large monolithic response.
- Moshi/S2S or ASR transcript text is evidence attached to the interpreter
  bundle, not a parallel conversation. It must not become `oracle_text`, a spend
  reason, a call payload, or durable user text unless interpreter/oracle judgment
  promotes it.

The immediate hackathon build should favor the fastest stable reflex path for
acknowledgement and turn-taking. Gemma should be used as an interpreter/evidence
lane when available. Moshi/S2S transcript output and external STT are auxiliary
evidence for Gemma/oracle work, not the normal reflex driver. Classic ASR should
be retained as a fallback and diagnostic lane, not as a mandatory proof that the
system heard the user.

## Demo Script

1. Join Discord voice with `/voice join`.
2. Ask Hermes to summarize its current operating status and confirm it can hear and speak.
3. Tell Hermes: "I am giving you a small budget. Provision what you need so you can call my phone."
4. Hermes confirms the spending envelope and uses Stripe-linked capabilities to provision a phone/VoIP service.
5. Hermes reports the chosen service and the action taken.
6. Hermes places a call to the user's phone.
7. The phone call starts with context preserved from the Discord conversation.

## Success Criteria

- Discord voice join succeeds without fallback.
- Hermes handles a live utterance through the reflex path and may post
  provenance-labeled transcript/interpreter evidence when available.
- Hermes gives a low-latency acknowledgement shortly after speech end.
- Hermes does not claim it lacks voice capability.
- Hermes replies in sentence-sized voice/text chunks.
- Raw-audio interpreter evidence, reflex transcript hypotheses, Moshi/S2S
  hypotheses, and ASR hypotheses remain separate in logs and durable records.
  Hypothesis-only text does not leak into persisted Hermes user messages or
  Stripe/NemoClaw/phone action payloads.
- The active Hermes model routes to the local PGX Nemotron 3 Super endpoint.
- Heavy planning/build/debug requests go directly through the active Hermes oracle, without the Gemma 12B + Nemotron Nano MoA path.
- Stripe-linked provisioning is constrained by an explicit budget.
- Spend/provision/call actions are represented as NemoClaw-style packets before
  live execution.
- The phone handoff preserves context from the Discord turn.

## Known Constraints

- Nemotron Super is memory-heavy on the PGX/GB10 and has a long cold-start path, but it is the preferred sponsor-aligned oracle for this demo shape.
- Running a fast reflex, Gemma interpreter, and Nemotron Super concurrently is
  the current target when hardware allows. Extra oracle candidates should stay
  stopped unless they are being used for fallback or measurement.
- vLLM reports model context according to the configured serving cap, not necessarily the model's architectural maximum.
- Long context and concurrency trade off directly. For a single-user demo, lower concurrency is the right choice.
- The current Nemotron Super vLLM path may leak reasoning-style text or
  `</think>` markers into assistant content. The demo path must filter this
  before Discord text or TTS output.
- Gemma is the target interpreter/evidence path. External STT should be
  retained only as fallback or additional evidence, not as the normal reflex
  driver. Moshi transcript output is also a hypothesis, not durable truth, but
  it should be valuable context for Gemma when paired with the raw voice clip in
  the same evidence bundle.
- Gemma's role should be called interpreter or evidence adjudicator, not
  background ASR. A corrected transcript from Gemma can become durable only when
  it is promoted with provenance and confidence; otherwise it remains evidence
  attached to the voice-session audit trail.
- The demo should not claim "ASR proved the command." The stronger claim is:
  raw audio, reflex hypothesis, optional Moshi/ASR hypotheses, and Gemma
  correction were preserved separately before any Stripe/NemoClaw/phone action
  became eligible for approval.
- External S2S/frontends should be judged by the same evidence rule. If a
  VoiceClaw/OpenClaw/Moshi-style bridge can provide raw-audio references and
  timing, Hermes passes them to Gemma with transcript hypotheses attached. If it
  can provide only text, the path is useful compatibility evidence, but not proof
  of the full raw-audio KAME interpreter loop.
- The sidecar and gateway must run from the same worktree/version to avoid realtime voice protocol mismatches.

## Immediate Build Priorities

1. Stabilize the local model stack for fast reflex, Gemma interpreter, and
   Nemotron 3 Super with single-user concurrency.
2. Keep the Hermes model interface normal: `/model` and `model.*` config select the oracle.
3. Keep MoA disabled in the live voice path unless a measured future variant proves it improves quality without damaging latency.
4. Make reflex acknowledgements real, visible, sentence-fragmented messages.
5. Wire the Stripe spending/provisioning flow into a demo-safe budgeted path.
6. Add a NemoClaw packet boundary for spending, provisioning, credential, and
   outbound-call actions.
7. Implement the external frontend/interpreter evidence adapter so raw audio,
   reflex intent, Moshi/S2S transcript hypotheses, classic ASR hypotheses, and
   correlation ids remain separate through oracle-job creation.
8. Add measurement for whether Moshi/open-S2S hypotheses helped or hurt Gemma's
   interpreter output, including clipped prefixes, names, numbers, and rejected
   hallucinated commands.
9. Implement the phone call handoff with context transfer from the Discord session.
10. Add a preflight command that checks PGX endpoints, sidecar health, Stripe readiness, voice provider config, and Discord gateway state.
