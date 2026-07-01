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
- **KAME-style voice architecture** with a low-latency reflex/interface model in front of Hermes' normal oracle model.
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

Second, the voice sidecar and streaming speech bridge handle speech conversion and playback. Today this path uses the local realtime voice sidecar plus the Cartesia bridge for streaming STT/TTS.

Third, the PGX model server runs vLLM containers:

- **Reflex/interface model:** `gemma-4-e2b-reflex` at `http://100.113.98.11:8001/v1`
- **Primary oracle model:** `nemotron-3-super-oracle` at `http://100.113.98.11:8004/v1`
- **Optional fallback model:** `gemma-4-12b-oracle` at `http://100.113.98.11:8002/v1` if Super is unavailable or too slow for the demo window.
- **Optional lightweight NVIDIA model:** `nemotron-3-nano-oracle` at `http://100.113.98.11:8003/v1` for later experiments, not the current live demo path.

Hermes itself remains the source of truth for the oracle. There is no separate `oracle_model` setting in the intended design; the oracle is whatever Hermes' active model is. For the hackathon setup, Hermes' active model is pointed at the PGX-hosted Nemotron 3 Super endpoint for the sponsor-aligned serious planning path.

The previous Gemma 12B + Nemotron Nano mixture-of-agents path is no longer the live demo strategy. It added too much orchestration latency to voice turns and made response timing harder to reason about. The demo path is now direct: Gemma E2B handles the low-latency reflex/interface loop, and Hermes' active model, currently Nemotron 3 Super, handles oracle work through the normal Hermes model path.

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

- The **reflex** model handles immediate acknowledgement, intent shaping, and concise narration of what it is asking the oracle to do.
- The **oracle** is Hermes' active model and handles tool use, memory, business logic, and longer reasoning.
- **Heavy requests** go directly to Hermes' active oracle model. For the hackathon target, that is Nemotron 3 Super, not an MoA wrapper.
- The reflex should produce real transcript-visible messages, not hidden filler audio.
- Voice output should be fragmented into sentence-level chunks so text and speech arrive incrementally instead of waiting for a large monolithic response.

The immediate hackathon build should use Gemma E2B's native-audio reflex path where it is stable. ASR is a fallback and evidence lane, not the normal reflex input path.

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
- Hermes hears a live utterance and posts a usable transcript.
- Hermes gives a low-latency acknowledgement shortly after speech end.
- Hermes does not claim it lacks voice capability.
- Hermes replies in sentence-sized voice/text chunks.
- The active Hermes model routes to the local PGX Nemotron 3 Super endpoint.
- Heavy planning/build/debug requests go directly through the active Hermes oracle, without the Gemma 12B + Nemotron Nano MoA path.
- Stripe-linked provisioning is constrained by an explicit budget.
- Spend/provision/call actions are represented as NemoClaw-style packets before
  live execution.
- The phone handoff preserves context from the Discord turn.

## Known Constraints

- Nemotron Super is memory-heavy on the PGX/GB10 and has a long cold-start path, but it is the preferred sponsor-aligned oracle for this demo shape.
- Running Gemma E2B and Nemotron Super concurrently is the current target. Gemma 12B and Nemotron Nano should stay stopped unless they are being used for fallback or measurement.
- vLLM reports model context according to the configured serving cap, not necessarily the model's architectural maximum.
- Long context and concurrency trade off directly. For a single-user demo, lower concurrency is the right choice.
- The current Nemotron Super vLLM path may leak reasoning-style text or
  `</think>` markers into assistant content. The demo path must filter this
  before Discord text or TTS output.
- Native-audio Gemma E2B reflex is the target input path. External STT should be retained only as fallback/evidence, not as the normal reflex driver.
- The sidecar and gateway must run from the same worktree/version to avoid realtime voice protocol mismatches.

## Immediate Build Priorities

1. Stabilize the PGX vLLM stack at `64K` context for Gemma E2B plus Nemotron 3 Super with single-user concurrency.
2. Keep the Hermes model interface normal: `/model` and `model.*` config select the oracle.
3. Keep MoA disabled in the live voice path unless a measured future variant proves it improves quality without damaging latency.
4. Make reflex acknowledgements real, visible, sentence-fragmented messages.
5. Wire the Stripe spending/provisioning flow into a demo-safe budgeted path.
6. Add a NemoClaw packet boundary for spending, provisioning, credential, and
   outbound-call actions.
7. Implement the phone call handoff with context transfer from the Discord session.
8. Add a preflight command that checks PGX endpoints, sidecar health, Stripe readiness, voice provider config, and Discord gateway state.
