# Hermes VoiceOps Business Agent Hackathon Summary

## One-Line Pitch

Hermes becomes a VoiceOps hackathon proof targeting Spark/PGX: a household and business operator that can be spoken to over Discord, given a spending budget through Stripe, provision services for itself, and carry the same operating context into a phone call.

## Demo Shape

The demo starts in Discord voice. The user joins a voice channel and talks naturally to Hermes. Hermes listens, acknowledges quickly through the reflex, sends raw-audio evidence plus labeled witness hypotheses to the Gemma interpreter, uses the active Hermes `/model` oracle for durable work, and replies by voice in the channel. Open S2S, STT, TTS, or hosted realtime providers may appear in the demo only as role-labeled components: reflex, interpreter, auxiliary witness, outbound TTS, or degraded fallback.

The demo should make the three-tier claim plainly: the reflex answers the floor,
Gemma interprets the accepted raw-audio cut, and Hermes' active `/model` does
the business work. If Moshi, OpenClaw, VoiceClaw, or classic ASR emits text, the
text is shown as a witness attached to the same audio cut, not as the user's
durable prompt. This lets the video show fast acknowledgement without implying
that the first transcript-like string gets to spend money, provision services,
place calls, or write memory.

The simplest demo phrasing is: "Hermes sends raw voice plus what the realtime
frontend thought it heard to Gemma; only Gemma's promoted interpretation reaches
the active Hermes model." That is the safety story. It keeps Moshi/Open-S2S text
useful for clipped starts, names, numbers, and code-switches without making it
the transcript of record or the thing that authorizes Stripe, NemoClaw, phone,
memory, files, tools, or external messages.

The design should not be described as a parallel STT race. The useful
three-tier description is: the reflex controls the live floor, Gemma interprets
the accepted raw-audio cut with any frontend transcript hypotheses attached,
and Hermes' active `/model` performs the business operation from promoted
evidence. If Moshi/open-S2S text exists, the demo should show it as "what the
frontend believed it heard," then show Gemma accepting, correcting, or
rejecting it before Stripe, NemoClaw, phone, memory, file, tool, or external
message payloads are eligible.

The user then gives Hermes a spending budget. Hermes uses Stripe-backed spending controls and skills to provision a service it needs, such as a VoIP provider account or phone-number-capable communications service.

Once provisioned, Hermes calls the user's phone and continues with the same context from Discord. The handoff demonstrates that Hermes is not just a chatbot in one channel; it is an operating agent that can acquire tools, pay for services, and act across real communication surfaces.

## What We Are Building

We are building a voice-first operations layer for Hermes Agent:

- **Realtime Discord voice interface** for speaking to Hermes in a live channel.
- **KAME-style voice architecture** with a low-latency reflex, auxiliary
  transcript evidence, a Gemma interpreter/evidence lane, and Hermes' normal
  oracle model. Raw audio is the interpreter's primary evidence; Moshi/S2S or
  ASR transcripts are labeled hypotheses, not the control path, scheduler, or
  user message of record.
- **Witness-assisted direct-audio interpretation** where Moshi/Open-S2S text,
  reflex captions, and classic ASR output can help Gemma recover clipped starts,
  names, numbers, and code-switched terms, but only
  `interpreter_promoted`/`oracle_promoted` fields become durable or actionable.
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

First, Discord handles live voice I/O. The Hermes gateway joins a voice channel, receives Discord audio frames, forwards them to the realtime voice sidecar, may post provenance-labeled rough caption or transcript-hypothesis status when enabled, and plays synthesized replies back into the channel.

Second, the voice sidecar and streaming speech bridge handle speech conversion,
playback, and bring-up fallbacks. Cartesia remains useful as a cloud TTS,
comparison, or degraded fallback baseline, but the intended KAME path is not
Cartesia-STT-driven: the reflex owns live floor control, Gemma interprets
clipped raw audio with labeled transcript hypotheses, and Hermes' active oracle
owns durable work.

Third, the local model server runs reproducible model containers:

- **Fast reflex model:** Moshi/PersonaPlex-class S2S or smaller floor-control
  model for barge-in, immediate acknowledgement, and rough transcript
  hypotheses.
- **Optional witness/fallback transcript evidence:** Moshi/S2S transcript output or classic ASR
  output may be passed to Gemma as witness context, and to the oracle only as
  labeled audit context or promoted evidence. It does not drive the reflex turn,
  is not required for voice to work, and must be labeled as hypothesis context
  when attached to raw audio. This evidence is collected opportunistically; the
  demo should not wait for ASR before acknowledging the user or creating a
  raw-audio interpreter request.
- **Interpreter/evidence model:** Gemma 4 E2B/E4B/12B-style audio-multimodal
  model for raw-audio review, multilingual correction, entity extraction, and
  oracle request patches.
- **Active Hermes oracle:** Nemotron 3 Super or the current Hermes active model
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
frontend-witness/reflex/S2S/classic-ASR hypotheses without blocking the voice
loop, and Hermes' active model handles oracle work through the normal Hermes
model path. Moshi/S2S or ASR transcript output is witness context for Gemma and
oracle-visible only as labeled audit context or promoted evidence, not a
replacement for raw-audio interpretation. If the Moshi-style frontend emits a
transcript, it should be attached to the same interpreter request as the clipped
audio segment so Gemma can compare what the live reflex thought it heard
against the waveform. The demo must keep those fields separate in logs and
prompts: raw audio is primary interpreter evidence, Moshi/S2S and ASR text are
labeled hypotheses, and only interpreter/oracle judgment can promote wording
into a durable user request or tool-critical argument.

The current implementation target should be described as evidence-bundle KAME,
not "Moshi STT" and not "Gemma ASR." The user-facing reflex is allowed to be an
open S2S model that responds quickly and emits transcript-looking text, but that
text remains a reflex or auxiliary hypothesis. Gemma is the post-cut
interpreter: it reads the clipped waveform, speaker/timing metadata, the reflex
route, the acknowledgement already spoken, and any Moshi/open-S2S or classic ASR
hypotheses. It promotes only the evidence that is safe to hand to Hermes'
active `/model`.

So the correct demo sentence is not "we replaced STT with Moshi" or "Gemma is
our ASR." The correct sentence is: "Hermes sends raw voice plus frontend witness
text to a Gemma interpreter, and only the promoted result reaches the normal
Hermes oracle." That framing is important because it explains why the system can
use Moshi-style text as context while still rejecting hallucinated commands,
wrong-speaker captions, and clipped-prefix mistakes before Stripe, NemoClaw, or
phone execution.

In the video and operator docs, avoid presenting this as an ASR race. The
stronger claim is that Hermes can hear through multiple sensors while trusting
only the promoted result. The packet order is raw voice, metadata, reflex
state, then transcript hypotheses. The visible result is a witness decision
beside the promoted wording: Moshi/Open-S2S text may be accepted as supporting
evidence, corrected by audio, or rejected as diagnostic only. Only the promoted
interpreter/oracle fields can become the spend reason, provider choice, phone
script, durable transcript, memory write, file write, external message, or tool
argument.

The demo should describe this as witness-assisted interpretation. If Moshi,
VoiceClaw, OpenClaw, or another open-S2S frontend emits an STT-like transcript,
Hermes should send that text to Gemma with the same raw voice clip, not around
Gemma and not as a second Hermes turn. Gemma then decides whether the witness
was useful, incomplete, wrong-speaker, stale, or hallucinated. The active Hermes
oracle receives promoted transcript/intent/entities and compact labeled audit
context; it does not receive unpromoted Moshi text as the durable user prompt.

The same applies when the text came from the reflex itself. A reflex transcript
is a useful witness and routing explanation, not a privileged transcript of
record. The demo artifact should show raw voice, timing, speaker metadata,
reflex route, and every Moshi/OpenClaw/VoiceClaw/reflex/classic-ASR text claim
merged into one interpreter packet before any wording is promoted for Stripe,
NemoClaw, phone, memory, files, or durable chat history.

The open-model strategy is role-based. Moshi/PersonaPlex-class models are reflex
candidates. Gemma 4 audio-multimodal is the interpreter candidate.
Nemotron/Riva-style ASR is auxiliary evidence or fallback. Magpie/Riva,
Piper-class, Cartesia, or another TTS provider only affects outbound speech.
Ultravox/Qwen Omni-style models remain watchlist candidates until local latency
and Discord-audio robustness are measured. None of these providers replaces the
Hermes oracle; the oracle remains the active `/model`.

The concrete implementation packet is one evidence bundle per speech cut. Raw
audio and timing are primary. The reflex route, the acknowledgement already
spoken, and transcript-like text from Moshi, VoiceClaw/OpenClaw, or classic ASR
attach to that bundle with source and authority labels. A field called "Moshi
STT" is still stored as a hypothesis: useful context for Gemma, not the user
message of record.

The packet also records when each witness arrived: `before_raw_audio`,
`with_raw_audio`, or `after_interpreter_start`. The demo artifact should show
that all three phases converge on one `turn_id`, one `audio_segment_ref`, one
`evidence_bundle_id`, and one oracle job. This is the practical proof that
Moshi/OpenClaw/VoiceClaw text is interpreter context beside raw voice, not a
parallel STT-first conversation.

The practical interpreter rule is: include the Moshi/OpenClaw/VoiceClaw witness
text when it exists, but place it after the raw waveform and timing metadata in
the Gemma request. The witness can help recover clipped starts, names, numbers,
and code-switched words; it can also be rejected when it is stale,
wrong-speaker, or hallucinated. The demo artifact should show that adjudication
explicitly rather than hiding the witness text or treating it as canonical STT.

If a VoiceClaw/OpenClaw/Moshi-compatible frontend can only provide text, the
demo should label that turn as degraded compatibility mode. That path can be
useful for bring-up, but it is not full raw-audio KAME and should not be used
as proof for Stripe spend, NemoClaw execution, phone-call payloads, or local
Spark readiness.

The right mental model is witness context. Moshi/OpenClaw/VoiceClaw text tells
Gemma what the realtime frontend believed it heard. That is valuable evidence,
especially for clipped starts, names, numbers, and code-switched speech, but it
is still compared against the waveform and speaker/timing metadata before it can
become action-authoritative.

For the hackathon narrative, describe this as three-tier sensor fan-in:

- Reflex: fastest voice frontend, acknowledges immediately, controls barge-in,
  and may emit a witness transcript.
- Interpreter: Gemma reads raw voice plus witness transcripts in one bundle and
  promotes only corrected evidence.
- Oracle: Hermes' active `/model` performs the real business work, with
  Nemotron 3 Super as the preferred Spark-local NVIDIA target when available.

That wording matters. "Moshi STT drives Hermes" sounds like a voice-message
bot. "Raw voice plus frontend witness evidence is promoted by Gemma before
Hermes spends money or places calls" is the safety and architecture story.

The current demo architecture should therefore be described as:

```text
Discord/phone audio
  -> fast reflex: VAD, barge-in, acknowledgement, local status narration
  -> one evidence bundle: raw audio + timing + reflex route + witness text
  -> Gemma interpreter: accept, correct, or reject witness hypotheses
  -> Hermes active /model oracle: business action, Stripe, NemoClaw, phone
```

The demo artifact should show the interpreter input order explicitly:
`raw_audio -> metadata -> reflex -> transcript_hypotheses`. If raw audio is not
present, that turn is a degraded compatibility path. It may still be useful for
fallback narration or captions, but it is not the proof path for full KAME,
Stripe spending, NemoClaw execution, phone calls, memory, files, external
messages, or Spark-local readiness.

Moshi, VoiceClaw, OpenClaw, and classic ASR are not competing "hearing layers"
in the story. They are witness producers. If they emit text, that text helps
Gemma understand what the live frontend thought it heard, especially when the
start was clipped or the utterance includes names, numbers, or code-switching.
The artifact should also show the failure case: a hallucinated or wrong-speaker
witness is preserved for audit but rejected before it can shape spend,
provisioning, phone-call payloads, durable history, memory, files, or tool
arguments.

The hackathon demo should make the latency story explicit. The reflex is allowed
to acknowledge and narrate the queued work before Moshi/STT text arrives. The
interpreter bundle can then absorb transcript hypotheses early, inline, or late
without creating another Hermes turn. This is how the demo can feel immediate
while still showing a safety trail before Stripe, NemoClaw, phone, memory, file,
or external-message actions execute.

The evidence story should be equally explicit. A successful voice artifact
should show the accepted speech cut's `audio_segment_ref`, VAD/energy gate
decision, reflex acknowledgement, any Moshi/OpenClaw/VoiceClaw/classic-ASR
witness hypotheses, Gemma's accepted/corrected/rejected witness outcome, and
the promoted evidence that the active Hermes `/model` used for the business
operation. That is the difference between a voice-message bot and an agent that
can safely spend money or call a phone from spoken context.

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

The demo artifacts should make the authority gate visible. Stripe spend,
provider provisioning, NemoClaw action packets, outbound phone-call payloads,
memory writes, and external messages should reference `interpreter_promoted` or
`oracle_promoted` evidence before they are eligible to execute. Hypothesis-only
text may explain what the frontend thought it heard, but it cannot become the
spend reason, provider choice, phone script, durable transcript, or tool
argument by itself. The artifact bundle should expose
`unpromoted_witness_sink_checks` for spend, phone, NemoClaw, tool, memory, file,
message, and durable history, and should keep `unpromoted_witness_sink_values`
empty. It should also show that broad Hermes tools were deferred behind
`tool_search`/bridge tools until the active oracle needed them, reducing
context pressure during the live voice session.

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
- The **interpreter** model, preferably Gemma 4, reviews raw audio plus
  frontend-witness/reflex/S2S/classic-ASR transcript hypotheses and produces
  corrected transcript, multilingual intent, entities, confidence, and oracle
  request patches.
- The **oracle** is Hermes' active model and handles tool use, memory, business logic, and longer reasoning.
- **Heavy requests** go directly to Hermes' active oracle model. For the hackathon target, that is Nemotron 3 Super, not an MoA wrapper.
- The reflex should produce real user-visible acknowledgements and status, with
  optional rough caption or transcript-hypothesis text labeled as non-durable
  evidence, not hidden filler audio.
- Voice output should be fragmented into sentence-level chunks so text and speech arrive incrementally instead of waiting for a large monolithic response.
- Moshi/S2S or ASR transcript text is evidence attached to the interpreter
  bundle, not a parallel conversation. It must not become `oracle_text`, a spend
  reason, a call payload, or durable user text unless interpreter/oracle judgment
  promotes it.
- The interpreter prompt must explicitly tell Gemma that raw audio is primary
  and that witness transcripts are clues about what the frontend believed it
  heard, not authoritative user messages.

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
- When a Moshi/OpenClaw/VoiceClaw witness transcript is present, the artifact
  shows it sharing the raw-audio turn's `evidence_bundle_id` rather than
  creating a second Hermes turn. Text-only witness turns are labeled degraded.
- In a multi-human voice channel, witness text is bound to the accepted
  speaker/channel before it can influence promoted evidence. Wrong-speaker,
  wrong-channel, stale, or ambiguous-speaker hypotheses remain audit-only and
  cannot become spend, phone, memory, file, durable-history, or tool payloads.
- A low-energy or VAD-rejected packet with transcript-looking text is suppressed
  before barge-in, interpreter scheduling, oracle scheduling, and transcript
  promotion. The artifact records the suppression so room-tone hallucinations
  cannot become spend, phone, memory, or message intent.
- Partial witness text is superseded by the final same-source/same-kind witness
  in the active evidence bundle, with the partial retained only as provenance.
- The interpreter evidence artifact records whether each witness transcript was
  accepted as support, corrected by raw audio, or rejected/diagnostic-only
  before any Stripe/NemoClaw/phone/tool action uses the wording.
- The oracle request artifact shows promoted interpreter/oracle fields as the
  action source, with unpromoted Moshi/S2S/ASR text retained only as labeled
  audit context.
- The active Hermes model routes through the normal `/model` path, preferably
  to local Nemotron 3 Super when available; any hosted or smaller fallback is
  labeled clearly in the artifact.
- Heavy planning/build/debug requests go directly through the active Hermes oracle, without the Gemma 12B + Nemotron Nano MoA path.
- Stripe-linked provisioning is constrained by an explicit budget.
- Spend/provision/call actions are represented as NemoClaw-style packets before
  live execution.
- The phone handoff preserves context from the Discord turn.

## Headless Readiness Plan

The hackathon package should be runnable without live services first, then
closed by external evidence gates:

- generate the Milestone 0 package headlessly
- run `scripts/voiceops_plan_run.py --dry-audit --package-audit`
- keep readiness blocked until live Discord voice evidence, spend/provisioning
  preflight evidence, and local deployment evidence for the selected
  reflex/interpreter/oracle runtime are attached
- use the package audit to reject false-ready claims, transcript-only action
  payloads, missing promoted evidence, missing `witness_context` hypothesis
  markers, and missing provider-role disclosure

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
- The normal demo path should be described as raw-audio KAME with
  transcript-hypothesis fan-in, not "Moshi STT" or "Gemma ASR." The visible proof
  should show that transcript-looking text is preserved with provenance and then
  either promoted, corrected, or rejected before spend/provisioning/call actions.
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
- Open S2S alternatives are candidates by role, not replacements for the
  interpreter/oracle contract. A fast frontend can improve acknowledgement and
  witness quality, but it cannot promote its own transcript into spend,
  provisioning, phone, memory, file, durable history, or tool arguments.
- Open S2S and STT/TTS alternatives must be described by role in the demo plan:
  reflex, interpreter, optional witness/fallback transcript evidence, outbound
  TTS, or degraded fallback. A fast transcript is not enough to authorize
  spending, provisioning, phone calls, memory writes, files, or durable user
  history.
- External frontend `ask_brain` bridges should still create a normalized
  Hermes oracle-request boundary after acceptance. Placeholders and safe status
  packets are transport state only. Durable session history may keep job ids,
  authority labels, counts, and promoted interpreter/oracle fields, but raw
  Moshi/S2S/ASR/reflex hypothesis strings must not survive as durable user
  wording or action arguments unless promoted.
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
8. Ensure partial transcript hypotheses collapse into a final active witness
   with provenance, rather than becoming parallel oracle context.
9. Add measurement for whether Moshi/open-S2S hypotheses helped or hurt Gemma's
   interpreter output, including clipped prefixes, names, numbers, and rejected
   hallucinated commands.
10. Prove the noise gate rejects low-energy witness text without triggering
   barge-in, interpreter requests, oracle jobs, or durable transcripts.
11. Add role-based provider comparison output so the artifact can say which
   component was used for reflex, interpreter, optional witness/fallback
   transcript evidence, outbound TTS, and degraded fallback in the recorded run.
12. Add one demo artifact where a Moshi/open-S2S transcript and raw voice are
   submitted together as one interpreter packet, proving the text stayed
   witness context until Gemma promoted or rejected it.
13. Implement the phone call handoff with context transfer from the Discord session.
14. Add a preflight command that checks PGX endpoints, sidecar health, Stripe readiness, voice provider config, and Discord gateway state.
