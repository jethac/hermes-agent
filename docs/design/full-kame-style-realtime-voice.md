# Full KAME-Style Realtime Voice Design

Status: design draft; packet contract active in `docs/kame-session-v1.md`
Target branch: `wip/full-kame-reflex-voice`
Target deployment: one DGX Spark as the intended local appliance path. Hackathon
artifacts may prove the workflow and evidence gates before Spark-local
readiness is proven; cloud providers remain allowed only as clearly labeled
bring-up fallbacks.
Preferred local reflex: fastest stable floor-control model, such as Moshi/PersonaPlex-class S2S or a smaller timing/noise-gated model
Preferred local interpreter: Gemma 4 E2B/E4B/12B audio-multimodal, consuming bounded raw-audio cuts plus witness hypotheses
Preferred local oracle target: Hermes active `/model`, with Nemotron 3 Super as the first Spark-local NVIDIA target to measure before readiness claims

## Latest Design Lock

2026-07-05: the intended KAME shape is a three-tier hearing stack with
witness fan-in, not a parallel STT architecture.

```text
live audio
  -> reflex: lowest-latency floor control, barge-in, ack, provisional route
  -> interpreter: Gemma direct-audio review of the accepted speech cut
  -> oracle: Hermes active /model, selected through the normal Hermes model UI
```

Moshi/Open-S2S, VoiceClaw/OpenClaw, reflex captions, and classic ASR may still
produce transcript-looking text. That text is useful, and should be supplied to
Gemma when it belongs to the same accepted speech cut as the waveform. It is
not a second Hermes turn, not `oracle_text`, not the transcript of record, and
not action authority. It belongs in `transcript_hypotheses[]` as
`role = "witness_context"`, `authority = "hypothesis"`, and
`tool_authority = false`.

The interpreter packet should therefore contain raw audio first, then timing,
energy, speaker, channel, and transport metadata, then reflex state, then
Moshi/Open-S2S/reflex/classic-ASR witness text. Gemma can use that witness to
repair clipped starts, names, numbers, code-switched speech, or rough intent,
but durable wording starts only at `interpreter_promoted` or
`oracle_promoted`. Stripe, NemoClaw, phone, memory, file, message, and tool
payloads must never be populated from unpromoted witness text.

The design intentionally allows a Moshi/open-S2S transcript to improve Gemma's
understanding without making it the system transcript. Treat provider text as
an evidence claim about a bounded audio segment: "this frontend believed the
human said X during this time window." The interpreter compares that claim with
the waveform, energy gate, speaker/channel metadata, reflex route, and session
state. It may accept the claim, correct it from audio, or reject it as stale,
wrong-speaker, wrong-channel, low-energy, hallucinated, or provider-conflicting.
The oracle receives only the interpreter's promoted result and compact witness
audit metadata.

This means "Moshi STT" is adapter-edge shorthand only. Inside Hermes the
supported operation is "attach the Moshi witness to the raw-audio interpreter
bundle." If the frontend provides text without raw audio, that is degraded
compatibility mode, useful for captions or clarification, but insufficient for
full KAME evidence or high-risk action gates.

The interpreter prompt must make that authority boundary explicit. Gemma should
be asked to judge the waveform first, then use Moshi/Open-S2S/reflex/classic-ASR
text only as a witness claim about what a frontend believed it heard. The prompt
must require one adjudication per hypothesis, must allow the witness to repair
clipped prefixes or ambiguous names/numbers, and must also allow rejection when
energy, timing, speaker, channel, waveform, or session evidence disagrees. It
must not ask Gemma to run "ASR" as a sibling lane or to pick the earliest text as
truth. The only output eligible for Hermes' active `/model` is promoted
interpreter wording, intent, entities, confidence, and compact witness audit
metadata.

Classic ASR therefore remains an optional evidence/fallback producer, not a
normal-path dependency. A missing Moshi/classic-ASR transcript must not delay the
reflex acknowledgement or raw-audio interpreter submission when the speech gate
accepted a cut. A present transcript must improve or explain interpretation
without gaining authority by being fast.

The implementation target is now **witness-assisted direct-audio**. This means
Hermes should prefer a single interpreter packet that contains the bounded raw
audio cut plus every same-cut hearing hypothesis the frontend can provide. A
Moshi/Open-S2S transcript, VoiceClaw/OpenClaw caption, reflex caption, or
classic-ASR string may be sent to Gemma with the waveform, but only as context
for adjudication. It is not a separate ASR lane, not a second Hermes user turn,
and not a fallback source of action authority.

The normative packet contract for that behavior is `docs/kame-session-v1.md`.
When this design document and implementation details disagree, prefer the
contract language there: same accepted speech cut, raw audio primary,
transcript-looking fields as `transcript_hypotheses[]`, explicit witness
adjudication, and only promoted interpreter/oracle fields reaching Hermes'
active `/model` or any action sink.

Runtime enforcement follows the same authority model. A raw-audio job cannot
become executable just because the incoming request labels its text as
`interpreter_promoted`, `oracle_promoted`, `gemma_interpreter`, or any similar
trusted-looking source. For full KAME turns, promotion is a job-owned evidence
transition: Hermes must observe interpreter/oracle evidence attached to the
same speech cut and then patch the queued job. Until that happens, the job may
be visible to the reflex but the Hermes `/model` runner and every irreversible
action sink stay gated.

The interpreter should emit two different classes of output:

- `witness_adjudications`: per-hypothesis decisions such as
  `accepted_as_supporting_evidence`, `corrected_by_audio`, or
  `rejected_or_diagnostic_only`
- `interpreter_promoted`: the corrected wording, intent, entities, confidence,
  and compact audit summary that Hermes' active `/model` may use

This separation is mandatory for household/business actions. The oracle can be
slow, local, hosted, Nemotron, or any model selected by Hermes' normal `/model`
interface; the voice stack must still hand it only promoted evidence plus
compact provenance, never raw witness strings masquerading as verified speech.

Prompting rule: do not prompt Gemma to "choose the transcript" or "run ASR in
parallel." Prompt it to adjudicate an evidence bundle. The prompt should state
that raw audio is primary, witness text is context, every hypothesis needs an
outcome, and only `interpreter_promoted` fields can be used by Hermes'
active `/model`.

## Source Of Truth

The current architecture is **reflex -> interpreter -> oracle**.

- The reflex is the low-latency live interface. It owns VAD-adjacent timing,
  barge-in, acknowledgement, provisional route, and short narration.
- The interpreter is Gemma-style direct-audio adjudication. It receives the
  clipped waveform first, then timing/speaker/channel metadata, then reflex
  state, then any transcript-looking witness text.
- The oracle is Hermes' active `/model`. Voice config must not introduce a
  separate `oracle_model`; the existing Hermes model selector remains the
  business/action brain.

Current amendment, 2026-07-05: the intended shape is a three-tier
direct-audio interpreter loop with witness transcripts as context. It is not a
parallel ASR design. The reflex may be Moshi/Open-S2S-like and may emit a
transcript-looking string, but that text is a sensor reading from the live
interface. If raw audio is available, Hermes must attach the string to the same
interpreter bundle as `transcript_hypotheses[]`; it must not turn the string
into a second Hermes turn, a durable user message, `oracle_text`, a spend
reason, a phone payload, or a tool argument.

The target packet order is intentionally strict:

```text
raw_audio
metadata: VAD, energy, speaker, channel, transport, timing
reflex: route, acknowledgement already spoken, provisional intent
transcript_hypotheses: Moshi/Open-S2S/reflex/classic-ASR witness strings
```

Gemma receives that packet as a direct-audio interpreter, not as an ASR service.
It may use the Moshi/Open-S2S witness to recover clipped starts, names,
numbers, code-switched terms, and rough intent. It must also be free to reject
that witness when waveform, timing, energy, speaker, channel, or session
evidence disagrees. Only Gemma's `interpreter_promoted` fields, or later
`oracle_promoted` fields, can become durable/actionable wording for the active
Hermes `/model`.

Moshi/Open-S2S output is allowed and useful, but it is not the control path.
When a Moshi-like frontend can provide both raw voice and an STT-looking string
for the same accepted speech cut, Hermes should send both to Gemma in one
interpreter bundle. The text is `frontend_witness_hypothesis` or a narrower
source-specific hypothesis. It tells Gemma what the realtime frontend believed
it heard; it does not become durable user text, `oracle_text`, a tool argument,
spend reason, phone payload, memory/file content, or external message unless
Gemma or the Hermes oracle promotes it.

Classic ASR is retained only as fallback, diagnostics, captions, or explicit
literal-evidence support. The normal KAME path must not wait for ASR before
acknowledging the user, creating the raw-audio interpreter request, or queuing a
provisional oracle job from the reflex route. Missing ASR evidence is not a
normal-path failure when raw audio, speech-gate evidence, and reflex route are
available.

The evidence proof for this design is one `turn_id`, one `audio_segment_ref`,
one `evidence_bundle_id`, and one oracle job per accepted speech cut. Early,
inline, and late Moshi/Open-S2S/VoiceClaw/OpenClaw/reflex/classic-ASR text must
merge into that same bundle as witness evidence. It must never fork a second
Hermes turn.

The current direct-audio interpreter packet should be understood as
witness-assisted, not ASR-first. A Moshi/Open-S2S transcript can be extremely
useful, but only when it is bound to the same accepted speech cut as the
waveform and passed to Gemma as context. The adapter must preserve the
frontend's text under `transcript_hypotheses[]` with source, latency,
confidence when available, speaker/channel guesses, `arrival_phase`,
`role = "witness_context"`, `authority = "hypothesis"`, and
`tool_authority = false`. Gemma may then accept it as support, correct it from
the waveform, or reject it as diagnostic-only. Until that promotion happens,
the text cannot become `oracle_text`, durable history, a Stripe/NemoClaw spend
reason, a phone payload, a tool argument, a memory/file write, or an external
message.

This is deliberately different from running Moshi as a parallel STT lane. The
fast reflex can acknowledge and route immediately after speech end, the
Moshi/Open-S2S witness can arrive before/with/after the raw-audio packet, and
the evidence merger must still produce one bundle and one oracle job. If the
frontend can provide only text and no waveform, that is degraded compatibility,
not the full KAME path and not sufficient for high-risk action gates.

## Canonical Short Form

Current decision as of 2026-07-05:

```text
Discord/phone/desktop audio
  -> reflex: floor control, VAD/energy gate, barge-in, ack, provisional route
  -> interpreter bundle: raw audio + metadata + reflex state + transcript hypotheses
  -> Gemma interpreter: accept, correct, or reject the witness text
  -> Hermes active /model: durable reasoning, tools, memory, spend, calls
```

The Moshi/Open-S2S transcript question has a narrow answer: yes, provide it to
Gemma with the raw voice, but only as same-cut witness context. It is useful
because it tells the interpreter what the live frontend believed it heard. It is
dangerous if treated as a parallel STT lane, a second Hermes turn, or a durable
user message. Normalized packets therefore put provider text under
`transcript_hypotheses[]` with `role = "witness_context"`,
`authority = "hypothesis"`, and `tool_authority = false`; promoted text must
come later from `interpreter_promoted` or `oracle_promoted` fields.

The implementation should optimize for the fastest post-speech acknowledgement
from the reflex, then let Gemma adjudicate the accepted speech cut
asynchronously. It should not wait for Moshi/STT text before acknowledging or
creating the interpreter bundle when raw audio, speech-gate evidence, and a
reflex route are already available.

### Current Implementation Directive

The next implementation pass should treat Moshi/Open-S2S text as an optional
witness attached to the direct-audio interpreter packet, not as the voice
pipeline itself. The adapter should submit the accepted raw-audio cut to Gemma
as soon as the speech gate closes, then include any same-cut Moshi/reflex/
classic-ASR text under `transcript_hypotheses[]` when it is available.

This makes Moshi useful without giving it authority. It can help Gemma recover
clipped prefixes, names, numbers, code-switched phrases, and rough intent, but
it cannot create a separate Hermes turn, populate `oracle_text`, or feed
Stripe, NemoClaw, phone, file, memory, message, or tool payloads. Only
`interpreter_promoted` or `oracle_promoted` fields may reach the active Hermes
`/model`.

Acceptance evidence for this directive should show one `turn_id`, one
`audio_segment_ref`, one `evidence_bundle_id`, one `evidence_merge_key`, and
one oracle job lifecycle across all witness arrival phases:
`before_raw_audio`, `with_raw_audio`, and `after_interpreter_start`.

2026-07-05 amendment: the intended shape is not "reflex plus STT plus oracle"
and not "Gemma ASR in parallel." It is a three-tier sensor-fan-in loop:

```text
live audio -> reflex floor control
accepted speech cut -> Gemma direct-audio interpreter
promoted evidence -> Hermes active /model oracle
```

If the reflex or an open S2S frontend such as Moshi also emits an STT-looking
string, that string should travel with the same accepted speech cut as
`transcript_hypotheses[]`. It is a sensor reading from the frontend, not a
second prompt. The interpreter receives the waveform first, metadata second,
reflex state third, and witness text last. This lets Gemma use the witness for
clipped starts, names, numbers, and code-switched phrases while still rejecting
hallucinated, stale, wrong-speaker, wrong-channel, low-energy, or
waveform-conflicting text before the active Hermes oracle sees the request.

The practical implementation consequence is that adapters should prefer a
single bundle shape over provider-specific "STT" pathways. A Moshi, OpenClaw,
VoiceClaw, reflex-caption, or classic-ASR string becomes
`frontend_witness_hypothesis`, `s2s_transcript_hypothesis`,
`reflex_transcript_hypothesis`, or `classic_asr_hypothesis` on the same
`turn_id` and `audio_segment_ref`. Only `interpreter_promoted` or
`oracle_promoted` fields may become durable chat text, Stripe/NemoClaw spend
reasons, phone payloads, tool arguments, memory/file writes, or external
messages.

Adapter acceptance rule: when a frontend can provide raw audio plus a
Moshi/Open-S2S transcript, preserve both signals and send one interpreter
packet. Do not translate the Moshi string into the user message, do not start a
separate STT-first oracle turn, and do not wait for a classic-ASR result before
the reflex acknowledges. The Gemma interpreter should see the waveform first
and the Moshi text as a labeled witness clue. The oracle should see only
promoted transcript/intent/entity fields and compact audit metadata, never raw
unpromoted witness text as the prompt.

2026-07-05 amendment: the Moshi/Open-S2S transcript is explicitly interpreter
context, not a parallel STT lane. If the frontend produces a same-cut waveform
and text, Hermes sends both to Gemma in one packet: raw audio first, metadata
second, reflex state third, `transcript_hypotheses[]` last. The text should be
classified as `frontend_witness_hypothesis` by default, with `source = "moshi"`
or the precise provider name, `role = "witness_context"`,
`authority = "hypothesis"`, and `tool_authority = false`. The interpreter may
accept, correct, or reject that witness. Only `interpreter_promoted` or
`oracle_promoted` fields can become durable user wording, Stripe/NemoClaw
spend reasons, phone payloads, memory/file writes, external messages, or tool
arguments.

## Current Decision Record

Date: 2026-07-04

The KAME voice design is now a three-tier sensor fan-in system:

1. **Reflex:** the lowest-latency live voice loop. It owns VAD-adjacent floor
   control, barge-in, acknowledgement, short status narration, and provisional
   routing.
2. **Interpreter:** a Gemma-style direct-audio model. It receives one bounded
   speech cut after the energy/noise gate, with raw audio first and every
   transcript-looking side signal as labeled witness context.
3. **Oracle:** Hermes' active `/model`. Voice config must not choose a separate
   oracle model; `/model` and the normal Hermes provider config remain the
   authority for durable reasoning and tools.

This is not an STT-first pipeline and not a parallel ASR race. If Moshi,
OpenClaw, VoiceClaw, a reflex caption path, or classic ASR emits text for the
same accepted speech cut, Hermes should pass that text to Gemma alongside the
raw waveform. The text is `frontend_witness_hypothesis`,
`reflex_transcript_hypothesis`, `s2s_transcript_hypothesis`, or
`classic_asr_hypothesis` context. It is never the transcript of record merely
because it arrived first.

The normal interpreter payload is:

```text
raw_audio
metadata: VAD, energy, speaker, channel, timing
reflex: route, acknowledgement already spoken, provisional intent
transcript_hypotheses: Moshi/OpenClaw/VoiceClaw/reflex/classic-ASR witnesses
```

Gemma may use a Moshi/Open-S2S witness to recover clipped starts, names,
numbers, code-switching, or a rough intent. Gemma must also be able to reject
that witness when the waveform, speaker/channel metadata, energy gate, or
conversation state contradicts it. Only `interpreter_promoted` or
`oracle_promoted` fields may reach durable history, Stripe, NemoClaw, phone
payloads, memory, files, external messages, or tool arguments.

Dedicated ASR remains useful, but only as fallback, captions, diagnostics,
literal checks, or degraded text-only compatibility. It should not block
reflex acknowledgement, create a second Hermes turn, schedule a second oracle
job, or satisfy high-risk action gates when raw audio is available.

Moshi-style STT output should therefore be treated as a witness, not a fourth
tier. The intended packet is not `Moshi transcript -> Hermes`; it is
`raw audio + Moshi witness -> Gemma interpreter -> promoted evidence -> Hermes
oracle`. This keeps Moshi useful for clipped prefixes, rough intent, names,
numbers, and code-switched words while preserving the raw-audio authority
boundary. A fast Moshi transcript can be displayed or attached immediately, but
Hermes should not let it become the user message until the interpreter accepts
or corrects it.

If a frontend can only provide the Moshi text and no waveform, the session is
explicitly degraded. The text-only path may hold the floor, ask for
confirmation, or preserve audit context, but it must fail closed for
Stripe/NemoClaw/phone/file/memory/message actions and must not be counted as a
full KAME demonstration.

Canonical rule: every provider is assigned to a role before it is trusted.
Open S2S systems such as Moshi, VoiceClaw, OpenClaw Talk, Ultravox-like
frontends, or hosted realtime APIs may be reflexes, witness producers,
interpreters, TTS providers, or degraded fallbacks. They are not allowed to
silently become the Hermes oracle, the transcript of record, or an action
authority source.

2026-07-05 authority amendment: external realtime frontends are consult
bridges, not Hermes tool runners. `kame_session_v1` may accept only
`ask_brain`, `ask_hermes_oracle`, `agent_consult`, or
`openclaw_agent_consult` as bridge tools. Direct requests for Stripe, NemoClaw,
phone, file, shell, memory, message, credential, or provider-provisioning tools
must be rejected with an auditable `tool.result` and must not create an oracle
job. A frontend can say what it heard, what it already told the user, and what
it wants Hermes to consider; it cannot spend, call, write, provision, or mutate
state by bypassing the active Hermes oracle.

This is the same boundary as the Moshi witness rule. Moshi/Open-S2S text can
help the interpreter understand the raw waveform, but both the text and the
frontend's requested action remain hypothesis/context until Gemma or the
Hermes oracle promotes them. The allowed bridge tools carry evidence and
consultation envelopes only; promoted action still flows through normal Hermes
tool routing, approvals, Stripe/NemoClaw safety, and durable audit.

Canonical current design: three-tier sensor fan-in, not STT-first. The reflex
is the always-warm live interface; Gemma is the post-cut audio interpreter; the
Hermes active `/model` is the oracle. The normal unit of work is one
energy-gated speech cut with raw audio as primary evidence. Moshi/OpenClaw/
VoiceClaw transcript-like text and classic ASR text may accompany that cut, but
only as labeled witness hypotheses inside the same interpreter bundle. They do
not schedule a second Hermes turn, become durable user text, authorize tools,
or carry spend/call/file/memory/message authority before raw-audio-grounded
interpreter promotion.

Authoritative architecture statement: do not implement KAME as a parallel ASR
stack. There are three roles:

1. Reflex: fast live floor control, barge-in, acknowledgement, provisional
   route, and optional transcript-looking witness text.
2. Interpreter: Gemma-style direct-audio adjudication over the clipped waveform,
   timing/speaker metadata, reflex state, and every transcript hypothesis.
3. Oracle: Hermes' active `/model`, selected through the existing model
   interface, with authority over tools, memory, files, spend, calls, approvals,
   and durable outcomes.

Moshi/Open-S2S text can be extremely useful, but only as `transcript_hypotheses[]`
attached to the accepted raw-audio cut. Gemma may use it to recover clipped
prefixes, names, numbers, or code-switched words; Gemma may also reject it when
energy, speaker, channel, timing, or waveform evidence disagrees. The first text
string to arrive is never the transcript of record.

Implementation naming rule: "Moshi STT" is adapter-edge shorthand only, and
"Gemma ASR" is the wrong abstraction for the normal path. Internal packet
fields, evidence artifacts, prompts, tests, and demo copy should describe
Moshi/Open-S2S/classic-ASR strings as witness hypotheses and Gemma output as
`interpreter_promoted` direct-audio interpretation.

Current operator decision: when a Moshi/OpenClaw/VoiceClaw-style frontend can
expose both the clipped waveform and an STT-like string for the same speech cut,
send both to Gemma in one interpreter evidence bundle. The raw waveform goes
first. The transcript-looking text is `frontend_witness_hypothesis` unless the
adapter can prove a narrower source label. It is context for direct-audio
interpretation, not a parallel ASR turn, not a scheduler, not a fourth agent,
and not the user's durable transcript until Gemma promotes raw-audio-grounded
evidence for the active Hermes oracle.

Compatibility decision: "Moshi STT" is acceptable language only at the adapter
edge. Inside Hermes it must be normalized as witness evidence. The useful shape
is not reflex plus ASR plus oracle; it is reflex, interpreter, and oracle, with
Moshi/open-S2S/classic-ASR text attached to the interpreter packet as one of the
sensor readings for the accepted raw-audio cut. This lets Gemma use a fast
frontend's best guess without letting the first transcript-looking string win
the turn.

The endpointer/noise gate is part of the contract, not an implementation
detail. The reflex should maintain an adaptive energy floor, ignore silence and
low-energy non-speech packets for barge-in and cut creation, and hand the
interpreter one bounded segment per user speech cut. This is what makes a small
direct-audio interpreter viable: Gemma should receive intentional speech plus
timing metadata, not every Discord packet, room tone, or harmonic artifact.

This also means the runtime should not block reflex acknowledgement waiting for
Moshi/STT text. If the waveform and reflex route are available, create the
bundle and let the interpreter start. Transcript hypotheses can attach before
the cut, with the cut, or later as late evidence, but only raw-audio-grounded
interpreter evidence can promote them into action text. If raw audio is absent,
the request is degraded text-only compatibility mode and cannot satisfy full
KAME readiness or high-risk action gates.

2026-07-03 amendment: the design is now explicitly evidence-bundle KAME. The
fast reflex may be Moshi/open-S2S-like and may produce an STT-looking text
string, but that text is not "the transcript" for Hermes. It is a hypothesis
attached to the same interpreter bundle as the clipped waveform, reflex route,
spoken acknowledgement, timing metadata, speaker metadata, and optional classic
ASR output. Gemma-style raw-audio interpretation is the promotion step for
durable wording. Hermes' active `/model` remains the only oracle and action
brain.

2026-07-03 operating decision: do not require ASR evidence in the normal path.
The runtime should collect Moshi/open-S2S or classic ASR text when available,
but only as transcript hypotheses attached to the same `turn_id` and
`audio_segment_ref` as the raw voice. The interpreter receives the raw waveform
and those hypotheses together; it decides which wording, if any, is promoted for
the oracle. This keeps the fast reflex from waiting on STT while still giving
Gemma useful hints about names, numbers, clipped prefixes, code-switching, and
hallucinated commands.

2026-07-03 Moshi-context amendment: a Moshi "STT" string is acceptable and
useful only as context for the interpreter. Treat it as what the live
frontend/reflex believed it heard, not as the user message. The normal packet is
still raw voice first: clipped waveform, timing, speaker/channel metadata,
reflex route, spoken acknowledgement, then transcript hypotheses including
Moshi/open-S2S text when available. Gemma may use that text to improve
multilingual interpretation or recover clipped prefixes, but it must be able to
reject the text when the waveform, energy gate, speaker identity, or later
context disagrees.

2026-07-03 sensor-fan-in amendment: the intended shape is not "keep STT in
parallel" as an old control path. It is one raw-audio turn with multiple
sensors attached. The reflex is the realtime sensor and floor controller;
Moshi/open-S2S text is a hypothesis about what that live sensor believed it
heard; classic ASR is optional diagnostic or fallback evidence; and Gemma is the
direct-audio interpreter that compares those signals before any durable wording
or tool-critical argument is promoted. When raw audio and Moshi text are both
available, send both to the interpreter in one bundle.

2026-07-03 implementation contract: do not build a second STT conversation beside
KAME. The only normal merge point is the interpreter evidence bundle keyed by
`turn_id` and `audio_segment_ref`. A Moshi/OpenClaw/VoiceClaw transcript-like
string may be useful context because it tells Gemma what the realtime frontend
believed it heard, but it is not a prerequisite, not a scheduler, not a user
message, and not a durable transcript. Dedicated ASR is fallback, diagnostic,
caption, or high-risk literal-evidence support only. When raw audio is available,
the interpreter request should include the waveform first and every transcript
string only as a provenance-labeled hypothesis.

Implementation note: `evidence_bundle_id` is stable for the speech cut even when
raw audio arrives late. The audio-aware join proof is `evidence_merge_key`, a
separate hash over the session, turn, and `audio_segment_ref`.

2026-07-03 witness-context amendment: Moshi/OpenClaw/VoiceClaw transcript text
is best understood as witness testimony from the realtime frontend. It should
be sent to Gemma beside the clipped waveform, not hidden and not promoted. The
interpreter prompt should explicitly ask Gemma to compare the witness text
against raw audio, VAD/energy timing, speaker identity, the reflex route, and
the acknowledgement already spoken. If the witness text helps recover a clipped
prefix, name, number, code-switch, or command, Gemma can promote corrected
wording. If it conflicts with the waveform or appears hallucinated, Gemma must
leave it as diagnostic evidence or reject it.

2026-07-03 action-authority amendment: high-risk VoiceOps actions must not
inherit authority from any transcript hypothesis. Stripe spend reasons,
provider selections, NemoClaw action packets, phone-call payloads, memory writes,
file writes, and external messages require `interpreter_promoted` or
`oracle_promoted` evidence fields. The artifact should show the raw-audio
bundle, any witness transcript hypotheses, the promotion source, and the policy
decision that allowed the action. If only transcript hypotheses are available,
Hermes may ask a clarification or prepare a draft, but it must not execute,
approve, or claim readiness for irreversible work.

2026-07-05 NemoClaw contract amendment: a NemoClaw approval packet is not valid
because it is internally self-consistent. It must match the VoiceOps action
contract: known action id, expected provider, allowlisted command shape,
required preflight gates, expected approval artifact, promoted KAME evidence,
and tool-disclosure reference. Unknown actions, shell-like substitutions, or
swapped provider/gate combinations must fail before operator approval is even
requested.

2026-07-05 setup-evidence provenance amendment: redacted Milestone 2 setup
evidence is accepted only when the section and its referenced source artifact
both identify themselves as `redacted_setup_evidence` and the section hash,
source artifact hash, and collector attestation agree. The manifest path is
metadata, not authority; the source artifact must be self-describing so a copied
or renamed JSON file cannot satisfy setup proof by position alone.

2026-07-05 channel route-payload amendment: channel approval routes must declare
what kind of payload they can carry before any operator review can approve live
egress. Each route must include a payload policy, allowed payload classes,
`payload_digest_required = true`, and `raw_witness_text_allowed = false`.
Customer-visible routes may carry only redacted approved payload classes.
Phone-handoff routes may carry references and summaries, not raw phone numbers
or transcript text. Spend, provisioning, credential, and account-mutation routes
must keep `outbound_payload_allowed = false` and may emit only blocked-intent or
operator-escalation evidence.

2026-07-05 live witness-metadata amendment: live KAME evidence cannot merely say
that transcript hypotheses were labeled. Each `transcript_hypotheses[]` item
must carry `text_digest`, `role = "witness_context"`,
`authority = "hypothesis"`, `promotion_required =
"interpreter_promoted_or_oracle_promoted"`, `tool_authority = false`,
`arrival_phase`, `latency_ms`, `confidence`, `speaker_or_actor_ref`, and
`channel_or_surface_ref`. This lets live evidence bind frontend/Moshi/Open-S2S
witness text to a specific speaker, channel, timing window, and digest without
turning raw witness text into durable user text or action authority. The
top-level `interpreter_adjudication_outcomes` set must exactly cover the
per-hypothesis adjudications; a rejected witness row cannot be summarized as an
accepted interpreter outcome.

2026-07-04 frontend-witness amendment: when a Moshi/open-S2S frontend exposes
an "STT" string, Hermes should classify it as a frontend witness transcript, not
as a dedicated ASR result. The preferred packet is raw audio plus witness text in
one interpreter request. The witness text gives Gemma a compact view of what
the realtime interface believed it heard, while the waveform remains the
primary evidence. This does not add a fourth agent, does not require classic
ASR, and does not allow transcript-only scheduling when raw audio is available.

2026-07-04 three-tier amendment: the field-aligned shape is reflex, interpreter,
oracle. The reflex may be Moshi-like, VoiceClaw-like, or a simpler local
floor-control model. It is allowed to emit a transcript-looking witness, but
that witness is context for the interpreter, not the interpreter itself. Gemma
receives the clipped waveform, VAD/energy timing, reflex route, spoken
acknowledgement, and all witness transcript hypotheses in one bundle, then
promotes or rejects wording for the active Hermes oracle. If the raw waveform is
missing, the turn is degraded compatibility mode, not full KAME.

2026-07-04 operator decision: the next implementation target is a three-tier
sensor-fan-in contract, not a parallel STT pipeline. The fast reflex answers the
floor, the Gemma interpreter judges the evidence bundle, and Hermes' active
`/model` remains the only oracle. Moshi/open-S2S transcript text is valuable
because it captures what the live frontend believed it heard, so it should be
included in the interpreter request beside the raw voice. It is still witness
context: it cannot schedule a separate Hermes turn, cannot become durable chat
history, and cannot authorize tools, spend, calls, memory, files, or messages
unless Gemma promotes raw-audio-grounded evidence for the active oracle.

2026-07-04 protocol decision: the transport-neutral packet is
`kame_session_v1`. A Moshi/open-S2S witness can arrive before the accepted cut,
with the accepted cut, or after the interpreter request has started, but all
three timing cases must merge into the same `turn_id`, `audio_segment_ref`,
`evidence_bundle_id`, and `evidence_merge_key`. The active interpreter context
contains at most one same-source, same-kind final hypothesis for a speech cut;
partials are retained only as superseded provenance. The acceptance proof must
show witness-before-cut, witness-with-cut, and witness-after-cut cases without
duplicate oracle jobs or durable user turns.

2026-07-04 same-turn convergence amendment: the strongest Moshi/Open-S2S shape
is raw voice plus witness text in one interpreter bundle, but adapters are
allowed to discover those pieces in different orders. If a text-only reflex or
frontend packet creates provisional queue state before the raw waveform is
ready, the later raw-audio packet must update that same job, not submit a new
Hermes turn. The runtime proof is one oracle job lifecycle for the speech cut,
one `INTERFACE_ORACLE_REQUEST` durable record, and an
`INTERFACE_ORACLE_UPDATE` carrying the merged raw-audio/witness evidence. This
keeps Moshi-style "STT" useful as context without letting it race the waveform
or fork the oracle.

2026-07-04 partial-supersession amendment: active partial hypotheses are not
valid completed evidence. A partial such as a clipped "hey" may appear in
streaming UI, telemetry, or latency traces while the cut is unstable, but it
must be replaced in interpreter context when the final same-source/same-kind
hypothesis arrives for the same speech cut. The final hypothesis should retain
`superseded_partial_texts`/`superseded_partial_count` for auditability. VoiceOps
and alpha evidence should reject completed packets that still expose active
partials.

2026-07-04 Moshi raw-voice join decision: yes, a Moshi/Open-S2S transcript may
and should be provided to the interpreter alongside the raw voice when both
refer to the same accepted speech cut. That transcript is not "STT evidence" in
the old Hermes sense. It is a `frontend_witness_hypothesis`: a compact report
of what the live voice frontend believed it heard. The Gemma interpreter should
receive the waveform, timing, speaker/channel metadata, reflex route,
acknowledgement already spoken, and the Moshi witness in one packet. It may use
the witness to recover clipped starts, names, numbers, or code-switched terms,
but only its promoted fields may reach durable history or the active Hermes
oracle as user wording.

2026-07-04 implementation clarification: this does not mean "run Gemma ASR in
parallel with Moshi." It means run one accepted speech cut through one
direct-audio interpreter packet. The packet carries the waveform first, then
metadata, then the reflex route/acknowledgement, then optional transcript
hypotheses. Moshi, VoiceClaw/OpenClaw, classic ASR, or hosted realtime text can
all be helpful witnesses, but they are never the prompt head, scheduler,
durable transcript, or action authority. The interpreter output must keep two
things separate: witness adjudication (`accepted_as_supporting_evidence`,
`corrected_by_audio`, or `rejected_or_diagnostic_only`) and promoted fields
(`interpreter_corrected_transcript`, normalized intent, entities, confidence)
that may be handed to Hermes' active `/model`.

2026-07-04 consolidation: the three-tier design is reflex, interpreter, oracle,
not reflex plus two competing transcript systems. The reflex can be extremely
small and latency-biased; it only needs to notice speech state, stop playback,
acknowledge, and form a provisional route. Gemma's job is multilingual
direct-audio interpretation of the accepted cut, including corrected transcript
output when needed. Moshi/Open-S2S "STT" should be supplied to Gemma as witness
context beside the waveform, never as a sibling control path. Classic ASR is
kept for fallback, diagnostics, captions, or high-risk literal checks. If the
interpreter produces transcript text, that text is `interpreter_promoted`; if a
frontend produces transcript text, it is only a hypothesis until Gemma or the
oracle promotes it.

2026-07-04 artifact amendment: full-KAME evidence must prove the negative
case, not only the promoted happy path. A live turn that includes Moshi,
VoiceClaw/OpenClaw, reflex, or classic-ASR witness text should expose
`unpromoted_witness_sink_checks` with `spend_clean`, `phone_clean`,
`nemoclaw_clean`, `tool_clean`, `memory_clean`, `file_clean`, `message_clean`,
and `durable_history_clean`, plus an empty `unpromoted_witness_sink_values`
object. This is the headless proof that witness text stayed context for Gemma
and did not leak into Stripe spend reasons, provider choices, NemoClaw packets,
phone scripts, tool arguments, memory/file writes, external messages, or
durable Hermes history before interpreter/oracle promotion.

| Tier | Primary input | Immediate output | Authority boundary |
| --- | --- | --- | --- |
| Reflex | live audio, VAD/energy, current session state | acknowledgement, barge-in, route, rough intent, optional witness transcript | floor control only; provisional `reflex_hypothesis` |
| Interpreter | clipped raw audio plus reflex/Moshi/OpenClaw/VoiceClaw/classic-ASR hypotheses | corrected transcript candidate, entities, language notes, disagreement flags, oracle request patch | first promotion point: `interpreter_promoted` |
| Oracle | promoted request plus labeled evidence bundle, using Hermes' active `/model` | tools, memory, files, spend/provisioning plans, calls, durable outcome | action authority: `oracle_promoted` after policy checks |

The practical prompt shape for Gemma should be explicit: "Audio is primary.
Witness transcripts describe what frontend sensors believed they heard. Use
them as clues for clipped starts, names, numbers, code-switching, and noisy
audio, but reject or downgrade them when they conflict with the waveform,
speaker identity, VAD/energy timing, or conversation state." This instruction
belongs in the interpreter prompt. The oracle prompt should receive promoted
wording and labeled evidence, not raw witness text masquerading as verified
user speech.

The normalized interpreter request should also make this trust order explicit
with `interpreter_input_order = ["raw_audio", "metadata", "reflex", "transcript_hypotheses"]`.
If raw audio is missing, the order and mode must say so and the turn is
degraded compatibility. That degraded path is useful for bring-up and captions,
but it must not satisfy full KAME, Stripe, NemoClaw, phone, memory, file,
external-message, or durable-history gates.

2026-07-04 confirmed packet contract: when a Moshi/OpenClaw/VoiceClaw-style
frontend can provide both waveform and transcript-looking text, Hermes should
send both to the interpreter in one request. The transcript-looking text is
`frontend_witness_hypothesis` unless the adapter can prove a narrower source
label. The interpreter packet must carry:

- one `turn_id`
- one `audio_segment_ref` for the clipped waveform
- VAD/energy timing and speaker/channel metadata
- reflex route and acknowledgement already spoken
- zero or more transcript hypotheses with source, timing, confidence when
  available, partial/final state, `authority = "hypothesis"`, and
  `tool_authority = false`

For Moshi specifically, the adapter should preserve the vendor transcript text
as `source = "moshi"` plus `kind = "frontend_witness_hypothesis"` and attach
the frontend's timing, confidence, speaker/channel guess, and partial/final
state when available. The same raw-audio `turn_id` and `evidence_merge_key`
must be used for the waveform and the Moshi witness so the interpreter can
compare them as one speech cut. A Moshi transcript without a matching waveform
is degraded text-only evidence, not direct-audio KAME evidence.

Queue/status surfaces may expose a compact `provisional_request_summary` so the
reflex can narrate what it is asking the oracle to do, but that summary is not
the user's verified transcript. It must carry `authority = "reflex_hypothesis"`
and `tool_authority = false`, and it cannot satisfy spend, call, tool, memory,
file, or external-message action gates before raw-audio-grounded interpreter
promotion.

Partial witness hypotheses are active only until a newer same-source,
same-kind witness supersedes them. When Moshi, OpenClaw, VoiceClaw, or classic
ASR emits a partial such as "what is three to the" and later emits the final
"what is three to the power of seventeen" for the same speech cut, the active
interpreter bundle should contain the final witness only. The partial text is
kept as `superseded_partial_texts`/`superseded_partial_count` provenance on
the retained final hypothesis, not as a second active candidate and not as
durable user text.

This packet is the only normal merge point. It is valid for the reflex to use a
witness transcript for local narration, but invalid for that transcript to
create a second Hermes user turn, patch `oracle_text`, or satisfy any action
approval before raw-audio-grounded interpreter promotion. When raw audio is
missing, the same adapter may still submit a degraded compatibility request, but
the request must be labeled text-only, cannot count as full KAME readiness, and
cannot satisfy high-risk action gates through oracle-only promotion.

2026-07-04 interpreter-context decision: if a Moshi/OpenClaw/VoiceClaw-style
frontend can emit transcript-like text for the same speech cut as the raw
audio, send that text to the Gemma interpreter as context in
`transcript_hypotheses[]`. Do not run a separate STT-first turn and do not wait
for that text before acknowledging the user. The interpreter prompt should see
the waveform first, then speaker/channel/VAD metadata, then the reflex route and
spoken acknowledgement, then witness transcript hypotheses. This ordering is
intentional: it lets Gemma use the witness as a clue while keeping raw voice as
the higher-authority signal.

The prompt and schema should make three outcomes explicit for every witness
transcript:

- `accepted_as_supporting_evidence`: the witness agrees with the waveform and
  helps recover wording.
- `corrected_by_audio`: the witness was useful but incomplete or slightly
  wrong.
- `rejected_or_diagnostic_only`: the witness appears clipped, hallucinated,
  stale, wrong-speaker, or contradicted by the waveform.

Only the first two outcomes may contribute to `interpreter_promoted` wording,
and only after the interpreter emits the promoted fields. A rejected or
diagnostic witness remains visible in the audit bundle but cannot become
durable user text, a phone script, a spend reason, a provider choice, a memory
write, a file write, or a tool argument.
The positive acceptance proof must be concrete: if the reflex heard only "three
to the power of seventeen" and an early frontend witness heard "what is three to
the power of seventeen," the durable wording still comes from Gemma's promoted
`interpreter_corrected_transcript`, not from the witness. The same bundle should
also preserve `interpreter_normalized_intent`, entities such as `3^17`, and the
fact that the witness remains hypothesis authority.
Rejected frontend witnesses should also carry typed `rejection_reasons` such as
`ambiguous_speaker`, `wrong_speaker`, `wrong_channel`, or `stale_witness` when
speaker/channel/timing metadata proves the conflict. These reasons are audit
evidence, not prompts for the oracle to reinterpret the rejected text.

## Current Implementation Target

The next implementation pass should optimize for a fast floor response while
keeping the evidence boundary strict:

1. The reflex listens continuously through VAD/energy gating, handles barge-in,
   and speaks a short acknowledgement without waiting for classic ASR.
2. The speech cut creates exactly one interpreter evidence bundle keyed by
   `turn_id` and `audio_segment_ref`.
3. The bundle includes raw audio first, then speaker/channel/VAD timing, then
   reflex route and acknowledgement, then Moshi/OpenClaw/VoiceClaw/classic-ASR
   transcript hypotheses with provenance labels.
4. Each transcript hypothesis carries arrival phase when known:
   `before_raw_audio`, `with_raw_audio`, or `after_interpreter_start`. The
   same phase list must remain visible on oracle job status, bounded job
   updates, live-evidence reports, and package audits.
5. Gemma interprets that bundle and emits promoted wording, entities,
   confidence, disagreements, witness adjudication, and an oracle request patch.
6. Hermes sends only promoted wording and labeled evidence to the active
   `/model` oracle for durable work and external effects.

This keeps a three-tier user experience without creating three competing
conversations. The user hears the reflex quickly. Gemma gets both the waveform
and the Moshi/open-S2S witness text. The oracle receives compact, promoted
business intent rather than raw partials or duplicate STT turns.

Moshi-style transcript output should therefore be implemented as a witness
attachment API, not as an ASR replacement API. The adapter may pass the Moshi
string to Gemma, and should do so when it belongs to the same accepted speech
cut, but it must arrive under `transcript_hypotheses[]` with hypothesis
authority. The interpreter packet should make the comparison explicit: raw
audio first, then timing and speaker/channel metadata, then the reflex route and
acknowledgement, then the Moshi/open-S2S witness. A helpful Moshi string can
improve the promoted wording, but the durable wording is still the interpreter
promotion result.

The acceptance fixture for this path should include a positive and negative
case. Positive: raw audio is clipped, Moshi supplies a same-turn witness, Gemma
accepts or corrects it, and the promoted fields reach the active Hermes
`/model`. Negative: Moshi supplies a hallucinated, ambiguous-speaker,
wrong-speaker, stale, or energy-inconsistent string; Gemma rejects it as
diagnostic evidence; no spend, phone, file, memory, external-message, tool, or
durable-history sink receives that unpromoted text.

The fixture should also include all three witness arrival phases. A
`before_raw_audio` witness proves the pending bundle can hold context before
the accepted cut. A `with_raw_audio` witness proves inline context reaches the
interpreter in the intended order. An `after_interpreter_start` witness proves
late evidence patches the existing oracle job rather than creating another
turn. These are runtime lifecycle tests, not model-quality tests.

## Provider Role Matrix

Provider and model selection must be described by role, not by brand. This is
the implementation map for open S2S, STT/TTS, and hosted realtime alternatives:

| Role | Valid providers | Contract |
| --- | --- | --- |
| Reflex / floor control | Moshi/PersonaPlex-class S2S, VoiceClaw/OpenClaw-style frontend, small local timing model, hosted realtime fallback | Owns VAD-adjacent turn-taking, barge-in, acknowledgement, and provisional route. May emit witness text, but has no tool authority. |
| Interpreter / evidence | Gemma 4 audio-multimodal family, or another direct-audio model that can accept bounded clips | Receives raw audio first and transcript hypotheses second. Emits promoted transcript, intent, entities, confidence, and disagreement flags. |
| Optional witness/fallback transcript evidence | Moshi/OpenClaw/VoiceClaw transcript-looking output, classic ASR, Nemotron/Riva ASR, cloud ASR fallback | Captions and clues only. Must stay `authority = "hypothesis"` until accepted or corrected by the interpreter. |
| Oracle / brain | Hermes active `/model` only | Owns tools, memory, files, approvals, spend, phone calls, and durable outcomes. Voice config must not add `oracle_model`. |
| Outbound TTS | Magpie/Riva, Piper-class local TTS, Cartesia/cloud fallback, hosted realtime voice output | Speaks approved text. TTS choice does not change transcript authority or oracle model. |
| Degraded compatibility | text-only VoiceClaw/OpenClaw/Moshi bridge, text-only hosted realtime callback | May draft, clarify, or produce low-risk status. Cannot satisfy full KAME, Stripe, NemoClaw, phone, memory, file, or message gates. |

This matrix is intentionally conservative. A model may be excellent at several
tasks, but a single request must still name which role it is playing. For
example, a Moshi transcript may be a strong witness for Gemma, but it is still
not the user's durable message. A hosted realtime API may be a useful bring-up
reflex or TTS path, but it does not replace Hermes' active `/model` as the
oracle.

## Non-Negotiable Implementation Rules

- Do not run a separate STT-first Hermes conversation when raw audio is
  available.
- Do not call Gemma "ASR" in code, docs, or demo copy. Its role is interpreter
  or evidence adjudicator.
- Do not call Moshi/OpenClaw/VoiceClaw transcript-looking output "the
  transcript" unless the authority label is visible and non-authoritative.
- Do not create a VoiceOps-specific `oracle_model`; use Hermes `/model`.
- Do not let text-only frontend evidence close full-KAME, Stripe, NemoClaw,
  phone, memory, file, external-message, or durable-history readiness gates.
- Do not let a witness transcript schedule a second oracle job for the same
  speech cut.
- Do not treat acknowledgement latency and oracle latency as the same metric.
  Reflex acknowledgement is allowed to be fast while the oracle continues in
  the background.

## Required Artifact Proof

Every headless or recorded proof of this design should show the same facts:

- one `turn_id`, one `evidence_bundle_id`, and one `evidence_merge_key` for the
  speech cut
- accepted `audio_segment_ref`, time range, VAD result, and energy/noise-gate
  decision
- reflex acknowledgement and provisional route
- transcript hypotheses with `kind`, `source`, timing/confidence when
  available, partial/final state, `authority = "hypothesis"`, and
  `tool_authority = false`
- `witness_arrival_phase` for before-cut, with-cut, and after-interpreter-start
  evidence timing
- `raw_audio_interpreter_evidence_observed = true` for full-KAME acceptance
- interpreter adjudication for each witness:
  `accepted_as_supporting_evidence`, `corrected_by_audio`, or
  `rejected_or_diagnostic_only`
- `transcript_only_witness_rejected_for_full_kame = true` when raw audio is
  missing
- no duplicate oracle job or durable user turn from transcript hypotheses alone
- promoted `interpreter_corrected_transcript`, normalized intent, entities, and
  confidence before those fields reach the active Hermes oracle
- sink-specific checks proving unpromoted witness text did not enter Stripe
  spend reasons, provider selections, NemoClaw action packets, phone payloads,
  tool arguments, memory writes, file writes, external messages, or durable
  user history
- explicit degraded-mode labeling when raw audio is missing

## Three-Tier Sensor Fan-In Contract

The latest design decision is to keep the architecture three-tier while
avoiding a separate STT-first control path:

1. **Reflex**: always warm, shortest-latency floor controller. It detects
   speech, handles barge-in, acknowledges quickly, narrates what it is asking
   Hermes to do, and may emit a rough transcript-looking witness.
2. **Interpreter**: Gemma-style direct-audio model. It receives the clipped raw
   waveform after the energy/VAD gate, then compares that audio with reflex,
   Moshi/OpenClaw/VoiceClaw, and classic-ASR hypotheses attached to the same
   cut.
3. **Oracle**: Hermes' active `/model`. It receives promoted wording, intent,
   entities, and compact labeled evidence; it owns tools, memory, files,
   approvals, spending, phone calls, and durable outcomes.

Moshi/open-S2S transcript text is therefore useful, but only as context for
the interpreter. It should be stored as `frontend_witness_hypothesis` unless an
adapter can prove a narrower label. The packet must preserve the source name
such as `moshi`, timing, confidence when available, partial/final state, and
speaker/channel guess when available. It must also keep
`authority = "hypothesis"` and `tool_authority = false`.

The presence of a Moshi transcript must never make Hermes skip the raw-audio
interpreter. When raw audio is available, the transcript-looking text is sent
beside the waveform in the same interpreter request. When raw audio is missing,
the request is degraded compatibility mode and cannot prove full KAME behavior
or satisfy high-risk VoiceOps action gates.

Classic ASR remains valuable for captions, diagnostics, literal spelling,
names, numbers, code-switching checks, and degraded fallback. It is not the
normal scheduler and should not block acknowledgement, cut creation, or
raw-audio interpretation. The first transcript to arrive is latency evidence,
not authority.

Acceptance artifacts for this contract must show:

- the accepted cut's `audio_segment_ref`, time range, VAD decision, and
  energy/noise-gate decision
- raw audio, reflex route, acknowledgement, and witness hypotheses sharing one
  `turn_id`, one `evidence_bundle_id`, and one `evidence_merge_key`
- no duplicate oracle job when a witness arrives early, inline, or late
- positive witness use, where Gemma accepts or corrects witness text after
  comparing it to audio
- adversarial witness rejection, where Gemma rejects stale, hallucinated,
  ambiguous-speaker, wrong-speaker, wrong-channel, or energy-inconsistent text
  without allowing it to become durable history or an action argument

## Purpose

Hermes currently has KAME-compatible realtime voice plumbing: Discord voice
transport, a realtime sidecar, streaming speech provider bridges, barge-in
handling, mixer playback, latency metrics, and the early interface/oracle
session boundary. Full KAME production readiness still requires proving the
live runtime shape end to end: a fast reflex in the user-facing loop, a
Gemma-style direct-audio interpreter over accepted speech cuts, Hermes' active
`/model` as the oracle, and evidence that transcript-looking side channels stay
non-authoritative until promoted.

Full KAME-style means:

1. The human speaks to a fast interface model, also called the reflex.
2. The reflex owns the realtime conversation loop: listening, interruption,
   acknowledgements, short local replies, floor control, and rough transcript
   hypotheses.
3. A non-blocking interpreter stage, preferably Gemma 4 audio-multimodal,
   reviews clipped raw audio plus any reflex/Moshi transcript hypotheses to
   produce corrected multilingual evidence, entities, confidence, and oracle
   request patches. The raw audio remains the primary input; transcript
   hypotheses are supporting context.
4. Hermes's oracle remains the brain: tools, memory, files, long reasoning,
   project context, and durable task execution.
5. The reflex and interpreter broker compact requests to the oracle instead of
   forcing every spoken fragment through the full Hermes context.

The goal is a voice system that feels immediate while preserving Hermes's existing agent capabilities.

## Current Architecture Decision

The current target is a three-tier system with sensor fan-in:

1. **Reflex:** a very fast live-audio interface that owns floor control,
   immediate acknowledgement, barge-in, and short narration of what it is asking
   Hermes to do.
2. **Interpreter:** a Gemma-style audio-multimodal evidence adjudicator that
   receives the clipped raw voice plus any transcript-like witness text from the
   reflex, Moshi/open-S2S, VoiceClaw/OpenClaw, or classic ASR.
3. **Oracle:** Hermes' active `/model`, unchanged by voice config, which owns
   tools, memory, spend, provisioning, calls, files, and durable outcomes.

This is not a parallel STT conversation. Moshi/open-S2S transcript output can be
extremely useful, but it is witness context: "what the realtime frontend believed
it heard." It should travel beside the raw waveform in the same interpreter
bundle. It should not create a second Hermes turn, block the acknowledgement,
drive the scheduler, or become durable user wording before Gemma/interpreter or
Hermes/oracle promotion.

Classic ASR has the same authority class in full KAME mode. It remains useful for
fallback, captions, diagnostics, and high-risk literal-evidence checks, but it is
not required before the reflex acknowledges or before a raw-audio interpreter
request starts. If the system lacks raw audio and has only text, the turn is
degraded compatibility mode.

## Three-Tier Voice Contract

The intended runtime is three-tier, not STT-first and not a single direct-audio
model pretending to own every job:

1. **Reflex:** the always-warm realtime interface. It listens to live audio,
   controls the floor, handles barge-in, speaks immediate acknowledgements, and
   emits rough transcript hypotheses when the chosen S2S/reflex model can do
   so. Moshi/PersonaPlex-class models belong here when their audio path is
   stable enough.
2. **Interpreter:** the audio evidence adjudicator. Gemma 4 E2B/E4B/12B-style
   multimodal models belong here: they receive the clipped raw audio segment
   plus labeled frontend-witness, reflex, S2S, or classic-ASR hypotheses, then
   produce corrected multilingual evidence, transcript candidates, and oracle
   request patches. This is the layer that can behave like multilingual
   transcript adjudication for Hermes, but it is not the live endpointer and
   not a required ASR proof. Raw audio is the primary signal; transcripts are
   context, not authority.
3. **Oracle:** Hermes' active model, selected through the normal `/model`
   interface. It owns tools, memory, approvals, files, spend, provisioning,
   phone calls, and durable business logic. Voice config must not introduce a
   separate `oracle_model` setting.

This split is what makes the system KAME-style. A direct Gemma audio request can
help the interpreter, and a legacy cloud speech/TTS bridge can prove transport
or degraded fallback behavior, but neither is the full reflex/oracle
architecture by itself. A transcript
producer, even one embedded in the reflex S2S model, is a sensor feeding the
bundle; it is not an extra tier with authority.

The current design choice is deliberately not "Gemma as reflex" and not "ASR in
front of the reflex." The reflex must be the fastest reliable live-audio loop we
can run, even if it only produces a route, acknowledgement, and rough hypothesis.
Gemma is the interpreter after the cut: it receives the waveform and any
transcript-like side channels, then decides what wording is safe to offer to the
oracle. This keeps the voice loop immediate without pretending an early
transcript is ground truth.

The practical shape is therefore allowed to look "parallel" only inside one
evidence bundle. A Moshi/open-S2S frontend can speak quickly and emit a rough
transcript. A classic ASR fallback can produce literal text for captions,
diagnostics, or high-risk literal checks. Gemma can consume both, plus the raw
audio, as context. Only the interpreter promotion result may become
durable user text, Stripe/NemoClaw spend rationale, phone-call payload, memory,
file content, or tool argument. When raw audio is available, that promotion
must be grounded in the interpreter's direct-audio judgment, not in whichever
transcript hypothesis arrived first.

This is the important distinction for Moshi-style output: the transcript is not
the thing Hermes acts on. It is a compact observation from the live interface,
useful precisely because it tells Gemma what the realtime voice model thought it
heard at the moment it chose an acknowledgement or route. That observation may
help recover clipped starts, names, numbers, or code-switched phrases, but it
can also be wrong or hallucinated. The interpreter must see it beside the raw
audio, not instead of the raw audio.

In implementation terms, the packet should look like one raw-audio turn with
multiple evidence attachments, not multiple turns competing for authority:

```text
speech cut
  -> raw_audio: primary interpreter evidence
  -> reflex: route, acknowledgement, provisional intent
  -> transcript_hypotheses[]: Moshi/open-S2S/classic-ASR clues
  -> Gemma interpreter: corrected evidence and oracle request patch
  -> Hermes active /model: action, tools, memory, spend, calls
```

If the Moshi text arrives first, it can help the reflex narrate what it thinks
it is doing, but it still cannot bypass the interpreter merge point for durable
history or tool-critical arguments.

If the raw audio is unavailable, the session is no longer in full KAME mode for
that turn. A text-only Moshi/OpenClaw/VoiceClaw bridge can still submit an
`ask_brain` compatibility request, but the request must be marked degraded,
must preserve the source as `hypothesis` or `fallback_text`, and may only
draft, clarify, or produce low-risk status. High-risk tools, spend,
provisioning, files, memory writes, external messages, and calls fail closed on
text-only fallback evidence; oracle-only promotion of that text is not enough
to make it action authority.

## Signal Authority Rules

The realtime stack may produce several text-like artifacts for one spoken turn.
They are not equivalent. The interpreter request is the merge point, and every
input must keep provenance until a later layer promotes it.

| Signal | Producer | Used For | Authority |
| --- | --- | --- | --- |
| `raw_audio` | transport/session cut | interpreter evidence, replay/debug, disagreement checks | primary interpreter input |
| `reflex_intent` | live reflex | routing, immediate acknowledgement, provisional job envelope creation | provisional routing |
| `reflex_transcript_hypothesis` | live reflex | early clue for the Gemma interpreter, user-visible rough caption when desired | hypothesis only |
| `s2s_transcript_hypothesis` | named S2S witness producer bound to the same raw-audio cut | what that S2S witness believed it heard, kept inside `transcript_hypotheses[]` | hypothesis only |
| `frontend_witness_hypothesis` | Moshi/VoiceClaw/OpenClaw or any ambiguous S2S/reflex frontend exposing STT-like text | umbrella label for "what the frontend believed it heard" when the exact producer is ambiguous | hypothesis only |
| `classic_asr_hypothesis` | dedicated ASR fallback/evidence lane | literal wording comparison, captions, diagnostics | optional hypothesis |
| `interpreter_corrected_transcript` | Gemma-style interpreter | durable user request candidate and tool-critical wording | first promoted transcript |
| promoted oracle text / final result | Hermes active `/model` | tool use, memory, files, spend, calls, durable outcome | action authority after policy checks |

This allows a three-tier design without making STT the control path. Ambiguous
Moshi/OpenClaw/VoiceClaw transcript-looking text should default to
`frontend_witness_hypothesis`, with `reflex_transcript_hypothesis` and
`s2s_transcript_hypothesis` reserved for producers the adapter can identify
precisely. A narrower S2S label does not create a distinct transcript lane; it
only preserves provenance for one same-packet witness. These hypotheses are
valuable because they summarize what the live frontend believed it heard,
including dropped prefixes or code-switched phrases. They must travel beside
the raw audio into the Gemma interpreter, not replace the raw audio and not
become a separate oracle prompt. Classic ASR follows the same rule and is kept
primarily for fallback, diagnostics, and literal-evidence checks.

The important distinction is not "Moshi instead of ASR" versus "classic ASR".
Both are transcript-like side channels. They can be useful evidence, especially
when the raw audio contains names, numbers, code-switched phrases, or clipped
prefixes, but neither can certify what the user said. The interpreter owns that
promotion step.

If the reflex has enough signal to acknowledge or create a provisional
background job envelope, it should do so immediately. The envelope is status and
queueing state, not action authority. The interpreter can attach corrected
evidence before the job starts, or as a bounded late update before any oracle
work commits irreversible spend, provisioning, message, memory, file, or call
arguments.

The frontend transcript adapter should therefore be boring and strict:

- attach Moshi/open-S2S text to the current interpreter bundle by `turn_id` and
  `audio_segment_ref`
- label it as `authority = "hypothesis"` with source, timing, confidence, and
  partial/final state when available
- never schedule an oracle job from that text alone when raw audio exists
- never overwrite `oracle_text`, durable transcript, spend reason, call text,
  memory text, or file content without raw-audio-grounded interpreter promotion
- keep the original raw-audio reference available for replay, disagreement
  checks, and later audit

Moshi/VoiceClaw/OpenClaw transcript output is especially valuable because it
describes what the live interface model thought it heard at the moment it
decided how to respond. That makes it better than a generic caption for
debugging missed prefixes, code-switching, and hallucinated commands. It is
still only a hypothesis. The correct packet shape is raw audio plus
provenance-labeled hypotheses, not "pick whichever transcript arrived first."

When the frontend is Moshi-like and produces both speech and text, the text
should default to `frontend_witness_hypothesis` alongside the clipped waveform
unless the adapter can prove it came from the live reflex model itself or from
a distinct caption/S2S component. The interpreter may use it to notice that
the live frontend dropped "hey Hermes", misheard a name, or invented a command,
but the hypothesis must not become durable user text until the interpreter
promotes it. This is the practical way to use Moshi transcript-looking output:
it is evidence about what the frontend believed it heard, not a replacement for
raw audio and not a second user message.

## Interpreter Evidence Bundle Contract

Each speech cut creates one interpreter evidence bundle. This is the contract
between Discord, VoiceClaw/OpenClaw-style frontends, Moshi/open-S2S frontends,
classic ASR fallbacks, the Gemma interpreter, and the Hermes oracle job manager.
The bundle is keyed by `turn_id` and the raw-audio reference; every transcript
string is attached to that same bundle instead of becoming its own conversation.
Runtime status exposes both a stable `evidence_bundle_id` and an audio-aware
`evidence_merge_key` so late witness/audio attachment can be audited without
turning raw artifact paths or transcripts into durable identifiers.

The acknowledgement path is allowed to run ahead of transcript evidence. When
VAD/energy gating has accepted a raw audio cut and the reflex has a route, the
reflex may acknowledge the user and create the oracle-job envelope without
waiting for Moshi/open-S2S, reflex transcript, or classic-ASR hypotheses. Those
hypotheses attach to the same bundle when they arrive and can help the
interpreter correct or reject wording, but they are not the scheduler and they
must not create a second durable user turn.

Canonical shape:

```json
{
  "turn_id": "voice-turn-id",
  "session_id": "voice-session-id",
  "audio_segment_ref": "artifact-or-buffer-ref",
  "evidence_bundle_id": "kame-bundle-id",
  "evidence_merge_key": "kame-merge-session-turn-audio",
  "audio": {
    "segment_ref": "artifact-or-buffer-ref",
    "codec": "pcm_s16le",
    "sample_rate_hz": 16000,
    "channels": 1,
    "time_range_ms": [12840, 15320],
    "vad": {"speech_start_ms": 12840, "speech_end_ms": 15320},
    "authority": "primary_audio"
  },
  "speaker": {
    "platform": "discord",
    "channel_user_id": "discord-user-id",
    "display_name": "jetha",
    "is_bot": false
  },
  "channel": {
    "transport": "discord_voice",
    "guild_id": "discord-guild-id",
    "channel_id": "discord-channel-id",
    "surface": "desk_voice"
  },
  "reflex": {
    "route": "defer",
    "intent": "calculate a power",
    "interface_already_said": "I'm checking that.",
    "ack_event_id": "voice-ack-001",
    "authority": "reflex_hypothesis"
  },
  "transcript_hypotheses": [
    {
      "kind": "reflex_transcript_hypothesis",
      "source": "moshi-reflex",
      "text": "three to the power of seventeen",
      "partial": false,
      "time_range_ms": [12920, 15300],
      "latency_ms": 110,
      "confidence": 0.68,
      "authority": "hypothesis",
      "tool_authority": false
    },
    {
      "kind": "frontend_witness_hypothesis",
      "source": "moshi",
      "text": "what is three to the power of seventeen",
      "partial": false,
      "time_range_ms": [12840, 15320],
      "latency_ms": 145,
      "confidence": 0.78,
      "authority": "hypothesis",
      "tool_authority": false
    }
  ],
  "interpreter_input_order": ["raw_audio", "metadata", "reflex", "transcript_hypotheses"],
  "interpreter_prompt_policy": {
    "version": "raw_audio_compare_v1",
    "raw_audio_primary": true,
    "witness_transcripts_context_only": true,
    "require_witness_adjudication": true
  },
  "interpreter": {
    "model": "gemma-4-audio",
    "status": "pending"
  },
  "oracle_job_id": "voice-oracle-001"
}
```

`audio.segment_ref` mirrors the canonical `audio_segment_ref` for adapters that
nest transport audio metadata; both values must identify the same accepted
speech cut. `evidence_bundle_id` remains stable for the logical turn, while
`evidence_merge_key` proves the raw-audio/witness join for the specific
session, turn, and audio reference.

If an operator calls the Moshi side channel "Moshi STT", the runtime should
store it as `frontend_witness_hypothesis` unless the adapter can prove it came
from the live reflex itself or from a named same-bundle S2S witness producer.
The name can appear in `source`; hypothesis kind/source describes provenance,
while `authority` stays `hypothesis` and `tool_authority` stays `false`.

If the adapter cannot confidently tell whether the string came from the reflex
model itself or from a named S2S witness producer, it should use
`frontend_witness_hypothesis`. That is still a hypothesis and still attaches to
the raw-audio bundle. It is deliberately safer than guessing `classic_asr` or
promoting the text to a user turn.

The interpreter prompt should receive the bundle in three sections:

1. Primary audio: `audio_segment_ref`, timing, VAD/energy metadata, speaker, and
   channel context.
2. Live interface context: reflex route, provisional intent, acknowledgement
   already spoken, and interruption/playback state.
3. Hypotheses: Moshi/open-S2S, reflex, VoiceClaw/OpenClaw, and classic ASR text
   with source, timing, partial/final state, confidence when available, and
   explicit authority labels.

The prompt must tell the interpreter that hypotheses may be clipped, stale,
hallucinated, or from the wrong speaker. Gemma may use them to recover names,
numbers, prefixes, and code-switched phrases, but it must prefer the raw audio
when the signals disagree and must report material disagreements.

`interpreter_prompt_policy.version = "raw_audio_compare_v1"` is the normal
policy for this packet. It means "compare witness text to raw voice" rather
than "continue from this transcript." This is the concrete implementation of the
Moshi-context decision: send the Moshi/Open-S2S transcript beside the raw
waveform, let Gemma use it as a clue, and require an adjudication outcome before
the text can influence durable user wording or any action field.

Lifecycle rules:

- partial transcript hypotheses attach to the pending bundle with
  `partial = true`; a final hypothesis from the same source and kind supersedes
  the partial for active interpreter context, while preserving the superseded
  partial only as timing/provenance on the retained final hypothesis
- acknowledgement and oracle-job creation do not wait for Moshi/open-S2S or
  classic ASR hypotheses when raw audio plus reflex routing is enough
- late transcript hypotheses attach to the same bundle and can update a queued
  oracle job only through interpreter evidence
- no hypothesis text may become durable user text, `oracle_text`, a tool
  argument, a spend reason, a call/message payload, or a memory write without
  `interpreter_promoted` or later `oracle_promoted` authority
- text-only external `ask_brain` calls remain compatibility inputs; they are
  useful, but they do not satisfy the full raw-audio KAME interpreter path
- the interpreter bundle must carry a `degraded_reason` when raw audio is
  missing, unavailable, clipped below the configured evidence floor, or rejected
  by speaker/energy checks; degraded turns may draft or ask clarifications, but
  they cannot satisfy readiness or high-risk action gates on witness text alone

## Model Assumptions To Validate

This design relies on the following external model and serving assumptions:

- Moshi/PersonaPlex-class S2S models can provide very low-latency floor
  control and rough transcript hypotheses, but should not be trusted as durable
  transcript truth or granted broad tools.
- Moshi-class transcript output, when available, is auxiliary evidence for the
  interpreter and oracle. It must not become a separate STT-first control path,
  and Hermes must still function when that transcript is absent, late, or
  contradicted by raw-audio interpretation.
- Gemma 4 E2B/E4B/12B supports text, image, and audio input, and produces text output.
- Gemma 4 E2B/E4B use a USM-style Conformer audio encoder.
- Gemma 4 audio input is bounded; Google's model card currently lists a 30 second audio limit.
- vLLM exposes Gemma 4 multimodal serving controls through `--limit-mm-per-prompt`, including audio prompt limits and audio memory allocation controls.
- The oracle is selected through Hermes's normal `/model` flow. Voice realtime
  config must not add a separate `oracle_model` selector.
- Nemotron 3 Super is the preferred first Spark-local NVIDIA oracle target to
  validate for the hackathon and appliance path. Gemma 4 26B-A4B remains a
  comparison candidate if it proves better for Hermes-style work.
- Nemotron 3 Ultra is a hosted or future multi-Spark fallback unless measured
  one-Spark evidence proves otherwise.

These assumptions must be checked against the exact model checkpoint and runtime before implementation is considered complete.

## Open S2S And Speech Alternative Matrix

These candidates are alternatives to a hosted Gemini Live-style frontend. They
must be evaluated by their fit to the KAME authority model, not just by whether
they can produce text quickly.

| Candidate | Best KAME Role | Why It Matters | Current Decision |
| --- | --- | --- | --- |
| Moshi / PersonaPlex-class S2S | Reflex / floor-control candidate | Open speech-text dialogue stack; can potentially speak quickly and expose what the live voice model believed it heard. | Evaluate as reflex only. Treat ambiguous STT-like output as `frontend_witness_hypothesis` unless the adapter can prove `reflex_transcript_hypothesis` or a distinct `s2s_transcript_hypothesis`; never durable user text. |
| Ultravox-class speech LLM | Interpreter or reflex watchlist | Direct speech understanding can reduce dependence on a separate ASR stage. | Watchlist until local latency, noise-gate behavior, and Discord audio robustness are measured. Do not grant tool authority. |
| Qwen Omni-class any-to-any model | S2S / interpreter watchlist | Can combine multimodal perception and speech output in one model family. | Watchlist for local serving complexity and latency. It may become a reflex/interpreter candidate, but Hermes still keeps the oracle boundary. |
| Gemma 4 E2B/E4B/12B audio-multimodal | Interpreter / evidence adjudicator | Strong fit for raw-audio plus text-context interpretation after a VAD cut. | Preferred interpreter lane. It receives raw audio first and transcript hypotheses second. |
| Nemotron Speech / Riva ASR | Optional witness/fallback transcript evidence | Purpose-built streaming ASR can provide literal text, timestamps, and diagnostics. | Use as fallback/evidence/caption lane, not the normal KAME scheduler. |
| Magpie / Riva-style TTS | Outbound speech | Local or NVIDIA-aligned TTS helps remove cloud TTS from the final appliance path. | Evaluate for first-audio latency and voice quality. TTS does not change transcript authority. |
| Piper / other small local TTS | Cheap local outbound speech | May run on the gateway host or Spark with low operational complexity. | Candidate fallback if quality and first-audio latency beat hosted TTS for short reflex acknowledgements. |
| Cartesia or other hosted STT/TTS | Bring-up and provider comparison | Useful baseline when local speech stack is unstable. | Allowed as labeled fallback/comparison only; not the target KAME control path. |

The evaluation question for every candidate is:

```text
Can it improve floor control, raw-audio interpretation, or speech output without
turning a hypothesis into action authority?
```

If the answer is no, the component may still be useful as a degraded bridge or
diagnostic source, but it should not be part of the normal full-KAME path.

## System Shape

```text
Discord voice / VoiceClaw / OpenClaw Talk / phone-SIP / desktop mic
  -> transport adapter
  -> KAME interface session
       -> streaming audio input
       -> VAD / turn detector
       -> fast reflex / floor-control model
            -> immediate ack / local control / rough transcript hypothesis
       -> optional witness/fallback transcript hypothesis sources
            -> Moshi/S2S transcript hypothesis or classic ASR hypothesis
       -> interpreter lane
            -> Gemma 4 audio model over raw clip
            -> optional context: reflex/Moshi transcript hypothesis
            -> optional fallback context: classic ASR hypothesis
            -> corrected transcript / entities / oracle request patch
       -> speech planner
       -> TTS or native speech output
       -> oracle router
            -> interpreter evidence
            -> provenance-labeled transcript hypotheses only after interpreter merge
            -> Hermes gateway / oracle session
            -> tools, MCP, memory, files, project context
       <- oracle hints, tool results, final answer
  -> mixer / playback / captions
```

The existing Discord realtime sidecar can become the first KAME interface
session host. It must not become the architecture boundary. The durable boundary
is a transport-neutral KAME session protocol that can be driven by Discord,
VoiceClaw, OpenClaw Talk, telephony bridges, desktop mic/speaker, web, or future
mobile clients.

VoiceClaw and OpenClaw are useful reference points because their public
architecture already resembles a prompt-mediated KAME split: a realtime voice
frontend speaks to the user, then calls an `ask_brain` or
`openclaw_agent_consult` style bridge for real agent work. Hermes should support
that frontend shape, but strengthen the backend contract:

- `ask_brain` maps to typed Hermes oracle jobs, not raw hidden chat turns
- placeholder tool results map to explicit acknowledgement and job status events
- injected brain results map to `oracle.job.completed` or `oracle.job.failed`
  events with relevance checks before speech
- frontend transcript sync maps to the Hermes voice-session ledger, not directly
  to durable chat history
- external realtime clients never gain direct file, shell, memory, payment, or
  provisioning authority

This lets VoiceClaw/OpenClaw-style clients become front doors for Hermes KAME
without weakening the reflex/oracle authority boundary.

## Moshi / Open-S2S Transcript Context Contract

Open speech-to-speech frontends are useful when they can stay warm, respond
quickly, and keep the user-oriented voice loop separate from Hermes' heavier
oracle loop. They are not useful if Hermes treats their transcript side channel
as a replacement for the raw waveform or as an independent user message.

When a Moshi, VoiceClaw, OpenClaw, or similar S2S frontend emits text, Hermes
should handle it as:

1. `frontend_witness_hypothesis` by default. This is the normal Moshi/Open-S2S
   adapter label for transcript-looking text that reports what the frontend
   believed it heard.
2. `reflex_transcript_hypothesis` only when the adapter can prove the text came
   from the live reflex model's own hearing for the same accepted speech cut.
3. `s2s_transcript_hypothesis` only when the adapter can prove a named S2S
   witness producer emitted the text for the same accepted speech cut. This is
   still same-bundle context, not a separate STT lane.
4. `classic_asr_hypothesis` only for a dedicated ASR provider used for fallback,
   diagnostics, captions, or literal wording checks.

All four labels are context for the interpreter. They are allowed to improve
Gemma's correction, entity extraction, language notes, and oracle request patch.
They are not allowed to:

- create a second oracle turn
- overwrite `oracle_text` directly
- become a spend reason, call payload, memory write, or tool argument by
  themselves
- block acknowledgement while Hermes waits for them
- hide disagreement with the raw audio, energy gate, speaker identity, or later
  interpreter/oracle judgment

This is the answer to the "can we provide the Moshi STT transcript as context
along with raw voice?" question: yes, that is exactly the desired packet shape.
The raw voice clip and timing metadata are the primary interpreter evidence;
Moshi/open-S2S text is a labeled clue that Gemma may accept, correct, or reject.
The implementation should prefer calling this `frontend_witness_hypothesis`
unless the adapter can prove the exact producer. That keeps the semantics stable
across Moshi, OpenClaw, VoiceClaw, classic-ASR captions, and future open-S2S
frontends: the text is what a frontend believed it heard, not what Hermes has
verified the user said.

The prompt should make that visible to Gemma in plain language: "The audio is
primary. The witness transcript is what the frontend believed it heard. Use it
as a clue, especially for clipped prefixes, names, numbers, and code-switching,
but reject it when it conflicts with the waveform, speaker, energy timing, or
conversation state." That instruction belongs in the interpreter prompt, not in
the oracle prompt, because Hermes' active `/model` should receive promoted
evidence rather than raw witness text masquerading as the user message.

Do not implement this as two conversations where one lane asks Hermes from raw
audio and another lane asks Hermes from Moshi text. The frontend may emit
multiple sensor events, but the KAME session must join them by `turn_id` and
`audio_segment_ref` before oracle authority. If the Moshi hypothesis arrives
before the raw clip is finalized, attach it to the pending bundle. If it arrives
after the interpreter has started, attach it as late evidence. In neither case
does it become a new user turn.

The interpreter request should make that hierarchy visible in the wire format and
prompt. Put the raw audio reference and timing fields in the primary input
section, then put Moshi/open-S2S/classic-ASR text under a separate
`transcript_hypotheses[]` field with `authority = "hypothesis"`. Older
adapter-edge names such as `auxiliary_transcript_hypotheses` should be
normalized into that field before the interpreter sees the packet. The prompt
must explicitly tell the interpreter that transcript hypotheses can be wrong,
clipped, hallucinated, stale, or from a different speaker, and that it should
prefer raw-audio interpretation when the signals disagree.

This also resolves the "fast reflex plus Gemma multilingual ASR" shape. Gemma is
not merely a background ASR process and should not block acknowledgement. It is
the interpreter that sees the clipped waveform and the evidence bundle after the
reflex has already handled floor control. When Gemma emits a corrected
transcript, language note, number/name correction, or tool-critical entity, that
output can update a queued oracle job or become an audited late correction. The
classic ASR lane remains optional because it is a comparison/fallback source,
not a prerequisite for routing.

Implementation-wise, this should be modeled as transcript evidence attaching to
the current `turn_id` / `audio_segment_ref`, not as a parallel STT conversation.
The adapter can emit transcript evidence before the raw clip is finalized, with
the raw clip, or after the interpreter has already started. In every case, Hermes
records it as a hypothesis on the same interpreter bundle and never schedules a
second oracle request from that text alone.

Acceptance proof: the headless async-oracle smoke must exercise all three
arrival phases (`before_raw_audio`, `with_raw_audio`, and
`after_interpreter_start`) and report `witness_fusion_*` fields proving one
stable `evidence_bundle_id`, one accepted/started/completed oracle job per
speech cut, and no duplicate oracle job. The bundle id is keyed by
`session_id` and `turn_id`; raw audio refs and degraded/primary status may
arrive later and must update the bundle rather than changing its identity.

External frontend adapters must preserve that shape instead of flattening it into
one text turn. A VoiceClaw/OpenClaw/Moshi-style bridge may send an `ask_brain`
request early, but the Hermes adapter should treat it as an interpreter/oracle
job envelope with explicit evidence fields:

- `audio_segment_ref` and `audio_time_range_ms` when the frontend can expose the
  clipped waveform or a replayable artifact
- `reflex_intent` and `interface_already_said` for what the live interface chose
  and already spoke
- `reflex_transcript_hypothesis` for the reflex model's own hearing
- `transcript_hypotheses[]` for frontend-witness, Moshi/open-S2S caption,
  reflex, classic-ASR, or other transcript-like side channels, each with kind,
  source, timing/confidence when available, and `authority = "hypothesis"`
- `frontend_session_id`, `frontend_turn_id`, and `tool_call_id` for correlation,
  cancellation, and terminal result delivery

Every evidence field should also carry an authority label. The minimum labels
are `primary_audio`, `reflex_hypothesis`, `hypothesis`,
`interpreter_promoted`, `oracle_promoted`, and `diagnostic_only`.
Transcript-like side channels use `authority = "hypothesis"` plus a `kind`
such as `frontend_witness_hypothesis`, `reflex_transcript_hypothesis`,
`s2s_transcript_hypothesis`, or `classic_asr_hypothesis`. The scheduler may
use provisional reflex intent to create a queue envelope or narrate work, but
durable replay must make clear which fields were hypotheses and which field
actually drove the oracle/tool action.

If an external frontend can provide only text and no replayable audio reference,
Hermes may still run it through the compatibility path, but that turn is degraded
evidence. It must not satisfy the full KAME raw-audio interpreter gate, and it
must not promote the transcript into durable user text without interpreter or
oracle judgment.

External frontend history has a separate runtime contract. An accepted
VoiceClaw/OpenClaw/Moshi `ask_brain` bridge request must emit a normalized
`INTERFACE_ORACLE_REQUEST` event with the job id, turn id, source, correlation
ids, evidence status, audio references when available, and authority labels.
Accepted/queued placeholders, safe reflex status, and transcript sync packets
remain transport/session state. Durable Hermes voice history may retain counts,
authority labels, job ids, and promoted interpreter/oracle fields, but raw
Moshi/S2S/ASR/reflex hypothesis strings must not appear in durable conversation
records unless a trusted interpreter promotes them from raw-audio-grounded
evidence.

If a Moshi-style frontend provides both audio and transcript text, the audio
reference should be preferred even when the transcript appears cleaner. The
transcript is useful because it shows what the realtime model believed it heard;
it is not proof that the user said those words.

If the frontend can stream partial Moshi transcripts before the audio cut is
finalized, Hermes should attach them to the same pending `turn_id` and mark them
`partial = true`. A later final transcript from the same source and kind
replaces the partial as the only active interpreter hypothesis. The audit ledger
should keep the superseded partial text and timing/provenance needed to debug
clipped starts, duplicated words, and hallucinated commands.

Runtime merge algorithm:

1. Allocate `turn_id` at speech start and `audio_segment_ref` when the session
   has a replayable cut or buffer reference.
2. Attach every frontend text observation to that pending turn as a hypothesis:
   `reflex_transcript_hypothesis`, `s2s_transcript_hypothesis`,
   `classic_asr_hypothesis`, or `frontend_witness_hypothesis`.
3. Start the interpreter as soon as raw audio plus timing metadata are available;
   do not wait for Moshi/OpenClaw/VoiceClaw/classic-ASR text.
4. If witness text arrives before the interpreter starts, include it in the
   initial Gemma request. If it arrives after start, attach it as late evidence
   on the same bundle and allow only bounded queued-job patches or pre-action
   corrections.
5. If witness text arrives without raw audio, mark the turn degraded and keep
   the text out of high-risk action authority; oracle-only promotion of
   text-only witness content does not satisfy the full KAME action boundary.

Acceptance for this merge path is concrete: one speech cut must produce exactly
one evidence bundle id, no more than one oracle job unless the reflex explicitly
routes multiple tasks, and zero durable user-message writes sourced only from a
Moshi/OpenClaw/VoiceClaw/classic-ASR hypothesis. The logs and readiness artifacts
should make that visible by reporting the shared `turn_id`, `audio_segment_ref`,
`evidence_bundle_id`, hypothesis count, promoted source, and degraded reason when
raw audio is missing.

Multi-human calls add one more mandatory proof: witness text must be bound to
the accepted speaker and channel before it can be useful. In a Discord call with
two humans, a Moshi/open-S2S or classic-ASR string from the wrong user, a stale
buffer, or an ambiguous speaker segment is still valuable diagnostic evidence,
but it must be adjudicated as `rejected_or_diagnostic_only` with a reason such
as `wrong_speaker`, `wrong_channel`, `stale_witness`, or `ambiguous_speaker`.
The interpreter may compare that rejected witness against the raw audio, but the
oracle must never receive it as promoted user wording, spend rationale, phone
payload, memory/file content, or a tool argument.

The headless proof for that rule lives in the async oracle smoke and the voice
operator readiness artifact. It must include a second-human witness whose text is
high-risk enough to matter, such as a spend or phone command, and prove all of
the following in one run: `witness_fusion_multi_speaker_witness_smoke_ok`,
`witness_fusion_multi_speaker_wrong_witness_rejected`,
`witness_fusion_multi_speaker_bound_to_second_human`, and
`witness_fusion_multi_speaker_action_sinks_clean`. The rejected witness remains
available as audit context with `tool_authority: false`, role `witness_context`,
and typed rejection reasons, but the promoted oracle/action text must stay bound
to the accepted speaker's raw-audio cut.

## Responsibilities

### Reflex / Interface Model

The reflex is optimized for latency, turn-taking, and conversational control. It
should be small enough to stay warm beside the oracle and interpreter on the DGX
Spark, or cheap enough to run on the gateway host. The preferred reflex class is
Moshi/PersonaPlex-style S2S or an even smaller timing/classifier path that can
produce immediate acknowledgements and rough transcript hypotheses. Reflex
quality is judged by floor control and response latency, not by long-form
reasoning.

The reflex transcript is a hypothesis. It is useful because it arrives early and
captures what the live model thought it heard, but it must not become the
unquestioned transcript of record. Moshi/S2S transcript output belongs in this
same class: helpful context, not authority. The interpreter and auxiliary
transcript-hypothesis lanes are not equal authorities: the interpreter may correct
the reflex hypothesis using auxiliary transcript hypotheses as labeled context
before the oracle executes tool calls or external actions.

If the reflex is a Moshi-style S2S model that emits a transcript or STT-like
side channel, Hermes should pass that text to the interpreter beside the raw
audio segment. This is useful even when the transcript is imperfect because it
captures what the realtime model believed it heard, including timing,
code-switching, dropped prefixes, and possible hallucinations. The interpreter
prompt must label it explicitly as a hypothesis, not as ground truth:

- `raw_audio`: the clipped utterance and timing metadata
- `reflex_transcript_hypothesis`: what the live reflex thought it heard
- `frontend_witness_hypothesis`: ambiguous Moshi/OpenClaw/VoiceClaw-style text,
  meaning what the realtime frontend believed it heard
- `s2s_transcript_hypothesis`: a proven named S2S witness output for the same
  accepted raw-audio cut, if separate from the reflex hypothesis
- `classic_asr_hypothesis`: optional fallback/evidence transcript, if enabled

The interpreter may use all of those signals, but its output must identify
disagreements instead of silently averaging them away.

The presence of a Moshi/S2S transcript must be opportunistic, not blocking. A
Gemma interpreter request can proceed with only the clipped raw audio and timing
metadata; any Moshi/S2S or classic ASR hypothesis that arrives later can be
attached as late evidence before irreversible oracle actions.

This also means there should be no separate "wait for ASR evidence" phase in the
normal KAME path. Waiting for transcript text is allowed only in explicit
fallback/debug modes or when a high-risk action policy asks for additional
literal evidence before approval.

Approval and operator surfaces must preserve the same boundary. A pending
Stripe, provisioning, phone, message, file, or memory action should carry
promoted interpreter/oracle evidence plus a tool-disclosure reference. The UI
may show Moshi/OpenClaw/VoiceClaw/reflex/ASR hypotheses as witness context, but
the approval contract is invalid if those hypotheses are the only evidence for
the action text, spend reason, provider choice, phone payload, or durable user
request.

At runtime, the async oracle job manager should expose the same rule as a compact
action gate on approval waits. `oracle.job.waiting_for_approval` payloads and
approval-related tool progress events should include
`voiceops.runtime_kame_action_gate.v1` with `ok=false` until promoted evidence
has been consumed before the irreversible boundary and `tool_disclosure_ref =
"tool_disclosure"` is present. The gate may list reflex/Moshi/OpenClaw/VoiceClaw
or ASR hypotheses as present witness context, but those labels are not accepted
authorities and cannot make the gate pass.

Headless acceptance for this runtime gate is explicit: the async oracle smoke
must report `runtime_kame_action_gate_*` fields proving that hypothesis-only
approval waits fail closed with `missing_promoted_evidence`, while approval
waits backed by consumed `interpreter_promoted` evidence and
`tool_disclosure_ref = "tool_disclosure"` pass.

It owns:

- turn detection interpretation and speech boundary decisions
- barge-in behavior and cancellation of current speech
- immediate acknowledgements such as "one second", "got it", and "checking"
- local handling for greetings, repeats, clarification questions, and low-risk conversational glue
- short spoken style, normally one or two sentences unless the user asks for more
- first-pass spoken user intent and rough transcript hypothesis
- deciding whether the oracle is likely needed
- summarizing long oracle output into voice-appropriate responses
- ephemeral conversational state for the live voice session

It must not own:

- durable transcript truth
- durable memory writes
- filesystem or project changes
- MCP/tool execution authority
- long-running task planning
- claims about capabilities that differ from Hermes's actual runtime state

Initially the interface should have no direct tool access. If direct tools are added later, they should be narrow, explicitly scoped, and auditable.

### Interpreter / Audio Evidence Model

The interpreter is optimized for audio understanding, multilingual robustness,
and oracle evidence quality. The preferred interpreter candidate is Gemma 4
E2B/E4B/12B because it can process bounded raw audio with text context and
produce structured text, while staying separate from the low-latency reflex and
the durable Hermes oracle.

Gemma 4 should be treated as an audio-understanding and routing/evidence model,
not as the whole speech stack. The public Gemma 4 model descriptions describe
multimodal input with text output, so Gemma can consume a clipped utterance plus
the reflex's transcript hypothesis and produce corrected evidence, but TTS is
still needed for spoken output.

Gemma 4 E2B's audio path should be treated as a buffered segment encoder, not as a streaming endpointer. It can ingest a cut audio segment and reason over it, but the realtime sensor remains VAD/endpointer logic. The hot path is:

1. VAD detects speech start, speech energy, and speech end.
2. The session cuts a bounded audio segment.
3. The reflex immediately acknowledges or controls the floor from live audio.
4. The interpreter encodes the segment plus the reflex/Moshi transcript
   hypotheses and any available optional witness/fallback transcript evidence.
5. The interpreter emits corrected transcript, intent, entities, disagreement
   flags, and an oracle request patch.
6. The oracle job uses the best available request and may accept late
   interpreter evidence before irreversible tool execution.

Mid-speech backchannels require rolling windows and should be deferred until the cut-segment path is stable.

STT should not feed the reflex in normal full KAME mode. A second interpretation stream in front of the reflex creates disagreement risk between "what the reflex heard" and "what STT transcribed." The reflex's live audio path is the primary truth for floor control, but not for durable transcript or tool arguments.

Moshi-class transcript output is different from classic STT operationally but
must be treated the same way semantically: it is a hypothesis attached to the
same interpreter request as the clipped waveform. It should not be a separate
turn, a separate oracle prompt, or a competing source of durable transcript
truth. The interpreter may use it to recover prefixes, names, numbers, or
code-switched phrases, but it must be able to reject it when the raw audio
contradicts it. If a Moshi transcript contains words the user did not say, the
disagreement must be visible in interpreter evidence and must not be written as
durable user text.

The interpreter receives:

- clipped raw audio
- speaker and channel metadata for the canonical turn context
- reflex transcript hypothesis
- optional Moshi/S2S transcript hypothesis if it differs from the reflex
- optional classic ASR transcript hypothesis
- language and speaker metadata
- reflex route, acknowledgement, and "interface already said" text
- current oracle job/status context

The interpreter emits:

- corrected transcript or transcript alternatives
- normalized intent and route confidence
- entities, numbers, names, URLs, code terms, and language notes
- disagreement flags between raw audio, reflex transcript, Moshi/S2S transcript,
  and ASR
- oracle request patch or clarification recommendation

The interpreter may attach evidence to a queued oracle job before it starts or
send a bounded update to a running oracle job. It must not stall the reflex
acknowledgement and must not receive broad Hermes tools.

The interpreter input bundle is the durable boundary between "live hearing" and
"evidence for action." Each speech cut should create at most one bundle:

```json
{
  "turn_id": "voice-turn-id",
  "audio_segment_ref": "artifact-or-buffer-ref",
  "audio_time_range_ms": [12840, 15320],
  "speaker_metadata": {
    "platform": "discord",
    "channel_user_id": "discord-user-id",
    "display_name": "jetha",
    "is_bot": false
  },
  "channel_metadata": {
    "transport": "discord_voice",
    "guild_id": "discord-guild-id",
    "channel_id": "discord-channel-id"
  },
  "reflex_route": "defer",
  "reflex_intent": "calculate a power",
  "transcript_hypotheses": [
    {
      "source": "reflex",
      "kind": "reflex_transcript_hypothesis",
      "role": "witness_context",
      "text": "three to the power of seventeen",
      "partial": false,
      "arrival_phase": "with_raw_audio",
      "authority": "hypothesis",
      "promotion_required": "interpreter_promoted_or_oracle_promoted",
      "tool_authority": false
    },
    {
      "source": "moshi",
      "kind": "frontend_witness_hypothesis",
      "role": "witness_context",
      "text": "what is three to the power of seventeen",
      "partial": false,
      "confidence": 0.78,
      "latency_ms": 140,
      "arrival_phase": "with_raw_audio",
      "authority": "hypothesis",
      "promotion_required": "interpreter_promoted_or_oracle_promoted",
      "tool_authority": false
    }
  ],
  "interface_already_said": "I'm checking that."
}
```

The bundle preserves provenance. The raw audio reference is the primary
evidence. Reflex and Moshi/S2S transcript strings are low-latency hints. Classic
ASR output, when present, is another hint. The interpreter owns the decision to
promote any wording into `interpreter_corrected_transcript`; nothing else in
the bundle is durable user text by default.

`transcript_hypotheses[]` is the canonical normalized field for every
transcript-looking witness. Older names such as
`auxiliary_transcript_hypotheses`, `reflex_transcript_hypothesis`,
`transcript.partial`, `transcript.final`, `stt_text`, or provider-specific
`query` fields are adapter-edge aliases only. They may be accepted at ingress,
but the interpreter packet, oracle job record, readiness artifacts, and package
audits should expose one canonical hypothesis list with explicit `kind`,
`source`, `role`, `authority`, `promotion_required`, `tool_authority`, timing,
and arrival-phase metadata.

### Oracle

The oracle is whatever Hermes is configured to use. Today that may be Kimi K2.6
through Hermes. For the DGX Spark path, the first preferred local NVIDIA oracle
target is Nemotron 3 Super, selected through Hermes's normal `/model` flow after
registering the local OpenAI-compatible endpoint. Gemma 4 26B-A4B remains a
comparison candidate, not the VoiceOps-specific oracle selector. Nemotron 3
Ultra is only a hosted or future multi-Spark fallback unless measured local
evidence proves a one-Spark path. This design must not be read as a claim that
the Spark-local oracle deployment has already been validated.

It owns:

- durable user intent handling
- tools, MCP, files, memory, and project context
- long context reasoning
- high-accuracy answers
- plans and task execution
- durable transcript commits

The oracle should receive structured requests from the interface instead of raw streaming audio fragments.

### Optional Witness/Fallback Transcript Hypothesis Inputs

Optional witness/fallback transcript inputs are not the old STT-first voice pipeline. They are
optional hypothesis fields attached to the same raw-audio interpreter bundle
when Hermes needs exact wording or when a fast S2S/reflex model already produced
a useful but untrusted transcript hypothesis. Moshi-class transcript output and
classic ASR output both enter through this bundle.

These inputs are allowed to improve the interpreter and oracle requests; they
are not a prerequisite for a voice turn. In full KAME mode the ordering is:

1. VAD/endpointer cuts the user turn.
2. Reflex responds or submits work from live audio.
3. Interpreter receives raw audio plus any available transcript hypotheses.
4. Classic ASR evidence may arrive only as optional fallback or comparison
   context.

The absence of ASR evidence must not block the reflex acknowledgement or prevent
oracle job creation when the raw audio and reflex route are sufficient.
The useful question for Moshi/open-S2S and ASR is whether their hypotheses help
the interpreter recover clipped prefixes, names, numbers, code switches, or
intent without being captured by hallucinated text. They are measured as
evidence quality and fallback value, not as the control path.

Modes:

- `disabled`: no separate transcript evidence is run; the oracle receives
  reflex intent and, optionally, the audio segment reference. This disables only
  transcript side-channel collection; it does not disable raw-audio
  interpretation when an interpreter is configured and an audio segment is
  available.
- `from_reflex`: only transcript hypotheses emitted by the reflex/S2S model are
  forwarded as evidence.
- `on_escalation`: dedicated ASR runs only after the reflex chooses `defer` or
  `oracle_direct`.
- `speculative`: dedicated ASR may start at speech end in parallel with the
  reflex, but its output is discarded for local turns and never drives the
  reflex.
- `debug`: transcript evidence runs for comparison, captions, and diagnostics.
- `fallback`: transcript evidence feeds the reflex only when the realtime reflex
  audio path is unavailable.

Default target mode: `from_reflex` when the reflex produces a transcript,
otherwise `disabled`. Dedicated ASR should be enabled only for explicit
fallback, diagnostics, captions, or literal-evidence checks, and its output must
stay off the acknowledgement critical path.

`speculative` can be enabled if measurements show that waiting until after the
reflex decision delays oracle requests. Even then, transcript evidence remains
an interpreter hypothesis input plus optional labeled audit context for the
oracle, not a reflex dependency and not a peer conversation path.

`speculative` is also not a request to make ASR authoritative. It exists only to
hide optional comparison latency behind the reflex decision. If speculative ASR
or a Moshi transcript disagrees with raw-audio interpretation, the disagreement
must be visible in interpreter evidence and the oracle request must prefer the
interpreter-promoted wording for tool-critical arguments.

Acceptance gates:

- a voice turn may acknowledge and create an oracle job without ASR evidence
- Moshi/S2S transcript evidence must be labeled `authority = "hypothesis"`
- Moshi/OpenClaw/VoiceClaw witness text is accepted only as interpreter context
  joined to the same `turn_id` and `audio_segment_ref` as the raw audio
- transcript-only witness evidence must fail the full KAME live-evidence gate;
  it can be preserved for audit, captions, clarification, or degraded fallback,
  but it does not prove that Hermes heard raw voice or that Gemma interpreted it
- live-evidence readiness artifacts must expose
  `raw_audio_interpreter_evidence_observed` and
  `transcript_only_witness_rejected_for_full_kame` so this failure is explicit,
  not inferred from generic missing fields
- interpreter prompts must include raw audio whenever an audio segment is
  available, even when Moshi or ASR produced a complete-looking transcript
- interpreter prompts explicitly identify witness transcripts as non-authority
  and ask the model to compare them against raw audio before promotion
- this prompt policy is a first-class field, currently
  `interpreter_prompt_policy.version = "raw_audio_compare_v1"`, and readiness
  artifacts must prove it is visible alongside the prompt input order
- readiness artifacts must expose KAME speech-end-to-first-audio metrics for
  defer acknowledgements and local reflex replies, so acknowledgement latency is
  measured from the reflex decision path rather than inferred from generic mixer
  timing
- rejected frontend witnesses must preserve typed rejection reasons for
  speaker, channel, and timing conflicts, including `ambiguous_speaker`,
  `wrong_speaker`, `wrong_channel`, and `stale_witness` when those conflicts are
  present
- a `rejected_or_diagnostic_only` witness without `rejection_reasons[]` is not
  acceptable live evidence; allowed reason codes are `ambiguous_speaker`,
  `wrong_speaker`, `wrong_channel`, `stale_witness`, `timing_conflict`,
  `low_energy_non_speech`, `waveform_conflict`, and `provider_conflict`
- text-only VoiceClaw/OpenClaw compatibility requests are marked degraded when
  no raw audio is available
- oracle jobs must distinguish hypothesis entries inside
  `transcript_hypotheses[]` from `interpreter_corrected_transcript`
- durable transcript writes and tool-critical arguments must use interpreter or
  oracle judgment, not raw Moshi/ASR text alone
- readiness artifacts must prove unpromoted witness text is absent from spend,
  call/phone, tool-argument, memory/file, and external-message action sinks

## Session Contract

The interface session should use explicit events between transport, speech, interface, and oracle layers. This prevents Discord-specific behavior from leaking into model policy.

Core input events:

- `session.start`
- `audio.input.chunk`
- `speech.start`
- `speech.energy`
- `speech.end`
- `barge_in.detected`
- `playback.started`
- `playback.stopped`
- `playback.cursor`
- `transport.buffer.cleared`
- `session.stop`

Auxiliary transcript/fallback evidence events:

- `transcript.partial` legacy provider-stream alias for a partial hypothesis
- `transcript.final` legacy provider-stream alias for a provider-final
  hypothesis
- `reflex.transcript.hypothesis`

Those names are legacy-compatible event names. In KAME mode they normalize into
`transcript_hypotheses[]`, not verified transcript events. Each event must carry
explicit provenance: source, authority, turn id, audio segment id when
available, timing, confidence when available, and whether the evidence arrived
before the raw-audio cut, with the raw-audio cut, or after interpreter start.
`transcript.final` means final for that provider stream only; it does not mean
verified user text and must not directly patch `oracle_text`.

Interpreter evidence events:

- `interpreter.evidence.started`
- `interpreter.evidence.final`
- `interpreter.evidence.patch`

These transcript events are disabled unless transcript evidence mode is
`from_reflex`, `on_escalation`, `speculative`, `debug`, or `fallback`. They must
not make the normal KAME path STT-first. Interpreter evidence events are governed
by the interpreter lane and may exist for raw-audio turns even when auxiliary
transcript evidence is disabled. Reflex/Moshi transcript hypotheses and
interpreter evidence are separate: transcript hypotheses are early and
non-durable by default, while interpreter evidence is the corrected
audio-understanding artifact offered to the oracle.

Interface events:

- `interface.intent.partial`
- `interface.intent.final`
- `interface.reply.local`
- `interface.reply.defer`
- `interface.oracle.submit`
- `interface.oracle.cancel_job`
- `interface.oracle.update_job`
- `interface.commit`

Oracle events:

- `oracle.job.accepted`
- `oracle.job.queued`
- `oracle.job.started`
- `oracle.job.progress`
- `oracle.job.waiting_for_approval`
- `oracle.job.completed`
- `oracle.job.failed`
- `oracle.job.cancel_requested`
- `oracle.job.cancelled`
- `oracle.job.result_suppressed`

Output events:

- `assistant.caption.partial`
- `assistant.caption.final`
- `assistant.audio.chunk`
- `assistant.audio.end`
- `session.metrics`

Transport adapters may expose provider-specific ids, but those ids must be
normalized at the session boundary. The internal ledger should be able to
correlate:

- audio chunk ids and timestamps
- speech-start, speech-end, and semantic endpoint decisions
- provider response ids or model item ids
- playback cursor and actual heard/unheard text spans
- barge-in, truncation, and downstream buffer-clear events
- tool calls, NemoClaw checks, approvals, oracle job ids, and result events
- handoff summaries between Discord, VoiceClaw, phone, WhatsApp, or other
  clients

Only final user intents, committed assistant responses, user-visible oracle job
results, cancellations, approvals, and external-action outcomes should enter
durable Hermes conversation state. Oracle job lifecycle detail belongs in the
voice-session task log/audit ledger. Partial transcripts, backchannels,
cancelled utterances, status polls, progress fragments, and interrupted audio
should stay ephemeral unless debugging is enabled. Debugging may persist these
items only to an audit/debug ledger, never directly to durable Hermes
conversation history or a verified user transcript.

## Routing Policy

The interface model should classify each turn into one of four paths.

`local`: The interface answers immediately without the oracle.

Use for greetings, short status checks, repeats, clarification prompts, "can you hear me", and low-risk conversational glue.

`defer`: The interface speaks a short acknowledgement and submits a background
oracle job. The voice loop remains live.

Use for ordinary Hermes questions, tasks, code/project questions, memory-dependent questions, and anything needing current Hermes context.

`oracle_direct`: The interface submits a background oracle job immediately. It
may speak a very short acknowledgement, but it must not pretend to know the
answer.

Use for high-stakes answers, tool use, filesystem work, MCP actions, long reasoning, or anything where a local guess would create confusion.

`reject_or_clarify`: The interface asks for missing information or refuses unsafe instructions before involving the oracle.

Use when the spoken turn is incomplete, ambiguous, unsafe, or impossible to route.

The default should be conservative: local replies are allowed only when the answer does not depend on Hermes state. If unsure, acknowledge quickly and escalate.

## Oracle Job Request Shape

The interface should submit a compact structured oracle job request:

```json
{
  "job_id": "voice-oracle-001",
  "session_id": "voice-session-id",
  "turn_id": "turn-id",
  "source": "discord",
  "user_id": "discord-user-id",
  "priority": "normal",
  "route": "defer",
  "audio_segment_ref": "artifact-or-buffer-ref",
  "evidence_bundle_id": "voice-session-id:turn-id",
  "evidence_merge_key": "audio-aware-join-proof",
  "provisional_request_summary": {
    "text": "compact provisional request summary for queueing and narration",
    "authority": "reflex_hypothesis",
    "tool_authority": false
  },
  "reflex_intent": "compact live intent",
  "transcript_hypotheses": [
    {
      "source": "reflex",
      "kind": "reflex_transcript_hypothesis",
      "role": "witness_context",
      "text": "three to the power of seventeen",
      "partial": false,
      "arrival_phase": "with_raw_audio",
      "authority": "hypothesis",
      "promotion_required": "interpreter_promoted_or_oracle_promoted",
      "tool_authority": false
    },
    {
      "source": "moshi",
      "kind": "frontend_witness_hypothesis",
      "role": "witness_context",
      "text": "three to the power of seventeen",
      "partial": false,
      "confidence": 0.74,
      "arrival_phase": "with_raw_audio",
      "authority": "hypothesis",
      "promotion_required": "interpreter_promoted_or_oracle_promoted",
      "tool_authority": false
    }
  ],
  "interpreter_corrected_transcript": "what is three to the power of seventeen",
  "interpreter_authority": "interpreter_promoted",
  "interpreter_confidence": 0.94,
  "interpreter_disagreements": ["reflex transcript omitted request prefix"],
  "interpreter_entities": [{"type": "math_expression", "value": "3^17"}],
  "interpreter_language_notes": ["English utterance with math expression"],
  "transcript": "promoted interpreter/oracle transcript candidate, not raw S2S/ASR text",
  "transcript_source": "interpreter",
  "transcript_confidence": 0.92,
  "intent": "normalized user request",
  "intent_source": "interpreter_audio",
  "mode": "voice",
  "urgency": "interactive",
  "interface_already_said": "One second, checking that now.",
  "conversation_summary": "ephemeral live voice summary",
  "metadata": {},
  "requested_response_style": {
    "spoken": true,
    "max_sentences": 2,
    "allow_followup_offer": false
  },
  "cancellation_token": "turn-cancel-token"
}
```

This gives the oracle enough state to answer without receiving every partial
audio event or every backchannel. The request should carry the reflex's
early hypotheses, the interpreter's corrected evidence when available, and any
optional witness/fallback transcript hypothesis when enabled. The oracle should prefer
interpreter-promoted evidence for tool arguments. Moshi/S2S or classic ASR
transcripts are auxiliary witness context, or fallback/diagnostic context only
in explicitly degraded mode. They may help explain interpreter provenance, but
durable text and tool arguments require `interpreter_promoted` or
`oracle_promoted` authority.

`provisional_request_summary` may start from a compact reflex intent so the job
can be queued without waiting. It is not durable user text and has no tool
authority. Irreversible tool, spend, provisioning, file, memory, message, or
call actions must use `interpreter_promoted` or `oracle_promoted` evidence
grounded in the raw-audio bundle.

If promoted interpreter evidence is already available before dispatch, it must
be folded into the oracle request before the job starts. The scheduler should
not delay acknowledgement or ordinary low-risk job start solely to wait for
transcript or interpreter evidence. If the interpreter produces a corrected
transcript or intent while a job is still queued, the scheduler should update
the promoted `transcript`, `transcript_source`, `transcript_confidence`,
`intent`, and relevant metadata before dispatch. The provisional summary may be
retained only as labeled pre-promotion provenance. Late evidence for a running
job should be delivered as a bounded update and audited before irreversible
tool, spend, provisioning, or call actions rely on the earlier request.

The scheduler must distinguish promoted interpreter output from auxiliary
hypotheses inside the same update envelope. A Moshi/OpenClaw/VoiceClaw/classic
ASR source may attach transcript text, confidence, timing, and disagreements to
the queued job, but that text remains `authority = "hypothesis"` and must not
rewrite promoted oracle text, durable transcript, normalized intent, spend
reason, or call/message payload. Only the trusted interpreter promotion source,
or a later oracle judgment, can make the wording action-authoritative.

## Barge-In

Barge-in should be triggered by actual user speech, not by the first decoded packet in a new audio buffer.

Required gates:

- decoded PCM frame energy exceeds configured RMS threshold
- speech-like energy persists for a configured minimum duration
- the active speaker is not the bot
- optional VAD/speech classifier agrees when available
- playback is currently active or oracle job result speech is currently streaming

On barge-in:

1. stop mixer playback within the configured deadline
2. cancel in-flight TTS generation for the interrupted response
3. clear downstream transport/carrier playback buffers when the adapter has
   queued audio beyond Hermes's mixer
4. record the playback cursor so durable transcript can distinguish what was
   spoken from what was only generated
5. request oracle job cancellation only when the user explicitly cancels backend
   work or the job result is no longer useful
6. keep already spoken text as ephemeral history
7. do not commit interrupted assistant text as a full assistant response

Target: playback stop within 150 ms of confirmed speech.

## Latency Budget

The design should measure every turn with monotonic timestamps. Minimum required spans:

- transport receive time to normalized session event
- speech boundary to bounded audio segment ready
- audio segment ready to reflex decision
- audio segment ready to interpreter evidence started/final
- reflex hypothesis to interpreter correction
- transcript evidence spans when Moshi/S2S or ASR transcript evidence is enabled
- oracle job accepted, queued, started, waiting, completed, failed, and cancelled
  lifecycle spans
- interface decision to local first audio
- interface decision to oracle job accepted
- oracle job started to oracle first token
- oracle first token to first TTS audio
- first TTS audio to Discord playback start
- speech boundary to first audible assistant audio
- barge-in speech confirmed to playback stopped
- playback stopped to downstream transport buffer cleared
- user speech end to reflex acknowledgement
- user speech end to oracle job accepted

Targets for the DGX Spark implementation:

- local acknowledgement first audio: under 500 ms from speech boundary
- local complete response first audio: under 1 second from speech boundary
- oracle escalation acknowledgement: under 500 ms from speech boundary
- simple oracle response first audio: under 3 seconds when warm
- tool/context oracle response first audio: under 6 to 8 seconds when warm
- barge-in stop: under 150 ms from confirmed speech

These are product targets, not assumptions. The system must log actual p50/p90 values per provider and model.

## Local Deployment Target Example

The preferred local-first end state is one DGX Spark running the complete
stack. This section is a hardware profile and launch-plan shape, not the KAME
architecture contract and not a validated deployment claim:

```text
Hermes gateway
Realtime voice sidecar / KAME session manager
Fast reflex server
Gemma interpreter server
Oracle LLM server
Optional witness/fallback transcript hypothesis server
Streaming TTS server
Metrics/log collector
```

The oracle should stay warm. Model swapping during an interactive voice session is expected to damage latency more than it helps memory usage.

Preferred first local oracle track:

- vLLM serving Nemotron 3 Super with measured one-Spark settings once evidence
  exists, or the best measured Hermes oracle candidate selected through `/model`
- Gemma 4 26B-A4B benchmarked as a comparison candidate, not as a separate
  realtime voice oracle setting
- OpenAI-compatible endpoint consumed by Hermes's existing model provider path
- fixed context and KV settings chosen for interactive work, not maximum benchmark context

Preferred reflex track:

- Moshi/PersonaPlex-class S2S or a smaller timing/classifier model as the first
  floor-control candidate
- reflex should be optimized for endpoint-to-ack latency and interruption
  behavior, not literal transcript authority
- rough transcript hypothesis captured as early, non-durable context
- text-only fallback through streaming STT only when the realtime reflex is
  unavailable or too unstable; this is degraded compatibility, not full KAME
- the model must be good at barge-in, immediate acknowledgements, concise voice
  responses, and following the Hermes capability contract

Preferred interpreter track:

- Gemma 4 E2B/E4B/12B as the first audio-understanding evidence candidate
- treat Gemma transcript output as the promoted direct-audio interpretation
  result, not as a parallel ASR feed racing Moshi text or the oracle
- raw audio plus frontend-witness/reflex/S2S transcript hypotheses as normal
  input
- optional classic ASR transcript hypothesis as an additional comparison input
- outputs corrected transcript, entities, language notes, confidence, and
  oracle request patches
- must be late-bindable so it can update queued/running oracle jobs without
  blocking the reflex acknowledgement

Preferred speech track:

- use Cartesia or another cloud bridge only as a fallback or
  provider-comparison path while local speech is being validated
- do not use Cartesia STT as primary KAME input, scheduler, or control path
- do not build a second "Gemma ASR" pipeline beside the interpreter; Gemma's
  transcript-like output is part of the interpreter result
- evaluate local transcript hypothesis sources and TTS separately before combining them
- do not feed STT into the reflex in normal full KAME mode
- use ASR as fallback or additional optional witness/fallback transcript evidence for escalated
  turns, not as the realtime interface
- use STT as reflex input only for text-only fallback, explicit debug/audit sessions, or provider comparisons
- do not treat native S2S as the entire agent unless it beats the three-lane
  KAME path on latency, controllability, interpreter evidence quality, and
  interruption behavior

Resource policy:

- reserve memory for the oracle first
- fast reflex second
- Gemma interpreter third
- speech models fourth
- prefer quantized reflex/interpreter/speech components over evicting the oracle
- keep provider processes separately restartable
- fail closed to legacy voice or text when the KAME session cannot start

## Configuration

Add a first-class engine mode:

```toml
[voice.realtime]
engine = "kame_interface_oracle"
transport = "discord"

[voice.realtime.interface]
provider = "openai_compatible"
base_url = "http://127.0.0.1:PORT/v1"
model = "moshi-or-personaplex-reflex"
temperature = 0.2
max_output_tokens = 160
timeout_ms = 800
audio_input = "auto"

[voice.realtime.interpreter]
provider = "openai_compatible"
base_url = "http://127.0.0.1:PORT/v1"
model = "gemma-4-E2B-it"
audio_input = "required"
include_transcript_hypotheses = "when_available"
witness_transcript_policy = "context_only"
prompt_input_order = ["raw_audio", "metadata", "reflex", "transcript_hypotheses"]
timeout_ms = 2000
late_bind_to_oracle_jobs = true

[voice.realtime.transcript_evidence]
mode = "witness_hypotheses"
dedicated_asr_mode = "disabled"
sources = ["reflex", "frontend_witness"]
# Provider names such as "moshi" belong in each hypothesis source field.
ambiguous_frontend_text_kind = "frontend_witness_hypothesis"
# Add "asr" only for explicit fallback, diagnostics, or literal-evidence checks.
attach_to_interpreter_bundle = true
schedule_oracle_from_transcript = false
promote_without_interpreter = false

[voice.realtime.oracle_policy]
mode = "hermes_active_model"
timeout_ms = 60000
max_spoken_sentences = 2

# The local DGX Spark oracle endpoint is registered in Hermes's normal model
# provider config and selected with `/model`, not through realtime voice.
# Do not add an `oracle_model` setting here.

[voice.realtime.oracle_jobs]
enabled = true
max_concurrent = 4
queue_limit = 16
default_priority = "normal"
overflow_policy = "queue"
shutdown_timeout_seconds = 2
speak_terminal_results = true

[voice.realtime.routing]
allow_local_greetings = true
allow_local_clarifications = true
require_oracle_for_tools = true
require_oracle_for_memory = true
require_oracle_for_files = true
local_confidence_threshold = 0.75

[voice.realtime.barge_in]
min_rms = 350
min_speech_ms = 120
stop_playback_deadline_ms = 150

[voice.realtime.input_noise_gate]
enabled = true
min_rms = 350
start_ms = 120
hangover_ms = 320
preroll_ms = 120

[voice.realtime.metrics]
enabled = true
log_turn_spans = true
log_provider_spans = true
```

External frontend compatibility should be configured separately from provider
secrets:

```toml
[voice.realtime.frontend_api]
enabled = false
protocol = "kame_session_v1"
allow_voiceclaw_style_bridge = true
require_auth = true
allow_direct_tools = false
```

When enabled, this API accepts normalized session events and emits the same
reflex/oracle/job stream used by Discord. It is not a second agent loop and it
does not expose Hermes tools directly.

The wire contract is `kame_session_v1`, documented in
`docs/kame-session-v1.md`. Headless smoke artifacts must expose both the
protocol id and the contract path so package audit can detect frontend bridge
drift.

The GUI environment page should expose provider/model/base URL settings for:

- reflex model
- reflex audio input mode
- interpreter model
- interpreter audio input mode
- interpreter late-binding policy
- transcript evidence mode: disabled, from_reflex, on_escalation, speculative,
  debug, or fallback
- transcript evidence authority: hypotheses only; promotion requires
  interpreter or oracle judgment
- local oracle provider registration hints and base URL for Hermes model config,
  while the active oracle still comes from Hermes `/model`
- auxiliary transcript provider/model
- TTS provider/model/voice
- oracle job capacity, queue policy, active/running/waiting/queued counts, and
  cancellation state
- barge-in thresholds
- input noise gate thresholds
- routing policy
- local/cloud fallback behavior

Secrets should stay in environment variables or the existing secrets mechanism, not plain config files.

## Implementation Plan

### Phase 1: KAME-Ready Session Boundary

Refactor the realtime sidecar around the event contract while preserving current behavior.

Deliverables:

- transport-neutral session object
- explicit transcript-hypothesis/provenance events when enabled
- explicit playback lifecycle events
- explicit cancellation tokens
- latency span logging for every stage
- no regression in Discord join, leave, playback, TTS, legacy STT fallback, or
  transcript-hypothesis behavior

### Phase 2: Fast Reflex MVP

Insert a fast reflex between user turns and Hermes oracle calls. The preferred
implementation is a Moshi/PersonaPlex-class S2S or smaller realtime model that
can acknowledge, control floor state, classify local/defer/oracle_direct/clarify
routes, and emit a rough transcript hypothesis. A transcript-fed path is
acceptable only as an explicitly degraded text-only fallback while the realtime
reflex is unavailable or being validated; it cannot satisfy full KAME readiness
or high-risk action gates.

Deliverables:

- reflex prompt and JSON routing schema
- local/defer/oracle_direct/clarify routing
- short local replies through existing TTS path
- structured oracle requests to Hermes
- rough transcript hypothesis surfaced as non-durable evidence
- acknowledgement and oracle-job envelope creation proven not to wait on
  Moshi/open-S2S or classic-ASR hypotheses when raw audio plus route are ready
- tests with fake reflex and fake oracle
- metrics showing which turns avoided the oracle

This phase is a degraded pre-KAME reflex/oracle MVP. It proves the
interface-model boundary and async oracle bridge, but it cannot claim full KAME
readiness and cannot authorize high-risk work from transcript-only or
hypothesis-only evidence. The full three-tier KAME design is not complete until
Phase 3 adds raw-audio Gemma interpretation.

### Phase 3: Gemma Interpreter Evidence Lane

Add Gemma 4 as the non-blocking interpreter over the clipped raw audio plus
frontend-witness, reflex, S2S, and optional classic-ASR hypotheses. It should
produce corrected transcript, entities, language notes, confidence,
disagreement flags, and oracle request patches.

The Moshi/S2S transcript is a companion signal for this interpreter request,
not a replacement for the waveform and not a second path into the oracle. In
normal KAME mode, the interpreter receives both raw audio and any Moshi-style
text the live voice layer produced, then decides whether that text should be
accepted, corrected, rejected, or left as diagnostic-only evidence.

If a provider labels that text as "STT," Hermes still normalizes it as witness
context. The field may be valuable because it captures what the live frontend
believed it heard before Gemma has reviewed the waveform, but it is not the
control path. The interpreter request must put raw audio and timing metadata
first, then reflex state, then Moshi/OpenClaw/VoiceClaw/reflex/classic-ASR
hypotheses. The first transcript-looking string to arrive is useful evidence,
not the user message of record.

Every normalized transcript hypothesis must carry the same non-authoritative
witness contract: `role = "witness_context"`, `authority = "hypothesis"`,
`promotion_required = "interpreter_promoted_or_oracle_promoted"`, and
`tool_authority = false`. The exact fields matter because they are what
runtime status, live evidence, VoiceOps readiness reports, plan-run artifacts,
and package audits use to prove that Moshi/STT-looking text remained context
for Gemma rather than becoming `oracle_text`, durable history, or an action
argument by field name.

If the same frontend can provide only text and no raw audio reference, it is a
degraded compatibility input, not a full KAME turn. The system may preserve that
text as witness context for audit or clarification, but high-risk approval
paths must fail closed until promoted interpreter or oracle evidence exists.
The generated headless artifacts should prove this with a degraded text-only
action-gate case, separate from the primary-audio hypothesis-only and promoted
evidence cases.

Deliverables:

- Gemma interpreter prompt and JSON evidence schema
- raw audio plus reflex/Moshi transcript hypothesis input path
- optional classic ASR hypothesis comparison input when enabled
- corrected transcript/entity output attached to oracle jobs
- late-binding update path for queued/running oracle jobs
- tests proving the interpreter does not block reflex acknowledgement
- evidence comparing oracle outcomes with reflex-only, interpreter, and
  interpreter-plus-auxiliary-transcript evidence
- fixtures proving Moshi/reflex transcript text is supplied to the interpreter
  beside raw audio, remains hypothesis authority, and cannot create a second
  oracle turn

Acceptance tests for this phase should use the canonical envelope from
`docs/kame-session-v1.md`: accepted raw-audio cut, metadata, reflex state, then
`transcript_hypotheses[]`. The passing case must show Moshi/Open-S2S witness
text helping Gemma produce `interpreter_promoted` wording while the raw witness
string stays out of `oracle_text`, durable history, Stripe/NemoClaw reasons,
phone payloads, files, memory, external messages, and direct tool arguments.
The failing cases must cover witness-before-audio, witness-after-interpreter,
text-only degraded compatibility, wrong speaker/channel, low-energy non-speech,
and a hallucinated command from the witness source.

### Phase 4: Streaming Interface Behavior

Let the interface observe VAD state, audio segment lifecycle, and playback state without committing partials. Partial transcripts are available only in debug or fallback modes.

Deliverables:

- early acknowledgement while the oracle warms or thinks
- interruption-aware response cancellation
- prevention of duplicate responses when partials change
- spoken summary of long oracle output
- configurable voice response length policy

### Phase 5: DGX Spark Local Oracle

Prepare Hermes's oracle to run through a local OpenAI-compatible server on the
Spark, with readiness withheld until measured evidence exists.

Deliverables:

- vLLM or SGLang launch profile for Nemotron 3 Super as the first preferred
  Spark-local NVIDIA oracle target
- model-provider registration so Hermes's active `/model` selection, not
  realtime voice config, chooses that endpoint
- warm-start and health-check scripts
- preflight that confirms model, context, and endpoint readiness
- latency comparison against current cloud oracle path
- documented memory and context settings
- proof that hosted Nemotron 3 Ultra or other hosted fallbacks are labeled as
  `/model` fallbacks and not counted as one-Spark readiness evidence

### Phase 6: DGX Spark Local Reflex, Interpreter, And Speech

Move the reflex, interpreter, then transcript evidence/TTS onto the Spark as
resources allow.

Deliverables:

- Moshi/PersonaPlex-class reflex launch profile
- Gemma 4 interpreter launch profile
- local reflex benchmark matrix
- local interpreter benchmark matrix, with Gemma 4 as the default candidate
- local auxiliary transcript benchmark matrix
- local TTS benchmark matrix
- reflex acknowledgement latency comparison
- interpreter correction latency and quality comparison
- auxiliary transcript latency and literal-accuracy comparison
- all-local smoke test
- cloud fallback retained behind config
- one-command launch profile for the full local stack

### Phase 7: Native Realtime Provider Watch

Evaluate native speech-to-speech or live multimodal providers as reflex or
interpreter candidates, not as replacements for Hermes's oracle contract.

Deliverables:

- provider adapter contract
- tool/oracle integration proof
- interruption test
- capability honesty test
- measured latency and evidence-quality comparison

### Phase 8: External Realtime Frontend Bridge

Make VoiceClaw/OpenClaw-style clients first-class frontends for Hermes KAME
without bypassing Hermes authority.

Deliverables:

- KAME session API specification for audio/text input, playback control,
  reflex replies, oracle job submission, status, cancellation, and terminal
  results; current draft: `docs/kame-session-v1.md`
- compatibility map from `ask_brain`/`openclaw_agent_consult` to Hermes oracle
  jobs
- auth and scope rules for external realtime clients
- durable promoted user-turn resume contract: recent promoted turns verbatim,
  older turns summarized, durable ledger remains authoritative, and hypothesis
  fields remain evidence only
- tests proving external clients cannot receive direct Hermes file, shell,
  memory, payment, or provisioning tools
- replay fixture showing Discord and an external frontend preserve one audit id
  across the same VoiceOps task

## Test Plan

Unit tests:

- routing matrix for local/defer/oracle_direct/clarify
- reflex rough transcript hypothesis is non-durable by default
- interpreter evidence schema validation
- interpreter compares raw audio, reflex/Moshi transcript hypotheses, and ASR
  hypotheses without treating any one as automatic truth
- Moshi/OpenClaw/VoiceClaw/reflex transcript text is normalized as
  `transcript_hypotheses[]` even when the provider labels it "STT"
- oracle request contains distinct reflex intent, interpreter evidence, and
  auxiliary transcript fields
- reflex JSON schema validation
- oracle request construction
- interrupted response commit behavior
- barge-in RMS and duration gates
- fallback path when interface provider is unavailable
- config scoping and defaults

Integration tests:

- fake Discord audio input to bounded audio segment to local reply
- fake Discord audio input to reflex acknowledgement to interpreter evidence to
  oracle job request
- fake external KAME frontend input to oracle job request to spoken/status events
- external frontend `ask_brain` compatibility path creates a Hermes oracle job
  and never exposes direct tools
- `defer` submits a background oracle job and returns acknowledgement promptly
- `oracle_direct` submits a background oracle job without blocking the next
  reflex turn
- `max_concurrent=4` starts four jobs and queues the fifth
- `/voice status` includes oracle job active/running/queued/waiting capacity and
  state
- late output from cancelled jobs is not spoken or committed
- sidecar unavailable fallback
- TTS unavailable fallback
- realtime reflex unavailable degraded fallback to transcript-fed routing; this is
  pre-KAME compatibility and must not be presented as full KAME evidence
- fail-closed sidecar failure after oracle-job acceptance emits `SESSION_ERROR`
  and drains/cancels active oracle jobs with the configured shutdown timeout
- escalated turn includes interpreter evidence and optional frontend-witness,
  reflex, S2S, or classic-ASR hypotheses when configured
- interpreter evidence can patch a queued/running oracle job without blocking
  the immediate reflex acknowledgement
- oracle timeout with spoken status
- cancellation during oracle stream
- cancellation during TTS playback
- unflagged high-risk oracle tool calls/results fail closed, including nested
  `function.name` event payloads, while low-risk oracle tool progress still
  streams normally

Manual smoke tests:

- `/voice join` starts KAME mode when all dependencies are healthy
- `/voice status` reports reflex, interpreter, oracle, transcript evidence,
  TTS, and fallback state
- greeting is answered locally
- project/tool question escalates to Hermes oracle
- user speech during bot playback stops audio quickly
- interrupted assistant response is not committed as complete
- local/cloud fallback reason is visible and accurate

Production review must also include KAME-specific evidence checks. A generic
voice production review is not enough for this branch. Required KAME gates
prove the role contract first; hardware-specific benchmark evidence is attached
as local deployment evidence:

- local deployment benchmark evidence accepted by the generated KAME matrix
  validator for the selected runtime
- Moshi/PersonaPlex-class reflex or equivalent fast floor-control launch
  evidence from the selected runtime
- Gemma 4 interpreter launch evidence from the selected runtime
- evidence that Gemma interpreter corrects or confirms reflex transcript
  hypotheses from raw audio before tool-critical oracle work
- evidence that Moshi/OpenClaw/VoiceClaw transcript text, when present, is
  attached to the same raw-audio interpreter packet as context rather than sent
  as a second Hermes turn
- evidence that every transcript hypothesis is normalized with
  `role = "witness_context"`, `authority = "hypothesis"`,
  `promotion_required = "interpreter_promoted_or_oracle_promoted"`, and
  `tool_authority = false`
- Nemotron 3 Super evaluated as the preferred Spark-local oracle target selected
  through Hermes `/model`
- `max_concurrent=4` measured against the Nemotron 3 Super endpoint, or
  explicitly marked as needing evidence
- hosted Nemotron 3 Ultra excluded from one-Spark readiness claims unless local
  evidence proves otherwise
- oracle outcome comparison with reflex-only, Gemma interpreter, and
  Gemma-plus-optional-auxiliary-transcript evidence
- all-local DGX Spark smoke with oracle, reflex, raw-audio interpreter, TTS,
  and sidecar together when claiming one-Spark readiness; auxiliary transcript
  evidence is optional comparison or fallback evidence when enabled
- async KAME VoiceOps proof coverage for single-bundle witness fusion,
  interpreter prompt ordering/policy, unpromoted hypothesis action-sink
  rejection, runtime KAME action gates, unflagged high-risk tool fail-closed
  handling, and KAME first-audio latency metrics
- live Discord smoke for the full KAME path under production credentials

The `kame_dgx_benchmark_evidence` production-review check must reference a
local JSON artifact from the DGX Spark benchmark validator with `ok=true` and
passing coverage for the required KAME matrix rows and async KAME VoiceOps proof
keys.

## Acceptance Criteria

The full implementation is acceptable when:

- a lightweight interface model is actually in the live path
- a Gemma interpreter/evidence lane can process raw audio plus reflex transcript
  hypotheses without blocking immediate acknowledgements
- the oracle is not called for simple local turns
- all tool, file, memory, and project questions still go through Hermes oracle authority
- the system never tells the user it lacks voice when voice is active
- barge-in responds to real speech energy, not silent packet arrival
- local acknowledgements are consistently fast
- oracle latency is measured and visible instead of guessed
- oracle job capacity reports active, running, queued, and waiting-for-approval
  work separately
- sidecar shutdown leaves no orphan sessions or playback
- Discord fallback is explicit and understandable
- all reflex, interpreter, optional witness/fallback transcript, TTS, routing, fallback, and
  local-provider target choices are configurable from config and GUI
- Hermes oracle model selection remains the existing `/model` mechanism, not a separate realtime voice setting
- the full stack has a documented one-DGX-Spark launch path, with readiness
  claims gated on measured evidence rather than the existence of the docs

## Current Gap Summary

Already present:

- Discord voice join/leave path
- realtime sidecar path
- streaming speech provider bridge path for fallback and bring-up
- Cartesia and ElevenLabs-style provider configuration for fallback,
  comparison, and outbound speech experiments
- mixer playback path
- speech-energy-gated barge-in
- fallback to legacy behavior when sidecar startup fails
- focused realtime voice tests
- early latency measurement logs
- KAME interface/oracle engine in the live session path
- structured interface-to-oracle request contract
- local/defer/oracle_direct/reject_or_clarify routing policy
- optional witness/fallback transcript-hypothesis lane for escalated turns
- ephemeral versus durable transcript policy at the session boundary
- oracle hint streaming back to live interface providers
- explicit KAME provenance for realtime-reflex, interpreter, hypothesis
  witnesses, and degraded text-only fallback turns
- visible degraded frontend fallback state when raw audio or the realtime
  reflex/interpreter path is unavailable
- DGX Spark launch/profile generation for reflex, interpreter, oracle,
  auxiliary transcript, and TTS targets
- benchmark matrix templates for local reflex, interpreter, and speech candidates
- GUI coverage for KAME reflex, interpreter, auxiliary transcript, TTS,
  routing, barge-in, fallback, and local provider target settings
- oracle job manager evidence for background execution, queueing, cancellation,
  and status reporting

Remaining for full KAME production readiness:

- Moshi/PersonaPlex-class or equivalent local reflex launch evidence from the
  actual target runtime
- Gemma 4 interpreter launch evidence from the actual DGX Spark runtime
- direct-audio interpreter evidence showing raw audio first, metadata/reflex
  second, and Moshi/STT text only as optional `transcript_hypotheses[]`
- witness-assisted direct-audio evidence showing Moshi/reflex text included in
  Gemma context with raw voice, plus accepted/corrected/rejected adjudication
- DGX Spark / Nemotron 3 Super `max_concurrent=4` capacity evidence
- benchmark evidence comparing reflex-only, Gemma interpreter, and Gemma plus
  optional witness/fallback transcript evidence in oracle outcomes
- benchmark evidence comparing interpreter correction against reflex transcript
  hypotheses for multilingual/code-switched turns
- all-local DGX Spark smoke evidence with the oracle, reflex, raw-audio
  interpreter, and TTS services running together; optional witness/fallback transcript evidence
  remains optional comparison or fallback evidence
- live Discord smoke evidence for the full KAME path under production credentials
