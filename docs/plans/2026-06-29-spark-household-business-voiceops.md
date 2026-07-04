# Spark Household and Business VoiceOps Goal

## North Star

Run a household and small business from Hermes VoiceOps, conversationally, with
DGX Spark as the preferred local-first compute target.

Hermes should become a local-first operating layer that can listen, decide, act, spend, provision, monitor, and escalate across the channels the user already lives in. The first live control surface is Discord voice. WhatsApp and phone/SMS are the next operational surfaces. The Spark is the preferred private compute base, not the voice architecture itself. Stripe is the controlled spend and provisioning rail.

This is not a voice feature. Voice is the control plane. The product is trusted operational agency.

## Product Thesis

Hermes VoiceOps is a local-first operator for daily life and business:

- Household: bills, subscriptions, maintenance, calendar conflicts, errands, urgent alerts, home services.
- Business: customer ops, vendor setup, SaaS provisioning, recurring reviews, payments, reporting, and incident response.
- Voice surfaces: Discord live voice for the desk, WhatsApp for mobile chat, phone/SMS for urgent fallback.
- Compute: DGX Spark is the preferred local deployment target for the KAME
  reflex, speech stack, and local models where practical. The architecture is
  still role-defined: reflex, Gemma-style interpreter, and Hermes active
  `/model` oracle.
- Reasoning: Nemotron 3 Super is the preferred Spark-local NVIDIA candidate for
  Hermes's active oracle, while Hermes `/model` remains authoritative. A clearly
  labeled hosted `/model` fallback is acceptable only when the local Spark path
  is unavailable, and no VoiceOps-specific `oracle_model` selector should exist.
- Safety: NemoClaw is the preferred sponsor-aligned execution boundary for spend, provisioning, and network-capable action packets.
- Spend: Stripe Link, Stripe Projects, and MPP/402 become the controlled path for paying, provisioning, and recording approvals.
- Audit: every planned, approved, held, executed, failed, or rolled-back action is durable and inspectable.

The hackathon entry is Milestone 0: a public proof that this shape is useful, viable, and presentable.

## Current KAME Invariant

The VoiceOps architecture is now fixed around three roles:

- reflex: the always-warm voice interface that handles floor control,
  VAD/energy gating, barge-in, immediate acknowledgement, and provisional
  route/status narration
- interpreter: Gemma-style direct-audio adjudication over one accepted speech
  cut, receiving raw audio first and transcript hypotheses last
- oracle: Hermes' active `/model`, selected through the normal Hermes model
  interface, with authority over durable reasoning, tools, memory, spend, phone
  calls, and external messages

Moshi/Open-S2S "STT" is not a fourth role. When available, it is a witness
hypothesis attached to the same raw-audio interpreter bundle. It can help Gemma
recover clipped starts, names, numbers, and code-switched wording, but it cannot
create a second Hermes turn, become `oracle_text`, or authorize Stripe,
NemoClaw, phone, file, memory, message, or tool payloads before interpreter or
oracle promotion.

Current amendment, 2026-07-05: provide Moshi/Open-S2S transcript output to the
Gemma interpreter with the raw voice when both refer to the same accepted speech
cut. That is the intended witness-context path. It is not an ASR-first path.
The demo and implementation should show the user getting an immediate reflex
acknowledgement, then one raw-audio interpreter bundle, then promoted evidence
flowing into the active Hermes `/model`.

For VoiceOps actions, this distinction is a safety requirement. A Moshi,
OpenClaw, VoiceClaw, reflex, or classic-ASR string can help Gemma interpret a
service name, dollar amount, phone number, approval phrase, or code-switched
instruction. It cannot itself become a spend reason, provider choice, phone
script, account-provisioning payload, memory/file write, external message, or
tool argument. Headless and live artifacts must prove that unpromoted witness
text stayed out of those sinks even when it helped the interpreter produce
promoted wording.

## Spark Model Strategy

There are two related but distinct model strategies.

### Hackathon Strategy

Use Nemotron 3 Super visibly as the preferred Spark-local Hermes oracle candidate
for the demo because it is sponsor-aligned, has a documented candidate
one-Spark serving path, and communicates serious agentic reasoning on the
NVIDIA target. The demo should show Nemotron 3 Super as the planner behind the
budgeted VoIP provisioning workflow when the local Spark path is available,
without claiming that local deployment has been validated.

This does not mean VoiceOps adds a separate model selector. Nemotron 3 Super
should still be selected through Hermes's normal `/model` flow. If the local
Super endpoint is not ready, a hosted model is acceptable only as a clearly
labeled `/model` fallback. The goal is to prove the VoiceOps workflow shape:
Hermes can carry a Discord voice request into a safe, budgeted, tool-using
business operation with NVIDIA/Stripe integrations visible. One-Spark readiness
still requires measured local Spark benchmark evidence.

Hackathon stack:

- Nemotron 3 Super as the preferred Spark-local serious reasoning/planning path
- clearly labeled hosted `/model` fallback if Super local serving is not ready
- NemoClaw as the safe execution boundary
- Stripe Skills as the spend and provisioning rail
- Discord voice as the live interface
- DGX Spark as the target local operating base and NVIDIA story

### One-Spark Local Strategy

For the long-term household/business appliance, Nemotron 3 Super is the preferred Spark-local NVIDIA oracle/model target, but still require benchmark evidence before claiming one-Spark readiness. One-Spark readiness still requires measured local Spark benchmark evidence. A clearly labeled hosted `/model` fallback is acceptable only when the local Spark path is unavailable. Nemotron 3 Ultra is not the one-Spark target; keep it as hosted or future multi-Spark context unless local evidence proves otherwise.

Target KAME layout:

- Source of truth: the voice architecture is reflex -> Gemma interpreter ->
  Hermes active `/model` oracle. Moshi/Open-S2S transcript-looking output may
  be sent to Gemma beside the raw voice as witness context, but it is not the
  user's durable message and not an action-authority source. Classic ASR is a
  fallback/diagnostic/literal-evidence support lane, not a required normal-path
  gate.
- Reflex/interface: the fastest stable always-warm live-audio path available,
  such as Moshi/PersonaPlex-class S2S or an even smaller timing/noise-gated
  classifier path, optimized for turn-taking, barge-in, immediate
  acknowledgement, and rough transcript hypotheses. This model owns live floor
  control, not tool execution or durable transcript truth. Its first
  responsibility is a calibrated energy/noise gate: silence, room tone, Discord
  receiver artifacts, and low-energy non-speech frames must not create barge-in
  events, interpreter requests, or durable transcript candidates.
- Interpreter/evidence: Gemma 4 E2B/E4B/12B-style audio-multimodal model,
  run non-blocking after each speech cut to adjudicate raw audio plus
  frontend-witness/reflex/S2S/classic-ASR hypotheses into corrected
  transcript, multilingual intent, entities, confidence, and oracle request
  patches. Raw audio is the primary signal; Moshi/S2S and classic ASR
  transcripts are labeled hypotheses supplied as context in the same
  interpreter evidence bundle, not prerequisites, separate turns, or authority.
- Oracle/brain: whatever Hermes `/model` selects, with Nemotron 3 Super as the
  first preferred local NVIDIA candidate to evaluate on DGX Spark. VoiceOps
  submits oracle jobs to the active Hermes model; it does not configure a
  separate oracle model.
- Speech: local transcript hypothesis evidence and TTS where practical, with
  Moshi/S2S or classic ASR transcripts used as witness/fallback context for
  the interpreter rather than reflex input in full KAME mode. The oracle may see
  promoted wording or labeled audit context, but raw transcript hypotheses do
  not carry action authority. The system must not require ASR evidence before
  acknowledging or submitting work when the raw-audio/reflex path is available.
- Sensor fan-in: the normal voice turn is not "Moshi STT plus Hermes" and not a
  separate ASR conversation. It is one clipped raw-audio turn, one reflex route,
  and one interpreter bundle. Moshi/open-S2S, VoiceClaw/OpenClaw, reflex, and
  classic ASR text all enter as witness hypotheses when available.
- Three-tier pivot: use a fast reflex for live floor control, Gemma for
  direct-audio interpretation of the accepted speech cut, and Hermes' active
  `/model` as the oracle. A Moshi/open-S2S transcript may be provided to Gemma
  with the raw voice as same-bundle context, but it must not become a separate
  STT lane, a second Hermes turn, or a durable prompt. This is the preferred
  way to use Moshi-style text: it tells Gemma what the frontend believed it
  heard while raw audio remains the higher-authority signal.
- Evidence: a speech cut should preserve raw audio, reflex hypothesis,
  Moshi/S2S hypothesis, optional ASR hypothesis, and interpreter correction as
  distinct provenance-labeled fields. When raw audio is available, only
  raw-audio-grounded interpreter evidence can promote a transcript hypothesis
  into durable user text or tool-critical arguments for the oracle to act on.
- Degraded compatibility: if a VoiceClaw/OpenClaw/Moshi-style frontend can
  provide only text and no raw audio reference, Hermes may still draft,
  clarify, or produce status in degraded mode, but the turn must be marked
  degraded. Text-only witness content must not become full-KAME durable user
  text or any action source without raw-audio-grounded interpreter promotion.
  It cannot satisfy full KAME readiness, Stripe/NemoClaw/phone action gates, or
  promoted transcript evidence.
- Merge point: the Gemma interpreter request is the normal merge point for raw
  audio plus transcript hypotheses. Moshi/S2S transcript output should be
  supplied as context beside the waveform, not as a replacement for the waveform
  and not as a second oracle prompt. Classic ASR uses the same path only when
  enabled for fallback, diagnostics, captions, or literal-evidence checks. A
  Moshi transcript that arrives without a matching raw-audio segment is useful
  compatibility evidence, but it is not the normal path and cannot prove full
  KAME behavior.
- External frontend packet shape: VoiceClaw/OpenClaw/Moshi-style bridges should
  send raw-audio references and timing when available, plus any transcript-like
  strings as labeled hypotheses. A text-only `ask_brain` bridge is useful for
  compatibility, but it is degraded evidence and does not prove the full KAME
  raw-audio interpreter path.
- Promotion rule: no Moshi/S2S, VoiceClaw/OpenClaw, or classic ASR transcript
  can become durable user text, `oracle_text`, a tool argument, a spend reason,
  or a call/message payload without interpreter or oracle judgment. The
  transcript that arrives first is never automatically the transcript of
  record. Headless VoiceOps artifacts must expose sink-specific checks showing
  unpromoted witness text is absent from spend, phone, NemoClaw, tool, memory,
  file, external-message, and durable-history action payloads.
- Persistence rule: this promotion rule applies to both visible chat history and
  internal oracle recovery records. Hypothesis strings may live in the
  voice-session audit ledger with provenance, but they must not be replayed as
  verified user messages or durable Hermes conversation turns.
- Three-tier rule: the long-term shape is reflex, interpreter, oracle. A
  two-tier reflex/oracle bridge is acceptable as a bring-up step, but it must not
  pretend that a Moshi/open-S2S transcript is equivalent to Gemma raw-audio
  interpretation. The interpreter bundle is where raw voice and transcript-like
  side channels are compared.
- Evidence-bundle rule: a Moshi/open-S2S transcript can and should be provided
  to Gemma alongside raw voice when available, but it is context, not control.
  Gemma's role is interpreter/evidence adjudicator, not a blocking ASR service.
  The reflex may propose a queue envelope and narrate work from provisional
  intent, while durable user wording and tool-critical fields require
  raw-audio-grounded interpreter promotion before the oracle treats them as
  action evidence.
- No-ASR-gate rule: the normal KAME path must not require ASR evidence before
  acknowledgement, job creation, or raw-audio interpretation. Dedicated ASR is
  fallback, diagnostics, captions, or high-risk literal-evidence support. Moshi,
  VoiceClaw/OpenClaw, and ASR transcript text all enter the same bundle as
  hypotheses, and none of them can become the user message of record by arriving
  first.
- Moshi-context rule: if a frontend exposes "Moshi STT" or another
  transcript-looking field, store the vendor name as source metadata and store
  the text as a hypothesis attached to the same raw-audio bundle. Do not fork a
  second Hermes turn, do not overwrite the raw-audio route, and do not let that
  text become a spend reason, phone-call payload, memory entry, file write, or
  durable chat message unless Gemma promotes raw-audio-grounded evidence for
  the active Hermes oracle.
- Witness-context rule: Moshi/OpenClaw/VoiceClaw transcript output is witness
  testimony from the realtime frontend, not the verified user utterance. Gemma
  should receive that witness text beside the clipped waveform, speaker/channel
  metadata, VAD/energy timing, reflex route, and spoken acknowledgement. The
  interpreter may use it to recover clipped prefixes, names, numbers,
  code-switches, or commands, but must also be free to reject it as hallucinated
  or wrong-speaker evidence.
- External frontend tool-boundary rule: VoiceClaw/OpenClaw/Moshi-style
  frontends are consult bridges only. They may submit `ask_brain`,
  `ask_hermes_oracle`, `agent_consult`, or `openclaw_agent_consult` envelopes
  with raw-audio evidence, witness text, reflex status, and correlation ids.
  They must not call Stripe, NemoClaw, phone, file, shell, memory, message,
  credential, or provider-provisioning tools directly. Unsupported tool names
  must be rejected with an auditable `tool.result` and must not create an
  oracle job.
- Witness-adjudication rule: each witness transcript should receive an
  interpreter outcome before it can influence durable/actionable wording:
  `accepted_as_supporting_evidence`, `corrected_by_audio`, or
  `rejected_or_diagnostic_only`. Only accepted or corrected witness text may
  contribute to `interpreter_promoted` fields, and only after Gemma has compared
  it against the raw waveform and metadata. Rejected/diagnostic witness text may
  stay in the audit bundle, but it must not become a spend reason, phone script,
  provider choice, file/memory content, durable user text, or tool argument.
- Witness-assisted interpreter rule: when Moshi/open-S2S produces an STT-like
  transcript for a speech cut, use it to help Gemma interpret the same clipped
  raw voice. Do not run Moshi text as a second Hermes conversation and do not
  send it to the oracle as the user message. The interpreter request should show
  raw audio first and the Moshi/OpenClaw/VoiceClaw/classic-ASR strings as
  labeled `transcript_hypotheses[]`; the interpreter response should show which
  hypotheses were accepted, corrected, rejected, or kept diagnostic-only.
- Three-tier sensor-fan-in decision: the product architecture is reflex,
  interpreter, oracle. It is not reflex plus a separate Gemma-ASR lane plus the
  oracle, and it is not Moshi-STT driving Hermes. The reflex may hear quickly
  and offer a witness. Gemma judges the accepted raw-audio cut with that witness
  as context. Hermes' active `/model` receives only promoted wording/intent plus
  compact labeled audit context before doing business work.
- Reflex-witness rule: if the low-latency reflex model itself emits
  transcript-looking text, treat it exactly like other witness text. It can
  explain the reflex route and help Gemma repair clipped starts or code-switched
  terms, but it is not durable user text and must not bypass the same raw-audio
  interpreter promotion boundary.
- Moshi-context acceptance rule: the artifact should prove that a Moshi/open-S2S
  transcript can be attached to the Gemma interpreter packet beside raw voice
  without becoming a second turn. Cover witness-before-cut, witness-with-cut,
  and witness-after-interpreter-start. All cases must preserve the same
  `turn_id`, `audio_segment_ref`, `evidence_bundle_id`, and
  `evidence_merge_key`, and must prove no duplicate oracle job or durable user
  message. A positive case should show Gemma promoting corrected wording after
  comparing raw audio and Moshi text. A negative case should show hallucinated,
  wrong-speaker, wrong-channel, stale, or low-energy witness text retained only
  as audit evidence and absent from spend, phone, NemoClaw, tool, memory, file,
  external-message, and durable-history sinks.
- Moshi-context metadata rule: every Moshi/Open-S2S witness row in the package
  must expose provider `source`, latency, confidence when available,
  `arrival_phase`, speaker/channel guesses when available,
  `role = "witness_context"`, `authority = "hypothesis"`,
  `promotion_required = "interpreter_promoted_or_oracle_promoted"`, and
  `tool_authority = false`. The package should also show the interpreter
  outcome for that row: `accepted_as_supporting_evidence`,
  `corrected_by_audio`, or `rejected_or_diagnostic_only`.
- Degraded-witness rule: if a Moshi/Open-S2S frontend provides text without a
  matching waveform, the packet can be retained as compatibility evidence but
  must fail full-KAME readiness and high-risk action gates. It may support a
  clarification or status response, but it cannot become a spend reason,
  provider choice, phone script, memory/file content, durable user message, or
  tool argument without later raw-audio-grounded interpreter/oracle promotion.
- Witness-rejection evidence rule: every `rejected_or_diagnostic_only`
  hypothesis must preserve typed `rejection_reasons[]`. At minimum, artifacts
  should accept `ambiguous_speaker`, `wrong_speaker`, `wrong_channel`,
  `stale_witness`, `timing_conflict`, `low_energy_non_speech`,
  `waveform_conflict`, and `provider_conflict`. A rejected Moshi/OpenClaw/
  VoiceClaw/reflex/classic-ASR witness without one of those concrete reasons is
  incomplete evidence, even if the rejected text stayed out of action sinks.
- Same-turn convergence rule: if a Moshi/OpenClaw/VoiceClaw/reflex packet
  creates provisional queue state before the raw voice artifact is ready, the
  later raw-audio packet must update that same job rather than submit a second
  oracle request. The proof should show one durable KAME request, one
  accepted/started/completed oracle lifecycle, a raw-audio-aware
  `evidence_merge_key`, and an update event identifying the evidence-bundle
  merge.
- Oracle-context rule: the active Hermes `/model` should receive promoted
  transcript/intent/entities plus compact labeled audit context. It should not
  receive unpromoted Moshi/STT witness strings as if they were the durable user
  prompt, because that recreates the old STT-first failure mode under a new
  name.
- Alternative-provider rule: Gemini Live-style hosted realtime APIs, Moshi,
  Ultravox, Qwen Omni, Nemotron/Riva ASR, Magpie/Riva TTS, Piper, Cartesia, and
  similar systems are provider candidates, not authority models. Each candidate
  must be assigned to one role: reflex, interpreter, auxiliary transcript
  evidence, outbound TTS, or degraded fallback. None may bypass the
  promoted-evidence rule.
- Sensor-fan-in rule: the long-term stack should collect observations for one
  speech cut rather than run multiple user turns in parallel. The fast reflex
  owns live floor control, Moshi/open-S2S transcript text records what that
  frontend believed it heard, optional classic ASR is fallback or diagnostic
  evidence, and Gemma receives those hypotheses beside the clipped waveform.
  The first transcript to arrive is not the transcript of record.
- Noise-gated raw-audio rule: every full-KAME turn should expose the energy gate
  decision that produced the speech cut. Headless artifacts should show that the
  system ignored silence/non-speech packets, preserved the selected
  `audio_segment_ref`, and sent that bounded raw segment to the interpreter
  before treating any Moshi/S2S/ASR text as actionable evidence.
- Task-state rule: the reflex needs a compact job-status projection with safe
  capacity counts, job ids, states, priorities, and ordinal-friendly spoken
  labels. It must name at least the first four active oracle jobs and a queued
  fifth job so the user can say "cancel the fourth one" or "make job five high
  priority" without giving the reflex raw transcript hypotheses, speaker/channel
  metadata, hidden reasoning, full oracle outputs, or tool traces.
- Interpreter prompt rule: raw audio and timing metadata are the primary
  interpreter input. Moshi/open-S2S and classic-ASR text must be passed in a
  labeled hypotheses field with source, timing/confidence when available,
  partial/final state, and `authority = "hypothesis"` so Gemma can accept,
  correct, or reject it instead of silently treating it as the user message.
- Interpreter prompt policy rule: every complete raw-audio KAME bundle should
  expose `interpreter_input_order = ["raw_audio", "metadata", "reflex", "transcript_hypotheses"]`
  and `interpreter_prompt_policy.version = "raw_audio_compare_v1"`. This makes
  the Moshi-context decision auditable: Moshi/Open-S2S text is sent beside raw
  voice as a clue for Gemma, not ahead of raw voice as a transcript prompt and
  not around Gemma as a separate Hermes turn.
- Witness role-marker rule: every normalized transcript hypothesis should carry
  the exact witness-context contract: `role = "witness_context"`,
  `authority = "hypothesis"`,
  `promotion_required = "interpreter_promoted_or_oracle_promoted"`, and
  `tool_authority = false`. This lets package audits prove that vendor text was
  included for interpretation without being smuggled into `oracle_text`, spend
  reasons, provider choices, phone scripts, memory/file writes, external
  messages, tool arguments, or durable history.
- Bundle schema rule: every speech cut creates one interpreter evidence bundle
  keyed by `turn_id` and `audio_segment_ref`. The bundle carries primary audio,
  VAD/energy timing, speaker metadata, channel/transport metadata, reflex route
  and acknowledgement, and a `transcript_hypotheses[]` list for Moshi/open-S2S,
  VoiceClaw/OpenClaw, reflex, and classic ASR text. Speaker/channel metadata is
  canonical evidence, not optional decoration, because it is needed to reject
  ambiguous-speaker transcripts, wrong-speaker transcripts, cross-channel
  replay, stale captions, and misattributed approvals. The implementation
  target is the canonical contract in
  `docs/design/full-kame-style-realtime-voice.md`; duplicated text from any
  transcript side channel must attach to that bundle, not spawn a new Hermes
  turn.
- Multi-speaker rule: Discord voice, phone bridges, and future WhatsApp voice
  sessions must treat speaker/channel attribution as part of the authority
  boundary. A transcript hypothesis from a second human in the same call may be
  shown as audit context, but it cannot update the accepted user's durable
  transcript, oracle request, approval packet, Stripe spend reason, phone
  payload, memory/file content, or tool arguments unless the interpreter
  promotes it from the matching raw-audio cut. Wrong-speaker, wrong-channel,
  stale, and ambiguous-speaker witnesses should fail closed with typed rejection
  reasons.
- Witness-fusion acceptance rule: when raw audio and Moshi/OpenClaw/VoiceClaw
  witness text are both present, artifacts should prove a single stable
  `evidence_bundle_id`, a single interpreter merge path, and no duplicate oracle
  job. If witness text arrives early, it waits on the pending bundle; if it
  arrives late, it is late evidence on the same bundle. If raw audio is missing,
  the turn is degraded and cannot close Stripe, NemoClaw, phone, memory, file,
  or external-message action gates on hypothesis text alone.
  Each witness should retain `arrival_phase` as one of `before_raw_audio`,
  `with_raw_audio`, or `after_interpreter_start`, and that phase must remain
  visible in runtime job status, bounded oracle-job updates, live evidence, and
  package-audit output.
  Partial witness text is active only until a final same-source/same-kind
  witness arrives for the same speech cut. The final witness should become the
  only active frontend hypothesis, while the partial survives only as
  superseded-partial provenance for audit and debugging.
  Headless acceptance requires `witness_fusion_timing_preserves_single_bundle`
  to pass for early, inline, and late witness arrival, with the same bundle id
  surviving degraded-to-primary audio updates. It must also prove the positive
  path: a clipped reflex transcript such as "three to the power of seventeen"
  plus an early Moshi/OpenClaw/VoiceClaw witness such as "what is three to the
  power of seventeen" can become durable only after Gemma promotes
  `interpreter_corrected_transcript`, `interpreter_normalized_intent`, and any
  entities such as `3^17`. The witness text remains hypothesis authority even
  when it helped; the promoted interpreter fields are the durable evidence.
- Moshi-context acceptance matrix: headless proof should cover four cases for
  transcript-looking frontend output. First, helpful witness text attached to
  raw audio improves interpreter recovery without becoming authority. Second,
  wrong, stale, low-energy, wrong-speaker, or wrong-channel witness text is
  rejected with typed reasons while remaining visible in the audit bundle.
  Third, witness text that arrives before or after the waveform merges into the
  same evidence bundle and does not create another Hermes turn. Fourth,
  text-only Moshi/OpenClaw/VoiceClaw compatibility mode reports
  `degraded_text_only`, preserves the hypothesis, and fails high-risk action
  gates until raw-audio-grounded evidence exists.
- Three-tier sensor fan-in rule: the long-term architecture remains reflex,
  interpreter, oracle. A very fast Moshi/OpenClaw/VoiceClaw-style frontend is
  the reflex candidate and may emit transcript-looking text, but that text is
  witness context for Gemma, not a separate STT turn. Gemma receives the clipped
  waveform plus the witness text in one interpreter packet. Classic ASR, when
  enabled, uses the same hypothesis lane for fallback, diagnostics, captions,
  or literal-evidence checks. Hermes' active `/model` receives only promoted
  wording, intent, entities, and compact labeled evidence.
- Accepted-cut evidence rule: headless and live artifacts must expose the
  accepted speech cut's `audio_segment_ref`, time range, VAD decision, and
  energy/noise-gate decision. The artifact must show that transcript hypotheses
  attached after that raw-audio gate rather than serving as the gate. Silence,
  room tone, harmonic artifacts, and low-energy non-speech packets must not
  create barge-in, interpreter requests, oracle jobs, durable transcript
  candidates, or high-risk action evidence.
- Live-evidence gate rule: transcript-only witness evidence must not close the
  full KAME live voice gate. A Moshi/OpenClaw/VoiceClaw/STT string without raw
  audio segment evidence and interpreter evidence is useful for audit,
  clarification, captions, or degraded fallback, but it cannot prove Hermes
  heard the user or satisfy Stripe, NemoClaw, phone, memory, file, or external
  message readiness.
- Action-authority rule: Stripe spend, provider provisioning, NemoClaw action
  packets, phone-call payloads, memory writes, file writes, and external
  messages require `interpreter_promoted` or `oracle_promoted` evidence for the
  action text and rationale. Hypothesis-only evidence can prepare a draft,
  request clarification, or populate an audit trail, but it cannot authorize
  irreversible work.
- NemoClaw action-contract rule: approval packets must validate against a
  static allowlist of known action ids, providers, command shapes, required
  preflight gates, and approval artifacts. A packet that invents a new action or
  swaps in a shell-like command is invalid even if its internal command hashes
  and approval contracts are self-consistent.
- Channel route-payload rule: every Discord, WhatsApp, SMS, phone, spend,
  provisioning, credential, and status route must declare its payload policy,
  allowed payload classes, payload digest requirement, and raw-witness-text
  prohibition. High-risk spend, provisioning, credential, and account-mutation
  routes must deny channel egress by default; they can emit only blocked-intent
  or operator-escalation evidence, not customer-visible payloads.
- Live witness-metadata rule: every `transcript_hypotheses[]` item in live KAME
  evidence must include `text_digest`, `role = witness_context`,
  `authority = hypothesis`, `promotion_required =
  interpreter_promoted_or_oracle_promoted`, `tool_authority = false`,
  `arrival_phase`, `latency_ms`, `confidence`, `speaker_or_actor_ref`, and
  `channel_or_surface_ref`. A hypothesis that lacks this metadata is not valid
  full-KAME live evidence, even if raw audio and interpreter fields are present.
  The top-level `interpreter_adjudication_outcomes` set must exactly match the
  per-hypothesis adjudications so an artifact cannot summarize a rejected
  witness as accepted.
- Operator-state rule: pending approval records must carry the same promoted
  KAME evidence used by the NemoClaw packet, plus a reference to the artifact's
  tool-disclosure proof. The operator-state artifact itself must include that
  tool-disclosure proof so a dashboard or GUI cannot silently lose the context
  pressure guard. A dashboard or GUI approval row that drops this evidence is
  invalid even if the visible command, budget, and provider fields look correct.
- Runtime approval-gate rule: live async oracle jobs should expose the same
  boundary as `voiceops.runtime_kame_action_gate.v1` on approval waits and
  approval-related tool progress, failing closed until promoted evidence was
  consumed before the irreversible action boundary and `tool_disclosure_ref =
  "tool_disclosure"` is present.
  Headless acceptance requires `runtime_kame_action_gate_enforced` to prove
  hypothesis-only approvals fail closed, degraded text-only frontend approvals
  preserve witness text while failing closed, and consumed promoted interpreter
  evidence passes.
- Tool-pressure rule: high-risk action artifacts should also prove that broad
  Hermes tools were not carried through the live voice context unnecessarily.
  The VoiceOps package should record the `tool_search`/bridge-tool deferral
  proof, especially that core tools are hidden behind discovery until the active
  Hermes oracle actually needs them.
- Tool-pressure acceptance rule: the headless VoiceOps artifacts must prove
  voice-scoped `tools.tool_search.defer_core = all` against the full Hermes core
  tool list, not only a small sample. The proof should show only
  `tool_search`, `tool_describe`, and `tool_call` as model-visible bridge tools,
  zero non-bridge core tools visible, all core tools hidden, and a positive
  estimated schema-token reduction.
- Fallbacks: hosted `/model` providers, Kimi, Cartesia, or other cloud providers are acceptable during bring-up and demos when they are labeled clearly.

The public demo should prefer Nemotron 3 Super on Spark for sponsor fit while allowing a clearly labeled hosted fallback only if needed. The private appliance roadmap benchmarks Super and other Spark-friendly models for the local brain.

Evidence notes:

- NVIDIA's Nemotron deployment guide lists a "Nemotron 3 Super on DGX Spark" path for a single DGX Spark with 128 GB unified memory using vLLM and TensorRT-LLM with NVFP4 and MTP.
- NVIDIA describes Nemotron 3 Super as a 120B-total, 12B-active hybrid MoE model for agentic reasoning.
- NVIDIA describes Nemotron 3 Ultra as a 550B model; for VoiceOps, Ultra is only an optional hosted/upstream fallback and must not be used as Spark-local readiness proof.
- Public DGX Spark reports support this split: Super has one-Spark reports, while Ultra reports and forum guidance point toward multi-Spark operation.

These notes are planning inputs. They are not repository-local validation of the
current Hermes voice stack, and they must not be used to claim DGX Spark
readiness without measured local evidence from the target runtime.

## Operating Domains

### Household Operations

Hermes should help with:

- upcoming bills, renewals, subscriptions, and household budget reviews
- maintenance reminders, vendor calls, service scheduling, and quote comparison
- calendar conflicts, travel logistics, shopping lists, and recurring tasks
- urgent alerts routed to Discord, WhatsApp, phone, or SMS depending on severity

### Business Operations

Hermes should help with:

- provisioning project services such as databases, hosting, telephony, auth, AI APIs, and observability
- customer follow-up, invoices, payment links, status reporting, and support triage
- vendor spend tracking, renewals, receipts, and approval packets
- daily or weekly operations reviews with durable action logs

### Communications

The same operator should be reachable through multiple surfaces:

- Discord realtime voice for live desktop interaction
- Discord text/status updates for approvals and audit summaries
- WhatsApp for mobile commands and low-friction follow-up
- phone/SMS for urgent escalation and non-Discord stakeholders

Discord is the first live surface, not the product boundary. The product
boundary is a KAME session protocol that can be driven by Discord, VoiceClaw,
OpenClaw Talk, phone/SIP, WhatsApp voice notes, desktop mic/speaker, or future
clients. Transport adapters should normalize channel-specific audio, text,
playback, interruption, authorization, and handoff events into the same internal
session contract before any reflex or oracle policy runs.

The concrete session contract is `docs/kame-session-v1.md`. For hackathon and
headless proof purposes, it must show that Moshi/open-S2S witness text can
arrive before, with, or after the accepted audio cut and still attach to the
same raw-audio evidence bundle. The visible product story is not "Moshi did
STT, then Hermes acted"; it is "the reflex answered quickly, the interpreter
compared raw voice with witness text, and only promoted evidence reached the
Hermes oracle and action gates."

VoiceClaw/OpenClaw lessons to absorb:

- a realtime voice frontend can act as a true reflex if it has only a narrow
  `ask_brain`-style bridge to the capable agent
- that bridge must become typed Hermes oracle jobs rather than an unstructured
  chat completion if we want cancellation, status, audit, and spend safety
- the live frontend should keep recent turns compact, summarize older turns,
  and avoid carrying the whole Hermes context into the voice loop
- mobile/watch/desktop/phone surfaces matter; Hermes should expose a stable
  KAME adapter API instead of making every client embed Discord-specific logic
- telephony is a first-class adapter with codec, jitter, playback, transfer,
  DTMF, authorization, and redaction concerns, not a thin webhook script
- open S2S transcript output should be preserved as "what the live interface
  believed it heard" and passed beside raw audio to the interpreter; it should
  not become the authoritative user request just because it arrived quickly

### Money and Provisioning

The agent should eventually operate against real economic rails:

- Stripe Projects for service provisioning and credential sync
- Stripe Link CLI for user-approved purchases
- MPP/402 for paid agent-facing services
- budget caps, approval thresholds, and spend reasons before any charge
- receipts, credential locations, and rollback notes after each action

## Architecture

### Reflex

The reflex is the lightweight KAME interface model. It is optimized for
low-latency voice behavior, not deep reasoning or verbatim transcription.

Target:

- Moshi/PersonaPlex-class S2S, a small local realtime model, or an even simpler
  tuned timing/classifier path on the configured realtime runtime
- owns turn-taking, floor control, barge-in, short acknowledgements, intent triage, and local conversational glue
- emits a rough transcript hypothesis when available
- may answer locally only for low-risk interface turns
- sends structured requests to the oracle for real work

The reflex is not the brain and should not gain broad tool authority early.
The reflex transcript is a hypothesis, not durable truth. Moshi/S2S transcript
output belongs in the same non-durable hypothesis class unless the Gemma
interpreter or oracle-visible outcome promotes it.

When the reflex stack provides a Moshi-style transcript, Hermes should attach it
to the interpreter request beside the clipped raw audio. The transcript is
useful because it captures the realtime model's hearing of the turn, including
timing and code-switching context, but it must be labeled as a hypothesis so
Gemma can contradict it when the waveform says otherwise.

The reflex path must remain usable without a transcript. If a Moshi-style model
emits only audio or produces a hallucinated transcript, the system should still
make floor-control decisions from live audio/VAD and let the interpreter decide
what evidence is safe to pass to the oracle.

This is why the Moshi transcript belongs beside the waveform rather than in
front of the system. It can tell Gemma what the reflex believed it heard, but it
must not force the oracle down a false path when the waveform, energy gate,
speaker metadata, or later interpreter correction disagrees.

For implementation, treat Moshi text as a sensor event attached to the current
`turn_id`, not as a user message. If it arrives before the audio cut is ready,
hold it on the pending bundle. If it arrives after Gemma has started, attach it
as late evidence. It should never cause a second Hermes turn, duplicate oracle
job, or independent spend/provisioning/call request.

This is the core KAME rule for VoiceOps: the live voice model may speak and
route, the interpreter may promote evidence, and Hermes' active oracle may act.
No transcript hypothesis from Moshi, VoiceClaw/OpenClaw, or classic ASR should
skip the interpreter merge point and become durable user text on its own.

For implementation and demo language, avoid calling the Moshi output "the STT
result" unless the authority label is visible. It is better described as "what
the reflex believed it heard." That makes it safe and useful context for Gemma:
Gemma can recover clipped prefixes, compare code-switched wording, or reject a
hallucinated command while the user still gets the fast acknowledgement from the
reflex.

The implementation target is therefore not "run Moshi STT, then ask Hermes."
It is "keep raw voice as primary evidence, attach Moshi's transcript hypothesis,
and ask the interpreter to adjudicate both." That distinction matters for the
hackathon demo because a misheard service name, dollar amount, phone number, or
approval phrase must be shown as a rejected or corrected hypothesis before any
Stripe/NemoClaw/phone action becomes eligible.

Current naming rule: call ambiguous Moshi/open-S2S text a
`frontend_witness_hypothesis`, not authoritative STT. If the adapter knows the
text came from the live reflex model, use `reflex_transcript_hypothesis`; if it
knows the text came from a separate caption/S2S side channel, use
`s2s_transcript_hypothesis`. If it cannot tell, preserve the text as
`frontend_witness_hypothesis` with `authority = "hypothesis"` and attach it to
the same raw-audio interpreter bundle. None of those names permits transcript
only scheduling, durable user text, spend reasons, phone payloads, or tool
arguments without raw-audio-grounded interpreter promotion when audio is
available.

Confirmed packet rule for the hackathon build: the demo should show one spoken
request producing one KAME evidence bundle. If Moshi/OpenClaw/VoiceClaw-style
text exists, it is shown as frontend witness context beside the raw voice clip,
not as the message Hermes acted on. The safe story for judges is: the reflex
answers quickly, Gemma compares raw audio and witness text, then Hermes' active
`/model` receives only promoted wording plus labeled evidence before Stripe,
NemoClaw, phone, memory, file, or messaging actions can proceed.

The proof should cover all witness timing cases: witness-before-cut,
witness-with-cut, and witness-after-cut. In each case, the same `turn_id`,
`audio_segment_ref`, `evidence_bundle_id`, and `evidence_merge_key` should
survive, partial text should be superseded by same-source final text in active
interpreter context, and no duplicate oracle job or durable user message should
be created from the witness alone.

The proof should also cover text-first/raw-audio-later convergence for the same
speech cut. A provisional external frontend or reflex envelope may start
background work, but the raw-audio evidence and Moshi/Open-S2S witness update
must coalesce into the running job. A second `INTERFACE_ORACLE_REQUEST` for the
same `turn_id` is a failure because it forks the oracle context and defeats the
single-bundle KAME trust model.

### Interpreter

The interpreter is the audio-understanding evidence lane. Gemma 4 is the
preferred candidate here because its audio-multimodal path can reason over a
bounded utterance, compare that raw audio against the reflex transcript
hypothesis, and produce higher-quality multilingual evidence for the oracle
without blocking the user's immediate acknowledgement.

Target inputs:

- clipped raw audio segment and timing metadata
- speaker metadata and channel/transport metadata
- reflex transcript hypothesis, if the reflex produced one
- Moshi/S2S transcript hypothesis, if the reflex stack produced one
- frontend witness transcript hypothesis, if the adapter has STT-like text but
  cannot confidently identify whether it came from the reflex or a side channel
- optional classic ASR transcript hypothesis
- reflex route, acknowledgement, and "interface already said" text
- current oracle job/status context

Target outputs:

- corrected transcript or transcript alternatives
- normalized intent and route confidence
- entities, numbers, names, URLs, code terms, and language notes
- disagreement flags between audio, reflex transcript, Moshi/S2S transcript,
  and classic ASR
- oracle request patch or clarification recommendation

The Moshi/S2S transcript is most valuable as evidence of what the live interface
model thought it heard. It should be passed to Gemma in the same interpreter
request as the raw voice clip, never committed directly as durable user text,
and never allowed to trigger spend, provisioning, credential, call, or messaging
actions without interpreter/oracle confirmation.

That makes the Moshi/open-S2S transcript a useful companion to raw voice, not a
replacement for raw voice. In the normal path, Hermes can send Gemma the clipped
audio, the reflex route, the acknowledgement already spoken, and the Moshi
transcript hypothesis together. Gemma may use the transcript to recover words
the reflex clipped, but it must also be able to reject transcript text that does
not match the waveform or speaker context.

When the frontend exposes a transcript-looking field under a vendor name such
as "Moshi STT", Hermes should not special-case it as authoritative STT. Store
the source name for diagnostics, but classify the field by provenance and
authority. Use `frontend_witness_hypothesis` by default when the adapter cannot
prove the producer. Use `reflex_transcript_hypothesis` only if it came from the
live reflex model's own hearing, or `s2s_transcript_hypothesis` only if it came
from a distinct S2S/caption side channel. All of these sit beside the same raw
audio in the interpreter evidence bundle with `authority = "hypothesis"` and
`tool_authority = false`.

The interpreter may attach evidence to an oracle job before it starts, or submit
a patch/update if the oracle job is already running. It must not stall the
reflex acknowledgement, and it must not receive broad Hermes tools. If evidence
arrives while the job is still queued, the scheduler should fold corrected
transcript, intent, entities, and confidence into the oracle request before
execution; if it arrives late, it should be audited as a bounded update before
any irreversible spend, provisioning, call, or credential action relies on
earlier text.

The fold-in rule is source-sensitive. A trusted interpreter promotion can patch
queued `oracle_text`, transcript, intent, and tool-critical fields. Moshi,
VoiceClaw/OpenClaw, classic ASR, and other transcript-side-channel sources can
only add labeled hypotheses to the same evidence bundle unless the interpreter
or oracle later promotes them.

The scheduler must also guard persistence. Before promotion, use
`provisional_request_summary` for queueing and narration rather than
`oracle_text`. That summary is not persistable as verified user text and has no
tool authority. The persisted Hermes user message should be the promoted
interpreter/oracle wording. Durable oracle records should preserve raw
hypothesis fields only in the voice-session audit trail, not as replayable
verified user turns.

### Oracle

The oracle is Hermes's active model, selected through the existing Hermes `/model` flow. There should not be a separate `oracle_model` setting for VoiceOps. If the user points Hermes at Nemotron 3 Super, hosted Nemotron 3 Ultra, Kimi, or another provider, that is the active Hermes oracle for that run. Hosted selections do not count as Spark-local readiness evidence.

Target:

- Nemotron 3 Super as the preferred Spark-local hackathon demo and sponsor-aligned planning path, selected through `/model`
- a clearly labeled hosted fallback selected through `/model` when local Super is unavailable
- current cloud model for bring-up if needed
- local Nemotron 3 Super on DGX Spark when it proves good enough for Hermes work and beats alternatives on latency, quality, and reliability
- owns memory, tools, files, long reasoning, project context, and durable task execution

### Speech

The speech layer should support the KAME design without turning back into a simple STT-first pipeline.

Target:

- VAD/endpointer drives turn cuts
- the fast reflex consumes live audio for floor control and immediate response
- Gemma interpreter consumes clipped audio plus reflex/Moshi transcript
  hypotheses for multilingual correction and oracle evidence
- dedicated classic ASR is an optional witness/fallback transcript-hypothesis lane and
  fallback, not the reflex driver
- speculative ASR or S2S transcript capture may run to hide latency, but the
  result stays a labeled hypothesis until raw-audio-grounded interpreter
  promotion
- local Nemotron Speech or equivalent streaming ASR only as optional fallback,
  diagnostic, caption, or literal-evidence hypothesis input
- local Magpie/Riva-style TTS when available
- Cartesia or similar cloud TTS remains an acceptable bring-up fallback

The operating rule is one voice turn, one evidence bundle. Moshi/open-S2S text,
classic ASR text, and reflex transcript text may arrive at different times, but
they all attach to the same raw-audio turn as hypotheses. They do not schedule a
second oracle turn, overwrite `oracle_text`, or become spend/call/tool arguments
until Gemma promotes raw-audio-grounded evidence for the active Hermes oracle.
This lets the demo keep the acknowledgement path fast while still using
transcript-like output as useful context for multilingual correction and
business-critical actions.

Current design rule: if a Moshi/OpenClaw/VoiceClaw-style frontend provides both
raw voice and an STT-like transcript for the same speech cut, pass both to the
Gemma interpreter in one packet. The transcript is a witness hypothesis about
what the realtime frontend believed it heard. It is valuable context for Gemma,
but it is not a required ASR proof, not a separate user turn, and not an action
authority source until raw-audio-grounded interpreter promotion.

Provider selection must be role-based:

| Role | Preferred Direction | Notes |
| --- | --- | --- |
| Reflex / floor control | Moshi/PersonaPlex-class S2S or smaller local timing model | Judge on acknowledgement latency, barge-in behavior, noise rejection, and whether its transcript-like text helps Gemma without becoming authority. |
| Interpreter / evidence | Gemma 4 E2B/E4B/12B audio-multimodal | Receives raw audio first, witness transcripts second; can promote, correct, reject, or downgrade transcript text. |
| Optional witness/fallback transcript evidence | Nemotron/Riva ASR, Moshi/open-S2S text, or classic ASR | Optional support for captions, diagnostics, literal checks, and fallback. Not a control path in full KAME mode. |
| Outbound TTS | Magpie/Riva-style local TTS, Piper-class local TTS, or hosted fallback | Pick by first-audio latency, quality, and operational stability. TTS provider changes require no authority-model change. |
| Degraded fallback | Cartesia or text-only VoiceClaw/OpenClaw/Moshi bridge | Useful for demos/bring-up, but must be labeled degraded when raw audio is missing. |

The voice runtime must distinguish transport state from agent state. VAD,
semantic endpointing, playback buffers, provider response ids, and carrier
playout cursors are session-layer facts. The reflex/oracle policy consumes
normalized events and must not depend on Discord packet timing, Twilio media
frames, provider conversation state, or a particular hosted realtime API as the
source of truth.

### Skills and Tools

Hermes's existing skills and tool system remain the action layer. VoiceOps should compose those capabilities instead of bypassing them.

Voice turns must not carry the full Hermes tool surface by default. The normal Discord/CLI toolset is too large for low-latency local oracle models and pushes long sessions into unnecessary context compaction. VoiceOps needs progressive tool disclosure:

- the durable voice/oracle conversation should keep only conversation state, selected tool results, and user-visible action history
- most or all Hermes core tools may be hidden behind `tool_search` when a feature flag enables core deferral
- raw MCP and plugin tools should stay behind `tool_search` unless they are explicitly selected for the current turn
- the long-term router is an ephemeral tool-selection oracle: it receives the reflex intent, compact session summary, platform metadata, and a compact tool catalog, then returns `no_tools`, a small set of toolsets, or exact tool names
- the ephemeral router must not perform real tool calls and must not persist its own transcript into the user conversation
- actual tool invocation still happens in the real Hermes oracle session, so approvals, guardrails, NemoClaw checks, audit logging, and session state remain authoritative

The first implementation target is a conservative feature flag: `tools.tool_search.defer_core: all`, used with `tools.tool_search.enabled: 'on'`. This collapses core Hermes tools behind the same bridge used for MCP/plugin progressive disclosure. A later VoiceOps milestone should add the separate ephemeral router so the main oracle sees only the selected small tool surface for that turn, instead of needing to call `tool_search` itself.

For realtime voice, tool exposure should also carry voice UX metadata:

- `latency_class`: fast, medium, slow, streaming, or background
- `requires_confirmation`: whether the user must approve before execution
- `interruptible`: whether barge-in can safely stop the operation or only stop
  speech playback
- `side_effect`: none, local_write, network, spend, provisioning, message, or
  call
- `redaction_policy`: what may be spoken, shown in Discord text, or stored in
  audit logs
- `fallback_speech`: concise language the reflex can say while the oracle,
  NemoClaw, or a skill works

This copies the useful part of VoiceClaw/OpenClaw's `ask_brain` split without
letting the realtime model directly inherit the whole Hermes tool surface.

Initial important skills:

- Stripe Link CLI
- Stripe Projects
- cron and recurring jobs
- WhatsApp bridge
- Discord gateway
- future Twilio/phone path provisioned through Stripe Projects

### NemoClaw Execution Boundary

For the hackathon story, NemoClaw should be visible as the safe execution layer between an agent plan and real external effects.

It should wrap or present:

- Stripe Projects provisioning packets
- Stripe Link spend requests
- VoIP/phone-provider setup
- outbound message or phone-call actions
- network and credential access decisions

The video does not need to prove every NemoClaw policy in depth, but it should make clear that the agent is not receiving unchecked authority just because the user spoke a command.

The local packet validator is part of that story. It does not only check JSON
shape and command hashes; it rejects unknown action ids, unexpected providers,
unexpected command forms, wrong preflight gates, and wrong approval artifacts
before the packet can become approval evidence.

### Audit Ledger

Every operational action needs a durable record:

- requested by whom
- proposed by which model or skill
- budget and approval policy used
- exact command or API action
- approval status
- result, receipt, credential location, or rollback note
- channel where the user was notified

The audit ledger is part of the product, not debug output.

## Trust Model

Hermes should be useful because it can act, but safe because action is scoped.

Allowed without approval:

- summarize status
- draft plans
- create local reminders or dry-run action queues
- collect evidence
- prepare approval packets
- post non-sensitive summaries to configured channels

Requires explicit approval:

- any payment or purchase
- service provisioning that can create a bill
- credential rotation or deletion
- vendor contact on the user's behalf
- external messages to customers, vendors, or household members

Forbidden without a separate future design:

- hidden spend
- raw card data in model context
- unbounded recurring charges
- irreversible deletion
- broad autonomous tool execution from the reflex
- pretending an action happened when it only reached dry-run

Default mode is dry-run. Live spend should go through Stripe/Link approval or an equivalent user-visible authorization.

## Milestone 0: Hackathon Proof

Goal for the June 30, 2026 submission: show a serious local agent operating system, not a toy voice demo.

Demo request:

```text
Hermes, I am giving you 200 dollars to use through Stripe Skills. Provision yourself a VoIP provider account, then call my phone with this same context so we can continue outside Discord.
```

Required proof:

- Discord voice is the live front door.
- Hermes gives an immediate KAME-style acknowledgement.
- Nemotron 3 Super is visible as the preferred Spark-local serious planning
  candidate selected through Hermes `/model`, with any hosted fallback labeled
  clearly if used and no claim that Spark-local execution has been validated.
- NemoClaw is visible as the safe execution boundary before billable or network-capable actions.
- Hermes converts the spoken budget into a spend policy.
- Stripe Projects action is queued to provision a VoIP-capable provider account, such as Twilio voice.
- Stripe Link action is queued for a gated service-credit spend request.
- Hermes preserves the Discord context for the phone handoff.
- Hermes queues or performs an outbound call to the user's phone with the same context.
- The audit ledger shows every action and approval requirement.
- WhatsApp and phone/SMS appear as reachable follow-on surfaces.
- The DGX Spark target is explicit in the story and artifacts.

Recorded artifact must show:

- one `turn_id`, one `audio_segment_ref`, one `evidence_bundle_id`, and one
  `evidence_merge_key` for the spoken budget/provisioning request
- reflex acknowledgement and provisional route before oracle completion
- Moshi/OpenClaw/VoiceClaw/reflex/classic-ASR transcript-looking text, if any,
  preserved only as transcript hypotheses or frontend witness context
- Gemma interpreter adjudication for those witnesses:
  `accepted_as_supporting_evidence`, `corrected_by_audio`, or
  `rejected_or_diagnostic_only`
- promoted transcript, intent, entities, and confidence as the source that
  reaches the active Hermes `/model`
- Stripe spend reason, provider selection, NemoClaw action packet, phone-call
  payload, tool arguments, memory/file writes, and external messages free of
  unpromoted witness text
- `unpromoted_witness_sink_checks` proving spend, phone, NemoClaw, tool,
  memory, file, message, and durable-history sinks are clean, plus empty
  `unpromoted_witness_sink_values`
- readiness gaps still marked external when live Discord, spend/provisioning,
  or Spark/PGX evidence has not actually been collected

Video spine:

1. User joins Discord voice.
2. User gives Hermes a fixed amount of spending money through Stripe Skills.
3. Hermes acknowledges the budget and explains that live spend requires approval.
4. Hermes uses Nemotron 3 Super for the plan, or clearly labels a hosted `/model` fallback.
5. Hermes presents a NemoClaw-safe action packet.
6. Hermes queues Stripe Projects to provision VoIP.
7. Hermes queues a Link-gated spend request for service credit.
8. Hermes calls the user's phone and continues from the same Discord context.

Headless command:

```bash
uv run python scripts/hackathon_voiceops_demo.py --output-dir artifacts/hackathon-voiceops-demo/current
```

The command writes:

- `voiceops-demo.json`
- `voiceops-demo.md`
- `audit-ledger.jsonl`
- `demo-script.md`
- `nemoclaw-action-packet.json`
- `nemoclaw-action-packet.validation.json`
- `phone-context.json`
- `milestone2-execution-plan.json`
- `readiness-report.json`
- `readiness-report.md`
- `readiness-closure-summary.json`
- `readiness-closure-summary.md`
- `operator-handoff-preview.json`
- `operator-handoff-preview.md`
- `operator-dashboard.html`
- `operator-state.json`
- `operator-state-events.jsonl`
- `recording-runbook.md`
- `submission-writeup.md`
- `stripe-actions-dry-run.sh`

The generated shell script is dry-run by construction. It prints the Stripe/Projects commands instead of executing them. The NemoClaw validation artifact is also local static validation only: it checks the packet schema, approval contracts, command hashes, dry-run command alignment, blocked capabilities, and no-write safety flags without running commands, calling the network, spending, provisioning, reading credentials, or placing calls. The demo package, readiness report, local closure summary, and operator handoff preview are schema-tagged so the Milestone 0 artifact directory can be reviewed without first opening the global plan-run index. The readiness report is non-invasive: it checks local prerequisites and env shape from process env, repo `.env`, and explicit `--env-file` values by presence only. It does not read `/Users/jethac/.hermes/hermes-agent`, print secrets, provision, purchase, call, or mutate credentials. The handoff preview is also non-invasive: it lists the ordered safe evidence-collection phases, current blockers, command safety labels, must-not rules, and final reindex command for live Discord voice, spend/provisioning preflight, and DGX Spark evidence. The HTML dashboard is a static recording surface and does not require a web server.
The recording runbook gives the shot list, fallback recording path, submission checklist, and tweet draft without requiring live spend or live provisioning. The submission writeup gives concise public copy for the tweet/thread/form.

## Milestone 1: Real Voice Operator

Make `/voice join` usable as the daily control surface:

- stable Discord receive/playback lifecycle
- real barge-in based on speech energy, not silent packet arrival
- KAME fallback state visible when reflex, interpreter, optional
  witness/fallback transcript lane, or TTS paths are unavailable
- voice replies short by default
- latency metrics from user speech end to reflex response, interpreter evidence,
  oracle response, and TTS playback
- voice capability prompt context so Hermes does not claim it cannot hear or speak

Live voice evidence must prove the three-tier KAME contract, not just generic
Discord voice readiness: a fast reflex acknowledgement, one accepted raw-audio
bundle, Gemma-style direct-audio interpreter promotion, and one Hermes
active-`/model` oracle job for the speech cut. Transcript-only evidence from
Moshi/Open-S2S, VoiceClaw/OpenClaw, reflex captions, or classic ASR is useful
as witness context, but it cannot satisfy this milestone without raw-audio
promotion.

Headless command:

```bash
uv run python scripts/voiceops_voice_operator.py --output-dir artifacts/voiceops-voice-operator/current
```

The command writes:

- `voice-operator-readiness.json`
- `voice-operator-readiness.md`
- `discord-loopback-smoke.json`
- `async-oracle-smoke.json`
- `discord-session-cleanup-smoke.json`
- `voice-operator-events.jsonl`
- `live-voice-evidence-template.json`
- `live-voice-evidence.example.json`
- `live-voice-evidence-scaffold/manifest.json`
- `live-probe-closure-plan.json`
- `live-probe-closure-plan.md`

The voice-operator artifact runs the in-memory Discord realtime voice loopback smoke. It verifies lifecycle, receiver callback wiring, PCM conversion, mixer playback, barge-in signaling, latency metrics, transcript-hypothesis non-promotion, and sidecar shutdown without Discord network access, provider sidecar network access, credential reads, sends, or calls. It must still report that a live Discord `/voice join` probe is required before claiming production readiness.

After Discord env/config and the production sidecar are ready, run the one-shot live-evidence closure command. It invokes `hermes doctor --realtime-voice-report`, derives sidecar/live-turn evidence from that report, writes a manifest, and strict-validates the bundle:

```bash
uv run python -m hermes_cli.realtime_voice_live_evidence \
  --output-dir artifacts/realtime-voice-evidence/live-current \
  --run-doctor-report \
  --require-inbound \
  --wait-seconds 5
```

If an existing `hermes doctor --realtime-voice-report` JSON file already exists, derive sidecar and live-turn VoiceOps evidence from it without rerunning Discord probes:

```bash
uv run --extra dev --extra voice hermes doctor \
  --realtime-voice \
  --realtime-voice-smoke \
  --discord-voice-live-probe \
  --discord-voice-live-probe-require-inbound \
  --discord-voice-live-probe-wait-seconds 5 \
  --realtime-voice-report artifacts/realtime-voice-evidence/live-current/realtime-voice-doctor-report.json
```

```bash
uv run python -m hermes_cli.realtime_voice_live_evidence \
  --output-dir artifacts/realtime-voice-evidence/live-current \
  --from-realtime-voice-report artifacts/realtime-voice-evidence/live-current/realtime-voice-doctor-report.json
```

This writes `sidecar-session.from-realtime-report.json`, `live-turn.from-realtime-report.json`, and `realtime-voice-report-validation.json`, and references any `discord_live_probe` section only if the doctor report actually contains one. The derivation uses the alpha report validator before emitting passing evidence, copies the reported sidecar mode instead of translating loopback or diagnostic modes into production, omits raw transcripts and assistant text from generated live-turn evidence, preserves redacted KAME lineage ids when the source report provides them, and still lets strict validation report missing Discord join/playback/receiver gates when the doctor report lacks a real Discord probe.

When live evidence exists, ingest supplied artifacts without running Discord from the generator:

```bash
uv run python -m hermes_cli.realtime_voice_live_evidence \
  --output-dir artifacts/realtime-voice-evidence/live-current \
  --require-live-discord \
  --require-inbound \
  --wait-seconds 5 \
  --sidecar-session-evidence artifacts/realtime-voice-evidence/live-current/sidecar-session.json \
  --live-turn-evidence artifacts/realtime-voice-evidence/live-current/live-turn.json
```

Before indexing the collected bundle, validate it offline with the same strict live-evidence contract used by the voice-operator ingester:

```bash
uv run python -m hermes_cli.realtime_voice_live_evidence \
  --audit-only \
  --live-evidence-manifest artifacts/realtime-voice-evidence/live-current/manifest.json
```

```bash
uv run python -m hermes_cli.realtime_voice_live_evidence \
  --output-dir artifacts/realtime-voice-evidence/live-current \
  --validate-live-evidence \
  --live-evidence-manifest artifacts/realtime-voice-evidence/live-current/manifest.json
```

Use `--live-evidence-manifest` for the preferred manifest-first offline audit and validation path; split section-file arguments remain available for direct debugging before a manifest exists.

```bash
uv run python scripts/voiceops_voice_operator.py \
  --output-dir artifacts/voiceops-voice-operator/current \
  --live-evidence artifacts/realtime-voice-evidence/live-current/manifest.json
```

The supplied evidence path is read-only. It may be a `voiceops.realtime_voice_live_evidence_manifest.v1` `manifest.json` from `hermes_cli.realtime_voice_live_evidence`, a combined `voiceops.milestone1.live_voice_evidence.v1` evidence JSON file with nested live evidence sections, or individual evidence JSON files. It must prove Discord join/playback, inbound receiver frames or speech-start, Discord probe latencies, production sidecar session start/close, sidecar `sidecar_mode: production`, redacted session identity, observed healthcheck and provider transport, visible fallback reason, and one live conversational turn with native-audio/reflex evidence, concrete redacted KAME lineage ids (`turn_id`, `audio_segment_ref`, `evidence_bundle_id`, and `evidence_merge_key`), provenance-labeled transcript and interpreter metadata when available, assistant audio, barge-in, short spoken reply, no voice-capability denial, first-audio latency, and barge-in stop latency. Transcript-only evidence must not close KAME readiness. Each `discord_live_probe`, `sidecar_session`, and `live_turn` section must carry both a verifiable `source_artifact` and a non-placeholder `collector_attestation` with collector name/version, run id, command argv, git commit, timestamp window, raw/redacted SHA-256 hashes, and parent manifest hash. For manifest packages, `reports` refs and nested `source_artifact` refs must be relative paths inside the manifest/report package; absolute paths, `~`, parent traversal, symlink escapes, and process-working-directory fallback are rejected. For non-manifest ingestion, pass one `--live-evidence` per section or combined file; each standalone section file must include `kind` or `evidence_type`, a verifiable `source_artifact`, and the collector attestation. Per-section manifest reports must also identify themselves with `kind` or `evidence_type` values such as `discord_live_probe`, `sidecar_session`, or `live_turn`, unless the report is a combined expanded `voiceops.milestone1.live_voice_evidence.v1` object with nested sections. Manifest ingestion is preferred because manifest reports record the actual referenced report path as provenance. Manifest `reports` may reference per-section artifacts or one combined evidence artifact; placeholder source paths inside referenced artifacts are not trusted as provenance because the ingester records the actual manifest-resolved file path. Template source artifact names such as `discord-live-probe.json`, `voice-status-or-sidecar-report.json`, `sidecar-session.json`, `voice-turn-evidence.json`, and `live-turn.json` are rejected until replaced by real resolved artifact paths, and `example_only` or placeholder collector attestations are rejected. It must not include Discord tokens, provider secrets, full phone numbers, raw transcript text, private transcript text containing secrets, or private assistant reply text. Referenced live source artifacts must be JSON and are scanned for forbidden raw transcript/reply fields, secret-like values, phone-like values, and assistant voice-capability denial text before their hashes are accepted. `--audit-only` performs no Discord network call, no report derivation, and no persistent artifact writes under `--output-dir`; it prints schema `voiceops.realtime_voice_live_evidence_audit.v1` and returns nonzero until strict validation passes. `--validate-live-evidence` performs no Discord network call and writes `live-evidence-validation.json` with schema `voiceops.realtime_voice_live_evidence_validation.v1`; it exists to produce a durable validation artifact before `scripts/voiceops_voice_operator.py` updates the readiness artifacts. `--from-realtime-voice-report` writes derivation metadata with schema `voiceops.realtime_voice_report_derivation.v1` and must not claim production sidecar evidence from loopback or diagnostic sidecar modes. The live-evidence collector references sidecar and live-turn files in its manifest and runs strict validation whenever optional or derived evidence is supplied; it does not sanitize supplied files or embed their contents. The generated `.example.json` file is only a populated redacted shape for operators; validators reject `example_only: true` evidence until real artifact references replace it. The generated `live-voice-evidence-scaffold/` directory is the preferred starting point for manual live evidence: replace its Discord, sidecar, and live-turn section files with real redacted observations and remove every `example_only` marker before ingesting `live-voice-evidence-scaffold/manifest.json`.

## Milestone 2: Real Spend and Provisioning

Turn the dry-run queue into controlled live operations:

- run the non-mutating provisioning preflight before any live spend or provider action
- verify Stripe Link CLI auth and approval flow
- verify Stripe Projects plugin and catalog on the target machine
- run `stripe projects list` and safe catalog discovery headlessly
- execute one low-risk live VoIP provisioning path only after explicit approval
- queue or perform one outbound phone call with a preserved context packet
- record receipts and generated credential locations without exposing secrets
- add rollback/deprovision notes to the ledger

Preflight command:

```bash
uv run python scripts/voiceops_provisioning_probe.py --output-dir artifacts/voiceops-provisioning/current
```

The command writes:

- `provisioning-readiness.json`
- `provisioning-readiness.md`
- `safe-command-manifest.json`
- `read-only-discovery.json`
- `read-only-discovery.md`
- `read-only-discovery.manifest.json`
- `audit-ledger.read-only-discovery.jsonl`
- `milestone2-execution-plan.json`
- `milestone2-execution-plan.md`
- `provisioning-preflight-evidence.template.json`
- `provisioning-preflight-evidence.example.json`
- `provisioning-preflight-evidence.manifest.example.json`
- `provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json`
- `post-approval-receipts.template.json`
- `post-approval-receipts.example.json`
- `post-approval-receipts.validation.json`
- `nemoclaw-action-packet.validation.json`
- `post-approval-receipts-scaffold/post-approval-receipts.json`
- `audit-ledger.post-approval.jsonl`
- `setup-closure-plan.json`
- `setup-closure-plan.md`

The default preflight is non-mutating and only checks PATH/env presence, env-key presence, local optional Stripe Skills bundle contracts, command policy, and phone-handoff configuration shape. It writes a `not_supplied` NemoClaw action-packet validation artifact unless `--nemoclaw-action-packet` is provided. It verifies that `optional-skills/payments/stripe-projects`, `optional-skills/payments/stripe-link-cli`, and `optional-skills/payments/mpp-agent` exist with the expected Link approval, HTTP 402/SPT, Projects billing, `.env`, and wallet/key secrecy safety terms. It blocks live spend, provider provisioning, credential retrieval, outbound phone calls, account mutation, and network tunnels. If active command probing is needed, it must be explicitly enabled with `--run-command-probes`; that mode is still limited to isolated version/help subprocess probes and must not be treated as approval for `stripe projects add`, Link spend creation, card retrieval, MPP payment, SMS, or phone calls.

To validate the generated NemoClaw packet without running provider commands, ingest it explicitly:

```bash
uv run python scripts/voiceops_provisioning_probe.py \
  --output-dir artifacts/voiceops-provisioning/current \
  --env-file .env \
  --no-command-probes \
  --nemoclaw-action-packet artifacts/hackathon-voiceops-demo/current/nemoclaw-action-packet.json
```

Display-only discovery is a separate opt-in path and is required before Milestone 2 can be considered ready for live provisioning approval. Use `--run-readonly-discovery` only for the exact allowlisted commands `stripe projects list --limit 10` and `link-cli auth status`. These commands run with an isolated temporary `HOME`, so `link-cli auth status` is an isolated auth-status attempt and does not prove the operator's normal local CLI auth state. The probe redacts command output, writes `read-only-discovery.json`, `read-only-discovery.md`, `read-only-discovery.manifest.json`, and `audit-ledger.read-only-discovery.jsonl`, and still does not grant approval for spend, provisioning, credential retrieval, messages, or calls.

```bash
uv run python scripts/voiceops_provisioning_probe.py \
  --output-dir artifacts/voiceops-provisioning/current \
  --env-file .env \
  --run-readonly-discovery
```

After a read-only discovery run exists, later closure/index runs should ingest the redacted manifest instead of rerunning network-capable discovery:

```bash
uv run python scripts/voiceops_provisioning_probe.py \
  --output-dir artifacts/voiceops-provisioning/current \
  --env-file .env \
  --read-only-discovery-evidence artifacts/voiceops-provisioning/current/read-only-discovery.manifest.json
```

When local setup and account/capability evidence exists, ingest supplied evidence without running live spend or provider mutations:

```bash
uv run python scripts/voiceops_provisioning_probe.py \
  --output-dir artifacts/voiceops-provisioning/current \
  --env-file .env \
  --preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-evidence.json
```

Or, when evidence is split by provider/domain:

```bash
uv run python scripts/voiceops_provisioning_probe.py \
  --output-dir artifacts/voiceops-provisioning/current \
  --env-file .env \
  --preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json
```

The supplied evidence path is read-only. It may be one complete `voiceops.milestone2.preflight_evidence.v1` JSON file or a `voiceops.milestone2.preflight_evidence_manifest.v1` manifest that references separate redacted section files for Stripe Projects, Stripe Link, MPP/NemoClaw, phone handoff, and rollback ownership. It must contain account aliases, capability booleans, provider references, credential-location references, rollback owners, and a `source_artifact` for every redacted evidence section. Every section must also include `source_artifact_kind: redacted_setup_evidence`, `source_artifact_sha256`, `source_artifact_redacted_at`, and `collector_attestation`; the referenced source artifact must also declare `artifact_kind: redacted_setup_evidence`. The SHA-256 must match the referenced redacted JSON source artifact, the attestation redacted hash must match that SHA-256, and the redaction and collection timestamps must be parseable with timezone information. Source artifacts must exist, be UTF-8 JSON, be marked redacted or carry a redaction policy, and resolve as relative paths inside the supplied evidence or manifest package; absolute paths, `~`, parent traversal, symlink escapes, and process-working-directory fallback are rejected. Collector attestations must identify the collector name/version, run id, command argv, git commit, timestamp window, raw/redacted SHA-256 hashes, and parent manifest hash; placeholder or `example_only` attestations are rejected. It must not contain Stripe secrets, provider tokens, raw card data, full phone numbers, or proof of unapproved live spend. The generated `.example.json` and `.manifest.example.json` files show redacted completed shapes for headless setup, but they are rejected as proof while `example_only: true` remains present.
The generated `provisioning-preflight-scaffold/` directory is the preferred operator starting point for split evidence: replace each section report and redacted source artifact with real local setup proof, refresh the SHA-256 fields, and remove every `example_only` marker before ingesting the manifest.

```bash
uv run python scripts/voiceops_provisioning_probe.py \
  --refresh-preflight-source-hashes artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json
```

The refresh helper reads and writes only local JSON evidence files. It refreshes `source_artifact_sha256` and `collector_attestation.redacted_artifact_sha256` together. It does not inspect env files, run command probes, perform network I/O, provision providers, spend money, retrieve credentials, send messages, or place calls.

The Milestone 2 execution plan is also non-mutating. It is the post-approval contract for the first live provisioning flow: readiness gates, display-only discovery commands, approval-required Stripe/Link/phone actions, receipt schema, credential-location schema, rollback/deprovision notes, and phone-context linkage. It must never claim that spend, provisioning, credential retrieval, outbound messages, or phone calls have already executed.

When approved actions have real redacted post-approval evidence, ingest the receipt bundle with `--post-approval-receipts` without running provider commands:

```bash
uv run python scripts/voiceops_provisioning_probe.py \
  --output-dir artifacts/voiceops-provisioning/current \
  --env-file .env \
  --post-approval-receipts artifacts/voiceops-provisioning/current/post-approval-receipts.json
```

To actually execute approved Stripe/Link actions, use the bounded post-approval executor instead of hand-running commands. It requires a validated NemoClaw packet, the generated Milestone 2 execution plan, a redacted `approval-decisions.json` containing explicit `approve_once`, `deny`, or `hold` decisions, and the live confirmation string. Without `--execute`, `approve_once` decisions fail closed; with `--execute`, only exact allowlisted Stripe Projects, Stripe Link, and queued phone-handoff commands that match the packet and plan can run. The executor writes `post-approval-receipts.json`, per-action approval decision artifacts, and `stripe-executor-report.json`; the normal receipt validator then decides whether the evidence closes the gate.

```bash
uv run python scripts/voiceops_stripe_executor.py \
  --nemoclaw-action-packet artifacts/hackathon-voiceops-demo/current/nemoclaw-action-packet.json \
  --execution-plan artifacts/voiceops-provisioning/current/milestone2-execution-plan.json \
  --approval-decisions artifacts/voiceops-provisioning/current/approval-decisions.json \
  --output-dir artifacts/voiceops-provisioning/current \
  --execute \
  --confirm-live-actions execute-approved-voiceops-stripe-actions
```

The receipt bundle uses `voiceops.milestone2.post_approval_receipts.v1` and must contain redacted `receipts`, `credential_locations`, `rollback_receipts`, and `audit_events`. The validator rejects `example_only`, raw secret/token/card/phone fields, command hash mismatches, unknown action ids, duplicate credential-location refs, duplicate rollback refs, duplicate audit event ids, missing audit events, missing approval-decision provenance, attempted execution without `decision: approve_once`, and missing credential-location or rollback refs for executed actions. Held, denied, and skipped decision receipts may omit execution-only credential and rollback artifacts, but they still need explicit `decision`, `decision_by`, `decision_at`, `approval_decision_ref`, and `approval_decision_sha256` fields. When receipts are loaded from a file, each `approval_decision_ref` must resolve to a package-local redacted JSON artifact whose SHA-256 matches `approval_decision_sha256`; absolute paths, home expansion, parent traversal, missing files, unredacted files, secret-like values, and phone-like values are rejected. A valid bundle writes `post-approval-receipts.validation.json` and `audit-ledger.post-approval.jsonl`; receipt validation itself still does not execute spend, provisioning, credential retrieval, messages, or calls.

## Milestone 3: Multi-Channel Operations

Make the same operator reachable beyond Discord:

- generate and review the multi-channel policy artifact before enabling new egress surfaces
- WhatsApp Cloud setup path validated for command and approval messages
- phone/SMS path designed around Twilio or equivalent provisioning
- VoiceClaw/OpenClaw-compatible KAME bridge shape documented so external
  realtime clients can submit user turns, receive reflex speech/status, and
  track oracle jobs without bypassing Hermes authority; the current wire
  contract is `docs/kame-session-v1.md`
- channel-specific authorization rules
- escalation policy for urgent household/business events
- consistent audit IDs across Discord, WhatsApp, and phone/SMS

Policy command:

```bash
uv run python scripts/voiceops_channel_policy.py --output-dir artifacts/voiceops-channel-policy/current
```

The command writes:

- `channel-policy.json`
- `channel-policy.md`
- `channel-policy-review.json`
- `channel-policy-review.md`

The policy artifacts are static and headless. They read no secrets, perform no network I/O, send no Discord/WhatsApp/SMS messages, and place no calls. They define channel authorization, approval routing, escalation levels, audit ID continuity, redaction rules, and a pending human review packet for Discord, WhatsApp, and phone/SMS before those surfaces are used for real operations. The review packet does not enable egress; it records the signoffs and gates that must be satisfied before a separate runtime approval can do that.

The policy review must carry the same KAME witness contract as Discord voice:
transcript hypotheses are referenced by source, digest, arrival phase, latency,
confidence, speaker/channel binding, and adjudication metadata. Raw witness
text is not allowed as WhatsApp/SMS/phone/Discord egress content, and
hypothesis-only wording cannot satisfy the promoted-evidence gate.

## Milestone 4: Local Deployment Evidence

Prove the three KAME roles on the selected local runtime, with one DGX Spark as
the preferred evidence target:

- local reflex model launch evidence
- local Gemma interpreter launch evidence
- local Hermes oracle endpoint registered through normal `/model` selection
- Nemotron 3 Super evaluated as the preferred local NVIDIA brain
- local TTS bridge evidence
- optional witness/fallback transcript evidence only for fallback, diagnostics,
  captions, or comparison runs
- all-local smoke with oracle, reflex, raw-audio interpreter, TTS, and sidecar
  together
- benchmark evidence accepted by the generated local/Spark matrix validator

The target Spark shape is the same three-tier KAME contract as the Discord
voice design: a fast reflex owns floor control, Gemma-style direct-audio
interpretation owns transcript promotion, and Hermes' active `/model` owns
oracle work. Moshi/open-S2S transcript text is allowed and useful, but only as a
frontend witness attached to the same raw-audio interpreter bundle. It must not
count as a separate Spark readiness role, create a second Hermes turn, replace
raw-audio evidence, or satisfy spend/provisioning/phone approval gates without
raw-audio-grounded interpreter promotion.

Headless command:

```bash
uv run python scripts/voiceops_spark_matrix.py --output-dir artifacts/voiceops-spark-matrix/current
```

The command writes:

- `spark-model-matrix.json`
- `spark-model-matrix.md`
- `spark-benchmark-evidence-template.json`
- `spark-benchmark-evidence.example.json`
- `spark-benchmark-scaffold/spark-benchmark-evidence.json`
- `spark-matrix-closure-plan.json`
- `spark-matrix-closure-plan.md`
- `spark-operator-runbook.md`

When benchmark evidence exists, pass it with repeated `--evidence path/to/evidence.json` arguments. Use `--refresh-source-hashes path/to/evidence.json` after replacing redacted raw source artifacts so the local JSON evidence file refreshes `source_artifact_sha256` and `collector_attestation.redacted_artifact_sha256` together without running benchmarks, network I/O, or Spark execution. Use `--lint-evidence --evidence path/to/evidence.json` next when the operator wants a no-write packaging/readiness check; lint mode prints schema `voiceops.spark_evidence_lint.v1`, performs no artifact writes, performs no network I/O, and returns nonzero until the supplied evidence proves one-Spark readiness. The matrix accepts its native `voiceops.spark_benchmark_evidence.v1` records and adapts the generated KAME DGX Spark benchmark evidence shape when provenance is present. Local readiness requires more than role metrics: evidence must identify the hardware/locality, model, measurement time, source artifact, source artifact SHA-256, collector attestation, verification state, and an all-local stack smoke proving reflex, interpreter, oracle, TTS, and sidecar ran together on one DGX Spark. Auxiliary Moshi/S2S or ASR transcript evidence is useful fallback context for the interpreter, but it is not a required readiness role. The referenced redacted raw source artifact must exist beside the supplied evidence file or at an absolute path, must be readable UTF-8 JSON, must explicitly set `redacted: true`, must not contain likely secrets or phone-like values, must not carry `example_only: true`, and its SHA-256 must match both `source_artifact_sha256` and `collector_attestation.redacted_artifact_sha256`. Collector attestations must identify the collector name/version, run id, command argv, git commit, timestamp window, raw/redacted SHA-256 hashes, and parent manifest hash; placeholder or `example_only` attestations are rejected. The stack smoke must also prove KAME routing: the oracle is selected by Hermes `/model`, oracle authority routes include tools/files/memory/project context, the interface input source includes `native_audio`, the reflex provider proves the chosen low-latency S2S or timing path, and the interpreter provider proves Gemma raw-audio serving. The smoke metrics must include `speech_end_to_first_audio_ms <= 1500`, `barge_in_stop_ms <= 150`, `local_turns >= 1`, `local_turn_oracle_calls == 0`, `oracle_bound_turns >= 1`, and `oracle_bound_oracle_calls >= oracle_bound_turns`. For adapted KAME smoke, local reflex turns must not call the oracle, while oracle-bound turns must route through Hermes `/model` authority. Until measured evidence is supplied, the matrix must mark local Spark roles and `all_local_stack_smoke` as needing evidence rather than claiming readiness. The generated `.example.json` file is a passing-looking guide for measured Spark artifacts, but all `example_only: true` entries are rejected by the matrix. The generated Spark closure plan is the Milestone 4 checklist that the readiness closure index should point at for missing model, speech, and all-local stack-smoke proof.
The generated `spark-benchmark-scaffold/` directory is the preferred DGX operator starting point: it contains a wrapper benchmark evidence file plus placeholder raw-source artifacts that resolve correctly but are rejected until replaced with measured DGX Spark output and all `example_only` markers are removed.
The generated `spark-operator-runbook.md` is the step-by-step DGX collection runbook: start the local KAME stack, run `scripts/dgx_spark_gemma4_voice_eval.sh`, replace scaffold source artifacts with redacted measured outputs, refresh source and attestation hashes with `--refresh-source-hashes`, lint the evidence with `--lint-evidence`, validate/write the matrix with `--evidence`, and re-index the full VoiceOps plan.

## Milestone 5: Operator Dashboard

Add an inspectable operations surface:

- current mode: dry-run, approval-required, or live
- active voice surface and fallback reason
- budget status
- pending approvals
- recent audit events
- provisioned services
- upcoming household/business tasks

The dashboard should support the operator workflow. It should not become a marketing page.

Headless command:

```bash
uv run python scripts/voiceops_operator_state.py --output-dir artifacts/voiceops-operator-state/current
```

The command writes:

- `operator-state.json`
- `operator-state.md`
- `operator-state-events.jsonl`

The operator-state generator is artifact-only. It does not read environment secrets, perform network I/O, send Discord/WhatsApp/SMS messages, place calls, provision services, or spend money. It gives the recording dashboard and future GUI a durable state contract for current mode, active/fallback voice surface, budget status, pending approvals, audit events, planned/provisioned services, and household/business tasks. The artifact must include `voiceops.tool_disclosure_proof.v1`, and each pending approval must include `voiceops.kame_action_evidence.v1` with the action-specific promoted fields, `interpreter_promoted` and `oracle_promoted` evidence labels, `hypotheses_allowed_for_action=false`, `transcript_hypotheses_promoted=false`, and `tool_disclosure_ref=tool_disclosure`. Moshi/open-S2S, VoiceClaw/OpenClaw, reflex, or classic-ASR transcript hypotheses can be shown in the audit trail, but they cannot satisfy an operator approval row without promotion.

For the hackathon demo, the generated `operator-dashboard.html` from Milestone 0 should visibly show the same operator state shape: current mode, active voice surface, fallback reason, full budget status, pending approvals, recent audit events, planned services, and a link to `operator-state.json`. Pending approval rows must also show a compact KAME evidence summary with promoted evidence labels, the raw-audio reference, and `tool_disclosure_ref`, and package audit must reject dashboard rows that drift from `operator-state.json`.

## Headless Plan Run

Run every currently headless VoiceOps milestone artifact generator and write one evidence index:

No-write preflight:

```bash
uv run python scripts/voiceops_plan_run.py --artifact-root artifacts --output-dir artifacts/voiceops-plan/current --dry-audit --package-audit
```

`--dry-audit` builds the same plan summary in a temporary artifact root, prints the readiness gaps, closure status, safety flags, current-environment blockers, and ordered `next_actions`, then removes the temporary artifacts on exit. It does not write the requested artifact paths, and it refuses `--run-command-probes` and `--run-readonly-discovery` so it cannot silently become a subprocess or network-capable probe. Its `ok` field means no hard validation failures, not readiness; use `readiness_ok` or `closure_status: complete` for readiness automation. The `next_actions` records are machine-readable and include each remaining gate, whether the current host can run it, current environment blockers, the first safe evidence command, any separate diagnostic command, and the success check.

This is the headless readiness plan for the current pivot: generate the package
and package audit first, then close the external gates with collected live
Discord voice evidence, spend/provisioning preflight evidence, and local
deployment evidence for the selected runtime. Transcript-only or frontend-only
demos must stay blocked from full KAME and local-readiness claims.

Artifact-writing indexer with final package audit:

```bash
uv run python scripts/voiceops_plan_run.py --artifact-root artifacts --output-dir artifacts/voiceops-plan/current --package-audit
```

The command writes:

- all Milestone 0 demo artifacts under `artifacts/hackathon-voiceops-demo/current`
- all Milestone 1 voice-operator artifacts under `artifacts/voiceops-voice-operator/current`
- all Milestone 2 provisioning preflight artifacts under `artifacts/voiceops-provisioning/current`
- all Milestone 3 channel policy artifacts under `artifacts/voiceops-channel-policy/current`
- all Milestone 4 Spark matrix artifacts under `artifacts/voiceops-spark-matrix/current`
- all Milestone 5 operator-state artifacts under `artifacts/voiceops-operator-state/current`
- `voiceops-plan-run.json`
- `voiceops-plan-run.md`
- `readiness-closure-index.json`
- `readiness-closure-index.md`
- `operator-handoff.json`
- `operator-handoff.md`
- `package-audit.json`
- `package-audit.md`

The audited plan run is artifact-only but writes the artifact tree above. It should surface readiness gaps such as missing Stripe/phone local setup or missing DGX Spark benchmark evidence, but those gaps must not cause live spend, provider provisioning, outbound messaging, calls, or secret reads. Use `--dry-audit --package-audit` first when the operator wants the same status check and package consistency check without persistent artifact writes.

The readiness closure index is the top-level next-action map for the remaining external evidence gates. It must keep live Discord voice evidence, Stripe/MPP/phone provisioning evidence, and DGX Spark benchmark evidence separate, list the required proof shape for each gate including collector attestation requirements, point at the relevant evidence templates and closure plans, emit ordered `next_actions`, and continue to report `needs_external_evidence` until supplied artifacts prove the live gates. It must never collapse missing live evidence into a single ready claim.

The operator handoff is the ordered execution runbook derived from the closure index. It must list the live Discord voice, spend/provisioning preflight, and local Spark stack phases in order, include exact collection and re-index commands, identify expected artifacts and success checks, and state that the handoff does not change readiness by itself. The final re-index command must include `--package-audit`, and package audit is part of final headless verification after every closure re-index.

The test suite includes a closure rehearsal with redacted local fixtures for all three remaining gates. It proves that supplied live voice evidence, provisioning preflight evidence, and Spark benchmark evidence with valid collector attestations can drive `readiness_gaps: []` and `remaining_gates: []` without credentials, live Discord, provider actions, phone calls, network I/O, or DGX Spark execution. Fixture evidence must be reported as `evidence_mode: fixture_rehearsal`, must keep `readiness_ok: false`, and must not produce `closure_status: complete`. Real readiness still requires replacing those fixtures with actual collected evidence and closing review gaps.

Standalone package consistency audit:

```bash
uv run python scripts/voiceops_artifact_package_audit.py --artifact-root artifacts --audit-only
```

Artifact-writing audit:

```bash
uv run python scripts/voiceops_artifact_package_audit.py \
  --artifact-root artifacts \
  --output-dir artifacts/voiceops-package-audit/current
```

The package audit is local and static. It reads the generated VoiceOps package and checks cross-artifact consistency between the demo readiness report, demo closure summary, plan closure index, operator state, dashboard HTML, NemoClaw packet validation, audit ledger, and dry-run shell metadata. It also revalidates the generated live voice, provisioning preflight, post-approval, and Spark benchmark scaffolds so manifest-local path contracts, source-artifact hashes, attestation hashes, and `example_only` guards cannot drift after package generation. It catches contradictions such as live-ready claims while closure gates remain, mismatched NemoClaw/operator approval contracts, executed audit rows in a dry-run package, external service provisioning claims without receipts, and missing non-live dashboard status. `--audit-only` performs no persistent writes. Prefer `voiceops_plan_run.py --package-audit` for normal headless operation so the plan index and package audit are generated and evaluated together.

When evidence exists, rerun the same indexer with the relevant read-only artifacts instead of hand-editing the index:

```bash
uv run python scripts/voiceops_plan_run.py --artifact-root artifacts \
  --output-dir artifacts/voiceops-plan/current \
  --package-audit \
  --voice-live-evidence artifacts/realtime-voice-evidence/live-current/manifest.json
```

```bash
uv run python scripts/voiceops_plan_run.py --artifact-root artifacts \
  --output-dir artifacts/voiceops-plan/current \
  --package-audit \
  --env-file .env \
  --read-only-discovery-evidence artifacts/voiceops-provisioning/current/read-only-discovery.manifest.json \
  --provisioning-preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-evidence.json
```

```bash
uv run python scripts/voiceops_plan_run.py --artifact-root artifacts \
  --output-dir artifacts/voiceops-plan/current \
  --package-audit \
  --env-file .env \
  --read-only-discovery-evidence artifacts/voiceops-provisioning/current/read-only-discovery.manifest.json \
  --provisioning-preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json
```

```bash
uv run python scripts/voiceops_plan_run.py --artifact-root artifacts \
  --output-dir artifacts/voiceops-plan/current \
  --package-audit \
  --env-file .env \
  --read-only-discovery-evidence artifacts/voiceops-provisioning/current/read-only-discovery.manifest.json \
  --provisioning-preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json \
  --post-approval-receipts artifacts/voiceops-provisioning/current/post-approval-receipts.json
```

```bash
uv run python scripts/voiceops_plan_run.py --artifact-root artifacts \
  --output-dir artifacts/voiceops-plan/current \
  --package-audit \
  --evidence artifacts/voiceops-spark-matrix/current/spark-benchmark-scaffold/spark-benchmark-evidence.json
```

If local setup discovery needs bounded binary/version checks, run the provisioning probe with explicit opt-in and then re-index its evidence:

```bash
uv run python scripts/voiceops_plan_run.py --artifact-root artifacts \
  --output-dir artifacts/voiceops-plan/current \
  --package-audit \
  --env-file .env \
  --run-command-probes
```

If local setup discovery needs authenticated display-only catalog/auth checks, keep it separate from version/help probes and run the exact read-only allowlist:

```bash
uv run python scripts/voiceops_plan_run.py --artifact-root artifacts \
  --output-dir artifacts/voiceops-plan/current \
  --package-audit \
  --env-file .env \
  --run-readonly-discovery
```

## Success Criteria

Short term:

- the hackathon demo can be recorded in 1-3 minutes
- the artifacts are generated headlessly
- no live spend occurs by default
- the story clearly ties Spark, Hermes, Stripe, and voice together

Medium term:

- the user can ask in Discord voice for a real household/business operation and receive a budgeted approval packet
- the same request can be followed up from WhatsApp
- one approved provisioning flow executes and records a complete ledger trail

Long term:

- the system can run daily household and business operations from one DGX Spark with minimal cloud dependence
- Hermes remains model-flexible through `/model`
- VoiceClaw/OpenClaw-style realtime frontends can use Hermes as the durable
  KAME backend without gaining direct spend, shell, file, or memory authority
- the user can trust the system because every action is scoped, approved, reversible where possible, and audited

## Non-Goals

- building a generic voice assistant detached from real operations
- making Discord or any one sidecar the permanent VoiceOps architecture boundary
- creating a separate VoiceOps oracle model setting
- allowing the reflex broad tool or spend authority
- blocking the hackathon proof on fully local Gemma audio serving
- blocking the hackathon proof on real purchases
- hiding dry-run status to make the demo look more complete than it is
- claiming full KAME, Stripe/NemoClaw/phone readiness, or Spark-local readiness
  from transcript-only, frontend-only, or hosted-fallback demos
