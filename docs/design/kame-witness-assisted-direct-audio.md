# KAME Witness-Assisted Direct Audio

Status: design amendment
Date: 2026-07-05
Related contract: `docs/kame-session-v1.md`

## Decision

Full KAME voice should use a three-tier hearing stack with witness fan-in:

```text
live audio
  -> reflex: floor control, VAD/energy gate, barge-in, acknowledgement
  -> interpreter: Gemma direct-audio adjudication of the accepted speech cut
  -> oracle: Hermes active /model for durable reasoning and tools
```

Moshi/Open-S2S, VoiceClaw/OpenClaw, reflex captions, and classic ASR may emit
transcript-looking text. When raw audio for the same accepted speech cut exists,
Hermes should provide that text to the Gemma interpreter with the waveform as
`transcript_hypotheses[]`. This is the normal full-KAME path:
`witness_assisted_direct_audio`.

That text is a witness claim about what a frontend believed it heard. It is not
the transcript of record, not a second Hermes user turn, not `oracle_text`, not
a spend reason, and not tool authority.

## Moshi Transcript Context Lock

The current design explicitly allows a Moshi/Open-S2S transcript to be sent to
Gemma beside the raw voice, but only as same-cut interpreter context. The useful
operation is:

```text
accepted raw audio
  + timing/speaker/channel/energy metadata
  + reflex acknowledgement and provisional route
  + Moshi/Open-S2S transcript hypothesis
  -> Gemma direct-audio interpreter
```

This is the preferred way to use Moshi-style "STT" output. It gives Gemma a
fast witness report of what the live frontend believed it heard while keeping
the waveform as primary evidence. It is not a parallel ASR control path and it
does not create a fourth model tier.

The adapter must therefore preserve provider text under
`transcript_hypotheses[]` with the same `turn_id`, `audio_segment_ref`,
`evidence_bundle_id`, and `evidence_merge_key` as the waveform. The row must
remain `role = "witness_context"`, `authority = "hypothesis"`, and
`tool_authority = false` until Gemma emits `interpreter_promoted` fields or the
Hermes oracle emits `oracle_promoted` fields.

Headless acceptance should prove both sides of the rule: one passing
`witness_assisted_direct_audio` case where Moshi/Open-S2S text improves or
supports Gemma's promoted interpretation, and one negative case where text-only
or conflicting Moshi/Open-S2S output remains degraded or diagnostic-only and
cannot reach Stripe, NemoClaw, phone, memory, file, message, durable history,
or tool sinks.

## Current Architecture Choice

The practical design choice is:

```text
fast reflex model
  -> owns floor timing, barge-in, noise rejection, acknowledgement, and rough route
Gemma direct-audio interpreter
  -> receives the accepted waveform plus same-cut witness text
Hermes active /model
  -> receives only promoted wording, intent, entities, and compact audit metadata
```

Moshi/Open-S2S text is valuable in this shape, but not because it replaces the
interpreter. It is the frontend's witness statement: "this is what the live
voice layer thought it heard during this audio cut." Send it to Gemma with the
raw voice so Gemma can use it for multilingual/code-switched recovery, clipped
prefixes, names, numbers, and rough intent. Do not send it to the Hermes oracle
as the user prompt, and do not schedule a second Hermes turn from that text.

This also means the reflex model does not need to be the best literal ASR model.
Its first job is latency and turn-taking. The transcript-like strings it emits,
or that a Moshi/Open-S2S companion emits, are useful only after they are bound
to the same accepted speech cut and labeled as hypotheses. Gemma is the first
component allowed to promote wording for Hermes' durable/actionable model path.

## Current Design Amendment

The direct answer to the Moshi question is yes: when a Moshi/Open-S2S frontend
can provide both the clipped waveform and a transcript-like string for the same
accepted speech cut, Hermes should provide both to Gemma. The waveform is the
primary interpreter evidence. The transcript-like string is context about what
the realtime frontend believed it heard.

This is not "keeping STT in parallel." It is one interpreter bundle with sensor
fan-in. The Moshi/Open-S2S string may help Gemma recover a clipped prefix,
name, number, language switch, or rough intent, but it stays a witness until
Gemma emits `interpreter_promoted` evidence or the Hermes oracle emits
`oracle_promoted` evidence. The adapter must therefore preserve the original
text as a hypothesis with source, timing, speaker/channel binding, and digest,
while durable prompts and action payloads use only promoted wording.

For implementation, the rule is:

```text
same speech cut:
  raw waveform -> Gemma primary evidence
  Moshi/Open-S2S/reflex/classic-ASR text -> transcript_hypotheses[]
  Gemma adjudication -> interpreter_promoted
  Hermes active /model -> durable oracle work
```

If the Moshi/Open-S2S text arrives before the waveform, hold it on the pending
bundle. If it arrives with the waveform, include it after metadata and reflex
state. If it arrives after Gemma has started, append it as late evidence on the
same bundle and job. In all three cases it must not create a second Hermes turn
or become a spend, phone, file, memory, message, or tool argument by arriving
first.

## 2026-07-05 Runtime Lock

The current implementation target is not "replace STT with Gemma" and not
"run STT beside Gemma." It is:

```text
energy/noise gate accepts one speech cut
  -> reflex answers the floor immediately
  -> Gemma receives the raw cut plus same-cut witness context
  -> Hermes active /model receives only promoted interpreter output
```

Moshi, Open-S2S, VoiceClaw/OpenClaw, reflex-caption, or classic-ASR text should
be provided to Gemma when it belongs to the same accepted speech cut. That is
the desired context path. The adapter must attach the text to the interpreter
packet as a hypothesis, not submit it as a competing Hermes prompt.

This makes the fast text useful without making it authoritative. The witness
can help Gemma recover a clipped prefix, name, number, code-switched phrase, or
rough intent. It can also be rejected as stale, wrong-speaker, wrong-channel,
low-energy, hallucinated, or contradicted by the waveform. The oracle never
sees raw witness text as the user request unless Gemma or the oracle promotes
it.

## Why This Shape

The reflex needs to be fast enough to answer the floor as soon as the user stops
talking. Waiting for a dedicated ASR result before acknowledgement makes the
conversation feel slow.

Raw audio still needs a higher-authority interpretation step before the system
spends money, provisions services, places calls, writes files, writes memory, or
sends external messages. A Moshi/Open-S2S transcript can be useful context, but
it can also be clipped, stale, wrong-speaker, wrong-channel, low-energy, or
hallucinated. Gemma gets both the waveform and the witness text so it can accept,
correct, or reject the witness.

This is not a parallel STT architecture. The accepted speech cut is the unit of
work. All sensor text attaches to that unit.

## Three-Tier Sensor Fan-In

The current design should be read as three tiers, not as an ASR pipeline with
extra steps:

| Tier | Latency Role | Authority |
| --- | --- | --- |
| Reflex | Detect speech, reject noise, handle barge-in, and acknowledge immediately after a valid cut. | Provisional only; may explain what it is asking the oracle to do. |
| Gemma interpreter | Consume the bounded raw-audio cut plus same-cut Moshi/Open-S2S/reflex/classic-ASR witnesses. | Can promote corrected transcript, intent, entities, confidence, and compact audit metadata. |
| Hermes active `/model` | Perform durable reasoning, tools, memory, spend, calls, files, and external messages. | Full oracle authority through the normal Hermes model selection path. |

This means Gemma is not a sibling "multilingual ASR" job running beside the
reflex. Gemma is the promotion gate for the accepted cut. Moshi/Open-S2S text
is still useful, and in practice should be sent to Gemma with the raw voice, but
only as a witness claim. The runtime should optimize for:

```text
valid speech cut -> immediate reflex acknowledgement
valid speech cut -> raw-audio interpreter packet
same-cut Moshi/Open-S2S text -> transcript_hypotheses[] on that packet
Gemma promotion -> Hermes active /model oracle work
```

If a future reflex model is strong enough to emit useful multilingual text, its
text follows the same rule. It is `reflex_transcript_hypothesis`, not durable
truth. If a Moshi/Open-S2S model emits the text, it is
`frontend_witness_hypothesis` or `s2s_transcript_hypothesis`, not the prompt.
If classic ASR emits the text, it is `classic_asr_hypothesis`, not a release
condition for high-risk action gates.

The practical reason is reliability. The reflex is optimized for timing, not
legal/financial/action authority. Moshi-style text can be clipped or
hallucinated. Classic ASR can mis-handle code-switches, names, and low-quality
Discord audio. Gemma gets all of those claims beside the waveform and decides
what, if anything, becomes promoted evidence.

## Packet Contract

The interpreter packet order is fixed:

```text
raw_audio
metadata: VAD, energy, speaker, channel, transport, timing
reflex: route, acknowledgement already spoken, provisional intent
transcript_hypotheses: Moshi/Open-S2S/reflex/classic-ASR witness text
```

Every witness hypothesis must be bound to the same speech cut with:

- `turn_id`
- `audio_segment_ref`
- `evidence_bundle_id`
- `evidence_merge_key`
- `speaker_or_actor_ref`
- `channel_or_surface_ref`
- `arrival_phase`: `before_raw_audio`, `with_raw_audio`, or
  `after_interpreter_start`

Every witness hypothesis must also carry:

- `role = "witness_context"`
- `authority = "hypothesis"`
- `promotion_required = "interpreter_promoted_or_oracle_promoted"`
- `tool_authority = false`
- source, kind, digest, latency when available, and confidence when available

The Gemma interpreter may receive the raw witness string in its ephemeral
interpreter prompt because the point is to compare the frontend's claimed text
against the waveform. That raw string is still not durable/actionable evidence.
Persisted artifacts, status packets, oracle prompts, memory, files, Stripe,
NemoClaw, phone, message, and tool sinks should carry only the witness source,
timing, digest, confidence, arrival phase, speaker/channel binding,
adjudication, and any later promoted wording. If promoted wording is needed,
Gemma must emit it under `interpreter_promoted`, or the Hermes oracle must emit
it under `oracle_promoted`.

Provider field names do not carry authority. Adapter-edge names such as `stt`,
`stt_text`, `caption`, `transcript`, `query`, or `user_text` normalize into
`transcript_hypotheses[]` when raw audio exists. If raw audio is missing, the
turn is degraded text-only compatibility and cannot satisfy full-KAME or
high-risk action gates.

The normalizer must preserve only bounded provenance for those adapter-edge
names. A `provider_alias_key` such as `stt`, `caption`, `transcript`, `query`,
or `user_text` may appear on the hypothesis row and in compact audit output,
but it is never an authority label. The headless package should prove every
alias enters `transcript_hypotheses[]`, carries witness/hypothesis metadata,
and remains absent from `oracle_text` and all action sinks until
`interpreter_promoted` or `oracle_promoted` evidence exists.

## Gemma Interpreter Output

Gemma is prompted as an evidence adjudicator, not as a sibling ASR service. It
must treat raw audio as primary evidence and each transcript hypothesis as a
witness claim.

The interpreter output must keep these fields separate:

- `witness_adjudications`: per-hypothesis outcomes such as
  `accepted_as_supporting_evidence`, `corrected_by_audio`, or
  `rejected_or_diagnostic_only`
- `interpreter_promoted`: corrected wording, intent, entities, confidence, and
  compact audit metadata eligible for Hermes' active `/model`

Rejected hypotheses should include typed reasons, for example
`wrong_speaker`, `wrong_channel`, `stale_witness`, `timing_conflict`,
`low_energy_non_speech`, `waveform_conflict`, or `provider_conflict`.

## Runtime Rules

- A valid speech gate may trigger reflex acknowledgement before Moshi/STT text
  arrives.
- Interpreter submission should not wait for Moshi/STT text when raw audio,
  speech-gate metadata, and a reflex route are already available.
- Early, inline, and late witness text must merge into one evidence bundle and
  one oracle job lifecycle.
- A witness-only packet may create provisional/degraded state, but it must not
  schedule durable oracle work or create a second Hermes turn.
- Hermes' active `/model` remains the oracle. Voice config must not introduce a
  separate `oracle_model`.
- External frontends may call only narrow consult/bridge tools such as
  `ask_brain`, `ask_hermes_oracle`, `agent_consult`, or
  `openclaw_agent_consult`; they may not directly invoke Stripe, NemoClaw,
  phone, file, shell, memory, message, credential, or provisioning tools.

## Action Authority

Unpromoted witness text must be absent from:

- durable Hermes user history
- `oracle_text`
- Stripe spend reasons and provisioning payloads
- NemoClaw action packets
- phone/SMS/WhatsApp/Discord outbound message payloads
- memory writes
- file writes
- tool arguments

Those sinks may use only `interpreter_promoted` or `oracle_promoted` wording,
intent, entities, confidence, and compact labeled audit metadata.

## Implementation Work

The next implementation pass should prove:

1. The energy/noise gate suppresses silence, room tone, echo, harmonic artifacts,
   and low-energy non-speech before barge-in, interpreter scheduling, oracle
   scheduling, or transcript promotion.
2. Moshi/Open-S2S witness text can be attached before, with, or after the raw
   audio cut without producing duplicate Hermes turns.
3. Gemma receives raw audio before witness text and emits both
   `witness_adjudications` and `interpreter_promoted`.
4. Logs and artifacts show one `turn_id`, one `audio_segment_ref`, one
   `evidence_bundle_id`, and one oracle job lifecycle.
5. Unpromoted witness text is sink-checked out of Stripe, NemoClaw, phone,
   memory, file, message, tool, and durable-history payloads.
6. Latency metrics distinguish speech-end-to-reflex-ack, audio-cut-to-Gemma
   submission, witness arrival, Gemma promotion, oracle start, oracle first
   token, TTS first audio, and playback completion.
   Artifacts should expose these as `kame_latency_breakdown_segments_ms` with
   required keys: `speech_end_to_reflex_ack_ms`,
   `audio_cut_to_interpreter_submit_ms`, `witness_arrival_ms`,
   `interpreter_submit_to_promotion_ms`, `promotion_to_oracle_start_ms`,
   `oracle_start_to_first_token_ms`, `first_token_to_tts_first_audio_ms`,
   `tts_first_audio_to_playback_start_ms`, and
   `playback_start_to_completion_ms`.

7. A Moshi/Open-S2S witness can improve Gemma's promoted wording without
   appearing verbatim in `oracle_text` or any action sink before promotion.
8. A conflicting Moshi/Open-S2S witness is rejected or kept diagnostic-only
   while the raw-audio-grounded promoted wording continues to the oracle.
9. A text-only Moshi/Open-S2S packet is visibly degraded compatibility and does
   not satisfy full-KAME, Stripe, NemoClaw, phone, file, memory, message, or
   tool readiness.
10. The final package audit fails closed if witness text appears in any
    witness-assisted action sink value, or if a sink value lacks an
    `interpreter_promoted` or `oracle_promoted` source, even when summary
    booleans claim the sinks are clean.
11. Provider aliases `stt`, `caption`, `transcript`, `query`, and `user_text`
    all normalize into `transcript_hypotheses[]` with `provider_alias_key`
    provenance and fail package audit if they leak into `oracle_text` or any
    action sink.

The headless proof must include at least three witness timing cases for the
same contract: witness before raw audio, witness with raw audio, and witness
after interpreter start. All three cases must converge on one `turn_id`, one
`audio_segment_ref`, one `evidence_bundle_id`, one `evidence_merge_key`, and
one oracle job lifecycle. A fourth negative case must prove that witness-only
text without raw audio is labeled degraded and fails full-KAME/high-risk action
readiness.

## Non-Goals

- Do not build a separate ASR-first Hermes conversation beside the raw-audio
  interpreter path.
- Do not wait for Moshi/classic-ASR text before acknowledging the user.
- Do not treat Gemma as the Hermes oracle.
- Do not route unpromoted transcript-looking strings into business actions.
- Do not claim full-KAME readiness from text-only frontend compatibility.
