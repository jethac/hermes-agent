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

Provider field names do not carry authority. Adapter-edge names such as `stt`,
`stt_text`, `caption`, `transcript`, `query`, or `user_text` normalize into
`transcript_hypotheses[]` when raw audio exists. If raw audio is missing, the
turn is degraded text-only compatibility and cannot satisfy full-KAME or
high-risk action gates.

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

## Non-Goals

- Do not build a separate ASR-first Hermes conversation beside the raw-audio
  interpreter path.
- Do not wait for Moshi/classic-ASR text before acknowledging the user.
- Do not treat Gemma as the Hermes oracle.
- Do not route unpromoted transcript-looking strings into business actions.
- Do not claim full-KAME readiness from text-only frontend compatibility.
