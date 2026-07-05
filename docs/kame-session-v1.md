# KAME Session v1 Contract

Status: draft contract for external realtime frontends
Owner: Hermes realtime voice / VoiceOps
Protocol id: `kame_session_v1`
Design amendment: `docs/design/kame-witness-assisted-direct-audio.md`

This contract is the transport-neutral boundary for Discord voice,
VoiceClaw/OpenClaw-style frontends, phone/SIP, WhatsApp voice, desktop mic, and
future clients. It lets a fast frontend act as the reflex without inheriting
Hermes' tools or durable transcript authority.

## Current Role Assignment

`kame_session_v1` treats transcript-looking provider text as witness context,
not as the control path. A fast frontend may be Moshi/Open-S2S-like, may speak
an immediate acknowledgement, and may emit a string that looks like STT. If the
same packet family also carries raw audio, Hermes must bind the string to that
raw-audio cut as a hypothesis for the Gemma interpreter.

The normal path is:

```text
reflex floor control -> Gemma direct-audio interpreter -> Hermes active /model
```

It is not:

```text
STT transcript -> Hermes prompt
```

and it is not a separate "Gemma ASR" lane running beside the reflex. Gemma's
role is interpreter/adjudicator: compare the waveform, timing metadata,
speaker/channel evidence, reflex route, and witness text, then emit promoted
wording or reject the witness. The active Hermes `/model` should receive only
promoted transcript/intent/entity fields plus compact audit metadata.

## Canonical Packet Invariant

One accepted speech cut produces one evidence bundle and one oracle job
lifecycle. Raw audio is primary interpreter evidence. Any transcript-looking
text from Moshi, Open-S2S, VoiceClaw/OpenClaw, the reflex, or classic ASR is a
same-cut witness hypothesis inside that bundle.

The required normal-path input order is:

```text
raw_audio -> metadata -> reflex -> transcript_hypotheses
```

That ordering is not cosmetic. It prevents an adapter field named `transcript`,
`stt_text`, or `query` from becoming the durable user prompt before Gemma has
compared it with the waveform and speaker/channel/timing evidence. If raw audio
is present, provider text must be normalized into `transcript_hypotheses[]` with
`role = "witness_context"`, `authority = "hypothesis"`, and
`tool_authority = false`. If raw audio is absent, the turn is degraded
text-only compatibility and cannot satisfy full-KAME or high-risk action gates.

## Moshi / Open-S2S Witness Contract

The supported way to use Moshi-style output is to send the clipped waveform and
the frontend's transcript-looking text to the interpreter together. The
transcript is not "the ASR result" from Hermes' point of view. It is a
same-cut witness reading that helps Gemma compare what the live frontend
believed it heard against the actual waveform.

This is also the preferred way to use open speech-to-speech systems that expose
caption, transcript, or command text. The provider may call the field `stt`,
`transcript`, `caption`, `query`, or `user_text`; Hermes must normalize it by
role, not by provider vocabulary. If the same accepted speech cut has a
waveform, the provider text is a witness supplied to the direct-audio
interpreter. If the waveform is absent, the packet is degraded text-only
compatibility and cannot close full-KAME or high-risk action gates.

A normal Moshi/Open-S2S packet therefore has three binding requirements:

- the waveform and witness share the same `turn_id`, `audio_segment_ref`,
  `evidence_bundle_id`, and `evidence_merge_key`
- the witness appears in `transcript_hypotheses[]` with `source`,
  `latency_ms`, confidence when available, `arrival_phase`,
  `role = "witness_context"`, `authority = "hypothesis"`,
  `promotion_required = "interpreter_promoted_or_oracle_promoted"`, and
  `tool_authority = false`
- the interpreter response records an adjudication outcome before the text can
  shape durable/actionable wording: `accepted_as_supporting_evidence`,
  `corrected_by_audio`, or `rejected_or_diagnostic_only`

This lets Moshi/Open-S2S help with clipped prefixes, names, numbers,
code-switches, and rough intent without letting a fast hallucinated transcript
spend money, provision services, place calls, write memory/files, or become the
durable Hermes user message. If Moshi text arrives before the audio cut or
after the interpreter has started, it must merge into the same bundle and job.
If no raw audio exists, the turn is degraded compatibility mode and high-risk
action gates fail closed.

The adapter should treat this as witness attachment, not transcript
translation. A field named `stt`, `transcript`, `caption`, `query`, or
`user_text` may be useful, but it is never trusted by name. When raw audio is
present, the adapter binds that text to the same accepted cut as
`transcript_hypotheses[]` and records enough provenance for Gemma to adjudicate
it: source, source kind, text digest, arrival phase, confidence when available,
latency when available, speaker/channel guesses, and the shared
`evidence_merge_key`.

This preserves the useful part of Moshi/Open-S2S output without reintroducing
an ASR control path. Gemma sees the waveform and the frontend's best guess in
one packet, then decides whether the witness was supporting, incomplete,
wrong-speaker, wrong-channel, stale, low-energy, hallucinated, or contradicted
by the waveform. Until that decision creates `interpreter_promoted` or
`oracle_promoted` evidence, the original witness text is audit context only.

Adapter-edge vocabulary is intentionally lossy. The provider may name the text
field `stt`, `caption`, `transcript`, `query`, or `user_text`, but those names
are normalized away before Hermes builds durable state. The normalized row must
record `provider_alias_key` only as provenance and must still carry
`role = "witness_context"`, `authority = "hypothesis"`,
`promotion_required = "interpreter_promoted_or_oracle_promoted"`, and
`tool_authority = false`. Package audits should reject any implementation where
those aliases bypass `transcript_hypotheses[]`, appear in `oracle_text`, or
populate Stripe, NemoClaw, phone, file, memory, message, durable-history, or
tool sinks before promotion.

Implementation rule: prefer submitting the raw-audio interpreter packet over
waiting for Moshi/Open-S2S text. Witness text is valuable context, but it is not
the critical path for acknowledgement or interpreter scheduling. The adapter may
attach witness text before, with, or after raw audio, provided the merge keys
prove that every attachment belongs to the same accepted speech cut.

Runtime action-gate invariant: whenever an oracle job enters
`waiting_for_approval`, its full status and audit event must include a
`voiceops.runtime_kame_action_gate.v1` decision, even if the approval packet is
otherwise empty. Sparse approval waits fail closed until promoted interpreter or
oracle evidence has been consumed before the irreversible action and the tool
disclosure reference is present. Reflex-visible status may continue to redact
the approval packet.

### Witness Source Taxonomy

Use the narrowest non-authoritative kind the adapter can prove:

- `frontend_witness_hypothesis`: default for Moshi/OpenClaw/VoiceClaw/open-S2S
  text when the adapter cannot prove whether it came from the reflex model, a
  caption head, or an internal ASR head
- `s2s_transcript_hypothesis`: provider text from a known speech-to-speech
  transcript/caption output
- `reflex_transcript_hypothesis`: text emitted by the low-latency reflex model
  as part of floor control or provisional routing
- `classic_asr_hypothesis`: text emitted by a dedicated ASR service used for
  fallback, captions, diagnostics, or literal-evidence support

All four kinds remain `authority = "hypothesis"` until Gemma emits
`interpreter_promoted` evidence for the same raw-audio cut or the Hermes oracle
emits `oracle_promoted` evidence. Source kind is provenance, not permission.

Interpreter prompts for this packet must be phrased as witness adjudication, not
ASR selection. The prompt contract is:

- raw audio is primary evidence for the accepted speech cut
- `transcript_hypotheses[]` are frontend/reflex/ASR witness claims, not verified
  user text
- each hypothesis must receive an adjudication outcome and, if rejected, typed
  rejection reasons
- accepted or corrected witness content may influence only
  `interpreter_promoted` fields
- unpromoted witness text must not be copied into `oracle_text`, durable
  history, tool arguments, spend reasons, phone payloads, memory/file writes, or
  external messages

This is what lets Hermes provide Moshi/Open-S2S transcript text to Gemma beside
raw voice without reintroducing a parallel STT control path.

## Direct-Audio Interpreter Context Rule

The answer to "can we provide Moshi STT as context with raw voice?" is yes, but
only under the same-cut witness contract. A Moshi/Open-S2S string is useful
because it is a fast report of what the live interface believed it heard. It is
not the user message, not a second STT conversation, and not a scheduler.

The runtime should therefore avoid any "ASR evidence required" gate on the
normal KAME path. The required evidence is the accepted raw-audio cut, speech
gate metadata, reflex state, and later Gemma promotion. Moshi/Open-S2S text and
classic ASR text are optional witness inputs. They can improve interpretation,
serve captions, support diagnostics, or provide degraded compatibility when raw
audio is absent, but they are not prerequisites for reflex acknowledgement,
interpreter submission, or oracle job creation when raw audio is available.

This is the normative rule for open S2S frontends. If the frontend can provide
raw audio and transcript-like text for one accepted speech cut, Hermes should
send both to the Gemma interpreter. If it can provide only text, Hermes may use
that text for degraded compatibility, captions, or clarification, but it cannot
claim full KAME readiness and cannot release high-risk action gates.

The canonical interpreter envelope is therefore:

```text
accepted_speech_cut:
  raw_audio: required when full KAME is claimed
  metadata: VAD, energy, speaker, channel, transport, timing
  reflex: route, acknowledgement, interruption/playback state
  transcript_hypotheses: optional Moshi/Open-S2S/reflex/classic-ASR witnesses

Gemma interpreter output:
  interpreter_promoted: corrected transcript, intent, entities, confidence
  witness_adjudications: accepted, corrected, or rejected hypothesis outcomes

Hermes oracle input:
  active /model receives promoted wording plus compact labeled evidence only
```

If the packet lacks `raw_audio`, it is degraded compatibility even when the
frontend calls the text "STT." If the packet has `raw_audio`, every
transcript-looking string must be attached after metadata and reflex state as a
hypothesis. The implementation should never pick the earliest text field as the
Hermes prompt just because it arrived before Gemma finished.

The normal runtime should therefore build this packet:

```text
accepted raw-audio cut
  + VAD/energy/speaker/channel metadata
  + reflex route and acknowledgement already spoken
  + Moshi/Open-S2S/reflex/classic-ASR transcript hypotheses
  -> Gemma direct-audio interpreter
  -> interpreter_promoted wording, intent, entities, and confidence
  -> Hermes active /model oracle
```

This is a three-tier system with sensor fan-in, not a four-tier
reflex-plus-STT-plus-interpreter-plus-oracle stack. The speech gate and reflex
own live timing. Gemma owns post-cut direct-audio adjudication. Hermes' active
`/model` owns durable reasoning and tools. Transcript-looking text from Moshi,
OpenClaw, VoiceClaw, the reflex, or classic ASR is a witness sensor attached to
the accepted cut. It can help Gemma recover clipped prefixes, names, numbers,
code-switched phrases, or rough intent, but it remains hypothesis authority
until `interpreter_promoted` or `oracle_promoted` fields exist.

Adapters must not wait for Moshi/classic-ASR text before acknowledging the user
or creating a raw-audio interpreter request when an accepted speech cut and
reflex route are already available. If witness text arrives before the cut, hold
it on the pending bundle. If it arrives with the cut, include it after raw
audio, metadata, and reflex state. If it arrives after the interpreter starts,
append it as late evidence on the same bundle. All three timings must preserve
one `turn_id`, one `audio_segment_ref`, one `evidence_bundle_id`, and one
oracle job lifecycle.

The implementation contract is "attach, do not translate." If the frontend has
both a waveform and Moshi/Open-S2S text, the adapter should preserve both
signals in one interpreter packet rather than turning the text into a Hermes
message first:

```json
{
  "audio": {
    "segment_ref": "artifact://voice/turn-042.wav",
    "time_range_ms": [12800, 15320],
    "primary_interpreter_evidence": true
  },
  "interpreter_input_order": [
    "raw_audio",
    "metadata",
    "reflex",
    "transcript_hypotheses"
  ],
  "transcript_hypotheses": [
    {
      "kind": "frontend_witness_hypothesis",
      "source": "moshi",
      "text_digest": "sha256:redacted",
      "role": "witness_context",
      "authority": "hypothesis",
      "promotion_required": "interpreter_promoted_or_oracle_promoted",
      "tool_authority": false,
      "arrival_phase": "with_raw_audio",
      "latency_ms": 94,
      "confidence": null,
      "speaker_or_actor_ref": "speaker:jetha",
      "channel_or_surface_ref": "discord:general"
    }
  ]
}
```

The raw witness string may be present in the ephemeral Gemma interpreter request
and in a redacted, access-controlled source artifact for audit. That is the
only place raw Moshi/Open-S2S text is allowed before promotion: the interpreter
needs the literal string so it can compare what the frontend believed it heard
against the waveform. Normal Hermes oracle prompts, egress messages,
Stripe/NemoClaw reasons, phone payloads, memory writes, file writes, durable
history, and tool arguments should carry only the digest, source/timing
metadata, adjudication outcome, and promoted wording.

This is the allowed exception to the "no raw witness text outside action
authority" rule. The Gemma request may contain the literal Moshi/Open-S2S or
classic-ASR string because comparison requires the actual claim. That privilege
does not extend to status packets, durable history, Hermes `/model` prompts, or
action sinks. Those surfaces should see a digest and adjudication metadata until
promotion exists.

This distinction is mandatory for the witness-assisted design. The interpreter
can see raw witness text; the rest of Hermes sees redacted witness metadata
until `interpreter_promoted` or `oracle_promoted` wording exists.

### Adapter Checklist

For each accepted speech cut, an adapter should execute this sequence:

1. Create or locate the pending bundle by `turn_id`, `audio_segment_ref`, and
   `evidence_merge_key`.
2. Store the bounded waveform as `primary_interpreter_evidence`.
3. Attach VAD, energy, speaker, channel, transport, and timing metadata.
4. Attach reflex state: acknowledgement already spoken, provisional route,
   interruption/playback state, and any provisional intent.
5. Normalize any provider field named `stt`, `transcript`, `caption`, `query`,
   `user_text`, or equivalent into `transcript_hypotheses[]`.
6. Preserve literal witness text only inside the interpreter request and
   controlled source artifacts; expose only digest and metadata elsewhere until
   promotion.
7. Submit the interpreter packet without waiting for late witness text when raw
   audio and reflex state are available.
8. Merge late witness text into the same bundle and job as late evidence; never
   schedule a second Hermes turn from the witness.
9. Accept only `interpreter_promoted` or `oracle_promoted` fields into durable
   user history, oracle prompts, tool arguments, Stripe/NemoClaw packets, phone
   payloads, memory/file writes, or external messages.

This checklist is the concrete interpretation of "provide Moshi STT to the
interpreter along with raw voice." The Moshi/Open-S2S text is allowed to be
literal context for Gemma. It is not allowed to become the user prompt or
action payload by adapter convenience.

### Three-Tier Runtime Acceptance

A conforming full-KAME turn proves these properties:

- the energy/noise gate accepted one bounded speech cut before interpreter or
  oracle scheduling
- the reflex acknowledgement did not wait for Moshi, Open-S2S, or classic-ASR
  witness text
- the interpreter packet used `mode = "witness_assisted_direct_audio"` and
  `interpreter_input_order = ["raw_audio", "metadata", "reflex",
  "transcript_hypotheses"]`
- every transcript-looking provider field normalized into
  `transcript_hypotheses[]` with hypothesis authority
- early, inline, and late witnesses merged into the same bundle and oracle job
- the Hermes `/model` oracle started only after `interpreter_promoted` or
  `oracle_promoted` wording existed
- Stripe, NemoClaw, phone, file, memory, message, and tool sinks contained no
  unpromoted witness text

An implementation that schedules one Hermes turn from the witness transcript
and another from the waveform is non-conforming, even if both turns are later
deduplicated. The accepted speech cut, not the first text string, is the unit of
work.

### Witness-Assisted Direct-Audio Profile

Adapters that can supply both waveform and transcript-like text should use the
`witness_assisted_direct_audio` profile. The profile has one control rule:
submit one raw-audio interpreter packet for the accepted speech cut, then attach
all same-cut transcript-like provider output as witness hypotheses. The adapter
must not translate the provider text into a Hermes user message before Gemma
adjudicates the packet.

Required profile fields:

- `mode = "witness_assisted_direct_audio"`
- `turn_id`, `audio_segment_ref`, `evidence_bundle_id`, and
  `evidence_merge_key`
- `audio.primary_interpreter_evidence = true`
- `interpreter_input_order = ["raw_audio", "metadata", "reflex",
  "transcript_hypotheses"]`
- `transcript_hypotheses[]` rows with `source`, `kind`, `text_digest`,
  `role = "witness_context"`, `authority = "hypothesis"`,
  `promotion_required = "interpreter_promoted_or_oracle_promoted"`,
  `tool_authority = false`, `arrival_phase`, `speaker_or_actor_ref`, and
  `channel_or_surface_ref`

Recommended profile fields:

- `latency_ms` and `confidence` when the provider reports them
- `audio_time_range_ms` for the portion of the speech cut the witness claims
  to describe
- `partial_state`, `superseded_partial_count`, and
  `superseded_partial_text_digests` when streaming partials collapse into a
  final same-source hypothesis
- `energy_gate` and `vad` metadata proving that silence, echo, and low-energy
  artifacts did not create the accepted cut

The interpreter response must keep witness decisions separate from promoted
speech:

```json
{
  "witness_adjudications": [
    {
      "text_digest": "sha256:redacted",
      "source": "moshi",
      "adjudication": "corrected_by_audio",
      "rejection_reasons": []
    }
  ],
  "interpreter_promoted": {
    "corrected_transcript_digest": "sha256:redacted",
    "intent": "provision_budgeted_phone_handoff",
    "entities": ["budget", "phone_handoff"],
    "confidence": 0.82
  }
}
```

Only `interpreter_promoted` or later `oracle_promoted` fields may feed Hermes'
active `/model`, durable history, Stripe/NemoClaw spend reasons, phone payloads,
memory/file writes, external messages, or tool arguments. Raw witness strings
remain audit context even when they helped Gemma produce the promoted fields.

## Current Contract

`kame_session_v1` carries one accepted speech cut, not one transcript. The
preferred packet contains the clipped waveform plus all same-cut witness text:

1. raw audio reference and timing
2. VAD, energy, speaker, channel, and transport metadata
3. reflex route and acknowledgement already spoken
4. transcript hypotheses from Moshi/open-S2S, VoiceClaw/OpenClaw, reflex text,
   or classic ASR

In other words: yes, a Moshi/Open-S2S "STT" string can travel beside the raw
voice. It is interpreter context, not the user's message. A full-KAME packet
with both signals must preserve the same speech-cut identity for the waveform
and the witness string, then ask the interpreter to accept, correct, or reject
the witness before any durable/actionable field is produced.

This allows a Moshi-style frontend to provide both audio and an STT-looking
string without making the string authoritative. The string is a witness claim:
what that frontend believed it heard. It is sent to the Gemma interpreter as
context beside the waveform. It must not become `oracle_text`, durable user
history, a spend reason, a phone payload, a memory/file write, an external
message, or a tool argument unless Gemma or the Hermes oracle promotes it.

If raw audio exists, witness text must merge into the same `turn_id`,
`audio_segment_ref`, `evidence_bundle_id`, and `evidence_merge_key`. Early,
inline, and late witness packets update the same bundle and job. They do not
create a second Hermes turn or second oracle job. If raw audio is missing, the
packet is degraded text-only compatibility mode and cannot satisfy full KAME or
high-risk action gates.

Same-turn convergence is required even when the first packet was text-only and
the raw audio arrives moments later. The first packet may create provisional
queue state, but a later packet for the same accepted speech cut must update
that existing job through the evidence bundle instead of emitting a second
oracle request. In event terms, Hermes should expose the late/raw-audio merge
as a bounded job update, not as another `INTERFACE_ORACLE_REQUEST`.

The canonical voice turn is one raw-audio evidence bundle with optional sensor
attachments. A Moshi/open-S2S transcript, VoiceClaw/OpenClaw text, or classic
ASR string is a witness attached to that bundle. It is not a separate Hermes
message, not a scheduler input, and not action authority.

This is the compatibility rule for Moshi-style systems: if a frontend can send
both waveform and transcript-looking text, Hermes should send both to the Gemma
interpreter in the same packet. The transcript-looking text records what the
frontend believed it heard; the waveform remains primary. If the frontend can
send only text, the packet is degraded text-only compatibility mode, not
full-KAME evidence and not a high-risk action gate input.

The preferred Moshi/Open-S2S shape is therefore not "run STT, then ask
Hermes." It is "send the clipped waveform plus the frontend's transcript
hypothesis to the interpreter." The hypothesis is useful context for Gemma, but
it remains `authority = "hypothesis"` until the interpreter promotes
raw-audio-grounded wording.

This contract intentionally supports a three-tier sensor-fan-in system rather
than a fourth STT authority lane. A fast reflex may emit a transcript-looking
string while it controls the floor. A Gemma-style interpreter then receives the
same accepted raw-audio cut plus that string and any other witness hypotheses.
Hermes' active `/model` only receives promoted wording, intent, entities, and
compact labeled audit context. Raw witness strings must not be replayed as the
oracle prompt.

2026-07-05 clarification: a Moshi/open-S2S transcript can be valuable precisely
because it is paired with the raw voice. Treat it like a same-cut witness, not a
replacement for STT and not an extra Hermes message. The normalized packet must
keep `audio.segment_ref` as primary evidence and place any Moshi/OpenClaw/
VoiceClaw/reflex/classic-ASR text under `transcript_hypotheses[]` with
`role = "witness_context"`, `authority = "hypothesis"`, and
`tool_authority = false`. Provider field names such as `stt_text`,
`transcript`, or `query` are adapter-edge names only; they must not survive as
verified user text, `oracle_text`, action arguments, or durable history unless
the interpreter or oracle promotes them.

2026-07-05 adapter rule: a Moshi/Open-S2S transcript may be sent with raw voice,
but only as interpreter context. Normalize it into `transcript_hypotheses[]`
with `kind = "frontend_witness_hypothesis"` unless the adapter can prove a
narrower producer, set `source` to the provider name, and include
`role = "witness_context"`, `authority = "hypothesis"`, and
`tool_authority = false`. The same `turn_id`, `audio_segment_ref`,
`evidence_bundle_id`, and `evidence_merge_key` must bind the witness to the
waveform. The witness can help Gemma recover clipped prefixes, names, numbers,
and code-switching, but it cannot become durable user text, `oracle_text`, or
an action payload before interpreter/oracle promotion.

The same rule applies when Gemma emits transcript-like text. In KAME, Gemma is
the direct-audio interpreter, not a second ASR service racing the frontend. Its
corrected transcript, entities, and intent become useful to Hermes only when
they are emitted as `interpreter_promoted` fields for the accepted raw-audio cut.
Moshi/Open-S2S text, reflex captions, and classic ASR text remain
`transcript_hypotheses[]` that Gemma may use, correct, or reject.

For adapters that expose a Moshi-style "STT" field, the field name is
misleading from Hermes' point of view. Hermes should normalize it as witness
context for the same direct-audio interpreter packet, not as a separate ASR
result and not as the user's prompt. The raw waveform, speech gate, speaker
metadata, and reflex route are the authoritative inputs; the Moshi text is a
witness claim the interpreter can use, correct, or reject.

The practical adapter rule is simple: if the frontend has raw audio, send raw
audio. Do not downgrade the turn to text just because the frontend also has a
Moshi/open-S2S transcript. Build one interpreter packet with the accepted
waveform first, then metadata, reflex state, and transcript hypotheses. The
Moshi text should help the interpreter compare, recover, and explain; it should
not replace the waveform or short-circuit interpretation.

## Raw-Audio Promotion Enforcement

For raw-audio or witness-assisted direct-audio turns, promotion is not a field
the frontend can self-attest. A request may carry `source`,
`intent_source`, `transcript_source`, or `oracle_text_source` values that look
like `gemma_interpreter`, `interpreter_promoted`, or `oracle_promoted`, but
those labels are advisory until Hermes has a job-owned interpreter or oracle
evidence record for the same `turn_id`, `audio_segment_ref`,
`evidence_bundle_id`, and `evidence_merge_key`.

The scheduler may accept the job and expose it to the reflex immediately, but it
must keep the Hermes `/model` runner out of durable/actionable work until
promoted evidence is attached and consumed. Hypothesis-only packets, including
Moshi/Open-S2S text, VoiceClaw/OpenClaw text, reflex captions, and classic ASR
text, can update the bundle but cannot unlock oracle execution by naming
themselves as promoted. This rule is what keeps external frontends useful as
hearing witnesses without letting a fast caption become a spend reason, phone
payload, tool argument, memory/file write, external message, or durable user
turn.

Only when the frontend genuinely lacks the waveform should Hermes enter
degraded text-only compatibility mode. That mode may ask clarifying questions,
show captions, or create a provisional audit trail, but it cannot claim full
KAME readiness or authorize high-risk work. If raw audio arrives later for the
same accepted cut, the degraded packet must merge into the existing evidence
bundle instead of creating a second Hermes turn.

## Implementation Checklist

Use this checklist for every Discord, phone, WhatsApp, VoiceClaw/OpenClaw, or
Moshi/Open-S2S adapter that claims full KAME behavior:

- create one accepted speech cut from the VAD/energy gate before interpreter
  scheduling
- keep raw audio as `primary_interpreter_evidence`
- normalize every transcript-looking provider field into
  `transcript_hypotheses[]`, including fields named `stt_text`, `transcript`,
  `caption`, `query`, or provider-specific equivalents
- bind waveform and witnesses with the same `turn_id`, `audio_segment_ref`,
  `evidence_bundle_id`, and `evidence_merge_key`
- preserve interpreter input order as `raw_audio`, `metadata`, `reflex`,
  `transcript_hypotheses`
- set each witness row to `role = "witness_context"`,
  `authority = "hypothesis"`,
  `promotion_required = "interpreter_promoted_or_oracle_promoted"`, and
  `tool_authority = false`
- retain source, kind, digest, latency when available, confidence when
  available, arrival phase, speaker binding, and channel binding for each
  witness
- require Gemma/interpreter adjudication for every witness before that witness
  can shape promoted wording
- pass only `interpreter_promoted` or `oracle_promoted` wording, intent,
  entities, confidence, and compact audit metadata to Hermes' active `/model`
- reject or downgrade text-only packets as degraded compatibility unless raw
  audio later merges into the same bundle

The adapter fails the contract if a Moshi/Open-S2S/reflex/classic-ASR string
directly becomes `oracle_text`, a durable user message, a Stripe/NemoClaw spend
reason, a phone payload, a memory/file write, an external message, or a tool
argument before promotion. It also fails if early, inline, or late witness text
creates a second Hermes turn or duplicate oracle job for the same accepted
speech cut.

## Roles

External frontends may provide:

- live audio and playback control
- reflex acknowledgements and provisional routes
- transcript-looking witness text from Moshi/open-S2S, VoiceClaw/OpenClaw, or
  classic ASR side channels
- `ask_brain` / `openclaw_agent_consult` style oracle requests
- cancellation and status correlation ids

External frontends may submit reflex job envelopes for Hermes to normalize, but
transcript-looking fields inside those envelopes remain
`transcript_hypotheses[]`. A full KAME envelope requires the same raw-audio
interpreter packet described below. Text-only envelopes are degraded
compatibility and cannot create durable user text or action authority.

External frontends must not receive or execute direct Hermes file, shell,
memory, payment, provisioning, phone, message, or credential tools.
The only bridge tools accepted at this boundary are `ask_brain`,
`ask_hermes_oracle`, `agent_consult`, and `openclaw_agent_consult`. Any other
tool name, including Stripe, NemoClaw, phone, file, shell, memory, message, or
provider-provisioning tools, must return a rejected `tool.result` with
`accepted = false` and must not create an oracle job. Tool authority starts only
after Hermes' active oracle receives promoted evidence and routes a normal
Hermes tool call.

The intended deployment may have three model roles, but it must still produce
one logical Hermes turn per accepted speech cut:

- fast reflex: floor control, acknowledgement, route, and optional witness text
- Gemma interpreter: raw-audio-grounded transcript/intent/entity promotion
- Hermes active `/model`: oracle authority for tools and durable outcomes

No frontend transcript source, including Moshi "STT," may create a second turn
or bypass the Gemma interpreter when raw audio exists.

## Input Events

Every spoken turn should use one `turn_id`. If raw audio is available, every
witness transcript for that speech cut must share the same `audio_segment_ref`.
The accepted raw-audio cut must come from the configured VAD/energy/noise gate
before Hermes schedules direct-audio interpretation; silence, playback echo,
room tone, and low-energy artifacts are not valid interpreter input by
themselves.

The frontend may send evidence in any of these orders:

- witness text before the accepted audio cut exists
- witness text in the same packet as the accepted audio cut
- witness text after the interpreter request has already started

All three cases must converge on the same `turn_id`, `audio_segment_ref`,
`evidence_bundle_id`, and `evidence_merge_key`. Early witness text is held on a
pending bundle. Late witness text is appended as late evidence on the existing
bundle. Neither case may create a duplicate oracle job or durable user turn.
If a provisional external `ask_brain` envelope has already started an oracle job
for that speech cut, later raw-audio or Moshi/Open-S2S witness evidence must
coalesce into the running job and refresh the request metadata used by the
oracle. The externally visible proof is one accepted/started/completed oracle
job, one durable request record, and a follow-up update marked as an evidence
bundle merge.

Minimum external frontend evidence/job-envelope shape:

```json
{
  "protocol": "kame_session_v1",
  "session_id": "voice-session-001",
  "turn_id": "voice-turn-001",
  "audio_segment_ref": "artifact://frontend/turn-001.wav",
  "evidence_bundle_id": "kame-bundle-001",
  "evidence_merge_key": "kame-merge-session-turn-audio",
  "tool_name": "ask_brain",
  "tool_call_id": "frontend-call-001",
  "audit_id": "frontend-audit-001",
  "source_audit_id": "discord-audit-001",
  "parent_audit_id": "voiceops-root-001",
  "arguments": {
    "provisional_request_summary": "prepare the phone handoff",
    "reflex_intent": "Prepare the phone handoff.",
    "kind": "reflex_hypothesis",
    "authority": "hypothesis",
    "tool_authority": false,
    "interface_already_said": "I'm preparing the handoff.",
    "requested_response_style": {"spoken": true, "max_sentences": 1}
  },
  "audio": {
    "segment_ref": "artifact://frontend/turn-001.wav",
    "codec": "pcm_s16le",
    "sample_rate_hz": 16000,
    "channels": 1,
    "time_range_ms": [120, 1840],
    "vad": {"speech_start_ms": 120, "speech_end_ms": 1840},
    "authority": "primary_audio"
  },
  "speaker": {
    "platform": "discord",
    "channel_user_id": "redacted-user",
    "display_name": "redacted",
    "is_bot": false
  },
  "channel": {
    "transport": "discord_voice",
    "guild_id": "redacted-guild",
    "channel_id": "redacted-channel",
    "surface": "desk_voice"
  },
  "interpreter_input_order": ["raw_audio", "metadata", "reflex", "transcript_hypotheses"],
  "interpreter_prompt_policy": {
    "version": "raw_audio_compare_v1",
    "raw_audio_primary": true,
    "witness_transcripts_context_only": true,
    "require_witness_adjudication": true
  },
  "transcript_hypotheses": [
    {
      "kind": "frontend_witness_hypothesis",
      "source": "moshi",
      "role": "witness_context",
      "text": "prepare phone handoff",
      "partial": false,
      "confidence": 0.78,
      "latency_ms": 140,
      "arrival_phase": "with_raw_audio",
      "audio_time_range_ms": [120, 1840],
      "speaker_guess": {"platform": "discord", "channel_user_id": "redacted-user"},
      "channel_guess": {"transport": "discord_voice", "channel_id": "redacted-channel"},
      "authority": "hypothesis",
      "promotion_required": "interpreter_promoted_or_oracle_promoted",
      "tool_authority": false
    }
  ]
}
```

`audio.segment_ref` is a transport-local copy of the canonical
`audio_segment_ref`; when both are present they must match. `evidence_bundle_id`
is stable for the logical speech cut, including witness-before-audio and
late-witness updates. `evidence_merge_key` is the audio-aware merge proof over
the session, turn, and `audio_segment_ref`.

`interpreter_input_order` and `interpreter_prompt_policy` are required whenever
raw audio is available. They make the Moshi/Open-S2S context rule machine
checkable: the interpreter receives the waveform first, then metadata, then
reflex state, then transcript hypotheses, and the prompt explicitly treats
those hypotheses as context to compare against audio. A frontend witness may
help Gemma recover a clipped prefix, name, number, or code-switch, but the
packet must still require witness adjudication before any durable wording or
tool-critical field is promoted.

`tool_name = "ask_brain"` is adapter-edge compatibility, not a direct Hermes
tool grant. `arguments.provisional_request_summary` and
`arguments.reflex_intent` are provisional frontend route/context fields for the
job envelope. They are allowed to help create queue state or a spoken
acknowledgement, but they are not durable user text, not `oracle_text`, and not
tool/action authority. Oracle text and high-risk action text must come from
`interpreter_promoted` or `oracle_promoted` evidence.
Queue state may be created from the reflex route plus accepted raw-audio packet.
If the summary was derived from Moshi/Open-S2S/VoiceClaw transcript-looking
text, it remains hypothesis context and must not schedule a job or populate
`oracle_text` by itself.

If the adapter cannot prove whether transcript-looking text came from the live
reflex model or a sibling caption/S2S lane, it must use
`frontend_witness_hypothesis`. Vendor names belong in `source`, not in the
authority model.

Adapters must apply the speech gate before treating transcript-looking text as
turn evidence. If a packet carries explicit negative speech evidence such as
`speech_confirmed = false`, `vad_speech = false`, or low RMS/duration below the
configured speech gate, its witness text is diagnostic only. It must not trigger
barge-in, interpreter scheduling, oracle scheduling, durable transcript writes,
or tool/action fields. A later accepted speech cut may include a new witness
hypothesis with the same source, but the rejected low-energy text must not be
replayed into that turn.

When raw audio exists, transcript hypotheses are attached to that same turn as
interpreter context. They do not create a second Hermes turn, do not schedule a
parallel oracle request, and do not become durable transcript text unless the
interpreter or oracle explicitly promotes them.

The same rule applies when the transcript-looking text is produced by the
reflex model itself. A reflex transcript can explain why the interface routed a
turn and can help the interpreter recover clipped starts, but it is still one
entry in `transcript_hypotheses[]`. It does not outrank the waveform and must
not be replayed as `arguments.query`, `oracle_text`, durable history, or a tool
argument unless promoted by interpreter/oracle evidence.

The interpreter request should preserve input ordering:

1. primary raw audio reference and timing
2. speaker/channel/VAD/energy metadata
3. reflex route and acknowledgement already spoken
4. transcript hypotheses from Moshi/open-S2S, VoiceClaw/OpenClaw, reflex, or
   classic ASR

This ordering is part of the trust model. It tells the interpreter that
transcript-looking text is context for comparing against the waveform, not the
user message of record.

Normalized packets should carry that ordering explicitly as
`interpreter_input_order = ["raw_audio", "metadata", "reflex", "transcript_hypotheses"]`.
This is intentionally boring metadata, but it prevents later adapters from
turning Moshi text, classic ASR, or a reflex caption into the first prompt item
Gemma sees. If raw audio is absent, the packet must omit `raw_audio` from the
order, set degraded compatibility mode, and fail full-KAME/action-gate proofs.

Normalized packets should also carry
`interpreter_prompt_policy.version = "raw_audio_compare_v1"` for the normal
path. That policy means the interpreter is asked to compare witness transcripts
against raw audio, not to trust them as pre-transcribed user text. Adapters that
cannot provide raw audio must use a degraded policy value and must not claim
full KAME readiness.

## Transcript Hypothesis Semantics

Adapters should prefer these `kind` values:

- `frontend_witness_hypothesis`: ambiguous transcript-looking output from a
  Moshi/open-S2S, VoiceClaw, OpenClaw, or hosted realtime frontend when the
  exact producer is not proven.
- `reflex_transcript_hypothesis`: text proven to come from the live reflex
  model's own hearing.
- `s2s_transcript_hypothesis`: text proven to come from a named S2S witness
  producer associated with the same accepted raw-audio cut. This is a
  provenance label inside `transcript_hypotheses[]`, not a separate STT lane or
  control path.
- `classic_asr_hypothesis`: text from a dedicated ASR provider retained for
  captions, diagnostics, literal checks, or degraded fallback.

Prefer `frontend_witness_hypothesis` for Moshi/Open-S2S text unless the adapter
can prove the narrower producer. Narrower labels are useful for audit and
quality scoring only; they do not change scheduling, authority, or packet
merge behavior.

Every hypothesis must carry `role = "witness_context"`,
`authority = "hypothesis"`,
`promotion_required = "interpreter_promoted_or_oracle_promoted"`, and
`tool_authority = false`. These exact fields are the normalized contract, not
just explanatory labels. They let downstream artifact checks prove that Moshi,
OpenClaw, VoiceClaw, reflex, or classic-ASR text was provided as interpreter
context rather than promoted by name. Source names such as `moshi`, `openclaw`,
`voiceclaw`, `riva`, or `cartesia` are provenance, not authority labels.
Every hypothesis should also carry `arrival_phase` when the adapter can
determine it: `before_raw_audio`, `with_raw_audio`, or
`after_interpreter_start`. Arrival phase is merge evidence, not authority. It
must survive status updates, job evidence bundles, readiness reports, and
package audits so Hermes can prove early, inline, and late witness text all
attached to one speech cut without spawning duplicate oracle jobs.

Partial hypotheses are active only until a same-source, same-kind final
hypothesis for the same speech cut arrives. The final hypothesis replaces the
partial in active interpreter context; the partial survives only as superseded
provenance for audit and latency debugging.

Do not add a separate hypothesis kind for "Gemma ASR" in the normal path. Gemma
is the interpreter; its transcript-like output belongs in the interpreter result
with `authority = "interpreter_promoted"` when accepted. If a separate Gemma
service is deliberately run as a diagnostic ASR experiment, normalize its output
as `classic_asr_hypothesis` or another explicit hypothesis source and keep
`tool_authority = false`.

In multi-speaker sessions, a transcript hypothesis is not attachable to a
speech cut merely because it arrived near the same time. The adapter must bind
the hypothesis to the accepted cut by `turn_id`, `audio_segment_ref`, and the
best available speaker/channel evidence. If the hypothesis speaker or channel
does not match the accepted cut, the interpreter may retain it only as
`rejected_or_diagnostic_only` evidence with an explicit reason such as
`wrong_speaker`, `wrong_channel`, `stale_witness`, or `ambiguous_speaker`.
Ambiguous speaker evidence must not become durable user text or action
authority.

Every `rejected_or_diagnostic_only` hypothesis must carry
`rejection_reasons[]`. Valid reason codes are:

- `ambiguous_speaker`
- `wrong_speaker`
- `wrong_channel`
- `stale_witness`
- `timing_conflict`
- `low_energy_non_speech`
- `waveform_conflict`
- `provider_conflict`

These reason codes are part of the evidence contract. They make it possible to
prove that Moshi/Open-S2S, VoiceClaw/OpenClaw, reflex, or classic-ASR text was
rejected because of concrete audio/session evidence, not merely ignored after
it failed to fit a later oracle response.

The interpreter must adjudicate each active hypothesis with one of:

- `accepted_as_supporting_evidence`
- `corrected_by_audio`
- `rejected_or_diagnostic_only`

Only accepted or corrected hypotheses may contribute to promoted interpreter
fields, and only after the interpreter emits those promoted fields. Rejected or
diagnostic hypotheses remain visible in audit records but cannot become durable
history, tool arguments, spend reasons, provider selections, phone payloads,
memory writes, file writes, or external messages.

The interpreter result should preserve the witness decision separately from the
promoted wording. The minimum result shape is:

```json
{
  "authority": "interpreter_promoted",
  "interpreter_input_order": ["raw_audio", "metadata", "reflex", "transcript_hypotheses"],
  "interpreter_corrected_transcript": "prepare the phone handoff",
  "interpreter_normalized_intent": "Prepare a phone handoff.",
  "promoted_fields_authority": {
    "interpreter_corrected_transcript": "interpreter_promoted",
    "interpreter_normalized_intent": "interpreter_promoted"
  },
  "witness_adjudications": [
    {
      "source": "moshi",
      "kind": "frontend_witness_hypothesis",
      "arrival_phase": "with_raw_audio",
      "outcome": "corrected_by_audio",
      "rejection_reasons": [],
      "authority": "hypothesis",
      "tool_authority": false
    }
  ]
}
```

This separation is what lets Hermes include a Moshi/Open-S2S transcript beside
raw voice without making that text the user's message. A useful witness can
shape promoted wording only through `interpreter_promoted` fields. A bad
witness remains visible for debugging and safety review, but does not reach the
oracle as durable user text.

### Moshi Witness Attachment

When a Moshi-like frontend exposes both the clipped waveform and a transcript
string for the same user utterance, the adapter should send both to Hermes. The
waveform remains `audio.segment_ref` and primary interpreter evidence. The text
should be stored in `transcript_hypotheses[]` with:

- `source = "moshi"` or the narrower vendor/source identifier
- `kind = "frontend_witness_hypothesis"` unless the adapter can prove a more
  specific producer
- `authority = "hypothesis"`
- `role = "witness_context"`
- `promotion_required = "interpreter_promoted_or_oracle_promoted"`
- `tool_authority = false`
- timing, confidence, partial/final state, speaker guess, and channel guess
  when available
- arrival phase: `before_raw_audio`, `with_raw_audio`, or
  `after_interpreter_start`

This Moshi text is useful because it tells Gemma what the live interface model
believed it heard. It must be attached to the same `turn_id`,
`audio_segment_ref`, `evidence_bundle_id`, and `evidence_merge_key` as the raw
audio. It may arrive before, with, or after the accepted cut, but it must not
create a second Hermes turn, replace the waveform, block reflex
acknowledgement, or become `oracle_text` before interpreter or oracle
promotion.

Arrival phase controls only merge handling. A before-cut witness waits on the
pending bundle. A with-cut witness is inline interpreter context. An
after-start witness is late evidence on the same job. None of those phases may
start a standalone Hermes turn, create a second oracle job, or make the witness
eligible for durable history or action sinks without interpreter/oracle
promotion.

If the Moshi text conflicts with the waveform, speaker/channel metadata,
energy/VAD decision, or current session state, the interpreter should mark it
`rejected_or_diagnostic_only` with a typed reason. If it helps recover a clipped
prefix, name, number, or code-switched phrase, the promoted wording still comes
from `interpreter_promoted` fields, not directly from the Moshi string.

Adapters should avoid naming this field `transcript`, `stt_text`, or
`oracle_text` in normalized packets. Use `transcript_hypotheses[]` with
`kind = "frontend_witness_hypothesis"` unless a narrower producer is proven.
This makes the raw audio plus witness text visible to the interpreter without
accidentally turning the witness into a durable Hermes user message.

## Output Events

Hermes emits normalized session/job events back to the frontend. The canonical
wire event names are dotted `kame_session_v1` event strings. Uppercase names are
legacy compatibility aliases for internal enum names and older
VoiceClaw/OpenClaw-style adapters.

- accepted placeholder: `tool.result` with `accepted = true`, `job_id`,
  `tool_call_id`, and provider/source fields
- oracle lifecycle: `oracle.job.accepted`, `oracle.job.queued`,
  `oracle.job.started`, `oracle.job.waiting_for_approval`,
  `oracle.job.completed`, `oracle.job.failed`, `oracle.job.cancelled`
- cancellation feedback: `interface.oracle.cancel_job`
- bounded updates: `interface.oracle.update_job`
- speech/playback events from the normal realtime voice stream

Compatibility aliases:

| Canonical event | Compatibility alias |
| --- | --- |
| `tool.result` | `TOOL_RESULT` |
| `oracle.job.accepted` | `ORACLE_JOB_ACCEPTED` |
| `oracle.job.queued` | `ORACLE_JOB_QUEUED` |
| `oracle.job.started` | `ORACLE_JOB_STARTED` |
| `oracle.job.waiting_for_approval` | `ORACLE_JOB_WAITING_FOR_APPROVAL` |
| `oracle.job.completed` | `ORACLE_JOB_COMPLETED` |
| `oracle.job.failed` | `ORACLE_JOB_FAILED` |
| `oracle.job.cancelled` | `ORACLE_JOB_CANCELLED` |
| `interface.oracle.cancel_job` | `INTERFACE_ORACLE_CANCEL` |
| `interface.oracle.update_job` | `INTERFACE_ORACLE_UPDATE` |

Terminal events must preserve `tool_call_id`, `audit_id`, `source_audit_id`,
and `parent_audit_id` so the frontend can correlate placeholders, progress,
and final results without treating witness text as durable user history.

## Authority Rules

- Raw audio is the primary interpreter evidence when `audio.segment_ref` exists.
- Transcript hypotheses are clues for the Gemma interpreter, not durable user
  messages.
- Moshi/open-S2S/reflex/classic-ASR text may arrive asynchronously as witness
  or fallback context, but it must not block reflex acknowledgement or
  oracle-job envelope creation when the accepted raw audio and reflex route are
  sufficient.
- Negative speech evidence outranks transcript-looking text. Low-energy witness
  text remains diagnostic and has no scheduling or tool authority.
- Text-only external requests are degraded compatibility mode.
- `ask_brain` maps to a typed Hermes oracle job, not a hidden chat turn.
- `arguments.query` and `arguments.intent` are provisional route/context fields,
  not verified user text.
- Frontend witness text cannot become `oracle_text`, spend reason, provider
  selection, NemoClaw action packet, phone payload, tool argument, memory write,
  file write, external message, or durable user history without
  `interpreter_promoted` or `oracle_promoted` evidence.
- A frontend may receive status and terminal results, but not direct Hermes
  tools.

## Required Proof Fields

The headless smoke, VoiceOps readiness report, plan-run projection, and package
audit should expose:

- `external_frontend_protocol = "kame_session_v1"`
- `external_frontend_protocol_contract = "docs/kame-session-v1.md"`
- `external_frontend_tool = "ask_brain"` or
  `external_frontend_input_source = "openclaw_agent_consult"`
- shared `turn_id`, `audio_segment_ref`, `evidence_bundle_id`, and
  `evidence_merge_key`
- transcript hypotheses with `tool_authority = false`
- interpreter prompt/input order showing raw audio before witness text
- acknowledgement and oracle-job envelope creation were not blocked on ASR or
  Moshi/open-S2S transcript hypotheses when raw audio plus reflex route existed
- witness arrival phase: before cut, with cut, or after interpreter start
- witness metadata for source, confidence, latency, partial/final state,
  audio time range, speaker guess, and channel guess when available
- multi-speaker binding proof showing accepted speaker/channel matches, or a
  typed rejection reason for wrong-speaker, wrong-channel, stale, or ambiguous
  witness text
- typed `rejection_reasons[]` for every
  `rejected_or_diagnostic_only` witness hypothesis
- transcript-only degraded mode rejected for full-KAME/action gates
- terminal correlation by `tool_call_id`
- audit-id continuity from request to status and terminal event
- direct-tool authority absence
- timing-order proof for witness-before-cut, witness-with-cut, and
  witness-after-cut cases
- interpreter adjudication outcome for every active transcript hypothesis
- Moshi/open-S2S transcript context, when present, attached beside raw audio as
  a same-bundle hypothesis rather than routed as a second user turn
- explicit proof that provider STT-looking fields were normalized into
  `transcript_hypotheses[]` and not forwarded as verified `oracle_text`
- duplicate-turn and duplicate-oracle-job suppression for transcript hypotheses
  that arrive before, with, or after the accepted audio cut
- sink checks proving rejected or unpromoted witness text did not enter spend,
  provider, NemoClaw, phone, tool, memory, file, message, or durable-history
  payloads

For live evidence and package-audit artifacts, the sink proof should use this
concrete shape:

```json
{
  "unpromoted_witness_sink_checks": {
    "spend_clean": true,
    "phone_clean": true,
    "nemoclaw_clean": true,
    "tool_clean": true,
    "memory_clean": true,
    "file_clean": true,
    "message_clean": true,
    "durable_history_clean": true
  },
  "unpromoted_witness_sink_values": {}
}
```

`unpromoted_witness_sink_values` must stay empty for a passing full-KAME
artifact. If any rejected, diagnostic-only, or unpromoted Moshi/Open-S2S,
VoiceClaw/OpenClaw, reflex, or classic-ASR witness text appears in one of those
sinks, the artifact should fail the Stripe/NemoClaw/phone/tool/memory/file/
message/durable-history action gate even if the rest of the voice turn worked.
