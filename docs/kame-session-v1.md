# KAME Session v1 Contract

Status: draft contract for external realtime frontends
Owner: Hermes realtime voice / VoiceOps
Protocol id: `kame_session_v1`

This contract is the transport-neutral boundary for Discord voice,
VoiceClaw/OpenClaw-style frontends, phone/SIP, WhatsApp voice, desktop mic, and
future clients. It lets a fast frontend act as the reflex without inheriting
Hermes' tools or durable transcript authority.

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
metadata, and reflex route are the authoritative inputs; the Moshi text is an
auxiliary claim the interpreter can use, correct, or reject.

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
    "authority": "reflex_hypothesis",
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
      "text": "prepare phone handoff",
      "partial": false,
      "confidence": 0.78,
      "latency_ms": 140,
      "arrival_phase": "with_raw_audio",
      "audio_time_range_ms": [120, 1840],
      "speaker_guess": {"platform": "discord", "channel_user_id": "redacted-user"},
      "channel_guess": {"transport": "discord_voice", "channel_id": "redacted-channel"},
      "authority": "hypothesis",
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
- `s2s_transcript_hypothesis`: text proven to come from a distinct S2S caption
  or transcript side channel.
- `classic_asr_hypothesis`: text from a dedicated ASR provider retained for
  captions, diagnostics, literal checks, or degraded fallback.

Every hypothesis must carry `authority = "hypothesis"` and
`tool_authority = false`. Source names such as `moshi`, `openclaw`,
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
- Moshi/open-S2S/reflex/classic-ASR text may arrive in parallel as witness or
  fallback context, but it must not block reflex acknowledgement or oracle-job
  envelope creation when the accepted raw audio and reflex route are sufficient.
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
- transcript-only degraded mode rejected for full-KAME/action gates
- terminal correlation by `tool_call_id`
- audit-id continuity from request to status and terminal event
- direct-tool authority absence
- timing-order proof for witness-before-cut, witness-with-cut, and
  witness-after-cut cases
- interpreter adjudication outcome for every active transcript hypothesis
- Moshi/open-S2S transcript context, when present, attached beside raw audio as
  a same-bundle hypothesis rather than routed as a second user turn
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
