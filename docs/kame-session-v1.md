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

## Roles

External frontends may provide:

- live audio and playback control
- reflex acknowledgements and provisional routes
- transcript-looking witness text from Moshi/open-S2S, VoiceClaw/OpenClaw, or
  classic ASR side channels
- `ask_brain` / `openclaw_agent_consult` style oracle requests
- cancellation and status correlation ids

External frontends must not receive or execute direct Hermes file, shell,
memory, payment, provisioning, phone, message, or credential tools.

## Input Events

Every spoken turn should use one `turn_id`. If raw audio is available, every
witness transcript for that speech cut must share the same `audio_segment_ref`.

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
  "tool_name": "ask_brain",
  "tool_call_id": "frontend-call-001",
  "audit_id": "frontend-audit-001",
  "source_audit_id": "discord-audit-001",
  "parent_audit_id": "voiceops-root-001",
  "arguments": {
    "query": "prepare the phone handoff",
    "intent": "Prepare the phone handoff.",
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
  "transcript_hypotheses": [
    {
      "kind": "frontend_witness_hypothesis",
      "source": "moshi",
      "text": "prepare phone handoff",
      "partial": false,
      "confidence": 0.78,
      "latency_ms": 140,
      "audio_time_range_ms": [120, 1840],
      "speaker_guess": {"platform": "discord", "channel_user_id": "redacted-user"},
      "channel_guess": {"transport": "discord_voice", "channel_id": "redacted-channel"},
      "authority": "hypothesis",
      "tool_authority": false
    }
  ]
}
```

`arguments.query` and `arguments.intent` are provisional frontend route/context
fields for the job envelope. They are allowed to help create queue state or a
spoken acknowledgement, but they are not durable user text, not `oracle_text`,
and not tool/action authority. Oracle text and high-risk action text must come
from `interpreter_promoted` or `oracle_promoted` evidence.

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

The interpreter request should preserve input ordering:

1. primary raw audio reference and timing
2. speaker/channel/VAD/energy metadata
3. reflex route and acknowledgement already spoken
4. transcript hypotheses from Moshi/open-S2S, VoiceClaw/OpenClaw, reflex, or
   classic ASR

This ordering is part of the trust model. It tells the interpreter that
transcript-looking text is context for comparing against the waveform, not the
user message of record.

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

Partial hypotheses are active only until a same-source, same-kind final
hypothesis for the same speech cut arrives. The final hypothesis replaces the
partial in active interpreter context; the partial survives only as superseded
provenance for audit and latency debugging.

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

## Output Events

Hermes emits normalized session/job events back to the frontend:

- accepted placeholder: `TOOL_RESULT` with `accepted = true`, `job_id`,
  `tool_call_id`, and provider/source fields
- oracle lifecycle: `ORACLE_JOB_ACCEPTED`, `ORACLE_JOB_QUEUED`,
  `ORACLE_JOB_STARTED`, `ORACLE_JOB_WAITING_FOR_APPROVAL`,
  `ORACLE_JOB_COMPLETED`, `ORACLE_JOB_FAILED`, `ORACLE_JOB_CANCELLED`
- cancellation feedback: `INTERFACE_ORACLE_CANCEL`
- bounded updates: `INTERFACE_ORACLE_UPDATE`
- speech/playback events from the normal realtime voice stream

Terminal events must preserve `tool_call_id`, `audit_id`, `source_audit_id`,
and `parent_audit_id` so the frontend can correlate placeholders, progress,
and final results without treating witness text as durable user history.

## Authority Rules

- Raw audio is the primary interpreter evidence when `audio.segment_ref` exists.
- Transcript hypotheses are clues for the Gemma interpreter, not durable user
  messages.
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
- sink checks proving rejected or unpromoted witness text did not enter spend,
  provider, NemoClaw, phone, tool, memory, file, message, or durable-history
  payloads
