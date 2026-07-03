# Full KAME-Style Realtime Voice Design

Status: design draft
Target branch: `wip/full-kame-reflex-voice`
Target deployment: one DGX Spark as the intended local appliance path; cloud providers remain allowed only as clearly labeled bring-up fallbacks
Preferred local reflex: fastest stable floor-control model, such as Moshi/PersonaPlex-class S2S or a smaller timing/noise-gated model
Preferred local interpreter: Gemma 4 E2B/E4B/12B audio-multimodal
Preferred local oracle target: Hermes active `/model`, with Nemotron 3 Super as the first Spark-local NVIDIA target to measure before readiness claims

Current pivot: raw voice is the normal evidence path into the interpreter.
Moshi/open-S2S or classic STT text may accompany that raw voice as labeled
context, but it must not become the scheduler, the durable transcript, or a
second prompt competing with the interpreter. The runtime may look
three-tier-ish rather than perfectly staged: fast reflex, direct-audio
interpreter, optional/fallback transcript hypotheses as evidence, and Hermes's
active model as oracle.

Current clarification: Moshi/STT transcript capture is not a fourth agent lane.
It is an attachment producer for the interpreter evidence bundle. The session
may collect transcript text opportunistically, but the unit of work remains one
speech cut with raw audio as primary evidence, plus whatever labeled hypotheses
were available before or after the cut.

2026-07-03 amendment: the design is now explicitly evidence-bundle KAME. The
fast reflex may be Moshi/open-S2S-like and may produce an STT-looking text
string, but that text is not "the transcript" for Hermes. It is a hypothesis
attached to the same interpreter bundle as the clipped waveform, reflex route,
spoken acknowledgement, timing metadata, speaker metadata, and optional classic
ASR output. Gemma-style raw-audio interpretation is the promotion step for
durable wording. Hermes' active `/model` remains the only oracle and action
brain.

## Purpose

Hermes currently has KAME-compatible realtime voice plumbing: Discord voice transport, a realtime sidecar, streaming STT/TTS provider bridges, barge-in handling, mixer playback, and latency metrics. It is not yet a full KAME-style implementation because there is no lightweight, low-latency interface model acting as the human-facing conversational front end.

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
   plus any labeled Moshi/reflex/STT transcript hypotheses, then produce
   corrected multilingual evidence, transcript candidates, and oracle request
   patches. This is the layer that can behave like multilingual transcript
   adjudication for Hermes, but it is not the live endpointer and not a
   required ASR proof. Raw audio is the primary signal; transcripts are
   context, not authority.
3. **Oracle:** Hermes' active model, selected through the normal `/model`
   interface. It owns tools, memory, approvals, files, spend, provisioning,
   phone calls, and durable business logic. Voice config must not introduce a
   separate `oracle_model` setting.

This split is what makes the system KAME-style. A direct Gemma audio request can
help the interpreter, and a cloud STT/TTS bridge can prove transport behavior,
but neither is the full reflex/oracle architecture by itself.

The current design choice is deliberately not "Gemma as reflex" and not "ASR in
front of the reflex." The reflex must be the fastest reliable live-audio loop we
can run, even if it only produces a route, acknowledgement, and rough hypothesis.
Gemma is the interpreter after the cut: it receives the waveform and any
transcript-like side channels, then decides what wording is safe to offer to the
oracle. This keeps the voice loop immediate without pretending an early
transcript is ground truth.

The practical shape is therefore allowed to look "parallel" inside the evidence
bundle without becoming STT-first. A Moshi/open-S2S frontend can speak quickly
and emit a rough transcript. A classic ASR fallback can produce literal text for
captions or diagnostics. Gemma can consume both, plus the raw audio, as context.
Only the interpreter/oracle promotion result may become durable user text,
Stripe/NemoClaw spend rationale, phone-call payload, memory, file content, or
tool argument.

## Signal Authority Rules

The realtime stack may produce several text-like artifacts for one spoken turn.
They are not equivalent. The interpreter request is the merge point, and every
input must keep provenance until a later layer promotes it.

| Signal | Producer | Used For | Authority |
| --- | --- | --- | --- |
| `raw_audio` | transport/session cut | interpreter evidence, replay/debug, disagreement checks | primary interpreter input |
| `reflex_intent` | live reflex | routing, immediate acknowledgement, oracle-job creation | provisional routing |
| `reflex_transcript_hypothesis` | live reflex | early clue for Gemma/oracle, user-visible rough caption when desired | hypothesis only |
| `s2s_transcript_hypothesis` | Moshi/VoiceClaw/OpenClaw-style frontend | what the realtime voice model thought it heard | hypothesis only |
| `classic_asr_hypothesis` | dedicated ASR fallback/evidence lane | literal wording comparison, captions, diagnostics | optional hypothesis |
| `interpreter_corrected_transcript` | Gemma-style interpreter | durable user request candidate and tool-critical wording | first promoted transcript |
| `oracle_text` / final result | Hermes active `/model` | tool use, memory, files, spend, calls, durable outcome | action authority after policy checks |

This allows a three-tier design without making STT the control path. Moshi/S2S
transcripts are valuable because they summarize the live reflex's hearing of
the turn, including dropped prefixes or code-switched phrases. They must travel
beside the raw audio into the Gemma interpreter, not replace the raw audio and
not become a separate oracle prompt. Classic ASR follows the same rule and is
kept primarily for fallback, diagnostics, and literal-evidence checks.

The important distinction is not "Moshi instead of ASR" versus "classic ASR".
Both are transcript-like side channels. They can be useful evidence, especially
when the raw audio contains names, numbers, code-switched phrases, or clipped
prefixes, but neither can certify what the user said. The interpreter owns that
promotion step.

If the reflex has enough signal to acknowledge or create a background oracle
job, it should do so immediately. The interpreter can attach corrected evidence
before the job starts, or as a bounded late update before irreversible spend,
provisioning, message, memory, file, or call actions rely on the earlier text.

Moshi/VoiceClaw/OpenClaw transcript output is especially valuable because it
describes what the live interface model thought it heard at the moment it
decided how to respond. That makes it better than a generic caption for
debugging missed prefixes, code-switching, and hallucinated commands. It is
still only a hypothesis. The correct packet shape is raw audio plus
provenance-labeled hypotheses, not "pick whichever transcript arrived first."

## Interpreter Evidence Bundle Contract

Each speech cut creates one interpreter evidence bundle. This is the contract
between Discord, VoiceClaw/OpenClaw-style frontends, Moshi/open-S2S frontends,
classic ASR fallbacks, the Gemma interpreter, and the Hermes oracle job manager.
The bundle is keyed by `turn_id` and the raw-audio reference; every transcript
string is attached to that same bundle instead of becoming its own conversation.

Canonical shape:

```json
{
  "turn_id": "voice-turn-id",
  "session_id": "voice-session-id",
  "audio_segment_ref": "artifact-or-buffer-ref",
  "audio": {
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
      "authority": "reflex_hypothesis"
    },
    {
      "kind": "s2s_transcript_hypothesis",
      "source": "moshi",
      "text": "what is three to the power of seventeen",
      "partial": false,
      "time_range_ms": [12840, 15320],
      "latency_ms": 145,
      "confidence": 0.78,
      "authority": "auxiliary_hypothesis"
    }
  ],
  "interpreter": {
    "model": "gemma-4-audio",
    "status": "pending"
  },
  "oracle_job_id": "voice-oracle-001"
}
```

If an operator calls the Moshi side channel "Moshi STT", the runtime should
still store it as `s2s_transcript_hypothesis` or
`reflex_transcript_hypothesis`, depending on whether it came from the live
reflex itself or a distinct transcript output. The name can appear in `source`;
the authority stays `reflex_hypothesis` or `auxiliary_hypothesis`.

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

Lifecycle rules:

- partial transcript hypotheses attach to the pending bundle with
  `partial = true`; a final hypothesis from the same source supersedes the
  partial for interpreter context while preserving audit timing
- acknowledgement and oracle-job creation do not wait for Moshi/open-S2S or
  classic ASR hypotheses when raw audio plus reflex routing is enough
- late transcript hypotheses attach to the same bundle and can update a queued
  oracle job only through interpreter evidence
- no hypothesis text may become durable user text, `oracle_text`, a tool
  argument, a spend reason, a call/message payload, or a memory write without
  `interpreter_promoted` or later `oracle_promoted` authority
- text-only external `ask_brain` calls remain compatibility inputs; they are
  useful, but they do not satisfy the full raw-audio KAME interpreter path

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

## System Shape

```text
Discord voice / VoiceClaw / OpenClaw Talk / phone-SIP / desktop mic
  -> transport adapter
  -> KAME interface session
       -> streaming audio input
       -> VAD / turn detector
       -> fast reflex / floor-control model
            -> immediate ack / local control / rough transcript hypothesis
       -> optional auxiliary transcript hypothesis sources
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

1. `reflex_transcript_hypothesis` when it is the live reflex model's own hearing
   of the turn.
2. `s2s_transcript_hypothesis` when it is a distinct transcript or caption side
   channel from the same frontend.
3. `classic_asr_hypothesis` only for a dedicated ASR provider used for fallback,
   diagnostics, captions, or literal wording checks.

All three fields are context for the interpreter. They are allowed to improve
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

The interpreter request should make that hierarchy visible in the wire format and
prompt. Put the raw audio reference and timing fields in the primary input
section, then put Moshi/open-S2S/classic-ASR text under a separate
`transcript_hypotheses` or `auxiliary_transcript_hypotheses` field with
`authority = "hypothesis"`. The prompt must explicitly tell the interpreter
that transcript hypotheses can be wrong, clipped, hallucinated, stale, or from a
different speaker, and that it should prefer raw-audio interpretation when the
signals disagree.

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

External frontend adapters must preserve that shape instead of flattening it into
one text turn. A VoiceClaw/OpenClaw/Moshi-style bridge may send an `ask_brain`
request early, but the Hermes adapter should treat it as an interpreter/oracle
job envelope with explicit evidence fields:

- `audio_segment_ref` and `audio_time_range_ms` when the frontend can expose the
  clipped waveform or a replayable artifact
- `reflex_intent` and `interface_already_said` for what the live interface chose
  and already spoke
- `reflex_transcript_hypothesis` for the reflex model's own hearing
- `auxiliary_transcript_hypotheses[]` for Moshi/open-S2S captions, classic ASR,
  or other transcript-like side channels, each with source, timing/confidence
  when available, and `authority = "hypothesis"`
- `frontend_session_id`, `frontend_turn_id`, and `tool_call_id` for correlation,
  cancellation, and terminal result delivery

Every evidence field should also carry an authority label. The minimum labels
are `primary_audio`, `reflex_hypothesis`, `auxiliary_hypothesis`,
`interpreter_promoted`, `oracle_promoted`, and `diagnostic_only`. The scheduler
may use provisional reflex intent to queue or narrate work, but durable replay
must make clear which fields were hypotheses and which field actually drove the
oracle/tool action.

If an external frontend can provide only text and no replayable audio reference,
Hermes may still run it through the compatibility path, but that turn is degraded
evidence. It must not satisfy the full KAME raw-audio interpreter gate, and it
must not promote the transcript into durable user text without interpreter or
oracle judgment.

If a Moshi-style frontend provides both audio and transcript text, the audio
reference should be preferred even when the transcript appears cleaner. The
transcript is useful because it shows what the realtime model believed it heard;
it is not proof that the user said those words.

If the frontend can stream partial Moshi transcripts before the audio cut is
finalized, Hermes should attach them to the same pending `turn_id` and mark them
`partial = true`. A later final transcript from the same source replaces or
supersedes the partial for interpreter context, but the audit ledger should keep
the timing/provenance needed to debug clipped starts, duplicated words, and
hallucinated commands.

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
- `s2s_transcript_hypothesis`: Moshi/OpenClaw/VoiceClaw-style transcript output,
  if distinct from the reflex hypothesis
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
   hypotheses and any available auxiliary transcript evidence.
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
  "reflex_transcript_hypothesis": "three to the power of seventeen",
  "auxiliary_transcript_hypotheses": [
    {
      "source": "moshi",
      "text": "what is three to the power of seventeen",
      "confidence": 0.78,
      "latency_ms": 140,
      "authority": "hypothesis"
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

### Auxiliary Transcript Hypothesis Inputs

Auxiliary transcript inputs are not the old STT-first voice pipeline. They are
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

Modes:

- `disabled`: no separate transcript evidence is run; the oracle receives
  reflex intent and, optionally, the audio segment reference. This disables only
  the auxiliary transcript lane; it does not disable raw-audio interpretation
  when an interpreter is configured and an audio segment is available.
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
fallback, diagnostics, or literal-evidence checks.

`speculative` can be enabled if measurements show that waiting until after the
reflex decision delays oracle requests. Even then, transcript evidence remains
an interpreter/oracle hypothesis input, not a reflex dependency and not a peer
conversation path.

`speculative` is also not a request to make ASR authoritative. It exists only to
hide optional comparison latency behind the reflex decision. If speculative ASR
or a Moshi transcript disagrees with raw-audio interpretation, the disagreement
must be visible in interpreter evidence and the oracle request must prefer the
interpreter-promoted wording for tool-critical arguments.

Acceptance gates:

- a voice turn may acknowledge and create an oracle job without ASR evidence
- Moshi/S2S transcript evidence must be labeled `authority = "hypothesis"`
- interpreter prompts must include raw audio whenever an audio segment is
  available, even when Moshi or ASR produced a complete-looking transcript
- oracle jobs must distinguish `reflex_transcript_hypothesis`,
  `auxiliary_transcript_hypotheses`, and `interpreter_corrected_transcript`
- durable transcript writes and tool-critical arguments must use interpreter or
  oracle judgment, not raw Moshi/ASR text alone

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

- `transcript.partial`
- `transcript.final`
- `reflex.transcript.hypothesis`

Those names are legacy-compatible event names. In KAME mode each transcript event
must carry explicit provenance: source, authority, turn id, audio segment id when
available, timing, confidence when available, and whether the evidence arrived
before or after the interpreter/oracle job. `transcript.final` means final for
that provider stream only; it does not mean verified user text.

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
  "oracle_text": "provisional action request text; requires interpreter/oracle promotion before irreversible action",
  "reflex_intent": "compact live intent",
  "reflex_transcript_hypothesis": "three to the power of seventeen",
  "auxiliary_transcript_hypotheses": [
    {"source": "moshi", "text": "three to the power of seventeen", "confidence": 0.74}
  ],
  "interpreter_corrected_transcript": "what is three to the power of seventeen",
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
auxiliary transcript hypothesis when enabled. The oracle should prefer
interpreter evidence for tool arguments, using Moshi/S2S or classic ASR
transcripts as supporting literal evidence while preserving the reflex route and
"interface already said" context.

`oracle_text` may start from a compact reflex intent so the job can be queued
without waiting. That text remains provisional until interpreter evidence or
oracle judgment promotes it. Irreversible tool, spend, provisioning, file,
memory, message, or call actions must use the promoted transcript/intent fields
or explicitly record that the oracle accepted responsibility for acting on a
provisional request.

Queued interpreter evidence must be folded into the oracle request before the
job starts. If the interpreter produces a corrected transcript or intent while a
job is still queued, the scheduler should update `oracle_text`, `transcript`,
`transcript_source`, `transcript_confidence`, `intent`, and relevant metadata
before dispatch. Late evidence for a running job should be delivered as a
bounded update and audited before irreversible tool, spend, provisioning, or
call actions rely on the earlier request.

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

## DGX Spark Deployment Target

The intended end state is one DGX Spark running the complete stack. This section
is a target architecture and launch-plan shape, not a validated deployment
claim:

```text
Hermes gateway
Realtime voice sidecar / KAME session manager
Fast reflex server
Gemma interpreter server
Oracle LLM server
Optional auxiliary transcript hypothesis server
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
- rough transcript hypothesis captured as early, non-durable context
- text-only fallback through streaming STT only when the realtime reflex is
  unavailable or too unstable
- the model must be good at barge-in, immediate acknowledgements, concise voice
  responses, and following the Hermes capability contract

Preferred interpreter track:

- Gemma 4 E2B/E4B/12B as the first audio-understanding evidence candidate
- raw audio plus reflex/Moshi transcript hypotheses as normal input
- optional classic ASR transcript hypothesis as an additional comparison input
- outputs corrected transcript, entities, language notes, confidence, and
  oracle request patches
- must be late-bindable so it can update queued/running oracle jobs without
  blocking the reflex acknowledgement

Preferred speech track:

- use Cartesia or another cloud bridge only as a fallback or provider-comparison path while local speech is being validated
- evaluate local transcript hypothesis sources and TTS separately before combining them
- do not feed STT into the reflex in normal full KAME mode
- use ASR as fallback or additional auxiliary transcript hypothesis evidence for escalated
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
include_reflex_transcript_hypothesis = true
include_auxiliary_transcript_hypotheses = "when_available"
timeout_ms = 2000
late_bind_to_oracle_jobs = true

[voice.realtime.transcript_evidence]
mode = "from_reflex"
dedicated_asr_mode = "disabled"
sources = ["reflex", "moshi"]
# Add "asr" only for explicit fallback, diagnostics, or literal-evidence checks.
attach_to_interpreter_bundle = true
schedule_oracle_from_transcript = false
promote_without_interpreter = false

[voice.realtime.oracle]
mode = "hermes_active_oracle"
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
- no regression in Discord join, leave, playback, STT, TTS, or fallback behavior

### Phase 2: Fast Reflex MVP

Insert a fast reflex between user turns and Hermes oracle calls. The preferred
implementation is a Moshi/PersonaPlex-class S2S or smaller realtime model that
can acknowledge, control floor state, classify local/defer/oracle_direct/clarify
routes, and emit a rough transcript hypothesis. A text-only STT-fed path is
acceptable only as a fallback while the realtime reflex is unavailable or being
validated.

Deliverables:

- reflex prompt and JSON routing schema
- local/defer/oracle_direct/clarify routing
- short local replies through existing TTS path
- structured oracle requests to Hermes
- rough transcript hypothesis surfaced as non-durable evidence
- tests with fake reflex and fake oracle
- metrics showing which turns avoided the oracle

This phase is a KAME-compatible reflex/oracle MVP. It proves the interface-model
boundary and async oracle bridge, but the full three-tier KAME design is not
complete until Phase 3 adds raw-audio Gemma interpretation.

### Phase 3: Gemma Interpreter Evidence Lane

Add Gemma 4 as the non-blocking interpreter over the clipped raw audio plus the
reflex and Moshi/S2S transcript hypotheses. It should produce corrected
transcript, entities, language notes, confidence, disagreement flags, and oracle
request patches.

Deliverables:

- Gemma interpreter prompt and JSON evidence schema
- raw audio plus reflex/Moshi transcript hypothesis input path
- optional classic ASR hypothesis comparison input when enabled
- corrected transcript/entity output attached to oracle jobs
- late-binding update path for queued/running oracle jobs
- tests proving the interpreter does not block reflex acknowledgement
- evidence comparing oracle outcomes with reflex-only, interpreter, and
  interpreter-plus-auxiliary-transcript evidence

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
  results
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
- realtime reflex unavailable fallback to STT-fed routing
- escalated turn includes interpreter evidence and optional auxiliary transcript
  evidence when configured
- interpreter evidence can patch a queued/running oracle job without blocking
  the immediate reflex acknowledgement
- oracle timeout with spoken status
- cancellation during oracle stream
- cancellation during TTS playback

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
voice production review is not enough for this branch. Required KAME gates are:

- DGX Spark benchmark evidence accepted by the generated KAME matrix validator
- Moshi/PersonaPlex-class reflex or equivalent fast floor-control launch
  evidence from the target runtime
- Gemma 4 interpreter launch evidence from the target DGX Spark runtime
- evidence that Gemma interpreter corrects or confirms reflex transcript
  hypotheses from raw audio before tool-critical oracle work
- Nemotron 3 Super evaluated as the preferred Spark-local oracle target selected
  through Hermes `/model`
- `max_concurrent=4` measured against the Nemotron 3 Super endpoint, or
  explicitly marked as needing evidence
- hosted Nemotron 3 Ultra excluded from one-Spark readiness claims unless local
  evidence proves otherwise
- oracle outcome comparison with reflex-only, Gemma interpreter, and
  Gemma-plus-optional-auxiliary-transcript evidence
- all-local DGX Spark smoke with oracle, reflex, raw-audio interpreter, TTS,
  and sidecar together; auxiliary transcript evidence is optional comparison or
  fallback evidence when enabled
- live Discord smoke for the full KAME path under production credentials

The `kame_dgx_benchmark_evidence` production-review check must reference a
local JSON artifact from the DGX Spark benchmark validator with `ok=true` and
passing coverage for the required KAME matrix rows.

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
- all reflex, interpreter, auxiliary transcript hypothesis, TTS, routing, fallback, and
  local-provider target choices are configurable from config and GUI
- Hermes oracle model selection remains the existing `/model` mechanism, not a separate realtime voice setting
- the full stack has a documented one-DGX-Spark launch path, with readiness
  claims gated on measured evidence rather than the existence of the docs

## Current Gap Summary

Already present:

- Discord voice join/leave path
- realtime sidecar path
- streaming STT/TTS provider bridge path
- Cartesia and ElevenLabs-style provider configuration
- mixer playback path
- speech-energy-gated barge-in
- fallback to legacy behavior when sidecar startup fails
- focused realtime voice tests
- early latency measurement logs
- KAME interface/oracle engine in the live session path
- structured interface-to-oracle request contract
- local/defer/oracle_direct/reject_or_clarify routing policy
- auxiliary transcript-hypothesis lane for escalated turns
- ephemeral versus durable transcript policy at the session boundary
- oracle hint streaming back to live interface providers
- explicit KAME provenance for realtime-reflex, interpreter, and STT-fed fallback turns
- visible frontend fallback state when KAME uses local STT as the reflex fallback
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
- DGX Spark / Nemotron 3 Super `max_concurrent=4` capacity evidence
- benchmark evidence comparing reflex-only, Gemma interpreter, and
  Gemma-plus-optional-auxiliary-transcript oracle outcomes
- benchmark evidence comparing interpreter correction against reflex transcript
  hypotheses for multilingual/code-switched turns
- all-local DGX Spark smoke evidence with the oracle, reflex, raw-audio
  interpreter, and TTS services running together; auxiliary transcript evidence
  remains optional comparison or fallback evidence
- live Discord smoke evidence for the full KAME path under production credentials
