---
title: "goal: Make the KAME reflex and Hermes oracle truly async"
status: active
date: 2026-07-02
type: goal
target_repo: hermes-agent
successor_design: docs/design/full-kame-style-realtime-voice.md
---

# goal: Make the KAME reflex and Hermes oracle truly async

## Summary

The KAME voice branch has the right authority boundary: the human talks to a
fast reflex/interface model, a non-blocking direct-audio interpreter stage turns
raw audio plus optional transcript hypotheses into corrected multilingual
evidence, and Hermes' active model owns oracle work: tools, memory, files,
approvals, planning, and durable outcomes. This is allowed to be
three-tier-ish in implementation, because transcript side channels can arrive
before, during, or after the interpreter request. Raw audio is the primary
interpreter input; transcript hypotheses are optional context and fallback
evidence.

Current decision: implement this as three-tier sensor fan-in. The reflex owns
floor control and may emit a Moshi/open-S2S-style witness transcript. The Gemma
interpreter receives the clipped waveform, VAD/energy timing, speaker/channel
metadata, reflex route, acknowledgement already spoken, and every witness
transcript hypothesis in one evidence bundle. Hermes' active `/model` is the
oracle. There is no separate `oracle_model`, no second Hermes turn from Moshi
text, and no ASR proof requirement before acknowledgement or job creation when
raw audio is available.

The Moshi transcript is useful context, not control. It should help Gemma detect
clipped prefixes, names, numbers, code-switching, and hallucinated commands,
but it must remain `reflex_hypothesis` or `auxiliary_hypothesis` until Gemma or
the active oracle promotes it. Hypothesis-only text may support diagnostics,
captions, or clarification. It must not become `oracle_text`, durable user
history, a spend reason, call payload, memory/file content, or a tool argument
by arriving first.

The next gap is runtime behavior. The reflex and oracle are not yet truly async.
The user should be able to keep talking to the reflex while one or more oracle
jobs continue in the background. The reflex should stay conversational,
interruptible, and aware of background work state instead of becoming a modal
front door that waits for every oracle turn to finish.

Target behavior:

```text
user keeps speaking -> reflex keeps listening/responding
                  \-> oracle task 1 runs in background
                  \-> oracle task 2 runs in background
                  \-> oracle task 3 runs in background
                  \-> oracle task 4 runs in background
```

The practical first high-end capacity target is four concurrent oracle jobs for
a DGX Spark running Nemotron-3 Super. That limit should be configurable,
visible in status, and enforced by the scheduler, but this plan does not claim
the DGX Spark deployment has been validated.

## Thesis

The model the user talks to should not be the model that does the work.

The reflex is the live interface. It owns:

- listening and turn-taking
- short acknowledgements
- barge-in and spoken interruption behavior
- clarification questions
- task intake
- task status narration
- priority/cancel/update requests from the user

The interpreter is the evidence lane. It owns:

- raw clipped audio review after each speech cut
- comparison against the reflex/Moshi transcript hypotheses
- optional comparison against classic ASR hypotheses when enabled
- corrected transcript alternatives
- multilingual intent, entities, numbers, names, URLs, code terms, and language
  notes
- bounded oracle request patches that can arrive after the reflex
  acknowledgement

The interpreter does not require a separate ASR proof before it can help the
oracle. A Moshi/S2S transcript, when available, is passed as hypothesis context
inside the same evidence bundle as the raw audio. Classic ASR is retained for
fallback, diagnostics, or literal wording checks, not as the normal reflex
driver.

Moshi/S2S transcript evidence is the same kind of optional support signal. The
interpreter should not wait for it before acknowledging the user or starting
raw-audio interpretation. If it arrives late, the oracle job records it as late
evidence and may use it only before irreversible tool, spend, memory, or file
actions.

2026-07-03 amendment: model this as evidence-bundle KAME. A Moshi/open-S2S
transcript is allowed to accompany the raw audio, and should be preserved when
available, but it is never the scheduler or durable user text by itself. Gemma
is the non-blocking interpreter/evidence adjudicator, not a mandatory ASR gate.
It consumes the clipped waveform, reflex route, spoken acknowledgement,
speaker/channel/timing metadata, and any transcript hypotheses, then promotes
corrected wording only when confidence and provenance justify it.

This means the architecture is three-tier-ish, but not three independent
conversations. The reflex owns live floor control, the Gemma-style interpreter
adjudicates the cut waveform and may promote a corrected transcript candidate,
and the active Hermes model selected through existing `/model` and config flows
owns oracle action. Voice config must not add a separate `oracle_model` setting.
Moshi/S2S and classic ASR text are evidence fields inside the interpreter
bundle, not separate prompts racing the oracle and not a required precondition
for acknowledging the user.

The evidence bundle must preserve provenance:

- raw audio reference: primary evidence
- speaker and channel metadata: canonical turn context
- reflex transcript: low-latency hypothesis
- Moshi/S2S transcript: auxiliary hypothesis
- classic ASR transcript: optional fallback or comparison hypothesis
- interpreter correction: first durable transcript candidate

Each field must also carry an explicit authority label in logs and durable job
records. The minimum labels are `primary_audio`, `reflex_hypothesis`,
`auxiliary_hypothesis`, `interpreter_promoted`, `oracle_promoted`, and
`diagnostic_only`. A field can be useful before it is authoritative; it just
cannot be replayed later as if the user verified it.

The oracle may see all of those fields, but only after they are labeled. It must
never receive a Moshi or ASR transcript as if it were the user's verified
utterance.

Operationally, the interpreter request is the only place those signals merge.
The reflex may acknowledge or create an oracle job from live audio and a compact
intent before transcript evidence exists. A Moshi/S2S transcript, if the
frontend emits one, is attached to that same interpreter bundle beside the raw
audio so Gemma can compare "what the live voice model thought it heard" against
the waveform. Classic ASR follows the same path when enabled. Neither Moshi nor
ASR should create a second oracle turn, overwrite `oracle_text` directly, or
block acknowledgement. Only interpreter evidence or later oracle judgment may
promote a hypothesis into durable user text or tool-critical arguments.

The implementation should reflect this in the request shape, not only in
comments. Raw audio and timing metadata are primary interpreter inputs.
Moshi/open-S2S and classic-ASR text must live under a labeled hypotheses field,
with source, timing, confidence when available, partial/final state, and
`authority = "hypothesis"`. The interpreter prompt must be allowed to reject a
hypothesis that disagrees with the waveform, speaker identity, energy gate, or
later context.

The canonical wire target is the interpreter evidence bundle documented in
`docs/design/full-kame-style-realtime-voice.md`. For this goal, the scheduler
should treat `turn_id` plus `audio_segment_ref` as the merge key. Reflex
transcripts, Moshi/open-S2S transcript text, VoiceClaw/OpenClaw text, and
classic ASR text are all entries in `transcript_hypotheses[]` with explicit
authority labels. Vendor names such as "Moshi STT" belong in `source`, not in
the authority model.

For this goal, transcript evidence is a side-channel sensor, not a scheduler.
It can arrive before, with, or after the interpreter result. The scheduler
should preserve it with provenance, but should not wait for it when the reflex
has already routed the turn and a raw-audio interpreter request is possible.
The first available transcript is useful for latency analysis; it is not
authority.

Implementation corollary: remove "wait for ASR evidence" from the normal KAME
path. Moshi/open-S2S transcript text and classic ASR text are allowed to enrich
the interpreter request, but neither should delay acknowledgement, create a
second oracle request, or be treated as the user's durable message before
interpreter/oracle promotion.

Moshi-context corollary: if the live S2S/frontend emits a transcript-looking
field, Hermes should preserve it because it is useful evidence about what the
reflex believed it heard. That evidence must be attached to the same raw-audio
interpreter bundle, not routed as a second user turn. The interpreter receives
the waveform and the Moshi hypothesis together, then decides whether to accept,
correct, reject, or downgrade the text to diagnostic-only evidence. This lets a
fast reflex remain useful without turning Moshi into the durable transcript
authority.

Sensor-fan-in corollary: this is not the old STT-first pipeline and not a
separate parallel ASR conversation. The runtime may collect several observations
for one speech cut: live reflex route, Moshi/open-S2S transcript hypothesis,
optional classic ASR hypothesis, VAD/energy timing, and the clipped waveform.
Those observations merge by `turn_id` and `audio_segment_ref` before oracle
authority. The interpreter, not the first transcript to arrive, decides what
wording can become durable or tool-critical.

Current operating rule: collect ASR-like evidence opportunistically, but do not
require it. A transcript-looking string from Moshi, OpenClaw, VoiceClaw, or a
dedicated ASR provider is useful only as context for the direct-audio
interpreter. It must attach to the same raw-audio turn and must never create a
parallel user turn, delay the reflex acknowledgement, or patch queued oracle text
unless a trusted interpreter source or later oracle judgment promotes it.

2026-07-04 naming rule: ambiguous STT-like text from Moshi, OpenClaw,
VoiceClaw, or another open-S2S frontend should be stored as
`frontend_witness_hypothesis` when the adapter cannot prove whether it came from
the reflex model itself or from a sibling caption/ASR side channel. That name is
intentionally non-authoritative. It means "what the realtime frontend believed
it heard," and it is valid only as interpreter context beside the raw audio.
It must not create transcript-only oracle scheduling, durable user history, or
tool/payment/phone arguments.

Witness-context rule: treat Moshi/OpenClaw/VoiceClaw transcript text as what the
realtime frontend believed it heard. It should be preserved because Gemma can
use it to compare against the waveform, recover clipped starts, catch
code-switched names or numbers, and explain disagreements. It is not the user
message of record. The interpreter receives the witness text and raw audio in
one bundle and decides whether to promote, correct, reject, or downgrade it to
diagnostic-only evidence.

Three-tier rule: the desired implementation is reflex, interpreter, oracle. A
Moshi/OpenClaw/VoiceClaw transcript-looking field is not a fourth tier and not a
parallel STT conversation. It is witness evidence attached to the interpreter
bundle. The reflex keeps floor control and acknowledgement fast; Gemma compares
raw audio against witness text after the cut; Hermes' active `/model` acts only
on promoted evidence. If the raw waveform is unavailable, the turn must be
marked degraded compatibility mode and must not count as full KAME evidence.

Action-authority rule: any Stripe, NemoClaw, provider provisioning, phone-call,
memory, file, or external-message action that depends on spoken content must use
`interpreter_promoted` or `oracle_promoted` evidence for the action text and
rationale. A reflex/Moshi/OpenClaw/VoiceClaw/classic-ASR hypothesis may support
the explanation trail, but it cannot by itself become a spend reason, provider
selection, phone payload, durable user transcript, or tool argument.

Operator approval rule: the same promotion boundary must be present in user
review surfaces. A pending approval row, dashboard state packet, or GUI action
card is incomplete unless it carries promoted KAME action evidence and a
tool-disclosure reference. The UI may display Moshi/OpenClaw/VoiceClaw/reflex
or classic-ASR hypotheses for auditability, but it must not let those
hypotheses satisfy the approval contract.

Runtime approval rule: async oracle jobs must attach a compact
`voiceops.runtime_kame_action_gate.v1` result whenever a job enters
`waiting_for_approval`. The gate must fail closed when promoted interpreter or
oracle evidence is absent, when interpreter evidence was not consumed before the
approval/action boundary, or when the approval lacks `tool_disclosure_ref =
"tool_disclosure"`. Reflex, Moshi/OpenClaw/VoiceClaw, or classic-ASR hypotheses
may remain visible in the gate as witness context, but their presence must not
make the gate pass without promoted evidence.
Headless acceptance requires `runtime_kame_action_gate_enforced` to prove both
paths: hypothesis-only approvals fail closed, and consumed promoted interpreter
evidence plus `tool_disclosure_ref = "tool_disclosure"` passes.

The oracle is the worker. It owns:

- tool execution
- memory and file access
- durable planning
- project context
- approvals and external side effects
- final evidence-backed task outcomes

The oracle is not a voice-specific model selector. It is whatever Hermes's
active model/config currently selects, including local or hosted providers. A
Spark-local Nemotron endpoint is registered through the normal Hermes model
provider path and selected with `/model`; KAME voice only submits structured
oracle jobs to that active Hermes oracle.

Async scheduling is what makes that boundary real. Without it, the architecture
can still degrade into:

```text
user speaks -> reflex routes -> oracle runs -> live voice waits
```

That is better isolated than direct realtime-provider tool use, but it is not
yet the full KAME experience.

VoiceClaw and OpenClaw show a useful adjacent pattern: the realtime voice
frontend can call an `ask_brain` or `agent_consult` tool, return a quick
placeholder, and inject the real answer later. Hermes should borrow that UX
shape, but not its weaker durability model. In this goal, `ask_brain` becomes a
typed oracle job with capacity, status, cancellation, relevance checks, audit
records, and durable/non-durable transcript rules.

Open S2S frontends may also produce transcript-like text. Hermes should preserve
that output as a hypothesis from the live interface and pass it to the Gemma
interpreter beside the raw audio segment. It must not become a separate oracle
turn or the durable user transcript just because it arrived before interpreter
evidence.

When raw audio and a Moshi/open-S2S transcript are both available, the preferred
interpreter input is both signals together. The transcript tells Gemma what the
realtime frontend believed it heard; the waveform and timing metadata remain
the primary evidence. If the transcript arrives late, it is late evidence on
the same oracle job, not a reason to create or replay another Hermes turn.

When raw audio is missing, the request must be marked degraded. Text-only
VoiceClaw/OpenClaw or Moshi compatibility paths are useful bring-up paths, but
they do not prove full KAME behavior and must not lower the promoted-evidence
bar for irreversible actions.

## User Experience Goal

During a live Discord voice session:

1. The user asks Hermes to do a work item.
2. The reflex acknowledges immediately and submits an oracle job.
3. The oracle job runs in the background.
4. The user keeps talking to the reflex without waiting.
5. The user can ask "what are you working on?", "cancel the second one",
   "make the Stripe task highest priority", or "also check the logs".
6. The reflex answers from live task state and schedules/cancels/prioritizes
   without blocking the voice loop.
7. When oracle jobs finish, the reflex can summarize the result in speech and
   commit the durable outcome through Hermes session state.

## Non-Goals

- Do not give the reflex direct file, memory, MCP, shell, or payment authority.
- Do not run a second hidden Hermes agent loop outside the existing oracle path.
- Do not persist every reflex partial, acknowledgement, or local routing
  hypothesis as durable conversation history.
- Do not require all deployments to support four concurrent oracle jobs. Four
  is the DGX Spark / Nemotron-3 Super target, not a universal default.
- Do not make async voice work depend on a specific vendor realtime API.

## Required Concepts

### Oracle Job

An oracle job is a structured unit of backend Hermes work created by a reflex
route. It is not the same thing as a spoken turn. One spoken conversation can
create many oracle jobs.

For external realtime clients, an oracle job is also the compatibility target
for VoiceClaw/OpenClaw-style brain calls. A client may submit an
`ask_brain`-shaped request, but Hermes must translate it into the same
`OracleJob` shape used by Discord KAME sessions before it reaches the oracle.
No external frontend should receive direct Hermes tools as a shortcut around
the job manager.

Minimum fields:

- `job_id`
- `session_id`
- `created_at`
- `updated_at`
- `state`
- `priority`
- `route`
- `oracle_text`
- `audio_segment_ref`
- `audio_time_range_ms`
- `speaker_metadata`
- `channel_metadata`
- `reflex_intent`
- `reflex_transcript_hypothesis`
- `auxiliary_transcript_hypotheses`
- `interpreter_corrected_transcript`
- `interpreter_confidence`
- `interpreter_entities`
- `interpreter_disagreements`
- `interface_already_said`
- `requested_response_style`
- `metadata`
- `result_summary`
- `error`
- `cancel_reason`

### Job States

Initial states:

- `queued`: accepted but not yet running.
- `running`: assigned to an oracle worker.
- `waiting_for_approval`: blocked on a Hermes approval or action boundary.
- `completed`: finished with a durable result.
- `failed`: terminal error.
- `cancel_requested`: user or runtime requested cancellation.
- `cancelled`: cancellation completed; late results must be ignored.

### Capacity

The scheduler must enforce a concurrency cap.

Target config shape:

```yaml
voice:
  realtime:
    oracle_jobs:
      max_concurrent: 4
      queue_limit: 16
      default_priority: normal
      overflow_policy: queue
      shutdown_timeout_seconds: 2
```

The first intended high-end capacity target is:

```text
DGX Spark / Nemotron-3 Super: max_concurrent=4
```

Other hardware and cloud providers can use lower values. A local laptop may
default to one running oracle job. This is a target configuration, not evidence
that the Spark/Nemotron deployment has already passed capacity validation.

Approval-blocked jobs count against `max_concurrent` until the approval wait
resolves, fails, or is cancelled. This is intentional: a job waiting at an
approval boundary still owns user intent, tool context, and a pending external
effect. Letting approval-blocked jobs release capacity would allow a spoken
approval queue to grow behind the user's back and would make spend/provisioning
state harder to reason about. The reflex status view must show
  `waiting_for_approval` separately from `running` so the user can distinguish
  model compute from approval-blocked work. Status must also expose active
  capacity, because approval-blocked jobs intentionally hold scheduler slots
  even when no model tokens are currently streaming.

Queued jobs are ordered by explicit priority first and FIFO within the same
priority. Spoken reprioritization may move a queued job ahead before capacity
frees. The scheduler does not yet group or reorder by inferred user intent; that
requires a separate policy because it would let the system decide that one
spoken task is more important than another without an explicit user control.

### Reflex Status View

The reflex needs a compact, live-readable status view, not direct access to
agent internals.

Example:

```json
{
  "capacity": {"active": 2, "running": 1, "max_concurrent": 4, "queued": 1, "waiting_for_approval": 1},
  "jobs": [
    {
      "job_id": "voice-oracle-001",
      "state": "running",
      "priority": "high",
      "intent": "check deployment status",
      "spoken_status": "checking the deployment"
    }
  ]
}
```

The reflex can use this to answer status questions and decide whether to accept
new work, ask for prioritization, or suggest cancellation.

The status surface must be speakable and referential, not just machine-readable.
For the first live deployment target it should expose at least four active jobs
and any queued fifth job with stable ordinal labels, for example "job one
running", "job four running", and "job five queued". Those labels let the human
and reflex use commands like "cancel the fourth one" or "make job five high
priority" without exposing raw transcript hypotheses, speaker metadata,
interpreter evidence strings, hidden reasoning, or full oracle result text.
Additional jobs can be summarized behind a bounded "+N more" suffix.
Spoken control commands should resolve against this same reflex-safe projection
so "the fourth one" means the fourth job the user could hear or see, not a
hidden raw scheduler ordering.
When a job is waiting for approval, the reflex-safe projection should include a
compact approval reason so the user can distinguish "model is still thinking"
from "Hermes is blocked on spend/provisioning approval" without receiving raw
approval payloads, credentials, or tool arguments.

The interpreter gets a different compact view: the current turn id, clipped
audio reference, speaker/channel metadata, reflex route and acknowledgement,
optional reflex/Moshi/S2S transcript hypotheses, optional ASR hypothesis, and
active job id. Those fields should arrive as one evidence bundle for the speech
cut, not as independent prompts racing each other. The interpreter can perform
the durable multilingual transcript adjudication from this bundle, but it is not
the streaming endpointer and does not need tool schemas or broad Hermes state.

The compact interpreter view should keep raw audio first, live interface context
second, and hypotheses third. That ordering matters: it tells Gemma that Moshi,
VoiceClaw/OpenClaw, reflex, and ASR text are clues to compare against the
waveform, not replacement user messages. The job manager can use the promoted
interpreter output to patch a queued oracle job, but raw hypothesis text must
remain audit evidence unless interpreter or oracle judgment promotes it.
That promotion rule applies even when a transcript-looking field is delivered
inside an `interpreter_evidence` update: sources such as Moshi, VoiceClaw,
OpenClaw, and classic ASR can enrich the evidence bundle, but they cannot patch
`oracle_text`, durable transcript, normalized intent, or tool-critical arguments
unless a trusted interpreter source explicitly promotes the wording.

## Routing Behavior

### `local`

The reflex answers immediately. No oracle job is created.

Examples:

- greetings
- "can you hear me?"
- "repeat that"
- short clarification

### `defer`

The reflex speaks a concise acknowledgement and creates a background oracle job.
The voice loop remains live.

Example:

```text
User: "Check the Stripe provisioning logs and tell me if the VoIP account is ready."
Reflex: "I'm checking the provisioning logs."
Scheduler: creates oracle job
```

If Gemma interpreter evidence arrives before the job starts, the scheduler folds
it into the job request before execution. That fold-in should update the
oracle-facing transcript, transcript source, transcript confidence, normalized
intent, entities, disagreement metadata, and `oracle_text` when the corrected
evidence changes what Hermes is about to do. If it arrives after the job starts,
the job manager attaches it as a bounded update for tool-critical checks and
final audit. That evidence may include the raw-audio interpretation, Moshi/S2S
transcript hypotheses, and optional ASR hypotheses, but the raw audio plus
interpreter judgment remains the higher-authority evidence path.

Queued-job fold-in must keep the original hypothesis fields as evidence rather
than deleting them. The oracle request should be able to show: "the reflex
thought it heard X, Moshi/ASR suggested Y, Gemma promoted Z." That is the only
way to debug missed prefixes, truncated turns, and hallucinated command text
without bloating durable Hermes chat history.

### `oracle_direct`

The reflex creates a background oracle job immediately. It may speak a very
short acknowledgement when useful, but it must not pretend to know the answer.

Use for:

- tools
- memory
- files
- code
- deployment
- payment/provisioning
- any nontrivial factual or project-context answer

### `reject_or_clarify`

The reflex asks for missing information or refuses unsafe instructions before
creating an oracle job.

## Cancellation and Barge-In

Barge-in and cancellation are related but not identical.

Barge-in means the user interrupts current speech playback. It should stop or
mark stale the current spoken generation.

Cancellation means the user or system wants to stop backend work. It should
move one or more oracle jobs through `cancel_requested` to `cancelled` where
possible.

Rules:

- A user saying "stop talking" should stop playback, not necessarily cancel all
  running oracle jobs.
- A user saying "cancel that log check" should cancel the matching oracle job.
- A user saying "stop everything" should stop playback and request cancellation
  of active/queued oracle jobs for the current voice session.
- Late oracle output from a cancelled job must not be spoken as if current.
- Late oracle output from a cancelled job must not be persisted as a completed
  assistant result.

## Persistence Rules

Durable Hermes history should record:

- final user requests that created oracle jobs
- committed oracle job results
- user-visible cancellations
- approvals and executed tool calls
- final spoken summaries when they materially differ from raw oracle output
- interpreter corrections only when they materially affect the durable user
  request, a tool argument, an approval, or a final outcome

Durable Hermes history should not record:

- transient reflex acknowledgements unless they are intentionally visible
- partial reflex hypotheses
- cancelled drafts
- stale late oracle output
- every status poll
- reflex transcript hypotheses as raw durable user text; durable wording should
  be an interpreter/oracle-corrected transcript that may cite the hypothesis as
  supporting evidence
- Moshi/S2S transcript hypotheses as raw durable user text; durable wording
  should be an interpreter/oracle-corrected transcript that may cite the
  hypothesis as supporting evidence

The voice-session task log should still retain enough event detail to reconstruct
what happened without bloating the oracle context: normalized frontend events,
oracle job lifecycle, progress/status fragments, playback cursor updates,
barge-in/truncation events, and cross-channel handoff summaries. This is the
lesson from hosted voice stacks and VoiceClaw-style transcript sync: provider
conversation state is not the authoritative ledger, because it may include text
the user did not hear, ASR hypotheses that were later contradicted, or Moshi
transcripts that were useful only as interpreter context, while omitting
transport buffers that were cleared after barge-in.

## Proposed Implementation Shape

### 1. Add An Oracle Job Manager

Create a small scheduler owned by the realtime voice session or sidecar process.
It should not be a new model tool.

Responsibilities:

- allocate `job_id`
- accept reflex-to-oracle requests
- enforce `max_concurrent`
- queue overflow according to policy
- run Hermes oracle calls in background tasks
- expose compact status
- support cancellation
- emit lifecycle events

Candidate module names:

- `agent/realtime_voice_oracle_jobs.py`
- or a contained class inside `agent/realtime_voice_reference_sidecar.py` until
  the shape stabilizes.

Prefer a focused module once behavior crosses tests and state transitions.

### 2. Convert Reflex Escalation To Job Submission

Today `defer` / `oracle_direct` risks behaving like a handoff. It should become
job submission:

```text
reflex payload -> KameOracleRequest -> OracleJobManager.submit(...)
```

The reflex response path should return immediately after acknowledgement and
job creation.

Interpreter evidence must be late-bindable. The job manager should accept
`InterpreterEvidence` for queued jobs and fold it into the oracle request before
execution starts. For running jobs, it should attach the evidence as a bounded
update and record whether the oracle consumed it before any irreversible tool or
spend action.

### 3. Stream Oracle Job Events Back To The Reflex Session

Initial events:

- `oracle.job.accepted`
- `oracle.job.queued`
- `oracle.job.started`
- `oracle.job.interpreter_evidence_attached`
- `oracle.job.interpreter_evidence_late`
- `oracle.job.progress`
- `oracle.job.waiting_for_approval`
- `oracle.job.completed`
- `oracle.job.failed`
- `oracle.job.cancel_requested`
- `oracle.job.cancelled`
- `oracle.job.result_suppressed`

These events should be visible to `/voice status`, smoke reports, and the
reflex status view. The same event stream should be available to future
VoiceClaw/OpenClaw-compatible clients so they can show progress and receive
terminal results without scraping Discord text.

### 4. Add Reflex Commands Over Task State

The reflex needs safe internal operations, not general tools:

- list active jobs
- summarize active jobs
- request cancellation by id or matching intent
- reprioritize queued jobs
- attach a user clarification/update to an existing job

These operations modify the oracle job manager only. They do not execute
external side effects.

### 4a. Add Interpreter Evidence Updates

The interpreter needs a narrow job-manager operation:

- attach corrected transcript, entities, confidence, disagreement flags, and
  transcript-hypothesis provenance to a job
- patch queued oracle request text only from promoted interpreter evidence or
  later oracle judgment; raw Moshi/S2S/ASR hypotheses remain evidence
- mark running jobs with late evidence and expose whether it was consumed
- never execute external side effects

This operation is internal to the KAME session and does not expose Hermes tools.

### 5. Gate Speech From Completed Jobs

When a job completes:

- If the session is still live and the result is still relevant, the reflex may
  speak a concise summary.
- If the user has moved on or the result was superseded, store the result and
  summarize only if asked.
- If the job was cancelled, discard late result speech.

This requires generation/job relevance checks, not just "oracle returned text".

Default policy: `oracle_jobs.speak_terminal_results` is enabled, so completed
jobs may speak a concise terminal summary when the current playback generation
is still relevant. Operators can disable unsolicited terminal summaries with
`oracle_jobs.speak_terminal_results: false`; in that mode completed results stay
durable and visible through reflex status questions, but they are not spoken
automatically.

### 6. Normalize Frontend Brain Calls Into Oracle Jobs

Add a compatibility adapter for realtime frontends that already expose an
`ask_brain` or `agent_consult` concept.

Responsibilities:

- accept compact user intent, optional transcript evidence, frontend session id,
  and the text already spoken by the reflex
- preserve nested evidence bundles from external S2S frontends, including
  `audio_segment_ref`, `audio_time_range_ms`, `reflex_transcript_hypothesis`,
  `auxiliary_transcript_hypotheses`, and correlation ids, instead of flattening
  them into a single transcript string
- create a normal oracle job with a Hermes session id and audit id
- return an immediate accepted/queued/status placeholder to the frontend,
  including the same reflex-safe status projection used by spoken job controls
  and excluding raw transcript hypotheses, speaker/channel metadata, hidden
  reasoning, full oracle text, and tool traces
- stream job progress, failure, cancellation, and terminal result events back
  through the same KAME event stream
- reject or ignore requests for direct Hermes tool schemas from the frontend

This keeps VoiceClaw/OpenClaw interoperability aligned with Hermes authority
instead of adding a second hidden agent path.

## Acceptance Criteria

### Core Async Behavior

- A user can submit four oracle jobs in one voice session while the reflex stays
  responsive.
- The fifth job obeys configured overflow policy: queued, rejected politely, or
  requires reprioritization.
- Reflex status can report running and queued jobs without calling the oracle.
- Reflex status names at least four active jobs and the queued fifth job with
  ordinal labels that match spoken control commands.
- Reflex can accept a new local conversational turn while oracle jobs run.
- Reflex can create a new oracle job while another oracle job is running.

### Cancellation

- User can cancel one running job without cancelling unrelated jobs.
- User can cancel a queued job before it starts.
- User can stop speech playback without cancelling background jobs.
- User can say "stop everything" and request cancellation for all current
  session jobs.
- Late output from cancelled jobs is dropped from speech and durable completed
  history.
- In `fallback_policy = "fail_closed"` mode, sidecar session errors, event-stream
  failures, and send failures request bounded oracle-job shutdown so accepted
  KAME work is cancelled instead of running after the realtime frontend is gone.

### Result Handling

- Completed job result is available to the reflex status view.
- Completed job can be spoken as a concise summary when still relevant.
- Long oracle output is summarized for speech while preserving full result in
  diagnostics or durable task evidence where appropriate.
- Failed job is reported in voice without crashing the session.

### Capacity And Hardware

- Config exposes `max_concurrent`.
- Config exposes a bounded oracle-job shutdown timeout so voice/session close
  does not hang on a non-cooperative oracle worker.
- Status reports running/queued/max capacity.
- Status reports active capacity separately from running jobs when approval
  waits or cancellation waits consume slots, and exposes explicit
  `waiting_for_approval` and `cancel_requested` counts.
- Tests cover `max_concurrent=1` and `max_concurrent=4`.
- DGX Spark / Nemotron-3 Super target documents `max_concurrent=4` as the first
  intended high-end local deployment setting, pending measured validation.

### Invariants

- Reflex never receives direct tool/file/memory authority.
- Hermes active model selection remains the oracle selection path.
- Existing prompt caching and message-role invariants are not broken.
- Legacy one-shot voice and non-KAME realtime voice paths continue to work.

## Test Plan

### Unit Tests

Add tests for the oracle job manager:

- submit creates queued/running jobs with stable ids
- `max_concurrent=1` queues the second job
- `max_concurrent=4` starts four jobs and queues the fifth
- cancelling queued job prevents execution
- cancelling running job calls oracle interrupt where available
- late result from cancelled job is ignored
- failed job records error and emits failure event
- status view redacts raw tool traces and hidden reasoning
- reflex-safe status projection includes compact capacity, safe job ids, states,
  priorities, ordinal-friendly labels, and terminal summaries without raw
  evidence metadata

### Sidecar / Session Tests

Extend realtime voice tests:

- `defer` route submits background job and returns acknowledgement promptly
- interpreter evidence arriving after acknowledgement attaches to the same job
- `oracle_direct` submits job without blocking next reflex turn
- local turn during running oracle job is handled locally
- status question during running jobs uses job manager state
- status question with four running jobs and one queued job speaks job one
  through job five so follow-up commands can target them
- completed job emits a speakable summary event
- barge-in during job-result speech stops playback but does not cancel job
- explicit cancellation during job run cancels only the requested job
- playback cursor/truncation state prevents unheard generated text from being
  committed as durable assistant history
- reflex transcript hypotheses are not durable unless promoted by interpreter
  or oracle-visible outcome
- interpreter correction can update a queued job before execution starts,
  including corrections informed by Moshi/S2S transcript evidence
- late interpreter correction is recorded for a running job without creating a
  duplicate oracle job

### Discord Tests

Extend Discord realtime tests:

- two spoken tasks in one session create two oracle jobs
- `/voice status` includes oracle job capacity and state
- leaving the voice channel requests cancellation/drain for active jobs
- sidecar crash marks active jobs failed/cancelled without hanging gateway
- fail-closed sidecar send failure after oracle-job acceptance emits
  `SESSION_ERROR` and transitions the accepted job out of active capacity

### External Frontend Tests

Add KAME frontend compatibility tests:

- VoiceClaw-style `ask_brain` request creates an oracle job, not a direct tool
  call
- VoiceClaw/OpenClaw-style requests with nested `arguments` preserve raw-audio
  references, timing, reflex intent, and transcript hypotheses in the oracle job
  evidence bundle
- Moshi/S2S or classic ASR transcript strings in external frontend payloads are
  stored as hypothesis evidence and do not overwrite `oracle_text` or durable
  user transcript fields by themselves
- accepted/queued placeholder returns before the oracle completes
- completed result streams back as a terminal job event and can be summarized
  for speech
- cancellation from the frontend cancels the matching job only
- frontend transcript sync does not write status polls or placeholders into
  durable Hermes chat history
- external frontend accepted/queued placeholders and safe status responses stay
  transport/session state only; durable history starts at the normalized oracle
  job request, approval/action records, cancellation/failure, or terminal result
- normalized external frontend oracle requests preserve job id, turn id, source,
  tool/correlation id, evidence status, and audio references, while durable
  terminal job records strip raw Moshi/S2S/ASR/reflex hypothesis text unless
  interpreter/oracle judgment promoted it
- direct file, shell, memory, payment, and provisioning tools are not exposed to
  the frontend realtime model

### Evidence

Add a local smoke report mode that proves:

- four fake oracle jobs can run concurrently
- reflex still accepts a local "can you hear me?" turn while jobs run
- Gemma interpreter evidence can arrive after the reflex acknowledgement and
  before/after oracle job start without blocking the voice loop, using raw audio
  plus Moshi/S2S or ASR transcript hypotheses when available
- interpreter evidence provenance survives queued and late updates:
  raw-audio ref, time range, reflex hypothesis, auxiliary hypotheses, promoted
  transcript, confidence, entities, and disagreement flags remain distinct
- Moshi/open-S2S transcript text can be included as auxiliary context alongside
  raw voice, but tests must prove it is not promoted to `oracle_text`, durable
  transcript, spend reason, call payload, or tool argument without interpreter or
  oracle judgment
- three-tier sensor fan-in preserves exactly one evidence bundle per speech cut:
  reflex route, raw audio, Moshi/open-S2S witness text, optional ASR fallback
  text, interpreter promotion, and oracle action are distinct fields with
  explicit authority labels
- a Moshi/open-S2S transcript that arrives before the audio cut, with the audio
  cut, or after interpreter start attaches to the same `turn_id` /
  `audio_segment_ref` and never creates a second Hermes turn
- the raw-audio-plus-Moshi witness fixture exposes one stable
  `evidence_bundle_id`, one interpreter merge path, and no duplicate oracle job
  when the witness transcript arrives before, with, or after the raw audio
- degraded text-only VoiceClaw/OpenClaw/Moshi fixtures preserve hypothesis text
  for audit and clarification, but report `degraded_reason` and fail high-risk
  action gates until interpreter/oracle promotion exists
- queued oracle job updates are visible in the reflex/status event stream
- one job can be cancelled while others complete
- queued oracle jobs can be cancelled before worker execution
- late cancelled output is not played
- completed oracle results are visible in a later reflex status query
- an external KAME frontend fixture can submit a brain request, receive an
  acknowledgement, observe job status, and receive the final result without
  direct tool authority

For real DGX Spark evidence, which is required before any validated deployment
claim:

- run `max_concurrent=4` against the Nemotron-3 Super oracle endpoint
- capture latency and memory metrics
- record whether four concurrent jobs are actually viable with the configured
  context window and vLLM settings

## Policy Decisions

### Job State Persistence

Durable Hermes history records user-visible intent and outcomes:

- final user requests that create oracle jobs
- concise completed job summaries when they are user-visible
- cancellations, failures, approvals, and executed tool results that affect the
  user's durable understanding of the task

The auxiliary voice-session task log records lifecycle and scheduling detail:

- accepted, queued, started, progress, waiting-for-approval, completed, failed,
  cancel-requested, and cancelled events
- active/running/queued/waiting capacity snapshots
- status polls, progress fragments, hidden routing metadata, and diagnostic
  timing

This keeps Hermes conversation history useful without carrying every voice-loop
state transition through the oracle context. The audit/task log can be indexed
or summarized later, but it is not the same thing as assistant chat history.

### Job Updates

User updates to a job attach to the job, not to a separate durable follow-up
message by default.

- Queued-job updates are folded into the original oracle request before worker
  execution starts.
- Running-job updates are delivered through the oracle update hook when the
  active oracle supports it, and always remain visible in job status.
- Updates become durable Hermes history only when they are user-visible,
  approval-relevant, or materially change the final task outcome.

This preserves the spoken control model: "also check the receipt" modifies the
existing background job instead of creating a surprise second oracle turn.

### Default Capacity

Default `max_concurrent` is one for non-DGX and unknown local machines. Four is
the first intended DGX Spark / Nemotron-3 Super target and must be enabled by
config plus measured evidence before being claimed as production-ready.

Status and readiness artifacts must show active, running, queued, and
waiting-for-approval capacity separately so approval-blocked work cannot be
mistaken for free capacity.

## Rollout Plan

1. Implement the job manager with fake oracle workers and focused tests.
2. Wire KAME `defer` / `oracle_direct` routes to job submission.
3. Add status and cancellation events.
4. Add speech gating for completed/cancelled jobs.
5. Add interpreter evidence attachment and late-update semantics.
6. Add Discord `/voice status` job visibility.
7. Add local fake-oracle smoke evidence.
8. Add external KAME frontend compatibility tests for VoiceClaw/OpenClaw-style
   brain requests.
9. Collect measured `max_concurrent=4` evidence on DGX Spark / Nemotron-3 Super.
10. Tune defaults and update the full KAME design with measured capacity.

## Definition Of Done

This goal is complete when a live KAME voice session can keep the reflex
responsive while multiple Hermes oracle jobs run in the background, with
bounded capacity, cancellation, status reporting, result summarization, and
tests proving that cancelled or stale oracle output cannot leak into speech or
durable completed history.
