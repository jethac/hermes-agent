# Full KAME-Style Realtime Voice Design

Status: design draft
Target branch: `wip/full-kame-reflex-voice`
Target deployment: one DGX Spark, with cloud providers allowed only as bring-up fallbacks
Preferred local reflex: Moshi/PersonaPlex-class fast S2S or smaller timing model
Preferred local interpreter: Gemma 4 E2B/E4B/12B audio-multimodal
Preferred local oracle target: Hermes active `/model`, with Nemotron 3 Super as the first Spark-local NVIDIA target to validate

## Purpose

Hermes currently has KAME-compatible realtime voice plumbing: Discord voice transport, a realtime sidecar, streaming STT/TTS provider bridges, barge-in handling, mixer playback, and latency metrics. It is not yet a full KAME-style implementation because there is no lightweight, low-latency interface model acting as the human-facing conversational front end.

Full KAME-style means:

1. The human speaks to a fast interface model, also called the reflex.
2. The reflex owns the realtime conversation loop: listening, interruption,
   acknowledgements, short local replies, floor control, and rough transcript
   hypotheses.
3. A parallel interpreter lane, preferably Gemma 4 audio-multimodal, reviews
   clipped raw audio plus the reflex transcript hypothesis to produce corrected
   multilingual evidence, entities, confidence, and oracle request patches.
4. Hermes's oracle remains the brain: tools, memory, files, long reasoning,
   project context, and durable task execution.
5. The reflex and interpreter broker compact requests to the oracle instead of
   forcing every spoken fragment through the full Hermes context.

The goal is a voice system that feels immediate while preserving Hermes's existing agent capabilities.

## Model Assumptions To Validate

This design relies on the following external model and serving assumptions:

- Moshi/PersonaPlex-class S2S models can provide very low-latency floor
  control and rough transcript hypotheses, but should not be trusted as durable
  transcript truth or granted broad tools.
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
       -> interpreter lane
            -> Gemma 4 audio model over raw clip + reflex transcript hypothesis
            -> corrected transcript / entities / oracle request patch
       -> speech planner
       -> TTS or native speech output
       -> oracle router
            -> interpreter evidence
            -> optional classic ASR evidence
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
unquestioned transcript of record. The interpreter and optional ASR lanes may
correct it before the oracle executes tool calls or external actions.

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
4. The interpreter encodes the segment plus the reflex transcript hypothesis.
5. The interpreter emits corrected transcript, intent, entities, disagreement
   flags, and an oracle request patch.
6. The oracle job uses the best available request and may accept late
   interpreter evidence before irreversible tool execution.

Mid-speech backchannels require rolling windows and should be deferred until the cut-segment path is stable.

STT should not feed the reflex in normal full KAME mode. A second interpretation stream in front of the reflex creates disagreement risk between "what the reflex heard" and "what STT transcribed." The reflex's live audio path is the primary truth for floor control, but not for durable transcript or tool arguments.

The interpreter receives:

- clipped raw audio
- reflex transcript hypothesis
- optional classic ASR transcript hypothesis
- language and speaker metadata
- reflex route, acknowledgement, and "interface already said" text
- current oracle job/status context

The interpreter emits:

- corrected transcript or transcript alternatives
- normalized intent and route confidence
- entities, numbers, names, URLs, code terms, and language notes
- disagreement flags between raw audio, reflex transcript, and ASR
- oracle request patch or clarification recommendation

The interpreter may attach evidence to a queued oracle job before it starts or
send a bounded update to a running oracle job. It must not stall the reflex
acknowledgement and must not receive broad Hermes tools.

### Oracle

The oracle is whatever Hermes is configured to use. Today that may be Kimi K2.6
through Hermes. For the DGX Spark path, the first preferred local NVIDIA oracle
target is Nemotron 3 Super, selected through Hermes's normal `/model` flow after
registering the local OpenAI-compatible endpoint. Gemma 4 26B-A4B remains a
comparison candidate, not the VoiceOps-specific oracle selector. Nemotron 3
Ultra is only a hosted or future multi-Spark fallback unless local evidence
proves a one-Spark path.

It owns:

- durable user intent handling
- tools, MCP, files, memory, and project context
- long context reasoning
- high-accuracy answers
- plans and task execution
- durable transcript commits

The oracle should receive structured requests from the interface instead of raw streaming audio fragments.

### Oracle-Verbatim ASR Lane

The oracle-verbatim lane is not the old STT-first voice pipeline. It is a lower-level evidence path used only when Hermes needs exact wording.

Modes:

- `disabled`: no ASR is run; the oracle receives reflex intent and, optionally, the audio segment reference.
- `on_escalation`: ASR runs only after the reflex chooses `defer` or `oracle_direct`.
- `speculative`: ASR may start at speech end in parallel with the reflex, but its output is discarded for local turns and never drives the reflex.
- `debug`: ASR runs for comparison, captions, and diagnostics.
- `fallback`: ASR feeds the reflex only when the realtime reflex audio path is unavailable.

Default target mode: `on_escalation`.

`speculative` can be enabled if measurements show that waiting until after the reflex decision delays oracle requests. Even then, ASR is an oracle evidence lane, not a reflex dependency.

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

Optional ASR/fallback evidence events:

- `transcript.partial`
- `transcript.final`
- `reflex.transcript.hypothesis`
- `interpreter.evidence.started`
- `interpreter.evidence.final`
- `interpreter.evidence.patch`

These transcript events are disabled unless ASR mode is `on_escalation`,
`speculative`, `debug`, or `fallback`. They must not make the normal KAME path
STT-first. Reflex transcript hypotheses and interpreter evidence are separate:
the reflex hypothesis is early and non-durable by default, while interpreter
evidence is the corrected audio-understanding artifact offered to the oracle.

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
should stay ephemeral unless debugging is enabled.

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
  "oracle_text": "backend Hermes request text",
  "reflex_intent": "compact live intent",
  "reflex_transcript_hypothesis": "three to the power of seventeen",
  "interpreter_corrected_transcript": "what is three to the power of seventeen",
  "interpreter_confidence": 0.94,
  "interpreter_disagreements": ["reflex transcript omitted request prefix"],
  "interpreter_entities": [{"type": "math_expression", "value": "3^17"}],
  "interpreter_language_notes": ["English utterance with math expression"],
  "transcript": "clean final user utterance",
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
early hypothesis, the interpreter's corrected evidence when available, and any
classic ASR transcript hypothesis when enabled. The oracle should prefer
interpreter/classic-ASR literal evidence for tool arguments while preserving the
reflex route and "interface already said" context.

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
- ASR transcript spans when ASR mode is enabled
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

The end state is one DGX Spark running the complete stack:

```text
Hermes gateway
Realtime voice sidecar / KAME session manager
Fast reflex server
Gemma interpreter server
Oracle LLM server
Streaming ASR server
Streaming TTS server
Metrics/log collector
```

The oracle should stay warm. Model swapping during an interactive voice session is expected to damage latency more than it helps memory usage.

Preferred first local oracle track:

- vLLM serving Nemotron 3 Super with the best validated one-Spark settings, or
  the best validated Hermes oracle candidate selected through `/model`
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
- raw audio plus reflex transcript hypothesis as normal input
- optional classic ASR transcript hypothesis as an additional comparison input
- outputs corrected transcript, entities, language notes, confidence, and
  oracle request patches
- must be late-bindable so it can update queued/running oracle jobs without
  blocking the reflex acknowledgement

Preferred speech track:

- keep Cartesia or another cloud bridge as the baseline while local speech is being validated
- evaluate local streaming ASR and TTS separately before combining them
- do not feed STT into the reflex in normal full KAME mode
- use ASR as fallback or additional oracle-verbatim evidence for escalated
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
include_asr_hypothesis = "when_available"
timeout_ms = 2000
late_bind_to_oracle_jobs = true

[voice.realtime.asr]
mode = "on_escalation"

[voice.realtime.oracle]
mode = "hermes_active_oracle"
timeout_ms = 60000
max_spoken_sentences = 2

# The local DGX Spark oracle endpoint is registered in Hermes's normal model
# provider config and selected with `/model`, not through realtime voice.

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
- ASR mode: disabled, on_escalation, speculative, debug, or fallback
- local oracle provider target and base URL, when registering a local endpoint for the active Hermes `/model` selection
- ASR provider/model
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
- explicit transcript finalization events
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

This phase is the first point where the system can honestly be called KAME-style.

### Phase 3: Gemma Interpreter Evidence Lane

Add Gemma 4 as a parallel interpreter over the clipped raw audio plus the reflex
transcript hypothesis. It should produce corrected transcript, entities,
language notes, confidence, disagreement flags, and oracle request patches.

Deliverables:

- Gemma interpreter prompt and JSON evidence schema
- raw audio plus reflex transcript hypothesis input path
- optional classic ASR hypothesis comparison input
- corrected transcript/entity output attached to oracle jobs
- late-binding update path for queued/running oracle jobs
- tests proving the interpreter does not block reflex acknowledgement
- evidence comparing oracle outcomes with reflex-only, interpreter, and
  interpreter-plus-ASR evidence

### Phase 4: Streaming Interface Behavior

Let the interface observe VAD state, audio segment lifecycle, and playback state without committing partials. Partial transcripts are available only in debug or fallback modes.

Deliverables:

- early acknowledgement while the oracle warms or thinks
- interruption-aware response cancellation
- prevention of duplicate responses when partials change
- spoken summary of long oracle output
- configurable voice response length policy

### Phase 5: DGX Spark Local Oracle

Run Hermes's oracle through a local OpenAI-compatible server on the Spark.

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

Move the reflex, interpreter, then ASR/TTS onto the Spark as resources allow.

Deliverables:

- Moshi/PersonaPlex-class reflex launch profile
- Gemma 4 interpreter launch profile
- local reflex benchmark matrix
- local interpreter benchmark matrix, with Gemma 4 as the default candidate
- local ASR benchmark matrix
- local TTS benchmark matrix
- reflex acknowledgement latency comparison
- interpreter correction latency and quality comparison
- oracle-verbatim ASR latency and literal-accuracy comparison
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
- transcript resume contract: recent turns verbatim, older turns summarized,
  durable ledger remains authoritative
- tests proving external clients cannot receive direct Hermes file, shell,
  memory, payment, or provisioning tools
- replay fixture showing Discord and an external frontend preserve one audit id
  across the same VoiceOps task

## Test Plan

Unit tests:

- routing matrix for local/defer/oracle_direct/clarify
- reflex rough transcript hypothesis is non-durable by default
- interpreter evidence schema validation
- interpreter compares raw audio, reflex transcript hypothesis, and ASR
  hypothesis without treating any one as automatic truth
- oracle request contains distinct reflex intent, interpreter evidence, and ASR
  transcript fields
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
- escalated turn includes interpreter evidence and optional oracle-verbatim ASR
  evidence when configured
- interpreter evidence can patch a queued/running oracle job without blocking
  the immediate reflex acknowledgement
- oracle timeout with spoken status
- cancellation during oracle stream
- cancellation during TTS playback

Manual smoke tests:

- `/voice join` starts KAME mode when all dependencies are healthy
- `/voice status` reports reflex, interpreter, oracle, ASR, TTS, and fallback state
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
- `max_concurrent=4` validated against the Nemotron 3 Super endpoint, or
  explicitly marked as needing evidence
- hosted Nemotron 3 Ultra excluded from one-Spark readiness claims unless local
  evidence proves otherwise
- oracle outcome comparison with reflex-only, Gemma interpreter, and
  Gemma-plus-ASR evidence
- all-local DGX Spark smoke with oracle, reflex, interpreter, ASR, TTS, and
  sidecar together
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
- all reflex, interpreter, ASR, TTS, routing, fallback, and local-provider
  target choices are configurable from config and GUI
- Hermes oracle model selection remains the existing `/model` mechanism, not a separate realtime voice setting
- the full stack has a documented one-DGX-Spark launch path

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
- oracle-verbatim ASR lane for escalated turns
- ephemeral versus durable transcript policy at the session boundary
- oracle hint streaming back to live interface providers
- explicit KAME provenance for realtime-reflex, interpreter, and STT-fed fallback turns
- visible frontend fallback state when KAME uses local STT as the reflex fallback
- DGX Spark launch/profile generation for reflex, interpreter, oracle, ASR, and
  TTS targets
- benchmark matrix templates for local reflex, interpreter, and speech candidates
- GUI coverage for KAME reflex, interpreter, ASR, TTS, routing, barge-in,
  fallback, and local provider target settings
- oracle job manager evidence for background execution, queueing, cancellation,
  and status reporting
- DGX Spark / Nemotron 3 Super `max_concurrent=4` capacity evidence

Remaining for full KAME production readiness:

- Moshi/PersonaPlex-class or equivalent local reflex launch evidence from the
  actual target runtime
- Gemma 4 interpreter launch evidence from the actual DGX Spark runtime
- DGX Spark / Nemotron 3 Super `max_concurrent=4` capacity evidence
- benchmark evidence comparing reflex-only, Gemma interpreter, and
  Gemma-plus-ASR oracle outcomes
- benchmark evidence comparing interpreter correction against reflex transcript
  hypotheses for multilingual/code-switched turns
- all-local DGX Spark smoke evidence with the oracle, reflex, interpreter, ASR,
  and TTS services running together
- live Discord smoke evidence for the full KAME path under production credentials
