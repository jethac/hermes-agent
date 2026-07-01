# Full KAME-Style Realtime Voice Design

Status: design draft
Target branch: `wip/full-kame-reflex-voice`
Target deployment: one DGX Spark, with cloud providers allowed only as bring-up fallbacks
Preferred local reflex: Gemma 4 E2B
Preferred local oracle target: Hermes active oracle, ideally Gemma 4 26B-A4B when it is good enough for Hermes work

## Purpose

Hermes currently has KAME-compatible realtime voice plumbing: Discord voice transport, a realtime sidecar, streaming STT/TTS provider bridges, barge-in handling, mixer playback, and latency metrics. It is not yet a full KAME-style implementation because there is no lightweight, low-latency interface model acting as the human-facing conversational front end.

Full KAME-style means:

1. The human speaks to a fast interface model, also called the reflex.
2. The reflex owns the realtime conversation loop: listening, interruption, acknowledgements, short local replies, turn shaping, and escalation decisions.
3. Hermes's oracle remains the brain: tools, memory, files, long reasoning, project context, and durable task execution.
4. The reflex summarizes and brokers requests to the oracle instead of forcing every spoken fragment through the full Hermes context.

The goal is a voice system that feels immediate while preserving Hermes's existing agent capabilities.

## Model Assumptions To Validate

This design relies on the following external model and serving assumptions:

- Gemma 4 E2B supports text, image, and audio input, and produces text output.
- Gemma 4 E2B/E4B use a USM-style Conformer audio encoder.
- Gemma 4 audio input is bounded; Google's model card currently lists a 30 second audio limit.
- vLLM exposes Gemma 4 multimodal serving controls through `--limit-mm-per-prompt`, including audio prompt limits and audio memory allocation controls.
- Gemma 4 26B-A4B is the preferred first local oracle candidate on DGX Spark, but Hermes's configured oracle remains authoritative.

These assumptions must be checked against the exact model checkpoint and runtime before implementation is considered complete.

## System Shape

```text
Discord voice / desktop mic
  -> transport adapter
  -> KAME interface session
       -> streaming audio input
       -> VAD / turn detector
       -> native audio encoder
       -> reflex / interface LLM
       -> speech planner
       -> TTS or native speech output
       -> oracle router
            -> optional oracle-verbatim ASR evidence
            -> Hermes gateway / oracle session
            -> tools, MCP, memory, files, project context
       <- oracle hints, tool results, final answer
  -> mixer / playback / captions
```

The existing Discord realtime sidecar can become the first KAME interface session host. The sidecar should remain transport-neutral enough that the same session engine can later serve desktop mic/speaker, web, or another realtime transport.

## Responsibilities

### Reflex / Interface Model

The reflex is optimized for latency, turn-taking, and conversational control. It should be small enough to stay warm beside the oracle on the DGX Spark. The preferred local reflex candidate is Gemma 4 E2B because it is small, multimodal, supports native audio input on the small-model track, and has native function-calling support.

Gemma 4 E2B should be treated as an audio-understanding and routing model, not as the whole speech stack. The public Gemma 4 model descriptions describe multimodal input with text output, so E2B can let the reflex consume audio directly and produce structured intent, but TTS is still needed for spoken output.

Gemma 4 E2B's audio path should be treated as a buffered segment encoder, not as a streaming endpointer. It can ingest a cut audio segment and reason over it, but the realtime sensor remains VAD/endpointer logic. The hot path is:

1. VAD detects speech start, speech energy, and speech end.
2. The session cuts a bounded audio segment.
3. Gemma 4 E2B encodes the segment and emits intent/routing output.
4. The reflex either answers locally or escalates to the oracle.

Mid-speech backchannels require rolling windows and should be deferred until the cut-segment path is stable.

STT should not feed the reflex in normal full KAME mode. A second interpretation stream in front of the reflex creates disagreement risk between "what the reflex heard" and "what STT transcribed." The reflex's audio interpretation is the primary truth for turn routing and floor control.

Dedicated ASR still has a role below the reflex: oracle-verbatim evidence. When a turn escalates to the oracle, Hermes may need the literal words, names, numbers, code identifiers, tool arguments, or JA/EN code-switched technical speech. For those cases, a dedicated multilingual ASR result can accompany the reflex's intent as evidence for the oracle. It must not override the reflex's routing decision, and it must be labeled as a transcript hypothesis rather than treated as ground truth.

It owns:

- turn detection interpretation and speech boundary decisions
- barge-in behavior and cancellation of current speech
- immediate acknowledgements such as "one second", "got it", and "checking"
- local handling for greetings, repeats, clarification questions, and low-risk conversational glue
- short spoken style, normally one or two sentences unless the user asks for more
- compression of spoken user intent into an oracle request
- deciding whether the oracle is needed
- summarizing long oracle output into voice-appropriate responses
- ephemeral conversational state for the live voice session

It must not own:

- durable memory writes
- filesystem or project changes
- MCP/tool execution authority
- long-running task planning
- claims about capabilities that differ from Hermes's actual runtime state

Initially the interface should have no direct tool access. If direct tools are added later, they should be narrow, explicitly scoped, and auditable.

### Oracle

The oracle is whatever Hermes is configured to use. Today that may be Kimi K2.6 through Hermes. The DGX Spark target is to make the active Hermes oracle local, ideally Gemma 4 26B-A4B when it proves good enough for Hermes-style work.

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
- `fallback`: ASR feeds the reflex only when audio-native reflex serving is unavailable.

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
- `transcript.partial`
- `transcript.final`
- `barge_in.detected`
- `playback.started`
- `playback.stopped`
- `session.stop`

Interface events:

- `interface.intent.partial`
- `interface.intent.final`
- `interface.reply.local`
- `interface.reply.defer`
- `interface.oracle.request`
- `interface.oracle.cancel`
- `interface.commit`

Oracle events:

- `oracle.accepted`
- `oracle.hint`
- `oracle.tool_call`
- `oracle.tool_result`
- `oracle.response.partial`
- `oracle.response.final`
- `oracle.error`

Output events:

- `assistant.caption.partial`
- `assistant.caption.final`
- `assistant.audio.chunk`
- `assistant.audio.end`
- `session.metrics`

Only final user intents, committed assistant responses, oracle requests, and oracle results should enter durable Hermes conversation state. Partial transcripts, backchannels, cancelled utterances, and interrupted audio should stay ephemeral unless debugging is enabled.

## Routing Policy

The interface model should classify each turn into one of four paths.

`local`: The interface answers immediately without the oracle.

Use for greetings, short status checks, repeats, clarification prompts, "can you hear me", and low-risk conversational glue.

`defer`: The interface speaks a short acknowledgement and starts an oracle request.

Use for ordinary Hermes questions, tasks, code/project questions, memory-dependent questions, and anything needing current Hermes context.

`oracle_direct`: The interface does not attempt a substantive local answer and hands off to the oracle immediately.

Use for high-stakes answers, tool use, filesystem work, MCP actions, long reasoning, or anything where a local guess would create confusion.

`reject_or_clarify`: The interface asks for missing information or refuses unsafe instructions before involving the oracle.

Use when the spoken turn is incomplete, ambiguous, unsafe, or impossible to route.

The default should be conservative: local replies are allowed only when the answer does not depend on Hermes state. If unsure, acknowledge quickly and escalate.

## Oracle Request Shape

The interface should send a compact structured request:

```json
{
  "session_id": "voice-session-id",
  "turn_id": "turn-id",
  "source": "discord",
  "user_id": "discord-user-id",
  "transcript": "clean final user utterance",
  "transcript_source": "asr_or_reflex",
  "transcript_confidence": 0.92,
  "intent": "normalized user request",
  "intent_source": "reflex_audio",
  "mode": "voice",
  "urgency": "interactive",
  "interface_already_said": "One second, checking that now.",
  "conversation_summary": "ephemeral live voice summary",
  "requested_response_style": {
    "spoken": true,
    "max_sentences": 2,
    "allow_followup_offer": false
  },
  "cancellation_token": "turn-cancel-token"
}
```

This gives the oracle enough state to answer without receiving every partial audio event or every backchannel. When ASR is enabled, the request should carry both the reflex's interpreted intent and the ASR transcript hypothesis so the oracle can prefer literal wording for tool arguments while still preserving reflex routing context.

## Barge-In

Barge-in should be triggered by actual user speech, not by the first decoded packet in a new audio buffer.

Required gates:

- decoded PCM frame energy exceeds configured RMS threshold
- speech-like energy persists for a configured minimum duration
- the active speaker is not the bot
- optional VAD/speech classifier agrees when available
- playback is currently active or an oracle response is currently streaming

On barge-in:

1. stop mixer playback within the configured deadline
2. cancel in-flight TTS generation for the interrupted response
3. propagate cancellation to the oracle request if the response is no longer useful
4. keep already spoken text as ephemeral history
5. do not commit interrupted assistant text as a full assistant response

Target: playback stop within 150 ms of confirmed speech.

## Latency Budget

The design should measure every turn with monotonic timestamps. Minimum required spans:

- first decoded PCM to first partial transcript
- speech boundary to final transcript
- final transcript to interface decision
- interface decision to local first audio
- interface decision to oracle request accepted
- oracle accepted to oracle first token
- oracle first token to first TTS audio
- first TTS audio to Discord playback start
- speech boundary to first audible assistant audio
- barge-in speech confirmed to playback stopped

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
Interface LLM server
Oracle LLM server
Streaming ASR server
Streaming TTS server
Metrics/log collector
```

The oracle should stay warm. Model swapping during an interactive voice session is expected to damage latency more than it helps memory usage.

Preferred first local oracle track:

- vLLM serving Gemma 4 26B-A4B or the best validated Hermes oracle candidate
- OpenAI-compatible endpoint consumed by Hermes's existing model provider path
- fixed context and KV settings chosen for interactive work, not maximum benchmark context

Preferred interface model track:

- Gemma 4 E2B as the first reflex candidate
- direct audio input to the reflex when the serving runtime supports it
- text-only fallback through streaming STT when audio-native serving is unavailable or too slow
- other small local models only if E2B fails latency, routing, or capability-honesty tests
- the model must be good at routing, concise voice responses, and following the Hermes capability contract

Preferred speech track:

- keep Cartesia or another cloud bridge as the baseline while local speech is being validated
- evaluate local streaming ASR and TTS separately before combining them
- do not feed STT into the reflex in normal full KAME mode
- use ASR as oracle-verbatim evidence for escalated turns, not as the realtime interface
- use STT as reflex input only for text-only fallback, explicit debug/audit sessions, or provider comparisons
- do not adopt a native speech-to-speech model as the primary path until it beats the text-oracle KAME path on latency, controllability, and interruption behavior

Resource policy:

- reserve memory for the oracle first
- interface model second
- speech models third
- prefer quantized interface/speech components over evicting the oracle
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
model = "gemma-4-E2B-it"
temperature = 0.2
max_output_tokens = 160
timeout_ms = 800
audio_input = "auto"
asr_mode = "on_escalation"

[voice.realtime.oracle]
mode = "hermes_active_oracle"
timeout_ms = 60000
max_spoken_sentences = 2

# The local DGX Spark oracle endpoint is registered in Hermes's normal model
# provider config and selected with `/model`, not through realtime voice.

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

The GUI environment page should expose provider/model/base URL settings for:

- interface model
- interface audio input mode
- ASR mode: disabled, on_escalation, speculative, debug, or fallback
- local oracle provider target and base URL, when registering a local endpoint for the active Hermes `/model` selection
- ASR provider/model
- TTS provider/model/voice
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

### Phase 2: Interface LLM MVP

Insert the reflex between user turns and Hermes oracle calls. The preferred implementation is Gemma 4 E2B consuming cut audio segments directly. A text-only STT-fed path is acceptable only as a fallback while audio-native serving is unavailable or being validated.

Deliverables:

- interface prompt and JSON routing schema
- local/defer/oracle_direct/clarify routing
- short local replies through existing TTS path
- structured oracle requests to Hermes
- oracle-verbatim ASR lane for escalated turns
- tests with fake interface and fake oracle
- metrics showing which turns avoided the oracle
- evidence comparing E2B direct-audio routing against STT-fed fallback routing
- evidence comparing oracle outcomes with and without ASR transcript hypotheses

This phase is the first point where the system can honestly be called KAME-style.

### Phase 3: Streaming Interface Behavior

Let the interface observe VAD state, audio segment lifecycle, and playback state without committing partials. Partial transcripts are available only in debug or fallback modes.

Deliverables:

- early acknowledgement while the oracle warms or thinks
- interruption-aware response cancellation
- prevention of duplicate responses when partials change
- spoken summary of long oracle output
- configurable voice response length policy

### Phase 4: DGX Spark Local Oracle

Run Hermes's oracle through a local OpenAI-compatible server on the Spark.

Deliverables:

- vLLM or SGLang launch profile for the local oracle provider target used by Hermes's active `/model` selection
- warm-start and health-check scripts
- preflight that confirms model, context, and endpoint readiness
- latency comparison against current cloud oracle path
- documented memory and context settings

### Phase 5: DGX Spark Local Interface And Speech

Move the interface model, then ASR/TTS, onto the Spark.

Deliverables:

- Gemma 4 E2B reflex launch profile
- local interface model benchmark matrix, with E2B as the default candidate
- local ASR benchmark matrix
- local TTS benchmark matrix
- direct-audio reflex versus STT-fed fallback latency comparison
- oracle-verbatim ASR latency and literal-accuracy comparison
- all-local smoke test
- cloud fallback retained behind config
- one-command launch profile for the full local stack

### Phase 6: Native Realtime Provider Watch

Evaluate native speech-to-speech or live multimodal providers only as alternatives to the interface session, not as replacements for Hermes's oracle contract.

Deliverables:

- provider adapter contract
- tool/oracle integration proof
- interruption test
- capability honesty test
- measured latency comparison

## Test Plan

Unit tests:

- routing matrix for local/defer/oracle_direct/clarify
- reflex direct-audio routing parity against transcript-fed routing
- oracle request contains distinct reflex intent and ASR transcript fields
- interface JSON schema validation
- oracle request construction
- interrupted response commit behavior
- barge-in RMS and duration gates
- fallback path when interface provider is unavailable
- config scoping and defaults

Integration tests:

- fake Discord audio input to final transcript to local reply
- fake Discord audio input to oracle request to spoken response
- sidecar unavailable fallback
- TTS unavailable fallback
- audio-native reflex unavailable fallback to STT-fed routing
- escalated turn includes oracle-verbatim ASR evidence when configured
- oracle timeout with spoken status
- cancellation during oracle stream
- cancellation during TTS playback

Manual smoke tests:

- `/voice join` starts KAME mode when all dependencies are healthy
- `/voice status` reports interface, oracle, ASR, TTS, and fallback state
- greeting is answered locally
- project/tool question escalates to Hermes oracle
- user speech during bot playback stops audio quickly
- interrupted assistant response is not committed as complete
- local/cloud fallback reason is visible and accurate

Production review must also include KAME-specific evidence checks. A generic
voice production review is not enough for this branch. Required KAME gates are:

- DGX Spark benchmark evidence accepted by the generated KAME matrix validator
- Gemma 4 E2B direct-audio reflex launch evidence from the target DGX Spark runtime
- oracle outcome comparison with and without oracle-verbatim ASR transcript hypotheses
- all-local DGX Spark smoke with oracle, interface, ASR, TTS, and sidecar together
- live Discord smoke for the full KAME path under production credentials

The `kame_dgx_benchmark_evidence` production-review check must reference a
local JSON artifact from the DGX Spark benchmark validator with `ok=true` and
passing coverage for the required KAME matrix rows.

## Acceptance Criteria

The full implementation is acceptable when:

- a lightweight interface model is actually in the live path
- the oracle is not called for simple local turns
- all tool, file, memory, and project questions still go through Hermes oracle authority
- the system never tells the user it lacks voice when voice is active
- barge-in responds to real speech energy, not silent packet arrival
- local acknowledgements are consistently fast
- oracle latency is measured and visible instead of guessed
- sidecar shutdown leaves no orphan sessions or playback
- Discord fallback is explicit and understandable
- all interface, ASR, TTS, routing, fallback, and local-provider target choices are configurable from config and GUI
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
- explicit KAME provenance for native-audio and STT-fed fallback interface turns
- visible frontend fallback state when KAME uses local STT as the interface fallback
- DGX Spark launch/profile generation for interface, oracle, ASR, and TTS targets
- benchmark matrix templates for local interface and speech candidates
- GUI coverage for KAME interface, ASR, TTS, routing, barge-in, fallback, and local provider target settings

Remaining for full KAME production readiness:

- Gemma 4 E2B local reflex launch evidence from the actual DGX Spark runtime
- benchmark evidence comparing E2B direct-audio routing against STT-fed fallback routing
- benchmark evidence comparing oracle outcomes with and without ASR transcript hypotheses
- all-local DGX Spark smoke evidence with the oracle, interface, ASR, and TTS services running together
- live Discord smoke evidence for the full KAME path under production credentials
