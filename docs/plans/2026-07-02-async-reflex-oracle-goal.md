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
fast reflex/interface model, while Hermes' active oracle model owns real work:
tools, memory, files, approvals, planning, and durable outcomes.

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

The practical first capacity target is four concurrent oracle jobs for a DGX
Spark running Nemotron-3 Super. That limit should be configurable, visible in
status, and enforced by the scheduler.

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

The oracle is the worker. It owns:

- tool execution
- memory and file access
- durable planning
- project context
- approvals and external side effects
- final evidence-backed task outcomes

Async scheduling is what makes that boundary real. Without it, the architecture
can still degrade into:

```text
user speaks -> reflex routes -> oracle runs -> live voice waits
```

That is better isolated than direct realtime-provider tool use, but it is not
yet the full KAME experience.

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

Minimum fields:

- `job_id`
- `session_id`
- `created_at`
- `updated_at`
- `state`
- `priority`
- `route`
- `oracle_text`
- `reflex_intent`
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

The first deployment target is:

```text
DGX Spark / Nemotron-3 Super: max_concurrent=4
```

Other hardware and cloud providers can use lower values. A local laptop may
default to one running oracle job.

### Reflex Status View

The reflex needs a compact, live-readable status view, not direct access to
agent internals.

Example:

```json
{
  "capacity": {"running": 2, "max_concurrent": 4, "queued": 1},
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

Durable Hermes history should not record:

- transient reflex acknowledgements unless they are intentionally visible
- partial reflex hypotheses
- cancelled drafts
- stale late oracle output
- every status poll

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

### 3. Stream Oracle Job Events Back To The Reflex Session

Initial events:

- `oracle.job.accepted`
- `oracle.job.queued`
- `oracle.job.started`
- `oracle.job.progress`
- `oracle.job.waiting_for_approval`
- `oracle.job.completed`
- `oracle.job.failed`
- `oracle.job.cancel_requested`
- `oracle.job.cancelled`

These events should be visible to `/voice status`, smoke reports, and the
reflex status view.

### 4. Add Reflex Commands Over Task State

The reflex needs safe internal operations, not general tools:

- list active jobs
- summarize active jobs
- request cancellation by id or matching intent
- reprioritize queued jobs
- attach a user clarification/update to an existing job

These operations modify the oracle job manager only. They do not execute
external side effects.

### 5. Gate Speech From Completed Jobs

When a job completes:

- If the session is still live and the result is still relevant, the reflex may
  speak a concise summary.
- If the user has moved on or the result was superseded, store the result and
  summarize only if asked.
- If the job was cancelled, discard late result speech.

This requires generation/job relevance checks, not just "oracle returned text".

## Acceptance Criteria

### Core Async Behavior

- A user can submit four oracle jobs in one voice session while the reflex stays
  responsive.
- The fifth job obeys configured overflow policy: queued, rejected politely, or
  requires reprioritization.
- Reflex status can report running and queued jobs without calling the oracle.
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
- Tests cover `max_concurrent=1` and `max_concurrent=4`.
- DGX Spark / Nemotron-3 Super target documents `max_concurrent=4` as the first
  intended high-end local deployment setting.

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

### Sidecar / Session Tests

Extend realtime voice tests:

- `defer` route submits background job and returns acknowledgement promptly
- `oracle_direct` submits job without blocking next reflex turn
- local turn during running oracle job is handled locally
- status question during running jobs uses job manager state
- completed job emits a speakable summary event
- barge-in during job-result speech stops playback but does not cancel job
- explicit cancellation during job run cancels only the requested job

### Discord Tests

Extend Discord realtime tests:

- two spoken tasks in one session create two oracle jobs
- `/voice status` includes oracle job capacity and state
- leaving the voice channel requests cancellation/drain for active jobs
- sidecar crash marks active jobs failed/cancelled without hanging gateway

### Evidence

Add a local smoke report mode that proves:

- four fake oracle jobs can run concurrently
- reflex still accepts a local "can you hear me?" turn while jobs run
- queued oracle job updates are visible in the reflex/status event stream
- one job can be cancelled while others complete
- queued oracle jobs can be cancelled before worker execution
- late cancelled output is not played
- completed oracle results are visible in a later reflex status query

For real DGX Spark evidence:

- run `max_concurrent=4` against the Nemotron-3 Super oracle endpoint
- capture latency and memory metrics
- record whether four concurrent jobs are actually viable with the configured
  context window and vLLM settings

## Open Questions

- Should queued jobs be FIFO by default, priority based, or grouped by user
  intent?
- Should the reflex be allowed to auto-summarize completed background jobs, or
  should it wait for a quiet moment / user prompt?
- How much job state should be committed to Hermes history versus an auxiliary
  voice-session task log?
- Should job updates be attached to the original oracle prompt or represented
  as separate follow-up messages?
- What is the right default `max_concurrent` for non-DGX local machines?
- Should approval-blocked jobs count against `max_concurrent`?

## Rollout Plan

1. Implement the job manager with fake oracle workers and focused tests.
2. Wire KAME `defer` / `oracle_direct` routes to job submission.
3. Add status and cancellation events.
4. Add speech gating for completed/cancelled jobs.
5. Add Discord `/voice status` job visibility.
6. Add local fake-oracle smoke evidence.
7. Validate `max_concurrent=4` on DGX Spark / Nemotron-3 Super.
8. Tune defaults and update the full KAME design with measured capacity.

## Definition Of Done

This goal is complete when a live KAME voice session can keep the reflex
responsive while multiple Hermes oracle jobs run in the background, with
bounded capacity, cancellation, status reporting, result summarization, and
tests proving that cancelled or stale oracle output cannot leak into speech or
durable completed history.
