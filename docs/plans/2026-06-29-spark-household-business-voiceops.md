# Spark Household and Business VoiceOps Goal

## North Star

Run a household and small business from one DGX Spark, conversationally.

Hermes should become a local-first operating layer that can listen, decide, act, spend, provision, monitor, and escalate across the channels the user already lives in. The first live control surface is Discord voice. WhatsApp and phone/SMS are the next operational surfaces. The Spark is the private compute base. Stripe is the controlled spend and provisioning rail.

This is not a voice feature. Voice is the control plane. The product is trusted operational agency.

## Product Thesis

Hermes VoiceOps is a local-first operator for daily life and business:

- Household: bills, subscriptions, maintenance, calendar conflicts, errands, urgent alerts, home services.
- Business: customer ops, vendor setup, SaaS provisioning, recurring reviews, payments, reporting, and incident response.
- Voice surfaces: Discord live voice for the desk, WhatsApp for mobile chat, phone/SMS for urgent fallback.
- Compute: DGX Spark runs the KAME reflex, speech stack, and preferred local models where practical.
- Reasoning: Nemotron 3 Super is the preferred Spark-local NVIDIA oracle/model target, while Hermes `/model` remains authoritative. A clearly labeled hosted `/model` fallback is acceptable only when the local Spark path is unavailable.
- Safety: NemoClaw is the preferred sponsor-aligned execution boundary for spend, provisioning, and network-capable action packets.
- Spend: Stripe Link, Stripe Projects, and MPP/402 become the controlled path for paying, provisioning, and recording approvals.
- Audit: every planned, approved, held, executed, failed, or rolled-back action is durable and inspectable.

The hackathon entry is Milestone 0: a public proof that this shape is useful, viable, and presentable.

## Spark Model Strategy

There are two related but distinct model strategies.

### Hackathon Strategy

Use Nemotron 3 Super visibly as the preferred Spark-local Hermes oracle/model path for the demo because it is sponsor-aligned, has a credible one-Spark serving path, and communicates serious agentic reasoning on the NVIDIA target. The demo should show Nemotron 3 Super as the planner behind the budgeted VoIP provisioning workflow when the local Spark path is available.

This does not mean VoiceOps adds a separate model selector. Nemotron 3 Super should still be selected through Hermes's normal `/model` flow. If the local Super endpoint is not ready, a hosted model is acceptable only as a clearly labeled `/model` fallback. The goal is to prove the VoiceOps workflow shape: Hermes can carry a Discord voice request into a safe, budgeted, tool-using business operation with NVIDIA/Stripe integrations visible. One-Spark readiness still requires measured local Spark benchmark evidence.

Hackathon stack:

- Nemotron 3 Super as the preferred Spark-local serious reasoning/planning path
- clearly labeled hosted `/model` fallback if Super local serving is not ready
- NemoClaw as the safe execution boundary
- Stripe Skills as the spend and provisioning rail
- Discord voice as the live interface
- DGX Spark as the target local operating base and NVIDIA story

### One-Spark Local Strategy

For the long-term household/business appliance, treat Nemotron 3 Super as the first preferred Spark-local NVIDIA oracle target, but still require benchmark evidence before claiming one-Spark readiness. Nemotron 3 Ultra is not the one-Spark target; keep it as hosted or future multi-Spark context unless local evidence proves otherwise.

Target KAME layout:

- Reflex/interface: Gemma 4 E2B or E4B-style audio-native model, always warm, optimized for turn-taking and routing.
- Oracle/brain: whatever Hermes `/model` selects, with Nemotron 3 Super as the first preferred local NVIDIA candidate to evaluate on DGX Spark.
- Speech: local ASR/TTS where practical, with ASR used as oracle evidence rather than reflex input in full KAME mode.
- Fallbacks: hosted `/model` providers, Kimi, Cartesia, or other cloud providers are acceptable during bring-up and demos when they are labeled clearly.

The public demo should prefer Nemotron 3 Super on Spark for sponsor fit while allowing a clearly labeled hosted fallback only if needed. The private appliance roadmap benchmarks Super and other Spark-friendly models for the local brain.

Evidence notes:

- NVIDIA's Nemotron deployment guide lists a "Nemotron 3 Super on DGX Spark" path for a single DGX Spark with 128 GB unified memory using vLLM and TensorRT-LLM with NVFP4 and MTP.
- NVIDIA describes Nemotron 3 Super as a 120B-total, 12B-active hybrid MoE model for agentic reasoning.
- NVIDIA describes Nemotron 3 Ultra as a 550B model; for VoiceOps, Ultra is only an optional hosted/upstream fallback and must not be used as Spark-local readiness proof.
- Public DGX Spark reports support this split: Super has one-Spark reports, while Ultra reports and forum guidance point toward multi-Spark operation.

## Operating Domains

### Household Operations

Hermes should help with:

- upcoming bills, renewals, subscriptions, and household budget reviews
- maintenance reminders, vendor calls, service scheduling, and quote comparison
- calendar conflicts, travel logistics, shopping lists, and recurring tasks
- urgent alerts routed to Discord, WhatsApp, phone, or SMS depending on severity

### Business Operations

Hermes should help with:

- provisioning project services such as databases, hosting, telephony, auth, AI APIs, and observability
- customer follow-up, invoices, payment links, status reporting, and support triage
- vendor spend tracking, renewals, receipts, and approval packets
- daily or weekly operations reviews with durable action logs

### Communications

The same operator should be reachable through multiple surfaces:

- Discord realtime voice for live desktop interaction
- Discord text/status updates for approvals and audit summaries
- WhatsApp for mobile commands and low-friction follow-up
- phone/SMS for urgent escalation and non-Discord stakeholders

### Money and Provisioning

The agent should eventually operate against real economic rails:

- Stripe Projects for service provisioning and credential sync
- Stripe Link CLI for user-approved purchases
- MPP/402 for paid agent-facing services
- budget caps, approval thresholds, and spend reasons before any charge
- receipts, credential locations, and rollback notes after each action

## Architecture

### Reflex

The reflex is the lightweight KAME interface model. It is optimized for low-latency voice behavior, not deep reasoning.

Target:

- Gemma 4 E2B or E4B-style audio-native model on DGX Spark
- owns turn-taking, floor control, barge-in, short acknowledgements, intent triage, and local conversational glue
- may answer locally only for low-risk interface turns
- sends structured requests to the oracle for real work

The reflex is not the brain and should not gain broad tool authority early.

### Oracle

The oracle is Hermes's active model, selected through the existing Hermes `/model` flow. There should not be a separate `oracle_model` setting for VoiceOps. If the user points Hermes at Nemotron 3 Super, hosted Nemotron 3 Ultra, Kimi, or another provider, that is the active Hermes oracle for that run. Hosted selections do not count as Spark-local readiness evidence.

Target:

- Nemotron 3 Super as the preferred Spark-local hackathon demo and sponsor-aligned planning path, selected through `/model`
- a clearly labeled hosted fallback selected through `/model` when local Super is unavailable
- current cloud model for bring-up if needed
- local Nemotron 3 Super on DGX Spark when it proves good enough for Hermes work and beats alternatives on latency, quality, and reliability
- owns memory, tools, files, long reasoning, project context, and durable task execution

### Speech

The speech layer should support the KAME design without turning back into a simple STT-first pipeline.

Target:

- VAD/endpointer drives turn cuts
- audio-native reflex consumes user audio in full KAME mode
- dedicated ASR is an oracle-verbatim evidence lane, not the reflex driver
- local Nemotron Speech or equivalent streaming ASR for durable transcript evidence
- local Magpie/Riva-style TTS when available
- Cartesia or similar cloud TTS remains an acceptable bring-up fallback

### Skills and Tools

Hermes's existing skills and tool system remain the action layer. VoiceOps should compose those capabilities instead of bypassing them.

Initial important skills:

- Stripe Link CLI
- Stripe Projects
- cron and recurring jobs
- WhatsApp bridge
- Discord gateway
- future Twilio/phone path provisioned through Stripe Projects

### NemoClaw Execution Boundary

For the hackathon story, NemoClaw should be visible as the safe execution layer between an agent plan and real external effects.

It should wrap or present:

- Stripe Projects provisioning packets
- Stripe Link spend requests
- VoIP/phone-provider setup
- outbound message or phone-call actions
- network and credential access decisions

The video does not need to prove every NemoClaw policy in depth, but it should make clear that the agent is not receiving unchecked authority just because the user spoke a command.

### Audit Ledger

Every operational action needs a durable record:

- requested by whom
- proposed by which model or skill
- budget and approval policy used
- exact command or API action
- approval status
- result, receipt, credential location, or rollback note
- channel where the user was notified

The audit ledger is part of the product, not debug output.

## Trust Model

Hermes should be useful because it can act, but safe because action is scoped.

Allowed without approval:

- summarize status
- draft plans
- create local reminders or dry-run action queues
- collect evidence
- prepare approval packets
- post non-sensitive summaries to configured channels

Requires explicit approval:

- any payment or purchase
- service provisioning that can create a bill
- credential rotation or deletion
- vendor contact on the user's behalf
- external messages to customers, vendors, or household members

Forbidden without a separate future design:

- hidden spend
- raw card data in model context
- unbounded recurring charges
- irreversible deletion
- broad autonomous tool execution from the reflex
- pretending an action happened when it only reached dry-run

Default mode is dry-run. Live spend should go through Stripe/Link approval or an equivalent user-visible authorization.

## Milestone 0: Hackathon Proof

Goal for the June 30, 2026 submission: show a serious local agent operating system, not a toy voice demo.

Demo request:

```text
Hermes, I am giving you 200 dollars to use through Stripe Skills. Provision yourself a VoIP provider account, then call my phone with this same context so we can continue outside Discord.
```

Required proof:

- Discord voice is the live front door.
- Hermes gives an immediate KAME-style acknowledgement.
- Nemotron 3 Super is visible as the preferred Spark-local serious planning/oracle path for the demo, with any hosted fallback labeled clearly if used.
- NemoClaw is visible as the safe execution boundary before billable or network-capable actions.
- Hermes converts the spoken budget into a spend policy.
- Stripe Projects action is queued to provision a VoIP-capable provider account, such as Twilio voice.
- Stripe Link action is queued for a gated service-credit spend request.
- Hermes preserves the Discord context for the phone handoff.
- Hermes queues or performs an outbound call to the user's phone with the same context.
- The audit ledger shows every action and approval requirement.
- WhatsApp and phone/SMS appear as reachable follow-on surfaces.
- The DGX Spark target is explicit in the story and artifacts.

Video spine:

1. User joins Discord voice.
2. User gives Hermes a fixed amount of spending money through Stripe Skills.
3. Hermes acknowledges the budget and explains that live spend requires approval.
4. Hermes uses Nemotron 3 Super for the plan, or clearly labels a hosted `/model` fallback.
5. Hermes presents a NemoClaw-safe action packet.
6. Hermes queues Stripe Projects to provision VoIP.
7. Hermes queues a Link-gated spend request for service credit.
8. Hermes calls the user's phone and continues from the same Discord context.

Headless command:

```bash
uv run python scripts/hackathon_voiceops_demo.py --output-dir artifacts/hackathon-voiceops-demo/current
```

The command writes:

- `voiceops-demo.json`
- `voiceops-demo.md`
- `audit-ledger.jsonl`
- `demo-script.md`
- `nemoclaw-action-packet.json`
- `nemoclaw-action-packet.validation.json`
- `phone-context.json`
- `milestone2-execution-plan.json`
- `readiness-report.json`
- `readiness-report.md`
- `readiness-closure-summary.json`
- `readiness-closure-summary.md`
- `operator-handoff-preview.json`
- `operator-handoff-preview.md`
- `operator-dashboard.html`
- `operator-state.json`
- `operator-state-events.jsonl`
- `recording-runbook.md`
- `submission-writeup.md`
- `stripe-actions-dry-run.sh`

The generated shell script is dry-run by construction. It prints the Stripe/Projects commands instead of executing them. The NemoClaw validation artifact is also local static validation only: it checks the packet schema, approval contracts, command hashes, dry-run command alignment, blocked capabilities, and no-write safety flags without running commands, calling the network, spending, provisioning, reading credentials, or placing calls. The demo package, readiness report, local closure summary, and operator handoff preview are schema-tagged so the Milestone 0 artifact directory can be reviewed without first opening the global plan-run index. The readiness report is non-invasive: it checks local prerequisites and env shape from process env, repo `.env`, and explicit `--env-file` values by presence only. It does not read `/Users/jethac/.hermes/hermes-agent`, print secrets, provision, purchase, call, or mutate credentials. The handoff preview is also non-invasive: it lists the ordered safe evidence-collection phases, current blockers, command safety labels, must-not rules, and final reindex command for live Discord voice, spend/provisioning preflight, and DGX Spark evidence. The HTML dashboard is a static recording surface and does not require a web server.
The recording runbook gives the shot list, fallback recording path, submission checklist, and tweet draft without requiring live spend or live provisioning. The submission writeup gives concise public copy for the tweet/thread/form.

## Milestone 1: Real Voice Operator

Make `/voice join` usable as the daily control surface:

- stable Discord receive/playback lifecycle
- real barge-in based on speech energy, not silent packet arrival
- KAME fallback state visible when audio-native reflex is unavailable
- voice replies short by default
- latency metrics from user speech end to reflex response, oracle response, and TTS playback
- voice capability prompt context so Hermes does not claim it cannot hear or speak

Headless command:

```bash
uv run python scripts/voiceops_voice_operator.py --output-dir artifacts/voiceops-voice-operator/current
```

The command writes:

- `voice-operator-readiness.json`
- `voice-operator-readiness.md`
- `discord-loopback-smoke.json`
- `voice-operator-events.jsonl`
- `live-voice-evidence-template.json`
- `live-voice-evidence.example.json`
- `live-voice-evidence-scaffold/manifest.json`
- `live-probe-closure-plan.json`
- `live-probe-closure-plan.md`

The voice-operator artifact runs the in-memory Discord realtime voice loopback smoke. It verifies lifecycle, receiver callback wiring, PCM conversion, mixer playback, barge-in signaling, latency metrics, and sidecar shutdown without Discord network access, provider sidecar network access, credential reads, sends, or calls. It must still report that a live Discord `/voice join` probe is required before claiming production readiness.

If an existing `hermes doctor --realtime-voice-report` JSON file exists, derive sidecar and live-turn VoiceOps evidence from it without running Discord probes:

```bash
uv run --extra dev --extra voice hermes doctor \
  --realtime-voice \
  --realtime-voice-smoke \
  --discord-voice-live-probe \
  --discord-voice-live-probe-require-inbound \
  --discord-voice-live-probe-wait-seconds 5 \
  --realtime-voice-report artifacts/realtime-voice-evidence/live-current/realtime-voice-doctor-report.json
```

```bash
uv run python -m hermes_cli.realtime_voice_live_evidence \
  --output-dir artifacts/realtime-voice-evidence/live-current \
  --from-realtime-voice-report artifacts/realtime-voice-evidence/live-current/realtime-voice-doctor-report.json
```

This writes `sidecar-session.from-realtime-report.json`, `live-turn.from-realtime-report.json`, and `realtime-voice-report-validation.json`, and references any `discord_live_probe` section only if the doctor report actually contains one. The derivation uses the alpha report validator before emitting passing evidence, copies the reported sidecar mode instead of translating loopback or diagnostic modes into production, omits raw transcripts and assistant text from generated live-turn evidence, and still lets strict validation report missing Discord join/playback/receiver gates when the doctor report lacks a real Discord probe.

When live evidence exists, ingest supplied artifacts without running Discord from the generator:

```bash
uv run python -m hermes_cli.realtime_voice_live_evidence \
  --output-dir artifacts/realtime-voice-evidence/live-current \
  --require-live-discord \
  --require-inbound \
  --wait-seconds 5 \
  --sidecar-session-evidence artifacts/realtime-voice-evidence/live-current/sidecar-session.json \
  --live-turn-evidence artifacts/realtime-voice-evidence/live-current/live-turn.json
```

Before indexing the collected bundle, validate it offline with the same strict live-evidence contract used by the voice-operator ingester:

```bash
uv run python -m hermes_cli.realtime_voice_live_evidence \
  --audit-only \
  --discord-live-probe-evidence artifacts/realtime-voice-evidence/live-current/discord-live-probe.json \
  --sidecar-session-evidence artifacts/realtime-voice-evidence/live-current/sidecar-session.json \
  --live-turn-evidence artifacts/realtime-voice-evidence/live-current/live-turn.json
```

```bash
uv run python -m hermes_cli.realtime_voice_live_evidence \
  --output-dir artifacts/realtime-voice-evidence/live-current \
  --validate-live-evidence \
  --discord-live-probe-evidence artifacts/realtime-voice-evidence/live-current/discord-live-probe.json \
  --sidecar-session-evidence artifacts/realtime-voice-evidence/live-current/sidecar-session.json \
  --live-turn-evidence artifacts/realtime-voice-evidence/live-current/live-turn.json
```

```bash
uv run python scripts/voiceops_voice_operator.py \
  --output-dir artifacts/voiceops-voice-operator/current \
  --live-evidence artifacts/realtime-voice-evidence/live-current/manifest.json
```

The supplied evidence path is read-only. It may be a `voiceops.realtime_voice_live_evidence_manifest.v1` `manifest.json` from `hermes_cli.realtime_voice_live_evidence`, a combined `voiceops.milestone1.live_voice_evidence.v1` evidence JSON file with nested live evidence sections, or individual evidence JSON files. It must prove Discord join/playback, inbound receiver frames or speech-start, Discord probe latencies, production sidecar session start/close, sidecar `sidecar_mode: production`, redacted session identity, observed healthcheck and provider transport, visible fallback reason, and one live conversational turn with transcript, assistant audio, barge-in, short spoken reply, no voice-capability denial, first-audio latency, and barge-in stop latency. Each `discord_live_probe`, `sidecar_session`, and `live_turn` section must carry both a `source_artifact` that resolves as an absolute path or beside the supplied evidence file and a non-placeholder `collector_attestation` with collector name/version, run id, command argv, git commit, timestamp window, raw/redacted SHA-256 hashes, and parent manifest hash. For non-manifest ingestion, pass one `--live-evidence` per section or combined file; each standalone section file must include `kind` or `evidence_type`, a verifiable `source_artifact`, and the collector attestation. Per-section manifest reports must also identify themselves with `kind` or `evidence_type` values such as `discord_live_probe`, `sidecar_session`, or `live_turn`, unless the report is a combined expanded `voiceops.milestone1.live_voice_evidence.v1` object with nested sections. Manifest ingestion is preferred because manifest reports record the actual referenced report path as provenance. Manifest `reports` may reference per-section artifacts or one combined evidence artifact; placeholder source paths inside referenced artifacts are not trusted as provenance because the ingester records the actual manifest-resolved file path. Template source artifact names such as `discord-live-probe.json`, `voice-status-or-sidecar-report.json`, `sidecar-session.json`, `voice-turn-evidence.json`, and `live-turn.json` are rejected until replaced by real resolved artifact paths, and `example_only` or placeholder collector attestations are rejected. It must not include Discord tokens, provider secrets, full phone numbers, or private transcript text containing secrets. `--audit-only` performs no Discord network call, no report derivation, and no persistent artifact writes under `--output-dir`; it prints schema `voiceops.realtime_voice_live_evidence_audit.v1` and returns nonzero until strict validation passes. `--validate-live-evidence` performs no Discord network call and writes `live-evidence-validation.json` with schema `voiceops.realtime_voice_live_evidence_validation.v1`; it exists to produce a durable validation artifact before `scripts/voiceops_voice_operator.py` updates the readiness artifacts. `--from-realtime-voice-report` writes derivation metadata with schema `voiceops.realtime_voice_report_derivation.v1` and must not claim production sidecar evidence from loopback or diagnostic sidecar modes. The live-evidence collector references sidecar and live-turn files in its manifest and runs strict validation whenever optional or derived evidence is supplied; it does not sanitize supplied files or embed their contents. The generated `.example.json` file is only a populated redacted shape for operators; validators reject `example_only: true` evidence until real artifact references replace it. The generated `live-voice-evidence-scaffold/` directory is the preferred starting point for manual live evidence: replace its Discord, sidecar, and live-turn section files with real redacted observations and remove every `example_only` marker before ingesting `live-voice-evidence-scaffold/manifest.json`.

## Milestone 2: Real Spend and Provisioning

Turn the dry-run queue into controlled live operations:

- run the non-mutating provisioning preflight before any live spend or provider action
- verify Stripe Link CLI auth and approval flow
- verify Stripe Projects plugin and catalog on the target machine
- run `stripe projects list` and safe catalog discovery headlessly
- execute one low-risk live VoIP provisioning path only after explicit approval
- queue or perform one outbound phone call with a preserved context packet
- record receipts and generated credential locations without exposing secrets
- add rollback/deprovision notes to the ledger

Preflight command:

```bash
uv run python scripts/voiceops_provisioning_probe.py --output-dir artifacts/voiceops-provisioning/current
```

The command writes:

- `provisioning-readiness.json`
- `provisioning-readiness.md`
- `safe-command-manifest.json`
- `read-only-discovery.json`
- `read-only-discovery.md`
- `read-only-discovery.manifest.json`
- `audit-ledger.read-only-discovery.jsonl`
- `milestone2-execution-plan.json`
- `milestone2-execution-plan.md`
- `provisioning-preflight-evidence.template.json`
- `provisioning-preflight-evidence.example.json`
- `provisioning-preflight-evidence.manifest.example.json`
- `provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json`
- `post-approval-receipts.template.json`
- `post-approval-receipts.example.json`
- `post-approval-receipts.validation.json`
- `nemoclaw-action-packet.validation.json`
- `post-approval-receipts-scaffold/post-approval-receipts.json`
- `audit-ledger.post-approval.jsonl`
- `setup-closure-plan.json`
- `setup-closure-plan.md`

The default preflight is non-mutating and only checks PATH/env presence, env-key presence, local optional Stripe Skills bundle contracts, command policy, and phone-handoff configuration shape. It writes a `not_supplied` NemoClaw action-packet validation artifact unless `--nemoclaw-action-packet` is provided. It verifies that `optional-skills/payments/stripe-projects`, `optional-skills/payments/stripe-link-cli`, and `optional-skills/payments/mpp-agent` exist with the expected Link approval, HTTP 402/SPT, Projects billing, `.env`, and wallet/key secrecy safety terms. It blocks live spend, provider provisioning, credential retrieval, outbound phone calls, account mutation, and network tunnels. If active command probing is needed, it must be explicitly enabled with `--run-command-probes`; that mode is still limited to isolated version/help subprocess probes and must not be treated as approval for `stripe projects add`, Link spend creation, card retrieval, MPP payment, SMS, or phone calls.

To validate the generated NemoClaw packet without running provider commands, ingest it explicitly:

```bash
uv run python scripts/voiceops_provisioning_probe.py \
  --output-dir artifacts/voiceops-provisioning/current \
  --env-file .env \
  --no-command-probes \
  --nemoclaw-action-packet artifacts/hackathon-voiceops-demo/current/nemoclaw-action-packet.json
```

Display-only discovery is a separate opt-in path and is required before Milestone 2 can be considered ready for live provisioning approval. Use `--run-readonly-discovery` only for the exact allowlisted commands `stripe projects list --limit 10` and `link-cli auth status`. These commands run with an isolated temporary `HOME`, so `link-cli auth status` is an isolated auth-status attempt and does not prove the operator's normal local CLI auth state. The probe redacts command output, writes `read-only-discovery.json`, `read-only-discovery.md`, `read-only-discovery.manifest.json`, and `audit-ledger.read-only-discovery.jsonl`, and still does not grant approval for spend, provisioning, credential retrieval, messages, or calls.

```bash
uv run python scripts/voiceops_provisioning_probe.py \
  --output-dir artifacts/voiceops-provisioning/current \
  --env-file .env \
  --run-readonly-discovery
```

After a read-only discovery run exists, later closure/index runs should ingest the redacted manifest instead of rerunning network-capable discovery:

```bash
uv run python scripts/voiceops_provisioning_probe.py \
  --output-dir artifacts/voiceops-provisioning/current \
  --env-file .env \
  --read-only-discovery-evidence artifacts/voiceops-provisioning/current/read-only-discovery.manifest.json
```

When local setup and account/capability evidence exists, ingest supplied evidence without running live spend or provider mutations:

```bash
uv run python scripts/voiceops_provisioning_probe.py \
  --output-dir artifacts/voiceops-provisioning/current \
  --env-file .env \
  --preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-evidence.json
```

Or, when evidence is split by provider/domain:

```bash
uv run python scripts/voiceops_provisioning_probe.py \
  --output-dir artifacts/voiceops-provisioning/current \
  --env-file .env \
  --preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json
```

The supplied evidence path is read-only. It may be one complete `voiceops.milestone2.preflight_evidence.v1` JSON file or a `voiceops.milestone2.preflight_evidence_manifest.v1` manifest that references separate redacted section files for Stripe Projects, Stripe Link, MPP/NemoClaw, phone handoff, and rollback ownership. It must contain account aliases, capability booleans, provider references, credential-location references, rollback owners, and a `source_artifact` for every redacted evidence section. Every section must also include `source_artifact_kind: redacted_setup_evidence`, `source_artifact_sha256`, `source_artifact_redacted_at`, and `collector_attestation`; the SHA-256 must match the referenced redacted JSON source artifact, the attestation redacted hash must match that SHA-256, and the redaction and collection timestamps must be parseable with timezone information. Source artifacts must exist, be UTF-8 JSON, be marked redacted or carry a redaction policy, and resolve as absolute paths or paths relative to the supplied evidence/manifest file; the validator must not fall back to unrelated files in the process working directory. Collector attestations must identify the collector name/version, run id, command argv, git commit, timestamp window, raw/redacted SHA-256 hashes, and parent manifest hash; placeholder or `example_only` attestations are rejected. It must not contain Stripe secrets, provider tokens, raw card data, full phone numbers, or proof of unapproved live spend. The generated `.example.json` and `.manifest.example.json` files show redacted completed shapes for headless setup, but they are rejected as proof while `example_only: true` remains present.
The generated `provisioning-preflight-scaffold/` directory is the preferred operator starting point for split evidence: replace each section report and redacted source artifact with real local setup proof, refresh the SHA-256 fields, and remove every `example_only` marker before ingesting the manifest.

```bash
uv run python scripts/voiceops_provisioning_probe.py \
  --refresh-preflight-source-hashes artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json
```

The refresh helper reads and writes only local JSON evidence files. It does not inspect env files, run command probes, perform network I/O, provision providers, spend money, retrieve credentials, send messages, or place calls.

The Milestone 2 execution plan is also non-mutating. It is the post-approval contract for the first live provisioning flow: readiness gates, display-only discovery commands, approval-required Stripe/Link/phone actions, receipt schema, credential-location schema, rollback/deprovision notes, and phone-context linkage. It must never claim that spend, provisioning, credential retrieval, outbound messages, or phone calls have already executed.

When approved actions have real redacted post-approval evidence, ingest the receipt bundle with `--post-approval-receipts` without running provider commands:

```bash
uv run python scripts/voiceops_provisioning_probe.py \
  --output-dir artifacts/voiceops-provisioning/current \
  --env-file .env \
  --post-approval-receipts artifacts/voiceops-provisioning/current/post-approval-receipts.json
```

The receipt bundle uses `voiceops.milestone2.post_approval_receipts.v1` and must contain redacted `receipts`, `credential_locations`, `rollback_receipts`, and `audit_events`. The validator rejects `example_only`, raw secret/token/card/phone fields, command hash mismatches, unknown action ids, duplicate credential-location refs, duplicate rollback refs, duplicate audit event ids, missing audit events, and missing credential-location or rollback refs for executed actions. Held, denied, and skipped decision receipts may omit execution-only credential and rollback artifacts. A valid bundle writes `post-approval-receipts.validation.json` and `audit-ledger.post-approval.jsonl`; it still does not execute spend, provisioning, credential retrieval, messages, or calls.

## Milestone 3: Multi-Channel Operations

Make the same operator reachable beyond Discord:

- generate and review the multi-channel policy artifact before enabling new egress surfaces
- WhatsApp Cloud setup path validated for command and approval messages
- phone/SMS path designed around Twilio or equivalent provisioning
- channel-specific authorization rules
- escalation policy for urgent household/business events
- consistent audit IDs across Discord, WhatsApp, and phone/SMS

Policy command:

```bash
uv run python scripts/voiceops_channel_policy.py --output-dir artifacts/voiceops-channel-policy/current
```

The command writes:

- `channel-policy.json`
- `channel-policy.md`
- `channel-policy-review.json`
- `channel-policy-review.md`

The policy artifacts are static and headless. They read no secrets, perform no network I/O, send no Discord/WhatsApp/SMS messages, and place no calls. They define channel authorization, approval routing, escalation levels, audit ID continuity, redaction rules, and a pending human review packet for Discord, WhatsApp, and phone/SMS before those surfaces are used for real operations. The review packet does not enable egress; it records the signoffs and gates that must be satisfied before a separate runtime approval can do that.

## Milestone 4: Local Spark Stack

Move as much of the stack as possible onto one DGX Spark:

- local reflex model launch evidence
- local Hermes oracle endpoint registered through normal `/model` selection
- Nemotron 3 Super evaluated as the preferred local NVIDIA brain
- local ASR/TTS bridge evidence
- all-local smoke with oracle, interface, ASR, TTS, and sidecar together
- benchmark evidence accepted by the generated DGX Spark matrix validator

Headless command:

```bash
uv run python scripts/voiceops_spark_matrix.py --output-dir artifacts/voiceops-spark-matrix/current
```

The command writes:

- `spark-model-matrix.json`
- `spark-model-matrix.md`
- `spark-benchmark-evidence-template.json`
- `spark-benchmark-evidence.example.json`
- `spark-benchmark-scaffold/spark-benchmark-evidence.json`
- `spark-matrix-closure-plan.json`
- `spark-matrix-closure-plan.md`
- `spark-operator-runbook.md`

When benchmark evidence exists, pass it with repeated `--evidence path/to/evidence.json` arguments. Use `--lint-evidence --evidence path/to/evidence.json` first when the operator wants a no-write packaging/readiness check; lint mode prints schema `voiceops.spark_evidence_lint.v1`, performs no artifact writes, performs no network I/O, and returns nonzero until the supplied evidence proves one-Spark readiness. The matrix accepts its native `voiceops.spark_benchmark_evidence.v1` records and adapts the generated KAME DGX Spark benchmark evidence shape when provenance is present. Local readiness requires more than role metrics: evidence must identify the hardware/locality, model, measurement time, source artifact, source artifact SHA-256, collector attestation, verification state, and an all-local stack smoke proving reflex, oracle, ASR, TTS, and sidecar ran together on one DGX Spark. The referenced redacted raw source artifact must exist beside the supplied evidence file or at an absolute path, must be readable UTF-8 JSON, must be marked redacted or carry a redaction policy, must not carry `example_only: true`, and its SHA-256 must match both `source_artifact_sha256` and `collector_attestation.redacted_artifact_sha256`. Collector attestations must identify the collector name/version, run id, command argv, git commit, timestamp window, raw/redacted SHA-256 hashes, and parent manifest hash; placeholder or `example_only` attestations are rejected. The stack smoke must also prove KAME routing: the oracle is selected by Hermes `/model`, oracle authority routes include tools/files/memory/project context, the interface input source includes `native_audio`, and the reflex provider includes `vllm`. The smoke metrics must include `speech_end_to_first_audio_ms <= 1500`, `barge_in_stop_ms <= 150`, `local_turns >= 1`, `local_turn_oracle_calls == 0`, `oracle_bound_turns >= 1`, and `oracle_bound_oracle_calls >= oracle_bound_turns`. For adapted KAME smoke, local reflex turns must not call the oracle, while oracle-bound turns must route through Hermes `/model` authority. Until measured evidence is supplied, the matrix must mark local Spark roles and `all_local_stack_smoke` as needing evidence rather than claiming readiness. The generated `.example.json` file is a passing-looking guide for measured Spark artifacts, but all `example_only: true` entries are rejected by the matrix. The generated Spark closure plan is the Milestone 4 checklist that the readiness closure index should point at for missing model, speech, and all-local stack-smoke proof.
The generated `spark-benchmark-scaffold/` directory is the preferred DGX operator starting point: it contains a wrapper benchmark evidence file plus placeholder raw-source artifacts that resolve correctly but are rejected until replaced with measured DGX Spark output and all `example_only` markers are removed.
The generated `spark-operator-runbook.md` is the step-by-step DGX collection runbook: start the local KAME stack, run `scripts/dgx_spark_gemma4_voice_eval.sh`, replace scaffold source artifacts with redacted measured outputs, lint the evidence with `--lint-evidence`, validate/write the matrix with `--evidence`, and re-index the full VoiceOps plan.

## Milestone 5: Operator Dashboard

Add an inspectable operations surface:

- current mode: dry-run, approval-required, or live
- active voice surface and fallback reason
- budget status
- pending approvals
- recent audit events
- provisioned services
- upcoming household/business tasks

The dashboard should support the operator workflow. It should not become a marketing page.

Headless command:

```bash
uv run python scripts/voiceops_operator_state.py --output-dir artifacts/voiceops-operator-state/current
```

The command writes:

- `operator-state.json`
- `operator-state.md`
- `operator-state-events.jsonl`

The operator-state generator is artifact-only. It does not read environment secrets, perform network I/O, send Discord/WhatsApp/SMS messages, place calls, provision services, or spend money. It gives the recording dashboard and future GUI a durable state contract for current mode, active/fallback voice surface, budget status, pending approvals, audit events, planned/provisioned services, and household/business tasks.

For the hackathon demo, the generated `operator-dashboard.html` from Milestone 0 should visibly show the same operator state shape: current mode, active voice surface, fallback reason, full budget status, pending approvals, recent audit events, planned services, and a link to `operator-state.json`.

## Headless Plan Run

Run every currently headless VoiceOps milestone artifact generator and write one evidence index:

No-write preflight:

```bash
uv run python scripts/voiceops_plan_run.py --artifact-root artifacts --output-dir artifacts/voiceops-plan/current --dry-audit --package-audit
```

`--dry-audit` builds the same plan summary in a temporary artifact root, prints the readiness gaps, closure status, safety flags, current-environment blockers, and ordered `next_actions`, then removes the temporary artifacts on exit. It does not write the requested artifact paths, and it refuses `--run-command-probes` and `--run-readonly-discovery` so it cannot silently become a subprocess or network-capable probe. Its `ok` field means no hard validation failures, not readiness; use `readiness_ok` or `closure_status: complete` for readiness automation. The `next_actions` records are machine-readable and include each remaining gate, whether the current host can run it, current environment blockers, the first safe evidence command, any separate diagnostic command, and the success check.

Artifact-writing indexer with final package audit:

```bash
uv run python scripts/voiceops_plan_run.py --artifact-root artifacts --output-dir artifacts/voiceops-plan/current --package-audit
```

The command writes:

- all Milestone 0 demo artifacts under `artifacts/hackathon-voiceops-demo/current`
- all Milestone 1 voice-operator artifacts under `artifacts/voiceops-voice-operator/current`
- all Milestone 2 provisioning preflight artifacts under `artifacts/voiceops-provisioning/current`
- all Milestone 3 channel policy artifacts under `artifacts/voiceops-channel-policy/current`
- all Milestone 4 Spark matrix artifacts under `artifacts/voiceops-spark-matrix/current`
- all Milestone 5 operator-state artifacts under `artifacts/voiceops-operator-state/current`
- `voiceops-plan-run.json`
- `voiceops-plan-run.md`
- `readiness-closure-index.json`
- `readiness-closure-index.md`
- `operator-handoff.json`
- `operator-handoff.md`
- `package-audit.json`
- `package-audit.md`

The audited plan run is artifact-only but writes the artifact tree above. It should surface readiness gaps such as missing Stripe/phone local setup or missing DGX Spark benchmark evidence, but those gaps must not cause live spend, provider provisioning, outbound messaging, calls, or secret reads. Use `--dry-audit --package-audit` first when the operator wants the same status check and package consistency check without persistent artifact writes.

The readiness closure index is the top-level next-action map for the remaining external evidence gates. It must keep live Discord voice evidence, Stripe/MPP/phone provisioning evidence, and DGX Spark benchmark evidence separate, list the required proof shape for each gate including collector attestation requirements, point at the relevant evidence templates and closure plans, emit ordered `next_actions`, and continue to report `needs_external_evidence` until supplied artifacts prove the live gates. It must never collapse missing live evidence into a single ready claim.

The operator handoff is the ordered execution runbook derived from the closure index. It must list the live Discord voice, spend/provisioning preflight, and local Spark stack phases in order, include exact collection and re-index commands, identify expected artifacts and success checks, and state that the handoff does not change readiness by itself. The final re-index command must include `--package-audit`, and package audit is part of final headless verification after every closure re-index.

The test suite includes a closure rehearsal with redacted local fixtures for all three remaining gates. It proves that supplied live voice evidence, provisioning preflight evidence, and Spark benchmark evidence with valid collector attestations can drive `readiness_gaps: []`, `closure_status: complete`, and `remaining_gates: []` without credentials, live Discord, provider actions, phone calls, network I/O, or DGX Spark execution. Real readiness still requires replacing those fixtures with actual collected evidence.

Standalone package consistency audit:

```bash
uv run python scripts/voiceops_artifact_package_audit.py --artifact-root artifacts --audit-only
```

Artifact-writing audit:

```bash
uv run python scripts/voiceops_artifact_package_audit.py \
  --artifact-root artifacts \
  --output-dir artifacts/voiceops-package-audit/current
```

The package audit is local and static. It reads the generated VoiceOps package and checks cross-artifact consistency between the demo readiness report, demo closure summary, plan closure index, operator state, dashboard HTML, NemoClaw packet validation, audit ledger, and dry-run shell metadata. It catches contradictions such as live-ready claims while closure gates remain, mismatched NemoClaw/operator approval contracts, executed audit rows in a dry-run package, external service provisioning claims without receipts, and missing non-live dashboard status. `--audit-only` performs no persistent writes. Prefer `voiceops_plan_run.py --package-audit` for normal headless operation so the plan index and package audit are generated and evaluated together.

When evidence exists, rerun the same indexer with the relevant read-only artifacts instead of hand-editing the index:

```bash
uv run python scripts/voiceops_plan_run.py --artifact-root artifacts \
  --output-dir artifacts/voiceops-plan/current \
  --package-audit \
  --voice-live-evidence artifacts/realtime-voice-evidence/live-current/manifest.json
```

```bash
uv run python scripts/voiceops_plan_run.py --artifact-root artifacts \
  --output-dir artifacts/voiceops-plan/current \
  --package-audit \
  --env-file .env \
  --read-only-discovery-evidence artifacts/voiceops-provisioning/current/read-only-discovery.manifest.json \
  --provisioning-preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-evidence.json
```

```bash
uv run python scripts/voiceops_plan_run.py --artifact-root artifacts \
  --output-dir artifacts/voiceops-plan/current \
  --package-audit \
  --env-file .env \
  --read-only-discovery-evidence artifacts/voiceops-provisioning/current/read-only-discovery.manifest.json \
  --provisioning-preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json
```

```bash
uv run python scripts/voiceops_plan_run.py --artifact-root artifacts \
  --output-dir artifacts/voiceops-plan/current \
  --package-audit \
  --env-file .env \
  --read-only-discovery-evidence artifacts/voiceops-provisioning/current/read-only-discovery.manifest.json \
  --provisioning-preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json \
  --post-approval-receipts artifacts/voiceops-provisioning/current/post-approval-receipts.json
```

```bash
uv run python scripts/voiceops_plan_run.py --artifact-root artifacts \
  --output-dir artifacts/voiceops-plan/current \
  --package-audit \
  --evidence artifacts/voiceops-spark-matrix/current/spark-benchmark-scaffold/spark-benchmark-evidence.json
```

If local setup discovery needs bounded binary/version checks, run the provisioning probe with explicit opt-in and then re-index its evidence:

```bash
uv run python scripts/voiceops_plan_run.py --artifact-root artifacts \
  --output-dir artifacts/voiceops-plan/current \
  --package-audit \
  --env-file .env \
  --run-command-probes
```

If local setup discovery needs authenticated display-only catalog/auth checks, keep it separate from version/help probes and run the exact read-only allowlist:

```bash
uv run python scripts/voiceops_plan_run.py --artifact-root artifacts \
  --output-dir artifacts/voiceops-plan/current \
  --package-audit \
  --env-file .env \
  --run-readonly-discovery
```

## Success Criteria

Short term:

- the hackathon demo can be recorded in 1-3 minutes
- the artifacts are generated headlessly
- no live spend occurs by default
- the story clearly ties Spark, Hermes, Stripe, and voice together

Medium term:

- the user can ask in Discord voice for a real household/business operation and receive a budgeted approval packet
- the same request can be followed up from WhatsApp
- one approved provisioning flow executes and records a complete ledger trail

Long term:

- the system can run daily household and business operations from one DGX Spark with minimal cloud dependence
- Hermes remains model-flexible through `/model`
- the user can trust the system because every action is scoped, approved, reversible where possible, and audited

## Non-Goals

- building a generic voice assistant detached from real operations
- creating a separate VoiceOps oracle model setting
- allowing the reflex broad tool or spend authority
- blocking the hackathon proof on fully local Gemma audio serving
- blocking the hackathon proof on real purchases
- hiding dry-run status to make the demo look more complete than it is
