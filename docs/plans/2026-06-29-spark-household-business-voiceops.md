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
- Reasoning: Nemotron 3 Ultra is the hackathon-visible oracle/model target, while Hermes `/model` remains authoritative.
- Safety: NemoClaw is the preferred sponsor-aligned execution boundary for spend, provisioning, and network-capable action packets.
- Spend: Stripe Link, Stripe Projects, and MPP/402 become the controlled path for paying, provisioning, and recording approvals.
- Audit: every planned, approved, held, executed, failed, or rolled-back action is durable and inspectable.

The hackathon entry is Milestone 0: a public proof that this shape is useful, viable, and presentable.

## Spark Model Strategy

There are two related but distinct model strategies.

### Hackathon Strategy

Use Nemotron 3 Ultra visibly as the Hermes oracle/model path for the demo because it is sponsor-aligned and communicates serious agentic reasoning. The demo should show Nemotron 3 Ultra as the planner behind the budgeted VoIP provisioning workflow.

This does not mean VoiceOps adds a separate model selector. Nemotron 3 Ultra should still be selected through Hermes's normal `/model` flow. The goal is to prove that Hermes can carry a Discord voice request into a safe, budgeted, tool-using business operation with NVIDIA/Stripe integrations visible.

Hackathon stack:

- Nemotron 3 Ultra as the serious reasoning/planning path
- NemoClaw as the safe execution boundary
- Stripe Skills as the spend and provisioning rail
- Discord voice as the live interface
- DGX Spark as the target local operating base and NVIDIA story

### One-Spark Local Strategy

For the long-term household/business appliance, do not assume Nemotron 3 Ultra is the local one-Spark brain unless NVIDIA provides a practical serving path for this hardware. The actual one-Spark local oracle must be chosen by benchmark and operational feel.

Target KAME layout:

- Reflex/interface: Gemma 4 E2B or E4B-style audio-native model, always warm, optimized for turn-taking and routing.
- Oracle/brain: whatever Hermes `/model` selects, with Gemma 4 26B-A4B as the first preferred local candidate to evaluate on DGX Spark.
- Speech: local ASR/TTS where practical, with ASR used as oracle evidence rather than reflex input in full KAME mode.
- Fallbacks: hosted Nemotron, Kimi, Cartesia, or other cloud providers are acceptable during bring-up and demos when they are labeled clearly.

The public demo can use Nemotron 3 Ultra for sponsor fit while the private appliance roadmap benchmarks Gemma 4 26B-A4B and other Spark-friendly models for the local brain.

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

The oracle is Hermes's active model, selected through the existing Hermes `/model` flow. There should not be a separate `oracle_model` setting for VoiceOps. If the user points Hermes at Kimi, Gemma 4 26B-A4B, Nemotron, or another provider, that is the oracle.

Target:

- Nemotron 3 Ultra for the hackathon demo and sponsor-aligned planning path, selected through `/model`
- current cloud model for bring-up if needed
- local Gemma 4 26B-A4B on DGX Spark when it proves good enough for Hermes work and beats alternatives on latency, quality, and reliability
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
- Nemotron 3 Ultra is visible as the serious planning/oracle path for the demo.
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
4. Hermes uses Nemotron 3 Ultra for the plan.
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
- `phone-context.json`
- `readiness-report.json`
- `readiness-report.md`
- `stripe-actions-dry-run.sh`

The generated shell script is dry-run by construction. It prints the Stripe/Projects commands instead of executing them. The readiness report is non-invasive: it checks local prerequisites and env shape, but it does not provision, purchase, call, or mutate credentials.

## Milestone 1: Real Voice Operator

Make `/voice join` usable as the daily control surface:

- stable Discord receive/playback lifecycle
- real barge-in based on speech energy, not silent packet arrival
- KAME fallback state visible when audio-native reflex is unavailable
- voice replies short by default
- latency metrics from user speech end to reflex response, oracle response, and TTS playback
- voice capability prompt context so Hermes does not claim it cannot hear or speak

## Milestone 2: Real Spend and Provisioning

Turn the dry-run queue into controlled live operations:

- verify Stripe Link CLI auth and approval flow
- verify Stripe Projects plugin and catalog on the target machine
- run `stripe projects list` and safe catalog discovery headlessly
- execute one low-risk live VoIP provisioning path only after explicit approval
- queue or perform one outbound phone call with a preserved context packet
- record receipts and generated credential locations without exposing secrets
- add rollback/deprovision notes to the ledger

## Milestone 3: Multi-Channel Operations

Make the same operator reachable beyond Discord:

- WhatsApp Cloud setup path validated for command and approval messages
- phone/SMS path designed around Twilio or equivalent provisioning
- channel-specific authorization rules
- escalation policy for urgent household/business events
- consistent audit IDs across Discord, WhatsApp, and phone/SMS

## Milestone 4: Local Spark Stack

Move as much of the stack as possible onto one DGX Spark:

- local reflex model launch evidence
- local Hermes oracle endpoint registered through normal `/model` selection
- Gemma 4 26B-A4B evaluated as the preferred local brain
- local ASR/TTS bridge evidence
- all-local smoke with oracle, interface, ASR, TTS, and sidecar together
- benchmark evidence accepted by the generated DGX Spark matrix validator

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
