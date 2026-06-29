# Spark Household and Business VoiceOps Plan

## Goal

Build a hackathon-ready Hermes entry that shows one DGX Spark running the private operating layer for a household and small business. The user talks to it through Discord voice first, then WhatsApp and phone as follow-on surfaces. Hermes plans work, gates spend, provisions services, and leaves an audit trail.

This should present as a useful operating system, not as a voice demo. Voice is the front door. Spark is the local compute base. Stripe/NVIDIA/Hermes are the business substrate.

## Product Thesis

Hermes VoiceOps is a local-first operator for daily life and business:

- Household: bills, subscriptions, maintenance, appointments, urgent alerts.
- Business: vendor setup, SaaS provisioning, customer ops, payments, recurring reviews.
- Voice surfaces: Discord live voice for the desk, WhatsApp for mobile chat, phone/SMS for urgent fallback.
- Compute: DGX Spark runs the KAME reflex, speech stack, and preferred local models where practical.
- Spend: Stripe Link and Stripe Projects become the controlled path for paying, provisioning, and recording approvals.

## Architecture Tracks

### Track 1: Spark-Powered Local Operator

- KAME reflex: Gemma 4 E2B/E4B-style audio-native interface model for fast turn handling and intent triage.
- Oracle: whatever Hermes is already using through `/model`; no separate `oracle_model` setting.
- Preferred local target: Gemma 4 26B-A4B on DGX Spark for the brain when available.
- Speech: local Nemotron Speech/Magpie/Riva-style ASR/TTS path, with Cartesia as a demo fallback.
- Evidence: benchmark and smoke artifacts from `scripts/dgx_spark_gemma4_voice_eval.sh`.

### Track 2: Voice Control Surfaces

- Discord: primary live voice demo through `/voice join`.
- WhatsApp: existing Hermes WhatsApp bridge and WhatsApp Cloud setup for mobile control.
- Phone/SMS: provision Twilio through Stripe Projects, then feed call summaries and urgent confirmations into the same ops queue.
- Barge-in: keep current KAME barge-in work visible as a quality differentiator.

### Track 3: Real Operations With Money

- Stripe Projects: provision Twilio, Neon, Vercel, or other services into a project with credential sync.
- Stripe Link CLI: gated spend requests for purchases or paid API credits.
- MPP/402: pay agent-facing services without exposing raw card data to the model.
- Audit ledger: every planned, approved, held, or executed action gets a durable event.
- Budget policy: default dry-run; live spend only after explicit user approval.

## Headless Demo Command

Run:

```bash
uv run python scripts/hackathon_voiceops_demo.py --output-dir artifacts/hackathon-voiceops-demo/current
```

The command writes:

- `voiceops-demo.json`
- `voiceops-demo.md`
- `audit-ledger.jsonl`
- `demo-script.md`
- `stripe-actions-dry-run.sh`

The generated shell script is dry-run by construction. It prints the Stripe/Projects commands instead of executing them.

## 90-Second Submission Flow

1. In Discord voice: "Hermes, set up tomorrow's paid household and business operations. Keep spend under 200 dollars."
2. Hermes gives a short reflex acknowledgement immediately.
3. Hermes produces the oracle plan: Twilio for phone/SMS escalation, Neon for the ops ledger, Link-gated spend for service credit, and recurring household review.
4. Show the generated audit ledger and budget totals.
5. Show WhatsApp/phone as the same operator surface for mobile and urgent paths.
6. Close on the DGX Spark thesis: private local operations, real spend controls, and voice-native interaction.

## Scope for June 30, 2026

Must have:

- Headless artifact generator.
- Demo script and dry-run Stripe action queue.
- Existing Discord voice branch as the live interaction path.
- Existing WhatsApp setup path referenced as the mobile follow-on.
- Clear audit trail and spend policy.

Nice to have:

- Live Stripe Projects catalog/list output if credentials are ready.
- Live Discord voice run with KAME fallback state visible.
- A screenshot or dashboard view of the generated ledger.

Do not block on:

- Fully local phone call audio.
- Fully local audio-native Gemma reflex serving.
- Real purchases.
- New GUI surface unless it is already cheap to expose the artifacts.

## Judging Fit

- Usefulness: runs real household and business workflows, not a toy chat.
- Viability: local-first Spark path, explicit budget gates, Stripe approval flow, durable audit trail.
- Presentation: voice-controlled operator with real provisioning and payment affordances.

