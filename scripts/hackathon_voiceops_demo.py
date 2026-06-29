#!/usr/bin/env python3
"""Generate a headless VoiceOps hackathon demo package.

The demo is intentionally credential-free by default. It emits the artifacts
needed to record a 1-3 minute submission while keeping live Stripe/Projects
execution behind explicit operator approval.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REQUEST = (
    "Hermes, I am giving you 200 dollars to use through Stripe Skills. "
    "Provision yourself a VoIP provider account, then call my phone with "
    "this same context so we can continue outside Discord."
)


@dataclass(frozen=True)
class VoiceSurface:
    channel: str
    role: str
    implementation: str
    status: str


@dataclass(frozen=True)
class SpendPolicy:
    name: str
    limit_cents: int
    approval_required_over_cents: int
    currency: str = "usd"


@dataclass(frozen=True)
class OpsAction:
    action_id: str
    provider: str
    command: str
    purpose: str
    estimated_cents: int
    requires_approval: bool
    status: str


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    actor: str
    action: str
    amount_cents: int
    status: str
    evidence: str


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _dollars(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def _slug(value: str) -> str:
    chars = []
    for ch in value.lower():
        if ch.isalnum():
            chars.append(ch)
        elif chars and chars[-1] != "-":
            chars.append("-")
    return "".join(chars).strip("-") or "voiceops"


def _surface_matrix() -> list[VoiceSurface]:
    return [
        VoiceSurface(
            channel="discord",
            role="primary realtime voice room",
            implementation="Hermes Discord gateway /voice join plus KAME realtime voice sidecar",
            status="implemented-on-branch",
        ),
        VoiceSurface(
            channel="whatsapp",
            role="mobile household command channel",
            implementation="existing WhatsApp bridge and WhatsApp Cloud setup path",
            status="repo-supported",
        ),
        VoiceSurface(
            channel="phone",
            role="outbound call handoff with the same operational context",
            implementation="Stripe Projects provisions Twilio or another VoIP provider; Hermes queues the call through the phone bridge",
            status="demo-call-queued",
        ),
    ]


def _sponsor_stack(oracle_model: str) -> dict[str, Any]:
    return {
        "nemotron_3_ultra": {
            "role": "hackathon-visible Hermes oracle/model target for serious planning and reasoning",
            "selection": oracle_model,
            "note": "Configured through Hermes' normal /model flow; VoiceOps does not introduce a separate oracle_model setting.",
        },
        "nemoclaw": {
            "role": "safe execution boundary for agent actions that touch tools, credentials, network, and spend",
            "demo_use": "wrap or present the Stripe/VoIP provisioning plan as a sandboxed execution packet before approval",
        },
        "stripe_skills": {
            "role": "controlled economic rail",
            "skills": ["stripe-projects", "stripe-link-cli", "mpp-agent"],
            "demo_use": "provision VoIP service, request approved spend, and preserve receipts/audit events",
        },
    }


def _spark_stack(oracle_model: str, reflex_model: str) -> dict[str, Any]:
    return {
        "compute": "1x NVIDIA DGX Spark target",
        "local_first": True,
        "reflex": {
            "model": reflex_model,
            "role": "low-latency KAME interface model for turn handling, intent triage, and floor control",
            "input": "native audio when available; explicit local STT fallback state otherwise",
        },
        "oracle": {
            "model": oracle_model,
            "role": "Hermes active model selected by /model; no separate oracle_model setting",
            "interface_contract": "receives committed intent, transcript evidence, spend policy, and tool plan",
        },
        "speech": {
            "asr": "Nemotron Speech or equivalent local streaming ASR for durable transcript evidence",
            "tts": "local Magpie/Riva-style TTS target with Cartesia cloud fallback for the demo",
        },
        "guardrails": [
            "budget caps",
            "approval gates",
            "audit ledger",
            "dry-run by default",
            "live spend only through Stripe Link approval",
        ],
    }


def _ops_actions(total_budget_cents: int) -> list[OpsAction]:
    base_actions = [
        OpsAction(
            action_id="grant-spend-budget",
            provider="voiceops-policy",
            command="record spend cap usd:200 approval_required:true",
            purpose="bind the spoken Discord budget to the approval and audit policy",
            estimated_cents=0,
            requires_approval=False,
            status="ready",
        ),
        OpsAction(
            action_id="provision-voip-provider",
            provider="stripe-projects",
            command="stripe projects add twilio/voice",
            purpose="provision a VoIP-capable provider account for outbound calls and SMS fallback",
            estimated_cents=2500,
            requires_approval=True,
            status="queued",
        ),
        OpsAction(
            action_id="buy-service-credit",
            provider="stripe-link-cli",
            command=(
                "link-cli spend-request create --merchant-name ExampleOps "
                "--merchant-url https://example.invalid --amount 4900 --request-approval"
            ),
            purpose="buy a prepaid operations API credit only after Link approval",
            estimated_cents=4900,
            requires_approval=True,
            status="queued",
        ),
        OpsAction(
            action_id="persist-call-context",
            provider="hermes-audit-ledger",
            command="write context packet for outbound phone handoff",
            purpose="preserve the Discord conversation, budget, approval state, and VoIP provisioning result for the phone call",
            estimated_cents=0,
            requires_approval=False,
            status="ready",
        ),
        OpsAction(
            action_id="call-user-phone",
            provider="voiceops-phone-bridge",
            command="queue outbound call --context artifacts/hackathon-voiceops-demo/current/voiceops-demo.json",
            purpose="call the user's phone and continue with the same Discord context",
            estimated_cents=0,
            requires_approval=True,
            status="queued",
        ),
        OpsAction(
            action_id="publish-status",
            provider="hermes-gateway",
            command="post summary to Discord and WhatsApp with phone-call audit ID",
            purpose="send the user a cross-channel approval packet and call handoff summary",
            estimated_cents=0,
            requires_approval=False,
            status="ready",
        ),
    ]
    running = 0
    selected: list[OpsAction] = []
    for action in base_actions:
        if running + action.estimated_cents <= total_budget_cents:
            selected.append(action)
            running += action.estimated_cents
        else:
            selected.append(
                OpsAction(
                    action_id=action.action_id,
                    provider=action.provider,
                    command=action.command,
                    purpose=action.purpose,
                    estimated_cents=action.estimated_cents,
                    requires_approval=True,
                    status="held-budget",
                )
            )
    return selected


def _audit_events(actions: Iterable[OpsAction]) -> list[AuditEvent]:
    events: list[AuditEvent] = []
    for index, action in enumerate(actions, start=1):
        events.append(
            AuditEvent(
                event_id=f"evt-{index:03d}",
                actor="hermes-voiceops",
                action=action.action_id,
                amount_cents=action.estimated_cents,
                status=action.status,
                evidence=f"action:{action.provider}:{action.action_id}",
            )
        )
    return events


def _nemoclaw_action_packet(demo: dict[str, Any]) -> dict[str, Any]:
    approval_actions = [action for action in demo["ops_actions"] if action["requires_approval"]]
    return {
        "packet_id": "voiceops-nemoclaw-demo-001",
        "runtime": "NemoClaw",
        "mode": "dry_run_until_user_approval",
        "source_channel": "discord_voice",
        "oracle_model": demo["sponsor_stack"]["nemotron_3_ultra"]["selection"],
        "spend_policy": demo["spend_policy"],
        "allowed_capabilities": [
            "stripe_projects_catalog",
            "stripe_projects_voip_provisioning_after_approval",
            "stripe_link_spend_request_after_approval",
            "phone_call_queue_after_approval",
            "discord_and_whatsapp_status_post",
        ],
        "blocked_capabilities": [
            "raw_card_data_in_model_context",
            "unapproved_purchase",
            "unapproved_recurring_charge",
            "unapproved_credential_deletion",
            "unbounded_network_access",
        ],
        "approval_required_actions": approval_actions,
        "dry_run_commands": [action["command"] for action in approval_actions],
        "audit_event_ids": [event["event_id"] for event in demo["audit_events"]],
    }


def _phone_context_packet(demo: dict[str, Any]) -> dict[str, Any]:
    approval_actions = [action for action in demo["ops_actions"] if action["requires_approval"]]
    return {
        "handoff_id": "voiceops-phone-handoff-001",
        "source_channel": "discord_voice",
        "target_channel": "phone",
        "status": "queued_requires_approval",
        "context_summary": (
            "The user gave Hermes a 200 dollar Stripe Skills budget in Discord voice, "
            "asked Hermes to provision a VoIP provider account, and asked Hermes to "
            "call their phone with the same context."
        ),
        "spoken_opening": (
            "I am continuing from Discord. You gave me a 200 dollar budget to "
            "provision VoIP through Stripe Skills, and I am waiting on your approval "
            "before live spend."
        ),
        "budget": demo["spend_policy"],
        "pending_approvals": [
            {
                "action_id": action["action_id"],
                "provider": action["provider"],
                "estimated_cents": action["estimated_cents"],
                "purpose": action["purpose"],
            }
            for action in approval_actions
        ],
        "audit_event_ids": [event["event_id"] for event in demo["audit_events"]],
    }


def build_demo(args: argparse.Namespace) -> dict[str, Any]:
    actions = _ops_actions(args.budget_cents)
    approval_total = sum(action.estimated_cents for action in actions if action.requires_approval and action.status == "queued")
    ready_total = sum(action.estimated_cents for action in actions if action.status in {"queued", "ready"})
    policy = SpendPolicy(
        name="household-business-daily-ops",
        limit_cents=args.budget_cents,
        approval_required_over_cents=args.approval_required_over_cents,
    )
    return {
        "generated_at": _utc_now(),
        "demo": {
            "name": args.demo_name,
            "request": args.request,
            "operator": "Hermes VoiceOps",
            "submission_theme": "give a Spark-powered Hermes agent spending money over Discord, let it provision VoIP through Stripe Skills, then continue by phone",
        },
        "sponsor_stack": _sponsor_stack(args.oracle_model),
        "spark_stack": _spark_stack(args.oracle_model, args.reflex_model),
        "voice_surfaces": [asdict(surface) for surface in _surface_matrix()],
        "spend_policy": asdict(policy),
        "ops_actions": [asdict(action) for action in actions],
        "audit_events": [asdict(event) for event in _audit_events(actions)],
        "totals": {
            "ready_or_queued_cents": ready_total,
            "approval_required_cents": approval_total,
            "held_budget_cents": sum(action.estimated_cents for action in actions if action.status == "held-budget"),
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, events: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events), encoding="utf-8")


def _markdown(demo: dict[str, Any]) -> str:
    totals = demo["totals"]
    lines = [
        f"# {demo['demo']['name']}",
        "",
        "## One-line pitch",
        "",
        "Hermes VoiceOps turns a DGX Spark into a local-first operator for a household and business, controlled by live voice in Discord with WhatsApp and phone escalation paths.",
        "",
        "## Sponsor stack",
        "",
        f"- Nemotron 3 Ultra: {demo['sponsor_stack']['nemotron_3_ultra']['role']}",
        f"- NemoClaw: {demo['sponsor_stack']['nemoclaw']['role']}",
        f"- Stripe Skills: {demo['sponsor_stack']['stripe_skills']['demo_use']}",
        "",
        "## Demo request",
        "",
        f"> {demo['demo']['request']}",
        "",
        "## Spark stack",
        "",
        f"- Compute: {demo['spark_stack']['compute']}",
        f"- Reflex: {demo['spark_stack']['reflex']['model']} for low-latency KAME interface behavior",
        f"- Oracle: {demo['spark_stack']['oracle']['model']} via Hermes' normal active model selection",
        f"- Speech: {demo['spark_stack']['speech']['asr']} plus {demo['spark_stack']['speech']['tts']}",
        "",
        "## Voice surfaces",
        "",
    ]
    for surface in demo["voice_surfaces"]:
        lines.append(f"- {surface['channel']}: {surface['role']} ({surface['status']})")
    lines.extend([
        "",
        "## Spend controls",
        "",
        f"- Budget: {_dollars(demo['spend_policy']['limit_cents'])}",
        f"- Approval threshold: {_dollars(demo['spend_policy']['approval_required_over_cents'])}",
        f"- Approval-required queued spend: {_dollars(totals['approval_required_cents'])}",
        f"- Held over budget: {_dollars(totals['held_budget_cents'])}",
        "",
        "## Action queue",
        "",
    ])
    for action in demo["ops_actions"]:
        approval = "approval required" if action["requires_approval"] else "no approval needed"
        lines.append(
            f"- {action['action_id']}: {action['provider']} -> {action['status']}, "
            f"{_dollars(action['estimated_cents'])}, {approval}"
        )
    lines.extend([
        "",
        "## Evidence artifacts",
        "",
        "- `nemoclaw-action-packet.json`: sandbox and approval frame for billable/network-capable actions",
        "- `phone-context.json`: outbound phone-call handoff context preserved from Discord",
        "",
        "## 90-second video beat sheet",
        "",
        "1. Join Discord voice and give Hermes a fixed Stripe Skills budget.",
        "2. Show Hermes producing a KAME reflex acknowledgement immediately, then a Nemotron-backed operating plan.",
        "3. Show the NemoClaw/sandboxed action packet before anything billable runs.",
        "4. Show the Stripe/Projects queue for VoIP provisioning and a Link-gated service-credit spend.",
        "5. Show Hermes preserving the Discord context and queuing an outbound phone call.",
        "6. Close by continuing the same task from the phone-call surface.",
        "",
    ])
    return "\n".join(lines)


def _demo_script(demo: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Demo Script",
            "",
            "User, spoken in Discord:",
            "",
            f"  {demo['demo']['request']}",
            "",
            "Hermes reflex reply:",
            "",
            "  I heard you. I will keep this under 200 dollars and ask before anything billable runs.",
            "",
            "Hermes oracle reply, using the hackathon sponsor stack:",
            "",
            "  I prepared a NemoClaw-safe action packet, queued Stripe Projects to provision a VoIP provider, and queued a Link-gated spend request for service credit. I also preserved this Discord context for the outbound phone call.",
            "",
            "Phone handoff:",
            "",
            "  Hermes calls the user's phone and says: I am continuing from Discord. You gave me a 200 dollar budget to provision VoIP through Stripe Skills, and I am waiting on your approval before live spend.",
            "",
            "Close:",
            "",
            "  This is one Spark-powered Hermes operator carrying context across Discord, Stripe-provisioned VoIP, WhatsApp, and phone.",
            "",
        ]
    )


def _stripe_script(actions: Iterable[dict[str, Any]]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Dry-run action queue generated for the Hermes VoiceOps hackathon demo.",
        "# Remove the leading 'printf' only after explicit user approval.",
        "",
    ]
    for action in actions:
        command = action["command"]
        quoted = shlex.quote(command)
        lines.append(f"printf '%s\\n' {quoted}")
    lines.append("")
    return "\n".join(lines)


def write_demo(output_dir: Path, demo: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "voiceops-demo.json",
        "markdown": output_dir / "voiceops-demo.md",
        "audit_ledger": output_dir / "audit-ledger.jsonl",
        "demo_script": output_dir / "demo-script.md",
        "nemoclaw_packet": output_dir / "nemoclaw-action-packet.json",
        "phone_context": output_dir / "phone-context.json",
        "stripe_actions": output_dir / "stripe-actions-dry-run.sh",
    }
    _write_json(paths["json"], demo)
    paths["markdown"].write_text(_markdown(demo), encoding="utf-8")
    _write_jsonl(paths["audit_ledger"], demo["audit_events"])
    paths["demo_script"].write_text(_demo_script(demo), encoding="utf-8")
    _write_json(paths["nemoclaw_packet"], _nemoclaw_action_packet(demo))
    _write_json(paths["phone_context"], _phone_context_packet(demo))
    paths["stripe_actions"].write_text(_stripe_script(demo["ops_actions"]), encoding="utf-8")
    paths["stripe_actions"].chmod(0o755)
    return {key: str(path) for key, path in paths.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hackathon-voiceops-demo/current"))
    parser.add_argument("--demo-name", default="Hermes VoiceOps on DGX Spark")
    parser.add_argument("--request", default=DEFAULT_REQUEST)
    parser.add_argument("--budget-cents", type=int, default=20_000)
    parser.add_argument("--approval-required-over-cents", type=int, default=1_000)
    parser.add_argument("--oracle-model", default="Nemotron 3 Ultra via Hermes /model for the hackathon demo")
    parser.add_argument("--reflex-model", default="Gemma 4 E2B audio-native reflex on Spark")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.budget_cents < 0:
        raise SystemExit("--budget-cents must be non-negative")
    if args.approval_required_over_cents < 0:
        raise SystemExit("--approval-required-over-cents must be non-negative")
    demo = build_demo(args)
    paths = write_demo(args.output_dir, demo)
    print(json.dumps({"ok": True, "output_dir": str(args.output_dir), "artifacts": paths}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
