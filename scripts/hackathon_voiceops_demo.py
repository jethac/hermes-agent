#!/usr/bin/env python3
"""Generate a headless VoiceOps hackathon demo package.

The demo is intentionally credential-free by default. It emits the artifacts
needed to record a 1-3 minute submission while keeping live Stripe/Projects
execution behind explicit operator approval.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import shlex
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.voiceops_provisioning_probe import build_milestone2_execution_plan


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


@dataclass(frozen=True)
class ReadinessCheck:
    check_id: str
    status: str
    required_for_video: bool
    detail: str
    next_step: str


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _dollars(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def _h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _slug(value: str) -> str:
    chars = []
    for ch in value.lower():
        if ch.isalnum():
            chars.append(ch)
        elif chars and chars[-1] != "-":
            chars.append("-")
    return "".join(chars).strip("-") or "voiceops"


def _env_present(env: Mapping[str, str], key: str) -> bool:
    return bool(str(env.get(key) or "").strip())


def _env_truthy(env: Mapping[str, str], key: str) -> bool:
    return str(env.get(key) or "").strip().lower() in {"1", "true", "yes", "on"}


def _which_any(which: Callable[[str], str | None], commands: Iterable[str]) -> str | None:
    for command in commands:
        path = which(command)
        if path:
            return path
    return None


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return values
    except OSError:
        return values
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _merge_env_sources(
    env: Mapping[str, str],
    env_files: Iterable[Path],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    merged = dict(env)
    sources: list[dict[str, Any]] = [{"kind": "process", "loaded": True, "key_count": len(env)}]
    for path in env_files:
        parsed = _parse_env_file(path)
        exists = path.exists()
        # Mirror Hermes readiness semantics: env files can prove configured
        # state even when the current shell did not export those variables.
        merged.update(parsed)
        sources.append(
            {
                "kind": "env_file",
                "path": str(path),
                "exists": exists,
                "loaded": bool(parsed),
                "key_count": len(parsed),
            }
        )
    return merged, sources


def _default_readiness_env_files(hermes_home: Path | None = None) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[1]
    resolved_home = hermes_home or Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
    return [repo_root / ".env", resolved_home / ".env"]


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


def _active_model_path(active_model: str) -> dict[str, Any]:
    normalized = active_model.lower()
    is_super = "nemotron" in normalized and "super" in normalized
    is_ultra = "nemotron" in normalized and "ultra" in normalized
    if is_super:
        return {
            "active_model": active_model,
            "selected_by": "Hermes /model",
            "path": "spark_local_nemotron_3_super",
            "status": "preferred_local",
            "recording_ready": True,
            "label": "Nemotron 3 Super on DGX Spark",
        }
    if is_ultra:
        return {
            "active_model": active_model,
            "selected_by": "Hermes /model",
            "path": "hosted_nemotron_3_ultra_fallback",
            "status": "hosted_fallback",
            "recording_ready": True,
            "label": "Nemotron 3 Ultra hosted fallback",
        }
    return {
        "active_model": active_model,
        "selected_by": "Hermes /model",
        "path": "non_nvidia_fallback",
        "status": "needs_nemotron_selection",
        "recording_ready": False,
        "label": "Switch Hermes to Nemotron 3 Super or hosted Nemotron 3 Ultra",
    }


def _sponsor_stack(active_model: str) -> dict[str, Any]:
    active_path = _active_model_path(active_model)
    return {
        "hermes_active_model": active_path,
        "nemotron_3_super": {
            "role": "preferred Spark-local NVIDIA oracle target for serious planning and reasoning",
            "selection": active_model if active_path["path"] == "spark_local_nemotron_3_super" else "not selected",
            "note": "Configured through Hermes' normal /model flow; VoiceOps does not introduce a separate oracle_model setting.",
        },
        "nemotron_3_ultra_hosted_fallback": {
            "role": "hosted fallback when the local Nemotron 3 Super Spark path is unavailable or still under benchmark",
            "selection": active_model if active_path["path"] == "hosted_nemotron_3_ultra_fallback" else "available fallback via Hermes /model hosted provider path",
            "note": "Fallback is still selected through Hermes' normal /model flow, not a VoiceOps-specific model setting.",
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


def _spark_stack(active_model: str, reflex_model: str) -> dict[str, Any]:
    return {
        "compute": "1x NVIDIA DGX Spark target",
        "local_first": True,
        "reflex": {
            "model": reflex_model,
            "role": "low-latency KAME interface model for turn handling, intent triage, and floor control",
            "input": "native audio when available; explicit local STT fallback state otherwise",
        },
        "oracle": {
            "model": active_model,
            "selected_by": "Hermes /model",
            "active_model_path": _active_model_path(active_model),
            "role": "Hermes active model selected by /model; no separate oracle_model setting",
            "interface_contract": "receives committed intent, transcript evidence, spend policy, and tool plan",
            "preferred_local_target": "Nemotron 3 Super on DGX Spark",
            "hosted_fallback": "Nemotron 3 Ultra via Hermes /model hosted provider path",
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
        "hermes_active_model": demo["sponsor_stack"]["hermes_active_model"]["active_model"],
        "model_selected_by": "Hermes /model",
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


def _operator_state_packet(demo: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    phone_context = _phone_context_packet(demo)
    discord_check = next((check for check in readiness["checks"] if check["check_id"] == "discord_voice"), None)
    fallback_reason = (
        discord_check["detail"]
        if discord_check and discord_check["status"] != "pass"
        else "Phone/SMS voice calls remain disabled until provisioning approval and channel policy review pass."
    )
    pending_approvals = [
        {
            "approval_id": f"voiceops-demo-{action['action_id']}",
            "action_id": action["action_id"],
            "provider": action["provider"],
            "title": action["purpose"],
            "estimated_cents": action["estimated_cents"],
            "currency": demo["spend_policy"]["currency"],
            "status": "pending",
            "artifact_ref": "nemoclaw-action-packet.json",
        }
        for action in demo["ops_actions"]
        if action["requires_approval"] and action["status"] == "queued"
    ]
    planned_services = [
        {
            "service_id": action["action_id"],
            "name": action["purpose"],
            "provider": action["provider"],
            "status": action["status"],
            "artifact_ref": "voiceops-demo.json",
        }
        for action in demo["ops_actions"]
        if action["action_id"] in {"provision-voip-provider", "buy-service-credit", "call-user-phone", "publish-status"}
    ]
    return {
        "schema_version": "voiceops.operator_state.v1",
        "generated_at": _utc_now(),
        "artifact_only": True,
        "current_mode": "approval-required",
        "mode": {
            "artifact_only": True,
            "bounded": True,
            "env_presence_inspection": True,
            "env_secret_values_emitted": False,
            "headless": True,
            "live_spend": False,
            "network_io": False,
            "outbound_calls": False,
            "outbound_sends": False,
            "provisioning": False,
        },
        "active_voice_surface": {
            "surface_id": phone_context["source_channel"],
            "display_name": "Discord voice",
            "status": "active_for_demo" if readiness["ready_for_recording"] else "needs_setup",
            "fallback_surface_id": phone_context["target_channel"],
            "fallback_reason": fallback_reason,
        },
        "budget_status": {
            "currency": demo["spend_policy"]["currency"],
            "current_mode": "approval-required",
            "limit_cents": demo["spend_policy"]["limit_cents"],
            "approval_required_over_cents": demo["spend_policy"]["approval_required_over_cents"],
            "queued_cents": demo["totals"]["approval_required_cents"],
            "ready_or_queued_cents": demo["totals"]["ready_or_queued_cents"],
            "held_budget_cents": demo["totals"]["held_budget_cents"],
            "remaining_before_approval_cents": max(
                0, demo["spend_policy"]["limit_cents"] - demo["totals"]["ready_or_queued_cents"]
            ),
            "status": "no_live_spend_without_explicit_approval",
        },
        "pending_approvals": pending_approvals,
        "recent_audit_events": demo["audit_events"][-5:],
        "planned_services": planned_services,
        "provisioned_services": [
            {
                "service_id": "repo_local_demo_artifacts",
                "name": "Repo-local VoiceOps demo artifacts",
                "provider": "filesystem",
                "status": "provisioned",
                "artifact_ref": "operator-dashboard.html",
            }
        ],
        "upcoming_tasks": [
            {
                "task_id": "household-budget-review",
                "domain": "household",
                "title": "Review household spending requests before approval",
                "status": "planned",
                "requires_approval": False,
            },
            {
                "task_id": "business-phone-handoff",
                "domain": "business",
                "title": "Complete VoIP handoff setup after explicit approval",
                "status": "blocked_on_approval",
                "requires_approval": True,
            },
        ],
    }


def _demo_milestone2_report(demo: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    checks = []
    for check in readiness["checks"]:
        if check["check_id"] in {
            "stripe_projects_cli",
            "stripe_link_cli",
            "nemoclaw_boundary",
            "phone_handoff",
        }:
            area = {
                "stripe_projects_cli": "stripe_projects",
                "stripe_link_cli": "stripe_link",
                "nemoclaw_boundary": "mpp",
                "phone_handoff": "phone_handoff",
            }[check["check_id"]]
            check_id = {
                "stripe_projects_cli": "stripe_projects_cli",
                "stripe_link_cli": "stripe_link_cli",
                "nemoclaw_boundary": "mpp_agent",
                "phone_handoff": "phone_target",
            }[check["check_id"]]
            checks.append(
                {
                    "check_id": check_id,
                    "area": area,
                    "status": "pass" if check["status"] == "pass" else "fail",
                    "required": check["required_for_video"] or check_id in {"phone_target"},
                    "detail": check["detail"],
                    "next_step": check["next_step"],
                    "evidence": {"source": "readiness-report.json"},
                }
            )
    existing = {check["check_id"] for check in checks}
    if "stripe_cli" not in existing:
        stripe_projects = next(check for check in checks if check["check_id"] == "stripe_projects_cli")
        checks.insert(
            0,
            {
                "check_id": "stripe_cli",
                "area": "stripe_projects",
                "status": stripe_projects["status"],
                "required": True,
                "detail": stripe_projects["detail"],
                "next_step": stripe_projects["next_step"],
                "evidence": {"source": "readiness-report.json"},
            },
        )
    if "phone_provider" not in existing:
        phone_ready = next(check for check in checks if check["check_id"] == "phone_target")
        checks.append(
            {
                "check_id": "phone_provider",
                "area": "phone_handoff",
                "status": phone_ready["status"],
                "required": True,
                "detail": "demo provider readiness follows phone handoff readiness",
                "next_step": "Run the provisioning preflight before any approved live phone handoff.",
                "evidence": {"source": "readiness-report.json"},
            }
        )
    required_failures = [check["check_id"] for check in checks if check["required"] and check["status"] != "pass"]
    return {
        "generated_at": readiness["generated_at"],
        "ready": not required_failures,
        "required_failures": required_failures,
        "checks": checks,
        "probe": {
            "name": "voiceops_demo_milestone2_readiness",
            "non_mutating": True,
            "bounded": True,
            "run_commands": False,
            "blocked_capabilities": [
                "live_spend",
                "provider_provisioning",
                "credential_retrieval",
                "outbound_phone_calls",
                "account_mutation",
            ],
        },
        "demo_refs": {
            "phone_context": "phone-context.json",
            "nemoclaw_packet": "nemoclaw-action-packet.json",
            "budget_cents": demo["spend_policy"]["limit_cents"],
        },
    }


def build_readiness_report(
    demo: dict[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    env_files: Iterable[Path] = (),
    which: Callable[[str], str | None] = shutil.which,
) -> dict[str, Any]:
    env, env_sources = _merge_env_sources(os.environ if env is None else env, env_files)
    checks: list[ReadinessCheck] = []

    hermes_path = _which_any(which, ["hermes"])
    checks.append(
        ReadinessCheck(
            check_id="hermes_cli",
            status="pass" if hermes_path else "warn",
            required_for_video=False,
            detail=f"hermes command found at {hermes_path}" if hermes_path else "hermes command not found on PATH",
            next_step="Use this repo with uv for artifacts, or install/point the system hermes command at this branch.",
        )
    )

    discord_ok = _env_present(env, "DISCORD_BOT_TOKEN") and (
        _env_present(env, "DISCORD_VOICE_CHANNEL_ID") or _env_present(env, "DISCORD_VOICE_CHANNEL_NAME")
    )
    checks.append(
        ReadinessCheck(
            check_id="discord_voice",
            status="pass" if discord_ok else "fail",
            required_for_video=True,
            detail=(
                "DISCORD_BOT_TOKEN and a voice channel selector are present"
                if discord_ok
                else "missing DISCORD_BOT_TOKEN or DISCORD_VOICE_CHANNEL_ID/DISCORD_VOICE_CHANNEL_NAME"
            ),
            next_step="Set Discord gateway env, restart Hermes gateway, then use /voice join in the recording server.",
        )
    )

    checks.append(
        ReadinessCheck(
            check_id="nemotron_3_super_or_ultra_active_model",
            status="pass" if demo["sponsor_stack"]["hermes_active_model"]["recording_ready"] else "fail",
            required_for_video=True,
            detail=(
                f"Hermes active model path is {demo['sponsor_stack']['hermes_active_model']['label']}: "
                f"{demo['sponsor_stack']['hermes_active_model']['active_model']}"
            ),
            next_step="Before recording, switch Hermes with /model so the visible model path is Nemotron 3 Super on Spark, or Ultra as the hosted fallback.",
        )
    )

    nemoclaw_path = _which_any(which, ["nemoclaw", "openshell"])
    checks.append(
        ReadinessCheck(
            check_id="nemoclaw_boundary",
            status="pass" if nemoclaw_path else "warn",
            required_for_video=False,
            detail=(
                f"NemoClaw/OpenShell command found at {nemoclaw_path}"
                if nemoclaw_path
                else "no nemoclaw or openshell command found; generated packet still demonstrates the policy boundary"
            ),
            next_step="If available, record the action packet inside NemoClaw/OpenShell; otherwise show nemoclaw-action-packet.json as the approval boundary.",
        )
    )

    stripe_path = _which_any(which, ["stripe"])
    stripe_projects_verified = _env_truthy(env, "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED")
    checks.append(
        ReadinessCheck(
            check_id="stripe_projects_cli",
            status="pass" if stripe_path and stripe_projects_verified else "fail",
            required_for_video=True,
            detail=(
                f"stripe CLI found at {stripe_path}; Projects help verification marker is present"
                if stripe_path and stripe_projects_verified
                else "stripe CLI found, but VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED is not set"
                if stripe_path
                else "stripe CLI not found"
            ),
            next_step=(
                "Run `stripe projects --help` or the provisioning preflight command, then set "
                "VOICEOPS_STRIPE_PROJECTS_HELP_VERIFIED=true before recording this as ready."
            ),
        )
    )

    link_path = _which_any(which, ["link-cli"])
    checks.append(
        ReadinessCheck(
            check_id="stripe_link_cli",
            status="pass" if link_path else "fail",
            required_for_video=True,
            detail=(
                f"link-cli found at {link_path}"
                if link_path
                else "link-cli not found on PATH; npx is not treated as ready because it may fetch packages"
            ),
            next_step="Install a pinned @stripe/link-cli binary, then authenticate Link before any live spend.",
        )
    )

    whatsapp_ready = _env_truthy(env, "WHATSAPP_ENABLED") or (
        _env_present(env, "WHATSAPP_CLOUD_PHONE_NUMBER_ID") and _env_present(env, "WHATSAPP_CLOUD_ACCESS_TOKEN")
    )
    checks.append(
        ReadinessCheck(
            check_id="whatsapp_followup",
            status="pass" if whatsapp_ready else "warn",
            required_for_video=False,
            detail=(
                "WhatsApp env indicates a configured bridge or Cloud API path"
                if whatsapp_ready
                else "WhatsApp is not configured; keep it as a roadmap/follow-on surface in the demo"
            ),
            next_step="Run `hermes whatsapp` or the WhatsApp Cloud setup if mobile follow-up will be shown live.",
        )
    )

    phone_ready = _env_present(env, "VOICEOPS_DEMO_PHONE_NUMBER") or _env_present(env, "TWILIO_ACCOUNT_SID")
    checks.append(
        ReadinessCheck(
            check_id="phone_handoff",
            status="pass" if phone_ready else "warn",
            required_for_video=False,
            detail=(
                "phone target/provider env is present"
                if phone_ready
                else "no VOICEOPS_DEMO_PHONE_NUMBER or TWILIO_ACCOUNT_SID; generated phone-context.json remains dry-run evidence"
            ),
            next_step="Set VOICEOPS_DEMO_PHONE_NUMBER and complete approved VoIP provisioning before attempting a live call.",
        )
    )

    check_dicts = [asdict(check) for check in checks]
    required_failures = [check for check in check_dicts if check["required_for_video"] and check["status"] != "pass"]
    return {
        "generated_at": _utc_now(),
        "ready_for_recording": not required_failures,
        "required_failures": [check["check_id"] for check in required_failures],
        "env_sources": env_sources,
        "checks": check_dicts,
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
        "sponsor_stack": _sponsor_stack(args.active_model),
        "spark_stack": _spark_stack(args.active_model, args.reflex_model),
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
        f"- Nemotron 3 Super: {demo['sponsor_stack']['nemotron_3_super']['role']}",
        f"- Nemotron 3 Ultra fallback: {demo['sponsor_stack']['nemotron_3_ultra_hosted_fallback']['role']}",
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
        "- `readiness-report.json`: local recording prerequisite report",
        "- `milestone2-execution-plan.json`: post-approval execution contract for VoIP provisioning, spend request, call receipt, credential reference, and rollback",
        "- `operator-dashboard.html`: static recording dashboard for budget, approvals, guardrails, and handoff state",
        "- `operator-state.json`: machine-readable current mode, surface, budget, approval, service, and task state",
        "- `operator-state-events.jsonl`: append-friendly view of recent operator audit events",
        "- `recording-runbook.md`: shot list, fallback plan, and submission checklist",
        "- `submission-writeup.md`: public submission copy for the tweet/thread/form",
        "",
        "## 90-second video beat sheet",
        "",
        "1. Join Discord voice and give Hermes a fixed Stripe Skills budget.",
        "2. Show Hermes producing a KAME reflex acknowledgement immediately, then a Nemotron 3 Super-backed operating plan.",
        "3. Show the NemoClaw/sandboxed action packet before anything billable runs.",
        "4. Show the Stripe/Projects queue for VoIP provisioning and a Link-gated service-credit spend.",
        "5. Show Hermes preserving the Discord context and queuing an outbound phone call.",
        "6. Close by continuing the same task from the phone-call surface.",
        "",
    ])
    return "\n".join(lines)


def _submission_writeup(demo: dict[str, Any]) -> str:
    policy = demo["spend_policy"]
    approval_cents = demo["totals"]["approval_required_cents"]
    lines = [
        "# Hermes VoiceOps Submission Writeup",
        "",
        "## Short Description",
        "",
        "Hermes VoiceOps is a local-first household and business operator controlled by live Discord voice, targeted at one NVIDIA DGX Spark. In the demo, the user gives Hermes a fixed budget through Stripe Skills, Hermes prepares a NemoClaw-safe plan, queues VoIP provisioning through Stripe Projects, and preserves context for a phone handoff.",
        "",
        "## What The Demo Shows",
        "",
        "- Discord voice is the live front door.",
        f"- {demo['sponsor_stack']['nemotron_3_super']['selection']} is the visible serious planning path.",
        f"- {demo['sponsor_stack']['nemotron_3_ultra_hosted_fallback']['selection']} is the hosted fallback if the local Spark path is unavailable.",
        "- NemoClaw frames billable and network-capable actions before execution.",
        "- Stripe Skills provide the spend and provisioning rail.",
        f"- The spoken budget becomes a {_dollars(policy['limit_cents'])} spend policy with approval required over {_dollars(policy['approval_required_over_cents'])}.",
        f"- {_dollars(approval_cents)} of queued spend remains approval-gated.",
        "- Phone and WhatsApp are treated as follow-on operational surfaces, not separate assistants.",
        "",
        "## Why It Matters",
        "",
        "The goal is not a generic voice assistant. The goal is an operator that can run real household and small-business workflows while keeping money, credentials, network effects, and external communication behind explicit controls and an audit trail.",
        "",
        "## Spark Strategy",
        "",
        f"- Compute target: {demo['spark_stack']['compute']}",
        f"- Reflex/interface: {demo['spark_stack']['reflex']['model']}",
        f"- Oracle/brain: {demo['spark_stack']['oracle']['model']}",
        f"- Speech path: {demo['spark_stack']['speech']['asr']} plus {demo['spark_stack']['speech']['tts']}",
        "",
        "## Safety Story",
        "",
        "- Dry-run by default.",
        "- Live spend requires user-visible approval.",
        "- Reflex model does not get broad tool or spend authority.",
        "- NemoClaw packet lists blocked capabilities such as raw card data in model context and unapproved purchases.",
        "- Audit ledger records proposed actions, budget impact, approval state, and handoff evidence.",
        "",
        "## Demo Prompt",
        "",
        f"> {demo['demo']['request']}",
        "",
        "## Submission Tweet",
        "",
        "Hermes VoiceOps: I give a Spark-powered Hermes agent a budget over Discord voice, it builds a NemoClaw-safe plan, queues Stripe Skills provisioning for VoIP, and preserves context for a phone handoff. Local-first household/business ops, with spend gated by approval. @NousResearch",
        "",
    ]
    return "\n".join(lines)


def _recording_runbook(demo: dict[str, Any], readiness: dict[str, Any]) -> str:
    failures = readiness["required_failures"]
    fallback = (
        "Use the static dashboard plus generated dry-run packets; narrate the missing required checks directly."
        if failures
        else "Record live Discord voice first, then cut to the dashboard and generated action packets."
    )
    lines = [
        "# VoiceOps Recording Runbook",
        "",
        "## Goal",
        "",
        "Record a 1-3 minute hackathon video showing Hermes as a Spark-powered household and business operator: Discord voice in, Stripe Skills budget and provisioning plan, NemoClaw safety boundary, and phone handoff with preserved context.",
        "",
        "## Regenerate Artifacts",
        "",
        "```bash",
        "uv run python scripts/hackathon_voiceops_demo.py --output-dir artifacts/hackathon-voiceops-demo/current",
        "```",
        "",
        "Open `artifacts/hackathon-voiceops-demo/current/operator-dashboard.html` directly in a browser for the recording surface.",
        "",
        "## Readiness Gate",
        "",
        f"- Ready for recording: {'yes' if readiness['ready_for_recording'] else 'no'}",
        f"- Required failures: {', '.join(failures) if failures else 'none'}",
        f"- Recording fallback: {fallback}",
        "",
        "Do not show terminal panes or files that contain secrets. Do not run live spend or provisioning unless the user explicitly approves it during the recording.",
        "",
        "## Spoken Demo Prompt",
        "",
        "Say this in Discord voice:",
        "",
        f"> {demo['demo']['request']}",
        "",
        "## Shot List",
        "",
        "1. Discord voice: join the voice channel and say the prompt above.",
        "2. Reflex response: show Hermes acknowledging the budget immediately and stating that billable actions require approval.",
        "3. Oracle path: show Nemotron 3 Super selected through Hermes' normal `/model` flow or visible in the generated dashboard; use Ultra only as the hosted fallback.",
        "4. NemoClaw boundary: show `nemoclaw-action-packet.json` or the dashboard NemoClaw section before any billable/network-capable action.",
        "5. Stripe Skills: show queued Stripe Projects VoIP provisioning and Link-gated service-credit spend in `stripe-actions-dry-run.sh` or the dashboard approval queue.",
        "6. Phone handoff: show `phone-context.json` and narrate that the same Discord context is preserved for the outbound call.",
        "7. Spark story: show or state that the target appliance is one DGX Spark running the reflex, speech stack, and preferred local model path.",
        "",
        "## If Live Voice Is Not Ready",
        "",
        "Use the generated package as the demo evidence instead of pretending the system is live:",
        "",
        "- `operator-dashboard.html` for the recording surface",
        "- `voiceops-demo.md` for the concise story",
        "- `nemoclaw-action-packet.json` for the approval boundary",
        "- `stripe-actions-dry-run.sh` for non-mutating Stripe/Projects commands",
        "- `phone-context.json` for the handoff context",
        "- `milestone2-execution-plan.json` for receipts, credential references, approval gates, and rollback/deprovision notes",
        "- `audit-ledger.jsonl` for durable action evidence",
        "",
        "## Submission Checklist",
        "",
        "- Video is between 1 and 3 minutes.",
        "- Video tags `@NousResearch` in the tweet.",
        "- Short writeup mentions DGX Spark, Discord voice, Nemotron 3 Super, Ultra hosted fallback, NemoClaw, and Stripe Skills.",
        "- Submit the tweet link in the Nous Discord submissions channel.",
        "- Fill out the hackathon Typeform submission.",
        "- Confirm no API keys, phone numbers, tokens, or account secrets are visible.",
        "",
        "## Tweet Draft",
        "",
        "Hermes VoiceOps: I give a Spark-powered Hermes agent a budget over Discord voice, it builds a NemoClaw-safe plan, queues Stripe Skills provisioning for VoIP, and preserves context for a phone handoff. Local-first household/business ops, with spend gated by approval. @NousResearch",
        "",
    ]
    return "\n".join(lines)


def _readiness_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# VoiceOps Recording Readiness",
        "",
        f"- Ready for recording: {'yes' if report['ready_for_recording'] else 'no'}",
        f"- Required failures: {', '.join(report['required_failures']) if report['required_failures'] else 'none'}",
        "",
        "## Env Sources",
        "",
    ]
    for source in report.get("env_sources") or []:
        if source.get("kind") == "process":
            lines.append(f"- process env: {source.get('key_count', 0)} keys visible")
        else:
            state = "loaded" if source.get("loaded") else "missing or empty"
            lines.append(f"- {source.get('path')}: {state} ({source.get('key_count', 0)} keys)")
    lines.extend([
        "",
        "## Checks",
        "",
    ])
    for check in report["checks"]:
        required = "required" if check["required_for_video"] else "optional"
        lines.extend(
            [
                f"### {check['check_id']}",
                "",
                f"- Status: {check['status']}",
                f"- Scope: {required}",
                f"- Detail: {check['detail']}",
                f"- Next step: {check['next_step']}",
                "",
            ]
        )
    return "\n".join(lines)


def _status_class(status: Any) -> str:
    normalized = str(status or "").strip().lower().replace("_", "-")
    if normalized in {"pass", "ready", "queued", "demo-call-queued", "implemented-on-branch", "repo-supported"}:
        return "ok"
    if normalized in {"warn", "held-budget", "queued-requires-approval"}:
        return "warn"
    if normalized in {"fail", "failed"}:
        return "fail"
    return "neutral"


def _dashboard_html(demo: dict[str, Any], readiness: dict[str, Any]) -> str:
    nemoclaw = _nemoclaw_action_packet(demo)
    phone_context = _phone_context_packet(demo)
    operator_state = _operator_state_packet(demo, readiness)
    budget_status = operator_state["budget_status"]
    voice_surface = operator_state["active_voice_surface"]
    approval_cents = demo["totals"]["approval_required_cents"]
    limit_cents = max(int(demo["spend_policy"]["limit_cents"] or 0), 1)
    approval_percent = min(100, int(round((approval_cents / limit_cents) * 100)))
    readiness_label = "Ready" if readiness["ready_for_recording"] else "Needs setup"
    held_actions = [action for action in demo["ops_actions"] if action["status"] == "held-budget"]
    pending_rows = []
    for approval in operator_state["pending_approvals"]:
        pending_rows.append(
            "<tr>"
            f"<td>{_h(approval['action_id'])}</td>"
            f"<td>{_h(approval['provider'])}</td>"
            f"<td>{_h(_dollars(approval['estimated_cents']))}</td>"
            f"<td>{_h(approval['title'])}</td>"
            "</tr>"
        )
    if not pending_rows:
        pending_rows.append("<tr><td colspan=\"4\">No queued approvals.</td></tr>")
    action_rows = []
    for action in demo["ops_actions"]:
        approval = "approval required" if action["requires_approval"] else "no approval"
        action_rows.append(
            "<tr>"
            f"<td>{_h(action['action_id'])}</td>"
            f"<td>{_h(action['provider'])}</td>"
            f"<td><span class=\"pill {_status_class(action['status'])}\">{_h(action['status'])}</span></td>"
            f"<td>{_h(_dollars(action['estimated_cents']))}</td>"
            f"<td>{_h(approval)}</td>"
            "</tr>"
        )
    audit_rows = []
    for event in operator_state["recent_audit_events"]:
        audit_rows.append(
            "<tr>"
            f"<td>{_h(event['event_id'])}</td>"
            f"<td>{_h(event['action'])}</td>"
            f"<td><span class=\"pill {_status_class(event['status'])}\">{_h(event['status'])}</span></td>"
            f"<td>{_h(_dollars(event['amount_cents']))}</td>"
            f"<td>{_h(event['evidence'])}</td>"
            "</tr>"
        )
    planned_service_rows = []
    for service in operator_state["planned_services"]:
        planned_service_rows.append(
            "<tr>"
            f"<td>{_h(service['service_id'])}</td>"
            f"<td>{_h(service['provider'])}</td>"
            f"<td><span class=\"pill {_status_class(service['status'])}\">{_h(service['status'])}</span></td>"
            f"<td>{_h(service['name'])}</td>"
            "</tr>"
        )
    provisioned_service_rows = []
    for service in operator_state["provisioned_services"]:
        provisioned_service_rows.append(
            "<tr>"
            f"<td>{_h(service['service_id'])}</td>"
            f"<td>{_h(service['provider'])}</td>"
            f"<td><span class=\"pill {_status_class(service['status'])}\">{_h(service['status'])}</span></td>"
            f"<td>{_h(service['name'])}</td>"
            "</tr>"
        )
    readiness_items = []
    for check in readiness["checks"]:
        required = "required" if check["required_for_video"] else "optional"
        readiness_items.append(
            "<li>"
            f"<span class=\"pill {_status_class(check['status'])}\">{_h(check['status'])}</span>"
            f"<strong>{_h(check['check_id'])}</strong>"
            f"<small>{_h(required)} - {_h(check['detail'])}</small>"
            "</li>"
        )
    guardrail_items = "".join(f"<li>{_h(item)}</li>" for item in nemoclaw["blocked_capabilities"])
    surfaces = "".join(
        "<li>"
        f"<span>{_h(surface['channel'])}</span>"
        f"<strong>{_h(surface['role'])}</strong>"
        f"<small>{_h(surface['status'])}</small>"
        "</li>"
        for surface in demo["voice_surfaces"]
    )
    held_action_text = ", ".join(action["action_id"] for action in held_actions) if held_actions else "none"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_h(demo['demo']['name'])}</title>
  <style>
    :root {{
      --bg: #f7f8fb;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #5f6b7a;
      --line: #d9e0ea;
      --green: #0f7b5f;
      --green-bg: #dff5ec;
      --amber: #9a5b00;
      --amber-bg: #fff1d2;
      --red: #a62f2f;
      --red-bg: #ffe0df;
      --blue: #2458a6;
      --blue-bg: #e6eefc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    main {{
      width: min(1280px, calc(100vw - 48px));
      margin: 0 auto;
      padding: 28px 0 36px;
    }}
    header {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 20px;
      align-items: end;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
      margin-bottom: 18px;
    }}
    h1 {{ margin: 0; font-size: 30px; line-height: 1.1; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.45; }}
    .grid {{
      display: grid;
      grid-template-columns: 1.45fr 0.9fr;
      gap: 18px;
      align-items: start;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .panel, .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .metric small, li small {{
      display: block;
      color: var(--muted);
      margin-top: 5px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }}
    .metric strong {{ display: block; font-size: 22px; margin-top: 5px; }}
    .stack {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      overflow-wrap: anywhere;
    }}
    th {{ color: var(--muted); font-weight: 600; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      border-radius: 6px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .ok {{ background: var(--green-bg); color: var(--green); }}
    .warn {{ background: var(--amber-bg); color: var(--amber); }}
    .fail {{ background: var(--red-bg); color: var(--red); }}
    .neutral {{ background: var(--blue-bg); color: var(--blue); }}
    ul {{
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 10px;
    }}
    li {{
      border-top: 1px solid var(--line);
      padding-top: 10px;
      overflow-wrap: anywhere;
    }}
    li:first-child {{ border-top: 0; padding-top: 0; }}
    li strong {{ display: block; margin-top: 5px; }}
    .bar {{
      height: 10px;
      background: #edf1f6;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 10px;
    }}
    .bar span {{
      display: block;
      height: 100%;
      width: {approval_percent}%;
      background: var(--green);
    }}
    .side {{ display: grid; gap: 18px; }}
    .section-gap {{ display: grid; gap: 18px; }}
    @media (max-width: 860px) {{
      main {{ width: min(100vw - 28px, 760px); }}
      header, .grid, .metrics, .stack {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>{_h(demo['demo']['name'])}</h1>
        <p>{_h(demo['demo']['submission_theme'])}</p>
      </div>
      <span class="pill {_status_class('pass' if readiness['ready_for_recording'] else 'warn')}">{_h(readiness_label)}</span>
    </header>

    <section class="metrics">
      <div class="metric"><small>Budget</small><strong>{_h(_dollars(demo['spend_policy']['limit_cents']))}</strong></div>
      <div class="metric"><small>Approval queued</small><strong>{_h(_dollars(approval_cents))}</strong><div class="bar"><span></span></div></div>
      <div class="metric"><small>Required failures</small><strong>{_h(len(readiness['required_failures']))}</strong></div>
      <div class="metric"><small>Audit events</small><strong>{_h(len(demo['audit_events']))}</strong></div>
    </section>

    <section class="stack">
      <div class="panel"><h2>Nemotron 3 Super</h2><p>{_h(demo['sponsor_stack']['nemotron_3_super']['role'])}</p></div>
      <div class="panel"><h2>NemoClaw</h2><p>{_h(demo['sponsor_stack']['nemoclaw']['demo_use'])}</p></div>
      <div class="panel"><h2>Stripe Skills</h2><p>{_h(demo['sponsor_stack']['stripe_skills']['demo_use'])}</p></div>
    </section>

    <section class="grid">
      <div class="section-gap">
        <div class="panel">
          <h2>Discord Voice Request</h2>
          <p>{_h(demo['demo']['request'])}</p>
        </div>
        <div class="panel">
          <h2>Current Mode</h2>
          <table>
            <tbody>
              <tr><th>NemoClaw Mode</th><td>{_h(nemoclaw['mode'])}</td></tr>
              <tr><th>Operator State</th><td>{_h(operator_state['current_mode'])}</td></tr>
              <tr><th>Active Voice Surface</th><td>{_h(voice_surface['surface_id'])}</td></tr>
              <tr><th>Target Surface</th><td>{_h(voice_surface['fallback_surface_id'])}</td></tr>
              <tr><th>Fallback Reason</th><td>{_h(voice_surface['fallback_reason'])}</td></tr>
            </tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Budget Status</h2>
          <table>
            <tbody>
              <tr><th>Limit</th><td>{_h(_dollars(budget_status['limit_cents']))}</td></tr>
              <tr><th>Approval threshold</th><td>{_h(_dollars(budget_status['approval_required_over_cents']))}</td></tr>
              <tr><th>Queued approval spend</th><td>{_h(_dollars(budget_status['queued_cents']))}</td></tr>
              <tr><th>Ready or queued spend</th><td>{_h(_dollars(budget_status['ready_or_queued_cents']))}</td></tr>
              <tr><th>Remaining before approval</th><td>{_h(_dollars(budget_status['remaining_before_approval_cents']))}</td></tr>
              <tr><th>Held over budget</th><td>{_h(_dollars(budget_status['held_budget_cents']))} ({_h(held_action_text)})</td></tr>
            </tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Pending Approvals</h2>
          <table>
            <thead><tr><th>Action</th><th>Provider</th><th>Spend</th><th>Purpose</th></tr></thead>
            <tbody>{''.join(pending_rows)}</tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Action Ledger</h2>
          <table>
            <thead><tr><th>Action</th><th>Provider</th><th>Status</th><th>Spend</th><th>Gate</th></tr></thead>
            <tbody>{''.join(action_rows)}</tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Voice Surfaces</h2>
          <ul>{surfaces}</ul>
        </div>
        <div class="panel">
          <h2>Recent Audit Events</h2>
          <table>
            <thead><tr><th>Event</th><th>Action</th><th>Status</th><th>Amount</th><th>Evidence</th></tr></thead>
            <tbody>{''.join(audit_rows)}</tbody>
          </table>
        </div>
      </div>

      <aside class="side">
        <div class="panel">
          <h2>Planned Services</h2>
          <table>
            <thead><tr><th>Action</th><th>Provider</th><th>Status</th><th>Purpose</th></tr></thead>
            <tbody>{''.join(planned_service_rows)}</tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Provisioned Services</h2>
          <table>
            <thead><tr><th>Service</th><th>Provider</th><th>Status</th><th>Capability</th></tr></thead>
            <tbody>{''.join(provisioned_service_rows)}</tbody>
          </table>
        </div>
        <div class="panel">
          <h2>Readiness</h2>
          <ul>{''.join(readiness_items)}</ul>
        </div>
        <div class="panel">
          <h2>NemoClaw Blocks</h2>
          <ul>{guardrail_items}</ul>
        </div>
        <div class="panel">
          <h2>Phone Handoff</h2>
          <p>{_h(phone_context['spoken_opening'])}</p>
          <p><a href="milestone2-execution-plan.json">Post-approval execution packet</a> records the approval gates, receipt schema, credential-location schema, and rollback notes.</p>
        </div>
        <div class="panel">
          <h2>Upcoming Tasks</h2>
          <p>Persistent household and business task state is bundled in <a href="operator-state.json">operator-state.json</a> and can also be generated independently by the Milestone 5 operator-state command.</p>
        </div>
      </aside>
    </section>
  </main>
</body>
</html>
"""


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


def write_demo(
    output_dir: Path,
    demo: dict[str, Any],
    *,
    readiness_env_files: Iterable[Path] = (),
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    readiness = build_readiness_report(demo, env_files=readiness_env_files)
    paths = {
        "json": output_dir / "voiceops-demo.json",
        "markdown": output_dir / "voiceops-demo.md",
        "audit_ledger": output_dir / "audit-ledger.jsonl",
        "demo_script": output_dir / "demo-script.md",
        "nemoclaw_packet": output_dir / "nemoclaw-action-packet.json",
        "phone_context": output_dir / "phone-context.json",
        "readiness_json": output_dir / "readiness-report.json",
        "readiness_markdown": output_dir / "readiness-report.md",
        "dashboard": output_dir / "operator-dashboard.html",
        "operator_state": output_dir / "operator-state.json",
        "operator_state_events": output_dir / "operator-state-events.jsonl",
        "milestone2_execution_plan": output_dir / "milestone2-execution-plan.json",
        "recording_runbook": output_dir / "recording-runbook.md",
        "submission_writeup": output_dir / "submission-writeup.md",
        "stripe_actions": output_dir / "stripe-actions-dry-run.sh",
    }
    _write_json(paths["json"], demo)
    paths["markdown"].write_text(_markdown(demo), encoding="utf-8")
    _write_jsonl(paths["audit_ledger"], demo["audit_events"])
    paths["demo_script"].write_text(_demo_script(demo), encoding="utf-8")
    _write_json(paths["nemoclaw_packet"], _nemoclaw_action_packet(demo))
    _write_json(paths["phone_context"], _phone_context_packet(demo))
    _write_json(paths["readiness_json"], readiness)
    paths["readiness_markdown"].write_text(_readiness_markdown(readiness), encoding="utf-8")
    _write_json(paths["milestone2_execution_plan"], build_milestone2_execution_plan(_demo_milestone2_report(demo, readiness)))
    operator_state = _operator_state_packet(demo, readiness)
    _write_json(paths["operator_state"], operator_state)
    _write_jsonl(paths["operator_state_events"], operator_state["recent_audit_events"])
    paths["dashboard"].write_text(_dashboard_html(demo, readiness), encoding="utf-8")
    paths["recording_runbook"].write_text(_recording_runbook(demo, readiness), encoding="utf-8")
    paths["submission_writeup"].write_text(_submission_writeup(demo), encoding="utf-8")
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
    parser.add_argument(
        "--active-model",
        dest="active_model",
        default="Nemotron 3 Super local on DGX Spark via Hermes /model",
        help="Hermes active model selected through /model.",
    )
    parser.add_argument("--reflex-model", default="Gemma 4 E2B audio-native reflex on Spark")
    parser.add_argument(
        "--hermes-home",
        type=Path,
        default=Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")),
        help="Hermes home whose .env should be considered for readiness without printing secrets.",
    )
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        type=Path,
        help="Additional .env file to include in the readiness presence check.",
    )
    parser.add_argument(
        "--no-default-env-files",
        action="store_true",
        help="Only use process env and explicit --env-file values for readiness.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.budget_cents < 0:
        raise SystemExit("--budget-cents must be non-negative")
    if args.approval_required_over_cents < 0:
        raise SystemExit("--approval-required-over-cents must be non-negative")
    demo = build_demo(args)
    env_files = [] if args.no_default_env_files else _default_readiness_env_files(args.hermes_home)
    env_files.extend(args.env_file)
    paths = write_demo(args.output_dir, demo, readiness_env_files=env_files)
    print(json.dumps({"ok": True, "output_dir": str(args.output_dir), "artifacts": paths}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
