#!/usr/bin/env python3
"""Probe VoiceOps provisioning readiness without mutating external systems.

This script checks local CLI and configuration readiness for the dry-run path
that would eventually cover Stripe Projects, Stripe Link, MPP, and phone
handoff. It never runs spend, provisioning, credential retrieval, or outbound
call commands. Active command probes are limited to bounded version/help calls.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


DEFAULT_OUTPUT_DIR = Path("artifacts/voiceops-provisioning/current")
FORBIDDEN_ENV_ROOT = Path("/Users/jethac/.hermes/hermes-agent").expanduser()

BLOCKED_CAPABILITIES = [
    "live_spend",
    "provider_provisioning",
    "credential_retrieval",
    "outbound_phone_calls",
    "account_mutation",
    "network_tunnels",
]

MUTATING_COMMAND_PATTERNS = [
    "projects add",
    "spend-request create",
    "payment",
    "charge",
    "checkout",
    "provision",
    "buy",
    "purchase",
    "call create",
    "calls create",
    "messages create",
    "login",
    "whoami",
    "credential",
    "secret",
    "token",
]

SAFE_PROBE_ARGS = {"--version", "-v", "version", "--help", "-h", "help"}

SECRET_KEY_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|KEY|PASSWORD|AUTH)[A-Z0-9_]*)\s*=\s*([^\s,;]+)"
)
SECRET_VALUE_RE = re.compile(
    r"(?i)\b(?:sk|pk|rk|whsec|AC|SG|xox[baprs]|gh[pousr])[_-]?[A-Za-z0-9][A-Za-z0-9_\-]{8,}\b"
)
BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{8,}")
PHONE_RE = re.compile(r"(?<!\d)\+?[1-9]\d[\d .()\-]{7,}\d(?!\d)")


@dataclass(frozen=True)
class CommandProbe:
    probe_id: str
    area: str
    argv: list[str]
    required: bool
    purpose: str


@dataclass(frozen=True)
class ReadinessCheck:
    check_id: str
    area: str
    status: str
    required: bool
    detail: str
    next_step: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


CommandRunner = Callable[[Sequence[str], int], CommandResult]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _redact(value: Any) -> str:
    text = str(value)
    text = SECRET_KEY_RE.sub(r"\1=<redacted>", text)
    text = BEARER_RE.sub("Bearer <redacted>", text)
    text = SECRET_VALUE_RE.sub("<redacted>", text)
    text = PHONE_RE.sub("<redacted-phone>", text)
    return text


def _excerpt(value: Any, limit: int = 240) -> str:
    text = _redact(value).replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _presence_env(env: Mapping[str, str]) -> dict[str, str]:
    return {key: "present" for key, value in env.items() if str(value or "").strip()}


def _env_present(env: Mapping[str, str], key: str) -> bool:
    return bool(str(env.get(key) or "").strip())


def _present_keys(env: Mapping[str, str], keys: Iterable[str]) -> list[str]:
    return sorted(key for key in keys if _env_present(env, key))


def _parse_env_file(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve(strict=False)
    if resolved == FORBIDDEN_ENV_ROOT or FORBIDDEN_ENV_ROOT in resolved.parents:
        raise ValueError(f"refusing to inspect forbidden Hermes worktree path: {resolved}")
    values: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                if key and value.strip():
                    values[key] = "present"
    except (FileNotFoundError, OSError):
        return {}
    return values


def _default_env_files() -> list[Path]:
    return [Path(__file__).resolve().parents[1] / ".env"]


def _merge_env_sources(env: Mapping[str, str], env_files: Iterable[Path]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    merged = _presence_env(env)
    sources: list[dict[str, Any]] = [{"kind": "process", "loaded": True, "key_count": len(merged)}]
    for path in env_files:
        parsed = _parse_env_file(path)
        merged.update(parsed)
        sources.append(
            {
                "kind": "env_file",
                "path": str(path),
                "exists": path.exists(),
                "loaded": bool(parsed),
                "key_count": len(parsed),
            }
        )
    return merged, sources


def _which_any(which: Callable[[str], str | None], commands: Iterable[str]) -> tuple[str | None, str | None]:
    for command in commands:
        path = which(command)
        if path:
            return command, path
    return None, None


def _validate_safe_probe_command(argv: Sequence[str]) -> None:
    if not argv:
        raise ValueError("empty probe command")
    joined = " ".join(argv).lower()
    for pattern in MUTATING_COMMAND_PATTERNS:
        if pattern in joined:
            raise ValueError(f"refusing mutating or credential-sensitive probe command: {joined}")
    if not any(arg in SAFE_PROBE_ARGS for arg in argv[1:]):
        raise ValueError(f"probe command must be version/help only: {joined}")


def _subprocess_runner(argv: Sequence[str], timeout_seconds: int) -> CommandResult:
    _validate_safe_probe_command(argv)
    with tempfile.TemporaryDirectory(prefix="voiceops-probe-home-") as home:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": home,
            "XDG_CONFIG_HOME": str(Path(home) / ".config"),
            "XDG_CACHE_HOME": str(Path(home) / ".cache"),
            "XDG_DATA_HOME": str(Path(home) / ".local" / "share"),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", ""),
            "STRIPE_CLI_TELEMETRY_OPTOUT": "1",
            "NO_COLOR": "1",
        }
        try:
            completed = subprocess.run(
                list(argv),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
                cwd=str(Path(__file__).resolve().parents[1]),
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                exit_code=124,
                stdout=exc.stdout if isinstance(exc.stdout, str) else "",
                stderr=exc.stderr if isinstance(exc.stderr, str) else "probe timed out",
                timed_out=True,
            )
        except OSError as exc:
            return CommandResult(exit_code=127, stderr=str(exc))
    return CommandResult(exit_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


def _command_manifest() -> list[CommandProbe]:
    return [
        CommandProbe(
            probe_id="stripe_cli_version",
            area="stripe_projects",
            argv=["stripe", "--version"],
            required=True,
            purpose="Confirm the Stripe CLI binary is callable without account lookup.",
        ),
        CommandProbe(
            probe_id="stripe_projects_help",
            area="stripe_projects",
            argv=["stripe", "projects", "--help"],
            required=True,
            purpose="Confirm the Projects subcommand/plugin is locally available without provisioning.",
        ),
        CommandProbe(
            probe_id="stripe_link_cli_version",
            area="stripe_link",
            argv=["link-cli", "--version"],
            required=True,
            purpose="Confirm the Stripe Link CLI binary is callable without creating spend requests.",
        ),
        CommandProbe(
            probe_id="mppx_version",
            area="mpp",
            argv=["mppx", "--version"],
            required=True,
            purpose="Confirm an MPP CLI is callable without paying a request or creating an account.",
        ),
        CommandProbe(
            probe_id="twilio_cli_version",
            area="phone_handoff",
            argv=["twilio", "--version"],
            required=False,
            purpose="Optionally confirm a phone provider CLI is callable without calls or messages.",
        ),
    ]


def _run_probe(
    probe: CommandProbe,
    *,
    which: Callable[[str], str | None],
    runner: CommandRunner,
    timeout_seconds: int,
    run_commands: bool,
) -> dict[str, Any]:
    _validate_safe_probe_command(probe.argv)
    executable = probe.argv[0]
    path = which(executable)
    result: dict[str, Any] = {
        "probe_id": probe.probe_id,
        "area": probe.area,
        "argv": probe.argv,
        "required": probe.required,
        "purpose": probe.purpose,
        "found": bool(path),
        "path": path,
        "executed": False,
        "status": "missing",
    }
    if not path:
        return result
    if not run_commands:
        result["status"] = "found"
        return result
    command_result = runner(probe.argv, timeout_seconds)
    result.update(
        {
            "executed": True,
            "exit_code": command_result.exit_code,
            "timed_out": command_result.timed_out,
            "stdout_excerpt": _excerpt(command_result.stdout),
            "stderr_excerpt": _excerpt(command_result.stderr),
            "status": "pass" if command_result.exit_code == 0 and not command_result.timed_out else "fail",
        }
    )
    return result


def _probe_by_id(command_results: list[dict[str, Any]], probe_id: str) -> dict[str, Any]:
    return next(item for item in command_results if item["probe_id"] == probe_id)


def _probe_ok(command_result: dict[str, Any], *, run_commands: bool) -> bool:
    if run_commands:
        return command_result.get("status") == "pass"
    return command_result.get("found") is True


def build_probe_report(
    *,
    env: Mapping[str, str] | None = None,
    env_files: Iterable[Path] | None = None,
    which: Callable[[str], str | None] = shutil.which,
    runner: CommandRunner = _subprocess_runner,
    run_commands: bool = False,
    timeout_seconds: int = 3,
) -> dict[str, Any]:
    env, env_sources = _merge_env_sources(os.environ if env is None else env, _default_env_files() if env_files is None else env_files)
    command_results = [
        _run_probe(
            probe,
            which=which,
            runner=runner,
            timeout_seconds=timeout_seconds,
            run_commands=run_commands,
        )
        for probe in _command_manifest()
    ]

    checks: list[ReadinessCheck] = []
    stripe_cli = _probe_by_id(command_results, "stripe_cli_version")
    stripe_projects = _probe_by_id(command_results, "stripe_projects_help")
    checks.append(
        ReadinessCheck(
            check_id="stripe_cli",
            area="stripe_projects",
            status="pass" if _probe_ok(stripe_cli, run_commands=run_commands) else "fail",
            required=True,
            detail="Stripe CLI is available for safe local probes" if stripe_cli["found"] else "stripe CLI not found on PATH",
            next_step="Install the Stripe CLI before attempting approved Projects provisioning.",
            evidence={"probe_id": "stripe_cli_version", "path": stripe_cli.get("path")},
        )
    )
    checks.append(
        ReadinessCheck(
            check_id="stripe_projects_cli",
            area="stripe_projects",
            status="pass" if _probe_ok(stripe_projects, run_commands=run_commands) else "fail",
            required=True,
            detail=(
                "Stripe Projects subcommand/help is available"
                if _probe_ok(stripe_projects, run_commands=run_commands)
                else "stripe projects help probe did not pass"
            ),
            next_step="Install or enable the Stripe Projects plugin/subcommand; do not run `stripe projects add` until approved.",
            evidence={"probe_id": "stripe_projects_help", "path": stripe_projects.get("path")},
        )
    )

    link_cli = _probe_by_id(command_results, "stripe_link_cli_version")
    npx_path = which("npx")
    checks.append(
        ReadinessCheck(
            check_id="stripe_link_cli",
            area="stripe_link",
            status="pass" if _probe_ok(link_cli, run_commands=run_commands) else "fail",
            required=True,
            detail=(
                "link-cli is available for safe local probes"
                if link_cli["found"]
                else "link-cli not found on PATH; npx is present but is not treated as ready because it may fetch packages"
                if npx_path
                else "link-cli not found on PATH"
            ),
            next_step="Install a pinned @stripe/link-cli binary before creating any approved spend request.",
            evidence={"probe_id": "stripe_link_cli_version", "path": link_cli.get("path"), "npx_path": npx_path},
        )
    )

    mpp_probe = _probe_by_id(command_results, "mppx_version")
    fallback_mpp_name, fallback_mpp_path = _which_any(which, ["mpp", "mpp-agent", "nemoclaw", "openshell"])
    mpp_ready = _probe_ok(mpp_probe, run_commands=run_commands) or bool(fallback_mpp_path)
    checks.append(
        ReadinessCheck(
            check_id="mpp_agent",
            area="mpp",
            status="pass" if mpp_ready else "fail",
            required=True,
            detail=(
                "mppx is available for safe local probes"
                if _probe_ok(mpp_probe, run_commands=run_commands)
                else f"{fallback_mpp_name} found as an MPP/sandbox boundary fallback"
                if fallback_mpp_path
                else "no mppx, mpp, mpp-agent, nemoclaw, or openshell command found"
            ),
            next_step="Install the MPP/NemoClaw boundary CLI before approving network-capable provisioning actions.",
            evidence={"probe_id": "mppx_version", "path": mpp_probe.get("path"), "fallback_path": fallback_mpp_path},
        )
    )

    phone_target_keys = [
        "VOICEOPS_DEMO_PHONE_NUMBER",
        "TWILIO_PHONE_NUMBER",
        "VAPI_PHONE_NUMBER_ID",
        "BLAND_PHONE_NUMBER",
    ]
    phone_provider_keys = [
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "TWILIO_PHONE_NUMBER_SID",
        "VAPI_API_KEY",
        "VAPI_PHONE_NUMBER_ID",
        "BLAND_API_KEY",
    ]
    target_present = _present_keys(env, phone_target_keys)
    provider_present = _present_keys(env, phone_provider_keys)
    provider_name, provider_path = _which_any(which, ["twilio", "vapi", "bland"])
    checks.append(
        ReadinessCheck(
            check_id="phone_target",
            area="phone_handoff",
            status="pass" if target_present else "fail",
            required=True,
            detail="phone handoff target configuration is present" if target_present else "no phone handoff target env key is present",
            next_step="Set a phone target env key before any approved outbound handoff.",
            evidence={"present_env_keys": target_present},
        )
    )
    checks.append(
        ReadinessCheck(
            check_id="phone_provider",
            area="phone_handoff",
            status="pass" if (provider_present or provider_path) else "fail",
            required=True,
            detail=(
                "phone provider env/tooling is present"
                if (provider_present or provider_path)
                else "no phone provider env key or provider CLI found"
            ),
            next_step="Configure a phone provider such as Twilio, Vapi, or Bland before an approved live handoff.",
            evidence={"present_env_keys": provider_present, "provider_cli": provider_name, "provider_cli_path": provider_path},
        )
    )

    check_dicts = [asdict(check) for check in checks]
    required_failures = [check["check_id"] for check in check_dicts if check["required"] and check["status"] != "pass"]
    area_status: dict[str, str] = {}
    for area in sorted({check["area"] for check in check_dicts}):
        area_checks = [check for check in check_dicts if check["area"] == area]
        area_status[area] = "pass" if all(check["status"] == "pass" for check in area_checks if check["required"]) else "fail"
    return {
        "generated_at": _utc_now(),
        "probe": {
            "name": "voiceops_provisioning_readiness",
            "non_mutating": True,
            "bounded": True,
            "run_commands": run_commands,
            "timeout_seconds": timeout_seconds,
            "active_probe_policy": "version_help_only",
            "blocked_capabilities": BLOCKED_CAPABILITIES,
        },
        "ready": not required_failures,
        "required_failures": required_failures,
        "area_status": area_status,
        "env_sources": env_sources,
        "checks": check_dicts,
        "command_probes": command_results,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# VoiceOps Provisioning Readiness Probe",
        "",
        f"- Ready: {'yes' if report['ready'] else 'no'}",
        "- Mode: non-mutating, bounded, PATH/env presence by default; version/help probes only when explicitly enabled",
        f"- Required failures: {', '.join(report['required_failures']) if report['required_failures'] else 'none'}",
        "",
        "## Safety Boundary",
        "",
    ]
    for capability in report["probe"]["blocked_capabilities"]:
        lines.append(f"- Blocks {capability}")
    lines.extend(["", "## Areas", ""])
    for area, status in sorted(report["area_status"].items()):
        lines.append(f"- {area}: {status}")
    lines.extend(["", "## Checks", ""])
    for check in report["checks"]:
        scope = "required" if check["required"] else "optional"
        lines.extend(
            [
                f"### {check['check_id']}",
                "",
                f"- Area: {check['area']}",
                f"- Status: {check['status']}",
                f"- Scope: {scope}",
                f"- Detail: {_redact(check['detail'])}",
                f"- Next step: {_redact(check['next_step'])}",
                "",
            ]
        )
    lines.extend(["## Command Probes", ""])
    for probe in report["command_probes"]:
        executed = "executed" if probe["executed"] else "not executed"
        lines.append(f"- {probe['probe_id']}: {probe['status']} ({executed}) `{' '.join(probe['argv'])}`")
    lines.append("")
    return "\n".join(lines)


def _safe_command_manifest_json() -> dict[str, Any]:
    return {
        "policy": "Default mode executes no vendor commands. If enabled, only isolated HOME version/help probes are allowed. Mutating, spend, provisioning, credential, and call commands are refused.",
        "blocked_patterns": MUTATING_COMMAND_PATTERNS,
        "commands": [asdict(probe) for probe in _command_manifest()],
    }


def build_milestone2_execution_plan(report: dict[str, Any]) -> dict[str, Any]:
    checks = {check["check_id"]: check for check in report["checks"]}
    gates = [
        {
            "gate_id": check_id,
            "area": checks[check_id]["area"],
            "status": checks[check_id]["status"],
            "required": checks[check_id]["required"],
            "detail": checks[check_id]["detail"],
            "next_step": checks[check_id]["next_step"],
        }
        for check_id in (
            "stripe_cli",
            "stripe_projects_cli",
            "stripe_link_cli",
            "mpp_agent",
            "phone_target",
            "phone_provider",
        )
    ]
    return {
        "generated_at": _utc_now(),
        "schema_version": "voiceops.milestone2.execution_plan.v1",
        "artifact_id": "voiceops-m2-execution-plan",
        "plan_id": "voiceops-m2-post-approval-execution-plan",
        "milestone": "milestone_2_real_spend_and_provisioning",
        "command": "uv run python scripts/voiceops_provisioning_probe.py --output-dir artifacts/voiceops-provisioning/current",
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "artifact_only": True,
        "mode": {
            "artifact_only": True,
            "headless": True,
            "bounded": True,
        },
        "source_readiness_artifact": "provisioning-readiness.json",
        "source_phone_context_artifact": "phone-context.json",
        "source_nemoclaw_artifact": "nemoclaw-action-packet.json",
        "safety": {
            "network_io": False,
            "env_secret_reads": False,
            "live_spend": False,
            "provider_provisioning": False,
            "credential_retrieval": False,
            "outbound_phone_calls": False,
            "account_mutation": False,
        },
        "blocked_capabilities": [
            "live_spend",
            "provider_provisioning",
            "credential_retrieval",
            "outbound_calls",
            "outbound_messages",
            "network_tunnels",
            "raw_card_data",
            "unapproved_recurring_charges",
        ],
        "preflight": {
            "command": "uv run python scripts/voiceops_provisioning_probe.py --output-dir artifacts/voiceops-provisioning/current",
            "readiness_artifact": "provisioning-readiness.json",
            "run_command_probes_default": False,
            "active_probe_policy": "version_help_only",
            "run_command_probes_does_not_grant_approval": True,
        },
        "demo_refs": {
            "voiceops_demo": "voiceops-demo.json",
            "nemoclaw_packet": "nemoclaw-action-packet.json",
            "phone_context": "phone-context.json",
            "audit_ledger": "audit-ledger.jsonl",
            "stripe_actions_dry_run": "stripe-actions-dry-run.sh",
        },
        "spend_policy": {
            "currency": "usd",
            "budget_cap_cents": 20_000,
            "approval_threshold_cents": 1_000,
            "queued_cents": 7_400,
            "held_cents": 0,
            "status": "no_live_spend_without_explicit_approval",
        },
        "readiness_gates": gates,
        "read_only_discovery": [
            {
                "step_id": "stripe-projects-catalog-discovery",
                "command": "stripe projects list --limit 10",
                "status": "not_executed",
                "requires": ["stripe_cli", "stripe_projects_cli", "mpp_agent"],
                "purpose": "Confirm available Projects catalog entries before choosing a VoIP provider.",
                "allowed_after": "operator opts into a read-only discovery run",
                "records_to": "audit-ledger.jsonl",
            },
            {
                "step_id": "stripe-link-auth-status",
                "command": "link-cli auth status",
                "status": "not_executed",
                "requires": ["stripe_link_cli", "mpp_agent"],
                "purpose": "Confirm Link approval capability without creating a spend request.",
                "allowed_after": "operator opts into a read-only auth-status run",
                "records_to": "audit-ledger.jsonl",
            },
        ],
        "approval_required_actions": [
            {
                "action_id": "provision-voip-provider",
                "provider": "stripe-projects",
                "command": "stripe projects add twilio/voice",
                "status": "blocked_until_explicit_approval",
                "requires": ["stripe_cli", "stripe_projects_cli", "mpp_agent"],
                "approval_artifact": "nemoclaw-action-packet.json",
                "expected_receipt_ref": "receipts.provision_voip_provider",
                "rollback_ref": "rollback_plan.deprovision_voip_provider",
            },
            {
                "action_id": "buy-service-credit",
                "provider": "stripe-link-cli",
                "command": "link-cli spend-request create --merchant-name ExampleOps --merchant-url https://example.invalid --amount 4900 --request-approval",
                "status": "blocked_until_explicit_approval",
                "requires": ["stripe_link_cli", "mpp_agent"],
                "approval_artifact": "nemoclaw-action-packet.json",
                "expected_receipt_ref": "receipts.buy_service_credit",
                "rollback_ref": "rollback_plan.refund_or_cancel_service_credit",
            },
            {
                "action_id": "call-user-phone",
                "provider": "voiceops-phone-bridge",
                "command": "queue outbound call --context phone-context.json",
                "status": "blocked_until_explicit_approval",
                "requires": ["phone_target", "phone_provider", "mpp_agent"],
                "approval_artifact": "phone-context.json",
                "expected_receipt_ref": "receipts.call_user_phone",
                "rollback_ref": "rollback_plan.cancel_or_end_phone_handoff",
            },
        ],
        "execution_steps": [
            {
                "step_id": "bind-spend-policy",
                "provider": "voiceops-policy",
                "purpose": "Bind spoken budget and approval threshold to the action packet.",
                "estimated_cents": 0,
                "requires_approval": False,
                "status": "planned_not_executed",
                "evidence_required": ["voiceops-demo.json", "audit-ledger.jsonl"],
                "rollback_or_deprovision_note": "Append a superseding policy event if the operator changes the budget.",
                "audit_event_id": "evt-001",
            },
            {
                "step_id": "provision-voip-provider",
                "provider": "stripe-projects",
                "purpose": "Provision a VoIP-capable provider account after approval.",
                "estimated_cents": 2500,
                "requires_approval": True,
                "status": "blocked_until_explicit_approval",
                "evidence_required": ["nemoclaw-action-packet.json", "provisioning-readiness.json"],
                "rollback_or_deprovision_note": "Disable calling, then deprovision or suspend provider resources after approval.",
                "audit_event_id": "evt-002",
            },
            {
                "step_id": "buy-service-credit",
                "provider": "stripe-link-cli",
                "purpose": "Request approved prepaid service credit through Link.",
                "estimated_cents": 4900,
                "requires_approval": True,
                "status": "blocked_until_explicit_approval",
                "evidence_required": ["nemoclaw-action-packet.json", "stripe-actions-dry-run.sh"],
                "rollback_or_deprovision_note": "Cancel pending spend request or record refund path.",
                "audit_event_id": "evt-003",
            },
            {
                "step_id": "persist-call-context",
                "provider": "hermes-audit-ledger",
                "purpose": "Persist Discord context for phone handoff.",
                "estimated_cents": 0,
                "requires_approval": False,
                "status": "planned_not_executed",
                "evidence_required": ["phone-context.json", "audit-ledger.jsonl"],
                "rollback_or_deprovision_note": "Append a corrected context packet if the handoff request changes.",
                "audit_event_id": "evt-004",
            },
            {
                "step_id": "call-user-phone",
                "provider": "voiceops-phone-bridge",
                "purpose": "Queue or place outbound call with preserved Discord context after approval.",
                "estimated_cents": 0,
                "requires_approval": True,
                "status": "blocked_until_explicit_approval",
                "evidence_required": ["phone-context.json", "channel-policy.json"],
                "rollback_or_deprovision_note": "Cancel queued call, or end active call and record call receipt.",
                "audit_event_id": "evt-005",
            },
            {
                "step_id": "publish-status",
                "provider": "hermes-gateway",
                "purpose": "Post redacted approval and handoff status to configured channels.",
                "estimated_cents": 0,
                "requires_approval": True,
                "status": "blocked_until_explicit_approval",
                "evidence_required": ["channel-policy.json", "audit-ledger.jsonl"],
                "rollback_or_deprovision_note": "Post a correction event and preserve the original audit id.",
                "audit_event_id": "evt-006",
            },
        ],
        "approval_gates": [
            {
                "gate_id": "stripe-projects-provisioning",
                "action_ids": ["provision-voip-provider"],
                "requires_human_approval": True,
                "reason": "Provider provisioning can create billable resources.",
            },
            {
                "gate_id": "stripe-link-spend",
                "action_ids": ["buy-service-credit"],
                "requires_human_approval": True,
                "reason": "Spend request can move money or reserve budget.",
            },
            {
                "gate_id": "phone-call-handoff",
                "action_ids": ["call-user-phone"],
                "requires_human_approval": True,
                "reason": "Outbound calls cross channel boundaries and may expose context.",
            },
            {
                "gate_id": "outbound-status-messages",
                "action_ids": ["publish-status"],
                "requires_human_approval": True,
                "reason": "External/customer-visible messages require channel-policy approval.",
            },
        ],
        "command_policy": {
            "default": "forbid_execution",
            "mutating_commands_are_display_only": True,
            "dry_run_shell_artifact": "stripe-actions-dry-run.sh",
            "forbidden_command_patterns": MUTATING_COMMAND_PATTERNS,
        },
        "redaction_policy": {
            "secrets": "redacted_or_absent",
            "phone_numbers": "redacted_or_hashed",
            "card_like_values": "redacted_or_absent",
        },
        "receipt_schema": {
            "required_fields": [
                "receipt_id",
                "action_id",
                "provider",
                "status",
                "approved_by",
                "approved_at",
                "executed_at",
                "amount_cents",
                "currency",
                "external_reference",
                "audit_event_id",
            ],
            "secret_policy": "receipts must contain references and redacted summaries only; never raw credentials, card data, tokens, or full phone numbers",
        },
        "credential_location_schema": {
            "required_fields": [
                "credential_ref_id",
                "provider",
                "service_id",
                "storage_backend",
                "secret_name_or_path",
                "created_by_action_id",
                "rotation_due",
            ],
            "allowed_storage_backends": ["hermes_secret_store", "system_keychain", "provider_managed"],
            "forbidden_fields": ["raw_secret", "raw_token", "raw_card_data", "raw_phone_number"],
        },
        "rollback_plan": {
            "deprovision_voip_provider": [
                "Record provider account/project id from receipt.",
                "Disable outbound calling before deleting resources.",
                "Delete or suspend VoIP provider project only after operator approval.",
                "Append rollback receipt and credential cleanup reference to audit-ledger.jsonl.",
            ],
            "refund_or_cancel_service_credit": [
                "Record Link spend request id and merchant reference.",
                "Cancel pending request or record refund path if already captured.",
                "Append refund/cancel status to audit-ledger.jsonl.",
            ],
            "cancel_or_end_phone_handoff": [
                "Cancel queued call if not started.",
                "If connected, end the call and preserve the call receipt id.",
                "Post redacted status to Discord/WhatsApp according to channel policy.",
            ],
        },
        "audit_requirements": [
            "append a child audit event before every approved external action",
            "append result, receipt reference, credential location reference, and rollback reference after every action",
            "preserve source Discord audit ids when handing context to phone or WhatsApp",
            "mark skipped, denied, or held actions explicitly rather than implying execution",
        ],
    }


def _execution_plan_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# VoiceOps Milestone 2 Execution Plan",
        "",
        f"- Plan ID: {plan['plan_id']}",
        f"- Schema: {plan['schema_version']}",
        "- Mode: artifact-only; no live spend, provisioning, credential retrieval, or outbound calls",
        f"- Phone context: `{plan['source_phone_context_artifact']}`",
        f"- Readiness source: `{plan['source_readiness_artifact']}`",
        f"- Budget cap: {plan['spend_policy']['currency']} {plan['spend_policy']['budget_cap_cents'] / 100:.2f}",
        f"- Queued spend: {plan['spend_policy']['currency']} {plan['spend_policy']['queued_cents'] / 100:.2f}",
        "",
        "## Demo References",
        "",
    ]
    for label, artifact in sorted(plan["demo_refs"].items()):
        lines.append(f"- {label}: `{artifact}`")
    lines.extend([
        "",
        "## Readiness Gates",
        "",
    ])
    for gate in plan["readiness_gates"]:
        lines.append(f"- {gate['gate_id']}: {gate['status']} ({gate['area']})")
    lines.extend(["", "## Read-Only Discovery", ""])
    for step in plan["read_only_discovery"]:
        lines.extend(
            [
                f"### {step['step_id']}",
                "",
                f"- Status: {step['status']}",
                f"- Command: `{step['command']}`",
                f"- Allowed after: {step['allowed_after']}",
                "",
            ]
        )
    lines.extend(["## Approval-Required Actions", ""])
    for action in plan["approval_required_actions"]:
        lines.extend(
            [
                f"### {action['action_id']}",
                "",
                f"- Provider: {action['provider']}",
                f"- Status: {action['status']}",
                f"- Command: `{action['command']}`",
                f"- Approval artifact: `{action['approval_artifact']}`",
                f"- Receipt ref: `{action['expected_receipt_ref']}`",
                f"- Rollback ref: `{action['rollback_ref']}`",
                "",
            ]
        )
    lines.extend(["## Execution Steps", ""])
    for step in plan["execution_steps"]:
        lines.extend(
            [
                f"### {step['step_id']}",
                "",
                f"- Provider: {step['provider']}",
                f"- Status: {step['status']}",
                f"- Requires approval: {'yes' if step['requires_approval'] else 'no'}",
                f"- Evidence: {', '.join(step['evidence_required'])}",
                f"- Rollback/deprovision: {step['rollback_or_deprovision_note']}",
                "",
            ]
        )
    lines.extend(["## Approval Gates", ""])
    for gate in plan["approval_gates"]:
        lines.extend(
            [
                f"### {gate['gate_id']}",
                "",
                f"- Actions: {', '.join(gate['action_ids'])}",
                f"- Requires human approval: {'yes' if gate['requires_human_approval'] else 'no'}",
                f"- Reason: {gate['reason']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Command Policy",
            "",
            f"- Default: {plan['command_policy']['default']}",
            f"- Mutating commands display-only: {plan['command_policy']['mutating_commands_are_display_only']}",
            f"- Dry-run shell artifact: `{plan['command_policy']['dry_run_shell_artifact']}`",
            "",
        ]
    )
    lines.extend(
        [
            "## Receipt Schema",
            "",
            "- Required fields: " + ", ".join(plan["receipt_schema"]["required_fields"]),
            f"- Secret policy: {plan['receipt_schema']['secret_policy']}",
            "",
            "## Credential Location Schema",
            "",
            "- Required fields: " + ", ".join(plan["credential_location_schema"]["required_fields"]),
            "- Forbidden fields: " + ", ".join(plan["credential_location_schema"]["forbidden_fields"]),
            "",
            "## Rollback Plan",
            "",
        ]
    )
    for rollback_id, steps in plan["rollback_plan"].items():
        lines.append(f"### {rollback_id}")
        lines.extend(f"- {step}" for step in steps)
        lines.append("")
    lines.extend(["## Audit Requirements", ""])
    lines.extend(f"- {requirement}" for requirement in plan["audit_requirements"])
    lines.append("")
    return "\n".join(lines)


def write_probe_artifacts(output_dir: Path, report: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_plan = build_milestone2_execution_plan(report)
    paths = {
        "json": output_dir / "provisioning-readiness.json",
        "markdown": output_dir / "provisioning-readiness.md",
        "command_manifest": output_dir / "safe-command-manifest.json",
        "execution_plan_json": output_dir / "milestone2-execution-plan.json",
        "execution_plan_markdown": output_dir / "milestone2-execution-plan.md",
    }
    _write_json(paths["json"], report)
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    _write_json(paths["command_manifest"], _safe_command_manifest_json())
    _write_json(paths["execution_plan_json"], execution_plan)
    paths["execution_plan_markdown"].write_text(_execution_plan_markdown(execution_plan), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-file", action="append", default=None, type=Path, help="Env file to inspect for key presence only.")
    parser.add_argument("--timeout-seconds", type=int, default=3)
    parser.add_argument(
        "--run-command-probes",
        action="store_true",
        help="Opt in to isolated version/help subprocess probes. Default inspects PATH/env presence only.",
    )
    parser.add_argument(
        "--no-command-probes",
        action="store_false",
        dest="run_command_probes",
        help="Compatibility alias for the default: inspect PATH/env presence only.",
    )
    parser.set_defaults(run_command_probes=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_probe_report(
        env_files=args.env_file,
        run_commands=args.run_command_probes,
        timeout_seconds=args.timeout_seconds,
    )
    paths = write_probe_artifacts(args.output_dir, report)
    print(
        json.dumps(
            {"ok": True, "ready": report["ready"], "output_dir": str(args.output_dir), "artifacts": paths},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
