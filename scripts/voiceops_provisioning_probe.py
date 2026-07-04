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
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence


DEFAULT_OUTPUT_DIR = Path("artifacts/voiceops-provisioning/current")
DEFAULT_COMMAND_PROBE_TIMEOUT_SECONDS = 3
DEFAULT_READONLY_DISCOVERY_TIMEOUT_SECONDS = 20
FORBIDDEN_ENV_ROOT = Path("/Users/jethac/.hermes/hermes-agent").expanduser()
PREFLIGHT_EVIDENCE_SCHEMA_VERSION = "voiceops.milestone2.preflight_evidence.v1"
PREFLIGHT_EVIDENCE_MANIFEST_SCHEMA_VERSION = "voiceops.milestone2.preflight_evidence_manifest.v1"
POST_APPROVAL_RECEIPTS_SCHEMA_VERSION = "voiceops.milestone2.post_approval_receipts.v1"
DEFAULT_SOURCE_VOICE_SESSION_ID = "discord:voiceops-demo:general"
DEFAULT_SOURCE_ORACLE_JOB_ID = "voice-oracle-voiceops-demo-001"
ACTION_PARENT_AUDIT_EVENT_IDS = {
    "provision-voip-provider": "evt-002",
    "buy-service-credit": "evt-003",
    "call-user-phone": "evt-005",
    "publish-status": "evt-006",
}
READ_ONLY_DISCOVERY_SCHEMA_VERSION = "voiceops.milestone2.read_only_discovery.v1"
READ_ONLY_DISCOVERY_MANIFEST_SCHEMA_VERSION = "voiceops.milestone2.read_only_discovery_manifest.v1"
NEMOCLAW_ACTION_PACKET_VALIDATION_SCHEMA_VERSION = "voiceops.nemoclaw_action_packet_validation.v1"
POST_APPROVAL_RECEIPT_STATUSES = {"executed", "failed", "held", "denied", "skipped", "rolled_back"}
POST_APPROVAL_NON_EXECUTED_STATUSES = {"held", "denied", "skipped"}
POST_APPROVAL_ATTEMPTED_EXECUTION_STATUSES = {"executed", "failed", "rolled_back"}
PREFLIGHT_REDACTED_SOURCE_SCHEMA_VERSION = "voiceops.milestone2.redacted_source_artifact.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
KAME_PROMOTED_AUTHORITIES = ("interpreter_promoted", "oracle_promoted")
KAME_REJECTED_AUTHORITIES = ("reflex_hypothesis", "auxiliary_hypothesis", "diagnostic_only", "hypothesis")
KAME_ACTION_PROMOTED_FIELDS = {
    "provision-voip-provider": ("user_request", "oracle_action_plan", "provider_selection"),
    "buy-service-credit": ("user_request", "oracle_action_plan", "spend_reason"),
    "call-user-phone": ("user_request", "oracle_action_plan", "phone_handoff_context"),
    "enable-whatsapp-egress": ("user_request", "oracle_action_plan", "channel_policy"),
    "publish-status": ("user_request", "oracle_action_plan", "channel_policy"),
}
TOOL_DISCLOSURE_TEST_REFS = (
    "tests/tools/test_tool_search.py::TestAssembly::test_defer_core_all_hides_core_behind_bridge",
    "tests/agent/test_realtime_voice_oracle.py::test_voice_oracle_applies_scoped_tool_search_override",
)
TOOL_DISCLOSURE_BRIDGE_TOOL_NAMES = ("tool_call", "tool_describe", "tool_search")

PHONE_TARGET_ENV_KEYS = [
    "VOICEOPS_DEMO_PHONE_NUMBER",
    "TWILIO_PHONE_NUMBER",
    "VAPI_PHONE_NUMBER_ID",
    "BLAND_PHONE_NUMBER",
]
PHONE_PROVIDER_ENV_KEYS = [
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_PHONE_NUMBER_SID",
    "VAPI_API_KEY",
    "VAPI_PHONE_NUMBER_ID",
    "BLAND_API_KEY",
]

BLOCKED_CAPABILITIES = [
    "live_spend",
    "provider_provisioning",
    "credential_retrieval",
    "outbound_phone_calls",
    "account_mutation",
    "network_tunnels",
]

PAYMENT_SKILL_CONTRACTS = {
    "stripe-projects": {
        "path": Path("optional-skills/payments/stripe-projects/SKILL.md"),
        "required_terms": [
            "stripe projects add",
            "twilio",
            "billing",
            ".env",
            "never commit",
        ],
    },
    "stripe-link-cli": {
        "path": Path("optional-skills/payments/stripe-link-cli/SKILL.md"),
        "required_terms": [
            "--request-approval",
            "http 402",
            "shared_payment_token",
            "do not print card details",
        ],
    },
    "mpp-agent": {
        "path": Path("optional-skills/payments/mpp-agent/SKILL.md"),
        "required_terms": [
            "http 402",
            "www-authenticate",
            "method=\"stripe\"",
            "wallet keys never enter agent context",
        ],
    },
}

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

SAFE_PROBE_ARGV_TUPLES = {
    ("stripe", "--version"),
    ("stripe", "projects", "--help"),
    ("link-cli", "--version"),
    ("mppx", "--version"),
    ("twilio", "--version"),
}
READONLY_DISCOVERY_ARGV_TUPLES = {
    ("stripe", "projects", "list", "--limit", "10"),
    ("link-cli", "auth", "status"),
}
SETUP_CLOSURE_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "stripe_cli": {
        "category": "local_tooling",
        "accepted_binaries": ["stripe"],
        "safe_probe_commands": [["stripe", "--version"]],
        "accepted_env_keys": [],
        "operator_action": "Install the Stripe CLI on PATH.",
        "proof": "A rerun records stripe_cli_version as found or pass.",
    },
    "stripe_projects_cli": {
        "category": "local_tooling",
        "accepted_binaries": ["stripe"],
        "safe_probe_commands": [["stripe", "projects", "--help"]],
        "accepted_env_keys": [],
        "operator_action": "Install or enable the Stripe Projects plugin/subcommand.",
        "proof": "A rerun records stripe_projects_help as found or pass.",
    },
    "stripe_link_cli": {
        "category": "local_tooling",
        "accepted_binaries": ["link-cli"],
        "safe_probe_commands": [["link-cli", "--version"]],
        "accepted_env_keys": [],
        "operator_action": "Install a pinned Stripe Link CLI binary on PATH.",
        "proof": "A rerun records stripe_link_cli_version as found or pass.",
    },
    "mpp_agent": {
        "category": "execution_boundary",
        "accepted_binaries": ["mppx", "mpp", "mpp-agent", "nemoclaw", "openshell"],
        "safe_probe_commands": [["mppx", "--version"]],
        "accepted_env_keys": [],
        "operator_action": "Install the MPP/NemoClaw boundary CLI on PATH.",
        "proof": "A rerun records mppx_version as found/pass or an accepted fallback boundary CLI.",
    },
    "stripe_skills_bundle": {
        "category": "local_skill_bundle",
        "accepted_binaries": [],
        "safe_probe_commands": [],
        "accepted_env_keys": [],
        "operator_action": "Restore or update the local optional Stripe Skills bundle with Projects, Link, and MPP safety contracts.",
        "proof": "A rerun records stripe_skills_bundle as pass with all required SKILL.md files and safety terms present.",
    },
    "phone_target": {
        "category": "configuration",
        "accepted_binaries": [],
        "safe_probe_commands": [],
        "accepted_env_keys": [
            *PHONE_TARGET_ENV_KEYS,
        ],
        "operator_action": "Set one phone handoff target env key in a repo-local env file or launch environment.",
        "proof": "A rerun records at least one accepted phone target key as present without emitting its value.",
    },
    "phone_provider": {
        "category": "configuration_or_tooling",
        "accepted_binaries": ["twilio", "vapi", "bland"],
        "safe_probe_commands": [["twilio", "--version"]],
        "accepted_env_keys": [
            *PHONE_PROVIDER_ENV_KEYS,
        ],
        "operator_action": "Configure a phone provider env key set or install an accepted provider CLI.",
        "proof": "A rerun records accepted provider env-key presence or provider CLI availability.",
    },
    "stripe_projects_account": {
        "category": "redacted_preflight_evidence",
        "accepted_binaries": [],
        "safe_probe_commands": [],
        "accepted_env_keys": [],
        "operator_action": "Fill the Stripe Projects account/capability section with a redacted source_artifact reference.",
        "proof": "A rerun loads preflight evidence with Stripe Projects source artifact, account ref, catalog timestamp, VoIP candidate, and approval-gated create capability.",
    },
    "stripe_link_approval_capability": {
        "category": "redacted_preflight_evidence",
        "accepted_binaries": [],
        "safe_probe_commands": [],
        "accepted_env_keys": [],
        "operator_action": "Fill the Stripe Link account and approval capability section with a redacted source_artifact reference.",
        "proof": "A rerun loads preflight evidence with Link source artifact, account ref, approval capability confirmed, currency, and budget coverage.",
    },
    "mpp_approval_boundary": {
        "category": "redacted_preflight_evidence",
        "accepted_binaries": [],
        "safe_probe_commands": [],
        "accepted_env_keys": [],
        "operator_action": "Fill the MPP/NemoClaw approval-boundary section with a redacted source_artifact reference.",
        "proof": "A rerun loads preflight evidence with MPP/NemoClaw source artifact, boundary tool, policy ref, and approval packet ref.",
    },
    "phone_provider_account": {
        "category": "redacted_preflight_evidence",
        "accepted_binaries": [],
        "safe_probe_commands": [],
        "accepted_env_keys": [],
        "operator_action": "Fill the phone provider account and target-reference section with a redacted source_artifact reference.",
        "proof": "A rerun loads preflight evidence with phone handoff source artifact, provider, provider account ref, and phone target ref.",
    },
    "credential_location_reference": {
        "category": "redacted_preflight_evidence",
        "accepted_binaries": [],
        "safe_probe_commands": [],
        "accepted_env_keys": [],
        "operator_action": "Fill the non-secret credential location reference with a redacted source_artifact reference.",
        "proof": "A rerun loads preflight evidence with a phone handoff source artifact and credential location ref, never a raw credential.",
    },
    "rollback_owner_refs": {
        "category": "redacted_preflight_evidence",
        "accepted_binaries": [],
        "safe_probe_commands": [],
        "accepted_env_keys": [],
        "operator_action": "Fill rollback owner refs for deprovision, refund/cancel, and call cancellation with a redacted source_artifact reference.",
        "proof": "A rerun loads preflight evidence with rollback source artifact and all rollback owner refs present.",
    },
}
PREFLIGHT_EVIDENCE_REQUIRED_DOT_PATHS = [
    "stripe_projects.source_artifact",
    "stripe_projects.source_artifact_kind",
    "stripe_projects.source_artifact_sha256",
    "stripe_projects.source_artifact_redacted_at",
    "stripe_projects.account_ref",
    "stripe_projects.projects_catalog_checked_at",
    "stripe_projects.voip_provider_candidate",
    "stripe_projects.can_create_project_after_approval",
    "stripe_link.source_artifact",
    "stripe_link.source_artifact_kind",
    "stripe_link.source_artifact_sha256",
    "stripe_link.source_artifact_redacted_at",
    "stripe_link.account_ref",
    "stripe_link.approval_capability_confirmed",
    "stripe_link.max_approved_cents",
    "stripe_link.currency",
    "mpp.source_artifact",
    "mpp.source_artifact_kind",
    "mpp.source_artifact_sha256",
    "mpp.source_artifact_redacted_at",
    "mpp.boundary_tool",
    "mpp.policy_ref",
    "mpp.approval_packet_ref",
    "phone_handoff.source_artifact",
    "phone_handoff.source_artifact_kind",
    "phone_handoff.source_artifact_sha256",
    "phone_handoff.source_artifact_redacted_at",
    "phone_handoff.provider",
    "phone_handoff.provider_account_ref",
    "phone_handoff.phone_target_ref",
    "phone_handoff.credential_location_ref",
    "rollback.source_artifact",
    "rollback.source_artifact_kind",
    "rollback.source_artifact_sha256",
    "rollback.source_artifact_redacted_at",
    "rollback.deprovision_owner",
    "rollback.refund_or_cancel_owner",
    "rollback.call_cancel_owner",
]
PREFLIGHT_EVIDENCE_SECTIONS = ("stripe_projects", "stripe_link", "mpp", "phone_handoff", "rollback")
PREFLIGHT_SOURCE_ARTIFACT_KIND = "redacted_setup_evidence"
PREFLIGHT_SOURCE_ARTIFACT_BASE_FIELD = "_voiceops_source_artifact_base_path"
PREFLIGHT_EXAMPLE_SOURCE_ARTIFACT_SHA256 = "0" * 64
PREFLIGHT_REFERENCE_FIELDS = {
    "stripe_projects.account_ref": "must be a non-empty reference string",
    "stripe_link.account_ref": "must be a non-empty reference string",
    "mpp.policy_ref": "must be a non-empty reference string",
    "phone_handoff.provider_account_ref": "must be a non-empty reference string",
    "phone_handoff.phone_target_ref": "must be a non-empty reference string",
    "phone_handoff.credential_location_ref": "must be a non-empty credential location reference string",
    "rollback.deprovision_owner": "must be a non-empty owner reference string",
    "rollback.refund_or_cancel_owner": "must be a non-empty owner reference string",
    "rollback.call_cancel_owner": "must be a non-empty owner reference string",
}
PREFLIGHT_ALLOWED_BOUNDARY_TOOLS = {"mppx", "mpp", "mpp-agent", "nemoclaw", "openshell"}
PREFLIGHT_ALLOWED_PHONE_PROVIDERS = {"twilio", "vapi", "bland"}
COLLECTOR_ATTESTATION_REQUIRED_FIELDS = (
    "collector_name",
    "collector_version",
    "run_id",
    "command_argv",
    "git_commit",
    "started_at",
    "finished_at",
    "raw_artifact_sha256",
    "redacted_artifact_sha256",
    "parent_manifest_sha256",
)
DEFAULT_VOIP_PROVIDER_CANDIDATE = "twilio/voice"
VOIP_PROVIDER_CANDIDATE_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]{0,80}$")

SECRET_KEY_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|KEY|PASSWORD|AUTH)[A-Z0-9_]*)\s*=\s*([^\s,;]+)"
)
SECRET_VALUE_RE = re.compile(
    r"(?i)\b(?:sk|pk|rk|whsec|AC|SG|xox[baprs]|gh[pousr])[_-]?[A-Za-z0-9][A-Za-z0-9_\-]{8,}\b"
)
PREFLIGHT_SECRET_VALUE_RE = re.compile(
    r"\b(?:sk|pk|rk|whsec|xox[baprs]|gh[pousr])[_-][A-Za-z0-9][A-Za-z0-9_\-]{8,}\b|"
    r"\bAC[A-Za-z0-9]{8,}\b|"
    r"\bSG[A-Za-z0-9]{8,}\b"
)
GENERIC_SECRET_REF_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:api[-_]?key|token|secret|password|auth)"
    r"(?:[\s:=/._-]+)[A-Za-z0-9][A-Za-z0-9._\-]{7,}"
)
SECRET_PATH_RE = re.compile(r"(?i)(?:^|[.\[\]_-])(?:credential|auth|token|secret|key|password)(?:$|[.\[\]_-])")
SENSITIVE_PATH_SECRET_VALUE_RE = re.compile(
    r"(?i)\b(?:live|prod|secret|token|api[-_]?key|auth)[._-][A-Za-z0-9][A-Za-z0-9._\-]{11,}\b|"
    r"\b[A-Za-z0-9+/=_-]{32,}\b"
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


def _skill_frontmatter_name(text: str) -> str | None:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, separator, value = line.partition(":")
        if separator and key.strip() == "name":
            return value.strip().strip("'\"")
    return None


def load_payment_skill_bundle_evidence(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    skill_reports: dict[str, dict[str, Any]] = {}
    missing_files: list[str] = []
    wrong_names: list[str] = []
    missing_terms: list[str] = []
    for skill_name, contract in PAYMENT_SKILL_CONTRACTS.items():
        relative_path = contract["path"]
        path = repo_root / relative_path
        exists = path.is_file()
        report = {
            "path": str(relative_path),
            "exists": exists,
            "frontmatter_name": None,
            "required_terms_present": {},
        }
        if not exists:
            missing_files.append(str(relative_path))
            skill_reports[skill_name] = report
            continue
        text = path.read_text(encoding="utf-8")
        frontmatter_name = _skill_frontmatter_name(text)
        report["frontmatter_name"] = frontmatter_name
        if frontmatter_name != skill_name:
            wrong_names.append(f"{skill_name}:frontmatter_name")
        normalized = text.lower()
        required_terms_present = {
            term: term.lower() in normalized
            for term in contract["required_terms"]
        }
        report["required_terms_present"] = required_terms_present
        missing_terms.extend(
            f"{skill_name}:{term}"
            for term, present in required_terms_present.items()
            if not present
        )
        skill_reports[skill_name] = report
    issues = [
        *[f"missing_skill_file:{path}" for path in missing_files],
        *[f"wrong_skill_frontmatter:{name}" for name in wrong_names],
        *[f"missing_skill_safety_term:{term}" for term in missing_terms],
    ]
    return {
        "schema_version": "voiceops.milestone2.payment_skill_bundle_evidence.v1",
        "repo_root": str(repo_root),
        "non_mutating": True,
        "network_io": False,
        "secret_values_emitted": False,
        "required_skills": sorted(PAYMENT_SKILL_CONTRACTS),
        "skills": skill_reports,
        "issues": sorted(issues),
        "status": "pass" if not issues else "fail",
    }


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


def build_preflight_evidence_template() -> dict[str, Any]:
    def section_source_template() -> dict[str, Any]:
        return {
            "source_artifact": None,
            "source_artifact_kind": PREFLIGHT_SOURCE_ARTIFACT_KIND,
            "source_artifact_sha256": None,
            "source_artifact_redacted_at": None,
            "collector_attestation": {
                "collector_name": None,
                "collector_version": None,
                "run_id": None,
                "command_argv": [],
                "git_commit": None,
                "started_at": None,
                "finished_at": None,
                "raw_artifact_sha256": None,
                "redacted_artifact_sha256": None,
                "parent_manifest_sha256": None,
            },
        }

    return {
        "schema_version": PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
        "redaction_policy": "references and aliases only; no raw secrets, cards, tokens, or full phone numbers",
        "stripe_projects": {
            **section_source_template(),
            "account_ref": None,
            "projects_catalog_checked_at": None,
            "voip_provider_candidate": "twilio/voice",
            "can_create_project_after_approval": False,
        },
        "stripe_link": {
            **section_source_template(),
            "account_ref": None,
            "approval_capability_confirmed": False,
            "max_approved_cents": 0,
            "currency": "usd",
        },
        "mpp": {
            **section_source_template(),
            "boundary_tool": None,
            "policy_ref": None,
            "approval_packet_ref": "nemoclaw-action-packet.json",
        },
        "phone_handoff": {
            **section_source_template(),
            "provider": None,
            "provider_account_ref": None,
            "phone_target_ref": None,
            "credential_location_ref": None,
        },
        "rollback": {
            **section_source_template(),
            "deprovision_owner": None,
            "refund_or_cancel_owner": None,
            "call_cancel_owner": None,
        },
    }


def _example_collector_attestation(*, section_name: str, redacted_sha256: str) -> dict[str, Any]:
    return {
        "example_only": True,
        "collector_name": "voiceops_provisioning_probe_manual_redacted_export",
        "collector_version": "example",
        "run_id": f"example-{section_name}-run",
        "command_argv": ["uv", "run", "python", "scripts/voiceops_provisioning_probe.py", "--output-dir", "artifacts/voiceops-provisioning/current"],
        "git_commit": "0" * 40,
        "started_at": "2026-06-29T00:00:00Z",
        "finished_at": "2026-06-29T00:00:01Z",
        "raw_artifact_sha256": "0" * 64,
        "redacted_artifact_sha256": redacted_sha256,
        "parent_manifest_sha256": "0" * 64,
    }


def build_preflight_evidence_example() -> dict[str, Any]:
    example = build_preflight_evidence_template()
    example["example_only"] = True
    example["redaction_policy"] = "example only; copy refs from real account/capability evidence and remove example_only before ingest"
    example["stripe_projects"].update(
        {
            "account_ref": "stripe-projects-account-ref-demo",
            "source_artifact": "artifacts/voiceops-provisioning/current/stripe-projects-catalog-redacted.json",
            "source_artifact_sha256": PREFLIGHT_EXAMPLE_SOURCE_ARTIFACT_SHA256,
            "source_artifact_redacted_at": "2026-06-29T00:00:00Z",
            "collector_attestation": _example_collector_attestation(
                section_name="stripe_projects",
                redacted_sha256=PREFLIGHT_EXAMPLE_SOURCE_ARTIFACT_SHA256,
            ),
            "projects_catalog_checked_at": "2026-06-29T00:00:00Z",
            "voip_provider_candidate": "twilio/voice",
            "can_create_project_after_approval": True,
        }
    )
    example["stripe_link"].update(
        {
            "account_ref": "stripe-link-account-ref-demo",
            "source_artifact": "artifacts/voiceops-provisioning/current/stripe-link-approval-redacted.json",
            "source_artifact_sha256": PREFLIGHT_EXAMPLE_SOURCE_ARTIFACT_SHA256,
            "source_artifact_redacted_at": "2026-06-29T00:00:00Z",
            "collector_attestation": _example_collector_attestation(
                section_name="stripe_link",
                redacted_sha256=PREFLIGHT_EXAMPLE_SOURCE_ARTIFACT_SHA256,
            ),
            "approval_capability_confirmed": True,
            "max_approved_cents": 20_000,
            "currency": "usd",
        }
    )
    example["mpp"].update(
        {
            "boundary_tool": "nemoclaw",
            "source_artifact": "artifacts/voiceops-provisioning/current/nemoclaw-boundary-redacted.json",
            "source_artifact_sha256": PREFLIGHT_EXAMPLE_SOURCE_ARTIFACT_SHA256,
            "source_artifact_redacted_at": "2026-06-29T00:00:00Z",
            "collector_attestation": _example_collector_attestation(
                section_name="mpp",
                redacted_sha256=PREFLIGHT_EXAMPLE_SOURCE_ARTIFACT_SHA256,
            ),
            "policy_ref": "voiceops-policy-ref-demo",
            "approval_packet_ref": "nemoclaw-action-packet.json",
        }
    )
    example["phone_handoff"].update(
        {
            "provider": "twilio",
            "source_artifact": "artifacts/voiceops-provisioning/current/phone-handoff-redacted.json",
            "source_artifact_sha256": PREFLIGHT_EXAMPLE_SOURCE_ARTIFACT_SHA256,
            "source_artifact_redacted_at": "2026-06-29T00:00:00Z",
            "collector_attestation": _example_collector_attestation(
                section_name="phone_handoff",
                redacted_sha256=PREFLIGHT_EXAMPLE_SOURCE_ARTIFACT_SHA256,
            ),
            "provider_account_ref": "twilio-account-ref-demo",
            "phone_target_ref": "operator-phone-ref-demo",
            "credential_location_ref": "1password://VoiceOps/Twilio Demo Credential Ref",
        }
    )
    example["rollback"].update(
        {
            "deprovision_owner": "operator-ref-demo",
            "source_artifact": "artifacts/voiceops-provisioning/current/rollback-owners-redacted.json",
            "source_artifact_sha256": PREFLIGHT_EXAMPLE_SOURCE_ARTIFACT_SHA256,
            "source_artifact_redacted_at": "2026-06-29T00:00:00Z",
            "collector_attestation": _example_collector_attestation(
                section_name="rollback",
                redacted_sha256=PREFLIGHT_EXAMPLE_SOURCE_ARTIFACT_SHA256,
            ),
            "refund_or_cancel_owner": "operator-ref-demo",
            "call_cancel_owner": "operator-ref-demo",
        }
    )
    return example


def build_preflight_evidence_manifest_example() -> dict[str, Any]:
    return {
        "schema_version": PREFLIGHT_EVIDENCE_MANIFEST_SCHEMA_VERSION,
        "example_only": True,
        "redaction_policy": "example only; reference real redacted section files and remove example_only before ingest",
        "reports": {
            "stripe_projects": "path/to/stripe-projects-evidence.json",
            "stripe_link": "path/to/stripe-link-evidence.json",
            "mpp": "path/to/nemoclaw-boundary-evidence.json",
            "phone_handoff": "path/to/phone-handoff-evidence.json",
            "rollback": "path/to/rollback-owner-evidence.json",
        },
        "notes": "Each referenced file may contain either the section object itself or an object with the matching section key.",
    }


def write_preflight_evidence_scaffold(output_dir: Path) -> dict[str, Path]:
    scaffold_dir = output_dir / "provisioning-preflight-scaffold"
    sections_dir = scaffold_dir / "sections"
    sources_dir = scaffold_dir / "sources"
    sections_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)

    evidence = build_preflight_evidence_example()
    evidence["redaction_policy"] = "example only; replace aliases with real redacted refs and remove every example_only marker"
    reports: dict[str, str] = {}
    paths: dict[str, Path] = {}
    source_names = {
        "stripe_projects": "stripe-projects-redacted-source.json",
        "stripe_link": "stripe-link-redacted-source.json",
        "mpp": "nemoclaw-boundary-redacted-source.json",
        "phone_handoff": "phone-handoff-redacted-source.json",
        "rollback": "rollback-owners-redacted-source.json",
    }
    section_names = {
        "stripe_projects": "stripe-projects-evidence.json",
        "stripe_link": "stripe-link-evidence.json",
        "mpp": "nemoclaw-boundary-evidence.json",
        "phone_handoff": "phone-handoff-evidence.json",
        "rollback": "rollback-owner-evidence.json",
    }
    for section_name in PREFLIGHT_EVIDENCE_SECTIONS:
        section = dict(evidence[section_name])
        source_path = sources_dir / source_names[section_name]
        source_payload = {
            "schema_version": PREFLIGHT_REDACTED_SOURCE_SCHEMA_VERSION,
            "example_only": True,
            "section": section_name,
            "redacted": True,
            "redaction_policy": "example only; references only, no raw secrets, tokens, cards, or full phone numbers",
            "summary": f"Replace this with real redacted {section_name} setup evidence.",
        }
        _write_json(source_path, source_payload)
        section["example_only"] = True
        section["source_artifact"] = f"sources/{source_path.name}"
        section["source_artifact_kind"] = PREFLIGHT_SOURCE_ARTIFACT_KIND
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        section["source_artifact_sha256"] = source_sha256
        section["source_artifact_redacted_at"] = "2026-06-29T00:00:00Z"
        section["collector_attestation"] = _example_collector_attestation(
            section_name=section_name,
            redacted_sha256=source_sha256,
        )
        section_path = sections_dir / section_names[section_name]
        _write_json(
            section_path,
            {
                "example_only": True,
                "redaction_policy": "example only; remove every example_only marker after replacing refs with real evidence",
                section_name: section,
            },
        )
        reports[section_name] = f"sections/{section_path.name}"
        paths[f"scaffold_{section_name}_section"] = section_path
        paths[f"scaffold_{section_name}_source"] = source_path

    manifest_path = scaffold_dir / "provisioning-preflight-evidence.manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": PREFLIGHT_EVIDENCE_MANIFEST_SCHEMA_VERSION,
            "example_only": True,
            "redaction_policy": "example only; this scaffold is rejected until all example_only markers are removed",
            "reports": reports,
            "notes": "Two-layer scaffold: section reports reference separate redacted source artifacts with matching SHA-256 fields.",
        },
    )
    paths["preflight_evidence_scaffold_manifest"] = manifest_path
    return paths


def _dot_get(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _field_present(payload: Mapping[str, Any], path: str) -> bool:
    value = _dot_get(payload, path)
    if isinstance(value, bool):
        return value is True
    if isinstance(value, int | float):
        return value > 0
    return bool(str(value or "").strip())


def _walk_strings(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_strings(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            yield from _walk_strings(child, child_prefix)
    elif isinstance(value, str):
        yield prefix, value


def _preflight_secret_issues(payload: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    for path, value in _walk_strings(payload):
        if (
            SECRET_KEY_RE.search(f"{path}={value}")
            or BEARER_RE.search(value)
            or PREFLIGHT_SECRET_VALUE_RE.search(value)
            or GENERIC_SECRET_REF_RE.search(value)
            or _secret_like_sensitive_path_value(path, value)
        ):
            issues.append(f"{path}: secret-like value")
        elif (
            not path.endswith("_checked_at")
            and not path.endswith("_redacted_at")
            and not path.endswith("_sha256")
            and ".collector_attestation." not in path
            and PHONE_RE.search(value)
        ):
            issues.append(f"{path}: phone-like value")
    return issues


def _secret_like_sensitive_path_value(path: str, value: str) -> bool:
    if path.endswith("_checked_at") or path.endswith("_redacted_at") or path.endswith("_sha256"):
        return False
    if not SECRET_PATH_RE.search(path):
        return False
    return bool(SENSITIVE_PATH_SECRET_VALUE_RE.search(value))


def _example_only_presence_issues(value: Any, prefix: str = "") -> list[str]:
    issues: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_prefix = f"{prefix}.{key_text}" if prefix else key_text
            if key_text == "example_only":
                if not prefix:
                    issues.append("example_only evidence is not accepted")
                elif prefix in PREFLIGHT_EVIDENCE_SECTIONS:
                    issues.append(f"{prefix}: example_only evidence is not accepted")
                else:
                    issues.append(f"{child_prefix}: example_only field is not accepted")
            issues.extend(_example_only_presence_issues(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            issues.extend(_example_only_presence_issues(child, child_prefix))
    return issues


def _parse_preflight_timestamp(value: Any) -> dt.datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def _timestamp_issues(payload: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    for path in ["stripe_projects.projects_catalog_checked_at"]:
        value = _dot_get(payload, path)
        if str(value or "").strip() and _parse_preflight_timestamp(value) is None:
            issues.append(f"{path}: invalid timestamp")
    for section_name in PREFLIGHT_EVIDENCE_SECTIONS:
        path = f"{section_name}.source_artifact_redacted_at"
        value = _dot_get(payload, path)
        if str(value or "").strip() and _parse_preflight_timestamp(value) is None:
            issues.append(f"{path}: invalid timestamp")
    return issues


def _preflight_field_value_issues(payload: Mapping[str, Any], evidence_path: Path) -> list[str]:
    issues: list[str] = []
    candidate = str(_dot_get(payload, "stripe_projects.voip_provider_candidate") or "").strip()
    if candidate and not VOIP_PROVIDER_CANDIDATE_RE.fullmatch(candidate):
        issues.append("stripe_projects.voip_provider_candidate: invalid provider candidate")
    for path, message in PREFLIGHT_REFERENCE_FIELDS.items():
        value = _dot_get(payload, path)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            issues.append(f"{path}: {message}")
    for path in (
        "stripe_projects.can_create_project_after_approval",
        "stripe_link.approval_capability_confirmed",
    ):
        value = _dot_get(payload, path)
        if value is not None and value is not True:
            issues.append(f"{path}: must be true")
    max_approved_cents = _dot_get(payload, "stripe_link.max_approved_cents")
    if max_approved_cents is not None and (
        not isinstance(max_approved_cents, int) or isinstance(max_approved_cents, bool) or max_approved_cents < 20_000
    ):
        issues.append("stripe_link.max_approved_cents: must be an integer >= 20000")
    currency = str(_dot_get(payload, "stripe_link.currency") or "").strip().lower()
    if currency and currency != "usd":
        issues.append("stripe_link.currency: must be usd")
    boundary_tool = str(_dot_get(payload, "mpp.boundary_tool") or "").strip().lower()
    if boundary_tool and boundary_tool not in PREFLIGHT_ALLOWED_BOUNDARY_TOOLS:
        issues.append("mpp.boundary_tool: must be one of mppx,mpp,mpp-agent,nemoclaw,openshell")
    provider = str(_dot_get(payload, "phone_handoff.provider") or "").strip().lower()
    if provider and provider not in PREFLIGHT_ALLOWED_PHONE_PROVIDERS:
        issues.append("phone_handoff.provider: must be one of twilio,vapi,bland")
    approval_packet_ref = _dot_get(payload, "mpp.approval_packet_ref")
    if approval_packet_ref is not None:
        if not isinstance(approval_packet_ref, str) or not approval_packet_ref.strip():
            issues.append("mpp.approval_packet_ref: must be a non-empty relative artifact reference")
        else:
            ref_issues = _relative_artifact_ref_issues(approval_packet_ref, base_path=evidence_path)
            issues.extend(f"mpp.approval_packet_ref:{issue}" for issue in ref_issues)
    return issues


def load_preflight_evidence(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "loaded": False,
            "path": None,
            "fields_present": [],
            "missing_fields": PREFLIGHT_EVIDENCE_REQUIRED_DOT_PATHS,
            "validation_issues": [],
            "voip_provider_candidate": None,
            "redaction_policy": "not_loaded",
        }
    resolved = path.expanduser().resolve(strict=False)
    if resolved == FORBIDDEN_ENV_ROOT or FORBIDDEN_ENV_ROOT in resolved.parents:
        raise ValueError(f"refusing to inspect forbidden Hermes worktree path: {resolved}")
    try:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "loaded": False,
            "path": str(path),
            "fields_present": [],
            "missing_fields": PREFLIGHT_EVIDENCE_REQUIRED_DOT_PATHS,
            "validation_issues": [
                "preflight evidence file not found",
                f"preflight evidence file not found at {resolved}",
            ],
            "voip_provider_candidate": None,
            "redaction_policy": "references_only",
        }
    except json.JSONDecodeError as exc:
        return {
            "loaded": False,
            "path": str(path),
            "fields_present": [],
            "missing_fields": PREFLIGHT_EVIDENCE_REQUIRED_DOT_PATHS,
            "validation_issues": [f"preflight evidence JSON parse failed: {exc.msg}"],
            "voip_provider_candidate": None,
            "redaction_policy": "references_only",
        }
    if not isinstance(raw_payload, Mapping):
        return {
            "loaded": False,
            "path": str(path),
            "fields_present": [],
            "missing_fields": PREFLIGHT_EVIDENCE_REQUIRED_DOT_PATHS,
            "validation_issues": ["preflight evidence root must be an object"],
            "voip_provider_candidate": None,
            "redaction_policy": "references_only",
        }
    raw_payload, manifest_issues = _expand_preflight_evidence_manifest(path, raw_payload)
    fields_present = [field for field in PREFLIGHT_EVIDENCE_REQUIRED_DOT_PATHS if _field_present(raw_payload, field)]
    missing_fields = [field for field in PREFLIGHT_EVIDENCE_REQUIRED_DOT_PATHS if field not in fields_present]
    validation_issues = [
        *manifest_issues,
        *_preflight_secret_issues(raw_payload),
        *_timestamp_issues(raw_payload),
        *_preflight_field_value_issues(raw_payload, path),
    ]
    if str(raw_payload.get("schema_version") or "") != PREFLIGHT_EVIDENCE_SCHEMA_VERSION:
        validation_issues.append("missing_or_invalid_schema_version")
    validation_issues.extend(_example_only_presence_issues(raw_payload))
    validation_issues.extend(_source_artifact_issues(raw_payload, path))
    candidate = str(_dot_get(raw_payload, "stripe_projects.voip_provider_candidate") or "").strip()
    return {
        "loaded": True,
        "path": str(path),
        "fields_present": fields_present,
        "missing_fields": missing_fields,
        "validation_issues": validation_issues,
        "voip_provider_candidate": candidate if VOIP_PROVIDER_CANDIDATE_RE.fullmatch(candidate) else None,
        "redaction_policy": "references_only",
    }


def _expand_preflight_evidence_manifest(path: Path, payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[str]]:
    reports = payload.get("reports")
    if not isinstance(reports, Mapping):
        return payload, []

    expanded: dict[str, Any] = {"schema_version": PREFLIGHT_EVIDENCE_SCHEMA_VERSION}
    issues: list[str] = []
    manifest_schema = str(payload.get("schema_version") or "")
    if not manifest_schema:
        issues.append("preflight_evidence_manifest:missing_schema_version")
    elif manifest_schema != PREFLIGHT_EVIDENCE_MANIFEST_SCHEMA_VERSION:
        issues.append("preflight_evidence_manifest:invalid_schema_version")
    if "example_only" in payload:
        expanded["example_only"] = payload["example_only"]
    for section_name, report_path_value in reports.items():
        if section_name not in PREFLIGHT_EVIDENCE_SECTIONS:
            issues.append(f"preflight_evidence_manifest:{section_name}:unknown_section")
            continue
        report_path_text = str(report_path_value or "").strip()
        if not report_path_text:
            issues.append(f"preflight_evidence_manifest:{section_name}:empty_report_path")
            continue
        path_issues = _relative_artifact_ref_issues(report_path_text, base_path=path)
        if path_issues:
            issues.extend(f"preflight_evidence_manifest:{section_name}:report_path:{issue}" for issue in path_issues)
            continue
        report_path = _resolve_manifest_report_path(path, report_path_text)
        loaded, report_issues = _load_preflight_manifest_report(report_path, section_name, source_base_path=path)
        issues.extend(f"preflight_evidence_manifest:{section_name}:{issue}" for issue in report_issues)
        if loaded is not None:
            expanded[section_name] = loaded
    return expanded if expanded else payload, issues


def _resolve_manifest_report_path(manifest_path: Path, report_path_text: str) -> Path:
    return manifest_path.parent / report_path_text


def _relative_artifact_ref_issues(ref_text: str, *, base_path: Path) -> list[str]:
    issues: list[str] = []
    ref = str(ref_text or "").strip()
    if not ref:
        return ["empty"]
    if ref.startswith("~"):
        issues.append("user_home_not_allowed")
    ref_path = Path(ref)
    if ref_path.is_absolute():
        issues.append("absolute_path_not_allowed")
    if ".." in ref_path.parts:
        issues.append("parent_traversal_not_allowed")
    if issues:
        return sorted(set(issues))
    base_dir = base_path.expanduser().resolve(strict=False).parent
    candidate = base_dir / ref
    try:
        resolved_candidate = candidate.resolve(strict=True)
    except OSError:
        resolved_candidate = candidate.resolve(strict=False)
    resolved_base = base_dir.resolve(strict=False)
    if resolved_candidate != resolved_base and resolved_base not in resolved_candidate.parents:
        issues.append("path_escape_not_allowed")
    return sorted(set(issues))


def _load_preflight_manifest_report(
    path: Path,
    section_name: str,
    *,
    source_base_path: Path | None = None,
) -> tuple[Mapping[str, Any] | None, list[str]]:
    resolved = path.expanduser().resolve(strict=False)
    if resolved == FORBIDDEN_ENV_ROOT or FORBIDDEN_ENV_ROOT in resolved.parents:
        raise ValueError(f"refusing to inspect forbidden Hermes worktree path: {resolved}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, ["evidence file not found", f"evidence file not found at {resolved}"]
    except json.JSONDecodeError as exc:
        return None, [f"evidence JSON parse failed: {exc.msg}"]
    if not isinstance(payload, Mapping):
        return None, ["evidence root must be an object"]
    issues: list[str] = []
    if "example_only" in payload:
        issues.append("example_only evidence is not accepted")
    section = payload.get(section_name)
    if isinstance(section, Mapping):
        section = _with_preflight_section_source_artifact(section, source_base_path or path)
        if "example_only" in section:
            issues.append("example_only evidence is not accepted")
        return section, issues
    return _with_preflight_section_source_artifact(payload, source_base_path or path), issues


def _with_preflight_section_source_artifact(section: Mapping[str, Any], path: Path) -> dict[str, Any]:
    section_copy = dict(section)
    source_artifact = str(section_copy.get("source_artifact") or "").strip()
    if source_artifact:
        section_copy[PREFLIGHT_SOURCE_ARTIFACT_BASE_FIELD] = str(path)
    return section_copy


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return bool(re.fullmatch(r"[0-9a-f]{64}", text))


def _collector_attestation_command_arg_is_sensitive(value: str) -> bool:
    return bool(
        BEARER_RE.search(value)
        or PREFLIGHT_SECRET_VALUE_RE.search(value)
        or SECRET_VALUE_RE.search(value)
        or PHONE_RE.search(value)
    )


def _collector_attestation_issues(
    section: Mapping[str, Any],
    *,
    section_name: str,
    expected_redacted_sha256: str,
) -> list[str]:
    issues: list[str] = []
    attestation = section.get("collector_attestation")
    if not isinstance(attestation, Mapping):
        return [f"{section_name}.collector_attestation: missing"]
    if attestation.get("example_only") is True:
        issues.append(f"{section_name}.collector_attestation: example_only")
    for field in COLLECTOR_ATTESTATION_REQUIRED_FIELDS:
        if field not in attestation:
            issues.append(f"{section_name}.collector_attestation.{field}: missing")
    for field in ("collector_name", "collector_version", "run_id", "git_commit"):
        value = str(attestation.get(field) or "").strip()
        if not value or value.lower() in {"placeholder", "example", "replace-me", "unknown"}:
            issues.append(f"{section_name}.collector_attestation.{field}: invalid")
    command_argv = attestation.get("command_argv")
    if not isinstance(command_argv, list) or not command_argv or not all(isinstance(item, str) and item for item in command_argv):
        issues.append(f"{section_name}.collector_attestation.command_argv: invalid")
    elif any(_collector_attestation_command_arg_is_sensitive(item) for item in command_argv):
        issues.append(f"{section_name}.collector_attestation.command_argv: secret_or_phone_like_value")
    started_at = _parse_preflight_timestamp(attestation.get("started_at"))
    finished_at = _parse_preflight_timestamp(attestation.get("finished_at"))
    if started_at is None:
        issues.append(f"{section_name}.collector_attestation.started_at: invalid")
    if finished_at is None:
        issues.append(f"{section_name}.collector_attestation.finished_at: invalid")
    if started_at is not None and finished_at is not None and started_at > finished_at:
        issues.append(f"{section_name}.collector_attestation.finished_at: before_started_at")
    for field in ("raw_artifact_sha256", "redacted_artifact_sha256", "parent_manifest_sha256"):
        if not _valid_sha256(attestation.get(field)):
            issues.append(f"{section_name}.collector_attestation.{field}: invalid")
    if (
        _valid_sha256(attestation.get("redacted_artifact_sha256"))
        and _valid_sha256(expected_redacted_sha256)
        and str(attestation.get("redacted_artifact_sha256")).strip().lower() != expected_redacted_sha256
    ):
        issues.append(f"{section_name}.collector_attestation.redacted_artifact_sha256: mismatch")
    return issues


def _source_artifact_issues(payload: Mapping[str, Any], evidence_path: Path) -> list[str]:
    issues: list[str] = []
    for section_name in PREFLIGHT_EVIDENCE_SECTIONS:
        section = payload.get(section_name)
        if not isinstance(section, Mapping):
            continue
        source_artifact = str(section.get("source_artifact") or "").strip()
        source_kind = str(section.get("source_artifact_kind") or "").strip()
        source_sha256 = str(section.get("source_artifact_sha256") or "").strip()
        if not source_artifact:
            issues.append(f"{section_name}.source_artifact: missing")
            continue
        if source_kind != PREFLIGHT_SOURCE_ARTIFACT_KIND:
            issues.append(f"{section_name}.source_artifact_kind: invalid")
        source_base_path = Path(str(section.get(PREFLIGHT_SOURCE_ARTIFACT_BASE_FIELD) or evidence_path))
        source_path_issues = _relative_artifact_ref_issues(source_artifact, base_path=source_base_path)
        if source_path_issues:
            issues.extend(f"{section_name}.source_artifact:{issue}" for issue in source_path_issues)
            continue
        source_path = _resolve_source_artifact_path(source_artifact, source_base_path)
        if not source_path.exists():
            issues.append(f"{section_name}.source_artifact: artifact not found")
            issues.append(f"{section_name}.source_artifact: artifact not found at {source_path.resolve(strict=False)}")
            continue
        try:
            artifact_bytes = source_path.read_bytes()
        except OSError as exc:
            issues.append(f"{section_name}.source_artifact: artifact unreadable: {exc.strerror or exc}")
            continue
        actual_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
            issues.append(f"{section_name}.source_artifact_sha256: invalid")
        elif source_sha256 != actual_sha256:
            issues.append(f"{section_name}.source_artifact_sha256: mismatch")
        issues.extend(
            _collector_attestation_issues(
                section,
                section_name=section_name,
                expected_redacted_sha256=actual_sha256,
            )
        )
        issues.extend(
            f"{section_name}.source_artifact:{issue}"
            for issue in _redacted_artifact_issues(artifact_bytes, section_name=section_name)
        )
    return issues


def _resolve_source_artifact_path(source_artifact: str, evidence_path: Path) -> Path:
    return evidence_path.parent / source_artifact


def _refresh_preflight_section_hashes(
    payload: MutableMapping[str, Any],
    evidence_path: Path,
    *,
    allowed_section: str | None = None,
    source_base_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    updates: list[dict[str, Any]] = []
    issues: list[str] = []
    section_names = [allowed_section] if allowed_section else list(PREFLIGHT_EVIDENCE_SECTIONS)
    for section_name in section_names:
        if section_name is None:
            continue
        section_candidate = payload.get(section_name)
        section: MutableMapping[str, Any] | None
        if isinstance(section_candidate, MutableMapping):
            section = section_candidate
        elif allowed_section == section_name:
            section = payload
        else:
            section = None
        if not isinstance(section, MutableMapping):
            continue
        source_artifact = str(section.get("source_artifact") or "").strip()
        if not source_artifact:
            issues.append(f"{section_name}.source_artifact: missing")
            continue
        source_ref_base = source_base_path or evidence_path
        source_path_issues = _relative_artifact_ref_issues(source_artifact, base_path=source_ref_base)
        if source_path_issues:
            issues.extend(f"{section_name}.source_artifact:{issue}" for issue in source_path_issues)
            continue
        source_path = _resolve_source_artifact_path(source_artifact, source_ref_base)
        if not source_path.exists():
            issues.append(f"{section_name}.source_artifact: artifact not found")
            issues.append(f"{section_name}.source_artifact: artifact not found at {source_path.resolve(strict=False)}")
            continue
        try:
            source_bytes = source_path.read_bytes()
        except OSError as exc:
            issues.append(f"{section_name}.source_artifact: artifact unreadable: {exc.strerror or exc}")
            continue
        artifact_issues = _redacted_artifact_issues(source_bytes, section_name=section_name)
        if artifact_issues:
            issues.extend(f"{section_name}.source_artifact:{issue}" for issue in artifact_issues)
            continue
        previous_sha256 = str(section.get("source_artifact_sha256") or "")
        new_sha256 = hashlib.sha256(source_bytes).hexdigest()
        section["source_artifact_sha256"] = new_sha256
        attestation = section.get("collector_attestation")
        previous_attestation_sha256: str | None = None
        attestation_changed = False
        if isinstance(attestation, MutableMapping):
            previous_attestation_sha256 = str(attestation.get("redacted_artifact_sha256") or "")
            attestation["redacted_artifact_sha256"] = new_sha256
            attestation_changed = previous_attestation_sha256 != new_sha256
        updates.append(
            {
                "section": section_name,
                "section_file": str(evidence_path),
                "source_artifact": source_artifact,
                "source_artifact_path": str(source_path),
                "previous_sha256": previous_sha256,
                "source_artifact_sha256": new_sha256,
                "previous_collector_attestation_redacted_artifact_sha256": previous_attestation_sha256,
                "collector_attestation_redacted_artifact_sha256": new_sha256 if isinstance(attestation, Mapping) else None,
                "collector_attestation_changed": attestation_changed,
                "changed": previous_sha256 != new_sha256,
            }
        )
    return updates, issues


def _refresh_preflight_hashes_in_file(
    path: Path,
    *,
    allowed_section: str | None = None,
    source_base_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [], [f"{path}: file not found"]
    except json.JSONDecodeError as exc:
        return [], [f"{path}: JSON parse failed: {exc.msg}"]
    if not isinstance(payload, MutableMapping):
        return [], [f"{path}: root must be an object"]
    updates, issues = _refresh_preflight_section_hashes(
        payload,
        path,
        allowed_section=allowed_section,
        source_base_path=source_base_path,
    )
    if issues:
        return [], issues
    if updates:
        _write_json(path, payload)
    return updates, issues


def refresh_preflight_source_hashes(path: Path) -> dict[str, Any]:
    """Refresh source_artifact_sha256 fields for local redacted preflight evidence."""

    resolved = path.expanduser().resolve(strict=False)
    if resolved == FORBIDDEN_ENV_ROOT or FORBIDDEN_ENV_ROOT in resolved.parents:
        raise ValueError(f"refusing to inspect forbidden Hermes worktree path: {resolved}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "ok": False,
            "schema_version": "voiceops.milestone2.preflight_hash_refresh.v1",
            "artifact_id": "voiceops-m2-preflight-hash-refresh",
            "target_path": str(path),
            "issues": ["target file not found"],
            "updates": [],
        }
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "schema_version": "voiceops.milestone2.preflight_hash_refresh.v1",
            "artifact_id": "voiceops-m2-preflight-hash-refresh",
            "target_path": str(path),
            "issues": [f"target JSON parse failed: {exc.msg}"],
            "updates": [],
        }
    if not isinstance(payload, Mapping):
        return {
            "ok": False,
            "schema_version": "voiceops.milestone2.preflight_hash_refresh.v1",
            "artifact_id": "voiceops-m2-preflight-hash-refresh",
            "target_path": str(path),
            "issues": ["target root must be an object"],
            "updates": [],
        }

    issues: list[str] = []
    updates: list[dict[str, Any]] = []
    reports = payload.get("reports")
    if isinstance(reports, Mapping):
        for section_name, report_path_value in reports.items():
            if section_name not in PREFLIGHT_EVIDENCE_SECTIONS:
                issues.append(f"preflight_evidence_manifest:{section_name}:unknown_section")
                continue
            report_path_text = str(report_path_value or "").strip()
            if not report_path_text:
                issues.append(f"preflight_evidence_manifest:{section_name}:empty_report_path")
                continue
            path_issues = _relative_artifact_ref_issues(report_path_text, base_path=path)
            if path_issues:
                issues.extend(f"preflight_evidence_manifest:{section_name}:report_path:{issue}" for issue in path_issues)
                continue
            report_path = _resolve_manifest_report_path(path, report_path_text)
            file_updates, file_issues = _refresh_preflight_hashes_in_file(
                report_path,
                allowed_section=section_name,
                source_base_path=path,
            )
            updates.extend(file_updates)
            issues.extend(f"preflight_evidence_manifest:{section_name}:{issue}" for issue in file_issues)
    else:
        if not isinstance(payload, MutableMapping):
            issues.append("target root must be mutable")
        else:
            updates, issues = _refresh_preflight_section_hashes(payload, path)
            if issues:
                updates = []
            elif updates:
                _write_json(path, payload)

    return {
        "ok": not issues,
        "schema_version": "voiceops.milestone2.preflight_hash_refresh.v1",
        "artifact_id": "voiceops-m2-preflight-hash-refresh",
        "generated_at": _utc_now(),
        "target_path": str(path),
        "manifest_mode": isinstance(reports, Mapping),
        "non_mutating_external_systems": True,
        "network_io": False,
        "env_secret_reads": False,
        "provider_provisioning": False,
        "live_spend": False,
        "updates": updates,
        "issues": issues,
    }


def _redacted_artifact_issues(artifact_bytes: bytes, *, section_name: str | None = None) -> list[str]:
    try:
        artifact = json.loads(artifact_bytes.decode("utf-8"))
    except UnicodeDecodeError:
        return ["artifact must be utf-8 JSON"]
    except json.JSONDecodeError as exc:
        return [f"artifact JSON parse failed: {exc.msg}"]
    if not isinstance(artifact, Mapping):
        return ["artifact root must be an object"]
    issues: list[str] = []
    if str(artifact.get("schema_version") or "") != PREFLIGHT_REDACTED_SOURCE_SCHEMA_VERSION:
        issues.append("missing_or_invalid_schema_version")
    if section_name is not None and str(artifact.get("section") or "") != section_name:
        issues.append("section_mismatch")
    issues.extend(_example_only_presence_issues(artifact))
    redaction_policy = str(artifact.get("redaction_policy") or "").lower()
    redacted_flag = artifact.get("redacted")
    if redacted_flag is not True and not (redacted_flag is None and _strict_affirmative_redaction_policy(redaction_policy)):
        issues.append("artifact is not marked redacted")
    issues.extend(_preflight_secret_issues(artifact))
    return issues


def _strict_affirmative_redaction_policy(policy: str) -> bool:
    policy = " ".join(policy.lower().split())
    if not policy:
        return False
    if re.search(r"\b(?:not redacted|unredacted|no redaction|without redaction)\b", policy):
        return False
    has_reference_scope = any(phrase in policy for phrase in ("references only", "refs only", "aliases only", "redacted"))
    has_no_raw = any(phrase in policy for phrase in ("no raw", "never raw", "without raw"))
    has_sensitive_scope = any(
        term in policy
        for term in ("secret", "token", "credential", "card", "phone", "password", "api key", "api-key")
    )
    return has_reference_scope and has_no_raw and has_sensitive_scope


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
    if tuple(argv) not in SAFE_PROBE_ARGV_TUPLES:
        raise ValueError(f"probe command must match the allowlisted manifest exactly: {joined}")


def _validate_readonly_discovery_command(argv: Sequence[str]) -> None:
    if not argv:
        raise ValueError("empty read-only discovery command")
    joined = " ".join(argv).lower()
    if tuple(argv) not in READONLY_DISCOVERY_ARGV_TUPLES:
        raise ValueError(f"read-only discovery command must match the allowlisted manifest exactly: {joined}")
    for pattern in MUTATING_COMMAND_PATTERNS:
        if pattern in joined and pattern not in {"token", "secret", "credential", "login", "whoami"}:
            raise ValueError(f"refusing mutating discovery command: {joined}")


def _isolated_subprocess_runner(
    argv: Sequence[str],
    timeout_seconds: int,
    validator: Callable[[Sequence[str]], None],
) -> CommandResult:
    validator(argv)
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


def _subprocess_runner(argv: Sequence[str], timeout_seconds: int) -> CommandResult:
    return _isolated_subprocess_runner(argv, timeout_seconds, _validate_safe_probe_command)


def _readonly_discovery_subprocess_runner(argv: Sequence[str], timeout_seconds: int) -> CommandResult:
    return _isolated_subprocess_runner(argv, timeout_seconds, _validate_readonly_discovery_command)


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


def _readonly_discovery_manifest() -> list[CommandProbe]:
    return [
        CommandProbe(
            probe_id="stripe_projects_catalog_list",
            area="stripe_projects",
            argv=["stripe", "projects", "list", "--limit", "10"],
            required=False,
            purpose="Optionally confirm visible Stripe Projects catalog entries without creating a project.",
        ),
        CommandProbe(
            probe_id="stripe_link_auth_status",
            area="stripe_link",
            argv=["link-cli", "auth", "status"],
            required=False,
            purpose="Optionally confirm Link auth/account status without creating a spend request.",
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
            "timeout_seconds": timeout_seconds,
            "stdout_excerpt": _excerpt(command_result.stdout),
            "stderr_excerpt": _excerpt(command_result.stderr),
            "status": "pass" if command_result.exit_code == 0 and not command_result.timed_out else "fail",
        }
    )
    return result


def _run_readonly_discovery_probe(
    probe: CommandProbe,
    *,
    which: Callable[[str], str | None],
    runner: CommandRunner,
    timeout_seconds: int,
    run_discovery: bool,
) -> dict[str, Any]:
    _validate_readonly_discovery_command(probe.argv)
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
        "status": "missing" if not path else "not_requested",
        "non_mutating": True,
        "does_not_grant_approval": True,
        "redacted_outputs_only": True,
    }
    if not path or not run_discovery:
        return result
    command_result = runner(probe.argv, timeout_seconds)
    result.update(
        {
            "executed": True,
            "exit_code": command_result.exit_code,
            "timed_out": command_result.timed_out,
            "timeout_seconds": timeout_seconds,
            "stdout_excerpt": _excerpt(command_result.stdout),
            "stderr_excerpt": _excerpt(command_result.stderr),
            "status": "pass" if command_result.exit_code == 0 and not command_result.timed_out else "fail",
        }
    )
    return result


def _build_readonly_discovery_report(
    *,
    which: Callable[[str], str | None],
    runner: CommandRunner,
    timeout_seconds: int,
    run_discovery: bool,
) -> dict[str, Any]:
    probes = [
        _run_readonly_discovery_probe(
            probe,
            which=which,
            runner=runner,
            timeout_seconds=timeout_seconds,
            run_discovery=run_discovery,
        )
        for probe in _readonly_discovery_manifest()
    ]
    executed = [probe for probe in probes if probe["executed"]]
    failed = [probe["probe_id"] for probe in executed if probe["status"] != "pass"]
    missing = [probe["probe_id"] for probe in probes if probe["status"] == "missing"]
    status = "not_requested"
    if run_discovery:
        status = "fail" if failed else "needs_tools" if missing else "pass"
    report = {
        "schema_version": READ_ONLY_DISCOVERY_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "run_requested": run_discovery,
        "non_mutating": True,
        "does_not_grant_approval": True,
        "redacted_outputs_only": True,
        "required_for_live_provisioning_approval": True,
        "auth_context": "isolated_home",
        "proves_existing_local_auth": False,
        "network_io_possible": run_discovery,
        "timeout_seconds": timeout_seconds if run_discovery else None,
        "status": status,
        "failed_probe_ids": failed,
        "timed_out_probe_ids": [
            probe["probe_id"]
            for probe in executed
            if probe.get("timed_out") is True
        ],
        "missing_probe_ids": missing,
        "allowlisted_commands": [list(command) for command in sorted(READONLY_DISCOVERY_ARGV_TUPLES)],
        "blocked_capabilities": BLOCKED_CAPABILITIES,
        "probes": probes,
    }
    report["collector_attestation"] = _readonly_discovery_collector_attestation(report)
    return report


def _readonly_discovery_not_loaded() -> dict[str, Any]:
    return _build_readonly_discovery_report(
        which=lambda _command: None,
        runner=lambda _argv, _timeout_seconds: CommandResult(exit_code=127),
        timeout_seconds=0,
        run_discovery=False,
    )


def _readonly_discovery_report_redacted_sha256(discovery: Mapping[str, Any]) -> str:
    attested_payload = dict(discovery)
    attested_payload.pop("collector_attestation", None)
    encoded = json.dumps(attested_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _readonly_discovery_collector_attestation(discovery: Mapping[str, Any]) -> dict[str, Any]:
    redacted_sha256 = _readonly_discovery_report_redacted_sha256(discovery)
    timestamp = _utc_now()
    return {
        "collector_name": "scripts.voiceops_provisioning_probe",
        "collector_version": "voiceops.milestone2.read_only_discovery.v1",
        "run_id": f"read-only-discovery-{redacted_sha256[:12]}",
        "command_argv": [sys.executable, "scripts/voiceops_provisioning_probe.py"],
        "git_commit": "unavailable",
        "started_at": str(discovery.get("generated_at") or timestamp),
        "finished_at": timestamp,
        "raw_artifact_sha256": redacted_sha256,
        "redacted_artifact_sha256": redacted_sha256,
        "parent_manifest_sha256": redacted_sha256,
    }


def _readonly_discovery_export_payload(discovery: Mapping[str, Any]) -> dict[str, Any]:
    """Return a standalone discovery report that can be re-ingested later."""

    payload = dict(discovery)
    if payload.get("loaded_from_evidence") is True and "source_network_io_possible" in payload:
        payload["network_io_possible"] = bool(payload.get("source_network_io_possible"))
    for key in (
        "loaded_from_evidence",
        "evidence_path",
        "source_network_io_possible",
        "validation_issues",
    ):
        payload.pop(key, None)
    payload["collector_attestation"] = _readonly_discovery_collector_attestation(payload)
    return payload


def _readonly_discovery_validation_issues(discovery: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    if str(discovery.get("schema_version") or "") != READ_ONLY_DISCOVERY_SCHEMA_VERSION:
        issues.append("read_only_discovery:missing_or_invalid_schema_version")
    issues.extend(
        _collector_attestation_issues(
            discovery,
            section_name="read_only_discovery",
            expected_redacted_sha256=_readonly_discovery_report_redacted_sha256(discovery),
        )
    )
    required_flags = {
        "non_mutating": True,
        "does_not_grant_approval": True,
        "redacted_outputs_only": True,
        "required_for_live_provisioning_approval": True,
        "proves_existing_local_auth": False,
    }
    for key, expected in required_flags.items():
        if discovery.get(key) is not expected:
            issues.append(f"read_only_discovery:{key}_must_be_{str(expected).lower()}")
    if str(discovery.get("auth_context") or "") != "isolated_home":
        issues.append("read_only_discovery:auth_context_must_be_isolated_home")
    if str(discovery.get("status") or "") != "pass":
        issues.append("read_only_discovery:status_not_pass")
    if list(discovery.get("blocked_capabilities") or []) != BLOCKED_CAPABILITIES:
        issues.append("read_only_discovery:blocked_capabilities_mismatch")
    allowlisted = [list(command) for command in sorted(READONLY_DISCOVERY_ARGV_TUPLES)]
    if discovery.get("allowlisted_commands") != allowlisted:
        issues.append("read_only_discovery:allowlisted_commands_mismatch")
    probes = discovery.get("probes")
    if not isinstance(probes, list):
        return [*issues, "read_only_discovery:probes_not_list"]
    expected_by_id = {probe.probe_id: probe for probe in _readonly_discovery_manifest()}
    seen: set[str] = set()
    for index, probe in enumerate(probes):
        if not isinstance(probe, Mapping):
            issues.append(f"read_only_discovery:probes[{index}]:not_object")
            continue
        probe_id = str(probe.get("probe_id") or "")
        expected = expected_by_id.get(probe_id)
        if expected is None:
            issues.append(f"read_only_discovery:{probe_id or index}:unknown_probe_id")
            continue
        seen.add(probe_id)
        if list(probe.get("argv") or []) != list(expected.argv):
            issues.append(f"read_only_discovery:{probe_id}:argv_mismatch")
        if str(probe.get("status") or "") != "pass":
            issues.append(f"read_only_discovery:{probe_id}:status_not_pass")
        if probe.get("executed") is not True:
            issues.append(f"read_only_discovery:{probe_id}:not_executed")
    missing = sorted(set(expected_by_id) - seen)
    if missing:
        issues.append(f"read_only_discovery:missing_probes:{','.join(missing)}")
    return issues


def _load_readonly_discovery_report(path: Path) -> tuple[Mapping[str, Any] | None, list[str]]:
    resolved = path.expanduser().resolve(strict=False)
    if resolved == FORBIDDEN_ENV_ROOT or FORBIDDEN_ENV_ROOT in resolved.parents:
        raise ValueError(f"refusing to inspect forbidden Hermes worktree path: {resolved}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, ["read_only_discovery:file_not_found"]
    except json.JSONDecodeError as exc:
        return None, [f"read_only_discovery:json_parse_failed:{exc.msg}"]
    if not isinstance(payload, Mapping):
        return None, ["read_only_discovery:root_must_be_object"]
    if str(payload.get("schema_version") or "") == READ_ONLY_DISCOVERY_MANIFEST_SCHEMA_VERSION:
        report_ref = str(payload.get("report") or "").strip()
        if not report_ref:
            return None, ["read_only_discovery_manifest:missing_report"]
        report_path_issues = _relative_artifact_ref_issues(report_ref, base_path=path)
        if report_path_issues:
            return None, [
                f"read_only_discovery_manifest:report_path:{issue}" for issue in report_path_issues
            ]
        report_path = _resolve_manifest_report_path(path, report_ref)
        expected_report_sha256 = str(payload.get("report_sha256") or "").strip().lower()
        manifest_issues: list[str] = []
        if not expected_report_sha256:
            manifest_issues.append("read_only_discovery_manifest:report_sha256:missing")
        elif not _valid_sha256(expected_report_sha256):
            manifest_issues.append("read_only_discovery_manifest:report_sha256:invalid")
        else:
            try:
                actual_report_sha256 = _file_sha256(report_path)
            except OSError:
                actual_report_sha256 = None
            if actual_report_sha256 is not None and expected_report_sha256 != actual_report_sha256:
                manifest_issues.append("read_only_discovery_manifest:report_sha256:mismatch")
        report, issues = _load_readonly_discovery_report(report_path)
        if str(payload.get("status") or "") != "pass":
            manifest_issues.append("read_only_discovery_manifest:status_not_pass")
        if payload.get("does_not_grant_approval") is not True:
            manifest_issues.append("read_only_discovery_manifest:does_not_grant_approval_must_be_true")
        if payload.get("redacted_outputs_only") is not True:
            manifest_issues.append("read_only_discovery_manifest:redacted_outputs_only_must_be_true")
        return report, [*manifest_issues, *issues]
    return payload, _readonly_discovery_validation_issues(payload)


def load_readonly_discovery_evidence(path: Path | None) -> dict[str, Any]:
    if path is None:
        return _readonly_discovery_not_loaded()
    report, issues = _load_readonly_discovery_report(path)
    if report is None:
        discovery = _readonly_discovery_not_loaded()
        discovery["status"] = "fail"
        discovery["validation_issues"] = issues
        discovery["evidence_path"] = str(path)
        return discovery
    discovery = dict(report)
    source_network_io_possible = bool(discovery.get("network_io_possible"))
    discovery["loaded_from_evidence"] = True
    discovery["evidence_path"] = str(path)
    discovery["source_network_io_possible"] = source_network_io_possible
    discovery["network_io_possible"] = False
    discovery["validation_issues"] = issues
    if issues:
        discovery["status"] = "fail"
    return discovery


def _nemoclaw_gate_issues(packet: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    gate = packet.get("kame_evidence_gate") if isinstance(packet.get("kame_evidence_gate"), Mapping) else {}
    if not gate:
        return ["missing_kame_evidence_gate"]
    if gate.get("requires_promoted_evidence") is not True:
        issues.append("kame_evidence_gate.requires_promoted_evidence_not_true")
    if gate.get("hypotheses_allowed_for_action") is not False:
        issues.append("kame_evidence_gate.hypotheses_allowed_for_action_not_false")
    accepted = set(gate.get("accepted_authorities") or [])
    for authority in KAME_PROMOTED_AUTHORITIES:
        if authority not in accepted:
            issues.append(f"kame_evidence_gate.missing_accepted_authority:{authority}")
    rejected = set(gate.get("rejected_authorities") or [])
    for authority in KAME_REJECTED_AUTHORITIES:
        if authority not in rejected:
            issues.append(f"kame_evidence_gate.missing_rejected_authority:{authority}")
    merge_keys = set(gate.get("merge_key_fields") or [])
    for field in ("turn_id", "audio_segment_ref"):
        if field not in merge_keys:
            issues.append(f"kame_evidence_gate.missing_merge_key:{field}")
    return issues


def _tool_disclosure_issues(packet: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    proof = packet.get("tool_disclosure") if isinstance(packet.get("tool_disclosure"), Mapping) else {}
    if not proof:
        return ["missing_tool_disclosure"]
    if proof.get("ok") is not True:
        issues.append("tool_disclosure.ok_not_true")
    config = proof.get("config") if isinstance(proof.get("config"), Mapping) else {}
    if config.get("enabled") != "on":
        issues.append("tool_disclosure.config.enabled_not_on")
    if config.get("defer_core") != "all":
        issues.append("tool_disclosure.config.defer_core_not_all")
    visible = set(proof.get("visible_tool_names") or [])
    for tool_name in TOOL_DISCLOSURE_BRIDGE_TOOL_NAMES:
        if tool_name not in visible:
            issues.append(f"tool_disclosure.visible_tool_missing:{tool_name}")
    visible_non_bridge = set(proof.get("visible_non_bridge_tool_names") or [])
    if visible_non_bridge:
        issues.append("tool_disclosure.visible_non_bridge_tools_present")
    if proof.get("broad_core_tools_visible") is not False:
        issues.append("tool_disclosure.broad_core_tools_visible")
    hidden = set(proof.get("hidden_core_tool_names") or [])
    input_core = set(proof.get("input_core_tools") or [])
    for tool_name in input_core or {"read_file", "terminal"}:
        if tool_name not in hidden:
            issues.append(f"tool_disclosure.hidden_core_tool_missing:{tool_name}")
    if input_core and hidden != input_core:
        issues.append("tool_disclosure.hidden_core_tool_set_mismatch")
    if proof.get("core_tools_hidden_all") is not True:
        issues.append("tool_disclosure.core_tools_hidden_all_not_true")
    if proof.get("hidden_core_tool_count") != len(hidden):
        issues.append("tool_disclosure.hidden_core_tool_count_mismatch")
    if proof.get("input_core_tool_count") != len(input_core):
        issues.append("tool_disclosure.input_core_tool_count_mismatch")
    if proof.get("deferred_count") != len(hidden):
        issues.append("tool_disclosure.deferred_count_mismatch")
    if int(proof.get("token_reduction_estimate") or 0) <= 0:
        issues.append("tool_disclosure.missing_token_reduction")
    refs = set(proof.get("external_test_refs") or [])
    for ref in TOOL_DISCLOSURE_TEST_REFS:
        if ref not in refs:
            issues.append(f"tool_disclosure.missing_external_test_ref:{ref}")
    return issues


def _evidence_label_values(payload: Any) -> Iterable[str]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key == "evidence_label":
                yield str(value)
            else:
                yield from _evidence_label_values(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _evidence_label_values(item)


def _kame_action_evidence_issues(action: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    action_id = str(action.get("action_id") or "")
    evidence = action.get("kame_evidence") if isinstance(action.get("kame_evidence"), Mapping) else {}
    if not evidence:
        return [f"{action_id}:missing_kame_evidence"]
    if evidence.get("schema_version") != "voiceops.kame_action_evidence.v1":
        issues.append(f"{action_id}:kame_evidence_schema_invalid")
    if evidence.get("action_id") != action_id:
        issues.append(f"{action_id}:kame_evidence_action_id_mismatch")
    if not str(evidence.get("turn_id") or "").strip():
        issues.append(f"{action_id}:kame_evidence_missing_turn_id")
    if not str(evidence.get("audio_segment_ref") or "").strip():
        issues.append(f"{action_id}:kame_evidence_missing_audio_segment_ref")
    if evidence.get("hypotheses_allowed_for_action") is not False:
        issues.append(f"{action_id}:kame_evidence_hypotheses_allowed")
    if evidence.get("transcript_hypotheses_promoted") is not False:
        issues.append(f"{action_id}:kame_evidence_transcript_hypotheses_promoted")
    required_promotions = set(evidence.get("required_promotions") or [])
    for authority in KAME_PROMOTED_AUTHORITIES:
        if authority not in required_promotions:
            issues.append(f"{action_id}:kame_evidence_missing_required_promotion:{authority}")
    promoted_fields = evidence.get("promoted_fields") if isinstance(evidence.get("promoted_fields"), Mapping) else {}
    for field in KAME_ACTION_PROMOTED_FIELDS.get(action_id, ("user_request", "oracle_action_plan")):
        promoted = promoted_fields.get(field) if isinstance(promoted_fields.get(field), Mapping) else {}
        if not promoted:
            issues.append(f"{action_id}:missing_promoted_field:{field}")
            continue
        authority = str(promoted.get("evidence_label") or "")
        if authority not in KAME_PROMOTED_AUTHORITIES:
            issues.append(f"{action_id}:promoted_field_authority_invalid:{field}:{authority}")
    for authority in _evidence_label_values(promoted_fields):
        if authority in KAME_REJECTED_AUTHORITIES:
            issues.append(f"{action_id}:promoted_field_uses_rejected_authority:{authority}")
    if action.get("tool_disclosure_ref") != "tool_disclosure":
        issues.append(f"{action_id}:tool_disclosure_ref_not_tool_disclosure")
    return issues


def validate_nemoclaw_action_packet(packet: Mapping[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    if packet.get("schema_version") != "voiceops.nemoclaw_action_packet.v1":
        issues.append("missing_or_invalid_schema_version")
    if packet.get("artifact_id") != "voiceops-nemoclaw-action-packet":
        issues.append("missing_or_invalid_artifact_id")
    if packet.get("runtime") != "NemoClaw":
        issues.append("runtime_not_nemoclaw")
    if packet.get("mode") != "dry_run_until_user_approval":
        issues.append("mode_not_dry_run_until_user_approval")
    issues.extend(_example_only_presence_issues(packet))
    issues.extend(_preflight_secret_issues(packet))
    issues.extend(_nemoclaw_gate_issues(packet))
    issues.extend(_tool_disclosure_issues(packet))

    safety = packet.get("safety") if isinstance(packet.get("safety"), Mapping) else {}
    for key in ("live_spend", "provider_provisioning", "credential_retrieval", "outbound_phone_calls", "network_io"):
        if safety.get(key) is not False:
            issues.append(f"safety.{key}_not_false")
    if safety.get("requires_operator_approval") is not True:
        issues.append("safety.requires_operator_approval_not_true")
    if safety.get("default_decision") != "hold":
        issues.append("safety.default_decision_not_hold")

    required_blocked = [
        "raw_card_data_in_model_context",
        "unapproved_purchase",
        "unbounded_network_access",
    ]
    blocked_capabilities = set(packet.get("blocked_capabilities") or [])
    for required_block in required_blocked:
        if required_block not in blocked_capabilities:
            issues.append(f"missing_blocked_capability:{required_block}")

    actions = packet.get("approval_required_actions") if isinstance(packet.get("approval_required_actions"), list) else []
    contracts = packet.get("approval_contracts") if isinstance(packet.get("approval_contracts"), Mapping) else {}
    dry_run_commands = packet.get("dry_run_commands") if isinstance(packet.get("dry_run_commands"), list) else []
    action_ids = {str(action.get("action_id")) for action in actions if isinstance(action, Mapping)}
    if not actions:
        issues.append("approval_required_actions_empty")
    if set(contracts) != action_ids:
        issues.append("approval_contracts_do_not_match_actions")

    for action in actions:
        if not isinstance(action, Mapping):
            issues.append("approval_required_action_not_object")
            continue
        action_id = str(action.get("action_id") or "")
        command = str(action.get("command") or "")
        contract = action.get("approval_contract") if isinstance(action.get("approval_contract"), Mapping) else {}
        indexed_contract = contracts.get(action_id)
        if action.get("requires_approval") is not True:
            issues.append(f"{action_id}:requires_approval_not_true")
        if action.get("status") not in {"queued", "held-budget"}:
            issues.append(f"{action_id}:status_not_queued_or_held_budget")
        if not command:
            issues.append(f"{action_id}:missing_command")
        elif command not in dry_run_commands:
            issues.append(f"{action_id}:command_missing_from_dry_run_commands")
        if not contract:
            issues.append(f"{action_id}:missing_approval_contract")
            continue
        if indexed_contract != contract:
            issues.append(f"{action_id}:approval_contract_index_mismatch")
        if contract.get("approval_artifact") != "nemoclaw-action-packet.json":
            issues.append(f"{action_id}:approval_artifact_not_packet")
        if contract.get("allowed_decisions") != ["approve_once", "deny", "hold"]:
            issues.append(f"{action_id}:allowed_decisions_invalid")
        if contract.get("default_decision") != "hold":
            issues.append(f"{action_id}:default_decision_not_hold")
        if contract.get("status") not in {"pending", "blocked"}:
            issues.append(f"{action_id}:approval_status_not_pending_or_blocked")
        if contract.get("approved_by_ref") is not None:
            issues.append(f"{action_id}:approved_by_ref_present")
        expected_hash = hashlib.sha256(command.encode("utf-8")).hexdigest()
        if contract.get("command_sha256") != expected_hash:
            issues.append(f"{action_id}:command_sha256_mismatch")
        if not contract.get("required_preflight_gates"):
            issues.append(f"{action_id}:missing_required_preflight_gates")
        issues.extend(_kame_action_evidence_issues(action))
    if set(dry_run_commands) != {str(action.get("command")) for action in actions if isinstance(action, Mapping)}:
        issues.append("dry_run_commands_do_not_match_approval_actions")

    return {
        "schema_version": NEMOCLAW_ACTION_PACKET_VALIDATION_SCHEMA_VERSION,
        "artifact_id": "voiceops-nemoclaw-action-packet-validation",
        "packet_id": packet.get("packet_id"),
        "ok": not issues,
        "status": "valid" if not issues else "invalid",
        "mode": "local_static_validation_only",
        "loaded": True,
        "path": None,
        "safety": {
            "executes_commands": False,
            "network_io": False,
            "live_spend": False,
            "provider_provisioning": False,
            "credential_retrieval": False,
            "outbound_phone_calls": False,
            "secret_values_emitted": False,
        },
        "validation_issues": sorted(set(issues)),
        "warnings": warnings,
        "validated_contract_count": len(actions),
        "dry_run_command_count": len(dry_run_commands),
        "required_blocked_capabilities": required_blocked,
    }


def load_nemoclaw_action_packet(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "schema_version": NEMOCLAW_ACTION_PACKET_VALIDATION_SCHEMA_VERSION,
            "artifact_id": "voiceops-nemoclaw-action-packet-validation",
            "ok": False,
            "status": "not_supplied",
            "mode": "local_static_validation_only",
            "loaded": False,
            "path": None,
            "safety": {
                "executes_commands": False,
                "network_io": False,
                "live_spend": False,
                "provider_provisioning": False,
                "credential_retrieval": False,
                "outbound_phone_calls": False,
                "secret_values_emitted": False,
            },
            "validation_issues": [],
            "warnings": [],
            "validated_contract_count": 0,
            "dry_run_command_count": 0,
            "required_blocked_capabilities": [
                "raw_card_data_in_model_context",
                "unapproved_purchase",
                "unbounded_network_access",
            ],
        }
    resolved = path.expanduser().resolve(strict=False)
    if resolved == FORBIDDEN_ENV_ROOT or FORBIDDEN_ENV_ROOT in resolved.parents:
        raise ValueError(f"refusing to inspect forbidden Hermes worktree path: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError:
        report = load_nemoclaw_action_packet(None)
        report.update(
            {
                "status": "not_found",
                "path": str(path),
                "validation_issues": ["nemoclaw_action_packet:file_not_found"],
            }
        )
        return report
    except json.JSONDecodeError as exc:
        report = load_nemoclaw_action_packet(None)
        report.update(
            {
                "status": "invalid",
                "path": str(path),
                "validation_issues": [f"nemoclaw_action_packet:json_parse_failed:{exc.msg}"],
            }
        )
        return report
    if not isinstance(payload, Mapping):
        report = load_nemoclaw_action_packet(None)
        report.update(
            {
                "status": "invalid",
                "path": str(path),
                "validation_issues": ["nemoclaw_action_packet:root_must_be_object"],
            }
        )
        return report
    report = validate_nemoclaw_action_packet(payload)
    report["path"] = str(path)
    return report


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
    repo_root: Path = REPO_ROOT,
    preflight_evidence_path: Path | None = None,
    read_only_discovery_evidence_path: Path | None = None,
    post_approval_receipts_path: Path | None = None,
    nemoclaw_action_packet_path: Path | None = None,
    which: Callable[[str], str | None] = shutil.which,
    runner: CommandRunner = _subprocess_runner,
    readonly_discovery_runner: CommandRunner = _readonly_discovery_subprocess_runner,
    run_commands: bool = False,
    run_readonly_discovery: bool = False,
    timeout_seconds: int | None = None,
    readonly_discovery_timeout_seconds: int | None = None,
) -> dict[str, Any]:
    env, env_sources = _merge_env_sources(os.environ if env is None else env, _default_env_files() if env_files is None else env_files)
    payment_skill_bundle = load_payment_skill_bundle_evidence(repo_root)
    command_timeout_seconds = (
        DEFAULT_COMMAND_PROBE_TIMEOUT_SECONDS
        if timeout_seconds is None
        else timeout_seconds
    )
    effective_readonly_discovery_timeout_seconds = (
        DEFAULT_READONLY_DISCOVERY_TIMEOUT_SECONDS
        if readonly_discovery_timeout_seconds is None
        else readonly_discovery_timeout_seconds
    )
    command_results = [
        _run_probe(
            probe,
            which=which,
            runner=runner,
            timeout_seconds=command_timeout_seconds,
            run_commands=run_commands,
        )
        for probe in _command_manifest()
    ]
    readonly_discovery = (
        load_readonly_discovery_evidence(read_only_discovery_evidence_path)
        if read_only_discovery_evidence_path is not None
        else _build_readonly_discovery_report(
            which=which,
            runner=readonly_discovery_runner,
            timeout_seconds=effective_readonly_discovery_timeout_seconds,
            run_discovery=run_readonly_discovery,
        )
    )

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
    checks.append(
        ReadinessCheck(
            check_id="stripe_skills_bundle",
            area="stripe_skills",
            status=payment_skill_bundle["status"],
            required=True,
            detail=(
                "Stripe Projects, Link, and MPP optional skills are present with required safety terms"
                if payment_skill_bundle["status"] == "pass"
                else "Stripe Skills bundle safety check failed: "
                + ", ".join(payment_skill_bundle["issues"])
            ),
            next_step="Restore or update optional payment skills before approving Stripe/Link/MPP provisioning flows.",
            evidence=payment_skill_bundle,
        )
    )

    target_present = _present_keys(env, PHONE_TARGET_ENV_KEYS)
    provider_present = _present_keys(env, PHONE_PROVIDER_ENV_KEYS)
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

    preflight_evidence = load_preflight_evidence(preflight_evidence_path)
    nemoclaw_action_packet = load_nemoclaw_action_packet(nemoclaw_action_packet_path)

    def preflight_check(
        check_id: str,
        area: str,
        required_fields: list[str],
        detail: str,
        next_step: str,
    ) -> ReadinessCheck:
        missing = [field for field in required_fields if field in preflight_evidence["missing_fields"]]
        status = "pass" if preflight_evidence["loaded"] and not missing and not preflight_evidence["validation_issues"] else "fail"
        issue_detail = "; ".join(preflight_evidence["validation_issues"])
        return ReadinessCheck(
            check_id=check_id,
            area=area,
            status=status,
            required=True,
            detail=detail if status == "pass" else issue_detail or f"missing preflight evidence fields: {', '.join(missing)}",
            next_step=next_step,
            evidence={
                "preflight_evidence_loaded": preflight_evidence["loaded"],
                "preflight_evidence_path": preflight_evidence["path"],
                "required_fields": required_fields,
                "missing_fields": missing,
            },
        )

    checks.extend(
        [
            preflight_check(
                "stripe_projects_account",
                "stripe_projects",
                [
                    "stripe_projects.source_artifact",
                    "stripe_projects.source_artifact_kind",
                    "stripe_projects.source_artifact_sha256",
                    "stripe_projects.source_artifact_redacted_at",
                    "stripe_projects.account_ref",
                    "stripe_projects.projects_catalog_checked_at",
                    "stripe_projects.voip_provider_candidate",
                    "stripe_projects.can_create_project_after_approval",
                ],
                "Stripe Projects account/capability evidence is present",
                "Fill `provisioning-preflight-evidence.template.json` with redacted Stripe Projects references and source_artifact.",
            ),
            preflight_check(
                "stripe_link_approval_capability",
                "stripe_link",
                [
                    "stripe_link.source_artifact",
                    "stripe_link.source_artifact_kind",
                    "stripe_link.source_artifact_sha256",
                    "stripe_link.source_artifact_redacted_at",
                    "stripe_link.account_ref",
                    "stripe_link.approval_capability_confirmed",
                    "stripe_link.max_approved_cents",
                    "stripe_link.currency",
                ],
                "Stripe Link approval capability evidence is present",
                "Fill Link source_artifact, account, approval capability, max approved cents, and currency in the preflight evidence file.",
            ),
            preflight_check(
                "mpp_approval_boundary",
                "mpp",
                [
                    "mpp.source_artifact",
                    "mpp.source_artifact_kind",
                    "mpp.source_artifact_sha256",
                    "mpp.source_artifact_redacted_at",
                    "mpp.boundary_tool",
                    "mpp.policy_ref",
                    "mpp.approval_packet_ref",
                ],
                "MPP/NemoClaw approval boundary evidence is present",
                "Fill source_artifact, boundary tool, policy ref, and approval packet ref in the preflight evidence file.",
            ),
            preflight_check(
                "phone_provider_account",
                "phone_handoff",
                [
                    "phone_handoff.source_artifact",
                    "phone_handoff.source_artifact_kind",
                    "phone_handoff.source_artifact_sha256",
                    "phone_handoff.source_artifact_redacted_at",
                    "phone_handoff.provider",
                    "phone_handoff.provider_account_ref",
                    "phone_handoff.phone_target_ref",
                ],
                "Phone provider account and target evidence is present",
                "Fill source_artifact, provider, provider account ref, and phone target ref in the preflight evidence file.",
            ),
            preflight_check(
                "credential_location_reference",
                "phone_handoff",
                [
                    "phone_handoff.source_artifact",
                    "phone_handoff.source_artifact_kind",
                    "phone_handoff.source_artifact_sha256",
                    "phone_handoff.source_artifact_redacted_at",
                    "phone_handoff.credential_location_ref",
                ],
                "Credential location reference evidence is present",
                "Fill a source_artifact and non-secret credential location ref in the preflight evidence file.",
            ),
            preflight_check(
                "rollback_owner_refs",
                "rollback",
                [
                    "rollback.source_artifact",
                    "rollback.source_artifact_kind",
                    "rollback.source_artifact_sha256",
                    "rollback.source_artifact_redacted_at",
                    "rollback.deprovision_owner",
                    "rollback.refund_or_cancel_owner",
                    "rollback.call_cancel_owner",
                ],
                "Rollback owner refs are present",
                "Fill all rollback owner refs in the preflight evidence file.",
            ),
        ]
    )

    check_dicts = [asdict(check) for check in checks]
    required_failures = [check["check_id"] for check in check_dicts if check["required"] and check["status"] != "pass"]
    if readonly_discovery["status"] != "pass":
        required_failures.append("read_only_discovery_passed")
    if nemoclaw_action_packet["loaded"] and nemoclaw_action_packet["status"] != "valid":
        required_failures.append("nemoclaw_action_packet_valid")
    area_status: dict[str, str] = {}
    for area in sorted({check["area"] for check in check_dicts}):
        area_checks = [check for check in check_dicts if check["area"] == area]
        area_status[area] = "pass" if all(check["status"] == "pass" for check in area_checks if check["required"]) else "fail"
    area_status["read_only_discovery"] = "pass" if readonly_discovery["status"] == "pass" else "fail"
    area_status["nemoclaw_action_packet"] = (
        "pass"
        if nemoclaw_action_packet["status"] == "valid"
        else "not_supplied"
        if not nemoclaw_action_packet["loaded"]
        else "fail"
    )
    ready = not required_failures
    report = {
        "generated_at": _utc_now(),
        "probe": {
            "name": "voiceops_provisioning_readiness",
            "non_mutating": True,
            "bounded": True,
            "run_commands": run_commands,
            "run_readonly_discovery": run_readonly_discovery,
            "read_only_discovery_evidence_path": str(read_only_discovery_evidence_path) if read_only_discovery_evidence_path else None,
            "nemoclaw_action_packet_path": str(nemoclaw_action_packet_path) if nemoclaw_action_packet_path else None,
            "timeout_seconds": command_timeout_seconds,
            "read_only_discovery_timeout_seconds": effective_readonly_discovery_timeout_seconds,
            "active_probe_policy": "version_help_only",
            "read_only_discovery_policy": "exact_allowlist_only",
            "blocked_capabilities": BLOCKED_CAPABILITIES,
        },
        "status": "ready" if ready else "needs_setup",
        "ready": ready,
        "required_failures": required_failures,
        "preflight_evidence_loaded": preflight_evidence["loaded"],
        "preflight_evidence_missing_fields": preflight_evidence["missing_fields"],
        "area_status": area_status,
        "payment_skill_bundle": payment_skill_bundle,
        "preflight_evidence": preflight_evidence,
        "env_sources": env_sources,
        "checks": check_dicts,
        "command_probes": command_results,
        "read_only_discovery": readonly_discovery,
        "nemoclaw_action_packet": nemoclaw_action_packet,
    }
    execution_plan = build_milestone2_execution_plan(report)
    report["post_approval_receipts"] = load_post_approval_receipts(post_approval_receipts_path, execution_plan)
    if report["post_approval_receipts"]["loaded"] and report["post_approval_receipts"]["status"] != "valid":
        report["ready"] = False
        report["status"] = "needs_setup"
        if "post_approval_receipts_valid" not in report["required_failures"]:
            report["required_failures"].append("post_approval_receipts_valid")
        report["area_status"]["post_approval_receipts"] = "fail"
    elif report["post_approval_receipts"]["loaded"]:
        report["area_status"]["post_approval_receipts"] = "pass"
    return report


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# VoiceOps Provisioning Readiness Probe",
        "",
        f"- Ready: {'yes' if report['ready'] else 'no'}",
        "- Mode: non-mutating, bounded, PATH/env presence by default; version/help probes only when explicitly enabled",
        f"- Read-only discovery requested: {'yes' if report['read_only_discovery']['run_requested'] else 'no'}",
        f"- Read-only discovery status: {report['read_only_discovery']['status']}",
        "- Read-only discovery auth context: isolated HOME; does not prove the operator's normal CLI auth state",
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
    lines.extend(
        [
            "",
            "## Preflight Evidence",
            "",
            f"- Loaded: {'yes' if report['preflight_evidence']['loaded'] else 'no'}",
            f"- Path: `{report['preflight_evidence']['path'] or 'not provided'}`",
            "- Redaction policy: references only; no raw secrets, cards, tokens, or full phone numbers",
            "- Missing fields: "
            + (
                ", ".join(report["preflight_evidence"]["missing_fields"])
                if report["preflight_evidence"]["missing_fields"]
                else "none"
            ),
            "- Validation issues: "
            + (
                ", ".join(_redact(issue) for issue in report["preflight_evidence"]["validation_issues"])
                if report["preflight_evidence"]["validation_issues"]
                else "none"
            ),
            "",
            "## Post-Approval Receipts",
            "",
            f"- Loaded: {'yes' if report['post_approval_receipts']['loaded'] else 'no'}",
            f"- Status: {report['post_approval_receipts']['status']}",
            f"- Validation issues: "
            + (
                ", ".join(_redact(issue) for issue in report["post_approval_receipts"]["validation_issues"])
                if report["post_approval_receipts"]["validation_issues"]
                else "none"
            ),
        ]
    )
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
    lines.extend(["", "## Read-Only Discovery", ""])
    discovery = report["read_only_discovery"]
    lines.extend(
        [
            f"- Status: {discovery['status']}",
            f"- Requested: {'yes' if discovery['run_requested'] else 'no'}",
            f"- Does not grant approval: {'yes' if discovery['does_not_grant_approval'] else 'no'}",
            f"- Redacted outputs only: {'yes' if discovery['redacted_outputs_only'] else 'no'}",
            f"- Missing probes: {', '.join(discovery['missing_probe_ids']) if discovery['missing_probe_ids'] else 'none'}",
        ]
    )
    for probe in discovery["probes"]:
        executed = "executed" if probe["executed"] else "not executed"
        lines.append(f"- {probe['probe_id']}: {probe['status']} ({executed}) `{' '.join(probe['argv'])}`")
    lines.append("")
    return "\n".join(lines)


def build_setup_closure_plan(report: dict[str, Any]) -> dict[str, Any]:
    """Describe the exact evidence needed to close Milestone 2 setup gaps."""

    requirements: list[dict[str, Any]] = []
    for check in report["checks"]:
        if not check["required"]:
            continue
        closure = SETUP_CLOSURE_REQUIREMENTS[check["check_id"]]
        satisfied = check["status"] == "pass"
        requirements.append(
            {
                "check_id": check["check_id"],
                "area": check["area"],
                "category": closure["category"],
                "status": check["status"],
                "closure_state": "satisfied" if satisfied else "needs_setup",
                "detail": check["detail"],
                "operator_action": closure["operator_action"],
                "next_step": check["next_step"],
                "accepted_binaries": closure["accepted_binaries"],
                "accepted_env_keys": closure["accepted_env_keys"],
                "safe_probe_commands": closure["safe_probe_commands"],
                "proof": closure["proof"],
                "required_fields": check.get("evidence", {}).get("required_fields", []),
                "missing_fields": check.get("evidence", {}).get("missing_fields", []),
                "evidence_artifacts": [
                    "provisioning-readiness.json",
                    "provisioning-readiness.md",
                    "provisioning-preflight-evidence.template.json",
                    "provisioning-preflight-evidence.example.json",
                    "provisioning-preflight-evidence.manifest.example.json",
                    "provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json",
                    "setup-closure-plan.json",
                ],
            }
        )
    return {
        "generated_at": _utc_now(),
        "schema_version": "voiceops.milestone2.setup_closure.v1",
        "artifact_id": "voiceops-m2-setup-closure",
        "milestone": "milestone_2_real_spend_and_provisioning_preflight",
        "ready": report["ready"],
        "remaining_failures": report["required_failures"],
        "source_readiness_artifact": "provisioning-readiness.json",
        "preflight_evidence_template": "provisioning-preflight-evidence.template.json",
        "preflight_evidence_example": "provisioning-preflight-evidence.example.json",
        "preflight_evidence_manifest_example": "provisioning-preflight-evidence.manifest.example.json",
        "preflight_evidence_scaffold_manifest": (
            "provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json"
        ),
        "evidence_contract": {
            "preflight_schema_version": PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
            "manifest_schema_version": PREFLIGHT_EVIDENCE_MANIFEST_SCHEMA_VERSION,
            "required_sections": list(PREFLIGHT_EVIDENCE_SECTIONS),
            "required_section_field": "source_artifact",
            "required_section_provenance_fields": [
                "source_artifact_kind",
                "source_artifact_sha256",
                "source_artifact_redacted_at",
                "collector_attestation",
            ],
            "collector_attestation_required_fields": list(COLLECTOR_ATTESTATION_REQUIRED_FIELDS),
            "placeholder_collector_attestation_accepted": False,
            "source_artifact_kind": PREFLIGHT_SOURCE_ARTIFACT_KIND,
            "source_artifacts_must_exist": True,
            "source_artifact_sha256_must_match": True,
            "source_artifacts_must_be_redacted_json": True,
            "source_artifact_resolution": "package-contained paths relative to the supplied evidence/manifest file; absolute paths, user-home expansion, parent traversal, symlink escapes, and process cwd fallback are rejected",
            "manifest_report_resolution": "package-contained paths relative to the supplied manifest file; absolute paths, user-home expansion, parent traversal, symlink escapes, and process cwd fallback are rejected",
            "example_only_accepted": False,
            "secret_like_values_accepted": False,
            "full_phone_numbers_accepted": False,
            "read_only_discovery_schema_version": "voiceops.milestone2.read_only_discovery.v1",
            "read_only_discovery_grants_approval": False,
            "post_approval_receipts_schema_version": POST_APPROVAL_RECEIPTS_SCHEMA_VERSION,
            "post_approval_collector_attestation_required": True,
            "post_approval_collector_attestation_redacted_sha256_must_match": True,
            "post_approval_collector_attestation_required_fields": list(COLLECTOR_ATTESTATION_REQUIRED_FIELDS),
            "post_approval_decision_provenance_required_fields": [
                "decision",
                "decision_by",
                "decision_at",
                "approval_decision_ref",
                "approval_decision_sha256",
            ],
            "post_approval_attempted_execution_decision": "approve_once",
            "post_approval_linkage_ids_must_be_unique": [
                "credential_locations[].credential_ref_id",
                "rollback_receipts[].rollback_ref",
                "audit_events[].audit_event_id",
            ],
        },
        "mode": {
            "artifact_only": True,
            "headless": True,
            "non_mutating": True,
            "secret_values_emitted": False,
        },
        "safety": {
            "live_spend": False,
            "provider_provisioning": False,
            "credential_retrieval": False,
            "outbound_phone_calls": False,
            "account_mutation": False,
        },
        "rerun_commands": {
            "presence_only": "uv run python scripts/voiceops_provisioning_probe.py --output-dir artifacts/voiceops-provisioning/current --env-file .env",
            "bounded_version_help": "uv run python scripts/voiceops_provisioning_probe.py --output-dir artifacts/voiceops-provisioning/current --env-file .env --run-command-probes",
            "read_only_discovery": "uv run python scripts/voiceops_provisioning_probe.py --output-dir artifacts/voiceops-provisioning/current --env-file .env --run-readonly-discovery",
            "validate_nemoclaw_action_packet": "uv run python scripts/voiceops_provisioning_probe.py --output-dir artifacts/voiceops-provisioning/current --env-file .env --no-command-probes --nemoclaw-action-packet artifacts/hackathon-voiceops-demo/current/nemoclaw-action-packet.json",
            "with_preflight_evidence": "uv run python scripts/voiceops_provisioning_probe.py --output-dir artifacts/voiceops-provisioning/current --env-file .env --preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-evidence.json",
            "with_preflight_manifest": "uv run python scripts/voiceops_provisioning_probe.py --output-dir artifacts/voiceops-provisioning/current --env-file .env --preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json",
            "plan_index": "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts --output-dir artifacts/voiceops-plan/current --env-file .env --provisioning-preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-evidence.json",
            "plan_index_manifest": "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts --output-dir artifacts/voiceops-plan/current --env-file .env --provisioning-preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json",
            "plan_index_manifest_and_post_approval_receipts": "uv run python scripts/voiceops_plan_run.py --artifact-root artifacts --output-dir artifacts/voiceops-plan/current --env-file .env --read-only-discovery-evidence artifacts/voiceops-provisioning/current/read-only-discovery.manifest.json --provisioning-preflight-evidence artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json --post-approval-receipts artifacts/voiceops-provisioning/current/post-approval-receipts.json",
            "refresh_preflight_source_hashes": "uv run python scripts/voiceops_provisioning_probe.py --refresh-preflight-source-hashes artifacts/voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json",
            "source_artifact_sha256": "shasum -a 256 path/to/redacted-source-artifact.json",
        },
        "operator_must_not": [
            "paste secret values into chat or artifact files",
            "use /Users/jethac/.hermes/hermes-agent as an env-file source",
            "run mutating Stripe Projects, Link spend, provider provisioning, or phone-call commands before approval",
        ],
        "requirements": requirements,
    }


def _setup_closure_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# VoiceOps Milestone 2 Setup Closure Plan",
        "",
        f"- Ready: {'yes' if plan['ready'] else 'no'}",
        f"- Remaining failures: {', '.join(plan['remaining_failures']) if plan['remaining_failures'] else 'none'}",
        "- Mode: artifact-only, headless, non-mutating, no secret values emitted",
        f"- Source readiness: `{plan['source_readiness_artifact']}`",
        "",
        "## Evidence Artifacts",
        "",
        f"- Template: `{plan['preflight_evidence_template']}`",
        f"- Example: `{plan['preflight_evidence_example']}`",
        f"- Manifest example: `{plan['preflight_evidence_manifest_example']}`",
        f"- Two-layer scaffold: `{plan['preflight_evidence_scaffold_manifest']}`",
        "",
        "## Evidence Contract",
        "",
    ]
    for key, value in sorted(plan["evidence_contract"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
        "## Rerun Commands",
        "",
        ]
    )
    for label, command in plan["rerun_commands"].items():
        lines.append(f"- {label}: `{command}`")
    lines.extend(["", "## Do Not", ""])
    lines.extend(f"- {item}" for item in plan["operator_must_not"])
    lines.extend(["", "## Requirements", ""])
    for requirement in plan["requirements"]:
        lines.extend(
            [
                f"### {requirement['check_id']}",
                "",
                f"- Area: {requirement['area']}",
                f"- Category: {requirement['category']}",
                f"- Status: {requirement['status']}",
                f"- Closure state: {requirement['closure_state']}",
                f"- Operator action: {_redact(requirement['operator_action'])}",
                f"- Proof: {_redact(requirement['proof'])}",
                f"- Accepted binaries: {', '.join(requirement['accepted_binaries']) or 'none'}",
                f"- Accepted env keys: {', '.join(requirement['accepted_env_keys']) or 'none'}",
            ]
        )
        if requirement["required_fields"]:
            lines.append(f"- Required fields: {', '.join(f'`{field}`' for field in requirement['required_fields'])}")
        if requirement["missing_fields"]:
            lines.append(f"- Missing fields: {', '.join(f'`{field}`' for field in requirement['missing_fields'])}")
        if requirement["safe_probe_commands"]:
            commands = [" ".join(command) for command in requirement["safe_probe_commands"]]
            lines.append(f"- Safe probe commands: {', '.join(f'`{command}`' for command in commands)}")
        else:
            lines.append("- Safe probe commands: none")
        lines.append("")
    return "\n".join(lines)


def _safe_command_manifest_json() -> dict[str, Any]:
    return {
        "schema_version": "voiceops.milestone2.safe_command_manifest.v1",
        "policy": "Default mode executes no vendor commands. If enabled, isolated HOME version/help probes and separately opted-in exact read-only discovery commands are allowed. Mutating, spend, provisioning, credential, and call commands are refused.",
        "blocked_patterns": MUTATING_COMMAND_PATTERNS,
        "commands": [asdict(probe) for probe in _command_manifest()],
        "version_help_commands": [asdict(probe) for probe in _command_manifest()],
        "read_only_discovery_commands": [asdict(probe) for probe in _readonly_discovery_manifest()],
        "read_only_discovery_policy": {
            "requires_explicit_flag": "--run-readonly-discovery",
            "exact_argv_allowlist_only": True,
            "does_not_grant_approval": True,
            "redacted_outputs_only": True,
        },
    }


def _command_sha256(command: str) -> str:
    return hashlib.sha256(command.encode("utf-8")).hexdigest()


def build_kame_evidence_gate() -> dict[str, Any]:
    return {
        "schema_version": "voiceops.kame_evidence_gate.v1",
        "requires_promoted_evidence": True,
        "accepted_authorities": list(KAME_PROMOTED_AUTHORITIES),
        "rejected_authorities": list(KAME_REJECTED_AUTHORITIES),
        "hypotheses_allowed_for_action": False,
        "merge_key_fields": ["turn_id", "audio_segment_ref"],
        "raw_audio_required_for_full_kame": True,
        "text_only_bridge_mode": "degraded_requires_oracle_responsibility",
    }


def build_registered_core_tool_schema_defs() -> tuple[list[dict[str, Any]], list[str]]:
    """Return actual registered Hermes core tool schemas for disclosure proof.

    This intentionally bypasses availability checks. VoiceOps tool-pressure
    proof is about the broad core schema surface that would enter context, not
    whether this host currently has optional env-gated backends configured.
    """

    from toolsets import _HERMES_CORE_TOOLS
    from tools.registry import discover_builtin_tools, registry

    discover_builtin_tools()
    tool_defs: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in sorted(_HERMES_CORE_TOOLS):
        entry = registry.get_entry(name)
        if entry is None:
            missing.append(name)
            continue
        schema_with_name = {**entry.schema, "name": entry.name}
        if entry.dynamic_schema_overrides is not None:
            try:
                overrides = entry.dynamic_schema_overrides()
            except Exception:
                overrides = None
            if isinstance(overrides, dict):
                schema_with_name.update(overrides)
        tool_defs.append({"type": "function", "function": schema_with_name})
    return tool_defs, missing


def build_tool_disclosure_proof() -> dict[str, Any]:
    from toolsets import _HERMES_CORE_TOOLS
    from tools.tool_search import bridge_tool_schemas, estimate_tokens_from_schemas

    core_tools = sorted(_HERMES_CORE_TOOLS)
    visible_tools = list(TOOL_DISCLOSURE_BRIDGE_TOOL_NAMES)
    core_tool_defs, missing_core_tools = build_registered_core_tool_schema_defs()
    input_schema_tokens = estimate_tokens_from_schemas(core_tool_defs)
    visible_schema_tokens = estimate_tokens_from_schemas(bridge_tool_schemas(len(core_tools)))
    return {
        "schema_version": "voiceops.tool_disclosure_proof.v1",
        "ok": not missing_core_tools and len(core_tool_defs) == len(core_tools),
        "scenario": "all_core_tools_deferred_behind_tool_search",
        "schema_source": "registered_core_tool_schemas",
        "representative_schema": False,
        "missing_registered_core_tools": missing_core_tools,
        "config": {"enabled": "on", "defer_core": "all"},
        "input_core_tools": core_tools,
        "visible_tool_names": visible_tools,
        "visible_non_bridge_tool_names": [],
        "hidden_core_tool_names": core_tools,
        "bridge_tool_names": visible_tools,
        "input_core_tool_count": len(core_tools),
        "hidden_core_tool_count": len(core_tools),
        "bridge_tool_count": len(visible_tools),
        "core_tools_hidden_all": True,
        "broad_core_tools_visible": False,
        "deferred_count": len(core_tools),
        "input_schema_tokens": input_schema_tokens,
        "visible_schema_tokens": visible_schema_tokens,
        "token_reduction_estimate": max(0, input_schema_tokens - visible_schema_tokens),
        "external_test_refs": list(TOOL_DISCLOSURE_TEST_REFS),
    }


def build_kame_action_evidence(
    action_id: str,
    source_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_context = source_context or {}
    source_voice_session_id = str(source_context.get("source_voice_session_id") or DEFAULT_SOURCE_VOICE_SESSION_ID)
    source_oracle_job_id = str(source_context.get("source_oracle_job_id") or DEFAULT_SOURCE_ORACLE_JOB_ID)
    fields = KAME_ACTION_PROMOTED_FIELDS.get(
        action_id,
        ("user_request", "oracle_action_plan", "action_rationale"),
    )
    promoted_fields = {
        "user_request": {
            "evidence_label": "interpreter_promoted",
            "source": "gemma_raw_audio_interpreter",
            "ref": "interpreter_evidence.corrected_transcript",
        },
        "oracle_action_plan": {
            "evidence_label": "oracle_promoted",
            "source": "hermes_active_model",
            "ref": "oracle_job.promoted_action_plan",
        },
        "provider_selection": {
            "evidence_label": "oracle_promoted",
            "source": "hermes_active_model",
            "ref": "oracle_job.provider_selection",
        },
        "spend_reason": {
            "evidence_label": "oracle_promoted",
            "source": "hermes_active_model",
            "ref": "oracle_job.spend_reason",
        },
        "phone_handoff_context": {
            "evidence_label": "oracle_promoted",
            "source": "hermes_active_model",
            "ref": "oracle_job.phone_handoff_context",
        },
        "channel_policy": {
            "evidence_label": "oracle_promoted",
            "source": "hermes_active_model",
            "ref": "oracle_job.channel_policy",
        },
        "action_rationale": {
            "evidence_label": "oracle_promoted",
            "source": "hermes_active_model",
            "ref": "oracle_job.action_rationale",
        },
    }
    return {
        "schema_version": "voiceops.kame_action_evidence.v1",
        "action_id": action_id,
        "turn_id": str(source_context.get("turn_id") or "voiceops-demo-turn-budget"),
        "audio_segment_ref": str(
            source_context.get("audio_segment_ref") or "artifact://voiceops-demo/discord-budget-turn.wav"
        ),
        "source_voice_session_id": source_voice_session_id,
        "source_oracle_job_id": source_oracle_job_id,
        "required_promotions": list(KAME_PROMOTED_AUTHORITIES),
        "hypotheses_allowed_for_action": False,
        "transcript_hypotheses_promoted": False,
        "hypothesis_sources": [
            "reflex_transcript_hypothesis",
            "s2s_transcript_hypothesis",
            "classic_asr_hypothesis",
        ],
        "promotion_required_before": list(fields),
        "promoted_fields": {
            field: promoted_fields[field]
            for field in fields
        },
    }


def _execution_approval_contract(
    *,
    action_id: str,
    command: str,
    required_preflight_gates: list[str],
    approval_artifact: str,
    ttl_seconds: int = 1800,
) -> dict[str, Any]:
    return {
        "approval_id": f"voiceops-m2-{action_id}",
        "action_id": action_id,
        "approval_channel": "discord_voice_operator_confirmation",
        "approval_artifact": approval_artifact,
        "approved_by_ref": None,
        "command_sha256": _command_sha256(command),
        "required_preflight_gates": required_preflight_gates,
        "allowed_decisions": ["approve_once", "deny", "hold"],
        "default_decision": "hold",
        "ttl_seconds": ttl_seconds,
        "status": "not_approved",
    }


def _expected_post_approval_evidence(action: Mapping[str, Any]) -> dict[str, Any]:
    action_id = str(action["action_id"])
    receipt_ref = str(action["expected_receipt_ref"])
    rollback_ref = str(action["rollback_ref"])
    approval_contract = action["approval_contract"]
    credential_location_ref = action.get("credential_location_ref")
    required_schemas = ["receipt_schema", rollback_ref]
    if credential_location_ref:
        required_schemas.append("credential_location_schema")
    return {
        "action_id": action_id,
        "approval_id": approval_contract["approval_id"],
        "approval_contract_ref": f"approval_contracts.{action_id}",
        "command_sha256": approval_contract["command_sha256"],
        "execution_status": "not_executed",
        "expected_receipt_ref": receipt_ref,
        "receipt": None,
        "credential_location_ref": credential_location_ref,
        "credential_location": None,
        "rollback_ref": rollback_ref,
        "rollback_receipt": None,
        "lineage": dict(action.get("lineage") or {}),
        "required_schemas": required_schemas,
        "audit_update_required": True,
        "secret_policy": "redacted references only; no raw tokens, card data, credentials, or full phone numbers",
    }


def _lineage_for_action(demo_refs: Mapping[str, Any], action_id: str) -> dict[str, Any]:
    return {
        "source_voice_session_id": str(
            demo_refs.get("source_voice_session_id") or DEFAULT_SOURCE_VOICE_SESSION_ID
        ),
        "source_oracle_job_id": str(
            demo_refs.get("source_oracle_job_id") or DEFAULT_SOURCE_ORACLE_JOB_ID
        ),
        "parent_audit_event_id": str(
            demo_refs.get(f"{action_id}_parent_audit_event_id")
            or ACTION_PARENT_AUDIT_EVENT_IDS.get(action_id)
            or ""
        ),
    }


def _selected_voip_provider_candidate(report: Mapping[str, Any]) -> dict[str, Any]:
    preflight = report.get("preflight_evidence")
    if isinstance(preflight, Mapping) and preflight.get("loaded") and not preflight.get("validation_issues"):
        path_text = str(preflight.get("path") or "")
        candidate = str(preflight.get("voip_provider_candidate") or "").strip()
        if VOIP_PROVIDER_CANDIDATE_RE.fullmatch(candidate):
            return {
                "candidate": candidate,
                "source": "preflight_evidence",
                "source_artifact": path_text or None,
                "default_used": candidate == DEFAULT_VOIP_PROVIDER_CANDIDATE,
            }
    return {
        "candidate": DEFAULT_VOIP_PROVIDER_CANDIDATE,
        "source": "default",
        "source_artifact": None,
        "default_used": True,
    }


def build_milestone2_execution_plan(report: dict[str, Any]) -> dict[str, Any]:
    checks = {check["check_id"]: check for check in report["checks"]}
    demo_refs = {
        "voiceops_demo": "voiceops-demo.json",
        "nemoclaw_packet": "nemoclaw-action-packet.json",
        "phone_context": "phone-context.json",
        "audit_ledger": "audit-ledger.jsonl",
        "stripe_actions_dry_run": "stripe-actions-dry-run.sh",
    }
    supplied_demo_refs = report.get("demo_refs")
    if isinstance(supplied_demo_refs, Mapping):
        demo_refs.update(
            {
                key: value
                for key, value in supplied_demo_refs.items()
                if isinstance(key, str) and isinstance(value, (str, int))
            }
        )
    budget_cap_cents = int(demo_refs.get("budget_cents") or 20_000)
    approval_threshold_cents = int(demo_refs.get("approval_threshold_cents") or 1_000)
    queued_cents = int(demo_refs.get("queued_cents") or 7_400)
    held_cents = int(demo_refs.get("held_cents") or 0)
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
    provider_selection = _selected_voip_provider_candidate(report)
    provision_command = f"stripe projects add {provider_selection['candidate']}"
    spend_command = (
        "link-cli spend-request create --merchant-name ExampleOps "
        "--merchant-url https://example.invalid --amount 4900 --request-approval"
    )
    call_command = "queue outbound call --context phone-context.json"
    publish_command = "post redacted approval and handoff status to configured channels"
    approval_required_actions = [
        {
            "action_id": "provision-voip-provider",
            "provider": "stripe-projects",
            "command": provision_command,
            "status": "blocked_until_explicit_approval",
            "requires": ["stripe_cli", "stripe_projects_cli", "mpp_agent"],
            "approval_artifact": "nemoclaw-action-packet.json",
            "expected_receipt_ref": "receipts.provision_voip_provider",
            "credential_location_ref": "credential_locations.voip_provider",
            "credential_location_required": True,
            "credential_location_schema_ref": "credential_location_schema",
            "rollback_ref": "rollback_plan.deprovision_voip_provider",
            "approval_contract": _execution_approval_contract(
                action_id="provision-voip-provider",
                command=provision_command,
                required_preflight_gates=["stripe_cli", "stripe_projects_cli", "mpp_agent"],
                approval_artifact="nemoclaw-action-packet.json",
            ),
        },
        {
            "action_id": "buy-service-credit",
            "provider": "stripe-link-cli",
            "command": spend_command,
            "status": "blocked_until_explicit_approval",
            "requires": ["stripe_link_cli", "mpp_agent"],
            "approval_artifact": "nemoclaw-action-packet.json",
            "expected_receipt_ref": "receipts.buy_service_credit",
            "credential_location_ref": "credential_locations.stripe_link",
            "credential_location_required": True,
            "credential_location_schema_ref": "credential_location_schema",
            "rollback_ref": "rollback_plan.refund_or_cancel_service_credit",
            "approval_contract": _execution_approval_contract(
                action_id="buy-service-credit",
                command=spend_command,
                required_preflight_gates=["stripe_link_cli", "mpp_agent"],
                approval_artifact="nemoclaw-action-packet.json",
                ttl_seconds=900,
            ),
        },
        {
            "action_id": "call-user-phone",
            "provider": "voiceops-phone-bridge",
            "command": call_command,
            "status": "blocked_until_explicit_approval",
            "requires": ["phone_target", "phone_provider", "mpp_agent"],
            "approval_artifact": "phone-context.json",
            "expected_receipt_ref": "receipts.call_user_phone",
            "credential_location_ref": "credential_locations.phone_bridge",
            "credential_location_required": True,
            "credential_location_schema_ref": "credential_location_schema",
            "rollback_ref": "rollback_plan.cancel_or_end_phone_handoff",
            "approval_contract": _execution_approval_contract(
                action_id="call-user-phone",
                command=call_command,
                required_preflight_gates=["phone_target", "phone_provider", "mpp_agent", "channel_policy"],
                approval_artifact="phone-context.json",
                ttl_seconds=900,
            ),
        },
        {
            "action_id": "publish-status",
            "provider": "hermes-gateway",
            "command": publish_command,
            "status": "blocked_until_explicit_approval",
            "requires": ["channel_policy", "mpp_agent"],
            "approval_artifact": "channel-policy.json",
            "expected_receipt_ref": "receipts.publish_status",
            "credential_location_ref": None,
            "credential_location_required": False,
            "credential_location_schema_ref": None,
            "rollback_ref": "rollback_plan.correct_or_remove_status_message",
            "approval_contract": _execution_approval_contract(
                action_id="publish-status",
                command=publish_command,
                required_preflight_gates=["channel_policy", "mpp_agent"],
                approval_artifact="channel-policy.json",
                ttl_seconds=900,
            ),
        },
    ]
    for action in approval_required_actions:
        action["approval_id"] = action["approval_contract"]["approval_id"]
        action["command_sha256"] = action["approval_contract"]["command_sha256"]
        action["lineage"] = _lineage_for_action(demo_refs, str(action["action_id"]))
        action["kame_evidence"] = build_kame_action_evidence(str(action["action_id"]), demo_refs)
        action["tool_disclosure_ref"] = "tool_disclosure"
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
        "kame_evidence_gate": build_kame_evidence_gate(),
        "tool_disclosure": build_tool_disclosure_proof(),
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
            "preflight_evidence_template": "provisioning-preflight-evidence.template.json",
            "required_evidence": [
                "stripe_projects_account",
                "stripe_link_approval_capability",
                "mpp_approval_boundary",
                "phone_provider_account",
                "credential_location_reference",
                "rollback_owner_refs",
            ],
            "run_command_probes_default": False,
            "active_probe_policy": "version_help_only",
            "run_command_probes_does_not_grant_approval": True,
        },
        "demo_refs": demo_refs,
        "spend_policy": {
            "currency": "usd",
            "budget_cap_cents": budget_cap_cents,
            "approval_threshold_cents": approval_threshold_cents,
            "queued_cents": queued_cents,
            "held_cents": held_cents,
            "status": "no_live_spend_without_explicit_approval",
        },
        "readiness_gates": gates,
        "provider_selection": {
            "voip_provider_candidate": provider_selection["candidate"],
            "source": provider_selection["source"],
            "source_artifact": provider_selection["source_artifact"],
            "default_used": provider_selection["default_used"],
            "command_sha256": _command_sha256(provision_command),
        },
        "read_only_discovery": [
            {
                "step_id": "stripe-projects-catalog-discovery",
                "command": "stripe projects list --limit 10",
                "status": "not_executed",
                "requires": ["stripe_cli", "stripe_projects_cli", "mpp_agent"],
                "purpose": "Confirm available Projects catalog entries before choosing a VoIP provider.",
                "allowed_after": "operator opts into a read-only discovery run",
                "records_to": "audit-ledger.read-only-discovery.jsonl",
            },
            {
                "step_id": "stripe-link-auth-status",
                "command": "link-cli auth status",
                "status": "not_executed",
                "requires": ["stripe_link_cli", "mpp_agent"],
                "purpose": "Confirm Link approval capability without creating a spend request.",
                "allowed_after": "operator opts into a read-only auth-status run",
                "records_to": "audit-ledger.read-only-discovery.jsonl",
            },
        ],
        "approval_required_actions": approval_required_actions,
        "approval_contracts": {
            action["action_id"]: action["approval_contract"]
            for action in approval_required_actions
        },
        "receipts": {
            action["expected_receipt_ref"].split(".", 1)[1]: {
                "status": "not_executed",
                "receipt": None,
                "schema_ref": "receipt_schema",
                "action_id": action["action_id"],
                "approval_id": action["approval_id"],
                "command_sha256": action["command_sha256"],
                "credential_location_ref": action["credential_location_ref"],
                "rollback_ref": action["rollback_ref"],
                "approval_contract_ref": f"approval_contracts.{action['action_id']}",
                "lineage": dict(action["lineage"]),
            }
            for action in approval_required_actions
        },
        "credential_locations": {
            str(action["credential_location_ref"]).split(".", 1)[1]: {
                "status": "not_created",
                "credential_location": None,
                "schema_ref": "credential_location_schema",
                "action_id": action["action_id"],
            }
            for action in approval_required_actions
            if action.get("credential_location_ref")
        },
        "expected_post_approval_evidence": {
            action["action_id"]: _expected_post_approval_evidence(action)
            for action in approval_required_actions
        },
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
                "approval_id",
                "provider",
                "status",
                "decision",
                "decision_by",
                "decision_at",
                "approval_decision_ref",
                "approval_decision_sha256",
                "executed_at",
                "command_sha256",
                "amount_cents",
                "currency",
                "approval_artifact",
                "external_reference",
                "credential_location_ref",
                "rollback_ref",
                "audit_event_id",
                "source_voice_session_id",
                "source_oracle_job_id",
                "parent_audit_event_id",
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
            "correct_or_remove_status_message": [
                "Record the message id from the status-post receipt.",
                "Post a correction or remove the message according to channel policy.",
                "Append the correction/removal receipt to audit-ledger.jsonl.",
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
    provider_selection = plan["provider_selection"]
    lines.extend(
        [
            "## Provider Selection",
            "",
            f"- VoIP provider candidate: `{provider_selection['voip_provider_candidate']}`",
            f"- Source: {provider_selection['source']}",
            f"- Default used: {'yes' if provider_selection['default_used'] else 'no'}",
            f"- Provision command SHA-256: `{provider_selection['command_sha256']}`",
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


def _read_only_discovery_markdown(discovery: dict[str, Any]) -> str:
    lines = [
        "# VoiceOps Read-Only Discovery",
        "",
        f"- Schema: `{discovery['schema_version']}`",
        f"- Status: {discovery['status']}",
        f"- Requested: {'yes' if discovery['run_requested'] else 'no'}",
        f"- Non-mutating: {'yes' if discovery['non_mutating'] else 'no'}",
        f"- Does not grant approval: {'yes' if discovery['does_not_grant_approval'] else 'no'}",
        f"- Redacted outputs only: {'yes' if discovery['redacted_outputs_only'] else 'no'}",
        f"- Timeout seconds: {discovery.get('timeout_seconds') if discovery.get('timeout_seconds') is not None else 'not applicable'}",
        f"- Failed probes: {', '.join(discovery['failed_probe_ids']) if discovery['failed_probe_ids'] else 'none'}",
        f"- Timed-out probes: {', '.join(discovery.get('timed_out_probe_ids', [])) if discovery.get('timed_out_probe_ids') else 'none'}",
        f"- Missing probes: {', '.join(discovery['missing_probe_ids']) if discovery['missing_probe_ids'] else 'none'}",
        "",
        "## Probes",
        "",
    ]
    for probe in discovery["probes"]:
        executed = "executed" if probe["executed"] else "not executed"
        lines.extend(
            [
                f"### {probe['probe_id']}",
                "",
                f"- Area: {probe['area']}",
                f"- Status: {probe['status']}",
                f"- Command: `{' '.join(probe['argv'])}`",
                f"- Execution: {executed}",
                f"- Timed out: {'yes' if probe.get('timed_out') else 'no'}",
                f"- Purpose: {_redact(probe['purpose'])}",
                "",
            ]
        )
    return "\n".join(lines)


def _read_only_discovery_manifest(discovery: dict[str, Any], *, report_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "voiceops.milestone2.read_only_discovery_manifest.v1",
        "generated_at": _utc_now(),
        "report": "read-only-discovery.json",
        "report_sha256": report_sha256,
        "markdown": "read-only-discovery.md",
        "audit_ledger": "audit-ledger.read-only-discovery.jsonl",
        "run_requested": discovery["run_requested"],
        "status": discovery["status"],
        "failed_probe_ids": discovery["failed_probe_ids"],
        "timed_out_probe_ids": discovery.get("timed_out_probe_ids", []),
        "missing_probe_ids": discovery["missing_probe_ids"],
        "timeout_seconds": discovery.get("timeout_seconds"),
        "does_not_grant_approval": discovery["does_not_grant_approval"],
        "redacted_outputs_only": discovery["redacted_outputs_only"],
        "probes": [
            {
                "probe_id": probe["probe_id"],
                "command": probe["argv"],
                "status": probe["status"],
                "executed": probe["executed"],
                "timed_out": probe.get("timed_out", False),
            }
            for probe in discovery["probes"]
        ],
    }


def _read_only_discovery_ledger_rows(discovery: dict[str, Any]) -> list[dict[str, Any]]:
    if not discovery["run_requested"]:
        return []
    rows: list[dict[str, Any]] = []
    for probe in discovery["probes"]:
        if not probe["executed"]:
            continue
        rows.append(
            {
                "audit_event_id": f"readonly-{probe['probe_id']}",
                "event_type": "read_only_discovery_probe",
                "probe_id": probe["probe_id"],
                "command_sha256": _command_sha256(" ".join(probe["argv"])),
                "status": probe["status"],
                "executed": probe["executed"],
                "timed_out": probe.get("timed_out", False),
                "does_not_grant_approval": True,
                "redacted_outputs_only": True,
                "artifact_ref": "read-only-discovery.json",
            }
        )
    return rows


def build_post_approval_receipts_template(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": POST_APPROVAL_RECEIPTS_SCHEMA_VERSION,
        "redaction_policy": "references and redacted summaries only; no raw credentials, tokens, card data, or full phone numbers",
        "collector_attestation": None,
        "receipts": [],
        "credential_locations": [],
        "rollback_receipts": [],
        "audit_events": [],
        "expected_actions": sorted(plan.get("approval_contracts", {}).keys()),
        "notes": "Populate only after explicit approval and execution; this validator is read-only and does not execute provider actions.",
    }


def build_post_approval_receipts_example(plan: Mapping[str, Any]) -> dict[str, Any]:
    example = build_post_approval_receipts_template(plan)
    example["example_only"] = True
    action_estimates = {
        str(step.get("step_id") or ""): step.get("estimated_cents")
        for step in plan.get("execution_steps", [])
        if isinstance(step, Mapping)
    }
    actions = list(plan["approval_required_actions"])
    example["receipts"] = []
    example["credential_locations"] = []
    example["rollback_receipts"] = []
    example["audit_events"] = []
    for action in actions:
        action_id = str(action["action_id"])
        action_ref = action_id.replace("-", "_")
        receipt_id = f"receipt-example-{action_id}"
        audit_event_id = f"audit-example-{action_id}"
        credential_ref = action.get("credential_location_ref")
        estimated_cents = action_estimates.get(action_id, 0)
        if not isinstance(estimated_cents, int):
            estimated_cents = 0
        example["receipts"].append(
            {
                "receipt_id": receipt_id,
                "action_id": action_id,
                "approval_id": action["approval_id"],
                "provider": action["provider"],
                "status": "executed",
                "decision": "approve_once",
                "decision_by": "operator-ref-demo",
                "decision_at": "2026-06-29T00:00:00Z",
                "approval_decision_ref": f"approval-decision-ref-demo-{action_id}",
                "approval_decision_sha256": "a" * 64,
                "executed_at": "2026-06-29T00:00:30Z",
                "command_sha256": action["command_sha256"],
                "amount_cents": estimated_cents,
                "currency": "usd",
                "external_reference": f"provider-resource-ref-demo-{action_id}",
                "credential_location_ref": credential_ref,
                "rollback_ref": action["rollback_ref"],
                "audit_event_id": audit_event_id,
                **dict(action.get("lineage") or {}),
                "approval_artifact": action["approval_artifact"],
                "redacted_summary": "Example only; replace with real redacted receipt refs.",
            }
        )
        if action.get("credential_location_required"):
            example["credential_locations"].append(
                {
                    "credential_ref_id": credential_ref,
                    "provider": action["provider"],
                    "service_id": f"provider-resource-ref-demo-{action_id}",
                    "storage_backend": "provider_managed",
                    "secret_name_or_path": f"credential-location-ref-demo-{action_ref}",
                    "created_by_action_id": action_id,
                    "rotation_due": "2026-09-29T00:00:00Z",
                    "redacted": True,
                    "lineage": dict(action.get("lineage") or {}),
                }
            )
        example["rollback_receipts"].append(
            {
                "rollback_ref": action["rollback_ref"],
                "status": "not_run",
                "owner_ref": "operator-ref-demo",
                "notes": "Example only; rollback not needed for this sample.",
                "lineage": dict(action.get("lineage") or {}),
            }
        )
        example["audit_events"].append(
            {
                "audit_event_id": audit_event_id,
                "action_id": action_id,
                "receipt_id": receipt_id,
                "status": "executed",
                "provider": action["provider"],
                "artifact_ref": "post-approval-receipts.example.json",
                "operator_next_step": "Replace this example with real redacted post-approval evidence.",
                **dict(action.get("lineage") or {}),
            }
        )
    example["collector_attestation"] = _example_collector_attestation(
        section_name="post_approval_receipts",
        redacted_sha256="0" * 64,
    )
    return example


def write_post_approval_receipts_scaffold(output_dir: Path, plan: Mapping[str, Any]) -> dict[str, Path]:
    scaffold_dir = output_dir / "post-approval-receipts-scaffold"
    scaffold_dir.mkdir(parents=True, exist_ok=True)
    scaffold_path = scaffold_dir / "post-approval-receipts.json"
    payload = build_post_approval_receipts_example(plan)
    payload["notes"] = "Scaffold only; replace with real redacted post-approval receipts and remove every example_only marker."
    _write_json(scaffold_path, payload)
    return {"post_approval_receipts_scaffold": scaffold_path}


def _post_approval_receipts_validation_base(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "expected_action_ids": sorted(str(action["action_id"]) for action in plan["approval_required_actions"]),
        "receipt_count": 0,
        "credential_location_count": 0,
        "rollback_receipt_count": 0,
        "audit_event_count": 0,
        "ledger_rows": [],
    }


def load_post_approval_receipts(path: Path | None, plan: Mapping[str, Any]) -> dict[str, Any]:
    if path is None:
        return {
            **_post_approval_receipts_validation_base(plan),
            "loaded": False,
            "path": None,
            "status": "not_supplied",
            "validation_issues": [],
            "redaction_policy": "not_loaded",
        }
    resolved = path.expanduser().resolve(strict=False)
    if resolved == FORBIDDEN_ENV_ROOT or FORBIDDEN_ENV_ROOT in resolved.parents:
        raise ValueError(f"refusing to inspect forbidden Hermes worktree path: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            **_post_approval_receipts_validation_base(plan),
            "loaded": False,
            "path": str(path),
            "status": "not_found",
            "validation_issues": ["post_approval_receipts:file_not_found"],
            "redaction_policy": "references_only",
        }
    except json.JSONDecodeError as exc:
        return {
            **_post_approval_receipts_validation_base(plan),
            "loaded": False,
            "path": str(path),
            "status": "invalid",
            "validation_issues": [f"post_approval_receipts:json_parse_failed:{exc.msg}"],
            "redaction_policy": "references_only",
        }
    if not isinstance(payload, Mapping):
        return {
            **_post_approval_receipts_validation_base(plan),
            "loaded": False,
            "path": str(path),
            "status": "invalid",
            "validation_issues": ["post_approval_receipts:root_must_be_object"],
            "redaction_policy": "references_only",
        }
    report = validate_post_approval_receipts(payload, plan, receipt_path=resolved)
    report["loaded"] = True
    report["path"] = str(path)
    report["redaction_policy"] = "references_only"
    return report


def _post_approval_receipt_secret_issues(payload: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    forbidden_names = {"raw_secret", "raw_token", "raw_card_data", "raw_phone_number"}
    for path, value in _walk_strings(payload):
        name = path.rsplit(".", 1)[-1].split("[", 1)[0]
        if name in forbidden_names:
            issues.append(f"{path}:forbidden_raw_field")
            continue
        if name in {"approval_artifact", "artifact_ref"}:
            continue
        if BEARER_RE.search(value) or PREFLIGHT_SECRET_VALUE_RE.search(value) or SECRET_VALUE_RE.search(value):
            issues.append(f"{path}:secret-like value")
        elif not (name.endswith("_at") or name.endswith("_due") or name.endswith("_sha256")) and PHONE_RE.search(value):
            issues.append(f"{path}:phone-like value")
    return issues


def _post_approval_receipts_redacted_sha256(payload: Mapping[str, Any]) -> str:
    attested_payload = dict(payload)
    attested_payload.pop("collector_attestation", None)
    encoded = json.dumps(attested_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _duplicate_nonempty_field_values(items: Iterable[Any], field: str) -> list[str]:
    values = [
        str(item.get(field) or "")
        for item in items
        if isinstance(item, Mapping) and str(item.get(field) or "")
    ]
    return sorted(value for value in set(values) if values.count(value) > 1)


def _approval_decision_artifact_issues(
    receipt: Mapping[str, Any],
    *,
    receipt_id: str,
    receipt_path: Path | None,
) -> list[str]:
    if receipt_path is None:
        return []
    decision_ref = str(receipt.get("approval_decision_ref") or "").strip()
    if not decision_ref:
        return []
    issues: list[str] = []
    ref_issues = _relative_artifact_ref_issues(decision_ref, base_path=receipt_path)
    if ref_issues:
        return [f"post_approval_receipts:{receipt_id}:approval_decision_ref:{issue}" for issue in ref_issues]
    decision_path = _resolve_source_artifact_path(decision_ref, receipt_path)
    if not decision_path.exists():
        return [
            f"post_approval_receipts:{receipt_id}:approval_decision_ref:file_not_found",
            f"post_approval_receipts:{receipt_id}:approval_decision_ref:file_not_found_at:{decision_path.resolve(strict=False)}",
        ]
    try:
        decision_bytes = decision_path.read_bytes()
    except OSError as exc:
        return [f"post_approval_receipts:{receipt_id}:approval_decision_ref:file_unreadable:{exc.strerror or exc}"]
    actual_sha256 = hashlib.sha256(decision_bytes).hexdigest()
    expected_sha256 = str(receipt.get("approval_decision_sha256") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) and expected_sha256 != actual_sha256:
        issues.append(f"post_approval_receipts:{receipt_id}:approval_decision_sha256_mismatch")
    try:
        decision_payload = json.loads(decision_bytes.decode("utf-8"))
    except UnicodeDecodeError:
        issues.append(f"post_approval_receipts:{receipt_id}:approval_decision_ref:not_utf8_json")
        return issues
    except json.JSONDecodeError as exc:
        issues.append(f"post_approval_receipts:{receipt_id}:approval_decision_ref:json_parse_failed:{exc.msg}")
        return issues
    if not isinstance(decision_payload, Mapping):
        issues.append(f"post_approval_receipts:{receipt_id}:approval_decision_ref:root_must_be_object")
        return issues
    if decision_payload.get("redacted") is not True and not _strict_affirmative_redaction_policy(
        str(decision_payload.get("redaction_policy") or "")
    ):
        issues.append(f"post_approval_receipts:{receipt_id}:approval_decision_ref:not_redacted")
    issues.extend(
        f"post_approval_receipts:{receipt_id}:approval_decision_ref:{issue}"
        for issue in _example_only_presence_issues(decision_payload)
    )
    issues.extend(
        f"post_approval_receipts:{receipt_id}:approval_decision_ref:{issue}"
        for issue in _post_approval_receipt_secret_issues(decision_payload)
    )
    for field in ("action_id", "approval_id", "decision", "decision_by", "decision_at"):
        if field not in decision_payload:
            issues.append(f"post_approval_receipts:{receipt_id}:approval_decision_ref:{field}_missing")
        elif decision_payload.get(field) != receipt.get(field):
            issues.append(f"post_approval_receipts:{receipt_id}:approval_decision_ref:{field}_mismatch")
    return issues


def _lineage_issues(prefix: str, item: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    for field in ("source_voice_session_id", "source_oracle_job_id", "parent_audit_event_id"):
        actual_value = str(item.get(field) or "")
        expected_value = str(expected.get(field) or "")
        if actual_value != expected_value:
            issues.append(f"{prefix}:{field}_mismatch")
    return issues


def validate_post_approval_receipts(
    payload: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    if str(payload.get("schema_version") or "") != POST_APPROVAL_RECEIPTS_SCHEMA_VERSION:
        issues.append("post_approval_receipts:missing_or_invalid_schema_version")
    issues.extend(f"post_approval_receipts:{issue}" for issue in _example_only_presence_issues(payload))
    issues.extend(f"post_approval_receipts:{issue}" for issue in _post_approval_receipt_secret_issues(payload))
    issues.extend(
        _collector_attestation_issues(
            payload,
            section_name="post_approval_receipts",
            expected_redacted_sha256=_post_approval_receipts_redacted_sha256(payload),
        )
    )

    receipts = payload.get("receipts")
    credential_locations = payload.get("credential_locations")
    rollback_receipts = payload.get("rollback_receipts")
    audit_events = payload.get("audit_events")
    if not isinstance(receipts, list):
        issues.append("post_approval_receipts:receipts_not_list")
        receipts = []
    if not isinstance(credential_locations, list):
        issues.append("post_approval_receipts:credential_locations_not_list")
        credential_locations = []
    if not isinstance(rollback_receipts, list):
        issues.append("post_approval_receipts:rollback_receipts_not_list")
        rollback_receipts = []
    if not isinstance(audit_events, list):
        issues.append("post_approval_receipts:audit_events_not_list")
        audit_events = []

    actions = {action["action_id"]: action for action in plan["approval_required_actions"]}
    expected_action_ids = set(actions)
    action_estimates = {
        str(step.get("step_id") or ""): step.get("estimated_cents")
        for step in plan.get("execution_steps", [])
        if isinstance(step, Mapping)
    }
    spend_currency = str(plan.get("spend_policy", {}).get("currency") or "")
    budget_cap_cents = plan.get("spend_policy", {}).get("budget_cap_cents")
    required_receipt_fields = set(plan["receipt_schema"]["required_fields"])
    non_executed_optional_receipt_fields = {
        "amount_cents",
        "credential_location_ref",
        "currency",
        "executed_at",
        "external_reference",
        "rollback_ref",
    }
    required_credential_fields = set(plan["credential_location_schema"]["required_fields"])
    forbidden_credential_fields = set(plan["credential_location_schema"]["forbidden_fields"])
    duplicate_credential_refs = _duplicate_nonempty_field_values(credential_locations, "credential_ref_id")
    duplicate_rollback_refs = _duplicate_nonempty_field_values(rollback_receipts, "rollback_ref")
    duplicate_audit_event_ids = _duplicate_nonempty_field_values(audit_events, "audit_event_id")
    if duplicate_credential_refs:
        issues.append(f"post_approval_receipts:duplicate_credential_refs:{','.join(duplicate_credential_refs)}")
    if duplicate_rollback_refs:
        issues.append(f"post_approval_receipts:duplicate_rollback_refs:{','.join(duplicate_rollback_refs)}")
    if duplicate_audit_event_ids:
        issues.append(f"post_approval_receipts:duplicate_audit_event_ids:{','.join(duplicate_audit_event_ids)}")
    credential_by_ref = {
        str(item.get("credential_ref_id") or ""): item
        for item in credential_locations
        if isinstance(item, Mapping)
    }
    rollback_refs = {
        str(item.get("rollback_ref") or "")
        for item in rollback_receipts
        if isinstance(item, Mapping)
    }
    audit_by_id = {
        str(item.get("audit_event_id") or ""): item
        for item in audit_events
        if isinstance(item, Mapping)
    }
    receipt_ids: list[str] = []
    receipt_action_ids: list[str] = []
    total_amount_cents = 0
    ledger_rows: list[dict[str, Any]] = []
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, Mapping):
            issues.append(f"post_approval_receipts:receipts[{index}]:not_object")
            continue
        receipt_id = str(receipt.get("receipt_id") or "")
        receipt_ids.append(receipt_id)
        action_id = str(receipt.get("action_id") or "")
        if action_id:
            receipt_action_ids.append(action_id)
        action = actions.get(action_id)
        status = str(receipt.get("status") or "")
        status_required_fields = required_receipt_fields
        if status in POST_APPROVAL_NON_EXECUTED_STATUSES:
            status_required_fields = required_receipt_fields.difference(non_executed_optional_receipt_fields)
        missing = sorted(field for field in status_required_fields if field not in receipt)
        if missing:
            issues.append(f"post_approval_receipts:{receipt_id or index}:missing_fields:{','.join(missing)}")
        if action is None:
            issues.append(f"post_approval_receipts:{receipt_id or index}:unknown_action_id")
            continue
        command = str(action.get("command") or "")
        expected_command_hash = hashlib.sha256(command.encode("utf-8")).hexdigest()
        if action.get("command_sha256") != expected_command_hash:
            issues.append(f"post_approval_receipts:{receipt_id}:plan_command_sha256_mismatch")
        contract = action.get("approval_contract") if isinstance(action.get("approval_contract"), Mapping) else {}
        if contract and contract.get("command_sha256") != expected_command_hash:
            issues.append(f"post_approval_receipts:{receipt_id}:approval_contract_command_sha256_mismatch")
        receipt_slot_path = str(action.get("expected_receipt_ref") or "")
        receipt_slot = _dot_get(plan, receipt_slot_path) if receipt_slot_path else None
        if isinstance(receipt_slot, Mapping) and receipt_slot.get("command_sha256") != expected_command_hash:
            issues.append(f"post_approval_receipts:{receipt_id}:receipt_slot_command_sha256_mismatch")
        action_lineage = action.get("lineage") if isinstance(action.get("lineage"), Mapping) else {}
        issues.extend(_lineage_issues(f"post_approval_receipts:{receipt_id or index}", receipt, action_lineage))
        if receipt.get("approval_id") != action["approval_id"]:
            issues.append(f"post_approval_receipts:{receipt_id}:approval_id_mismatch")
        if receipt.get("command_sha256") != action["command_sha256"]:
            issues.append(f"post_approval_receipts:{receipt_id}:command_sha256_mismatch")
        if receipt.get("provider") != action["provider"]:
            issues.append(f"post_approval_receipts:{receipt_id}:provider_mismatch")
        if receipt.get("approval_artifact") != action["approval_artifact"]:
            issues.append(f"post_approval_receipts:{receipt_id}:approval_artifact_mismatch")
        should_validate_rollback_ref = status not in POST_APPROVAL_NON_EXECUTED_STATUSES or "rollback_ref" in receipt
        if should_validate_rollback_ref and receipt.get("rollback_ref") != action["rollback_ref"]:
            issues.append(f"post_approval_receipts:{receipt_id}:rollback_ref_mismatch")
        should_validate_credential_ref = (
            status not in POST_APPROVAL_NON_EXECUTED_STATUSES or "credential_location_ref" in receipt
        )
        if should_validate_credential_ref and receipt.get("credential_location_ref") != action.get(
            "credential_location_ref"
        ):
            issues.append(f"post_approval_receipts:{receipt_id}:credential_location_ref_mismatch")
        decision = str(receipt.get("decision") or "")
        allowed_decisions = action.get("approval_contract", {}).get("allowed_decisions")
        if decision not in allowed_decisions:
            issues.append(f"post_approval_receipts:{receipt_id}:invalid_decision")
        expected_decision = _expected_post_approval_decision(status)
        if expected_decision and decision != expected_decision:
            issues.append(f"post_approval_receipts:{receipt_id}:decision_mismatch:{expected_decision}")
        decision_ref = str(receipt.get("approval_decision_ref") or "").strip()
        if not decision_ref:
            issues.append(f"post_approval_receipts:{receipt_id}:missing_approval_decision_ref")
        decision_sha256 = str(receipt.get("approval_decision_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", decision_sha256):
            issues.append(f"post_approval_receipts:{receipt_id}:invalid_approval_decision_sha256")
        issues.extend(
            _approval_decision_artifact_issues(
                receipt,
                receipt_id=receipt_id or str(index),
                receipt_path=receipt_path,
            )
        )
        if (
            status not in POST_APPROVAL_NON_EXECUTED_STATUSES
            and action.get("credential_location_required")
            and str(receipt.get("credential_location_ref") or "") not in credential_by_ref
        ):
            issues.append(f"post_approval_receipts:{receipt_id}:missing_credential_location")
        if (
            status not in POST_APPROVAL_NON_EXECUTED_STATUSES
            and str(receipt.get("rollback_ref") or "") not in rollback_refs
        ):
            issues.append(f"post_approval_receipts:{receipt_id}:missing_rollback_receipt")
        audit_id = str(receipt.get("audit_event_id") or "")
        audit_event = audit_by_id.get(audit_id)
        if audit_event is None:
            issues.append(f"post_approval_receipts:{receipt_id}:missing_audit_event")
        else:
            if audit_event.get("action_id") != action_id:
                issues.append(f"post_approval_receipts:{receipt_id}:audit_action_id_mismatch")
            if audit_event.get("receipt_id") != receipt_id:
                issues.append(f"post_approval_receipts:{receipt_id}:audit_receipt_id_mismatch")
            if audit_event.get("status") != receipt.get("status"):
                issues.append(f"post_approval_receipts:{receipt_id}:audit_status_mismatch")
            if audit_event.get("provider") != action["provider"]:
                issues.append(f"post_approval_receipts:{receipt_id}:audit_provider_mismatch")
            issues.extend(
                _lineage_issues(
                    f"post_approval_receipts:{receipt_id}:audit_event",
                    audit_event,
                    action_lineage,
                )
            )
        if status not in POST_APPROVAL_RECEIPT_STATUSES:
            issues.append(f"post_approval_receipts:{receipt_id}:invalid_status")
        decision_at = _parse_preflight_timestamp(receipt.get("decision_at"))
        executed_at_value = receipt.get("executed_at")
        executed_at = _parse_preflight_timestamp(executed_at_value)
        if decision_at is None:
            issues.append(f"post_approval_receipts:{receipt_id}:invalid_decision_at")
        if status not in POST_APPROVAL_NON_EXECUTED_STATUSES and executed_at is None:
            issues.append(f"post_approval_receipts:{receipt_id}:invalid_executed_at")
        if (
            status in POST_APPROVAL_NON_EXECUTED_STATUSES
            and executed_at_value not in {None, ""}
            and executed_at is None
        ):
            issues.append(f"post_approval_receipts:{receipt_id}:invalid_executed_at")
        if decision_at is not None and executed_at is not None and decision_at > executed_at:
            issues.append(f"post_approval_receipts:{receipt_id}:decision_after_executed")
        amount = receipt.get("amount_cents")
        if amount is None and status in POST_APPROVAL_NON_EXECUTED_STATUSES:
            pass
        elif not isinstance(amount, int) or amount < 0:
            issues.append(f"post_approval_receipts:{receipt_id}:invalid_amount_cents")
        else:
            total_amount_cents += amount
            estimate = action_estimates.get(action_id)
            if isinstance(estimate, int) and amount > estimate:
                issues.append(f"post_approval_receipts:{receipt_id}:amount_exceeds_estimate")
        if (status not in POST_APPROVAL_NON_EXECUTED_STATUSES or "currency" in receipt) and str(
            receipt.get("currency") or ""
        ) != spend_currency:
            issues.append(f"post_approval_receipts:{receipt_id}:currency_mismatch")
        ledger_rows.append(
            {
                "audit_event_id": audit_id,
                "receipt_id": receipt_id,
                "action_id": action_id,
                "approval_id": str(receipt.get("approval_id") or ""),
                "status": status,
                "provider": str(receipt.get("provider") or ""),
                "external_reference": str(receipt.get("external_reference") or ""),
                "credential_location_ref": str(receipt.get("credential_location_ref") or ""),
                "rollback_ref": str(receipt.get("rollback_ref") or ""),
            }
        )

    duplicate_receipts = sorted(receipt_id for receipt_id in set(receipt_ids) if receipt_id and receipt_ids.count(receipt_id) > 1)
    if duplicate_receipts:
        issues.append(f"post_approval_receipts:duplicate_receipt_ids:{','.join(duplicate_receipts)}")
    duplicate_actions = sorted(action_id for action_id in set(receipt_action_ids) if action_id and receipt_action_ids.count(action_id) > 1)
    if duplicate_actions:
        issues.append(f"post_approval_receipts:duplicate_action_receipts:{','.join(duplicate_actions)}")
    missing_action_receipts = sorted(expected_action_ids.difference(set(receipt_action_ids)))
    if missing_action_receipts:
        issues.append(f"post_approval_receipts:missing_receipts_for_actions:{','.join(missing_action_receipts)}")
    if isinstance(budget_cap_cents, int) and total_amount_cents > budget_cap_cents:
        issues.append("post_approval_receipts:total_amount_exceeds_budget")
    action_by_rollback_ref = {
        str(action.get("rollback_ref") or ""): action
        for action in actions.values()
        if str(action.get("rollback_ref") or "")
    }
    for index, credential in enumerate(credential_locations):
        if not isinstance(credential, Mapping):
            issues.append(f"post_approval_receipts:credential_locations[{index}]:not_object")
            continue
        credential_id = str(credential.get("credential_ref_id") or f"index-{index}")
        missing = sorted(field for field in required_credential_fields if field not in credential)
        if missing:
            issues.append(f"post_approval_receipts:{credential_id}:missing_credential_fields:{','.join(missing)}")
        forbidden_present = sorted(field for field in forbidden_credential_fields if field in credential)
        if forbidden_present:
            issues.append(f"post_approval_receipts:{credential_id}:forbidden_credential_fields:{','.join(forbidden_present)}")
        if credential.get("storage_backend") not in plan["credential_location_schema"]["allowed_storage_backends"]:
            issues.append(f"post_approval_receipts:{credential_id}:invalid_storage_backend")
        action = actions.get(str(credential.get("created_by_action_id") or ""))
        if action is not None:
            expected_lineage = action.get("lineage") if isinstance(action.get("lineage"), Mapping) else {}
            actual_lineage = credential.get("lineage") if isinstance(credential.get("lineage"), Mapping) else credential
            issues.extend(
                _lineage_issues(
                    f"post_approval_receipts:{credential_id}:lineage",
                    actual_lineage,
                    expected_lineage,
                )
            )
    for index, rollback in enumerate(rollback_receipts):
        if not isinstance(rollback, Mapping):
            continue
        rollback_ref = str(rollback.get("rollback_ref") or "")
        action = action_by_rollback_ref.get(rollback_ref)
        if action is None:
            continue
        expected_lineage = action.get("lineage") if isinstance(action.get("lineage"), Mapping) else {}
        actual_lineage = rollback.get("lineage") if isinstance(rollback.get("lineage"), Mapping) else rollback
        issues.extend(
            _lineage_issues(
                f"post_approval_receipts:rollback_receipts[{index}]:lineage",
                actual_lineage,
                expected_lineage,
            )
        )
    for audit_id, event in audit_by_id.items():
        if not audit_id:
            issues.append("post_approval_receipts:audit_event_missing_id")
            continue
        if not str(event.get("artifact_ref") or "").strip():
            issues.append(f"post_approval_receipts:{audit_id}:missing_artifact_ref")
        if not str(event.get("operator_next_step") or "").strip():
            issues.append(f"post_approval_receipts:{audit_id}:missing_operator_next_step")

    return {
        **_post_approval_receipts_validation_base(plan),
        "status": "valid" if not issues else "invalid",
        "validation_issues": sorted(set(issues)),
        "receipt_count": len(receipts),
        "credential_location_count": len(credential_locations),
        "rollback_receipt_count": len(rollback_receipts),
        "audit_event_count": len(audit_events),
        "ledger_rows": ledger_rows if not issues else [],
    }


def _expected_post_approval_decision(status: str) -> str:
    if status in POST_APPROVAL_ATTEMPTED_EXECUTION_STATUSES:
        return "approve_once"
    if status == "denied":
        return "deny"
    if status in {"held", "skipped"}:
        return "hold"
    return ""


def write_probe_artifacts(output_dir: Path, report: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_plan = build_milestone2_execution_plan(report)
    setup_closure = build_setup_closure_plan(report)
    paths = {
        "json": output_dir / "provisioning-readiness.json",
        "markdown": output_dir / "provisioning-readiness.md",
        "command_manifest": output_dir / "safe-command-manifest.json",
        "read_only_discovery_json": output_dir / "read-only-discovery.json",
        "read_only_discovery_markdown": output_dir / "read-only-discovery.md",
        "read_only_discovery_manifest": output_dir / "read-only-discovery.manifest.json",
        "read_only_discovery_audit_ledger": output_dir / "audit-ledger.read-only-discovery.jsonl",
        "execution_plan_json": output_dir / "milestone2-execution-plan.json",
        "execution_plan_markdown": output_dir / "milestone2-execution-plan.md",
        "post_approval_receipts_template": output_dir / "post-approval-receipts.template.json",
        "post_approval_receipts_example": output_dir / "post-approval-receipts.example.json",
        "post_approval_receipts_validation": output_dir / "post-approval-receipts.validation.json",
        "post_approval_audit_ledger": output_dir / "audit-ledger.post-approval.jsonl",
        "nemoclaw_action_packet_validation": output_dir / "nemoclaw-action-packet.validation.json",
        "preflight_evidence_template": output_dir / "provisioning-preflight-evidence.template.json",
        "preflight_evidence_example": output_dir / "provisioning-preflight-evidence.example.json",
        "preflight_evidence_manifest_example": output_dir / "provisioning-preflight-evidence.manifest.example.json",
        "setup_closure_json": output_dir / "setup-closure-plan.json",
        "setup_closure_markdown": output_dir / "setup-closure-plan.md",
    }
    _write_json(paths["json"], report)
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    _write_json(paths["command_manifest"], _safe_command_manifest_json())
    read_only_discovery_export = _readonly_discovery_export_payload(report["read_only_discovery"])
    _write_json(paths["read_only_discovery_json"], read_only_discovery_export)
    read_only_discovery_report_sha256 = _file_sha256(paths["read_only_discovery_json"])
    paths["read_only_discovery_markdown"].write_text(
        _read_only_discovery_markdown(read_only_discovery_export),
        encoding="utf-8",
    )
    _write_json(
        paths["read_only_discovery_manifest"],
        _read_only_discovery_manifest(
            read_only_discovery_export,
            report_sha256=read_only_discovery_report_sha256,
        ),
    )
    _write_jsonl(
        paths["read_only_discovery_audit_ledger"],
        _read_only_discovery_ledger_rows(read_only_discovery_export),
    )
    _write_json(paths["execution_plan_json"], execution_plan)
    paths["execution_plan_markdown"].write_text(_execution_plan_markdown(execution_plan), encoding="utf-8")
    _write_json(paths["post_approval_receipts_template"], build_post_approval_receipts_template(execution_plan))
    _write_json(paths["post_approval_receipts_example"], build_post_approval_receipts_example(execution_plan))
    paths.update(write_post_approval_receipts_scaffold(output_dir, execution_plan))
    _write_json(paths["post_approval_receipts_validation"], report["post_approval_receipts"])
    _write_jsonl(paths["post_approval_audit_ledger"], report["post_approval_receipts"].get("ledger_rows", []))
    _write_json(paths["nemoclaw_action_packet_validation"], report["nemoclaw_action_packet"])
    _write_json(paths["preflight_evidence_template"], build_preflight_evidence_template())
    _write_json(paths["preflight_evidence_example"], build_preflight_evidence_example())
    _write_json(paths["preflight_evidence_manifest_example"], build_preflight_evidence_manifest_example())
    paths.update(write_preflight_evidence_scaffold(output_dir))
    _write_json(paths["setup_closure_json"], setup_closure)
    paths["setup_closure_markdown"].write_text(_setup_closure_markdown(setup_closure), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-file", action="append", default=None, type=Path, help="Env file to inspect for key presence only.")
    parser.add_argument(
        "--preflight-evidence",
        type=Path,
        default=None,
        help="Redacted Milestone 2 account/capability evidence JSON; values must be refs only, never secrets.",
    )
    parser.add_argument(
        "--post-approval-receipts",
        type=Path,
        default=None,
        help="Read-only redacted post-approval receipt bundle; validates receipts and writes local ledger artifacts only.",
    )
    parser.add_argument(
        "--nemoclaw-action-packet",
        type=Path,
        default=None,
        help="Read-only NemoClaw action packet JSON to validate without running commands or granting approval.",
    )
    parser.add_argument(
        "--read-only-discovery-evidence",
        "--readonly-discovery-evidence",
        type=Path,
        default=None,
        dest="read_only_discovery_evidence",
        help=(
            "Read-only redacted discovery report or manifest from a prior allowlisted discovery run; "
            "ingests evidence without running discovery commands."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help=(
            "Timeout for fast local version/help probes. Read-only discovery uses "
            f"{DEFAULT_READONLY_DISCOVERY_TIMEOUT_SECONDS}s by default because provider CLIs may perform network I/O."
        ),
    )
    parser.add_argument(
        "--readonly-discovery-timeout-seconds",
        "--read-only-discovery-timeout-seconds",
        type=int,
        default=None,
        dest="readonly_discovery_timeout_seconds",
        help=(
            "Timeout for exact allowlisted read-only provider discovery; defaults to "
            f"{DEFAULT_READONLY_DISCOVERY_TIMEOUT_SECONDS}s."
        ),
    )
    parser.add_argument(
        "--run-command-probes",
        action="store_true",
        help="Opt in to isolated version/help subprocess probes. Default inspects PATH/env presence only.",
    )
    parser.add_argument(
        "--run-readonly-discovery",
        "--run-read-only-discovery",
        action="store_true",
        dest="run_readonly_discovery",
        help="Opt in to exact allowlisted read-only discovery commands with redacted output artifacts.",
    )
    parser.add_argument(
        "--refresh-preflight-source-hashes",
        type=Path,
        default=None,
        help=(
            "Refresh source_artifact_sha256 fields in a local preflight evidence file or manifest, "
            "then exit without probing CLIs or external systems."
        ),
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
    if args.refresh_preflight_source_hashes is not None:
        result = refresh_preflight_source_hashes(args.refresh_preflight_source_hashes)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1
    report = build_probe_report(
        env_files=args.env_file,
        preflight_evidence_path=args.preflight_evidence,
        read_only_discovery_evidence_path=args.read_only_discovery_evidence,
        post_approval_receipts_path=args.post_approval_receipts,
        nemoclaw_action_packet_path=args.nemoclaw_action_packet,
        run_commands=args.run_command_probes,
        run_readonly_discovery=args.run_readonly_discovery,
        timeout_seconds=args.timeout_seconds,
        readonly_discovery_timeout_seconds=args.readonly_discovery_timeout_seconds,
    )
    paths = write_probe_artifacts(args.output_dir, report)
    print(
        json.dumps(
            {
                "ok": True,
                "ok_meaning": "probe completed; readiness is reported by ready/status/required_failures",
                "ready": report["ready"],
                "status": report["status"],
                "required_failures": report["required_failures"],
                "area_status": report["area_status"],
                "read_only_discovery_status": report["read_only_discovery"]["status"],
                "read_only_discovery_failed_probe_ids": report["read_only_discovery"]["failed_probe_ids"],
                "read_only_discovery_missing_probe_ids": report["read_only_discovery"]["missing_probe_ids"],
                "read_only_discovery_timed_out_probe_ids": report["read_only_discovery"].get("timed_out_probe_ids", []),
                "output_dir": str(args.output_dir),
                "setup_closure_json": paths["setup_closure_json"],
                "artifacts": paths,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
