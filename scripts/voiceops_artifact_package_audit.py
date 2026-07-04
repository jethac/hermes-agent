#!/usr/bin/env python3
"""Audit the generated VoiceOps artifact package for cross-file consistency.

This script is intentionally local-only. It reads generated JSON/Markdown/HTML
artifacts and never inspects env files, contacts providers, runs Discord, spends
money, provisions services, retrieves credentials, sends messages, or places
calls.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shlex
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.voiceops_channel_policy import CHANNEL_IDS, validate_policy
from scripts.voiceops_operator_state import validate_operator_state
from scripts.voiceops_provisioning_probe import load_preflight_evidence, validate_post_approval_receipts
from scripts.voiceops_spark_matrix import build_matrix
from scripts.voiceops_voice_operator import _load_live_evidence, validate_live_probe_evidence, validate_voice_operator_report


DEFAULT_ARTIFACT_ROOT = Path("artifacts")
DEFAULT_OUTPUT_DIR = Path("artifacts/voiceops-package-audit/current")
AUDIT_SCHEMA_VERSION = "voiceops.artifact_package_audit.v1"
EXPECTED_HANDOFF_PHASES = (
    (1, "live_discord_voice"),
    (2, "spend_and_provisioning_preflight"),
    (3, "local_spark_stack"),
)
EXPECTED_REVIEW_PHASES = (
    (1, "multi_channel_policy_review"),
)
EXPECTED_PROVIDER_ROLE_AUTHORITIES = {
    "reflex": "reflex_hypothesis",
    "interpreter": "interpreter_promoted",
    "oracle": "oracle_promoted",
    "auxiliary_transcript_evidence": "auxiliary_hypothesis",
    "outbound_tts": "playback_only",
    "degraded_fallback": "fallback_text_or_diagnostic_only",
}
EXPECTED_SPARK_SCAFFOLD_LINT_ISSUES = {
    "collector_attestation_example_only_not_accepted",
    "collector_attestation_invalid:collector_version",
    "example_only_evidence_not_accepted",
    "missing_benchmark_evidence",
    "missing_oracle_authority_proof",
    "no_single_evidence_record_satisfies_targets",
    "source_artifact_example_only_not_accepted",
}
LOCAL_MODEL_MARKERS = ("local", "dgx", "spark", "localhost", "127.0.0.1", "vllm")
HOSTED_MODEL_MARKERS = ("hosted", "cloud", "provider", "remote", "api", "nous")
SECRET_SCAN_PATTERNS = (
    ("openai_or_stripe_secret_key", re.compile(r"\bsk_(?:live|test|car)_[A-Za-z0-9_-]{12,}\b")),
    ("openai_project_key", re.compile(r"\bsk-proj-[A-Za-z0-9_-]{12,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{12,}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b")),
    ("github_fine_grained_token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b")),
    ("sendgrid_api_key", re.compile(r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b")),
    ("discord_bot_token", re.compile(r"\b[MNO][A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,}\b")),
    ("e164_phone_number", re.compile(r"(?<![A-Za-z0-9_])\+[1-9][0-9]{9,14}\b")),
)
EXPECTED_PACKAGE_ARTIFACTS = (
    "hackathon-voiceops-demo/current/audit-ledger.jsonl",
    "hackathon-voiceops-demo/current/demo-script.md",
    "hackathon-voiceops-demo/current/milestone2-execution-plan.json",
    "hackathon-voiceops-demo/current/nemoclaw-action-packet.json",
    "hackathon-voiceops-demo/current/nemoclaw-action-packet.validation.json",
    "hackathon-voiceops-demo/current/operator-dashboard.html",
    "hackathon-voiceops-demo/current/operator-handoff-preview.json",
    "hackathon-voiceops-demo/current/operator-handoff-preview.md",
    "hackathon-voiceops-demo/current/operator-state-events.jsonl",
    "hackathon-voiceops-demo/current/operator-state.json",
    "hackathon-voiceops-demo/current/phone-context.json",
    "hackathon-voiceops-demo/current/readiness-closure-summary.json",
    "hackathon-voiceops-demo/current/readiness-closure-summary.md",
    "hackathon-voiceops-demo/current/readiness-report.json",
    "hackathon-voiceops-demo/current/readiness-report.md",
    "hackathon-voiceops-demo/current/recording-runbook.md",
    "hackathon-voiceops-demo/current/stripe-actions-dry-run.sh",
    "hackathon-voiceops-demo/current/submission-writeup.md",
    "hackathon-voiceops-demo/current/voiceops-demo.json",
    "hackathon-voiceops-demo/current/voiceops-demo.md",
    "voiceops-channel-policy/current/channel-policy-review.json",
    "voiceops-channel-policy/current/channel-policy-review.md",
    "voiceops-channel-policy/current/channel-policy.json",
    "voiceops-channel-policy/current/channel-policy.md",
    "voiceops-operator-state/current/operator-state-events.jsonl",
    "voiceops-operator-state/current/operator-state.json",
    "voiceops-operator-state/current/operator-state.md",
    "voiceops-plan/current/operator-handoff.json",
    "voiceops-plan/current/operator-handoff.md",
    "voiceops-plan/current/readiness-closure-index.json",
    "voiceops-plan/current/readiness-closure-index.md",
    "voiceops-plan/current/voiceops-plan-run.json",
    "voiceops-plan/current/voiceops-plan-run.md",
    "voiceops-provisioning/current/audit-ledger.post-approval.jsonl",
    "voiceops-provisioning/current/audit-ledger.read-only-discovery.jsonl",
    "voiceops-provisioning/current/milestone2-execution-plan.json",
    "voiceops-provisioning/current/milestone2-execution-plan.md",
    "voiceops-provisioning/current/nemoclaw-action-packet.validation.json",
    "voiceops-provisioning/current/post-approval-receipts-scaffold/post-approval-receipts.json",
    "voiceops-provisioning/current/post-approval-receipts.example.json",
    "voiceops-provisioning/current/post-approval-receipts.template.json",
    "voiceops-provisioning/current/post-approval-receipts.validation.json",
    "voiceops-provisioning/current/provisioning-preflight-evidence.example.json",
    "voiceops-provisioning/current/provisioning-preflight-evidence.manifest.example.json",
    "voiceops-provisioning/current/provisioning-preflight-evidence.template.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sections/nemoclaw-boundary-evidence.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sections/phone-handoff-evidence.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sections/rollback-owner-evidence.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sections/stripe-link-evidence.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sections/stripe-projects-evidence.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sources/nemoclaw-boundary-redacted-source.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sources/phone-handoff-redacted-source.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sources/rollback-owners-redacted-source.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sources/stripe-link-redacted-source.json",
    "voiceops-provisioning/current/provisioning-preflight-scaffold/sources/stripe-projects-redacted-source.json",
    "voiceops-provisioning/current/provisioning-readiness.json",
    "voiceops-provisioning/current/provisioning-readiness.md",
    "voiceops-provisioning/current/read-only-discovery.json",
    "voiceops-provisioning/current/read-only-discovery.manifest.json",
    "voiceops-provisioning/current/read-only-discovery.md",
    "voiceops-provisioning/current/safe-command-manifest.json",
    "voiceops-provisioning/current/setup-closure-plan.json",
    "voiceops-provisioning/current/setup-closure-plan.md",
    "voiceops-spark-matrix/current/spark-benchmark-evidence-template.json",
    "voiceops-spark-matrix/current/spark-benchmark-evidence.example.json",
    "voiceops-spark-matrix/current/spark-benchmark-scaffold/sources/all-local-stack-smoke-raw.json",
    "voiceops-spark-matrix/current/spark-benchmark-scaffold/sources/asr-nemotron-speech-raw.json",
    "voiceops-spark-matrix/current/spark-benchmark-scaffold/sources/interpreter-gemma4-e2b-raw.json",
    "voiceops-spark-matrix/current/spark-benchmark-scaffold/sources/interpreter-gemma4-e4b-raw.json",
    "voiceops-spark-matrix/current/spark-benchmark-scaffold/sources/oracle-nemotron3-super-raw.json",
    "voiceops-spark-matrix/current/spark-benchmark-scaffold/sources/reflex-moshi-s2s-raw.json",
    "voiceops-spark-matrix/current/spark-benchmark-scaffold/sources/tts-magpie-local-raw.json",
    "voiceops-spark-matrix/current/spark-benchmark-scaffold/spark-benchmark-evidence.json",
    "voiceops-spark-matrix/current/spark-matrix-closure-plan.json",
    "voiceops-spark-matrix/current/spark-matrix-closure-plan.md",
    "voiceops-spark-matrix/current/spark-model-matrix.json",
    "voiceops-spark-matrix/current/spark-model-matrix.md",
    "voiceops-spark-matrix/current/spark-operator-runbook.md",
    "voiceops-voice-operator/current/discord-loopback-smoke.json",
    "voiceops-voice-operator/current/async-oracle-smoke.json",
    "voiceops-voice-operator/current/discord-session-cleanup-smoke.json",
    "voiceops-voice-operator/current/sidecar-fail-closed-smoke.json",
    "voiceops-voice-operator/current/live-probe-closure-plan.json",
    "voiceops-voice-operator/current/live-probe-closure-plan.md",
    "voiceops-voice-operator/current/live-voice-evidence-scaffold/manifest.json",
    "voiceops-voice-operator/current/live-voice-evidence-scaffold/sections/discord-live-probe.json",
    "voiceops-voice-operator/current/live-voice-evidence-scaffold/sections/live-turn.json",
    "voiceops-voice-operator/current/live-voice-evidence-scaffold/sections/sidecar-session.json",
    "voiceops-voice-operator/current/live-voice-evidence-template.json",
    "voiceops-voice-operator/current/live-voice-evidence.example.json",
    "voiceops-voice-operator/current/voice-operator-events.jsonl",
    "voiceops-voice-operator/current/voice-operator-readiness.json",
    "voiceops-voice-operator/current/voice-operator-readiness.md",
)
OPTIONAL_PACKAGE_ARTIFACTS = (
    "voiceops-provisioning/current/approval-decisions.json",
    "voiceops-provisioning/current/approval-decisions/provision-voip-provider.json",
    "voiceops-provisioning/current/post-approval-receipts.json",
    "voiceops-provisioning/current/stripe-executor-report.json",
)
AUDITED_PACKAGE_DIRS = tuple(
    sorted(
        {
            "/".join(relative_path.split("/")[:2])
            for relative_path in (*EXPECTED_PACKAGE_ARTIFACTS, *OPTIONAL_PACKAGE_ARTIFACTS)
        }
    )
)


def _read_json(path: Path, issues: list[str], label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        issues.append(f"{label}:missing:{path}")
        return {}
    except json.JSONDecodeError as exc:
        issues.append(f"{label}:json_parse_failed:{exc.msg}")
        return {}
    if not isinstance(payload, dict):
        issues.append(f"{label}:root_must_be_object")
        return {}
    return payload


def _read_text(path: Path, issues: list[str], label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        issues.append(f"{label}:missing:{path}")
    except UnicodeDecodeError:
        issues.append(f"{label}:not_utf8:{path}")
    return ""


def _read_jsonl(path: Path, issues: list[str], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = _read_text(path, issues, label)
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(f"{label}:line_{line_number}:json_parse_failed:{exc.msg}")
            continue
        if not isinstance(row, dict):
            issues.append(f"{label}:line_{line_number}:row_must_be_object")
            continue
        rows.append(row)
    return rows


def _audit_expected_package_artifacts(artifact_root: Path, issues: list[str]) -> list[str]:
    checked_artifacts: list[str] = []
    for relative_path in EXPECTED_PACKAGE_ARTIFACTS:
        path = artifact_root / relative_path
        label = f"package_artifact:{relative_path}"
        if path.suffix == ".json":
            _read_json(path, issues, label)
        elif path.suffix == ".jsonl":
            _read_jsonl(path, issues, label)
        else:
            text = _read_text(path, issues, label)
            if not text.strip():
                issues.append(f"{label}:empty")
        checked_artifacts.append(str(path))
    for relative_path in OPTIONAL_PACKAGE_ARTIFACTS:
        path = artifact_root / relative_path
        if not path.exists():
            continue
        label = f"package_artifact:{relative_path}"
        if path.suffix == ".json":
            _read_json(path, issues, label)
        elif path.suffix == ".jsonl":
            _read_jsonl(path, issues, label)
        else:
            text = _read_text(path, issues, label)
            if not text.strip():
                issues.append(f"{label}:empty")
        checked_artifacts.append(str(path))
    return checked_artifacts


def _audit_no_unexpected_package_artifacts(
    artifact_root: Path,
    checked_artifacts: list[str],
    issues: list[str],
) -> None:
    expected_paths = {Path(path).resolve(strict=False) for path in checked_artifacts}
    for relative_dir in AUDITED_PACKAGE_DIRS:
        directory = artifact_root / relative_dir
        if not directory.exists():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            if path.resolve(strict=False) not in expected_paths:
                issues.append(f"package_artifact:unexpected:{_artifact_label(artifact_root, path)}")


def _audit_no_secret_like_values(artifact_root: Path, checked_artifacts: list[str], issues: list[str]) -> None:
    for path_text in checked_artifacts:
        path = Path(path_text)
        label = _artifact_label(artifact_root, path)
        try:
            text = path.read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for rule_id, pattern in SECRET_SCAN_PATTERNS:
                if pattern.search(line):
                    issues.append(f"secret_scan:{label}:line_{line_number}:{rule_id}")


def _artifact_label(artifact_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(artifact_root))
    except ValueError:
        return str(path)


def _dollars(cents: Any) -> str:
    try:
        value = int(cents)
    except (TypeError, ValueError):
        value = 0
    return f"${value / 100:,.2f}"


def _squash_text(value: str) -> str:
    return " ".join(html.unescape(value).split())


class _DashboardTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: dict[str, list[list[str]]] = {}
        self.current_heading: str | None = None
        self._capture_heading = False
        self._heading_parts: list[str] = []
        self._table_heading: str | None = None
        self._table_rows: list[list[str]] = []
        self._row_cells: list[str] | None = None
        self._capture_cell = False
        self._cell_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "h2":
            self._capture_heading = True
            self._heading_parts = []
        elif tag == "table":
            self._table_heading = self.current_heading
            self._table_rows = []
        elif tag == "tr" and self._table_heading is not None:
            self._row_cells = []
        elif tag in {"td", "th"} and self._row_cells is not None:
            self._capture_cell = True
            self._cell_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "h2" and self._capture_heading:
            self.current_heading = _squash_text("".join(self._heading_parts))
            self._capture_heading = False
        elif tag in {"td", "th"} and self._capture_cell:
            if self._row_cells is not None:
                self._row_cells.append(_squash_text("".join(self._cell_parts)))
            self._capture_cell = False
            self._cell_parts = []
        elif tag == "tr" and self._row_cells is not None:
            if self._row_cells:
                self._table_rows.append(self._row_cells)
            self._row_cells = None
        elif tag == "table" and self._table_heading is not None:
            self.tables[self._table_heading] = self._table_rows
            self._table_heading = None
            self._table_rows = []

    def handle_data(self, data: str) -> None:
        if self._capture_heading:
            self._heading_parts.append(data)
        if self._capture_cell:
            self._cell_parts.append(data)


def _dashboard_tables(dashboard_html: str) -> dict[str, list[list[str]]]:
    parser = _DashboardTableParser()
    parser.feed(dashboard_html)
    parser.close()
    return parser.tables


def _dashboard_metrics(dashboard_html: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    pattern = re.compile(
        r'<div class="metric"><small>(?P<label>.*?)</small><strong>(?P<value>.*?)</strong>',
        re.DOTALL,
    )
    for match in pattern.finditer(dashboard_html):
        metrics[_squash_text(match.group("label"))] = _squash_text(match.group("value"))
    return metrics


def _assert_dashboard_table(
    tables: Mapping[str, list[list[str]]],
    heading: str,
    expected_rows: list[list[str]],
    issues: list[str],
) -> None:
    observed_rows = tables.get(heading)
    if observed_rows is None:
        issues.append(f"dashboard:{heading}:missing_table")
        return
    if observed_rows != expected_rows:
        issues.append(f"dashboard:{heading}:rows_mismatch")


def _approval_evidence_summary(approval: Mapping[str, Any]) -> str:
    evidence = approval.get("kame_evidence") if isinstance(approval.get("kame_evidence"), Mapping) else {}
    promoted_fields = evidence.get("promoted_fields") if isinstance(evidence.get("promoted_fields"), Mapping) else {}
    labels = sorted(
        {
            str(field.get("evidence_label"))
            for field in promoted_fields.values()
            if isinstance(field, Mapping) and field.get("evidence_label")
        }
    )
    label_text = "+".join(labels) if labels else "missing_promoted_evidence"
    audio_ref = str(evidence.get("audio_segment_ref") or "missing_audio_ref")
    tool_ref = str(approval.get("tool_disclosure_ref") or "missing_tool_disclosure")
    return f"{label_text}; audio={audio_ref}; tool={tool_ref}"


def _audit_dashboard_consistency(
    *,
    demo: Mapping[str, Any],
    readiness: Mapping[str, Any],
    operator_state: Mapping[str, Any],
    dashboard_html: str,
    issues: list[str],
) -> None:
    metrics = _dashboard_metrics(dashboard_html)
    expected_metrics = {
        "Budget": _dollars((demo.get("spend_policy") or {}).get("limit_cents")),
        "Approval queued": _dollars((demo.get("totals") or {}).get("approval_required_cents")),
        "Live/Spark gaps": str(len(readiness.get("live_demo_missing_evidence") or [])),
        "Audit events": str(len(demo.get("audit_events") or [])),
    }
    for label, expected in expected_metrics.items():
        if metrics.get(label) != expected:
            issues.append(f"dashboard:metric:{label}:mismatch")

    tables = _dashboard_tables(dashboard_html)
    budget_status = operator_state.get("budget_status") if isinstance(operator_state.get("budget_status"), Mapping) else {}
    held_actions = [
        str(action.get("action_id"))
        for action in demo.get("ops_actions", [])
        if isinstance(action, Mapping) and action.get("status") == "held-budget"
    ]
    held_action_text = ", ".join(held_actions) if held_actions else "none"
    _assert_dashboard_table(
        tables,
        "Budget Status",
        [
            ["Limit", _dollars(budget_status.get("approved_budget_cents"))],
            ["Approval threshold", _dollars(budget_status.get("approval_required_over_cents"))],
            ["Reserved approval spend", _dollars(budget_status.get("reserved_cents"))],
            ["Spent", _dollars(budget_status.get("spent_cents"))],
            ["Remaining before approval", _dollars(budget_status.get("remaining_cents"))],
            ["Held over budget", f"{_dollars(budget_status.get('held_budget_cents'))} ({held_action_text})"],
        ],
        issues,
    )
    _assert_dashboard_table(
        tables,
        "Pending Approvals",
        [["Action", "Provider", "Spend", "Purpose", "Evidence"]]
        + [
            [
                str(approval.get("action_id")),
                str(approval.get("provider")),
                _dollars(approval.get("budget_impact_cents")),
                str(approval.get("title")),
                _approval_evidence_summary(approval),
            ]
            for approval in operator_state.get("pending_approvals", [])
            if isinstance(approval, Mapping)
        ],
        issues,
    )
    _assert_dashboard_table(
        tables,
        "Action Ledger",
        [["Action", "Provider", "Status", "Spend", "Gate"]]
        + [
            [
                str(action.get("action_id")),
                str(action.get("provider")),
                str(action.get("status")),
                _dollars(action.get("estimated_cents")),
                "approval required" if action.get("requires_approval") else "no approval",
            ]
            for action in demo.get("ops_actions", [])
            if isinstance(action, Mapping)
        ],
        issues,
    )
    _assert_dashboard_table(
        tables,
        "Recent Audit Events",
        [["Event", "Action", "Status", "Amount", "Evidence"]]
        + [
            [
                str(event.get("audit_id")),
                str(event.get("event_type")),
                str(event.get("status")),
                _dollars(event.get("amount_cents")),
                str(event.get("summary")),
            ]
            for event in operator_state.get("recent_audit_events", [])
            if isinstance(event, Mapping)
        ],
        issues,
    )
    _assert_dashboard_table(
        tables,
        "Planned Services",
        [["Action", "Provider", "Status", "Purpose"]]
        + [
            [
                str(service.get("service_id")),
                str(service.get("provider")),
                str(service.get("status")),
                str(service.get("display_name")),
            ]
            for service in operator_state.get("planned_services", [])
            if isinstance(service, Mapping)
        ],
        issues,
    )
    _assert_dashboard_table(
        tables,
        "Provisioned Services",
        [["Service", "Provider", "Status", "Capability"]]
        + [
            [
                str(service.get("service_id")),
                str(service.get("provider")),
                str(service.get("status")),
                str(service.get("display_name")),
            ]
            for service in operator_state.get("provisioned_services", [])
            if isinstance(service, Mapping)
        ],
        issues,
    )


def _dry_run_metadata_rows(script_text: str, issues: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prefix = "# voiceops-action-metadata "
    for line_number, line in enumerate(script_text.splitlines(), start=1):
        if not line.startswith(prefix):
            continue
        try:
            row = json.loads(line.removeprefix(prefix))
        except json.JSONDecodeError as exc:
            issues.append(f"stripe_actions:metadata_line_{line_number}:json_parse_failed:{exc.msg}")
            continue
        if not isinstance(row, dict):
            issues.append(f"stripe_actions:metadata_line_{line_number}:row_must_be_object")
            continue
        rows.append(row)
    executable_lines = [
        line
        for line in script_text.splitlines()
        if line.strip() and not line.startswith("#") and not line.startswith("printf ")
    ]
    if executable_lines != ["set -euo pipefail"]:
        issues.append("stripe_actions:unexpected_executable_lines")
    return rows


def _approval_contract_subset(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "approval_id": contract.get("approval_id"),
        "command_sha256": contract.get("command_sha256"),
        "default_decision": contract.get("default_decision"),
        "approved_by_ref": contract.get("approved_by_ref"),
    }


def _audit_static_readiness(
    *,
    demo: Mapping[str, Any],
    readiness: Mapping[str, Any],
    spark_matrix: Mapping[str, Any],
    demo_closure: Mapping[str, Any],
    plan_closure: Mapping[str, Any],
    operator_state: Mapping[str, Any],
    dashboard_html: str,
    issues: list[str],
) -> None:
    remaining_gates = plan_closure.get("remaining_gates") or []
    closure_incomplete = bool(remaining_gates) or plan_closure.get("closure_status") != "complete"
    recording_readiness = demo.get("recording_readiness") if isinstance(demo.get("recording_readiness"), Mapping) else {}
    if closure_incomplete:
        if readiness.get("live_demo_ready") is not False:
            issues.append("readiness:live_demo_ready_not_false_with_remaining_gates")
        if recording_readiness.get("live_demo_ready") is not False:
            issues.append("demo:live_demo_ready_not_false_with_remaining_gates")
        if readiness.get("ready_for_recording_scope") != "static_artifact_recording_only":
            issues.append("readiness:scope_not_static_artifact_recording_only")
        if recording_readiness.get("ready_for_recording_scope") != "static_artifact_recording_only":
            issues.append("demo:scope_not_static_artifact_recording_only")
        if not recording_readiness.get("live_demo_missing_evidence"):
            issues.append("demo:missing_live_demo_missing_evidence")
        voice_surface = operator_state.get("active_voice_surface")
        if isinstance(voice_surface, Mapping) and voice_surface.get("status") == "active_for_demo":
            if readiness.get("ready_for_recording_scope") != "static_artifact_recording_only":
                issues.append("operator_state:active_for_demo_without_static_recording_scope")
            if not voice_surface.get("fallback_reason"):
                issues.append("operator_state:active_for_demo_without_visible_fallback_reason")
        sponsor_stack = demo.get("sponsor_stack") if isinstance(demo.get("sponsor_stack"), Mapping) else {}
        active_path = (
            sponsor_stack.get("hermes_active_model")
            if isinstance(sponsor_stack.get("hermes_active_model"), Mapping)
            else {}
        )
        spark_boundary = (
            "Spark target selected, live evidence pending"
            if active_path.get("spark_local") is True
            else "Hosted fallback selected, Spark-local evidence pending"
        )
        rejected_spark_boundary = (
            "Hosted fallback selected, Spark-local evidence pending"
            if active_path.get("spark_local") is True
            else "Spark target selected, live evidence pending"
        )
        required_dashboard_tokens = [
            "static dry-run package",
            "Static package ready",
            "Live/Spark gaps",
            spark_boundary,
            "scripted_static_ack_until_live_voice_evidence",
            "needs_live_probe",
            "needs_setup",
            "needs_evidence",
        ]
        for token in required_dashboard_tokens:
            if token not in dashboard_html:
                issues.append(f"dashboard:missing_non_live_token:{token}")
        if rejected_spark_boundary in dashboard_html:
            issues.append("dashboard:contradicts_active_model_path")

    demo_gate_ids = {str(gate.get("gate_id")) for gate in demo_closure.get("gates", []) if isinstance(gate, Mapping)}
    plan_gate_ids = {str(gate.get("gate_id")) for gate in plan_closure.get("gates", []) if isinstance(gate, Mapping)}
    if demo_gate_ids != plan_gate_ids:
        issues.append("closure:gates_mismatch_between_demo_and_plan")
    _audit_spark_model_claims(demo=demo, readiness=readiness, spark_matrix=spark_matrix, issues=issues)


def _audit_spark_model_claims(
    *,
    demo: Mapping[str, Any],
    readiness: Mapping[str, Any],
    spark_matrix: Mapping[str, Any],
    issues: list[str],
) -> None:
    recording_readiness = demo.get("recording_readiness") if isinstance(demo.get("recording_readiness"), Mapping) else {}
    sponsor_stack = demo.get("sponsor_stack") if isinstance(demo.get("sponsor_stack"), Mapping) else {}
    active_path = (
        sponsor_stack.get("hermes_active_model")
        if isinstance(sponsor_stack.get("hermes_active_model"), Mapping)
        else {}
    )
    active_model = str(active_path.get("active_model") or "").lower()
    spark_stack = demo.get("spark_stack") if isinstance(demo.get("spark_stack"), Mapping) else {}
    reflex = spark_stack.get("reflex") if isinstance(spark_stack.get("reflex"), Mapping) else {}
    interpreter = spark_stack.get("interpreter") if isinstance(spark_stack.get("interpreter"), Mapping) else {}
    reflex_model = str(reflex.get("model") or "").lower()
    interpreter_model = str(interpreter.get("model") or "").lower()
    spark_local = active_path.get("spark_local")
    hosted_marker_present = any(marker in active_model for marker in HOSTED_MODEL_MARKERS)
    local_marker_present = any(marker in active_model for marker in LOCAL_MODEL_MARKERS)
    if spark_local is True and hosted_marker_present:
        issues.append("spark_model_claim:spark_local_true_for_hosted_model")
    if spark_local is True and not local_marker_present:
        issues.append("spark_model_claim:spark_local_true_without_local_marker")
    if active_path.get("fallback_used") is True and spark_local is True:
        issues.append("spark_model_claim:fallback_used_but_spark_local_true")
    if "gemma" in reflex_model:
        issues.append("spark_model_claim:gemma_model_in_reflex_role")
    if "gemma" not in interpreter_model:
        issues.append("spark_model_claim:gemma_interpreter_missing")
    expected_status = (
        "target_selected_needs_benchmark_evidence"
        if spark_local is True
        else "hosted_or_nonlocal_path_not_spark_evidence"
    )
    if readiness.get("spark_local_evidence_status") != expected_status:
        issues.append("spark_model_claim:readiness_status_mismatch")
    for key in ("spark_local_readiness", "spark_benchmark_required", "spark_readiness_source"):
        if recording_readiness.get(key) != readiness.get(key):
            issues.append(f"spark_model_claim:demo_{key}_mismatch")
    if not spark_matrix:
        return
    matrix_ready = spark_matrix.get("ready_for_one_spark_demo") is True
    if readiness.get("spark_local_readiness") is not matrix_ready:
        issues.append("spark_model_claim:spark_local_readiness_mismatch")
    expected_benchmark_required = not matrix_ready
    if readiness.get("spark_benchmark_required") is not expected_benchmark_required:
        issues.append("spark_model_claim:spark_benchmark_required_mismatch")
    if readiness.get("spark_readiness_source") != "voiceops_spark_matrix.ready_for_one_spark_demo":
        issues.append("spark_model_claim:readiness_source_mismatch")
    missing_evidence = readiness.get("live_demo_missing_evidence") or []
    if matrix_ready is False and "local_spark_stack_matrix" not in missing_evidence:
        issues.append("spark_model_claim:missing_m4_live_evidence_gap")


def _audit_provider_role_matrix(
    *,
    demo: Mapping[str, Any],
    dashboard_html: str,
    issues: list[str],
) -> None:
    matrix = demo.get("provider_role_matrix")
    if not isinstance(matrix, list):
        issues.append("provider_role_matrix:missing_or_not_list")
        return
    role_items = {str(item.get("role")): item for item in matrix if isinstance(item, Mapping)}
    expected_roles = set(EXPECTED_PROVIDER_ROLE_AUTHORITIES)
    if set(role_items) != expected_roles:
        issues.append("provider_role_matrix:roles_mismatch")

    spark_stack = demo.get("spark_stack") if isinstance(demo.get("spark_stack"), Mapping) else {}
    reflex = spark_stack.get("reflex") if isinstance(spark_stack.get("reflex"), Mapping) else {}
    interpreter = spark_stack.get("interpreter") if isinstance(spark_stack.get("interpreter"), Mapping) else {}
    oracle = spark_stack.get("oracle") if isinstance(spark_stack.get("oracle"), Mapping) else {}
    expected_selected = {
        "reflex": reflex.get("model"),
        "interpreter": interpreter.get("model"),
        "oracle": oracle.get("model"),
    }

    for role, expected_authority in EXPECTED_PROVIDER_ROLE_AUTHORITIES.items():
        item = role_items.get(role)
        if not isinstance(item, Mapping):
            issues.append(f"provider_role_matrix:{role}:missing")
            continue
        if item.get("authority") != expected_authority:
            issues.append(f"provider_role_matrix:{role}:authority_mismatch")
        if role in expected_selected and item.get("selected_label") != expected_selected[role]:
            issues.append(f"provider_role_matrix:{role}:selected_label_mismatch")
        for field in ("selected_label", "candidate_class", "primary_signal", "status"):
            if not str(item.get(field) or "").strip():
                issues.append(f"provider_role_matrix:{role}:{field}_missing")
        for field in ("allowed_outputs", "must_not", "evidence_required"):
            values = item.get(field)
            if not isinstance(values, list) or not values:
                issues.append(f"provider_role_matrix:{role}:{field}_missing")

    oracle_must_not = role_items.get("oracle", {}).get("must_not") if isinstance(role_items.get("oracle"), Mapping) else []
    if not any("oracle_model" in str(value) for value in oracle_must_not or []):
        issues.append("provider_role_matrix:oracle:missing_oracle_model_boundary")
    aux_must_not = (
        role_items.get("auxiliary_transcript_evidence", {}).get("must_not")
        if isinstance(role_items.get("auxiliary_transcript_evidence"), Mapping)
        else []
    )
    if not any("schedule a second Hermes turn" in str(value) for value in aux_must_not or []):
        issues.append("provider_role_matrix:auxiliary_transcript_evidence:missing_no_second_turn_boundary")
    if not any("spend reason" in str(value) for value in aux_must_not or []):
        issues.append("provider_role_matrix:auxiliary_transcript_evidence:missing_high_risk_boundary")

    for token in ("Provider Roles", *EXPECTED_PROVIDER_ROLE_AUTHORITIES.keys(), *EXPECTED_PROVIDER_ROLE_AUTHORITIES.values()):
        if token not in dashboard_html:
            issues.append(f"dashboard:missing_provider_role_token:{token}")


def _audit_action_consistency(
    *,
    demo: Mapping[str, Any],
    packet: Mapping[str, Any],
    packet_validation: Mapping[str, Any],
    operator_state: Mapping[str, Any],
    audit_rows: list[dict[str, Any]],
    operator_state_event_rows: list[dict[str, Any]],
    dry_run_rows: list[dict[str, Any]],
    issues: list[str],
) -> None:
    if packet_validation.get("status") != "valid" or packet_validation.get("ok") is not True:
        issues.append("nemoclaw:validation_not_valid")
    if audit_rows != list(demo.get("audit_events") or []):
        issues.append("audit_ledger:rows_mismatch_demo_audit_events")
    if operator_state_event_rows != list(operator_state.get("recent_audit_events") or []):
        issues.append("operator_state_events:rows_mismatch_operator_state")

    actions = {
        str(action.get("action_id")): action
        for action in packet.get("approval_required_actions", [])
        if isinstance(action, Mapping)
    }
    contracts = packet.get("approval_contracts") if isinstance(packet.get("approval_contracts"), Mapping) else {}
    pending = {
        str(item.get("action_id")): item
        for item in operator_state.get("pending_approvals", [])
        if isinstance(item, Mapping)
    }
    state_contracts = (
        operator_state.get("approval_contracts")
        if isinstance(operator_state.get("approval_contracts"), Mapping)
        else {}
    )
    audit_by_action = {str(row.get("action")): row for row in audit_rows}
    dry_run_by_action = {str(row.get("action_id")): row for row in dry_run_rows}

    if set(actions) != set(pending):
        issues.append("approvals:pending_action_ids_do_not_match_nemoclaw_actions")
    for action_id, action in actions.items():
        command = str(action.get("command") or "")
        contract = action.get("approval_contract") if isinstance(action.get("approval_contract"), Mapping) else {}
        indexed_contract = contracts.get(action_id)
        pending_item = pending.get(action_id, {})
        pending_contract = (
            pending_item.get("approval_contract") if isinstance(pending_item.get("approval_contract"), Mapping) else {}
        )
        state_contract = state_contracts.get(action_id) if isinstance(state_contracts, Mapping) else None
        expected_hash = hashlib.sha256(command.encode("utf-8")).hexdigest()
        if contract.get("command_sha256") != expected_hash:
            issues.append(f"nemoclaw:{action_id}:command_sha256_mismatch")
        if indexed_contract != contract:
            issues.append(f"nemoclaw:{action_id}:indexed_contract_mismatch")
        if state_contract != contract:
            issues.append(f"operator_state:{action_id}:approval_contract_mismatch")
        if _approval_contract_subset(pending_contract) != _approval_contract_subset(contract):
            issues.append(f"operator_state:{action_id}:pending_contract_mismatch")
        action_evidence = action.get("kame_evidence") if isinstance(action.get("kame_evidence"), Mapping) else {}
        action_tool_ref = action.get("tool_disclosure_ref")
        if pending_item.get("kame_evidence") != action_evidence:
            issues.append(f"operator_state:{action_id}:pending_kame_evidence_mismatch")
        if pending_item.get("tool_disclosure_ref") != action_tool_ref:
            issues.append(f"operator_state:{action_id}:pending_tool_disclosure_ref_mismatch")
        if action_tool_ref != "tool_disclosure":
            issues.append(f"nemoclaw:{action_id}:tool_disclosure_ref_missing")
        if pending_item.get("execution_status") != "not_executed":
            issues.append(f"operator_state:{action_id}:pending_approval_executed")
        if pending_item.get("status") not in {"pending", "held"}:
            issues.append(f"operator_state:{action_id}:pending_status_invalid")
        if contract.get("default_decision") != "hold":
            issues.append(f"approvals:{action_id}:default_decision_not_hold")
        if contract.get("approved_by_ref") is not None:
            issues.append(f"approvals:{action_id}:approved_by_ref_present")

        audit = audit_by_action.get(action_id)
        if not audit:
            issues.append(f"audit_ledger:{action_id}:missing")
        else:
            if audit.get("approval_required") is not True:
                issues.append(f"audit_ledger:{action_id}:approval_required_not_true")
            if audit.get("approval_status") not in {"pending_operator_approval", "held_budget"}:
                issues.append(f"audit_ledger:{action_id}:approval_status_invalid")
            if audit.get("result") == "executed":
                issues.append(f"audit_ledger:{action_id}:result_executed")
            if audit.get("status") not in {"queued", "held-budget", "blocked"}:
                issues.append(f"audit_ledger:{action_id}:status_invalid")
            if audit.get("kame_evidence") != action_evidence:
                issues.append(f"audit_ledger:{action_id}:kame_evidence_mismatch")
            if audit.get("tool_disclosure_ref") != action_tool_ref:
                issues.append(f"audit_ledger:{action_id}:tool_disclosure_ref_mismatch")

        dry_run = dry_run_by_action.get(action_id)
        if not dry_run:
            issues.append(f"stripe_actions:{action_id}:metadata_missing")
        else:
            if dry_run.get("command") != command:
                issues.append(f"stripe_actions:{action_id}:command_mismatch")
            if dry_run.get("command_sha256") != expected_hash:
                issues.append(f"stripe_actions:{action_id}:command_sha256_mismatch")
            if dry_run.get("provider_command_executes") is not False:
                issues.append(f"stripe_actions:{action_id}:provider_command_executes_not_false")
            if dry_run.get("execution_mode") != "dry_run_printf_only":
                issues.append(f"stripe_actions:{action_id}:execution_mode_invalid")

    for row in audit_rows:
        if row.get("result") == "executed":
            issues.append(f"audit_ledger:{row.get('action')}:unexpected_executed_result")


def _audit_service_claims(operator_state: Mapping[str, Any], issues: list[str]) -> None:
    for issue in validate_operator_state(dict(operator_state)):
        issues.append(f"operator_state:validation:{issue}")
    for service in operator_state.get("planned_services", []):
        if not isinstance(service, Mapping):
            continue
        if service.get("external") is True:
            if service.get("execution_status") != "not_executed":
                issues.append(f"planned_services:{service.get('service_id')}:external_execution_claim")
            if service.get("status") not in {"approval_required", "planned", "queued"}:
                issues.append(f"planned_services:{service.get('service_id')}:external_status_invalid")
    for service in operator_state.get("provisioned_services", []):
        if not isinstance(service, Mapping):
            continue
        if service.get("external") is True:
            issues.append(f"provisioned_services:{service.get('service_id')}:external_service_claimed_provisioned")
        if service.get("execution_status") not in {"local_artifact_written", "not_executed"}:
            issues.append(f"provisioned_services:{service.get('service_id')}:execution_status_invalid")


def _audit_execution_plan_contracts(execution_plan: Mapping[str, Any], issues: list[str]) -> None:
    contracts = (
        execution_plan.get("approval_contracts")
        if isinstance(execution_plan.get("approval_contracts"), Mapping)
        else {}
    )
    receipts = execution_plan.get("receipts") if isinstance(execution_plan.get("receipts"), Mapping) else {}
    for action in execution_plan.get("approval_required_actions", []):
        if not isinstance(action, Mapping):
            issues.append("execution_plan:approval_required_action_not_object")
            continue
        action_id = str(action.get("action_id") or "unknown")
        command = str(action.get("command") or "")
        expected_hash = hashlib.sha256(command.encode("utf-8")).hexdigest()
        if action.get("command_sha256") != expected_hash:
            issues.append(f"execution_plan:{action_id}:command_sha256_mismatch")
        contract = action.get("approval_contract") if isinstance(action.get("approval_contract"), Mapping) else {}
        if contract and contract.get("command_sha256") != expected_hash:
            issues.append(f"execution_plan:{action_id}:approval_contract_command_sha256_mismatch")
        indexed_contract = contracts.get(action_id) if isinstance(contracts, Mapping) else None
        if isinstance(indexed_contract, Mapping) and indexed_contract.get("command_sha256") != expected_hash:
            issues.append(f"execution_plan:{action_id}:indexed_contract_command_sha256_mismatch")
        receipt_ref = str(action.get("expected_receipt_ref") or "")
        receipt_key = receipt_ref.split(".", 1)[1] if receipt_ref.startswith("receipts.") else ""
        receipt_slot = receipts.get(receipt_key) if isinstance(receipts, Mapping) else None
        if isinstance(receipt_slot, Mapping) and receipt_slot.get("command_sha256") != expected_hash:
            issues.append(f"execution_plan:{action_id}:receipt_slot_command_sha256_mismatch")


def _audit_execution_plan_approval_surfaces(
    *,
    execution_plan: Mapping[str, Any],
    packet: Mapping[str, Any],
    channel_policy: Mapping[str, Any],
    issues: list[str],
) -> None:
    packet_action_ids = {
        str(action.get("action_id"))
        for action in packet.get("approval_required_actions", [])
        if isinstance(action, Mapping) and action.get("action_id")
    }
    route_ids = {
        str(route.get("route_id"))
        for route in channel_policy.get("approval_routing", [])
        if isinstance(route, Mapping) and route.get("route_id")
    }
    policy_scope = channel_policy.get("scope") if isinstance(channel_policy.get("scope"), Mapping) else {}
    channel_policy_ready = (
        policy_scope.get("real_egress_enabled") is False
        and policy_scope.get("review_required_for_real_egress") is True
        and policy_scope.get("review_status") == "pending_human_review"
        and "customer_visible_outbound" in route_ids
    )

    for action in execution_plan.get("approval_required_actions", []):
        if not isinstance(action, Mapping):
            continue
        action_id = str(action.get("action_id") or "unknown")
        approval_artifact = str(action.get("approval_artifact") or "")
        if action_id in packet_action_ids:
            continue
        if approval_artifact == "nemoclaw-action-packet.json":
            issues.append(f"execution_plan:{action_id}:missing_nemoclaw_packet_action")
        elif approval_artifact == "channel-policy.json":
            requires = {str(item) for item in action.get("requires", []) if item}
            contract = action.get("approval_contract") if isinstance(action.get("approval_contract"), Mapping) else {}
            required_gates = {str(item) for item in contract.get("required_preflight_gates", []) if item}
            if "channel_policy" not in requires and "channel_policy" not in required_gates:
                issues.append(f"execution_plan:{action_id}:channel_policy_gate_missing")
            if not channel_policy_ready:
                issues.append(f"execution_plan:{action_id}:channel_policy_surface_not_ready")
        else:
            issues.append(f"execution_plan:{action_id}:unknown_approval_artifact:{approval_artifact or 'missing'}")


def _audit_spark_evidence_scaffold(evidence_path: Path, issues: list[str]) -> None:
    matrix = build_matrix([evidence_path])
    if matrix.get("ready_for_one_spark_demo") is True:
        issues.append("spark_evidence_scaffold:unexpectedly_valid")
    if not evidence_path.exists():
        issues.append(f"spark_evidence_scaffold:missing:{evidence_path}")

    unexpected_issues: set[str] = set()
    observed_issues: set[str] = set()
    for load_issue in matrix.get("evidence_load_issues", []):
        unexpected_issues.add(f"evidence_load:{load_issue}")
    for evaluation in matrix.get("evaluations", []):
        scope = str(evaluation.get("candidate_id") or "unknown_candidate")
        for issue in evaluation.get("issues", []):
            issue_text = str(issue)
            observed_issues.add(issue_text)
            if issue_text not in EXPECTED_SPARK_SCAFFOLD_LINT_ISSUES:
                unexpected_issues.add(f"{scope}:{issue_text}")
    stack_smoke = matrix.get("stack_smoke") if isinstance(matrix.get("stack_smoke"), Mapping) else {}
    for issue in stack_smoke.get("issues", []):
        issue_text = str(issue)
        observed_issues.add(issue_text)
        if issue_text not in EXPECTED_SPARK_SCAFFOLD_LINT_ISSUES:
            unexpected_issues.add(f"stack_smoke:{issue_text}")

    if "example_only_evidence_not_accepted" not in observed_issues:
        unexpected_issues.add("missing_example_only_rejection")
    for issue in sorted(unexpected_issues):
        issues.append(f"spark_evidence_scaffold:{issue}")


def _iter_plan_run_commands(value: Any) -> list[str]:
    commands: list[str] = []
    if isinstance(value, str):
        if "scripts/voiceops_plan_run.py" in value:
            commands.append(value)
    elif isinstance(value, Mapping):
        for nested in value.values():
            commands.extend(_iter_plan_run_commands(nested))
    elif isinstance(value, list):
        for nested in value:
            commands.extend(_iter_plan_run_commands(nested))
    return commands


def _audit_plan_run_package_audit_summary(
    *,
    artifact_root: Path,
    plan_run: Mapping[str, Any],
    checked_artifacts: list[str],
    issues: list[str],
) -> None:
    package_audit = plan_run.get("package_audit")
    if package_audit is None:
        return
    if not isinstance(package_audit, Mapping):
        issues.append("plan_run:package_audit_summary_not_object")
        return
    expected = {
        "ok": True,
        "status": "pass",
        "issues": [],
        "checked_artifact_count": len(checked_artifacts),
    }
    for key, expected_value in expected.items():
        if package_audit.get(key) != expected_value:
            issues.append(f"plan_run:package_audit_summary_mismatch:{key}")
    artifacts = package_audit.get("artifacts")
    if artifacts is not None:
        if not isinstance(artifacts, Mapping):
            issues.append("plan_run:package_audit_artifacts_not_object")
            return
        expected_dir = artifact_root / "voiceops-package-audit" / "current"
        for key in ("json", "markdown"):
            path_text = artifacts.get(key)
            if not isinstance(path_text, str) or not path_text:
                issues.append(f"plan_run:package_audit_artifact_missing:{key}")
                continue
            path = Path(path_text)
            try:
                resolved = path.resolve()
                expected_resolved = expected_dir.resolve()
            except OSError:
                issues.append(f"plan_run:package_audit_artifact_unresolvable:{key}")
                continue
            if not (resolved == expected_resolved or expected_resolved in resolved.parents):
                issues.append(f"plan_run:package_audit_artifact_outside_output_dir:{key}")
            if not path.is_file():
                issues.append(f"plan_run:package_audit_artifact_missing_file:{key}")


def _audit_plan_consistency(
    *,
    demo: Mapping[str, Any],
    demo_closure: Mapping[str, Any],
    demo_handoff: Mapping[str, Any],
    plan_run: Mapping[str, Any],
    plan_closure: Mapping[str, Any],
    plan_handoff: Mapping[str, Any],
    issues: list[str],
) -> None:
    if plan_run.get("artifact_id") != "voiceops-plan-run":
        issues.append("plan_run:artifact_id_mismatch")
    if plan_run.get("artifact_only") is not True:
        issues.append("plan_run:artifact_only_not_true")
    if plan_closure.get("artifact_only") is not True:
        issues.append("plan_closure:artifact_only_not_true")
    if plan_run.get("ok") is not True:
        issues.append("plan_run:ok_not_true")
    if plan_run.get("hard_failures") != []:
        issues.append("plan_run:hard_failures_not_empty")
    if plan_run.get("closure_index") != plan_closure:
        issues.append("plan_run:closure_index_mismatch")
    if plan_run.get("closure_status") != plan_closure.get("closure_status"):
        issues.append("plan_run:closure_status_mismatch")
    if plan_run.get("readiness_gaps") != plan_closure.get("readiness_gaps"):
        issues.append("plan_run:readiness_gaps_mismatch")
    expected_readiness_ok = (
        plan_closure.get("closure_status") == "complete"
        and plan_closure.get("readiness_gaps") == []
    )
    if plan_run.get("readiness_ok") != expected_readiness_ok:
        issues.append("plan_run:readiness_ok_mismatch")
    if plan_run.get("current_environment_blockers") != plan_closure.get(
        "current_environment_blockers"
    ):
        issues.append("plan_run:current_environment_blockers_mismatch")
    expected_remaining_gate_ids = [
        str(gate.get("gate_id"))
        for gate in plan_closure.get("remaining_gates", [])
        if isinstance(gate, Mapping)
    ]
    if plan_run.get("remaining_gates") != expected_remaining_gate_ids:
        issues.append("plan_run:remaining_gates_mismatch")
    if plan_run.get("next_actions") != plan_closure.get("next_actions"):
        issues.append("plan_run:next_actions_mismatch")
    if plan_run.get("review_actions") != plan_closure.get("review_actions"):
        issues.append("plan_run:review_actions_mismatch")
    _audit_next_action_command_order("plan_run", plan_run.get("next_actions"), issues)
    _audit_next_action_command_order("plan_closure", plan_closure.get("next_actions"), issues)
    _audit_review_actions("plan_run", plan_run.get("review_actions"), issues)
    _audit_review_actions("plan_closure", plan_closure.get("review_actions"), issues)
    _audit_plan_safety("plan_run", plan_run.get("safety"), issues)
    _audit_plan_safety("plan_closure", plan_closure.get("safety"), issues)
    if plan_handoff != plan_closure.get("operator_handoff"):
        issues.append("operator_handoff:mismatch_with_closure")
    blockers_ref = plan_handoff.get("diagnostic_blockers_ref")
    if blockers_ref and (
        blockers_ref not in plan_run
        and blockers_ref not in plan_closure
    ):
        issues.append("operator_handoff:diagnostic_blockers_ref_unresolvable")
    _audit_plan_model_args(demo=demo, plan_run=plan_run, issues=issues)
    for label, payload in (
        ("demo_closure", demo_closure),
        ("demo_handoff", demo_handoff),
        ("plan_closure", plan_closure),
        ("operator_handoff", plan_handoff),
    ):
        for command in _iter_plan_run_commands(payload):
            if "--package-audit" not in command:
                issues.append(f"{label}:plan_run_command_missing_package_audit")
            _audit_plan_command_model_args(label=label, command=command, plan_run=plan_run, issues=issues)

    plan_final_command = plan_handoff.get("final_reindex_command")
    if demo_handoff.get("final_reindex_command") != plan_final_command:
        issues.append("demo_handoff:final_reindex_command_mismatch")
    package_audit_command = plan_handoff.get("final_package_audit_command")
    if not package_audit_command or "voiceops_artifact_package_audit.py" not in str(package_audit_command):
        issues.append("operator_handoff:missing_final_package_audit_command")
    if demo_handoff.get("final_package_audit_command") != package_audit_command:
        issues.append("demo_handoff:final_package_audit_command_mismatch")
    final_success_signal = str(plan_handoff.get("final_success_signal") or "")
    if "package_audit.status is pass" not in final_success_signal:
        issues.append("operator_handoff:final_success_signal_missing_package_audit")
    if demo_handoff.get("final_success_signal") != plan_handoff.get("final_success_signal"):
        issues.append("demo_handoff:final_success_signal_mismatch")
    _audit_handoff_phase_contract("operator_handoff", plan_handoff, issues)
    _audit_handoff_phase_contract("demo_handoff", demo_handoff, issues)
    _audit_handoff_validation_command_safety(
        "operator_handoff",
        plan_closure.get("next_actions"),
        plan_handoff,
        issues,
    )
    _audit_handoff_validation_command_safety(
        "demo_handoff",
        plan_closure.get("next_actions"),
        demo_handoff,
        issues,
    )
    _audit_handoff_review_phase_contract("operator_handoff", plan_handoff, issues)
    _audit_handoff_review_phase_contract("demo_handoff", demo_handoff, issues)
    if demo_handoff.get("review_phases") != plan_handoff.get("review_phases"):
        issues.append("demo_handoff:review_phases_mismatch")
    demo_phases = _handoff_phases_by_id(demo_handoff)
    plan_phases = _handoff_phases_by_id(plan_handoff)
    if set(demo_phases) != set(plan_phases):
        issues.append("demo_handoff:phase_ids_mismatch")
        return
    for phase_id, plan_phase in plan_phases.items():
        demo_phase = demo_phases[phase_id]
        for key in ("order", "commands", "expected_artifacts", "success_check"):
            if demo_phase.get(key) != plan_phase.get(key):
                issues.append(f"demo_handoff:{phase_id}:{key}_mismatch")


def _audit_plan_model_args(*, demo: Mapping[str, Any], plan_run: Mapping[str, Any], issues: list[str]) -> None:
    plan_args = plan_run.get("plan_args")
    if plan_args is None:
        return
    if not isinstance(plan_args, Mapping):
        issues.append("plan_run:plan_args_not_object")
        return
    active_model_arg = plan_args.get("active_model")
    reflex_model_arg = plan_args.get("reflex_model")
    interpreter_model_arg = plan_args.get("interpreter_model")
    sponsor_stack = demo.get("sponsor_stack") if isinstance(demo.get("sponsor_stack"), Mapping) else {}
    active_path = (
        sponsor_stack.get("hermes_active_model")
        if isinstance(sponsor_stack.get("hermes_active_model"), Mapping)
        else {}
    )
    spark_stack = demo.get("spark_stack") if isinstance(demo.get("spark_stack"), Mapping) else {}
    reflex = spark_stack.get("reflex") if isinstance(spark_stack.get("reflex"), Mapping) else {}
    interpreter = spark_stack.get("interpreter") if isinstance(spark_stack.get("interpreter"), Mapping) else {}
    if active_model_arg is not None and active_model_arg != active_path.get("active_model"):
        issues.append("plan_run:active_model_arg_mismatch_demo")
    if reflex_model_arg is not None and reflex_model_arg != reflex.get("model"):
        issues.append("plan_run:reflex_model_arg_mismatch_demo")
    if interpreter_model_arg is not None and interpreter_model_arg != interpreter.get("model"):
        issues.append("plan_run:interpreter_model_arg_mismatch_demo")


def _audit_plan_command_model_args(
    *,
    label: str,
    command: str,
    plan_run: Mapping[str, Any],
    issues: list[str],
) -> None:
    plan_args = plan_run.get("plan_args")
    if not isinstance(plan_args, Mapping):
        return
    expected = {
        "--active-model": plan_args.get("active_model"),
        "--reflex-model": plan_args.get("reflex_model"),
        "--interpreter-model": plan_args.get("interpreter_model"),
    }
    if all(value is None for value in expected.values()):
        return
    try:
        parts = shlex.split(command)
    except ValueError:
        issues.append(f"{label}:plan_run_command_parse_failed")
        return
    for flag, expected_value in expected.items():
        if expected_value is None:
            continue
        field = flag.removeprefix("--").replace("-", "_")
        equals_prefix = f"{flag}="
        equals_values = [part.removeprefix(equals_prefix) for part in parts if part.startswith(equals_prefix)]
        if flag not in parts and not equals_values:
            issues.append(f"{label}:plan_run_command_missing_{field}_arg")
            continue
        if equals_values:
            observed = equals_values[0]
        else:
            index = parts.index(flag)
            observed = parts[index + 1] if index + 1 < len(parts) else None
        if observed != expected_value:
            issues.append(f"{label}:plan_run_command_{field}_arg_mismatch")


def _audit_plan_safety(label: str, safety: Any, issues: list[str]) -> None:
    if not isinstance(safety, Mapping):
        issues.append(f"{label}:safety_missing")
        return
    for key in (
        "env_secret_values_emitted",
        "mutating_network_io",
        "live_spend",
        "provider_provisioning",
        "outbound_calls",
    ):
        if safety.get(key) is not False:
            issues.append(f"{label}:safety_{key}_not_false")
    if safety.get("outbound_sends", safety.get("outbound_messages", False)) is not False:
        issues.append(f"{label}:safety_outbound_sends_not_false")
    if safety.get("read_only_discovery_grants_approval") is not False:
        issues.append(f"{label}:safety_read_only_discovery_grants_approval_not_false")
    network_io = safety.get("network_io")
    network_scope = safety.get("network_io_scope")
    if network_io not in {False, True}:
        issues.append(f"{label}:safety_network_io_not_boolean")
    elif network_io is True and network_scope != "allowlisted_read_only_discovery":
        issues.append(f"{label}:safety_network_io_scope_invalid")
    elif network_io is False and network_scope != "none":
        issues.append(f"{label}:safety_network_io_scope_invalid")
    if label == "plan_closure" and safety.get("spark_execution") is not False:
        issues.append(f"{label}:safety_spark_execution_not_false")


def _handoff_phases_by_id(handoff: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    phases = handoff.get("phases")
    if not isinstance(phases, list):
        return {}
    return {
        str(phase.get("phase_id")): phase
        for phase in phases
        if isinstance(phase, Mapping) and str(phase.get("phase_id") or "").strip()
    }


def _handoff_review_phases_by_id(handoff: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    phases = handoff.get("review_phases")
    if not isinstance(phases, list):
        return {}
    return {
        str(phase.get("phase_id")): phase
        for phase in phases
        if isinstance(phase, Mapping) and str(phase.get("phase_id") or "").strip()
    }


def _audit_handoff_phase_contract(label: str, handoff: Mapping[str, Any], issues: list[str]) -> None:
    phases = handoff.get("phases")
    if not isinstance(phases, list):
        issues.append(f"{label}:phases_missing")
        return
    observed = [
        (phase.get("order"), phase.get("phase_id"))
        for phase in phases
        if isinstance(phase, Mapping)
    ]
    if observed != list(EXPECTED_HANDOFF_PHASES):
        issues.append(f"{label}:phase_order_mismatch")
    for expected_order, expected_phase_id in EXPECTED_HANDOFF_PHASES:
        phase = next(
            (
                item
                for item in phases
                if isinstance(item, Mapping)
                and item.get("order") == expected_order
                and item.get("phase_id") == expected_phase_id
            ),
            None,
        )
        if not isinstance(phase, Mapping):
            issues.append(f"{label}:{expected_phase_id}:phase_missing")
            continue
        if not isinstance(phase.get("blocked_by_current_environment"), Mapping):
            issues.append(f"{label}:{expected_phase_id}:missing_environment_blockers")
        _audit_handoff_command_order(label, expected_phase_id, phase, issues)


def _audit_handoff_review_phase_contract(label: str, handoff: Mapping[str, Any], issues: list[str]) -> None:
    phases = handoff.get("review_phases")
    if not isinstance(phases, list):
        issues.append(f"{label}:review_phases_missing")
        return
    observed = [
        (phase.get("order"), phase.get("phase_id"))
        for phase in phases
        if isinstance(phase, Mapping)
    ]
    if observed != list(EXPECTED_REVIEW_PHASES):
        issues.append(f"{label}:review_phase_order_mismatch")
    phase = _handoff_review_phases_by_id(handoff).get("multi_channel_policy_review")
    if not isinstance(phase, Mapping):
        issues.append(f"{label}:missing_channel_policy_review_phase")
        return
    if phase.get("changes_readiness_by_itself") is not False:
        issues.append(f"{label}:channel_policy_review_changes_readiness")
    if phase.get("changes_policy_by_itself") is not False:
        issues.append(f"{label}:channel_policy_review_changes_policy")
    if phase.get("real_egress_enabled") is not False:
        issues.append(f"{label}:channel_policy_review_enables_egress")
    review_command = str(phase.get("review_command") or "")
    first_safe_command = str(phase.get("first_safe_command") or "")
    if "voiceops_channel_policy.py" not in first_safe_command:
        issues.append(f"{label}:channel_policy_review_first_safe_command_invalid")
    if review_command != first_safe_command:
        issues.append(f"{label}:channel_policy_review_command_mismatch")
    artifacts = phase.get("review_artifacts")
    if not isinstance(artifacts, list) or "artifacts/voiceops-channel-policy/current/channel-policy-review.json" not in artifacts:
        issues.append(f"{label}:channel_policy_review_artifact_missing")
    required_review = set(phase.get("required_review") or [])
    if required_review != {"business_owner", "channel_owner", "privacy_reviewer", "security_owner"}:
        issues.append(f"{label}:channel_policy_review_signoffs_mismatch")


def _audit_handoff_command_order(
    label: str,
    phase_id: str,
    phase: Mapping[str, Any],
    issues: list[str],
) -> None:
    commands = phase.get("commands")
    if not isinstance(commands, list) or not commands:
        issues.append(f"{label}:{phase_id}:commands_missing")
        return
    command_texts = [str(command) for command in commands]
    first_safe_command = str(phase.get("first_safe_command") or "")
    first_evidence_command = str(phase.get("first_evidence_command") or "")
    if not first_safe_command:
        issues.append(f"{label}:{phase_id}:missing_first_safe_command")
    elif command_texts[0] != first_safe_command and (
        not phase.get("diagnostic_command") or len(command_texts) < 2 or command_texts[1] != first_safe_command
    ):
        issues.append(f"{label}:{phase_id}:first_safe_command_not_first")
    if phase_id == "live_discord_voice":
        _audit_ordered_handoff_command(
            label,
            phase_id,
            command_texts,
            first_safe_command,
            required_marker="--audit-only",
            issue_suffix="first_safe_command_not_no_write_audit",
            issues=issues,
        )
        _audit_ordered_handoff_command(
            label,
            phase_id,
            command_texts,
            first_evidence_command,
            required_marker="--run-doctor-report",
            issue_suffix="first_evidence_command_not_live_closure",
            issues=issues,
        )
        _audit_handoff_command_precedence(
            label,
            phase_id,
            command_texts,
            first_safe_command,
            first_evidence_command,
            issue_suffix="no_write_audit_not_before_live_collection",
            issues=issues,
        )
    elif phase_id == "local_spark_stack":
        _audit_ordered_handoff_command(
            label,
            phase_id,
            command_texts,
            first_safe_command,
            required_marker="--lint-evidence",
            issue_suffix="first_safe_command_not_spark_lint",
            issues=issues,
        )
        _audit_ordered_handoff_command(
            label,
            phase_id,
            command_texts,
            first_evidence_command,
            required_marker="dgx_spark_gemma4_voice_eval",
            issue_suffix="first_evidence_command_not_dgx_eval",
            issues=issues,
        )
        _audit_handoff_command_precedence(
            label,
            phase_id,
            command_texts,
            first_safe_command,
            first_evidence_command,
            issue_suffix="spark_lint_not_before_dgx_eval",
            issues=issues,
        )
    if first_safe_command and first_evidence_command and first_safe_command != first_evidence_command:
        try:
            safe_index = command_texts.index(first_safe_command)
            evidence_index = command_texts.index(first_evidence_command)
        except ValueError:
            return
        if safe_index >= evidence_index:
            issues.append(f"{label}:{phase_id}:first_safe_command_not_before_first_evidence")


def _audit_handoff_validation_command_safety(
    label: str,
    next_actions: Any,
    handoff: Mapping[str, Any],
    issues: list[str],
) -> None:
    if not isinstance(next_actions, list):
        issues.append(f"{label}:next_actions_missing_for_command_safety")
        return
    phases_by_gate = {
        str(phase.get("gate_id")): phase
        for phase in handoff.get("phases", [])
        if isinstance(phase, Mapping) and str(phase.get("gate_id") or "").strip()
    }
    for action in next_actions:
        if not isinstance(action, Mapping):
            continue
        gate_id = str(action.get("gate_id") or "").strip()
        phase = phases_by_gate.get(gate_id)
        if not isinstance(phase, Mapping):
            issues.append(f"{label}:{gate_id}:missing_phase_for_command_safety")
            continue
        command_safety = phase.get("command_safety")
        if not isinstance(command_safety, Mapping):
            issues.append(f"{label}:{gate_id}:missing_command_safety")
            continue
        validation_commands = action.get("validation_commands")
        if not isinstance(validation_commands, Mapping):
            issues.append(f"{label}:{gate_id}:validation_commands_not_object")
            continue
        for command_key in validation_commands:
            if command_key not in command_safety:
                issues.append(f"{label}:{gate_id}:validation_command_missing_safety:{command_key}")


def _audit_handoff_command_precedence(
    label: str,
    phase_id: str,
    commands: list[str],
    safe_command: str,
    evidence_command: str,
    *,
    issue_suffix: str,
    issues: list[str],
) -> None:
    try:
        safe_index = commands.index(safe_command)
        evidence_index = commands.index(evidence_command)
    except ValueError:
        return
    if safe_index >= evidence_index:
        issues.append(f"{label}:{phase_id}:{issue_suffix}")


def _audit_next_action_command_order(label: str, actions: Any, issues: list[str]) -> None:
    if not isinstance(actions, list):
        issues.append(f"{label}:next_actions_missing")
        return
    actions_by_gate = {
        str(action.get("gate_id")): action
        for action in actions
        if isinstance(action, Mapping)
    }
    live = actions_by_gate.get("live_discord_voice_operator")
    if isinstance(live, Mapping):
        live_safe = str(live.get("first_safe_command") or "")
        live_evidence = str(live.get("first_evidence_command") or "")
        if "--audit-only" not in live_safe:
            issues.append(f"{label}:live_discord_voice_operator:first_safe_command_not_no_write_audit")
        if "--run-doctor-report" not in live_evidence:
            issues.append(f"{label}:live_discord_voice_operator:first_evidence_command_not_realtime_voice_closure")
        if live_safe == live_evidence:
            issues.append(f"{label}:live_discord_voice_operator:first_safe_command_equals_first_evidence")
    spark = actions_by_gate.get("local_spark_stack_matrix")
    if isinstance(spark, Mapping):
        spark_safe = str(spark.get("first_safe_command") or "")
        spark_evidence = str(spark.get("first_evidence_command") or "")
        if "--lint-evidence" not in spark_safe:
            issues.append(f"{label}:local_spark_stack_matrix:first_safe_command_not_spark_lint")
        if "dgx_spark_gemma4_voice_eval" not in spark_evidence:
            issues.append(f"{label}:local_spark_stack_matrix:first_evidence_command_not_dgx_eval")
        if spark_safe == spark_evidence:
            issues.append(f"{label}:local_spark_stack_matrix:first_safe_command_equals_first_evidence")


def _audit_review_actions(label: str, actions: Any, issues: list[str]) -> None:
    if not isinstance(actions, list):
        issues.append(f"{label}:review_actions_missing")
        return
    actions_by_phase = {
        str(action.get("phase_id")): action
        for action in actions
        if isinstance(action, Mapping)
    }
    action = actions_by_phase.get("multi_channel_policy_review")
    if not isinstance(action, Mapping):
        issues.append(f"{label}:missing_channel_policy_review_action")
        return
    if action.get("status") != "pending_human_review":
        issues.append(f"{label}:channel_policy_review_status_not_pending")
    if action.get("changes_readiness_by_itself") is not False:
        issues.append(f"{label}:channel_policy_review_changes_readiness")
    if action.get("changes_policy_by_itself") is not False:
        issues.append(f"{label}:channel_policy_review_changes_policy")
    if action.get("real_egress_enabled") is not False:
        issues.append(f"{label}:channel_policy_review_enables_egress")
    command = str(action.get("review_command") or "")
    if "voiceops_channel_policy.py" not in command:
        issues.append(f"{label}:channel_policy_review_command_invalid")


def _audit_ordered_handoff_command(
    label: str,
    phase_id: str,
    commands: list[str],
    command: str,
    *,
    required_marker: str,
    issue_suffix: str,
    issues: list[str],
) -> None:
    if not command:
        issues.append(f"{label}:{phase_id}:missing_{issue_suffix.removeprefix('first_')}")
        return
    if command not in commands:
        issues.append(f"{label}:{phase_id}:{issue_suffix}_not_listed")
    if required_marker not in command:
        issues.append(f"{label}:{phase_id}:{issue_suffix}")


def _audit_channel_policy(policy: Mapping[str, Any], review: Mapping[str, Any], issues: list[str]) -> None:
    for issue in validate_policy(dict(policy)):
        issues.append(f"channel_policy:validation:{issue}")

    scope = policy.get("scope") if isinstance(policy.get("scope"), Mapping) else {}
    if scope.get("real_egress_enabled") is not False:
        issues.append("channel_policy:real_egress_enabled_not_false")
    if scope.get("review_required_for_real_egress") is not True:
        issues.append("channel_policy:review_required_for_real_egress_not_true")
    if scope.get("review_status") != "pending_human_review":
        issues.append("channel_policy:review_status_not_pending")
    if set(scope.get("channels") or []) != set(CHANNEL_IDS):
        issues.append("channel_policy:scope_channels_mismatch")

    channel_ids = {
        str(channel.get("channel_id"))
        for channel in policy.get("channel_authorization", [])
        if isinstance(channel, Mapping)
    }
    if channel_ids != set(CHANNEL_IDS):
        issues.append("channel_policy:channel_authorization_mismatch")
    policy_channels = {
        str(channel.get("channel_id")): channel
        for channel in policy.get("channel_authorization", [])
        if isinstance(channel, Mapping)
    }
    if review.get("schema_version") != "voiceops.multi_channel_policy_review.v1":
        issues.append("channel_policy_review:schema_version_mismatch")
    if review.get("artifact_id") != "voiceops-m3-channel-policy-review":
        issues.append("channel_policy_review:artifact_id_mismatch")
    for key in ("milestone", "policy_id", "policy_version"):
        if review.get(key) != policy.get(key):
            issues.append(f"channel_policy_review:{key}_mismatch")
    if review.get("artifact_only") is not True:
        issues.append("channel_policy_review:artifact_only_not_true")
    if review.get("policy_ref") != "channel-policy.json":
        issues.append("channel_policy_review:policy_ref_mismatch")
    if review.get("review_status") != "pending_human_review":
        issues.append("channel_policy_review:review_status_not_pending")
    if review.get("real_egress_enabled") is not False:
        issues.append("channel_policy_review:real_egress_enabled_not_false")
    if review.get("changes_policy") is not False:
        issues.append("channel_policy_review:changes_policy_not_false")
    decision_options = set(review.get("decision_options") or [])
    if {"request_changes", "deny", "approve_dry_run_only"} - decision_options:
        issues.append("channel_policy_review:decision_options_missing_safe_choices")
    review_channels = {
        str(channel.get("channel_id")): channel
        for channel in review.get("per_channel_review", [])
        if isinstance(channel, Mapping)
    }
    if set(review_channels) != set(CHANNEL_IDS):
        issues.append("channel_policy_review:per_channel_mismatch")
    for channel_id, channel in review_channels.items():
        if channel.get("live_egress_enabled") is not False:
            issues.append(f"channel_policy_review:{channel_id}:live_egress_enabled_not_false")
        if channel.get("review_status") != "pending_human_review":
            issues.append(f"channel_policy_review:{channel_id}:review_status_not_pending")
        policy_channel = policy_channels.get(channel_id, {})
        if set(channel.get("required_evidence") or []) != set(policy_channel.get("evidence_required") or []):
            issues.append(f"channel_policy_review:{channel_id}:required_evidence_mismatch")
        route_map = policy.get("approval_route_map") if isinstance(policy.get("approval_route_map"), Mapping) else {}
        if dict(channel.get("approval_routes_to_confirm") or {}) != dict(route_map.get(channel_id) or {}):
            issues.append(f"channel_policy_review:{channel_id}:approval_routes_mismatch")
        if set(channel.get("blocked_capabilities_to_confirm") or []) != set(policy_channel.get("prohibited_actions") or []):
            issues.append(f"channel_policy_review:{channel_id}:blocked_capabilities_mismatch")
    phone_review = review_channels.get("phone_sms", {})
    phone_routes = set((phone_review.get("approval_routes_to_confirm") or {}).keys())
    if {"any_sms_send", "approved_phone_handoff_call", "customer_visible_handoff"} - phone_routes:
        issues.append("channel_policy_review:phone_sms:approval_routes_mismatch")
    blocked_capabilities = set(scope.get("blocked_capabilities") or [])
    if {"sms_send_without_approval", "voice_call"} - blocked_capabilities:
        issues.append("channel_policy:phone_sms_blocked_capabilities_missing")

    required_signoff_roles = {"business_owner", "channel_owner", "security_owner", "privacy_reviewer"}
    signoff_roles = {
        str(signoff.get("role"))
        for signoff in review.get("required_signoffs", [])
        if isinstance(signoff, Mapping) and signoff.get("required") is True
    }
    if signoff_roles != required_signoff_roles:
        issues.append("channel_policy_review:required_signoffs_mismatch")
    has_package_audit_review_command = False
    for command in review.get("review_commands", []):
        if isinstance(command, str) and "scripts/voiceops_plan_run.py" in command:
            if "--package-audit" in command:
                has_package_audit_review_command = True
            else:
                issues.append("channel_policy_review:plan_run_command_missing_package_audit")
    if not has_package_audit_review_command:
        issues.append("channel_policy_review:missing_package_audit_review_command")


def _resolve_package_artifact_path(artifact_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == artifact_root.name:
        return artifact_root.parent / path
    return artifact_root / path


def _audit_post_approval_receipt_scaffold(
    scaffold: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    scaffold_path: Path,
    issues: list[str],
) -> None:
    if scaffold.get("example_only") is not True:
        issues.append("post_approval_receipts_scaffold:must_remain_example_only")
    recomputed_scaffold = validate_post_approval_receipts(scaffold, execution_plan, receipt_path=scaffold_path)
    scaffold_issues = recomputed_scaffold.get("validation_issues")
    if recomputed_scaffold.get("status") != "invalid":
        issues.append("post_approval_receipts_scaffold:unexpectedly_valid")
    if not any("example_only" in str(issue) for issue in scaffold_issues or []):
        issues.append("post_approval_receipts_scaffold:missing_example_only_validation_issue")
    for issue in scaffold_issues or []:
        issue_text = str(issue)
        if (
            ":approval_decision_ref:" in issue_text
            and not issue_text.endswith(":file_not_found")
            and ":file_not_found_at:" not in issue_text
        ):
            issues.append(f"post_approval_receipts_scaffold:{issue_text}")


def _audit_post_approval_receipt_validation(
    *,
    artifact_root: Path,
    execution_plan: Mapping[str, Any],
    scaffold: Mapping[str, Any],
    stored_validation: Mapping[str, Any],
    issues: list[str],
) -> None:
    scaffold_path = (
        artifact_root
        / "voiceops-provisioning"
        / "current"
        / "post-approval-receipts-scaffold"
        / "post-approval-receipts.json"
    )
    _audit_post_approval_receipt_scaffold(scaffold, execution_plan, scaffold_path, issues)
    if stored_validation.get("loaded") is True:
        receipt_path_text = str(stored_validation.get("path") or "").strip()
        if not receipt_path_text:
            issues.append("post_approval_receipts_validation:loaded_without_path")
            return
        receipt_path = _resolve_package_artifact_path(artifact_root, receipt_path_text)
        receipt_payload = _read_json(receipt_path, issues, "post_approval_receipts")
        if not isinstance(receipt_payload, Mapping) or not receipt_payload:
            issues.append("post_approval_receipts_validation:loaded_receipts_empty_or_invalid")
            return
        recomputed = validate_post_approval_receipts(receipt_payload, execution_plan, receipt_path=receipt_path)
        for key in (
            "status",
            "validation_issues",
            "receipt_count",
            "credential_location_count",
            "rollback_receipt_count",
            "audit_event_count",
            "ledger_rows",
        ):
            if stored_validation.get(key) != recomputed.get(key):
                issues.append(f"post_approval_receipts_validation:{key}_mismatch")
        return

    if stored_validation.get("status") != "not_supplied":
        issues.append("post_approval_receipts_validation:status_not_not_supplied_without_loaded_receipts")
    if stored_validation.get("validation_issues") not in ([], None):
        issues.append("post_approval_receipts_validation:issues_present_without_loaded_receipts")
    if stored_validation.get("ledger_rows") not in ([], None):
        issues.append("post_approval_receipts_validation:ledger_rows_present_without_loaded_receipts")
    for key in ("receipt_count", "credential_location_count", "rollback_receipt_count", "audit_event_count"):
        if stored_validation.get(key) not in (0, None):
            issues.append(f"post_approval_receipts_validation:{key}_nonzero_without_loaded_receipts")


def _audit_preflight_evidence_scaffold(
    *,
    manifest_path: Path,
    issues: list[str],
) -> None:
    validation = load_preflight_evidence(manifest_path)
    validation_issues = validation.get("validation_issues") or []
    if validation.get("loaded") is not True:
        issues.append("preflight_evidence_scaffold:manifest_not_loaded")
    if not any("example_only" in str(issue) for issue in validation_issues):
        issues.append("preflight_evidence_scaffold:missing_example_only_validation_issue")
    for issue in validation_issues:
        issue_text = str(issue)
        if issue_text.startswith("preflight_evidence_manifest:") and "example_only" not in issue_text:
            issues.append(f"preflight_evidence_scaffold:{issue_text}")


def _audit_voice_operator_artifact_consistency(
    *,
    readiness: Mapping[str, Any],
    discord_loopback_smoke: Mapping[str, Any],
    async_oracle_smoke: Mapping[str, Any],
    discord_session_cleanup_smoke: Mapping[str, Any],
    sidecar_fail_closed_smoke: Mapping[str, Any],
    issues: list[str],
) -> None:
    for issue in validate_voice_operator_report(dict(readiness)):
        issues.append(f"voice_operator_readiness:{issue}")
    expected_payloads = {
        "smoke": discord_loopback_smoke,
        "async_oracle_smoke": async_oracle_smoke,
        "discord_session_cleanup_smoke": discord_session_cleanup_smoke,
        "sidecar_fail_closed_smoke": sidecar_fail_closed_smoke,
    }
    for field, standalone_payload in expected_payloads.items():
        if readiness.get(field) != standalone_payload:
            issues.append(f"voice_operator_readiness:{field}_standalone_artifact_mismatch")
    _audit_voice_operator_proof_consistency(readiness=readiness, issues=issues)


def _compare_proof_fields(
    *,
    label: str,
    proof: Mapping[str, Any],
    expected: Mapping[str, Any],
    issues: list[str],
) -> None:
    for field, expected_value in expected.items():
        if proof.get(field) != expected_value:
            issues.append(f"voice_operator_readiness:proofs.{label}.{field}_mismatch")


def _audit_voice_operator_proof_consistency(*, readiness: Mapping[str, Any], issues: list[str]) -> None:
    proofs = readiness.get("proofs") if isinstance(readiness.get("proofs"), Mapping) else {}
    smoke = readiness.get("smoke") if isinstance(readiness.get("smoke"), Mapping) else {}
    async_smoke = readiness.get("async_oracle_smoke") if isinstance(readiness.get("async_oracle_smoke"), Mapping) else {}
    cleanup_smoke = (
        readiness.get("discord_session_cleanup_smoke")
        if isinstance(readiness.get("discord_session_cleanup_smoke"), Mapping)
        else {}
    )
    sidecar_fail_closed_smoke = (
        readiness.get("sidecar_fail_closed_smoke")
        if isinstance(readiness.get("sidecar_fail_closed_smoke"), Mapping)
        else {}
    )
    pcm_proof = proofs.get("pcm_conversion") if isinstance(proofs.get("pcm_conversion"), Mapping) else {}
    _compare_proof_fields(
        label="pcm_conversion",
        proof=pcm_proof,
        expected={
            "input_pcm48_stereo_bytes": smoke.get("input_pcm48_bytes"),
            "sidecar_pcm16_mono_bytes": smoke.get("sidecar_pcm16_bytes"),
            "sidecar_pcm16_first_sample": smoke.get("sidecar_pcm16_first_sample"),
            "sidecar_pcm16_checksum": smoke.get("sidecar_pcm16_checksum"),
            "sentinel_expected_first_sample": 450,
        },
        issues=issues,
    )
    barge_in_proof = proofs.get("barge_in_energy") if isinstance(proofs.get("barge_in_energy"), Mapping) else {}
    _compare_proof_fields(
        label="barge_in_energy",
        proof=barge_in_proof,
        expected={
            "reaction_proven": bool(smoke.get("barge_in_sent")),
            "speech_energy_event_forwarded": bool(smoke.get("speech_energy_sent")),
            "energy_gate_proven_by_smoke": bool(async_smoke.get("energy_gate_smoke_ok")),
            "energy_gate_ignored_non_speech_packets": async_smoke.get(
                "energy_gate_ignored_non_speech_packets"
            ),
            "energy_gate_low_energy_witness_source": async_smoke.get(
                "energy_gate_low_energy_witness_source"
            ),
            "energy_gate_low_energy_witness_promoted": async_smoke.get(
                "energy_gate_low_energy_witness_promoted"
            ),
            "energy_gate_low_energy_witness_suppressed": async_smoke.get(
                "energy_gate_low_energy_witness_suppressed"
            ),
            "energy_gate_barge_in_events": async_smoke.get("energy_gate_barge_in_events"),
            "energy_gate_oracle_work_events": async_smoke.get("energy_gate_oracle_work_events"),
            "stop_called": int(smoke.get("mixer_stop_calls") or 0) >= 1,
        },
        issues=issues,
    )
    async_proof = proofs.get("async_oracle_jobs") if isinstance(proofs.get("async_oracle_jobs"), Mapping) else {}
    _compare_proof_fields(
        label="async_oracle_jobs",
        proof=async_proof,
        expected={
            "kind": async_smoke.get("kind"),
            "scenario": async_smoke.get("scenario"),
            "max_running": async_smoke.get("max_running"),
            "max_worker_overlap": async_smoke.get("max_worker_overlap"),
            "worker_overlap_proved": bool(async_smoke.get("worker_overlap_proved")),
            "worker_overlap_within_capacity": bool(async_smoke.get("worker_overlap_within_capacity")),
            "noncooperative_cancel_overlap_observed": bool(
                async_smoke.get("noncooperative_cancel_overlap_observed")
            ),
            "started_jobs": async_smoke.get("started_jobs"),
            "queued_jobs": async_smoke.get("queued_jobs"),
            "completed_jobs": async_smoke.get("completed_jobs"),
            "failed_jobs": async_smoke.get("failed_jobs"),
            "cancelled_jobs": async_smoke.get("cancelled_jobs"),
            "shutdown_timeout_configured_ms": async_smoke.get("shutdown_timeout_configured_ms"),
            "shutdown_close_elapsed_ms": async_smoke.get("shutdown_close_elapsed_ms"),
            "shutdown_bounded_close_observed": bool(async_smoke.get("shutdown_bounded_close_observed")),
            "shutdown_forced_cancel_observed": bool(async_smoke.get("shutdown_forced_cancel_observed")),
            "shutdown_close_cancel_entered": bool(async_smoke.get("shutdown_close_cancel_entered")),
            "shutdown_cancelled_jobs": async_smoke.get("shutdown_cancelled_jobs"),
            "queued_cancel_smoke_ok": bool(async_smoke.get("queued_cancel_smoke_ok")),
            "queued_cancel_observed": bool(async_smoke.get("queued_cancel_observed")),
            "queued_cancelled_before_start": bool(async_smoke.get("queued_cancelled_before_start")),
            "queued_cancel_not_sent_to_oracle": bool(async_smoke.get("queued_cancel_not_sent_to_oracle")),
            "queued_cancel_reason": async_smoke.get("queued_cancel_reason"),
            "queued_cancel_target_job_id": async_smoke.get("queued_cancel_target_job_id"),
            "queued_cancel_running_completed": bool(async_smoke.get("queued_cancel_running_completed")),
            "approval_capacity_smoke_ok": bool(async_smoke.get("approval_capacity_smoke_ok")),
            "approval_capacity_waiting_observed": bool(async_smoke.get("approval_capacity_waiting_observed")),
            "approval_capacity_followup_queued": bool(async_smoke.get("approval_capacity_followup_queued")),
            "approval_capacity_active_visible": bool(async_smoke.get("approval_capacity_active_visible")),
            "approval_capacity_misleading_running_capacity": bool(
                async_smoke.get("approval_capacity_misleading_running_capacity")
            ),
            "approval_capacity_status_text": async_smoke.get("approval_capacity_status_text"),
            "approval_capacity_followup_started_after_approval": bool(
                async_smoke.get("approval_capacity_followup_started_after_approval")
            ),
            "approval_capacity_completed_jobs": async_smoke.get("approval_capacity_completed_jobs"),
            "approval_capacity_failed_gate_suppressed": bool(
                async_smoke.get("approval_capacity_failed_gate_suppressed")
            ),
            "approval_capacity_failed_jobs": async_smoke.get("approval_capacity_failed_jobs"),
            "approval_capacity_max_concurrent": async_smoke.get("approval_capacity_max_concurrent"),
            "approval_cancel_capacity_smoke_ok": bool(async_smoke.get("approval_cancel_capacity_smoke_ok")),
            "approval_cancel_waiting_observed": bool(async_smoke.get("approval_cancel_waiting_observed")),
            "approval_cancel_followup_queued": bool(async_smoke.get("approval_cancel_followup_queued")),
            "approval_cancel_requested_observed": bool(async_smoke.get("approval_cancel_requested_observed")),
            "approval_cancel_cancelled_observed": bool(async_smoke.get("approval_cancel_cancelled_observed")),
            "approval_cancel_late_output_attempted": bool(async_smoke.get("approval_cancel_late_output_attempted")),
            "approval_cancel_completed_after_cancel": bool(async_smoke.get("approval_cancel_completed_after_cancel")),
            "approval_cancel_late_result_spoken": bool(async_smoke.get("approval_cancel_late_result_spoken")),
            "approval_cancel_followup_started_before_cancel_drained": bool(
                async_smoke.get("approval_cancel_followup_started_before_cancel_drained")
            ),
            "approval_cancel_followup_started_after_cancel": bool(
                async_smoke.get("approval_cancel_followup_started_after_cancel")
            ),
            "approval_cancel_active_visible": bool(async_smoke.get("approval_cancel_active_visible")),
            "approval_cancel_misleading_running_capacity": bool(
                async_smoke.get("approval_cancel_misleading_running_capacity")
            ),
            "approval_cancel_status_text": async_smoke.get("approval_cancel_status_text"),
            "approval_cancel_max_concurrent": async_smoke.get("approval_cancel_max_concurrent"),
            "cancel_drain_capacity_smoke_ok": bool(async_smoke.get("cancel_drain_capacity_smoke_ok")),
            "cancel_drain_requested_observed": bool(async_smoke.get("cancel_drain_requested_observed")),
            "cancel_drain_cancelled_observed": bool(async_smoke.get("cancel_drain_cancelled_observed")),
            "cancel_drain_followup_queued": bool(async_smoke.get("cancel_drain_followup_queued")),
            "cancel_drain_active_visible": bool(async_smoke.get("cancel_drain_active_visible")),
            "cancel_drain_misleading_running_capacity": bool(
                async_smoke.get("cancel_drain_misleading_running_capacity")
            ),
            "cancel_drain_status_text": async_smoke.get("cancel_drain_status_text"),
            "cancel_drain_followup_started_after_cancel": bool(
                async_smoke.get("cancel_drain_followup_started_after_cancel")
            ),
            "cancel_drain_max_concurrent": async_smoke.get("cancel_drain_max_concurrent"),
            "local_turn_committed": bool(async_smoke.get("local_turn_committed")),
            "local_turn_during_running_jobs_observed": bool(
                async_smoke.get("local_turn_during_running_jobs_observed")
            ),
            "local_turn_active_job_count": async_smoke.get("local_turn_active_job_count"),
            "playback_stop_committed": bool(async_smoke.get("playback_stop_committed")),
            "playback_stop_jobs_still_running": bool(async_smoke.get("playback_stop_jobs_still_running")),
            "playback_stop_cancelled_jobs": bool(async_smoke.get("playback_stop_cancelled_jobs")),
            "playback_stop_does_not_cancel_jobs": bool(async_smoke.get("playback_stop_does_not_cancel_jobs")),
            "status_turn_committed": bool(async_smoke.get("status_turn_committed")),
            "status_turn_queued_visible": bool(async_smoke.get("status_turn_queued_visible")),
            "status_turn_no_oracle_request": bool(async_smoke.get("status_turn_no_oracle_request")),
            "status_turn_oracle_request_count_before": async_smoke.get(
                "status_turn_oracle_request_count_before"
            ),
            "status_turn_oracle_request_count_after": async_smoke.get(
                "status_turn_oracle_request_count_after"
            ),
            "status_text": async_smoke.get("status_text"),
            "terminal_status_committed": bool(async_smoke.get("terminal_status_committed")),
            "completed_result_status_visible": bool(async_smoke.get("completed_result_status_visible")),
            "terminal_status_text": async_smoke.get("terminal_status_text"),
            "fifth_job_id": async_smoke.get("fifth_job_id"),
            "fifth_job_queued": bool(async_smoke.get("fifth_job_queued")),
            "fifth_job_started_after_capacity_freed": bool(
                async_smoke.get("fifth_job_started_after_capacity_freed")
            ),
            "cancelled_job_id": async_smoke.get("cancelled_job_id"),
            "late_cancelled_output_attempted": bool(async_smoke.get("late_cancelled_output_attempted")),
            "cancelled_result_spoken": bool(async_smoke.get("cancelled_result_spoken")),
            "cancelled_result_committed": bool(async_smoke.get("cancelled_result_committed")),
            "cancelled_result_progress_leaked": bool(async_smoke.get("cancelled_result_progress_leaked")),
            "cancelled_result_durable_completed": bool(async_smoke.get("cancelled_result_durable_completed")),
            "cancelled_result_durable_text": bool(async_smoke.get("cancelled_result_durable_text")),
            "durable_cancelled_record_present": bool(async_smoke.get("durable_cancelled_record_present")),
            "durable_completed_jobs": async_smoke.get("durable_completed_jobs"),
            "approval_wait_observed": bool(async_smoke.get("approval_wait_observed")),
            "approval_status_committed": bool(async_smoke.get("approval_status_committed")),
            "approval_tool_progress_observed": bool(async_smoke.get("approval_tool_progress_observed")),
            "approval_payload_redacted": bool(async_smoke.get("approval_payload_redacted")),
            "approval_secret_leaked": bool(async_smoke.get("approval_secret_leaked")),
            "approval_secret_canary_checked": bool(async_smoke.get("approval_secret_canary_checked")),
            "approval_completed": bool(async_smoke.get("approval_completed")),
            "approval_gate_failed_closed": bool(async_smoke.get("approval_gate_failed_closed")),
            "approval_result_suppressed": bool(async_smoke.get("approval_result_suppressed")),
            "approval_status_text": async_smoke.get("approval_status_text"),
            "failed_job_reported": bool(async_smoke.get("failed_job_reported")),
            "failed_job_spoken": bool(async_smoke.get("failed_job_spoken")),
            "durable_failed_record_present": bool(async_smoke.get("durable_failed_record_present")),
            "session_survived_failed_job": bool(async_smoke.get("session_survived_failed_job")),
            "queued_job_update_observed": bool(async_smoke.get("queued_job_update_observed")),
            "running_job_update_observed": bool(async_smoke.get("running_job_update_observed")),
            "running_update_latest_update_visible": bool(async_smoke.get("running_update_latest_update_visible")),
            "running_update_latest_update_text": async_smoke.get("running_update_latest_update_text"),
            "running_update_reached_oracle": bool(async_smoke.get("running_update_reached_oracle")),
            "running_update_delivery_metadata_ok": bool(
                async_smoke.get("running_update_delivery_metadata_ok")
            ),
            "queued_update_latest_update_visible": bool(async_smoke.get("queued_update_latest_update_visible")),
            "queued_update_latest_update_text": async_smoke.get("queued_update_latest_update_text"),
            "queued_update_started_with_priority": bool(async_smoke.get("queued_update_started_with_priority")),
            "queued_update_reached_oracle": bool(async_smoke.get("queued_update_reached_oracle")),
            "queued_interpreter_fold_in_observed": bool(
                async_smoke.get("queued_interpreter_fold_in_observed")
            ),
            "queued_interpreter_fold_in_oracle_text": async_smoke.get(
                "queued_interpreter_fold_in_oracle_text"
            ),
            "queued_interpreter_fold_in_transcript_source": async_smoke.get(
                "queued_interpreter_fold_in_transcript_source"
            ),
            "queued_interpreter_fold_in_transcript_confidence": async_smoke.get(
                "queued_interpreter_fold_in_transcript_confidence"
            ),
            "queued_interpreter_fold_in_oracle_text_source": async_smoke.get(
                "queued_interpreter_fold_in_oracle_text_source"
            ),
            "queued_interpreter_fold_in_evidence_authority": dict(
                async_smoke.get("queued_interpreter_fold_in_evidence_authority") or {}
            ),
            "verbose_result_spoken_bounded": bool(async_smoke.get("verbose_result_spoken_bounded")),
            "verbose_result_committed_bounded": bool(async_smoke.get("verbose_result_committed_bounded")),
            "verbose_result_commit_marked_truncated": bool(
                async_smoke.get("verbose_result_commit_marked_truncated")
            ),
            "verbose_full_result_durable": bool(async_smoke.get("verbose_full_result_durable")),
            "verbose_full_result_chars": async_smoke.get("verbose_full_result_chars"),
            "verbose_spoken_result": async_smoke.get("verbose_spoken_result"),
            "terminal_result_policy_smoke_ok": bool(async_smoke.get("terminal_result_policy_smoke_ok")),
            "terminal_result_auto_summarize_default": bool(
                async_smoke.get("terminal_result_auto_summarize_default")
            ),
            "terminal_result_default_event_count": async_smoke.get("terminal_result_default_event_count"),
            "terminal_result_default_spoken": bool(async_smoke.get("terminal_result_default_spoken")),
            "terminal_result_suppression_config": async_smoke.get("terminal_result_suppression_config"),
            "terminal_result_suppressed": bool(async_smoke.get("terminal_result_suppressed")),
            "terminal_result_unsolicited_event_count": async_smoke.get(
                "terminal_result_unsolicited_event_count"
            ),
            "terminal_result_unsolicited_spoken": bool(async_smoke.get("terminal_result_unsolicited_spoken")),
            "terminal_result_status_available": bool(async_smoke.get("terminal_result_status_available")),
            "terminal_result_status_text": async_smoke.get("terminal_result_status_text"),
            "unflagged_high_risk_tool_smoke_ok": bool(async_smoke.get("unflagged_high_risk_tool_smoke_ok")),
            "unflagged_high_risk_tool_suppressed": bool(async_smoke.get("unflagged_high_risk_tool_suppressed")),
            "unflagged_high_risk_tool_failed_closed": bool(
                async_smoke.get("unflagged_high_risk_tool_failed_closed")
            ),
            "unflagged_high_risk_tool_suppression_reason": async_smoke.get(
                "unflagged_high_risk_tool_suppression_reason"
            ),
            "unflagged_high_risk_tool_progress_suppressed": bool(
                async_smoke.get("unflagged_high_risk_tool_progress_suppressed")
            ),
            "unflagged_high_risk_tool_payload_redacted": bool(
                async_smoke.get("unflagged_high_risk_tool_payload_redacted")
            ),
            "unflagged_high_risk_tool_spoken_payload_clean": bool(
                async_smoke.get("unflagged_high_risk_tool_spoken_payload_clean")
            ),
            "unflagged_high_risk_tool_failure_spoken": bool(
                async_smoke.get("unflagged_high_risk_tool_failure_spoken")
            ),
            "unflagged_high_risk_tool_secret_canary_checked": bool(
                async_smoke.get("unflagged_high_risk_tool_secret_canary_checked")
            ),
            "unflagged_high_risk_tool_spoken": async_smoke.get("unflagged_high_risk_tool_spoken") or [],
            "external_frontend_bridge_smoke_ok": bool(async_smoke.get("external_frontend_bridge_smoke_ok")),
            "external_frontend_request_accepted": bool(async_smoke.get("external_frontend_request_accepted")),
            "external_frontend_tool_result_observed": bool(
                async_smoke.get("external_frontend_tool_result_observed")
            ),
            "external_frontend_protocol": async_smoke.get("external_frontend_protocol"),
            "external_frontend_protocol_contract": async_smoke.get(
                "external_frontend_protocol_contract"
            ),
            "external_frontend_job_id": async_smoke.get("external_frontend_job_id"),
            "external_frontend_provider": async_smoke.get("external_frontend_provider"),
            "external_frontend_tool": async_smoke.get("external_frontend_tool"),
            "external_frontend_tool_call_id": async_smoke.get("external_frontend_tool_call_id"),
            "external_frontend_completion_tool_call_id": async_smoke.get(
                "external_frontend_completion_tool_call_id"
            ),
            "external_frontend_status_tool_call_id": async_smoke.get("external_frontend_status_tool_call_id"),
            "external_frontend_terminal_correlation_observed": bool(
                async_smoke.get("external_frontend_terminal_correlation_observed")
            ),
            "external_frontend_audit_id": async_smoke.get("external_frontend_audit_id"),
            "external_frontend_source_audit_id": async_smoke.get("external_frontend_source_audit_id"),
            "external_frontend_parent_audit_id": async_smoke.get("external_frontend_parent_audit_id"),
            "external_frontend_status_audit_id": async_smoke.get("external_frontend_status_audit_id"),
            "external_frontend_completion_audit_id": async_smoke.get(
                "external_frontend_completion_audit_id"
            ),
            "external_frontend_audit_id_continuity_observed": bool(
                async_smoke.get("external_frontend_audit_id_continuity_observed")
            ),
            "external_frontend_accepted_observed": bool(async_smoke.get("external_frontend_accepted_observed")),
            "external_frontend_started_observed": bool(async_smoke.get("external_frontend_started_observed")),
            "external_frontend_completion_observed": bool(async_smoke.get("external_frontend_completion_observed")),
            "external_frontend_status_state": async_smoke.get("external_frontend_status_state"),
            "external_frontend_source_reached_oracle": bool(
                async_smoke.get("external_frontend_source_reached_oracle")
            ),
            "external_frontend_input_source": async_smoke.get("external_frontend_input_source"),
            "external_frontend_provisional_request_summary": async_smoke.get(
                "external_frontend_provisional_request_summary"
            )
            or {},
            "external_frontend_status_provisional_request_summary": async_smoke.get(
                "external_frontend_status_provisional_request_summary"
            )
            or {},
            "external_frontend_provisional_request_summary_non_authoritative": bool(
                async_smoke.get("external_frontend_provisional_request_summary_non_authoritative")
            ),
            "external_frontend_evidence_bundle_propagated": bool(
                async_smoke.get("external_frontend_evidence_bundle_propagated")
            ),
            "external_frontend_evidence_bundle_id": async_smoke.get(
                "external_frontend_evidence_bundle_id"
            ),
            "external_frontend_evidence_bundle_id_stable": bool(
                async_smoke.get("external_frontend_evidence_bundle_id_stable")
            ),
            "external_frontend_evidence_merge_key": async_smoke.get(
                "external_frontend_evidence_merge_key"
            ),
            "external_frontend_evidence_merge_key_propagated": bool(
                async_smoke.get("external_frontend_evidence_merge_key_propagated")
            ),
            "external_frontend_evidence_bundle_single_turn": bool(
                async_smoke.get("external_frontend_evidence_bundle_single_turn")
            ),
            "external_frontend_evidence_bundle_status": async_smoke.get(
                "external_frontend_evidence_bundle_status"
            ),
            "external_frontend_evidence_bundle_transcript_hypotheses_count": async_smoke.get(
                "external_frontend_evidence_bundle_transcript_hypotheses_count"
            ),
            "external_frontend_witness_kind": async_smoke.get("external_frontend_witness_kind"),
            "external_frontend_witness_kind_frontend_hypothesis": bool(
                async_smoke.get("external_frontend_witness_kind_frontend_hypothesis")
            ),
            "external_frontend_witness_metadata": dict(
                async_smoke.get("external_frontend_witness_metadata") or {}
            ),
            "external_frontend_witness_metadata_complete": bool(
                async_smoke.get("external_frontend_witness_metadata_complete")
            ),
            "external_frontend_witness_confidence": async_smoke.get(
                "external_frontend_witness_confidence"
            ),
            "external_frontend_witness_latency_ms": async_smoke.get(
                "external_frontend_witness_latency_ms"
            ),
            "external_frontend_witness_partial": async_smoke.get("external_frontend_witness_partial"),
            "external_frontend_witness_audio_time_range_ms": async_smoke.get(
                "external_frontend_witness_audio_time_range_ms"
            )
            or [],
            "external_frontend_witness_speaker": dict(
                async_smoke.get("external_frontend_witness_speaker") or {}
            ),
            "external_frontend_witness_channel": dict(
                async_smoke.get("external_frontend_witness_channel") or {}
            ),
            "external_frontend_witness_tool_authority_false": bool(
                async_smoke.get("external_frontend_witness_tool_authority_false")
            ),
            "external_frontend_direct_tool_authority_exposed": bool(
                async_smoke.get("external_frontend_direct_tool_authority_exposed")
            ),
            "unpromoted_hypothesis_evidence_bundle_id": async_smoke.get(
                "unpromoted_hypothesis_evidence_bundle_id"
            ),
            "unpromoted_hypothesis_single_bundle_observed": bool(
                async_smoke.get("unpromoted_hypothesis_single_bundle_observed")
            ),
            "unpromoted_hypothesis_status_bundle_status": async_smoke.get(
                "unpromoted_hypothesis_status_bundle_status"
            ),
            "unpromoted_hypothesis_status_bundle_transcript_hypotheses_count": async_smoke.get(
                "unpromoted_hypothesis_status_bundle_transcript_hypotheses_count"
            ),
            "unpromoted_hypothesis_tool_authority": async_smoke.get(
                "unpromoted_hypothesis_tool_authority"
            ),
            "unpromoted_hypothesis_tool_authority_false": bool(
                async_smoke.get("unpromoted_hypothesis_tool_authority_false")
            ),
            "unpromoted_hypothesis_action_sink_keys_checked": async_smoke.get(
                "unpromoted_hypothesis_action_sink_keys_checked"
            )
            or [],
            "unpromoted_hypothesis_action_sinks_clean": bool(
                async_smoke.get("unpromoted_hypothesis_action_sinks_clean")
            ),
            "unpromoted_hypothesis_not_spend_reason": bool(
                async_smoke.get("unpromoted_hypothesis_not_spend_reason")
            ),
            "unpromoted_hypothesis_not_spend_payload": bool(
                async_smoke.get("unpromoted_hypothesis_not_spend_payload")
            ),
            "unpromoted_hypothesis_not_provider_selection": bool(
                async_smoke.get("unpromoted_hypothesis_not_provider_selection")
            ),
            "unpromoted_hypothesis_not_nemoclaw_action_packet": bool(
                async_smoke.get("unpromoted_hypothesis_not_nemoclaw_action_packet")
            ),
            "unpromoted_hypothesis_not_phone_call_payload": bool(
                async_smoke.get("unpromoted_hypothesis_not_phone_call_payload")
            ),
            "unpromoted_hypothesis_not_call_payload": bool(
                async_smoke.get("unpromoted_hypothesis_not_call_payload")
            ),
            "unpromoted_hypothesis_not_tool_arguments": bool(
                async_smoke.get("unpromoted_hypothesis_not_tool_arguments")
            ),
            "unpromoted_hypothesis_not_memory_write": bool(
                async_smoke.get("unpromoted_hypothesis_not_memory_write")
            ),
            "unpromoted_hypothesis_not_file_write": bool(
                async_smoke.get("unpromoted_hypothesis_not_file_write")
            ),
            "unpromoted_hypothesis_not_message_payload": bool(
                async_smoke.get("unpromoted_hypothesis_not_message_payload")
            ),
            "witness_fusion_timing_smoke_ok": bool(async_smoke.get("witness_fusion_timing_smoke_ok")),
            "witness_fusion_arrival_phases": async_smoke.get("witness_fusion_arrival_phases") or [],
            "witness_fusion_case_job_ids": async_smoke.get("witness_fusion_case_job_ids") or {},
            "witness_fusion_turn_ids": async_smoke.get("witness_fusion_turn_ids") or {},
            "witness_fusion_audio_segment_refs": async_smoke.get("witness_fusion_audio_segment_refs") or {},
            "witness_fusion_evidence_merge_keys": async_smoke.get("witness_fusion_evidence_merge_keys") or {},
            "witness_fusion_merge_key_observed": bool(async_smoke.get("witness_fusion_merge_key_observed")),
            "witness_fusion_audio_metadata": async_smoke.get("witness_fusion_audio_metadata") or {},
            "witness_fusion_bundle_audio_metadata": async_smoke.get(
                "witness_fusion_bundle_audio_metadata"
            )
            or {},
            "witness_fusion_accepted_audio_gate_observed": bool(
                async_smoke.get("witness_fusion_accepted_audio_gate_observed")
            ),
            "witness_fusion_early_initial_bundle_id": async_smoke.get(
                "witness_fusion_early_initial_bundle_id"
            ),
            "witness_fusion_early_final_bundle_id": async_smoke.get("witness_fusion_early_final_bundle_id"),
            "witness_fusion_early_single_bundle": bool(async_smoke.get("witness_fusion_early_single_bundle")),
            "witness_fusion_with_bundle_id": async_smoke.get("witness_fusion_with_bundle_id"),
            "witness_fusion_with_single_bundle": bool(async_smoke.get("witness_fusion_with_single_bundle")),
            "witness_fusion_late_initial_bundle_id": async_smoke.get("witness_fusion_late_initial_bundle_id"),
            "witness_fusion_late_final_bundle_id": async_smoke.get("witness_fusion_late_final_bundle_id"),
            "witness_fusion_late_single_bundle": bool(async_smoke.get("witness_fusion_late_single_bundle")),
            "witness_fusion_no_duplicate_oracle_jobs": bool(
                async_smoke.get("witness_fusion_no_duplicate_oracle_jobs")
            ),
            "witness_fusion_partial_superseded_by_final": bool(
                async_smoke.get("witness_fusion_partial_superseded_by_final")
            ),
            "witness_fusion_partial_active_hypothesis": async_smoke.get(
                "witness_fusion_partial_active_hypothesis"
            )
            or {},
            "witness_fusion_adjudications": async_smoke.get("witness_fusion_adjudications") or {},
            "witness_fusion_rejection_reasons": async_smoke.get("witness_fusion_rejection_reasons") or {},
            "witness_fusion_adjudication_outcomes_observed": bool(
                async_smoke.get("witness_fusion_adjudication_outcomes_observed")
            ),
            "witness_fusion_accepted_counts": async_smoke.get("witness_fusion_accepted_counts") or {},
            "witness_fusion_started_counts": async_smoke.get("witness_fusion_started_counts") or {},
            "witness_fusion_completed_counts": async_smoke.get("witness_fusion_completed_counts") or {},
            "energy_gate_smoke_ok": bool(async_smoke.get("energy_gate_smoke_ok")),
            "energy_gate_policy": async_smoke.get("energy_gate_policy") or {},
            "energy_gate_ignored_packet_rms": async_smoke.get("energy_gate_ignored_packet_rms"),
            "energy_gate_ignored_packet_duration_ms": async_smoke.get(
                "energy_gate_ignored_packet_duration_ms"
            ),
            "energy_gate_ignored_packet_speech_confirmed": async_smoke.get(
                "energy_gate_ignored_packet_speech_confirmed"
            ),
            "energy_gate_ignored_packet_vad_speech": async_smoke.get(
                "energy_gate_ignored_packet_vad_speech"
            ),
            "energy_gate_ignored_non_speech_packets": async_smoke.get(
                "energy_gate_ignored_non_speech_packets"
            ),
            "energy_gate_low_energy_witness_text": async_smoke.get(
                "energy_gate_low_energy_witness_text"
            ),
            "energy_gate_low_energy_witness_source": async_smoke.get(
                "energy_gate_low_energy_witness_source"
            ),
            "energy_gate_low_energy_witness_promoted": async_smoke.get(
                "energy_gate_low_energy_witness_promoted"
            ),
            "energy_gate_low_energy_witness_suppressed": async_smoke.get(
                "energy_gate_low_energy_witness_suppressed"
            ),
            "energy_gate_barge_in_events": async_smoke.get("energy_gate_barge_in_events"),
            "energy_gate_interpreter_requests": async_smoke.get("energy_gate_interpreter_requests"),
            "energy_gate_oracle_work_events": async_smoke.get("energy_gate_oracle_work_events"),
            "energy_gate_oracle_requests": async_smoke.get("energy_gate_oracle_requests"),
            "energy_gate_raw_packet_buffered_without_turn": bool(
                async_smoke.get("energy_gate_raw_packet_buffered_without_turn")
            ),
            "energy_gate_event_types": async_smoke.get("energy_gate_event_types") or [],
            "runtime_kame_action_gate_smoke_ok": bool(async_smoke.get("runtime_kame_action_gate_smoke_ok")),
            "runtime_kame_action_gate_waiting_events": async_smoke.get("runtime_kame_action_gate_waiting_events"),
            "runtime_kame_action_gate_hypothesis_only_ok": async_smoke.get(
                "runtime_kame_action_gate_hypothesis_only_ok"
            ),
            "runtime_kame_action_gate_hypothesis_only_issues": async_smoke.get(
                "runtime_kame_action_gate_hypothesis_only_issues"
            )
            or [],
            "runtime_kame_action_gate_hypothesis_only_rejected_authorities": async_smoke.get(
                "runtime_kame_action_gate_hypothesis_only_rejected_authorities"
            )
            or [],
            "runtime_kame_action_gate_degraded_text_only_ok": async_smoke.get(
                "runtime_kame_action_gate_degraded_text_only_ok"
            ),
            "runtime_kame_action_gate_degraded_text_only_issues": async_smoke.get(
                "runtime_kame_action_gate_degraded_text_only_issues"
            )
            or [],
            "runtime_kame_action_gate_degraded_text_only_rejected_authorities": async_smoke.get(
                "runtime_kame_action_gate_degraded_text_only_rejected_authorities"
            )
            or [],
            "runtime_kame_action_gate_degraded_text_only_status": async_smoke.get(
                "runtime_kame_action_gate_degraded_text_only_status"
            ),
            "runtime_kame_action_gate_degraded_text_only_reason": async_smoke.get(
                "runtime_kame_action_gate_degraded_text_only_reason"
            ),
            "runtime_kame_action_gate_degraded_text_only_raw_audio_available": async_smoke.get(
                "runtime_kame_action_gate_degraded_text_only_raw_audio_available"
            ),
            "runtime_kame_action_gate_degraded_text_only_preserves_hypothesis": bool(
                async_smoke.get("runtime_kame_action_gate_degraded_text_only_preserves_hypothesis")
            ),
            "runtime_kame_action_gate_promoted_ok": async_smoke.get("runtime_kame_action_gate_promoted_ok"),
            "runtime_kame_action_gate_promoted_issues": async_smoke.get(
                "runtime_kame_action_gate_promoted_issues"
            )
            or [],
            "runtime_kame_action_gate_promoted_authorities": async_smoke.get(
                "runtime_kame_action_gate_promoted_authorities"
            )
            or [],
            "runtime_kame_action_gate_promoted_consumed_before_action": bool(
                async_smoke.get("runtime_kame_action_gate_promoted_consumed_before_action")
            ),
            "runtime_kame_action_gate_self_attested_ok": async_smoke.get(
                "runtime_kame_action_gate_self_attested_ok"
            ),
            "runtime_kame_action_gate_self_attested_issues": async_smoke.get(
                "runtime_kame_action_gate_self_attested_issues"
            )
            or [],
            "runtime_kame_action_gate_self_attested_authorities": async_smoke.get(
                "runtime_kame_action_gate_self_attested_authorities"
            )
            or [],
            "runtime_kame_action_gate_self_attested_consumed_before_action": bool(
                async_smoke.get("runtime_kame_action_gate_self_attested_consumed_before_action")
            ),
            "runtime_kame_action_gate_missing_tool_disclosure_ok": async_smoke.get(
                "runtime_kame_action_gate_missing_tool_disclosure_ok"
            ),
            "runtime_kame_action_gate_missing_tool_disclosure_issues": async_smoke.get(
                "runtime_kame_action_gate_missing_tool_disclosure_issues"
            )
            or [],
            "runtime_kame_action_gate_missing_tool_disclosure_authorities": async_smoke.get(
                "runtime_kame_action_gate_missing_tool_disclosure_authorities"
            )
            or [],
            "runtime_kame_action_gate_tool_disclosure_ref_observed": bool(
                async_smoke.get("runtime_kame_action_gate_tool_disclosure_ref_observed")
            ),
            "runtime_kame_action_gate_schema_versions": async_smoke.get(
                "runtime_kame_action_gate_schema_versions"
            )
            or [],
            "audit_scalar_smoke_ok": bool(async_smoke.get("audit_scalar_smoke_ok")),
            "audit_scalar_payload_redacted": bool(async_smoke.get("audit_scalar_payload_redacted")),
            "audit_scalar_secret_canary_checked": bool(async_smoke.get("audit_scalar_secret_canary_checked")),
            "audit_scalar_result_text_omitted": bool(async_smoke.get("audit_scalar_result_text_omitted")),
            "audit_scalar_completed_event_seen": bool(async_smoke.get("audit_scalar_completed_event_seen")),
            "audit_scalar_waiting_event_seen": bool(async_smoke.get("audit_scalar_waiting_event_seen")),
            "audit_scalar_row_count": async_smoke.get("audit_scalar_row_count"),
        },
        issues=issues,
    )
    cleanup_proof = proofs.get("discord_session_cleanup") if isinstance(proofs.get("discord_session_cleanup"), Mapping) else {}
    _compare_proof_fields(
        label="discord_session_cleanup",
        proof=cleanup_proof,
        expected={
            "scenario": cleanup_smoke.get("scenario"),
            "cancel_all_before_session_closed": bool(cleanup_smoke.get("cancel_all_before_session_closed")),
            "session_closed_sent": bool(cleanup_smoke.get("session_closed_sent")),
            "sidecar_closed": bool(cleanup_smoke.get("sidecar_closed")),
            "sidecar_close_calls": cleanup_smoke.get("sidecar_close_calls"),
            "degraded_job_state": cleanup_smoke.get("degraded_job_state"),
            "degraded_job_error": cleanup_smoke.get("degraded_job_error"),
            "event_order": cleanup_smoke.get("event_order") or [],
        },
        issues=issues,
    )
    sidecar_fail_closed_proof = (
        proofs.get("sidecar_fail_closed") if isinstance(proofs.get("sidecar_fail_closed"), Mapping) else {}
    )
    _compare_proof_fields(
        label="sidecar_fail_closed",
        proof=sidecar_fail_closed_proof,
        expected={
            "scenario": sidecar_fail_closed_smoke.get("scenario"),
            "fallback_policy": sidecar_fail_closed_smoke.get("fallback_policy"),
            "request_accepted": bool(sidecar_fail_closed_smoke.get("request_accepted")),
            "job_id": sidecar_fail_closed_smoke.get("job_id"),
            "cancelled_observed": bool(sidecar_fail_closed_smoke.get("cancelled_observed")),
            "cancel_reason": sidecar_fail_closed_smoke.get("cancel_reason"),
            "session_error_observed": bool(sidecar_fail_closed_smoke.get("session_error_observed")),
            "session_error_reason": sidecar_fail_closed_smoke.get("session_error_reason"),
            "session_error_sidecar": sidecar_fail_closed_smoke.get("session_error_sidecar"),
            "error_redacted": bool(sidecar_fail_closed_smoke.get("error_redacted")),
            "error_mentions_fail_closed": bool(sidecar_fail_closed_smoke.get("error_mentions_fail_closed")),
            "active_capacity_after_failure": sidecar_fail_closed_smoke.get("active_capacity_after_failure"),
            "job_state_after_failure": sidecar_fail_closed_smoke.get("job_state_after_failure"),
            "sidecar_removed": bool(sidecar_fail_closed_smoke.get("sidecar_removed")),
            "sidecar_closed": bool(sidecar_fail_closed_smoke.get("sidecar_closed")),
            "sidecar_close_calls": sidecar_fail_closed_smoke.get("sidecar_close_calls"),
            "oracle_requests_seen": sidecar_fail_closed_smoke.get("oracle_requests_seen"),
            "event_order": sidecar_fail_closed_smoke.get("event_order") or [],
            "test_refs": sidecar_fail_closed_smoke.get("test_refs") or [],
        },
        issues=issues,
    )


def _audit_live_evidence_scaffold(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    sections: Mapping[str, Mapping[str, Any]],
    issues: list[str],
) -> None:
    if manifest.get("example_only") is not True:
        issues.append("live_evidence_scaffold:manifest_must_remain_example_only")
    if str(manifest.get("overall_status") or "") == "live_evidence_supplied_not_readiness_claim":
        issues.append("live_evidence_scaffold:manifest_unexpected_live_claim")
    for section_name, section in sections.items():
        if section.get("example_only") is not True:
            issues.append(f"live_evidence_scaffold:{section_name}:must_remain_example_only")
        attestation = section.get("collector_attestation")
        if not isinstance(attestation, Mapping) or attestation.get("example_only") is not True:
            issues.append(f"live_evidence_scaffold:{section_name}:collector_attestation_must_remain_example_only")

    manifest_validation = _load_live_evidence([manifest_path])
    manifest_issues = manifest_validation.get("issues")
    if manifest_validation.get("overall_status") == "live_evidence_supplied_not_readiness_claim":
        issues.append("live_evidence_scaffold:manifest_unexpectedly_valid")
    if not any("example_only" in str(issue) for issue in manifest_issues or []):
        issues.append("live_evidence_scaffold:manifest_missing_example_only_validation_issue")
    for issue in manifest_issues or []:
        issue_text = str(issue)
        if issue_text.startswith("live_evidence_manifest:") and "example_only" not in issue_text:
            issues.append(f"live_evidence_scaffold:{issue_text}")

    evidence = {
        "schema_version": "voiceops.milestone1.live_voice_evidence.v1",
        "discord_live_probe": dict(sections.get("discord_live_probe") or {}),
        "sidecar_session": dict(sections.get("sidecar_session") or {}),
        "live_turn": dict(sections.get("live_turn") or {}),
    }
    validation = validate_live_probe_evidence(evidence)
    if validation.get("overall_status") == "live_evidence_supplied_not_readiness_claim":
        issues.append("live_evidence_scaffold:unexpectedly_valid")
    if not any("example_only" in str(issue) for issue in validation.get("issues") or []):
        issues.append("live_evidence_scaffold:missing_example_only_validation_issue")


def _require_markdown_tokens(label: str, markdown: str, tokens: Mapping[str, str], issues: list[str]) -> None:
    for issue, token in tokens.items():
        if token not in markdown:
            issues.append(f"{label}:{issue}")


def _require_markdown_any_token(label: str, markdown: str, issue: str, tokens: tuple[str, ...], issues: list[str]) -> None:
    if not any(token in markdown for token in tokens):
        issues.append(f"{label}:{issue}")


def _reject_markdown_tokens(label: str, markdown: str, tokens: Mapping[str, str], issues: list[str]) -> None:
    for issue, token in tokens.items():
        if token in markdown:
            issues.append(f"{label}:{issue}")


def _audit_markdown_consistency(
    *,
    spark_local_target_selected: bool,
    demo_markdown: str,
    demo_script_markdown: str,
    recording_runbook_markdown: str,
    submission_writeup_markdown: str,
    closure_markdown: str,
    operator_handoff_markdown: str,
    demo_handoff_markdown: str,
    dashboard_html: str,
    channel_policy_markdown: str,
    channel_review_markdown: str,
    issues: list[str],
) -> None:
    spark_boundary = (
        "Spark target selected, live evidence pending"
        if spark_local_target_selected
        else "Hosted fallback selected, Spark-local evidence pending"
    )
    rejected_spark_boundary = (
        "Hosted fallback selected, Spark-local evidence pending"
        if spark_local_target_selected
        else "Spark target selected, live evidence pending"
    )
    _require_markdown_tokens(
        "demo_markdown",
        demo_markdown,
        {
            "missing_static_dry_run_status": "static dry-run package",
            "missing_spark_evidence_boundary": spark_boundary,
            "missing_approval_gate": "spend/provisioning gated by approval",
        },
        issues,
    )
    _require_markdown_tokens(
        "recording_runbook_markdown",
        recording_runbook_markdown,
        {
            "missing_static_dry_run_status": "static dry-run VoiceOps package",
            "missing_spark_evidence_boundary": spark_boundary,
            "missing_secret_policy": "Do not show terminal panes or files that contain secrets",
        },
        issues,
    )
    _require_markdown_tokens(
        "submission_writeup_markdown",
        submission_writeup_markdown,
        {
            "missing_static_dry_run_status": "static dry-run package",
            "missing_spark_evidence_boundary": spark_boundary,
            "missing_spend_gate": "Spend gated by approval",
        },
        issues,
    )
    _require_markdown_tokens(
        "closure_markdown",
        closure_markdown,
        {
            "missing_needs_external_evidence": "needs_external_evidence",
            "missing_final_package_audit_command": "Final package audit command",
            "missing_package_audit_status_signal": "package_audit.status is pass",
            "missing_package_audit_flag": "--package-audit",
        },
        issues,
    )
    _require_markdown_any_token(
        "closure_markdown",
        closure_markdown,
        "missing_artifact_only_safety",
        (
            "artifact-only; no network I/O",
            "artifact-only; read-only discovery network possible only when explicitly requested",
        ),
        issues,
    )
    _require_markdown_tokens(
        "operator_handoff_markdown",
        operator_handoff_markdown,
        {
            "missing_final_package_audit_command": "Final package audit command",
            "missing_package_audit_status_signal": "package_audit.status is pass",
            "missing_package_audit_flag": "--package-audit",
            "missing_secret_policy": "never paste secret values into artifacts",
        },
        issues,
    )
    _require_markdown_tokens(
        "demo_handoff_markdown",
        demo_handoff_markdown,
        {
            "missing_package_audit_section": "Package audit:",
            "missing_package_audit_status_signal": "package_audit.status is pass",
            "missing_package_audit_flag": "--package-audit",
            "missing_no_secret_policy": "never paste secret values into artifacts",
        },
        issues,
    )
    _require_markdown_tokens(
        "channel_policy_markdown",
        channel_policy_markdown,
        {
            "missing_artifact_only_safety": "artifact-only; no network, secret reads, sends, SMS, or calls",
            "missing_validation_pass": "Validation: pass",
            "missing_approval_routing": "## Approval Routing",
            "missing_customer_visible_route": "customer_visible_outbound",
            "missing_phone_handoff_route": "approved_phone_handoff_call",
            "missing_audit_id_continuity": "Never overwrite an existing audit_id",
            "missing_phone_redaction": "phone_number: `<redacted-phone>`",
        },
        issues,
    )
    _require_markdown_tokens(
        "channel_policy_review_markdown",
        channel_review_markdown,
        {
            "missing_pending_review": "Review status: pending_human_review",
            "missing_no_real_egress": "Real egress enabled: False",
            "missing_operator_must_not_send": "send Discord, WhatsApp, SMS, or phone traffic from this generated packet",
            "missing_package_audit_flag": "--package-audit",
            "missing_phone_handoff_route": "approved_phone_handoff_call",
        },
        issues,
    )
    _reject_markdown_tokens(
        "spark_public_copy",
        "\n".join(
            [
                demo_markdown,
                demo_script_markdown,
                recording_runbook_markdown,
                submission_writeup_markdown,
                demo_handoff_markdown,
                dashboard_html,
            ]
        ),
        {
            "contradicts_active_model_path": rejected_spark_boundary,
            "claims_running_spark_appliance_without_evidence": "target appliance is one DGX Spark running",
            "claims_spark_powered_operator_without_evidence": "Spark-powered Hermes operator",
            "claims_turns_spark_into_operator_without_evidence": "turns a DGX Spark into",
        },
        issues,
    )
    _reject_markdown_tokens(
        "channel_policy_review_markdown",
        channel_review_markdown,
        {
            "contradicts_pending_review": "Review status: approved",
            "contradicts_no_real_egress": "Real egress enabled: True",
            "contradicts_no_live_egress": "Live egress enabled: True",
        },
        issues,
    )


def audit_package(artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    checked_artifacts = _audit_expected_package_artifacts(artifact_root, issues)
    _audit_no_unexpected_package_artifacts(artifact_root, checked_artifacts, issues)
    _audit_no_secret_like_values(artifact_root, checked_artifacts, issues)
    demo_dir = artifact_root / "hackathon-voiceops-demo" / "current"
    plan_dir = artifact_root / "voiceops-plan" / "current"
    provisioning_dir = artifact_root / "voiceops-provisioning" / "current"
    voice_dir = artifact_root / "voiceops-voice-operator" / "current"
    channel_dir = artifact_root / "voiceops-channel-policy" / "current"
    spark_dir = artifact_root / "voiceops-spark-matrix" / "current"

    demo = _read_json(demo_dir / "voiceops-demo.json", issues, "voiceops_demo")
    demo_markdown = _read_text(demo_dir / "voiceops-demo.md", issues, "voiceops_demo_markdown")
    demo_script_markdown = _read_text(demo_dir / "demo-script.md", issues, "demo_script_markdown")
    readiness = _read_json(demo_dir / "readiness-report.json", issues, "readiness_report")
    demo_closure = _read_json(demo_dir / "readiness-closure-summary.json", issues, "demo_closure")
    demo_handoff = _read_json(demo_dir / "operator-handoff-preview.json", issues, "demo_handoff")
    demo_handoff_markdown = _read_text(demo_dir / "operator-handoff-preview.md", issues, "demo_handoff_markdown")
    recording_runbook_markdown = _read_text(demo_dir / "recording-runbook.md", issues, "recording_runbook_markdown")
    submission_writeup_markdown = _read_text(demo_dir / "submission-writeup.md", issues, "submission_writeup_markdown")
    operator_state = _read_json(demo_dir / "operator-state.json", issues, "operator_state")
    packet = _read_json(demo_dir / "nemoclaw-action-packet.json", issues, "nemoclaw_packet")
    packet_validation = _read_json(
        demo_dir / "nemoclaw-action-packet.validation.json",
        issues,
        "nemoclaw_packet_validation",
    )
    plan_run = _read_json(plan_dir / "voiceops-plan-run.json", issues, "plan_run")
    plan_closure = _read_json(plan_dir / "readiness-closure-index.json", issues, "plan_closure")
    plan_closure_markdown = _read_text(plan_dir / "readiness-closure-index.md", issues, "plan_closure_markdown")
    plan_handoff = _read_json(plan_dir / "operator-handoff.json", issues, "operator_handoff")
    plan_handoff_markdown = _read_text(plan_dir / "operator-handoff.md", issues, "operator_handoff_markdown")
    execution_plan = _read_json(provisioning_dir / "milestone2-execution-plan.json", issues, "milestone2_execution_plan")
    post_approval_scaffold = _read_json(
        provisioning_dir / "post-approval-receipts-scaffold" / "post-approval-receipts.json",
        issues,
        "post_approval_receipts_scaffold",
    )
    post_approval_validation = _read_json(
        provisioning_dir / "post-approval-receipts.validation.json",
        issues,
        "post_approval_receipts_validation",
    )
    voice_operator_readiness = _read_json(
        voice_dir / "voice-operator-readiness.json",
        issues,
        "voice_operator_readiness",
    )
    discord_loopback_smoke = _read_json(
        voice_dir / "discord-loopback-smoke.json",
        issues,
        "discord_loopback_smoke",
    )
    async_oracle_smoke = _read_json(
        voice_dir / "async-oracle-smoke.json",
        issues,
        "async_oracle_smoke",
    )
    discord_session_cleanup_smoke = _read_json(
        voice_dir / "discord-session-cleanup-smoke.json",
        issues,
        "discord_session_cleanup_smoke",
    )
    sidecar_fail_closed_smoke = _read_json(
        voice_dir / "sidecar-fail-closed-smoke.json",
        issues,
        "sidecar_fail_closed_smoke",
    )
    live_scaffold_dir = voice_dir / "live-voice-evidence-scaffold"
    live_scaffold_manifest = _read_json(live_scaffold_dir / "manifest.json", issues, "live_evidence_scaffold_manifest")
    live_scaffold_sections = {
        "discord_live_probe": _read_json(
            live_scaffold_dir / "sections" / "discord-live-probe.json",
            issues,
            "live_evidence_scaffold_discord_live_probe",
        ),
        "sidecar_session": _read_json(
            live_scaffold_dir / "sections" / "sidecar-session.json",
            issues,
            "live_evidence_scaffold_sidecar_session",
        ),
        "live_turn": _read_json(
            live_scaffold_dir / "sections" / "live-turn.json",
            issues,
            "live_evidence_scaffold_live_turn",
        ),
    }
    spark_matrix = _read_json(spark_dir / "spark-model-matrix.json", issues, "spark_matrix")
    channel_policy = _read_json(channel_dir / "channel-policy.json", issues, "channel_policy")
    channel_review = _read_json(channel_dir / "channel-policy-review.json", issues, "channel_policy_review")
    channel_policy_markdown = _read_text(channel_dir / "channel-policy.md", issues, "channel_policy_markdown")
    channel_review_markdown = _read_text(
        channel_dir / "channel-policy-review.md",
        issues,
        "channel_policy_review_markdown",
    )
    dashboard_html = _read_text(demo_dir / "operator-dashboard.html", issues, "operator_dashboard")
    audit_rows = _read_jsonl(demo_dir / "audit-ledger.jsonl", issues, "audit_ledger")
    operator_state_event_rows = _read_jsonl(demo_dir / "operator-state-events.jsonl", issues, "operator_state_events")
    dry_run_rows = _dry_run_metadata_rows(
        _read_text(demo_dir / "stripe-actions-dry-run.sh", issues, "stripe_actions"),
        issues,
    )
    sponsor_stack = demo.get("sponsor_stack") if isinstance(demo.get("sponsor_stack"), Mapping) else {}
    active_path = (
        sponsor_stack.get("hermes_active_model")
        if isinstance(sponsor_stack.get("hermes_active_model"), Mapping)
        else {}
    )
    spark_local_target_selected = active_path.get("spark_local") is True

    _audit_static_readiness(
        demo=demo,
        readiness=readiness,
        spark_matrix=spark_matrix,
        demo_closure=demo_closure,
        plan_closure=plan_closure,
        operator_state=operator_state,
        dashboard_html=dashboard_html,
        issues=issues,
    )
    _audit_dashboard_consistency(
        demo=demo,
        readiness=readiness,
        operator_state=operator_state,
        dashboard_html=dashboard_html,
        issues=issues,
    )
    _audit_provider_role_matrix(
        demo=demo,
        dashboard_html=dashboard_html,
        issues=issues,
    )
    _audit_action_consistency(
        demo=demo,
        packet=packet,
        packet_validation=packet_validation,
        operator_state=operator_state,
        audit_rows=audit_rows,
        operator_state_event_rows=operator_state_event_rows,
        dry_run_rows=dry_run_rows,
        issues=issues,
    )
    _audit_service_claims(operator_state, issues)
    _audit_plan_consistency(
        demo=demo,
        demo_closure=demo_closure,
        demo_handoff=demo_handoff,
        plan_run=plan_run,
        plan_closure=plan_closure,
        plan_handoff=plan_handoff,
        issues=issues,
    )
    _audit_plan_run_package_audit_summary(
        artifact_root=artifact_root,
        plan_run=plan_run,
        checked_artifacts=checked_artifacts,
        issues=issues,
    )
    _audit_channel_policy(channel_policy, channel_review, issues)
    _audit_execution_plan_contracts(execution_plan, issues)
    _audit_execution_plan_approval_surfaces(
        execution_plan=execution_plan,
        packet=packet,
        channel_policy=channel_policy,
        issues=issues,
    )
    _audit_post_approval_receipt_validation(
        artifact_root=artifact_root,
        execution_plan=execution_plan,
        scaffold=post_approval_scaffold,
        stored_validation=post_approval_validation,
        issues=issues,
    )
    _audit_preflight_evidence_scaffold(
        manifest_path=(
            artifact_root
            / "voiceops-provisioning"
            / "current"
            / "provisioning-preflight-scaffold"
            / "provisioning-preflight-evidence.manifest.json"
        ),
        issues=issues,
    )
    _audit_voice_operator_artifact_consistency(
        readiness=voice_operator_readiness,
        discord_loopback_smoke=discord_loopback_smoke,
        async_oracle_smoke=async_oracle_smoke,
        discord_session_cleanup_smoke=discord_session_cleanup_smoke,
        sidecar_fail_closed_smoke=sidecar_fail_closed_smoke,
        issues=issues,
    )
    _audit_live_evidence_scaffold(
        manifest_path=live_scaffold_dir / "manifest.json",
        manifest=live_scaffold_manifest,
        sections=live_scaffold_sections,
        issues=issues,
    )
    _audit_spark_evidence_scaffold(
        spark_dir / "spark-benchmark-scaffold" / "spark-benchmark-evidence.json",
        issues,
    )
    _audit_markdown_consistency(
        spark_local_target_selected=spark_local_target_selected,
        demo_markdown=demo_markdown,
        demo_script_markdown=demo_script_markdown,
        recording_runbook_markdown=recording_runbook_markdown,
        submission_writeup_markdown=submission_writeup_markdown,
        closure_markdown=plan_closure_markdown,
        operator_handoff_markdown=plan_handoff_markdown,
        demo_handoff_markdown=demo_handoff_markdown,
        dashboard_html=dashboard_html,
        channel_policy_markdown=channel_policy_markdown,
        channel_review_markdown=channel_review_markdown,
        issues=issues,
    )

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "artifact_id": "voiceops-artifact-package-audit",
        "artifact_root": str(artifact_root),
        "mode": "local_static_package_audit_only",
        "readiness_claim": False,
        "readiness_scope": "static_package_consistency_only",
        "readiness_note": (
            "Package audit pass means generated artifacts are internally consistent and secret-safe; "
            "it does not satisfy live Discord, spend/provisioning, or DGX Spark evidence gates."
        ),
        "safety": {
            "env_files_read": False,
            "secret_values_emitted": False,
            "network_io": False,
            "discord_io": False,
            "provider_provisioning": False,
            "live_spend": False,
            "outbound_messages": False,
            "outbound_calls": False,
            "spark_execution": False,
        },
        "ok": not issues,
        "status": "pass" if not issues else "fail",
        "issues": sorted(set(issues)),
        "warnings": warnings,
        "checked_artifacts": checked_artifacts,
        "checked_artifact_count": len(checked_artifacts),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# VoiceOps Artifact Package Audit",
        "",
        f"- Status: {report['status']}",
        f"- Readiness claim: {'yes' if report.get('readiness_claim') else 'no'}",
        f"- Readiness scope: `{report.get('readiness_scope', 'static_package_consistency_only')}`",
        f"- Note: {report.get('readiness_note', '')}",
        f"- Artifact root: `{report['artifact_root']}`",
        f"- Checked artifacts: {report['checked_artifact_count']}",
        "- Network I/O: no",
        "- Provider provisioning: no",
        "- Live spend: no",
        "- Outbound messages/calls: no",
        "",
        "## Issues",
        "",
    ]
    issues = report.get("issues") or []
    if issues:
        lines.extend(f"- `{issue}`" for issue in issues)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_audit(output_dir: Path, report: Mapping[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "package-audit.json",
        "markdown": output_dir / "package-audit.md",
    }
    paths["json"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Print the audit report without writing package-audit artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_package(args.artifact_root)
    if args.audit_only:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        paths = write_audit(args.output_dir, report)
        print(json.dumps({"ok": report["ok"], "status": report["status"], "artifacts": paths}, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
