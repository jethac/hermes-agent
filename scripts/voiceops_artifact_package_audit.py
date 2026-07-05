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

from scripts.voiceops_channel_policy import (
    CHANNEL_IDS,
    REVIEW_DECISION_ARTIFACT_ID,
    REVIEW_DECISION_SCHEMA_VERSION,
    REQUIRED_KAME_DESIGN_REFERENCE,
    REQUIRED_KAME_INPUT_ORDER,
    REQUIRED_KAME_INTERPRETER_PROFILE,
    REQUIRED_KAME_LINEAGE_FIELDS,
    REQUIRED_KAME_PROMOTED_AUTHORITIES,
    REQUIRED_TRANSCRIPT_HYPOTHESIS_CONTRACT,
    REQUIRED_TRANSCRIPT_HYPOTHESIS_FIELDS,
    REQUIRED_UNPROMOTED_WITNESS_SINK_CHECKS,
    stable_review_sha256,
    validate_channel_policy_review_decision,
    validate_policy,
)
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
    "auxiliary_transcript_evidence": "hypothesis",
    "outbound_tts": "playback_only",
    "degraded_fallback": "fallback_text_or_diagnostic_only",
}

VOICE_SCOPED_ORACLE_SELECTOR_BOUNDARY_TOKENS = (
    "oracle_model",
    "preferred_local_oracle_model",
    "oracle_provider",
    "oracle_provider_name",
    "oracle_base_url",
    "oracle_api_mode",
)
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
VALID_WITNESS_ADJUDICATION_OUTCOMES = {
    "accepted_as_supporting_evidence",
    "corrected_by_audio",
    "rejected_or_diagnostic_only",
}
VALID_WITNESS_REJECTION_REASONS = {
    "ambiguous_speaker",
    "wrong_speaker",
    "wrong_channel",
    "stale_witness",
    "timing_conflict",
    "low_energy_non_speech",
    "waveform_conflict",
    "provider_conflict",
}
VALID_WITNESS_ARRIVAL_PHASES = {
    "before_raw_audio",
    "with_raw_audio",
    "after_interpreter_start",
}
EXPECTED_PROVIDER_TEXT_ALIAS_KEYS = ("stt", "caption", "transcript", "query", "user_text")
EXPECTED_KAME_LATENCY_BREAKDOWN_SEGMENTS = (
    "speech_end_to_reflex_ack_ms",
    "audio_cut_to_interpreter_submit_ms",
    "witness_arrival_ms",
    "interpreter_submit_to_promotion_ms",
    "promotion_to_oracle_start_ms",
    "oracle_start_to_first_token_ms",
    "first_token_to_tts_first_audio_ms",
    "tts_first_audio_to_playback_start_ms",
    "playback_start_to_completion_ms",
)
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
    "voiceops-channel-policy/current/channel-policy-review-decision.json",
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
    "voiceops-voice-operator/current/tool-disclosure-smoke.json",
    "voiceops-voice-operator/current/ephemeral-tool-router-smoke.json",
    "voiceops-voice-operator/current/interpreter-request-packet.json",
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
    "voiceops-channel-policy/current/operator-channel-policy-review-decision.json",
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
    oracle_boundary_text = " ".join(str(value) for value in oracle_must_not or [])
    missing_oracle_boundary_tokens = [
        token for token in VOICE_SCOPED_ORACLE_SELECTOR_BOUNDARY_TOKENS if token not in oracle_boundary_text
    ]
    if missing_oracle_boundary_tokens:
        issues.append("provider_role_matrix:oracle:missing_voice_scoped_oracle_selector_boundary")
        for token in missing_oracle_boundary_tokens:
            issues.append(f"provider_role_matrix:oracle:missing_oracle_selector_boundary:{token}")
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


def _audit_phone_context_contract(
    *,
    demo: Mapping[str, Any],
    phone_context: Mapping[str, Any],
    packet: Mapping[str, Any],
    issues: list[str],
) -> None:
    if phone_context.get("schema_version") != "voiceops.phone_context.v1":
        issues.append("phone_context:missing_or_invalid_schema_version")
    if phone_context.get("artifact_id") != "voiceops-phone-context":
        issues.append("phone_context:missing_or_invalid_artifact_id")
    if phone_context.get("target_channel") != "phone":
        issues.append("phone_context:target_channel_not_phone")
    if phone_context.get("status") != "queued_requires_approval":
        issues.append("phone_context:status_not_queued_requires_approval")
    if phone_context.get("context_authority") != "oracle_promoted":
        issues.append("phone_context:context_authority_not_oracle_promoted")
    if phone_context.get("context_authority_ref") != "oracle_job.phone_handoff_context":
        issues.append("phone_context:context_authority_ref_mismatch")
    if phone_context.get("transcript_hypotheses_allowed") is not False:
        issues.append("phone_context:transcript_hypotheses_allowed_not_false")
    if phone_context.get("transcript_hypotheses") != []:
        issues.append("phone_context:transcript_hypotheses_not_empty")
    if phone_context.get("raw_witness_text_allowed") is not False:
        issues.append("phone_context:raw_witness_text_allowed_not_false")
    if phone_context.get("phone_payload_policy") != "promoted_context_reference_only":
        issues.append("phone_context:phone_payload_policy_mismatch")
    if phone_context.get("channel_policy_ref") != "channel_policy.routes.approved_phone_handoff_call":
        issues.append("phone_context:channel_policy_ref_mismatch")
    if phone_context.get("credential_location_ref") != "credential_locations.phone_bridge":
        issues.append("phone_context:credential_location_ref_mismatch")
    if phone_context.get("tool_disclosure_ref") != "tool_disclosure":
        issues.append("phone_context:tool_disclosure_ref_mismatch")

    source_context = phone_context.get("source_context") if isinstance(phone_context.get("source_context"), Mapping) else {}
    demo_source_context = demo.get("source_context") if isinstance(demo.get("source_context"), Mapping) else {}
    required_context_fields = (
        "source_voice_session_id",
        "source_oracle_job_id",
        "turn_id",
        "audio_segment_ref",
        "evidence_bundle_id",
        "evidence_merge_key",
    )
    for field in required_context_fields:
        if not str(source_context.get(field) or "").strip():
            issues.append(f"phone_context:source_context:{field}_missing")
        if phone_context.get(field) != source_context.get(field):
            issues.append(f"phone_context:{field}_top_level_mismatch")
        if field in demo_source_context and demo_source_context.get(field) != source_context.get(field):
            issues.append(f"phone_context:source_context:{field}_demo_mismatch")

    actions = {
        str(action.get("action_id")): action
        for action in packet.get("approval_required_actions", [])
        if isinstance(action, Mapping)
    }
    pending = {
        str(action.get("action_id")): action
        for action in phone_context.get("pending_approvals", [])
        if isinstance(action, Mapping)
    }
    if set(pending) != set(actions):
        issues.append("phone_context:pending_approvals_do_not_match_nemoclaw_actions")
    call_approval = pending.get("call-user-phone")
    if not isinstance(call_approval, Mapping):
        issues.append("phone_context:call_user_phone_pending_approval_missing")
        return
    call_evidence = (
        call_approval.get("kame_evidence")
        if isinstance(call_approval.get("kame_evidence"), Mapping)
        else {}
    )
    promoted_fields = (
        call_evidence.get("promoted_fields")
        if isinstance(call_evidence.get("promoted_fields"), Mapping)
        else {}
    )
    phone_field = (
        promoted_fields.get("phone_handoff_context")
        if isinstance(promoted_fields.get("phone_handoff_context"), Mapping)
        else {}
    )
    if phone_field.get("evidence_label") != "oracle_promoted":
        issues.append("phone_context:call_user_phone_handoff_context_not_oracle_promoted")
    if call_evidence.get("hypotheses_allowed_for_action") is not False:
        issues.append("phone_context:call_user_phone_hypotheses_allowed_for_action_not_false")
    if call_evidence.get("transcript_hypotheses_promoted") is not False:
        issues.append("phone_context:call_user_phone_transcript_hypotheses_promoted_not_false")
    if call_approval.get("tool_disclosure_ref") != "tool_disclosure":
        issues.append("phone_context:call_user_phone_tool_disclosure_ref_mismatch")


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


def _audit_spark_matrix_kame_contract(spark_matrix: Mapping[str, Any], issues: list[str]) -> None:
    policy = spark_matrix.get("policy") if isinstance(spark_matrix.get("policy"), Mapping) else {}
    required_fields = {
        "kind",
        "source",
        "text_digest",
        "role",
        "authority",
        "promotion_required",
        "tool_authority",
        "arrival_phase",
        "latency_ms",
        "confidence",
        "speaker_or_actor_ref",
        "channel_or_surface_ref",
    }
    if policy.get("raw_audio_primary_interpreter_evidence") is not True:
        issues.append("spark_matrix:kame_contract:raw_audio_primary_not_true")
    if policy.get("transcript_hypotheses_are_witness_context") is not True:
        issues.append("spark_matrix:kame_contract:witness_context_not_true")
    if policy.get("transcript_only_counts_for_one_spark_readiness") is not False:
        issues.append("spark_matrix:kame_contract:transcript_only_counts_for_readiness")
    observed_fields = {
        str(item)
        for item in policy.get("transcript_hypothesis_required_fields", [])
        if str(item).strip()
    }
    missing_fields = sorted(required_fields.difference(observed_fields))
    if missing_fields:
        issues.append("spark_matrix:kame_contract:missing_transcript_hypothesis_fields:" + ",".join(missing_fields))
    contract = policy.get("transcript_hypothesis_contract") if isinstance(policy.get("transcript_hypothesis_contract"), Mapping) else {}
    expected_contract = {
        "role": "witness_context",
        "authority": "hypothesis",
        "promotion_required": "interpreter_promoted_or_oracle_promoted",
        "tool_authority": False,
    }
    for key, expected in expected_contract.items():
        if contract.get(key) != expected:
            issues.append(f"spark_matrix:kame_contract:{key}_mismatch")


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
    if plan_run.get("review_gaps") != plan_closure.get("review_gaps"):
        issues.append("plan_run:review_gaps_mismatch")
    if plan_run.get("evidence_mode") != plan_closure.get("evidence_mode"):
        issues.append("plan_run:evidence_mode_mismatch")
    expected_readiness_ok = (
        plan_closure.get("closure_status") == "complete"
        and plan_closure.get("readiness_gaps") == []
        and plan_closure.get("review_gaps") == []
        and plan_closure.get("evidence_mode") != "fixture_rehearsal"
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
    _audit_review_action_handoff_consistency(
        "operator_handoff",
        plan_closure.get("review_actions"),
        plan_handoff,
        issues,
    )
    _audit_review_action_handoff_consistency(
        "demo_handoff",
        plan_closure.get("review_actions"),
        demo_handoff,
        issues,
    )
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
        _audit_artifact_ref_list(label, f"{expected_phase_id}:expected_artifacts", phase.get("expected_artifacts"), issues)
        _audit_artifact_ref_list(label, f"{expected_phase_id}:optional_artifacts", phase.get("optional_artifacts"), issues)
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
        for field in ("status", "can_run_here_now"):
            if action.get(field) != phase.get(field):
                issues.append(f"{label}:{gate_id}:{field}_mismatch_with_phase")
        action_blockers = action.get("blocked_by_current_environment")
        phase_blockers = phase.get("blocked_by_current_environment")
        if not isinstance(action_blockers, Mapping):
            issues.append(f"{label}:{gate_id}:blocked_by_current_environment_not_object")
        elif not isinstance(phase_blockers, Mapping):
            issues.append(f"{label}:{gate_id}:phase_blocked_by_current_environment_not_object")
        else:
            for blocker_key, blocker_value in action_blockers.items():
                if phase_blockers.get(blocker_key) != blocker_value:
                    issues.append(
                        f"{label}:{gate_id}:blocked_by_current_environment_mismatch:{blocker_key}"
                    )
        for field in (
            "expected_artifacts",
            "optional_artifacts",
            "success_check",
            "first_safe_command",
            "first_evidence_command",
        ):
            action_value = (action.get(field) or []) if field == "optional_artifacts" else action.get(field)
            phase_value = (phase.get(field) or []) if field == "optional_artifacts" else phase.get(field)
            if action_value != phase_value:
                issues.append(f"{label}:{gate_id}:{field}_mismatch_with_phase")
        if action.get("primary_next_command") != action.get("first_safe_command"):
            issues.append(f"{label}:{gate_id}:primary_next_command_mismatch")
        if action.get("primary_evidence_command") != action.get("first_evidence_command"):
            issues.append(f"{label}:{gate_id}:primary_evidence_command_mismatch")
        secret_policy = str(action.get("secret_policy") or "")
        if "never include secret values" not in secret_policy:
            issues.append(f"{label}:{gate_id}:secret_policy_missing_no_secret_rule")
        operator_step = str(action.get("operator_step") or "").strip()
        if not operator_step:
            issues.append(f"{label}:{gate_id}:operator_step_missing")
        validation_commands = action.get("validation_commands")
        if not isinstance(validation_commands, Mapping):
            issues.append(f"{label}:{gate_id}:validation_commands_not_object")
            continue
        for command_key in validation_commands:
            if command_key not in command_safety:
                issues.append(f"{label}:{gate_id}:validation_command_missing_safety:{command_key}")


def _audit_review_action_handoff_consistency(
    label: str,
    review_actions: Any,
    handoff: Mapping[str, Any],
    issues: list[str],
) -> None:
    if not isinstance(review_actions, list):
        issues.append(f"{label}:review_actions_missing_for_review_phase_consistency")
        return
    phases_by_id = _handoff_review_phases_by_id(handoff)
    actions_by_id = {
        str(action.get("phase_id")): action
        for action in review_actions
        if isinstance(action, Mapping) and str(action.get("phase_id") or "").strip()
    }
    for phase_id, action in actions_by_id.items():
        phase = phases_by_id.get(phase_id)
        if not isinstance(phase, Mapping):
            issues.append(f"{label}:{phase_id}:missing_review_phase_for_action")
            continue
        for field in (
            "order",
            "milestone",
            "status",
            "decision_artifact",
            "decision_status",
            "can_run_here_now",
            "blocked_by_current_environment",
            "first_safe_command",
            "review_command",
            "review_artifacts",
            "required_review",
            "success_check",
            "changes_readiness_by_itself",
            "changes_policy_by_itself",
            "real_egress_enabled",
        ):
            if action.get(field) != phase.get(field):
                issues.append(f"{label}:{phase_id}:{field}_mismatch_with_review_phase")
        secret_policy = str(action.get("secret_policy") or "")
        if "never paste channel credentials or message recipient values" not in secret_policy:
            issues.append(f"{label}:{phase_id}:secret_policy_missing_review_secret_rule")


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
    for gate_id, action in actions_by_gate.items():
        _audit_artifact_ref_list(label, f"{gate_id}:expected_artifacts", action.get("expected_artifacts"), issues)
        _audit_artifact_ref_list(label, f"{gate_id}:optional_artifacts", action.get("optional_artifacts"), issues)
    live = actions_by_gate.get("live_discord_voice_operator")
    if isinstance(live, Mapping):
        live_safe = str(live.get("first_safe_command") or "")
        live_evidence = str(live.get("first_evidence_command") or "")
        live_primary = str(live.get("primary_next_command") or "")
        live_primary_evidence = str(live.get("primary_evidence_command") or "")
        if live_primary != live_safe:
            issues.append(f"{label}:live_discord_voice_operator:primary_next_command_mismatch")
        if live_primary_evidence != live_evidence:
            issues.append(f"{label}:live_discord_voice_operator:primary_evidence_command_mismatch")
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
        spark_primary = str(spark.get("primary_next_command") or "")
        spark_primary_evidence = str(spark.get("primary_evidence_command") or "")
        if spark_primary != spark_safe:
            issues.append(f"{label}:local_spark_stack_matrix:primary_next_command_mismatch")
        if spark_primary_evidence != spark_evidence:
            issues.append(f"{label}:local_spark_stack_matrix:primary_evidence_command_mismatch")
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
    status = action.get("status")
    if status not in {"pending_human_review", "operator_review_accepted"}:
        issues.append(f"{label}:channel_policy_review_status_invalid")
    if status == "operator_review_accepted" and action.get("decision_status") != "accepted":
        issues.append(f"{label}:channel_policy_review_decision_not_accepted")
    if status == "operator_review_accepted" and not str(action.get("decision_artifact") or ""):
        issues.append(f"{label}:channel_policy_review_decision_artifact_missing")
    if action.get("changes_readiness_by_itself") is not False:
        issues.append(f"{label}:channel_policy_review_changes_readiness")
    if action.get("changes_policy_by_itself") is not False:
        issues.append(f"{label}:channel_policy_review_changes_policy")
    if action.get("real_egress_enabled") is not False:
        issues.append(f"{label}:channel_policy_review_enables_egress")
    command = str(action.get("review_command") or "")
    if "voiceops_channel_policy.py" not in command:
        issues.append(f"{label}:channel_policy_review_command_invalid")


def _audit_artifact_ref_list(label: str, field: str, value: Any, issues: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, list):
        issues.append(f"{label}:{field}_not_list")
        return
    normalized = [str(item) for item in value]
    duplicates = sorted({item for item in normalized if normalized.count(item) > 1})
    for duplicate in duplicates:
        issues.append(f"{label}:{field}_duplicate:{duplicate}")


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


def _audit_channel_policy(
    policy: Mapping[str, Any],
    review: Mapping[str, Any],
    decision_scaffold: Mapping[str, Any],
    *,
    review_path: Path,
    issues: list[str],
) -> None:
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
    policy_routes = {
        str(route.get("route_id")): route
        for route in policy.get("approval_routing", [])
        if isinstance(route, Mapping)
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
    policy_kame_gate = policy.get("kame_action_evidence_gate") if isinstance(policy.get("kame_action_evidence_gate"), Mapping) else {}
    review_kame_gate = review.get("kame_action_evidence_gate") if isinstance(review.get("kame_action_evidence_gate"), Mapping) else {}
    if dict(review_kame_gate) != dict(policy_kame_gate):
        issues.append("channel_policy_review:kame_action_evidence_gate_mismatch")
    if review_kame_gate:
        if set(review_kame_gate.get("accepted_promoted_authorities") or []) != REQUIRED_KAME_PROMOTED_AUTHORITIES:
            issues.append("channel_policy_review:kame_gate_promoted_authorities_mismatch")
        if review_kame_gate.get("design_reference") != REQUIRED_KAME_DESIGN_REFERENCE:
            issues.append("channel_policy_review:kame_gate_design_reference_mismatch")
        if review_kame_gate.get("required_interpreter_profile") != REQUIRED_KAME_INTERPRETER_PROFILE:
            issues.append("channel_policy_review:kame_gate_interpreter_profile_mismatch")
        if review_kame_gate.get("required_interpreter_input_order") != REQUIRED_KAME_INPUT_ORDER:
            issues.append("channel_policy_review:kame_gate_input_order_mismatch")
        missing_hypothesis_fields = REQUIRED_TRANSCRIPT_HYPOTHESIS_FIELDS - set(
            review_kame_gate.get("required_transcript_hypothesis_fields") or []
        )
        if missing_hypothesis_fields:
            issues.append(
                "channel_policy_review:kame_gate_missing_transcript_hypothesis_fields:"
                + ",".join(sorted(missing_hypothesis_fields))
            )
        hypothesis_contract = (
            review_kame_gate.get("transcript_hypothesis_contract")
            if isinstance(review_kame_gate.get("transcript_hypothesis_contract"), Mapping)
            else {}
        )
        for field, expected_value in sorted(REQUIRED_TRANSCRIPT_HYPOTHESIS_CONTRACT.items()):
            if hypothesis_contract.get(field) != expected_value:
                issues.append(f"channel_policy_review:kame_gate_transcript_hypothesis_contract_mismatch:{field}")
        if review_kame_gate.get("raw_transcript_text_allowed_in_channel_egress") is not False:
            issues.append("channel_policy_review:kame_gate_raw_transcript_text_allowed_in_channel_egress")
        missing_lineage = REQUIRED_KAME_LINEAGE_FIELDS - set(review_kame_gate.get("required_lineage_fields") or [])
        if missing_lineage:
            issues.append(f"channel_policy_review:kame_gate_missing_lineage:{','.join(sorted(missing_lineage))}")
        sink_checks = review_kame_gate.get("requires_unpromoted_witness_sink_checks")
        if not isinstance(sink_checks, Mapping):
            sink_checks = {}
        missing_sink_checks = {
            sink for sink in REQUIRED_UNPROMOTED_WITNESS_SINK_CHECKS if sink_checks.get(sink) is not True
        }
        if missing_sink_checks:
            issues.append(
                "channel_policy_review:kame_gate_missing_unpromoted_sink_checks:"
                + ",".join(sorted(missing_sink_checks))
            )
        if review_kame_gate.get("degraded_text_only_allowed_for_action") is not False:
            issues.append("channel_policy_review:kame_gate_degraded_text_allows_action")
        if review_kame_gate.get("unpromoted_witness_may_enter_payloads") is not False:
            issues.append("channel_policy_review:kame_gate_unpromoted_witness_allows_payloads")
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
        expected_payload_classes = {
            approval_item: policy_routes.get(str(route_id), {}).get("payload_classes")
            for approval_item, route_id in dict(route_map.get(channel_id) or {}).items()
        }
        if dict(channel.get("route_payload_classes_to_confirm") or {}) != expected_payload_classes:
            issues.append(f"channel_policy_review:{channel_id}:route_payload_classes_mismatch")
        if channel.get("kame_evidence_gate_to_confirm") != policy_kame_gate.get("gate_id"):
            issues.append(f"channel_policy_review:{channel_id}:kame_evidence_gate_mismatch")
        checklist = [str(item).lower() for item in channel.get("checklist") or []]
        if not any("interpreter_promoted" in item and "oracle_promoted" in item for item in checklist):
            issues.append(f"channel_policy_review:{channel_id}:missing_promoted_evidence_checklist")
        if not any(
            "transcript hypotheses" in item
            and "source" in item
            and "text_digest" in item
            and "arrival_phase" in item
            for item in checklist
        ):
            issues.append(f"channel_policy_review:{channel_id}:missing_transcript_hypothesis_metadata_checklist")
        if not any("unpromoted witness" in item and "absent" in item for item in checklist):
            issues.append(f"channel_policy_review:{channel_id}:missing_unpromoted_witness_checklist")
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

    if decision_scaffold.get("schema_version") != REVIEW_DECISION_SCHEMA_VERSION:
        issues.append("channel_policy_review_decision_scaffold:schema_version_mismatch")
    if decision_scaffold.get("artifact_id") != REVIEW_DECISION_ARTIFACT_ID:
        issues.append("channel_policy_review_decision_scaffold:artifact_id_mismatch")
    for key in ("milestone", "policy_id", "policy_version"):
        if decision_scaffold.get(key) != review.get(key):
            issues.append(f"channel_policy_review_decision_scaffold:{key}_mismatch")
    if decision_scaffold.get("review_artifact_ref") != "channel-policy-review.json":
        issues.append("channel_policy_review_decision_scaffold:review_artifact_ref_mismatch")
    if decision_scaffold.get("review_artifact_sha256") != hashlib.sha256(
        review_path.read_bytes()
    ).hexdigest():
        issues.append("channel_policy_review_decision_scaffold:review_artifact_sha256_mismatch")
    if decision_scaffold.get("review_artifact_stable_sha256") != stable_review_sha256(review):
        issues.append("channel_policy_review_decision_scaffold:review_artifact_stable_sha256_mismatch")
    if decision_scaffold.get("decision") != "pending_operator_review":
        issues.append("channel_policy_review_decision_scaffold:decision_not_pending")
    if decision_scaffold.get("review_status") != "pending_human_review":
        issues.append("channel_policy_review_decision_scaffold:review_status_not_pending")
    review_decision_effects = (
        review.get("decision_effects") if isinstance(review.get("decision_effects"), Mapping) else {}
    )
    scaffold_decision_effects = (
        decision_scaffold.get("decision_effects")
        if isinstance(decision_scaffold.get("decision_effects"), Mapping)
        else {}
    )
    if scaffold_decision_effects != review_decision_effects:
        issues.append("channel_policy_review_decision_scaffold:decision_effects_mismatch")
    live_effect = scaffold_decision_effects.get("approve_live_egress_after_external_credentials_are_bound")
    if not isinstance(live_effect, Mapping):
        issues.append("channel_policy_review_decision_scaffold:live_egress_effect_missing")
    else:
        if live_effect.get("permits_real_egress_now") is not False:
            issues.append("channel_policy_review_decision_scaffold:live_egress_effect_permits_egress")
        if live_effect.get("requires_runtime_credential_binding") is not True:
            issues.append("channel_policy_review_decision_scaffold:live_egress_effect_missing_credential_binding")
        if live_effect.get("requires_separate_runtime_approval") is not True:
            issues.append("channel_policy_review_decision_scaffold:live_egress_effect_missing_runtime_approval")
    if decision_scaffold.get("artifact_only") is not True:
        issues.append("channel_policy_review_decision_scaffold:artifact_only_not_true")
    if decision_scaffold.get("changes_policy") is not False:
        issues.append("channel_policy_review_decision_scaffold:changes_policy_not_false")
    if decision_scaffold.get("changes_readiness_by_itself") is not False:
        issues.append("channel_policy_review_decision_scaffold:changes_readiness_by_itself_not_false")
    if decision_scaffold.get("real_egress_enabled") is not False:
        issues.append("channel_policy_review_decision_scaffold:real_egress_enabled_not_false")
    scaffold_gate = (
        decision_scaffold.get("kame_action_evidence_gate")
        if isinstance(decision_scaffold.get("kame_action_evidence_gate"), Mapping)
        else {}
    )
    for key in ("gate_id", "design_reference", "required_interpreter_profile"):
        if scaffold_gate.get(key) != review_kame_gate.get(key):
            issues.append(f"channel_policy_review_decision_scaffold:kame_gate_{key}_mismatch")
    if scaffold_gate.get("raw_transcript_text_allowed_in_channel_egress") is not False:
        issues.append("channel_policy_review_decision_scaffold:kame_gate_allows_raw_transcript_text")
    if scaffold_gate.get("unpromoted_witness_may_enter_payloads") is not False:
        issues.append("channel_policy_review_decision_scaffold:kame_gate_allows_unpromoted_witness")
    signoffs = decision_scaffold.get("signoffs")
    if not isinstance(signoffs, list):
        issues.append("channel_policy_review_decision_scaffold:signoffs_not_list")
    else:
        scaffold_roles = {
            str(signoff.get("role"))
            for signoff in signoffs
            if isinstance(signoff, Mapping) and str(signoff.get("role") or "").strip()
        }
        if scaffold_roles != required_signoff_roles:
            issues.append("channel_policy_review_decision_scaffold:signoff_roles_mismatch")
        if any(isinstance(signoff, Mapping) and signoff.get("approved") is True for signoff in signoffs):
            issues.append("channel_policy_review_decision_scaffold:signoff_preapproved")
    scaffold_validation = validate_channel_policy_review_decision(
        decision_scaffold,
        review=review,
        review_path=review_path,
    )
    if "decision_not_review_closing" not in scaffold_validation:
        issues.append("channel_policy_review_decision_scaffold:unexpectedly_review_closing")
    if "decision_review_status_not_approved" not in scaffold_validation:
        issues.append("channel_policy_review_decision_scaffold:unexpectedly_approved")


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
    plan_run: Mapping[str, Any],
    discord_loopback_smoke: Mapping[str, Any],
    async_oracle_smoke: Mapping[str, Any],
    discord_session_cleanup_smoke: Mapping[str, Any],
    sidecar_fail_closed_smoke: Mapping[str, Any],
    tool_disclosure_smoke: Mapping[str, Any],
    ephemeral_tool_router_smoke: Mapping[str, Any],
    interpreter_request_packet: Mapping[str, Any],
    issues: list[str],
) -> None:
    for issue in validate_voice_operator_report(dict(readiness)):
        issues.append(f"voice_operator_readiness:{issue}")
    expected_payloads = {
        "smoke": discord_loopback_smoke,
        "async_oracle_smoke": async_oracle_smoke,
        "discord_session_cleanup_smoke": discord_session_cleanup_smoke,
        "sidecar_fail_closed_smoke": sidecar_fail_closed_smoke,
        "tool_disclosure_smoke": tool_disclosure_smoke,
        "ephemeral_tool_router_smoke": ephemeral_tool_router_smoke,
        "interpreter_request_packet": interpreter_request_packet,
    }
    for field, standalone_payload in expected_payloads.items():
        if readiness.get(field) != standalone_payload:
            issues.append(f"voice_operator_readiness:{field}_standalone_artifact_mismatch")
    _audit_voice_operator_proof_consistency(readiness=readiness, issues=issues)
    _audit_voice_operator_tool_disclosure_plan_projection(
        readiness=readiness,
        plan_run=plan_run,
        issues=issues,
    )
    _audit_voice_operator_async_oracle_plan_projection(
        readiness=readiness,
        plan_run=plan_run,
        issues=issues,
    )
    _audit_voice_operator_ephemeral_router_plan_projection(
        readiness=readiness,
        plan_run=plan_run,
        issues=issues,
    )


def _voice_operator_plan_details(plan_run: Mapping[str, Any]) -> Mapping[str, Any]:
    voice_result = next(
        (
            result
            for result in plan_run.get("results", [])
            if isinstance(result, Mapping)
            and result.get("milestone") == "milestone_1_real_voice_operator"
        ),
        {},
    )
    return voice_result.get("details") if isinstance(voice_result.get("details"), Mapping) else {}


def _audit_voice_operator_async_oracle_plan_projection(
    *,
    readiness: Mapping[str, Any],
    plan_run: Mapping[str, Any],
    issues: list[str],
) -> None:
    proof = (
        readiness.get("proofs", {}).get("async_oracle_jobs")
        if isinstance(readiness.get("proofs"), Mapping)
        and isinstance(readiness.get("proofs", {}).get("async_oracle_jobs"), Mapping)
        else {}
    )
    details = _voice_operator_plan_details(plan_run)
    projected = details.get("async_oracle_smoke") if isinstance(details.get("async_oracle_smoke"), Mapping) else {}
    _audit_plan_projection_witness_text_redacted(projected=projected, issues=issues)
    _audit_provider_text_alias_normalization(
        async_smoke=projected,
        prefix="plan_run:voice_operator.async_oracle_smoke",
        issues=issues,
    )
    _audit_kame_latency_breakdown(
        async_smoke=projected,
        prefix="plan_run:voice_operator.async_oracle_smoke",
        issues=issues,
    )
    _audit_reflex_ack_transcript(
        async_smoke=projected,
        prefix="plan_run:voice_operator.async_oracle_smoke",
        issues=issues,
    )
    _audit_minimum_interpreter_packet(
        async_smoke=projected,
        prefix="plan_run:voice_operator.async_oracle_smoke",
        issues=issues,
    )
    _audit_asr_optional_normal_path(
        async_smoke=projected,
        prefix="plan_run:voice_operator.async_oracle_smoke",
        issues=issues,
    )
    _audit_promoted_request_summary_contract(
        proof.get("external_frontend_promoted_request_summary"),
        label=(
            "voice_operator_readiness:proofs.async_oracle_jobs."
            "external_frontend_promoted_request_summary"
        ),
        issues=issues,
    )
    _audit_promoted_request_summary_contract(
        projected.get("external_frontend_promoted_request_summary"),
        label="plan_run:voice_operator.async_oracle_smoke.external_frontend_promoted_request_summary",
        issues=issues,
    )
    _audit_provisional_request_summary_contract(
        proof.get("external_frontend_provisional_request_summary"),
        label=(
            "voice_operator_readiness:proofs.async_oracle_jobs."
            "external_frontend_provisional_request_summary"
        ),
        issues=issues,
    )
    _audit_provisional_request_summary_contract(
        projected.get("external_frontend_provisional_request_summary"),
        label="plan_run:voice_operator.async_oracle_smoke.external_frontend_provisional_request_summary",
        issues=issues,
    )
    _audit_provisional_request_summary_contract(
        proof.get("external_frontend_status_provisional_request_summary"),
        label=(
            "voice_operator_readiness:proofs.async_oracle_jobs."
            "external_frontend_status_provisional_request_summary"
        ),
        issues=issues,
    )
    _audit_provisional_request_summary_contract(
        projected.get("external_frontend_status_provisional_request_summary"),
        label="plan_run:voice_operator.async_oracle_smoke.external_frontend_status_provisional_request_summary",
        issues=issues,
    )
    expected = {
        "status_bounded_overflow_visible": proof.get("status_bounded_overflow_visible"),
        "status_bounded_overflow_visible_job_count": proof.get(
            "status_bounded_overflow_visible_job_count"
        ),
        "status_bounded_overflow_hidden_job_count": proof.get(
            "status_bounded_overflow_hidden_job_count"
        ),
        "status_bounded_overflow_more_spoken_status": proof.get(
            "status_bounded_overflow_more_spoken_status"
        ),
        "status_bounded_overflow_last_visible_ordinal": proof.get(
            "status_bounded_overflow_last_visible_ordinal"
        ),
        "status_bounded_overflow_last_visible_label": proof.get(
            "status_bounded_overflow_last_visible_label"
        ),
        "status_bounded_overflow_hidden_ids_absent": proof.get(
            "status_bounded_overflow_hidden_ids_absent"
        ),
        "witness_fusion_timing_smoke_ok": proof.get("witness_fusion_timing_smoke_ok"),
        "witness_fusion_arrival_phases": list(proof.get("witness_fusion_arrival_phases") or []),
        "witness_fusion_case_job_ids": proof.get("witness_fusion_case_job_ids"),
        "witness_fusion_turn_ids": proof.get("witness_fusion_turn_ids"),
        "witness_fusion_audio_segment_refs": proof.get("witness_fusion_audio_segment_refs"),
        "witness_fusion_evidence_merge_keys": proof.get("witness_fusion_evidence_merge_keys"),
        "witness_fusion_merge_key_observed": proof.get("witness_fusion_merge_key_observed"),
        "witness_fusion_same_turn_convergence_ok": proof.get(
            "witness_fusion_same_turn_convergence_ok"
        ),
        "witness_fusion_same_turn_arrival_phases": list(
            proof.get("witness_fusion_same_turn_arrival_phases") or []
        ),
        "witness_fusion_same_turn_lineage": proof.get("witness_fusion_same_turn_lineage"),
        "witness_fusion_same_turn_phase_lineage": proof.get(
            "witness_fusion_same_turn_phase_lineage"
        ),
        "witness_fusion_same_turn_bundle_ids_by_phase": proof.get(
            "witness_fusion_same_turn_bundle_ids_by_phase"
        ),
        "witness_fusion_same_turn_job_ids_by_phase": proof.get(
            "witness_fusion_same_turn_job_ids_by_phase"
        ),
        "witness_fusion_same_turn_single_bundle": proof.get(
            "witness_fusion_same_turn_single_bundle"
        ),
        "witness_fusion_same_turn_one_oracle_job": proof.get(
            "witness_fusion_same_turn_one_oracle_job"
        ),
        "witness_fusion_same_turn_oracle_job_counts": proof.get(
            "witness_fusion_same_turn_oracle_job_counts"
        ),
        "witness_fusion_same_turn_no_duplicate_oracle_job": proof.get(
            "witness_fusion_same_turn_no_duplicate_oracle_job"
        ),
        "witness_fusion_same_turn_expected_merge_key": proof.get(
            "witness_fusion_same_turn_expected_merge_key"
        ),
        "witness_fusion_audio_metadata": proof.get("witness_fusion_audio_metadata"),
        "witness_fusion_bundle_audio_metadata": proof.get("witness_fusion_bundle_audio_metadata"),
        "witness_fusion_accepted_audio_gate_observed": proof.get(
            "witness_fusion_accepted_audio_gate_observed"
        ),
        "witness_fusion_early_single_bundle": proof.get("witness_fusion_early_single_bundle"),
        "witness_fusion_interpreter_prompt_input_order": list(
            proof.get("witness_fusion_interpreter_prompt_input_order") or []
        ),
        "witness_fusion_interpreter_prompt_input_order_expected": list(
            proof.get("witness_fusion_interpreter_prompt_input_order_expected") or []
        ),
        "witness_fusion_interpreter_prompt_input_order_visible": proof.get(
            "witness_fusion_interpreter_prompt_input_order_visible"
        ),
        "witness_fusion_interpreter_prompt_policy": proof.get(
            "witness_fusion_interpreter_prompt_policy"
        ),
        "witness_fusion_interpreter_prompt_policy_expected": proof.get(
            "witness_fusion_interpreter_prompt_policy_expected"
        ),
        "witness_fusion_interpreter_prompt_policy_version": proof.get(
            "witness_fusion_interpreter_prompt_policy_version"
        ),
        "witness_fusion_interpreter_prompt_policy_visible": proof.get(
            "witness_fusion_interpreter_prompt_policy_visible"
        ),
        "witness_fusion_with_single_bundle": proof.get("witness_fusion_with_single_bundle"),
        "witness_fusion_late_single_bundle": proof.get("witness_fusion_late_single_bundle"),
        "witness_fusion_no_duplicate_oracle_jobs": proof.get(
            "witness_fusion_no_duplicate_oracle_jobs"
        ),
        "witness_fusion_partial_superseded_by_final": proof.get(
            "witness_fusion_partial_superseded_by_final"
        ),
        "witness_fusion_partial_active_hypothesis": _redact_projected_witness_hypothesis(
            proof.get("witness_fusion_partial_active_hypothesis")
        ),
        "witness_fusion_adjudications": proof.get("witness_fusion_adjudications"),
        "witness_fusion_rejection_reasons": proof.get("witness_fusion_rejection_reasons"),
        "witness_fusion_adjudication_outcomes_observed": proof.get(
            "witness_fusion_adjudication_outcomes_observed"
        ),
        "runtime_kame_action_gate_smoke_ok": proof.get("runtime_kame_action_gate_smoke_ok"),
        "runtime_kame_action_gate_hypothesis_only_ok": proof.get(
            "runtime_kame_action_gate_hypothesis_only_ok"
        ),
        "runtime_kame_action_gate_hypothesis_only_issues": list(
            proof.get("runtime_kame_action_gate_hypothesis_only_issues") or []
        ),
        "runtime_kame_action_gate_degraded_text_only_ok": proof.get(
            "runtime_kame_action_gate_degraded_text_only_ok"
        ),
        "runtime_kame_action_gate_degraded_text_only_issues": list(
            proof.get("runtime_kame_action_gate_degraded_text_only_issues") or []
        ),
        "runtime_kame_action_gate_degraded_text_only_status": proof.get(
            "runtime_kame_action_gate_degraded_text_only_status"
        ),
        "runtime_kame_action_gate_degraded_text_only_reason": proof.get(
            "runtime_kame_action_gate_degraded_text_only_reason"
        ),
        "runtime_kame_action_gate_degraded_text_only_raw_audio_available": proof.get(
            "runtime_kame_action_gate_degraded_text_only_raw_audio_available"
        ),
        "runtime_kame_action_gate_degraded_text_only_preserves_hypothesis": proof.get(
            "runtime_kame_action_gate_degraded_text_only_preserves_hypothesis"
        ),
        "runtime_kame_action_gate_degraded_oracle_promoted_ok": proof.get(
            "runtime_kame_action_gate_degraded_oracle_promoted_ok"
        ),
        "runtime_kame_action_gate_degraded_oracle_promoted_issues": list(
            proof.get("runtime_kame_action_gate_degraded_oracle_promoted_issues") or []
        ),
        "runtime_kame_action_gate_degraded_oracle_promoted_authorities": list(
            proof.get("runtime_kame_action_gate_degraded_oracle_promoted_authorities") or []
        ),
        "runtime_kame_action_gate_degraded_oracle_promoted_status": proof.get(
            "runtime_kame_action_gate_degraded_oracle_promoted_status"
        ),
        "runtime_kame_action_gate_degraded_oracle_promoted_raw_audio_available": proof.get(
            "runtime_kame_action_gate_degraded_oracle_promoted_raw_audio_available"
        ),
        "runtime_kame_action_gate_degraded_oracle_promoted_consumed_before_action": proof.get(
            "runtime_kame_action_gate_degraded_oracle_promoted_consumed_before_action"
        ),
        "runtime_kame_action_gate_promoted_ok": proof.get("runtime_kame_action_gate_promoted_ok"),
        "runtime_kame_action_gate_promoted_authorities": list(
            proof.get("runtime_kame_action_gate_promoted_authorities") or []
        ),
        "runtime_kame_action_gate_promoted_consumed_before_action": proof.get(
            "runtime_kame_action_gate_promoted_consumed_before_action"
        ),
        "runtime_kame_action_gate_self_attested_ok": proof.get(
            "runtime_kame_action_gate_self_attested_ok"
        ),
        "runtime_kame_action_gate_self_attested_issues": list(
            proof.get("runtime_kame_action_gate_self_attested_issues") or []
        ),
        "runtime_kame_action_gate_self_attested_authorities": list(
            proof.get("runtime_kame_action_gate_self_attested_authorities") or []
        ),
        "runtime_kame_action_gate_self_attested_consumed_before_action": proof.get(
            "runtime_kame_action_gate_self_attested_consumed_before_action"
        ),
        "runtime_kame_action_gate_missing_tool_disclosure_ok": proof.get(
            "runtime_kame_action_gate_missing_tool_disclosure_ok"
        ),
        "runtime_kame_action_gate_missing_tool_disclosure_issues": list(
            proof.get("runtime_kame_action_gate_missing_tool_disclosure_issues") or []
        ),
        "runtime_kame_action_gate_missing_tool_disclosure_authorities": list(
            proof.get("runtime_kame_action_gate_missing_tool_disclosure_authorities") or []
        ),
        "runtime_kame_action_gate_tool_disclosure_ref_observed": proof.get(
            "runtime_kame_action_gate_tool_disclosure_ref_observed"
        ),
    }
    for field, expected_value in expected.items():
        if projected.get(field) != expected_value:
            issues.append(f"plan_run:voice_operator.async_oracle_smoke.{field}_mismatch")


def _audit_voice_operator_ephemeral_router_plan_projection(
    *,
    readiness: Mapping[str, Any],
    plan_run: Mapping[str, Any],
    issues: list[str],
) -> None:
    proof = (
        readiness.get("proofs", {}).get("ephemeral_tool_router")
        if isinstance(readiness.get("proofs"), Mapping)
        and isinstance(readiness.get("proofs", {}).get("ephemeral_tool_router"), Mapping)
        else {}
    )
    details = _voice_operator_plan_details(plan_run)
    projected = details.get("ephemeral_tool_router") if isinstance(details.get("ephemeral_tool_router"), Mapping) else {}
    expected = {
        "ok": proof.get("ok"),
        "router_mode": proof.get("router_mode"),
        "provider_network": proof.get("provider_network"),
        "model_call": proof.get("model_call"),
        "router_call_count": proof.get("router_call_count"),
        "selected_voiceops_toolsets": proof.get("selected_voiceops_toolsets"),
        "selected_no_tools_toolsets": proof.get("selected_no_tools_toolsets"),
        "router_transcript_persistent": proof.get("router_transcript_persistent"),
        "router_tool_calls_allowed": proof.get("router_tool_calls_allowed"),
        "test_ref_count": len(proof.get("external_test_refs") or []),
    }
    for field, expected_value in expected.items():
        if projected.get(field) != expected_value:
            issues.append(f"plan_run:voice_operator.ephemeral_tool_router.{field}_mismatch")


def _audit_voice_operator_tool_disclosure_plan_projection(
    *,
    readiness: Mapping[str, Any],
    plan_run: Mapping[str, Any],
    issues: list[str],
) -> None:
    proof = (
        readiness.get("proofs", {}).get("tool_disclosure")
        if isinstance(readiness.get("proofs"), Mapping)
        and isinstance(readiness.get("proofs", {}).get("tool_disclosure"), Mapping)
        else {}
    )
    details = _voice_operator_plan_details(plan_run)
    projected = details.get("tool_disclosure") if isinstance(details.get("tool_disclosure"), Mapping) else {}
    expected = {
        "ok": proof.get("ok"),
        "schema_source": proof.get("schema_source"),
        "representative_schema": proof.get("representative_schema"),
        "missing_registered_core_tools": proof.get("missing_registered_core_tools") or [],
        "config": proof.get("config"),
        "input_core_tools": proof.get("input_core_tools"),
        "visible_tool_names": proof.get("visible_tool_names"),
        "visible_non_bridge_tool_names": proof.get("visible_non_bridge_tool_names") or [],
        "bridge_tool_names": proof.get("bridge_tool_names") or [],
        "hidden_core_tool_names": proof.get("hidden_core_tool_names"),
        "input_core_tool_count": proof.get("input_core_tool_count"),
        "hidden_core_tool_count": proof.get("hidden_core_tool_count"),
        "bridge_tool_count": proof.get("bridge_tool_count"),
        "core_tools_hidden_all": proof.get("core_tools_hidden_all"),
        "broad_core_tools_visible": proof.get("broad_core_tools_visible"),
        "deferred_count": proof.get("deferred_count"),
        "deferred_tokens": proof.get("deferred_tokens"),
        "input_schema_tokens": proof.get("input_schema_tokens"),
        "visible_schema_tokens": proof.get("visible_schema_tokens"),
        "token_reduction_estimate": proof.get("token_reduction_estimate"),
        "test_ref_count": len(proof.get("external_test_refs") or []),
    }
    for field, expected_value in expected.items():
        if projected.get(field) != expected_value:
            issues.append(f"plan_run:voice_operator.tool_disclosure.{field}_mismatch")


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


def _audit_provisional_request_summary_contract(
    summary: Any,
    *,
    label: str,
    issues: list[str],
) -> None:
    if not isinstance(summary, Mapping):
        issues.append(f"{label}_not_object")
        return
    if summary.get("kind") != "reflex_hypothesis":
        issues.append(f"{label}_missing_reflex_kind")
    if summary.get("authority") == "reflex_hypothesis":
        issues.append(f"{label}_uses_kind_as_authority")
    if summary.get("authority") != "hypothesis":
        issues.append(f"{label}_authority_not_hypothesis")
    if summary.get("tool_authority") is not False:
        issues.append(f"{label}_tool_authority_not_false")


def _audit_promoted_request_summary_contract(
    summary: Any,
    *,
    label: str,
    issues: list[str],
) -> None:
    if not isinstance(summary, Mapping):
        issues.append(f"{label}_not_object")
        return
    if summary.get("authority") not in {"interpreter_promoted", "oracle_promoted"}:
        issues.append(f"{label}_authority_not_promoted")
    if summary.get("tool_authority") is not False:
        issues.append(f"{label}_tool_authority_not_false")


def _audit_external_frontend_hypothesis_adjudication(
    hypotheses: list[Any],
    *,
    issues: list[str],
    label: str,
) -> None:
    for index, hypothesis in enumerate(hypotheses):
        if not isinstance(hypothesis, Mapping):
            issues.append(f"voice_operator_readiness:{label}.{index}_not_object")
            continue
        _audit_transcript_hypothesis_contract(
            hypothesis,
            issues=issues,
            label=f"voice_operator_readiness:{label}.{index}",
        )
        adjudication = str(hypothesis.get("adjudication") or "").strip()
        if not adjudication:
            issues.append(f"voice_operator_readiness:{label}.{index}_missing_adjudication")
            continue
        if adjudication not in VALID_WITNESS_ADJUDICATION_OUTCOMES:
            issues.append(f"voice_operator_readiness:{label}.{index}_invalid_adjudication")
            continue
        if adjudication != "rejected_or_diagnostic_only":
            continue
        raw_reasons = hypothesis.get("rejection_reasons")
        if isinstance(raw_reasons, str):
            reasons = [raw_reasons]
        elif isinstance(raw_reasons, list):
            reasons = [str(reason).strip() for reason in raw_reasons if str(reason).strip()]
        else:
            reasons = []
        if not reasons:
            issues.append(f"voice_operator_readiness:{label}.{index}_missing_rejection_reasons")
            continue
        invalid_reasons = [
            reason for reason in reasons if reason not in VALID_WITNESS_REJECTION_REASONS
        ]
        if invalid_reasons:
            issues.append(f"voice_operator_readiness:{label}.{index}_invalid_rejection_reasons")


def _audit_transcript_hypothesis_contract(
    hypothesis: Mapping[str, Any],
    *,
    issues: list[str],
    label: str,
) -> None:
    for field, expected_value in REQUIRED_TRANSCRIPT_HYPOTHESIS_CONTRACT.items():
        if hypothesis.get(field) != expected_value:
            issues.append(f"{label}_{field}_mismatch")
    arrival_phase = str(hypothesis.get("arrival_phase") or "").strip()
    if not arrival_phase:
        issues.append(f"{label}_missing_arrival_phase")
    elif arrival_phase not in VALID_WITNESS_ARRIVAL_PHASES:
        issues.append(f"{label}_invalid_arrival_phase")


def _canonical_witness_adjudication_rows(hypotheses: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, Mapping):
            continue
        adjudication = str(hypothesis.get("adjudication") or "").strip()
        if not adjudication:
            continue
        row: dict[str, Any] = {
            "source": str(hypothesis.get("source") or "").strip(),
            "kind": str(hypothesis.get("kind") or "").strip(),
            "text_digest": str(hypothesis.get("text_digest") or "").strip(),
            "adjudication": adjudication,
        }
        reasons = hypothesis.get("rejection_reasons")
        if isinstance(reasons, list):
            compact_reasons = [
                str(reason).strip()
                for reason in reasons
                if str(reason).strip()
            ]
            if compact_reasons:
                row["rejection_reasons"] = compact_reasons
        rows.append({key: value for key, value in row.items() if value not in ("", [], {})})
    return rows


def _iter_nested_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        strings: list[str] = []
        for item in value.values():
            strings.extend(_iter_nested_strings(item))
        return strings
    if isinstance(value, list):
        strings: list[str] = []
        for item in value:
            strings.extend(_iter_nested_strings(item))
        return strings
    return []


def _iter_nested_authorities(value: Any) -> list[str]:
    authorities: list[str] = []
    if isinstance(value, Mapping):
        raw_authority = value.get("authority")
        if isinstance(raw_authority, str) and raw_authority.strip():
            authorities.append(raw_authority.strip())
        for item in value.values():
            authorities.extend(_iter_nested_authorities(item))
    elif isinstance(value, list):
        for item in value:
            authorities.extend(_iter_nested_authorities(item))
    return authorities


def _sha256_text_digest(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _redact_projected_witness_hypothesis(value: Any) -> Any:
    if isinstance(value, list):
        return [_redact_projected_witness_hypothesis(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    redacted = {
        key: _redact_projected_witness_hypothesis(item)
        for key, item in value.items()
        if key not in {"text", "superseded_partial_texts"}
    }
    text = value.get("text")
    if isinstance(text, str) and text:
        redacted.setdefault("text_digest", _sha256_text_digest(text))
        redacted["text_redacted"] = True
    elif "text" in value:
        redacted["text_redacted"] = True
    superseded_texts = value.get("superseded_partial_texts")
    if isinstance(superseded_texts, (list, tuple)):
        redacted["superseded_partial_text_digests"] = [
            _sha256_text_digest(str(item))
            for item in superseded_texts
            if str(item)
        ]
        redacted["superseded_partial_texts_redacted"] = True
    return redacted


def _redacted_witness_text_projection(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        return {"text_redacted": False, "text_digest": None}
    return {"text_redacted": True, "text_digest": _sha256_text_digest(value)}


def _audit_plan_projection_witness_text_redacted(
    *,
    projected: Mapping[str, Any],
    issues: list[str],
) -> None:
    frontend = projected.get("external_frontend_witness_metadata")
    if not isinstance(frontend, Mapping):
        issues.append(
            "plan_run:voice_operator.async_oracle_smoke.external_frontend_witness_metadata_not_object"
        )
    else:
        if "text" in frontend:
            issues.append(
                "plan_run:voice_operator.async_oracle_smoke.external_frontend_witness_metadata_raw_text_present"
            )
        if frontend.get("text_redacted") is not True:
            issues.append(
                "plan_run:voice_operator.async_oracle_smoke.external_frontend_witness_metadata_text_not_redacted"
            )

    partial = projected.get("witness_fusion_partial_active_hypothesis")
    if not isinstance(partial, Mapping):
        issues.append(
            "plan_run:voice_operator.async_oracle_smoke.witness_fusion_partial_active_hypothesis_not_object"
        )
    else:
        if "text" in partial:
            issues.append(
                "plan_run:voice_operator.async_oracle_smoke.witness_fusion_partial_active_hypothesis_raw_text_present"
            )
        if "superseded_partial_texts" in partial:
            issues.append(
                "plan_run:voice_operator.async_oracle_smoke.witness_fusion_partial_active_hypothesis_raw_partials_present"
            )
        if partial.get("text_redacted") is not True:
            issues.append(
                "plan_run:voice_operator.async_oracle_smoke.witness_fusion_partial_active_hypothesis_text_not_redacted"
            )
    if "energy_gate_low_energy_witness_text" in projected:
        issues.append(
            "plan_run:voice_operator.async_oracle_smoke.energy_gate_low_energy_witness_raw_text_present"
        )
    energy_projection = projected.get("energy_gate_low_energy_witness_text_projection")
    if not isinstance(energy_projection, Mapping):
        issues.append(
            "plan_run:voice_operator.async_oracle_smoke.energy_gate_low_energy_witness_text_projection_missing"
        )
    elif energy_projection.get("text_redacted") is not True:
        issues.append(
            "plan_run:voice_operator.async_oracle_smoke.energy_gate_low_energy_witness_text_not_redacted"
        )


def _audit_witness_assisted_action_sinks(
    *,
    async_smoke: Mapping[str, Any],
    issues: list[str],
) -> None:
    sink_values = async_smoke.get("witness_assisted_voiceops_action_sink_values")
    if not isinstance(sink_values, Mapping):
        issues.append(
            "voice_operator_readiness:async_oracle_smoke.witness_assisted_voiceops_action_sink_values_not_object"
        )
        return

    witness_text = str(
        async_smoke.get("witness_assisted_voiceops_action_witness_text") or ""
    ).strip()
    allowed_authorities = {"interpreter_promoted", "oracle_promoted"}

    for sink_key, sink_value in sink_values.items():
        sink_label = str(sink_key)
        if witness_text:
            for nested_text in _iter_nested_strings(sink_value):
                if witness_text in nested_text:
                    issues.append(
                        "voice_operator_readiness:async_oracle_smoke."
                        f"witness_assisted_voiceops_action_sink_values_raw_witness_present:{sink_label}"
                    )
                    break

        authorities = _iter_nested_authorities(sink_value)
        if not authorities and "." in sink_label:
            parent_value = sink_values.get(sink_label.split(".", 1)[0])
            authorities = _iter_nested_authorities(parent_value)

        if not authorities:
            issues.append(
                "voice_operator_readiness:async_oracle_smoke."
                f"witness_assisted_voiceops_action_sink_values_missing_promoted_source:{sink_label}"
            )
            continue

        invalid_authorities = [
            authority for authority in authorities if authority not in allowed_authorities
        ]
        if invalid_authorities:
            issues.append(
                "voice_operator_readiness:async_oracle_smoke."
                f"witness_assisted_voiceops_action_sink_values_unpromoted_source:{sink_label}"
            )


def _audit_provider_text_alias_normalization(
    *,
    async_smoke: Mapping[str, Any],
    prefix: str,
    issues: list[str],
) -> None:
    expected = list(EXPECTED_PROVIDER_TEXT_ALIAS_KEYS)
    if async_smoke.get("provider_text_alias_normalization_smoke_ok") is not True:
        issues.append(f"{prefix}.provider_text_alias_normalization_smoke_not_ok")
    if async_smoke.get("provider_text_alias_keys_expected") != expected:
        issues.append(f"{prefix}.provider_text_alias_keys_expected_mismatch")
    if async_smoke.get("provider_text_alias_keys_observed") != expected:
        issues.append(f"{prefix}.provider_text_alias_keys_observed_mismatch")
    if async_smoke.get("provider_text_alias_hypothesis_count") != len(expected):
        issues.append(f"{prefix}.provider_text_alias_hypothesis_count_mismatch")
    if async_smoke.get("provider_text_alias_hypothesis_contract_ok") is not True:
        issues.append(f"{prefix}.provider_text_alias_hypothesis_contract_not_ok")
    if async_smoke.get("provider_text_alias_no_oracle_text_leak") is not True:
        issues.append(f"{prefix}.provider_text_alias_oracle_text_leak")


def _non_negative_metric(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value >= 0


def _audit_kame_latency_breakdown(
    *,
    async_smoke: Mapping[str, Any],
    prefix: str,
    issues: list[str],
) -> None:
    expected = list(EXPECTED_KAME_LATENCY_BREAKDOWN_SEGMENTS)
    if async_smoke.get("kame_latency_breakdown_smoke_ok") is not True:
        issues.append(f"{prefix}.kame_latency_breakdown_smoke_not_ok")
    if async_smoke.get("kame_latency_breakdown_required_segments") != expected:
        issues.append(f"{prefix}.kame_latency_breakdown_required_segments_mismatch")

    segments = async_smoke.get("kame_latency_breakdown_segments_ms")
    if not isinstance(segments, Mapping):
        issues.append(f"{prefix}.kame_latency_breakdown_segments_not_object")
        segments = {}
    for key in expected:
        if not _non_negative_metric(segments.get(key)):
            issues.append(f"{prefix}.kame_latency_breakdown_segment_missing_or_invalid:{key}")

    timeline = async_smoke.get("kame_latency_breakdown_timeline_ms")
    if not isinstance(timeline, Mapping):
        issues.append(f"{prefix}.kame_latency_breakdown_timeline_not_object")
    elif not all(_non_negative_metric(value) for value in timeline.values()):
        issues.append(f"{prefix}.kame_latency_breakdown_timeline_invalid_metric")

    if async_smoke.get("kame_latency_breakdown_monotonic") is not True:
        issues.append(f"{prefix}.kame_latency_breakdown_not_monotonic")
    if not _non_negative_metric(async_smoke.get("kame_latency_breakdown_total_ms")):
        issues.append(f"{prefix}.kame_latency_breakdown_total_missing_or_invalid")


def _audit_reflex_ack_transcript(
    *,
    async_smoke: Mapping[str, Any],
    prefix: str,
    issues: list[str],
) -> None:
    if async_smoke.get("reflex_ack_transcript_smoke_ok") is not True:
        issues.append(f"{prefix}.reflex_ack_transcript_smoke_not_ok")
    if async_smoke.get("reflex_ack_transcript_visible") is not True:
        issues.append(f"{prefix}.reflex_ack_transcript_not_visible")
    record = async_smoke.get("reflex_ack_transcript_record")
    if not isinstance(record, Mapping):
        issues.append(f"{prefix}.reflex_ack_transcript_record_not_object")
        record = {}
    audit_record = async_smoke.get("reflex_ack_transcript_audit_record")
    if not isinstance(audit_record, Mapping):
        issues.append(f"{prefix}.reflex_ack_transcript_audit_record_not_object")
        audit_record = {}
    expected_record_fields = {
        "schema_version": "voiceops.reflex_ack_transcript.v1",
        "speaker": "assistant_reflex",
        "text_source": "reflex_acknowledgement",
        "authority": "reflex_hypothesis",
        "durability": "visible_transcript_and_audit",
        "provisional": True,
        "action_authority": False,
        "tool_authority": False,
        "spoken": True,
        "visible_to_user": True,
    }
    for key, expected_value in expected_record_fields.items():
        if record.get(key) != expected_value:
            issues.append(f"{prefix}.reflex_ack_transcript_record_{key}_mismatch")
    if not str(record.get("text") or "").strip():
        issues.append(f"{prefix}.reflex_ack_transcript_record_text_missing")
    if record.get("turn_id") != async_smoke.get("reflex_ack_turn_id"):
        issues.append(f"{prefix}.reflex_ack_transcript_turn_id_mismatch")
    if record.get("oracle_job_id") != async_smoke.get("reflex_ack_oracle_job_id"):
        issues.append(f"{prefix}.reflex_ack_transcript_oracle_job_id_mismatch")
    if audit_record.get("event_id") != record.get("audit_event_id"):
        issues.append(f"{prefix}.reflex_ack_transcript_audit_event_mismatch")
    if audit_record.get("event") != "reflex.ack.transcript_recorded":
        issues.append(f"{prefix}.reflex_ack_transcript_audit_event_name_mismatch")
    if audit_record.get("turn_id") != record.get("turn_id"):
        issues.append(f"{prefix}.reflex_ack_transcript_audit_turn_id_mismatch")
    if audit_record.get("oracle_job_id") != record.get("oracle_job_id"):
        issues.append(f"{prefix}.reflex_ack_transcript_audit_oracle_job_id_mismatch")
    if audit_record.get("authority") != "reflex_hypothesis":
        issues.append(f"{prefix}.reflex_ack_transcript_audit_authority_mismatch")
    if audit_record.get("action_authority") is not False:
        issues.append(f"{prefix}.reflex_ack_transcript_audit_action_authority_not_false")
    if audit_record.get("tool_authority") is not False:
        issues.append(f"{prefix}.reflex_ack_transcript_audit_tool_authority_not_false")


def _audit_minimum_interpreter_packet(
    *,
    async_smoke: Mapping[str, Any],
    prefix: str,
    issues: list[str],
) -> None:
    if async_smoke.get("minimum_interpreter_packet_smoke_ok") is not True:
        issues.append(f"{prefix}.minimum_interpreter_packet_smoke_not_ok")
    packet = async_smoke.get("minimum_interpreter_packet")
    if not isinstance(packet, Mapping):
        issues.append(f"{prefix}.minimum_interpreter_packet_not_object")
        packet = {}
    metadata = packet.get("metadata") if isinstance(packet.get("metadata"), Mapping) else {}
    reflex = packet.get("reflex") if isinstance(packet.get("reflex"), Mapping) else {}
    hypotheses = (
        packet.get("transcript_hypotheses")
        if isinstance(packet.get("transcript_hypotheses"), list)
        else []
    )
    if packet.get("schema_version") != "voiceops.minimum_interpreter_packet.v1":
        issues.append(f"{prefix}.minimum_interpreter_packet_schema_mismatch")
    if packet.get("mode") != "witness_assisted_direct_audio":
        issues.append(f"{prefix}.minimum_interpreter_packet_mode_mismatch")
    if packet.get("interpreter_input_order") != [
        "raw_audio",
        "metadata",
        "reflex",
        "transcript_hypotheses",
    ]:
        issues.append(f"{prefix}.minimum_interpreter_packet_input_order_mismatch")
    if packet.get("audio_segment_ref") != async_smoke.get("external_frontend_audio_segment_ref"):
        issues.append(f"{prefix}.minimum_interpreter_packet_audio_segment_ref_mismatch")
    if metadata.get("evidence_bundle_id") != async_smoke.get("external_frontend_evidence_bundle_id"):
        issues.append(f"{prefix}.minimum_interpreter_packet_bundle_id_mismatch")
    if metadata.get("evidence_merge_key") != async_smoke.get("external_frontend_evidence_merge_key"):
        issues.append(f"{prefix}.minimum_interpreter_packet_merge_key_mismatch")
    if metadata.get("vad_speech") is not True:
        issues.append(f"{prefix}.minimum_interpreter_packet_vad_speech_not_true")
    if metadata.get("energy_gate") != "accepted":
        issues.append(f"{prefix}.minimum_interpreter_packet_energy_gate_mismatch")
    if reflex.get("authority") != "reflex_hypothesis":
        issues.append(f"{prefix}.minimum_interpreter_packet_reflex_authority_mismatch")
    if reflex.get("tool_authority") is not False:
        issues.append(f"{prefix}.minimum_interpreter_packet_reflex_tool_authority_not_false")
    if not hypotheses:
        issues.append(f"{prefix}.minimum_interpreter_packet_hypotheses_missing")
    if len(hypotheses) != int(async_smoke.get("minimum_interpreter_packet_witness_count") or -1):
        issues.append(f"{prefix}.minimum_interpreter_packet_witness_count_mismatch")
    for index, hypothesis in enumerate(hypotheses):
        if not isinstance(hypothesis, Mapping):
            issues.append(f"{prefix}.minimum_interpreter_packet_hypothesis_not_object:{index}")
            continue
        if "text" in hypothesis:
            issues.append(f"{prefix}.minimum_interpreter_packet_hypothesis_text_not_redacted:{index}")
        expected = {
            "role": "witness_context",
            "authority": "hypothesis",
            "promotion_required": "interpreter_promoted_or_oracle_promoted",
            "tool_authority": False,
            "text_redacted": True,
        }
        for key, expected_value in expected.items():
            if hypothesis.get(key) != expected_value:
                issues.append(f"{prefix}.minimum_interpreter_packet_hypothesis_{key}_mismatch:{index}")
    packet_text = json.dumps(packet, sort_keys=True, default=str).lower()
    if "prepare an external came hand off" in packet_text:
        issues.append(f"{prefix}.minimum_interpreter_packet_raw_witness_text_present")
    if async_smoke.get("minimum_interpreter_packet_text_redacted") is not True:
        issues.append(f"{prefix}.minimum_interpreter_packet_text_redacted_not_true")
    if async_smoke.get("minimum_interpreter_packet_raw_audio_primary") is not True:
        issues.append(f"{prefix}.minimum_interpreter_packet_raw_audio_primary_not_true")
    if async_smoke.get("minimum_interpreter_packet_hypotheses_authority") is not True:
        issues.append(f"{prefix}.minimum_interpreter_packet_hypotheses_authority_not_true")


def _audit_asr_optional_normal_path(
    *,
    async_smoke: Mapping[str, Any],
    prefix: str,
    issues: list[str],
) -> None:
    if async_smoke.get("asr_optional_normal_path_smoke_ok") is not True:
        issues.append(f"{prefix}.asr_optional_normal_path_smoke_not_ok")
    if async_smoke.get("asr_optional_normal_path_audio_segment_ref") != "artifact://voiceclaw/no-asr.wav":
        issues.append(f"{prefix}.asr_optional_normal_path_audio_segment_ref_mismatch")
    if list(async_smoke.get("asr_optional_normal_path_prompt_input_order") or [])[:3] != [
        "raw_audio",
        "metadata",
        "reflex",
    ]:
        issues.append(f"{prefix}.asr_optional_normal_path_prompt_input_order_mismatch")
    if async_smoke.get("asr_optional_normal_path_classic_asr_absent") is not True:
        issues.append(f"{prefix}.asr_optional_normal_path_classic_asr_not_absent")
    raw_kinds = async_smoke.get("asr_optional_normal_path_hypothesis_kinds")
    kinds = {str(value) for value in raw_kinds if value} if isinstance(raw_kinds, (list, tuple)) else set()
    raw_sources = async_smoke.get("asr_optional_normal_path_hypothesis_sources")
    sources = (
        {str(value).lower() for value in raw_sources if value}
        if isinstance(raw_sources, (list, tuple))
        else set()
    )
    if "classic_asr_hypothesis" in kinds:
        issues.append(f"{prefix}.asr_optional_normal_path_classic_asr_kind_present")
    if any("asr" in source for source in sources):
        issues.append(f"{prefix}.asr_optional_normal_path_asr_source_present")
    if async_smoke.get("asr_optional_normal_path_raw_audio_available") is not True:
        issues.append(f"{prefix}.asr_optional_normal_path_raw_audio_not_available")
    if async_smoke.get("asr_optional_normal_path_raw_audio_authority") != "primary_audio":
        issues.append(f"{prefix}.asr_optional_normal_path_raw_audio_authority_mismatch")
    if async_smoke.get("asr_optional_normal_path_interpreter_profile") != "witness_assisted_direct_audio":
        issues.append(f"{prefix}.asr_optional_normal_path_interpreter_profile_mismatch")
    if async_smoke.get("asr_optional_normal_path_interpreter_promoted_authority") != "interpreter_promoted":
        issues.append(f"{prefix}.asr_optional_normal_path_interpreter_promoted_authority_mismatch")
    if async_smoke.get("asr_optional_normal_path_started_observed") is not True:
        issues.append(f"{prefix}.asr_optional_normal_path_started_not_observed")
    if async_smoke.get("asr_optional_normal_path_completed_observed") is not True:
        issues.append(f"{prefix}.asr_optional_normal_path_completed_not_observed")
    if async_smoke.get("asr_optional_normal_path_status_state") != "completed":
        issues.append(f"{prefix}.asr_optional_normal_path_status_not_completed")


def _audit_witness_fusion_multi_speaker_binding(
    *,
    async_smoke: Mapping[str, Any],
    issues: list[str],
) -> None:
    prefix = "voice_operator_readiness:async_oracle_smoke"
    required_true_fields = (
        "witness_fusion_multi_speaker_witness_smoke_ok",
        "witness_fusion_multi_speaker_wrong_witness_rejected",
        "witness_fusion_multi_speaker_bound_to_second_human",
        "witness_fusion_multi_speaker_action_sinks_clean",
    )
    for field in required_true_fields:
        if async_smoke.get(field) is not True:
            issues.append(f"{prefix}.{field}_not_true")

    wrong_witness_text = str(
        async_smoke.get("witness_fusion_multi_speaker_wrong_witness_text") or ""
    ).strip()
    wrong_witness = async_smoke.get("witness_fusion_multi_speaker_wrong_witness")
    if not isinstance(wrong_witness, Mapping):
        issues.append(f"{prefix}.witness_fusion_multi_speaker_wrong_witness_missing")
        return

    if wrong_witness_text and wrong_witness.get("text") != wrong_witness_text:
        issues.append(f"{prefix}.witness_fusion_multi_speaker_wrong_witness_text_mismatch")
    if wrong_witness_text and wrong_witness.get("text_digest") != _sha256_text_digest(
        wrong_witness_text
    ):
        issues.append(f"{prefix}.witness_fusion_multi_speaker_wrong_witness_digest_mismatch")
    if wrong_witness.get("adjudication") != "rejected_or_diagnostic_only":
        issues.append(f"{prefix}.witness_fusion_multi_speaker_wrong_witness_not_rejected")
    if wrong_witness.get("authority") != "hypothesis":
        issues.append(f"{prefix}.witness_fusion_multi_speaker_wrong_witness_authority_mismatch")
    if wrong_witness.get("role") != "witness_context":
        issues.append(f"{prefix}.witness_fusion_multi_speaker_wrong_witness_role_mismatch")
    if wrong_witness.get("tool_authority") is not False:
        issues.append(f"{prefix}.witness_fusion_multi_speaker_wrong_witness_tool_authority_not_false")
    if wrong_witness.get("promotion_required") != "interpreter_promoted_or_oracle_promoted":
        issues.append(f"{prefix}.witness_fusion_multi_speaker_wrong_witness_promotion_required_mismatch")

    raw_reasons = wrong_witness.get("rejection_reasons")
    if isinstance(raw_reasons, str):
        reasons = [raw_reasons]
    elif isinstance(raw_reasons, list):
        reasons = [str(reason).strip() for reason in raw_reasons if str(reason).strip()]
    else:
        reasons = []
    if not reasons:
        issues.append(f"{prefix}.witness_fusion_multi_speaker_wrong_witness_missing_rejection_reasons")
    else:
        invalid_reasons = [
            reason for reason in reasons if reason not in VALID_WITNESS_REJECTION_REASONS
        ]
        if invalid_reasons:
            issues.append(f"{prefix}.witness_fusion_multi_speaker_wrong_witness_invalid_rejection_reasons")
        if not {"ambiguous_speaker", "wrong_speaker", "wrong_channel", "stale_witness"}.intersection(
            reasons
        ):
            issues.append(f"{prefix}.witness_fusion_multi_speaker_wrong_witness_no_binding_rejection_reason")

    accepted_speaker = async_smoke.get("witness_fusion_multi_speaker_accepted_speaker")
    wrong_speaker = async_smoke.get("witness_fusion_multi_speaker_wrong_witness_speaker")
    if not isinstance(accepted_speaker, Mapping) or not isinstance(wrong_speaker, Mapping):
        issues.append(f"{prefix}.witness_fusion_multi_speaker_speaker_metadata_missing")
    else:
        accepted_user_id = str(accepted_speaker.get("channel_user_id") or "").strip()
        wrong_user_id = str(wrong_speaker.get("channel_user_id") or "").strip()
        if not accepted_user_id or not wrong_user_id or accepted_user_id == wrong_user_id:
            issues.append(f"{prefix}.witness_fusion_multi_speaker_not_bound_to_second_human")
        if "ambiguous_speaker" in reasons and wrong_speaker.get("ambiguous") is not True:
            issues.append(f"{prefix}.witness_fusion_multi_speaker_ambiguous_speaker_not_marked")

    accepted_channel = async_smoke.get("witness_fusion_multi_speaker_accepted_channel")
    wrong_channel = async_smoke.get("witness_fusion_multi_speaker_wrong_witness_channel")
    if not isinstance(accepted_channel, Mapping) or not isinstance(wrong_channel, Mapping):
        issues.append(f"{prefix}.witness_fusion_multi_speaker_channel_metadata_missing")
    elif "wrong_channel" in reasons:
        accepted_channel_id = str(accepted_channel.get("channel_id") or "").strip()
        wrong_channel_id = str(wrong_channel.get("channel_id") or "").strip()
        if not accepted_channel_id or not wrong_channel_id or accepted_channel_id == wrong_channel_id:
            issues.append(f"{prefix}.witness_fusion_multi_speaker_wrong_channel_not_proven")

    promoted_text = str(async_smoke.get("witness_fusion_multi_speaker_promoted_text") or "")
    if wrong_witness_text and wrong_witness_text in promoted_text:
        issues.append(f"{prefix}.witness_fusion_multi_speaker_wrong_witness_in_promoted_text")

    sink_values = async_smoke.get("witness_assisted_voiceops_action_sink_values")
    if wrong_witness_text and isinstance(sink_values, Mapping):
        for sink_key, sink_value in sink_values.items():
            for nested_text in _iter_nested_strings(sink_value):
                if wrong_witness_text in nested_text:
                    issues.append(
                        f"{prefix}.witness_fusion_multi_speaker_wrong_witness_in_action_sink:{sink_key}"
                    )
                    break


def _audit_voice_operator_proof_consistency(*, readiness: Mapping[str, Any], issues: list[str]) -> None:
    proofs = readiness.get("proofs") if isinstance(readiness.get("proofs"), Mapping) else {}
    smoke = readiness.get("smoke") if isinstance(readiness.get("smoke"), Mapping) else {}
    async_smoke = readiness.get("async_oracle_smoke") if isinstance(readiness.get("async_oracle_smoke"), Mapping) else {}
    _audit_promoted_request_summary_contract(
        async_smoke.get("external_frontend_promoted_request_summary"),
        label="voice_operator_readiness:async_oracle_smoke.external_frontend_promoted_request_summary",
        issues=issues,
    )
    _audit_provisional_request_summary_contract(
        async_smoke.get("external_frontend_provisional_request_summary"),
        label="voice_operator_readiness:async_oracle_smoke.external_frontend_provisional_request_summary",
        issues=issues,
    )
    _audit_provisional_request_summary_contract(
        async_smoke.get("external_frontend_status_provisional_request_summary"),
        label="voice_operator_readiness:async_oracle_smoke.external_frontend_status_provisional_request_summary",
        issues=issues,
    )
    async_unpromoted_sink_values = async_smoke.get("unpromoted_hypothesis_action_sink_values")
    if not isinstance(async_unpromoted_sink_values, Mapping):
        issues.append("voice_operator_readiness:async_oracle_smoke.unpromoted_hypothesis_action_sink_values_not_object")
        async_unpromoted_sink_values = {}
    elif async_unpromoted_sink_values:
        issues.append("voice_operator_readiness:async_oracle_smoke.unpromoted_hypothesis_action_sink_values_not_empty")
    _audit_witness_assisted_action_sinks(async_smoke=async_smoke, issues=issues)
    _audit_provider_text_alias_normalization(
        async_smoke=async_smoke,
        prefix="voice_operator_readiness:async_oracle_smoke",
        issues=issues,
    )
    _audit_kame_latency_breakdown(
        async_smoke=async_smoke,
        prefix="voice_operator_readiness:async_oracle_smoke",
        issues=issues,
    )
    _audit_reflex_ack_transcript(
        async_smoke=async_smoke,
        prefix="voice_operator_readiness:async_oracle_smoke",
        issues=issues,
    )
    _audit_minimum_interpreter_packet(
        async_smoke=async_smoke,
        prefix="voice_operator_readiness:async_oracle_smoke",
        issues=issues,
    )
    _audit_asr_optional_normal_path(
        async_smoke=async_smoke,
        prefix="voice_operator_readiness:async_oracle_smoke",
        issues=issues,
    )
    _audit_witness_fusion_multi_speaker_binding(async_smoke=async_smoke, issues=issues)
    external_frontend_transcript_hypotheses = async_smoke.get(
        "external_frontend_transcript_hypotheses"
    )
    if (
        not isinstance(external_frontend_transcript_hypotheses, list)
        or not external_frontend_transcript_hypotheses
    ):
        if external_frontend_transcript_hypotheses is None:
            issues.append(
                "voice_operator_readiness:async_oracle_smoke.external_frontend_transcript_hypotheses_missing"
            )
        elif not isinstance(external_frontend_transcript_hypotheses, list):
            issues.append(
                "voice_operator_readiness:async_oracle_smoke.external_frontend_transcript_hypotheses_not_list"
            )
        else:
            issues.append(
                "voice_operator_readiness:async_oracle_smoke.external_frontend_transcript_hypotheses_empty"
            )
    else:
        _audit_external_frontend_hypothesis_adjudication(
            external_frontend_transcript_hypotheses,
            issues=issues,
            label="async_oracle_smoke.external_frontend_transcript_hypotheses",
        )
    legacy_external_frontend_hypotheses = async_smoke.get(
        "external_frontend_auxiliary_transcript_hypotheses"
    )
    if (
        isinstance(legacy_external_frontend_hypotheses, list)
        and legacy_external_frontend_hypotheses
        and external_frontend_transcript_hypotheses != legacy_external_frontend_hypotheses
    ):
        issues.append(
            "voice_operator_readiness:async_oracle_smoke.external_frontend_transcript_hypotheses_auxiliary_mismatch"
        )
    if async_smoke.get("external_frontend_mode") != "witness_assisted_direct_audio":
        issues.append("voice_operator_readiness:async_oracle_smoke.external_frontend_mode_mismatch")
    if async_smoke.get("external_frontend_interpreter_profile") != "witness_assisted_direct_audio":
        issues.append(
            "voice_operator_readiness:async_oracle_smoke.external_frontend_interpreter_profile_mismatch"
        )
    if async_smoke.get("external_frontend_interpreter_input_order") != [
        "raw_audio",
        "metadata",
        "reflex",
        "transcript_hypotheses",
    ]:
        issues.append(
            "voice_operator_readiness:async_oracle_smoke.external_frontend_interpreter_input_order_mismatch"
        )
    if async_smoke.get("external_frontend_witness_direct_audio_profile_ok") is not True:
        issues.append(
            "voice_operator_readiness:async_oracle_smoke.external_frontend_witness_direct_audio_profile_not_ok"
        )
    witness_adjudications = async_smoke.get("external_frontend_witness_adjudications")
    if not isinstance(witness_adjudications, list) or not witness_adjudications:
        issues.append(
            "voice_operator_readiness:async_oracle_smoke.external_frontend_witness_adjudications_missing"
        )
    elif (
        isinstance(external_frontend_transcript_hypotheses, list)
        and external_frontend_transcript_hypotheses
        and witness_adjudications
        != _canonical_witness_adjudication_rows(external_frontend_transcript_hypotheses)
    ):
        issues.append(
            "voice_operator_readiness:async_oracle_smoke.external_frontend_witness_adjudications_mismatch"
        )
    interpreter_promoted = async_smoke.get("external_frontend_interpreter_promoted")
    if not isinstance(interpreter_promoted, Mapping):
        issues.append(
            "voice_operator_readiness:async_oracle_smoke.external_frontend_interpreter_promoted_missing"
        )
    elif interpreter_promoted.get("authority") != "interpreter_promoted":
        issues.append(
            "voice_operator_readiness:async_oracle_smoke.external_frontend_interpreter_promoted_authority_mismatch"
        )
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
            "energy_gate_low_energy_witness_adjudication": async_smoke.get(
                "energy_gate_low_energy_witness_adjudication"
            ),
            "energy_gate_low_energy_witness_rejection_reasons": async_smoke.get(
                "energy_gate_low_energy_witness_rejection_reasons"
            )
            or [],
            "energy_gate_low_energy_witness_authority": async_smoke.get(
                "energy_gate_low_energy_witness_authority"
            ),
            "energy_gate_low_energy_witness_tool_authority": async_smoke.get(
                "energy_gate_low_energy_witness_tool_authority"
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
    if "external_frontend_transcript_hypotheses" not in async_proof:
        issues.append(
            "voice_operator_readiness:proofs.async_oracle_jobs.external_frontend_transcript_hypotheses_missing"
        )
    elif isinstance(async_proof.get("external_frontend_transcript_hypotheses"), list):
        _audit_external_frontend_hypothesis_adjudication(
            async_proof["external_frontend_transcript_hypotheses"],
            issues=issues,
            label="proofs.async_oracle_jobs.external_frontend_transcript_hypotheses",
        )
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
            "status_ordinal_labels_visible": bool(async_smoke.get("status_ordinal_labels_visible")),
            "status_ordinal_labels": list(async_smoke.get("status_ordinal_labels") or []),
            "status_bounded_overflow_visible": bool(
                async_smoke.get("reflex_status_overflow_smoke_ok")
            ),
            "status_bounded_overflow_visible_job_count": async_smoke.get(
                "reflex_status_overflow_visible_job_count"
            ),
            "status_bounded_overflow_hidden_job_count": async_smoke.get(
                "reflex_status_overflow_hidden_job_count"
            ),
            "status_bounded_overflow_more_spoken_status": async_smoke.get(
                "reflex_status_overflow_more_spoken_status"
            ),
            "status_bounded_overflow_last_visible_ordinal": async_smoke.get(
                "reflex_status_overflow_last_visible_ordinal"
            ),
            "status_bounded_overflow_last_visible_label": async_smoke.get(
                "reflex_status_overflow_last_visible_label"
            ),
            "status_bounded_overflow_hidden_ids_absent": bool(
                async_smoke.get("reflex_status_overflow_hidden_ids_absent")
            ),
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
            "approval_tool_progress_kame_gate_present": bool(
                async_smoke.get("approval_tool_progress_kame_gate_present")
            ),
            "approval_tool_progress_kame_gate_schema_version": async_smoke.get(
                "approval_tool_progress_kame_gate_schema_version"
            ),
            "approval_tool_progress_kame_gate_failed_closed": bool(
                async_smoke.get("approval_tool_progress_kame_gate_failed_closed")
            ),
            "approval_tool_progress_kame_gate_issues": async_smoke.get(
                "approval_tool_progress_kame_gate_issues"
            )
            or [],
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
            "unflagged_high_risk_tool_cases": async_smoke.get("unflagged_high_risk_tool_cases") or [],
            "unflagged_high_risk_tool_case_count": async_smoke.get("unflagged_high_risk_tool_case_count"),
            "unflagged_high_risk_tool_categories": async_smoke.get("unflagged_high_risk_tool_categories") or [],
            "unflagged_high_risk_tool_names": async_smoke.get("unflagged_high_risk_tool_names") or [],
            "unflagged_high_risk_tool_all_cases_failed_closed": bool(
                async_smoke.get("unflagged_high_risk_tool_all_cases_failed_closed")
            ),
            "unflagged_high_risk_tool_all_progress_suppressed": bool(
                async_smoke.get("unflagged_high_risk_tool_all_progress_suppressed")
            ),
            "unflagged_high_risk_tool_all_payloads_redacted": bool(
                async_smoke.get("unflagged_high_risk_tool_all_payloads_redacted")
            ),
            "unflagged_high_risk_tool_all_spoken_payloads_clean": bool(
                async_smoke.get("unflagged_high_risk_tool_all_spoken_payloads_clean")
            ),
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
            "unflagged_high_risk_tool_name": async_smoke.get("unflagged_high_risk_tool_name"),
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
            "external_frontend_mode": async_smoke.get("external_frontend_mode"),
            "external_frontend_interpreter_profile": async_smoke.get(
                "external_frontend_interpreter_profile"
            ),
            "external_frontend_interpreter_input_order": async_smoke.get(
                "external_frontend_interpreter_input_order"
            )
            or [],
            "external_frontend_witness_direct_audio_profile_ok": bool(
                async_smoke.get("external_frontend_witness_direct_audio_profile_ok")
            ),
            "external_frontend_witness_adjudications": async_smoke.get(
                "external_frontend_witness_adjudications"
            )
            or [],
            "external_frontend_interpreter_promoted": dict(
                async_smoke.get("external_frontend_interpreter_promoted") or {}
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
            "external_frontend_promoted_request_summary": async_smoke.get(
                "external_frontend_promoted_request_summary"
            )
            or {},
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
            "external_frontend_transcript_hypotheses": async_smoke.get(
                "external_frontend_transcript_hypotheses"
            )
            or [],
            "external_frontend_auxiliary_transcript_hypotheses": async_smoke.get(
                "external_frontend_auxiliary_transcript_hypotheses"
            )
            or [],
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
            "external_frontend_witness_role_context": bool(
                async_smoke.get("external_frontend_witness_role_context")
            ),
            "external_frontend_witness_promotion_required": bool(
                async_smoke.get("external_frontend_witness_promotion_required")
            ),
            "external_frontend_direct_tool_authority_exposed": bool(
                async_smoke.get("external_frontend_direct_tool_authority_exposed")
            ),
            "external_frontend_direct_tool_rejected": bool(
                async_smoke.get("external_frontend_direct_tool_rejected")
            ),
            "external_frontend_direct_tool_rejected_tool": async_smoke.get(
                "external_frontend_direct_tool_rejected_tool"
            ),
            "external_frontend_direct_tool_rejection_reason": async_smoke.get(
                "external_frontend_direct_tool_rejection_reason"
            ),
            "external_frontend_direct_tool_created_oracle_job": bool(
                async_smoke.get("external_frontend_direct_tool_created_oracle_job")
            ),
            "external_frontend_tool_result_payload_safe": bool(
                async_smoke.get("external_frontend_tool_result_payload_safe")
            ),
            "external_frontend_reflex_status_payload_safe": bool(
                async_smoke.get("external_frontend_reflex_status_payload_safe")
            ),
            "external_frontend_placeholder_payload_safe": bool(
                async_smoke.get("external_frontend_placeholder_payload_safe")
            ),
            "external_frontend_tool_result_forbidden_paths": async_smoke.get(
                "external_frontend_tool_result_forbidden_paths"
            ),
            "external_frontend_reflex_status_forbidden_paths": async_smoke.get(
                "external_frontend_reflex_status_forbidden_paths"
            ),
            "external_frontend_placeholder": async_smoke.get("external_frontend_placeholder"),
            "external_frontend_placeholder_forbidden_paths": async_smoke.get(
                "external_frontend_placeholder_forbidden_paths"
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
            "unpromoted_hypothesis_action_sink_values": dict(async_unpromoted_sink_values),
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
            "unpromoted_hypothesis_not_durable_history": bool(
                async_smoke.get("unpromoted_hypothesis_not_durable_history")
            ),
            "witness_assisted_voiceops_action_smoke_ok": bool(
                async_smoke.get("witness_assisted_voiceops_action_smoke_ok")
            ),
            "witness_assisted_voiceops_action_gate_ok": bool(
                async_smoke.get("witness_assisted_voiceops_action_gate_ok")
            ),
            "witness_assisted_voiceops_action_gate_authorities": async_smoke.get(
                "witness_assisted_voiceops_action_gate_authorities"
            )
            or [],
            "witness_assisted_voiceops_action_consumed_before_action": bool(
                async_smoke.get("witness_assisted_voiceops_action_consumed_before_action")
            ),
            "witness_assisted_voiceops_action_single_bundle": bool(
                async_smoke.get("witness_assisted_voiceops_action_single_bundle")
            ),
            "witness_assisted_voiceops_action_witness_text": async_smoke.get(
                "witness_assisted_voiceops_action_witness_text"
            ),
            "witness_assisted_voiceops_action_promoted_text": async_smoke.get(
                "witness_assisted_voiceops_action_promoted_text"
            ),
            "witness_assisted_voiceops_action_witness_authority": async_smoke.get(
                "witness_assisted_voiceops_action_witness_authority"
            ),
            "witness_assisted_voiceops_action_witness_role_context": bool(
                async_smoke.get("witness_assisted_voiceops_action_witness_role_context")
            ),
            "witness_assisted_voiceops_action_witness_tool_authority_false": bool(
                async_smoke.get("witness_assisted_voiceops_action_witness_tool_authority_false")
            ),
            "witness_assisted_voiceops_action_witness_adjudication": async_smoke.get(
                "witness_assisted_voiceops_action_witness_adjudication"
            ),
            "witness_assisted_voiceops_action_promoted_authorities": async_smoke.get(
                "witness_assisted_voiceops_action_promoted_authorities"
            )
            or [],
            "witness_assisted_voiceops_action_sink_keys_checked": async_smoke.get(
                "witness_assisted_voiceops_action_sink_keys_checked"
            )
            or [],
            "witness_assisted_voiceops_action_sinks_clean": bool(
                async_smoke.get("witness_assisted_voiceops_action_sinks_clean")
            ),
            "witness_assisted_voiceops_action_sink_values": dict(
                async_smoke.get("witness_assisted_voiceops_action_sink_values") or {}
            ),
            "witness_assisted_voiceops_action_raw_witness_absent": bool(
                async_smoke.get("witness_assisted_voiceops_action_raw_witness_absent")
            ),
            "witness_assisted_voiceops_action_promoted_text_present": bool(
                async_smoke.get("witness_assisted_voiceops_action_promoted_text_present")
            ),
            "witness_fusion_timing_smoke_ok": bool(async_smoke.get("witness_fusion_timing_smoke_ok")),
            "witness_fusion_arrival_phases": async_smoke.get("witness_fusion_arrival_phases") or [],
            "witness_arrival_phase": async_smoke.get("witness_fusion_arrival_phases") or [],
            "witness_fusion_case_job_ids": async_smoke.get("witness_fusion_case_job_ids") or {},
            "witness_fusion_turn_ids": async_smoke.get("witness_fusion_turn_ids") or {},
            "witness_fusion_audio_segment_refs": async_smoke.get("witness_fusion_audio_segment_refs") or {},
            "witness_fusion_evidence_merge_keys": async_smoke.get("witness_fusion_evidence_merge_keys") or {},
            "witness_fusion_merge_key_observed": bool(async_smoke.get("witness_fusion_merge_key_observed")),
            "witness_fusion_same_turn_convergence_ok": bool(
                async_smoke.get("witness_fusion_same_turn_convergence_ok")
            ),
            "witness_fusion_same_turn_arrival_phases": async_smoke.get(
                "witness_fusion_same_turn_arrival_phases"
            )
            or [],
            "witness_fusion_same_turn_lineage": async_smoke.get("witness_fusion_same_turn_lineage") or {},
            "witness_fusion_same_turn_phase_lineage": async_smoke.get(
                "witness_fusion_same_turn_phase_lineage"
            )
            or {},
            "witness_fusion_same_turn_bundle_ids_by_phase": async_smoke.get(
                "witness_fusion_same_turn_bundle_ids_by_phase"
            )
            or {},
            "witness_fusion_same_turn_job_ids_by_phase": async_smoke.get(
                "witness_fusion_same_turn_job_ids_by_phase"
            )
            or {},
            "witness_fusion_same_turn_single_bundle": bool(
                async_smoke.get("witness_fusion_same_turn_single_bundle")
            ),
            "witness_fusion_same_turn_one_oracle_job": bool(
                async_smoke.get("witness_fusion_same_turn_one_oracle_job")
            ),
            "witness_fusion_same_turn_oracle_job_counts": async_smoke.get(
                "witness_fusion_same_turn_oracle_job_counts"
            )
            or {},
            "witness_fusion_same_turn_no_duplicate_oracle_job": bool(
                async_smoke.get("witness_fusion_same_turn_no_duplicate_oracle_job")
            ),
            "witness_fusion_same_turn_expected_merge_key": async_smoke.get(
                "witness_fusion_same_turn_expected_merge_key"
            )
            or "",
            "witness_fusion_multi_speaker_witness_smoke_ok": bool(
                async_smoke.get("witness_fusion_multi_speaker_witness_smoke_ok")
            ),
            "witness_fusion_multi_speaker_wrong_witness_text": str(
                async_smoke.get("witness_fusion_multi_speaker_wrong_witness_text") or ""
            ),
            "witness_fusion_multi_speaker_wrong_witness": dict(
                async_smoke.get("witness_fusion_multi_speaker_wrong_witness") or {}
            ),
            "witness_fusion_multi_speaker_wrong_witness_rejected": bool(
                async_smoke.get("witness_fusion_multi_speaker_wrong_witness_rejected")
            ),
            "witness_fusion_multi_speaker_wrong_witness_speaker": dict(
                async_smoke.get("witness_fusion_multi_speaker_wrong_witness_speaker") or {}
            ),
            "witness_fusion_multi_speaker_wrong_witness_channel": dict(
                async_smoke.get("witness_fusion_multi_speaker_wrong_witness_channel") or {}
            ),
            "witness_fusion_multi_speaker_accepted_speaker": dict(
                async_smoke.get("witness_fusion_multi_speaker_accepted_speaker") or {}
            ),
            "witness_fusion_multi_speaker_accepted_channel": dict(
                async_smoke.get("witness_fusion_multi_speaker_accepted_channel") or {}
            ),
            "witness_fusion_multi_speaker_bound_to_second_human": bool(
                async_smoke.get("witness_fusion_multi_speaker_bound_to_second_human")
            ),
            "witness_fusion_multi_speaker_action_sinks_clean": bool(
                async_smoke.get("witness_fusion_multi_speaker_action_sinks_clean")
            ),
            "witness_fusion_multi_speaker_promoted_text": str(
                async_smoke.get("witness_fusion_multi_speaker_promoted_text") or ""
            ),
            "witness_fusion_audio_metadata": async_smoke.get("witness_fusion_audio_metadata") or {},
            "witness_fusion_bundle_audio_metadata": async_smoke.get(
                "witness_fusion_bundle_audio_metadata"
            )
            or {},
            "witness_fusion_accepted_audio_gate_observed": bool(
                async_smoke.get("witness_fusion_accepted_audio_gate_observed")
            ),
            "raw_audio_interpreter_evidence_observed": bool(
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
            "interpreter_adjudication_outcomes": async_smoke.get("witness_fusion_adjudications") or {},
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
            "energy_gate_low_energy_witness_adjudication": async_smoke.get(
                "energy_gate_low_energy_witness_adjudication"
            ),
            "energy_gate_low_energy_witness_rejection_reasons": async_smoke.get(
                "energy_gate_low_energy_witness_rejection_reasons"
            )
            or [],
            "energy_gate_low_energy_witness_authority": async_smoke.get(
                "energy_gate_low_energy_witness_authority"
            ),
            "energy_gate_low_energy_witness_tool_authority": async_smoke.get(
                "energy_gate_low_energy_witness_tool_authority"
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
            "transcript_only_witness_rejected_for_full_kame": (
                async_smoke.get("runtime_kame_action_gate_degraded_text_only_ok") is False
                and async_smoke.get("runtime_kame_action_gate_degraded_text_only_status")
                == "degraded_text_only"
                and async_smoke.get("runtime_kame_action_gate_degraded_text_only_raw_audio_available")
                is False
                and bool(async_smoke.get("runtime_kame_action_gate_degraded_text_only_preserves_hypothesis"))
            ),
            "runtime_kame_action_gate_degraded_oracle_promoted_ok": async_smoke.get(
                "runtime_kame_action_gate_degraded_oracle_promoted_ok"
            ),
            "runtime_kame_action_gate_degraded_oracle_promoted_issues": async_smoke.get(
                "runtime_kame_action_gate_degraded_oracle_promoted_issues"
            ),
            "runtime_kame_action_gate_degraded_oracle_promoted_authorities": async_smoke.get(
                "runtime_kame_action_gate_degraded_oracle_promoted_authorities"
            ),
            "runtime_kame_action_gate_degraded_oracle_promoted_rejected_authorities": async_smoke.get(
                "runtime_kame_action_gate_degraded_oracle_promoted_rejected_authorities"
            ),
            "runtime_kame_action_gate_degraded_oracle_promoted_status": async_smoke.get(
                "runtime_kame_action_gate_degraded_oracle_promoted_status"
            ),
            "runtime_kame_action_gate_degraded_oracle_promoted_raw_audio_available": async_smoke.get(
                "runtime_kame_action_gate_degraded_oracle_promoted_raw_audio_available"
            ),
            "runtime_kame_action_gate_degraded_oracle_promoted_consumed_before_action": bool(
                async_smoke.get("runtime_kame_action_gate_degraded_oracle_promoted_consumed_before_action")
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
            "durable_resume_contract_smoke_ok": bool(
                async_smoke.get("durable_resume_contract_smoke_ok")
            ),
            "durable_resume_contract_schema_version": async_smoke.get(
                "durable_resume_contract_schema_version"
            ),
            "durable_resume_promoted_turn_count": async_smoke.get(
                "durable_resume_promoted_turn_count"
            ),
            "durable_resume_recent_promoted_turns_verbatim": bool(
                async_smoke.get("durable_resume_recent_promoted_turns_verbatim")
            ),
            "durable_resume_recent_promoted_turns": async_smoke.get(
                "durable_resume_recent_promoted_turns"
            )
            or [],
            "durable_resume_older_turns_summarized": bool(
                async_smoke.get("durable_resume_older_turns_summarized")
            ),
            "durable_resume_older_promoted_turn_count": async_smoke.get(
                "durable_resume_older_promoted_turn_count"
            ),
            "durable_resume_older_promoted_turn_summary": async_smoke.get(
                "durable_resume_older_promoted_turn_summary"
            ),
            "durable_resume_hypothesis_replay_absent": bool(
                async_smoke.get("durable_resume_hypothesis_replay_absent")
            ),
            "durable_resume_ledger_authoritative": bool(
                async_smoke.get("durable_resume_ledger_authoritative")
            ),
            "hypothesis_final_durable_message_smoke_ok": bool(
                async_smoke.get("hypothesis_final_durable_message_smoke_ok")
            ),
            "hypothesis_final_durable_messages_empty": bool(
                async_smoke.get("hypothesis_final_durable_messages_empty")
            ),
            "hypothesis_final_durable_message_count": async_smoke.get(
                "hypothesis_final_durable_message_count"
            ),
            "hypothesis_final_without_adapter_flag_non_durable": bool(
                async_smoke.get("hypothesis_final_without_adapter_flag_non_durable")
            ),
            "hypothesis_final_witness_intent_non_durable": bool(
                async_smoke.get("hypothesis_final_witness_intent_non_durable")
            ),
            "explicit_asr_fallback_final_remains_durable": bool(
                async_smoke.get("explicit_asr_fallback_final_remains_durable")
            ),
            "explicit_asr_fallback_durable_messages": async_smoke.get(
                "explicit_asr_fallback_durable_messages"
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
            "missing_transcript_hypothesis_fields": "Required transcript hypothesis fields",
            "missing_raw_witness_text_ban": "raw witness text is not allowed as outbound payload content",
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
            "missing_transcript_hypothesis_fields": "Required transcript hypothesis fields",
            "missing_raw_witness_text_ban": "Raw witness text is not allowed in channel egress",
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
    phone_context = _read_json(demo_dir / "phone-context.json", issues, "phone_context")
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
    tool_disclosure_smoke = _read_json(
        voice_dir / "tool-disclosure-smoke.json",
        issues,
        "tool_disclosure_smoke",
    )
    ephemeral_tool_router_smoke = _read_json(
        voice_dir / "ephemeral-tool-router-smoke.json",
        issues,
        "ephemeral_tool_router_smoke",
    )
    interpreter_request_packet = _read_json(
        voice_dir / "interpreter-request-packet.json",
        issues,
        "interpreter_request_packet",
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
    channel_review_path = channel_dir / "channel-policy-review.json"
    channel_review = _read_json(channel_review_path, issues, "channel_policy_review")
    channel_review_decision_scaffold = _read_json(
        channel_dir / "channel-policy-review-decision.json",
        issues,
        "channel_policy_review_decision_scaffold",
    )
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
    _audit_phone_context_contract(
        demo=demo,
        phone_context=phone_context,
        packet=packet,
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
    _audit_channel_policy(
        channel_policy,
        channel_review,
        channel_review_decision_scaffold,
        review_path=channel_review_path,
        issues=issues,
    )
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
        plan_run=plan_run,
        discord_loopback_smoke=discord_loopback_smoke,
        async_oracle_smoke=async_oracle_smoke,
        discord_session_cleanup_smoke=discord_session_cleanup_smoke,
        sidecar_fail_closed_smoke=sidecar_fail_closed_smoke,
        tool_disclosure_smoke=tool_disclosure_smoke,
        ephemeral_tool_router_smoke=ephemeral_tool_router_smoke,
        interpreter_request_packet=interpreter_request_packet,
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
    _audit_spark_matrix_kame_contract(spark_matrix, issues)
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
