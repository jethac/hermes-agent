from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path
from typing import Sequence

import pytest

from scripts.voiceops_provisioning_probe import (
    CommandResult,
    PREFLIGHT_EVIDENCE_REQUIRED_DOT_PATHS,
    build_preflight_evidence_example,
    build_preflight_evidence_manifest_example,
    build_preflight_evidence_template,
    build_milestone2_execution_plan,
    build_post_approval_receipts_example,
    build_post_approval_receipts_template,
    build_probe_report,
    load_post_approval_receipts,
    load_preflight_evidence,
    parse_args,
    refresh_preflight_source_hashes,
    validate_post_approval_receipts,
    write_preflight_evidence_scaffold,
    write_probe_artifacts,
    _validate_safe_probe_command,
    _validate_readonly_discovery_command,
)


def _dot_get(payload, ref):
    cursor = payload
    for part in ref.split("."):
        cursor = cursor[part]
    return cursor


def _complete_preflight_evidence() -> dict[str, object]:
    evidence = build_preflight_evidence_template()
    evidence["stripe_projects"].update(
        {
            "account_ref": "stripe-account-ref-demo",
            "projects_catalog_checked_at": "2026-06-29T00:00:00Z",
            "can_create_project_after_approval": True,
        }
    )
    evidence["stripe_link"].update(
        {
            "account_ref": "stripe-link-account-ref-demo",
            "approval_capability_confirmed": True,
            "max_approved_cents": 20_000,
        }
    )
    evidence["mpp"].update({"boundary_tool": "nemoclaw", "policy_ref": "voiceops-policy-demo"})
    evidence["phone_handoff"].update(
        {
            "provider": "twilio",
            "provider_account_ref": "twilio-account-ref-demo",
            "phone_target_ref": "phone-target-ref-demo",
            "credential_location_ref": "keychain-ref-demo",
        }
    )
    evidence["rollback"].update(
        {
            "deprovision_owner": "operator",
            "refund_or_cancel_owner": "operator",
            "call_cancel_owner": "operator",
        }
    )
    return evidence


def _write_preflight_evidence(tmp_path: Path, payload: dict[str, object] | None = None) -> Path:
    payload = payload or _complete_preflight_evidence()
    for section_name in ("stripe_projects", "stripe_link", "mpp", "phone_handoff", "rollback"):
        section = payload.get(section_name)
        if not isinstance(section, dict):
            continue
        source_path = tmp_path / f"{section_name}-source.json"
        source_payload = {
            "schema_version": "voiceops.milestone2.redacted_source_artifact.v1",
            "section": section_name,
            "redacted": True,
            "redaction_policy": "references only; no raw secrets, tokens, or full phone numbers",
        }
        source_bytes = json.dumps(source_payload, sort_keys=True).encode("utf-8")
        source_path.write_bytes(source_bytes)
        if not section.get("source_artifact"):
            section["source_artifact"] = source_path.name
        if not section.get("source_artifact_kind"):
            section["source_artifact_kind"] = "redacted_setup_evidence"
        if not section.get("source_artifact_sha256"):
            section["source_artifact_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        if not section.get("source_artifact_redacted_at"):
            section["source_artifact_redacted_at"] = "2026-06-29T00:00:00Z"
    evidence_path = tmp_path / "preflight-evidence.json"
    evidence_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return evidence_path


def test_probe_passes_with_safe_local_tools_and_redacts_outputs(tmp_path):
    calls: list[list[str]] = []

    def fake_which(command: str) -> str | None:
        paths = {
            "stripe": "/usr/local/bin/stripe",
            "link-cli": "/usr/local/bin/link-cli",
            "mppx": "/usr/local/bin/mppx",
            "twilio": "/usr/local/bin/twilio",
        }
        return paths.get(command)

    def fake_runner(argv: Sequence[str], _timeout_seconds: int) -> CommandResult:
        calls.append(list(argv))
        return CommandResult(
            exit_code=0,
            stdout="ok STRIPE_SECRET_KEY=sk_live_123456789abcdef phone +15551234567",
            stderr="Bearer token_123456789abcdef",
        )

    report = build_probe_report(
        env={
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "TWILIO_ACCOUNT_SID": "AC123456789abcdef",
            "TWILIO_AUTH_TOKEN": "secret-token",
        },
        env_files=[],
        preflight_evidence_path=_write_preflight_evidence(tmp_path),
        which=fake_which,
        runner=fake_runner,
        run_commands=True,
    )

    assert report["status"] == "ready"
    assert report["ready"] is True
    assert report["required_failures"] == []
    assert report["preflight_evidence_loaded"] is True
    assert report["preflight_evidence_missing_fields"] == []
    assert calls == [
        ["stripe", "--version"],
        ["stripe", "projects", "--help"],
        ["link-cli", "--version"],
        ["mppx", "--version"],
        ["twilio", "--version"],
    ]
    assert all(any(arg in {"--version", "--help"} for arg in call[1:]) for call in calls)
    joined_calls = " ".join(" ".join(call) for call in calls)
    for forbidden in ["projects add", "spend-request create", "provision", "call create", "credential"]:
        assert forbidden not in joined_calls

    serialized = json.dumps(report)
    assert "sk_live_123456789abcdef" not in serialized
    assert "+15551234567" not in serialized
    assert "secret-token" not in serialized
    assert "<redacted" in serialized


def test_readonly_discovery_runs_exact_allowlist_and_does_not_grant_readiness(tmp_path):
    calls: list[list[str]] = []

    def fake_which(command: str) -> str | None:
        return f"/usr/local/bin/{command}" if command in {"stripe", "link-cli", "mppx"} else None

    def fake_readonly_runner(argv: Sequence[str], _timeout_seconds: int) -> CommandResult:
        calls.append(list(argv))
        return CommandResult(
            exit_code=0,
            stdout="catalog twilio sk_live_123456789abcdef phone +15551234567",
            stderr="Bearer token_123456789abcdef",
        )

    report = build_probe_report(
        env={"VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567", "TWILIO_ACCOUNT_SID": "AC123"},
        env_files=[],
        which=fake_which,
        runner=lambda _argv, _timeout_seconds: CommandResult(exit_code=0),
        readonly_discovery_runner=fake_readonly_runner,
        run_readonly_discovery=True,
    )

    assert calls == [
        ["stripe", "projects", "list", "--limit", "10"],
        ["link-cli", "auth", "status"],
    ]
    assert report["read_only_discovery"]["run_requested"] is True
    assert report["read_only_discovery"]["does_not_grant_approval"] is True
    assert report["read_only_discovery"]["status"] == "pass"
    assert report["ready"] is False
    assert "stripe_projects_account" in report["required_failures"]
    serialized = json.dumps(report)
    assert "sk_live_123456789abcdef" not in serialized
    assert "+15551234567" not in serialized
    assert "token_123456789abcdef" not in serialized
    assert "<redacted" in serialized


def test_probe_reports_required_failures_without_running_missing_tools():
    calls: list[list[str]] = []

    report = build_probe_report(
        env={},
        env_files=[],
        which=lambda _command: None,
        runner=lambda argv, _timeout_seconds: calls.append(list(argv)) or CommandResult(exit_code=0),
    )

    assert report["status"] == "needs_setup"
    assert report["ready"] is False
    assert report["preflight_evidence_loaded"] is False
    assert report["preflight_evidence_missing_fields"] == PREFLIGHT_EVIDENCE_REQUIRED_DOT_PATHS
    assert set(report["required_failures"]) == {
        "credential_location_reference",
        "mpp_approval_boundary",
        "stripe_cli",
        "stripe_projects_cli",
        "stripe_projects_account",
        "stripe_link_cli",
        "stripe_link_approval_capability",
        "mpp_agent",
        "phone_target",
        "phone_provider",
        "phone_provider_account",
        "rollback_owner_refs",
    }
    assert calls == []


def test_probe_treats_no_command_probes_as_path_presence_only(tmp_path):
    def fake_which(command: str) -> str | None:
        return f"/bin/{command}" if command in {"stripe", "link-cli", "mppx"} else None

    report = build_probe_report(
        env={"VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567", "VAPI_API_KEY": "secret"},
        env_files=[],
        preflight_evidence_path=_write_preflight_evidence(tmp_path),
        which=fake_which,
        runner=lambda _argv, _timeout_seconds: (_ for _ in ()).throw(AssertionError("runner should not be called")),
    )

    assert report["ready"] is True
    assert all(probe["executed"] is False for probe in report["command_probes"])
    assert {probe["status"] for probe in report["command_probes"] if probe["found"]} == {"found"}


def test_cli_env_presence_alone_does_not_complete_real_preflight():
    def fake_which(command: str) -> str | None:
        return f"/bin/{command}" if command in {"stripe", "link-cli", "mppx"} else None

    report = build_probe_report(
        env={"VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567", "VAPI_API_KEY": "secret"},
        env_files=[],
        which=fake_which,
        runner=lambda _argv, _timeout_seconds: (_ for _ in ()).throw(AssertionError("runner should not be called")),
    )

    assert report["ready"] is False
    assert {"stripe_projects_account", "stripe_link_approval_capability", "mpp_approval_boundary"} <= set(
        report["required_failures"]
    )
    assert report["preflight_evidence"]["loaded"] is False
    assert "stripe_cli" not in report["required_failures"]


def test_write_probe_artifacts(tmp_path):
    report = build_probe_report(
        env={"VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567", "TWILIO_ACCOUNT_SID": "AC123"},
        env_files=[],
        which=lambda command: f"/bin/{command}" if command in {"stripe", "link-cli", "mppx"} else None,
        runner=lambda _argv, _timeout_seconds: CommandResult(exit_code=0, stdout="version 1.0"),
    )
    paths = write_probe_artifacts(tmp_path, report)

    required_paths = {
        "command_manifest",
        "execution_plan_json",
        "execution_plan_markdown",
        "json",
        "markdown",
        "post_approval_audit_ledger",
        "post_approval_receipts_example",
        "post_approval_receipts_scaffold",
        "post_approval_receipts_template",
        "post_approval_receipts_validation",
        "preflight_evidence_example",
        "preflight_evidence_manifest_example",
        "preflight_evidence_scaffold_manifest",
        "preflight_evidence_template",
        "read_only_discovery_audit_ledger",
        "read_only_discovery_json",
        "read_only_discovery_manifest",
        "read_only_discovery_markdown",
        "setup_closure_json",
        "setup_closure_markdown",
    }
    scaffold_sections = ("stripe_projects", "stripe_link", "mpp", "phone_handoff", "rollback")
    scaffold_paths = {
        f"scaffold_{section_name}_{kind}"
        for section_name in scaffold_sections
        for kind in ("section", "source")
    }
    assert required_paths <= set(paths)
    assert scaffold_paths <= set(paths)
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    execution_plan = json.loads(Path(paths["execution_plan_json"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(paths["command_manifest"]).read_text(encoding="utf-8"))
    discovery = json.loads(Path(paths["read_only_discovery_json"]).read_text(encoding="utf-8"))
    discovery_manifest = json.loads(Path(paths["read_only_discovery_manifest"]).read_text(encoding="utf-8"))
    setup_closure = json.loads(Path(paths["setup_closure_json"]).read_text(encoding="utf-8"))
    post_approval_template = json.loads(Path(paths["post_approval_receipts_template"]).read_text(encoding="utf-8"))
    post_approval_example = json.loads(Path(paths["post_approval_receipts_example"]).read_text(encoding="utf-8"))
    post_approval_scaffold = json.loads(Path(paths["post_approval_receipts_scaffold"]).read_text(encoding="utf-8"))
    post_approval_validation = json.loads(Path(paths["post_approval_receipts_validation"]).read_text(encoding="utf-8"))
    preflight_example = json.loads(Path(paths["preflight_evidence_example"]).read_text(encoding="utf-8"))
    preflight_manifest_example = json.loads(Path(paths["preflight_evidence_manifest_example"]).read_text(encoding="utf-8"))
    preflight_scaffold_manifest_path = Path(paths["preflight_evidence_scaffold_manifest"])
    preflight_scaffold_manifest = json.loads(preflight_scaffold_manifest_path.read_text(encoding="utf-8"))
    preflight_template = json.loads(Path(paths["preflight_evidence_template"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    execution_markdown = Path(paths["execution_plan_markdown"]).read_text(encoding="utf-8")
    setup_markdown = Path(paths["setup_closure_markdown"]).read_text(encoding="utf-8")
    assert payload["probe"]["non_mutating"] is True
    assert payload["preflight_evidence"]["loaded"] is False
    assert discovery["schema_version"] == "voiceops.milestone2.read_only_discovery.v1"
    assert discovery["run_requested"] is False
    assert discovery["does_not_grant_approval"] is True
    assert discovery_manifest["schema_version"] == "voiceops.milestone2.read_only_discovery_manifest.v1"
    assert discovery_manifest["audit_ledger"] == "audit-ledger.read-only-discovery.jsonl"
    assert Path(paths["read_only_discovery_audit_ledger"]).read_text(encoding="utf-8") == ""
    assert "read_only_discovery_commands" in manifest
    assert "version_help_commands" in manifest
    assert "VoiceOps Provisioning Readiness Probe" in markdown
    assert execution_plan["schema_version"] == "voiceops.milestone2.execution_plan.v1"
    assert "stripe_projects_account" in execution_plan["preflight"]["required_evidence"]
    assert "phone-context.json" in json.dumps(execution_plan)
    assert "VoiceOps Milestone 2 Execution Plan" in execution_markdown
    assert post_approval_template["schema_version"] == "voiceops.milestone2.post_approval_receipts.v1"
    assert post_approval_template["receipts"] == []
    assert post_approval_example["example_only"] is True
    assert post_approval_example["expected_actions"] == sorted(execution_plan["approval_contracts"])
    assert post_approval_scaffold["expected_actions"] == sorted(execution_plan["approval_contracts"])
    expected_action_ids = {action["action_id"] for action in execution_plan["approval_required_actions"]}
    credential_required_action_ids = {
        action["action_id"]
        for action in execution_plan["approval_required_actions"]
        if action["credential_location_required"]
    }
    assert {receipt["action_id"] for receipt in post_approval_example["receipts"]} == expected_action_ids
    assert {receipt["action_id"] for receipt in post_approval_scaffold["receipts"]} == expected_action_ids
    assert {event["action_id"] for event in post_approval_example["audit_events"]} == expected_action_ids
    assert {event["action_id"] for event in post_approval_scaffold["audit_events"]} == expected_action_ids
    assert {item["created_by_action_id"] for item in post_approval_example["credential_locations"]} == (
        credential_required_action_ids
    )
    assert {item["created_by_action_id"] for item in post_approval_scaffold["credential_locations"]} == (
        credential_required_action_ids
    )
    assert {item["rollback_ref"] for item in post_approval_example["rollback_receipts"]} == {
        action["rollback_ref"] for action in execution_plan["approval_required_actions"]
    }
    example_validation = validate_post_approval_receipts(post_approval_example, execution_plan)
    assert "example_only evidence is not accepted" in " ".join(example_validation["validation_issues"])
    assert not any("missing_receipts_for_actions" in issue for issue in example_validation["validation_issues"])
    assert not any("missing_credential_location" in issue for issue in example_validation["validation_issues"])
    assert not any("missing_rollback_receipt" in issue for issue in example_validation["validation_issues"])
    assert not any("missing_audit_event" in issue for issue in example_validation["validation_issues"])
    assert post_approval_scaffold["example_only"] is True
    assert post_approval_validation["status"] == "not_supplied"
    assert Path(paths["post_approval_audit_ledger"]).read_text(encoding="utf-8") == ""
    assert setup_closure["schema_version"] == "voiceops.milestone2.setup_closure.v1"
    assert setup_closure["preflight_evidence_template"] == "provisioning-preflight-evidence.template.json"
    assert setup_closure["preflight_evidence_example"] == "provisioning-preflight-evidence.example.json"
    assert setup_closure["preflight_evidence_manifest_example"] == "provisioning-preflight-evidence.manifest.example.json"
    assert (
        setup_closure["preflight_evidence_scaffold_manifest"]
        == "provisioning-preflight-scaffold/provisioning-preflight-evidence.manifest.json"
    )
    assert setup_closure["evidence_contract"]["preflight_schema_version"] == "voiceops.milestone2.preflight_evidence.v1"
    assert setup_closure["evidence_contract"]["read_only_discovery_grants_approval"] is False
    assert setup_closure["evidence_contract"]["required_section_field"] == "source_artifact"
    assert setup_closure["evidence_contract"]["source_artifacts_must_exist"] is True
    assert "provisioning-preflight-evidence.manifest.json" in setup_closure["rerun_commands"]["with_preflight_manifest"]
    assert "--run-readonly-discovery" in setup_closure["rerun_commands"]["read_only_discovery"]
    assert "--refresh-preflight-source-hashes" in setup_closure["rerun_commands"]["refresh_preflight_source_hashes"]
    assert (
        "provisioning-preflight-evidence.manifest.json"
        in setup_closure["rerun_commands"]["plan_index_manifest_and_post_approval_receipts"]
    )
    assert "--post-approval-receipts" in setup_closure["rerun_commands"][
        "plan_index_manifest_and_post_approval_receipts"
    ]
    assert setup_closure["rerun_commands"]["source_artifact_sha256"].startswith("shasum -a 256")
    assert "VoiceOps Milestone 2 Setup Closure Plan" in setup_markdown
    assert "Manifest example" in setup_markdown
    assert "Two-layer scaffold" in setup_markdown
    assert "source_artifact" in setup_markdown
    assert "`stripe_projects.account_ref`" in setup_markdown
    assert "`stripe_link.max_approved_cents`" in setup_markdown
    assert "`phone_handoff.credential_location_ref`" in setup_markdown
    assert "`rollback.deprovision_owner`" in setup_markdown
    assert preflight_example["example_only"] is True
    assert len(preflight_example["stripe_projects"]["source_artifact_sha256"]) == 64
    assert set(preflight_example["stripe_projects"]["source_artifact_sha256"]) == {"0"}
    assert preflight_manifest_example["example_only"] is True
    assert preflight_manifest_example["reports"]["stripe_projects"].endswith("stripe-projects-evidence.json")
    assert preflight_scaffold_manifest["example_only"] is True
    assert (
        preflight_scaffold_manifest["reports"]["stripe_projects"]
        == "sections/stripe-projects-evidence.json"
    )
    scaffold_issues = load_preflight_evidence(preflight_scaffold_manifest_path)["validation_issues"]
    assert "example_only evidence is not accepted" in scaffold_issues
    assert "stripe_projects: example_only evidence is not accepted" in scaffold_issues
    assert all("artifact not found" not in issue for issue in scaffold_issues)
    stripe_section_path = preflight_scaffold_manifest_path.parent / preflight_scaffold_manifest["reports"][
        "stripe_projects"
    ]
    stripe_section = json.loads(stripe_section_path.read_text(encoding="utf-8"))["stripe_projects"]
    stripe_source_path = stripe_section_path.parent / stripe_section["source_artifact"]
    stripe_source = json.loads(stripe_source_path.read_text(encoding="utf-8"))
    assert stripe_source["redacted"] is True
    assert hashlib.sha256(stripe_source_path.read_bytes()).hexdigest() == stripe_section["source_artifact_sha256"]
    assert "example_only evidence is not accepted" in load_preflight_evidence(
        Path(paths["preflight_evidence_example"])
    )["validation_issues"]
    assert preflight_template["schema_version"] == "voiceops.milestone2.preflight_evidence.v1"
    assert "projects add" in manifest["blocked_patterns"]
    assert "+15551234567" not in json.dumps(payload)


def test_preflight_evidence_example_is_not_accepted_as_proof(tmp_path):
    evidence_path = _write_preflight_evidence(tmp_path, build_preflight_evidence_example())

    report = build_probe_report(
        env={"VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567", "TWILIO_ACCOUNT_SID": "AC123"},
        env_files=[],
        preflight_evidence_path=evidence_path,
        which=lambda command: f"/bin/{command}" if command in {"stripe", "link-cli", "mppx"} else None,
        runner=lambda _argv, _timeout_seconds: CommandResult(exit_code=0),
    )

    assert report["ready"] is False
    assert report["preflight_evidence_loaded"] is True
    assert report["preflight_evidence_missing_fields"] == []
    assert "example_only evidence is not accepted" in report["preflight_evidence"]["validation_issues"]
    assert {
        "credential_location_reference",
        "mpp_approval_boundary",
        "phone_provider_account",
        "rollback_owner_refs",
        "stripe_link_approval_capability",
        "stripe_projects_account",
    } <= set(report["required_failures"])


def test_preflight_evidence_rejects_missing_or_invalid_schema(tmp_path):
    evidence = _complete_preflight_evidence()
    evidence.pop("schema_version")
    missing_schema_path = _write_preflight_evidence(tmp_path, evidence)

    missing_schema = load_preflight_evidence(missing_schema_path)

    assert missing_schema["missing_fields"] == []
    assert "missing_or_invalid_schema_version" in missing_schema["validation_issues"]

    evidence["schema_version"] = "wrong.schema.v1"
    invalid_schema_path = tmp_path / "invalid-schema-preflight.json"
    invalid_schema_path.write_text(json.dumps(evidence), encoding="utf-8")

    invalid_schema = load_preflight_evidence(invalid_schema_path)

    assert invalid_schema["missing_fields"] == []
    assert "missing_or_invalid_schema_version" in invalid_schema["validation_issues"]


def test_preflight_evidence_rejects_complete_shape_without_source_artifacts(tmp_path):
    evidence = _complete_preflight_evidence()
    evidence_path = tmp_path / "synthetic-preflight.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    loaded = load_preflight_evidence(evidence_path)
    report = build_probe_report(
        env={"VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567", "TWILIO_ACCOUNT_SID": "AC123"},
        env_files=[],
        preflight_evidence_path=evidence_path,
        which=lambda command: f"/bin/{command}" if command in {"stripe", "link-cli", "mppx"} else None,
        runner=lambda _argv, _timeout_seconds: CommandResult(exit_code=0),
    )

    assert "stripe_projects.source_artifact" in loaded["missing_fields"]
    assert "stripe_link.source_artifact" in loaded["missing_fields"]
    assert "mpp.source_artifact" in loaded["missing_fields"]
    assert "phone_handoff.source_artifact" in loaded["missing_fields"]
    assert "rollback.source_artifact" in loaded["missing_fields"]
    assert "stripe_projects.source_artifact: missing" in loaded["validation_issues"]
    assert report["ready"] is False
    assert {
        "credential_location_reference",
        "mpp_approval_boundary",
        "phone_provider_account",
        "rollback_owner_refs",
        "stripe_link_approval_capability",
        "stripe_projects_account",
    } <= set(report["required_failures"])


def test_preflight_evidence_rejects_synthetic_schema_valid_refs_without_provenance(tmp_path):
    evidence = _complete_preflight_evidence()
    for section_name in ("stripe_projects", "stripe_link", "mpp", "phone_handoff", "rollback"):
        source_path = tmp_path / f"{section_name}-redacted.json"
        source_path.write_text(json.dumps({"redacted": True, "section": section_name}), encoding="utf-8")
        evidence[section_name]["source_artifact"] = source_path.name

    evidence_path = tmp_path / "synthetic-refs-preflight.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    loaded = load_preflight_evidence(evidence_path)
    report = build_probe_report(
        env={"VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567", "TWILIO_ACCOUNT_SID": "AC123"},
        env_files=[],
        preflight_evidence_path=evidence_path,
        which=lambda command: f"/bin/{command}" if command in {"stripe", "link-cli", "mppx"} else None,
        runner=lambda _argv, _timeout_seconds: CommandResult(exit_code=0),
    )

    assert report["ready"] is False
    assert "stripe_projects.source_artifact_sha256" in loaded["missing_fields"]
    assert "stripe_projects.source_artifact_redacted_at" in loaded["missing_fields"]
    assert "stripe_projects.source_artifact_sha256: invalid" in loaded["validation_issues"]
    assert "stripe_projects_account" in report["required_failures"]


def test_preflight_evidence_rejects_source_artifact_without_schema_or_matching_section(tmp_path):
    evidence = _complete_preflight_evidence()
    stripe_source = tmp_path / "stripe-projects-redacted.json"
    stripe_source_bytes = json.dumps(
        {
            "redacted": True,
            "section": "stripe_projects",
            "redaction_policy": "references only; no raw secrets, tokens, or full phone numbers",
        },
        sort_keys=True,
    ).encode("utf-8")
    stripe_source.write_bytes(stripe_source_bytes)
    link_source = tmp_path / "stripe-link-redacted.json"
    link_source_bytes = json.dumps(
        {
            "schema_version": "voiceops.milestone2.redacted_source_artifact.v1",
            "section": "mpp",
            "redacted": True,
            "redaction_policy": "references only; no raw secrets, tokens, or full phone numbers",
        },
        sort_keys=True,
    ).encode("utf-8")
    link_source.write_bytes(link_source_bytes)
    evidence["stripe_projects"].update(
        {
            "source_artifact": stripe_source.name,
            "source_artifact_kind": "redacted_setup_evidence",
            "source_artifact_sha256": hashlib.sha256(stripe_source_bytes).hexdigest(),
            "source_artifact_redacted_at": "2026-06-29T00:00:00Z",
        }
    )
    evidence["stripe_link"].update(
        {
            "source_artifact": link_source.name,
            "source_artifact_kind": "redacted_setup_evidence",
            "source_artifact_sha256": hashlib.sha256(link_source_bytes).hexdigest(),
            "source_artifact_redacted_at": "2026-06-29T00:00:00Z",
        }
    )
    evidence_path = _write_preflight_evidence(tmp_path, evidence)

    loaded = load_preflight_evidence(evidence_path)

    assert "stripe_projects.source_artifact:missing_or_invalid_schema_version" in loaded["validation_issues"]
    assert "stripe_link.source_artifact:section_mismatch" in loaded["validation_issues"]


def test_preflight_evidence_rejects_invalid_timestamps(tmp_path):
    evidence = _complete_preflight_evidence()
    evidence["stripe_projects"]["projects_catalog_checked_at"] = "June 29 2026"
    evidence["stripe_link"]["source_artifact_redacted_at"] = "not-a-timestamp"
    evidence_path = _write_preflight_evidence(tmp_path, evidence)

    loaded = load_preflight_evidence(evidence_path)
    report = build_probe_report(
        env={"VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567", "TWILIO_ACCOUNT_SID": "AC123"},
        env_files=[],
        preflight_evidence_path=evidence_path,
        which=lambda command: f"/bin/{command}" if command in {"stripe", "link-cli", "mppx"} else None,
        runner=lambda _argv, _timeout_seconds: CommandResult(exit_code=0),
    )

    assert "stripe_projects.projects_catalog_checked_at: invalid timestamp" in loaded["validation_issues"]
    assert "stripe_link.source_artifact_redacted_at: invalid timestamp" in loaded["validation_issues"]
    assert report["ready"] is False
    assert "stripe_projects_account" in report["required_failures"]
    assert "stripe_link_approval_capability" in report["required_failures"]


def test_preflight_evidence_manifest_merges_redacted_section_files(tmp_path):
    sections = _complete_preflight_evidence()
    for section_name, section in sections.items():
        if not isinstance(section, dict):
            continue
        source_path = tmp_path / f"{section_name}-source.json"
        source_bytes = json.dumps(
            {
                "schema_version": "voiceops.milestone2.redacted_source_artifact.v1",
                "section": section_name,
                "redacted": True,
                "redaction_policy": "references only; no raw secrets or tokens",
            },
            sort_keys=True,
        ).encode("utf-8")
        source_path.write_bytes(source_bytes)
        section["source_artifact"] = source_path.name
        section["source_artifact_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        section["source_artifact_redacted_at"] = "2026-06-29T00:00:00Z"
    (tmp_path / "stripe-projects.json").write_text(
        json.dumps({"redacted": True, "stripe_projects": sections["stripe_projects"]}),
        encoding="utf-8",
    )
    (tmp_path / "stripe-link.json").write_text(json.dumps({"redacted": True, **sections["stripe_link"]}), encoding="utf-8")
    (tmp_path / "mpp.json").write_text(json.dumps({"redacted": True, "mpp": sections["mpp"]}), encoding="utf-8")
    (tmp_path / "phone.json").write_text(json.dumps({"redacted": True, "phone_handoff": sections["phone_handoff"]}), encoding="utf-8")
    (tmp_path / "rollback.json").write_text(json.dumps({"redacted": True, **sections["rollback"]}), encoding="utf-8")
    manifest_path = tmp_path / "preflight-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.milestone2.preflight_evidence_manifest.v1",
                "reports": {
                    "stripe_projects": "stripe-projects.json",
                    "stripe_link": "stripe-link.json",
                    "mpp": "mpp.json",
                    "phone_handoff": "phone.json",
                    "rollback": "rollback.json",
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_preflight_evidence(manifest_path)
    report = build_probe_report(
        env={"VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567", "TWILIO_ACCOUNT_SID": "AC123"},
        env_files=[],
        preflight_evidence_path=manifest_path,
        which=lambda command: f"/bin/{command}" if command in {"stripe", "link-cli", "mppx"} else None,
        runner=lambda _argv, _timeout_seconds: CommandResult(exit_code=0),
    )

    assert loaded["loaded"] is True
    assert loaded["missing_fields"] == []
    assert loaded["validation_issues"] == []
    assert report["ready"] is True


def test_refresh_preflight_manifest_source_sha256_updates_section_files(tmp_path):
    paths = write_preflight_evidence_scaffold(tmp_path)
    manifest_path = Path(paths["preflight_evidence_scaffold_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for section_name, report_path in manifest["reports"].items():
        section_path = manifest_path.parent / report_path
        section_payload = json.loads(section_path.read_text(encoding="utf-8"))
        section = section_payload[section_name]
        source_path = section_path.parent / section["source_artifact"]
        source_path.write_text(
            json.dumps(
                {
                    "schema_version": "voiceops.milestone2.redacted_source_artifact.v1",
                    "section": section_name,
                    "redacted": True,
                    "redaction_policy": "references only; no raw secrets, tokens, cards, or full phone numbers",
                    "summary": f"operator replaced this with redacted {section_name} setup evidence",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    section_path = manifest_path.parent / manifest["reports"]["stripe_projects"]
    section_payload = json.loads(section_path.read_text(encoding="utf-8"))
    section = section_payload["stripe_projects"]
    source_path = section_path.parent / section["source_artifact"]

    before = load_preflight_evidence(manifest_path)
    result = refresh_preflight_source_hashes(manifest_path)
    after_payload = json.loads(section_path.read_text(encoding="utf-8"))
    after_section = after_payload["stripe_projects"]
    after = load_preflight_evidence(manifest_path)

    assert "stripe_projects.source_artifact_sha256: mismatch" in before["validation_issues"]
    assert result["ok"] is True
    assert result["schema_version"] == "voiceops.milestone2.preflight_hash_refresh.v1"
    assert result["network_io"] is False
    assert result["env_secret_reads"] is False
    assert result["provider_provisioning"] is False
    assert result["live_spend"] is False
    assert result["manifest_mode"] is True
    update = next(item for item in result["updates"] if item["section"] == "stripe_projects")
    assert update["changed"] is True
    assert after_section["source_artifact_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert "stripe_projects.source_artifact_sha256: mismatch" not in after["validation_issues"]


def test_refresh_preflight_manifest_source_sha256_refuses_wrong_source_section(tmp_path):
    paths = write_preflight_evidence_scaffold(tmp_path)
    manifest_path = Path(paths["preflight_evidence_scaffold_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    section_path = manifest_path.parent / manifest["reports"]["stripe_projects"]
    section_payload = json.loads(section_path.read_text(encoding="utf-8"))
    section = section_payload["stripe_projects"]
    source_path = section_path.parent / section["source_artifact"]
    source_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.milestone2.redacted_source_artifact.v1",
                "section": "stripe_link",
                "redacted": True,
                "redaction_policy": "references only; no raw secrets, tokens, cards, or full phone numbers",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    previous_sha = section["source_artifact_sha256"]

    result = refresh_preflight_source_hashes(manifest_path)
    after_section = json.loads(section_path.read_text(encoding="utf-8"))["stripe_projects"]

    assert result["ok"] is False
    assert "preflight_evidence_manifest:stripe_projects:stripe_projects.source_artifact:section_mismatch" in result["issues"]
    assert after_section["source_artifact_sha256"] == previous_sha


def test_refresh_preflight_source_hashes_refuses_forbidden_paths():
    forbidden = Path("/Users/jethac/.hermes/hermes-agent/preflight-evidence.manifest.json")

    with pytest.raises(ValueError, match="forbidden Hermes worktree"):
        refresh_preflight_source_hashes(forbidden)


def test_preflight_evidence_manifest_requires_explicit_section_source_sha(tmp_path):
    sections = _complete_preflight_evidence()
    sections["stripe_projects"]["source_artifact_redacted_at"] = "2026-06-29T00:00:00Z"
    (tmp_path / "stripe-projects.json").write_text(
        json.dumps({"stripe_projects": sections["stripe_projects"]}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "preflight-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.milestone2.preflight_evidence_manifest.v1",
                "reports": {"stripe_projects": "stripe-projects.json"},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_preflight_evidence(manifest_path)

    assert "stripe_projects.source_artifact" in loaded["missing_fields"]
    assert "stripe_projects.source_artifact_sha256" in loaded["missing_fields"]
    assert "stripe_projects.source_artifact: missing" in loaded["validation_issues"]


def test_preflight_evidence_manifest_rejects_missing_or_invalid_schema(tmp_path):
    sections = _complete_preflight_evidence()
    (tmp_path / "stripe-link.json").write_text(json.dumps(sections["stripe_link"]), encoding="utf-8")
    base_manifest = {"reports": {"stripe_link": "stripe-link.json"}}
    missing_schema_path = tmp_path / "missing-schema-manifest.json"
    missing_schema_path.write_text(json.dumps(base_manifest), encoding="utf-8")

    missing_schema = load_preflight_evidence(missing_schema_path)

    assert "preflight_evidence_manifest:missing_schema_version" in missing_schema["validation_issues"]

    invalid_schema_path = tmp_path / "invalid-schema-manifest.json"
    invalid_schema_path.write_text(json.dumps({**base_manifest, "schema_version": "wrong.schema.v1"}), encoding="utf-8")

    invalid_schema = load_preflight_evidence(invalid_schema_path)

    assert "preflight_evidence_manifest:invalid_schema_version" in invalid_schema["validation_issues"]


def test_preflight_evidence_manifest_rejects_example_only_referenced_sections(tmp_path):
    sections = _complete_preflight_evidence()
    sections["stripe_link"]["example_only"] = False
    (tmp_path / "stripe-link.json").write_text(json.dumps(sections["stripe_link"]), encoding="utf-8")
    sections["mpp"]["example_only"] = False
    (tmp_path / "mpp.json").write_text(json.dumps({"mpp": sections["mpp"]}), encoding="utf-8")
    manifest_path = tmp_path / "preflight-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.milestone2.preflight_evidence_manifest.v1",
                "example_only": False,
                "reports": {
                    "stripe_link": "stripe-link.json",
                    "mpp": "mpp.json",
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_preflight_evidence(manifest_path)

    assert "example_only evidence is not accepted" in loaded["validation_issues"]
    assert "preflight_evidence_manifest:stripe_link:example_only evidence is not accepted" in loaded["validation_issues"]
    assert "preflight_evidence_manifest:mpp:example_only evidence is not accepted" in loaded["validation_issues"]
    assert "stripe_link: example_only evidence is not accepted" in loaded["validation_issues"]
    assert "mpp: example_only evidence is not accepted" in loaded["validation_issues"]


def test_preflight_evidence_manifest_prefers_manifest_relative_paths(monkeypatch, tmp_path):
    manifest_dir = tmp_path / "manifest-dir"
    cwd_dir = tmp_path / "cwd"
    manifest_dir.mkdir()
    cwd_dir.mkdir()
    manifest_section = {"account_ref": "stripe-link-account-ref-demo", "approval_capability_confirmed": True, "max_approved_cents": 20_000, "currency": "usd"}
    cwd_section = {"account_ref": "sk_live_wrong_file_should_not_load", "approval_capability_confirmed": True, "max_approved_cents": 20_000, "currency": "usd"}
    (manifest_dir / "stripe-link.json").write_text(json.dumps(manifest_section), encoding="utf-8")
    (cwd_dir / "stripe-link.json").write_text(json.dumps(cwd_section), encoding="utf-8")
    manifest_path = manifest_dir / "preflight-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.milestone2.preflight_evidence_manifest.v1",
                "reports": {"stripe_link": "stripe-link.json"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(cwd_dir)

    loaded = load_preflight_evidence(manifest_path)

    assert "stripe_link.account_ref" in loaded["fields_present"]
    assert not any("secret-like value" in issue for issue in loaded["validation_issues"])


def test_preflight_evidence_manifest_does_not_fallback_to_cwd(monkeypatch, tmp_path):
    manifest_dir = tmp_path / "manifest-dir"
    cwd_dir = tmp_path / "cwd"
    manifest_dir.mkdir()
    cwd_dir.mkdir()
    cwd_section = {
        "account_ref": "stripe-link-account-ref-demo",
        "approval_capability_confirmed": True,
        "max_approved_cents": 20_000,
        "currency": "usd",
    }
    (cwd_dir / "stripe-link.json").write_text(json.dumps(cwd_section), encoding="utf-8")
    manifest_path = manifest_dir / "preflight-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.milestone2.preflight_evidence_manifest.v1",
                "reports": {"stripe_link": "stripe-link.json"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(cwd_dir)

    loaded = load_preflight_evidence(manifest_path)

    assert "preflight_evidence_manifest:stripe_link:evidence file not found" in loaded["validation_issues"]
    assert "stripe_link.account_ref" in loaded["missing_fields"]


def test_preflight_evidence_manifest_does_not_fallback_to_basename(tmp_path):
    manifest_dir = tmp_path / "manifest-dir"
    manifest_dir.mkdir()
    (manifest_dir / "stripe-link.json").write_text(
        json.dumps(
            {
                "account_ref": "stripe-link-account-ref-demo",
                "approval_capability_confirmed": True,
                "max_approved_cents": 20_000,
                "currency": "usd",
            }
        ),
        encoding="utf-8",
    )
    manifest_path = manifest_dir / "preflight-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.milestone2.preflight_evidence_manifest.v1",
                "reports": {"stripe_link": "sections/stripe-link.json"},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_preflight_evidence(manifest_path)

    assert "preflight_evidence_manifest:stripe_link:evidence file not found" in loaded["validation_issues"]
    assert "stripe_link.account_ref" in loaded["missing_fields"]


def test_preflight_evidence_manifest_rejects_example_or_invalid_sections(tmp_path):
    bad_section = tmp_path / "stripe-projects.json"
    bad_section.write_text("[]", encoding="utf-8")
    manifest_path = tmp_path / "preflight-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                **build_preflight_evidence_manifest_example(),
                "reports": {
                    "stripe_projects": str(bad_section),
                    "stripe_link": "missing-link.json",
                    "unknown": "ignored.json",
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_preflight_evidence(manifest_path)

    assert loaded["loaded"] is True
    assert "example_only evidence is not accepted" in loaded["validation_issues"]
    assert "preflight_evidence_manifest:stripe_projects:evidence root must be an object" in loaded["validation_issues"]
    assert "preflight_evidence_manifest:stripe_link:evidence file not found" in loaded["validation_issues"]
    assert "preflight_evidence_manifest:unknown:unknown_section" in loaded["validation_issues"]
    assert "stripe_projects.account_ref" in loaded["missing_fields"]


def test_milestone2_execution_plan_defines_safety_gates_receipts_and_rollback():
    report = build_probe_report(
        env={
            "VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567",
            "TWILIO_ACCOUNT_SID": "AC123456789abcdef",
            "TWILIO_AUTH_TOKEN": "secret-token",
            "STRIPE_SECRET_KEY": "sk_live_123456789abcdef",
        },
        env_files=[],
        which=lambda command: f"/bin/{command}" if command in {"stripe", "link-cli", "mppx"} else None,
        runner=lambda _argv, _timeout_seconds: CommandResult(exit_code=0),
    )
    plan = build_milestone2_execution_plan(report)

    assert plan["schema_version"] == "voiceops.milestone2.execution_plan.v1"
    assert plan["artifact_id"] == "voiceops-m2-execution-plan"
    assert plan["mode"] == {"artifact_only": True, "bounded": True, "headless": True}
    assert plan["safety"] == {
        "account_mutation": False,
        "credential_retrieval": False,
        "env_secret_reads": False,
        "live_spend": False,
        "network_io": False,
        "outbound_phone_calls": False,
        "provider_provisioning": False,
    }
    assert {
        "live_spend",
        "provider_provisioning",
        "credential_retrieval",
        "outbound_calls",
        "outbound_messages",
        "network_tunnels",
        "raw_card_data",
        "unapproved_recurring_charges",
    } <= set(plan["blocked_capabilities"])
    assert plan["preflight"]["run_command_probes_default"] is False
    assert plan["preflight"]["run_command_probes_does_not_grant_approval"] is True
    assert "provisioning-preflight-evidence.template.json" == plan["preflight"]["preflight_evidence_template"]
    assert plan["demo_refs"] == {
        "audit_ledger": "audit-ledger.jsonl",
        "nemoclaw_packet": "nemoclaw-action-packet.json",
        "phone_context": "phone-context.json",
        "stripe_actions_dry_run": "stripe-actions-dry-run.sh",
        "voiceops_demo": "voiceops-demo.json",
    }
    assert {
        "approval_id",
        "command_sha256",
        "credential_location_ref",
        "receipt_id",
        "rollback_ref",
    } <= set(plan["receipt_schema"]["required_fields"])
    assert "credential_ref_id" in plan["credential_location_schema"]["required_fields"]
    assert "raw_secret" in plan["credential_location_schema"]["forbidden_fields"]
    assert {
        "audit-ledger.read-only-discovery.jsonl",
    } == {step["records_to"] for step in plan["read_only_discovery"]}
    assert {
        "deprovision_voip_provider",
        "refund_or_cancel_service_credit",
        "cancel_or_end_phone_handoff",
        "correct_or_remove_status_message",
    } <= set(plan["rollback_plan"])

    risky_steps = [
        step
        for step in plan["execution_steps"]
        if step["provider"] in {"stripe-projects", "stripe-link-cli", "voiceops-phone-bridge", "hermes-gateway"}
    ]
    assert risky_steps
    assert all(step["requires_approval"] is True for step in risky_steps)
    assert all(step["status"] == "blocked_until_explicit_approval" for step in risky_steps)
    assert {gate["gate_id"] for gate in plan["approval_gates"]} == {
        "outbound-status-messages",
        "phone-call-handoff",
        "stripe-link-spend",
        "stripe-projects-provisioning",
    }
    assert {step["step_id"] for step in risky_steps} <= {
        action["action_id"] for action in plan["approval_required_actions"]
    }
    assert set(plan["approval_contracts"]) == {
        action["action_id"] for action in plan["approval_required_actions"]
    }
    for action in plan["approval_required_actions"]:
        receipt_slot = _dot_get(plan, action["expected_receipt_ref"])
        rollback_slot = _dot_get(plan, action["rollback_ref"])
        assert receipt_slot["status"] == "not_executed"
        assert receipt_slot["schema_ref"] == "receipt_schema"
        assert receipt_slot["action_id"] == action["action_id"]
        assert receipt_slot["approval_id"] == action["approval_id"]
        assert receipt_slot["command_sha256"] == hashlib.sha256(action["command"].encode("utf-8")).hexdigest()
        assert receipt_slot["command_sha256"] == action["command_sha256"]
        assert receipt_slot["credential_location_ref"] == action["credential_location_ref"]
        assert receipt_slot["rollback_ref"] == action["rollback_ref"]
        assert rollback_slot
        contract = action["approval_contract"]
        assert contract == plan["approval_contracts"][action["action_id"]]
        assert action["approval_id"] == contract["approval_id"]
        assert action["command_sha256"] == contract["command_sha256"]
        assert contract["command_sha256"] == hashlib.sha256(action["command"].encode("utf-8")).hexdigest()
        assert len(contract["command_sha256"]) == 64
        assert contract["approved_by_ref"] is None
        assert contract["allowed_decisions"] == ["approve_once", "deny", "hold"]
        if action["credential_location_required"]:
            credential_slot = _dot_get(plan, action["credential_location_ref"])
            assert action["credential_location_schema_ref"] == "credential_location_schema"
            assert credential_slot["status"] == "not_created"
            assert credential_slot["schema_ref"] == "credential_location_schema"
        else:
            assert action["credential_location_ref"] is None
            assert action["credential_location_schema_ref"] is None
    assert set(plan["expected_post_approval_evidence"]) == {
        action["action_id"] for action in plan["approval_required_actions"]
    }
    assert all(
        evidence["execution_status"] == "not_executed"
        for evidence in plan["expected_post_approval_evidence"].values()
    )
    for action_id, evidence in plan["expected_post_approval_evidence"].items():
        action = next(item for item in plan["approval_required_actions"] if item["action_id"] == action_id)
        assert evidence["approval_id"] == action["approval_id"]
        assert evidence["command_sha256"] == action["command_sha256"]
        assert evidence["expected_receipt_ref"] == action["expected_receipt_ref"]
        assert evidence["credential_location_ref"] == action["credential_location_ref"]
        assert evidence["rollback_ref"] == action["rollback_ref"]

    serialized = json.dumps(plan)
    assert "sk_live_123456789abcdef" not in serialized
    assert "+15551234567" not in serialized
    assert "secret-token" not in serialized


def test_post_approval_receipts_validate_redacted_bundle_and_emit_ledger(tmp_path):
    report = build_probe_report(env={}, env_files=[], which=lambda _command: None)
    plan = build_milestone2_execution_plan(report)
    actions = plan["approval_required_actions"]
    estimates = {
        step["step_id"]: step["estimated_cents"]
        for step in plan["execution_steps"]
        if step.get("requires_approval") is True
    }
    payload = {
        "schema_version": "voiceops.milestone2.post_approval_receipts.v1",
        "redaction_policy": "references only",
        "receipts": [
            {
                "receipt_id": f"receipt-{action['action_id']}-001",
                "action_id": action["action_id"],
                "approval_id": action["approval_id"],
                "provider": action["provider"],
                "status": "executed",
                "approved_by": "operator-ref-demo",
                "approved_at": "2026-06-29T00:00:00Z",
                "executed_at": "2026-06-29T00:00:30Z",
                "command_sha256": action["command_sha256"],
                "amount_cents": estimates.get(action["action_id"], 0),
                "currency": "usd",
                "approval_artifact": action["approval_artifact"],
                "external_reference": f"provider-resource-ref-{action['action_id']}",
                "credential_location_ref": action["credential_location_ref"],
                "rollback_ref": action["rollback_ref"],
                "audit_event_id": f"audit-{action['action_id']}-001",
            }
            for action in actions
        ],
        "credential_locations": [
            {
                "credential_ref_id": action["credential_location_ref"],
                "provider": action["provider"],
                "service_id": f"provider-resource-ref-{action['action_id']}",
                "storage_backend": "provider_managed",
                "secret_name_or_path": f"credential-location-ref-{action['action_id']}",
                "created_by_action_id": action["action_id"],
                "rotation_due": "2026-09-29T00:00:00Z",
            }
            for action in actions
            if action["credential_location_required"]
        ],
        "rollback_receipts": [
            {
                "rollback_ref": action["rollback_ref"],
                "status": "not_run",
                "owner_ref": "operator-ref-demo",
                "notes": "No rollback run.",
            }
            for action in actions
        ],
        "audit_events": [
            {
                "audit_event_id": f"audit-{action['action_id']}-001",
                "action_id": action["action_id"],
                "receipt_id": f"receipt-{action['action_id']}-001",
                "status": "executed",
                "provider": action["provider"],
                "artifact_ref": "post-approval-receipts.json",
                "operator_next_step": "Review provider dashboard and rollback window.",
            }
            for action in actions
        ],
    }
    receipt_path = tmp_path / "post-approval-receipts.json"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_post_approval_receipts(receipt_path, plan)
    report_with_receipts = build_probe_report(
        env={},
        env_files=[],
        post_approval_receipts_path=receipt_path,
        which=lambda _command: None,
    )
    paths = write_probe_artifacts(tmp_path / "out", report_with_receipts)
    ledger_rows = Path(paths["post_approval_audit_ledger"]).read_text(encoding="utf-8").splitlines()

    assert loaded["status"] == "valid"
    assert loaded["validation_issues"] == []
    assert loaded["receipt_count"] == 4
    assert loaded["ledger_rows"][0]["receipt_id"] == "receipt-provision-voip-provider-001"
    assert report_with_receipts["post_approval_receipts"]["status"] == "valid"
    assert len(ledger_rows) == 4
    assert json.loads(ledger_rows[0])["audit_event_id"] == "audit-provision-voip-provider-001"
    assert "+15551234567" not in json.dumps(loaded)


def test_post_approval_receipts_reject_examples_secrets_and_mismatches():
    report = build_probe_report(env={}, env_files=[], which=lambda _command: None)
    plan = build_milestone2_execution_plan(report)
    example = build_post_approval_receipts_example(plan)
    bad = json.loads(json.dumps(example))
    bad.pop("example_only")
    bad["receipts"][0]["command_sha256"] = "0" * 64
    bad["receipts"][0]["provider"] = "wrong-provider"
    bad["receipts"][0]["currency"] = "eur"
    bad["receipts"][0]["amount_cents"] = 999999
    bad["receipts"][0]["approval_artifact"] = "wrong-approval.json"
    bad["receipts"][0]["approved_at"] = "2026-06-29T00:01:00Z"
    bad["receipts"][0]["executed_at"] = "2026-06-29T00:00:30Z"
    bad["receipts"][0]["external_reference"] = "sk_live_123456789abcdef"
    bad["receipts"] = bad["receipts"][:-1]
    bad["audit_events"][0]["provider"] = "wrong-provider"
    bad["audit_events"][0]["status"] = "held"
    bad["credential_locations"][0]["raw_secret"] = "secret"

    example_result = validate_post_approval_receipts(example, plan)
    bad_result = validate_post_approval_receipts(bad, plan)

    assert example_result["status"] == "invalid"
    assert any("example_only evidence is not accepted" in issue for issue in example_result["validation_issues"])
    assert bad_result["status"] == "invalid"
    assert "post_approval_receipts:receipt-example-provision-voip-provider:command_sha256_mismatch" in bad_result[
        "validation_issues"
    ]
    assert "post_approval_receipts:receipt-example-provision-voip-provider:provider_mismatch" in bad_result[
        "validation_issues"
    ]
    assert "post_approval_receipts:receipt-example-provision-voip-provider:approval_artifact_mismatch" in bad_result[
        "validation_issues"
    ]
    assert "post_approval_receipts:receipt-example-provision-voip-provider:currency_mismatch" in bad_result[
        "validation_issues"
    ]
    assert "post_approval_receipts:receipt-example-provision-voip-provider:amount_exceeds_estimate" in bad_result[
        "validation_issues"
    ]
    assert "post_approval_receipts:receipt-example-provision-voip-provider:approved_after_executed" in bad_result[
        "validation_issues"
    ]
    assert "post_approval_receipts:receipt-example-provision-voip-provider:audit_status_mismatch" in bad_result[
        "validation_issues"
    ]
    assert "post_approval_receipts:receipt-example-provision-voip-provider:audit_provider_mismatch" in bad_result[
        "validation_issues"
    ]
    assert any(issue.startswith("post_approval_receipts:missing_receipts_for_actions:") for issue in bad_result["validation_issues"])
    assert any("secret-like value" in issue for issue in bad_result["validation_issues"])
    assert any("forbidden_raw_field" in issue for issue in bad_result["validation_issues"])
    assert bad_result["ledger_rows"] == []


def test_probe_loads_env_file_key_presence_without_values(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "VOICEOPS_DEMO_PHONE_NUMBER=+15551234567",
                "TWILIO_AUTH_TOKEN=secret-token",
                "TWILIO_ACCOUNT_SID=AC123456789abcdef",
            ]
        ),
        encoding="utf-8",
    )

    report = build_probe_report(
        env={},
        env_files=[env_file],
        preflight_evidence_path=_write_preflight_evidence(tmp_path),
        which=lambda command: f"/bin/{command}" if command in {"stripe", "link-cli", "mppx"} else None,
        runner=lambda _argv, _timeout_seconds: CommandResult(exit_code=0),
    )

    serialized = json.dumps(report)
    assert report["ready"] is True
    assert report["env_sources"][1]["loaded"] is True
    assert "VOICEOPS_DEMO_PHONE_NUMBER" in serialized
    assert "+15551234567" not in serialized
    assert "secret-token" not in serialized


def test_probe_refuses_forbidden_hermes_agent_env_path():
    forbidden = Path("/Users/jethac/.hermes/hermes-agent/.env")

    with pytest.raises(ValueError, match="forbidden Hermes worktree"):
        build_probe_report(env={}, env_files=[forbidden], which=lambda _command: None)


def test_probe_refuses_forbidden_hermes_agent_preflight_evidence_path():
    forbidden = Path("/Users/jethac/.hermes/hermes-agent/preflight-evidence.json")

    with pytest.raises(ValueError, match="forbidden Hermes worktree"):
        build_probe_report(env={}, env_files=[], preflight_evidence_path=forbidden, which=lambda _command: None)


def test_preflight_evidence_rejects_secret_like_values(tmp_path):
    evidence = _complete_preflight_evidence()
    evidence["phone_handoff"]["credential_location_ref"] = "sk_live_123456789abcdef"
    evidence["phone_handoff"]["phone_target_ref"] = "+15551234567"
    evidence_path = _write_preflight_evidence(tmp_path, evidence)

    report = build_probe_report(
        env={"VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567", "VAPI_API_KEY": "secret"},
        env_files=[],
        preflight_evidence_path=evidence_path,
        which=lambda command: f"/bin/{command}" if command in {"stripe", "link-cli", "mppx"} else None,
    )

    serialized = json.dumps(report)
    assert report["ready"] is False
    assert "credential_location_reference" in report["required_failures"]
    assert "sk_live_123456789abcdef" not in serialized
    assert "+15551234567" not in serialized
    assert "secret-like value" in serialized
    assert "phone-like value" in serialized


def test_preflight_evidence_rejects_generic_secret_shaped_refs(tmp_path):
    evidence = _complete_preflight_evidence()
    evidence["phone_handoff"]["credential_location_ref"] = "api-key-live-abcdefghijklmnopqrstuvwxyz"
    evidence["mpp"]["policy_ref"] = "auth-token-prod-abcdefghijklmnopqrstuvwxyz"
    evidence_path = _write_preflight_evidence(tmp_path, evidence)

    loaded = load_preflight_evidence(evidence_path)

    assert "phone_handoff.credential_location_ref: secret-like value" in loaded["validation_issues"]
    assert "mpp.policy_ref: secret-like value" in loaded["validation_issues"]


def test_preflight_evidence_rejects_not_redacted_artifacts_and_nested_example_only(tmp_path):
    evidence = _complete_preflight_evidence()
    evidence_path = _write_preflight_evidence(tmp_path, evidence)
    stripe_source = tmp_path / "stripe_projects-source.json"
    stripe_source.write_text(
        json.dumps(
            {
                "redacted": False,
                "redaction_policy": "not redacted",
                "nested": {"example_only": False},
            }
        ),
        encoding="utf-8",
    )
    evidence["stripe_projects"]["source_artifact_sha256"] = hashlib.sha256(stripe_source.read_bytes()).hexdigest()
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    loaded = load_preflight_evidence(evidence_path)

    assert "stripe_projects.source_artifact:artifact is not marked redacted" in loaded["validation_issues"]
    assert "stripe_projects.source_artifact:nested.example_only: example_only field is not accepted" in loaded["validation_issues"]


def test_command_probe_validation_requires_exact_manifest_argv():
    _validate_safe_probe_command(["stripe", "--version"])
    _validate_safe_probe_command(["stripe", "projects", "--help"])

    with pytest.raises(ValueError, match="allowlisted manifest exactly"):
        _validate_safe_probe_command(["stripe", "customers", "--help"])
    with pytest.raises(ValueError, match="allowlisted manifest exactly"):
        _validate_safe_probe_command(["unknown-cli", "--version"])


def test_readonly_discovery_validation_requires_exact_manifest_argv():
    _validate_readonly_discovery_command(["stripe", "projects", "list", "--limit", "10"])
    _validate_readonly_discovery_command(["link-cli", "auth", "status"])

    with pytest.raises(ValueError, match="allowlisted manifest exactly"):
        _validate_readonly_discovery_command(["stripe", "projects", "add", "twilio/voice"])
    with pytest.raises(ValueError, match="allowlisted manifest exactly"):
        _validate_readonly_discovery_command(["link-cli", "auth", "status", "--json"])


def test_probe_cli_smoke_no_command_probes(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "voiceops_provisioning_probe.py"
    result = subprocess.run(
        ["python", str(script), "--output-dir", str(tmp_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert Path(payload["artifacts"]["json"]).exists()
    assert Path(payload["artifacts"]["markdown"]).exists()


def test_parse_args_defaults_to_requested_artifact_dir():
    args = parse_args([])

    assert args.output_dir == Path("artifacts/voiceops-provisioning/current")
    assert args.preflight_evidence is None
    assert args.post_approval_receipts is None
    assert args.refresh_preflight_source_hashes is None
    assert args.run_command_probes is False
    assert args.run_readonly_discovery is False
