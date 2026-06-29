from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

import pytest

from scripts.voiceops_provisioning_probe import (
    CommandResult,
    build_probe_report,
    parse_args,
    write_probe_artifacts,
)


def test_probe_passes_with_safe_local_tools_and_redacts_outputs():
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
        which=fake_which,
        runner=fake_runner,
        run_commands=True,
    )

    assert report["ready"] is True
    assert report["required_failures"] == []
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


def test_probe_reports_required_failures_without_running_missing_tools():
    calls: list[list[str]] = []

    report = build_probe_report(
        env={},
        env_files=[],
        which=lambda _command: None,
        runner=lambda argv, _timeout_seconds: calls.append(list(argv)) or CommandResult(exit_code=0),
    )

    assert report["ready"] is False
    assert set(report["required_failures"]) == {
        "stripe_cli",
        "stripe_projects_cli",
        "stripe_link_cli",
        "mpp_agent",
        "phone_target",
        "phone_provider",
    }
    assert calls == []


def test_probe_treats_no_command_probes_as_path_presence_only():
    def fake_which(command: str) -> str | None:
        return f"/bin/{command}" if command in {"stripe", "link-cli", "mppx"} else None

    report = build_probe_report(
        env={"VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567", "VAPI_API_KEY": "secret"},
        env_files=[],
        which=fake_which,
        runner=lambda _argv, _timeout_seconds: (_ for _ in ()).throw(AssertionError("runner should not be called")),
    )

    assert report["ready"] is True
    assert all(probe["executed"] is False for probe in report["command_probes"])
    assert {probe["status"] for probe in report["command_probes"] if probe["found"]} == {"found"}


def test_write_probe_artifacts(tmp_path):
    report = build_probe_report(
        env={"VOICEOPS_DEMO_PHONE_NUMBER": "+15551234567", "TWILIO_ACCOUNT_SID": "AC123"},
        env_files=[],
        which=lambda command: f"/bin/{command}" if command in {"stripe", "link-cli", "mppx"} else None,
        runner=lambda _argv, _timeout_seconds: CommandResult(exit_code=0, stdout="version 1.0"),
    )
    paths = write_probe_artifacts(tmp_path, report)

    assert set(paths) == {"json", "markdown", "command_manifest"}
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(paths["command_manifest"]).read_text(encoding="utf-8"))
    markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert payload["probe"]["non_mutating"] is True
    assert "VoiceOps Provisioning Readiness Probe" in markdown
    assert "projects add" in manifest["blocked_patterns"]
    assert "+15551234567" not in json.dumps(payload)


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
    assert args.run_command_probes is False
