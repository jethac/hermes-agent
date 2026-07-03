import json

from hermes_cli import realtime_voice_production_review
from hermes_cli.realtime_voice_production_review import (
    KAME_DGX_BENCHMARK_EVIDENCE_CHECK,
    KAME_DGX_REQUIRED_BENCHMARK_COVERAGE,
    REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS,
)


def _write_dgx_benchmark_validation_artifact(tmp_path, *, ok=True):
    artifact = tmp_path / "dgx-benchmark-validation.json"
    artifact.write_text(
        json.dumps(
            {
                "ok": ok,
                "benchmark_evidence": {
                    "ok": ok,
                    "issues": [] if ok else ["interface_direct_audio_latency: failed"],
                    "coverage": {
                        key: ok
                        for key in KAME_DGX_REQUIRED_BENCHMARK_COVERAGE
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return artifact


def _review_evidence(*, dgx_artifact=None):
    evidence = {
        key: {
            "notes": f"{key} passed in production review.",
            "artifacts": [f"./artifacts/realtime-voice-review/{key}.md"],
        }
        for key in REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS
    }
    if dgx_artifact is not None:
        evidence[KAME_DGX_BENCHMARK_EVIDENCE_CHECK] = {
            "notes": "DGX Spark benchmark validator passed.",
            "artifacts": [str(dgx_artifact)],
        }
    return evidence


def _passed_review_report(tmp_path, *, reviewer="qa@example.test"):
    return realtime_voice_production_review.build_production_review_report(
        evidence=_review_evidence(dgx_artifact=_write_dgx_benchmark_validation_artifact(tmp_path)),
        reviewer=reviewer,
        passed_checks=REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS,
    )


def test_production_review_template_defaults_to_pending_checks(tmp_path, capsys):
    report_path = tmp_path / "review.json"

    result = realtime_voice_production_review.main(
        [
            str(report_path),
            "--write-template",
            "--reviewer",
            "qa@example.test",
        ]
    )

    assert result == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["kind"] == "realtime_voice_production_review"
    assert report["reviewer"] == "qa@example.test"
    assert set(report["checks"]) == set(REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS)
    assert set(report["evidence"]) == set(REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS)
    assert all(value is False for value in report["checks"].values())
    assert all(value == {"notes": "", "artifacts": []} for value in report["evidence"].values())
    output = capsys.readouterr().out
    assert "Pending check(s)" in output


def test_production_review_template_accepts_evidence_flags_for_all_passed_checks(
    monkeypatch,
    tmp_path,
    capsys,
):
    report_path = tmp_path / "review.json"
    saved = {}
    monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {"voice": {"realtime": {"enabled": True}}})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.setdefault("config", cfg))
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: tmp_path / "config.yaml")
    dgx_artifact = _write_dgx_benchmark_validation_artifact(tmp_path)
    evidence_args = []
    for key in REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS:
        evidence_args.extend(["--evidence-note", f"{key}=Reviewed {key}."])
        artifact = str(dgx_artifact) if key == KAME_DGX_BENCHMARK_EVIDENCE_CHECK else f"./artifacts/{key}.md"
        evidence_args.extend(["--evidence-artifact", f"{key}={artifact}"])
    evidence_args.extend(["--evidence-artifact", "desktop_reconnect_recovery=./artifacts/reconnect-video.txt"])

    result = realtime_voice_production_review.main(
        [
            str(report_path),
            "--write-template",
            "--reviewer",
            "qa@example.test",
            "--all-passed",
            "--apply",
            *evidence_args,
        ]
    )

    assert result == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert all(value is True for value in report["checks"].values())
    assert report["evidence"]["human_en_ja_conversations"] == {
        "notes": "Reviewed human_en_ja_conversations.",
        "artifacts": ["./artifacts/human_en_ja_conversations.md"],
    }
    assert report["evidence"]["desktop_reconnect_recovery"]["artifacts"] == [
        "./artifacts/desktop_reconnect_recovery.md",
        "./artifacts/reconnect-video.txt",
    ]
    assert saved["config"]["voice"]["realtime"]["production_review_report"] == str(report_path)
    output = capsys.readouterr().out
    assert "Realtime voice production review OK" in output
    assert "Updated realtime voice production_review_report" in output


def test_production_review_template_rejects_invalid_evidence_flag(tmp_path, capsys):
    report_path = tmp_path / "review.json"

    result = realtime_voice_production_review.main(
        [
            str(report_path),
            "--write-template",
            "--reviewer",
            "qa@example.test",
            "--all-passed",
            "--evidence-note",
            "not_a_check=reviewed",
        ]
    )

    assert result == 1
    assert not report_path.exists()
    assert "unknown check: not_a_check" in capsys.readouterr().err


def test_production_review_validation_reports_missing_checks(tmp_path, capsys):
    report_path = tmp_path / "review.json"
    report_path.write_text(
        json.dumps(
            {
                "kind": "realtime_voice_production_review",
                "reviewer": "qa@example.test",
                "reviewed_at": "2026-06-08T00:00:00Z",
                "checks": {"human_en_ja_conversations": True},
            }
        ),
        encoding="utf-8",
    )

    result = realtime_voice_production_review.main([str(report_path)])

    assert result == 1
    error = capsys.readouterr().err
    assert "Realtime voice production review failed" in error
    assert "review_check_missing:noisy_room_and_headset_coverage" in error


def test_production_review_requires_desktop_reconnect_recovery_check(tmp_path, capsys):
    report_path = tmp_path / "review.json"
    checks = {
        key: True
        for key in REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS
        if key != "desktop_reconnect_recovery"
    }
    report_path.write_text(
        json.dumps(
            {
                "kind": "realtime_voice_production_review",
                "reviewer": "qa@example.test",
                "reviewed_at": "2026-06-08T00:00:00Z",
                "checks": checks,
                "evidence": _review_evidence(),
            }
        ),
        encoding="utf-8",
    )

    result = realtime_voice_production_review.main([str(report_path)])

    assert result == 1
    assert "review_check_missing:desktop_reconnect_recovery" in capsys.readouterr().err


def test_production_review_requires_kame_dgx_benchmark_evidence_check(tmp_path, capsys):
    report_path = tmp_path / "review.json"
    checks = {
        key: True
        for key in REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS
        if key != "kame_dgx_benchmark_evidence"
    }
    report_path.write_text(
        json.dumps(
            {
                "kind": "realtime_voice_production_review",
                "reviewer": "qa@example.test",
                "reviewed_at": "2026-06-08T00:00:00Z",
                "checks": checks,
                "evidence": _review_evidence(dgx_artifact=_write_dgx_benchmark_validation_artifact(tmp_path)),
            }
        ),
        encoding="utf-8",
    )

    result = realtime_voice_production_review.main([str(report_path)])

    assert result == 1
    assert "review_check_missing:kame_dgx_benchmark_evidence" in capsys.readouterr().err


def test_production_review_validation_requires_evidence_for_passed_checks(tmp_path, capsys):
    report_path = tmp_path / "review.json"
    report_path.write_text(
        json.dumps(
            realtime_voice_production_review.build_production_review_report(
                reviewer="qa@example.test",
                passed_checks=REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS,
            )
        ),
        encoding="utf-8",
    )

    result = realtime_voice_production_review.main([str(report_path)])

    assert result == 1
    error = capsys.readouterr().err
    assert "review_evidence_missing:human_en_ja_conversations" in error
    assert "review_evidence_missing:desktop_reconnect_recovery" in error


def test_production_review_validation_accepts_string_evidence_for_manual_checks(tmp_path, capsys):
    report_path = tmp_path / "review.json"
    dgx_artifact = _write_dgx_benchmark_validation_artifact(tmp_path)
    evidence = {key: f"reviewed {key}" for key in REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS}
    evidence[KAME_DGX_BENCHMARK_EVIDENCE_CHECK] = {
        "notes": "DGX Spark benchmark validator passed.",
        "artifacts": [str(dgx_artifact)],
    }
    report_path.write_text(
        json.dumps(
            {
                "kind": "realtime_voice_production_review",
                "reviewer": "qa@example.test",
                "reviewed_at": "2026-06-08T00:00:00Z",
                "checks": {key: True for key in REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS},
                "evidence": evidence,
            }
        ),
        encoding="utf-8",
    )

    result = realtime_voice_production_review.main([str(report_path)])

    assert result == 0
    assert "Realtime voice production review OK" in capsys.readouterr().out


def test_production_review_rejects_kame_benchmark_note_without_validator_json(tmp_path, capsys):
    report_path = tmp_path / "review.json"
    evidence = _review_evidence()
    evidence[KAME_DGX_BENCHMARK_EVIDENCE_CHECK] = "reviewed by hand"
    report_path.write_text(
        json.dumps(
            {
                "kind": "realtime_voice_production_review",
                "reviewer": "qa@example.test",
                "reviewed_at": "2026-06-08T00:00:00Z",
                "checks": {key: True for key in REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS},
                "evidence": evidence,
            }
        ),
        encoding="utf-8",
    )

    result = realtime_voice_production_review.main([str(report_path)])

    assert result == 1
    assert (
        "review_evidence_invalid:kame_dgx_benchmark_evidence:requires_local_validator_json"
        in capsys.readouterr().err
    )


def test_production_review_rejects_failed_kame_benchmark_validator_json(tmp_path, capsys):
    report_path = tmp_path / "review.json"
    evidence = _review_evidence(dgx_artifact=_write_dgx_benchmark_validation_artifact(tmp_path, ok=False))
    report_path.write_text(
        json.dumps(
            {
                "kind": "realtime_voice_production_review",
                "reviewer": "qa@example.test",
                "reviewed_at": "2026-06-08T00:00:00Z",
                "checks": {key: True for key in REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS},
                "evidence": evidence,
            }
        ),
        encoding="utf-8",
    )

    result = realtime_voice_production_review.main([str(report_path)])

    assert result == 1
    error = capsys.readouterr().err
    assert "review_evidence_invalid:kame_dgx_benchmark_evidence" in error
    assert "validator_not_ok" in error


def test_production_review_requires_async_kame_voiceops_coverage_in_benchmark_artifact(tmp_path, capsys):
    report_path = tmp_path / "review.json"
    artifact = _write_dgx_benchmark_validation_artifact(tmp_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["benchmark_evidence"]["coverage"]["async_oracle_witness_fusion_single_bundle"] = False
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    evidence = _review_evidence(dgx_artifact=artifact)
    report_path.write_text(
        json.dumps(
            {
                "kind": "realtime_voice_production_review",
                "reviewer": "qa@example.test",
                "reviewed_at": "2026-06-08T00:00:00Z",
                "checks": {key: True for key in REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS},
                "evidence": evidence,
            }
        ),
        encoding="utf-8",
    )

    result = realtime_voice_production_review.main([str(report_path)])

    assert result == 1
    error = capsys.readouterr().err
    assert "review_evidence_invalid:kame_dgx_benchmark_evidence" in error
    assert "validator_not_ok" in error
    assert "async_oracle_witness_fusion_single_bundle" in KAME_DGX_REQUIRED_BENCHMARK_COVERAGE
    assert "async_oracle_runtime_kame_action_gate_enforced" in KAME_DGX_REQUIRED_BENCHMARK_COVERAGE
    assert "async_oracle_unpromoted_hypothesis_action_sinks_clean" in KAME_DGX_REQUIRED_BENCHMARK_COVERAGE


def test_production_review_resolves_kame_benchmark_artifact_relative_to_report(tmp_path, monkeypatch, capsys):
    report_dir = tmp_path / "review"
    report_dir.mkdir()
    artifacts_dir = report_dir / "artifacts"
    artifacts_dir.mkdir()
    dgx_artifact = _write_dgx_benchmark_validation_artifact(artifacts_dir)
    report_path = report_dir / "review.json"
    evidence = _review_evidence()
    evidence[KAME_DGX_BENCHMARK_EVIDENCE_CHECK] = {
        "notes": "DGX Spark benchmark validator passed.",
        "artifacts": ["artifacts/dgx-benchmark-validation.json"],
    }
    report_path.write_text(
        json.dumps(
            {
                "kind": "realtime_voice_production_review",
                "reviewer": "qa@example.test",
                "reviewed_at": "2026-06-08T00:00:00Z",
                "checks": {key: True for key in REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS},
                "evidence": evidence,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert dgx_artifact.exists()

    result = realtime_voice_production_review.main([str(report_path)])

    assert result == 0
    assert "Realtime voice production review OK" in capsys.readouterr().out


def test_production_review_validation_accepts_all_checks(tmp_path, capsys):
    report_path = tmp_path / "review.json"
    report_path.write_text(
        json.dumps(_passed_review_report(tmp_path)),
        encoding="utf-8",
    )

    result = realtime_voice_production_review.main([str(report_path)])

    assert result == 0
    assert "Realtime voice production review OK" in capsys.readouterr().out


def test_production_review_apply_updates_config_after_validation(monkeypatch, tmp_path, capsys):
    report_path = tmp_path / "review.json"
    report_path.write_text(
        json.dumps(_passed_review_report(tmp_path)),
        encoding="utf-8",
    )
    saved = {}
    monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {"voice": {"realtime": {"enabled": True}}})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.setdefault("config", cfg))
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: tmp_path / "config.yaml")

    result = realtime_voice_production_review.main([str(report_path), "--apply"])

    assert result == 0
    assert saved["config"]["voice"]["realtime"]["enabled"] is True
    assert saved["config"]["voice"]["realtime"]["production_review_report"] == str(report_path)
    assert "Updated realtime voice production_review_report" in capsys.readouterr().out


def test_production_review_apply_skips_pending_template(monkeypatch, tmp_path, capsys):
    report_path = tmp_path / "review.json"
    saved = {}
    monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.setdefault("config", cfg))
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: tmp_path / "config.yaml")

    result = realtime_voice_production_review.main(
        [
            str(report_path),
            "--write-template",
            "--reviewer",
            "qa@example.test",
            "--apply",
        ]
    )

    assert result == 0
    assert saved == {}
    output = capsys.readouterr().out
    assert "Pending check(s)" in output
    assert "Config not updated" in output


def test_production_review_template_refuses_existing_file_without_overwrite(tmp_path, capsys):
    report_path = tmp_path / "review.json"
    report_path.write_text("{}", encoding="utf-8")

    result = realtime_voice_production_review.main([str(report_path), "--write-template"])

    assert result == 1
    assert "already exists" in capsys.readouterr().err
