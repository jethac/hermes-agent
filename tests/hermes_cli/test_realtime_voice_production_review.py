import json

from hermes_cli import realtime_voice_production_review
from hermes_cli.realtime_voice_production_review import (
    REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS,
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
    assert all(value is False for value in report["checks"].values())
    output = capsys.readouterr().out
    assert "Pending check(s)" in output


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


def test_production_review_validation_accepts_all_checks(tmp_path, capsys):
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

    assert result == 0
    assert "Realtime voice production review OK" in capsys.readouterr().out


def test_production_review_template_refuses_existing_file_without_overwrite(tmp_path, capsys):
    report_path = tmp_path / "review.json"
    report_path.write_text("{}", encoding="utf-8")

    result = realtime_voice_production_review.main([str(report_path), "--write-template"])

    assert result == 1
    assert "already exists" in capsys.readouterr().err
