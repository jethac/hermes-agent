import json

from agent.realtime_voice_smoke_report import (
    ALPHA_REQUIRED_AUDIO_FIXTURES,
    ALPHA_REQUIRED_BARGE_IN_TEXTS,
    ALPHA_REQUIRED_TTS_TEXTS,
    load_realtime_voice_smoke_report,
    summarize_realtime_voice_smoke_report_runs,
    validate_realtime_voice_alpha_report_runs,
    validate_realtime_voice_alpha_report,
    validate_realtime_voice_smoke_report,
)
from hermes_cli.realtime_voice_report import main as realtime_voice_report_main


def _valid_alpha_report():
    entries = [
        {
            "kind": "protocol",
            "ok": True,
            "ready_ms": 12,
            "transcript_final_ms": 25,
            "events": ["frontend.state", "transcript.final"],
            "error": None,
        }
    ]
    for fixture in ALPHA_REQUIRED_AUDIO_FIXTURES:
        entries.append(
            {
                "kind": "audio_fixture",
                "ok": True,
                "fixture": fixture,
                "codec": "webm_opus",
                "audio_bytes": 1234,
                "transcript_partial_ms": 90,
                "transcript_final_ms": 180,
                "target_ms": 300,
                "events": ["frontend.state", "transcript.partial", "transcript.final"],
                "error": None,
            }
        )
    for text in ALPHA_REQUIRED_TTS_TEXTS:
        entries.append(
            {
                "kind": "tts",
                "ok": True,
                "text": text,
                "first_audio_ms": 250,
                "target_ms": 900,
                "output_audio_bytes": 4321,
                "events": ["frontend.state", "audio.output.chunk"],
                "error": None,
            }
        )
    for text in ALPHA_REQUIRED_BARGE_IN_TEXTS:
        entries.append(
            {
                "kind": "barge_in",
                "ok": True,
                "text": text,
                "barge_in_ack_ms": 45,
                "target_ms": 150,
                "events": ["frontend.state", "barge_in"],
                "error": None,
            }
        )
    return entries


def test_realtime_voice_alpha_report_accepts_required_en_ja_smokes():
    assert validate_realtime_voice_alpha_report(_valid_alpha_report()) == []


def test_realtime_voice_alpha_report_requires_all_required_fixtures_and_phrases():
    report = [
        entry
        for entry in _valid_alpha_report()
        if entry.get("fixture") != "./fixtures/realtime-voice/ja/tool-question.webm"
        and entry.get("text") != "音声で会話できますか？"
        and entry.get("kind") != "barge_in"
    ]

    issues = validate_realtime_voice_alpha_report(report)

    formatted = [issue.format() for issue in issues]
    assert any("missing required fixture" in issue for issue in formatted)
    assert any("missing required text" in issue for issue in formatted)


def test_realtime_voice_alpha_report_rejects_barge_in_target_misses():
    report = _valid_alpha_report()
    report[-1]["barge_in_ack_ms"] = 250

    issues = validate_realtime_voice_alpha_report(report)

    assert any("barge_in_ack_ms 250 exceeds target 150" in issue.format() for issue in issues)


def test_realtime_voice_smoke_report_rejects_latency_target_misses():
    report = _valid_alpha_report()
    report[1]["transcript_partial_ms"] = 450

    issues = validate_realtime_voice_smoke_report(report)

    assert any("exceeds target 300" in issue.format() for issue in issues)


def test_load_realtime_voice_smoke_report_round_trips_unicode(tmp_path):
    path = tmp_path / "voice-smoke.json"
    expected = _valid_alpha_report()
    path.write_text(json.dumps(expected, ensure_ascii=False), encoding="utf-8")

    assert load_realtime_voice_smoke_report(path) == expected


def test_realtime_voice_report_cli_validates_alpha_report(tmp_path, capsys):
    path = tmp_path / "voice-smoke.json"
    path.write_text(json.dumps(_valid_alpha_report(), ensure_ascii=False), encoding="utf-8")

    assert realtime_voice_report_main([str(path), "--alpha"]) == 0
    output = capsys.readouterr().out
    assert "Realtime voice smoke report OK" in output
    assert "audio_to_partial_transcript: p50=90ms p95=90ms max=90ms n=4" in output
    assert "barge_in_ack: p50=45ms p95=45ms max=45ms n=1" in output


def test_realtime_voice_report_cli_enforces_minimum_alpha_runs(tmp_path, capsys):
    path = tmp_path / "voice-smoke.json"
    path.write_text(json.dumps(_valid_alpha_report(), ensure_ascii=False), encoding="utf-8")

    assert realtime_voice_report_main([str(path), "--alpha", "--min-runs", "3"]) == 1
    assert "requires at least 3 run(s), found 1" in capsys.readouterr().err


def test_realtime_voice_alpha_report_runs_accept_multiple_reports(tmp_path):
    runs = []
    for index in range(3):
        path = tmp_path / f"voice-smoke-{index}.json"
        path.write_text(json.dumps(_valid_alpha_report(), ensure_ascii=False), encoding="utf-8")
        runs.append((str(path), load_realtime_voice_smoke_report(path)))

    assert validate_realtime_voice_alpha_report_runs(runs, min_runs=3) == []


def test_realtime_voice_report_run_summary_counts_latency_distributions(tmp_path):
    runs = []
    for index, partial_ms in enumerate((80, 90, 120)):
        report = _valid_alpha_report()
        for entry in report:
            if entry.get("kind") == "audio_fixture":
                entry["transcript_partial_ms"] = partial_ms
        path = tmp_path / f"voice-smoke-{index}.json"
        path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        runs.append((str(path), load_realtime_voice_smoke_report(path)))

    summary = summarize_realtime_voice_smoke_report_runs(runs)

    assert summary["runs"] == 3
    assert summary["entries"] == 30
    assert summary["kinds"]["audio_fixture"] == {"entries": 12, "ok": 12, "failed": 0}
    assert summary["latency_ms"]["audio_to_partial_transcript"] == {
        "count": 12,
        "p50": 90,
        "p95": 120,
        "max": 120,
    }


def test_realtime_voice_report_cli_returns_nonzero_for_failed_report(tmp_path, capsys):
    path = tmp_path / "voice-smoke.json"
    report = _valid_alpha_report()
    report[0]["ok"] = False
    report[0]["error"] = "protocol failed"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    assert realtime_voice_report_main([str(path), "--alpha"]) == 1
    assert "protocol failed" in capsys.readouterr().err
