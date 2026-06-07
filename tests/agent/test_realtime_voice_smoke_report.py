import json

from agent.realtime_voice_smoke_report import (
    ALPHA_REQUIRED_AUDIO_FIXTURES,
    ALPHA_REQUIRED_TTS_TEXTS,
    load_realtime_voice_smoke_report,
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
    return entries


def test_realtime_voice_alpha_report_accepts_required_en_ja_smokes():
    assert validate_realtime_voice_alpha_report(_valid_alpha_report()) == []


def test_realtime_voice_alpha_report_requires_all_required_fixtures_and_phrases():
    report = [
        entry
        for entry in _valid_alpha_report()
        if entry.get("fixture") != "./fixtures/realtime-voice/ja/tool-question.webm"
        and entry.get("text") != "音声で会話できますか？"
    ]

    issues = validate_realtime_voice_alpha_report(report)

    formatted = [issue.format() for issue in issues]
    assert any("missing required fixture" in issue for issue in formatted)
    assert any("missing required text" in issue for issue in formatted)


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
    assert "Realtime voice smoke report OK" in capsys.readouterr().out


def test_realtime_voice_report_cli_returns_nonzero_for_failed_report(tmp_path, capsys):
    path = tmp_path / "voice-smoke.json"
    report = _valid_alpha_report()
    report[0]["ok"] = False
    report[0]["error"] = "protocol failed"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    assert realtime_voice_report_main([str(path), "--alpha"]) == 1
    assert "protocol failed" in capsys.readouterr().err
