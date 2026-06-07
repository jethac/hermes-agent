import json
import types

from agent.realtime_voice_smoke_report import (
    ALPHA_REQUIRED_AUDIO_FIXTURES,
    ALPHA_REQUIRED_BARGE_IN_TEXTS,
    ALPHA_REQUIRED_TTS_TEXTS,
)
from hermes_cli import realtime_voice_alpha_evidence


def _valid_alpha_report():
    entries = [
        {
            "kind": "manifest",
            "ok": True,
        },
        {
            "kind": "protocol",
            "ok": True,
            "ready_ms": 12,
            "transcript_final_ms": 25,
            "events": ["frontend.state", "transcript.final"],
        },
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
            }
        )
    return entries


def test_alpha_evidence_runner_collects_and_validates_runs(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run_doctor(args):
        calls.append(args)
        with open(args.realtime_voice_report, "w", encoding="utf-8") as handle:
            json.dump(_valid_alpha_report(), handle, ensure_ascii=False)

    monkeypatch.setitem(__import__("sys").modules, "hermes_cli.doctor", types.SimpleNamespace(run_doctor=fake_run_doctor))

    result = realtime_voice_alpha_evidence.main(
        [
            "--output-dir",
            str(tmp_path),
            "--runs",
            "3",
            "--prefix",
            "alpha",
        ]
    )

    assert result == 0
    assert [path.name for path in sorted(tmp_path.glob("*.json"))] == [
        "alpha-001.json",
        "alpha-002.json",
        "alpha-003.json",
    ]
    assert len(calls) == 3
    assert all(call.realtime_voice is True for call in calls)
    assert all(call.realtime_voice_alpha is True for call in calls)
    assert all(call.realtime_voice_audio_codec == "webm_opus" for call in calls)
    output = capsys.readouterr().out
    assert "Realtime voice alpha evidence OK" in output
    assert "audio_to_partial_transcript" in output


def test_alpha_evidence_runner_refuses_to_overwrite_existing_report(tmp_path, capsys):
    (tmp_path / "alpha-001.json").write_text("[]", encoding="utf-8")

    result = realtime_voice_alpha_evidence.main(
        [
            "--output-dir",
            str(tmp_path),
            "--runs",
            "1",
            "--prefix",
            "alpha",
        ]
    )

    assert result == 1
    assert "already exists" in capsys.readouterr().err


def test_alpha_evidence_runner_returns_validation_failure(monkeypatch, tmp_path, capsys):
    def fake_run_doctor(args):
        report = _valid_alpha_report()
        report = [entry for entry in report if entry.get("kind") != "barge_in"]
        with open(args.realtime_voice_report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False)

    monkeypatch.setitem(__import__("sys").modules, "hermes_cli.doctor", types.SimpleNamespace(run_doctor=fake_run_doctor))

    result = realtime_voice_alpha_evidence.main(
        [
            "--output-dir",
            str(tmp_path),
            "--runs",
            "1",
            "--prefix",
            "alpha",
        ]
    )

    assert result == 1
    assert "missing required text" in capsys.readouterr().err
