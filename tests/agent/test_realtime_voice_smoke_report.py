import json
from datetime import datetime, timezone
from itertools import count

from agent.realtime_voice_smoke_report import (
    ALPHA_REQUIRED_AUDIO_FIXTURES,
    ALPHA_REQUIRED_AUDIO_FIXTURE_TEXTS,
    ALPHA_REQUIRED_AUDIO_SESSION_FIXTURES,
    ALPHA_REQUIRED_BARGE_IN_TEXTS,
    ALPHA_REQUIRED_SESSION_TURN_TEXTS,
    ALPHA_REQUIRED_SESSION_TURN_METADATA,
    ALPHA_REQUIRED_TTS_METADATA,
    ALPHA_REQUIRED_TTS_TEXTS,
    load_realtime_voice_smoke_report,
    summarize_realtime_voice_smoke_report_runs,
    validate_realtime_voice_alpha_report_runs,
    validate_realtime_voice_alpha_report,
    validate_realtime_voice_smoke_report,
)
from hermes_cli.realtime_voice_report import main as realtime_voice_report_main


_RUN_ID_COUNTER = count(1)


def _next_run_id():
    return f"test-run-{next(_RUN_ID_COUNTER):04d}"


def _valid_manifest():
    return {
        "kind": "manifest",
        "ok": True,
        "run_id": _next_run_id(),
        "collected_at": "2026-06-08T00:00:00Z",
        "available": True,
        "conversation_quality": {
            "live_like": True,
            "mode": "streaming_text",
            "reason": "streaming_stt_tts",
            "sidecar_verified": True,
        },
        "quality_targets_ms": {
            "audio_to_partial_transcript_ms": 300,
            "final_transcript_to_first_text_ms": 500,
            "final_transcript_to_first_audio_ms": 900,
            "barge_in_ack_ms": 150,
        },
        "sidecar": {
            "healthy": True,
            "health": {
                "ok": True,
                "capabilities": {
                    "streaming_stt": True,
                    "streaming_tts": True,
                    "tts": True,
                    "native_s2s": False,
                    "output_languages": ["en", "ja"],
                }
            },
        },
    }


def _valid_alpha_report():
    entries = [
        _valid_manifest(),
        {
            "kind": "protocol",
            "ok": True,
            "ready_ms": 12,
            "transcript_final_ms": 25,
            "events": ["frontend.state", "transcript.final"],
            "error": None,
        }
    ]
    for text in ALPHA_REQUIRED_SESSION_TURN_TEXTS:
        entries.append(
            {
                "kind": "session_turn",
                "ok": True,
                "text": text,
                **ALPHA_REQUIRED_SESSION_TURN_METADATA[text],
                "transcript_final_ms": 10,
                "first_text_ms": 90,
                "first_text_target_ms": 500,
                "first_audio_ms": 250,
                "first_audio_target_ms": 900,
                "output_audio_bytes": 4321,
                "events": [
                    "session.started",
                    "transcript.final",
                    "assistant.text.partial",
                    "audio.output.chunk",
                ],
                "error": None,
            }
        )
    for fixture in ALPHA_REQUIRED_AUDIO_FIXTURES:
        entries.append(
            {
                "kind": "audio_fixture",
                "ok": True,
                "fixture": fixture,
                "codec": "webm_opus",
                "audio_bytes": 1234,
                "final_text": ALPHA_REQUIRED_AUDIO_FIXTURE_TEXTS[fixture],
                "transcript_partial_ms": 90,
                "transcript_final_ms": 180,
                "target_ms": 300,
                "events": ["frontend.state", "transcript.partial", "transcript.final"],
                "error": None,
            }
        )
    for fixture in ALPHA_REQUIRED_AUDIO_SESSION_FIXTURES:
        entries.append(
            {
                "kind": "audio_session",
                "ok": True,
                "fixture": fixture,
                "codec": "webm_opus",
                "audio_bytes": 1234,
                "final_text": ALPHA_REQUIRED_AUDIO_FIXTURE_TEXTS[fixture],
                "transcript_partial_ms": 90,
                "transcript_final_ms": 180,
                "target_ms": 300,
                "first_text_ms": 90,
                "first_text_target_ms": 500,
                "first_audio_ms": 250,
                "first_audio_target_ms": 900,
                "output_audio_bytes": 4321,
                "events": [
                    "session.started",
                    "frontend.state",
                    "transcript.partial",
                    "transcript.final",
                    "assistant.text.partial",
                    "audio.output.chunk",
                ],
                "error": None,
            }
        )
    for text in ALPHA_REQUIRED_TTS_TEXTS:
        metadata = ALPHA_REQUIRED_TTS_METADATA[text]
        entries.append(
            {
                "kind": "tts",
                "ok": True,
                "text": text,
                **metadata,
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
                "audio_after_barge_in_bytes": 0,
                "target_ms": 150,
                "events": ["frontend.state", "barge_in"],
                "error": None,
            }
        )
    return entries


def test_realtime_voice_alpha_report_accepts_required_en_ja_smokes():
    assert validate_realtime_voice_alpha_report(_valid_alpha_report()) == []


def test_realtime_voice_alpha_report_accepts_manifest_entry():
    report = [
        _valid_manifest(),
        *_valid_alpha_report(),
    ]

    assert validate_realtime_voice_alpha_report(report) == []


def test_realtime_voice_alpha_report_requires_manifest_entry():
    report = [
        entry
        for entry in _valid_alpha_report()
        if entry.get("kind") != "manifest"
    ]

    issues = validate_realtime_voice_alpha_report(report)

    assert any("missing manifest entry" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_requires_live_like_manifest():
    manifest = _valid_manifest()
    manifest["conversation_quality"] = {
        "live_like": False,
        "mode": "turn_based_text",
        "reason": "utterance_stt_tts",
    }
    report = [manifest, *_valid_alpha_report()[1:]]

    issues = validate_realtime_voice_alpha_report(report)

    assert any("manifest was not live-like" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_rejects_loose_manifest_quality_targets():
    manifest = _valid_manifest()
    manifest["quality_targets_ms"]["final_transcript_to_first_audio_ms"] = 1200
    report = [manifest, *_valid_alpha_report()[1:]]

    issues = validate_realtime_voice_alpha_report(report)

    assert any("quality target final_transcript_to_first_audio_ms 1200 exceeds alpha ceiling 900" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_requires_live_sidecar_manifest_capabilities():
    manifest = _valid_manifest()
    manifest["sidecar"]["health"]["capabilities"] = {
        "utterance_stt": True,
        "tts": True,
        "output_languages": ["en", "ja"],
    }
    report = [manifest, *_valid_alpha_report()[1:]]

    issues = validate_realtime_voice_alpha_report(report)

    assert any("missing native_s2s or streaming_stt+tts" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_requires_sidecar_health_ok():
    manifest = _valid_manifest()
    manifest["sidecar"]["health"]["ok"] = False
    report = [manifest, *_valid_alpha_report()[1:]]

    issues = validate_realtime_voice_alpha_report(report)

    assert any("manifest sidecar health was not ok" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_requires_manifest_run_identity():
    manifest = _valid_manifest()
    manifest.pop("run_id")
    manifest.pop("collected_at")
    report = [manifest, *_valid_alpha_report()[1:]]

    issues = validate_realtime_voice_alpha_report(report)

    formatted = [issue.format() for issue in issues]
    assert any("missing valid evidence run_id" in issue for issue in formatted)
    assert any("missing collected_at timestamp" in issue for issue in formatted)


def test_realtime_voice_alpha_report_requires_manifest_output_language_evidence():
    manifest = _valid_manifest()
    manifest["sidecar"]["health"]["capabilities"]["output_languages"] = ["en"]
    report = [
        manifest,
        *_valid_alpha_report()[1:],
    ]

    issues = validate_realtime_voice_alpha_report(report)

    assert any("missing TTS model route for ja" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_accepts_manifest_regional_output_languages():
    manifest = _valid_manifest()
    manifest["sidecar"]["health"]["frontend"] = {"tts_model_languages": ["en-US", "ja-JP"]}
    manifest["sidecar"]["health"]["capabilities"]["output_languages"] = []
    report = [
        manifest,
        *_valid_alpha_report()[1:],
    ]

    assert validate_realtime_voice_alpha_report(report) == []


def test_realtime_voice_alpha_report_requires_all_required_fixtures_and_phrases():
    report = [
        entry
        for entry in _valid_alpha_report()
        if entry.get("fixture") != "./fixtures/realtime-voice/ja/tool-question.webm"
        and entry.get("text") != "音声で会話できますか？"
        and entry.get("kind") != "barge_in"
        and entry.get("kind") != "session_turn"
    ]

    issues = validate_realtime_voice_alpha_report(report)

    formatted = [issue.format() for issue in issues]
    assert any("missing required fixture" in issue for issue in formatted)
    assert any("missing required text" in issue for issue in formatted)


def test_realtime_voice_alpha_report_requires_session_turn_smoke():
    report = [entry for entry in _valid_alpha_report() if entry.get("kind") != "session_turn"]

    issues = validate_realtime_voice_alpha_report(report)

    assert any("session_turn: missing required text" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_requires_audio_session_smoke():
    report = [entry for entry in _valid_alpha_report() if entry.get("kind") != "audio_session"]

    issues = validate_realtime_voice_alpha_report(report)

    assert any("audio_session: missing required fixture" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_rejects_wrong_audio_session_final_text():
    report = _valid_alpha_report()
    for entry in report:
        if entry.get("kind") == "audio_session" and entry.get("fixture") == "./fixtures/realtime-voice/ja/hello.webm":
            entry["final_text"] = "こんにちは、別の人です。"
            break

    issues = validate_realtime_voice_alpha_report(report)

    assert any("final_text did not match expected fixture transcript" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_rejects_slow_audio_session_first_audio():
    report = _valid_alpha_report()
    for entry in report:
        if entry.get("kind") == "audio_session":
            entry["first_audio_ms"] = 950
            break

    issues = validate_realtime_voice_alpha_report(report)

    assert any("audio_session: first_audio_ms 950 exceeds target 900" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_rejects_slow_session_first_text():
    report = _valid_alpha_report()
    for entry in report:
        if entry.get("kind") == "session_turn":
            entry["first_text_ms"] = 650
            break

    issues = validate_realtime_voice_alpha_report(report)

    assert any("first_text_ms 650 exceeds target 500" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_requires_session_turn_language_metadata():
    report = _valid_alpha_report()
    for entry in report:
        if entry.get("kind") == "session_turn" and entry.get("text") == "こんにちは、Hermesです。":
            entry.pop("language")
            break

    issues = validate_realtime_voice_alpha_report(report)

    assert any("missing language=ja metadata" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_requires_tts_language_metadata():
    report = _valid_alpha_report()
    for entry in report:
        if entry.get("kind") == "tts" and entry.get("text") == "こんにちは、Hermesです。":
            entry.pop("language", None)
            break

    issues = validate_realtime_voice_alpha_report(report)

    assert any("missing language=ja metadata" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_requires_audio_fixture_final_text():
    report = _valid_alpha_report()
    for entry in report:
        if entry.get("fixture") == "./fixtures/realtime-voice/en/hello.webm":
            entry["final_text"] = ""
            break

    issues = validate_realtime_voice_alpha_report(report)

    assert any("missing final_text for required fixture" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_rejects_wrong_audio_fixture_final_text():
    report = _valid_alpha_report()
    for entry in report:
        if entry.get("fixture") == "./fixtures/realtime-voice/ja/hello.webm":
            entry["final_text"] = "こんにちは、別の人です。"
            break

    issues = validate_realtime_voice_alpha_report(report)

    assert any("final_text did not match expected fixture transcript" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_accepts_normalized_audio_fixture_text():
    report = _valid_alpha_report()
    for entry in report:
        if entry.get("fixture") == "./fixtures/realtime-voice/en/tool-question.webm":
            entry["final_text"] = "what files can hermes see in this workspace"
            break

    assert validate_realtime_voice_alpha_report(report) == []


def test_realtime_voice_alpha_report_rejects_barge_in_target_misses():
    report = _valid_alpha_report()
    report[-1]["barge_in_ack_ms"] = 250

    issues = validate_realtime_voice_alpha_report(report)

    assert any("barge_in_ack_ms 250 exceeds target 150" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_rejects_audio_after_barge_in():
    report = _valid_alpha_report()
    report[-1]["audio_after_barge_in_bytes"] = 128
    report[-1]["events"] = ["frontend.state", "barge_in", "audio.output.chunk"]

    issues = validate_realtime_voice_alpha_report(report)

    assert any("audio.output.chunk arrived after barge_in (128 byte(s))" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_requires_barge_in_audio_quiet_field():
    report = _valid_alpha_report()
    report[-1].pop("audio_after_barge_in_bytes")

    issues = validate_realtime_voice_alpha_report(report)

    assert any("missing audio_after_barge_in_bytes" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_rejects_loose_entry_targets():
    report = _valid_alpha_report()
    for entry in report:
        if entry.get("kind") == "tts":
            entry["first_audio_ms"] = 950
            entry["target_ms"] = 1200
            break

    issues = validate_realtime_voice_alpha_report(report)

    assert any("target_ms 1200 exceeds alpha ceiling 900" in issue.format() for issue in issues)


def test_realtime_voice_smoke_report_rejects_latency_target_misses():
    report = _valid_alpha_report()
    for entry in report:
        if entry.get("kind") == "audio_fixture":
            entry["transcript_partial_ms"] = 450
            break

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
    assert "audio_to_partial_transcript: p50=90ms p95=90ms max=90ms n=6" in output
    assert "final_transcript_to_first_text: p50=90ms p95=90ms max=90ms n=4" in output
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


def test_realtime_voice_alpha_report_runs_reject_mixed_stack_manifests(tmp_path):
    runs = []
    for index in range(3):
        report = json.loads(json.dumps(_valid_alpha_report(), ensure_ascii=False))
        if index == 1:
            manifest = report[0]
            manifest["engine"] = "native_s2s_oracle"
            manifest["frontend_provider"] = "native_s2s"
            manifest["conversation_quality"]["mode"] = "native_s2s"
            manifest["sidecar"]["mode"] = "external"
            manifest["sidecar"]["health"]["capabilities"] = {
                "native_s2s": True,
                "output_languages": ["en", "ja"],
            }
        path = tmp_path / f"voice-smoke-{index}.json"
        path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        runs.append((str(path), load_realtime_voice_smoke_report(path)))

    issues = validate_realtime_voice_alpha_report_runs(runs, min_runs=3)

    assert any("mixed realtime voice stack manifests" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_runs_reject_duplicate_run_ids(tmp_path):
    runs = []
    for index in range(3):
        report = _valid_alpha_report()
        report[0]["run_id"] = "duplicated-run-id"
        path = tmp_path / f"voice-smoke-{index}.json"
        path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        runs.append((str(path), load_realtime_voice_smoke_report(path)))

    issues = validate_realtime_voice_alpha_report_runs(runs, min_runs=3)

    assert any("alpha runs reused evidence run_id" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_runs_reject_stale_evidence_when_max_age_is_set(tmp_path):
    runs = []
    for index in range(3):
        report = _valid_alpha_report()
        report[0]["collected_at"] = "2026-05-01T00:00:00Z"
        path = tmp_path / f"voice-smoke-{index}.json"
        path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        runs.append((str(path), load_realtime_voice_smoke_report(path)))

    issues = validate_realtime_voice_alpha_report_runs(
        runs,
        min_runs=3,
        max_collected_age_days=14,
        now=datetime(2026, 6, 8, tzinfo=timezone.utc),
    )

    assert any("alpha run evidence is older than 14 day(s)" in issue.format() for issue in issues)


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
    assert summary["entries"] == 42
    assert summary["kinds"]["audio_fixture"] == {"entries": 12, "ok": 12, "failed": 0}
    assert summary["kinds"]["audio_session"] == {"entries": 6, "ok": 6, "failed": 0}
    assert summary["kinds"]["session_turn"] == {"entries": 6, "ok": 6, "failed": 0}
    assert summary["latency_ms"]["audio_to_partial_transcript"] == {
        "count": 18,
        "p50": 90,
        "p95": 120,
        "max": 120,
    }
    assert summary["latency_ms"]["final_transcript_to_first_text"] == {
        "count": 12,
        "p50": 90,
        "p95": 90,
        "max": 90,
    }


def test_realtime_voice_report_cli_returns_nonzero_for_failed_report(tmp_path, capsys):
    path = tmp_path / "voice-smoke.json"
    report = _valid_alpha_report()
    for entry in report:
        if entry.get("kind") == "protocol":
            entry["ok"] = False
            entry["error"] = "protocol failed"
            break
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    assert realtime_voice_report_main([str(path), "--alpha"]) == 1
    assert "protocol failed" in capsys.readouterr().err
