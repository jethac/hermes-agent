import json

from hermes_cli import realtime_voice_dgx_report


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _alpha_report(*, first_audio_ms):
    return [
        {"kind": "manifest", "ok": True, "run_id": "dgx-report-test-0001"},
        {
            "kind": "session_turn",
            "ok": True,
            "text": "Hello from Hermes.",
            "first_audio_ms": first_audio_ms,
            "first_text_ms": 120,
            "route": "local",
            "reflex_provider": "vllm",
            "interface_input_source": "native_audio",
        },
        {
            "kind": "barge_in",
            "ok": True,
            "text": "Hello from Hermes.",
            "barge_in_ack_ms": 45,
        },
    ]


def test_dgx_voice_report_prefers_local_when_within_cartesia_latency(tmp_path):
    _write_json(
        tmp_path / "oracle-gemma4-probe.json",
        {
            "ok": True,
            "model": "gemma-4-26B-A4B-it",
            "base_url": "http://spark.local:8001/v1",
            "elapsed_ms": 900,
            "tokens_per_second": 32,
        },
    )
    _write_json(tmp_path / "cartesia-alpha" / "cartesia-001.json", _alpha_report(first_audio_ms=1000))
    _write_json(tmp_path / "local-speech-alpha" / "local-001.json", _alpha_report(first_audio_ms=1180))

    report = realtime_voice_dgx_report.build_dgx_voice_recommendation_report(artifact_dir=tmp_path)

    assert report["recommendation"]["decision"] == "prefer_local_speech"
    assert report["tracks"]["oracle"]["status"] == "passed"
    assert report["tracks"]["cartesia"]["latency_ms"]["first_audio_p50"] == 1000
    assert report["tracks"]["local_speech"]["latency_ms"]["first_audio_p50"] == 1180


def test_dgx_voice_report_keeps_cartesia_when_local_is_too_slow(tmp_path):
    _write_json(
        tmp_path / "oracle-gemma4-probe.json",
        {"ok": True, "elapsed_ms": 900, "tokens_per_second": 32},
    )
    _write_json(tmp_path / "cartesia-alpha" / "cartesia-001.json", _alpha_report(first_audio_ms=1000))
    _write_json(tmp_path / "local-speech-alpha" / "local-001.json", _alpha_report(first_audio_ms=1800))

    report = realtime_voice_dgx_report.build_dgx_voice_recommendation_report(artifact_dir=tmp_path)

    assert report["recommendation"]["decision"] == "keep_cartesia_baseline"
    assert "slower than" in report["recommendation"]["reason"]


def test_dgx_voice_report_requires_oracle_before_voice_frontend_choice(tmp_path):
    _write_json(
        tmp_path / "oracle-gemma4-probe.json",
        {"ok": False, "error": "connection refused"},
    )
    _write_json(tmp_path / "cartesia-alpha" / "cartesia-001.json", _alpha_report(first_audio_ms=1000))

    report = realtime_voice_dgx_report.build_dgx_voice_recommendation_report(artifact_dir=tmp_path)

    assert report["recommendation"]["decision"] == "fix_oracle_first"
    assert report["tracks"]["oracle"]["status"] == "failed"


def test_dgx_voice_report_main_writes_json_and_markdown(tmp_path):
    _write_json(
        tmp_path / "oracle-gemma4-probe.json",
        {"ok": True, "elapsed_ms": 900, "tokens_per_second": 32},
    )
    output = tmp_path / "recommendation.json"
    markdown = tmp_path / "recommendation.md"

    result = realtime_voice_dgx_report.main(
        [
            "--artifact-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--markdown-output",
            str(markdown),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert payload["recommendation"]["decision"] == "collect_voice_frontend_evidence"
    assert "DGX Spark KAME Voice Recommendation" in markdown.read_text(encoding="utf-8")
