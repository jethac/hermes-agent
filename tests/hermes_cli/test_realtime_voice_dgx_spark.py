import json
import argparse
from pathlib import Path

from hermes_cli import realtime_voice_dgx_spark
from hermes_cli.subcommands.voice import build_voice_parser


def _manifest(tmp_path: Path) -> dict:
    return realtime_voice_dgx_spark.build_dgx_spark_stack_manifest(
        repo_dir=tmp_path / "repo",
        hermes_home=tmp_path / "home",
        interface_base_url="http://spark.local:8000/v1",
        interface_model="gemma-4-E2B-it",
        interface_context_tokens=8192,
        interface_gpu_memory_utilization=0.18,
        oracle_base_url="http://spark.local:8001/v1",
        oracle_model="gemma-4-26B-A4B-it",
        oracle_context_tokens=32768,
        oracle_gpu_memory_utilization=0.62,
        sidecar_base_url="http://spark.local:8765",
        asr_base_url="http://spark.local:8767",
        tts_base_url="http://spark.local:8768",
        asr_mode="on_escalation",
        vllm_image="vllm/vllm-openai:gemma4-cu130",
        hermes_image="ghcr.io/astral-sh/uv:python3.12-bookworm-slim",
        model_cache_dir="/models",
    )


def _passing_benchmark_evidence() -> list[dict]:
    return [
        {
            "kind": "kame_benchmark_result",
            "category": "interface",
            "input": "direct_audio",
            "metrics": {
                "speech_end_to_interface_decision_ms": 320,
                "speech_end_to_local_first_audio_ms": 480,
                "routing_accuracy": 0.94,
            },
        },
        {
            "kind": "kame_benchmark_result",
            "category": "interface",
            "input": "stt_fallback",
            "metrics": {
                "speech_end_to_transcript_ms": 190,
                "transcript_to_interface_decision_ms": 280,
                "routing_accuracy": 0.91,
            },
        },
        {
            "kind": "kame_benchmark_result",
            "category": "oracle",
            "metrics": {
                "oracle_request_to_accepted_ms": 40,
                "oracle_accepted_to_first_token_ms": 780,
                "oracle_first_token_to_first_audio_ms": 220,
            },
        },
        {
            "kind": "kame_benchmark_result",
            "category": "oracle_outcome",
            "asr_hypothesis": "without_asr_hypothesis",
            "metrics": {
                "task_success_rate": 0.78,
                "literal_argument_accuracy": 0.72,
                "tool_argument_error_rate": 0.21,
            },
        },
        {
            "kind": "kame_benchmark_result",
            "category": "oracle_outcome",
            "asr_hypothesis": "with_asr_hypothesis",
            "metrics": {
                "task_success_rate": 0.84,
                "literal_argument_accuracy": 0.9,
                "tool_argument_error_rate": 0.08,
            },
        },
        {
            "kind": "kame_benchmark_result",
            "category": "speech",
            "role": "oracle_verbatim_asr",
            "metrics": {
                "speech_end_to_asr_final_ms": 110,
                "literal_accuracy_names_numbers_code": 0.88,
            },
        },
        {
            "kind": "kame_benchmark_result",
            "category": "speech",
            "role": "tts",
            "metrics": {
                "tts_request_to_first_audio_ms": 160,
                "tts_request_to_audio_end_ms": 620,
            },
        },
        {"kind": "kame_smoke_result", "name": "all_local_smoke", "ok": True},
        {"kind": "kame_smoke_result", "name": "cloud_fallback_smoke", "ok": True},
    ]


def test_manifest_describes_full_kame_dgx_spark_stack(tmp_path):
    manifest = _manifest(tmp_path)

    assert manifest["kind"] == "kame_dgx_spark_stack"
    assert manifest["target"]["hardware"] == "1x DGX Spark"
    assert manifest["engine"]["name"] == "kame_interface_oracle"
    assert manifest["engine"]["interface_audio_input"] == "native_audio"
    assert manifest["engine"]["asr_mode"] == "on_escalation"
    assert manifest["engine"]["max_spoken_sentences"] == 2
    assert manifest["roles"]["interface"]["model"] == "gemma-4-E2B-it"
    assert manifest["roles"]["interface"]["limit_mm_per_prompt"] == {"audio": 1}
    assert manifest["roles"]["oracle"]["preferred_local_model"] == "gemma-4-26B-A4B-it"
    assert manifest["roles"]["asr"]["role"] == "oracle_verbatim_evidence"
    assert manifest["roles"]["asr"]["default_adapter"] == "loopback_smoke_bridge"
    assert manifest["roles"]["asr"]["production_replacement"] == "local_streaming_asr"
    assert manifest["roles"]["asr"]["feeds_reflex"] is False
    assert manifest["roles"]["tts"]["default_adapter"] == "loopback_smoke_bridge"
    assert manifest["roles"]["tts"]["production_replacement"] == "local_streaming_tts"
    assert "all_local_smoke" in manifest["evidence_required"]


def test_rendered_compose_has_reflex_oracle_and_sidecar_without_secret_material(tmp_path):
    compose = realtime_voice_dgx_spark.render_dgx_spark_compose(_manifest(tmp_path))

    assert "kame-interface-vllm:" in compose
    assert "kame-oracle-vllm:" in compose
    assert "hermes-realtime-sidecar:" in compose
    assert "kame-asr-bridge:" in compose
    assert "kame-tts-bridge:" in compose
    assert "gemma-4-E2B-it" in compose
    assert "gemma-4-26B-A4B-it" in compose
    assert "--limit-mm-per-prompt" in compose
    assert '{"audio":1}' in compose
    assert "HERMES_VOICE_STREAMING_STT_BASE_URL" in compose
    assert "oracle-verbatim-asr" in compose
    assert "hermes_cli.realtime_voice_loopback_bridge" in compose
    assert "API_KEY" not in compose
    assert "sk_" not in compose


def test_writer_emits_headless_artifact_pack(tmp_path):
    output_dir = tmp_path / "out"
    written = realtime_voice_dgx_spark.write_dgx_spark_stack_artifacts(
        output_dir,
        _manifest(tmp_path),
    )

    assert set(written) == {
        "manifest",
        "compose",
        "env_example",
        "launch",
        "benchmark_matrix",
        "benchmark_evidence_template",
    }
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "compose.yaml").is_file()
    assert (output_dir / ".env.example").is_file()
    assert (output_dir / "launch-local-stack.sh").is_file()
    assert (output_dir / "benchmark-matrix.json").is_file()
    assert (output_dir / "benchmark-evidence-template.json").is_file()
    assert (output_dir / "launch-local-stack.sh").stat().st_mode & 0o111

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    launch = (output_dir / "launch-local-stack.sh").read_text(encoding="utf-8")
    matrix = json.loads((output_dir / "benchmark-matrix.json").read_text(encoding="utf-8"))
    evidence_template = json.loads((output_dir / "benchmark-evidence-template.json").read_text(encoding="utf-8"))
    assert manifest["roles"]["interface"]["audio_input"] == "native_audio"
    assert manifest["engine"]["max_spoken_sentences"] == 2
    assert "HERMES_KAME_MAX_SPOKEN_SENTENCES=2" in (output_dir / ".env.example").read_text(encoding="utf-8")
    assert "HERMES_DGX_SPARK_APPLY_PROFILE" in launch
    assert "hermes_cli.realtime_voice_profile --preset kame --apply" in launch
    assert "--kame-interface-audio-input native_audio" in launch
    assert "--kame-asr-mode on_escalation" in launch
    assert "--sidecar-host spark.local" in launch
    assert "--sidecar-port 8765" in launch
    assert "docker compose --env-file .env.example -f compose.yaml up" in launch
    assert matrix["candidates"]["interface"][0]["input"] == "direct_audio"
    assert matrix["candidates"]["interface"][1]["input"] == "stt_fallback"
    assert matrix["candidates"]["oracle_outcome"][0]["asr_hypothesis"] == "without_asr_hypothesis"
    assert matrix["candidates"]["oracle_outcome"][1]["asr_hypothesis"] == "with_asr_hypothesis"
    assert evidence_template[0]["kind"] == "kame_benchmark_result"
    assert evidence_template[0]["category"] == "interface"
    assert evidence_template[0]["input"] == "direct_audio"
    assert evidence_template[0]["metrics"]["speech_end_to_interface_decision_ms"] is None
    assert evidence_template[-2]["name"] == "all_local_smoke"
    assert evidence_template[-2]["ok"] is False


def test_benchmark_evidence_template_matches_matrix_and_does_not_pass_validation(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path))
    template = realtime_voice_dgx_spark.build_dgx_spark_benchmark_evidence_template(matrix)

    assert [entry["input"] for entry in template if entry.get("category") == "interface"] == [
        "direct_audio",
        "stt_fallback",
    ]
    assert [entry["asr_hypothesis"] for entry in template if entry.get("category") == "oracle_outcome"] == [
        "without_asr_hypothesis",
        "with_asr_hypothesis",
    ]
    assert {entry["role"] for entry in template if entry.get("category") == "speech"} == {
        "oracle_verbatim_asr",
        "tts",
    }

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, template)

    assert result["ok"] is False
    assert "interface:direct_audio: missing or invalid metric speech_end_to_interface_decision_ms" in result["issues"]
    assert "all_local_smoke: missing passing smoke result" in result["issues"]


def test_preflight_checks_openai_models_and_health_urls(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)
    seen_urls: list[str] = []

    class _Response:
        status = 200

        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        seen_urls.append(request.full_url)
        if request.full_url.endswith("/models") and ":8000" in request.full_url:
            return _Response({"data": [{"id": "gemma-4-E2B-it"}]})
        if request.full_url.endswith("/models") and ":8001" in request.full_url:
            return _Response({"data": [{"id": "gemma-4-26B-A4B-it"}]})
        if request.full_url.endswith("/health") and ":8765" in request.full_url:
            return _Response(
                {
                    "ok": True,
                    "capabilities": {
                        "vllm_audio_frontend": True,
                        "tts": True,
                        "streaming_stt_bridge": True,
                        "streaming_tts_bridge": True,
                    },
                    "frontend": {
                        "streaming_stt_bridge": {"healthy": True},
                        "streaming_tts_bridge": {"healthy": True},
                    },
                }
            )
        return _Response({"ok": True})

    monkeypatch.setattr(realtime_voice_dgx_spark.urllib.request, "urlopen", fake_urlopen)

    preflight = realtime_voice_dgx_spark.preflight_dgx_spark_stack(
        manifest,
        timeout_seconds=0.1,
    )

    assert preflight["ok"] is True
    assert "http://spark.local:8000/v1/models" in seen_urls
    assert "http://spark.local:8001/v1/models" in seen_urls
    assert "http://spark.local:8765/health" in seen_urls
    assert "http://spark.local:8767/health" in seen_urls
    assert "http://spark.local:8768/health" in seen_urls
    assert preflight["checks"]["sidecar_health"]["field_misses"] == []


def test_preflight_fails_when_sidecar_lacks_kame_reflex_capability(monkeypatch, tmp_path):
    manifest = _manifest(tmp_path)

    class _Response:
        status = 200

        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/models") and ":8000" in request.full_url:
            return _Response({"data": [{"id": "gemma-4-E2B-it"}]})
        if request.full_url.endswith("/models") and ":8001" in request.full_url:
            return _Response({"data": [{"id": "gemma-4-26B-A4B-it"}]})
        if request.full_url.endswith("/health") and ":8765" in request.full_url:
            return _Response(
                {
                    "ok": True,
                    "capabilities": {
                        "vllm_audio_frontend": False,
                        "tts": True,
                        "streaming_stt_bridge": True,
                        "streaming_tts_bridge": True,
                    },
                    "frontend": {
                        "streaming_stt_bridge": {"healthy": True},
                        "streaming_tts_bridge": {"healthy": True},
                    },
                }
            )
        return _Response({"ok": True})

    monkeypatch.setattr(realtime_voice_dgx_spark.urllib.request, "urlopen", fake_urlopen)

    preflight = realtime_voice_dgx_spark.preflight_dgx_spark_stack(
        manifest,
        timeout_seconds=0.1,
    )

    assert preflight["ok"] is False
    assert preflight["checks"]["sidecar_health"]["ok"] is False
    assert {
        "path": "capabilities.vllm_audio_frontend",
        "expected": True,
        "actual": False,
    } in preflight["checks"]["sidecar_health"]["field_misses"]


def test_benchmark_evidence_validator_accepts_complete_comparison_matrix(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path))

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(
        matrix,
        _passing_benchmark_evidence(),
    )

    assert result["ok"] is True
    assert result["issues"] == []
    assert result["coverage"]["interface_direct_audio_vs_stt_fallback"] is True
    assert result["coverage"]["oracle_outcomes_with_and_without_asr_hypotheses"] is True
    assert result["coverage"]["oracle_verbatim_asr_latency_and_literal_accuracy"] is True
    assert result["coverage"]["local_asr_tts_benchmark_matrix"] is True
    assert result["coverage"]["all_local_smoke"] is True
    assert result["coverage"]["cloud_fallback_smoke"] is True


def test_benchmark_evidence_validator_requires_stt_fallback_and_smoke(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path))
    evidence = [
        entry
        for entry in _passing_benchmark_evidence()
        if entry.get("input") != "stt_fallback" and entry.get("name") != "cloud_fallback_smoke"
    ]

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert "interface:stt_fallback: missing benchmark result" in result["issues"]
    assert "interface_direct_audio_vs_stt_fallback: requires direct_audio and stt_fallback results" in result["issues"]
    assert "cloud_fallback_smoke: missing passing smoke result" in result["issues"]


def test_benchmark_evidence_validator_requires_oracle_asr_outcome_comparison(tmp_path):
    matrix = realtime_voice_dgx_spark.build_dgx_spark_benchmark_matrix(_manifest(tmp_path))
    evidence = [
        entry
        for entry in _passing_benchmark_evidence()
        if entry.get("asr_hypothesis") != "with_asr_hypothesis"
    ]

    result = realtime_voice_dgx_spark.validate_dgx_spark_benchmark_evidence(matrix, evidence)

    assert result["ok"] is False
    assert "oracle_outcome:with_asr_hypothesis: missing benchmark result" in result["issues"]
    assert (
        "oracle_outcomes_with_and_without_asr_hypotheses: "
        "requires with_asr_hypothesis and without_asr_hypothesis results"
    ) in result["issues"]


def test_main_validates_benchmark_evidence_file(tmp_path, capsys):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(_passing_benchmark_evidence()), encoding="utf-8")

    exit_code = realtime_voice_dgx_spark.main(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--repo-dir",
            str(tmp_path / "repo"),
            "--hermes-home",
            str(tmp_path / "home"),
            "--benchmark-evidence",
            str(evidence_path),
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert result["ok"] is True
    assert result["benchmark_evidence"]["ok"] is True


def test_main_fails_on_incomplete_benchmark_evidence_file(tmp_path, capsys):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps([]), encoding="utf-8")

    exit_code = realtime_voice_dgx_spark.main(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--repo-dir",
            str(tmp_path / "repo"),
            "--hermes-home",
            str(tmp_path / "home"),
            "--benchmark-evidence",
            str(evidence_path),
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert result["ok"] is False
    assert result["benchmark_evidence"]["ok"] is False
    assert result["benchmark_evidence"]["issues"]


def test_main_writes_files_and_reports_json(tmp_path, capsys):
    exit_code = realtime_voice_dgx_spark.main(
        [
            "--output-dir",
            str(tmp_path / "out"),
            "--repo-dir",
            str(tmp_path / "repo"),
            "--hermes-home",
            str(tmp_path / "home"),
        ]
    )

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert Path(result["written"]["manifest"]).is_file()
    assert Path(result["written"]["compose"]).is_file()


def test_voice_subcommand_exposes_dgx_spark_launch_profile(tmp_path):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    calls = []

    def cmd_voice(args):
        calls.append(args)

    build_voice_parser(subparsers, cmd_voice=cmd_voice)

    args = parser.parse_args(
        [
            "voice",
            "dgx-spark",
            "--output-dir",
            str(tmp_path / "out"),
            "--interface-model",
            "gemma-4-E2B-it",
            "--oracle-model",
            "gemma-4-26B-A4B-it",
            "--check",
        ]
    )
    args.func(args)

    assert args.command == "voice"
    assert args.voice_command == "dgx-spark"
    assert args.output_dir == str(tmp_path / "out")
    assert args.interface_model == "gemma-4-E2B-it"
    assert args.oracle_model == "gemma-4-26B-A4B-it"
    assert args.check is True
    assert calls == [args]
