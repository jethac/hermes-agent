import json
from pathlib import Path

from hermes_cli import realtime_voice_dgx_spark


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

    assert set(written) == {"manifest", "compose", "env_example", "launch", "benchmark_matrix"}
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "compose.yaml").is_file()
    assert (output_dir / ".env.example").is_file()
    assert (output_dir / "launch-local-stack.sh").is_file()
    assert (output_dir / "benchmark-matrix.json").is_file()
    assert (output_dir / "launch-local-stack.sh").stat().st_mode & 0o111

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    matrix = json.loads((output_dir / "benchmark-matrix.json").read_text(encoding="utf-8"))
    assert manifest["roles"]["interface"]["audio_input"] == "native_audio"
    assert manifest["engine"]["max_spoken_sentences"] == 2
    assert "HERMES_KAME_MAX_SPOKEN_SENTENCES=2" in (output_dir / ".env.example").read_text(encoding="utf-8")
    assert matrix["candidates"]["interface"][0]["input"] == "direct_audio"
    assert matrix["candidates"]["interface"][1]["input"] == "stt_fallback"


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
