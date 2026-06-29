import subprocess
from pathlib import Path

from hermes_cli.realtime_voice_dgx_spark import _kame_preflight_content_schema_issues


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dgx_spark_gemma4_voice_eval.sh"


def test_dgx_spark_eval_script_generates_full_kame_launch_pack():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "Track 0: full KAME DGX Spark launch pack" in text
    assert "python -m hermes_cli.realtime_voice_dgx_spark" in text
    assert '--output-dir "$KAME_STACK_DIR"' in text
    assert 'DGX_SPARK_KAME_CHECK:-0' in text
    assert "--interface-base-url" in text
    assert "--interface-api-key-env" in text
    assert "DGX_SPARK_INTERFACE_API_KEY_ENV" in text
    assert "--interface-max-audio-seconds" in text
    assert "DGX_SPARK_INTERFACE_MAX_AUDIO_SECONDS" in text
    assert "DGX_SPARK_KAME_BENCHMARK_EVIDENCE" in text
    assert '"$KAME_STACK_DIR/validate-benchmark-evidence.sh" "$DGX_SPARK_KAME_BENCHMARK_EVIDENCE"' in text
    assert "track 0 KAME benchmark evidence validation" in text
    assert "scripts/voiceops_spark_matrix.py" in text
    assert '--output-dir "$VOICEOPS_MATRIX_DIR"' in text
    assert '--evidence "$DGX_SPARK_KAME_BENCHMARK_EVIDENCE"' in text
    assert "track 0 VoiceOps Spark matrix verdict generated" in text
    assert "track 0 VoiceOps local one-Spark readiness verdict" in text
    assert "ready_for_one_spark_demo" in text
    assert "KAME benchmark matrix: $KAME_STACK_DIR/benchmark-matrix.json" in text
    assert "KAME benchmark evidence template: $KAME_STACK_DIR/benchmark-evidence-template.json" in text
    assert "KAME benchmark validator: $KAME_STACK_DIR/validate-benchmark-evidence.sh" in text
    assert "VoiceOps Spark matrix: $ARTIFACT_DIR/voiceops-spark-matrix/spark-model-matrix.json" in text
    assert "VoiceOps Spark matrix markdown: $ARTIFACT_DIR/voiceops-spark-matrix/spark-model-matrix.md" in text
    assert "--oracle-base-url" in text
    assert "--asr-base-url" in text
    assert "--asr-model" in text
    assert "--asr-module" in text
    assert "--asr-adapter" in text
    assert "--tts-base-url" in text
    assert "--tts-model" in text
    assert "--tts-module" in text
    assert "--tts-adapter" in text
    assert "DGX_SPARK_LOCAL_VOICE_STT_MODEL" in text
    assert "DGX_SPARK_LOCAL_VOICE_TTS_MODEL" in text
    assert "--preset nvidia_speech" in text
    assert "--provider local_speech" in text
    assert "Full KAME stack pack: $KAME_STACK_DIR" in text
    assert "python -m hermes_cli.realtime_voice_oracle_probe" in text
    assert "--output \"$ARTIFACT_DIR/oracle-probe.json\"" in text
    assert "Track A: configured oracle probe, diagnostic only" in text
    assert "track A configured oracle probe, diagnostic only" in text
    assert "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4" in text
    assert "gemma-4-26B-A4B-it" not in text
    assert "gemma-4-26b-a4b-it" not in text
    assert "python -m hermes_cli.realtime_voice_dgx_report" in text
    assert "--output \"$ARTIFACT_DIR/recommendation.json\"" in text
    assert "--markdown-output \"$ARTIFACT_DIR/recommendation.md\"" in text
    assert "DGX Spark KAME recommendation report" in text
    assert "Recommendation JSON: $ARTIFACT_DIR/recommendation.json" in text
    assert "Recommendation Markdown: $ARTIFACT_DIR/recommendation.md" in text
    assert "CARTESIA_API_KEY" not in text.split("Track 0: full KAME DGX Spark launch pack", 1)[1].split(
        "Track A: configured oracle probe, diagnostic only",
        1,
    )[0]


def test_dgx_spark_eval_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_dgx_spark_kame_preflight_uses_reflex_schema():
    assert _kame_preflight_content_schema_issues(
        '{"route":"reject_or_clarify","intent":"preflight audio probe","text":"","route_confidence":0.8,"local_reply":"Say that again?"}'
    ) == []
    assert _kame_preflight_content_schema_issues(
        '{"route":"local","intent":"preflight audio probe","text":"","route_confidence":false}'
    ) == ["route_confidence must be numeric", "local_reply is required for local or reject_or_clarify"]
