import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dgx_spark_gemma4_voice_eval.sh"


def test_dgx_spark_eval_script_generates_full_kame_launch_pack():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "Track 0: full KAME DGX Spark launch pack" in text
    assert "python -m hermes_cli.realtime_voice_dgx_spark" in text
    assert '--output-dir "$KAME_STACK_DIR"' in text
    assert 'DGX_SPARK_KAME_CHECK:-0' in text
    assert "--interface-base-url" in text
    assert "--oracle-base-url" in text
    assert "--asr-base-url" in text
    assert "--tts-base-url" in text
    assert "Full KAME stack pack: $KAME_STACK_DIR" in text
    assert "CARTESIA_API_KEY" not in text.split("Track 0: full KAME DGX Spark launch pack", 1)[1].split(
        "Track A: Gemma 4 oracle probe",
        1,
    )[0]


def test_dgx_spark_eval_script_is_valid_bash():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
