from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.voiceops_spark_matrix import build_matrix, parse_args, write_matrix


def test_spark_matrix_defaults_to_needing_evidence(tmp_path):
    matrix = build_matrix()
    paths = write_matrix(tmp_path, matrix)

    assert matrix["hardware_target"] == "1x NVIDIA DGX Spark"
    assert matrix["policy"]["oracle_selected_by"] == "Hermes /model"
    assert matrix["ready_for_one_spark_demo"] is False
    assert matrix["role_status"] == {
        "asr": "needs_evidence",
        "oracle": "needs_evidence",
        "reflex": "needs_evidence",
        "tts": "needs_evidence",
    }
    assert {candidate["candidate_id"] for candidate in matrix["candidates"]} >= {
        "reflex-gemma4-e2b",
        "oracle-gemma4-26b-a4b",
        "oracle-nemotron3-ultra-hosted",
        "asr-nemotron-speech",
        "tts-magpie-local",
        "tts-cartesia-cloud-fallback",
    }
    assert set(paths) == {"json", "markdown", "evidence_template"}
    assert "VoiceOps DGX Spark Model Matrix" in Path(paths["markdown"]).read_text(encoding="utf-8")
    template = json.loads(Path(paths["evidence_template"]).read_text(encoding="utf-8"))
    assert template["evidence"][0]["verified"] is False
    assert "replace null metrics" in template["evidence"][0]["notes"]


def test_spark_matrix_validates_matching_evidence(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "evidence": [
                    {
                        "candidate_id": "reflex-gemma4-e2b",
                        "verified": True,
                        "metrics": {
                            "first_token_ms": 700,
                            "intent_latency_ms": 1100,
                            "steady_state_memory_gb": 20,
                        },
                    },
                    {
                        "candidate_id": "oracle-gemma4-26b-a4b",
                        "verified": True,
                        "metrics": {
                            "decode_tok_s": 24,
                            "prefill_tok_s": 3100,
                            "first_token_ms": 2100,
                            "steady_state_memory_gb": 86,
                        },
                    },
                    {
                        "candidate_id": "asr-nemotron-speech",
                        "verified": True,
                        "metrics": {
                            "asr_delta_ms": 30,
                            "final_transcript_ms": 600,
                            "word_error_rate": 0.08,
                        },
                    },
                    {
                        "candidate_id": "tts-magpie-local",
                        "verified": True,
                        "metrics": {
                            "tts_first_audio_ms": 200,
                            "underrun_count": 0,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    matrix = build_matrix([evidence_path])
    evaluations = {evaluation["candidate_id"]: evaluation for evaluation in matrix["evaluations"]}

    assert matrix["ready_for_one_spark_demo"] is True
    assert evaluations["reflex-gemma4-e2b"]["status"] == "validated"
    assert evaluations["oracle-gemma4-26b-a4b"]["status"] == "validated"
    assert evaluations["asr-nemotron-speech"]["status"] == "validated"
    assert evaluations["tts-magpie-local"]["status"] == "validated"


def test_spark_matrix_fails_unverified_or_slow_evidence(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "candidate_id": "oracle-gemma4-26b-a4b",
                "verified": False,
                "metrics": {
                    "decode_tok_s": 10,
                    "prefill_tok_s": 1000,
                    "first_token_ms": 5000,
                    "steady_state_memory_gb": 120,
                },
            }
        ),
        encoding="utf-8",
    )

    matrix = build_matrix([evidence_path])
    evaluation = next(item for item in matrix["evaluations"] if item["candidate_id"] == "oracle-gemma4-26b-a4b")

    assert evaluation["status"] == "fails_target"
    assert "evidence_not_verified" in evaluation["issues"]
    assert "target_failed:decode_tok_s" in evaluation["issues"]
    assert matrix["role_status"]["oracle"] == "needs_evidence"


def test_spark_matrix_cli_smoke(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "voiceops_spark_matrix.py"
    result = subprocess.run(
        ["python", str(script), "--output-dir", str(tmp_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert Path(payload["artifacts"]["json"]).exists()
    assert Path(payload["artifacts"]["evidence_template"]).exists()


def test_spark_matrix_parse_args_accepts_repeated_evidence(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    args = parse_args(["--evidence", str(first), "--evidence", str(second)])

    assert args.evidence == [first, second]
