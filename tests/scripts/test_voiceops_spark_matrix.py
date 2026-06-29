from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.voiceops_spark_matrix import build_matrix, parse_args, write_matrix


def _base_evidence(candidate_id: str, *, model: str, locality: str = "local_spark") -> dict:
    return {
        "schema_version": "voiceops.spark_benchmark_evidence.v1",
        "candidate_id": candidate_id,
        "hardware": "1x NVIDIA DGX Spark" if locality == "local_spark" else "hosted",
        "locality": locality,
        "model": model,
        "engine": "test engine",
        "verified": True,
        "measured_at": "2026-06-29T00:00:00Z",
        "source_artifact": "artifacts/test/raw.json",
        "metrics": {},
    }


def _stack_smoke() -> dict:
    return {
        "schema_version": "voiceops.spark_benchmark_evidence.v1",
        "kind": "voiceops_spark_stack_smoke",
        "hardware": "1x NVIDIA DGX Spark",
        "locality": "local_spark",
        "verified": True,
        "measured_at": "2026-06-29T00:00:00Z",
        "source_artifact": "artifacts/test/stack-smoke.json",
        "oracle_selected_by": "Hermes /model",
        "components": {
            "reflex": True,
            "oracle": True,
            "asr": True,
            "tts": True,
            "sidecar": True,
        },
        "metrics": {
            "speech_end_to_first_audio_ms": 900,
            "barge_in_stop_ms": 90,
        },
    }


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
        "oracle-nemotron3-super-local",
        "oracle-nemotron3-ultra-hosted",
        "asr-nemotron-speech",
        "tts-magpie-local",
        "tts-cartesia-cloud-fallback",
    }
    oracle_candidates = {candidate["candidate_id"]: candidate for candidate in matrix["candidates"] if candidate["role"] == "oracle"}
    assert oracle_candidates["oracle-nemotron3-super-local"]["priority"] == 1
    assert oracle_candidates["oracle-nemotron3-super-local"]["locality"] == "local_spark"
    assert oracle_candidates["oracle-nemotron3-ultra-hosted"]["priority"] == 2
    assert oracle_candidates["oracle-nemotron3-ultra-hosted"]["locality"] == "hosted"
    assert set(paths) == {"json", "markdown", "evidence_example", "evidence_template"}
    assert "VoiceOps DGX Spark Model Matrix" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "all_local_stack_smoke: needs_evidence" in Path(paths["markdown"]).read_text(encoding="utf-8")
    example = json.loads(Path(paths["evidence_example"]).read_text(encoding="utf-8"))
    template = json.loads(Path(paths["evidence_template"]).read_text(encoding="utf-8"))
    assert example["example_only"] is True
    assert all(item["example_only"] is True for item in example["evidence"])
    assert template["evidence"][0]["verified"] is False
    assert template["evidence"][0]["schema_version"] == "voiceops.spark_benchmark_evidence.v1"
    assert template["evidence"][0]["model"]
    assert template["evidence"][-1]["kind"] == "voiceops_spark_stack_smoke"
    assert "replace null metrics" in template["evidence"][0]["notes"]


def test_spark_matrix_validates_matching_evidence(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "evidence": [
                    {
                        **_base_evidence("reflex-gemma4-e2b", model="Gemma 4 E2B audio-native"),
                        "metrics": {"first_token_ms": 700, "intent_latency_ms": 1100, "steady_state_memory_gb": 20},
                    },
                    {
                        **_base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super"),
                        "metrics": {
                            "decode_tok_s": 24,
                            "prefill_tok_s": 3100,
                            "first_token_ms": 2100,
                            "steady_state_memory_gb": 86,
                        },
                    },
                    {
                        **_base_evidence("asr-nemotron-speech", model="Nemotron Speech streaming"),
                        "metrics": {"asr_delta_ms": 30, "final_transcript_ms": 600, "word_error_rate": 0.08},
                    },
                    {
                        **_base_evidence("tts-magpie-local", model="Magpie local TTS"),
                        "metrics": {"tts_first_audio_ms": 200, "underrun_count": 0},
                    },
                    _stack_smoke(),
                ]
            }
        ),
        encoding="utf-8",
    )

    matrix = build_matrix([evidence_path])
    evaluations = {evaluation["candidate_id"]: evaluation for evaluation in matrix["evaluations"]}

    assert matrix["ready_for_one_spark_demo"] is True
    assert evaluations["reflex-gemma4-e2b"]["status"] == "validated"
    assert evaluations["oracle-nemotron3-super-local"]["status"] == "validated"
    assert evaluations["asr-nemotron-speech"]["status"] == "validated"
    assert evaluations["tts-magpie-local"]["status"] == "validated"
    assert matrix["stack_smoke"]["status"] == "validated"


def test_spark_matrix_example_is_not_accepted_as_proof(tmp_path):
    paths = write_matrix(tmp_path, build_matrix())

    matrix = build_matrix([Path(paths["evidence_example"])])
    evaluations = {evaluation["candidate_id"]: evaluation for evaluation in matrix["evaluations"]}

    assert matrix["ready_for_one_spark_demo"] is False
    assert evaluations["reflex-gemma4-e2b"]["status"] == "fails_target"
    assert evaluations["oracle-nemotron3-super-local"]["status"] == "fails_target"
    assert evaluations["asr-nemotron-speech"]["status"] == "fails_target"
    assert evaluations["tts-magpie-local"]["status"] == "fails_target"
    assert "example_only_evidence_not_accepted" in evaluations["oracle-nemotron3-super-local"]["issues"]
    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "example_only_evidence_not_accepted" in matrix["stack_smoke"]["issues"]
    assert matrix["role_status"] == {
        "asr": "needs_evidence",
        "oracle": "needs_evidence",
        "reflex": "needs_evidence",
        "tts": "needs_evidence",
    }


def test_spark_matrix_fails_unverified_or_slow_evidence(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.spark_benchmark_evidence.v1",
                "candidate_id": "oracle-nemotron3-super-local",
                "hardware": "1x NVIDIA DGX Spark",
                "locality": "local_spark",
                "model": "Gemma 4 26B-A4B",
                "engine": "test engine",
                "verified": False,
                "measured_at": "2026-06-29T00:00:00Z",
                "source_artifact": "artifacts/test/raw.json",
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
    evaluation = next(item for item in matrix["evaluations"] if item["candidate_id"] == "oracle-nemotron3-super-local")

    assert evaluation["status"] == "fails_target"
    assert "evidence_not_verified" in evaluation["issues"]
    assert "model_mismatch" in evaluation["issues"]
    assert "target_failed:decode_tok_s" in evaluation["issues"]
    assert matrix["role_status"]["oracle"] == "needs_evidence"


def test_spark_matrix_rejects_ultra_model_for_super_local_oracle_gate(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "evidence": [
                    {
                        **_base_evidence("reflex-gemma4-e2b", model="Gemma 4 E2B audio-native"),
                        "metrics": {"first_token_ms": 700, "intent_latency_ms": 1100, "steady_state_memory_gb": 20},
                    },
                    {
                        **_base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Ultra"),
                        "metrics": {
                            "decode_tok_s": 24,
                            "prefill_tok_s": 3100,
                            "first_token_ms": 2100,
                            "steady_state_memory_gb": 86,
                        },
                    },
                    {
                        **_base_evidence("asr-nemotron-speech", model="Nemotron Speech streaming"),
                        "metrics": {"asr_delta_ms": 30, "final_transcript_ms": 600, "word_error_rate": 0.08},
                    },
                    {
                        **_base_evidence("tts-magpie-local", model="Magpie local TTS"),
                        "metrics": {"tts_first_audio_ms": 200, "underrun_count": 0},
                    },
                    _stack_smoke(),
                ]
            }
        ),
        encoding="utf-8",
    )

    matrix = build_matrix([evidence_path])
    evaluations = {evaluation["candidate_id"]: evaluation for evaluation in matrix["evaluations"]}
    oracle_evaluation = evaluations["oracle-nemotron3-super-local"]

    assert oracle_evaluation["status"] == "fails_target"
    assert "model_mismatch" in oracle_evaluation["issues"]
    assert matrix["role_status"]["reflex"] == "validated"
    assert matrix["role_status"]["oracle"] == "needs_evidence"
    assert matrix["role_status"]["asr"] == "validated"
    assert matrix["role_status"]["tts"] == "validated"
    assert matrix["stack_smoke"]["status"] == "validated"
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_hosted_ultra_does_not_validate_local_oracle_role(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.spark_benchmark_evidence.v1",
                "candidate_id": "oracle-nemotron3-ultra-hosted",
                "hardware": "hosted",
                "locality": "hosted",
                "model": "Nemotron 3 Ultra",
                "engine": "hosted provider",
                "verified": True,
                "measured_at": "2026-06-29T00:00:00Z",
                "source_artifact": "artifacts/test/hosted.json",
                "metrics": {
                    "first_token_ms": 900,
                    "tool_plan_quality": 5,
                },
            }
        ),
        encoding="utf-8",
    )

    matrix = build_matrix([evidence_path])
    evaluations = {evaluation["candidate_id"]: evaluation for evaluation in matrix["evaluations"]}

    assert evaluations["oracle-nemotron3-ultra-hosted"]["status"] == "validated"
    assert matrix["role_status"]["oracle"] == "needs_evidence"
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_rejects_unproven_all_local_stack_smoke(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    incomplete = _stack_smoke()
    incomplete["components"]["tts"] = False
    incomplete["metrics"]["barge_in_stop_ms"] = 300
    evidence_path.write_text(json.dumps({"evidence": [incomplete]}), encoding="utf-8")

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "missing_components:tts" in matrix["stack_smoke"]["issues"]
    assert "target_failed:barge_in_stop_ms" in matrix["stack_smoke"]["issues"]
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_adapts_kame_benchmark_evidence_with_provenance(tmp_path):
    evidence_path = tmp_path / "kame-evidence.json"
    common = {
        "hardware": "1x DGX Spark",
        "verified": True,
        "measured_at": "2026-06-29T00:00:00Z",
        "source_artifact": "artifacts/kame/raw.json",
    }
    evidence_path.write_text(
        json.dumps(
            [
                {
                    **common,
                    "kind": "kame_model_assumption_result",
                    "name": "oracle_authority",
                    "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
                    "ok": True,
                },
                {
                    **common,
                    "kind": "kame_benchmark_result",
                    "category": "interface",
                    "model": "gemma-4-E2B-it",
                    "metrics": {
                        "kame_interface_model_request_ms": 220,
                        "speech_end_to_interface_decision_p90_ms": 600,
                        "steady_state_memory_gb": 18,
                    },
                },
                {
                    **common,
                    "kind": "kame_benchmark_result",
                    "category": "oracle",
                    "metrics": {
                        "decode_tok_s": 24,
                        "prefill_tok_s": 3100,
                        "oracle_accepted_to_first_token_ms": 1200,
                        "steady_state_memory_gb": 92,
                    },
                },
                {
                    **common,
                    "kind": "kame_benchmark_result",
                    "category": "speech",
                    "role": "oracle_verbatim_asr",
                    "model": "Nemotron Speech streaming",
                    "metrics": {
                        "speech_end_to_asr_final_ms": 80,
                        "speech_end_to_asr_final_p90_ms": 150,
                        "literal_accuracy_names_numbers_code": 0.9,
                    },
                },
                {
                    **common,
                    "kind": "kame_benchmark_result",
                    "category": "speech",
                    "role": "tts",
                    "model": "Magpie local TTS",
                    "metrics": {
                        "tts_request_to_first_audio_ms": 180,
                        "underrun_count": 0,
                    },
                },
                {
                    **common,
                    "kind": "kame_smoke_result",
                    "name": "all_local_smoke",
                    "ok": True,
                    "local_turns": 2,
                    "local_turn_oracle_calls": 0,
                    "oracle_bound_turns": 4,
                    "oracle_bound_oracle_calls": 4,
                    "oracle_authority_routes": ["tools", "files", "memory", "project_context"],
                    "interface_input_sources": ["native_audio"],
                    "reflex_providers": ["vllm"],
                    "metrics": {
                        "speech_end_to_first_audio_ms": 900,
                        "barge_in_stop_ms": 90,
                    },
                },
            ]
        ),
        encoding="utf-8",
    )

    matrix = build_matrix([evidence_path])
    evaluations = {evaluation["candidate_id"]: evaluation for evaluation in matrix["evaluations"]}

    assert evaluations["reflex-gemma4-e2b"]["status"] == "validated"
    assert evaluations["oracle-nemotron3-super-local"]["status"] == "validated"
    assert evaluations["asr-nemotron-speech"]["status"] == "validated"
    assert evaluations["tts-magpie-local"]["status"] == "validated"
    assert matrix["stack_smoke"]["status"] == "validated"
    assert matrix["ready_for_one_spark_demo"] is True


def test_spark_matrix_rejects_kame_smoke_without_reflex_bypass_and_oracle_authority(tmp_path):
    evidence_path = tmp_path / "kame-evidence.json"
    common = {
        "hardware": "1x DGX Spark",
        "verified": True,
        "measured_at": "2026-06-29T00:00:00Z",
        "source_artifact": "artifacts/kame/raw.json",
    }
    passing_entries = [
        {
            **common,
            "kind": "kame_benchmark_result",
            "category": "interface",
            "model": "gemma-4-E2B-it",
            "metrics": {
                "kame_interface_model_request_ms": 220,
                "speech_end_to_interface_decision_p90_ms": 600,
                "steady_state_memory_gb": 18,
            },
        },
        {
            **common,
            "kind": "kame_benchmark_result",
            "category": "oracle",
            "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
            "metrics": {
                "decode_tok_s": 24,
                "prefill_tok_s": 3100,
                "oracle_accepted_to_first_token_ms": 1200,
                "steady_state_memory_gb": 92,
            },
        },
        {
            **common,
            "kind": "kame_benchmark_result",
            "category": "speech",
            "role": "oracle_verbatim_asr",
            "model": "Nemotron Speech streaming",
            "metrics": {
                "speech_end_to_asr_final_ms": 80,
                "speech_end_to_asr_final_p90_ms": 150,
                "literal_accuracy_names_numbers_code": 0.9,
            },
        },
        {
            **common,
            "kind": "kame_benchmark_result",
            "category": "speech",
            "role": "tts",
            "model": "Magpie local TTS",
            "metrics": {
                "tts_request_to_first_audio_ms": 180,
                "underrun_count": 0,
            },
        },
    ]
    evidence_path.write_text(
        json.dumps(
            passing_entries
            + [
                {
                    **common,
                    "kind": "kame_smoke_result",
                    "name": "all_local_smoke",
                    "ok": True,
                    "local_turns": 2,
                    "local_turn_oracle_calls": 1,
                    "oracle_bound_turns": 4,
                    "oracle_bound_oracle_calls": 4,
                    "oracle_authority_routes": ["tools"],
                    "interface_input_sources": ["streaming_stt"],
                    "reflex_providers": ["local_stt"],
                    "metrics": {
                        "speech_end_to_first_audio_ms": 900,
                        "barge_in_stop_ms": 90,
                    },
                },
            ]
        ),
        encoding="utf-8",
    )

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "target_failed:local_turn_oracle_calls" in matrix["stack_smoke"]["issues"]
    assert "missing_oracle_authority_routes:files,memory,project_context" in matrix["stack_smoke"]["issues"]
    assert "missing_interface_input_source:native_audio" in matrix["stack_smoke"]["issues"]
    assert "missing_reflex_provider:vllm" in matrix["stack_smoke"]["issues"]
    assert matrix["ready_for_one_spark_demo"] is False


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
    assert Path(payload["artifacts"]["evidence_example"]).exists()
    assert Path(payload["artifacts"]["evidence_template"]).exists()


def test_spark_matrix_parse_args_accepts_repeated_evidence(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    args = parse_args(["--evidence", str(first), "--evidence", str(second)])

    assert args.evidence == [first, second]
