from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.voiceops_spark_matrix import build_matrix, parse_args, write_matrix


def _source_artifact_sha256(relative: str) -> str:
    payload = json.dumps({"redacted": True, "source": relative}).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture(autouse=True)
def _spark_raw_source_artifacts(tmp_path):
    for relative in [
        "artifacts/test/raw.json",
        "artifacts/test/stack-smoke.json",
        "artifacts/test/hosted.json",
        "artifacts/test/cartesia.json",
        "artifacts/kame/raw.json",
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"redacted": True, "source": relative}), encoding="utf-8")


def _base_evidence(candidate_id: str, *, model: str, locality: str = "local_spark") -> dict:
    evidence = {
        "schema_version": "voiceops.spark_benchmark_evidence.v1",
        "candidate_id": candidate_id,
        "hardware": "1x NVIDIA DGX Spark" if locality == "local_spark" else "hosted",
        "locality": locality,
        "model": model,
        "engine": "test engine",
        "verified": True,
        "measured_at": "2026-06-29T00:00:00Z",
        "source_artifact": "artifacts/test/raw.json",
        "source_artifact_sha256": _source_artifact_sha256("artifacts/test/raw.json"),
        "metrics": {},
    }
    if candidate_id == "oracle-nemotron3-super-local":
        evidence["oracle_selected_by"] = "Hermes /model"
    return evidence


def _stack_smoke() -> dict:
    return {
        "schema_version": "voiceops.spark_benchmark_evidence.v1",
        "kind": "voiceops_spark_stack_smoke",
        "hardware": "1x NVIDIA DGX Spark",
        "locality": "local_spark",
        "verified": True,
        "measured_at": "2026-06-29T00:00:00Z",
        "source_artifact": "artifacts/test/stack-smoke.json",
        "source_artifact_sha256": _source_artifact_sha256("artifacts/test/stack-smoke.json"),
        "oracle_selected_by": "Hermes /model",
        "oracle_authority_routes": ["tools", "files", "memory", "project_context"],
        "interface_input_sources": ["native_audio"],
        "reflex_providers": ["vllm"],
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
            "local_turns": 2,
            "local_turn_oracle_calls": 0,
            "oracle_bound_turns": 4,
            "oracle_bound_oracle_calls": 4,
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
        "reflex-gemma4-e4b",
        "oracle-nemotron3-super-local",
        "oracle-nemotron3-ultra-hosted",
        "asr-nemotron-speech",
        "tts-magpie-local",
        "tts-cartesia-cloud-fallback",
    }
    reflex_candidates = {candidate["candidate_id"]: candidate for candidate in matrix["candidates"] if candidate["role"] == "reflex"}
    assert reflex_candidates["reflex-gemma4-e2b"]["priority"] == 1
    assert reflex_candidates["reflex-gemma4-e4b"]["priority"] == 1
    assert reflex_candidates["reflex-gemma4-e4b"]["locality"] == "local_spark"
    oracle_candidates = {candidate["candidate_id"]: candidate for candidate in matrix["candidates"] if candidate["role"] == "oracle"}
    assert oracle_candidates["oracle-nemotron3-super-local"]["priority"] == 1
    assert oracle_candidates["oracle-nemotron3-super-local"]["locality"] == "local_spark"
    assert oracle_candidates["oracle-nemotron3-ultra-hosted"]["priority"] == 2
    assert oracle_candidates["oracle-nemotron3-ultra-hosted"]["locality"] == "hosted"
    required_paths = {
        "closure_json",
        "closure_markdown",
        "evidence_example",
        "evidence_scaffold",
        "evidence_template",
        "json",
        "markdown",
        "operator_runbook",
    }
    assert required_paths <= set(paths)
    assert {
        "scaffold_source_reflex-gemma4-e2b",
        "scaffold_source_reflex-gemma4-e4b",
        "scaffold_source_oracle-nemotron3-super-local",
        "scaffold_source_asr-nemotron-speech",
        "scaffold_source_tts-magpie-local",
        "scaffold_source_voiceops_spark_stack_smoke",
    } <= set(paths)
    assert "VoiceOps DGX Spark Model Matrix" in Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "all_local_stack_smoke: needs_evidence" in Path(paths["markdown"]).read_text(encoding="utf-8")
    closure = json.loads(Path(paths["closure_json"]).read_text(encoding="utf-8"))
    closure_markdown = Path(paths["closure_markdown"]).read_text(encoding="utf-8")
    operator_runbook = Path(paths["operator_runbook"]).read_text(encoding="utf-8")
    example = json.loads(Path(paths["evidence_example"]).read_text(encoding="utf-8"))
    scaffold_path = Path(paths["evidence_scaffold"])
    scaffold = json.loads(scaffold_path.read_text(encoding="utf-8"))
    template = json.loads(Path(paths["evidence_template"]).read_text(encoding="utf-8"))
    assert closure["schema_version"] == "voiceops.milestone4.spark_matrix_closure.v1"
    assert closure["status"] == "needs_external_evidence"
    assert closure["ready"] is False
    assert closure["source_matrix_artifact"] == "spark-model-matrix.json"
    assert closure["mode"]["spark_execution"] is False
    assert closure["mode"]["network_io"] is False
    assert closure["missing_gates"] == [
        "asr:needs_evidence",
        "oracle:needs_evidence",
        "reflex:needs_evidence",
        "tts:needs_evidence",
        "all_local_stack_smoke",
    ]
    assert closure["missing_roles"] == [
        "asr:needs_evidence",
        "oracle:needs_evidence",
        "reflex:needs_evidence",
        "tts:needs_evidence",
    ]
    assert {
        "schema_version",
        "oracle_selected_by",
        "oracle_authority_routes",
        "interface_input_sources",
        "reflex_providers",
        "metrics.local_turns",
        "metrics.local_turn_oracle_calls",
        "metrics.oracle_bound_turns",
        "metrics.oracle_bound_oracle_calls",
    } <= set(closure["required_stack_smoke_fields"])
    assert closure["all_local_stack_smoke"]["required_components"] == ["reflex", "oracle", "asr", "tts", "sidecar"]
    assert closure["evidence_contract"]["preferred_local_oracle_candidate_id"] == "oracle-nemotron3-super-local"
    assert closure["evidence_contract"]["preferred_local_oracle_model"] == "Nemotron 3 Super"
    assert closure["evidence_contract"]["non_counting_fallback_oracle_models"] == ["Nemotron 3 Ultra"]
    assert closure["evidence_contract"]["source_artifacts_must_exist"] is True
    assert closure["evidence_contract"]["source_artifact_readable"] is True
    assert closure["evidence_contract"]["source_artifact_sha256_must_match"] is True
    assert closure["evidence_contract"]["source_artifact_resolution"].endswith("supplied benchmark evidence file")
    assert closure["benchmark_evidence_shape"]["evidence"][0]["schema_version"] == "voiceops.spark_benchmark_evidence.v1"
    assert closure["benchmark_evidence_shape"]["evidence"][0]["candidate_id"] == "oracle-nemotron3-super-local"
    assert closure["benchmark_evidence_shape"]["evidence"][1]["kind"] == "voiceops_spark_stack_smoke"
    assert "scripts/dgx_spark_gemma4_voice_eval.sh" == closure["rerun_commands"]["dgx_eval"]
    assert "VoiceOps Milestone 4 Spark Matrix Closure" in closure_markdown
    assert "spark-benchmark-scaffold/spark-benchmark-evidence.json" in closure_markdown
    assert "hosted or multi-Spark Nemotron 3 Ultra fallback evidence" in closure_markdown
    assert '"evidence": [' in closure_markdown
    assert "voiceops.spark_benchmark_evidence.v1" in closure_markdown
    assert "oracle_authority_routes" in closure_markdown
    assert "source_artifact" in closure_markdown
    assert "source_artifact_sha256" in closure_markdown
    assert "source_artifacts_must_exist" in closure_markdown
    assert "VoiceOps DGX Spark Operator Runbook" in operator_runbook
    assert "scripts/dgx_spark_gemma4_voice_eval.sh" in operator_runbook
    assert "spark-benchmark-scaffold/spark-benchmark-evidence.json" in operator_runbook
    assert "uv run python scripts/voiceops_spark_matrix.py" in operator_runbook
    assert "uv run python scripts/voiceops_plan_run.py" in operator_runbook
    assert "Nemotron 3 Super is the preferred one-Spark oracle candidate" in operator_runbook
    assert "Hosted or multi-Spark Nemotron 3 Ultra evidence" in operator_runbook
    assert "`loopback_smoke_bridge`" in operator_runbook
    assert "protocol-only smoke checks" in operator_runbook
    assert "HERMES_DGX_SPARK_ASR_MODULE" in operator_runbook
    assert "HERMES_DGX_SPARK_TTS_ADAPTER" in operator_runbook
    assert "must remain unverified for local ASR/TTS roles" in operator_runbook
    assert "`speech_end_to_first_audio_ms <= 1500`" in operator_runbook
    assert "`barge_in_stop_ms <= 150`" in operator_runbook
    assert "`local_turn_oracle_calls == 0`" in operator_runbook
    assert "`oracle_bound_oracle_calls >= oracle_bound_turns`" in operator_runbook
    assert example["example_only"] is True
    assert all(item["example_only"] is True for item in example["evidence"])
    assert scaffold["example_only"] is True
    assert scaffold["evidence"][0]["source_artifact"] == "sources/reflex-gemma4-e2b-raw.json"
    scaffold_matrix = build_matrix([scaffold_path])
    scaffold_evaluations = {
        evaluation["candidate_id"]: evaluation
        for evaluation in scaffold_matrix["evaluations"]
    }
    assert scaffold_matrix["ready_for_one_spark_demo"] is False
    assert "example_only_evidence_not_accepted" in scaffold_evaluations["reflex-gemma4-e2b"]["issues"]
    assert "example_only_evidence_not_accepted" in scaffold_evaluations["reflex-gemma4-e4b"]["issues"]
    assert "source_artifact_not_found" not in scaffold_evaluations["reflex-gemma4-e2b"]["issues"]
    assert "source_artifact_not_found" not in scaffold_evaluations["reflex-gemma4-e4b"]["issues"]
    assert "source_artifact_not_found" not in scaffold_matrix["stack_smoke"]["issues"]
    scaffold_source = json.loads(Path(paths["scaffold_source_reflex-gemma4-e2b"]).read_text(encoding="utf-8"))
    assert scaffold_source["redacted"] is True
    assert template["evidence"][0]["verified"] is False
    assert template["evidence"][0]["schema_version"] == "voiceops.spark_benchmark_evidence.v1"
    assert template["evidence"][0]["model"]
    assert "source_artifact_sha256" in template["evidence"][0]
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

    paths = write_matrix(tmp_path / "validated", matrix)
    closure = json.loads(Path(paths["closure_json"]).read_text(encoding="utf-8"))
    assert closure["status"] == "complete"
    assert closure["ready"] is True
    assert closure["missing_gates"] == []


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


def test_spark_matrix_wrapper_example_is_not_accepted_as_proof(tmp_path):
    evidence_path = tmp_path / "wrapped-example.json"
    evidence_path.write_text(
        json.dumps(
            {
                "example_only": True,
                "evidence": [
                    {
                        **_base_evidence("reflex-gemma4-e2b", model="Gemma 4 E2B"),
                        "metrics": {
                            "first_token_ms": 200,
                            "intent_latency_ms": 600,
                            "steady_state_memory_gb": 18,
                        },
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
                ],
            }
        ),
        encoding="utf-8",
    )

    matrix = build_matrix([evidence_path])
    evaluations = {evaluation["candidate_id"]: evaluation for evaluation in matrix["evaluations"]}

    assert matrix["ready_for_one_spark_demo"] is False
    assert "example_only_evidence_not_accepted" in evaluations["reflex-gemma4-e2b"]["issues"]
    assert "example_only_evidence_not_accepted" in evaluations["oracle-nemotron3-super-local"]["issues"]
    assert "example_only_evidence_not_accepted" in matrix["stack_smoke"]["issues"]


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
                "source_artifact_sha256": _source_artifact_sha256("artifacts/test/raw.json"),
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


def test_spark_matrix_rejects_candidate_with_missing_source_artifact(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    evidence["source_artifact"] = "missing/raw-oracle.json"
    evidence["metrics"] = {
        "decode_tok_s": 24,
        "prefill_tok_s": 3100,
        "first_token_ms": 2100,
        "steady_state_memory_gb": 86,
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    matrix = build_matrix([evidence_path])
    evaluation = next(item for item in matrix["evaluations"] if item["candidate_id"] == "oracle-nemotron3-super-local")

    assert evaluation["status"] == "fails_target"
    assert "source_artifact_not_found" in evaluation["issues"]
    assert matrix["role_status"]["oracle"] == "needs_evidence"


def test_spark_matrix_rejects_candidate_with_mismatched_source_artifact_hash(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    evidence["source_artifact_sha256"] = "0" * 64
    evidence["metrics"] = {
        "decode_tok_s": 24,
        "prefill_tok_s": 3100,
        "first_token_ms": 2100,
        "steady_state_memory_gb": 86,
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    matrix = build_matrix([evidence_path])
    evaluation = next(item for item in matrix["evaluations"] if item["candidate_id"] == "oracle-nemotron3-super-local")

    assert evaluation["status"] == "fails_target"
    assert "source_artifact_sha256_mismatch" in evaluation["issues"]
    assert matrix["role_status"]["oracle"] == "needs_evidence"


def test_spark_matrix_rejects_candidate_with_example_only_source_artifact(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    source_path = tmp_path / "artifacts/test/example-source.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps({"example_only": True, "redacted": True}), encoding="utf-8")
    evidence = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    evidence["source_artifact"] = "artifacts/test/example-source.json"
    evidence["source_artifact_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    evidence["metrics"] = {
        "decode_tok_s": 24,
        "prefill_tok_s": 3100,
        "first_token_ms": 2100,
        "steady_state_memory_gb": 86,
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    matrix = build_matrix([evidence_path])
    evaluation = next(item for item in matrix["evaluations"] if item["candidate_id"] == "oracle-nemotron3-super-local")

    assert evaluation["status"] == "fails_target"
    assert "source_artifact_example_only_not_accepted" in evaluation["issues"]
    assert matrix["role_status"]["oracle"] == "needs_evidence"


def test_spark_matrix_rejects_candidate_with_invalid_measured_at(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    evidence["measured_at"] = "yesterday"
    evidence["metrics"] = {
        "decode_tok_s": 24,
        "prefill_tok_s": 3100,
        "first_token_ms": 2100,
        "steady_state_memory_gb": 86,
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    matrix = build_matrix([evidence_path])
    evaluation = next(item for item in matrix["evaluations"] if item["candidate_id"] == "oracle-nemotron3-super-local")

    assert evaluation["status"] == "fails_target"
    assert "invalid_measured_at" in evaluation["issues"]
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
                "source_artifact_sha256": _source_artifact_sha256("artifacts/test/hosted.json"),
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


def test_spark_matrix_cartesia_fallback_does_not_validate_local_tts_role(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": "voiceops.spark_benchmark_evidence.v1",
                "candidate_id": "tts-cartesia-cloud-fallback",
                "hardware": "hosted",
                "locality": "hosted",
                "model": "Cartesia cloud TTS",
                "engine": "Cartesia API",
                "verified": True,
                "measured_at": "2026-06-29T00:00:00Z",
                "source_artifact": "artifacts/test/cartesia.json",
                "source_artifact_sha256": _source_artifact_sha256("artifacts/test/cartesia.json"),
                "metrics": {
                    "tts_first_audio_ms": 250,
                    "cutoff_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    matrix = build_matrix([evidence_path])
    evaluations = {evaluation["candidate_id"]: evaluation for evaluation in matrix["evaluations"]}

    assert evaluations["tts-cartesia-cloud-fallback"]["status"] == "validated"
    assert matrix["role_status"]["tts"] == "needs_evidence"
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_cartesia_fallback_does_not_make_stack_ready_without_local_tts(tmp_path):
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
                        **_base_evidence("tts-cartesia-cloud-fallback", model="Cartesia cloud TTS", locality="hosted"),
                        "metrics": {"tts_first_audio_ms": 250, "cutoff_count": 0},
                    },
                    _stack_smoke(),
                ]
            }
        ),
        encoding="utf-8",
    )

    matrix = build_matrix([evidence_path])
    evaluations = {evaluation["candidate_id"]: evaluation for evaluation in matrix["evaluations"]}

    assert evaluations["tts-cartesia-cloud-fallback"]["status"] == "validated"
    assert evaluations["tts-magpie-local"]["status"] == "needs_evidence"
    assert matrix["role_status"] == {
        "asr": "validated",
        "oracle": "validated",
        "reflex": "validated",
        "tts": "needs_evidence",
    }
    assert matrix["stack_smoke"]["status"] == "validated"
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


def test_spark_matrix_rejects_stack_smoke_without_provenance(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    incomplete = _stack_smoke()
    incomplete.pop("source_artifact")
    incomplete.pop("measured_at")
    evidence_path.write_text(json.dumps({"evidence": [incomplete]}), encoding="utf-8")

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "missing_source_artifact" in matrix["stack_smoke"]["issues"]
    assert "missing_measured_at" in matrix["stack_smoke"]["issues"]
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_rejects_stack_smoke_with_missing_source_artifact(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    incomplete = _stack_smoke()
    incomplete["source_artifact"] = "missing/stack-smoke.json"
    evidence_path.write_text(json.dumps({"evidence": [incomplete]}), encoding="utf-8")

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "source_artifact_not_found" in matrix["stack_smoke"]["issues"]
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_rejects_stack_smoke_with_mismatched_source_artifact_hash(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    incomplete = _stack_smoke()
    incomplete["source_artifact_sha256"] = "f" * 64
    evidence_path.write_text(json.dumps({"evidence": [incomplete]}), encoding="utf-8")

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "source_artifact_sha256_mismatch" in matrix["stack_smoke"]["issues"]
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_rejects_stack_smoke_with_example_only_source_artifact(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    source_path = tmp_path / "artifacts/test/example-stack-smoke.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps({"example_only": True, "redacted": True}), encoding="utf-8")
    incomplete = _stack_smoke()
    incomplete["source_artifact"] = "artifacts/test/example-stack-smoke.json"
    incomplete["source_artifact_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    evidence_path.write_text(json.dumps({"evidence": [incomplete]}), encoding="utf-8")

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "source_artifact_example_only_not_accepted" in matrix["stack_smoke"]["issues"]
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_rejects_stack_smoke_with_timezone_less_measured_at(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    incomplete = _stack_smoke()
    incomplete["measured_at"] = "2026-06-29T00:00:00"
    evidence_path.write_text(json.dumps({"evidence": [incomplete]}), encoding="utf-8")

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "invalid_measured_at" in matrix["stack_smoke"]["issues"]
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_rejects_native_stack_smoke_without_kame_routing_proof(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    incomplete = _stack_smoke()
    incomplete.pop("oracle_authority_routes")
    incomplete.pop("interface_input_sources")
    incomplete.pop("reflex_providers")
    evidence_path.write_text(json.dumps({"evidence": [incomplete]}), encoding="utf-8")

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "missing_oracle_authority_routes:files,memory,project_context,tools" in matrix["stack_smoke"]["issues"]
    assert "missing_interface_input_source:native_audio" in matrix["stack_smoke"]["issues"]
    assert "missing_reflex_provider:vllm" in matrix["stack_smoke"]["issues"]
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_rejects_stack_smoke_above_closure_first_audio_target(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    too_slow = _stack_smoke()
    too_slow["metrics"]["speech_end_to_first_audio_ms"] = 1501
    evidence_path.write_text(json.dumps({"evidence": [too_slow]}), encoding="utf-8")

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "target_failed:speech_end_to_first_audio_ms" in matrix["stack_smoke"]["issues"]
    assert matrix["ready_for_one_spark_demo"] is False


@pytest.mark.parametrize(
    ("metrics", "expected_status", "expected_issue"),
    [
        ({"speech_end_to_first_audio_ms": 1500}, "validated", None),
        ({"barge_in_stop_ms": 150}, "validated", None),
        ({"local_turns": 1}, "validated", None),
        ({"local_turns": 0}, "fails_target", "missing_or_failed:local_turns"),
        ({"oracle_bound_turns": 1, "oracle_bound_oracle_calls": 1}, "validated", None),
        ({"oracle_bound_turns": 0}, "fails_target", "missing_or_failed:oracle_bound_turns"),
        ({"oracle_bound_turns": 2, "oracle_bound_oracle_calls": 2}, "validated", None),
        ({"oracle_bound_turns": 2, "oracle_bound_oracle_calls": 1}, "fails_target", "target_failed:oracle_bound_oracle_calls"),
    ],
)
def test_spark_matrix_stack_smoke_threshold_boundaries(tmp_path, metrics, expected_status, expected_issue):
    evidence_path = tmp_path / "evidence.json"
    smoke = _stack_smoke()
    smoke["metrics"].update(metrics)
    evidence_path.write_text(json.dumps({"evidence": [smoke]}), encoding="utf-8")

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == expected_status
    if expected_issue is None:
        assert matrix["stack_smoke"]["issues"] == []
    else:
        assert expected_issue in matrix["stack_smoke"]["issues"]


def test_spark_matrix_adapts_kame_benchmark_evidence_with_provenance(tmp_path):
    evidence_path = tmp_path / "kame-evidence.json"
    common = {
        "hardware": "1x DGX Spark",
        "locality": "local_spark",
        "verified": True,
        "measured_at": "2026-06-29T00:00:00Z",
        "source_artifact": "artifacts/kame/raw.json",
        "source_artifact_sha256": _source_artifact_sha256("artifacts/kame/raw.json"),
    }
    evidence_path.write_text(
        json.dumps(
            [
                {
                    **common,
                    "kind": "kame_model_assumption_result",
                    "name": "oracle_authority",
                    "validated_by": "oracle_models_probe",
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
                    "oracle_selected_by": "Hermes /model",
                    "components": {
                        "reflex": True,
                        "oracle": True,
                        "asr": True,
                        "tts": True,
                        "sidecar": True,
                    },
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


def test_spark_matrix_adapts_kame_e4b_interface_evidence(tmp_path):
    evidence_path = tmp_path / "kame-e4b-evidence.json"
    evidence_path.write_text(
        json.dumps(
            [
                {
                    "kind": "kame_benchmark_result",
                    "category": "interface",
                    "model": "gemma-4-E4B-it",
                    "hardware": "1x DGX Spark",
                    "locality": "local_spark",
                    "verified": True,
                    "measured_at": "2026-06-29T00:00:00Z",
                    "source_artifact": "artifacts/kame/raw.json",
                    "source_artifact_sha256": _source_artifact_sha256("artifacts/kame/raw.json"),
                    "metrics": {
                        "kame_interface_model_request_ms": 350,
                        "speech_end_to_interface_decision_p90_ms": 900,
                        "steady_state_memory_gb": 34,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    matrix = build_matrix([evidence_path])
    evaluations = {evaluation["candidate_id"]: evaluation for evaluation in matrix["evaluations"]}

    assert evaluations["reflex-gemma4-e2b"]["status"] == "needs_evidence"
    assert evaluations["reflex-gemma4-e4b"]["status"] == "validated"
    assert matrix["role_status"]["reflex"] == "validated"
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_rejects_protocol_only_kame_speech_evidence(tmp_path):
    evidence_path = tmp_path / "kame-evidence.json"
    common = {
        "hardware": "1x DGX Spark",
        "locality": "local_spark",
        "verified": True,
        "measured_at": "2026-06-29T00:00:00Z",
        "source_artifact": "artifacts/kame/raw.json",
        "source_artifact_sha256": _source_artifact_sha256("artifacts/kame/raw.json"),
        "adapter": "loopback_smoke_bridge",
        "protocol_smoke_only": True,
    }
    evidence_path.write_text(
        json.dumps(
            [
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
        ),
        encoding="utf-8",
    )

    matrix = build_matrix([evidence_path])
    evaluations = {evaluation["candidate_id"]: evaluation for evaluation in matrix["evaluations"]}

    assert evaluations["asr-nemotron-speech"]["status"] == "fails_target"
    assert "protocol_smoke_only_not_accepted" in evaluations["asr-nemotron-speech"]["issues"]
    assert "loopback_speech_evidence_not_accepted" in evaluations["asr-nemotron-speech"]["issues"]
    assert evaluations["tts-magpie-local"]["status"] == "fails_target"
    assert matrix["role_status"]["asr"] == "needs_evidence"
    assert matrix["role_status"]["tts"] == "needs_evidence"


def test_spark_matrix_rejects_kame_oracle_benchmark_without_hermes_model_authority(tmp_path):
    evidence_path = tmp_path / "kame-evidence.json"
    evidence_path.write_text(
        json.dumps(
            [
                {
                    "hardware": "1x DGX Spark",
                    "locality": "local_spark",
                    "verified": True,
                    "measured_at": "2026-06-29T00:00:00Z",
                    "source_artifact": "artifacts/kame/raw.json",
                    "source_artifact_sha256": _source_artifact_sha256("artifacts/kame/raw.json"),
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
            ]
        ),
        encoding="utf-8",
    )

    matrix = build_matrix([evidence_path])
    oracle = next(item for item in matrix["evaluations"] if item["candidate_id"] == "oracle-nemotron3-super-local")

    assert oracle["status"] == "fails_target"
    assert "missing_oracle_authority_proof" in oracle["issues"]
    assert matrix["role_status"]["oracle"] == "needs_evidence"


def test_spark_matrix_rejects_kame_evidence_without_explicit_spark_locality_and_routing(tmp_path):
    evidence_path = tmp_path / "kame-evidence.json"
    common = {
        "verified": True,
        "measured_at": "2026-06-29T00:00:00Z",
        "source_artifact": "artifacts/kame/raw.json",
        "source_artifact_sha256": _source_artifact_sha256("artifacts/kame/raw.json"),
    }
    evidence_path.write_text(
        json.dumps(
            [
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
    oracle = next(item for item in matrix["evaluations"] if item["candidate_id"] == "oracle-nemotron3-super-local")

    assert oracle["status"] == "fails_target"
    assert "hardware_mismatch" in oracle["issues"]
    assert "locality_mismatch" in oracle["issues"]
    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "hardware_mismatch" in matrix["stack_smoke"]["issues"]
    assert "locality_mismatch" in matrix["stack_smoke"]["issues"]
    assert "oracle_not_selected_by_model_flow" in matrix["stack_smoke"]["issues"]
    assert "missing_components:oracle,tts" in matrix["stack_smoke"]["issues"]
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_rejects_unverified_kame_oracle_model_assumption(tmp_path):
    evidence_path = tmp_path / "kame-evidence.json"
    common = {
        "hardware": "1x DGX Spark",
        "locality": "local_spark",
        "verified": True,
        "measured_at": "2026-06-29T00:00:00Z",
        "source_artifact": "artifacts/kame/raw.json",
        "source_artifact_sha256": _source_artifact_sha256("artifacts/kame/raw.json"),
    }
    evidence_path.write_text(
        json.dumps(
            [
                {
                    **common,
                    "kind": "kame_model_assumption_result",
                    "name": "oracle_authority",
                    "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
                    "ok": False,
                    "verified": False,
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
            ]
        ),
        encoding="utf-8",
    )

    matrix = build_matrix([evidence_path])
    oracle = next(item for item in matrix["evaluations"] if item["candidate_id"] == "oracle-nemotron3-super-local")

    assert oracle["status"] == "fails_target"
    assert "model_mismatch" in oracle["issues"]


def test_spark_matrix_rejects_kame_smoke_without_reflex_bypass_and_oracle_authority(tmp_path):
    evidence_path = tmp_path / "kame-evidence.json"
    common = {
        "hardware": "1x DGX Spark",
        "verified": True,
        "measured_at": "2026-06-29T00:00:00Z",
        "source_artifact": "artifacts/kame/raw.json",
        "source_artifact_sha256": _source_artifact_sha256("artifacts/kame/raw.json"),
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
    assert Path(payload["artifacts"]["closure_json"]).exists()
    assert Path(payload["artifacts"]["closure_markdown"]).exists()
    assert Path(payload["artifacts"]["json"]).exists()
    assert Path(payload["artifacts"]["evidence_example"]).exists()
    assert Path(payload["artifacts"]["evidence_scaffold"]).exists()
    assert Path(payload["artifacts"]["evidence_template"]).exists()


def test_spark_matrix_parse_args_accepts_repeated_evidence(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    args = parse_args(["--evidence", str(first), "--evidence", str(second)])

    assert args.evidence == [first, second]
