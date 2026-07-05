from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.voiceops_spark_matrix import (
    build_matrix,
    parse_args,
    refresh_spark_source_hashes,
    write_evidence_scaffold,
    write_matrix,
)


SOURCE_KEYS_BY_ARTIFACT = {
    "artifacts/test/raw.json": [
        "reflex-moshi-s2s",
        "interpreter-gemma4-e2b",
        "oracle-nemotron3-super-local",
        "asr-nemotron-speech",
        "tts-magpie-local",
    ],
    "artifacts/test/stack-smoke.json": ["voiceops_spark_stack_smoke"],
    "artifacts/test/hosted.json": ["oracle-nemotron3-ultra-hosted"],
    "artifacts/test/cartesia.json": ["tts-cartesia-cloud-fallback"],
    "artifacts/kame/raw.json": [
        "reflex-moshi-s2s",
        "interpreter-gemma4-e2b",
        "interpreter-gemma4-e4b",
        "oracle-nemotron3-super-local",
        "asr-nemotron-speech",
        "tts-magpie-local",
        "voiceops_spark_stack_smoke",
    ],
}


def _source_artifact_payload(relative: str) -> dict:
    source_keys = SOURCE_KEYS_BY_ARTIFACT.get(relative, [relative])
    payload = {"redacted": True, "source": relative, "source_keys": source_keys}
    if "voiceops_spark_stack_smoke" in source_keys:
        payload["kame_turns"] = _stack_smoke_source_turns()
    return payload


def _witness_hypothesis(
    source: str,
    *,
    kind: str = "frontend_witness_hypothesis",
    phase: str = "with_raw_audio",
    digest_seed: str = "witness",
    latency_ms: int = 94,
    confidence: float | None = 0.88,
) -> dict:
    return {
        "kind": kind,
        "source": source,
        "text_digest": "sha256:" + hashlib.sha256(digest_seed.encode("utf-8")).hexdigest(),
        "role": "witness_context",
        "authority": "hypothesis",
        "promotion_required": "interpreter_promoted_or_oracle_promoted",
        "tool_authority": False,
        "arrival_phase": phase,
        "latency_ms": latency_ms,
        "confidence": confidence,
        "speaker_or_actor_ref": "speaker:jetha",
        "channel_or_surface_ref": "discord:general",
    }


def _stack_smoke_source_turns() -> list[dict]:
    return [
        {
            "turn_id": "local-001",
            "route": "local",
            "oracle_called": False,
            "audio_segment_ref": "artifact://redacted/local-001.wav",
            "audio_time_range_ms": [100, 900],
            "reflex_transcript_hypothesis": _witness_hypothesis(
                "moshi",
                kind="reflex_transcript_hypothesis",
                digest_seed="redacted local greeting hypothesis",
            ),
            "auxiliary_transcript_hypotheses": [],
        },
        {
            "turn_id": "oracle-001",
            "route": "defer",
            "oracle_called": True,
            "oracle_calls": 1,
            "audio_segment_ref": "artifact://redacted/oracle-001.wav",
            "audio_time_range_ms": [1200, 3300],
            "reflex_transcript_hypothesis": _witness_hypothesis(
                "moshi",
                kind="reflex_transcript_hypothesis",
                digest_seed="redacted reflex hypothesis",
            ),
            "auxiliary_transcript_hypotheses": [
                _witness_hypothesis(
                    "classic_asr_fallback_optional",
                    kind="classic_asr_hypothesis",
                    phase="after_interpreter_start",
                    digest_seed="redacted auxiliary hypothesis",
                    latency_ms=310,
                    confidence=0.76,
                )
            ],
            "interpreter_evidence": {
                "source": "gemma_interpreter",
                "corrected_transcript": "[redacted interpreter correction]",
                "confidence": 0.91,
                "disagreements": ["reflex hypothesis corrected by raw audio"],
            },
            "interpreter_corrected_transcript": "[redacted interpreter correction]",
            "tool_critical_text_source": "gemma_interpreter",
        },
    ]


def _source_artifact_sha256(relative: str) -> str:
    payload = json.dumps(_source_artifact_payload(relative)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _collector_attestation(name: str, redacted_sha256: str) -> dict:
    return {
        "collector_name": "pytest_dgx_spark_fixture",
        "collector_version": "test-v1",
        "run_id": f"test-{name}-run",
        "command_argv": ["pytest", "tests/scripts/test_voiceops_spark_matrix.py"],
        "git_commit": "a" * 40,
        "started_at": "2026-06-29T00:00:00Z",
        "finished_at": "2026-06-29T00:00:01Z",
        "raw_artifact_sha256": "b" * 64,
        "redacted_artifact_sha256": redacted_sha256,
        "parent_manifest_sha256": "c" * 64,
    }


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
        path.write_text(json.dumps(_source_artifact_payload(relative)), encoding="utf-8")


def _base_evidence(candidate_id: str, *, model: str, locality: str = "local_spark") -> dict:
    source_artifact = {
        "oracle-nemotron3-ultra-hosted": "artifacts/test/hosted.json",
        "tts-cartesia-cloud-fallback": "artifacts/test/cartesia.json",
    }.get(candidate_id, "artifacts/test/raw.json")
    evidence = {
        "schema_version": "voiceops.spark_benchmark_evidence.v1",
        "candidate_id": candidate_id,
        "hardware": "1x NVIDIA DGX Spark" if locality == "local_spark" else "hosted",
        "locality": locality,
        "model": model,
        "engine": "test engine",
        "verified": True,
        "measured_at": "2026-06-29T00:00:00Z",
        "source_artifact": source_artifact,
        "source_artifact_sha256": _source_artifact_sha256(source_artifact),
        "collector_attestation": _collector_attestation(
            candidate_id,
            _source_artifact_sha256(source_artifact),
        ),
        "metrics": {},
    }
    if candidate_id == "oracle-nemotron3-super-local":
        evidence["oracle_selected_by"] = "Hermes /model"
    return evidence


def _reflex_evidence() -> dict:
    return {
        **_base_evidence("reflex-moshi-s2s", model="Moshi/PersonaPlex-class low-latency S2S"),
        "metrics": {"ack_latency_ms": 250, "barge_in_stop_ms": 90, "steady_state_memory_gb": 16},
    }


def _interpreter_e2b_evidence() -> dict:
    return {
        **_base_evidence("interpreter-gemma4-e2b", model="Gemma 4 E2B audio-native interpreter"),
        "metrics": {"audio_interpretation_ms": 900, "evidence_patch_ms": 1200, "steady_state_memory_gb": 24},
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
        "source_artifact_sha256": _source_artifact_sha256("artifacts/test/stack-smoke.json"),
        "collector_attestation": _collector_attestation(
            "stack-smoke",
            _source_artifact_sha256("artifacts/test/stack-smoke.json"),
        ),
        "oracle_selected_by": "Hermes /model",
        "oracle_authority_routes": ["tools", "files", "memory", "project_context"],
        "interface_input_sources": ["native_audio"],
        "reflex_providers": ["moshi"],
        "interpreter_providers": ["vllm", "gemma"],
        "auxiliary_transcript_sources": ["moshi_hypothesis", "classic_asr_fallback_optional"],
        "components": {
            "reflex": True,
            "interpreter": True,
            "oracle": True,
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
    assert matrix["policy"]["raw_audio_primary_interpreter_evidence"] is True
    assert matrix["policy"]["transcript_hypotheses_are_witness_context"] is True
    assert matrix["policy"]["transcript_only_counts_for_one_spark_readiness"] is False
    assert {
        "kind",
        "source",
        "text_digest",
        "role",
        "authority",
        "promotion_required",
        "tool_authority",
        "arrival_phase",
        "latency_ms",
        "confidence",
        "speaker_or_actor_ref",
        "channel_or_surface_ref",
    } <= set(matrix["policy"]["transcript_hypothesis_required_fields"])
    assert matrix["policy"]["transcript_hypothesis_contract"] == {
        "role": "witness_context",
        "authority": "hypothesis",
        "promotion_required": "interpreter_promoted_or_oracle_promoted",
        "tool_authority": False,
    }
    assert matrix["ready_for_one_spark_demo"] is False
    assert matrix["role_status"] == {
        "interpreter": "needs_evidence",
        "oracle": "needs_evidence",
        "reflex": "needs_evidence",
        "tts": "needs_evidence",
    }
    assert {candidate["candidate_id"] for candidate in matrix["candidates"]} >= {
        "reflex-moshi-s2s",
        "interpreter-gemma4-e2b",
        "interpreter-gemma4-e4b",
        "oracle-nemotron3-super-local",
        "oracle-nemotron3-ultra-hosted",
        "asr-nemotron-speech",
        "tts-magpie-local",
        "tts-cartesia-cloud-fallback",
    }
    reflex_candidates = {candidate["candidate_id"]: candidate for candidate in matrix["candidates"] if candidate["role"] == "reflex"}
    assert reflex_candidates["reflex-moshi-s2s"]["priority"] == 1
    assert reflex_candidates["reflex-moshi-s2s"]["locality"] == "local_spark"
    interpreter_candidates = {
        candidate["candidate_id"]: candidate for candidate in matrix["candidates"] if candidate["role"] == "interpreter"
    }
    assert interpreter_candidates["interpreter-gemma4-e2b"]["priority"] == 1
    assert interpreter_candidates["interpreter-gemma4-e4b"]["priority"] == 1
    assert interpreter_candidates["interpreter-gemma4-e4b"]["locality"] == "local_spark"
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
        "scaffold_source_reflex-moshi-s2s",
        "scaffold_source_interpreter-gemma4-e2b",
        "scaffold_source_interpreter-gemma4-e4b",
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
        "interpreter:needs_evidence",
        "oracle:needs_evidence",
        "reflex:needs_evidence",
        "tts:needs_evidence",
        "all_local_stack_smoke",
    ]
    assert closure["missing_roles"] == [
        "interpreter:needs_evidence",
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
        "interpreter_providers",
        "auxiliary_transcript_sources",
        "metrics.local_turns",
        "metrics.local_turn_oracle_calls",
        "metrics.oracle_bound_turns",
        "metrics.oracle_bound_oracle_calls",
    } <= set(closure["required_stack_smoke_fields"])
    assert closure["all_local_stack_smoke"]["required_components"] == ["reflex", "interpreter", "oracle", "tts", "sidecar"]
    assert closure["all_local_stack_smoke"]["required_reflex_provider"] == "s2s_or_timing"
    assert closure["all_local_stack_smoke"]["required_interpreter_provider"] == "gemma_audio"
    assert closure["all_local_stack_smoke"]["auxiliary_transcript_sources_optional"] is True
    assert closure["all_local_stack_smoke"]["required_source_artifact_contract"][
        "requires_transcript_hypotheses_witness_context_contract"
    ] is True
    assert closure["all_local_stack_smoke"]["required_source_artifact_contract"][
        "raw_transcript_text_counts_for_readiness"
    ] is False
    assert {
        "kind",
        "source",
        "text_digest",
        "role",
        "authority",
        "promotion_required",
        "tool_authority",
        "arrival_phase",
        "latency_ms",
        "confidence",
        "speaker_or_actor_ref",
        "channel_or_surface_ref",
    } <= set(
        closure["all_local_stack_smoke"]["required_source_artifact_contract"][
            "required_transcript_hypothesis_fields"
        ]
    )
    assert closure["evidence_contract"]["preferred_local_oracle_candidate_id"] == "oracle-nemotron3-super-local"
    assert closure["evidence_contract"]["preferred_local_oracle_candidate_model"] == "Nemotron 3 Super"
    assert "preferred_local_oracle_model" not in closure["evidence_contract"]
    assert closure["evidence_contract"]["non_counting_fallback_oracle_models"] == ["Nemotron 3 Ultra"]
    assert closure["evidence_contract"]["source_artifacts_must_exist"] is True
    assert closure["evidence_contract"]["source_artifact_readable"] is True
    assert closure["evidence_contract"]["source_artifact_sha256_must_match"] is True
    assert closure["evidence_contract"]["benchmark_evidence_not_before"] == "2026-06-29T00:00:00+00:00"
    assert closure["evidence_contract"]["measured_at_must_be_in_evidence_window"] is True
    assert closure["evidence_contract"]["collector_attestation_timestamps_must_be_in_evidence_window"] is True
    assert closure["evidence_contract"]["collector_attestation_required_for_one_spark_readiness"] is True
    assert closure["evidence_contract"]["placeholder_collector_attestation_accepted"] is False
    assert closure["evidence_contract"]["example_only_accepted"] is False
    assert closure["evidence_contract"]["source_artifact_resolution"].endswith("supplied benchmark evidence file")
    assert closure["benchmark_evidence_shape"]["evidence"][0]["schema_version"] == "voiceops.spark_benchmark_evidence.v1"
    assert closure["benchmark_evidence_shape"]["evidence"][0]["candidate_id"] == "oracle-nemotron3-super-local"
    assert closure["benchmark_evidence_shape"]["evidence"][1]["kind"] == "voiceops_spark_stack_smoke"
    assert scaffold["scaffold_entries"]["reflex-moshi-s2s"]["role"] == "reflex"
    assert scaffold["scaffold_entries"]["reflex-moshi-s2s"]["source_artifact"] == "sources/reflex-moshi-s2s-raw.json"
    assert scaffold["scaffold_entries"]["reflex-moshi-s2s"]["secret_values_allowed"] is False
    assert scaffold["scaffold_entries"]["reflex-moshi-s2s"]["full_phone_numbers_allowed"] is False
    assert {
        "metric": "ack_latency_ms",
        "operator": "<=",
        "value": 350,
        "unit": "ms",
    } in scaffold["scaffold_entries"]["reflex-moshi-s2s"]["required_metrics"]
    assert scaffold["scaffold_entries"]["oracle-nemotron3-super-local"]["role"] == "oracle"
    assert scaffold["scaffold_entries"]["oracle-nemotron3-super-local"]["locality"] == "local_spark"
    assert scaffold["scaffold_entries"]["oracle-nemotron3-super-local"][
        "collector_attestation_required_fields"
    ] == [
        "collector_name",
        "collector_version",
        "run_id",
        "command_argv",
        "git_commit",
        "started_at",
        "finished_at",
        "raw_artifact_sha256",
        "redacted_artifact_sha256",
        "parent_manifest_sha256",
    ]
    assert scaffold["scaffold_entries"]["voiceops_spark_stack_smoke"]["required_components"] == [
        "reflex",
        "interpreter",
        "oracle",
        "tts",
        "sidecar",
    ]
    assert scaffold["scaffold_entries"]["voiceops_spark_stack_smoke"][
        "source_artifact_must_include_kame_turn_contract"
    ] is True
    assert scaffold["scaffold_entries"]["voiceops_spark_stack_smoke"][
        "transcript_hypotheses_are_witness_context"
    ] is True
    assert scaffold["completion_check"]["remove_every_example_only_marker"] is True
    assert "--refresh-source-hashes" in scaffold["completion_check"]["refresh_source_hashes_command"]
    assert "--lint-evidence" in scaffold["completion_check"]["lint_command"]
    assert "spark-benchmark-scaffold/spark-benchmark-evidence.json" in scaffold["completion_check"][
        "validate_command"
    ]
    assert any("Hosted fallback rows" in note for note in scaffold["operator_notes"])
    assert any("ASR rows are optional witness/fallback evidence" in note for note in scaffold["operator_notes"])
    assert "scripts/dgx_spark_gemma4_voice_eval.sh" == closure["rerun_commands"]["dgx_eval"]
    assert "--refresh-source-hashes" in closure["rerun_commands"]["refresh_source_hashes"]
    assert "--lint-evidence" in closure["rerun_commands"]["lint_evidence"]
    assert "spark-benchmark-scaffold/spark-benchmark-evidence.json" in closure["rerun_commands"]["lint_evidence"]
    assert "VoiceOps Milestone 4 Spark Matrix Closure" in closure_markdown
    assert "spark-benchmark-scaffold/spark-benchmark-evidence.json" in closure_markdown
    assert "path/to/spark-benchmark-evidence.json" not in closure_markdown
    assert "hosted or multi-Spark Nemotron 3 Ultra fallback evidence" in closure_markdown
    assert '"evidence": [' in closure_markdown
    assert "voiceops.spark_benchmark_evidence.v1" in closure_markdown
    assert "oracle_authority_routes" in closure_markdown
    assert "source_artifact" in closure_markdown
    assert "source_artifact_sha256" in closure_markdown
    assert "collector_attestation" in closure_markdown
    assert "collector_name" in closure_markdown
    assert "collector_version" in closure_markdown
    assert "command_argv" in closure_markdown
    assert "redacted_artifact_sha256" in closure_markdown
    assert "parent_manifest_sha256" in closure_markdown
    assert "source_artifacts_must_exist" in closure_markdown
    assert "VoiceOps DGX Spark Operator Runbook" in operator_runbook
    assert "scripts/dgx_spark_gemma4_voice_eval.sh" in operator_runbook
    assert "spark-benchmark-scaffold/spark-benchmark-evidence.json" in operator_runbook
    assert "path/to/spark-benchmark-evidence.json" not in operator_runbook
    assert "uv run python scripts/voiceops_spark_matrix.py" in operator_runbook
    assert "--refresh-source-hashes" in operator_runbook
    assert "uv run python scripts/voiceops_plan_run.py" in operator_runbook
    assert "collector_attestation.redacted_artifact_sha256" in operator_runbook
    assert "Nemotron 3 Super is the preferred one-Spark oracle candidate" in operator_runbook
    assert "Hosted or multi-Spark Nemotron 3 Ultra evidence" in operator_runbook
    assert "`loopback_smoke_bridge`" in operator_runbook
    assert "protocol-only smoke checks" in operator_runbook
    assert "HERMES_DGX_SPARK_ASR_MODULE" in operator_runbook
    assert "HERMES_DGX_SPARK_TTS_ADAPTER" in operator_runbook
    assert "must remain unverified for local transcript/TTS evidence" in operator_runbook
    assert "`speech_end_to_first_audio_ms <= 1500`" in operator_runbook
    assert "`barge_in_stop_ms <= 150`" in operator_runbook
    assert "`local_turn_oracle_calls == 0`" in operator_runbook
    assert "`oracle_bound_oracle_calls >= oracle_bound_turns`" in operator_runbook
    assert example["example_only"] is True
    assert all(item["example_only"] is True for item in example["evidence"])
    assert scaffold["example_only"] is True
    assert scaffold["evidence"][0]["source_artifact"] == "sources/reflex-moshi-s2s-raw.json"
    scaffold_matrix = build_matrix([scaffold_path])
    scaffold_evaluations = {
        evaluation["candidate_id"]: evaluation
        for evaluation in scaffold_matrix["evaluations"]
    }
    assert scaffold_matrix["ready_for_one_spark_demo"] is False
    assert "example_only_evidence_not_accepted" in scaffold_evaluations["reflex-moshi-s2s"]["issues"]
    assert "example_only_evidence_not_accepted" in scaffold_evaluations["interpreter-gemma4-e2b"]["issues"]
    assert "example_only_evidence_not_accepted" in scaffold_evaluations["interpreter-gemma4-e4b"]["issues"]
    assert "source_artifact_not_found" not in scaffold_evaluations["interpreter-gemma4-e2b"]["issues"]
    assert "source_artifact_not_found" not in scaffold_evaluations["interpreter-gemma4-e4b"]["issues"]
    assert "source_artifact_not_found" not in scaffold_matrix["stack_smoke"]["issues"]
    scaffold_source = json.loads(Path(paths["scaffold_source_interpreter-gemma4-e2b"]).read_text(encoding="utf-8"))
    assert scaffold_source["redacted"] is True
    assert template["evidence"][0]["verified"] is False
    assert template["evidence"][0]["schema_version"] == "voiceops.spark_benchmark_evidence.v1"
    assert template["evidence"][0]["model"]
    assert "source_artifact_sha256" in template["evidence"][0]
    assert "collector_attestation" in template["evidence"][0]
    assert template["evidence"][-1]["kind"] == "voiceops_spark_stack_smoke"
    assert "replace null metrics" in template["evidence"][0]["notes"]


def test_spark_scaffold_removes_stale_example_sources_but_keeps_measured_sources(tmp_path):
    sources_dir = tmp_path / "spark-benchmark-scaffold" / "sources"
    sources_dir.mkdir(parents=True)
    stale_example = sources_dir / "reflex-gemma4-e2b-raw.json"
    stale_example.write_text(json.dumps({"example_only": True, "redacted": True}), encoding="utf-8")
    measured_extra = sources_dir / "operator-measured-extra.json"
    measured_extra.write_text(json.dumps({"example_only": False, "redacted": True}), encoding="utf-8")

    paths = write_evidence_scaffold(tmp_path)

    assert not stale_example.exists()
    assert measured_extra.exists()
    assert Path(paths["scaffold_source_reflex-moshi-s2s"]).exists()
    scaffold = json.loads(Path(paths["evidence_scaffold"]).read_text(encoding="utf-8"))
    source_refs = {item["source_artifact"] for item in scaffold["evidence"]}
    assert "sources/reflex-gemma4-e2b-raw.json" not in source_refs
    assert "sources/reflex-moshi-s2s-raw.json" in source_refs


def test_spark_matrix_validates_matching_evidence(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "evidence": [
                    _reflex_evidence(),
                    _interpreter_e2b_evidence(),
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
    assert evaluations["reflex-moshi-s2s"]["status"] == "validated"
    assert evaluations["interpreter-gemma4-e2b"]["status"] == "validated"
    assert evaluations["oracle-nemotron3-super-local"]["status"] == "validated"
    assert evaluations["asr-nemotron-speech"]["status"] == "validated"
    assert evaluations["tts-magpie-local"]["status"] == "validated"
    assert matrix["stack_smoke"]["status"] == "validated"

    paths = write_matrix(tmp_path / "validated", matrix)
    closure = json.loads(Path(paths["closure_json"]).read_text(encoding="utf-8"))
    assert closure["status"] == "complete"
    assert closure["ready"] is True
    assert closure["missing_gates"] == []


def test_spark_matrix_requires_single_coherent_candidate_record(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    fast_decode_bad_prefill = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    fast_decode_bad_prefill["metrics"] = {
        "decode_tok_s": 24,
        "prefill_tok_s": 1000,
        "first_token_ms": 2100,
        "steady_state_memory_gb": 86,
    }
    fast_prefill_bad_decode = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    fast_prefill_bad_decode["metrics"] = {
        "decode_tok_s": 10,
        "prefill_tok_s": 3100,
        "first_token_ms": 2100,
        "steady_state_memory_gb": 86,
    }
    evidence_path.write_text(
        json.dumps({"evidence": [fast_decode_bad_prefill, fast_prefill_bad_decode]}),
        encoding="utf-8",
    )

    matrix = build_matrix([evidence_path])
    evaluation = next(item for item in matrix["evaluations"] if item["candidate_id"] == "oracle-nemotron3-super-local")

    assert evaluation["status"] == "fails_target"
    assert "no_single_evidence_record_satisfies_targets" in evaluation["issues"]
    assert "target_failed:decode_tok_s" in evaluation["issues"]
    assert "target_failed:prefill_tok_s" in evaluation["issues"]
    assert [record["status"] for record in evaluation["record_results"]] == ["rejected", "rejected"]
    assert matrix["role_status"]["oracle"] == "needs_evidence"


def test_spark_matrix_reports_missing_evidence_file_without_crashing(tmp_path):
    evidence_path = tmp_path / "missing-evidence.json"

    matrix = build_matrix([evidence_path])
    paths = write_matrix(tmp_path / "matrix", matrix)
    closure_markdown = Path(paths["closure_markdown"]).read_text(encoding="utf-8")

    assert matrix["ready_for_one_spark_demo"] is False
    assert matrix["evidence_load_issues"] == [f"evidence_file_not_found:{evidence_path}"]
    assert f"evidence_file_not_found:{evidence_path}" in closure_markdown


def test_spark_matrix_example_is_not_accepted_as_proof(tmp_path):
    paths = write_matrix(tmp_path, build_matrix())

    matrix = build_matrix([Path(paths["evidence_example"])])
    evaluations = {evaluation["candidate_id"]: evaluation for evaluation in matrix["evaluations"]}

    assert matrix["ready_for_one_spark_demo"] is False
    assert evaluations["reflex-moshi-s2s"]["status"] == "fails_target"
    assert evaluations["interpreter-gemma4-e2b"]["status"] == "fails_target"
    assert evaluations["oracle-nemotron3-super-local"]["status"] == "fails_target"
    assert evaluations["asr-nemotron-speech"]["status"] == "fails_target"
    assert evaluations["tts-magpie-local"]["status"] == "fails_target"
    assert "example_only_evidence_not_accepted" in evaluations["oracle-nemotron3-super-local"]["issues"]
    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "example_only_evidence_not_accepted" in matrix["stack_smoke"]["issues"]
    assert matrix["role_status"] == {
        "interpreter": "needs_evidence",
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
                    _reflex_evidence(),
                    _interpreter_e2b_evidence(),
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
    assert "example_only_evidence_not_accepted" in evaluations["interpreter-gemma4-e2b"]["issues"]
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
    assert f"source_artifact_not_found_path:{tmp_path / 'missing/raw-oracle.json'}" in evaluation["issues"]
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


def test_spark_matrix_rejects_candidate_with_missing_source_artifact_identity(tmp_path):
    source_path = tmp_path / "artifacts/test/no-identity.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(json.dumps({"redacted": True, "source": "generic raw output"}), encoding="utf-8")
    evidence_path = tmp_path / "evidence.json"
    evidence = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    evidence["source_artifact"] = "artifacts/test/no-identity.json"
    evidence["source_artifact_sha256"] = source_sha256
    evidence["collector_attestation"]["redacted_artifact_sha256"] = source_sha256
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
    assert "source_artifact_identity_missing" in evaluation["issues"]
    assert matrix["role_status"]["oracle"] == "needs_evidence"


def test_spark_matrix_rejects_candidate_with_mismatched_source_artifact_identity(tmp_path):
    source_path = tmp_path / "artifacts/test/wrong-identity.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps({"redacted": True, "source": "wrong raw output", "source_key": "interpreter-gemma4-e2b"}),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "evidence.json"
    evidence = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    evidence["source_artifact"] = "artifacts/test/wrong-identity.json"
    evidence["source_artifact_sha256"] = source_sha256
    evidence["collector_attestation"]["redacted_artifact_sha256"] = source_sha256
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
    assert "source_artifact_identity_mismatch" in evaluation["issues"]
    assert matrix["role_status"]["oracle"] == "needs_evidence"


def test_spark_matrix_rejects_candidate_source_artifact_with_only_generic_kind_identity(tmp_path):
    source_path = tmp_path / "artifacts/test/generic-kind-identity.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps({"redacted": True, "source": "generic KAME output", "source_key": "kame_benchmark_result"}),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "evidence.json"
    evidence = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    evidence["kind"] = "kame_benchmark_result"
    evidence["source_artifact"] = "artifacts/test/generic-kind-identity.json"
    evidence["source_artifact_sha256"] = source_sha256
    evidence["collector_attestation"]["redacted_artifact_sha256"] = source_sha256
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
    assert "source_artifact_identity_mismatch" in evaluation["issues"]
    assert matrix["role_status"]["oracle"] == "needs_evidence"


def test_spark_matrix_rejects_candidate_with_stale_source_and_attestation_hashes(tmp_path):
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    source_path = sources_dir / "oracle.json"
    source_path.write_text(
        json.dumps({"redacted": True, "source": "old oracle output"}, sort_keys=True),
        encoding="utf-8",
    )
    old_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    evidence = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    evidence["source_artifact"] = "sources/oracle.json"
    evidence["source_artifact_sha256"] = old_sha256
    evidence["collector_attestation"]["redacted_artifact_sha256"] = old_sha256
    evidence["metrics"] = {
        "decode_tok_s": 24,
        "prefill_tok_s": 3100,
        "first_token_ms": 2100,
        "steady_state_memory_gb": 86,
    }
    evidence_path = tmp_path / "spark-benchmark-evidence.json"
    evidence_path.write_text(json.dumps({"evidence": [evidence]}, indent=2, sort_keys=True), encoding="utf-8")
    source_path.write_text(
        json.dumps({"redacted": True, "source": "updated oracle output"}, sort_keys=True),
        encoding="utf-8",
    )

    matrix = build_matrix([evidence_path])
    evaluation = next(item for item in matrix["evaluations"] if item["candidate_id"] == "oracle-nemotron3-super-local")

    assert evaluation["status"] == "fails_target"
    assert "source_artifact_sha256_mismatch" in evaluation["issues"]
    assert "collector_attestation_redacted_sha256_mismatch" in evaluation["issues"]
    assert matrix["role_status"]["oracle"] == "needs_evidence"


def test_spark_matrix_refresh_source_hashes_updates_candidate_attestation(tmp_path):
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    source_path = sources_dir / "oracle.json"
    source_path.write_text(
        json.dumps(
            {"redacted": True, "source": "old oracle output", "source_key": "oracle-nemotron3-super-local"},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    old_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    evidence = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    evidence["source_artifact"] = "sources/oracle.json"
    evidence["source_artifact_sha256"] = old_sha256
    evidence["collector_attestation"]["redacted_artifact_sha256"] = old_sha256
    evidence_path = tmp_path / "spark-benchmark-evidence.json"
    evidence_path.write_text(json.dumps({"evidence": [evidence]}, indent=2, sort_keys=True), encoding="utf-8")
    source_path.write_text(
        json.dumps(
            {"redacted": True, "source": "updated oracle output", "source_key": "oracle-nemotron3-super-local"},
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = refresh_spark_source_hashes(evidence_path)
    refreshed = json.loads(evidence_path.read_text(encoding="utf-8"))["evidence"][0]
    new_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    assert result["ok"] is True
    assert result["schema_version"] == "voiceops.spark_evidence_hash_refresh.v1"
    assert result["network_io"] is False
    assert result["spark_execution"] is False
    assert result["artifact_writes"] is True
    assert result["updates"][0]["item"] == "oracle-nemotron3-super-local"
    assert result["updates"][0]["changed"] is True
    assert result["updates"][0]["collector_attestation_changed"] is True
    assert refreshed["source_artifact_sha256"] == new_sha256
    assert refreshed["collector_attestation"]["redacted_artifact_sha256"] == new_sha256


def test_spark_matrix_rejects_candidate_with_source_artifact_not_explicitly_redacted(tmp_path):
    source_path = tmp_path / "artifacts/test/policy-only-source.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "redaction_policy": "operator claims this artifact was scrubbed",
                "source_key": "oracle-nemotron3-super-local",
            }
        ),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "evidence.json"
    evidence = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    evidence["source_artifact"] = "artifacts/test/policy-only-source.json"
    evidence["source_artifact_sha256"] = source_sha256
    evidence["collector_attestation"]["redacted_artifact_sha256"] = source_sha256
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
    assert "source_artifact_not_redacted" in evaluation["issues"]
    assert matrix["role_status"]["oracle"] == "needs_evidence"


def test_spark_matrix_rejects_candidate_source_artifact_with_secret_and_phone_like_values(tmp_path):
    source_path = tmp_path / "artifacts/test/leaky-source.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "redacted": True,
                "source_key": "oracle-nemotron3-super-local",
                "api_key": "sk-" + ("a" * 24),
                "transcript": "call back at +14155552671",
            }
        ),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "evidence.json"
    evidence = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    evidence["source_artifact"] = "artifacts/test/leaky-source.json"
    evidence["source_artifact_sha256"] = source_sha256
    evidence["collector_attestation"]["redacted_artifact_sha256"] = source_sha256
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
    assert "source_artifact_contains_likely_secret" in evaluation["issues"]
    assert "source_artifact_contains_phone_like_value" in evaluation["issues"]
    assert matrix["role_status"]["oracle"] == "needs_evidence"


def test_spark_matrix_refresh_source_hashes_rejects_unsafe_source_without_writing(tmp_path):
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    source_path = sources_dir / "oracle.json"
    source_path.write_text(
        json.dumps(
            {"redacted": True, "source": "old oracle output", "source_key": "oracle-nemotron3-super-local"},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    old_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    evidence = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    evidence["source_artifact"] = "sources/oracle.json"
    evidence["source_artifact_sha256"] = old_sha256
    evidence["collector_attestation"]["redacted_artifact_sha256"] = old_sha256
    evidence_path = tmp_path / "spark-benchmark-evidence.json"
    evidence_path.write_text(json.dumps({"evidence": [evidence]}, indent=2, sort_keys=True), encoding="utf-8")
    before_refresh = evidence_path.read_text(encoding="utf-8")
    source_path.write_text(
        json.dumps(
            {
                "redaction_policy": "policy text is not an explicit redacted marker",
                "source_key": "oracle-nemotron3-super-local",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = refresh_spark_source_hashes(evidence_path)
    refreshed = json.loads(evidence_path.read_text(encoding="utf-8"))["evidence"][0]

    assert result["ok"] is False
    assert result["artifact_writes"] is False
    assert result["updates"] == []
    assert result["issues"] == ["oracle-nemotron3-super-local:source_artifact_not_redacted"]
    assert evidence_path.read_text(encoding="utf-8") == before_refresh
    assert refreshed["source_artifact_sha256"] == old_sha256
    assert refreshed["collector_attestation"]["redacted_artifact_sha256"] == old_sha256


def test_spark_matrix_refresh_source_hashes_rejects_leaky_redacted_source_without_writing(tmp_path):
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    source_path = sources_dir / "oracle.json"
    source_path.write_text(
        json.dumps(
            {"redacted": True, "source": "old oracle output", "source_key": "oracle-nemotron3-super-local"},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    old_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    evidence = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    evidence["source_artifact"] = "sources/oracle.json"
    evidence["source_artifact_sha256"] = old_sha256
    evidence["collector_attestation"]["redacted_artifact_sha256"] = old_sha256
    evidence_path = tmp_path / "spark-benchmark-evidence.json"
    evidence_path.write_text(json.dumps({"evidence": [evidence]}, indent=2, sort_keys=True), encoding="utf-8")
    before_refresh = evidence_path.read_text(encoding="utf-8")
    source_path.write_text(
        json.dumps(
            {
                "redacted": True,
                "source_key": "oracle-nemotron3-super-local",
                "api_key": "sk-" + ("b" * 24),
                "caller_phone": "(415) 555-2671",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = refresh_spark_source_hashes(evidence_path)
    refreshed = json.loads(evidence_path.read_text(encoding="utf-8"))["evidence"][0]

    assert result["ok"] is False
    assert result["artifact_writes"] is False
    assert result["updates"] == []
    assert "oracle-nemotron3-super-local:source_artifact_contains_likely_secret" in result["issues"]
    assert "oracle-nemotron3-super-local:source_artifact_contains_phone_like_value" in result["issues"]
    assert evidence_path.read_text(encoding="utf-8") == before_refresh
    assert refreshed["source_artifact_sha256"] == old_sha256
    assert refreshed["collector_attestation"]["redacted_artifact_sha256"] == old_sha256


def test_spark_matrix_rejects_candidate_without_collector_attestation(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    evidence.pop("collector_attestation")
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
    assert "missing_collector_attestation" in evaluation["issues"]
    assert matrix["role_status"]["oracle"] == "needs_evidence"


def test_spark_matrix_rejects_candidate_with_placeholder_collector_attestation(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    evidence["collector_attestation"]["example_only"] = True
    evidence["collector_attestation"]["collector_version"] = "example"
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
    assert "collector_attestation_example_only_not_accepted" in evaluation["issues"]
    assert "collector_attestation_invalid:collector_version" in evaluation["issues"]
    assert matrix["role_status"]["oracle"] == "needs_evidence"


def test_spark_matrix_rejects_candidate_with_inverted_collector_attestation_window(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    evidence["collector_attestation"]["started_at"] = "2026-06-29T00:00:02Z"
    evidence["collector_attestation"]["finished_at"] = "2026-06-29T00:00:01Z"
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
    assert "collector_attestation_invalid:timestamp_window" in evaluation["issues"]
    assert matrix["role_status"]["oracle"] == "needs_evidence"


def test_spark_matrix_rejects_candidate_with_sensitive_collector_attestation_command_argv(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    evidence["collector_attestation"]["command_argv"] = [
        "voiceops-spark-collector",
        "--notify=+15551234567",
    ]
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
    assert "collector_attestation_secret_or_phone_like_command_argv" in evaluation["issues"]
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


def test_spark_matrix_rejects_stale_candidate_timestamps(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence = _base_evidence("oracle-nemotron3-super-local", model="Nemotron 3 Super")
    evidence["measured_at"] = "2026-06-28T23:59:59Z"
    evidence["collector_attestation"]["started_at"] = "2026-06-28T23:59:57Z"
    evidence["collector_attestation"]["finished_at"] = "2026-06-28T23:59:58Z"
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
    assert "stale_measured_at" in evaluation["issues"]
    assert "collector_attestation_stale:started_at" in evaluation["issues"]
    assert "collector_attestation_stale:finished_at" in evaluation["issues"]
    assert matrix["role_status"]["oracle"] == "needs_evidence"
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_rejects_ultra_model_for_super_local_oracle_gate(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "evidence": [
                    _reflex_evidence(),
                    {
                        **_base_evidence("interpreter-gemma4-e2b", model="Gemma 4 E2B audio-native interpreter"),
                        "metrics": {"audio_interpretation_ms": 900, "evidence_patch_ms": 1200, "steady_state_memory_gb": 24},
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
    assert matrix["role_status"]["interpreter"] == "validated"
    assert matrix["role_status"]["oracle"] == "needs_evidence"
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
                    _reflex_evidence(),
                    {
                        **_base_evidence("interpreter-gemma4-e2b", model="Gemma 4 E2B audio-native interpreter"),
                        "metrics": {"audio_interpretation_ms": 900, "evidence_patch_ms": 1200, "steady_state_memory_gb": 24},
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
        "interpreter": "validated",
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
    assert f"source_artifact_not_found_path:{tmp_path / 'missing/stack-smoke.json'}" in matrix["stack_smoke"]["issues"]
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


def test_spark_matrix_rejects_stack_smoke_with_mismatched_source_artifact_identity(tmp_path):
    source_path = tmp_path / "artifacts/test/wrong-stack-identity.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps({"redacted": True, "source": "wrong stack raw output", "source_key": "tts-magpie-local"}),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "evidence.json"
    incomplete = _stack_smoke()
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    incomplete["source_artifact"] = "artifacts/test/wrong-stack-identity.json"
    incomplete["source_artifact_sha256"] = source_sha256
    incomplete["collector_attestation"]["redacted_artifact_sha256"] = source_sha256
    evidence_path.write_text(json.dumps({"evidence": [incomplete]}), encoding="utf-8")

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "source_artifact_identity_mismatch" in matrix["stack_smoke"]["issues"]
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_rejects_stack_smoke_source_without_kame_turn_contract(tmp_path):
    source_path = tmp_path / "artifacts/test/stack-smoke-no-turns.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "redacted": True,
                "source_key": "voiceops_spark_stack_smoke",
                "summary": "component-only stack smoke, no turn evidence",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    evidence_path = tmp_path / "evidence.json"
    incomplete = _stack_smoke()
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    incomplete["source_artifact"] = "artifacts/test/stack-smoke-no-turns.json"
    incomplete["source_artifact_sha256"] = source_sha256
    incomplete["collector_attestation"]["redacted_artifact_sha256"] = source_sha256
    evidence_path.write_text(json.dumps({"evidence": [incomplete]}), encoding="utf-8")

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "source_artifact_missing_kame_turns" in matrix["stack_smoke"]["issues"]
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_rejects_stack_smoke_source_with_unlabeled_transcript_and_missing_interpreter(tmp_path):
    source_path = tmp_path / "artifacts/test/stack-smoke-weak-turns.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_payload = _source_artifact_payload("artifacts/test/stack-smoke.json")
    source_payload["source_key"] = "voiceops_spark_stack_smoke"
    source_payload["source_keys"] = ["voiceops_spark_stack_smoke"]
    source_payload["kame_turns"][0]["reflex_transcript_hypothesis"] = "hello"
    source_payload["kame_turns"][1].pop("interpreter_evidence")
    source_payload["kame_turns"][1].pop("interpreter_corrected_transcript")
    source_payload["kame_turns"][1]["auxiliary_transcript_hypotheses"] = [
        {"source": "classic_asr_fallback_optional", "text": "[redacted auxiliary hypothesis]"}
    ]
    source_payload["kame_turns"][1]["tool_critical_text_source"] = "asr"
    source_path.write_text(json.dumps(source_payload, sort_keys=True), encoding="utf-8")
    evidence_path = tmp_path / "evidence.json"
    incomplete = _stack_smoke()
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    incomplete["source_artifact"] = "artifacts/test/stack-smoke-weak-turns.json"
    incomplete["source_artifact_sha256"] = source_sha256
    incomplete["collector_attestation"]["redacted_artifact_sha256"] = source_sha256
    evidence_path.write_text(json.dumps({"evidence": [incomplete]}), encoding="utf-8")

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "source_artifact_reflex_transcript_not_hypothesis" in matrix["stack_smoke"]["issues"]
    assert "source_artifact_auxiliary_transcript_not_hypothesis" in matrix["stack_smoke"]["issues"]
    assert "source_artifact_missing_interpreter_corrected_transcript" in matrix["stack_smoke"]["issues"]
    assert "source_artifact_missing_gemma_interpreter_evidence" in matrix["stack_smoke"]["issues"]
    assert "source_artifact_tool_critical_text_not_interpreter_or_oracle_judgment" in matrix["stack_smoke"]["issues"]
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_accepts_stack_smoke_source_with_canonical_transcript_hypotheses(tmp_path):
    source_path = tmp_path / "artifacts/test/stack-smoke-canonical-turns.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_payload = _source_artifact_payload("artifacts/test/stack-smoke.json")
    source_payload["source_key"] = "voiceops_spark_stack_smoke"
    source_payload["source_keys"] = ["voiceops_spark_stack_smoke"]
    for turn in source_payload["kame_turns"]:
        hypotheses = [turn.pop("reflex_transcript_hypothesis")]
        hypotheses.extend(turn.pop("auxiliary_transcript_hypotheses", []))
        turn["transcript_hypotheses"] = hypotheses
    source_path.write_text(json.dumps(source_payload, sort_keys=True), encoding="utf-8")
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    evidence_path = tmp_path / "evidence.json"
    stack_smoke = _stack_smoke()
    stack_smoke["source_artifact"] = "artifacts/test/stack-smoke-canonical-turns.json"
    stack_smoke["source_artifact_sha256"] = source_sha256
    stack_smoke["collector_attestation"]["redacted_artifact_sha256"] = source_sha256
    evidence_path.write_text(json.dumps({"evidence": [stack_smoke]}), encoding="utf-8")

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == "validated"
    assert "source_artifact_reflex_transcript_not_hypothesis" not in matrix["stack_smoke"]["issues"]
    assert "source_artifact_auxiliary_transcript_not_hypothesis" not in matrix["stack_smoke"]["issues"]
    assert "source_artifact_transcript_hypotheses_not_hypothesis" not in matrix["stack_smoke"]["issues"]


def test_spark_matrix_rejects_stack_smoke_source_with_incomplete_witness_contract(tmp_path):
    source_path = tmp_path / "artifacts/test/stack-smoke-incomplete-witness.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_payload = _source_artifact_payload("artifacts/test/stack-smoke.json")
    source_payload["source_key"] = "voiceops_spark_stack_smoke"
    source_payload["source_keys"] = ["voiceops_spark_stack_smoke"]
    source_payload["kame_turns"][0]["reflex_transcript_hypothesis"].pop("text_digest")
    source_payload["kame_turns"][1]["auxiliary_transcript_hypotheses"][0]["tool_authority"] = True
    source_payload["kame_turns"][1]["auxiliary_transcript_hypotheses"][0]["role"] = "verified_transcript"
    source_path.write_text(json.dumps(source_payload, sort_keys=True), encoding="utf-8")
    evidence_path = tmp_path / "evidence.json"
    incomplete = _stack_smoke()
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    incomplete["source_artifact"] = "artifacts/test/stack-smoke-incomplete-witness.json"
    incomplete["source_artifact_sha256"] = source_sha256
    incomplete["collector_attestation"]["redacted_artifact_sha256"] = source_sha256
    evidence_path.write_text(json.dumps({"evidence": [incomplete]}), encoding="utf-8")

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "source_artifact_reflex_transcript_not_hypothesis" in matrix["stack_smoke"]["issues"]
    assert "source_artifact_auxiliary_transcript_not_hypothesis" in matrix["stack_smoke"]["issues"]
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_rejects_stack_smoke_with_stale_source_and_attestation_hashes(tmp_path):
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    source_path = sources_dir / "stack-smoke.json"
    source_path.write_text(
        json.dumps({"redacted": True, "source": "old stack smoke"}, sort_keys=True),
        encoding="utf-8",
    )
    old_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    stack_smoke = _stack_smoke()
    stack_smoke["source_artifact"] = "sources/stack-smoke.json"
    stack_smoke["source_artifact_sha256"] = old_sha256
    stack_smoke["collector_attestation"]["redacted_artifact_sha256"] = old_sha256
    evidence_path = tmp_path / "spark-benchmark-evidence.json"
    evidence_path.write_text(json.dumps({"evidence": [stack_smoke]}, indent=2, sort_keys=True), encoding="utf-8")
    source_path.write_text(
        json.dumps({"redacted": True, "source": "updated stack smoke"}, sort_keys=True),
        encoding="utf-8",
    )

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "source_artifact_sha256_mismatch" in matrix["stack_smoke"]["issues"]
    assert "collector_attestation_redacted_sha256_mismatch" in matrix["stack_smoke"]["issues"]
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_rejects_stack_smoke_without_collector_attestation(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    incomplete = _stack_smoke()
    incomplete.pop("collector_attestation")
    evidence_path.write_text(json.dumps({"evidence": [incomplete]}), encoding="utf-8")

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "missing_collector_attestation" in matrix["stack_smoke"]["issues"]
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_rejects_stack_smoke_with_inverted_collector_attestation_window(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    stack_smoke = _stack_smoke()
    stack_smoke["collector_attestation"]["started_at"] = "2026-06-29T00:00:02Z"
    stack_smoke["collector_attestation"]["finished_at"] = "2026-06-29T00:00:01Z"
    evidence_path.write_text(json.dumps({"evidence": [stack_smoke]}), encoding="utf-8")

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "collector_attestation_invalid:timestamp_window" in matrix["stack_smoke"]["issues"]
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_rejects_stack_smoke_with_sensitive_collector_attestation_command_argv(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    stack_smoke = _stack_smoke()
    stack_smoke["collector_attestation"]["command_argv"] = [
        "voiceops-spark-collector",
        "--notify=+15551234567",
    ]
    evidence_path.write_text(json.dumps({"evidence": [stack_smoke]}), encoding="utf-8")

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "collector_attestation_secret_or_phone_like_command_argv" in matrix["stack_smoke"]["issues"]
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_refresh_source_hashes_updates_stack_smoke_attestation(tmp_path):
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    source_path = sources_dir / "stack-smoke.json"
    source_path.write_text(
        json.dumps(
            {"redacted": True, "source": "old stack smoke", "source_key": "voiceops_spark_stack_smoke"},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    old_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    stack_smoke = _stack_smoke()
    stack_smoke["source_artifact"] = "sources/stack-smoke.json"
    stack_smoke["source_artifact_sha256"] = old_sha256
    stack_smoke["collector_attestation"]["redacted_artifact_sha256"] = old_sha256
    evidence_path = tmp_path / "spark-benchmark-evidence.json"
    evidence_path.write_text(json.dumps({"evidence": [stack_smoke]}, indent=2, sort_keys=True), encoding="utf-8")
    source_path.write_text(
        json.dumps(
            {"redacted": True, "source": "updated stack smoke", "source_key": "voiceops_spark_stack_smoke"},
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = refresh_spark_source_hashes(evidence_path)
    refreshed = json.loads(evidence_path.read_text(encoding="utf-8"))["evidence"][0]
    new_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()

    assert result["ok"] is True
    assert result["updates"][0]["item"] == "voiceops_spark_stack_smoke"
    assert result["updates"][0]["collector_attestation_changed"] is True
    assert refreshed["source_artifact_sha256"] == new_sha256
    assert refreshed["collector_attestation"]["redacted_artifact_sha256"] == new_sha256


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


def test_spark_matrix_rejects_stale_stack_smoke_timestamps(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    stale = _stack_smoke()
    stale["measured_at"] = "2026-06-28T23:59:59Z"
    stale["collector_attestation"]["started_at"] = "2026-06-28T23:59:57Z"
    stale["collector_attestation"]["finished_at"] = "2026-06-28T23:59:58Z"
    evidence_path.write_text(json.dumps({"evidence": [stale]}), encoding="utf-8")

    matrix = build_matrix([evidence_path])

    assert matrix["stack_smoke"]["status"] == "fails_target"
    assert "stale_measured_at" in matrix["stack_smoke"]["issues"]
    assert "collector_attestation_stale:started_at" in matrix["stack_smoke"]["issues"]
    assert "collector_attestation_stale:finished_at" in matrix["stack_smoke"]["issues"]
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
    assert "missing_reflex_provider:s2s_or_timing" in matrix["stack_smoke"]["issues"]
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
        "collector_attestation": _collector_attestation(
            "kame-adapted",
            _source_artifact_sha256("artifacts/kame/raw.json"),
        ),
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
                    "model": "Moshi S2S",
                    "metrics": {
                        "ack_latency_ms": 240,
                        "barge_in_stop_ms": 80,
                        "steady_state_memory_gb": 16,
                    },
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
                        "interpreter": True,
                        "tts": True,
                        "sidecar": True,
                    },
                    "oracle_authority_routes": ["tools", "files", "memory", "project_context"],
                    "interface_input_sources": ["native_audio"],
                    "reflex_providers": ["moshi"],
                    "interpreter_providers": ["vllm", "gemma"],
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

    assert evaluations["interpreter-gemma4-e2b"]["status"] == "validated"
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
                    "collector_attestation": _collector_attestation(
                        "kame-e4b",
                        _source_artifact_sha256("artifacts/kame/raw.json"),
                    ),
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

    assert evaluations["interpreter-gemma4-e2b"]["status"] == "needs_evidence"
    assert evaluations["interpreter-gemma4-e4b"]["status"] == "validated"
    assert matrix["role_status"]["interpreter"] == "validated"
    assert matrix["role_status"]["reflex"] == "needs_evidence"
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
    assert "asr" not in matrix["role_status"]
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
                    "reflex_providers": ["moshi"],
                    "interpreter_providers": ["vllm", "gemma"],
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
    assert "missing_reflex_provider:s2s_or_timing" in matrix["stack_smoke"]["issues"]
    assert "missing_interpreter_provider:gemma_audio" in matrix["stack_smoke"]["issues"]
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


def test_spark_matrix_cli_writes_artifacts_for_missing_evidence_file(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "voiceops_spark_matrix.py"
    missing_evidence = tmp_path / "missing-evidence.json"
    output_dir = tmp_path / "matrix"

    result = subprocess.run(
        ["python", str(script), "--output-dir", str(output_dir), "--evidence", str(missing_evidence)],
        check=False,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["ok"] is False
    assert payload["evidence_load_issues"] == [f"evidence_file_not_found:{missing_evidence}"]
    assert Path(payload["artifacts"]["json"]).exists()
    matrix = json.loads(Path(payload["artifacts"]["json"]).read_text(encoding="utf-8"))
    assert matrix["ready_for_one_spark_demo"] is False


def test_spark_matrix_lint_evidence_is_no_write_for_valid_evidence(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "voiceops_spark_matrix.py"
    evidence_path = tmp_path / "evidence.json"
    output_dir = tmp_path / "matrix"
    evidence_path.write_text(
        json.dumps(
            {
                "evidence": [
                    _reflex_evidence(),
                    {
                        **_base_evidence("interpreter-gemma4-e2b", model="Gemma 4 E2B audio-native interpreter"),
                        "metrics": {"audio_interpretation_ms": 900, "evidence_patch_ms": 1200, "steady_state_memory_gb": 24},
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

    result = subprocess.run(
        [
            "python",
            str(script),
            "--lint-evidence",
            "--output-dir",
            str(output_dir),
            "--evidence",
            str(evidence_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "voiceops.spark_evidence_lint.v1"
    assert payload["ok"] is True
    assert payload["artifact_writes"] is False
    assert payload["ready_for_one_spark_demo"] is True
    assert payload["role_status"] == {
        "interpreter": "validated",
        "oracle": "validated",
        "reflex": "validated",
        "tts": "validated",
    }
    assert payload["stack_smoke"]["status"] == "validated"
    assert not output_dir.exists()


def test_spark_matrix_lint_evidence_reports_missing_file_without_writes(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "voiceops_spark_matrix.py"
    missing_evidence = tmp_path / "missing-evidence.json"
    output_dir = tmp_path / "matrix"

    result = subprocess.run(
        [
            "python",
            str(script),
            "--lint-evidence",
            "--output-dir",
            str(output_dir),
            "--evidence",
            str(missing_evidence),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["ok"] is False
    assert payload["artifact_writes"] is False
    assert payload["evidence_load_issues"] == [f"evidence_file_not_found:{missing_evidence}"]
    assert payload["ready_for_one_spark_demo"] is False
    assert not output_dir.exists()


def test_spark_matrix_parse_args_accepts_repeated_evidence(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    args = parse_args(["--evidence", str(first), "--evidence", str(second)])

    assert args.evidence == [first, second]


def test_spark_matrix_parse_args_accepts_refresh_source_hashes(tmp_path):
    evidence = tmp_path / "spark-benchmark-evidence.json"
    args = parse_args(["--refresh-source-hashes", str(evidence)])

    assert args.refresh_source_hashes == evidence
    assert args.evidence == []
    assert args.lint_evidence is False


def test_spark_matrix_parse_args_rejects_refresh_combined_with_evidence(tmp_path):
    evidence = tmp_path / "spark-benchmark-evidence.json"
    with pytest.raises(SystemExit):
        parse_args(["--refresh-source-hashes", str(evidence), "--evidence", str(evidence)])


def test_spark_matrix_parse_args_lint_requires_evidence():
    with pytest.raises(SystemExit):
        parse_args(["--lint-evidence"])
