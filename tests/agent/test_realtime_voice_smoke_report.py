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
    realtime_voice_alpha_manifest_fingerprint,
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
                "events": ["frontend.state", "barge_in.detected"],
                "error": None,
            }
        )
    return entries


def _discord_live_probe_report(*, inbound: bool = False):
    return {
        "kind": "discord_live_probe",
        "ok": True,
        "guild_name": "jetha dev server",
        "voice_channel_name": "General",
        "connect_perm": True,
        "speak_perm": True,
        "members_before": 2 if inbound else 0,
        "connected": True,
        "opus_loaded": True,
        "accepted_audio_source": True,
        "played": True,
        "playing_during_probe": True,
        "receiver_started": True,
        "receiver_frames": 12 if inbound else 0,
        "receiver_speech_start": 1 if inbound else 0,
        "inbound_observed": inbound,
        "members_after": 2 if inbound else 0,
        "disconnected": True,
        "require_inbound": inbound,
        "wait_seconds": 5.0,
        "failure_reason": None,
        "error": None,
    }


def _add_kame_route_evidence(report):
    route_by_kind = {
        "audio_session": iter(("local", "oracle_direct")),
        "session_turn": iter(("defer", "reject_or_clarify")),
    }
    for entry in report:
        if entry.get("kind") in route_by_kind:
            entry["route"] = next(route_by_kind[entry["kind"]])
            entry["interface_input_source"] = "native_audio"
            entry["reflex_provider"] = "vllm"
    return report


def _valid_async_oracle_smoke_entry():
    return {
        "kind": "async_oracle_smoke",
        "ok": True,
        "scenario": "async_kame_oracle_jobs_fake",
        "max_worker_overlap": 4,
        "worker_overlap_proved": True,
        "worker_overlap_within_capacity": True,
        "fifth_job_queued": True,
        "fifth_job_started_after_capacity_freed": True,
        "queued_job_update_observed": True,
        "queued_update_reached_oracle": True,
        "spoken_cancel_control_observed": True,
        "queued_cancel_smoke_ok": True,
        "completed_result_status_visible": True,
        "failed_job_reported": True,
        "verbose_result_spoken_bounded": True,
        "verbose_full_result_durable": True,
    }


def test_realtime_voice_alpha_report_accepts_required_en_ja_smokes():
    assert validate_realtime_voice_alpha_report(_valid_alpha_report()) == []


def test_realtime_voice_alpha_report_requires_async_oracle_smoke_when_requested():
    issues = validate_realtime_voice_alpha_report(
        _valid_alpha_report(),
        require_async_oracle_smoke=True,
    )

    assert any("missing async oracle smoke proof" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_accepts_async_oracle_smoke_when_requested():
    report = [*_valid_alpha_report(), _valid_async_oracle_smoke_entry()]

    assert validate_realtime_voice_alpha_report(report, require_async_oracle_smoke=True) == []


def test_realtime_voice_alpha_report_rejects_weak_async_oracle_smoke():
    weak_async_smoke = _valid_async_oracle_smoke_entry()
    weak_async_smoke["max_worker_overlap"] = 3
    weak_async_smoke["queued_update_reached_oracle"] = False
    report = [*_valid_alpha_report(), weak_async_smoke]

    issues = validate_realtime_voice_alpha_report(
        report,
        require_async_oracle_smoke=True,
    )

    assert any("four concurrent oracle jobs" in issue.format() for issue in issues)
    assert any("queued_update_reached_oracle" in issue.format() for issue in issues)


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


def test_realtime_voice_alpha_report_accepts_declared_provider_partial_latency_ceiling():
    report = _valid_alpha_report()
    manifest = report[0]
    manifest["quality_targets_ms"]["audio_to_partial_transcript_ms"] = 1000
    manifest["quality_target_ceilings_ms"] = {"audio_to_partial_transcript_ms": 1000}
    for entry in report:
        if entry.get("kind") in {"audio_fixture", "audio_session"}:
            entry["target_ms"] = 1000
            entry["transcript_partial_ms"] = 831

    assert validate_realtime_voice_alpha_report(report) == []


def test_realtime_voice_alpha_report_accepts_fast_final_when_short_utterance_has_no_partial():
    report = _valid_alpha_report()
    manifest = report[0]
    manifest["quality_targets_ms"]["audio_to_partial_transcript_ms"] = 1000
    manifest["quality_target_ceilings_ms"] = {"audio_to_partial_transcript_ms": 1000}
    for entry in report:
        if entry.get("kind") in {"audio_fixture", "audio_session"} and entry.get("fixture") == "./fixtures/realtime-voice/ja/hello.webm":
            entry["target_ms"] = 1000
            entry["transcript_partial_ms"] = None
            entry["transcript_final_ms"] = 679
            entry["events"] = [event for event in entry["events"] if event != "transcript.partial"]
            if entry.get("kind") == "audio_session":
                entry["ok"] = False
                entry["error"] = "unknown realtime voice error"

    assert validate_realtime_voice_alpha_report(report) == []


def test_realtime_voice_alpha_report_requires_live_sidecar_manifest_capabilities():
    manifest = _valid_manifest()
    manifest["sidecar"]["health"]["capabilities"] = {
        "utterance_stt": True,
        "tts": True,
        "output_languages": ["en", "ja"],
    }
    report = [manifest, *_valid_alpha_report()[1:]]

    issues = validate_realtime_voice_alpha_report(report)

    assert any("missing native_s2s, streaming_stt+tts, or kame_reflex+tts" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_accepts_kame_reflex_manifest_capabilities():
    manifest = _valid_manifest()
    manifest["engine"] = "kame_interface_oracle"
    manifest["frontend_provider"] = "gemma4"
    manifest["frontend_model"] = "gemma-4-E2B-it"
    manifest["interface_audio_input"] = "native_audio"
    manifest["asr_mode"] = "on_escalation"
    manifest["preferred_local_oracle_model"] = "gemma-4-26B-A4B-it"
    manifest["conversation_quality"] = {
        "live_like": True,
        "mode": "kame_reflex",
        "reason": "audio_reflex_tts",
        "sidecar_verified": True,
    }
    manifest["sidecar"]["health"]["frontend"] = {
        "provider": "vllm",
        "model": "gemma-4-E2B-it",
    }
    manifest["sidecar"]["health"]["capabilities"] = {
        "utterance_stt": True,
        "streaming_stt": False,
        "tts": True,
        "native_s2s": False,
        "vllm_audio_frontend": True,
        "output_languages": ["en", "ja"],
    }
    report = _add_kame_route_evidence([manifest, *_valid_alpha_report()[1:]])

    assert validate_realtime_voice_alpha_report(report) == []


def test_realtime_voice_alpha_report_accepts_kame_native_audio_without_partial_transcripts():
    manifest = _valid_manifest()
    manifest["engine"] = "kame_interface_oracle"
    manifest["frontend_provider"] = "gemma4"
    manifest["frontend_model"] = "gemma-4-E2B-it"
    manifest["interface_audio_input"] = "native_audio"
    manifest["asr_mode"] = "on_escalation"
    manifest["conversation_quality"] = {
        "live_like": True,
        "mode": "kame_reflex",
        "reason": "audio_reflex_tts",
        "sidecar_verified": True,
    }
    manifest["sidecar"]["health"]["frontend"] = {
        "provider": "vllm",
        "model": "gemma-4-E2B-it",
    }
    manifest["sidecar"]["health"]["capabilities"] = {
        "utterance_stt": True,
        "streaming_stt": False,
        "tts": True,
        "native_s2s": False,
        "vllm_audio_frontend": True,
        "output_languages": ["en", "ja"],
    }
    report = _add_kame_route_evidence([manifest, *_valid_alpha_report()[1:]])
    for entry in report:
        if entry.get("kind") == "audio_session":
            entry["transcript_partial_ms"] = None
            entry["transcript_final_ms"] = 450
            entry["events"] = [event for event in entry["events"] if event != "transcript.partial"]

    assert validate_realtime_voice_alpha_report(report) == []


def test_realtime_voice_alpha_report_requires_kame_oracle_avoidance_evidence():
    manifest = _valid_manifest()
    manifest["engine"] = "kame_interface_oracle"
    manifest["frontend_provider"] = "gemma4"
    manifest["frontend_model"] = "gemma-4-E2B-it"
    manifest["interface_audio_input"] = "native_audio"
    manifest["asr_mode"] = "on_escalation"
    manifest["conversation_quality"] = {
        "live_like": True,
        "mode": "kame_reflex",
        "reason": "audio_reflex_tts",
        "sidecar_verified": True,
    }
    manifest["sidecar"]["health"]["frontend"] = {
        "provider": "vllm",
        "model": "gemma-4-E2B-it",
    }
    manifest["sidecar"]["health"]["capabilities"] = {
        "utterance_stt": True,
        "streaming_stt": False,
        "tts": True,
        "native_s2s": False,
        "vllm_audio_frontend": True,
        "output_languages": ["en", "ja"],
    }
    report = [manifest, *_valid_alpha_report()[1:]]
    for entry in report:
        if entry.get("kind") in {"audio_session", "session_turn"}:
            entry["route"] = "oracle_direct"

    issues = validate_realtime_voice_alpha_report(report)

    assert any(
        "missing oracle-avoiding local or clarify route evidence" in issue.format()
        for issue in issues
    )


def test_realtime_voice_alpha_report_requires_kame_native_audio_reflex_provenance():
    manifest = _valid_manifest()
    manifest["engine"] = "kame_interface_oracle"
    manifest["frontend_provider"] = "gemma4"
    manifest["frontend_model"] = "gemma-4-E2B-it"
    manifest["interface_audio_input"] = "native_audio"
    manifest["asr_mode"] = "on_escalation"
    manifest["conversation_quality"] = {
        "live_like": True,
        "mode": "kame_reflex",
        "reason": "audio_reflex_tts",
        "sidecar_verified": True,
    }
    manifest["sidecar"]["health"]["frontend"] = {
        "provider": "vllm",
        "model": "gemma-4-E2B-it",
    }
    manifest["sidecar"]["health"]["capabilities"] = {
        "utterance_stt": True,
        "streaming_stt": False,
        "tts": True,
        "native_s2s": False,
        "vllm_audio_frontend": True,
        "output_languages": ["en", "ja"],
    }
    report = _add_kame_route_evidence([manifest, *_valid_alpha_report()[1:]])
    for entry in report:
        if entry.get("kind") in {"audio_session", "session_turn"}:
            entry["interface_input_source"] = "streaming_stt"
            entry["reflex_provider"] = "streaming_stt"
            entry["interface_audio_input_fallback"] = True

    issues = validate_realtime_voice_alpha_report(report)

    assert any("missing native-audio reflex route evidence" in issue.format() for issue in issues)
    assert any("missing vLLM reflex provider route evidence" in issue.format() for issue in issues)
    assert any("KAME route evidence used only fallback reflex input" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_rejects_malformed_kame_reflex_route_evidence():
    manifest = _valid_manifest()
    manifest["engine"] = "kame_interface_oracle"
    manifest["frontend_provider"] = "gemma4"
    manifest["frontend_model"] = "gemma-4-E2B-it"
    manifest["interface_audio_input"] = "native_audio"
    manifest["asr_mode"] = "on_escalation"
    manifest["conversation_quality"] = {
        "live_like": True,
        "mode": "kame_reflex",
        "reason": "audio_reflex_tts",
        "sidecar_verified": True,
    }
    manifest["sidecar"]["health"]["frontend"] = {
        "provider": "vllm",
        "model": "gemma-4-E2B-it",
    }
    manifest["sidecar"]["health"]["capabilities"] = {
        "utterance_stt": True,
        "streaming_stt": False,
        "tts": True,
        "native_s2s": False,
        "vllm_audio_frontend": True,
        "output_languages": ["en", "ja"],
    }
    report = _add_kame_route_evidence([manifest, *_valid_alpha_report()[1:]])
    for entry in report:
        if entry.get("kind") == "audio_session":
            entry["reflex_validation_error"] = "invalid_json"
            break

    issues = validate_realtime_voice_alpha_report(report)

    assert any(
        "KAME route evidence includes malformed reflex output" in issue.format()
        and "invalid_json=1" in issue.format()
        for issue in issues
    )


def test_realtime_voice_alpha_report_rejects_kame_voice_capability_denial_output():
    manifest = _valid_manifest()
    manifest["engine"] = "kame_interface_oracle"
    manifest["frontend_provider"] = "gemma4"
    manifest["frontend_model"] = "gemma-4-E2B-it"
    manifest["interface_audio_input"] = "native_audio"
    manifest["asr_mode"] = "on_escalation"
    manifest["conversation_quality"] = {
        "live_like": True,
        "mode": "kame_reflex",
        "reason": "audio_reflex_tts",
        "sidecar_verified": True,
    }
    manifest["sidecar"]["health"]["frontend"] = {
        "provider": "vllm",
        "model": "gemma-4-E2B-it",
    }
    manifest["sidecar"]["health"]["capabilities"] = {
        "utterance_stt": True,
        "streaming_stt": False,
        "tts": True,
        "native_s2s": False,
        "vllm_audio_frontend": True,
        "output_languages": ["en", "ja"],
    }
    report = _add_kame_route_evidence([manifest, *_valid_alpha_report()[1:]])
    for entry in report:
        if entry.get("kind") == "session_turn":
            entry["final_text"] = "I cannot hear you or speak in Discord voice."
            break

    issues = validate_realtime_voice_alpha_report(report)

    assert any(
        "assistant output denied live voice capability" in issue.format()
        and "session_turn.final_text" in issue.format()
        for issue in issues
    )


def test_realtime_voice_alpha_report_allows_kame_user_hear_me_text():
    manifest = _valid_manifest()
    manifest["engine"] = "kame_interface_oracle"
    manifest["frontend_provider"] = "gemma4"
    manifest["frontend_model"] = "gemma-4-E2B-it"
    manifest["interface_audio_input"] = "native_audio"
    manifest["asr_mode"] = "on_escalation"
    manifest["conversation_quality"] = {
        "live_like": True,
        "mode": "kame_reflex",
        "reason": "audio_reflex_tts",
        "sidecar_verified": True,
    }
    manifest["sidecar"]["health"]["frontend"] = {
        "provider": "vllm",
        "model": "gemma-4-E2B-it",
    }
    manifest["sidecar"]["health"]["capabilities"] = {
        "utterance_stt": True,
        "streaming_stt": False,
        "tts": True,
        "native_s2s": False,
        "vllm_audio_frontend": True,
        "output_languages": ["en", "ja"],
    }
    report = _add_kame_route_evidence([manifest, *_valid_alpha_report()[1:]])
    report.append({"kind": "debug_note", "ok": True, "text": "I cannot hear you clearly?"})

    assert validate_realtime_voice_alpha_report(report) == []


def test_realtime_voice_alpha_manifest_fingerprint_includes_kame_interface_base_url():
    manifest = _valid_manifest()
    manifest["engine"] = "kame_interface_oracle"
    manifest["frontend_provider"] = "gemma4"
    manifest["frontend_model"] = "gemma-4-E2B-it"
    manifest["interface_base_url"] = "http://spark-a.local:8000/v1"
    manifest["interface_audio_input"] = "native_audio"
    manifest["interface_temperature"] = 0.2
    manifest["interface_max_output_tokens"] = 160
    manifest["interface_timeout_seconds"] = 0.8
    manifest["interface_max_audio_seconds"] = 30.0
    manifest["asr_mode"] = "on_escalation"
    manifest["conversation_quality"] = {
        "live_like": True,
        "mode": "kame_reflex",
        "reason": "audio_reflex_tts",
        "sidecar_verified": True,
    }
    manifest["sidecar"]["health"]["frontend"] = {
        "provider": "vllm",
        "model": "gemma-4-E2B-it",
    }
    manifest["sidecar"]["health"]["capabilities"] = {
        "utterance_stt": True,
        "streaming_stt": False,
        "tts": True,
        "native_s2s": False,
        "vllm_audio_frontend": True,
        "output_languages": ["en", "ja"],
    }
    other = dict(manifest)
    other["interface_base_url"] = "http://spark-b.local:8000/v1"

    assert realtime_voice_alpha_manifest_fingerprint(manifest) != realtime_voice_alpha_manifest_fingerprint(other)


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


def test_realtime_voice_alpha_report_accepts_native_assistant_audio_chunks():
    report = _valid_alpha_report()
    for entry in report:
        events = entry.get("events")
        if isinstance(events, list):
            entry["events"] = [
                "assistant.audio.chunk" if event == "audio.output.chunk" else event
                for event in events
            ]

    assert validate_realtime_voice_alpha_report(report) == []


def test_realtime_voice_alpha_report_accepts_japanese_hermes_phonetic_variants():
    report = _valid_alpha_report()
    replacements = {
        "./fixtures/realtime-voice/ja/hello.webm": "こんにちは。ハルメスです。",
        "./fixtures/realtime-voice/ja/tool-question.webm": "ハーメスはこのワークスペースで何を確認できますか？",
    }
    for entry in report:
        if entry.get("kind") in {"audio_fixture", "audio_session"} and entry.get("fixture") in replacements:
            entry["final_text"] = replacements[entry["fixture"]]

    assert validate_realtime_voice_alpha_report(report) == []


def test_realtime_voice_alpha_report_rejects_barge_in_target_misses():
    report = _valid_alpha_report()
    report[-1]["barge_in_ack_ms"] = 250

    issues = validate_realtime_voice_alpha_report(report)

    assert any("barge_in_ack_ms 250 exceeds target 150" in issue.format() for issue in issues)


def test_realtime_voice_alpha_report_rejects_audio_after_barge_in():
    report = _valid_alpha_report()
    report[-1]["audio_after_barge_in_bytes"] = 128
    report[-1]["events"] = ["frontend.state", "barge_in.detected", "audio.output.chunk"]

    issues = validate_realtime_voice_alpha_report(report)

    assert any(
        "output audio event arrived after barge_in.detected (128 byte(s))" in issue.format()
        for issue in issues
    )


def test_realtime_voice_alpha_report_accepts_legacy_barge_in_event_name():
    report = _valid_alpha_report()
    report[-1]["events"] = ["frontend.state", "barge_in"]

    assert validate_realtime_voice_alpha_report(report) == []


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
    report = _valid_alpha_report()
    route_by_kind = {
        "audio_session": iter(("local", "oracle_direct")),
        "session_turn": iter(("defer", "reject_or_clarify")),
    }
    metric_added = False
    for entry in report:
        if entry.get("kind") in route_by_kind:
            entry["route"] = next(route_by_kind[entry["kind"]])
            entry["interface_input_source"] = "native_audio"
            entry["reflex_provider"] = "vllm"
        if entry.get("kind") == "session_turn" and not metric_added:
            entry["metrics"] = {
                "eou_to_final_transcript_ms": 18,
                "kame_final_transcript_to_interface_decision_ms": 27,
                "kame_interface_decision_to_oracle_accepted_ms": 35,
                "kame_oracle_accepted_to_first_token_ms": 120,
            }
            metric_added = True
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    assert realtime_voice_report_main([str(path), "--alpha"]) == 0
    output = capsys.readouterr().out
    assert "Realtime voice smoke report OK" in output
    assert "audio_to_partial_transcript: p50=90ms p90=90ms p95=90ms max=90ms n=6" in output
    assert "final_transcript_to_first_text: p50=90ms p90=90ms p95=90ms max=90ms n=4" in output
    assert "speech_boundary_to_final_transcript: p50=18ms p90=18ms p95=18ms max=18ms n=1" in output
    assert "final_transcript_to_interface_decision: p50=27ms p90=27ms p95=27ms max=27ms n=1" in output
    assert "interface_decision_to_oracle_accepted: p50=35ms p90=35ms p95=35ms max=35ms n=1" in output
    assert "oracle_accepted_to_first_token: p50=120ms p90=120ms p95=120ms max=120ms n=1" in output
    assert "barge_in_ack: p50=45ms p90=45ms p95=45ms max=45ms n=1" in output
    assert "kame_routes: total=4 oracle_avoided=2 oracle_required=2 avoidance=50.0%" in output
    assert "local=1 defer=1 oracle_direct=1 reject_or_clarify=1" in output
    assert "kame_reflex: total=4 native_audio=4 vllm=4 fallback=0 sources native_audio=4 providers vllm=4" in output
    assert "stack unknown_engine|unknown_frontend|unknown_model|unknown_oracle|unknown_tts|unknown_tts_model" in output


def test_realtime_voice_report_cli_requires_async_oracle_smoke_when_requested(tmp_path, capsys):
    path = tmp_path / "voice-smoke.json"
    path.write_text(json.dumps(_valid_alpha_report(), ensure_ascii=False), encoding="utf-8")

    assert realtime_voice_report_main([str(path), "--alpha", "--require-async-oracle-smoke"]) == 1
    assert "missing async oracle smoke proof" in capsys.readouterr().err

    report = [*_valid_alpha_report(), _valid_async_oracle_smoke_entry()]
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    assert realtime_voice_report_main([str(path), "--alpha", "--require-async-oracle-smoke"]) == 0
    assert "Realtime voice smoke report OK" in capsys.readouterr().out


def test_realtime_voice_report_cli_enforces_minimum_alpha_runs(tmp_path, capsys):
    path = tmp_path / "voice-smoke.json"
    path.write_text(json.dumps(_valid_alpha_report(), ensure_ascii=False), encoding="utf-8")

    assert realtime_voice_report_main([str(path), "--alpha", "--min-runs", "3"]) == 1
    assert "requires at least 3 run(s), found 1" in capsys.readouterr().err


def test_realtime_voice_report_cli_validates_discord_live_probe(tmp_path, capsys):
    path = tmp_path / "discord-live.json"
    path.write_text(
        json.dumps(
            [
                {"kind": "manifest", "ok": True},
                _discord_live_probe_report(inbound=True),
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert realtime_voice_report_main([str(path), "--discord-live-probe", "--require-inbound"]) == 0
    assert "Realtime voice smoke report OK: 1 result(s) across 1 run(s)" in capsys.readouterr().out


def test_realtime_voice_report_cli_rejects_discord_live_probe_without_inbound(
    tmp_path,
    capsys,
):
    path = tmp_path / "discord-live.json"
    probe = _discord_live_probe_report(inbound=False)
    probe["ok"] = False
    probe["require_inbound"] = True
    probe["failure_reason"] = "inbound_required_but_no_other_members"
    probe["error"] = "live Discord voice probe did not satisfy invariants: inbound_required_but_no_other_members"
    path.write_text(json.dumps([{"kind": "manifest", "ok": True}, probe], ensure_ascii=False), encoding="utf-8")

    assert realtime_voice_report_main([str(path), "--discord-live-probe", "--require-inbound"]) == 1
    error = capsys.readouterr().err
    assert "no passing probe" in error
    assert "inbound speech not observed" in error
    assert "inbound_required_but_no_other_members" in error


def test_realtime_voice_report_cli_rejects_missing_discord_live_probe(tmp_path, capsys):
    path = tmp_path / "voice-smoke.json"
    path.write_text(json.dumps(_valid_alpha_report(), ensure_ascii=False), encoding="utf-8")

    assert realtime_voice_report_main([str(path), "--discord-live-probe"]) == 1
    assert "missing Discord live probe result" in capsys.readouterr().err


def test_realtime_voice_report_cli_apply_requires_alpha(tmp_path, capsys):
    path = tmp_path / "voice-smoke.json"
    path.write_text(json.dumps(_valid_alpha_report(), ensure_ascii=False), encoding="utf-8")

    assert realtime_voice_report_main([str(path), "--apply-production-evidence"]) == 1
    assert "--apply-production-evidence requires --alpha" in capsys.readouterr().err


def test_realtime_voice_report_cli_applies_validated_production_evidence(
    monkeypatch,
    tmp_path,
    capsys,
):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    saved = {}
    for index in range(3):
        path = reports_dir / f"voice-smoke-{index}.json"
        path.write_text(json.dumps(_valid_alpha_report(), ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {"voice": {"realtime": {"enabled": True}}})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.setdefault("config", cfg))
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: tmp_path / "config.yaml")

    assert (
        realtime_voice_report_main(
            [
                *[str(path) for path in sorted(reports_dir.glob("*.json"))],
                "--alpha",
                "--min-runs",
                "3",
                "--apply-production-evidence",
            ]
        )
        == 0
    )

    assert saved["config"]["voice"]["realtime"]["production_evidence_report"] == str(reports_dir)
    assert "Updated realtime voice production_evidence_report" in capsys.readouterr().out


def test_realtime_voice_report_cli_apply_rejects_loopback_evidence(tmp_path, capsys):
    reports = []
    for index in range(3):
        report = _valid_alpha_report()
        for entry in report:
            entry["evidence_provider"] = "loopback"
            entry["loopback_validation"] = True
        path = tmp_path / f"voice-smoke-{index}.json"
        path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        reports.append(str(path))

    assert (
        realtime_voice_report_main(
            [
                *reports,
                "--alpha",
                "--min-runs",
                "3",
                "--apply-production-evidence",
            ]
        )
        == 1
    )
    assert "loopback validation cannot satisfy production evidence" in capsys.readouterr().err


def test_realtime_voice_report_cli_apply_rejects_mixed_report_directories(
    monkeypatch,
    tmp_path,
    capsys,
):
    saved = {}
    reports = []
    for index in range(3):
        report_dir = tmp_path / f"reports-{index}"
        report_dir.mkdir()
        path = report_dir / "voice-smoke.json"
        path.write_text(json.dumps(_valid_alpha_report(), ensure_ascii=False), encoding="utf-8")
        reports.append(str(path))

    monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {"voice": {"realtime": {"enabled": True}}})
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.setdefault("config", cfg))

    assert (
        realtime_voice_report_main(
            [
                *reports,
                "--alpha",
                "--min-runs",
                "3",
                "--apply-production-evidence",
            ]
        )
        == 1
    )

    assert saved == {}
    assert "multiple reports requires all reports to share one directory" in capsys.readouterr().err


def test_realtime_voice_alpha_report_runs_accept_multiple_reports(tmp_path):
    runs = []
    for index in range(3):
        path = tmp_path / f"voice-smoke-{index}.json"
        path.write_text(json.dumps(_valid_alpha_report(), ensure_ascii=False), encoding="utf-8")
        runs.append((str(path), load_realtime_voice_smoke_report(path)))

    assert validate_realtime_voice_alpha_report_runs(runs, min_runs=3) == []


def test_realtime_voice_alpha_report_runs_can_reject_loopback_for_production(tmp_path):
    runs = []
    for index in range(3):
        report = _valid_alpha_report()
        for entry in report:
            entry["evidence_provider"] = "loopback"
            entry["loopback_validation"] = True
        path = tmp_path / f"voice-smoke-{index}.json"
        path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        runs.append((str(path), load_realtime_voice_smoke_report(path)))

    assert validate_realtime_voice_alpha_report_runs(runs, min_runs=3) == []

    issues = validate_realtime_voice_alpha_report_runs(
        runs,
        min_runs=3,
        allow_loopback_validation=False,
    )

    assert any("loopback validation cannot satisfy production evidence" in issue.format() for issue in issues)


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
        route_by_kind = {
            "audio_session": iter(("local", "oracle_direct")),
            "session_turn": iter(("defer", "reject_or_clarify")),
        }
        for entry in report:
            if entry.get("kind") in route_by_kind:
                entry["route"] = next(route_by_kind[entry["kind"]])
                if entry["route"] == "reject_or_clarify":
                    entry["interface_input_source"] = "streaming_stt"
                    entry["reflex_provider"] = "streaming_stt"
                    entry["interface_audio_input_fallback"] = True
                else:
                    entry["interface_input_source"] = "native_audio"
                    entry["reflex_provider"] = "vllm"
            if entry.get("kind") == "audio_fixture":
                entry["transcript_partial_ms"] = partial_ms
            if entry.get("kind") == "audio_session":
                entry["metrics"] = {
                    "eou_to_final_transcript_ms": 18 + index,
                    "kame_speech_end_to_interface_decision_ms": 20 + index,
                    "oracle_verbatim_asr_ms": 35 + index,
                }
            if entry.get("kind") == "session_turn":
                entry["metrics"] = {
                    "kame_final_transcript_to_interface_decision_ms": 25 + index,
                    "kame_speech_end_to_local_first_audio_ms": 90 + index,
                    "kame_speech_end_to_first_audio_ms": 160 + index,
                    "kame_interface_decision_to_local_first_audio_ms": 70 + index,
                    "kame_interface_decision_to_first_audio_ms": 140 + index,
                    "kame_interface_decision_to_oracle_accepted_ms": 30 + index,
                    "kame_oracle_accepted_to_first_token_ms": 120 + index,
                    "kame_oracle_first_token_to_first_spoken_text_ms": 40 + index,
                    "kame_oracle_first_token_to_first_tts_audio_ms": 115 + index,
                    "kame_first_tts_audio_to_playback_start_ms": 12 + index,
                    "kame_speech_end_to_playback_start_ms": 172 + index,
                    "kame_oracle_total_stream_ms": 300 + index,
                }
            if entry.get("kind") == "barge_in":
                entry["metrics"] = {
                    "barge_in_confirmed_to_playback_stopped_ms": 18 + index,
                }
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
        "p90": 120,
        "p95": 120,
        "max": 120,
    }
    assert summary["latency_ms"]["final_transcript_to_first_text"] == {
        "count": 12,
        "p50": 90,
        "p90": 90,
        "p95": 90,
        "max": 90,
    }
    assert summary["latency_ms"]["speech_end_to_interface_decision"] == {
        "count": 6,
        "p50": 21,
        "p90": 22,
        "p95": 22,
        "max": 22,
    }
    assert summary["latency_ms"]["speech_boundary_to_final_transcript"] == {
        "count": 6,
        "p50": 19,
        "p90": 20,
        "p95": 20,
        "max": 20,
    }
    assert summary["latency_ms"]["final_transcript_to_interface_decision"] == {
        "count": 6,
        "p50": 26,
        "p90": 27,
        "p95": 27,
        "max": 27,
    }
    assert summary["latency_ms"]["speech_end_to_local_first_audio"]["p90"] == 92
    assert summary["latency_ms"]["speech_end_to_first_audio"]["p90"] == 162
    assert summary["latency_ms"]["interface_decision_to_local_first_audio"]["p90"] == 72
    assert summary["latency_ms"]["interface_decision_to_first_audio"]["p90"] == 142
    assert summary["latency_ms"]["interface_decision_to_oracle_accepted"]["p90"] == 32
    assert summary["latency_ms"]["oracle_accepted_to_first_token"]["p90"] == 122
    assert summary["latency_ms"]["oracle_first_token_to_first_spoken_text"]["p90"] == 42
    assert summary["latency_ms"]["oracle_first_token_to_first_tts_audio"]["p90"] == 117
    assert summary["latency_ms"]["first_tts_audio_to_playback_start"]["p90"] == 14
    assert summary["latency_ms"]["speech_end_to_playback_start"]["p90"] == 174
    assert summary["latency_ms"]["oracle_total_stream"]["p90"] == 302
    assert summary["latency_ms"]["oracle_verbatim_asr"]["p90"] == 37
    assert summary["latency_ms"]["barge_in_confirmed_to_playback_stopped"] == {
        "count": 3,
        "p50": 19,
        "p90": 20,
        "p95": 20,
        "max": 20,
    }
    assert summary["kame_routes"] == {
        "total": 12,
        "counts": {
            "local": 3,
            "defer": 3,
            "oracle_direct": 3,
            "reject_or_clarify": 3,
        },
        "oracle_avoided": 6,
        "oracle_required": 6,
        "oracle_avoidance_rate": 0.5,
    }
    assert summary["kame_reflex_provenance"] == {
        "total": 12,
        "input_sources": {"native_audio": 9, "streaming_stt": 3},
        "reflex_providers": {"streaming_stt": 3, "vllm": 9},
        "native_audio": 9,
        "vllm": 9,
        "fallback": 3,
        "fallback_only": False,
    }
    stack_summary = summary["latency_by_stack"][
        "unknown_engine|unknown_frontend|unknown_model|unknown_oracle|unknown_tts|unknown_tts_model"
    ]
    assert stack_summary["runs"] == 3
    assert len(stack_summary["report_labels"]) == 3
    assert stack_summary["stack"] == {
        "engine": "",
        "frontend_provider": "",
        "frontend_model": "",
        "interface_audio_input": "",
        "asr_mode": "",
        "asr_provider": "",
        "asr_model": "",
        "preferred_local_oracle_model": "",
        "tts_provider": "",
        "tts_model": "",
        "tts_voice": "",
    }
    assert stack_summary["latency_ms"]["audio_to_partial_transcript"]["p90"] == 120
    assert stack_summary["latency_ms"]["oracle_accepted_to_first_token"]["p90"] == 122
    assert stack_summary["latency_ms"]["barge_in_confirmed_to_playback_stopped"]["p90"] == 20
    assert stack_summary["kame_routes"]["oracle_avoided"] == 6
    assert stack_summary["kame_routes"]["oracle_required"] == 6
    assert stack_summary["kame_reflex_provenance"]["native_audio"] == 9
    assert stack_summary["kame_reflex_provenance"]["fallback"] == 3


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
