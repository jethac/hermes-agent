"""Tests for hermes_cli.doctor."""

import json
import os
import sys
import types
import io
import contextlib
from argparse import Namespace
from types import SimpleNamespace

import pytest

import hermes_cli.doctor as doctor
import hermes_cli.gateway as gateway_cli
from hermes_cli import doctor as doctor_mod
from hermes_cli.doctor import _has_provider_env_config
from agent.realtime_voice_smoke import (
    RealtimeVoiceSidecarSmokeResult,
    realtime_voice_smoke_result_payload,
)
from agent.realtime_voice_smoke_report import (
    ALPHA_REQUIRED_AUDIO_FIXTURES,
    ALPHA_REQUIRED_AUDIO_SESSION_FIXTURES,
    ALPHA_REQUIRED_BARGE_IN_TEXTS,
    ALPHA_REQUIRED_TTS_TEXTS,
)


class TestDoctorPlatformHints:
    def test_termux_package_hint(self, monkeypatch):
        monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
        monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
        assert doctor._is_termux() is True
        assert doctor._python_install_cmd() == "python -m pip install"
        assert doctor._system_package_install_cmd("ripgrep") == "pkg install ripgrep"

    def test_non_termux_package_hint_defaults_to_apt(self, monkeypatch):
        monkeypatch.delenv("TERMUX_VERSION", raising=False)
        monkeypatch.setenv("PREFIX", "/usr")
        monkeypatch.setattr(sys, "platform", "linux")
        assert doctor._is_termux() is False
        assert doctor._python_install_cmd() == "uv pip install"
        assert doctor._system_package_install_cmd("ripgrep") == "sudo apt install ripgrep"


class TestProviderEnvDetection:
    def test_detects_openai_api_key(self):
        content = "OPENAI_BASE_URL=http://localhost:1234/v1\nOPENAI_API_KEY=***"
        assert _has_provider_env_config(content)

    def test_detects_custom_endpoint_without_openrouter_key(self):
        content = "OPENAI_BASE_URL=http://localhost:8080/v1\n"
        assert _has_provider_env_config(content)

    def test_detects_kimi_cn_api_key(self):
        content = "KIMI_CN_API_KEY=sk-test\n"
        assert _has_provider_env_config(content)

    def test_returns_false_when_no_provider_settings(self):
        content = "TERMINAL_ENV=local\n"
        assert not _has_provider_env_config(content)


class TestDoctorToolAvailabilitySummary:
    def test_missing_api_key_summary_ignores_disabled_toolsets(self, monkeypatch):
        unavailable = [
            {"name": "rl", "missing_vars": ["TINKER_API_KEY"]},
            {"name": "web", "missing_vars": ["EXA_API_KEY"]},
        ]
        monkeypatch.setattr(doctor, "_enabled_cli_toolsets_for_doctor", lambda: {"web"})

        filtered = doctor._missing_api_key_toolsets_for_summary(unavailable)

        assert [item["name"] for item in filtered] == ["web"]

    def test_missing_api_key_summary_falls_back_when_config_unavailable(self, monkeypatch):
        unavailable = [
            {"name": "rl", "missing_vars": ["TINKER_API_KEY"]},
            {"name": "web", "missing_vars": ["EXA_API_KEY"]},
        ]
        monkeypatch.setattr(doctor, "_enabled_cli_toolsets_for_doctor", lambda: None)

        filtered = doctor._missing_api_key_toolsets_for_summary(unavailable)

        assert [item["name"] for item in filtered] == ["rl", "web"]


class TestRealtimeVoiceReadiness:
    def test_realtime_voice_cli_values_accept_single_and_repeated_args(self):
        assert doctor._realtime_voice_cli_values(None) == []
        assert doctor._realtime_voice_cli_values("") == []
        assert doctor._realtime_voice_cli_values("hello.webm") == ["hello.webm"]
        assert doctor._realtime_voice_cli_values(["en.webm", "ja.webm"]) == ["en.webm", "ja.webm"]
        assert doctor._realtime_voice_cli_values(["en.webm", ["ja.webm"]]) == ["en.webm", "ja.webm"]

    def test_realtime_voice_unique_values_preserves_order_and_dedupes(self):
        assert doctor._realtime_voice_unique_values(
            ["a", "b"],
            ["b", "c", ""],
            ["a", "d"],
        ) == ["a", "b", "c", "d"]

    def test_realtime_voice_smoke_config_includes_kame_role_metadata(self, monkeypatch):
        import hermes_cli.web_server as web_server

        config = {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "engine": "text_oracle_tts",
                    "sidecar_base_url": "http://voice.local:8765",
                    "frontend_provider": "gemma4",
                    "frontend_model": "google/gemma-4-E4B-it-qat-w4a16-ct",
                    "interface_temperature": 0.35,
                    "interface_max_output_tokens": 96,
                    "interface_timeout_seconds": 0.7,
                    "interface_max_audio_seconds": 22,
                }
            }
        }
        monkeypatch.setattr(
            doctor,
            "load_config",
            lambda: config,
            raising=False,
        )
        monkeypatch.setattr(web_server, "load_config", lambda: config, raising=False)
        monkeypatch.setattr(doctor, "_realtime_voice_sidecar_token", lambda realtime: "", raising=False)

        smoke_config = doctor._realtime_voice_smoke_config()

        assert smoke_config.frontend_provider == "gemma4"
        assert smoke_config.frontend_model == "google/gemma-4-E4B-it-qat-w4a16-ct"
        assert smoke_config.interface_temperature == 0.35
        assert smoke_config.interface_max_output_tokens == 96
        assert smoke_config.interface_timeout_seconds == 0.7
        assert smoke_config.interface_max_audio_seconds == 22
        assert smoke_config.metadata["voice_architecture"] == "kame_frontend_oracle"
        assert smoke_config.metadata["frontend_role"] == "low_latency_voice_interface"
        assert smoke_config.metadata["oracle_role"] == "hermes_backend_oracle"
        assert smoke_config.metadata["frontend_provider"] == "gemma4"
        assert smoke_config.metadata["frontend_model"] == "google/gemma-4-E4B-it-qat-w4a16-ct"
        assert smoke_config.metadata["interface_temperature"] == 0.35
        assert smoke_config.metadata["interface_max_output_tokens"] == 96
        assert smoke_config.metadata["interface_timeout_seconds"] == 0.7
        assert smoke_config.metadata["interface_max_audio_seconds"] == 22.0
        assert "oracle_model" not in smoke_config.to_wire()
        assert "oracle_model" not in smoke_config.metadata

    def test_realtime_voice_smoke_result_payload_is_json_safe(self):
        payload = realtime_voice_smoke_result_payload(
            RealtimeVoiceSidecarSmokeResult(
                ok=True,
                ready_ms=12,
                transcript_partial_ms=90,
                transcript_final_ms=180,
                first_text_ms=95,
                first_audio_ms=250,
                barge_in_ack_ms=45,
                final_text="こんにちは",
                audio_bytes=1234,
                output_audio_bytes=4321,
                audio_after_barge_in_bytes=0,
                events=("frontend.state", "transcript.final"),
                first_audio_metrics={"tts_synthesis_ms": 123},
            ),
            kind="audio_fixture",
        )

        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
        assert payload["kind"] == "audio_fixture"
        assert payload["events"] == ["frontend.state", "transcript.final"]
        assert payload["first_text_ms"] == 95
        assert payload["barge_in_ack_ms"] == 45
        assert payload["audio_after_barge_in_bytes"] == 0
        assert payload["first_audio_metrics"] == {"tts_synthesis_ms": 123}
        assert payload["transport"] is None
        assert payload["error"] is None

    def test_realtime_voice_smoke_result_payload_includes_diagnostics(self):
        payload = realtime_voice_smoke_result_payload(
            RealtimeVoiceSidecarSmokeResult(
                ok=True,
                events=("frontend.state",),
                first_audio_metrics={"tts_synthesis_ms": 123},
                transport="discord_voice",
            ),
            kind="protocol",
        )

        assert payload["transport"] == "discord_voice"
        assert payload["first_audio_metrics"] == {"tts_synthesis_ms": 123}

    def test_realtime_voice_report_payload_preserves_transport(self):
        result = RealtimeVoiceSidecarSmokeResult(ok=True, transport="discord_voice")

        payload = doctor._realtime_voice_smoke_report_payload(result, kind="protocol")

        assert payload["transport"] == "discord_voice"

    def test_realtime_voice_result_with_transport_uses_config_metadata(self):
        result = RealtimeVoiceSidecarSmokeResult(ok=True)
        config = SimpleNamespace(metadata={"transport": "discord_voice"})

        tagged = doctor._realtime_voice_result_with_transport(result, config)

        assert tagged.transport == "discord_voice"
        assert result.transport == ""

    def test_discord_realtime_voice_smoke_report_payload_is_json_safe(self):
        payload = doctor._discord_realtime_voice_smoke_report_payload(
            SimpleNamespace(
                ok=True,
                mode="discord_loopback",
                transport="discord_voice",
                input_pcm48_bytes=3840,
                sidecar_pcm16_bytes=640,
                sidecar_pcm16_first_sample=450,
                sidecar_pcm16_checksum=195,
                mixer_frames=1,
                mixer_frame_bytes=3840,
                barge_in_sent=True,
                speech_energy_sent=True,
                mixer_stop_calls=2,
                sidecar_closed=True,
                shutdown_elapsed_ms=1,
                shutdown_bounded=True,
                shutdown_timed_out=False,
                events=("transcript.final", "barge_in"),
                error="",
            )
        )

        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
        assert payload["kind"] == "discord_bridge"
        assert payload["transport"] == "discord_voice"
        assert payload["sidecar_pcm16_bytes"] == 640
        assert payload["sidecar_pcm16_first_sample"] == 450
        assert payload["sidecar_pcm16_checksum"] == 195
        assert payload["barge_in_sent"] is True
        assert payload["speech_energy_sent"] is True
        assert payload["sidecar_closed"] is True
        assert payload["shutdown_bounded"] is True
        assert payload["shutdown_timed_out"] is False
        assert payload["error"] is None

    def test_discord_voice_live_probe_report_payload_is_json_safe(self):
        payload = doctor._discord_voice_live_probe_report_payload(
            SimpleNamespace(
                ok=True,
                guild_name="jetha dev server",
                voice_channel_name="General",
                connect_perm=True,
                speak_perm=True,
                members_before=2,
                connected=True,
                opus_loaded=True,
                accepted_audio_source=True,
                played=True,
                playing_during_probe=True,
                receiver_started=True,
                receiver_frames=12,
                receiver_speech_start=1,
                inbound_observed=True,
                members_after=2,
                disconnected=True,
                require_inbound=True,
                wait_seconds=5.0,
                latency_metrics_ms={
                    "connect_ms": 420,
                    "playback_observed_ms": 180,
                    "inbound_observed_ms": 900,
                    "disconnect_ms": 120,
                },
                failure_reason="",
                error="",
            )
        )

        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
        assert payload["kind"] == "discord_live_probe"
        assert payload["guild_name"] == "jetha dev server"
        assert payload["voice_channel_name"] == "General"
        assert payload["receiver_frames"] == 12
        assert payload["inbound_observed"] is True
        assert payload["latency_metrics_ms"] == {
            "connect_ms": 420,
            "playback_observed_ms": 180,
            "inbound_observed_ms": 900,
            "disconnect_ms": 120,
        }
        assert payload["failure_reason"] is None
        assert payload["error"] is None

    def test_non_strict_disabled_realtime_voice_is_informational(self, monkeypatch, capsys):
        monkeypatch.setattr(doctor, "load_config", lambda: {}, raising=False)
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_status_payload",
            lambda: {
                "enabled": False,
                "available": False,
                "engine": "text_oracle_tts",
                "conversation_quality": {
                    "mode": "local_turn_based",
                    "reason": "disabled",
                    "live_like": False,
                },
                "language_support": {
                    "production_languages": ["en", "ja"],
                    "best_effort_languages": True,
                },
                "sidecar": {"mode": "none"},
            },
        )
        issues = []

        doctor._check_realtime_voice_readiness(issues, strict=False)

        output = capsys.readouterr().out
        assert "Realtime voice disabled" in output
        assert "skipping live voice readiness gate" in output
        assert issues == []

    def test_strict_requires_enabled_available_live_like_voice(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_status_payload",
            lambda: {
                "enabled": True,
                "available": False,
                "unavailable_reason": "live_like_required",
                "engine": "text_oracle_tts",
                "require_live_like": True,
                "conversation_quality": {
                    "mode": "turn_based_text",
                    "reason": "utterance_stt_tts",
                    "live_like": False,
                },
                "production_readiness": {
                    "ready": False,
                    "level": "not_ready",
                    "issues": ["live_like_required", "not_live_like", "missing_evidence_report"],
                },
                "language_support": {
                    "production_languages": ["en"],
                    "best_effort_languages": False,
                },
                "sidecar": {
                    "mode": "external",
                    "healthy": False,
                },
            },
        )
        issues = []

        doctor._check_realtime_voice_readiness(issues, strict=True)

        output = capsys.readouterr().out
        assert "Realtime voice preflight" in output
        assert "live_like_required" in output
        assert "turn_based_text" in output
        assert "Realtime voice production readiness" in output
        assert "missing production target(s): ja" in output
        assert "Best-effort languages disabled" in output
        assert "Voice sidecar health" in output
        assert "realtime_voice_profile --preset deepgram --apply --generate-bridge-token" in output
        assert "realtime_voice_profile --preset elevenlabs --apply --generate-bridge-token" in output
        assert "realtime_voice_deepgram_bridge --check --strict --production-en-ja" in output
        assert "realtime_voice_elevenlabs_bridge --check --strict --production-en-ja" in output
        assert "realtime_voice_alpha_evidence --runs 3 --apply" in output
        assert any("live-like" in issue for issue in issues)
        assert any("production_evidence_report" in issue for issue in issues)
        assert any("English and Japanese" in issue for issue in issues)
        assert any("best_effort_languages" in issue for issue in issues)

    def test_strict_accepts_portable_live_like_voice_config(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_status_payload",
            lambda: {
                "enabled": True,
                "available": True,
                "engine": "text_oracle_tts",
                "frontend_provider": "gemma4",
                "frontend_model": "google/gemma-4-E4B-it-qat-w4a16-ct",
                "conversation_quality": {
                    "mode": "streaming_text",
                    "reason": "streaming_stt_tts",
                    "live_like": True,
                },
                "production_readiness": {
                    "ready": True,
                    "level": "production_ready",
                    "issues": [],
                },
                "quality_targets_ms": {
                    "audio_to_partial_transcript_ms": 300,
                    "final_transcript_to_first_text_ms": 500,
                    "final_transcript_to_first_audio_ms": 900,
                    "barge_in_ack_ms": 150,
                },
                "language_support": {
                    "production_languages": ["en", "ja"],
                    "production_scripts": ["Latn", "Jpan"],
                    "best_effort_languages": True,
                },
                "sidecar": {
                    "mode": "external",
                    "healthy": True,
                },
            },
        )
        issues = []

        doctor._check_realtime_voice_readiness(issues, strict=True)

        output = capsys.readouterr().out
        assert "Realtime voice enabled" in output
        assert "Realtime voice preflight" in output
        assert "Live conversation quality" in output
        assert "Realtime voice latency targets" in output
        assert "Realtime voice production readiness" in output
        assert "Portable voice provider naming" in output
        assert "realtime_voice_profile --preset deepgram" not in output
        assert issues == []

    def test_strict_guides_missing_realtime_voice_production_review(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_status_payload",
            lambda: {
                "enabled": True,
                "available": True,
                "engine": "text_oracle_tts",
                "frontend_provider": "gemma4",
                "conversation_quality": {
                    "mode": "streaming_text",
                    "reason": "streaming_stt_tts",
                    "live_like": True,
                },
                "production_readiness": {
                    "ready": False,
                    "level": "evidence_ready",
                    "evidence_ready": True,
                    "issues": ["missing_production_review_report"],
                },
                "quality_targets_ms": {
                    "audio_to_partial_transcript_ms": 300,
                    "final_transcript_to_first_text_ms": 500,
                    "final_transcript_to_first_audio_ms": 900,
                    "barge_in_ack_ms": 150,
                },
                "language_support": {
                    "production_languages": ["en", "ja"],
                    "production_scripts": ["Latn", "Jpan"],
                    "best_effort_languages": True,
                },
                "sidecar": {
                    "mode": "external",
                    "healthy": True,
                },
            },
        )
        issues = []

        doctor._check_realtime_voice_readiness(issues, strict=True)

        output = capsys.readouterr().out
        assert "Realtime voice production readiness" in output
        assert "missing_production_review_report" in output
        assert "Production launch review:" in output
        assert "realtime_voice_production_review ./artifacts/realtime-voice-production-review.json --write-template" in output
        assert "realtime_voice_production_review ./artifacts/realtime-voice-production-review.json --apply" in output
        assert any("production_review_report" in issue for issue in issues)
        assert not any("production_evidence_report" in issue for issue in issues)

    def test_strict_guides_missing_realtime_voice_production_evidence(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_status_payload",
            lambda: {
                "enabled": True,
                "available": True,
                "engine": "text_oracle_tts",
                "frontend_provider": "gemma4",
                "conversation_quality": {
                    "mode": "streaming_text",
                    "reason": "streaming_stt_tts",
                    "live_like": True,
                },
                "production_readiness": {
                    "ready": False,
                    "level": "live_like",
                    "issues": ["missing_evidence_report"],
                },
                "quality_targets_ms": {
                    "audio_to_partial_transcript_ms": 300,
                    "final_transcript_to_first_text_ms": 500,
                    "final_transcript_to_first_audio_ms": 900,
                    "barge_in_ack_ms": 150,
                },
                "language_support": {
                    "production_languages": ["en", "ja"],
                    "production_scripts": ["Latn", "Jpan"],
                    "best_effort_languages": True,
                },
                "sidecar": {
                    "mode": "external",
                    "healthy": True,
                },
            },
        )
        issues = []

        doctor._check_realtime_voice_readiness(issues, strict=True)

        output = capsys.readouterr().out
        assert "Realtime voice production readiness" in output
        assert "missing_evidence_report" in output
        assert "Production evidence:" in output
        assert "realtime_voice_fixture_pack --output-dir ./fixtures/realtime-voice" in output
        assert "realtime_voice_alpha_evidence --runs 3 --apply" in output
        assert "realtime_voice_alpha_evidence --runs 3 --apply --provider elevenlabs --start-bridge" in output
        assert "realtime_voice_report ./artifacts/realtime-voice-evidence/*.json --alpha --min-runs 3" in output
        assert (
            "realtime_voice_report ./artifacts/realtime-voice-evidence/*.json --alpha "
            "--min-runs 3 --apply-production-evidence"
        ) in output
        assert "realtime_voice_live_evidence --require-live-discord --require-openai-realtime" in output
        assert any("production_evidence_report" in issue for issue in issues)
        assert "Production launch review:" not in output

    def test_strict_rejects_loose_realtime_voice_latency_targets(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_status_payload",
            lambda: {
                "enabled": True,
                "available": True,
                "engine": "text_oracle_tts",
                "frontend_provider": "gemma4",
                "conversation_quality": {
                    "mode": "streaming_text",
                    "reason": "streaming_stt_tts",
                    "live_like": True,
                },
                "production_readiness": {
                    "ready": False,
                    "level": "live_like",
                    "issues": ["loose_quality_targets"],
                },
                "quality_targets_ms": {
                    "audio_to_partial_transcript_ms": 800,
                    "final_transcript_to_first_text_ms": 500,
                    "final_transcript_to_first_audio_ms": 900,
                    "barge_in_ack_ms": 150,
                },
                "language_support": {
                    "production_languages": ["en", "ja"],
                    "best_effort_languages": True,
                },
                "sidecar": {
                    "mode": "external",
                    "healthy": True,
                },
            },
        )
        issues = []

        doctor._check_realtime_voice_readiness(issues, strict=True)

        output = capsys.readouterr().out
        assert "Realtime voice latency targets" in output
        assert "audio_to_partial_transcript_ms=800ms>300ms" in output
        assert any("quality_targets_ms" in issue for issue in issues)

    def test_strict_accepts_kame_realtime_voice_latency_targets(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_status_payload",
            lambda: {
                "enabled": True,
                "available": True,
                "engine": "kame_interface_oracle",
                "frontend_provider": "gemma4",
                "frontend_model": "gemma-4-E2B-it",
                "conversation_quality": {
                    "mode": "kame_reflex",
                    "reason": "audio_reflex_tts",
                    "live_like": True,
                    "kame_reflex": True,
                },
                "production_readiness": {
                    "ready": True,
                    "level": "production_ready",
                    "issues": [],
                },
                "quality_targets_ms": {
                    "audio_to_partial_transcript_ms": 300,
                    "final_transcript_to_first_text_ms": 500,
                    "final_transcript_to_first_audio_ms": 900,
                    "barge_in_ack_ms": 150,
                    "barge_in_confirmed_to_playback_stopped_ms": 150,
                    "kame_speech_end_to_interface_decision_ms": 500,
                    "kame_final_transcript_to_interface_decision_ms": 500,
                    "kame_interface_decision_to_local_first_audio_ms": 500,
                    "kame_speech_end_to_local_first_audio_ms": 1000,
                    "kame_interface_decision_to_defer_first_audio_ms": 500,
                    "kame_speech_end_to_defer_first_audio_ms": 500,
                    "kame_interface_decision_to_oracle_accepted_ms": 500,
                    "kame_oracle_first_token_to_first_tts_audio_ms": 1000,
                    "kame_first_tts_audio_to_playback_start_ms": 150,
                    "kame_speech_end_to_first_audio_ms": 3000,
                    "kame_speech_end_to_playback_start_ms": 3000,
                },
                "language_support": {
                    "production_languages": ["en", "ja"],
                    "production_scripts": ["Latn", "Jpan"],
                    "best_effort_languages": True,
                },
                "sidecar": {
                    "mode": "external",
                    "healthy": True,
                },
            },
        )
        issues = []

        doctor._check_realtime_voice_readiness(issues, strict=True)

        output = capsys.readouterr().out
        assert "Realtime voice latency targets" in output
        assert "kame_speech_end_to_interface_decision_ms=500ms" in output
        assert "kame_final_transcript_to_interface_decision_ms=500ms" in output
        assert "kame_speech_end_to_defer_first_audio_ms=500ms" in output
        assert "kame_speech_end_to_first_audio_ms=3000ms" in output
        assert issues == []

    def test_strict_rejects_missing_kame_realtime_voice_latency_targets(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_status_payload",
            lambda: {
                "enabled": True,
                "available": True,
                "engine": "kame_interface_oracle",
                "frontend_provider": "gemma4",
                "conversation_quality": {
                    "mode": "kame_reflex",
                    "reason": "audio_reflex_tts",
                    "live_like": True,
                    "kame_reflex": True,
                },
                "production_readiness": {
                    "ready": False,
                    "level": "live_like",
                    "issues": ["loose_quality_targets"],
                },
                "quality_targets_ms": {
                    "audio_to_partial_transcript_ms": 300,
                    "final_transcript_to_first_text_ms": 500,
                    "final_transcript_to_first_audio_ms": 900,
                    "barge_in_ack_ms": 150,
                },
                "language_support": {
                    "production_languages": ["en", "ja"],
                    "production_scripts": ["Latn", "Jpan"],
                    "best_effort_languages": True,
                },
                "sidecar": {
                    "mode": "external",
                    "healthy": True,
                },
            },
        )
        issues = []

        doctor._check_realtime_voice_readiness(issues, strict=True)

        output = capsys.readouterr().out
        assert "Realtime voice latency targets" in output
        assert "missing: barge_in_confirmed_to_playback_stopped_ms" in output
        assert "kame_speech_end_to_interface_decision_ms" in output
        assert any("quality_targets_ms" in issue for issue in issues)

    def test_strict_warns_on_machine_named_voice_provider(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_status_payload",
            lambda: {
                "enabled": True,
                "available": True,
                "engine": "text_oracle_tts",
                "frontend_provider": "pgx-workstation",
                "conversation_quality": {
                    "mode": "streaming_text",
                    "reason": "streaming_stt_tts",
                    "live_like": True,
                },
                "production_readiness": {
                    "ready": True,
                    "level": "production_ready",
                    "issues": [],
                },
                "quality_targets_ms": {
                    "audio_to_partial_transcript_ms": 300,
                    "final_transcript_to_first_text_ms": 500,
                    "final_transcript_to_first_audio_ms": 900,
                    "barge_in_ack_ms": 150,
                },
                "language_support": {
                    "production_languages": ["en", "ja"],
                    "best_effort_languages": True,
                },
                "sidecar": {
                    "mode": "external",
                    "healthy": True,
                },
            },
        )
        issues = []

        doctor._check_realtime_voice_readiness(issues, strict=True)

        output = capsys.readouterr().out
        assert "Portable voice provider naming" in output
        assert "machine or accelerator name" in output
        assert any("workstation/GPU" in issue for issue in issues)

    def test_sidecar_smoke_reports_success(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_smoke_config",
            lambda: SimpleNamespace(sidecar_connect_timeout_seconds=2.0),
        )
        monkeypatch.setattr(
            doctor,
            "_run_realtime_voice_sidecar_smoke_sync",
            lambda _config, *, timeout_seconds: SimpleNamespace(
                events=("frontend.state", "transcript.final"),
                ok=True,
                ready_ms=12,
                transcript_final_ms=25,
            ),
        )
        issues = []

        doctor._check_realtime_voice_sidecar_smoke(issues)

        output = capsys.readouterr().out
        assert "Realtime voice sidecar smoke" in output
        assert "ready=12ms" in output
        assert "transcript.final=25ms" in output
        assert issues == []

    def test_sidecar_smoke_records_failure_issue(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_smoke_config",
            lambda: SimpleNamespace(sidecar_connect_timeout_seconds=2.0),
        )
        monkeypatch.setattr(
            doctor,
            "_run_realtime_voice_sidecar_smoke_sync",
            lambda _config, *, timeout_seconds: SimpleNamespace(
                error="timed out after 2s waiting for transcript.final",
                events=("frontend.state",),
                ok=False,
            ),
        )
        issues = []

        doctor._check_realtime_voice_sidecar_smoke(issues)

        output = capsys.readouterr().out
        assert "Realtime voice sidecar smoke" in output
        assert "timed out after 2s" in output
        assert "events=frontend.state" in output
        assert any("protocol smoke failure" in issue for issue in issues)

    def test_discord_realtime_voice_smoke_reports_success(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_run_discord_realtime_voice_smoke_sync",
            lambda: SimpleNamespace(
                ok=True,
                mode="discord_loopback",
                transport="discord_voice",
                input_pcm48_bytes=3840,
                sidecar_pcm16_bytes=640,
                sidecar_pcm16_first_sample=450,
                sidecar_pcm16_checksum=195,
                mixer_frames=1,
                mixer_frame_bytes=3840,
                barge_in_sent=True,
                speech_energy_sent=True,
                mixer_stop_calls=2,
                sidecar_closed=True,
                shutdown_elapsed_ms=1,
                shutdown_bounded=True,
                shutdown_timed_out=False,
                events=("transcript.final", "barge_in"),
                error="",
            ),
        )
        issues = []
        reports = []

        doctor._check_discord_realtime_voice_smoke(issues, reports=reports)

        output = capsys.readouterr().out
        assert "Discord realtime voice bridge smoke" in output
        assert "pcm48=3840B" in output
        assert "pcm16=640B" in output
        assert "barge_in=yes" in output
        assert issues == []
        assert reports == [
            {
                "kind": "discord_bridge",
                "ok": True,
                "mode": "discord_loopback",
                "transport": "discord_voice",
                "input_pcm48_bytes": 3840,
                "sidecar_pcm16_bytes": 640,
                "sidecar_pcm16_first_sample": 450,
                "sidecar_pcm16_checksum": 195,
                "mixer_frames": 1,
                "mixer_frame_bytes": 3840,
                "barge_in_sent": True,
                "speech_energy_sent": True,
                "mixer_stop_calls": 2,
                "sidecar_closed": True,
                "shutdown_elapsed_ms": 1,
                "shutdown_bounded": True,
                "shutdown_timed_out": False,
                "events": ["transcript.final", "barge_in"],
                "error": None,
            }
        ]

    def test_discord_realtime_voice_smoke_records_failure_issue(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_run_discord_realtime_voice_smoke_sync",
            lambda: SimpleNamespace(
                ok=False,
                mode="discord_loopback",
                transport="discord_voice",
                input_pcm48_bytes=3840,
                sidecar_pcm16_bytes=0,
                sidecar_pcm16_first_sample=None,
                sidecar_pcm16_checksum=0,
                mixer_frames=0,
                mixer_frame_bytes=0,
                barge_in_sent=False,
                speech_energy_sent=False,
                mixer_stop_calls=0,
                sidecar_closed=False,
                shutdown_elapsed_ms=None,
                shutdown_bounded=False,
                shutdown_timed_out=False,
                events=("transcript.partial",),
                error="discord realtime loopback smoke did not satisfy invariants",
            ),
        )
        issues = []

        doctor._check_discord_realtime_voice_smoke(issues)

        output = capsys.readouterr().out
        assert "Discord realtime voice bridge smoke" in output
        assert "did not satisfy invariants" in output
        assert "events=transcript.partial" in output
        assert any("Discord realtime voice bridge smoke failure" in issue for issue in issues)

    def test_discord_voice_live_probe_reports_success(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_run_discord_voice_live_probe_sync",
            lambda **kwargs: SimpleNamespace(
                ok=True,
                guild_name="jetha dev server",
                voice_channel_name="General",
                connect_perm=True,
                speak_perm=True,
                members_before=2,
                connected=True,
                opus_loaded=True,
                accepted_audio_source=True,
                played=True,
                playing_during_probe=True,
                receiver_started=True,
                receiver_frames=15,
                receiver_speech_start=1,
                inbound_observed=True,
                members_after=2,
                disconnected=True,
                require_inbound=kwargs["require_inbound"],
                wait_seconds=kwargs["wait_seconds"],
                failure_reason="",
                error="",
            ),
        )
        issues = []
        reports = []

        doctor._check_discord_voice_live_probe(
            issues,
            reports=reports,
            wait_seconds=5.0,
            require_inbound=True,
            voice_channel_name="General",
        )

        output = capsys.readouterr().out
        assert "Discord live voice probe" in output
        assert "jetha dev server / General" in output
        assert "receiver_frames=15" in output
        assert "inbound=yes" in output
        assert issues == []
        assert reports[0]["kind"] == "discord_live_probe"
        assert reports[0]["ok"] is True
        assert reports[0]["receiver_frames"] == 15
        assert reports[0]["require_inbound"] is True

    def test_discord_voice_live_probe_explains_empty_channel_failure(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_run_discord_voice_live_probe_sync",
            lambda **_kwargs: SimpleNamespace(
                ok=False,
                guild_name="jetha dev server",
                voice_channel_name="General",
                connect_perm=True,
                speak_perm=True,
                members_before=0,
                connected=True,
                opus_loaded=True,
                accepted_audio_source=True,
                played=True,
                playing_during_probe=True,
                receiver_started=True,
                receiver_frames=0,
                receiver_speech_start=0,
                inbound_observed=False,
                members_after=0,
                disconnected=True,
                require_inbound=True,
                wait_seconds=1.0,
                failure_reason="inbound_required_but_no_other_members",
                error=(
                    "live Discord voice probe did not satisfy invariants: "
                    "inbound_required_but_no_other_members"
                ),
            ),
        )
        issues = []
        reports = []

        doctor._check_discord_voice_live_probe(
            issues,
            reports=reports,
            wait_seconds=1.0,
            require_inbound=True,
        )

        output = capsys.readouterr().out
        assert "Discord live voice probe" in output
        assert "inbound_required_but_no_other_members" in output
        assert any("controlled second Discord client" in issue for issue in issues)
        assert reports[0]["kind"] == "discord_live_probe"
        assert reports[0]["failure_reason"] == "inbound_required_but_no_other_members"

    def test_audio_fixture_smoke_reports_success_with_partial_target(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_smoke_config",
            lambda: SimpleNamespace(
                metadata={"quality_targets_ms": {"audio_to_partial_transcript_ms": 300}},
                sidecar_connect_timeout_seconds=2.0,
            ),
        )
        monkeypatch.setattr(
            doctor,
            "_run_realtime_voice_sidecar_audio_smoke_sync",
            lambda _config, *, audio_fixture_path, audio_codec, timeout_seconds: SimpleNamespace(
                audio_bytes=1234,
                events=("frontend.state", "transcript.partial", "transcript.final"),
                ok=True,
                transcript_final_ms=180,
                transcript_partial_ms=90,
            ),
        )
        issues = []

        doctor._check_realtime_voice_audio_fixture_smoke(
            issues,
            audio_fixture_path="hello.webm",
            audio_codec="webm_opus",
        )

        output = capsys.readouterr().out
        assert "Realtime voice audio fixture smoke" in output
        assert "bytes=1234" in output
        assert "transcript.partial=90ms <= 300ms" in output
        assert issues == []

    def test_audio_fixture_smoke_appends_machine_readable_report(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_smoke_config",
            lambda: SimpleNamespace(
                metadata={"quality_targets_ms": {"audio_to_partial_transcript_ms": 300}},
                sidecar_connect_timeout_seconds=2.0,
            ),
        )
        monkeypatch.setattr(
            doctor,
            "_run_realtime_voice_sidecar_audio_smoke_sync",
            lambda _config, *, audio_fixture_path, audio_codec, timeout_seconds: SimpleNamespace(
                audio_bytes=1234,
                events=("frontend.state", "transcript.partial", "transcript.final"),
                ok=True,
                transcript_final_ms=180,
                transcript_partial_ms=90,
            ),
        )
        issues = []
        reports = []

        doctor._check_realtime_voice_audio_fixture_smoke(
            issues,
            audio_fixture_path="hello-ja.webm",
            audio_codec="webm_opus",
            reports=reports,
        )

        assert issues == []
        assert reports == [
            {
                "kind": "audio_fixture",
                "ok": True,
                "ready_ms": None,
                "transcript_partial_ms": 90,
                "transcript_final_ms": 180,
                "first_text_ms": None,
                "first_audio_ms": None,
                "barge_in_ack_ms": None,
                "final_text": "",
                "assistant_final_text": None,
                "audio_bytes": 1234,
                "output_audio_bytes": 0,
                "audio_after_barge_in_bytes": 0,
                "events": ["frontend.state", "transcript.partial", "transcript.final"],
                "first_audio_metrics": None,
                "metrics": None,
                "route": None,
                "interface_input_source": None,
                "reflex_provider": None,
                "reflex_validation_error": None,
                "transport": None,
                "error": None,
                "fixture": "hello-ja.webm",
                "codec": "webm_opus",
                "target_ms": 300,
                "timeout_seconds": 2.0,
            }
        ]
        capsys.readouterr()

    def test_audio_fixture_smoke_records_missing_partial_issue(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_smoke_config",
            lambda: SimpleNamespace(
                metadata={"quality_targets_ms": {"audio_to_partial_transcript_ms": 300}},
                sidecar_connect_timeout_seconds=2.0,
            ),
        )
        monkeypatch.setattr(
            doctor,
            "_run_realtime_voice_sidecar_audio_smoke_sync",
            lambda _config, *, audio_fixture_path, audio_codec, timeout_seconds: SimpleNamespace(
                audio_bytes=1234,
                events=("frontend.state", "transcript.final"),
                ok=True,
                transcript_final_ms=180,
                transcript_partial_ms=None,
            ),
        )
        issues = []

        doctor._check_realtime_voice_audio_fixture_smoke(
            issues,
            audio_fixture_path="hello.webm",
            audio_codec="webm_opus",
        )

        output = capsys.readouterr().out
        assert "missing transcript.partial" in output
        assert any("audio fixture smoke failure" in issue for issue in issues)

    def test_audio_fixture_smoke_records_partial_target_miss(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_smoke_config",
            lambda: SimpleNamespace(
                metadata={"quality_targets_ms": {"audio_to_partial_transcript_ms": 300}},
                sidecar_connect_timeout_seconds=2.0,
            ),
        )
        monkeypatch.setattr(
            doctor,
            "_run_realtime_voice_sidecar_audio_smoke_sync",
            lambda _config, *, audio_fixture_path, audio_codec, timeout_seconds: SimpleNamespace(
                audio_bytes=1234,
                events=("frontend.state", "transcript.partial", "transcript.final"),
                ok=True,
                transcript_final_ms=500,
                transcript_partial_ms=450,
            ),
        )
        issues = []

        doctor._check_realtime_voice_audio_fixture_smoke(
            issues,
            audio_fixture_path="hello.webm",
            audio_codec="webm_opus",
        )

        output = capsys.readouterr().out
        assert "transcript.partial=450ms exceeds target 300ms" in output
        assert any("audio fixture smoke failure" in issue for issue in issues)

    def test_tts_smoke_reports_success_with_first_audio_target(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_smoke_config",
            lambda: SimpleNamespace(
                metadata={"quality_targets_ms": {"final_transcript_to_first_audio_ms": 900}},
                sidecar_connect_timeout_seconds=2.0,
            ),
        )
        monkeypatch.setattr(
            doctor,
            "_run_realtime_voice_sidecar_tts_smoke_sync",
            lambda _config, *, text, metadata=None, timeout_seconds: SimpleNamespace(
                events=("frontend.state", "audio.output.chunk"),
                first_audio_ms=250,
                ok=True,
                output_audio_bytes=4321,
            ),
        )
        issues = []

        doctor._check_realtime_voice_tts_smoke(issues, text="Hello from Hermes.")

        output = capsys.readouterr().out
        assert "Realtime voice TTS smoke" in output
        assert "audio.bytes=4321" in output
        assert "first_audio=250ms <= 900ms" in output
        assert issues == []

    def test_tts_smoke_records_first_audio_target_miss(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_smoke_config",
            lambda: SimpleNamespace(
                metadata={"quality_targets_ms": {"final_transcript_to_first_audio_ms": 900}},
                sidecar_connect_timeout_seconds=2.0,
            ),
        )
        monkeypatch.setattr(
            doctor,
            "_run_realtime_voice_sidecar_tts_smoke_sync",
            lambda _config, *, text, metadata=None, timeout_seconds: SimpleNamespace(
                events=("frontend.state", "audio.output.chunk"),
                first_audio_ms=1200,
                ok=True,
                output_audio_bytes=4321,
            ),
        )
        issues = []

        doctor._check_realtime_voice_tts_smoke(issues, text="Hello from Hermes.")

        output = capsys.readouterr().out
        assert "first_audio=1200ms exceeds target 900ms" in output
        assert any("TTS smoke failure" in issue for issue in issues)

    def test_barge_in_smoke_reports_success_with_ack_target(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_smoke_config",
            lambda: SimpleNamespace(
                metadata={"quality_targets_ms": {"barge_in_ack_ms": 150}},
                sidecar_connect_timeout_seconds=2.0,
            ),
        )
        monkeypatch.setattr(
            doctor,
            "_run_realtime_voice_sidecar_barge_in_smoke_sync",
            lambda _config, *, text, timeout_seconds: SimpleNamespace(
                barge_in_ack_ms=45,
                events=("frontend.state", "barge_in"),
                ok=True,
            ),
        )
        issues = []

        doctor._check_realtime_voice_barge_in_smoke(issues, text="Hello from Hermes.")

        output = capsys.readouterr().out
        assert "Realtime voice barge-in smoke" in output
        assert "barge_in_ack=45ms <= 150ms" in output
        assert issues == []

    def test_session_turn_smoke_reports_success_with_text_and_audio_targets(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_smoke_config",
            lambda: SimpleNamespace(
                metadata={
                    "quality_targets_ms": {
                        "final_transcript_to_first_text_ms": 500,
                        "final_transcript_to_first_audio_ms": 900,
                    }
                },
                sidecar_connect_timeout_seconds=2.0,
            ),
        )
        monkeypatch.setattr(
            doctor,
            "_run_realtime_voice_session_turn_smoke_sync",
            lambda _config, *, text, timeout_seconds: SimpleNamespace(
                events=("session.started", "transcript.final", "assistant.text.partial", "audio.output.chunk"),
                first_audio_ms=240,
                first_text_ms=80,
                ok=True,
                output_audio_bytes=4321,
            ),
        )
        issues = []

        doctor._check_realtime_voice_session_turn_smoke(issues, text="Hello from Hermes.")

        output = capsys.readouterr().out
        assert "Realtime voice session turn smoke" in output
        assert "first_text=80ms <= 500ms" in output
        assert "first_audio=240ms <= 900ms" in output
        assert issues == []

    def test_session_turn_smoke_records_first_text_target_miss(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_smoke_config",
            lambda: SimpleNamespace(
                metadata={
                    "quality_targets_ms": {
                        "final_transcript_to_first_text_ms": 500,
                        "final_transcript_to_first_audio_ms": 900,
                    }
                },
                sidecar_connect_timeout_seconds=2.0,
            ),
        )
        monkeypatch.setattr(
            doctor,
            "_run_realtime_voice_session_turn_smoke_sync",
            lambda _config, *, text, timeout_seconds: SimpleNamespace(
                events=("session.started", "transcript.final", "assistant.text.partial", "audio.output.chunk"),
                first_audio_ms=240,
                first_text_ms=650,
                ok=True,
                output_audio_bytes=4321,
            ),
        )
        issues = []

        doctor._check_realtime_voice_session_turn_smoke(issues, text="Hello from Hermes.")

        output = capsys.readouterr().out
        assert "first_text=650ms exceeds target 500ms" in output
        assert any("session turn smoke failure" in issue for issue in issues)

    def test_session_audio_smoke_reports_success_with_all_targets(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_smoke_config",
            lambda: SimpleNamespace(
                metadata={
                    "quality_targets_ms": {
                        "audio_to_partial_transcript_ms": 300,
                        "final_transcript_to_first_text_ms": 500,
                        "final_transcript_to_first_audio_ms": 900,
                    }
                },
                sidecar_connect_timeout_seconds=2.0,
            ),
        )
        monkeypatch.setattr(
            doctor,
            "_run_realtime_voice_session_audio_smoke_sync",
            lambda _config, *, audio_fixture_path, audio_codec, timeout_seconds: SimpleNamespace(
                audio_bytes=1234,
                events=(
                    "session.started",
                    "frontend.state",
                    "transcript.partial",
                    "transcript.final",
                    "assistant.text.partial",
                    "audio.output.chunk",
                ),
                first_audio_ms=240,
                first_text_ms=80,
                ok=True,
                output_audio_bytes=4321,
                transcript_final_ms=180,
                transcript_partial_ms=90,
            ),
        )
        issues = []

        doctor._check_realtime_voice_session_audio_smoke(
            issues,
            audio_fixture_path="hello.webm",
            audio_codec="webm_opus",
        )

        output = capsys.readouterr().out
        assert "Realtime voice audio session smoke" in output
        assert "transcript.partial=90ms <= 300ms" in output
        assert "first_text=80ms <= 500ms" in output
        assert "first_audio=240ms <= 900ms" in output
        assert issues == []

    def test_session_audio_smoke_appends_machine_readable_report(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_smoke_config",
            lambda: SimpleNamespace(
                metadata={
                    "quality_targets_ms": {
                        "audio_to_partial_transcript_ms": 300,
                        "final_transcript_to_first_text_ms": 500,
                        "final_transcript_to_first_audio_ms": 900,
                    }
                },
                sidecar_connect_timeout_seconds=2.0,
            ),
        )
        monkeypatch.setattr(
            doctor,
            "_run_realtime_voice_session_audio_smoke_sync",
            lambda _config, *, audio_fixture_path, audio_codec, timeout_seconds: SimpleNamespace(
                audio_bytes=1234,
                events=("session.started", "transcript.partial", "transcript.final", "assistant.text.partial", "audio.output.chunk"),
                final_text="こんにちは、Hermesです。",
                first_audio_ms=240,
                first_text_ms=80,
                ok=True,
                output_audio_bytes=4321,
                transcript_final_ms=180,
                transcript_partial_ms=90,
            ),
        )
        issues = []
        reports = []

        doctor._check_realtime_voice_session_audio_smoke(
            issues,
            audio_fixture_path="hello-ja.webm",
            audio_codec="webm_opus",
            reports=reports,
        )

        assert issues == []
        assert reports == [
            {
                "kind": "audio_session",
                "ok": True,
                "ready_ms": None,
                "transcript_partial_ms": 90,
                "transcript_final_ms": 180,
                "first_text_ms": 80,
                "first_audio_ms": 240,
                "barge_in_ack_ms": None,
                "final_text": "こんにちは、Hermesです。",
                "assistant_final_text": None,
                "audio_bytes": 1234,
                "output_audio_bytes": 4321,
                "audio_after_barge_in_bytes": 0,
                "events": [
                    "session.started",
                    "transcript.partial",
                    "transcript.final",
                    "assistant.text.partial",
                    "audio.output.chunk",
                ],
                "first_audio_metrics": None,
                "metrics": None,
                "route": None,
                "interface_input_source": None,
                "reflex_provider": None,
                "reflex_validation_error": None,
                "transport": None,
                "error": None,
                "fixture": "hello-ja.webm",
                "codec": "webm_opus",
                "target_ms": 300,
                "first_text_target_ms": 500,
                "first_audio_target_ms": 900,
                "timeout_seconds": 2.0,
            }
        ]
        capsys.readouterr()

    def test_session_audio_smoke_records_first_audio_target_miss(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_smoke_config",
            lambda: SimpleNamespace(
                metadata={
                    "quality_targets_ms": {
                        "audio_to_partial_transcript_ms": 300,
                        "final_transcript_to_first_text_ms": 500,
                        "final_transcript_to_first_audio_ms": 900,
                    }
                },
                sidecar_connect_timeout_seconds=2.0,
            ),
        )
        monkeypatch.setattr(
            doctor,
            "_run_realtime_voice_session_audio_smoke_sync",
            lambda _config, *, audio_fixture_path, audio_codec, timeout_seconds: SimpleNamespace(
                audio_bytes=1234,
                events=("session.started", "transcript.partial", "transcript.final", "assistant.text.partial", "audio.output.chunk"),
                first_audio_ms=1200,
                first_text_ms=80,
                ok=True,
                output_audio_bytes=4321,
                transcript_final_ms=180,
                transcript_partial_ms=90,
            ),
        )
        issues = []

        doctor._check_realtime_voice_session_audio_smoke(
            issues,
            audio_fixture_path="hello.webm",
            audio_codec="webm_opus",
        )

        output = capsys.readouterr().out
        assert "first_audio=1200ms exceeds target 900ms" in output
        assert any("audio session smoke failure" in issue for issue in issues)

    def test_barge_in_smoke_records_ack_target_miss(self, monkeypatch, capsys):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_smoke_config",
            lambda: SimpleNamespace(
                metadata={"quality_targets_ms": {"barge_in_ack_ms": 150}},
                sidecar_connect_timeout_seconds=2.0,
            ),
        )
        monkeypatch.setattr(
            doctor,
            "_run_realtime_voice_sidecar_barge_in_smoke_sync",
            lambda _config, *, text, timeout_seconds: SimpleNamespace(
                barge_in_ack_ms=240,
                events=("frontend.state", "barge_in"),
                ok=True,
            ),
        )
        issues = []

        doctor._check_realtime_voice_barge_in_smoke(issues, text="Hello from Hermes.")

        output = capsys.readouterr().out
        assert "barge_in_ack=240ms exceeds target 150ms" in output
        assert any("barge-in smoke failure" in issue for issue in issues)

    def test_realtime_voice_report_writer_preserves_unicode(self, tmp_path):
        report_path = tmp_path / "voice-smoke.json"

        doctor._write_realtime_voice_report(
            report_path,
            [{"kind": "tts", "ok": True, "final_text": "こんにちは"}],
        )

        assert json.loads(report_path.read_text(encoding="utf-8")) == [
            {"kind": "tts", "ok": True, "final_text": "こんにちは"}
        ]

    def test_realtime_voice_evidence_manifest_is_sanitized(self, monkeypatch):
        monkeypatch.setattr(
            doctor,
            "_realtime_voice_status_payload",
            lambda: {
                "engine": "text_oracle_tts",
                "frontend_provider": "reference",
                "frontend_model": "portable-voice",
                "interface_temperature": 0.25,
                "interface_max_output_tokens": 128,
                "interface_timeout_seconds": 0.9,
                "interface_max_audio_seconds": 18.0,
                "available": True,
                "require_live_like": True,
                "quality_targets_ms": {
                    "audio_to_partial_transcript_ms": 300,
                    "final_transcript_to_first_audio_ms": 900,
                    "barge_in_ack_ms": 150,
                },
                "conversation_quality": {
                    "live_like": True,
                    "reason": "streaming_stt_tts",
                    "streaming_stt": True,
                    "tts": True,
                },
                "language_support": {
                    "production_languages": ["en", "ja"],
                    "best_effort_languages": True,
                },
                "sidecar": {
                    "mode": "managed_loopback",
                    "base_url": "http://user:secret@voice.local:8765",
                    "healthy": True,
                    "health": {
                        "ok": True,
                        "kind": "reference",
                        "frontend": {
                            "provider": "streaming_stt",
                            "model": "portable-asr",
                            "tts_model_languages": ["ja", "en", "https://voice.local/secret"],
                            "streaming_tts_bridge": {
                                "configured": True,
                                "healthy": True,
                                "model": "portable-voice",
                            },
                            "unsafe_url": "http://user:secret@voice.local",
                        },
                        "capabilities": {
                            "streaming_stt": True,
                            "streaming_stt_bridge": True,
                            "tts": True,
                            "streaming_tts": True,
                            "streaming_tts_bridge": True,
                            "output_languages": ["en", "ja", "token=secret"],
                            "token": "secret",
                        },
                    },
                },
                "production_readiness": {
                    "ready": False,
                    "level": "not_ready",
                    "issues": ["missing_evidence_report"],
                    "evidence": {
                        "runs": 0,
                        "min_runs": 3,
                        "verified": False,
                        "report_path": "C:/secret/report.json",
                    },
                },
            },
        )

        report = [{"kind": "tts", "ok": True}]
        doctor._append_realtime_voice_evidence_manifest(report)

        manifest = report[0]
        assert manifest["kind"] == "manifest"
        assert manifest["ok"] is True
        assert manifest["interface_temperature"] == 0.25
        assert manifest["interface_max_output_tokens"] == 128
        assert manifest["interface_timeout_seconds"] == 0.9
        assert manifest["interface_max_audio_seconds"] == 18.0
        assert manifest["conversation_quality"]["live_like"] is True
        assert manifest["sidecar"] == {
            "mode": "managed_loopback",
            "healthy": True,
            "health": {
                "ok": True,
                "kind": "reference",
                "frontend": {
                    "provider": "streaming_stt",
                    "model": "portable-asr",
                    "tts_model_languages": ["ja", "en"],
                    "languages": [],
                    "scripts": [],
                    "streaming_tts_bridge": {
                        "configured": True,
                        "healthy": True,
                        "model": "portable-voice",
                    },
                },
                "capabilities": {
                    "utterance_stt": False,
                    "streaming_stt": True,
                    "tts": True,
                    "native_s2s": False,
                    "vllm_audio_frontend": False,
                    "input_languages": [],
                    "output_languages": ["en", "ja"],
                    "scripts": [],
                    "streaming_stt_bridge": True,
                    "streaming_tts": True,
                    "streaming_tts_bridge": True,
                },
                "local": {"stt": False, "tts": False},
            },
        }
        assert "secret" not in json.dumps(manifest)
        assert "unsafe_url" not in json.dumps(manifest)
        assert report[1] == {"kind": "tts", "ok": True}

    def test_realtime_voice_alpha_expands_required_smokes(self):
        smoke, audio, tts, barge = doctor._realtime_voice_expand_alpha_smokes(
            alpha=True,
            smoke=False,
            audio_fixtures=["./fixtures/realtime-voice/en/hello.webm", "custom.webm"],
            tts_texts=["Hello from Hermes.", "Extra phrase."],
            barge_in_texts=[],
        )

        assert smoke is True
        assert audio == [*ALPHA_REQUIRED_AUDIO_FIXTURES, "custom.webm"]
        assert tts == [*ALPHA_REQUIRED_TTS_TEXTS, "Extra phrase."]
        assert barge == list(ALPHA_REQUIRED_BARGE_IN_TEXTS)
        assert doctor._realtime_voice_alpha_audio_session_fixtures() == list(
            ALPHA_REQUIRED_AUDIO_SESSION_FIXTURES
        )
        assert doctor._realtime_voice_alpha_session_turn_texts() == [
            "Hello from Hermes.",
            "こんにちは、Hermesです。",
        ]


class TestDoctorEnvFileEncoding:
    """Regression for #18637 (bug 3): `hermes doctor` crashed on Windows
    Chinese locale (GBK) because `.env` was read with Path.read_text() which
    defaults to the system locale encoding, not UTF-8."""

    def test_doctor_reads_env_as_utf8_even_when_locale_is_not_utf8(
        self, monkeypatch, tmp_path
    ):
        import pathlib

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        # Write a UTF-8 .env containing an em dash (U+2014 = e2 80 94). The
        # 0x94 byte is exactly the one the issue reporter hit: it's invalid
        # as a GBK trailing byte in this position, so locale-default reads
        # raise UnicodeDecodeError on Chinese Windows.
        env_path = hermes_home / ".env"
        env_path.write_text(
            "OPENAI_API_KEY=sk-test  # em-dash here — should not crash\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", hermes_home)

        orig_read_text = pathlib.Path.read_text

        def gbk_like_read_text(self, encoding=None, errors=None, **kwargs):
            # Simulate a GBK locale: refuse to decode this specific UTF-8
            # .env unless the caller pins encoding="utf-8".
            if self == env_path and encoding != "utf-8":
                raise UnicodeDecodeError(
                    "gbk", b"\x94", 0, 1, "illegal multibyte sequence"
                )
            return orig_read_text(self, encoding=encoding, errors=errors, **kwargs)

        monkeypatch.setattr(pathlib.Path, "read_text", gbk_like_read_text)

        # Short-circuit the expensive tool-availability probe — we only
        # need doctor to reach the .env read without crashing.
        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: (_ for _ in ()).throw(SystemExit(0)),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

        # Run doctor. If the .env read still uses locale encoding, this
        # raises UnicodeDecodeError and the test fails.
        with pytest.raises(SystemExit):
            doctor_mod.run_doctor(Namespace(fix=False))


class TestDoctorToolAvailabilityOverrides:
    def test_marks_honcho_available_when_configured(self, monkeypatch):
        monkeypatch.setattr(doctor, "_honcho_is_configured_for_doctor", lambda: True)

        available, unavailable = doctor._apply_doctor_tool_availability_overrides(
            [],
            [{"name": "honcho", "env_vars": [], "tools": ["query_user_context"]}],
        )

        assert available == ["honcho"]
        assert unavailable == []

    def test_leaves_honcho_unavailable_when_not_configured(self, monkeypatch):
        monkeypatch.setattr(doctor, "_honcho_is_configured_for_doctor", lambda: False)

        honcho_entry = {"name": "honcho", "env_vars": [], "tools": ["query_user_context"]}
        available, unavailable = doctor._apply_doctor_tool_availability_overrides(
            [],
            [honcho_entry],
        )

        assert available == []
        assert unavailable == [honcho_entry]

    def test_marks_kanban_available_only_when_missing_worker_env_gate(self, monkeypatch):
        monkeypatch.setattr(doctor, "_honcho_is_configured_for_doctor", lambda: False)
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

        available, unavailable = doctor._apply_doctor_tool_availability_overrides(
            [],
            [{"name": "kanban", "env_vars": [], "tools": ["kanban_show"]}],
        )

        assert available == ["kanban"]
        assert unavailable == []

    def test_leaves_kanban_unavailable_when_worker_env_is_set(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "probe")
        kanban_entry = {"name": "kanban", "env_vars": [], "tools": ["kanban_show"]}

        available, unavailable = doctor._apply_doctor_tool_availability_overrides(
            [],
            [kanban_entry],
        )

        assert available == []
        assert unavailable == [kanban_entry]

    def test_leaves_non_worker_kanban_failure_unavailable(self, monkeypatch):
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        kanban_entry = {"name": "kanban", "env_vars": [], "tools": ["kanban_show", "not_a_kanban_tool"]}

        available, unavailable = doctor._apply_doctor_tool_availability_overrides(
            [],
            [kanban_entry],
        )

        assert available == []
        assert unavailable == [kanban_entry]

    def test_kanban_doctor_detail_explains_worker_gate(self, monkeypatch):
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

        assert doctor._doctor_tool_availability_detail("kanban") == "(runtime-gated; loaded only for dispatcher-spawned workers)"


class TestHonchoDoctorConfigDetection:
    def test_reports_configured_when_enabled_with_api_key(self, monkeypatch):
        fake_config = SimpleNamespace(enabled=True, api_key="***")

        monkeypatch.setattr(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            lambda: fake_config,
        )

        assert doctor._honcho_is_configured_for_doctor()

    def test_reports_not_configured_without_api_key(self, monkeypatch):
        fake_config = SimpleNamespace(enabled=True, api_key="")

        monkeypatch.setattr(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            lambda: fake_config,
        )

        assert not doctor._honcho_is_configured_for_doctor()


def test_run_doctor_sets_interactive_env_for_tool_checks(monkeypatch, tmp_path):
    """Doctor should present CLI-gated tools as available in CLI context."""
    project_root = tmp_path / "project"
    hermes_home = tmp_path / ".hermes"
    project_root.mkdir()
    hermes_home.mkdir()

    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", hermes_home)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    seen = {}

    def fake_check_tool_availability(*args, **kwargs):
        seen["interactive"] = os.getenv("HERMES_INTERACTIVE")
        raise SystemExit(0)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=fake_check_tool_availability,
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    with pytest.raises(SystemExit):
        doctor_mod.run_doctor(Namespace(fix=False))

    assert seen["interactive"] == "1"


def test_check_gateway_service_linger_warns_when_disabled(monkeypatch, tmp_path, capsys):
    unit_path = tmp_path / "hermes-gateway.service"
    unit_path.write_text("[Unit]\n")

    monkeypatch.setattr(gateway_cli, "is_linux", lambda: True)
    monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda: unit_path)
    monkeypatch.setattr(gateway_cli, "get_systemd_linger_status", lambda: (False, ""))

    issues = []
    doctor._check_gateway_service_linger(issues)

    out = capsys.readouterr().out
    assert "Gateway Service" in out
    assert "Systemd linger disabled" in out
    assert "loginctl enable-linger" in out
    assert issues == [
        "Enable linger for the gateway user service: sudo loginctl enable-linger $USER"
    ]


def test_check_gateway_service_linger_skips_when_service_not_installed(monkeypatch, tmp_path, capsys):
    unit_path = tmp_path / "missing.service"

    monkeypatch.setattr(gateway_cli, "is_linux", lambda: True)
    monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda: unit_path)

    issues = []
    doctor._check_gateway_service_linger(issues)

    out = capsys.readouterr().out
    assert out == ""
    assert issues == []


# ── Memory provider section (doctor should only check the *active* provider) ──


class TestDoctorMemoryProviderSection:
    """The ◆ Memory Provider section should respect memory.provider config."""

    def _make_hermes_home(self, tmp_path, provider=""):
        """Create a minimal HERMES_HOME with config.yaml."""
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        import yaml
        config = {"memory": {"provider": provider}} if provider else {"memory": {}}
        (home / "config.yaml").write_text(yaml.dump(config))
        return home

    def _run_doctor_and_capture(self, monkeypatch, tmp_path, provider=""):
        """Run doctor and capture stdout."""
        home = self._make_hermes_home(tmp_path, provider)
        monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
        monkeypatch.setattr(doctor_mod, "_DHH", str(home))
        (tmp_path / "project").mkdir(exist_ok=True)

        # Stub tool availability (returns empty) so doctor runs past it
        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: ([], []),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

        # Stub auth checks to avoid real API calls
        try:
            from hermes_cli import auth as _auth_mod
            monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
            monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
            monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
        except Exception:
            pass

        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=False))
        return buf.getvalue()

    def test_no_provider_shows_builtin_ok(self, monkeypatch, tmp_path):
        out = self._run_doctor_and_capture(monkeypatch, tmp_path, provider="")
        assert "Memory Provider" in out
        assert "Built-in memory active" in out
        # Should NOT mention Honcho or Mem0 errors
        assert "Honcho API key" not in out
        assert "Mem0" not in out

    def test_honcho_provider_not_installed_shows_fail(self, monkeypatch, tmp_path):
        # Make honcho import fail
        monkeypatch.setitem(
            sys.modules, "plugins.memory.honcho.client", None
        )
        out = self._run_doctor_and_capture(monkeypatch, tmp_path, provider="honcho")
        assert "Memory Provider" in out
        # Should show failure since honcho is set but not importable
        assert "Built-in memory active" not in out

    def test_mem0_provider_not_installed_shows_fail(self, monkeypatch, tmp_path):
        # Make mem0 import fail
        monkeypatch.setitem(sys.modules, "plugins.memory.mem0", None)
        out = self._run_doctor_and_capture(monkeypatch, tmp_path, provider="mem0")
        assert "Memory Provider" in out
        assert "Built-in memory active" not in out


def test_run_doctor_termux_treats_docker_and_browser_warnings_as_expected(monkeypatch, tmp_path):
    helper = TestDoctorMemoryProviderSection()
    monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")

    real_which = doctor_mod.shutil.which

    def fake_which(cmd):
        if cmd in {"docker", "node", "npm"}:
            return None
        return real_which(cmd)

    monkeypatch.setattr(doctor_mod.shutil, "which", fake_which)

    out = helper._run_doctor_and_capture(monkeypatch, tmp_path, provider="")

    assert "Docker backend is not available inside Termux" in out
    assert "Node.js not found (browser tools are optional in the tested Termux path)" in out
    assert "Install Node.js on Termux with: pkg install nodejs" in out
    assert "Termux browser setup:" in out
    assert "1) pkg install nodejs" in out
    assert "2) npm install -g agent-browser" in out
    assert "3) agent-browser install" in out
    assert "Termux compatibility fallbacks:" in out
    assert "use .[termux-all] for broad compatibility" in out
    assert "Matrix E2EE extra is excluded on Termux" in out
    assert "Local faster-whisper extra is excluded on Termux" in out
    assert "STT fallback: use Groq Whisper (set GROQ_API_KEY) or OpenAI Whisper (set VOICE_TOOLS_OPENAI_KEY)." in out
    assert "docker not found (optional)" not in out


def test_run_doctor_accepts_named_provider_from_providers_section(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)

    import yaml

    (home / "config.yaml").write_text(
        yaml.dump(
            {
                "model": {
                    "provider": "volcengine-plan",
                    "default": "doubao-seed-2.0-code",
                },
                "providers": {
                    "volcengine-plan": {
                        "name": "volcengine-plan",
                        "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
                        "default_model": "doubao-seed-2.0-code",
                        "models": {"doubao-seed-2.0-code": {}},
                    }
                },
            }
        )
    )

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    (tmp_path / "project").mkdir(exist_ok=True)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    try:
        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except Exception:
        pass

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))

    out = buf.getvalue()
    assert "model.provider 'volcengine-plan' is not a recognised provider" not in out


def test_run_doctor_accepts_bare_custom_provider(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: custom\n"
        "  default: local-model\n"
        "  base_url: http://localhost:8000/v1\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    (tmp_path / "project").mkdir(exist_ok=True)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    try:
        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except Exception:
        pass

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))

    out = buf.getvalue()
    assert "model.provider 'custom' is not a recognised provider" not in out


def test_run_doctor_flags_missing_credentials_for_active_openrouter_provider(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: openrouter\n"
        "  default: openai/gpt-4.1-mini\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    (tmp_path / "project").mkdir(exist_ok=True)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        from hermes_cli import auth as _auth_mod

        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_minimax_oauth_auth_status", lambda: {})
    except Exception:
        pass

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))

    out = buf.getvalue()
    assert "model.provider 'openrouter' is set but no API key is configured" in out
    assert "No credentials found for provider 'openrouter'." in out


@pytest.mark.parametrize(
    ("provider", "default_model"),
    [
        ("opencode-zen", "anthropic/claude-sonnet-4.6"),
        ("kilocode", "anthropic/claude-sonnet-4.6"),
        ("kimi-coding", "kimi-k2"),
        ("nvidia", "qwen/qwen3.5-122b-a10b"),
    ],
)
def test_run_doctor_accepts_hermes_provider_ids_that_catalog_aliases(
    monkeypatch, tmp_path, provider, default_model
):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n"
        f"  provider: {provider}\n"
        f"  default: {default_model}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    (tmp_path / "project").mkdir(exist_ok=True)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    try:
        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except Exception:
        pass

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))

    out = buf.getvalue()
    assert f"model.provider '{provider}' is not a recognised provider" not in out
    assert f"model.provider '{provider}' is unknown" not in out
    if provider in {"opencode-zen", "kilocode", "nvidia"}:
        assert (
            f"model.default '{default_model}' uses a vendor/model slug but provider is '{provider}'"
            not in out
        )


def test_run_doctor_accepts_vendor_slugs_for_named_custom_provider(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: custom:hpc-ai\n"
        "  default: deepseek/deepseek-v4-flash\n"
        "custom_providers:\n"
        "  - name: hpc-ai\n"
        "    base_url: https://hpc-ai.example/v1\n"
        "    api_key: test-key\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    (tmp_path / "project").mkdir(exist_ok=True)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    try:
        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except Exception:
        pass

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))

    out = buf.getvalue()
    assert "model.provider 'custom:hpc-ai' is not a recognised provider" not in out
    assert "model.provider 'custom:hpc-ai' is unknown" not in out
    assert (
        "model.default 'deepseek/deepseek-v4-flash' uses a vendor/model slug but provider is "
        "'custom:hpc-ai'"
        not in out
    )
    assert "Either set model.provider to 'openrouter', or drop the vendor prefix." not in out




def test_run_doctor_accepts_kimi_coding_cn_provider(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text("KIMI_CN_API_KEY=***\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: kimi-coding-cn\n"
        "  default: kimi-k2.6\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    (tmp_path / "project").mkdir(exist_ok=True)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    try:
        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_auth_status", lambda provider: {"logged_in": True})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except Exception:
        pass

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))

    out = buf.getvalue()
    assert "model.provider 'kimi-coding-cn' is not a recognised provider" not in out


def test_run_doctor_termux_does_not_mark_browser_available_without_agent_browser(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)

    monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda cmd: "/data/data/com.termux/files/usr/bin/node" if cmd in {"node", "npm"} else None)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: (["terminal"], [{"name": "browser", "env_vars": [], "tools": ["browser_navigate"]}]),
        TOOLSET_REQUIREMENTS={
            "terminal": {"name": "terminal"},
            "browser": {"name": "browser"},
        },
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    try:
        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except Exception:
        pass

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))
    out = buf.getvalue()

    assert "✓ browser" not in out
    assert "browser" in out
    assert "system dependency not met" in out
    assert "agent-browser is not installed (expected in the tested Termux path)" in out
    assert "npm install -g agent-browser && agent-browser install" in out


def test_run_doctor_kimi_cn_env_is_detected_and_probe_is_null_safe(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
    (home / ".env").write_text("KIMI_CN_API_KEY=sk-test\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    monkeypatch.setenv("KIMI_CN_API_KEY", "sk-test")

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    try:
        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except Exception:
        pass

    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, headers, timeout))
        return types.SimpleNamespace(status_code=200)

    import httpx
    monkeypatch.setattr(httpx, "get", fake_get)

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))
    out = buf.getvalue()

    assert "API key or custom endpoint configured" in out
    assert "Kimi / Moonshot (China)" in out
    assert "str expected, not NoneType" not in out
    assert any(url == "https://api.moonshot.cn/v1/models" for url, _, _ in calls)


def test_run_doctor_dashscope_retries_china_endpoint_after_intl_unauthorized(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
    (home / ".env").write_text("DASHSCOPE_API_KEY=sk-test\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    try:
        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except ImportError:
        pass

    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, headers, timeout))
        status = 200 if "dashscope.aliyuncs.com" in url else 401
        return types.SimpleNamespace(status_code=status)

    import httpx
    monkeypatch.setattr(httpx, "get", fake_get)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))
    out = buf.getvalue()

    assert "Alibaba/DashScope" in out
    assert "invalid API key" not in out
    assert any(
        url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models"
        for url, _, _ in calls
    )
    assert any(
        url == "https://dashscope.aliyuncs.com/compatible-mode/v1/models"
        for url, _, _ in calls
    )


@pytest.mark.parametrize("base_url", [None, "https://opencode.ai/zen/go/v1"])
def test_run_doctor_opencode_go_skips_invalid_models_probe(monkeypatch, tmp_path, base_url):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
    (home / ".env").write_text("OPENCODE_GO_API_KEY=***\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-test")
    if base_url:
        monkeypatch.setenv("OPENCODE_GO_BASE_URL", base_url)
    else:
        monkeypatch.delenv("OPENCODE_GO_BASE_URL", raising=False)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    try:
        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
    except ImportError:
        pass

    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append((url, headers, timeout))
        return types.SimpleNamespace(status_code=200)

    import httpx
    monkeypatch.setattr(httpx, "get", fake_get)

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))
    out = buf.getvalue()

    assert any(
        "OpenCode Go" in line and "(key configured)" in line
        for line in out.splitlines()
    )
    assert not any(url == "https://opencode.ai/zen/go/v1/models" for url, _, _ in calls)
    assert not any("opencode" in url.lower() and "models" in url.lower() for url, _, _ in calls)


class TestGitHubTokenCheck:
    """Tests for GitHub token / gh auth detection in doctor."""

    def test_no_token_and_not_gh_authenticated_shows_warn(self, monkeypatch, tmp_path):
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("PATH", "/nonexistent")  # gh not found

        from hermes_cli.doctor import run_doctor
        import io, contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_doctor(Namespace(fix=False))
        out = buf.getvalue()

        assert "No GITHUB_TOKEN" in out
        assert "60 req/hr" in out

    def test_token_env_present_shows_ok(self, monkeypatch, tmp_path):
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123")
        monkeypatch.setenv("PATH", "/nonexistent")  # gh not found

        from hermes_cli.doctor import run_doctor
        import io, contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_doctor(Namespace(fix=False))
        out = buf.getvalue()

        assert "GitHub token configured" in out

    def test_gh_authenticated_without_env_token_shows_ok(self, monkeypatch, tmp_path):
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        # No GITHUB_TOKEN or GH_TOKEN
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)

        # Mock gh to return success
        import shutil
        real_which = shutil.which
        def mock_which(cmd):
            return "/usr/local/bin/gh" if cmd == "gh" else real_which(cmd)
        monkeypatch.setattr(shutil, "which", mock_which)

        call_log = []
        def mock_run(cmd, **kwargs):
            call_log.append(cmd)
            if cmd[:2] == ["gh", "auth"]:
                result = types.SimpleNamespace(returncode=0, stdout="", stderr="")
            else:
                result = types.SimpleNamespace(returncode=1, stdout="", stderr="")
            return result

        import subprocess
        monkeypatch.setattr(subprocess, "run", mock_run)

        from hermes_cli.doctor import run_doctor
        import io, contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_doctor(Namespace(fix=False))
        out = buf.getvalue()

        assert "gh auth" in str(call_log) or any(c[0] == "gh" for c in call_log), f"gh not called: {call_log}"
        assert "GitHub authenticated via gh CLI" in out or "token configured" in out


def _run_doctor_with_healthy_oauth_fallback(
    monkeypatch,
    tmp_path,
    *,
    env_key: str,
    bad_key: str,
    failing_host: str,
    minimax_oauth_status: dict,
    xai_oauth_status: dict | None = None,
) -> str:
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "model:\n"
        "  provider: nous\n"
        "  default: moonshotai/kimi-k2.6\n",
        encoding="utf-8",
    )
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)

    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
    monkeypatch.setattr(doctor_mod, "_DHH", str(home))
    monkeypatch.setenv(env_key, bad_key)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_CN_API_KEY", raising=False)
    monkeypatch.setenv(env_key, bad_key)

    fake_model_tools = types.SimpleNamespace(
        check_tool_availability=lambda *a, **kw: ([], []),
        TOOLSET_REQUIREMENTS={},
    )
    monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

    from hermes_cli import auth as _auth_mod

    monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {"logged_in": True})
    monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
    monkeypatch.setattr(_auth_mod, "get_minimax_oauth_auth_status", lambda: minimax_oauth_status)
    _xai_status = xai_oauth_status if xai_oauth_status is not None else {}
    monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: _xai_status)

    def fake_get(url, headers=None, timeout=None):
        status = 401 if failing_host in url else 200
        return types.SimpleNamespace(status_code=status)

    import httpx

    monkeypatch.setattr(httpx, "get", fake_get)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))
    return buf.getvalue()


@pytest.mark.parametrize(
    ("env_key", "bad_key", "failing_host", "minimax_oauth_status", "xai_oauth_status", "unexpected_issue"),
    [
        (
            "MINIMAX_API_KEY",
            "bad-minimax-key",
            "minimax.io",
            {"logged_in": True, "region": "global"},
            None,
            "Check MINIMAX_API_KEY in .env",
        ),
        (
            "XAI_API_KEY",
            "bad-xai-key",
            "api.x.ai",
            {},
            {"logged_in": True, "auth_mode": "oauth_pkce"},
            "Check XAI_API_KEY in .env",
        ),
    ],
)
def test_run_doctor_ignores_invalid_direct_keys_when_oauth_fallback_is_healthy(
    monkeypatch,
    tmp_path,
    env_key,
    bad_key,
    failing_host,
    minimax_oauth_status,
    xai_oauth_status,
    unexpected_issue,
):
    out = _run_doctor_with_healthy_oauth_fallback(
        monkeypatch,
        tmp_path,
        env_key=env_key,
        bad_key=bad_key,
        failing_host=failing_host,
        minimax_oauth_status=minimax_oauth_status,
        xai_oauth_status=xai_oauth_status,
    )

    assert "invalid API key" in out
    assert unexpected_issue not in out


def test_has_healthy_oauth_fallback_returns_false_for_unknown_provider():
    from hermes_cli.doctor import _has_healthy_oauth_fallback_for_apikey_provider
    assert _has_healthy_oauth_fallback_for_apikey_provider("unknown-provider") is False


class TestHasHealthyOauthFallbackForXai:
    def test_returns_true_when_xai_oauth_healthy(self, monkeypatch):
        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {"logged_in": True})
        from hermes_cli.doctor import _has_healthy_oauth_fallback_for_apikey_provider
        assert _has_healthy_oauth_fallback_for_apikey_provider("xai") is True

    def test_returns_false_when_xai_oauth_not_logged_in(self, monkeypatch):
        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {"logged_in": False})
        from hermes_cli.doctor import _has_healthy_oauth_fallback_for_apikey_provider
        assert _has_healthy_oauth_fallback_for_apikey_provider("xai") is False

    def test_returns_false_when_xai_oauth_returns_none(self, monkeypatch):
        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: None)
        from hermes_cli.doctor import _has_healthy_oauth_fallback_for_apikey_provider
        assert _has_healthy_oauth_fallback_for_apikey_provider("xai") is False

    def test_returns_false_when_xai_import_unavailable(self, monkeypatch):
        import sys
        # Simulate get_xai_oauth_auth_status missing from auth module
        monkeypatch.delattr("hermes_cli.auth.get_xai_oauth_auth_status", raising=False)
        # Force doctor module to re-import the function
        monkeypatch.delitem(sys.modules, "hermes_cli.doctor", raising=False)
        from hermes_cli.doctor import _has_healthy_oauth_fallback_for_apikey_provider
        assert _has_healthy_oauth_fallback_for_apikey_provider("xai") is False


# ---------------------------------------------------------------------------
# ◆ Auth Providers — xAI OAuth display in run_doctor()
# ---------------------------------------------------------------------------


class TestDoctorXaiOAuthStatus:
    """The ◆ Auth Providers section must show xAI OAuth login state.

    xAI OAuth is checked in a *separate* try/except block so that an import
    failure (or runtime exception) cannot silence the Nous / Codex / Gemini /
    MiniMax rows that were already printed above it.
    """

    def _run(self, monkeypatch, tmp_path, *, xai_auth_fn) -> str:
        """Run doctor with a controlled xAI auth callable; return stdout."""
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
        project = tmp_path / "project"
        project.mkdir(exist_ok=True)

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
        monkeypatch.setattr(doctor_mod, "_DHH", str(home))

        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: ([], []),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {"logged_in": False})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {"logged_in": False})
        monkeypatch.setattr(_auth_mod, "get_minimax_oauth_auth_status", lambda: {"logged_in": False})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", xai_auth_fn)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=False))
        return buf.getvalue()

    def test_logged_in_shows_ok(self, monkeypatch, tmp_path):
        out = self._run(
            monkeypatch, tmp_path,
            xai_auth_fn=lambda: {"logged_in": True},
        )
        assert "xAI OAuth" in out
        assert "(logged in)" in out

    def test_not_logged_in_shows_warn(self, monkeypatch, tmp_path):
        out = self._run(
            monkeypatch, tmp_path,
            xai_auth_fn=lambda: {"logged_in": False},
        )
        assert "xAI OAuth" in out
        assert "(not logged in)" in out

    def test_error_shown_when_not_logged_in_and_error_present(self, monkeypatch, tmp_path):
        out = self._run(
            monkeypatch, tmp_path,
            xai_auth_fn=lambda: {"logged_in": False, "error": "refresh token expired"},
        )
        assert "xAI OAuth" in out
        assert "refresh token expired" in out

    def test_no_error_line_when_error_key_absent(self, monkeypatch, tmp_path):
        out = self._run(
            monkeypatch, tmp_path,
            xai_auth_fn=lambda: {"logged_in": False},
        )
        assert "xAI OAuth" in out
        # The check_info line is only emitted when the "error" key is present.
        # Pick a token that would appear in no ordinary doctor output.
        assert "refresh token expired" not in out

    def test_logged_in_does_not_emit_not_logged_in_on_xai_line(self, monkeypatch, tmp_path):
        out = self._run(
            monkeypatch, tmp_path,
            xai_auth_fn=lambda: {"logged_in": True},
        )
        assert "xAI OAuth" in out
        # The xAI OAuth line itself must say "(logged in)", not "(not logged in)".
        xai_line = next(l for l in out.splitlines() if "xAI OAuth" in l)
        assert "(logged in)" in xai_line
        assert "(not logged in)" not in xai_line

    def test_import_failure_does_not_crash_doctor(self, monkeypatch, tmp_path):
        """Doctor must not crash when get_xai_oauth_auth_status cannot be imported."""
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
        project = tmp_path / "project"
        project.mkdir(exist_ok=True)

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
        monkeypatch.setattr(doctor_mod, "_DHH", str(home))

        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: ([], []),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {"logged_in": False})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {"logged_in": False})
        monkeypatch.setattr(_auth_mod, "get_minimax_oauth_auth_status", lambda: {"logged_in": False})
        monkeypatch.delattr(_auth_mod, "get_xai_oauth_auth_status", raising=False)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=False))
        out = buf.getvalue()
        # The ◆ Auth Providers header must still appear — other providers unaffected.
        assert "Auth Providers" in out

    def test_import_failure_does_not_affect_other_providers(self, monkeypatch, tmp_path):
        """Nous / Codex / Gemini / MiniMax rows must survive an xAI import failure."""
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
        project = tmp_path / "project"
        project.mkdir(exist_ok=True)

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
        monkeypatch.setattr(doctor_mod, "_DHH", str(home))

        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: ([], []),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {"logged_in": True})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {"logged_in": False})
        monkeypatch.setattr(_auth_mod, "get_minimax_oauth_auth_status", lambda: {"logged_in": False})
        monkeypatch.delattr(_auth_mod, "get_xai_oauth_auth_status", raising=False)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=False))
        out = buf.getvalue()
        assert "Nous Portal auth" in out
        assert "logged in" in out

    def test_function_raises_does_not_crash_doctor(self, monkeypatch, tmp_path):
        """A runtime exception from get_xai_oauth_auth_status must be swallowed."""
        def _raise():
            raise RuntimeError("simulated xAI status failure")

        out = self._run(monkeypatch, tmp_path, xai_auth_fn=_raise)
        assert "Auth Providers" in out

    def test_function_returns_none_does_not_crash_doctor(self, monkeypatch, tmp_path):
        """None return is normalised to {} via `or {}` — must not AttributeError."""
        out = self._run(monkeypatch, tmp_path, xai_auth_fn=lambda: None)
        # None → {} → logged_in falsy → shows not-logged-in warn
        assert "xAI OAuth" in out
        assert "(not logged in)" in out


# ---------------------------------------------------------------------------
# ◆ Auth Providers — codex CLI import hint placement (issue #27975)
# ---------------------------------------------------------------------------


class TestDoctorCodexCliHintPlacement:
    """The `codex CLI not installed` hint belongs under OpenAI Codex auth.

    Regression for #27975: the hint used to be emitted as a standalone block
    after all auth-provider rows, so it visually attached to whichever
    provider happened to print last (MiniMax OAuth in the reported repro),
    reading as remediation for an unrelated provider.
    """

    def _run(self, monkeypatch, tmp_path, *, codex_logged_in: bool, codex_cli_present: bool) -> str:
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
        project = tmp_path / "project"
        project.mkdir(exist_ok=True)

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
        monkeypatch.setattr(doctor_mod, "_DHH", str(home))

        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: ([], []),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

        from hermes_cli import auth as _auth_mod
        monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {"logged_in": False})
        monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {"logged_in": codex_logged_in})
        monkeypatch.setattr(_auth_mod, "get_minimax_oauth_auth_status", lambda: {"logged_in": False})
        monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {"logged_in": False})

        real_which = doctor_mod.shutil.which
        monkeypatch.setattr(
            doctor_mod.shutil,
            "which",
            lambda cmd: ("/usr/local/bin/codex" if codex_cli_present else None) if cmd == "codex" else real_which(cmd),
        )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=False))
        return buf.getvalue()

    @staticmethod
    def _hint_line() -> str:
        return "codex CLI not installed"

    def test_hint_appears_under_codex_auth_when_missing(self, monkeypatch, tmp_path):
        out = self._run(monkeypatch, tmp_path, codex_logged_in=False, codex_cli_present=False)
        lines = out.splitlines()
        codex_idx = next(i for i, l in enumerate(lines) if "OpenAI Codex auth" in l)
        hint_idx = next(i for i, l in enumerate(lines) if self._hint_line() in l)
        minimax_idx = next(i for i, l in enumerate(lines) if "MiniMax OAuth" in l)
        # Hint must sit between Codex auth and the next provider row (#27975).
        assert codex_idx < hint_idx < minimax_idx

    def test_hint_suppressed_when_codex_cli_present(self, monkeypatch, tmp_path):
        out = self._run(monkeypatch, tmp_path, codex_logged_in=False, codex_cli_present=True)
        assert "OpenAI Codex auth" in out
        assert self._hint_line() not in out

    def test_hint_suppressed_when_codex_logged_in(self, monkeypatch, tmp_path):
        out = self._run(monkeypatch, tmp_path, codex_logged_in=True, codex_cli_present=False)
        assert "OpenAI Codex auth" in out
        assert "(logged in)" in out
        assert self._hint_line() not in out

    def test_hint_never_attaches_to_minimax_row(self, monkeypatch, tmp_path):
        out = self._run(monkeypatch, tmp_path, codex_logged_in=False, codex_cli_present=False)
        # The hint belongs to the Codex auth row that precedes it, never to the
        # MiniMax row that follows (#27975). The MiniMax row itself must not be
        # the hint line, and the hint must sit strictly above MiniMax.
        lines = [l for l in out.splitlines() if l.strip()]
        codex_idx = next(i for i, l in enumerate(lines) if "OpenAI Codex auth" in l)
        hint_idx = next(i for i, l in enumerate(lines) if self._hint_line() in l)
        minimax_idx = next(i for i, l in enumerate(lines) if "MiniMax OAuth" in l)
        # Hint sits under Codex and above MiniMax; the MiniMax row is not the hint.
        assert codex_idx < hint_idx < minimax_idx
        assert self._hint_line() not in lines[minimax_idx]


class TestDoctorStaleMaxIterationsDrift:
    """Regression for #17534: a stale HERMES_MAX_ITERATIONS in .env shadows
    agent.max_turns in config.yaml. The repro symptom is config.yaml saying
    400 while the gateway activity line reads N/90. Doctor must detect the
    drift, and `--fix` must remove the .env ghost (config.yaml wins).

    The detector reads the .env FILE directly, NOT os.environ — the gateway
    startup bridge can already have overridden os.environ to the config value,
    so the ghost is only visible in the file.
    """

    def _run_config_section(self, monkeypatch, tmp_path, *, fix, ghost, cfg_turns,
                            os_environ_value=None):
        import pathlib
        import contextlib
        import io
        from argparse import Namespace

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir(parents=True)
        (hermes_home / "config.yaml").write_text(
            f"agent:\n  max_turns: {cfg_turns}\n", encoding="utf-8"
        )
        env_lines = ["OPENAI_API_KEY=sk-test\n"]
        if ghost is not None:
            env_lines.append(f"HERMES_MAX_ITERATIONS={ghost}\n")
        (hermes_home / ".env").write_text("".join(env_lines), encoding="utf-8")

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", hermes_home)
        monkeypatch.setattr(doctor_mod, "get_hermes_home", lambda: hermes_home)
        # Point the config helpers at the temp home.
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        if os_environ_value is not None:
            # Simulate the gateway bridge having already overridden os.environ.
            monkeypatch.setenv("HERMES_MAX_ITERATIONS", str(os_environ_value))
        else:
            monkeypatch.delenv("HERMES_MAX_ITERATIONS", raising=False)

        # Short-circuit at the Tool Availability stage — the drift check runs
        # well before it in the Configuration Files section.
        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: (_ for _ in ()).throw(SystemExit(0)),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
            doctor_mod.run_doctor(Namespace(fix=fix))
        return buf.getvalue(), hermes_home

    def test_detects_drift_warn_only(self, monkeypatch, tmp_path):
        out, hermes_home = self._run_config_section(
            monkeypatch, tmp_path, fix=False, ghost=90, cfg_turns=400,
            os_environ_value=400,  # bridge contaminated os.environ
        )
        assert "HERMES_MAX_ITERATIONS=90" in out
        assert "shadows" in out
        # Warn-only must NOT mutate .env.
        assert "HERMES_MAX_ITERATIONS=90" in (hermes_home / ".env").read_text(encoding="utf-8")

    def test_fix_removes_ghost(self, monkeypatch, tmp_path):
        out, hermes_home = self._run_config_section(
            monkeypatch, tmp_path, fix=True, ghost=90, cfg_turns=400,
            os_environ_value=400,
        )
        assert "Removed stale HERMES_MAX_ITERATIONS" in out
        env_after = (hermes_home / ".env").read_text(encoding="utf-8")
        assert "HERMES_MAX_ITERATIONS" not in env_after
        assert "OPENAI_API_KEY=sk-test" in env_after  # other keys preserved

    def test_no_drift_when_values_match(self, monkeypatch, tmp_path):
        out, _ = self._run_config_section(
            monkeypatch, tmp_path, fix=False, ghost=400, cfg_turns=400,
        )
        assert "shadows" not in out

    def test_no_drift_when_ghost_absent(self, monkeypatch, tmp_path):
        out, _ = self._run_config_section(
            monkeypatch, tmp_path, fix=False, ghost=None, cfg_turns=400,
        )
        assert "shadows" not in out


def test_npm_audit_fix_hint_avoids_crashing_workspace_flag(monkeypatch, tmp_path):
    """`hermes doctor` must not hand users `npm audit fix --workspace <name>`:
    that exact form crashes npm with "Cannot read properties of null (reading
    'edgesOut')" (an arborist bug with workspace-filtered audit fix).

    It must not recommend root-level `npm audit fix` for workspace advisories
    either: current npm can crash there too with "Cannot read properties of null
    (reading 'isDescendantOf')" on this tree. The safe guidance is that these
    build-tool advisories clear via the lockfile/package bump.

    Regression for user reports where doctor flagged the web/ui-tui workspaces
    and the suggested fix command errored out.
    """
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    project = tmp_path / "project"
    (project / "node_modules").mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)

    # Only npm is "installed" — keeps the rest of run_doctor's external checks
    # quiet without affecting the npm-audit branch under test.
    monkeypatch.setattr(
        doctor_mod.shutil, "which", lambda cmd: "/usr/bin/npm" if cmd == "npm" else None
    )

    def mock_run(cmd, **kwargs):
        if "audit" in cmd and "--workspace" in cmd:
            payload = (
                '{"metadata": {"vulnerabilities": '
                '{"critical": 0, "high": 2, "moderate": 0}}}'
            )
            return SimpleNamespace(returncode=1, stdout=payload, stderr="")
        if "audit" in cmd:
            payload = (
                '{"metadata": {"vulnerabilities": '
                '{"critical": 0, "high": 0, "moderate": 0}}}'
            )
            return SimpleNamespace(returncode=0, stdout=payload, stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    import subprocess

    monkeypatch.setattr(subprocess, "run", mock_run)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doctor_mod.run_doctor(Namespace(fix=False))
    out = buf.getvalue()

    # The workspace vulnerability is still reported ...
    assert "web workspace" in out
    # ... but the remediation must NOT use the npm-crashing per-workspace form
    # (`npm audit fix --workspace web` / `--workspace ui-tui`).
    assert "npm audit fix --workspace web" not in out
    assert "npm audit fix --workspace ui-tui" not in out
    # ... and it must not point at the root-level form either: npm can crash
    # there too with `isDescendantOf` on this monorepo tree.
    assert "npm audit fix" not in out
    # ... and explains the workspace advisories are build-time tooling whose
    # manual remediation may hit a known npm arborist crash, so the user isn't
    # left thinking a crashing command means a broken Hermes install.
    assert "build-time tooling" in out
    assert "known npm bug" in out
    assert "lockfile bump" in out
