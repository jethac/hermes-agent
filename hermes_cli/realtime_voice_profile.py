"""Build or apply a portable realtime voice profile for Hermes."""

from __future__ import annotations

import argparse
import copy
import secrets
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_EVIDENCE_REPORT = "./artifacts/realtime-voice-evidence"
DEFAULT_STREAMING_STT_MODEL = "portable-streaming-asr"
DEFAULT_STREAMING_TTS_MODEL = "portable-streaming-voice"
DEFAULT_STREAMING_STT_TOKEN_ENV = "HERMES_STREAMING_STT_BRIDGE_TOKEN"
DEFAULT_STREAMING_TTS_TOKEN_ENV = "HERMES_STREAMING_TTS_BRIDGE_TOKEN"
DEFAULT_DEEPGRAM_BRIDGE_BASE_URL = "http://127.0.0.1:8766"
DEFAULT_DEEPGRAM_STT_MODEL = "nova-3"
DEFAULT_DEEPGRAM_TTS_MODEL = "aura-2-thalia-en"
DEFAULT_ELEVENLABS_BRIDGE_BASE_URL = "http://127.0.0.1:8767"
DEFAULT_ELEVENLABS_STT_MODEL = "scribe_v2_realtime"
DEFAULT_ELEVENLABS_TTS_MODEL = "eleven_flash_v2_5"
DEFAULT_CARTESIA_BRIDGE_BASE_URL = "http://127.0.0.1:8769"
DEFAULT_CARTESIA_STT_MODEL = "ink-2"
DEFAULT_CARTESIA_TTS_MODEL = "sonic-3.5"
DEFAULT_NVIDIA_SPEECH_STT_BRIDGE_BASE_URL = "http://127.0.0.1:8767"
DEFAULT_NVIDIA_SPEECH_TTS_BRIDGE_BASE_URL = "http://127.0.0.1:8768"
DEFAULT_NVIDIA_SPEECH_STT_MODEL = "nemotron-speech-streaming-0.6b"
DEFAULT_NVIDIA_SPEECH_TTS_MODEL = "magpie-local-streaming-tts"
DEFAULT_OPENAI_REALTIME_MODEL = "gpt-realtime-2"
DEFAULT_OPENAI_REALTIME_VOICE = "marin"
DEFAULT_OPENAI_REALTIME_TRANSCRIPTION_MODEL = "gpt-realtime-whisper"
DEFAULT_GEMINI_LIVE_MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_GEMINI_LIVE_VOICE = "Puck"
DEFAULT_KAME_REFLEX_MODEL = "gemma-4-E2B-it"
DEFAULT_KAME_ORACLE_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
DEFAULT_KAME_ORACLE_PROVIDER_NAME = "KAME Local Oracle"
DEFAULT_KAME_ORACLE_JOBS = {
    "enabled": True,
    "max_concurrent": 4,
    "queue_limit": 16,
    "default_priority": "normal",
    "overflow_policy": "queue",
    "shutdown_timeout_seconds": 2.0,
    "speak_terminal_results": True,
    "audit_ledger_path": "",
}
KAME_QUALITY_TARGETS_MS = {
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
}


def _parse_bool(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected one of true, false, 1, 0, yes, no, on, or off")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print or apply a portable Hermes realtime voice profile"
    )
    parser.add_argument(
        "--preset",
        choices=(
            "generic",
            "deepgram",
            "elevenlabs",
            "cartesia",
            "nvidia_speech",
            "openai",
            "gemini",
            "kame",
        ),
        default="generic",
        help="Provider preset for common portable realtime voice stacks",
    )
    parser.add_argument(
        "--bridge-base-url",
        default="",
        help="Shared bridge base URL used by provider presets, for example http://127.0.0.1:8766",
    )
    parser.add_argument(
        "--streaming-stt-base-url",
        default="",
        help="Compatible streaming STT bridge base URL, for example http://127.0.0.1:8766",
    )
    parser.add_argument(
        "--streaming-tts-base-url",
        default="",
        help="Compatible streaming TTS bridge base URL, for example http://127.0.0.1:8766",
    )
    parser.add_argument(
        "--streaming-stt-model",
        default=DEFAULT_STREAMING_STT_MODEL,
        help="Streaming STT model label for diagnostics",
    )
    parser.add_argument(
        "--streaming-tts-model",
        default=DEFAULT_STREAMING_TTS_MODEL,
        help="Streaming TTS model label for diagnostics",
    )
    parser.add_argument(
        "--streaming-tts-voice",
        default="",
        help="Streaming TTS voice identifier for diagnostics and KAME spoken output",
    )
    parser.add_argument(
        "--streaming-stt-token-env",
        default=DEFAULT_STREAMING_STT_TOKEN_ENV,
        help="Environment variable containing the streaming STT bridge bearer token",
    )
    parser.add_argument(
        "--streaming-tts-token-env",
        default=DEFAULT_STREAMING_TTS_TOKEN_ENV,
        help="Environment variable containing the streaming TTS bridge bearer token",
    )
    parser.add_argument(
        "--openai-realtime-model",
        default=DEFAULT_OPENAI_REALTIME_MODEL,
        help="OpenAI Realtime model for --preset openai",
    )
    parser.add_argument(
        "--openai-realtime-voice",
        default=DEFAULT_OPENAI_REALTIME_VOICE,
        help="OpenAI Realtime voice for --preset openai",
    )
    parser.add_argument(
        "--openai-realtime-transcription-model",
        default=DEFAULT_OPENAI_REALTIME_TRANSCRIPTION_MODEL,
        help="OpenAI Realtime transcription model for --preset openai",
    )
    parser.add_argument(
        "--openai-realtime-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the OpenAI Realtime API key",
    )
    parser.add_argument(
        "--gemini-live-model",
        default=DEFAULT_GEMINI_LIVE_MODEL,
        help="Gemini Live model for --preset gemini",
    )
    parser.add_argument(
        "--gemini-live-voice",
        default=DEFAULT_GEMINI_LIVE_VOICE,
        help="Gemini Live voice for --preset gemini",
    )
    parser.add_argument(
        "--gemini-live-api-key-env",
        default="GEMINI_API_KEY",
        help="Environment variable containing the Gemini Live API key",
    )
    parser.add_argument(
        "--gemini-live-google-search",
        action="store_true",
        help="Allow Gemini Live to use Google Search as a frontend context tool",
    )
    parser.add_argument(
        "--disable-gemini-live-oracle-tool",
        action="store_true",
        help="Disable Gemini Live's KAME ask_hermes_oracle bridge tool",
    )
    parser.add_argument(
        "--kame-reflex-model",
        default=DEFAULT_KAME_REFLEX_MODEL,
        help="Reflex/interface model for --preset kame",
    )
    parser.add_argument(
        "--kame-interface-provider",
        default="gemma4",
        help="Interface/reflex provider label for --preset kame",
    )
    parser.add_argument(
        "--kame-interface-audio-input",
        default="auto",
        choices=("auto", "native_audio", "text_fallback"),
        help="How the KAME reflex receives user input",
    )
    parser.add_argument(
        "--kame-interface-base-url",
        default="",
        help="OpenAI-compatible base URL for the KAME reflex/interface model",
    )
    parser.add_argument(
        "--kame-interface-api-key-env",
        default="HERMES_KAME_INTERFACE_API_KEY",
        help="Environment variable containing the KAME reflex/interface endpoint bearer token",
    )
    parser.add_argument(
        "--kame-interface-max-audio-seconds",
        type=float,
        default=30.0,
        help="Maximum native-audio segment seconds sent to the KAME reflex model",
    )
    parser.add_argument(
        "--kame-interface-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the KAME reflex/interface model",
    )
    parser.add_argument(
        "--kame-interface-max-output-tokens",
        type=int,
        default=160,
        help="Maximum text tokens emitted by the KAME reflex/interface model",
    )
    parser.add_argument(
        "--kame-interface-timeout-seconds",
        type=float,
        default=0.8,
        help="Timeout for one KAME reflex/interface request",
    )
    parser.add_argument(
        "--kame-asr-mode",
        default="on_escalation",
        choices=("disabled", "from_reflex", "on_escalation", "speculative", "debug", "fallback"),
        help="ASR role for --preset kame",
    )
    parser.add_argument(
        "--kame-asr-provider",
        default="streaming_stt",
        help="ASR provider label for --preset kame oracle-verbatim evidence",
    )
    parser.add_argument(
        "--kame-preferred-local-oracle-model",
        default=DEFAULT_KAME_ORACLE_MODEL,
        help="Preferred local oracle provider target label for --preset kame; Hermes still chooses the active oracle through /model",
    )
    parser.add_argument(
        "--kame-oracle-provider",
        default="",
        help="Provider id to register a local KAME oracle endpoint; empty keeps Hermes' active oracle provider set unchanged",
    )
    parser.add_argument(
        "--kame-oracle-api-mode",
        default="chat_completions",
        choices=("chat_completions", "anthropic_messages", "codex_responses"),
        help="API contract used by the registered local KAME oracle endpoint",
    )
    parser.add_argument(
        "--kame-oracle-timeout-seconds",
        type=float,
        default=60.0,
        help="Timeout for escalated Hermes oracle requests in KAME mode",
    )
    parser.add_argument(
        "--kame-max-spoken-sentences",
        type=int,
        default=2,
        help="Maximum spoken sentences for KAME oracle responses",
    )
    parser.add_argument(
        "--kame-voice-response-policy",
        default="sentence_cap",
        choices=("sentence_cap", "brief_summary", "full"),
        help="How KAME should shape spoken oracle responses",
    )
    parser.add_argument(
        "--kame-tts-provider",
        default="streaming_tts",
        help="TTS provider label for --preset kame spoken output",
    )
    parser.add_argument(
        "--kame-fallback-policy",
        default="legacy_voice",
        choices=("legacy_voice", "text_only", "fail_closed"),
        help="How KAME should degrade when the realtime sidecar or reflex path is unavailable",
    )
    parser.add_argument(
        "--kame-allow-local-greetings",
        type=_parse_bool,
        default=True,
        help="Whether KAME may answer simple greetings locally",
    )
    parser.add_argument(
        "--kame-allow-local-clarifications",
        type=_parse_bool,
        default=True,
        help="Whether KAME may ask clarification prompts locally",
    )
    parser.add_argument(
        "--kame-require-oracle-for-tools",
        type=_parse_bool,
        default=True,
        help="Whether tool-related turns must escalate to Hermes' oracle",
    )
    parser.add_argument(
        "--kame-require-oracle-for-memory",
        type=_parse_bool,
        default=True,
        help="Whether memory-related turns must escalate to Hermes' oracle",
    )
    parser.add_argument(
        "--kame-require-oracle-for-files",
        type=_parse_bool,
        default=True,
        help="Whether file/project turns must escalate to Hermes' oracle",
    )
    parser.add_argument(
        "--kame-local-confidence-threshold",
        type=float,
        default=0.75,
        help="Minimum reflex confidence required for local KAME replies",
    )
    parser.add_argument(
        "--kame-barge-in-min-rms",
        type=int,
        default=350,
        help="Minimum 16-bit PCM RMS amplitude required before KAME barge-in",
    )
    parser.add_argument(
        "--kame-barge-in-min-speech-ms",
        type=int,
        default=120,
        help="Milliseconds of sustained speech required before KAME barge-in",
    )
    parser.add_argument(
        "--kame-barge-in-stop-playback-deadline-ms",
        type=int,
        default=150,
        help="Target milliseconds to stop playback after confirmed KAME barge-in",
    )
    parser.add_argument(
        "--kame-metrics-enabled",
        type=_parse_bool,
        default=True,
        help="Whether KAME realtime turn/provider metrics are enabled",
    )
    parser.add_argument(
        "--kame-log-turn-spans",
        type=_parse_bool,
        default=True,
        help="Whether KAME logs per-turn latency spans",
    )
    parser.add_argument(
        "--kame-log-provider-spans",
        type=_parse_bool,
        default=True,
        help="Whether KAME logs provider latency spans",
    )
    parser.add_argument(
        "--disable-kame-metrics",
        action="store_true",
        help="Disable KAME realtime turn/provider span metrics in the generated profile",
    )
    parser.add_argument(
        "--disable-kame-turn-span-logs",
        action="store_true",
        help="Disable KAME per-turn latency span logging in the generated profile",
    )
    parser.add_argument(
        "--disable-kame-provider-span-logs",
        action="store_true",
        help="Disable KAME provider latency span logging in the generated profile",
    )
    parser.add_argument(
        "--kame-oracle-base-url",
        default="",
        help=(
            "OpenAI-compatible base URL for the local Hermes oracle model. "
            "When set with --preset kame --apply, Hermes' main model provider is pointed at this endpoint."
        ),
    )
    parser.add_argument(
        "--kame-oracle-provider-name",
        default=DEFAULT_KAME_ORACLE_PROVIDER_NAME,
        help="Display name to register for the local KAME oracle custom provider",
    )
    parser.add_argument("--sidecar-host", default="127.0.0.1")
    parser.add_argument("--sidecar-port", type=int, default=8765)
    parser.add_argument(
        "--production-evidence-report",
        default=DEFAULT_EVIDENCE_REPORT,
        help="Report file or directory to use once alpha evidence has been collected",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the profile into ~/.hermes/config.yaml instead of printing YAML",
    )
    parser.add_argument(
        "--generate-bridge-token",
        action="store_true",
        help="With --apply, generate the shared streaming bridge bearer token in ~/.hermes/.env",
    )
    parser.add_argument(
        "--force-bridge-token",
        action="store_true",
        help="With --generate-bridge-token, replace an existing bridge bearer token",
    )
    parser.add_argument(
        "--allow-template-urls",
        action="store_true",
        help="Allow placeholder streaming bridge URLs when printing a template",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preset = _profile_preset_values(args)
    try:
        if args.preset == "openai":
            profile = build_openai_realtime_voice_profile(
                model=preset["openai_realtime_model"],
                voice=preset["openai_realtime_voice"],
                transcription_model=preset["openai_realtime_transcription_model"],
                api_key_env=args.openai_realtime_api_key_env,
                sidecar_host=args.sidecar_host,
                sidecar_port=args.sidecar_port,
                production_evidence_report=args.production_evidence_report,
            )
        elif args.preset == "gemini":
            profile = build_gemini_live_voice_profile(
                model=preset["gemini_live_model"],
                voice=preset["gemini_live_voice"],
                api_key_env=args.gemini_live_api_key_env,
                google_search=bool(args.gemini_live_google_search),
                oracle_tool=not bool(args.disable_gemini_live_oracle_tool),
                sidecar_host=args.sidecar_host,
                sidecar_port=args.sidecar_port,
                production_evidence_report=args.production_evidence_report,
            )
        elif args.preset == "kame":
            profile = build_kame_realtime_voice_profile(
                reflex_provider=str(args.kame_interface_provider or "gemma4"),
                reflex_model=str(args.kame_reflex_model or DEFAULT_KAME_REFLEX_MODEL),
                interface_base_url=str(args.kame_interface_base_url or ""),
                interface_api_key_env=str(args.kame_interface_api_key_env or "HERMES_KAME_INTERFACE_API_KEY"),
                interface_audio_input=str(args.kame_interface_audio_input or "auto"),
                interface_max_audio_seconds=float(args.kame_interface_max_audio_seconds or 30.0),
                interface_temperature=float(args.kame_interface_temperature or 0.2),
                interface_max_output_tokens=int(args.kame_interface_max_output_tokens or 160),
                interface_timeout_seconds=float(args.kame_interface_timeout_seconds or 0.8),
                asr_mode=str(args.kame_asr_mode or "on_escalation"),
                asr_provider=str(args.kame_asr_provider or "streaming_stt"),
                preferred_local_oracle_model=str(
                    args.kame_preferred_local_oracle_model or DEFAULT_KAME_ORACLE_MODEL
                ),
                oracle_provider=str(args.kame_oracle_provider or ""),
                oracle_api_mode=str(args.kame_oracle_api_mode or "chat_completions"),
                oracle_timeout_seconds=float(args.kame_oracle_timeout_seconds or 60.0),
                max_spoken_sentences=int(args.kame_max_spoken_sentences or 2),
                voice_response_policy=str(args.kame_voice_response_policy or "sentence_cap"),
                tts_provider=str(args.kame_tts_provider or "streaming_tts"),
                fallback_policy=str(args.kame_fallback_policy or "legacy_voice"),
                oracle_base_url=str(args.kame_oracle_base_url or ""),
                oracle_provider_name=str(
                    args.kame_oracle_provider_name or DEFAULT_KAME_ORACLE_PROVIDER_NAME
                ),
                streaming_stt_base_url=preset["streaming_stt_base_url"],
                streaming_tts_base_url=preset["streaming_tts_base_url"],
                streaming_stt_model=preset["streaming_stt_model"],
                streaming_tts_model=preset["streaming_tts_model"],
                streaming_tts_voice=str(args.streaming_tts_voice or ""),
                streaming_stt_token_env=args.streaming_stt_token_env,
                streaming_tts_token_env=args.streaming_tts_token_env,
                barge_in_min_rms=int(args.kame_barge_in_min_rms or 350),
                barge_in_min_speech_ms=int(args.kame_barge_in_min_speech_ms or 120),
                barge_in_stop_playback_deadline_ms=int(
                    args.kame_barge_in_stop_playback_deadline_ms or 150
                ),
                allow_local_greetings=bool(args.kame_allow_local_greetings),
                allow_local_clarifications=bool(args.kame_allow_local_clarifications),
                require_oracle_for_tools=bool(args.kame_require_oracle_for_tools),
                require_oracle_for_memory=bool(args.kame_require_oracle_for_memory),
                require_oracle_for_files=bool(args.kame_require_oracle_for_files),
                local_confidence_threshold=float(args.kame_local_confidence_threshold or 0.75),
                metrics_enabled=bool(args.kame_metrics_enabled) and not bool(args.disable_kame_metrics),
                metrics_log_turn_spans=bool(args.kame_log_turn_spans)
                and not bool(args.disable_kame_turn_span_logs),
                metrics_log_provider_spans=bool(args.kame_log_provider_spans)
                and not bool(args.disable_kame_provider_span_logs),
                sidecar_host=args.sidecar_host,
                sidecar_port=args.sidecar_port,
                production_evidence_report=args.production_evidence_report,
            )
        else:
            profile = build_realtime_voice_live_like_profile(
                streaming_stt_base_url=preset["streaming_stt_base_url"],
                streaming_tts_base_url=preset["streaming_tts_base_url"],
                streaming_stt_model=preset["streaming_stt_model"],
                streaming_tts_model=preset["streaming_tts_model"],
                streaming_tts_voice=str(args.streaming_tts_voice or ""),
                streaming_stt_token_env=args.streaming_stt_token_env,
                streaming_tts_token_env=args.streaming_tts_token_env,
                sidecar_host=args.sidecar_host,
                sidecar_port=args.sidecar_port,
                production_evidence_report=args.production_evidence_report,
                allow_template_urls=bool(args.allow_template_urls and not args.apply),
            )
    except ValueError as exc:
        print(f"Realtime voice profile failed: {exc}", file=sys.stderr)
        return 1

    if args.apply:
        config_path = apply_realtime_voice_profile(profile)
        print(f"Updated realtime voice profile in {config_path}")
        if args.generate_bridge_token:
            bridge_token_envs = _realtime_voice_bridge_token_envs(profile)
            primary_bridge_token_env = bridge_token_envs[0] if bridge_token_envs else ""
            token_result = ensure_realtime_voice_bridge_tokens(
                bridge_token_envs,
                force=bool(args.force_bridge_token),
            )
            if args.preset == "deepgram":
                ensure_deepgram_bridge_token_env(primary_bridge_token_env)
            if args.preset == "cartesia":
                ensure_cartesia_bridge_token_env(primary_bridge_token_env)
            if args.preset == "nvidia_speech":
                ensure_nvidia_speech_bridge_token_envs(bridge_token_envs)
            if token_result == "created":
                print(f"Generated realtime voice bridge token in {_format_env_list(bridge_token_envs)}")
            elif token_result == "existing":
                print(f"Realtime voice bridge token already configured in {_format_env_list(bridge_token_envs)}")
        print("Next:")
        if args.preset == "deepgram":
            if not args.generate_bridge_token:
                print("  python -m hermes_cli.realtime_voice_deepgram_bridge --generate-token")
            print("  python -m hermes_cli.realtime_voice_deepgram_bridge --check --strict --production-en-ja")
            print(
                "  python -m hermes_cli.realtime_voice_deepgram_bridge "
                "--host 127.0.0.1 --port 8766 --production-en-ja"
            )
            print("  python -m hermes_cli.realtime_voice_alpha_evidence --runs 3 --apply --start-deepgram-bridge")
        elif args.preset == "elevenlabs":
            if not args.generate_bridge_token:
                print("  python -m hermes_cli.realtime_voice_elevenlabs_bridge --generate-token")
            print("  python -m hermes_cli.realtime_voice_elevenlabs_bridge --check --strict --production-en-ja")
            print(
                "  python -m hermes_cli.realtime_voice_elevenlabs_bridge "
                "--host 127.0.0.1 --port 8767 --production-en-ja"
            )
            print("  python -m hermes_cli.realtime_voice_alpha_evidence --runs 3 --apply --provider elevenlabs --start-bridge")
        elif args.preset == "cartesia":
            if not args.generate_bridge_token:
                print("  python -m hermes_cli.realtime_voice_cartesia_bridge --generate-token")
            print("  python -m hermes_cli.realtime_voice_cartesia_bridge --check --strict --production-en-ja")
            print(
                "  python -m hermes_cli.realtime_voice_cartesia_bridge "
                "--host 127.0.0.1 --port 8769 --production-en-ja"
            )
            print("  python -m hermes_cli.realtime_voice_alpha_evidence --runs 3 --apply --provider cartesia --start-bridge")
        elif args.preset == "nvidia_speech":
            if not args.generate_bridge_token:
                print("  python -m hermes_cli.realtime_voice_profile --preset nvidia_speech --apply --generate-bridge-token")
            print("  export HERMES_NEMOTRON_SPEECH_UPSTREAM_BASE_URL=http://<spark-host>:<asr-port>")
            print("  export HERMES_MAGPIE_TTS_UPSTREAM_BASE_URL=http://<spark-host>:<tts-port>")
            print("  python -m hermes_cli.realtime_voice_nemotron_speech_bridge --check --strict --production-en-ja")
            print("  python -m hermes_cli.realtime_voice_magpie_tts_bridge --check --strict --production-en-ja")
            print(
                "  python -m hermes_cli.realtime_voice_nemotron_speech_bridge "
                "--host 127.0.0.1 --port 8767 --production-en-ja"
            )
            print(
                "  python -m hermes_cli.realtime_voice_magpie_tts_bridge "
                "--host 127.0.0.1 --port 8768 --production-en-ja"
            )
            print("  python -m hermes_cli.realtime_voice_alpha_evidence --runs 3 --apply --provider local_speech --start-bridge")
        elif args.preset == "openai":
            print("  export OPENAI_API_KEY=...")
            print("  export DISCORD_BOT_TOKEN=... DISCORD_GUILD_ID=... DISCORD_VOICE_CHANNEL_ID=...")
            print("  python -m hermes_cli.realtime_voice_sidecar --host 127.0.0.1 --port 8765")
            print("  python -m hermes_cli.realtime_voice_alpha_evidence --runs 3 --apply")
        elif args.preset == "gemini":
            print("  export GEMINI_API_KEY=...")
            print("  export DISCORD_BOT_TOKEN=... DISCORD_GUILD_ID=... DISCORD_VOICE_CHANNEL_ID=...")
            print("  python -m hermes_cli.realtime_voice_sidecar --host 127.0.0.1 --port 8765")
        elif args.preset == "kame":
            print("  start the Gemma 4 E2B reflex runtime behind the realtime sidecar")
            print("  python -m hermes_cli.realtime_voice_sidecar --host 127.0.0.1 --port 8765")
            print("  python -m hermes_cli.realtime_voice_alpha_evidence --runs 3 --apply")
        else:
            print("  python -m hermes_cli.realtime_voice_alpha_evidence --runs 3 --apply --start-bridge")
        live_provider_flag = "--require-gemini-live" if args.preset == "gemini" else "--require-openai-realtime"
        print(
            "  python -m hermes_cli.realtime_voice_live_evidence "
            f"--require-live-discord {live_provider_flag}"
        )
        print("  python -m hermes_cli.realtime_voice_fixture_pack --output-dir ./fixtures/realtime-voice")
        return 0

    print(yaml.safe_dump({"voice": {"realtime": profile}}, sort_keys=False, allow_unicode=True).rstrip())
    return 0


def _realtime_voice_bridge_token_envs(profile: Mapping[str, Any]) -> list[str]:
    envs: list[str] = []
    for key in ("streaming_stt_token_env", "streaming_tts_token_env"):
        env_name = _clean_env_name(str(profile.get(key) or ""))
        if env_name and env_name not in envs:
            envs.append(env_name)
    if not envs:
        raise ValueError("streaming bridge token env is required")
    return envs


def _format_env_list(env_names: list[str]) -> str:
    return ", ".join(env_names)


def ensure_realtime_voice_bridge_tokens(token_envs: list[str], *, force: bool = False) -> str:
    env_names = [_clean_env_name(token_env) for token_env in token_envs]
    env_names = [env_name for env_name in dict.fromkeys(env_names) if env_name]
    if not env_names:
        raise ValueError("streaming bridge token env is required")
    from hermes_cli.config import load_env, save_env_value

    env = load_env()
    existing = [str(env.get(env_name) or "") for env_name in env_names]
    if all(existing) and not force:
        return "existing"
    token = next((value for value in existing if value and not force), "") or secrets.token_urlsafe(32)
    for env_name in env_names:
        if force or not str(env.get(env_name) or ""):
            save_env_value(env_name, token)
    return "created"


def ensure_realtime_voice_bridge_token(token_env: str, *, force: bool = False) -> str:
    return ensure_realtime_voice_bridge_tokens([token_env], force=force)


def ensure_deepgram_bridge_token_env(token_env: str) -> None:
    env_name = _clean_env_name(token_env)
    if not env_name or env_name == DEFAULT_STREAMING_STT_TOKEN_ENV:
        return

    from hermes_cli.config import save_env_value

    save_env_value("HERMES_DEEPGRAM_BRIDGE_TOKEN_ENV", env_name)


def ensure_cartesia_bridge_token_env(token_env: str) -> None:
    env_name = _clean_env_name(token_env)
    if not env_name or env_name in {DEFAULT_STREAMING_STT_TOKEN_ENV, DEFAULT_STREAMING_TTS_TOKEN_ENV}:
        return

    from hermes_cli.config import save_env_value

    save_env_value("HERMES_CARTESIA_BRIDGE_TOKEN_ENV", env_name)


def ensure_nvidia_speech_bridge_token_envs(token_envs: list[str]) -> None:
    env_names = [_clean_env_name(token_env) for token_env in token_envs]
    env_names = [env_name for env_name in dict.fromkeys(env_names) if env_name]
    if not env_names:
        return
    stt_env = env_names[0]
    tts_env = env_names[-1]

    from hermes_cli.config import load_env, save_env_value

    env_on_disk = load_env()
    if not str(env_on_disk.get("HERMES_NEMOTRON_SPEECH_BRIDGE_TOKEN") or ""):
        save_env_value("HERMES_NEMOTRON_SPEECH_BRIDGE_TOKEN", f"${{{stt_env}}}")
    if not str(env_on_disk.get("HERMES_MAGPIE_TTS_BRIDGE_TOKEN") or ""):
        save_env_value("HERMES_MAGPIE_TTS_BRIDGE_TOKEN", f"${{{tts_env}}}")


def _profile_preset_values(args: argparse.Namespace) -> dict[str, str]:
    streaming_stt_base_url = str(args.streaming_stt_base_url or "")
    streaming_tts_base_url = str(args.streaming_tts_base_url or "")
    streaming_stt_model = str(args.streaming_stt_model or DEFAULT_STREAMING_STT_MODEL)
    streaming_tts_model = str(args.streaming_tts_model or DEFAULT_STREAMING_TTS_MODEL)

    if args.preset == "generic":
        return {
            "streaming_stt_base_url": streaming_stt_base_url,
            "streaming_tts_base_url": streaming_tts_base_url,
            "streaming_stt_model": streaming_stt_model,
            "streaming_tts_model": streaming_tts_model,
        }

    if args.preset == "openai":
        return {
            "streaming_stt_base_url": streaming_stt_base_url,
            "streaming_tts_base_url": streaming_tts_base_url,
            "streaming_stt_model": streaming_stt_model,
            "streaming_tts_model": streaming_tts_model,
            "openai_realtime_model": str(args.openai_realtime_model or DEFAULT_OPENAI_REALTIME_MODEL),
            "openai_realtime_voice": str(args.openai_realtime_voice or DEFAULT_OPENAI_REALTIME_VOICE),
            "openai_realtime_transcription_model": str(
                args.openai_realtime_transcription_model or DEFAULT_OPENAI_REALTIME_TRANSCRIPTION_MODEL
            ),
        }
    if args.preset == "gemini":
        return {
            "streaming_stt_base_url": streaming_stt_base_url,
            "streaming_tts_base_url": streaming_tts_base_url,
            "streaming_stt_model": streaming_stt_model,
            "streaming_tts_model": streaming_tts_model,
            "gemini_live_model": str(args.gemini_live_model or DEFAULT_GEMINI_LIVE_MODEL),
            "gemini_live_voice": str(args.gemini_live_voice or DEFAULT_GEMINI_LIVE_VOICE),
        }

    if args.preset == "kame":
        return {
            "streaming_stt_base_url": streaming_stt_base_url,
            "streaming_tts_base_url": streaming_tts_base_url,
            "streaming_stt_model": streaming_stt_model,
            "streaming_tts_model": streaming_tts_model,
        }

    if args.preset == "elevenlabs":
        bridge_base_url = _clean_url(str(args.bridge_base_url or DEFAULT_ELEVENLABS_BRIDGE_BASE_URL))
        return {
            "streaming_stt_base_url": streaming_stt_base_url or bridge_base_url,
            "streaming_tts_base_url": streaming_tts_base_url or streaming_stt_base_url or bridge_base_url,
            "streaming_stt_model": (
                DEFAULT_ELEVENLABS_STT_MODEL
                if streaming_stt_model == DEFAULT_STREAMING_STT_MODEL
                else streaming_stt_model
            ),
            "streaming_tts_model": (
                DEFAULT_ELEVENLABS_TTS_MODEL
                if streaming_tts_model == DEFAULT_STREAMING_TTS_MODEL
                else streaming_tts_model
            ),
        }

    if args.preset == "cartesia":
        bridge_base_url = _clean_url(str(args.bridge_base_url or DEFAULT_CARTESIA_BRIDGE_BASE_URL))
        return {
            "streaming_stt_base_url": streaming_stt_base_url or bridge_base_url,
            "streaming_tts_base_url": streaming_tts_base_url or streaming_stt_base_url or bridge_base_url,
            "streaming_stt_model": (
                DEFAULT_CARTESIA_STT_MODEL
                if streaming_stt_model == DEFAULT_STREAMING_STT_MODEL
                else streaming_stt_model
            ),
            "streaming_tts_model": (
                DEFAULT_CARTESIA_TTS_MODEL
                if streaming_tts_model == DEFAULT_STREAMING_TTS_MODEL
                else streaming_tts_model
            ),
        }

    if args.preset == "nvidia_speech":
        shared_bridge_base_url = _clean_url(str(args.bridge_base_url or ""))
        return {
            "streaming_stt_base_url": (
                streaming_stt_base_url
                or shared_bridge_base_url
                or DEFAULT_NVIDIA_SPEECH_STT_BRIDGE_BASE_URL
            ),
            "streaming_tts_base_url": (
                streaming_tts_base_url
                or shared_bridge_base_url
                or DEFAULT_NVIDIA_SPEECH_TTS_BRIDGE_BASE_URL
            ),
            "streaming_stt_model": (
                DEFAULT_NVIDIA_SPEECH_STT_MODEL
                if streaming_stt_model == DEFAULT_STREAMING_STT_MODEL
                else streaming_stt_model
            ),
            "streaming_tts_model": (
                DEFAULT_NVIDIA_SPEECH_TTS_MODEL
                if streaming_tts_model == DEFAULT_STREAMING_TTS_MODEL
                else streaming_tts_model
            ),
        }

    bridge_base_url = _clean_url(str(args.bridge_base_url or DEFAULT_DEEPGRAM_BRIDGE_BASE_URL))
    return {
        "streaming_stt_base_url": streaming_stt_base_url or bridge_base_url,
        "streaming_tts_base_url": streaming_tts_base_url or streaming_stt_base_url or bridge_base_url,
        "streaming_stt_model": (
            DEFAULT_DEEPGRAM_STT_MODEL
            if streaming_stt_model == DEFAULT_STREAMING_STT_MODEL
            else streaming_stt_model
        ),
        "streaming_tts_model": (
            DEFAULT_DEEPGRAM_TTS_MODEL
            if streaming_tts_model == DEFAULT_STREAMING_TTS_MODEL
            else streaming_tts_model
        ),
    }


def build_realtime_voice_live_like_profile(
    *,
    streaming_stt_base_url: str = "",
    streaming_tts_base_url: str = "",
    streaming_stt_model: str = DEFAULT_STREAMING_STT_MODEL,
    streaming_tts_model: str = DEFAULT_STREAMING_TTS_MODEL,
    streaming_tts_voice: str = "",
    streaming_stt_token_env: str = DEFAULT_STREAMING_STT_TOKEN_ENV,
    streaming_tts_token_env: str = DEFAULT_STREAMING_TTS_TOKEN_ENV,
    sidecar_host: str = "127.0.0.1",
    sidecar_port: int = 8765,
    production_evidence_report: str = DEFAULT_EVIDENCE_REPORT,
    allow_template_urls: bool = False,
) -> dict[str, Any]:
    stt_url = _clean_url(streaming_stt_base_url)
    tts_url = _clean_url(streaming_tts_base_url)
    if allow_template_urls:
        stt_url = stt_url or "http://127.0.0.1:8766"
        tts_url = tts_url or stt_url
    if not stt_url:
        raise ValueError("--streaming-stt-base-url is required for a live-like profile")
    if not tts_url:
        raise ValueError("--streaming-tts-base-url is required for a live-like profile")

    port = int(sidecar_port or 8765)
    if port <= 0 or port > 65535:
        raise ValueError("--sidecar-port must be between 1 and 65535")

    return {
        "enabled": True,
        "engine": "text_oracle_tts",
        "input_codec": "webm_opus",
        "output_codec": "opus",
        "input_buffer_limit_bytes": 8 * 1024 * 1024,
        "input_frame_ms": 100,
        "silence_timeout_ms": 650,
        "speech_level_threshold": 0.075,
        "barge_in_min_speech_ms": 120,
        "barge_in_min_rms": 350,
        "barge_in_stop_playback_deadline_ms": 150,
        "pre_roll_ms": 300,
        "require_live_like": True,
        "frontend_provider": "reference",
        "frontend_model": str(streaming_stt_model or DEFAULT_STREAMING_STT_MODEL),
        "interface_temperature": 0.2,
        "interface_max_output_tokens": 160,
        "interface_timeout_seconds": 0.8,
        "interface_max_audio_seconds": 30.0,
        "sidecar_base_url": "",
        "spark_base_url": "",
        "sidecar_host": str(sidecar_host or "127.0.0.1"),
        "sidecar_port": port,
        "sidecar_autostart": True,
        "sidecar_connect_timeout_seconds": 10.0,
        "vllm_base_url": "",
        "vllm_model": "",
        "asr_base_url": stt_url,
        "streaming_stt_base_url": stt_url,
        "streaming_stt_model": str(streaming_stt_model or DEFAULT_STREAMING_STT_MODEL),
        "streaming_stt_token_env": _clean_env_name(streaming_stt_token_env),
        "tts_base_url": tts_url,
        "streaming_tts_base_url": tts_url,
        "streaming_tts_model": str(streaming_tts_model or DEFAULT_STREAMING_TTS_MODEL),
        "streaming_tts_voice": str(streaming_tts_voice or ""),
        "streaming_tts_token_env": _clean_env_name(streaming_tts_token_env),
        "production_languages": ["en", "ja"],
        "production_scripts": ["Latn", "Jpan"],
        "best_effort_languages": True,
        "production_evidence_report": str(production_evidence_report or DEFAULT_EVIDENCE_REPORT),
        "production_evidence_min_runs": 3,
        "turn_acknowledgement": {
            "enabled": True,
            "text": "One moment.",
        },
        "routing": {
            "allow_local_greetings": True,
            "allow_local_clarifications": True,
            "require_oracle_for_tools": True,
            "require_oracle_for_memory": True,
            "require_oracle_for_files": True,
            "local_confidence_threshold": 0.75,
        },
        "metrics": {
            "enabled": True,
            "log_turn_spans": True,
            "log_provider_spans": True,
        },
        "quality_targets_ms": {
            "audio_to_partial_transcript_ms": 300,
            "final_transcript_to_first_text_ms": 500,
            "final_transcript_to_first_audio_ms": 900,
            "barge_in_ack_ms": 150,
        },
    }


def build_openai_realtime_voice_profile(
    *,
    model: str = DEFAULT_OPENAI_REALTIME_MODEL,
    voice: str = DEFAULT_OPENAI_REALTIME_VOICE,
    transcription_model: str = DEFAULT_OPENAI_REALTIME_TRANSCRIPTION_MODEL,
    api_key_env: str = "OPENAI_API_KEY",
    sidecar_host: str = "127.0.0.1",
    sidecar_port: int = 8765,
    production_evidence_report: str = DEFAULT_EVIDENCE_REPORT,
) -> dict[str, Any]:
    port = int(sidecar_port or 8765)
    if port <= 0 or port > 65535:
        raise ValueError("--sidecar-port must be between 1 and 65535")
    return {
        "enabled": True,
        "engine": "text_oracle_tts",
        "input_codec": "webm_opus",
        "output_codec": "opus",
        "input_buffer_limit_bytes": 8 * 1024 * 1024,
        "input_frame_ms": 100,
        "silence_timeout_ms": 650,
        "speech_level_threshold": 0.075,
        "barge_in_min_speech_ms": 120,
        "barge_in_min_rms": 350,
        "barge_in_stop_playback_deadline_ms": 150,
        "pre_roll_ms": 300,
        "require_live_like": True,
        "frontend_provider": "openai_realtime",
        "frontend_model": str(model or DEFAULT_OPENAI_REALTIME_MODEL),
        "interface_temperature": 0.2,
        "interface_max_output_tokens": 160,
        "interface_timeout_seconds": 0.8,
        "sidecar_base_url": "",
        "spark_base_url": "",
        "sidecar_host": str(sidecar_host or "127.0.0.1"),
        "sidecar_port": port,
        "sidecar_autostart": True,
        "sidecar_connect_timeout_seconds": 10.0,
        "openai_realtime_api_key_env": _clean_env_name(api_key_env) or "OPENAI_API_KEY",
        "openai_realtime_base_url": "wss://api.openai.com/v1/realtime",
        "openai_realtime_voice": str(voice or DEFAULT_OPENAI_REALTIME_VOICE),
        "openai_realtime_transcription_model": str(
            transcription_model or DEFAULT_OPENAI_REALTIME_TRANSCRIPTION_MODEL
        ),
        "production_languages": ["en", "ja"],
        "production_scripts": ["Latn", "Jpan"],
        "best_effort_languages": True,
        "production_evidence_report": str(production_evidence_report or DEFAULT_EVIDENCE_REPORT),
        "production_evidence_min_runs": 3,
        "turn_acknowledgement": {
            "enabled": True,
            "text": "One moment.",
        },
        "routing": {
            "allow_local_greetings": True,
            "allow_local_clarifications": True,
            "require_oracle_for_tools": True,
            "require_oracle_for_memory": True,
            "require_oracle_for_files": True,
            "local_confidence_threshold": 0.75,
        },
        "metrics": {
            "enabled": True,
            "log_turn_spans": True,
            "log_provider_spans": True,
        },
        "quality_targets_ms": {
            "audio_to_partial_transcript_ms": 300,
            "final_transcript_to_first_text_ms": 500,
            "final_transcript_to_first_audio_ms": 900,
            "barge_in_ack_ms": 150,
        },
    }


def build_gemini_live_voice_profile(
    *,
    model: str = DEFAULT_GEMINI_LIVE_MODEL,
    voice: str = DEFAULT_GEMINI_LIVE_VOICE,
    api_key_env: str = "GEMINI_API_KEY",
    google_search: bool = False,
    oracle_tool: bool = True,
    sidecar_host: str = "127.0.0.1",
    sidecar_port: int = 8765,
    production_evidence_report: str = DEFAULT_EVIDENCE_REPORT,
) -> dict[str, Any]:
    port = int(sidecar_port or 8765)
    if port <= 0 or port > 65535:
        raise ValueError("--sidecar-port must be between 1 and 65535")
    return {
        "enabled": True,
        "engine": "text_oracle_tts",
        "input_codec": "webm_opus",
        "output_codec": "opus",
        "input_buffer_limit_bytes": 8 * 1024 * 1024,
        "input_frame_ms": 100,
        "silence_timeout_ms": 650,
        "speech_level_threshold": 0.075,
        "barge_in_min_speech_ms": 120,
        "barge_in_min_rms": 350,
        "barge_in_stop_playback_deadline_ms": 150,
        "pre_roll_ms": 300,
        "require_live_like": True,
        "frontend_provider": "gemini_live",
        "frontend_model": str(model or DEFAULT_GEMINI_LIVE_MODEL),
        "interface_temperature": 0.2,
        "interface_max_output_tokens": 160,
        "interface_timeout_seconds": 0.8,
        "sidecar_base_url": "",
        "spark_base_url": "",
        "sidecar_host": str(sidecar_host or "127.0.0.1"),
        "sidecar_port": port,
        "sidecar_autostart": True,
        "sidecar_connect_timeout_seconds": 10.0,
        "gemini_live_api_key_env": _clean_env_name(api_key_env) or "GEMINI_API_KEY",
        "gemini_live_base_url": (
            "wss://generativelanguage.googleapis.com/ws/"
            "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        ),
        "gemini_live_voice": str(voice or DEFAULT_GEMINI_LIVE_VOICE),
        "gemini_live_google_search": bool(google_search),
        "gemini_live_oracle_tool": bool(oracle_tool),
        "production_languages": ["en", "ja"],
        "production_scripts": ["Latn", "Jpan"],
        "best_effort_languages": True,
        "production_evidence_report": str(production_evidence_report or DEFAULT_EVIDENCE_REPORT),
        "production_evidence_min_runs": 3,
        "turn_acknowledgement": {
            "enabled": True,
            "text": "One moment.",
        },
        "routing": {
            "allow_local_greetings": True,
            "allow_local_clarifications": True,
            "require_oracle_for_tools": True,
            "require_oracle_for_memory": True,
            "require_oracle_for_files": True,
            "local_confidence_threshold": 0.75,
        },
        "metrics": {
            "enabled": True,
            "log_turn_spans": True,
            "log_provider_spans": True,
        },
        "quality_targets_ms": {
            "audio_to_partial_transcript_ms": 300,
            "final_transcript_to_first_text_ms": 500,
            "final_transcript_to_first_audio_ms": 900,
            "barge_in_ack_ms": 150,
        },
    }


def build_kame_realtime_voice_profile(
    *,
    reflex_provider: str = "gemma4",
    reflex_model: str = DEFAULT_KAME_REFLEX_MODEL,
    vllm_model: str = "",
    interface_api_key_env: str = "HERMES_KAME_INTERFACE_API_KEY",
    interface_base_url: str = "",
    interface_temperature: float = 0.2,
    interface_max_output_tokens: int = 160,
    interface_timeout_seconds: float = 0.8,
    interface_audio_input: str = "auto",
    interface_max_audio_seconds: float = 30.0,
    asr_mode: str = "on_escalation",
    asr_provider: str = "streaming_stt",
    preferred_local_oracle_model: str = DEFAULT_KAME_ORACLE_MODEL,
    oracle_provider: str = "",
    oracle_api_mode: str = "chat_completions",
    oracle_timeout_seconds: float = 60.0,
    max_spoken_sentences: int = 2,
    voice_response_policy: str = "sentence_cap",
    tts_provider: str = "streaming_tts",
    tts_model: str = "",
    tts_voice: str = "",
    fallback_policy: str = "legacy_voice",
    oracle_base_url: str = "",
    oracle_provider_name: str = DEFAULT_KAME_ORACLE_PROVIDER_NAME,
    streaming_stt_base_url: str = "",
    streaming_tts_base_url: str = "",
    streaming_stt_model: str = DEFAULT_STREAMING_STT_MODEL,
    streaming_tts_model: str = DEFAULT_STREAMING_TTS_MODEL,
    streaming_tts_voice: str = "",
    streaming_stt_token_env: str = DEFAULT_STREAMING_STT_TOKEN_ENV,
    streaming_tts_token_env: str = DEFAULT_STREAMING_TTS_TOKEN_ENV,
    sidecar_host: str = "127.0.0.1",
    sidecar_port: int = 8765,
    production_evidence_report: str = DEFAULT_EVIDENCE_REPORT,
    allow_local_greetings: bool = True,
    allow_local_clarifications: bool = True,
    require_oracle_for_tools: bool = True,
    require_oracle_for_memory: bool = True,
    require_oracle_for_files: bool = True,
    local_confidence_threshold: float = 0.75,
    barge_in_min_rms: int = 350,
    barge_in_min_speech_ms: int = 120,
    barge_in_stop_playback_deadline_ms: int = 150,
    metrics_enabled: bool = True,
    metrics_log_turn_spans: bool = True,
    metrics_log_provider_spans: bool = True,
) -> dict[str, Any]:
    port = int(sidecar_port or 8765)
    if port <= 0 or port > 65535:
        raise ValueError("--sidecar-port must be between 1 and 65535")
    audio_mode = str(interface_audio_input or "auto").strip() or "auto"
    if audio_mode not in {"auto", "native_audio", "text_fallback"}:
        raise ValueError("--kame-interface-audio-input must be auto, native_audio, or text_fallback")
    asr = str(asr_mode or "on_escalation").strip() or "on_escalation"
    if asr not in {"disabled", "from_reflex", "on_escalation", "speculative", "debug", "fallback"}:
        raise ValueError("--kame-asr-mode must be disabled, from_reflex, on_escalation, speculative, debug, or fallback")
    response_policy = str(voice_response_policy or "sentence_cap").strip().lower().replace("-", "_")
    if response_policy not in {"sentence_cap", "brief_summary", "full"}:
        raise ValueError("--kame-voice-response-policy must be sentence_cap, brief_summary, or full")
    fallback = str(fallback_policy or "legacy_voice").strip().lower().replace("-", "_")
    if fallback not in {"legacy_voice", "text_only", "fail_closed"}:
        raise ValueError("--kame-fallback-policy must be legacy_voice, text_only, or fail_closed")
    try:
        max_audio_seconds = float(interface_max_audio_seconds)
    except (TypeError, ValueError):
        raise ValueError("--kame-interface-max-audio-seconds must be a number")
    if max_audio_seconds < 1.0 or max_audio_seconds > 30.0:
        raise ValueError("--kame-interface-max-audio-seconds must be between 1 and 30")
    try:
        temperature = float(interface_temperature)
    except (TypeError, ValueError):
        raise ValueError("--kame-interface-temperature must be a number")
    if temperature < 0.0 or temperature > 2.0:
        raise ValueError("--kame-interface-temperature must be between 0 and 2")
    try:
        max_output_tokens = int(interface_max_output_tokens)
    except (TypeError, ValueError):
        raise ValueError("--kame-interface-max-output-tokens must be a positive integer")
    if max_output_tokens <= 0:
        raise ValueError("--kame-interface-max-output-tokens must be a positive integer")
    try:
        timeout_seconds = float(interface_timeout_seconds)
    except (TypeError, ValueError):
        raise ValueError("--kame-interface-timeout-seconds must be a positive number")
    if timeout_seconds <= 0:
        raise ValueError("--kame-interface-timeout-seconds must be a positive number")
    try:
        oracle_timeout = float(oracle_timeout_seconds)
    except (TypeError, ValueError):
        raise ValueError("--kame-oracle-timeout-seconds must be a positive number")
    if oracle_timeout <= 0:
        raise ValueError("--kame-oracle-timeout-seconds must be a positive number")
    try:
        spoken_sentences = int(max_spoken_sentences)
    except (TypeError, ValueError):
        raise ValueError("--kame-max-spoken-sentences must be a positive integer")
    if spoken_sentences <= 0:
        raise ValueError("--kame-max-spoken-sentences must be a positive integer")
    oracle_api = str(oracle_api_mode or "chat_completions").strip().lower()
    if oracle_api not in {"chat_completions", "anthropic_messages", "codex_responses"}:
        raise ValueError("--kame-oracle-api-mode must be chat_completions, anthropic_messages, or codex_responses")
    try:
        confidence_threshold = float(local_confidence_threshold)
    except (TypeError, ValueError):
        raise ValueError("--kame-local-confidence-threshold must be a number")
    if confidence_threshold < 0.0 or confidence_threshold > 1.0:
        raise ValueError("--kame-local-confidence-threshold must be between 0 and 1")
    try:
        min_rms = int(barge_in_min_rms)
    except (TypeError, ValueError):
        raise ValueError("--kame-barge-in-min-rms must be a non-negative integer")
    if min_rms < 0:
        raise ValueError("--kame-barge-in-min-rms must be a non-negative integer")
    try:
        min_speech_ms = int(barge_in_min_speech_ms)
    except (TypeError, ValueError):
        raise ValueError("--kame-barge-in-min-speech-ms must be a positive integer")
    if min_speech_ms <= 0:
        raise ValueError("--kame-barge-in-min-speech-ms must be a positive integer")
    try:
        stop_deadline_ms = int(barge_in_stop_playback_deadline_ms)
    except (TypeError, ValueError):
        raise ValueError("--kame-barge-in-stop-playback-deadline-ms must be a positive integer")
    if stop_deadline_ms <= 0:
        raise ValueError("--kame-barge-in-stop-playback-deadline-ms must be a positive integer")

    interface_url = _clean_url(interface_base_url)
    oracle_url = _clean_url(oracle_base_url)
    asr_url = _clean_url(streaming_stt_base_url)
    tts_url = _clean_url(streaming_tts_base_url)
    local_oracle_model = str(preferred_local_oracle_model or DEFAULT_KAME_ORACLE_MODEL)
    reflex_model_label = str(reflex_model or DEFAULT_KAME_REFLEX_MODEL)
    served_vllm_model = str(vllm_model or reflex_model_label)
    tts_model_label = str(tts_model or streaming_tts_model or DEFAULT_STREAMING_TTS_MODEL)
    tts_voice_label = str(tts_voice or streaming_tts_voice or "")
    profile = {
        "enabled": True,
        "engine": "kame_interface_oracle",
        "input_codec": "webm_opus",
        "output_codec": "opus",
        "input_buffer_limit_bytes": 8 * 1024 * 1024,
        "input_frame_ms": 100,
        "silence_timeout_ms": 650,
        "speech_level_threshold": 0.075,
        "barge_in_min_speech_ms": min_speech_ms,
        "barge_in_min_rms": min_rms,
        "barge_in_stop_playback_deadline_ms": stop_deadline_ms,
        "pre_roll_ms": 300,
        "require_live_like": True,
        "frontend_provider": str(reflex_provider or "gemma4"),
        "frontend_model": reflex_model_label,
        "interface_base_url": interface_url,
        "interface_api_key_env": _clean_env_name(interface_api_key_env) or "HERMES_KAME_INTERFACE_API_KEY",
        "vllm_base_url": interface_url,
        "vllm_model": served_vllm_model,
        "interface_temperature": temperature,
        "interface_max_output_tokens": max_output_tokens,
        "interface_timeout_seconds": timeout_seconds,
        "interface_max_audio_seconds": max_audio_seconds,
        "interface_audio_input": audio_mode,
        "asr_mode": asr,
        "asr_provider": str(asr_provider or "streaming_stt"),
        "asr_model": str(streaming_stt_model or DEFAULT_STREAMING_STT_MODEL),
        "asr_base_url": asr_url,
        "preferred_local_oracle_model": local_oracle_model,
        "oracle_provider": str(oracle_provider or ""),
        "oracle_api_mode": oracle_api,
        "oracle_timeout_seconds": oracle_timeout,
        "max_spoken_sentences": spoken_sentences,
        "voice_response_policy": response_policy,
        "tts_provider": str(tts_provider or "streaming_tts"),
        "tts_model": tts_model_label,
        "tts_voice": tts_voice_label,
        "tts_base_url": tts_url,
        "fallback_policy": fallback,
        "sidecar_base_url": "",
        "spark_base_url": "",
        "sidecar_host": str(sidecar_host or "127.0.0.1"),
        "sidecar_port": port,
        "sidecar_autostart": True,
        "sidecar_connect_timeout_seconds": 10.0,
        "streaming_stt_base_url": asr_url,
        "streaming_stt_model": str(streaming_stt_model or DEFAULT_STREAMING_STT_MODEL),
        "streaming_stt_token_env": _clean_env_name(streaming_stt_token_env),
        "streaming_tts_base_url": tts_url,
        "streaming_tts_model": str(streaming_tts_model or tts_model_label),
        "streaming_tts_voice": str(streaming_tts_voice or tts_voice_label),
        "streaming_tts_token_env": _clean_env_name(streaming_tts_token_env),
        "production_languages": ["en", "ja"],
        "production_scripts": ["Latn", "Jpan"],
        "best_effort_languages": True,
        "production_evidence_report": str(production_evidence_report or DEFAULT_EVIDENCE_REPORT),
        "production_evidence_min_runs": 3,
        "turn_acknowledgement": {
            "enabled": True,
            "text": "One moment.",
        },
        "routing": {
            "allow_local_greetings": bool(allow_local_greetings),
            "allow_local_clarifications": bool(allow_local_clarifications),
            "require_oracle_for_tools": bool(require_oracle_for_tools),
            "require_oracle_for_memory": bool(require_oracle_for_memory),
            "require_oracle_for_files": bool(require_oracle_for_files),
            "local_confidence_threshold": confidence_threshold,
        },
        "metrics": {
            "enabled": bool(metrics_enabled),
            "log_turn_spans": bool(metrics_log_turn_spans),
            "log_provider_spans": bool(metrics_log_provider_spans),
        },
        "oracle_jobs": copy.deepcopy(DEFAULT_KAME_ORACLE_JOBS),
        "output_events": {
            "caption_aliases": False,
            "audio_aliases": False,
        },
        "quality_targets_ms": dict(KAME_QUALITY_TARGETS_MS),
    }
    if oracle_url:
        profile.update(
            {
                "oracle_provider": str(oracle_provider or "custom").strip() or "custom",
                "oracle_provider_name": str(
                    oracle_provider_name or DEFAULT_KAME_ORACLE_PROVIDER_NAME
                ).strip()
                or DEFAULT_KAME_ORACLE_PROVIDER_NAME,
                "oracle_base_url": oracle_url,
                "oracle_api_mode": oracle_api,
            }
        )
    profile.update(_kame_nested_config(profile))
    return profile


def apply_realtime_voice_profile(profile: Mapping[str, Any]) -> Path:
    from hermes_cli.config import get_config_path, read_raw_config, save_config

    config = read_raw_config()
    if not isinstance(config, dict):
        config = {}
    updated = merge_realtime_voice_profile(config, profile)
    save_config(updated)
    return get_config_path()


def merge_realtime_voice_profile(
    config: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    updated = copy.deepcopy(dict(config))
    voice = updated.get("voice")
    if not isinstance(voice, dict):
        voice = {}
    else:
        voice = copy.deepcopy(voice)
    realtime = voice.get("realtime")
    if not isinstance(realtime, dict):
        realtime = {}
    else:
        realtime = copy.deepcopy(realtime)
    realtime.update(dict(profile))
    voice["realtime"] = realtime
    updated["voice"] = voice
    discord = updated.get("discord")
    if not isinstance(discord, dict):
        discord = {}
    else:
        discord = copy.deepcopy(discord)
    discord_rt = discord.get("realtime_voice")
    if not isinstance(discord_rt, dict):
        discord_rt = {}
    else:
        discord_rt = copy.deepcopy(discord_rt)
    sidecar_url = _reference_sidecar_base_url(profile)
    discord_rt.update(
        {
            "enabled": bool(profile.get("enabled")),
            "engine": profile.get("engine"),
            "sidecar_base_url": sidecar_url,
            "sidecar_token_env": str(profile.get("sidecar_token_env") or "HERMES_VOICE_SIDECAR_TOKEN"),
            "frontend_provider": profile.get("frontend_provider"),
            "frontend_model": profile.get("frontend_model"),
            "interface_base_url": profile.get("interface_base_url") or profile.get("vllm_base_url"),
            "interface_api_key_env": profile.get("interface_api_key_env") or "HERMES_KAME_INTERFACE_API_KEY",
            "interface_temperature": profile.get("interface_temperature", 0.2),
            "interface_max_output_tokens": profile.get("interface_max_output_tokens", 160),
            "interface_timeout_seconds": profile.get("interface_timeout_seconds", 0.8),
            "interface_max_audio_seconds": profile.get("interface_max_audio_seconds", 30.0),
            "interface_audio_input": profile.get("interface_audio_input"),
            "asr_mode": profile.get("asr_mode"),
            "asr_provider": profile.get("asr_provider"),
            "asr_model": profile.get("asr_model"),
            "asr_base_url": profile.get("asr_base_url") or profile.get("streaming_stt_base_url"),
            "streaming_stt_base_url": profile.get("streaming_stt_base_url") or profile.get("asr_base_url"),
            "streaming_stt_token_env": profile.get("streaming_stt_token_env") or DEFAULT_STREAMING_STT_TOKEN_ENV,
            "oracle_provider": profile.get("oracle_provider"),
            "oracle_provider_name": profile.get("oracle_provider_name"),
            "preferred_local_oracle_model": profile.get("preferred_local_oracle_model"),
            "oracle_base_url": profile.get("oracle_base_url"),
            "oracle_api_mode": profile.get("oracle_api_mode"),
            "oracle_timeout_seconds": profile.get("oracle_timeout_seconds", 60.0),
            "max_spoken_sentences": profile.get("max_spoken_sentences", 2),
            "voice_response_policy": profile.get("voice_response_policy", "sentence_cap"),
            "tts_provider": profile.get("tts_provider"),
            "tts_model": profile.get("tts_model"),
            "tts_voice": profile.get("tts_voice"),
            "tts_base_url": profile.get("tts_base_url") or profile.get("streaming_tts_base_url"),
            "streaming_tts_base_url": profile.get("streaming_tts_base_url") or profile.get("tts_base_url"),
            "streaming_tts_token_env": profile.get("streaming_tts_token_env") or DEFAULT_STREAMING_TTS_TOKEN_ENV,
            "fallback_policy": profile.get("fallback_policy"),
            "barge_in_min_rms": profile.get("barge_in_min_rms", 350),
            "barge_in_min_speech_ms": profile.get("barge_in_min_speech_ms", 120),
            "barge_in_stop_playback_deadline_ms": profile.get("barge_in_stop_playback_deadline_ms", 150),
            "routing": copy.deepcopy(profile.get("routing") if isinstance(profile.get("routing"), dict) else {}),
            "metrics": copy.deepcopy(profile.get("metrics") if isinstance(profile.get("metrics"), dict) else {}),
            "sidecar_connect_timeout_seconds": profile.get("sidecar_connect_timeout_seconds", 10.0),
            "sidecar_close_timeout_seconds": profile.get("sidecar_close_timeout_seconds", 2.0),
        }
    )
    if str(profile.get("engine") or "") == "kame_interface_oracle":
        discord_rt.update(copy.deepcopy(_kame_nested_config(profile)))
    discord["realtime_voice"] = discord_rt
    updated["discord"] = discord
    _merge_kame_oracle_provider_config(updated, profile)
    return updated


def _kame_nested_config(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the documented nested KAME config shape alongside legacy flat keys."""

    timeout_seconds = _positive_float_or_default(profile.get("interface_timeout_seconds"), 0.8)
    oracle_timeout_seconds = _positive_float_or_default(profile.get("oracle_timeout_seconds"), 60.0)
    interface_base_url = profile.get("interface_base_url") or profile.get("vllm_base_url") or ""
    streaming_stt_base_url = profile.get("asr_base_url") or profile.get("streaming_stt_base_url") or ""
    streaming_tts_base_url = profile.get("tts_base_url") or profile.get("streaming_tts_base_url") or ""
    oracle_base_url = profile.get("oracle_base_url") or ""
    turn_acknowledgement = _mapping_or_default(
        profile.get("turn_acknowledgement"),
        {"enabled": True, "text": "One moment."},
    )
    routing = _mapping_or_default(
        profile.get("routing"),
        {
            "allow_local_greetings": True,
            "allow_local_clarifications": True,
            "require_oracle_for_tools": True,
            "require_oracle_for_memory": True,
            "require_oracle_for_files": True,
            "local_confidence_threshold": 0.75,
        },
    )
    metrics = _mapping_or_default(
        profile.get("metrics"),
        {
            "enabled": True,
            "log_turn_spans": True,
            "log_provider_spans": True,
        },
    )
    oracle_jobs = _mapping_or_default(profile.get("oracle_jobs"), DEFAULT_KAME_ORACLE_JOBS)
    output_events = _mapping_or_default(
        profile.get("output_events"),
        {
            "caption_aliases": False,
            "audio_aliases": False,
        },
    )
    quality_targets_ms = _mapping_or_default(profile.get("quality_targets_ms"), KAME_QUALITY_TARGETS_MS)
    return {
        "interface": {
            "provider": profile.get("frontend_provider") or "",
            "base_url": interface_base_url,
            "api_key_env": profile.get("interface_api_key_env") or "HERMES_KAME_INTERFACE_API_KEY",
            "model": profile.get("frontend_model") or "",
            "temperature": profile.get("interface_temperature", 0.2),
            "max_output_tokens": profile.get("interface_max_output_tokens", 160),
            "timeout_ms": int(round(timeout_seconds * 1000)),
            "max_audio_seconds": profile.get("interface_max_audio_seconds", 30.0),
            "audio_input": profile.get("interface_audio_input") or "auto",
            "asr_mode": profile.get("asr_mode") or "on_escalation",
        },
        "oracle": {
            "mode": "local_openai_compatible" if oracle_base_url else "hermes_active_oracle",
            "provider": profile.get("oracle_provider") or "",
            "provider_name": profile.get("oracle_provider_name") or "",
            "preferred_local_model": profile.get("preferred_local_oracle_model") or DEFAULT_KAME_ORACLE_MODEL,
            "base_url": oracle_base_url,
            "api_mode": profile.get("oracle_api_mode") or "chat_completions",
            "timeout_ms": int(round(oracle_timeout_seconds * 1000)),
            "max_spoken_sentences": profile.get("max_spoken_sentences", 2),
            "voice_response_policy": profile.get("voice_response_policy") or "sentence_cap",
        },
        "asr": {
            "provider": profile.get("asr_provider") or "",
            "model": profile.get("asr_model") or profile.get("streaming_stt_model") or "",
            "base_url": streaming_stt_base_url,
            "token_env": profile.get("streaming_stt_token_env") or DEFAULT_STREAMING_STT_TOKEN_ENV,
        },
        "tts": {
            "provider": profile.get("tts_provider") or "",
            "model": profile.get("tts_model") or profile.get("streaming_tts_model") or "",
            "voice": profile.get("tts_voice") or profile.get("streaming_tts_voice") or "",
            "base_url": streaming_tts_base_url,
            "token_env": profile.get("streaming_tts_token_env") or DEFAULT_STREAMING_TTS_TOKEN_ENV,
        },
        "barge_in": {
            "min_rms": profile.get("barge_in_min_rms", 350),
            "min_speech_ms": profile.get("barge_in_min_speech_ms", 120),
            "stop_playback_deadline_ms": profile.get("barge_in_stop_playback_deadline_ms", 150),
        },
        "turn_acknowledgement": turn_acknowledgement,
        "routing": routing,
        "metrics": metrics,
        "oracle_jobs": oracle_jobs,
        "output_events": output_events,
        "quality_targets_ms": quality_targets_ms,
    }


def _mapping_or_default(value: Any, default: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    return copy.deepcopy(dict(default))


def _positive_float_or_default(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _merge_kame_oracle_provider_config(updated: dict[str, Any], profile: Mapping[str, Any]) -> None:
    """Register the local KAME oracle endpoint without changing Hermes' active model.

    Hermes' existing model config and /model flow remain authoritative. A DGX
    KAME launch profile may add the local endpoint as a selectable custom
    provider, but it must not silently rewrite model.default.
    """

    if str(profile.get("engine") or "") != "kame_interface_oracle":
        return
    oracle_base_url = _clean_url(str(profile.get("oracle_base_url") or ""))
    if not oracle_base_url:
        return
    local_oracle_model = str(profile.get("preferred_local_oracle_model") or DEFAULT_KAME_ORACLE_MODEL).strip()
    if not local_oracle_model:
        local_oracle_model = DEFAULT_KAME_ORACLE_MODEL
    oracle_api_mode = (
        str(profile.get("oracle_api_mode") or "chat_completions").strip()
        or "chat_completions"
    )

    provider_name = str(
        profile.get("oracle_provider_name") or DEFAULT_KAME_ORACLE_PROVIDER_NAME
    ).strip() or DEFAULT_KAME_ORACLE_PROVIDER_NAME
    custom_entry = {
        "name": provider_name,
        "base_url": oracle_base_url,
        "model": local_oracle_model,
        "api_mode": oracle_api_mode,
    }
    raw_custom_providers = updated.get("custom_providers")
    custom_providers = (
        copy.deepcopy(raw_custom_providers)
        if isinstance(raw_custom_providers, list)
        else []
    )
    provider_name_key = provider_name.lower()
    oracle_base_key = oracle_base_url.rstrip("/")
    replaced = False
    for index, entry in enumerate(custom_providers):
        if not isinstance(entry, dict):
            continue
        entry_name = str(entry.get("name") or "").strip().lower()
        entry_base = str(entry.get("base_url") or "").strip().rstrip("/")
        if entry_name == provider_name_key or entry_base == oracle_base_key:
            merged_entry = copy.deepcopy(entry)
            merged_entry.update(custom_entry)
            custom_providers[index] = merged_entry
            replaced = True
            break
    if not replaced:
        custom_providers.append(custom_entry)
    updated["custom_providers"] = custom_providers


def _reference_sidecar_base_url(profile: Mapping[str, Any]) -> str:
    explicit = _clean_url(str(profile.get("sidecar_base_url") or ""))
    if explicit:
        return explicit
    host = str(profile.get("sidecar_host") or "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int(profile.get("sidecar_port") or 8765)
    except (TypeError, ValueError):
        port = 8765
    if port <= 0 or port > 65535:
        port = 8765
    return f"http://{host}:{port}"


def _clean_url(value: str) -> str:
    text = str(value or "").strip()
    return text.rstrip("/")


def _clean_env_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if not text.replace("_", "").isalnum() or text[0].isdigit():
        raise ValueError(f"invalid environment variable name: {text}")
    return text


if __name__ == "__main__":
    raise SystemExit(main())
