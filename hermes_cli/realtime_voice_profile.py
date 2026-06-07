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
DEFAULT_DEEPGRAM_BRIDGE_BASE_URL = "http://127.0.0.1:8766"
DEFAULT_DEEPGRAM_STT_MODEL = "nova-3"
DEFAULT_DEEPGRAM_TTS_MODEL = "aura-2-thalia-en"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print or apply a portable Hermes realtime voice profile"
    )
    parser.add_argument(
        "--preset",
        choices=("generic", "deepgram"),
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
        "--streaming-stt-token-env",
        default="HERMES_STREAMING_STT_BRIDGE_TOKEN",
        help="Environment variable containing the streaming STT bridge bearer token",
    )
    parser.add_argument(
        "--streaming-tts-token-env",
        default="HERMES_STREAMING_STT_BRIDGE_TOKEN",
        help="Environment variable containing the streaming TTS bridge bearer token",
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
        profile = build_realtime_voice_live_like_profile(
            streaming_stt_base_url=preset["streaming_stt_base_url"],
            streaming_tts_base_url=preset["streaming_tts_base_url"],
            streaming_stt_model=preset["streaming_stt_model"],
            streaming_tts_model=preset["streaming_tts_model"],
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
        bridge_token_env = str(profile.get("streaming_stt_token_env") or "").strip()
        if args.generate_bridge_token:
            token_result = ensure_realtime_voice_bridge_token(
                bridge_token_env,
                force=bool(args.force_bridge_token),
            )
            if args.preset == "deepgram":
                ensure_deepgram_bridge_token_env(bridge_token_env)
            if token_result == "created":
                print(f"Generated realtime voice bridge token in {bridge_token_env}")
            elif token_result == "existing":
                print(f"Realtime voice bridge token already configured in {bridge_token_env}")
        print("Next:")
        if args.preset == "deepgram":
            if not args.generate_bridge_token:
                print("  python -m hermes_cli.realtime_voice_deepgram_bridge --generate-token")
            print("  python -m hermes_cli.realtime_voice_deepgram_bridge --check --strict --production-en-ja")
            print(
                "  python -m hermes_cli.realtime_voice_deepgram_bridge "
                "--host 127.0.0.1 --port 8766 --production-en-ja"
            )
        print("  python -m hermes_cli.realtime_voice_fixture_pack --output-dir ./fixtures/realtime-voice")
        print("  python -m hermes_cli.realtime_voice_alpha_evidence --runs 3")
        return 0

    print(yaml.safe_dump({"voice": {"realtime": profile}}, sort_keys=False, allow_unicode=True).rstrip())
    return 0


def ensure_realtime_voice_bridge_token(token_env: str, *, force: bool = False) -> str:
    env_name = _clean_env_name(token_env)
    if not env_name:
        raise ValueError("streaming bridge token env is required")

    from hermes_cli.config import load_env, save_env_value

    existing = str(load_env().get(env_name) or "")
    if existing and not force:
        return "existing"
    save_env_value(env_name, secrets.token_urlsafe(32))
    return "created"


def ensure_deepgram_bridge_token_env(token_env: str) -> None:
    env_name = _clean_env_name(token_env)
    if not env_name or env_name == "HERMES_STREAMING_STT_BRIDGE_TOKEN":
        return

    from hermes_cli.config import save_env_value

    save_env_value("HERMES_DEEPGRAM_BRIDGE_TOKEN_ENV", env_name)


def _profile_preset_values(args: argparse.Namespace) -> dict[str, str]:
    streaming_stt_base_url = str(args.streaming_stt_base_url or "")
    streaming_tts_base_url = str(args.streaming_tts_base_url or "")
    streaming_stt_model = str(args.streaming_stt_model or DEFAULT_STREAMING_STT_MODEL)
    streaming_tts_model = str(args.streaming_tts_model or DEFAULT_STREAMING_TTS_MODEL)

    if args.preset != "deepgram":
        return {
            "streaming_stt_base_url": streaming_stt_base_url,
            "streaming_tts_base_url": streaming_tts_base_url,
            "streaming_stt_model": streaming_stt_model,
            "streaming_tts_model": streaming_tts_model,
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
    streaming_stt_token_env: str = "HERMES_STREAMING_STT_BRIDGE_TOKEN",
    streaming_tts_token_env: str = "HERMES_STREAMING_STT_BRIDGE_TOKEN",
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
        "pre_roll_ms": 300,
        "require_live_like": True,
        "frontend_provider": "reference",
        "frontend_model": str(streaming_stt_model or DEFAULT_STREAMING_STT_MODEL),
        "sidecar_base_url": "",
        "spark_base_url": "",
        "sidecar_host": str(sidecar_host or "127.0.0.1"),
        "sidecar_port": port,
        "sidecar_autostart": True,
        "sidecar_connect_timeout_seconds": 10.0,
        "vllm_base_url": "",
        "vllm_model": "",
        "streaming_stt_base_url": stt_url,
        "streaming_stt_model": str(streaming_stt_model or DEFAULT_STREAMING_STT_MODEL),
        "streaming_stt_token_env": _clean_env_name(streaming_stt_token_env),
        "streaming_tts_base_url": tts_url,
        "streaming_tts_model": str(streaming_tts_model or DEFAULT_STREAMING_TTS_MODEL),
        "streaming_tts_token_env": _clean_env_name(streaming_tts_token_env),
        "production_languages": ["en", "ja"],
        "production_scripts": ["Latn", "Jpan"],
        "best_effort_languages": True,
        "production_evidence_report": str(production_evidence_report or DEFAULT_EVIDENCE_REPORT),
        "production_evidence_min_runs": 3,
        "quality_targets_ms": {
            "audio_to_partial_transcript_ms": 300,
            "final_transcript_to_first_text_ms": 500,
            "final_transcript_to_first_audio_ms": 900,
            "barge_in_ack_ms": 150,
        },
    }


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
    return updated


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
