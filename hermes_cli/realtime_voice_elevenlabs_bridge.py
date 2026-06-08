"""Command entrypoint for the Hermes ElevenLabs realtime voice bridge."""

from __future__ import annotations

import argparse
import os
import secrets

from agent.realtime_voice_elevenlabs_bridge import (
    create_elevenlabs_realtime_bridge_app,
    elevenlabs_bridge_config_from_env,
    elevenlabs_bridge_prerequisite_issues,
)


DEFAULT_PRODUCTION_EN_JA_LANGUAGE = "auto"
DEFAULT_PRODUCTION_EN_JA_OUTPUT_LANGUAGES = "en,ja"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Hermes-compatible ElevenLabs realtime voice bridge")
    parser.add_argument("--host", default=os.environ.get("HERMES_ELEVENLABS_BRIDGE_HOST", "127.0.0.1"))
    parser.add_argument("--port", default=int(os.environ.get("HERMES_ELEVENLABS_BRIDGE_PORT", "8767")), type=int)
    parser.add_argument("--model", default=os.environ.get("HERMES_ELEVENLABS_STT_MODEL", "scribe_v2_realtime"))
    parser.add_argument("--tts-model", default=os.environ.get("HERMES_ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5"))
    parser.add_argument(
        "--voice-id",
        default=os.environ.get("ELEVENLABS_VOICE_ID") or os.environ.get("HERMES_ELEVENLABS_VOICE_ID", ""),
    )
    parser.add_argument("--language", default=os.environ.get("HERMES_ELEVENLABS_LANGUAGE", ""))
    parser.add_argument(
        "--output-format",
        default=os.environ.get("HERMES_ELEVENLABS_OUTPUT_FORMAT", "pcm_24000"),
    )
    parser.add_argument(
        "--output-languages",
        default=os.environ.get("HERMES_ELEVENLABS_OUTPUT_LANGUAGES", ""),
        help="Comma-separated output language tags advertised by bridge health, for example en,ja",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check bridge prerequisites and exit without starting the server",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="With --check, also require a bridge bearer token",
    )
    parser.add_argument(
        "--require-output-languages",
        default=os.environ.get("HERMES_ELEVENLABS_REQUIRE_OUTPUT_LANGUAGES", ""),
        help="Comma-separated TTS output languages that --check must verify, for example en,ja",
    )
    parser.add_argument(
        "--production-en-ja",
        action="store_true",
        help="Use Hermes EN/JA production defaults and require en,ja input/output in --check",
    )
    parser.add_argument(
        "--token-env",
        default=None,
        help="Environment variable used for the bridge bearer token",
    )
    parser.add_argument(
        "--generate-token",
        action="store_true",
        help="Generate and store a bridge bearer token in ~/.hermes/.env, then exit",
    )
    parser.add_argument(
        "--force-token",
        action="store_true",
        help="With --generate-token, replace an existing bridge bearer token",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_bridge_env()
    output_languages = str(args.output_languages or "").strip()
    if args.production_en_ja and not output_languages:
        output_languages = DEFAULT_PRODUCTION_EN_JA_OUTPUT_LANGUAGES
    required_output_languages = str(args.require_output_languages or "").strip()
    if args.production_en_ja and not required_output_languages:
        required_output_languages = DEFAULT_PRODUCTION_EN_JA_OUTPUT_LANGUAGES
    required_input_languages = "en,ja" if args.production_en_ja else ""

    if args.model:
        os.environ["HERMES_ELEVENLABS_STT_MODEL"] = args.model
    if args.tts_model:
        os.environ["HERMES_ELEVENLABS_TTS_MODEL"] = args.tts_model
    if args.voice_id:
        os.environ["HERMES_ELEVENLABS_VOICE_ID"] = args.voice_id
    configured_language = str(args.language or os.environ.get("HERMES_ELEVENLABS_LANGUAGE") or "").strip()
    if args.production_en_ja and not configured_language:
        os.environ["HERMES_ELEVENLABS_LANGUAGE"] = DEFAULT_PRODUCTION_EN_JA_LANGUAGE
    elif configured_language:
        os.environ["HERMES_ELEVENLABS_LANGUAGE"] = configured_language
    if args.output_format:
        os.environ["HERMES_ELEVENLABS_OUTPUT_FORMAT"] = args.output_format
    if output_languages:
        os.environ["HERMES_ELEVENLABS_OUTPUT_LANGUAGES"] = output_languages
    token_env = str(
        args.token_env
        or os.environ.get("HERMES_ELEVENLABS_BRIDGE_TOKEN_ENV")
        or "HERMES_STREAMING_STT_BRIDGE_TOKEN"
    ).strip()
    if args.token_env:
        os.environ["HERMES_ELEVENLABS_BRIDGE_TOKEN_ENV"] = token_env

    if args.generate_token:
        existing = os.environ.get(token_env)
        if existing and not args.force_token:
            print(f"ElevenLabs realtime voice bridge token already configured in {token_env}")
            return 0
        from hermes_cli.config import save_env_value

        save_env_value(token_env, secrets.token_urlsafe(32))
        if args.token_env:
            save_env_value("HERMES_ELEVENLABS_BRIDGE_TOKEN_ENV", token_env)
        print(f"ElevenLabs realtime voice bridge token stored in {token_env}")
        return 0

    runtime = elevenlabs_bridge_config_from_env()
    if args.check:
        issues = elevenlabs_bridge_prerequisite_issues(
            runtime,
            require_auth_token=bool(args.strict),
            required_input_languages=_parse_required_languages(required_input_languages),
            required_output_languages=_parse_required_languages(required_output_languages),
        )
        if issues:
            print(f"ElevenLabs realtime voice bridge check failed: {len(issues)} issue(s)")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        print("ElevenLabs realtime voice bridge check OK")
        print(f"  stt_model: {runtime.model}")
        print(f"  tts_model: {runtime.tts_model}")
        print(f"  voice_id: {'configured' if runtime.voice_id else 'not configured'}")
        print(f"  language: {runtime.language or 'auto'}")
        print(
            "  output_languages: "
            f"{','.join(runtime.output_languages) if runtime.output_languages else 'inferred from TTS model'}"
        )
        print(f"  auth_token: {'configured' if runtime.auth_token else 'not configured'}")
        return 0

    import uvicorn

    app = create_elevenlabs_realtime_bridge_app(runtime)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _load_bridge_env() -> None:
    try:
        from hermes_cli.config import load_env
    except Exception:
        return
    try:
        os.environ.update(load_env())
    except Exception:
        return


def _parse_required_languages(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").replace(" ", ",").split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
