"""Command entrypoint for the Hermes Deepgram streaming STT bridge."""

from __future__ import annotations

import argparse
import os
import secrets

from agent.realtime_voice_deepgram_bridge import (
    create_deepgram_streaming_stt_bridge_app,
    deepgram_bridge_config_from_env,
    deepgram_bridge_prerequisite_issues,
)


DEFAULT_PRODUCTION_EN_JA_TTS_MODEL_BY_LANGUAGE = "ja:aura-2-fujin-ja,en:aura-2-thalia-en"
DEFAULT_PRODUCTION_EN_JA_STT_LANGUAGE = "multi"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Hermes-compatible Deepgram streaming STT bridge")
    parser.add_argument("--host", default=os.environ.get("HERMES_DEEPGRAM_BRIDGE_HOST", "127.0.0.1"))
    parser.add_argument("--port", default=int(os.environ.get("HERMES_DEEPGRAM_BRIDGE_PORT", "8766")), type=int)
    parser.add_argument("--model", default=os.environ.get("HERMES_DEEPGRAM_MODEL", "nova-3"))
    parser.add_argument("--tts-model", default=os.environ.get("HERMES_DEEPGRAM_TTS_MODEL", "aura-2-thalia-en"))
    parser.add_argument(
        "--tts-model-by-language",
        default=os.environ.get("HERMES_DEEPGRAM_TTS_MODEL_BY_LANGUAGE", ""),
        help="Comma-separated language:model overrides for streaming TTS, for example ja:<japanese-tts-model>",
    )
    parser.add_argument(
        "--output-languages",
        default=os.environ.get("HERMES_DEEPGRAM_OUTPUT_LANGUAGES", ""),
        help="Comma-separated output language tags advertised by bridge health, for example en,ja",
    )
    parser.add_argument("--language", default=os.environ.get("HERMES_DEEPGRAM_LANGUAGE", ""))
    parser.add_argument(
        "--tts-sample-rate-hz",
        default=int(os.environ.get("HERMES_DEEPGRAM_TTS_SAMPLE_RATE_HZ", "24000")),
        type=int,
    )
    parser.add_argument(
        "--endpointing-ms",
        default=int(os.environ.get("HERMES_DEEPGRAM_ENDPOINTING_MS", "80")),
        type=int,
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
        default=os.environ.get("HERMES_DEEPGRAM_REQUIRE_OUTPUT_LANGUAGES", ""),
        help="Comma-separated TTS output languages that --check must verify, for example en,ja",
    )
    parser.add_argument(
        "--production-en-ja",
        action="store_true",
        help="Use the supported Hermes EN/JA production TTS route preset and require en,ja output in --check",
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
    tts_model_by_language = str(args.tts_model_by_language or "").strip()
    if args.production_en_ja and not tts_model_by_language:
        tts_model_by_language = DEFAULT_PRODUCTION_EN_JA_TTS_MODEL_BY_LANGUAGE
    required_output_languages = str(args.require_output_languages or "").strip()
    if args.production_en_ja and not required_output_languages:
        required_output_languages = "en,ja"
    required_input_languages = "en,ja" if args.production_en_ja else ""

    if args.model:
        os.environ["HERMES_DEEPGRAM_MODEL"] = args.model
    if args.tts_model:
        os.environ["HERMES_DEEPGRAM_TTS_MODEL"] = args.tts_model
    if tts_model_by_language:
        os.environ["HERMES_DEEPGRAM_TTS_MODEL_BY_LANGUAGE"] = tts_model_by_language
    if args.output_languages:
        os.environ["HERMES_DEEPGRAM_OUTPUT_LANGUAGES"] = args.output_languages
    configured_language = str(args.language or os.environ.get("HERMES_DEEPGRAM_LANGUAGE") or "").strip()
    if args.production_en_ja and not configured_language:
        os.environ["HERMES_DEEPGRAM_LANGUAGE"] = DEFAULT_PRODUCTION_EN_JA_STT_LANGUAGE
    elif configured_language:
        os.environ["HERMES_DEEPGRAM_LANGUAGE"] = configured_language
    if args.tts_sample_rate_hz:
        os.environ["HERMES_DEEPGRAM_TTS_SAMPLE_RATE_HZ"] = str(args.tts_sample_rate_hz)
    if args.endpointing_ms:
        os.environ["HERMES_DEEPGRAM_ENDPOINTING_MS"] = str(args.endpointing_ms)
    token_env = str(
        args.token_env
        or os.environ.get("HERMES_DEEPGRAM_BRIDGE_TOKEN_ENV")
        or "HERMES_STREAMING_STT_BRIDGE_TOKEN"
    ).strip()
    if args.token_env:
        os.environ["HERMES_DEEPGRAM_BRIDGE_TOKEN_ENV"] = token_env

    if args.generate_token:
        existing = os.environ.get(token_env)
        if existing and not args.force_token:
            print(f"Deepgram realtime voice bridge token already configured in {token_env}")
            return 0
        from hermes_cli.config import save_env_value

        save_env_value(token_env, secrets.token_urlsafe(32))
        if args.token_env:
            save_env_value("HERMES_DEEPGRAM_BRIDGE_TOKEN_ENV", token_env)
        print(f"Deepgram realtime voice bridge token stored in {token_env}")
        return 0

    runtime = deepgram_bridge_config_from_env()
    if args.check:
        issues = deepgram_bridge_prerequisite_issues(
            runtime,
            require_auth_token=bool(args.strict),
            required_input_languages=_parse_required_languages(required_input_languages),
            required_output_languages=_parse_required_languages(required_output_languages),
        )
        if issues:
            print(f"Deepgram realtime voice bridge check failed: {len(issues)} issue(s)")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        print("Deepgram realtime voice bridge check OK")
        print(f"  model: {runtime.model}")
        print(f"  tts_model: {runtime.tts_model}")
        print(
            "  tts_model_by_language: "
            f"{','.join(sorted(runtime.tts_model_by_language)) if runtime.tts_model_by_language else 'not configured'}"
        )
        print(
            "  output_languages: "
            f"{','.join(runtime.output_languages) if runtime.output_languages else 'inferred from TTS models'}"
        )
        print(f"  language: {runtime.language or 'default(en)'}")
        print(f"  auth_token: {'configured' if runtime.auth_token else 'not configured'}")
        return 0

    import uvicorn

    app = create_deepgram_streaming_stt_bridge_app(runtime)
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
