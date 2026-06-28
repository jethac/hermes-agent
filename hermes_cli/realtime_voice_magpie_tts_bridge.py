"""Command entrypoint for a local Magpie-compatible streaming TTS bridge."""

from __future__ import annotations

import argparse
import os

from agent.realtime_voice_local_speech_bridge import (
    create_local_speech_proxy_bridge_app,
    local_speech_proxy_config_from_env,
    local_speech_proxy_prerequisite_issues,
    probe_local_speech_upstream_health,
)


ENV_PREFIX = "HERMES_MAGPIE_TTS"
DEFAULT_MODEL = "magpie-local-streaming-tts"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Hermes-compatible proxy for a local Magpie TTS service. "
            "The upstream service must expose /health and /v1/streaming-tts/session."
        )
    )
    parser.add_argument("--host", default=os.environ.get(f"{ENV_PREFIX}_BRIDGE_HOST", "127.0.0.1"))
    parser.add_argument("--port", default=int(os.environ.get(f"{ENV_PREFIX}_BRIDGE_PORT", "8768")), type=int)
    parser.add_argument("--model", default=os.environ.get(f"{ENV_PREFIX}_MODEL", DEFAULT_MODEL))
    parser.add_argument("--upstream-base-url", default=os.environ.get(f"{ENV_PREFIX}_UPSTREAM_BASE_URL", ""))
    parser.add_argument("--upstream-token", default=os.environ.get(f"{ENV_PREFIX}_UPSTREAM_TOKEN", ""))
    parser.add_argument("--output-languages", default=os.environ.get(f"{ENV_PREFIX}_OUTPUT_LANGUAGES", "en,ja"))
    parser.add_argument("--production-en-ja", action="store_true", help="Advertise EN/JA production coverage")
    parser.add_argument("--check", action="store_true", help="Check bridge prerequisites and exit")
    parser.add_argument("--strict", action="store_true", help="With --check, require bridge bearer token")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ[f"{ENV_PREFIX}_MODEL"] = args.model
    if args.upstream_base_url:
        os.environ[f"{ENV_PREFIX}_UPSTREAM_BASE_URL"] = args.upstream_base_url
    if args.upstream_token:
        os.environ[f"{ENV_PREFIX}_UPSTREAM_TOKEN"] = args.upstream_token
    if args.output_languages:
        os.environ[f"{ENV_PREFIX}_OUTPUT_LANGUAGES"] = args.output_languages
    if args.production_en_ja:
        os.environ[f"{ENV_PREFIX}_OUTPUT_LANGUAGES"] = "en,ja"

    runtime = local_speech_proxy_config_from_env(
        provider="magpie_tts",
        role="tts",
        default_model=DEFAULT_MODEL,
        env_prefix=ENV_PREFIX,
        default_output_languages=("en", "ja"),
    )
    if args.check:
        upstream_health = probe_local_speech_upstream_health(runtime) if runtime.upstream_base_url else None
        issues = local_speech_proxy_prerequisite_issues(
            runtime,
            require_auth_token=bool(args.strict),
            upstream_health=upstream_health,
        )
        if issues:
            print(f"Magpie TTS bridge check failed: {len(issues)} issue(s)")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        print("Magpie TTS bridge check OK")
        print(f"  model: {runtime.model}")
        print(f"  upstream_base_url: {runtime.upstream_base_url or 'not configured'}")
        print(f"  output_languages: {','.join(runtime.output_languages) or 'not configured'}")
        print(f"  auth_token: {'configured' if runtime.auth_token else 'not configured'}")
        return 0

    import uvicorn

    app = create_local_speech_proxy_bridge_app(runtime)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
