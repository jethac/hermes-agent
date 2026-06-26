"""Command entrypoint for the Hermes loopback streaming voice bridge."""

from __future__ import annotations

import argparse
import os

from agent.realtime_voice_loopback_bridge import (
    create_loopback_streaming_bridge_app,
    loopback_bridge_config_from_env,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local Hermes loopback streaming voice bridge")
    parser.add_argument("--host", default=os.environ.get("HERMES_LOOPBACK_BRIDGE_HOST", "127.0.0.1"))
    parser.add_argument("--port", default=int(os.environ.get("HERMES_LOOPBACK_BRIDGE_PORT", "8768")), type=int)
    parser.add_argument("--check", action="store_true", help="Check loopback bridge configuration and exit")
    parser.add_argument("--production-en-ja", action="store_true", help="Advertise EN/JA loopback coverage")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.production_en_ja:
        os.environ.setdefault("HERMES_LOOPBACK_INPUT_LANGUAGES", "en,ja")
        os.environ.setdefault("HERMES_LOOPBACK_OUTPUT_LANGUAGES", "en,ja")
    runtime = loopback_bridge_config_from_env()
    if args.check:
        print("Loopback realtime voice bridge check OK")
        print(f"  input_languages: {','.join(runtime.input_languages)}")
        print(f"  output_languages: {','.join(runtime.output_languages)}")
        print(f"  auth_token: {'configured' if runtime.auth_token else 'not configured'}")
        return 0

    import uvicorn

    app = create_loopback_streaming_bridge_app(runtime)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
