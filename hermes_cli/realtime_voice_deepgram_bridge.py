"""Command entrypoint for the Hermes Deepgram streaming STT bridge."""

from __future__ import annotations

import argparse
import os

from agent.realtime_voice_deepgram_bridge import (
    create_deepgram_streaming_stt_bridge_app,
    deepgram_bridge_config_from_env,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Hermes-compatible Deepgram streaming STT bridge")
    parser.add_argument("--host", default=os.environ.get("HERMES_DEEPGRAM_BRIDGE_HOST", "127.0.0.1"))
    parser.add_argument("--port", default=int(os.environ.get("HERMES_DEEPGRAM_BRIDGE_PORT", "8766")), type=int)
    parser.add_argument("--model", default=os.environ.get("HERMES_DEEPGRAM_MODEL", "nova-3"))
    parser.add_argument("--tts-model", default=os.environ.get("HERMES_DEEPGRAM_TTS_MODEL", "aura-2-thalia-en"))
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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.model:
        os.environ["HERMES_DEEPGRAM_MODEL"] = args.model
    if args.tts_model:
        os.environ["HERMES_DEEPGRAM_TTS_MODEL"] = args.tts_model
    if args.language:
        os.environ["HERMES_DEEPGRAM_LANGUAGE"] = args.language
    if args.tts_sample_rate_hz:
        os.environ["HERMES_DEEPGRAM_TTS_SAMPLE_RATE_HZ"] = str(args.tts_sample_rate_hz)
    if args.endpointing_ms:
        os.environ["HERMES_DEEPGRAM_ENDPOINTING_MS"] = str(args.endpointing_ms)

    import uvicorn

    app = create_deepgram_streaming_stt_bridge_app(deepgram_bridge_config_from_env())
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
