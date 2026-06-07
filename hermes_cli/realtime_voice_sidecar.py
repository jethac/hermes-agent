"""Command entrypoint for the Hermes realtime voice reference sidecar."""

from __future__ import annotations

import argparse
import os

from agent.realtime_voice_reference_sidecar import create_reference_sidecar_app, runtime_config_from_env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Hermes realtime voice reference sidecar")
    parser.add_argument("--host", default=os.environ.get("HERMES_VOICE_SIDECAR_HOST", "127.0.0.1"))
    parser.add_argument("--port", default=int(os.environ.get("HERMES_VOICE_SIDECAR_PORT", "8765")), type=int)
    parser.add_argument("--vllm-base-url", default=os.environ.get("HERMES_VOICE_VLLM_BASE_URL", ""))
    parser.add_argument("--vllm-model", default=os.environ.get("HERMES_VOICE_VLLM_MODEL", ""))
    parser.add_argument(
        "--streaming-stt-base-url",
        default=os.environ.get("HERMES_VOICE_STREAMING_STT_BASE_URL", ""),
        help="Optional compatible streaming STT bridge base URL",
    )
    parser.add_argument(
        "--streaming-stt-model",
        default=os.environ.get("HERMES_VOICE_STREAMING_STT_MODEL", ""),
        help="Optional streaming STT model name for diagnostics",
    )
    parser.add_argument(
        "--streaming-tts-base-url",
        default=os.environ.get("HERMES_VOICE_STREAMING_TTS_BASE_URL", ""),
        help="Optional compatible streaming TTS bridge base URL",
    )
    parser.add_argument(
        "--streaming-tts-model",
        default=os.environ.get("HERMES_VOICE_STREAMING_TTS_MODEL", ""),
        help="Optional streaming TTS model name for diagnostics",
    )
    parser.add_argument("--input-languages", default=os.environ.get("HERMES_VOICE_INPUT_LANGUAGES", ""))
    parser.add_argument("--output-languages", default=os.environ.get("HERMES_VOICE_OUTPUT_LANGUAGES", ""))
    parser.add_argument("--scripts", default=os.environ.get("HERMES_VOICE_SCRIPTS", ""))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.vllm_base_url:
        os.environ["HERMES_VOICE_VLLM_BASE_URL"] = args.vllm_base_url
    if args.vllm_model:
        os.environ["HERMES_VOICE_VLLM_MODEL"] = args.vllm_model
    if args.streaming_stt_base_url:
        os.environ["HERMES_VOICE_STREAMING_STT_BASE_URL"] = args.streaming_stt_base_url
    if args.streaming_stt_model:
        os.environ["HERMES_VOICE_STREAMING_STT_MODEL"] = args.streaming_stt_model
    if args.streaming_tts_base_url:
        os.environ["HERMES_VOICE_STREAMING_TTS_BASE_URL"] = args.streaming_tts_base_url
    if args.streaming_tts_model:
        os.environ["HERMES_VOICE_STREAMING_TTS_MODEL"] = args.streaming_tts_model
    if args.input_languages:
        os.environ["HERMES_VOICE_INPUT_LANGUAGES"] = args.input_languages
    if args.output_languages:
        os.environ["HERMES_VOICE_OUTPUT_LANGUAGES"] = args.output_languages
    if args.scripts:
        os.environ["HERMES_VOICE_SCRIPTS"] = args.scripts

    import uvicorn

    app = create_reference_sidecar_app(runtime_config_from_env())
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
