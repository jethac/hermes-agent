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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.vllm_base_url:
        os.environ["HERMES_VOICE_VLLM_BASE_URL"] = args.vllm_base_url
    if args.vllm_model:
        os.environ["HERMES_VOICE_VLLM_MODEL"] = args.vllm_model

    import uvicorn

    app = create_reference_sidecar_app(runtime_config_from_env())
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
