"""Generate launchd plists for the realtime voice provider bridge and sidecar."""

from __future__ import annotations

import argparse
import plistlib
import shlex
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "./artifacts/realtime-voice-launchd"
BRIDGE_LABEL = "ai.hermes.realtime-voice.elevenlabs-bridge"
SIDECAR_LABEL = "ai.hermes.realtime-voice.sidecar"
DEFAULT_KAME_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_KAME_REFLEX_MODEL = "gemma-4-E2B-it"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate macOS LaunchAgent plists for Hermes realtime voice"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where plist files will be written",
    )
    parser.add_argument(
        "--repo-dir",
        default=".",
        help="Hermes Agent checkout used as the service working directory",
    )
    parser.add_argument(
        "--hermes-home",
        default="~/.hermes",
        help="Hermes home containing .env and logs",
    )
    parser.add_argument(
        "--uv-bin",
        default="uv",
        help="uv executable used to run the service modules",
    )
    parser.add_argument("--bridge-host", default="127.0.0.1")
    parser.add_argument("--bridge-port", type=int, default=8767)
    parser.add_argument("--sidecar-host", default="127.0.0.1")
    parser.add_argument("--sidecar-port", type=int, default=8765)
    parser.add_argument(
        "--profile",
        choices=("elevenlabs", "kame"),
        default="elevenlabs",
        help="Launch profile to generate",
    )
    parser.add_argument(
        "--interface-base-url",
        default="",
        help="OpenAI-compatible base URL for the KAME interface/reflex model",
    )
    parser.add_argument(
        "--vllm-base-url",
        default=DEFAULT_KAME_VLLM_BASE_URL,
        help="Backward-compatible alias for --interface-base-url with --profile kame",
    )
    parser.add_argument(
        "--vllm-model",
        default=DEFAULT_KAME_REFLEX_MODEL,
        help="Gemma audio reflex model for --profile kame",
    )
    parser.add_argument(
        "--streaming-tts-base-url",
        default="",
        help="Optional compatible streaming TTS bridge URL for --profile kame",
    )
    parser.add_argument("--stt-model", default="scribe_v2_realtime")
    parser.add_argument("--tts-model", default="eleven_flash_v2_5")
    parser.add_argument("--languages", default="en,ja")
    parser.add_argument("--scripts", default="Latn,Jpan")
    parser.add_argument(
        "--no-extra-dev",
        action="store_true",
        help="Omit --extra dev from generated uv commands",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(args.repo_dir).expanduser().resolve()
    hermes_home = Path(args.hermes_home).expanduser()

    bridge_path = output_dir / f"{BRIDGE_LABEL}.plist"
    sidecar_path = output_dir / f"{SIDECAR_LABEL}.plist"
    if args.profile == "elevenlabs":
        bridge_path.write_bytes(
            plistlib.dumps(
                build_elevenlabs_bridge_plist(
                    repo_dir=repo_dir,
                    hermes_home=hermes_home,
                    uv_bin=str(args.uv_bin),
                    host=str(args.bridge_host),
                    port=int(args.bridge_port),
                    include_dev_extra=not bool(args.no_extra_dev),
                ),
                sort_keys=False,
            )
        )
        sidecar = build_realtime_voice_sidecar_plist(
            repo_dir=repo_dir,
            hermes_home=hermes_home,
            uv_bin=str(args.uv_bin),
            host=str(args.sidecar_host),
            port=int(args.sidecar_port),
            bridge_base_url=f"http://{args.bridge_host}:{int(args.bridge_port)}",
            stt_model=str(args.stt_model),
            tts_model=str(args.tts_model),
            languages=str(args.languages),
            scripts=str(args.scripts),
            include_dev_extra=not bool(args.no_extra_dev),
        )
        wrote_paths = [bridge_path, sidecar_path]
    else:
        sidecar = build_kame_realtime_voice_sidecar_plist(
            repo_dir=repo_dir,
            hermes_home=hermes_home,
            uv_bin=str(args.uv_bin),
            host=str(args.sidecar_host),
            port=int(args.sidecar_port),
            interface_base_url=str(args.interface_base_url),
            vllm_base_url=str(args.vllm_base_url),
            vllm_model=str(args.vllm_model),
            streaming_tts_base_url=str(args.streaming_tts_base_url),
            tts_model=str(args.tts_model),
            languages=str(args.languages),
            scripts=str(args.scripts),
            include_dev_extra=not bool(args.no_extra_dev),
        )
        wrote_paths = [sidecar_path]
    sidecar_path.write_bytes(plistlib.dumps(sidecar, sort_keys=False))

    for path in wrote_paths:
        print(f"Wrote {path}")
    print("Install with:")
    print(f"  cp {sidecar_path} ~/Library/LaunchAgents/")
    if args.profile == "elevenlabs":
        print(f"  cp {bridge_path} ~/Library/LaunchAgents/")
        print(f"  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/{bridge_path.name}")
    print(f"  launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/{sidecar_path.name}")
    return 0


def build_elevenlabs_bridge_plist(
    *,
    repo_dir: Path,
    hermes_home: Path,
    uv_bin: str,
    host: str = "127.0.0.1",
    port: int = 8767,
    include_dev_extra: bool = True,
) -> dict[str, Any]:
    command = _shell_prelude(repo_dir=repo_dir, hermes_home=hermes_home)
    command += "exec " + _uv_python_module_command(
        uv_bin,
        "hermes_cli.realtime_voice_elevenlabs_bridge",
        [
            "--host",
            host,
            "--port",
            str(port),
            "--production-en-ja",
        ],
        include_dev_extra=include_dev_extra,
    )
    return _launchd_plist(
        label=BRIDGE_LABEL,
        command=command,
        repo_dir=repo_dir,
        hermes_home=hermes_home,
        stdout_name="realtime-voice-elevenlabs-bridge.log",
        stderr_name="realtime-voice-elevenlabs-bridge.error.log",
    )


def build_realtime_voice_sidecar_plist(
    *,
    repo_dir: Path,
    hermes_home: Path,
    uv_bin: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    bridge_base_url: str = "http://127.0.0.1:8767",
    stt_model: str = "scribe_v2_realtime",
    tts_model: str = "eleven_flash_v2_5",
    languages: str = "en,ja",
    scripts: str = "Latn,Jpan",
    include_dev_extra: bool = True,
) -> dict[str, Any]:
    command = _shell_prelude(repo_dir=repo_dir, hermes_home=hermes_home)
    command += (
        'export HERMES_VOICE_STREAMING_STT_TOKEN="${HERMES_VOICE_STREAMING_STT_TOKEN:-$HERMES_STREAMING_STT_BRIDGE_TOKEN}"; '
        'export HERMES_VOICE_STREAMING_TTS_TOKEN="${HERMES_VOICE_STREAMING_TTS_TOKEN:-${HERMES_STREAMING_TTS_BRIDGE_TOKEN:-$HERMES_STREAMING_STT_BRIDGE_TOKEN}}"; '
    )
    command += "exec " + _uv_python_module_command(
        uv_bin,
        "hermes_cli.realtime_voice_sidecar",
        [
            "--host",
            host,
            "--port",
            str(port),
            "--streaming-stt-base-url",
            bridge_base_url,
            "--streaming-stt-model",
            stt_model,
            "--streaming-tts-base-url",
            bridge_base_url,
            "--streaming-tts-model",
            tts_model,
            "--input-languages",
            languages,
            "--output-languages",
            languages,
            "--scripts",
            scripts,
        ],
        include_dev_extra=include_dev_extra,
    )
    return _launchd_plist(
        label=SIDECAR_LABEL,
        command=command,
        repo_dir=repo_dir,
        hermes_home=hermes_home,
        stdout_name="realtime-voice-sidecar.log",
        stderr_name="realtime-voice-sidecar.error.log",
    )


def build_kame_realtime_voice_sidecar_plist(
    *,
    repo_dir: Path,
    hermes_home: Path,
    uv_bin: str,
    host: str = "127.0.0.1",
    port: int = 8765,
    interface_base_url: str = "",
    vllm_base_url: str = DEFAULT_KAME_VLLM_BASE_URL,
    vllm_model: str = DEFAULT_KAME_REFLEX_MODEL,
    streaming_tts_base_url: str = "",
    tts_model: str = "portable-streaming-voice",
    languages: str = "en,ja",
    scripts: str = "Latn,Jpan",
    include_dev_extra: bool = True,
) -> dict[str, Any]:
    effective_interface_base_url = str(interface_base_url or vllm_base_url or DEFAULT_KAME_VLLM_BASE_URL)
    command = _shell_prelude(repo_dir=repo_dir, hermes_home=hermes_home)
    command += (
        "export HERMES_KAME_INTERFACE_BASE_URL="
        f"{shlex.quote(effective_interface_base_url)}; "
        'export HERMES_VOICE_VLLM_BASE_URL="${HERMES_VOICE_VLLM_BASE_URL:-$HERMES_KAME_INTERFACE_BASE_URL}"; '
    )
    args = [
        "--host",
        host,
        "--port",
        str(port),
        "--interface-base-url",
        effective_interface_base_url,
        "--vllm-base-url",
        effective_interface_base_url,
        "--vllm-model",
        vllm_model,
        "--input-languages",
        languages,
        "--output-languages",
        languages,
        "--scripts",
        scripts,
    ]
    if streaming_tts_base_url:
        command += (
            'export HERMES_VOICE_STREAMING_TTS_TOKEN="${HERMES_VOICE_STREAMING_TTS_TOKEN:-${HERMES_STREAMING_TTS_BRIDGE_TOKEN:-$HERMES_STREAMING_STT_BRIDGE_TOKEN}}"; '
        )
        args.extend(
            [
                "--streaming-tts-base-url",
                streaming_tts_base_url,
                "--streaming-tts-model",
                tts_model,
            ]
        )
    command += "exec " + _uv_python_module_command(
        uv_bin,
        "hermes_cli.realtime_voice_sidecar",
        args,
        include_dev_extra=include_dev_extra,
    )
    return _launchd_plist(
        label=SIDECAR_LABEL,
        command=command,
        repo_dir=repo_dir,
        hermes_home=hermes_home,
        stdout_name="realtime-voice-sidecar.log",
        stderr_name="realtime-voice-sidecar.error.log",
    )


def _shell_prelude(*, repo_dir: Path, hermes_home: Path) -> str:
    env_path = hermes_home / ".env"
    return (
        f"set -a; [ -f {shlex.quote(str(env_path))} ] && . {shlex.quote(str(env_path))}; "
        "set +a; "
        f"cd {shlex.quote(str(repo_dir))}; "
    )


def _uv_python_module_command(
    uv_bin: str,
    module: str,
    args: list[str],
    *,
    include_dev_extra: bool,
) -> str:
    parts = [uv_bin, "run"]
    if include_dev_extra:
        parts.extend(["--extra", "dev"])
    parts.extend(["--extra", "voice", "python", "-m", module, *args])
    return " ".join(shlex.quote(str(part)) for part in parts)


def _launchd_plist(
    *,
    label: str,
    command: str,
    repo_dir: Path,
    hermes_home: Path,
    stdout_name: str,
    stderr_name: str,
) -> dict[str, Any]:
    log_dir = hermes_home / "logs"
    return {
        "Label": label,
        "ProgramArguments": ["/bin/zsh", "-lc", command],
        "WorkingDirectory": str(repo_dir),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": str(log_dir / stdout_name),
        "StandardErrorPath": str(log_dir / stderr_name),
        "EnvironmentVariables": {
            "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "HERMES_HOME": str(hermes_home),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
