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
    sidecar_path.write_bytes(
        plistlib.dumps(
            build_realtime_voice_sidecar_plist(
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
            ),
            sort_keys=False,
        )
    )

    print(f"Wrote {bridge_path}")
    print(f"Wrote {sidecar_path}")
    print("Install with:")
    print(f"  cp {bridge_path} ~/Library/LaunchAgents/")
    print(f"  cp {sidecar_path} ~/Library/LaunchAgents/")
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
        'export HERMES_VOICE_STREAMING_TTS_TOKEN="${HERMES_VOICE_STREAMING_TTS_TOKEN:-$HERMES_STREAMING_STT_BRIDGE_TOKEN}"; '
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
