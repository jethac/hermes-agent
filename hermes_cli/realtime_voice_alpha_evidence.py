"""Run repeated realtime voice alpha evidence checks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from agent.realtime_voice_smoke_report import (
    ALPHA_REQUIRED_AUDIO_FIXTURES,
    KAME_LATENCY_REPORT_LABELS,
    load_realtime_voice_smoke_report,
    summarize_realtime_voice_smoke_report_runs,
    validate_realtime_voice_alpha_report_runs,
)
from agent.realtime_voice_errors import sanitize_realtime_voice_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect repeated Hermes realtime voice private-alpha evidence reports"
    )
    parser.add_argument(
        "--output-dir",
        default="./artifacts/realtime-voice-evidence",
        help="Directory where per-run JSON reports will be written",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of independent alpha evidence runs to collect",
    )
    parser.add_argument(
        "--prefix",
        default="realtime-voice-alpha",
        help="Report filename prefix",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting numeric suffix for report filenames",
    )
    parser.add_argument(
        "--audio-codec",
        choices=("webm_opus", "opus", "pcm16"),
        default="webm_opus",
        help="Codec for the required alpha audio fixtures",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing report files",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="After validation succeeds, set voice.realtime.production_evidence_report in config.yaml",
    )
    parser.add_argument(
        "--provider",
        choices=("deepgram", "elevenlabs", "cartesia", "loopback", "local_speech"),
        default="deepgram",
        help="Streaming voice bridge provider to start when --start-bridge is set",
    )
    parser.add_argument(
        "--start-bridge",
        action="store_true",
        help="Start the configured local streaming STT/TTS provider bridge for this evidence run",
    )
    parser.add_argument(
        "--bridge-host",
        default="",
        help="Host for --start-bridge; defaults to the configured streaming bridge URL host",
    )
    parser.add_argument(
        "--bridge-port",
        type=int,
        default=0,
        help="Port for --start-bridge; defaults to the configured streaming bridge URL port",
    )
    parser.add_argument(
        "--bridge-timeout-seconds",
        type=float,
        default=15.0,
        help="Seconds to wait for an auto-started provider bridge to become healthy",
    )
    parser.add_argument(
        "--start-deepgram-bridge",
        action="store_true",
        help="Backward-compatible alias for --provider deepgram --start-bridge",
    )
    parser.add_argument(
        "--deepgram-bridge-host",
        default="",
        help="Backward-compatible alias for --bridge-host with --start-deepgram-bridge",
    )
    parser.add_argument(
        "--deepgram-bridge-port",
        type=int,
        default=0,
        help="Backward-compatible alias for --bridge-port with --start-deepgram-bridge",
    )
    parser.add_argument(
        "--deepgram-bridge-timeout-seconds",
        type=float,
        default=None,
        help="Backward-compatible alias for --bridge-timeout-seconds with --start-deepgram-bridge",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_count = max(1, int(args.runs or 1))
    start_index = max(1, int(args.start_index or 1))
    output_dir = Path(args.output_dir).expanduser()

    report_paths = [
        output_dir / f"{args.prefix}-{index:03d}.json"
        for index in range(start_index, start_index + run_count)
    ]
    existing = [path for path in report_paths if path.exists()]
    if existing and not args.overwrite:
        print(
            "Realtime voice alpha evidence failed: report file already exists "
            f"({existing[0]}); pass --overwrite or choose another --start-index",
            file=sys.stderr,
        )
        return 1

    missing_fixtures = missing_required_audio_fixtures()
    if missing_fixtures:
        print(
            "Realtime voice alpha evidence failed: missing required audio fixture(s)",
            file=sys.stderr,
        )
        for fixture in missing_fixtures:
            print(f"  - {fixture}", file=sys.stderr)
        print_realtime_voice_fixture_setup_hint()
        return 1

    try:
        from hermes_cli.doctor import _realtime_voice_smoke_config, run_doctor

        _realtime_voice_smoke_config()
    except Exception as exc:
        print(
            "Realtime voice alpha evidence failed: realtime voice smoke is not configured "
            f"({sanitize_realtime_voice_error(exc)})",
            file=sys.stderr,
        )
        print_realtime_voice_live_setup_hint()
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with managed_streaming_bridge_for_evidence(args):
            with managed_realtime_voice_sidecar_for_evidence():
                live_like_issue = realtime_voice_live_like_preflight_issue()
                if live_like_issue:
                    print(
                        "Realtime voice alpha evidence failed: live-like realtime voice is not ready "
                        f"({live_like_issue})",
                        file=sys.stderr,
                    )
                    print_realtime_voice_live_setup_hint()
                    return 1
                for ordinal, report_path in enumerate(report_paths, start=1):
                    print(f"Realtime voice alpha evidence run {ordinal}/{run_count}: {report_path}")
                    run_doctor(
                        Namespace(
                            fix=False,
                            ack=None,
                            realtime_voice=True,
                            realtime_voice_alpha=True,
                            realtime_voice_smoke=False,
                            realtime_voice_audio_fixture=None,
                            realtime_voice_audio_codec=args.audio_codec,
                            realtime_voice_tts_smoke=None,
                            realtime_voice_barge_in_smoke=None,
                            realtime_voice_report=str(report_path),
                        )
                    )
                    if not report_path.exists():
                        print(
                            f"Realtime voice alpha evidence failed: {report_path} was not written",
                            file=sys.stderr,
                        )
                        return 1
                    _annotate_realtime_voice_alpha_report_for_provider(
                        report_path,
                        provider=_evidence_bridge_provider(args),
                    )
    except RuntimeError:
        return 1

    runs = [(str(path), load_realtime_voice_smoke_report(path)) for path in report_paths]
    issues = validate_realtime_voice_alpha_report_runs(runs, min_runs=run_count)
    if issues:
        print(f"Realtime voice alpha evidence failed: {len(issues)} issue(s)", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue.format()}", file=sys.stderr)
        return 1

    _print_summary(runs)
    if args.apply:
        if _evidence_bridge_provider(args) == "loopback":
            print(
                "Realtime voice alpha evidence failed: loopback validation cannot be applied as production evidence",
                file=sys.stderr,
            )
            return 1
        config_path = apply_realtime_voice_production_evidence_report(output_dir)
        print(f"Updated realtime voice production_evidence_report in {config_path}")
    return 0


def missing_required_audio_fixtures() -> list[str]:
    return [
        fixture
        for fixture in ALPHA_REQUIRED_AUDIO_FIXTURES
        if not Path(fixture).expanduser().is_file()
    ]


def apply_realtime_voice_production_evidence_report(report_path: str | Path) -> Path:
    from hermes_cli.config import get_config_path, read_raw_config, save_config

    path = Path(report_path).expanduser()
    config = read_raw_config()
    if not isinstance(config, dict):
        config = {}
    voice = config.get("voice")
    if not isinstance(voice, dict):
        voice = {}
    realtime = voice.get("realtime")
    if not isinstance(realtime, dict):
        realtime = {}
    realtime["production_evidence_report"] = str(path)
    voice["realtime"] = realtime
    config["voice"] = voice
    save_config(config)
    return get_config_path()


def _annotate_realtime_voice_alpha_report_for_provider(
    report_path: str | Path,
    *,
    provider: str,
) -> None:
    """Add provider-specific evidence metadata before alpha validation."""
    if provider not in {"elevenlabs", "cartesia", "loopback"}:
        return
    path = Path(report_path).expanduser()
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(entries, list):
        return
    if provider == "loopback":
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry["evidence_provider"] = "loopback"
            entry["loopback_validation"] = True
        path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return
    partial_ceiling_ms = 1000
    for entry in entries:
        if isinstance(entry, dict) and entry.get("kind") == "manifest":
            ceilings = entry.get("quality_target_ceilings_ms")
            if not isinstance(ceilings, dict):
                ceilings = {}
            ceilings["audio_to_partial_transcript_ms"] = max(
                int(ceilings.get("audio_to_partial_transcript_ms") or 0),
                partial_ceiling_ms,
            )
            entry["quality_target_ceilings_ms"] = ceilings
            break
    for entry in entries:
        if isinstance(entry, dict) and entry.get("kind") in {"audio_fixture", "audio_session"}:
            entry["target_ms"] = max(int(entry.get("target_ms") or 0), partial_ceiling_ms)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@contextmanager
def managed_streaming_bridge_for_evidence(args: argparse.Namespace) -> Iterator[None]:
    """Optionally start the configured local streaming provider bridge for evidence runs."""

    provider = _evidence_bridge_provider(args)
    start_bridge = bool(getattr(args, "start_bridge", False) or getattr(args, "start_deepgram_bridge", False))
    if not start_bridge:
        yield
        return

    proc = None
    try:
        from hermes_cli import web_server

        realtime = web_server._realtime_voice_config_dict()
        env_on_disk = web_server.load_env()
        if provider == "local_speech":
            with managed_local_speech_bridges_for_evidence(args, realtime, env_on_disk):
                yield
            return
        bridge_url = _configured_streaming_bridge_url(realtime)
        if not bridge_url:
            raise RuntimeError("voice.realtime.streaming_stt_base_url is required")
        token = _streaming_bridge_token_for_evidence(realtime, env_on_disk)
        if provider == "deepgram":
            prerequisite_issues = _deepgram_bridge_prerequisite_issues_for_evidence(env_on_disk)
        else:
            prerequisite_issues = _streaming_bridge_prerequisite_issues_for_evidence(provider, env_on_disk)
        if prerequisite_issues:
            raise RuntimeError(
                f"{_bridge_provider_label(provider)} bridge prerequisite check failed: "
                + "; ".join(prerequisite_issues)
            )
        healthy = (
            _deepgram_bridge_healthy(bridge_url, token=token)
            if provider == "deepgram"
            else _streaming_bridge_healthy(bridge_url, token=token)
        )
        if not healthy:
            host, port = (
                _deepgram_bridge_bind(args, bridge_url)
                if provider == "deepgram"
                else _streaming_bridge_bind(args, bridge_url, provider=provider)
            )
            proc = (
                _spawn_deepgram_bridge_for_evidence(host, port, env_on_disk)
                if provider == "deepgram"
                else _spawn_streaming_bridge_for_evidence(provider, host, port, env_on_disk)
            )
            if provider == "deepgram":
                _wait_for_deepgram_bridge_health(
                    bridge_url,
                    token=token,
                    proc=proc,
                    timeout_seconds=max(0.1, float(_bridge_timeout_seconds(args) or 15.0)),
                )
            else:
                _wait_for_streaming_bridge_health(
                    bridge_url,
                    token=token,
                    proc=proc,
                    provider=provider,
                    timeout_seconds=max(0.1, float(_bridge_timeout_seconds(args) or 15.0)),
                )
    except Exception as exc:
        _terminate_process(proc)
        label = _bridge_provider_label(provider)
        print(
            f"Realtime voice alpha evidence failed: {label} bridge is not ready "
            f"({sanitize_realtime_voice_error(exc)})",
            file=sys.stderr,
        )
        raise RuntimeError(f"{label} bridge is not ready") from exc
    try:
        yield
    finally:
        _terminate_process(proc)


@contextmanager
def managed_deepgram_bridge_for_evidence(args: argparse.Namespace) -> Iterator[None]:
    """Backward-compatible Deepgram-only wrapper for evidence runs."""

    setattr(args, "provider", "deepgram")
    setattr(args, "start_bridge", getattr(args, "start_deepgram_bridge", False))
    with managed_streaming_bridge_for_evidence(args):
        yield


@contextmanager
def managed_local_speech_bridges_for_evidence(
    args: argparse.Namespace,
    realtime: dict[str, Any],
    env_on_disk: dict[str, str],
) -> Iterator[None]:
    """Start separate local ASR and TTS proxy bridges for DGX-style evidence."""

    asr_url = str(realtime.get("streaming_stt_base_url") or "").strip().rstrip("/")
    tts_url = str(realtime.get("streaming_tts_base_url") or "").strip().rstrip("/")
    if not asr_url:
        raise RuntimeError("voice.realtime.streaming_stt_base_url is required for local_speech")
    if not tts_url:
        raise RuntimeError("voice.realtime.streaming_tts_base_url is required for local_speech")
    if int(getattr(args, "bridge_port", 0) or getattr(args, "deepgram_bridge_port", 0) or 0) > 0:
        raise RuntimeError(
            "--provider local_speech uses separate ASR/TTS bridge URLs; configure "
            "voice.realtime.streaming_stt_base_url and streaming_tts_base_url instead of --bridge-port"
        )
    prerequisite_issues = _local_speech_bridge_prerequisite_issues_for_evidence(env_on_disk)
    if prerequisite_issues:
        raise RuntimeError(
            "Local speech bridge prerequisite check failed: "
            + "; ".join(prerequisite_issues)
        )
    token = _streaming_bridge_token_for_evidence(realtime, env_on_disk)
    procs: list[subprocess.Popen] = []
    try:
        if not _streaming_bridge_healthy(asr_url, token=token):
            asr_host, asr_port = _local_speech_bridge_bind(args, asr_url, provider="nemotron_speech")
            asr_proc = _spawn_streaming_bridge_for_evidence(
                "nemotron_speech",
                asr_host,
                asr_port,
                env_on_disk,
            )
            procs.append(asr_proc)
            _wait_for_streaming_bridge_health(
                asr_url,
                token=token,
                proc=asr_proc,
                provider="nemotron_speech",
                timeout_seconds=max(0.1, float(_bridge_timeout_seconds(args) or 15.0)),
            )
        if not _streaming_bridge_healthy(tts_url, token=token):
            tts_host, tts_port = _local_speech_bridge_bind(args, tts_url, provider="magpie_tts")
            tts_proc = _spawn_streaming_bridge_for_evidence(
                "magpie_tts",
                tts_host,
                tts_port,
                env_on_disk,
            )
            procs.append(tts_proc)
            _wait_for_streaming_bridge_health(
                tts_url,
                token=token,
                proc=tts_proc,
                provider="magpie_tts",
                timeout_seconds=max(0.1, float(_bridge_timeout_seconds(args) or 15.0)),
            )
        yield
    finally:
        for proc in reversed(procs):
            _terminate_process(proc)


def _evidence_bridge_provider(args: argparse.Namespace) -> str:
    if getattr(args, "start_deepgram_bridge", False):
        return "deepgram"
    provider = str(getattr(args, "provider", "deepgram") or "deepgram").strip().lower()
    if provider not in {"deepgram", "elevenlabs", "cartesia", "loopback", "local_speech"}:
        raise RuntimeError("--provider must be deepgram, elevenlabs, cartesia, loopback, or local_speech")
    return provider


def _bridge_provider_label(provider: str) -> str:
    if provider == "elevenlabs":
        return "ElevenLabs"
    if provider == "cartesia":
        return "Cartesia"
    if provider == "loopback":
        return "Loopback"
    if provider == "local_speech":
        return "Local speech"
    if provider == "nemotron_speech":
        return "Nemotron Speech"
    if provider == "magpie_tts":
        return "Magpie TTS"
    return "Deepgram"


def _bridge_default_port(provider: str) -> int:
    if provider == "loopback":
        return 8768
    if provider == "cartesia":
        return 8769
    if provider == "nemotron_speech":
        return 8767
    if provider == "magpie_tts":
        return 8768
    return 8767 if provider == "elevenlabs" else 8766


def _bridge_timeout_seconds(args: argparse.Namespace) -> float:
    if getattr(args, "deepgram_bridge_timeout_seconds", None) is not None:
        return float(getattr(args, "deepgram_bridge_timeout_seconds") or 15.0)
    return float(getattr(args, "bridge_timeout_seconds", 15.0) or 15.0)


def _configured_streaming_bridge_url(realtime: dict[str, Any]) -> str:
    return str(
        realtime.get("streaming_stt_base_url")
        or realtime.get("streaming_tts_base_url")
        or ""
    ).strip().rstrip("/")


def _local_speech_bridge_bind(
    args: argparse.Namespace,
    bridge_url: str,
    *,
    provider: str,
) -> tuple[str, int]:
    parsed = urlparse(bridge_url)
    legacy_host = str(getattr(args, "deepgram_bridge_host", "") or "").strip()
    host = str(getattr(args, "bridge_host", "") or legacy_host or "").strip()
    if not host:
        host = parsed.hostname or "127.0.0.1"
    try:
        port = int(parsed.port or _bridge_default_port(provider))
    except ValueError:
        port = _bridge_default_port(provider)
    if port <= 0 or port > 65535:
        raise RuntimeError(f"{_bridge_provider_label(provider)} bridge port must be between 1 and 65535")
    configured_host = parsed.hostname or ""
    explicit_host = str(getattr(args, "bridge_host", "") or legacy_host or "").strip()
    if (
        configured_host
        and configured_host not in {"127.0.0.1", "localhost", "::1"}
        and not explicit_host
    ):
        raise RuntimeError(
            f"configured {_bridge_provider_label(provider)} bridge URL is not loopback; "
            "start that bridge on its host or pass --bridge-host for an explicit local bind"
        )
    return host, port


def _configured_deepgram_bridge_url(realtime: dict[str, Any]) -> str:
    return _configured_streaming_bridge_url(realtime)


def _streaming_bridge_token_for_evidence(
    realtime: dict[str, Any],
    env_on_disk: dict[str, str],
) -> str:
    token_env = str(
        realtime.get("streaming_stt_token_env")
        or realtime.get("streaming_tts_token_env")
        or "HERMES_STREAMING_STT_BRIDGE_TOKEN"
    ).strip()
    if not token_env:
        return ""
    return str(env_on_disk.get(token_env) or os.environ.get(token_env) or "")


def _streaming_bridge_bind(
    args: argparse.Namespace,
    bridge_url: str,
    *,
    provider: str,
) -> tuple[str, int]:
    parsed = urlparse(bridge_url)
    legacy_host = str(getattr(args, "deepgram_bridge_host", "") or "").strip()
    host = str(getattr(args, "bridge_host", "") or legacy_host or "").strip()
    if not host:
        host = parsed.hostname or "127.0.0.1"
    legacy_port = int(getattr(args, "deepgram_bridge_port", 0) or 0)
    port = int(getattr(args, "bridge_port", 0) or legacy_port or 0)
    if port <= 0:
        try:
            port = int(parsed.port or _bridge_default_port(provider))
        except ValueError:
            port = _bridge_default_port(provider)
    if port <= 0 or port > 65535:
        raise RuntimeError("--bridge-port must be between 1 and 65535")
    configured_host = parsed.hostname or ""
    explicit_host = str(getattr(args, "bridge_host", "") or legacy_host or "").strip()
    if (
        configured_host
        and configured_host not in {"127.0.0.1", "localhost", "::1"}
        and not explicit_host
    ):
        raise RuntimeError(
            "configured streaming bridge URL is not loopback; start that bridge on its host "
            "or pass --bridge-host for an explicit local bind"
        )
    return host, port


def _deepgram_bridge_bind(args: argparse.Namespace, bridge_url: str) -> tuple[str, int]:
    return _streaming_bridge_bind(args, bridge_url, provider="deepgram")


def _spawn_streaming_bridge_for_evidence(
    provider: str,
    host: str,
    port: int,
    env_on_disk: dict[str, str],
) -> subprocess.Popen:
    log_path = _streaming_bridge_log_path(provider)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "ab", buffering=0)
    label = _bridge_provider_label(provider)
    log_file.write(f"\n=== {label} realtime voice bridge started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode())
    child_env = {
        **os.environ,
        **env_on_disk,
        "HERMES_NONINTERACTIVE": "1",
    }
    module = (
        "hermes_cli.realtime_voice_elevenlabs_bridge"
        if provider == "elevenlabs"
        else "hermes_cli.realtime_voice_cartesia_bridge"
        if provider == "cartesia"
        else "hermes_cli.realtime_voice_nemotron_speech_bridge"
        if provider == "nemotron_speech"
        else "hermes_cli.realtime_voice_magpie_tts_bridge"
        if provider == "magpie_tts"
        else "hermes_cli.realtime_voice_loopback_bridge"
        if provider == "loopback"
        else "hermes_cli.realtime_voice_deepgram_bridge"
    )
    command = [
        sys.executable,
        "-m",
        module,
        "--host",
        str(host),
        "--port",
        str(port),
        "--production-en-ja",
    ]
    popen_kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        "env": child_env,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        popen_kwargs["start_new_session"] = True
    try:
        return subprocess.Popen(command, **popen_kwargs)
    finally:
        log_file.close()


def _spawn_deepgram_bridge_for_evidence(
    host: str,
    port: int,
    env_on_disk: dict[str, str],
) -> subprocess.Popen:
    return _spawn_streaming_bridge_for_evidence("deepgram", host, port, env_on_disk)


def _deepgram_bridge_prerequisite_issues_for_evidence(env_on_disk: dict[str, str]) -> list[str]:
    from agent.realtime_voice_deepgram_bridge import (
        deepgram_bridge_config_from_env,
        deepgram_bridge_prerequisite_issues,
    )
    from hermes_cli.realtime_voice_deepgram_bridge import (
        DEFAULT_PRODUCTION_EN_JA_STT_LANGUAGE,
        DEFAULT_PRODUCTION_EN_JA_TTS_MODEL_BY_LANGUAGE,
    )

    merged_env = {
        **os.environ,
        **env_on_disk,
    }
    if not str(merged_env.get("HERMES_DEEPGRAM_TTS_MODEL_BY_LANGUAGE") or "").strip():
        merged_env["HERMES_DEEPGRAM_TTS_MODEL_BY_LANGUAGE"] = DEFAULT_PRODUCTION_EN_JA_TTS_MODEL_BY_LANGUAGE
    if not str(merged_env.get("HERMES_DEEPGRAM_LANGUAGE") or "").strip():
        merged_env["HERMES_DEEPGRAM_LANGUAGE"] = DEFAULT_PRODUCTION_EN_JA_STT_LANGUAGE

    with _temporary_environ(merged_env):
        runtime = deepgram_bridge_config_from_env()
        return deepgram_bridge_prerequisite_issues(
            runtime,
            require_auth_token=True,
            required_input_languages=("en", "ja"),
            required_output_languages=("en", "ja"),
        )


def _streaming_bridge_prerequisite_issues_for_evidence(
    provider: str,
    env_on_disk: dict[str, str],
) -> list[str]:
    if provider == "loopback":
        return []
    if provider == "elevenlabs":
        return _elevenlabs_bridge_prerequisite_issues_for_evidence(env_on_disk)
    if provider == "cartesia":
        return _cartesia_bridge_prerequisite_issues_for_evidence(env_on_disk)
    if provider == "local_speech":
        return _local_speech_bridge_prerequisite_issues_for_evidence(env_on_disk)
    return _deepgram_bridge_prerequisite_issues_for_evidence(env_on_disk)


def _local_speech_bridge_prerequisite_issues_for_evidence(env_on_disk: dict[str, str]) -> list[str]:
    from agent.realtime_voice_local_speech_bridge import (
        local_speech_proxy_config_from_env,
        local_speech_proxy_prerequisite_issues,
    )
    from hermes_cli.realtime_voice_magpie_tts_bridge import DEFAULT_MODEL as DEFAULT_MAGPIE_TTS_MODEL
    from hermes_cli.realtime_voice_nemotron_speech_bridge import DEFAULT_MODEL as DEFAULT_NEMOTRON_SPEECH_MODEL

    merged_env = {
        **os.environ,
        **env_on_disk,
    }
    with _temporary_environ(merged_env):
        asr_runtime = local_speech_proxy_config_from_env(
            provider="nemotron_speech",
            role="stt",
            default_model=DEFAULT_NEMOTRON_SPEECH_MODEL,
            env_prefix="HERMES_NEMOTRON_SPEECH",
            default_input_languages=("en", "ja"),
        )
        tts_runtime = local_speech_proxy_config_from_env(
            provider="magpie_tts",
            role="tts",
            default_model=DEFAULT_MAGPIE_TTS_MODEL,
            env_prefix="HERMES_MAGPIE_TTS",
            default_output_languages=("en", "ja"),
        )
        return [
            *(
                f"nemotron_speech: {issue}"
                for issue in local_speech_proxy_prerequisite_issues(
                    asr_runtime,
                    require_auth_token=True,
                )
            ),
            *(
                f"magpie_tts: {issue}"
                for issue in local_speech_proxy_prerequisite_issues(
                    tts_runtime,
                    require_auth_token=True,
                )
            ),
        ]


def _elevenlabs_bridge_prerequisite_issues_for_evidence(env_on_disk: dict[str, str]) -> list[str]:
    from agent.realtime_voice_elevenlabs_bridge import (
        elevenlabs_bridge_config_from_env,
        elevenlabs_bridge_prerequisite_issues,
    )
    from hermes_cli.realtime_voice_elevenlabs_bridge import DEFAULT_PRODUCTION_EN_JA_OUTPUT_LANGUAGES

    merged_env = {
        **os.environ,
        **env_on_disk,
    }
    if not str(merged_env.get("HERMES_ELEVENLABS_OUTPUT_LANGUAGES") or "").strip():
        merged_env["HERMES_ELEVENLABS_OUTPUT_LANGUAGES"] = DEFAULT_PRODUCTION_EN_JA_OUTPUT_LANGUAGES
    if not str(merged_env.get("HERMES_ELEVENLABS_REQUIRE_OUTPUT_LANGUAGES") or "").strip():
        merged_env["HERMES_ELEVENLABS_REQUIRE_OUTPUT_LANGUAGES"] = DEFAULT_PRODUCTION_EN_JA_OUTPUT_LANGUAGES
    if not str(merged_env.get("HERMES_ELEVENLABS_LANGUAGE") or "").strip():
        merged_env["HERMES_ELEVENLABS_LANGUAGE"] = "auto"

    with _temporary_environ(merged_env):
        runtime = elevenlabs_bridge_config_from_env()
        return elevenlabs_bridge_prerequisite_issues(
            runtime,
            require_auth_token=True,
            required_input_languages=("en", "ja"),
            required_output_languages=("en", "ja"),
        )


def _cartesia_bridge_prerequisite_issues_for_evidence(env_on_disk: dict[str, str]) -> list[str]:
    from agent.realtime_voice_cartesia_bridge import (
        cartesia_bridge_config_from_env,
        cartesia_bridge_prerequisite_issues,
    )
    from hermes_cli.realtime_voice_cartesia_bridge import DEFAULT_PRODUCTION_EN_JA_OUTPUT_LANGUAGES

    merged_env = {
        **os.environ,
        **env_on_disk,
    }
    if not str(merged_env.get("HERMES_CARTESIA_OUTPUT_LANGUAGES") or "").strip():
        merged_env["HERMES_CARTESIA_OUTPUT_LANGUAGES"] = DEFAULT_PRODUCTION_EN_JA_OUTPUT_LANGUAGES
    if not str(merged_env.get("HERMES_CARTESIA_REQUIRE_OUTPUT_LANGUAGES") or "").strip():
        merged_env["HERMES_CARTESIA_REQUIRE_OUTPUT_LANGUAGES"] = DEFAULT_PRODUCTION_EN_JA_OUTPUT_LANGUAGES

    with _temporary_environ(merged_env):
        runtime = cartesia_bridge_config_from_env()
        return cartesia_bridge_prerequisite_issues(
            runtime,
            require_auth_token=True,
            required_output_languages=("en", "ja"),
        )


@contextmanager
def _temporary_environ(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update({key: str(value) for key, value in values.items()})
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _streaming_bridge_log_path(provider: str) -> Path:
    provider_name = (
        provider
        if provider in {"deepgram", "elevenlabs", "cartesia", "loopback", "nemotron_speech", "magpie_tts"}
        else "deepgram"
    )
    try:
        from hermes_cli import web_server

        action_log_dir = getattr(web_server, "_ACTION_LOG_DIR", None)
        if action_log_dir:
            return Path(action_log_dir) / f"realtime-voice-{provider_name}-bridge.log"
    except Exception:
        pass
    return Path.home() / ".hermes" / "logs" / f"realtime-voice-{provider_name}-bridge.log"


def _deepgram_bridge_log_path() -> Path:
    return _streaming_bridge_log_path("deepgram")


def _wait_for_streaming_bridge_health(
    bridge_url: str,
    *,
    token: str,
    proc: subprocess.Popen,
    provider: str,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "health check did not run"
    label = _bridge_provider_label(provider)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{label} bridge exited with code {proc.returncode}")
        try:
            if _streaming_bridge_healthy(bridge_url, token=token):
                return
        except Exception as exc:
            last_error = sanitize_realtime_voice_error(exc)
        time.sleep(0.2)
    raise RuntimeError(f"{label} bridge health did not become ready at {bridge_url}/health: {last_error}")


def _wait_for_deepgram_bridge_health(
    bridge_url: str,
    *,
    token: str,
    proc: subprocess.Popen,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "health check did not run"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Deepgram bridge exited with code {proc.returncode}")
        try:
            if _deepgram_bridge_healthy(bridge_url, token=token):
                return
        except Exception as exc:
            last_error = sanitize_realtime_voice_error(exc)
        time.sleep(0.2)
    raise RuntimeError(f"Deepgram bridge health did not become ready at {bridge_url}/health: {last_error}")


def _streaming_bridge_healthy(bridge_url: str, *, token: str = "") -> bool:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(f"{bridge_url.rstrip('/')}/health", headers=headers)
    try:
        with urlopen(request, timeout=1.0) as response:
            if int(getattr(response, "status", 0) or 0) != 200:
                return False
            try:
                payload = json.loads(response.read().decode("utf-8"))
            except Exception:
                return False
            if not isinstance(payload, dict):
                return False
            return payload.get("ok") is True
    except (OSError, URLError):
        return False


def _deepgram_bridge_healthy(bridge_url: str, *, token: str = "") -> bool:
    return _streaming_bridge_healthy(bridge_url, token=token)


def _terminate_process(proc: Any) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


@contextmanager
def managed_realtime_voice_sidecar_for_evidence() -> Iterator[None]:
    """Start the configured managed loopback sidecar for CLI evidence runs."""

    proc = None
    try:
        from hermes_cli import web_server

        realtime = web_server._realtime_voice_config_dict()
        base_url = web_server._realtime_voice_sidecar_base_url(realtime)
        if web_server._realtime_voice_should_autostart_sidecar(realtime, base_url):
            env_on_disk = web_server.load_env()
            token = web_server._realtime_voice_sidecar_token(realtime, env_on_disk)
            was_healthy = web_server._realtime_voice_sidecar_healthy(base_url, token=token)
            if not was_healthy:
                web_server._ensure_realtime_voice_sidecar(realtime)
                candidate = getattr(web_server, "_VOICE_SIDECAR_PROC", None)
                if candidate is not None and candidate.poll() is None:
                    proc = candidate
    except Exception as exc:
        print(
            "Realtime voice alpha evidence failed: managed sidecar is not ready "
            f"({sanitize_realtime_voice_error(exc)})",
            file=sys.stderr,
        )
        raise RuntimeError("managed realtime voice sidecar is not ready") from exc
    try:
        yield
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()


def realtime_voice_live_like_preflight_issue() -> str:
    try:
        from hermes_cli import web_server

        status = web_server._realtime_voice_status_payload()
    except AttributeError:
        return ""
    except Exception as exc:
        return f"status unavailable: {sanitize_realtime_voice_error(exc)}"
    if not isinstance(status, dict):
        return "status unavailable"
    if status.get("enabled") is not True:
        return "disabled"
    unavailable = str(status.get("unavailable_reason") or "").strip()
    conversation_quality = status.get("conversation_quality")
    conversation_quality = conversation_quality if isinstance(conversation_quality, dict) else {}
    if status.get("available") is not True and unavailable:
        mode = str(conversation_quality.get("mode") or "unknown")
        reason = str(conversation_quality.get("reason") or "unknown")
        return f"{unavailable}; mode={mode}; reason={reason}"
    if conversation_quality.get("live_like") is not True:
        mode = str(conversation_quality.get("mode") or "unknown")
        reason = str(conversation_quality.get("reason") or "unknown")
        return (
            f"not_live_like; mode={mode}; reason={reason}; "
            "configure streaming STT/TTS or native S2S"
        )
    return ""


def print_realtime_voice_live_setup_hint() -> None:
    print("Portable live setup:", file=sys.stderr)
    print(
        "  python -m hermes_cli.realtime_voice_profile --preset deepgram --apply --generate-bridge-token",
        file=sys.stderr,
    )
    print("  set DEEPGRAM_API_KEY=...", file=sys.stderr)
    print("  python -m hermes_cli.realtime_voice_deepgram_bridge --check --strict --production-en-ja", file=sys.stderr)
    print(
        "  python -m hermes_cli.realtime_voice_deepgram_bridge --host 127.0.0.1 --port 8766 --production-en-ja",
        file=sys.stderr,
    )
    print("  python -m hermes_cli.realtime_voice_fixture_pack --output-dir ./fixtures/realtime-voice", file=sys.stderr)
    print(
        "  python -m hermes_cli.realtime_voice_alpha_evidence --runs 3 --apply --start-deepgram-bridge",
        file=sys.stderr,
    )


def print_realtime_voice_fixture_setup_hint() -> None:
    print("Fixture setup:", file=sys.stderr)
    print("  python -m hermes_cli.realtime_voice_fixture_pack --output-dir ./fixtures/realtime-voice", file=sys.stderr)


def _print_summary(runs: list[tuple[str, list[dict[str, Any]]]]) -> None:
    summary = summarize_realtime_voice_smoke_report_runs(runs)
    print(
        "Realtime voice alpha evidence OK: "
        f"{summary.get('entries')} smoke result(s) across {summary.get('runs')} run(s)"
    )
    latency = summary.get("latency_ms")
    if not isinstance(latency, dict):
        return
    for label in (
        "audio_to_partial_transcript",
        "final_transcript_to_first_text",
        "final_transcript_to_first_audio",
        "barge_in_ack",
        *KAME_LATENCY_REPORT_LABELS,
    ):
        metric = latency.get(label)
        if not isinstance(metric, dict) or not metric.get("count"):
            continue
        print(
            f"  {label}: p50={metric.get('p50')}ms "
            f"p90={metric.get('p90')}ms p95={metric.get('p95')}ms "
            f"max={metric.get('max')}ms n={metric.get('count')}"
        )
    latency_by_stack = summary.get("latency_by_stack")
    if isinstance(latency_by_stack, dict):
        for stack_key, stack_summary in sorted(latency_by_stack.items()):
            if not isinstance(stack_summary, dict):
                continue
            stack = stack_summary.get("stack") if isinstance(stack_summary.get("stack"), dict) else {}
            stack_latency = stack_summary.get("latency_ms") if isinstance(stack_summary.get("latency_ms"), dict) else {}
            print(
                "  stack "
                f"{stack_key}: frontend={stack.get('frontend_provider') or 'unknown'}/"
                f"{stack.get('frontend_model') or 'unknown'} "
                f"oracle={stack.get('oracle_authority') or 'Hermes /model'} "
                f"tts={stack.get('tts_provider') or 'unknown'}/{stack.get('tts_model') or 'unknown'}"
            )
            for label, metric in sorted(stack_latency.items()):
                if not isinstance(metric, dict) or not metric.get("count"):
                    continue
                print(
                    f"    {label}: p50={metric.get('p50')}ms "
                    f"p90={metric.get('p90')}ms p95={metric.get('p95')}ms "
                    f"max={metric.get('max')}ms n={metric.get('count')}"
                )


if __name__ == "__main__":
    raise SystemExit(main())
