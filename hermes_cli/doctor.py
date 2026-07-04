"""
Doctor command for hermes CLI.

Diagnoses issues with Hermes Agent setup.
"""

import os
import sys
import asyncio
import json
import subprocess
import shutil
import uuid
from argparse import Namespace
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from hermes_cli.config import get_project_root, get_hermes_home, get_env_path
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_constants import display_hermes_home
from hermes_constants import agent_browser_runnable

PROJECT_ROOT = get_project_root()
HERMES_HOME = get_hermes_home()
_DHH = display_hermes_home()  # user-facing display path (e.g. ~/.hermes or ~/.hermes/profiles/coder)

# Load environment variables from ~/.hermes/.env so API key checks work
_env_path = get_env_path()
load_hermes_dotenv(hermes_home=_env_path.parent, project_env=PROJECT_ROOT / ".env")

from hermes_cli.colors import Colors, color
from hermes_cli.models import _HERMES_USER_AGENT
from hermes_constants import OPENROUTER_MODELS_URL
from utils import base_url_host_matches


_PROVIDER_ENV_HINTS = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_TOKEN",
    "OPENAI_BASE_URL",
    "NOUS_API_KEY",
    "GLM_API_KEY",
    "ZAI_API_KEY",
    "Z_AI_API_KEY",
    "KIMI_API_KEY",
    "KIMI_CN_API_KEY",
    "GMI_API_KEY",
    "MINIMAX_API_KEY",
    "MINIMAX_CN_API_KEY",
    "KILOCODE_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "HF_TOKEN",
    "OPENCODE_ZEN_API_KEY",
    "OPENCODE_GO_API_KEY",
    "XIAOMI_API_KEY",
    "TOKENHUB_API_KEY",
)

_REALTIME_VOICE_LIVE_TARGETS_MS = {
    "audio_to_partial_transcript_ms": 300,
    "final_transcript_to_first_text_ms": 500,
    "final_transcript_to_first_audio_ms": 900,
    "barge_in_ack_ms": 150,
}
_REALTIME_VOICE_KAME_TARGETS_MS = {
    **_REALTIME_VOICE_LIVE_TARGETS_MS,
    "barge_in_confirmed_to_playback_stopped_ms": 150,
    "kame_speech_end_to_interface_decision_ms": 500,
    "kame_final_transcript_to_interface_decision_ms": 500,
    "kame_interface_decision_to_local_first_audio_ms": 500,
    "kame_speech_end_to_local_first_audio_ms": 1000,
    "kame_interface_decision_to_defer_first_audio_ms": 500,
    "kame_speech_end_to_defer_first_audio_ms": 500,
    "kame_interface_decision_to_oracle_accepted_ms": 500,
    "kame_oracle_first_token_to_first_tts_audio_ms": 1000,
    "kame_first_tts_audio_to_playback_start_ms": 150,
    "kame_speech_end_to_first_audio_ms": 3000,
    "kame_speech_end_to_playback_start_ms": 3000,
}


def _realtime_voice_alpha_audio_fixtures() -> list[str]:
    from agent.realtime_voice_smoke_report import ALPHA_REQUIRED_AUDIO_FIXTURES

    return list(ALPHA_REQUIRED_AUDIO_FIXTURES)


def _realtime_voice_alpha_audio_session_fixtures() -> list[str]:
    from agent.realtime_voice_smoke_report import ALPHA_REQUIRED_AUDIO_SESSION_FIXTURES

    return list(ALPHA_REQUIRED_AUDIO_SESSION_FIXTURES)


def _realtime_voice_alpha_tts_texts() -> list[str]:
    from agent.realtime_voice_smoke_report import ALPHA_REQUIRED_TTS_TEXTS

    return list(ALPHA_REQUIRED_TTS_TEXTS)


def _realtime_voice_alpha_barge_in_texts() -> list[str]:
    from agent.realtime_voice_smoke_report import ALPHA_REQUIRED_BARGE_IN_TEXTS

    return list(ALPHA_REQUIRED_BARGE_IN_TEXTS)


def _realtime_voice_alpha_session_turn_texts() -> list[str]:
    from agent.realtime_voice_smoke_report import ALPHA_REQUIRED_SESSION_TURN_TEXTS

    return list(ALPHA_REQUIRED_SESSION_TURN_TEXTS)


from hermes_constants import is_termux as _is_termux


def _python_install_cmd() -> str:
    return "python -m pip install" if _is_termux() else "uv pip install"


def _system_package_install_cmd(pkg: str) -> str:
    if _is_termux():
        return f"pkg install {pkg}"
    if sys.platform == "darwin":
        return f"brew install {pkg}"
    return f"sudo apt install {pkg}"


def _safe_which(cmd: str) -> str | None:
    """shutil.which wrapper resilient to platform monkeypatching in tests."""
    try:
        return shutil.which(cmd)
    except Exception:
        return None


def _termux_browser_setup_steps(node_installed: bool) -> list[str]:
    steps: list[str] = []
    step = 1
    if not node_installed:
        steps.append(f"{step}) pkg install nodejs")
        step += 1
    steps.append(f"{step}) npm install -g agent-browser")
    steps.append(f"{step + 1}) agent-browser install")
    return steps


def _termux_install_all_fallback_notes() -> list[str]:
    return [
        "Termux install profile: use .[termux-all] for broad compatibility (installer default on Termux).",
        "Matrix E2EE extra is excluded on Termux (python-olm currently fails to build).",
        "Local faster-whisper extra is excluded on Termux (ctranslate2/av build path unavailable).",
        "STT fallback: use Groq Whisper (set GROQ_API_KEY) or OpenAI Whisper (set VOICE_TOOLS_OPENAI_KEY).",
    ]


def _has_provider_env_config(content: str) -> bool:
    """Return True when ~/.hermes/.env contains provider auth/base URL settings."""
    return any(key in content for key in _PROVIDER_ENV_HINTS)


def _honcho_is_configured_for_doctor() -> bool:
    """Return True when Honcho is configured, even if this process has no active session."""
    try:
        from plugins.memory.honcho.client import HonchoClientConfig

        cfg = HonchoClientConfig.from_global_config()
        return bool(cfg.enabled and (cfg.api_key or cfg.base_url))
    except Exception:
        return False


def _is_kanban_worker_env_gate(item: dict) -> bool:
    """Return True when Kanban is unavailable only because this is not a worker process."""
    if item.get("name") != "kanban":
        return False
    if os.environ.get("HERMES_KANBAN_TASK"):
        return False

    tools = item.get("tools") or []
    return bool(tools) and all(str(tool).startswith("kanban_") for tool in tools)


def _doctor_tool_availability_detail(toolset: str) -> str:
    """Optional explanatory suffix for toolsets whose doctor status needs context."""
    if toolset == "kanban" and not os.environ.get("HERMES_KANBAN_TASK"):
        return "(runtime-gated; loaded only for dispatcher-spawned workers)"
    return ""


def _apply_doctor_tool_availability_overrides(available: list[str], unavailable: list[dict]) -> tuple[list[str], list[dict]]:
    """Adjust runtime-gated tool availability for doctor diagnostics."""
    updated_available = list(available)
    updated_unavailable = []
    for item in unavailable:
        name = item.get("name")
        if _is_kanban_worker_env_gate(item):
            if "kanban" not in updated_available:
                updated_available.append("kanban")
            continue
        if name == "honcho" and _honcho_is_configured_for_doctor():
            if "honcho" not in updated_available:
                updated_available.append("honcho")
            continue
        updated_unavailable.append(item)
    return updated_available, updated_unavailable


def _has_healthy_oauth_fallback_for_apikey_provider(provider_label: str) -> bool:
    """Return True when a direct API-key probe failure is non-blocking.

    Some provider families support both a direct API-key path and a separate
    OAuth runtime path. When the OAuth path is already healthy, doctor should
    still show a failed API-key connectivity row, but it should not promote
    that direct-key problem into the final blocking summary.
    """
    normalized = (provider_label or "").strip().lower()
    if normalized == "minimax":
        try:
            from hermes_cli.auth import get_minimax_oauth_auth_status
            return bool((get_minimax_oauth_auth_status() or {}).get("logged_in"))
        except Exception:
            return False
    if normalized == "xai":
        try:
            from hermes_cli.auth import get_xai_oauth_auth_status
            return bool((get_xai_oauth_auth_status() or {}).get("logged_in"))
        except Exception:
            return False
    return False


def check_ok(text: str, detail: str = ""):
    print(f"  {color('✓', Colors.GREEN)} {text}" + (f" {color(detail, Colors.DIM)}" if detail else ""))

def check_warn(text: str, detail: str = ""):
    print(f"  {color('⚠', Colors.YELLOW)} {text}" + (f" {color(detail, Colors.DIM)}" if detail else ""))

def check_fail(text: str, detail: str = ""):
    print(f"  {color('✗', Colors.RED)} {text}" + (f" {color(detail, Colors.DIM)}" if detail else ""))

def check_info(text: str):
    print(f"    {color('→', Colors.CYAN)} {text}")


def _section(title: str) -> None:
    """Print a doctor section banner: blank line + bold cyan ◆ title."""
    print()
    print(color(f"◆ {title}", Colors.CYAN, Colors.BOLD))


def _fail_and_issue(text: str, detail: str, fix: str, issues: list[str]) -> None:
    """Emit a check_fail and append the corresponding fix instruction."""
    check_fail(text, detail)
    issues.append(fix)


def _enabled_cli_toolsets_for_doctor() -> set[str] | None:
    """Return toolsets enabled for the CLI, or None if config resolution fails."""
    try:
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools

        return {str(toolset) for toolset in _get_platform_tools(load_config() or {}, "cli")}
    except Exception:
        return None


def _missing_api_key_toolsets_for_summary(unavailable: list[dict]) -> list[dict]:
    """Filter unavailable API-key toolsets to those enabled for the CLI."""
    api_key_unavailable = [
        item for item in unavailable
        if item.get("missing_vars") or item.get("env_vars")
    ]
    enabled_toolsets = _enabled_cli_toolsets_for_doctor()
    if enabled_toolsets is None:
        return api_key_unavailable
    return [
        item for item in api_key_unavailable
        if str(item.get("name") or "") in enabled_toolsets
    ]


def _read_pyproject_version() -> str | None:
    """Read the ``version = "..."`` from ``pyproject.toml`` at the project root.

    Returns None when running from an installed wheel (no pyproject.toml ships
    with the package) or when the file can't be parsed. Reads only the
    ``[project]`` version, ignoring any version strings that appear in other
    tables.
    """
    pyproject = PROJECT_ROOT / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return None
    in_project = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            continue
        if in_project and line.startswith("version") and "=" in line:
            value = line.split("=", 1)[1]
            value = value.split("#", 1)[0].strip().strip("\"'")
            return value or None
    return None


def _check_version_consistency(issues: list[str]) -> None:
    """Verify pyproject.toml version matches hermes_cli.__version__.

    A git conflict resolution (reset/merge) can revert one file without the
    other, leaving ``hermes --version`` reporting a stale version while
    ``pyproject.toml`` is current. Detect that drift so users can re-sync.
    Silent no-op for installed wheels where pyproject.toml isn't present.
    """
    try:
        from hermes_cli import __version__ as init_version
    except Exception:
        return
    pyproject_version = _read_pyproject_version()
    if pyproject_version is None:
        # Installed wheel or unreadable pyproject — nothing to cross-check.
        return
    if pyproject_version == init_version:
        check_ok("Version files consistent", f"({init_version})")
    else:
        _fail_and_issue(
            "Version mismatch between source files",
            f"(pyproject.toml {pyproject_version} != hermes_cli/__init__.py {init_version})",
            "Re-sync version files (e.g. run 'hermes update', or set "
            "hermes_cli/__init__.py __version__ to match pyproject.toml)",
            issues,
        )


def _check_s6_supervision(issues: list[str]) -> None:
    """Inside a container under our s6 /init, surface what s6 sees.

    Runs as a counterpart to :func:`_check_gateway_service_linger` for
    the systemd-on-host case. No-op everywhere except in the s6
    container so host runs aren't cluttered with irrelevant output.

    Reports:
      - Whether the main-hermes and dashboard static services are up
      - How many per-profile gateway slots are registered (via
        ``S6ServiceManager.list_profile_gateways()``) and how many are
        currently supervised as ``up``
    """
    try:
        from hermes_cli.service_manager import (
            S6ServiceManager,
            detect_service_manager,
        )
    except Exception:
        return

    if detect_service_manager() != "s6":
        return

    _section("s6 Supervision")

    mgr = S6ServiceManager()

    # Static services. They live under /run/service/ via s6-rc symlinks,
    # so the same s6-svstat probe works.
    for static in ("main-hermes", "dashboard"):
        if mgr.is_running(static):
            check_ok(f"{static}: up")
        else:
            check_info(f"{static}: down (expected if not enabled via env)")

    profiles = mgr.list_profile_gateways()
    if not profiles:
        check_info("No per-profile gateways registered yet — create one with `hermes profile create <name>`")
        return

    up_count = sum(1 for p in profiles if mgr.is_running(f"gateway-{p}"))
    check_ok(
        f"Per-profile gateways: {up_count}/{len(profiles)} supervised up"
        + (f" ({', '.join(sorted(profiles))})" if len(profiles) <= 8 else "")
    )


def check_certificates() -> None:
    """Verify the certifi CA bundle is loadable.

    Surfaces the SSLConfigurationError user-friendly path before they hit
    a wall of tracebacks on the first outbound HTTPS call.
    """
    try:
        from agent.ssl_guard import verify_ca_bundle_with_fallback
        from agent.errors import SSLConfigurationError
        verify_ca_bundle_with_fallback()
        check_ok("SSL CA certificate bundle is valid")
    except SSLConfigurationError as e:
        check_fail("SSL CA certificate bundle is broken", str(e))
    except Exception as e:
        check_warn("SSL certificate check skipped", str(e))


def _check_gateway_service_linger(issues: list[str]) -> None:
    """Warn when a systemd user gateway service will stop after logout.

    Skipped inside a container running under s6 — the linger concept
    (user-systemd surviving SSH logout) doesn't apply there, and the
    s6 supervision state is surfaced separately by
    ``_check_s6_supervision``.
    """
    try:
        from hermes_cli.gateway import (
            get_systemd_linger_status,
            get_systemd_unit_path,
            is_linux,
        )
        from hermes_cli.service_manager import detect_service_manager
    except Exception as e:
        check_warn("Gateway service linger", f"(could not import gateway helpers: {e})")
        return

    if not is_linux():
        return

    # Inside a container under our s6 /init, _check_s6_supervision
    # reports the live supervision state; the linger warning would be
    # confusing here (no systemd, no logout, no "lingering" concept).
    if detect_service_manager() == "s6":
        return

    unit_path = get_systemd_unit_path()
    if not unit_path.exists():
        return

    _section("Gateway Service")
    linger_enabled, linger_detail = get_systemd_linger_status()
    if linger_enabled is True:
        check_ok("Systemd linger enabled", "(gateway service survives logout)")
    elif linger_enabled is False:
        check_warn("Systemd linger disabled", "(gateway may stop after logout)")
        check_info("Run: sudo loginctl enable-linger $USER")
        issues.append("Enable linger for the gateway user service: sudo loginctl enable-linger $USER")
    else:
        check_warn("Could not verify systemd linger", f"({linger_detail})")


def _realtime_voice_status_payload() -> Mapping[str, Any]:
    from hermes_cli.web_server import _realtime_voice_status_payload as payload

    return payload(probe_health=True)


def _realtime_voice_bool(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def _realtime_voice_clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            token = item.strip()
            if token:
                result.append(token)
    return result


def _realtime_voice_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer():
        int_value = int(value)
        return int_value if int_value > 0 else None
    return None


def _realtime_voice_provider_label(payload: Mapping[str, Any]) -> str:
    provider = str(payload.get("frontend_provider") or "").strip()
    if provider:
        return provider
    sidecar = payload.get("sidecar")
    if isinstance(sidecar, Mapping):
        health = sidecar.get("health")
        if isinstance(health, Mapping):
            frontend = health.get("frontend")
            if isinstance(frontend, Mapping):
                health_provider = str(frontend.get("provider") or "").strip()
                if health_provider:
                    return health_provider
    return ""


def _realtime_voice_has_machine_named_provider(payload: Mapping[str, Any]) -> bool:
    text = " ".join(
        part
        for part in (
            _realtime_voice_provider_label(payload),
            str(payload.get("frontend_model") or ""),
        )
        if part
    ).lower()
    if not text:
        return False
    machine_tokens = (
        "dgx",
        "pgx",
        "spark",
        "workstation",
        "gpu",
        "rtx",
        "a100",
        "h100",
        "h200",
        "b200",
    )
    return any(token in text for token in machine_tokens)


def _realtime_voice_enabled_in_config() -> bool:
    try:
        cfg = load_config()
    except Exception:
        return False
    voice = cfg.get("voice") if isinstance(cfg, dict) else {}
    realtime = voice.get("realtime") if isinstance(voice, dict) else {}
    return isinstance(realtime, dict) and realtime.get("enabled") is True


def _check_realtime_voice_quality_targets(payload: Mapping[str, Any], issues: list[str], *, strict: bool) -> None:
    raw_targets = payload.get("quality_targets_ms")
    targets = raw_targets if isinstance(raw_targets, Mapping) else {}
    required_targets = _realtime_voice_required_quality_targets(payload)
    missing: list[str] = []
    invalid: list[str] = []
    loose: list[str] = []
    display_parts: list[str] = []

    for key, max_ms in required_targets.items():
        value = _realtime_voice_positive_int(targets.get(key))
        if value is None:
            if key in targets:
                invalid.append(key)
            else:
                missing.append(key)
            continue
        display_parts.append(f"{key}={value}ms")
        if value > max_ms:
            loose.append(f"{key}={value}ms>{max_ms}ms")

    detail = f"({', '.join(display_parts)})" if display_parts else "(none)"
    if not missing and not invalid and not loose:
        check_ok("Realtime voice latency targets", detail)
        return

    problem_parts: list[str] = []
    if missing:
        problem_parts.append(f"missing: {', '.join(missing)}")
    if invalid:
        problem_parts.append(f"invalid: {', '.join(invalid)}")
    if loose:
        problem_parts.append(f"loose: {', '.join(loose)}")
    problem = "; ".join(problem_parts)

    if strict:
        _fail_and_issue(
            "Realtime voice latency targets",
            f"({problem})",
            "Set voice.realtime.quality_targets_ms to the live-readiness targets or stricter values",
            issues,
        )
    else:
        check_warn("Realtime voice latency targets", f"({problem})")


def _realtime_voice_required_quality_targets(payload: Mapping[str, Any]) -> Mapping[str, int]:
    conversation_quality = payload.get("conversation_quality")
    if not isinstance(conversation_quality, Mapping):
        conversation_quality = {}
    if (
        str(payload.get("engine") or "").strip() == "kame_interface_oracle"
        or conversation_quality.get("kame_reflex") is True
        or str(payload.get("voice_architecture") or "").strip() == "kame_frontend_oracle"
    ):
        return _REALTIME_VOICE_KAME_TARGETS_MS
    return _REALTIME_VOICE_LIVE_TARGETS_MS


def _check_realtime_voice_production_readiness(
    payload: Mapping[str, Any],
    issues: list[str],
    *,
    strict: bool,
) -> None:
    readiness = payload.get("production_readiness")
    readiness = readiness if isinstance(readiness, Mapping) else {}
    ready = readiness.get("ready") is True
    level = str(readiness.get("level") or "unknown")
    raw_issues = readiness.get("issues")
    readiness_issues = (
        [str(issue) for issue in raw_issues if str(issue or "").strip()]
        if isinstance(raw_issues, list)
        else []
    )
    detail = f"({level})" if not readiness_issues else f"({level}: {', '.join(readiness_issues[:4])})"

    if ready:
        check_ok("Realtime voice production readiness", detail)
        return
    if strict:
        _fail_and_issue(
            "Realtime voice production readiness",
            detail,
            _realtime_voice_production_readiness_fix(readiness_issues),
            issues,
        )
    else:
        check_warn("Realtime voice production readiness", detail)
    _emit_realtime_voice_production_evidence_hint(readiness_issues)
    _emit_realtime_voice_production_review_hint(readiness_issues)


def _realtime_voice_production_readiness_fix(readiness_issues: list[str]) -> str:
    if _realtime_voice_has_production_review_issue(readiness_issues):
        return (
            "Provide a valid voice.realtime.production_review_report generated by the "
            "realtime voice production review CLI"
        )
    return "Provide a valid voice.realtime.production_evidence_report generated by realtime EN/JA smoke checks"


def _realtime_voice_has_production_evidence_issue(readiness_issues: list[str]) -> bool:
    return any(
        issue == "missing_evidence_report"
        or issue.startswith("invalid_evidence_report:")
        or issue.startswith("evidence_report:")
        for issue in readiness_issues
    )


def _emit_realtime_voice_production_evidence_hint(readiness_issues: list[str]) -> None:
    if not _realtime_voice_has_production_evidence_issue(readiness_issues):
        return
    check_info("Production evidence:")
    check_info("  python -m hermes_cli.realtime_voice_fixture_pack --output-dir ./fixtures/realtime-voice")
    check_info("  python -m hermes_cli.realtime_voice_alpha_evidence --runs 3 --apply --start-deepgram-bridge")
    check_info(
        "  python -m hermes_cli.realtime_voice_alpha_evidence "
        "--runs 3 --apply --provider elevenlabs --start-bridge"
    )
    check_info(
        "  python -m hermes_cli.realtime_voice_report "
        "./artifacts/realtime-voice-evidence/*.json --alpha --min-runs 3"
    )
    check_info(
        "  python -m hermes_cli.realtime_voice_report "
        "./artifacts/realtime-voice-evidence/*.json --alpha --min-runs 3 --apply-production-evidence"
    )
    check_info(
        "  python -m hermes_cli.realtime_voice_live_evidence "
        "--require-live-discord --require-openai-realtime"
    )


def _realtime_voice_has_production_review_issue(readiness_issues: list[str]) -> bool:
    return any(
        issue == "missing_production_review_report"
        or issue.startswith("invalid_production_review_report:")
        or issue.startswith("review_check_missing:")
        or issue == "missing_evidence"
        or issue.startswith("review_evidence_missing:")
        for issue in readiness_issues
    )


def _emit_realtime_voice_production_review_hint(readiness_issues: list[str]) -> None:
    if not _realtime_voice_has_production_review_issue(readiness_issues):
        return
    check_info("Production launch review:")
    check_info(
        "  python -m hermes_cli.realtime_voice_production_review "
        "./artifacts/realtime-voice-production-review.json --write-template --reviewer \"Realtime voice QA\""
    )
    check_info(
        "  python -m hermes_cli.realtime_voice_production_review "
        "./artifacts/realtime-voice-production-review.json --apply"
    )


def _emit_realtime_voice_live_setup_hint(payload: Mapping[str, Any]) -> None:
    quality = payload.get("conversation_quality")
    quality = quality if isinstance(quality, Mapping) else {}
    mode = str(quality.get("mode") or "")
    if quality.get("live_like") is True:
        return
    if mode not in {
        "local_turn_based",
        "turn_based_text",
        "limited_sidecar",
        "unverified_sidecar",
        "unknown",
    }:
        return
    check_info("Portable live setup:")
    check_info("  python -m hermes_cli.realtime_voice_profile --preset deepgram --apply --generate-bridge-token")
    check_info("  set DEEPGRAM_API_KEY=...")
    check_info("  python -m hermes_cli.realtime_voice_deepgram_bridge --check --strict --production-en-ja")
    check_info("  python -m hermes_cli.realtime_voice_deepgram_bridge --host 127.0.0.1 --port 8766 --production-en-ja")
    check_info("  python -m hermes_cli.realtime_voice_alpha_evidence --runs 3 --apply --start-deepgram-bridge")
    check_info("  or:")
    check_info("  python -m hermes_cli.realtime_voice_profile --preset elevenlabs --apply --generate-bridge-token")
    check_info("  set ELEVENLABS_API_KEY=... and ELEVENLABS_VOICE_ID=...")
    check_info("  python -m hermes_cli.realtime_voice_elevenlabs_bridge --check --strict --production-en-ja")
    check_info("  python -m hermes_cli.realtime_voice_elevenlabs_bridge --host 127.0.0.1 --port 8767 --production-en-ja")
    check_info("  python -m hermes_cli.realtime_voice_alpha_evidence --runs 3 --apply --provider elevenlabs --start-bridge")
    check_info("  or:")
    check_info("  python -m hermes_cli.realtime_voice_profile --preset cartesia --apply --generate-bridge-token")
    check_info("  set CARTESIA_API_KEY=... and CARTESIA_VOICE_ID=...")
    check_info("  python -m hermes_cli.realtime_voice_cartesia_bridge --check --strict --production-en-ja")
    check_info("  python -m hermes_cli.realtime_voice_cartesia_bridge --host 127.0.0.1 --port 8769 --production-en-ja")
    check_info("  python -m hermes_cli.realtime_voice_alpha_evidence --runs 3 --apply --provider cartesia --start-bridge")
    check_info("  or:")
    check_info("  python -m hermes_cli.realtime_voice_profile --preset gemini --apply")
    check_info("  set GEMINI_API_KEY=...")
    check_info("  python -m hermes_cli.realtime_voice_sidecar --host 127.0.0.1 --port 8765")
    check_info("  python -m hermes_cli.realtime_voice_live_evidence --require-live-discord --require-gemini-live")


def _realtime_voice_cli_values(value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(_realtime_voice_cli_values(item))
        return result
    text = str(value).strip()
    return [text] if text else []


def _realtime_voice_unique_values(*groups: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
    return result


def _realtime_voice_expand_alpha_smokes(
    *,
    alpha: bool,
    smoke: bool,
    audio_fixtures: list[str],
    tts_texts: list[str],
    barge_in_texts: list[str],
) -> tuple[bool, list[str], list[str], list[str]]:
    if not alpha:
        return smoke, audio_fixtures, tts_texts, barge_in_texts
    return (
        True,
        _realtime_voice_unique_values(_realtime_voice_alpha_audio_fixtures(), audio_fixtures),
        _realtime_voice_unique_values(_realtime_voice_alpha_tts_texts(), tts_texts),
        _realtime_voice_unique_values(_realtime_voice_alpha_barge_in_texts(), barge_in_texts),
    )


def _realtime_voice_smoke_report_payload(result: Any, *, kind: str, **extra: Any) -> dict[str, Any]:
    mapping = result if isinstance(result, Mapping) else {}
    if not mapping:
        try:
            from agent.realtime_voice_smoke import (
                RealtimeVoiceSidecarSmokeResult,
                realtime_voice_smoke_result_payload,
            )

            if isinstance(result, RealtimeVoiceSidecarSmokeResult):
                payload = realtime_voice_smoke_result_payload(result, kind=kind)
                payload.update(extra)
                return payload
        except Exception:
            pass
    payload = {
        "kind": kind,
        "ok": bool(mapping.get("ok", getattr(result, "ok", False))),
        "ready_ms": mapping.get("ready_ms", getattr(result, "ready_ms", None)),
        "transcript_partial_ms": mapping.get(
            "transcript_partial_ms",
            getattr(result, "transcript_partial_ms", None),
        ),
        "transcript_final_ms": mapping.get(
            "transcript_final_ms",
            getattr(result, "transcript_final_ms", None),
        ),
        "first_text_ms": mapping.get("first_text_ms", getattr(result, "first_text_ms", None)),
        "first_audio_ms": mapping.get("first_audio_ms", getattr(result, "first_audio_ms", None)),
        "barge_in_ack_ms": mapping.get("barge_in_ack_ms", getattr(result, "barge_in_ack_ms", None)),
        "final_text": str(mapping.get("final_text", getattr(result, "final_text", "")) or ""),
        "assistant_final_text": str(
            mapping.get("assistant_final_text", getattr(result, "assistant_final_text", "")) or ""
        ) or None,
        "audio_bytes": int(mapping.get("audio_bytes", getattr(result, "audio_bytes", 0)) or 0),
        "output_audio_bytes": int(
            mapping.get("output_audio_bytes", getattr(result, "output_audio_bytes", 0)) or 0
        ),
        "audio_after_barge_in_bytes": int(
            mapping.get(
                "audio_after_barge_in_bytes",
                getattr(result, "audio_after_barge_in_bytes", 0),
            )
            or 0
        ),
        "events": list(tuple(mapping.get("events", getattr(result, "events", ())) or ())),
        "first_audio_metrics": (
            dict(mapping.get("first_audio_metrics", getattr(result, "first_audio_metrics", None)) or {})
            or None
        ),
        "metrics": (
            dict(
                mapping.get("metrics")
                or mapping.get("first_audio_metrics", getattr(result, "first_audio_metrics", None))
                or {}
            )
            or None
        ),
        "route": str(mapping.get("route", getattr(result, "route", "")) or "") or None,
        "interface_input_source": str(
            mapping.get("interface_input_source", getattr(result, "interface_input_source", "")) or ""
        ) or None,
        "reflex_provider": str(
            mapping.get("reflex_provider", getattr(result, "reflex_provider", "")) or ""
        ) or None,
        "reflex_validation_error": str(
            mapping.get("reflex_validation_error", getattr(result, "reflex_validation_error", "")) or ""
        ) or None,
        "transport": str(mapping.get("transport", getattr(result, "transport", "")) or "") or None,
        "error": str(mapping.get("error", getattr(result, "error", "")) or "") or None,
    }
    payload.update(_realtime_voice_kame_report_fields(result))
    payload.update(extra)
    return payload


def _realtime_voice_kame_report_fields(result: Any) -> dict[str, Any]:
    mapping = result if isinstance(result, Mapping) else {}

    def field(name: str, default: Any = None) -> Any:
        if name in mapping:
            return mapping.get(name)
        return getattr(result, name, default)

    fields: dict[str, Any] = {}
    for key in ("turn_id", "audio_segment_ref", "evidence_bundle_id", "evidence_merge_key"):
        text = str(field(key, "") or "").strip()
        if text:
            fields[key] = text
    for key in ("audio_segment_ref_observed", "interpreter_evidence_observed", "transcript_hypotheses_labeled"):
        value = field(key, None)
        if value is not None:
            fields[key] = bool(value)
    for key in ("witness_arrival_phases", "interpreter_input_order", "interpreter_adjudication_outcomes"):
        value = field(key, None)
        if isinstance(value, (list, tuple)):
            fields[key] = [str(item) for item in value if str(item or "").strip()]
    hypotheses = field("transcript_hypotheses", None)
    if isinstance(hypotheses, (list, tuple)):
        fields["transcript_hypotheses"] = [
            dict(item)
            for item in hypotheses
            if isinstance(item, Mapping)
        ]
    promoted = field("promoted_evidence_authority", None)
    if isinstance(promoted, Mapping):
        fields["promoted_evidence_authority"] = {
            str(key): str(value)
            for key, value in promoted.items()
            if str(key or "").strip() and str(value or "").strip()
        }
    for key in ("speaker", "channel"):
        value = field(key, None)
        if isinstance(value, Mapping) and value:
            fields[key] = {
                str(item_key): item_value
                for item_key, item_value in value.items()
                if str(item_key or "").strip()
            }
    return fields


def _append_realtime_voice_smoke_report(
    reports: list[dict[str, Any]] | None,
    result: Any,
    *,
    kind: str,
    **extra: Any,
) -> None:
    if reports is None:
        return
    reports.append(_realtime_voice_smoke_report_payload(result, kind=kind, **extra))


def _discord_realtime_voice_smoke_report_payload(result: Any, **extra: Any) -> dict[str, Any]:
    mapping = result if isinstance(result, Mapping) else {}
    payload = {
        "kind": "discord_bridge",
        "ok": bool(mapping.get("ok", getattr(result, "ok", False))),
        "mode": str(mapping.get("mode", getattr(result, "mode", "")) or ""),
        "transport": str(mapping.get("transport", getattr(result, "transport", "")) or "") or None,
        "input_pcm48_bytes": int(
            mapping.get("input_pcm48_bytes", getattr(result, "input_pcm48_bytes", 0)) or 0
        ),
        "sidecar_pcm16_bytes": int(
            mapping.get("sidecar_pcm16_bytes", getattr(result, "sidecar_pcm16_bytes", 0)) or 0
        ),
        "sidecar_pcm16_first_sample": mapping.get(
            "sidecar_pcm16_first_sample",
            getattr(result, "sidecar_pcm16_first_sample", None),
        ),
        "sidecar_pcm16_checksum": int(
            mapping.get("sidecar_pcm16_checksum", getattr(result, "sidecar_pcm16_checksum", 0)) or 0
        ),
        "mixer_frames": int(mapping.get("mixer_frames", getattr(result, "mixer_frames", 0)) or 0),
        "mixer_frame_bytes": int(
            mapping.get("mixer_frame_bytes", getattr(result, "mixer_frame_bytes", 0)) or 0
        ),
        "barge_in_sent": bool(mapping.get("barge_in_sent", getattr(result, "barge_in_sent", False))),
        "speech_energy_sent": bool(
            mapping.get("speech_energy_sent", getattr(result, "speech_energy_sent", False))
        ),
        "mixer_stop_calls": int(
            mapping.get("mixer_stop_calls", getattr(result, "mixer_stop_calls", 0)) or 0
        ),
        "sidecar_closed": bool(mapping.get("sidecar_closed", getattr(result, "sidecar_closed", False))),
        "shutdown_elapsed_ms": mapping.get(
            "shutdown_elapsed_ms",
            getattr(result, "shutdown_elapsed_ms", None),
        ),
        "shutdown_bounded": bool(
            mapping.get("shutdown_bounded", getattr(result, "shutdown_bounded", False))
        ),
        "shutdown_timed_out": bool(
            mapping.get("shutdown_timed_out", getattr(result, "shutdown_timed_out", False))
        ),
        "events": list(tuple(mapping.get("events", getattr(result, "events", ())) or ())),
        "error": str(mapping.get("error", getattr(result, "error", "")) or "") or None,
    }
    payload.update(extra)
    return payload


def _append_discord_realtime_voice_smoke_report(
    reports: list[dict[str, Any]] | None,
    result: Any,
    **extra: Any,
) -> None:
    if reports is None:
        return
    reports.append(_discord_realtime_voice_smoke_report_payload(result, **extra))


def _discord_voice_live_probe_report_payload(result: Any, **extra: Any) -> dict[str, Any]:
    mapping = result if isinstance(result, Mapping) else {}

    def _int_field(name: str) -> int:
        return int(mapping.get(name, getattr(result, name, 0)) or 0)

    def _bool_field(name: str) -> bool:
        return bool(mapping.get(name, getattr(result, name, False)))

    def _str_field(name: str) -> str:
        return str(mapping.get(name, getattr(result, name, "")) or "")

    payload = {
        "kind": "discord_live_probe",
        "ok": _bool_field("ok"),
        "guild_name": _str_field("guild_name"),
        "voice_channel_name": _str_field("voice_channel_name"),
        "connect_perm": _bool_field("connect_perm"),
        "speak_perm": _bool_field("speak_perm"),
        "members_before": _int_field("members_before"),
        "connected": _bool_field("connected"),
        "opus_loaded": _bool_field("opus_loaded"),
        "accepted_audio_source": _bool_field("accepted_audio_source"),
        "played": _bool_field("played"),
        "playing_during_probe": _bool_field("playing_during_probe"),
        "receiver_started": _bool_field("receiver_started"),
        "receiver_frames": _int_field("receiver_frames"),
        "receiver_speech_start": _int_field("receiver_speech_start"),
        "inbound_observed": _bool_field("inbound_observed"),
        "members_after": _int_field("members_after"),
        "disconnected": _bool_field("disconnected"),
        "require_inbound": _bool_field("require_inbound"),
        "wait_seconds": float(mapping.get("wait_seconds", getattr(result, "wait_seconds", 0.0)) or 0.0),
        "latency_metrics_ms": dict(mapping.get("latency_metrics_ms", getattr(result, "latency_metrics_ms", {})) or {}),
        "failure_reason": _str_field("failure_reason") or None,
        "error": _str_field("error") or None,
    }
    payload.update(extra)
    return payload


def _append_discord_voice_live_probe_report(
    reports: list[dict[str, Any]] | None,
    result: Any,
    **extra: Any,
) -> None:
    if reports is None:
        return
    reports.append(_discord_voice_live_probe_report_payload(result, **extra))


def _write_realtime_voice_report(path: str | os.PathLike[str], reports: list[dict[str, Any]]) -> None:
    report_path = Path(path).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _realtime_voice_evidence_manifest_payload() -> dict[str, Any]:
    """Return sanitized realtime voice stack context for smoke evidence reports."""
    try:
        status = dict(_realtime_voice_status_payload())
    except Exception as exc:
        return {
            "kind": "manifest",
            "ok": False,
            "schema_version": 1,
            "error": f"could not read realtime voice status: {exc}",
        }

    sidecar = status.get("sidecar") if isinstance(status.get("sidecar"), Mapping) else {}
    sidecar_health = _realtime_voice_manifest_health(sidecar.get("health"))
    production = status.get("production_readiness") if isinstance(status.get("production_readiness"), Mapping) else {}
    evidence = production.get("evidence") if isinstance(production.get("evidence"), Mapping) else {}
    manifest = {
        "kind": "manifest",
        "ok": True,
        "schema_version": 1,
        "source": "hermes doctor --realtime-voice-report",
        "run_id": uuid.uuid4().hex,
        "collected_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "engine": str(status.get("engine") or ""),
        "frontend_provider": str(status.get("frontend_provider") or ""),
        "frontend_model": str(status.get("frontend_model") or ""),
        "interface_temperature": status.get("interface_temperature"),
        "interface_max_output_tokens": status.get("interface_max_output_tokens"),
        "interface_timeout_seconds": status.get("interface_timeout_seconds"),
        "interface_audio_input": str(status.get("interface_audio_input") or ""),
        "interface_max_audio_seconds": status.get("interface_max_audio_seconds"),
        "asr_mode": str(status.get("asr_mode") or ""),
        "preferred_local_oracle_model": str(status.get("preferred_local_oracle_model") or ""),
        "available": status.get("available") is True,
        "unavailable_reason": str(status.get("unavailable_reason") or "") or None,
        "require_live_like": status.get("require_live_like") is True,
        "quality_targets_ms": status.get("quality_targets_ms") if isinstance(status.get("quality_targets_ms"), Mapping) else {},
        "conversation_quality": (
            status.get("conversation_quality")
            if isinstance(status.get("conversation_quality"), Mapping)
            else {}
        ),
        "language_support": (
            status.get("language_support")
            if isinstance(status.get("language_support"), Mapping)
            else {}
        ),
        "sidecar": {
            "mode": str(sidecar.get("mode") or "none"),
            "healthy": sidecar.get("healthy"),
            "health": sidecar_health,
        },
        "production_readiness": {
            "ready": production.get("ready") is True,
            "level": str(production.get("level") or ""),
            "issues": [str(issue) for issue in production.get("issues", [])]
            if isinstance(production.get("issues"), list)
            else [],
            "evidence": {
                "runs": evidence.get("runs"),
                "min_runs": evidence.get("min_runs"),
                "verified": evidence.get("verified") is True,
            },
        },
    }
    return _strip_none_values(manifest)


def _append_realtime_voice_evidence_manifest(reports: list[dict[str, Any]] | None) -> None:
    if reports is None:
        return
    reports[:] = [entry for entry in reports if entry.get("kind") != "manifest"]
    reports.insert(0, _realtime_voice_evidence_manifest_payload())


def _realtime_voice_manifest_health(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    frontend = value.get("frontend") if isinstance(value.get("frontend"), Mapping) else {}
    capabilities = value.get("capabilities") if isinstance(value.get("capabilities"), Mapping) else {}
    local = value.get("local") if isinstance(value.get("local"), Mapping) else {}
    result: dict[str, Any] = {
        "ok": value.get("ok") is True,
        "kind": str(value.get("kind") or "") or None,
        "frontend": {
            "provider": str(frontend.get("provider") or "") or None,
            "model": str(frontend.get("model") or "") or None,
            "tts_model_languages": _realtime_voice_manifest_metadata_list(
                frontend.get("tts_model_languages")
            ),
            "languages": _realtime_voice_manifest_metadata_list(frontend.get("languages")),
            "scripts": _realtime_voice_manifest_metadata_list(frontend.get("scripts")),
        },
        "capabilities": {
            "utterance_stt": capabilities.get("utterance_stt") is True,
            "streaming_stt": capabilities.get("streaming_stt") is True,
            "tts": capabilities.get("tts") is True,
            "native_s2s": capabilities.get("native_s2s") is True,
            "vllm_audio_frontend": capabilities.get("vllm_audio_frontend") is True,
            "input_languages": _realtime_voice_manifest_metadata_list(capabilities.get("input_languages")),
            "output_languages": _realtime_voice_manifest_metadata_list(capabilities.get("output_languages")),
            "scripts": _realtime_voice_manifest_metadata_list(capabilities.get("scripts")),
        },
        "local": {
            "stt": local.get("stt") is True,
            "tts": local.get("tts") is True,
        },
    }
    for key in ("streaming_stt_bridge", "streaming_tts", "streaming_tts_bridge"):
        if key in capabilities:
            result["capabilities"][key] = capabilities.get(key) is True
    stt_bridge = frontend.get("streaming_stt_bridge")
    if isinstance(stt_bridge, Mapping):
        result["frontend"]["streaming_stt_bridge"] = {
            "configured": stt_bridge.get("configured") is True,
            "healthy": stt_bridge.get("healthy") is True,
        }
    tts_bridge = frontend.get("streaming_tts_bridge")
    if isinstance(tts_bridge, Mapping):
        result["frontend"]["streaming_tts_bridge"] = {
            "configured": tts_bridge.get("configured") is True,
            "healthy": tts_bridge.get("healthy") is True,
            "model": str(tts_bridge.get("model") or "") or None,
        }
    return _strip_none_values(result)


def _realtime_voice_manifest_metadata_list(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value] if isinstance(value, str) else []
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or len(text) > 64:
            continue
        if not all(char.isalnum() or char in {"-", "_", "."} for char in text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _strip_none_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_none_values(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_strip_none_values(item) for item in value]
    return value


def _realtime_voice_smoke_config():
    from agent.realtime_voice import RealtimeVoiceASRMode, RealtimeVoiceEngineKind, RealtimeVoiceSessionConfig, VoiceAudioCodec
    from hermes_cli.web_server import (
        _positive_float_config,
        _positive_int_config,
        _realtime_voice_config_dict,
        _realtime_voice_conversation_quality_payload,
        _realtime_voice_language_support_payload,
        _realtime_voice_metrics_policy_payload,
        _realtime_voice_quality_targets_payload,
        _realtime_voice_routing_policy_payload,
        _realtime_voice_sidecar_base_url,
        _realtime_voice_sidecar_token,
        _realtime_voice_turn_acknowledgement_payload,
        _truthy_config,
    )

    realtime = _realtime_voice_config_dict()
    base_url = _realtime_voice_sidecar_base_url(realtime)
    if not base_url:
        raise RuntimeError("realtime voice smoke requires voice.realtime.sidecar_base_url")
    engine = str(realtime.get("engine") or RealtimeVoiceEngineKind.TEXT_ORACLE_TTS.value)
    sidecar_token = _realtime_voice_sidecar_token(realtime)
    routing_policy = _realtime_voice_routing_policy_payload(realtime)
    metrics_policy = _realtime_voice_metrics_policy_payload(realtime)
    return RealtimeVoiceSessionConfig(
        session_id="voice-doctor-smoke",
        engine=RealtimeVoiceEngineKind(engine),
        input_codec=VoiceAudioCodec(str(realtime.get("input_codec") or VoiceAudioCodec.WEBM_OPUS.value)),
        output_codec=VoiceAudioCodec(str(realtime.get("output_codec") or VoiceAudioCodec.OPUS.value)),
        input_buffer_limit_bytes=_positive_int_config(
            realtime.get("input_buffer_limit_bytes"),
            default=8 * 1024 * 1024,
        ),
        frontend_provider=str(realtime.get("frontend_provider") or "") or None,
        frontend_model=str(realtime.get("frontend_model") or "") or None,
        interface_temperature=_positive_float_config(
            realtime.get("interface_temperature"),
            default=0.2,
        ),
        interface_max_output_tokens=_positive_int_config(
            realtime.get("interface_max_output_tokens"),
            default=160,
        ),
        interface_timeout_seconds=_positive_float_config(
            realtime.get("interface_timeout_seconds"),
            default=0.8,
        ),
        interface_max_audio_seconds=_positive_float_config(
            realtime.get("interface_max_audio_seconds"),
            default=30.0,
        ),
        interface_audio_input=str(realtime.get("interface_audio_input") or "") or None,
        asr_mode=RealtimeVoiceASRMode(str(realtime.get("asr_mode") or RealtimeVoiceASRMode.ON_ESCALATION.value)),
        preferred_local_oracle_model=str(realtime.get("preferred_local_oracle_model") or "") or None,
        oracle_timeout_seconds=_positive_float_config(
            realtime.get("oracle_timeout_seconds"),
            default=60.0,
        ),
        max_spoken_sentences=_positive_int_config(
            realtime.get("max_spoken_sentences"),
            default=2,
        ),
        tts_provider=str(realtime.get("tts_provider") or "") or None,
        sidecar_base_url=str(base_url or "") or None,
        sidecar_token=str(sidecar_token or "") or None,
        sidecar_connect_timeout_seconds=_positive_float_config(
            realtime.get("sidecar_connect_timeout_seconds"),
            default=10.0,
        ),
        metadata={
            "source": "doctor_smoke",
            "transport": str(realtime.get("transport") or "doctor_smoke"),
            "voice_architecture": "kame_frontend_oracle",
            "frontend_role": "low_latency_voice_interface",
            "oracle_role": "hermes_backend_oracle",
            "frontend_provider": str(realtime.get("frontend_provider") or "") or None,
            "frontend_model": str(realtime.get("frontend_model") or "") or None,
            "interface_temperature": _positive_float_config(
                realtime.get("interface_temperature"),
                default=0.2,
            ),
            "interface_max_output_tokens": _positive_int_config(
                realtime.get("interface_max_output_tokens"),
                default=160,
            ),
            "interface_timeout_seconds": _positive_float_config(
                realtime.get("interface_timeout_seconds"),
                default=0.8,
            ),
            "interface_max_audio_seconds": _positive_float_config(
                realtime.get("interface_max_audio_seconds"),
                default=30.0,
            ),
            "interface_audio_input": str(realtime.get("interface_audio_input") or "") or None,
            "asr_mode": str(realtime.get("asr_mode") or "") or None,
            "preferred_local_oracle_model": str(realtime.get("preferred_local_oracle_model") or "") or None,
            "oracle_timeout_seconds": _positive_float_config(
                realtime.get("oracle_timeout_seconds"),
                default=60.0,
            ),
            "max_spoken_sentences": _positive_int_config(
                realtime.get("max_spoken_sentences"),
                default=2,
            ),
            "language_support": _realtime_voice_language_support_payload(realtime),
            "quality_targets_ms": _realtime_voice_quality_targets_payload(realtime),
            "routing": routing_policy,
            "metrics": metrics_policy,
            "turn_acknowledgement": _realtime_voice_turn_acknowledgement_payload(realtime),
            "conversation_quality": _realtime_voice_conversation_quality_payload(
                engine=engine,
                base_url=str(base_url or ""),
                health_payload=None,
            ),
            "require_live_like": _truthy_config(realtime.get("require_live_like"), default=False),
        },
    )


def _run_realtime_voice_sidecar_smoke_sync(config, *, timeout_seconds: float):
    from agent.realtime_voice_smoke import run_realtime_voice_sidecar_smoke

    result = asyncio.run(
        run_realtime_voice_sidecar_smoke(
            config,
            timeout_seconds=timeout_seconds,
        )
    )
    return _realtime_voice_result_with_transport(result, config)


def _run_realtime_voice_sidecar_audio_smoke_sync(
    config,
    *,
    audio_fixture_path: str,
    audio_codec: str,
    timeout_seconds: float,
):
    from agent.realtime_voice import VoiceAudioCodec
    from agent.realtime_voice_smoke import run_realtime_voice_sidecar_smoke

    audio_path = Path(audio_fixture_path).expanduser()
    if not audio_path.exists() or not audio_path.is_file():
        raise RuntimeError(f"audio fixture not found: {audio_fixture_path}")
    audio = audio_path.read_bytes()
    if not audio:
        raise RuntimeError(f"audio fixture is empty: {audio_fixture_path}")

    result = asyncio.run(
        run_realtime_voice_sidecar_smoke(
            config,
            audio=audio,
            audio_codec=VoiceAudioCodec(str(audio_codec)),
            timeout_seconds=timeout_seconds,
        )
    )
    return _realtime_voice_result_with_transport(result, config)


def _run_realtime_voice_sidecar_tts_smoke_sync(
    config,
    *,
    text: str,
    metadata: Mapping[str, str] | None = None,
    timeout_seconds: float,
):
    from agent.realtime_voice_smoke import run_realtime_voice_sidecar_tts_smoke

    result = asyncio.run(
        run_realtime_voice_sidecar_tts_smoke(
            config,
            text=text,
            metadata=metadata,
            timeout_seconds=timeout_seconds,
        )
    )
    return _realtime_voice_result_with_transport(result, config)


def _run_realtime_voice_sidecar_barge_in_smoke_sync(
    config,
    *,
    text: str,
    timeout_seconds: float,
):
    from agent.realtime_voice_smoke import run_realtime_voice_sidecar_barge_in_smoke

    result = asyncio.run(
        run_realtime_voice_sidecar_barge_in_smoke(
            config,
            text=text,
            timeout_seconds=timeout_seconds,
        )
    )
    return _realtime_voice_result_with_transport(result, config)


def _run_realtime_voice_session_turn_smoke_sync(
    config,
    *,
    text: str,
    timeout_seconds: float,
):
    from agent.realtime_voice_smoke import (
        realtime_voice_smoke_text_metadata,
        run_realtime_voice_session_turn_smoke,
    )

    result = asyncio.run(
        run_realtime_voice_session_turn_smoke(
            config,
            answer=text,
            metadata=realtime_voice_smoke_text_metadata(text),
            transcript=text,
            timeout_seconds=timeout_seconds,
        )
    )
    return _realtime_voice_result_with_transport(result, config)


def _run_realtime_voice_session_audio_smoke_sync(
    config,
    *,
    audio_fixture_path: str,
    audio_codec: str,
    timeout_seconds: float,
):
    from agent.realtime_voice import VoiceAudioCodec
    from agent.realtime_voice_smoke import run_realtime_voice_session_audio_smoke
    from agent.realtime_voice_smoke_report import ALPHA_REQUIRED_AUDIO_FIXTURE_TEXTS

    audio_path = Path(audio_fixture_path).expanduser()
    if not audio_path.exists() or not audio_path.is_file():
        raise RuntimeError(f"audio fixture not found: {audio_fixture_path}")
    audio = audio_path.read_bytes()
    if not audio:
        raise RuntimeError(f"audio fixture is empty: {audio_fixture_path}")

    expected_text = ALPHA_REQUIRED_AUDIO_FIXTURE_TEXTS.get(audio_fixture_path, "Hello from Hermes.")
    result = asyncio.run(
        run_realtime_voice_session_audio_smoke(
            config,
            audio=audio,
            audio_codec=VoiceAudioCodec(str(audio_codec)),
            answer=expected_text,
            timeout_seconds=timeout_seconds,
        )
    )
    return _realtime_voice_result_with_transport(result, config)


def _run_discord_realtime_voice_smoke_sync():
    from hermes_cli.discord_realtime_voice_smoke import run_discord_realtime_voice_smoke

    return asyncio.run(run_discord_realtime_voice_smoke())


def _run_discord_voice_live_probe_sync(
    *,
    wait_seconds: float,
    require_inbound: bool,
    voice_channel_id: str = "",
    voice_channel_name: str = "",
):
    from hermes_cli.discord_voice_live_probe import run_discord_voice_live_probe

    args = Namespace(
        guild_id=os.environ.get("DISCORD_GUILD_ID", ""),
        text_channel_id=os.environ.get("DISCORD_HOME_CHANNEL", ""),
        voice_channel_id=str(voice_channel_id or os.environ.get("DISCORD_VOICE_CHANNEL_ID", "")),
        voice_channel_name=str(
            voice_channel_name or os.environ.get("DISCORD_VOICE_CHANNEL_NAME", "General")
        ),
        wait_seconds=max(0.0, float(wait_seconds or 0.0)),
        require_inbound=bool(require_inbound),
    )
    return asyncio.run(run_discord_voice_live_probe(args))


def _realtime_voice_result_with_transport(result: Any, config: Any) -> Any:
    metadata = getattr(config, "metadata", None)
    transport = ""
    if isinstance(metadata, Mapping):
        transport = str(metadata.get("transport") or "").strip()
    if not transport:
        return result
    try:
        from agent.realtime_voice_smoke import RealtimeVoiceSidecarSmokeResult

        if isinstance(result, RealtimeVoiceSidecarSmokeResult):
            return replace(result, transport=transport)
    except Exception:
        pass
    return result


def _check_discord_realtime_voice_smoke(
    issues: list[str],
    *,
    reports: list[dict[str, Any]] | None = None,
) -> None:
    try:
        result = _run_discord_realtime_voice_smoke_sync()
    except Exception as exc:
        _append_discord_realtime_voice_smoke_report(
            reports,
            {"ok": False, "error": f"failed: {exc}"},
        )
        _fail_and_issue(
            "Discord realtime voice bridge smoke",
            f"(failed: {exc})",
            f"Fix Discord realtime voice bridge smoke failure: {exc}",
            issues,
        )
        return

    _append_discord_realtime_voice_smoke_report(reports, result)
    if result.ok:
        check_ok(
            "Discord realtime voice bridge smoke",
            (
                f"(pcm48={result.input_pcm48_bytes}B, "
                f"pcm16={result.sidecar_pcm16_bytes}B, "
                f"mixer_frames={result.mixer_frames}, "
                f"barge_in={_realtime_voice_bool(result.barge_in_sent)})"
            ),
        )
        return

    detail = result.error or "discord realtime bridge smoke failed"
    if result.events:
        detail = f"{detail}; events={','.join(result.events)}"
    _fail_and_issue(
        "Discord realtime voice bridge smoke",
        f"({detail})",
        f"Fix Discord realtime voice bridge smoke failure: {detail}",
        issues,
    )


def _check_discord_voice_live_probe(
    issues: list[str],
    *,
    reports: list[dict[str, Any]] | None = None,
    wait_seconds: float = 2.0,
    require_inbound: bool = False,
    voice_channel_id: str = "",
    voice_channel_name: str = "",
) -> None:
    try:
        result = _run_discord_voice_live_probe_sync(
            wait_seconds=wait_seconds,
            require_inbound=require_inbound,
            voice_channel_id=voice_channel_id,
            voice_channel_name=voice_channel_name,
        )
    except Exception as exc:
        _append_discord_voice_live_probe_report(
            reports,
            {
                "ok": False,
                "error": f"failed: {exc}",
                "wait_seconds": wait_seconds,
                "require_inbound": require_inbound,
            },
        )
        _fail_and_issue(
            "Discord live voice probe",
            f"(failed: {exc})",
            f"Fix Discord live voice probe failure: {exc}",
            issues,
        )
        return

    _append_discord_voice_live_probe_report(reports, result)
    if result.ok:
        inbound = "yes" if result.inbound_observed else "no"
        check_ok(
            "Discord live voice probe",
            (
                f"({result.guild_name} / {result.voice_channel_name}, "
                f"members={result.members_before}->{result.members_after}, "
                f"receiver_frames={result.receiver_frames}, inbound={inbound})"
            ),
        )
        return

    detail = result.error or result.failure_reason or "discord live voice probe failed"
    fix = f"Fix Discord live voice probe failure: {detail}"
    if getattr(result, "failure_reason", "") == "inbound_required_but_no_other_members":
        fix = "Rerun while a human or controlled second Discord client is speaking in the voice channel"
    elif getattr(result, "failure_reason", "") == "inbound_required_but_no_frames":
        fix = "Verify Discord receiver callback wiring and rerun while inbound speech is active"
    _fail_and_issue(
        "Discord live voice probe",
        f"({detail})",
        fix,
        issues,
    )


def _check_realtime_voice_sidecar_smoke(
    issues: list[str],
    *,
    reports: list[dict[str, Any]] | None = None,
) -> None:
    try:
        config = _realtime_voice_smoke_config()
    except Exception as exc:
        _append_realtime_voice_smoke_report(
            reports,
            {"ok": False, "error": f"could not build config: {exc}"},
            kind="protocol",
        )
        _fail_and_issue(
            "Realtime voice sidecar smoke",
            f"(could not build config: {exc})",
            f"Fix realtime voice smoke configuration: {exc}",
            issues,
        )
        return

    timeout_seconds = min(max(float(config.sidecar_connect_timeout_seconds or 10.0), 1.0), 15.0)
    try:
        result = _run_realtime_voice_sidecar_smoke_sync(config, timeout_seconds=timeout_seconds)
    except Exception as exc:
        _append_realtime_voice_smoke_report(
            reports,
            {"ok": False, "error": f"failed: {exc}"},
            kind="protocol",
            timeout_seconds=timeout_seconds,
        )
        _fail_and_issue(
            "Realtime voice sidecar smoke",
            f"(failed: {exc})",
            f"Fix realtime voice sidecar protocol smoke failure: {exc}",
            issues,
        )
        return
    _append_realtime_voice_smoke_report(
        reports,
        result,
        kind="protocol",
        timeout_seconds=timeout_seconds,
    )
    if result.ok:
        ready = f"{result.ready_ms}ms" if result.ready_ms is not None else "unknown"
        final = f"{result.transcript_final_ms}ms" if result.transcript_final_ms is not None else "unknown"
        check_ok("Realtime voice sidecar smoke", f"(ready={ready}, transcript.final={final})")
        return

    detail = result.error or "sidecar smoke failed"
    if result.events:
        detail = f"{detail}; events={','.join(result.events)}"
    _fail_and_issue(
        "Realtime voice sidecar smoke",
        f"({detail})",
        f"Fix realtime voice sidecar protocol smoke failure: {detail}",
        issues,
    )


def _check_realtime_voice_session_turn_smoke(
    issues: list[str],
    *,
    text: str,
    reports: list[dict[str, Any]] | None = None,
) -> None:
    try:
        config = _realtime_voice_smoke_config()
    except Exception as exc:
        _append_realtime_voice_smoke_report(
            reports,
            {"ok": False, "error": f"could not build config: {exc}"},
            kind="session_turn",
            text=text,
        )
        _fail_and_issue(
            "Realtime voice session turn smoke",
            f"(could not build config: {exc})",
            f"Fix realtime voice session turn smoke configuration: {exc}",
            issues,
        )
        return

    timeout_seconds = min(max(float(config.sidecar_connect_timeout_seconds or 10.0), 1.0), 30.0)
    try:
        result = _run_realtime_voice_session_turn_smoke_sync(
            config,
            text=text,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        _append_realtime_voice_smoke_report(
            reports,
            {"ok": False, "error": f"failed: {exc}"},
            kind="session_turn",
            text=text,
            timeout_seconds=timeout_seconds,
        )
        _fail_and_issue(
            "Realtime voice session turn smoke",
            f"(failed: {exc})",
            f"Fix realtime voice session turn smoke failure: {exc}",
            issues,
        )
        return

    first_text_target_ms = _quality_target_from_config(config, "final_transcript_to_first_text_ms", 500)
    first_audio_target_ms = _quality_target_from_config(config, "final_transcript_to_first_audio_ms", 900)
    from agent.realtime_voice_smoke import realtime_voice_smoke_text_metadata

    _append_realtime_voice_smoke_report(
        reports,
        result,
        kind="session_turn",
        text=text,
        **realtime_voice_smoke_text_metadata(text),
        first_text_target_ms=first_text_target_ms,
        first_audio_target_ms=first_audio_target_ms,
        timeout_seconds=timeout_seconds,
    )
    if (
        result.ok
        and result.first_text_ms is not None
        and result.first_audio_ms is not None
        and result.first_text_ms <= first_text_target_ms
        and result.first_audio_ms <= first_audio_target_ms
    ):
        check_ok(
            "Realtime voice session turn smoke",
            (
                f"(first_text={result.first_text_ms}ms <= {first_text_target_ms}ms, "
                f"first_audio={result.first_audio_ms}ms <= {first_audio_target_ms}ms)"
            ),
        )
        return

    detail = getattr(result, "error", "") or "session turn smoke failed"
    if result.ok and result.first_text_ms is not None and result.first_text_ms > first_text_target_ms:
        detail = f"first_text={result.first_text_ms}ms exceeds target {first_text_target_ms}ms"
    elif result.ok and result.first_audio_ms is not None and result.first_audio_ms > first_audio_target_ms:
        detail = f"first_audio={result.first_audio_ms}ms exceeds target {first_audio_target_ms}ms"
    events = tuple(getattr(result, "events", ()) or ())
    if events:
        detail = f"{detail}; events={','.join(events)}"
    _fail_and_issue(
        "Realtime voice session turn smoke",
        f"({detail})",
        f"Fix realtime voice session turn smoke failure: {detail}",
        issues,
    )


def _check_realtime_voice_barge_in_smoke(
    issues: list[str],
    *,
    text: str,
    reports: list[dict[str, Any]] | None = None,
) -> None:
    try:
        config = _realtime_voice_smoke_config()
    except Exception as exc:
        _append_realtime_voice_smoke_report(
            reports,
            {"ok": False, "error": f"could not build config: {exc}"},
            kind="barge_in",
            text=text,
        )
        _fail_and_issue(
            "Realtime voice barge-in smoke",
            f"(could not build config: {exc})",
            f"Fix realtime voice barge-in smoke configuration: {exc}",
            issues,
        )
        return

    timeout_seconds = min(max(float(config.sidecar_connect_timeout_seconds or 10.0), 1.0), 30.0)
    try:
        result = _run_realtime_voice_sidecar_barge_in_smoke_sync(
            config,
            text=text,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        _append_realtime_voice_smoke_report(
            reports,
            {"ok": False, "error": f"failed: {exc}"},
            kind="barge_in",
            text=text,
            timeout_seconds=timeout_seconds,
        )
        _fail_and_issue(
            "Realtime voice barge-in smoke",
            f"(failed: {exc})",
            f"Fix realtime voice barge-in smoke failure: {exc}",
            issues,
        )
        return

    target_ms = _quality_target_from_config(config, "barge_in_ack_ms", 150)
    _append_realtime_voice_smoke_report(
        reports,
        result,
        kind="barge_in",
        text=text,
        target_ms=target_ms,
        timeout_seconds=timeout_seconds,
    )
    if result.ok and result.barge_in_ack_ms is not None and result.barge_in_ack_ms <= target_ms:
        check_ok(
            "Realtime voice barge-in smoke",
            f"(barge_in_ack={result.barge_in_ack_ms}ms <= {target_ms}ms)",
        )
        return

    detail = getattr(result, "error", "") or "barge-in smoke failed"
    if result.ok and result.barge_in_ack_ms is not None:
        detail = f"barge_in_ack={result.barge_in_ack_ms}ms exceeds target {target_ms}ms"
    events = tuple(getattr(result, "events", ()) or ())
    if events:
        detail = f"{detail}; events={','.join(events)}"
    _fail_and_issue(
        "Realtime voice barge-in smoke",
        f"({detail})",
        f"Fix realtime voice barge-in smoke failure: {detail}",
        issues,
    )


def _check_realtime_voice_tts_smoke(
    issues: list[str],
    *,
    text: str,
    reports: list[dict[str, Any]] | None = None,
) -> None:
    try:
        config = _realtime_voice_smoke_config()
    except Exception as exc:
        _append_realtime_voice_smoke_report(
            reports,
            {"ok": False, "error": f"could not build config: {exc}"},
            kind="tts",
            text=text,
        )
        _fail_and_issue(
            "Realtime voice TTS smoke",
            f"(could not build config: {exc})",
            f"Fix realtime voice TTS smoke configuration: {exc}",
            issues,
        )
        return

    timeout_seconds = min(max(float(config.sidecar_connect_timeout_seconds or 10.0), 1.0), 30.0)
    from agent.realtime_voice_smoke import realtime_voice_smoke_text_metadata

    metadata = realtime_voice_smoke_text_metadata(text)
    try:
        result = _run_realtime_voice_sidecar_tts_smoke_sync(
            config,
            text=text,
            metadata=metadata,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        _append_realtime_voice_smoke_report(
            reports,
            {"ok": False, "error": f"failed: {exc}"},
            kind="tts",
            text=text,
            **metadata,
            timeout_seconds=timeout_seconds,
        )
        _fail_and_issue(
            "Realtime voice TTS smoke",
            f"(failed: {exc})",
            f"Fix realtime voice TTS smoke failure: {exc}",
            issues,
        )
        return

    target_ms = _quality_target_from_config(config, "final_transcript_to_first_audio_ms", 900)
    _append_realtime_voice_smoke_report(
        reports,
        result,
        kind="tts",
        text=text,
        **metadata,
        target_ms=target_ms,
        timeout_seconds=timeout_seconds,
    )
    if result.ok and result.first_audio_ms is not None and result.first_audio_ms <= target_ms:
        check_ok(
            "Realtime voice TTS smoke",
            (
                f"(audio.bytes={result.output_audio_bytes}, first_audio={result.first_audio_ms}ms"
                f" <= {target_ms}ms)"
            ),
        )
        return

    detail = getattr(result, "error", "") or "TTS smoke failed"
    if result.ok and result.first_audio_ms is not None:
        detail = f"first_audio={result.first_audio_ms}ms exceeds target {target_ms}ms"
    events = tuple(getattr(result, "events", ()) or ())
    if events:
        detail = f"{detail}; events={','.join(events)}"
    _fail_and_issue(
        "Realtime voice TTS smoke",
        f"({detail})",
        f"Fix realtime voice TTS smoke failure: {detail}",
        issues,
    )


def _check_realtime_voice_session_audio_smoke(
    issues: list[str],
    *,
    audio_fixture_path: str,
    audio_codec: str,
    reports: list[dict[str, Any]] | None = None,
) -> None:
    try:
        config = _realtime_voice_smoke_config()
    except Exception as exc:
        _append_realtime_voice_smoke_report(
            reports,
            {"ok": False, "error": f"could not build config: {exc}"},
            kind="audio_session",
            fixture=audio_fixture_path,
            codec=audio_codec,
        )
        _fail_and_issue(
            "Realtime voice audio session smoke",
            f"(could not build config: {exc})",
            f"Fix realtime voice audio session smoke configuration: {exc}",
            issues,
        )
        return

    timeout_seconds = min(max(float(config.sidecar_connect_timeout_seconds or 10.0), 1.0), 30.0)
    try:
        result = _run_realtime_voice_session_audio_smoke_sync(
            config,
            audio_fixture_path=audio_fixture_path,
            audio_codec=audio_codec,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        _append_realtime_voice_smoke_report(
            reports,
            {"ok": False, "error": f"failed: {exc}"},
            kind="audio_session",
            fixture=audio_fixture_path,
            codec=audio_codec,
            timeout_seconds=timeout_seconds,
        )
        _fail_and_issue(
            "Realtime voice audio session smoke",
            f"(failed: {exc})",
            f"Fix realtime voice audio session smoke failure: {exc}",
            issues,
        )
        return

    partial_target_ms = _quality_target_from_config(config, "audio_to_partial_transcript_ms", 300)
    first_text_target_ms = _quality_target_from_config(config, "final_transcript_to_first_text_ms", 500)
    first_audio_target_ms = _quality_target_from_config(config, "final_transcript_to_first_audio_ms", 900)
    _append_realtime_voice_smoke_report(
        reports,
        result,
        kind="audio_session",
        fixture=audio_fixture_path,
        codec=audio_codec,
        target_ms=partial_target_ms,
        first_text_target_ms=first_text_target_ms,
        first_audio_target_ms=first_audio_target_ms,
        timeout_seconds=timeout_seconds,
    )
    kame_native_audio_reflex = _realtime_voice_smoke_result_uses_kame_native_audio_reflex(result)
    if (
        result.ok
        and (
            result.transcript_partial_ms is not None
            or kame_native_audio_reflex
        )
        and result.first_text_ms is not None
        and result.first_audio_ms is not None
        and (
            result.transcript_partial_ms is None
            or result.transcript_partial_ms <= partial_target_ms
        )
        and result.first_text_ms <= first_text_target_ms
        and result.first_audio_ms <= first_audio_target_ms
    ):
        partial_detail = (
            "native_audio_reflex"
            if result.transcript_partial_ms is None
            else f"transcript.partial={result.transcript_partial_ms}ms <= {partial_target_ms}ms"
        )
        check_ok(
            "Realtime voice audio session smoke",
            (
                f"(bytes={result.audio_bytes}, {partial_detail}, first_text={result.first_text_ms}ms "
                f"<= {first_text_target_ms}ms, first_audio={result.first_audio_ms}ms "
                f"<= {first_audio_target_ms}ms)"
            ),
        )
        return

    detail = getattr(result, "error", "") or "audio session smoke failed"
    if result.ok and result.transcript_partial_ms is None and not kame_native_audio_reflex:
        detail = "missing transcript.partial before transcript.final"
    elif result.ok and result.transcript_partial_ms is not None and result.transcript_partial_ms > partial_target_ms:
        detail = f"transcript.partial={result.transcript_partial_ms}ms exceeds target {partial_target_ms}ms"
    elif result.ok and result.first_text_ms is not None and result.first_text_ms > first_text_target_ms:
        detail = f"first_text={result.first_text_ms}ms exceeds target {first_text_target_ms}ms"
    elif result.ok and result.first_audio_ms is not None and result.first_audio_ms > first_audio_target_ms:
        detail = f"first_audio={result.first_audio_ms}ms exceeds target {first_audio_target_ms}ms"
    events = tuple(getattr(result, "events", ()) or ())
    if events:
        detail = f"{detail}; events={','.join(events)}"
    _fail_and_issue(
        "Realtime voice audio session smoke",
        f"({detail})",
        f"Fix realtime voice audio session smoke failure: {detail}",
        issues,
    )


def _realtime_voice_smoke_result_uses_kame_native_audio_reflex(result: Any) -> bool:
    mapping = result if isinstance(result, Mapping) else {}
    route = str(mapping.get("route", getattr(result, "route", "")) or "").strip().lower()
    input_source = str(
        mapping.get("interface_input_source", getattr(result, "interface_input_source", "")) or ""
    ).strip().lower()
    return route in {"local", "defer", "oracle_direct", "reject_or_clarify"} and input_source == "native_audio"


def _quality_target_from_config(config, key: str, default: int) -> int:
    metadata = getattr(config, "metadata", None)
    targets = metadata.get("quality_targets_ms") if isinstance(metadata, Mapping) else {}
    value = _realtime_voice_positive_int(targets.get(key)) if isinstance(targets, Mapping) else None
    return value if value is not None else default


def _check_realtime_voice_audio_fixture_smoke(
    issues: list[str],
    *,
    audio_fixture_path: str,
    audio_codec: str,
    reports: list[dict[str, Any]] | None = None,
) -> None:
    try:
        config = _realtime_voice_smoke_config()
    except Exception as exc:
        _append_realtime_voice_smoke_report(
            reports,
            {"ok": False, "error": f"could not build config: {exc}"},
            kind="audio_fixture",
            fixture=audio_fixture_path,
            codec=audio_codec,
        )
        _fail_and_issue(
            "Realtime voice audio fixture smoke",
            f"(could not build config: {exc})",
            f"Fix realtime voice audio smoke configuration: {exc}",
            issues,
        )
        return

    timeout_seconds = min(max(float(config.sidecar_connect_timeout_seconds or 10.0), 1.0), 30.0)
    try:
        result = _run_realtime_voice_sidecar_audio_smoke_sync(
            config,
            audio_fixture_path=audio_fixture_path,
            audio_codec=audio_codec,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        _append_realtime_voice_smoke_report(
            reports,
            {"ok": False, "error": f"failed: {exc}"},
            kind="audio_fixture",
            fixture=audio_fixture_path,
            codec=audio_codec,
            timeout_seconds=timeout_seconds,
        )
        _fail_and_issue(
            "Realtime voice audio fixture smoke",
            f"(failed: {exc})",
            f"Fix realtime voice audio fixture smoke failure: {exc}",
            issues,
        )
        return

    target_ms = _quality_target_from_config(config, "audio_to_partial_transcript_ms", 300)
    _append_realtime_voice_smoke_report(
        reports,
        result,
        kind="audio_fixture",
        fixture=audio_fixture_path,
        codec=audio_codec,
        target_ms=target_ms,
        timeout_seconds=timeout_seconds,
    )
    if result.ok and result.transcript_partial_ms is not None and result.transcript_partial_ms <= target_ms:
        final = f"{result.transcript_final_ms}ms" if result.transcript_final_ms is not None else "unknown"
        check_ok(
            "Realtime voice audio fixture smoke",
            (
                f"(bytes={result.audio_bytes}, transcript.partial={result.transcript_partial_ms}ms"
                f" <= {target_ms}ms, transcript.final={final})"
            ),
        )
        return

    detail = getattr(result, "error", "") or "audio fixture smoke failed"
    if result.ok and result.transcript_partial_ms is None:
        detail = f"missing transcript.partial before transcript.final; transcript.final={result.transcript_final_ms}ms"
    elif result.ok and result.transcript_partial_ms is not None:
        detail = f"transcript.partial={result.transcript_partial_ms}ms exceeds target {target_ms}ms"
    events = tuple(getattr(result, "events", ()) or ())
    if events:
        detail = f"{detail}; events={','.join(events)}"
    _fail_and_issue(
        "Realtime voice audio fixture smoke",
        f"({detail})",
        f"Fix realtime voice audio fixture smoke failure: {detail}",
        issues,
    )


def _check_realtime_voice_readiness(issues: list[str], *, strict: bool = False) -> None:
    """Report whether realtime voice is configured for portable live conversation."""
    _section("Realtime Voice")

    if not strict and not _realtime_voice_enabled_in_config():
        check_warn("Realtime voice disabled", "(skipping live voice readiness gate)")
        check_info("Realtime voice preflight skipped because realtime voice is disabled")
        return

    try:
        payload = _realtime_voice_status_payload()
    except Exception as exc:
        check_warn("Realtime voice status", f"(could not check: {exc})")
        if strict:
            issues.append(f"Realtime voice status could not be checked: {exc}")
        return

    enabled = payload.get("enabled") is True
    available = payload.get("available") is True
    unavailable_reason = str(payload.get("unavailable_reason") or "").strip()
    engine = str(payload.get("engine") or "text_oracle_tts")
    require_live_like = payload.get("require_live_like") is True
    quality = payload.get("conversation_quality")
    quality = quality if isinstance(quality, Mapping) else {}
    mode = str(quality.get("mode") or "unknown")
    reason = str(quality.get("reason") or "unknown")
    live_like = quality.get("live_like")

    if enabled:
        check_ok("Realtime voice enabled", f"({engine})")
    elif strict:
        _fail_and_issue(
            "Realtime voice disabled",
            "(enable voice.realtime.enabled for live voice readiness)",
            "Enable voice.realtime.enabled before using realtime voice",
            issues,
        )
    else:
        check_warn("Realtime voice disabled", "(skipping live voice readiness gate)")

    if enabled and available:
        check_ok("Realtime voice preflight", "(available)")
    elif enabled:
        _fail_and_issue(
            "Realtime voice preflight",
            f"(unavailable: {unavailable_reason or 'unknown'})",
            f"Fix realtime voice availability: {unavailable_reason or 'unknown'}",
            issues,
        )
    elif strict:
        check_fail("Realtime voice preflight", "(unavailable because realtime voice is disabled)")
    else:
        check_info("Realtime voice preflight skipped because realtime voice is disabled")

    quality_detail = f"({mode}: {reason}, live-like: {_realtime_voice_bool(live_like)})"
    if live_like is True:
        check_ok("Live conversation quality", quality_detail)
    elif enabled and (strict or require_live_like):
        _fail_and_issue(
            "Live conversation quality",
            quality_detail,
            "Configure a streaming STT/TTS sidecar or native S2S sidecar; turn-based voice is not live-like",
            issues,
        )
        _emit_realtime_voice_live_setup_hint(payload)
    elif enabled:
        check_warn("Live conversation quality", quality_detail)
    else:
        check_info(f"Live conversation quality not evaluated {quality_detail}")

    _check_realtime_voice_quality_targets(payload, issues, strict=strict)
    _check_realtime_voice_production_readiness(payload, issues, strict=strict)

    language_support = payload.get("language_support")
    language_support = language_support if isinstance(language_support, Mapping) else {}
    production_languages = _realtime_voice_clean_list(language_support.get("production_languages"))
    best_effort = language_support.get("best_effort_languages") is True
    missing_targets = [lang for lang in ("en", "ja") if lang not in production_languages]
    if not missing_targets:
        detail = f"(production: {', '.join(production_languages) or 'none'}; best-effort: {_realtime_voice_bool(best_effort)})"
        check_ok("Realtime voice language policy", detail)
    else:
        check_warn(
            "Realtime voice language policy",
            f"(missing production target(s): {', '.join(missing_targets)})",
        )
        if strict:
            issues.append("Realtime voice production language targets should include English and Japanese")
    if production_languages and best_effort is not True:
        check_warn("Best-effort languages disabled", "(non-target languages may be rejected or hidden)")
        if strict:
            issues.append("Set voice.realtime.best_effort_languages: true unless intentionally limiting speech languages")

    sidecar = payload.get("sidecar")
    sidecar = sidecar if isinstance(sidecar, Mapping) else {}
    sidecar_mode = str(sidecar.get("mode") or "none")
    sidecar_healthy = sidecar.get("healthy")
    if sidecar_mode == "none":
        check_info("Voice sidecar not configured")
    elif sidecar_healthy is True:
        check_ok("Voice sidecar health", f"({sidecar_mode})")
    elif enabled:
        _fail_and_issue(
            "Voice sidecar health",
            f"({sidecar_mode}, healthy: {_realtime_voice_bool(sidecar_healthy)})",
            "Start or fix the configured realtime voice sidecar",
            issues,
        )
    else:
        check_warn("Voice sidecar health", f"({sidecar_mode}, healthy: {_realtime_voice_bool(sidecar_healthy)})")

    if _realtime_voice_has_machine_named_provider(payload):
        check_warn(
            "Portable voice provider naming",
            "(provider/model appears to encode a machine or accelerator name)",
        )
        if strict:
            issues.append(
                "Use capability names like sidecar, reference, gemma4, vllm, or native_s2s; "
                "do not encode a workstation/GPU name into realtime voice provider config"
            )
    else:
        check_ok("Portable voice provider naming", "(capability-based)")


_APIKEY_PROVIDERS_CACHE: list | None = None


def _build_apikey_providers_list() -> list:
    """Build the API-key provider health-check list once and cache it.

    Tuple format: (name, env_vars, default_url, base_env, supports_models_endpoint)
    Base list augmented with any ProviderProfile with auth_type="api_key" not
    already present — adding plugins/model-providers/<name>/ is sufficient to get into doctor.
    """
    _static = [
        ("Z.AI / GLM",      ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"), "https://api.z.ai/api/paas/v4/models", "GLM_BASE_URL", True),
        ("Kimi / Moonshot",  ("KIMI_API_KEY",),                              "https://api.moonshot.ai/v1/models",   "KIMI_BASE_URL", True),
        ("StepFun Step Plan", ("STEPFUN_API_KEY",),                          "https://api.stepfun.ai/step_plan/v1/models", "STEPFUN_BASE_URL", True),
        ("Kimi / Moonshot (China)", ("KIMI_CN_API_KEY",),                    "https://api.moonshot.cn/v1/models",   None, True),
        ("Arcee AI",         ("ARCEEAI_API_KEY",),                           "https://api.arcee.ai/api/v1/models",  "ARCEE_BASE_URL", True),
        ("GMI Cloud",        ("GMI_API_KEY",),                               "https://api.gmi-serving.com/v1/models", "GMI_BASE_URL", True),
        ("DeepSeek",         ("DEEPSEEK_API_KEY",),                          "https://api.deepseek.com/v1/models",  "DEEPSEEK_BASE_URL", True),
        ("Hugging Face",     ("HF_TOKEN",),                                  "https://router.huggingface.co/v1/models", "HF_BASE_URL", True),
        ("NVIDIA NIM",       ("NVIDIA_API_KEY",),                            "https://integrate.api.nvidia.com/v1/models", "NVIDIA_BASE_URL", True),
        ("Alibaba/DashScope", ("DASHSCOPE_API_KEY",),                        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models", "DASHSCOPE_BASE_URL", True),
        # MiniMax global: /v1 endpoint supports /models.
        ("MiniMax",          ("MINIMAX_API_KEY",),                           "https://api.minimax.io/v1/models",    "MINIMAX_BASE_URL", True),
        # MiniMax CN: /v1 endpoint does NOT support /models (returns 404).
        ("MiniMax (China)",  ("MINIMAX_CN_API_KEY",),                        "https://api.minimaxi.com/v1/models",  "MINIMAX_CN_BASE_URL", False),
        ("Kilo Code",        ("KILOCODE_API_KEY",),                          "https://api.kilo.ai/api/gateway/models", "KILOCODE_BASE_URL", True),
        ("OpenCode Zen",     ("OPENCODE_ZEN_API_KEY",),                      "https://opencode.ai/zen/v1/models",  "OPENCODE_ZEN_BASE_URL", True),
        # OpenCode Go has no shared /models endpoint; skip the health check.
        ("OpenCode Go",      ("OPENCODE_GO_API_KEY",),                       None,                                  "OPENCODE_GO_BASE_URL", False),
    ]
    _known_names = {t[0] for t in _static}
    # Also index by profile canonical name so profiles without display_name
    # don't create duplicate entries for providers already in the static list.
    _known_canonical: set[str] = set()
    _name_to_canonical = {
        "Z.AI / GLM": "zai", "Kimi / Moonshot": "kimi-coding",
        "StepFun Step Plan": "stepfun", "Kimi / Moonshot (China)": "kimi-coding-cn",
        "Arcee AI": "arcee", "GMI Cloud": "gmi", "DeepSeek": "deepseek",
        "Hugging Face": "huggingface", "NVIDIA NIM": "nvidia",
        "Alibaba/DashScope": "alibaba", "MiniMax": "minimax",
        "MiniMax (China)": "minimax-cn",
        "Kilo Code": "kilocode", "OpenCode Zen": "opencode-zen",
        "OpenCode Go": "opencode-go",
    }
    for _label, _canonical in _name_to_canonical.items():
        _known_canonical.add(_canonical)
    # Providers that already have a dedicated health check above the generic
    # API-key loop (with custom headers/auth). Skip their pluggable profiles
    # here so the generic Bearer-auth loop doesn't run a duplicate, broken
    # check (e.g. Anthropic native API requires x-api-key, not Bearer).
    _dedicated_canonical = {"anthropic", "openrouter", "bedrock"}
    _known_canonical.update(_dedicated_canonical)
    try:
        from providers import list_providers
        from providers.base import ProviderProfile as _PP
        try:
            from hermes_cli.providers import normalize_provider as _normalize_provider
        except Exception:  # pragma: no cover - normalization is best-effort
            def _normalize_provider(_name: str) -> str:
                return (_name or "").strip().lower()
        for _pp in list_providers():
            if not isinstance(_pp, _PP) or _pp.auth_type != "api_key" or not _pp.env_vars:
                continue
            _label = _pp.display_name or _pp.name
            if _label in _known_names or _pp.name in _known_canonical:
                continue
            _candidates = {_normalize_provider(_pp.name)}
            for _alias in (_pp.aliases or ()):
                _candidates.add(_normalize_provider(_alias))
            if _candidates & _dedicated_canonical:
                continue
            # Separate API-key vars from base-URL override vars — the health-check
            # loop sends the first found value as Authorization: Bearer, so a URL
            # string must never be picked.
            _key_vars = tuple(
                v for v in _pp.env_vars
                if not v.endswith("_BASE_URL") and not v.endswith("_URL")
            )
            _base_var = next(
                (v for v in _pp.env_vars if v.endswith("_BASE_URL") or v.endswith("_URL")),
                None,
            )
            if not _key_vars:
                continue
            _models_url = (
                (_pp.models_url or (_pp.base_url.rstrip("/") + "/models"))
                if _pp.base_url else None
            )
            _hc = getattr(_pp, "supports_health_check", True)
            _static.append((_label, _key_vars, _models_url, _base_var, _hc))
    except Exception:
        pass
    return _static


def managed_scope_check() -> None:
    """Report the active managed scope (resolved dir + pinned key counts).

    Silent when no managed scope is present. When the managed directory was
    resolved from the HERMES_MANAGED_DIR override (rather than the system
    default), that is surfaced too — a redirected scope is the documented
    foot-gun (see docs/design/managed-scope.md §7) and an operator should see it.
    """
    try:
        from hermes_cli import managed_scope
        managed_dir = managed_scope.get_managed_dir()
    except Exception:  # noqa: BLE001 — diagnostics must never crash
        return
    if managed_dir is None:
        return
    n_cfg = len(managed_scope.managed_config_keys())
    n_env = len(managed_scope.load_managed_env())
    check_ok(
        f"Managed scope active: {n_cfg} config key(s), {n_env} env key(s) "
        f"pinned by {managed_dir}"
    )
    if os.environ.get("HERMES_MANAGED_DIR", "").strip():
        check_info(f"managed dir set via HERMES_MANAGED_DIR={managed_dir}")


def run_doctor(args):
    """Run diagnostic checks."""
    should_fix = getattr(args, 'fix', False)
    ack_target = getattr(args, 'ack', None)
    realtime_voice_alpha = getattr(args, 'realtime_voice_alpha', False)
    smoke_realtime_voice = getattr(args, 'realtime_voice_smoke', False)
    smoke_discord_realtime_voice = getattr(args, 'discord_realtime_voice_smoke', False)
    discord_voice_live_probe = getattr(args, 'discord_voice_live_probe', False)
    realtime_voice_audio_fixtures = _realtime_voice_cli_values(
        getattr(args, 'realtime_voice_audio_fixture', None)
    )
    realtime_voice_tts_smokes = _realtime_voice_cli_values(
        getattr(args, 'realtime_voice_tts_smoke', None)
    )
    realtime_voice_barge_in_smokes = _realtime_voice_cli_values(
        getattr(args, 'realtime_voice_barge_in_smoke', None)
    )
    (
        smoke_realtime_voice,
        realtime_voice_audio_fixtures,
        realtime_voice_tts_smokes,
        realtime_voice_barge_in_smokes,
    ) = _realtime_voice_expand_alpha_smokes(
        alpha=realtime_voice_alpha,
        smoke=smoke_realtime_voice,
        audio_fixtures=realtime_voice_audio_fixtures,
        tts_texts=realtime_voice_tts_smokes,
        barge_in_texts=realtime_voice_barge_in_smokes,
    )
    realtime_voice_report_path = getattr(args, 'realtime_voice_report', None)
    realtime_voice_reports: list[dict[str, Any]] | None = [] if realtime_voice_report_path else None
    strict_realtime_voice = (
        getattr(args, 'realtime_voice', False)
        or smoke_realtime_voice
        or smoke_discord_realtime_voice
        or bool(realtime_voice_audio_fixtures)
        or bool(realtime_voice_tts_smokes)
        or bool(realtime_voice_barge_in_smokes)
    )

    # Doctor runs from the interactive CLI, so CLI-gated tool availability
    # checks (like cronjob management) should see the same context as `hermes`.
    os.environ.setdefault("HERMES_INTERACTIVE", "1")

    # Handle `hermes doctor --ack <id>` as a fast path. Persist the ack and
    # return without running the rest of the diagnostics — the user has
    # already seen the advisory and just wants to silence it.
    if ack_target:
        from hermes_cli.security_advisories import (
            ADVISORIES,
            ack_advisory,
        )
        valid_ids = {a.id for a in ADVISORIES}
        if ack_target not in valid_ids:
            print(color(
                f"Unknown advisory ID: {ack_target!r}. Known IDs: "
                f"{', '.join(sorted(valid_ids)) or '(none)'}",
                Colors.RED,
            ))
            sys.exit(2)
        if ack_advisory(ack_target):
            print(color(
                f"  ✓ Acknowledged advisory {ack_target}. "
                f"It will no longer trigger startup banners.",
                Colors.GREEN,
            ))
        else:
            print(color(
                f"  ✗ Failed to persist ack for {ack_target}. "
                f"Check ~/.hermes/config.yaml is writable.",
                Colors.RED,
            ))
            sys.exit(1)
        return

    issues = []
    manual_issues = []  # issues that can't be auto-fixed
    fixed_count = 0

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│                 🩺 Hermes Doctor                        │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))

    _section("Security Advisories")
    try:
        from hermes_cli.security_advisories import (
            detect_compromised,
            filter_unacked,
            full_remediation_text,
            get_acked_ids,
        )
        all_hits = detect_compromised()
        fresh_hits = filter_unacked(all_hits)
        if fresh_hits:
            for hit in fresh_hits:
                check_fail(
                    f"{hit.advisory.title}",
                    f"({hit.package}=={hit.installed_version})",
                )
                # Print the full remediation block, indented under the
                # check_fail header so it reads as a single section.
                for line in full_remediation_text(hit):
                    if line:
                        print(f"    {color(line, Colors.YELLOW)}")
                    else:
                        print()
                # Funnel into the action list so the summary block surfaces it
                # for users who scroll past the section.
                manual_issues.append(
                    f"Resolve security advisory {hit.advisory.id}: "
                    f"uninstall {hit.package}=={hit.installed_version} and "
                    f"rotate credentials, then run "
                    f"`hermes doctor --ack {hit.advisory.id}`."
                )
            # Acked-but-still-installed: show as informational so the user
            # knows the package is still on disk after the ack.
            acked_ids = get_acked_ids()
            for h in all_hits:
                if h.advisory.id in acked_ids:
                    check_warn(
                        f"{h.package}=={h.installed_version} still installed "
                        f"(advisory {h.advisory.id} acknowledged)",
                    )
        else:
            check_ok("No active security advisories")
    except Exception as e:
        # Never let a bug in the advisory check block the rest of doctor.
        check_warn(f"Security advisory check failed: {e}")

    _section("MCP Server Security")
    try:
        from hermes_cli.config import load_config
        from hermes_cli.mcp_security import validate_mcp_server_entry

        servers = load_config().get("mcp_servers") or {}
        suspicious = 0
        if isinstance(servers, dict):
            for name, entry in sorted(servers.items()):
                if not isinstance(entry, dict):
                    continue
                issues_found = validate_mcp_server_entry(name, entry)
                if not issues_found:
                    continue
                suspicious += 1
                check_warn(f"MCP server '{name}' has suspicious stdio command", "; ".join(issues_found))
                manual_issues.append(
                    f"Review/remove mcp_servers.{name} in config.yaml; rotate any credentials that may have been exposed."
                )
        if suspicious == 0:
            check_ok("No suspicious MCP stdio commands")
    except Exception as e:
        check_warn(f"MCP security check failed: {e}")
    
    _section("Python Environment")
    py_version = sys.version_info
    if py_version >= (3, 11):
        check_ok(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    elif py_version >= (3, 10):
        check_ok(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
        check_warn("Python 3.11+ recommended for RL Training tools (tinker requires >= 3.11)")
    elif py_version >= (3, 8):
        check_warn(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}", "(3.10+ recommended)")
    else:
        _fail_and_issue(
            f"Python {py_version.major}.{py_version.minor}.{py_version.micro}",
            "(3.10+ required)",
            "Upgrade Python to 3.10+",
            issues,
        )
    
    # Check if in virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        check_ok("Virtual environment active")
    else:
        check_warn("Not in virtual environment", "(recommended)")

    # Detect drift between pyproject.toml and hermes_cli/__init__.py versions
    # (a git conflict resolution can silently revert one but not the other).
    _check_version_consistency(issues)

    _section("SSL / CA Certificates")
    check_certificates()

    _section("Required Packages")
    required_packages = [
        ("openai", "OpenAI SDK"),
        ("rich", "Rich (terminal UI)"),
        ("dotenv", "python-dotenv"),
        ("yaml", "PyYAML"),
        ("httpx", "HTTPX"),
    ]
    
    optional_packages = [
        ("croniter", "Croniter (cron expressions)"),
        ("telegram", "python-telegram-bot"),
        ("discord", "discord.py"),
    ]
    
    for module, name in required_packages:
        try:
            __import__(module)
            check_ok(name)
        except ImportError:
            _fail_and_issue(name, "(missing)", f"Install {name}: {_python_install_cmd()} {module}", issues)
    
    for module, name in optional_packages:
        try:
            __import__(module)
            check_ok(name, "(optional)")
        except ImportError:
            check_warn(name, "(optional, not installed)")
    
    _section("Configuration Files")
    # Managed scope (administrator-pinned config/env), when present.
    managed_scope_check()
    # Check ~/.hermes/.env (primary location for user config)
    env_path = HERMES_HOME / '.env'
    if env_path.exists():
        check_ok(f"{_DHH}/.env file exists")
        
        # Check for common issues. Pin encoding to UTF-8 because .env files are
        # written as UTF-8 everywhere in the codebase, while Path.read_text()
        # defaults to the system locale — which crashes on non-UTF-8 Windows
        # locales (e.g. GBK) as soon as the file contains any non-ASCII byte.
        content = env_path.read_text(encoding="utf-8")
        if _has_provider_env_config(content):
            check_ok("API key or custom endpoint configured")
        else:
            check_warn(f"No API key found in {_DHH}/.env")
            issues.append("Run 'hermes setup' to configure API keys")
    else:
        # Also check project root as fallback
        fallback_env = PROJECT_ROOT / '.env'
        if fallback_env.exists():
            check_ok(".env file exists (in project directory)")
        else:
            check_fail(f"{_DHH}/.env file missing")
            if should_fix:
                env_path.parent.mkdir(parents=True, exist_ok=True)
                env_path.touch()
                # .env holds API keys — restrict to owner-only access from
                # creation. touch() obeys umask which is commonly 0o022,
                # leaving the file world-readable; tighten explicitly.
                try:
                    os.chmod(str(env_path), 0o600)
                except OSError:
                    pass
                check_ok(f"Created empty {_DHH}/.env")
                check_info("Run 'hermes setup' to configure API keys")
                fixed_count += 1
            else:
                check_info("Run 'hermes setup' to create one")
                issues.append("Run 'hermes setup' to create .env")
    
    # Check ~/.hermes/config.yaml (primary) or project cli-config.yaml (fallback)
    config_path = HERMES_HOME / 'config.yaml'
    if config_path.exists():
        check_ok(f"{_DHH}/config.yaml exists")

        # Validate model.provider and model.default values
        try:
            import yaml as _yaml
            cfg = _yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            model_section = cfg.get("model") or {}
            provider_raw = (model_section.get("provider") or "").strip()
            provider = provider_raw.lower()
            default_model = (model_section.get("default") or model_section.get("model") or "").strip()

            known_providers: set = set()
            try:
                from hermes_cli.auth import (
                    PROVIDER_REGISTRY,
                    resolve_provider as _resolve_auth_provider,
                )
                known_providers = set(PROVIDER_REGISTRY.keys()) | {"openrouter", "custom", "auto"}
            except Exception:
                _resolve_auth_provider = None
                pass
            try:
                from hermes_cli.config import get_compatible_custom_providers as _compatible_custom_providers
                from hermes_cli.providers import (
                    normalize_provider as _normalize_catalog_provider,
                    resolve_provider_full as _resolve_provider_full,
                )
            except Exception:
                _compatible_custom_providers = None
                _normalize_catalog_provider = None
                _resolve_provider_full = None

            custom_providers = []
            if _compatible_custom_providers is not None:
                try:
                    custom_providers = _compatible_custom_providers(cfg)
                except Exception:
                    custom_providers = []

            user_providers = cfg.get("providers")
            if isinstance(user_providers, dict):
                known_providers.update(str(name).strip().lower() for name in user_providers if str(name).strip())
            for entry in custom_providers:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name") or "").strip()
                if name:
                    known_providers.add("custom:" + name.lower().replace(" ", "-"))

            valid_provider_ids = set(known_providers)
            provider_ids_to_accept = {provider} if provider else set()
            if _normalize_catalog_provider is not None:
                for known_provider in known_providers:
                    try:
                        valid_provider_ids.add(_normalize_catalog_provider(known_provider))
                    except Exception:
                        continue

            runtime_provider = provider
            if (
                provider
                and _resolve_auth_provider is not None
                and provider not in {"auto", "custom"}
            ):
                try:
                    runtime_provider = _resolve_auth_provider(provider)
                    provider_ids_to_accept.add(runtime_provider)
                except Exception:
                    runtime_provider = provider

            catalog_provider = provider
            if (
                provider
                and _resolve_provider_full is not None
                and provider not in {"auto", "custom"}
            ):
                provider_def = _resolve_provider_full(provider, user_providers, custom_providers)
                catalog_provider = provider_def.id if provider_def is not None else None
                if catalog_provider is not None:
                    provider_ids_to_accept.add(catalog_provider)

            if provider and provider != "auto":
                if catalog_provider is None or (
                    known_providers
                    and not (provider_ids_to_accept & valid_provider_ids)
                ):
                    known_list = ", ".join(sorted(known_providers)) if known_providers else "(unavailable)"
                    _fail_and_issue(
                        f"model.provider '{provider_raw}' is not a recognised provider",
                        f"(known: {known_list})",
                        (
                            f"model.provider '{provider_raw}' is unknown. "
                            f"Valid providers: {known_list}. "
                            f"Fix: run 'hermes config set model.provider <valid_provider>'"
                        ),
                        issues,
                    )

            # Warn if model is set to a provider-prefixed name on a provider that doesn't use them.
            # Vendor/model slugs are valid on aggregator-style providers and on any custom
            # provider — bare "custom" or a named "custom:<name>" that fronts an OpenAI-compatible
            # aggregator (e.g. custom:hpc-ai serving deepseek/deepseek-v4-flash) requires the prefix.
            provider_for_policy = runtime_provider or catalog_provider
            provider_policy_id = str(provider_for_policy or "").strip().lower()
            providers_accepting_vendor_slugs = {
                "openrouter",
                "auto",
                "kilocode",
                "opencode-zen",
                "huggingface",
                "lmstudio",
                "nous",
                "nvidia",
            }
            provider_accepts_vendor_slug = (
                provider_policy_id in providers_accepting_vendor_slugs
                or provider_policy_id == "custom"
                or provider_policy_id.startswith("custom:")
            )
            if (
                default_model
                and "/" in default_model
                and provider_policy_id
                and not provider_accepts_vendor_slug
            ):
                check_warn(
                    f"model.default '{default_model}' uses a vendor/model slug but provider is '{provider_raw}'",
                    "(vendor-prefixed slugs belong to aggregators like openrouter)",
                )
                issues.append(
                    f"model.default '{default_model}' is vendor-prefixed but model.provider is '{provider_raw}'. "
                    "Either set model.provider to 'openrouter', or drop the vendor prefix."
                )

            # Check credentials for the configured provider.
            # Limit to API-key providers in PROVIDER_REGISTRY — other provider
            # types (OAuth, SDK, anthropic/custom/auto) have their own env-var
            # checks elsewhere in doctor, and get_auth_status() returns a bare
            # {logged_in: False} for anything it doesn't explicitly dispatch,
            # which would produce false positives.
            if runtime_provider and runtime_provider not in ("auto", "custom"):
                try:
                    if runtime_provider == "openrouter":
                        from hermes_cli.config import get_env_value

                        configured = bool(
                            str(get_env_value("OPENROUTER_API_KEY") or "").strip()
                            or str(get_env_value("OPENAI_API_KEY") or "").strip()
                        )
                    else:
                        from hermes_cli.auth import PROVIDER_REGISTRY, get_auth_status

                        pconfig = PROVIDER_REGISTRY.get(runtime_provider)
                        configured = True
                        if pconfig and getattr(pconfig, "auth_type", "") == "api_key":
                            status = get_auth_status(runtime_provider) or {}
                            configured = bool(
                                status.get("configured")
                                or status.get("logged_in")
                                or status.get("api_key")
                            )
                    if not configured:
                        _fail_and_issue(
                            f"model.provider '{runtime_provider}' is set but no API key is configured",
                            "(check ~/.hermes/.env or run 'hermes setup')",
                            (
                                f"No credentials found for provider '{runtime_provider}'. "
                                f"Run 'hermes setup' or set the provider's API key in {_DHH}/.env, "
                                f"or switch providers with 'hermes config set model.provider <name>'"
                            ),
                            issues,
                        )
                except Exception:
                    pass

        except Exception as e:
            check_warn("Could not validate model/provider config", f"({e})")
    else:
        fallback_config = PROJECT_ROOT / 'cli-config.yaml'
        if fallback_config.exists():
            check_ok("cli-config.yaml exists (in project directory)")
        else:
            if should_fix:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                example_config = PROJECT_ROOT / 'cli-config.yaml.example'
                if example_config.exists():
                    shutil.copy2(str(example_config), str(config_path))
                    check_ok(f"Created {_DHH}/config.yaml from cli-config.yaml.example")
                else:
                    from hermes_cli.config import DEFAULT_CONFIG, save_config
                    save_config(DEFAULT_CONFIG)
                    check_ok(f"Created {_DHH}/config.yaml from defaults")
                fixed_count += 1
            else:
                check_warn("config.yaml not found", "(using defaults)")

    # Check config version and stale keys
    config_path = HERMES_HOME / 'config.yaml'
    if config_path.exists():
        try:
            from hermes_cli.config import check_config_version, migrate_config
            current_ver, latest_ver = check_config_version()
            if current_ver < latest_ver:
                check_warn(
                    f"Config version outdated (v{current_ver} → v{latest_ver})",
                    "(new settings available)"
                )
                if should_fix:
                    try:
                        migrate_config(interactive=False, quiet=False)
                        check_ok("Config migrated to latest version")
                        fixed_count += 1
                    except Exception as mig_err:
                        check_warn(f"Auto-migration failed: {mig_err}")
                        issues.append("Run 'hermes setup' to migrate config")
                else:
                    issues.append("Run 'hermes doctor --fix' or 'hermes setup' to migrate config")
            else:
                check_ok(f"Config version up to date (v{current_ver})")
        except Exception:
            pass

        # Detect stale root-level model keys (known bug source — PR #4329)
        try:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}
            stale_root_keys = [k for k in ("provider", "base_url") if k in raw_config and isinstance(raw_config[k], str)]
            if stale_root_keys:
                check_warn(
                    f"Stale root-level config keys: {', '.join(stale_root_keys)}",
                    "(should be under 'model:' section)"
                )
                if should_fix:
                    # Coerce scalar/None ``model:`` into a dict before mutation —
                    # ``setdefault("model", {})`` would return an existing scalar
                    # and then ``model_section[k] = ...`` would raise TypeError.
                    raw_model = raw_config.get("model")
                    if isinstance(raw_model, dict):
                        model_section = raw_model
                    elif isinstance(raw_model, str) and raw_model.strip():
                        model_section = {"default": raw_model.strip()}
                        raw_config["model"] = model_section
                    else:
                        model_section = {}
                        raw_config["model"] = model_section
                    for k in stale_root_keys:
                        if not model_section.get(k):
                            model_section[k] = raw_config.pop(k)
                        else:
                            raw_config.pop(k)
                    from utils import atomic_yaml_write
                    atomic_yaml_write(config_path, raw_config)
                    check_ok("Migrated stale root-level keys into model section")
                    fixed_count += 1
                else:
                    issues.append("Stale root-level provider/base_url in config.yaml — run 'hermes doctor --fix'")
        except Exception:
            pass

        # Detect stale HERMES_MAX_ITERATIONS ghost in .env shadowing
        # agent.max_turns in config.yaml (issue #17534). The setup wizard
        # used to dual-write the iteration budget to both stores; users who
        # later edit only config.yaml are left with a .env ghost. The gateway
        # bridge normally derives HERMES_MAX_ITERATIONS from agent.max_turns
        # at startup, but if that bridge bails (any earlier config-parse
        # error), the stale .env value silently wins and the agent runs at the
        # wrong budget — e.g. config says 400 but the activity line reads N/90.
        # Read the .env FILE directly (load_env), not get_env_value/os.environ,
        # which the startup bridge may already have overridden.
        try:
            import yaml
            from hermes_cli.config import load_env, remove_env_value
            with open(config_path, encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}
            agent_cfg = raw_config.get("agent")
            cfg_max_turns = (
                agent_cfg.get("max_turns")
                if isinstance(agent_cfg, dict)
                else None
            )
            # Legacy root-level key counts too.
            if cfg_max_turns is None:
                cfg_max_turns = raw_config.get("max_turns")
            env_ghost = load_env().get("HERMES_MAX_ITERATIONS")
            drift = (
                cfg_max_turns is not None
                and env_ghost is not None
                and str(cfg_max_turns).strip() != str(env_ghost).strip()
            )
            if drift:
                check_warn(
                    f"HERMES_MAX_ITERATIONS={env_ghost} in .env shadows "
                    f"agent.max_turns={cfg_max_turns} in config.yaml",
                    "(stale ghost from an earlier `hermes setup` run)",
                )
                if should_fix:
                    if remove_env_value("HERMES_MAX_ITERATIONS"):
                        check_ok(
                            "Removed stale HERMES_MAX_ITERATIONS from .env "
                            f"(config.yaml agent.max_turns={cfg_max_turns} is now authoritative)"
                        )
                        fixed_count += 1
                    else:
                        check_warn("Could not remove HERMES_MAX_ITERATIONS from .env")
                        manual_issues.append(
                            "Manually delete the HERMES_MAX_ITERATIONS line from "
                            f"{_DHH}/.env — config.yaml agent.max_turns is authoritative."
                        )
                else:
                    issues.append(
                        "Stale HERMES_MAX_ITERATIONS in .env shadows config.yaml — "
                        "run 'hermes doctor --fix'"
                    )
        except Exception:
            pass

        # Validate config structure (catches malformed custom_providers, etc.)
        try:
            from hermes_cli.config import validate_config_structure
            config_issues = validate_config_structure()
            if config_issues:
                _section("Config Structure")
                for ci in config_issues:
                    if ci.severity == "error":
                        check_fail(ci.message)
                    else:
                        check_warn(ci.message)
                    # Show the hint indented
                    for hint_line in ci.hint.splitlines():
                        check_info(hint_line)
                    issues.append(ci.message)
        except Exception:
            pass

    _section("xAI Model Retirement (May 15, 2026)")

    try:
        from hermes_cli.config import load_config
        from hermes_cli.xai_retirement import (
            MIGRATION_GUIDE_URL,
            find_retired_xai_refs,
            format_issue,
        )

        _xai_cfg = load_config()
        retired_refs = find_retired_xai_refs(_xai_cfg)
        if not retired_refs:
            check_ok("No retired xAI models in config")
        else:
            for ref in retired_refs:
                check_warn(format_issue(ref))
            check_info(f"Migration guide: {MIGRATION_GUIDE_URL}")
            manual_issues.append(
                f"Update {len(retired_refs)} retired xAI model reference(s) "
                f"in config.yaml — see {MIGRATION_GUIDE_URL}"
            )
    except Exception as _xai_check_err:
        check_warn("xAI retirement check skipped", f"({_xai_check_err})")

    _section("Auth Providers")

    try:
        from hermes_cli.auth import (
            get_nous_auth_status,
            get_codex_auth_status,
            get_minimax_oauth_auth_status,
        )

        nous_status = get_nous_auth_status()
        if nous_status.get("logged_in"):
            check_ok("Nous Portal auth", "(logged in)")
        else:
            check_warn("Nous Portal auth", "(not logged in)")

        codex_status = get_codex_auth_status()
        if codex_status.get("logged_in"):
            check_ok("OpenAI Codex auth", "(logged in)")
        else:
            check_warn("OpenAI Codex auth", "(not logged in)")
            if codex_status.get("error"):
                check_info(codex_status["error"])
            # Native OAuth uses Hermes' own device-code flow — the Codex CLI is
            # only needed to import existing tokens from ~/.codex/auth.json.
            # Attach the hint to the Codex auth row so it doesn't read as
            # remediation for whichever provider happens to print next (#27975).
            if not _safe_which("codex"):
                check_info(
                    "codex CLI not installed "
                    "(optional — only required to import tokens "
                    "from an existing Codex CLI login)"
                )

        minimax_status = get_minimax_oauth_auth_status()
        if minimax_status.get("logged_in"):
            region = minimax_status.get("region", "global")
            check_ok("MiniMax OAuth", f"(logged in, region={region})")
        else:
            check_warn("MiniMax OAuth", "(not logged in)")
    except Exception as e:
        check_warn("Auth provider status", f"(could not check: {e})")

    # xAI OAuth — separate try/except so an import failure here cannot
    # disrupt the already-printed Nous/Codex/Gemini/MiniMax rows above.
    try:
        from hermes_cli.auth import get_xai_oauth_auth_status
        xai_oauth_status = get_xai_oauth_auth_status() or {}
        if xai_oauth_status.get("logged_in"):
            check_ok("xAI OAuth", "(logged in)")
        else:
            check_warn("xAI OAuth", "(not logged in)")
            if xai_oauth_status.get("error"):
                check_info(xai_oauth_status["error"])
    except Exception:
        pass

    _check_realtime_voice_readiness(issues, strict=strict_realtime_voice)
    if smoke_realtime_voice:
        _check_realtime_voice_sidecar_smoke(issues, reports=realtime_voice_reports)
        session_turn_texts = (
            _realtime_voice_alpha_session_turn_texts()
            if realtime_voice_alpha
            else ["Hello from Hermes."]
        )
        for realtime_voice_session_turn_smoke in session_turn_texts:
            _check_realtime_voice_session_turn_smoke(
                issues,
                text=realtime_voice_session_turn_smoke,
                reports=realtime_voice_reports,
            )
        if realtime_voice_alpha:
            for realtime_voice_audio_session_fixture in _realtime_voice_alpha_audio_session_fixtures():
                _check_realtime_voice_session_audio_smoke(
                    issues,
                    audio_fixture_path=realtime_voice_audio_session_fixture,
                    audio_codec=str(getattr(args, 'realtime_voice_audio_codec', "webm_opus")),
                    reports=realtime_voice_reports,
                )
    for realtime_voice_audio_fixture in realtime_voice_audio_fixtures:
        _check_realtime_voice_audio_fixture_smoke(
            issues,
            audio_fixture_path=realtime_voice_audio_fixture,
            audio_codec=str(getattr(args, 'realtime_voice_audio_codec', "webm_opus")),
            reports=realtime_voice_reports,
        )
    for realtime_voice_tts_smoke in realtime_voice_tts_smokes:
        _check_realtime_voice_tts_smoke(
            issues,
            text=realtime_voice_tts_smoke,
            reports=realtime_voice_reports,
        )
    for realtime_voice_barge_in_smoke in realtime_voice_barge_in_smokes:
        _check_realtime_voice_barge_in_smoke(
            issues,
            text=realtime_voice_barge_in_smoke,
            reports=realtime_voice_reports,
        )
    if smoke_discord_realtime_voice:
        _check_discord_realtime_voice_smoke(issues, reports=realtime_voice_reports)
    if discord_voice_live_probe:
        _check_discord_voice_live_probe(
            issues,
            reports=realtime_voice_reports,
            wait_seconds=float(getattr(args, 'discord_voice_live_probe_wait_seconds', 2.0) or 0.0),
            require_inbound=bool(getattr(args, 'discord_voice_live_probe_require_inbound', False)),
            voice_channel_id=str(getattr(args, 'discord_voice_live_probe_channel_id', "") or ""),
            voice_channel_name=str(getattr(args, 'discord_voice_live_probe_channel_name', "") or ""),
        )
    if realtime_voice_report_path:
        try:
            _append_realtime_voice_evidence_manifest(realtime_voice_reports)
            _write_realtime_voice_report(realtime_voice_report_path, realtime_voice_reports or [])
            check_ok("Realtime voice smoke report", f"({realtime_voice_report_path})")
        except Exception as exc:
            _fail_and_issue(
                "Realtime voice smoke report",
                f"(failed: {exc})",
                f"Fix realtime voice smoke report path: {exc}",
                issues,
            )

    _section("Directory Structure")
    hermes_home = HERMES_HOME
    if hermes_home.exists():
        check_ok(f"{_DHH} directory exists")
    elif should_fix:
        hermes_home.mkdir(parents=True, exist_ok=True)
        check_ok(f"Created {_DHH} directory")
        fixed_count += 1
    else:
        check_warn(f"{_DHH} not found", "(will be created on first use)")
    
    # Check expected subdirectories
    expected_subdirs = ["cron", "sessions", "logs", "skills", "memories"]
    for subdir_name in expected_subdirs:
        subdir_path = hermes_home / subdir_name
        if subdir_path.exists():
            check_ok(f"{_DHH}/{subdir_name}/ exists")
        elif should_fix:
            subdir_path.mkdir(parents=True, exist_ok=True)
            check_ok(f"Created {_DHH}/{subdir_name}/")
            fixed_count += 1
        else:
            check_warn(f"{_DHH}/{subdir_name}/ not found", "(will be created on first use)")
    
    # Check for SOUL.md persona file
    soul_path = hermes_home / "SOUL.md"
    if soul_path.exists():
        content = soul_path.read_text(encoding="utf-8").strip()
        # Check if it's just the template comments (no real content)
        lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith(("<!--", "-->", "#"))]
        if lines:
            check_ok(f"{_DHH}/SOUL.md exists (persona configured)")
        else:
            check_info(f"{_DHH}/SOUL.md exists but is empty — edit it to customize personality")
    else:
        check_warn(f"{_DHH}/SOUL.md not found", "(create it to give Hermes a custom personality)")
        if should_fix:
            soul_path.parent.mkdir(parents=True, exist_ok=True)
            soul_path.write_text(
                "# Hermes Agent Persona\n\n"
                "<!-- Edit this file to customize how Hermes communicates. -->\n\n"
                "You are Hermes, a helpful AI assistant.\n",
                encoding="utf-8",
            )
            check_ok(f"Created {_DHH}/SOUL.md with basic template")
            fixed_count += 1
    
    # Check memory directory
    memories_dir = hermes_home / "memories"
    if memories_dir.exists():
        check_ok(f"{_DHH}/memories/ directory exists")
        memory_file = memories_dir / "MEMORY.md"
        user_file = memories_dir / "USER.md"
        if memory_file.exists():
            size = len(memory_file.read_text(encoding="utf-8").strip())
            check_ok(f"MEMORY.md exists ({size} chars)")
        else:
            check_info("MEMORY.md not created yet (will be created when the agent first writes a memory)")
        if user_file.exists():
            size = len(user_file.read_text(encoding="utf-8").strip())
            check_ok(f"USER.md exists ({size} chars)")
        else:
            check_info("USER.md not created yet (will be created when the agent first writes a memory)")
    else:
        check_warn(f"{_DHH}/memories/ not found", "(will be created on first use)")
        if should_fix:
            memories_dir.mkdir(parents=True, exist_ok=True)
            check_ok(f"Created {_DHH}/memories/")
            fixed_count += 1
    
    # Check SQLite session store
    state_db_path = hermes_home / "state.db"
    if state_db_path.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(state_db_path))
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            count = cursor.fetchone()[0]
            conn.close()
            check_ok(f"{_DHH}/state.db exists ({count} sessions)")

            # FTS write-health probe (#50502): `SELECT COUNT(*)` above succeeds
            # even when the FTS index is corrupt and every message write fails
            # through the triggers. `_db_opens_cleanly` now drives a rolled-back
            # write so this otherwise-silent corruption class is surfaced (and
            # repaired in place with --fix).
            from hermes_state import _db_opens_cleanly, repair_state_db_schema

            _write_reason = _db_opens_cleanly(state_db_path)
            if _write_reason is not None:
                check_warn(
                    f"{_DHH}/state.db fails a write-health probe (FTS index may be corrupt)",
                    f"({_write_reason})",
                )
                if should_fix:
                    report = repair_state_db_schema(state_db_path)
                    if report.get("repaired"):
                        backup_name = (
                            Path(report["backup_path"]).name
                            if report.get("backup_path") else "n/a"
                        )
                        check_ok(
                            "Repaired state.db FTS write health",
                            f"(strategy: {report.get('strategy')}; backup: {backup_name})",
                        )
                        fixed_count += 1
                    else:
                        check_warn(
                            "state.db FTS write-health repair did not recover automatically",
                            f"({report.get('error')}; backup: {report.get('backup_path')})",
                        )
                        issues.append(
                            "state.db FTS write corruption and auto-repair failed — "
                            "restore from the backup copy beside state.db"
                        )
                else:
                    issues.append(
                        "state.db FTS write corruption — run 'hermes doctor --fix' "
                        "(or 'hermes sessions repair') to rebuild the FTS index"
                    )
        except Exception as e:
            from hermes_state import is_malformed_db_error, repair_state_db_schema

            if is_malformed_db_error(e):
                # sqlite_master itself is malformed (e.g. duplicate
                # messages_fts) — every statement fails before it runs, so
                # this is NOT a plain FTS-index rebuild. Repair sqlite_master
                # in place (backup first; sessions/messages preserved).
                check_warn(
                    f"{_DHH}/state.db schema is malformed (sessions hidden until repaired)",
                    f"({e})",
                )
                if should_fix:
                    report = repair_state_db_schema(state_db_path)
                    if report.get("repaired"):
                        try:
                            conn = sqlite3.connect(str(state_db_path))
                            count = conn.execute(
                                "SELECT COUNT(*) FROM sessions"
                            ).fetchone()[0]
                            conn.close()
                        except Exception:
                            count = "?"
                        backup_name = (
                            Path(report["backup_path"]).name
                            if report.get("backup_path") else "n/a"
                        )
                        check_ok(
                            f"Repaired state.db schema ({count} sessions recovered)",
                            f"(strategy: {report.get('strategy')}; backup: {backup_name})",
                        )
                        fixed_count += 1
                    else:
                        check_warn(
                            "state.db schema repair did not recover automatically",
                            f"({report.get('error')}; backup: {report.get('backup_path')})",
                        )
                        issues.append(
                            "state.db schema malformed and auto-repair failed — "
                            "restore from the backup copy beside state.db"
                        )
                else:
                    issues.append(
                        "state.db schema malformed — run 'hermes doctor --fix' "
                        "(or 'hermes sessions repair') to recover hidden sessions"
                    )
            else:
                check_warn(f"{_DHH}/state.db exists but has issues: {e}")
    else:
        check_info(f"{_DHH}/state.db not created yet (will be created on first session)")

    # Check WAL file size (unbounded growth indicates missed checkpoints)
    wal_path = hermes_home / "state.db-wal"
    if wal_path.exists():
        try:
            wal_size = wal_path.stat().st_size
            if wal_size > 50 * 1024 * 1024:  # 50 MB
                check_warn(
                    f"WAL file is large ({wal_size // (1024*1024)} MB)",
                    "(may indicate missed checkpoints)"
                )
                if should_fix:
                    import sqlite3
                    conn = sqlite3.connect(str(state_db_path))
                    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    conn.close()
                    new_size = wal_path.stat().st_size if wal_path.exists() else 0
                    check_ok(f"WAL checkpoint performed ({wal_size // 1024}K → {new_size // 1024}K)")
                    fixed_count += 1
                else:
                    issues.append("Large WAL file — run 'hermes doctor --fix' to checkpoint")
            elif wal_size > 10 * 1024 * 1024:  # 10 MB
                check_info(f"WAL file is {wal_size // (1024*1024)} MB (normal for active sessions)")
        except Exception:
            pass

    _check_gateway_service_linger(issues)
    _check_s6_supervision(issues)

    if sys.platform != "win32":
        _section("Command Installation")
        # Determine the venv entry point location
        _venv_bin = None
        for _venv_name in ("venv", ".venv"):
            _candidate = PROJECT_ROOT / _venv_name / "bin" / "hermes"
            if _candidate.exists():
                _venv_bin = _candidate
                break

        # Determine the expected command link directory (mirrors install.sh logic)
        _prefix = os.environ.get("PREFIX", "")
        _is_termux_env = bool(os.environ.get("TERMUX_VERSION")) or "com.termux/files/usr" in _prefix
        if _is_termux_env and _prefix:
            _cmd_link_dir = Path(_prefix) / "bin"
            _cmd_link_display = "$PREFIX/bin"
        else:
            _cmd_link_dir = Path.home() / ".local" / "bin"
            _cmd_link_display = "~/.local/bin"
        _cmd_link = _cmd_link_dir / "hermes"

        if _venv_bin is None:
            check_warn(
                "Venv entry point not found",
                "(hermes not in venv/bin/ or .venv/bin/ — reinstall with pip install -e '.[all]')"
            )
            manual_issues.append(
                f"Reinstall entry point: cd {PROJECT_ROOT} && source venv/bin/activate && pip install -e '.[all]'"
            )
        else:
            check_ok(f"Venv entry point exists ({_venv_bin.relative_to(PROJECT_ROOT)})")

            # Check the symlink at the command link location
            if _cmd_link.is_symlink():
                _target = _cmd_link.resolve()
                _expected = _venv_bin.resolve()
                if _target == _expected:
                    check_ok(f"{_cmd_link_display}/hermes → correct target")
                else:
                    check_warn(
                        f"{_cmd_link_display}/hermes points to wrong target",
                        f"(→ {_target}, expected → {_expected})"
                    )
                    if should_fix:
                        _cmd_link.unlink()
                        _cmd_link.symlink_to(_venv_bin)
                        check_ok(f"Fixed symlink: {_cmd_link_display}/hermes → {_venv_bin}")
                        fixed_count += 1
                    else:
                        issues.append(f"Broken symlink at {_cmd_link_display}/hermes — run 'hermes doctor --fix'")
            elif _cmd_link.exists():
                # It's a regular file, not a symlink — possibly a wrapper script
                check_ok(f"{_cmd_link_display}/hermes exists (non-symlink)")
            else:
                check_fail(
                    f"{_cmd_link_display}/hermes not found",
                    "(hermes command may not work outside the venv)"
                )
                if should_fix:
                    _cmd_link_dir.mkdir(parents=True, exist_ok=True)
                    _cmd_link.symlink_to(_venv_bin)
                    check_ok(f"Created symlink: {_cmd_link_display}/hermes → {_venv_bin}")
                    fixed_count += 1

                    # Check if the link dir is on PATH
                    _path_dirs = os.environ.get("PATH", "").split(os.pathsep)
                    if str(_cmd_link_dir) not in _path_dirs:
                        check_warn(
                            f"{_cmd_link_display} is not on your PATH",
                            "(add it to your shell config: export PATH=\"$HOME/.local/bin:$PATH\")"
                        )
                        manual_issues.append(f"Add {_cmd_link_display} to your PATH")
                else:
                    issues.append(f"Missing {_cmd_link_display}/hermes symlink — run 'hermes doctor --fix'")

    _section("External Tools")
    # Git
    if _safe_which("git"):
        check_ok("git")
    else:
        check_warn("git not found", "(optional)")
    
    # ripgrep (optional, for faster file search)
    if _safe_which("rg"):
        check_ok("ripgrep (rg)", "(faster file search)")
    else:
        check_warn("ripgrep (rg) not found", "(file search uses grep fallback)")
        check_info(f"Install for faster search: {_system_package_install_cmd('ripgrep')}")
    
    # Docker (optional)
    terminal_env = os.getenv("TERMINAL_ENV", "local")
    try:
        from hermes_constants import is_container as _is_container
        running_in_container = _is_container()
    except Exception:
        running_in_container = False

    if running_in_container:
        # Inside our container the Docker terminal backend is not
        # configured by default (Docker-in-Docker isn't set up); the
        # local backend is the intended one. Skip the noisy "docker
        # not found" warning. If the user has explicitly chosen
        # TERMINAL_ENV=docker inside the container they likely mounted
        # /var/run/docker.sock, so fall through to the normal check.
        if terminal_env != "docker":
            check_info(
                "Running inside a container — using local terminal backend "
                "(docker-in-docker is not configured by default)"
            )
            # Skip to next section; Docker isn't relevant here.
            terminal_env = "local"
    if terminal_env == "docker":
        if _safe_which("docker"):
            # Check if docker daemon is running
            try:
                result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
            except subprocess.TimeoutExpired:
                result = None
            if result is not None and result.returncode == 0:
                check_ok("docker", "(daemon running)")
            else:
                _fail_and_issue("docker daemon not running", "", "Start Docker daemon", issues)
        else:
            _fail_and_issue(
                "docker not found",
                "(required for TERMINAL_ENV=docker)",
                "Install Docker or change TERMINAL_ENV",
                issues,
            )
    elif _safe_which("docker"):
        check_ok("docker", "(optional)")
    elif _is_termux():
        check_info("Docker backend is not available inside Termux (expected on Android)")
    elif running_in_container:
        pass  # already explained above
    else:
        check_warn("docker not found", "(optional)")
    
    # SSH (if using ssh backend)
    if terminal_env == "ssh":
        ssh_host = os.getenv("TERMINAL_SSH_HOST")
        if ssh_host:
            ssh_user = os.getenv("TERMINAL_SSH_USER")
            ssh_port = os.getenv("TERMINAL_SSH_PORT")
            ssh_key = os.getenv("TERMINAL_SSH_KEY")
            target = f"{ssh_user}@{ssh_host}" if ssh_user else ssh_host
            cmd = ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes"]
            if ssh_port:
                cmd += ["-p", ssh_port]
            if ssh_key:
                cmd += ["-i", os.path.expanduser(ssh_key)]
            cmd += [target, "echo ok"]
            # Try to connect
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=15
                )
            except subprocess.TimeoutExpired:
                result = None
            if result is not None and result.returncode == 0:
                check_ok(f"SSH connection to {ssh_host}")
            else:
                _fail_and_issue(f"SSH connection to {ssh_host}", "", f"Check SSH configuration for {ssh_host}", issues)
        else:
            _fail_and_issue(
                "TERMINAL_SSH_HOST not set",
                "(required for TERMINAL_ENV=ssh)",
                "Set TERMINAL_SSH_HOST in .env",
                issues,
            )
    
    # Daytona (if using daytona backend)
    if terminal_env == "daytona":
        daytona_key = os.getenv("DAYTONA_API_KEY")
        if daytona_key:
            check_ok("Daytona API key", "(configured)")
        else:
            _fail_and_issue(
                "DAYTONA_API_KEY not set",
                "(required for TERMINAL_ENV=daytona)",
                "Set DAYTONA_API_KEY environment variable",
                issues,
            )
        try:
            from daytona import Daytona  # noqa: F401 — SDK presence check
            check_ok("daytona SDK", "(installed)")
        except ImportError:
            _fail_and_issue(
                "daytona SDK not installed",
                "(pip install daytona)",
                "Install daytona SDK: pip install daytona",
                issues,
            )

    # Node.js + agent-browser (for browser automation tools)
    if _safe_which("node"):
        check_ok("Node.js")
        # Check if agent-browser is installed
        agent_browser_path = PROJECT_ROOT / "node_modules" / "agent-browser"
        agent_browser_ok = False
        _which_ab = shutil.which("agent-browser")
        if agent_browser_path.exists():
            check_ok("agent-browser (Node.js)", "(browser automation)")
            agent_browser_ok = True
        elif _which_ab and agent_browser_runnable(_which_ab):
            check_ok("agent-browser", "(browser automation)")
            agent_browser_ok = True
        elif _which_ab:
            # Found on PATH but won't run — almost always a dangling global
            # symlink left behind by agent-browser's npm postinstall after a
            # `hermes update` wiped node_modules (issue #48521).
            check_warn(
                "agent-browser found but not runnable",
                f"(broken symlink at {_which_ab}? run: npm install)",
            )
        elif _is_termux():
            check_info("agent-browser is not installed (expected in the tested Termux path)")
            check_info("Install it manually later with: npm install -g agent-browser && agent-browser install")
            check_info("Termux browser setup:")
            for step in _termux_browser_setup_steps(node_installed=True):
                check_info(step)
        else:
            check_warn("agent-browser not installed", "(run: npm install)")

        # Chromium presence — the browser tools silently fail to register when
        # agent-browser is found but no Playwright-managed Chromium is on disk
        # (tools/browser_tool.py::check_browser_requirements filters them out
        # before the agent ever sees them).  Reuse the exact predicate it uses
        # so the two checks cannot diverge.  Skip on Termux (not a tested
        # path).
        if agent_browser_ok and not _is_termux():
            try:
                # Lazy import: browser_tool is a ~150KB module we don't want
                # to eagerly load in every `hermes doctor` invocation.
                from tools.browser_tool import (
                    _chromium_installed,
                    _is_camofox_mode,
                    _get_cloud_provider,
                    _get_cdp_override,
                    _using_lightpanda_engine,
                )
            except Exception:
                # If browser_tool can't even import, that's a separate bug
                # surfaced elsewhere; don't crash doctor.
                pass
            else:
                # Only warn about Chromium if the installed engine actually
                # requires it: Camofox, CDP override, a cloud provider, or
                # Lightpanda all bypass the local Chromium requirement.
                skip_chromium_check = (
                    _is_camofox_mode()
                    or bool(_get_cdp_override())
                    or _get_cloud_provider() is not None
                    or _using_lightpanda_engine()
                )
                if not skip_chromium_check:
                    if _chromium_installed():
                        check_ok("Playwright Chromium", "(browser engine)")
                    else:
                        check_warn(
                            "Playwright Chromium not installed",
                            "(browser_* tools will be hidden from the agent)",
                        )
                        if sys.platform == "win32":
                            check_info(
                                f"Install with: cd {PROJECT_ROOT} && "
                                "npx playwright install chromium"
                            )
                        else:
                            check_info(
                                f"Install with: cd {PROJECT_ROOT} && "
                                "npx playwright install --with-deps chromium"
                            )
    elif _is_termux():
        check_info("Node.js not found (browser tools are optional in the tested Termux path)")
        check_info("Install Node.js on Termux with: pkg install nodejs")
        check_info("Termux browser setup:")
        for step in _termux_browser_setup_steps(node_installed=False):
            check_info(step)
    else:
        check_warn("Node.js not found", "(optional, needed for browser tools)")
    
    # npm audit for all Node.js packages
    _npm_bin = _safe_which("npm")
    if _npm_bin:
        # Each entry: (cwd, label, extra_audit_args)
        # PROJECT_ROOT is audited with --workspaces=false so that the apps/*
        # glob (which pulls in Electron, node-pty, etc.) is never resolved
        # for a routine security check. The web and ui-tui workspaces are
        # audited separately via --workspace flags. See #38772.
        # The WhatsApp bridge may live under a writable HERMES_HOME mirror
        # instead of the (possibly read-only) install tree in Docker — resolve
        # it through the shared helper so we audit the dir that actually holds
        # node_modules. See #49561.
        try:
            from gateway.platforms.whatsapp_common import resolve_whatsapp_bridge_dir
            _whatsapp_bridge_dir = resolve_whatsapp_bridge_dir()
        except Exception:
            _whatsapp_bridge_dir = PROJECT_ROOT / "scripts" / "whatsapp-bridge"
        npm_audit_targets = [
            (PROJECT_ROOT, "Browser tools (agent-browser)", ["--workspaces=false"]),
            (PROJECT_ROOT, "web workspace", ["--workspace", "web"]),
            (PROJECT_ROOT, "ui-tui workspace", ["--workspace", "ui-tui"]),
            (_whatsapp_bridge_dir, "WhatsApp bridge", []),
        ]
        for npm_dir, label, audit_extra in npm_audit_targets:
            # For workspace-scoped audits run from PROJECT_ROOT the
            # node_modules check must use the workspace root; standalone dirs
            # (whatsapp-bridge) check their own node_modules.
            check_dir = PROJECT_ROOT if audit_extra else npm_dir
            if not (check_dir / "node_modules").exists():
                continue
            try:
                # Use resolved absolute path so Windows can execute
                # npm.cmd (CreateProcessW can't run bare .cmd names).
                audit_result = subprocess.run(
                    [_npm_bin, "audit", "--json", *audit_extra],
                    cwd=str(npm_dir),
                    capture_output=True, text=True, timeout=30,
                )
                import json as _json
                audit_data = _json.loads(audit_result.stdout) if audit_result.stdout.strip() else {}
                vuln_count = audit_data.get("metadata", {}).get("vulnerabilities", {})
                critical = vuln_count.get("critical", 0)
                high = vuln_count.get("high", 0)
                moderate = vuln_count.get("moderate", 0)
                total = critical + high + moderate
                # Determine a scoped fix command for the remediation hint.
                if audit_extra and audit_extra[0] == "--workspace":
                    # Detection (`npm audit --workspace <name>`) is read-only and
                    # safe, but `npm audit fix --workspace <name>` crashes on
                    # current npm with "Cannot read properties of null (reading
                    # 'edgesOut')" — an arborist bug with workspace-filtered
                    # audit fix. The root-level `npm audit fix` can crash on the
                    # same tree with "isDescendantOf", so do not hand the user a
                    # manual fix command for these build-tool advisories.
                    fix_cmd = None
                elif audit_extra == ["--workspaces=false"]:
                    fix_cmd = f"cd {npm_dir} && npm audit fix --workspaces=false"
                else:
                    fix_cmd = f"cd {npm_dir} && npm audit fix"
                if total == 0:
                    check_ok(f"{label} deps", "(no known vulnerabilities)")
                elif critical > 0 or high > 0:
                    if fix_cmd:
                        vuln_detail = (
                            f"{critical} critical, {high} high, {moderate} moderate — run: {fix_cmd}"
                        )
                    else:
                        vuln_detail = (
                            f"{critical} critical, {high} high, {moderate} moderate — "
                            "build-tool advisory; clears via lockfile bump"
                        )
                    check_warn(
                        f"{label} deps",
                        f"({vuln_detail})"
                    )
                    if audit_extra and audit_extra[0] == "--workspace":
                        # The web/ui-tui workspace advisories are in build-time
                        # tooling (esbuild/vite, etc.), not runtime code that ships
                        # to users. Manual npm remediation may error with a known
                        # arborist crash (edgesOut / isDescendantOf) on this monorepo
                        # tree — in that case it is an npm bug, not a Hermes one.
                        check_info(
                            "  ^ build-time tooling (not runtime); if manual npm remediation "
                            "errors with an arborist crash it's a known npm bug — clears "
                            "via a lockfile bump"
                        )
                    issues.append(
                        f"{label} has {total} npm "
                        f"{'vulnerability' if total == 1 else 'vulnerabilities'}"
                    )
                else:
                    check_ok(
                        f"{label} deps",
                        f"({moderate} moderate "
                        f"{'vulnerability' if moderate == 1 else 'vulnerabilities'})",
                    )
            except Exception:
                pass

    if _is_termux():
        check_info("Termux compatibility fallbacks:")
        for note in _termux_install_all_fallback_notes():
            check_info(note)

    _section("API Connectivity")
    # Refactor: every connectivity probe below is HTTP-bound and fully
    # independent. Running them in series spent ~5s wall on a typical
    # workstation (2s of that was boto3's IMDS lookup for AWS credentials,
    # which times out unless you're actually on EC2). Threading them with
    # a small executor pool collapses the section to roughly the slowest
    # single probe — about 2s — without changing the output format.
    #
    # Each ``_probe_*`` helper is a pure function: takes its inputs,
    # makes one HTTP/SDK call, returns a ``_ConnectivityResult`` carrying
    # the line(s) to print and any issue strings to append. No globals,
    # no shared mutable state, no printing inside the workers.
    import concurrent.futures as _futures
    from collections import namedtuple as _namedtuple

    _ConnectivityResult = _namedtuple(
        "_ConnectivityResult", ["label", "lines", "issues"]
    )
    _probes: list = []  # list of (label, callable) submitted in display order

    def _probe_openrouter() -> _ConnectivityResult:
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            return _ConnectivityResult(
                "OpenRouter API",
                [(color("⚠", Colors.YELLOW), "OpenRouter API",
                  color("(not configured)", Colors.DIM))],
                [],
            )
        try:
            import httpx
            r = httpx.get(
                OPENROUTER_MODELS_URL,
                headers={"Authorization": f"Bearer {key}"},
                timeout=10,
            )
            if r.status_code == 200:
                return _ConnectivityResult(
                    "OpenRouter API",
                    [(color("✓", Colors.GREEN), "OpenRouter API", "")],
                    [],
                )
            if r.status_code == 401:
                return _ConnectivityResult(
                    "OpenRouter API",
                    [(color("✗", Colors.RED), "OpenRouter API",
                      color("(invalid API key)", Colors.DIM))],
                    ["Check OPENROUTER_API_KEY in .env"],
                )
            if r.status_code == 402:
                return _ConnectivityResult(
                    "OpenRouter API",
                    [(color("✗", Colors.RED), "OpenRouter API",
                      color("(out of credits — payment required)", Colors.DIM))],
                    ["OpenRouter account has insufficient credits. "
                     "Fix: run 'hermes config set model.provider <provider>' "
                     "to switch providers, or fund your OpenRouter account "
                     "at https://openrouter.ai/settings/credits"],
                )
            if r.status_code == 429:
                return _ConnectivityResult(
                    "OpenRouter API",
                    [(color("✗", Colors.RED), "OpenRouter API",
                      color("(rate limited)", Colors.DIM))],
                    ["OpenRouter rate limit hit — consider switching to "
                     "a different provider or waiting"],
                )
            return _ConnectivityResult(
                "OpenRouter API",
                [(color("✗", Colors.RED), "OpenRouter API",
                  color(f"(HTTP {r.status_code})", Colors.DIM))],
                [],
            )
        except Exception as e:
            return _ConnectivityResult(
                "OpenRouter API",
                [(color("✗", Colors.RED), "OpenRouter API",
                  color(f"({e})", Colors.DIM))],
                ["Check network connectivity"],
            )

    def _probe_anthropic() -> _ConnectivityResult:
        from hermes_cli.auth import get_anthropic_key
        key = get_anthropic_key()
        if not key:
            return _ConnectivityResult("Anthropic API", [], [])
        try:
            import httpx
            from agent.anthropic_adapter import (
                _is_oauth_token,
                _COMMON_BETAS,
                _OAUTH_ONLY_BETAS,
                _CONTEXT_1M_BETA,
            )
            headers = {"anthropic-version": "2023-06-01"}
            is_oauth = _is_oauth_token(key)
            if is_oauth:
                headers["Authorization"] = f"Bearer {key}"
                headers["anthropic-beta"] = ",".join(_COMMON_BETAS + _OAUTH_ONLY_BETAS)
            else:
                headers["x-api-key"] = key
            r = httpx.get(
                "https://api.anthropic.com/v1/models",
                headers=headers, timeout=10,
            )
            # Reactive recovery: OAuth subscriptions without 1M context reject the
            # request with 400 "long context beta is not yet available for this
            # subscription". Retry once with that beta stripped so the doctor
            # check doesn't falsely report Anthropic as unreachable.
            if (
                is_oauth
                and r.status_code == 400
                and "long context beta" in r.text.lower()
                and "not yet available" in r.text.lower()
            ):
                headers["anthropic-beta"] = ",".join(
                    [b for b in _COMMON_BETAS if b != _CONTEXT_1M_BETA]
                    + list(_OAUTH_ONLY_BETAS)
                )
                r = httpx.get(
                    "https://api.anthropic.com/v1/models",
                    headers=headers, timeout=10,
                )
            if r.status_code == 200:
                return _ConnectivityResult(
                    "Anthropic API",
                    [(color("✓", Colors.GREEN), "Anthropic API", "")],
                    [],
                )
            if r.status_code == 401:
                return _ConnectivityResult(
                    "Anthropic API",
                    [(color("✗", Colors.RED), "Anthropic API",
                      color("(invalid API key)", Colors.DIM))],
                    [],
                )
            return _ConnectivityResult(
                "Anthropic API",
                [(color("⚠", Colors.YELLOW), "Anthropic API",
                  color("(couldn't verify)", Colors.DIM))],
                [],
            )
        except Exception as e:
            return _ConnectivityResult(
                "Anthropic API",
                [(color("⚠", Colors.YELLOW), "Anthropic API",
                  color(f"({e})", Colors.DIM))],
                [],
            )

    def _probe_apikey_provider(pname, env_vars, default_url, base_env,
                               supports_health_check) -> _ConnectivityResult:
        key = ""
        for ev in env_vars:
            key = os.getenv(ev, "")
            if key:
                break
        if not key:
            return _ConnectivityResult(pname, [], [])
        label = pname.ljust(20)
        if not supports_health_check:
            return _ConnectivityResult(
                pname,
                [(color("✓", Colors.GREEN), label,
                  color("(key configured)", Colors.DIM))],
                [],
            )
        try:
            import httpx
            base = os.getenv(base_env, "") if base_env else ""
            # Auto-detect Kimi Code keys (sk-kimi-) → api.kimi.com/coding/v1
            # (OpenAI-compat surface, which exposes /models for health check).
            if not base and key.startswith("sk-kimi-"):
                base = "https://api.kimi.com/coding/v1"
            # Anthropic-compat endpoints (/anthropic, api.kimi.com/coding
            # with no /v1) don't support /models. Rewrite to OpenAI-compat
            # /v1 surface for health checks.
            if base and base.rstrip("/").endswith("/anthropic"):
                from agent.auxiliary_client import _to_openai_base_url
                base = _to_openai_base_url(base)
            if base_url_host_matches(base, "api.kimi.com") and base.rstrip("/").endswith("/coding"):
                base = base.rstrip("/") + "/v1"
            url = (base.rstrip("/") + "/models") if base else default_url
            headers = {
                "Authorization": f"Bearer {key}",
                "User-Agent": _HERMES_USER_AGENT,
            }
            if base_url_host_matches(base, "api.kimi.com"):
                headers["User-Agent"] = "claude-code/0.1.0"
            # Google's Generative Language API (generativelanguage.googleapis.com)
            # rejects ``Authorization: Bearer <api-key>`` with 401
            # ``ACCESS_TOKEN_TYPE_UNSUPPORTED`` — that header is reserved for
            # OAuth 2 access tokens, not plain API keys. Plain keys use
            # ``x-goog-api-key`` (or ``?key=``). Without this, a perfectly valid
            # GOOGLE_API_KEY/GEMINI_API_KEY always shows red in ``hermes doctor``.
            if url and base_url_host_matches(url, "generativelanguage.googleapis.com"):
                headers.pop("Authorization", None)
                headers["x-goog-api-key"] = key
            r = httpx.get(url, headers=headers, timeout=10)
            if (
                pname == "Alibaba/DashScope"
                and not base
                and r.status_code == 401
            ):
                r = httpx.get(
                    "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
                    headers=headers, timeout=10,
                )
            if r.status_code == 200:
                return _ConnectivityResult(
                    pname,
                    [(color("✓", Colors.GREEN), label, "")],
                    [],
                )
            if r.status_code == 401:
                return _ConnectivityResult(
                    pname,
                    [(color("✗", Colors.RED), label,
                      color("(invalid API key)", Colors.DIM))],
                    [f"Check {env_vars[0]} in .env"],
                )
            return _ConnectivityResult(
                pname,
                [(color("⚠", Colors.YELLOW), label,
                  color(f"(HTTP {r.status_code})", Colors.DIM))],
                [],
            )
        except Exception as e:
            return _ConnectivityResult(
                pname,
                [(color("⚠", Colors.YELLOW), label,
                  color(f"({e})", Colors.DIM))],
                [],
            )

    def _probe_bedrock() -> _ConnectivityResult:
        try:
            from agent.bedrock_adapter import (
                has_aws_credentials,
                resolve_aws_auth_env_var,
                resolve_bedrock_region,
            )
        except ImportError:
            return _ConnectivityResult("AWS Bedrock", [], [])
        if not has_aws_credentials():
            return _ConnectivityResult("AWS Bedrock", [], [])
        auth_var = resolve_aws_auth_env_var()
        region = resolve_bedrock_region()
        label = "AWS Bedrock".ljust(20)
        try:
            import boto3
            from botocore.config import Config as _BotoConfig
            # Trim retries on the actual Bedrock API call so a transient
            # failure doesn't pad the doctor run by 30+ seconds.
            cfg = _BotoConfig(
                connect_timeout=5,
                read_timeout=10,
                retries={"max_attempts": 1},
            )
            client = boto3.client("bedrock", region_name=region, config=cfg)
            resp = client.list_foundation_models()
            n = len(resp.get("modelSummaries", []))
            return _ConnectivityResult(
                "AWS Bedrock",
                [(color("✓", Colors.GREEN), label,
                  color(f"({auth_var}, {region}, {n} models)", Colors.DIM))],
                [],
            )
        except ImportError:
            return _ConnectivityResult(
                "AWS Bedrock",
                [(color("⚠", Colors.YELLOW), label,
                  color(f"(boto3 not installed — {sys.executable} -m pip install boto3)",
                        Colors.DIM))],
                [f"Install boto3 for Bedrock: {sys.executable} -m pip install boto3"],
            )
        except Exception as e:
            err_name = type(e).__name__
            return _ConnectivityResult(
                "AWS Bedrock",
                [(color("⚠", Colors.YELLOW), label,
                  color(f"({err_name}: {e})", Colors.DIM))],
                [f"AWS Bedrock: {err_name} — check IAM permissions for "
                 f"bedrock:ListFoundationModels"],
            )

    def _probe_azure_entra() -> _ConnectivityResult:
        """Probe Azure Foundry Entra ID auth, parallel to ``_probe_bedrock``.

        Skipped unless the active config has ``model.provider:
        azure-foundry`` AND ``model.auth_mode: entra_id`` — we don't probe
        the token-service / CLI chain for users on plain API-key Azure.

        Bounded by a 10s timeout (via
        :func:`agent.azure_identity_adapter.describe_active_credential`)
        so a slow token service can't pad the doctor run.
        """
        label = "Azure Foundry (Entra ID)".ljust(28)
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            model_cfg = cfg.get("model") if isinstance(cfg, dict) else {}
            if not isinstance(model_cfg, dict):
                return _ConnectivityResult("Azure Foundry (Entra ID)", [], [])
            cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
            auth_mode = str(model_cfg.get("auth_mode") or "").strip().lower()
            if cfg_provider != "azure-foundry" or auth_mode != "entra_id":
                return _ConnectivityResult("Azure Foundry (Entra ID)", [], [])
        except Exception:
            return _ConnectivityResult("Azure Foundry (Entra ID)", [], [])

        try:
            from agent.azure_identity_adapter import (
                EntraIdentityConfig,
                SCOPE_AI_AZURE_DEFAULT,
                describe_active_credential,
                has_azure_identity_installed,
            )
        except Exception as exc:
            return _ConnectivityResult(
                "Azure Foundry (Entra ID)",
                [(color("⚠", Colors.YELLOW), label,
                  color(f"(adapter import failed: {exc})", Colors.DIM))],
                [f"Azure Foundry adapter import failed: {exc}"],
            )

        if not has_azure_identity_installed():
            return _ConnectivityResult(
                "Azure Foundry (Entra ID)",
                [(color("⚠", Colors.YELLOW), label,
                  color("(azure-identity not installed)", Colors.DIM))],
                [f"Install azure-identity: {sys.executable} -m pip install azure-identity"],
            )

        base_url = str(model_cfg.get("base_url") or "").strip()
        entra_cfg = model_cfg.get("entra") or {}
        if not isinstance(entra_cfg, dict):
            entra_cfg = {}
        scope = (
            str(entra_cfg.get("scope") or "").strip()
            or SCOPE_AI_AZURE_DEFAULT
        )
        config = EntraIdentityConfig(
            scope=scope,
        )
        info = describe_active_credential(config=config, timeout_seconds=10.0)
        if info.get("ok"):
            env_sources = info.get("env_sources") or []
            tag = ", ".join(env_sources) if env_sources else "default credential chain"
            return _ConnectivityResult(
                "Azure Foundry (Entra ID)",
                [(color("✓", Colors.GREEN), label,
                  color(f"({tag}, scope={scope})", Colors.DIM))],
                [],
            )
        err = info.get("error") or "credential chain exhausted"
        hint = info.get("hint") or (
            "Run `az login`, set AZURE_TENANT_ID/AZURE_CLIENT_ID/"
            "AZURE_CLIENT_SECRET, or attach a managed identity to this VM."
        )
        return _ConnectivityResult(
            "Azure Foundry (Entra ID)",
            [(color("⚠", Colors.YELLOW), label,
              color(f"({err})", Colors.DIM))],
            [f"Azure Foundry Entra: {err}. {hint}"],
        )

    # Build the probe submission list in display order
    _probes.append(("OpenRouter API", _probe_openrouter))
    _probes.append(("Anthropic API", _probe_anthropic))

    global _APIKEY_PROVIDERS_CACHE
    if _APIKEY_PROVIDERS_CACHE is None:
        _APIKEY_PROVIDERS_CACHE = _build_apikey_providers_list()
    for _entry in _APIKEY_PROVIDERS_CACHE:
        _pname, _env_vars, _default_url, _base_env, _supports = _entry
        # Capture loop vars by binding default args — without this, all closures
        # would share the final iteration's values and every probe would hit
        # the last provider's URL.
        _probes.append((_pname, lambda p=_pname, e=_env_vars, u=_default_url,
                                       b=_base_env, s=_supports:
                                _probe_apikey_provider(p, e, u, b, s)))

    _probes.append(("AWS Bedrock", _probe_bedrock))
    _probes.append(("Azure Foundry (Entra ID)", _probe_azure_entra))

    # Print a single status line so users see something happening, then
    # fan out. ``\r`` clears it once the first real result line lands.
    print(f"  {color(f'Running {len(_probes)} connectivity checks in parallel…', Colors.DIM)}",
          end="", flush=True)

    # Disable boto3's EC2 instance-metadata-service probe for the duration
    # of the parallel block. boto's default credential chain tries
    # 169.254.169.254 with a multi-second timeout when we're not on EC2,
    # which dominated the section's wall time before this fix
    # (~2s on a developer laptop, even with the rest parallelized).
    # Set on the parent thread before submitting work so the env-var
    # mutation never races with another worker. has_aws_credentials() in
    # the bedrock probe already gates on real env-var creds, so IMDS is
    # never the legitimate source for `hermes doctor`.
    _imds_prev = os.environ.get("AWS_EC2_METADATA_DISABLED")
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    try:
        # 8 workers is plenty — each probe is a single HTTP call plus a TLS
        # handshake. More than that wastes thread-startup cost and risks
        # noisy output if anything ever printed from inside a worker.
        with _futures.ThreadPoolExecutor(max_workers=8,
                                         thread_name_prefix="doctor-probe") as _ex:
            _futures_in_order = [_ex.submit(_fn) for _, _fn in _probes]
            _results = [_f.result() for _f in _futures_in_order]
    finally:
        if _imds_prev is None:
            os.environ.pop("AWS_EC2_METADATA_DISABLED", None)
        else:
            os.environ["AWS_EC2_METADATA_DISABLED"] = _imds_prev

    # Clear the "Running …" line and print all results in submission order.
    print("\r" + " " * 70 + "\r", end="")
    for _r in _results:
        for _glyph, _label, _detail in _r.lines:
            if _detail:
                print(f"  {_glyph} {_label} {_detail}")
            else:
                print(f"  {_glyph} {_label}")
        _issues_to_add = list(_r.issues)
        if _issues_to_add and _has_healthy_oauth_fallback_for_apikey_provider(_r.label):
            _issues_to_add = []
        for _issue in _issues_to_add:
            issues.append(_issue)

    _section("Tool Availability")
    try:
        # Add project root to path for imports
        sys.path.insert(0, str(PROJECT_ROOT))
        from model_tools import check_tool_availability, TOOLSET_REQUIREMENTS
        
        available, unavailable = check_tool_availability()
        available, unavailable = _apply_doctor_tool_availability_overrides(available, unavailable)
        
        for tid in available:
            info = TOOLSET_REQUIREMENTS.get(tid, {})
            check_ok(info.get("name", tid), _doctor_tool_availability_detail(tid))
        
        for item in unavailable:
            env_vars = item.get("missing_vars") or item.get("env_vars") or []
            if env_vars:
                vars_str = ", ".join(env_vars)
                check_warn(item["name"], f"(missing {vars_str})")
            else:
                check_warn(item["name"], "(system dependency not met)")

        # Count missing API-key requirements only for toolsets enabled in the
        # current CLI platform. Default-off or explicitly disabled toolsets may
        # still show warnings above, but should not pollute the final summary.
        api_disabled = _missing_api_key_toolsets_for_summary(unavailable)
        if api_disabled:
            issues.append("Run 'hermes setup' to configure missing API keys for full tool access")
    except Exception as e:
        check_warn("Could not check tool availability", f"({e})")
    
    _section("Skills Hub")
    hub_dir = HERMES_HOME / "skills" / ".hub"
    if hub_dir.exists():
        check_ok("Skills Hub directory exists")
        lock_file = hub_dir / "lock.json"
        if lock_file.exists():
            try:
                import json
                lock_data = json.loads(lock_file.read_text())
                count = len(lock_data.get("installed", {}))
                check_ok(f"Lock file OK ({count} hub-installed skill(s))")
            except Exception:
                check_warn("Lock file", "(corrupted or unreadable)")
        quarantine = hub_dir / "quarantine"
        q_count = sum(1 for d in quarantine.iterdir() if d.is_dir()) if quarantine.exists() else 0
        if q_count > 0:
            check_warn(f"{q_count} skill(s) in quarantine", "(pending review)")
    else:
        check_warn("Skills Hub directory not initialized", "(run: hermes skills list)")

    from hermes_cli.config import get_env_value

    def _gh_authenticated() -> bool:
        """Check if gh CLI is authenticated via token file or device flow."""
        try:
            result = subprocess.run(
                ["gh", "auth", "status", "--json", "authenticated"],
                capture_output=True, timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    github_token = get_env_value("GITHUB_TOKEN") or get_env_value("GH_TOKEN")
    if github_token:
        check_ok("GitHub token configured (authenticated API access)")
    elif _gh_authenticated():
        check_ok("GitHub authenticated via gh CLI", "(full API access — no GITHUB_TOKEN needed)")
    else:
        check_warn("No GITHUB_TOKEN", f"(60 req/hr rate limit — set in {_DHH}/.env for better rates)")

    _section("Memory Provider")
    _active_memory_provider = ""
    try:
        import yaml as _yaml
        _mem_cfg_path = HERMES_HOME / "config.yaml"
        if _mem_cfg_path.exists():
            with open(_mem_cfg_path, encoding="utf-8") as _f:
                _raw_cfg = _yaml.safe_load(_f) or {}
            try:
                from hermes_cli import managed_scope
                _raw_cfg = managed_scope.apply_managed_overlay(_raw_cfg)
            except Exception:
                pass
            _active_memory_provider = (_raw_cfg.get("memory") or {}).get("provider", "")
    except Exception:
        pass

    if not _active_memory_provider:
        check_ok("Built-in memory active", "(no external provider configured — this is fine)")
    elif _active_memory_provider == "honcho":
        try:
            from plugins.memory.honcho.client import HonchoClientConfig, resolve_config_path
            hcfg = HonchoClientConfig.from_global_config()
            _honcho_cfg_path = resolve_config_path()

            if not _honcho_cfg_path.exists():
                # Config file missing — but env var fallback may have resolved it.
                # Only warn if the config didn't actually resolve from env vars.
                if hcfg.api_key or hcfg.base_url:
                    check_ok(
                        "Honcho configured via environment variables",
                        f"config file {_honcho_cfg_path} not found, using HONCHO_API_KEY env var",
                    )
                else:
                    check_warn("Honcho config not found", "run: hermes memory setup")
            elif not hcfg.enabled:
                check_info(f"Honcho disabled (set enabled: true in {_honcho_cfg_path} to activate)")
            elif not (hcfg.api_key or hcfg.base_url):
                _fail_and_issue(
                    "Honcho API key or base URL not set",
                    "run: hermes memory setup",
                    "No Honcho API key — run 'hermes memory setup'",
                    issues,
                )
            else:
                from plugins.memory.honcho.client import get_honcho_client, reset_honcho_client
                reset_honcho_client()
                try:
                    get_honcho_client(hcfg)
                    check_ok(
                        "Honcho connected",
                        f"workspace={hcfg.workspace_id} mode={hcfg.recall_mode} freq={hcfg.write_frequency}",
                    )
                except Exception as _e:
                    _fail_and_issue("Honcho connection failed", str(_e), f"Honcho unreachable: {_e}", issues)
        except ImportError:
            _fail_and_issue(
                "honcho-ai not installed",
                "pip install honcho-ai",
                "Honcho is set as memory provider but honcho-ai is not installed",
                issues,
            )
        except Exception as _e:
            check_warn("Honcho check failed", str(_e))
    elif _active_memory_provider == "mem0":
        try:
            from plugins.memory.mem0 import _load_config as _load_mem0_config
            mem0_cfg = _load_mem0_config()
            mem0_key = mem0_cfg.get("api_key", "")
            if mem0_key:
                check_ok("Mem0 API key configured")
                check_info(f"user_id={mem0_cfg.get('user_id', '?')}  agent_id={mem0_cfg.get('agent_id', '?')}")
            else:
                _fail_and_issue(
                    "Mem0 API key not set",
                    "(set MEM0_API_KEY in .env or run hermes memory setup)",
                    "Mem0 is set as memory provider but API key is missing",
                    issues,
                )
        except ImportError:
            _fail_and_issue(
                "Mem0 plugin not loadable",
                "pip install mem0ai",
                "Mem0 is set as memory provider but mem0ai is not installed",
                issues,
            )
        except Exception as _e:
            check_warn("Mem0 check failed", str(_e))
    else:
        # Generic check for other memory providers (openviking, hindsight, etc.)
        try:
            from plugins.memory import load_memory_provider
            _provider = load_memory_provider(_active_memory_provider)
            if _provider and _provider.is_available():
                check_ok(f"{_active_memory_provider} provider active")
            elif _provider:
                check_warn(f"{_active_memory_provider} configured but not available", "run: hermes memory status")
            else:
                check_warn(f"{_active_memory_provider} plugin not found", "run: hermes memory setup")
        except Exception as _e:
            check_warn(f"{_active_memory_provider} check failed", str(_e))

    try:
        from hermes_cli.profiles import list_profiles, _get_wrapper_dir, profile_exists
        import re as _re

        named_profiles = [p for p in list_profiles() if not p.is_default]
        if named_profiles:
            _section("Profiles")
            check_ok(f"{len(named_profiles)} profile(s) found")
            wrapper_dir = _get_wrapper_dir()
            for p in named_profiles:
                parts = []
                if p.gateway_running:
                    parts.append("gateway running")
                if p.model:
                    parts.append(p.model[:30])
                if not (p.path / "config.yaml").exists():
                    parts.append("⚠ missing config")
                if not (p.path / ".env").exists():
                    parts.append("no .env")
                wrapper = wrapper_dir / p.name
                if not wrapper.exists():
                    parts.append("no alias")
                status = ", ".join(parts) if parts else "configured"
                check_ok(f"  {p.name}: {status}")

            # Check for orphan wrappers
            if wrapper_dir.is_dir():
                for wrapper in wrapper_dir.iterdir():
                    if not wrapper.is_file():
                        continue
                    try:
                        content = wrapper.read_text()
                        if "hermes -p" in content:
                            _m = _re.search(r"hermes -p (\S+)", content)
                            if _m and not profile_exists(_m.group(1)):
                                check_warn(f"Orphan alias: {wrapper.name} → profile '{_m.group(1)}' no longer exists")
                    except Exception:
                        pass
    except ImportError:
        pass
    except Exception:
        pass

    print()
    remaining_issues = issues + manual_issues
    if should_fix and fixed_count > 0:
        print(color("─" * 60, Colors.GREEN))
        print(color(f"  Fixed {fixed_count} issue(s).", Colors.GREEN, Colors.BOLD), end="")
        if remaining_issues:
            print(color(f" {len(remaining_issues)} issue(s) require manual intervention.", Colors.YELLOW, Colors.BOLD))
        else:
            print()
        print()
        if remaining_issues:
            for i, issue in enumerate(remaining_issues, 1):
                print(f"  {i}. {issue}")
            print()
    elif remaining_issues:
        print(color("─" * 60, Colors.YELLOW))
        print(color(f"  Found {len(remaining_issues)} issue(s) to address:", Colors.YELLOW, Colors.BOLD))
        print()
        for i, issue in enumerate(remaining_issues, 1):
            print(f"  {i}. {issue}")
        print()
        if not should_fix:
            print(color("  Tip: run 'hermes doctor --fix' to auto-fix what's possible.", Colors.DIM))
    else:
        print(color("─" * 60, Colors.GREEN))
        print(color("  All checks passed! 🎉", Colors.GREEN, Colors.BOLD))
    
    print()
