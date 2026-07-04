"""
Status command for hermes CLI.

Shows the status of all Hermes Agent components.
"""

import os
import sys
import subprocess  # noqa: F401 — re-exported for tests that monkeypatch status.subprocess to guard against regressions
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

from hermes_cli.auth import AuthError, resolve_provider
from hermes_cli.colors import Colors, color
from hermes_cli.config import get_env_path, get_env_value, get_hermes_home, load_config
from hermes_cli.models import provider_label
from hermes_cli.nous_account import (
    format_nous_portal_entitlement_message,
    get_nous_portal_account_info,
)
from hermes_cli.nous_subscription import get_nous_subscription_features
from hermes_cli.runtime_provider import resolve_requested_provider
from hermes_constants import OPENROUTER_MODELS_URL
from tools.tool_backend_helpers import managed_nous_tools_enabled

REALTIME_VOICE_KAME_STATUS_LATENCIES = (
    ("speech_boundary_to_final_transcript", "final_transcript"),
    ("final_transcript_to_interface_decision", "reflex"),
    ("interface_decision_to_local_first_audio", "local_audio"),
    ("interface_decision_to_first_audio", "kame_audio"),
    ("interface_decision_to_oracle_accepted", "oracle_accept"),
    ("oracle_accepted_to_first_token", "oracle_token"),
    ("oracle_first_token_to_first_spoken_text", "oracle_speech"),
    ("oracle_first_token_to_first_tts_audio", "oracle_tts"),
    ("first_tts_audio_to_playback_start", "playback"),
    ("speech_end_to_playback_start", "audible"),
    ("oracle_verbatim_asr", "asr_record"),
    ("barge_in_confirmed_to_playback_stopped", "barge_stop"),
)

def check_mark(ok: bool) -> str:
    if ok:
        return color("✓", Colors.GREEN)
    return color("✗", Colors.RED)

def redact_key(key: str) -> str:
    """Redact an API key for display.

    Thin wrapper over :func:`agent.redact.mask_secret`. Preserves the
    "(not set)" placeholder in dim color to match ``hermes config``'s
    output (previously this variant was missing the DIM color —
    consolidated via PR that also introduced ``mask_secret``).
    """
    from agent.redact import mask_secret
    return mask_secret(key, empty=color("(not set)", Colors.DIM))


def _format_iso_timestamp(value) -> str:
    """Format ISO timestamps for status output, converting to local timezone."""
    if not value or not isinstance(value, str):
        return "(unknown)"
    from datetime import datetime, timezone
    text = value.strip()
    if not text:
        return "(unknown)"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return value
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _configured_model_label(config: dict) -> str:
    """Return the configured default model from config.yaml."""
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        model = (model_cfg.get("default") or model_cfg.get("name") or "").strip()
    elif isinstance(model_cfg, str):
        model = model_cfg.strip()
    else:
        model = ""
    return model or "(not set)"


def _effective_provider_label() -> str:
    """Return the provider label matching current CLI runtime resolution."""
    requested = resolve_requested_provider()
    try:
        effective = resolve_provider(requested)
    except AuthError:
        effective = requested or "auto"

    if effective == "openrouter" and get_env_value("OPENAI_BASE_URL"):
        effective = "custom"

    return provider_label(effective)


def _realtime_voice_status_payload() -> Mapping[str, Any]:
    from hermes_cli.web_server import _realtime_voice_status_payload as payload

    return payload(probe_health=True)


def _format_realtime_voice_bool(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def _print_realtime_voice_status() -> None:
    print()
    print(color("◆ Realtime Voice", Colors.CYAN, Colors.BOLD))

    try:
        payload = _realtime_voice_status_payload()
    except Exception as exc:
        print(f"  Status:       {color('unknown', Colors.DIM)}")
        print(f"  Detail:       could not check realtime voice ({exc})")
        return

    enabled = payload.get("enabled") is True
    available = payload.get("available") is True
    unavailable_reason = str(payload.get("unavailable_reason") or "").strip()
    engine = str(payload.get("engine") or "text_oracle_tts")
    quality = payload.get("conversation_quality") if isinstance(payload.get("conversation_quality"), Mapping) else {}
    production = payload.get("production_readiness") if isinstance(payload.get("production_readiness"), Mapping) else {}
    sidecar = payload.get("sidecar") if isinstance(payload.get("sidecar"), Mapping) else {}

    if not enabled:
        status_label = "disabled"
    elif available:
        status_label = "available"
    else:
        status_label = f"unavailable ({unavailable_reason or 'unknown'})"

    print(f"  Status:       {check_mark(enabled and available)} {status_label}")
    print(f"  Engine:       {engine}")
    mode = str(quality.get("mode") or "unknown")
    reason = str(quality.get("reason") or "unknown")
    live_like = quality.get("live_like")
    live_label = _format_realtime_voice_bool(live_like)
    print(f"  Quality:      {check_mark(live_like is True)} {mode} ({reason})")
    print(f"  Live-like:    {live_label}")
    production_ready = production.get("ready")
    production_level = str(production.get("level") or "unknown")
    production_issues = production.get("issues") if isinstance(production.get("issues"), list) else []
    issue_suffix = f" ({', '.join(str(issue) for issue in production_issues[:3])})" if production_issues else ""
    print(f"  Production:   {check_mark(production_ready is True)} {production_level}{issue_suffix}")
    evidence_line = _realtime_voice_evidence_line(production)
    if evidence_line:
        print(f"  Evidence:     {evidence_line}")
    review_line = _realtime_voice_launch_review_line(production)
    if review_line:
        print(f"  Review:       {review_line}")
    print(f"  Require live: {_format_realtime_voice_bool(payload.get('require_live_like'))}")
    sidecar_mode = str(sidecar.get("mode") or "none")
    healthy = sidecar.get("healthy")
    print(f"  Sidecar:      {sidecar_mode} (healthy: {_format_realtime_voice_bool(healthy)})")


def _realtime_voice_evidence_line(production: Mapping[str, Any]) -> str:
    evidence = production.get("evidence")
    if not isinstance(evidence, Mapping):
        return ""
    runs = evidence.get("runs")
    min_runs = evidence.get("min_runs")
    if not isinstance(runs, int) or not isinstance(min_runs, int):
        return ""
    parts = [f"runs {runs}/{min_runs}"]
    summary = evidence.get("summary")
    latency = summary.get("latency_ms") if isinstance(summary, Mapping) else None
    if isinstance(latency, Mapping):
        metric_parts = [
            _realtime_voice_latency_summary_part(latency, "audio_to_partial_transcript", "partial"),
            _realtime_voice_latency_summary_part(latency, "final_transcript_to_first_text", "text"),
            _realtime_voice_latency_summary_part(latency, "final_transcript_to_first_audio", "audio"),
            _realtime_voice_latency_summary_part(latency, "barge_in_ack", "barge"),
            *(
                _realtime_voice_latency_summary_part(latency, key, label)
                for key, label in REALTIME_VOICE_KAME_STATUS_LATENCIES
            ),
        ]
        parts.extend(part for part in metric_parts if part)
    route_line = _realtime_voice_kame_route_summary_part(summary.get("kame_routes") if isinstance(summary, Mapping) else None)
    if route_line:
        parts.append(route_line)
    provenance_line = _realtime_voice_kame_reflex_provenance_part(
        summary.get("kame_reflex_provenance") if isinstance(summary, Mapping) else None
    )
    if provenance_line:
        parts.append(provenance_line)
    parts.extend(_realtime_voice_stack_latency_summary_parts(summary))
    return "; ".join(parts)


def _realtime_voice_launch_review_line(production: Mapping[str, Any]) -> str:
    review = production.get("launch_review")
    if not isinstance(review, Mapping) or review.get("required") is not True:
        return ""
    if review.get("verified") is True:
        reviewed_at = str(review.get("reviewed_at") or "").strip()
        return f"passed ({reviewed_at})" if reviewed_at else "passed"
    issues = review.get("issues") if isinstance(review.get("issues"), list) else []
    if issues:
        return f"pending ({', '.join(str(issue) for issue in issues[:3])})"
    return "pending"


def _realtime_voice_latency_summary_part(latency: Mapping[str, Any], key: str, label: str) -> str:
    value = latency.get(key)
    if not isinstance(value, Mapping) or not value.get("count"):
        return ""
    spans = []
    for percentile in ("p50", "p90", "p95"):
        if value.get(percentile) is not None:
            spans.append(f"{percentile}={value.get(percentile)}ms")
    if value.get("max") is not None:
        spans.append(f"max={value.get('max')}ms")
    if not spans:
        return ""
    return f"{label} {' '.join(spans)}"


def _realtime_voice_stack_latency_summary_parts(summary: Any) -> list[str]:
    if not isinstance(summary, Mapping):
        return []
    latency_by_stack = summary.get("latency_by_stack")
    if not isinstance(latency_by_stack, Mapping):
        return []
    parts: list[str] = []
    for stack_key, stack_summary in sorted(latency_by_stack.items()):
        if not isinstance(stack_summary, Mapping):
            continue
        stack_latency = stack_summary.get("latency_ms")
        if not isinstance(stack_latency, Mapping):
            continue
        audio = _realtime_voice_latency_summary_part(
            stack_latency,
            "interface_decision_to_first_audio",
            "kame_audio",
        )
        if not audio:
            audio = _realtime_voice_latency_summary_part(
                stack_latency,
                "interface_decision_to_local_first_audio",
                "local_audio",
            )
        if not audio:
            audio = _realtime_voice_latency_summary_part(
                stack_latency,
                "final_transcript_to_first_audio",
                "audio",
            )
        if not audio:
            audio = _realtime_voice_latency_summary_part(
                stack_latency,
                "audio_to_partial_transcript",
                "partial",
            )
        if not audio:
            continue
        stack = stack_summary.get("stack") if isinstance(stack_summary.get("stack"), Mapping) else {}
        frontend = _stack_label(stack, "frontend_provider", "frontend_model", default="unknown_frontend")
        oracle = str(stack.get("oracle_authority") or "Hermes /model").strip() if isinstance(stack, Mapping) else ""
        tts = _stack_label(stack, "tts_provider", "tts_model", default="unknown_tts")
        stack_id = str(stack_key or "unknown_stack")
        provenance = _realtime_voice_kame_reflex_provenance_part(stack_summary.get("kame_reflex_provenance"))
        provenance_suffix = f" {provenance}" if provenance else ""
        parts.append(
            f"stack {stack_id} {audio} frontend={frontend} "
            f"oracle={oracle or 'unknown'} tts={tts}{provenance_suffix}"
        )
    return parts


def _realtime_voice_kame_route_summary_part(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    total = _positive_int(value.get("total"))
    if total <= 0:
        return ""
    counts = value.get("counts") if isinstance(value.get("counts"), Mapping) else {}
    route_parts = [
        f"{route}={_positive_int(counts.get(route))}"
        for route in ("local", "defer", "oracle_direct", "reject_or_clarify")
        if _positive_int(counts.get(route)) > 0
    ]
    oracle_avoided = _positive_int(value.get("oracle_avoided"))
    oracle_required = _positive_int(value.get("oracle_required"))
    rate = value.get("oracle_avoidance_rate")
    try:
        rate_text = f"{float(rate) * 100:.1f}%"
    except (TypeError, ValueError):
        rate_text = "unknown"
    return (
        f"kame_routes total={total} oracle_avoided={oracle_avoided} "
        f"oracle_required={oracle_required} avoidance={rate_text} "
        + " ".join(route_parts)
    ).strip()


def _realtime_voice_kame_reflex_provenance_part(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    total = _positive_int(value.get("total"))
    if total <= 0:
        return ""
    input_sources = value.get("input_sources") if isinstance(value.get("input_sources"), Mapping) else {}
    reflex_providers = value.get("reflex_providers") if isinstance(value.get("reflex_providers"), Mapping) else {}
    source_parts = [
        f"{source}={_positive_int(count)}"
        for source, count in sorted(input_sources.items())
        if _positive_int(count) > 0
    ]
    provider_parts = [
        f"{provider}={_positive_int(count)}"
        for provider, count in sorted(reflex_providers.items())
        if _positive_int(count) > 0
    ]
    fallback = _positive_int(value.get("fallback"))
    fallback_label = "fallback_only" if value.get("fallback_only") is True else f"fallback={fallback}"
    parts = [
        f"kame_reflex total={total}",
        f"native_audio={_positive_int(value.get('native_audio'))}",
        f"vllm={_positive_int(value.get('vllm'))}",
        fallback_label,
    ]
    if source_parts:
        parts.append("sources " + " ".join(source_parts))
    if provider_parts:
        parts.append("providers " + " ".join(provider_parts))
    return " ".join(parts)


def _stack_label(stack: Mapping[str, Any], provider_key: str, model_key: str, *, default: str) -> str:
    provider = str(stack.get(provider_key) or "").strip()
    model = str(stack.get(model_key) or "").strip()
    if provider and model:
        return f"{provider}/{model}"
    return provider or model or default


def _positive_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


from hermes_constants import is_termux as _is_termux


def show_status(args):
    """Show status of all Hermes Agent components."""
    deep = getattr(args, 'deep', False)

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│                 ⚕ Hermes Agent Status                  │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))

    # =========================================================================
    # Environment
    # =========================================================================
    print()
    print(color("◆ Environment", Colors.CYAN, Colors.BOLD))
    print(f"  Project:      {PROJECT_ROOT}")
    print(f"  Python:       {sys.version.split()[0]}")

    env_path = get_env_path()
    print(f"  .env file:    {check_mark(env_path.exists())} {'exists' if env_path.exists() else 'not found'}")

    try:
        config = load_config()
    except Exception:
        config = {}

    print(f"  Model:        {_configured_model_label(config)}")
    print(f"  Provider:     {_effective_provider_label()}")

    # =========================================================================
    # API Keys
    # =========================================================================
    print()
    print(color("◆ API Keys", Colors.CYAN, Colors.BOLD))

    # Values may be a single env var name (str) or a tuple of alternates (first found wins).
    keys: dict[str, str | tuple[str, ...]] = {
        "OpenRouter": "OPENROUTER_API_KEY",
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN"),
        "Google / Gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        "DeepSeek": "DEEPSEEK_API_KEY",
        "xAI / Grok": "XAI_API_KEY",
        "NVIDIA NIM": "NVIDIA_API_KEY",
        "Z.AI / GLM": "GLM_API_KEY",
        "Kimi": "KIMI_API_KEY",
        "StepFun Step Plan": "STEPFUN_API_KEY",
        "MiniMax": "MINIMAX_API_KEY",
        "MiniMax-CN": "MINIMAX_CN_API_KEY",
        "Firecrawl": "FIRECRAWL_API_KEY",
        "Tavily": "TAVILY_API_KEY",
        "Browser Use": "BROWSER_USE_API_KEY",  # Optional — local browser works without this
        "Browserbase": "BROWSERBASE_API_KEY",  # Optional — direct credentials only
        "FAL": "FAL_KEY",
        "ElevenLabs": "ELEVENLABS_API_KEY",
        "GitHub": "GITHUB_TOKEN",
    }

    def _resolve_env(env_ref) -> str:
        """Return first non-empty env var value from a str or tuple of names."""
        if isinstance(env_ref, tuple):
            for candidate in env_ref:
                v = get_env_value(candidate) or ""
                if v:
                    return v
            return ""
        return get_env_value(env_ref) or ""

    for name, env_ref in keys.items():
        # Anthropic already has a dedicated lookup below; keep that as the
        # single source of truth (it also resolves OAuth tokens), skip here
        # so we don't print two "Anthropic" rows.
        if name == "Anthropic":
            continue
        value = _resolve_env(env_ref)
        has_key = bool(value)
        display = redact_key(value)
        print(f"  {name:<12}  {check_mark(has_key)} {display}")

    from hermes_cli.auth import get_anthropic_key
    anthropic_value = get_anthropic_key()
    anthropic_display = redact_key(anthropic_value)
    print(f"  {'Anthropic':<12}  {check_mark(bool(anthropic_value))} {anthropic_display}")

    _print_realtime_voice_status()

    # =========================================================================
    # Auth Providers (OAuth)
    # =========================================================================
    print()
    print(color("◆ Auth Providers", Colors.CYAN, Colors.BOLD))

    try:
        from hermes_cli.auth import (
            get_nous_auth_status,
            get_codex_auth_status,
            get_qwen_auth_status,
            get_minimax_oauth_auth_status,
        )
        nous_status = get_nous_auth_status()
        codex_status = get_codex_auth_status()
        qwen_status = get_qwen_auth_status()
        minimax_status = get_minimax_oauth_auth_status()
    except Exception:
        nous_status = {}
        codex_status = {}
        qwen_status = {}
        minimax_status = {}

    nous_account_info = None
    if (
        nous_status.get("logged_in")
        or nous_status.get("access_token")
        or nous_status.get("portal_base_url")
        or nous_status.get("inference_credential_present")
        or nous_status.get("error_code")
    ):
        try:
            nous_account_info = get_nous_portal_account_info()
        except Exception:
            nous_account_info = None

    nous_logged_in = bool(
        nous_status.get("logged_in")
        or (nous_account_info and nous_account_info.logged_in)
    )
    nous_inference_present = bool(
        nous_status.get("inference_credential_present")
        or (nous_account_info and nous_account_info.inference_credential_present)
    )
    nous_error = nous_status.get("error")
    if nous_logged_in:
        nous_label = "logged in"
    elif nous_inference_present:
        nous_label = "not logged in (Nous inference key configured)"
    else:
        nous_label = "not logged in (run: hermes portal)"
    print(
        f"  {'Nous Portal':<12}  {check_mark(nous_logged_in)} "
        f"{nous_label}"
    )
    portal_url = nous_status.get("portal_base_url") or "(unknown)"
    inference_url = (
        nous_status.get("inference_base_url")
        or (nous_account_info.inference_base_url if nous_account_info else None)
    )
    access_exp = _format_iso_timestamp(nous_status.get("access_expires_at"))
    key_exp = _format_iso_timestamp(nous_status.get("agent_key_expires_at"))
    refresh_label = "yes" if nous_status.get("has_refresh_token") else "no"
    if nous_logged_in or portal_url != "(unknown)" or nous_error:
        print(f"    Portal URL: {portal_url}")
    if nous_inference_present and inference_url:
        print(f"    Inference:  {inference_url}")
    if nous_logged_in or nous_status.get("access_expires_at"):
        print(f"    Access exp: {access_exp}")
    if nous_logged_in or nous_inference_present or nous_status.get("agent_key_expires_at"):
        print(f"    Key exp:    {key_exp}")
    if nous_logged_in or nous_status.get("has_refresh_token"):
        print(f"    Refresh:    {refresh_label}")
    if nous_error:
        print(f"    Error:      {nous_error}")

    codex_logged_in = bool(codex_status.get("logged_in"))
    print(
        f"  {'OpenAI Codex':<12}  {check_mark(codex_logged_in)} "
        f"{'logged in' if codex_logged_in else 'not logged in (run: hermes model)'}"
    )
    codex_auth_file = codex_status.get("auth_store")
    if codex_auth_file:
        print(f"    Auth file:  {codex_auth_file}")
    codex_last_refresh = _format_iso_timestamp(codex_status.get("last_refresh"))
    if codex_status.get("last_refresh"):
        print(f"    Refreshed:  {codex_last_refresh}")
    if codex_status.get("error") and not codex_logged_in:
        print(f"    Error:      {codex_status.get('error')}")

    qwen_logged_in = bool(qwen_status.get("logged_in"))
    print(
        f"  {'Qwen OAuth':<12}  {check_mark(qwen_logged_in)} "
        f"{'logged in' if qwen_logged_in else 'not logged in (run: qwen auth qwen-oauth)'}"
    )
    qwen_auth_file = qwen_status.get("auth_file")
    if qwen_auth_file:
        print(f"    Auth file:  {qwen_auth_file}")
    qwen_exp = qwen_status.get("expires_at_ms")
    if qwen_exp:
        from datetime import datetime, timezone
        print(f"    Access exp: {datetime.fromtimestamp(int(qwen_exp) / 1000, tz=timezone.utc).isoformat()}")
    if qwen_status.get("error") and not qwen_logged_in:
        print(f"    Error:      {qwen_status.get('error')}")

    minimax_logged_in = bool(minimax_status.get("logged_in"))
    print(
        f"  {'MiniMax OAuth':<12}  {check_mark(minimax_logged_in)} "
        f"{'logged in' if minimax_logged_in else 'not logged in (run: hermes auth add minimax-oauth)'}"
    )
    minimax_region = minimax_status.get("region")
    if minimax_logged_in and minimax_region:
        print(f"    Region:     {minimax_region}")
    minimax_exp = minimax_status.get("expires_at")
    if minimax_exp:
        print(f"    Access exp: {minimax_exp}")
    if minimax_status.get("error") and not minimax_logged_in:
        print(f"    Error:      {minimax_status.get('error')}")

    # xAI OAuth — separate try/except so an import failure here cannot
    # disrupt the already-printed Nous/Codex/Qwen/MiniMax rows above.
    try:
        from hermes_cli.auth import get_xai_oauth_auth_status
        xai_oauth_status = get_xai_oauth_auth_status() or {}
    except Exception:
        xai_oauth_status = {}

    xai_oauth_logged_in = bool(xai_oauth_status.get("logged_in"))
    print(
        f"  {'xAI OAuth':<12}  {check_mark(xai_oauth_logged_in)} "
        f"{'logged in' if xai_oauth_logged_in else 'not logged in (run: hermes auth add xai-oauth)'}"
    )
    xai_auth_file = xai_oauth_status.get("auth_store")
    if xai_auth_file:
        print(f"    Auth file:  {xai_auth_file}")
    if xai_oauth_status.get("last_refresh"):
        print(f"    Refreshed:  {_format_iso_timestamp(xai_oauth_status.get('last_refresh'))}")
    if xai_oauth_status.get("error") and not xai_oauth_logged_in:
        print(f"    Error:      {xai_oauth_status.get('error')}")

    # =========================================================================
    # Nous Subscription Features
    # =========================================================================
    if managed_nous_tools_enabled():
        features = get_nous_subscription_features(config)
        print()
        print(color("◆ Nous Tool Gateway", Colors.CYAN, Colors.BOLD))
        if not features.nous_auth_present:
            print("  Nous Portal   ✗ not logged in")
        else:
            print("  Nous Portal   ✓ managed tools available")
        for feature in features.items():
            if feature.managed_by_nous:
                state = "active via Nous subscription"
            elif feature.active:
                current = feature.current_provider or "configured provider"
                state = f"active via {current}"
            elif feature.included_by_default and features.nous_auth_present:
                state = "included by subscription, not currently selected"
            elif feature.key == "modal" and features.nous_auth_present:
                state = "available via subscription (optional)"
            else:
                state = "not configured"
            print(f"  {feature.label:<15} {check_mark(feature.available or feature.active or feature.managed_by_nous)} {state}")
    elif nous_logged_in or nous_inference_present:
        # Nous OAuth without entitlement, or an opaque inference key without
        # Portal account information, cannot enable the Tool Gateway.
        print()
        print(color("◆ Nous Tool Gateway", Colors.CYAN, Colors.BOLD))
        message = format_nous_portal_entitlement_message(
            nous_account_info,
            capability="managed web, image, TTS, STT, browser, and Modal tools",
        )
        if message:
            for line in message.splitlines():
                print(f"  {line}")

    # =========================================================================
    # API-Key Providers
    # =========================================================================
    print()
    print(color("◆ API-Key Providers", Colors.CYAN, Colors.BOLD))

    apikey_providers = {
        "Z.AI / GLM":       ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
        "Kimi / Moonshot":  ("KIMI_API_KEY",),
        "StepFun Step Plan": ("STEPFUN_API_KEY",),
        "MiniMax":          ("MINIMAX_API_KEY",),
        "MiniMax (China)":  ("MINIMAX_CN_API_KEY",),
    }
    for pname, env_vars in apikey_providers.items():
        key_val = ""
        for ev in env_vars:
            key_val = get_env_value(ev) or ""
            if key_val:
                break
        configured = bool(key_val)
        label = "configured" if configured else "not configured (run: hermes model)"
        print(f"  {pname:<16} {check_mark(configured)} {label}")

    # LM Studio reachability — only probe when it's the active provider so
    # users with foreign configs don't see noise. Auth rejection vs. silent
    # empty list is the most common LM Studio support case.
    if _effective_provider_label() == "LM Studio":
        from hermes_cli.models import probe_lmstudio_models
        model_cfg = config.get("model")
        base = (model_cfg.get("base_url") if isinstance(model_cfg, dict) else None) or get_env_value("LM_BASE_URL") or "http://127.0.0.1:1234/v1"
        try:
            models = probe_lmstudio_models(api_key=get_env_value("LM_API_KEY") or "", base_url=base, timeout=1.5)
            if models is None:
                ok, msg = False, f"unreachable at {base}"
            else:
                ok, msg = True, f"reachable ({len(models)} model(s)) at {base}"
        except AuthError:
            ok, msg = False, "auth rejected — set LM_API_KEY"
        print(f"  {'LM Studio':<16} {check_mark(ok)} {msg}")

    # =========================================================================
    # Terminal Configuration
    # =========================================================================
    print()
    print(color("◆ Terminal Backend", Colors.CYAN, Colors.BOLD))

    terminal_cfg = config.get("terminal", {}) if isinstance(config.get("terminal"), dict) else {}
    terminal_env = os.getenv("TERMINAL_ENV", "")
    if not terminal_env:
        terminal_env = terminal_cfg.get("backend", "local")
    print(f"  Backend:      {terminal_env}")

    if terminal_env == "ssh":
        ssh_host = os.getenv("TERMINAL_SSH_HOST", "")
        ssh_user = os.getenv("TERMINAL_SSH_USER", "")
        print(f"  SSH Host:     {ssh_host or '(not set)'}")
        print(f"  SSH User:     {ssh_user or '(not set)'}")
    elif terminal_env == "docker":
        docker_image = os.getenv("TERMINAL_DOCKER_IMAGE", "python:3.11-slim")
        print(f"  Docker Image: {docker_image}")
    elif terminal_env == "daytona":
        daytona_image = os.getenv("TERMINAL_DAYTONA_IMAGE", "nikolaik/python-nodejs:python3.11-nodejs20")
        print(f"  Daytona Image: {daytona_image}")

    sudo_password = os.getenv("SUDO_PASSWORD", "")
    print(f"  Sudo:         {check_mark(bool(sudo_password))} {'enabled' if sudo_password else 'disabled'}")

    # =========================================================================
    # Messaging Platforms
    # =========================================================================
    print()
    print(color("◆ Messaging Platforms", Colors.CYAN, Colors.BOLD))

    platforms = {
        "Telegram": ("TELEGRAM_BOT_TOKEN", "TELEGRAM_HOME_CHANNEL"),
        "Discord": ("DISCORD_BOT_TOKEN", "DISCORD_HOME_CHANNEL"),
        "WhatsApp": ("WHATSAPP_ENABLED", None),
        "Signal": ("SIGNAL_HTTP_URL", "SIGNAL_HOME_CHANNEL"),
        "Slack": ("SLACK_BOT_TOKEN", None),
        "Email": ("EMAIL_ADDRESS", "EMAIL_HOME_ADDRESS"),
        "SMS": ("TWILIO_ACCOUNT_SID", "SMS_HOME_CHANNEL"),
        "DingTalk": ("DINGTALK_CLIENT_ID", None),
        "Feishu": ("FEISHU_APP_ID", "FEISHU_HOME_CHANNEL"),
        "WeCom": ("WECOM_BOT_ID", "WECOM_HOME_CHANNEL"),
        "WeCom Callback": ("WECOM_CALLBACK_CORP_ID", None),
        "Weixin": ("WEIXIN_ACCOUNT_ID", "WEIXIN_HOME_CHANNEL"),
        "BlueBubbles": ("BLUEBUBBLES_SERVER_URL", "BLUEBUBBLES_HOME_CHANNEL"),
        "QQBot": ("QQ_APP_ID", "QQ_HOME_CHANNEL"),
        "Yuanbao": ("YUANBAO_APP_ID", "YUANBAO_HOME_CHANNEL"),
    }

    for name, (token_var, home_var) in platforms.items():
        token = os.getenv(token_var, "")
        has_token = bool(token)
        
        home_channel = ""
        if home_var:
            home_channel = os.getenv(home_var, "")
        # Back-compat: QQBot home channel was renamed from QQ_HOME_CHANNEL to QQBOT_HOME_CHANNEL
        if not home_channel and home_var == "QQBOT_HOME_CHANNEL":
            home_channel = os.getenv("QQ_HOME_CHANNEL", "")
        
        status = "configured" if has_token else "not configured"
        if home_channel:
            status += f" (home: {home_channel})"
        
        print(f"  {name:<12}  {check_mark(has_token)} {status}")

    # Plugin-registered platforms
    try:
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.plugin_entries():
            configured = entry.check_fn()
            status_str = "configured" if configured else "not configured"
            label = entry.label
            print(f"  {label:<12}  {check_mark(configured)} {status_str} (plugin)")
    except Exception:
        pass

    # =========================================================================
    # Gateway Status
    # =========================================================================
    print()
    print(color("◆ Gateway Service", Colors.CYAN, Colors.BOLD))

    try:
        from hermes_cli.gateway import get_gateway_runtime_snapshot, _format_gateway_pids

        snapshot = get_gateway_runtime_snapshot()
        is_running = snapshot.running
        print(f"  Status:       {check_mark(is_running)} {'running' if is_running else 'stopped'}")
        print(f"  Manager:      {snapshot.manager}")
        if snapshot.gateway_pids:
            print(f"  PID(s):       {_format_gateway_pids(snapshot.gateway_pids)}")
        if snapshot.has_process_service_mismatch:
            print("  Service:      installed but not managing the current running gateway")
        elif _is_termux() and not snapshot.gateway_pids:
            print("  Start with:   hermes gateway")
            print("  Note:         Android may stop background jobs when Termux is suspended")
        elif snapshot.service_installed and not snapshot.service_running:
            print("  Service:      installed but stopped")
    except Exception:
        if _is_termux():
            print(f"  Status:       {color('unknown', Colors.DIM)}")
            print("  Manager:      Termux / manual process")
        elif sys.platform.startswith('linux'):
            print(f"  Status:       {color('unknown', Colors.DIM)}")
            print("  Manager:      systemd/manual")
        elif sys.platform == 'darwin':
            print(f"  Status:       {color('unknown', Colors.DIM)}")
            print("  Manager:      launchd")
        else:
            print(f"  Status:       {color('N/A', Colors.DIM)}")
            print("  Manager:      (not supported on this platform)")

    # =========================================================================
    # Cron Jobs
    # =========================================================================
    print()
    print(color("◆ Scheduled Jobs", Colors.CYAN, Colors.BOLD))

    jobs_file = get_hermes_home() / "cron" / "jobs.json"
    if jobs_file.exists():
        import json
        try:
            with open(jobs_file, encoding="utf-8") as f:
                data = json.load(f)
                jobs = data.get("jobs", [])
                enabled_jobs = [j for j in jobs if j.get("enabled", True)]
                print(f"  Jobs:         {len(enabled_jobs)} active, {len(jobs)} total")
        except Exception:
            print("  Jobs:         (error reading jobs file)")
    else:
        print("  Jobs:         0")

    # =========================================================================
    # Sessions
    # =========================================================================
    print()
    print(color("◆ Sessions", Colors.CYAN, Colors.BOLD))

    sessions_file = get_hermes_home() / "sessions" / "sessions.json"
    if sessions_file.exists():
        import json
        try:
            with open(sessions_file, encoding="utf-8") as f:
                data = json.load(f)
                print(f"  Active:       {len(data)} session(s)")
        except Exception:
            print("  Active:       (error reading sessions file)")
    else:
        print("  Active:       0")

    # =========================================================================
    # Deep checks
    # =========================================================================
    if deep:
        print()
        print(color("◆ Deep Checks", Colors.CYAN, Colors.BOLD))
        
        # Check OpenRouter connectivity
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        if openrouter_key:
            try:
                import httpx
                response = httpx.get(
                    OPENROUTER_MODELS_URL,
                    headers={"Authorization": f"Bearer {openrouter_key}"},
                    timeout=10
                )
                ok = response.status_code == 200
                print(f"  OpenRouter:   {check_mark(ok)} {'reachable' if ok else f'error ({response.status_code})'}")
            except Exception as e:
                print(f"  OpenRouter:   {check_mark(False)} error: {e}")
        
        # Check gateway port
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', 18789))
            sock.close()
            # Port in use = gateway likely running
            port_in_use = result == 0
            # This is informational, not necessarily bad
            print(f"  Port 18789:   {'in use' if port_in_use else 'available'}")
        except OSError:
            pass

    print()
    print(color("─" * 60, Colors.DIM))
    print(color("  Run 'hermes doctor' for detailed diagnostics", Colors.DIM))
    print(color("  Run 'hermes setup' to configure", Colors.DIM))
    print()
