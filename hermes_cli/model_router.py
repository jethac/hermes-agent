"""Lightweight request routing helpers for multi-model Hermes setups."""

from __future__ import annotations

from typing import Any


DEFAULT_HEAVY_KEYWORDS = (
    "architecture",
    "audit",
    "build",
    "debug",
    "deep",
    "design",
    "diagnose",
    "evaluate",
    "implement",
    "investigate",
    "plan",
    "prd",
    "production",
    "provision",
    "refactor",
    "research",
    "security",
    "strategy",
    "thorough",
)


def normalize_model_router_config(raw: Any) -> dict[str, Any]:
    """Return a tolerant model-router config with stable defaults."""

    if not isinstance(raw, dict):
        raw = {}
    mode = str(raw.get("mode") or "heavy_moa").strip().lower().replace("-", "_")
    if mode not in {"off", "heavy_moa"}:
        mode = "heavy_moa"
    keywords = raw.get("heavy_keywords")
    if not isinstance(keywords, list):
        keywords = list(DEFAULT_HEAVY_KEYWORDS)
    cleaned_keywords = [str(item).strip().lower() for item in keywords if str(item or "").strip()]
    exclude_prefixes = raw.get("exclude_prefixes")
    if not isinstance(exclude_prefixes, list):
        exclude_prefixes = ["/"]
    cleaned_exclude_prefixes = [
        str(item).strip()
        for item in exclude_prefixes
        if str(item or "").strip()
    ]
    try:
        min_chars = int(raw.get("heavy_min_chars", 900))
    except (TypeError, ValueError):
        min_chars = 900
    return {
        "enabled": bool(raw.get("enabled", False)),
        "mode": mode,
        "heavy_moa_preset": str(raw.get("heavy_moa_preset") or "gemma-nemotron").strip() or "gemma-nemotron",
        "heavy_min_chars": max(0, min_chars),
        "heavy_keywords": cleaned_keywords,
        "exclude_prefixes": cleaned_exclude_prefixes,
    }


def is_heavy_request(user_message: Any, router_config: Any) -> bool:
    """Heuristic gate for turns that should get multi-model advice."""

    cfg = normalize_model_router_config(router_config)
    if not cfg["enabled"] or cfg["mode"] == "off":
        return False
    text = str(user_message or "").strip()
    if not text:
        return False
    if any(text.startswith(prefix) for prefix in cfg["exclude_prefixes"]):
        return False
    if len(text) >= int(cfg["heavy_min_chars"]):
        return True
    lowered = text.lower()
    return any(keyword in lowered for keyword in cfg["heavy_keywords"])


def moa_config_for_request(config: dict[str, Any] | None, user_message: Any) -> dict[str, Any] | None:
    """Return the configured heavy-request MoA preset for this turn, if any."""

    config = config or {}
    router = normalize_model_router_config(config.get("model_router"))
    if not is_heavy_request(user_message, router):
        return None

    from hermes_cli.moa_config import resolve_moa_preset

    return resolve_moa_preset(config.get("moa") or {}, router["heavy_moa_preset"])
