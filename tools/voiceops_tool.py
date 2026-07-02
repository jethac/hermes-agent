"""VoiceOps planning tools.

These tools prepare bounded, non-mutating VoiceOps action artifacts. They do
not execute spend, provisioning, credential retrieval, outbound messages, or
phone calls.
"""

from __future__ import annotations

import json
from typing import Any

from tools.registry import registry, tool_error


MAX_BUDGET_CENTS = 1_000_000


def check_voiceops_requirements() -> bool:
    return True


def _int_arg(args: dict[str, Any], key: str, *, default: int) -> int:
    raw = args.get(key, default)
    if isinstance(raw, bool):
        raise ValueError(f"{key} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{key} must be non-negative")
    if value > MAX_BUDGET_CENTS:
        raise ValueError(f"{key} must be <= {MAX_BUDGET_CENTS}")
    return value


def _prepare_voiceops_action_packet(**kwargs: Any) -> dict[str, Any]:
    from scripts.hackathon_voiceops_demo import prepare_voiceops_action_packet

    return prepare_voiceops_action_packet(**kwargs)


def voiceops_prepare_action_packet_tool(args: dict[str, Any], **_kw: Any) -> str:
    """Prepare a redacted VoiceOps NemoClaw packet and execution plan."""
    try:
        request = str(args.get("request") or "").strip()
        if not request:
            return tool_error("request is required")
        budget_cents = _int_arg(args, "budget_cents", default=20_000)
        approval_required_over_cents = _int_arg(args, "approval_required_over_cents", default=1_000)
        active_model = str(args.get("active_model") or "").strip()
        reflex_model = str(args.get("reflex_model") or "").strip()
        payload = _prepare_voiceops_action_packet(
            request=request,
            budget_cents=budget_cents,
            approval_required_over_cents=approval_required_over_cents,
            active_model=active_model or "Hermes active model selected through /model",
            reflex_model=reflex_model or "KAME reflex model selected by realtime voice config",
            env={},
            env_files=(),
            which=lambda _command: None,
        )
    except Exception as exc:
        return tool_error(str(exc))
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


VOICEOPS_PREPARE_ACTION_PACKET_SCHEMA = {
    "name": "voiceops_prepare_action_packet",
    "description": (
        "Prepare a non-mutating VoiceOps action packet for a spoken household/"
        "business operations request. Returns a NemoClaw-safe approval packet, "
        "Milestone 2 execution plan, phone handoff context, and validation. "
        "This tool never spends money, provisions services, reads credentials, "
        "sends messages, places calls, or performs network I/O; it only prepares "
        "approval-gated artifacts for the real Hermes oracle session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "The user's spoken VoiceOps request to prepare safely.",
            },
            "budget_cents": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_BUDGET_CENTS,
                "description": "Total user-granted budget in cents. Default is 20000.",
            },
            "approval_required_over_cents": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_BUDGET_CENTS,
                "description": "Spend threshold above which approval is required. Default is 1000.",
            },
            "active_model": {
                "type": "string",
                "description": "Optional label for Hermes' active /model selection; not a separate oracle model.",
            },
            "reflex_model": {
                "type": "string",
                "description": "Optional label for the KAME reflex/interface model.",
            },
        },
        "required": ["request"],
    },
}


registry.register(
    name="voiceops_prepare_action_packet",
    toolset="voiceops",
    schema=VOICEOPS_PREPARE_ACTION_PACKET_SCHEMA,
    handler=voiceops_prepare_action_packet_tool,
    check_fn=check_voiceops_requirements,
    emoji="",
    max_result_size_chars=120_000,
)
