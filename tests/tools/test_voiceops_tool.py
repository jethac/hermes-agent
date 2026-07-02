import json


def test_voiceops_prepare_action_packet_tool_registered_and_returns_safe_packet():
    from model_tools import (
        get_all_tool_names,
        get_tool_definitions,
        get_toolset_for_tool,
        handle_function_call,
    )

    assert "voiceops_prepare_action_packet" in get_all_tool_names()
    assert get_toolset_for_tool("voiceops_prepare_action_packet") == "voiceops"
    schemas = {
        tool["function"]["name"]: tool["function"]
        for tool in get_tool_definitions(enabled_toolsets=["voiceops"])
    }
    assert schemas["voiceops_prepare_action_packet"]["parameters"]["required"] == ["request"]

    payload = json.loads(
        handle_function_call(
            "voiceops_prepare_action_packet",
            {
                "request": (
                    "Hermes, I am giving you 200 dollars to use through Stripe Skills. "
                    "Provision yourself a VoIP provider account, then call my phone with "
                    "this same context so we can continue outside Discord."
                ),
                "budget_cents": 20_000,
                "active_model": "Nemotron 3 Super via Hermes /model",
            },
        )
    )

    assert payload["schema_version"] == "voiceops.action_packet_preparation.v1"
    assert payload["safety"] == {
        "executes_commands": False,
        "network_io": False,
        "live_spend": False,
        "provider_provisioning": False,
        "credential_retrieval": False,
        "outbound_phone_calls": False,
        "secret_values_emitted": False,
        "requires_operator_approval": True,
    }
    assert payload["nemoclaw_action_packet_validation"]["status"] == "valid"
    action_ids = {
        action["action_id"]
        for action in payload["nemoclaw_action_packet"]["approval_required_actions"]
    }
    assert {"provision-voip-provider", "buy-service-credit", "call-user-phone"} <= action_ids
    assert payload["milestone2_execution_plan"]["spend_policy"]["budget_cap_cents"] == 20_000
    assert payload["nemoclaw_action_packet"]["model_selected_by"] == "Hermes /model"
    assert "oracle_model" not in payload["nemoclaw_action_packet"]


def test_voiceops_toolset_is_configurable_and_default_off():
    from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS, _get_platform_tools
    from toolsets import TOOLSETS, resolve_toolset

    configurable_keys = {key for key, _label, _description in CONFIGURABLE_TOOLSETS}

    assert "voiceops" in TOOLSETS
    assert "voiceops" in configurable_keys
    assert resolve_toolset("voiceops") == ["voiceops_prepare_action_packet"]
    assert "voiceops" not in _get_platform_tools({}, "cli")
    assert "voiceops" in _get_platform_tools({"platform_toolsets": {"cli": ["voiceops"]}}, "cli")


def test_voiceops_prepare_action_packet_tool_uses_non_probing_boundary(monkeypatch):
    from model_tools import handle_function_call
    import tools.voiceops_tool as voiceops_tool

    calls = {}

    def fake_prepare(**kwargs):
        calls.update(kwargs)
        assert kwargs["which"]("stripe") is None
        return {
            "schema_version": "voiceops.action_packet_preparation.v1",
            "safety": {"network_io": False},
        }

    monkeypatch.setattr(voiceops_tool, "_prepare_voiceops_action_packet", fake_prepare)

    payload = json.loads(
        handle_function_call(
            "voiceops_prepare_action_packet",
            {
                "request": "prepare a VoiceOps packet",
                "budget_cents": 1234,
            },
        )
    )

    assert payload["schema_version"] == "voiceops.action_packet_preparation.v1"
    assert calls["env"] == {}
    assert calls["env_files"] == ()
    assert calls["budget_cents"] == 1234


def test_voiceops_prepare_action_packet_tool_requires_request():
    from model_tools import handle_function_call

    result = json.loads(handle_function_call("voiceops_prepare_action_packet", {}))

    assert "error" in result
    assert result["error"] == "request is required"


def test_voiceops_prepare_action_packet_tool_rejects_invalid_budget():
    from model_tools import handle_function_call

    result = json.loads(
        handle_function_call(
            "voiceops_prepare_action_packet",
            {
                "request": "prepare a VoiceOps packet",
                "budget_cents": -1,
            },
        )
    )

    assert "error" in result
    assert "budget_cents must be non-negative" in result["error"]
