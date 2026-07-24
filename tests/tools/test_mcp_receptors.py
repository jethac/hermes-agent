"""Receptor (per-agent MCP tool scoping) tests.

Covers the three contract points of ``agents.<id>.receptors``:

(a) a profile with NO receptors key sees ZERO MCP tools (fail-closed),
(b) a profile with receptors sees exactly the intersection of the global
    MCP surface and its receptor patterns,
(c) ``["*"]`` sees everything,

plus the call-time enforcement guard and the no-profile legacy path.
"""

import json

import pytest

import model_tools
from agent.profile import AgentProfile, use_profile, _build_profile
from tools import mcp_tool
from tools.mcp_tool import (
    _forget_mcp_tool_server,
    _track_mcp_tool_server,
    _wrap_with_receptor_guard,
    mcp_prefixed_tool_name,
)
from tools.registry import registry


FAKE_SERVERS = {
    # server -> tools
    "gbrain": ["search_notes", "write_note"],
    "stripe-link": ["create_payment"],
    "sensei": ["scan_inbox", "begin_diagnostic"],
}


def _schema(name):
    return {
        "name": name,
        "description": f"fake {name}",
        "parameters": {"type": "object", "properties": {}},
    }


@pytest.fixture()
def fake_mcp_registry():
    """Register a fake multi-server MCP surface into the live registry."""
    registered = []
    for server, tools in FAKE_SERVERS.items():
        toolset = f"mcp-{server}"
        for tool in tools:
            prefixed = mcp_prefixed_tool_name(server, tool)
            registry.register(
                name=prefixed,
                toolset=toolset,
                schema=_schema(prefixed),
                handler=_wrap_with_receptor_guard(
                    server, tool, lambda args, **kw: "ok"
                ),
                check_fn=None,
                is_async=False,
                description=f"fake {tool}",
            )
            _track_mcp_tool_server(prefixed, server, tool)
            registered.append(prefixed)
    model_tools._clear_tool_defs_cache()
    try:
        yield registered
    finally:
        for name in registered:
            try:
                registry.deregister(name)
            except Exception:
                pass
            _forget_mcp_tool_server(name)
        model_tools._clear_tool_defs_cache()


def _mcp_names(defs):
    return {
        t["function"]["name"]
        for t in defs
        if t["function"]["name"].startswith(mcp_tool.MCP_TOOL_NAME_PREFIX)
        and mcp_tool.get_mcp_tool_provenance(t["function"]["name"]) is not None
    }


ALL_FAKE = {
    mcp_prefixed_tool_name(s, t) for s, tools in FAKE_SERVERS.items() for t in tools
}


def _defs():
    return model_tools.get_tool_definitions(
        enabled_toolsets=[f"mcp-{s}" for s in FAKE_SERVERS],
        quiet_mode=True,
    )


class TestReceptorVisibility:
    def test_no_receptors_key_sees_zero_mcp_tools(self, fake_mcp_registry):
        profile = AgentProfile(id="locked")  # receptors=None → fail-closed
        with use_profile(profile):
            assert _mcp_names(_defs()) & ALL_FAKE == set()

    def test_empty_receptors_sees_zero_mcp_tools(self, fake_mcp_registry):
        profile = AgentProfile(id="locked", receptors=[])
        with use_profile(profile):
            assert _mcp_names(_defs()) & ALL_FAKE == set()

    def test_receptors_see_exact_intersection(self, fake_mcp_registry):
        profile = AgentProfile(
            id="scoped", receptors=["gbrain", "sensei.scan_*"]
        )
        with use_profile(profile):
            names = _mcp_names(_defs()) & ALL_FAKE
        assert names == {
            mcp_prefixed_tool_name("gbrain", "search_notes"),
            mcp_prefixed_tool_name("gbrain", "write_note"),
            mcp_prefixed_tool_name("sensei", "scan_inbox"),
        }

    def test_star_sees_all(self, fake_mcp_registry):
        profile = AgentProfile(id="root", receptors=["*"])
        with use_profile(profile):
            names = _mcp_names(_defs()) & ALL_FAKE
        assert names == ALL_FAKE

    def test_no_profile_bound_is_legacy_unfiltered(self, fake_mcp_registry):
        # Legacy single-agent path: no ContextVar profile → no filtering.
        assert _mcp_names(_defs()) & ALL_FAKE == ALL_FAKE

    def test_cache_hit_is_refiltered_per_profile(self, fake_mcp_registry):
        # Same cache key, different profiles: each must see its own subset.
        with use_profile(AgentProfile(id="root", receptors=["*"])):
            assert _mcp_names(_defs()) & ALL_FAKE == ALL_FAKE
        with use_profile(AgentProfile(id="locked")):
            assert _mcp_names(_defs()) & ALL_FAKE == set()
        with use_profile(AgentProfile(id="money", receptors=["stripe-link"])):
            assert _mcp_names(_defs()) & ALL_FAKE == {
                mcp_prefixed_tool_name("stripe-link", "create_payment")
            }


class TestReceptorCallGuard:
    def test_denied_call_is_blocked(self):
        handler = _wrap_with_receptor_guard(
            "stripe-link", "create_payment", lambda args, **kw: "charged"
        )
        with use_profile(AgentProfile(id="agent", receptors=["gbrain"])):
            out = json.loads(handler({}))
        assert "not available" in out["error"]

    def test_allowed_call_passes_through(self):
        handler = _wrap_with_receptor_guard(
            "gbrain", "search_notes", lambda args, **kw: "found"
        )
        with use_profile(AgentProfile(id="ea", receptors=["gbrain"])):
            assert handler({}) == "found"

    def test_no_profile_passes_through(self):
        handler = _wrap_with_receptor_guard(
            "gbrain", "search_notes", lambda args, **kw: "found"
        )
        assert handler({}) == "found"


class TestAllowsMcpGrammar:
    def test_server_glob(self):
        p = AgentProfile(id="x", receptors=["gb*"])
        assert p.allows_mcp("gbrain", "anything")
        assert not p.allows_mcp("sensei", "anything")

    def test_server_dot_tool_glob(self):
        p = AgentProfile(id="x", receptors=["sensei.scan_*"])
        assert p.allows_mcp("sensei", "scan_inbox")
        assert not p.allows_mcp("sensei", "begin_diagnostic")
        # Server-level question (empty tool) still acknowledges the server.
        assert p.allows_mcp("sensei")

    def test_build_profile_parses_receptors(self):
        p = _build_profile("ea", {"receptors": ["gbrain", "sensei"]})
        assert p.receptors == ["gbrain", "sensei"]
        assert "receptors" not in p.config_overrides

    def test_build_profile_malformed_receptors_fail_closed(self):
        p = _build_profile("ea", {"receptors": 42})
        assert p.receptors == []
        assert not p.allows_mcp("gbrain", "search_notes")

    def test_build_profile_no_receptors_is_none(self):
        p = _build_profile("ea", {})
        assert p.receptors is None
        assert not p.allows_mcp("gbrain", "search_notes")
