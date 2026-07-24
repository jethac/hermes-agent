"""Agent profile + ContextVar plumbing for single-gateway-multi-agent.

An ``AgentProfile`` bundles every piece of per-agent state that used to be
process-scoped via ``HERMES_HOME``:

* home directory (governs SOUL.md, memory dir, skills dir, sessions.json)
* model / provider / api_key_env
* enabled / disabled toolsets
* free-form config overrides

The active profile is propagated through async code via a ``ContextVar``.
Path getters (``get_hermes_home``, ``get_skills_dir``, ``get_memory_dir``,
SOUL.md reader, etc.) honor the ContextVar **first**, falling back to the
``HERMES_HOME`` env var when no profile is set — so single-profile installs
see zero behavior change.

This module is import-safe: it has no module-level side effects and depends
only on stdlib + ``hermes_constants``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


DEFAULT_AGENT_ID = "main"


@dataclass
class AgentProfile:
    """A named agent with its own filesystem root, model, and toolset config.

    ``id`` is the routing key (matches ``SessionSource.agent_id``).
    ``home_dir`` is the root that replaces ``HERMES_HOME`` when this profile
    is active in the current ContextVar.  When ``home_dir`` is None the
    profile inherits the process-wide ``HERMES_HOME``, which is the right
    default for the legacy "main" agent.
    """

    id: str = DEFAULT_AGENT_ID
    home_dir: Optional[Path] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    enabled_toolsets: Optional[List[str]] = None
    disabled_toolsets: Optional[List[str]] = None
    receptors: Optional[List[str]] = None
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    def allows_mcp(self, server_name: str, tool_name: str = "") -> bool:
        """Receptor check: may this profile see/call *tool_name* on *server_name*?

        ``receptors`` grammar (per ``agents.<id>.receptors`` in config.yaml):

        * ``None`` (key absent)  → **fail-closed**: NO MCP tools at all.
        * ``[]``                 → same as absent: no MCP tools.
        * ``["*"]``              → all servers, all tools.
        * ``["gbrain"]``         → every tool on server ``gbrain``.
        * ``["sensei.scan_*"]``  → fnmatch glob on ``server.tool``; the part
          before the FIRST dot is the server pattern, the rest the tool
          pattern (MCP tool names conventionally contain no dots).

        *server_name* must be the RAW config key from ``mcp_servers:`` (not
        the sanitized ``mcp__``-prefix component). An empty *tool_name* asks
        the server-level question "is any tool on this server allowed?" and
        matches ``server.tool`` patterns on the server part alone.
        """
        receptors = self.receptors
        if not receptors:
            return False  # fail-closed: no receptors key → no MCP surface
        from fnmatch import fnmatchcase
        for raw in receptors:
            pat = str(raw).strip()
            if not pat:
                continue
            if pat == "*":
                return True
            if "." in pat:
                server_pat, tool_pat = pat.split(".", 1)
                if not fnmatchcase(server_name, server_pat):
                    continue
                if not tool_name or fnmatchcase(tool_name, tool_pat):
                    return True
            elif fnmatchcase(server_name, pat):
                return True
        return False

    @property
    def resolved_home(self) -> Path:
        """Return the effective home directory for this profile.

        Falls back to the process-wide ``HERMES_HOME`` when ``home_dir``
        is unset.  This keeps the default profile a thin shell over the
        existing env-driven layout.
        """
        if self.home_dir is not None:
            return Path(self.home_dir).expanduser()
        return get_hermes_home()

    @property
    def soul_md_path(self) -> Path:
        return self.resolved_home / "SOUL.md"

    @property
    def memory_dir(self) -> Path:
        return self.resolved_home / "memories"

    @property
    def skills_dir(self) -> Path:
        return self.resolved_home / "skills"

    @property
    def sessions_path(self) -> Path:
        return self.resolved_home / "sessions.json"


_current_agent_profile: ContextVar[Optional[AgentProfile]] = ContextVar(
    "hermes_current_agent_profile", default=None
)


def get_active_profile() -> Optional[AgentProfile]:
    """Return the profile bound to the current async context, or None.

    None means "no profile bound" — callers should fall back to
    ``HERMES_HOME`` env-var behavior, which is exactly what the legacy
    single-profile install expects.
    """
    return _current_agent_profile.get()


def set_active_profile(profile: Optional[AgentProfile]):
    """Bind *profile* to the current ContextVar; returns the reset token.

    Prefer ``use_profile()`` (context manager) over this raw setter.
    """
    return _current_agent_profile.set(profile)


@contextmanager
def use_profile(profile: Optional[AgentProfile]) -> Iterator[Optional[AgentProfile]]:
    """Bind *profile* for the duration of the ``with`` block.

    Uses ContextVar.set/reset so the binding propagates through ``await``
    and ``asyncio.gather``, but does **not** leak to sibling tasks unless
    they are spawned with ``copy_context()`` from within the block.

    Passing ``None`` is a no-op restoration: callers that hit a route
    without a profile (legacy single-agent path) can simply skip the
    ``with``.
    """
    if profile is None:
        yield None
        return
    token = _current_agent_profile.set(profile)
    try:
        yield profile
    finally:
        _current_agent_profile.reset(token)


def load_agent_registry(config: Any) -> Dict[str, AgentProfile]:
    """Build an ``id -> AgentProfile`` registry from a ``GatewayConfig``.

    Reads ``config.agents`` (a dict of id -> kwargs) and constructs an
    ``AgentProfile`` for each.  Always returns at least the default
    profile under ``DEFAULT_AGENT_ID`` so single-agent installs work
    without any config changes.

    Unknown kwargs in the agent dicts are forwarded to ``config_overrides``
    so future per-agent settings don't require a registry update.
    """
    registry: Dict[str, AgentProfile] = {}

    raw_agents: Dict[str, Any] = {}
    if config is not None:
        raw_agents = getattr(config, "agents", None) or {}

    for agent_id, raw in (raw_agents or {}).items():
        if not isinstance(raw, dict):
            logger.warning(
                "agents.%s ignored: expected dict, got %s",
                agent_id, type(raw).__name__,
            )
            continue
        profile = _build_profile(agent_id, raw)
        registry[agent_id] = profile

    if DEFAULT_AGENT_ID not in registry:
        registry[DEFAULT_AGENT_ID] = AgentProfile(id=DEFAULT_AGENT_ID)

    return registry


def _build_profile(agent_id: str, raw: Dict[str, Any]) -> AgentProfile:
    """Construct an AgentProfile from a raw config dict.

    Known keys are extracted; everything else lands in ``config_overrides``
    so downstream code can pick up custom keys without touching this file.
    """
    raw = dict(raw)  # Copy so pops don't mutate the caller's dict.
    home_dir_raw = raw.pop("home_dir", None)
    home_dir = Path(home_dir_raw).expanduser() if home_dir_raw else None

    model = raw.pop("model", None)
    provider = raw.pop("provider", None)
    base_url = raw.pop("base_url", None)
    api_key_env = raw.pop("api_key_env", None)
    enabled_toolsets = raw.pop("enabled_toolsets", None)
    disabled_toolsets = raw.pop("disabled_toolsets", None)

    receptors_raw = raw.pop("receptors", None)
    receptors: Optional[List[str]]
    if receptors_raw is None:
        receptors = None
    elif isinstance(receptors_raw, str):
        receptors = [receptors_raw]
    elif isinstance(receptors_raw, (list, tuple)):
        receptors = [str(r) for r in receptors_raw]
    else:
        logger.warning(
            "agents.%s.receptors ignored: expected list, got %s (fail-closed)",
            agent_id, type(receptors_raw).__name__,
        )
        receptors = []  # malformed → fail-closed, not fail-open

    return AgentProfile(
        id=agent_id,
        home_dir=home_dir,
        model=model,
        provider=provider,
        base_url=base_url,
        api_key_env=api_key_env,
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
        receptors=receptors,
        config_overrides=raw,
    )
