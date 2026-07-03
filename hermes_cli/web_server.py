"""
Hermes Agent — Web UI server.

Provides a FastAPI backend serving the Vite/React frontend and REST API
endpoints for managing configuration, environment variables, and sessions.

Usage:
    python -m hermes_cli.main web          # Start on http://127.0.0.1:9119
    python -m hermes_cli.main web --port 8080
"""

from contextlib import asynccontextmanager, contextmanager, suppress

import asyncio
import base64
import binascii
from dataclasses import dataclass
from datetime import datetime, timezone
import hmac
import importlib.util
import json
import logging
import mimetypes
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import zipfile

from hermes_cli._subprocess_compat import windows_detach_flags, windows_hide_flags
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hermes_cli import __version__, __release_date__
from hermes_cli.config import (
    cfg_get,
    DEFAULT_CONFIG,
    OPTIONAL_ENV_VARS,
    clear_model_endpoint_credentials,
    get_config_path,
    get_env_path,
    get_hermes_home,
    load_config,
    load_env,
    read_raw_config,
    save_config,
    save_env_value,
    remove_env_value,
    check_config_version,
    detect_install_method,
    format_docker_update_message,
    recommended_update_command_for_method,
    redact_key,
    write_platform_config_field,
    _deep_merge,
)
from hermes_cli.memory_providers import (
    MemoryProvider,
    ProviderField,
    get_memory_provider,
)
from gateway.status import (
    derive_gateway_busy,
    derive_gateway_drainable,
    get_running_pid,
    get_runtime_status_running_pid,
    parse_active_agents,
    read_runtime_status,
)
from hermes_cli.realtime_voice_production_review import (
    REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS,
    load_production_review_report,
    validate_production_review_report,
)
from agent.realtime_voice_errors import sanitize_realtime_voice_error
from utils import env_var_enabled

try:
    from fastapi import (
        FastAPI, File, Form, HTTPException, Request, UploadFile,
        WebSocket, WebSocketDisconnect,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError:
    # First try lazy-installing the dashboard extras. Only the user actually
    # running `hermes dashboard` needs fastapi+uvicorn; lazy install keeps
    # them out of every other install path. After install, re-import.
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("tool.dashboard", prompt=False)
        from fastapi import (
            FastAPI, File, Form, HTTPException, Request, UploadFile,
            WebSocket, WebSocketDisconnect,
        )
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel
    except Exception:
        raise SystemExit(
            "Web UI requires fastapi and uvicorn.\n"
            f"Install with: {sys.executable} -m pip install 'fastapi' 'uvicorn[standard]'"
        )

WEB_DIST = Path(os.environ["HERMES_WEB_DIST"]) if "HERMES_WEB_DIST" in os.environ else Path(__file__).parent / "web_dist"
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-channel subscriber registry used by /api/pub (PTY-side gateway → dashboard)
# and /api/events (dashboard → browser sidebar).  Keyed by an opaque channel id
# the chat tab generates on mount; entries auto-evict when the last subscriber
# drops AND the publisher has disconnected.
#
# State lives on app.state (not module-level globals) so that asyncio.Lock is
# created on the running event loop during lifespan startup.  A module-level
# asyncio.Lock() binds to whatever loop was active at import time, which breaks
# when the same module is used across TestClient instances or uvicorn reloads.
# ---------------------------------------------------------------------------

def _start_desktop_cron_ticker(stop_event: "threading.Event", interval: int = 60) -> None:
    """Tick the cron scheduler from inside the desktop dashboard backend.

    The scheduler tick loop normally lives in ``hermes gateway run`` — but the
    desktop app spawns a ``hermes dashboard`` backend, not a gateway, so a cron
    a user creates in the app would never fire. We run the resolved cron
    scheduler provider here (no live adapters; delivery falls back to the
    per-platform send path).

    Cross-process safe: the built-in provider's ``cron.scheduler.tick`` takes
    the ``cron/.tick.lock`` file lock, so this never double-fires alongside a
    real gateway on the same HERMES_HOME — whichever process grabs the lock
    first wins the tick.
    """
    from cron.scheduler_provider import resolve_cron_scheduler

    provider = resolve_cron_scheduler()
    _log.info("Desktop cron scheduler started (provider=%s, interval=%ds)", provider.name, interval)
    provider.start(stop_event, interval=interval)


def _warm_gateway_module() -> None:
    try:
        import hermes_cli.gateway  # noqa: F401
    except Exception:
        pass


def _resolve_restart_drain_timeout() -> float:
    try:
        from hermes_cli.gateway import _get_restart_drain_timeout
        return _get_restart_drain_timeout()
    except ImportError:
        from gateway.restart import DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
        return DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT


@asynccontextmanager
async def _lifespan(app: "FastAPI"):
    app.state.event_channels = {}  # dict[str, set]
    app.state.event_lock = asyncio.Lock()
    app.state.pty_active_session_files = {}  # dict[str, Path]
    # Serializes chat-argv resolution so concurrent /api/pty connections
    # don't trigger overlapping ``npm install`` / ``npm run build`` work.
    # On app.state (not a module global) so the Lock binds to the running
    # event loop during lifespan startup — see _get_event_state's docstring.
    app.state.chat_argv_lock = asyncio.Lock()

    # Fire hermes_cli.gateway import into a background thread so the event
    # loop is not blocked and HERMES_DASHBOARD_READY fires without delay.
    # On a cold Windows install the module chain triggers .pyc compilation
    # and Defender real-time scans that can stall the event loop for 15-30s.
    # Running in an executor means the cost is paid in a worker thread while
    # the server socket is already open and accepting probes.
    asyncio.get_event_loop().run_in_executor(None, _warm_gateway_module)

    # Desktop-spawned backends (HERMES_DESKTOP=1) fire cron jobs themselves,
    # since the app has no gateway running the scheduler. Server `hermes
    # dashboard` is unaffected — it relies on its own gateway.
    cron_stop: "threading.Event | None" = None
    cron_thread: "threading.Thread | None" = None
    if os.getenv("HERMES_DESKTOP") == "1":
        cron_stop = threading.Event()
        cron_thread = threading.Thread(
            target=_start_desktop_cron_ticker,
            args=(cron_stop,),
            daemon=True,
            name="desktop-cron-ticker",
        )
        cron_thread.start()

    try:
        yield
    finally:
        if cron_stop is not None:
            cron_stop.set()


def _get_event_state(app: "FastAPI"):
    """Return (event_channels, event_lock) from app.state.

    Lazily initialises the state if the lifespan hasn't run (e.g. when
    TestClient is constructed without a ``with`` block).  The lifespan
    path is preferred because it guarantees the Lock is created on the
    correct event loop, but the lazy path lets existing non-``with``
    TestClient usages keep working.
    """
    try:
        return app.state.event_channels, app.state.event_lock
    except AttributeError:
        app.state.event_channels = {}
        app.state.event_lock = asyncio.Lock()
        return app.state.event_channels, app.state.event_lock


def _get_chat_argv_lock(app: "FastAPI") -> asyncio.Lock:
    """Return the chat-argv resolution lock from app.state.

    Mirrors :func:`_get_event_state`: prefers the lifespan-initialised Lock
    (created on the correct event loop) but lazily initialises it for
    non-``with`` TestClient usages.
    """
    try:
        return app.state.chat_argv_lock
    except AttributeError:
        app.state.chat_argv_lock = asyncio.Lock()
        return app.state.chat_argv_lock


def _get_pty_active_session_files(app: "FastAPI") -> dict[str, Path]:
    """Return channel -> active-session-file state for dashboard PTYs."""
    try:
        return app.state.pty_active_session_files
    except AttributeError:
        app.state.pty_active_session_files = {}
        return app.state.pty_active_session_files


app = FastAPI(title="Hermes Agent", version=__version__, lifespan=_lifespan)

# Memory-provider OAuth connect routes live in the memory layer, not here.
from hermes_cli.memory_oauth import router as _memory_oauth_router  # noqa: E402

app.include_router(_memory_oauth_router)

# ---------------------------------------------------------------------------
# Session token for protecting sensitive endpoints (reveal).
# The desktop shell mints the token and injects it via
# HERMES_DASHBOARD_SESSION_TOKEN so its main process can authenticate the
# /api calls it makes on the user's behalf; otherwise we generate one fresh
# on every server start. Either way it dies when the process exits and is
# injected into the SPA HTML so only the legitimate web UI can use it.
# ---------------------------------------------------------------------------
_SESSION_TOKEN = os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN") or secrets.token_urlsafe(32)
_SESSION_HEADER_NAME = "X-Hermes-Session-Token"

# In-browser Chat tab (/chat, /api/pty, /api/ws, …).  Always enabled: the
# desktop app and the dashboard's own Chat tab both drive the agent over the
# `/api/ws` + `/api/pty` WebSockets, so the embedded-chat surface is an
# unconditional part of the dashboard.  Kept as a module-level constant (rather
# than inlining ``True`` at every gate) so the WS endpoints and the SPA token
# injection share a single, testable seam.
_DASHBOARD_EMBEDDED_CHAT_ENABLED = True

# Simple rate limiter for the reveal endpoint
_reveal_timestamps: List[float] = []
_REVEAL_MAX_PER_WINDOW = 5
_REVEAL_WINDOW_SECONDS = 30

# CORS: restrict to localhost origins only.  The web UI is intended to run
# locally; binding to 0.0.0.0 with allow_origins=["*"] would let any website
# read/modify config and secrets.

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Endpoints that do NOT require the session token.  Everything else under
# /api/ is gated by the auth middleware below.
#
# This list is defined in ``hermes_cli.dashboard_auth.public_paths`` so the
# OAuth gate middleware can honour the same allowlist — keeping the two
# gates in lockstep avoids drift like the wildcard-subdomain regression
# where ``/api/status`` was public under the legacy gate but 401'd under
# the OAuth gate (breaking the portal's liveness probe).
#
# Keep the upstream list minimal — only truly non-sensitive, read-only
# endpoints belong there.
# ---------------------------------------------------------------------------
from hermes_cli.dashboard_auth.public_paths import (
    PUBLIC_API_PATHS as _PUBLIC_API_PATHS,
)


def _has_valid_session_token(request: Request) -> bool:
    """True if the request carries a valid dashboard session token.

    The dedicated session header avoids collisions with reverse proxies that
    already use ``Authorization`` (for example Caddy ``basic_auth``). We still
    accept the legacy Bearer path for backward compatibility with older
    dashboard bundles.
    """
    session_header = request.headers.get(_SESSION_HEADER_NAME, "")
    if session_header and hmac.compare_digest(
        session_header.encode(),
        _SESSION_TOKEN.encode(),
    ):
        return True

    auth = request.headers.get("authorization", "")
    expected = f"Bearer {_SESSION_TOKEN}"
    return hmac.compare_digest(auth.encode(), expected.encode())


# Routes that may also authenticate via a ``?token=`` query param, for download
# links opened by the OS shell or a new browser tab where the session header
# can't be set. Kept narrow — same query-token tradeoff as the /api/pty WS.
_QUERY_TOKEN_API_PATHS: frozenset[str] = frozenset({"/api/files/download"})


def _has_valid_query_token(request: Request, path: str) -> bool:
    if path not in _QUERY_TOKEN_API_PATHS:
        return False
    token = request.query_params.get("token", "")
    return bool(token) and hmac.compare_digest(token.encode(), _SESSION_TOKEN.encode())


def _require_token(request: Request) -> None:
    """Authorize a sensitive endpoint, raising 401 if the caller isn't allowed.

    Two auth schemes protect the dashboard, exactly one active per bind:

    * **Loopback / ``--insecure`` mode** (``auth_required`` False): the
      ephemeral ``_SESSION_TOKEN`` is injected into the SPA HTML and echoed
      back via ``X-Hermes-Session-Token`` (or the legacy ``Bearer`` header).
      Validate it here.
    * **Gated / OAuth mode** (``auth_required`` True): ``_SESSION_TOKEN`` is
      NOT injected (the SPA authenticates with a session cookie), so there is
      no token to check. The ``gated_auth_middleware`` has already verified the
      cookie before the request reached this handler — any non-public ``/api/``
      route it lets through carries a verified ``request.state.session``. The
      legacy ``auth_middleware`` likewise short-circuits in this mode. Requiring
      the (absent) token here would 401 every cookie-authenticated request,
      making plugin install/enable/disable and the other ``_require_token``
      endpoints permanently unreachable behind the gate. Defer to the gate.
    """
    if getattr(request.app.state, "auth_required", False):
        # Gate is authoritative. It attaches ``request.state.session`` on
        # success and 401s otherwise, so a request that reached us is already
        # authenticated. Belt-and-braces: confirm the session is present.
        if getattr(request.state, "session", None) is not None:
            return
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not _has_valid_session_token(request):
        raise HTTPException(status_code=401, detail="Unauthorized")


# Accepted Host header values for loopback binds. DNS rebinding attacks
# point a victim browser at an attacker-controlled hostname (evil.test)
# which resolves to 127.0.0.1 after a TTL flip — bypassing same-origin
# checks because the browser now considers evil.test and our dashboard
# "same origin". Validating the Host header at the app layer rejects any
# request whose Host isn't one we bound for. See GHSA-ppp5-vxwm-4cf7.
_LOOPBACK_HOST_VALUES: frozenset = frozenset({
    "localhost", "127.0.0.1", "::1",
})


def should_require_auth(host: str, allow_public: bool = False) -> bool:
    """Return True iff the dashboard auth gate must be active.

    Truth table:
      host == loopback        → False (no auth — local-only, trusted operator)
      host != loopback        → True  (gate engages — OAuth or password required)

    "Loopback" is 127.0.0.1, localhost, ::1. RFC1918 / CGNAT / link-local are
    deliberately treated as PUBLIC — a hostile device on the same LAN is exactly
    the threat model the gate is designed for.

    ``allow_public`` (the legacy ``--insecure`` escape hatch) NO LONGER disables
    the gate. It is accepted for backward-compat with old launch scripts and
    desktop shells but is ignored: a non-loopback bind ALWAYS requires an auth
    provider (OAuth or the bundled password provider). This closes the
    unauthenticated-public-dashboard hole behind the June 2026 ``hermes-0day``
    MCP-persistence campaign, where ``--insecure --host 0.0.0.0`` left the
    config/MCP/agent surface open to internet scanners.
    """
    return host not in _LOOPBACK_HOST_VALUES


def _is_accepted_host(host_header: str, bound_host: str) -> bool:
    """True if the Host header targets the interface we bound to.

    Accepts:
    - Exact bound host (with or without port suffix)
    - Loopback aliases when bound to loopback
    - Any host when bound to 0.0.0.0 (explicit opt-in to non-loopback,
      no protection possible at this layer)
    """
    if not host_header:
        return False
    # Strip port suffix. IPv6 addresses use bracket notation:
    #   [::1]         — no port
    #   [::1]:9119    — with port
    # Plain hosts/v4:
    #   localhost:9119
    #   127.0.0.1:9119
    h = host_header.strip()
    if h.startswith("["):
        # IPv6 bracketed — port (if any) follows "]:"
        close = h.find("]")
        if close != -1:
            host_only = h[1:close]  # strip brackets
        else:
            host_only = h.strip("[]")
    else:
        host_only = h.rsplit(":", 1)[0] if ":" in h else h
    host_only = host_only.lower()

    # 0.0.0.0 bind means operator explicitly opted into all-interfaces
    # (requires --insecure per web_server.start_server). No Host-layer
    # defence can protect that mode; rely on operator network controls.
    if bound_host in {"0.0.0.0", "::"}:
        return True

    # Loopback bind: accept the loopback names
    bound_lc = bound_host.lower()
    if bound_lc in _LOOPBACK_HOST_VALUES:
        return host_only in _LOOPBACK_HOST_VALUES

    # Explicit non-loopback bind: require exact host match
    return host_only == bound_lc


@app.middleware("http")
async def host_header_middleware(request: Request, call_next):
    """Reject requests whose Host header doesn't match the bound interface.

    Defends against DNS rebinding: a victim browser on a localhost
    dashboard is tricked into fetching from an attacker hostname that
    TTL-flips to 127.0.0.1. CORS and same-origin checks don't help —
    the browser now treats the attacker origin as same-origin with the
    dashboard. Host-header validation at the app layer catches it.

    See GHSA-ppp5-vxwm-4cf7.
    """
    # Store the bound host on app.state so this middleware can read it —
    # set by start_server() at listen time.
    bound_host = getattr(app.state, "bound_host", None)
    if bound_host:
        host_header = request.headers.get("host", "")
        if not _is_accepted_host(host_header, bound_host):
            return JSONResponse(
                status_code=400,
                content={
                    "detail": (
                        "Invalid Host header. Dashboard requests must use "
                        "the hostname the server was bound to."
                    ),
                },
            )
    return await call_next(request)


@app.middleware("http")
async def _plugin_api_runtime_gate(request: Request, call_next):
    """Block requests to disabled plugin API routes at request time.

    :func:`_mount_plugin_api_routes` gates at import time, but if a plugin
    is disabled *after* the dashboard is already running, its FastAPI router
    remains mounted until restart.  This middleware enforces the enabled/
    disabled policy on every request to ``/api/plugins/{name}/...`` so that
    runtime config changes take effect immediately.

    Registered BEFORE the auth middlewares (so it executes AFTER them): a
    request that hasn't cleared auth must get auth's 401 first, never this
    gate's 404 — otherwise an unauthenticated caller could fingerprint which
    plugins are installed/enabled by reading the status code. We only reach
    the enabled/disabled check for a request that auth already let through.
    """
    path = request.url.path
    if path.startswith("/api/plugins/"):
        # Only gate authenticated requests. Unauthenticated ones fall
        # through so auth_middleware / the OAuth gate return 401 first and
        # this route can't be used as a plugin-name oracle.
        _authed = (
            getattr(request.state, "token_authenticated", False)
            or getattr(request.app.state, "auth_required", False)
            or _has_valid_session_token(request)
            or _has_valid_query_token(request, path)
        )
        if _authed:
            # Extract plugin name from /api/plugins/<name>/...
            parts = path.split("/")
            # parts: ['', 'api', 'plugins', '<name>', ...]
            if len(parts) >= 4:
                plugin_name = parts[3]
                if plugin_name:
                    try:
                        from hermes_cli.plugins_cmd import (
                            _get_enabled_set,
                            _get_disabled_set,
                        )
                        enabled_set = _get_enabled_set()
                        disabled_set = _get_disabled_set()
                    except Exception:
                        enabled_set = set()
                        disabled_set = set()
                    # Determine plugin source.  Check the cached plugin list;
                    # if not found, assume user plugin (safe default — blocks).
                    plugins = _get_dashboard_plugins()
                    plugin = next(
                        (p for p in plugins if p.get("name") == plugin_name),
                        None,
                    )
                    source = plugin.get("source") if plugin else "user"
                    if source == "user":
                        if plugin_name in disabled_set or plugin_name not in enabled_set:
                            return JSONResponse(
                                status_code=404,
                                content={"detail": "Plugin not found"},
                            )
                    elif source == "bundled":
                        if plugin_name in disabled_set:
                            return JSONResponse(
                                status_code=404,
                                content={"detail": "Plugin not found"},
                            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Dashboard OAuth auth gate — engaged only when start_server flags the
# bind as non-loopback-without-insecure.  No-op pass-through in loopback
# mode so the legacy auth_middleware (below) handles those binds via
# the injected ``_SESSION_TOKEN``.  Registered between host_header and
# auth_middleware so the order is: host check → cookie auth → token auth.
# ---------------------------------------------------------------------------


@app.middleware("http")
async def _dashboard_auth_gate(request: Request, call_next):
    from hermes_cli.dashboard_auth.middleware import gated_auth_middleware
    return await gated_auth_middleware(request, call_next)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Require the session token on all /api/ routes except the public list."""
    # A request already authenticated by the token-auth seam (a service caller
    # presenting a bearer token on a registered token route) carries
    # ``token_authenticated`` — never bounce it through the cookie/session gate.
    if getattr(request.state, "token_authenticated", False):
        return await call_next(request)
    # When the OAuth gate is active, cookie-based auth (gated_auth_middleware
    # above) is authoritative.  The legacy _SESSION_TOKEN path is loopback-only
    # and is skipped here so the gate's session attachment isn't overridden.
    if getattr(request.app.state, "auth_required", False):
        return await call_next(request)
    path = request.url.path
    if path.startswith("/api/") and path not in _PUBLIC_API_PATHS:
        if not _has_valid_session_token(request) and not _has_valid_query_token(request, path):
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
            )
    return await call_next(request)


@app.middleware("http")
async def _token_auth_seam(request: Request, call_next):
    """Outermost auth seam: non-interactive bearer-token auth for opted-in routes.

    Registered LAST so it runs FIRST (Starlette middleware is outermost-last).
    A registered token route is fully owned here — authenticate by token,
    attach the principal + ``token_authenticated`` flag, and let the downstream
    cookie/session gates skip enforcement. Non-token routes pass straight
    through untouched.
    """
    from hermes_cli.dashboard_auth.token_auth import token_auth_middleware
    return await token_auth_middleware(request, call_next)


# ---------------------------------------------------------------------------
# Config schema — auto-generated from DEFAULT_CONFIG
# ---------------------------------------------------------------------------

# Manual overrides for fields that need select options or custom types
_SCHEMA_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "model": {
        "type": "string",
        "description": "Default model (e.g. anthropic/claude-sonnet-4.6)",
        "category": "general",
    },
    "model_context_length": {
        "type": "number",
        "description": "Context window override (0 = auto-detect from model metadata)",
        "category": "general",
    },
    "terminal.backend": {
        "type": "select",
        "description": "Terminal execution backend",
        "options": ["local", "docker", "ssh", "modal", "daytona", "singularity"],
    },
    "terminal.modal_mode": {
        "type": "select",
        "description": "Modal sandbox mode",
        "options": ["sandbox", "function"],
    },
    "tts.provider": {
        "type": "select",
        "description": "Text-to-speech provider",
        "options": ["edge", "elevenlabs", "openai", "neutts"],
    },
    "stt.provider": {
        "type": "select",
        "description": "Speech-to-text provider",
        # "mistral" temporarily removed — mistralai PyPI package quarantined
        # (malicious 2.4.6 release on 2026-05-12). Restore once available.
        "options": ["local", "groq", "openai", "xai", "elevenlabs"],
    },
    "stt.elevenlabs.model_id": {
        "type": "select",
        "description": "ElevenLabs Scribe model",
        "options": ["scribe_v2", "scribe_v1"],
    },
    "tools.tool_search.defer_core": {
        "type": "select",
        "description": "Whether tool_search may defer core Hermes tools behind tool_call",
        "options": ["off", "all"],
        "category": "tools",
    },
    "voice.realtime.enabled": {
        "type": "boolean",
        "description": "Enable the realtime voice websocket path",
        "category": "voice",
    },
    "voice.realtime.engine": {
        "type": "select",
        "description": "Realtime voice engine",
        "options": ["text_oracle_tts", "native_s2s_oracle", "kame_interface_oracle"],
        "category": "voice",
    },
    "voice.realtime.input_codec": {
        "type": "select",
        "description": "Realtime microphone wire codec",
        "options": ["webm_opus", "opus", "pcm16"],
        "category": "voice",
    },
    "voice.realtime.output_codec": {
        "type": "select",
        "description": "Realtime assistant audio wire codec",
        "options": ["opus", "webm_opus", "pcm16"],
        "category": "voice",
    },
    "voice.realtime.input_buffer_limit_bytes": {
        "type": "number",
        "description": "Maximum local realtime input audio buffered before dropping a turn",
        "category": "voice",
    },
    "voice.realtime.input_frame_ms": {
        "type": "number",
        "description": "Desktop microphone frame duration for realtime voice capture",
        "category": "voice",
    },
    "voice.realtime.silence_timeout_ms": {
        "type": "number",
        "description": "Milliseconds of silence before realtime voice closes the current user turn",
        "category": "voice",
    },
    "voice.realtime.speech_level_threshold": {
        "type": "number",
        "description": "Normalized microphone level required to start realtime speech capture",
        "category": "voice",
    },
    "voice.realtime.barge_in_min_speech_ms": {
        "type": "number",
        "description": "Milliseconds of sustained speech over playback before barge-in",
        "category": "voice",
    },
    "voice.realtime.barge_in_min_rms": {
        "type": "number",
        "description": "Minimum 16-bit PCM RMS amplitude required before Discord voice barge-in",
        "category": "voice",
    },
    "voice.realtime.barge_in_stop_playback_deadline_ms": {
        "type": "number",
        "description": "Maximum milliseconds allowed to stop assistant playback after confirmed barge-in",
        "category": "voice",
    },
    "voice.realtime.pre_roll_ms": {
        "type": "number",
        "description": "Milliseconds of local microphone pre-roll kept before speech starts",
        "category": "voice",
    },
    "voice.realtime.require_live_like": {
        "type": "boolean",
        "description": "Require native S2S or streaming STT/TTS before realtime voice is considered available",
        "category": "voice",
    },
    "voice.realtime.frontend_provider": {
        "type": "select",
        "description": "Low-latency realtime interface provider",
        "options": ["reference", "gemma4", "openai_realtime", "gemini_live"],
        "category": "voice",
    },
    "voice.realtime.frontend_model": {
        "type": "string",
        "description": "Low-latency realtime interface model",
        "category": "voice",
    },
    "voice.realtime.interface_base_url": {
        "type": "string",
        "description": "OpenAI-compatible base URL for the low-latency KAME interface model",
        "category": "voice",
    },
    "voice.realtime.interface_temperature": {
        "type": "number",
        "description": "Sampling temperature for the low-latency KAME interface model",
        "category": "voice",
    },
    "voice.realtime.interface_max_output_tokens": {
        "type": "number",
        "description": "Maximum output tokens for one KAME interface routing decision",
        "category": "voice",
    },
    "voice.realtime.interface_timeout_seconds": {
        "type": "number",
        "description": "Seconds to wait for the KAME interface model before treating it as unavailable",
        "category": "voice",
    },
    "voice.realtime.interface_max_audio_seconds": {
        "type": "number",
        "description": "Maximum native-audio segment seconds sent to the KAME interface model",
        "category": "voice",
    },
    "voice.realtime.interface_audio_input": {
        "type": "select",
        "description": "How the KAME reflex receives user input",
        "options": ["auto", "native_audio", "text_fallback"],
        "category": "voice",
    },
    "voice.realtime.asr_mode": {
        "type": "select",
        "description": "ASR role in KAME realtime voice",
        "options": ["disabled", "on_escalation", "speculative", "debug", "fallback"],
        "category": "voice",
    },
    "voice.realtime.asr_provider": {
        "type": "select",
        "description": "ASR provider used for oracle-verbatim evidence or text fallback",
        "options": ["streaming_stt", "deepgram", "elevenlabs", "cartesia", "nvidia_speech", "local_speech", "reference"],
        "category": "voice",
    },
    "voice.realtime.asr_model": {
        "type": "string",
        "description": "ASR model used for oracle-verbatim evidence or text fallback",
        "category": "voice",
    },
    "voice.realtime.asr_base_url": {
        "type": "string",
        "description": "Base URL for the KAME ASR evidence or text-fallback service; mirrors streaming_stt_base_url",
        "category": "voice",
    },
    "voice.realtime.oracle_provider": {
        "type": "string",
        "description": "Local oracle provider registration for KAME realtime voice",
        "category": "voice",
    },
    "voice.realtime.oracle_provider_name": {
        "type": "string",
        "description": "Display name for the KAME local oracle provider",
        "category": "voice",
    },
    "voice.realtime.preferred_local_oracle_model": {
        "type": "string",
        "description": "Preferred local oracle provider target label for KAME realtime voice; Hermes still chooses the active oracle through /model",
        "category": "voice",
    },
    "voice.realtime.oracle_base_url": {
        "type": "string",
        "description": "OpenAI-compatible base URL for registering a KAME local oracle endpoint",
        "category": "voice",
    },
    "voice.realtime.oracle_api_mode": {
        "type": "select",
        "description": "Wire protocol used by the registered KAME local oracle endpoint",
        "options": ["chat_completions", "anthropic_messages", "codex_responses"],
        "category": "voice",
    },
    "voice.realtime.oracle_timeout_seconds": {
        "type": "number",
        "description": "Seconds to wait for a Hermes oracle voice response before speaking a timeout status",
        "category": "voice",
    },
    "voice.realtime.max_spoken_sentences": {
        "type": "number",
        "description": "Maximum spoken sentences for KAME oracle voice responses",
        "category": "voice",
    },
    "voice.realtime.voice_response_policy": {
        "type": "select",
        "description": "How KAME shapes spoken oracle responses",
        "options": ["sentence_cap", "brief_summary", "full"],
        "category": "voice",
    },
    "voice.realtime.tts_provider": {
        "type": "select",
        "description": "TTS provider used for KAME spoken output",
        "options": ["streaming_tts", "deepgram", "elevenlabs", "cartesia", "nvidia_speech", "local_speech", "reference"],
        "category": "voice",
    },
    "voice.realtime.tts_model": {
        "type": "string",
        "description": "TTS model used for KAME spoken output",
        "category": "voice",
    },
    "voice.realtime.tts_voice": {
        "type": "string",
        "description": "TTS voice identifier used for KAME spoken output",
        "category": "voice",
    },
    "voice.realtime.tts_base_url": {
        "type": "string",
        "description": "Base URL for the KAME TTS service; mirrors streaming_tts_base_url",
        "category": "voice",
    },
    "voice.realtime.fallback_policy": {
        "type": "select",
        "description": "Fallback behavior when the KAME realtime sidecar or local stack is unavailable",
        "options": ["legacy_voice", "text_only", "fail_closed"],
        "category": "voice",
    },
    "voice.realtime.production_evidence_report": {
        "type": "string",
        "description": "Path to a verified realtime voice smoke report required for production readiness",
        "category": "voice",
    },
    "voice.realtime.production_evidence_min_runs": {
        "type": "number",
        "description": "Minimum verified realtime voice smoke report runs required for production readiness",
        "category": "voice",
    },
    "voice.realtime.production_evidence_max_age_days": {
        "type": "number",
        "description": "Maximum age in days for realtime voice production evidence before it must be re-collected",
        "category": "voice",
    },
    "voice.realtime.production_review_report": {
        "type": "string",
        "description": "Path to a realtime voice production launch-review JSON report required after smoke evidence passes",
        "category": "voice",
    },
    "voice.realtime.production_languages": {
        "type": "list",
        "description": "BCP-47 language tags Hermes treats as production realtime voice targets; defaults to English and Japanese",
        "category": "voice",
    },
    "voice.realtime.production_scripts": {
        "type": "list",
        "description": "ISO 15924 scripts covered by production realtime voice acceptance; defaults to Latn and Jpan",
        "category": "voice",
    },
    "voice.realtime.best_effort_languages": {
        "type": "boolean",
        "description": "Allow other languages through captions, prompts, and provider auto-detection without claiming production quality",
        "category": "voice",
    },
    "voice.realtime.turn_acknowledgement.enabled": {
        "type": "boolean",
        "description": "Speak a short acknowledgement immediately after a final transcript while the full voice response is generated",
        "category": "voice",
    },
    "voice.realtime.turn_acknowledgement.text": {
        "type": "string",
        "description": "Short spoken acknowledgement used for realtime voice turns before the final response is ready",
        "category": "voice",
    },
    "voice.realtime.routing.allow_local_greetings": {
        "type": "boolean",
        "description": "Allow the KAME reflex to answer greetings and hear-me checks without the oracle",
        "category": "voice",
    },
    "voice.realtime.routing.allow_local_clarifications": {
        "type": "boolean",
        "description": "Allow the KAME reflex to ask short clarification questions without the oracle",
        "category": "voice",
    },
    "voice.realtime.routing.require_oracle_for_tools": {
        "type": "boolean",
        "description": "Require Hermes oracle authority for tool-using voice turns",
        "category": "voice",
    },
    "voice.realtime.routing.require_oracle_for_memory": {
        "type": "boolean",
        "description": "Require Hermes oracle authority for memory-dependent voice turns",
        "category": "voice",
    },
    "voice.realtime.routing.require_oracle_for_files": {
        "type": "boolean",
        "description": "Require Hermes oracle authority for file or project voice turns",
        "category": "voice",
    },
    "voice.realtime.routing.local_confidence_threshold": {
        "type": "number",
        "description": "Minimum reflex confidence required for local KAME replies",
        "category": "voice",
    },
    "voice.realtime.metrics.enabled": {
        "type": "boolean",
        "description": "Enable realtime voice turn and provider metrics",
        "category": "voice",
    },
    "voice.realtime.metrics.log_turn_spans": {
        "type": "boolean",
        "description": "Log realtime voice turn latency spans",
        "category": "voice",
    },
    "voice.realtime.metrics.log_provider_spans": {
        "type": "boolean",
        "description": "Log realtime voice provider latency spans",
        "category": "voice",
    },
    "voice.realtime.oracle_jobs.enabled": {
        "type": "boolean",
        "description": "Run KAME oracle turns as bounded background jobs instead of blocking the live reflex",
        "category": "voice",
    },
    "voice.realtime.oracle_jobs.max_concurrent": {
        "type": "number",
        "description": "Maximum concurrent Hermes oracle jobs for one realtime voice session",
        "category": "voice",
    },
    "voice.realtime.oracle_jobs.queue_limit": {
        "type": "number",
        "description": "Maximum queued KAME oracle jobs accepted while workers are busy",
        "category": "voice",
    },
    "voice.realtime.oracle_jobs.default_priority": {
        "type": "select",
        "description": "Default priority for spoken KAME oracle jobs",
        "options": ["high", "normal", "low"],
        "category": "voice",
    },
    "voice.realtime.oracle_jobs.overflow_policy": {
        "type": "select",
        "description": "Behavior when the KAME oracle job queue is full",
        "options": ["queue", "reject", "reprioritize"],
        "category": "voice",
    },
    "voice.realtime.oracle_jobs.shutdown_timeout_seconds": {
        "type": "number",
        "description": "Seconds to wait for background oracle jobs during voice session shutdown",
        "category": "voice",
    },
    "voice.realtime.oracle_jobs.speak_terminal_results": {
        "type": "boolean",
        "description": "Speak concise completed or failed oracle job results when still relevant",
        "category": "voice",
    },
    "voice.realtime.oracle_jobs.audit_ledger_path": {
        "type": "text",
        "description": "Optional JSONL path for redacted VoiceOps oracle-job lifecycle audit events",
        "category": "voice",
    },
    "voice.realtime.oracle_tool_router.enabled": {
        "type": "boolean",
        "description": "Let realtime voice choose narrow Hermes oracle toolsets from reflex intent metadata",
        "category": "voice",
    },
    "voice.realtime.oracle_tool_router.mode": {
        "type": "select",
        "description": "Realtime voice oracle tool routing mode",
        "options": ["deterministic"],
        "category": "voice",
    },
    "voice.realtime.oracle_tool_router.voiceops_toolsets": {
        "type": "list",
        "description": "Toolsets exposed to the Hermes oracle for VoiceOps spend/provision/phone requests",
        "category": "voice",
    },
    "voice.realtime.oracle_tool_router.default_toolsets": {
        "type": "list",
        "description": "Optional toolsets exposed to the Hermes oracle for non-VoiceOps realtime voice turns",
        "category": "voice",
    },
    "voice.realtime.oracle_tool_router.tool_search.enabled": {
        "type": "select",
        "description": "Tool-search activation for realtime voice oracle sessions",
        "options": ["auto", "on", "off"],
        "category": "voice",
    },
    "voice.realtime.oracle_tool_router.tool_search.defer_core": {
        "type": "select",
        "description": "Whether realtime voice oracle sessions defer core Hermes tools behind tool_search/tool_call",
        "options": ["off", "all"],
        "category": "voice",
    },
    "voice.realtime.output_events.caption_aliases": {
        "type": "boolean",
        "description": "Emit assistant.caption.partial/final aliases alongside legacy assistant text events",
        "category": "voice",
    },
    "voice.realtime.output_events.audio_aliases": {
        "type": "boolean",
        "description": "Emit assistant.audio.chunk aliases alongside legacy audio.output.chunk events",
        "category": "voice",
    },
    "voice.realtime.quality_targets_ms.audio_to_partial_transcript_ms": {
        "type": "number",
        "description": "Target milliseconds from user audio to first partial transcript",
        "category": "voice",
    },
    "voice.realtime.quality_targets_ms.final_transcript_to_first_text_ms": {
        "type": "number",
        "description": "Target milliseconds from final transcript to first assistant text",
        "category": "voice",
    },
    "voice.realtime.quality_targets_ms.final_transcript_to_first_audio_ms": {
        "type": "number",
        "description": "Target milliseconds from final transcript to first assistant audio",
        "category": "voice",
    },
    "voice.realtime.quality_targets_ms.barge_in_ack_ms": {
        "type": "number",
        "description": "Target milliseconds for backend barge-in acknowledgement",
        "category": "voice",
    },
    "voice.realtime.quality_targets_ms.barge_in_confirmed_to_playback_stopped_ms": {
        "type": "number",
        "description": "Target milliseconds from confirmed speech barge-in to Discord playback stopped",
        "category": "voice",
    },
    "voice.realtime.quality_targets_ms.kame_speech_end_to_interface_decision_ms": {
        "type": "number",
        "description": "Target milliseconds from speech boundary to KAME reflex routing decision",
        "category": "voice",
    },
    "voice.realtime.quality_targets_ms.kame_interface_decision_to_local_first_audio_ms": {
        "type": "number",
        "description": "Target milliseconds from KAME local decision to first local reply audio",
        "category": "voice",
    },
    "voice.realtime.quality_targets_ms.kame_speech_end_to_local_first_audio_ms": {
        "type": "number",
        "description": "Target milliseconds from speech boundary to first local KAME reply audio",
        "category": "voice",
    },
    "voice.realtime.quality_targets_ms.kame_final_transcript_to_interface_decision_ms": {
        "type": "number",
        "description": "Target milliseconds from final transcript to KAME interface routing decision",
        "category": "voice",
    },
    "voice.realtime.quality_targets_ms.kame_interface_decision_to_defer_first_audio_ms": {
        "type": "number",
        "description": "Target milliseconds from KAME defer decision to first acknowledgement audio",
        "category": "voice",
    },
    "voice.realtime.quality_targets_ms.kame_speech_end_to_defer_first_audio_ms": {
        "type": "number",
        "description": "Target milliseconds from speech boundary to first KAME defer acknowledgement audio",
        "category": "voice",
    },
    "voice.realtime.quality_targets_ms.kame_interface_decision_to_oracle_accepted_ms": {
        "type": "number",
        "description": "Target milliseconds from KAME routing decision to Hermes oracle acceptance",
        "category": "voice",
    },
    "voice.realtime.quality_targets_ms.kame_oracle_first_token_to_first_tts_audio_ms": {
        "type": "number",
        "description": "Target milliseconds from Hermes oracle first token to first synthesized TTS audio",
        "category": "voice",
    },
    "voice.realtime.quality_targets_ms.kame_first_tts_audio_to_playback_start_ms": {
        "type": "number",
        "description": "Target milliseconds from first synthesized TTS audio to Discord playback start",
        "category": "voice",
    },
    "voice.realtime.quality_targets_ms.kame_speech_end_to_first_audio_ms": {
        "type": "number",
        "description": "Target milliseconds from speech boundary to first synthesized KAME assistant audio",
        "category": "voice",
    },
    "voice.realtime.quality_targets_ms.kame_speech_end_to_playback_start_ms": {
        "type": "number",
        "description": "Target milliseconds from speech boundary to first audible KAME assistant audio",
        "category": "voice",
    },
    "voice.realtime.sidecar_base_url": {
        "type": "string",
        "description": "Remote or local realtime voice sidecar URL",
        "category": "voice",
    },
    "voice.realtime.sidecar_token_env": {
        "type": "string",
        "description": "Environment variable containing the realtime voice sidecar bearer token",
        "category": "voice",
    },
    "voice.realtime.sidecar_connect_timeout_seconds": {
        "type": "number",
        "description": "Seconds to wait when connecting to a realtime voice sidecar",
        "category": "voice",
    },
    "voice.realtime.sidecar_close_timeout_seconds": {
        "type": "number",
        "description": "Seconds to wait while closing a realtime voice sidecar session",
        "category": "voice",
    },
    "voice.realtime.vllm_base_url": {
        "type": "string",
        "description": "OpenAI-compatible vLLM base URL for a local KAME audio reflex",
        "category": "voice",
    },
    "voice.realtime.vllm_model": {
        "type": "string",
        "description": "Local KAME audio reflex model served by vLLM",
        "category": "voice",
    },
    "voice.realtime.interface_api_key_env": {
        "type": "string",
        "description": "Environment variable containing the KAME interface/reflex endpoint bearer token; secrets stay outside config",
        "category": "voice",
    },
    "voice.realtime.streaming_stt_base_url": {
        "type": "string",
        "description": "Optional compatible streaming STT bridge URL used by the managed reference sidecar",
        "category": "voice",
    },
    "voice.realtime.streaming_stt_model": {
        "type": "string",
        "description": "Optional streaming STT bridge model name for diagnostics",
        "category": "voice",
    },
    "voice.realtime.streaming_stt_token_env": {
        "type": "string",
        "description": "Environment variable containing the streaming STT bridge bearer token",
        "category": "voice",
    },
    "voice.realtime.streaming_tts_base_url": {
        "type": "string",
        "description": "Optional compatible streaming TTS bridge URL used by the managed reference sidecar",
        "category": "voice",
    },
    "voice.realtime.streaming_tts_model": {
        "type": "string",
        "description": "Optional streaming TTS bridge model name for diagnostics",
        "category": "voice",
    },
    "voice.realtime.streaming_tts_voice": {
        "type": "string",
        "description": "Optional streaming TTS bridge voice identifier for diagnostics",
        "category": "voice",
    },
    "voice.realtime.streaming_tts_token_env": {
        "type": "string",
        "description": "Environment variable containing the streaming TTS bridge bearer token",
        "category": "voice",
    },
    "voice.realtime.openai_realtime_api_key_env": {
        "type": "string",
        "description": "Environment variable containing the OpenAI Realtime API key; secrets stay outside config",
        "category": "voice",
    },
    "voice.realtime.openai_realtime_base_url": {
        "type": "string",
        "description": "OpenAI Realtime WebSocket base URL for managed reference-sidecar sessions",
        "category": "voice",
    },
    "voice.realtime.openai_realtime_voice": {
        "type": "string",
        "description": "OpenAI Realtime voice used by the low-latency frontend provider",
        "category": "voice",
    },
    "voice.realtime.openai_realtime_transcription_model": {
        "type": "string",
        "description": "OpenAI Realtime transcription model used for frontend speech-to-text events",
        "category": "voice",
    },
    "voice.realtime.gemini_live_api_key_env": {
        "type": "string",
        "description": "Environment variable containing the Gemini Live API key; secrets stay outside config",
        "category": "voice",
    },
    "voice.realtime.gemini_live_base_url": {
        "type": "string",
        "description": "Gemini Live WebSocket base URL for managed reference-sidecar sessions",
        "category": "voice",
    },
    "voice.realtime.gemini_live_voice": {
        "type": "string",
        "description": "Gemini Live voice used by the low-latency frontend provider",
        "category": "voice",
    },
    "voice.realtime.gemini_live_google_search": {
        "type": "boolean",
        "description": "Allow the Gemini Live frontend provider to use Google Search as a frontend context tool",
        "category": "voice",
    },
    "voice.realtime.gemini_live_oracle_tool": {
        "type": "boolean",
        "description": "Expose only the KAME ask_hermes_oracle bridge tool to Gemini Live; Hermes still owns real tool execution",
        "category": "voice",
    },
    "voice.realtime.spark_base_url": {
        "type": "string",
        "description": "Deprecated alias for voice.realtime.sidecar_base_url",
        "category": "voice",
    },
    "discord.realtime_voice.enabled": {
        "type": "boolean",
        "description": "Enable Discord voice-channel realtime sidecar mode",
        "category": "discord",
    },
    "discord.realtime_voice.sidecar_base_url": {
        "type": "string",
        "description": "Discord realtime voice sidecar URL; falls back to voice.realtime.sidecar_base_url when unset",
        "category": "discord",
    },
    "discord.realtime_voice.sidecar_token": {
        "type": "string",
        "description": "Bearer token for the Discord realtime voice sidecar; prefer HERMES_REALTIME_VOICE_SIDECAR_TOKEN for secrets",
        "category": "discord",
    },
    "discord.realtime_voice.frontend_provider": {
        "type": "select",
        "description": "Low-latency realtime voice frontend provider override for Discord voice sessions",
        "options": ["reference", "gemma4", "openai_realtime", "gemini_live"],
        "category": "discord",
    },
    "discord.realtime_voice.frontend_model": {
        "type": "string",
        "description": "Low-latency realtime interface model override for Discord voice sessions",
        "category": "discord",
    },
    "discord.realtime_voice.interface_base_url": {
        "type": "string",
        "description": "OpenAI-compatible base URL for the Discord KAME interface model",
        "category": "discord",
    },
    "discord.realtime_voice.interface_temperature": {
        "type": "number",
        "description": "Sampling temperature for the Discord KAME interface model",
        "category": "discord",
    },
    "discord.realtime_voice.interface_max_output_tokens": {
        "type": "number",
        "description": "Maximum output tokens for one Discord KAME interface routing decision",
        "category": "discord",
    },
    "discord.realtime_voice.interface_timeout_seconds": {
        "type": "number",
        "description": "Seconds to wait for the Discord KAME interface model before treating it as unavailable",
        "category": "discord",
    },
    "discord.realtime_voice.interface_max_audio_seconds": {
        "type": "number",
        "description": "Maximum native-audio segment seconds sent to the Discord KAME interface model",
        "category": "discord",
    },
    "discord.realtime_voice.interface_api_key_env": {
        "type": "string",
        "description": "Environment variable containing the Discord KAME interface/reflex endpoint bearer token; secrets stay outside config",
        "category": "discord",
    },
    "discord.realtime_voice.interface_audio_input": {
        "type": "select",
        "description": "How the Discord KAME reflex receives user input",
        "options": ["auto", "native_audio", "text_fallback"],
        "category": "discord",
    },
    "discord.realtime_voice.asr_mode": {
        "type": "select",
        "description": "ASR role in Discord KAME realtime voice",
        "options": ["disabled", "on_escalation", "speculative", "debug", "fallback"],
        "category": "discord",
    },
    "discord.realtime_voice.asr_provider": {
        "type": "select",
        "description": "ASR provider override for Discord oracle-verbatim evidence",
        "options": ["streaming_stt", "deepgram", "elevenlabs", "cartesia", "nvidia_speech", "local_speech", "reference"],
        "category": "discord",
    },
    "discord.realtime_voice.asr_model": {
        "type": "string",
        "description": "ASR model override for Discord oracle-verbatim evidence",
        "category": "discord",
    },
    "discord.realtime_voice.asr_base_url": {
        "type": "string",
        "description": "Base URL for the Discord KAME ASR evidence or text-fallback service; mirrors streaming_stt_base_url",
        "category": "discord",
    },
    "discord.realtime_voice.oracle_provider": {
        "type": "string",
        "description": "Local oracle provider registration for Discord KAME realtime voice",
        "category": "discord",
    },
    "discord.realtime_voice.oracle_provider_name": {
        "type": "string",
        "description": "Display name for the Discord KAME local oracle provider",
        "category": "discord",
    },
    "discord.realtime_voice.preferred_local_oracle_model": {
        "type": "string",
        "description": "Preferred local oracle provider target label for Discord KAME realtime voice; Hermes still chooses the active oracle through /model",
        "category": "discord",
    },
    "discord.realtime_voice.oracle_base_url": {
        "type": "string",
        "description": "OpenAI-compatible base URL for registering the Discord KAME local oracle endpoint",
        "category": "discord",
    },
    "discord.realtime_voice.oracle_api_mode": {
        "type": "select",
        "description": "Wire protocol used by the registered Discord KAME local oracle endpoint",
        "options": ["chat_completions", "anthropic_messages", "codex_responses"],
        "category": "discord",
    },
    "discord.realtime_voice.oracle_timeout_seconds": {
        "type": "number",
        "description": "Seconds to wait for a Discord realtime voice oracle response before speaking a timeout status",
        "category": "discord",
    },
    "discord.realtime_voice.max_spoken_sentences": {
        "type": "number",
        "description": "Maximum spoken sentences for Discord KAME oracle voice responses",
        "category": "discord",
    },
    "discord.realtime_voice.voice_response_policy": {
        "type": "select",
        "description": "How Discord KAME shapes spoken oracle responses",
        "options": ["sentence_cap", "brief_summary", "full"],
        "category": "discord",
    },
    "discord.realtime_voice.tts_provider": {
        "type": "select",
        "description": "TTS provider override for Discord realtime voice fallback speech",
        "options": ["streaming_tts", "deepgram", "elevenlabs", "cartesia", "nvidia_speech", "local_speech", "reference"],
        "category": "discord",
    },
    "discord.realtime_voice.tts_model": {
        "type": "string",
        "description": "TTS model override for Discord KAME spoken output",
        "category": "discord",
    },
    "discord.realtime_voice.tts_voice": {
        "type": "string",
        "description": "TTS voice identifier override for Discord KAME spoken output",
        "category": "discord",
    },
    "discord.realtime_voice.tts_base_url": {
        "type": "string",
        "description": "Base URL for the Discord KAME TTS service; mirrors streaming_tts_base_url",
        "category": "discord",
    },
    "discord.realtime_voice.fallback_policy": {
        "type": "select",
        "description": "Fallback behavior when the Discord KAME realtime sidecar or local stack is unavailable",
        "options": ["legacy_voice", "text_only", "fail_closed"],
        "category": "discord",
    },
    "discord.realtime_voice.barge_in_min_speech_ms": {
        "type": "number",
        "description": "Minimum speech duration before Discord voice barge-in can interrupt playback",
        "category": "discord",
    },
    "discord.realtime_voice.barge_in_min_rms": {
        "type": "number",
        "description": "Minimum Discord PCM RMS energy required before barge-in can interrupt playback",
        "category": "discord",
    },
    "discord.realtime_voice.barge_in_stop_playback_deadline_ms": {
        "type": "number",
        "description": "Target maximum Discord playback stop latency after barge-in",
        "category": "discord",
    },
    "discord.realtime_voice.sidecar_connect_timeout_seconds": {
        "type": "number",
        "description": "Seconds to wait when connecting Discord voice to the realtime sidecar",
        "category": "discord",
    },
    "discord.realtime_voice.sidecar_close_timeout_seconds": {
        "type": "number",
        "description": "Seconds to wait while closing a Discord realtime sidecar session",
        "category": "discord",
    },
    "discord.realtime_voice.turn_acknowledgement.enabled": {
        "type": "boolean",
        "description": "Speak a short Discord KAME acknowledgement while the oracle response is generated",
        "category": "discord",
    },
    "discord.realtime_voice.turn_acknowledgement.text": {
        "type": "string",
        "description": "Short spoken Discord KAME acknowledgement before the final response is ready",
        "category": "discord",
    },
    "discord.realtime_voice.metrics.enabled": {
        "type": "boolean",
        "description": "Enable Discord KAME realtime voice turn and provider metrics",
        "category": "discord",
    },
    "discord.realtime_voice.metrics.log_turn_spans": {
        "type": "boolean",
        "description": "Log Discord KAME realtime voice turn latency spans",
        "category": "discord",
    },
    "discord.realtime_voice.metrics.log_provider_spans": {
        "type": "boolean",
        "description": "Log Discord KAME realtime voice provider latency spans",
        "category": "discord",
    },
    "discord.realtime_voice.oracle_jobs.enabled": {
        "type": "boolean",
        "description": "Run Discord KAME oracle turns as bounded background jobs instead of blocking the live reflex",
        "category": "discord",
    },
    "discord.realtime_voice.oracle_jobs.max_concurrent": {
        "type": "number",
        "description": "Maximum concurrent Hermes oracle jobs for one Discord realtime voice session",
        "category": "discord",
    },
    "discord.realtime_voice.oracle_jobs.queue_limit": {
        "type": "number",
        "description": "Maximum queued Discord KAME oracle jobs accepted while workers are busy",
        "category": "discord",
    },
    "discord.realtime_voice.oracle_jobs.default_priority": {
        "type": "select",
        "description": "Default priority for spoken Discord KAME oracle jobs",
        "options": ["high", "normal", "low"],
        "category": "discord",
    },
    "discord.realtime_voice.oracle_jobs.overflow_policy": {
        "type": "select",
        "description": "Behavior when the Discord KAME oracle job queue is full",
        "options": ["queue", "reject", "reprioritize"],
        "category": "discord",
    },
    "discord.realtime_voice.oracle_jobs.shutdown_timeout_seconds": {
        "type": "number",
        "description": "Seconds to wait for Discord background oracle jobs during voice session shutdown",
        "category": "discord",
    },
    "discord.realtime_voice.oracle_jobs.speak_terminal_results": {
        "type": "boolean",
        "description": "Speak concise completed or failed Discord oracle job results when still relevant",
        "category": "discord",
    },
    "discord.realtime_voice.oracle_jobs.audit_ledger_path": {
        "type": "text",
        "description": "Optional JSONL path for redacted Discord VoiceOps oracle-job lifecycle audit events",
        "category": "discord",
    },
    "discord.realtime_voice.oracle_tool_router.enabled": {
        "type": "boolean",
        "description": "Let Discord realtime voice choose narrow Hermes oracle toolsets from reflex intent metadata",
        "category": "discord",
    },
    "discord.realtime_voice.oracle_tool_router.mode": {
        "type": "select",
        "description": "Discord realtime voice oracle tool routing mode",
        "options": ["deterministic"],
        "category": "discord",
    },
    "discord.realtime_voice.oracle_tool_router.voiceops_toolsets": {
        "type": "list",
        "description": "Toolsets exposed to the Hermes oracle for Discord VoiceOps spend/provision/phone requests",
        "category": "discord",
    },
    "discord.realtime_voice.oracle_tool_router.default_toolsets": {
        "type": "list",
        "description": "Optional toolsets exposed to the Hermes oracle for non-VoiceOps Discord realtime voice turns",
        "category": "discord",
    },
    "discord.realtime_voice.oracle_tool_router.tool_search.enabled": {
        "type": "select",
        "description": "Tool-search activation for Discord realtime voice oracle sessions",
        "options": ["auto", "on", "off"],
        "category": "discord",
    },
    "discord.realtime_voice.oracle_tool_router.tool_search.defer_core": {
        "type": "select",
        "description": "Whether Discord realtime voice oracle sessions defer core Hermes tools behind tool_search/tool_call",
        "options": ["off", "all"],
        "category": "discord",
    },
    "discord.realtime_voice.output_events.caption_aliases": {
        "type": "boolean",
        "description": "Emit assistant.caption.partial/final aliases for Discord realtime voice sessions",
        "category": "discord",
    },
    "discord.realtime_voice.output_events.audio_aliases": {
        "type": "boolean",
        "description": "Emit assistant.audio.chunk aliases for Discord realtime voice sessions",
        "category": "discord",
    },
    "discord.realtime_voice.routing.allow_local_greetings": {
        "type": "boolean",
        "description": "Allow the Discord KAME reflex to answer greetings and hear-me checks without the oracle",
        "category": "discord",
    },
    "discord.realtime_voice.routing.allow_local_clarifications": {
        "type": "boolean",
        "description": "Allow the Discord KAME reflex to ask short clarification questions without the oracle",
        "category": "discord",
    },
    "discord.realtime_voice.routing.require_oracle_for_tools": {
        "type": "boolean",
        "description": "Require Hermes oracle authority for Discord tool-using voice turns",
        "category": "discord",
    },
    "discord.realtime_voice.routing.require_oracle_for_memory": {
        "type": "boolean",
        "description": "Require Hermes oracle authority for Discord memory-dependent voice turns",
        "category": "discord",
    },
    "discord.realtime_voice.routing.require_oracle_for_files": {
        "type": "boolean",
        "description": "Require Hermes oracle authority for Discord file or project voice turns",
        "category": "discord",
    },
    "discord.realtime_voice.routing.local_confidence_threshold": {
        "type": "number",
        "description": "Minimum reflex confidence required for local Discord KAME replies",
        "category": "discord",
    },
    "display.skin": {
        "type": "select",
        "description": "CLI visual theme",
        "options": ["default", "ares", "mono", "slate"],
    },
    "dashboard.theme": {
        "type": "select",
        "description": "Web dashboard visual theme",
        "options": ["default", "midnight", "ember", "mono", "cyberpunk", "rose"],
    },
    "display.resume_display": {
        "type": "select",
        "description": "How resumed sessions display history",
        "options": ["minimal", "full", "off"],
    },
    "display.busy_input_mode": {
        "type": "select",
        "description": "Input behavior while agent is running",
        "options": ["interrupt", "queue", "steer"],
    },
    "memory.provider": {
        "type": "select",
        "description": "Memory provider plugin",
        "options": ["builtin", "honcho"],
    },
    "approvals.mode": {
        "type": "select",
        "description": "Dangerous command approval mode",
        "options": ["ask", "yolo", "deny"],
    },
    "context.engine": {
        "type": "select",
        "description": "Context management engine",
        "options": ["default", "custom"],
    },
    "human_delay.mode": {
        "type": "select",
        "description": "Simulated typing delay mode",
        "options": ["off", "typing", "fixed"],
    },
    "logging.level": {
        "type": "select",
        "description": "Log level for agent.log",
        "options": ["DEBUG", "INFO", "WARNING", "ERROR"],
    },
    "agent.service_tier": {
        "type": "select",
        "description": "API service tier (OpenAI/Anthropic)",
        "options": ["", "auto", "default", "flex"],
    },
    "delegation.reasoning_effort": {
        "type": "select",
        "description": "Reasoning effort for delegated subagents",
        "options": ["", "low", "medium", "high"],
    },
    "updates.non_interactive_local_changes": {
        "type": "select",
        "description": (
            "When the chat app / gateway updates Hermes (no terminal prompt), "
            "what to do with uncommitted local source edits. 'stash' keeps them "
            "and re-applies them after the update; 'discard' throws them away. "
            "Terminal updates always ask, regardless of this setting."
        ),
        "options": ["stash", "discard"],
    },
}

# Categories with fewer fields get merged into "general" to avoid tab sprawl.
_CATEGORY_MERGE: Dict[str, str] = {
    "privacy": "security",
    "context": "agent",
    "skills": "agent",
    "cron": "agent",
    "network": "agent",
    "checkpoints": "agent",
    "approvals": "security",
    "human_delay": "display",
    "dashboard": "display",
    "code_execution": "agent",
    "prompt_caching": "agent",
    "goals": "agent",
    "updates": "general",
    # `onboarding.profile_build` is the only schema-surfaced onboarding field
    # (`onboarding.seen` is an internal latch dict, not a user setting), so fold
    # it into the agent tab rather than spawning a one-field orphan category.
    "onboarding": "agent",
    # Only `telegram.reactions` currently lives under telegram — fold it in
    # with the other messaging-platform config (discord) so it isn't an
    # orphan tab of one field.
    "telegram": "discord",
    # `computer_use.cua_telemetry` is the only schema-surfaced computer_use
    # field — fold it into the agent tab rather than spawning a one-field
    # orphan category.
    "computer_use": "agent",
}

# Display order for tabs — unlisted categories sort alphabetically after these.
_CATEGORY_ORDER = [
    "general", "agent", "terminal", "display", "delegation",
    "memory", "compression", "security", "browser", "voice",
    "tts", "stt", "logging", "discord", "auxiliary",
]


def _infer_type(value: Any) -> str:
    """Infer a UI field type from a Python value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "number"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "string"


def _build_schema_from_config(
    config: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, Dict[str, Any]]:
    """Walk DEFAULT_CONFIG and produce a flat dot-path → field schema dict."""
    schema: Dict[str, Dict[str, Any]] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key

        # Skip internal / version keys
        if full_key in {"_config_version",}:
            continue

        # Category is the first path component for nested keys, or "general"
        # for top-level scalar fields (model, toolsets, timezone, etc.).
        if prefix:
            category = prefix.split(".")[0]
        elif isinstance(value, dict):
            category = key
        else:
            category = "general"

        if isinstance(value, dict):
            # Recurse into nested dicts
            schema.update(_build_schema_from_config(value, full_key))
        else:
            entry: Dict[str, Any] = {
                "type": _infer_type(value),
                "description": full_key.replace(".", " → ").replace("_", " ").title(),
                "category": category,
            }
            # Apply manual overrides
            if full_key in _SCHEMA_OVERRIDES:
                entry.update(_SCHEMA_OVERRIDES[full_key])
            # Merge small categories
            entry["category"] = _CATEGORY_MERGE.get(entry["category"], entry["category"])
            schema[full_key] = entry
    return schema


CONFIG_SCHEMA = _build_schema_from_config(DEFAULT_CONFIG)

# Inject virtual fields that don't live in DEFAULT_CONFIG but are surfaced
# by the normalize/denormalize cycle.  Insert model_context_length right after
# the "model" key so it renders adjacent in the frontend.
_mcl_entry = _SCHEMA_OVERRIDES["model_context_length"]
_ordered_schema: Dict[str, Dict[str, Any]] = {}
for _k, _v in CONFIG_SCHEMA.items():
    _ordered_schema[_k] = _v
    if _k == "model":
        _ordered_schema["model_context_length"] = _mcl_entry
CONFIG_SCHEMA = _ordered_schema
for _schema_key, _schema_entry in _SCHEMA_OVERRIDES.items():
    if (
        _schema_key.startswith("voice.realtime.quality_targets_ms.")
        or _schema_key.startswith("discord.realtime_voice.oracle_jobs.")
    ) and _schema_key not in CONFIG_SCHEMA:
        CONFIG_SCHEMA[_schema_key] = dict(_schema_entry)


class ConfigUpdate(BaseModel):
    config: dict
    profile: Optional[str] = None


class RealtimeVoiceProfileApply(BaseModel):
    preset: str
    model: str = ""
    voice: str = ""
    api_key_env: str = ""
    interface_provider: str = "gemma4"
    interface_base_url: str = ""
    interface_api_key_env: str = ""
    vllm_model: str = ""
    interface_temperature: float = 0.2
    interface_max_output_tokens: int = 160
    interface_timeout_seconds: float = 0.8
    interface_audio_input: str = "auto"
    interface_max_audio_seconds: float = 30.0
    asr_mode: str = "on_escalation"
    asr_provider: str = "streaming_stt"
    oracle_provider: str = ""
    preferred_local_oracle_model: str = ""
    oracle_base_url: str = ""
    oracle_api_mode: str = "chat_completions"
    oracle_timeout_seconds: float = 60.0
    max_spoken_sentences: int = 2
    oracle_provider_name: str = ""
    voice_response_policy: str = "sentence_cap"
    fallback_policy: str = "legacy_voice"
    streaming_stt_base_url: str = ""
    asr_base_url: str = ""
    streaming_tts_base_url: str = ""
    tts_base_url: str = ""
    streaming_stt_model: str = ""
    streaming_stt_token_env: str = ""
    tts_provider: str = "streaming_tts"
    tts_model: str = ""
    tts_voice: str = ""
    streaming_tts_model: str = ""
    streaming_tts_voice: str = ""
    streaming_tts_token_env: str = ""
    barge_in_min_rms: int = 350
    barge_in_min_speech_ms: int = 120
    barge_in_stop_playback_deadline_ms: int = 150
    allow_local_greetings: bool = True
    allow_local_clarifications: bool = True
    require_oracle_for_tools: bool = True
    require_oracle_for_memory: bool = True
    require_oracle_for_files: bool = True
    local_confidence_threshold: float = 0.75
    metrics_enabled: bool = True
    metrics_log_turn_spans: bool = True
    metrics_log_provider_spans: bool = True
    google_search: bool = False
    oracle_tool: bool = True
    enable_discord: bool = False
    profile: Optional[str] = None


class RealtimeVoiceSmokeRequest(BaseModel):
    require_discord: bool = False
    require_inbound: bool = False
    wait_seconds: float = 5.0
    output_dir: str = ""
    profile: Optional[str] = None


class EnvVarUpdate(BaseModel):
    key: str
    value: str
    profile: Optional[str] = None
    # Optional bearer key for the connectivity probe of a custom/local endpoint
    # (``key == "OPENAI_BASE_URL"``). Self-hosted endpoints that gate
    # ``/v1/models`` behind auth otherwise look "reachable but empty"; sending
    # the key lets the probe enumerate the served models. Ignored for the
    # regular PUT /api/env path (which only reads key/value).
    api_key: str = ""


class EnvVarDelete(BaseModel):
    key: str
    profile: Optional[str] = None


class EnvVarReveal(BaseModel):
    key: str
    profile: Optional[str] = None


class MemoryProviderConfigUpdate(BaseModel):
    values: Dict[str, str] = {}


class MessagingPlatformUpdate(BaseModel):
    enabled: Optional[bool] = None
    env: Dict[str, str] = {}
    clear_env: List[str] = []
    # Explicit body profile beats the query param injected by the global
    # dashboard profile switcher (same precedence as other scoped writes).
    profile: Optional[str] = None


class TelegramOnboardingStart(BaseModel):
    bot_name: Optional[str] = None


class TelegramOnboardingApply(BaseModel):
    allowed_user_ids: List[str]
    profile: Optional[str] = None


class AudioTranscriptionRequest(BaseModel):
    data_url: str
    mime_type: Optional[str] = None


class ManagedFileUpload(BaseModel):
    path: str
    data_url: str
    overwrite: bool = True


class ManagedDirectoryCreate(BaseModel):
    path: str


class ManagedFileDelete(BaseModel):
    path: str
    recursive: bool = False


_AUDIO_MIME_EXTENSIONS: Dict[str, str] = {
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/m4a": ".m4a",
    "audio/mp3": ".mp3",
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/webm": ".webm",
    "audio/x-m4a": ".m4a",
    "audio/x-wav": ".wav",
    "video/webm": ".webm",
}
_MAX_TRANSCRIPTION_UPLOAD_BYTES = 25 * 1024 * 1024


def _audio_extension_for_mime(mime_type: str) -> str:
    normalized = (mime_type or "").split(";", 1)[0].strip().lower()
    return _AUDIO_MIME_EXTENSIONS.get(normalized, ".webm")


class ModelAssignment(BaseModel):
    """Payload for POST /api/model/set — assign a provider/model to a slot.

    scope="main"        → writes model.provider + model.default
    scope="auxiliary"   → writes auxiliary.<task>.provider + auxiliary.<task>.model
    scope="auxiliary" with task=""  → applied to every auxiliary.* slot
    scope="auxiliary" with task="__reset__"  → resets every slot to provider="auto"
    """
    scope: str
    provider: str
    model: str
    task: str = ""
    # Optional OpenAI-compatible endpoint URL. Only honored for custom/local
    # providers on the main slot — lets the GUI configure a self-hosted endpoint
    # (vLLM, llama.cpp, Ollama, …) that needs no API key. The runtime resolver
    # reads model.base_url from config (it ignores OPENAI_BASE_URL), so this is
    # the path that actually wires a local endpoint into resolution.
    base_url: str = ""
    # Optional API key for a custom/local endpoint. Persisted to
    # ``model.api_key`` (where the runtime resolver reads it) so a self-hosted
    # endpoint that requires auth works from the GUI — mirrors the key the
    # ``hermes model`` custom flow collects. Honored only on the main slot for
    # custom/local providers.
    api_key: str = ""
    confirm_expensive_model: bool = False
    profile: Optional[str] = None


class MoaModelSlot(BaseModel):
    provider: str = ""
    model: str = ""


class MoaPresetPayload(BaseModel):
    reference_models: list[MoaModelSlot] = []
    aggregator: MoaModelSlot = MoaModelSlot()
    reference_temperature: float = 0.6
    aggregator_temperature: float = 0.4
    max_tokens: int = 4096
    enabled: bool = True


class MoaConfigPayload(BaseModel):
    default_preset: str = "default"
    active_preset: str = ""
    presets: dict[str, MoaPresetPayload] = {}
    # Backward-compatible flat payload fields used by older dashboard/desktop
    # clients during this PR's transition window.
    reference_models: list[MoaModelSlot] = []
    aggregator: MoaModelSlot = MoaModelSlot()
    reference_temperature: float = 0.6
    aggregator_temperature: float = 0.4
    max_tokens: int = 4096
    enabled: bool = True
    profile: Optional[str] = None


def _normalize_main_model_assignment(provider: str, model: str) -> tuple[str, str]:
    """Normalize a main-slot (provider, model) pair before persisting.

    The Models page has two assignment paths and only one of them was safe:

    - The "Change" picker sends a real Hermes provider slug — fine.
    - The per-card "Use as → Main model" menu sends ``entry.provider``
      from the analytics rows, falling back to the model's VENDOR prefix
      (``modelVendor("anthropic/claude-opus-4.6") == "anthropic"``) when
      the session row has no ``billing_provider`` (older sessions, NULL
      rows).  That wrote ``provider: anthropic`` +
      ``default: anthropic/claude-opus-4.6`` to config — a vendor-prefixed
      OpenRouter slug on the NATIVE Anthropic provider.  New sessions then
      400 against api.anthropic.com ("model: anthropic/claude-opus-4.6 not
      found") and the user reads it as "changing models does nothing".

    Two repairs, both at this single chokepoint so every caller inherits:

    1. Vendor-name → Hermes-provider mapping: when the provider string is
       not a known Hermes provider/alias (e.g. ``moonshotai``, ``x-ai`` is
       known but ``poolside`` isn't) but the model is a vendor-prefixed
       aggregator slug, keep the user's CURRENT aggregator if they're on
       one, else fall back to openrouter.
    2. Model-format normalization for the resolved provider via
       ``normalize_model_for_provider`` (e.g. ``anthropic/claude-opus-4.6``
       on native anthropic → ``claude-opus-4-6``).
    """
    from hermes_cli.models import _KNOWN_PROVIDER_NAMES, normalize_provider
    from hermes_cli.model_normalize import normalize_model_for_provider

    prov_in = (provider or "").strip()
    model_in = (model or "").strip()
    canonical = normalize_provider(prov_in)

    if canonical not in _KNOWN_PROVIDER_NAMES and "/" in model_in:
        # Vendor prefix posing as a provider (analytics fallback). Resolve
        # against the user's current provider when it's an aggregator that
        # serves vendor-prefixed slugs; otherwise default to openrouter.
        try:
            cur_cfg = load_config().get("model", {})
            cur_provider = (
                str(cur_cfg.get("provider", "") or "").strip().lower()
                if isinstance(cur_cfg, dict) else ""
            )
        except Exception:
            cur_provider = ""
        from hermes_cli.models import _AGGREGATOR_PROVIDERS
        if cur_provider and normalize_provider(cur_provider) in _AGGREGATOR_PROVIDERS:
            canonical = normalize_provider(cur_provider)
            prov_in = cur_provider
        else:
            canonical = "openrouter"
            prov_in = "openrouter"

    # Custom/user-config providers keep the model verbatim — the registry
    # normalizer doesn't know their namespaces.
    if canonical in _KNOWN_PROVIDER_NAMES and not canonical.startswith("custom"):
        try:
            normalized_model = normalize_model_for_provider(model_in, canonical)
            if normalized_model:
                model_in = normalized_model
        except Exception:
            _log.debug("model normalization failed for %s/%s", prov_in, model_in, exc_info=True)

    return prov_in, model_in


def _apply_main_model_assignment(
    model_cfg: "Any", provider: str, model: str, base_url: str = "", api_key: str = ""
) -> dict:
    """Apply a main-slot model assignment to a ``model`` config dict in place.

    Sets ``provider``/``default``, then reconciles ``base_url``:

    - An explicitly supplied ``base_url`` is always persisted (covers
      ``custom``/local endpoints and any provider whose key is bound to a
      non-default host).
    - Otherwise, a stale ``base_url`` is cleared ONLY when switching to a
      *different* provider — that URL belonged to the old provider. When the
      provider is unchanged and no new URL is supplied, the existing
      ``base_url`` is preserved. This keeps a user's custom endpoint (e.g. a
      Xiaomi MiMo Token Plan host, ``https://token-plan-*.xiaomimimo.com/v1``)
      alive when they merely re-pick a model under the same provider — picking
      a model previously wiped it, forcing the registry default and breaking
      Token Plan keys.

    The runtime resolver reads ``model.base_url`` from config (it ignores
    ``OPENAI_BASE_URL``) and only honors it when the configured provider matches
    and the pool entry is on the registry default, so preserving it here is what
    lets the override actually route. The hardcoded ``context_length`` override
    is always dropped since the new model may have a different context window.

    Returns the same dict (coerced to a fresh dict if the input wasn't one) so
    callers can assign it straight back onto the model config.
    """
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    prev_provider = str(model_cfg.get("provider") or "").strip().lower()
    new_provider = provider.strip().lower()
    model_cfg["provider"] = provider
    model_cfg["default"] = model
    if base_url.strip():
        model_cfg["base_url"] = base_url.strip()
    elif model_cfg.get("base_url") and new_provider != prev_provider:
        # Switching providers: the old URL belonged to the old provider, drop
        # it so the new provider's default endpoint is used. Same-provider
        # re-assignment keeps the user's configured base_url intact.
        model_cfg["base_url"] = ""
    # The endpoint key follows the same lifecycle as base_url: an explicit key
    # is always persisted; an existing key is dropped only when switching to a
    # different provider (it belonged to the old endpoint), and preserved on a
    # same-provider re-pick so re-selecting a model doesn't wipe the key.
    if api_key.strip():
        model_cfg["api_key"] = api_key.strip()
        model_cfg.pop("api", None)
    elif model_cfg.get("api_key") and new_provider != prev_provider:
        clear_model_endpoint_credentials(model_cfg, clear_api_mode=False)
    if new_provider != prev_provider:
        clear_model_endpoint_credentials(model_cfg, clear_api_key=False)
    model_cfg.pop("context_length", None)
    return model_cfg


_GATEWAY_HEALTH_URL = os.getenv("GATEWAY_HEALTH_URL")
try:
    _GATEWAY_HEALTH_TIMEOUT = float(os.getenv("GATEWAY_HEALTH_TIMEOUT", "3"))
except (ValueError, TypeError):
    _log.warning(
        "Invalid GATEWAY_HEALTH_TIMEOUT value %r — using default 3.0s",
        os.getenv("GATEWAY_HEALTH_TIMEOUT"),
    )
    _GATEWAY_HEALTH_TIMEOUT = 3.0

# DEPRECATED (scheduled for removal): GATEWAY_HEALTH_URL / GATEWAY_HEALTH_TIMEOUT.
# Cross-container / cross-host gateway liveness detection will be folded into a
# first-class dashboard config key so it's no longer Docker-adjacent lore buried
# in env vars.  The env vars still work for now so existing Compose deployments
# don't break.  Do not add new callers — wire new uses through the planned
# config surface.


def _probe_gateway_health() -> tuple[bool, dict | None]:
    """Probe the gateway via its HTTP health endpoint (cross-container).

    .. deprecated::
        Driven by the deprecated ``GATEWAY_HEALTH_URL`` /
        ``GATEWAY_HEALTH_TIMEOUT`` env vars.  Scheduled for removal alongside
        a move to a first-class dashboard config key.  See
        :data:`_GATEWAY_HEALTH_URL` for context.

    Uses ``/health/detailed`` first (returns full state), falling back to
    the simpler ``/health`` endpoint.  Returns ``(is_alive, body_dict)``.

    Accepts any of these as ``GATEWAY_HEALTH_URL``:
    - ``http://gateway:8642``                (base URL — recommended)
    - ``http://gateway:8642/health``         (explicit health path)
    - ``http://gateway:8642/health/detailed`` (explicit detailed path)

    This is a **blocking** call — run via ``run_in_executor`` from async code.
    """
    if not _GATEWAY_HEALTH_URL:
        return False, None

    # Normalise to base URL so we always probe the right paths regardless of
    # whether the user included /health or /health/detailed in the env var.
    base = _GATEWAY_HEALTH_URL.rstrip("/")
    if base.endswith("/health/detailed"):
        base = base[: -len("/health/detailed")]
    elif base.endswith("/health"):
        base = base[: -len("/health")]

    for path in (f"{base}/health/detailed", f"{base}/health"):
        try:
            req = urllib.request.Request(path, method="GET")
            with urllib.request.urlopen(req, timeout=_GATEWAY_HEALTH_TIMEOUT) as resp:
                if resp.status == 200:
                    body = json.loads(resp.read())
                    return True, body
        except Exception:
            continue
    return False, None


# Image MIME types this endpoint will serve. Extension-allowlisted so an
# authenticated caller can't pull non-image files through it.
_MEDIA_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".bmp": "image/bmp",
    ".ico": "image/x-icon",
}
_MEDIA_MAX_BYTES = 25 * 1024 * 1024
_MANAGED_FILES_ROOT_ENV = "HERMES_DASHBOARD_FILES_ROOT"
_MANAGED_FILE_MAX_BYTES = 100 * 1024 * 1024
_HOSTED_MANAGED_FILES_ROOT = Path("/opt/data")


@dataclass(frozen=True)
class ManagedFilesPolicy:
    default_path: Path
    locked_root: Path | None
    can_change_path: bool


_FS_READDIR_HIDDEN = {
    ".git",
    ".hg",
    ".svn",
    ".cache",
    ".next",
    ".turbo",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
}
_FS_DATA_URL_MAX_BYTES = 16 * 1024 * 1024
_FS_TEXT_SOURCE_MAX_BYTES = 64 * 1024 * 1024
_FS_TEXT_PREVIEW_MAX_BYTES = 512 * 1024
# Upper bound for the in-app spot editor's save. The editor only opens
# non-truncated text (<= the preview cap), so this is a safety ceiling against
# a pasted-in megablob, not the expected payload size.
_FS_TEXT_WRITE_MAX_BYTES = 8 * 1024 * 1024
_FS_PREVIEW_LANGUAGE_BY_EXT = {
    ".c": "c",
    ".conf": "ini",
    ".cpp": "cpp",
    ".css": "css",
    ".csv": "csv",
    ".go": "go",
    ".graphql": "graphql",
    ".h": "c",
    ".hpp": "cpp",
    ".html": "html",
    ".java": "java",
    ".js": "javascript",
    ".json": "json",
    ".jsx": "jsx",
    ".kt": "kotlin",
    ".lua": "lua",
    ".md": "markdown",
    ".mjs": "javascript",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".sh": "shell",
    ".sql": "sql",
    ".svg": "xml",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".txt": "text",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".zsh": "shell",
}
_FS_MIME_TYPES = {
    ".avi": "video/x-msvideo",
    ".bmp": "image/bmp",
    ".flac": "audio/flac",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".m4a": "audio/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp3": "audio/mpeg",
    ".mp4": "video/mp4",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg; codecs=opus",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".wav": "audio/wav",
    ".webm": "video/webm",
    ".webp": "image/webp",
}


def _fs_path(raw_path: str) -> Path:
    raw = str(raw_path or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Path is required")
    if "\0" in raw:
        raise HTTPException(status_code=400, detail="Invalid path")
    try:
        if raw.lower().startswith("file:"):
            parsed = urllib.parse.urlparse(raw)
            if parsed.netloc and parsed.netloc not in {"", "localhost"}:
                raise ValueError
            raw = urllib.request.url2pathname(parsed.path)
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid path")


def _fs_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _FS_MIME_TYPES:
        return _FS_MIME_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _fs_looks_binary(data: bytes) -> bool:
    if not data:
        return False
    if b"\0" in data:
        return True
    suspicious = sum(1 for byte in data if byte < 32 and byte not in {9, 10, 13})
    return suspicious / len(data) > 0.12


def _fs_regular_file(path: Path) -> tuple[Path, os.stat_result]:
    target = _fs_path(str(path))
    try:
        st = target.stat()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except NotADirectoryError:
        raise HTTPException(status_code=404, detail="File not found")
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not readable")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "Invalid path")
    if stat.S_ISDIR(st.st_mode):
        raise HTTPException(status_code=400, detail="Path points to a directory")
    if not stat.S_ISREG(st.st_mode):
        raise HTTPException(status_code=400, detail="Only regular files can be read")
    return target, st


def _fs_find_git_root(start: Path) -> str | None:
    directory = start
    for _ in range(50):
        try:
            if (directory / ".git").exists():
                return str(directory)
        except OSError:
            return None
        parent = directory.parent
        if parent == directory:
            return None
        directory = parent
    return None


def _fs_default_cwd() -> str:
    cfg_terminal = load_config().get("terminal") or {}
    raw = str(cfg_terminal.get("cwd") or os.environ.get("TERMINAL_CWD") or "").strip()
    if raw and raw not in {".", "auto", "cwd"}:
        try:
            candidate = Path(raw).expanduser().resolve(strict=False)
            if candidate.is_dir():
                return str(candidate)
        except (OSError, RuntimeError):
            pass
    return str(Path.cwd())


def _fs_git_branch(cwd: str) -> str:
    try:
        run_kwargs: Dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "timeout": 2,
            "check": False,
        }
        if sys.platform == "win32":
            run_kwargs["creationflags"] = windows_hide_flags()
        result = subprocess.run(
            ["git", "-C", cwd, "branch", "--show-current"],
            **run_kwargs,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _media_serve_roots() -> list[Path]:
    """Directories ``GET /api/media`` is allowed to read from.

    Confined to where the agent and attach pipeline actually write media on the
    gateway host — its images dir and cache subtree. This stops an authenticated
    client from reading image-extension files anywhere on disk (e.g. a renamed
    key or a screenshot outside the cache) merely because the suffix passes the
    allowlist.
    """
    home = get_hermes_home()
    roots = [home / "images", home / "screenshots", home / "cache"]
    out: list[Path] = []
    for root in roots:
        try:
            out.append(root.resolve())
        except (OSError, RuntimeError):
            continue
    return out


@app.get("/api/media")
async def get_media(path: str):
    """Return a gateway-local image file as a base64 data URL.

    Lets remote clients (the desktop app over the network, or the web dashboard
    in a browser) display images the agent wrote to *this* machine's filesystem
    — they can't read the gateway's local disk directly.

    Auth-gated by the session token like every other /api route. Restricted to
    an image-extension allowlist, a size cap, AND the gateway's own media roots
    (resolved, symlink-safe) so it can't be used to read arbitrary files.
    """
    try:
        target = Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid path")

    if target.suffix.lower() not in _MEDIA_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported media type")

    roots = _media_serve_roots()
    if not any(target == root or root in target.parents for root in roots):
        raise HTTPException(status_code=403, detail="Path outside media roots")

    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if target.stat().st_size > _MEDIA_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    encoded = base64.b64encode(target.read_bytes()).decode("ascii")
    return {"data_url": f"data:{_MEDIA_CONTENT_TYPES[target.suffix.lower()]};base64,{encoded}"}


def _canonical_path(path: Path, *, require_exists: bool = False) -> Path:
    try:
        return path.expanduser().resolve(strict=require_exists)
    except FileNotFoundError:
        if require_exists:
            raise HTTPException(status_code=404, detail="Path not found")
        raise
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid path")


def _ensure_managed_root(raw_path: str | Path) -> Path:
    root = Path(raw_path).expanduser()
    try:
        root.mkdir(parents=True, exist_ok=True)
        resolved = root.resolve()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=f"Managed files root is unavailable: {exc}")
    if not resolved.is_dir():
        raise HTTPException(status_code=500, detail="Managed files root is not a directory")
    return resolved


def _path_is_under(root: Path, target: Path) -> bool:
    return target == root or root in target.parents


def _path_text(raw_path: str | None) -> str:
    text = str(raw_path or "").strip()
    if "\x00" in text:
        raise HTTPException(status_code=400, detail="Invalid path")
    return text


def _local_dashboard_request(request: Request) -> bool:
    if getattr(request.app.state, "auth_required", False):
        return False
    host = (request.url.hostname or "").lower()
    client_host = (request.client.host if request.client else "").lower()
    local_hosts = {"", "localhost", "127.0.0.1", "::1", "testserver", "testclient"}
    return host in local_hosts or client_host in local_hosts


def _default_hermes_root_is_opt_data() -> bool:
    raw = os.environ.get("HERMES_HOME", "").strip()
    if not raw:
        return False
    try:
        from hermes_constants import get_default_hermes_root

        root = get_default_hermes_root().expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        root = Path(raw).expanduser().resolve(strict=False)
    return root == _HOSTED_MANAGED_FILES_ROOT


def _dashboard_local_update_managed_externally() -> bool:
    """Return true when the dashboard should not offer ``hermes update``.

    Containerized dashboards are updated by the outer launcher/image, not by an
    in-browser local update action. Keep this dashboard capability separate
    from install-method detection: manual git/pip installs inside containers can
    still behave like their actual install method in the CLI.

    However, when the install method is ``git`` (a bind-mounted checkout inside
    a container — e.g. the hermes-webui image sharing the Hermes source tree),
    the dashboard's ``hermes update`` button is the correct update path and
    should not be suppressed. Other containerized install methods remain
    externally managed unless their apply path is proven safe inside the
    running container filesystem.
    """
    if _default_hermes_root_is_opt_data():
        return True
    try:
        from hermes_constants import is_container

        if not is_container():
            return False
    except Exception:
        return False
    # We are inside a container, but the install may still be self-managed.
    # If the install method is git, the dashboard update button works against
    # the mounted checkout and should be offered. Keep pip blocked inside
    # containers: its apply path mutates the running container filesystem and
    # is not the bind-mounted checkout case this gate is meant to recover.
    try:
        method = detect_install_method(PROJECT_ROOT)
        if method == "git":
            return False
    except Exception:
        pass
    return True


def _managed_files_policy(request: Request, *, create_root: bool = True) -> ManagedFilesPolicy:
    raw_forced_root = os.environ.get(_MANAGED_FILES_ROOT_ENV, "").strip()
    if raw_forced_root:
        root = _ensure_managed_root(raw_forced_root) if create_root else _canonical_path(Path(raw_forced_root))
        return ManagedFilesPolicy(default_path=root, locked_root=root, can_change_path=False)

    # Remote/OAuth access does not imply a hosted container. Users can expose a
    # local dashboard through the auth gate (for example a macOS launchd install)
    # and still expect the Files page to browse their local home directory. Lock
    # to /opt/data only when the installation's Hermes root is actually /opt/data
    # (the container/hosted layout) or when HERMES_DASHBOARD_FILES_ROOT is set.
    if _default_hermes_root_is_opt_data():
        root = _ensure_managed_root(_HOSTED_MANAGED_FILES_ROOT) if create_root else _HOSTED_MANAGED_FILES_ROOT
        return ManagedFilesPolicy(default_path=root, locked_root=root, can_change_path=False)

    home = _canonical_path(Path.home())
    return ManagedFilesPolicy(default_path=home, locked_root=None, can_change_path=True)


def _resolve_managed_path(
    raw_path: str | None,
    request: Request,
    *,
    for_write: bool = False,
) -> tuple[ManagedFilesPolicy, Path, str]:
    policy = _managed_files_policy(request)
    text = _path_text(raw_path)
    root = policy.locked_root

    if root is not None and (not text or text in {".", "/"}):
        candidate = root
    elif not text:
        candidate = policy.default_path
    else:
        candidate = Path(text).expanduser()
        if root is not None and not candidate.is_absolute():
            if any(part == ".." for part in candidate.parts):
                raise HTTPException(status_code=400, detail="Path cannot contain '..'")
            candidate = root / candidate
        elif not candidate.is_absolute():
            raise HTTPException(status_code=400, detail="Path must be absolute")

    if ".." in candidate.parts:
        raise HTTPException(status_code=400, detail="Path cannot contain '..'")

    if for_write and not candidate.exists():
        parent = _canonical_path(candidate.parent)
        resolved = parent / candidate.name
    else:
        resolved = _canonical_path(candidate, require_exists=not for_write)

    if root is not None and not _path_is_under(root, resolved):
        raise HTTPException(status_code=403, detail="Path outside managed files root")

    return policy, resolved, str(resolved)


def _managed_response_meta(policy: ManagedFilesPolicy) -> Dict[str, Any]:
    locked_root = str(policy.locked_root) if policy.locked_root is not None else None
    return {
        "root": locked_root,
        "locked_root": locked_root,
        "can_change_path": policy.can_change_path,
    }


def _managed_file_entry(policy: ManagedFilesPolicy, target: Path) -> Dict[str, Any]:
    try:
        resolved = target.resolve()
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid path")
    if policy.locked_root is not None and not _path_is_under(policy.locked_root, resolved):
        raise HTTPException(status_code=403, detail="Path outside managed files root")

    try:
        st = resolved.stat()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not stat path: {exc}")

    is_dir = resolved.is_dir()
    mime_type = None if is_dir else (mimetypes.guess_type(resolved.name)[0] or "application/octet-stream")
    return {
        "name": target.name or resolved.name or str(resolved),
        "path": str(resolved),
        "is_directory": is_dir,
        "size": None if is_dir else st.st_size,
        "mtime": st.st_mtime,
        "mime_type": mime_type,
    }


def _decode_data_url(data_url: str) -> tuple[bytes, str]:
    text = (data_url or "").strip()
    if not text.startswith("data:") or "," not in text:
        raise HTTPException(status_code=400, detail="Upload payload must be a data URL")
    header, encoded = text.split(",", 1)
    mime_type = header[5:].split(";", 1)[0] or "application/octet-stream"
    if ";base64" not in header:
        raise HTTPException(status_code=400, detail="Upload payload must be base64 encoded")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Upload payload is not valid base64")
    if len(data) > _MANAGED_FILE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File is too large")
    return data, mime_type


@app.get("/api/files")
async def list_managed_files(request: Request, path: Optional[str] = None):
    policy, target, display_path = _resolve_managed_path(path, request)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")

    try:
        entries = [_managed_file_entry(policy, child) for child in target.iterdir()]
    except PermissionError:
        raise HTTPException(status_code=403, detail="Directory is not readable")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read directory: {exc}")

    entries.sort(key=lambda item: (not item["is_directory"], str(item["name"]).lower()))
    locked_root = policy.locked_root
    parent = None
    if target.parent != target and (locked_root is None or target != locked_root):
        parent = str(target.parent)
    return {
        "path": display_path,
        "parent": parent,
        "entries": entries,
        **_managed_response_meta(policy),
    }


@app.get("/api/files/read")
async def read_managed_file(request: Request, path: str):
    policy, target, display_path = _resolve_managed_path(path, request)
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")

    try:
        size = target.stat().st_size
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not stat file: {exc}")
    if size > _MANAGED_FILE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File is too large")

    mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    try:
        encoded = base64.b64encode(target.read_bytes()).decode("ascii")
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not readable")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read file: {exc}")

    return {
        "name": target.name,
        "path": display_path,
        "size": size,
        "mime_type": mime_type,
        "data_url": f"data:{mime_type};base64,{encoded}",
        **_managed_response_meta(policy),
    }


@app.get("/api/files/download")
async def download_managed_file(request: Request, path: str):
    """Stream a managed file as an attachment download.

    Remote clients (desktop app, browser dashboard) open agent-written files
    that live on *this* gateway's disk, not theirs. Auth-gated like every other
    managed-files route — ``auth_middleware`` additionally accepts the session
    token as a ``?token=`` query param here so a shell/browser-opened download
    (which can't set the session header) still authenticates. See ``/api/pty``
    for the same query-token precedent.
    """
    policy, target, _display_path = _resolve_managed_path(path, request)
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")

    try:
        size = target.stat().st_size
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not stat file: {exc}")
    if size > _MANAGED_FILE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File is too large")

    mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"

    return FileResponse(
        path=str(target),
        media_type=mime_type,
        filename=target.name,
        content_disposition_type="attachment",
    )


@app.post("/api/files/upload")
async def upload_managed_file(payload: ManagedFileUpload, request: Request):
    policy, target, display_path = _resolve_managed_path(payload.path, request, for_write=True)
    if target.exists() and target.is_dir():
        raise HTTPException(status_code=409, detail="A directory already exists at that path")
    if target.exists() and not payload.overwrite:
        raise HTTPException(status_code=409, detail="File already exists")

    data, _mime_type = _decode_data_url(payload.data_url)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not writable")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not write file: {exc}")

    return {
        "ok": True,
        "entry": _managed_file_entry(policy, target),
        "path": display_path,
        **_managed_response_meta(policy),
    }


# Stream uploads to disk in fixed-size chunks. The legacy JSON endpoint above
# buffers the whole file as a base64 data URL in a JSON body, which (a) inflates
# the payload ~33%, (b) holds the entire file (plus its decoded copy) in memory,
# and (c) reliably trips upstream proxy body-size/timeout limits with a 502 on
# large backup archives (NS-501). This multipart endpoint reads the request body
# in 1 MiB chunks straight to a temp file, enforces the size cap as it goes, and
# atomically renames into place — constant memory, no base64 inflation.
_UPLOAD_CHUNK_BYTES = 1024 * 1024


@app.post("/api/files/upload-stream")
async def upload_managed_file_stream(
    request: Request,
    file: UploadFile = File(...),
    path: str = Form(...),
    overwrite: bool = Form(True),
):
    policy, target, display_path = _resolve_managed_path(path, request, for_write=True)
    if target.exists() and target.is_dir():
        raise HTTPException(status_code=409, detail="A directory already exists at that path")
    if target.exists() and not overwrite:
        raise HTTPException(status_code=409, detail="File already exists")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not writable")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not create parent directory: {exc}")

    # Write to a sibling temp file first so a partial/aborted upload never
    # clobbers an existing file, then atomically rename into place.
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".upload", dir=str(target.parent)
    )
    tmp_path = Path(tmp_name)
    total = 0
    renamed = False
    try:
        with os.fdopen(tmp_fd, "wb") as out:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MANAGED_FILE_MAX_BYTES:
                    raise HTTPException(status_code=413, detail="File is too large")
                out.write(chunk)
        os.replace(tmp_path, target)
        renamed = True
    except HTTPException:
        raise
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not writable")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not write file: {exc}")
    finally:
        # Clean up the temp file on every non-success exit, including
        # BaseException paths the `except` clauses above don't catch — most
        # importantly asyncio.CancelledError when a browser aborts a large
        # upload mid-stream (the exact NS-501 scenario). os.replace clears
        # tmp_path on success, so only unlink when the rename didn't happen.
        if not renamed:
            tmp_path.unlink(missing_ok=True)
        await file.close()

    return {
        "ok": True,
        "entry": _managed_file_entry(policy, target),
        "path": display_path,
        **_managed_response_meta(policy),
    }


@app.post("/api/files/mkdir")
async def create_managed_directory(payload: ManagedDirectoryCreate, request: Request):
    policy, target, display_path = _resolve_managed_path(payload.path, request, for_write=True)
    if target.exists() and not target.is_dir():
        raise HTTPException(status_code=409, detail="A file already exists at that path")

    try:
        target.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise HTTPException(status_code=403, detail="Directory is not writable")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not create directory: {exc}")

    return {
        "ok": True,
        "entry": _managed_file_entry(policy, target),
        "path": display_path,
        **_managed_response_meta(policy),
    }


@app.delete("/api/files")
async def delete_managed_file(payload: ManagedFileDelete, request: Request):
    policy, target, display_path = _resolve_managed_path(payload.path, request)
    if policy.locked_root is not None and target == policy.locked_root:
        raise HTTPException(status_code=400, detail="Cannot delete the managed files root")
    if target.parent == target:
        raise HTTPException(status_code=400, detail="Cannot delete the filesystem root")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")

    try:
        if target.is_dir():
            if payload.recursive:
                shutil.rmtree(target)
            else:
                target.rmdir()
        else:
            target.unlink()
    except OSError as exc:
        status_code = 409 if target.is_dir() and not payload.recursive else 500
        raise HTTPException(status_code=status_code, detail=f"Could not delete path: {exc}")

    return {"ok": True, "path": display_path, **_managed_response_meta(policy)}


@app.get("/api/fs/list")
async def fs_list(path: str):
    target = _fs_path(path)
    try:
        entries = []
        with os.scandir(target) as scan:
            for entry in scan:
                if entry.name in _FS_READDIR_HIDDEN:
                    continue
                entries.append({
                    "name": entry.name,
                    "path": str(target / entry.name),
                    "isDirectory": entry.is_dir(follow_symlinks=False),
                })
        entries.sort(key=lambda item: (not item["isDirectory"], item["name"].lower(), item["name"]))
        return {"entries": entries}
    except FileNotFoundError:
        return {"entries": [], "error": "ENOENT"}
    except NotADirectoryError:
        return {"entries": [], "error": "ENOTDIR"}
    except PermissionError:
        return {"entries": [], "error": "EACCES"}
    except OSError as exc:
        return {"entries": [], "error": getattr(exc, "strerror", None) or "read-error"}


@app.get("/api/fs/read-text")
async def fs_read_text(path: str):
    target, st = _fs_regular_file(_fs_path(path))
    if st.st_size > _FS_TEXT_SOURCE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    bytes_to_read = min(st.st_size, _FS_TEXT_PREVIEW_MAX_BYTES)
    try:
        with target.open("rb") as handle:
            data = handle.read(bytes_to_read)
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not readable")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "File read failed")
    return {
        "binary": _fs_looks_binary(data[:4096]),
        "byteSize": st.st_size,
        "language": _FS_PREVIEW_LANGUAGE_BY_EXT.get(target.suffix.lower(), "text"),
        "mimeType": _fs_mime_type(target),
        "path": str(target),
        "text": data.decode("utf-8", errors="replace"),
        "truncated": st.st_size > _FS_TEXT_PREVIEW_MAX_BYTES,
    }


class FsWriteText(BaseModel):
    path: str
    content: str


@app.post("/api/fs/write-text")
async def fs_write_text(payload: FsWriteText):
    """Overwrite (or create) a UTF-8 text file for the in-app spot editor.

    Mirrors the local Electron ``hermes:fs:writeText`` hardening: the path is
    resolved + validated by ``_fs_path``, the parent directory must already
    exist (we never build directory trees), only regular files may be replaced,
    and the payload is size-capped. The write is staged to a sibling temp file
    and ``os.replace``-d into place so a crash mid-write can't truncate the
    original. Stale-on-disk detection is the client's job (re-read before save),
    so both transports behave identically.
    """
    target = _fs_path(payload.path)
    text = payload.content or ""
    if len(text.encode("utf-8")) > _FS_TEXT_WRITE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Content too large")

    try:
        st: Optional[os.stat_result] = target.stat()
    except FileNotFoundError:
        st = None
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not writable")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "Invalid path")

    if st is not None and stat.S_ISDIR(st.st_mode):
        raise HTTPException(status_code=400, detail="Path points to a directory")
    if st is not None and not stat.S_ISREG(st.st_mode):
        raise HTTPException(status_code=400, detail="Only regular files can be written")
    if not target.parent.is_dir():
        raise HTTPException(status_code=400, detail="Parent directory does not exist")

    tmp = target.with_name(f".{target.name}.hermes-tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    except PermissionError:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=403, detail="File is not writable")
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not write file: {exc}")

    return {"ok": True, "path": str(target), "byteSize": len(text.encode("utf-8"))}


@app.get("/api/fs/read-data-url")
async def fs_read_data_url(path: str):
    target, st = _fs_regular_file(_fs_path(path))
    if st.st_size > _FS_DATA_URL_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    try:
        encoded = base64.b64encode(target.read_bytes()).decode("ascii")
    except PermissionError:
        raise HTTPException(status_code=403, detail="File is not readable")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "File read failed")
    return {"dataUrl": f"data:{_fs_mime_type(target)};base64,{encoded}"}


@app.get("/api/fs/git-root")
async def fs_git_root(path: str):
    target = _fs_path(path)
    try:
        st = target.stat()
        start = target if stat.S_ISDIR(st.st_mode) else target.parent
    except OSError:
        start = target
    return {"root": _fs_find_git_root(start)}


@app.get("/api/fs/default-cwd")
async def fs_default_cwd():
    cwd = _fs_default_cwd()
    return {"cwd": cwd, "branch": _fs_git_branch(cwd)}


# ---------------------------------------------------------------------------
# Git ops — the remote half of the desktop coding rail + review pane.
#
# The desktop runs these as Electron-local git on the user's machine; over a
# remote gateway that's the wrong filesystem, so we mirror them here (same auth
# gate + path hardening as /api/fs). Logic lives in ``hermes_cli.web_git``;
# these are thin, executor-offloaded wrappers (git/gh can block).
# ---------------------------------------------------------------------------

from hermes_cli import web_git as _web_git  # noqa: E402


async def _git_op(fn, *args):
    """Run a (blocking) git op off the event loop; map a failed mutation to 400."""
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, fn, *args)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "git operation failed")


def _git_path(path: str) -> str:
    return str(_fs_path(path))


class GitPathBody(BaseModel):
    path: str


class GitFileBody(BaseModel):
    path: str
    file: Optional[str] = None


class GitCommitBody(BaseModel):
    path: str
    message: str
    push: bool = False


class GitWorktreeAddBody(BaseModel):
    path: str
    name: Optional[str] = None
    branch: Optional[str] = None
    base: Optional[str] = None
    existingBranch: Optional[str] = None


class GitWorktreeRemoveBody(BaseModel):
    path: str
    worktreePath: str
    force: bool = False


class GitBranchSwitchBody(BaseModel):
    path: str
    branch: str


@app.get("/api/git/status")
async def git_status_route(path: str):
    return await _git_op(_web_git.repo_status, _git_path(path))


@app.get("/api/git/worktrees")
async def git_worktrees_route(path: str):
    return {"worktrees": await _git_op(_web_git.worktree_list, _git_path(path))}


@app.get("/api/git/branches")
async def git_branches_route(path: str):
    return {"branches": await _git_op(_web_git.branch_list, _git_path(path))}


@app.get("/api/git/review/list")
async def git_review_list_route(path: str, scope: str = "uncommitted", base: Optional[str] = None):
    return await _git_op(_web_git.review_list, _git_path(path), scope, base)


@app.get("/api/git/review/diff")
async def git_review_diff_route(
    path: str, file: str, scope: str = "uncommitted", base: Optional[str] = None, staged: bool = False
):
    return {"diff": await _git_op(_web_git.review_diff, _git_path(path), file, scope, base, staged)}


@app.get("/api/git/file-diff")
async def git_file_diff_route(path: str, file: str):
    return {"diff": await _git_op(_web_git.file_diff_vs_head, _git_path(path), file)}


@app.get("/api/git/review/commit-context")
async def git_commit_context_route(path: str):
    return await _git_op(_web_git.review_commit_context, _git_path(path))


@app.get("/api/git/review/rev-parse")
async def git_rev_parse_route(path: str, ref: Optional[str] = None):
    return {"sha": await _git_op(_web_git.review_rev_parse, _git_path(path), ref)}


@app.get("/api/git/review/ship-info")
async def git_ship_info_route(path: str):
    return await _git_op(_web_git.review_ship_info, _git_path(path))


@app.post("/api/git/review/stage")
async def git_stage_route(body: GitFileBody):
    return await _git_op(_web_git.review_stage, _git_path(body.path), body.file)


@app.post("/api/git/review/unstage")
async def git_unstage_route(body: GitFileBody):
    return await _git_op(_web_git.review_unstage, _git_path(body.path), body.file)


@app.post("/api/git/review/revert")
async def git_revert_route(body: GitFileBody):
    return await _git_op(_web_git.review_revert, _git_path(body.path), body.file)


@app.post("/api/git/review/commit")
async def git_commit_route(body: GitCommitBody):
    return await _git_op(_web_git.review_commit, _git_path(body.path), body.message, body.push)


@app.post("/api/git/review/push")
async def git_push_route(body: GitPathBody):
    return await _git_op(_web_git.review_push, _git_path(body.path))


@app.post("/api/git/review/create-pr")
async def git_create_pr_route(body: GitPathBody):
    return await _git_op(_web_git.review_create_pr, _git_path(body.path))


@app.post("/api/git/worktree/add")
async def git_worktree_add_route(body: GitWorktreeAddBody):
    options = {
        key: value
        for key, value in {
            "name": body.name,
            "branch": body.branch,
            "base": body.base,
            "existingBranch": body.existingBranch,
        }.items()
        if value
    }
    return await _git_op(_web_git.worktree_add, _git_path(body.path), options)


@app.post("/api/git/worktree/remove")
async def git_worktree_remove_route(body: GitWorktreeRemoveBody):
    return await _git_op(
        _web_git.worktree_remove, _git_path(body.path), _git_path(body.worktreePath), body.force
    )


@app.post("/api/git/branch/switch")
async def git_branch_switch_route(body: GitBranchSwitchBody):
    return await _git_op(_web_git.branch_switch, _git_path(body.path), body.branch)


@app.get("/api/status")
async def get_status(profile: Optional[str] = None):
    status_scope = None
    requested_profile = (profile or "").strip()
    # Plain /api/status stays the machine-level public liveness probe. The
    # dashboard adds ?profile= when its management switcher targets another
    # profile, so its gateway badge reflects the selected profile.
    #
    # Use the config-only (contextvar) scope, NOT _profile_scope: this handler
    # awaits the remote-health probe, and _profile_scope swaps process-global
    # skills-module attributes that a concurrent request would cross-restore
    # across that await. Status only resolves get_hermes_home() at call time
    # (config/env/gateway state), which the task-local contextvar covers.
    if requested_profile and requested_profile.lower() != "current":
        status_scope = _config_profile_scope(requested_profile)
        status_scope.__enter__()

    try:
        current_ver, latest_ver = check_config_version()
        # --- Gateway liveness detection ---
        # Try local PID check first (same-host).  If that fails and a remote
        # GATEWAY_HEALTH_URL is configured, probe the gateway over HTTP so the
        # dashboard works when the gateway runs in a separate container.
        gateway_pid = get_running_pid()
        gateway_running = gateway_pid is not None
        remote_health_body: dict | None = None

        if not gateway_running and _GATEWAY_HEALTH_URL:
            loop = asyncio.get_running_loop()
            alive, remote_health_body = await loop.run_in_executor(
                None, _probe_gateway_health
            )
            if alive:
                gateway_running = True
                # PID from the remote container (display only — not locally valid)
                if remote_health_body:
                    gateway_pid = remote_health_body.get("pid")

        gateway_state = None
        gateway_platforms: dict = {}
        gateway_exit_reason = None
        gateway_updated_at = None
        configured_gateway_platforms: set[str] | None = None
        try:
            from gateway.config import load_gateway_config

            gateway_config = load_gateway_config()
            configured_gateway_platforms = {
                platform.value for platform in gateway_config.get_connected_platforms()
            }
        except Exception:
            configured_gateway_platforms = None

        # Prefer the detailed health endpoint response (has full state) when the
        # local runtime status file is absent or stale (cross-container).
        local_runtime = read_runtime_status()
        runtime = local_runtime
        if runtime is None and remote_health_body and remote_health_body.get("gateway_state"):
            runtime = remote_health_body
        # The runtime-status PID fallback validates liveness with a local
        # os.kill() probe, so it must only run against the LOCAL status file —
        # never the remote health body, whose PID belongs to another host and
        # is display-only. (Running os.kill on a remote PID is both wrong and
        # trips the test live-system guard.)
        if not gateway_running and local_runtime is not None:
            runtime_pid = get_runtime_status_running_pid(local_runtime)
            if runtime_pid is not None:
                gateway_running = True
                gateway_pid = runtime_pid

        if runtime:
            gateway_state = runtime.get("gateway_state")
            gateway_platforms = runtime.get("platforms") or {}
            if configured_gateway_platforms is not None:
                gateway_platforms = {
                    key: value
                    for key, value in gateway_platforms.items()
                    if key in configured_gateway_platforms
                }
            gateway_exit_reason = runtime.get("exit_reason")
            gateway_updated_at = runtime.get("updated_at")
            if not gateway_running:
                gateway_state = gateway_state if gateway_state in {"stopped", "startup_failed"} else "stopped"
                gateway_platforms = {}
            elif gateway_running and remote_health_body is not None:
                # The health probe confirmed the gateway is alive, but the local
                # runtime status file may be stale (cross-container).  Override
                # stopped/None state so the dashboard shows the correct badge.
                if gateway_state in {None, "stopped"}:
                    gateway_state = "running"

        # If there was no runtime info at all but the health probe confirmed alive,
        # ensure we still report the gateway as running (no shared volume scenario).
        if gateway_running and gateway_state is None and remote_health_body is not None:
            gateway_state = "running"

        active_sessions = 0
        try:
            from hermes_state import SessionDB
            db = SessionDB()
            try:
                sessions = db.list_sessions_rich(limit=50)
                now = time.time()
                active_sessions = sum(
                    1 for s in sessions
                    if s.get("ended_at") is None
                    and (now - s.get("last_active", s.get("started_at", 0))) < 300
                )
            finally:
                db.close()
        except Exception:
            pass

        # Busy/drainable readout (NAS lifecycle-safety gate).  active_agents is
        # the in-flight gateway-turn count the gateway now persists at every
        # turn boundary; gateway_busy/gateway_drainable are derived from it +
        # liveness via the single shared contract in gateway.status.  Liveness
        # keys off gateway_running (a live PID/health probe), NEVER
        # gateway_updated_at — a healthy idle gateway never advances that.
        active_agents = parse_active_agents((runtime or {}).get("active_agents", 0))
        gateway_busy = derive_gateway_busy(
            gateway_running=gateway_running,
            gateway_state=gateway_state,
            active_agents=active_agents,
        )
        gateway_drainable = derive_gateway_drainable(
            gateway_running=gateway_running,
            gateway_state=gateway_state,
        )
        # Resolved drain timeout (seconds) so NAS can size its poll deadline
        # without out-of-band knowledge.  Offload to a thread: on a cold
        # Windows install the first import of hermes_cli.gateway blocks the
        # asyncio event loop for 15-30s (.pyc compilation + Defender scans),
        # exceeding the desktop handshake's 15s socket timeout.  After the
        # first call the module is in sys.modules and run_in_executor returns
        # in microseconds.
        restart_drain_timeout = await asyncio.get_running_loop().run_in_executor(
            None, _resolve_restart_drain_timeout
        )

        # Dashboard auth gate (Phase 7): surface whether the gate is engaged
        # and which providers are registered so ``hermes status`` and the
        # SPA's StatusPage can show "OAuth gate ON via Nous Research" or
        # "loopback only — no auth gate" with no extra round trips.
        auth_required = bool(getattr(app.state, "auth_required", False))
        auth_providers: list[str] = []
        try:
            from hermes_cli.dashboard_auth import list_providers as _list_providers
            auth_providers = [p.name for p in _list_providers()]
        except Exception:
            # Module not importable yet (early startup) — leave as [].
            pass

        # Always-public liveness + auth-gate shape. Safe for external uptime
        # probes (NAS's wildcard-subdomain liveness probe), the SPA's pre-login
        # bootstrap, and anyone who can curl the host — i.e. exactly the audience
        # ``PUBLIC_API_PATHS`` documents this endpoint as serving.
        status = {
            "version": __version__,
            "release_date": __release_date__,
            "config_version": current_ver,
            "latest_config_version": latest_ver,
            "can_update_hermes": not _dashboard_local_update_managed_externally(),
            "gateway_running": gateway_running,
            "gateway_state": gateway_state,
            "gateway_platforms": gateway_platforms,
            "gateway_exit_reason": gateway_exit_reason,
            "gateway_updated_at": gateway_updated_at,
            "active_agents": active_agents,
            "gateway_busy": gateway_busy,
            "gateway_drainable": gateway_drainable,
            "restart_drain_timeout": restart_drain_timeout,
            "active_sessions": active_sessions,
            "auth_required": auth_required,
            "auth_providers": auth_providers,
        }

        # Absolute host paths, the gateway PID, and the internal gateway health
        # URL are deployment recon a liveness probe never needs. ``/api/status``
        # is in ``PUBLIC_API_PATHS`` so it bypasses dashboard auth; on a
        # network-exposed (gated) bind that means *any* unauthenticated caller
        # reaches it, and leaking host metadata there contradicts the allowlist's
        # own contract ("version, gateway state, active session count, and the
        # dashboard auth-gate shape. No bodies, no session content, no secrets").
        # Surface this detail only on a loopback / ``--insecure`` bind, where the
        # dashboard is local-only and the caller is already inside the trust
        # envelope — the same loopback/gated split ``should_require_auth`` draws.
        if not auth_required:
            status.update({
                "hermes_home": str(get_hermes_home()),
                "config_path": str(get_config_path()),
                "env_path": str(get_env_path()),
                "gateway_pid": gateway_pid,
                "gateway_health_url": _GATEWAY_HEALTH_URL,
            })

        return status
    finally:
        if status_scope is not None:
            status_scope.__exit__(*sys.exc_info())


_WINDOWS_11_MIN_BUILD = 22000


def _windows_build_number(version: str, platform_label: str) -> Optional[int]:
    """Extract the Windows NT build number from stdlib platform strings."""
    for value in (version or "", platform_label or ""):
        match = re.search(r"(?:^|[^\d])10\.0\.(\d{5,})(?:[^\d]|$)", value)
        if not match:
            continue
        try:
            return int(match.group(1))
        except ValueError:
            continue
    return None


def _display_system_platform(
    *,
    system: str,
    release: str,
    version: str,
    platform_label: str,
) -> Dict[str, str]:
    """Return host OS fields for display while preserving stdlib detail."""
    if system == "Windows" and release == "10":
        build = _windows_build_number(version, platform_label)
        if build is not None and build >= _WINDOWS_11_MIN_BUILD:
            platform_label = re.sub(
                r"^Windows-10(?=-)",
                "Windows-11",
                platform_label,
                count=1,
            )
            release = "11"

    return {
        "os": system,
        "os_release": release,
        "os_version": version,
        "platform": platform_label,
    }


@app.get("/api/system/stats")
async def get_system_stats():
    """Host + process system stats for the System page.

    OS / Python / host identity from stdlib; CPU / memory / disk / uptime from
    psutil when available, with graceful degradation when it isn't.  Read-only
    and non-sensitive (no env values, no paths beyond the hermes home root).
    """
    import platform as _platform

    info: Dict[str, Any] = {
        **_display_system_platform(
            system=_platform.system(),
            release=_platform.release(),
            version=_platform.version(),
            platform_label=_platform.platform(),
        ),
        "arch": _platform.machine(),
        "hostname": _platform.node(),
        "python_version": _platform.python_version(),
        "python_impl": _platform.python_implementation(),
        "hermes_version": __version__,
        "cpu_count": os.cpu_count(),
    }

    # psutil enriches the picture when present; everything below is optional.
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        info["memory"] = {
            "total": vm.total,
            "available": vm.available,
            "used": vm.used,
            "percent": vm.percent,
        }
        try:
            du = psutil.disk_usage(str(get_hermes_home()))
            info["disk"] = {
                "total": du.total,
                "used": du.used,
                "free": du.free,
                "percent": du.percent,
            }
        except Exception:
            pass
        try:
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            la = getattr(psutil, "getloadavg", None)
            if la:
                info["load_avg"] = list(la())
        except Exception:
            pass
        try:
            boot = psutil.boot_time()
            info["uptime_seconds"] = int(time.time() - boot)
        except Exception:
            pass
        try:
            proc = psutil.Process()
            info["process"] = {
                "pid": proc.pid,
                "rss": proc.memory_info().rss,
                "create_time": int(proc.create_time()),
                "num_threads": proc.num_threads(),
            }
        except Exception:
            pass
        info["psutil"] = True
    except Exception:
        info["psutil"] = False
        # stdlib-only fallbacks for load average + uptime where the kernel
        # exposes them.
        try:
            info["load_avg"] = list(os.getloadavg())
        except (OSError, AttributeError):
            pass

    return info


# ---------------------------------------------------------------------------
# Curator endpoints — background skill-maintenance status + controls.
#
# The curator periodically reviews skills (archive stale, prune, pin).  The
# dashboard surfaces its state and the pause/resume/run-now controls that
# `hermes curator` exposes.
# ---------------------------------------------------------------------------


@app.get("/api/curator")
async def get_curator_status():
    try:
        from agent import curator
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Curator unavailable: {exc}")
    try:
        state = curator.load_state()
    except Exception:
        state = {}
    return {
        "enabled": _safe_call(curator, "is_enabled", True),
        "paused": _safe_call(curator, "is_paused", False),
        "interval_hours": _safe_call(curator, "get_interval_hours", None),
        "last_run_at": state.get("last_run_at"),
        "min_idle_hours": _safe_call(curator, "get_min_idle_hours", None),
        "stale_after_days": _safe_call(curator, "get_stale_after_days", None),
        "archive_after_days": _safe_call(curator, "get_archive_after_days", None),
    }


class CuratorPause(BaseModel):
    paused: bool


@app.put("/api/curator/paused")
async def set_curator_paused(body: CuratorPause):
    from agent import curator

    curator.set_paused(bool(body.paused))
    return {"ok": True, "paused": bool(body.paused)}


@app.post("/api/curator/run")
async def run_curator():
    """Trigger a curator review now (backgrounded; tail via action status)."""
    try:
        proc = _spawn_hermes_action(["curator", "run"], "curator-run")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to run curator: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "curator-run"}


@app.get("/api/learning/graph")
async def get_learning_graph(profile: Optional[str] = None):
    """Learning graph payload for the desktop panel.

    Profile-scoped view of learned, non-base skills plus memory chunks, with
    graph links derived from skill relations and memory-skill overlap.
    """
    try:
        from agent.learning_graph import build_learning_graph

        with _profile_scope(profile):
            return build_learning_graph()
    except Exception:
        _log.exception("GET /api/learning/graph failed")
        raise HTTPException(status_code=500, detail="Failed to build learning graph")


class LearningNodeRef(BaseModel):
    id: str
    profile: Optional[str] = None


class LearningNodeEdit(BaseModel):
    id: str
    content: str
    profile: Optional[str] = None


@app.get("/api/learning/node")
async def get_learning_node(id: str, profile: Optional[str] = None):
    """Current content of a journey node (skill SKILL.md or memory chunk), for an edit prefill."""
    from agent.learning_mutations import node_detail

    with _profile_scope(profile):
        res = node_detail(id)
    if not res.get("ok"):
        raise HTTPException(status_code=404, detail=res.get("message", "not found"))
    return res


@app.delete("/api/learning/node")
async def delete_learning_node(body: LearningNodeRef):
    """Delete a journey node — skills are archived (restorable), memories removed."""
    from agent.learning_mutations import delete_node

    with _profile_scope(body.profile):
        res = delete_node(body.id)
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("message", "delete failed"))
    return res


@app.put("/api/learning/node")
async def update_learning_node(body: LearningNodeEdit):
    """Rewrite a journey node's content (SKILL.md or memory chunk)."""
    from agent.learning_mutations import edit_node

    with _profile_scope(body.profile):
        res = edit_node(body.id, body.content)
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("message", "edit failed"))
    return res


def _safe_call(mod, fn_name: str, default):
    try:
        fn = getattr(mod, fn_name, None)
        return fn() if callable(fn) else default
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Portal endpoint — Nous Portal auth + Tool Gateway routing status (read-only).
# ---------------------------------------------------------------------------


@app.get("/api/portal")
async def get_portal_status():
    cfg = load_config() or {}
    auth: Dict[str, Any] = {}
    try:
        from hermes_cli.auth import get_nous_auth_status

        auth = get_nous_auth_status() or {}
    except Exception:
        auth = {}

    features = []
    try:
        from hermes_cli.nous_subscription import get_nous_subscription_features

        feats = get_nous_subscription_features(cfg)
        if feats is not None:
            for feat in feats.items():
                if getattr(feat, "managed_by_nous", False):
                    state = "via Nous Portal"
                elif getattr(feat, "active", False) and getattr(feat, "current_provider", None):
                    state = feat.current_provider
                elif getattr(feat, "active", False):
                    state = "active"
                else:
                    state = "not configured"
                features.append({"label": getattr(feat, "label", ""), "state": state})
    except Exception:
        _log.exception("portal features failed")

    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    return {
        "logged_in": bool(auth.get("logged_in")),
        "portal_url": auth.get("portal_base_url"),
        "inference_url": auth.get("inference_base_url"),
        "provider": str((model_cfg or {}).get("provider") or ""),
        "subscription_url": "https://portal.nousresearch.com/manage-subscription",
        "features": features,
    }


# ---------------------------------------------------------------------------
# Diagnostics: prompt-size, support dump, debug upload, config migrate.
# All produce text output, so they spawn background actions tailed via
# /api/actions/<name>/status.
# ---------------------------------------------------------------------------


@app.post("/api/ops/prompt-size")
async def run_prompt_size():
    try:
        proc = _spawn_hermes_action(["prompt-size"], "prompt-size")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "prompt-size"}


@app.post("/api/ops/dump")
async def run_dump():
    try:
        proc = _spawn_hermes_action(["dump"], "dump")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "dump"}


@app.post("/api/ops/config-migrate")
async def run_config_migrate():
    try:
        proc = _spawn_hermes_action(["config", "migrate"], "config-migrate")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "config-migrate"}


class DebugShareRequest(BaseModel):
    # Redaction is ON by default — force-mode scrubs credential-shaped tokens
    # out of log content before it leaves the machine. The toggle exists so an
    # operator who knows the logs are clean can opt out for fuller fidelity.
    redact: bool = True
    # Recent log lines included in the summary tail (full logs are separate).
    lines: int = 200


@app.post("/api/ops/debug-share")
async def run_debug_share_endpoint(body: DebugShareRequest | None = None):
    """Upload a redacted debug report + full logs and return the paste URLs.

    Unlike the other diagnostics actions (doctor, dump, prompt-size) this is
    *synchronous*: the whole point of ``debug share`` is the set of shareable
    URLs it produces, so we run the upload in a worker thread and return the
    structured ``{urls, failures, redacted, ...}`` payload directly. The
    dashboard renders those as real, copyable links instead of scraping a log
    tail. Pastes auto-delete after 6 hours (handled inside the share core).
    """
    from hermes_cli.debug import build_debug_share

    req = body or DebugShareRequest()
    try:
        result = await asyncio.to_thread(
            build_debug_share,
            log_lines=max(1, min(int(req.lines), 5000)),
            redact=bool(req.redact),
        )
    except RuntimeError as exc:
        # Required summary-report upload failed (offline / paste service down).
        raise HTTPException(status_code=502, detail=f"Upload failed: {exc}")
    except Exception as exc:
        _log.exception("debug share failed")
        raise HTTPException(status_code=500, detail=f"Failed: {exc}")

    return {
        "ok": True,
        "urls": result.urls,
        "failures": result.failures,
        "redacted": result.redacted,
        "auto_delete_seconds": result.auto_delete_seconds,
    }


# ---------------------------------------------------------------------------
# Gateway + update actions (invoked from the Status page).
#
# Both commands are spawned as detached subprocesses so the HTTP request
# returns immediately.  stdin is closed (``DEVNULL``) so any stray ``input()``
# calls fail fast with EOF rather than hanging forever.  stdout/stderr are
# streamed to a per-action log file under ``~/.hermes/logs/<action>.log`` so
# the dashboard can tail them back to the user.
# ---------------------------------------------------------------------------

_ACTION_LOG_DIR: Path = get_hermes_home() / "logs"

# Short ``name`` (from the URL) → absolute log file path.
_ACTION_LOG_FILES: Dict[str, str] = {
    "gateway-restart": "gateway-restart.log",
    "gateway-start": "gateway-start.log",
    "gateway-stop": "gateway-stop.log",
    "hermes-update": "hermes-update.log",
    "doctor": "action-doctor.log",
    "security-audit": "action-security-audit.log",
    "backup": "action-backup.log",
    "import": "action-import.log",
    "checkpoints-prune": "action-checkpoints-prune.log",
    "skills-install": "action-skills-install.log",
    "skills-uninstall": "action-skills-uninstall.log",
    "skills-update": "action-skills-update.log",
    "curator-run": "action-curator-run.log",
    "prompt-size": "action-prompt-size.log",
    "dump": "action-dump.log",
    "config-migrate": "action-config-migrate.log",
    "tools-post-setup": "action-tools-post-setup.log",
}

# ``name`` → most recently spawned Popen handle.  Used so ``status`` can
# report liveness and exit code without shelling out to ``ps``.
_ACTION_PROCS: Dict[str, subprocess.Popen] = {}
_ACTION_COMMANDS: Dict[str, Tuple[str, ...]] = {}

# ``name`` → completed synthetic action result for actions the server handled
# without spawning a subprocess (for example, unsupported Docker updates).
_ACTION_RESULTS: Dict[str, Dict[str, Any]] = {}

_VOICE_SIDECAR_PROC: Optional[subprocess.Popen] = None
_VOICE_SIDECAR_LOCK = threading.Lock()
_LOCAL_VOICE_SIDECAR_PROVIDERS = {"local", "reference", "provider", "sidecar", "gemma", "gemma4", "lmstudio", "vllm"}
_REALTIME_VOICE_DEFAULT_PRODUCTION_LANGUAGES = ("en", "ja")
_REALTIME_VOICE_DEFAULT_PRODUCTION_SCRIPTS = ("Latn", "Jpan")
_REALTIME_VOICE_DEFAULT_QUALITY_TARGETS_MS = {
    "audio_to_partial_transcript_ms": 300,
    "final_transcript_to_first_text_ms": 500,
    "final_transcript_to_first_audio_ms": 900,
    "barge_in_ack_ms": 150,
}
_REALTIME_VOICE_KAME_QUALITY_TARGETS_MS = {
    **_REALTIME_VOICE_DEFAULT_QUALITY_TARGETS_MS,
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
_REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS = REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS
_VOICE_SIDECAR_HEALTH_TIMEOUT = 0.75
_VOICE_SIDECAR_START_TIMEOUT = 8.0


def _record_completed_action(name: str, message: str, exit_code: int = 1) -> None:
    """Record a non-spawned action result and write it to the action log."""
    log_file_name = _ACTION_LOG_FILES[name]
    _ACTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _ACTION_LOG_DIR / log_file_name
    with open(log_path, "ab", buffering=0) as log_file:
        log_file.write(
            f"\n=== {name} completed {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode()
        )
        log_file.write(message.encode("utf-8", errors="replace"))
        if not message.endswith("\n"):
            log_file.write(b"\n")
    _ACTION_PROCS.pop(name, None)
    _ACTION_COMMANDS.pop(name, None)
    _ACTION_RESULTS[name] = {"exit_code": exit_code, "pid": None}


def _dashboard_spawn_executable() -> str:
    """Prefer pythonw.exe for detached dashboard actions on Windows."""
    if sys.platform != "win32":
        return sys.executable
    exe = sys.executable
    if exe.lower().endswith("python.exe"):
        pythonw = os.path.join(os.path.dirname(exe), "pythonw.exe")
        if os.path.isfile(pythonw):
            return pythonw
    return exe


def _spawn_hermes_action(subcommand: List[str], name: str) -> subprocess.Popen:
    """Spawn ``hermes <subcommand>`` detached and record the Popen handle.

    Uses the running interpreter's ``hermes_cli.main`` module so the action
    inherits the same venv/PYTHONPATH the web server is using.
    """
    log_file_name = _ACTION_LOG_FILES[name]
    _ACTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _ACTION_LOG_DIR / log_file_name
    log_file = open(log_path, "ab", buffering=0)
    log_file.write(
        f"\n=== {name} started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode()
    )

    cmd = [_dashboard_spawn_executable(), "-m", "hermes_cli.main", *subcommand]

    popen_kwargs: Dict[str, Any] = {
        "cwd": str(PROJECT_ROOT),
        "stdin": subprocess.DEVNULL,
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        "env": {**os.environ, "HERMES_NONINTERACTIVE": "1"},
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = windows_detach_flags()
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    # The child inherits its own duplicated fd for stdout/stderr, so the
    # parent's handle can be released immediately — otherwise we leak one
    # fd per spawned action.
    log_file.close()
    _ACTION_RESULTS.pop(name, None)
    _ACTION_COMMANDS[name] = tuple(subcommand)
    _ACTION_PROCS[name] = proc
    return proc


def _tail_lines(path: Path, n: int) -> List[str]:
    """Return the last ``n`` lines of ``path``.  Reads the whole file — fine
    for our small per-action logs.  Binary-decoded with ``errors='replace'``
    so log corruption doesn't 500 the endpoint."""
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    return lines[-n:] if n > 0 else lines


def _gateway_subcommand(profile: Optional[str], verb: str) -> List[str]:
    return _profile_cli_args(profile) + ["gateway", verb]


def _gateway_display_command(profile: Optional[str], verb: str) -> str:
    return " ".join(["hermes", *_gateway_subcommand(profile, verb)])


# Slack member IDs (users U..., Enterprise Grid W...). Kept in sync with the
# frontend SLACK_MEMBER_ID_RE in web/src/pages/ChannelsPage.tsx.
_SLACK_MEMBER_ID_RE = re.compile(r"[UW][A-Z0-9]{2,}")


def _validate_messaging_env_value(platform_id: str, key: str, value: str) -> None:
    """Reject platform credentials that are clearly in the wrong field."""
    if platform_id != "slack" or not value:
        return

    if key == "SLACK_BOT_TOKEN" and not value.startswith("xoxb-"):
        raise HTTPException(
            status_code=400,
            detail="Slack Bot Token must start with xoxb-. Paste the bot token from OAuth & Permissions.",
        )
    if key == "SLACK_APP_TOKEN" and not value.startswith("xapp-"):
        raise HTTPException(
            status_code=400,
            detail="Slack App Token must start with xapp-. Paste the app-level token from Basic Information > App-Level Tokens.",
        )
    if key == "SLACK_ALLOWED_USERS":
        # Mirror the gateway's parse (gateway/platforms/slack.py): split on comma,
        # strip, and drop empty entries so a trailing/interior comma isn't rejected
        # here when the runtime would accept it. "*" is the allow-all wildcard.
        user_ids = [part.strip() for part in value.split(",") if part.strip()]
        invalid = [
            user_id
            for user_id in user_ids
            if user_id != "*" and not _SLACK_MEMBER_ID_RE.fullmatch(user_id)
        ]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail="Slack allowed user IDs must be comma-separated member IDs like U01ABC2DEF3.",
            )


def _spawn_gateway_restart(profile: Optional[str] = None) -> Tuple[subprocess.Popen, bool]:
    """Spawn ``hermes gateway restart``, reusing an in-flight restart.

    Multiple dashboard paths can request a restart in quick succession
    (restart button double-click, or a stale cached frontend firing its own
    restart after the server already auto-restarted post-onboarding). Two
    concurrent ``hermes gateway restart`` children race each other on the
    manual kill-and-start path, so reuse the live one instead.

    Returns ``(proc, reused)``.
    """
    subcommand = _gateway_subcommand(profile, "restart")
    existing = _ACTION_PROCS.get("gateway-restart")
    if existing is not None and existing.poll() is None:
        existing_command = _ACTION_COMMANDS.get("gateway-restart")
        if existing_command is None or existing_command == tuple(subcommand):
            return existing, True
        raise RuntimeError("gateway restart already in progress for another profile")
    return _spawn_hermes_action(subcommand, "gateway-restart"), False


def _restart_gateway_after_webhook_enable(profile: Optional[str] = None) -> dict[str, Any]:
    """Best-effort gateway restart after enabling the webhook platform."""
    try:
        proc, reused = _spawn_gateway_restart(profile)
    except Exception as exc:
        _log.exception("Failed to auto-restart gateway after enabling webhooks")
        return {
            "restart_started": False,
            "restart_error": str(exc),
        }
    if reused:
        _log.info(
            "Webhook enable: reusing in-flight gateway restart (pid %s)",
            proc.pid,
        )
    return {
        "restart_started": True,
        "restart_action": "gateway-restart",
        "restart_pid": proc.pid,
    }


@app.post("/api/gateway/restart")
async def restart_gateway(profile: Optional[str] = None):
    """Kick off a ``hermes gateway restart`` in the background."""
    try:
        proc, _reused = _spawn_gateway_restart(profile)
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn gateway restart")
        raise HTTPException(status_code=500, detail=f"Failed to restart gateway: {exc}")
    return {
        "ok": True,
        "pid": proc.pid,
        "name": "gateway-restart",
    }


@app.post("/api/gateway/drain")
async def gateway_drain(request: Request):
    """Begin or cancel an external (NAS-driven) gateway drain.

    Authenticated by the non-interactive token-auth seam: the
    ``dashboard_auth/drain`` plugin registers this exact path as a token route
    and verifies the ``Authorization`` bearer secret. If that plugin isn't
    active (no ``HERMES_DASHBOARD_DRAIN_SECRET``), the route is NOT a token
    route, so on a gated bind the cookie gate handles it (a browser session can
    still drive it from the dashboard) and on a loopback bind the legacy
    session-token gate applies — either way it is never unauthenticated on a
    network-exposed bind.

    Body: ``{"action": "drain"}`` (begin) or ``{"action": "cancel"}`` (cancel).
    Begin writes the ``.drain_request.json`` marker the gateway's
    ``_drain_control_watcher`` observes (flip to ``draining`` + refuse new
    turns); cancel removes it (revert to ``running`` + re-accept). Idempotent
    on both sides. This endpoint only writes/removes the marker — the gateway
    process owns the actual state transition (there is no HTTP control channel
    into the running gateway; the marker IS the channel, decisions.md Q-B).

    The force-override (D6: "unless a user commands it") is NOT here — an
    immediate, drain-skipping action maps onto the existing
    ``POST /api/gateway/restart`` force path, which supersedes a drain.
    """
    from gateway.drain_control import (
        clear_drain_request,
        drain_requested,
        write_drain_request,
    )

    try:
        body = await request.json()
    except Exception:
        body = {}
    action = str((body or {}).get("action", "drain")).strip().lower()

    # Attribute the request to the verified token principal when present
    # (token-auth seam attaches it); fall back to a generic label otherwise.
    principal_obj = getattr(request.state, "token_principal", None)
    principal = getattr(principal_obj, "principal", None) or "dashboard"

    if action == "cancel":
        existed = clear_drain_request()
        _log.info("Gateway drain CANCEL requested by %s (existed=%s)", principal, existed)
        return {"ok": True, "action": "cancel", "was_draining": existed}

    if action != "drain":
        raise HTTPException(
            status_code=400,
            detail=f"Unknown drain action {action!r}; expected 'drain' or 'cancel'",
        )

    payload = write_drain_request(
        principal=str(principal),
        suppress_notification=bool((body or {}).get("suppress_notification", False)),
    )
    _log.info(
        "Gateway drain BEGIN requested by %s (suppress_notification=%s)",
        principal,
        payload["suppress_notification"],
    )
    return {
        "ok": True,
        "action": "drain",
        "requested_at": payload["requested_at"],
        # Echo so a caller polling /api/status knows the marker is now set;
        # the gateway watcher flips gateway_state -> draining within ~1s.
        "draining": drain_requested(),
        "suppress_notification": payload["suppress_notification"],
    }


@app.post("/api/hermes/update")
async def update_hermes():
    """Kick off ``hermes update`` in the background."""
    if _dashboard_local_update_managed_externally():
        message = (
            "Hermes updates are managed outside this dashboard in "
            "containerized environments. The built-in local updater is "
            "disabled here."
        )
        _record_completed_action("hermes-update", message, exit_code=1)
        return {
            "ok": False,
            "pid": None,
            "name": "hermes-update",
            "error": "dashboard_update_managed_externally",
            "message": message,
            "update_command": "managed outside dashboard",
        }

    install_method = detect_install_method(PROJECT_ROOT)
    if install_method == "docker":
        message = format_docker_update_message()
        _record_completed_action("hermes-update", message, exit_code=1)
        return {
            "ok": False,
            "pid": None,
            "name": "hermes-update",
            "error": "docker_update_unsupported",
            "message": message,
            "update_command": recommended_update_command_for_method(install_method),
        }

    try:
        proc = _spawn_hermes_action(["update"], "hermes-update")
    except Exception as exc:
        _log.exception("Failed to spawn hermes update")
        raise HTTPException(status_code=500, detail=f"Failed to start update: {exc}")
    return {
        "ok": True,
        "pid": proc.pid,
        "name": "hermes-update",
    }


def _recent_upstream_commits(n: int = 20) -> List[Dict[str, Any]]:
    """Commits the local checkout is behind ``origin/main`` by, newest first.

    Logs the SAME range the behind-count uses (``HEAD..origin/main`` — see
    ``banner._check_via_local_git``), NOT the branch's ``@{upstream}``. On a
    feature-branch checkout ``@{upstream}`` is the branch's own tip (zero
    commits), which would leave the changelog empty even though the count is
    non-zero. Pinning to ``origin/main`` keeps count and changelog consistent.

    Best-effort: returns [] if not a git checkout, origin/main is unreachable,
    or git is unavailable. Never raises into the request path.
    """
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "log",
                "--format=%H%x1f%s%x1f%an%x1f%ct",
                "HEAD..origin/main",
                f"-n{int(n)}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return []
        rows: List[Dict[str, Any]] = []
        for line in out.stdout.splitlines():
            if not line.strip():
                continue
            parts = (line.split("\x1f") + ["", "", "", "0"])[:4]
            sha, summary, author, at = parts
            rows.append(
                {
                    "sha": sha[:7],
                    "summary": summary,
                    "author": author,
                    "at": int(at or 0),
                }
            )
        return rows
    except Exception:
        return []


@app.get("/api/hermes/update/check")
async def check_hermes_update(force: bool = False):
    """Report whether a Hermes update is available, without applying it.

    Powers the dashboard's "check before you update" flow: the System page
    shows the commit-behind count and asks the user to confirm before
    ``POST /api/hermes/update`` actually runs ``hermes update``.

    Returns:
        install_method: 'git' | 'pip' | 'docker' | 'nixos' | 'homebrew' | ...
        current_version: installed Hermes version string
        behind: commits behind upstream (>=1), 0 if up to date,
                -1 if behind by an unknown count (nix/pypi), or null if the
                check could not run (offline, no remote, etc.)
        update_available: convenience bool (behind is non-zero and not null)
        can_apply: True when the dashboard's update button can apply it
                   in place (git/pip); False for docker/nix/homebrew where the
                   user must update out-of-band
        update_command: the recommended command for this install method
        message: human-readable guidance for non-applyable methods
        commits: for git/pip installs that are behind, a list of the commits
                 the local checkout is behind upstream by — each
                 {sha, summary, author, at}. Absent/empty otherwise. The
                 desktop's remote update overlay renders this as "what's
                 changed". Additive: existing consumers ignore it.
    """
    if _dashboard_local_update_managed_externally():
        return {
            "install_method": "managed-runtime",
            "current_version": __version__,
            "behind": None,
            "update_available": False,
            "can_apply": False,
            "update_command": "managed outside dashboard",
            "message": (
                "Hermes updates are managed outside this dashboard in "
                "containerized environments."
            ),
        }

    install_method = detect_install_method(PROJECT_ROOT)
    update_command = recommended_update_command_for_method(install_method)

    payload: Dict[str, Any] = {
        "install_method": install_method,
        "current_version": __version__,
        "behind": None,
        "update_available": False,
        "can_apply": install_method in ("git", "pip"),
        "update_command": update_command,
        "message": None,
    }

    if install_method == "docker":
        payload["message"] = format_docker_update_message()
        return payload

    # banner.check_for_updates() handles git / pypi / nix-revision paths and
    # caches the result for 6h. ``force`` busts the cache so the "Check now"
    # button reflects reality immediately.
    try:
        from hermes_cli.banner import check_for_updates

        if force:
            try:
                (get_hermes_home() / ".update_check").unlink()
            except OSError:
                pass

        behind = await asyncio.to_thread(check_for_updates)
    except Exception:
        _log.exception("Update check failed")
        behind = None

    payload["behind"] = behind
    if behind is None:
        payload["message"] = "Couldn't reach the update source — try again later."
    elif behind == 0:
        payload["message"] = "You're on the latest version."
    else:
        payload["update_available"] = True
        # Enrich with the actual commits we're behind by, so the desktop's
        # remote update overlay can show "what's changed". git/pip only;
        # best-effort (empty list on any failure).
        if install_method in ("git", "pip"):
            payload["commits"] = await asyncio.to_thread(_recent_upstream_commits)

    return payload


@app.post("/api/audio/transcribe")
async def transcribe_audio_upload(payload: AudioTranscriptionRequest):
    data_url = (payload.data_url or "").strip()
    if not data_url.startswith("data:") or "," not in data_url:
        raise HTTPException(status_code=400, detail="Invalid audio payload")

    header, encoded = data_url.split(",", 1)
    if ";base64" not in header:
        raise HTTPException(
            status_code=400, detail="Audio payload must be base64 encoded"
        )

    mime_type = (
        payload.mime_type or header[5:].split(";", 1)[0] or "audio/webm"
    ).strip()
    normalized_mime_type = mime_type.split(";", 1)[0].lower()
    if not (
        normalized_mime_type.startswith("audio/")
        or normalized_mime_type == "video/webm"
    ):
        raise HTTPException(
            status_code=400, detail="Payload must be an audio recording"
        )

    try:
        audio_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Audio payload is not valid base64")

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio recording is empty")
    if len(audio_bytes) > _MAX_TRANSCRIPTION_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio recording is too large")

    temp_path = ""
    try:
        suffix = _audio_extension_for_mime(mime_type)
        with tempfile.NamedTemporaryFile(
            prefix="hermes-desktop-voice-",
            suffix=suffix,
            delete=False,
        ) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        from tools.transcription_tools import transcribe_audio

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, transcribe_audio, temp_path)
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Desktop voice transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    if not result.get("success"):
        raise HTTPException(
            status_code=400,
            detail=result.get("error") or "Transcription failed",
        )

    return {
        "ok": True,
        "transcript": str(result.get("transcript") or "").strip(),
        "provider": result.get("provider"),
    }


class TTSSpeakRequest(BaseModel):
    text: str


def _elevenlabs_voice_label(voice: Dict[str, Any]) -> str:
    name = str(voice.get("name") or voice.get("voice_id") or "Voice").strip()
    category = str(voice.get("category") or "").strip()

    return f"{name} ({category})" if category else name


# Collapses repeated identical ElevenLabs voice-list failures (the desktop
# re-polls on every settings open/focus) to a single log line. Re-arms on
# success or when the error signature changes, so a real new failure is seen.
_voice_list_last_error: Optional[str] = None


def _voice_list_error_logged_once(signature: Optional[str]) -> bool:
    """Return True if ``signature`` is new and should be logged now.

    Passing ``None`` clears the latch (call on success). Idempotent per
    signature: the same error logs once until it changes.
    """
    global _voice_list_last_error
    if signature is None:
        _voice_list_last_error = None
        return False
    if signature == _voice_list_last_error:
        return False
    _voice_list_last_error = signature
    return True


@app.get("/api/audio/elevenlabs/voices")
async def get_elevenlabs_voices():
    """Return ElevenLabs voices when an API key is configured.

    The desktop UI uses this for the ``tts.elevenlabs.voice_id`` dropdown.
    Only non-secret voice metadata is returned; the API key stays server-side.
    """
    api_key = (load_env().get("ELEVENLABS_API_KEY") or os.environ.get("ELEVENLABS_API_KEY") or "").strip()
    if not api_key:
        return {"available": False, "voices": []}

    request = urllib.request.Request(
        "https://api.elevenlabs.io/v1/voices",
        headers={
            "Accept": "application/json",
            "xi-api-key": api_key,
        },
    )

    try:
        loop = asyncio.get_running_loop()

        def _fetch() -> Dict[str, Any]:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))

        payload = await loop.run_in_executor(None, _fetch)
    except urllib.error.HTTPError as exc:
        # An auth failure (bad/expired/scoped key) is a persistent,
        # user-fixable state, not a transient blip — the desktop polls this on
        # every settings open/focus, so a per-poll WARNING floods the log
        # (#voice-list-401-spam). Treat 401/403 as "integration unavailable":
        # report it to the UI with a 200 and log at most once until the error
        # signature changes (see _voice_list_error_logged_once).
        if exc.code in (401, 403):
            if _voice_list_error_logged_once(f"http-{exc.code}"):
                _log.info(
                    "ElevenLabs voices unavailable: %s — check ELEVENLABS_API_KEY", exc
                )
            return {"available": False, "voices": [], "error": "unauthorized"}
        if _voice_list_error_logged_once(f"http-{exc.code}"):
            _log.warning("ElevenLabs voice list failed: %s", exc)
        raise HTTPException(status_code=502, detail="Could not load ElevenLabs voices")
    except Exception as exc:
        if _voice_list_error_logged_once(str(exc)):
            _log.warning("ElevenLabs voice list failed: %s", exc)
        raise HTTPException(status_code=502, detail="Could not load ElevenLabs voices")
    _voice_list_error_logged_once(None)  # success — re-arm logging for next failure

    voices = []
    for voice in payload.get("voices") or []:
        if not isinstance(voice, dict):
            continue

        voice_id = str(voice.get("voice_id") or "").strip()
        if not voice_id:
            continue

        voices.append({
            "voice_id": voice_id,
            "name": str(voice.get("name") or voice_id),
            "label": _elevenlabs_voice_label(voice),
        })

    voices.sort(key=lambda item: str(item.get("label") or "").lower())
    return {"available": True, "voices": voices}


@app.post("/api/audio/speak")
async def speak_text(payload: TTSSpeakRequest):
    """Synthesize speech and return audio as base64 data URL.

    Used by the desktop voice-conversation mode to play back assistant
    responses without exposing the on-disk file path. Reuses the
    existing TTS provider chain (Edge / OpenAI / ElevenLabs / etc.)
    configured in ``~/.hermes/config.yaml`` under ``tts.``.
    """
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        from tools.tts_tool import text_to_speech_tool
        loop = asyncio.get_running_loop()
        result_json = await loop.run_in_executor(None, text_to_speech_tool, text)
    except Exception as exc:
        _log.exception("Desktop voice TTS failed")
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {exc}")

    try:
        result = json.loads(result_json) if isinstance(result_json, str) else result_json
    except Exception:
        raise HTTPException(status_code=500, detail="Invalid TTS response")

    if not result.get("success"):
        raise HTTPException(
            status_code=400,
            detail=result.get("error") or "Speech synthesis failed",
        )

    file_path = result.get("file_path")
    if not file_path or not os.path.isfile(file_path):
        raise HTTPException(status_code=500, detail="Audio file missing")

    ext = os.path.splitext(file_path)[1].lower()
    mime_type = {
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
    }.get(ext, "audio/mpeg")

    try:
        with open(file_path, "rb") as fh:
            audio_bytes = fh.read()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read audio: {exc}")
    finally:
        try:
            os.unlink(file_path)
        except OSError:
            pass

    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return {
        "ok": True,
        "data_url": f"data:{mime_type};base64,{encoded}",
        "mime_type": mime_type,
        "provider": result.get("provider"),
    }


@app.get("/api/actions/{name}/status")
async def get_action_status(name: str, lines: int = 200):
    """Tail an action log and report whether the process is still running."""
    log_file_name = _ACTION_LOG_FILES.get(name)
    if log_file_name is None:
        raise HTTPException(status_code=404, detail=f"Unknown action: {name}")

    log_path = _ACTION_LOG_DIR / log_file_name
    tail = _tail_lines(log_path, min(max(lines, 1), 2000))

    proc = _ACTION_PROCS.get(name)
    if proc is None:
        result = _ACTION_RESULTS.get(name)
        running = False
        exit_code = result.get("exit_code") if result else None
        pid = result.get("pid") if result else None
    else:
        exit_code = proc.poll()
        running = exit_code is None
        pid = proc.pid
        if exit_code is not None:
            try:
                proc.wait(timeout=1)
            except Exception:
                pass
            _ACTION_RESULTS[name] = {"exit_code": exit_code, "pid": pid}
            _ACTION_PROCS.pop(name, None)
            _ACTION_COMMANDS.pop(name, None)

    return {
        "name": name,
        "running": running,
        "exit_code": exit_code,
        "pid": pid,
        "lines": tail,
    }


@app.get("/api/sessions")
async def get_sessions(
    limit: int = 20,
    offset: int = 0,
    min_messages: int = 0,
    archived: str = "exclude",
    order: str = "created",
    source: str = None,
    exclude_sources: str = None,
    cwd_prefix: str = None,
    profile: Optional[str] = None,
):
    """List sessions.

    ``archived`` controls how soft-archived sessions are treated:
    ``exclude`` (default) hides them, ``only`` returns just the archived ones
    (used by the desktop "Archived sessions" settings panel), and ``include``
    returns both.

    ``order`` controls pagination order: ``created`` (default, by original
    start time) or ``recent`` (by latest activity across the compression
    chain). ``recent`` keeps a long-running conversation on the first page
    after it auto-compresses into a fresh continuation id.
    """
    if archived not in ("exclude", "only", "include"):
        raise HTTPException(
            status_code=400,
            detail="archived must be one of: exclude, only, include",
        )
    if order not in ("created", "recent"):
        raise HTTPException(
            status_code=400,
            detail="order must be one of: created, recent",
        )
    profile_name: Optional[str] = None
    if profile:
        profile_name, _ = _cron_profile_home(profile)
    try:
        db = _open_session_db_for_profile(profile)
        try:
            min_message_count = max(0, min_messages)
            archived_only = archived == "only"
            include_archived = archived == "include"
            # Optional source scoping: ``source`` includes a single class,
            # ``exclude_sources`` (comma-separated) drops classes. The desktop
            # uses these to split recents (exclude=cron) from the cron-jobs
            # section (source=cron) into two independent lists.
            exclude_list = [s for s in (exclude_sources or "").split(",") if s.strip()]
            sessions = db.list_sessions_rich(
                source=source or None,
                exclude_sources=exclude_list or None,
                cwd_prefix=(cwd_prefix or None),
                limit=limit,
                offset=offset,
                min_message_count=min_message_count,
                include_archived=include_archived,
                archived_only=archived_only,
                order_by_last_active=order == "recent",
            )
            total = db.session_count(
                source=source or None,
                cwd_prefix=(cwd_prefix or None),
                exclude_sources=exclude_list or None,
                min_message_count=min_message_count,
                include_archived=include_archived,
                archived_only=archived_only,
                exclude_children=True,
            )
            now = time.time()
            for s in sessions:
                s["is_active"] = (
                    s.get("ended_at") is None
                    and (now - s.get("last_active", s.get("started_at", 0))) < 300
                )
                if profile_name:
                    s["profile"] = profile_name
                    s["is_default_profile"] = profile_name == "default"
                # SQLite stores the flag as 0/1; expose a real JSON boolean.
                s["archived"] = bool(s.get("archived"))
            return {"sessions": sessions, "total": total, "limit": limit, "offset": offset}
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception:
        _log.exception("GET /api/sessions failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/profiles/sessions")
def get_profiles_sessions(
    limit: int = 20,
    offset: int = 0,
    min_messages: int = 0,
    archived: str = "exclude",
    order: str = "recent",
    profile: str = "all",
    source: str = None,
    exclude_sources: str = None,
):
    """Unified, read-only session list aggregated across ALL profiles.

    Intentionally process-light: this opens each profile's ``state.db`` directly
    from disk — it does NOT spawn a dashboard backend per profile. Each returned
    session is tagged with its owning ``profile`` so the desktop renders one
    browsable list and only spins up a profile's backend when the user actually
    interacts (sends a message). A user with a single (default) profile gets the
    same rows as ``/api/sessions``, just tagged ``profile="default"``.
    """
    if archived not in ("exclude", "only", "include"):
        raise HTTPException(status_code=400, detail="archived must be one of: exclude, only, include")
    if order not in ("created", "recent"):
        raise HTTPException(status_code=400, detail="order must be one of: created, recent")

    from hermes_state import SessionDB
    from hermes_cli import profiles as profiles_mod

    targets: List[Tuple[str, Path]] = []
    if profile and profile != "all":
        name, home = _cron_profile_home(profile)
        targets.append((name, home))
    else:
        try:
            infos = profiles_mod.list_profiles()
            targets = [(info.name, info.path) for info in infos]
        except Exception:
            _log.exception("GET /api/profiles/sessions: list_profiles failed")
            targets = []
        if not targets:
            targets.append(("default", profiles_mod.get_profile_dir("default")))

    min_message_count = max(0, min_messages)
    archived_only = archived == "only"
    include_archived = archived == "include"
    # Source scoping (see /api/sessions): recents pass exclude_sources=cron,
    # the cron-jobs section passes source=cron — two independent lists so
    # newest cron sessions can't starve the recents page.
    source_filter = source or None
    exclude_list = [s for s in (exclude_sources or "").split(",") if s.strip()]
    # Over-fetch per profile so the merged+sorted window is correct for the
    # requested page. Capped so a huge profile can't blow up the response.
    per_profile = min(max(limit + offset, limit), 500)

    merged: List[Dict[str, Any]] = []
    total = 0
    profile_totals: Dict[str, int] = {}
    errors: List[Dict[str, str]] = []
    now = time.time()
    for name, home in targets:
        db_path = Path(home) / "state.db"
        if not db_path.exists():
            continue
        try:
            # Read-only: this loop runs on every sidebar refresh, so it must
            # never DDL/write-lock another profile's live DB (see SessionDB
            # read_only docstring).
            db = SessionDB(db_path=db_path, read_only=True)
        except Exception as exc:
            errors.append({"profile": name, "error": str(exc)})
            continue
        try:
            rows = db.list_sessions_rich(
                source=source_filter,
                exclude_sources=exclude_list or None,
                limit=per_profile,
                offset=0,
                min_message_count=min_message_count,
                include_archived=include_archived,
                archived_only=archived_only,
                order_by_last_active=order == "recent",
            )
            profile_total = db.session_count(
                source=source_filter,
                exclude_sources=exclude_list or None,
                min_message_count=min_message_count,
                include_archived=include_archived,
                archived_only=archived_only,
                exclude_children=True,
            )
            total += profile_total
            profile_totals[name] = profile_total
            for s in rows:
                s["profile"] = name
                s["is_default_profile"] = name == "default"
                s["is_active"] = (
                    s.get("ended_at") is None
                    and (now - s.get("last_active", s.get("started_at", 0))) < 300
                )
                s["archived"] = bool(s.get("archived"))
                merged.append(s)
        except Exception as exc:
            errors.append({"profile": name, "error": str(exc)})
        finally:
            db.close()

    sort_key = "last_active" if order == "recent" else "started_at"
    merged.sort(key=lambda s: s.get(sort_key) or s.get("started_at") or 0, reverse=True)
    window = merged[offset:offset + limit]
    return {
        "sessions": window,
        "total": total,
        "profile_totals": profile_totals,
        "limit": limit,
        "offset": offset,
        "errors": errors,
    }


@app.get("/api/sessions/search")
async def search_sessions(q: str = "", limit: int = 20, profile: Optional[str] = None):
    """Search sessions by ID plus full-text message content using FTS5.

    Direct session-id matches are surfaced first, then FTS message-content
    matches. Results are deduped by compression lineage, not by raw
    ``session_id``. Auto-compression rotates a conversation onto a fresh
    session id (and leaves the old segment's messages in the FTS index), so one
    logical chat can own many ``sessions`` rows that all match the same query.
    Branches also use ``parent_session_id``, but they are real alternate
    conversations; don't collapse branch-specific hits back into the parent.
    """
    if not q or not q.strip():
        return {"results": []}
    try:
        db = _open_session_db_for_profile(profile)
        try:
            safe_limit = max(1, min(int(limit or 20), 100))

            # Walk parent_session_id to the compression root, memoized so a
            # chain of compression segments only costs one walk. We deliberately
            # stop at branch/delegate edges: those sessions may diverge from the
            # parent and should remain searchable on their own.
            root_cache: dict = {}

            def compression_root(session_id: str) -> str:
                if not session_id:
                    return session_id
                if session_id in root_cache:
                    return root_cache[session_id]
                chain = []
                cur = session_id
                visited = set()
                root = session_id
                while cur and cur not in visited:
                    visited.add(cur)
                    chain.append(cur)
                    if cur in root_cache:
                        root = root_cache[cur]
                        break
                    try:
                        s = db.get_session(cur)
                    except Exception:
                        s = None
                    if not s:
                        root = cur
                        break
                    parent = s.get("parent_session_id") if isinstance(s, dict) else None
                    if not parent:
                        root = cur
                        break
                    try:
                        parent_session = db.get_session(parent)
                    except Exception:
                        parent_session = None
                    if not parent_session:
                        root = cur
                        break
                    parent_ended_at = parent_session.get("ended_at")
                    started_at = s.get("started_at")
                    is_compression_edge = (
                        parent_session.get("end_reason") == "compression"
                        and parent_ended_at is not None
                        and started_at is not None
                        and started_at >= parent_ended_at
                    )
                    if not is_compression_edge:
                        root = cur
                        break
                    cur = parent
                for node in chain:
                    root_cache[node] = root
                return root

            tip_cache: dict = {}

            def lineage_tip(root_id: str) -> str:
                if root_id in tip_cache:
                    return tip_cache[root_id]
                tip = root_id
                try:
                    resolved = db.get_compression_tip(root_id)
                    if resolved:
                        tip = resolved
                except Exception:
                    pass
                tip_cache[root_id] = tip
                return tip

            # Both ID matches and content matches share one keyspace, keyed by
            # compression lineage root, so an id-hit and a content-hit on the
            # same logical conversation collapse to a single result. The first
            # hit for a lineage wins; ID matches run first and take priority.
            seen: dict = {}

            def add_lineage_result(raw_sid: str, payload: dict) -> None:
                if not raw_sid:
                    return
                root = compression_root(raw_sid)
                if root in seen or len(seen) >= safe_limit:
                    return
                payload = dict(payload)
                payload["session_id"] = lineage_tip(root)
                payload["lineage_root"] = root
                seen[root] = payload

            # Direct ID matches first: users often paste a session id from CLI,
            # logs, or another Hermes surface. FTS can't find those unless the
            # id happens to appear in message text. search_sessions_by_id is
            # SQL-bounded, so this stays cheap even with thousands of sessions.
            for row in db.search_sessions_by_id(q, limit=safe_limit, include_archived=True):
                sid = row.get("id")
                preview = (row.get("preview") or "").strip()
                snippet = preview or f"Session ID: {sid}"
                add_lineage_result(
                    sid,
                    {
                        "snippet": snippet,
                        "role": None,
                        "source": row.get("source"),
                        "model": row.get("model"),
                        "session_started": row.get("started_at"),
                    },
                )

            # Auto-add prefix wildcards so partial words match
            # e.g. "nimb" → "nimb*" matches "nimby"
            # Preserve quoted phrases and existing wildcards as-is
            import re
            terms = []
            for token in re.findall(r'"[^"]*"|\S+', q.strip()):
                if token.startswith('"') or token.endswith("*"):
                    terms.append(token)
                else:
                    terms.append(token + "*")
            prefix_query = " ".join(terms)
            # Over-fetch so lineage dedup can still surface `limit` distinct
            # conversations even when several hits collapse onto one root.
            fetch_limit = max(safe_limit * 5, 50)
            matches = db.search_messages(query=prefix_query, limit=fetch_limit)

            for m in matches:
                if len(seen) >= safe_limit:
                    break
                add_lineage_result(
                    m["session_id"],
                    {
                        "snippet": m.get("snippet", ""),
                        "role": m.get("role"),
                        "source": m.get("source"),
                        "model": m.get("model"),
                        "session_started": m.get("session_started"),
                    },
                )
            return {"results": list(seen.values())}
        finally:
            db.close()
    except HTTPException:
        raise
    except Exception:
        _log.exception("GET /api/sessions/search failed")
        raise HTTPException(status_code=500, detail="Search failed")


def _normalize_config_for_web(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize config for the web UI.

    Hermes supports ``model`` as either a bare string (``"anthropic/claude-sonnet-4"``)
    or a dict (``{default: ..., provider: ..., base_url: ...}``).  The schema is built
    from DEFAULT_CONFIG where ``model`` is a string, but user configs often have the
    dict form.  Normalize to the string form so the frontend schema matches.

    Also surfaces ``model_context_length`` as a top-level field so the web UI can
    display and edit it.  A value of 0 means "auto-detect".
    """
    config = dict(config)  # shallow copy
    model_val = config.get("model")
    if isinstance(model_val, dict):
        # Extract context_length before flattening the dict
        ctx_len = model_val.get("context_length", 0)
        config["model"] = model_val.get("default", model_val.get("name", ""))
        config["model_context_length"] = ctx_len if isinstance(ctx_len, int) else 0
    else:
        config["model_context_length"] = 0
    return config


def _memory_provider_config_path(provider: MemoryProvider) -> Path:
    return get_hermes_home() / provider.name / "config.json"


def _read_memory_provider_file(provider: MemoryProvider) -> Dict[str, Any]:
    path = _memory_provider_config_path(provider)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _log.warning("Failed to read memory provider config from %s", path, exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def _read_field_value(field: ProviderField, data: Dict[str, Any]) -> str:
    """Resolve the stored value for a non-secret field, honoring legacy reads."""

    for source_key in (field.key, *field.aliases):
        value = data.get(source_key)
        if value:
            return str(value)

    env_on_disk = load_env()
    for env_key in field.env_fallbacks:
        value = env_on_disk.get(env_key)
        if value:
            return str(value)

    return field.default


def _field_is_set(field: ProviderField, data: Dict[str, Any]) -> bool:
    """Whether a secret field has a value anywhere it may have been written."""

    env_on_disk = load_env()
    for env_key in (field.env_key, *field.env_fallbacks):
        if env_key and env_on_disk.get(env_key):
            return True
    return any(data.get(source_key) for source_key in (field.key, *field.aliases))


def _memory_provider_payload(provider: MemoryProvider) -> Dict[str, Any]:
    data = _read_memory_provider_file(provider)
    fields: List[Dict[str, Any]] = []

    for field in provider.fields:
        entry: Dict[str, Any] = {
            "key": field.key,
            "label": field.label,
            "kind": field.kind,
            "description": field.description,
            "placeholder": field.placeholder,
            "options": [
                {"value": opt.value, "label": opt.label, "description": opt.description}
                for opt in field.options
            ],
        }

        if field.is_secret:
            # Secrets are write-only over the API; only expose whether one is set.
            entry["value"] = ""
            entry["is_set"] = _field_is_set(field, data)
        else:
            value = _read_field_value(field, data)
            if field.kind == "select" and value not in field.allowed_values():
                value = field.default
            entry["value"] = value
            entry["is_set"] = bool(value)

        fields.append(entry)

    return {"name": provider.name, "label": provider.label, "fields": fields}


def _coerce_field_value(field: ProviderField, raw: str) -> str:
    """Validate and normalize a submitted non-secret value, or raise ValueError."""

    value = (raw or "").strip()
    if field.kind == "select":
        if not value:
            value = field.default
        if value not in field.allowed_values():
            raise ValueError(f"Invalid value for '{field.key}'")
        return value
    return value or field.default


@app.get("/api/memory/providers/{name}/config")
async def get_memory_provider_config(name: str):
    provider = get_memory_provider(name)
    if provider is None:
        # Undeclared providers (e.g. builtin) have no config surface. Return an
        # empty schema so the generic panel simply renders nothing.
        return {"name": name, "label": name, "fields": []}
    return _memory_provider_payload(provider)


@app.put("/api/memory/providers/{name}/config")
async def update_memory_provider_config(name: str, body: MemoryProviderConfigUpdate):
    provider = get_memory_provider(name)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Unknown memory provider: {name}")

    values = body.values or {}

    try:
        existing = _read_memory_provider_file(provider)
        json_values: Dict[str, Any] = {}
        secrets: Dict[str, str] = {}

        for field in provider.fields:
            if field.is_secret:
                submitted = (values.get(field.key) or "").strip()
                if submitted and field.env_key:
                    secrets[field.env_key] = submitted
                continue

            raw = (
                values[field.key]
                if field.key in values
                else str(existing.get(field.key, field.default))
            )
            json_values[field.key] = _coerce_field_value(field, raw)

        config = load_config()
        memory_config = config.get("memory")
        if not isinstance(memory_config, dict):
            memory_config = {}
            config["memory"] = memory_config
        memory_config["provider"] = provider.name
        save_config(config)

        path = _memory_provider_config_path(provider)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing.update(json_values)
        from utils import atomic_json_write

        atomic_json_write(path, existing, mode=0o600)

        for env_key, secret in secrets.items():
            save_env_value(env_key, secret)

        return {"ok": True}
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        _log.exception("PUT /api/memory/providers/%s/config failed", name)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/config")
async def get_config(profile: Optional[str] = None):
    with _profile_scope(profile):
        config = _normalize_config_for_web(load_config())
    # Strip internal keys that the frontend shouldn't see or send back
    return {k: v for k, v in config.items() if not k.startswith("_")}


@app.get("/api/config/defaults")
async def get_defaults():
    return DEFAULT_CONFIG


@app.get("/api/config/schema")
async def get_schema():
    return {"fields": CONFIG_SCHEMA, "category_order": _CATEGORY_ORDER}


_EMPTY_MODEL_INFO: dict = {
    "model": "",
    "provider": "",
    "auto_context_length": 0,
    "config_context_length": 0,
    "effective_context_length": 0,
    "capabilities": {},
}


@app.get("/api/model/info")
def get_model_info(profile: Optional[str] = None):
    """Return resolved model metadata for the currently configured model.

    Calls the same context-length resolution chain the agent uses, so the
    frontend can display "Auto-detected: 200K" alongside the override field.
    Also returns model capabilities (vision, reasoning, tools) when available.
    """
    try:
        with _profile_scope(profile):
            cfg = load_config()
        model_cfg = cfg.get("model", "")

        # Extract model name and provider from the config
        if isinstance(model_cfg, dict):
            model_name = model_cfg.get("default", model_cfg.get("name", ""))
            provider = model_cfg.get("provider", "")
            base_url = model_cfg.get("base_url", "")
            config_ctx = model_cfg.get("context_length")
        else:
            model_name = str(model_cfg) if model_cfg else ""
            provider = ""
            base_url = ""
            config_ctx = None

        if not model_name:
            return dict(_EMPTY_MODEL_INFO, provider=provider)

        # Resolve auto-detected context length (pass config_ctx=None to get
        # purely auto-detected value, then separately report the override)
        try:
            from agent.model_metadata import get_model_context_length
            auto_ctx = get_model_context_length(
                model=model_name,
                base_url=base_url,
                provider=provider,
                config_context_length=None,  # ignore override — we want auto value
            )
        except Exception:
            auto_ctx = 0

        config_ctx_int = 0
        if isinstance(config_ctx, int) and config_ctx > 0:
            config_ctx_int = config_ctx

        # Effective is what the agent actually uses
        effective_ctx = config_ctx_int if config_ctx_int > 0 else auto_ctx

        # Try to get model capabilities from models.dev
        caps = {}
        try:
            from agent.models_dev import get_model_capabilities
            mc = get_model_capabilities(provider=provider, model=model_name)
            if mc is not None:
                caps = {
                    "supports_tools": mc.supports_tools,
                    "supports_vision": mc.supports_vision,
                    "supports_reasoning": mc.supports_reasoning,
                    "context_window": mc.context_window,
                    "max_output_tokens": mc.max_output_tokens,
                    "model_family": mc.model_family,
                }
        except Exception:
            pass

        return {
            "model": model_name,
            "provider": provider,
            "auto_context_length": auto_ctx,
            "config_context_length": config_ctx_int,
            "effective_context_length": effective_ctx,
            "capabilities": caps,
        }
    except HTTPException:
        # Unknown/invalid profile must surface as 404, not degrade into a
        # 200 with empty model info (which would render as "no model set").
        raise
    except Exception:
        _log.exception("GET /api/model/info failed")
        return dict(_EMPTY_MODEL_INFO)


# ---------------------------------------------------------------------------
# Model assignment — pick provider+model for main slot or auxiliary slots.
# Mirrors the model.options JSON-RPC from tui_gateway but uses REST so the
# Models page (which has no chat PTY open) can drive it.
# ---------------------------------------------------------------------------

# Canonical auxiliary task slots. Keep in sync with DEFAULT_CONFIG["auxiliary"]
# in hermes_cli/config.py — listed here for deterministic ordering in the UI.
_AUX_TASK_SLOTS: Tuple[str, ...] = (
    "vision",
    "web_extract",
    "compression",
    "skills_hub",
    "approval",
    "mcp",
    "title_generation",
    "triage_specifier",
    "kanban_decomposer",
    "profile_describer",
    "curator",
)


@app.get("/api/model/options")
def get_model_options(profile: Optional[str] = None, refresh: bool = False):
    """Return authenticated providers + their curated model lists.

    REST equivalent of the ``model.options`` JSON-RPC on tui_gateway, so the
    dashboard Models page can render the picker without a live chat session.
    The response shape matches ``model.options`` 1:1 so ``ModelPickerDialog``
    can share the same types.

    ``profile`` scopes the picker context (current model/provider, custom
    providers from config, per-profile .env auth state) so the Models page
    reads the SAME profile /api/model/set writes.

    ``refresh`` busts the per-provider model-id disk cache so every row
    re-fetches its live catalog — used by the picker's explicit "Refresh
    Models" control. Normal opens leave it false to stay on the 1h cache.
    """
    try:
        from hermes_cli.inventory import build_models_payload, load_picker_context

        # include_unconfigured + picker_hints + canonical_order mirror the
        # tui_gateway `model.options` JSON-RPC handler exactly, so every GUI
        # surface fed by this endpoint (Settings → Model, the first-run
        # onboarding picker) sees the SAME full provider universe `hermes model`
        # exposes — not just the authenticated subset. Unconfigured providers
        # come back as skeleton rows carrying `authenticated=False` +
        # `auth_type`/`key_env`/`warning` so the GUI can render a setup
        # affordance instead of hiding the provider entirely.
        with _profile_scope(profile):
            return build_models_payload(
                load_picker_context(),
                include_unconfigured=True,
                picker_hints=True,
                canonical_order=True,
                pricing=True,
                capabilities=True,
                refresh=bool(refresh),
            )
    except HTTPException:
        raise
    except Exception:
        _log.exception("GET /api/model/options failed")
        raise HTTPException(status_code=500, detail="Failed to list model options")


@app.get("/api/model/recommended-default")
def get_recommended_default_model(provider: str = ""):
    """Return the recommended default model for a freshly-authenticated provider.

    Mirrors the model-curation `hermes model` does so GUI onboarding lands on a
    sensible default instead of blindly taking the first curated entry. For
    Nous this honors the user's free/paid tier: free users get a free model,
    paid users get the full curated default. For any other provider it falls
    back to the first curated model (same as before).

    Response: {"provider": str, "model": str, "free_tier": bool | None}
    where free_tier is True/False for Nous and None otherwise. `model` may be
    empty if nothing could be resolved (caller degrades gracefully).
    """
    slug = (provider or "").strip().lower()

    if slug == "nous":
        try:
            from hermes_cli.models import (
                get_curated_nous_model_ids,
                get_pricing_for_provider,
                check_nous_free_tier,
                partition_nous_models_by_tier,
                union_with_portal_free_recommendations,
                union_with_portal_paid_recommendations,
            )
            from hermes_cli.auth import get_provider_auth_state

            model_ids = get_curated_nous_model_ids()
            pricing = get_pricing_for_provider("nous") or {}
            free_tier = check_nous_free_tier(force_fresh=True)

            portal_url = ""
            try:
                state = get_provider_auth_state("nous") or {}
                portal_url = state.get("portal_base_url", "") or ""
            except Exception:
                portal_url = ""

            if free_tier:
                model_ids, pricing = union_with_portal_free_recommendations(
                    model_ids, pricing, portal_url
                )
                model_ids, _unavailable = partition_nous_models_by_tier(
                    model_ids, pricing, free_tier=True
                )
            else:
                model_ids, pricing = union_with_portal_paid_recommendations(
                    model_ids, pricing, portal_url
                )

            model = model_ids[0] if model_ids else ""
            return {"provider": "nous", "model": model, "free_tier": bool(free_tier)}
        except Exception:
            _log.exception("GET /api/model/recommended-default (nous) failed")
            return {"provider": "nous", "model": "", "free_tier": None}

    # Non-Nous: first curated model for the provider, matching prior behaviour.
    try:
        from hermes_cli.inventory import build_models_payload, load_picker_context

        payload = build_models_payload(load_picker_context())
        for row in payload.get("providers", []):
            if str(row.get("slug", "")).lower() == slug:
                models = row.get("models") or []
                return {"provider": slug, "model": models[0] if models else "", "free_tier": None}
        return {"provider": slug, "model": "", "free_tier": None}
    except Exception:
        _log.exception("GET /api/model/recommended-default failed")
        return {"provider": slug, "model": "", "free_tier": None}


@app.get("/api/model/auxiliary")
def get_auxiliary_models(profile: Optional[str] = None):
    """Return current auxiliary task assignments.

    Shape:
      {
        "tasks": [
          {"task": "vision", "provider": "auto", "model": "", "base_url": ""},
          ...
        ],
        "main": {"provider": "openrouter", "model": "anthropic/claude-opus-4.7"},
      }

    ``profile`` scopes the read — without it, the Models page would show
    the dashboard profile's auxiliary pins while /api/model/set wrote the
    selected profile's (read/write asymmetry).
    """
    try:
        with _profile_scope(profile):
            cfg = load_config()
        aux_cfg = cfg.get("auxiliary", {})
        if not isinstance(aux_cfg, dict):
            aux_cfg = {}

        tasks = []
        for slot in _AUX_TASK_SLOTS:
            slot_cfg = aux_cfg.get(slot, {}) if isinstance(aux_cfg.get(slot), dict) else {}
            tasks.append({
                "task": slot,
                "provider": str(slot_cfg.get("provider", "auto") or "auto"),
                "model": str(slot_cfg.get("model", "") or ""),
                "base_url": str(slot_cfg.get("base_url", "") or ""),
            })

        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, dict):
            main = {
                "provider": str(model_cfg.get("provider", "") or ""),
                "model": str(model_cfg.get("default", model_cfg.get("name", "")) or ""),
            }
        else:
            main = {"provider": "", "model": str(model_cfg) if model_cfg else ""}

        return {"tasks": tasks, "main": main}
    except HTTPException:
        raise
    except Exception:
        _log.exception("GET /api/model/auxiliary failed")
        raise HTTPException(status_code=500, detail="Failed to read auxiliary config")


@app.get("/api/model/moa")
def get_moa_models(profile: Optional[str] = None):
    """Return the configured Mixture-of-Agents provider/model slots."""
    try:
        from hermes_cli.moa_config import normalize_moa_config

        with _profile_scope(profile):
            cfg = load_config()
            return normalize_moa_config(cfg.get("moa") if isinstance(cfg, dict) else {})
    except HTTPException:
        raise
    except Exception:
        _log.exception("GET /api/model/moa failed")
        raise HTTPException(status_code=500, detail="Failed to read MoA config")


@app.put("/api/model/moa")
def set_moa_models(body: MoaConfigPayload, profile: Optional[str] = None):
    """Persist the Mixture-of-Agents provider/model slots."""
    try:
        from hermes_cli.moa_config import normalize_moa_config

        with _profile_scope(body.profile or profile):
            cfg = load_config()
            if body.presets:
                raw = {
                    "default_preset": body.default_preset,
                    "active_preset": body.active_preset,
                    "presets": {
                        name: {
                            "reference_models": [slot.dict() for slot in preset.reference_models],
                            "aggregator": preset.aggregator.dict(),
                            "reference_temperature": preset.reference_temperature,
                            "aggregator_temperature": preset.aggregator_temperature,
                            "max_tokens": preset.max_tokens,
                            "enabled": preset.enabled,
                        }
                        for name, preset in body.presets.items()
                    },
                }
            else:
                raw = {
                    "reference_models": [slot.dict() for slot in body.reference_models],
                    "aggregator": body.aggregator.dict(),
                    "reference_temperature": body.reference_temperature,
                    "aggregator_temperature": body.aggregator_temperature,
                    "max_tokens": body.max_tokens,
                    "enabled": body.enabled,
                }
            normalized = normalize_moa_config(raw)
            cfg["moa"] = normalized
            save_config(cfg)
            return {"ok": True, **normalized}
    except HTTPException:
        raise
    except Exception:
        _log.exception("PUT /api/model/moa failed")
        raise HTTPException(status_code=500, detail="Failed to save MoA config")


@app.post("/api/model/set")
async def set_model_assignment(body: ModelAssignment, profile: Optional[str] = None):
    """Assign a model to the main slot or an auxiliary task slot.

    Writes to ``~/.hermes/config.yaml`` — applies to **new** sessions only.
    The currently running chat PTY (if any) is not affected; use the
    ``/model`` slash command inside a chat to hot-swap that specific session.
    """
    scope = (body.scope or "").strip().lower()
    provider = (body.provider or "").strip()
    model = (body.model or "").strip()
    task = (body.task or "").strip().lower()
    base_url = (body.base_url or "").strip()
    api_key = (body.api_key or "").strip()

    if scope not in {"main", "auxiliary"}:
        raise HTTPException(status_code=400, detail="scope must be 'main' or 'auxiliary'")

    try:
        # Expensive-model warning runs BEFORE the profile scope is entered:
        # _profile_scope must never be held across an await (the RLock is
        # reentrant per-thread, so a second coroutine interleaving on the
        # event-loop thread could cross-restore the module globals).
        if model and not body.confirm_expensive_model:
            try:
                from hermes_cli.model_cost_guard import expensive_model_warning

                # Pricing lookup can hit models.dev / a /models endpoint on a
                # cache miss — keep it off the event loop.
                warning = await asyncio.to_thread(
                    expensive_model_warning,
                    model,
                    provider=provider,
                    base_url=base_url,
                )
            except Exception:
                warning = None
            if warning is not None:
                return {
                    "ok": False,
                    "scope": scope,
                    "provider": provider,
                    "model": model,
                    "confirm_required": True,
                    "confirm_message": warning.message,
                }

        def _apply_assignment():
            with _profile_scope(body.profile or profile):
                return _apply_model_assignment_sync(
                    scope, provider, model, task, base_url, api_key
                )

        return await asyncio.to_thread(_apply_assignment)
    except HTTPException:
        raise
    except Exception:
        _log.exception("POST /api/model/set failed")
        raise HTTPException(status_code=500, detail="Failed to save model assignment")


def _apply_model_assignment_sync(
    scope: str, provider: str, model: str, task: str, base_url: str, api_key: str = ""
):
    """Synchronous body of POST /api/model/set.

    Runs inside ``_profile_scope`` (in a worker thread) so every
    load_config/save_config lands in the requested profile.  Raises
    HTTPException for validation errors — the async wrapper re-raises them.
    """
    cfg = load_config()

    if scope == "main":
        if not provider or not model:
            raise HTTPException(status_code=400, detail="provider and model required for main")
        provider, model = _normalize_main_model_assignment(provider, model)
        model_cfg = _apply_main_model_assignment(
            cfg.get("model", {}), provider, model, base_url, api_key
        )
        cfg["model"] = model_cfg

        # When switching the main provider to Nous, mirror the CLI's
        # post-model-selection behaviour (hermes_cli/main.py
        # prompt_enable_tool_gateway / tools_config apply_nous_managed_defaults):
        # auto-route any *unconfigured* tools through the Nous Tool Gateway.
        # This is purely additive — apply_nous_managed_defaults skips every
        # tool where the user already has a direct key (FIRECRAWL_API_KEY,
        # FAL_KEY, etc.) or an explicit backend/provider in config, so it
        # never overwrites a user's own setup. GUI users thus land on the
        # gateway the same way CLI users do, without a separate prompt.
        gateway_tools: list[str] = []
        if provider.strip().lower() == "nous":
            try:
                from hermes_cli.nous_subscription import apply_nous_managed_defaults
                from hermes_cli.tools_config import _get_platform_tools

                enabled = _get_platform_tools(
                    cfg, "cli", include_default_mcp_servers=False
                )
                changed = apply_nous_managed_defaults(
                    cfg,
                    enabled_toolsets=enabled,
                    force_fresh=True,
                )
                gateway_tools = sorted(changed)
            except Exception:
                # Portal lookup hiccups / non-subscriber / non-nous gating
                # must never block saving the model assignment.
                _log.debug("apply_nous_managed_defaults skipped", exc_info=True)

        save_config(cfg)

        # Register a named ``custom_providers`` entry for a custom/local
        # endpoint, mirroring the ``hermes model`` custom flow
        # (_save_custom_provider). Without this the endpoint only lives in
        # ``model.*`` and the picker has no proper ready row for it — the
        # GUI then surfaces a "needs setup" dead-end on the bare ``custom``
        # provider. Dedups by base_url, so re-saving is idempotent.
        if provider.strip().lower() in {"custom", "local"} and base_url:
            try:
                from hermes_cli.main import _auto_provider_name, _save_custom_provider

                _save_custom_provider(
                    base_url,
                    api_key,
                    model,
                    name=_auto_provider_name(base_url),
                )
            except Exception:
                # Never block the assignment on the bookkeeping write —
                # model.* is already persisted and routable.
                _log.debug("custom_providers registration skipped", exc_info=True)

        # Surface auxiliary slots still pinned to a *different* provider than
        # the new main one. Switching the main model does NOT touch aux pins
        # (they're independent, sticky per-task overrides — see
        # auxiliary_client._resolve_auto). A user who switches main away from
        # a now-unpaid provider (e.g. nous with $0 balance) keeps paying 402s
        # on every background aux call until they reset those pins. We never
        # auto-clear them — pinning aux to a cheaper/different model is a
        # legitimate config — but we tell the caller so the UI can offer a
        # "reset to main" nudge instead of silently burning credits.
        new_provider = provider.strip().lower()
        stale_aux: list[dict] = []
        aux_cfg = cfg.get("auxiliary", {})
        if isinstance(aux_cfg, dict):
            for slot in _AUX_TASK_SLOTS:
                slot_cfg = aux_cfg.get(slot)
                if not isinstance(slot_cfg, dict):
                    continue
                slot_provider = str(slot_cfg.get("provider", "") or "").strip()
                if (
                    slot_provider
                    and slot_provider.lower() not in {"auto", ""}
                    and slot_provider.lower() != new_provider
                ):
                    stale_aux.append({
                        "task": slot,
                        "provider": slot_provider,
                        "model": str(slot_cfg.get("model", "") or ""),
                    })

        return {
            "ok": True,
            "scope": "main",
            "provider": provider,
            "model": model,
            "base_url": model_cfg.get("base_url", ""),
            "gateway_tools": gateway_tools,
            "stale_aux": stale_aux,
        }

    # scope == "auxiliary"
    aux = cfg.get("auxiliary")
    if not isinstance(aux, dict):
        aux = {}

    if task == "__reset__":
        # Reset every slot to provider="auto", model="" — keeps other fields intact.
        for slot in _AUX_TASK_SLOTS:
            slot_cfg = aux.get(slot)
            if not isinstance(slot_cfg, dict):
                slot_cfg = {}
            slot_cfg["provider"] = "auto"
            slot_cfg["model"] = ""
            slot_cfg.pop("base_url", None)
            clear_model_endpoint_credentials(slot_cfg)
            aux[slot] = slot_cfg
        cfg["auxiliary"] = aux
        save_config(cfg)
        return {"ok": True, "scope": "auxiliary", "reset": True}

    if not provider:
        raise HTTPException(status_code=400, detail="provider required for auxiliary")

    targets = [task] if task else list(_AUX_TASK_SLOTS)
    for slot in targets:
        if slot not in _AUX_TASK_SLOTS:
            raise HTTPException(status_code=400, detail=f"unknown auxiliary task: {slot}")
        slot_cfg = aux.get(slot)
        if not isinstance(slot_cfg, dict):
            slot_cfg = {}
        prev_provider = str(slot_cfg.get("provider") or "").strip().lower()
        new_provider = provider.strip().lower()
        slot_cfg["provider"] = provider
        slot_cfg["model"] = model
        if new_provider != prev_provider and new_provider != "custom":
            slot_cfg.pop("base_url", None)
            clear_model_endpoint_credentials(slot_cfg)
        aux[slot] = slot_cfg

    cfg["auxiliary"] = aux
    save_config(cfg)
    return {
        "ok": True,
        "scope": "auxiliary",
        "tasks": targets,
        "provider": provider,
        "model": model,
    }




def _infer_provider_on_model_change(model_val: str, prev_provider: str) -> tuple[str, str]:
    """Infer which provider serves ``model_val`` when the flat Config-page Model
    field changes, given the previously-saved ``prev_provider``.

    Returns ``(provider, model)``; ``provider`` is empty when no switch is
    warranted (leave the existing provider untouched). Two signals, in order:

    1. Curated-catalog detection (``detect_provider_for_model``) — handles the
       ~28 OpenRouter-curated models and direct provider-static catalogs.
    2. Vendor-slug heuristic — a ``vendor/model`` slug cannot belong to a
       single-model / non-aggregator provider (e.g. ``ollama-local``). When the
       current provider is not an aggregator that serves vendor-prefixed slugs,
       route to an aggregator. ``_normalize_main_model_assignment`` (called by
       the caller) keeps the user's current aggregator when they're already on
       one, else falls back to openrouter — the same chokepoint logic as
       ``POST /api/model/set``.
    """
    name = (model_val or "").strip()
    if not name:
        return "", name
    try:
        from hermes_cli.models import (
            _AGGREGATOR_PROVIDERS,
            detect_provider_for_model,
            normalize_provider,
        )
    except Exception:
        return "", name

    try:
        detected = detect_provider_for_model(name, prev_provider)
    except Exception:
        detected = None
    if detected:
        return detected[0], detected[1]

    # Vendor-prefixed slug under a non-aggregator provider → reassign. Use a
    # sentinel "openrouter" here; _normalize_main_model_assignment resolves the
    # real aggregator (keeps a current aggregator, else openrouter).
    if "/" in name:
        try:
            cur_is_aggregator = normalize_provider(prev_provider) in _AGGREGATOR_PROVIDERS
        except Exception:
            cur_is_aggregator = False
        if not cur_is_aggregator:
            return "openrouter", name

    return "", name


def _denormalize_config_from_web(config: Dict[str, Any]) -> Dict[str, Any]:
    """Reverse _normalize_config_for_web before saving.

    Reconstructs ``model`` as a dict by reading the current on-disk config
    to recover model subkeys (provider, base_url, api_mode, etc.) that were
    stripped from the GET response.  The frontend only sees model as a flat
    string; the rest is preserved transparently.

    Also handles ``model_context_length`` — writes it back into the model dict
    as ``context_length``.  A value of 0 or absent means "auto-detect" (omitted
    from the dict so get_model_context_length() uses its normal resolution).
    """
    config = dict(config)
    # Remove any _model_meta that might have leaked in (shouldn't happen
    # with the stripped GET response, but be defensive)
    config.pop("_model_meta", None)

    # Extract and remove model_context_length before processing model
    ctx_override = config.pop("model_context_length", 0)
    if not isinstance(ctx_override, int):
        try:
            ctx_override = int(ctx_override)
        except (TypeError, ValueError):
            ctx_override = 0

    model_val = config.get("model")
    if isinstance(model_val, str) and model_val:
        # Read the current disk config to recover model subkeys
        try:
            disk_config = load_config()
            disk_model = disk_config.get("model")
            if isinstance(disk_model, dict):
                prev_default = str(disk_model.get("default") or "").strip()
                prev_provider = str(disk_model.get("provider") or "").strip()
                # When the model name actually changed, re-detect which
                # provider serves it. The Config-page Model field is a flat
                # string with no provider info, so without this a user who
                # picks an OpenRouter model while their default provider is
                # ollama-local keeps the stale provider and 404s. Only fires
                # on a real model change so saving unrelated config fields
                # never overwrites an explicit provider.
                if model_val != prev_default and prev_provider:
                    new_provider, resolved_model = _infer_provider_on_model_change(
                        model_val, prev_provider
                    )
                    if new_provider and new_provider.strip().lower() != prev_provider.lower():
                        # Route through the canonical assignment chokepoints so
                        # the model is normalized for the new provider and stale
                        # base_url/api_mode/api_key are cleared on the switch
                        # (and preserved on a same-provider re-pick).
                        norm_provider, norm_model = _normalize_main_model_assignment(
                            new_provider, resolved_model
                        )
                        disk_model = _apply_main_model_assignment(
                            disk_model, norm_provider, norm_model
                        )
                        model_val = norm_model
                # Preserve all subkeys, update default with the new value
                disk_model["default"] = model_val
                # Write context_length into the model dict (0 = remove/auto)
                if ctx_override > 0:
                    disk_model["context_length"] = ctx_override
                else:
                    disk_model.pop("context_length", None)
                config["model"] = disk_model
            # Model was previously a bare string — upgrade to dict if
            # user is setting a context_length override
            elif ctx_override > 0:
                config["model"] = {
                    "default": model_val,
                    "context_length": ctx_override,
                }
        except Exception:
            pass  # can't read disk config — just use the string form
    return config


@app.put("/api/config")
async def update_config(body: ConfigUpdate, profile: Optional[str] = None):
    try:
        with _profile_scope(body.profile or profile):
            # The dashboard form is schema-driven (see CONFIG_SCHEMA). Any root
            # key absent from the schema — most visibly ``custom_providers``, but
            # also ``agent.personalities``, ``terminal.lifetime_seconds``, etc. —
            # is not sent in the PUT body. A full-replace save would silently
            # drop those keys. Deep-merge incoming over what's on disk so the
            # frontend can only overwrite what it explicitly sends.
            existing = read_raw_config()
            incoming = _denormalize_config_from_web(body.config)
            save_config(_deep_merge(existing, incoming))
        return {"ok": True}
    except HTTPException:
        raise
    except Exception:
        _log.exception("PUT /api/config failed")
        raise HTTPException(status_code=500, detail="Internal server error")


def _catalog_provider_env_metadata() -> dict:
    """Map provider env vars → desktop card metadata, derived from the catalog.

    Returns ``{env_var: {provider, provider_label, description, url, is_password,
    advanced}}`` for every API-key provider in the unified ``provider_catalog()``
    (i.e. the ``hermes model`` universe). This is what lets the desktop Keys tab
    render a card for a provider even when its env var was never hand-added to
    ``OPTIONAL_ENV_VARS`` — closing the drift where CLI-configurable providers
    (openai-api, kilocode, novita, tencent-tokenhub, copilot, …) were missing
    from the GUI.

    Hand ``OPTIONAL_ENV_VARS`` prose is layered ON TOP of this in the endpoint;
    this only supplies membership + grouping + sensible fallbacks.
    """
    try:
        from hermes_cli.provider_catalog import provider_catalog
    except Exception:
        return {}

    # Env vars already declared with a NON-provider category (e.g. the shared
    # GITHUB_TOKEN, which is a Skills-Hub "tool" credential) must not be
    # promoted into a provider card. Copilot lists GITHUB_TOKEN among its auth
    # aliases, but its provider card uses the provider-owned COPILOT_GITHUB_TOKEN.
    try:
        from hermes_cli.config import OPTIONAL_ENV_VARS as _OPT
    except Exception:
        _OPT = {}
    _non_provider_keys = {
        k for k, v in _OPT.items()
        if (v or {}).get("category") and (v or {}).get("category") != "provider"
    }

    meta: dict = {}
    for d in provider_catalog():
        if d.tab != "keys":
            continue
        # API-key vars: the first is the primary (password) field; any aliases
        # are kept as additional password fields so users can clear them too.
        for env_var in d.api_key_env_vars:
            if env_var in _non_provider_keys:
                continue  # don't hijack a shared tool/messaging credential
            meta.setdefault(
                env_var,
                {
                    "provider": d.slug,
                    "provider_label": d.label,
                    "description": d.description,
                    "url": d.signup_url or None,
                    "is_password": True,
                    "advanced": False,
                    "category": "provider",
                },
            )
        # Base-URL override is an advanced, non-secret field for the same card.
        if d.base_url_env_var:
            meta.setdefault(
                d.base_url_env_var,
                {
                    "provider": d.slug,
                    "provider_label": d.label,
                    "description": f"{d.label} base URL override",
                    "url": None,
                    "is_password": False,
                    "advanced": True,
                    "category": "provider",
                },
            )

        # AWS-SDK providers (Bedrock) authenticate via the AWS credential chain
        # rather than a pasted API key, so they have no api_key_env_vars. Tag
        # their AWS_* settings to the provider card so they still appear on the
        # Keys tab (otherwise Bedrock — a `hermes model` provider — would be
        # invisible in the desktop app).
        if d.auth_type == "aws_sdk":
            for aws_var in ("AWS_REGION", "AWS_PROFILE"):
                existing = meta.get(aws_var, {})
                meta[aws_var] = {
                    "provider": d.slug,
                    "provider_label": d.label,
                    "description": existing.get("description") or f"{d.label} ({aws_var})",
                    "url": existing.get("url"),
                    "is_password": False,
                    "advanced": existing.get("advanced", True),
                    "category": "provider",
                }

        # Vertex AI authenticates via OAuth2 (service-account JSON or ADC), not a
        # pasted API key, so it also has no api_key_env_vars. Tag its credential
        # env var to the provider card so it appears on the Keys tab (otherwise
        # Vertex — a `hermes model` provider — would be invisible in the desktop
        # app). The value is a filesystem path, not a secret string, so it is
        # not a password field.
        if d.auth_type == "vertex":
            existing = meta.get("VERTEX_CREDENTIALS_PATH", {})
            meta["VERTEX_CREDENTIALS_PATH"] = {
                "provider": d.slug,
                "provider_label": d.label,
                "description": existing.get("description")
                or f"{d.label} — service account JSON path (or use ADC)",
                "url": existing.get("url"),
                "is_password": False,
                "advanced": existing.get("advanced", True),
                "category": "provider",
            }
    return meta


@app.get("/api/env")
async def get_env_vars(profile: Optional[str] = None):
    with _profile_scope(profile):
        env_on_disk = load_env()
    channel_keys = _channel_managed_env_keys()
    catalog_meta = _catalog_provider_env_metadata()

    def _row(var_name: str, info: dict, *, custom: bool = False) -> dict:
        value = env_on_disk.get(var_name)
        cat_meta = catalog_meta.get(var_name) or {}
        # Hand OPTIONAL_ENV_VARS prose wins where present; the catalog fills any
        # gaps (description/url) and always supplies provider grouping hints.
        return {
            "is_set": bool(value),
            "redacted_value": redact_key(value) if value else None,
            "description": info.get("description") or cat_meta.get("description", ""),
            "url": info.get("url") if info.get("url") is not None else cat_meta.get("url"),
            "category": info.get("category") or cat_meta.get("category", ""),
            "is_password": info.get("password", cat_meta.get("is_password", False)),
            "tools": info.get("tools", []),
            "advanced": info.get("advanced", cat_meta.get("advanced", False)),
            # True when this var is a messaging-platform credential owned by a
            # Channels page card. The Keys/Env page uses this to hide it and
            # avoid duplicating the (richer) Channels configuration UI.
            "channel_managed": var_name in channel_keys,
            # Provider grouping hints derived from the unified provider catalog
            # so the desktop Keys tab groups by the SAME provider identity the
            # CLI `hermes model` picker uses (not desktop-only prefix guesses).
            "provider": cat_meta.get("provider", ""),
            "provider_label": cat_meta.get("provider_label", ""),
            # True when this key exists in the user's .env but is NOT in any
            # catalog (OPTIONAL_ENV_VARS or the provider catalog) — an
            # arbitrary/custom env var the user added directly. Surfaced so the
            # Keys page can list (and let the user manage) them instead of
            # hiding everything it doesn't recognise.
            "custom": custom,
        }

    result = {}
    for var_name, info in OPTIONAL_ENV_VARS.items():
        result[var_name] = _row(var_name, info)
    # Synthesize rows for catalog provider env vars that have no hand entry in
    # OPTIONAL_ENV_VARS — these are the providers that were CLI-configurable but
    # invisible in the desktop app until now.
    for var_name in catalog_meta:
        if var_name not in result:
            result[var_name] = _row(var_name, {})
    # Surface arbitrary/custom keys the user set in .env that aren't in any
    # catalog. These are always "set" (they're on disk). Treated as secrets by
    # default (is_password=True → redacted, reveal-gated) since an unrecognised
    # key could hold anything. Channel-managed credentials are excluded — those
    # belong to the Channels page. This makes the "add a custom key" surface
    # round-trip: a key added there reappears here under its own section.
    for var_name in env_on_disk:
        if var_name in result or var_name in channel_keys:
            continue
        row = _row(var_name, {}, custom=True)
        row["category"] = "custom"
        row["is_password"] = True
        result[var_name] = row
    return result


@app.put("/api/env")
async def set_env_var(body: EnvVarUpdate, profile: Optional[str] = None):
    try:
        with _profile_scope(body.profile or profile):
            save_env_value(body.key, body.value)
        return {"ok": True, "key": body.key}
    except ValueError as exc:
        # save_env_value raises ValueError for invalid names and for keys
        # on the denylist (LD_PRELOAD, PATH, PYTHONPATH, …). Surface the
        # message to the SPA so the user understands why the write was
        # refused instead of seeing an opaque 500.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        _log.exception("PUT /api/env failed")
        raise HTTPException(status_code=500, detail="Internal server error")


# Live credential probes keyed by env var. Each entry is (method, url, auth)
# where auth is "bearer" (Authorization header) or "query" (?key=). A cheap
# read-only models/key call that 401s on a bad token — enough to catch a
# mistyped key before it's persisted. Providers absent from this map (or local
# endpoints) are not network-validated; the client treats those as "unknown".
_CREDENTIAL_PROBES: dict[str, tuple[str, str]] = {
    "OPENROUTER_API_KEY": ("https://openrouter.ai/api/v1/key", "bearer"),
    "OPENAI_API_KEY": ("https://api.openai.com/v1/models", "bearer"),
    "XAI_API_KEY": ("https://api.x.ai/v1/models", "bearer"),
    "GEMINI_API_KEY": ("https://generativelanguage.googleapis.com/v1beta/models", "query"),
}


def _parse_model_ids(resp: "Any") -> List[str]:
    """Extract model ids from an OpenAI-compatible ``/v1/models`` response.

    Tolerant of the common shapes: ``{"data": [{"id": ...}]}`` (OpenAI / vLLM /
    llama.cpp) and a bare ``{"data": ["id", ...]}``. Returns ``[]`` on any
    parse/HTTP error so a slightly non-standard endpoint never hard-blocks.
    """
    try:
        if not resp.is_success:
            return []
        payload = resp.json()
    except Exception:
        return []
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        return []
    ids: List[str] = []
    for item in data:
        if isinstance(item, dict):
            mid = str(item.get("id") or "").strip()
        else:
            mid = str(item or "").strip()
        if mid:
            ids.append(mid)
    return ids


@app.post("/api/providers/validate")
async def validate_provider_credential(body: EnvVarUpdate, request: Request):
    """Live-probe a provider credential before it's saved.

    Returns {ok, reachable, message}. ok=True means the provider accepted the
    key; ok=False + reachable=True means the key is bad (caller should block);
    reachable=False means the network probe couldn't run (caller may save with
    a warning rather than hard-blocking offline users).
    """
    _require_token(request)
    import httpx

    key = (body.key or "").strip()
    value = (body.value or "").strip()
    if not value:
        return {"ok": False, "reachable": True, "message": "Enter a value first."}

    # Local / custom endpoint: validate connectivity, not auth — any HTTP
    # response (even 401) proves the endpoint is up. Also surface the model
    # ids the endpoint advertises (OpenAI ``/v1/models`` shape) so the GUI can
    # auto-pick a default without asking the user to type a model name.
    if key == "OPENAI_BASE_URL":
        url = value.rstrip("/") + "/models"
        # Send the optional API key so endpoints that require auth on
        # ``/v1/models`` (many hosted OpenAI-compatible servers) still enumerate
        # their models instead of returning an empty list behind a 401.
        api_key = (body.api_key or "").strip()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        try:
            with httpx.Client(timeout=httpx.Timeout(8.0)) as client:
                resp = client.get(url, headers=headers)
            return {"ok": True, "reachable": True, "message": "", "models": _parse_model_ids(resp)}
        except Exception:
            return {"ok": False, "reachable": False, "message": f"Could not reach {url}."}

    probe = _CREDENTIAL_PROBES.get(key)
    if not probe:
        # No probe for this provider — can't validate, don't block.
        return {"ok": True, "reachable": False, "message": ""}

    url, auth = probe
    headers = {"Accept": "application/json"}
    params = {}
    if auth == "bearer":
        headers["Authorization"] = f"Bearer {value}"
    else:
        params["key"] = value

    try:
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            resp = client.get(url, headers=headers, params=params)
    except Exception:
        return {"ok": False, "reachable": False, "message": "Could not reach the provider to verify the key."}

    if resp.status_code in (401, 403):
        return {"ok": False, "reachable": True, "message": "That API key was rejected. Double-check it and try again."}
    if resp.status_code == 429 or resp.is_success:
        # 429 = key is valid but rate-limited; success = valid.
        return {"ok": True, "reachable": True, "message": ""}
    return {"ok": False, "reachable": True, "message": f"Provider returned HTTP {resp.status_code} for this key."}


@app.delete("/api/env")
async def remove_env_var(body: EnvVarDelete, profile: Optional[str] = None):
    try:
        with _profile_scope(body.profile or profile):
            removed = remove_env_value(body.key)
        if not removed:
            raise HTTPException(status_code=404, detail=f"{body.key} not found in .env")
        return {"ok": True, "key": body.key}
    except HTTPException:
        raise
    except ValueError as exc:
        # remove_env_value raises ValueError for invalid key names. Surface
        # the message to the SPA so the user understands why the delete was
        # refused instead of seeing an opaque 500. Mirrors PUT /api/env.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        _log.exception("DELETE /api/env failed")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/env/reveal")
async def reveal_env_var(
    body: EnvVarReveal, request: Request, profile: Optional[str] = None
):
    """Return the real (unredacted) value of a single env var.

    Protected by:
    - Ephemeral session token (generated per server start, injected into SPA)
    - Rate limiting (max 5 reveals per 30s window)
    - Audit logging
    """
    # --- Token check ---
    _require_token(request)

    # --- Rate limit ---
    now = time.time()
    cutoff = now - _REVEAL_WINDOW_SECONDS
    _reveal_timestamps[:] = [t for t in _reveal_timestamps if t > cutoff]
    if len(_reveal_timestamps) >= _REVEAL_MAX_PER_WINDOW:
        raise HTTPException(status_code=429, detail="Too many reveal requests. Try again shortly.")
    _reveal_timestamps.append(now)

    # --- Reveal ---
    with _profile_scope(body.profile or profile):
        env_on_disk = load_env()
    value = env_on_disk.get(body.key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"{body.key} not found in .env")

    _log.info("env/reveal: %s", body.key)
    return {"key": body.key, "value": value}


# Entries omit fields they don't need to override; the catalog builder fills
# in env_vars from OPTIONAL_ENV_VARS via prefix matching when not specified,
# and pulls required_env from a plugin's PlatformEntry when available.
_PLATFORM_OVERRIDES: dict[str, dict[str, Any]] = {
    "telegram": {
        "name": "Telegram",
        "description": "Run Hermes from Telegram DMs, groups, and topics.",
        "docs_url": "https://core.telegram.org/bots/features#botfather",
        "env_vars": ("TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_USERS", "TELEGRAM_PROXY"),
        "required_env": ("TELEGRAM_BOT_TOKEN",),
    },
    "discord": {
        "name": "Discord",
        "description": "Connect Hermes to Discord DMs, channels, and threads.",
        "docs_url": "https://discord.com/developers/applications",
        "env_vars": (
            "DISCORD_BOT_TOKEN",
            "DISCORD_ALLOWED_USERS",
            "DISCORD_REPLY_TO_MODE",
        ),
        "required_env": ("DISCORD_BOT_TOKEN",),
    },
    "slack": {
        "name": "Slack",
        "description": "Use Hermes from Slack via Socket Mode. Add allowed Slack member IDs so connected bots can respond.",
        "docs_url": "https://api.slack.com/apps",
        "env_vars": ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_ALLOWED_USERS"),
        "required_env": ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"),
    },
    "mattermost": {
        "name": "Mattermost",
        "description": "Connect Hermes to Mattermost channels and direct messages.",
        "docs_url": "https://mattermost.com/deploy/",
        "env_vars": ("MATTERMOST_URL", "MATTERMOST_TOKEN", "MATTERMOST_ALLOWED_USERS"),
        "required_env": ("MATTERMOST_URL", "MATTERMOST_TOKEN"),
    },
    "matrix": {
        "name": "Matrix",
        "description": "Use Hermes in Matrix rooms and direct messages.",
        "docs_url": "https://matrix.org/ecosystem/servers/",
        "env_vars": (
            "MATRIX_HOMESERVER",
            "MATRIX_ACCESS_TOKEN",
            "MATRIX_USER_ID",
            "MATRIX_ALLOWED_USERS",
        ),
        "required_env": ("MATRIX_HOMESERVER", "MATRIX_ACCESS_TOKEN", "MATRIX_USER_ID"),
    },
    "signal": {
        "name": "Signal",
        "description": "Connect through a signal-cli REST bridge.",
        "docs_url": "https://github.com/bbernhard/signal-cli-rest-api",
        "env_vars": ("SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT", "SIGNAL_ALLOWED_USERS"),
        "required_env": ("SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT"),
    },
    "whatsapp": {
        "name": "WhatsApp",
        "description": "Use Hermes through the bundled WhatsApp bridge with QR-based auth.",
        "docs_url": "https://github.com/tulir/whatsmeow",
        "env_vars": ("WHATSAPP_ENABLED", "WHATSAPP_MODE", "WHATSAPP_ALLOWED_USERS"),
        "required_env": (),
    },
    "homeassistant": {
        "name": "Home Assistant",
        "description": "Control your smart home from Hermes via Home Assistant.",
        "docs_url": "https://www.home-assistant.io/docs/authentication/",
        "env_vars": ("HASS_URL", "HASS_TOKEN"),
        "required_env": ("HASS_URL", "HASS_TOKEN"),
    },
    "email": {
        "name": "Email",
        "description": "Talk to Hermes through an IMAP/SMTP mailbox.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/",
        "env_vars": (
            "EMAIL_ADDRESS",
            "EMAIL_PASSWORD",
            "EMAIL_IMAP_HOST",
            "EMAIL_SMTP_HOST",
        ),
        "required_env": (
            "EMAIL_ADDRESS",
            "EMAIL_PASSWORD",
            "EMAIL_IMAP_HOST",
            "EMAIL_SMTP_HOST",
        ),
    },
    "sms": {
        "name": "SMS (Twilio)",
        "description": "Send and receive text messages via Twilio.",
        "docs_url": "https://www.twilio.com/console",
        "env_vars": ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"),
        "required_env": ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"),
    },
    "dingtalk": {
        "name": "DingTalk",
        "description": "Connect Hermes to DingTalk groups (钉钉).",
        "docs_url": "https://open.dingtalk.com/document/orgapp/the-robot-development-process",
        "env_vars": ("DINGTALK_CLIENT_ID", "DINGTALK_CLIENT_SECRET"),
        "required_env": ("DINGTALK_CLIENT_ID", "DINGTALK_CLIENT_SECRET"),
    },
    "feishu": {
        "name": "Feishu / Lark",
        "description": "Use Hermes inside Feishu / Lark.",
        "docs_url": "https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/intro",
        "env_vars": (
            "FEISHU_APP_ID",
            "FEISHU_APP_SECRET",
            "FEISHU_ENCRYPT_KEY",
            "FEISHU_VERIFICATION_TOKEN",
        ),
        "required_env": ("FEISHU_APP_ID", "FEISHU_APP_SECRET"),
    },
    "google_chat": {
        "name": "Google Chat",
        "description": "Connect Hermes to Google Chat via Cloud Pub/Sub.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/google_chat",
    },
    "wecom": {
        "name": "WeCom (group bot)",
        "description": "Send-only WeCom group bot via webhook.",
        "docs_url": "https://developer.work.weixin.qq.com/document/path/91770",
        "env_vars": ("WECOM_BOT_ID", "WECOM_SECRET"),
        "required_env": ("WECOM_BOT_ID",),
    },
    "wecom_callback": {
        "name": "WeCom (app)",
        "description": "Two-way WeCom integration via callback app.",
        "docs_url": "https://developer.work.weixin.qq.com/document/path/90930",
        "env_vars": (
            "WECOM_CALLBACK_CORP_ID",
            "WECOM_CALLBACK_CORP_SECRET",
            "WECOM_CALLBACK_AGENT_ID",
            "WECOM_CALLBACK_TOKEN",
            "WECOM_CALLBACK_ENCODING_AES_KEY",
        ),
        "required_env": (
            "WECOM_CALLBACK_CORP_ID",
            "WECOM_CALLBACK_CORP_SECRET",
            "WECOM_CALLBACK_AGENT_ID",
        ),
    },
    "weixin": {
        "name": "Weixin / WeChat (Personal)",
        "description": "Connect a personal WeChat account through Tencent's iLink Bot API.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/weixin/",
        "env_vars": ("WEIXIN_ACCOUNT_ID", "WEIXIN_TOKEN", "WEIXIN_BASE_URL"),
        "required_env": ("WEIXIN_ACCOUNT_ID", "WEIXIN_TOKEN"),
    },
    "bluebubbles": {
        "name": "BlueBubbles (iMessage)",
        "description": "Use Hermes through iMessage via a BlueBubbles server.",
        "docs_url": "https://bluebubbles.app/",
        "env_vars": (
            "BLUEBUBBLES_SERVER_URL",
            "BLUEBUBBLES_PASSWORD",
            "BLUEBUBBLES_ALLOWED_USERS",
        ),
        "required_env": ("BLUEBUBBLES_SERVER_URL", "BLUEBUBBLES_PASSWORD"),
    },
    "qqbot": {
        "name": "QQ Bot",
        "description": "Connect Hermes to a QQ Bot from the QQ Open Platform.",
        "docs_url": "https://q.qq.com",
        "env_vars": ("QQ_APP_ID", "QQ_CLIENT_SECRET", "QQ_ALLOWED_USERS"),
        "required_env": ("QQ_APP_ID", "QQ_CLIENT_SECRET"),
    },
    # Teams ships as a platform plugin, so its name/env vars come from the
    # plugin registry. Only the docs link needs an override here so the
    # Channels page can point at the Microsoft Teams setup guide.
    "teams": {
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/teams",
    },
    "yuanbao": {
        "name": "Yuanbao (元宝)",
        "description": "Connect Hermes to Tencent Yuanbao.",
        "docs_url": "",
        "required_env": (),
    },
    "api_server": {
        "name": "API server",
        "description": "Expose Hermes as an OpenAI-compatible HTTP API for tools like Open WebUI.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/",
        "env_vars": (
            "API_SERVER_ENABLED",
            "API_SERVER_KEY",
            "API_SERVER_PORT",
            "API_SERVER_HOST",
            "API_SERVER_MODEL_NAME",
        ),
        "required_env": (),
    },
    "webhook": {
        "name": "Webhooks",
        "description": "Receive events from GitHub, GitLab, and other webhook sources.",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/webhooks/",
        "env_vars": ("WEBHOOK_ENABLED", "WEBHOOK_PORT", "WEBHOOK_SECRET"),
        "required_env": (),
    },
}

# Display order: well-known platforms surface first; unknown plugins fall to
# the end alphabetically.
_PLATFORM_ORDER: tuple[str, ...] = (
    "telegram",
    "discord",
    "slack",
    "mattermost",
    "matrix",
    "whatsapp",
    "signal",
    "bluebubbles",
    "homeassistant",
    "email",
    "sms",
    "dingtalk",
    "feishu",
    "google_chat",
    "wecom",
    "wecom_callback",
    "weixin",
    "qqbot",
    "yuanbao",
    "api_server",
    "webhook",
)

# Display labels for env vars not in OPTIONAL_ENV_VARS (HOME_CHANNEL_*, bridge
# toggles, Twilio, HASS, Email, etc.). Anything missing from OPTIONAL_ENV_VARS
# falls back here so the UI can still render a friendly label.
_MESSAGING_ENV_FALLBACKS: dict[str, dict[str, Any]] = {
    "SIGNAL_HTTP_URL": {
        "description": "signal-cli REST API base URL, e.g. http://127.0.0.1:8080",
        "prompt": "Signal bridge URL",
        "url": "https://github.com/bbernhard/signal-cli-rest-api",
    },
    "SIGNAL_ACCOUNT": {
        "description": "Signal account phone number registered with the bridge",
        "prompt": "Signal account",
    },
    "SIGNAL_ALLOWED_USERS": {
        "description": "Comma-separated Signal users allowed to use the bot",
        "prompt": "Allowed Signal users",
    },
    "WHATSAPP_ENABLED": {
        "description": "Enable the WhatsApp gateway adapter",
        "prompt": "Enable WhatsApp",
        "advanced": True,
    },
    "WHATSAPP_MODE": {
        "description": "WhatsApp bridge mode",
        "prompt": "WhatsApp mode",
        "advanced": True,
    },
    "WHATSAPP_ALLOWED_USERS": {
        "description": "Comma-separated WhatsApp users allowed to use the bot",
        "prompt": "Allowed WhatsApp users",
    },
    "HASS_URL": {
        "description": "Home Assistant base URL, e.g. https://homeassistant.local:8123",
        "prompt": "Home Assistant URL",
    },
    "HASS_TOKEN": {
        "description": "Long-lived access token from Home Assistant (Profile → Security)",
        "prompt": "Home Assistant access token",
        "password": True,
    },
    "EMAIL_ADDRESS": {
        "description": "Email address to send and receive from",
        "prompt": "Email address",
    },
    "EMAIL_PASSWORD": {
        "description": "Email account password or app password",
        "prompt": "Email password",
        "password": True,
    },
    "EMAIL_IMAP_HOST": {
        "description": "IMAP server host (e.g. imap.gmail.com)",
        "prompt": "IMAP host",
    },
    "EMAIL_SMTP_HOST": {
        "description": "SMTP server host (e.g. smtp.gmail.com)",
        "prompt": "SMTP host",
    },
    "TWILIO_ACCOUNT_SID": {
        "description": "Twilio Account SID",
        "prompt": "Twilio Account SID",
        "url": "https://www.twilio.com/console",
    },
    "TWILIO_AUTH_TOKEN": {
        "description": "Twilio Auth Token",
        "prompt": "Twilio Auth Token",
        "password": True,
    },
    "WECOM_BOT_ID": {"description": "WeCom group bot ID", "prompt": "WeCom Bot ID"},
    "WECOM_SECRET": {
        "description": "WeCom group bot secret",
        "prompt": "WeCom Secret",
        "password": True,
    },
    "WECOM_CALLBACK_CORP_ID": {
        "description": "WeCom corp ID",
        "prompt": "WeCom Corp ID",
    },
    "WECOM_CALLBACK_CORP_SECRET": {
        "description": "WeCom app corp secret",
        "prompt": "WeCom Corp Secret",
        "password": True,
    },
    "WECOM_CALLBACK_AGENT_ID": {
        "description": "WeCom app agent ID",
        "prompt": "WeCom Agent ID",
    },
    "WECOM_CALLBACK_TOKEN": {
        "description": "WeCom callback verification token",
        "prompt": "WeCom Token",
    },
    "WECOM_CALLBACK_ENCODING_AES_KEY": {
        "description": "WeCom callback AES encoding key",
        "prompt": "WeCom AES Key",
        "password": True,
    },
    "WEIXIN_ACCOUNT_ID": {
        "description": "iLink Bot account ID obtained through QR login in hermes gateway setup",
        "prompt": "iLink Bot account ID",
    },
    "WEIXIN_TOKEN": {
        "description": "iLink Bot token obtained through QR login in hermes gateway setup",
        "prompt": "iLink Bot token",
        "password": True,
    },
    "WEIXIN_BASE_URL": {
        "description": "iLink API base URL saved by QR login (default: https://ilinkai.weixin.qq.com)",
        "prompt": "iLink API base URL",
    },
    "FEISHU_APP_ID": {"description": "Feishu / Lark app ID", "prompt": "App ID"},
    "FEISHU_APP_SECRET": {
        "description": "Feishu / Lark app secret",
        "prompt": "App secret",
        "password": True,
    },
    "FEISHU_ENCRYPT_KEY": {
        "description": "Feishu / Lark encrypt key",
        "prompt": "Encrypt key",
        "password": True,
    },
    "FEISHU_VERIFICATION_TOKEN": {
        "description": "Feishu / Lark verification token",
        "prompt": "Verification token",
        "password": True,
    },
    "DINGTALK_CLIENT_ID": {
        "description": "DingTalk client ID (App key)",
        "prompt": "Client ID",
    },
    "DINGTALK_CLIENT_SECRET": {
        "description": "DingTalk client secret (App secret)",
        "prompt": "Client secret",
        "password": True,
    },
}


def _messaging_platform_catalog() -> tuple[dict[str, Any], ...]:
    """Build the messaging catalog from the gateway's Platform enum + plugin registry.

    Built-in platforms come from ``gateway.config.Platform`` (LOCAL is excluded).
    Plugin platforms come from ``gateway.platform_registry.plugin_entries()``,
    which lets newly installed adapters (e.g. IRC) appear without a code change
    here. Per-platform UI metadata (description, docs URL, env-var picks) lives
    in :data:`_PLATFORM_OVERRIDES`; anything not overridden gets reasonable
    defaults derived from the platform id and required_env.
    """
    from gateway.config import Platform

    seen: set[str] = set()
    entries: list[dict[str, Any]] = []

    for member in Platform.__members__.values():
        if member.value == "local":
            continue
        if member.value in seen:
            continue
        seen.add(member.value)
        entries.append(_build_catalog_entry(member.value))

    try:
        from gateway.platform_registry import platform_registry

        for plugin_entry in platform_registry.plugin_entries():
            if plugin_entry.name in seen:
                continue
            seen.add(plugin_entry.name)
            entries.append(_build_catalog_entry(plugin_entry.name, plugin_entry))
    except Exception:
        _log.debug("plugin platform registry unavailable", exc_info=True)

    order = {pid: idx for idx, pid in enumerate(_PLATFORM_ORDER)}
    entries.sort(
        key=lambda e: (order.get(e["id"], len(_PLATFORM_ORDER)), e["name"].lower())
    )
    return tuple(entries)


def _channel_managed_env_keys() -> frozenset[str]:
    """Env-var keys owned by a Channels page platform card.

    The Channels page is the canonical surface for configuring messaging
    platform credentials (with connection status, test, enable toggle and
    gateway restart). The Keys/Env page consults this set to hide those vars
    so the same fields aren't duplicated in a plainer UI. Best-effort: if the
    gateway catalog can't be built, nothing is flagged and Keys shows it all.
    """
    try:
        keys: set[str] = set()
        for entry in _messaging_platform_catalog():
            keys.update(entry.get("env_vars", ()))
        return frozenset(keys)
    except Exception:
        _log.debug("could not build channel-managed env key set", exc_info=True)
        return frozenset()


# Cross-cutting gateway / relay knobs stay on the Keys → Settings tab even though
# they use the ``messaging`` category in OPTIONAL_ENV_VARS. Platform-scoped vars
# (``DISCORD_*``, ``MATRIX_*``, …) are owned by the Messaging UI instead.
_MESSAGING_KEYS_PAGE_KEYS = frozenset({
    "GATEWAY_ALLOW_ALL_USERS",
    "GATEWAY_PROXY_KEY",
    "GATEWAY_PROXY_URL",
})


def _platform_env_prefixes(platform_id: str) -> tuple[str, ...]:
    """Env-var prefixes owned by a messaging platform card."""
    aliases: dict[str, tuple[str, ...]] = {
        "email": ("EMAIL_",),
        "homeassistant": ("HASS_",),
        "qqbot": ("QQ_", "QQBOT_"),
        "sms": ("TWILIO_",),
        "wecom": ("WECOM_BOT_", "WECOM_SECRET"),
        "wecom_callback": ("WECOM_CALLBACK_",),
    }
    if platform_id in aliases:
        return aliases[platform_id]
    return (platform_id.upper().replace("-", "_") + "_",)


def _discover_platform_env_vars(platform_id: str) -> tuple[str, ...]:
    """All messaging-category env vars for a platform (override + plugin + prefix)."""
    prefixes = _platform_env_prefixes(platform_id)
    keys: list[str] = []
    for name, info in OPTIONAL_ENV_VARS.items():
        if info.get("category") != "messaging":
            continue
        if name in _MESSAGING_KEYS_PAGE_KEYS:
            continue
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        keys.append(name)
    return tuple(sorted(set(keys)))


def _merge_platform_env_vars(
    platform_id: str,
    override: dict[str, Any],
    plugin_entry: Any | None,
) -> tuple[str, ...]:
    """Canonical env-var list for a messaging platform card."""
    discovered = _discover_platform_env_vars(platform_id)
    if "env_vars" in override:
        return tuple(dict.fromkeys((*override["env_vars"], *discovered)))
    if plugin_entry is not None and plugin_entry.required_env:
        return tuple(dict.fromkeys((*tuple(plugin_entry.required_env), *discovered)))
    return discovered


def _build_catalog_entry(
    platform_id: str, plugin_entry: Any | None = None
) -> dict[str, Any]:
    override = _PLATFORM_OVERRIDES.get(platform_id, {})

    env_vars = _merge_platform_env_vars(platform_id, override, plugin_entry)

    if "required_env" in override:
        required_env = tuple(override["required_env"])
    elif plugin_entry is not None:
        required_env = tuple(plugin_entry.required_env or ())
    else:
        required_env = ()

    if override.get("name"):
        name = override["name"]
    elif plugin_entry is not None and plugin_entry.label:
        name = plugin_entry.label
    else:
        name = platform_id.replace("_", " ").title()

    description = override.get("description")
    if not description and plugin_entry is not None:
        description = plugin_entry.install_hint or ""

    return {
        "id": platform_id,
        "name": name,
        "description": description or "",
        "docs_url": override.get("docs_url", ""),
        "env_vars": env_vars,
        "required_env": required_env,
    }


def _catalog_lookup(platform_id: str) -> dict[str, Any] | None:
    for entry in _messaging_platform_catalog():
        if entry["id"] == platform_id:
            return entry
    return None


def _messaging_env_info(key: str) -> dict[str, Any]:
    info = OPTIONAL_ENV_VARS.get(key) or _MESSAGING_ENV_FALLBACKS.get(key) or {}
    return {
        "description": info.get("description", ""),
        "prompt": info.get("prompt", key),
        "help": info.get("help", ""),
        "url": info.get("url"),
        "is_password": info.get("password", False),
        "advanced": info.get("advanced", False),
    }


def _gateway_platform_config(platform_id: str):
    from gateway.config import Platform, load_gateway_config

    config = load_gateway_config()
    platform = Platform(platform_id)
    platform_config = config.platforms.get(platform)
    return config, platform, platform_config


def _messaging_platform_payload(
    entry: dict[str, Any],
    env_on_disk: dict[str, str],
    runtime: dict | None,
    scoped: bool = False,
) -> dict[str, Any]:
    platform_id = entry["id"]
    runtime_platforms = runtime.get("platforms") if runtime else {}
    runtime_platform = (
        runtime_platforms.get(platform_id, {})
        if isinstance(runtime_platforms, dict)
        else {}
    )
    gateway_running = (
        get_running_pid() is not None
        or get_runtime_status_running_pid(runtime) is not None
    )
    env_vars = []

    for key in entry["env_vars"]:
        # When profile-scoped, judge only the profile's own .env — the
        # dashboard process's os.environ carries the ROOT install's .env
        # (loaded at startup) and would falsely report the root credentials
        # as the profile's.
        value = env_on_disk.get(key) or ("" if scoped else os.getenv(key, ""))
        env_vars.append(
            {
                "key": key,
                "required": key in entry["required_env"],
                "is_set": bool(value),
                "redacted_value": redact_key(value) if value else None,
                **_messaging_env_info(key),
            }
        )

    if scoped:
        # Profile-scoped view: derive enablement/configuration from the
        # profile's config.yaml + .env only. load_gateway_config()'s
        # env-override layer reads os.environ and would leak the root
        # install's tokens into the profile's reported state.
        try:
            cfg = load_config()
            platforms_cfg = cfg.get("platforms") or {}
            plat_cfg = platforms_cfg.get(platform_id)
            if not isinstance(plat_cfg, dict):
                plat_cfg = {}
            enabled = bool(plat_cfg.get("enabled"))
            hc = plat_cfg.get("home_channel")
            home_channel = hc if isinstance(hc, dict) else None
        except Exception:
            enabled = False
            home_channel = None
        configured = all(env_on_disk.get(key) for key in entry["required_env"])
    else:
        try:
            gateway_config, platform, platform_config = _gateway_platform_config(
                platform_id
            )
            enabled = bool(platform_config and platform_config.enabled)
            configured = bool(
                platform_config
                and gateway_config._is_platform_connected(platform, platform_config)
            )
            home_channel = (
                platform_config.home_channel.to_dict()
                if platform_config and platform_config.home_channel
                else None
            )
        except Exception:
            enabled = False
            configured = all(
                env_on_disk.get(key) or os.getenv(key, "")
                for key in entry["required_env"]
            )
            home_channel = None

    state = (
        runtime_platform.get("state") if isinstance(runtime_platform, dict) else None
    )
    runtime_gateway_state = runtime.get("gateway_state") if isinstance(runtime, dict) else None
    runtime_gateway_error = runtime.get("exit_reason") if isinstance(runtime, dict) else None
    if not enabled:
        state = "disabled"
    elif not configured:
        state = "not_configured"
    elif gateway_running and not state:
        state = "pending_restart"
    elif (
        not gateway_running
        and not state
        and runtime_gateway_state == "startup_failed"
    ):
        state = "startup_failed"
    elif not gateway_running and not state:
        state = "gateway_stopped"

    error_code = (
        runtime_platform.get("error_code")
        if isinstance(runtime_platform, dict)
        else None
    )
    error_message = (
        runtime_platform.get("error_message")
        if isinstance(runtime_platform, dict)
        else None
    )
    if state == "startup_failed":
        error_code = error_code or "startup_failed"
        error_message = error_message or runtime_gateway_error

    return {
        "id": platform_id,
        "name": entry["name"],
        "description": entry["description"],
        "docs_url": entry["docs_url"],
        "enabled": enabled,
        "configured": configured,
        "gateway_running": gateway_running,
        "state": state,
        "error_code": error_code,
        "error_message": error_message,
        "updated_at": (
            runtime_platform.get("updated_at")
            if isinstance(runtime_platform, dict)
            else None
        ),
        "home_channel": home_channel,
        "env_vars": env_vars,
    }


def _write_platform_enabled(platform_id: str, enabled: bool) -> None:
    write_platform_config_field(platform_id, "enabled", enabled)


_TELEGRAM_ONBOARDING_DEFAULT_URL = "https://setup.hermes-agent.nousresearch.com"
_TELEGRAM_ONBOARDING_USER_AGENT = f"HermesDashboard/{__version__}"
_TELEGRAM_USER_ID_RE = re.compile(r"^\d+$")


@dataclass
class _TelegramOnboardingPairing:
    poll_token: str
    expires_at: str
    expires_at_ts: float
    bot_token: str | None = None
    bot_username: str | None = None
    owner_user_id: str | None = None


_telegram_onboarding_pairings: dict[str, _TelegramOnboardingPairing] = {}
_telegram_onboarding_lock = threading.RLock()


def _telegram_onboarding_base_url() -> str:
    return (
        os.getenv("TELEGRAM_ONBOARDING_URL", _TELEGRAM_ONBOARDING_DEFAULT_URL)
        .strip()
        .rstrip("/")
    )


def _parse_expiry_ts(value: str) -> float:
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except Exception:
        return time.time() + 600


def _prune_telegram_onboarding_pairings() -> None:
    now = time.time()
    expired = [
        pairing_id
        for pairing_id, record in _telegram_onboarding_pairings.items()
        if record.expires_at_ts <= now
    ]
    for pairing_id in expired:
        _telegram_onboarding_pairings.pop(pairing_id, None)


def _normalize_telegram_user_id(value: Any) -> str | None:
    normalized = str(value or "").strip()
    if _TELEGRAM_USER_ID_RE.fullmatch(normalized):
        return normalized
    return None


def _telegram_onboarding_error_message(error: str, fallback: str) -> str:
    return {
        "not_found": "Telegram pairing was not found. Start a new setup.",
        "expired": "Telegram setup expired. Start a new setup.",
        "claimed": "Telegram setup was already claimed. Start a new setup.",
        "unauthorized": "Telegram setup service rejected this request.",
        "telegram_manager_bot_token_not_configured": "Telegram setup service is not configured.",
        "telegram_token_fetch_failed": "Telegram could not finish bot setup. Try again.",
    }.get(error, fallback)


def _telegram_onboarding_request_sync(
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    bearer_token: str | None = None,
) -> dict[str, Any]:
    import httpx

    headers = {
        "Accept": "application/json",
        "User-Agent": _TELEGRAM_ONBOARDING_USER_AGENT,
    }
    request_kwargs: dict[str, Any] = {}
    if body is not None:
        headers["Content-Type"] = "application/json"
        request_kwargs["json"] = body
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"

    url = f"{_telegram_onboarding_base_url()}{path}"
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            response = client.request(
                method,
                url,
                headers=headers,
                **request_kwargs,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            parsed = exc.response.json()
        except Exception:
            parsed = {}
        error = str(parsed.get("error") or parsed.get("status") or "")
        detail = _telegram_onboarding_error_message(
            error,
            "Telegram setup service returned an error.",
        )
        status_code = 404 if exc.response.status_code == 404 else 502
        if error in {"expired", "claimed"}:
            status_code = 410
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail="Telegram setup service is unavailable. Try again shortly.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Telegram setup service is unavailable. Try again shortly.",
        ) from exc

    try:
        parsed = response.json()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Telegram setup service returned an invalid response.",
        ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=502,
            detail="Telegram setup service returned an invalid response.",
        )
    return parsed


async def _telegram_onboarding_request(
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    bearer_token: str | None = None,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _telegram_onboarding_request_sync,
        method,
        path,
        body=body,
        bearer_token=bearer_token,
    )


@app.post("/api/messaging/telegram/onboarding/start")
async def start_telegram_onboarding(body: TelegramOnboardingStart):
    bot_name = (body.bot_name or "Hermes Agent").strip() or "Hermes Agent"
    payload = await _telegram_onboarding_request(
        "POST",
        "/v1/telegram/pairings",
        body={"bot_name": bot_name},
    )

    pairing_id = str(payload.get("pairing_id") or "").strip()
    poll_token = str(payload.get("poll_token") or "").strip()
    expires_at = str(payload.get("expires_at") or "").strip()
    deep_link = str(payload.get("deep_link") or "").strip()
    qr_payload = str(payload.get("qr_payload") or deep_link).strip()
    suggested_username = str(payload.get("suggested_username") or "").strip()
    if not pairing_id or not poll_token or not expires_at or not deep_link:
        raise HTTPException(
            status_code=502,
            detail="Telegram setup service returned an incomplete response.",
        )

    with _telegram_onboarding_lock:
        _prune_telegram_onboarding_pairings()
        _telegram_onboarding_pairings[pairing_id] = _TelegramOnboardingPairing(
            poll_token=poll_token,
            expires_at=expires_at,
            expires_at_ts=_parse_expiry_ts(expires_at),
        )

    return {
        "pairing_id": pairing_id,
        "suggested_username": suggested_username,
        "deep_link": deep_link,
        "qr_payload": qr_payload,
        "expires_at": expires_at,
    }


@app.get("/api/messaging/telegram/onboarding/{pairing_id}")
async def get_telegram_onboarding_status(pairing_id: str):
    with _telegram_onboarding_lock:
        _prune_telegram_onboarding_pairings()
        record = _telegram_onboarding_pairings.get(pairing_id)
        if not record:
            raise HTTPException(
                status_code=404,
                detail="Telegram setup session was not found. Start a new setup.",
            )
        if record.bot_token:
            return {
                "status": "ready",
                "bot_username": record.bot_username,
                "owner_user_id": record.owner_user_id,
                "expires_at": record.expires_at,
            }
        poll_token = record.poll_token

    payload = await _telegram_onboarding_request(
        "GET",
        f"/v1/telegram/pairings/{urllib.parse.quote(pairing_id, safe='')}",
        bearer_token=poll_token,
    )
    status = str(payload.get("status") or "").strip()
    if status == "waiting":
        with _telegram_onboarding_lock:
            current = _telegram_onboarding_pairings.get(pairing_id)
            expires_at = current.expires_at if current else ""
        return {"status": "waiting", "expires_at": expires_at}

    if status == "ready":
        bot_token = str(payload.get("token") or "").strip()
        bot_username = str(payload.get("bot_username") or "").strip()
        if not bot_token:
            raise HTTPException(
                status_code=502,
                detail="Telegram setup service returned an incomplete response.",
            )
        owner_user_id = _normalize_telegram_user_id(payload.get("owner_user_id"))
        with _telegram_onboarding_lock:
            record = _telegram_onboarding_pairings.get(pairing_id)
            if not record:
                raise HTTPException(
                    status_code=404,
                    detail="Telegram setup session was not found. Start a new setup.",
                )
            record.bot_token = bot_token
            record.bot_username = bot_username or None
            record.owner_user_id = owner_user_id
            return {
                "status": "ready",
                "bot_username": record.bot_username,
                "owner_user_id": record.owner_user_id,
                "expires_at": record.expires_at,
            }

    if status in {"expired", "claimed"}:
        with _telegram_onboarding_lock:
            _telegram_onboarding_pairings.pop(pairing_id, None)
        raise HTTPException(
            status_code=410,
            detail=_telegram_onboarding_error_message(
                status,
                "Telegram setup is no longer available. Start a new setup.",
            ),
        )

    raise HTTPException(
        status_code=502,
        detail="Telegram setup service returned an unknown status.",
    )


def _restart_gateway_after_telegram_onboarding(profile: Optional[str] = None) -> dict[str, Any]:
    """Best-effort gateway restart after saving Telegram QR onboarding.

    The QR flow naturally pulls users into Telegram on another device. If the
    saved token waits on a separate dashboard restart click, Hermes appears
    broken from the chat side. Keep the config save authoritative, but report
    restart failures so the UI can fall back to the existing manual banner.
    """
    try:
        proc, reused = _spawn_gateway_restart(profile)
    except Exception as exc:
        _log.exception("Failed to auto-restart gateway after Telegram onboarding")
        return {
            "restart_started": False,
            "restart_error": str(exc),
        }
    if reused:
        _log.info(
            "Telegram onboarding: reusing in-flight gateway restart (pid %s)",
            proc.pid,
        )
    return {
        "restart_started": True,
        "restart_action": "gateway-restart",
        "restart_pid": proc.pid,
    }


@app.post("/api/messaging/telegram/onboarding/{pairing_id}/apply")
async def apply_telegram_onboarding(
    pairing_id: str, body: TelegramOnboardingApply, profile: Optional[str] = None
):
    allowed_user_ids = []
    seen = set()
    for raw_id in body.allowed_user_ids:
        normalized = _normalize_telegram_user_id(raw_id)
        if not normalized:
            raise HTTPException(
                status_code=400,
                detail="Allowed Telegram user IDs must be numeric.",
            )
        if normalized not in seen:
            seen.add(normalized)
            allowed_user_ids.append(normalized)
    if not allowed_user_ids:
        raise HTTPException(
            status_code=400,
            detail="Add at least one allowed Telegram user ID.",
        )

    with _telegram_onboarding_lock:
        _prune_telegram_onboarding_pairings()
        record = _telegram_onboarding_pairings.get(pairing_id)
        if not record:
            raise HTTPException(
                status_code=404,
                detail="Telegram setup session was not found. Start a new setup.",
            )
        bot_token = record.bot_token
        bot_username = record.bot_username
        if not bot_token:
            raise HTTPException(
                status_code=409,
                detail="Telegram setup is not ready yet.",
            )

    effective_profile = body.profile or profile
    try:
        with _profile_scope(effective_profile):
            save_env_value("TELEGRAM_BOT_TOKEN", bot_token)
            save_env_value("TELEGRAM_ALLOWED_USERS", ",".join(allowed_user_ids))
            _write_platform_enabled("telegram", True)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _log.exception("Telegram onboarding apply failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to save Telegram setup.",
        ) from exc

    with _telegram_onboarding_lock:
        _telegram_onboarding_pairings.pop(pairing_id, None)

    restart_result = _restart_gateway_after_telegram_onboarding(effective_profile)

    return {
        "ok": True,
        "platform": "telegram",
        "bot_username": bot_username,
        "needs_restart": not restart_result["restart_started"],
        **restart_result,
    }


@app.delete("/api/messaging/telegram/onboarding/{pairing_id}")
async def cancel_telegram_onboarding(pairing_id: str):
    with _telegram_onboarding_lock:
        _telegram_onboarding_pairings.pop(pairing_id, None)
    return {"ok": True}


@app.get("/api/messaging/platforms")
async def get_messaging_platforms(profile: Optional[str] = None):
    # Profile-scoped so the dashboard's global profile switcher shows the
    # TARGET profile's channel credentials/state, not the root install's.
    # Inside _profile_scope, load_env()/read_runtime_status()/get_running_pid()
    # all resolve against the requested profile's HERMES_HOME.
    with _profile_scope(profile) as scoped_dir:
        env_on_disk = load_env()
        runtime = read_runtime_status()
        return {
            "env_path": str(get_env_path()),
            "gateway_start_command": _gateway_display_command(profile, "start"),
            "platforms": [
                _messaging_platform_payload(
                    entry, env_on_disk, runtime, scoped=scoped_dir is not None
                )
                for entry in _messaging_platform_catalog()
            ]
        }


@app.put("/api/messaging/platforms/{platform_id}")
async def update_messaging_platform(
    platform_id: str, body: MessagingPlatformUpdate, profile: Optional[str] = None
):
    entry = _catalog_lookup(platform_id)
    if not entry:
        raise HTTPException(
            status_code=404, detail=f"Unknown messaging platform: {platform_id}"
        )

    allowed_env = set(entry["env_vars"])
    try:
        with _profile_scope(body.profile or profile):
            for key in body.clear_env:
                if key not in allowed_env:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{key} is not configurable for {entry['name']}",
                    )
                remove_env_value(key)

            for key, value in body.env.items():
                if key not in allowed_env:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{key} is not configurable for {entry['name']}",
                    )
                trimmed = value.strip()
                if trimmed:
                    _validate_messaging_env_value(platform_id, key, trimmed)
                    save_env_value(key, trimmed)

            if body.enabled is not None:
                _write_platform_enabled(platform_id, body.enabled)

        return {"ok": True, "platform": platform_id}
    except HTTPException:
        raise
    except Exception:
        _log.exception("PUT /api/messaging/platforms/%s failed", platform_id)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/messaging/platforms/{platform_id}/test")
async def test_messaging_platform(platform_id: str, profile: Optional[str] = None):
    entry = _catalog_lookup(platform_id)
    if not entry:
        raise HTTPException(
            status_code=404, detail=f"Unknown messaging platform: {platform_id}"
        )

    with _profile_scope(profile) as scoped_dir:
        env_on_disk = load_env()
        payload = _messaging_platform_payload(
            entry, env_on_disk, read_runtime_status(), scoped=scoped_dir is not None
        )
    if not payload["enabled"]:
        message = f"{entry['name']} is disabled. Enable it, then restart the gateway."
        return {"ok": False, "state": payload["state"], "message": message}
    if not payload["configured"]:
        missing = [
            field["key"]
            for field in payload["env_vars"]
            if field["required"] and not field["is_set"]
        ]
        message = (
            f"Missing required setup: {', '.join(missing)}"
            if missing
            else "Platform setup is incomplete."
        )
        return {"ok": False, "state": payload["state"], "message": message}
    if not payload["gateway_running"]:
        return {
            "ok": False,
            "state": payload["state"],
            "message": "Gateway is not running. Restart the gateway to connect this platform.",
        }
    if payload["state"] == "connected":
        return {
            "ok": True,
            "state": payload["state"],
            "message": f"{entry['name']} is connected.",
        }
    if payload.get("error_message"):
        return {
            "ok": False,
            "state": payload["state"],
            "message": payload["error_message"],
        }
    return {
        "ok": False,
        "state": payload["state"],
        "message": "Setup looks complete, but the gateway has not reported a connection yet. Restart the gateway.",
    }


# ---------------------------------------------------------------------------
# OAuth provider endpoints — status + disconnect (Phase 1)
# ---------------------------------------------------------------------------
#
# Phase 1 surfaces *which OAuth providers exist* and whether each is
# connected, plus a disconnect button. The actual login flow (PKCE for
# Anthropic, device-code for Nous/Codex) still runs in the CLI for now;
# Phase 2 will add in-browser flows. For unconnected providers we return
# the canonical ``hermes auth add <provider>`` command so the dashboard
# can surface a one-click copy.


def _truncate_token(value: Optional[str], visible: int = 6) -> str:
    """Return ``...XXXXXX`` (last N chars) for safe display in the UI.

    We never expose more than the trailing ``visible`` characters of an
    OAuth access token. JWT prefixes (the part before the first dot) are
    stripped first when present so the visible suffix is always part of
    the signing region rather than a meaningless header chunk.

    Returns the Entra-ID placeholder when handed a callable (Azure Foundry
    bearer provider) — the callable is NEVER invoked here.
    """
    if not value:
        return ""
    if callable(value) and not isinstance(value, str):
        # Entra ID bearer provider — never reveal a minted token in the UI.
        return "<entra-id-bearer>"
    s = str(value)
    if "." in s and s.count(".") >= 2:
        # Looks like a JWT — show the trailing piece of the signature only.
        s = s.rsplit(".", 1)[-1]
    if len(s) <= visible:
        return s
    return f"…{s[-visible:]}"


def _anthropic_oauth_status() -> Dict[str, Any]:
    """Status for the "Anthropic API Key" catalog entry.

    Two sources, in priority order:
    1. ``~/.hermes/.anthropic_oauth.json`` — Hermes-managed PKCE flow (what
       this entry's Connect button writes)
    2. ``ANTHROPIC_API_KEY`` → ``ANTHROPIC_TOKEN`` → ``CLAUDE_CODE_OAUTH_TOKEN``
       env vars (registry order) — from ``.env``, the shell, or an external
       secret source like Bitwarden (whose keys are injected into the process
       env during ``load_hermes_dotenv()``, so the same check covers them)

    Claude Code's ``~/.claude/.credentials.json`` is deliberately NOT read
    here — it has its own dedicated catalog entry (``claude-code`` →
    ``_claude_code_only_status``). Reporting it under the API-key entry
    double-counts the token and shadows a real ANTHROPIC_API_KEY.
    """
    try:
        from agent.anthropic_adapter import (
            read_hermes_oauth_credentials,
            _HERMES_OAUTH_FILE,
        )
    except ImportError:
        read_hermes_oauth_credentials = None  # type: ignore
        _HERMES_OAUTH_FILE = None  # type: ignore

    hermes_creds = None
    if read_hermes_oauth_credentials:
        try:
            hermes_creds = read_hermes_oauth_credentials()
        except Exception:
            hermes_creds = None
    if hermes_creds and hermes_creds.get("accessToken"):
        return {
            "logged_in": True,
            "source": "hermes_pkce",
            "source_label": f"Hermes PKCE ({_HERMES_OAUTH_FILE})",
            "token_preview": _truncate_token(hermes_creds.get("accessToken")),
            "expires_at": hermes_creds.get("expiresAt"),
            "has_refresh_token": bool(hermes_creds.get("refreshToken")),
        }

    # Env-var / secret-source path. ``get_env_value`` checks the process
    # environment first (where Bitwarden-sourced secrets land) then .env.
    env_var_order: tuple = ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        env_var_order = PROVIDER_REGISTRY["anthropic"].api_key_env_vars
    except (ImportError, KeyError):
        pass
    try:
        from hermes_cli.config import get_env_value
    except ImportError:
        get_env_value = None  # type: ignore
    try:
        from hermes_cli.env_loader import format_secret_source_suffix
    except ImportError:
        format_secret_source_suffix = None  # type: ignore

    for var in env_var_order:
        value = (get_env_value(var) if get_env_value else None) or os.getenv(var)
        if not value:
            continue
        suffix = format_secret_source_suffix(var) if format_secret_source_suffix else ""
        return {
            "logged_in": True,
            "source": "env_var",
            "source_label": f"{var}{suffix}",
            "token_preview": _truncate_token(value),
            "expires_at": None,
            "has_refresh_token": False,
        }
    return {"logged_in": False, "source": None}


def _claude_code_only_status() -> Dict[str, Any]:
    """Surface Claude Code CLI credentials as their own provider entry.

    Independent of the Anthropic entry above so users can see whether their
    Claude Code subscription tokens are actively flowing into Hermes even
    when they also have a separate Hermes-managed PKCE login.
    """
    try:
        from agent.anthropic_adapter import read_claude_code_credentials
        creds = read_claude_code_credentials()
    except Exception:
        creds = None
    if creds and creds.get("accessToken"):
        return {
            "logged_in": True,
            "source": "claude_code_cli",
            "source_label": "~/.claude/.credentials.json",
            "token_preview": _truncate_token(creds.get("accessToken")),
            "expires_at": creds.get("expiresAt"),
            "has_refresh_token": bool(creds.get("refreshToken")),
        }
    return {"logged_in": False, "source": None}


def _copilot_acp_status() -> Dict[str, Any]:
    """Status for copilot-acp — credentials are owned by the Copilot CLI.

    There is no cheap programmatic credential probe for the ACP subprocess, so
    this is a read-only "managed by the Copilot CLI" card (like claude-code):
    Hermes never claims a login state it can't verify.
    """
    return {
        "logged_in": False,
        "source": "copilot_cli",
        "source_label": "Managed by the GitHub Copilot CLI",
        "token_preview": None,
        "expires_at": None,
        "has_refresh_token": False,
    }


# Explicit, hand-tuned OAuth/account provider cards. These carry the bits that
# can't be derived from the unified provider catalog: the OAuth ``flow`` shape,
# the per-provider ``status_fn``, the ``cli_command`` fallback, and curated
# display order. They are the OVERRIDE BASE for ``_build_oauth_catalog()``,
# which unions them with every accounts-tab provider in ``provider_catalog()``
# so newly-added OAuth/external providers appear automatically (no hand edit).
# This tuple also still includes two entries that are NOT catalog providers but
# must show on the Accounts tab: the api-key Anthropic PKCE card and the
# synthetic ``claude-code`` subscription row.
# ``flow`` describes the OAuth shape so the modal can pick the right UI:
# ``pkce`` = open URL + paste callback code, ``device_code`` = show code +
# verification URL + poll, ``external`` = read-only (delegated to a third-party
# CLI like Claude Code or Qwen), ``loopback`` = 127.0.0.1 callback listener.
_OAUTH_PROVIDER_CATALOG: tuple[Dict[str, Any], ...] = (
    {
        "id": "nous",
        "name": "Nous Portal",
        "flow": "device_code",
        "cli_command": "hermes auth add nous",
        "docs_url": "https://portal.nousresearch.com",
        "status_fn": None,  # dispatched via auth.get_nous_auth_status
    },
    {
        "id": "openai-codex",
        "name": "OpenAI OAuth (ChatGPT)",
        "flow": "device_code",
        "cli_command": "hermes auth add openai-codex",
        "docs_url": "https://platform.openai.com/docs",
        "status_fn": None,  # dispatched via auth.get_codex_auth_status
    },
    {
        "id": "qwen-oauth",
        "name": "Qwen (via Qwen CLI)",
        "flow": "external",
        "cli_command": "hermes auth add qwen-oauth",
        "docs_url": "https://github.com/QwenLM/qwen-code",
        "status_fn": None,  # dispatched via auth.get_qwen_auth_status
    },
    {
        "id": "minimax-oauth",
        "name": "MiniMax (OAuth)",
        # MiniMax's flow is structurally device-code (verification URI +
        # user code, backend polls the token endpoint) with a PKCE
        # extension for code-binding. The dashboard renders the same UX
        # as Nous's device-code flow; the PKCE bit is a security
        # extension that doesn't change the operator experience.
        "flow": "device_code",
        "cli_command": "hermes auth add minimax-oauth",
        "docs_url": "https://www.minimax.io",
        "status_fn": None,  # dispatched via auth.get_minimax_oauth_auth_status
    },
    {
        "id": "xai-oauth",
        "name": "xAI Grok OAuth (SuperGrok / Premium+)",
        # Loopback PKCE: the desktop's local backend binds a 127.0.0.1
        # callback server, the client opens the browser, and the redirect
        # lands back on the loopback listener — no code to copy/paste.
        "flow": "loopback",
        "cli_command": "hermes auth add xai-oauth",
        "docs_url": "https://hermes-agent.nousresearch.com/docs/guides/xai-grok-oauth",
        "status_fn": None,  # dispatched via auth.get_xai_oauth_auth_status
    },
    {
        "id": "copilot-acp",
        "name": "GitHub Copilot (ACP)",
        "flow": "external",
        "cli_command": "copilot /login",
        "docs_url": "https://docs.github.com/en/copilot",
        "status_fn": _copilot_acp_status,
    },
    # ── Anthropic / Claude entries sit at the bottom: the API-key path
    # first, then the subscription OAuth path (which only works with extra
    # usage credits on top of a Claude Max plan — see disclaimer in name).
    {
        "id": "anthropic",
        "name": "Anthropic API Key",
        "flow": "pkce",
        "cli_command": "hermes auth add anthropic",
        "docs_url": "https://docs.claude.com/en/api/getting-started",
        "status_fn": _anthropic_oauth_status,
    },
    {
        "id": "claude-code",
        "name": "Anthropic OAuth: Required Extra Usage Credits to Use Subscription",
        "flow": "external",
        "cli_command": "claude setup-token",
        "docs_url": "https://docs.claude.com/en/docs/claude-code",
        "status_fn": _claude_code_only_status,
    },
)


def _resolve_provider_status(provider_id: str, status_fn) -> Dict[str, Any]:
    """Dispatch to the right status helper for an OAuth provider entry."""
    if status_fn is not None:
        try:
            return status_fn()
        except Exception as e:
            return {"logged_in": False, "error": str(e)}
    try:
        from hermes_cli import auth as hauth
        if provider_id == "nous":
            raw = hauth.get_nous_auth_status()
            return {
                "logged_in": bool(raw.get("logged_in")),
                "source": "nous_portal",
                "source_label": raw.get("portal_base_url") or "Nous Portal",
                "token_preview": _truncate_token(raw.get("access_token")),
                "expires_at": raw.get("access_expires_at"),
                "has_refresh_token": bool(raw.get("has_refresh_token")),
            }
        if provider_id == "openai-codex":
            raw = hauth.get_codex_auth_status()
            return {
                "logged_in": bool(raw.get("logged_in")),
                "source": raw.get("source") or "openai_codex",
                "source_label": raw.get("auth_mode") or "OpenAI Codex",
                "token_preview": _truncate_token(raw.get("api_key")),
                "expires_at": None,
                "has_refresh_token": False,
                "last_refresh": raw.get("last_refresh"),
            }
        if provider_id == "qwen-oauth":
            raw = hauth.get_qwen_auth_status()
            return {
                "logged_in": bool(raw.get("logged_in")),
                "source": "qwen_cli",
                "source_label": raw.get("auth_store_path") or "Qwen CLI",
                "token_preview": _truncate_token(raw.get("access_token")),
                "expires_at": raw.get("expires_at"),
                "has_refresh_token": bool(raw.get("has_refresh_token")),
            }
        if provider_id == "minimax-oauth":
            raw = hauth.get_minimax_oauth_auth_status()
            return {
                "logged_in": bool(raw.get("logged_in")),
                "source": "minimax_oauth",
                "source_label": f"MiniMax ({raw.get('region', 'global')})",
                "token_preview": None,
                "expires_at": raw.get("expires_at"),
                "has_refresh_token": True,
            }
        if provider_id == "xai-oauth":
            raw = hauth.get_xai_oauth_auth_status()
            # source_label is meant to be a human-readable origin (auth-store
            # path / credential source), not the internal auth_mode string
            # ("oauth_pkce"). Prefer the store path, then the source slug.
            return {
                "logged_in": bool(raw.get("logged_in")),
                "source": raw.get("source") or "xai_oauth",
                "source_label": raw.get("auth_store") or raw.get("source") or "xAI Grok OAuth",
                "token_preview": _truncate_token(raw.get("api_key")),
                "expires_at": None,
                "has_refresh_token": True,
                "last_refresh": raw.get("last_refresh"),
            }
        # No hand-written branch for this provider id: fall through to the
        # canonical slug-driven dispatcher so accounts-tab providers derived
        # from the unified catalog (which carry status_fn=None) still reflect
        # real login state instead of rendering permanently logged-out. This
        # closes the membership-auto-extends-but-status-doesn't gap: add an
        # OAuth/account provider plugin and its card shows the right state.
        raw = hauth.get_auth_status(provider_id)
        if isinstance(raw, dict) and "logged_in" in raw:
            return {
                "logged_in": bool(raw.get("logged_in")),
                "source": raw.get("source") or raw.get("provider") or provider_id,
                "source_label": (
                    raw.get("source_label")
                    or raw.get("auth_store")
                    or raw.get("auth_store_path")
                    or raw.get("base_url")
                    or raw.get("name")
                    or ""
                ),
                "token_preview": _truncate_token(
                    raw.get("access_token") or raw.get("api_key")
                ),
                "expires_at": raw.get("expires_at") or raw.get("access_expires_at"),
                "has_refresh_token": bool(raw.get("has_refresh_token")),
            }
    except Exception as e:
        return {"logged_in": False, "error": str(e)}
    return {"logged_in": False}


def _oauth_provider_disconnect_command(provider: Dict[str, Any]) -> Optional[str]:
    """Shell command that clears an external provider's credentials.

    External providers store their credentials outside Hermes, so the disconnect
    API deliberately refuses them (we never delete files another CLI owns on the
    user's behalf via a silent API call). For the ones we know how to clear we
    instead hand the GUI a command it can *run in the embedded terminal* — the
    user sees exactly what executes, and Hermes then stops resolving the token.

    Claude Code has no scriptable logout (only the interactive ``/logout``), so
    we remove the credential the same way logout does: the macOS Keychain entry
    (``Claude Code-credentials``) and/or the ``~/.claude/.credentials.json``
    file — the two sources ``read_claude_code_credentials()`` consults. Returns
    None for providers we can't safely clear (the GUI shows a manual hint).
    """
    if provider.get("flow") != "external":
        return None
    if provider.get("id") == "claude-code":
        rm_file = "rm -f ~/.claude/.credentials.json"
        if sys.platform == "darwin":
            return f'security delete-generic-password -s "Claude Code-credentials" 2>/dev/null; {rm_file}'
        return rm_file
    return None


def _oauth_provider_disconnect_hint(provider: Dict[str, Any], status: Dict[str, Any]) -> Optional[str]:
    """Return the manual disconnect path when the API cannot clear this provider."""
    if provider.get("flow") == "external":
        if _oauth_provider_disconnect_command(provider):
            # The GUI offers a one-click "run in terminal" path; this hint is the
            # fallback wording for surfaces that only show text.
            return "Managed outside Hermes — run the disconnect command to remove it."
        return "Managed by that provider's CLI; remove it there."
    if status.get("source") == "env_var":
        return "Remove the API key from Settings → Keys instead."
    return None


def _build_oauth_catalog() -> list[Dict[str, Any]]:
    """Build the Accounts-tab provider list.

    MEMBERSHIP is the union of:
      1. ``_OAUTH_PROVIDER_CATALOG`` — the explicit, hand-tuned cards that carry
         bespoke flow / status_fn / cli_command (including the api-key Anthropic
         PKCE card and the synthetic claude-code subscription row, which are not
         catalog providers), and
      2. every accounts-tab provider in the unified ``provider_catalog()`` (the
         ``hermes model`` universe) — so any OAuth/external provider added as a
         plugin appears automatically, with sensible defaults, even if no
         explicit card was written for it.

    The explicit catalog wins on metadata; the unified catalog guarantees we
    never silently drop a provider the CLI picker offers. Order: explicit cards
    first (their curated order), then any catalog-only providers appended in
    ``hermes model`` order.
    """
    rows: list[Dict[str, Any]] = []
    seen: set[str] = set()

    # 1. Explicit hand-tuned cards (authoritative metadata + curated order).
    for entry in _OAUTH_PROVIDER_CATALOG:
        if entry["id"] in seen:
            continue
        seen.add(entry["id"])
        rows.append(dict(entry))

    # 2. Catalog accounts-providers not already covered — keeps the Accounts tab
    #    in lockstep with the `hermes model` universe (zero-edit for new plugins).
    try:
        from hermes_cli.provider_catalog import provider_catalog
        for d in provider_catalog():
            if d.tab != "accounts" or d.slug in seen:
                continue
            seen.add(d.slug)
            rows.append({
                "id": d.slug,
                "name": d.label,
                "flow": "external",
                "cli_command": f"hermes auth add {d.slug}",
                "docs_url": d.signup_url or "",
                "status_fn": None,
            })
    except Exception:
        pass

    return rows


@app.get("/api/providers/oauth")
async def list_oauth_providers(profile: Optional[str] = None):
    """Enumerate every OAuth-capable LLM provider with current status.

    Response shape (per provider):
        id              stable identifier (used in DELETE path)
        name            human label
        flow            "pkce" | "device_code" | "external" | "loopback"
        cli_command     fallback CLI command for users to run manually
        disconnect_command  shell command that clears an external provider's
                            creds (run in the embedded terminal), else null
        docs_url        external docs/portal link for the "Learn more" link
        status:
          logged_in        bool — currently has usable creds
          source           short slug ("hermes_pkce", "claude_code", ...)
          source_label     human-readable origin (file path, env var name)
          token_preview    last N chars of the token, never the full token
          expires_at       ISO timestamp string or null
          has_refresh_token bool

    Membership is derived from the unified provider_catalog() so this stays in
    sync with the `hermes model` picker; _OAUTH_OVERRIDES supplies per-provider
    flow/status/cli metadata.
    """
    with _profile_scope(profile):
        providers = []
        for p in _build_oauth_catalog():
            status = _resolve_provider_status(p["id"], p.get("status_fn"))
            disconnect_hint = _oauth_provider_disconnect_hint(p, status)
            providers.append({
                "id": p["id"],
                "name": p["name"],
                "flow": p["flow"],
                "cli_command": p["cli_command"],
                "docs_url": p["docs_url"],
                "disconnect_hint": disconnect_hint,
                "disconnect_command": _oauth_provider_disconnect_command(p),
                "disconnectable": disconnect_hint is None,
                "status": status,
            })
        return {"providers": providers}


@app.delete("/api/providers/oauth/{provider_id}")
async def disconnect_oauth_provider(
    provider_id: str,
    request: Request,
    profile: Optional[str] = None,
):
    """Disconnect an OAuth provider. Token-protected (matches /env/reveal)."""
    _require_token(request)

    with _profile_scope(profile):
        catalog_by_id = {p["id"]: p for p in _build_oauth_catalog()}
        provider = catalog_by_id.get(provider_id)
        if provider is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown provider: {provider_id}. "
                       f"Available: {', '.join(sorted(catalog_by_id))}",
            )

        disconnect_hint = _oauth_provider_disconnect_hint(provider, {})
        if disconnect_hint:
            raise HTTPException(
                status_code=400,
                detail=f"{provider['name']} cannot be disconnected automatically. {disconnect_hint}",
            )

        status = _resolve_provider_status(provider_id, provider.get("status_fn"))
        disconnect_hint = _oauth_provider_disconnect_hint(provider, status)
        if disconnect_hint:
            raise HTTPException(
                status_code=400,
                detail=f"{provider['name']} cannot be disconnected automatically. {disconnect_hint}",
            )

        # Anthropic clears only the Hermes-managed PKCE file and auth-store entry.
        # The separate claude-code catalog row is external/read-only and rejected
        # above so we never pretend to remove ~/.claude/* credentials owned by the CLI.
        if provider_id == "anthropic":
            cleared = False
            try:
                from agent.anthropic_adapter import _HERMES_OAUTH_FILE
                if _HERMES_OAUTH_FILE.exists():
                    _HERMES_OAUTH_FILE.unlink()
                    cleared = True
            except Exception:
                pass
            # Also clear the credential pool entry if present.
            try:
                from hermes_cli.auth import clear_provider_auth
                cleared = clear_provider_auth("anthropic") or cleared
            except Exception:
                pass
            _log.info("oauth/disconnect: %s", provider_id)
            return {"ok": bool(cleared), "provider": provider_id}

        try:
            from hermes_cli.auth import clear_provider_auth, invalidate_nous_auth_status_cache
            cleared = clear_provider_auth(provider_id)
            if provider_id == "nous":
                invalidate_nous_auth_status_cache()
            _log.info("oauth/disconnect: %s (cleared=%s)", provider_id, cleared)
            return {"ok": bool(cleared), "provider": provider_id}
        except Exception as e:
            _log.exception("disconnect %s failed", provider_id)
            raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# OAuth Phase 2 — in-browser PKCE & device-code flows
# ---------------------------------------------------------------------------
#
# Two flow shapes are supported:
#
#   PKCE (Anthropic):
#     1. POST /api/providers/oauth/anthropic/start
#          → server generates code_verifier + challenge, builds claude.ai
#            authorize URL, stashes verifier in _oauth_sessions[session_id]
#          → returns { session_id, flow: "pkce", auth_url }
#     2. UI opens auth_url in a new tab. User authorizes, copies code.
#     3. POST /api/providers/oauth/anthropic/submit { session_id, code }
#          → server exchanges (code + verifier) → tokens at console.anthropic.com
#          → persists to ~/.hermes/.anthropic_oauth.json AND credential pool
#          → returns { ok: true, status: "approved" }
#
#   Device code (Nous, OpenAI Codex):
#     1. POST /api/providers/oauth/{nous|openai-codex}/start
#          → server hits provider's device-auth endpoint
#          → gets { user_code, verification_url, device_code, interval, expires_in }
#          → spawns background poller thread that polls the token endpoint
#            every `interval` seconds until approved/expired
#          → stores poll status in _oauth_sessions[session_id]
#          → returns { session_id, flow: "device_code", user_code,
#                      verification_url, expires_in, poll_interval }
#     2. UI opens verification_url in a new tab and shows user_code.
#     3. UI polls GET /api/providers/oauth/{provider}/poll/{session_id}
#          every 2s until status != "pending".
#     4. On "approved" the background thread has already saved creds; UI
#        refreshes the providers list.
#
#   Loopback PKCE (xAI Grok):
#     1. POST /api/providers/oauth/xai-oauth/start
#          → server binds a 127.0.0.1 callback listener, builds the xAI
#            authorize URL, spawns a background worker waiting on the redirect
#          → returns { session_id, flow: "loopback", auth_url, expires_in }
#     2. UI opens auth_url in the browser. There is NO user_code/code to
#        paste — the redirect lands back on the loopback listener.
#     3. UI polls GET /api/providers/oauth/{provider}/poll/{session_id}
#          (same endpoint as device_code) until status != "pending".
#     4. The worker exchanges the code, persists creds, sets "approved".
#        DELETE /sessions/{id} cancels: the worker bails before persisting
#        and the callback server is shut down to free the port immediately.
#
# Sessions are kept in-memory only (single-process FastAPI) and time out
# after 15 minutes. A periodic cleanup runs on each /start call to GC
# expired sessions so the dict doesn't grow without bound.

_OAUTH_SESSION_TTL_SECONDS = 15 * 60
_oauth_sessions: Dict[str, Dict[str, Any]] = {}
_oauth_sessions_lock = threading.Lock()

# Import OAuth constants from canonical source instead of duplicating.
# Guarded so hermes web still starts if anthropic_adapter is unavailable;
# Phase 2 endpoints will return 501 in that case.
try:
    from agent.anthropic_adapter import (
        _OAUTH_CLIENT_ID as _ANTHROPIC_OAUTH_CLIENT_ID,
        _OAUTH_TOKEN_URL as _ANTHROPIC_OAUTH_TOKEN_URL,
        _OAUTH_TOKEN_URLS as _ANTHROPIC_OAUTH_TOKEN_URLS,
        _OAUTH_REDIRECT_URI as _ANTHROPIC_OAUTH_REDIRECT_URI,
        _OAUTH_SCOPES as _ANTHROPIC_OAUTH_SCOPES,
        _generate_pkce as _generate_pkce_pair,
    )
    _ANTHROPIC_OAUTH_AVAILABLE = True
except ImportError:
    _ANTHROPIC_OAUTH_AVAILABLE = False
_ANTHROPIC_OAUTH_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"


def _gc_oauth_sessions() -> None:
    """Drop expired sessions. Called opportunistically on /start."""
    cutoff = time.time() - _OAUTH_SESSION_TTL_SECONDS
    with _oauth_sessions_lock:
        stale = [sid for sid, sess in _oauth_sessions.items() if sess["created_at"] < cutoff]
        for sid in stale:
            _oauth_sessions.pop(sid, None)


def _oauth_profile_name(profile: Optional[str]) -> Optional[str]:
    requested = (profile or "").strip()
    if not requested or requested.lower() == "current":
        return None
    return requested


def _validate_oauth_profile(profile: Optional[str]) -> None:
    profile_name = _oauth_profile_name(profile)
    if profile_name:
        _resolve_profile_dir(profile_name)


def _new_oauth_session(
    provider_id: str,
    flow: str,
    profile: Optional[str] = None,
) -> tuple[str, Dict[str, Any]]:
    """Create + register a new OAuth session, return (session_id, session_dict)."""
    sid = secrets.token_urlsafe(16)
    profile_name = _oauth_profile_name(profile)
    sess = {
        "session_id": sid,
        "provider": provider_id,
        "flow": flow,
        "profile": profile_name,
        "created_at": time.time(),
        "status": "pending",  # pending | approved | denied | expired | error
        "error_message": None,
    }
    with _oauth_sessions_lock:
        _oauth_sessions[sid] = sess
    return sid, sess


def _oauth_session_profile(
    session_id: str,
    fallback: Optional[str] = None,
) -> Optional[str]:
    """Return the profile that owns an OAuth session, if one was provided."""
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
        profile = sess.get("profile") if sess else None
    return profile or _oauth_profile_name(fallback)


def _save_anthropic_oauth_creds(access_token: str, refresh_token: str, expires_at_ms: int) -> None:
    """Persist Anthropic PKCE creds to both Hermes file AND credential pool.

    Mirrors what auth_commands.add_command does so the dashboard flow leaves
    the system in the same state as ``hermes auth add anthropic``.
    """
    from agent.anthropic_adapter import _HERMES_OAUTH_FILE
    payload = {
        "accessToken": access_token,
        "refreshToken": refresh_token,
        "expiresAt": expires_at_ms,
    }
    _HERMES_OAUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _HERMES_OAUTH_FILE.with_name(
        f"{_HERMES_OAUTH_FILE.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, _HERMES_OAUTH_FILE)
        try:
            _HERMES_OAUTH_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
    # Best-effort credential-pool insert. Failure here doesn't invalidate
    # the file write — pool registration only matters for the rotation
    # strategy, not for runtime credential resolution.
    try:
        from agent.credential_pool import (
            PooledCredential,
            load_pool,
            AUTH_TYPE_OAUTH,
            SOURCE_MANUAL,
        )
        import uuid
        pool = load_pool("anthropic")
        # Avoid duplicate entries: delete any prior dashboard-issued OAuth entry
        existing = [e for e in pool.entries() if getattr(e, "source", "").startswith(f"{SOURCE_MANUAL}:dashboard_pkce")]
        for e in existing:
            try:
                pool.remove_entry(getattr(e, "id", ""))
            except Exception:
                pass
        entry = PooledCredential(
            provider="anthropic",
            id=uuid.uuid4().hex[:6],
            label="dashboard PKCE",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source=f"{SOURCE_MANUAL}:dashboard_pkce",
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at_ms=expires_at_ms,
        )
        pool.add_entry(entry)
    except Exception as e:
        _log.warning("anthropic pool add (dashboard) failed: %s", e)


def _start_anthropic_pkce(profile: Optional[str] = None) -> Dict[str, Any]:
    """Begin PKCE flow. Returns the auth URL the UI should open."""
    if not _ANTHROPIC_OAUTH_AVAILABLE:
        raise HTTPException(status_code=501, detail="Anthropic OAuth not available (missing adapter)")
    verifier, challenge = _generate_pkce_pair()
    sid, sess = _new_oauth_session("anthropic", "pkce", profile=profile)
    sess["verifier"] = verifier
    sess["state"] = verifier  # Anthropic round-trips verifier as state
    params = {
        "code": "true",
        "client_id": _ANTHROPIC_OAUTH_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": _ANTHROPIC_OAUTH_REDIRECT_URI,
        "scope": _ANTHROPIC_OAUTH_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    auth_url = f"{_ANTHROPIC_OAUTH_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
    return {
        "session_id": sid,
        "flow": "pkce",
        "auth_url": auth_url,
        "expires_in": _OAUTH_SESSION_TTL_SECONDS,
    }


def _submit_anthropic_pkce(
    session_id: str,
    code_input: str,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Exchange authorization code for tokens. Persists on success."""
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
    if not sess or sess["provider"] != "anthropic" or sess["flow"] != "pkce":
        raise HTTPException(status_code=404, detail="Unknown or expired session")
    if sess["status"] != "pending":
        return {"ok": False, "status": sess["status"], "message": sess.get("error_message")}

    # Anthropic's redirect callback page formats the code as `<code>#<state>`.
    # Strip the state suffix if present (we already have the verifier server-side).
    parts = code_input.strip().split("#", 1)
    code = parts[0].strip()
    if not code:
        return {"ok": False, "status": "error", "message": "No code provided"}
    state_from_callback = parts[1] if len(parts) > 1 else ""

    exchange_data = json.dumps({
        "grant_type": "authorization_code",
        "client_id": _ANTHROPIC_OAUTH_CLIENT_ID,
        "code": code,
        "state": state_from_callback or sess["state"],
        "redirect_uri": _ANTHROPIC_OAUTH_REDIRECT_URI,
        "code_verifier": sess["verifier"],
    }).encode()
    # Anthropic migrated the OAuth token endpoint to platform.claude.com;
    # console.anthropic.com now 404s. Try the new host first, then fall back.
    result = None
    last_exc = None
    for _endpoint in _ANTHROPIC_OAUTH_TOKEN_URLS:
        req = urllib.request.Request(
            _endpoint,
            data=exchange_data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "hermes-dashboard/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                result = json.loads(resp.read().decode())
            break
        except Exception as e:
            last_exc = e
            continue
    if result is None:
        with _oauth_sessions_lock:
            sess["status"] = "error"
            sess["error_message"] = f"Token exchange failed: {last_exc}"
        return {"ok": False, "status": "error", "message": sess["error_message"]}

    access_token = result.get("access_token", "")
    refresh_token = result.get("refresh_token", "")
    expires_in = int(result.get("expires_in") or 3600)
    if not access_token:
        with _oauth_sessions_lock:
            sess["status"] = "error"
            sess["error_message"] = "No access token returned"
        return {"ok": False, "status": "error", "message": sess["error_message"]}

    expires_at_ms = int(time.time() * 1000) + (expires_in * 1000)
    try:
        with _profile_scope(_oauth_session_profile(session_id, profile)):
            _save_anthropic_oauth_creds(access_token, refresh_token, expires_at_ms)
    except Exception as e:
        with _oauth_sessions_lock:
            sess["status"] = "error"
            sess["error_message"] = f"Save failed: {e}"
        return {"ok": False, "status": "error", "message": sess["error_message"]}
    with _oauth_sessions_lock:
        sess["status"] = "approved"
    _log.info("oauth/pkce: anthropic login completed (session=%s)", session_id)
    return {"ok": True, "status": "approved"}


async def _start_device_code_flow(
    provider_id: str,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Initiate a device-code flow (Nous, OpenAI Codex, or MiniMax).

    Calls the provider's device-auth endpoint via the existing CLI helpers,
    then spawns a background poller. Returns the user-facing display fields
    so the UI can render the verification page link + user code.
    """
    if provider_id == "nous":
        from hermes_cli.auth import (
            _request_device_code,
            PROVIDER_REGISTRY,
        )
        import httpx
        pconfig = PROVIDER_REGISTRY["nous"]
        portal_base_url = (
            os.getenv("HERMES_PORTAL_BASE_URL")
            or os.getenv("NOUS_PORTAL_BASE_URL")
            or pconfig.portal_base_url
        ).rstrip("/")
        client_id = pconfig.client_id
        scope = pconfig.scope

        def _do_nous_device_request():
            with httpx.Client(
                timeout=httpx.Timeout(15.0),
                headers={"Accept": "application/json"},
            ) as client:
                return (
                    _request_device_code(
                        client=client,
                        portal_base_url=portal_base_url,
                        client_id=client_id,
                        scope=scope,
                    ),
                    scope,
                )

        device_data, effective_scope = await asyncio.get_running_loop().run_in_executor(
            None, _do_nous_device_request
        )
        sid, sess = _new_oauth_session("nous", "device_code", profile=profile)
        sess["device_code"] = str(device_data["device_code"])
        sess["interval"] = int(device_data["interval"])
        sess["expires_at"] = time.time() + int(device_data["expires_in"])
        sess["portal_base_url"] = portal_base_url
        sess["client_id"] = client_id
        sess["scope"] = effective_scope
        threading.Thread(
            target=_nous_poller, args=(sid,), daemon=True, name=f"oauth-poll-{sid[:6]}"
        ).start()
        return {
            "session_id": sid,
            "flow": "device_code",
            "user_code": str(device_data["user_code"]),
            "verification_url": str(device_data["verification_uri_complete"]),
            "expires_in": int(device_data["expires_in"]),
            "poll_interval": int(device_data["interval"]),
        }

    if provider_id == "openai-codex":
        # Codex uses fixed OpenAI device-auth endpoints; reuse the helper.
        sid, _ = _new_oauth_session("openai-codex", "device_code", profile=profile)
        # Use the helper but in a thread because it polls inline.
        # We can't extract just the start step without refactoring auth.py,
        # so we run the full helper in a worker and proxy the user_code +
        # verification_url back via the session dict. The helper prints
        # to stdout — we capture nothing here, just status.
        threading.Thread(
            target=_codex_full_login_worker, args=(sid,), daemon=True,
            name=f"oauth-codex-{sid[:6]}",
        ).start()
        # Block briefly until the worker has populated the user_code, OR error.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with _oauth_sessions_lock:
                s = _oauth_sessions.get(sid)
            if s and (s.get("user_code") or s["status"] != "pending"):
                break
            await asyncio.sleep(0.1)
        with _oauth_sessions_lock:
            s = _oauth_sessions.get(sid, {})
        if s.get("status") == "error":
            raise HTTPException(status_code=500, detail=s.get("error_message") or "device-auth failed")
        if not s.get("user_code"):
            raise HTTPException(status_code=504, detail="device-auth timed out before returning a user code")
        return {
            "session_id": sid,
            "flow": "device_code",
            "user_code": s["user_code"],
            "verification_url": s["verification_url"],
            "expires_in": int(s.get("expires_in") or 900),
            "poll_interval": int(s.get("interval") or 5),
        }

    if provider_id == "minimax-oauth":
        # MiniMax uses a device-code-style flow (verification URI + user
        # code + background poll) with a PKCE extension on top. From the
        # operator's perspective it's identical to Nous's device-code
        # flow; the PKCE bit (verifier + challenge from
        # _minimax_pkce_pair) is a security extension that binds the
        # token exchange to the original session.
        from hermes_cli.auth import (
            _minimax_pkce_pair,
            _minimax_request_user_code,
            MINIMAX_OAUTH_CLIENT_ID,
            MINIMAX_OAUTH_GLOBAL_BASE,
        )
        import httpx
        verifier, challenge, state = _minimax_pkce_pair()
        portal_base_url = (
            os.getenv("MINIMAX_PORTAL_BASE_URL") or MINIMAX_OAUTH_GLOBAL_BASE
        ).rstrip("/")
        def _do_minimax_request():
            with httpx.Client(
                timeout=httpx.Timeout(15.0),
                headers={"Accept": "application/json"},
                follow_redirects=True,
            ) as client:
                return _minimax_request_user_code(
                    client=client,
                    portal_base_url=portal_base_url,
                    client_id=MINIMAX_OAUTH_CLIENT_ID,
                    code_challenge=challenge,
                    state=state,
                )
        device_data = await asyncio.get_event_loop().run_in_executor(
            None, _do_minimax_request
        )
        sid, sess = _new_oauth_session("minimax-oauth", "device_code", profile=profile)
        # The CLI flow names this `interval_ms` because MiniMax's
        # `interval` field is in milliseconds (defensive default 2000ms
        # in _minimax_poll_token).
        interval_raw = device_data.get("interval")
        sess["interval_ms"] = (
            int(interval_raw) if interval_raw is not None else None
        )
        sess["user_code"] = str(device_data["user_code"])
        sess["code_verifier"] = verifier
        sess["state"] = state
        sess["portal_base_url"] = portal_base_url
        sess["client_id"] = MINIMAX_OAUTH_CLIENT_ID
        sess["region"] = "global"
        # `expired_in` from MiniMax is overloaded — could be a unix-ms
        # timestamp OR a seconds-from-now duration. Mirror the heuristic
        # in _minimax_poll_token. Stash the raw value for the poller;
        # compute a derived expires_at + UI-friendly expires_in seconds.
        expired_in_raw = int(device_data["expired_in"])
        sess["expired_in_raw"] = expired_in_raw
        if expired_in_raw > 1_000_000_000_000:  # likely unix-ms
            expires_at_ts = expired_in_raw / 1000.0
            expires_in_seconds = max(0, int(expires_at_ts - time.time()))
        else:
            expires_at_ts = time.time() + expired_in_raw
            expires_in_seconds = expired_in_raw
        sess["expires_at"] = expires_at_ts
        threading.Thread(
            target=_minimax_poller,
            args=(sid,),
            daemon=True,
            name=f"oauth-poll-{sid[:6]}",
        ).start()
        return {
            "session_id": sid,
            "flow": "device_code",
            "user_code": str(device_data["user_code"]),
            "verification_url": str(device_data["verification_uri"]),
            "expires_in": expires_in_seconds,
            "poll_interval": max(2, (sess["interval_ms"] or 2000) // 1000),
        }

    raise HTTPException(status_code=400, detail=f"Provider {provider_id} does not support device-code flow")


# xAI Grok OAuth uses a loopback-redirect PKCE flow (RFC 8252). Unlike the
# device-code providers there is no user_code to display: the local backend
# binds a 127.0.0.1 callback server, the client opens the authorize URL in
# the browser, and the redirect lands back on the loopback listener. The
# background worker waits for that callback, exchanges the code, and persists
# the tokens exactly like `hermes auth add xai-oauth`.
_XAI_LOOPBACK_TIMEOUT_SECONDS = 300.0


def _start_xai_loopback_flow(profile: Optional[str] = None) -> Dict[str, Any]:
    """Begin the xAI loopback PKCE flow.

    Binds the local callback server, builds the authorize URL, and spawns a
    background worker that waits for the redirect and finishes the exchange.
    Returns the authorize URL for the client to open in the browser.
    """
    from hermes_cli import auth as hauth

    discovery = hauth._xai_oauth_discovery()
    server, thread, callback_result, redirect_uri = hauth._xai_start_callback_server()
    try:
        hauth._xai_validate_loopback_redirect_uri(redirect_uri)
        verifier = hauth._oauth_pkce_code_verifier()
        challenge = hauth._oauth_pkce_code_challenge(verifier)
        state = secrets.token_hex(16)
        nonce = secrets.token_hex(16)
        authorize_url = hauth._xai_oauth_build_authorize_url(
            authorization_endpoint=discovery["authorization_endpoint"],
            redirect_uri=redirect_uri,
            code_challenge=challenge,
            state=state,
            nonce=nonce,
        )
    except Exception:
        # Binding succeeded but URL construction failed — release the socket
        # and join the serving thread so we don't leak a listener (or a
        # lingering daemon thread) on the loopback port.
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            pass
        try:
            thread.join(timeout=1.0)
        except Exception:
            pass
        raise

    sid, sess = _new_oauth_session("xai-oauth", "loopback", profile=profile)
    sess["server"] = server
    sess["thread"] = thread
    sess["callback_result"] = callback_result
    sess["redirect_uri"] = redirect_uri
    sess["verifier"] = verifier
    sess["challenge"] = challenge
    sess["state"] = state
    sess["token_endpoint"] = discovery["token_endpoint"]
    sess["discovery"] = discovery
    sess["expires_at"] = time.time() + _XAI_LOOPBACK_TIMEOUT_SECONDS
    threading.Thread(
        target=_xai_loopback_worker, args=(sid,), daemon=True,
        name=f"oauth-xai-{sid[:6]}",
    ).start()
    return {
        "session_id": sid,
        "flow": "loopback",
        "auth_url": authorize_url,
        "expires_in": int(_XAI_LOOPBACK_TIMEOUT_SECONDS),
    }


def _xai_loopback_worker(session_id: str) -> None:
    """Wait for the xAI loopback callback, exchange the code, persist tokens."""
    from datetime import datetime, timezone

    from hermes_cli import auth as hauth

    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
    if not sess:
        return

    def _fail(message: str) -> None:
        with _oauth_sessions_lock:
            s = _oauth_sessions.get(session_id)
            if s is not None:
                s["status"] = "error"
                s["error_message"] = message

    def _cancelled() -> bool:
        # The session is removed from the registry when the user cancels
        # (DELETE /sessions/{id}). If that happened while we were blocked on
        # the callback or token exchange, abort instead of persisting tokens
        # the user no longer wants.
        with _oauth_sessions_lock:
            return session_id not in _oauth_sessions

    try:
        callback = hauth._xai_wait_for_callback(
            sess["server"],
            sess["thread"],
            sess["callback_result"],
            timeout_seconds=_XAI_LOOPBACK_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        _fail(f"xAI authorization timed out: {exc}")
        return

    if _cancelled():
        return

    if callback.get("error"):
        detail = callback.get("error_description") or callback["error"]
        _fail(f"xAI authorization failed: {detail}")
        return
    if callback.get("state") != sess["state"]:
        _fail("xAI authorization failed: state mismatch.")
        return
    code = str(callback.get("code") or "").strip()
    if not code:
        _fail("xAI authorization failed: missing authorization code.")
        return

    try:
        payload = hauth._xai_oauth_exchange_code_for_tokens(
            token_endpoint=sess["token_endpoint"],
            code=code,
            redirect_uri=sess["redirect_uri"],
            code_verifier=sess["verifier"],
            code_challenge=sess["challenge"],
        )
        access_token = str(payload.get("access_token", "") or "").strip()
        refresh_token = str(payload.get("refresh_token", "") or "").strip()
        if not access_token or not refresh_token:
            _fail("xAI token exchange did not return the expected tokens.")
            return
        base_url = hauth._xai_validate_inference_base_url(
            os.getenv("HERMES_XAI_BASE_URL", "").strip().rstrip("/")
            or os.getenv("XAI_BASE_URL", "").strip().rstrip("/"),
            fallback=hauth.DEFAULT_XAI_OAUTH_BASE_URL,
        )
        last_refresh = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        tokens = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "id_token": str(payload.get("id_token", "") or "").strip(),
            "expires_in": payload.get("expires_in"),
            "token_type": str(payload.get("token_type") or "Bearer").strip() or "Bearer",
        }
        if _cancelled():
            return
        with _profile_scope(_oauth_session_profile(session_id)):
            hauth._save_xai_oauth_tokens(
                tokens,
                discovery=sess.get("discovery"),
                redirect_uri=sess["redirect_uri"],
                last_refresh=last_refresh,
            )
            _add_xai_oauth_pool_entry(access_token, refresh_token, base_url, last_refresh)
    except Exception as exc:
        _fail(f"xAI token exchange failed: {exc}")
        return

    with _oauth_sessions_lock:
        s = _oauth_sessions.get(session_id)
        if s is not None:
            s["status"] = "approved"
    _log.info("oauth/loopback: xai-oauth login completed (session=%s)", session_id)


def _add_xai_oauth_pool_entry(
    access_token: str, refresh_token: str, base_url: str, last_refresh: str
) -> None:
    """Mirror `hermes auth add xai-oauth`'s credential-pool insert.

    Best-effort: the auth-store write in _save_xai_oauth_tokens is the source
    of truth for runtime resolution; the pool entry only matters for the
    rotation strategy.
    """
    try:
        import uuid

        from agent.credential_pool import (
            PooledCredential,
            load_pool,
            AUTH_TYPE_OAUTH,
            SOURCE_MANUAL,
        )
        pool = load_pool("xai-oauth")
        existing = [
            e for e in pool.entries()
            if getattr(e, "source", "").startswith(f"{SOURCE_MANUAL}:dashboard_xai_pkce")
        ]
        for e in existing:
            try:
                pool.remove_entry(getattr(e, "id", ""))
            except Exception:
                pass
        entry = PooledCredential(
            provider="xai-oauth",
            id=uuid.uuid4().hex[:6],
            label="dashboard PKCE",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source=f"{SOURCE_MANUAL}:dashboard_xai_pkce",
            access_token=access_token,
            refresh_token=refresh_token,
            base_url=base_url,
            last_refresh=last_refresh,
        )
        pool.add_entry(entry)
    except Exception as e:
        _log.warning("xai-oauth pool add (dashboard) failed: %s", e)


def _nous_poller(session_id: str) -> None:
    """Background poller that drives a Nous device-code flow to completion."""
    from hermes_cli.auth import (
        _poll_for_token,
        refresh_nous_oauth_from_state,
    )
    from datetime import datetime, timezone
    import httpx
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
    if not sess:
        return
    portal_base_url = sess["portal_base_url"]
    client_id = sess["client_id"]
    device_code = sess["device_code"]
    interval = sess["interval"]
    scope = sess.get("scope")
    expires_in = max(60, int(sess["expires_at"] - time.time()))
    try:
        with httpx.Client(timeout=httpx.Timeout(15.0), headers={"Accept": "application/json"}) as client:
            token_data = _poll_for_token(
                client=client,
                portal_base_url=portal_base_url,
                client_id=client_id,
                device_code=device_code,
                expires_in=expires_in,
                poll_interval=interval,
            )
        # Same post-processing as _nous_device_code_login (validate/refresh JWT)
        now = datetime.now(timezone.utc)
        token_ttl = int(token_data.get("expires_in") or 0)
        auth_state = {
            "portal_base_url": portal_base_url,
            "inference_base_url": token_data.get("inference_base_url"),
            "client_id": client_id,
            "scope": token_data.get("scope") or scope,
            "token_type": token_data.get("token_type", "Bearer"),
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token"),
            "obtained_at": now.isoformat(),
            "expires_at": (
                datetime.fromtimestamp(now.timestamp() + token_ttl, tz=timezone.utc).isoformat()
                if token_ttl else None
            ),
            "expires_in": token_ttl,
        }
        with _profile_scope(_oauth_session_profile(session_id)):
            full_state = refresh_nous_oauth_from_state(
                auth_state,
                timeout_seconds=15.0,
                force_refresh=False,
            )
            from hermes_cli.auth import persist_nous_credentials
            persist_nous_credentials(full_state)
        with _oauth_sessions_lock:
            sess["status"] = "approved"
        _log.info("oauth/device: nous login completed (session=%s)", session_id)
    except Exception as e:
        _log.warning("nous device-code poll failed (session=%s): %s", session_id, e)
        with _oauth_sessions_lock:
            sess["status"] = "error"
            sess["error_message"] = str(e)


def _minimax_poller(session_id: str) -> None:
    """Background poller that drives a MiniMax OAuth flow to completion.

    Mirrors `_nous_poller` but calls the MiniMax-specific token endpoint,
    which uses a PKCE-style ``code_verifier`` + ``user_code`` rather than
    the ``device_code`` field used by Nous. On success, builds the same
    auth_state dict that ``_minimax_oauth_login`` (the CLI flow) builds
    and persists via ``_minimax_save_auth_state`` — so the dashboard
    path leaves the system in the same state as
    ``hermes auth add minimax-oauth``.
    """
    from hermes_cli.auth import (
        _minimax_poll_token,
        _minimax_resolve_token_expiry_unix,
        _minimax_save_auth_state,
        MINIMAX_OAUTH_GLOBAL_INFERENCE,
        MINIMAX_OAUTH_SCOPE,
    )
    from datetime import datetime, timezone
    import httpx
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
    if not sess:
        return
    portal_base_url = sess["portal_base_url"]
    client_id = sess["client_id"]
    user_code = sess["user_code"]
    code_verifier = sess["code_verifier"]
    interval_ms = sess.get("interval_ms")
    expired_in_raw = sess["expired_in_raw"]
    try:
        with httpx.Client(
            timeout=httpx.Timeout(15.0),
            headers={"Accept": "application/json"},
            follow_redirects=True,
        ) as client:
            token_data = _minimax_poll_token(
                client=client,
                portal_base_url=portal_base_url,
                client_id=client_id,
                user_code=user_code,
                code_verifier=code_verifier,
                expired_in=expired_in_raw,
                interval_ms=interval_ms,
            )
        # Build the auth_state dict in the same shape as the CLI flow's
        # `_minimax_oauth_login` so `_minimax_save_auth_state` writes
        # the canonical record. Region is fixed to "global" for the
        # dashboard path; cn-region operators can still use the CLI
        # flow which supports `--region cn`.
        now = datetime.now(timezone.utc)
        expires_at_ts = _minimax_resolve_token_expiry_unix(
            int(token_data["expired_in"]), now=now,
        )
        expires_in_s = max(0, int(expires_at_ts - now.timestamp()))
        auth_state = {
            "provider": "minimax-oauth",
            "region": sess.get("region", "global"),
            "portal_base_url": portal_base_url,
            "inference_base_url": MINIMAX_OAUTH_GLOBAL_INFERENCE,
            "client_id": client_id,
            "scope": MINIMAX_OAUTH_SCOPE,
            "token_type": token_data.get("token_type", "Bearer"),
            "access_token": token_data["access_token"],
            "refresh_token": token_data["refresh_token"],
            "resource_url": token_data.get("resource_url"),
            "obtained_at": now.isoformat(),
            "expires_at": datetime.fromtimestamp(
                expires_at_ts, tz=timezone.utc
            ).isoformat(),
            "expires_in": expires_in_s,
        }
        with _profile_scope(_oauth_session_profile(session_id)):
            _minimax_save_auth_state(auth_state)
        with _oauth_sessions_lock:
            sess["status"] = "approved"
        _log.info("oauth/device: minimax login completed (session=%s)", session_id)
    except Exception as e:
        _log.warning("minimax device-code poll failed (session=%s): %s", session_id, e)
        with _oauth_sessions_lock:
            sess["status"] = "error"
            sess["error_message"] = str(e)


def _codex_full_login_worker(session_id: str) -> None:
    """Run the complete OpenAI Codex device-code flow.

    Codex doesn't use the standard OAuth device-code endpoints; it has its
    own ``/api/accounts/deviceauth/usercode`` (JSON body, returns
    ``device_auth_id``) and ``/api/accounts/deviceauth/token`` (JSON body
    polled until 200). On success the response carries an
    ``authorization_code`` + ``code_verifier`` that get exchanged at
    CODEX_OAUTH_TOKEN_URL with grant_type=authorization_code.

    The flow is replicated inline (rather than calling
    _codex_device_code_login) because that helper prints/blocks/polls in a
    single function — we need to surface the user_code to the dashboard the
    moment we receive it, well before polling completes.
    """
    try:
        import httpx
        from hermes_cli.auth import (
            CODEX_OAUTH_CLIENT_ID,
            CODEX_OAUTH_TOKEN_URL,
            DEFAULT_CODEX_BASE_URL,
        )
        issuer = "https://auth.openai.com"

        # Step 1: request device code
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            resp = client.post(
                f"{issuer}/api/accounts/deviceauth/usercode",
                json={"client_id": CODEX_OAUTH_CLIENT_ID},
                headers={"Content-Type": "application/json"},
            )
        if resp.status_code != 200:
            raise RuntimeError(f"deviceauth/usercode returned {resp.status_code}")
        device_data = resp.json()
        user_code = device_data.get("user_code", "")
        device_auth_id = device_data.get("device_auth_id", "")
        poll_interval = max(3, int(device_data.get("interval", "5")))
        if not user_code or not device_auth_id:
            raise RuntimeError("device-code response missing user_code or device_auth_id")
        verification_url = f"{issuer}/codex/device"
        with _oauth_sessions_lock:
            sess = _oauth_sessions.get(session_id)
            if not sess:
                return
            sess["user_code"] = user_code
            sess["verification_url"] = verification_url
            sess["device_auth_id"] = device_auth_id
            sess["interval"] = poll_interval
            sess["expires_in"] = 15 * 60  # OpenAI's effective limit
            sess["expires_at"] = time.time() + sess["expires_in"]

        # Step 2: poll until authorized
        deadline = time.monotonic() + sess["expires_in"]
        code_resp = None
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            while time.monotonic() < deadline:
                time.sleep(poll_interval)
                poll = client.post(
                    f"{issuer}/api/accounts/deviceauth/token",
                    json={"device_auth_id": device_auth_id, "user_code": user_code},
                    headers={"Content-Type": "application/json"},
                )
                if poll.status_code == 200:
                    code_resp = poll.json()
                    break
                if poll.status_code in {403, 404}:
                    continue  # user hasn't authorized yet
                raise RuntimeError(f"deviceauth/token poll returned {poll.status_code}")

        if code_resp is None:
            with _oauth_sessions_lock:
                sess["status"] = "expired"
                sess["error_message"] = "Device code expired before approval"
            return

        # Step 3: exchange authorization_code for tokens
        authorization_code = code_resp.get("authorization_code", "")
        code_verifier = code_resp.get("code_verifier", "")
        if not authorization_code or not code_verifier:
            raise RuntimeError("device-auth response missing authorization_code/code_verifier")
        with httpx.Client(timeout=httpx.Timeout(15.0)) as client:
            token_resp = client.post(
                CODEX_OAUTH_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": authorization_code,
                    "redirect_uri": f"{issuer}/deviceauth/callback",
                    "client_id": CODEX_OAUTH_CLIENT_ID,
                    "code_verifier": code_verifier,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if token_resp.status_code != 200:
            raise RuntimeError(f"token exchange returned {token_resp.status_code}")
        tokens = token_resp.json()
        access_token = tokens.get("access_token", "")
        refresh_token = tokens.get("refresh_token", "")
        if not access_token:
            raise RuntimeError("token exchange did not return access_token")

        from hermes_cli.auth import _save_codex_tokens

        with _profile_scope(_oauth_session_profile(session_id)):
            _save_codex_tokens({
                "access_token": access_token,
                "refresh_token": refresh_token,
            })
        with _oauth_sessions_lock:
            sess["status"] = "approved"
        _log.info("oauth/device: openai-codex login completed (session=%s)", session_id)
    except Exception as e:
        _log.warning("codex device-code worker failed (session=%s): %s", session_id, e)
        with _oauth_sessions_lock:
            s = _oauth_sessions.get(session_id)
            if s:
                s["status"] = "error"
                s["error_message"] = str(e)


@app.post("/api/providers/oauth/{provider_id}/start")
async def start_oauth_login(
    provider_id: str,
    request: Request,
    profile: Optional[str] = None,
):
    """Initiate an OAuth login flow. Token-protected."""
    _require_token(request)
    _gc_oauth_sessions()
    _validate_oauth_profile(profile)
    valid = {p["id"] for p in _OAUTH_PROVIDER_CATALOG}
    if provider_id not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown provider {provider_id}")
    catalog_entry = next(p for p in _OAUTH_PROVIDER_CATALOG if p["id"] == provider_id)
    if catalog_entry["flow"] == "external":
        raise HTTPException(
            status_code=400,
            detail=f"{provider_id} uses an external CLI; run `{catalog_entry['cli_command']}` manually",
        )
    try:
        # The pkce branch is gated on provider_id == "anthropic" because
        # `_start_anthropic_pkce()` is hardcoded to the Anthropic flow.
        # Routing any other future pkce-flagged provider through it would
        # silently launch the Anthropic OAuth flow (the bug fixed in this
        # change for MiniMax). New PKCE providers must add their own
        # start function and an explicit branch here.
        if catalog_entry["flow"] == "pkce" and provider_id == "anthropic":
            return _start_anthropic_pkce(profile=profile)
        if catalog_entry["flow"] == "device_code":
            return await _start_device_code_flow(provider_id, profile=profile)
        if catalog_entry["flow"] == "loopback" and provider_id == "xai-oauth":
            return await asyncio.get_running_loop().run_in_executor(
                None, _start_xai_loopback_flow, profile,
            )
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("oauth/start %s failed", provider_id)
        raise HTTPException(status_code=500, detail=str(e))
    raise HTTPException(status_code=400, detail="Unsupported flow")


class OAuthSubmitBody(BaseModel):
    session_id: str
    code: str


@app.post("/api/providers/oauth/{provider_id}/submit")
async def submit_oauth_code(
    provider_id: str,
    body: OAuthSubmitBody,
    request: Request,
    profile: Optional[str] = None,
):
    """Submit the auth code for PKCE flows. Token-protected."""
    _require_token(request)
    if provider_id == "anthropic":
        return await asyncio.get_running_loop().run_in_executor(
            None, _submit_anthropic_pkce, body.session_id, body.code, profile,
        )
    raise HTTPException(status_code=400, detail=f"submit not supported for {provider_id}")


@app.get("/api/providers/oauth/{provider_id}/poll/{session_id}")
async def poll_oauth_session(
    provider_id: str,
    session_id: str,
    profile: Optional[str] = None,
):
    """Poll a session's status (no auth — read-only state).

    Shared by the device-code flows (Nous, OpenAI Codex, MiniMax) and the
    loopback flow (xAI Grok). Both surface progress through the same
    background-worker-updated ``status`` field, so a single poll endpoint
    serves them all.
    """
    with _oauth_sessions_lock:
        sess = _oauth_sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if sess["provider"] != provider_id:
        raise HTTPException(status_code=400, detail="Provider mismatch for session")
    return {
        "session_id": session_id,
        "status": sess["status"],
        "error_message": sess.get("error_message"),
        "expires_at": sess.get("expires_at"),
    }


@app.delete("/api/providers/oauth/sessions/{session_id}")
async def cancel_oauth_session(
    session_id: str,
    request: Request,
    profile: Optional[str] = None,
):
    """Cancel a pending OAuth session. Token-protected."""
    _require_token(request)
    with _oauth_sessions_lock:
        sess = _oauth_sessions.pop(session_id, None)
    if sess is None:
        return {"ok": False, "message": "session not found"}
    # Loopback sessions own a bound 127.0.0.1 callback server. Without an
    # explicit shutdown the worker would keep that port held until
    # _xai_wait_for_callback times out (up to 5 min). Free it immediately so
    # an orphaned listener can't block a subsequent sign-in attempt.
    if sess.get("flow") == "loopback":
        # The worker is blocked in _xai_wait_for_callback, which polls
        # callback_result rather than the server state. Flag the result as
        # cancelled so that loop returns on its next tick instead of spinning
        # until the timeout — otherwise repeated cancel/retry piles up daemon
        # threads. (_cancelled() in the worker then short-circuits before any
        # persist.)
        result = sess.get("callback_result")
        if isinstance(result, dict):
            result["error"] = result.get("error") or "cancelled"
        server = sess.get("server")
        thread = sess.get("thread")
        try:
            if server is not None:
                server.shutdown()
                server.server_close()
        except Exception:
            pass
        try:
            if thread is not None:
                thread.join(timeout=1.0)
        except Exception:
            pass
    return {"ok": True, "session_id": session_id}


# ---------------------------------------------------------------------------
# Session detail endpoints
# ---------------------------------------------------------------------------



def _session_latest_descendant(session_id: str):
    """Resolve a session id to the newest child leaf session.

    /model may create child sessions. Dashboard refresh should continue the
    newest child instead of reopening the old parent.
    """
    from hermes_state import SessionDB

    def row_get(row, key, index):
        if isinstance(row, dict):
            return row.get(key)
        try:
            return row[key]
        except Exception:
            try:
                return row[index]
            except Exception:
                return None

    db = SessionDB()
    try:
        sid = db.resolve_session_id(session_id)
        if not sid or not db.get_session(sid):
            return None, []

        conn = (
            getattr(db, "conn", None)
            or getattr(db, "_conn", None)
            or getattr(db, "connection", None)
            or getattr(db, "_connection", None)
        )

        rows = []
        if conn is not None:
            raw_rows = conn.execute(
                "SELECT id, parent_session_id, started_at FROM sessions"
            ).fetchall()
            for row in raw_rows:
                rows.append({
                    "id": row_get(row, "id", 0),
                    "parent_session_id": row_get(row, "parent_session_id", 1),
                    "started_at": row_get(row, "started_at", 2),
                })
        else:
            rows = db.list_sessions_rich(limit=10000, offset=0)

        children = {}
        for row in rows:
            rid = row.get("id")
            parent = row.get("parent_session_id")
            if rid and parent:
                children.setdefault(parent, []).append(row)

        def started(row):
            try:
                return float(row.get("started_at") or 0)
            except Exception:
                return 0.0

        current = sid
        path = [sid]
        seen = {sid}

        while children.get(current):
            candidates = [r for r in children[current] if r.get("id") not in seen]
            if not candidates:
                break
            candidates.sort(key=started, reverse=True)
            current = candidates[0]["id"]
            path.append(current)
            seen.add(current)

        return current, path
    finally:
        db.close()


# CRITICAL — every literal-path route below MUST be declared BEFORE the
# templated ``/api/sessions/{session_id}`` family that follows. FastAPI/
# Starlette match routes in registration order, and the ``{session_id}``
# pattern is unconstrained — it would otherwise swallow e.g.
# ``DELETE /api/sessions/empty``, ``POST /api/sessions/bulk-delete``, or
# ``GET /api/sessions/stats`` as "operate on the session with id
# 'empty'" / "'bulk-delete'" / "'stats'", which would 404 (or worse,
# succeed and delete the wrong row). Same story as the older
# ``/api/sessions/search`` endpoint up at line ~1191. If you split or
# reorder this block, move every route in it together.
class BulkDeleteSessions(BaseModel):
    ids: List[str]
    profile: Optional[str] = None


@app.post("/api/sessions/bulk-delete")
async def bulk_delete_sessions_endpoint(body: BulkDeleteSessions):
    """Delete every session in ``body.ids`` in a single DB transaction.

    Backs the dashboard's bulk-select-and-delete flow on the sessions
    page. POST (not DELETE) because most HTTP clients refuse to send a
    request body on DELETE and a body is the natural shape for a list
    of IDs — Starlette accepts both, but POSTing a list keeps proxies,
    curl, and the browser ``fetch`` API consistent.

    Per-row contract matches :meth:`SessionDB.delete_sessions`:

    * Unknown IDs are silently skipped (the response ``deleted`` count
      reflects what really happened, not the input length). This is
      deliberate — UI selection state can race against another tab's
      delete, and we'd rather succeed-on-the-rest than fail-the-whole-
      batch.
    * Children of every deleted parent are orphaned, not cascade-
      deleted.
    * Active and archived sessions ARE deleted when explicitly
      selected — unlike ``DELETE /api/sessions/empty``, the user
      hand-picked the rows so we trust the selection.
    * Like the other session-delete endpoints, this does NOT pass a
      ``sessions_dir`` through; on-disk transcript / request-dump
      cleanup runs at the CLI/agent layer on the next prune pass.

    The response carries the actual deleted count, so the dashboard
    can surface it in a toast. The IDs that were removed are not
    echoed back because the client already knows what it asked to
    delete (unknown IDs are silently skipped — see contract above)
    and can prune its in-memory list directly from the request.
    """
    # Enforce a hard cap so a runaway/typo'd selection can't lock the
    # DB writer for an extended window. The dashboard pages 20 rows
    # at a time; 500 covers a "select all on every page in a
    # reasonable scrollback" worst case without opening the door to
    # multi-thousand-row transactions.
    if len(body.ids) > 500:
        raise HTTPException(
            status_code=400,
            detail="ids must contain at most 500 entries",
        )
    db = _open_session_db_for_profile(body.profile)
    try:
        deleted = db.delete_sessions(body.ids)
        return {"ok": True, "deleted": deleted}
    finally:
        db.close()


@app.get("/api/sessions/empty/count")
async def count_empty_sessions_endpoint(profile: Optional[str] = None):
    """Return the number of empty, ended, non-archived sessions.

    Drives the dashboard's "Delete empty (N)" button — when N is 0 the
    UI hides the affordance so users aren't presented with a button
    that does nothing. Cheap, single-COUNT query.
    """
    db = _open_session_db_for_profile(profile)
    try:
        return {"count": db.count_empty_sessions()}
    finally:
        db.close()


@app.delete("/api/sessions/empty")
async def delete_empty_sessions_endpoint(profile: Optional[str] = None):
    """Delete every empty (``message_count == 0``), ended,
    non-archived session in a single transaction.

    Safety contract mirrors :meth:`SessionDB.delete_empty_sessions`:

    * Active sessions are skipped (``ended_at IS NULL``) so a live
      agent isn't yanked mid-handshake.
    * Archived sessions are skipped — the user explicitly chose to
      keep those rows.
    * Children of deleted parents are orphaned, not cascade-deleted.

    Like the single-session ``DELETE /api/sessions/{id}`` endpoint
    below, this doesn't pass a ``sessions_dir`` through — the on-disk
    transcript / request-dump cleanup is wired at the CLI/agent layer
    but the web server historically leaves file cleanup to the next
    prune-on-startup pass. Matching that pre-existing trade-off keeps
    the two delete endpoints' DB-vs-disk behaviour consistent.
    """
    db = _open_session_db_for_profile(profile)
    try:
        deleted = db.delete_empty_sessions()
        return {"ok": True, "deleted": deleted}
    finally:
        db.close()


@app.get("/api/sessions/stats")
async def get_session_stats(profile: Optional[str] = None):
    """Session-store statistics for the Sessions page (mirrors `hermes sessions stats`).

    Registered before ``/api/sessions/{session_id}`` so the literal ``stats``
    path isn't captured as a session id by the parameterized route.
    """
    db = _open_session_db_for_profile(profile)
    try:
        total = db.session_count(include_archived=True)
        active_store = db.session_count(include_archived=False)
        archived = db.session_count(archived_only=True)
        messages = db.message_count()
        by_source: Dict[str, int] = {}
        try:
            for s in db.list_sessions_rich(limit=10000, include_archived=True):
                src = str(s.get("source") or "cli")
                by_source[src] = by_source.get(src, 0) + 1
        except Exception:
            pass
        return {
            "total": total,
            "active_store": active_store,
            "archived": archived,
            "messages": messages,
            "by_source": by_source,
        }
    finally:
        db.close()


def _open_session_db_for_profile(profile: Optional[str]):
    """Open a SessionDB for read paths, optionally for another profile.

    ``profile`` None/empty → this process's own ``state.db`` (the common,
    single-profile case). A named profile opens that profile's on-disk
    ``state.db`` directly so the primary backend can serve cross-profile reads
    (transcripts, detail) without spawning that profile's backend.
    """
    from hermes_state import SessionDB
    if not profile:
        return SessionDB()
    _name, home = _cron_profile_home(profile)
    return SessionDB(db_path=Path(home) / "state.db")


@app.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str, profile: Optional[str] = None):
    db = _open_session_db_for_profile(profile)
    try:
        sid = db.resolve_session_id(session_id)
        session = db.get_session(sid) if sid else None
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        if profile:
            session["profile"] = _cron_profile_home(profile)[0]
        return session
    finally:
        db.close()



@app.get("/api/sessions/{session_id}/latest-descendant")
async def get_session_latest_descendant(session_id: str):
    latest, path = _session_latest_descendant(session_id)
    if not latest:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "requested_session_id": path[0] if path else session_id,
        "session_id": latest,
        "path": path,
        "changed": bool(path and latest != path[0]),
    }

@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, profile: Optional[str] = None):
    db = _open_session_db_for_profile(profile)
    try:
        sid = db.resolve_session_id(session_id)
        if not sid:
            raise HTTPException(status_code=404, detail="Session not found")
        sid = db.resolve_resume_session_id(sid)
        messages = db.get_messages(sid)
        return {"session_id": sid, "messages": messages}
    finally:
        db.close()


@app.delete("/api/sessions/{session_id}")
async def delete_session_endpoint(session_id: str, profile: Optional[str] = None):
    # ``profile`` deletes a session belonging to another (local) profile by
    # opening its state.db directly. Remote profiles never reach here — the
    # desktop routes their DELETE to the remote backend. Omit for current/default.
    db = _open_session_db_for_profile(profile)
    try:
        # Resolve exact ids / unique prefixes like every other session endpoint
        # (detail, messages, rename, export all do). A session that no longer
        # exists is an idempotent success: DELETE's contract is "ensure it's
        # gone", and the desktop optimistically removes the row then RESTORES it
        # on any error — so a 404 on an already-absent row resurrected a ghost
        # row and surfaced "session not found". /goal + auto-compression churn
        # leaves transient empty rows (reaped by empty-session hygiene) that
        # race the sidebar snapshot, which is exactly when this fired. Mirrors
        # the bulk-delete endpoint, which already treats ghost ids as success.
        sid = db.resolve_session_id(session_id)
        if not sid:
            return {"ok": True, "already_absent": True}
        db.delete_session(sid)
        return {"ok": True}
    finally:
        db.close()


class SessionRename(BaseModel):
    title: Optional[str] = None
    archived: Optional[bool] = None
    # Mutate a session belonging to another profile (opens its state.db). Omit
    # for the current/default profile.
    profile: Optional[str] = None


@app.patch("/api/sessions/{session_id}")
async def rename_session_endpoint(session_id: str, body: SessionRename):
    """Update a session: rename (or clear its title) and/or archive it.

    ``title`` renames (empty/null clears the title); ``archived`` soft-hides or
    restores the session. Either field may be omitted. ``profile`` targets
    another profile's session.
    """
    db = _open_session_db_for_profile(body.profile)
    try:
        sid = db.resolve_session_id(session_id)
        if not sid:
            raise HTTPException(status_code=404, detail="Session not found")
        if body.title is None and body.archived is None:
            raise HTTPException(
                status_code=400,
                detail="Nothing to update; provide 'title' and/or 'archived'.",
            )
        if body.title is not None:
            try:
                db.set_session_title(sid, body.title or "")
            except ValueError as e:
                # Title too long, invalid characters, or already in use.
                raise HTTPException(status_code=400, detail=str(e))
        if body.archived is not None:
            db.set_session_archived(sid, body.archived)
        result = {"ok": True, "title": db.get_session_title(sid) or ""}
        if body.archived is not None:
            result["archived"] = bool(body.archived)
        return result
    finally:
        db.close()


@app.get("/api/sessions/{session_id}/export")
async def export_session_endpoint(session_id: str, profile: Optional[str] = None):
    """Export a single session (metadata + messages) as JSON."""
    db = _open_session_db_for_profile(profile)
    try:
        sid = db.resolve_session_id(session_id)
        if not sid:
            raise HTTPException(status_code=404, detail="Session not found")
        data = db.export_session(sid)
        if data is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return data
    finally:
        db.close()


class SessionPrune(BaseModel):
    older_than_days: int = 90
    source: Optional[str] = None
    profile: Optional[str] = None


@app.post("/api/sessions/prune")
async def prune_sessions_endpoint(body: SessionPrune):
    """Delete ended sessions older than N days (mirrors `hermes sessions prune`)."""
    if body.older_than_days < 1:
        raise HTTPException(status_code=400, detail="older_than_days must be >= 1")
    profile_home = _cron_profile_home(body.profile)[1] if body.profile else get_hermes_home()
    db = _open_session_db_for_profile(body.profile)
    try:
        sessions_dir = profile_home / "sessions"
        removed = db.prune_sessions(
            older_than_days=body.older_than_days,
            source=(body.source or None),
            sessions_dir=sessions_dir if sessions_dir.exists() else None,
        )
        return {"ok": True, "removed": removed}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Log viewer endpoint
# ---------------------------------------------------------------------------


@app.get("/api/logs")
async def get_logs(
    file: str = "agent",
    lines: int = 100,
    level: Optional[str] = None,
    component: Optional[str] = None,
    search: Optional[str] = None,
):
    from hermes_cli.logs import _read_tail, LOG_FILES

    log_name = LOG_FILES.get(file)
    if not log_name:
        raise HTTPException(status_code=400, detail=f"Unknown log file: {file}")
    log_path = get_hermes_home() / "logs" / log_name
    if not log_path.exists():
        return {"file": file, "lines": []}

    try:
        from hermes_logging import COMPONENT_PREFIXES
    except ImportError:
        COMPONENT_PREFIXES = {}

    # Normalize "ALL" / "all" / empty → no filter. _matches_filters treats an
    # empty tuple as "must match a prefix" (startswith(()) is always False),
    # so passing () instead of None silently drops every line.
    min_level = level if level and level.upper() != "ALL" else None
    if component and component.lower() != "all":
        comp_prefixes = COMPONENT_PREFIXES.get(component)
        if comp_prefixes is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown component: {component}. "
                       f"Available: {', '.join(sorted(COMPONENT_PREFIXES))}",
            )
    else:
        comp_prefixes = None

    has_filters = bool(min_level or comp_prefixes or search)
    result = _read_tail(
        log_path, min(lines, 500) if not search else 2000,
        has_filters=has_filters,
        min_level=min_level,
        component_prefixes=comp_prefixes,
    )
    # Post-filter by search term (case-insensitive substring match).
    # _read_tail doesn't support free-text search, so we filter here and
    # trim to the requested line count afterward.
    if search:
        needle = search.lower()
        result = [l for l in result if needle in l.lower()][-min(lines, 500):]
    return {"file": file, "lines": result}


# ---------------------------------------------------------------------------
# Cron job management endpoints
# ---------------------------------------------------------------------------


class CronJobCreate(BaseModel):
    prompt: str = ""
    schedule: str
    name: str = ""
    deliver: str = "local"
    skills: Optional[List[str]] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    base_url: Optional[str] = None
    script: Optional[str] = None
    context_from: Optional[Any] = None
    enabled_toolsets: Optional[List[str]] = None
    workdir: Optional[str] = None
    no_agent: bool = False


class CronJobUpdate(BaseModel):
    updates: dict


def _cron_optional_text(value: Any, *, strip_trailing_slash: bool = False) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if strip_trailing_slash:
        text = text.rstrip("/")
    return text or None


def _cron_string_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        raw_items = re.split(r"[\n,]", value)
    elif isinstance(value, (list, tuple)):
        raw_items = value
    else:
        return None
    items = [str(item).strip() for item in raw_items if str(item).strip()]
    return items or None


def _normalize_dashboard_cron_script(value: Any, profile_home: Path) -> Optional[str]:
    """Validate a dashboard-selected cron script against the profile sandbox."""
    text = _cron_optional_text(value)
    if not text:
        return None

    scripts_root = (profile_home / "scripts").resolve()
    raw_path = Path(text).expanduser()
    candidate = raw_path.resolve() if raw_path.is_absolute() else (scripts_root / raw_path).resolve()
    try:
        relative = candidate.relative_to(scripts_root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"script must be inside {scripts_root}",
        ) from exc
    if not candidate.exists():
        raise HTTPException(status_code=400, detail=f"script does not exist: {candidate}")
    if not candidate.is_file():
        raise HTTPException(status_code=400, detail=f"script is not a file: {candidate}")
    return str(relative)


def _validate_dashboard_cron_effective_job(job: Dict[str, Any]) -> None:
    prompt = _cron_optional_text(job.get("prompt"))
    script = _cron_optional_text(job.get("script"))
    skills = _cron_string_list(job.get("skills")) or _cron_string_list(job.get("skill"))
    no_agent = bool(job.get("no_agent"))

    if no_agent:
        if not script:
            raise HTTPException(
                status_code=400,
                detail="no_agent=True requires a script",
            )
        return

    if not (prompt or skills or script):
        raise HTTPException(
            status_code=400,
            detail="agent cron jobs require a prompt, skill, or script",
        )


def _normalize_dashboard_cron_updates(
    updates: Dict[str, Any],
    profile_home: Path,
) -> Dict[str, Any]:
    """Normalize dashboard JSON into cron.jobs.update_job's storage shape.

    This intentionally stays in the dashboard adapter layer: cron/jobs.py is the
    source of truth for scheduling behaviour; the dashboard only translates form
    payloads into the shapes that existing core functions already accept.
    """
    normalized = dict(updates or {})

    for key in ("model", "provider", "workdir"):
        if key in normalized:
            normalized[key] = _cron_optional_text(normalized[key])
    if "script" in normalized:
        normalized["script"] = _normalize_dashboard_cron_script(
            normalized["script"],
            profile_home,
        )
    if "base_url" in normalized:
        normalized["base_url"] = _cron_optional_text(
            normalized["base_url"], strip_trailing_slash=True
        )
    if "deliver" in normalized:
        normalized["deliver"] = _cron_optional_text(normalized["deliver"]) or "local"
    if "context_from" in normalized:
        normalized["context_from"] = _cron_string_list(normalized["context_from"])
    if "enabled_toolsets" in normalized:
        normalized["enabled_toolsets"] = _cron_string_list(normalized["enabled_toolsets"])
    return normalized


def _validate_dashboard_cron_context_from(
    refs: Optional[List[str]],
    profile_name: str,
) -> None:
    if not refs:
        return
    for ref in refs:
        if not _call_cron_for_profile(profile_name, "get_job", ref):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"context_from job '{ref}' not found in profile "
                    f"'{profile_name}'"
                ),
            )


_CRON_PROFILE_LOCK = threading.RLock()


def _cron_profile_dicts() -> List[Dict[str, Any]]:
    """Return dashboard profile records, falling back to a directory scan."""
    from hermes_cli import profiles as profiles_mod
    try:
        return [_profile_to_dict(p) for p in profiles_mod.list_profiles()]
    except Exception:
        _log.exception("Failed to list profiles for cron dashboard; falling back to directory scan")
        return _fallback_profile_dicts(profiles_mod)


def _cron_profile_home(profile: Optional[str]) -> Tuple[str, Path]:
    """Resolve a profile query value to (profile_name, HERMES_HOME)."""
    from hermes_cli import profiles as profiles_mod

    raw = (profile or "default").strip() or "default"
    try:
        canon = profiles_mod.normalize_profile_name(raw)
        profiles_mod.validate_profile_name(canon)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not profiles_mod.profile_exists(canon):
        raise HTTPException(status_code=404, detail=f"Profile '{canon}' does not exist.")
    return canon, profiles_mod.get_profile_dir(canon)


def _annotate_cron_job(job: Dict[str, Any], profile: str, home: Path) -> Dict[str, Any]:
    annotated = dict(job)
    annotated["profile"] = profile
    annotated["profile_name"] = profile
    annotated["hermes_home"] = str(home)
    annotated["is_default_profile"] = profile == "default"
    return annotated


def _call_cron_for_profile(target_profile: Optional[str], func_name: str, *args, **kwargs):
    """Run cron.jobs helpers against the selected profile's cron directory.

    cron.jobs keeps CRON_DIR/JOBS_FILE/OUTPUT_DIR as module globals resolved
    from the process HERMES_HOME at import time. The dashboard is a single
    process that can inspect many profiles, so temporarily retarget those
    globals while holding a lock and restore them immediately after the call.
    """
    profile_name, home = _cron_profile_home(target_profile)
    with _CRON_PROFILE_LOCK:
        from cron import jobs as cron_jobs
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        old_cron_dir = cron_jobs.CRON_DIR
        old_jobs_file = cron_jobs.JOBS_FILE
        old_output_dir = cron_jobs.OUTPUT_DIR
        token = set_hermes_home_override(str(home))
        cron_jobs.CRON_DIR = home / "cron"
        cron_jobs.JOBS_FILE = cron_jobs.CRON_DIR / "jobs.json"
        cron_jobs.OUTPUT_DIR = cron_jobs.CRON_DIR / "output"
        try:
            result = getattr(cron_jobs, func_name)(*args, **kwargs)
        finally:
            cron_jobs.CRON_DIR = old_cron_dir
            cron_jobs.JOBS_FILE = old_jobs_file
            cron_jobs.OUTPUT_DIR = old_output_dir
            reset_hermes_home_override(token)

    if isinstance(result, list):
        return [_annotate_cron_job(j, profile_name, home) for j in result]
    if isinstance(result, dict):
        return _annotate_cron_job(result, profile_name, home)
    return result


def _find_cron_job_profile(job_id: str) -> Optional[str]:
    for profile in _cron_profile_dicts():
        name = str(profile.get("name") or "")
        if not name:
            continue
        jobs = _call_cron_for_profile(name, "list_jobs", True)
        if any(j.get("id") == job_id or j.get("name") == job_id for j in jobs):
            return name
    return None


@app.get("/api/cron/jobs")
async def list_cron_jobs(profile: str = "all"):
    requested = (profile or "all").strip()
    if requested.lower() != "all":
        return _call_cron_for_profile(requested, "list_jobs", True)

    jobs: List[Dict[str, Any]] = []
    for item in _cron_profile_dicts():
        name = str(item.get("name") or "")
        if not name:
            continue
        try:
            jobs.extend(_call_cron_for_profile(name, "list_jobs", True))
        except Exception:
            _log.exception("Failed to list cron jobs for profile %s", name)
    return jobs


@app.get("/api/cron/jobs/{job_id}")
async def get_cron_job(job_id: str, profile: Optional[str] = None):
    selected = profile or _find_cron_job_profile(job_id)
    if not selected:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _call_cron_for_profile(selected, "get_job", job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/cron/jobs/{job_id}/runs")
async def list_cron_job_runs(job_id: str, profile: Optional[str] = None, limit: int = 20):
    """Run sessions produced by a cron job, newest first.

    Cron runs are stored as ordinary sessions whose id is
    ``cron_{job_id}_{timestamp}`` (see cron/scheduler.run_job). A job's history
    is therefore every session whose id carries that prefix; ``source='cron'``
    narrows it and the id prefix binds it to this job. Powers the run-history
    list under each job in the desktop cron detail. Same row shape as
    ``/api/sessions`` so the frontend can reuse SessionInfo.

    Backed by ``SessionDB.list_cron_job_runs`` — a bounded ``[prefix, hi)``
    id-range scan, not the compression-chain CTE used for the recents list,
    so the cost scales with the requested window and not the (unbounded) total
    cron history.
    """
    selected = profile or _find_cron_job_profile(job_id)
    # job_id may be a human name; resolve to the canonical id used in run-session ids.
    canonical = job_id
    if selected:
        job = _call_cron_for_profile(selected, "get_job", job_id)
        if job and job.get("id"):
            canonical = str(job["id"])

    try:
        limit_n = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        limit_n = 20

    db = _open_session_db_for_profile(selected)
    try:
        runs = db.list_cron_job_runs(canonical, limit=limit_n, offset=0)
        now = time.time()
        for s in runs:
            s["is_active"] = (
                s.get("ended_at") is None
                and (now - s.get("last_active", s.get("started_at", 0))) < 300
            )
            s["archived"] = bool(s.get("archived"))
            if selected:
                s["profile"] = selected
        return {"runs": runs, "limit": limit_n}
    finally:
        db.close()


@app.post("/api/cron/jobs")
async def create_cron_job(body: CronJobCreate, profile: str = "default"):
    try:
        profile_name, profile_home = _cron_profile_home(profile)
        script = _normalize_dashboard_cron_script(body.script, profile_home)
        skills = _cron_string_list(body.skills)
        context_from = _cron_string_list(body.context_from)
        _validate_dashboard_cron_context_from(context_from, profile_name)
        no_agent = bool(body.no_agent)
        _validate_dashboard_cron_effective_job({
            "prompt": body.prompt,
            "skills": skills,
            "script": script,
            "no_agent": no_agent,
        })
        return _call_cron_for_profile(
            profile_name,
            "create_job",
            prompt=body.prompt or "",
            schedule=body.schedule,
            name=body.name,
            deliver=_cron_optional_text(body.deliver) or "local",
            skills=skills,
            model=_cron_optional_text(body.model),
            provider=_cron_optional_text(body.provider),
            base_url=_cron_optional_text(body.base_url, strip_trailing_slash=True),
            script=script,
            context_from=context_from,
            enabled_toolsets=_cron_string_list(body.enabled_toolsets),
            workdir=_cron_optional_text(body.workdir),
            no_agent=no_agent,
        )
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("POST /api/cron/jobs failed")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/cron/delivery-targets")
async def get_cron_delivery_targets():
    """Delivery targets the cron dropdown should offer.

    Always includes the implicit ``local`` option. Beyond that, the list is
    derived dynamically from the configured gateway platforms via
    ``cron.scheduler.cron_delivery_targets()`` — no hardcoded platform list. A
    configured platform that hasn't set its cron home channel is still returned
    with ``home_target_set: false`` so the UI can surface it as "configure a
    home channel first" rather than hiding it.
    """
    targets = [
        {
            "id": "local",
            "name": "Local (save only)",
            "home_target_set": True,
            "home_env_var": None,
        }
    ]
    try:
        from cron.scheduler import cron_delivery_targets

        targets.extend(cron_delivery_targets())
    except Exception:
        _log.exception("GET /api/cron/delivery-targets failed")
    return {"targets": targets}


@app.put("/api/cron/jobs/{job_id}")
async def update_cron_job(job_id: str, body: CronJobUpdate, profile: Optional[str] = None):
    selected = profile or _find_cron_job_profile(job_id)
    if not selected:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        profile_name, profile_home = _cron_profile_home(selected)
        existing = _call_cron_for_profile(profile_name, "get_job", job_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Job not found")
        updates = _normalize_dashboard_cron_updates(
            body.updates,
            profile_home,
        )
        if "context_from" in updates:
            _validate_dashboard_cron_context_from(
                updates.get("context_from"),
                profile_name,
            )
        execution_fields = {"prompt", "skill", "skills", "script", "no_agent"}
        if execution_fields.intersection(updates):
            effective = {**existing, **updates}
            if "skills" in updates and "skill" not in updates:
                effective["skill"] = None
            _validate_dashboard_cron_effective_job(effective)
        job = _call_cron_for_profile(profile_name, "update_job", job_id, updates)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/cron/jobs/{job_id}/pause")
async def pause_cron_job(job_id: str, profile: Optional[str] = None):
    selected = profile or _find_cron_job_profile(job_id)
    if not selected:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _call_cron_for_profile(selected, "pause_job", job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/cron/jobs/{job_id}/resume")
async def resume_cron_job(job_id: str, profile: Optional[str] = None):
    selected = profile or _find_cron_job_profile(job_id)
    if not selected:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _call_cron_for_profile(selected, "resume_job", job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/cron/jobs/{job_id}/trigger")
async def trigger_cron_job(job_id: str, profile: Optional[str] = None):
    selected = profile or _find_cron_job_profile(job_id)
    if not selected:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _call_cron_for_profile(selected, "trigger_job", job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.delete("/api/cron/jobs/{job_id}")
async def delete_cron_job(job_id: str, profile: Optional[str] = None):
    selected = profile or _find_cron_job_profile(job_id)
    if not selected:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        removed = _call_cron_for_profile(selected, "remove_job", job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True}


def _fire_cron_job_for_profile(profile: str, job_id: str) -> bool:
    """Run ONE due cron job end-to-end for ``profile`` via the resolved
    scheduler provider's ``fire_due`` (store CAS claim + ``run_one_job``).

    Retargets the ``cron.jobs`` module globals to the profile's cron dir under
    the shared lock — same mechanism as ``_call_cron_for_profile`` — so the
    claim and the run operate on the right profile's ``jobs.json``. Runs with
    no live adapters; delivery falls back to the per-platform send path (the
    dashboard process has no gateway adapter handles, exactly like the desktop
    cron path above).
    """
    _profile_name, home = _cron_profile_home(profile)
    with _CRON_PROFILE_LOCK:
        from cron import jobs as cron_jobs
        from cron.scheduler_provider import resolve_cron_scheduler

        old_cron_dir = cron_jobs.CRON_DIR
        old_jobs_file = cron_jobs.JOBS_FILE
        old_output_dir = cron_jobs.OUTPUT_DIR
        cron_jobs.CRON_DIR = home / "cron"
        cron_jobs.JOBS_FILE = cron_jobs.CRON_DIR / "jobs.json"
        cron_jobs.OUTPUT_DIR = cron_jobs.CRON_DIR / "output"
        try:
            provider = resolve_cron_scheduler()
            return bool(provider.fire_due(job_id, adapters=None, loop=None))
        finally:
            cron_jobs.CRON_DIR = old_cron_dir
            cron_jobs.JOBS_FILE = old_jobs_file
            cron_jobs.OUTPUT_DIR = old_output_dir


@app.post("/api/cron/fire")
async def cron_fire_webhook(request: Request):
    """Chronos managed-cron fire webhook (NAS -> agent).

    Authenticated by a short-lived NAS-minted JWT (verified by the pluggable
    Chronos fire-verifier), NOT the dashboard session cookie — so this path is
    in ``PUBLIC_API_PATHS`` to bypass the dashboard auth gate, and the JWT is
    the real gate. This is the inbound half of scale-to-zero managed cron: NAS
    POSTs here at fire time, the agent verifies, claims the job (store CAS, so
    at-most-once across replicas / on a NAS retry), runs it, and re-arms the
    next one-shot.

    Lives on the dashboard app (not the api_server adapter) because the
    dashboard is the agent's always-reachable public HTTP surface on hosted
    deployments; the gateway may be idle/scaled down.

    Returns 202 immediately and runs the job in the background so a long agent
    turn never trips NAS's HTTP timeout.
    """
    from plugins.cron_providers.chronos.verify import get_fire_verifier

    auth = request.headers.get("Authorization", "")
    token = auth[7:].strip() if auth.startswith("Bearer ") else ""

    cfg = load_config()
    claims = get_fire_verifier()(
        token=token,
        expected_audience=cfg_get(cfg, "cron", "chronos", "expected_audience", default=""),
        jwks_or_key=cfg_get(cfg, "cron", "chronos", "nas_jwks_url", default="") or None,
        issuer=cfg_get(cfg, "cron", "chronos", "portal_url", default="") or None,
    )
    if claims is None:
        return JSONResponse({"error": "invalid fire token"}, status_code=401)

    try:
        body = await request.json()
    except Exception:
        body = {}
    job_id = (body or {}).get("job_id") if isinstance(body, dict) else None
    if not job_id:
        return JSONResponse({"error": "missing job_id"}, status_code=400)

    profile = _find_cron_job_profile(job_id)
    if not profile:
        # Job is gone (cancelled / completed) — nothing to fire. 200 so NAS
        # does not retry a fire that is intentionally absent.
        return JSONResponse({"status": "gone", "job_id": job_id}, status_code=200)

    # Run in the background; the store CAS claim inside fire_due de-dupes a
    # NAS/scheduler retry that arrives while this is in flight.
    asyncio.create_task(
        asyncio.to_thread(_fire_cron_job_for_profile, profile, job_id)
    )
    return JSONResponse({"status": "accepted", "job_id": job_id}, status_code=202)


# ---------------------------------------------------------------------------
# Automation Blueprints — parameterized automation blueprints. The dashboard renders the
# slot schema as a form; submitting instantiates a real cron job via the same
# create_job path. See cron/blueprint_catalog.py for the single source of truth.
# ---------------------------------------------------------------------------
class AutomationBlueprintInstantiate(BaseModel):
    blueprint: str                      # blueprint key, e.g. "morning-brief"
    values: Dict[str, Any] = {}      # filled slot values from the form


@app.get("/api/cron/blueprints")
async def list_cron_blueprints():
    """Return the blueprint catalog as form schemas for the dashboard gallery.

    The ``deliver`` slot's options are rewritten from the user's actually
    configured gateway platforms (plus the universal origin/local/all), so the
    form never offers a platform that isn't connected.
    """
    try:
        from cron.blueprint_catalog import CATALOG, blueprint_catalog_entry

        deliver_options = None
        try:
            from cron.scheduler import cron_delivery_targets

            platforms = [t["id"] for t in cron_delivery_targets() if t.get("id")]
            deliver_options = ["origin", "local", *platforms]
        except Exception:
            _log.debug("cron_delivery_targets unavailable; using static deliver options", exc_info=True)

        entries = []
        for r in CATALOG:
            entry = blueprint_catalog_entry(r)
            if deliver_options:
                for f in entry.get("fields", []):
                    if f.get("name") == "deliver":
                        f["options"] = deliver_options
            entries.append(entry)
        return {"blueprints": entries}
    except Exception as e:
        _log.exception("GET /api/cron/blueprints failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cron/blueprints/instantiate")
async def instantiate_blueprint(body: AutomationBlueprintInstantiate, profile: str = "default"):
    """Fill a blueprint's slots and create the cron job (form-submit path)."""
    try:
        from cron.blueprint_catalog import fill_blueprint, get_blueprint, BlueprintFillError

        blueprint = get_blueprint(body.blueprint)
        if blueprint is None:
            raise HTTPException(status_code=404, detail=f"Unknown blueprint: {body.blueprint}")
        try:
            spec = fill_blueprint(blueprint, body.values)
        except BlueprintFillError as exc:
            # Field-level validation error — 422 so the form can show it inline.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        # Blueprint-created jobs deliver to the dashboard's configured target by
        # default; the form's deliver slot overrides via spec["deliver"].
        spec.pop("origin", None)
        return _call_cron_for_profile(profile, "create_job", **spec)
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("POST /api/cron/blueprints/instantiate failed")
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# MCP server endpoints — list / add / remove / test.
#
# Wraps the same config data layer the CLI uses (hermes_cli.mcp_config), so
# servers managed here show up under `hermes mcp list` and vice versa.  Secrets
# in stdio `env` blocks are redacted on read; the agent picks them up from
# config.yaml at session start exactly as with CLI-added servers.
# ---------------------------------------------------------------------------


class MCPServerCreate(BaseModel):
    name: str
    url: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = []
    # env: KEY=VALUE map for stdio servers (API keys, etc.)
    env: Dict[str, str] = {}
    # auth: "oauth" | "header" | None
    auth: Optional[str] = None
    profile: Optional[str] = None


def _redact_mcp_env(env: Dict[str, Any]) -> Dict[str, str]:
    """Mask secret-shaped MCP env values for read responses."""
    out: Dict[str, str] = {}
    for k, v in (env or {}).items():
        try:
            out[str(k)] = redact_key(str(v)) if v else ""
        except Exception:
            out[str(k)] = "***"
    return out


def _mcp_server_summary(name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    transport = "http" if cfg.get("url") else ("stdio" if cfg.get("command") else "unknown")
    return {
        "name": name,
        "transport": transport,
        "url": cfg.get("url"),
        "command": cfg.get("command"),
        "args": list(cfg.get("args") or []),
        "env": _redact_mcp_env(cfg.get("env") or {}),
        "auth": cfg.get("auth"),
        "enabled": cfg.get("enabled", True) is not False,
        # Tool selection: list of enabled tool names, or None = all.
        "tools": cfg.get("tools"),
    }


@app.get("/api/mcp/servers")
async def list_mcp_servers(profile: Optional[str] = None):
    from hermes_cli.mcp_config import _get_mcp_servers

    with _profile_scope(profile):
        servers = _get_mcp_servers()
    return {
        "servers": [
            _mcp_server_summary(name, cfg) for name, cfg in sorted(servers.items())
        ]
    }


@app.post("/api/mcp/servers")
async def add_mcp_server(body: MCPServerCreate, profile: Optional[str] = None):
    from hermes_cli.mcp_config import _get_mcp_servers, _save_mcp_server

    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Server name is required")
    with _profile_scope(body.profile or profile):
        existing = _get_mcp_servers()
    if name in existing:
        raise HTTPException(status_code=409, detail=f"Server '{name}' already exists")
    if not body.url and not body.command:
        raise HTTPException(
            status_code=400,
            detail="Provide either a URL (HTTP/SSE server) or a command (stdio server)",
        )

    server_config: Dict[str, Any] = {}
    if body.url:
        server_config["url"] = body.url.strip()
    if body.command:
        server_config["command"] = body.command.strip()
        if body.args:
            server_config["args"] = list(body.args)
    if body.env:
        server_config["env"] = dict(body.env)
    if body.auth:
        server_config["auth"] = body.auth

    try:
        with _profile_scope(body.profile or profile):
            if not _save_mcp_server(name, server_config):
                raise HTTPException(
                    status_code=400,
                    detail=f"Server '{name}' rejected: suspicious command/args configuration",
                )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("POST /api/mcp/servers failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _mcp_server_summary(name, server_config)


@app.delete("/api/mcp/servers/{name}")
async def remove_mcp_server(name: str, profile: Optional[str] = None):
    from hermes_cli.mcp_config import _remove_mcp_server

    with _profile_scope(profile):
        removed = _remove_mcp_server(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")
    return {"ok": True}


@app.post("/api/mcp/servers/{name}/test")
async def test_mcp_server(name: str, profile: Optional[str] = None):
    """Connect to the server, list its tools, disconnect.  Returns tool list."""
    from hermes_cli.mcp_config import _get_mcp_servers, _probe_single_server

    with _profile_scope(profile):
        servers = _get_mcp_servers()
    if name not in servers:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")

    def _probe_scoped():
        # Re-enter the scope INSIDE the worker thread so call-time
        # resolution during the probe — env-placeholder expansion in
        # _resolve_mcp_server_config reading the profile's .env — sees the
        # selected profile, matching the config the server was saved into.
        # (asyncio.to_thread copies contextvars, but entering explicitly
        # keeps the lock-protected SKILLS_DIR swap balanced per-thread.)
        # The probe's dedicated MCP event-loop thread is covered too:
        # _run_on_mcp_loop wraps scheduled coroutines with the caller's
        # HERMES_HOME override (see mcp_tool._wrap_with_home_override), so
        # OAuth token stores resolve against the selected profile as well.
        with _profile_scope(profile):
            return _probe_single_server(name, servers[name])

    try:
        # Probe blocks on a dedicated MCP event loop — run in a thread so the
        # FastAPI event loop is never blocked.
        tools = await asyncio.to_thread(_probe_scoped)
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "tools": [],
        }
    return {
        "ok": True,
        "tools": [{"name": t, "description": d} for t, d in tools],
    }


class MCPEnabledToggle(BaseModel):
    enabled: bool
    profile: Optional[str] = None


@app.put("/api/mcp/servers/{name}/enabled")
async def set_mcp_server_enabled(
    name: str, body: MCPEnabledToggle, profile: Optional[str] = None
):
    """Enable or disable an MCP server (takes effect on next session/gateway).

    Toggles the ``enabled`` key on the server's config.yaml entry — the same
    flag the agent reads at startup.  Disabled servers stay in config so they
    can be re-enabled without re-entering their settings.
    """
    with _profile_scope(body.profile or profile):
        cfg = load_config()
        servers = cfg.get("mcp_servers")
        if not isinstance(servers, dict) or name not in servers:
            raise HTTPException(status_code=404, detail=f"Server '{name}' not found")
        if not isinstance(servers[name], dict):
            raise HTTPException(status_code=400, detail="Malformed server config")
        servers[name]["enabled"] = bool(body.enabled)
        save_config(cfg)
    return {"ok": True, "name": name, "enabled": bool(body.enabled)}


@app.get("/api/mcp/catalog")
async def list_mcp_catalog(profile: Optional[str] = None):
    """Browse the Nous-approved MCP catalog (the optional-mcps/ manifests).

    Each entry reports whether it's already installed and enabled so the UI
    can show install / enabled state inline.  This is the same catalog
    `hermes mcp catalog` / `hermes mcp install` read.  ``profile`` scopes
    the installed/enabled annotations (the catalog itself is repo-shipped
    and identical for every profile).
    """
    try:
        from hermes_cli import mcp_catalog
    except Exception as exc:
        _log.exception("mcp_catalog import failed")
        raise HTTPException(status_code=500, detail=f"Catalog unavailable: {exc}")

    entries = []
    try:
        with _profile_scope(profile):
            catalog_entries = list(mcp_catalog.list_catalog())
            installed_state = {
                e.name: (mcp_catalog.is_installed(e.name), mcp_catalog.is_enabled(e.name))
                for e in catalog_entries
            }
        for entry in catalog_entries:
            auth = entry.auth
            transport = entry.transport
            install = entry.install
            entries.append({
                "name": entry.name,
                "description": entry.description,
                "source": entry.source,
                "transport": transport.type,
                "auth_type": getattr(auth, "type", "none"),
                # Env vars the user must supply (names + prompts only, never values).
                "required_env": [
                    {"name": e.name, "prompt": e.prompt, "required": e.required}
                    for e in getattr(auth, "env", []) or []
                ],
                # Transport details so the UI can show exactly what connects/runs.
                # The trust model (docs: user-guide/features/mcp) tells users to
                # inspect command/args/url and the install bootstrap before
                # installing — surface them rather than hiding them in the repo.
                "command": transport.command,
                "args": list(transport.args or []),
                "url": transport.url,
                # Git bootstrap (present only for entries that clone + build).
                "install_url": install.url if install else None,
                "install_ref": install.ref if install else None,
                "bootstrap": list(install.bootstrap) if install else [],
                # Default tool pre-selection hint and post-install guidance.
                "default_enabled": list(entry.tools.default_enabled)
                if entry.tools.default_enabled is not None
                else None,
                "post_install": entry.post_install or "",
                "needs_install": entry.install is not None,
                "installed": installed_state.get(entry.name, (False, False))[0],
                "enabled": installed_state.get(entry.name, (False, False))[1],
            })
    except HTTPException:
        # Unknown/invalid profile → 404, not a silently-empty catalog.
        raise
    except Exception:
        _log.exception("list_mcp_catalog failed")

    diagnostics = []
    try:
        diagnostics = [
            {"name": n, "kind": k, "message": m}
            for (n, k, m) in mcp_catalog.catalog_diagnostics()
        ]
    except Exception:
        pass

    return {"entries": entries, "diagnostics": diagnostics}


class MCPCatalogInstall(BaseModel):
    name: str
    # env: KEY=VALUE map for catalog entries that declare required env vars.
    env: Dict[str, str] = {}
    enable: bool = True
    profile: Optional[str] = None


@app.post("/api/mcp/catalog/install")
async def install_mcp_catalog_entry(body: MCPCatalogInstall, profile: Optional[str] = None):
    """Install a catalog MCP into config.yaml.

    For HTTP/stdio entries with required env vars, those are written to .env
    via the standard env path so the agent can read them at session start.
    Entries that need a git bootstrap (``needs_install``) are installed via
    the CLI action path because the clone can take time.
    """
    from hermes_cli import mcp_catalog

    name = (body.name or "").strip()
    entry = mcp_catalog.get_entry(name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No catalog entry '{name}'")

    # Persist any supplied env vars first (catalog entries declare which names
    # they need; we only write the ones the user provided).
    effective_profile = body.profile or profile
    if body.env:
        with _profile_scope(effective_profile):
            for k, v in body.env.items():
                if v:
                    save_env_value(k, v)

    # Git-bootstrap entries can take a while to clone — run via the background
    # action path so the request returns immediately and the UI can tail logs.
    # The -p subprocess rebinds HERMES_HOME-derived paths in the child.
    if entry.install is not None:
        try:
            proc = _spawn_hermes_action(
                _profile_cli_args(effective_profile) + ["mcp", "install", name],
                "mcp-install",
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Install failed: {exc}")
        return {"ok": True, "name": name, "background": True, "action": "mcp-install"}

    # No git step — install synchronously via the catalog API. install_entry
    # routes through load_config/save_config + save_env_value, all call-time
    # resolvers, so the context override scopes it. Wrap the to_thread body
    # in the scope INSIDE the thread (contextvars don't propagate into
    # to_thread the other way around — asyncio.to_thread copies context, so
    # setting it here works; keep it explicit for clarity).
    def _install_scoped():
        with _profile_scope(effective_profile):
            mcp_catalog.install_entry(entry, enable=body.enable)

    try:
        await asyncio.to_thread(_install_scoped)
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("install_mcp_catalog_entry failed")
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "name": name, "background": False}


# Register the mcp-install action log so /api/actions/mcp-install/status works.
_ACTION_LOG_FILES.setdefault("mcp-install", "action-mcp-install.log")
_ACTION_LOG_FILES.setdefault("computer-use-grant", "action-computer-use-grant.log")


# ---------------------------------------------------------------------------
# Pairing endpoints — approve / revoke / list messaging pairing codes.
#
# These are how a remote admin onboards messaging users (Telegram, Discord, …)
# without shell access.  Wraps gateway.pairing.PairingStore directly.
# ---------------------------------------------------------------------------


class PairingApprove(BaseModel):
    platform: str
    code: str


class PairingRevoke(BaseModel):
    platform: str
    user_id: str


def _pairing_store():
    from gateway.pairing import PairingStore

    return PairingStore()


@app.get("/api/pairing")
async def list_pairing():
    store = _pairing_store()
    return {
        "pending": store.list_pending(),
        "approved": store.list_approved(),
    }


@app.post("/api/pairing/approve")
async def approve_pairing(body: PairingApprove):
    store = _pairing_store()
    platform = (body.platform or "").lower().strip()
    code = (body.code or "").upper().strip()
    if not platform or not code:
        raise HTTPException(status_code=400, detail="platform and code are required")

    result = store.approve_code(platform, code)
    if result:
        return {"ok": True, "user": result}
    if store._is_locked_out(platform):
        raise HTTPException(
            status_code=429,
            detail=f"Platform '{platform}' is locked out after too many failed approvals.",
        )
    raise HTTPException(
        status_code=404,
        detail=f"Code '{code}' not found or expired for platform '{platform}'.",
    )


@app.post("/api/pairing/revoke")
async def revoke_pairing(body: PairingRevoke):
    store = _pairing_store()
    platform = (body.platform or "").lower().strip()
    if not platform or not body.user_id:
        raise HTTPException(status_code=400, detail="platform and user_id are required")
    if store.revoke(platform, body.user_id):
        return {"ok": True}
    raise HTTPException(
        status_code=404,
        detail=f"User {body.user_id} not found in approved list for {platform}.",
    )


@app.post("/api/pairing/clear-pending")
async def clear_pending_pairing():
    store = _pairing_store()
    count = store.clear_pending()
    return {"ok": True, "cleared": count}


# ---------------------------------------------------------------------------
# Webhook subscription endpoints — list / subscribe / remove.
#
# Wraps the same JSON store the CLI uses (hermes_cli.webhook); the webhook
# adapter hot-reloads it without a gateway restart.  Per-route HMAC secrets
# are redacted on read and surfaced once on create.
# ---------------------------------------------------------------------------


class WebhookCreate(BaseModel):
    name: str
    description: Optional[str] = None
    events: List[str] = []
    prompt: Optional[str] = None
    skills: List[str] = []
    deliver: str = "log"
    deliver_only: bool = False
    deliver_chat_id: Optional[str] = None
    # secret: omit to auto-generate
    secret: Optional[str] = None


def _webhook_route_summary(name: str, route: Dict[str, Any], base_url: str) -> Dict[str, Any]:
    return {
        "name": name,
        "description": route.get("description", ""),
        "events": list(route.get("events") or []),
        "deliver": route.get("deliver", "log"),
        "deliver_only": bool(route.get("deliver_only")),
        "prompt": route.get("prompt", ""),
        "skills": list(route.get("skills") or []),
        "created_at": route.get("created_at"),
        "url": f"{base_url}/webhooks/{name}",
        # Secret is masked on read; full value only returned on create.
        "secret_set": bool(route.get("secret")),
        # Default-enabled; only an explicit enabled:false turns a route off.
        "enabled": route.get("enabled", True) is not False,
    }


@app.get("/api/webhooks")
async def list_webhooks():
    import hermes_cli.webhook as wh

    base_url = wh._get_webhook_base_url()
    subs = wh._load_subscriptions()
    return {
        "enabled": wh._is_webhook_enabled(),
        "base_url": base_url,
        "subscriptions": [
            _webhook_route_summary(name, route, base_url)
            for name, route in subs.items()
        ],
    }


@app.post("/api/webhooks/enable")
async def enable_webhooks():
    try:
        _write_platform_enabled("webhook", True)
    except Exception as exc:
        _log.exception("Failed to enable webhook platform from dashboard")
        raise HTTPException(
            status_code=500,
            detail="Failed to enable webhook platform.",
        ) from exc

    restart_result = _restart_gateway_after_webhook_enable()
    return {
        "ok": True,
        "platform": "webhook",
        "enabled": True,
        "needs_restart": not restart_result["restart_started"],
        **restart_result,
    }


@app.post("/api/webhooks")
async def create_webhook(body: WebhookCreate):
    import re as _re
    import secrets as _secrets
    import time as _time
    import hermes_cli.webhook as wh

    if not wh._is_webhook_enabled():
        raise HTTPException(
            status_code=400,
            detail="Webhook platform is not enabled. Enable it from the Webhooks page first.",
        )

    name = (body.name or "").strip().lower().replace(" ", "-")
    if not _re.match(r"^[a-z0-9][a-z0-9_-]*$", name):
        raise HTTPException(
            status_code=400,
            detail="Invalid name. Use lowercase alphanumeric with hyphens/underscores.",
        )

    if body.deliver_only and body.deliver == "log":
        raise HTTPException(
            status_code=400,
            detail="Direct delivery requires a real target (telegram, discord, …), not 'log'.",
        )

    secret = body.secret or _secrets.token_urlsafe(32)
    route: Dict[str, Any] = {
        "description": body.description or f"Dashboard-created subscription: {name}",
        "events": [e.strip() for e in body.events if e.strip()],
        "secret": secret,
        "prompt": body.prompt or "",
        "skills": [s.strip() for s in body.skills if s.strip()],
        "deliver": body.deliver or "log",
        "created_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
    }
    if body.deliver_only:
        route["deliver_only"] = True
    if body.deliver_chat_id:
        route["deliver_extra"] = {"chat_id": body.deliver_chat_id}

    subs = wh._load_subscriptions()
    subs[name] = route
    wh._save_subscriptions(subs)

    base_url = wh._get_webhook_base_url()
    summary = _webhook_route_summary(name, route, base_url)
    # Surface the secret exactly once, on create.
    summary["secret"] = secret
    return summary


@app.delete("/api/webhooks/{name}")
async def delete_webhook(name: str):
    import hermes_cli.webhook as wh

    key = (name or "").strip().lower()
    subs = wh._load_subscriptions()
    if key not in subs:
        raise HTTPException(status_code=404, detail=f"No subscription named '{key}'")
    del subs[key]
    wh._save_subscriptions(subs)
    return {"ok": True}


class WebhookEnabledToggle(BaseModel):
    enabled: bool


@app.put("/api/webhooks/{name}/enabled")
async def set_webhook_enabled(name: str, body: WebhookEnabledToggle):
    """Enable or disable a webhook route.

    Disabled routes stay in the subscriptions file (so they can be
    re-enabled) but the gateway rejects incoming events with 403.  The
    gateway hot-reloads the subscriptions file, so this takes effect on the
    next event without a restart.
    """
    import hermes_cli.webhook as wh

    key = (name or "").strip().lower()
    subs = wh._load_subscriptions()
    if key not in subs:
        raise HTTPException(status_code=404, detail=f"No subscription named '{key}'")
    subs[key]["enabled"] = bool(body.enabled)
    wh._save_subscriptions(subs)
    return {"ok": True, "name": key, "enabled": bool(body.enabled)}


# ---------------------------------------------------------------------------
# Gateway lifecycle endpoints — start / stop.
#
# restart + update already exist above; these complete the lifecycle so a
# remote admin can bring the gateway up or down without shell access.  Both
# spawn the real `hermes gateway <verb>` so behaviour matches the CLI exactly.
# Status is already surfaced by /api/status (gateway_running/state/platforms).
# ---------------------------------------------------------------------------


@app.post("/api/gateway/start")
async def start_gateway(profile: Optional[str] = None):
    try:
        proc = _spawn_hermes_action(_gateway_subcommand(profile, "start"), "gateway-start")
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn gateway start")
        raise HTTPException(status_code=500, detail=f"Failed to start gateway: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "gateway-start"}


@app.post("/api/gateway/stop")
async def stop_gateway(profile: Optional[str] = None):
    try:
        proc = _spawn_hermes_action(_gateway_subcommand(profile, "stop"), "gateway-stop")
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn gateway stop")
        raise HTTPException(status_code=500, detail=f"Failed to stop gateway: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "gateway-stop"}


# ---------------------------------------------------------------------------
# Credential pool endpoints — list / add / remove rotation keys.
#
# The credential pool (auth.json -> credential_pool.<provider>[]) holds the
# rotating API keys the agent round-robins through.  Secrets are redacted on
# read; only the agent ever sees the raw values at session start.
# ---------------------------------------------------------------------------


class CredentialPoolAdd(BaseModel):
    provider: str
    # api_key for API-key providers; OAuth pooling stays CLI-only (it needs
    # an interactive browser flow that doesn't belong in a single POST).
    api_key: str
    label: Optional[str] = None


def _pool_entry_summary(entry: Any, index: int) -> Dict[str, Any]:
    """Redacted, display-safe view of one PooledCredential.

    ``index`` is 1-based to match CredentialPool.remove_index().
    """
    token = getattr(entry, "access_token", "") or ""
    return {
        "index": index,
        "id": getattr(entry, "id", None),
        "label": getattr(entry, "label", None),
        "auth_type": getattr(entry, "auth_type", None),
        "source": getattr(entry, "source", None),
        "priority": getattr(entry, "priority", 0),
        "last_status": getattr(entry, "last_status", None),
        "request_count": getattr(entry, "request_count", 0),
        "token_preview": redact_key(token) if token else "",
        "has_refresh": bool(getattr(entry, "refresh_token", None)),
    }


@app.get("/api/credentials/pool")
async def list_credential_pool():
    from agent.credential_pool import load_pool
    from hermes_cli.auth import read_credential_pool

    providers = []
    # read_credential_pool(None) lists every provider that has pooled entries;
    # load_pool() then gives us the rich PooledCredential objects per provider.
    raw_pool = read_credential_pool()
    for provider_id in sorted(raw_pool.keys()):
        try:
            pool = load_pool(provider_id)
        except Exception:
            _log.exception("load_pool(%s) failed", provider_id)
            continue
        entries = pool.entries()
        if not entries:
            continue
        providers.append({
            "provider": provider_id,
            "entries": [
                _pool_entry_summary(e, i) for i, e in enumerate(entries, start=1)
            ],
        })
    return {"providers": providers}


@app.post("/api/credentials/pool")
async def add_credential_pool_entry(body: CredentialPoolAdd):
    import uuid as _uuid
    from agent.credential_pool import (
        load_pool,
        PooledCredential,
        AUTH_TYPE_API_KEY,
        SOURCE_MANUAL,
    )

    provider = (body.provider or "").strip().lower()
    api_key = (body.api_key or "").strip()
    if not provider or not api_key:
        raise HTTPException(status_code=400, detail="provider and api_key are required")

    try:
        pool = load_pool(provider)
        label = (body.label or "").strip() or f"key #{len(pool.entries()) + 1}"
        entry = PooledCredential(
            provider=provider,
            id=_uuid.uuid4().hex[:6],
            label=label,
            auth_type=AUTH_TYPE_API_KEY,
            priority=0,
            source=SOURCE_MANUAL,
            access_token=api_key,
        )
        pool.add_entry(entry)
    except Exception as exc:
        _log.exception("POST /api/credentials/pool failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "provider": provider, "count": len(pool.entries())}


@app.delete("/api/credentials/pool/{provider}/{index}")
async def remove_credential_pool_entry(provider: str, index: int):
    """Remove a pool entry.  ``index`` is 1-based (matches the list response)."""
    from agent.credential_pool import load_pool

    provider = (provider or "").strip().lower()
    try:
        pool = load_pool(provider)
        removed = pool.remove_index(index)
    except Exception as exc:
        _log.exception("DELETE /api/credentials/pool failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if removed is None:
        raise HTTPException(status_code=404, detail="No pool entry at that index")
    return {"ok": True, "provider": provider, "count": len(pool.entries())}


# ---------------------------------------------------------------------------
# Memory provider endpoints — status / list providers / select / disable / reset.
#
# Selecting a provider only writes config.memory.provider (full interactive
# provider setup, with its API-key prompts, stays on the CLI via
# `hermes memory setup`).  The dashboard covers the common admin actions:
# see which provider is active, switch the built-in store on/off, and wipe
# built-in memory files.
# ---------------------------------------------------------------------------


class MemoryProviderSelect(BaseModel):
    # "" or "built-in" disables the external provider (built-in only).
    provider: str


class MemoryReset(BaseModel):
    # "all" | "memory" | "user"
    target: str = "all"


@app.get("/api/memory")
async def get_memory_status():
    from plugins.memory import discover_memory_providers

    cfg = load_config()
    active = ""
    mem = cfg.get("memory")
    if isinstance(mem, dict):
        active = str(mem.get("provider") or "")

    providers = []
    try:
        for name, description, configured in discover_memory_providers():
            providers.append({
                "name": name,
                "description": description,
                "configured": bool(configured),
            })
    except Exception:
        _log.exception("discover_memory_providers failed")

    # Built-in memory file sizes (so the UI can show what a reset would erase).
    mem_dir = get_hermes_home() / "memories"
    files = {}
    for fname, key in (("MEMORY.md", "memory"), ("USER.md", "user")):
        path = mem_dir / fname
        files[key] = path.stat().st_size if path.exists() else 0

    return {
        "active": active,
        "providers": providers,
        "builtin_files": files,
    }


@app.put("/api/memory/provider")
async def set_memory_provider(body: MemoryProviderSelect):
    provider = (body.provider or "").strip()
    if provider.lower() in {"built-in", "builtin", "none"}:
        provider = ""

    if provider:
        from plugins.memory import discover_memory_providers

        valid = {name for name, _d, _c in discover_memory_providers()}
        if provider not in valid:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown memory provider '{provider}'. Run `hermes memory setup` to configure a new one.",
            )

    cfg = load_config()
    if not isinstance(cfg.get("memory"), dict):
        cfg["memory"] = {}
    cfg["memory"]["provider"] = provider
    save_config(cfg)
    return {"ok": True, "active": provider}


@app.post("/api/memory/reset")
async def reset_memory(body: MemoryReset):
    target = (body.target or "all").strip().lower()
    if target not in {"all", "memory", "user"}:
        raise HTTPException(status_code=400, detail="target must be all, memory, or user")

    mem_dir = get_hermes_home() / "memories"
    deleted = []
    targets = []
    if target in {"all", "memory"}:
        targets.append("MEMORY.md")
    if target in {"all", "user"}:
        targets.append("USER.md")
    for fname in targets:
        path = mem_dir / fname
        if path.exists():
            try:
                path.unlink()
                deleted.append(fname)
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"Could not delete {fname}: {exc}")
    return {"ok": True, "deleted": deleted}


# ---------------------------------------------------------------------------
# Operations endpoints — doctor / security audit / backup / import /
# checkpoints / hooks.
#
# Diagnostic and maintenance commands.  The long-running / text-output ones
# (doctor, security audit, backup, import, skills install) are spawned as
# background actions whose logs the dashboard tails via
# /api/actions/{name}/status — same pattern as gateway restart and update.
# The cheap, structured reads (hooks list, checkpoints list) return JSON
# directly.
# ---------------------------------------------------------------------------


@app.post("/api/ops/doctor")
async def run_doctor():
    try:
        proc = _spawn_hermes_action(["doctor"], "doctor")
    except Exception as exc:
        _log.exception("Failed to spawn doctor")
        raise HTTPException(status_code=500, detail=f"Failed to run doctor: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "doctor"}


@app.post("/api/ops/security-audit")
async def run_security_audit():
    try:
        proc = _spawn_hermes_action(["security", "audit"], "security-audit")
    except Exception as exc:
        _log.exception("Failed to spawn security audit")
        raise HTTPException(status_code=500, detail=f"Failed to run security audit: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "security-audit"}


class BackupRequest(BaseModel):
    # Optional output path; defaults to a timestamped zip in the home dir.
    output: Optional[str] = None


def _dashboard_backup_dir() -> Path:
    return get_hermes_home() / "backups"


def _new_dashboard_backup_path() -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    return _dashboard_backup_dir() / f"hermes-backup-{stamp}-{secrets.token_hex(4)}.zip"


@app.post("/api/ops/backup")
async def run_backup(body: BackupRequest):
    args = ["backup"]
    archive: Optional[Path] = None
    if body.output:
        args.append(body.output.strip())
    else:
        archive = _new_dashboard_backup_path()
        try:
            archive.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not create backup directory: {exc}",
            )
        args.append(str(archive))
    try:
        proc = _spawn_hermes_action(args, "backup")
    except Exception as exc:
        _log.exception("Failed to spawn backup")
        raise HTTPException(status_code=500, detail=f"Failed to run backup: {exc}")
    response = {"ok": True, "pid": proc.pid, "name": "backup"}
    if archive is not None:
        response["archive"] = str(archive)
    return response


@app.get("/api/ops/backup/download")
async def download_dashboard_backup(archive: str):
    try:
        backup_dir = _dashboard_backup_dir().expanduser().resolve(strict=False)
        target = Path(archive).expanduser().resolve(strict=True)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Backup not found")
    except (OSError, RuntimeError):
        raise HTTPException(status_code=400, detail="Invalid backup path")

    if not _path_is_under(backup_dir, target):
        raise HTTPException(status_code=403, detail="Backup is outside the dashboard backup directory")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Backup not found")

    return FileResponse(
        path=str(target),
        media_type="application/zip",
        filename=target.name,
        content_disposition_type="attachment",
    )


class ImportRequest(BaseModel):
    archive: str
    # Pass --force to `hermes import`. The spawned action runs with
    # stdin=DEVNULL, so the CLI's interactive "Continue? [y/N]" overwrite
    # prompt hits EOF and auto-aborts ("Aborted.", exit 1) whenever the
    # target already has a config — which it always does when the dashboard
    # itself is running from it. The dashboard shows its own confirm modal
    # before calling this endpoint, then sends force=True so the restore
    # proceeds non-interactively.
    force: bool = False


@app.post("/api/ops/import")
async def run_import(body: ImportRequest):
    archive = (body.archive or "").strip()
    if not archive:
        raise HTTPException(status_code=400, detail="archive path is required")
    if not os.path.isfile(archive):
        raise HTTPException(status_code=404, detail=f"Archive not found: {archive}")
    args = ["import", archive]
    if body.force:
        args.append("--force")
    try:
        proc = _spawn_hermes_action(args, "import")
    except Exception as exc:
        _log.exception("Failed to spawn import")
        raise HTTPException(status_code=500, detail=f"Failed to run import: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "import"}


def _safe_backup_upload_name(filename: str | None) -> str:
    name = Path(filename or "backup.zip").name.strip()
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    if not name:
        name = "backup.zip"
    if not name.lower().endswith(".zip"):
        name = f"{name}.zip"
    return name


@app.post("/api/ops/import-upload")
async def run_import_upload(
    file: UploadFile = File(...),
    force: bool = Form(False),
):
    staging_dir = _dashboard_backup_dir()
    try:
        staging_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not create import staging directory: {exc}",
        )

    safe_name = _safe_backup_upload_name(file.filename)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = staging_dir / f"dashboard-import-{stamp}-{secrets.token_hex(4)}-{safe_name}"
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".upload",
        dir=str(staging_dir),
    )
    tmp_path = Path(tmp_name)
    total = 0
    renamed = False
    try:
        with os.fdopen(tmp_fd, "wb") as out:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MANAGED_FILE_MAX_BYTES:
                    raise HTTPException(status_code=413, detail="Archive is too large")
                out.write(chunk)
        os.replace(tmp_path, target)
        renamed = True
    except HTTPException:
        raise
    except PermissionError:
        raise HTTPException(
            status_code=403,
            detail="Import staging directory is not writable",
        )
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not write uploaded archive: {exc}",
        )
    finally:
        if not renamed:
            tmp_path.unlink(missing_ok=True)
        await file.close()

    if not zipfile.is_zipfile(target):
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Uploaded archive is not a valid zip file",
        )

    args = ["import", str(target)]
    if force:
        args.append("--force")
    try:
        proc = _spawn_hermes_action(args, "import")
    except Exception as exc:
        _log.exception("Failed to spawn import")
        raise HTTPException(status_code=500, detail=f"Failed to run import: {exc}")
    return {
        "ok": True,
        "pid": proc.pid,
        "name": "import",
        "archive": str(target),
        "uploaded_bytes": total,
    }


@app.get("/api/ops/hooks")
async def list_hooks():
    """List configured shell hooks from config.yaml with consent + health.

    Reports each hook's allowlist (consent) status and whether the script is
    currently executable, plus the set of valid hook events so the create
    form can offer them.
    """
    from hermes_cli.config import load_config as _load_config
    from agent import shell_hooks

    try:
        from hermes_cli.plugins import VALID_HOOKS
        valid_events = sorted(VALID_HOOKS)
    except Exception:
        valid_events = []

    specs = []
    try:
        specs = shell_hooks.iter_configured_hooks(_load_config())
    except Exception:
        _log.exception("iter_configured_hooks failed")

    out = []
    for spec in specs:
        entry = None
        try:
            entry = shell_hooks.allowlist_entry_for(spec.event, spec.command)
        except Exception:
            pass
        executable = False
        try:
            executable = shell_hooks.script_is_executable(spec.command)
        except Exception:
            pass
        out.append({
            "event": spec.event,
            "matcher": spec.matcher,
            "command": spec.command,
            "timeout": spec.timeout,
            "allowed": entry is not None,
            "approved_at": (entry or {}).get("approved_at"),
            "executable": executable,
        })

    return {"hooks": out, "valid_events": valid_events}


class HookCreate(BaseModel):
    event: str
    command: str
    matcher: Optional[str] = None
    timeout: Optional[int] = None
    # approve: write the consent allowlist entry too (the operator using the
    # authenticated dashboard is giving consent). Without it the hook is
    # configured but won't fire until approved.
    approve: bool = True


@app.post("/api/ops/hooks")
async def create_hook(body: HookCreate):
    """Add a shell hook to config.yaml (and optionally approve it).

    Shell hooks run arbitrary commands, so this is a privileged action: it
    writes to the ``hooks:`` config block and, when ``approve`` is set, records
    consent in the allowlist so the hook actually fires.  Takes effect on the
    next session / gateway restart.
    """
    from agent import shell_hooks

    event = (body.event or "").strip()
    command = (body.command or "").strip()
    if not event or not command:
        raise HTTPException(status_code=400, detail="event and command are required")

    try:
        from hermes_cli.plugins import VALID_HOOKS
        if event not in VALID_HOOKS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown event '{event}'. Valid: {', '.join(sorted(VALID_HOOKS))}",
            )
    except HTTPException:
        raise
    except Exception:
        pass

    cfg = load_config()
    hooks_cfg = cfg.get("hooks")
    if not isinstance(hooks_cfg, dict):
        hooks_cfg = {}
        cfg["hooks"] = hooks_cfg
    entries = hooks_cfg.get(event)
    if not isinstance(entries, list):
        entries = []
        hooks_cfg[event] = entries

    new_entry: Dict[str, Any] = {"command": command}
    if body.matcher:
        new_entry["matcher"] = body.matcher
    if body.timeout is not None:
        new_entry["timeout"] = int(body.timeout)
    entries.append(new_entry)
    save_config(cfg)

    approved = False
    if body.approve:
        try:
            shell_hooks._record_approval(event, command)
            approved = True
        except Exception:
            _log.exception("hook consent record failed")

    return {"ok": True, "event": event, "command": command, "approved": approved}


class HookDelete(BaseModel):
    event: str
    command: str


@app.delete("/api/ops/hooks")
async def delete_hook(body: HookDelete):
    """Remove a hook from config.yaml and revoke its consent allowlist entry."""
    from agent import shell_hooks

    event = (body.event or "").strip()
    command = (body.command or "").strip()
    if not event or not command:
        raise HTTPException(status_code=400, detail="event and command are required")

    cfg = load_config()
    hooks_cfg = cfg.get("hooks")
    removed = False
    if isinstance(hooks_cfg, dict) and isinstance(hooks_cfg.get(event), list):
        before = len(hooks_cfg[event])
        hooks_cfg[event] = [
            e for e in hooks_cfg[event]
            if not (isinstance(e, dict) and e.get("command") == command)
        ]
        removed = len(hooks_cfg[event]) < before
        if not hooks_cfg[event]:
            del hooks_cfg[event]
        if not hooks_cfg:
            cfg.pop("hooks", None)
        save_config(cfg)

    # Revoke consent regardless so a re-add re-prompts.
    try:
        shell_hooks.revoke(command)
    except Exception:
        pass

    if not removed:
        raise HTTPException(status_code=404, detail="No matching hook found")
    return {"ok": True}


@app.get("/api/ops/checkpoints")
async def list_checkpoints():
    """List the /rollback shadow store checkpoints (read-only)."""
    # Checkpoints live under <hermes_home>/checkpoints/.  Surface a count +
    # total size so the dashboard can show what a prune would reclaim; the
    # actual prune is a spawned action so confirmation/pruning logic stays
    # in one place (the CLI).
    cp_dir = get_hermes_home() / "checkpoints"
    sessions = []
    total_bytes = 0
    if cp_dir.is_dir():
        for child in sorted(cp_dir.iterdir()):
            if not child.is_dir():
                continue
            size = 0
            count = 0
            for f in child.rglob("*"):
                if f.is_file():
                    try:
                        size += f.stat().st_size
                        count += 1
                    except OSError:
                        pass
            total_bytes += size
            sessions.append({
                "session": child.name,
                "files": count,
                "bytes": size,
            })
    return {"sessions": sessions, "total_bytes": total_bytes}


@app.post("/api/ops/checkpoints/prune")
async def prune_checkpoints():
    try:
        proc = _spawn_hermes_action(["checkpoints", "prune"], "checkpoints-prune")
    except Exception as exc:
        _log.exception("Failed to spawn checkpoints prune")
        raise HTTPException(status_code=500, detail=f"Failed to prune checkpoints: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "checkpoints-prune"}


# ---------------------------------------------------------------------------
# Skills hub endpoints — search / install / uninstall / update.
#
# Search and install touch the network (GitHub, hub sources) and run the same
# complex source-router pipeline the CLI uses, so they're spawned as background
# actions whose logs the dashboard tails.  The already-installed skill list +
# enable/disable toggle live in the existing /api/skills endpoints.
# ---------------------------------------------------------------------------


class SkillInstallRequest(BaseModel):
    identifier: str
    profile: Optional[str] = None


def _profile_cli_args(profile: Optional[str]) -> List[str]:
    """Return ``["-p", <name>]`` for a validated non-default profile.

    Hub install/uninstall/update run in a fresh ``hermes`` subprocess, and
    ``_apply_profile_override()`` reads ``-p`` from argv in the child — the
    only mechanism that reaches import-time-bound globals like
    ``skills_hub.SKILLS_DIR``. Empty/"current" means the dashboard's own
    profile (no args, legacy behavior).
    """
    requested = (profile or "").strip()
    if not requested or requested.lower() in {"current", "default"}:
        return []
    from hermes_cli import profiles as profiles_mod
    _resolve_profile_dir(requested)
    return ["-p", profiles_mod.normalize_profile_name(requested)]


@app.post("/api/skills/hub/install")
async def install_skill_hub(body: SkillInstallRequest, profile: Optional[str] = None):
    identifier = (body.identifier or "").strip()
    if not identifier:
        raise HTTPException(status_code=400, detail="identifier is required")
    try:
        proc = _spawn_hermes_action(
            _profile_cli_args(body.profile or profile)
            + ["skills", "install", identifier, "--yes"],
            "skills-install",
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn skills install")
        raise HTTPException(status_code=500, detail=f"Failed to install skill: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "skills-install"}


class SkillUninstallRequest(BaseModel):
    name: str
    profile: Optional[str] = None


@app.post("/api/skills/hub/uninstall")
async def uninstall_skill_hub(body: SkillUninstallRequest, profile: Optional[str] = None):
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    try:
        proc = _spawn_hermes_action(
            _profile_cli_args(body.profile or profile) + ["skills", "uninstall", name, "--yes"],
            "skills-uninstall",
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn skills uninstall")
        raise HTTPException(status_code=500, detail=f"Failed to uninstall skill: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "skills-uninstall"}


class SkillsUpdateRequest(BaseModel):
    profile: Optional[str] = None


@app.post("/api/skills/hub/update")
async def update_skills_hub(
    body: Optional[SkillsUpdateRequest] = None, profile: Optional[str] = None
):
    try:
        effective = (body.profile if body else None) or profile
        proc = _spawn_hermes_action(
            _profile_cli_args(effective) + ["skills", "update"], "skills-update"
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn skills update")
        raise HTTPException(status_code=500, detail=f"Failed to update skills: {exc}")
    return {"ok": True, "pid": proc.pid, "name": "skills-update"}


# Human-readable labels for each hub source id (matches `hermes skills search`
# provenance).  Keep in sync with create_source_router()'s source list.
_SKILL_HUB_SOURCE_LABELS = {
    "official": "Official (Nous)",
    "hermes-index": "Hermes Index",
    "skills-sh": "skills.sh",
    "well-known": "Well-Known",
    "url": "Direct URL",
    "github": "GitHub",
    "clawhub": "ClawHub",
    "claude-marketplace": "Claude Marketplace",
    "lobehub": "LobeHub",
    "browse-sh": "browse.sh",
}


def _skill_meta_to_payload(m) -> dict:
    return {
        "name": m.name,
        "description": m.description,
        "source": m.source,
        "identifier": m.identifier,
        "trust_level": m.trust_level,
        "repo": m.repo,
        "tags": list(m.tags or []),
    }


def _installed_hub_identifiers(profile: Optional[str] = None) -> dict:
    """Map identifier -> installed lock entry for hub-installed skills.

    Lets the UI mark search results that are already installed.  Scoped to
    ``profile``'s skills/.hub/lock.json when provided (HubLockFile takes an
    explicit path, sidestepping the import-time LOCK_FILE binding).
    Best-effort: returns an empty dict if the lock file can't be read.
    """
    try:
        from tools.skills_hub import HubLockFile

        requested = (profile or "").strip()
        if requested and requested.lower() != "current":
            profile_dir = _resolve_profile_dir(requested)
            lock = HubLockFile(profile_dir / "skills" / ".hub" / "lock.json")
        else:
            lock = HubLockFile()
        out = {}
        for entry in lock.list_installed():
            ident = entry.get("identifier")
            if ident:
                out[ident] = {
                    "name": entry.get("name"),
                    "trust_level": entry.get("trust_level"),
                    "scan_verdict": entry.get("scan_verdict"),
                }
        return out
    except Exception:
        return {}


@app.get("/api/skills/hub/sources")
async def list_skills_hub_sources(profile: Optional[str] = None):
    """List the configured skill-hub sources and installed-skill provenance.

    Gives the dashboard something to show BEFORE a search runs — which hubs
    are wired up, their trust tier, and a set of featured skills pulled from
    the centralized index (zero extra API calls).  Without this the Browse-hub
    tab is a blank page with no indication it's even connected to anything.
    ``profile`` scopes the installed-skill provenance to that profile.
    """

    def _run():
        from tools.skills_hub import create_source_router

        sources = create_source_router()
        out = []
        index_available = False
        featured = []
        for src in sources:
            sid = src.source_id()
            entry = {
                "id": sid,
                "label": _SKILL_HUB_SOURCE_LABELS.get(sid, sid),
            }
            # GitHub exposes a rate-limit flag; the index an availability flag.
            if sid == "github":
                try:
                    entry["rate_limited"] = bool(getattr(src, "is_rate_limited", False))
                except Exception:
                    entry["rate_limited"] = False
            if sid == "hermes-index":
                try:
                    index_available = bool(getattr(src, "is_available", False))
                except Exception:
                    index_available = False
                entry["available"] = index_available
                # Empty-query search on the index returns featured/popular skills.
                if index_available:
                    try:
                        featured = [
                            _skill_meta_to_payload(m) for m in src.search("", limit=12)
                        ]
                    except Exception:
                        featured = []
            out.append(entry)
        return {
            "sources": out,
            "index_available": index_available,
            "featured": featured,
            "installed": _installed_hub_identifiers(profile),
        }

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("skills hub sources listing failed")
        raise HTTPException(status_code=502, detail=f"Hub sources failed: {exc}")


@app.get("/api/skills/hub/search")
async def search_skills_hub(
    q: str = "", source: str = "all", limit: int = 20, profile: Optional[str] = None
):
    """Search the skill hub across all configured sources.

    Network-bound (parallel source search); runs in a thread so the FastAPI
    loop isn't blocked.  Returns structured results the UI installs by
    identifier via POST /api/skills/hub/install, previews via
    /api/skills/hub/preview, and scans via /api/skills/hub/scan.
    """
    query = (q or "").strip()
    if not query:
        return {"results": [], "source_counts": {}, "timed_out": [], "installed": {}}

    def _run():
        from tools.skills_hub import create_source_router, parallel_search_sources

        sources = create_source_router()
        capped = min(max(limit, 1), 50)
        all_results, source_counts, timed_out = parallel_search_sources(
            sources, query=query, source_filter=source or "all", overall_timeout=30
        )

        # Dedupe by identifier, preferring higher trust (mirrors unified_search).
        _rank = {"builtin": 2, "trusted": 1, "community": 0}
        seen = {}
        for r in all_results:
            if r.identifier not in seen:
                seen[r.identifier] = r
            elif _rank.get(r.trust_level, 0) > _rank.get(seen[r.identifier].trust_level, 0):
                seen[r.identifier] = r
        deduped = list(seen.values())[:capped]

        return {
            "results": [_skill_meta_to_payload(m) for m in deduped],
            "source_counts": source_counts,
            "timed_out": timed_out,
            "installed": _installed_hub_identifiers(profile),
        }

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("skills hub search failed")
        raise HTTPException(status_code=502, detail=f"Hub search failed: {exc}")


@app.get("/api/skills/hub/preview")
async def preview_skill_hub(identifier: str = ""):
    """Fetch a hub skill's SKILL.md content + metadata for in-dashboard reading.

    Resolves the identifier across configured sources (same path the CLI
    installer uses), then returns the rendered SKILL.md text and the file
    manifest WITHOUT installing anything.  This is the 'read the actual skill
    before installing' affordance the Browse-hub tab was missing.
    """
    ident = (identifier or "").strip()
    if not ident:
        raise HTTPException(status_code=400, detail="identifier is required")

    def _run():
        from hermes_cli.skills_hub import _resolve_source_meta_and_bundle
        from tools.skills_hub import create_source_router

        sources = create_source_router()
        meta, bundle, _src = _resolve_source_meta_and_bundle(ident, sources)
        if not bundle and not meta:
            return None

        files = {}
        skill_md = ""
        if bundle:
            for rel, content in (bundle.files or {}).items():
                if isinstance(content, bytes):
                    # Some sources (e.g. official optional skills) store every
                    # file as bytes.  Decode text so SKILL.md / docs render;
                    # only fall back to a placeholder for genuinely-binary data.
                    try:
                        files[rel] = content.decode("utf-8")
                    except UnicodeDecodeError:
                        files[rel] = "(binary file)"
                else:
                    files[rel] = content
            skill_md = files.get("SKILL.md", "") or ""

        m = meta or bundle
        return {
            "name": getattr(m, "name", ident),
            "description": getattr(m, "description", "") or "",
            "source": getattr(m, "source", "") or "",
            "identifier": getattr(m, "identifier", ident) or ident,
            "trust_level": getattr(m, "trust_level", "community") or "community",
            "repo": getattr(m, "repo", None),
            "tags": list(getattr(m, "tags", None) or []),
            "skill_md": skill_md,
            "files": sorted(files.keys()),
        }

    try:
        result = await asyncio.to_thread(_run)
    except Exception as exc:
        _log.exception("skills hub preview failed")
        raise HTTPException(status_code=502, detail=f"Hub preview failed: {exc}")
    if result is None:
        raise HTTPException(status_code=404, detail=f"Skill not found: {ident}")
    return result


@app.get("/api/skills/hub/scan")
async def scan_skill_hub(identifier: str = ""):
    """Run the install-time security scan on a hub skill WITHOUT installing it.

    Fetches the bundle, quarantines it, and runs the same `scan_skill` /
    `should_allow_install` pipeline the CLI installer uses — then cleans up the
    quarantine.  Returns the verdict, per-finding detail, trust tier, and the
    install-policy decision so the dashboard can show a visual safety result
    on demand (the 'scan' button the Browse-hub tab was missing).
    """
    ident = (identifier or "").strip()
    if not ident:
        raise HTTPException(status_code=400, detail="identifier is required")

    def _run():
        import shutil as _shutil

        from hermes_cli.skills_hub import _resolve_source_meta_and_bundle
        from tools.skills_hub import create_source_router, quarantine_bundle
        from tools.skills_guard import scan_skill, should_allow_install

        sources = create_source_router()
        meta, bundle, _src = _resolve_source_meta_and_bundle(ident, sources)
        if not bundle:
            return None

        if bundle.source == "official":
            scan_source = "official"
        else:
            scan_source = (
                getattr(bundle, "identifier", "")
                or getattr(meta, "identifier", "")
                or ident
            )

        q_path = None
        try:
            q_path = quarantine_bundle(bundle)
            result = scan_skill(q_path, source=scan_source)
        finally:
            if q_path is not None:
                _shutil.rmtree(q_path, ignore_errors=True)

        allowed, reason = should_allow_install(result, force=False)
        # `allowed` may be None ("ask") for agent-created/dangerous gates.
        if allowed is True:
            policy = "allow"
        elif allowed is None:
            policy = "ask"
        else:
            policy = "block"

        findings = [
            {
                "severity": f.severity,
                "category": f.category,
                "file": f.file,
                "line": f.line,
                "description": f.description,
            }
            for f in result.findings
        ]
        # Per-severity tally for an at-a-glance summary.
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for f in result.findings:
            if f.severity in counts:
                counts[f.severity] += 1

        return {
            "name": result.skill_name,
            "identifier": ident,
            "source": result.source,
            "trust_level": result.trust_level,
            "verdict": result.verdict,
            "summary": result.summary,
            "policy": policy,
            "policy_reason": reason,
            "findings": findings,
            "severity_counts": counts,
        }

    try:
        result = await asyncio.to_thread(_run)
    except Exception as exc:
        _log.exception("skills hub scan failed")
        raise HTTPException(status_code=502, detail=f"Hub scan failed: {exc}")
    if result is None:
        raise HTTPException(status_code=404, detail=f"Skill not found: {ident}")
    return result


# ---------------------------------------------------------------------------
# Profile management endpoints (minimal — list/create/rename/delete + SOUL.md)
# ---------------------------------------------------------------------------


class ProfileCreate(BaseModel):
    name: str
    clone_from: Optional[str] = None
    # Backward compatibility for older dashboard/desktop clients. New clients
    # send clone_from="default" (or another profile name) explicitly.
    clone_from_default: bool = False
    clone_all: bool = False
    no_skills: bool = False
    description: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    # Profile-builder additions — all optional, all applied best-effort AFTER
    # the profile directory exists, so a hiccup in any of them never 500s the
    # create (the user can fix it from the relevant dashboard page afterward).
    # MCP servers to write into the new profile's config.yaml.
    mcp_servers: List["MCPServerCreate"] = []
    # Built-in / optional skills to KEEP active. When this list is non-empty,
    # the builder uses "replace" semantics: the bundle is seeded, then every
    # seeded skill NOT in this list is added to the profile's disabled list.
    # Empty list = leave the seeded bundle untouched (legacy behaviour).
    keep_skills: List[str] = []
    # Skills-hub identifiers to install into the new profile. Installed async
    # via a subprocess scoped to the profile (`hermes -p <name> skills install`)
    # because skills_hub.SKILLS_DIR is import-time-bound and the HERMES_HOME
    # override can't redirect it. Returns spawned PIDs for the UI to poll.
    hub_skills: List[str] = []


class ProfileRename(BaseModel):
    new_name: str


class ProfileSoulUpdate(BaseModel):
    content: str


class ProfileActiveUpdate(BaseModel):
    name: str


class ProfileDescriptionUpdate(BaseModel):
    description: str = ""


class ProfileModelUpdate(BaseModel):
    provider: str
    model: str


class ProfileDescribeAuto(BaseModel):
    overwrite: bool = False


def _profile_attr(info, name: str, default: Any = None) -> Any:
    try:
        return getattr(info, name)
    except Exception:
        return default


def _profile_to_dict(info) -> Dict[str, Any]:
    return {
        "name": _profile_attr(info, "name", ""),
        "path": str(_profile_attr(info, "path", "")),
        "is_default": bool(_profile_attr(info, "is_default", False)),
        "model": _profile_attr(info, "model"),
        "provider": _profile_attr(info, "provider"),
        "has_env": bool(_profile_attr(info, "has_env", False)),
        "skill_count": int(_profile_attr(info, "skill_count", 0) or 0),
        "gateway_running": bool(_profile_attr(info, "gateway_running", False)),
        "description": _profile_attr(info, "description", "") or "",
        "description_auto": bool(_profile_attr(info, "description_auto", False)),
        "distribution_name": _profile_attr(info, "distribution_name"),
        "distribution_version": _profile_attr(info, "distribution_version"),
        "distribution_source": _profile_attr(info, "distribution_source"),
        "has_alias": _profile_attr(info, "alias_path") is not None,
    }


def _fallback_profile_dicts(profiles_mod) -> List[Dict[str, Any]]:
    def _safe(callable_, default):
        try:
            return callable_()
        except Exception:
            return default

    profiles: List[Dict[str, Any]] = []
    default_home = profiles_mod._get_default_hermes_home()
    if default_home.is_dir():
        model, provider = _safe(lambda: profiles_mod._read_config_model(default_home), (None, None))
        profiles.append({
            "name": "default",
            "path": str(default_home),
            "is_default": True,
            "model": model,
            "provider": provider,
            "has_env": (default_home / ".env").exists(),
            "skill_count": _safe(lambda: profiles_mod._count_skills(default_home), 0),
            "gateway_running": _safe(lambda: profiles_mod._check_gateway_running(default_home), False),
            "description": _safe(lambda: profiles_mod.read_profile_meta(default_home).get("description", ""), ""),
            "description_auto": _safe(lambda: profiles_mod.read_profile_meta(default_home).get("description_auto", False), False),
            "distribution_name": None,
            "distribution_version": None,
            "distribution_source": None,
            "has_alias": False,
        })

    profiles_root = profiles_mod._get_profiles_root()
    if profiles_root.is_dir():
        for entry in sorted(profiles_root.iterdir()):
            if not entry.is_dir() or not profiles_mod._PROFILE_ID_RE.match(entry.name):
                continue
            model, provider = _safe(lambda entry=entry: profiles_mod._read_config_model(entry), (None, None))
            profiles.append({
                "name": entry.name,
                "path": str(entry),
                "is_default": False,
                "model": model,
                "provider": provider,
                "has_env": (entry / ".env").exists(),
                "skill_count": _safe(lambda entry=entry: profiles_mod._count_skills(entry), 0),
                "gateway_running": _safe(lambda entry=entry: profiles_mod._check_gateway_running(entry), False),
                "description": _safe(lambda entry=entry: profiles_mod.read_profile_meta(entry).get("description", ""), ""),
                "description_auto": _safe(lambda entry=entry: profiles_mod.read_profile_meta(entry).get("description_auto", False), False),
                "distribution_name": None,
                "distribution_version": None,
                "distribution_source": None,
                "has_alias": False,
            })

    return profiles


def _resolve_profile_dir(name: str) -> Path:
    """Validate ``name`` and resolve to its directory or raise an HTTPException."""
    from hermes_cli import profiles as profiles_mod
    try:
        profiles_mod.validate_profile_name(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not profiles_mod.profile_exists(name):
        raise HTTPException(status_code=404, detail=f"Profile '{name}' does not exist.")
    return profiles_mod.get_profile_dir(name)


def _profile_setup_command(name: str) -> str:
    """Return the shell command used to configure a profile in the CLI."""
    _resolve_profile_dir(name)
    return "hermes setup" if name == "default" else f"{name} setup"


def _write_profile_model(profile_dir: Path, provider: str, model: str) -> None:
    """Write the main model assignment into a specific profile's config.yaml.

    Scopes ``load_config``/``save_config`` to ``profile_dir`` via the
    context-local HERMES_HOME override so the write lands in the target
    profile's config rather than the dashboard process's active profile.
    Clears any stale ``base_url`` / ``context_length`` the same way
    ``POST /api/model/set`` does, since the new model may differ.
    """
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    token = set_hermes_home_override(str(profile_dir))
    try:
        provider, model = _normalize_main_model_assignment(provider, model)
        cfg = load_config()
        cfg["model"] = _apply_main_model_assignment(cfg.get("model", {}), provider, model)
        save_config(cfg)
    finally:
        reset_hermes_home_override(token)


def _write_profile_mcp_servers(profile_dir: Path, servers: List["MCPServerCreate"]) -> int:
    """Write MCP server entries into a specific profile's config.yaml.

    Scopes ``load_config``/``save_config`` to ``profile_dir`` via the
    context-local HERMES_HOME override (same mechanism as
    ``_write_profile_model``) so the entries land in the target profile's
    config rather than the dashboard process's active profile.

    Mirrors the per-server shape the ``POST /api/mcp/servers`` endpoint builds,
    but batched so the whole profile-create write is a single config save.
    Returns the number of servers written.
    """
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.mcp_security import validate_mcp_server_entry

    written = 0
    token = set_hermes_home_override(str(profile_dir))
    try:
        cfg = load_config()
        mcp = cfg.setdefault("mcp_servers", {})
        for server in servers:
            name = (server.name or "").strip()
            if not name:
                continue
            entry: Dict[str, Any] = {}
            if server.url:
                entry["url"] = server.url
            if server.command:
                entry["command"] = server.command
            if server.args:
                entry["args"] = list(server.args)
            if server.env:
                entry["env"] = dict(server.env)
            if server.auth:
                entry["auth"] = server.auth
            if not entry:
                # Nothing usable to write (neither url nor command) — skip
                # rather than persist an empty, unusable server stanza.
                continue
            issues = validate_mcp_server_entry(name, entry)
            if issues:
                _log.warning("Profile-create: skipping MCP server '%s': %s", name, "; ".join(issues))
                continue
            mcp[name] = entry
            written += 1
        if written:
            save_config(cfg)
        elif not mcp:
            # We created an empty mcp_servers dict but wrote nothing — don't
            # leave a stray empty key in the new profile's config.
            cfg.pop("mcp_servers", None)
            save_config(cfg)
    finally:
        reset_hermes_home_override(token)
    return written


def _disable_unselected_skills(profile_dir: Path, keep: List[str]) -> int:
    """Disable every installed skill in ``profile_dir`` not in ``keep``.

    Profiles manage skill activation via a *disabled* list — all installed
    skills are active by default and users opt out. The builder's skill step
    uses "replace" semantics: the user picks exactly which seeded built-in /
    optional skills stay active, and everything else gets added to the disabled
    list. (Hub skills are installed separately via subprocess and are active on
    install.) Scoped to the profile via the HERMES_HOME override. Returns the
    number of skills newly disabled.
    """
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.skills_config import get_disabled_skills, save_disabled_skills

    keep_set = {s.strip() for s in keep if s and s.strip()}
    disabled_count = 0
    token = set_hermes_home_override(str(profile_dir))
    try:
        installed: List[str] = []
        skills_root = profile_dir / "skills"
        if skills_root.is_dir():
            for md in skills_root.rglob("SKILL.md"):
                installed.append(md.parent.name)
        cfg = load_config()
        disabled = get_disabled_skills(cfg)
        for name in installed:
            if name not in keep_set and name not in disabled:
                disabled.add(name)
                disabled_count += 1
        if disabled_count:
            save_disabled_skills(cfg, disabled)
    finally:
        reset_hermes_home_override(token)
    return disabled_count


@app.get("/api/profiles")
async def list_profiles_endpoint():
    from hermes_cli import profiles as profiles_mod
    try:
        loop = asyncio.get_running_loop()
        profiles = await loop.run_in_executor(None, profiles_mod.list_profiles)
        return {"profiles": [_profile_to_dict(p) for p in profiles]}
    except Exception:
        _log.exception("GET /api/profiles failed; falling back to profile directory scan")
        return {"profiles": _fallback_profile_dicts(profiles_mod)}


@app.post("/api/profiles")
async def create_profile_endpoint(body: ProfileCreate):
    from hermes_cli import profiles as profiles_mod
    explicit_source = (body.clone_from or "").strip()
    if explicit_source:
        # Duplicating a specific profile: clone its config/skills/SOUL (or full
        # state when clone_all) from the named source rather than "default".
        clone = True
        clone_from = explicit_source
        clone_config = not body.clone_all
    elif body.clone_all:
        # Preserve the dashboard's historical clone-all behavior: a full-copy
        # request with no explicit dropdown source copies from default.
        clone = True
        clone_from = "default"
        clone_config = False
    else:
        clone = body.clone_from_default
        clone_from = "default" if clone else None
        clone_config = clone
    try:
        path = profiles_mod.create_profile(
            name=body.name,
            clone_from=clone_from,
            clone_all=body.clone_all,
            clone_config=clone_config,
            no_skills=body.no_skills,
            description=body.description,
        )
        # Match the CLI's profile-create flow: fresh named profiles get the
        # bundled skills installed. When cloning from default, create_profile()
        # has already copied the source profile's skills, including any
        # user-installed skills. When no_skills=True, create_profile() wrote
        # the opt-out marker and seed_profile_skills() will no-op.
        if not clone:
            profiles_mod.seed_profile_skills(path, quiet=True)

        # Match the CLI's profile-create flow: named profiles should get a
        # wrapper in ~/.local/bin when the alias is safe to create.
        collision = profiles_mod.check_alias_collision(body.name)
        if not collision:
            profiles_mod.create_wrapper_script(body.name)
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("POST /api/profiles failed")
        raise HTTPException(status_code=500, detail=str(e))

    # Optional explicit model assignment for the new profile. Best-effort:
    # the profile already exists, so a model-write hiccup must not 500 the
    # whole create — the user can set the model later from the Models page
    # or `<profile> setup`.
    provider = (body.provider or "").strip()
    model = (body.model or "").strip()
    model_set = False
    if provider and model:
        try:
            _write_profile_model(path, provider, model)
            model_set = True
        except Exception:
            _log.exception("Setting model for new profile %s failed", body.name)

    # Optional MCP servers. Best-effort, same rationale as model assignment.
    mcp_written = 0
    if body.mcp_servers:
        try:
            mcp_written = _write_profile_mcp_servers(path, body.mcp_servers)
        except Exception:
            _log.exception("Writing MCP servers for new profile %s failed", body.name)

    # Optional "keep" skill selection — replace semantics. When the builder
    # sends an explicit keep list, disable every seeded skill not in it.
    # Best-effort. Skipped when keep_skills is empty (legacy: keep the bundle).
    skills_disabled = 0
    if body.keep_skills:
        try:
            skills_disabled = _disable_unselected_skills(path, body.keep_skills)
        except Exception:
            _log.exception("Applying skill selection for new profile %s failed", body.name)

    # Optional skills-hub installs. Spawned async, scoped to the new profile
    # via `-p <name>` (a fresh subprocess re-binds skills_hub.SKILLS_DIR to the
    # profile's HERMES_HOME at import). Returns PIDs for the UI to poll.
    hub_installs: List[Dict[str, Any]] = []
    for identifier in body.hub_skills:
        ident = (identifier or "").strip()
        if not ident:
            continue
        try:
            proc = _spawn_hermes_action(
                ["-p", body.name, "skills", "install", ident, "--yes"],
                "skills-install",
            )
            hub_installs.append({"identifier": ident, "pid": proc.pid})
        except Exception:
            _log.exception(
                "Spawning hub-skill install %s for new profile %s failed",
                ident,
                body.name,
            )
            hub_installs.append({"identifier": ident, "pid": None})

    return {
        "ok": True,
        "name": body.name,
        "path": str(path),
        "model_set": model_set,
        "mcp_written": mcp_written,
        "skills_disabled": skills_disabled,
        "hub_installs": hub_installs,
    }


@app.get("/api/profiles/active")
async def get_active_profile_endpoint():
    """Return the sticky active profile and the profile this dashboard
    process is currently running as.

    ``active`` is the sticky default written by ``hermes profile use`` —
    the profile new CLI invocations pick up. ``current`` is the profile
    the running dashboard/gateway is scoped to (derived from HERMES_HOME).
    """
    from hermes_cli import profiles as profiles_mod
    try:
        active = profiles_mod.get_active_profile() or "default"
    except Exception:
        active = "default"
    try:
        current = profiles_mod.get_active_profile_name() or "default"
    except Exception:
        current = "default"
    return {"active": active, "current": current}


@app.post("/api/profiles/active")
async def set_active_profile_endpoint(body: ProfileActiveUpdate):
    """Set the sticky active profile (mirrors ``hermes profile use``).

    Note: this does not retarget the already-running dashboard process —
    it changes which profile subsequent CLI commands and gateways use.
    """
    from hermes_cli import profiles as profiles_mod
    try:
        profiles_mod.set_active_profile(body.name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("POST /api/profiles/active failed")
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "active": profiles_mod.normalize_profile_name(body.name)}


@app.get("/api/profiles/{name}/setup-command")
async def get_profile_setup_command(name: str):
    return {"command": _profile_setup_command(name)}


@app.post("/api/profiles/{name}/open-terminal")
async def open_profile_terminal_endpoint(name: str):
    try:
        command = _profile_setup_command(name)

        if sys.platform.startswith("win"):
            subprocess.Popen(["cmd.exe", "/c", "start", "", command])
        elif sys.platform == "darwin":
            escaped = command.replace("\\", "\\\\").replace('"', '\\"')
            applescript = (
                'tell application "Terminal"\n'
                "activate\n"
                f'do script "{escaped}"\n'
                "end tell"
            )
            subprocess.Popen(["osascript", "-e", applescript])
        else:
            terminal_commands = [
                ("x-terminal-emulator", ["x-terminal-emulator", "-e", "sh", "-lc", command]),
                ("gnome-terminal", ["gnome-terminal", "--", "sh", "-lc", command]),
                ("konsole", ["konsole", "-e", "sh", "-lc", command]),
                ("xfce4-terminal", ["xfce4-terminal", "-e", f"sh -lc '{command}'"]),
                ("mate-terminal", ["mate-terminal", "-e", f"sh -lc '{command}'"]),
                ("lxterminal", ["lxterminal", "-e", f"sh -lc '{command}'"]),
                ("tilix", ["tilix", "-e", "sh", "-lc", command]),
                ("alacritty", ["alacritty", "-e", "sh", "-lc", command]),
                ("kitty", ["kitty", "sh", "-lc", command]),
                ("xterm", ["xterm", "-e", "sh", "-lc", command]),
            ]
            for executable, popen_args in terminal_commands:
                if subprocess.call(
                    ["which", executable],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ) == 0:
                    subprocess.Popen(popen_args)
                    break
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No supported terminal emulator found",
                )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        _log.exception("POST /api/profiles/%s/open-terminal failed", name)
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "command": command}


@app.patch("/api/profiles/{name}")
async def rename_profile_endpoint(name: str, body: ProfileRename):
    from hermes_cli import profiles as profiles_mod
    try:
        path = profiles_mod.rename_profile(name, body.new_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, FileExistsError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("PATCH /api/profiles/%s failed", name)
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "name": body.new_name, "path": str(path)}


@app.delete("/api/profiles/{name}")
async def delete_profile_endpoint(name: str):
    """Delete a profile. The dashboard collects the user's confirmation in
    its own dialog before this request, so we always pass ``yes=True`` to
    skip the CLI's interactive prompt."""
    from hermes_cli import profiles as profiles_mod
    try:
        path = profiles_mod.delete_profile(name, yes=True)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _log.exception("DELETE /api/profiles/%s failed", name)
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "path": str(path)}


@app.get("/api/profiles/{name}/soul")
async def get_profile_soul(name: str):
    soul_path = _resolve_profile_dir(name) / "SOUL.md"
    if soul_path.exists():
        try:
            return {"content": soul_path.read_text(encoding="utf-8"), "exists": True}
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Could not read SOUL.md: {e}")
    return {"content": "", "exists": False}


@app.put("/api/profiles/{name}/soul")
async def update_profile_soul(name: str, body: ProfileSoulUpdate):
    soul_path = _resolve_profile_dir(name) / "SOUL.md"
    try:
        soul_path.write_text(body.content, encoding="utf-8")
    except OSError as e:
        _log.exception("PUT /api/profiles/%s/soul failed", name)
        raise HTTPException(status_code=500, detail=f"Could not write SOUL.md: {e}")
    return {"ok": True}


@app.put("/api/profiles/{name}/description")
async def update_profile_description_endpoint(name: str, body: ProfileDescriptionUpdate):
    """Set or clear a profile's role description (kanban routing signal).

    Empty string clears the description. Non-empty stores it as a
    user-authored description (``description_auto: false``) so the
    auto-describer won't overwrite it on a sweep.
    """
    from hermes_cli import profiles as profiles_mod
    profile_dir = _resolve_profile_dir(name)
    text = (body.description or "").strip()
    try:
        profiles_mod.write_profile_meta(
            profile_dir,
            description=text,
            description_auto=False,
        )
    except Exception as e:
        _log.exception("PUT /api/profiles/%s/description failed", name)
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "description": text, "description_auto": False}


@app.put("/api/profiles/{name}/model")
async def update_profile_model_endpoint(name: str, body: ProfileModelUpdate):
    """Set the main model (``model.default`` + ``model.provider``) for a
    specific profile's config.yaml, without touching the dashboard's own
    active profile. Mirrors ``POST /api/model/set`` (main scope) but scoped
    to the named profile via the HERMES_HOME override.
    """
    profile_dir = _resolve_profile_dir(name)
    provider = (body.provider or "").strip()
    model = (body.model or "").strip()
    if not provider or not model:
        raise HTTPException(status_code=400, detail="provider and model are required")
    try:
        _write_profile_model(profile_dir, provider, model)
    except Exception as e:
        _log.exception("PUT /api/profiles/%s/model failed", name)
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": True, "provider": provider, "model": model}


@app.post("/api/profiles/{name}/describe-auto")
async def describe_profile_auto_endpoint(name: str, body: ProfileDescribeAuto):
    """Auto-generate a profile's description via the auxiliary LLM
    (``auxiliary.profile_describer``). Mirrors ``hermes profile describe
    <name> --auto``.

    A failed generation (no aux client, LLM error, …) is returned as
    ``ok: false`` with a reason rather than an HTTP error so the UI can
    surface it inline and let the operator fix config and retry.
    """
    _resolve_profile_dir(name)
    try:
        from hermes_cli import profile_describer
        outcome = profile_describer.describe_profile(name, overwrite=bool(body.overwrite))
    except Exception as e:
        _log.exception("POST /api/profiles/%s/describe-auto failed", name)
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "ok": bool(outcome.ok),
        "reason": outcome.reason,
        "description": outcome.description,
        # Only a successful generation is an auto-authored description. A failed
        # sweep leaves any existing description untouched, so don't claim it's
        # auto-generated.
        "description_auto": bool(outcome.ok),
    }


# ---------------------------------------------------------------------------
# Skills & Tools endpoints
#
# Every read/write below accepts an optional ``profile`` query param so the
# dashboard can manage ANY profile's skills/toolsets, not just the profile
# the dashboard process happens to be running under. Without this, "Set as
# active" on the Profiles page (which only flips the sticky ``active_profile``
# file for FUTURE CLI/gateway invocations) misled users into thinking skill
# toggles would land in the activated profile — they silently wrote into the
# dashboard's own config instead. See _profile_scope() for the mechanism.
# ---------------------------------------------------------------------------


_SKILLS_PROFILE_LOCK = threading.RLock()


@contextmanager
def _profile_scope(profile: Optional[str]):
    """Scope config + skill-directory resolution to ``profile`` for one request.

    Two seams must be redirected for skills/toolsets endpoints:

    1. ``load_config``/``save_config`` resolve ``get_hermes_home()`` at call
       time — the context-local override from ``set_hermes_home_override``
       reaches them (same pattern as ``_write_profile_model``).
    2. ``tools.skills_tool`` and ``tools.skill_manager_tool`` bind
       ``SKILLS_DIR`` at import time, so the override CANNOT reach them.
       Like ``_call_cron_for_profile`` does for cron's module globals,
       temporarily retarget both under a lock and restore them
       immediately after.

    ``profile`` of None/""/"current" means "the dashboard's own profile" —
    config resolution is untouched, but the skill-module globals are still
    retargeted to the *current* ``get_hermes_home()`` so writes land in the
    live home even when the import-time binding is stale (e.g. the process
    imported the modules before a HERMES_HOME override, or under test
    isolation).
    """
    requested = (profile or "").strip()

    from hermes_constants import (
        get_hermes_home,
        set_hermes_home_override,
        reset_hermes_home_override,
    )
    from tools import skills_tool as _skills_tool
    from tools import skill_manager_tool as _skill_mgr

    token = None
    if not requested or requested.lower() == "current":
        profile_dir = get_hermes_home()
    else:
        profile_dir = _resolve_profile_dir(requested)
        token = set_hermes_home_override(str(profile_dir))

    with _SKILLS_PROFILE_LOCK:
        old_home = _skills_tool.HERMES_HOME
        old_skills_dir = _skills_tool.SKILLS_DIR
        old_mgr_home = _skill_mgr.HERMES_HOME
        old_mgr_skills_dir = _skill_mgr.SKILLS_DIR
        _skills_tool.HERMES_HOME = profile_dir
        _skills_tool.SKILLS_DIR = profile_dir / "skills"
        _skill_mgr.HERMES_HOME = profile_dir
        _skill_mgr.SKILLS_DIR = profile_dir / "skills"
        try:
            yield profile_dir if token is not None else None
        finally:
            _skills_tool.HERMES_HOME = old_home
            _skills_tool.SKILLS_DIR = old_skills_dir
            _skill_mgr.HERMES_HOME = old_mgr_home
            _skill_mgr.SKILLS_DIR = old_mgr_skills_dir
            if token is not None:
                reset_hermes_home_override(token)


@contextmanager
def _config_profile_scope(profile: Optional[str]):
    """Await-safe, config-only profile scope for handlers that ``await``.

    Unlike ``_profile_scope`` this touches ONLY the context-local
    ``set_hermes_home_override`` contextvar — it does NOT swap the
    process-global ``skills_tool``/``skill_manager`` module attributes.
    Those globals are shared across all event-loop tasks, so holding them
    across an ``await`` lets a concurrent skills request restore THIS
    request's profile dir on its ``finally`` (cross-contamination). The
    contextvar override is task-local and survives an ``await`` cleanly,
    which is all endpoints that resolve ``get_hermes_home()`` at call time
    (config, env, gateway status) actually need.

    None/""/"current" means the dashboard's own profile — no override.
    """
    requested = (profile or "").strip()
    if not requested or requested.lower() == "current":
        yield None
        return

    from hermes_constants import (
        set_hermes_home_override,
        reset_hermes_home_override,
    )

    profile_dir = _resolve_profile_dir(requested)
    token = set_hermes_home_override(str(profile_dir))
    try:
        yield profile_dir
    finally:
        reset_hermes_home_override(token)


class SkillToggle(BaseModel):
    name: str
    enabled: bool
    profile: Optional[str] = None


@app.get("/api/skills")
async def get_skills(profile: Optional[str] = None):
    from tools.skills_tool import _find_all_skills
    from hermes_cli.skills_config import get_disabled_skills
    with _profile_scope(profile):
        config = load_config()
        disabled = get_disabled_skills(config)
        skills = _find_all_skills(skip_disabled=True)
    for s in skills:
        s["enabled"] = s["name"] not in disabled
    return skills


@app.put("/api/skills/toggle")
async def toggle_skill(body: SkillToggle, profile: Optional[str] = None):
    from hermes_cli.skills_config import get_disabled_skills, save_disabled_skills
    with _profile_scope(body.profile or profile):
        config = load_config()
        disabled = get_disabled_skills(config)
        if body.enabled:
            disabled.discard(body.name)
        else:
            disabled.add(body.name)
        save_disabled_skills(config, disabled)
    return {"ok": True, "name": body.name, "enabled": body.enabled}


class SkillCreate(BaseModel):
    name: str
    content: str
    category: Optional[str] = None
    profile: Optional[str] = None


class SkillContentUpdate(BaseModel):
    name: str
    content: str
    profile: Optional[str] = None


def _clear_skills_prompt_cache() -> None:
    """Best-effort: invalidate the skills system-prompt snapshot after a write.

    Mirrors what ``skill_manage`` does so a dashboard-authored skill is picked
    up by the next session without a manual cache reset.
    """
    try:
        from agent.prompt_builder import clear_skills_system_prompt_cache
        clear_skills_system_prompt_cache(clear_snapshot=True)
    except Exception:
        pass


@app.get("/api/skills/content")
async def get_skill_content(name: str, profile: Optional[str] = None):
    """Return the raw SKILL.md text for a skill, for the dashboard editor."""
    from tools.skill_manager_tool import _find_skill

    with _profile_scope(profile):
        found = _find_skill(name)
        if not found:
            raise HTTPException(status_code=404, detail=f"Skill '{name}' not found.")
        skill_md = found["path"] / "SKILL.md"
        if not skill_md.exists():
            raise HTTPException(status_code=404, detail=f"Skill '{name}' has no SKILL.md.")
        try:
            content = skill_md.read_text(encoding="utf-8")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"name": name, "content": content, "path": str(skill_md)}


@app.post("/api/skills")
async def create_skill(body: SkillCreate):
    """Create a new custom skill (SKILL.md) from the dashboard editor.

    Calls the same validated write path as the agent's ``skill_manage``
    tool (frontmatter validation, name/category validation, size limit,
    optional security scan) — but bypasses the agent write-approval gate:
    a write from the authenticated dashboard IS the user acting directly.
    """
    from tools.skill_manager_tool import _create_skill

    with _profile_scope(body.profile):
        result = _create_skill(body.name, body.content, body.category or None)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to create skill."))
    _clear_skills_prompt_cache()
    return result


@app.put("/api/skills/content")
async def update_skill_content(body: SkillContentUpdate):
    """Replace the SKILL.md of an existing skill (full rewrite) from the editor."""
    from tools.skill_manager_tool import _edit_skill

    with _profile_scope(body.profile):
        result = _edit_skill(body.name, body.content)
    if not result.get("success"):
        err = result.get("error", "Failed to update skill.")
        status = 404 if "not found" in str(err).lower() else 400
        raise HTTPException(status_code=status, detail=err)
    _clear_skills_prompt_cache()
    return result


@app.get("/api/tools/toolsets")
async def get_toolsets(profile: Optional[str] = None):
    from hermes_cli.tools_config import (
        _get_effective_configurable_toolsets,
        _get_platform_tools,
        _toolset_has_keys,
        gui_toolset_label,
    )
    from toolsets import resolve_toolset

    with _profile_scope(profile):
        config = load_config()
        enabled_toolsets = _get_platform_tools(
            config,
            "cli",
            include_default_mcp_servers=False,
        )
    result = []
    for name, label, desc in _get_effective_configurable_toolsets():
        try:
            tools = sorted(set(resolve_toolset(name)))
        except Exception:
            tools = []
        is_enabled = name in enabled_toolsets
        result.append({
            "name": name,
            "label": gui_toolset_label(label),
            "description": desc,
            "enabled": is_enabled,
            "available": is_enabled,
            "configured": _toolset_has_keys(name, config),
            "tools": tools,
        })
    return result


class ToolsetToggle(BaseModel):
    enabled: bool
    profile: Optional[str] = None


@app.put("/api/tools/toolsets/{name}")
async def toggle_toolset(name: str, body: ToolsetToggle, profile: Optional[str] = None):
    """Enable/disable a configurable toolset for the desktop (cli) platform.

    Persists to ``platform_toolsets.cli`` via the same ``_save_platform_tools``
    helper the CLI ``hermes tools`` picker uses, so the GUI and CLI stay in
    lockstep. Scoped to ``body.profile`` when provided. Returns 400 for
    unknown toolset keys.
    """
    from hermes_cli.tools_config import (
        _get_effective_configurable_toolsets,
        _get_platform_tools,
        _save_platform_tools,
    )

    valid = {ts_key for ts_key, _, _ in _get_effective_configurable_toolsets()}
    if name not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown toolset: {name}")

    with _profile_scope(body.profile or profile):
        config = load_config()
        enabled = set(
            _get_platform_tools(config, "cli", include_default_mcp_servers=False)
        )
        if body.enabled:
            enabled.add(name)
        else:
            enabled.discard(name)
        _save_platform_tools(config, "cli", enabled)
    return {"ok": True, "name": name, "enabled": body.enabled}


@app.get("/api/tools/toolsets/{name}/config")
async def get_toolset_config(name: str, profile: Optional[str] = None):
    """Return the provider matrix + key status for a toolset's config panel.

    Surfaces the same provider rows the CLI ``hermes tools`` picker shows
    (via ``_visible_providers``), each with its ``env_vars`` annotated with
    current ``is_set`` state so the GUI can render provider selection + key
    entry. Toolsets without a ``TOOL_CATEGORIES`` entry return an empty
    provider list and ``has_category: false``. Returns 400 for unknown keys.
    """
    from hermes_cli.tools_config import (
        TOOL_CATEGORIES,
        _get_effective_configurable_toolsets,
        _is_provider_active,
        _visible_providers,
    )
    from hermes_cli.config import get_env_value

    valid = {ts_key for ts_key, _, _ in _get_effective_configurable_toolsets()}
    if name not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown toolset: {name}")

    with _profile_scope(profile):
        config = load_config()
        cat = TOOL_CATEGORIES.get(name)
        providers = []
        active_provider = None
        if cat:
            for prov in _visible_providers(cat, config, force_fresh=True):
                env_vars = [
                    {
                        "key": e["key"],
                        "prompt": e.get("prompt", e["key"]),
                        "url": e.get("url"),
                        "default": e.get("default"),
                        "is_set": bool(get_env_value(e["key"])),
                    }
                    for e in prov.get("env_vars", [])
                ]
                # Surface the same active-provider determination the CLI picker
                # uses (``_is_provider_active``) so the GUI highlights the provider
                # actually written to config (e.g. web.backend), not just the first
                # keyless one in the list.
                is_active = _is_provider_active(prov, config, force_fresh=True)
                if is_active and active_provider is None:
                    active_provider = prov["name"]
                providers.append({
                    "name": prov["name"],
                    "badge": prov.get("badge", ""),
                    "tag": prov.get("tag", ""),
                    "env_vars": env_vars,
                    "post_setup": prov.get("post_setup"),
                    "requires_nous_auth": bool(prov.get("requires_nous_auth")),
                    "is_active": is_active,
                })
    return {
        "name": name,
        "has_category": cat is not None,
        "providers": providers,
        "active_provider": active_provider,
    }


class ToolsetProviderSelect(BaseModel):
    provider: str
    profile: Optional[str] = None


@app.put("/api/tools/toolsets/{name}/provider")
async def select_toolset_provider(
    name: str, body: ToolsetProviderSelect, profile: Optional[str] = None
):
    """Persist a provider selection for a toolset (no key prompting).

    Delegates to ``apply_provider_selection`` — the shared, non-interactive
    core extracted from the CLI configurator — so the GUI and ``hermes tools``
    write identical config keys (``web.backend``, ``tts.provider``, etc.).
    API keys and post-setup flows are handled by separate endpoints. Returns
    400 for unknown toolset or provider names.
    """
    from hermes_cli.tools_config import (
        apply_provider_selection,
        _get_effective_configurable_toolsets,
    )

    valid = {ts_key for ts_key, _, _ in _get_effective_configurable_toolsets()}
    if name not in valid:
        raise HTTPException(status_code=400, detail=f"Unknown toolset: {name}")

    with _profile_scope(body.profile or profile):
        config = load_config()
        try:
            apply_provider_selection(name, body.provider, config)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc).strip('"'))
        save_config(config)
    return {"ok": True, "name": name, "provider": body.provider}


class ToolsetEnvUpdate(BaseModel):
    env: Dict[str, str]
    profile: Optional[str] = None


@app.put("/api/tools/toolsets/{name}/env")
async def save_toolset_env(name: str, body: ToolsetEnvUpdate, profile: Optional[str] = None):
    """Persist API keys for a toolset's provider env vars.

    Writes each ``key: value`` to ``~/.hermes/.env`` via ``save_env_value`` —
    the same store ``hermes tools`` writes when it prompts for keys. Keys are
    validated against the env-var allowlist for the toolset's category (the
    union of every visible provider's ``env_vars``), so the GUI can't write an
    arbitrary env var through this endpoint. A blank value is treated as
    "leave unchanged" and skipped. Returns the saved/skipped key lists and the
    refreshed ``is_set`` status. Returns 400 for unknown toolset or env keys.
    """
    from hermes_cli.tools_config import (
        TOOL_CATEGORIES,
        _get_effective_configurable_toolsets,
        _visible_providers,
    )
    from hermes_cli.config import get_env_value, save_env_value

    valid_ts = {ts_key for ts_key, _, _ in _get_effective_configurable_toolsets()}
    if name not in valid_ts:
        raise HTTPException(status_code=400, detail=f"Unknown toolset: {name}")

    with _profile_scope(body.profile or profile):
        config = load_config()
        cat = TOOL_CATEGORIES.get(name)
        allowed: set[str] = set()
        if cat:
            for prov in _visible_providers(cat, config, force_fresh=True):
                for e in prov.get("env_vars", []):
                    allowed.add(e["key"])

        unknown = [k for k in body.env if k not in allowed]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown env var(s) for toolset {name}: {', '.join(sorted(unknown))}",
            )

        saved: List[str] = []
        skipped: List[str] = []
        for key, value in body.env.items():
            if value and value.strip():
                try:
                    save_env_value(key, value.strip())
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc))
                saved.append(key)
            else:
                skipped.append(key)

        status = {k: bool(get_env_value(k)) for k in allowed}
    return {"ok": True, "name": name, "saved": saved, "skipped": skipped, "is_set": status}


class ToolsetPostSetup(BaseModel):
    key: str
    profile: Optional[str] = None


@app.post("/api/tools/toolsets/{name}/post-setup")
async def run_toolset_post_setup(
    name: str, body: ToolsetPostSetup, profile: Optional[str] = None
):
    """Spawn a provider's post-setup install hook as a background action.

    Post-setup hooks (npm install for browser/Camofox, pip install for
    KittenTTS/Piper/ddgs, cua-driver fetch, etc.) are long-running and
    text-output, so this follows the spawn-action pattern: it launches
    ``hermes tools post-setup <key>`` and the frontend tails the log via
    ``GET /api/actions/tools-post-setup/status``. The ``key`` is validated
    against the declared post-setup allowlist before spawning. Returns 400
    for unknown toolset or post-setup key.

    ``profile`` spawns the hook as ``hermes -p <profile> tools post-setup``.
    Most hooks install machine-level artifacts (repo node_modules, shared
    pip packages) where the scope is inert, but hooks that read config or
    write per-profile state must see the same HERMES_HOME the rest of the
    drawer's writes targeted — so the scope is threaded for consistency.
    """
    from hermes_cli.tools_config import (
        _get_effective_configurable_toolsets,
        valid_post_setup_keys,
    )

    valid_ts = {ts_key for ts_key, _, _ in _get_effective_configurable_toolsets()}
    if name not in valid_ts:
        raise HTTPException(status_code=400, detail=f"Unknown toolset: {name}")

    if body.key not in valid_post_setup_keys():
        raise HTTPException(
            status_code=400, detail=f"Unknown post-setup key: {body.key}"
        )

    try:
        proc = _spawn_hermes_action(
            _profile_cli_args(body.profile or profile)
            + ["tools", "post-setup", body.key],
            "tools-post-setup",
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn tools post-setup")
        raise HTTPException(
            status_code=500, detail=f"Failed to run post-setup: {exc}"
        )
    return {"ok": True, "pid": proc.pid, "name": "tools-post-setup", "key": body.key}


# ---------------------------------------------------------------------------
# Computer Use (cua-driver) — cross-platform readiness + macOS permission grant
#
# cua-driver runs on macOS, Windows, and Linux. The desktop card reflects
# per-OS readiness: on macOS the Accessibility + Screen Recording TCC grants
# (which attach to cua-driver's OWN identity, com.trycua.driver — not Hermes,
# so no app entitlement is involved); elsewhere, driver health from
# `cua-driver doctor`. The grant flow is macOS-only (no TCC toggles to request
# on Windows/Linux).
# ---------------------------------------------------------------------------


@app.get("/api/tools/computer-use/status")
async def get_computer_use_status(profile: Optional[str] = None):
    """Cross-platform Computer Use readiness for the desktop card.

    See ``tools.computer_use.permissions.computer_use_status`` for the payload
    shape. Read-only and fast (shells ``cua-driver doctor`` + macOS
    ``permissions status``).
    """
    from tools.computer_use.permissions import computer_use_status

    with _profile_scope(profile):
        return computer_use_status()


@app.post("/api/tools/computer-use/permissions/grant")
async def grant_computer_use_permissions(profile: Optional[str] = None):
    """Spawn ``hermes computer-use permissions grant`` as a background action.

    macOS-only: ``cua-driver permissions grant`` launches CuaDriver via
    LaunchServices so the TCC dialog is attributed to com.trycua.driver, then
    waits for approval. The frontend polls ``GET /api/actions/computer-use-
    grant/status`` and re-reads ``/status`` once it exits. Windows/Linux have
    no TCC toggles to grant, so this returns 400 there.
    """
    if sys.platform != "darwin":
        raise HTTPException(
            status_code=400,
            detail="Computer Use permission grants are a macOS concept.",
        )
    try:
        proc = _spawn_hermes_action(
            _profile_cli_args(profile)
            + ["computer-use", "permissions", "grant"],
            "computer-use-grant",
        )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("Failed to spawn computer-use permissions grant")
        raise HTTPException(
            status_code=500, detail=f"Failed to request permissions: {exc}"
        )
    return {"ok": True, "pid": proc.pid, "name": "computer-use-grant"}


# ---------------------------------------------------------------------------
# Raw YAML config endpoint
# ---------------------------------------------------------------------------


class RawConfigUpdate(BaseModel):
    yaml_text: str
    profile: Optional[str] = None


@app.get("/api/config/raw")
async def get_config_raw(profile: Optional[str] = None):
    """Raw config.yaml text plus its resolved path.

    ``path`` is resolved inside ``_profile_scope`` so the Config page header
    shows the file the switched profile actually reads/writes — /api/status's
    ``config_path`` is machine-global and always reports the dashboard
    process's own profile, which is wrong under the global profile switcher.
    """
    with _profile_scope(profile):
        path = get_config_path()
    if not path.exists():
        return {"yaml": "", "path": str(path)}
    return {"yaml": path.read_text(encoding="utf-8"), "path": str(path)}


@app.put("/api/config/raw")
async def update_config_raw(body: RawConfigUpdate, profile: Optional[str] = None):
    try:
        parsed = yaml.safe_load(body.yaml_text)
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="YAML must be a mapping")
        with _profile_scope(body.profile or profile):
            save_config(parsed)
        return {"ok": True}
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")


# ---------------------------------------------------------------------------
# Token / cost analytics endpoint
# ---------------------------------------------------------------------------


@app.get("/api/analytics/usage")
async def get_usage_analytics(days: int = 30, profile: Optional[str] = None):
    from agent.insights import InsightsEngine

    db = _open_session_db_for_profile(profile)
    try:
        cutoff = time.time() - (days * 86400)
        cur = db._conn.execute("""
            SELECT date(started_at, 'unixepoch') as day,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   SUM(cache_read_tokens) as cache_read_tokens,
                   SUM(reasoning_tokens) as reasoning_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as actual_cost,
                   COUNT(*) as sessions,
                   SUM(COALESCE(api_call_count, 0)) as api_calls
            FROM sessions WHERE started_at > ?
            GROUP BY day ORDER BY day
        """, (cutoff,))
        daily = [dict(r) for r in cur.fetchall()]

        cur2 = db._conn.execute("""
            SELECT model,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COUNT(*) as sessions,
                   SUM(COALESCE(api_call_count, 0)) as api_calls
            FROM sessions WHERE started_at > ? AND model IS NOT NULL
            GROUP BY model ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC
        """, (cutoff,))
        by_model = [dict(r) for r in cur2.fetchall()]

        cur3 = db._conn.execute("""
            SELECT SUM(input_tokens) as total_input,
                   SUM(output_tokens) as total_output,
                   SUM(cache_read_tokens) as total_cache_read,
                   SUM(reasoning_tokens) as total_reasoning,
                   COALESCE(SUM(estimated_cost_usd), 0) as total_estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as total_actual_cost,
                   COUNT(*) as total_sessions,
                   SUM(COALESCE(api_call_count, 0)) as total_api_calls
            FROM sessions WHERE started_at > ?
        """, (cutoff,))
        totals = dict(cur3.fetchone())
        insights_report = InsightsEngine(db).generate(days=days)
        skills = insights_report.get("skills", {
            "summary": {
                "total_skill_loads": 0,
                "total_skill_edits": 0,
                "total_skill_actions": 0,
                "distinct_skills_used": 0,
            },
            "top_skills": [],
        })

        return {
            "daily": daily,
            "by_model": by_model,
            "totals": totals,
            "period_days": days,
            "skills": skills,
        }
    finally:
        db.close()


@app.get("/api/analytics/models")
async def get_models_analytics(days: int = 30, profile: Optional[str] = None):
    """Rich per-model analytics for the Models dashboard page.

    Returns token/cost/session breakdown per model plus capability metadata
    from models.dev (context window, vision, tools, reasoning, etc.).
    """
    db = _open_session_db_for_profile(profile)
    try:
        cutoff = time.time() - (days * 86400)

        cur = db._conn.execute("""
            SELECT model,
                   billing_provider,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   SUM(cache_read_tokens) as cache_read_tokens,
                   SUM(reasoning_tokens) as reasoning_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as actual_cost,
                   COUNT(*) as sessions,
                   SUM(COALESCE(api_call_count, 0)) as api_calls,
                   SUM(tool_call_count) as tool_calls,
                   MAX(started_at) as last_used_at,
                   AVG(input_tokens + output_tokens) as avg_tokens_per_session
            FROM sessions WHERE started_at > ? AND model IS NOT NULL AND model != ''
            GROUP BY model, billing_provider
            ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC
        """, (cutoff,))
        raw_rows = [dict(r) for r in cur.fetchall()]

        # Session rows can be created before the first billable provider call
        # finishes. If that early row records only the model name, and a later
        # row for the same model has real accounting + billing_provider, the
        # Models page used to show a duplicate "0 tokens / — API calls" card
        # next to the real provider card. Fold those session-only rows into
        # the single accounted provider row when the ownership is unambiguous.
        rows_by_model: Dict[str, List[Dict[str, Any]]] = {}
        for row in raw_rows:
            rows_by_model.setdefault(row.get("model") or "", []).append(row)

        rows: List[Dict[str, Any]] = []
        for model_rows in rows_by_model.values():
            provider_rows = [r for r in model_rows if r.get("billing_provider")]
            if len(provider_rows) == 1:
                target = provider_rows[0]
                for row in model_rows:
                    if row is target or row.get("billing_provider"):
                        continue
                    has_usage = any(
                        (row.get(key) or 0) != 0
                        for key in (
                            "input_tokens",
                            "output_tokens",
                            "cache_read_tokens",
                            "reasoning_tokens",
                            "estimated_cost",
                            "actual_cost",
                            "api_calls",
                            "tool_calls",
                        )
                    )
                    if has_usage:
                        continue
                    target["sessions"] = (target.get("sessions") or 0) + (row.get("sessions") or 0)
                    target["last_used_at"] = max(target.get("last_used_at") or 0, row.get("last_used_at") or 0)
                    total_tokens = (target.get("input_tokens") or 0) + (target.get("output_tokens") or 0)
                    sessions = target.get("sessions") or 0
                    target["avg_tokens_per_session"] = total_tokens / sessions if sessions else 0
                rows.append(target)
                rows.extend(
                    r for r in model_rows
                    if r is not target
                    and (r.get("billing_provider") or any(
                        (r.get(key) or 0) != 0
                        for key in (
                            "input_tokens",
                            "output_tokens",
                            "cache_read_tokens",
                            "reasoning_tokens",
                            "estimated_cost",
                            "actual_cost",
                            "api_calls",
                            "tool_calls",
                        )
                    ))
                )
            else:
                rows.extend(model_rows)

        rows.sort(
            key=lambda r: (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0),
            reverse=True,
        )

        models = []
        for row in rows:
            provider = row.get("billing_provider") or ""
            model_name = row["model"]
            caps = {}
            try:
                from agent.models_dev import get_model_capabilities
                mc = get_model_capabilities(provider=provider, model=model_name)
                if mc is not None:
                    caps = {
                        "supports_tools": mc.supports_tools,
                        "supports_vision": mc.supports_vision,
                        "supports_reasoning": mc.supports_reasoning,
                        "context_window": mc.context_window,
                        "max_output_tokens": mc.max_output_tokens,
                        "model_family": mc.model_family,
                    }
            except Exception:
                pass

            models.append({
                "model": model_name,
                "provider": provider,
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "cache_read_tokens": row["cache_read_tokens"],
                "reasoning_tokens": row["reasoning_tokens"],
                "estimated_cost": row["estimated_cost"],
                "actual_cost": row["actual_cost"],
                "sessions": row["sessions"],
                "api_calls": row["api_calls"],
                "tool_calls": row["tool_calls"],
                "last_used_at": row["last_used_at"],
                "avg_tokens_per_session": row["avg_tokens_per_session"],
                "capabilities": caps,
            })

        totals_cur = db._conn.execute("""
            SELECT COUNT(DISTINCT model) as distinct_models,
                   SUM(input_tokens) as total_input,
                   SUM(output_tokens) as total_output,
                   SUM(cache_read_tokens) as total_cache_read,
                   SUM(reasoning_tokens) as total_reasoning,
                   COALESCE(SUM(estimated_cost_usd), 0) as total_estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as total_actual_cost,
                   COUNT(*) as total_sessions,
                   SUM(COALESCE(api_call_count, 0)) as total_api_calls
            FROM sessions WHERE started_at > ? AND model IS NOT NULL AND model != ''
        """, (cutoff,))
        totals = dict(totals_cur.fetchone())

        return {
            "models": models,
            "totals": totals,
            "period_days": days,
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# /api/pty — PTY-over-WebSocket bridge for the dashboard "Chat" tab.
#
# The endpoint spawns the same ``hermes --tui`` binary the CLI uses, behind
# a POSIX pseudo-terminal, and forwards bytes + resize escapes across a
# WebSocket.  The browser renders the ANSI through xterm.js (see
# web/src/pages/ChatPage.tsx).
#
# Auth: ``?token=<session_token>`` query param (browsers can't set
# Authorization on the WS upgrade).  Same ephemeral ``_SESSION_TOKEN`` as
# REST.  Localhost-only — we defensively reject non-loopback clients even
# though uvicorn binds to 127.0.0.1.
# ---------------------------------------------------------------------------

# PTY bridge: POSIX uses pty_bridge (fcntl/termios/ptyprocess); native Windows
# uses win_pty_bridge (pywinpty/ConPTY, already a declared dependency).  Both
# expose the same public surface — spawn/read/write/resize/close/is_available —
# so the /api/pty WebSocket handler needs no platform guards.
if sys.platform.startswith("win"):
    try:
        from hermes_cli.win_pty_bridge import WinPtyBridge as PtyBridge, PtyUnavailableError
        _PTY_BRIDGE_AVAILABLE = True
    except ImportError:  # pragma: no cover - pywinpty missing
        PtyBridge = None  # type: ignore[assignment]
        _PTY_BRIDGE_AVAILABLE = False

        class PtyUnavailableError(RuntimeError):  # type: ignore[no-redef]
            """Stub when win_pty_bridge cannot be imported."""
            pass
else:
    try:
        from hermes_cli.pty_bridge import PtyBridge, PtyUnavailableError
        _PTY_BRIDGE_AVAILABLE = True
    except ImportError:  # pragma: no cover - dev env without ptyprocess
        PtyBridge = None  # type: ignore[assignment]
        _PTY_BRIDGE_AVAILABLE = False

        class PtyUnavailableError(RuntimeError):  # type: ignore[no-redef]
            """Stub on platforms where pty_bridge can't be imported."""
            pass

_RESIZE_RE = re.compile(rb"\x1b\[RESIZE:(\d+);(\d+)\]")
_PTY_READ_CHUNK_TIMEOUT = 0.2
_VALID_CHANNEL_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
# Starlette's TestClient reports the peer as "testclient"; treat it as
# loopback so tests don't need to rewrite request scope.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})


def _ws_client_reason(ws: "WebSocket") -> Optional[str]:
    """Return a rejection reason for the client IP, or None when allowed.

    Reasons are short machine-parseable tokens logged on the rejection path
    so a "WS keeps closing" report can be diagnosed from agent.log without a
    repro. ``None`` means the peer IP passed this gate.

    See :func:`_ws_client_is_allowed` for the full policy rationale.
    """
    if getattr(app.state, "auth_required", False):
        return None
    bound_host = (getattr(app.state, "bound_host", "") or "").strip().lower()
    if bound_host and bound_host not in _LOOPBACK_HOSTS:
        return None
    client_host = ws.client.host if ws.client else ""
    if not client_host:
        # Fail-closed: a loopback-bound dashboard with auth disabled must
        # not accept a WebSocket with no identifiable peer. ASGI servers
        # behind a misconfigured proxy or unix socket can deliver
        # ws.client == None or "" — treating that as "allowed" would let
        # an unidentified peer reach a loopback-only surface.
        return f"missing_or_empty_peer bound={bound_host or '?'}"
    if client_host in _LOOPBACK_HOSTS:
        return None
    return f"peer_not_loopback peer={client_host} bound={bound_host or '?'}"


def _ws_client_is_allowed(ws: "WebSocket") -> bool:
    """Check if the WebSocket client IP is acceptable.

    Loopback bind: only loopback clients allowed — the legacy
    ``?token=<_SESSION_TOKEN>`` path is the only auth we have, so we
    don't want LAN hosts guessing tokens.

    Explicit non-loopback bind (``--host 0.0.0.0``, ``--host ::``, or a
    specific address such as a Tailscale/LAN IP, always with
    ``--insecure``): allow any peer. The operator explicitly opted into
    non-loopback exposure, so the loopback-only peer restriction does not
    apply. DNS-rebinding is still blocked by the Host/Origin guard in
    :func:`_ws_host_origin_is_allowed`, which mirrors the HTTP layer and
    requires the Host header to match the bound interface — the same
    defence ``_is_accepted_host`` applies to non-loopback HTTP requests.

    Gated mode: any peer is allowed — uvicorn's ``proxy_headers=True``
    (enabled when the OAuth gate is active so cookies can pick up
    ``X-Forwarded-Proto``) rewrites ``ws.client.host`` to the
    X-Forwarded-For value, which is the real internet client IP. The
    OAuth gate + single-use ``?ticket=`` is the auth at that point; the
    Host/Origin guard in :func:`_ws_host_origin_is_allowed` is what
    blocks DNS-rebinding here, not the peer IP.
    """
    if getattr(app.state, "auth_required", False):
        return True
    # Any explicit non-loopback bind (0.0.0.0, ::, or a specific LAN /
    # Tailscale address) means the operator opted into non-loopback
    # access via --insecure.  The loopback-only peer gate only applies to
    # an actual loopback bind; otherwise the WS handshake is rejected even
    # though same-bind HTTP requests pass _is_accepted_host.
    bound_host = (getattr(app.state, "bound_host", "") or "").strip().lower()
    if bound_host and bound_host not in _LOOPBACK_HOSTS:
        return True
    client_host = ws.client.host if ws.client else ""
    if not client_host:
        # Fail-closed: see _ws_client_reason for rationale. An empty
        # client_host on a loopback-bound dashboard with auth disabled
        # must be rejected, not accepted as a default-allow.
        return False
    return client_host in _LOOPBACK_HOSTS


def _ws_host_origin_reason(ws: "WebSocket") -> Optional[str]:
    """Return a Host/Origin rejection reason, or None when allowed.

    Mirrors :func:`_ws_host_origin_is_allowed` but yields a short
    machine-parseable token (``host_mismatch …`` / ``origin_mismatch …``)
    on rejection so the close path can log *why* the upgrade was refused.
    """
    bound_host = getattr(app.state, "bound_host", None)
    if not bound_host:
        return None

    host_header = ws.headers.get("host", "")
    if not _is_accepted_host(host_header, bound_host):
        return f"host_mismatch host={host_header or '?'} bound={bound_host}"

    origin = ws.headers.get("origin", "")
    if not origin:
        return None

    parsed = urllib.parse.urlparse(origin)
    if parsed.scheme not in {"http", "https"}:
        # Non-web origin (packaged Electron: file://, null, app://). The
        # upstream credential check is the real auth boundary; trust it.
        # See _ws_host_origin_is_allowed for the full rationale.
        return None

    if not parsed.netloc:
        return f"origin_mismatch origin={origin} bound={bound_host}"

    if not _is_accepted_host(parsed.netloc, bound_host):
        return f"origin_mismatch origin={origin} bound={bound_host}"
    return None


def _ws_host_origin_is_allowed(ws: "WebSocket") -> bool:
    """Apply the dashboard Host/Origin guard to WebSocket upgrades.

    FastAPI HTTP middleware does not run for WebSocket routes, so the
    DNS-rebinding Host check used for normal dashboard HTTP requests must be
    repeated here before accepting the upgrade.  Browsers also send an Origin
    header on WebSocket handshakes; when present, require it to target the
    same bound dashboard host.
    """
    return _ws_host_origin_reason(ws) is None


def _ws_request_reason(ws: "WebSocket") -> Optional[str]:
    """First Host/Origin or peer-IP rejection reason, or None when allowed."""
    return _ws_host_origin_reason(ws) or _ws_client_reason(ws)


def _ws_request_is_allowed(ws: "WebSocket") -> bool:
    """Return True when the WebSocket upgrade matches dashboard boundaries."""
    return _ws_host_origin_is_allowed(ws) and _ws_client_is_allowed(ws)


def _ws_auth_mode() -> str:
    """Short label for the active WS auth mode — logged on every connection."""
    if getattr(app.state, "auth_required", False):
        return "gated"
    bound_host = (getattr(app.state, "bound_host", "") or "").strip().lower()
    if bound_host and bound_host not in _LOOPBACK_HOSTS:
        return "insecure"
    return "loopback"


def _ws_auth_reason(ws: "WebSocket") -> tuple[Optional[str], str]:
    """Validate WS-upgrade auth; return ``(reason, credential)``.

    ``reason`` is None when the credential is accepted, else a short
    machine-parseable token explaining the rejection (``no_credential``,
    ``token_mismatch``, ``ticket_invalid``, ``internal_invalid``).
    ``credential`` names which credential type was presented (``ticket``,
    ``internal``, ``token``, or ``none``) so the accepted path can log *how*
    a peer authed, not just that it did.

    Loopback / ``--insecure``: legacy ``?token=<_SESSION_TOKEN>`` query
    parameter, constant-time compared.

    Gated (public bind, no ``--insecure``): one of two credentials —

    * ``?ticket=<single-use>`` — a browser-minted, single-use, 30s-TTL ticket
      consumed against the dashboard-auth ticket store. This is what the SPA
      (and native clients) use.
    * ``?internal=<process-credential>`` — the process-lifetime internal
      credential, used only by WS clients the server spawns itself (the
      embedded-TUI PTY child attaching to ``/api/ws`` and ``/api/pub``). It
      is multi-use and never expires so the child can reconnect, and is never
      injected into the SPA — see ``dashboard_auth.ws_tickets`` for the
      threat model.

    The legacy ``?token=`` path is unconditionally rejected in gated mode
    (the SPA bundle isn't carrying the token any longer, and a leaked
    ``_SESSION_TOKEN`` must not grant WS access once the gate is engaged).

    Audit-logs the rejection so operators can debug "WS keeps closing"
    issues from the log.
    """
    auth_required = bool(getattr(app.state, "auth_required", False))
    if auth_required:
        # Lazy import — keeps this function importable in test harnesses
        # that don't bring in the dashboard_auth layer.
        from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
        from hermes_cli.dashboard_auth.ws_tickets import (
            TicketInvalid,
            consume_internal_credential,
            consume_ticket,
        )

        # Server-spawned children (PTY child → /api/ws, /api/pub) present the
        # multi-use internal credential rather than a single-use ticket, so
        # they survive reconnects and slow cold boots.
        internal = ws.query_params.get("internal", "")
        if internal:
            try:
                consume_internal_credential(internal)
                return None, "internal"
            except TicketInvalid as exc:
                audit_log(
                    AuditEvent.WS_TICKET_REJECTED,
                    reason=f"internal: {exc}",
                    ip=(ws.client.host if ws.client else ""),
                    path=ws.url.path,
                )
                return "internal_invalid", "internal"

        ticket = ws.query_params.get("ticket", "")
        if not ticket:
            return "no_credential", "none"

        try:
            consume_ticket(ticket)
            return None, "ticket"
        except TicketInvalid as exc:
            audit_log(
                AuditEvent.WS_TICKET_REJECTED,
                reason=str(exc),
                ip=(ws.client.host if ws.client else ""),
                path=ws.url.path,
            )
            return "ticket_invalid", "ticket"

    token = ws.query_params.get("token", "")
    if not token:
        return "no_credential", "none"
    if hmac.compare_digest(token.encode(), _SESSION_TOKEN.encode()):
        return None, "token"
    return "token_mismatch", "token"


def _ws_auth_ok(ws: "WebSocket") -> bool:
    """True when the WS-upgrade credential is accepted. See _ws_auth_reason."""
    return _ws_auth_reason(ws)[0] is None

# Per-channel subscriber registry used by /api/pub (PTY-side gateway → dashboard)
# and /api/events (dashboard → browser sidebar).  Keyed by an opaque channel id
# the chat tab generates on mount; entries auto-evict when the last subscriber
# drops AND the publisher has disconnected.
# (Channel state and the chat-argv lock are initialised in _lifespan on app
# startup — see _get_event_state / _get_chat_argv_lock above.)


def _resolve_chat_argv(
    resume: Optional[str] = None,
    sidecar_url: Optional[str] = None,
    profile: Optional[str] = None,
    active_session_file: Optional[str] = None,
) -> tuple[list[str], Optional[str], Optional[dict]]:
    """Resolve the argv + cwd + env for the chat PTY.

    Default: whatever ``hermes --tui`` would run.  Tests monkeypatch this
    function to inject a tiny fake command (``cat``, ``sh -c 'printf …'``)
    so nothing has to build Node or the TUI bundle.

    Session resume is propagated via the ``HERMES_TUI_RESUME`` env var —
    matching what ``hermes_cli.main._launch_tui`` does for the CLI path.
    Appending ``--resume <id>`` to argv doesn't work because ``ui-tui`` does
    not parse its argv.

    ``HERMES_TUI_GATEWAY_URL`` is injected so the PTY child can attach to
    this process's in-memory ``tui_gateway`` instance instead of spawning
    its own Python gateway subprocess.

    `sidecar_url` (when set) is forwarded as ``HERMES_TUI_SIDECAR_URL`` so
    the spawned ``tui_gateway.entry`` can mirror dispatcher emits to the
    dashboard's ``/api/pub`` endpoint (see :func:`pub_ws`).

    `active_session_file` (when set) is forwarded as
    ``HERMES_TUI_ACTIVE_SESSION_FILE``. The TUI writes the current session id
    there whenever it creates/resumes/switches sessions, giving the dashboard a
    small cross-process breadcrumb for reconnecting after an unexpected browser
    WebSocket close.

    `profile` (when set) scopes the ENTIRE chat to that profile by pointing
    ``HERMES_HOME`` at the profile dir in the child env. Every spawned
    process (the TUI and the ``tui_gateway.entry`` it launches) resolves
    ``get_hermes_home()`` from that env var at its own import, so the child
    binds the profile's config, skills, memory, and state.db from the start
    — the same propagation ``hermes -p <name>`` performs. The in-process
    ``HERMES_TUI_GATEWAY_URL`` attach is SKIPPED for scoped chats: the
    dashboard's in-memory gateway runs under the dashboard's own profile,
    so a profile-scoped chat must spawn its own gateway subprocess.
    """
    from hermes_cli.main import PROJECT_ROOT, _make_tui_argv

    profile_dir: Optional[Path] = None
    requested = (profile or "").strip()
    if requested and requested.lower() != "current":
        profile_dir = _resolve_profile_dir(requested)

    argv, cwd = _make_tui_argv(PROJECT_ROOT / "ui-tui", tui_dev=False)
    env = os.environ.copy()
    try:
        from hermes_cli.config import apply_terminal_config_to_env
        apply_terminal_config_to_env(env=env)
    except Exception:
        _log.debug("Failed to apply terminal config bridge for dashboard chat", exc_info=True)
    env.setdefault("NODE_ENV", "production")
    # Browser-embedded chat should prefer stable wheel-based scrollback over
    # native terminal mouse tracking. When mouse tracking is enabled, wheel
    # events are consumed by the TUI and forwarded as terminal input, which
    # makes browser-side transcript scrolling feel broken. Keep the terminal
    # build unchanged for native CLI usage; only disable mouse tracking for
    # the dashboard PTY path.
    env.setdefault("HERMES_TUI_DISABLE_MOUSE", "1")
    env.setdefault("HERMES_TUI_INLINE", "1")
    env["HERMES_TUI_DASHBOARD"] = "1"

    if profile_dir is not None:
        env["HERMES_HOME"] = str(profile_dir)

    if resume:
        latest_resume, _latest_path = _session_latest_descendant(resume)
        if latest_resume:
            resume = latest_resume
        env["HERMES_TUI_RESUME"] = resume

    if sidecar_url:
        env["HERMES_TUI_SIDECAR_URL"] = sidecar_url

    if active_session_file:
        env["HERMES_TUI_ACTIVE_SESSION_FILE"] = active_session_file

    # Profile-scoped chats must NOT attach to the dashboard's in-memory
    # gateway — it runs under the dashboard's own profile. Without the
    # attach URL, gatewayClient spawns its own `tui_gateway.entry`, which
    # inherits the profile HERMES_HOME set above.
    if profile_dir is None:
        if gateway_ws_url := _build_gateway_ws_url():
            env["HERMES_TUI_GATEWAY_URL"] = gateway_ws_url

    return list(argv), str(cwd) if cwd else None, env


def _build_gateway_ws_url() -> Optional[str]:
    """ws:// URL the PTY child should attach to for JSON-RPC gateway traffic.

    Loopback / ``--insecure``: ``?token=<_SESSION_TOKEN>``.

    Gated mode: the legacy token path is rejected by ``_ws_auth_ok``, so the
    server-spawned PTY child authenticates with the process-lifetime internal
    credential (``?internal=``). It must NOT use a single-use browser ticket:
    the child reads this URL once at startup and reuses it on every reconnect,
    and a 30s-TTL ticket can expire before a slow cold boot even dials.
    """
    host = getattr(app.state, "bound_host", None)
    port = getattr(app.state, "bound_port", None)

    if not host or not port:
        return None

    netloc = (
        f"[{host}]:{port}"
        if ":" in host and not host.startswith("[")
        else f"{host}:{port}"
    )

    if getattr(app.state, "auth_required", False):
        from hermes_cli.dashboard_auth.ws_tickets import internal_ws_credential

        qs = urllib.parse.urlencode({"internal": internal_ws_credential()})
    else:
        qs = urllib.parse.urlencode({"token": _SESSION_TOKEN})

    return f"ws://{netloc}/api/ws?{qs}"


async def _resolve_chat_argv_async(
    resume: Optional[str] = None,
    sidecar_url: Optional[str] = None,
    profile: Optional[str] = None,
    active_session_file: Optional[str] = None,
) -> tuple[list[str], Optional[str], Optional[dict]]:
    """Resolve chat argv without blocking the dashboard event loop.

    ``_resolve_chat_argv`` may run ``npm install`` / ``npm run build`` through
    ``_make_tui_argv``.  Keep that synchronous work off the WebSocket event
    loop so reverse proxies and existing dashboard connections can continue
    to exchange keepalives while the TUI launch command is prepared.  The
    async lock preserves the previous one-build-at-a-time behavior when
    multiple browser tabs connect at once without occupying worker threads
    while queued connections wait.
    """
    kwargs = {
        "resume": resume,
        "sidecar_url": sidecar_url,
        "profile": profile,
    }
    if active_session_file is not None:
        kwargs["active_session_file"] = active_session_file

    async with _get_chat_argv_lock(app):
        return await asyncio.to_thread(
            _resolve_chat_argv,
            **kwargs,
        )


def _build_sidecar_url(channel: str) -> Optional[str]:
    """ws:// URL the PTY child should publish events to, or None when unbound.

    Loopback / ``--insecure``: uses ``?token=<_SESSION_TOKEN>``.

    Gated mode: authenticates with the process-lifetime internal credential
    (``?internal=``), the same one ``_build_gateway_ws_url`` uses. The PTY
    child is a server-spawned process we trust; the credential is multi-use
    and never expires, so the child can reconnect ``/api/pub`` without a new
    URL. (This previously minted a single-use 30s ticket, which meant the
    child could not reconnect and could miss the window on a slow cold boot.)
    Connections authenticated this way are recorded under the
    ``server-internal`` identity in the audit log.
    """
    host = getattr(app.state, "bound_host", None)
    port = getattr(app.state, "bound_port", None)

    if not host or not port:
        return None

    netloc = f"[{host}]:{port}" if ":" in host and not host.startswith("[") else f"{host}:{port}"

    if getattr(app.state, "auth_required", False):
        # Gated mode — use the internal credential so the WS upgrade survives
        # _ws_auth_ok and the child can reconnect.
        from hermes_cli.dashboard_auth.ws_tickets import internal_ws_credential

        qs = urllib.parse.urlencode(
            {"internal": internal_ws_credential(), "channel": channel}
        )
    else:
        qs = urllib.parse.urlencode({"token": _SESSION_TOKEN, "channel": channel})

    return f"ws://{netloc}/api/pub?{qs}"


async def _broadcast_event(app: Any, channel: str, payload: str) -> None:
    """Fan out one publisher frame to every subscriber on `channel`."""
    event_channels, event_lock = _get_event_state(app)
    async with event_lock:
        subs = list(event_channels.get(channel, ()))

    for sub in subs:
        try:
            await sub.send_text(payload)
        except Exception:
            # Subscriber went away mid-send; the /api/events finally clause
            # will remove it from the registry on its next iteration.
            _log.warning("broadcast send failed for subscriber on %s", channel, exc_info=True)


def _channel_or_close_code(ws: WebSocket) -> Optional[str]:
    """Return the channel id from the query string or None if invalid."""
    channel = ws.query_params.get("channel", "")

    return channel if _VALID_CHANNEL_RE.match(channel) else None


def _active_session_file_for_channel(app: "FastAPI", channel: str) -> Path:
    """Return the per-channel file where a dashboard TUI writes its active sid."""
    files = _get_pty_active_session_files(app)
    existing = files.get(channel)
    if existing is not None:
        return existing

    fd, raw_path = tempfile.mkstemp(prefix="hermes-pty-active-", suffix=".json")
    os.close(fd)
    path = Path(raw_path)
    files[channel] = path
    return path


def _read_active_session_file(path: Path) -> Optional[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    session_id = str(data.get("session_id") or "").strip()
    return session_id or None


def _forget_active_session_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _ws_close_reason(text: str) -> str:
    """Clamp a WS close reason to the protocol's 123-byte UTF-8 limit.

    RFC 6455 caps the close-frame reason at 123 bytes; uvicorn raises if a
    longer string is passed. Our reasons embed an attacker-controlled origin,
    so truncate defensively rather than crash the close handler.
    """
    encoded = text.encode("utf-8", "replace")
    if len(encoded) <= 123:
        return text
    return encoded[:120].decode("utf-8", "ignore") + "..."


def _realtime_voice_sidecar_base_url(realtime: Dict[str, Any]) -> str:
    sidecar_base_url = str(realtime.get("sidecar_base_url") or "").strip()
    if sidecar_base_url:
        return sidecar_base_url.rstrip("/")

    spark_base_url = str(realtime.get("spark_base_url") or "").strip()
    if spark_base_url:
        return spark_base_url.rstrip("/")

    provider = str(realtime.get("frontend_provider") or "").lower()
    if provider in _LOCAL_VOICE_SIDECAR_PROVIDERS:
        sidecar_host = str(realtime.get("sidecar_host") or "127.0.0.1")
        sidecar_port = int(realtime.get("sidecar_port") or 8765)
        return f"http://{sidecar_host}:{sidecar_port}"

    return ""


def _truthy_config(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _positive_float_config(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _positive_int_config(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _bounded_int_config(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    parsed = _positive_int_config(value, default=default)
    return min(maximum, max(minimum, parsed))


def _bounded_nonnegative_int_config(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        parsed = default
    else:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
    if parsed < 0:
        parsed = default
    return min(maximum, max(minimum, parsed))


def _bounded_float_config(value: Any, *, default: float, minimum: float, maximum: float) -> float:
    parsed = _positive_float_config(value, default=default)
    return min(maximum, max(minimum, parsed))


def _realtime_voice_sidecar_is_loopback(base_url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(base_url)
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def _realtime_voice_should_autostart_sidecar(realtime: Dict[str, Any], base_url: str) -> bool:
    provider = str(realtime.get("frontend_provider") or "").lower()
    if provider not in _LOCAL_VOICE_SIDECAR_PROVIDERS:
        return False
    if not _truthy_config(realtime.get("sidecar_autostart"), default=True):
        return False
    if not base_url or not _realtime_voice_sidecar_is_loopback(base_url):
        return False
    return True


def _realtime_voice_sidecar_mode(realtime: Dict[str, Any], base_url: str) -> str:
    if _realtime_voice_should_autostart_sidecar(realtime, base_url):
        return "managed_loopback"
    if base_url:
        return "external"
    return "none"


def _realtime_voice_sidecar_health_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/health"


def _realtime_voice_sidecar_token_env(realtime: Dict[str, Any]) -> str:
    return str(
        realtime.get("sidecar_token_env")
        or realtime.get("spark_token_env")
        or "HERMES_VOICE_SIDECAR_TOKEN"
    )


def _realtime_voice_sidecar_token(
    realtime: Dict[str, Any],
    env_on_disk: Optional[Dict[str, str]] = None,
) -> str:
    token_env = _realtime_voice_sidecar_token_env(realtime)
    env = env_on_disk if env_on_disk is not None else load_env()
    return str(env.get(token_env) or os.environ.get(token_env) or "")


def _realtime_voice_sidecar_health_request(base_url: str, token: str = ""):
    url = _realtime_voice_sidecar_health_url(base_url)
    if not token:
        return url
    return urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})


def _redact_realtime_voice_url(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = urllib.parse.urlparse(value)
    except Exception:
        return value

    netloc = parsed.netloc
    if "@" in netloc:
        _, host_part = netloc.rsplit("@", 1)
        netloc = f"***@{host_part}"
    return urllib.parse.urlunparse((parsed.scheme, netloc, parsed.path, "", "", ""))


def _realtime_voice_sidecar_healthy(
    base_url: str,
    timeout: float = _VOICE_SIDECAR_HEALTH_TIMEOUT,
    token: str = "",
) -> bool:
    return _realtime_voice_sidecar_health_probe(
        base_url,
        timeout=timeout,
        token=token,
    )[0]


def _realtime_voice_sidecar_health_payload(
    base_url: str,
    timeout: float = _VOICE_SIDECAR_HEALTH_TIMEOUT,
    token: str = "",
) -> Optional[Dict[str, Any]]:
    return _realtime_voice_sidecar_health_probe(base_url, timeout=timeout, token=token)[1]


def _realtime_voice_sidecar_health_probe(
    base_url: str,
    timeout: float = _VOICE_SIDECAR_HEALTH_TIMEOUT,
    token: str = "",
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    if not base_url:
        return False, None
    try:
        with urllib.request.urlopen(_realtime_voice_sidecar_health_request(base_url, token), timeout=timeout) as response:
            status = int(getattr(response, "status", 200))
            healthy = 200 <= status < 300
            if not healthy or not hasattr(response, "read"):
                return healthy, None
            raw = response.read()
    except Exception:
        return False, None

    try:
        payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
    except Exception:
        return False, None
    if not isinstance(payload, dict):
        return False, None
    sanitized = _sanitize_realtime_voice_sidecar_health(payload)
    return sanitized.get("ok") is True, sanitized


_REALTIME_VOICE_HEALTH_METADATA_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _sanitize_realtime_voice_health_metadata(value: Any, *, limit: int = 32) -> List[str]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        return []

    sanitized: List[str] = []
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        token = candidate.strip()
        if not _REALTIME_VOICE_HEALTH_METADATA_RE.fullmatch(token):
            continue
        key = token.lower()
        if key in seen:
            continue
        sanitized.append(token)
        seen.add(key)
        if len(sanitized) >= limit:
            break
    return sanitized


def _realtime_voice_metadata_list_or_default(value: Any, default: Sequence[str]) -> List[str]:
    sanitized = _sanitize_realtime_voice_health_metadata(value)
    return sanitized or list(default)


def _realtime_voice_language_support_payload(realtime: Dict[str, Any]) -> Dict[str, Any]:
    production_languages = _realtime_voice_metadata_list_or_default(
        realtime.get("production_languages") or realtime.get("target_languages"),
        _REALTIME_VOICE_DEFAULT_PRODUCTION_LANGUAGES,
    )
    production_scripts = _realtime_voice_metadata_list_or_default(
        realtime.get("production_scripts") or realtime.get("target_scripts"),
        _REALTIME_VOICE_DEFAULT_PRODUCTION_SCRIPTS,
    )
    best_effort_languages = _truthy_config(realtime.get("best_effort_languages"), default=True)
    return {
        "production_languages": production_languages,
        "production_scripts": production_scripts,
        "best_effort_languages": best_effort_languages,
        "sidecar_languages_are_diagnostics": True,
    }


def _realtime_voice_quality_targets_payload(realtime: Dict[str, Any]) -> Dict[str, int]:
    raw = realtime.get("quality_targets_ms") or realtime.get("latency_targets_ms")
    if str(realtime.get("engine") or "").strip() == "kame_interface_oracle":
        targets = dict(_REALTIME_VOICE_KAME_QUALITY_TARGETS_MS)
    else:
        targets = dict(_REALTIME_VOICE_DEFAULT_QUALITY_TARGETS_MS)
    if not isinstance(raw, dict):
        return targets

    for key in targets:
        parsed = _positive_int_config(raw.get(key), default=targets[key])
        targets[key] = parsed
    for key, value in raw.items():
        if key in targets:
            continue
        if key not in _REALTIME_VOICE_KAME_QUALITY_TARGETS_MS:
            continue
        parsed = _positive_int_config(value, default=0)
        if parsed > 0:
            targets[str(key)] = parsed
    return targets


def _realtime_voice_turn_acknowledgement_payload(realtime: Dict[str, Any]) -> Dict[str, Any]:
    raw = realtime.get("turn_acknowledgement")
    if not isinstance(raw, dict):
        raw = {}
    enabled = _truthy_config(raw.get("enabled"), default=False)
    text = str(raw.get("text") or "One moment.").strip()
    return {
        "enabled": enabled,
        "text": text[:120] if text else "One moment.",
    }


def _realtime_voice_routing_policy_payload(realtime: Mapping[str, Any]) -> Dict[str, Any]:
    raw = realtime.get("routing") if isinstance(realtime, Mapping) else {}
    if not isinstance(raw, Mapping):
        raw = {}
    return {
        "allow_local_greetings": _truthy_config(raw.get("allow_local_greetings"), default=True),
        "allow_local_clarifications": _truthy_config(raw.get("allow_local_clarifications"), default=True),
        "require_oracle_for_tools": _truthy_config(raw.get("require_oracle_for_tools"), default=True),
        "require_oracle_for_memory": _truthy_config(raw.get("require_oracle_for_memory"), default=True),
        "require_oracle_for_files": _truthy_config(raw.get("require_oracle_for_files"), default=True),
        "local_confidence_threshold": _bounded_float_config(
            raw.get("local_confidence_threshold"),
            default=0.75,
            minimum=0.0,
            maximum=1.0,
        ),
    }


def _realtime_voice_metrics_policy_payload(realtime: Mapping[str, Any]) -> Dict[str, Any]:
    raw = realtime.get("metrics") if isinstance(realtime, Mapping) else {}
    if not isinstance(raw, Mapping):
        raw = {}
    return {
        "enabled": _truthy_config(raw.get("enabled"), default=True),
        "log_turn_spans": _truthy_config(raw.get("log_turn_spans"), default=True),
        "log_provider_spans": _truthy_config(raw.get("log_provider_spans"), default=True),
    }


def _realtime_voice_oracle_jobs_payload(realtime: Mapping[str, Any]) -> Dict[str, Any]:
    raw = realtime.get("oracle_jobs") if isinstance(realtime, Mapping) else {}
    if not isinstance(raw, Mapping):
        raw = {}
    default_priority = str(raw.get("default_priority") or "normal").strip().lower()
    if default_priority not in {"high", "normal", "low"}:
        default_priority = "normal"
    overflow_policy = str(raw.get("overflow_policy") or "queue").strip().lower()
    if overflow_policy not in {"queue", "reject", "reprioritize"}:
        overflow_policy = "queue"
    return {
        "enabled": _truthy_config(raw.get("enabled"), default=False),
        "max_concurrent": _bounded_int_config(
            raw.get("max_concurrent"),
            default=1,
            minimum=1,
            maximum=16,
        ),
        "queue_limit": _bounded_int_config(
            raw.get("queue_limit"),
            default=16,
            minimum=0,
            maximum=256,
        ),
        "default_priority": default_priority,
        "overflow_policy": overflow_policy,
        "shutdown_timeout_seconds": _positive_float_config(
            raw.get("shutdown_timeout_seconds"),
            default=2.0,
        ),
        "speak_terminal_results": _truthy_config(raw.get("speak_terminal_results"), default=True),
        "audit_ledger_path": str(raw.get("audit_ledger_path") or "").strip(),
    }


def _realtime_voice_kame_stack_payload(
    *,
    engine: str,
    provider: str,
    frontend_model: str,
    interface_base_url: str,
    interface_timeout_seconds: float,
    interface_max_audio_seconds: float,
    interface_audio_input: str,
    realtime: Mapping[str, Any],
    asr_mode: str,
    asr_provider: str,
    asr_model: str,
    asr_base_url: str,
    preferred_local_oracle_model: str,
    oracle_timeout_seconds: float,
    tts_provider: str,
    tts_model: str,
    tts_voice: str,
    tts_base_url: str,
    fallback_policy: str,
) -> Dict[str, Any]:
    interpreter = _mapping_config(realtime.get("interpreter") if isinstance(realtime, Mapping) else {})
    transcript_evidence = _mapping_config(realtime.get("transcript_evidence") if isinstance(realtime, Mapping) else {})
    interpreter_provider = str(
        _first_realtime_voice_config_value(
            realtime,
            ("interpreter_provider",),
            ("interpreter", "provider"),
            default="",
        )
        or ""
    )
    interpreter_model = str(
        _first_realtime_voice_config_value(
            realtime,
            ("interpreter_model",),
            ("interpreter", "model"),
            default="",
        )
        or ""
    )
    interpreter_audio_input = str(
        _first_realtime_voice_config_value(
            realtime,
            ("interpreter_audio_input",),
            ("interpreter", "audio_input"),
            default="native_audio",
        )
        or "native_audio"
    )
    interpreter_base_url = str(
        _first_realtime_voice_config_value(
            realtime,
            ("interpreter_base_url",),
            ("interpreter", "base_url"),
            default="",
        )
        or ""
    )
    transcript_mode = str(transcript_evidence.get("mode") or asr_mode or "")
    transcript_sources = transcript_evidence.get("sources")
    if not isinstance(transcript_sources, list):
        transcript_sources = []

    return {
        "engine": engine,
        "reflex": {
            "provider": provider or "",
            "model": frontend_model or "",
            "audio_input": interface_audio_input or "auto",
            "base_url_configured": bool(interface_base_url),
            "timeout_seconds": interface_timeout_seconds,
            "max_audio_seconds": interface_max_audio_seconds,
            "role": "low_latency_floor_control",
        },
        "interpreter": {
            "provider": interpreter_provider,
            "model": interpreter_model,
            "audio_input": interpreter_audio_input,
            "base_url_configured": bool(interpreter_base_url),
            "role": str(interpreter.get("role") or "raw_audio_evidence_adjudicator"),
        },
        "transcript_evidence": {
            "mode": transcript_mode,
            "provider": asr_provider or "",
            "model": asr_model or "",
            "sources": [str(source) for source in transcript_sources if str(source or "").strip()],
            "base_url_configured": bool(asr_base_url),
            "authority": str(transcript_evidence.get("authority") or "hypothesis"),
            "schedule_oracle_from_transcript": _truthy_config(
                transcript_evidence.get("schedule_oracle_from_transcript"),
                default=False,
            ),
        },
        "oracle": {
            "mode": "hermes_active_model",
            "preferred_local_model": preferred_local_oracle_model or "",
            "timeout_seconds": oracle_timeout_seconds,
        },
        "tts": {
            "provider": tts_provider or "",
            "model": tts_model or "",
            "voice_configured": bool(tts_voice),
            "base_url_configured": bool(tts_base_url),
        },
        "fallback_policy": fallback_policy,
    }


def _realtime_voice_oracle_tool_router_payload(realtime: Mapping[str, Any]) -> Dict[str, Any]:
    raw = realtime.get("oracle_tool_router") if isinstance(realtime, Mapping) else {}
    if not isinstance(raw, Mapping):
        raw = {}
    mode = str(raw.get("mode") or "deterministic").strip().lower()
    if mode not in {"deterministic"}:
        mode = "deterministic"
    return {
        "enabled": _truthy_config(raw.get("enabled"), default=True),
        "mode": mode,
        "voiceops_toolsets": _sanitize_realtime_voice_toolset_list(
            raw.get("voiceops_toolsets"),
            default=["voiceops"],
        ),
        "default_toolsets": _sanitize_realtime_voice_toolset_list(
            raw.get("default_toolsets"),
            default=[],
        ),
        "tool_search": _realtime_voice_oracle_tool_search_payload(raw.get("tool_search")),
    }


def _realtime_voice_oracle_tool_search_payload(raw: Any) -> Dict[str, Any]:
    if raw is False:
        return {"enabled": "off", "defer_core": "off"}
    if not isinstance(raw, Mapping):
        raw = {}

    enabled = str(raw.get("enabled") or "on").strip().lower()
    if enabled not in {"auto", "on", "off"}:
        enabled = "on"
    defer_core = str(raw.get("defer_core") or "all").strip().lower()
    if defer_core not in {"off", "all"}:
        defer_core = "all"
    return {"enabled": enabled, "defer_core": defer_core}


def _sanitize_realtime_voice_toolset_list(value: Any, *, default: Sequence[str]) -> List[str]:
    if isinstance(value, str):
        candidates = re.split(r"[, ]+", value)
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        candidates = list(default)

    sanitized: List[str] = []
    seen = set()
    for candidate in candidates:
        token = str(candidate or "").strip()
        if not token or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", token):
            continue
        key = token.lower()
        if key in seen:
            continue
        sanitized.append(token)
        seen.add(key)
        if len(sanitized) >= 16:
            break
    return sanitized


def _realtime_voice_output_events_payload(realtime: Mapping[str, Any]) -> Dict[str, Any]:
    raw = realtime.get("output_events") if isinstance(realtime, Mapping) else {}
    if not isinstance(raw, Mapping):
        raw = {}
    caption_events = realtime.get("caption_events") if isinstance(realtime, Mapping) else {}
    if not isinstance(caption_events, Mapping):
        caption_events = {}
    return {
        "caption_aliases": _truthy_config(
            raw.get("caption_aliases", caption_events.get("enabled")),
            default=False,
        ),
        "audio_aliases": _truthy_config(raw.get("audio_aliases"), default=False),
    }


def _sanitize_realtime_voice_sidecar_health(payload: Dict[str, Any]) -> Dict[str, Any]:
    frontend = payload.get("frontend") if isinstance(payload.get("frontend"), dict) else {}
    capabilities = payload.get("capabilities") if isinstance(payload.get("capabilities"), dict) else {}
    local = payload.get("local") if isinstance(payload.get("local"), dict) else {}
    vllm_frontend = frontend.get("vllm_audio_frontend")
    vllm_frontend_configured = (
        vllm_frontend.get("configured") is True
        if isinstance(vllm_frontend, dict)
        else capabilities.get("vllm_audio_frontend_configured") is True
    )
    vllm_audio_frontend = (
        vllm_frontend.get("healthy") is True
        if isinstance(vllm_frontend, dict)
        else capabilities.get("vllm_audio_frontend") is True
    )

    result = {
        "ok": payload.get("ok") is True,
        "kind": str(payload.get("kind") or "") or None,
        "frontend": {
            "provider": str(frontend.get("provider") or "") or None,
            "model": str(frontend.get("model") or "") or None,
            "tts_model_languages": _sanitize_realtime_voice_health_metadata(
                frontend.get("tts_model_languages")
            ),
            "languages": _sanitize_realtime_voice_health_metadata(
                frontend.get("languages", frontend.get("language"))
            ),
            "scripts": _sanitize_realtime_voice_health_metadata(frontend.get("scripts", frontend.get("script"))),
        },
        "capabilities": {
            "utterance_stt": capabilities.get("utterance_stt") is True,
            "streaming_stt": capabilities.get("streaming_stt") is True,
            "tts": capabilities.get("tts") is True,
            "native_s2s": capabilities.get("native_s2s") is True,
            "vllm_audio_frontend": vllm_audio_frontend,
            "input_languages": _sanitize_realtime_voice_health_metadata(
                capabilities.get("input_languages", capabilities.get("languages"))
            ),
            "output_languages": _sanitize_realtime_voice_health_metadata(
                capabilities.get("output_languages", capabilities.get("tts_languages"))
            ),
            "scripts": _sanitize_realtime_voice_health_metadata(capabilities.get("scripts")),
        },
        "local": {
            "stt": local.get("stt") is True,
            "tts": local.get("tts") is True,
        },
    }
    if "streaming_stt_bridge" in capabilities:
        result["capabilities"]["streaming_stt_bridge"] = capabilities.get("streaming_stt_bridge") is True
    if "streaming_tts" in capabilities:
        result["capabilities"]["streaming_tts"] = capabilities.get("streaming_tts") is True
    if "streaming_tts_bridge" in capabilities:
        result["capabilities"]["streaming_tts_bridge"] = capabilities.get("streaming_tts_bridge") is True
    if vllm_frontend_configured:
        result["capabilities"]["vllm_audio_frontend_configured"] = True

    stt_bridge = frontend.get("streaming_stt_bridge")
    if isinstance(stt_bridge, dict):
        result["frontend"]["streaming_stt_bridge"] = {
            "configured": stt_bridge.get("configured") is True,
            "healthy": stt_bridge.get("healthy") is True,
        }
    tts_bridge = frontend.get("streaming_tts_bridge")
    if isinstance(tts_bridge, dict):
        result["frontend"]["streaming_tts_bridge"] = {
            "configured": tts_bridge.get("configured") is True,
            "healthy": tts_bridge.get("healthy") is True,
            "model": str(tts_bridge.get("model") or "") or None,
        }
    if isinstance(vllm_frontend, dict):
        result["frontend"]["vllm_audio_frontend"] = {
            "configured": vllm_frontend.get("configured") is True,
            "healthy": vllm_frontend.get("healthy") is True,
            "model": str(vllm_frontend.get("model") or "") or None,
        }
    return result


def _realtime_voice_sidecar_capability_error(
    *,
    engine: str,
    health_payload: Optional[Dict[str, Any]],
    expected_kame_reflex_model: str = "",
) -> str:
    if not health_payload:
        return "sidecar_missing_capabilities"
    if health_payload.get("ok") is not True:
        return "sidecar_not_ready"

    capabilities = health_payload.get("capabilities") if isinstance(health_payload.get("capabilities"), dict) else {}
    if engine == "native_s2s_oracle":
        return "" if capabilities.get("native_s2s") is True else "sidecar_missing_native_s2s"
    if engine == "kame_interface_oracle":
        if capabilities.get("vllm_audio_frontend") is not True:
            return "sidecar_missing_kame_reflex"
        if not _realtime_voice_kame_reflex_model_matches(
            health_payload,
            expected_kame_reflex_model,
        ):
            return "sidecar_kame_reflex_model_mismatch"
        if capabilities.get("tts") is not True:
            return "sidecar_missing_tts"
        return ""

    has_stt = capabilities.get("utterance_stt") is True or capabilities.get("streaming_stt") is True
    if not has_stt:
        return "sidecar_missing_stt"
    if capabilities.get("tts") is not True:
        return "sidecar_missing_tts"
    return ""


def _realtime_voice_conversation_quality_payload(
    *,
    engine: str,
    base_url: str,
    health_payload: Optional[Dict[str, Any]],
    expected_kame_reflex_model: str = "",
) -> Dict[str, Any]:
    sidecar_ready = isinstance(health_payload, dict) and health_payload.get("ok") is True
    capabilities = health_payload.get("capabilities") if sidecar_ready else {}
    capabilities = capabilities if isinstance(capabilities, dict) else {}
    native_s2s = engine == "native_s2s_oracle" and capabilities.get("native_s2s") is True
    kame_reflex = (
        engine == "kame_interface_oracle"
        and capabilities.get("vllm_audio_frontend") is True
        and _realtime_voice_kame_reflex_model_matches(
            health_payload,
            expected_kame_reflex_model,
        )
    )
    streaming_stt = capabilities.get("streaming_stt") is True
    utterance_stt = capabilities.get("utterance_stt") is True
    tts = capabilities.get("tts") is True

    if native_s2s:
        mode = "native_s2s"
        reason = "native_s2s"
        live_like = True
    elif kame_reflex and tts:
        mode = "kame_reflex"
        reason = "audio_reflex_tts"
        live_like = True
    elif streaming_stt and tts:
        mode = "streaming_text"
        reason = "streaming_stt_tts"
        live_like = True
    elif utterance_stt and tts:
        mode = "turn_based_text"
        reason = "utterance_stt_tts"
        live_like = False
    elif base_url and isinstance(health_payload, dict):
        mode = "unready_sidecar"
        reason = "sidecar_not_ready"
        live_like = False
    elif base_url and health_payload:
        mode = "limited_sidecar"
        reason = "missing_streaming_stt_or_tts"
        live_like = False
    elif base_url:
        mode = "unverified_sidecar"
        reason = "sidecar_unverified"
        live_like = False
    else:
        mode = "local_turn_based"
        reason = "local_stt_tts"
        live_like = False

    return {
        "mode": mode,
        "reason": reason,
        "live_like": live_like,
        "partial_transcripts": native_s2s or streaming_stt,
        "barge_in": True,
        "native_s2s": native_s2s,
        "kame_reflex": kame_reflex,
        "streaming_stt": streaming_stt,
        "utterance_stt": utterance_stt,
        "tts": tts,
        "sidecar_verified": sidecar_ready,
    }


def _realtime_voice_kame_reflex_model_matches(
    health_payload: Optional[Mapping[str, Any]],
    expected_model: str = "",
) -> bool:
    expected = str(expected_model or "").strip()
    if not expected:
        return True
    if not isinstance(health_payload, Mapping):
        return False
    frontend = health_payload.get("frontend") if isinstance(health_payload.get("frontend"), Mapping) else {}
    vllm_frontend = frontend.get("vllm_audio_frontend") if isinstance(frontend, Mapping) else {}
    reported = ""
    if isinstance(vllm_frontend, Mapping):
        reported = str(vllm_frontend.get("model") or "").strip()
    if not reported and isinstance(frontend, Mapping):
        reported = str(frontend.get("model") or "").strip()
    if not reported:
        return False
    expected_keys = _realtime_voice_model_identity_keys(expected)
    reported_keys = _realtime_voice_model_identity_keys(reported)
    return bool(expected_keys.intersection(reported_keys))


def _realtime_voice_model_identity_keys(value: str) -> set[str]:
    normalized = str(value or "").strip().lower().replace("\\", "/")
    if not normalized:
        return set()
    compact = normalized.removesuffix("/")
    tail = compact.rsplit("/", 1)[-1]
    return {key for key in {compact, tail} if key}


def _realtime_voice_production_readiness_payload(
    *,
    enabled: bool,
    available: bool,
    unavailable_reason: str,
    language_support: Dict[str, Any],
    quality_targets_ms: Dict[str, int],
    conversation_quality: Dict[str, Any],
    evidence: Dict[str, Any],
    launch_review: Dict[str, Any],
) -> Dict[str, Any]:
    issues: List[str] = []
    if not enabled:
        issues.append("disabled")
    if not available and unavailable_reason:
        issues.append(unavailable_reason)
    if conversation_quality.get("live_like") is not True:
        issues.append("not_live_like")
    if conversation_quality.get("sidecar_verified") is not True:
        issues.append("sidecar_unverified")

    production_languages = language_support.get("production_languages")
    if not isinstance(production_languages, list):
        production_languages = []
    if not _realtime_voice_language_policy_has(production_languages, "en"):
        issues.append("missing_english")
    if not _realtime_voice_language_policy_has(production_languages, "ja"):
        issues.append("missing_japanese")

    production_scripts = language_support.get("production_scripts")
    if not isinstance(production_scripts, list):
        production_scripts = []
    if "Latn" not in production_scripts:
        issues.append("missing_latin_script")
    if "Jpan" not in production_scripts:
        issues.append("missing_japanese_script")

    if language_support.get("best_effort_languages") is not True:
        issues.append("best_effort_languages_disabled")

    required_quality_targets = (
        _REALTIME_VOICE_KAME_QUALITY_TARGETS_MS
        if conversation_quality.get("kame_reflex") is True
        else _REALTIME_VOICE_DEFAULT_QUALITY_TARGETS_MS
    )
    loose_targets = [
        key
        for key, default in required_quality_targets.items()
        if int(quality_targets_ms.get(key, default)) > default
    ]
    if loose_targets:
        issues.append("loose_quality_targets")

    evidence_issues = evidence.get("issues") if isinstance(evidence.get("issues"), list) else []
    issues.extend(str(issue) for issue in evidence_issues if str(issue or "").strip())

    issues = list(dict.fromkeys(issues))
    evidence_ready = available and not issues
    if evidence_ready:
        review_issues = (
            launch_review.get("issues")
            if isinstance(launch_review.get("issues"), list)
            else []
        )
        issues.extend(str(issue) for issue in review_issues if str(issue or "").strip())
        issues = list(dict.fromkeys(issues))
    ready = evidence_ready and launch_review.get("verified") is True and not issues
    if ready:
        level = "production_ready"
    elif evidence_ready:
        level = "evidence_ready"
    elif conversation_quality.get("live_like") is True:
        level = "live_like"
    else:
        level = "not_ready"
    return {
        "ready": ready,
        "level": level,
        "issues": issues,
        "evidence_ready": evidence_ready,
        "required_languages": list(_REALTIME_VOICE_DEFAULT_PRODUCTION_LANGUAGES),
        "required_scripts": list(_REALTIME_VOICE_DEFAULT_PRODUCTION_SCRIPTS),
        "required_quality_targets_ms": dict(required_quality_targets),
        "evidence": evidence,
        "launch_review": launch_review,
    }


def _realtime_voice_production_launch_review_payload(
    realtime: Dict[str, Any],
    *,
    required: bool,
    evidence: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    required_checks = dict(_REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS)
    raw_path = str(
        realtime.get("production_review_report")
        or realtime.get("production_launch_review_report")
        or ""
    ).strip()
    if not required:
        return {
            "required": False,
            "configured": bool(raw_path),
            "verified": False,
            "report_path": raw_path or None,
            "required_checks": required_checks,
            "issues": [],
        }
    if not raw_path:
        return {
            "required": True,
            "configured": False,
            "verified": False,
            "report_path": None,
            "required_checks": required_checks,
            "issues": ["missing_production_review_report"],
        }

    try:
        report = load_production_review_report(raw_path)
    except Exception as exc:
        return {
            "required": True,
            "configured": True,
            "verified": False,
            "report_path": raw_path,
            "required_checks": required_checks,
            "issues": [f"invalid_production_review_report:{sanitize_realtime_voice_error(exc)}"],
        }

    issues = validate_production_review_report(report, report_path=raw_path)
    reviewed_at = report.get("reviewed_at")
    reviewer = report.get("reviewer")
    raw_review_evidence = report.get("evidence")
    review_evidence = raw_review_evidence if isinstance(raw_review_evidence, Mapping) else {}
    reviewed_at_dt = _realtime_voice_parse_timestamp(reviewed_at)
    production_evidence = evidence if isinstance(evidence, Mapping) else {}
    newest_collected_at = str(production_evidence.get("newest_collected_at") or "").strip()
    newest_collected_dt = _realtime_voice_parse_timestamp(newest_collected_at)
    if newest_collected_dt is not None:
        if reviewed_at_dt is None:
            issues.append("reviewed_at_invalid")
        elif reviewed_at_dt < newest_collected_dt:
            issues.append("review_predates_evidence")
    return {
        "required": True,
        "configured": True,
        "verified": not issues,
        "report_path": raw_path,
        "reviewed_at": str(reviewed_at or "") or None,
        "reviewer": str(reviewer or "") or None,
        "required_checks": required_checks,
        "evidence": _realtime_voice_production_review_evidence_summary(review_evidence),
        "issues": issues,
    }


def _realtime_voice_production_review_evidence_summary(evidence: Mapping[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key in _REALTIME_VOICE_PRODUCTION_REVIEW_CHECKS:
        entry = evidence.get(key)
        if isinstance(entry, str):
            summary[key] = {
                "notes": bool(entry.strip()),
                "artifacts": 0,
            }
            continue
        if not isinstance(entry, Mapping):
            summary[key] = {
                "notes": False,
                "artifacts": 0,
            }
            continue
        artifacts = entry.get("artifacts")
        artifact_count = (
            sum(1 for item in artifacts if str(item or "").strip())
            if isinstance(artifacts, list)
            else 0
        )
        summary[key] = {
            "notes": bool(str(entry.get("notes") or "").strip()),
            "artifacts": artifact_count,
        }
    return summary


def _realtime_voice_production_evidence_payload(
    realtime: Dict[str, Any],
    *,
    current_manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    min_runs = _positive_int_config(realtime.get("production_evidence_min_runs"), default=3)
    max_age_days = _positive_int_config(realtime.get("production_evidence_max_age_days"), default=14)
    raw_path = str(
        realtime.get("production_evidence_report")
        or realtime.get("alpha_evidence_report")
        or ""
    ).strip()
    if not raw_path:
        return {
            "configured": False,
            "verified": False,
            "report_path": None,
            "runs": 0,
            "min_runs": min_runs,
            "max_age_days": max_age_days,
            "issues": ["missing_evidence_report"],
        }

    report_path = Path(raw_path).expanduser()
    try:
        from agent.realtime_voice_smoke_report import (
            RealtimeVoiceSmokeReportIssue,
            load_realtime_voice_smoke_report_runs,
            realtime_voice_alpha_manifest_fingerprint,
            summarize_realtime_voice_smoke_report_runs,
            validate_realtime_voice_alpha_report_runs,
        )

        runs = load_realtime_voice_smoke_report_runs(report_path)
        issues = validate_realtime_voice_alpha_report_runs(
            runs,
            min_runs=min_runs,
            max_collected_age_days=max_age_days,
            allow_loopback_validation=False,
        )
        if current_manifest is not None:
            evidence_manifest = _realtime_voice_first_evidence_manifest(runs)
            if evidence_manifest is not None and (
                realtime_voice_alpha_manifest_fingerprint(evidence_manifest)
                != realtime_voice_alpha_manifest_fingerprint(current_manifest)
            ):
                issues.append(
                    RealtimeVoiceSmokeReportIssue(
                        "evidence",
                        "current realtime stack does not match evidence manifest",
                        "current_stack",
                    )
                )
        summary = summarize_realtime_voice_smoke_report_runs(runs)
        newest_collected_at = _realtime_voice_newest_evidence_collected_at(runs)
    except Exception as exc:
        return {
            "configured": True,
            "verified": False,
            "report_path": raw_path,
            "min_runs": min_runs,
            "max_age_days": max_age_days,
            "issues": [f"invalid_evidence_report:{sanitize_realtime_voice_error(exc)}"],
        }

    formatted_issues = [f"evidence_report:{issue.format()}" for issue in issues]
    return {
        "configured": True,
        "verified": not formatted_issues,
        "report_path": raw_path,
        "runs": len(runs),
        "min_runs": min_runs,
        "max_age_days": max_age_days,
        "entries": sum(len(entries) for _label, entries in runs),
        "newest_collected_at": newest_collected_at,
        "summary": summary,
        "issues": formatted_issues,
    }


def _realtime_voice_newest_evidence_collected_at(
    runs: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> Optional[str]:
    newest_dt: Optional[datetime] = None
    newest_value = ""
    for _label, entries in runs:
        manifest = _realtime_voice_first_manifest_entry(entries)
        if manifest is None:
            continue
        value = str(manifest.get("collected_at") or "").strip()
        parsed = _realtime_voice_parse_timestamp(value)
        if parsed is None:
            continue
        if newest_dt is None or parsed > newest_dt:
            newest_dt = parsed
            newest_value = value
    return newest_value or None


def _realtime_voice_first_manifest_entry(
    entries: Sequence[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    for entry in entries:
        if str(entry.get("kind") or "").strip() == "manifest":
            return entry
    return None


def _realtime_voice_parse_timestamp(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _realtime_voice_current_evidence_manifest(
    *,
    engine: str,
    frontend_provider: str,
    frontend_model: str,
    interface_base_url: str = "",
    interface_temperature: Optional[float] = None,
    interface_max_output_tokens: Optional[int] = None,
    interface_timeout_seconds: Optional[float] = None,
    interface_max_audio_seconds: Optional[float] = None,
    interface_audio_input: str = "",
    asr_mode: str = "",
    asr_provider: str = "",
    asr_model: str = "",
    asr_base_url: str = "",
    oracle_provider: str = "",
    oracle_provider_name: str = "",
    preferred_local_oracle_model: str = "",
    oracle_base_url: str = "",
    oracle_api_mode: str = "",
    max_spoken_sentences: int = 2,
    voice_response_policy: str = "sentence_cap",
    tts_provider: str = "",
    tts_model: str = "",
    tts_voice: str = "",
    tts_base_url: str = "",
    fallback_policy: str = "",
    routing_policy: Optional[Mapping[str, Any]] = None,
    metrics_policy: Optional[Mapping[str, Any]] = None,
    sidecar_mode: str = "",
    health_payload: Optional[Mapping[str, Any]] = None,
    conversation_quality: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    conversation_quality = conversation_quality if isinstance(conversation_quality, Mapping) else {}
    return {
        "kind": "manifest",
        "engine": str(engine or ""),
        "frontend_provider": str(frontend_provider or ""),
        "frontend_model": str(frontend_model or ""),
        "interface_base_url": _redact_realtime_voice_url(str(interface_base_url or "")) if interface_base_url else "",
        "interface_temperature": interface_temperature,
        "interface_max_output_tokens": interface_max_output_tokens,
        "interface_timeout_seconds": interface_timeout_seconds,
        "interface_max_audio_seconds": interface_max_audio_seconds,
        "interface_audio_input": str(interface_audio_input or ""),
        "asr_mode": str(asr_mode or ""),
        "asr_provider": str(asr_provider or ""),
        "asr_model": str(asr_model or ""),
        "asr_base_url": _redact_realtime_voice_url(str(asr_base_url or "")) if asr_base_url else "",
        "oracle_provider": str(oracle_provider or ""),
        "oracle_provider_name": str(oracle_provider_name or ""),
        "preferred_local_oracle_model": str(preferred_local_oracle_model or ""),
        "oracle_base_url": _redact_realtime_voice_url(str(oracle_base_url or "")) if oracle_base_url else "",
        "oracle_api_mode": str(oracle_api_mode or ""),
        "max_spoken_sentences": int(max_spoken_sentences or 2),
        "voice_response_policy": str(voice_response_policy or "sentence_cap"),
        "tts_provider": str(tts_provider or ""),
        "tts_model": str(tts_model or ""),
        "tts_voice": str(tts_voice or ""),
        "tts_base_url": _redact_realtime_voice_url(str(tts_base_url or "")) if tts_base_url else "",
        "fallback_policy": str(fallback_policy or ""),
        "routing": dict(routing_policy) if isinstance(routing_policy, Mapping) else {},
        "metrics": dict(metrics_policy) if isinstance(metrics_policy, Mapping) else {},
        "conversation_quality": {
            "mode": str(conversation_quality.get("mode") or ""),
        },
        "sidecar": {
            "mode": str(sidecar_mode or "none"),
            "health": health_payload if isinstance(health_payload, Mapping) else {},
        },
    }


def _realtime_voice_first_evidence_manifest(
    runs: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> Optional[Mapping[str, Any]]:
    for _label, entries in runs:
        for entry in entries:
            if str(entry.get("kind") or "").strip() == "manifest":
                return entry
    return None


def _realtime_voice_language_policy_has(languages: Sequence[Any], target: str) -> bool:
    target = target.lower()
    for language in languages:
        primary = str(language or "").strip().lower().replace("_", "-").split("-", 1)[0]
        if primary == target:
            return True
    return False


def _realtime_voice_sidecar_command(realtime: Dict[str, Any]) -> List[str]:
    sidecar_host = str(realtime.get("sidecar_host") or "127.0.0.1")
    sidecar_port = str(int(realtime.get("sidecar_port") or 8765))
    provider = str(realtime.get("frontend_provider") or "").strip().lower()
    cmd = [
        sys.executable,
        "-m",
        "hermes_cli.realtime_voice_sidecar",
        "--host",
        sidecar_host,
        "--port",
        sidecar_port,
    ]

    interface_base_url = str(realtime.get("interface_base_url") or realtime.get("vllm_base_url") or "").strip()
    vllm_model = _realtime_voice_kame_vllm_model(realtime)
    if interface_base_url:
        cmd.extend(["--interface-base-url", interface_base_url])
        cmd.extend(["--vllm-base-url", interface_base_url])
    if vllm_model:
        cmd.extend(["--vllm-model", vllm_model])
    streaming_stt_base_url = str(realtime.get("streaming_stt_base_url") or "").strip()
    streaming_stt_model = str(realtime.get("streaming_stt_model") or "").strip()
    if streaming_stt_base_url:
        cmd.extend(["--streaming-stt-base-url", streaming_stt_base_url])
    if streaming_stt_model:
        cmd.extend(["--streaming-stt-model", streaming_stt_model])
    streaming_tts_base_url = str(realtime.get("streaming_tts_base_url") or "").strip()
    streaming_tts_model = str(realtime.get("streaming_tts_model") or "").strip()
    if streaming_tts_base_url:
        cmd.extend(["--streaming-tts-base-url", streaming_tts_base_url])
    if streaming_tts_model:
        cmd.extend(["--streaming-tts-model", streaming_tts_model])
    openai_realtime_base_url = str(realtime.get("openai_realtime_base_url") or "").strip()
    openai_realtime_model = str(
        (realtime.get("frontend_model") if provider in {"openai_realtime", "openai"} else None)
        or realtime.get("openai_realtime_model")
        or ""
    ).strip()
    openai_realtime_voice = str(realtime.get("openai_realtime_voice") or "").strip()
    openai_realtime_transcription_model = str(realtime.get("openai_realtime_transcription_model") or "").strip()
    if openai_realtime_base_url:
        cmd.extend(["--openai-realtime-base-url", openai_realtime_base_url])
    if openai_realtime_model:
        cmd.extend(["--openai-realtime-model", openai_realtime_model])
    if openai_realtime_voice:
        cmd.extend(["--openai-realtime-voice", openai_realtime_voice])
    if openai_realtime_transcription_model:
        cmd.extend(["--openai-realtime-transcription-model", openai_realtime_transcription_model])
    gemini_live_base_url = str(realtime.get("gemini_live_base_url") or "").strip()
    gemini_live_model = str(
        (realtime.get("frontend_model") if provider in {"gemini_live", "gemini"} else None)
        or realtime.get("gemini_live_model")
        or ""
    ).strip()
    gemini_live_voice = str(realtime.get("gemini_live_voice") or "").strip()
    if gemini_live_base_url:
        cmd.extend(["--gemini-live-base-url", gemini_live_base_url])
    if gemini_live_model:
        cmd.extend(["--gemini-live-model", gemini_live_model])
    if gemini_live_voice:
        cmd.extend(["--gemini-live-voice", gemini_live_voice])
    if _truthy_config(realtime.get("gemini_live_google_search"), default=False):
        cmd.append("--gemini-live-google-search")
    if _truthy_config(realtime.get("gemini_live_oracle_tool"), default=True) is False:
        cmd.append("--disable-gemini-live-oracle-tool")
    input_languages = _realtime_voice_sidecar_metadata_arg(_realtime_voice_sidecar_input_languages(realtime))
    output_languages = _realtime_voice_sidecar_metadata_arg(_realtime_voice_sidecar_output_languages(realtime))
    scripts = _realtime_voice_sidecar_metadata_arg(_realtime_voice_sidecar_scripts(realtime))
    if input_languages:
        cmd.extend(["--input-languages", input_languages])
    if output_languages:
        cmd.extend(["--output-languages", output_languages])
    if scripts:
        cmd.extend(["--scripts", scripts])
    return cmd


def _realtime_voice_sidecar_input_languages(realtime: Dict[str, Any]) -> Any:
    return (
        realtime.get("input_languages")
        or realtime.get("languages")
        or realtime.get("production_languages")
        or realtime.get("target_languages")
    )


def _realtime_voice_sidecar_output_languages(realtime: Dict[str, Any]) -> Any:
    return (
        realtime.get("output_languages")
        or realtime.get("tts_languages")
        or realtime.get("languages")
        or realtime.get("production_languages")
        or realtime.get("target_languages")
    )


def _realtime_voice_sidecar_scripts(realtime: Dict[str, Any]) -> Any:
    return realtime.get("scripts") or realtime.get("production_scripts") or realtime.get("target_scripts")


def _realtime_voice_sidecar_metadata_arg(value: Any) -> str:
    return ",".join(_sanitize_realtime_voice_health_metadata(value))


def _spawn_realtime_voice_sidecar(realtime: Dict[str, Any], env_on_disk: Dict[str, str]) -> subprocess.Popen:
    log_path = _ACTION_LOG_DIR / "realtime-voice-sidecar.log"
    _ACTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "ab", buffering=0)
    log_file.write(f"\n=== realtime voice sidecar started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode())

    child_env = {
        **os.environ,
        **env_on_disk,
        "HERMES_NONINTERACTIVE": "1",
    }
    provider = str(realtime.get("frontend_provider") or "").strip().lower()
    sidecar_token = _realtime_voice_sidecar_token(realtime, env_on_disk)
    if sidecar_token:
        child_env["HERMES_VOICE_SIDECAR_TOKEN"] = sidecar_token
    interface_base_url = str(realtime.get("interface_base_url") or realtime.get("vllm_base_url") or "").strip()
    vllm_model = _realtime_voice_kame_vllm_model(realtime)
    if interface_base_url:
        child_env["HERMES_KAME_INTERFACE_BASE_URL"] = interface_base_url
        child_env["HERMES_VOICE_VLLM_BASE_URL"] = interface_base_url
    if vllm_model:
        child_env["HERMES_VOICE_VLLM_MODEL"] = vllm_model
    interface_token_env = str(realtime.get("interface_api_key_env") or "HERMES_KAME_INTERFACE_API_KEY").strip()
    interface_token = _realtime_voice_env_value(
        env_on_disk,
        interface_token_env,
        "HERMES_KAME_INTERFACE_TOKEN",
        "HERMES_VOICE_VLLM_API_KEY",
        "HERMES_VOICE_VLLM_TOKEN",
    )
    if interface_token:
        child_env["HERMES_KAME_INTERFACE_API_KEY"] = interface_token
        child_env["HERMES_VOICE_VLLM_TOKEN"] = interface_token
    streaming_stt_base_url = str(realtime.get("streaming_stt_base_url") or "").strip()
    streaming_stt_model = str(realtime.get("streaming_stt_model") or "").strip()
    streaming_stt_token_env = str(realtime.get("streaming_stt_token_env") or "HERMES_VOICE_STREAMING_STT_TOKEN")
    streaming_stt_token = str(env_on_disk.get(streaming_stt_token_env) or os.environ.get(streaming_stt_token_env) or "")
    if streaming_stt_base_url:
        child_env["HERMES_VOICE_STREAMING_STT_BASE_URL"] = streaming_stt_base_url
    if streaming_stt_model:
        child_env["HERMES_VOICE_STREAMING_STT_MODEL"] = streaming_stt_model
    if streaming_stt_token:
        child_env["HERMES_VOICE_STREAMING_STT_TOKEN"] = streaming_stt_token
    streaming_tts_base_url = str(realtime.get("streaming_tts_base_url") or "").strip()
    streaming_tts_model = str(realtime.get("streaming_tts_model") or "").strip()
    streaming_tts_token_env = str(realtime.get("streaming_tts_token_env") or "HERMES_STREAMING_TTS_BRIDGE_TOKEN")
    streaming_tts_token = _realtime_voice_env_value(
        env_on_disk,
        streaming_tts_token_env,
        "HERMES_VOICE_STREAMING_TTS_TOKEN",
        "HERMES_STREAMING_STT_BRIDGE_TOKEN",
    )
    if streaming_tts_base_url:
        child_env["HERMES_VOICE_STREAMING_TTS_BASE_URL"] = streaming_tts_base_url
    if streaming_tts_model:
        child_env["HERMES_VOICE_STREAMING_TTS_MODEL"] = streaming_tts_model
    if streaming_tts_token:
        child_env["HERMES_VOICE_STREAMING_TTS_TOKEN"] = streaming_tts_token
    openai_realtime_api_key_env = str(realtime.get("openai_realtime_api_key_env") or "OPENAI_API_KEY")
    openai_realtime_api_key = str(
        env_on_disk.get(openai_realtime_api_key_env)
        or os.environ.get(openai_realtime_api_key_env)
        or ""
    )
    openai_realtime_base_url = str(realtime.get("openai_realtime_base_url") or "").strip()
    openai_realtime_model = str(
        (realtime.get("frontend_model") if provider in {"openai_realtime", "openai"} else None)
        or realtime.get("openai_realtime_model")
        or ""
    ).strip()
    openai_realtime_voice = str(realtime.get("openai_realtime_voice") or "").strip()
    openai_realtime_transcription_model = str(realtime.get("openai_realtime_transcription_model") or "").strip()
    if openai_realtime_api_key:
        child_env["HERMES_OPENAI_REALTIME_API_KEY"] = openai_realtime_api_key
    if openai_realtime_base_url:
        child_env["HERMES_OPENAI_REALTIME_BASE_URL"] = openai_realtime_base_url
    if openai_realtime_model:
        child_env["HERMES_OPENAI_REALTIME_MODEL"] = openai_realtime_model
    if openai_realtime_voice:
        child_env["HERMES_OPENAI_REALTIME_VOICE"] = openai_realtime_voice
    if openai_realtime_transcription_model:
        child_env["HERMES_OPENAI_REALTIME_TRANSCRIPTION_MODEL"] = openai_realtime_transcription_model
    gemini_live_api_key_env = str(realtime.get("gemini_live_api_key_env") or "GEMINI_API_KEY")
    gemini_live_api_key = str(
        env_on_disk.get(gemini_live_api_key_env)
        or os.environ.get(gemini_live_api_key_env)
        or ""
    )
    gemini_live_base_url = str(realtime.get("gemini_live_base_url") or "").strip()
    gemini_live_model = str(
        (realtime.get("frontend_model") if provider in {"gemini_live", "gemini"} else None)
        or realtime.get("gemini_live_model")
        or ""
    ).strip()
    gemini_live_voice = str(realtime.get("gemini_live_voice") or "").strip()
    if gemini_live_api_key:
        child_env["HERMES_GEMINI_LIVE_API_KEY"] = gemini_live_api_key
    if gemini_live_base_url:
        child_env["HERMES_GEMINI_LIVE_BASE_URL"] = gemini_live_base_url
    if gemini_live_model:
        child_env["HERMES_GEMINI_LIVE_MODEL"] = gemini_live_model
    if gemini_live_voice:
        child_env["HERMES_GEMINI_LIVE_VOICE"] = gemini_live_voice
    if _truthy_config(realtime.get("gemini_live_google_search"), default=False):
        child_env["HERMES_GEMINI_LIVE_GOOGLE_SEARCH"] = "true"
    if _truthy_config(realtime.get("gemini_live_oracle_tool"), default=True) is False:
        child_env["HERMES_GEMINI_LIVE_ORACLE_TOOL"] = "false"
    input_languages = _realtime_voice_sidecar_metadata_arg(_realtime_voice_sidecar_input_languages(realtime))
    output_languages = _realtime_voice_sidecar_metadata_arg(_realtime_voice_sidecar_output_languages(realtime))
    scripts = _realtime_voice_sidecar_metadata_arg(_realtime_voice_sidecar_scripts(realtime))
    if input_languages:
        child_env["HERMES_VOICE_INPUT_LANGUAGES"] = input_languages
    if output_languages:
        child_env["HERMES_VOICE_OUTPUT_LANGUAGES"] = output_languages
    if scripts:
        child_env["HERMES_VOICE_SCRIPTS"] = scripts

    popen_kwargs: Dict[str, Any] = {
        "cwd": str(PROJECT_ROOT),
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
        return subprocess.Popen(_realtime_voice_sidecar_command(realtime), **popen_kwargs)
    finally:
        log_file.close()


def _realtime_voice_kame_vllm_model(realtime: Mapping[str, Any]) -> str:
    explicit_model = str(realtime.get("vllm_model") or "").strip()
    if explicit_model:
        return explicit_model
    provider = str(realtime.get("frontend_provider") or "").strip().lower()
    if provider not in {"gemma4", "vllm", "openai_compatible", "openai-compatible"}:
        return ""
    return str(realtime.get("frontend_model") or "").strip()


def _ensure_realtime_voice_sidecar(realtime: Dict[str, Any]) -> None:
    global _VOICE_SIDECAR_PROC

    base_url = _realtime_voice_sidecar_base_url(realtime)
    if not _realtime_voice_should_autostart_sidecar(realtime, base_url):
        return
    env_on_disk = load_env()
    sidecar_token = _realtime_voice_sidecar_token(realtime, env_on_disk)
    if _realtime_voice_sidecar_healthy(base_url, token=sidecar_token):
        return

    with _VOICE_SIDECAR_LOCK:
        if _realtime_voice_sidecar_healthy(base_url, token=sidecar_token):
            return
        if _VOICE_SIDECAR_PROC is not None and _VOICE_SIDECAR_PROC.poll() is None:
            deadline = time.monotonic() + _VOICE_SIDECAR_START_TIMEOUT
            while time.monotonic() < deadline:
                if _realtime_voice_sidecar_healthy(base_url, token=sidecar_token):
                    return
                time.sleep(0.2)
            raise RuntimeError(f"realtime voice sidecar is not healthy at {base_url}")

        _VOICE_SIDECAR_PROC = _spawn_realtime_voice_sidecar(realtime, env_on_disk)
        deadline = time.monotonic() + _VOICE_SIDECAR_START_TIMEOUT
        while time.monotonic() < deadline:
            if _realtime_voice_sidecar_healthy(base_url, token=sidecar_token):
                _log.info("realtime voice sidecar ready at %s", base_url)
                return
            if _VOICE_SIDECAR_PROC.poll() is not None:
                raise RuntimeError(
                    f"realtime voice sidecar exited with code {_VOICE_SIDECAR_PROC.returncode}; "
                    f"see {_ACTION_LOG_DIR / 'realtime-voice-sidecar.log'}"
                )
            time.sleep(0.2)
        raise RuntimeError(f"realtime voice sidecar did not become healthy at {base_url}")


def _realtime_voice_config_dict() -> Dict[str, Any]:
    cfg = load_config()
    voice = cfg.get("voice") if isinstance(cfg, dict) else {}
    realtime = voice.get("realtime") if isinstance(voice, dict) else {}
    return _normalize_realtime_voice_config(realtime if isinstance(realtime, dict) else {})


def _mapping_config(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _realtime_voice_interface_config(realtime: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping_config(realtime.get("interface") if isinstance(realtime, Mapping) else {})


def _realtime_voice_oracle_config(realtime: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping_config(realtime.get("oracle") if isinstance(realtime, Mapping) else {})


def _realtime_voice_barge_in_config(realtime: Mapping[str, Any]) -> Dict[str, Any]:
    return _mapping_config(realtime.get("barge_in") if isinstance(realtime, Mapping) else {})


def _first_realtime_voice_config_value(
    realtime: Mapping[str, Any],
    *paths: tuple[str, ...],
    default: Any = None,
) -> Any:
    if not isinstance(realtime, Mapping):
        return default
    for path in paths:
        current: Any = realtime
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                current = None
                break
            current = current.get(key)
        if current is not None:
            return current
    return default


def _set_realtime_voice_default(config: Dict[str, Any], key: str, value: Any) -> None:
    existing = config.get(key)
    if value is None or (existing is not None and not (isinstance(existing, str) and not existing.strip())):
        return
    if isinstance(value, str) and not value.strip():
        return
    config[key] = value


def _normalize_realtime_voice_config(realtime: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a flat runtime view of voice.realtime.

    The GUI and older profiles use flat keys such as ``frontend_model`` while
    the full KAME design documents nested ``interface``/``oracle``/``barge_in``
    sections.  Keep flat keys authoritative for compatibility, and fill missing
    flat runtime keys from the nested shape.
    """
    config = dict(realtime) if isinstance(realtime, Mapping) else {}
    interface = _realtime_voice_interface_config(config)
    oracle = _realtime_voice_oracle_config(config)
    asr = _mapping_config(config.get("asr"))
    tts = _mapping_config(config.get("tts"))
    barge_in = _realtime_voice_barge_in_config(config)

    _set_realtime_voice_default(config, "frontend_provider", interface.get("provider"))
    _set_realtime_voice_default(config, "frontend_model", interface.get("model"))
    _set_realtime_voice_default(config, "frontend_model", config.get("vllm_model"))
    _set_realtime_voice_default(config, "interface_base_url", interface.get("base_url"))
    _set_realtime_voice_default(config, "interface_api_key_env", interface.get("api_key_env"))
    _set_realtime_voice_default(config, "interface_base_url", config.get("vllm_base_url"))
    _set_realtime_voice_default(config, "vllm_base_url", config.get("interface_base_url"))
    _set_realtime_voice_default(config, "interface_temperature", interface.get("temperature"))
    _set_realtime_voice_default(config, "interface_max_output_tokens", interface.get("max_output_tokens"))
    _set_realtime_voice_default(config, "interface_max_audio_seconds", interface.get("max_audio_seconds"))
    if config.get("interface_timeout_seconds") is None:
        if interface.get("timeout_seconds") is not None:
            config["interface_timeout_seconds"] = interface.get("timeout_seconds")
        elif interface.get("timeout_ms") is not None:
            config["interface_timeout_seconds"] = _positive_float_config(interface.get("timeout_ms"), default=800.0) / 1000.0
    _set_realtime_voice_default(config, "interface_audio_input", interface.get("audio_input"))
    _set_realtime_voice_default(config, "asr_mode", interface.get("asr_mode"))

    _set_realtime_voice_default(config, "oracle_provider", oracle.get("provider"))
    _set_realtime_voice_default(config, "oracle_provider_name", oracle.get("provider_name"))
    _set_realtime_voice_default(config, "preferred_local_oracle_model", oracle.get("preferred_local_model"))
    _set_realtime_voice_default(config, "oracle_base_url", oracle.get("base_url"))
    _set_realtime_voice_default(config, "oracle_api_mode", oracle.get("api_mode"))
    _set_realtime_voice_default(config, "max_spoken_sentences", oracle.get("max_spoken_sentences"))
    _set_realtime_voice_default(config, "voice_response_policy", oracle.get("voice_response_policy") or oracle.get("response_policy"))
    if config.get("oracle_timeout_seconds") is None:
        if oracle.get("timeout_seconds") is not None:
            config["oracle_timeout_seconds"] = oracle.get("timeout_seconds")
        elif oracle.get("timeout_ms") is not None:
            config["oracle_timeout_seconds"] = _positive_float_config(oracle.get("timeout_ms"), default=60000.0) / 1000.0

    _set_realtime_voice_default(config, "asr_provider", asr.get("provider"))
    _set_realtime_voice_default(config, "asr_model", asr.get("model"))
    _set_realtime_voice_default(config, "asr_model", config.get("streaming_stt_model"))
    _set_realtime_voice_default(config, "asr_base_url", asr.get("base_url"))
    _set_realtime_voice_default(config, "asr_base_url", config.get("streaming_stt_base_url"))
    _set_realtime_voice_default(config, "streaming_stt_base_url", config.get("asr_base_url"))
    _set_realtime_voice_default(config, "streaming_stt_base_url", asr.get("base_url"))
    _set_realtime_voice_default(config, "streaming_stt_token_env", asr.get("token_env"))

    _set_realtime_voice_default(config, "tts_provider", tts.get("provider"))
    _set_realtime_voice_default(config, "tts_model", tts.get("model"))
    _set_realtime_voice_default(config, "tts_model", config.get("streaming_tts_model"))
    _set_realtime_voice_default(config, "tts_voice", tts.get("voice"))
    _set_realtime_voice_default(config, "tts_voice", config.get("streaming_tts_voice"))
    _set_realtime_voice_default(config, "tts_base_url", tts.get("base_url"))
    _set_realtime_voice_default(config, "tts_base_url", config.get("streaming_tts_base_url"))
    _set_realtime_voice_default(config, "streaming_tts_base_url", config.get("tts_base_url"))
    _set_realtime_voice_default(config, "streaming_tts_base_url", tts.get("base_url"))
    _set_realtime_voice_default(config, "streaming_tts_token_env", tts.get("token_env"))

    _set_realtime_voice_default(config, "barge_in_min_rms", barge_in.get("min_rms"))
    _set_realtime_voice_default(config, "barge_in_min_speech_ms", barge_in.get("min_speech_ms"))
    _set_realtime_voice_default(config, "barge_in_stop_playback_deadline_ms", barge_in.get("stop_playback_deadline_ms"))
    return config


def _realtime_voice_oracle_timeout_seconds(realtime: Mapping[str, Any]) -> float:
    if _first_realtime_voice_config_value(realtime, ("oracle_timeout_seconds",)) is not None:
        return _positive_float_config(realtime.get("oracle_timeout_seconds"), default=60.0)
    if _first_realtime_voice_config_value(realtime, ("oracle", "timeout_seconds")) is not None:
        return _positive_float_config(_first_realtime_voice_config_value(realtime, ("oracle", "timeout_seconds")), default=60.0)
    timeout_ms = _first_realtime_voice_config_value(realtime, ("oracle", "timeout_ms"))
    if timeout_ms is not None:
        parsed_ms = _positive_float_config(timeout_ms, default=60000.0)
        return parsed_ms / 1000.0
    return 60.0


def _realtime_voice_max_spoken_sentences(realtime: Mapping[str, Any]) -> int:
    return _positive_int_config(
        _first_realtime_voice_config_value(realtime, ("max_spoken_sentences",), ("oracle", "max_spoken_sentences")),
        default=2,
    )


def _realtime_voice_response_policy(realtime: Mapping[str, Any]) -> str:
    raw = _first_realtime_voice_config_value(
        realtime,
        ("voice_response_policy",),
        ("oracle", "voice_response_policy"),
        ("oracle", "response_policy"),
        default="sentence_cap",
    )
    value = str(raw or "sentence_cap").strip().lower().replace("-", "_")
    return value if value in {"sentence_cap", "brief_summary", "full"} else "sentence_cap"


def _realtime_voice_status_payload(*, probe_health: bool = True) -> Dict[str, Any]:
    realtime = _realtime_voice_config_dict()
    env_on_disk = load_env()
    enabled = realtime.get("enabled") is True
    engine = str(realtime.get("engine") or "text_oracle_tts")
    provider = str(_first_realtime_voice_config_value(realtime, ("frontend_provider",), ("interface", "provider"), default="") or "")
    frontend_model = str(_first_realtime_voice_config_value(realtime, ("frontend_model",), ("interface", "model"), default="") or "")
    interface_base_url = str(
        _first_realtime_voice_config_value(
            realtime,
            ("interface_base_url",),
            ("interface", "base_url"),
            ("vllm_base_url",),
            default="",
        )
        or ""
    )
    interface_temperature = _bounded_float_config(
        _first_realtime_voice_config_value(realtime, ("interface_temperature",), ("interface", "temperature")),
        default=0.2,
        minimum=0.0,
        maximum=2.0,
    )
    interface_max_output_tokens = _bounded_int_config(
        _first_realtime_voice_config_value(realtime, ("interface_max_output_tokens",), ("interface", "max_output_tokens")),
        default=160,
        minimum=1,
        maximum=4096,
    )
    interface_timeout_seconds = _positive_float_config(
        _first_realtime_voice_config_value(realtime, ("interface_timeout_seconds",), ("interface", "timeout_seconds")),
        default=(
            _positive_float_config(
                _first_realtime_voice_config_value(realtime, ("interface", "timeout_ms")),
                default=800.0,
            )
            / 1000.0
        ),
    )
    interface_max_audio_seconds = _bounded_float_config(
        _first_realtime_voice_config_value(
            realtime,
            ("interface_max_audio_seconds",),
            ("interface", "max_audio_seconds"),
        ),
        default=30.0,
        minimum=1.0,
        maximum=30.0,
    )
    interface_audio_input = str(
        _first_realtime_voice_config_value(realtime, ("interface_audio_input",), ("interface", "audio_input"), default="") or ""
    )
    interface_api_key_env = str(
        _first_realtime_voice_config_value(
            realtime,
            ("interface_api_key_env",),
            ("interface", "api_key_env"),
            default="HERMES_KAME_INTERFACE_API_KEY",
        )
        or "HERMES_KAME_INTERFACE_API_KEY"
    )
    interface_api_key_present = _realtime_voice_env_present_any(
        env_on_disk,
        interface_api_key_env,
        "HERMES_KAME_INTERFACE_TOKEN",
        "HERMES_VOICE_VLLM_API_KEY",
        "HERMES_VOICE_VLLM_TOKEN",
    )
    asr_mode = str(_first_realtime_voice_config_value(realtime, ("asr_mode",), ("interface", "asr_mode"), default="") or "")
    asr_provider = str(_first_realtime_voice_config_value(realtime, ("asr_provider",), ("asr", "provider"), default="") or "")
    asr_model = str(
        _first_realtime_voice_config_value(
            realtime,
            ("asr_model",),
            ("asr", "model"),
            ("streaming_stt_model",),
            default="",
        )
        or ""
    )
    asr_base_url = str(
        _first_realtime_voice_config_value(
            realtime,
            ("asr_base_url",),
            ("streaming_stt_base_url",),
            ("asr", "base_url"),
            default="",
        )
        or ""
    )
    preferred_local_oracle_model = str(
        _first_realtime_voice_config_value(
            realtime,
            ("preferred_local_oracle_model",),
            ("oracle", "preferred_local_model"),
            default="",
        )
        or ""
    )
    oracle_provider = str(_first_realtime_voice_config_value(realtime, ("oracle_provider",), ("oracle", "provider"), default="") or "")
    oracle_provider_name = str(
        _first_realtime_voice_config_value(realtime, ("oracle_provider_name",), ("oracle", "provider_name"), default="")
        or ""
    )
    oracle_base_url = str(_first_realtime_voice_config_value(realtime, ("oracle_base_url",), ("oracle", "base_url"), default="") or "")
    oracle_api_mode = str(
        _first_realtime_voice_config_value(realtime, ("oracle_api_mode",), ("oracle", "api_mode"), default="") or ""
    )
    oracle_timeout_seconds = _realtime_voice_oracle_timeout_seconds(realtime)
    max_spoken_sentences = _realtime_voice_max_spoken_sentences(realtime)
    voice_response_policy = _realtime_voice_response_policy(realtime)
    tts_provider = str(_first_realtime_voice_config_value(realtime, ("tts_provider",), ("tts", "provider"), default="") or "")
    tts_model = str(
        _first_realtime_voice_config_value(realtime, ("tts_model",), ("tts", "model"), ("streaming_tts_model",), default="") or ""
    )
    tts_voice = str(
        _first_realtime_voice_config_value(realtime, ("tts_voice",), ("tts", "voice"), ("streaming_tts_voice",), default="") or ""
    )
    tts_base_url = str(
        _first_realtime_voice_config_value(
            realtime,
            ("tts_base_url",),
            ("streaming_tts_base_url",),
            ("tts", "base_url"),
            default="",
        )
        or ""
    )
    fallback_policy = str(realtime.get("fallback_policy") or "legacy_voice")
    expected_kame_reflex_model = _realtime_voice_kame_vllm_model(realtime) or frontend_model
    require_live_like = _truthy_config(realtime.get("require_live_like"), default=False)
    language_support = _realtime_voice_language_support_payload(realtime)
    quality_targets_ms = _realtime_voice_quality_targets_payload(realtime)
    routing_policy = _realtime_voice_routing_policy_payload(realtime)
    metrics_policy = _realtime_voice_metrics_policy_payload(realtime)
    output_events_policy = _realtime_voice_output_events_payload(realtime)
    base_url = _realtime_voice_sidecar_base_url(realtime)
    connect_timeout_seconds = _positive_float_config(
        realtime.get("sidecar_connect_timeout_seconds"),
        default=10.0,
    )
    loopback = _realtime_voice_sidecar_is_loopback(base_url) if base_url else False
    autostart = _realtime_voice_should_autostart_sidecar(realtime, base_url)
    externally_managed = bool(base_url and not autostart)
    sidecar_token = _realtime_voice_sidecar_token(realtime) if probe_health and base_url else ""
    healthy: Optional[bool]
    health_payload: Optional[Dict[str, Any]]
    if probe_health and base_url:
        healthy, health_payload = _realtime_voice_sidecar_health_probe(base_url, token=sidecar_token)
    else:
        healthy = None
        health_payload = None
    sidecar_capability_error = (
        _realtime_voice_sidecar_capability_error(
            engine=engine,
            health_payload=health_payload,
            expected_kame_reflex_model=expected_kame_reflex_model,
        )
        if base_url and healthy is True
        else ""
    )
    conversation_quality = _realtime_voice_conversation_quality_payload(
        engine=engine,
        base_url=base_url,
        health_payload=health_payload,
        expected_kame_reflex_model=expected_kame_reflex_model,
    )
    sidecar_mode = _realtime_voice_sidecar_mode(realtime, base_url)

    current_evidence_manifest = _realtime_voice_current_evidence_manifest(
        engine=engine,
        frontend_provider=provider,
        frontend_model=frontend_model,
        interface_base_url=interface_base_url,
        interface_temperature=interface_temperature,
        interface_max_output_tokens=interface_max_output_tokens,
        interface_timeout_seconds=interface_timeout_seconds,
        interface_max_audio_seconds=interface_max_audio_seconds,
        interface_audio_input=interface_audio_input,
        asr_mode=asr_mode,
        asr_provider=asr_provider,
        asr_model=asr_model,
        asr_base_url=asr_base_url,
        oracle_provider=oracle_provider,
        oracle_provider_name=oracle_provider_name,
        preferred_local_oracle_model=preferred_local_oracle_model,
        oracle_base_url=oracle_base_url,
        oracle_api_mode=oracle_api_mode,
        max_spoken_sentences=max_spoken_sentences,
        voice_response_policy=voice_response_policy,
        tts_provider=tts_provider,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_base_url=tts_base_url,
        fallback_policy=fallback_policy,
        routing_policy=routing_policy,
        metrics_policy=metrics_policy,
        sidecar_mode=sidecar_mode,
        health_payload=health_payload,
        conversation_quality=conversation_quality,
    )
    kame_stack = _realtime_voice_kame_stack_payload(
        engine=engine,
        provider=provider,
        frontend_model=frontend_model,
        interface_base_url=interface_base_url,
        interface_timeout_seconds=interface_timeout_seconds,
        interface_max_audio_seconds=interface_max_audio_seconds,
        interface_audio_input=interface_audio_input,
        realtime=realtime,
        asr_mode=asr_mode,
        asr_provider=asr_provider,
        asr_model=asr_model,
        asr_base_url=asr_base_url,
        preferred_local_oracle_model=preferred_local_oracle_model,
        oracle_timeout_seconds=oracle_timeout_seconds,
        tts_provider=tts_provider,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_base_url=tts_base_url,
        fallback_policy=fallback_policy,
    )
    production_evidence = _realtime_voice_production_evidence_payload(
        realtime,
        current_manifest=current_evidence_manifest,
    )
    launch_review = _realtime_voice_production_launch_review_payload(
        realtime,
        required=production_evidence.get("verified") is True,
        evidence=production_evidence,
    )

    unavailable_reason = ""
    available = enabled
    if not enabled:
        unavailable_reason = "disabled"
    elif engine == "native_s2s_oracle" and not base_url:
        available = False
        unavailable_reason = "sidecar_required"
    elif base_url and not autostart and healthy is False:
        available = False
        unavailable_reason = "sidecar_unhealthy"
    elif sidecar_capability_error:
        available = False
        unavailable_reason = sidecar_capability_error
    elif require_live_like and conversation_quality.get("live_like") is not True:
        available = False
        unavailable_reason = "live_like_required"

    production_readiness = _realtime_voice_production_readiness_payload(
        enabled=enabled,
        available=available,
        unavailable_reason=unavailable_reason,
        language_support=language_support,
        quality_targets_ms=quality_targets_ms,
        conversation_quality=conversation_quality,
        evidence=production_evidence,
        launch_review=launch_review,
    )

    return {
        "enabled": enabled,
        "available": available,
        "unavailable_reason": unavailable_reason or None,
        "engine": engine,
        "input_codec": str(realtime.get("input_codec") or "webm_opus"),
        "output_codec": str(realtime.get("output_codec") or "opus"),
        "input_buffer_limit_bytes": _positive_int_config(
            realtime.get("input_buffer_limit_bytes"),
            default=8 * 1024 * 1024,
        ),
        "input_frame_ms": _bounded_int_config(
            realtime.get("input_frame_ms"),
            default=100,
            minimum=40,
            maximum=500,
        ),
        "silence_timeout_ms": _bounded_int_config(
            realtime.get("silence_timeout_ms"),
            default=650,
            minimum=250,
            maximum=2000,
        ),
        "speech_level_threshold": _bounded_float_config(
            realtime.get("speech_level_threshold"),
            default=0.075,
            minimum=0.005,
            maximum=1.0,
        ),
        "barge_in_min_speech_ms": _bounded_int_config(
            realtime.get("barge_in_min_speech_ms"),
            default=120,
            minimum=40,
            maximum=1000,
        ),
        "barge_in_min_rms": _bounded_int_config(
            realtime.get("barge_in_min_rms"),
            default=350,
            minimum=0,
            maximum=32767,
        ),
        "barge_in_stop_playback_deadline_ms": _bounded_int_config(
            realtime.get("barge_in_stop_playback_deadline_ms"),
            default=150,
            minimum=20,
            maximum=1000,
        ),
        "pre_roll_ms": _bounded_nonnegative_int_config(
            realtime.get("pre_roll_ms"),
            default=300,
            minimum=0,
            maximum=1000,
        ),
        "voice_architecture": "kame_frontend_oracle",
        "frontend_role": "low_latency_voice_interface",
        "oracle_role": "hermes_backend_oracle",
        "frontend_provider": provider or None,
        "frontend_model": frontend_model or None,
        "interface_base_url": _redact_realtime_voice_url(interface_base_url) if interface_base_url else None,
        "interface_temperature": interface_temperature,
        "interface_max_output_tokens": interface_max_output_tokens,
        "interface_timeout_seconds": interface_timeout_seconds,
        "interface_max_audio_seconds": interface_max_audio_seconds,
        "interface_api_key_env": interface_api_key_env,
        "interface_api_key_present": interface_api_key_present,
        "interface_audio_input": interface_audio_input or None,
        "asr_mode": asr_mode or None,
        "asr_provider": asr_provider or None,
        "asr_model": asr_model or None,
        "asr_base_url": _redact_realtime_voice_url(asr_base_url) if asr_base_url else None,
        "oracle_provider": oracle_provider or None,
        "oracle_provider_name": oracle_provider_name or None,
        "preferred_local_oracle_model": preferred_local_oracle_model or None,
        "oracle_base_url": _redact_realtime_voice_url(oracle_base_url) if oracle_base_url else None,
        "oracle_api_mode": oracle_api_mode or None,
        "oracle_timeout_seconds": oracle_timeout_seconds,
        "max_spoken_sentences": max_spoken_sentences,
        "voice_response_policy": voice_response_policy,
        "tts_provider": tts_provider or None,
        "tts_model": tts_model or None,
        "tts_voice": tts_voice or None,
        "tts_base_url": _redact_realtime_voice_url(tts_base_url) if tts_base_url else None,
        "fallback_policy": fallback_policy,
        "language_support": language_support,
        "quality_targets_ms": quality_targets_ms,
        "routing": routing_policy,
        "metrics": metrics_policy,
        "output_events": output_events_policy,
        "conversation_quality": conversation_quality,
        "production_readiness": production_readiness,
        "kame_stack": kame_stack,
        "require_live_like": require_live_like,
        "sidecar": {
            "mode": sidecar_mode,
            "base_url": _redact_realtime_voice_url(base_url),
            "health_url": _redact_realtime_voice_url(_realtime_voice_sidecar_health_url(base_url)) if base_url else "",
            "loopback": loopback,
            "autostart": autostart,
            "connect_timeout_seconds": connect_timeout_seconds,
            "healthy": healthy,
            "externally_managed": externally_managed,
            "health": health_payload,
        },
        "native_s2s": {
            "enabled": engine == "native_s2s_oracle",
            "sidecar_required": engine == "native_s2s_oracle",
        },
        "kame": {
            "enabled": engine == "kame_interface_oracle",
            "sidecar_required": engine == "kame_interface_oracle",
            "audio_reflex": conversation_quality.get("kame_reflex") is True,
            "interface_base_url": _redact_realtime_voice_url(interface_base_url) if interface_base_url else None,
            "interface_temperature": interface_temperature,
            "interface_max_output_tokens": interface_max_output_tokens,
            "interface_timeout_seconds": interface_timeout_seconds,
            "interface_max_audio_seconds": interface_max_audio_seconds,
            "interface_api_key_env": interface_api_key_env,
            "interface_api_key_present": interface_api_key_present,
            "asr_mode": asr_mode or None,
            "interface_audio_input": interface_audio_input or None,
            "asr_provider": asr_provider or None,
            "asr_model": asr_model or None,
            "asr_base_url": _redact_realtime_voice_url(asr_base_url) if asr_base_url else None,
            "oracle_provider": oracle_provider or None,
            "oracle_provider_name": oracle_provider_name or None,
            "preferred_local_oracle_model": preferred_local_oracle_model or None,
            "oracle_base_url": _redact_realtime_voice_url(oracle_base_url) if oracle_base_url else None,
            "oracle_api_mode": oracle_api_mode or None,
            "oracle_timeout_seconds": oracle_timeout_seconds,
            "max_spoken_sentences": max_spoken_sentences,
            "voice_response_policy": voice_response_policy,
            "tts_provider": tts_provider or None,
            "tts_model": tts_model or None,
            "tts_voice": tts_voice or None,
            "tts_base_url": _redact_realtime_voice_url(tts_base_url) if tts_base_url else None,
            "fallback_policy": fallback_policy,
            "routing": routing_policy,
            "metrics": metrics_policy,
            "stack": kame_stack,
        },
    }


def _realtime_voice_open_unavailable_reason(realtime: Dict[str, Any]) -> str:
    engine = str(realtime.get("engine") or "text_oracle_tts")
    base_url = _realtime_voice_sidecar_base_url(realtime)
    if engine == "native_s2s_oracle" and not base_url:
        return "sidecar_required"
    if not base_url:
        return "live_like_required" if _truthy_config(realtime.get("require_live_like"), default=False) else ""

    sidecar_token = _realtime_voice_sidecar_token(realtime)
    healthy, health_payload = _realtime_voice_sidecar_health_probe(base_url, token=sidecar_token)
    if not healthy:
        return "sidecar_unhealthy"
    expected_kame_reflex_model = _realtime_voice_kame_vllm_model(realtime) or str(
        _first_realtime_voice_config_value(
            realtime,
            ("frontend_model",),
            ("interface", "model"),
            default="",
        )
        or ""
    )
    capability_error = _realtime_voice_sidecar_capability_error(
        engine=engine,
        health_payload=health_payload,
        expected_kame_reflex_model=expected_kame_reflex_model,
    )
    if capability_error:
        return capability_error
    if _truthy_config(realtime.get("require_live_like"), default=False):
        conversation_quality = _realtime_voice_conversation_quality_payload(
            engine=engine,
            base_url=base_url,
            health_payload=health_payload,
            expected_kame_reflex_model=expected_kame_reflex_model,
        )
        if conversation_quality.get("live_like") is not True:
            return "live_like_required"
    return ""


def _realtime_voice_env_present(env_on_disk: Mapping[str, Any], key: str) -> bool:
    return bool(str(env_on_disk.get(key) or os.environ.get(key) or "").strip())


def _realtime_voice_env_value(
    env_on_disk: Mapping[str, Any],
    key: str,
    *fallback_keys: str,
) -> str:
    for candidate in (key, *fallback_keys):
        env_name = str(candidate or "").strip()
        if not env_name:
            continue
        value = str(env_on_disk.get(env_name) or os.environ.get(env_name) or "").strip()
        if value:
            return value
    return ""


def _realtime_voice_env_present_any(env_on_disk: Mapping[str, Any], key: str, *fallback_keys: str) -> bool:
    return bool(_realtime_voice_env_value(env_on_disk, key, *fallback_keys))


def _realtime_voice_provider_setup_rows(realtime: Mapping[str, Any], env_on_disk: Mapping[str, Any]) -> List[Dict[str, Any]]:
    openai_key_env = str(realtime.get("openai_realtime_api_key_env") or "OPENAI_API_KEY")
    gemini_key_env = str(realtime.get("gemini_live_api_key_env") or "GEMINI_API_KEY")
    interface_key_env = str(realtime.get("interface_api_key_env") or "HERMES_KAME_INTERFACE_API_KEY")
    stt_token_env = str(realtime.get("streaming_stt_token_env") or "HERMES_STREAMING_STT_BRIDGE_TOKEN")
    tts_token_env = str(realtime.get("streaming_tts_token_env") or "HERMES_STREAMING_TTS_BRIDGE_TOKEN")
    nemotron_upstream_env = "HERMES_NEMOTRON_SPEECH_UPSTREAM_BASE_URL"
    magpie_upstream_env = "HERMES_MAGPIE_TTS_UPSTREAM_BASE_URL"
    kame_model = (
        str(realtime.get("frontend_model") or "")
        if str(realtime.get("frontend_provider") or "").strip() == "gemma4"
        else ""
    )
    interface_base_url = str(realtime.get("interface_base_url") or realtime.get("vllm_base_url") or "")
    return [
        {
            "id": "kame",
            "provider": "gemma4",
            "label": "KAME Gemma reflex",
            "kind": "kame_interface_oracle",
            "model": kame_model or "gemma-4-E2B-it",
            "base_url": _redact_realtime_voice_url(interface_base_url),
            "api_key_env": interface_key_env,
            "api_key_present": _realtime_voice_env_present_any(
                env_on_disk,
                interface_key_env,
                "HERMES_KAME_INTERFACE_TOKEN",
                "HERMES_VOICE_VLLM_API_KEY",
                "HERMES_VOICE_VLLM_TOKEN",
            ),
            "voice": str(realtime.get("tts_voice") or realtime.get("streaming_tts_voice") or ""),
            "implemented": True,
        },
        {
            "id": "openai",
            "provider": "openai_realtime",
            "label": "OpenAI Realtime",
            "kind": "native_s2s",
            "model": str(realtime.get("frontend_model") or realtime.get("openai_realtime_model") or "gpt-realtime-2"),
            "voice": str(realtime.get("openai_realtime_voice") or "marin"),
            "api_key_env": openai_key_env,
            "api_key_present": _realtime_voice_env_present(env_on_disk, openai_key_env)
            or _realtime_voice_env_present(env_on_disk, "HERMES_OPENAI_REALTIME_API_KEY"),
            "implemented": True,
        },
        {
            "id": "gemini",
            "provider": "gemini_live",
            "label": "Gemini Live",
            "kind": "native_s2s",
            "model": str(realtime.get("frontend_model") or realtime.get("gemini_live_model") or "gemini-3.1-flash-live-preview"),
            "voice": str(realtime.get("gemini_live_voice") or "Puck"),
            "api_key_env": gemini_key_env,
            "api_key_present": _realtime_voice_env_present(env_on_disk, gemini_key_env)
            or _realtime_voice_env_present(env_on_disk, "HERMES_GEMINI_LIVE_API_KEY"),
            "implemented": True,
        },
        {
            "id": "elevenlabs",
            "provider": "reference",
            "label": "ElevenLabs bridge",
            "kind": "stt_tts_bridge",
            "model": str(realtime.get("streaming_tts_model") or "eleven_flash_v2_5"),
            "voice": str(realtime.get("streaming_tts_voice") or ""),
            "api_key_env": "ELEVENLABS_API_KEY",
            "api_key_present": _realtime_voice_env_present(env_on_disk, "ELEVENLABS_API_KEY"),
            "bridge_token_env": tts_token_env,
            "bridge_token_present": _realtime_voice_env_present_any(
                env_on_disk,
                tts_token_env,
                "HERMES_STREAMING_STT_BRIDGE_TOKEN",
            ),
            "implemented": True,
        },
        {
            "id": "deepgram",
            "provider": "reference",
            "label": "Deepgram bridge",
            "kind": "stt_tts_bridge",
            "model": str(realtime.get("streaming_stt_model") or "nova-3"),
            "voice": str(realtime.get("streaming_tts_model") or "aura-2-thalia-en"),
            "api_key_env": "DEEPGRAM_API_KEY",
            "api_key_present": _realtime_voice_env_present(env_on_disk, "DEEPGRAM_API_KEY"),
            "bridge_token_env": stt_token_env,
            "bridge_token_present": _realtime_voice_env_present(env_on_disk, stt_token_env),
            "implemented": True,
        },
        {
            "id": "nemotron_speech",
            "provider": "reference",
            "label": "Nemotron Speech bridge",
            "kind": "local_streaming_asr_bridge",
            "model": str(realtime.get("asr_model") or realtime.get("streaming_stt_model") or "nemotron-speech-streaming-0.6b"),
            "base_url": str(realtime.get("streaming_stt_base_url") or ""),
            "upstream_base_url_env": nemotron_upstream_env,
            "upstream_base_url_present": _realtime_voice_env_present(env_on_disk, nemotron_upstream_env),
            "upstream_token_env": "HERMES_NEMOTRON_SPEECH_UPSTREAM_TOKEN",
            "upstream_token_present": _realtime_voice_env_present(env_on_disk, "HERMES_NEMOTRON_SPEECH_UPSTREAM_TOKEN"),
            "bridge_token_env": stt_token_env,
            "bridge_token_present": _realtime_voice_env_present(env_on_disk, stt_token_env),
            "implemented": True,
        },
        {
            "id": "magpie_tts",
            "provider": "reference",
            "label": "Magpie TTS bridge",
            "kind": "local_streaming_tts_bridge",
            "model": str(realtime.get("tts_model") or realtime.get("streaming_tts_model") or "magpie-local-streaming-tts"),
            "voice": str(realtime.get("tts_voice") or realtime.get("streaming_tts_voice") or ""),
            "base_url": str(realtime.get("streaming_tts_base_url") or ""),
            "upstream_base_url_env": magpie_upstream_env,
            "upstream_base_url_present": _realtime_voice_env_present(env_on_disk, magpie_upstream_env),
            "upstream_token_env": "HERMES_MAGPIE_TTS_UPSTREAM_TOKEN",
            "upstream_token_present": _realtime_voice_env_present(env_on_disk, "HERMES_MAGPIE_TTS_UPSTREAM_TOKEN"),
            "bridge_token_env": tts_token_env,
            "bridge_token_present": _realtime_voice_env_present_any(
                env_on_disk,
                tts_token_env,
                "HERMES_STREAMING_STT_BRIDGE_TOKEN",
            ),
            "implemented": True,
        },
        {
            "id": "cartesia",
            "provider": "reference",
            "label": "Cartesia bridge",
            "kind": "stt_tts_bridge",
            "model": str(realtime.get("streaming_tts_model") or "sonic-3.5"),
            "voice": str(realtime.get("streaming_tts_voice") or ""),
            "api_key_env": "CARTESIA_API_KEY",
            "api_key_present": _realtime_voice_env_present(env_on_disk, "CARTESIA_API_KEY"),
            "voice_id_env": "CARTESIA_VOICE_ID",
            "voice_id_present": _realtime_voice_env_present(env_on_disk, "CARTESIA_VOICE_ID"),
            "bridge_token_env": tts_token_env,
            "bridge_token_present": _realtime_voice_env_present_any(
                env_on_disk,
                tts_token_env,
                "HERMES_STREAMING_STT_BRIDGE_TOKEN",
            ),
            "implemented": True,
        },
    ]


def _realtime_voice_discord_setup(realtime: Mapping[str, Any], env_on_disk: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = load_config()
    discord_cfg = cfg.get("discord") if isinstance(cfg, dict) else {}
    discord_rt = discord_cfg.get("realtime_voice") if isinstance(discord_cfg, dict) else {}
    if not isinstance(discord_rt, dict):
        discord_rt = {}
    sidecar_base_url = str(
        discord_rt.get("sidecar_base_url")
        or realtime.get("sidecar_base_url")
        or (
            f"http://{realtime.get('sidecar_host') or '127.0.0.1'}:{int(realtime.get('sidecar_port') or 8765)}"
            if realtime.get("sidecar_autostart") is True
            else ""
        )
    ).strip()
    return {
        "enabled": bool(discord_rt.get("enabled")),
        "sidecar_base_url": sidecar_base_url,
        "bot_token_present": _realtime_voice_env_present(env_on_disk, "DISCORD_BOT_TOKEN"),
        "guild_id_present": _realtime_voice_env_present(env_on_disk, "DISCORD_GUILD_ID"),
        "voice_channel_id_present": _realtime_voice_env_present(env_on_disk, "DISCORD_VOICE_CHANNEL_ID"),
        "voice_channel_name_present": _realtime_voice_env_present(env_on_disk, "DISCORD_VOICE_CHANNEL_NAME"),
    }


def _realtime_voice_setup_config_payload(
    realtime: Mapping[str, Any],
    env_on_disk: Mapping[str, Any],
    status: Mapping[str, Any],
) -> Dict[str, Any]:
    interface_key_env = str(realtime.get("interface_api_key_env") or "HERMES_KAME_INTERFACE_API_KEY")
    stt_token_env = str(realtime.get("streaming_stt_token_env") or "HERMES_STREAMING_STT_BRIDGE_TOKEN")
    tts_token_env = str(realtime.get("streaming_tts_token_env") or "HERMES_STREAMING_TTS_BRIDGE_TOKEN")
    nemotron_upstream_env = "HERMES_NEMOTRON_SPEECH_UPSTREAM_BASE_URL"
    magpie_upstream_env = "HERMES_MAGPIE_TTS_UPSTREAM_BASE_URL"
    return {
        "engine": str(realtime.get("engine") or status.get("engine") or "text_oracle_tts"),
        "interface": {
            "provider": str(realtime.get("frontend_provider") or ""),
            "model": str(realtime.get("frontend_model") or ""),
            "base_url": _redact_realtime_voice_url(
                str(realtime.get("interface_base_url") or realtime.get("vllm_base_url") or "")
            ),
            "api_key_env": interface_key_env,
            "api_key_present": _realtime_voice_env_present_any(
                env_on_disk,
                interface_key_env,
                "HERMES_KAME_INTERFACE_TOKEN",
                "HERMES_VOICE_VLLM_API_KEY",
                "HERMES_VOICE_VLLM_TOKEN",
            ),
            "audio_input": str(realtime.get("interface_audio_input") or ""),
            "temperature": status.get("interface_temperature"),
            "max_output_tokens": status.get("interface_max_output_tokens"),
            "timeout_seconds": status.get("interface_timeout_seconds"),
            "max_audio_seconds": status.get("interface_max_audio_seconds"),
        },
        "oracle": {
            "provider": str(realtime.get("oracle_provider") or ""),
            "provider_name": str(realtime.get("oracle_provider_name") or ""),
            "preferred_local_model": str(realtime.get("preferred_local_oracle_model") or ""),
            "base_url": _redact_realtime_voice_url(str(realtime.get("oracle_base_url") or "")),
            "api_mode": str(realtime.get("oracle_api_mode") or ""),
            "timeout_seconds": status.get("oracle_timeout_seconds"),
            "max_spoken_sentences": status.get("max_spoken_sentences"),
            "voice_response_policy": status.get("voice_response_policy"),
        },
        "asr": {
            "mode": str(realtime.get("asr_mode") or ""),
            "provider": str(realtime.get("asr_provider") or ""),
            "model": str(realtime.get("asr_model") or realtime.get("streaming_stt_model") or ""),
            "bridge_base_url": _redact_realtime_voice_url(str(realtime.get("streaming_stt_base_url") or "")),
            "bridge_token_env": stt_token_env,
            "bridge_token_present": _realtime_voice_env_present(env_on_disk, stt_token_env),
            "local_provider": "nemotron_speech",
            "local_upstream_base_url_env": nemotron_upstream_env,
            "local_upstream_base_url_present": _realtime_voice_env_present(env_on_disk, nemotron_upstream_env),
            "local_upstream_token_env": "HERMES_NEMOTRON_SPEECH_UPSTREAM_TOKEN",
            "local_upstream_token_present": _realtime_voice_env_present(
                env_on_disk,
                "HERMES_NEMOTRON_SPEECH_UPSTREAM_TOKEN",
            ),
        },
        "tts": {
            "provider": str(realtime.get("tts_provider") or ""),
            "model": str(realtime.get("tts_model") or realtime.get("streaming_tts_model") or ""),
            "voice": str(realtime.get("tts_voice") or realtime.get("streaming_tts_voice") or ""),
            "bridge_base_url": _redact_realtime_voice_url(str(realtime.get("streaming_tts_base_url") or "")),
            "bridge_token_env": tts_token_env,
            "bridge_token_present": _realtime_voice_env_present(env_on_disk, tts_token_env),
            "local_provider": "magpie_tts",
            "local_upstream_base_url_env": magpie_upstream_env,
            "local_upstream_base_url_present": _realtime_voice_env_present(env_on_disk, magpie_upstream_env),
            "local_upstream_token_env": "HERMES_MAGPIE_TTS_UPSTREAM_TOKEN",
            "local_upstream_token_present": _realtime_voice_env_present(env_on_disk, "HERMES_MAGPIE_TTS_UPSTREAM_TOKEN"),
        },
        "barge_in": {
            "min_rms": status.get("barge_in_min_rms"),
            "min_speech_ms": status.get("barge_in_min_speech_ms"),
            "stop_playback_deadline_ms": status.get("barge_in_stop_playback_deadline_ms"),
        },
        "routing": dict(status.get("routing") if isinstance(status.get("routing"), Mapping) else {}),
        "metrics": dict(status.get("metrics") if isinstance(status.get("metrics"), Mapping) else {}),
        "fallback_policy": str(realtime.get("fallback_policy") or status.get("fallback_policy") or "legacy_voice"),
    }


def _realtime_voice_setup_payload(profile: Optional[str] = None) -> Dict[str, Any]:
    with _profile_scope(profile):
        cfg = load_config()
        env_on_disk = load_env()
        realtime = _realtime_voice_config_dict()
        status = _realtime_voice_status_payload()
        return {
            "status": status,
            "providers": _realtime_voice_provider_setup_rows(realtime, env_on_disk),
            "config": _realtime_voice_setup_config_payload(realtime, env_on_disk, status),
            "discord": _realtime_voice_discord_setup(realtime, env_on_disk),
            "config_path": str(get_config_path()),
            "env_path": str(get_env_path()),
            "active_provider": str(realtime.get("frontend_provider") or ""),
            "active_model": str(realtime.get("frontend_model") or ""),
            "enabled": bool(realtime.get("enabled")),
            "profile": profile or "",
        }


def _realtime_voice_profile_for_request(body: RealtimeVoiceProfileApply) -> Dict[str, Any]:
    from hermes_cli.realtime_voice_profile import (
        DEFAULT_DEEPGRAM_STT_MODEL,
        DEFAULT_DEEPGRAM_TTS_MODEL,
        DEFAULT_CARTESIA_STT_MODEL,
        DEFAULT_CARTESIA_TTS_MODEL,
        DEFAULT_ELEVENLABS_STT_MODEL,
        DEFAULT_ELEVENLABS_TTS_MODEL,
        DEFAULT_GEMINI_LIVE_MODEL,
        DEFAULT_GEMINI_LIVE_VOICE,
        DEFAULT_KAME_ORACLE_MODEL,
        DEFAULT_KAME_REFLEX_MODEL,
        DEFAULT_NVIDIA_SPEECH_STT_BRIDGE_BASE_URL,
        DEFAULT_NVIDIA_SPEECH_STT_MODEL,
        DEFAULT_NVIDIA_SPEECH_TTS_BRIDGE_BASE_URL,
        DEFAULT_NVIDIA_SPEECH_TTS_MODEL,
        DEFAULT_OPENAI_REALTIME_MODEL,
        DEFAULT_OPENAI_REALTIME_TRANSCRIPTION_MODEL,
        DEFAULT_OPENAI_REALTIME_VOICE,
        DEFAULT_STREAMING_STT_TOKEN_ENV,
        DEFAULT_STREAMING_TTS_TOKEN_ENV,
        build_kame_realtime_voice_profile,
        build_gemini_live_voice_profile,
        build_openai_realtime_voice_profile,
        build_realtime_voice_live_like_profile,
    )

    preset = str(body.preset or "").strip().lower()
    requested_stt_base_url = body.asr_base_url or body.streaming_stt_base_url
    requested_tts_base_url = body.tts_base_url or body.streaming_tts_base_url
    if preset in {"openai", "openai_realtime"}:
        return build_openai_realtime_voice_profile(
            model=body.model or DEFAULT_OPENAI_REALTIME_MODEL,
            voice=body.voice or DEFAULT_OPENAI_REALTIME_VOICE,
            transcription_model=DEFAULT_OPENAI_REALTIME_TRANSCRIPTION_MODEL,
            api_key_env=body.api_key_env or "OPENAI_API_KEY",
        )
    if preset in {"gemini", "gemini_live"}:
        return build_gemini_live_voice_profile(
            model=body.model or DEFAULT_GEMINI_LIVE_MODEL,
            voice=body.voice or DEFAULT_GEMINI_LIVE_VOICE,
            api_key_env=body.api_key_env or "GEMINI_API_KEY",
            google_search=bool(body.google_search),
            oracle_tool=bool(body.oracle_tool),
        )
    if preset in {"kame", "kame_interface_oracle"}:
        return build_kame_realtime_voice_profile(
            reflex_provider=body.interface_provider or "gemma4",
            reflex_model=body.model or DEFAULT_KAME_REFLEX_MODEL,
            vllm_model=body.vllm_model,
            interface_api_key_env=body.interface_api_key_env or "HERMES_KAME_INTERFACE_API_KEY",
            interface_base_url=body.interface_base_url,
            interface_temperature=body.interface_temperature,
            interface_max_output_tokens=body.interface_max_output_tokens,
            interface_timeout_seconds=body.interface_timeout_seconds,
            interface_audio_input=body.interface_audio_input or "auto",
            interface_max_audio_seconds=body.interface_max_audio_seconds,
            asr_mode=body.asr_mode or "on_escalation",
            asr_provider=body.asr_provider or "streaming_stt",
            preferred_local_oracle_model=body.preferred_local_oracle_model or DEFAULT_KAME_ORACLE_MODEL,
            oracle_provider=body.oracle_provider,
            oracle_api_mode=body.oracle_api_mode,
            oracle_timeout_seconds=body.oracle_timeout_seconds,
            max_spoken_sentences=body.max_spoken_sentences,
            voice_response_policy=body.voice_response_policy or "sentence_cap",
            tts_provider=body.tts_provider or "streaming_tts",
            tts_model=body.tts_model,
            tts_voice=body.tts_voice,
            fallback_policy=body.fallback_policy or "legacy_voice",
            oracle_base_url=body.oracle_base_url,
            oracle_provider_name=body.oracle_provider_name,
            streaming_stt_base_url=requested_stt_base_url,
            streaming_tts_base_url=requested_tts_base_url,
            streaming_stt_model=body.streaming_stt_model or "portable-streaming-asr",
            streaming_tts_model=body.streaming_tts_model or "portable-streaming-voice",
            streaming_tts_voice=body.streaming_tts_voice or body.voice,
            streaming_stt_token_env=body.streaming_stt_token_env or DEFAULT_STREAMING_STT_TOKEN_ENV,
            streaming_tts_token_env=body.streaming_tts_token_env or DEFAULT_STREAMING_TTS_TOKEN_ENV,
            allow_local_greetings=body.allow_local_greetings,
            allow_local_clarifications=body.allow_local_clarifications,
            require_oracle_for_tools=body.require_oracle_for_tools,
            require_oracle_for_memory=body.require_oracle_for_memory,
            require_oracle_for_files=body.require_oracle_for_files,
            local_confidence_threshold=body.local_confidence_threshold,
            barge_in_min_rms=body.barge_in_min_rms,
            barge_in_min_speech_ms=body.barge_in_min_speech_ms,
            barge_in_stop_playback_deadline_ms=body.barge_in_stop_playback_deadline_ms,
            metrics_enabled=body.metrics_enabled,
            metrics_log_turn_spans=body.metrics_log_turn_spans,
            metrics_log_provider_spans=body.metrics_log_provider_spans,
        )
    if preset == "elevenlabs":
        return build_realtime_voice_live_like_profile(
            streaming_stt_base_url=requested_stt_base_url or "http://127.0.0.1:8767",
            streaming_tts_base_url=requested_tts_base_url or requested_stt_base_url or "http://127.0.0.1:8767",
            streaming_stt_model=body.streaming_stt_model or DEFAULT_ELEVENLABS_STT_MODEL,
            streaming_tts_model=body.streaming_tts_model or DEFAULT_ELEVENLABS_TTS_MODEL,
            streaming_tts_voice=body.streaming_tts_voice or body.voice,
            streaming_stt_token_env=body.streaming_stt_token_env or DEFAULT_STREAMING_STT_TOKEN_ENV,
            streaming_tts_token_env=body.streaming_tts_token_env or DEFAULT_STREAMING_TTS_TOKEN_ENV,
        )
    if preset == "deepgram":
        return build_realtime_voice_live_like_profile(
            streaming_stt_base_url=requested_stt_base_url or "http://127.0.0.1:8766",
            streaming_tts_base_url=requested_tts_base_url or requested_stt_base_url or "http://127.0.0.1:8766",
            streaming_stt_model=body.streaming_stt_model or DEFAULT_DEEPGRAM_STT_MODEL,
            streaming_tts_model=body.streaming_tts_model or DEFAULT_DEEPGRAM_TTS_MODEL,
            streaming_tts_voice=body.streaming_tts_voice or body.voice,
            streaming_stt_token_env=body.streaming_stt_token_env or DEFAULT_STREAMING_STT_TOKEN_ENV,
            streaming_tts_token_env=body.streaming_tts_token_env or DEFAULT_STREAMING_TTS_TOKEN_ENV,
        )
    if preset == "cartesia":
        return build_realtime_voice_live_like_profile(
            streaming_stt_base_url=requested_stt_base_url or "http://127.0.0.1:8769",
            streaming_tts_base_url=requested_tts_base_url or requested_stt_base_url or "http://127.0.0.1:8769",
            streaming_stt_model=body.streaming_stt_model or DEFAULT_CARTESIA_STT_MODEL,
            streaming_tts_model=body.streaming_tts_model or body.model or DEFAULT_CARTESIA_TTS_MODEL,
            streaming_tts_voice=body.streaming_tts_voice or body.voice,
            streaming_stt_token_env=body.streaming_stt_token_env or DEFAULT_STREAMING_STT_TOKEN_ENV,
            streaming_tts_token_env=body.streaming_tts_token_env or DEFAULT_STREAMING_TTS_TOKEN_ENV,
        )
    if preset == "nvidia_speech":
        return build_realtime_voice_live_like_profile(
            streaming_stt_base_url=requested_stt_base_url or DEFAULT_NVIDIA_SPEECH_STT_BRIDGE_BASE_URL,
            streaming_tts_base_url=requested_tts_base_url or DEFAULT_NVIDIA_SPEECH_TTS_BRIDGE_BASE_URL,
            streaming_stt_model=body.streaming_stt_model or DEFAULT_NVIDIA_SPEECH_STT_MODEL,
            streaming_tts_model=body.streaming_tts_model or body.model or DEFAULT_NVIDIA_SPEECH_TTS_MODEL,
            streaming_tts_voice=body.streaming_tts_voice or body.voice,
            streaming_stt_token_env=body.streaming_stt_token_env or DEFAULT_STREAMING_STT_TOKEN_ENV,
            streaming_tts_token_env=body.streaming_tts_token_env or DEFAULT_STREAMING_TTS_TOKEN_ENV,
        )
    if preset == "reference":
        return build_realtime_voice_live_like_profile(
            streaming_stt_base_url=requested_stt_base_url,
            streaming_tts_base_url=requested_tts_base_url,
            streaming_stt_model=body.streaming_stt_model or "portable-streaming-asr",
            streaming_tts_model=body.streaming_tts_model or "portable-streaming-voice",
            streaming_tts_voice=body.streaming_tts_voice or body.voice,
            streaming_stt_token_env=body.streaming_stt_token_env or DEFAULT_STREAMING_STT_TOKEN_ENV,
            streaming_tts_token_env=body.streaming_tts_token_env or DEFAULT_STREAMING_TTS_TOKEN_ENV,
        )
    raise HTTPException(status_code=400, detail=f"unknown realtime voice preset: {body.preset}")


def _apply_realtime_voice_profile_body(body: RealtimeVoiceProfileApply, profile: Optional[str]) -> Dict[str, Any]:
    from hermes_cli.realtime_voice_profile import merge_realtime_voice_profile

    target_profile = body.profile or profile
    profile_cfg = _realtime_voice_profile_for_request(body)
    with _profile_scope(target_profile):
        cfg = load_config()
        updated = merge_realtime_voice_profile(cfg, profile_cfg)
        if body.enable_discord:
            discord = updated.get("discord")
            if not isinstance(discord, dict):
                discord = {}
            realtime = updated.get("voice", {}).get("realtime", {}) if isinstance(updated.get("voice"), dict) else {}
            sidecar_base_url = (
                str(realtime.get("sidecar_base_url") or "").strip()
                or f"http://{realtime.get('sidecar_host') or '127.0.0.1'}:{int(realtime.get('sidecar_port') or 8765)}"
            )
            discord["realtime_voice"] = {
                **(discord.get("realtime_voice") if isinstance(discord.get("realtime_voice"), dict) else {}),
                "enabled": True,
                "sidecar_base_url": sidecar_base_url,
                "frontend_provider": realtime.get("frontend_provider") or "",
                "frontend_model": realtime.get("frontend_model") or "",
                "interface_api_key_env": realtime.get("interface_api_key_env") or "HERMES_KAME_INTERFACE_API_KEY",
                "interface_audio_input": realtime.get("interface_audio_input") or "",
                "asr_mode": realtime.get("asr_mode") or "on_escalation",
                "asr_provider": realtime.get("asr_provider") or "",
                "asr_model": realtime.get("asr_model") or "",
                "oracle_provider": realtime.get("oracle_provider") or "",
                "oracle_provider_name": realtime.get("oracle_provider_name") or "",
                "preferred_local_oracle_model": realtime.get("preferred_local_oracle_model") or "",
                "oracle_base_url": realtime.get("oracle_base_url") or "",
                "oracle_api_mode": realtime.get("oracle_api_mode") or "",
                "oracle_timeout_seconds": _positive_float_config(
                    realtime.get("oracle_timeout_seconds"),
                    default=60.0,
                ),
                "max_spoken_sentences": _positive_int_config(
                    realtime.get("max_spoken_sentences"),
                    default=2,
                ),
                "voice_response_policy": realtime.get("voice_response_policy") or "sentence_cap",
                "tts_provider": realtime.get("tts_provider") or "",
                "tts_model": realtime.get("tts_model") or "",
                "tts_voice": realtime.get("tts_voice") or "",
                "fallback_policy": realtime.get("fallback_policy") or "legacy_voice",
                "routing": dict(realtime.get("routing") if isinstance(realtime.get("routing"), dict) else {}),
                "metrics": dict(realtime.get("metrics") if isinstance(realtime.get("metrics"), dict) else {}),
            }
            updated["discord"] = discord
        save_config(updated)
        normalized = _normalize_config_for_web(load_config())
    return {
        "ok": True,
        "config": {k: v for k, v in normalized.items() if not k.startswith("_")},
        "setup": _realtime_voice_setup_payload(target_profile),
    }


def _realtime_voice_profile_home(profile: Optional[str]) -> str:
    if not profile:
        return str(get_hermes_home())
    try:
        from hermes_cli.profiles import resolve_profile_env

        return resolve_profile_env(profile)
    except Exception:
        return str(get_hermes_home())


def _realtime_voice_config_from_request(ws: WebSocket):
    """Build a realtime voice session config from profile config + query params."""
    from agent.realtime_voice import (
        RealtimeVoiceASRMode,
        RealtimeVoiceEngineKind,
        RealtimeVoiceSessionConfig,
        VoiceAudioCodec,
    )

    cfg = load_config()
    voice = cfg.get("voice") if isinstance(cfg, dict) else {}
    realtime = voice.get("realtime") if isinstance(voice, dict) else {}
    realtime = _normalize_realtime_voice_config(realtime if isinstance(realtime, dict) else {})

    session_id = (
        ws.query_params.get("session_id")
        or ws.query_params.get("session")
        or f"voice-{secrets.token_urlsafe(12)}"
    )
    engine = ws.query_params.get("engine") or realtime.get("engine") or "text_oracle_tts"
    input_codec = ws.query_params.get("input_codec") or realtime.get("input_codec") or "webm_opus"
    output_codec = ws.query_params.get("output_codec") or realtime.get("output_codec") or "opus"
    sidecar_base_url = _realtime_voice_sidecar_base_url(realtime)
    env = load_env()
    sidecar_token = _realtime_voice_sidecar_token(realtime, env)

    language_support = _realtime_voice_language_support_payload(realtime)
    quality_targets_ms = _realtime_voice_quality_targets_payload(realtime)
    routing_policy = _realtime_voice_routing_policy_payload(realtime)
    metrics_policy = _realtime_voice_metrics_policy_payload(realtime)
    output_events_policy = _realtime_voice_output_events_payload(realtime)
    oracle_jobs_policy = _realtime_voice_oracle_jobs_payload(realtime)
    oracle_tool_router_policy = _realtime_voice_oracle_tool_router_payload(realtime)
    turn_acknowledgement = _realtime_voice_turn_acknowledgement_payload(realtime)
    barge_in_policy = {
        "min_rms": _bounded_int_config(
            realtime.get("barge_in_min_rms"),
            default=350,
            minimum=0,
            maximum=32767,
        ),
        "min_speech_ms": _bounded_int_config(
            realtime.get("barge_in_min_speech_ms"),
            default=120,
            minimum=40,
            maximum=1000,
        ),
        "stop_playback_deadline_ms": _bounded_int_config(
            realtime.get("barge_in_stop_playback_deadline_ms"),
            default=150,
            minimum=20,
            maximum=1000,
        ),
    }
    interface_base_url = str(realtime.get("interface_base_url") or realtime.get("vllm_base_url") or "")
    asr_provider = str(realtime.get("asr_provider") or "")
    asr_model = str(realtime.get("asr_model") or realtime.get("streaming_stt_model") or "")
    asr_base_url = str(realtime.get("asr_base_url") or realtime.get("streaming_stt_base_url") or "")
    oracle_provider = str(realtime.get("oracle_provider") or "")
    oracle_provider_name = str(realtime.get("oracle_provider_name") or "")
    oracle_base_url = str(realtime.get("oracle_base_url") or "")
    oracle_api_mode = str(realtime.get("oracle_api_mode") or "")
    voice_response_policy = _realtime_voice_response_policy(realtime)
    tts_provider = str(realtime.get("tts_provider") or "")
    tts_model = str(realtime.get("tts_model") or realtime.get("streaming_tts_model") or "")
    tts_voice = str(realtime.get("tts_voice") or realtime.get("streaming_tts_voice") or "")
    tts_base_url = str(realtime.get("tts_base_url") or realtime.get("streaming_tts_base_url") or "")
    fallback_policy = str(realtime.get("fallback_policy") or "legacy_voice")
    require_live_like = _truthy_config(realtime.get("require_live_like"), default=False)
    evidence_configured = bool(
        str(
            realtime.get("production_evidence_report")
            or realtime.get("alpha_evidence_report")
            or ""
        ).strip()
    )
    health_payload = None
    if sidecar_base_url and (require_live_like or evidence_configured):
        health_payload = _realtime_voice_sidecar_health_payload(
            sidecar_base_url,
            token=sidecar_token,
        )
    conversation_quality = _realtime_voice_conversation_quality_payload(
        engine=str(engine),
        base_url=str(sidecar_base_url or ""),
        health_payload=health_payload,
        expected_kame_reflex_model=_realtime_voice_kame_vllm_model(realtime)
        or str(realtime.get("frontend_model") or ""),
    )
    current_evidence_manifest = _realtime_voice_current_evidence_manifest(
        engine=str(engine),
        frontend_provider=str(realtime.get("frontend_provider") or ""),
        frontend_model=str(realtime.get("frontend_model") or ""),
        interface_base_url=interface_base_url,
        interface_temperature=_bounded_float_config(
            realtime.get("interface_temperature"),
            default=0.2,
            minimum=0.0,
            maximum=2.0,
        ),
        interface_max_output_tokens=_bounded_int_config(
            realtime.get("interface_max_output_tokens"),
            default=160,
            minimum=1,
            maximum=4096,
        ),
        interface_timeout_seconds=_positive_float_config(
            realtime.get("interface_timeout_seconds"),
            default=0.8,
        ),
        interface_max_audio_seconds=_bounded_float_config(
            realtime.get("interface_max_audio_seconds"),
            default=30.0,
            minimum=1.0,
            maximum=30.0,
        ),
        interface_audio_input=str(realtime.get("interface_audio_input") or ""),
        asr_mode=str(realtime.get("asr_mode") or ""),
        asr_provider=asr_provider,
        asr_model=asr_model,
        asr_base_url=asr_base_url,
        oracle_provider=oracle_provider,
        oracle_provider_name=oracle_provider_name,
        preferred_local_oracle_model=str(realtime.get("preferred_local_oracle_model") or ""),
        oracle_base_url=oracle_base_url,
        oracle_api_mode=oracle_api_mode,
        max_spoken_sentences=_positive_int_config(realtime.get("max_spoken_sentences"), default=2),
        voice_response_policy=voice_response_policy,
        tts_provider=tts_provider,
        tts_model=tts_model,
        tts_voice=tts_voice,
        tts_base_url=tts_base_url,
        fallback_policy=fallback_policy,
        routing_policy=routing_policy,
        metrics_policy=metrics_policy,
        sidecar_mode=_realtime_voice_sidecar_mode(realtime, sidecar_base_url),
        health_payload=health_payload,
        conversation_quality=conversation_quality,
    )
    production_evidence = _realtime_voice_production_evidence_payload(
        realtime,
        current_manifest=current_evidence_manifest,
    )
    launch_review = _realtime_voice_production_launch_review_payload(
        realtime,
        required=production_evidence.get("verified") is True,
        evidence=production_evidence,
    )
    production_readiness = _realtime_voice_production_readiness_payload(
        enabled=realtime.get("enabled") is True,
        available=realtime.get("enabled") is True,
        unavailable_reason="",
        language_support=language_support,
        quality_targets_ms=quality_targets_ms,
        conversation_quality=conversation_quality,
        evidence=production_evidence,
        launch_review=launch_review,
    )

    return RealtimeVoiceSessionConfig(
        session_id=str(session_id),
        engine=RealtimeVoiceEngineKind(str(engine)),
        input_codec=VoiceAudioCodec(str(input_codec)),
        output_codec=VoiceAudioCodec(str(output_codec)),
        input_buffer_limit_bytes=_positive_int_config(
            realtime.get("input_buffer_limit_bytes"),
            default=8 * 1024 * 1024,
        ),
        frontend_provider=str(realtime.get("frontend_provider") or "") or None,
        frontend_model=str(realtime.get("frontend_model") or "") or None,
        interface_temperature=_bounded_float_config(
            realtime.get("interface_temperature"),
            default=0.2,
            minimum=0.0,
            maximum=2.0,
        ),
        interface_max_output_tokens=_bounded_int_config(
            realtime.get("interface_max_output_tokens"),
            default=160,
            minimum=1,
            maximum=4096,
        ),
        interface_timeout_seconds=_positive_float_config(
            realtime.get("interface_timeout_seconds"),
            default=0.8,
        ),
        interface_max_audio_seconds=_bounded_float_config(
            realtime.get("interface_max_audio_seconds"),
            default=30.0,
            minimum=1.0,
            maximum=30.0,
        ),
        interface_audio_input=str(realtime.get("interface_audio_input") or "") or None,
        interface_base_url=interface_base_url or None,
        asr_mode=RealtimeVoiceASRMode(str(realtime.get("asr_mode") or RealtimeVoiceASRMode.ON_ESCALATION.value)),
        asr_provider=asr_provider or None,
        asr_model=asr_model or None,
        asr_base_url=asr_base_url or None,
        preferred_local_oracle_model=str(realtime.get("preferred_local_oracle_model") or "") or None,
        oracle_timeout_seconds=_positive_float_config(
            realtime.get("oracle_timeout_seconds"),
            default=60.0,
        ),
        max_spoken_sentences=_positive_int_config(
            realtime.get("max_spoken_sentences"),
            default=2,
        ),
        voice_response_policy=voice_response_policy,
        tts_provider=tts_provider or None,
        tts_model=tts_model or None,
        tts_voice=tts_voice or None,
        tts_base_url=tts_base_url or None,
        fallback_policy=fallback_policy or None,
        sidecar_base_url=str(sidecar_base_url or "") or None,
        sidecar_token=str(sidecar_token or "") or None,
        sidecar_connect_timeout_seconds=_positive_float_config(
            realtime.get("sidecar_connect_timeout_seconds"),
            default=10.0,
        ),
        spark_base_url=str(sidecar_base_url or "") or None,
        spark_token=str(sidecar_token or "") or None,
        turn_acknowledgement=turn_acknowledgement,
        routing_policy=routing_policy,
        metrics_policy=metrics_policy,
        output_events=output_events_policy,
        quality_targets_ms=quality_targets_ms,
        oracle_jobs=oracle_jobs_policy,
        oracle_tool_router=oracle_tool_router_policy,
        barge_in_policy=barge_in_policy,
        metadata={
            "source": "desktop",
            "voice_architecture": "kame_frontend_oracle",
            "frontend_role": "low_latency_voice_interface",
            "oracle_role": "hermes_backend_oracle",
            "frontend_provider": str(realtime.get("frontend_provider") or "") or None,
            "frontend_model": str(realtime.get("frontend_model") or "") or None,
            "interface_base_url": _redact_realtime_voice_url(interface_base_url) if interface_base_url else None,
            "interface_temperature": _bounded_float_config(
                realtime.get("interface_temperature"),
                default=0.2,
                minimum=0.0,
                maximum=2.0,
            ),
            "interface_max_output_tokens": _bounded_int_config(
                realtime.get("interface_max_output_tokens"),
                default=160,
                minimum=1,
                maximum=4096,
            ),
            "interface_timeout_seconds": _positive_float_config(
                realtime.get("interface_timeout_seconds"),
                default=0.8,
            ),
            "interface_max_audio_seconds": _bounded_float_config(
                realtime.get("interface_max_audio_seconds"),
                default=30.0,
                minimum=1.0,
                maximum=30.0,
            ),
            "interface_audio_input": str(realtime.get("interface_audio_input") or "") or None,
            "asr_mode": str(realtime.get("asr_mode") or "") or None,
            "asr_provider": asr_provider or None,
            "asr_model": asr_model or None,
            "asr_base_url": _redact_realtime_voice_url(asr_base_url) if asr_base_url else None,
            "oracle_provider": oracle_provider or None,
            "oracle_provider_name": oracle_provider_name or None,
            "preferred_local_oracle_model": str(realtime.get("preferred_local_oracle_model") or "") or None,
            "oracle_base_url": _redact_realtime_voice_url(oracle_base_url) if oracle_base_url else None,
            "oracle_api_mode": oracle_api_mode or None,
            "oracle_timeout_seconds": _positive_float_config(
                realtime.get("oracle_timeout_seconds"),
                default=60.0,
            ),
            "max_spoken_sentences": _positive_int_config(
                realtime.get("max_spoken_sentences"),
                default=2,
            ),
            "voice_response_policy": voice_response_policy,
            "tts_provider": tts_provider or None,
            "tts_model": tts_model or None,
            "tts_voice": tts_voice or None,
            "tts_base_url": _redact_realtime_voice_url(tts_base_url) if tts_base_url else None,
            "fallback_policy": fallback_policy,
            "language_support": language_support,
            "quality_targets_ms": quality_targets_ms,
            "routing": routing_policy,
            "metrics": metrics_policy,
            "oracle_jobs": oracle_jobs_policy,
            "oracle_tool_router": oracle_tool_router_policy,
            "output_events": output_events_policy,
            "turn_acknowledgement": turn_acknowledgement,
            "conversation_quality": conversation_quality,
            "production_readiness": production_readiness,
            "require_live_like": require_live_like,
        },
    )


_REALTIME_VOICE_AUDIO_SEND_TIMEOUT_SECONDS = 2.0
_REALTIME_VOICE_EVENT_SEND_TIMEOUT_SECONDS = 2.0


def _realtime_voice_event_from_binary_frame(frame: bytes):
    from agent.realtime_voice import VoiceEventType, event_from_binary_audio_frame

    return event_from_binary_audio_frame(frame, expected_type=VoiceEventType.AUDIO_INPUT_CHUNK)


def _realtime_voice_event_from_ws_message(message: Dict[str, Any]):
    from agent.realtime_voice import VoiceEvent

    message_type = str(message.get("type") or "")
    if message_type == "websocket.disconnect":
        raise WebSocketDisconnect(code=int(message.get("code") or 1000))

    text = message.get("text")
    if isinstance(text, str):
        return VoiceEvent.from_wire(json.loads(text))

    frame = message.get("bytes")
    if isinstance(frame, bytes):
        return _realtime_voice_event_from_binary_frame(frame)

    raise ValueError("realtime voice websocket message must be JSON text or binary audio")


def _realtime_voice_binary_frame_from_event(event) -> Optional[bytes]:
    from agent.realtime_voice import binary_audio_frame_from_event, is_output_audio_event_type

    return binary_audio_frame_from_event(event) if is_output_audio_event_type(event.type) else None


async def _send_realtime_voice_server_event(ws: WebSocket, event) -> bool:
    frame = _realtime_voice_binary_frame_from_event(event)
    if frame is None:
        try:
            await asyncio.wait_for(
                ws.send_json(event.to_wire()),
                timeout=_REALTIME_VOICE_EVENT_SEND_TIMEOUT_SECONDS,
            )
            return True
        except asyncio.TimeoutError:
            _log.warning(
                "dropping realtime voice control event due to websocket backpressure session=%s sequence=%s type=%s",
                event.session_id,
                event.sequence,
                event.type.value,
            )
            return False

    try:
        await asyncio.wait_for(ws.send_bytes(frame), timeout=_REALTIME_VOICE_AUDIO_SEND_TIMEOUT_SECONDS)
        return True
    except asyncio.TimeoutError:
        from agent.realtime_voice import VoiceEvent, VoiceEventType

        _log.warning(
            "dropping realtime voice audio frame due to websocket backpressure session=%s sequence=%s",
            event.session_id,
            event.sequence,
        )
        try:
            await asyncio.wait_for(
                ws.send_json(
                    VoiceEvent(
                        type=VoiceEventType.FRONTEND_STATE,
                        session_id=event.session_id,
                        sequence=0,
                        payload={
                            "status": "degraded",
                            "reason": "desktop_audio_backpressure",
                            "sidecar": False,
                        },
                    ).to_wire()
                ),
                timeout=_REALTIME_VOICE_EVENT_SEND_TIMEOUT_SECONDS,
            )
        except Exception:
            pass
        return False


@app.get("/api/voice/realtime/setup")
async def realtime_voice_setup(profile: Optional[str] = None) -> Dict[str, Any]:
    """Return provider-aware realtime voice setup state for GUI surfaces."""
    return _realtime_voice_setup_payload(profile)


@app.post("/api/voice/realtime/profile")
async def realtime_voice_apply_profile(
    body: RealtimeVoiceProfileApply,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply a known realtime voice provider preset."""
    try:
        return _apply_realtime_voice_profile_body(body, profile)
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("POST /api/voice/realtime/profile failed")
        raise HTTPException(status_code=500, detail=sanitize_realtime_voice_error(exc))


@app.post("/api/voice/realtime/smoke")
async def realtime_voice_smoke(
    body: RealtimeVoiceSmokeRequest,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the focused realtime voice evidence collector from the GUI."""
    target_profile = body.profile or profile

    def _run() -> Dict[str, Any]:
        with _profile_scope(target_profile):
            env_on_disk = load_env()
            realtime = _realtime_voice_config_dict()
            provider = str(realtime.get("frontend_provider") or "").strip().lower()
            hermes_home = _realtime_voice_profile_home(target_profile)
            output_dir = str(body.output_dir or "").strip()
            if not output_dir:
                output_dir = str(Path(hermes_home) / "artifacts" / "realtime-voice-evidence" / "gui-current")
            env = {
                **os.environ,
                **{k: str(v) for k, v in env_on_disk.items() if v is not None},
                "HERMES_HOME": hermes_home,
            }
            cmd = [
                sys.executable,
                "-m",
                "hermes_cli.realtime_voice_live_evidence",
                "--output-dir",
                output_dir,
                "--wait-seconds",
                str(max(0.0, float(body.wait_seconds or 0.0))),
            ]
            if body.require_discord:
                cmd.append("--require-live-discord")
            if body.require_inbound:
                cmd.append("--require-inbound")
            if provider in {"openai", "openai_realtime"}:
                cmd.append("--require-openai-realtime")
            if provider in {"gemini", "gemini_live"}:
                cmd.append("--require-gemini-live")

        timeout = max(20.0, float(body.wait_seconds or 0.0) + 25.0)
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
            timeout=timeout,
        )
        payload: Dict[str, Any] = {}
        if result.stdout.strip():
            try:
                parsed = json.loads(result.stdout)
                if isinstance(parsed, dict):
                    payload = parsed
            except json.JSONDecodeError:
                payload = {}
        return {
            "ok": result.returncode == 0,
            "exit_code": result.returncode,
            "output_dir": output_dir,
            "stdout": result.stdout[-12000:],
            "stderr": result.stderr[-12000:],
            "result": payload,
        }

    try:
        return await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "exit_code": None,
            "output_dir": body.output_dir or "",
            "stdout": (exc.stdout or "")[-12000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-12000:] if isinstance(exc.stderr, str) else "",
            "result": {"issues": ["smoke_test_timed_out"]},
        }
    except Exception as exc:
        _log.exception("POST /api/voice/realtime/smoke failed")
        raise HTTPException(status_code=500, detail=sanitize_realtime_voice_error(exc))


@app.get("/api/voice/realtime/status")
async def realtime_voice_status() -> Dict[str, Any]:
    """Return realtime voice capability and sidecar health for preflight UI."""
    return _realtime_voice_status_payload()


@app.websocket("/api/voice/realtime")
async def realtime_voice_ws(ws: WebSocket) -> None:
    peer = ws.client.host if ws.client else "?"

    auth_reason, cred = _ws_auth_reason(ws)
    mode = _ws_auth_mode()
    if auth_reason is not None:
        _log.warning(
            "realtime voice auth rejected reason=%s mode=%s cred=%s peer=%s",
            auth_reason,
            mode,
            cred,
            peer,
        )
        await ws.close(code=4401, reason=_ws_close_reason(f"auth: {auth_reason}"))
        return

    request_reason = _ws_request_reason(ws)
    if request_reason is not None:
        _log.warning("realtime voice refused: %s peer=%s", request_reason, peer)
        await ws.close(code=4403, reason=_ws_close_reason(request_reason))
        return

    cfg = load_config()
    voice = cfg.get("voice") if isinstance(cfg, dict) else {}
    realtime = voice.get("realtime") if isinstance(voice, dict) else {}
    if not (isinstance(realtime, dict) and realtime.get("enabled") is True):
        await ws.close(code=4403, reason="realtime voice disabled")
        return

    try:
        _ensure_realtime_voice_sidecar(realtime)
    except Exception as exc:
        _log.warning("realtime voice sidecar failed to start", exc_info=True)
        await ws.close(code=1011, reason=_ws_close_reason(f"sidecar: {sanitize_realtime_voice_error(exc)}"))
        return

    unavailable_reason = _realtime_voice_open_unavailable_reason(realtime)
    if unavailable_reason:
        _log.warning("realtime voice refused: %s peer=%s", unavailable_reason, peer)
        await ws.close(code=1011, reason=_ws_close_reason(f"sidecar: {unavailable_reason}"))
        return

    try:
        config = _realtime_voice_config_from_request(ws)
    except Exception as exc:
        await ws.close(code=4400, reason=_ws_close_reason(f"bad config: {sanitize_realtime_voice_error(exc)}"))
        return

    await ws.accept()
    _log.info(
        "realtime voice accepted peer=%s mode=%s cred=%s session=%s engine=%s",
        peer,
        mode,
        cred,
        config.session_id,
        config.engine.value,
    )

    from agent.realtime_voice_session import RealtimeVoiceSession

    session = RealtimeVoiceSession(config)
    try:
        await session.start()
    except Exception as exc:
        _log.warning("realtime voice session failed to start", exc_info=True)
        error = sanitize_realtime_voice_error(exc)
        await ws.send_json(
            {
                "type": "session.error",
                "session_id": config.session_id,
                "sequence": 0,
                "payload": {"error": error},
            }
        )
        await ws.close(code=1011, reason=_ws_close_reason(error))
        return

    async def pump_events() -> None:
        from agent.realtime_voice import VoiceEventType, is_output_audio_event_type

        async for event in session.events():
            sent = await _send_realtime_voice_server_event(ws, event)
            if not sent and not is_output_audio_event_type(event.type):
                await ws.close(code=1011, reason="realtime voice desktop websocket backpressure")
                return
            if event.type == VoiceEventType.SESSION_ERROR:
                reason = str(event.payload.get("error") or "realtime voice session error")
                await ws.close(code=1011, reason=_ws_close_reason(reason))
                return

    event_task = asyncio.create_task(pump_events())

    try:
        while True:
            event = _realtime_voice_event_from_ws_message(await ws.receive())
            await session.receive_client_event(event)
            if event.type.value in {"session.stop", "session.closed"}:
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(event_task, timeout=1.0)
                with suppress(Exception):
                    await ws.close(code=1000, reason="session stopped")
                return
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        _log.warning("realtime voice websocket error", exc_info=True)
        try:
            error = sanitize_realtime_voice_error(exc)
            await ws.send_json(
                {
                    "type": "session.error",
                    "session_id": config.session_id,
                    "sequence": 0,
                    "payload": {"error": error},
                }
            )
        except Exception:
            pass
    finally:
        event_task.cancel()
        try:
            await event_task
        except (asyncio.CancelledError, Exception):
            pass
        await session.close()


@app.websocket("/api/pty")
async def pty_ws(ws: WebSocket) -> None:
    peer = ws.client.host if ws.client else "?"

    if not _DASHBOARD_EMBEDDED_CHAT_ENABLED:
        _log.info("pty refused: embedded chat disabled peer=%s", peer)
        await ws.close(code=4404, reason="embedded chat disabled")
        return

    # --- auth + host/origin/peer check (before accept so we can close
    #     cleanly AND tell the client WHY via the close code + reason).
    #     Each gate maps to a distinct close code so the log and the
    #     browser banner agree on the cause:
    #       4401 bad credential   4403 host/origin mismatch
    #       4408 peer not allowed  4404 chat disabled
    auth_reason, cred = _ws_auth_reason(ws)
    mode = _ws_auth_mode()
    if auth_reason is not None:
        _log.warning(
            "pty auth rejected reason=%s mode=%s cred=%s peer=%s",
            auth_reason, mode, cred, peer,
        )
        await ws.close(code=4401, reason=_ws_close_reason(f"auth: {auth_reason}"))
        return

    host_origin_reason = _ws_host_origin_reason(ws)
    if host_origin_reason is not None:
        _log.warning("pty refused: %s peer=%s", host_origin_reason, peer)
        await ws.close(code=4403, reason=_ws_close_reason(host_origin_reason))
        return

    client_reason = _ws_client_reason(ws)
    if client_reason is not None:
        _log.warning("pty refused: %s", client_reason)
        await ws.close(code=4408, reason=_ws_close_reason(client_reason))
        return

    await ws.accept()
    _log.info("pty accepted peer=%s mode=%s cred=%s", peer, mode, cred)

    # On native Windows, the POSIX PTY bridge can't be imported.  Tell the
    # client and close cleanly rather than pretending the feature works.
    if not _PTY_BRIDGE_AVAILABLE:
        await ws.send_text(
            "\r\n\x1b[31mChat unavailable: the embedded terminal requires a "
            "POSIX PTY, which native Windows Python doesn't provide.\x1b[0m\r\n"
            "\x1b[33mInstall Hermes inside WSL2 to use the dashboard's /chat "
            "tab — the rest of the dashboard works here.\x1b[0m\r\n"
        )
        await ws.close(code=1011)
        return

    # --- spawn PTY ------------------------------------------------------
    resume = ws.query_params.get("resume") or None
    profile = ws.query_params.get("profile") or None
    channel = _channel_or_close_code(ws)
    sidecar_url = _build_sidecar_url(channel) if channel else None
    force_fresh = (ws.query_params.get("fresh") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    active_session_file: Optional[Path] = None

    if channel:
        active_session_file = _active_session_file_for_channel(ws.app, channel)
        if force_fresh:
            resume = None
            _forget_active_session_file(active_session_file)
        elif not resume:
            resume = _read_active_session_file(active_session_file)

    resolve_kwargs = {
        "resume": resume,
        "sidecar_url": sidecar_url,
        "profile": profile,
    }
    if active_session_file is not None:
        resolve_kwargs["active_session_file"] = str(active_session_file)

    try:
        argv, cwd, env = await _resolve_chat_argv_async(**resolve_kwargs)
    except HTTPException as exc:
        # Unknown/invalid profile from _resolve_profile_dir.
        await ws.send_text(f"\r\n\x1b[31mChat unavailable: {exc.detail}\x1b[0m\r\n")
        await ws.close(code=1011)
        return
    except SystemExit as exc:
        # _make_tui_argv calls sys.exit(1) when node/npm is missing.
        await ws.send_text(f"\r\n\x1b[31mChat unavailable: {exc}\x1b[0m\r\n")
        await ws.close(code=1011)
        return


    try:
        bridge = await asyncio.to_thread(PtyBridge.spawn, argv, cwd=cwd, env=env)
    except PtyUnavailableError as exc:
        await ws.send_text(f"\r\n\x1b[31mChat unavailable: {exc}\x1b[0m\r\n")
        await ws.close(code=1011)
        return
    except (FileNotFoundError, OSError) as exc:
        await ws.send_text(f"\r\n\x1b[31mChat failed to start: {exc}\x1b[0m\r\n")
        await ws.close(code=1011)
        return

    loop = asyncio.get_running_loop()

    # --- reader task: PTY master → WebSocket ----------------------------
    async def pump_pty_to_ws() -> None:
        try:
            while True:
                chunk = await loop.run_in_executor(
                    None, bridge.read, _PTY_READ_CHUNK_TIMEOUT
                )
                if chunk is None:  # EOF
                    return
                if not chunk:  # no data this tick; yield control and retry
                    await asyncio.sleep(0)
                    continue
                try:
                    await ws.send_bytes(chunk)
                except Exception:
                    return
        finally:
            # The child has exited (EOF) or the send side broke.  Close the
            # WebSocket so the writer loop's ``ws.receive()`` returns instead
            # of blocking forever — otherwise, when the browser's socket is
            # half-open (no FIN delivered, common on macOS/launchd) the
            # handler never reaches its ``finally`` and the PTY's fds leak.
            # With dashboard auto-reconnect (#52962) every dropped socket then
            # stacks a fresh PTY on top of the orphaned one, exhausting fds.
            #
            # Reap the bridge here too (close() is idempotent): on child EOF the
            # writer loop's ``finally`` is the usual closer, but if the handler
            # task is cancelled the instant we close the WS, that ``finally``
            # can be skipped, leaking the PTY. Closing from the EOF path makes
            # the reap independent of that cancellation race (#54028).
            try:
                await asyncio.to_thread(bridge.close)
            except Exception:
                pass
            try:
                await ws.close()
            except Exception:
                pass

    reader_task = asyncio.create_task(pump_pty_to_ws())

    # --- writer loop: WebSocket → PTY master ----------------------------
    try:
        while True:
            try:
                msg = await ws.receive()
            except RuntimeError:
                # Raised when ws.receive() is called after the socket is
                # already disconnected (e.g. closed by the reader task above).
                break
            msg_type = msg.get("type")
            if msg_type == "websocket.disconnect":
                break
            raw = msg.get("bytes")
            if raw is None:
                text = msg.get("text")
                raw = text.encode("utf-8") if isinstance(text, str) else b""
            if not raw:
                continue

            # Resize escape is consumed locally, never written to the PTY.
            match = _RESIZE_RE.match(raw)
            if match and match.end() == len(raw):
                cols = int(match.group(1))
                rows = int(match.group(2))
                bridge.resize(cols=cols, rows=rows)
                continue

            bridge.write(raw)
    except WebSocketDisconnect:
        pass
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except (asyncio.CancelledError, Exception):
            pass
        await asyncio.to_thread(bridge.close)


# ---------------------------------------------------------------------------
# /api/ws — JSON-RPC WebSocket sidecar for the dashboard "Chat" tab.
#
# Drives the same `tui_gateway.dispatch` surface Ink uses over stdio, so the
# dashboard can render structured metadata (model badge, tool-call sidebar,
# slash launcher, session info) alongside the xterm.js terminal that PTY
# already paints. Both transports bind to the same session id when one is
# active, so a tool.start emitted by the agent fans out to both sinks.
# ---------------------------------------------------------------------------


@app.websocket("/api/ws")
async def gateway_ws(ws: WebSocket) -> None:
    if not _DASHBOARD_EMBEDDED_CHAT_ENABLED:
        await ws.close(code=4403)
        return

    if not _ws_auth_ok(ws):
        await ws.close(code=4401)
        return

    if not _ws_request_is_allowed(ws):
        await ws.close(code=4403)
        return

    from tui_gateway.ws import handle_ws

    await handle_ws(ws)


# ---------------------------------------------------------------------------
# /api/pub + /api/events — chat-tab event broadcast.
#
# The PTY-side ``tui_gateway.entry`` opens /api/pub at startup (driven by
# HERMES_TUI_SIDECAR_URL set in /api/pty's PTY env) and writes every
# dispatcher emit through it.  The dashboard fans those frames out to any
# subscriber that opened /api/events on the same channel id.  This is what
# gives the React sidebar its tool-call feed without breaking the PTY
# child's stdio handshake with Ink.
# ---------------------------------------------------------------------------


@app.websocket("/api/pub")
async def pub_ws(ws: WebSocket) -> None:
    if not _DASHBOARD_EMBEDDED_CHAT_ENABLED:
        await ws.close(code=4403)
        return

    if not _ws_auth_ok(ws):
        await ws.close(code=4401)
        return

    if not _ws_request_is_allowed(ws):
        await ws.close(code=4403)
        return

    channel = _channel_or_close_code(ws)
    if not channel:
        await ws.close(code=4400)
        return

    await ws.accept()

    try:
        while True:
            await _broadcast_event(ws.app, channel, await ws.receive_text())
    except WebSocketDisconnect:
        pass


@app.websocket("/api/events")
async def events_ws(ws: WebSocket) -> None:
    if not _DASHBOARD_EMBEDDED_CHAT_ENABLED:
        await ws.close(code=4403)
        return

    if not _ws_auth_ok(ws):
        await ws.close(code=4401)
        return

    if not _ws_request_is_allowed(ws):
        await ws.close(code=4403)
        return

    channel = _channel_or_close_code(ws)
    if not channel:
        await ws.close(code=4400)
        return

    await ws.accept()

    event_channels, event_lock = _get_event_state(ws.app)
    async with event_lock:
        event_channels.setdefault(channel, set()).add(ws)

    try:
        while True:
            # Subscribers don't speak — the receive() just blocks until
            # disconnect so the connection stays open as long as the
            # browser holds it.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with event_lock:
            subs = event_channels.get(channel)

            if subs is not None:
                subs.discard(ws)

                if not subs:
                    event_channels.pop(channel, None)


def _normalise_prefix(raw: Optional[str]) -> str:
    """Normalise an X-Forwarded-Prefix header value.

    Thin re-export of :func:`hermes_cli.dashboard_auth.prefix.normalise_prefix`
    — the single source of truth lives in the dashboard_auth package so
    the gate middleware, the OAuth routes, the cookie helpers, and the
    SPA mount all agree on validation rules.
    """
    from hermes_cli.dashboard_auth.prefix import normalise_prefix
    return normalise_prefix(raw)


def mount_spa(application: FastAPI):
    """Mount the built SPA. Falls back to index.html for client-side routing.

    The session token is injected into index.html via a ``<script>`` tag so
    the SPA can authenticate against protected API endpoints without a
    separate (unauthenticated) token-dispensing endpoint.

    When served behind a path-prefix reverse proxy (e.g.
    ``mission-control.tilos.com/hermes/*`` -> local Caddy -> :9119), the
    proxy injects ``X-Forwarded-Prefix: /hermes`` on every request. We
    rewrite the served ``index.html`` so absolute asset URLs (``/assets/...``)
    and the SPA's runtime ``__HERMES_BASE_PATH__`` honour that prefix
    without rebuilding the bundle.
    """
    if not WEB_DIST.exists():
        @application.get("/{full_path:path}")
        async def no_frontend(full_path: str):
            return JSONResponse(
                {"error": "Frontend not built. Run: cd web && npm run build"},
                status_code=404,
            )
        return

    _index_path = WEB_DIST / "index.html"

    def _serve_index(prefix: str = ""):
        """Return index.html with the session token + base-path injected.

        ``prefix`` is the normalised ``X-Forwarded-Prefix`` (e.g. ``/hermes``)
        or empty string when served at root.

        When the OAuth auth gate is active (``app.state.auth_required``),
        the legacy ``_SESSION_TOKEN`` is NOT injected — the SPA reads
        identity from ``/api/auth/me`` over cookie auth instead.  The
        ``__HERMES_AUTH_REQUIRED__`` flag lets the SPA pick the right
        auth scheme for /api/pty and /api/ws (ticket vs token).
        """
        html = _index_path.read_text(encoding="utf-8")
        chat_js = "true" if _DASHBOARD_EMBEDDED_CHAT_ENABLED else "false"
        gated = bool(getattr(app.state, "auth_required", False))
        gated_js = "true" if gated else "false"
        if gated:
            bootstrap_script = (
                f"<script>"
                f"window.__HERMES_DASHBOARD_EMBEDDED_CHAT__={chat_js};"
                f'window.__HERMES_BASE_PATH__="{prefix}";'
                f"window.__HERMES_AUTH_REQUIRED__={gated_js};"
                f"</script>"
            )
        else:
            bootstrap_script = (
                f'<script>window.__HERMES_SESSION_TOKEN__="{_SESSION_TOKEN}";'
                f"window.__HERMES_DASHBOARD_EMBEDDED_CHAT__={chat_js};"
                f'window.__HERMES_BASE_PATH__="{prefix}";'
                f"window.__HERMES_AUTH_REQUIRED__={gated_js};"
                f"</script>"
            )
        if prefix:
            # Rewrite absolute asset URLs baked into the Vite build so the
            # browser fetches them through the same proxy prefix.
            html = html.replace('href="/assets/', f'href="{prefix}/assets/')
            html = html.replace('src="/assets/', f'src="{prefix}/assets/')
            html = html.replace('href="/favicon.ico"', f'href="{prefix}/favicon.ico"')
            html = html.replace('href="/fonts/', f'href="{prefix}/fonts/')
            html = html.replace('href="/ds-assets/', f'href="{prefix}/ds-assets/')
            html = html.replace('src="/ds-assets/', f'src="{prefix}/ds-assets/')
        html = html.replace("</head>", f"{bootstrap_script}</head>", 1)
        return HTMLResponse(
            html,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    # When served behind a path-prefix proxy, the built CSS contains
    # absolute ``url(/fonts/...)`` and ``url(/ds-assets/...)`` references.
    # Browsers resolve those against the document origin, which means
    # under ``/hermes`` they'd hit ``mission-control.tilos.com/fonts/...``
    # (the MC Pages app), not the Hermes backend. Intercept CSS asset
    # requests BEFORE the StaticFiles mount and rewrite the absolute paths
    # when a prefix is in play.
    @application.get("/assets/{filename}.css")
    async def serve_css(filename: str, request: Request):
        css_path = WEB_DIST / "assets" / f"{filename}.css"
        if not css_path.is_file() or not css_path.resolve().is_relative_to(
            WEB_DIST.resolve()
        ):
            return JSONResponse({"error": "not found"}, status_code=404)
        prefix = _normalise_prefix(request.headers.get("x-forwarded-prefix"))
        css = css_path.read_text(encoding="utf-8")
        if prefix:
            for asset_dir in ("/fonts/", "/fonts-terminal/", "/ds-assets/", "/assets/"):
                css = css.replace(f"url({asset_dir}", f"url({prefix}{asset_dir}")
                css = css.replace(f"url(\"{asset_dir}", f"url(\"{prefix}{asset_dir}")
                css = css.replace(f"url('{asset_dir}", f"url('{prefix}{asset_dir}")
        return Response(content=css, media_type="text/css")

    application.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @application.get("/{full_path:path}")
    async def serve_spa(full_path: str, request: Request):
        prefix = _normalise_prefix(request.headers.get("x-forwarded-prefix"))
        # An unmatched /api/* path is a missing/renamed endpoint, NOT a
        # client-side route. Falling through to index.html here returns
        # `<!doctype html>` with status 200, which makes JSON clients (the
        # desktop app's fetchJson, dashboard fetch wrappers) blow up with an
        # opaque `SyntaxError: Unexpected token '<'`. Return a real 404 JSON
        # so the caller sees a clear "no such endpoint" instead.
        if full_path == "api" or full_path.startswith("api/"):
            return JSONResponse(
                {"detail": f"No such API endpoint: /{full_path}"},
                status_code=404,
            )
        file_path = WEB_DIST / full_path
        # Prevent path traversal via url-encoded sequences (%2e%2e/)
        if (
            full_path
            and file_path.resolve().is_relative_to(WEB_DIST.resolve())
            and file_path.exists()
            and file_path.is_file()
        ):
            return FileResponse(file_path)
        return _serve_index(prefix)


# ---------------------------------------------------------------------------
# Dashboard theme endpoints
# ---------------------------------------------------------------------------

# Built-in dashboard themes — label + description only.  The actual color
# definitions live in the frontend (web/src/themes/presets.ts).
_BUILTIN_DASHBOARD_THEMES = [
    {"name": "default",       "label": "Hermes Teal",         "description": "Classic dark teal — the canonical Hermes look"},
    {"name": "default-large", "label": "Hermes Teal (Large)", "description": "Hermes Teal with bigger fonts and roomier spacing"},
    {"name": "nous-blue",     "label": "Nous Blue",           "description": "Light mode — vivid Nous-blue accents on cream canvas"},
    {"name": "midnight",      "label": "Midnight",            "description": "Deep blue-violet with cool accents"},
    {"name": "ember",     "label": "Ember",          "description": "Warm crimson and bronze — forge vibes"},
    {"name": "mono",      "label": "Mono",           "description": "Clean grayscale — minimal and focused"},
    {"name": "cyberpunk", "label": "Cyberpunk",      "description": "Neon green on black — matrix terminal"},
    {"name": "rose",      "label": "Rosé",           "description": "Soft pink and warm ivory — easy on the eyes"},
]


def _parse_theme_layer(value: Any, default_hex: str, default_alpha: float = 1.0) -> Optional[Dict[str, Any]]:
    """Normalise a theme layer spec from YAML into `{hex, alpha}` form.

    Accepts shorthand (a bare hex string) or full dict form.  Returns
    ``None`` on garbage input so the caller can fall back to a built-in
    default rather than blowing up.
    """
    if value is None:
        return {"hex": default_hex, "alpha": default_alpha}
    if isinstance(value, str):
        return {"hex": value, "alpha": default_alpha}
    if isinstance(value, dict):
        hex_val = value.get("hex", default_hex)
        alpha_val = value.get("alpha", default_alpha)
        if not isinstance(hex_val, str):
            return None
        try:
            alpha_f = float(alpha_val)
        except (TypeError, ValueError):
            alpha_f = default_alpha
        return {"hex": hex_val, "alpha": max(0.0, min(1.0, alpha_f))}
    return None


_THEME_DEFAULT_TYPOGRAPHY: Dict[str, str] = {
    "fontSans": 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    "fontMono": 'ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace',
    "baseSize": "15px",
    "lineHeight": "1.55",
    "letterSpacing": "0",
}

_THEME_DEFAULT_LAYOUT: Dict[str, str] = {
    "radius": "0.5rem",
    "density": "comfortable",
}

_THEME_OVERRIDE_KEYS = {
    "card", "cardForeground", "popover", "popoverForeground",
    "primary", "primaryForeground", "secondary", "secondaryForeground",
    "muted", "mutedForeground", "accent", "accentForeground",
    "destructive", "destructiveForeground", "success", "warning",
    "border", "input", "ring",
}

# Well-known named asset slots themes can populate.  Any other keys under
# ``assets.custom`` are exposed as ``--theme-asset-custom-<key>`` CSS vars
# for plugin/shell use.
_THEME_NAMED_ASSET_KEYS = {"bg", "hero", "logo", "crest", "sidebar", "header"}

# Component-style buckets themes can override.  The value under each bucket
# is a mapping from camelCase property name to CSS string; each pair emits
# ``--component-<bucket>-<kebab-property>`` on :root.  The frontend's shell
# components (Card, App header, Backdrop, etc.) consume these vars so themes
# can restyle chrome (clip-path, border-image, segmented progress, etc.)
# without shipping their own CSS.
_THEME_COMPONENT_BUCKETS = {
    "card", "header", "footer", "sidebar", "tab",
    "progress", "badge", "backdrop", "page",
}

_THEME_LAYOUT_VARIANTS = {"standard", "cockpit", "tiled"}

# Cap on customCSS length so a malformed/oversized theme YAML can't blow up
# the response payload or the <style> tag.  32 KiB is plenty for every
# practical reskin (the Strike Freedom demo is ~2 KiB).
_THEME_CUSTOM_CSS_MAX = 32 * 1024


def _normalise_theme_definition(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalise a user theme YAML into the wire format `ThemeProvider`
    expects.  Returns ``None`` if the theme is unusable.

    Accepts both the full schema (palette/typography/layout) and a loose
    form with bare hex strings, so hand-written YAMLs stay friendly.
    """
    if not isinstance(data, dict):
        return None
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        return None

    # Palette
    palette_src = data.get("palette", {}) if isinstance(data.get("palette"), dict) else {}
    # Allow top-level `colors.background` as a shorthand too.
    colors_src = data.get("colors", {}) if isinstance(data.get("colors"), dict) else {}

    def _layer(key: str, default_hex: str, default_alpha: float = 1.0) -> Dict[str, Any]:
        spec = palette_src.get(key, colors_src.get(key))
        parsed = _parse_theme_layer(spec, default_hex, default_alpha)
        return parsed if parsed is not None else {"hex": default_hex, "alpha": default_alpha}

    palette = {
        "background": _layer("background", "#041c1c", 1.0),
        "midground": _layer("midground", "#ffe6cb", 1.0),
        "foreground": _layer("foreground", "#ffffff", 0.0),
        "warmGlow": palette_src.get("warmGlow") or data.get("warmGlow") or "rgba(255, 189, 56, 0.35)",
        "noiseOpacity": 1.0,
    }
    raw_noise = palette_src.get("noiseOpacity", data.get("noiseOpacity"))
    try:
        palette["noiseOpacity"] = float(raw_noise) if raw_noise is not None else 1.0
    except (TypeError, ValueError):
        palette["noiseOpacity"] = 1.0

    # Typography
    typo_src = data.get("typography", {}) if isinstance(data.get("typography"), dict) else {}
    typography = dict(_THEME_DEFAULT_TYPOGRAPHY)
    for key in ("fontSans", "fontMono", "fontDisplay", "fontUrl", "baseSize", "lineHeight", "letterSpacing"):
        val = typo_src.get(key)
        if isinstance(val, str) and val.strip():
            typography[key] = val

    # Layout
    layout_src = data.get("layout", {}) if isinstance(data.get("layout"), dict) else {}
    layout = dict(_THEME_DEFAULT_LAYOUT)
    radius = layout_src.get("radius")
    if isinstance(radius, str) and radius.strip():
        layout["radius"] = radius
    density = layout_src.get("density")
    if isinstance(density, str) and density in {"compact", "comfortable", "spacious"}:
        layout["density"] = density

    # Color overrides — keep only valid keys with string values.
    overrides_src = data.get("colorOverrides", {})
    color_overrides: Dict[str, str] = {}
    if isinstance(overrides_src, dict):
        for key, val in overrides_src.items():
            if key in _THEME_OVERRIDE_KEYS and isinstance(val, str) and val.strip():
                color_overrides[key] = val

    # Assets — named slots + arbitrary user-defined keys.  Values must be
    # strings (URLs or CSS ``url(...)``/``linear-gradient(...)`` expressions).
    # We don't fetch remote assets here; the frontend just injects them as
    # CSS vars.  Empty values are dropped so a theme can explicitly clear a
    # slot by setting ``hero: ""``.
    assets_out: Dict[str, Any] = {}
    assets_src = data.get("assets", {}) if isinstance(data.get("assets"), dict) else {}
    for key in _THEME_NAMED_ASSET_KEYS:
        val = assets_src.get(key)
        if isinstance(val, str) and val.strip():
            assets_out[key] = val
    custom_assets_src = assets_src.get("custom")
    if isinstance(custom_assets_src, dict):
        custom_assets: Dict[str, str] = {}
        for key, val in custom_assets_src.items():
            if (
                isinstance(key, str)
                and key.replace("-", "").replace("_", "").isalnum()
                and isinstance(val, str)
                and val.strip()
            ):
                custom_assets[key] = val
        if custom_assets:
            assets_out["custom"] = custom_assets

    # Custom CSS — raw CSS text the frontend injects as a scoped <style>
    # tag on theme apply.  Clipped to _THEME_CUSTOM_CSS_MAX to keep the
    # payload bounded.  We intentionally do NOT parse/sanitise the CSS
    # here — the dashboard is localhost-only and themes are user-authored
    # YAML in ~/.hermes/, same trust level as the config file itself.
    custom_css_val = data.get("customCSS")
    custom_css: Optional[str] = None
    if isinstance(custom_css_val, str) and custom_css_val.strip():
        custom_css = custom_css_val[:_THEME_CUSTOM_CSS_MAX]

    # Component style overrides — per-bucket dicts of camelCase CSS
    # property -> CSS string.  The frontend converts these into CSS vars
    # that shell components (Card, App header, Backdrop) consume.
    component_styles_src = data.get("componentStyles", {})
    component_styles: Dict[str, Dict[str, str]] = {}
    if isinstance(component_styles_src, dict):
        for bucket, props in component_styles_src.items():
            if bucket not in _THEME_COMPONENT_BUCKETS or not isinstance(props, dict):
                continue
            clean: Dict[str, str] = {}
            for prop, value in props.items():
                if (
                    isinstance(prop, str)
                    and prop.replace("-", "").replace("_", "").isalnum()
                    and isinstance(value, (str, int, float))
                    and str(value).strip()
                ):
                    clean[prop] = str(value)
            if clean:
                component_styles[bucket] = clean

    layout_variant_src = data.get("layoutVariant")
    layout_variant = (
        layout_variant_src
        if isinstance(layout_variant_src, str) and layout_variant_src in _THEME_LAYOUT_VARIANTS
        else "standard"
    )

    result: Dict[str, Any] = {
        "name": name,
        "label": data.get("label") or name,
        "description": data.get("description", ""),
        "palette": palette,
        "typography": typography,
        "layout": layout,
        "layoutVariant": layout_variant,
    }
    if color_overrides:
        result["colorOverrides"] = color_overrides
    if assets_out:
        result["assets"] = assets_out
    if custom_css is not None:
        result["customCSS"] = custom_css
    if component_styles:
        result["componentStyles"] = component_styles
    return result


def _discover_user_themes() -> list:
    """Scan ~/.hermes/dashboard-themes/*.yaml for user-created themes.

    Returns a list of fully-normalised theme definitions ready to ship
    to the frontend, so the client can apply them without a secondary
    round-trip or a built-in stub.
    """
    themes_dir = get_hermes_home() / "dashboard-themes"
    if not themes_dir.is_dir():
        return []
    result = []
    for f in sorted(themes_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        normalised = _normalise_theme_definition(data)
        if normalised is not None:
            result.append(normalised)
    return result


@app.get("/api/dashboard/themes")
async def get_dashboard_themes():
    """Return available themes and the currently active one.

    Built-in entries ship name/label/description only (the frontend owns
    their full definitions in `web/src/themes/presets.ts`).  User themes
    from `~/.hermes/dashboard-themes/*.yaml` ship with their full
    normalised definition under `definition`, so the client can apply
    them without a stub.
    """
    config = load_config()
    active = cfg_get(config, "dashboard", "theme", default="default")
    user_themes = _discover_user_themes()
    seen = set()
    themes = []
    for t in _BUILTIN_DASHBOARD_THEMES:
        seen.add(t["name"])
        themes.append(t)
    for t in user_themes:
        if t["name"] in seen:
            continue
        themes.append({
            "name": t["name"],
            "label": t["label"],
            "description": t["description"],
            "definition": t,
        })
        seen.add(t["name"])
    return {"themes": themes, "active": active}


class ThemeSetBody(BaseModel):
    name: str


@app.put("/api/dashboard/theme")
async def set_dashboard_theme(body: ThemeSetBody):
    """Set the active dashboard theme (persists to config.yaml)."""
    config = load_config()
    if "dashboard" not in config:
        config["dashboard"] = {}
    config["dashboard"]["theme"] = body.name
    save_config(config)
    return {"ok": True, "theme": body.name}


# Curated font-override ids. Kept in sync with FONT_CHOICES in
# web/src/themes/fonts.ts — the frontend owns the stacks + webfont URLs;
# the backend only needs the id allow-list so it can reject anything not
# in the vetted catalog (the font's webfont URL is injected as a <link>,
# so we never accept an arbitrary user-supplied id/URL here).
_FONT_DEFAULT_ID = "theme"
_FONT_CHOICES = frozenset({
    "system-sans", "system-serif", "system-mono",
    "inter", "ibm-plex-sans", "work-sans", "atkinson-hyperlegible", "dm-sans",
    "spectral", "fraunces", "source-serif",
    "jetbrains-mono", "ibm-plex-mono", "space-mono",
})


@app.get("/api/dashboard/font")
async def get_dashboard_font():
    """Return the active font override (``"theme"`` = use the theme's font)."""
    config = load_config()
    font = cfg_get(config, "dashboard", "font", default=_FONT_DEFAULT_ID)
    if font not in _FONT_CHOICES:
        font = _FONT_DEFAULT_ID
    return {"font": font}


class FontSetBody(BaseModel):
    font: str


@app.put("/api/dashboard/font")
async def set_dashboard_font(body: FontSetBody):
    """Set the dashboard font override (persists to config.yaml).

    Accepts any id in the curated catalog, or ``"theme"`` to clear the
    override and fall back to the active theme's own font. Unknown ids are
    coerced to ``"theme"`` rather than 400'd so a stale client can't wedge
    the picker.
    """
    font = body.font if body.font in _FONT_CHOICES else _FONT_DEFAULT_ID
    config = load_config()
    if "dashboard" not in config:
        config["dashboard"] = {}
    config["dashboard"]["font"] = font
    save_config(config)
    return {"ok": True, "font": font}


# ---------------------------------------------------------------------------
# Dashboard plugin system
# ---------------------------------------------------------------------------

def _safe_plugin_api_relpath(api_field: Any, *, dashboard_dir: Path) -> Optional[str]:
    """Validate the manifest's ``api`` field for the plugin loader.

    The web server later imports this file as a Python module via
    ``importlib.util.spec_from_file_location`` (arbitrary code
    execution by design — that's how plugins extend the backend).
    Pre-#29156 the field was used as-is, which meant:

    * An absolute path swallowed the plugin's dashboard directory
      entirely — ``Path('safe/dashboard') / '/tmp/evil.py'`` resolves
      to ``/tmp/evil.py``, so any attacker-controlled manifest could
      point the import at any Python file on disk (GHSA-5qr3-c538-wm9j).
    * A ``../..`` traversal could climb out of the plugin into
      neighbouring directories on the search path.

    Return the original string when the resolved path stays under
    ``dashboard_dir``; return ``None`` (with a warning logged at the
    call site) otherwise so the plugin still loads its static JS/CSS
    but its backend ``api`` is rejected.
    """
    if not isinstance(api_field, str) or not api_field.strip():
        return None
    candidate = Path(api_field)
    if candidate.is_absolute():
        return None
    try:
        resolved = (dashboard_dir / candidate).resolve()
        base = dashboard_dir.resolve()
    except (OSError, RuntimeError):
        return None
    try:
        resolved.relative_to(base)
    except ValueError:
        return None
    return api_field


def _discover_dashboard_plugins() -> list:
    """Scan plugins/*/dashboard/manifest.json for dashboard extensions.

    Checks three plugin sources (same as hermes_cli.plugins):
    1. User plugins:    ~/.hermes/plugins/<name>/dashboard/manifest.json
    2. Bundled plugins: <repo>/plugins/<name>/dashboard/manifest.json  (memory/, etc.)
    3. Project plugins: ./.hermes/plugins/  (only if HERMES_ENABLE_PROJECT_PLUGINS)
    """
    plugins = []
    seen_names: set = set()

    from hermes_cli.plugins import get_bundled_plugins_dir
    bundled_root = get_bundled_plugins_dir()
    search_dirs = [
        (get_hermes_home() / "plugins", "user"),
        (bundled_root / "memory", "bundled"),
        (bundled_root, "bundled"),
    ]
    # GHSA-5qr3-c538-wm9j (#29156): the previous ``os.environ.get(...)``
    # check treated *any* non-empty string as truthy, so ``=0``, ``=false``,
    # and ``=no`` — all of which the agent loader and operators correctly
    # read as "disabled" — silently *enabled* the untrusted project source
    # in the web server.  Combined with the absolute-path RCE primitive on
    # the manifest's ``api`` field (now patched below), this turned the
    # opt-in into a sticky always-on switch.  Use the shared truthy
    # semantics (``1`` / ``true`` / ``yes`` / ``on``) so the gate matches
    # ``hermes_cli/plugins.py`` and the documented user contract.
    if env_var_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
        search_dirs.append((Path.cwd() / ".hermes" / "plugins", "project"))

    for plugins_root, source in search_dirs:
        if not plugins_root.is_dir():
            continue
        for child in sorted(plugins_root.iterdir()):
            if not child.is_dir():
                continue
            manifest_file = child / "dashboard" / "manifest.json"
            if not manifest_file.exists():
                continue
            try:
                data = json.loads(manifest_file.read_text(encoding="utf-8"))
                name = data.get("name", child.name)
                if name in seen_names:
                    continue
                seen_names.add(name)
                # Tab options: ``path`` + ``position`` for a new tab, optional
                # ``override`` to replace a built-in route, and ``hidden`` to
                # register the plugin component/slots without adding a tab
                # (useful for slot-only plugins like a header-crest injector).
                raw_tab = data.get("tab", {}) if isinstance(data.get("tab"), dict) else {}
                tab_info = {
                    "path": raw_tab.get("path", f"/{name}"),
                    "position": raw_tab.get("position", "end"),
                }
                override_path = raw_tab.get("override")
                if isinstance(override_path, str) and override_path.startswith("/"):
                    tab_info["override"] = override_path
                if bool(raw_tab.get("hidden")):
                    tab_info["hidden"] = True
                # Slots: list of named slot locations this plugin populates.
                # The frontend exposes ``registerSlot(pluginName, slotName, Component)``
                # on window; plugins with non-empty slots call it from their JS bundle.
                slots_src = data.get("slots")
                slots: List[str] = []
                if isinstance(slots_src, list):
                    slots = [s for s in slots_src if isinstance(s, str) and s]
                # Validate ``api`` at discovery time so the value cached
                # on the plugin entry is already safe to feed into the
                # importer.  An attacker-controlled manifest can name
                # any absolute path or ``..`` traversal here — the
                # web server then imports that file as a Python module
                # (RCE, GHSA-5qr3-c538-wm9j).
                raw_api = data.get("api")
                dashboard_dir = child / "dashboard"
                safe_api = _safe_plugin_api_relpath(raw_api, dashboard_dir=dashboard_dir)
                if raw_api and safe_api is None:
                    _log.warning(
                        "Plugin %s: refusing unsafe api path %r (must be a "
                        "relative file inside the plugin's dashboard/ "
                        "directory); backend routes from this plugin will "
                        "not be mounted",
                        name, raw_api,
                    )
                plugins.append({
                    "name": name,
                    "label": data.get("label", name),
                    "description": data.get("description", ""),
                    "icon": data.get("icon", "Puzzle"),
                    "version": data.get("version", "0.0.0"),
                    "tab": tab_info,
                    "slots": slots,
                    "entry": data.get("entry", "dist/index.js"),
                    "css": data.get("css"),
                    "has_api": bool(safe_api),
                    "source": source,
                    "_dir": str(dashboard_dir),
                    "_api_file": safe_api,
                })
            except Exception as exc:
                _log.warning("Bad dashboard plugin manifest %s: %s", manifest_file, exc)
                continue
    return plugins


# Cache discovered plugins per-process (refresh on explicit re-scan).
_dashboard_plugins_cache: Optional[list] = None


def _get_dashboard_plugins(force_rescan: bool = False) -> list:
    global _dashboard_plugins_cache
    if _dashboard_plugins_cache is None or force_rescan:
        _dashboard_plugins_cache = _discover_dashboard_plugins()
    elif _dashboard_plugins_cache:
        if any(not Path(p["_dir"]).is_dir() for p in _dashboard_plugins_cache):
            _dashboard_plugins_cache = _discover_dashboard_plugins()
    return _dashboard_plugins_cache


@app.get("/api/dashboard/plugins")
async def get_dashboard_plugins():
    """Return discovered dashboard plugins (excludes user-hidden and non-enabled ones)."""
    plugins = _get_dashboard_plugins()
    # Read user's hidden plugins list from config.
    config = load_config()
    hidden: list = cfg_get(config, "dashboard", "hidden_plugins", default=[]) or []
    # Gate: only serve user plugins that are in plugins.enabled and not
    # in plugins.disabled.  This prevents the frontend from loading JS/CSS
    # from plugins the user has not explicitly activated.  (#46435)
    try:
        from hermes_cli.plugins_cmd import _get_enabled_set, _get_disabled_set
        enabled_set = _get_enabled_set()
        disabled_set = _get_disabled_set()
    except Exception:
        enabled_set = set()
        disabled_set = set()

    def _is_active(p: dict) -> bool:
        name = p.get("name", "")
        if name in hidden:
            return False
        if p.get("source") == "user":
            if name in disabled_set:
                return False
            if name not in enabled_set:
                return False
        elif p.get("source") == "bundled":
            if name in disabled_set:
                return False
        return True

    # Strip internal fields before sending to frontend.
    return [
        {k: v for k, v in p.items() if not k.startswith("_")}
        for p in plugins
        if _is_active(p)
    ]


@app.get("/api/dashboard/plugins/rescan")
async def rescan_dashboard_plugins():
    """Force re-scan of dashboard plugins."""
    plugins = _get_dashboard_plugins(force_rescan=True)
    return {"ok": True, "count": len(plugins)}


class _AgentPluginInstallBody(BaseModel):
    identifier: str
    force: bool = False
    enable: bool = True


def _strip_dashboard_manifest(p: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in p.items() if not k.startswith("_")}


def _merged_plugins_hub() -> Dict[str, Any]:
    """Agent discovery + dashboard manifests + optional provider picker metadata."""
    from hermes_cli.plugins_cmd import (
        _discover_all_plugins,
        _get_current_context_engine,
        _get_current_memory_provider,
        _discover_context_engines,
        _discover_memory_providers,
        _get_disabled_set,
        _get_enabled_set,
        _read_manifest as _read_plugin_manifest_at,
    )

    dashboard_list = _get_dashboard_plugins()
    dash_by_name = {str(p["name"]): p for p in dashboard_list}

    disabled_set = _get_disabled_set()
    enabled_set = _get_enabled_set()

    # Read user-hidden plugins from config for the user_hidden field.
    config = load_config()
    hidden_plugins: list = cfg_get(config, "dashboard", "hidden_plugins", default=[]) or []

    plugins_root_resolved = (get_hermes_home() / "plugins").resolve()
    rows: List[Dict[str, Any]] = []

    for name, version, description, source, dir_str, key in _discover_all_plugins():
        # Both the path-derived key (nested category plugins) and the bare
        # manifest name count for enabled/disabled state, matching the runtime
        # loader's back-compat lookup.
        aliases = {name}
        if key:
            aliases.add(key)
        if aliases & disabled_set:
            runtime_status = "disabled"
        elif aliases & enabled_set:
            runtime_status = "enabled"
        else:
            runtime_status = "inactive"

        dir_path = Path(dir_str)
        dm = dash_by_name.get(name)
        has_dash_manifest = dm is not None or (dir_path / "dashboard" / "manifest.json").exists()

        under_user_tree = False
        try:
            dir_path.resolve().relative_to(plugins_root_resolved)
            under_user_tree = True
        except ValueError:
            pass

        can_remove_update = (
            source in {"user", "git"} and under_user_tree and Path(dir_str).is_dir()
        )

        # Check if this plugin provides tools that require auth
        auth_required = False
        auth_command = ""
        manifest_data = _read_plugin_manifest_at(dir_path)
        provides_tools = manifest_data.get("provides_tools") or []
        if provides_tools:
            try:
                from tools.registry import registry
                for tname in provides_tools:
                    entry = registry.get_entry(tname)
                    if entry and entry.check_fn and not entry.check_fn():
                        auth_required = True
                        auth_command = f"hermes auth {name}"
                        break
            except Exception:
                pass

        rows.append({
            "name": name,
            "version": version or "",
            "description": description or "",
            "source": source,
            "runtime_status": runtime_status,
            "has_dashboard_manifest": has_dash_manifest,
            "dashboard_manifest": _strip_dashboard_manifest(dm) if dm else None,
            "path": dir_str,
            "can_remove": can_remove_update,
            "can_update_git": can_remove_update and (Path(dir_str) / ".git").exists(),
            "auth_required": auth_required,
            "auth_command": auth_command,
            "user_hidden": name in hidden_plugins,
        })

    agent_names = {r["name"] for r in rows}
    orphan_dashboard = [
        _strip_dashboard_manifest(p)
        for p in dashboard_list
        if str(p["name"]) not in agent_names
    ]

    memory_providers: List[Dict[str, str]] = []
    try:
        for n, desc in _discover_memory_providers():
            memory_providers.append({"name": n, "description": desc})
    except Exception:
        memory_providers = []

    context_engines: List[Dict[str, str]] = []
    try:
        for n, desc in _discover_context_engines():
            context_engines.append({"name": n, "description": desc})
    except Exception:
        context_engines = []

    return {
        "plugins": rows,
        "orphan_dashboard_plugins": orphan_dashboard,
        "providers": {
            "memory_provider": _get_current_memory_provider() or "",
            "memory_options": memory_providers,
            "context_engine": _get_current_context_engine(),
            "context_options": context_engines,
        },
    }


@app.get("/api/dashboard/plugins/hub")
async def get_plugins_hub(request: Request):
    """Unified agent plugins + dashboard extension metadata (session protected)."""
    _require_token(request)
    try:
        return _merged_plugins_hub()
    except Exception as exc:
        _log.warning("plugins/hub failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to build plugins hub.") from exc


@app.post("/api/dashboard/agent-plugins/install")
async def post_agent_plugin_install(request: Request, body: _AgentPluginInstallBody):
    _require_token(request)
    from hermes_cli.plugins_cmd import dashboard_install_plugin

    result = dashboard_install_plugin(
        body.identifier.strip(),
        force=body.force,
        enable=body.enable,
    )
    if not result.get("ok"):
        raise HTTPException(
            status_code=400,
            detail=result.get("error") or "Install failed.",
        )
    _get_dashboard_plugins(force_rescan=True)
    # Strip internal paths from the response
    result.pop("after_install_path", None)
    return result


def _validate_plugin_name(name: str) -> str:
    """Reject path-traversal attempts in plugin name URL parameters."""
    name = name.strip("/")
    if not name or ".." in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid plugin name.")
    return name


@app.post("/api/dashboard/agent-plugins/{name:path}/enable")
async def post_agent_plugin_enable(request: Request, name: str):
    _require_token(request)
    name = _validate_plugin_name(name)
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

    result = dashboard_set_agent_plugin_enabled(name, enabled=True)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or "Enable failed.")
    return result


@app.post("/api/dashboard/agent-plugins/{name:path}/disable")
async def post_agent_plugin_disable(request: Request, name: str):
    _require_token(request)
    name = _validate_plugin_name(name)
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

    result = dashboard_set_agent_plugin_enabled(name, enabled=False)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or "Disable failed.")
    return result


@app.post("/api/dashboard/agent-plugins/{name:path}/update")
async def post_agent_plugin_update(request: Request, name: str):
    _require_token(request)
    name = _validate_plugin_name(name)
    from hermes_cli.plugins_cmd import dashboard_update_user_plugin

    result = dashboard_update_user_plugin(name)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or "Update failed.")
    _get_dashboard_plugins(force_rescan=True)
    return result


@app.delete("/api/dashboard/agent-plugins/{name:path}")
async def delete_agent_plugin(request: Request, name: str):
    _require_token(request)
    name = _validate_plugin_name(name)
    from hermes_cli.plugins_cmd import dashboard_remove_user_plugin

    result = dashboard_remove_user_plugin(name)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or "Remove failed.")
    _get_dashboard_plugins(force_rescan=True)
    return result


class _PluginProvidersPutBody(BaseModel):
    memory_provider: Optional[str] = None
    context_engine: Optional[str] = None


@app.put("/api/dashboard/plugin-providers")
async def put_plugin_providers(request: Request, body: _PluginProvidersPutBody):
    """Persist memory provider / context engine selection (writes config.yaml)."""
    _require_token(request)
    from hermes_cli.plugins_cmd import (
        _save_context_engine,
        _save_memory_provider,
    )

    if body.memory_provider is not None:
        _save_memory_provider(body.memory_provider)
    if body.context_engine is not None:
        _save_context_engine(body.context_engine)
    return {"ok": True}


class _PluginVisibilityBody(BaseModel):
    hidden: bool


@app.post("/api/dashboard/plugins/{name:path}/visibility")
async def post_plugin_visibility(request: Request, name: str, body: _PluginVisibilityBody):
    """Toggle a plugin's sidebar visibility (persists to config.yaml dashboard.hidden_plugins)."""
    _require_token(request)
    name = _validate_plugin_name(name)

    config = load_config()
    if "dashboard" not in config or not isinstance(config.get("dashboard"), dict):
        config["dashboard"] = {}
    hidden_list: list = config["dashboard"].get("hidden_plugins") or []
    if not isinstance(hidden_list, list):
        hidden_list = []

    if body.hidden and name not in hidden_list:
        hidden_list.append(name)
    elif not body.hidden and name in hidden_list:
        hidden_list.remove(name)

    config["dashboard"]["hidden_plugins"] = hidden_list
    save_config(config)
    return {"ok": True, "name": name, "hidden": body.hidden}


@app.get("/dashboard-plugins/{plugin_name}/{file_path:path}")
async def serve_plugin_asset(plugin_name: str, file_path: str):
    """Serve static assets from a dashboard plugin directory.

    Only serves files from the plugin's ``dashboard/`` subdirectory.
    Path traversal is blocked by checking ``resolve().is_relative_to()``.

    Restricted to a browser-fetchable suffix allowlist (JS/CSS/JSON/HTML/
    SVG/PNG/JPG/WOFF). The dashboard loads plugin JS via ``<script src>``
    and CSS via ``<link href>``, neither of which can attach a custom
    auth header — so this route stays unauthenticated to keep the SPA
    working. But user-installed plugins ship a ``plugin_api.py``
    backend module that the browser never fetches; it's only imported
    by :func:`_mount_plugin_api_routes` at startup. Without a suffix
    allowlist, anyone on the loopback port can curl the ``.py`` source
    of a private third-party plugin. Reject everything outside the
    browser-asset set.

    User plugins must be in plugins.enabled before their assets are
    served. (#46435, GHSA-mcfc-hp25-cjv7)
    """
    plugins = _get_dashboard_plugins()
    plugin = next((p for p in plugins if p["name"] == plugin_name), None)
    if not plugin:
        raise HTTPException(status_code=404, detail="Plugin not found")

    # Gate: user plugins must be enabled to serve assets;
    # bundled plugins must not be explicitly disabled.
    try:
        from hermes_cli.plugins_cmd import _get_enabled_set, _get_disabled_set
        enabled_set = _get_enabled_set()
        disabled_set = _get_disabled_set()
    except Exception:
        enabled_set = set()
        disabled_set = set()
    if plugin.get("source") == "user":
        if plugin_name in disabled_set or plugin_name not in enabled_set:
            raise HTTPException(status_code=404, detail="Plugin not found")
    elif plugin.get("source") == "bundled":
        if plugin_name in disabled_set:
            raise HTTPException(status_code=404, detail="Plugin not found")

    base = Path(plugin["_dir"])
    target = (base / file_path).resolve()

    if not target.is_relative_to(base.resolve()):
        raise HTTPException(status_code=403, detail="Path traversal blocked")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Browser-asset suffix allowlist. Everything outside this set is
    # rejected with 404 so we don't leak ``.py`` backend sources, README
    # files, ``.env.example`` templates, etc. — none of which the SPA
    # actually fetches. Add to this set deliberately when a new asset
    # type comes up; do NOT change the default fallback.
    suffix = target.suffix.lower()
    content_types = {
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".html": "text/html",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".ico": "image/x-icon",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
        ".ttf": "font/ttf",
        ".otf": "font/otf",
        ".map": "application/json",
    }
    if suffix not in content_types:
        raise HTTPException(
            status_code=404,
            detail="File not found",
        )
    media_type = content_types[suffix]
    return FileResponse(
        target,
        media_type=media_type,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


def _mount_plugin_api_routes():
    """Import and mount backend API routes from plugins that declare them.

    Each plugin's ``api`` field points to a Python file that must expose
    a ``router`` (FastAPI APIRouter).  Routes are mounted under
    ``/api/plugins/<name>/``.

    Backend import is restricted to ``bundled`` and ``user`` sources.
    Project plugins (``./.hermes/plugins/``) ship with the CWD and are
    therefore attacker-controlled in any threat model where the user
    opens a malicious repo; they can extend the dashboard UI via
    static JS/CSS but their Python ``api`` file is never auto-imported
    by the web server.  See GHSA-5qr3-c538-wm9j (#29156).

    Additionally, user plugins must be explicitly enabled via the
    ``plugins.enabled`` allow-list in config.yaml before their backend
    code is imported. Without this gate, an installed-but-not-enabled
    plugin's Python code would execute at dashboard startup — a code
    execution vector that bypasses the user's intent. (#46435,
    GHSA-mcfc-hp25-cjv7)
    """
    # Load the enabled/disabled sets once for the loop.
    try:
        from hermes_cli.plugins_cmd import _get_enabled_set, _get_disabled_set
        enabled_set = _get_enabled_set()
        disabled_set = _get_disabled_set()
    except Exception:
        enabled_set = set()
        disabled_set = set()

    for plugin in _get_dashboard_plugins():
        api_file_name = plugin.get("_api_file")
        if not api_file_name:
            continue
        plugin_name = plugin.get("name", "")
        # Gate: user plugins must be in plugins.enabled and not in
        # plugins.disabled before we import their Python code.
        # Bundled plugins are trusted (they ship with the release) but
        # still respect an explicit disable.
        if plugin.get("source") == "user":
            if plugin_name in disabled_set:
                _log.debug(
                    "Plugin %s: skipping API mount (explicitly disabled)",
                    plugin_name,
                )
                continue
            if plugin_name not in enabled_set:
                _log.debug(
                    "Plugin %s: skipping API mount (not in plugins.enabled)",
                    plugin_name,
                )
                continue
        elif plugin.get("source") == "bundled":
            if plugin_name in disabled_set:
                _log.debug(
                    "Plugin %s: skipping API mount (explicitly disabled)",
                    plugin_name,
                )
                continue
        if plugin.get("source") == "project":
            _log.warning(
                "Plugin %s: ignoring backend api=%s (project plugins may "
                "not auto-import Python code; move the plugin to "
                "~/.hermes/plugins/ if you trust it)",
                plugin["name"], api_file_name,
            )
            continue
        dashboard_dir = Path(plugin["_dir"])
        api_path = dashboard_dir / api_file_name
        try:
            resolved_api = api_path.resolve()
            resolved_base = dashboard_dir.resolve()
            resolved_api.relative_to(resolved_base)
        except (OSError, RuntimeError, ValueError):
            # Discovery already filters this, but re-check here in case
            # ``_dir`` was tampered with after caching or a future caller
            # bypasses the validator.  Defence in depth keeps the import
            # primitive contained even if the upstream check regresses.
            _log.warning(
                "Plugin %s: refusing to import api file outside its "
                "dashboard directory (%s)", plugin["name"], api_path,
            )
            continue
        if not api_path.exists():
            _log.warning("Plugin %s declares api=%s but file not found", plugin["name"], api_file_name)
            continue
        try:
            module_name = f"hermes_dashboard_plugin_{plugin['name']}"
            spec = importlib.util.spec_from_file_location(module_name, api_path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            # Register in sys.modules BEFORE exec_module so pydantic/FastAPI
            # can resolve forward references (e.g. models defined in a file
            # that uses `from __future__ import annotations`). Without this,
            # TypeAdapter lazy-build fails at first request with
            # "is not fully defined" because the module namespace isn't
            # reachable by name for string-annotation resolution.
            sys.modules[module_name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
            router = getattr(mod, "router", None)
            if router is None:
                _log.warning("Plugin %s api file has no 'router' attribute", plugin["name"])
                continue
            app.include_router(router, prefix=f"/api/plugins/{plugin['name']}")
            _log.info("Mounted plugin API routes: /api/plugins/%s/", plugin["name"])
        except Exception as exc:
            _log.warning("Failed to load plugin %s API routes: %s", plugin["name"], exc)


# Mount plugin API routes before the SPA catch-all.
_mount_plugin_api_routes()

# Mount the dashboard auth routes (/login, /auth/*, /api/auth/*) before the
# SPA catch-all so /{full_path:path} doesn't swallow them.  These are
# always mounted — the gate middleware decides whether to enforce auth,
# not whether the routes exist.
from hermes_cli.dashboard_auth.routes import router as _dashboard_auth_router  # noqa: E402
app.include_router(_dashboard_auth_router)

mount_spa(app)


def _read_bound_port(server: "uvicorn.Server", fallback: int) -> int:
    """Read the OS-assigned port from a live uvicorn server socket.

    After ``server.startup()`` the socket is bound.  Returns the actual
    port so ephemeral (port-0) discovery works without a pre-bind TOCTOU.
    Falls back to *fallback* if the socket list is empty (shouldn't happen
    but guards against uvicorn internals changing).
    """
    if server.servers and server.servers[0].sockets:
        return server.servers[0].sockets[0].getsockname()[1]
    return fallback


def _write_dashboard_ready_file(actual_port: int) -> None:
    """Optionally publish the dashboard port through an atomic ready file.

    Windows Desktop can launch dashboard backends with ``pythonw.exe`` to avoid
    console flashes. That path cannot rely on stdout for the port announcement,
    so Electron passes ``HERMES_DESKTOP_READY_FILE`` and waits for this JSON.
    Normal CLI/dashboard launches still use the stdout READY line below.
    """
    target = os.environ.get("HERMES_DESKTOP_READY_FILE")
    if not target:
        return

    tmp_name = ""
    try:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"port": int(actual_port)}, separators=(",", ":"))
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
            tmp_name = fh.name
        os.replace(tmp_name, path)
    except Exception as exc:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except Exception:
                pass
        _log.warning("Failed to write dashboard ready file %r: %s", target, exc)


def _maybe_open_browser(
    host: str, actual_port: int, open_browser: bool, initial_profile: str
) -> None:
    """Open the dashboard URL in the user's browser if appropriate.

    Skips on headless Linux (no ``DISPLAY`` / ``WAYLAND_DISPLAY``) to avoid
    TUI browsers (links, lynx) that would SIGHUP the server process.
    Maps ``0.0.0.0`` / ``::`` binds to ``127.0.0.1`` so the browser opens
    a reachable URL.
    """
    if not open_browser:
        return

    import webbrowser

    _has_display = (
        sys.platform != "linux"
        or bool(os.environ.get("DISPLAY"))
        or bool(os.environ.get("WAYLAND_DISPLAY"))
    )
    if not _has_display:
        _log.debug(
            "Skipping browser-open: no DISPLAY or WAYLAND_DISPLAY detected "
            "(headless Linux). Pass --no-open to suppress this detection."
        )
        return

    _display_host = host if host not in ("0.0.0.0", "::") else "127.0.0.1"
    _open_url = f"http://{_display_host}:{actual_port}"
    if initial_profile:
        from urllib.parse import quote
        _open_url += f"/?profile={quote(initial_profile)}"

    def _open():
        try:
            time.sleep(1.0)
            webbrowser.open(_open_url)
        except Exception:
            pass

    threading.Thread(target=_open, daemon=True).start()


def start_server(
    host: str = "127.0.0.1",
    port: int = 9119,
    open_browser: bool = True,
    allow_public: bool = False,
    initial_profile: str = "",
):
    """Start the web UI server.

    ``initial_profile`` (when set) is appended to the auto-opened browser
    URL as ``?profile=<name>`` so the SPA's profile switcher preselects it
    — used when a profile alias (``<profile> dashboard``) routes to the
    machine dashboard.
    """
    import uvicorn

    try:
        from hermes_cli.nous_auth_keepalive import start_nous_auth_keepalive

        start_nous_auth_keepalive()
    except Exception as exc:
        _log.debug("Nous auth keepalive did not start: %s", exc)

    # Phase 0: stash the auth-gate flag on app.state so middleware / SPA-token
    # injection / WS-auth paths can branch on it consistently.  Phase 3.5
    # uses this to decide whether to refuse the bind, log the gate-on
    # banner, and enable uvicorn proxy_headers.
    app.state.auth_required = should_require_auth(host)

    # ``--insecure`` no longer disables the auth gate (June 2026 hardening:
    # the hermes-0day MCP-persistence campaign abused unauthenticated public
    # dashboards). If a caller still passes it, warn that it is now a no-op
    # rather than silently changing their expectation of an open bind.
    if allow_public and host not in _LOOPBACK_HOST_VALUES:
        _log.warning(
            "--insecure no longer bypasses dashboard authentication. A "
            "non-loopback bind (%s) now ALWAYS requires an auth provider "
            "(OAuth or the bundled password provider). Configure one — see "
            "below — or bind to 127.0.0.1 and reach it over an SSH tunnel / "
            "Tailscale.", host,
        )

    if app.state.auth_required:
        # The gate engages on every non-loopback bind. Require at least one
        # provider to be registered, else fail closed — there is no longer an
        # escape hatch that serves the dashboard without authentication.
        from hermes_cli.dashboard_auth import list_providers
        if not list_providers():
            # Surface the *specific* reason any bundled provider declined
            # to register (e.g. missing HERMES_DASHBOARD_OAUTH_CLIENT_ID).
            # Each provider plugin that ships with Hermes Agent exposes a
            # module-level ``LAST_SKIP_REASON`` string for this purpose;
            # without it the operator would only see "no providers" which
            # is misleading when the provider IS installed but unconfigured.
            skip_reasons: list[str] = []
            try:
                from plugins.dashboard_auth import nous as _nous_plugin

                if _nous_plugin.LAST_SKIP_REASON:
                    skip_reasons.append(
                        f"  • nous: {_nous_plugin.LAST_SKIP_REASON}"
                    )
            except Exception:
                pass

            _fix_hint = (
                "Configure an auth provider before exposing the dashboard:\n"
                "  • Password: set dashboard.basic_auth.username + "
                "password_hash in config.yaml\n"
                "    (hash with: python -c \"from "
                "plugins.dashboard_auth.basic import hash_password; "
                "print(hash_password('your-password'))\")\n"
                "  • OAuth: run `hermes dashboard register` (Nous Portal) or "
                "install a DashboardAuthProvider plugin.\n"
                "There is no unauthenticated public-bind option — to keep it "
                "local, bind 127.0.0.1 and tunnel in (SSH / Tailscale)."
            )
            if skip_reasons:
                raise SystemExit(
                    f"Refusing to bind dashboard to {host} — the auth gate "
                    f"engages on non-loopback binds, but no auth providers "
                    f"are registered.\n\n"
                    f"Bundled providers reported these issues:\n"
                    + "\n".join(skip_reasons)
                    + "\n\n"
                    + _fix_hint
                )
            raise SystemExit(
                f"Refusing to bind dashboard to {host} — the auth gate "
                f"engages on non-loopback binds, but no auth providers are "
                f"registered.\n\n" + _fix_hint
            )
        _log.info(
            "Dashboard binding to %s with auth gate enabled. Providers: %s",
            host,
            ", ".join(p.name for p in list_providers()),
        )

    # Record the bound host so host_header_middleware can validate incoming
    # Host headers against it. Defends against DNS rebinding (GHSA-ppp5-vxwm-4cf7).
    app.state.bound_host = host

    # ── Start uvicorn with direct Server API ─────────────────────────
    # We use uvicorn.Server directly (not uvicorn.run) so we can split
    # startup from the main loop.  After startup() the socket is actually
    # bound — we read the OS-assigned port from the live socket, print
    # HERMES_DASHBOARD_READY, open the browser, *then* serve.
    #
    # This eliminates the TOCTOU of the old pre-bind-then-close approach
    # (bind port 0 → close → uvicorn rebind): the socket is held by
    # uvicorn the entire time, so no other process can steal the port.
    #
    # For explicit non-zero ports, if the port is taken uvicorn catches
    # OSError inside create_server() and exits with a clear error — no
    # separate preflight probe needed.
    # Loopback binds are the Desktop case: a single local client, no reverse
    # proxy in front. A GIL-heavy agent turn can stall the event loop past 20s,
    # and uvicorn's ws keepalive ping runs on that same starved loop — so a
    # 20s ping timeout kills an otherwise-healthy local connection over a
    # recoverable stall (QW-1). Give loopback a longer 60s timeout / 30s
    # interval to ride out those stalls. Non-loopback binds sit behind a
    # Cloudflare Tunnel (idle timeout ~100s), so keep them at 20/20 to detect
    # half-open connections promptly and stay under the tunnel's idle window.
    _is_loopback = host in ("127.0.0.1", "localhost", "::1")
    config = uvicorn.Config(
        app, host=host, port=port, log_level="warning",
        # proxy_headers defaults to False so _ws_client_is_allowed sees
        # the real connection peer rather than X-Forwarded-For's rewritten
        # value (which would defeat the loopback gate when behind a reverse
        # proxy).  When the OAuth gate is active we are explicitly running
        # behind a TLS terminator (Fly.io) and need X-Forwarded-Proto to
        # decide cookie Secure flags, so we flip proxy_headers on for that
        # mode.
        proxy_headers=bool(app.state.auth_required),
        # Detect half-open WS connections (reverse-proxy 524, dropped
        # tunnels) within ~20-40s so WebSocketDisconnect fires the
        # disconnect→reap path.  20s stays under Cloudflare Tunnel's idle
        # timeout, keeping it warm.  Loopback gets a longer window (see above).
        ws_ping_interval=30.0 if _is_loopback else 20.0,
        ws_ping_timeout=60.0 if _is_loopback else 20.0,
    )
    server = uvicorn.Server(config)

    async def _serve():
        # Split startup from main_loop so we can read the bound port
        # after the socket is live (ephemeral port discovery).
        if not config.loaded:
            config.load()
        server.lifespan = config.lifespan_class(config)
        with server.capture_signals():
            await server.startup()
            if server.should_exit:
                return

            actual_port = _read_bound_port(server, fallback=port)
            app.state.bound_port = actual_port

            _write_dashboard_ready_file(actual_port)
            print(f"HERMES_DASHBOARD_READY port={actual_port}", flush=True)
            print(f"  Hermes Web UI → http://{host}:{actual_port}")
            _maybe_open_browser(host, actual_port, open_browser, initial_profile)

            # Collapse the peer-hangup teardown flood (#50005). When the Desktop
            # forcibly closes its WebSocket mid-write, asyncio logs a full
            # traceback per pending connection-lost callback — 50+ identical
            # WinError 10054 (ConnectionResetError) lines per disconnect on
            # Windows. This filter downgrades exactly that class to one debug
            # line and passes every other loop error through unchanged.
            try:
                from tui_gateway.loop_noise import install_loop_noise_filter

                install_loop_noise_filter(asyncio.get_running_loop())
            except Exception as exc:  # pragma: no cover - best-effort
                _log.debug("loop noise filter install skipped: %s", exc)

            # ── Loop heartbeat watchdog (CF-1) ───────────────────────────
            # Confirm the GIL-pressure hypothesis in production. Re-arm a 2s
            # tick and measure the drift between when it *should* fire and
            # when it actually does: a healthy loop drifts ~0, but a turn that
            # holds the GIL blocks the loop and the next tick fires late by the
            # stall duration. We log that so a stalled-loop WS drop is
            # diagnosable from the gateway log. Uses loop.time() (monotonic)
            # for drift, and call_later (not a task) so it dies with the loop —
            # nothing to cancel on shutdown.
            _hb_interval = 2.0
            _hb_stall_threshold = 5.0
            _hb_loop = asyncio.get_running_loop()

            def _loop_heartbeat(expected: float) -> None:
                now = _hb_loop.time()
                drift = now - expected
                if drift > _hb_stall_threshold:
                    _log.warning(
                        "event loop stalled %.1fs (GIL pressure suspected)",
                        drift,
                    )
                _hb_loop.call_later(
                    _hb_interval, _loop_heartbeat, now + _hb_interval
                )

            _hb_loop.call_later(
                _hb_interval, _loop_heartbeat, _hb_loop.time() + _hb_interval
            )

            await server.main_loop()
            if server.started:
                await server.shutdown()

    # On POSIX, keep the long-standing ``asyncio.run(_serve())`` behavior
    # unchanged — Python's default loop there is already a SelectorEventLoop
    # (or uvloop when uvicorn[standard] installs it), which is exactly what
    # uvicorn serves on. Touching that path would only widen the blast radius
    # for no benefit.
    #
    # On Windows it is broken: ``asyncio.run`` defaults to a ProactorEventLoop,
    # but uvicorn's socket-serving stack assumes a SelectorEventLoop on win32
    # (``uvicorn/loops/asyncio.py`` forces it, and ``uvicorn.Server.run`` threads
    # ``config.get_loop_factory()`` into its runner for exactly this reason).
    # Driving uvicorn on the proactor loop makes ``server.startup()`` bind a
    # socket that never accepts — the dashboard / desktop backend prints
    # "Skipping web UI build" and then hangs forever with the port LISTENING but
    # no TCP handshake completing (#50641). So *only on Windows* we mirror
    # uvicorn's own machinery and run on the loop factory it picks.
    if sys.platform != "win32":
        asyncio.run(_serve())
        return

    # Windows-only path. Resolve the runner + loop factory FIRST (and fall back
    # to a hand-installed Windows selector policy only when uvicorn predates the
    # loop-factory API, < 0.36). The actual serve call is then OUTSIDE the
    # try/except so genuine serve-time errors (port in use, KeyboardInterrupt)
    # propagate normally instead of being swallowed and double-run.
    try:
        from uvicorn._compat import asyncio_run as _runner

        _loop_factory = config.get_loop_factory()
    except Exception:
        _runner = None
        _loop_factory = None
        try:
            asyncio.set_event_loop_policy(
                asyncio.WindowsSelectorEventLoopPolicy()  # type: ignore[attr-defined]
            )
        except Exception:
            pass

    if _runner is not None:
        _runner(_serve(), loop_factory=_loop_factory)
    else:
        asyncio.run(_serve())
