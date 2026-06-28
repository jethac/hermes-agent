"""Local DGX speech proxy bridge for Hermes realtime voice.

This module exposes the Hermes streaming STT/TTS websocket contract while
fronting a local speech service that already speaks the same contract. It is
used by vendor-named DGX entrypoints such as Nemotron Speech and Magpie without
hard-coding undocumented vendor wire protocols into Hermes.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from starlette.requests import Request
from starlette.websockets import WebSocket, WebSocketDisconnect

from agent.realtime_voice import (
    RealtimeVoiceSessionConfig,
    VoiceEvent,
    VoiceEventType,
    binary_audio_frame_from_event,
    event_from_binary_audio_frame,
)
from agent.realtime_voice_errors import sanitize_realtime_voice_error
from agent.realtime_voice_sidecar import RealtimeVoiceSidecarClient


LOCAL_SPEECH_BRIDGE_ROLES = frozenset({"stt", "tts"})


@dataclass(frozen=True)
class LocalSpeechProxyBridgeConfig:
    """Runtime settings for a local Hermes-compatible speech proxy."""

    provider: str
    role: str
    model: str
    upstream_base_url: Optional[str] = None
    upstream_token: Optional[str] = None
    auth_token: Optional[str] = None
    connect_timeout_seconds: float = 10.0
    health_timeout_seconds: float = 0.5
    input_languages: tuple[str, ...] = ()
    output_languages: tuple[str, ...] = ()

    @property
    def session_path(self) -> str:
        if self.role == "stt":
            return "/v1/streaming-stt/session"
        if self.role == "tts":
            return "/v1/streaming-tts/session"
        raise ValueError(f"unsupported local speech bridge role: {self.role}")


HealthProbe = Callable[[LocalSpeechProxyBridgeConfig], Mapping[str, Any]]
ClientFactory = Callable[..., RealtimeVoiceSidecarClient]


def create_local_speech_proxy_bridge_app(
    runtime: LocalSpeechProxyBridgeConfig,
    *,
    health_probe: Optional[HealthProbe] = None,
    client_factory: Optional[ClientFactory] = None,
):
    """Create a Hermes-compatible local speech proxy app."""

    from fastapi import FastAPI, HTTPException

    if runtime.role not in LOCAL_SPEECH_BRIDGE_ROLES:
        raise ValueError(f"unsupported local speech bridge role: {runtime.role}")

    app = FastAPI(title=f"Hermes {runtime.provider} local speech bridge")
    probe = health_probe or probe_local_speech_upstream_health
    factory = client_factory or RealtimeVoiceSidecarClient

    @app.get("/health")
    async def health(request: Request):
        if not _authorized(request.headers, runtime.auth_token):
            raise HTTPException(status_code=401, detail="unauthorized")
        upstream_health = probe(runtime)
        return local_speech_proxy_health_payload(runtime, upstream_health=upstream_health)

    @app.websocket(runtime.session_path)
    async def streaming_session(ws: WebSocket):
        if not _authorized(ws.headers, runtime.auth_token):
            await ws.close(code=1008, reason="unauthorized")
            return
        if not runtime.upstream_base_url:
            await ws.close(code=1011, reason="local speech upstream unavailable")
            return
        await _run_proxy_session(ws, runtime, client_factory=factory)

    return app


def local_speech_proxy_health_payload(
    runtime: LocalSpeechProxyBridgeConfig,
    *,
    upstream_health: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    upstream = dict(upstream_health or {})
    upstream_ok = upstream.get("ok") is True
    role_ready = _upstream_supports_role(runtime, upstream)
    ready = bool(runtime.upstream_base_url and upstream_ok and role_ready)
    payload: dict[str, Any] = {
        "ok": ready,
        "kind": "local_speech_proxy_bridge",
        "frontend": {
            "provider": runtime.provider,
            "model": runtime.model,
            "role": runtime.role,
            "upstream_configured": bool(runtime.upstream_base_url),
            "upstream_healthy": upstream_ok,
        },
        "capabilities": {
            "native_s2s": False,
        },
    }
    if runtime.role == "stt":
        payload["capabilities"]["streaming_stt"] = ready
        payload["capabilities"]["utterance_stt"] = ready
        if runtime.input_languages:
            payload["capabilities"]["input_languages"] = list(runtime.input_languages)
            payload["frontend"]["languages"] = list(runtime.input_languages)
    if runtime.role == "tts":
        payload["capabilities"]["tts"] = ready
        payload["capabilities"]["streaming_tts"] = ready
        if runtime.output_languages:
            payload["capabilities"]["output_languages"] = list(runtime.output_languages)
            payload["frontend"]["tts_model_languages"] = list(runtime.output_languages)
    if upstream:
        payload["upstream"] = _sanitized_upstream_health(upstream)
    return payload


def probe_local_speech_upstream_health(runtime: LocalSpeechProxyBridgeConfig) -> Mapping[str, Any]:
    if not runtime.upstream_base_url:
        return {"ok": False, "error": "upstream_base_url_not_configured"}
    url = f"{runtime.upstream_base_url.rstrip('/')}/health"
    headers = {}
    if runtime.upstream_token:
        headers["Authorization"] = f"Bearer {runtime.upstream_token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=max(0.1, runtime.health_timeout_seconds)) as response:
            body = response.read()
            status = response.status
    except Exception as exc:
        return {"ok": False, "error": sanitize_realtime_voice_error(exc)}
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as exc:
        return {"ok": False, "status": status, "error": f"invalid_json: {sanitize_realtime_voice_error(exc)}"}
    if not isinstance(payload, Mapping):
        return {"ok": False, "status": status, "error": "health payload is not an object"}
    return dict(payload) | {"status": status}


def local_speech_proxy_prerequisite_issues(
    runtime: LocalSpeechProxyBridgeConfig,
    *,
    require_auth_token: bool = False,
    upstream_health: Optional[Mapping[str, Any]] = None,
) -> list[str]:
    issues: list[str] = []
    if runtime.role not in LOCAL_SPEECH_BRIDGE_ROLES:
        issues.append(f"unsupported role: {runtime.role}")
    if not runtime.upstream_base_url:
        issues.append(f"{_env_prefix(runtime)}_UPSTREAM_BASE_URL is required")
    if require_auth_token and not runtime.auth_token:
        issues.append(f"{_env_prefix(runtime)}_BRIDGE_TOKEN is required")
    if upstream_health is not None:
        if upstream_health.get("ok") is not True:
            issues.append("upstream health is not ok")
        if not _upstream_supports_role(runtime, upstream_health):
            issues.append(f"upstream does not advertise required {runtime.role} capability")
    return issues


def local_speech_proxy_config_from_env(
    *,
    provider: str,
    role: str,
    default_model: str,
    env_prefix: str,
    default_input_languages: tuple[str, ...] = (),
    default_output_languages: tuple[str, ...] = (),
) -> LocalSpeechProxyBridgeConfig:
    return LocalSpeechProxyBridgeConfig(
        provider=provider,
        role=role,
        model=os.environ.get(f"{env_prefix}_MODEL") or default_model,
        upstream_base_url=os.environ.get(f"{env_prefix}_UPSTREAM_BASE_URL") or None,
        upstream_token=os.environ.get(f"{env_prefix}_UPSTREAM_TOKEN") or None,
        auth_token=os.environ.get(f"{env_prefix}_BRIDGE_TOKEN")
        or os.environ.get("HERMES_STREAMING_STT_BRIDGE_TOKEN")
        or None,
        connect_timeout_seconds=float(os.environ.get(f"{env_prefix}_CONNECT_TIMEOUT_SECONDS") or 10.0),
        health_timeout_seconds=float(os.environ.get(f"{env_prefix}_HEALTH_TIMEOUT_SECONDS") or 0.5),
        input_languages=tuple(
            _parse_languages(os.environ.get(f"{env_prefix}_INPUT_LANGUAGES") or ",".join(default_input_languages))
        ),
        output_languages=tuple(
            _parse_languages(os.environ.get(f"{env_prefix}_OUTPUT_LANGUAGES") or ",".join(default_output_languages))
        ),
    )


async def _run_proxy_session(
    ws: WebSocket,
    runtime: LocalSpeechProxyBridgeConfig,
    *,
    client_factory: ClientFactory,
) -> None:
    await ws.accept()
    client: Optional[RealtimeVoiceSidecarClient] = None
    pump_task: Optional[asyncio.Task[None]] = None

    async def pump_events() -> None:
        assert client is not None
        async for event in client.events():
            frame = binary_audio_frame_from_event(event)
            if frame is not None and event.type == VoiceEventType.AUDIO_OUTPUT_CHUNK:
                await ws.send_bytes(frame)
                continue
            await ws.send_json(event.to_wire())

    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(code=int(message.get("code") or 1000))
            frame = message.get("bytes")
            if isinstance(frame, bytes):
                if runtime.role != "stt":
                    await ws.send_json({"type": "session.error", "payload": {"error": "invalid binary audio frame"}})
                    continue
                if client is None:
                    await ws.send_json({"type": "session.error", "payload": {"error": "missing session.config"}})
                    continue
                try:
                    event = event_from_binary_audio_frame(frame, expected_type=VoiceEventType.AUDIO_INPUT_CHUNK)
                except Exception:
                    await ws.send_json({"type": "session.error", "payload": {"error": "invalid binary audio frame"}})
                    continue
                await client.send_event(event)
                continue
            raw = message.get("text")
            if not isinstance(raw, str):
                await ws.send_json({"type": "session.error", "payload": {"error": "invalid websocket frame"}})
                continue
            data = json.loads(raw)
            if data.get("type") == "session.config":
                config = _downstream_config(runtime, RealtimeVoiceSessionConfig.from_wire(data.get("payload") or {}))
                client = client_factory(path=runtime.session_path)
                try:
                    await client.start(config)
                except Exception as exc:
                    await ws.send_json(
                        {
                            "type": "session.error",
                            "payload": {"error": sanitize_realtime_voice_error(exc)},
                        }
                    )
                    await ws.close(code=1011, reason="local speech upstream unavailable")
                    return
                pump_task = asyncio.create_task(pump_events())
                continue
            if client is None:
                await ws.send_json({"type": "session.error", "payload": {"error": "missing session.config"}})
                continue
            await client.send_event(VoiceEvent.from_wire(data))
    except WebSocketDisconnect:
        pass
    finally:
        if pump_task:
            pump_task.cancel()
        if client is not None:
            await client.close()


def _downstream_config(
    runtime: LocalSpeechProxyBridgeConfig,
    config: RealtimeVoiceSessionConfig,
) -> RealtimeVoiceSessionConfig:
    payload = config.to_wire()
    payload.update(
        {
            "frontend_provider": runtime.provider,
            "frontend_model": runtime.model,
            "sidecar_base_url": runtime.upstream_base_url,
            "sidecar_token": runtime.upstream_token,
            "sidecar_connect_timeout_seconds": runtime.connect_timeout_seconds,
            "spark_base_url": runtime.upstream_base_url,
            "spark_token": runtime.upstream_token,
        }
    )
    if runtime.role == "stt":
        payload["asr_provider"] = runtime.provider
        payload["asr_model"] = runtime.model
    if runtime.role == "tts":
        payload["tts_provider"] = runtime.provider
        payload["tts_model"] = runtime.model
    return RealtimeVoiceSessionConfig.from_wire(payload)


def _upstream_supports_role(runtime: LocalSpeechProxyBridgeConfig, upstream_health: Mapping[str, Any]) -> bool:
    capabilities = upstream_health.get("capabilities")
    if not isinstance(capabilities, Mapping):
        return False
    if runtime.role == "stt":
        return capabilities.get("streaming_stt") is True or capabilities.get("utterance_stt") is True
    if runtime.role == "tts":
        return capabilities.get("streaming_tts") is True or capabilities.get("tts") is True
    return False


def _sanitized_upstream_health(upstream: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in ("ok", "kind", "status", "frontend", "capabilities"):
        if key in upstream:
            safe[key] = upstream[key]
    if upstream.get("error"):
        safe["error"] = sanitize_realtime_voice_error(upstream.get("error"))
    return safe


def _authorized(headers: Mapping[str, str], token: Optional[str]) -> bool:
    if not token:
        return True
    return headers.get("authorization") == f"Bearer {token}"


def _env_prefix(runtime: LocalSpeechProxyBridgeConfig) -> str:
    return "HERMES_" + "".join(ch if ch.isalnum() else "_" for ch in runtime.provider.upper()).strip("_")


def _parse_languages(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").replace(" ", ",").split(",") if part.strip()]
