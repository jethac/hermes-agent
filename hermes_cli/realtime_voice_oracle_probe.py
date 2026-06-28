"""Probe an OpenAI-compatible Hermes oracle endpoint for voice readiness."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_PROMPT = (
    "You are Hermes's local oracle. In one short paragraph, explain your role "
    "in a KAME-style realtime voice session."
)
DEFAULT_SYSTEM_PROMPT = "You are a concise local Hermes oracle benchmark."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe an OpenAI-compatible Hermes oracle endpoint")
    parser.add_argument("--output", required=True, help="JSON file to write with probe results")
    parser.add_argument("--base-url", default=os.environ.get("DGX_SPARK_ORACLE_BASE_URL", ""))
    parser.add_argument("--model", default=os.environ.get("DGX_SPARK_ORACLE_MODEL", ""))
    parser.add_argument("--api-key", default=os.environ.get("DGX_SPARK_ORACLE_API_KEY", ""))
    parser.add_argument("--prompt", default=os.environ.get("DGX_SPARK_ORACLE_PROMPT", DEFAULT_PROMPT))
    parser.add_argument("--system-prompt", default=os.environ.get("DGX_SPARK_ORACLE_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("DGX_SPARK_ORACLE_MAX_TOKENS", "220")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("DGX_SPARK_ORACLE_TEMPERATURE", "0.2")))
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("DGX_SPARK_ORACLE_TIMEOUT_SECONDS", "120")),
        help="HTTP request timeout in seconds",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not str(args.base_url or "").strip() or not str(args.model or "").strip():
        result = {
            "ok": False,
            "error": "--base-url and --model are required",
            "base_url": str(args.base_url or ""),
            "model": str(args.model or ""),
        }
        _write_json(Path(args.output), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2

    result = probe_openai_compatible_oracle(
        base_url=str(args.base_url),
        model=str(args.model),
        api_key=str(args.api_key or ""),
        prompt=str(args.prompt),
        system_prompt=str(args.system_prompt),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        timeout_seconds=float(args.timeout),
    )
    _write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") is True else 1


def probe_openai_compatible_oracle(
    *,
    base_url: str,
    model: str,
    api_key: str = "",
    prompt: str = DEFAULT_PROMPT,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int = 220,
    temperature: float = 0.2,
    timeout_seconds: float = 120.0,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    """Call one non-streaming chat completion and return latency/usage evidence."""

    endpoint = chat_completions_url(base_url)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = build_chat_completion_payload(
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    started = time.perf_counter()
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        body = exc.read()
        status = int(exc.code)
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "base_url": base_url.rstrip("/"),
            "endpoint": endpoint,
            "model": model,
        }

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    data = _decode_json_body(body)
    content = _completion_content(data)
    usage = data.get("usage") if isinstance(data, dict) else {}
    completion_tokens = int((usage or {}).get("completion_tokens") or 0)
    tokens_per_second = completion_tokens / (elapsed_ms / 1000.0) if completion_tokens else None
    return {
        "ok": 200 <= status < 300,
        "status": status,
        "base_url": base_url.rstrip("/"),
        "endpoint": endpoint,
        "model": model,
        "elapsed_ms": round(elapsed_ms, 2),
        "completion_tokens": completion_tokens,
        "tokens_per_second": round(tokens_per_second, 2) if tokens_per_second else None,
        "content_preview": content[:1000],
        "usage": usage,
    }


def chat_completions_url(base_url: str) -> str:
    """Return a chat-completions URL for root or /v1 OpenAI-compatible base URLs."""

    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        raise ValueError("base_url is required")
    parsed = urllib.parse.urlsplit(normalized)
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        return normalized
    suffix = "/chat/completions" if path.endswith("/v1") else "/v1/chat/completions"
    return normalized + suffix


def build_chat_completion_payload(
    *,
    model: str,
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_tokens: int = 220,
    temperature: float = 0.2,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }


def _decode_json_body(body: bytes) -> dict[str, Any]:
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        return {"raw": body.decode("utf-8", errors="replace")[:4000]}
    return data if isinstance(data, dict) else {"raw": data}


def _completion_content(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        return ""
    return str(content or "")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
