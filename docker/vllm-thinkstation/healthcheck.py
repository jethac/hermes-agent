#!/usr/bin/env python3
"""Small dependency-free healthcheck for vLLM OpenAI-compatible servers."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: healthcheck.py URL", file=sys.stderr)
        return 2

    request = urllib.request.Request(sys.argv[1], headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status != 200:
                print(f"unexpected HTTP status: {response.status}", file=sys.stderr)
                return 1
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"healthcheck failed: {exc}", file=sys.stderr)
        return 1

    models = payload.get("data")
    if not isinstance(models, list) or not models:
        print("no models reported", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
