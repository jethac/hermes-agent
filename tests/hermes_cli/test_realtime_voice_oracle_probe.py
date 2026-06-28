import json

from hermes_cli import realtime_voice_oracle_probe


class _FakeResponse:
    status = 200

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._body


def test_chat_completions_url_accepts_root_v1_or_full_endpoint():
    assert (
        realtime_voice_oracle_probe.chat_completions_url("http://spark.local:8000")
        == "http://spark.local:8000/v1/chat/completions"
    )
    assert (
        realtime_voice_oracle_probe.chat_completions_url("http://spark.local:8000/v1")
        == "http://spark.local:8000/v1/chat/completions"
    )
    assert (
        realtime_voice_oracle_probe.chat_completions_url("http://spark.local:8000/v1/chat/completions")
        == "http://spark.local:8000/v1/chat/completions"
    )


def test_probe_openai_compatible_oracle_writes_latency_and_usage():
    calls = []

    def fake_urlopen(request, *, timeout):
        calls.append((request, timeout, json.loads(request.data.decode("utf-8"))))
        return _FakeResponse(
            {
                "choices": [{"message": {"content": "I am the local Hermes oracle."}}],
                "usage": {"completion_tokens": 12},
            }
        )

    result = realtime_voice_oracle_probe.probe_openai_compatible_oracle(
        base_url="http://spark.local:8000/v1",
        model="gemma-4-26B-A4B-it",
        api_key="spark-token",
        prompt="say hello",
        max_tokens=42,
        timeout_seconds=7.0,
        urlopen=fake_urlopen,
    )

    request, timeout, payload = calls[0]
    assert result["ok"] is True
    assert result["endpoint"] == "http://spark.local:8000/v1/chat/completions"
    assert result["content_preview"] == "I am the local Hermes oracle."
    assert result["completion_tokens"] == 12
    assert result["tokens_per_second"] is not None
    assert timeout == 7.0
    assert request.get_header("Authorization") == "Bearer spark-token"
    assert payload["model"] == "gemma-4-26B-A4B-it"
    assert payload["messages"][-1]["content"] == "say hello"
    assert payload["max_tokens"] == 42
    assert payload["stream"] is False


def test_main_requires_base_url_and_model(tmp_path):
    output = tmp_path / "oracle-probe.json"

    result = realtime_voice_oracle_probe.main(["--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result == 2
    assert payload["ok"] is False
    assert "--base-url and --model are required" in payload["error"]
