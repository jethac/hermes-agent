import sys
from types import SimpleNamespace

from hermes_cli import realtime_voice_sidecar


def test_sidecar_parser_prefers_canonical_interface_env(monkeypatch):
    monkeypatch.setenv("HERMES_KAME_INTERFACE_BASE_URL", "http://interface.local:8000/v1")
    monkeypatch.setenv("HERMES_VOICE_VLLM_BASE_URL", "http://legacy.local:8000/v1")

    args = realtime_voice_sidecar.build_parser().parse_args([])

    assert args.interface_base_url == "http://interface.local:8000/v1"
    assert args.vllm_base_url == "http://legacy.local:8000/v1"


def test_sidecar_main_mirrors_interface_base_url_to_legacy_runtime_env(monkeypatch):
    captured = {}

    def fake_runtime_config_from_env():
        captured["interface_base_url"] = __import__("os").environ.get("HERMES_KAME_INTERFACE_BASE_URL")
        captured["vllm_base_url"] = __import__("os").environ.get("HERMES_VOICE_VLLM_BASE_URL")
        return SimpleNamespace()

    def fake_create_reference_sidecar_app(runtime):
        captured["runtime"] = runtime
        return "app"

    fake_uvicorn = SimpleNamespace(
        run=lambda app, host, port: captured.update({"app": app, "host": host, "port": port})
    )
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr(realtime_voice_sidecar, "runtime_config_from_env", fake_runtime_config_from_env)
    monkeypatch.setattr(realtime_voice_sidecar, "create_reference_sidecar_app", fake_create_reference_sidecar_app)

    realtime_voice_sidecar.main(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "9876",
            "--interface-base-url",
            "http://interface.local:8000/v1",
            "--vllm-base-url",
            "http://legacy.local:8000/v1",
        ]
    )

    assert captured["interface_base_url"] == "http://interface.local:8000/v1"
    assert captured["vllm_base_url"] == "http://interface.local:8000/v1"
    assert captured["app"] == "app"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9876
