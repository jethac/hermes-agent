from hermes_cli.model_router import (
    is_heavy_request,
    moa_config_for_request,
    normalize_model_router_config,
)


def test_normalize_model_router_defaults_disabled_heavy_moa():
    cfg = normalize_model_router_config({})

    assert cfg["enabled"] is False
    assert cfg["mode"] == "heavy_moa"
    assert cfg["heavy_moa_preset"] == "gemma-nemotron"
    assert cfg["exclude_prefixes"] == ["/"]
    assert "debug" in cfg["heavy_keywords"]


def test_is_heavy_request_uses_keywords_and_length():
    router = {"enabled": True, "heavy_keywords": ["audit"], "heavy_min_chars": 20}

    assert is_heavy_request("please audit this", router) is True
    assert is_heavy_request("x" * 20, router) is True
    assert is_heavy_request("hello", router) is False
    assert is_heavy_request("please audit this", {"enabled": False}) is False
    assert is_heavy_request("/audit this", router) is False


def test_moa_config_for_request_returns_named_heavy_preset():
    cfg = {
        "model_router": {
            "enabled": True,
            "heavy_moa_preset": "gemma-nemotron",
            "heavy_keywords": ["heavy"],
        },
        "moa": {
            "presets": {
                "gemma-nemotron": {
                    "reference_models": [
                        {
                            "provider": "custom",
                            "model": "nemotron-3-nano-oracle",
                            "base_url": "http://pgx.local:8003/v1",
                        }
                    ],
                    "aggregator": {
                        "provider": "custom",
                        "model": "gemma-4-12b-oracle",
                        "base_url": "http://pgx.local:8002/v1",
                    },
                    "max_tokens": 2048,
                }
            }
        },
    }

    routed = moa_config_for_request(cfg, "heavy planning request")

    assert routed is not None
    assert routed["reference_models"][0]["model"] == "nemotron-3-nano-oracle"
    assert routed["reference_models"][0]["base_url"] == "http://pgx.local:8003/v1"
    assert routed["aggregator"]["model"] == "gemma-4-12b-oracle"


def test_moa_config_for_request_ignores_light_turns():
    cfg = {
        "model_router": {"enabled": True, "heavy_keywords": ["audit"]},
        "moa": {"presets": {"gemma-nemotron": {}}},
    }

    assert moa_config_for_request(cfg, "hello") is None
