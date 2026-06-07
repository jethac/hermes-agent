import yaml

from hermes_cli import realtime_voice_profile


def test_live_like_profile_contains_portable_streaming_bridge_config():
    profile = realtime_voice_profile.build_realtime_voice_live_like_profile(
        streaming_stt_base_url="http://streaming-stt.local:9000/",
        streaming_tts_base_url="http://streaming-tts.local:9001/",
        streaming_stt_model="nova-3",
        streaming_tts_model="aura-2-thalia-en",
    )

    assert profile["enabled"] is True
    assert profile["engine"] == "text_oracle_tts"
    assert profile["frontend_provider"] == "reference"
    assert profile["require_live_like"] is True
    assert profile["sidecar_autostart"] is True
    assert profile["streaming_stt_base_url"] == "http://streaming-stt.local:9000"
    assert profile["streaming_tts_base_url"] == "http://streaming-tts.local:9001"
    assert profile["streaming_stt_token_env"] == "HERMES_STREAMING_STT_BRIDGE_TOKEN"
    assert profile["streaming_tts_token_env"] == "HERMES_STREAMING_STT_BRIDGE_TOKEN"
    assert profile["production_languages"] == ["en", "ja"]
    assert profile["production_scripts"] == ["Latn", "Jpan"]
    assert profile["best_effort_languages"] is True
    assert profile["quality_targets_ms"]["audio_to_partial_transcript_ms"] == 300
    assert profile["production_evidence_min_runs"] == 3


def test_live_like_profile_requires_streaming_urls_for_applyable_profile():
    result = realtime_voice_profile.main(["--apply"])

    assert result == 1


def test_live_like_profile_can_print_template_with_placeholder_urls(capsys):
    result = realtime_voice_profile.main(["--allow-template-urls"])

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["streaming_stt_base_url"] == "http://127.0.0.1:8766"
    assert realtime["streaming_tts_base_url"] == "http://127.0.0.1:8766"


def test_deepgram_preset_prints_applyable_portable_profile(capsys):
    result = realtime_voice_profile.main(["--preset", "deepgram"])

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["streaming_stt_base_url"] == "http://127.0.0.1:8766"
    assert realtime["streaming_tts_base_url"] == "http://127.0.0.1:8766"
    assert realtime["streaming_stt_model"] == "nova-3"
    assert realtime["streaming_tts_model"] == "aura-2-thalia-en"
    assert realtime["streaming_stt_token_env"] == "HERMES_STREAMING_STT_BRIDGE_TOKEN"
    assert realtime["streaming_tts_token_env"] == "HERMES_STREAMING_STT_BRIDGE_TOKEN"
    assert realtime["require_live_like"] is True


def test_deepgram_preset_accepts_custom_bridge_base_url(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "deepgram",
            "--bridge-base-url",
            "http://voice-bridge.local:8766/",
        ]
    )

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["streaming_stt_base_url"] == "http://voice-bridge.local:8766"
    assert realtime["streaming_tts_base_url"] == "http://voice-bridge.local:8766"


def test_deepgram_preset_preserves_explicit_streaming_models(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "deepgram",
            "--streaming-stt-model",
            "nova-3-medical",
            "--streaming-tts-model",
            "custom-voice",
        ]
    )

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["streaming_stt_model"] == "nova-3-medical"
    assert realtime["streaming_tts_model"] == "custom-voice"


def test_merge_realtime_voice_profile_preserves_unrelated_config():
    existing = {
        "model": {"provider": "openrouter", "default": "gpt-5"},
        "voice": {
            "record_key": "ctrl+b",
            "realtime": {
                "enabled": False,
                "sidecar_base_url": "http://old.example.test",
            },
        },
    }
    profile = realtime_voice_profile.build_realtime_voice_live_like_profile(
        streaming_stt_base_url="http://streaming.local:8766",
        streaming_tts_base_url="http://streaming.local:8766",
    )

    merged = realtime_voice_profile.merge_realtime_voice_profile(existing, profile)

    assert merged["model"] == existing["model"]
    assert merged["voice"]["record_key"] == "ctrl+b"
    assert merged["voice"]["realtime"]["enabled"] is True
    assert merged["voice"]["realtime"]["sidecar_base_url"] == ""
    assert merged["voice"]["realtime"]["streaming_stt_base_url"] == "http://streaming.local:8766"


def test_apply_realtime_voice_profile_saves_merged_config(monkeypatch, tmp_path):
    saved = {}
    profile = realtime_voice_profile.build_realtime_voice_live_like_profile(
        streaming_stt_base_url="http://streaming.local:8766",
        streaming_tts_base_url="http://streaming.local:8766",
    )

    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"model": {"provider": "openrouter"}},
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.setdefault("config", cfg))
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: tmp_path / "config.yaml")

    path = realtime_voice_profile.apply_realtime_voice_profile(profile)

    assert path == tmp_path / "config.yaml"
    assert saved["config"]["model"]["provider"] == "openrouter"
    assert saved["config"]["voice"]["realtime"]["require_live_like"] is True


def test_deepgram_preset_apply_prints_bridge_next_steps(monkeypatch, tmp_path, capsys):
    saved = {}
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"model": {"provider": "openrouter"}},
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.setdefault("config", cfg))
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: tmp_path / "config.yaml")

    result = realtime_voice_profile.main(["--preset", "deepgram", "--apply"])

    assert result == 0
    realtime = saved["config"]["voice"]["realtime"]
    assert realtime["streaming_stt_base_url"] == "http://127.0.0.1:8766"
    assert realtime["streaming_tts_base_url"] == "http://127.0.0.1:8766"
    output = capsys.readouterr().out
    assert "realtime_voice_deepgram_bridge --generate-token" in output
    assert "realtime_voice_deepgram_bridge --check --strict" in output
    assert "--production-en-ja" in output
    assert "realtime_voice_deepgram_bridge --host 127.0.0.1 --port 8766 --production-en-ja" in output
    assert "realtime_voice_alpha_evidence --runs 3 --apply" in output


def test_deepgram_preset_apply_can_generate_bridge_token(monkeypatch, tmp_path, capsys):
    saved = {}
    env = {}
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"model": {"provider": "openrouter"}},
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.setdefault("config", cfg))
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: tmp_path / "config.yaml")
    monkeypatch.setattr("hermes_cli.config.load_env", lambda: dict(env))
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda key, value: env.setdefault(key, value))

    result = realtime_voice_profile.main(["--preset", "deepgram", "--apply", "--generate-bridge-token"])

    assert result == 0
    assert "HERMES_STREAMING_STT_BRIDGE_TOKEN" in env
    assert len(env["HERMES_STREAMING_STT_BRIDGE_TOKEN"]) >= 32
    output = capsys.readouterr().out
    assert "Generated realtime voice bridge token in HERMES_STREAMING_STT_BRIDGE_TOKEN" in output
    assert "realtime_voice_deepgram_bridge --generate-token" not in output
    assert env["HERMES_STREAMING_STT_BRIDGE_TOKEN"] not in output


def test_deepgram_preset_apply_persists_custom_bridge_token_env(monkeypatch, tmp_path, capsys):
    saved = {}
    env = {}
    writes = {}
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"model": {"provider": "openrouter"}},
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.setdefault("config", cfg))
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: tmp_path / "config.yaml")
    monkeypatch.setattr("hermes_cli.config.load_env", lambda: dict(env))
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda key, value: writes.setdefault(key, value))

    result = realtime_voice_profile.main(
        [
            "--preset",
            "deepgram",
            "--apply",
            "--generate-bridge-token",
            "--streaming-stt-token-env",
            "CUSTOM_BRIDGE_TOKEN",
            "--streaming-tts-token-env",
            "CUSTOM_BRIDGE_TOKEN",
        ]
    )

    assert result == 0
    assert "CUSTOM_BRIDGE_TOKEN" in writes
    assert writes["HERMES_DEEPGRAM_BRIDGE_TOKEN_ENV"] == "CUSTOM_BRIDGE_TOKEN"
    assert "CUSTOM_BRIDGE_TOKEN" == saved["config"]["voice"]["realtime"]["streaming_stt_token_env"]
    output = capsys.readouterr().out
    assert writes["CUSTOM_BRIDGE_TOKEN"] not in output


def test_deepgram_preset_apply_does_not_overwrite_existing_bridge_token(monkeypatch, tmp_path, capsys):
    saved = {}
    env = {"HERMES_STREAMING_STT_BRIDGE_TOKEN": "existing-token"}
    writes = []
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"model": {"provider": "openrouter"}},
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.setdefault("config", cfg))
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: tmp_path / "config.yaml")
    monkeypatch.setattr("hermes_cli.config.load_env", lambda: dict(env))
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda key, value: writes.append((key, value)))

    result = realtime_voice_profile.main(["--preset", "deepgram", "--apply", "--generate-bridge-token"])

    assert result == 0
    assert writes == []
    assert "already configured in HERMES_STREAMING_STT_BRIDGE_TOKEN" in capsys.readouterr().out
