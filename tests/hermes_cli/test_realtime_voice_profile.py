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
    assert profile["barge_in_min_rms"] == 350
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


def test_elevenlabs_preset_prints_applyable_portable_profile(capsys):
    result = realtime_voice_profile.main(["--preset", "elevenlabs"])

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["streaming_stt_base_url"] == "http://127.0.0.1:8767"
    assert realtime["streaming_tts_base_url"] == "http://127.0.0.1:8767"
    assert realtime["streaming_stt_model"] == "scribe_v2_realtime"
    assert realtime["streaming_tts_model"] == "eleven_flash_v2_5"
    assert realtime["streaming_stt_token_env"] == "HERMES_STREAMING_STT_BRIDGE_TOKEN"
    assert realtime["streaming_tts_token_env"] == "HERMES_STREAMING_STT_BRIDGE_TOKEN"
    assert realtime["require_live_like"] is True


def test_elevenlabs_preset_accepts_custom_bridge_base_url(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "elevenlabs",
            "--bridge-base-url",
            "http://voice-bridge.local:8767/",
        ]
    )

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["streaming_stt_base_url"] == "http://voice-bridge.local:8767"
    assert realtime["streaming_tts_base_url"] == "http://voice-bridge.local:8767"


def test_cartesia_preset_prints_applyable_portable_profile(capsys):
    result = realtime_voice_profile.main(["--preset", "cartesia"])

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["streaming_stt_base_url"] == "http://127.0.0.1:8769"
    assert realtime["streaming_tts_base_url"] == "http://127.0.0.1:8769"
    assert realtime["streaming_stt_model"] == "ink-2"
    assert realtime["streaming_tts_model"] == "sonic-3.5"
    assert realtime["streaming_stt_token_env"] == "HERMES_STREAMING_STT_BRIDGE_TOKEN"
    assert realtime["streaming_tts_token_env"] == "HERMES_STREAMING_STT_BRIDGE_TOKEN"
    assert realtime["require_live_like"] is True


def test_cartesia_preset_accepts_custom_bridge_base_url(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "cartesia",
            "--bridge-base-url",
            "http://voice-bridge.local:8769/",
        ]
    )

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["streaming_stt_base_url"] == "http://voice-bridge.local:8769"
    assert realtime["streaming_tts_base_url"] == "http://voice-bridge.local:8769"


def test_openai_preset_prints_managed_realtime_profile(capsys):
    result = realtime_voice_profile.main(["--preset", "openai"])

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["frontend_provider"] == "openai_realtime"
    assert realtime["frontend_model"] == "gpt-realtime-2"
    assert realtime["openai_realtime_api_key_env"] == "OPENAI_API_KEY"
    assert realtime["openai_realtime_base_url"] == "wss://api.openai.com/v1/realtime"
    assert realtime["openai_realtime_voice"] == "marin"
    assert realtime["openai_realtime_transcription_model"] == "gpt-realtime-whisper"
    assert realtime["sidecar_autostart"] is True
    assert realtime["require_live_like"] is True
    assert "streaming_stt_base_url" not in realtime


def test_openai_preset_accepts_explicit_model_voice_and_key_env(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "openai",
            "--openai-realtime-model",
            "gpt-realtime-2",
            "--openai-realtime-voice",
            "cedar",
            "--openai-realtime-transcription-model",
            "gpt-realtime-whisper",
            "--openai-realtime-api-key-env",
            "HERMES_OPENAI_KEY",
        ]
    )

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["frontend_model"] == "gpt-realtime-2"
    assert realtime["openai_realtime_voice"] == "cedar"
    assert realtime["openai_realtime_transcription_model"] == "gpt-realtime-whisper"
    assert realtime["openai_realtime_api_key_env"] == "HERMES_OPENAI_KEY"


def test_gemini_preset_prints_managed_realtime_profile(capsys):
    result = realtime_voice_profile.main(["--preset", "gemini"])

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["frontend_provider"] == "gemini_live"
    assert realtime["frontend_model"] == "gemini-3.1-flash-live-preview"
    assert realtime["gemini_live_api_key_env"] == "GEMINI_API_KEY"
    assert realtime["gemini_live_voice"] == "Puck"
    assert realtime["gemini_live_google_search"] is False
    assert realtime["gemini_live_oracle_tool"] is True
    assert realtime["sidecar_autostart"] is True
    assert realtime["require_live_like"] is True
    assert "streaming_stt_base_url" not in realtime


def test_gemini_preset_accepts_explicit_model_voice_key_env_and_tools(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "gemini",
            "--gemini-live-model",
            "gemini-3.1-flash-live-preview",
            "--gemini-live-voice",
            "Kore",
            "--gemini-live-api-key-env",
            "HERMES_GEMINI_KEY",
            "--gemini-live-google-search",
            "--disable-gemini-live-oracle-tool",
        ]
    )

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["frontend_model"] == "gemini-3.1-flash-live-preview"
    assert realtime["gemini_live_voice"] == "Kore"
    assert realtime["gemini_live_api_key_env"] == "HERMES_GEMINI_KEY"
    assert realtime["gemini_live_google_search"] is True
    assert realtime["gemini_live_oracle_tool"] is False


def test_kame_preset_prints_reflex_oracle_profile(capsys):
    result = realtime_voice_profile.main(["--preset", "kame"])

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["engine"] == "kame_interface_oracle"
    assert realtime["frontend_provider"] == "gemma4"
    assert realtime["frontend_model"] == "gemma-4-E2B-it"
    assert realtime["interface_audio_input"] == "auto"
    assert realtime["asr_mode"] == "on_escalation"
    assert realtime["preferred_local_oracle_model"] == "gemma-4-26B-A4B-it"
    assert realtime["oracle_timeout_seconds"] == 60.0
    assert realtime["max_spoken_sentences"] == 2
    assert realtime["sidecar_autostart"] is True
    assert realtime["require_live_like"] is True
    assert realtime["routing"] == {
        "allow_local_greetings": True,
        "allow_local_clarifications": True,
        "require_oracle_for_tools": True,
        "require_oracle_for_memory": True,
        "require_oracle_for_files": True,
        "local_confidence_threshold": 0.75,
    }
    assert realtime["metrics"] == {
        "enabled": True,
        "log_turn_spans": True,
        "log_provider_spans": True,
    }


def test_kame_profile_merge_copies_discord_scoped_runtime_fields():
    profile = realtime_voice_profile.build_kame_realtime_voice_profile(
        reflex_model="gemma-4-E2B-it",
        interface_audio_input="native_audio",
        asr_mode="speculative",
        preferred_local_oracle_model="gemma-4-26B-A4B-it",
    )

    merged = realtime_voice_profile.merge_realtime_voice_profile({}, profile)
    discord_rt = merged["discord"]["realtime_voice"]

    assert discord_rt["engine"] == "kame_interface_oracle"
    assert discord_rt["frontend_provider"] == "gemma4"
    assert discord_rt["frontend_model"] == "gemma-4-E2B-it"
    assert discord_rt["interface_audio_input"] == "native_audio"
    assert discord_rt["asr_mode"] == "speculative"
    assert discord_rt["preferred_local_oracle_model"] == "gemma-4-26B-A4B-it"
    assert discord_rt["oracle_timeout_seconds"] == 60.0
    assert discord_rt["max_spoken_sentences"] == 2
    assert discord_rt["routing"]["require_oracle_for_tools"] is True
    assert discord_rt["metrics"]["log_turn_spans"] is True


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
    assert merged["discord"]["realtime_voice"]["enabled"] is True
    assert merged["discord"]["realtime_voice"]["sidecar_base_url"] == "http://127.0.0.1:8765"
    assert merged["discord"]["realtime_voice"]["sidecar_token_env"] == "HERMES_VOICE_SIDECAR_TOKEN"


def test_merge_realtime_voice_profile_replaces_stale_discord_provider_bridge_url():
    existing = {
        "discord": {
            "realtime_voice": {
                "enabled": True,
                "sidecar_base_url": "http://127.0.0.1:8769",
                "sidecar_token": "existing-token",
            },
        },
    }
    profile = realtime_voice_profile.build_realtime_voice_live_like_profile(
        streaming_stt_base_url="http://127.0.0.1:8769",
        streaming_tts_base_url="http://127.0.0.1:8769",
        sidecar_host="127.0.0.1",
        sidecar_port=8765,
    )

    merged = realtime_voice_profile.merge_realtime_voice_profile(existing, profile)

    discord_rt = merged["discord"]["realtime_voice"]
    assert discord_rt["sidecar_base_url"] == "http://127.0.0.1:8765"
    assert discord_rt["sidecar_token"] == "existing-token"
    assert merged["voice"]["realtime"]["streaming_stt_base_url"] == "http://127.0.0.1:8769"


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
    assert "realtime_voice_live_evidence --require-live-discord --require-openai-realtime" in output


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


def test_elevenlabs_preset_apply_prints_bridge_next_steps(monkeypatch, tmp_path, capsys):
    saved = {}
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"model": {"provider": "openrouter"}},
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.setdefault("config", cfg))
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: tmp_path / "config.yaml")

    result = realtime_voice_profile.main(["--preset", "elevenlabs", "--apply"])

    assert result == 0
    realtime = saved["config"]["voice"]["realtime"]
    assert realtime["streaming_stt_base_url"] == "http://127.0.0.1:8767"
    assert realtime["streaming_tts_base_url"] == "http://127.0.0.1:8767"
    output = capsys.readouterr().out
    assert "realtime_voice_elevenlabs_bridge --generate-token" in output
    assert "realtime_voice_elevenlabs_bridge --check --strict --production-en-ja" in output
    assert "realtime_voice_elevenlabs_bridge --host 127.0.0.1 --port 8767 --production-en-ja" in output
    assert "realtime_voice_alpha_evidence --runs 3 --apply --provider elevenlabs --start-bridge" in output
    assert "realtime_voice_live_evidence --require-live-discord --require-openai-realtime" in output


def test_cartesia_preset_apply_prints_bridge_next_steps(monkeypatch, tmp_path, capsys):
    saved = {}
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"model": {"provider": "openrouter"}},
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.setdefault("config", cfg))
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: tmp_path / "config.yaml")

    result = realtime_voice_profile.main(["--preset", "cartesia", "--apply"])

    assert result == 0
    realtime = saved["config"]["voice"]["realtime"]
    assert realtime["streaming_stt_base_url"] == "http://127.0.0.1:8769"
    assert realtime["streaming_tts_base_url"] == "http://127.0.0.1:8769"
    assert realtime["barge_in_min_rms"] == 350
    output = capsys.readouterr().out
    assert "realtime_voice_cartesia_bridge --generate-token" in output
    assert "realtime_voice_cartesia_bridge --check --strict --production-en-ja" in output
    assert "realtime_voice_cartesia_bridge --host 127.0.0.1 --port 8769 --production-en-ja" in output
    assert "realtime_voice_alpha_evidence --runs 3 --apply --provider cartesia --start-bridge" in output
    assert "realtime_voice_live_evidence --require-live-discord --require-openai-realtime" in output


def test_openai_preset_apply_prints_live_evidence_next_steps(monkeypatch, tmp_path, capsys):
    saved = {}
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"model": {"provider": "openrouter"}},
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.setdefault("config", cfg))
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: tmp_path / "config.yaml")

    result = realtime_voice_profile.main(["--preset", "openai", "--apply"])

    assert result == 0
    realtime = saved["config"]["voice"]["realtime"]
    assert realtime["frontend_provider"] == "openai_realtime"
    assert realtime["barge_in_min_rms"] == 350
    output = capsys.readouterr().out
    assert "export OPENAI_API_KEY=..." in output
    assert "export DISCORD_BOT_TOKEN=... DISCORD_GUILD_ID=... DISCORD_VOICE_CHANNEL_ID=..." in output
    assert "realtime_voice_sidecar --host 127.0.0.1 --port 8765" in output
    assert "realtime_voice_live_evidence --require-live-discord --require-openai-realtime" in output


def test_gemini_preset_apply_prints_live_evidence_next_steps(monkeypatch, tmp_path, capsys):
    saved = {}
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"model": {"provider": "openrouter"}},
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.setdefault("config", cfg))
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: tmp_path / "config.yaml")

    result = realtime_voice_profile.main(["--preset", "gemini", "--apply"])

    assert result == 0
    realtime = saved["config"]["voice"]["realtime"]
    assert realtime["frontend_provider"] == "gemini_live"
    assert realtime["barge_in_min_rms"] == 350
    output = capsys.readouterr().out
    assert "export GEMINI_API_KEY=..." in output
    assert "export DISCORD_BOT_TOKEN=... DISCORD_GUILD_ID=... DISCORD_VOICE_CHANNEL_ID=..." in output
    assert "realtime_voice_sidecar --host 127.0.0.1 --port 8765" in output
    assert "realtime_voice_live_evidence --require-live-discord --require-gemini-live" in output
    assert "realtime_voice_live_evidence --require-live-discord --require-openai-realtime" not in output
