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
    assert profile["asr_base_url"] == "http://streaming-stt.local:9000"
    assert profile["streaming_tts_base_url"] == "http://streaming-tts.local:9001"
    assert profile["tts_base_url"] == "http://streaming-tts.local:9001"
    assert profile["streaming_stt_token_env"] == "HERMES_STREAMING_STT_BRIDGE_TOKEN"
    assert profile["streaming_tts_token_env"] == "HERMES_STREAMING_TTS_BRIDGE_TOKEN"
    assert profile["production_languages"] == ["en", "ja"]
    assert profile["production_scripts"] == ["Latn", "Jpan"]
    assert profile["best_effort_languages"] is True
    assert profile["interface_temperature"] == 0.2
    assert profile["interface_max_output_tokens"] == 160
    assert profile["interface_timeout_seconds"] == 0.8
    assert profile["barge_in_min_rms"] == 350
    assert profile["barge_in_stop_playback_deadline_ms"] == 150
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
    assert realtime["streaming_tts_token_env"] == "HERMES_STREAMING_TTS_BRIDGE_TOKEN"
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
    assert realtime["streaming_tts_token_env"] == "HERMES_STREAMING_TTS_BRIDGE_TOKEN"
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
    assert realtime["streaming_tts_token_env"] == "HERMES_STREAMING_TTS_BRIDGE_TOKEN"
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


def test_nvidia_speech_preset_prints_applyable_local_speech_profile(capsys):
    result = realtime_voice_profile.main(["--preset", "nvidia_speech"])

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["streaming_stt_base_url"] == "http://127.0.0.1:8767"
    assert realtime["streaming_tts_base_url"] == "http://127.0.0.1:8768"
    assert realtime["streaming_stt_model"] == "nemotron-speech-streaming-0.6b"
    assert realtime["streaming_tts_model"] == "magpie-local-streaming-tts"
    assert realtime["streaming_stt_token_env"] == "HERMES_STREAMING_STT_BRIDGE_TOKEN"
    assert realtime["streaming_tts_token_env"] == "HERMES_STREAMING_TTS_BRIDGE_TOKEN"
    assert realtime["require_live_like"] is True


def test_nvidia_speech_preset_accepts_custom_separate_bridge_urls(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "nvidia_speech",
            "--streaming-stt-base-url",
            "http://spark.local:8767/",
            "--streaming-tts-base-url",
            "http://spark.local:8768/",
            "--streaming-stt-model",
            "nemotron-custom",
            "--streaming-tts-model",
            "magpie-custom",
        ]
    )

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["streaming_stt_base_url"] == "http://spark.local:8767"
    assert realtime["streaming_tts_base_url"] == "http://spark.local:8768"
    assert realtime["streaming_stt_model"] == "nemotron-custom"
    assert realtime["streaming_tts_model"] == "magpie-custom"


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
    assert realtime["interface_api_key_env"] == "HERMES_KAME_INTERFACE_API_KEY"
    assert realtime["interface"]["api_key_env"] == "HERMES_KAME_INTERFACE_API_KEY"
    assert realtime["vllm_model"] == "gemma-4-E2B-it"
    assert realtime["interface_temperature"] == 0.2
    assert realtime["interface_max_output_tokens"] == 160
    assert realtime["interface_timeout_seconds"] == 0.8
    assert realtime["interface_max_audio_seconds"] == 30.0
    assert realtime["asr_mode"] == "from_reflex"
    assert realtime["asr_provider"] == "streaming_stt"
    assert realtime["asr_model"] == "portable-streaming-asr"
    assert realtime["asr_base_url"] == ""
    assert "preferred_local_oracle_model" not in realtime
    assert "oracle_provider" not in realtime
    assert "oracle_provider_name" not in realtime
    assert "oracle_base_url" not in realtime
    assert "oracle_api_mode" not in realtime
    assert realtime["oracle_timeout_seconds"] == 60.0
    assert realtime["max_spoken_sentences"] == 2
    assert realtime["voice_response_policy"] == "sentence_cap"
    assert realtime["tts_provider"] == "streaming_tts"
    assert realtime["tts_model"] == "portable-streaming-voice"
    assert realtime["tts_voice"] == ""
    assert realtime["tts_base_url"] == ""
    assert realtime["fallback_policy"] == "legacy_voice"
    assert realtime["sidecar_autostart"] is True
    assert realtime["require_live_like"] is True
    assert realtime["barge_in_stop_playback_deadline_ms"] == 150
    assert realtime["routing"] == {
        "allow_local_greetings": True,
        "allow_local_clarifications": True,
        "require_oracle_for_tools": True,
        "require_oracle_for_memory": True,
        "require_oracle_for_files": True,
        "local_confidence_threshold": 0.75,
    }
    assert realtime["oracle_jobs"] == {
        "enabled": True,
        "max_concurrent": 4,
        "queue_limit": 16,
        "default_priority": "normal",
        "overflow_policy": "queue",
        "shutdown_timeout_seconds": 2.0,
        "speak_terminal_results": True,
        "audit_ledger_path": "",
    }
    assert realtime["metrics"] == {
        "enabled": True,
        "log_turn_spans": True,
        "log_provider_spans": True,
    }
    assert realtime["turn_acknowledgement"] == {
        "enabled": True,
        "text": "One moment.",
    }
    assert realtime["output_events"] == {
        "caption_aliases": False,
        "audio_aliases": False,
    }
    assert realtime["interface"] == {
        "provider": "gemma4",
        "base_url": "",
        "api_key_env": "HERMES_KAME_INTERFACE_API_KEY",
        "model": "gemma-4-E2B-it",
        "temperature": 0.2,
        "max_output_tokens": 160,
        "timeout_ms": 800,
        "max_audio_seconds": 30.0,
        "audio_input": "auto",
        "asr_mode": "from_reflex",
    }
    assert realtime["oracle"] == {
        "mode": "hermes_active_model",
        "selected_by": "Hermes /model",
        "provider_registration": {
            "enabled": False,
            "provider": "",
            "provider_name": "",
            "preferred_local_model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
            "base_url": "",
            "api_mode": "chat_completions",
            "selection_authority": "Hermes /model",
        },
        "timeout_ms": 60000,
        "max_spoken_sentences": 2,
        "voice_response_policy": "sentence_cap",
    }
    assert realtime["barge_in"] == {
        "min_rms": 350,
        "min_speech_ms": 120,
        "stop_playback_deadline_ms": 150,
    }
    assert realtime["quality_targets_ms"] == {
        "audio_to_partial_transcript_ms": 300,
        "final_transcript_to_first_text_ms": 500,
        "final_transcript_to_first_audio_ms": 900,
        "barge_in_ack_ms": 150,
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
    assert "oracle_base_url" not in realtime


def test_kame_preset_can_print_local_oracle_provider_profile(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "kame",
            "--kame-oracle-base-url",
            "http://spark.local:8001/v1/",
            "--kame-oracle-provider-name",
            "Spark Oracle",
        ]
    )

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert "oracle_provider" not in realtime
    assert "oracle_provider_name" not in realtime
    assert "preferred_local_oracle_model" not in realtime
    assert "oracle_model" not in realtime
    assert "oracle_base_url" not in realtime
    assert "oracle_api_mode" not in realtime
    assert realtime["oracle"]["provider_registration"] == {
        "enabled": True,
        "provider": "custom",
        "provider_name": "Spark Oracle",
        "preferred_local_model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        "base_url": "http://spark.local:8001/v1",
        "api_mode": "chat_completions",
        "selection_authority": "Hermes /model",
    }


def test_kame_preset_can_override_interface_max_audio_seconds(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "kame",
            "--kame-interface-max-audio-seconds",
            "12",
        ]
    )

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    assert data["voice"]["realtime"]["interface_max_audio_seconds"] == 12.0


def test_kame_preset_can_set_interface_runtime_limits(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "kame",
            "--kame-interface-temperature",
            "0.1",
            "--kame-interface-max-output-tokens",
            "96",
            "--kame-interface-timeout-seconds",
            "1.25",
        ]
    )

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["interface_temperature"] == 0.1
    assert realtime["interface_max_output_tokens"] == 96
    assert realtime["interface_timeout_seconds"] == 1.25
    assert realtime["interface"]["temperature"] == 0.1
    assert realtime["interface"]["max_output_tokens"] == 96
    assert realtime["interface"]["timeout_ms"] == 1250


def test_kame_preset_can_set_interface_base_url(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "kame",
            "--kame-interface-base-url",
            "http://spark.local:8000/v1/",
        ]
    )

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["interface_base_url"] == "http://spark.local:8000/v1"
    assert realtime["vllm_base_url"] == "http://spark.local:8000/v1"


def test_kame_preset_can_set_interface_api_key_env(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "kame",
            "--kame-interface-api-key-env",
            "CUSTOM_KAME_INTERFACE_TOKEN",
        ]
    )

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["interface_api_key_env"] == "CUSTOM_KAME_INTERFACE_TOKEN"
    assert realtime["interface"]["api_key_env"] == "CUSTOM_KAME_INTERFACE_TOKEN"


def test_kame_preset_can_set_streaming_tts_voice(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "kame",
            "--streaming-tts-model",
            "magpie-local-streaming-tts",
            "--streaming-tts-voice",
            "spark-voice-1",
        ]
    )

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["tts_model"] == "magpie-local-streaming-tts"
    assert realtime["streaming_tts_model"] == "magpie-local-streaming-tts"
    assert realtime["tts_voice"] == "spark-voice-1"
    assert realtime["streaming_tts_voice"] == "spark-voice-1"


def test_kame_preset_can_set_barge_in_thresholds(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "kame",
            "--kame-barge-in-min-rms",
            "420",
            "--kame-barge-in-min-speech-ms",
            "160",
            "--kame-barge-in-stop-playback-deadline-ms",
            "140",
        ]
    )

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["barge_in_min_rms"] == 420
    assert realtime["barge_in_min_speech_ms"] == 160
    assert realtime["barge_in_stop_playback_deadline_ms"] == 140


def test_kame_preset_can_set_fallback_policy(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "kame",
            "--kame-fallback-policy",
            "fail_closed",
        ]
    )

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    assert data["voice"]["realtime"]["fallback_policy"] == "fail_closed"


def test_kame_preset_can_set_oracle_and_routing_policy(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "kame",
            "--kame-oracle-timeout-seconds",
            "42",
            "--kame-max-spoken-sentences",
            "1",
            "--kame-voice-response-policy",
            "brief_summary",
            "--kame-allow-local-greetings",
            "false",
            "--kame-allow-local-clarifications",
            "false",
            "--kame-require-oracle-for-tools",
            "true",
            "--kame-require-oracle-for-memory",
            "true",
            "--kame-require-oracle-for-files",
            "true",
            "--kame-local-confidence-threshold",
            "0.9",
        ]
    )

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["oracle_timeout_seconds"] == 42
    assert realtime["oracle"]["timeout_ms"] == 42000
    assert realtime["max_spoken_sentences"] == 1
    assert realtime["oracle"]["max_spoken_sentences"] == 1
    assert realtime["voice_response_policy"] == "brief_summary"
    assert realtime["oracle"]["voice_response_policy"] == "brief_summary"
    assert realtime["routing"] == {
        "allow_local_greetings": False,
        "allow_local_clarifications": False,
        "require_oracle_for_tools": True,
        "require_oracle_for_memory": True,
        "require_oracle_for_files": True,
        "local_confidence_threshold": 0.9,
    }


def test_kame_preset_can_disable_metrics(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "kame",
            "--disable-kame-metrics",
            "--disable-kame-turn-span-logs",
            "--disable-kame-provider-span-logs",
        ]
    )

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    assert data["voice"]["realtime"]["metrics"] == {
        "enabled": False,
        "log_turn_spans": False,
        "log_provider_spans": False,
    }


def test_kame_preset_can_set_metrics_booleans(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "kame",
            "--kame-metrics-enabled",
            "false",
            "--kame-log-turn-spans",
            "false",
            "--kame-log-provider-spans",
            "true",
        ]
    )

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    assert data["voice"]["realtime"]["metrics"] == {
        "enabled": False,
        "log_turn_spans": False,
        "log_provider_spans": True,
    }


def test_kame_preset_can_set_provider_labels(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "kame",
            "--kame-interface-provider",
            "openai_compatible",
            "--kame-asr-provider",
            "nemotron_speech",
            "--kame-tts-provider",
            "magpie_tts",
        ]
    )

    assert result == 0
    data = yaml.safe_load(capsys.readouterr().out)
    realtime = data["voice"]["realtime"]
    assert realtime["frontend_provider"] == "openai_compatible"
    assert realtime["asr_provider"] == "nemotron_speech"
    assert realtime["tts_provider"] == "magpie_tts"


def test_kame_preset_rejects_invalid_interface_max_audio_seconds(capsys):
    result = realtime_voice_profile.main(
        [
            "--preset",
            "kame",
            "--kame-interface-max-audio-seconds",
            "31",
        ]
    )

    assert result == 1
    assert "--kame-interface-max-audio-seconds must be between 1 and 30" in capsys.readouterr().err


def test_kame_profile_merge_copies_discord_scoped_runtime_fields():
    profile = realtime_voice_profile.build_kame_realtime_voice_profile(
        reflex_model="gemma-4-E2B-it",
        vllm_model="google/gemma-4-E2B-it",
        interface_api_key_env="CUSTOM_KAME_INTERFACE_TOKEN",
        interface_base_url="http://spark.local:8000/v1",
        interface_temperature=0.3,
        interface_max_output_tokens=96,
        interface_timeout_seconds=0.7,
        interface_audio_input="native_audio",
        asr_mode="speculative",
        streaming_stt_base_url="http://spark.local:8767",
        streaming_stt_token_env="CUSTOM_STT_TOKEN",
        oracle_provider="custom",
        oracle_api_mode="chat_completions",
        oracle_base_url="http://spark.local:8001/v1",
        oracle_provider_name="Spark Oracle",
        preferred_local_oracle_model="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        oracle_timeout_seconds=42,
        max_spoken_sentences=3,
        tts_model="magpie-local-streaming-tts",
        tts_voice="spark-voice-1",
        streaming_tts_base_url="http://spark.local:8768",
        streaming_tts_token_env="CUSTOM_TTS_TOKEN",
        barge_in_min_rms=410,
        barge_in_min_speech_ms=130,
    )

    merged = realtime_voice_profile.merge_realtime_voice_profile({}, profile)
    discord_rt = merged["discord"]["realtime_voice"]

    assert discord_rt["engine"] == "kame_interface_oracle"
    assert discord_rt["frontend_provider"] == "gemma4"
    assert discord_rt["frontend_model"] == "gemma-4-E2B-it"
    assert discord_rt["interface_base_url"] == "http://spark.local:8000/v1"
    assert discord_rt["interface_api_key_env"] == "CUSTOM_KAME_INTERFACE_TOKEN"
    assert discord_rt["interface_temperature"] == 0.3
    assert discord_rt["interface_max_output_tokens"] == 96
    assert discord_rt["interface_timeout_seconds"] == 0.7
    assert discord_rt["interface_max_audio_seconds"] == 30.0
    assert discord_rt["interface_audio_input"] == "native_audio"
    assert discord_rt["asr_mode"] == "speculative"
    assert discord_rt["asr_provider"] == "streaming_stt"
    assert discord_rt["asr_model"] == "portable-streaming-asr"
    assert discord_rt["asr_base_url"] == "http://spark.local:8767"
    assert discord_rt["streaming_stt_base_url"] == "http://spark.local:8767"
    assert discord_rt["streaming_stt_token_env"] == "CUSTOM_STT_TOKEN"
    assert "oracle_provider" not in discord_rt
    assert "oracle_provider_name" not in discord_rt
    assert "preferred_local_oracle_model" not in discord_rt
    assert "oracle_model" not in discord_rt
    assert "oracle_base_url" not in discord_rt
    assert "oracle_api_mode" not in discord_rt
    assert discord_rt["oracle_timeout_seconds"] == 42.0
    assert discord_rt["max_spoken_sentences"] == 3
    assert discord_rt["voice_response_policy"] == "sentence_cap"
    assert discord_rt["barge_in_min_rms"] == 410
    assert discord_rt["barge_in_min_speech_ms"] == 130
    assert discord_rt["barge_in_stop_playback_deadline_ms"] == 150
    assert discord_rt["tts_provider"] == "streaming_tts"
    assert discord_rt["tts_model"] == "magpie-local-streaming-tts"
    assert discord_rt["tts_voice"] == "spark-voice-1"
    assert discord_rt["tts_base_url"] == "http://spark.local:8768"
    assert discord_rt["streaming_tts_base_url"] == "http://spark.local:8768"
    assert discord_rt["streaming_tts_token_env"] == "CUSTOM_TTS_TOKEN"
    assert discord_rt["fallback_policy"] == "legacy_voice"
    assert discord_rt["routing"]["require_oracle_for_tools"] is True
    assert discord_rt["metrics"]["log_turn_spans"] is True
    assert discord_rt["oracle_jobs"] == {
        "enabled": True,
        "max_concurrent": 4,
        "queue_limit": 16,
        "default_priority": "normal",
        "overflow_policy": "queue",
        "shutdown_timeout_seconds": 2.0,
        "speak_terminal_results": True,
        "audit_ledger_path": "",
    }
    assert discord_rt["turn_acknowledgement"] == {
        "enabled": True,
        "text": "One moment.",
    }
    assert discord_rt["output_events"] == {
        "caption_aliases": False,
        "audio_aliases": False,
    }
    assert discord_rt["quality_targets_ms"]["kame_speech_end_to_interface_decision_ms"] == 500
    assert discord_rt["quality_targets_ms"]["barge_in_confirmed_to_playback_stopped_ms"] == 150
    assert discord_rt["interface"] == {
        "provider": "gemma4",
        "base_url": "http://spark.local:8000/v1",
        "api_key_env": "CUSTOM_KAME_INTERFACE_TOKEN",
        "model": "gemma-4-E2B-it",
        "temperature": 0.3,
        "max_output_tokens": 96,
        "timeout_ms": 700,
        "max_audio_seconds": 30.0,
        "audio_input": "native_audio",
        "asr_mode": "speculative",
    }
    assert discord_rt["oracle"] == {
        "mode": "hermes_active_model",
        "selected_by": "Hermes /model",
        "provider_registration": {
            "enabled": True,
            "provider": "custom",
            "provider_name": "Spark Oracle",
            "preferred_local_model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
            "base_url": "http://spark.local:8001/v1",
            "api_mode": "chat_completions",
            "selection_authority": "Hermes /model",
        },
        "timeout_ms": 42000,
        "max_spoken_sentences": 3,
        "voice_response_policy": "sentence_cap",
    }
    assert discord_rt["barge_in"] == {
        "min_rms": 410,
        "min_speech_ms": 130,
        "stop_playback_deadline_ms": 150,
    }


def test_kame_profile_merge_registers_local_oracle_without_changing_active_model():
    existing = {
        "model": {"provider": "openrouter", "default": "gpt-5"},
        "custom_providers": [
            {
                "name": "Unrelated",
                "base_url": "https://unrelated.example.test/v1",
                "model": "unrelated-model",
            }
        ],
    }
    profile = realtime_voice_profile.build_kame_realtime_voice_profile(
        preferred_local_oracle_model="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        oracle_base_url="http://spark.local:8001/v1/",
        oracle_provider_name="Spark Oracle",
    )

    merged = realtime_voice_profile.merge_realtime_voice_profile(existing, profile)

    assert merged["model"] == existing["model"]
    assert "oracle_provider" not in merged["voice"]["realtime"]
    assert "oracle_provider_name" not in merged["voice"]["realtime"]
    assert "preferred_local_oracle_model" not in merged["voice"]["realtime"]
    assert "oracle_base_url" not in merged["voice"]["realtime"]
    assert "oracle_api_mode" not in merged["voice"]["realtime"]
    assert merged["voice"]["realtime"]["oracle"] == {
        "mode": "hermes_active_model",
        "selected_by": "Hermes /model",
        "provider_registration": {
            "enabled": True,
            "provider": "custom",
            "provider_name": "Spark Oracle",
            "preferred_local_model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
            "base_url": "http://spark.local:8001/v1",
            "api_mode": "chat_completions",
            "selection_authority": "Hermes /model",
        },
        "timeout_ms": 60000,
        "max_spoken_sentences": 2,
        "voice_response_policy": "sentence_cap",
    }
    assert merged["custom_providers"] == [
        {
            "name": "Unrelated",
            "base_url": "https://unrelated.example.test/v1",
            "model": "unrelated-model",
        },
        {
            "name": "Spark Oracle",
            "base_url": "http://spark.local:8001/v1",
            "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
            "api_mode": "chat_completions",
        },
    ]


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
    assert "HERMES_STREAMING_TTS_BRIDGE_TOKEN" in env
    assert len(env["HERMES_STREAMING_STT_BRIDGE_TOKEN"]) >= 32
    assert env["HERMES_STREAMING_TTS_BRIDGE_TOKEN"] == env["HERMES_STREAMING_STT_BRIDGE_TOKEN"]
    output = capsys.readouterr().out
    assert (
        "Generated realtime voice bridge token in "
        "HERMES_STREAMING_STT_BRIDGE_TOKEN, HERMES_STREAMING_TTS_BRIDGE_TOKEN"
    ) in output
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
    assert writes == [("HERMES_STREAMING_TTS_BRIDGE_TOKEN", "existing-token")]
    assert (
        "Generated realtime voice bridge token in "
        "HERMES_STREAMING_STT_BRIDGE_TOKEN, HERMES_STREAMING_TTS_BRIDGE_TOKEN"
    ) in capsys.readouterr().out


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
    assert realtime["interface_temperature"] == 0.2
    assert realtime["interface_max_output_tokens"] == 160
    assert realtime["interface_timeout_seconds"] == 0.8
    assert realtime["barge_in_min_rms"] == 350
    assert realtime["barge_in_stop_playback_deadline_ms"] == 150
    output = capsys.readouterr().out
    assert "realtime_voice_cartesia_bridge --generate-token" in output
    assert "realtime_voice_cartesia_bridge --check --strict --production-en-ja" in output
    assert "realtime_voice_cartesia_bridge --host 127.0.0.1 --port 8769 --production-en-ja" in output
    assert "realtime_voice_alpha_evidence --runs 3 --apply --provider cartesia --start-bridge" in output
    assert "realtime_voice_live_evidence --require-live-discord --require-openai-realtime" in output


def test_nvidia_speech_preset_apply_prints_local_bridge_next_steps(monkeypatch, tmp_path, capsys):
    saved = {}
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"model": {"provider": "openrouter"}},
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.setdefault("config", cfg))
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: tmp_path / "config.yaml")

    result = realtime_voice_profile.main(["--preset", "nvidia_speech", "--apply"])

    assert result == 0
    realtime = saved["config"]["voice"]["realtime"]
    assert realtime["streaming_stt_base_url"] == "http://127.0.0.1:8767"
    assert realtime["streaming_tts_base_url"] == "http://127.0.0.1:8768"
    assert realtime["streaming_stt_model"] == "nemotron-speech-streaming-0.6b"
    assert realtime["streaming_tts_model"] == "magpie-local-streaming-tts"
    output = capsys.readouterr().out
    assert "realtime_voice_profile --preset nvidia_speech --apply --generate-bridge-token" in output
    assert "HERMES_NEMOTRON_SPEECH_UPSTREAM_BASE_URL" in output
    assert "HERMES_MAGPIE_TTS_UPSTREAM_BASE_URL" in output
    assert "realtime_voice_nemotron_speech_bridge --check --strict --production-en-ja" in output
    assert "realtime_voice_magpie_tts_bridge --check --strict --production-en-ja" in output
    assert "realtime_voice_nemotron_speech_bridge --host 127.0.0.1 --port 8767 --production-en-ja" in output
    assert "realtime_voice_magpie_tts_bridge --host 127.0.0.1 --port 8768 --production-en-ja" in output
    assert "realtime_voice_alpha_evidence --runs 3 --apply --provider local_speech --start-bridge" in output


def test_nvidia_speech_preset_generate_token_sets_bridge_alias_envs(monkeypatch, tmp_path):
    saved = {}
    env = {}
    writes = {}
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"model": {"provider": "openrouter"}},
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved.setdefault("config", cfg))
    monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: tmp_path / "config.yaml")
    monkeypatch.setattr("hermes_cli.config.load_env", lambda: {**env, **writes})
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda key, value: writes.setdefault(key, value))

    result = realtime_voice_profile.main(["--preset", "nvidia_speech", "--apply", "--generate-bridge-token"])

    assert result == 0
    assert "HERMES_STREAMING_STT_BRIDGE_TOKEN" in writes
    assert "HERMES_STREAMING_TTS_BRIDGE_TOKEN" in writes
    assert writes["HERMES_NEMOTRON_SPEECH_BRIDGE_TOKEN"] == "${HERMES_STREAMING_STT_BRIDGE_TOKEN}"
    assert writes["HERMES_MAGPIE_TTS_BRIDGE_TOKEN"] == "${HERMES_STREAMING_TTS_BRIDGE_TOKEN}"
    assert saved["config"]["voice"]["realtime"]["streaming_stt_model"] == "nemotron-speech-streaming-0.6b"


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
    assert realtime["interface_temperature"] == 0.2
    assert realtime["interface_max_output_tokens"] == 160
    assert realtime["interface_timeout_seconds"] == 0.8
    assert realtime["barge_in_min_rms"] == 350
    assert realtime["barge_in_stop_playback_deadline_ms"] == 150
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
    assert realtime["interface_temperature"] == 0.2
    assert realtime["interface_max_output_tokens"] == 160
    assert realtime["interface_timeout_seconds"] == 0.8
    assert realtime["barge_in_min_rms"] == 350
    assert realtime["barge_in_stop_playback_deadline_ms"] == 150
    output = capsys.readouterr().out
    assert "export GEMINI_API_KEY=..." in output
    assert "export DISCORD_BOT_TOKEN=... DISCORD_GUILD_ID=... DISCORD_VOICE_CHANNEL_ID=..." in output
    assert "realtime_voice_sidecar --host 127.0.0.1 --port 8765" in output
    assert "realtime_voice_live_evidence --require-live-discord --require-gemini-live" in output
    assert "realtime_voice_live_evidence --require-live-discord --require-openai-realtime" not in output
