from hermes_cli import realtime_voice_magpie_tts_bridge, realtime_voice_nemotron_speech_bridge


def test_nemotron_speech_bridge_check_requires_upstream(monkeypatch, capsys):
    monkeypatch.delenv("HERMES_NEMOTRON_SPEECH_UPSTREAM_BASE_URL", raising=False)
    monkeypatch.delenv("HERMES_NEMOTRON_SPEECH_BRIDGE_TOKEN", raising=False)
    monkeypatch.delenv("HERMES_STREAMING_STT_BRIDGE_TOKEN", raising=False)

    result = realtime_voice_nemotron_speech_bridge.main(["--check", "--strict"])

    output = capsys.readouterr().out
    assert result == 1
    assert "Nemotron Speech bridge check failed" in output
    assert "HERMES_NEMOTRON_SPEECH_UPSTREAM_BASE_URL is required" in output
    assert "HERMES_NEMOTRON_SPEECH_BRIDGE_TOKEN is required" in output


def test_nemotron_speech_bridge_check_accepts_ready_upstream(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_NEMOTRON_SPEECH_BRIDGE_TOKEN", "bridge-token")
    monkeypatch.setattr(
        realtime_voice_nemotron_speech_bridge,
        "probe_local_speech_upstream_health",
        lambda _runtime: {"ok": True, "capabilities": {"streaming_stt": True}},
    )

    result = realtime_voice_nemotron_speech_bridge.main(
        [
            "--check",
            "--strict",
            "--upstream-base-url",
            "http://127.0.0.1:9101",
            "--production-en-ja",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Nemotron Speech bridge check OK" in output
    assert "input_languages: en,ja" in output
    assert "bridge-token" not in output


def test_magpie_tts_bridge_check_requires_upstream(monkeypatch, capsys):
    monkeypatch.delenv("HERMES_MAGPIE_TTS_UPSTREAM_BASE_URL", raising=False)
    monkeypatch.delenv("HERMES_MAGPIE_TTS_BRIDGE_TOKEN", raising=False)
    monkeypatch.delenv("HERMES_STREAMING_STT_BRIDGE_TOKEN", raising=False)

    result = realtime_voice_magpie_tts_bridge.main(["--check", "--strict"])

    output = capsys.readouterr().out
    assert result == 1
    assert "Magpie TTS bridge check failed" in output
    assert "HERMES_MAGPIE_TTS_UPSTREAM_BASE_URL is required" in output
    assert "HERMES_MAGPIE_TTS_BRIDGE_TOKEN is required" in output


def test_magpie_tts_bridge_check_accepts_ready_upstream(monkeypatch, capsys):
    monkeypatch.setenv("HERMES_MAGPIE_TTS_BRIDGE_TOKEN", "bridge-token")
    monkeypatch.setattr(
        realtime_voice_magpie_tts_bridge,
        "probe_local_speech_upstream_health",
        lambda _runtime: {"ok": True, "capabilities": {"streaming_tts": True}},
    )

    result = realtime_voice_magpie_tts_bridge.main(
        [
            "--check",
            "--strict",
            "--upstream-base-url",
            "http://127.0.0.1:9102",
            "--production-en-ja",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "Magpie TTS bridge check OK" in output
    assert "output_languages: en,ja" in output
    assert "bridge-token" not in output
