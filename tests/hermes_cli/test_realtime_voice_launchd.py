import plistlib
from pathlib import Path

from hermes_cli import realtime_voice_launchd


def test_launchd_generator_writes_bridge_and_sidecar_plists(tmp_path, capsys):
    output_dir = tmp_path / "launchd"
    repo_dir = tmp_path / "repo"
    hermes_home = tmp_path / "hermes-home"
    repo_dir.mkdir()

    result = realtime_voice_launchd.main(
        [
            "--output-dir",
            str(output_dir),
            "--repo-dir",
            str(repo_dir),
            "--hermes-home",
            str(hermes_home),
            "--uv-bin",
            "/opt/homebrew/bin/uv",
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "ai.hermes.realtime-voice.elevenlabs-bridge.plist" in output
    assert "ai.hermes.realtime-voice.sidecar.plist" in output

    bridge = plistlib.loads(
        (output_dir / "ai.hermes.realtime-voice.elevenlabs-bridge.plist").read_bytes()
    )
    sidecar = plistlib.loads((output_dir / "ai.hermes.realtime-voice.sidecar.plist").read_bytes())

    assert bridge["Label"] == "ai.hermes.realtime-voice.elevenlabs-bridge"
    assert sidecar["Label"] == "ai.hermes.realtime-voice.sidecar"
    assert bridge["ProgramArguments"][:2] == ["/bin/zsh", "-lc"]
    assert sidecar["ProgramArguments"][:2] == ["/bin/zsh", "-lc"]
    assert bridge["WorkingDirectory"] == str(repo_dir.resolve())
    assert sidecar["WorkingDirectory"] == str(repo_dir.resolve())
    assert bridge["RunAtLoad"] is True
    assert sidecar["KeepAlive"] == {"SuccessfulExit": False}
    assert bridge["EnvironmentVariables"]["HERMES_HOME"] == str(hermes_home)
    assert sidecar["EnvironmentVariables"]["HERMES_HOME"] == str(hermes_home)
    assert bridge["StandardOutPath"] == str(
        hermes_home / "logs" / "realtime-voice-elevenlabs-bridge.log"
    )
    assert sidecar["StandardErrorPath"] == str(
        hermes_home / "logs" / "realtime-voice-sidecar.error.log"
    )


def test_launchd_bridge_command_loads_env_and_runs_elevenlabs_bridge(tmp_path):
    plist = realtime_voice_launchd.build_elevenlabs_bridge_plist(
        repo_dir=tmp_path / "repo with spaces",
        hermes_home=tmp_path / "home with spaces",
        uv_bin="uv",
    )

    command = plist["ProgramArguments"][2]

    assert ".env" in command
    assert "hermes_cli.realtime_voice_elevenlabs_bridge" in command
    assert "--host 127.0.0.1" in command
    assert "--port 8767" in command
    assert "--production-en-ja" in command
    assert "--extra dev --extra voice" in command
    assert "repo with spaces" in command


def test_launchd_sidecar_command_aliases_bridge_token_and_points_at_bridge(tmp_path):
    plist = realtime_voice_launchd.build_realtime_voice_sidecar_plist(
        repo_dir=tmp_path / "repo",
        hermes_home=tmp_path / "home",
        uv_bin="uv",
        bridge_base_url="http://127.0.0.1:8767",
        include_dev_extra=False,
    )

    command = plist["ProgramArguments"][2]

    assert "HERMES_VOICE_STREAMING_STT_TOKEN" in command
    assert "HERMES_STREAMING_STT_BRIDGE_TOKEN" in command
    assert "HERMES_VOICE_STREAMING_TTS_TOKEN" in command
    assert "hermes_cli.realtime_voice_sidecar" in command
    assert "--streaming-stt-base-url http://127.0.0.1:8767" in command
    assert "--streaming-tts-base-url http://127.0.0.1:8767" in command
    assert "--streaming-stt-model scribe_v2_realtime" in command
    assert "--streaming-tts-model eleven_flash_v2_5" in command
    assert "--input-languages en,ja" in command
    assert "--scripts Latn,Jpan" in command
    assert "--extra dev" not in command
    assert "--extra voice" in command


def test_launchd_plist_uses_absolute_resolved_repo_dir(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    plist = realtime_voice_launchd.build_elevenlabs_bridge_plist(
        repo_dir=Path(str(repo_dir)),
        hermes_home=tmp_path / "home",
        uv_bin="uv",
    )

    assert plist["WorkingDirectory"] == str(repo_dir)
