import asyncio
import json

from hermes_cli.discord_realtime_voice_smoke import main, run_discord_realtime_voice_smoke


def test_discord_realtime_voice_smoke_exercises_pcm_mixer_and_barge_in():
    result = asyncio.run(run_discord_realtime_voice_smoke())

    assert result.ok is True
    assert result.mode == "discord_loopback"
    assert result.transport == "discord_voice"
    assert result.input_pcm48_bytes == 3840
    assert result.sidecar_pcm16_bytes == 640
    assert result.mixer_frames >= 1
    assert result.mixer_frame_bytes == 3840
    assert result.barge_in_sent is True
    assert result.mixer_stop_calls >= 1
    assert "audio.output.chunk" in result.events
    assert result.evidence_context["git_commit"]
    assert "input_to_first_mixer_frame_ms" in result.latency_metrics_ms
    assert "barge_in_ack_ms" in result.latency_metrics_ms


def test_discord_realtime_voice_smoke_writes_report(tmp_path, capsys):
    report = tmp_path / "discord-smoke.json"

    exit_code = main(["--report", str(report)])

    assert exit_code == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["transport"] == "discord_voice"
    assert payload["sidecar_pcm16_bytes"] == 640
    assert payload["evidence_context"]["git_commit"]
    assert payload["latency_metrics_ms"]["barge_in_ack_ms"] >= 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
