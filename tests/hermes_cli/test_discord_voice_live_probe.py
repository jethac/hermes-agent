import json

from hermes_cli import discord_voice_live_probe
from hermes_cli.discord_voice_live_probe import DiscordVoiceLiveProbeResult


def test_discord_voice_live_probe_main_writes_report(monkeypatch, tmp_path, capsys):
    async def fake_probe(args):
        return DiscordVoiceLiveProbeResult(
            ok=True,
            guild_name="jetha dev server",
            voice_channel_name="General",
            connect_perm=True,
            speak_perm=True,
            connected=True,
            opus_loaded=True,
            accepted_audio_source=True,
            played=True,
            playing_during_probe=True,
            receiver_started=True,
            inbound_observed=False,
            disconnected=True,
            wait_seconds=float(args.wait_seconds),
        )

    monkeypatch.setattr(discord_voice_live_probe, "run_discord_voice_live_probe", fake_probe)
    report_path = tmp_path / "probe.json"

    result = discord_voice_live_probe.main([
        "--voice-channel-name",
        "General",
        "--wait-seconds",
        "0.5",
        "--report",
        str(report_path),
    ])

    assert result == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["voice_channel_name"] == "General"
    assert payload["accepted_audio_source"] is True
    assert "evidence_context" in payload
    assert "latency_metrics_ms" in payload
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_discord_voice_live_probe_requires_bot_token(monkeypatch):
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    args = discord_voice_live_probe.build_parser().parse_args([])

    import asyncio

    result = asyncio.run(discord_voice_live_probe.run_discord_voice_live_probe(args))

    assert result.ok is False
    assert result.error == "DISCORD_BOT_TOKEN is required"
    assert result.evidence_context["env_presence"]["DISCORD_BOT_TOKEN"] is False
    assert result.evidence_context["config_snapshot"]["voice_channel_name"] == "General"


def test_discord_voice_live_probe_failure_reason_identifies_empty_inbound_channel():
    reason = discord_voice_live_probe._probe_failure_reason(
        connect_perm=True,
        speak_perm=True,
        accepted_audio_source=True,
        playing_during_probe=True,
        disconnected=True,
        require_inbound=True,
        inbound_observed=False,
        members_before=0,
        members_after=1,
    )

    assert reason == "inbound_required_but_no_other_members"


def test_discord_voice_live_probe_failure_reason_identifies_missing_inbound_frames():
    reason = discord_voice_live_probe._probe_failure_reason(
        connect_perm=True,
        speak_perm=True,
        accepted_audio_source=True,
        playing_during_probe=True,
        disconnected=True,
        require_inbound=True,
        inbound_observed=False,
        members_before=2,
        members_after=3,
    )

    assert reason == "inbound_required_but_no_frames"
