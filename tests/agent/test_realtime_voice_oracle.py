from agent.realtime_voice_oracle import _voice_oracle_prompt


def test_voice_oracle_prompt_includes_discord_live_voice_capability_context():
    prompt = _voice_oracle_prompt(
        "Can you hear me?",
        {"transport": "discord_voice", "language": "en"},
    )

    assert "live Discord voice channel" in prompt
    assert "spoken reply will be sent back to the same voice channel" in prompt
    assert "Do not claim that you cannot hear, join, or speak in Discord voice" in prompt
    assert "language=en" in prompt


def test_voice_oracle_prompt_warns_against_generic_voice_denial():
    prompt = _voice_oracle_prompt("Hello", {})

    assert "live voice transport" in prompt
    assert "Do not deny live voice capability" in prompt


def test_voice_oracle_prompt_includes_kame_frontend_backend_roles():
    prompt = _voice_oracle_prompt(
        "Use your voice.",
        {
            "transport": "discord_voice",
            "voice_architecture": "kame_frontend_oracle",
            "frontend_provider": "elevenlabs",
            "frontend_model": "realtime-voice",
            "oracle_model": "deep-hermes",
        },
    )

    assert "low-latency realtime frontend model handles live speech" in prompt
    assert "elevenlabs realtime-voice" in prompt
    assert "Hermes backend oracle" in prompt
    assert "deep-hermes" in prompt
    assert "Do not describe the frontend as a separate user-visible bot" in prompt


def test_voice_oracle_prompt_includes_kame_brief_summary_policy():
    prompt = _voice_oracle_prompt(
        "Summarize the project state.",
        {
            "voice_architecture": "kame_frontend_oracle",
            "kame_requested_response_style": {
                "spoken": True,
                "max_sentences": 2,
                "policy": "brief_summary",
                "allow_followup_offer": False,
            },
        },
    )

    assert "Summarize long oracle output for speech" in prompt
    assert "at most 2 sentence(s)" in prompt
    assert "Requested response style: spoken=true; policy=brief_summary" in prompt
