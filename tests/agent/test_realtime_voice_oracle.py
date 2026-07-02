import asyncio
import threading

import run_agent

from agent.realtime_voice import RealtimeVoiceEngineKind, RealtimeVoiceSessionConfig
from agent.realtime_voice_kame import KameOracleRequest, KameRoute
from agent.realtime_voice_oracle import HermesRealtimeOracle
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
        },
    )

    assert "low-latency realtime frontend model handles live speech" in prompt
    assert "elevenlabs realtime-voice" in prompt
    assert "Hermes backend oracle" in prompt
    assert "active Hermes model" in prompt
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


def test_hermes_realtime_oracle_runs_concurrent_kame_requests_and_targets_interrupt(monkeypatch):
    calls = []
    running = 0
    max_running = 0
    release = threading.Event()
    condition = threading.Condition()

    class FakeAIAgent:
        def __init__(self, *, model, platform, session_id):
            self.model = model
            self.platform = platform
            self.session_id = session_id
            self.interrupts = []

        def run_conversation(self, prompt, *, persist_user_message=None, stream_callback=None):
            nonlocal running, max_running
            with condition:
                running += 1
                max_running = max(max_running, running)
                calls.append(
                    {
                        "agent": self,
                        "prompt": prompt,
                        "persist_user_message": persist_user_message,
                        "session_id": self.session_id,
                        "platform": self.platform,
                        "model": self.model,
                    }
                )
                condition.notify_all()
            if stream_callback is not None:
                stream_callback(f"delta for {persist_user_message};")
            assert release.wait(timeout=2), "test did not release fake realtime oracle"
            with condition:
                running -= 1
                condition.notify_all()
            return {"final_response": f"final for {persist_user_message}"}

        def interrupt(self, message):
            self.interrupts.append(message)

    def wait_for_running_count(count):
        with condition:
            return condition.wait_for(lambda: len(calls) >= count and running >= count, timeout=2)

    async def collect(oracle, request):
        chunks = []
        async for delta in oracle.stream_answer_for_request(request):
            chunks.append(delta)
        return "".join(chunks)

    async def run():
        monkeypatch.setattr(run_agent, "AIAgent", FakeAIAgent)
        oracle = HermesRealtimeOracle(
            RealtimeVoiceSessionConfig(
                session_id="voice-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                metadata={"transport": "discord_voice"},
            )
        )
        requests = [
            KameOracleRequest(
                session_id="voice-123",
                turn_id=f"voice-123:{index}",
                source="discord_voice",
                user_id="jetha",
                intent=f"Run task {index}.",
                route=KameRoute.ORACLE_DIRECT,
                transcript=f"run task {index}",
                transcript_source="reflex_audio",
                asr_transcript=f"task {index} exact evidence",
                asr_transcript_source="asr",
                cancellation_token=f"voice-123:{index}:cancel",
            )
            for index in range(1, 5)
        ]
        tasks = [asyncio.create_task(collect(oracle, request)) for request in requests]
        assert await asyncio.to_thread(wait_for_running_count, 4)

        oracle.interrupt_request(requests[1], "cancel task 2 only")
        interrupted = [
            call
            for call in calls
            if call["agent"].interrupts
        ]
        assert len(interrupted) == 1
        assert interrupted[0]["persist_user_message"] == "task 2 exact evidence"
        assert interrupted[0]["agent"].interrupts == ["cancel task 2 only"]

        release.set()
        results = await asyncio.gather(*tasks)
        assert results == [
            "delta for task 1 exact evidence;",
            "delta for task 2 exact evidence;",
            "delta for task 3 exact evidence;",
            "delta for task 4 exact evidence;",
        ]

        assert max_running == 4
        assert [call["session_id"] for call in calls] == ["voice-123"] * 4
        assert [call["platform"] for call in calls] == ["desktop_voice"] * 4
        assert [call["model"] for call in calls] == [""] * 4
        assert [call["persist_user_message"] for call in calls] == [
            "task 1 exact evidence",
            "task 2 exact evidence",
            "task 3 exact evidence",
            "task 4 exact evidence",
        ]
        assert all("live Discord voice channel" in call["prompt"] for call in calls)
        assert all("Hermes backend oracle" in call["prompt"] for call in calls)
        assert all("Verbatim ASR evidence" in call["prompt"] for call in calls)

    asyncio.run(run())


def test_voice_oracle_preserves_default_toolset_selection_for_general_voice_turn(monkeypatch):
    init_kwargs = []

    class FakeAIAgent:
        def __init__(self, **kwargs):
            init_kwargs.append(kwargs)

        def run_conversation(self, prompt, *, persist_user_message=None, stream_callback=None):
            if stream_callback is not None:
                stream_callback("ok")
            return {"final_response": "ok"}

    async def run():
        monkeypatch.setattr(run_agent, "AIAgent", FakeAIAgent)
        oracle = HermesRealtimeOracle(
            RealtimeVoiceSessionConfig(
                session_id="voice-general",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                oracle_tool_router={
                    "enabled": True,
                    "mode": "deterministic",
                    "voiceops_toolsets": ["voiceops"],
                    "default_toolsets": [],
                },
            )
        )
        chunks = []
        async for delta in oracle.stream_answer("tell me a short joke"):
            chunks.append(delta)
        assert chunks == ["ok"]

    asyncio.run(run())
    assert init_kwargs == [
        {
            "model": "",
            "platform": "desktop_voice",
            "session_id": "voice-general",
        }
    ]


def test_voice_oracle_applies_scoped_tool_search_override(monkeypatch):
    observed = []

    class FakeAIAgent:
        def __init__(self, **kwargs):
            from tools.tool_search import load_config

            cfg = load_config()
            observed.append((kwargs, cfg.enabled, cfg.defer_core))

        def run_conversation(self, prompt, *, persist_user_message=None, stream_callback=None):
            return {"final_response": "ok"}

    async def run():
        monkeypatch.setattr(run_agent, "AIAgent", FakeAIAgent)
        oracle = HermesRealtimeOracle(
            RealtimeVoiceSessionConfig(
                session_id="voice-tool-search",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                oracle_tool_router={
                    "enabled": True,
                    "mode": "deterministic",
                    "voiceops_toolsets": ["voiceops"],
                    "default_toolsets": [],
                    "tool_search": {
                        "enabled": "on",
                        "defer_core": "all",
                    },
                },
            )
        )
        assert await oracle.answer("tell me a short joke") == "ok"

    asyncio.run(run())
    assert observed == [
        (
            {
                "model": "",
                "platform": "desktop_voice",
                "session_id": "voice-tool-search",
            },
            "on",
            "all",
        )
    ]


def test_voice_oracle_router_disabled_does_not_apply_tool_search_override(monkeypatch):
    observed = []

    class FakeAIAgent:
        def __init__(self, **kwargs):
            from tools.tool_search import load_config

            cfg = load_config()
            observed.append((cfg.enabled, cfg.defer_core))

        def run_conversation(self, prompt, *, persist_user_message=None, stream_callback=None):
            return {"final_response": "ok"}

    async def run():
        monkeypatch.setattr(run_agent, "AIAgent", FakeAIAgent)
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"tools": {"tool_search": {"enabled": "off", "defer_core": "off"}}},
        )
        oracle = HermesRealtimeOracle(
            RealtimeVoiceSessionConfig(
                session_id="voice-tool-search-disabled",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                oracle_tool_router={
                    "enabled": False,
                    "tool_search": {
                        "enabled": "on",
                        "defer_core": "all",
                    },
                },
            )
        )
        assert await oracle.answer("tell me a short joke") == "ok"

    asyncio.run(run())
    assert observed == [("off", "off")]


def test_voice_oracle_routes_voiceops_request_to_voiceops_toolset(monkeypatch):
    init_kwargs = []
    prompts = []

    class FakeAIAgent:
        def __init__(self, **kwargs):
            init_kwargs.append(kwargs)

        def run_conversation(self, prompt, *, persist_user_message=None, stream_callback=None):
            prompts.append(prompt)
            if stream_callback is not None:
                stream_callback("preparing action packet")
            return {"final_response": "prepared"}

    async def run():
        monkeypatch.setattr(run_agent, "AIAgent", FakeAIAgent)
        oracle = HermesRealtimeOracle(
            RealtimeVoiceSessionConfig(
                session_id="voiceops-123",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                metadata={
                    "transport": "discord_voice",
                    "voice_architecture": "kame_frontend_oracle",
                },
                oracle_tool_router={
                    "enabled": True,
                    "mode": "deterministic",
                    "voiceops_toolsets": ["voiceops"],
                    "default_toolsets": [],
                },
            )
        )
        request = KameOracleRequest(
            session_id="voiceops-123",
            turn_id="voiceops-123:1",
            source="discord_voice",
            user_id="jetha",
            intent="Use Stripe spending money to provision VoIP and call my phone.",
            route=KameRoute.ORACLE_DIRECT,
            transcript="give yourself a Stripe budget, set up a VoIP account, and call my phone",
            transcript_source="reflex_audio",
            asr_transcript="give yourself a Stripe budget, set up a VoIP account, and call my phone",
            asr_transcript_source="asr",
            cancellation_token="voiceops-123:1:cancel",
        )
        chunks = []
        async for delta in oracle.stream_answer_for_request(request):
            chunks.append(delta)
        assert chunks == ["preparing action packet"]

    asyncio.run(run())
    assert init_kwargs == [
        {
            "model": "",
            "platform": "desktop_voice",
            "session_id": "voiceops-123",
            "enabled_toolsets": ["voiceops"],
        }
    ]
    assert "Hermes backend oracle" in prompts[0]
    assert "Verbatim ASR evidence" in prompts[0]


def test_voice_oracle_tool_router_can_be_disabled(monkeypatch):
    init_kwargs = []

    class FakeAIAgent:
        def __init__(self, **kwargs):
            init_kwargs.append(kwargs)

        def run_conversation(self, prompt, *, persist_user_message=None, stream_callback=None):
            if stream_callback is not None:
                stream_callback("ok")
            return {"final_response": "ok"}

    async def run():
        monkeypatch.setattr(run_agent, "AIAgent", FakeAIAgent)
        oracle = HermesRealtimeOracle(
            RealtimeVoiceSessionConfig(
                session_id="voiceops-disabled",
                engine=RealtimeVoiceEngineKind.KAME_INTERFACE_ORACLE,
                oracle_tool_router={
                    "enabled": False,
                    "mode": "deterministic",
                    "voiceops_toolsets": ["voiceops"],
                },
            )
        )
        request = KameOracleRequest(
            session_id="voiceops-disabled",
            turn_id="voiceops-disabled:1",
            source="discord_voice",
            user_id="jetha",
            intent="Use Stripe spending money to provision VoIP and call my phone.",
            route=KameRoute.ORACLE_DIRECT,
            transcript="give yourself a Stripe budget and call my phone",
            transcript_source="reflex_audio",
            asr_transcript="give yourself a Stripe budget and call my phone",
            asr_transcript_source="asr",
            cancellation_token="voiceops-disabled:1:cancel",
        )
        chunks = []
        async for delta in oracle.stream_answer_for_request(request):
            chunks.append(delta)
        assert chunks == ["ok"]

    asyncio.run(run())
    assert init_kwargs == [
        {
            "model": "",
            "platform": "desktop_voice",
            "session_id": "voiceops-disabled",
        }
    ]
