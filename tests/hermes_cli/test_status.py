from types import SimpleNamespace

from hermes_cli.status import show_status


def test_show_status_all_does_not_print_tavily_key_value(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sentinel = "NONSECRET_SENTINEL_VALUE_DO_NOT_PRINT_123456"
    monkeypatch.setenv("TAVILY_API_KEY", sentinel)

    show_status(SimpleNamespace(all=True, deep=False))

    output = capsys.readouterr().out
    assert "Tavily" in output
    assert sentinel not in output


def test_show_status_reports_realtime_voice_live_like(monkeypatch, capsys, tmp_path):
    from hermes_cli import status as status_mod

    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(
        status_mod,
        "_realtime_voice_status_payload",
        lambda: {
            "enabled": True,
            "available": True,
            "engine": "text_oracle_tts",
            "require_live_like": True,
            "conversation_quality": {
                "mode": "streaming_text",
                "reason": "streaming_stt_tts",
                "live_like": True,
            },
            "production_readiness": {
                "ready": True,
                "level": "production_ready",
                "issues": [],
                "evidence_ready": True,
                "evidence": {
                    "runs": 3,
                    "min_runs": 3,
                    "summary": {
                        "latency_ms": {
                            "audio_to_partial_transcript": {"count": 12, "p50": 80, "p90": 110, "p95": 120, "max": 140},
                            "final_transcript_to_first_text": {"count": 3, "p50": 140, "p90": 170, "p95": 180, "max": 200},
                            "final_transcript_to_first_audio": {"count": 12, "p50": 320, "p90": 400, "p95": 420, "max": 500},
                            "barge_in_ack": {"count": 3, "p50": 35, "p90": 50, "p95": 55, "max": 60},
                            "speech_boundary_to_final_transcript": {"count": 3, "p50": 24, "p90": 32, "p95": 34, "max": 38},
                            "final_transcript_to_interface_decision": {"count": 3, "p50": 45, "p90": 60, "p95": 65, "max": 70},
                            "interface_decision_to_oracle_accepted": {"count": 3, "p50": 30, "p90": 45, "p95": 50, "max": 55},
                            "oracle_accepted_to_first_token": {"count": 3, "p50": 180, "p90": 220, "p95": 240, "max": 260},
                            "barge_in_confirmed_to_playback_stopped": {"count": 3, "p50": 28, "p90": 42, "p95": 48, "max": 55},
                        },
                        "kame_routes": {
                            "total": 12,
                            "counts": {
                                "local": 4,
                                "defer": 3,
                                "oracle_direct": 4,
                                "reject_or_clarify": 1,
                            },
                            "oracle_avoided": 5,
                            "oracle_required": 7,
                            "oracle_avoidance_rate": 0.4167,
                        },
                        "kame_reflex_provenance": {
                            "total": 12,
                            "input_sources": {"native_audio": 10, "streaming_stt": 2},
                            "reflex_providers": {"vllm": 10, "streaming_stt": 2},
                            "native_audio": 10,
                            "vllm": 10,
                            "fallback": 2,
                            "fallback_only": False,
                        },
                        "latency_by_stack": {
                            "kame_interface_oracle|vllm|gemma-4-E2B-it|Hermes_model|cartesia|sonic-2": {
                                "stack": {
                                    "frontend_provider": "vllm",
                                    "frontend_model": "gemma-4-E2B-it",
                                    "oracle_authority": "Hermes /model",
                                    "tts_provider": "cartesia",
                                    "tts_model": "sonic-2",
                                },
                                "latency_ms": {
                                    "interface_decision_to_first_audio": {
                                        "count": 9,
                                        "p50": 170,
                                        "p90": 230,
                                        "p95": 260,
                                        "max": 300,
                                    },
                                    "final_transcript_to_first_audio": {
                                        "count": 9,
                                        "p50": 310,
                                        "p90": 390,
                                        "p95": 410,
                                        "max": 480,
                                    }
                                },
                                "kame_reflex_provenance": {
                                    "total": 12,
                                    "input_sources": {"native_audio": 10, "streaming_stt": 2},
                                    "reflex_providers": {"vllm": 10, "streaming_stt": 2},
                                    "native_audio": 10,
                                    "vllm": 10,
                                    "fallback": 2,
                                    "fallback_only": False,
                                },
                            }
                        },
                    },
                },
                "launch_review": {
                    "required": True,
                    "verified": True,
                    "reviewed_at": "2026-06-08T00:00:00Z",
                    "issues": [],
                },
            },
            "sidecar": {
                "mode": "external",
                "healthy": True,
            },
        },
        raising=False,
    )

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "◆ Realtime Voice" in output
    assert "Status:" in output
    assert "available" in output
    assert "Quality:" in output
    assert "streaming_text (streaming_stt_tts)" in output
    assert "Live-like:    yes" in output
    assert "Production:" in output
    assert "production_ready" in output
    assert "Evidence:" in output
    assert "runs 3/3" in output
    assert "partial p50=80ms p90=110ms p95=120ms max=140ms" in output
    assert "text p50=140ms p90=170ms p95=180ms max=200ms" in output
    assert "audio p50=320ms p90=400ms p95=420ms max=500ms" in output
    assert "barge p50=35ms p90=50ms p95=55ms max=60ms" in output
    assert "final_transcript p50=24ms p90=32ms p95=34ms max=38ms" in output
    assert "reflex p50=45ms p90=60ms p95=65ms max=70ms" in output
    assert "oracle_accept p50=30ms p90=45ms p95=50ms max=55ms" in output
    assert "oracle_token p50=180ms p90=220ms p95=240ms max=260ms" in output
    assert "barge_stop p50=28ms p90=42ms p95=48ms max=55ms" in output
    assert "kame_routes total=12 oracle_avoided=5 oracle_required=7 avoidance=41.7%" in output
    assert "local=4 defer=3 oracle_direct=4 reject_or_clarify=1" in output
    assert "kame_reflex total=12 native_audio=10 vllm=10 fallback=2" in output
    assert "sources native_audio=10 streaming_stt=2 providers streaming_stt=2 vllm=10" in output
    assert "stack kame_interface_oracle|vllm|gemma-4-E2B-it|Hermes_model|cartesia|sonic-2" in output
    assert "kame_audio p50=170ms p90=230ms p95=260ms max=300ms" in output
    assert "frontend=vllm/gemma-4-E2B-it oracle=Hermes /model tts=cartesia/sonic-2" in output
    assert "Review:" in output
    assert "passed (2026-06-08T00:00:00Z)" in output
    assert "Require live: yes" in output
    assert "Sidecar:      external (healthy: yes)" in output


def test_show_status_reports_realtime_voice_live_like_required(monkeypatch, capsys, tmp_path):
    from hermes_cli import status as status_mod

    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(
        status_mod,
        "_realtime_voice_status_payload",
        lambda: {
            "enabled": True,
            "available": False,
            "unavailable_reason": "live_like_required",
            "engine": "text_oracle_tts",
            "require_live_like": True,
            "conversation_quality": {
                "mode": "turn_based_text",
                "reason": "utterance_stt_tts",
                "live_like": False,
            },
            "production_readiness": {
                "ready": False,
                "level": "not_ready",
                "issues": ["live_like_required", "not_live_like"],
                "evidence": {
                    "configured": False,
                    "verified": False,
                    "report_path": None,
                    "runs": 0,
                    "min_runs": 3,
                    "issues": ["missing_evidence_report"],
                },
            },
            "sidecar": {
                "mode": "external",
                "healthy": True,
            },
        },
        raising=False,
    )

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "unavailable (live_like_required)" in output
    assert "turn_based_text (utterance_stt_tts)" in output
    assert "Live-like:    no" in output
    assert "not_ready (live_like_required, not_live_like)" in output
    assert "runs 0/3" in output
    assert "Require live: yes" in output


def test_show_status_termux_gateway_section_skips_systemctl(monkeypatch, capsys, tmp_path):
    from hermes_cli import status as status_mod
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(auth_mod, "get_nous_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)

    def _unexpected_systemctl(*args, **kwargs):
        raise AssertionError("systemctl should not be called in the Termux status view")

    monkeypatch.setattr(status_mod.subprocess, "run", _unexpected_systemctl)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Manager:      Termux / manual process" in output
    assert "Start with:   hermes gateway" in output
    assert "systemd (user)" not in output


def test_show_status_reports_nous_auth_error(monkeypatch, capsys, tmp_path):
    from hermes_cli import status as status_mod
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(
        auth_mod,
        "get_nous_auth_status",
        lambda: {
            "logged_in": False,
            "portal_base_url": "https://portal.nousresearch.com",
            "access_expires_at": "2026-04-20T01:00:51+00:00",
            "agent_key_expires_at": "2026-04-20T04:54:24+00:00",
            "has_refresh_token": True,
            "error": "Refresh session has been revoked",
        },
        raising=False,
    )
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Nous Portal   ✗ not logged in (run: hermes portal)" in output
    assert "Error:      Refresh session has been revoked" in output
    assert "Access exp:" in output
    assert "Key exp:" in output


def test_show_status_reports_nous_inference_key_without_portal_login(monkeypatch, capsys, tmp_path):
    from hermes_cli import status as status_mod
    from hermes_cli.nous_account import NousPortalAccountInfo
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(
        auth_mod,
        "get_nous_auth_status",
        lambda: {
            "logged_in": False,
            "inference_credential_present": True,
            "credential_source": "pool:manual opaque key",
            "inference_base_url": "https://inference.example.com/v1",
            "agent_key_expires_at": "2099-01-01T00:00:00+00:00",
        },
        raising=False,
    )
    monkeypatch.setattr(
        status_mod,
        "get_nous_portal_account_info",
        lambda: NousPortalAccountInfo(
            logged_in=False,
            source="inference_key",
            fresh=False,
            inference_credential_present=True,
            inference_base_url="https://inference.example.com/v1",
        ),
        raising=False,
    )
    monkeypatch.setattr(status_mod, "managed_nous_tools_enabled", lambda: False, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Nous Portal   ✗ not logged in (Nous inference key configured)" in output
    assert "Inference:  https://inference.example.com/v1" in output
    assert "Nous inference credentials are configured" in output


# ---------------------------------------------------------------------------
# Helpers shared by xAI OAuth status tests
# ---------------------------------------------------------------------------

def _base_xai_mocks(monkeypatch, tmp_path):
    """Set up the minimal environment for show_status, returning status_mod."""
    from hermes_cli import status as status_mod
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(auth_mod, "get_nous_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_minimax_oauth_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)
    return status_mod


class TestShowStatusXaiOAuth:
    """xAI OAuth row in hermes status."""

    # ------------------------------------------------------------------
    # Logged-in branch
    # ------------------------------------------------------------------

    def test_logged_in_shows_check_mark_and_label(self, monkeypatch, capsys, tmp_path):
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {"logged_in": True, "auth_store": "/a/auth.json"},
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "xAI OAuth" in out
        # The logged-in label must appear; the "not logged in" label must not
        assert "✓" in out or "logged in" in out
        assert "not logged in" not in out.split("xAI OAuth", 1)[1].split("\n")[0]

    def test_logged_in_shows_auth_store(self, monkeypatch, capsys, tmp_path):
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {"logged_in": True, "auth_store": "/home/u/.hermes/auth.json"},
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "Auth file:  /home/u/.hermes/auth.json" in out

    def test_logged_in_shows_last_refresh(self, monkeypatch, capsys, tmp_path):
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {
                                "logged_in": True,
                                "auth_store": "/a/auth.json",
                                "last_refresh": "2026-05-17T10:00:00+00:00",
                            },
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "Refreshed:" in out

    def test_logged_in_does_not_show_error_line(self, monkeypatch, capsys, tmp_path):
        """Error field must be suppressed when logged_in is True."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {
                                "logged_in": True,
                                "auth_store": "/a/auth.json",
                                "error": "stale-error-must-not-appear",
                            },
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        xai_section = out.split("xAI OAuth", 1)[1]
        assert "stale-error-must-not-appear" not in xai_section

    def test_no_auth_store_line_when_field_absent(self, monkeypatch, capsys, tmp_path):
        """Auth file line must not appear when auth_store is missing."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {"logged_in": True},
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        xai_section = out.split("xAI OAuth", 1)[1].split("◆", 1)[0]
        assert "Auth file:" not in xai_section

    def test_no_refreshed_line_when_last_refresh_absent(self, monkeypatch, capsys, tmp_path):
        """Refreshed line must not appear when last_refresh is not present."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {"logged_in": True, "auth_store": "/a/auth.json"},
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        xai_section = out.split("xAI OAuth", 1)[1].split("◆", 1)[0]
        assert "Refreshed:" not in xai_section

    # ------------------------------------------------------------------
    # Not-logged-in branch
    # ------------------------------------------------------------------

    def test_not_logged_in_shows_login_command(self, monkeypatch, capsys, tmp_path):
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {"logged_in": False, "error": "no credentials"},
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "not logged in (run: hermes auth add xai-oauth)" in out

    def test_not_logged_in_shows_error(self, monkeypatch, capsys, tmp_path):
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {"logged_in": False, "error": "Token has expired"},
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "Error:      Token has expired" in out

    def test_not_logged_in_omits_error_line_when_error_absent(self, monkeypatch, capsys, tmp_path):
        """No Error: line when not logged in but error key is missing."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {"logged_in": False},
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        xai_section = out.split("xAI OAuth", 1)[1].split("◆", 1)[0]
        assert "Error:" not in xai_section

    # ------------------------------------------------------------------
    # Resilience: import failure and runtime exception
    # ------------------------------------------------------------------

    def test_import_failure_does_not_crash_show_status(self, monkeypatch, capsys, tmp_path):
        """show_status must complete even when get_xai_oauth_auth_status cannot be imported."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.delattr(auth_mod, "get_xai_oauth_auth_status", raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "◆ Auth Providers" in out

    def test_import_failure_does_not_break_other_oauth_providers(self, monkeypatch, capsys, tmp_path):
        """Nous/Codex/MiniMax rows must still appear when xAI import fails."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_nous_auth_status",
                            lambda: {"logged_in": True}, raising=False)
        monkeypatch.delattr(auth_mod, "get_xai_oauth_auth_status", raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "Nous Portal" in out
        assert "MiniMax OAuth" in out

    def test_status_function_exception_does_not_crash(self, monkeypatch, capsys, tmp_path):
        """show_status must not propagate an exception raised by get_xai_oauth_auth_status."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)

        def _raises():
            raise RuntimeError("backend unreachable")

        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", _raises, raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "◆ Auth Providers" in out

    def test_status_function_returns_none_does_not_crash(self, monkeypatch, capsys, tmp_path):
        """get_xai_oauth_auth_status returning None must be handled gracefully."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: None, raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "xAI OAuth" in out
        assert "not logged in (run: hermes auth add xai-oauth)" in out
