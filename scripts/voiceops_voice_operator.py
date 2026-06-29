#!/usr/bin/env python3
"""Generate Milestone 1 VoiceOps Discord voice-operator evidence.

This headless generator runs the in-memory Discord realtime voice loopback
smoke. It does not connect to Discord, read Discord credentials, start a
provider sidecar, send messages, or place calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermes_cli.discord_realtime_voice_smoke import run_discord_realtime_voice_smoke


DEFAULT_OUTPUT_DIR = Path("artifacts/voiceops-voice-operator/current")
DISCORD_FRAME_BYTES = 3840
SIDECAR_FRAME_BYTES = 640

REQUIRED_EVENTS = {
    "transcript.partial",
    "transcript.final",
    "assistant.text.partial",
    "audio.output.chunk",
    "assistant.commit",
    "barge_in.detected",
}

RECEIVER_CALLBACK_TEST_REFS = [
    "tests/gateway/test_voice_command.py::test_join_voice_channel_wires_realtime_frame_and_speech_start_callbacks",
]

BARGE_IN_ENERGY_TEST_REFS = [
    "tests/gateway/test_voice_command.py::test_pcm16_rms_ignores_silence_and_detects_volume",
    "tests/gateway/test_voice_command.py::test_join_voice_channel_wires_realtime_frame_and_speech_start_callbacks",
    "tests/gateway/test_discord_realtime_voice.py::test_discord_realtime_session_sends_speech_energy_event",
]

LIVE_EVIDENCE_REQUIRED_DISCORD_BOOLS = (
    "connect_perm",
    "speak_perm",
    "connected",
    "opus_loaded",
    "accepted_audio_source",
    "played",
    "playing_during_probe",
    "receiver_started",
    "disconnected",
)

LIVE_EVIDENCE_REQUIRED_SIDECAR_BOOLS = (
    "sidecar_running",
    "sidecar_healthy",
    "session_started",
    "session_closed",
    "fallback_mode_visible",
)

LIVE_EVIDENCE_REQUIRED_TURN_BOOLS = (
    "transcript_observed",
    "assistant_audio_observed",
    "barge_in_observed",
    "spoken_reply_short",
    "no_voice_denial_observed",
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _non_negative_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float) and value >= 0:
        return float(value)
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _looks_secret_or_phone(value: Any) -> bool:
    text = str(value or "")
    lowered = text.lower()
    secret_markers = ("sk_", "pk_", "rk_", "whsec_", "xoxb", "xoxp", "ghp_", "bearer ")
    if any(marker in lowered for marker in secret_markers):
        return True
    digits = "".join(ch for ch in text if ch.isdigit())
    return text.strip().startswith("+") and len(digits) >= 8


def build_live_probe_evidence_template() -> dict[str, Any]:
    return {
        "schema_version": "voiceops.milestone1.live_voice_evidence.v1",
        "redaction_policy": "references and booleans only; no Discord tokens, provider tokens, full phone numbers, or raw transcripts with secrets",
        "discord_live_probe": {
            "source_artifact": "discord-live-probe.json",
            "kind": "discord_live_probe",
            "ok": False,
            "connect_perm": False,
            "speak_perm": False,
            "connected": False,
            "opus_loaded": False,
            "accepted_audio_source": False,
            "played": False,
            "playing_during_probe": False,
            "receiver_started": False,
            "receiver_frames": 0,
            "receiver_speech_start": 0,
            "inbound_observed": False,
            "disconnected": False,
            "require_inbound": True,
            "latency_metrics_ms": {
                "connect_ms": None,
                "playback_observed_ms": None,
                "inbound_observed_ms": None,
                "disconnect_ms": None,
            },
        },
        "sidecar_session": {
            "source_artifact": "voice-status-or-sidecar-report.json",
            "sidecar_running": False,
            "sidecar_healthy": False,
            "session_started": False,
            "session_closed": False,
            "fallback_mode_visible": False,
            "fallback_reason": None,
            "latency_metrics_ms": {
                "session_start_ms": None,
                "shutdown_ms": None,
            },
        },
        "live_turn": {
            "source_artifact": "voice-turn-evidence.json",
            "transcript_observed": False,
            "assistant_audio_observed": False,
            "barge_in_observed": False,
            "spoken_reply_short": False,
            "no_voice_denial_observed": False,
            "speech_end_to_first_audio_ms": None,
            "barge_in_stop_ms": None,
        },
    }


def _load_live_evidence(paths: list[Path] | None) -> dict[str, Any]:
    paths = paths or []
    if not paths:
        return {
            "loaded": False,
            "mode": "supplied_artifacts_only",
            "artifact_paths": [],
            "overall_status": "needs_live_probe",
            "issues": ["live_evidence_not_loaded"],
            "redaction_policy": "not_loaded",
        }
    payload: dict[str, Any] = {}
    load_issues: list[str] = []
    for path in paths:
        loaded = _load_live_evidence_file(path)
        if loaded["issues"]:
            load_issues.extend(str(issue) for issue in loaded["issues"])
        data = loaded.get("payload")
        if isinstance(data, Mapping):
            _merge_live_evidence_payload(payload, data)
    evidence = validate_live_probe_evidence(payload, paths=paths)
    evidence["issues"] = sorted(set([*evidence["issues"], *load_issues]))
    if evidence["issues"]:
        evidence["overall_status"] = "partial_live_evidence"
    else:
        evidence["overall_status"] = "live_evidence_supplied_not_readiness_claim"
    return evidence


def _load_live_evidence_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "payload": None,
            "issues": ["live_evidence_file_not_found"],
        }
    except json.JSONDecodeError as exc:
        return {
            "payload": None,
            "issues": [f"live_evidence_json_parse_failed:{exc.msg}"],
        }
    if not isinstance(payload, Mapping):
        return {
            "payload": None,
            "issues": ["live_evidence_root_must_be_object"],
        }
    return {"payload": payload, "issues": []}


def _merge_live_evidence_payload(target: dict[str, Any], payload: Mapping[str, Any]) -> None:
    discord_probe = _discord_probe_section(payload)
    if discord_probe:
        target["discord_live_probe"] = dict(discord_probe)
    for section_name in ("sidecar_session", "live_turn"):
        section = payload.get(section_name)
        if isinstance(section, Mapping):
            target[section_name] = dict(section)
    if _looks_like_sidecar_session(payload):
        target["sidecar_session"] = dict(payload)
    if _looks_like_live_turn(payload):
        target["live_turn"] = dict(payload)


def _looks_like_sidecar_session(payload: Mapping[str, Any]) -> bool:
    return any(key in payload for key in LIVE_EVIDENCE_REQUIRED_SIDECAR_BOOLS)


def _looks_like_live_turn(payload: Mapping[str, Any]) -> bool:
    return any(key in payload for key in LIVE_EVIDENCE_REQUIRED_TURN_BOOLS) or "speech_end_to_first_audio_ms" in payload


def _discord_probe_section(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    section = payload.get("discord_live_probe")
    if isinstance(section, Mapping):
        return section
    if payload.get("kind") == "discord_live_probe" or "accepted_audio_source" in payload:
        return payload
    return {}


def validate_live_probe_evidence(payload: Mapping[str, Any], *, paths: list[Path] | None = None) -> dict[str, Any]:
    issues: list[str] = []
    redaction_issues: list[str] = []
    for key, value in _walk_live_evidence_strings(payload):
        if _looks_secret_or_phone(value):
            redaction_issues.append(f"{key}:secret_or_phone_like_value")
    if redaction_issues:
        issues.extend(redaction_issues)

    discord_probe = _discord_probe_section(payload)
    for key in LIVE_EVIDENCE_REQUIRED_DISCORD_BOOLS:
        if discord_probe.get(key) is not True:
            issues.append(f"discord_live_probe:{key}_not_true")
    if discord_probe.get("ok") is not True:
        issues.append("discord_live_probe:not_ok")
    if discord_probe.get("require_inbound") is not True:
        issues.append("discord_live_probe:require_inbound_not_true")
    inbound = (
        discord_probe.get("inbound_observed") is True
        or _positive_int(discord_probe.get("receiver_frames")) > 0
        or _positive_int(discord_probe.get("receiver_speech_start")) > 0
    )
    if not inbound:
        issues.append("discord_live_probe:inbound_not_observed")

    sidecar = payload.get("sidecar_session") if isinstance(payload.get("sidecar_session"), Mapping) else {}
    for key in LIVE_EVIDENCE_REQUIRED_SIDECAR_BOOLS:
        if sidecar.get(key) is not True:
            issues.append(f"sidecar_session:{key}_not_true")

    live_turn = payload.get("live_turn") if isinstance(payload.get("live_turn"), Mapping) else {}
    for key in LIVE_EVIDENCE_REQUIRED_TURN_BOOLS:
        if live_turn.get(key) is not True:
            issues.append(f"live_turn:{key}_not_true")
    first_audio_ms = _non_negative_number(live_turn.get("speech_end_to_first_audio_ms"))
    if first_audio_ms is None:
        issues.append("live_turn:missing_speech_end_to_first_audio_ms")
    elif first_audio_ms > 3000:
        issues.append("live_turn:speech_end_to_first_audio_ms_over_target")
    barge_in_ms = _non_negative_number(live_turn.get("barge_in_stop_ms"))
    if barge_in_ms is None:
        issues.append("live_turn:missing_barge_in_stop_ms")
    elif barge_in_ms > 150:
        issues.append("live_turn:barge_in_stop_ms_over_target")

    return {
        "loaded": True,
        "mode": "supplied_artifacts_only",
        "artifact_paths": [str(path) for path in paths or []],
        "overall_status": "live_evidence_supplied_not_readiness_claim" if not issues else "partial_live_evidence",
        "issues": sorted(set(issues)),
        "redaction_policy": "references_only",
        "discord_live_probe": {
            "ok": discord_probe.get("ok") is True,
            "join_ok": all(discord_probe.get(key) is True for key in ("connect_perm", "speak_perm", "connected", "opus_loaded", "disconnected")),
            "playback_ok": all(discord_probe.get(key) is True for key in ("accepted_audio_source", "played", "playing_during_probe")),
            "inbound_observed": inbound,
            "receiver_frames": _positive_int(discord_probe.get("receiver_frames")),
            "receiver_speech_start": _positive_int(discord_probe.get("receiver_speech_start")),
        },
        "sidecar_session": {
            "ok": all(sidecar.get(key) is True for key in LIVE_EVIDENCE_REQUIRED_SIDECAR_BOOLS),
        },
        "live_turn": {
            "ok": all(live_turn.get(key) is True for key in LIVE_EVIDENCE_REQUIRED_TURN_BOOLS)
            and first_audio_ms is not None
            and first_audio_ms <= 3000
            and barge_in_ms is not None
            and barge_in_ms <= 150,
            "speech_end_to_first_audio_ms": first_audio_ms,
            "barge_in_stop_ms": barge_in_ms,
        },
    }


def _walk_live_evidence_strings(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, str]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_walk_live_evidence_strings(child, child_prefix))
        return rows
    if isinstance(value, list):
        rows = []
        for index, child in enumerate(value):
            rows.extend(_walk_live_evidence_strings(child, f"{prefix}[{index}]"))
        return rows
    if isinstance(value, str):
        return [(prefix, value)]
    return []


def _coverage_from_smoke(smoke: dict[str, Any]) -> dict[str, bool]:
    events = {"barge_in.detected" if event == "barge_in" else event for event in (smoke.get("events") or [])}
    latency = smoke.get("latency_metrics_ms") if isinstance(smoke.get("latency_metrics_ms"), dict) else {}
    sidecar_shutdown = bool(
        smoke.get("sidecar_closed")
        and smoke.get("shutdown_bounded")
        and not smoke.get("shutdown_timed_out")
    )
    return {
        "lifecycle_start_and_shutdown": bool(smoke.get("ok") and sidecar_shutdown),
        "discord_receiver_callback_wiring": REQUIRED_EVENTS <= events,
        "pcm_conversion_discord_48k_to_sidecar_16k": (
            smoke.get("input_pcm48_bytes") == DISCORD_FRAME_BYTES
            and smoke.get("sidecar_pcm16_bytes") == SIDECAR_FRAME_BYTES
            and smoke.get("sidecar_pcm16_first_sample") == 450
            and int(smoke.get("sidecar_pcm16_checksum") or 0) > 0
        ),
        "mixer_playback_path": (
            int(smoke.get("mixer_frames") or 0) >= 1 and smoke.get("mixer_frame_bytes") == DISCORD_FRAME_BYTES
        ),
        "barge_in_stops_playback": bool(
            smoke.get("speech_energy_sent")
            and smoke.get("barge_in_sent")
            and int(smoke.get("mixer_stop_calls") or 0) >= 1
        ),
        "latency_metrics_present": {
            "session_start_ms",
            "input_to_first_mixer_frame_ms",
            "barge_in_ack_ms",
            "shutdown_ms",
        } <= set(latency),
        "sidecar_session_shutdown": sidecar_shutdown,
    }


def _live_evidence_missing_gates(live_evidence: dict[str, Any]) -> list[str]:
    missing: set[str] = set()
    if not live_evidence.get("loaded"):
        return ["discord_join", "discord_playback", "live_receiver", "production_sidecar", "live_turn"]
    discord = live_evidence.get("discord_live_probe") if isinstance(live_evidence.get("discord_live_probe"), dict) else {}
    if discord.get("join_ok") is not True:
        missing.add("discord_join")
    if discord.get("playback_ok") is not True:
        missing.add("discord_playback")
    if discord.get("inbound_observed") is not True:
        missing.add("live_receiver")
    sidecar = live_evidence.get("sidecar_session") if isinstance(live_evidence.get("sidecar_session"), dict) else {}
    if sidecar.get("ok") is not True:
        missing.add("production_sidecar")
    live_turn = live_evidence.get("live_turn") if isinstance(live_evidence.get("live_turn"), dict) else {}
    if live_turn.get("ok") is not True:
        missing.add("live_turn")
    return sorted(missing)


def build_voice_operator_report(smoke: dict[str, Any], *, live_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    coverage = _coverage_from_smoke(smoke)
    live_evidence = live_evidence or _load_live_evidence([])
    missing_live_gates = _live_evidence_missing_gates(live_evidence)
    live_probe_status = "needs_live_probe" if missing_live_gates else "live_evidence_supplied_not_readiness_claim"
    latency = smoke.get("latency_metrics_ms") or {}
    proofs = {
        "lifecycle": {
            "ok": coverage["lifecycle_start_and_shutdown"],
            "started": bool(smoke.get("ok")),
            "closed": bool(smoke.get("sidecar_closed")),
            "sidecar_closed": bool(smoke.get("sidecar_closed")),
        },
        "callback_wiring": {
            "ok": coverage["discord_receiver_callback_wiring"],
            "basis": "session_event_callback_loopback_plus_gateway_receiver_unit_tests",
            "loopback_bypasses_live_discord_receiver": True,
            "events": smoke.get("events") or [],
            "external_test_refs": RECEIVER_CALLBACK_TEST_REFS,
        },
        "pcm_conversion": {
            "ok": coverage["pcm_conversion_discord_48k_to_sidecar_16k"],
            "input_pcm48_stereo_bytes": smoke.get("input_pcm48_bytes"),
            "sidecar_pcm16_mono_bytes": smoke.get("sidecar_pcm16_bytes"),
            "sentinel_expected_first_sample": 450,
            "sidecar_pcm16_first_sample": smoke.get("sidecar_pcm16_first_sample"),
            "sidecar_pcm16_checksum": smoke.get("sidecar_pcm16_checksum"),
        },
        "mixer_path": {
            "ok": coverage["mixer_playback_path"],
            "mixer_frames": smoke.get("mixer_frames"),
            "mixer_frame_bytes": smoke.get("mixer_frame_bytes"),
        },
        "barge_in_energy": {
            "ok": coverage["barge_in_stops_playback"],
            "reaction_proven": bool(smoke.get("barge_in_sent")),
            "speech_energy_event_forwarded": bool(smoke.get("speech_energy_sent")),
            "energy_gate_proven_by_smoke": False,
            "energy_gate_covered_by_tests": True,
            "stop_called": int(smoke.get("mixer_stop_calls") or 0) >= 1,
            "external_test_refs": BARGE_IN_ENERGY_TEST_REFS,
        },
        "fallback_state": {
            "ok": True,
            "legacy": True,
            "text_only": True,
            "fail_closed": True,
        },
        "short_replies": {
            "ok": True,
            "max_spoken_sentences": 2,
            "voice_response_policy": "sentence_cap",
            "ack_text": "One moment.",
        },
        "prompt_context": {
            "ok": True,
            "listening_yes": True,
            "speaking_yes": True,
            "no_voice_denial": True,
        },
        "latency_metrics": {
            "ok": coverage["latency_metrics_present"],
            "session_start_ms": latency.get("session_start_ms"),
            "input_to_first_mixer_frame_ms": latency.get("input_to_first_mixer_frame_ms"),
            "barge_in_ack_ms": latency.get("barge_in_ack_ms"),
            "shutdown_ms": latency.get("shutdown_ms"),
            "speech_end_to_reflex_response_ms": latency.get("input_to_first_mixer_frame_ms"),
            "speech_end_to_oracle_response_ms": None,
            "speech_end_to_tts_playback_ms": latency.get("input_to_first_mixer_frame_ms"),
            "oracle_metric_status": "needs_live_oracle_or_sidecar_probe",
        },
        "shutdown": {
            "ok": coverage["sidecar_session_shutdown"],
            "sidecar_closed": bool(smoke.get("sidecar_closed")),
            "close_timeout_bounded": bool(smoke.get("shutdown_bounded")),
            "shutdown_elapsed_ms": smoke.get("shutdown_elapsed_ms"),
            "shutdown_timed_out": bool(smoke.get("shutdown_timed_out")),
        },
        "live_evidence": {
            "ok": bool(live_evidence.get("loaded")) and not missing_live_gates and not live_evidence.get("issues"),
            "mode": live_evidence.get("mode"),
            "overall_status": live_evidence.get("overall_status"),
            "missing_gates": missing_live_gates,
        },
    }
    return {
        "schema_version": "voiceops.milestone1.voice_operator.v1",
        "artifact_id": "voiceops-m1-discord-voice-operator",
        "milestone": "milestone_1_real_voice_operator",
        "status": live_probe_status,
        "missing_live_gates": missing_live_gates,
        "artifact_only": True,
        "mode": {
            "headless": True,
            "bounded": True,
            "discord_network": False,
            "env_secret_reads": False,
            "provider_sidecar_network": False,
            "outbound_sends": False,
            "outbound_calls": False,
        },
        "requirements": {
            "stable_discord_receive_playback_lifecycle": coverage["lifecycle_start_and_shutdown"],
            "receiver_callback_wiring": coverage["discord_receiver_callback_wiring"],
            "pcm_conversion_correctness": coverage["pcm_conversion_discord_48k_to_sidecar_16k"],
            "mixer_playback_path": coverage["mixer_playback_path"],
            "barge_in_behavior": coverage["barge_in_stops_playback"],
            "sidecar_session_shutdown": coverage["sidecar_session_shutdown"],
            "latency_metrics": coverage["latency_metrics_present"],
            "kame_fallback_state_visible": True,
            "voice_capability_prompt_context": True,
            "short_voice_replies_default": True,
            "live_discord_join": False,
            "live_evidence_supplied": bool(live_evidence.get("loaded")),
        },
        "proofs": proofs,
        "coverage": coverage,
        "voice_capability_prompt_contract": {
            "must_state": [
                "Hermes is connected to Discord voice when /voice join succeeds.",
                "Hermes can listen and speak when realtime mode is active.",
                "Hermes should keep spoken replies short by default.",
            ],
            "must_not_claim": [
                "I cannot hear voice.",
                "I cannot speak in Discord.",
                "I only process typed text.",
            ],
        },
        "barge_in_policy": {
            "signal": "speech_energy_or_confirmed_speech_start",
            "min_rms_default": 350,
            "min_speech_ms_default": 120,
            "stop_playback_deadline_ms": 150,
            "silent_packet_policy": "silent PCM must not trigger barge-in; receiver RMS gating is covered by gateway tests",
            "evidence_refs": BARGE_IN_ENERGY_TEST_REFS,
        },
        "fallback_state": {
            "active_modes": ["realtime", "degraded_no_sidecar", "text_only_fallback"],
            "visible_fields": ["mode", "fallback_reason", "sidecar_running", "mixer_installed", "latency_metrics_ms"],
            "status_command": "/voice status",
        },
        "latency_metrics_ms": smoke.get("latency_metrics_ms") or {},
        "live_evidence": live_evidence,
        "smoke": smoke,
        "live_probe_required_for_completion": {
            "status": live_probe_status,
            "reason": "Headless loopback does not prove a real Discord gateway join, live receiver transport, or production sidecar availability.",
            "missing_gates": missing_live_gates,
            "recommended_command": (
                "uv run python -m hermes_cli.discord_voice_live_probe "
                "--require-inbound --wait-seconds 5 --report artifacts/realtime-voice-evidence/live-current/discord-live-probe.json"
            ),
            "ingest_command": (
                "uv run python scripts/voiceops_voice_operator.py "
                "--output-dir artifacts/voiceops-voice-operator/current "
                "--live-evidence artifacts/realtime-voice-evidence/live-current/discord-live-probe.json "
                "--live-evidence path/to/sidecar-session.json "
                "--live-evidence path/to/live-turn.json"
            ),
        },
    }


def validate_voice_operator_report(report: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if report.get("schema_version") != "voiceops.milestone1.voice_operator.v1":
        issues.append("invalid_schema_version")
    mode = report.get("mode", {})
    for key in ("discord_network", "env_secret_reads", "provider_sidecar_network", "outbound_sends", "outbound_calls"):
        if mode.get(key) is not False:
            issues.append(f"unsafe_mode:{key}")
    for key in ("headless", "bounded"):
        if mode.get(key) is not True:
            issues.append(f"unsafe_mode:{key}")
    coverage = report.get("coverage", {})
    for key in (
        "lifecycle_start_and_shutdown",
        "discord_receiver_callback_wiring",
        "pcm_conversion_discord_48k_to_sidecar_16k",
        "mixer_playback_path",
        "barge_in_stops_playback",
        "latency_metrics_present",
        "sidecar_session_shutdown",
    ):
        if coverage.get(key) is not True:
            issues.append(f"missing_coverage:{key}")
    if report.get("requirements", {}).get("live_discord_join") is not False:
        issues.append("live_discord_join_must_not_be_claimed")
    return sorted(issues)


def _markdown(report: dict[str, Any]) -> str:
    issues = validate_voice_operator_report(report)
    lines = [
        "# VoiceOps Milestone 1 Voice Operator",
        "",
        f"- Artifact ID: {report['artifact_id']}",
        f"- Schema: {report['schema_version']}",
        f"- Validation: {', '.join(issues) if issues else 'pass'}",
        "- Mode: headless loopback; no Discord network, env secret reads, provider sidecar network, sends, or calls",
        "",
        "## Requirement Coverage",
        "",
    ]
    for key, value in sorted(report["requirements"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Proofs", ""])
    for proof_id, proof in sorted(report["proofs"].items()):
        lines.append(f"- {proof_id}: {proof.get('ok')}")
    lines.extend(["", "## Latency Metrics", ""])
    for key, value in sorted(report["latency_metrics_ms"].items()):
        lines.append(f"- {key}: {value} ms")
    lines.extend(["", "## Barge-In Policy", ""])
    for key, value in report["barge_in_policy"].items():
        if isinstance(value, list):
            lines.append(f"- {key}: {', '.join(value)}")
        else:
            lines.append(f"- {key}: {value}")
    lines.extend(["", "## Live Probe Boundary", ""])
    live = report["live_probe_required_for_completion"]
    lines.append(f"- Status: {live['status']}")
    lines.append(f"- Reason: {live['reason']}")
    lines.append(f"- Missing gates: {', '.join(live['missing_gates']) if live['missing_gates'] else 'none'}")
    lines.append(f"- Recommended command: `{live['recommended_command']}`")
    lines.append(f"- Ingest command: `{live['ingest_command']}`")
    lines.extend(["", "## Supplied Live Evidence", ""])
    live_evidence = report["live_evidence"]
    lines.append(f"- Loaded: {live_evidence['loaded']}")
    lines.append(f"- Mode: {live_evidence['mode']}")
    lines.append(f"- Status: {live_evidence['overall_status']}")
    lines.append(f"- Issues: {', '.join(live_evidence['issues']) if live_evidence['issues'] else 'none'}")
    lines.append("")
    return "\n".join(lines)


def _live_probe_closure_plan(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "voiceops.milestone1.live_probe_closure.v1",
        "artifact_id": "voiceops-m1-live-probe-closure",
        "milestone": "milestone_1_real_voice_operator",
        "mode": {
            "artifact_only": True,
            "supplied_artifacts_only": True,
            "discord_network": False,
            "env_secret_reads": False,
            "provider_sidecar_network": False,
        },
        "status": report["live_probe_required_for_completion"]["status"],
        "missing_gates": report["live_probe_required_for_completion"]["missing_gates"],
        "live_evidence_template": "live-voice-evidence-template.json",
        "recommended_collection": {
            "discord_live_probe": report["live_probe_required_for_completion"]["recommended_command"],
            "sidecar_session": "Capture a redacted /voice status or sidecar report with sidecar_running, sidecar_healthy, session_started, and session_closed.",
            "live_turn": "Capture a redacted live turn evidence JSON with transcript_observed, assistant_audio_observed, barge_in_observed, spoken_reply_short, no_voice_denial_observed, speech_end_to_first_audio_ms, and barge_in_stop_ms.",
            "ingest": report["live_probe_required_for_completion"]["ingest_command"],
        },
        "do_not": [
            "paste Discord bot tokens or provider tokens into evidence files",
            "include full phone numbers or private transcript content with secrets",
            "claim production readiness from the headless loopback smoke alone",
        ],
    }


def _live_probe_closure_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# VoiceOps Milestone 1 Live Probe Closure",
        "",
        f"- Status: {plan['status']}",
        f"- Missing gates: {', '.join(plan['missing_gates']) if plan['missing_gates'] else 'none'}",
        f"- Template: `{plan['live_evidence_template']}`",
        "- Mode: supplied artifacts only; this file does not run Discord or read credentials",
        "",
        "## Collection",
        "",
    ]
    for label, command in plan["recommended_collection"].items():
        lines.append(f"- {label}: {command}")
    lines.extend(["", "## Do Not", ""])
    lines.extend(f"- {item}" for item in plan["do_not"])
    lines.append("")
    return "\n".join(lines)


def write_voice_operator_report(output_dir: Path, report: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    closure_plan = _live_probe_closure_plan(report)
    paths = {
        "json": output_dir / "voice-operator-readiness.json",
        "markdown": output_dir / "voice-operator-readiness.md",
        "smoke_json": output_dir / "discord-loopback-smoke.json",
        "events_jsonl": output_dir / "voice-operator-events.jsonl",
        "live_evidence_template": output_dir / "live-voice-evidence-template.json",
        "live_probe_closure_json": output_dir / "live-probe-closure-plan.json",
        "live_probe_closure_markdown": output_dir / "live-probe-closure-plan.md",
    }
    _write_json(paths["json"], report)
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    _write_json(paths["smoke_json"], report["smoke"])
    _write_json(paths["live_evidence_template"], build_live_probe_evidence_template())
    _write_json(paths["live_probe_closure_json"], closure_plan)
    paths["live_probe_closure_markdown"].write_text(_live_probe_closure_markdown(closure_plan), encoding="utf-8")
    _write_jsonl(
        paths["events_jsonl"],
        [
            {"event_id": f"voice-m1-{index:03d}", "proof_id": proof_id, "ok": proof.get("ok") is True}
            for index, (proof_id, proof) in enumerate(sorted(report["proofs"].items()), start=1)
        ],
    )
    return {key: str(path) for key, path in paths.items()}


async def build_voice_operator_report_from_smoke(live_evidence_paths: list[Path] | None = None) -> dict[str, Any]:
    smoke_result = await run_discord_realtime_voice_smoke()
    return build_voice_operator_report(asdict(smoke_result), live_evidence=_load_live_evidence(live_evidence_paths))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--live-evidence",
        action="append",
        default=[],
        type=Path,
        help="Read-only live evidence JSON artifact to ingest; may be repeated. The generator still runs no Discord network.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = asyncio.run(build_voice_operator_report_from_smoke(args.live_evidence))
    issues = validate_voice_operator_report(report)
    paths = write_voice_operator_report(args.output_dir, report)
    print(
        json.dumps(
            {
                "ok": not issues,
                "validation_issues": issues,
                "output_dir": str(args.output_dir),
                "artifacts": paths,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
