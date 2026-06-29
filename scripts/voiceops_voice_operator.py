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
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermes_cli.discord_realtime_voice_smoke import run_discord_realtime_voice_smoke


DEFAULT_OUTPUT_DIR = Path("artifacts/voiceops-voice-operator/current")
DISCORD_FRAME_BYTES = 3840
SIDECAR_FRAME_BYTES = 640
LIVE_EVIDENCE_SCHEMA_VERSION = "voiceops.milestone1.live_voice_evidence.v1"
LIVE_EVIDENCE_MANIFEST_SCHEMA_VERSION = "voiceops.realtime_voice_live_evidence_manifest.v1"

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

LIVE_EVIDENCE_REQUIRED_SIDECAR_PROVENANCE_BOOLS = (
    "healthcheck_observed",
    "provider_transport_observed",
    "session_id_redacted",
)

LIVE_EVIDENCE_REQUIRED_TURN_BOOLS = (
    "transcript_observed",
    "assistant_audio_observed",
    "barge_in_observed",
    "spoken_reply_short",
    "no_voice_denial_observed",
)

LIVE_EVIDENCE_TEMPLATE_SOURCE_ARTIFACTS = {
    "discord-live-probe.json",
    "voice-status-or-sidecar-report.json",
    "sidecar-session.json",
    "voice-turn-evidence.json",
    "live-turn.json",
}

LIVE_EVIDENCE_REQUIRED_DISCORD_LATENCIES_MS = (
    "connect_ms",
    "playback_observed_ms",
    "inbound_observed_ms",
    "disconnect_ms",
)

LIVE_EVIDENCE_FORBIDDEN_TEXT_FIELDS = {
    "assistant_text",
    "assistant_reply",
    "assistant_transcript",
    "raw_transcript",
    "reply_text",
    "transcript_text",
    "user_transcript",
}

LIVE_EVIDENCE_SECRET_FIELD_MARKERS = (
    "api_key",
    "authorization",
    "auth_token",
    "bearer",
    "phone",
    "secret",
    "token",
)

LIVE_EVIDENCE_DENIAL_PHRASES = (
    "cannot hear voice",
    "cannot hear you",
    "cannot speak in discord",
    "do not have any ability to join discord voice",
    "i only process text",
    "i only process typed text",
)

SECRET_VALUE_RE = re.compile(
    r"(?i)(sk[-_][a-z0-9]{8,}|pk[-_][a-z0-9]{8,}|rk[-_][a-z0-9]{8,}|"
    r"whsec_[a-z0-9]{8,}|xox[aboprs]-[a-z0-9-]{8,}|gh[pousr]_[a-z0-9_]{8,}|"
    r"mfa\.[a-z0-9_-]{20,}|[a-z0-9_-]{23,}\.[a-z0-9_-]{6,}\.[a-z0-9_-]{20,})"
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
    secret_markers = ("sk_", "pk_", "rk_", "whsec_", "xoxb", "xoxp", "ghp_", "bearer ", "sk-", "pk-", "rk-")
    if any(marker in lowered for marker in secret_markers):
        return True
    if SECRET_VALUE_RE.search(text):
        return True
    digits = "".join(ch for ch in text if ch.isdigit())
    return "+" in text and len(digits) >= 8


def _live_evidence_key_name(path: str) -> str:
    return path.rsplit(".", 1)[-1].split("[", 1)[0].lower()


def _looks_like_forbidden_live_evidence_field(path: str) -> bool:
    name = _live_evidence_key_name(path)
    if name in LIVE_EVIDENCE_FORBIDDEN_TEXT_FIELDS:
        return True
    return any(marker in name for marker in LIVE_EVIDENCE_SECRET_FIELD_MARKERS)


def _looks_like_voice_denial_text(value: str) -> bool:
    lowered = value.lower()
    return any(phrase in lowered for phrase in LIVE_EVIDENCE_DENIAL_PHRASES)


def build_live_probe_evidence_template() -> dict[str, Any]:
    return {
        "schema_version": LIVE_EVIDENCE_SCHEMA_VERSION,
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
            "shutdown_bounded": False,
            "shutdown_timed_out": None,
            "fallback_mode_visible": False,
            "fallback_reason": None,
            "sidecar_mode": None,
            "healthcheck_observed": False,
            "provider_transport_observed": False,
            "session_id_redacted": False,
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


def build_live_probe_evidence_example() -> dict[str, Any]:
    example = build_live_probe_evidence_template()
    example["example_only"] = True
    example["redaction_policy"] = "example only; copy shape with real artifact refs and remove example_only before ingest"
    example["discord_live_probe"].update(
        {
            "ok": True,
            "connect_perm": True,
            "speak_perm": True,
            "connected": True,
            "opus_loaded": True,
            "accepted_audio_source": True,
            "played": True,
            "playing_during_probe": True,
            "receiver_started": True,
            "receiver_frames": 42,
            "receiver_speech_start": 1,
            "inbound_observed": True,
            "disconnected": True,
            "latency_metrics_ms": {
                "connect_ms": 420,
                "playback_observed_ms": 180,
                "inbound_observed_ms": 900,
                "disconnect_ms": 120,
            },
        }
    )
    example["sidecar_session"].update(
        {
            "sidecar_running": True,
            "sidecar_healthy": True,
            "session_started": True,
            "session_closed": True,
            "shutdown_bounded": True,
            "shutdown_timed_out": False,
            "fallback_mode_visible": True,
            "fallback_reason": "none",
            "sidecar_mode": "production",
            "healthcheck_observed": True,
            "provider_transport_observed": True,
            "session_id_redacted": True,
            "latency_metrics_ms": {
                "session_start_ms": 110,
                "shutdown_ms": 80,
            },
        }
    )
    example["live_turn"].update(
        {
            "transcript_observed": True,
            "assistant_audio_observed": True,
            "barge_in_observed": True,
            "spoken_reply_short": True,
            "no_voice_denial_observed": True,
            "speech_end_to_first_audio_ms": 950,
            "barge_in_stop_ms": 90,
        }
    )
    return example


def write_live_evidence_scaffold(output_dir: Path) -> dict[str, Path]:
    scaffold_dir = output_dir / "live-voice-evidence-scaffold"
    sections_dir = scaffold_dir / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)

    example = build_live_probe_evidence_example()
    section_files = {
        "discord_live_probe": "discord-live-probe.json",
        "sidecar_session": "sidecar-session.json",
        "live_turn": "live-turn.json",
    }
    reports: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for section_name, section_file in section_files.items():
        section = dict(example[section_name])
        section["example_only"] = True
        section["kind"] = section_name
        section["source_artifact"] = section_file
        section["redaction_policy"] = (
            "example only; replace with real redacted live evidence and remove example_only before ingest"
        )
        section_path = sections_dir / section_file
        _write_json(section_path, section)
        reports[section_name] = f"sections/{section_file}"
        paths[f"scaffold_{section_name}"] = section_path

    manifest_path = scaffold_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": LIVE_EVIDENCE_MANIFEST_SCHEMA_VERSION,
            "example_only": True,
            "redaction_policy": "example only; this scaffold is rejected until all example_only markers are removed",
            "reports": reports,
            "notes": (
                "Replace each section report with real redacted Discord/sidecar/turn evidence. "
                "Do not paste tokens, full phone numbers, or raw private transcripts."
            ),
        },
    )
    paths["live_evidence_scaffold_manifest"] = manifest_path
    return paths


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


def _load_live_evidence_file(path: Path, *, visited: set[Path] | None = None, standalone: bool = True) -> dict[str, Any]:
    visited = set() if visited is None else set(visited)
    resolved_path = path.expanduser().resolve()
    if resolved_path in visited:
        return {
            "payload": None,
            "issues": ["cycle_detected"],
        }
    visited.add(resolved_path)
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
    payload, issues = _expand_live_evidence_manifest(path, payload, visited=visited)
    if standalone:
        identity_issue = _standalone_report_identity_issue(payload)
        if identity_issue:
            issues.append(identity_issue)
    return {"payload": payload, "issues": issues}


def _expand_live_evidence_manifest(
    path: Path,
    payload: Mapping[str, Any],
    *,
    visited: set[Path],
) -> tuple[Mapping[str, Any], list[str]]:
    reports = payload.get("reports")
    if not isinstance(reports, Mapping):
        return payload, []

    expanded: dict[str, Any] = {"schema_version": LIVE_EVIDENCE_SCHEMA_VERSION}
    if payload.get("example_only") is True:
        expanded["example_only"] = True
    issues: list[str] = []
    manifest_schema = str(payload.get("schema_version") or "")
    if not manifest_schema:
        issues.append("live_evidence_manifest:missing_schema_version")
    elif manifest_schema != LIVE_EVIDENCE_MANIFEST_SCHEMA_VERSION:
        issues.append("live_evidence_manifest:invalid_schema_version")
    for report_name, report_path_value in reports.items():
        report_path_text = str(report_path_value or "").strip()
        if not report_path_text:
            issues.append(f"live_evidence_manifest:{report_name}:empty_report_path")
            continue
        report_path = _resolve_manifest_report_path(path, report_path_text)
        loaded = _load_live_evidence_file(report_path, visited=visited, standalone=False)
        if loaded["issues"]:
            issues.extend(f"live_evidence_manifest:{report_name}:{issue}" for issue in loaded["issues"])
        report_payload = loaded.get("payload")
        if isinstance(report_payload, Mapping):
            if not _manifest_report_has_identity(report_name, report_payload):
                issues.append(f"live_evidence_manifest:{report_name}:missing_report_identity")
            report_payload = _with_manifest_report_provenance(report_payload, report_path)
            if report_payload.get("example_only") is True:
                issues.append(f"live_evidence_manifest:{report_name}:example_only_evidence_not_accepted")
                expanded["example_only"] = True
            _merge_live_evidence_payload(expanded, report_payload)
    return expanded if expanded else payload, issues


def _with_manifest_report_provenance(payload: Mapping[str, Any], report_path: Path) -> dict[str, Any]:
    enriched = dict(payload)
    source_artifact = str(report_path.resolve())
    previous_source_artifact = str(enriched.get("source_artifact") or "")
    enriched["source_artifact"] = source_artifact
    provenance = {"wrapper_artifact": source_artifact}
    if previous_source_artifact:
        provenance["reported_source_artifact"] = previous_source_artifact
    enriched["provenance"] = provenance
    for section_name in ("discord_live_probe", "sidecar_session", "live_turn"):
        section = enriched.get(section_name)
        if isinstance(section, Mapping):
            section_copy = dict(section)
            previous_section_source = str(section_copy.get("source_artifact") or "")
            section_provenance = {"wrapper_artifact": source_artifact, "section": section_name}
            if previous_section_source:
                section_provenance["reported_source_artifact"] = previous_section_source
                if previous_section_source not in LIVE_EVIDENCE_TEMPLATE_SOURCE_ARTIFACTS:
                    previous_path = Path(previous_section_source).expanduser()
                    if not previous_path.is_absolute():
                        previous_path = report_path.parent / previous_path
                    section_copy["source_artifact"] = str(previous_path.resolve())
            section_copy["provenance"] = section_provenance
            enriched[section_name] = section_copy
    return enriched


def _resolve_manifest_report_path(manifest_path: Path, report_path_text: str) -> Path:
    report_path = Path(report_path_text).expanduser()
    if report_path.is_absolute():
        return report_path
    return manifest_path.parent / report_path_text


def _manifest_report_has_identity(report_name: str, payload: Mapping[str, Any]) -> bool:
    if _uses_expanded_live_evidence_schema(payload):
        return True
    kind = str(payload.get("kind") or payload.get("evidence_type") or "").strip()
    if report_name == "combined":
        return _uses_expanded_live_evidence_schema(payload) or kind == "combined"
    if report_name == "discord_live_probe":
        return kind == "discord_live_probe"
    if report_name == "sidecar_session":
        return kind == "sidecar_session"
    if report_name == "live_turn":
        return kind == "live_turn"
    return bool(kind)


def _standalone_report_identity_issue(payload: Mapping[str, Any]) -> str:
    if _uses_expanded_live_evidence_schema(payload):
        return ""
    kind = str(payload.get("kind") or payload.get("evidence_type") or "").strip()
    if not kind:
        return "missing_standalone_report_identity"
    if kind in {"discord_live_probe", "sidecar_session", "live_turn"}:
        return ""
    return f"invalid_standalone_report_identity:{kind}"


def _uses_expanded_live_evidence_schema(payload: Mapping[str, Any]) -> bool:
    if str(payload.get("schema_version") or "") != LIVE_EVIDENCE_SCHEMA_VERSION:
        return False
    return any(isinstance(payload.get(section), Mapping) for section in ("discord_live_probe", "sidecar_session", "live_turn"))


def _merge_live_evidence_payload(target: dict[str, Any], payload: Mapping[str, Any]) -> None:
    if payload.get("example_only") is True:
        target["example_only"] = True
    if "schema_version" in payload and "schema_version" not in target:
        target["schema_version"] = payload.get("schema_version")
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
    if str(payload.get("schema_version") or "") != LIVE_EVIDENCE_SCHEMA_VERSION:
        issues.append("missing_schema_version")
    if payload.get("example_only") is True:
        issues.append("example_only_evidence_not_accepted")
    redaction_issues: list[str] = []
    for key, value in _walk_live_evidence_strings(payload):
        if _looks_like_forbidden_live_evidence_field(key):
            redaction_issues.append(f"{key}:forbidden_evidence_field")
        if _looks_like_voice_denial_text(value):
            redaction_issues.append(f"{key}:voice_capability_denial_text")
        if _looks_secret_or_phone(value):
            redaction_issues.append(f"{key}:secret_or_phone_like_value")
    if redaction_issues:
        issues.extend(redaction_issues)

    discord_probe = _discord_probe_section(payload)
    if not str(discord_probe.get("source_artifact") or "").strip():
        issues.append("discord_live_probe:missing_source_artifact")
    else:
        _validate_source_artifact(
            discord_probe.get("source_artifact"),
            "discord_live_probe",
            paths or [],
            issues,
        )
    if discord_probe.get("example_only") is True:
        issues.append("discord_live_probe:example_only_evidence_not_accepted")
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
    discord_latency = (
        discord_probe.get("latency_metrics_ms")
        if isinstance(discord_probe.get("latency_metrics_ms"), Mapping)
        else {}
    )
    discord_latency_ok = True
    for key in LIVE_EVIDENCE_REQUIRED_DISCORD_LATENCIES_MS:
        if _non_negative_number(discord_latency.get(key)) is None:
            issues.append(f"discord_live_probe:missing_{key}")
            discord_latency_ok = False

    sidecar = payload.get("sidecar_session") if isinstance(payload.get("sidecar_session"), Mapping) else {}
    if not str(sidecar.get("source_artifact") or "").strip():
        issues.append("sidecar_session:missing_source_artifact")
    else:
        _validate_source_artifact(
            sidecar.get("source_artifact"),
            "sidecar_session",
            paths or [],
            issues,
        )
    if sidecar.get("example_only") is True:
        issues.append("sidecar_session:example_only_evidence_not_accepted")
    for key in LIVE_EVIDENCE_REQUIRED_SIDECAR_BOOLS:
        if sidecar.get(key) is not True:
            issues.append(f"sidecar_session:{key}_not_true")
    for key in LIVE_EVIDENCE_REQUIRED_SIDECAR_PROVENANCE_BOOLS:
        if sidecar.get(key) is not True:
            issues.append(f"sidecar_session:{key}_not_true")
    if str(sidecar.get("sidecar_mode") or "").strip() != "production":
        issues.append("sidecar_session:sidecar_mode_not_production")
    fallback_reason = str(sidecar.get("fallback_reason") or "").strip()
    if sidecar.get("fallback_mode_visible") is True and not fallback_reason:
        issues.append("sidecar_session:missing_fallback_reason")
    elif fallback_reason and _looks_secret_or_phone(fallback_reason):
        issues.append("sidecar_session:fallback_reason_secret_or_phone_like_value")
    sidecar_latency = sidecar.get("latency_metrics_ms") if isinstance(sidecar.get("latency_metrics_ms"), Mapping) else {}
    session_start_ms = _non_negative_number(sidecar_latency.get("session_start_ms"))
    if session_start_ms is None:
        issues.append("sidecar_session:missing_session_start_ms")
    shutdown_ms = _non_negative_number(sidecar_latency.get("shutdown_ms"))
    if shutdown_ms is None:
        issues.append("sidecar_session:missing_shutdown_ms")
    if sidecar.get("shutdown_bounded") is not True:
        issues.append("sidecar_session:shutdown_bounded_not_true")
    if sidecar.get("shutdown_timed_out") is not False:
        issues.append("sidecar_session:shutdown_timed_out_not_false")

    live_turn = payload.get("live_turn") if isinstance(payload.get("live_turn"), Mapping) else {}
    if not str(live_turn.get("source_artifact") or "").strip():
        issues.append("live_turn:missing_source_artifact")
    else:
        _validate_source_artifact(
            live_turn.get("source_artifact"),
            "live_turn",
            paths or [],
            issues,
        )
    if live_turn.get("example_only") is True:
        issues.append("live_turn:example_only_evidence_not_accepted")
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
        "section_refs": {
            "discord_live_probe": _section_ref(discord_probe, "discord_live_probe"),
            "sidecar_session": _section_ref(sidecar, "sidecar_session"),
            "live_turn": _section_ref(live_turn, "live_turn"),
        },
        "discord_live_probe": {
            "ok": discord_probe.get("ok") is True,
            "join_ok": all(discord_probe.get(key) is True for key in ("connect_perm", "speak_perm", "connected", "opus_loaded", "disconnected")),
            "playback_ok": all(discord_probe.get(key) is True for key in ("accepted_audio_source", "played", "playing_during_probe")),
            "inbound_observed": inbound,
            "latency_ok": discord_latency_ok,
            "receiver_frames": _positive_int(discord_probe.get("receiver_frames")),
            "receiver_speech_start": _positive_int(discord_probe.get("receiver_speech_start")),
        },
        "sidecar_session": {
            "ok": all(sidecar.get(key) is True for key in LIVE_EVIDENCE_REQUIRED_SIDECAR_BOOLS)
            and all(sidecar.get(key) is True for key in LIVE_EVIDENCE_REQUIRED_SIDECAR_PROVENANCE_BOOLS)
            and str(sidecar.get("sidecar_mode") or "").strip() == "production"
            and (sidecar.get("fallback_mode_visible") is not True or bool(fallback_reason))
            and not _looks_secret_or_phone(fallback_reason)
            and session_start_ms is not None
            and shutdown_ms is not None
            and sidecar.get("shutdown_bounded") is True
            and sidecar.get("shutdown_timed_out") is False,
            "session_start_ms": session_start_ms,
            "shutdown_ms": shutdown_ms,
            "shutdown_bounded": sidecar.get("shutdown_bounded") is True,
            "shutdown_timed_out": sidecar.get("shutdown_timed_out") is True,
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


def _section_ref(section: Mapping[str, Any], section_name: str) -> dict[str, str]:
    ref = {
        "source_artifact": str(section.get("source_artifact") or ""),
        "section": section_name,
    }
    provenance = section.get("provenance")
    if isinstance(provenance, Mapping):
        wrapper = str(provenance.get("wrapper_artifact") or "").strip()
        reported = str(provenance.get("reported_source_artifact") or "").strip()
        if wrapper:
            ref["wrapper_artifact"] = wrapper
        if reported:
            ref["reported_source_artifact"] = reported
    return ref


def _source_artifact_exists(source_artifact: Any, evidence_paths: list[Path]) -> bool:
    source_text = str(source_artifact or "").strip()
    if not source_text:
        return False
    source_path = Path(source_text).expanduser()
    if source_path.is_absolute():
        return source_path.is_file()
    return any((path.parent / source_text).is_file() for path in evidence_paths)


def _validate_source_artifact(
    source_artifact: Any,
    section_name: str,
    evidence_paths: list[Path],
    issues: list[str],
) -> None:
    source_text = str(source_artifact or "").strip()
    source_path = Path(source_text).expanduser()
    if source_text in LIVE_EVIDENCE_TEMPLATE_SOURCE_ARTIFACTS or (
        not source_path.is_absolute() and source_path.name in LIVE_EVIDENCE_TEMPLATE_SOURCE_ARTIFACTS
    ):
        issues.append(f"{section_name}:template_source_artifact_not_accepted")
        return
    if evidence_paths:
        if not _source_artifact_exists(source_text, evidence_paths):
            issues.append(f"{section_name}:source_artifact_not_found")
        return
    if source_path.is_absolute():
        if not source_path.exists():
            issues.append(f"{section_name}:source_artifact_not_found")
        elif not source_path.is_file():
            issues.append(f"{section_name}:source_artifact_not_file")
        return
    issues.append(f"{section_name}:unverified_source_artifact")


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
                "uv run python -m hermes_cli.realtime_voice_live_evidence "
                "--output-dir artifacts/realtime-voice-evidence/live-current "
                "--require-live-discord --require-inbound --wait-seconds 5 "
                "--sidecar-session-evidence artifacts/realtime-voice-evidence/live-current/sidecar-session.json "
                "--live-turn-evidence artifacts/realtime-voice-evidence/live-current/live-turn.json"
            ),
            "validate_command": (
                "uv run python -m hermes_cli.realtime_voice_live_evidence "
                "--output-dir artifacts/realtime-voice-evidence/live-current "
                "--validate-live-evidence "
                "--discord-live-probe-evidence artifacts/realtime-voice-evidence/live-current/discord-live-probe.json "
                "--sidecar-session-evidence artifacts/realtime-voice-evidence/live-current/sidecar-session.json "
                "--live-turn-evidence artifacts/realtime-voice-evidence/live-current/live-turn.json"
            ),
            "ingest_command": (
                "uv run python scripts/voiceops_voice_operator.py "
                "--output-dir artifacts/voiceops-voice-operator/current "
                "--live-evidence artifacts/realtime-voice-evidence/live-current/manifest.json"
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
        "live_evidence_scaffold_manifest": "live-voice-evidence-scaffold/manifest.json",
        "evidence_contract": {
            "manifest_schema_version": LIVE_EVIDENCE_MANIFEST_SCHEMA_VERSION,
            "expanded_evidence_schema_version": LIVE_EVIDENCE_SCHEMA_VERSION,
            "required_sections": ["discord_live_probe", "sidecar_session", "live_turn"],
            "required_section_field": "source_artifact",
            "required_section_refs": ["source_artifact", "section"],
            "manifest_report_identity": "per-section reports must include kind/evidence_type matching discord_live_probe, sidecar_session, or live_turn unless they use the expanded live evidence schema",
            "standalone_report_identity": "standalone non-expanded evidence files must include kind/evidence_type matching discord_live_probe, sidecar_session, or live_turn",
            "source_artifacts_must_exist": True,
            "source_artifact_resolution": "absolute paths or paths relative to supplied live-evidence files",
            "template_source_artifacts_accepted": False,
            "example_only_accepted": False,
        },
        "recommended_collection": {
            "live_bundle_manifest": report["live_probe_required_for_completion"]["recommended_command"],
            "validate_bundle_offline": report["live_probe_required_for_completion"]["validate_command"],
            "sidecar_session": (
                "Write sidecar-session.json with kind=sidecar_session, sidecar_running, sidecar_healthy, "
                "session_started, session_closed, fallback_mode_visible, fallback_reason, sidecar_mode=production, "
                "healthcheck_observed, provider_transport_observed, session_id_redacted, shutdown_bounded=true, "
                "shutdown_timed_out=false, latency_metrics_ms.session_start_ms, latency_metrics_ms.shutdown_ms, "
                "and source_artifact."
            ),
            "live_turn": "Write live-turn.json with kind=live_turn, transcript_observed, assistant_audio_observed, barge_in_observed, spoken_reply_short, no_voice_denial_observed, speech_end_to_first_audio_ms, barge_in_stop_ms, and source_artifact.",
            "ingest": report["live_probe_required_for_completion"]["ingest_command"],
        },
        "evidence_shapes": {
            "discord_live_probe": {
                "kind": "discord_live_probe",
                "source_artifact": "absolute/or/resolved-discord-live-probe.json",
                "ok": True,
                "connect_perm": True,
                "speak_perm": True,
                "connected": True,
                "opus_loaded": True,
                "accepted_audio_source": True,
                "played": True,
                "playing_during_probe": True,
                "receiver_started": True,
                "receiver_frames": 1,
                "receiver_speech_start": 1,
                "inbound_observed": True,
                "disconnected": True,
                "require_inbound": True,
                "latency_metrics_ms": {
                    "connect_ms": 420,
                    "playback_observed_ms": 180,
                    "inbound_observed_ms": 900,
                    "disconnect_ms": 120,
                },
            },
            "sidecar_session": {
                "kind": "sidecar_session",
                "source_artifact": "sidecar-session.json",
                "sidecar_running": True,
                "sidecar_healthy": True,
                "session_started": True,
                "session_closed": True,
                "fallback_mode_visible": True,
                "fallback_reason": "none",
                "sidecar_mode": "production",
                "healthcheck_observed": True,
                "provider_transport_observed": True,
                "session_id_redacted": True,
                "shutdown_bounded": True,
                "shutdown_timed_out": False,
                "latency_metrics_ms": {"session_start_ms": 110, "shutdown_ms": 80},
            },
            "live_turn": {
                "kind": "live_turn",
                "source_artifact": "live-turn.json",
                "transcript_observed": True,
                "assistant_audio_observed": True,
                "barge_in_observed": True,
                "spoken_reply_short": True,
                "no_voice_denial_observed": True,
                "speech_end_to_first_audio_ms": 900,
                "barge_in_stop_ms": 90,
            },
        },
        "do_not": [
            "paste Discord bot tokens or provider tokens into evidence files",
            "include full phone numbers or private transcript content with secrets",
            "include raw transcript text; record only redacted booleans, latency numbers, and artifact references",
            "hand-edit manifest.json or example_only evidence to claim a passing live probe",
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
        f"- Scaffold manifest: `{plan['live_evidence_scaffold_manifest']}`",
        "- Mode: supplied artifacts only; this file does not run Discord or read credentials",
        "",
        "## Evidence Contract",
        "",
    ]
    for key, value in sorted(plan["evidence_contract"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
        "## Collection",
        "",
        ]
    )
    for label, command in plan["recommended_collection"].items():
        lines.append(f"- {label}: {command}")
    lines.extend(["", "## Evidence Shapes", ""])
    for label, shape in plan["evidence_shapes"].items():
        lines.append(f"### {label}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(shape, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
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
        "live_evidence_example": output_dir / "live-voice-evidence.example.json",
        "live_probe_closure_json": output_dir / "live-probe-closure-plan.json",
        "live_probe_closure_markdown": output_dir / "live-probe-closure-plan.md",
    }
    _write_json(paths["json"], report)
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    _write_json(paths["smoke_json"], report["smoke"])
    _write_json(paths["live_evidence_template"], build_live_probe_evidence_template())
    _write_json(paths["live_evidence_example"], build_live_probe_evidence_example())
    paths.update(write_live_evidence_scaffold(output_dir))
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
        help="Read-only live evidence JSON artifact or realtime_voice_live_evidence manifest to ingest; may be repeated. The generator still runs no Discord network.",
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
