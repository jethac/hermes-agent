"""Collect realtime voice live-evidence artifacts in one command."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RealtimeVoiceLiveEvidenceResult:
    ok: bool
    output_dir: str
    schema_version: str = "voiceops.realtime_voice_live_evidence_manifest.v1"
    require_live_discord: bool = False
    require_openai_realtime: bool = False
    require_gemini_live: bool = False
    reports: dict[str, str] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    live_probe_ok: bool | None = None
    live_probe_status: str = "not_run"
    evidence_context: dict[str, Any] = field(default_factory=dict)
    validate_live_evidence: bool = False
    strict_validation: dict[str, Any] = field(default_factory=dict)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Hermes realtime voice live-evidence artifacts")
    parser.add_argument(
        "--output-dir",
        default="./artifacts/realtime-voice-evidence/live-current",
        help="Directory where evidence JSON files will be written",
    )
    parser.add_argument(
        "--require-live-discord",
        action="store_true",
        help="Fail if the Discord live voice probe does not pass",
    )
    parser.add_argument(
        "--require-openai-realtime",
        action="store_true",
        help="Fail unless an OpenAI Realtime API key env var is present",
    )
    parser.add_argument(
        "--require-gemini-live",
        action="store_true",
        help="Fail unless a Gemini Live API key env var is present",
    )
    parser.add_argument("--guild-id", default=os.environ.get("DISCORD_GUILD_ID", ""))
    parser.add_argument("--text-channel-id", default=os.environ.get("DISCORD_HOME_CHANNEL", ""))
    parser.add_argument("--voice-channel-id", default=os.environ.get("DISCORD_VOICE_CHANNEL_ID", ""))
    parser.add_argument("--voice-channel-name", default=os.environ.get("DISCORD_VOICE_CHANNEL_NAME", "General"))
    parser.add_argument("--wait-seconds", type=float, default=2.0)
    parser.add_argument(
        "--require-inbound",
        action="store_true",
        help="Require inbound speech frames during the Discord live probe",
    )
    parser.add_argument(
        "--validate-live-evidence",
        action="store_true",
        help=(
            "Validate supplied Discord/sidecar/turn evidence with the strict VoiceOps ingester contract "
            "without connecting to Discord or requiring credentials"
        ),
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help=(
            "Validate supplied live evidence without running probes, deriving reports, or writing persistent "
            "artifacts under --output-dir"
        ),
    )
    parser.add_argument(
        "--discord-live-probe-evidence",
        type=Path,
        help="Optional read-only Discord live probe evidence JSON to reference from the manifest",
    )
    parser.add_argument(
        "--sidecar-session-evidence",
        type=Path,
        help="Optional read-only sidecar session evidence JSON to reference from the manifest",
    )
    parser.add_argument(
        "--live-turn-evidence",
        type=Path,
        help="Optional read-only live turn evidence JSON to reference from the manifest",
    )
    parser.add_argument(
        "--from-realtime-voice-report",
        type=Path,
        help=(
            "Offline: derive sidecar/live-turn evidence files from a JSON report written by "
            "hermes doctor --realtime-voice-report"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "audit_only", False):
        result = audit_realtime_voice_live_evidence(args)
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        return 0 if result["ok"] else 1
    result = asyncio.run(collect_realtime_voice_live_evidence(args))
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0 if result.ok else 1


def audit_realtime_voice_live_evidence(args: argparse.Namespace) -> dict[str, Any]:
    reports = {
        report_key: evidence_path
        for report_key, evidence_path in (
            ("discord_live_probe", getattr(args, "discord_live_probe_evidence", None)),
            ("sidecar_session", getattr(args, "sidecar_session_evidence", None)),
            ("live_turn", getattr(args, "live_turn_evidence", None)),
        )
        if evidence_path is not None
    }
    issues: list[str] = []
    if getattr(args, "from_realtime_voice_report", None) is not None:
        issues.append("audit_only: --from-realtime-voice-report is not supported because derivation writes files")
    if not reports:
        issues.append("audit_only: at least one live evidence file is required")

    strict_validation: dict[str, Any]
    if reports:
        with tempfile.TemporaryDirectory(prefix="hermes-live-evidence-audit-") as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            _write_json(
                manifest_path,
                {
                    "schema_version": "voiceops.realtime_voice_live_evidence_manifest.v1",
                    "reports": {
                        report_key: str(evidence_path.expanduser().resolve(strict=False))
                        for report_key, evidence_path in reports.items()
                    },
                },
            )
            strict_validation = _strict_live_evidence_validation(manifest_path)
    else:
        strict_validation = {
            "schema_version": "voiceops.realtime_voice_live_evidence_validation.v1",
            "manifest": None,
            "loaded": False,
            "overall_status": "partial_live_evidence",
            "issues": ["live_evidence_not_loaded"],
            "section_refs": {},
            "missing_gates": ["discord_join", "discord_playback", "live_receiver", "live_turn", "production_sidecar"],
        }

    validation_issues = [f"live_evidence_validation:{issue}" for issue in strict_validation.get("issues", [])]
    all_issues = sorted(set([*issues, *validation_issues]))
    ok = (
        not all_issues
        and strict_validation.get("overall_status") == "live_evidence_supplied_not_readiness_claim"
        and not strict_validation.get("missing_gates")
    )
    return {
        "schema_version": "voiceops.realtime_voice_live_evidence_audit.v1",
        "ok": ok,
        "artifact_writes": False,
        "discord_probe_run": False,
        "report_derivation_run": False,
        "output_dir": str(Path(args.output_dir).expanduser()),
        "reports": {report_key: str(path) for report_key, path in reports.items()},
        "issues": all_issues,
        "strict_validation": strict_validation,
    }


async def collect_realtime_voice_live_evidence(args: argparse.Namespace) -> RealtimeVoiceLiveEvidenceResult:
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, str] = {}
    issues: list[str] = []
    warnings: list[str] = []
    live_probe_ok: bool | None = None
    live_probe_status = "not_run"
    context = _evidence_context(args)
    derived_report_paths: dict[str, Path] = {}
    derived_evidence_supplied = False

    if getattr(args, "from_realtime_voice_report", None) is not None:
        derived_report_paths, derived_issues = _derive_live_evidence_from_realtime_voice_report(
            Path(args.from_realtime_voice_report),
            output_dir=output_dir,
        )
        issues.extend(derived_issues)
        if "discord_live_probe" in derived_report_paths:
            _attach_optional_evidence_report(
                reports=reports,
                issues=issues,
                report_key="discord_live_probe",
                path=derived_report_paths["discord_live_probe"],
            )
        derived_evidence_supplied = any(
            key in derived_report_paths for key in ("discord_live_probe", "sidecar_session", "live_turn")
        )
    elif getattr(args, "validate_live_evidence", False):
        if getattr(args, "discord_live_probe_evidence", None) is None:
            issues.append("discord_live_probe: evidence file is required for --validate-live-evidence")
        _attach_optional_evidence_report(
            reports=reports,
            issues=issues,
            report_key="discord_live_probe",
            path=getattr(args, "discord_live_probe_evidence", None),
        )
    else:
        loopback_report = output_dir / "discord-loopback.json"
        loopback_result = await _run_discord_loopback_smoke()
        _write_json(
            loopback_report,
            _with_report_identity(asdict(loopback_result), kind="discord_loopback", report_path=loopback_report),
        )
        reports["discord_loopback"] = _report_ref(output_dir, loopback_report)
        if not getattr(loopback_result, "ok", False):
            issues.append(f"discord_loopback: {getattr(loopback_result, 'error', '') or 'failed'}")

        live_report = output_dir / "discord-live-probe.json"
        live_result = await _run_discord_live_probe(args)
        live_probe_ok = bool(getattr(live_result, "ok", False))
        live_probe_status = "passed" if live_probe_ok else "failed"
        _write_json(
            live_report,
            _with_report_identity(asdict(live_result), kind="discord_live_probe", report_path=live_report),
        )
        reports["discord_live_probe"] = _report_ref(output_dir, live_report)
        if not live_probe_ok:
            message = f"discord_live_probe: {getattr(live_result, 'error', '') or 'failed'}"
            if args.require_live_discord:
                issues.append(message)
            else:
                warnings.append(message)

    if derived_report_paths:
        reports["realtime_voice_report_validation"] = _report_ref(
            output_dir,
            output_dir / "realtime-voice-report-validation.json",
        )
    _attach_optional_evidence_report(
        reports=reports,
        issues=issues,
        report_key="sidecar_session",
        path=derived_report_paths.get("sidecar_session") or getattr(args, "sidecar_session_evidence", None),
    )
    _attach_optional_evidence_report(
        reports=reports,
        issues=issues,
        report_key="live_turn",
        path=derived_report_paths.get("live_turn") or getattr(args, "live_turn_evidence", None),
    )

    if args.require_openai_realtime and not _openai_realtime_key_present():
        issues.append("openai_realtime: OPENAI_API_KEY or HERMES_OPENAI_REALTIME_API_KEY is required")
    if args.require_gemini_live and not _gemini_live_key_present():
        issues.append("gemini_live: GEMINI_API_KEY or HERMES_GEMINI_LIVE_API_KEY is required")

    strict_validation: dict[str, Any] = {}
    result = RealtimeVoiceLiveEvidenceResult(
        ok=not issues,
        output_dir=str(output_dir),
        require_live_discord=bool(args.require_live_discord),
        require_openai_realtime=bool(args.require_openai_realtime),
        require_gemini_live=bool(args.require_gemini_live),
        reports=reports,
        issues=issues,
        warnings=warnings,
        live_probe_ok=live_probe_ok,
        live_probe_status=live_probe_status,
        evidence_context=context,
        validate_live_evidence=bool(getattr(args, "validate_live_evidence", False)),
        strict_validation=strict_validation,
    )
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, asdict(result))
    optional_evidence_supplied = any(
        getattr(args, attr, None) is not None
        for attr in ("discord_live_probe_evidence", "sidecar_session_evidence", "live_turn_evidence")
    ) or derived_evidence_supplied
    if getattr(args, "validate_live_evidence", False) or optional_evidence_supplied:
        strict_validation = _strict_live_evidence_validation(manifest_path)
        strict_issues = [f"live_evidence_validation:{issue}" for issue in strict_validation.get("issues", [])]
        if strict_issues:
            issues.extend(strict_issues)
        result = RealtimeVoiceLiveEvidenceResult(
            ok=not issues and strict_validation.get("overall_status") == "live_evidence_supplied_not_readiness_claim",
            output_dir=str(output_dir),
            require_live_discord=bool(args.require_live_discord),
            require_openai_realtime=bool(args.require_openai_realtime),
            require_gemini_live=bool(args.require_gemini_live),
            reports=reports,
            issues=sorted(set(issues)),
            warnings=warnings,
            live_probe_ok=live_probe_ok,
            live_probe_status=live_probe_status,
            evidence_context=context,
            validate_live_evidence=bool(getattr(args, "validate_live_evidence", False)),
            strict_validation=strict_validation,
        )
        _write_json(output_dir / "live-evidence-validation.json", strict_validation)
        _write_json(manifest_path, asdict(result))
    return result


def _derive_live_evidence_from_realtime_voice_report(
    report_path: Path,
    *,
    output_dir: Path,
) -> tuple[dict[str, Path], list[str]]:
    from agent.realtime_voice_smoke_report import (
        load_realtime_voice_smoke_report,
        validate_realtime_voice_alpha_report,
    )

    reports: dict[str, Path] = {}
    issues: list[str] = []
    resolved = report_path.expanduser().resolve(strict=False)
    try:
        entries = load_realtime_voice_smoke_report(resolved)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, [f"realtime_voice_report: failed to load report: {exc}"]

    validation_issues = validate_realtime_voice_alpha_report(entries)
    validation_payload = {
        "kind": "realtime_voice_report_validation",
        "source_artifact": str(resolved),
        "schema_version": "voiceops.realtime_voice_report_derivation.v1",
        "alpha_valid": not validation_issues,
        "issue_count": len(validation_issues),
        "issues": [_format_report_issue(issue) for issue in validation_issues],
        "derived_sections": [],
    }

    if validation_issues:
        issues.extend(f"realtime_voice_report:{_format_report_issue(issue)}" for issue in validation_issues)

    discord = _derive_discord_probe_from_realtime_report(entries, resolved)
    if discord is not None:
        path = output_dir / "discord-live-probe.from-realtime-report.json"
        _write_json(path, discord)
        reports["discord_live_probe"] = path
        validation_payload["derived_sections"].append("discord_live_probe")

    sidecar = _derive_sidecar_session_from_realtime_report(
        entries,
        resolved,
        alpha_valid=not validation_issues,
    )
    if sidecar is not None:
        path = output_dir / "sidecar-session.from-realtime-report.json"
        _write_json(path, sidecar)
        reports["sidecar_session"] = path
        validation_payload["derived_sections"].append("sidecar_session")
    else:
        issues.append("realtime_voice_report: unable to derive sidecar_session evidence")

    live_turn = _derive_live_turn_from_realtime_report(
        entries,
        resolved,
        alpha_valid=not validation_issues,
    )
    if live_turn is not None:
        path = output_dir / "live-turn.from-realtime-report.json"
        _write_json(path, live_turn)
        reports["live_turn"] = path
        validation_payload["derived_sections"].append("live_turn")
    else:
        issues.append("realtime_voice_report: unable to derive live_turn evidence")

    _write_json(output_dir / "realtime-voice-report-validation.json", validation_payload)
    return reports, issues


def _format_report_issue(issue: object) -> str:
    formatter = getattr(issue, "format", None)
    if callable(formatter):
        return str(formatter())
    return str(issue)


def _derive_discord_probe_from_realtime_report(entries: list[dict[str, Any]], report_path: Path) -> dict[str, Any] | None:
    for entry in entries:
        if str(entry.get("kind") or "") != "discord_live_probe":
            continue
        payload = dict(entry)
        payload.setdefault("kind", "discord_live_probe")
        payload["source_artifact"] = str(report_path)
        return payload
    return None


def _derive_sidecar_session_from_realtime_report(
    entries: list[dict[str, Any]],
    report_path: Path,
    *,
    alpha_valid: bool,
) -> dict[str, Any] | None:
    manifest = _first_entry(entries, "manifest")
    protocol = _first_passing_entry(entries, "protocol") or _first_entry(entries, "protocol")
    bridge = _first_passing_entry(entries, "discord_bridge") or _first_entry(entries, "discord_bridge")
    if manifest is None and bridge is None:
        return None
    sidecar = manifest.get("sidecar") if isinstance(manifest, dict) and isinstance(manifest.get("sidecar"), dict) else {}
    health = sidecar.get("health") if isinstance(sidecar.get("health"), dict) else {}
    health_ok = health.get("ok") is True
    healthy = sidecar.get("healthy") is True or health_ok
    bridge_ok = isinstance(bridge, dict) and bridge.get("ok") is True
    protocol_ok = isinstance(protocol, dict) and protocol.get("ok") is True
    shutdown_ms = _first_number(
        bridge.get("shutdown_elapsed_ms") if isinstance(bridge, dict) else None,
        bridge.get("latency_metrics_ms", {}).get("shutdown_ms")
        if isinstance(bridge, dict) and isinstance(bridge.get("latency_metrics_ms"), dict)
        else None,
    )
    session_start_ms = _first_number(
        protocol.get("ready_ms") if isinstance(protocol, dict) else None,
        _entry_number(entries, "session_turn", "ready_ms"),
        _entry_number(entries, "audio_session", "ready_ms"),
        0,
    )
    sidecar_mode = str(sidecar.get("mode") or "").strip() or "none"
    provider_transport_observed = bool(
        alpha_valid
        and healthy
        and any(entry.get("ok") is True and _non_empty(entry.get("transport")) for entry in entries)
    )
    shutdown_bounded = isinstance(bridge, dict) and bridge.get("shutdown_bounded") is True
    shutdown_timed_out = bool(bridge.get("shutdown_timed_out")) if isinstance(bridge, dict) else True
    unavailable_reason = str(manifest.get("unavailable_reason") or "") if isinstance(manifest, dict) else ""
    conversation_quality = (
        manifest.get("conversation_quality")
        if isinstance(manifest, dict) and isinstance(manifest.get("conversation_quality"), dict)
        else {}
    )
    fallback_reason = (
        "none"
        if conversation_quality.get("live_like") is True and not unavailable_reason
        else "fallback_or_unavailable_redacted"
    )
    return {
        "kind": "sidecar_session",
        "source_artifact": str(report_path),
        "derived_from": "hermes doctor --realtime-voice-report",
        "sidecar_running": bool(alpha_valid and protocol_ok),
        "sidecar_healthy": bool(alpha_valid and healthy),
        "session_started": bool(alpha_valid and protocol_ok and _non_negative_number(session_start_ms) is not None),
        "session_closed": bool(alpha_valid and bridge_ok and bridge.get("sidecar_closed") is True)
        if isinstance(bridge, dict)
        else False,
        "fallback_mode_visible": True,
        "fallback_reason": fallback_reason,
        "sidecar_mode": sidecar_mode,
        "healthcheck_observed": bool(isinstance(sidecar.get("health"), dict) and sidecar.get("health")),
        "provider_transport_observed": provider_transport_observed,
        "session_id_redacted": True,
        "shutdown_bounded": bool(alpha_valid and shutdown_bounded),
        "shutdown_timed_out": bool(shutdown_timed_out),
        "latency_metrics_ms": {
            "session_start_ms": session_start_ms,
            "shutdown_ms": shutdown_ms,
        },
    }


def _derive_live_turn_from_realtime_report(
    entries: list[dict[str, Any]],
    report_path: Path,
    *,
    alpha_valid: bool,
) -> dict[str, Any] | None:
    turn_entries = [
        entry
        for entry in entries
        if str(entry.get("kind") or "") in {"session_turn", "audio_session"}
    ]
    barge_in = _first_passing_entry(entries, "barge_in") or _first_entry(entries, "barge_in")
    if not turn_entries and barge_in is None:
        return None
    first_audio_ms = _min_number(
        _nested_metric(entry, "kame_speech_end_to_first_audio_ms")
        for entry in turn_entries
        if entry.get("ok") is True
    )
    if first_audio_ms is None:
        first_audio_ms = _min_number(
            entry.get("first_audio_ms")
            for entry in turn_entries
        )
    barge_in_ms = _first_number(
        _nested_metric(barge_in, "barge_in_confirmed_to_playback_stopped_ms")
        if isinstance(barge_in, dict)
        else None,
        barge_in.get("barge_in_ack_ms")
        if isinstance(barge_in, dict) and _positive_int(barge_in.get("audio_after_barge_in_bytes")) == 0
        else None,
        _entry_number(entries, "discord_bridge", "barge_in_ack_ms"),
    )
    transcript_observed = any(
        entry.get("ok") is True
        and (
            _non_negative_number(entry.get("transcript_final_ms")) is not None
            or _non_empty(entry.get("final_text"))
            or _non_empty(entry.get("text"))
        )
        for entry in turn_entries
    )
    assistant_audio_observed = any(
        entry.get("ok") is True
        and (
            _positive_int(entry.get("output_audio_bytes")) > 0
            or "audio.output.chunk" in _event_set(entry)
            or "assistant.audio.chunk" in _event_set(entry)
        )
        for entry in turn_entries
    )
    texts = [
        str(entry.get("assistant_final_text") or entry.get("final_text") or entry.get("text") or "")
        for entry in turn_entries
    ]
    denial_observed = any(_looks_like_voice_denial(text) for text in texts if text)
    spoken_reply_short = any(0 < len(text) <= 240 for text in texts)
    return {
        "kind": "live_turn",
        "source_artifact": str(report_path),
        "derived_from": "hermes doctor --realtime-voice-report",
        "transcript_observed": bool(alpha_valid and transcript_observed),
        "assistant_audio_observed": bool(alpha_valid and assistant_audio_observed),
        "barge_in_observed": bool(alpha_valid and isinstance(barge_in, dict) and barge_in.get("ok") is True),
        "spoken_reply_short": bool(alpha_valid and spoken_reply_short),
        "no_voice_denial_observed": bool(alpha_valid and not denial_observed),
        "speech_end_to_first_audio_ms": first_audio_ms,
        "barge_in_stop_ms": barge_in_ms,
    }


def _first_entry(entries: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    for entry in entries:
        if str(entry.get("kind") or "") == kind:
            return entry
    return None


def _first_passing_entry(entries: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    for entry in entries:
        if str(entry.get("kind") or "") == kind and entry.get("ok") is True:
            return entry
    return None


def _entry_number(entries: list[dict[str, Any]], kind: str, key: str) -> float | None:
    return _min_number(entry.get(key) for entry in entries if str(entry.get("kind") or "") == kind)


def _nested_metric(entry: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(entry, dict):
        return None
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    first_audio_metrics = (
        entry.get("first_audio_metrics")
        if isinstance(entry.get("first_audio_metrics"), dict)
        else {}
    )
    return metrics.get(key) if key in metrics else first_audio_metrics.get(key)


def _min_number(values: Any) -> float | None:
    numbers = [_non_negative_number(value) for value in values]
    numbers = [number for number in numbers if number is not None]
    return min(numbers) if numbers else None


def _first_number(*values: Any) -> float | None:
    for value in values:
        number = _non_negative_number(value)
        if number is not None:
            return number
    return None


def _event_set(entry: dict[str, Any]) -> set[str]:
    events = entry.get("events") if isinstance(entry.get("events"), list) else []
    return {str(event) for event in events}


def _events_include(entries: list[dict[str, Any]], event_name: str) -> bool:
    return any(event_name in _event_set(entry) for entry in entries)


def _non_empty(value: Any) -> bool:
    return bool(str(value or "").strip())


def _positive_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


def _looks_like_voice_denial(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "cannot hear you",
            "can't hear you",
            "cannot listen",
            "can't listen",
            "cannot join voice",
            "can't join voice",
            "cannot speak in voice",
            "can't speak in voice",
        )
    )


def _with_report_identity(payload: dict[str, Any], *, kind: str, report_path: Path) -> dict[str, Any]:
    enriched = dict(payload)
    enriched.setdefault("kind", kind)
    enriched.setdefault("source_artifact", str(report_path))
    return enriched


def _report_ref(output_dir: Path, report_path: Path) -> str:
    try:
        return str(report_path.relative_to(output_dir))
    except ValueError:
        return str(report_path.resolve())


def _attach_optional_evidence_report(
    *,
    reports: dict[str, str],
    issues: list[str],
    report_key: str,
    path: Path | None,
) -> None:
    if path is None:
        return
    resolved = path.expanduser()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError:
        issues.append(f"{report_key}: evidence file not found")
        issues.append(f"{report_key}: evidence file not found at {resolved.resolve(strict=False)}")
        return
    except json.JSONDecodeError as exc:
        issues.append(f"{report_key}: evidence JSON parse failed: {exc.msg}")
        return
    if not isinstance(payload, dict):
        issues.append(f"{report_key}: evidence root must be an object")
        return
    if not _optional_evidence_has_identity(report_key, payload):
        issues.append(f"{report_key}: evidence file must include kind, evidence_type, or live evidence schema")
        return
    structural_issues = _optional_evidence_structural_issues(report_key, payload)
    if structural_issues:
        issues.extend(f"{report_key}: {issue}" for issue in structural_issues)
        return
    reports[report_key] = str(resolved.resolve())


def _optional_evidence_has_identity(report_key: str, payload: dict[str, Any]) -> bool:
    if (
        payload.get("schema_version") == "voiceops.milestone1.live_voice_evidence.v1"
        and any(isinstance(payload.get(section), dict) for section in ("discord_live_probe", "sidecar_session", "live_turn"))
    ):
        return True
    kind = str(payload.get("kind") or payload.get("evidence_type") or "").strip()
    return kind == report_key


def _optional_evidence_structural_issues(report_key: str, payload: dict[str, Any]) -> list[str]:
    if payload.get("example_only") is True:
        return ["example_only evidence is not accepted"]
    if report_key == "discord_live_probe":
        issues = _missing_required_optional_fields(
            payload,
            (
                "ok",
                "connect_perm",
                "speak_perm",
                "connected",
                "opus_loaded",
                "accepted_audio_source",
                "played",
                "playing_during_probe",
                "receiver_started",
                "inbound_observed",
                "disconnected",
                "require_inbound",
            ),
            nested_numbers=(
                "latency_metrics_ms.connect_ms",
                "latency_metrics_ms.playback_observed_ms",
                "latency_metrics_ms.inbound_observed_ms",
                "latency_metrics_ms.disconnect_ms",
            ),
        )
        receiver_frames = _non_negative_number(payload.get("receiver_frames"))
        receiver_speech_start = _non_negative_number(payload.get("receiver_speech_start"))
        if (receiver_frames is None or receiver_frames <= 0) and (
            receiver_speech_start is None or receiver_speech_start <= 0
        ):
            issues.append("receiver_frames or receiver_speech_start must be positive")
        return issues
    if report_key == "sidecar_session":
        return _missing_required_optional_fields(
            payload,
            (
                "sidecar_running",
                "sidecar_healthy",
                "session_started",
                "session_closed",
                "fallback_mode_visible",
                "shutdown_bounded",
                "healthcheck_observed",
                "provider_transport_observed",
                "session_id_redacted",
            ),
            exact_fields={"sidecar_mode": "production"},
            required_strings=("fallback_reason",),
            nested_numbers=("latency_metrics_ms.session_start_ms", "latency_metrics_ms.shutdown_ms"),
            false_fields=("shutdown_timed_out",),
        )
    if report_key == "live_turn":
        return _missing_required_optional_fields(
            payload,
            (
                "transcript_observed",
                "assistant_audio_observed",
                "barge_in_observed",
                "spoken_reply_short",
                "no_voice_denial_observed",
            ),
            nested_numbers=("speech_end_to_first_audio_ms", "barge_in_stop_ms"),
        )
    return []


def _strict_live_evidence_validation(manifest_path: Path) -> dict[str, Any]:
    from scripts.voiceops_voice_operator import _load_live_evidence

    evidence = _load_live_evidence([manifest_path])
    return {
        "schema_version": "voiceops.realtime_voice_live_evidence_validation.v1",
        "manifest": str(manifest_path),
        "loaded": evidence.get("loaded") is True,
        "overall_status": str(evidence.get("overall_status") or "partial_live_evidence"),
        "issues": list(evidence.get("issues") or []),
        "section_refs": evidence.get("section_refs") or {},
        "missing_gates": _strict_live_evidence_missing_gates(evidence),
    }


def _strict_live_evidence_missing_gates(evidence: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    discord = evidence.get("discord_live_probe") if isinstance(evidence.get("discord_live_probe"), dict) else {}
    if discord.get("join_ok") is not True:
        missing.append("discord_join")
    if discord.get("playback_ok") is not True:
        missing.append("discord_playback")
    if discord.get("inbound_observed") is not True:
        missing.append("live_receiver")
    sidecar = evidence.get("sidecar_session") if isinstance(evidence.get("sidecar_session"), dict) else {}
    if sidecar.get("ok") is not True:
        missing.append("production_sidecar")
    live_turn = evidence.get("live_turn") if isinstance(evidence.get("live_turn"), dict) else {}
    if live_turn.get("ok") is not True:
        missing.append("live_turn")
    return sorted(set(missing))


def _missing_required_optional_fields(
    payload: dict[str, Any],
    true_fields: tuple[str, ...],
    *,
    exact_fields: dict[str, str] | None = None,
    required_strings: tuple[str, ...] = (),
    nested_numbers: tuple[str, ...] = (),
    false_fields: tuple[str, ...] = (),
) -> list[str]:
    issues: list[str] = []
    for field in true_fields:
        if payload.get(field) is not True:
            issues.append(f"{field} must be true")
    for field in false_fields:
        if payload.get(field) is not False:
            issues.append(f"{field} must be false")
    for field, expected in (exact_fields or {}).items():
        if str(payload.get(field) or "").strip() != expected:
            issues.append(f"{field} must be {expected}")
    for field in required_strings:
        if not str(payload.get(field) or "").strip():
            issues.append(f"{field} must be a non-empty redacted string")
    for field in nested_numbers:
        value: Any = payload
        for part in field.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if _non_negative_number(value) is None:
            issues.append(f"{field} must be a non-negative number")
    return issues


def _non_negative_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


async def _run_discord_loopback_smoke() -> Any:
    from hermes_cli.discord_realtime_voice_smoke import run_discord_realtime_voice_smoke

    return await run_discord_realtime_voice_smoke()


async def _run_discord_live_probe(args: argparse.Namespace) -> Any:
    from hermes_cli.discord_voice_live_probe import run_discord_voice_live_probe

    probe_args = argparse.Namespace(
        guild_id=str(getattr(args, "guild_id", "") or ""),
        text_channel_id=str(getattr(args, "text_channel_id", "") or ""),
        voice_channel_id=str(getattr(args, "voice_channel_id", "") or ""),
        voice_channel_name=str(getattr(args, "voice_channel_name", "") or "General"),
        wait_seconds=float(getattr(args, "wait_seconds", 2.0) or 0.0),
        require_inbound=bool(getattr(args, "require_inbound", False)),
        report="",
    )
    return await run_discord_voice_live_probe(probe_args)


def _evidence_context(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "config_snapshot": {
            "guild_id_configured": _configured(getattr(args, "guild_id", "")),
            "text_channel_id_configured": _configured(getattr(args, "text_channel_id", "")),
            "voice_channel_id_configured": _configured(getattr(args, "voice_channel_id", "")),
            "voice_channel_name": str(getattr(args, "voice_channel_name", "") or ""),
            "wait_seconds": max(0.0, float(getattr(args, "wait_seconds", 0.0) or 0.0)),
            "require_inbound": bool(getattr(args, "require_inbound", False)),
        },
        "env_presence": {
            "DISCORD_BOT_TOKEN": bool(os.environ.get("DISCORD_BOT_TOKEN")),
            "DISCORD_GUILD_ID": bool(os.environ.get("DISCORD_GUILD_ID")),
            "DISCORD_HOME_CHANNEL": bool(os.environ.get("DISCORD_HOME_CHANNEL")),
            "DISCORD_VOICE_CHANNEL_ID": bool(os.environ.get("DISCORD_VOICE_CHANNEL_ID")),
            "DISCORD_VOICE_CHANNEL_NAME": bool(os.environ.get("DISCORD_VOICE_CHANNEL_NAME")),
            "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
            "HERMES_OPENAI_REALTIME_API_KEY": bool(os.environ.get("HERMES_OPENAI_REALTIME_API_KEY")),
            "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY")),
            "HERMES_GEMINI_LIVE_API_KEY": bool(os.environ.get("HERMES_GEMINI_LIVE_API_KEY")),
        },
    }


def _openai_realtime_key_present() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("HERMES_OPENAI_REALTIME_API_KEY"))


def _gemini_live_key_present() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("HERMES_GEMINI_LIVE_API_KEY"))


def _configured(value: Any) -> bool:
    return bool(str(value or "").strip())


def _git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ("git", *args),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
