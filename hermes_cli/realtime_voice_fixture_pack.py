"""Generate the required realtime voice alpha audio fixture pack."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from agent.realtime_voice_errors import sanitize_realtime_voice_error
from agent.realtime_voice_smoke_report import (
    ALPHA_REQUIRED_AUDIO_FIXTURE_TEXTS,
    ALPHA_REQUIRED_AUDIO_FIXTURES,
)


SynthesizeFn = Callable[..., Any]
ConvertFn = Callable[[Path, Path], None]


@dataclass(frozen=True)
class AlphaFixtureSpec:
    fixture: str
    text: str
    language: str
    locale: str
    script: str
    target_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Hermes realtime voice private-alpha audio fixtures"
    )
    parser.add_argument(
        "--output-dir",
        default="./fixtures/realtime-voice",
        help="Directory where en/*.webm and ja/*.webm fixtures will be written",
    )
    parser.add_argument(
        "--manifest",
        default="manifest.json",
        help="Manifest filename relative to --output-dir, or an absolute path",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing fixture files and manifest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fixture plan without generating audio",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser()
    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = output_dir / manifest_path

    if args.dry_run:
        for spec in alpha_fixture_specs(output_dir):
            print(f"{spec.fixture} -> {spec.target_path}: {spec.text}")
        return 0

    try:
        manifest = build_alpha_fixture_pack(
            output_dir=output_dir,
            manifest_path=manifest_path,
            overwrite=bool(args.overwrite),
        )
    except Exception as exc:
        print(
            f"Realtime voice fixture pack failed: {sanitize_realtime_voice_error(exc)}",
            file=sys.stderr,
        )
        return 1

    print(
        "Realtime voice fixture pack OK: "
        f"{len(manifest.get('fixtures') or [])} fixture(s) written"
    )
    print(f"Manifest: {manifest_path}")
    return 0


def alpha_fixture_specs(output_dir: str | Path = "./fixtures/realtime-voice") -> list[AlphaFixtureSpec]:
    root = Path(output_dir).expanduser()
    return [
        AlphaFixtureSpec(
            fixture=fixture,
            text=ALPHA_REQUIRED_AUDIO_FIXTURE_TEXTS[fixture],
            language=_fixture_language(fixture),
            locale=_fixture_locale(fixture),
            script=_fixture_script(fixture),
            target_path=_fixture_target_path(fixture, root),
        )
        for fixture in ALPHA_REQUIRED_AUDIO_FIXTURES
    ]


def build_alpha_fixture_pack(
    *,
    output_dir: str | Path = "./fixtures/realtime-voice",
    manifest_path: str | Path | None = None,
    overwrite: bool = False,
    synthesize_func: SynthesizeFn | None = None,
    convert_func: ConvertFn | None = None,
) -> dict[str, Any]:
    output_root = Path(output_dir).expanduser()
    manifest_file = Path(manifest_path).expanduser() if manifest_path else output_root / "manifest.json"
    specs = alpha_fixture_specs(output_root)

    existing = [spec.target_path for spec in specs if spec.target_path.exists()]
    if manifest_file.exists():
        existing.append(manifest_file)
    if existing and not overwrite:
        raise FileExistsError(
            f"{existing[0]} already exists; pass --overwrite or choose another --output-dir"
        )

    if synthesize_func is None:
        from tools.tts_tool import text_to_speech_tool as synthesize_func
    if convert_func is None:
        convert_func = convert_audio_to_webm_opus

    fixture_entries: list[dict[str, Any]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hermes-realtime-fixtures-") as tmp:
        tmp_dir = Path(tmp)
        for spec in specs:
            source_path, tts_result = _synthesize_source_audio(spec, tmp_dir, synthesize_func)
            spec.target_path.parent.mkdir(parents=True, exist_ok=True)
            convert_func(source_path, spec.target_path)
            if not spec.target_path.is_file() or spec.target_path.stat().st_size <= 0:
                raise RuntimeError(f"fixture conversion produced no output: {spec.target_path}")
            fixture_entries.append(
                {
                    "fixture": spec.fixture,
                    "path": str(spec.target_path),
                    "text": spec.text,
                    "language": spec.language,
                    "locale": spec.locale,
                    "script": spec.script,
                    "codec": "webm_opus",
                    "audio_bytes": spec.target_path.stat().st_size,
                    "source_provider": str(tts_result.get("provider") or ""),
                    "source_voice_compatible": bool(tts_result.get("voice_compatible")),
                }
            )

    manifest = {
        "kind": "realtime_voice_alpha_fixture_pack",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_root),
        "codec": "webm_opus",
        "fixtures": fixture_entries,
    }
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def convert_audio_to_webm_opus(source_path: Path, target_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to generate webm/opus alpha fixtures")
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-c:a",
            "libopus",
            "-ac",
            "1",
            "-ar",
            "48000",
            str(target_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"ffmpeg fixture conversion failed: {detail or result.returncode}")


def _synthesize_source_audio(
    spec: AlphaFixtureSpec,
    tmp_dir: Path,
    synthesize_func: SynthesizeFn,
) -> tuple[Path, Mapping[str, Any]]:
    source_path = tmp_dir / f"{spec.language}-{spec.target_path.stem}.mp3"
    raw_result = synthesize_func(text=spec.text, output_path=str(source_path))
    result = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
    if not isinstance(result, Mapping):
        raise RuntimeError(f"TTS returned an invalid result for {spec.fixture}")
    if result.get("success") is not True:
        raise RuntimeError(f"TTS failed for {spec.fixture}: {result.get('error') or 'unknown error'}")
    file_path = Path(str(result.get("file_path") or source_path)).expanduser()
    if not file_path.is_file() or file_path.stat().st_size <= 0:
        raise RuntimeError(f"TTS produced no source audio for {spec.fixture}: {file_path}")
    return file_path, result


def _fixture_target_path(fixture: str, output_dir: Path) -> Path:
    normalized = fixture[2:] if fixture.startswith("./") else fixture
    path = PurePosixPath(normalized)
    try:
        relative = path.relative_to(PurePosixPath("fixtures/realtime-voice"))
    except ValueError:
        relative = PurePosixPath(path.name)
    return output_dir.joinpath(*relative.parts)


def _fixture_language(fixture: str) -> str:
    return "ja" if "/ja/" in fixture else "en"


def _fixture_locale(fixture: str) -> str:
    return "ja-JP" if _fixture_language(fixture) == "ja" else "en-US"


def _fixture_script(fixture: str) -> str:
    return "Jpan" if _fixture_language(fixture) == "ja" else "Latn"


if __name__ == "__main__":
    raise SystemExit(main())
