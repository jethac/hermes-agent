import json
from pathlib import Path

from agent.realtime_voice_smoke_report import (
    ALPHA_REQUIRED_AUDIO_FIXTURE_TEXTS,
    ALPHA_REQUIRED_AUDIO_FIXTURES,
)
from hermes_cli import realtime_voice_fixture_pack


def test_fixture_pack_generates_required_en_ja_files_and_manifest(tmp_path):
    calls = []

    def fake_synthesize(*, text, output_path):
        calls.append((text, output_path))
        Path(output_path).write_bytes(f"audio:{text}".encode("utf-8"))
        return json.dumps(
            {
                "success": True,
                "file_path": output_path,
                "provider": "fake-tts",
                "voice_compatible": False,
            },
            ensure_ascii=False,
        )

    def fake_convert(source, target):
        target.write_bytes(b"webm:" + source.read_bytes())

    manifest = realtime_voice_fixture_pack.build_alpha_fixture_pack(
        output_dir=tmp_path / "fixtures" / "realtime-voice",
        overwrite=False,
        synthesize_func=fake_synthesize,
        convert_func=fake_convert,
    )

    assert [entry["fixture"] for entry in manifest["fixtures"]] == list(ALPHA_REQUIRED_AUDIO_FIXTURES)
    assert {call[0] for call in calls} == set(ALPHA_REQUIRED_AUDIO_FIXTURE_TEXTS.values())
    assert (tmp_path / "fixtures" / "realtime-voice" / "en" / "hello.webm").read_bytes().startswith(b"webm:")
    assert (tmp_path / "fixtures" / "realtime-voice" / "ja" / "tool-question.webm").is_file()
    manifest_path = tmp_path / "fixtures" / "realtime-voice" / "manifest.json"
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["kind"] == "realtime_voice_alpha_fixture_pack"
    assert saved["codec"] == "webm_opus"
    assert saved["fixtures"][0]["source_provider"] == "fake-tts"
    assert saved["fixtures"][0]["locale"] == "en-US"
    assert saved["fixtures"][-1]["locale"] == "ja-JP"


def test_fixture_pack_refuses_existing_files_without_overwrite(tmp_path):
    output_dir = tmp_path / "fixtures" / "realtime-voice"
    existing = output_dir / "en" / "hello.webm"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"existing")

    try:
        realtime_voice_fixture_pack.build_alpha_fixture_pack(
            output_dir=output_dir,
            synthesize_func=lambda **_: "{}",
            convert_func=lambda _source, _target: None,
        )
    except FileExistsError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("expected fixture pack generation to refuse existing files")


def test_fixture_pack_main_reports_tts_failure(monkeypatch, tmp_path, capsys):
    def failing_build(**_kwargs):
        raise RuntimeError("failed Bearer secret-token at http://user:pass@voice.local/v1?token=abc")

    monkeypatch.setattr(realtime_voice_fixture_pack, "build_alpha_fixture_pack", failing_build)

    result = realtime_voice_fixture_pack.main(
        [
            "--output-dir",
            str(tmp_path / "fixtures"),
        ]
    )

    assert result == 1
    error = capsys.readouterr().err
    assert "Realtime voice fixture pack failed" in error
    assert "secret-token" not in error
    assert "token=abc" not in error
