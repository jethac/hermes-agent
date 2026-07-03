---
title: "goal: Fix and harden Discord realtime voice after June 9 test"
status: active
date: 2026-06-25
type: goal
target_repo: hermes-agent
origin: June 9 Discord voice-channel live test
successor_design: docs/design/full-kame-style-realtime-voice.md
---

# goal: Fix and harden Discord realtime voice after June 9 test

## Summary

Fix the Discord live voice experience exposed by the June 9, 2026 test in the
`jetha dev server / #General` voice channel. The logs show Hermes joined the
voice channel, captured and transcribed live user speech, generated ElevenLabs
TTS, and played audio back into the channel. They also show major product and
architecture failures: the mixer failed to install, the conversation fell back
to the legacy Whisper plus normal-agent plus file-TTS path, responses were slow,
barge-in did not stop ongoing speech promptly, and Hermes told the user it could
not hear or speak in Discord voice while doing exactly that.

The goal is not just to make audio technically work. The goal is for `/voice
join` to create a coherent live voice session with explicit state, low-latency
feedback, correct model-facing context, reliable barge-in, observable fallback,
and tests that prevent the June 9 failure shape from returning.

This document records the June 9 hardening goal and its evidence trail. The
current KAME architecture target is tracked in
`docs/design/full-kame-style-realtime-voice.md`; that successor design preserves
the June 9 reliability requirements while adding the reflex/oracle split.

---

## June 9 Evidence

Primary log files:

- `/Users/jethac/.hermes/logs/gateway.log`
- `/Users/jethac/.hermes/logs/agent.log.1`
- `/Users/jethac/.hermes/logs/errors.log`

Key observations:

- `/voice join` was invoked at `2026-06-09 09:56:01`.
- `VoiceReceiver started` and Discord speaking events mapped the user SSRC.
- The mixer failed immediately:
  `Voice mixer failed to start: source must be an AudioSource not VoiceMixer`.
- Discord realtime voice session then logged as started, but user utterances
  were still processed through legacy local Whisper files such as
  `vc_listen_8g6vomxq.wav`.
- Response readiness was slow:
  - first substantive response: `19.3s`
  - second substantive response: `18.4s`
  - later short response: `7.8s`
- ElevenLabs TTS added more latency:
  - about `6.0s` for `tts_20260609_095643.mp3`
  - about `7.8s` for `tts_20260609_095752.mp3`
- The user's "barge-in" complaint was transcribed as "bargent", and Hermes kept
  playing/sending the prior long response before processing the interruption.
- Hermes played TTS in the voice channel while semantically claiming it lacked
  live Discord voice capability.

The important conclusion: the June 9 system had enough audio plumbing to be
dangerous, but not enough state discipline for the model, playback path, sidecar,
fallback, or interruption behavior to agree with each other.

## Execution Status

Updated 2026-06-25:

- Implemented explicit per-guild Discord voice session state and surfaced it
  through join/status/context paths.
- Hardened mixer startup fallback so realtime does not silently proceed without
  a mixer.
- Hardened sidecar startup, runtime degradation, and shutdown timeout behavior.
- Hardened model-facing live voice context and realtime oracle prompt so Discord
  voice sessions are not described as impossible.
- Hardened barge-in playback handling by stopping mixer/one-shot playback and
  dropping stale sidecar audio chunks when playback generations are available.
- Added central config defaults/schema for `voice.realtime.sidecar_close_timeout_seconds`
  and `discord.realtime_voice.*`.
- Added focused regression coverage for state transitions, degraded fallback,
  sidecar errors, close timeout, prompt context, and stale post-barge-in audio.
- Added Discord realtime runtime latency/quality-target state capture and
  surfaced it through `/voice status`.
- Added transport labeling to realtime voice smoke report payloads so future
  Discord-like smoke evidence can be distinguished from generic doctor smoke.
- Added first-audio diagnostic metric capture to smoke reports, including
  file-TTS synthesis duration (`tts_synthesis_ms`), so slow spoken-response
  paths can be split between oracle latency, TTS generation, and playback.
- Added configurable turn acknowledgements (`voice.realtime.turn_acknowledgement`)
  and prewarmed reference-sidecar acknowledgement audio so first spoken feedback
  is no longer gated by full response generation or a cold local TTS call.
- Added a deterministic local loopback streaming STT/TTS bridge for live-like
  protocol validation without external provider credentials. The alpha evidence
  runner can start it with `--provider loopback --start-bridge`, and reports are
  explicitly marked as loopback validation so they are not confused with
  production provider evidence.
- Added a local Discord realtime voice smoke command that exercises the Discord
  bridge path without Discord credentials: 48 kHz stereo PCM input is
  downsampled to 16 kHz sidecar audio, sidecar output is upsampled back into the
  mixer, and barge-in stops mixer speech and sends a realtime `BARGE_IN`.
- Added `hermes doctor --discord-realtime-voice-smoke` so the local Discord
  bridge smoke can run through the standard doctor/report path and produce
  machine-readable evidence beside realtime voice smoke reports.
- Added production-evidence guardrails so loopback-marked reports remain valid
  local protocol evidence but cannot satisfy the production readiness gate or
  be applied with `realtime_voice_alpha_evidence --provider loopback --apply`.
- Added an ElevenLabs realtime voice profile preset so the real-provider path
  can be configured with `realtime_voice_profile --preset elevenlabs` rather
  than hand-written generic bridge URLs.
- Added `hermes_cli.discord_voice_live_probe`, a bounded real Discord voice
  channel probe that joins a configured voice channel, installs
  `VoiceReceiver`, plays silent PCM through `VoiceMixer`, disconnects cleanly,
  and can optionally fail unless inbound speech frames are observed.
  Its reports now include `inbound_observed`, `members_after`, and
  `failure_reason` so empty-channel failures are distinguishable from receiver
  wiring failures.
- Verified:
  - `uv run --extra dev --extra voice python -m pytest tests/hermes_cli/test_discord_voice_live_probe.py tests/hermes_cli/test_doctor.py tests/hermes_cli/test_realtime_voice_profile.py tests/hermes_cli/test_realtime_voice_alpha_evidence.py tests/agent/test_realtime_voice_smoke_report.py tests/hermes_cli/test_web_server.py -q`
  - Result: `550 passed, 1 warning in 22.22s`.
  - `uv run --extra dev --extra voice python -m pytest tests/hermes_cli/test_discord_voice_live_probe.py -q`
  - Result: `4 passed, 1 warning in 0.21s`.
  - `uv run --extra dev --extra voice python -m pytest tests/agent/test_realtime_voice.py tests/agent/test_realtime_voice_smoke.py tests/agent/test_realtime_voice_loopback_bridge.py tests/agent/test_realtime_voice_smoke_report.py tests/hermes_cli/test_realtime_voice_alpha_evidence.py tests/hermes_cli/test_discord_realtime_voice_smoke.py tests/hermes_cli/test_discord_voice_live_probe.py tests/hermes_cli/test_doctor.py tests/hermes_cli/test_realtime_voice_profile.py tests/gateway/test_discord_realtime_voice.py tests/gateway/test_discord_voice_mixer.py tests/gateway/test_voice_command.py tests/agent/test_realtime_voice_oracle.py tests/hermes_cli/test_web_server.py tests/hermes_cli/test_config.py`
  - Result: `991 passed in 30.54s`.
  - `uv run --extra dev --extra voice python -m pytest tests/hermes_cli/test_discord_voice_live_probe.py -q`
  - Result: `2 passed, 1 warning in 0.21s`.
  - `uv run --extra dev --extra voice python -m pytest tests/agent/test_realtime_voice.py tests/agent/test_realtime_voice_smoke.py tests/agent/test_realtime_voice_loopback_bridge.py tests/agent/test_realtime_voice_smoke_report.py tests/hermes_cli/test_realtime_voice_alpha_evidence.py tests/hermes_cli/test_discord_realtime_voice_smoke.py tests/hermes_cli/test_doctor.py tests/hermes_cli/test_realtime_voice_profile.py tests/gateway/test_discord_realtime_voice.py tests/gateway/test_discord_voice_mixer.py tests/gateway/test_voice_command.py tests/agent/test_realtime_voice_oracle.py tests/hermes_cli/test_web_server.py tests/hermes_cli/test_config.py`
  - Result: `989 passed in 30.55s`.
  - `uv run --extra dev --extra voice python -m pytest tests/hermes_cli/test_doctor.py tests/hermes_cli/test_realtime_voice_profile.py tests/hermes_cli/test_realtime_voice_alpha_evidence.py tests/agent/test_realtime_voice_smoke_report.py tests/hermes_cli/test_web_server.py -q`
  - Result: `546 passed, 1 warning in 22.38s`.
  - `uv run --extra dev --extra voice python -m pytest tests/hermes_cli/test_realtime_voice_profile.py tests/hermes_cli/test_realtime_voice_alpha_evidence.py tests/agent/test_realtime_voice_smoke_report.py -q`
  - Result: `76 passed in 0.98s`.
  - `uv run --extra dev --extra voice python -m pytest tests/agent/test_realtime_voice_smoke_report.py tests/hermes_cli/test_realtime_voice_alpha_evidence.py tests/hermes_cli/test_web_server.py -q`
  - Result: `433 passed, 1 warning in 21.98s`.
  - `uv run --extra dev --extra voice python -m pytest tests/hermes_cli/test_doctor.py tests/hermes_cli/test_discord_realtime_voice_smoke.py`
  - Result: `100 passed in 6.28s`.
  - `uv run --extra dev --extra voice python -m pytest tests/agent/test_realtime_voice.py tests/agent/test_realtime_voice_smoke.py tests/agent/test_realtime_voice_loopback_bridge.py tests/hermes_cli/test_realtime_voice_alpha_evidence.py tests/hermes_cli/test_discord_realtime_voice_smoke.py tests/hermes_cli/test_doctor.py tests/gateway/test_discord_realtime_voice.py tests/gateway/test_discord_voice_mixer.py tests/gateway/test_voice_command.py tests/agent/test_realtime_voice_oracle.py tests/hermes_cli/test_web_server.py tests/hermes_cli/test_config.py`
  - Result: `912 passed, 21 skipped in 30.22s`.
  - `uv run --extra dev --extra voice python -m pytest tests/agent/test_realtime_voice_oracle.py tests/hermes_cli/test_doctor.py tests/hermes_cli/test_web_server.py tests/hermes_cli/test_config.py tests/gateway/test_discord_realtime_voice.py tests/gateway/test_discord_voice_mixer.py tests/gateway/test_voice_command.py`
  - Result: `775 passed, 21 skipped in 29.04s`.
  - `uv run --extra dev --extra voice python -m pytest tests/agent/test_realtime_voice.py tests/agent/test_realtime_voice_smoke.py tests/hermes_cli/test_doctor.py tests/gateway/test_discord_realtime_voice.py tests/gateway/test_discord_voice_mixer.py tests/gateway/test_voice_command.py tests/agent/test_realtime_voice_oracle.py tests/hermes_cli/test_web_server.py tests/hermes_cli/test_config.py`
  - Result: `881 passed, 21 skipped in 30.40s`.
  - `uv run --extra dev --extra voice python -m pytest tests/agent/test_realtime_voice.py tests/agent/test_realtime_voice_smoke.py tests/hermes_cli/test_doctor.py`
  - Result: `196 passed in 5.88s`.
  - `uv run --extra dev --extra voice python -m pytest tests/agent/test_realtime_voice.py tests/agent/test_realtime_voice_smoke.py tests/hermes_cli/test_doctor.py tests/gateway/test_discord_realtime_voice.py tests/gateway/test_discord_voice_mixer.py tests/gateway/test_voice_command.py tests/agent/test_realtime_voice_oracle.py tests/hermes_cli/test_web_server.py tests/hermes_cli/test_config.py`
  - Result: `882 passed, 21 skipped in 29.43s`.
  - `uv run --extra dev --extra voice python -m pytest tests/agent/test_realtime_voice.py tests/agent/test_realtime_voice_smoke.py tests/hermes_cli/test_doctor.py tests/gateway/test_discord_realtime_voice.py tests/gateway/test_voice_command.py tests/hermes_cli/test_web_server.py tests/hermes_cli/test_config.py`
  - Result: `863 passed, 21 skipped in 30.84s`.
  - `uv run --extra dev --extra voice python -m pytest tests/agent/test_realtime_voice.py tests/agent/test_realtime_voice_smoke.py tests/agent/test_realtime_voice_loopback_bridge.py tests/hermes_cli/test_realtime_voice_alpha_evidence.py tests/hermes_cli/test_doctor.py tests/gateway/test_discord_realtime_voice.py tests/gateway/test_discord_voice_mixer.py tests/gateway/test_voice_command.py tests/agent/test_realtime_voice_oracle.py tests/hermes_cli/test_web_server.py tests/hermes_cli/test_config.py`
  - Result: `907 passed, 21 skipped in 30.05s`.
  - `uv run --extra dev --extra voice python -m pytest tests/hermes_cli/test_discord_realtime_voice_smoke.py tests/gateway/test_discord_realtime_voice.py`
  - Result: `14 passed in 0.72s`.
  - `git diff --check` passed.
- Live protocol smoke configuration exists for `http://127.0.0.1:8765`.
  A bounded run failed when no sidecar was listening:
  `[Errno 61] Connect call failed ('127.0.0.1', 8765)`.
- After starting the local reference sidecar, direct protocol smoke passed:
  `ok=True`, `ready_ms=13`, `transcript_final_ms=13`,
  `transport=doctor_smoke`, events `frontend.state,transcript.final`.
- A report-producing doctor run with the sidecar available wrote
  `/tmp/hermes-realtime-voice-smoke-report.json` and showed the remaining
  latency gap: session turn first audio was `4465ms`, exceeding the `900ms`
  target, with events
  `session.started,transcript.final,assistant.text.partial,assistant.commit,frontend.state,audio.output.chunk`.
- After turn acknowledgement and sidecar prewarm work, a fresh report-producing
  doctor run with the local sidecar available passed the session-turn latency
  gate: `first_text=0ms <= 500ms`, `first_audio=669ms <= 900ms`. The report
  captured `first_audio_metrics` including `tts_synthesis_ms=668`,
  `tts_cache=prewarmed`, and `final_transcript_to_first_audio_ms=669`.
- With the loopback streaming bridge on `127.0.0.1:8768` and the reference
  sidecar pointed at it, doctor reported live-like streaming mode:
  `Live conversation quality (streaming_text: streaming_stt_tts, live-like: yes)`.
  The smoke report at `/tmp/hermes-realtime-voice-loopback-report.json` showed
  `ready_ms=15`, `transcript_final_ms=15`, `first_text=0ms`, and
  `first_audio=3ms`, with `first_audio_metrics` including
  `streaming_tts_ms=0`, `loopback=true`, and
  `final_transcript_to_first_audio_ms=3`.
- `uv run --extra dev --extra voice python -m hermes_cli.discord_realtime_voice_smoke --report /tmp/hermes-discord-realtime-voice-smoke.json`
  passed with `ok=true`, `transport=discord_voice`, `input_pcm48_bytes=3840`,
  `sidecar_pcm16_bytes=640`, `mixer_frames=1`, `mixer_frame_bytes=3840`,
  `barge_in_sent=true`, and events
  `transcript.partial,transcript.final,assistant.text.partial,audio.output.chunk,assistant.commit,barge_in`.
- `uv run --extra dev --extra voice hermes doctor --discord-realtime-voice-smoke --realtime-voice-report /tmp/hermes-discord-doctor-smoke.json`
  ran the Discord bridge smoke through doctor and wrote a report with
  `kind=discord_bridge`, `ok=true`, `transport=discord_voice`,
  `input_pcm48_bytes=3840`, `sidecar_pcm16_bytes=640`, `mixer_frames=1`,
  `mixer_frame_bytes=3840`, `barge_in_sent=true`, `mixer_stop_calls=2`, and
  events
  `transcript.partial,transcript.final,assistant.text.partial,audio.output.chunk,assistant.commit,barge_in`.
  The same doctor run still reported the expected remaining production-readiness
  issues: `not_live_like`, `sidecar_unverified`, `missing_evidence_report`, and
  unhealthy configured sidecar.
- `uv run --extra dev --extra voice python -m hermes_cli.realtime_voice_elevenlabs_bridge --check --strict --production-en-ja`
  passed with `stt_model=scribe_v2_realtime`, `tts_model=eleven_flash_v2_5`,
  configured voice ID, `language=auto`, output languages `en,ja`, and configured
  bridge auth token.
- Using a temporary `HERMES_HOME=/tmp/hermes-elevenlabs-evidence` so user config
  was not mutated, `realtime_voice_profile --preset elevenlabs --apply` wrote a
  live-like profile pointing at `http://127.0.0.1:8767`.
- With that temporary profile and the existing ElevenLabs credentials,
  `uv run --extra dev --extra voice python -m hermes_cli.realtime_voice_alpha_evidence --runs 3 --provider elevenlabs --start-bridge --output-dir /tmp/hermes-elevenlabs-evidence/reports --prefix elevenlabs-alpha --overwrite --bridge-timeout-seconds 30`
  collected real-provider EN/JA evidence and validated it:
  `Realtime voice alpha evidence OK: 42 smoke result(s) across 3 run(s)`.
  Summary:
  - `audio_to_partial_transcript`: `p50=730ms`, `p95=736ms`, `max=736ms`, `n=12`
  - `final_transcript_to_first_text`: `p50=0ms`, `p95=0ms`, `max=0ms`, `n=12`
  - `final_transcript_to_first_audio`: `p50=330ms`, `p95=412ms`, `max=451ms`, `n=24`
  - `barge_in_ack`: `p50=1ms`, `p95=1ms`, `max=1ms`, `n=3`
  The ElevenLabs evidence path declares a provider-specific
  `audio_to_partial_transcript_ms` ceiling of `1000ms`; the default live target
  remains `300ms`, so doctor displays the raw partial-latency misses while the
  alpha evidence validator accepts the provider-specific ceiling.
- `uv run --extra dev --extra voice python -m hermes_cli.realtime_voice_report /tmp/hermes-elevenlabs-evidence/reports/*.json --alpha --min-runs 3`
  passed with the same `42` result summary.
- The accepted ElevenLabs evidence was copied into the repo-local artifact path
  `artifacts/realtime-voice-evidence/`:
  `elevenlabs-alpha-001.json`, `elevenlabs-alpha-002.json`, and
  `elevenlabs-alpha-003.json`. A secret scan against those copied artifacts
  found no matches for key/token/secret/authorization patterns, and
  `uv run --extra dev --extra voice python -m hermes_cli.realtime_voice_report artifacts/realtime-voice-evidence/*.json --alpha --min-runs 3`
  passed with:
  `Realtime voice smoke report OK: 42 result(s) across 3 run(s)`,
  `audio_to_partial_transcript p50=730ms p95=736ms max=736ms n=12`,
  `final_transcript_to_first_text p50=0ms p95=0ms max=0ms n=12`,
  `final_transcript_to_first_audio p50=330ms p95=412ms max=451ms n=24`,
  and `barge_in_ack p50=1ms p95=1ms max=1ms n=3`.
- Added `realtime_voice_report --apply-production-evidence` so already-saved
  and validated production evidence can be wired into
  `voice.realtime.production_evidence_report` without rerunning provider calls.
  The apply path requires `--alpha`, uses the run-level validator, rejects
  loopback-marked evidence as production evidence, and applies a shared parent
  directory when multiple report files from one evidence bundle are passed.
  Doctor and the ElevenLabs realtime voice guide now surface this path.
- Verified the new apply path against the repo-local ElevenLabs evidence with a
  temporary `HERMES_HOME` so the real user config was not mutated:
  `HERMES_HOME="$(mktemp -d /tmp/hermes-report-apply-XXXXXX)" uv run --extra dev --extra voice python -m hermes_cli.realtime_voice_report artifacts/realtime-voice-evidence/*.json --alpha --min-runs 3 --apply-production-evidence`.
  Result: `Realtime voice smoke report OK: 42 result(s) across 3 run(s)`,
  followed by `Updated realtime voice production_evidence_report in
  /tmp/hermes-report-apply-*/config.yaml`; the temp config contained
  `voice.realtime.production_evidence_report: artifacts/realtime-voice-evidence`.
- Verified:
  - `uv run --extra dev --extra voice python -m pytest tests/agent/test_realtime_voice_smoke_report.py tests/hermes_cli/test_doctor.py -q`
  - Result: `141 passed, 1 warning in 5.41s`.
  - `uv run --extra dev --extra voice python -m pytest tests/agent/test_realtime_voice.py tests/agent/test_realtime_voice_smoke.py tests/agent/test_realtime_voice_loopback_bridge.py tests/agent/test_realtime_voice_smoke_report.py tests/hermes_cli/test_realtime_voice_alpha_evidence.py tests/hermes_cli/test_discord_realtime_voice_smoke.py tests/hermes_cli/test_discord_voice_live_probe.py tests/hermes_cli/test_doctor.py tests/hermes_cli/test_realtime_voice_profile.py tests/gateway/test_discord_realtime_voice.py tests/gateway/test_discord_voice_mixer.py tests/gateway/test_voice_command.py tests/agent/test_realtime_voice_oracle.py tests/hermes_cli/test_web_server.py tests/hermes_cli/test_config.py -q`
  - Result: `997 passed in 30.67s`.
- Added `hermes_cli.realtime_voice_launchd`, a repo-safe macOS LaunchAgent
  plist generator for the ElevenLabs bridge and Hermes reference sidecar. It
  writes reviewable plists to a chosen directory, loads `~/.hermes/.env`, runs
  from the selected checkout, writes logs under `~/.hermes/logs`, maps
  `HERMES_STREAMING_STT_BRIDGE_TOKEN` into the sidecar's streaming STT/TTS token
  environment variables, and does not install or bootstrap services by default.
  The ElevenLabs realtime voice guide now documents this persistent-service
  flow.
- Generated repo-local service artifacts:
  - `artifacts/realtime-voice-launchd/ai.hermes.realtime-voice.elevenlabs-bridge.plist`
  - `artifacts/realtime-voice-launchd/ai.hermes.realtime-voice.sidecar.plist`
  A secret scan across `artifacts/realtime-voice-launchd` and
  `artifacts/realtime-voice-evidence` found no key/token/secret/authorization
  matches, and both generated plists parsed successfully with `plistlib`.
- Verified:
  - `uv run --extra dev --extra voice python -m pytest tests/hermes_cli/test_realtime_voice_launchd.py tests/hermes_cli/test_realtime_voice_profile.py tests/hermes_cli/test_realtime_voice_alpha_evidence.py tests/agent/test_realtime_voice_smoke_report.py tests/hermes_cli/test_doctor.py -q`
  - Result: `182 passed, 1 warning in 5.75s`.
  - `uv run --extra dev --extra voice python -m pytest tests/agent/test_realtime_voice.py tests/agent/test_realtime_voice_smoke.py tests/agent/test_realtime_voice_loopback_bridge.py tests/agent/test_realtime_voice_smoke_report.py tests/hermes_cli/test_realtime_voice_alpha_evidence.py tests/hermes_cli/test_discord_realtime_voice_smoke.py tests/hermes_cli/test_discord_voice_live_probe.py tests/hermes_cli/test_doctor.py tests/hermes_cli/test_realtime_voice_profile.py tests/hermes_cli/test_realtime_voice_launchd.py tests/gateway/test_discord_realtime_voice.py tests/gateway/test_discord_voice_mixer.py tests/gateway/test_voice_command.py tests/agent/test_realtime_voice_oracle.py tests/hermes_cli/test_web_server.py tests/hermes_cli/test_config.py -q`
  - Result: `1001 passed in 30.48s`.
- A temp-home `hermes doctor --realtime-voice --realtime-voice-report /tmp/hermes-elevenlabs-evidence/final-doctor-report.json`
  confirmed the evidence exists but also confirmed live readiness is not active
  unless the provider bridge/sidecar is running; it reported
  `current realtime stack does not match evidence manifest` after the temporary
  bridge was shut down.
- Without mutating the real active config, manually starting the ElevenLabs
  bridge at `127.0.0.1:8767` and the reference sidecar at `127.0.0.1:8765`
  with streaming STT/TTS environment pointed at that bridge made active-config
  doctor readiness live-like:
  `uv run --extra dev --extra voice hermes doctor --realtime-voice --realtime-voice-smoke --realtime-voice-report /tmp/hermes-active-realtime-doctor-live.json`
  reported `Live conversation quality (streaming_text: streaming_stt_tts, live-like: yes)`,
  `Voice sidecar health (managed_loopback)`, sidecar protocol smoke
  `ready=27ms, transcript.final=27ms`, and session-turn smoke
  `first_text=0ms <= 500ms, first_audio=566ms <= 900ms`. The only realtime
  readiness issue left in that run was `missing_evidence_report`; the temporary
  processes were then stopped and a process check found no lingering
  `realtime_voice_elevenlabs_bridge`, `realtime_voice_sidecar`, `uvicorn`, or
  `discord_voice_live_probe` process.
- A read-only Discord gateway discovery using the configured bot token found
  `hermes-macmini#3355` in one guild, `jetha dev server`, with one voice
  channel, `General`; the bot has both `Connect` and `Speak` permissions there.
- A bounded live Discord voice-channel probe against `jetha dev server /
  General` loaded opus, joined the real voice channel, installed
  `VoiceReceiver`, played silent PCM through `VoiceMixer`, and disconnected
  cleanly:
  `opus_loaded=true`, `connect_perm=true`, `speak_perm=true`,
  `members_before=0`, `connected=true`, `accepted_audio_source=true`,
  `played=true`, `playing_during_probe=true`,
  `receiver_running_after_start=true`, `receiver_frames=0`,
  `receiver_speech_start=0`, `disconnected=true`.
  Because no other members were in the channel, this proves real Discord
  gateway/channel permissions, voice connect, mixer playback acceptance, receiver
  installation, and shutdown, but not live user speech capture.
- The reusable probe command
  `uv run --extra dev --extra voice python -m hermes_cli.discord_voice_live_probe --voice-channel-name General --wait-seconds 1 --report /tmp/hermes-discord-live-probe.json`
  passed with the same invariants:
  `ok=true`, `guild_name=jetha dev server`, `voice_channel_name=General`,
  `connect_perm=true`, `speak_perm=true`, `members_before=0`,
  `connected=true`, `opus_loaded=true`, `accepted_audio_source=true`,
  `played=true`, `playing_during_probe=true`, `receiver_started=true`,
  `receiver_frames=0`, `receiver_speech_start=0`, `disconnected=true`.
- The inbound-required form
  `uv run --extra dev --extra voice python -m hermes_cli.discord_voice_live_probe --voice-channel-name General --wait-seconds 1 --require-inbound --report /tmp/hermes-discord-live-probe-require-inbound.json`
  exited `1` with `ok=false` and
  `error="live Discord voice probe did not satisfy invariants"` while still
  proving join/play/receiver/leave. This is the expected result with
  `members_before=0`; rerun the same command while a human or controlled second
  client is speaking to prove inbound live speech capture.
- After probe diagnostic hardening, the inbound-required form
  `uv run --extra dev --extra voice python -m hermes_cli.discord_voice_live_probe --voice-channel-name General --wait-seconds 1 --require-inbound --report /tmp/hermes-discord-live-probe-require-inbound-v2.json`
  still exited `1` because the channel was empty, but now reports
  `failure_reason=inbound_required_but_no_other_members`,
  `inbound_observed=false`, `members_before=0`, `members_after=0`,
  `receiver_frames=0`, and `receiver_speech_start=0`.
- Added `hermes doctor --discord-voice-live-probe` so the bounded real Discord
  voice-channel probe can be run through doctor and captured in the standard
  realtime voice report format. The doctor flags support wait duration,
  voice-channel ID/name override, and an inbound-required mode for the final
  human-or-second-client speech proof. Reports use `kind=discord_live_probe`.
- Verified the doctor-integrated live probe without requiring inbound speech:
  `uv run --extra dev --extra voice hermes doctor --discord-voice-live-probe --discord-voice-live-probe-wait-seconds 1 --realtime-voice-report /tmp/hermes-discord-live-probe-doctor.json`.
  Result: doctor passed the Discord live probe for `jetha dev server / General`
  with `members=0->0`, `receiver_frames=0`, and `inbound=no`; the report
  contains `kind=discord_live_probe`, `ok=true`, `connect_perm=true`,
  `speak_perm=true`, `connected=true`, `opus_loaded=true`,
  `accepted_audio_source=true`, `played=true`, `playing_during_probe=true`,
  `receiver_started=true`, `disconnected=true`, and `require_inbound=false`.
- Verified the doctor-integrated inbound-required form:
  `uv run --extra dev --extra voice hermes doctor --discord-voice-live-probe --discord-voice-live-probe-require-inbound --discord-voice-live-probe-wait-seconds 1 --realtime-voice-report /tmp/hermes-discord-live-probe-doctor-require-inbound.json`.
  Result: the probe failed only on the expected empty-channel invariant,
  `failure_reason=inbound_required_but_no_other_members`, with the same
  join/play/receiver/leave invariants true and doctor surfacing the action:
  `Rerun while a human or controlled second Discord client is speaking in the
  voice channel`.
- Verified:
  - `uv run --extra dev --extra voice python -m pytest tests/hermes_cli/test_doctor.py tests/hermes_cli/test_discord_voice_live_probe.py -q`
  - Result: `105 passed, 1 warning in 4.98s`.
  - `uv run --extra dev --extra voice python -m pytest tests/agent/test_realtime_voice.py tests/agent/test_realtime_voice_smoke.py tests/agent/test_realtime_voice_loopback_bridge.py tests/agent/test_realtime_voice_smoke_report.py tests/hermes_cli/test_realtime_voice_alpha_evidence.py tests/hermes_cli/test_discord_realtime_voice_smoke.py tests/hermes_cli/test_discord_voice_live_probe.py tests/hermes_cli/test_doctor.py tests/hermes_cli/test_realtime_voice_profile.py tests/hermes_cli/test_realtime_voice_launchd.py tests/gateway/test_discord_realtime_voice.py tests/gateway/test_discord_voice_mixer.py tests/gateway/test_voice_command.py tests/agent/test_realtime_voice_oracle.py tests/hermes_cli/test_web_server.py tests/hermes_cli/test_config.py -q`
  - Result: `1004 passed in 30.57s`.
- Added `realtime_voice_report --discord-live-probe` and
  `--require-inbound` so saved doctor live-probe reports are independently
  machine-verifiable. The non-inbound validator checks connect/speak
  permissions, voice connection, Opus, mixer `AudioSource` acceptance, playback,
  receiver startup, and clean disconnect. The inbound-required validator also
  requires receiver frames or speech-start callbacks.
- Verified:
  - `uv run --extra dev --extra voice python -m pytest tests/agent/test_realtime_voice_smoke_report.py -q`
  - Result: `46 passed in 0.46s`.
  - `uv run --extra dev --extra voice python -m hermes_cli.realtime_voice_report /tmp/hermes-discord-live-probe-doctor.json --discord-live-probe`
  - Result: `Realtime voice smoke report OK: 1 result(s) across 1 run(s)`.
  - `uv run --extra dev --extra voice python -m hermes_cli.realtime_voice_report /tmp/hermes-discord-live-probe-doctor-require-inbound.json --discord-live-probe --require-inbound`
  - Result: exited `1` with `discord_live_probe: no passing probe
    (inbound_required_but_no_other_members)` and `discord_live_probe: inbound
    speech not observed (inbound_required_but_no_other_members)`, which is the
    expected empty-channel blocker.
  - `uv run --extra dev --extra voice python -m pytest tests/agent/test_realtime_voice.py tests/agent/test_realtime_voice_smoke.py tests/agent/test_realtime_voice_loopback_bridge.py tests/agent/test_realtime_voice_smoke_report.py tests/hermes_cli/test_realtime_voice_alpha_evidence.py tests/hermes_cli/test_discord_realtime_voice_smoke.py tests/hermes_cli/test_discord_voice_live_probe.py tests/hermes_cli/test_doctor.py tests/hermes_cli/test_realtime_voice_profile.py tests/hermes_cli/test_realtime_voice_launchd.py tests/gateway/test_discord_realtime_voice.py tests/gateway/test_discord_voice_mixer.py tests/gateway/test_voice_command.py tests/agent/test_realtime_voice_oracle.py tests/hermes_cli/test_web_server.py tests/hermes_cli/test_config.py -q`
  - Result: `1007 passed in 30.78s`.
- `git diff --check` passed.

Still remaining before this goal can be marked complete:

- Apply or point the real Hermes config at accepted production evidence, and run
  the ElevenLabs bridge/reference sidecar as services when validating persistent
  runtime readiness. The active config already points at the ElevenLabs bridge
  URL and becomes live-like when the bridge/sidecar are running, but
  `production_evidence_report` is still unset in the real config.
- Run a real Discord voice-channel validation while a human or controlled second
  Discord client speaks in the channel. The live probe now proves gateway
  permissions, join, mixer playback, receiver installation, and leave behavior,
  but not inbound live speech packets because `General` had zero other members
  during the probe.

---

## Goals

- G1. `/voice join` must establish one explicit voice mode per guild:
  `realtime_active`, `legacy_voice_active`, `degraded_no_sidecar`, or `failed`.
- G2. Realtime voice must not silently fall through into the legacy path while
  still presenting itself as realtime.
- G3. The model must know when it is in a live Discord voice session and must
  not deny voice-channel listening or speaking capability in that context.
- G4. Barge-in must stop currently playing speech immediately and supersede stale
  assistant/TTS output.
- G5. The mixer playback path must be the primary Discord realtime output path.
- G6. Sidecar startup, degradation, and shutdown must be explicit, observable,
  and tested.
- G7. Legacy Whisper plus file-TTS fallback must remain available, but only as a
  clearly reported degraded mode.
- G8. The voice experience must give fast spoken feedback before long work.
- G9. Logs and smoke reports must include enough latency and state information to
  diagnose regressions without replaying Discord manually.
- G10. Test coverage must reproduce the June 9 failure modes.

---

## Non-Goals

- Do not rewrite the entire Discord adapter.
- Do not remove legacy Discord voice STT/TTS; keep it as fallback.
- Do not require a specific external realtime provider for all users.
- Do not make the model responsible for transport state inference. The gateway
  must inject explicit, accurate state.
- Do not rely on logs alone as behavior contracts. Logs are evidence and
  observability; tests own correctness.

---

## Target Behavior

### Healthy Realtime Session

1. User runs `/voice join`.
2. Hermes joins the channel and installs `VoiceReceiver`.
3. Hermes installs a continuous `VoiceMixer` accepted by `discord.py`.
4. Hermes connects the realtime sidecar or provider bridge.
5. Hermes reports voice mode as realtime, not generic TTS.
6. User speech is streamed as audio frames to the realtime session.
7. Realtime path emits audio/reflex events first; transcript partial/final
   events are optional auxiliary evidence when enabled.
8. Hermes emits a fast acknowledgement for work that will take more than a
   short moment.
9. TTS/audio output is played through the mixer.
10. User speech during playback triggers barge-in and stops speech promptly.

### Degraded Legacy Session

1. User runs `/voice join`.
2. Hermes joins and starts the receiver.
3. Mixer or sidecar setup fails.
4. Hermes explicitly reports degraded mode and reason.
5. User speech is processed through legacy utterance buffering plus local STT.
6. TTS may use file playback.
7. Barge-in still stops one-shot playback as best effort.
8. `/voice status` shows degraded state and the active fallback reason.

### Failed Session

1. Join, receiver, permissions, Opus, mixer, and sidecar failures are surfaced.
2. Hermes must not say "voice mode enabled" if it cannot listen or speak.
3. The adapter must leave no half-open receiver, mixer, sidecar, or voice client.

---

## Implementation Units

### U1. Add an explicit Discord voice session state model

**Goal:** Make the adapter track the actual mode for each guild instead of
inferring from scattered dictionaries.

**Files:**

- `plugins/platforms/discord/adapter.py`
- `tests/gateway/test_voice_command.py`
- `tests/gateway/test_discord_realtime_voice.py`

**Approach:**

- Add a small state object per guild containing:
  - mode
  - voice channel id
  - text channel id
  - mixer installed
  - receiver running
  - sidecar running
  - fallback reason
  - last state transition timestamp
- Use this state in `/voice status`, prompt injection, legacy STT gating, and
  join/leave cleanup.
- Treat state transitions as the source of truth, not `_voice_mixers`,
  `_realtime_voice_sessions`, or `_voice_clients` independently.

**Acceptance criteria:**

- Tests prove realtime-active suppresses legacy STT.
- Tests prove degraded mode allows legacy STT.
- Tests prove failed startup does not leave stale session state.

### U2. Harden mixer startup and playback path

**Goal:** Ensure the continuous mixer is a valid Discord audio source and is the
primary playback path for realtime voice.

**Files:**

- `plugins/platforms/discord/voice_mixer.py`
- `plugins/platforms/discord/adapter.py`
- `tests/gateway/test_discord_voice_mixer.py`
- `tests/gateway/test_discord_realtime_voice.py`

**Approach:**

- Keep `VoiceMixer` as a `discord.AudioSource` subclass when Discord is present.
- Add a startup assertion or health signal after `vc.play(mixer)` succeeds.
- If mixer install fails, set degraded mode with reason
  `mixer_start_failed`.
- Ensure `play_in_voice_channel` routes to mixer when available.
- Ensure realtime audio chunks enqueue to mixer and stale chunks are discarded.

**Acceptance criteria:**

- A mocked Discord `VoiceClient.play()` requiring `AudioSource` accepts the mixer.
- Realtime output audio reaches `mixer.enqueue_speech_frame`.
- Mixer failure produces degraded state and visible status, not silent fallback.

### U3. Make sidecar startup, fallback, and shutdown explicit

**Goal:** Avoid "realtime session started" logs when the sidecar path is not
actually usable.

**Files:**

- `plugins/platforms/discord/adapter.py`
- `plugins/platforms/discord/realtime_voice.py`
- `agent/realtime_voice_session.py`
- `agent/realtime_voice_text_engine.py`
- `tests/gateway/test_discord_realtime_voice.py`

**Approach:**

- During `/voice join`, wait for sidecar readiness before marking
  `realtime_active`.
- If the sidecar is unavailable, set `degraded_no_sidecar` and record the
  sanitized error.
- On sidecar event stream failure, transition to degraded mode and stop sending
  realtime frames.
- On `/voice leave`, close the sidecar session before tearing down receiver and
  voice client.
- Add close timeouts so shutdown cannot hang.

**Acceptance criteria:**

- Sidecar unavailable creates degraded state and does not install a dead
  realtime session.
- Sidecar close is called on leave.
- Sidecar event stream failure transitions to fallback once and logs why.

### U4. Inject accurate live voice capability context

**Goal:** Prevent the model from claiming it cannot hear or speak in Discord
voice when live voice is active.

**Files:**

- `gateway/run.py`
- `plugins/platforms/discord/adapter.py`
- `agent/realtime_voice_oracle.py`
- `tests/gateway/test_voice_command.py`
- `tests/gateway/test_discord_channel_prompts.py`

**Approach:**

- Extend `get_voice_channel_context()` or add a new structured context helper
  that includes:
  - active mode
  - whether Hermes is listening to the voice channel
  - whether Hermes can speak back into the channel
  - whether the session is realtime or degraded
  - concise voice-response policy
- Inject this into `combined_ephemeral` for Discord voice turns.
- Add a stronger voice oracle prompt for realtime sessions:
  - answer naturally and briefly
  - you are in a live Discord voice session
  - do not deny live listening/speaking capability when state says active
  - mention degradation only if relevant

**Acceptance criteria:**

- Tests capture `ephemeral_system_prompt` and assert it contains live Discord
  voice capability facts when the bot is in a voice channel.
- Tests assert no live voice prompt is injected when the bot is not connected.
- Realtime oracle prompt includes live voice capability context.

### U5. Make barge-in immediate and generation-safe

**Goal:** User speech during playback must stop current speech and prevent stale
assistant output from continuing.

**Files:**

- `plugins/platforms/discord/adapter.py`
- `plugins/platforms/discord/realtime_voice.py`
- `agent/realtime_voice_text_engine.py`
- `agent/realtime_voice_session.py`
- `tests/gateway/test_discord_realtime_voice.py`
- `tests/gateway/test_voice_command.py`

**Approach:**

- On speech-start callback:
  - stop mixer speech if active
  - stop legacy `VoiceClient` playback if mixer is missing
  - send a `BARGE_IN` event even when no playback is active
  - include `playback_active`
- Increment playback generation on barge-in.
- Drop output chunks and assistant commits from older generations.
- Interrupt the active oracle turn when possible.

**Acceptance criteria:**

- User speech during mixer playback calls `stop_speech()` immediately.
- User speech during legacy playback calls `vc.stop()`.
- Stale output chunks after barge-in are ignored.
- Barge-in emits measurable `barge_in_ack_ms`.

### U6. Reduce perceived latency

**Goal:** Avoid June 9's 18-19s silent waits plus multi-second whole-response
TTS delay.

**Files:**

- `plugins/platforms/discord/adapter.py`
- `agent/realtime_voice_text_engine.py`
- `agent/realtime_voice_planner.py`
- `tools/tts_tool.py` if needed
- relevant gateway tests

**Approach:**

- Preflight lazy voice dependencies during `/voice join` or `hermes doctor`.
- Use a short acknowledgement for any request that will run tools or exceed a
  small latency threshold.
- Chunk spoken output by phrase/sentence instead of waiting for a whole final
  response.
- Prefer streaming sidecar TTS when available.
- Keep fallback file-TTS, but log it as fallback with timing.

**Acceptance criteria:**

- First spoken acknowledgement can be emitted independently of final response.
- Chunked TTS starts before full oracle answer completion in realtime mode.
- Logs include time to reflex acknowledgement, first assistant text/audio, and
  optional transcript-hypothesis latency when enabled.

### U7. Improve `/voice status` and user-visible diagnostics

**Goal:** The user should know whether they are testing realtime or fallback.

**Files:**

- `gateway/slash_commands.py`
- `plugins/platforms/discord/adapter.py`
- locale files if needed
- `tests/gateway/test_voice_command.py`

**Approach:**

- Extend `/voice status` to include:
  - mode
  - voice channel
  - participants
  - realtime sidecar status
  - mixer status
  - fallback reason
  - last barge-in time or status if useful
- On `/voice join`, return a precise status:
  - "Realtime voice connected"
  - "Voice connected in fallback mode: sidecar unavailable"
  - "Joined voice but cannot play audio: mixer failed"

**Acceptance criteria:**

- Status tests cover healthy realtime, degraded fallback, and failed mixer.
- Join response does not overstate capabilities.

### U8. Add observability and smoke reports

**Goal:** Make future Discord voice tests diagnosable from logs and artifacts.

**Files:**

- `plugins/platforms/discord/adapter.py`
- `plugins/platforms/discord/realtime_voice.py`
- `agent/realtime_voice_session.py`
- `agent/realtime_voice_smoke_report.py`
- `hermes_cli/subcommands/doctor.py`
- `tests/hermes_cli/test_doctor.py`

**Approach:**

- Add structured log fields for:
  - voice session mode
  - mixer start result
  - sidecar start result
  - first input frame timestamp
  - optional transcript-hypothesis partial/final latency, labeled as
    non-authoritative evidence
  - first audio output latency
  - TTS provider and duration
  - barge-in ack latency
  - fallback reason
- Extend realtime smoke reports to flag missing metrics and quality misses.

**Acceptance criteria:**

- A focused doctor/smoke test can fail on missing core metrics.
- Logs are sufficient to reconstruct the session state machine.

### U9. Expand regression coverage for the June 9 failure shape

**Goal:** Prevent this exact class of bug from coming back.

**Files:**

- `tests/gateway/test_discord_realtime_voice.py`
- `tests/gateway/test_voice_command.py`
- `tests/gateway/test_discord_voice_mixer.py`
- `tests/hermes_cli/test_doctor.py`

**Test scenarios:**

- Mixer must satisfy `discord.AudioSource`.
- Realtime-active sessions suppress legacy utterance processing.
- Sidecar unavailable falls back visibly.
- Speech-start stops mixer playback.
- Speech-start stops one-shot playback when mixer is missing.
- Barge-in sends event even when no playback is active.
- Stale TTS/audio output after barge-in is dropped.
- Prompt includes live Discord voice capability context.
- `/voice status` reports realtime/degraded/failed accurately.
- Sidecar close happens during `/voice leave`.
- Focused pytest suite includes all voice regression files.

---

## Milestones

### M1. Correctness and State

- Implement explicit voice session state.
- Harden mixer startup.
- Harden sidecar startup and fallback.
- Add status output for actual mode.

### M2. Model Awareness and Barge-In

- Inject live voice capability context.
- Strengthen realtime oracle prompt.
- Make barge-in stop playback and supersede stale output.
- Add generation-safe stale-output tests.

### M3. Latency and Observability

- Add immediate acknowledgements.
- Chunk spoken output.
- Preflight voice dependencies.
- Add structured latency logs and smoke report checks.

---

## Verification Commands

Run focused tests after each milestone:

```bash
uv run --extra dev --extra voice python -m pytest \
  tests/gateway/test_discord_realtime_voice.py \
  tests/gateway/test_discord_voice_mixer.py \
  tests/gateway/test_voice_command.py
```

Run broader conflict-sensitive tests after touching doctor, sidecar, or gateway
prompt paths:

```bash
uv run --extra dev --extra voice python -m pytest \
  tests/hermes_cli/test_doctor.py \
  tests/hermes_cli/test_web_server.py \
  tests/gateway/test_discord_realtime_voice.py \
  tests/gateway/test_discord_voice_mixer.py \
  tests/gateway/test_voice_command.py
```

---

## Done Criteria

- `/voice join` reports the real active mode.
- A healthy realtime session does not process utterances through legacy local
  Whisper.
- A degraded session reports why it degraded and uses legacy fallback
  intentionally.
- Hermes never denies live Discord voice capability when the injected state says
  it is currently listening/speaking in a live voice channel.
- Barge-in stops current speech within one callback turn and drops stale output.
- `/voice leave` closes sidecar, receiver, mixer, and voice client cleanly.
- Focused tests pass with exact results recorded in the PR or work log.
- Logs from a manual Discord test can reconstruct join, mode, STT, TTS,
  playback, barge-in, and shutdown timing.
