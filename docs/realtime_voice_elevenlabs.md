# Hermes Realtime Voice: ElevenLabs Bridge

Hermes can run live voice through its existing realtime sidecar contract with
ElevenLabs as a streaming speech provider fallback or comparison bridge:

```text
browser/desktop mic
  -> Hermes realtime websocket
  -> reference sidecar
  -> raw audio turn plus optional ElevenLabs classic_asr_hypothesis
  -> Gemma/interpreter evidence bundle in full KAME mode
  -> Hermes active /model oracle
  -> ElevenLabs realtime TTS bridge
  -> browser/desktop playback
```

This is useful for bring-up and provider-quality comparison, but it is not the
target full KAME control path. In full KAME mode, the reflex owns floor control,
Gemma interprets clipped raw audio plus labeled transcript hypotheses, and
Hermes' active `/model` remains the oracle. ElevenLabs STT output should enter
that path only as optional transcript-hypothesis context unless Hermes is
explicitly running a degraded text-only fallback.

Only explicit degraded text-only fallback mode may route provider STT into
Hermes text handling. That path is useful for bring-up, captions,
clarification, and low-risk drafting, but it is not full raw-audio KAME
evidence and cannot authorize spend, tools, calls, files, memory writes,
external messages, or durable user text by itself. Full KAME evidence should
preserve the raw audio reference and treat ElevenLabs text as
`classic_asr_hypothesis` context.

## Provider-Neutral Architecture

The realtime voice sidecar speaks the same provider-neutral streaming contract
regardless of speech provider:

- `GET /health`
- `WS /v1/streaming-stt/session`
- `WS /v1/streaming-tts/session`

Provider bridges sit at the edge of that contract. The current matrix is:

| Provider path | Status | Role |
| --- | --- | --- |
| Deepgram bridge | Existing | Streaming STT/TTS fallback and provider-comparison baseline |
| ElevenLabs bridge | Implemented | Streaming STT/TTS fallback and provider-comparison path |
| OpenAI Realtime | Implemented | Native speech-to-speech frontend provider behind the sidecar |
| Gemini Live | Implemented | Native speech-to-speech frontend provider with KAME-scoped bridge tools |
| Gemma/audio interpreter | Future | Interpreter/evidence lane over clipped raw audio plus labeled transcript hypotheses; not a speech frontend and not the Hermes oracle |
| Native speech-to-speech | Future | Another provider path, not required for first production readiness |

The design decisions are:

- Keep provider code at the edge.
- Keep Hermes' active `/model` as the only oracle; provider frontends and STT
  bridges only produce reflex, interpreter, TTS, or hypothesis evidence.
- Keep native-provider tool calls scoped to bridge tools such as
  `ask_hermes_oracle`; provider frontends do not execute arbitrary Hermes tools.
- Keep Gemma-style audio models on the interpreter/evidence lane unless a
  separate low-latency reflex implementation proves it can handle live floor
  control.
- Keep desktop/UI and speech inference split so the UI machine does not need to
  run the speech provider or model.
- Avoid English-only protocol assumptions. Production evidence currently targets
  English and Japanese first.
- Do not claim production readiness from static checks; require live audio
  evidence.

## Environment

Required:

```powershell
$env:ELEVENLABS_API_KEY = "<api key>"
$env:ELEVENLABS_VOICE_ID = "<voice id>"
```

Recommended for the Hermes sidecar bearer token:

```powershell
python -m hermes_cli.realtime_voice_elevenlabs_bridge --generate-token
```

Optional:

```powershell
$env:HERMES_ELEVENLABS_STT_MODEL = "scribe_v2_realtime"
$env:HERMES_ELEVENLABS_TTS_MODEL = "eleven_flash_v2_5"
$env:HERMES_ELEVENLABS_LANGUAGE = "auto"
$env:HERMES_ELEVENLABS_OUTPUT_LANGUAGES = "en,ja"
$env:HERMES_ELEVENLABS_OUTPUT_FORMAT = "pcm_24000"
```

Do not commit API keys, bearer tokens, voice IDs, or machine-specific hostnames.
Use environment variables or the normal Hermes secret configuration path.

## Local Checks

Check local bridge prerequisites:

```powershell
python -m hermes_cli.realtime_voice_elevenlabs_bridge --check --strict --production-en-ja
```

Start the ElevenLabs bridge:

```powershell
python -m hermes_cli.realtime_voice_elevenlabs_bridge --host 127.0.0.1 --port 8767 --production-en-ja
```

Point the Hermes reference sidecar at the bridge for both streaming STT and
streaming TTS:

```powershell
$env:HERMES_VOICE_STREAMING_STT_BASE_URL = "http://127.0.0.1:8767"
$env:HERMES_VOICE_STREAMING_TTS_BASE_URL = "http://127.0.0.1:8767"
$env:HERMES_VOICE_STREAMING_STT_MODEL = "scribe_v2_realtime"
$env:HERMES_VOICE_STREAMING_TTS_MODEL = "eleven_flash_v2_5"
$env:HERMES_VOICE_INPUT_LANGUAGES = "en,ja"
$env:HERMES_VOICE_OUTPUT_LANGUAGES = "en,ja"
python -m hermes_cli.realtime_voice_sidecar
```

## macOS LaunchAgent Services

For persistent local testing on macOS, generate reviewable LaunchAgent plists
for the ElevenLabs bridge and Hermes reference sidecar:

```powershell
python -m hermes_cli.realtime_voice_launchd --output-dir artifacts/realtime-voice-launchd --repo-dir .
```

The generator writes:

- `artifacts/realtime-voice-launchd/ai.hermes.realtime-voice.elevenlabs-bridge.plist`
- `artifacts/realtime-voice-launchd/ai.hermes.realtime-voice.sidecar.plist`

The services load `~/.hermes/.env`, run from the selected repo checkout, write
logs under `~/.hermes/logs`, and point the sidecar at the local ElevenLabs
bridge on `127.0.0.1:8767`. The sidecar maps
`HERMES_STREAMING_STT_BRIDGE_TOKEN` into its streaming STT/TTS token variables,
matching the profile preset and evidence runner defaults.

Install after reviewing the generated plists:

```powershell
cp artifacts/realtime-voice-launchd/*.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.hermes.realtime-voice.elevenlabs-bridge.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.hermes.realtime-voice.sidecar.plist
```

Inspect logs with:

```powershell
tail -f ~/.hermes/logs/realtime-voice-elevenlabs-bridge.log
tail -f ~/.hermes/logs/realtime-voice-sidecar.log
```

## Live Provider Validation

The bridge `--check` command verifies local configuration, the websocket Python
dependency, voice id presence, and Hermes bearer-token configuration. It does
not spend provider quota or prove the API key is active.

```powershell
python -m hermes_cli.realtime_voice_elevenlabs_bridge --check --strict --production-en-ja
```

For a minimal live key check, call ElevenLabs' voices endpoint with
`xi-api-key` and confirm a `200` response before running production evidence.
Do not commit the key or write it into repo-local config.

After bridge prerequisites pass, production alpha evidence can auto-start the
ElevenLabs provider bridge through the provider-neutral evidence runner:

```powershell
python -m hermes_cli.realtime_voice_alpha_evidence `
  --provider elevenlabs `
  --start-bridge `
  --runs 3 `
  --output-dir evidence/realtime-voice/elevenlabs
```

`--start-deepgram-bridge` remains supported as a backward-compatible alias for
Deepgram-only runs. New provider integrations should use `--provider <name>`
with `--start-bridge` so the evidence runner stays speech-provider-neutral.

Validate saved reports with:

```powershell
python -m hermes_cli.realtime_voice_report evidence/realtime-voice/elevenlabs/*.json --alpha --min-runs 3
```

After validation succeeds, point Hermes at the accepted evidence bundle:

```powershell
python -m hermes_cli.realtime_voice_report evidence/realtime-voice/elevenlabs/*.json --alpha --min-runs 3 --apply-production-evidence
```

This writes `voice.realtime.production_evidence_report` to the shared parent
directory of the validated reports. It refuses loopback-marked evidence and
does not rerun provider calls.

## Discord Live Validation

Local Discord bridge smoke validates PCM conversion, mixer playback, sidecar
events, and barge-in without using Discord credentials:

```powershell
python -m hermes_cli.discord_realtime_voice_smoke --report artifacts/realtime-voice-discord-bridge.json
```

For a real Discord gateway/channel check, use doctor so the probe is captured in
the normal realtime voice report format:

```powershell
hermes doctor --discord-voice-live-probe --discord-voice-live-probe-wait-seconds 5 --realtime-voice-report artifacts/realtime-voice-discord-live.json
```

Validate the saved probe report with:

```powershell
python -m hermes_cli.realtime_voice_report artifacts/realtime-voice-discord-live.json --discord-live-probe
```

That joins the configured voice channel, installs `VoiceReceiver`, plays mixer
audio, and leaves cleanly. To prove inbound receiver callbacks with live speech,
rerun while a human or controlled second Discord client is speaking:

```powershell
hermes doctor --discord-voice-live-probe --discord-voice-live-probe-require-inbound --discord-voice-live-probe-wait-seconds 15 --realtime-voice-report artifacts/realtime-voice-discord-live-inbound.json
```

Validate the inbound proof with:

```powershell
python -m hermes_cli.realtime_voice_report artifacts/realtime-voice-discord-live-inbound.json --discord-live-probe --require-inbound
```

If the channel is empty, the inbound-required form fails with
`inbound_required_but_no_other_members`; that proves the bot reached the channel
and receiver path but does not prove inbound speech frames.

For the final upstream evidence bundle, run the strict collector after setting
Discord and OpenAI Realtime credentials:

```powershell
$env:DISCORD_BOT_TOKEN = "<bot token>"
$env:DISCORD_GUILD_ID = "<guild id>"
$env:DISCORD_VOICE_CHANNEL_ID = "<voice channel id>"
$env:OPENAI_API_KEY = "<OpenAI Realtime API key>"
python -m hermes_cli.realtime_voice_live_evidence `
  --output-dir artifacts/realtime-voice-evidence/live-openai-discord `
  --require-live-discord `
  --require-openai-realtime `
  --require-inbound `
  --wait-seconds 15
```

The collector writes `manifest.json`, `discord-loopback.json`, and
`discord-live-probe.json`. Reports include the exact git commit and redacted
env/config readiness, but never write secret values.

For Gemini Live native speech-to-speech comparison evidence, use the Gemini
preset and strict collector flag:

```powershell
$env:GEMINI_API_KEY = "<Gemini API key>"
python -m hermes_cli.realtime_voice_profile --preset gemini --apply
python -m hermes_cli.realtime_voice_live_evidence `
  --output-dir artifacts/realtime-voice-evidence/live-gemini-discord `
  --require-live-discord `
  --require-gemini-live `
  --require-inbound `
  --wait-seconds 15
```

## ElevenLabs Bridge Evidence Checklist

An ElevenLabs bridge evidence bundle should include live, not mocked, runs that
prove:

1. `/health` reports streaming STT and streaming TTS capability.
2. Known English and Japanese audio fixtures produce provider transcript
   hypothesis events that match the expected fixture text after
   provider-normalized transcript matching.
3. The full Hermes realtime session path produces:
   - provenance-labeled `classic_asr_hypothesis` events,
   - Hermes assistant text only in explicit degraded text-only fallback mode,
   - streaming TTS audio chunks.
4. Barge-in sends an acknowledgement within the configured target and clears or
   resets queued TTS output.
5. Latency metrics are present for:
   - audio to provider transcript hypothesis,
   - raw-audio/interpreter or fallback-text input to first assistant text,
   - assistant text to first audio,
   - barge-in acknowledgement.

ElevenLabs can emit short Japanese utterances as a fast final transcript without
an earlier partial. The alpha evidence runner records an ElevenLabs-specific
partial-transcript ceiling in the report manifest and stamps STT entries with
that provider target. Deepgram keeps the stricter default target.

This evidence proves the ElevenLabs bridge path. It does not, by itself, prove
full KAME production readiness. Full KAME evidence must also show reflex
acknowledgement/floor-control timing, raw-audio interpreter evidence, preserved
transcript-hypothesis provenance, Hermes `/model` oracle routing, and
non-authoritative handling of ElevenLabs transcripts.

## Current Limits

- STT sends Hermes audio chunks as ElevenLabs `input_audio_chunk` websocket
  messages with manual commits on Hermes end-of-utterance.
- TTS uses ElevenLabs' realtime text-to-speech websocket and requests PCM output
  so Hermes can preserve its existing audio output contract.
- Barge-in resets the ElevenLabs TTS websocket to drop queued provider audio.
- `en,ja` is the production preset. Other languages can be advertised with
  `HERMES_ELEVENLABS_OUTPUT_LANGUAGES`, but they should not be claimed as
  production-ready until smoke evidence exists for them.
- Gemma/audio interpreters and native speech-to-speech frontends should
  implement the same sidecar bridge contract while preserving raw audio
  references.
  Moshi/OpenClaw/VoiceClaw transcript-like text should be attached as labeled
  hypotheses beside raw audio and must not become durable user text,
  `oracle_text`, or tool-critical arguments without interpreter/oracle promotion.
- In full KAME mode, ElevenLabs STT output should be attached as optional
  `classic_asr_hypothesis` evidence beside the raw audio, not promoted directly
  to durable user text or tool-critical oracle arguments.
