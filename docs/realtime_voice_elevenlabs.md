# Hermes Realtime Voice: ElevenLabs Bridge

Hermes can run live voice through its existing realtime sidecar contract with
ElevenLabs as the streaming speech provider:

```text
browser/desktop mic
  -> Hermes realtime websocket
  -> reference sidecar
  -> ElevenLabs realtime STT bridge
  -> Hermes oracle/backend/tool loop
  -> ElevenLabs realtime TTS bridge
  -> browser/desktop playback
```

This is a KAME-inspired realtime loop, but it deliberately keeps the Hermes
oracle/backend/tool layer in the middle. ElevenLabs handles the speech frontend
and speech output; it does not replace Hermes' configured backend model, tools,
or data access layer.

## Provider-Neutral Architecture

The realtime voice sidecar speaks the same provider-neutral streaming contract
regardless of speech provider:

- `GET /health`
- `WS /v1/streaming-stt/session`
- `WS /v1/streaming-tts/session`

Provider bridges sit at the edge of that contract. The current matrix is:

| Provider path | Status | Role |
| --- | --- | --- |
| Deepgram bridge | Existing | Streaming STT/TTS evidence baseline |
| ElevenLabs bridge | Implemented | Streaming STT plus realtime TTS |
| Gemma/audio frontend | Future | Remote audio-capable frontend bridge |
| Native speech-to-speech | Future | Another provider path, not required for first production readiness |

The design decisions are:

- Keep provider code at the edge.
- Keep the Hermes oracle in the middle.
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

## Evidence Checklist

A production evidence bundle should include live, not mocked, runs that prove:

1. `/health` reports streaming STT and streaming TTS capability.
2. Known English and Japanese audio fixtures produce final transcripts that
   match the expected fixture text after provider-normalized transcript matching.
3. The full Hermes realtime session path produces:
   - transcript final events,
   - Hermes assistant text,
   - streaming TTS audio chunks.
4. Barge-in sends an acknowledgement within the configured target and clears or
   resets queued TTS output.
5. Latency metrics are present for:
   - audio to partial or fast final transcript,
   - final transcript to first assistant text,
   - final transcript to first audio,
   - barge-in acknowledgement.

ElevenLabs can emit short Japanese utterances as a fast final transcript without
an earlier partial. The alpha evidence runner records an ElevenLabs-specific
partial-transcript ceiling in the report manifest and stamps STT entries with
that provider target. Deepgram keeps the stricter default target.

## Current Limits

- STT sends Hermes audio chunks as ElevenLabs `input_audio_chunk` websocket
  messages with manual commits on Hermes end-of-utterance.
- TTS uses ElevenLabs' realtime text-to-speech websocket and requests PCM output
  so Hermes can preserve its existing audio output contract.
- Barge-in resets the ElevenLabs TTS websocket to drop queued provider audio.
- `en,ja` is the production preset. Other languages can be advertised with
  `HERMES_ELEVENLABS_OUTPUT_LANGUAGES`, but they should not be claimed as
  production-ready until smoke evidence exists for them.
- Gemma/audio-frontends and native speech-to-speech providers should implement
  the same sidecar bridge contract instead of bypassing Hermes' oracle/backend
  loop.
