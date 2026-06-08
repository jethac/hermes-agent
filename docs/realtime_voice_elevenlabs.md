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

This keeps the Hermes oracle and tool loop in the middle. ElevenLabs handles
the fast speech frontend and speech output, but it does not replace Hermes'
backend model or data access layer.

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

For a minimal live key check, call ElevenLabs' voices endpoint with
`xi-api-key` and confirm a `200` response before running production evidence.
Do not commit the key or write it into repo-local config.

## Current Limits

- STT sends Hermes audio chunks as ElevenLabs `input_audio_chunk` websocket
  messages with manual commits on Hermes end-of-utterance.
- TTS uses ElevenLabs' realtime text-to-speech websocket and requests PCM
  output so Hermes can preserve its existing audio output contract.
- Barge-in resets the ElevenLabs TTS websocket to drop queued provider audio.
- `en,ja` is the production preset. Other languages can be advertised with
  `HERMES_ELEVENLABS_OUTPUT_LANGUAGES`, but they should not be claimed as
  production-ready until smoke evidence exists for them.
