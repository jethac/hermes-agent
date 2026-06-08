# KAME-Style Realtime Voice Handoff

This document is temporary implementation context for the
`wip/kame-style-realtime-voice` branch. Remove it once the realtime voice system
is running end-to-end and the durable docs cover the same ground.

## Goal

Hermes should support a live spoken conversation loop while preserving the
Hermes oracle/backend/tool layer as the source of agent behavior and data
access.

The target architecture is KAME-inspired, not a direct copy of one provider:

```text
desktop/browser mic
  -> Hermes realtime websocket
  -> realtime voice sidecar
  -> speech frontend/provider bridge
  -> Hermes oracle/backend/tool loop
  -> speech output/provider bridge
  -> desktop/browser playback
```

The speech frontend can be:

- Gemma 4 / audio-capable frontend model hosted remotely
- ElevenLabs realtime STT
- Deepgram realtime STT
- later native speech-to-speech provider

The speech output can be:

- ElevenLabs realtime TTS
- Deepgram realtime TTS
- another streaming TTS provider
- later native speech-to-speech output

The backend oracle should remain whatever Hermes is configured to use. Gemma 4
can act as an audio/frontend model when available, but it should not replace the
Hermes oracle unless the user explicitly configures Hermes that way.

## Current Branch State

Implemented in this branch:

- ElevenLabs realtime STT/TTS bridge module:
  `agent/realtime_voice_elevenlabs_bridge.py`
- CLI entrypoint:
  `hermes_cli/realtime_voice_elevenlabs_bridge.py`
- Provider-specific operator doc:
  `docs/realtime_voice_elevenlabs.md`
- Unit tests:
  `tests/agent/test_realtime_voice_elevenlabs_bridge.py`

The bridge follows the existing Deepgram bridge contract:

- `GET /health`
- `WS /v1/streaming-stt/session`
- `WS /v1/streaming-tts/session`

That means the existing Hermes reference sidecar can use the ElevenLabs bridge
through its generic streaming STT/TTS bridge settings.

## Important Design Decisions

- Keep provider code at the edge.
  Hermes' internal realtime protocol should stay provider-neutral.
- Keep the Hermes oracle in the middle.
  Speech providers should transcribe and speak; Hermes should still decide what
  to say and what tools/data to use.
- Keep desktop and inference split.
  The machine running the UI should not need to be the machine running audio or
  model inference.
- Avoid English-only assumptions in protocol fields.
  The practical production target is English and Japanese first, but language
  metadata should remain BCP-47-ish and provider-neutral.
- Do not claim production readiness from static checks.
  Require live audio evidence before marking this production-ready.

## Provider Notes

### ElevenLabs

ElevenLabs is now the first implemented alternate realtime provider bridge on
this branch.

The intended launch shape:

```powershell
python -m hermes_cli.realtime_voice_elevenlabs_bridge --host 127.0.0.1 --port 8767 --production-en-ja
```

Then configure the Hermes reference sidecar:

```powershell
$env:HERMES_VOICE_STREAMING_STT_BASE_URL = "http://127.0.0.1:8767"
$env:HERMES_VOICE_STREAMING_TTS_BASE_URL = "http://127.0.0.1:8767"
python -m hermes_cli.realtime_voice_sidecar
```

Do not commit API keys or voice IDs. Use environment variables or the normal
Hermes secret configuration path.

### Gemma 4 / Audio Frontend

The Gemma path is not implemented in this branch yet.

Expected shape:

- host Gemma 4 audio-capable inference on a network machine
- expose a Hermes-compatible streaming frontend bridge
- feed transcript/final-turn events into the same Hermes oracle path
- keep output either streaming TTS or native audio depending on model support

Do not tie this to one workstation name or local hardware setup. The design
should read as portable remote inference.

### Native S2S

Native speech-to-speech should be supported as another engine/provider path,
but not required for the first production path. The near-term practical target
is streaming STT plus Hermes oracle plus streaming TTS.

## What Still Needs Doing

1. Commit this handoff doc separately from implementation files.
2. Commit and push the ElevenLabs bridge implementation.
3. Run a live ElevenLabs bridge smoke with a valid API key and voice id:
   - `/health` returns ready
   - known PCM input produces partial and final transcript events
   - known assistant text produces playable audio chunks
   - barge-in clears or resets queued TTS output
4. Wire the evidence runner so provider bridges are not Deepgram-specific:
   - `--provider elevenlabs`
   - provider-specific startup command
   - provider-specific prerequisite check
   - provider-neutral evidence report fields
5. Run an end-to-end Hermes session:
   - browser/desktop mic input
   - Hermes oracle turn
   - provider TTS playback
   - user barge-in
6. Add the Gemma/frontend bridge path:
   - define expected websocket or HTTP streaming contract
   - make it use the same Hermes realtime sidecar events
   - document remote inference config
7. Replace this temporary handoff doc with durable docs:
   - provider matrix
   - setup guide
   - production evidence checklist
   - troubleshooting

## Validation Already Run

Local tests:

```powershell
python -m py_compile agent\realtime_voice_elevenlabs_bridge.py hermes_cli\realtime_voice_elevenlabs_bridge.py
python -m pytest tests\agent\test_realtime_voice_deepgram_bridge.py tests\agent\test_realtime_voice_elevenlabs_bridge.py -q -o addopts=
```

Observed result before this doc was written:

```text
41 passed
```

Live key check:

```text
GET /v1/user: 200
GET /v1/voices: 200
voice_count: 21
```

No API key was intentionally written to the repo. Re-run a secret scan before
committing or pushing.

## Removal Criteria

Delete this file when:

- the ElevenLabs bridge has live evidence,
- the provider-neutral evidence runner exists,
- the durable realtime voice docs explain the architecture and provider matrix,
- the remaining Gemma/native-S2S work is tracked in issues or durable docs.
