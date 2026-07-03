---
title: "Realtime Voice Implementation Guide"
description: "Implementation guide for the KAME-inspired realtime Hermes voice subsystem"
---

# Realtime Voice Implementation Guide

This guide is the working plan for adding realtime voice to Hermes. It assumes the PRD in [Realtime Voice PRD](./realtime-voice-prd.md).

## Current State

Desktop voice conversation is currently turn-based:

- `apps/desktop/src/app/chat/composer/hooks/use-mic-recorder.ts` records audio into a blob.
- `apps/desktop/src/app/chat/composer/hooks/use-voice-conversation.ts` waits for silence, uploads the blob, submits the transcript, then speaks chunks of streamed text.
- `hermes_cli/web_server.py` exposes `/api/audio/transcribe` and `/api/audio/speak`.
- `agent/transcription_provider.py` and `agent/tts_provider.py` define one-shot STT/TTS provider interfaces, with optional TTS byte streaming.

The new subsystem should not remove any of that. It adds a parallel realtime path.

## First Code Boundary

`agent/realtime_voice.py` defines the stable protocol primitives:

- `RealtimeVoiceSessionConfig`
- `RealtimeVoiceEngineKind`
- `VoiceAudioCodec`
- `VoiceEventType`
- `VoiceEvent`
- `AudioChunk`
- `RealtimeVoiceEngine`

Treat this module as the shared contract between the desktop app, FastAPI websocket endpoint, Hermes session runtime, and optional model sidecars.

## Implemented Surfaces

The realtime voice implementation now has both engine families behind the same protocol:

- `agent/realtime_voice_session.py` owns the session state machine, monotonically increasing client sequence validation, barge-in state, and the durable transcript boundary.
- `agent/realtime_voice_sidecar.py` implements the configured model-sidecar client for local, Gemma/vLLM, and remote STT/TTS frontends.
- `agent/realtime_voice_reference_sidecar.py` implements the reference sidecar server that can run on ordinary machines with configured STT/TTS providers, call a vLLM/Gemma audio endpoint when available, or bridge to a compatible streaming STT service while keeping Hermes' TTS/oracle boundary intact.
- `agent/realtime_voice_text_engine.py` implements the text-oracle path: audio or transcript input, streaming frontend events from a configured sidecar, local STT fallback via Hermes' existing transcription provider chain at utterance boundaries, streaming Hermes oracle deltas, speech planning, and chunked audio output via sidecar TTS or the existing TTS provider chain.
- `agent/realtime_voice_s2s_engine.py` implements the native S2S path as a websocket bridge to a local, remote, or cloud model sidecar. When the sidecar emits final transcript events, Hermes calls the configured oracle model and sends `oracle.hint` events back to the sidecar.
- `hermes_cli/web_server.py` exposes `/api/voice/realtime` behind the same websocket auth and Host/Origin guards as the dashboard chat websocket. For loopback `local`, `reference`, `sidecar`, `gemma`, `gemma4`, `lmstudio`, and `vllm` frontends, it can also supervise the reference sidecar process automatically.
- `apps/desktop/src/app/chat/composer/hooks/use-realtime-voice-session.ts` implements the desktop websocket client, microphone frame capture, simple VAD, playback queue, and barge-in cancellation.

The existing one-shot voice mode remains the fallback. Realtime voice is opt-in via `voice.realtime.enabled`.

Current limits:

- In-core local STT still uses Hermes' existing file-based transcription providers after an utterance boundary. True streaming STT is available through the configured sidecar protocol, not through the local provider chain.
- The native S2S model itself is not shipped in Hermes. Hermes provides the sidecar bridge and oracle hint stream; a local, remote, or cloud sidecar owns model inference.
- Binary audio frames are implemented for the desktop hot path, text-oracle sidecars, and native S2S sidecars. JSON/base64 remains a compatibility protocol for tests and older clients.

## Portability Boundary

The desktop app should only know about Hermes' `/api/voice/realtime` websocket. It should not know whether speech inference is local, a supervised loopback process, a LAN GPU host, or a provider endpoint.

This is a deployment invariant, not an optimization. The desktop must remain portable across laptops, thin clients, and machines without audio-model hardware. Moving Gemma, streaming TTS, or native S2S inference to another host should require only server-side Hermes profile changes and sidecar credentials, not a desktop rebuild or a different browser protocol.

Hermes owns:

- websocket auth and session lifecycle
- microphone/playback event protocol
- oracle calls, tool gates, memory, files, and permissions
- durable transcript commit policy

The voice inference process owns:

- streaming STT or audio understanding
- streaming TTS or native S2S audio generation
- model-specific media dependencies and GPU scheduling

This split is why `sidecar_base_url` remains server-side configuration. The desktop cannot point Hermes at an arbitrary inference host through query params, and public docs/settings should describe sidecars by capability (`sidecar`, `local`, `reference`, `gemma4`, `vllm`, `native_s2s`) rather than by a specific workstation or accelerator name.

### Portable STT/TTS Fallback Bridge

The managed reference sidecar can bridge to a portable streaming STT service through:

- `voice.realtime.streaming_stt_base_url`
- `voice.realtime.streaming_stt_model`
- `voice.realtime.streaming_stt_token_env`
- `voice.realtime.streaming_tts_base_url`
- `voice.realtime.streaming_tts_model`
- `voice.realtime.streaming_tts_token_env`

The downstream service must expose `GET /health` with `{"ok": true, "capabilities": {"streaming_stt": true}}` before the reference sidecar advertises `capabilities.streaming_stt: true`. It must also accept a websocket at `/v1/streaming-stt/session` using the same `session.config`, binary `audio.input.chunk`, `transcript.partial`, `transcript.final`, `barge_in`, and `session.error` event contract as the main Hermes sidecar protocol. Hermes forwards microphone chunks to that bridge, relays sanitized transcript metadata back to the desktop, and still uses the configured Hermes oracle plus TTS path for the assistant response.

If `streaming_tts_base_url` is configured, the same reference sidecar probes the bridge for `capabilities.tts: true` and opens `/v1/streaming-tts/session` for assistant speech. It sends `assistant.text.partial` events with `speak: true`, relays `audio.output.chunk` events back to the desktop, and forwards `barge_in` so the TTS bridge can clear pending audio. This keeps low-latency output portable in the same way as streaming STT.

This bridge is a fallback and provider-comparison path, not the full KAME control path. In full KAME mode, the reflex handles live floor control, Gemma receives clipped raw audio plus labeled transcript hypotheses, and Hermes's active `/model` remains the oracle. Streaming STT output is optional hypothesis evidence unless Hermes is explicitly running a text-oracle or fallback mode.

This is intentionally not a Gemma/vLLM shortcut. A vLLM/Gemma chat-completions audio endpoint can improve utterance interpretation, but it remains a buffered interpreter path until it emits live reflex decisions while audio is still arriving. Streaming STT plus TTS can make a useful text-oracle fallback, but the target production KAME claim requires a reflex, a raw-audio interpreter, and Hermes oracle authority.

Hermes ships a Deepgram-compatible bridge entrypoint for the first provider-backed streaming STT and streaming TTS path:

```bash
python -m pip install "hermes-agent[voice]"
python -m hermes_cli.realtime_voice_profile --preset deepgram --apply --generate-bridge-token
set DEEPGRAM_API_KEY=...
python -m hermes_cli.realtime_voice_deepgram_bridge --check --strict --production-en-ja
python -m hermes_cli.realtime_voice_deepgram_bridge --host 127.0.0.1 --port 8766 --production-en-ja
```

Then configure the Hermes realtime profile so the managed reference sidecar can bridge to it:

```yaml
voice:
  realtime:
    enabled: true
    frontend_provider: reference
    sidecar_host: 127.0.0.1
    sidecar_port: 8765
    streaming_stt_base_url: http://127.0.0.1:8766
    streaming_stt_model: nova-3
    streaming_stt_token_env: HERMES_STREAMING_STT_BRIDGE_TOKEN
    streaming_tts_base_url: http://127.0.0.1:8766
    streaming_tts_model: DEFAULT_TTS_MODEL
    streaming_tts_token_env: HERMES_STREAMING_STT_BRIDGE_TOKEN
    require_live_like: true
```

The same live-like profile can be generated without machine-specific names:

```bash
python -m hermes_cli.realtime_voice_profile --preset deepgram --apply --generate-bridge-token
```

This writes a capability-based `voice.realtime` profile, clears stale direct sidecar URLs, keeps the managed loopback reference sidecar portable, configures both streaming STT and streaming TTS through `http://127.0.0.1:8766`, stores the shared bridge bearer token without printing it, uses `nova-3` plus `aura-2-thalia-en` unless overridden, requires live-like streaming STT/TTS, and points production evidence at `./artifacts/realtime-voice-evidence` by default. Use `--bridge-base-url` when the bridge runs on another host, or use the generic `--streaming-stt-*` and `--streaming-tts-*` flags for a non-Deepgram provider. If you override the bridge token env with `--streaming-stt-token-env`, the Deepgram preset also persists `HERMES_DEEPGRAM_BRIDGE_TOKEN_ENV` so the bridge strict check reads the same token.

For Japanese validation, use a Japanese-capable Deepgram STT language setting and route Japanese TTS to a Japanese-capable model. `--production-en-ja` configures the bridge with Nova-3 `language=multi`, `ja:aura-2-fujin-ja,en:aura-2-thalia-en`, and a check that fails early if STT is locked to English or the configured TTS route metadata cannot satisfy Hermes' production EN/JA evidence gate. Override it with `--language`, `HERMES_DEEPGRAM_LANGUAGE`, `--tts-model-by-language`, or `HERMES_DEEPGRAM_TTS_MODEL_BY_LANGUAGE` when you want different provider behavior. If the bridge health probe cannot verify `streaming_stt: true`, Hermes keeps the profile below live-like status even though utterance STT and TTS may still work.

For local evidence collection, the alpha evidence command can start the configured Deepgram bridge itself:

```bash
python -m hermes_cli.realtime_voice_alpha_evidence \
  --output-dir ./artifacts/realtime-voice-evidence \
  --runs 3 \
  --apply \
  --start-deepgram-bridge
```

`--start-deepgram-bridge` runs the same strict EN/JA prerequisite check as `realtime_voice_deepgram_bridge --check --strict --production-en-ja`, then probes the configured `streaming_stt_base_url` with the saved bridge bearer token and accepts only `/health` responses with `ok: true`. It leaves an already healthy bridge alone. If the bridge is not running and the configured URL is loopback, it starts `hermes_cli.realtime_voice_deepgram_bridge --production-en-ja`, waits for `/health.ok`, then starts the managed reference sidecar and collects the evidence run. For remote or shared inference hosts, start the bridge on that host or pass an explicit bind host with `--deepgram-bridge-host`; Hermes still verifies the configured URL before accepting the run.

## Production-Readiness Ladder

Use this ladder when deciding what a profile may claim. The distinction matters because the same Hermes desktop can be connected to very different inference deployments.

| Tier | Claim | Required capability/evidence |
| --- | --- | --- |
| 0. One-shot fallback | Voice works as turn-based dictation and playback. | Existing `/api/audio/transcribe` and `/api/audio/speak` paths work; realtime preflight can fail without stranding the microphone. |
| 1. Portable realtime shell | The desktop can hold a realtime websocket session. | `/api/voice/realtime/status` is enabled; websocket auth and Host/Origin guards work; microphone frames, session state, barge-in events, playback generations, fallback, and durable transcript boundaries are covered by tests. |
| 2. Text-oracle fallback sidecar | Hermes can converse through STT/audio understanding and streaming TTS when the KAME stack is unavailable or under comparison. | Sidecar `/health` advertises STT or audio-understanding plus `tts: true`; sidecar auth is configured; `hermes doctor --realtime-voice` passes; smoke reports include protocol and TTS evidence. |
| 3. Provider-comparison live-like fallback | A provider bridge is good enough for private alpha fallback or baseline comparisons. | `conversation_quality.live_like` is true, normally from `streaming_stt: true` plus `tts: true`; `voice.realtime.require_live_like: true` passes; EN/JA audio fixture and TTS smoke reports meet latency targets. This does not prove full KAME readiness. |
| 4. KAME interpreter path | Gemma 4 E2B/E4B/12B or another audio-capable interpreter improves speech understanding without replacing Hermes. | The interpreter receives clipped raw audio plus labeled reflex/Moshi/ASR transcript hypotheses; Hermes still calls the configured backend oracle model for tools, memory, files, MCP, approvals, and durable answers. |
| 5. KAME reflex path | A native S2S or smaller realtime model handles floor control and immediate acknowledgement. | Sidecar/session health advertises reflex capability; barge-in, acknowledgement latency, rough transcript hypotheses, and fallback state are measured; transcript hypotheses remain non-authoritative until interpreter/oracle promotion. |
| 6. Full KAME production | The product can be compared to always-on commercial live voice APIs while preserving Hermes authority. | Multi-run latency distributions, barge-in reliability, raw-audio interpreter evidence, transcript-hypothesis provenance, remote sidecar degradation, TTS/provider failure behavior, EN/JA acceptance conversations, security review, and operator docs are all collected and reviewed. |

Tier 2 is useful but not enough to claim full KAME interaction if it is STT-first or utterance-based. Tier 3 is the first live-like fallback/baseline target. Tier 4 can be reached with a remote model host or provider endpoint, but Gemma is the interpreter/evidence lane, not the Hermes authority. Tier 5 is where the system first has a true live reflex. Tier 6 requires the three-tier reflex + interpreter + Hermes `/model` oracle contract with provenance-preserving evidence.

The ladder must stay portable. Documentation, config names, status payloads, doctor checks, and release notes should refer to capabilities such as `local`, `reference`, `sidecar`, `gemma4`, `vllm`, `streaming_stt`, `tts`, and `native_s2s`, not to a particular private workstation, accelerator product, tailnet host, or developer credential.

## Target File Layout

```text
agent/
  realtime_voice.py              # shared protocol and engine ABC
  realtime_voice_session.py      # session state machine, sequence, persistence boundary
  realtime_voice_oracle.py       # Hermes oracle adapter around AIAgent/context/tools
  realtime_voice_planner.py      # early-speech, commit, interruption policy
  realtime_voice_sidecar.py      # Gemma/STT/TTS sidecar websocket client
  realtime_voice_reference_sidecar.py # reference local/provider/vLLM sidecar server
  realtime_voice_text_engine.py  # STT -> oracle -> TTS implementation
  realtime_voice_s2s_engine.py   # native S2S sidecar bridge + oracle stream protocol

hermes_cli/
  web_server.py                  # /api/voice/realtime websocket endpoint
  realtime_voice_sidecar.py      # reference sidecar CLI

apps/desktop/src/app/chat/voice/
  realtime-voice-client.ts       # websocket client
  audio-input-worklet.ts         # frame capture
  use-realtime-voice-session.ts  # websocket client, frame capture, playback, barge-in
```

## Wire Protocol

Use websocket JSON frames for control events, transcript events, captions, metrics, and errors. Microphone chunks and assistant audio chunks use binary websocket frames on the desktop, text-oracle sidecar, and native S2S sidecar hot paths. JSON/base64 `audio.input.chunk` and `audio.output.chunk` remain valid for tests and compatibility clients.

For `audio.output.chunk`, `codec: pcm16` means raw little-endian signed 16-bit PCM with `sample_rate_hz` and `channels` metadata. The desktop wraps those bytes in a WAV container before browser playback. Sidecars that emit already-containerized audio should report `opus`/`webm_opus` plus a clean `mime_type`, or use a future explicit container codec, instead of labeling container bytes as `pcm16`.

Binary audio frame format:

```text
4-byte big-endian JSON header length
UTF-8 JSON VoiceEvent header without payload.data_b64
raw audio bytes
```

For client microphone input, Hermes normalizes binary frames back into `audio.input.chunk` events with `payload.data_b64` before validation and engine dispatch, so engines and sidecars keep the same semantic event contract while the desktop avoids base64 encoding microphone blobs.

For server assistant audio output, Hermes sends `audio.output.chunk` as a binary frame when `payload.data_b64` is present and valid. The desktop parses the JSON header and plays the raw trailing bytes directly. Non-audio events continue to use JSON.

Client events:

```json
{"type":"audio.input.chunk","session_id":"...","sequence":1,"payload":{"codec":"opus","data_b64":"..."}}
{"type":"barge_in","session_id":"...","sequence":2,"payload":{"reason":"user_speech"}}
{"type":"session.closed","session_id":"...","sequence":3,"payload":{"reason":"client_closed"}}
```

Client `audio.input.chunk` events may include a `transcript` string for browser or sidecar experiments that produce text before raw-audio interpretation. Set `end_of_utterance`, `final`, or `is_final` to `false` for partial transcript captions. Only text-oracle and fallback modes should start a Hermes oracle turn directly from a final transcript payload. In full KAME mode, transcript payloads are hypotheses attached to the clipped raw audio and interpreter evidence bundle.

When Hermes forwards microphone chunks to a text-oracle sidecar, it adds a server-owned `input_generation` to each sidecar-bound `audio.input.chunk`. Sidecars should echo that value on `transcript.partial` and `transcript.final` events. Hermes uses it to ignore stale speech-recognition results after barge-in or after a newer utterance has started. Desktop clients do not set or rely on this field.

Server events:

```json
{"type":"session.started","session_id":"...","sequence":1,"payload":{"engine":"text_oracle_tts"}}
{"type":"transcript.partial","session_id":"...","sequence":2,"payload":{"text":"what was that","stability":0.42}}
{"type":"transcript.final","session_id":"...","sequence":3,"payload":{"text":"what was that KAME paper about?"}}
{"type":"assistant.text.partial","session_id":"...","sequence":4,"payload":{"text":"KAME is interesting because ","playback_generation":1}}
{"type":"audio.output.chunk","session_id":"...","sequence":5,"payload":{"codec":"opus","data_b64":"...","playback_generation":1}}
{"type":"assistant.commit","session_id":"...","sequence":6,"payload":{"text":"KAME is interesting because ...","playback_generation":1}}
{"type":"oracle.hint","session_id":"...","sequence":7,"payload":{"text":"Use Hermes memory here","delta":"Use Hermes memory here","final":false,"source":"hermes","playback_generation":1}}
```

Server events may include a `metrics` object in the payload. The session layer annotates events with monotonic timing data such as `session_elapsed_ms`, `audio_to_partial_transcript_ms`, `audio_to_final_transcript_ms`, `eou_to_final_transcript_ms`, `final_transcript_to_first_text_ms`, `final_transcript_to_first_audio_ms`, and `barge_in_ack_ms`. Engines and sidecars should preserve any existing metric fields they provide; the session appends Hermes-observed timings before forwarding the event to the desktop. The desktop hook keeps the latest valid metrics as a realtime session snapshot and the active voice controls surface a compact quality readout against the PRD latency targets.

Hermes also annotates server events with `session_state` when the event implies a backend turn-state transition. Values mirror the session state machine (`listening`, `assistant_pending`, `speaking`, `closing`, `closed`) and are authoritative for desktop status display across text-oracle, remote sidecar, and native S2S engines. Desktop clients may still keep local playback state to avoid switching from speaking to listening while already-buffered audio is playing.

## Backend Endpoint

Add a FastAPI websocket endpoint in `hermes_cli/web_server.py`:

```python
@app.websocket("/api/voice/realtime")
async def realtime_voice_ws(ws: WebSocket) -> None:
    ...
```

Add a FastAPI status endpoint for desktop preflight and operator diagnostics:

```http
GET /api/voice/realtime/status
```

The status endpoint returns `enabled`, `available`, selected engine/codecs, frontend provider/model, native S2S requirement flags, and sanitized sidecar state:

```json
{
  "enabled": true,
  "available": true,
  "unavailable_reason": null,
  "engine": "text_oracle_tts",
  "frontend_provider": "gemma4",
  "language_support": {
    "production_languages": ["en", "ja"],
    "production_scripts": ["Latn", "Jpan"],
    "best_effort_languages": true,
    "sidecar_languages_are_diagnostics": true
  },
  "speech_level_threshold": 0.075,
  "barge_in_min_speech_ms": 120,
  "pre_roll_ms": 300,
  "quality_targets_ms": {
    "audio_to_partial_transcript_ms": 300,
    "final_transcript_to_first_text_ms": 500,
    "final_transcript_to_first_audio_ms": 900,
    "barge_in_ack_ms": 150
  },
  "sidecar": {
    "mode": "managed_loopback",
    "base_url": "http://127.0.0.1:8765",
    "autostart": true,
    "healthy": true,
    "health": {
      "kind": "reference",
      "frontend": {
        "provider": "vllm",
        "model": "google/gemma-4-E4B-it-qat-w4a16-ct",
        "languages": ["en", "ja"],
        "scripts": ["Latn", "Jpan"]
      },
      "capabilities": {
        "utterance_stt": true,
        "streaming_stt": false,
        "tts": true,
        "native_s2s": false,
        "vllm_audio_frontend": true,
        "input_languages": ["en", "ja"],
        "output_languages": ["en", "ja"],
        "scripts": ["Latn", "Jpan"]
      }
    }
  }
}
```

A managed loopback sidecar can be `available: true` while `healthy: false` because the websocket path will autostart it. An externally managed remote sidecar that is unhealthy is `available: false`, so the desktop should keep or return to the one-shot voice fallback. Health probes use the configured `sidecar_token_env` bearer token when present. When `/health` returns metadata, Hermes includes only sanitized `kind`, `frontend`, `capabilities`, and local provider flags; URLs, tokens, credentials, and arbitrary vendor fields are not forwarded.

When `available` is false, `unavailable_reason` is a stable machine-readable reason such as `disabled`, `sidecar_required`, `sidecar_unhealthy`, `sidecar_missing_capabilities`, `sidecar_missing_stt`, `sidecar_missing_tts`, or `sidecar_missing_native_s2s`. The desktop should use this only to choose fallback and diagnostics; it must not infer a specific machine or accelerator from the reason.

When realtime preflight is unavailable, the desktop should keep the voice conversation alive by switching to the one-shot voice fallback and surface `unavailable_reason` through the same fallback/degraded diagnostics used for runtime sidecar failures. This keeps local-only installs and remote-sidecar installs portable while still telling operators whether the problem is disabled config, missing native S2S, missing STT/TTS, or an unhealthy external sidecar.

When a sidecar is reachable, its `/health` response must include a JSON capability payload. Hermes uses `capabilities` for preflight gating: `native_s2s_oracle` requires `native_s2s: true`; `text_oracle_tts` sidecar mode requires either `utterance_stt` or `streaming_stt`, plus `tts: true`, because the sidecar is responsible for both speech understanding and streaming speech output on that fallback path. Full KAME mode gates on reflex/interpreter capabilities instead of requiring classic streaming STT. A healthy HTTP sidecar without capability metadata, or without the capabilities required for the selected mode, is reported as `available: false` with an `unavailable_reason`. Websocket session opens are refused with the same reason before the microphone session is accepted, including managed loopback sidecars after Hermes has autostarted them. Language and script arrays in sidecar health are diagnostics only; Hermes sanitizes and forwards them for operator visibility, but they do not grant tool authority or replace explicit voice/STT/TTS configuration.

`language_support` is Hermes' product-support contract, not a sidecar capability claim. By default, English and Japanese are the production acceptance languages and Latin/Japanese scripts are the production acceptance scripts. `best_effort_languages: true` means other clean language metadata may pass through captions, prompts, diagnostics, and provider auto-detection when the configured STT/frontend/TTS stack can handle it. The desktop should not hide realtime voice for non-target languages solely because they are outside this production list; it may label them best-effort. Operators can override `production_languages`, `production_scripts`, and `best_effort_languages` in `voice.realtime`.

`quality_targets_ms` is the active live-conversation quality contract. The desktop compares observed realtime metrics against these targets for its quality pill, and operators can tune them in `voice.realtime.quality_targets_ms` for slower local-only setups without changing the protocol.

`production_readiness` is stricter than `conversation_quality`. A profile may be `live_like` from sidecar capabilities, but it is not even `evidence_ready` until `voice.realtime.production_evidence_report` points at JSON reports that pass the alpha EN/JA verifier. It is not `production_ready` until `voice.realtime.production_review_report` also points at an evidence-backed launch-review JSON report covering human conversation, noisy-room/headset, remote sidecar, desktop reconnect recovery, provider failure, barge-in, tool policy, accessibility, security, and operator-doc checks. This prevents a healthy streaming sidecar from being marketed as Gemini Live-style production quality without both repeatable speech evidence and human/failure review.

Capture tuning is also server-owned. The desktop uses `speech_level_threshold`, `barge_in_min_speech_ms`, and `pre_roll_ms` from preflight status so microphone sensitivity, interruption confidence, and first-syllable preservation can be tuned per profile without rebuilding or reconfiguring the desktop.

Operator readiness gate:

```bash
hermes doctor --realtime-voice
hermes doctor --realtime-voice-smoke
hermes doctor --realtime-voice-alpha --realtime-voice-report ./voice-smoke-report.json
hermes doctor --realtime-voice-audio-fixture ./fixtures/hello-en.webm --realtime-voice-audio-fixture ./fixtures/hello-ja.webm --realtime-voice-audio-codec webm_opus
hermes doctor --realtime-voice-tts-smoke "Hello from Hermes." --realtime-voice-tts-smoke "こんにちは、Hermesです。"
hermes doctor --realtime-voice-barge-in-smoke "Hello from Hermes."
```

Use this before treating a profile as live-voice ready. The strict gate requires realtime voice to be enabled, preflight-available, live-like according to the same `conversation_quality` payload the desktop uses, and configured with latency targets no looser than the PRD live-conversation targets. It also checks that English and Japanese remain the production acceptance languages, that best-effort language pass-through is enabled unless deliberately disabled, that the configured sidecar is healthy, and that public provider naming stays capability-based rather than tied to a specific workstation or accelerator. Plain `hermes doctor` reports the same section informatively without failing ordinary installs that have not opted into realtime voice.

`--realtime-voice-smoke` implies the strict gate, then opens the configured sidecar websocket, sends a transcript-backed `audio.input.chunk`, and waits for `frontend.state` plus `transcript.final`. This is a protocol smoke, not a microphone/acoustic benchmark: it proves sidecar auth, session startup, event streaming, and basic transcript turn latency without requiring a particular GPU or audio device.

`--realtime-voice-alpha` expands to the documented private-alpha evidence set: protocol smoke, the four required English/Japanese STT audio fixtures, the required English/Japanese full audio-session smokes, the four required English/Japanese TTS phrases, and the required barge-in smoke. Use it with `--realtime-voice-report` for a single release-candidate run, then repeat with separate report filenames until the minimum run count is satisfied.

`--realtime-voice-audio-fixture` sends real audio bytes through the same websocket path and requires `transcript.partial` within `audio_to_partial_transcript_ms`, followed by `transcript.final` before timeout. Repeat the flag with short English and Japanese fixtures for release validation. This still does not prove end-user room acoustics or TTS quality, but it catches broken STT/audio-frontend deployments that a transcript-only protocol smoke cannot.

`--realtime-voice-tts-smoke` sends `assistant.text.partial` with `speak: true` to the configured sidecar and requires the first `audio.output.chunk` within `final_transcript_to_first_audio_ms`. Repeat it with a short English and Japanese phrase when validating a release profile so both TTS provider latency and multilingual voice configuration are covered. Built-in EN/JA alpha phrases carry sanitized `language`, `locale`, and `script` metadata, so provider bridges can exercise language-aware voice/model routing during evidence collection.

`--realtime-voice-barge-in-smoke` sends a spoken assistant chunk, immediately interrupts it with `barge_in`, and requires a `barge_in` acknowledgement within `barge_in_ack_ms`. This is a protocol/interruption smoke, not a full acoustic cancellation benchmark, but it prevents release evidence from omitting the interruption path.

`--realtime-voice-report` writes a JSON array for CI and release gates. The first entry is a sanitized `manifest` row that records the realtime stack context used for the run: engine, frontend provider/model, live-like quality reason, quality targets, language policy, sidecar mode, sidecar health capabilities, bridge capability flags, and current production evidence counters. It does not include sidecar URLs, bearer tokens, env var values, query strings, or arbitrary vendor fields.

For text-oracle and provider-bridge alpha validation, the manifest must show an
available live-like profile, a healthy verified sidecar with either
`native_s2s: true` or `streaming_stt: true` plus `tts: true`, and EN/JA output
routing evidence through `output_languages` or `tts_model_languages`. Manifest
`quality_targets_ms` and per-smoke `target_ms` values must not be looser than
the PRD ceilings: partial transcript <= 300 ms, first text <= 500 ms, first
audio <= 900 ms, and barge-in acknowledgement <= 150 ms. The remaining entries
record neutral smoke `kind` values (`protocol`, `session_turn`,
`audio_fixture`, `audio_session`, `tts`, or `barge_in`), `ok`, event names,
latency fields, byte counts, sanitized error text, and smoke-specific metadata
such as fixture path, codec, recognized `final_text`, phrase text,
language/script metadata, and target milliseconds. Required EN/JA audio
fixtures must match the expected transcript text after punctuation, case,
width, and whitespace normalization; required EN/JA session-turn smokes for
text-oracle modes must prove the Hermes transcript -> oracle text -> TTS path
emits both assistant text and audio with the expected language metadata.
Required EN/JA audio-session smokes for provider bridges must start with real
fixture audio bytes and, in one `RealtimeVoiceSession`, emit
`transcript.partial`, matching `transcript.final`, `assistant.text.partial`,
and `audio.output.chunk`. A sidecar that emits any unrelated `transcript.final`
is not enough.

Full KAME production evidence adds a separate requirement: the report must
prove reflex acknowledgement/floor-control timing, raw-audio interpreter
evidence, preserved transcript-hypothesis provenance, Hermes `/model` oracle
routing, and non-authoritative handling of Moshi/S2S or ASR transcript text.
Classic transcript fixtures can remain fallback/provider evidence, but they do
not replace the raw-audio interpreter gate. The schema is intentionally
language-neutral: English and Japanese are the first production acceptance
fixtures, but additional best-effort language fixtures can use the same report
format without changing Hermes protocol semantics.

After `python -m hermes_cli.realtime_voice_report ./artifacts/realtime-voice-alpha-*.json --alpha --min-runs 3` passes, set `voice.realtime.production_evidence_report` to either a verified report file or a directory containing verified report JSON files for the release profile. Production evidence defaults to `production_evidence_min_runs: 3`, so a single alpha report is useful evidence but not enough to claim production readiness. The reports must also share one realtime stack manifest; mixing native S2S, streaming STT/TTS, providers, frontend models, or sidecar capability profiles in one evidence bundle is rejected because it does not prove one deployable profile is evidence-ready. The verifier and `/api/voice/realtime/status` summarize p50, p95, max, and sample count for transcript partial latency, first-audio latency, and barge-in acknowledgement latency across the configured runs. `/api/voice/realtime/status`, `hermes status`, and strict `hermes doctor --realtime-voice` then surface the same evidence-backed `production_readiness` result. Without this path, a profile can still report `conversation_quality.live_like: true`, but `production_readiness.ready` remains false with `missing_evidence_report`.

When the evidence gate passes, `production_readiness.level` becomes `evidence_ready`, not `production_ready`. To make the final production claim, set `voice.realtime.production_review_report` to a JSON file like:

```bash
python -m hermes_cli.realtime_voice_production_review \
  ./artifacts/realtime-voice-production-review.json \
  --write-template \
  --reviewer "Realtime voice QA" \
  --pass-check desktop_reconnect_recovery \
  --evidence-note "desktop_reconnect_recovery=Killed the bridge during active playback; desktop stopped playback, released mic capture, cleared queued audio, and recovered." \
  --evidence-artifact "desktop_reconnect_recovery=./artifacts/realtime-voice-review/desktop-reconnect.md"
# Repeat --pass-check and evidence flags, or edit the JSON, for every required check before applying.
python -m hermes_cli.realtime_voice_production_review \
  ./artifacts/realtime-voice-production-review.json \
  --apply
```

```json
{
  "kind": "realtime_voice_production_review",
  "reviewer": "qa@example.com",
  "reviewed_at": "2026-06-08T00:00:00Z",
  "checks": {
    "human_en_ja_conversations": true,
    "noisy_room_and_headset_coverage": true,
    "remote_sidecar_latency_drill": true,
    "desktop_reconnect_recovery": true,
    "provider_failure_drill": true,
    "barge_in_reliability": true,
    "tool_call_policy_review": true,
    "accessibility_review": true,
    "security_review": true,
    "operator_docs_review": true
  },
  "evidence": {
    "human_en_ja_conversations": {
      "notes": "Three English and three Japanese human conversations completed against the release profile.",
      "artifacts": ["./artifacts/realtime-voice-review/human-en-ja.md"]
    },
    "desktop_reconnect_recovery": {
      "notes": "Killed the bridge during active playback; desktop stopped playback, released mic capture, cleared queued audio, and recovered.",
      "artifacts": ["./artifacts/realtime-voice-review/desktop-reconnect.md"]
    }
  }
}
```

Every passed production review check must include either non-empty `evidence.<check>.notes` or at least one artifact reference in `evidence.<check>.artifacts`; use repeated `--evidence-note CHECK=TEXT` and `--evidence-artifact CHECK=PATH_OR_URL` flags when writing the template from a script. The JSON example above omits the remaining evidence entries for brevity. Only after both `production_evidence_report` and `production_review_report` pass does `production_readiness.ready` become true and `production_readiness.level` become `production_ready`.

### Private Alpha Evidence Pack

The shortest path to an evidence-backed private alpha is days, not weeks, if the scope stays on the portable sidecar contract and the first production languages remain English and Japanese. Do not gate alpha on a particular workstation, GPU, or native S2S model; gate it on repeatable artifacts from the configured realtime sidecar profile.

Private alpha is not the same claim as Gemini Live-style production quality. Alpha readiness means one configured profile repeatedly proves live-like status, English/Japanese speech understanding, full audio-session flow from real fixture audio through Hermes oracle text to spoken output, English/Japanese spoken output, barge-in plumbing, sanitized metadata, and latency targets through the doctor/report path. That makes the profile `evidence_ready`. Production readiness also needs `production_review_report` to document repeated human conversation sessions, noisy-room and headset coverage, remote-sidecar failure drills, desktop reconnect recovery, provider/TTS outage behavior, transcript correction behavior, tool-call policy review, accessibility review, security review, and clear operator docs for fallback.

Minimum alpha fixture set:

- `fixtures/realtime-voice/en/hello.webm`: short English greeting or question.
- `fixtures/realtime-voice/en/tool-question.webm`: short English utterance that should exercise Hermes oracle context without executing unsafe tools from a partial transcript.
- `fixtures/realtime-voice/ja/hello.webm`: short Japanese greeting or question.
- `fixtures/realtime-voice/ja/tool-question.webm`: short Japanese utterance that should preserve Japanese captions and oracle guidance.
- Optional best-effort fixture in a non-target language, used only to prove clean metadata pass-through and no EN/JA-only rejection.

Minimum TTS phrase set:

- `Hello from Hermes.`
- `Can you hear me clearly?`
- `こんにちは、Hermesです。`
- `音声で会話できますか？`
- Optional best-effort phrase in a non-target language, marked non-blocking unless the configured provider claims production support for that language.

For a private alpha release candidate, collect one JSON report per profile/run. For production readiness, collect at least three passing runs and point `production_evidence_report` at the directory that contains them:

```bash
python -m hermes_cli.realtime_voice_fixture_pack \
  --output-dir ./fixtures/realtime-voice \
  --overwrite
python -m hermes_cli.realtime_voice_alpha_evidence \
  --output-dir ./artifacts/realtime-voice-evidence \
  --runs 3 \
  --apply \
  --start-deepgram-bridge
python -m hermes_cli.realtime_voice_report ./artifacts/realtime-voice-evidence/*.json --alpha --min-runs 3
```

`realtime_voice_fixture_pack` uses Hermes' configured TTS provider to generate the required English/Japanese input utterances and converts them to WebM/Opus with `ffmpeg`. Teams may replace those generated files with hand-recorded fixtures, but the filenames and utterance intent should stay stable so report validation remains comparable across machines and sidecar providers.

The alpha evidence helper preflights the four required audio fixture paths before starting any doctor run or sidecar smoke. The EN/JA hello fixtures are also reused by the full audio-session smoke so the report proves real audio can flow through STT, Hermes session/oracle text, and TTS in one live turn. If a fixture is missing, it fails immediately with the exact path to create, preserving the documented relative fixture identifiers used by report validation.

For a single debug run, the helper above is equivalent to:

```bash
hermes doctor \
  --realtime-voice-alpha \
  --realtime-voice-audio-codec webm_opus \
  --realtime-voice-report ./artifacts/realtime-voice-alpha-001.json
```

CI shape:

- Keep the normal unit tests in the regular test matrix: `tests/agent/test_realtime_voice.py`, `tests/hermes_cli/test_web_server.py::TestRealtimeVoiceWebSocket`, and the realtime desktop hook tests.
- Add a separate, opt-in realtime voice smoke workflow or manual job that starts the configured sidecar, runs the command above, verifies the JSON artifact with `python -m hermes_cli.realtime_voice_report ./artifacts/realtime-voice-alpha.json --alpha`, and uploads `artifacts/realtime-voice-alpha.json`.
- Treat EN/JA fixture, full audio-session, Hermes session-turn, TTS, and barge-in failures, missing `transcript.partial`, missing `transcript.final`, missing `assistant.text.partial`, missing `audio.output.chunk`, missing `barge_in`, post-barge-in `audio.output.chunk` bytes from the interrupted utterance, and target latency misses as release-blocking for private alpha.
- Treat non-target language fixture failures as non-blocking unless they reveal protocol rejection, translation-to-English behavior, metadata leakage, or a crash.
- Archive only latency metrics, event names, byte counts, sanitized errors, fixture identifiers, and configured smoke phrases. Do not archive raw user audio outside explicit opt-in fixtures.

Implementation notes:

- Reuse existing websocket host/origin/auth guards in `web_server.py`.
- Create one `RealtimeVoiceSession` per websocket.
- Validate every client event with `validate_client_event`.
- Never let recoverable model-sidecar failures kill the process; emit `frontend.state` with `status: "fallback"` or `status: "degraded"` and continue through local/provider STT/TTS where possible. Use `session.error` only for unrecoverable session failures.
- After forwarding `session.error`, Hermes closes the realtime websocket with an abnormal close code so the desktop cannot keep recording into a failed session.
- Sanitize exception text before sending websocket events or close reasons to the desktop, including errors after the websocket has already been accepted. Runtime voice events must not expose bearer tokens, URL credentials, query-string secrets, or provider keys.
- On disconnect, call `engine.close()`.
- Never expose sidecar bearer tokens, URL credentials, or query-string secrets through the status endpoint.
- Use the configured sidecar bearer token for both websocket sessions and `/health` probes.
- Bound realtime event queues inside Hermes and sidecars. Under pressure, drop queued audio chunks before control, transcript, error, or close events so memory stays bounded without hiding state changes from the desktop.
- Sanitize reference sidecar STT/TTS/vLLM runtime errors before emitting `session.error`; provider URLs, bearer tokens, and API keys must not cross the websocket boundary.

## Session State Machine

Suggested states:

```text
idle -> starting -> listening -> assistant_pending -> speaking -> listening -> closing -> closed
```

Keep these pieces of state:

- current final user segment
- partial transcript text and stability
- active assistant draft id
- active playback generation
- active sidecar input generation
- committed assistant text
- interrupted assistant text
- pending tool calls
- playback generation id
- last inbound and outbound sequence numbers

When a new final user turn or barge-in advances `playback_generation`, cancelled work from older generations must not emit assistant commits or audio. The text-oracle engine also treats incoming speech or transcript frames during an active answer as an implicit barge-in: it cancels the active oracle/TTS work, emits a `barge_in` acknowledgement, reserves that new `playback_generation` for the incoming turn, and then continues buffering or processing the new speech. The desktop should still send explicit `barge_in` as soon as local VAD detects sustained speech over playback, but backend cancellation must not depend on the client remembering that extra event.

When a sidecar handles microphone input, Hermes also advances an `input_generation` across utterances and barge-in boundaries; generation-aware sidecars echo it with transcript events so Hermes can drop late STT results before they start an obsolete oracle turn. The desktop keeps its own lightweight input generation for queued microphone sends and advances it on local barge-in, so delayed blob encoding or websocket backpressure cannot send stale audio from the interrupted turn after the new pre-roll has started. The Hermes session layer drops stale generated audio, assistant text, assistant commits, and generated final transcripts before forwarding events to the desktop.

The session owns persistence. Engines produce events; the session decides which events become durable Hermes messages.

## Hermes Oracle Adapter

The oracle is not Gemma by definition. It is a Hermes adapter that calls the
active Hermes model selected through normal `/model` and provider
configuration. Realtime voice must not introduce a separate `oracle_model`
selector.

Responsibilities:

- Build a voice-specific prompt wrapper around current transcript state.
- Include normal Hermes system prompt, memory, context, tools, and profile state.
- Enforce tool-call policy for partial vs final transcript.
- Return incremental text guidance to the planner.
- Cancel or supersede in-flight oracle calls on barge-in.

Partial transcript policy:

- Pure answer drafting is allowed.
- Read-only tools may be allowed after a stability threshold.
- Write, shell, browser, messaging, and external side-effect tools require final transcript or confirmation.

## Portable Text Oracle + Streaming TTS Fallback

Pipeline:

```text
receive audio chunk
  -> streaming STT frontend
  -> transcript partial/final events
  -> oracle call on final transcript, optionally on stable partials
  -> planner emits text chunks
  -> streaming TTS emits audio chunks
```

The text-oracle engine starts TTS at stable sentence or phrase boundaries instead of waiting for the full response. Prefer punctuation boundaries, including common non-ASCII sentence and phrase delimiters, and only use whitespace splits as a fallback. This lets the local/provider tier improve first-audio latency without assuming English text or a native S2S model.

This engine remains important for bring-up, machines without audio-model
capacity, provider comparison, and failover. It is not the full KAME path. In
full KAME mode, live floor control belongs to the reflex, Gemma-style
interpreter evidence receives raw audio plus labeled transcript hypotheses, and
STT text is optional hypothesis evidence rather than the normal reflex driver.

Do not make English the hidden default. The speech-understanding sidecar prompt must ask the model to preserve the speaker's language and script unless the user explicitly asks for translation. `transcript.partial`, `transcript.final`, `assistant.text.partial`, and `assistant.commit` payloads may carry optional `language`, `locale`, and `script` metadata, but downstream logic must not require it. The planner and chunker should work for languages without spaces between words and for punctuation such as `。`, `！？`, `؟`, `।`, `、`, `，`, `،`, and `；`.

Quality coverage is tiered, not protocol-limited. English and Japanese are the first production acceptance languages for speech input, assistant captions, barge-in, and spoken output. Other languages are best-effort based on the configured STT, frontend model, TTS voice, or native S2S sidecar; clean metadata such as `ko-KR` or `de-DE` should pass through diagnostics and captions, but Hermes may report degraded frontend state when the configured provider cannot serve that language well.

When sanitized transcript language metadata is available, the text-oracle engine carries it into assistant partial/commit events and the Hermes oracle prompt uses it as non-durable guidance to preserve the user's spoken language and script. The persisted user message remains the transcript text only; URLs, provider-specific fields, and malformed metadata must not enter the prompt or durable transcript.

TTS is also language-sensitive. Prefer provider auto-detection or configured multilingual voices where available. If a configured voice is known to be language-limited, report a degraded `frontend.state` or use a configured fallback voice/provider; do not silently translate assistant output into English to satisfy a voice.

If local/provider TTS fails after assistant text has already been planned, the text-oracle engine should emit `frontend.state` with `status: "degraded"` and continue as a text/caption-only turn instead of raising `session.error`. After the first TTS failure in a turn, skip the remaining queued TTS chunks so a broken provider cannot delay the assistant text commit. The user can keep speaking, and the committed assistant text remains durable. Reserve `session.error` for unrecoverable oracle, websocket, or session failures.

Provider choices should be config-driven:

```yaml
voice:
  realtime:
    enabled: true
    engine: text_oracle_tts
    frontend_provider: gemma4
    frontend_model: google/gemma-4-E4B-it-qat-w4a16-ct
    input_buffer_limit_bytes: 8388608
    input_frame_ms: 100
    silence_timeout_ms: 650
    speech_level_threshold: 0.075
    barge_in_min_speech_ms: 120
    pre_roll_ms: 300
    sidecar_host: 127.0.0.1
    sidecar_port: 8765
    sidecar_connect_timeout_seconds: 10
    vllm_base_url: "http://voice-gpu.local:8000/v1"
    vllm_model: google/gemma-4-E4B-it-qat-w4a16-ct
    tts_provider: edge
    production_languages: ["en", "ja"] # production acceptance targets
    production_scripts: ["Latn", "Jpan"]
    best_effort_languages: true        # allow non-target languages without claiming production quality
    production_evidence_report: ./artifacts/realtime-voice-evidence
    production_evidence_min_runs: 3
    production_review_report: ./artifacts/realtime-voice-production-review.json
    quality_targets_ms:
      audio_to_partial_transcript_ms: 300
      final_transcript_to_first_text_ms: 500
      final_transcript_to_first_audio_ms: 900
      barge_in_ack_ms: 150
    languages: ["en", "ja"]        # optional diagnostics for managed reference sidecars
    scripts: ["Latn", "Jpan"]      # optional diagnostics for managed reference sidecars
```

Do not add new core dependencies for provider-specific engines. Use extras or lazy install paths.

## Reference And Remote Inference Sidecars

Treat every sidecar as a model/media process, not as the Hermes authority. Hermes keeps the session, permissions, memory, and tool execution.

Implemented sidecar websocket:

```text
WS /v1/realtime-text/session
```

Hermes sends a `session.config` frame first, then forwards `audio.input.chunk`, `barge_in`, and assistant text chunks with `{"speak": true}`. Forwarded `audio.input.chunk` events include the active `input_generation`; sidecars should echo it on transcript events so Hermes can reject stale STT output after barge-in or newer input. Forwarded `barge_in` events include the active `playback_generation` so sidecars can cancel or tag stale output work deterministically. The sidecar returns `transcript.partial`, `transcript.final`, `frontend.state`, `audio.output.chunk`, or `session.error` events using the shared wire protocol.

Hermes sanitizes sidecar transcript payloads before forwarding them or starting oracle work. Transcript events may keep `text`, `confidence`, `stability`, `input_generation`, `playback_generation`, and sanitized `language`/`locale`/`script`; provider URLs, secrets, raw metadata blobs, and malformed language tokens are dropped.

Hermes and the reference sidecar use the same binary audio envelope as the desktop hot path for sidecar-facing `audio.input.chunk` and `audio.output.chunk` events: a 4-byte big-endian JSON header length, a UTF-8 `VoiceEvent` header without `payload.data_b64`, then the raw audio bytes. JSON/base64 audio events remain valid for compatibility sidecars and tests, and raw binary output bytes without the envelope are still accepted as legacy Opus output from older sidecars.

Reference and provider sidecars must treat `session.closed` as terminal. Cancel or drain active STT/TTS workers on close with a bounded timeout, and suppress late transcript, audio, error, or degraded-state events from cancelled workers after the terminal event has been queued.

Reference sidecar command:

```bash
python -m hermes_cli.realtime_voice_sidecar --host 127.0.0.1 --port 8765
```

Gemma/vLLM audio frontend:

```bash
python -m hermes_cli.realtime_voice_sidecar \
  --host 127.0.0.1 \
  --port 8765 \
  --vllm-base-url http://voice-gpu.local:8000/v1 \
  --vllm-model google/gemma-4-E4B-it-qat-w4a16-ct \
  --input-languages en,ja \
  --output-languages en,ja \
  --scripts Latn,Jpan
```

For ordinary local/provider sidecars, `--input-languages`, `--output-languages`, and `--scripts` are optional health diagnostics. They can also be set with `HERMES_VOICE_INPUT_LANGUAGES`, `HERMES_VOICE_OUTPUT_LANGUAGES`, `HERMES_VOICE_LANGUAGES`, and `HERMES_VOICE_SCRIPTS`. Hermes sanitizes and forwards these values through `/api/voice/realtime/status`; they do not imply model authority or replace explicit STT/TTS provider configuration.

The vLLM runtime must include audio dependencies. If the server returns `Invalid or unsupported audio file` and logs `Please install vllm[audio] for audio support`, install or bake `av`, `librosa`, `soundfile`, and `soxr` into the vLLM image.

When `HERMES_VOICE_SIDECAR_TOKEN` is set, the reference sidecar requires `Authorization: Bearer ...` on both `GET /health` and `WS /v1/realtime-text/session`. Hermes also accepts a custom `sidecar_token_env`; for managed loopback sidecars it resolves that value and passes it to the child as `HERMES_VOICE_SIDECAR_TOKEN` so the desktop process and the inference process can be split cleanly without changing the sidecar API.

Hermes config for no-special-hardware local mode:

```yaml
voice:
  realtime:
    enabled: true
    engine: text_oracle_tts
    frontend_provider: local
    input_buffer_limit_bytes: 8388608
    input_frame_ms: 100
    silence_timeout_ms: 650
    speech_level_threshold: 0.075
    barge_in_min_speech_ms: 120
    pre_roll_ms: 300
    sidecar_host: 127.0.0.1
    sidecar_port: 8765
    sidecar_autostart: true
```

With `sidecar_autostart: true`, Hermes checks `GET /health` on the loopback sidecar URL before accepting a realtime voice websocket. If the sidecar is absent, Hermes starts:

```bash
python -m hermes_cli.realtime_voice_sidecar --host 127.0.0.1 --port 8765
```

Hermes config for an externally managed remote inference sidecar:

```yaml
voice:
  realtime:
    enabled: true
    engine: text_oracle_tts
    frontend_provider: gemma4
    frontend_model: google/gemma-4-E4B-it-qat-w4a16-ct
    input_buffer_limit_bytes: 8388608
    input_frame_ms: 100
    silence_timeout_ms: 650
    speech_level_threshold: 0.075
    barge_in_min_speech_ms: 120
    pre_roll_ms: 300
    sidecar_base_url: "http://voice-inference.local:8765"
    sidecar_token_env: HERMES_VOICE_SIDECAR_TOKEN
    sidecar_connect_timeout_seconds: 10
    sidecar_autostart: false
```

For `gemma4` or `vllm` frontends, the same supervised sidecar can call a remote vLLM audio endpoint through `vllm_base_url` and `vllm_model`. If `sidecar_base_url` points at a non-loopback host, Hermes treats that as an externally managed inference host and does not spawn a local process. Hermes bounds realtime sidecar websocket startup with `sidecar_connect_timeout_seconds` so an unreachable remote inference host can fall back or fail quickly instead of leaving the desktop waiting with an open microphone path. Hermes also bounds sidecar websocket sends; a sidecar that stops accepting microphone chunks, TTS text, or oracle hints must produce fallback/degraded state or a session error instead of blocking the live receive loop indefinitely. The desktop captures microphone chunks at `input_frame_ms` intervals and closes a user turn after `silence_timeout_ms` of quiet; keep the defaults at 100 ms frames and 650 ms silence for low-latency conversation, and raise them only when a browser, room, or provider cannot keep up. `speech_level_threshold`, `barge_in_min_speech_ms`, and `pre_roll_ms` tune the browser VAD and interruption feel for different microphones and rooms. The in-core local STT fallback and the managed reference sidecar both bound unfinished utterance buffering with `input_buffer_limit_bytes`; when the cap is exceeded, Hermes clears the buffered audio and emits `frontend.state` with `status: "degraded"` and `reason: "input_buffer_limit_exceeded"` instead of storing audio without limit. Deprecated sidecar URL aliases remain accepted only for existing private profiles.

Use capability names for `frontend_provider`, not machine names. Prefer `sidecar`, `reference`, `local`, `gemma4`, `vllm`, or a concrete `frontend_model`; do not encode a workstation or GPU product name into the provider value. `sidecar` is the portable alias for "use the configured/default voice sidecar" and should get the same local loopback defaulting and managed-autostart behavior as `local` or `reference` when no explicit remote `sidecar_base_url` is set.

Suggested sidecar API expansion:

```text
POST /v1/audio/understand
WS   /v1/audio/stream-understand
WS   /v1/tts/stream
WS   /v1/s2s/session
GET  /health
```

Hermes sends audio and transcript state. The sidecar returns transcript/frontend state, local draft hints, or audio chunks. `GET /health` returns liveness plus sanitized capability metadata so Hermes can diagnose local, remote, and provider-backed sidecars without exposing secrets.

Native S2S sidecars should treat `oracle.hint` as a streaming hint channel from the Hermes oracle. Each hint includes accumulated `text`, the latest `delta`, `final`, `source: "hermes"`, and the active `playback_generation`. Sidecars can condition generation on these deltas immediately instead of waiting for the final hint. Hermes cancels the active hint stream and advances `playback_generation` on barge-in. If raw microphone audio reaches Hermes while native S2S output or oracle guidance is still active, Hermes sends a backend-generated `barge_in` to the sidecar before forwarding the new audio frame, so native deployments are not dependent on a perfectly timed desktop control event. If a native sidecar later emits generated transcript, assistant text, commit, or audio events with an older `playback_generation`, Hermes drops them before forwarding to the desktop or starting a new oracle hint.

Oracle hints are guidance, not the native sidecar's liveness boundary. If the Hermes oracle hint stream fails after the S2S session has started, Hermes should emit `frontend.state` with `status: "degraded"` and `reason: "oracle_hint_failed"` instead of `session.error`; the sidecar may continue speech generation from its native context while the desktop surfaces degraded guidance quality.

The native S2S websocket uses the same binary audio envelope for `audio.input.chunk` and `audio.output.chunk` as the text-oracle sidecar websocket. JSON/base64 remains valid for control and compatibility events, and raw binary output without an envelope remains a legacy Opus fallback only.

Security requirements:

- Bind to LAN-private interface only or require auth.
- Support bearer token on health checks and realtime websockets.
- Log model names and latencies, not raw audio by default.
- Make raw audio trace capture opt-in.

## Native S2S Engine

Native S2S is a first-class engine family behind the same session protocol. It does not need to wait for the text-oracle path to be "done"; the shared contract is the requirement.

The native engine owns low-level speaking:

```text
audio in -> S2S model -> audio out
                    ^
                    |
            Hermes oracle hints
```

The Hermes oracle stream should provide:

- likely intent
- canonical facts from tools/memory/files
- answer plan
- correction hints
- wait/stop/speak directives

Do not try to make the S2S model execute Hermes tools directly. It should receive oracle guidance, not bypass the agent runtime.

## Desktop Implementation

Add a realtime client separate from the existing dictation hook.

Responsibilities:

- Capture microphone frames with WebAudio or AudioWorklet.
- Wait for `session.started` before opening the microphone so remote sidecar/model startup does not capture audio before the backend voice engine is ready.
- Bound the desktop wait for `session.started`; if the backend accepts the websocket but never starts the session, close the socket and return to the one-shot voice fallback.
- Keep the analyser and local recorder active while listening so Hermes can send a short pre-roll when speech starts, but do not stream idle chunks to Hermes or a sidecar before accepted speech.
- Send frames over websocket with monotonically increasing sequence numbers.
- Serialize asynchronous audio encoding/sends so browser recorder chunks reach Hermes in capture order.
- Bound browser websocket backpressure by dropping only non-final audio frames when the send buffer is already behind; never drop `end_of_utterance` frames.
- Bound server-to-desktop sends. If a desktop cannot receive assistant audio promptly, drop that audio chunk, emit degraded `frontend.state`, and keep transcript/control events moving. If transcript, state, error, or close events cannot be delivered promptly, close the realtime websocket because the desktop no longer has a coherent live session.
- Snapshot the end-of-utterance flag before async blob encoding so the final recorder chunk cannot lose the user turn boundary.
- Request a final recorder chunk before silence stop and send an empty `end_of_utterance` marker if the browser stops without yielding one.
- Maintain and surface captions from transcript and assistant text events.
- Surface recoverable `frontend.state` fallback/degraded events without ending the voice session.
- Play `audio.output.chunk` through a queue.
- Bound the browser playback queue. If assistant audio arrives faster than the browser can play it, drop the oldest queued chunks for the current generation, surface degraded frontend state, and keep the session live instead of allowing seconds of playback latency to accumulate.
- Treat audio element end, error, and rejected `play()` promises as the same queue-settlement path: continue queued audio when present, otherwise return to listening or idle.
- Cancel playback immediately on local barge-in.
- Require a short sustained-speech window before local barge-in so isolated playback echo frames do not self-interrupt the assistant.
- Keep the microphone stream and analyser alive across assistant playback; stop and recreate only the per-utterance `MediaRecorder`.
- Restart the per-utterance `MediaRecorder` immediately when accepted speech begins after a prior silence stop, including barge-in before the previous assistant turn has committed.
- Track `playback_generation` and drop stale audio, assistant text, assistant commit, and generated final-transcript events from interrupted output.
- Show the latest realtime latency snapshot while a voice session is active.
- Treat `session.error` before `session.started` as realtime unavailable and return to the one-shot voice fallback. Treat `session.error` after `session.started` as an active-session fatal error.
- Treat `session.closed` as terminal. Engines should drain cancelled turn work before emitting it and must not enqueue late interrupted commits or sidecar reader errors after close.
- Fall back to the current MediaRecorder blob loop when realtime mode is unavailable or the websocket closes before `session.started`.

## Testing Plan

Unit tests:

- protocol serialization and validation
- session state transitions
- persistence boundary: partials are not committed
- barge-in cancels active playback generation
- tool policy rejects unsafe partial-transcript tool calls

Integration tests:

- websocket opens and closes cleanly
- fake STT emits partial/final transcript
- fake oracle streams text
- fake TTS streams audio chunks
- fake sidecar receives browser audio and assistant TTS chunks
- disconnect closes engine resources

Manual checks:

- Start desktop app.
- Enable realtime voice feature flag.
- Speak a short question.
- Confirm partial transcript appears before final transcript.
- Confirm first audio starts before full response completes.
- Interrupt playback and verify it stops.
- Kill or restart the realtime sidecar or bridge during active playback and verify the desktop stops playback, releases microphone capture, clears stale queued audio, and reconnects or falls back without recording into the failed session.
- Confirm durable transcript contains only final user text and committed assistant text.

Run focused tests with:

```bash
scripts/run_tests.sh tests/agent/test_realtime_voice.py
python -m pytest tests/hermes_cli/test_web_server.py::TestRealtimeVoiceWebSocket -q
npm run test:ui -- src/app/chat/composer/hooks/use-realtime-voice-session.test.ts
```

## Implementation Order

1. Land protocol primitives and docs.
2. Add fake in-process `RealtimeVoiceEngine` for websocket testing.
3. Add `/api/voice/realtime` endpoint behind `voice.realtime.enabled`. Done.
4. Add desktop websocket client and playback path. Done.
5. Add STT provider adapter. Done for the sidecar streaming path; local fallback still reuses Hermes' existing provider chain at utterance boundaries.
6. Add Hermes oracle adapter. Done.
7. Add TTS adapter. Done by reusing Hermes' existing provider chain.
8. Add barge-in and commit semantics. Done in the session and desktop playback layers.
9. Add reference sidecar server. Done for local/provider STT/TTS and optional vLLM audio frontend.
10. Add managed local sidecar lifecycle. Done for loopback local/reference/Gemma/vLLM frontends; remote inference sidecars remain externally managed.
11. Add remote inference sidecar adapter. Done for text-oracle Gemma/STT/TTS streaming, native S2S websocket bridging, and oracle hints.
12. Add native S2S engine. Done as a sidecar-backed engine; model-sidecar deployment is external to Hermes and remains a first-class production track.

## Review Checklist

- Existing one-shot voice mode still works.
- No raw audio is persisted by default.
- No partial transcript is committed to session history.
- Tool calls from unstable transcript are gated.
- Websocket auth matches existing dashboard boundaries.
- Sidecar credentials are profile-safe.
- Provider dependencies are optional or lazy-installed.
- No English-only assumptions in speech planning, transcript prompts, language metadata, or TTS fallback.
- Desktop reconnect recovery was exercised with real microphone/playback state, including microphone release, playback cancellation, stale queued audio invalidation, and clean return to realtime or one-shot fallback voice.
