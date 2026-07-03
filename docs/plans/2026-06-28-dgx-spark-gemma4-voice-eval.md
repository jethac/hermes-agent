---
title: "DGX Spark KAME realtime voice evaluation plan"
status: planned
date: 2026-06-28
type: plan
target_repo: hermes-agent
---

# DGX Spark KAME Realtime Voice Evaluation Plan

## Summary

Evaluate the best local DGX Spark stack for Hermes realtime voice while keeping
the KAME architecture intact:

- Hermes's selected oracle model is the brain.
- The realtime voice provider is only the low-latency interface layer.
- The interface layer hears, segments turns, handles barge-in, speaks, and
  delegates reasoning/actions to Hermes.

The preferred Spark-local large-model/oracle target is Nemotron 3 Super served
locally on DGX Spark. The preferred KAME split is now: a very fast
Moshi/PersonaPlex-class or smaller model handles reflex/floor-control behavior,
Gemma 4 E2B/E4B/12B handles raw-audio interpretation and multilingual evidence,
and Hermes' active `/model` handles oracle work. Gemma 4 26B-A4B remains a
comparison candidate, not the primary local brain. The voice layer should be
evaluated independently so a weak or experimental voice frontend cannot hide a
strong local oracle, and a good voice frontend cannot hide a weak oracle.

## Evidence Snapshot

Recent DGX Spark reports suggest this ranking for Hermes:

| Area | Best-supported path | Evidence summary | Risk |
| --- | --- | --- | --- |
| Local oracle | Nemotron 3 Super via Hermes `/model` to a local Spark endpoint | Sponsor-aligned serious reasoning target with an explicit Spark-local deployment path; must be benchmarked before one-Spark readiness is claimed. | Serving image/version and memory profile details matter. |
| Oracle comparison | Gemma 4 26B-A4B via vLLM | Reported around 24-40 decode tok/s, with strong prefill and workable memory use on single Spark. | Comparison target, not the primary VoiceOps brain. |
| Cloud voice baseline | Cartesia bridge | We already have a Hermes STT/TTS bridge. Good for proving the voice path while local speech work proceeds. | Cloud dependency; not a local-only answer. |
| Local voice pipeline | Nemotron Speech or Riva-like ASR + Magpie/Riva TTS | Pipecat/Nemotron/Magpie is the only well-instrumented Spark voice pipeline found, around 1.2s server-side voice-to-voice in reported runs. | Need a Hermes-compatible bridge; full Riva setup reports include install pain. |
| Reflex/floor-control S2S | Moshi/PersonaPlex-class models | Useful architecture fit for immediate acknowledgement and rough transcript hypotheses; Spark reports mention choppy/unusable full-duplex audio in some deployments. | Candidate for reflex only after stable audio/noise-gate validation; transcript output is evidence, not truth. |
| Direct speech LLM | Ultravox | No confirmed DGX Spark deployment numbers found. | Watchlist only. |
| TensorRT-LLM | Model support exists, but less practical user evidence than vLLM/SGLang/llama.cpp. | Revisit after vLLM baseline. |

The practical conclusion is to prove Nemotron 3 Super as the Spark-local oracle
first, then compare voice frontends and interpreter variants against that stable
brain. Moshi-class S2S stays on the reflex/interface track when audio quality is
stable. Gemma 4 E2B/E4B/12B stays on the interpreter/evidence track, consuming
raw audio plus any reflex/Moshi transcript hypotheses. Gemma 4 26B-A4B is kept
as a measured comparison oracle.

Evaluation must distinguish low-latency hearing from transcript authority. A
Moshi/S2S transcript can improve Gemma's interpretation because it captures what
the reflex believed it heard, but it must be measured as hypothesis evidence
beside the clipped waveform. Classic ASR is the same class of optional evidence
unless the system has fallen back to a text-only voice mode.

## Headless Runner

The repo-side unattended runner is:

```bash
scripts/dgx_spark_gemma4_voice_eval.sh
```

It writes artifacts under:

```text
artifacts/dgx-spark-gemma4-voice-eval/<timestamp>/
```

The runner never prompts. It uses environment variables and skips tracks whose
external prerequisites are absent. It also uses temporary `HERMES_HOME`
directories inside the artifact directory so it does not mutate the user's real
Hermes config.

Required repo-local validation always runs:

```bash
uv run python -m py_compile \
  hermes_cli/realtime_voice_profile.py \
  hermes_cli/realtime_voice_alpha_evidence.py \
  hermes_cli/realtime_voice_cartesia_bridge.py \
  hermes_cli/web_server.py \
  agent/realtime_voice_cartesia_bridge.py

uv run pytest \
  tests/agent/test_realtime_voice_cartesia_bridge.py \
  tests/hermes_cli/test_realtime_voice_profile.py \
  tests/hermes_cli/test_realtime_voice_alpha_evidence.py \
  tests/hermes_cli/test_realtime_voice_dgx_spark.py \
  tests/hermes_cli/test_web_server.py::TestRealtimeVoiceWebSocket \
  -q
```

## Track 0: Full KAME DGX Spark Launch Pack

Goal: keep the actual one-DGX-Spark launch path generated and preflightable
before evaluating individual oracle or speech components.

Runner behavior:

- Runs `python -m hermes_cli.realtime_voice_dgx_spark`.
- Writes the KAME artifact pack under `kame-stack/` in the evaluation artifact
  directory.
- Generates `compose.yaml`, `.env.example`, `launch-local-stack.sh`,
  `preflight-local-stack.sh`, `benchmark-matrix.json`, and benchmark evidence
  templates.
- Generates a three-tier KAME layout: a Moshi/PersonaPlex-class or smaller
  S2S/timing model as the reflex target, Gemma 4 E2B/E4B/12B as the
  interpreter/evidence target, and Nemotron 3 Super as the preferred local
  oracle provider target unless environment variables override generated
  endpoint/preflight targets. Hermes still selects the active oracle through
  its normal `/model` path.
- Runs endpoint preflight only when `DGX_SPARK_KAME_CHECK=1` is set, so artifact
  generation remains headless before services are online.
- Validates filled benchmark evidence with the generated stack-pack validator
  when `DGX_SPARK_KAME_BENCHMARK_EVIDENCE` points at a JSON evidence file.
- Writes `recommendation.json` and `recommendation.md` after Track A/B/C
  artifacts are collected, ranking Cartesia versus local speech evidence only
  after the oracle probe passes.

Useful variables:

```bash
export DGX_SPARK_REFLEX_BASE_URL=http://spark.local:7999/v1
export DGX_SPARK_REFLEX_MODEL=moshi-reflex-or-small-timing-model
export DGX_SPARK_INTERPRETER_BASE_URL=http://spark.local:8000/v1
export DGX_SPARK_INTERPRETER_MODEL=gemma-4-E2B-it
export DGX_SPARK_INTERPRETER_MAX_AUDIO_SECONDS=30
export DGX_SPARK_ORACLE_BASE_URL=http://spark.local:8001/v1
export DGX_SPARK_ORACLE_PROVIDER_TARGET=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4  # provider target hint, not a Hermes /model override
export DGX_SPARK_SIDECAR_BASE_URL=http://spark.local:8765
export DGX_SPARK_LOCAL_VOICE_BRIDGE_URL=http://spark.local:8767
export DGX_SPARK_LOCAL_TTS_BRIDGE_URL=http://spark.local:8768
export DGX_SPARK_KAME_CHECK=1
export DGX_SPARK_KAME_BENCHMARK_EVIDENCE=/path/to/filled-benchmark-evidence.json
```

Acceptance gates:

| Artifact/check | Target |
| --- | --- |
| KAME launch pack generation | Required |
| Reflex model | Defaults to a Moshi/PersonaPlex-class or smaller timing model when available |
| Interpreter model | Defaults to Gemma 4 E2B native audio |
| Local oracle provider target | Defaults to Nemotron 3 Super; Hermes still selects the active oracle through `/model` |
| Preflight | Required only when `DGX_SPARK_KAME_CHECK=1` |
| Benchmark matrix | Includes reflex timing, Gemma direct-audio interpreter evidence, Moshi/S2S transcript-hypothesis usefulness, and STT fallback comparison |
| Evidence validation | Required when `DGX_SPARK_KAME_BENCHMARK_EVIDENCE` is set |
| Recommendation report | Emits Track A/B/C decision and missing-evidence reasons |

## Track A: Nemotron 3 Super Oracle

Goal: verify that Nemotron 3 Super can serve as Hermes's local Spark brain
before any voice frontend evaluation.

Expected external service:

- OpenAI-compatible chat completions endpoint on the DGX Spark.
- Suggested first serving path: vLLM or TensorRT-LLM with the Spark-compatible
  Nemotron 3 Super recipe.
- Keep the model warm for the full evaluation; do not unload/swap between voice
  runs.

Headless variables:

```bash
export DGX_SPARK_ORACLE_BASE_URL=http://<spark-host>:8000
export DGX_SPARK_ORACLE_PROVIDER_TARGET=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4  # provider target hint, not a Hermes /model override
export DGX_SPARK_ORACLE_API_KEY=optional
export DGX_SPARK_ORACLE_TIMEOUT_SECONDS=120
export DGX_SPARK_ORACLE_MAX_TOKENS=220
```

Runner behavior:

- Runs `python -m hermes_cli.realtime_voice_oracle_probe`.
- Uses `DGX_SPARK_ORACLE_PROVIDER_TARGET` only as the provider probe model id;
  Hermes still uses the active `/model` selection for the oracle. The runner
  still accepts `DGX_SPARK_ORACLE_MODEL` and `DGX_SPARK_KAME_ORACLE_MODEL` as
  compatibility aliases.
- Calls `/v1/chat/completions`, accepting either root or `/v1` base URLs.
- Writes `oracle-probe.json`.
- Records elapsed milliseconds, completion tokens, approximate tokens/sec, and
  a response preview.

Acceptance gates:

| Metric | Target |
| --- | --- |
| Chat completion succeeds | Required |
| First simple oracle response | Under 5s wall time for first warmed request |
| Sustained decode | At least 20 tok/s for normal Hermes responses |
| Tool/oracle behavior | Must not refuse local assistant/tool context |
| Operational mode | Model remains warm through all Track B/C voice tests |

Manual DGX-side setup is intentionally outside the repo runner. The setup should
be captured as a separate Spark host playbook once the first known-good server
command is confirmed.

## Track B: Cartesia Cloud Voice Fallback / Provider Comparison

Goal: determine whether Hermes plus the selected oracle feels good when the
voice layer is a high-quality cloud STT/TTS fallback bridge. This isolates
oracle quality and Discord transport behavior from local audio stack issues,
but it is not the target full KAME control path.

Headless variables:

```bash
export CARTESIA_API_KEY=...
export CARTESIA_VOICE_ID=...
export DGX_SPARK_EVAL_RUNS=3
export DGX_SPARK_BRIDGE_TIMEOUT_SECONDS=30
```

Runner behavior when Cartesia variables exist:

```bash
HERMES_HOME=<artifact-home> \
  uv run python -m hermes_cli.realtime_voice_profile \
  --preset cartesia \
  --apply \
  --generate-bridge-token \
  --force-bridge-token

HERMES_HOME=<artifact-home> \
  uv run python -m hermes_cli.realtime_voice_cartesia_bridge \
  --check \
  --strict \
  --production-en-ja

HERMES_HOME=<artifact-home> \
  uv run python -m hermes_cli.realtime_voice_alpha_evidence \
  --runs 3 \
  --provider cartesia \
  --start-bridge \
  --output-dir <artifacts>/cartesia-alpha \
  --prefix cartesia \
  --overwrite
```

Runner behavior without Cartesia variables:

- Runs a loopback bridge protocol check and focused loopback bridge tests.
- Marks it explicitly as protocol-only smoke, not provider quality evidence.
- This fallback is meant to prove the local Hermes bridge contract is intact
  when no external credentials are available. It is not a substitute for Track B
  or Track C latency/quality evidence.

Acceptance gates:

| Metric | Target |
| --- | --- |
| Bridge prerequisite check | Passes strict mode |
| Alpha evidence runs | 3/3 pass |
| Audio-to-partial transcript | Under 300ms target where provider evidence exposes it |
| Final transcript to first audio | Under 1500ms end-to-end target |
| Barge-in | Stop/cancel under 150ms target |
| Model-facing context | Hermes never claims it cannot hear/speak while in voice mode |

Track B is the fastest path to answering: "Is the selected Hermes oracle,
preferably Nemotron 3 Super on Spark, good enough in a voice session before
local reflex, interpreter, and speech services are ready?"

## Track C: Local DGX KAME Voice Stack

Goal: replace the cloud fallback bridge with a local Spark KAME voice stack
while keeping the same Hermes oracle contract and the three-tier split.

Expected external service:

- A Hermes-compatible local reflex/interpreter/TTS stack already running on the
  DGX Spark, or reachable from this machine. A separate ASR or transcript bridge
  may be present only as optional transcript-hypothesis evidence, diagnostics,
  captions, or text-oracle fallback.
- First implementation candidates:
  - Moshi/PersonaPlex-class S2S or smaller timing/noise-gated model for reflex
    floor control and rough transcript hypotheses.
  - Gemma 4 E2B/E4B/12B for raw-audio interpreter evidence.
  - Nemotron Speech streaming ASR + Magpie/Riva-like TTS.
  - Riva/NVIDIA speech stack if installation is stable enough.
  - Pipecat as a reference latency harness, not as the Hermes brain.
- Hermes profile preset: `nvidia_speech`, which points the local speech lane at
  the Nemotron Speech ASR proxy and Magpie TTS proxy by default when the KAME
  reflex/interpreter path is unavailable.

For full KAME runs, the local speech bridge must not become the reflex control
path merely because it can transcribe. Moshi/S2S transcript output and
streaming ASR output should be recorded as transcript hypotheses. Gemma receives
those hypotheses beside the clipped raw audio and emits corrected evidence for
the Hermes oracle. The reflex still owns immediate acknowledgement and
barge-in/floor-control behavior.

The Track C report should therefore show separate timings for speech end to
reflex acknowledgement, speech end to raw-audio segment ready, raw-audio segment
ready to Gemma evidence, transcript-hypothesis arrival, and oracle job
accepted/started/completed. A passing ASR latency number does not prove KAME
readiness unless the raw-audio interpreter path and reflex acknowledgement path
also pass.

Headless variables:

```bash
export DGX_SPARK_LOCAL_VOICE_BRIDGE_URL=http://<spark-host>:8770
export DGX_SPARK_LOCAL_TTS_BRIDGE_URL=http://<spark-host>:8770
export DGX_SPARK_LOCAL_VOICE_STT_MODEL=nemotron-speech-streaming
export DGX_SPARK_LOCAL_VOICE_TTS_MODEL=magpie-or-riva-tts
export DGX_SPARK_EVAL_RUNS=3
```

Runner behavior:

```bash
HERMES_HOME=<artifact-home> \
  uv run python -m hermes_cli.realtime_voice_profile \
  --preset nvidia_speech \
  --streaming-stt-base-url "$DGX_SPARK_LOCAL_VOICE_BRIDGE_URL" \
  --streaming-tts-base-url "$DGX_SPARK_LOCAL_TTS_BRIDGE_URL" \
  --streaming-stt-model "$DGX_SPARK_LOCAL_VOICE_STT_MODEL" \
  --streaming-tts-model "$DGX_SPARK_LOCAL_VOICE_TTS_MODEL" \
  --apply \
  --generate-bridge-token \
  --force-bridge-token

HERMES_HOME=<artifact-home> \
  uv run python -m hermes_cli.realtime_voice_alpha_evidence \
  --runs 3 \
  --provider local_speech \
  --output-dir <artifacts>/local-speech-alpha \
  --prefix local-speech \
  --overwrite
```

Acceptance gates:

| Metric | Target |
| --- | --- |
| Local reflex health | Reports reflex/floor-control capability, or explicitly falls back |
| Gemma interpreter health | Reports raw-audio interpreter capability |
| Local speech bridge health | Reports streaming STT/TTS only when that optional lane is enabled |
| Alpha evidence runs | 3/3 pass |
| Reflex acknowledgement latency | Under 500ms from speech end |
| Interpreter evidence latency | Under 2000ms from audio segment ready |
| Local ASR partial latency | Under 300ms when optional ASR evidence is enabled |
| Final transcript latency | Under 700ms after speech end when text-oracle fallback is enabled |
| Local TTS first audio | Under 900ms from first assistant text |
| End-to-end spoken turn | Under 1500ms with the selected oracle warm |
| Barge-in | Under 150ms stop/cancel |
| Resource contention | Oracle decode does not collapse while speech bridge is active |

Track C only becomes the preferred path if it beats or gets close to Track B on
reflex feel and reliability while preserving the KAME authority boundary and
local-only operation.

## Evaluation Matrix

| Track | Brain | Voice frontend | Must run headless? | Purpose |
| --- | --- | --- | --- | --- |
| A | Nemotron 3 Super via local Spark endpoint | none | Yes | Prove preferred local oracle viability. |
| B | Nemotron 3 Super via local Spark endpoint | Cartesia bridge | Yes | Establish high-quality fallback/provider comparison. |
| C | Nemotron 3 Super via local Spark endpoint | Local DGX KAME stack | Yes | Test local-only target. |
| D | Gemma 4 26B-A4B via vLLM | best passing voice frontend | Yes | Compare non-NVIDIA local brain quality and latency. |

## Decision Rules

1. If Track A fails, do not evaluate local speech yet. Fix Nemotron 3 Super serving first.
2. If Track A passes and Track B feels good, Nemotron 3 Super is a plausible
   Hermes brain and local speech becomes an optimization.
3. If Track B passes but Track C fails, use Cartesia only as a labeled fallback and
   provider-comparison bridge while continuing local KAME stack development.
4. If Track C passes within 25% of Track B latency, prefer Track C for local DGX
   operation.
5. Do not make Moshi/PersonaPlex or Ultravox the durable transcript or tool
   authority. If a Moshi-class model is evaluated, measure it as a reflex:
   acknowledgement latency, barge-in, noise-gate behavior, and transcript
   hypothesis usefulness beside raw audio.
6. Do not rank a speech stack by transcript speed alone. Rank it by reflex
   immediacy, barge-in correctness, interpreter evidence quality, oracle
   routing correctness, and whether transcript hypotheses remain
   non-authoritative until promotion.

## Follow-up Implementation Work

After the first headless run:

1. If Track C is promising, add a managed local speech bridge launcher and tests
   equivalent to the Cartesia/ElevenLabs/Deepgram bridge tests.
2. Keep Moshi/Ultravox as explicit watchlist items until a confirmed Spark
   deployment shows stable realtime audio, but allow Moshi transcript output to
   feed the Gemma interpreter as labeled hypothesis evidence in non-authoritative
   tests.

## One-command Headless Run

With only repo-local validation and loopback fallback:

```bash
scripts/dgx_spark_gemma4_voice_eval.sh
```

Optional track failures are recorded in `summary.md` without aborting the whole
run. Set `DGX_SPARK_EVAL_STRICT=1` when a nonzero exit is desired for any
optional track failure.

With Nemotron 3 Super and Cartesia:

```bash
export DGX_SPARK_ORACLE_BASE_URL=http://spark.local:8000
export DGX_SPARK_ORACLE_PROVIDER_TARGET=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4  # provider target hint, not a Hermes /model override
export CARTESIA_API_KEY=...
export CARTESIA_VOICE_ID=...
scripts/dgx_spark_gemma4_voice_eval.sh
```

With Nemotron 3 Super, Cartesia, and local DGX speech:

```bash
export DGX_SPARK_ORACLE_BASE_URL=http://spark.local:8000
export DGX_SPARK_ORACLE_PROVIDER_TARGET=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4  # provider target hint, not a Hermes /model override
export CARTESIA_API_KEY=...
export CARTESIA_VOICE_ID=...
export DGX_SPARK_LOCAL_VOICE_BRIDGE_URL=http://spark.local:8770
export DGX_SPARK_LOCAL_VOICE_STT_MODEL=nemotron-speech-streaming
export DGX_SPARK_LOCAL_VOICE_TTS_MODEL=magpie-or-riva-tts
scripts/dgx_spark_gemma4_voice_eval.sh
```
