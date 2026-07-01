# Hermes ThinkStation vLLM Stack

This directory contains an isolated Docker Compose stack for serving the KAME
voice architecture on a ThinkStation PGX-class NVIDIA machine.

It exposes two OpenAI-compatible vLLM servers:

- `gemma4-e2b-reflex` on host port `8001`, intended for the low-latency
  audio-aware reflex/interface model.
- `nemotron-super-oracle` on host port `8002`, intended as a large local model
  endpoint that Hermes can select through its normal model/provider flow.

The stack is intentionally separate from the main Hermes `docker-compose.yml`.
It mounts only model/cache directories, so Python, CUDA, and ML package
experiments stay inside the container image instead of clobbering the host.

## Model Defaults

`nemotron-super-oracle` defaults to NVIDIA's official NVFP4 checkpoint:

```text
nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
```

`gemma4-e2b-reflex` defaults to Google's official checkpoint:

```text
google/gemma-4-E2B-it
```

The community `bg-digitalservices/Gemma-4-E2B-it-NVFP4` checkpoint was tested on
the PGX host with vLLM 0.24.0. vLLM recognized it as ModelOpt NVFP4, but failed
during tied-weight initialization. Keep that checkpoint as an explicit
experiment, not as the bring-up default.

## Requirements

- Docker Engine with the NVIDIA Container Toolkit installed.
- A recent NVIDIA driver with Blackwell FP4/NVFP4 support.
- Hugging Face access to the gated model repositories.
- Enough VRAM/unified memory for the selected model length and tensor parallel
  settings.

## Bring-Up

```bash
cd docker/vllm-thinkstation
cp .env.example .env
$EDITOR .env
docker compose --env-file .env up -d --build gemma4-e2b-reflex
docker compose --env-file .env logs -f gemma4-e2b-reflex
```

After the reflex endpoint is healthy:

```bash
curl http://127.0.0.1:8001/v1/models
```

Then start the oracle endpoint:

```bash
docker compose --env-file .env up -d --build nemotron-super-oracle
docker compose --env-file .env logs -f nemotron-super-oracle
curl http://127.0.0.1:8002/v1/models
```

Nemotron Super is large. The committed default is `65536` tokens with
single-user concurrency for the Hermes voice/oracle path. Drop
`NEMOTRON_SUPER_MAX_MODEL_LEN` to `32768` first if vLLM fails during initial
bring-up or KV cache allocation.

## Hermes Wiring

Use these OpenAI-compatible endpoints from Hermes:

```text
Reflex base URL: http://<thinkstation-host>:8001/v1
Reflex model:    gemma-4-e2b-reflex

Oracle base URL: http://<thinkstation-host>:8002/v1
Oracle model:    nemotron-3-super-oracle
```

The oracle endpoint is just a serving target. Hermes model selection should
still remain authoritative; `/model` should continue to choose the active
Hermes oracle.

## Useful Knobs

- `GEMMA4_E2B_LIMIT_MM_PER_PROMPT`: defaults to `{"audio":1,"image":0,"video":0}`
  so the reflex can accept one audio item per prompt without reserving image or
  video slots.
- `GEMMA4_E2B_MAX_MODEL_LEN`: defaults to `8192` because the reflex should stay
  small and resident beside the oracle. Raise it only if real audio prompts need
  more room.
- `NEMOTRON_SUPER_MAX_MODEL_LEN`: defaults to `65536` for the local oracle so
  Hermes can keep a large active context in the single-user demo. Lower this
  first if vLLM fails during KV cache allocation.
- `*_GPU_MEMORY_UTILIZATION`: the oracle defaults to `0.62` so Gemma E2B can
  stay warm beside it on a single GB10. Lower it if vLLM reports allocation
  failures; raise it only after confirming the other service still has
  headroom.
- `NEMOTRON_SUPER_TENSOR_PARALLEL_SIZE`: set this to the number of GPUs when
  you want vLLM to shard Nemotron Super across multiple GPUs.
- `*_EXTRA_ARGS`: append vLLM flags without editing Compose, for example
  scheduler, quantization, KV cache, or tool-calling flags.

The Nemotron Super default includes
`--max-num-seqs 4 --enforce-eager --enable-auto-tool-choice --tool-call-parser hermes`.
On the GB10/PGX with Gemma resident, vLLM reported only 25 available Mamba cache
blocks after real profiling; the upstream default of 256 concurrent sequences
cannot initialize in that memory split. Eager mode avoids CUDA graph capture
failures while preserving a usable single-user oracle endpoint. Auto tool choice
with the `hermes` parser is required because Hermes sends OpenAI-compatible tool
definitions and `tool_choice=auto` to the oracle provider.

## Known Serving Limitation

The Nemotron Super endpoint can emit reasoning-style text such as planning
phrases or `</think>` markers in normal assistant content when served through
this vLLM path. The API is healthy in that state, but Hermes must filter that
content before it reaches Discord text or TTS. Treat a clean `/v1/models`
response and successful tool-choice probe as serving checks, not as proof that
the model output is demo-ready.

## Smoke Test

```bash
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma-4-e2b-reflex",
    "messages": [{"role": "user", "content": "Say ready in three words."}],
    "max_tokens": 16
  }'
```

```bash
curl -s http://127.0.0.1:8002/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nemotron-3-super-oracle",
    "messages": [{"role": "user", "content": "Give one sentence about why local agents matter."}],
    "max_tokens": 64
  }'
```
