# Hermes ThinkStation vLLM Stack

This directory contains an isolated Docker Compose stack for serving the KAME
voice architecture on a ThinkStation PGX-class NVIDIA machine.

It exposes OpenAI-compatible vLLM servers:

- `gemma4-e2b-reflex` on host port `8001`, intended for the low-latency
  audio-aware reflex/interface model.
- `gemma4-12b-oracle` on host port `8002`, intended as the default local
  multimodal Hermes brain for responsive household/business operation.
- `nemotron-nano-oracle` on host port `8003`, intended as a second local
  NVIDIA oracle brain for tool-heavy reasoning, comparison, and routing.
- `nemotron-super-oracle` on host port `8004`, kept as an optional deep local
  model when memory and latency budget allow it.

The stack is intentionally separate from the main Hermes `docker-compose.yml`.
It mounts only model/cache directories, so Python, CUDA, and ML package
experiments stay inside the container image instead of clobbering the host.

## Model Defaults

`gemma4-12b-oracle` defaults to Google's official unified multimodal
checkpoint:

```text
google/gemma-4-12B-it
```

`gemma4-e2b-reflex` defaults to Google's official checkpoint:

```text
google/gemma-4-E2B-it
```

`nemotron-nano-oracle` defaults to NVIDIA's official NVFP4 checkpoint:

```text
nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
```

`nemotron-super-oracle` defaults to NVIDIA's official NVFP4 checkpoint:

```text
nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
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

Then start the default local oracle endpoint:

```bash
docker compose --env-file .env up -d --build gemma4-12b-oracle
docker compose --env-file .env logs -f gemma4-12b-oracle
curl http://127.0.0.1:8002/v1/models
```

Then start the secondary NVIDIA oracle endpoint:

```bash
docker compose --env-file .env up -d --build nemotron-nano-oracle
docker compose --env-file .env logs -f nemotron-nano-oracle
curl http://127.0.0.1:8003/v1/models
```

Nemotron Super is optional and large. Start it only when the default Gemma 12B
+ Nano layout leaves enough memory headroom:

```bash
docker compose --env-file .env up -d --build nemotron-super-oracle
docker compose --env-file .env logs -f nemotron-super-oracle
curl http://127.0.0.1:8004/v1/models
```

The committed default keeps all oracle endpoints at `65536` tokens so Hermes
can satisfy its local 64K-context requirement. Drop the selected oracle's
`*_MAX_MODEL_LEN` only if you also adjust Hermes model context expectations.

## Hermes Wiring

Use these OpenAI-compatible endpoints from Hermes:

```text
Reflex base URL: http://<thinkstation-host>:8001/v1
Reflex model:    gemma-4-e2b-reflex

Primary oracle base URL:   http://<thinkstation-host>:8002/v1
Primary oracle model:      gemma-4-12b-oracle

Secondary oracle base URL: http://<thinkstation-host>:8003/v1
Secondary oracle model:    nemotron-3-nano-oracle

Optional deep base URL:    http://<thinkstation-host>:8004/v1
Optional deep model:       nemotron-3-super-oracle
```

The oracle endpoint is just a serving target. Hermes model selection should
still remain authoritative; `/model` should continue to choose the active
Hermes oracle.

For the hackathon branch, heavy requests should use both local oracle brains.
Configure Hermes with Gemma 12B as the active `/model` target and the
`gemma-nemotron` MoA preset enabled for the model router. In that layout,
Nemotron Nano produces the second-model analysis for long or planning-heavy
turns, and Gemma 12B aggregates it into the normal Hermes response.

## Useful Knobs

- `GEMMA4_E2B_LIMIT_MM_PER_PROMPT`: defaults to `{"audio":1,"image":0,"video":0}`
  so the reflex can accept one audio item per prompt without reserving image or
  video slots.
- `GEMMA4_E2B_MAX_MODEL_LEN`: defaults to `8192` because the reflex should stay
  small and resident beside the oracle. Raise it only if real audio prompts need
  more room.
- `GEMMA4_12B_LIMIT_MM_PER_PROMPT`: defaults to `{"audio":1,"image":1,"video":0}`
  so the primary local brain can accept bounded multimodal inputs without
  reserving video capacity.
- `GEMMA4_12B_MAX_MODEL_LEN`: defaults to `65536` so Hermes can use it as the
  active local model without tripping the 64K context guard.
- `NEMOTRON_NANO_MAX_MODEL_LEN`: defaults to `65536` for the same reason.
- `*_GPU_MEMORY_UTILIZATION`: Gemma 12B defaults to `0.38` and Nemotron Nano to
  `0.28` so both can stay warm beside the reflex on a single-user GB10/PGX
  host. Lower them if vLLM reports allocation failures; raise one only after
  confirming the other services still have headroom.
- `NEMOTRON_NANO_TENSOR_PARALLEL_SIZE` and `NEMOTRON_SUPER_TENSOR_PARALLEL_SIZE`:
  set these to the number of GPUs when you want vLLM to shard an NVIDIA oracle
  across multiple GPUs.
- `NEMOTRON_SUPER_TENSOR_PARALLEL_SIZE`: set this to the number of GPUs when
  you want vLLM to shard Nemotron Super across multiple GPUs.
- `*_EXTRA_ARGS`: append vLLM flags without editing Compose, for example
  scheduler, quantization, KV cache, or tool-calling flags.

The Nemotron Nano and Super defaults include
`--max-num-seqs 4 --enforce-eager --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser nemotron_v3`.
On the GB10/PGX with Gemma resident, the upstream default of 256 concurrent
sequences is too aggressive for single-user voice. Eager mode avoids CUDA graph
capture failures while preserving usable oracle endpoints. Auto tool choice with
the `hermes` parser is required because Hermes sends OpenAI-compatible tool
definitions and `tool_choice=auto` to custom vLLM providers. The `nemotron_v3`
reasoning parser is required so vLLM returns thinking as structured `reasoning`
instead of mixing it into normal assistant `content`.

## Known Serving Limitation

The Dockerfile applies a small vLLM 0.24.0 compatibility patch for Gemma 4
audio dummy profiling. The installed Gemma 4 unified audio feature extractor
exposes `audio_samples_per_token` / `feature_size`, while vLLM's bundled
`gemma4_mm.py` path expects `fft_length`. Without the patch, Gemma 4 12B can
load weights and then crash during KV-cache profiling before `/v1/models`
comes up.

If `--reasoning-parser nemotron_v3` is removed or unsupported by the installed
vLLM image, the Nemotron endpoints can emit reasoning-style text such as
planning phrases or `</think>` markers in normal assistant content. The API is
healthy in that state, but Hermes must filter that content before it reaches
Discord text, TTS, or the Gemma reflex context. Treat a clean `/v1/models`
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
    "model": "gemma-4-12b-oracle",
    "messages": [{"role": "user", "content": "Give one sentence about why local agents matter."}],
    "max_tokens": 64
  }'
```

```bash
curl -s http://127.0.0.1:8003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nemotron-3-nano-oracle",
    "messages": [{"role": "user", "content": "Give one sentence about why local agents matter."}],
    "max_tokens": 64
  }'
```
