ARG VLLM_IMAGE=vllm/vllm-openai:latest

FROM ${VLLM_IMAGE}

LABEL org.opencontainers.image.title="Hermes ThinkStation vLLM serving"
LABEL org.opencontainers.image.description="OpenAI-compatible vLLM endpoints for Hermes KAME reflex and oracle experiments"

RUN VLLM_VERSION="$(python3 -c 'import importlib.metadata as metadata; print(metadata.version("vllm"))')" \
    && python3 -m pip install --no-cache-dir "vllm[audio]==${VLLM_VERSION}"

RUN python3 - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4_mm.py")
text = path.read_text()
old = "            audio_len = processor.feature_extractor.fft_length\n"
new = """            audio_len = getattr(processor.feature_extractor, \"fft_length\", None)
            if audio_len is None:
                audio_len = getattr(processor.feature_extractor, \"audio_samples_per_token\", None)
            if audio_len is None:
                audio_len = getattr(processor.feature_extractor, \"feature_size\", None)
            if audio_len is None:
                raise AttributeError(
                    \"Gemma4 audio feature extractor exposes neither fft_length, \"
                    \"audio_samples_per_token, nor feature_size\"
                )
"""
if old in text:
    path.write_text(text.replace(old, new, 1))
PY

COPY healthcheck.py /opt/hermes-vllm/healthcheck.py
COPY serve-model.sh /opt/hermes-vllm/serve-model.sh

RUN chmod 0755 /opt/hermes-vllm/healthcheck.py /opt/hermes-vllm/serve-model.sh
