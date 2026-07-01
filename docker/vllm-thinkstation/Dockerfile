ARG VLLM_IMAGE=vllm/vllm-openai:latest

FROM ${VLLM_IMAGE}

LABEL org.opencontainers.image.title="Hermes ThinkStation vLLM serving"
LABEL org.opencontainers.image.description="OpenAI-compatible vLLM endpoints for Hermes KAME reflex and oracle experiments"

COPY healthcheck.py /opt/hermes-vllm/healthcheck.py
COPY serve-model.sh /opt/hermes-vllm/serve-model.sh

RUN chmod 0755 /opt/hermes-vllm/healthcheck.py /opt/hermes-vllm/serve-model.sh
