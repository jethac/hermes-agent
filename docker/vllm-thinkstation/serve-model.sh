#!/usr/bin/env bash
set -euo pipefail

service="${1:-}"

append_extra_args() {
  local extra_args="${1:-}"
  if [[ -n "${extra_args}" ]]; then
    # Intentional shell-style splitting so .env can append simple vLLM flags.
    # Use Compose file edits for arguments that need embedded spaces.
    # shellcheck disable=SC2206
    EXTRA_ARGS=( ${extra_args} )
  else
    EXTRA_ARGS=()
  fi
}

case "${service}" in
  gemma4-e2b-reflex)
    append_extra_args "${GEMMA4_E2B_EXTRA_ARGS:-}"
    exec vllm serve "${GEMMA4_E2B_MODEL}" \
      --host 0.0.0.0 \
      --port 8000 \
      --served-model-name "${GEMMA4_E2B_SERVED_NAME}" \
      --trust-remote-code \
      --max-model-len "${GEMMA4_E2B_MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GEMMA4_E2B_GPU_MEMORY_UTILIZATION}" \
      --limit-mm-per-prompt "${GEMMA4_E2B_LIMIT_MM_PER_PROMPT}" \
      "${EXTRA_ARGS[@]}"
    ;;
  gemma4-12b-oracle)
    append_extra_args "${GEMMA4_12B_EXTRA_ARGS:-}"
    exec vllm serve "${GEMMA4_12B_MODEL}" \
      --host 0.0.0.0 \
      --port 8000 \
      --served-model-name "${GEMMA4_12B_SERVED_NAME}" \
      --trust-remote-code \
      --max-model-len "${GEMMA4_12B_MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GEMMA4_12B_GPU_MEMORY_UTILIZATION}" \
      --limit-mm-per-prompt "${GEMMA4_12B_LIMIT_MM_PER_PROMPT}" \
      "${EXTRA_ARGS[@]}"
    ;;
  nemotron-nano-oracle)
    append_extra_args "${NEMOTRON_NANO_EXTRA_ARGS:-}"
    exec vllm serve "${NEMOTRON_NANO_MODEL}" \
      --host 0.0.0.0 \
      --port 8000 \
      --served-model-name "${NEMOTRON_NANO_SERVED_NAME}" \
      --trust-remote-code \
      --max-model-len "${NEMOTRON_NANO_MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${NEMOTRON_NANO_GPU_MEMORY_UTILIZATION}" \
      --tensor-parallel-size "${NEMOTRON_NANO_TENSOR_PARALLEL_SIZE}" \
      "${EXTRA_ARGS[@]}"
    ;;
  nemotron-super-oracle)
    append_extra_args "${NEMOTRON_SUPER_EXTRA_ARGS:-}"
    exec vllm serve "${NEMOTRON_SUPER_MODEL}" \
      --host 0.0.0.0 \
      --port 8000 \
      --served-model-name "${NEMOTRON_SUPER_SERVED_NAME}" \
      --trust-remote-code \
      --max-model-len "${NEMOTRON_SUPER_MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${NEMOTRON_SUPER_GPU_MEMORY_UTILIZATION}" \
      --tensor-parallel-size "${NEMOTRON_SUPER_TENSOR_PARALLEL_SIZE}" \
      "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "unknown service '${service}'. expected gemma4-e2b-reflex, gemma4-12b-oracle, nemotron-nano-oracle, or nemotron-super-oracle" >&2
    exit 64
    ;;
esac
