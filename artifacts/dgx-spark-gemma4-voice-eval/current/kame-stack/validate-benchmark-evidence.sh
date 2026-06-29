#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ "$#" -ne 1 ]; then
  echo "usage: $0 /path/to/benchmark-evidence.json" >&2
  exit 2
fi

: "${HERMES_REPO_DIR:=/Volumes/MacMiniOffload/home/projects/hermes-agent/.codex-worktrees/full-kame-reflex-voice}"
: "${HERMES_HOME:=/Users/jethac/.hermes}"
: "${HERMES_PYTHON:=python3}"
: "${HERMES_KAME_INTERFACE_PROVIDER:=gemma4}"
: "${HERMES_KAME_INTERFACE_MODEL:=gemma-4-E2B-it}"
: "${HERMES_KAME_INTERFACE_BASE_URL:=http://spark.local:8000/v1}"
: "${HERMES_KAME_INTERFACE_API_KEY_ENV:=HERMES_KAME_INTERFACE_API_KEY}"
: "${HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS:=30.0}"
: "${HERMES_KAME_INTERFACE_TEMPERATURE:=0.2}"
: "${HERMES_KAME_INTERFACE_MAX_OUTPUT_TOKENS:=160}"
: "${HERMES_KAME_INTERFACE_TIMEOUT_SECONDS:=0.8}"
: "${HERMES_KAME_ASR_MODE:=on_escalation}"
: "${HERMES_KAME_MAX_SPOKEN_SENTENCES:=2}"
: "${HERMES_KAME_VOICE_RESPONSE_POLICY:=sentence_cap}"
: "${HERMES_KAME_FALLBACK_POLICY:=legacy_voice}"
: "${HERMES_KAME_ALLOW_LOCAL_GREETINGS:=true}"
: "${HERMES_KAME_ALLOW_LOCAL_CLARIFICATIONS:=true}"
: "${HERMES_KAME_REQUIRE_ORACLE_FOR_TOOLS:=true}"
: "${HERMES_KAME_REQUIRE_ORACLE_FOR_MEMORY:=true}"
: "${HERMES_KAME_REQUIRE_ORACLE_FOR_FILES:=true}"
: "${HERMES_KAME_LOCAL_CONFIDENCE_THRESHOLD:=0.75}"
: "${HERMES_KAME_BARGE_IN_MIN_RMS:=350}"
: "${HERMES_KAME_BARGE_IN_MIN_SPEECH_MS:=120}"
: "${HERMES_KAME_BARGE_IN_STOP_PLAYBACK_DEADLINE_MS:=150}"
: "${HERMES_KAME_METRICS_ENABLED:=true}"
: "${HERMES_KAME_LOG_TURN_SPANS:=true}"
: "${HERMES_KAME_LOG_PROVIDER_SPANS:=true}"
: "${HERMES_KAME_ORACLE_MODEL:=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}"
: "${HERMES_KAME_ORACLE_BASE_URL:=http://spark.local:8001/v1}"
: "${HERMES_KAME_ORACLE_TIMEOUT_SECONDS:=60.0}"
: "${HERMES_VOICE_STREAMING_STT_BASE_URL:=http://spark.local:8767}"
: "${HERMES_DGX_SPARK_ASR_PROVIDER:=streaming_stt}"
: "${HERMES_VOICE_STREAMING_STT_MODEL:=oracle-verbatim-asr}"
: "${HERMES_DGX_SPARK_ASR_MODULE:=hermes_cli.realtime_voice_loopback_bridge}"
: "${HERMES_DGX_SPARK_ASR_ADAPTER:=loopback_smoke_bridge}"
: "${HERMES_VOICE_STREAMING_TTS_BASE_URL:=http://spark.local:8768}"
: "${HERMES_DGX_SPARK_TTS_PROVIDER:=streaming_tts}"
: "${HERMES_VOICE_STREAMING_TTS_MODEL:=local-streaming-tts}"
: "${HERMES_DGX_SPARK_TTS_MODULE:=hermes_cli.realtime_voice_loopback_bridge}"
: "${HERMES_DGX_SPARK_TTS_ADAPTER:=loopback_smoke_bridge}"

cd "$HERMES_REPO_DIR"
"$HERMES_PYTHON" -m hermes_cli.realtime_voice_dgx_spark \
  --output-dir "$SCRIPT_DIR" \
  --repo-dir "$HERMES_REPO_DIR" \
  --hermes-home "$HERMES_HOME" \
  --interface-provider "$HERMES_KAME_INTERFACE_PROVIDER" \
  --interface-base-url "$HERMES_KAME_INTERFACE_BASE_URL" \
  --interface-model "$HERMES_KAME_INTERFACE_MODEL" \
  --interface-api-key-env "$HERMES_KAME_INTERFACE_API_KEY_ENV" \
  --interface-max-audio-seconds "$HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS" \
  --interface-temperature "$HERMES_KAME_INTERFACE_TEMPERATURE" \
  --interface-max-output-tokens "$HERMES_KAME_INTERFACE_MAX_OUTPUT_TOKENS" \
  --interface-timeout-seconds "$HERMES_KAME_INTERFACE_TIMEOUT_SECONDS" \
  --interface-context-tokens 8192 \
  --interface-gpu-memory-utilization 0.18 \
  --oracle-base-url "$HERMES_KAME_ORACLE_BASE_URL" \
  --oracle-model "$HERMES_KAME_ORACLE_MODEL" \
  --oracle-timeout-seconds "$HERMES_KAME_ORACLE_TIMEOUT_SECONDS" \
  --oracle-context-tokens 32768 \
  --oracle-gpu-memory-utilization 0.62 \
  --max-spoken-sentences "$HERMES_KAME_MAX_SPOKEN_SENTENCES" \
  --voice-response-policy "$HERMES_KAME_VOICE_RESPONSE_POLICY" \
  --fallback-policy "$HERMES_KAME_FALLBACK_POLICY" \
  --allow-local-greetings "$HERMES_KAME_ALLOW_LOCAL_GREETINGS" \
  --allow-local-clarifications "$HERMES_KAME_ALLOW_LOCAL_CLARIFICATIONS" \
  --require-oracle-for-tools "$HERMES_KAME_REQUIRE_ORACLE_FOR_TOOLS" \
  --require-oracle-for-memory "$HERMES_KAME_REQUIRE_ORACLE_FOR_MEMORY" \
  --require-oracle-for-files "$HERMES_KAME_REQUIRE_ORACLE_FOR_FILES" \
  --local-confidence-threshold "$HERMES_KAME_LOCAL_CONFIDENCE_THRESHOLD" \
  --barge-in-min-rms "$HERMES_KAME_BARGE_IN_MIN_RMS" \
  --barge-in-min-speech-ms "$HERMES_KAME_BARGE_IN_MIN_SPEECH_MS" \
  --barge-in-stop-playback-deadline-ms "$HERMES_KAME_BARGE_IN_STOP_PLAYBACK_DEADLINE_MS" \
  --metrics-enabled "$HERMES_KAME_METRICS_ENABLED" \
  --log-turn-spans "$HERMES_KAME_LOG_TURN_SPANS" \
  --log-provider-spans "$HERMES_KAME_LOG_PROVIDER_SPANS" \
  --sidecar-base-url http://spark.local:8765 \
  --asr-base-url "$HERMES_VOICE_STREAMING_STT_BASE_URL" \
  --asr-provider "$HERMES_DGX_SPARK_ASR_PROVIDER" \
  --asr-model "$HERMES_VOICE_STREAMING_STT_MODEL" \
  --asr-module "$HERMES_DGX_SPARK_ASR_MODULE" \
  --asr-adapter "$HERMES_DGX_SPARK_ASR_ADAPTER" \
  --tts-base-url "$HERMES_VOICE_STREAMING_TTS_BASE_URL" \
  --tts-provider "$HERMES_DGX_SPARK_TTS_PROVIDER" \
  --tts-model "$HERMES_VOICE_STREAMING_TTS_MODEL" \
  --tts-module "$HERMES_DGX_SPARK_TTS_MODULE" \
  --tts-adapter "$HERMES_DGX_SPARK_TTS_ADAPTER" \
  --asr-mode "$HERMES_KAME_ASR_MODE" \
  --vllm-image vllm/vllm-openai:gemma4-cu130 \
  --hermes-image ghcr.io/astral-sh/uv:python3.12-bookworm-slim \
  --model-cache-dir /Users/jethac/.cache/huggingface \
  --benchmark-evidence "$1"
