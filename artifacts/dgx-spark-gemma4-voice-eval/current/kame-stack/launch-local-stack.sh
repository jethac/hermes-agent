#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

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
: "${HERMES_DGX_SPARK_ASR_PROVIDER:=streaming_stt}"
: "${HERMES_VOICE_STREAMING_STT_BASE_URL:=http://spark.local:8767}"
: "${HERMES_VOICE_STREAMING_STT_MODEL:=oracle-verbatim-asr}"
: "${HERMES_DGX_SPARK_TTS_PROVIDER:=streaming_tts}"
: "${HERMES_VOICE_STREAMING_TTS_BASE_URL:=http://spark.local:8768}"
: "${HERMES_VOICE_STREAMING_TTS_MODEL:=local-streaming-tts}"
export HERMES_REPO_DIR HERMES_HOME

if [ "${HERMES_DGX_SPARK_APPLY_PROFILE:-1}" != "0" ]; then
  (
    cd "$HERMES_REPO_DIR"
    "$HERMES_PYTHON" -m hermes_cli.realtime_voice_profile --preset kame --apply \
      --kame-interface-provider "$HERMES_KAME_INTERFACE_PROVIDER" \
      --kame-reflex-model "$HERMES_KAME_INTERFACE_MODEL" \
      --kame-interface-base-url "$HERMES_KAME_INTERFACE_BASE_URL" \
      --kame-interface-api-key-env "$HERMES_KAME_INTERFACE_API_KEY_ENV" \
      --kame-interface-audio-input native_audio \
      --kame-interface-max-audio-seconds "$HERMES_KAME_INTERFACE_MAX_AUDIO_SECONDS" \
      --kame-interface-temperature "$HERMES_KAME_INTERFACE_TEMPERATURE" \
      --kame-interface-max-output-tokens "$HERMES_KAME_INTERFACE_MAX_OUTPUT_TOKENS" \
      --kame-interface-timeout-seconds "$HERMES_KAME_INTERFACE_TIMEOUT_SECONDS" \
      --kame-asr-mode "$HERMES_KAME_ASR_MODE" \
      --kame-asr-provider "$HERMES_DGX_SPARK_ASR_PROVIDER" \
      --kame-preferred-local-oracle-model "$HERMES_KAME_ORACLE_MODEL" \
      --kame-oracle-base-url "$HERMES_KAME_ORACLE_BASE_URL" \
      --kame-oracle-provider-name "KAME Local Oracle" \
      --kame-oracle-timeout-seconds "$HERMES_KAME_ORACLE_TIMEOUT_SECONDS" \
      --kame-max-spoken-sentences "$HERMES_KAME_MAX_SPOKEN_SENTENCES" \
      --kame-voice-response-policy "$HERMES_KAME_VOICE_RESPONSE_POLICY" \
      --kame-tts-provider "$HERMES_DGX_SPARK_TTS_PROVIDER" \
      --kame-fallback-policy "$HERMES_KAME_FALLBACK_POLICY" \
      --kame-allow-local-greetings "$HERMES_KAME_ALLOW_LOCAL_GREETINGS" \
      --kame-allow-local-clarifications "$HERMES_KAME_ALLOW_LOCAL_CLARIFICATIONS" \
      --kame-require-oracle-for-tools "$HERMES_KAME_REQUIRE_ORACLE_FOR_TOOLS" \
      --kame-require-oracle-for-memory "$HERMES_KAME_REQUIRE_ORACLE_FOR_MEMORY" \
      --kame-require-oracle-for-files "$HERMES_KAME_REQUIRE_ORACLE_FOR_FILES" \
      --kame-local-confidence-threshold "$HERMES_KAME_LOCAL_CONFIDENCE_THRESHOLD" \
      --kame-barge-in-min-rms "$HERMES_KAME_BARGE_IN_MIN_RMS" \
      --kame-barge-in-min-speech-ms "$HERMES_KAME_BARGE_IN_MIN_SPEECH_MS" \
      --kame-barge-in-stop-playback-deadline-ms "$HERMES_KAME_BARGE_IN_STOP_PLAYBACK_DEADLINE_MS" \
      --kame-metrics-enabled "$HERMES_KAME_METRICS_ENABLED" \
      --kame-log-turn-spans "$HERMES_KAME_LOG_TURN_SPANS" \
      --kame-log-provider-spans "$HERMES_KAME_LOG_PROVIDER_SPANS" \
      --streaming-stt-base-url "$HERMES_VOICE_STREAMING_STT_BASE_URL" \
      --streaming-stt-model "$HERMES_VOICE_STREAMING_STT_MODEL" \
      --streaming-tts-base-url "$HERMES_VOICE_STREAMING_TTS_BASE_URL" \
      --streaming-tts-model "$HERMES_VOICE_STREAMING_TTS_MODEL" \
      --sidecar-host spark.local \
      --sidecar-port 8765
  )
fi

cd "$SCRIPT_DIR"
docker compose --env-file .env.example -f compose.yaml up --remove-orphans "$@"

# Readiness check once services are up:
#   ./preflight-local-stack.sh
