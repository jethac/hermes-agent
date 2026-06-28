#!/usr/bin/env bash
set -euo pipefail

# Headless repo-side evaluator for DGX Spark + Gemma 4 + Hermes realtime voice.
#
# This script does not install or start DGX Spark model servers. It assumes any
# DGX-side oracle or local speech bridge is already reachable by URL and uses
# environment variables to decide which tracks to run.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARTIFACT_DIR="${DGX_SPARK_EVAL_ARTIFACT_DIR:-artifacts/dgx-spark-gemma4-voice-eval/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$ARTIFACT_DIR"

LOG="$ARTIFACT_DIR/run.log"
SUMMARY="$ARTIFACT_DIR/summary.md"
OPTIONAL_FAILURES=0

exec > >(tee -a "$LOG") 2>&1

note() {
  printf '\n== %s ==\n' "$*"
}

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

record_skip() {
  printf -- '- %s: SKIPPED - %s\n' "$1" "$2" >> "$SUMMARY"
}

record_pass() {
  printf -- '- %s: PASSED\n' "$1" >> "$SUMMARY"
}

record_fail() {
  OPTIONAL_FAILURES=1
  printf -- '- %s: FAILED - %s\n' "$1" "$2" >> "$SUMMARY"
}

cat > "$SUMMARY" <<EOF
# DGX Spark Gemma 4 Voice Eval

Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Repo: $ROOT
Artifacts: $ARTIFACT_DIR

## Results
EOF

note "Local repo validation"
run uv run python -m py_compile \
  hermes_cli/realtime_voice_profile.py \
  hermes_cli/realtime_voice_alpha_evidence.py \
  hermes_cli/realtime_voice_dgx_spark.py \
  hermes_cli/realtime_voice_oracle_probe.py \
  hermes_cli/realtime_voice_cartesia_bridge.py \
  hermes_cli/web_server.py \
  agent/realtime_voice_cartesia_bridge.py
run uv run pytest \
  tests/agent/test_realtime_voice_cartesia_bridge.py \
  tests/hermes_cli/test_realtime_voice_profile.py \
  tests/hermes_cli/test_realtime_voice_alpha_evidence.py \
  tests/hermes_cli/test_realtime_voice_dgx_spark.py \
  tests/hermes_cli/test_web_server.py::TestRealtimeVoiceWebSocket \
  -q
record_pass "local repo validation"

note "Track 0: full KAME DGX Spark launch pack"
KAME_STACK_DIR="$ARTIFACT_DIR/kame-stack"
KAME_CHECK_ARGS=()
if [[ "${DGX_SPARK_KAME_CHECK:-0}" == "1" ]]; then
  KAME_CHECK_ARGS+=(--check --timeout "${DGX_SPARK_KAME_CHECK_TIMEOUT_SECONDS:-2}")
fi
if run uv run python -m hermes_cli.realtime_voice_dgx_spark \
  --output-dir "$KAME_STACK_DIR" \
  --repo-dir "$ROOT" \
  --hermes-home "${DGX_SPARK_HERMES_HOME:-$HOME/.hermes}" \
  --interface-base-url "${DGX_SPARK_INTERFACE_BASE_URL:-http://spark.local:8000/v1}" \
  --interface-model "${DGX_SPARK_INTERFACE_MODEL:-gemma-4-E2B-it}" \
  --interface-max-audio-seconds "${DGX_SPARK_INTERFACE_MAX_AUDIO_SECONDS:-30}" \
  --interface-context-tokens "${DGX_SPARK_INTERFACE_CONTEXT_TOKENS:-8192}" \
  --interface-gpu-memory-utilization "${DGX_SPARK_INTERFACE_GPU_MEMORY_UTILIZATION:-0.18}" \
  --oracle-base-url "${DGX_SPARK_ORACLE_BASE_URL:-http://spark.local:8001/v1}" \
  --oracle-model "${DGX_SPARK_ORACLE_MODEL:-gemma-4-26B-A4B-it}" \
  --oracle-context-tokens "${DGX_SPARK_ORACLE_CONTEXT_TOKENS:-32768}" \
  --oracle-gpu-memory-utilization "${DGX_SPARK_ORACLE_GPU_MEMORY_UTILIZATION:-0.62}" \
  --sidecar-base-url "${DGX_SPARK_SIDECAR_BASE_URL:-http://spark.local:8765}" \
  --asr-base-url "${DGX_SPARK_LOCAL_VOICE_BRIDGE_URL:-http://spark.local:8767}" \
  --asr-model "${DGX_SPARK_LOCAL_VOICE_STT_MODEL:-oracle-verbatim-asr}" \
  --asr-module "${DGX_SPARK_ASR_MODULE:-hermes_cli.realtime_voice_loopback_bridge}" \
  --asr-adapter "${DGX_SPARK_ASR_ADAPTER:-loopback_smoke_bridge}" \
  --tts-base-url "${DGX_SPARK_LOCAL_TTS_BRIDGE_URL:-http://spark.local:8768}" \
  --tts-model "${DGX_SPARK_LOCAL_VOICE_TTS_MODEL:-local-streaming-tts}" \
  --tts-module "${DGX_SPARK_TTS_MODULE:-hermes_cli.realtime_voice_loopback_bridge}" \
  --tts-adapter "${DGX_SPARK_TTS_ADAPTER:-loopback_smoke_bridge}" \
  --asr-mode "${DGX_SPARK_ASR_MODE:-on_escalation}" \
  --vllm-image "${DGX_SPARK_VLLM_IMAGE:-vllm/vllm-openai:gemma4-cu130}" \
  --hermes-image "${DGX_SPARK_HERMES_IMAGE:-ghcr.io/astral-sh/uv:python3.12-bookworm-slim}" \
  --model-cache-dir "${DGX_SPARK_MODEL_CACHE_DIR:-${HOME}/.cache/huggingface}" \
  "${KAME_CHECK_ARGS[@]}"
then
  if [[ "${DGX_SPARK_KAME_CHECK:-0}" == "1" ]]; then
    record_pass "track 0 full KAME DGX Spark launch pack and preflight"
  else
    record_pass "track 0 full KAME DGX Spark launch pack"
  fi
else
  record_fail "track 0 full KAME DGX Spark launch pack" "KAME stack artifact generation or preflight failed"
fi

note "Track A: Gemma 4 oracle probe"
if [[ -n "${DGX_SPARK_ORACLE_BASE_URL:-}" && -n "${DGX_SPARK_ORACLE_MODEL:-}" ]]; then
  if run uv run python -m hermes_cli.realtime_voice_oracle_probe \
    --output "$ARTIFACT_DIR/oracle-gemma4-probe.json" \
    --base-url "$DGX_SPARK_ORACLE_BASE_URL" \
    --model "$DGX_SPARK_ORACLE_MODEL"
  then
    record_pass "track A gemma4 oracle probe"
  else
    record_fail "track A gemma4 oracle probe" "OpenAI-compatible oracle probe failed"
  fi
else
  record_skip "track A gemma4 oracle probe" "set DGX_SPARK_ORACLE_BASE_URL and DGX_SPARK_ORACLE_MODEL"
fi

note "Track B: Cartesia cloud voice bridge baseline"
TRACK_B_HOME="$ARTIFACT_DIR/hermes-home-cartesia"
mkdir -p "$TRACK_B_HOME"
if [[ -n "${CARTESIA_API_KEY:-}" && -n "${CARTESIA_VOICE_ID:-}" ]]; then
  if run env HERMES_HOME="$TRACK_B_HOME" uv run python -m hermes_cli.realtime_voice_profile \
    --preset cartesia \
    --apply \
    --generate-bridge-token \
    --force-bridge-token \
    --production-evidence-report "$ARTIFACT_DIR/cartesia-alpha"
  run env HERMES_HOME="$TRACK_B_HOME" uv run python -m hermes_cli.realtime_voice_cartesia_bridge \
    --check \
    --strict \
    --production-en-ja
  run env HERMES_HOME="$TRACK_B_HOME" uv run python -m hermes_cli.realtime_voice_alpha_evidence \
    --runs "${DGX_SPARK_EVAL_RUNS:-3}" \
    --provider cartesia \
    --start-bridge \
    --output-dir "$ARTIFACT_DIR/cartesia-alpha" \
    --prefix cartesia \
    --overwrite \
    --bridge-timeout-seconds "${DGX_SPARK_BRIDGE_TIMEOUT_SECONDS:-30}"
  then
    record_pass "track B cartesia cloud voice bridge"
  else
    record_fail "track B cartesia cloud voice bridge" "Cartesia profile/check/evidence command failed"
  fi
else
  record_skip "track B cartesia cloud voice bridge" "set CARTESIA_API_KEY and CARTESIA_VOICE_ID"
  note "Track B fallback: loopback protocol evidence"
  if run env HERMES_HOME="$TRACK_B_HOME" uv run python -m hermes_cli.realtime_voice_loopback_bridge \
    --check \
    --production-en-ja
  run uv run pytest tests/agent/test_realtime_voice_loopback_bridge.py -q
  then
    record_pass "track B fallback loopback protocol smoke"
  else
    record_fail "track B fallback loopback protocol smoke" "Loopback check or focused pytest failed"
  fi
fi

note "Track C: local DGX speech bridge"
TRACK_C_HOME="$ARTIFACT_DIR/hermes-home-local-speech"
mkdir -p "$TRACK_C_HOME"
if [[ -n "${DGX_SPARK_LOCAL_VOICE_BRIDGE_URL:-}" ]]; then
  LOCAL_STT_MODEL="${DGX_SPARK_LOCAL_VOICE_STT_MODEL:-nemotron-speech-streaming}"
  LOCAL_TTS_MODEL="${DGX_SPARK_LOCAL_VOICE_TTS_MODEL:-magpie-or-riva-tts}"
  if run env HERMES_HOME="$TRACK_C_HOME" uv run python -m hermes_cli.realtime_voice_profile \
    --preset generic \
    --streaming-stt-base-url "$DGX_SPARK_LOCAL_VOICE_BRIDGE_URL" \
    --streaming-tts-base-url "${DGX_SPARK_LOCAL_TTS_BRIDGE_URL:-$DGX_SPARK_LOCAL_VOICE_BRIDGE_URL}" \
    --streaming-stt-model "$LOCAL_STT_MODEL" \
    --streaming-tts-model "$LOCAL_TTS_MODEL" \
    --apply \
    --generate-bridge-token \
    --force-bridge-token \
    --production-evidence-report "$ARTIFACT_DIR/local-speech-alpha"
  run env HERMES_HOME="$TRACK_C_HOME" uv run python -m hermes_cli.realtime_voice_alpha_evidence \
    --runs "${DGX_SPARK_EVAL_RUNS:-3}" \
    --output-dir "$ARTIFACT_DIR/local-speech-alpha" \
    --prefix local-speech \
    --overwrite
  then
    record_pass "track C local DGX speech bridge"
  else
    record_fail "track C local DGX speech bridge" "Local speech profile/evidence command failed"
  fi
else
  record_skip "track C local DGX speech bridge" "set DGX_SPARK_LOCAL_VOICE_BRIDGE_URL"
fi

note "Track 0 evidence validation"
if [[ -n "${DGX_SPARK_KAME_BENCHMARK_EVIDENCE:-}" ]]; then
  if [[ -x "$KAME_STACK_DIR/validate-benchmark-evidence.sh" ]]; then
    if run "$KAME_STACK_DIR/validate-benchmark-evidence.sh" "$DGX_SPARK_KAME_BENCHMARK_EVIDENCE"; then
      record_pass "track 0 KAME benchmark evidence validation"
    else
      record_fail "track 0 KAME benchmark evidence validation" "Benchmark evidence did not satisfy KAME launch-pack gates"
    fi
  else
    record_fail "track 0 KAME benchmark evidence validation" "KAME benchmark validator is missing or not executable"
  fi
else
  record_skip "track 0 KAME benchmark evidence validation" "set DGX_SPARK_KAME_BENCHMARK_EVIDENCE to a filled benchmark evidence JSON"
fi

cat >> "$SUMMARY" <<EOF

Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)

## Key Artifact Paths

- Log: $LOG
- Full KAME stack pack: $KAME_STACK_DIR
- KAME benchmark matrix: $KAME_STACK_DIR/benchmark-matrix.json
- KAME benchmark evidence template: $KAME_STACK_DIR/benchmark-evidence-template.json
- KAME benchmark validator: $KAME_STACK_DIR/validate-benchmark-evidence.sh
- Oracle probe: $ARTIFACT_DIR/oracle-gemma4-probe.json
- Cartesia alpha: $ARTIFACT_DIR/cartesia-alpha
- Loopback alpha: $ARTIFACT_DIR/loopback-alpha
- Local speech alpha: $ARTIFACT_DIR/local-speech-alpha
EOF

note "Summary"
cat "$SUMMARY"

if [[ "${DGX_SPARK_EVAL_STRICT:-0}" == "1" && "$OPTIONAL_FAILURES" != "0" ]]; then
  exit 1
fi
