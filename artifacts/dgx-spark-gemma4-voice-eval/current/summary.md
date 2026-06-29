# DGX Spark Gemma 4 Voice Eval

Started: 2026-06-29T08:48:45Z
Repo: /Volumes/MacMiniOffload/home/projects/hermes-agent/.codex-worktrees/full-kame-reflex-voice
Artifacts: artifacts/dgx-spark-gemma4-voice-eval/current

## Results
- local repo validation: PASSED
- track 0 full KAME DGX Spark launch pack: PASSED
- track A configured oracle probe: SKIPPED - set DGX_SPARK_ORACLE_BASE_URL
- track B cartesia cloud voice bridge: SKIPPED - set CARTESIA_API_KEY and CARTESIA_VOICE_ID
- track B fallback loopback protocol smoke: PASSED
- track C local DGX speech bridge: SKIPPED - set DGX_SPARK_LOCAL_VOICE_BRIDGE_URL
- track 0 KAME benchmark evidence validation: SKIPPED - set DGX_SPARK_KAME_BENCHMARK_EVIDENCE to a filled benchmark evidence JSON
- DGX Spark KAME recommendation report: PASSED

Finished: 2026-06-29T08:48:49Z

## Key Artifact Paths

- Log: artifacts/dgx-spark-gemma4-voice-eval/current/run.log
- Full KAME stack pack: artifacts/dgx-spark-gemma4-voice-eval/current/kame-stack
- KAME benchmark matrix: artifacts/dgx-spark-gemma4-voice-eval/current/kame-stack/benchmark-matrix.json
- KAME benchmark evidence template: artifacts/dgx-spark-gemma4-voice-eval/current/kame-stack/benchmark-evidence-template.json
- KAME benchmark validator: artifacts/dgx-spark-gemma4-voice-eval/current/kame-stack/validate-benchmark-evidence.sh
- VoiceOps Spark matrix: artifacts/dgx-spark-gemma4-voice-eval/current/voiceops-spark-matrix/spark-model-matrix.json
- VoiceOps Spark matrix markdown: artifacts/dgx-spark-gemma4-voice-eval/current/voiceops-spark-matrix/spark-model-matrix.md
- Oracle probe: artifacts/dgx-spark-gemma4-voice-eval/current/oracle-probe.json
- Cartesia alpha: artifacts/dgx-spark-gemma4-voice-eval/current/cartesia-alpha
- Loopback alpha: artifacts/dgx-spark-gemma4-voice-eval/current/loopback-alpha
- Local speech alpha: artifacts/dgx-spark-gemma4-voice-eval/current/local-speech-alpha
- Recommendation JSON: artifacts/dgx-spark-gemma4-voice-eval/current/recommendation.json
- Recommendation Markdown: artifacts/dgx-spark-gemma4-voice-eval/current/recommendation.md
