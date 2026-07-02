import { codiconIcon } from '@/components/ui/codicon'
import { Brain, type IconComponent, Lock, MessageCircle, Mic, Monitor, Moon, Palette, Sun, Wrench } from '@/lib/icons'
import type { ThemeMode } from '@/themes/context'

import { defineFieldCopy } from './field-copy'
import type { DesktopConfigSection } from './types'

// Provider group definitions used to fold raw env-var names like
// ``XAI_API_KEY`` into a single "xAI" card with a friendly label, short
// description, and signup URL. Membership is determined by longest
// prefix match (see ``providerGroup`` in helpers.ts) so more specific
// prefixes (``MINIMAX_CN_``) correctly beat their general parents
// (``MINIMAX_``). New providers should be added here so they get their
// own card in Settings → Keys instead of being lumped into "Other".
interface ProviderPrefix {
  prefix: string
  name: string
  /** Optional one-line tagline shown beneath the group name. */
  description?: string
  /** Optional canonical signup/console URL surfaced from the card header. */
  docsUrl?: string
  /** Lower numbers float to the top of the providers list. */
  priority: number
}

export const EMPTY_SELECT_VALUE = '__hermes_empty__'
export const CONTROL_TEXT = 'text-xs'

export const PROVIDER_GROUPS: ProviderPrefix[] = [
  {
    prefix: 'NOUS_',
    name: 'Nous Portal',
    description: 'Hosted Hermes & Nous-trained models',
    docsUrl: 'https://portal.nousresearch.com',
    priority: 0
  },
  {
    prefix: 'OPENROUTER_',
    name: 'OpenRouter',
    description: 'Aggregator for hundreds of frontier models',
    docsUrl: 'https://openrouter.ai/keys',
    priority: 1
  },
  {
    prefix: 'ANTHROPIC_',
    name: 'Anthropic',
    description: 'Claude API access (Sonnet, Opus, Haiku)',
    docsUrl: 'https://console.anthropic.com/settings/keys',
    priority: 2
  },
  {
    prefix: 'XAI_',
    name: 'xAI',
    description: 'Grok models (use OAuth for SuperGrok / Premium+)',
    docsUrl: 'https://console.x.ai/',
    priority: 3
  },
  {
    prefix: 'GOOGLE_',
    name: 'Gemini',
    description: 'Google AI Studio (Gemini 1.5 / 2.0 / 2.5)',
    docsUrl: 'https://aistudio.google.com/app/apikey',
    priority: 4
  },
  { prefix: 'GEMINI_', name: 'Gemini', priority: 4 },
  {
    prefix: 'DEEPSEEK_',
    name: 'DeepSeek',
    description: 'Direct DeepSeek API (V3.x, R1)',
    docsUrl: 'https://platform.deepseek.com/api_keys',
    priority: 5
  },
  {
    prefix: 'DASHSCOPE_',
    name: 'DashScope (Qwen)',
    description: 'Alibaba Cloud DashScope — Qwen and multi-vendor models',
    docsUrl: 'https://modelstudio.console.alibabacloud.com/',
    priority: 6
  },
  { prefix: 'HERMES_QWEN_', name: 'DashScope (Qwen)', priority: 6 },
  {
    prefix: 'GLM_',
    name: 'GLM / Z.AI',
    description: 'Zhipu GLM-4.6 and Z.AI hosted endpoints',
    docsUrl: 'https://z.ai/',
    priority: 7
  },
  { prefix: 'ZAI_', name: 'GLM / Z.AI', priority: 7 },
  { prefix: 'Z_AI_', name: 'GLM / Z.AI', priority: 7 },
  {
    prefix: 'KIMI_',
    name: 'Kimi / Moonshot',
    description: 'Moonshot Kimi K2 / coding endpoints',
    docsUrl: 'https://platform.moonshot.cn/',
    priority: 8
  },
  {
    prefix: 'KIMI_CN_',
    name: 'Kimi (China)',
    description: 'Moonshot China endpoint',
    docsUrl: 'https://platform.moonshot.cn/',
    priority: 9
  },
  {
    prefix: 'MINIMAX_',
    name: 'MiniMax',
    description: 'MiniMax-M2 and Hailuo international endpoints',
    docsUrl: 'https://www.minimax.io/',
    priority: 10
  },
  {
    prefix: 'MINIMAX_CN_',
    name: 'MiniMax (China)',
    description: 'MiniMax mainland China endpoint',
    docsUrl: 'https://www.minimaxi.com/',
    priority: 11
  },
  {
    prefix: 'HF_',
    name: 'Hugging Face',
    description: 'Inference Providers — 20+ open models via router.huggingface.co',
    docsUrl: 'https://huggingface.co/settings/tokens',
    priority: 12
  },
  {
    prefix: 'OPENCODE_ZEN_',
    name: 'OpenCode Zen',
    description: 'Pay-as-you-go access to curated coding models',
    docsUrl: 'https://opencode.ai/auth',
    priority: 13
  },
  {
    prefix: 'OPENCODE_GO_',
    name: 'OpenCode Go',
    description: '$10/month subscription for open coding models',
    docsUrl: 'https://opencode.ai/auth',
    priority: 14
  },
  {
    prefix: 'NVIDIA_',
    name: 'NVIDIA NIM',
    description: 'build.nvidia.com or your own local NIM endpoint',
    docsUrl: 'https://build.nvidia.com/',
    priority: 15
  },
  {
    prefix: 'OLLAMA_',
    name: 'Ollama Cloud',
    description: 'Cloud-hosted open models from ollama.com',
    docsUrl: 'https://ollama.com/settings',
    priority: 16
  },
  {
    prefix: 'LM_',
    name: 'LM Studio',
    description: 'Local LM Studio server (OpenAI-compatible)',
    docsUrl: 'https://lmstudio.ai/docs/local-server',
    priority: 17
  },
  {
    prefix: 'STEPFUN_',
    name: 'StepFun',
    description: 'StepFun Step Plan coding models',
    docsUrl: 'https://platform.stepfun.com/',
    priority: 18
  },
  {
    prefix: 'XIAOMI_',
    name: 'Xiaomi MiMo',
    description: 'MiMo-V2.5 and Xiaomi proprietary models',
    docsUrl: 'https://platform.xiaomimimo.com',
    priority: 19
  },
  {
    prefix: 'ARCEEAI_',
    name: 'Arcee AI',
    description: 'Arcee-hosted small + medium models',
    docsUrl: 'https://chat.arcee.ai/',
    priority: 20
  },
  { prefix: 'ARCEE_', name: 'Arcee AI', priority: 20 },
  {
    prefix: 'GMI_',
    name: 'GMI Cloud',
    description: 'GMI Cloud GPU + model serving',
    docsUrl: 'https://www.gmicloud.ai/',
    priority: 21
  },
  {
    prefix: 'AZURE_FOUNDRY_',
    name: 'Azure Foundry',
    description: 'Azure AI Foundry custom endpoints (OpenAI / Anthropic-compatible)',
    docsUrl: 'https://ai.azure.com/',
    priority: 22
  },
  {
    prefix: 'AWS_',
    name: 'AWS Bedrock',
    description: 'Authenticate via AWS profile + region',
    docsUrl: 'https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html',
    priority: 23
  }
]

export const BUILTIN_PERSONALITIES = [
  'helpful',
  'concise',
  'technical',
  'creative',
  'teacher',
  'kawaii',
  'catgirl',
  'pirate',
  'shakespeare',
  'surfer',
  'noir',
  'uwu',
  'philosopher',
  'hype'
]

// Schema-side select overrides for desktop-relevant enum fields whose
// backend schema only declares a string type.
export const ENUM_OPTIONS: Record<string, string[]> = {
  'agent.image_input_mode': ['auto', 'native', 'text'],
  'approvals.mode': ['manual', 'smart', 'off'],
  'code_execution.mode': ['project', 'strict'],
  'context.engine': ['compressor', 'default', 'custom'],
  'delegation.reasoning_effort': ['', 'minimal', 'low', 'medium', 'high', 'xhigh'],
  'memory.provider': ['', 'builtin', 'hindsight', 'honcho'],
  // Terminal execution backends — kept in sync with the dispatch ladder in
  // tools/terminal_tool.py::_create_environment (local/docker/singularity/
  // modal/daytona/ssh). Remote backends need extra env (image, tokens, host).
  'terminal.backend': ['local', 'docker', 'singularity', 'modal', 'daytona', 'ssh'],
  'stt.elevenlabs.model_id': ['scribe_v2', 'scribe_v1'],
  'stt.local.model': ['tiny', 'base', 'small', 'medium', 'large-v3'],
  // Speech-to-text backends — kept in sync with the stt block in
  // hermes_cli/config.py (local/groq/openai/mistral/elevenlabs).
  'stt.provider': ['local', 'groq', 'openai', 'mistral', 'xai', 'elevenlabs'],
  'tts.openai.voice': ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],
  // Text-to-speech backends — kept in sync with the built-in source of truth
  // (agent/tts_registry.py::_BUILTIN_NAMES / tools/tts_tool.py::
  // BUILTIN_TTS_PROVIDERS). 'xai' is Grok TTS.
  'tts.provider': [
    'edge',
    'elevenlabs',
    'openai',
    'xai',
    'minimax',
    'mistral',
    'gemini',
    'neutts',
    'kittentts',
    'piper'
  ],
  'stt.openai.model': ['whisper-1', 'gpt-4o-mini-transcribe', 'gpt-4o-transcribe'],
  'stt.mistral.model': ['voxtral-mini-latest', 'voxtral-mini-2602'],
  'tts.openai.model': ['gpt-4o-mini-tts', 'tts-1', 'tts-1-hd'],
  'tts.elevenlabs.model_id': ['eleven_multilingual_v2', 'eleven_turbo_v2_5', 'eleven_flash_v2_5'],
  // NeuTTS local inference device.
  'tts.neutts.device': ['cpu', 'cuda', 'mps'],
  'updates.non_interactive_local_changes': ['stash', 'discard'],
  'voice.realtime.engine': ['text_oracle_tts', 'kame_interface_oracle', 'native_s2s_oracle'],
  'voice.realtime.frontend_provider': ['reference', 'gemma4', 'openai_realtime', 'gemini_live'],
  'voice.realtime.interface_audio_input': ['auto', 'native_audio', 'text_fallback'],
  'voice.realtime.asr_mode': ['disabled', 'on_escalation', 'speculative', 'debug', 'fallback'],
  'voice.realtime.asr_provider': [
    'streaming_stt',
    'deepgram',
    'elevenlabs',
    'cartesia',
    'nvidia_speech',
    'local_speech',
    'reference'
  ],
  'voice.realtime.oracle_api_mode': ['chat_completions', 'anthropic_messages', 'codex_responses'],
  'voice.realtime.voice_response_policy': ['sentence_cap', 'brief_summary', 'full'],
  'voice.realtime.tts_provider': [
    'streaming_tts',
    'deepgram',
    'elevenlabs',
    'cartesia',
    'nvidia_speech',
    'local_speech',
    'reference'
  ],
  'voice.realtime.fallback_policy': ['legacy_voice', 'text_only', 'fail_closed'],
  'voice.realtime.input_codec': ['webm_opus', 'opus', 'pcm16'],
  'voice.realtime.output_codec': ['opus', 'webm_opus', 'pcm16'],
  'voice.realtime.openai_realtime_voice': ['marin', 'cedar', 'alloy', 'verse'],
  'voice.realtime.gemini_live_voice': ['Puck', 'Charon', 'Kore', 'Fenrir', 'Aoede']
}

export const FIELD_LABELS: Record<string, string> = defineFieldCopy({
  model: 'Default Model',
  modelContextLength: 'Context Window',
  fallbackProviders: 'Fallback Models',
  toolsets: 'Enabled Toolsets',
  timezone: 'Timezone',
  display: {
    personality: 'Personality',
    showReasoning: 'Reasoning Blocks'
  },
  agent: {
    maxTurns: 'Max Agent Steps',
    imageInputMode: 'Image Attachments',
    apiMaxRetries: 'API Retries',
    serviceTier: 'Service Tier',
    toolUseEnforcement: 'Tool-Use Enforcement'
  },
  terminal: {
    cwd: 'Working Directory',
    backend: 'Execution Backend',
    timeout: 'Command Timeout',
    persistentShell: 'Persistent Shell',
    envPassthrough: 'Environment Passthrough',
    dockerImage: 'Docker Image',
    singularityImage: 'Singularity Image',
    modalImage: 'Modal Image',
    daytonaImage: 'Daytona Image'
  },
  fileReadMaxChars: 'File Read Limit',
  toolOutput: {
    maxBytes: 'Terminal Output Limit',
    maxLines: 'File Page Limit',
    maxLineLength: 'Line Length Limit'
  },
  codeExecution: {
    mode: 'Code Execution Mode'
  },
  approvals: {
    mode: 'Approval Mode',
    timeout: 'Approval Timeout',
    mcpReloadConfirm: 'Confirm MCP Reloads'
  },
  commandAllowlist: 'Command Allowlist',
  security: {
    redactSecrets: 'Redact Secrets',
    allowPrivateUrls: 'Allow Private URLs'
  },
  browser: {
    allowPrivateUrls: 'Browser Private URLs',
    autoLocalForPrivateUrls: 'Local Browser For Private URLs'
  },
  checkpoints: {
    enabled: 'File Checkpoints',
    maxSnapshots: 'Checkpoint Limit'
  },
  voice: {
    recordKey: 'Voice Shortcut',
    maxRecordingSeconds: 'Max Recording Length',
    autoTts: 'Read Responses Aloud',
    realtime: {
      enabled: 'Realtime Voice',
      engine: 'Realtime Engine',
      inputCodec: 'Realtime Input Codec',
      outputCodec: 'Realtime Output Codec',
      inputBufferLimitBytes: 'Realtime Input Buffer Limit',
      inputFrameMs: 'Realtime Frame Duration',
      silenceTimeoutMs: 'Realtime Silence Timeout',
      speechLevelThreshold: 'Speech Start Threshold',
      bargeInMinSpeechMs: 'Barge-In Speech Window',
      bargeInMinRms: 'Barge-In RMS',
      bargeInStopPlaybackDeadlineMs: 'Barge-In Stop Deadline',
      preRollMs: 'Mic Pre-Roll',
      requireLiveLike: 'Require Live-Like Voice',
      productionLanguages: 'Production Voice Languages',
      productionScripts: 'Production Voice Scripts',
      bestEffortLanguages: 'Best-Effort Other Languages',
      qualityTargetsMs: {
        audioToPartialTranscriptMs: 'ASR Target',
        finalTranscriptToFirstTextMs: 'First Text Target',
        finalTranscriptToFirstAudioMs: 'First Audio Target',
        bargeInAckMs: 'Barge-In Target'
      },
      frontendProvider: 'Frontend Provider',
      frontendModel: 'Frontend Model',
      interfaceTemperature: 'Interface Temperature',
      interfaceMaxOutputTokens: 'Interface Token Limit',
      interfaceTimeoutSeconds: 'Interface Timeout',
      interfaceMaxAudioSeconds: 'Interface Audio Limit',
      interfaceAudioInput: 'Interface Audio Input',
      asrMode: 'ASR Mode',
      asrProvider: 'ASR Provider',
      asrModel: 'ASR Model',
      oracleProvider: 'Oracle Provider',
      oracleProviderName: 'Oracle Provider Name',
      preferredLocalOracleModel: 'Preferred Local Oracle',
      oracleBaseUrl: 'Oracle Base URL',
      oracleApiMode: 'Oracle API Mode',
      oracleTimeoutSeconds: 'Oracle Timeout',
      maxSpokenSentences: 'Spoken Sentence Cap',
      voiceResponsePolicy: 'Voice Response Policy',
      ttsProvider: 'Realtime TTS Provider',
      ttsModel: 'Realtime TTS Model',
      ttsVoice: 'Realtime TTS Voice',
      fallbackPolicy: 'Realtime Fallback',
      turnAcknowledgement: {
        enabled: 'Oracle Acknowledgement',
        text: 'Acknowledgement Text'
      },
      routing: {
        allowLocalGreetings: 'Local Greetings',
        allowLocalClarifications: 'Local Clarifications',
        requireOracleForTools: 'Oracle For Tools',
        requireOracleForMemory: 'Oracle For Memory',
        requireOracleForFiles: 'Oracle For Files',
        localConfidenceThreshold: 'Local Confidence Threshold'
      },
      metrics: {
        enabled: 'Realtime Metrics',
        logTurnSpans: 'Log Turn Spans',
        logProviderSpans: 'Log Provider Spans'
      },
      outputEvents: {
        captionAliases: 'Caption Alias Events',
        audioAliases: 'Audio Alias Events'
      },
      sidecarBaseUrl: 'Voice Sidecar URL',
      sidecarHost: 'Managed Sidecar Host',
      sidecarPort: 'Managed Sidecar Port',
      sidecarAutostart: 'Autostart Voice Sidecar',
      sidecarTokenEnv: 'Voice Sidecar Token Env Var',
      sidecarConnectTimeoutSeconds: 'Voice Sidecar Connect Timeout',
      streamingSttBaseUrl: 'Streaming STT Bridge URL',
      streamingSttModel: 'Streaming STT Model',
      streamingSttTokenEnv: 'Streaming STT Token Env Var',
      streamingTtsBaseUrl: 'Streaming TTS Bridge URL',
      streamingTtsModel: 'Streaming TTS Model',
      streamingTtsTokenEnv: 'Streaming TTS Token Env Var',
      openaiRealtimeApiKeyEnv: 'OpenAI Realtime Key Env Var',
      openaiRealtimeBaseUrl: 'OpenAI Realtime URL',
      openaiRealtimeVoice: 'OpenAI Realtime Voice',
      openaiRealtimeTranscriptionModel: 'OpenAI Realtime Transcription',
      geminiLiveApiKeyEnv: 'Gemini Live Key Env Var',
      geminiLiveBaseUrl: 'Gemini Live URL',
      geminiLiveVoice: 'Gemini Live Voice',
      geminiLiveGoogleSearch: 'Gemini Live Google Search',
      geminiLiveOracleTool: 'Gemini Oracle Tool',
      sparkBaseUrl: 'Voice Sidecar URL',
      sparkTokenEnv: 'Voice Sidecar Token Env Var'
    }
  },
  stt: {
    enabled: 'Speech To Text',
    provider: 'Speech-To-Text Provider',
    local: {
      model: 'Local Transcription Model',
      language: 'Transcription Language'
    },
    openai: {
      model: 'OpenAI STT Model'
    },
    groq: {
      model: 'Groq STT Model'
    },
    mistral: {
      model: 'Mistral STT Model'
    },
    elevenlabs: {
      modelId: 'ElevenLabs STT Model',
      languageCode: 'ElevenLabs Language',
      tagAudioEvents: 'Tag Audio Events',
      diarize: 'Speaker Diarization'
    }
  },
  tts: {
    provider: 'Text-To-Speech Provider',
    edge: {
      voice: 'Edge Voice'
    },
    openai: {
      model: 'OpenAI TTS Model',
      voice: 'OpenAI Voice'
    },
    elevenlabs: {
      voiceId: 'ElevenLabs Voice',
      modelId: 'ElevenLabs Model'
    },
    xai: {
      voiceId: 'xAI (Grok) Voice',
      language: 'xAI Language'
    },
    minimax: {
      model: 'MiniMax TTS Model',
      voiceId: 'MiniMax Voice'
    },
    mistral: {
      model: 'Mistral TTS Model',
      voiceId: 'Mistral Voice'
    },
    gemini: {
      model: 'Gemini TTS Model',
      voice: 'Gemini Voice'
    },
    neutts: {
      model: 'NeuTTS Model',
      device: 'NeuTTS Device'
    },
    kittentts: {
      model: 'KittenTTS Model',
      voice: 'KittenTTS Voice'
    },
    piper: {
      voice: 'Piper Voice'
    }
  },
  memory: {
    memoryEnabled: 'Persistent Memory',
    userProfileEnabled: 'User Profile',
    memoryCharLimit: 'Memory Budget',
    userCharLimit: 'Profile Budget',
    provider: 'Memory Provider'
  },
  context: {
    engine: 'Context Engine'
  },
  compression: {
    enabled: 'Auto-Compression',
    threshold: 'Compression Threshold',
    targetRatio: 'Compression Target',
    protectLastN: 'Protected Recent Messages'
  },
  delegation: {
    model: 'Subagent Model',
    provider: 'Subagent Provider',
    maxIterations: 'Subagent Turn Limit',
    maxConcurrentChildren: 'Parallel Subagents',
    childTimeoutSeconds: 'Subagent Timeout',
    reasoningEffort: 'Subagent Reasoning Effort'
  },
  updates: {
    nonInteractiveLocalChanges: 'In-App Update Local Changes'
  }
})

export const FIELD_DESCRIPTIONS: Record<string, string> = defineFieldCopy({
  model: 'Used for new chats unless you pick a different model in the composer.',
  modelContextLength: "Leave at 0 to use the selected model's detected context window.",
  fallbackProviders: 'Backup provider:model entries to try if the default model fails.',
  display: {
    personality: 'Default assistant style for new sessions.',
    showReasoning: 'Show reasoning sections when the backend provides them.'
  },
  timezone: 'Used when Hermes needs local time context. Blank uses the system timezone.',
  agent: {
    imageInputMode: 'Controls how image attachments are sent to the model.',
    maxTurns: 'Upper bound for tool-calling turns before Hermes stops a run.'
  },
  terminal: {
    cwd: 'Default project folder for tool and terminal work.',
    persistentShell: 'Keep shell state between commands when the backend supports it.',
    envPassthrough: 'Environment variables to pass into tool execution.',
    dockerImage: 'Container image used when the execution backend is Docker.',
    singularityImage: 'Image used when the execution backend is Singularity.',
    modalImage: 'Image used when the execution backend is Modal.',
    daytonaImage: 'Image used when the execution backend is Daytona.'
  },
  codeExecution: {
    mode: 'How strictly code execution is scoped to the current project.'
  },
  fileReadMaxChars: 'Maximum characters Hermes can read from one file request.',
  approvals: {
    mode: 'How Hermes handles commands that need explicit approval.',
    timeout: 'How long approval prompts wait before timing out.'
  },
  security: {
    redactSecrets: 'Hide detected secrets from model-visible content when possible.'
  },
  checkpoints: {
    enabled: 'Create rollback snapshots before file edits.'
  },
  memory: {
    memoryEnabled: 'Save durable memories that can help future sessions.',
    userProfileEnabled: 'Maintain a compact profile of user preferences.'
  },
  context: {
    engine: 'Strategy for managing long conversations near the context limit.'
  },
  compression: {
    enabled: 'Summarize older context when conversations get large.'
  },
  voice: {
    autoTts: 'Automatically speak assistant responses.',
    realtime: {
      enabled: 'Use the KAME-inspired realtime websocket path instead of the turn-based record/transcribe/speak loop.',
      inputBufferLimitBytes: 'Maximum local realtime audio bytes Hermes buffers before dropping an unfinished turn.',
      inputFrameMs:
        'Microphone chunk duration in milliseconds for realtime voice. Lower values reduce latency but send more frames.',
      silenceTimeoutMs: 'Milliseconds of silence before Hermes closes the current realtime voice turn.',
      speechLevelThreshold:
        'Normalized microphone level required before realtime voice starts a user turn. Raise it in noisy rooms.',
      bargeInMinSpeechMs:
        'Milliseconds of sustained speech over assistant playback before Hermes sends a barge-in event.',
      bargeInMinRms: 'Minimum decoded PCM RMS amplitude required before Discord voice barge-in can interrupt playback.',
      bargeInStopPlaybackDeadlineMs:
        'Maximum milliseconds allowed to stop assistant playback after confirmed barge-in.',
      preRollMs:
        'Milliseconds of microphone audio retained before speech starts, so the first syllable is not clipped.',
      requireLiveLike:
        'Require native speech-to-speech or streaming STT/TTS before realtime voice is considered available.',
      productionLanguages:
        'Comma-separated BCP-47 language tags that Hermes treats as production acceptance targets. Defaults to English and Japanese.',
      productionScripts:
        'Comma-separated ISO 15924 script tags covered by production realtime voice acceptance. Defaults to Latn and Jpan.',
      bestEffortLanguages:
        'Allow other languages to pass through captions, prompts, and provider auto-detection without claiming production quality.',
      qualityTargetsMs: {
        audioToPartialTranscriptMs:
          'Milliseconds from user audio to the first partial transcript before realtime voice is considered slow.',
        finalTranscriptToFirstTextMs:
          'Milliseconds from final transcript to first assistant text before realtime voice is considered slow.',
        finalTranscriptToFirstAudioMs:
          'Milliseconds from final transcript to first assistant audio before realtime voice is considered slow.',
        bargeInAckMs:
          'Milliseconds for backend barge-in acknowledgement before realtime interruption is considered slow.'
      },
      frontendProvider: 'Low-latency realtime interface provider. Use gemma4 for full KAME reflex/oracle mode.',
      frontendModel: 'Low-latency realtime interface model, such as Gemma 4 E2B.',
      interfaceTemperature: 'Sampling temperature for one KAME interface routing decision.',
      interfaceMaxOutputTokens: 'Maximum output tokens for one KAME interface routing decision.',
      interfaceTimeoutSeconds: 'Seconds to wait for the KAME interface model before falling back.',
      interfaceMaxAudioSeconds: 'Maximum native-audio segment seconds sent to the KAME interface model.',
      interfaceAudioInput:
        'Whether the KAME reflex receives native audio, transcript fallback, or automatic selection.',
      asrMode: 'Controls when ASR runs as oracle-verbatim evidence instead of feeding the reflex.',
      asrProvider: 'ASR provider used for oracle-verbatim evidence or text fallback.',
      asrModel: 'ASR model used for oracle-verbatim evidence or text fallback.',
      oracleProvider: 'Optional local oracle provider registration for KAME realtime voice.',
      oracleProviderName: 'Display name for the local oracle provider added by KAME profile setup.',
      preferredLocalOracleModel:
        'Preferred local oracle provider target label for KAME realtime voice; Hermes still chooses the active oracle through /model.',
      oracleBaseUrl: 'OpenAI-compatible base URL for registering a local KAME oracle endpoint.',
      oracleApiMode: 'Wire protocol used by the registered local KAME oracle endpoint.',
      oracleTimeoutSeconds: 'Seconds to wait for an oracle voice response before speaking a timeout status.',
      maxSpokenSentences: 'Maximum spoken sentences for KAME oracle voice responses.',
      voiceResponsePolicy: 'How KAME shapes long oracle answers for spoken output.',
      ttsProvider: 'TTS provider used for KAME spoken output.',
      ttsModel: 'TTS model used for KAME spoken output.',
      ttsVoice: 'TTS voice identifier used for KAME spoken output.',
      fallbackPolicy: 'Fallback behavior when the KAME sidecar or local stack is unavailable.',
      turnAcknowledgement: {
        enabled: 'Speak a short acknowledgement while the oracle thinks.',
        text: 'Short acknowledgement text used before the final response is ready.'
      },
      routing: {
        allowLocalGreetings: 'Allow the KAME reflex to answer greetings and hear-me checks without the oracle.',
        allowLocalClarifications: 'Allow the KAME reflex to ask short clarification questions without the oracle.',
        requireOracleForTools: 'Require Hermes oracle authority for tool-using voice turns.',
        requireOracleForMemory: 'Require Hermes oracle authority for memory-dependent voice turns.',
        requireOracleForFiles: 'Require Hermes oracle authority for file or project voice turns.',
        localConfidenceThreshold: 'Minimum reflex confidence required before local KAME replies are allowed.'
      },
      metrics: {
        enabled: 'Enable realtime voice turn and provider metrics.',
        logTurnSpans: 'Log per-turn KAME latency spans.',
        logProviderSpans: 'Log provider-specific realtime voice spans.'
      },
      outputEvents: {
        captionAliases: 'Emit assistant.caption.partial/final aliases alongside legacy assistant text events.',
        audioAliases: 'Emit assistant.audio.chunk aliases alongside legacy audio.output.chunk events.'
      },
      sidecarBaseUrl:
        'LAN or private-network URL for a remote inference sidecar running Gemma, streaming TTS, or native S2S.',
      sidecarHost: 'Host used when Hermes autostarts the managed realtime voice sidecar.',
      sidecarPort: 'Port used when Hermes autostarts the managed realtime voice sidecar.',
      sidecarAutostart: 'Start the managed loopback sidecar automatically when realtime voice opens.',
      sidecarTokenEnv: 'Environment variable that contains the bearer token for the voice sidecar.',
      sidecarConnectTimeoutSeconds:
        'Seconds Hermes waits for a realtime voice sidecar websocket before falling back or failing.',
      streamingSttBaseUrl: 'Compatible streaming speech-to-text bridge URL for the reference sidecar.',
      streamingSttModel: 'Speech-to-text model label advertised by the streaming bridge.',
      streamingSttTokenEnv: 'Environment variable containing the streaming STT bridge bearer token.',
      streamingTtsBaseUrl: 'Compatible streaming text-to-speech bridge URL for the reference sidecar.',
      streamingTtsModel: 'Text-to-speech model label advertised by the streaming bridge.',
      streamingTtsTokenEnv: 'Environment variable containing the streaming TTS bridge bearer token.',
      openaiRealtimeApiKeyEnv: 'Environment variable containing the OpenAI Realtime API key.',
      openaiRealtimeBaseUrl: 'OpenAI Realtime WebSocket endpoint.',
      openaiRealtimeVoice: 'Voice used by OpenAI Realtime native speech output.',
      openaiRealtimeTranscriptionModel: 'OpenAI transcription model used for realtime input transcription.',
      geminiLiveApiKeyEnv: 'Environment variable containing the Gemini Live API key.',
      geminiLiveBaseUrl: 'Gemini Live WebSocket endpoint.',
      geminiLiveVoice: 'Voice used by Gemini Live native speech output.',
      geminiLiveGoogleSearch: 'Allow Gemini Live to use Google Search as a frontend context tool.',
      geminiLiveOracleTool: 'Allow Gemini Live to call the restricted Hermes oracle bridge tool.',
      sparkBaseUrl: 'Deprecated compatibility alias for the voice sidecar URL. Prefer voice.realtime.sidecar_base_url.'
    }
  },
  tts: {
    xai: {
      voiceId: 'xAI voice ID (e.g. eve) or a custom voice ID.',
      language: 'Spoken language code, e.g. en.'
    },
    neutts: {
      device: 'Local inference device for NeuTTS.'
    }
  },
  stt: {
    enabled: 'Enable local or provider-backed speech transcription.',
    elevenlabs: {
      languageCode: 'Optional ISO-639-3 language code. Blank lets ElevenLabs auto-detect.'
    }
  },
  updates: {
    nonInteractiveLocalChanges:
      'When Hermes updates itself from the app (no terminal prompt), keep local source edits (stash) or throw them away (discard). Terminal updates always ask.'
  }
})

// Curated desktop config surface: only fields a user might tune from the app.
export const SECTIONS: DesktopConfigSection[] = [
  {
    id: 'model',
    label: 'Model',
    icon: codiconIcon('hubot'),
    keys: ['model_context_length', 'fallback_providers']
  },
  {
    id: 'chat',
    label: 'Chat',
    icon: MessageCircle,
    keys: ['display.personality', 'timezone', 'display.show_reasoning', 'agent.image_input_mode']
  },
  {
    id: 'appearance',
    label: 'Appearance',
    icon: Palette,
    keys: []
  },
  {
    id: 'workspace',
    label: 'Workspace',
    icon: Monitor,
    keys: [
      'terminal.cwd',
      'code_execution.mode',
      'terminal.persistent_shell',
      'terminal.env_passthrough',
      'file_read_max_chars'
    ]
  },
  {
    id: 'safety',
    label: 'Safety',
    icon: Lock,
    keys: [
      'approvals.mode',
      'approvals.timeout',
      'approvals.mcp_reload_confirm',
      'command_allowlist',
      'security.redact_secrets',
      'security.allow_private_urls',
      'browser.allow_private_urls',
      'browser.auto_local_for_private_urls',
      'checkpoints.enabled'
    ]
  },
  {
    id: 'memory',
    label: 'Memory & Context',
    icon: Brain,
    keys: [
      'memory.memory_enabled',
      'memory.user_profile_enabled',
      'memory.memory_char_limit',
      'memory.user_char_limit',
      'memory.provider',
      'context.engine',
      'compression.enabled',
      'compression.threshold',
      'compression.target_ratio',
      'compression.protect_last_n'
    ]
  },
  {
    id: 'voice',
    label: 'Voice',
    icon: Mic,
    keys: [
      'tts.provider',
      'stt.enabled',
      'stt.provider',
      'voice.auto_tts',
      'tts.edge.voice',
      'tts.openai.model',
      'tts.openai.voice',
      'tts.elevenlabs.voice_id',
      'tts.elevenlabs.model_id',
      'tts.xai.voice_id',
      'tts.xai.language',
      'tts.minimax.model',
      'tts.minimax.voice_id',
      'tts.mistral.model',
      'tts.mistral.voice_id',
      'tts.gemini.model',
      'tts.gemini.voice',
      'tts.neutts.model',
      'tts.neutts.device',
      'tts.kittentts.model',
      'tts.kittentts.voice',
      'tts.piper.voice',
      'stt.local.model',
      'stt.local.language',
      'stt.openai.model',
      'stt.groq.model',
      'stt.mistral.model',
      'stt.elevenlabs.model_id',
      'stt.elevenlabs.language_code',
      'stt.elevenlabs.tag_audio_events',
      'stt.elevenlabs.diarize',
      'voice.realtime.enabled',
      'voice.realtime.engine',
      'voice.realtime.input_codec',
      'voice.realtime.output_codec',
      'voice.realtime.input_buffer_limit_bytes',
      'voice.realtime.input_frame_ms',
      'voice.realtime.silence_timeout_ms',
      'voice.realtime.speech_level_threshold',
      'voice.realtime.barge_in_min_speech_ms',
      'voice.realtime.barge_in_min_rms',
      'voice.realtime.barge_in_stop_playback_deadline_ms',
      'voice.realtime.pre_roll_ms',
      'voice.realtime.require_live_like',
      'voice.realtime.production_languages',
      'voice.realtime.production_scripts',
      'voice.realtime.best_effort_languages',
      'voice.realtime.quality_targets_ms.audio_to_partial_transcript_ms',
      'voice.realtime.quality_targets_ms.final_transcript_to_first_text_ms',
      'voice.realtime.quality_targets_ms.final_transcript_to_first_audio_ms',
      'voice.realtime.quality_targets_ms.barge_in_ack_ms',
      'voice.realtime.frontend_provider',
      'voice.realtime.frontend_model',
      'voice.realtime.interface_temperature',
      'voice.realtime.interface_max_output_tokens',
      'voice.realtime.interface_timeout_seconds',
      'voice.realtime.interface_max_audio_seconds',
      'voice.realtime.interface_audio_input',
      'voice.realtime.asr_mode',
      'voice.realtime.asr_provider',
      'voice.realtime.asr_model',
      'voice.realtime.oracle_provider',
      'voice.realtime.oracle_provider_name',
      'voice.realtime.preferred_local_oracle_model',
      'voice.realtime.oracle_base_url',
      'voice.realtime.oracle_api_mode',
      'voice.realtime.oracle_timeout_seconds',
      'voice.realtime.max_spoken_sentences',
      'voice.realtime.voice_response_policy',
      'voice.realtime.tts_provider',
      'voice.realtime.tts_model',
      'voice.realtime.tts_voice',
      'voice.realtime.fallback_policy',
      'voice.realtime.turn_acknowledgement.enabled',
      'voice.realtime.turn_acknowledgement.text',
      'voice.realtime.routing.allow_local_greetings',
      'voice.realtime.routing.allow_local_clarifications',
      'voice.realtime.routing.require_oracle_for_tools',
      'voice.realtime.routing.require_oracle_for_memory',
      'voice.realtime.routing.require_oracle_for_files',
      'voice.realtime.routing.local_confidence_threshold',
      'voice.realtime.metrics.enabled',
      'voice.realtime.metrics.log_turn_spans',
      'voice.realtime.metrics.log_provider_spans',
      'voice.realtime.oracle_tool_router.enabled',
      'voice.realtime.oracle_tool_router.mode',
      'voice.realtime.oracle_tool_router.voiceops_toolsets',
      'voice.realtime.oracle_tool_router.default_toolsets',
      'voice.realtime.output_events.caption_aliases',
      'voice.realtime.output_events.audio_aliases',
      'voice.realtime.sidecar_base_url',
      'voice.realtime.sidecar_host',
      'voice.realtime.sidecar_port',
      'voice.realtime.sidecar_autostart',
      'voice.realtime.sidecar_token_env',
      'voice.realtime.sidecar_connect_timeout_seconds',
      'voice.realtime.streaming_stt_base_url',
      'voice.realtime.streaming_stt_model',
      'voice.realtime.streaming_stt_token_env',
      'voice.realtime.streaming_tts_base_url',
      'voice.realtime.streaming_tts_model',
      'voice.realtime.streaming_tts_token_env',
      'voice.realtime.openai_realtime_api_key_env',
      'voice.realtime.openai_realtime_base_url',
      'voice.realtime.openai_realtime_voice',
      'voice.realtime.openai_realtime_transcription_model',
      'voice.realtime.gemini_live_api_key_env',
      'voice.realtime.gemini_live_base_url',
      'voice.realtime.gemini_live_voice',
      'voice.realtime.gemini_live_google_search',
      'voice.realtime.gemini_live_oracle_tool',
      'voice.record_key',
      'voice.max_recording_seconds'
    ]
  },
  {
    id: 'advanced',
    label: 'Advanced',
    icon: Wrench,
    keys: [
      'toolsets',
      'tools.tool_search.defer_core',
      'terminal.backend',
      'terminal.timeout',
      'terminal.docker_image',
      'terminal.singularity_image',
      'terminal.modal_image',
      'terminal.daytona_image',
      'tool_output.max_bytes',
      'tool_output.max_lines',
      'tool_output.max_line_length',
      'checkpoints.max_snapshots',
      'agent.max_turns',
      'agent.api_max_retries',
      'agent.service_tier',
      'agent.tool_use_enforcement',
      'delegation.model',
      'delegation.provider',
      'delegation.max_iterations',
      'delegation.max_concurrent_children',
      'delegation.child_timeout_seconds',
      'delegation.reasoning_effort',
      'updates.non_interactive_local_changes'
    ]
  }
]

export interface ModeOption {
  id: ThemeMode
  label: string
  icon: IconComponent
}

export const MODE_OPTIONS: ModeOption[] = [
  { id: 'light', label: 'Light', icon: Sun },
  { id: 'dark', label: 'Dark', icon: Moon },
  { id: 'system', label: 'System', icon: Monitor }
]
