import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  applyRealtimePlaybackQueueBackpressure,
  collectRealtimeVoiceCaption,
  collectRealtimeVoiceFrontendState,
  collectRealtimeVoiceMetrics,
  getRealtimeVoiceStatus,
  nextRealtimeVoicePlaybackGeneration,
  parseRealtimeVoiceServerMessage,
  queueRealtimeAudioTask,
  realtimeAudioInputPayload,
  realtimeBinaryAudioInputFrame,
  realtimeVoiceCloseAction,
  realtimeVoiceBargeInMinSpeechMs,
  realtimeVoiceInputFrameMs,
  realtimeVoiceEventGeneration,
  realtimeVoicePlaybackGeneration,
  realtimeVoicePlaybackQueueAction,
  realtimeVoicePreRollMs,
  realtimeVoicePreRollChunkLimit,
  realtimeVoiceSpeechLevelThreshold,
  realtimeVoiceSessionErrorAction,
  realtimeVoiceSessionReadyTimeoutMs,
  realtimeVoiceSessionStatus,
  realtimeVoiceSilenceTimeoutMs,
  realtimeVoiceConversationQualityFrontendState,
  realtimeVoiceUnavailableFrontendState,
  realtimeVoiceQualityTargets,
  realtimeVoiceQualityTargetsFromPayload,
  realtimeVoiceUrl,
  shouldDropQueuedRealtimeAudioInput,
  shouldDropStaleRealtimeVoiceEvent,
  shouldRestartRealtimeTurnRecorder,
  shouldSendRealtimeAudioFrame,
  shouldSendRealtimeVoiceEndMarker,
  shouldStartRealtimeTurnCapture,
  updateRealtimeVoiceBargeInGate
} from './use-realtime-voice-session'

import type { RealtimeVoiceLatencyMetrics, VoiceEvent } from './use-realtime-voice-session'

const { resolveGatewayWsUrlMock } = vi.hoisted(() => ({
  resolveGatewayWsUrlMock: vi.fn(async () => 'ws://127.0.0.1:9119/api/ws?token=t')
}))

vi.mock('@/lib/gateway-ws-url', () => ({
  resolveGatewayWsUrl: resolveGatewayWsUrlMock
}))

describe('realtimeVoiceUrl', () => {
  afterEach(() => {
    vi.clearAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('uses the gateway websocket origin and targets realtime voice for the active session', async () => {
    const getConnection = vi.fn(async () => ({ gatewayUrl: 'http://127.0.0.1:9119' }))

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { getConnection }
    })

    await expect(realtimeVoiceUrl('session-123')).resolves.toBe(
      'ws://127.0.0.1:9119/api/voice/realtime?token=t&session_id=session-123'
    )
    expect(getConnection).toHaveBeenCalledTimes(1)
    expect(resolveGatewayWsUrlMock).toHaveBeenCalledTimes(1)
  })

  it('loads realtime voice status through the authenticated desktop API bridge', async () => {
    const api = vi.fn(async () => ({
      available: false,
      enabled: true,
      engine: 'native_s2s_oracle',
      barge_in_min_speech_ms: 120,
      pre_roll_ms: 300,
      speech_level_threshold: 0.075,
      language_support: {
        best_effort_languages: true,
        production_languages: ['en', 'ja'],
        production_scripts: ['Latn', 'Jpan'],
        sidecar_languages_are_diagnostics: true
      },
      quality_targets_ms: {
        audio_to_partial_transcript_ms: 250,
        barge_in_ack_ms: 120,
        final_transcript_to_first_audio_ms: 850,
        final_transcript_to_first_text_ms: 450
      },
      sidecar: {
        health: {
          capabilities: {
            input_languages: ['en', 'ja'],
            output_languages: ['en', 'ja'],
            scripts: ['Latn', 'Jpan']
          },
          frontend: {
            languages: ['en', 'ja'],
            scripts: ['Latn', 'Jpan']
          }
        },
        mode: 'none'
      },
      unavailable_reason: 'sidecar_required'
    }))

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })

    await expect(getRealtimeVoiceStatus()).resolves.toMatchObject({
      available: false,
      enabled: true,
      engine: 'native_s2s_oracle',
      barge_in_min_speech_ms: 120,
      pre_roll_ms: 300,
      speech_level_threshold: 0.075,
      language_support: {
        production_languages: ['en', 'ja'],
        production_scripts: ['Latn', 'Jpan']
      },
      quality_targets_ms: {
        audio_to_partial_transcript_ms: 250,
        final_transcript_to_first_audio_ms: 850
      },
      sidecar: {
        health: {
          capabilities: {
            input_languages: ['en', 'ja'],
            output_languages: ['en', 'ja'],
            scripts: ['Latn', 'Jpan']
          }
        }
      },
      unavailable_reason: 'sidecar_required'
    })
    expect(api).toHaveBeenCalledWith({ path: '/api/voice/realtime/status' })
  })
})

describe('collectRealtimeVoiceMetrics', () => {
  const event = (payload: Record<string, unknown>, timestamp_ms = 1_234): VoiceEvent => ({
    payload,
    sequence: 1,
    session_id: 'voice-123',
    timestamp_ms,
    type: 'transcript.final'
  })

  it('maps backend latency metrics into a desktop snapshot', () => {
    const metrics = collectRealtimeVoiceMetrics({}, event({
      metrics: {
        audio_to_final_transcript_ms: 111,
        audio_to_partial_transcript_ms: 42,
        barge_in_ack_ms: 17,
        eou_to_final_transcript_ms: 29,
        final_transcript_to_first_audio_ms: 310,
        final_transcript_to_first_text_ms: 91,
        session_elapsed_ms: 1_000
      }
    }))

    expect(metrics).toEqual({
      audioToFinalTranscriptMs: 111,
      audioToPartialTranscriptMs: 42,
      bargeInAckMs: 17,
      eouToFinalTranscriptMs: 29,
      finalTranscriptToFirstAudioMs: 310,
      finalTranscriptToFirstTextMs: 91,
      sessionElapsedMs: 1_000,
      updatedAtMs: 1_234
    })
  })

  it('ignores malformed metric values and keeps the previous object when nothing changes', () => {
    const previous: RealtimeVoiceLatencyMetrics = { sessionElapsedMs: 50, updatedAtMs: 100 }

    expect(collectRealtimeVoiceMetrics(previous, event({}))).toBe(previous)
    expect(collectRealtimeVoiceMetrics(previous, event({ metrics: { session_elapsed_ms: -1 } }))).toBe(previous)
    expect(collectRealtimeVoiceMetrics(previous, event({ metrics: { session_elapsed_ms: 'fast' } }))).toBe(previous)
  })

  it('merges new valid metrics with an existing snapshot', () => {
    const previous: RealtimeVoiceLatencyMetrics = { audioToPartialTranscriptMs: 40, sessionElapsedMs: 90 }
    const metrics = collectRealtimeVoiceMetrics(previous, event({
      metrics: {
        final_transcript_to_first_audio_ms: 250,
        session_elapsed_ms: 120
      }
    }, 1_500))

    expect(metrics).toEqual({
      audioToPartialTranscriptMs: 40,
      finalTranscriptToFirstAudioMs: 250,
      sessionElapsedMs: 120,
      updatedAtMs: 1_500
    })
  })
})

describe('realtimeVoiceQualityTargets', () => {
  it('uses backend targets and defaults malformed values', () => {
    expect(realtimeVoiceQualityTargets({
      available: true,
      enabled: true,
      engine: 'text_oracle_tts',
      quality_targets_ms: {
        audio_to_partial_transcript_ms: 250,
        barge_in_ack_ms: -1,
        final_transcript_to_first_audio_ms: 0,
        final_transcript_to_first_text_ms: Number.NaN
      }
    })).toEqual({
      audio_to_partial_transcript_ms: 250,
      barge_in_ack_ms: 150,
      final_transcript_to_first_audio_ms: 900,
      final_transcript_to_first_text_ms: 500
    })
  })

  it('uses session-start payload targets without requiring status preflight', () => {
    expect(realtimeVoiceQualityTargetsFromPayload({
      audio_to_partial_transcript_ms: 220,
      barge_in_ack_ms: 130,
      final_transcript_to_first_audio_ms: 800,
      final_transcript_to_first_text_ms: 420
    })).toEqual({
      audio_to_partial_transcript_ms: 220,
      barge_in_ack_ms: 130,
      final_transcript_to_first_audio_ms: 800,
      final_transcript_to_first_text_ms: 420
    })
    expect(realtimeVoiceQualityTargetsFromPayload('slow')).toEqual({
      audio_to_partial_transcript_ms: 300,
      barge_in_ack_ms: 150,
      final_transcript_to_first_audio_ms: 900,
      final_transcript_to_first_text_ms: 500
    })
  })
})

describe('collectRealtimeVoiceCaption', () => {
  const event = (type: string, payload: Record<string, unknown>, timestamp_ms = 1_234): VoiceEvent => ({
    payload,
    sequence: 1,
    session_id: 'voice-123',
    timestamp_ms,
    type
  })

  it('tracks user partial and final transcript captions', () => {
    const partial = collectRealtimeVoiceCaption(null, event('transcript.partial', {
      language: 'ja',
      locale: 'ja-JP',
      script: 'Jpan',
      text: 'こんにちは'
    }))
    const final = collectRealtimeVoiceCaption(partial, event('transcript.final', { text: 'こんにちは Hermes' }, 1_500))

    expect(partial).toEqual({
      final: false,
      language: 'ja',
      locale: 'ja-JP',
      script: 'Jpan',
      speaker: 'user',
      text: 'こんにちは',
      updatedAtMs: 1_234
    })
    expect(final).toEqual({
      final: true,
      language: 'ja',
      locale: 'ja-JP',
      script: 'Jpan',
      speaker: 'user',
      text: 'こんにちは Hermes',
      updatedAtMs: 1_500
    })
  })

  it('filters malformed caption language metadata', () => {
    const caption = collectRealtimeVoiceCaption(null, event('transcript.final', {
      language: 'https://voice.local/secret',
      locale: 'ja-JP',
      script: 'bad/script',
      text: 'こんにちは'
    }))

    expect(caption).toEqual({
      final: true,
      locale: 'ja-JP',
      speaker: 'user',
      text: 'こんにちは',
      updatedAtMs: 1_234
    })
  })

  it('accumulates assistant text chunks until commit replaces the caption', () => {
    const first = collectRealtimeVoiceCaption(null, event('assistant.text.partial', { text: 'Answering ' }))
    const second = collectRealtimeVoiceCaption(first, event('assistant.text.partial', { text: 'now.' }, 1_300))
    const committed = collectRealtimeVoiceCaption(second, event('assistant.commit', { text: 'Answering now.' }, 1_600))

    expect(second).toEqual({
      final: false,
      speaker: 'assistant',
      text: 'Answering now.',
      updatedAtMs: 1_300
    })
    expect(committed).toEqual({
      final: true,
      speaker: 'assistant',
      text: 'Answering now.',
      updatedAtMs: 1_600
    })
  })

  it('accumulates assistant deltas from live provider-style partials', () => {
    const first = collectRealtimeVoiceCaption(null, event('assistant.text.partial', {
      delta: '回答します。',
      language: 'ja',
      locale: 'ja-JP',
      script: 'Jpan'
    }))
    const second = collectRealtimeVoiceCaption(first, event('assistant.text.partial', { delta: '続けます。' }, 1_300))

    expect(second).toEqual({
      final: false,
      language: 'ja',
      locale: 'ja-JP',
      script: 'Jpan',
      speaker: 'assistant',
      text: '回答します。続けます。',
      updatedAtMs: 1_300
    })
  })

  it('uses cumulative assistant text as the first caption when both text and delta are present', () => {
    const first = collectRealtimeVoiceCaption(null, event('assistant.text.partial', {
      delta: 'now.',
      text: 'Answering now.'
    }))
    const second = collectRealtimeVoiceCaption(first, event('assistant.text.partial', { delta: ' More.' }, 1_300))

    expect(first?.text).toBe('Answering now.')
    expect(second?.text).toBe('Answering now. More.')
  })

  it('clears assistant captions on barge-in but keeps user captions', () => {
    const assistant = collectRealtimeVoiceCaption(null, event('assistant.text.partial', { text: 'old answer' }))
    const user = collectRealtimeVoiceCaption(null, event('transcript.partial', { text: 'new question' }))

    expect(collectRealtimeVoiceCaption(assistant, event('barge_in', {}))).toBeNull()
    expect(collectRealtimeVoiceCaption(user, event('barge_in', {}))).toBe(user)
  })
})

describe('collectRealtimeVoiceFrontendState', () => {
  const event = (type: string, payload: Record<string, unknown>, timestamp_ms = 1_234): VoiceEvent => ({
    payload,
    sequence: 1,
    session_id: 'voice-123',
    timestamp_ms,
    type
  })

  it('tracks recoverable frontend fallback state', () => {
    expect(collectRealtimeVoiceFrontendState(null, event('frontend.state', {
      reason: 'sidecar_send_failed',
      status: 'fallback'
    }))).toEqual({
      reason: 'sidecar_send_failed',
      status: 'fallback',
      updatedAtMs: 1_234
    })
  })

  it('clears degraded state when the frontend reports ready', () => {
    const previous = {
      reason: 'sidecar_event_stream_failed',
      status: 'degraded' as const,
      updatedAtMs: 1_000
    }

    expect(collectRealtimeVoiceFrontendState(previous, event('frontend.state', { status: 'ready' }))).toBeNull()
    expect(collectRealtimeVoiceFrontendState(previous, event('transcript.partial', { text: 'hello' }))).toBe(previous)
  })

  it('marks realtime voice degraded when latency targets are missed', () => {
    expect(collectRealtimeVoiceFrontendState(null, event('assistant.text.partial', {
      quality_target_misses: [
        {
          actual_ms: 650,
          metric: 'final_transcript_to_first_text_ms',
          target_ms: 500
        }
      ],
      text: 'slow'
    }))).toEqual({
      reason: 'quality_target_missed',
      status: 'degraded',
      updatedAtMs: 1_234
    })
  })

  it('does not replace a fallback state with latency degradation', () => {
    const previous = {
      reason: 'sidecar_send_failed',
      status: 'fallback' as const,
      updatedAtMs: 1_000
    }

    expect(collectRealtimeVoiceFrontendState(previous, event('assistant.text.partial', {
      quality_target_misses: [
        {
          actual_ms: 650,
          metric: 'final_transcript_to_first_text_ms',
          target_ms: 500
        }
      ],
      text: 'slow'
    }))).toBe(previous)
  })
})

describe('realtimeVoiceUnavailableFrontendState', () => {
  it('does not report fallback state when realtime voice is enabled and available', () => {
    expect(realtimeVoiceUnavailableFrontendState({
      available: true,
      enabled: true,
      engine: 'text_oracle_tts'
    }, 1_234)).toBeNull()
  })

  it('uses the stable unavailable reason from the backend status payload', () => {
    expect(realtimeVoiceUnavailableFrontendState({
      available: false,
      enabled: true,
      engine: 'native_s2s_oracle',
      language_support: {
        best_effort_languages: true,
        production_languages: ['en', 'ja'],
        production_scripts: ['Latn', 'Jpan'],
        sidecar_languages_are_diagnostics: true
      },
      unavailable_reason: 'sidecar_missing_native_s2s'
    }, 1_234)).toEqual({
      reason: 'sidecar_missing_native_s2s',
      status: 'fallback',
      updatedAtMs: 1_234
    })
  })

  it('reports disabled realtime voice as fallback without treating languages as a gate', () => {
    expect(realtimeVoiceUnavailableFrontendState({
      available: false,
      enabled: false,
      engine: 'text_oracle_tts',
      language_support: {
        best_effort_languages: true,
        production_languages: ['en', 'ja']
      }
    }, 1_234)).toEqual({
      reason: 'disabled',
      status: 'fallback',
      updatedAtMs: 1_234
    })
  })
})

describe('realtimeAudioInputPayload', () => {
  it('preserves the caller snapshot of end-of-utterance state', () => {
    expect(realtimeAudioInputPayload({
      dataB64: 'abc',
      endOfUtterance: true,
      mimeType: 'audio/webm;codecs=opus'
    })).toEqual({
      channels: 1,
      codec: 'webm_opus',
      data_b64: 'abc',
      end_of_utterance: true,
      sample_rate_hz: 16000
    })
  })

  it('maps ogg recorder output to opus wire codec', () => {
    expect(realtimeAudioInputPayload({
      dataB64: 'abc',
      endOfUtterance: false,
      mimeType: 'audio/ogg;codecs=opus'
    })).toMatchObject({
      codec: 'opus',
      end_of_utterance: false
    })
  })
})

describe('realtimeBinaryAudioInputFrame', () => {
  it('encodes realtime audio as a JSON header plus raw bytes', () => {
    const audioData = new Uint8Array([1, 2, 3]).buffer
    const frame = realtimeBinaryAudioInputFrame({
      audioData,
      endOfUtterance: true,
      mimeType: 'audio/webm;codecs=opus',
      sequence: 7,
      sessionId: 'voice-123'
    })
    const bytes = new Uint8Array(frame)
    const headerLength = new DataView(frame).getUint32(0, false)
    const header = JSON.parse(new TextDecoder().decode(bytes.slice(4, 4 + headerLength)))

    expect(header).toEqual({
      payload: {
        channels: 1,
        codec: 'webm_opus',
        end_of_utterance: true,
        sample_rate_hz: 16000
      },
      sequence: 7,
      session_id: 'voice-123',
      type: 'audio.input.chunk'
    })
    expect(Array.from(bytes.slice(4 + headerLength))).toEqual([1, 2, 3])
  })
})

describe('parseRealtimeVoiceServerMessage', () => {
  it('parses legacy JSON websocket events', async () => {
    await expect(parseRealtimeVoiceServerMessage(JSON.stringify({
      payload: { engine: 'text_oracle_tts' },
      sequence: 1,
      session_id: 'voice-123',
      type: 'session.started'
    }))).resolves.toMatchObject({
      payload: { engine: 'text_oracle_tts' },
      sequence: 1,
      session_id: 'voice-123',
      type: 'session.started'
    })
  })

  it('parses binary server audio without base64 encoding the raw bytes', async () => {
    const header = new TextEncoder().encode(JSON.stringify({
      payload: {
        codec: 'opus',
        playback_generation: 2
      },
      sequence: 9,
      session_id: 'voice-123',
      type: 'audio.output.chunk'
    }))
    const frame = new Uint8Array(4 + header.byteLength + 3)

    new DataView(frame.buffer).setUint32(0, header.byteLength, false)
    frame.set(header, 4)
    frame.set(new Uint8Array([4, 5, 6]), 4 + header.byteLength)

    const event = await parseRealtimeVoiceServerMessage(frame.buffer)

    expect(event.type).toBe('audio.output.chunk')
    expect(event.payload?.data_b64).toBeUndefined()
    expect(Array.from(event.payload?.data_bytes as Uint8Array)).toEqual([4, 5, 6])
    expect(event.payload?.playback_generation).toBe(2)
  })
})

describe('shouldSendRealtimeAudioFrame', () => {
  it('sends ordinary audio when the websocket buffer is under the live threshold', () => {
    expect(shouldSendRealtimeAudioFrame({
      bufferedAmount: 128,
      endOfUtterance: false,
      maxBufferedBytes: 256
    })).toBe(true)
  })

  it('drops non-final audio when the websocket buffer is too far behind', () => {
    expect(shouldSendRealtimeAudioFrame({
      bufferedAmount: 512,
      endOfUtterance: false,
      maxBufferedBytes: 256
    })).toBe(false)
  })

  it('always sends end-of-utterance frames so turn boundaries survive backpressure', () => {
    expect(shouldSendRealtimeAudioFrame({
      bufferedAmount: 512,
      endOfUtterance: true,
      maxBufferedBytes: 256
    })).toBe(true)
  })
})

describe('applyRealtimePlaybackQueueBackpressure', () => {
  it('keeps queued playback unchanged while under the live backlog limit', () => {
    const queue = ['a', 'b']

    expect(applyRealtimePlaybackQueueBackpressure({ maxItems: 3, queue })).toEqual({
      dropped: 0,
      queue
    })
  })

  it('drops the oldest queued playback chunks when the browser falls behind', () => {
    expect(applyRealtimePlaybackQueueBackpressure({
      maxItems: 3,
      queue: ['old-1', 'old-2', 'new-1', 'new-2', 'new-3']
    })).toEqual({
      dropped: 2,
      queue: ['new-1', 'new-2', 'new-3']
    })
  })

  it('treats malformed limits as zero so callers fail closed', () => {
    expect(applyRealtimePlaybackQueueBackpressure({
      maxItems: Number.NaN,
      queue: ['stale']
    })).toEqual({
      dropped: 1,
      queue: []
    })
  })
})

describe('realtimeVoiceInputFrameMs', () => {
  it('defaults to low-latency 100 ms microphone frames', () => {
    expect(realtimeVoiceInputFrameMs(undefined)).toBe(100)
    expect(realtimeVoiceInputFrameMs('fast')).toBe(100)
  })

  it('rounds and clamps configured microphone frame duration', () => {
    expect(realtimeVoiceInputFrameMs(79.6)).toBe(80)
    expect(realtimeVoiceInputFrameMs(10)).toBe(40)
    expect(realtimeVoiceInputFrameMs(1_000)).toBe(500)
  })
})

describe('realtimeVoiceSilenceTimeoutMs', () => {
  it('defaults to a low-latency 650 ms end-of-utterance silence window', () => {
    expect(realtimeVoiceSilenceTimeoutMs(undefined)).toBe(650)
    expect(realtimeVoiceSilenceTimeoutMs('slow')).toBe(650)
  })

  it('rounds and clamps configured silence timeout duration', () => {
    expect(realtimeVoiceSilenceTimeoutMs(799.6)).toBe(800)
    expect(realtimeVoiceSilenceTimeoutMs(100)).toBe(250)
    expect(realtimeVoiceSilenceTimeoutMs(5_000)).toBe(2_000)
  })
})

describe('realtime voice capture tuning', () => {
  it('clamps speech level threshold from backend status', () => {
    expect(realtimeVoiceSpeechLevelThreshold(undefined)).toBe(0.075)
    expect(realtimeVoiceSpeechLevelThreshold(0)).toBe(0.075)
    expect(realtimeVoiceSpeechLevelThreshold(0.001)).toBe(0.005)
    expect(realtimeVoiceSpeechLevelThreshold(2)).toBe(1)
    expect(realtimeVoiceSpeechLevelThreshold(0.12)).toBe(0.12)
  })

  it('clamps barge-in speech duration from backend status', () => {
    expect(realtimeVoiceBargeInMinSpeechMs(undefined)).toBe(120)
    expect(realtimeVoiceBargeInMinSpeechMs(0)).toBe(120)
    expect(realtimeVoiceBargeInMinSpeechMs(10)).toBe(40)
    expect(realtimeVoiceBargeInMinSpeechMs(2_000)).toBe(1_000)
    expect(realtimeVoiceBargeInMinSpeechMs(149.6)).toBe(150)
  })

  it('clamps pre-roll duration from backend status', () => {
    expect(realtimeVoicePreRollMs(undefined)).toBe(300)
    expect(realtimeVoicePreRollMs(-1)).toBe(300)
    expect(realtimeVoicePreRollMs(0)).toBe(0)
    expect(realtimeVoicePreRollMs(2_000)).toBe(1_000)
    expect(realtimeVoicePreRollMs(249.6)).toBe(250)
  })
})

describe('realtimeVoiceSessionReadyTimeoutMs', () => {
  it('defaults to a bounded wait for session.started', () => {
    expect(realtimeVoiceSessionReadyTimeoutMs(null)).toBe(12_000)
    expect(realtimeVoiceSessionReadyTimeoutMs({
      available: true,
      enabled: true,
      engine: 'text_oracle_tts',
      sidecar: { connect_timeout_seconds: undefined }
    })).toBe(12_000)
  })

  it('derives the wait from sidecar connect timeout with guardrails', () => {
    expect(realtimeVoiceSessionReadyTimeoutMs({
      available: true,
      enabled: true,
      engine: 'text_oracle_tts',
      sidecar: { connect_timeout_seconds: 2.5 }
    })).toBe(4_500)
    expect(realtimeVoiceSessionReadyTimeoutMs({
      available: true,
      enabled: true,
      engine: 'text_oracle_tts',
      sidecar: { connect_timeout_seconds: 0 }
    })).toBe(3_000)
    expect(realtimeVoiceSessionReadyTimeoutMs({
      available: true,
      enabled: true,
      engine: 'text_oracle_tts',
      sidecar: { connect_timeout_seconds: 90 }
    })).toBe(30_000)
  })
})

describe('realtimeVoicePreRollChunkLimit', () => {
  it('keeps roughly 300 ms of local pre-roll using the active frame duration', () => {
    expect(realtimeVoicePreRollChunkLimit(undefined)).toBe(3)
    expect(realtimeVoicePreRollChunkLimit(80)).toBe(4)
    expect(realtimeVoicePreRollChunkLimit(500)).toBe(1)
  })

  it('uses configured pre-roll duration and allows disabling pre-roll', () => {
    expect(realtimeVoicePreRollChunkLimit(100, 500)).toBe(5)
    expect(realtimeVoicePreRollChunkLimit(100, 0)).toBe(0)
  })
})

describe('queueRealtimeAudioTask', () => {
  it('runs realtime audio sends in capture order even when later work is ready first', async () => {
    const order: string[] = []
    let releaseFirst!: () => void
    const firstReady = new Promise<void>(resolve => {
      releaseFirst = resolve
    })

    const first = queueRealtimeAudioTask(Promise.resolve(), async () => {
      order.push('first-start')
      await firstReady
      order.push('first-end')
    }, () => undefined)
    const second = queueRealtimeAudioTask(first, async () => {
      order.push('second')
    }, () => undefined)

    await Promise.resolve()
    await Promise.resolve()
    expect(order).toEqual(['first-start'])
    releaseFirst()
    await second

    expect(order).toEqual(['first-start', 'first-end', 'second'])
  })

  it('continues after a prior audio send error and reports the failure', async () => {
    const errors: unknown[] = []
    const order: string[] = []

    const first = queueRealtimeAudioTask(Promise.resolve(), async () => {
      throw new Error('encode failed')
    }, error => {
      errors.push(error)
    })
    const second = queueRealtimeAudioTask(first, async () => {
      order.push('second')
    }, error => {
      errors.push(error)
    })

    await second

    expect(errors).toHaveLength(1)
    expect(order).toEqual(['second'])
  })
})

describe('shouldDropQueuedRealtimeAudioInput', () => {
  it('keeps audio queued for the active input generation', () => {
    expect(shouldDropQueuedRealtimeAudioInput({
      activeGeneration: 3,
      queuedGeneration: 3
    })).toBe(false)
  })

  it('drops audio queued before a new input generation such as barge-in', () => {
    expect(shouldDropQueuedRealtimeAudioInput({
      activeGeneration: 4,
      queuedGeneration: 3
    })).toBe(true)
  })
})

describe('shouldSendRealtimeVoiceEndMarker', () => {
  it('sends a fallback end marker when silence stopped the recorder before a final chunk was sent', () => {
    expect(shouldSendRealtimeVoiceEndMarker({
      closingInput: false,
      sentEndOfUtterance: false,
      stoppedForSilence: true
    })).toBe(true)
  })

  it('does not send duplicate or shutdown end markers', () => {
    expect(shouldSendRealtimeVoiceEndMarker({
      closingInput: false,
      sentEndOfUtterance: true,
      stoppedForSilence: true
    })).toBe(false)
    expect(shouldSendRealtimeVoiceEndMarker({
      closingInput: true,
      sentEndOfUtterance: false,
      stoppedForSilence: true
    })).toBe(false)
    expect(shouldSendRealtimeVoiceEndMarker({
      closingInput: false,
      sentEndOfUtterance: false,
      stoppedForSilence: false
    })).toBe(false)
  })
})

describe('shouldStartRealtimeTurnCapture', () => {
  it('starts capture only after accepted speech while realtime is available', () => {
    expect(shouldStartRealtimeTurnCapture({
      acceptSpeech: true,
      busy: false,
      enabled: true,
      muted: false,
      turnCaptureActive: false
    })).toBe(true)
  })

  it('does not start capture while idle, muted, busy, disabled, or already capturing a turn', () => {
    expect(shouldStartRealtimeTurnCapture({
      acceptSpeech: false,
      busy: false,
      enabled: true,
      muted: false,
      turnCaptureActive: false
    })).toBe(false)
    expect(shouldStartRealtimeTurnCapture({
      acceptSpeech: true,
      busy: false,
      enabled: true,
      muted: true,
      turnCaptureActive: false
    })).toBe(false)
    expect(shouldStartRealtimeTurnCapture({
      acceptSpeech: true,
      busy: true,
      enabled: true,
      muted: false,
      turnCaptureActive: false
    })).toBe(false)
    expect(shouldStartRealtimeTurnCapture({
      acceptSpeech: true,
      busy: false,
      enabled: false,
      muted: false,
      turnCaptureActive: false
    })).toBe(false)
    expect(shouldStartRealtimeTurnCapture({
      acceptSpeech: true,
      busy: false,
      enabled: true,
      muted: false,
      turnCaptureActive: true
    })).toBe(false)
  })
})

describe('shouldRestartRealtimeTurnRecorder', () => {
  it('restarts the per-turn recorder when accepted speech arrives after a prior turn stopped it', () => {
    expect(shouldRestartRealtimeTurnRecorder({
      acceptSpeech: true,
      hasRecorder: false,
      streamAvailable: true,
      turnCaptureActive: false
    })).toBe(true)
  })

  it('keeps the current recorder when one is already active or speech is not accepted', () => {
    expect(shouldRestartRealtimeTurnRecorder({
      acceptSpeech: true,
      hasRecorder: true,
      streamAvailable: true,
      turnCaptureActive: false
    })).toBe(false)
    expect(shouldRestartRealtimeTurnRecorder({
      acceptSpeech: false,
      hasRecorder: false,
      streamAvailable: true,
      turnCaptureActive: false
    })).toBe(false)
    expect(shouldRestartRealtimeTurnRecorder({
      acceptSpeech: true,
      hasRecorder: false,
      streamAvailable: false,
      turnCaptureActive: false
    })).toBe(false)
    expect(shouldRestartRealtimeTurnRecorder({
      acceptSpeech: true,
      hasRecorder: false,
      streamAvailable: true,
      turnCaptureActive: true
    })).toBe(false)
  })
})

describe('realtimeVoiceCloseAction', () => {
  it('falls back when realtime closes before the backend session starts', () => {
    expect(realtimeVoiceCloseAction({
      closeCode: 1011,
      enabled: true,
      sessionStarted: false
    })).toBe('fallback')
  })

  it('treats active-session abnormal closes as fatal', () => {
    expect(realtimeVoiceCloseAction({
      closeCode: 1011,
      enabled: true,
      sessionStarted: true
    })).toBe('fatal')
  })

  it('ignores normal or disabled closes', () => {
    expect(realtimeVoiceCloseAction({
      closeCode: 1000,
      enabled: true,
      sessionStarted: true
    })).toBe('ignore')
    expect(realtimeVoiceCloseAction({
      closeCode: 1006,
      enabled: false,
      sessionStarted: false
    })).toBe('ignore')
  })

  it('does not fall back after a pre-start session error event', () => {
    expect(realtimeVoiceCloseAction({
      closeCode: 1011,
      enabled: true,
      sessionFailed: true,
      sessionStarted: false
    })).toBe('ignore')
  })
})

describe('realtimeVoiceSessionErrorAction', () => {
  it('falls back for startup errors before session.started', () => {
    expect(realtimeVoiceSessionErrorAction({ sessionStarted: false })).toBe('fallback')
  })

  it('treats active-session errors as fatal', () => {
    expect(realtimeVoiceSessionErrorAction({ sessionStarted: true })).toBe('fatal')
  })
})

describe('realtimeVoiceConversationQualityFrontendState', () => {
  it('marks available turn-based realtime as degraded instead of fallback', () => {
    expect(realtimeVoiceConversationQualityFrontendState({
      available: true,
      enabled: true,
      conversation_quality: {
        live_like: false,
        mode: 'turn_based_text',
        reason: 'utterance_stt_tts'
      }
    }, 1_234)).toEqual({
      reason: 'utterance_stt_tts',
      status: 'degraded',
      updatedAtMs: 1_234
    })
  })

  it('does not degrade unavailable or live-like realtime sessions', () => {
    expect(realtimeVoiceConversationQualityFrontendState({
      available: false,
      enabled: true,
      conversation_quality: {
        live_like: false,
        reason: 'sidecar_unhealthy'
      }
    }, 1_234)).toBeNull()
    expect(realtimeVoiceConversationQualityFrontendState({
      available: true,
      enabled: true,
      conversation_quality: {
        live_like: true,
        mode: 'streaming_text',
        reason: 'streaming_stt_tts'
      }
    }, 1_234)).toBeNull()
  })
})

describe('realtimeVoiceSessionStatus', () => {
  const event = (session_state: unknown): VoiceEvent => ({
    payload: { session_state },
    sequence: 1,
    session_id: 'voice-123',
    type: 'transcript.final'
  })

  it('maps backend realtime session states to desktop conversation status', () => {
    expect(realtimeVoiceSessionStatus(event('listening'))).toBe('listening')
    expect(realtimeVoiceSessionStatus(event('assistant_pending'))).toBe('thinking')
    expect(realtimeVoiceSessionStatus(event('speaking'))).toBe('speaking')
    expect(realtimeVoiceSessionStatus(event('closed'))).toBe('idle')
  })

  it('ignores missing or unknown backend session states', () => {
    expect(realtimeVoiceSessionStatus(event('unknown'))).toBeNull()
    expect(realtimeVoiceSessionStatus(event(null))).toBeNull()
  })
})

describe('realtime playback generation helpers', () => {
  const event = (type: string, playback_generation: unknown): VoiceEvent => ({
    payload: { playback_generation },
    sequence: 1,
    session_id: 'voice-123',
    type
  })

  it('parses numeric playback generations from wire payloads', () => {
    expect(realtimeVoicePlaybackGeneration({ playback_generation: 3 })).toBe(3)
    expect(realtimeVoicePlaybackGeneration({ playback_generation: '4' })).toBe(4)
    expect(realtimeVoicePlaybackGeneration({ playback_generation: true })).toBeNull()
    expect(realtimeVoicePlaybackGeneration({ playback_generation: -1 })).toBeNull()
    expect(realtimeVoicePlaybackGeneration({ playback_generation: 'old' })).toBeNull()
  })

  it('advances playback generation from numeric strings without over-incrementing', () => {
    expect(nextRealtimeVoicePlaybackGeneration(1, '4')).toBe(4)
    expect(nextRealtimeVoicePlaybackGeneration(4, '2')).toBe(4)
    expect(nextRealtimeVoicePlaybackGeneration(4, undefined)).toBe(5)
  })

  it('uses parsed wire generation for audio event queueing', () => {
    expect(realtimeVoiceEventGeneration({ playback_generation: '7' }, 3)).toBe(7)
    expect(realtimeVoiceEventGeneration({ playback_generation: true }, 3)).toBe(3)
    expect(realtimeVoiceEventGeneration(undefined, 3)).toBe(3)
  })

  it('drops stale generated assistant and transcript events', () => {
    expect(shouldDropStaleRealtimeVoiceEvent(event('assistant.text.partial', 1), 2)).toBe(true)
    expect(shouldDropStaleRealtimeVoiceEvent(event('assistant.commit', 1), 2)).toBe(true)
    expect(shouldDropStaleRealtimeVoiceEvent(event('audio.output.chunk', 1), 2)).toBe(true)
    expect(shouldDropStaleRealtimeVoiceEvent(event('transcript.final', 1), 2)).toBe(true)
  })

  it('keeps current, future, and ungenerational events', () => {
    expect(shouldDropStaleRealtimeVoiceEvent(event('assistant.text.partial', 2), 2)).toBe(false)
    expect(shouldDropStaleRealtimeVoiceEvent(event('assistant.text.partial', 3), 2)).toBe(false)
    expect(shouldDropStaleRealtimeVoiceEvent(event('transcript.partial', 1), 2)).toBe(false)
    expect(shouldDropStaleRealtimeVoiceEvent(event('assistant.text.partial', undefined), 2)).toBe(false)
  })
})

describe('realtimeVoicePlaybackQueueAction', () => {
  it('continues queued playback after a chunk ends or fails', () => {
    expect(realtimeVoicePlaybackQueueAction({
      enabled: true,
      hasQueuedAudio: true,
      muted: false
    })).toBe('play_next')
  })

  it('returns to listening when playback drains while realtime is enabled', () => {
    expect(realtimeVoicePlaybackQueueAction({
      enabled: true,
      hasQueuedAudio: false,
      muted: false
    })).toBe('listening')
  })

  it('returns to idle when playback drains while muted or disabled', () => {
    expect(realtimeVoicePlaybackQueueAction({
      enabled: true,
      hasQueuedAudio: false,
      muted: true
    })).toBe('idle')
    expect(realtimeVoicePlaybackQueueAction({
      enabled: false,
      hasQueuedAudio: false,
      muted: false
    })).toBe('idle')
  })
})

describe('updateRealtimeVoiceBargeInGate', () => {
  it('does not trigger barge-in on the first loud playback frame', () => {
    expect(updateRealtimeVoiceBargeInGate({
      isSpeechActive: true,
      minSpeechMs: 120,
      nowMs: 1_000,
      speechStartedAtMs: null
    })).toEqual({
      shouldBargeIn: false,
      speechStartedAtMs: 1_000
    })
  })

  it('triggers barge-in after speech remains active long enough', () => {
    expect(updateRealtimeVoiceBargeInGate({
      isSpeechActive: true,
      minSpeechMs: 120,
      nowMs: 1_121,
      speechStartedAtMs: 1_000
    })).toEqual({
      shouldBargeIn: true,
      speechStartedAtMs: 1_000
    })
  })

  it('resets the pending barge-in candidate when speech drops below threshold', () => {
    expect(updateRealtimeVoiceBargeInGate({
      isSpeechActive: false,
      minSpeechMs: 120,
      nowMs: 1_060,
      speechStartedAtMs: 1_000
    })).toEqual({
      shouldBargeIn: false,
      speechStartedAtMs: null
    })
  })
})
