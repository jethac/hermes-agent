import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  collectRealtimeVoiceCaption,
  collectRealtimeVoiceFrontendState,
  collectRealtimeVoiceMetrics,
  getRealtimeVoiceStatus,
  queueRealtimeAudioTask,
  realtimeAudioInputPayload,
  realtimeBinaryAudioInputFrame,
  realtimeVoiceCloseAction,
  realtimeVoicePlaybackGeneration,
  realtimeVoiceSessionStatus,
  realtimeVoiceUrl,
  shouldDropStaleRealtimeVoiceEvent,
  shouldSendRealtimeAudioFrame,
  shouldSendRealtimeVoiceEndMarker,
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
      sidecar: { mode: 'none' }
    }))

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })

    await expect(getRealtimeVoiceStatus()).resolves.toMatchObject({
      available: false,
      enabled: true,
      engine: 'native_s2s_oracle'
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

describe('collectRealtimeVoiceCaption', () => {
  const event = (type: string, payload: Record<string, unknown>, timestamp_ms = 1_234): VoiceEvent => ({
    payload,
    sequence: 1,
    session_id: 'voice-123',
    timestamp_ms,
    type
  })

  it('tracks user partial and final transcript captions', () => {
    const partial = collectRealtimeVoiceCaption(null, event('transcript.partial', { text: 'hello her' }))
    const final = collectRealtimeVoiceCaption(partial, event('transcript.final', { text: 'hello hermes' }, 1_500))

    expect(partial).toEqual({
      final: false,
      speaker: 'user',
      text: 'hello her',
      updatedAtMs: 1_234
    })
    expect(final).toEqual({
      final: true,
      speaker: 'user',
      text: 'hello hermes',
      updatedAtMs: 1_500
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
