import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  collectRealtimeVoiceMetrics,
  getRealtimeVoiceStatus,
  realtimeVoiceUrl
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
