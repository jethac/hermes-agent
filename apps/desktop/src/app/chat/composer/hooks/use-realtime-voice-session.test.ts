import { afterEach, describe, expect, it, vi } from 'vitest'

import { realtimeVoiceUrl } from './use-realtime-voice-session'

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
})
