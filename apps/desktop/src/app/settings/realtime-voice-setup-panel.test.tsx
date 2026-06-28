import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConfigRecord, RealtimeVoiceSetupResponse } from '@/hermes'

import { RealtimeVoiceSetupPanel } from './realtime-voice-setup-panel'

const hermesMocks = vi.hoisted(() => ({
  applyRealtimeVoiceProfile: vi.fn(),
  getRealtimeVoiceSetup: vi.fn(),
  runRealtimeVoiceSmoke: vi.fn()
}))

vi.mock('@/hermes', () => hermesMocks)

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

const setupResponse: RealtimeVoiceSetupResponse = {
  active_model: 'gemma-4-E2B-it',
  active_provider: 'gemma4',
  discord: {
    bot_token_present: true,
    enabled: true,
    guild_id_present: true,
    sidecar_base_url: 'http://127.0.0.1:8765',
    voice_channel_name_present: true
  },
  enabled: true,
  providers: [
    {
      api_key_present: true,
      id: 'kame',
      implemented: true,
      kind: 'kame_interface_oracle',
      label: 'KAME Gemma reflex',
      model: 'gemma-4-E2B-it',
      provider: 'gemma4'
    }
  ],
  status: {
    available: true,
    enabled: true,
    engine: 'kame_interface_oracle',
    frontend_model: 'gemma-4-E2B-it',
    frontend_provider: 'gemma4',
    sidecar: {
      autostart: true,
      base_url: 'http://127.0.0.1:8765',
      health: {
        capabilities: { kame_reflex: true, tts: true },
        frontend: { provider: 'gemma4' },
        ok: true
      },
      healthy: true,
      mode: 'managed'
    }
  }
}

const config: HermesConfigRecord = {
  discord: {
    realtime_voice: {
      enabled: true,
      sidecar_base_url: 'http://127.0.0.1:8765'
    }
  },
  voice: {
    realtime: {
      asr_mode: 'on_escalation',
      enabled: true,
      engine: 'kame_interface_oracle',
      frontend_model: 'gemma-4-E2B-it',
      frontend_provider: 'gemma4',
      interface_audio_input: 'native_audio',
      interface_max_audio_seconds: 24,
      oracle_base_url: 'http://spark.local:8000/v1',
      oracle_model: 'gemma-4-26B-A4B-it',
      routing: {
        allow_local_clarifications: false,
        allow_local_greetings: false,
        local_confidence_threshold: 0.88,
        require_oracle_for_files: true,
        require_oracle_for_memory: true,
        require_oracle_for_tools: true
      },
      streaming_stt_base_url: 'http://127.0.0.1:8766',
      streaming_stt_model: 'nova-3',
      streaming_tts_base_url: 'http://127.0.0.1:8769',
      streaming_tts_model: 'sonic-3.5',
      voice_response_policy: 'brief_summary'
    }
  }
}

describe('RealtimeVoiceSetupPanel', () => {
  beforeEach(() => {
    hermesMocks.getRealtimeVoiceSetup.mockResolvedValue(setupResponse)
    hermesMocks.applyRealtimeVoiceProfile.mockResolvedValue({
      config,
      ok: true,
      setup: setupResponse
    })
    hermesMocks.runRealtimeVoiceSmoke.mockResolvedValue({
      ok: true,
      output_dir: 'artifacts/realtime-voice-smoke/test'
    })
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('renders KAME routing controls and submits them with the profile apply request', async () => {
    const onConfigChange = vi.fn()
    render(<RealtimeVoiceSetupPanel config={config} onConfigChange={onConfigChange} />)

    expect(await screen.findByText('KAME Reflex / Oracle')).toBeTruthy()
    expect(screen.getByText('Interface audio input')).toBeTruthy()
    expect(screen.getByText('ASR mode')).toBeTruthy()
    expect(screen.getByText('Local confidence')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /apply provider/i }))

    await waitFor(() => expect(hermesMocks.applyRealtimeVoiceProfile).toHaveBeenCalledTimes(1))
    expect(hermesMocks.applyRealtimeVoiceProfile).toHaveBeenCalledWith(
      expect.objectContaining({
        allow_local_clarifications: false,
        allow_local_greetings: false,
        asr_mode: 'on_escalation',
        interface_audio_input: 'native_audio',
        interface_max_audio_seconds: 24,
        local_confidence_threshold: 0.88,
        oracle_base_url: 'http://spark.local:8000/v1',
        oracle_model: 'gemma-4-26B-A4B-it',
        preset: 'kame',
        require_oracle_for_files: true,
        require_oracle_for_memory: true,
        require_oracle_for_tools: true,
        streaming_stt_base_url: 'http://127.0.0.1:8766',
        streaming_stt_model: 'nova-3',
        streaming_tts_base_url: 'http://127.0.0.1:8769',
        streaming_tts_model: 'sonic-3.5',
        voice_response_policy: 'brief_summary'
      })
    )
  })
})
