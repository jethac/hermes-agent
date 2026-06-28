import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

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

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

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
      asr_provider: 'nemotron_speech',
      barge_in_min_rms: 420,
      barge_in_min_speech_ms: 160,
      barge_in_stop_playback_deadline_ms: 140,
      enabled: true,
      engine: 'kame_interface_oracle',
      fallback_policy: 'fail_closed',
      frontend_model: 'gemma-4-E2B-it',
      frontend_provider: 'gemma4',
      interface_api_key_env: 'CUSTOM_KAME_INTERFACE_TOKEN',
      interface_base_url: 'http://spark.local:8000/v1',
      interface_audio_input: 'native_audio',
      interface_max_audio_seconds: 24,
      interface_max_output_tokens: 96,
      interface_temperature: 0.3,
      interface_timeout_seconds: 0.7,
      metrics: {
        enabled: true,
        log_provider_spans: false,
        log_turn_spans: true
      },
      oracle_base_url: 'http://spark.local:8000/v1',
      oracle_api_mode: 'chat_completions',
      oracle_model: 'gemma-4-26B-A4B-it',
      oracle_provider: 'custom',
      oracle_provider_name: 'Spark Oracle',
      oracle_timeout_seconds: 42,
      max_spoken_sentences: 3,
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
      streaming_tts_voice: '5ee9feff-1265-424a-9d7f-8e4d431a12c7',
      tts_provider: 'cartesia',
      tts_model: 'sonic-3.5',
      tts_voice: '5ee9feff-1265-424a-9d7f-8e4d431a12c7',
      vllm_model: 'google/gemma-4-E2B-it',
      voice_response_policy: 'brief_summary'
    }
  }
}

const nestedKameConfig: HermesConfigRecord = {
  discord: {
    realtime_voice: {
      enabled: true,
      sidecar_base_url: 'http://127.0.0.1:8765'
    }
  },
  voice: {
    realtime: {
      enabled: true,
      engine: 'kame_interface_oracle',
      fallback_policy: 'fail_closed',
      interface: {
        api_key_env: 'NESTED_KAME_INTERFACE_TOKEN',
        asr_mode: 'on_escalation',
        audio_input: 'native_audio',
        base_url: 'http://spark.local:8001/v1',
        max_audio_seconds: 18,
        max_output_tokens: 88,
        model: 'google/gemma-4-E2B-it',
        provider: 'gemma4',
        temperature: 0.25,
        timeout_ms: 650
      },
      oracle: {
        api_mode: 'chat_completions',
        base_url: 'http://spark.local:8002/v1',
        max_spoken_sentences: 2,
        model: 'gemma-4-26B-A4B-it',
        preferred_local_model: 'gemma-4-26B-A4B-it',
        provider: 'custom',
        provider_name: 'Nested Spark Oracle',
        response_policy: 'brief_summary',
        timeout_ms: 45000
      },
      asr: {
        base_url: 'http://127.0.0.1:8770',
        model: 'nemotron-speech-streaming-0.6b',
        provider: 'nemotron_speech'
      },
      tts: {
        base_url: 'http://127.0.0.1:8771',
        model: 'magpie-tts',
        provider: 'nvidia_speech',
        voice: 'puck-local'
      },
      barge_in: {
        min_rms: 375,
        min_speech_ms: 130,
        stop_playback_deadline_ms: 145
      },
      metrics: {
        enabled: true,
        log_provider_spans: true,
        log_turn_spans: false
      },
      routing: {
        allow_local_clarifications: true,
        allow_local_greetings: false,
        local_confidence_threshold: 0.82,
        require_oracle_for_files: true,
        require_oracle_for_memory: true,
        require_oracle_for_tools: true
      }
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
    expect(screen.getByText('Interface base URL')).toBeTruthy()
    expect(screen.getByText('Interface provider')).toBeTruthy()
    expect(screen.getByText('Interface key env')).toBeTruthy()
    expect(screen.getByText('Served reflex model')).toBeTruthy()
    expect(screen.getByText('Interface temperature')).toBeTruthy()
    expect(screen.getByText('Interface token limit')).toBeTruthy()
    expect(screen.getByText('Interface timeout')).toBeTruthy()
    expect(screen.getByText('Interface audio input')).toBeTruthy()
    expect(screen.getByText('ASR mode')).toBeTruthy()
    expect(screen.getByText('ASR provider')).toBeTruthy()
    expect(screen.getByText('Fallback policy')).toBeTruthy()
    expect(screen.getByText('TTS voice')).toBeTruthy()
    expect(screen.getByText('TTS bridge voice')).toBeTruthy()
    expect(screen.getByText('TTS provider')).toBeTruthy()
    expect(screen.getByText('Oracle provider')).toBeTruthy()
    expect(screen.getByText('Oracle provider name')).toBeTruthy()
    expect(screen.getByText('Oracle API mode')).toBeTruthy()
    expect(screen.getByText('Oracle timeout')).toBeTruthy()
    expect(screen.getByText('Spoken sentence cap')).toBeTruthy()
    expect(screen.getByText('Barge-in RMS')).toBeTruthy()
    expect(screen.getByText('Local confidence')).toBeTruthy()
    expect(screen.getByText('Metrics enabled')).toBeTruthy()
    expect(screen.getByText('Turn span logs')).toBeTruthy()
    expect(screen.getByText('Provider span logs')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /apply provider/i }))

    await waitFor(() => expect(hermesMocks.applyRealtimeVoiceProfile).toHaveBeenCalledTimes(1))
    expect(hermesMocks.applyRealtimeVoiceProfile).toHaveBeenCalledWith(
      expect.objectContaining({
        allow_local_clarifications: false,
        allow_local_greetings: false,
        asr_mode: 'on_escalation',
        asr_provider: 'nemotron_speech',
        barge_in_min_rms: 420,
        barge_in_min_speech_ms: 160,
        barge_in_stop_playback_deadline_ms: 140,
        fallback_policy: 'fail_closed',
        interface_base_url: 'http://spark.local:8000/v1',
        interface_provider: 'gemma4',
        interface_api_key_env: 'CUSTOM_KAME_INTERFACE_TOKEN',
        interface_audio_input: 'native_audio',
        interface_max_audio_seconds: 24,
        interface_max_output_tokens: 96,
        interface_temperature: 0.3,
        interface_timeout_seconds: 0.7,
        local_confidence_threshold: 0.88,
        max_spoken_sentences: 3,
        metrics_enabled: true,
        metrics_log_provider_spans: false,
        metrics_log_turn_spans: true,
        oracle_base_url: 'http://spark.local:8000/v1',
        oracle_api_mode: 'chat_completions',
        oracle_model: 'gemma-4-26B-A4B-it',
        oracle_provider: 'custom',
        oracle_provider_name: 'Spark Oracle',
        oracle_timeout_seconds: 42,
        preset: 'kame',
        require_oracle_for_files: true,
        require_oracle_for_memory: true,
        require_oracle_for_tools: true,
        streaming_stt_base_url: 'http://127.0.0.1:8766',
        streaming_stt_model: 'nova-3',
        streaming_tts_base_url: 'http://127.0.0.1:8769',
        streaming_tts_model: 'sonic-3.5',
        streaming_tts_voice: '5ee9feff-1265-424a-9d7f-8e4d431a12c7',
        tts_provider: 'cartesia',
        tts_model: 'sonic-3.5',
        tts_voice: '5ee9feff-1265-424a-9d7f-8e4d431a12c7',
        vllm_model: 'google/gemma-4-E2B-it',
        voice_response_policy: 'brief_summary'
      })
    )
  })

  it('renders KAME provider controls as selectable provider lists', async () => {
    render(<RealtimeVoiceSetupPanel config={config} onConfigChange={vi.fn()} />)

    expect(await screen.findByText('KAME Reflex / Oracle')).toBeTruthy()

    const comboboxWithValue = (value: string) => {
      const combobox = screen
        .getAllByRole('combobox')
        .find(element => element.textContent?.includes(value))

      expect(combobox).toBeTruthy()

      return combobox as HTMLElement
    }

    fireEvent.click(comboboxWithValue('gemma4'))
    expect(await screen.findByText('openai_realtime')).toBeTruthy()
    expect(screen.getByText('gemini_live')).toBeTruthy()
    fireEvent.keyDown(document.activeElement ?? document.body, { key: 'Escape' })

    fireEvent.click(comboboxWithValue('nemotron_speech'))
    expect((await screen.findAllByText('nemotron_speech')).length).toBeGreaterThan(0)
    expect(screen.getByText('nvidia_speech')).toBeTruthy()
    expect(screen.getByText('streaming_stt')).toBeTruthy()
    fireEvent.keyDown(document.activeElement ?? document.body, { key: 'Escape' })

    fireEvent.click(comboboxWithValue('cartesia'))
    expect((await screen.findAllByText('cartesia')).length).toBeGreaterThan(0)
    expect(screen.getByText('streaming_tts')).toBeTruthy()
  })

  it('submits nested KAME design config without flattening values to defaults', async () => {
    render(<RealtimeVoiceSetupPanel config={nestedKameConfig} onConfigChange={vi.fn()} />)

    expect(await screen.findByText('KAME Reflex / Oracle')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /apply provider/i }))

    await waitFor(() => expect(hermesMocks.applyRealtimeVoiceProfile).toHaveBeenCalledTimes(1))
    expect(hermesMocks.applyRealtimeVoiceProfile).toHaveBeenCalledWith(
      expect.objectContaining({
        allow_local_clarifications: true,
        allow_local_greetings: false,
        asr_mode: 'on_escalation',
        asr_provider: 'nemotron_speech',
        barge_in_min_rms: 375,
        barge_in_min_speech_ms: 130,
        barge_in_stop_playback_deadline_ms: 145,
        fallback_policy: 'fail_closed',
        interface_api_key_env: 'NESTED_KAME_INTERFACE_TOKEN',
        interface_audio_input: 'native_audio',
        interface_base_url: 'http://spark.local:8001/v1',
        interface_max_audio_seconds: 18,
        interface_max_output_tokens: 88,
        interface_provider: 'gemma4',
        interface_temperature: 0.25,
        interface_timeout_seconds: 0.65,
        metrics_enabled: true,
        metrics_log_provider_spans: true,
        metrics_log_turn_spans: false,
        model: 'google/gemma-4-E2B-it',
        oracle_api_mode: 'chat_completions',
        oracle_base_url: 'http://spark.local:8002/v1',
        oracle_model: 'gemma-4-26B-A4B-it',
        oracle_provider: 'custom',
        oracle_provider_name: 'Nested Spark Oracle',
        oracle_timeout_seconds: 45,
        preset: 'kame',
        streaming_stt_base_url: 'http://127.0.0.1:8770',
        streaming_stt_model: 'nemotron-speech-streaming-0.6b',
        streaming_tts_base_url: 'http://127.0.0.1:8771',
        streaming_tts_model: 'magpie-tts',
        tts_model: 'magpie-tts',
        tts_provider: 'nvidia_speech',
        tts_voice: 'puck-local',
        vllm_model: 'google/gemma-4-E2B-it',
        voice_response_policy: 'brief_summary'
      })
    )
  })
})
