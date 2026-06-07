import { describe, expect, it } from 'vitest'

import { realtimeVoiceQuality } from './controls'

describe('realtimeVoiceQuality', () => {
  it('returns null until realtime metrics arrive', () => {
    expect(realtimeVoiceQuality({})).toBeNull()
  })

  it('marks metrics within PRD latency targets as good', () => {
    expect(realtimeVoiceQuality({
      audioToPartialTranscriptMs: 180,
      bargeInAckMs: 90,
      finalTranscriptToFirstAudioMs: 650
    })).toMatchObject({
      primaryMs: 650,
      state: 'good'
    })
  })

  it('marks the snapshot slow when any key realtime path misses its target', () => {
    expect(realtimeVoiceQuality({
      audioToPartialTranscriptMs: 180,
      bargeInAckMs: 170,
      finalTranscriptToFirstAudioMs: 650
    })).toMatchObject({
      primaryMs: 650,
      state: 'slow'
    })
  })

  it('uses transcript latency as the primary value until first audio is known', () => {
    expect(realtimeVoiceQuality({ audioToPartialTranscriptMs: 210 })).toMatchObject({
      primaryMs: 210,
      state: 'good'
    })
  })
})
