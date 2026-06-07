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
      finalTranscriptToFirstTextMs: 220,
      finalTranscriptToFirstAudioMs: 650
    })).toMatchObject({
      primaryMs: 650,
      state: 'slow'
    })
  })

  it('marks slow when assistant text misses its target before audio arrives', () => {
    expect(realtimeVoiceQuality({
      audioToPartialTranscriptMs: 180,
      finalTranscriptToFirstTextMs: 650
    })).toMatchObject({
      primaryMs: 180,
      state: 'slow'
    })
  })

  it('uses backend-provided quality targets when available', () => {
    expect(realtimeVoiceQuality({
      audioToPartialTranscriptMs: 180,
      bargeInAckMs: 170,
      finalTranscriptToFirstTextMs: 450,
      finalTranscriptToFirstAudioMs: 650
    }, {
      audio_to_partial_transcript_ms: 300,
      barge_in_ack_ms: 200,
      final_transcript_to_first_text_ms: 500,
      final_transcript_to_first_audio_ms: 700
    })).toMatchObject({
      primaryMs: 650,
      state: 'good'
    })
  })

  it('uses transcript latency as the primary value until first audio is known', () => {
    expect(realtimeVoiceQuality({ audioToPartialTranscriptMs: 210 })).toMatchObject({
      primaryMs: 210,
      state: 'good'
    })
  })
})
