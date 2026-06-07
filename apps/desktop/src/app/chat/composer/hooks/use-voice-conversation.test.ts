import { describe, expect, it } from 'vitest'

import { takeVoiceConversationSpeechChunk } from './use-voice-conversation'

describe('takeVoiceConversationSpeechChunk', () => {
  it('keeps the existing low-latency English sentence behavior', () => {
    expect(takeVoiceConversationSpeechChunk('Answering now. More is coming')).toEqual({
      chunk: 'Answering now.',
      remaining: 'More is coming'
    })
  })

  it('splits compact Japanese sentences without requiring trailing whitespace', () => {
    expect(takeVoiceConversationSpeechChunk('これは最初の返答です。続きもあります')).toEqual({
      chunk: 'これは最初の返答です。',
      remaining: '続きもあります'
    })
  })

  it('splits Arabic question punctuation without requiring an English question mark', () => {
    expect(takeVoiceConversationSpeechChunk('هذا رد طويل بما يكفي؟وهذه متابعة')).toEqual({
      chunk: 'هذا رد طويل بما يكفي؟',
      remaining: 'وهذه متابعة'
    })
  })

  it('uses non-ASCII phrase punctuation for languages without spaces', () => {
    const text =
      'これは多言語音声の計画で英語の空白に頼らず自然な句読点を探すための長い前置きです' +
      'これは多言語音声の計画で英語の空白に頼らず自然な句読点を探すための長い前置きです、' +
      'ここから先もまだかなり長く続くので句読点で先に読み上げられる必要があります'

    const result = takeVoiceConversationSpeechChunk(text)

    expect(result.chunk?.endsWith('です、')).toBe(true)
    expect(`${result.chunk}${result.remaining}`).toBe(text)
  })

  it('buffers unfinished speech unless forced', () => {
    expect(takeVoiceConversationSpeechChunk('This is still forming')).toEqual({
      chunk: null,
      remaining: 'This is still forming'
    })
    expect(takeVoiceConversationSpeechChunk('This is still forming', true)).toEqual({
      chunk: 'This is still forming',
      remaining: ''
    })
  })
})
