import { useCallback, useEffect, useRef, useState } from 'react'

import { resolveGatewayWsUrl } from '@/lib/gateway-ws-url'
import { notifyError } from '@/store/notifications'

import type { ConversationStatus } from './use-voice-conversation'

interface RealtimeVoiceOptions {
  busy: boolean
  enabled: boolean
  onFatalError?: () => void
  onUnavailable?: () => void
  sessionId?: null | string
}

export interface VoiceEvent {
  payload?: Record<string, unknown>
  sequence: number
  session_id: string
  timestamp_ms?: number
  type: string
}

interface PlaybackItem {
  blob: Blob
  generation: number
}

export interface RealtimeVoiceLatencyMetrics {
  audioToFinalTranscriptMs?: number
  audioToPartialTranscriptMs?: number
  bargeInAckMs?: number
  eouToFinalTranscriptMs?: number
  finalTranscriptToFirstAudioMs?: number
  finalTranscriptToFirstTextMs?: number
  sessionElapsedMs?: number
  updatedAtMs?: number
}

export interface RealtimeVoiceCaption {
  final: boolean
  speaker: 'assistant' | 'user'
  text: string
  updatedAtMs?: number
}

export interface RealtimeVoiceFrontendState {
  reason?: string
  status: 'degraded' | 'fallback'
  updatedAtMs?: number
}

export interface RealtimeVoiceStatus {
  available: boolean
  enabled: boolean
  engine: string
  sidecar?: {
    autostart?: boolean
    healthy?: boolean | null
    mode?: string
  }
}

type BrowserAudioContext = typeof AudioContext

const METRIC_KEYS = {
  audio_to_final_transcript_ms: 'audioToFinalTranscriptMs',
  audio_to_partial_transcript_ms: 'audioToPartialTranscriptMs',
  barge_in_ack_ms: 'bargeInAckMs',
  eou_to_final_transcript_ms: 'eouToFinalTranscriptMs',
  final_transcript_to_first_audio_ms: 'finalTranscriptToFirstAudioMs',
  final_transcript_to_first_text_ms: 'finalTranscriptToFirstTextMs',
  session_elapsed_ms: 'sessionElapsedMs'
} as const satisfies Record<string, keyof RealtimeVoiceLatencyMetrics>

const SPEECH_LEVEL_THRESHOLD = 0.075
const BARGE_IN_MIN_SPEECH_MS = 120
const GENERATION_EVENT_TYPES = new Set([
  'audio.output.chunk',
  'assistant.commit',
  'assistant.text.partial',
  'transcript.final'
])

interface RealtimeVoiceBargeInGateInput {
  isSpeechActive: boolean
  minSpeechMs?: number
  nowMs: number
  speechStartedAtMs: number | null
}

interface RealtimeAudioInputPayloadOptions {
  dataB64: string
  endOfUtterance: boolean
  mimeType: string
}

interface RealtimeEndMarkerInput {
  closingInput: boolean
  sentEndOfUtterance: boolean
  stoppedForSilence: boolean
}

interface RealtimeCloseActionInput {
  closeCode: number
  enabled: boolean
  sessionFailed?: boolean
  sessionStarted: boolean
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()

    reader.onerror = () => reject(reader.error || new Error('Could not read recorded audio'))
    reader.onload = () => {
      const value = String(reader.result || '')
      resolve(value.includes(',') ? value.split(',', 2)[1] : value)
    }
    reader.readAsDataURL(blob)
  })
}

function bytesFromBase64(value: string): Uint8Array {
  const raw = atob(value)
  const bytes = new Uint8Array(raw.length)

  for (let i = 0; i < raw.length; i += 1) {
    bytes[i] = raw.charCodeAt(i)
  }

  return bytes
}

export async function realtimeVoiceUrl(sessionId: string): Promise<string> {
  const conn = await window.hermesDesktop.getConnection()
  const wsUrl = await resolveGatewayWsUrl(window.hermesDesktop, conn)
  const url = new URL(wsUrl)

  url.pathname = '/api/voice/realtime'
  url.searchParams.set('session_id', sessionId)

  return url.toString()
}

export function getRealtimeVoiceStatus(): Promise<RealtimeVoiceStatus> {
  return window.hermesDesktop.api<RealtimeVoiceStatus>({ path: '/api/voice/realtime/status' })
}

function finiteNonNegativeMs(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null
}

export function collectRealtimeVoiceMetrics(
  previous: RealtimeVoiceLatencyMetrics,
  event: VoiceEvent
): RealtimeVoiceLatencyMetrics {
  const metrics = event.payload?.metrics

  if (!metrics || typeof metrics !== 'object' || Array.isArray(metrics)) {
    return previous
  }

  let changed = false
  const next: RealtimeVoiceLatencyMetrics = { ...previous }

  for (const [wireKey, stateKey] of Object.entries(METRIC_KEYS)) {
    const value = finiteNonNegativeMs((metrics as Record<string, unknown>)[wireKey])

    if (value !== null && next[stateKey] !== value) {
      next[stateKey] = value
      changed = true
    }
  }

  if (!changed) {
    return previous
  }

  next.updatedAtMs = finiteNonNegativeMs(event.timestamp_ms) ?? Date.now()

  return next
}

export function updateRealtimeVoiceBargeInGate({
  isSpeechActive,
  minSpeechMs = BARGE_IN_MIN_SPEECH_MS,
  nowMs,
  speechStartedAtMs
}: RealtimeVoiceBargeInGateInput): { shouldBargeIn: boolean; speechStartedAtMs: number | null } {
  if (!isSpeechActive) {
    return { shouldBargeIn: false, speechStartedAtMs: null }
  }

  const startedAt = speechStartedAtMs ?? nowMs

  return {
    shouldBargeIn: nowMs - startedAt >= minSpeechMs,
    speechStartedAtMs: startedAt
  }
}

export function realtimeAudioInputPayload({
  dataB64,
  endOfUtterance,
  mimeType
}: RealtimeAudioInputPayloadOptions): Record<string, unknown> {
  return {
    codec: mimeType.includes('ogg') ? 'opus' : 'webm_opus',
    sample_rate_hz: 16000,
    channels: 1,
    data_b64: dataB64,
    end_of_utterance: endOfUtterance
  }
}

export function shouldSendRealtimeVoiceEndMarker({
  closingInput,
  sentEndOfUtterance,
  stoppedForSilence
}: RealtimeEndMarkerInput): boolean {
  return stoppedForSilence && !closingInput && !sentEndOfUtterance
}

export function realtimeVoiceCloseAction({
  closeCode,
  enabled,
  sessionFailed = false,
  sessionStarted
}: RealtimeCloseActionInput): 'fallback' | 'fatal' | 'ignore' {
  if (!enabled || closeCode === 1000 || sessionFailed) {
    return 'ignore'
  }
  return sessionStarted ? 'fatal' : 'fallback'
}

export function realtimeVoicePlaybackGeneration(payload?: Record<string, unknown>): number | null {
  const value = payload?.playback_generation

  if (typeof value === 'boolean') {
    return null
  }
  if (typeof value === 'number' && Number.isInteger(value) && value >= 0) {
    return value
  }
  if (typeof value === 'string' && /^\d+$/.test(value)) {
    return Number(value)
  }

  return null
}

export function shouldDropStaleRealtimeVoiceEvent(event: VoiceEvent, activeGeneration: number): boolean {
  if (!GENERATION_EVENT_TYPES.has(event.type)) {
    return false
  }

  const generation = realtimeVoicePlaybackGeneration(event.payload)

  return generation !== null && generation < activeGeneration
}

export function collectRealtimeVoiceCaption(
  previous: RealtimeVoiceCaption | null,
  event: VoiceEvent
): RealtimeVoiceCaption | null {
  const rawText = typeof event.payload?.text === 'string' ? event.payload.text : ''
  const text = rawText.trim()
  const updatedAtMs = finiteNonNegativeMs(event.timestamp_ms) ?? Date.now()

  if (event.type === 'transcript.partial') {
    return text ? { final: false, speaker: 'user', text, updatedAtMs } : previous
  }
  if (event.type === 'transcript.final') {
    return text ? { final: true, speaker: 'user', text, updatedAtMs } : previous
  }
  if (event.type === 'assistant.text.partial') {
    if (!text) {
      return previous
    }
    const chunk = previous?.speaker === 'assistant' ? rawText : rawText.trimStart()
    const nextText = previous?.speaker === 'assistant' ? `${previous.text}${chunk}` : chunk

    return { final: false, speaker: 'assistant', text: nextText, updatedAtMs }
  }
  if (event.type === 'assistant.commit') {
    return text ? { final: true, speaker: 'assistant', text, updatedAtMs } : previous
  }
  if (event.type === 'barge_in') {
    return previous?.speaker === 'assistant' ? null : previous
  }

  return previous
}

export function collectRealtimeVoiceFrontendState(
  previous: RealtimeVoiceFrontendState | null,
  event: VoiceEvent
): RealtimeVoiceFrontendState | null {
  if (event.type !== 'frontend.state') {
    return previous
  }

  const status = typeof event.payload?.status === 'string' ? event.payload.status : ''
  if (status !== 'fallback' && status !== 'degraded') {
    return null
  }

  const reason = typeof event.payload?.reason === 'string' ? event.payload.reason : ''

  return {
    reason: reason || undefined,
    status,
    updatedAtMs: finiteNonNegativeMs(event.timestamp_ms) ?? Date.now()
  }
}

export function useRealtimeVoiceSession({ busy, enabled, onFatalError, onUnavailable, sessionId }: RealtimeVoiceOptions) {
  const [caption, setCaption] = useState<RealtimeVoiceCaption | null>(null)
  const [frontendState, setFrontendState] = useState<RealtimeVoiceFrontendState | null>(null)
  const [status, setStatus] = useState<ConversationStatus>('idle')
  const [level, setLevel] = useState(0)
  const [muted, setMuted] = useState(false)
  const [metrics, setMetrics] = useState<RealtimeVoiceLatencyMetrics>({})
  const socketRef = useRef<WebSocket | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserFrameRef = useRef<number | null>(null)
  const closingInputRef = useRef(false)
  const sequenceRef = useRef(0)
  const sessionRef = useRef(`voice-${Math.random().toString(36).slice(2)}`)
  const heardSpeechRef = useRef(false)
  const silenceStartedAtRef = useRef<number | null>(null)
  const sentEndOfUtteranceRef = useRef(false)
  const stoppingForSilenceRef = useRef(false)
  const sessionStartedRef = useRef(false)
  const sessionFailedRef = useRef(false)
  const playbackQueueRef = useRef<PlaybackItem[]>([])
  const playingRef = useRef<HTMLAudioElement | null>(null)
  const bargeInSpeechStartedAtRef = useRef<number | null>(null)
  const playbackGenerationRef = useRef(0)
  const enabledRef = useRef(enabled)
  const mutedRef = useRef(muted)
  const busyRef = useRef(busy)

  useEffect(() => {
    enabledRef.current = enabled
  }, [enabled])

  useEffect(() => {
    mutedRef.current = muted
  }, [muted])

  useEffect(() => {
    busyRef.current = busy
  }, [busy])

  const nextSequence = () => {
    sequenceRef.current += 1

    return sequenceRef.current
  }

  const sendEvent = useCallback((type: string, payload: Record<string, unknown> = {}) => {
    const socket = socketRef.current

    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return
    }

    socket.send(
      JSON.stringify({
        type,
        session_id: sessionRef.current,
        sequence: nextSequence(),
        payload
      })
    )
  }, [])

  const cleanupInput = useCallback(() => {
    closingInputRef.current = true
    const recorder = recorderRef.current
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop()
    }
    recorderRef.current = null
    if (analyserFrameRef.current) {
      window.cancelAnimationFrame(analyserFrameRef.current)
      analyserFrameRef.current = null
    }
    void audioContextRef.current?.close()
    audioContextRef.current = null
    streamRef.current?.getTracks().forEach(track => track.stop())
    streamRef.current = null
    recorderRef.current = null
    setLevel(0)
  }, [])

  const startRecorder = useCallback(
    (stream: MediaStream) => {
      if (recorderRef.current) {
        return
      }
      closingInputRef.current = false

      const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/ogg'].find(type =>
        MediaRecorder.isTypeSupported(type)
      )
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)

      recorderRef.current = recorder
      stoppingForSilenceRef.current = false
      sentEndOfUtteranceRef.current = false
      silenceStartedAtRef.current = null

      recorder.ondataavailable = event => {
        if (closingInputRef.current) {
          return
        }
        if (event.data.size <= 0) {
          return
        }

        const endOfUtterance = stoppingForSilenceRef.current
        const recorderMimeType = recorder.mimeType
        if (endOfUtterance) {
          sentEndOfUtteranceRef.current = true
        }

        void blobToBase64(event.data).then(data_b64 => {
          sendEvent('audio.input.chunk', realtimeAudioInputPayload({
            dataB64: data_b64,
            endOfUtterance,
            mimeType: recorderMimeType
          }))
        })
      }

      recorder.onstop = () => {
        const isCurrentRecorder = recorderRef.current === recorder
        const stoppedForSilence = stoppingForSilenceRef.current
        if (shouldSendRealtimeVoiceEndMarker({
          closingInput: closingInputRef.current,
          sentEndOfUtterance: sentEndOfUtteranceRef.current,
          stoppedForSilence
        })) {
          sendEvent('audio.input.chunk', realtimeAudioInputPayload({
            dataB64: '',
            endOfUtterance: true,
            mimeType: recorder.mimeType
          }))
          sentEndOfUtteranceRef.current = true
        }
        if (isCurrentRecorder) {
          recorderRef.current = null
        }
        stoppingForSilenceRef.current = false
        sentEndOfUtteranceRef.current = false
        if (!closingInputRef.current && isCurrentRecorder) {
          setStatus('thinking')
        }
      }

      recorder.start(250)
    },
    [sendEvent]
  )

  const advancePlaybackGeneration = useCallback((generation?: unknown) => {
    const next = typeof generation === 'number' && Number.isFinite(generation)
      ? Math.max(playbackGenerationRef.current, generation)
      : playbackGenerationRef.current + 1

    playbackGenerationRef.current = next

    return next
  }, [])

  const stopPlayback = useCallback((advanceGeneration = false) => {
    if (advanceGeneration) {
      advancePlaybackGeneration()
    }
    playbackQueueRef.current = []
    const audio = playingRef.current

    if (audio) {
      audio.pause()
      audio.src = ''
      playingRef.current = null
    }
  }, [advancePlaybackGeneration])

  const playNext = useCallback(() => {
    if (playingRef.current || !playbackQueueRef.current.length) {
      return
    }

    const item = playbackQueueRef.current.shift()
    if (!item) {
      return
    }
    if (item.generation < playbackGenerationRef.current) {
      playNext()

      return
    }

    const url = URL.createObjectURL(item.blob)
    const audio = new Audio(url)

    playingRef.current = audio
    setStatus('speaking')
    audio.onended = () => {
      URL.revokeObjectURL(url)
      playingRef.current = null
      if (playbackQueueRef.current.length) {
        playNext()
      } else {
        setStatus(enabledRef.current && !mutedRef.current ? 'listening' : 'idle')
      }
    }
    audio.onerror = () => {
      URL.revokeObjectURL(url)
      playingRef.current = null
      setStatus('idle')
    }
    void audio.play()
  }, [])

  const enqueueAudio = useCallback(
    (payload: Record<string, unknown>) => {
      const data = typeof payload.data_b64 === 'string' ? payload.data_b64 : ''

      if (!data) {
        return
      }
      const rawGeneration = payload.playback_generation
      const generation =
        typeof rawGeneration === 'number' && Number.isFinite(rawGeneration)
          ? rawGeneration
          : playbackGenerationRef.current

      if (generation < playbackGenerationRef.current) {
        return
      }
      if (generation > playbackGenerationRef.current) {
        stopPlayback()
        playbackGenerationRef.current = generation
      }

      const mimeType =
        typeof payload.mime_type === 'string'
          ? payload.mime_type
          : payload.codec === 'webm_opus'
            ? 'audio/webm;codecs=opus'
            : 'audio/ogg'
      const bytes = bytesFromBase64(data)
      const audioData = new ArrayBuffer(bytes.byteLength)

      new Uint8Array(audioData).set(bytes)

      playbackQueueRef.current.push({ blob: new Blob([audioData], { type: mimeType }), generation })
      playNext()
    },
    [playNext, stopPlayback]
  )

  const stopRecorderForTurn = useCallback(() => {
    const recorder = recorderRef.current

    if (!recorder || recorder.state === 'inactive') {
      return
    }

    stoppingForSilenceRef.current = true
    if (recorder.state === 'recording') {
      recorder.requestData()
    }
    recorder.stop()
  }, [])

  const startMeter = useCallback(
    (stream: MediaStream) => {
      if (analyserFrameRef.current) {
        return
      }

      const audioWindow = window as Window & { webkitAudioContext?: BrowserAudioContext }
      const AudioContextCtor = window.AudioContext || audioWindow.webkitAudioContext

      if (!AudioContextCtor) {
        return
      }

      const audioContext = new AudioContextCtor()
      const analyser = audioContext.createAnalyser()
      const source = audioContext.createMediaStreamSource(stream)
      const data = new Uint8Array(256)

      analyser.fftSize = 256
      source.connect(analyser)
      audioContextRef.current = audioContext

      const tick = () => {
        analyser.getByteTimeDomainData(data)

        let sum = 0

        for (const value of data) {
          const centered = value - 128
          sum += centered * centered
        }

        const normalized = Math.min(1, Math.sqrt(sum / data.length) / 42)
        const now = Date.now()

        setLevel(normalized)

        if (normalized >= SPEECH_LEVEL_THRESHOLD) {
          let acceptSpeech = true

          if (playingRef.current) {
            const gate = updateRealtimeVoiceBargeInGate({
              isSpeechActive: true,
              nowMs: now,
              speechStartedAtMs: bargeInSpeechStartedAtRef.current
            })

            bargeInSpeechStartedAtRef.current = gate.speechStartedAtMs
            acceptSpeech = gate.shouldBargeIn
            if (gate.shouldBargeIn) {
              bargeInSpeechStartedAtRef.current = null
              stopPlayback(true)
              sendEvent('barge_in', { reason: 'user_speech' })
            }
          } else {
            bargeInSpeechStartedAtRef.current = null
          }

          if (acceptSpeech && !recorderRef.current && enabledRef.current && !mutedRef.current && !busyRef.current) {
            startRecorder(stream)
            setStatus('listening')
          }

          if (acceptSpeech) {
            heardSpeechRef.current = true
            silenceStartedAtRef.current = null
          }
        } else if (heardSpeechRef.current) {
          bargeInSpeechStartedAtRef.current = null
          silenceStartedAtRef.current ??= now
          if (now - silenceStartedAtRef.current >= 1250) {
            if (recorderRef.current) {
              stopRecorderForTurn()
            }
            heardSpeechRef.current = false
            silenceStartedAtRef.current = null
          }
        } else {
          bargeInSpeechStartedAtRef.current = null
        }

        analyserFrameRef.current = window.requestAnimationFrame(tick)
      }

      tick()
    },
    [sendEvent, startRecorder, stopPlayback, stopRecorderForTurn]
  )

  const startListening = useCallback(async () => {
    if (!enabledRef.current || mutedRef.current || busyRef.current || recorderRef.current) {
      return
    }

    const permitted = await window.hermesDesktop?.requestMicrophoneAccess?.()

    if (permitted === false) {
      throw new Error('Microphone access denied.')
    }

    const stream =
      streamRef.current ||
      (await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true }
      }))

    streamRef.current = stream
    heardSpeechRef.current = false
    silenceStartedAtRef.current = null

    startMeter(stream)
    startRecorder(stream)
    setStatus('listening')
  }, [startMeter, startRecorder])

  const handleEvent = useCallback(
    (event: VoiceEvent) => {
      if (shouldDropStaleRealtimeVoiceEvent(event, playbackGenerationRef.current)) {
        return
      }

      const eventGeneration = realtimeVoicePlaybackGeneration(event.payload)
      if (
        eventGeneration !== null &&
        eventGeneration > playbackGenerationRef.current &&
        GENERATION_EVENT_TYPES.has(event.type)
      ) {
        stopPlayback()
        playbackGenerationRef.current = eventGeneration
      }

      setCaption(current => collectRealtimeVoiceCaption(current, event))
      setFrontendState(current => collectRealtimeVoiceFrontendState(current, event))
      setMetrics(current => collectRealtimeVoiceMetrics(current, event))

      if (event.type === 'audio.output.chunk') {
        enqueueAudio(event.payload || {})
      } else if (event.type === 'assistant.text.partial') {
        setStatus('speaking')
      } else if (event.type === 'assistant.commit') {
        if (!playingRef.current && !playbackQueueRef.current.length) {
          setStatus(enabledRef.current && !mutedRef.current ? 'listening' : 'idle')
        }
        if (enabledRef.current && !mutedRef.current) {
          void startListening()
        }
      } else if (event.type === 'session.error') {
        notifyError(new Error(String(event.payload?.error || 'Realtime voice failed')), 'Realtime voice failed')
        onFatalError?.()
      } else if (event.type === 'barge_in') {
        stopPlayback()
        advancePlaybackGeneration(event.payload?.playback_generation)
      }
    },
    [advancePlaybackGeneration, enqueueAudio, onFatalError, startListening, stopPlayback]
  )

  const start = useCallback(async () => {
    const preflight = await getRealtimeVoiceStatus().catch(() => null)

    if (preflight && (!preflight.enabled || !preflight.available)) {
      onUnavailable?.()

      return
    }

    sessionRef.current = sessionId || sessionRef.current
    setCaption(null)
    setFrontendState(null)
    setMetrics({})
    const url = await realtimeVoiceUrl(sessionRef.current)
    const socket = new WebSocket(url)

    sequenceRef.current = 0
    sessionStartedRef.current = false
    sessionFailedRef.current = false
    socketRef.current = socket

    let resolveSessionReady: ((ready: boolean) => void) | null = null
    const sessionReady = new Promise<boolean>(resolve => {
      resolveSessionReady = resolve
    })

    socket.onmessage = message => {
      try {
        const event = JSON.parse(String(message.data)) as VoiceEvent
        if (event.type === 'session.started') {
          sessionStartedRef.current = true
          resolveSessionReady?.(true)
          resolveSessionReady = null
        } else if (event.type === 'session.error' && !sessionStartedRef.current) {
          sessionFailedRef.current = true
          resolveSessionReady?.(false)
          resolveSessionReady = null
        }
        handleEvent(event)
      } catch (error) {
        sessionFailedRef.current = true
        resolveSessionReady?.(false)
        resolveSessionReady = null
        notifyError(error, 'Realtime voice failed')
        onFatalError?.()
      }
    }
    socket.onclose = close => {
      const action = realtimeVoiceCloseAction({
        closeCode: close.code,
        enabled: enabledRef.current,
        sessionFailed: sessionFailedRef.current,
        sessionStarted: sessionStartedRef.current
      })
      resolveSessionReady?.(false)
      resolveSessionReady = null
      if (action === 'fallback') {
        onUnavailable?.()
      } else if (action === 'fatal') {
        onFatalError?.()
      }
      setStatus('idle')
    }

    const opened = await new Promise<boolean>(resolve => {
      socket.onopen = () => resolve(true)
      socket.onerror = () => {
        resolve(false)
      }
    })

    if (!opened) {
      onUnavailable?.()
      socket.close()
      if (socketRef.current === socket) {
        socketRef.current = null
      }
      setStatus('idle')
      return
    }

    const ready = await sessionReady
    if (!ready || socketRef.current !== socket || socket.readyState !== WebSocket.OPEN) {
      socket.close()
      if (socketRef.current === socket) {
        socketRef.current = null
      }
      setStatus('idle')
      return
    }

    setMuted(false)
    await startListening()
  }, [handleEvent, onFatalError, onUnavailable, sessionId, startListening])

  const end = useCallback(async () => {
    sendEvent('session.closed', { reason: 'client_closed' })
    stopPlayback()
    cleanupInput()
    socketRef.current?.close(1000, 'client closed')
    socketRef.current = null
    sessionStartedRef.current = false
    sessionFailedRef.current = false
    setCaption(null)
    setFrontendState(null)
    setMuted(false)
    setStatus('idle')
  }, [cleanupInput, sendEvent, stopPlayback])

  const stopTurn = useCallback(() => {
    stopRecorderForTurn()
  }, [stopRecorderForTurn])

  const toggleMute = useCallback(() => {
    setMuted(value => {
      const next = !value

      if (next) {
        cleanupInput()
        setStatus('idle')
      } else if (enabledRef.current) {
        void startListening().catch(error => {
          notifyError(error, 'Could not restart realtime voice')
          onFatalError?.()
        })
      }

      return next
    })
  }, [cleanupInput, onFatalError, startListening])

  useEffect(() => {
    if (enabled) {
      void start().catch(error => {
        notifyError(error, 'Could not start realtime voice')
        onFatalError?.()
      })

      return
    }

    void end()
  }, [enabled, end, onFatalError, start])

  useEffect(() => () => void end(), [end])

  return { caption, end, frontendState, level, metrics, muted, start, status, stopTurn, toggleMute }
}
