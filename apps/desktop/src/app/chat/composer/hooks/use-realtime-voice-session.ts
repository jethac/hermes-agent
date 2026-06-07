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

interface VoiceEvent {
  payload?: Record<string, unknown>
  sequence: number
  session_id: string
  timestamp_ms?: number
  type: string
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

export function useRealtimeVoiceSession({ busy, enabled, onFatalError, onUnavailable, sessionId }: RealtimeVoiceOptions) {
  const [status, setStatus] = useState<ConversationStatus>('idle')
  const [level, setLevel] = useState(0)
  const [muted, setMuted] = useState(false)
  const socketRef = useRef<WebSocket | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserFrameRef = useRef<number | null>(null)
  const sequenceRef = useRef(0)
  const sessionRef = useRef(`voice-${Math.random().toString(36).slice(2)}`)
  const heardSpeechRef = useRef(false)
  const silenceStartedAtRef = useRef<number | null>(null)
  const stoppingForSilenceRef = useRef(false)
  const playbackQueueRef = useRef<Blob[]>([])
  const playingRef = useRef<HTMLAudioElement | null>(null)
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

  const stopPlayback = useCallback(() => {
    playbackQueueRef.current = []
    const audio = playingRef.current

    if (audio) {
      audio.pause()
      audio.src = ''
      playingRef.current = null
    }
  }, [])

  const playNext = useCallback(() => {
    if (playingRef.current || !playbackQueueRef.current.length) {
      return
    }

    const blob = playbackQueueRef.current.shift()
    if (!blob) {
      return
    }

    const url = URL.createObjectURL(blob)
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

      const mimeType =
        typeof payload.mime_type === 'string'
          ? payload.mime_type
          : payload.codec === 'webm_opus'
            ? 'audio/webm;codecs=opus'
            : 'audio/ogg'
      const bytes = bytesFromBase64(data)
      const audioData = new ArrayBuffer(bytes.byteLength)

      new Uint8Array(audioData).set(bytes)

      playbackQueueRef.current.push(new Blob([audioData], { type: mimeType }))
      playNext()
    },
    [playNext]
  )

  const stopRecorderForTurn = useCallback(() => {
    const recorder = recorderRef.current

    if (!recorder || recorder.state === 'inactive') {
      return
    }

    stoppingForSilenceRef.current = true
    recorder.stop()
  }, [])

  const startMeter = useCallback(
    (stream: MediaStream) => {
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

        if (normalized >= 0.075) {
          if (playingRef.current) {
            stopPlayback()
            sendEvent('barge_in', { reason: 'user_speech' })
          }
          heardSpeechRef.current = true
          silenceStartedAtRef.current = null
        } else if (heardSpeechRef.current) {
          silenceStartedAtRef.current ??= now
          if (now - silenceStartedAtRef.current >= 1250) {
            stopRecorderForTurn()

            return
          }
        }

        analyserFrameRef.current = window.requestAnimationFrame(tick)
      }

      tick()
    },
    [sendEvent, stopPlayback, stopRecorderForTurn]
  )

  const startListening = useCallback(async () => {
    if (!enabledRef.current || mutedRef.current || busyRef.current || recorderRef.current) {
      return
    }

    const permitted = await window.hermesDesktop?.requestMicrophoneAccess?.()

    if (permitted === false) {
      throw new Error('Microphone access denied.')
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true }
    })
    const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/ogg'].find(type =>
      MediaRecorder.isTypeSupported(type)
    )
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)

    streamRef.current = stream
    recorderRef.current = recorder
    heardSpeechRef.current = false
    stoppingForSilenceRef.current = false
    silenceStartedAtRef.current = null

    recorder.ondataavailable = event => {
      if (event.data.size <= 0) {
        return
      }

      void blobToBase64(event.data).then(data_b64 => {
        sendEvent('audio.input.chunk', {
          codec: recorder.mimeType.includes('ogg') ? 'opus' : 'webm_opus',
          sample_rate_hz: 16000,
          channels: 1,
          data_b64,
          end_of_utterance: stoppingForSilenceRef.current
        })
      })
    }

    recorder.onstop = () => {
      cleanupInput()
      setStatus('thinking')
    }

    recorder.start(250)
    startMeter(stream)
    setStatus('listening')
  }, [cleanupInput, sendEvent, startMeter])

  const handleEvent = useCallback(
    (event: VoiceEvent) => {
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
      }
    },
    [enqueueAudio, onFatalError, startListening]
  )

  const start = useCallback(async () => {
    const preflight = await getRealtimeVoiceStatus().catch(() => null)

    if (preflight && (!preflight.enabled || !preflight.available)) {
      onUnavailable?.()

      return
    }

    sessionRef.current = sessionId || sessionRef.current
    const url = await realtimeVoiceUrl(sessionRef.current)
    const socket = new WebSocket(url)

    sequenceRef.current = 0
    socketRef.current = socket

    socket.onmessage = message => {
      try {
        const event = JSON.parse(String(message.data)) as VoiceEvent
        handleEvent(event)
      } catch (error) {
        notifyError(error, 'Realtime voice failed')
        onFatalError?.()
      }
    }
    socket.onclose = close => {
      if (enabledRef.current && close.code !== 1000) {
        onFatalError?.()
      }
      setStatus('idle')
    }

    await new Promise<void>((resolve, reject) => {
      socket.onopen = () => resolve()
      socket.onerror = () => {
        onFatalError?.()
        reject(new Error('Could not start realtime voice session'))
      }
    })

    setMuted(false)
    await startListening()
  }, [handleEvent, onFatalError, onUnavailable, sessionId, startListening])

  const end = useCallback(async () => {
    sendEvent('session.closed', { reason: 'client_closed' })
    stopPlayback()
    cleanupInput()
    socketRef.current?.close(1000, 'client closed')
    socketRef.current = null
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

  return { end, level, muted, start, status, stopTurn, toggleMute }
}
