import { useCallback, useEffect, useRef, useState } from 'react'
import { resolveGatewayWsUrl } from '@hermes/shared'

import { notifyError } from '@/store/notifications'

import type { ConversationStatus } from './use-voice-conversation'

interface RealtimeVoiceOptions {
  busy: boolean
  enabled: boolean
  onFatalError?: () => void
  onUnavailable?: (state?: RealtimeVoiceFrontendState | null) => void
  sessionId?: null | string
}

export interface VoiceEvent {
  payload?: Record<string, unknown>
  sequence: number
  session_id: string
  timestamp_ms?: number
  type: string
}

interface BinaryVoiceEventPayload extends Record<string, unknown> {
  data_bytes?: Uint8Array
}

interface PlaybackItem {
  blob: Blob
  generation: number
}

interface PreRollItem {
  blob: Blob
  mimeType: string
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

export interface RealtimeVoiceQualityTargetsMs {
  audio_to_partial_transcript_ms?: number
  barge_in_ack_ms?: number
  final_transcript_to_first_audio_ms?: number
  final_transcript_to_first_text_ms?: number
}

export interface RealtimeVoiceConversationQuality {
  barge_in?: boolean
  live_like?: boolean
  mode?: string
  native_s2s?: boolean
  partial_transcripts?: boolean
  reason?: string
  sidecar_verified?: boolean
  streaming_stt?: boolean
  tts?: boolean
  utterance_stt?: boolean
}

export interface RealtimeVoiceLanguageMetadata {
  language?: string
  locale?: string
  script?: string
}

export interface RealtimeVoiceCaption {
  final: boolean
  language?: string
  locale?: string
  speaker: 'assistant' | 'user'
  script?: string
  text: string
  updatedAtMs?: number
}

export interface RealtimeVoiceFrontendState {
  reason?: string
  status: 'degraded' | 'fallback'
  updatedAtMs?: number
}

export interface RealtimeVoiceLanguageSupport {
  best_effort_languages?: boolean
  production_languages?: string[]
  production_scripts?: string[]
  sidecar_languages_are_diagnostics?: boolean
}

export interface RealtimeVoiceOracleJobCapacity {
  active?: number
  cancel_requested?: number
  max_concurrent?: number
  queue_limit?: number
  queued?: number
  running?: number
  waiting_for_approval?: number
}

export interface RealtimeVoiceOracleJobReflexItem {
  approval_reason?: string
  evidence_bundle_id?: string
  evidence_bundle_status?: string
  degraded_reason?: string
  job_id?: string
  ordinal?: number | string
  ordinal_label?: string
  priority?: string
  spoken_status?: string
  state?: string
}

export interface RealtimeVoiceOracleJobSessionState {
  capacity?: RealtimeVoiceOracleJobCapacity
  enabled?: boolean
  error?: string
  reflex?: {
    capacity?: RealtimeVoiceOracleJobCapacity
    jobs?: RealtimeVoiceOracleJobReflexItem[]
    more_jobs?: number
  }
  session_id?: string
}

export interface RealtimeVoiceOracleJobState {
  active_sessions?: number
  capacity?: RealtimeVoiceOracleJobCapacity
  sessions?: RealtimeVoiceOracleJobSessionState[]
}

export interface RealtimeVoiceStatus {
  available: boolean
  enabled: boolean
  engine: string
  barge_in_min_speech_ms?: number
  conversation_quality?: RealtimeVoiceConversationQuality
  input_buffer_limit_bytes?: number
  input_frame_ms?: number
  language_support?: RealtimeVoiceLanguageSupport
  pre_roll_ms?: number
  quality_targets_ms?: RealtimeVoiceQualityTargetsMs
  oracle_job_state?: RealtimeVoiceOracleJobState
  require_live_like?: boolean
  silence_timeout_ms?: number
  speech_level_threshold?: number
  sidecar?: {
    autostart?: boolean
    connect_timeout_seconds?: number
    health?: {
      capabilities?: {
        input_languages?: string[]
        native_s2s?: boolean
        output_languages?: string[]
        scripts?: string[]
        streaming_stt?: boolean
        tts?: boolean
        utterance_stt?: boolean
        vllm_audio_frontend?: boolean
      }
      frontend?: {
        languages?: string[]
        model?: null | string
        provider?: null | string
        scripts?: string[]
      }
      kind?: null | string
      local?: {
        stt?: boolean
        tts?: boolean
      }
      ok?: boolean
    }
    healthy?: boolean | null
    mode?: string
  }
  unavailable_reason?: null | string
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

const DEFAULT_REALTIME_SPEECH_LEVEL_THRESHOLD = 0.075
const MIN_REALTIME_SPEECH_LEVEL_THRESHOLD = 0.005
const MAX_REALTIME_SPEECH_LEVEL_THRESHOLD = 1
const DEFAULT_REALTIME_BARGE_IN_MIN_SPEECH_MS = 120
const MIN_REALTIME_BARGE_IN_MIN_SPEECH_MS = 40
const MAX_REALTIME_BARGE_IN_MIN_SPEECH_MS = 1_000
const MAX_REALTIME_AUDIO_BUFFERED_BYTES = 512 * 1024
const MAX_REALTIME_PLAYBACK_QUEUE_ITEMS = 24
const REALTIME_BINARY_HEADER_BYTES = 4
const REALTIME_BINARY_HEADER_LIMIT = 64 * 1024
const DEFAULT_REALTIME_INPUT_FRAME_MS = 100
const MIN_REALTIME_INPUT_FRAME_MS = 40
const MAX_REALTIME_INPUT_FRAME_MS = 500
const DEFAULT_REALTIME_SILENCE_TIMEOUT_MS = 650
const MIN_REALTIME_SILENCE_TIMEOUT_MS = 250
const MAX_REALTIME_SILENCE_TIMEOUT_MS = 2_000
const DEFAULT_REALTIME_SESSION_READY_TIMEOUT_MS = 12_000
const MIN_REALTIME_SESSION_READY_TIMEOUT_MS = 3_000
const MAX_REALTIME_SESSION_READY_TIMEOUT_MS = 30_000
const DEFAULT_REALTIME_PRE_ROLL_MS = 300
const MIN_REALTIME_PRE_ROLL_MS = 0
const MAX_REALTIME_PRE_ROLL_MS = 1_000
const MAX_REALTIME_RECONNECT_ATTEMPTS = 3
const REALTIME_RECONNECT_BASE_DELAY_MS = 400
const REALTIME_RECONNECT_MAX_DELAY_MS = 2_000
const REALTIME_LANGUAGE_METADATA_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/
const REALTIME_OUTPUT_AUDIO_MIME_TYPE_RE =
  /^audio\/[a-z0-9.+-]+(?:\s*;\s*[a-z0-9._-]+=[a-z0-9._-]+)*$/i
const GENERATION_EVENT_TYPES = new Set([
  'audio.output.chunk',
  'assistant.commit',
  'assistant.text.partial',
  'transcript.final'
])

const REALTIME_SESSION_STATUS: Record<string, ConversationStatus> = {
  assistant_pending: 'thinking',
  closed: 'idle',
  closing: 'idle',
  idle: 'idle',
  listening: 'listening',
  speaking: 'speaking',
  starting: 'idle'
}

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

interface RealtimeBinaryAudioInputFrameOptions {
  audioData: ArrayBuffer
  endOfUtterance: boolean
  mimeType: string
  sequence: number
  sessionId: string
}

interface RealtimeAudioFrameBackpressureInput {
  bufferedAmount: number
  endOfUtterance: boolean
  maxBufferedBytes?: number
}

interface RealtimePlaybackQueueBackpressureInput<T> {
  maxItems?: number
  queue: T[]
}

interface RealtimeTurnCaptureStartInput {
  acceptSpeech: boolean
  busy: boolean
  enabled: boolean
  muted: boolean
  turnCaptureActive: boolean
}

interface RealtimeTurnRecorderRestartInput {
  acceptSpeech: boolean
  hasRecorder: boolean
  streamAvailable: boolean
  turnCaptureActive: boolean
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

interface RealtimeSessionErrorActionInput {
  sessionStarted: boolean
}

interface RealtimeReconnectInput {
  attempts: number
  busy: boolean
  enabled: boolean
  muted: boolean
}

interface RealtimePlaybackQueueActionInput {
  enabled: boolean
  hasQueuedAudio: boolean
  muted: boolean
}

interface RealtimeQueuedAudioInputGeneration {
  activeGeneration: number
  queuedGeneration: number
}

function bytesFromBase64(value: string): Uint8Array {
  const raw = atob(value)
  const bytes = new Uint8Array(raw.length)

  for (let i = 0; i < raw.length; i += 1) {
    bytes[i] = raw.charCodeAt(i)
  }

  return bytes
}

export function queueRealtimeAudioTask(
  previous: Promise<void>,
  task: () => Promise<void>,
  onError: (error: unknown) => void
): Promise<void> {
  return previous
    .catch(() => undefined)
    .then(task)
    .catch(error => {
      onError(error)
    })
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

export function realtimeVoiceOutputAudioMimeType(payload: Record<string, unknown>): string {
  const codec = typeof payload.codec === 'string' ? payload.codec.trim().toLowerCase() : ''
  if (codec === 'pcm16') {
    return 'audio/wav'
  }

  const mimeType = typeof payload.mime_type === 'string' ? payload.mime_type.trim().toLowerCase() : ''

  if (mimeType.length > 0 && mimeType.length <= 128 && REALTIME_OUTPUT_AUDIO_MIME_TYPE_RE.test(mimeType)) {
    return mimeType
  }

  if (codec === 'webm_opus') {
    return 'audio/webm;codecs=opus'
  }

  return 'audio/ogg'
}

function realtimeVoiceAudioSampleRateHz(payload: Record<string, unknown>): number {
  const value = payload.sample_rate_hz

  return typeof value === 'number' && Number.isFinite(value) && value >= 8_000 && value <= 384_000
    ? Math.round(value)
    : 16_000
}

function realtimeVoiceAudioChannels(payload: Record<string, unknown>): number {
  const value = payload.channels

  return typeof value === 'number' && Number.isFinite(value) && value >= 1 && value <= 8
    ? Math.round(value)
    : 1
}

export function realtimeVoiceOutputAudioBlob(payload: Record<string, unknown>, bytes: Uint8Array): Blob {
  const codec = typeof payload.codec === 'string' ? payload.codec.trim().toLowerCase() : ''
  const audioBytes = codec === 'pcm16'
    ? wavContainerForPcm16(
      bytes,
      realtimeVoiceAudioSampleRateHz(payload),
      realtimeVoiceAudioChannels(payload)
    )
    : bytes
  const audioData = new ArrayBuffer(audioBytes.byteLength)

  new Uint8Array(audioData).set(audioBytes)

  return new Blob([audioData], { type: realtimeVoiceOutputAudioMimeType(payload) })
}

export function wavContainerForPcm16(
  pcmBytes: Uint8Array,
  sampleRateHz = 16_000,
  channels = 1
): Uint8Array {
  const headerBytes = 44
  const wav = new Uint8Array(headerBytes + pcmBytes.byteLength)
  const view = new DataView(wav.buffer)
  const writeAscii = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      wav[offset + index] = value.charCodeAt(index)
    }
  }
  const bitsPerSample = 16
  const blockAlign = channels * bitsPerSample / 8

  writeAscii(0, 'RIFF')
  view.setUint32(4, 36 + pcmBytes.byteLength, true)
  writeAscii(8, 'WAVE')
  writeAscii(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, channels, true)
  view.setUint32(24, sampleRateHz, true)
  view.setUint32(28, sampleRateHz * blockAlign, true)
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, bitsPerSample, true)
  writeAscii(36, 'data')
  view.setUint32(40, pcmBytes.byteLength, true)
  wav.set(pcmBytes, headerBytes)

  return wav
}

function finiteNonNegativeMs(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null
}

function finitePositiveMs(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null
}

function realtimeVoiceHasQualityTargetMiss(payload?: Record<string, unknown>): boolean {
  if (Array.isArray(payload?.quality_target_misses) && payload.quality_target_misses.length > 0) {
    return true
  }

  const summary = payload?.quality_summary
  if (!summary || typeof summary !== 'object' || Array.isArray(summary)) {
    return false
  }

  const targetMissCount = (summary as Record<string, unknown>).target_miss_count

  return typeof targetMissCount === 'number' && Number.isFinite(targetMissCount) && targetMissCount > 0
}

function realtimeVoiceLanguageMetadata(payload?: Record<string, unknown>): RealtimeVoiceLanguageMetadata {
  const metadata: RealtimeVoiceLanguageMetadata = {}

  for (const key of ['language', 'locale', 'script'] as const) {
    const value = payload?.[key]

    if (typeof value !== 'string') {
      continue
    }

    const token = value.trim()
    if (REALTIME_LANGUAGE_METADATA_RE.test(token)) {
      metadata[key] = token
    }
  }

  return metadata
}

function mergeRealtimeVoiceLanguageMetadata(
  previous: RealtimeVoiceCaption | null,
  metadata: RealtimeVoiceLanguageMetadata,
  speaker: RealtimeVoiceCaption['speaker']
): RealtimeVoiceLanguageMetadata {
  if (Object.keys(metadata).length > 0 || previous?.speaker !== speaker) {
    return metadata
  }

  return {
    ...(previous.language ? { language: previous.language } : {}),
    ...(previous.locale ? { locale: previous.locale } : {}),
    ...(previous.script ? { script: previous.script } : {})
  }
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

export function realtimeVoiceQualityTargetsFromPayload(raw: unknown): RealtimeVoiceQualityTargetsMs {
  const targets =
    raw && typeof raw === 'object' && !Array.isArray(raw) ? raw as RealtimeVoiceQualityTargetsMs : undefined

  return {
    audio_to_partial_transcript_ms:
      finitePositiveMs(targets?.audio_to_partial_transcript_ms) ?? 300,
    barge_in_ack_ms: finitePositiveMs(targets?.barge_in_ack_ms) ?? 150,
    final_transcript_to_first_audio_ms:
      finitePositiveMs(targets?.final_transcript_to_first_audio_ms) ?? 900,
    final_transcript_to_first_text_ms:
      finitePositiveMs(targets?.final_transcript_to_first_text_ms) ?? 500
  }
}

export function realtimeVoiceQualityTargets(status: RealtimeVoiceStatus | null): RealtimeVoiceQualityTargetsMs {
  return realtimeVoiceQualityTargetsFromPayload(status?.quality_targets_ms)
}

export function updateRealtimeVoiceBargeInGate({
  isSpeechActive,
  minSpeechMs = DEFAULT_REALTIME_BARGE_IN_MIN_SPEECH_MS,
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

export function realtimeBinaryAudioInputFrame({
  audioData,
  endOfUtterance,
  mimeType,
  sequence,
  sessionId
}: RealtimeBinaryAudioInputFrameOptions): ArrayBuffer {
  const payload = realtimeAudioInputPayload({
    dataB64: '',
    endOfUtterance,
    mimeType
  })
  Reflect.deleteProperty(payload, 'data_b64')

  const header = new TextEncoder().encode(JSON.stringify({
    type: 'audio.input.chunk',
    session_id: sessionId,
    sequence,
    payload
  }))
  const frame = new Uint8Array(4 + header.byteLength + audioData.byteLength)
  const view = new DataView(frame.buffer)

  view.setUint32(0, header.byteLength, false)
  frame.set(header, 4)
  frame.set(new Uint8Array(audioData), 4 + header.byteLength)

  return frame.buffer
}

function arrayBufferFromRealtimeMessageData(data: unknown): Promise<ArrayBuffer | null> {
  if (data instanceof ArrayBuffer) {
    return Promise.resolve(data)
  }
  if (ArrayBuffer.isView(data)) {
    const bytes = data as ArrayBufferView
    const copy = new ArrayBuffer(bytes.byteLength)

    new Uint8Array(copy).set(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength))

    return Promise.resolve(copy)
  }
  if (data instanceof Blob) {
    return data.arrayBuffer()
  }

  return Promise.resolve(null)
}

export async function parseRealtimeVoiceServerMessage(data: unknown): Promise<VoiceEvent> {
  if (typeof data === 'string') {
    return JSON.parse(data) as VoiceEvent
  }

  const frame = await arrayBufferFromRealtimeMessageData(data)
  if (!frame || frame.byteLength < REALTIME_BINARY_HEADER_BYTES) {
    throw new Error('Realtime voice message must be JSON text or binary audio')
  }

  const headerLength = new DataView(frame).getUint32(0, false)
  if (headerLength <= 0 || headerLength > REALTIME_BINARY_HEADER_LIMIT) {
    throw new Error('Realtime voice binary header length is invalid')
  }
  const headerEnd = REALTIME_BINARY_HEADER_BYTES + headerLength
  if (frame.byteLength < headerEnd) {
    throw new Error('Realtime voice binary header is truncated')
  }

  const bytes = new Uint8Array(frame)
  const event = JSON.parse(
    new TextDecoder().decode(bytes.slice(REALTIME_BINARY_HEADER_BYTES, headerEnd))
  ) as VoiceEvent
  const payload = (event.payload && typeof event.payload === 'object' ? event.payload : {}) as BinaryVoiceEventPayload

  payload.data_bytes = bytes.slice(headerEnd)
  event.payload = payload

  return event
}

export function shouldSendRealtimeAudioFrame({
  bufferedAmount,
  endOfUtterance,
  maxBufferedBytes = MAX_REALTIME_AUDIO_BUFFERED_BYTES
}: RealtimeAudioFrameBackpressureInput): boolean {
  return endOfUtterance || bufferedAmount <= maxBufferedBytes
}

export function applyRealtimePlaybackQueueBackpressure<T>({
  maxItems = MAX_REALTIME_PLAYBACK_QUEUE_ITEMS,
  queue
}: RealtimePlaybackQueueBackpressureInput<T>): { dropped: number; queue: T[] } {
  const limit = Number.isFinite(maxItems) ? Math.max(0, Math.floor(maxItems)) : 0

  if (queue.length <= limit) {
    return { dropped: 0, queue }
  }

  const dropped = queue.length - limit

  return { dropped, queue: queue.slice(dropped) }
}

export function realtimeVoiceInputFrameMs(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return DEFAULT_REALTIME_INPUT_FRAME_MS
  }

  return Math.min(
    MAX_REALTIME_INPUT_FRAME_MS,
    Math.max(MIN_REALTIME_INPUT_FRAME_MS, Math.round(value))
  )
}

export function realtimeVoiceSilenceTimeoutMs(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return DEFAULT_REALTIME_SILENCE_TIMEOUT_MS
  }

  return Math.min(
    MAX_REALTIME_SILENCE_TIMEOUT_MS,
    Math.max(MIN_REALTIME_SILENCE_TIMEOUT_MS, Math.round(value))
  )
}

export function realtimeVoiceSpeechLevelThreshold(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    return DEFAULT_REALTIME_SPEECH_LEVEL_THRESHOLD
  }

  return Math.min(MAX_REALTIME_SPEECH_LEVEL_THRESHOLD, Math.max(MIN_REALTIME_SPEECH_LEVEL_THRESHOLD, value))
}

export function realtimeVoiceBargeInMinSpeechMs(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    return DEFAULT_REALTIME_BARGE_IN_MIN_SPEECH_MS
  }

  return Math.min(
    MAX_REALTIME_BARGE_IN_MIN_SPEECH_MS,
    Math.max(MIN_REALTIME_BARGE_IN_MIN_SPEECH_MS, Math.round(value))
  )
}

export function realtimeVoicePreRollMs(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    return DEFAULT_REALTIME_PRE_ROLL_MS
  }

  return Math.min(MAX_REALTIME_PRE_ROLL_MS, Math.max(MIN_REALTIME_PRE_ROLL_MS, Math.round(value)))
}

export function realtimeVoiceSessionReadyTimeoutMs(status: RealtimeVoiceStatus | null): number {
  const connectTimeoutSeconds = status?.sidecar?.connect_timeout_seconds
  if (typeof connectTimeoutSeconds !== 'number' || !Number.isFinite(connectTimeoutSeconds)) {
    return DEFAULT_REALTIME_SESSION_READY_TIMEOUT_MS
  }

  return Math.min(
    MAX_REALTIME_SESSION_READY_TIMEOUT_MS,
    Math.max(MIN_REALTIME_SESSION_READY_TIMEOUT_MS, Math.round(connectTimeoutSeconds * 1000 + 2_000))
  )
}

export function realtimeVoicePreRollChunkLimit(
  inputFrameMs: unknown,
  preRollMs: unknown = DEFAULT_REALTIME_PRE_ROLL_MS
): number {
  const durationMs = realtimeVoicePreRollMs(preRollMs)
  if (durationMs <= 0) {
    return 0
  }

  return Math.max(1, Math.ceil(durationMs / realtimeVoiceInputFrameMs(inputFrameMs)))
}

export function shouldStartRealtimeTurnCapture({
  acceptSpeech,
  busy,
  enabled,
  muted,
  turnCaptureActive
}: RealtimeTurnCaptureStartInput): boolean {
  return acceptSpeech && enabled && !muted && !busy && !turnCaptureActive
}

export function shouldRestartRealtimeTurnRecorder({
  acceptSpeech,
  hasRecorder,
  streamAvailable,
  turnCaptureActive
}: RealtimeTurnRecorderRestartInput): boolean {
  return acceptSpeech && streamAvailable && !hasRecorder && !turnCaptureActive
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
  sessionFailed = false
}: RealtimeCloseActionInput): 'fallback' | 'fatal' | 'ignore' {
  if (!enabled || closeCode === 1000 || sessionFailed) {
    return 'ignore'
  }
  return 'fallback'
}

export function realtimeVoiceSessionErrorAction(_input: RealtimeSessionErrorActionInput): 'fallback' | 'fatal' {
  return 'fallback'
}

export function realtimeVoiceFailureFrontendState(
  reason: string,
  updatedAtMs = Date.now()
): RealtimeVoiceFrontendState {
  const cleanReason = reason.trim()

  return {
    reason: cleanReason || 'realtime_voice_failed',
    status: 'fallback',
    updatedAtMs
  }
}

export function shouldReconnectRealtimeVoiceSession({
  attempts,
  busy,
  enabled,
  muted
}: RealtimeReconnectInput): boolean {
  return enabled && !muted && !busy && attempts < MAX_REALTIME_RECONNECT_ATTEMPTS
}

export function realtimeVoiceReconnectDelayMs(
  attempts: number,
  baseDelayMs = REALTIME_RECONNECT_BASE_DELAY_MS,
  maxDelayMs = REALTIME_RECONNECT_MAX_DELAY_MS
): number {
  const cleanAttempts = Math.max(0, Math.floor(Number.isFinite(attempts) ? attempts : 0))
  const delay = Math.max(0, baseDelayMs) * (2 ** cleanAttempts)

  return Math.min(Math.max(0, maxDelayMs), delay)
}

export function realtimeVoiceReconnectFrontendState(
  attempts: number,
  updatedAtMs = Date.now()
): RealtimeVoiceFrontendState {
  return {
    reason: `reconnecting_${Math.max(1, attempts)}`,
    status: 'degraded',
    updatedAtMs
  }
}

export function realtimeVoicePlaybackQueueAction({
  enabled,
  hasQueuedAudio,
  muted
}: RealtimePlaybackQueueActionInput): 'idle' | 'listening' | 'play_next' {
  if (hasQueuedAudio) {
    return 'play_next'
  }

  return enabled && !muted ? 'listening' : 'idle'
}

export function shouldDropQueuedRealtimeAudioInput({
  activeGeneration,
  queuedGeneration
}: RealtimeQueuedAudioInputGeneration): boolean {
  return queuedGeneration !== activeGeneration
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

export function nextRealtimeVoicePlaybackGeneration(current: number, generation?: unknown): number {
  const parsed = realtimeVoicePlaybackGeneration({ playback_generation: generation })

  return parsed !== null ? Math.max(current, parsed) : current + 1
}

export function realtimeVoiceEventGeneration(
  payload: Record<string, unknown> | undefined,
  activeGeneration: number
): number {
  return realtimeVoicePlaybackGeneration(payload) ?? activeGeneration
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
  const metadata = realtimeVoiceLanguageMetadata(event.payload)

  if (event.type === 'transcript.partial') {
    const language = mergeRealtimeVoiceLanguageMetadata(previous, metadata, 'user')

    return text ? { final: false, speaker: 'user', text, updatedAtMs, ...language } : previous
  }
  if (event.type === 'transcript.final') {
    const language = mergeRealtimeVoiceLanguageMetadata(previous, metadata, 'user')

    return text ? { final: true, speaker: 'user', text, updatedAtMs, ...language } : previous
  }
  if (event.type === 'assistant.text.partial') {
    const language = mergeRealtimeVoiceLanguageMetadata(previous, metadata, 'assistant')
    const rawDelta = typeof event.payload?.delta === 'string' ? event.payload.delta : ''
    const hasDelta = rawDelta.length > 0
    if (!text && !hasDelta) {
      return previous
    }
    if (hasDelta) {
      const chunk = previous?.speaker === 'assistant' ? rawDelta : rawDelta.trimStart()
      const prefix = previous?.speaker === 'assistant' ? previous.text : rawText.trimStart()
      const nextText = previous?.speaker === 'assistant' ? `${prefix}${chunk}` : prefix || chunk

      return { final: false, speaker: 'assistant', text: nextText, updatedAtMs, ...language }
    }

    const chunk = previous?.speaker === 'assistant' ? rawText : rawText.trimStart()
    const nextText = previous?.speaker === 'assistant' ? `${previous.text}${chunk}` : chunk

    return { final: false, speaker: 'assistant', text: nextText, updatedAtMs, ...language }
  }
  if (event.type === 'assistant.commit') {
    const language = mergeRealtimeVoiceLanguageMetadata(previous, metadata, 'assistant')

    return text ? { final: true, speaker: 'assistant', text, updatedAtMs, ...language } : previous
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
    if (realtimeVoiceHasQualityTargetMiss(event.payload)) {
      if (previous?.status === 'fallback') {
        return previous
      }

      return {
        reason: 'quality_target_missed',
        status: 'degraded',
        updatedAtMs: finiteNonNegativeMs(event.timestamp_ms) ?? Date.now()
      }
    }

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

export function realtimeVoiceUnavailableFrontendState(
  status: RealtimeVoiceStatus | null,
  updatedAtMs = Date.now()
): RealtimeVoiceFrontendState | null {
  if (!status || (status.enabled && status.available)) {
    return null
  }

  const reason =
    typeof status.unavailable_reason === 'string' && status.unavailable_reason.trim()
      ? status.unavailable_reason.trim()
      : status.enabled
        ? 'unavailable'
        : 'disabled'

  return {
    reason,
    status: 'fallback',
    updatedAtMs
  }
}

export function realtimeVoiceConversationQualityFrontendState(
  status: Pick<RealtimeVoiceStatus, 'available' | 'conversation_quality' | 'enabled'> | null,
  updatedAtMs = Date.now()
): RealtimeVoiceFrontendState | null {
  if (!status || !status.enabled || !status.available) {
    return null
  }

  const quality = status.conversation_quality
  if (!quality || quality.live_like !== false) {
    return null
  }

  const reason = typeof quality.reason === 'string' && quality.reason.trim()
    ? quality.reason.trim()
    : 'not_live_like'

  return {
    reason,
    status: 'degraded',
    updatedAtMs
  }
}

export function realtimeVoiceSessionStatus(event: VoiceEvent): ConversationStatus | null {
  const state = typeof event.payload?.session_state === 'string' ? event.payload.session_state : ''

  return REALTIME_SESSION_STATUS[state] ?? null
}

export function useRealtimeVoiceSession({ busy, enabled, onFatalError, onUnavailable, sessionId }: RealtimeVoiceOptions) {
  const [caption, setCaption] = useState<RealtimeVoiceCaption | null>(null)
  const [frontendState, setFrontendState] = useState<RealtimeVoiceFrontendState | null>(null)
  const [status, setStatus] = useState<ConversationStatus>('idle')
  const [level, setLevel] = useState(0)
  const [muted, setMuted] = useState(false)
  const [metrics, setMetrics] = useState<RealtimeVoiceLatencyMetrics>({})
  const [qualityTargets, setQualityTargets] = useState<RealtimeVoiceQualityTargetsMs>(realtimeVoiceQualityTargets(null))
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
  const turnCaptureActiveRef = useRef(false)
  const preRollChunksRef = useRef<PreRollItem[]>([])
  const sessionStartedRef = useRef(false)
  const sessionFailedRef = useRef(false)
  const playbackQueueRef = useRef<PlaybackItem[]>([])
  const playingRef = useRef<HTMLAudioElement | null>(null)
  const bargeInSpeechStartedAtRef = useRef<number | null>(null)
  const playbackGenerationRef = useRef(0)
  const enabledRef = useRef(enabled)
  const mutedRef = useRef(muted)
  const busyRef = useRef(busy)
  const audioSendChainRef = useRef<Promise<void>>(Promise.resolve())
  const serverEventChainRef = useRef<Promise<void>>(Promise.resolve())
  const audioInputGenerationRef = useRef(0)
  const reconnectAttemptsRef = useRef(0)
  const reconnectTimerRef = useRef<number | null>(null)
  const startRef = useRef<(() => Promise<void>) | null>(null)
  const inputFrameMsRef = useRef(DEFAULT_REALTIME_INPUT_FRAME_MS)
  const silenceTimeoutMsRef = useRef(DEFAULT_REALTIME_SILENCE_TIMEOUT_MS)
  const speechLevelThresholdRef = useRef(DEFAULT_REALTIME_SPEECH_LEVEL_THRESHOLD)
  const bargeInMinSpeechMsRef = useRef(DEFAULT_REALTIME_BARGE_IN_MIN_SPEECH_MS)
  const preRollMsRef = useRef(DEFAULT_REALTIME_PRE_ROLL_MS)

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

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
  }, [])

  const scheduleReconnect = useCallback(() => {
    if (!shouldReconnectRealtimeVoiceSession({
      attempts: reconnectAttemptsRef.current,
      busy: busyRef.current,
      enabled: enabledRef.current,
      muted: mutedRef.current
    })) {
      return false
    }

    reconnectAttemptsRef.current += 1
    const attempt = reconnectAttemptsRef.current

    clearReconnectTimer()
    setFrontendState(realtimeVoiceReconnectFrontendState(attempt))
    reconnectTimerRef.current = window.setTimeout(() => {
      reconnectTimerRef.current = null
      if (!shouldReconnectRealtimeVoiceSession({
        attempts: attempt - 1,
        busy: busyRef.current,
        enabled: enabledRef.current,
        muted: mutedRef.current
      })) {
        return
      }
      const restart = startRef.current
      if (!restart) {
        return
      }
      void restart().catch(error => {
        notifyError(error, 'Could not reconnect realtime voice')
        onFatalError?.()
      })
    }, realtimeVoiceReconnectDelayMs(attempt - 1))

    return true
  }, [clearReconnectTimer, onFatalError])

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

  const queueAudioInput = useCallback(
    (blob: Blob, endOfUtterance: boolean, mimeType: string) => {
      const audioInputGeneration = audioInputGenerationRef.current
      audioSendChainRef.current = queueRealtimeAudioTask(
        audioSendChainRef.current,
        async () => {
          if (shouldDropQueuedRealtimeAudioInput({
            activeGeneration: audioInputGenerationRef.current,
            queuedGeneration: audioInputGeneration
          })) {
            return
          }
          if (closingInputRef.current && !endOfUtterance) {
            return
          }

          const socket = socketRef.current
          if (
            !socket ||
            socket.readyState !== WebSocket.OPEN ||
            !shouldSendRealtimeAudioFrame({
              bufferedAmount: socket.bufferedAmount,
              endOfUtterance
            })
          ) {
            return
          }

          const audioData = blob.size > 0 ? await blob.arrayBuffer() : new ArrayBuffer(0)
          if (shouldDropQueuedRealtimeAudioInput({
            activeGeneration: audioInputGenerationRef.current,
            queuedGeneration: audioInputGeneration
          })) {
            return
          }

          socket.send(realtimeBinaryAudioInputFrame({
            audioData,
            endOfUtterance,
            mimeType,
            sequence: nextSequence(),
            sessionId: sessionRef.current
          }))
        },
        error => {
          notifyError(error, 'Realtime voice failed')
          onFatalError?.()
        }
      )
    },
    [onFatalError]
  )

  const retainPreRollChunk = useCallback((blob: Blob, mimeType: string) => {
    const chunks = preRollChunksRef.current

    chunks.push({ blob, mimeType })
    chunks.splice(
      0,
      Math.max(0, chunks.length - realtimeVoicePreRollChunkLimit(inputFrameMsRef.current, preRollMsRef.current))
    )
  }, [])

  const flushPreRollChunks = useCallback(() => {
    const chunks = preRollChunksRef.current

    preRollChunksRef.current = []
    for (const chunk of chunks) {
      queueAudioInput(chunk.blob, false, chunk.mimeType)
    }
  }, [queueAudioInput])

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
    turnCaptureActiveRef.current = false
    preRollChunksRef.current = []
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
      turnCaptureActiveRef.current = false
      preRollChunksRef.current = []
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
          queueAudioInput(event.data, endOfUtterance, recorderMimeType)
          return
        }

        if (!turnCaptureActiveRef.current) {
          retainPreRollChunk(event.data, recorderMimeType)
          return
        }

        queueAudioInput(event.data, endOfUtterance, recorderMimeType)
      }

      recorder.onstop = () => {
        const isCurrentRecorder = recorderRef.current === recorder
        const stoppedForSilence = stoppingForSilenceRef.current
        if (shouldSendRealtimeVoiceEndMarker({
          closingInput: closingInputRef.current,
          sentEndOfUtterance: sentEndOfUtteranceRef.current,
          stoppedForSilence
        })) {
          queueAudioInput(new Blob(), true, recorder.mimeType)
          sentEndOfUtteranceRef.current = true
        }
        if (isCurrentRecorder) {
          recorderRef.current = null
        }
        stoppingForSilenceRef.current = false
        sentEndOfUtteranceRef.current = false
        turnCaptureActiveRef.current = false
        preRollChunksRef.current = []
        if (!closingInputRef.current && isCurrentRecorder) {
          setStatus('thinking')
        }
      }

      recorder.start(inputFrameMsRef.current)
    },
    [queueAudioInput, retainPreRollChunk]
  )

  const advancePlaybackGeneration = useCallback((generation?: unknown) => {
    const next = nextRealtimeVoicePlaybackGeneration(playbackGenerationRef.current, generation)

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

  const resetTransportForRecoverableFailure = useCallback(() => {
    stopPlayback(true)
    cleanupInput()
    audioInputGenerationRef.current += 1
    audioSendChainRef.current = Promise.resolve()
    serverEventChainRef.current = Promise.resolve()
    heardSpeechRef.current = false
    silenceStartedAtRef.current = null
    sentEndOfUtteranceRef.current = false
    stoppingForSilenceRef.current = false
    bargeInSpeechStartedAtRef.current = null
  }, [cleanupInput, stopPlayback])

  const handleRecoverableFailure = useCallback(
    (reason: string, updatedAtMs = Date.now()) => {
      resetTransportForRecoverableFailure()
      if (scheduleReconnect()) {
        return true
      }

      const state = realtimeVoiceFailureFrontendState(reason, updatedAtMs)
      setFrontendState(state)
      onUnavailable?.(state)

      return false
    },
    [onUnavailable, resetTransportForRecoverableFailure, scheduleReconnect]
  )

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
    let settled = false

    playingRef.current = audio
    setStatus('speaking')

    const settle = () => {
      if (settled) {
        return
      }
      settled = true
      URL.revokeObjectURL(url)
      if (playingRef.current === audio) {
        playingRef.current = null
      }

      const action = realtimeVoicePlaybackQueueAction({
        enabled: enabledRef.current,
        hasQueuedAudio: Boolean(playbackQueueRef.current.length),
        muted: mutedRef.current
      })
      if (action === 'play_next') {
        playNext()
      } else {
        setStatus(action)
      }
    }

    audio.onended = settle
    audio.onerror = settle
    void audio.play().catch(settle)
  }, [])

  const enqueueAudio = useCallback(
    (payload: Record<string, unknown>) => {
      const binaryData = payload.data_bytes instanceof Uint8Array ? payload.data_bytes : null
      const data = typeof payload.data_b64 === 'string' ? payload.data_b64 : ''

      if (!binaryData && !data) {
        return
      }
      const generation = realtimeVoiceEventGeneration(payload, playbackGenerationRef.current)

      if (generation < playbackGenerationRef.current) {
        return
      }
      if (generation > playbackGenerationRef.current) {
        stopPlayback()
        playbackGenerationRef.current = generation
      }

      const bytes = binaryData ?? bytesFromBase64(data)

      playbackQueueRef.current.push({ blob: realtimeVoiceOutputAudioBlob(payload, bytes), generation })
      const backpressure = applyRealtimePlaybackQueueBackpressure({ queue: playbackQueueRef.current })

      if (backpressure.dropped > 0) {
        playbackQueueRef.current = backpressure.queue
        setFrontendState({
          reason: 'playback_queue_backpressure',
          status: 'degraded',
          updatedAtMs: Date.now()
        })
      }
      playNext()
    },
    [playNext, stopPlayback]
  )

  const stopRecorderForTurn = useCallback(() => {
    const recorder = recorderRef.current

    if (!recorder || recorder.state === 'inactive' || !turnCaptureActiveRef.current) {
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

        if (normalized >= speechLevelThresholdRef.current) {
          let acceptSpeech = true

          if (playingRef.current) {
            const gate = updateRealtimeVoiceBargeInGate({
              isSpeechActive: true,
              minSpeechMs: bargeInMinSpeechMsRef.current,
              nowMs: now,
              speechStartedAtMs: bargeInSpeechStartedAtRef.current
            })

            bargeInSpeechStartedAtRef.current = gate.speechStartedAtMs
            acceptSpeech = gate.shouldBargeIn
            if (gate.shouldBargeIn) {
              bargeInSpeechStartedAtRef.current = null
              audioInputGenerationRef.current += 1
              stopPlayback(true)
              sendEvent('barge_in', { reason: 'user_speech' })
            }
          } else {
            bargeInSpeechStartedAtRef.current = null
          }

          if (shouldStartRealtimeTurnCapture({
            acceptSpeech,
            busy: busyRef.current,
            enabled: enabledRef.current,
            muted: mutedRef.current,
            turnCaptureActive: turnCaptureActiveRef.current
          })) {
            if (shouldRestartRealtimeTurnRecorder({
              acceptSpeech,
              hasRecorder: Boolean(recorderRef.current),
              streamAvailable: Boolean(streamRef.current),
              turnCaptureActive: turnCaptureActiveRef.current
            }) && streamRef.current) {
              startRecorder(streamRef.current)
            }
            turnCaptureActiveRef.current = true
            flushPreRollChunks()
            setStatus('listening')
          }

          if (acceptSpeech) {
            heardSpeechRef.current = true
            silenceStartedAtRef.current = null
          }
        } else if (heardSpeechRef.current) {
          bargeInSpeechStartedAtRef.current = null
          silenceStartedAtRef.current ??= now
          if (now - silenceStartedAtRef.current >= silenceTimeoutMsRef.current) {
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

      const sessionStatus = realtimeVoiceSessionStatus(event)
      if (sessionStatus) {
        const hasPlayback = Boolean(playingRef.current || playbackQueueRef.current.length)
        if (sessionStatus !== 'listening' || event.type !== 'assistant.commit' || !hasPlayback) {
          setStatus(sessionStatus)
        }
      }

      if (event.type === 'audio.output.chunk') {
        enqueueAudio(event.payload || {})
      } else if (event.type === 'session.started') {
        setQualityTargets(realtimeVoiceQualityTargetsFromPayload(event.payload?.quality_targets_ms))
        const qualityState = realtimeVoiceConversationQualityFrontendState({
          available: true,
          conversation_quality: event.payload?.conversation_quality as RealtimeVoiceConversationQuality | undefined,
          enabled: true
        }, finiteNonNegativeMs(event.timestamp_ms) ?? Date.now())
        if (qualityState) {
          setFrontendState(current => current ?? qualityState)
        }
      } else if (event.type === 'assistant.text.partial') {
        if (!sessionStatus) {
          setStatus('speaking')
        }
      } else if (event.type === 'assistant.commit') {
        if (!sessionStatus && !playingRef.current && !playbackQueueRef.current.length) {
          setStatus(enabledRef.current && !mutedRef.current ? 'listening' : 'idle')
        }
        if (enabledRef.current && !mutedRef.current) {
          void startListening()
        }
      } else if (event.type === 'session.error') {
        notifyError(new Error(String(event.payload?.error || 'Realtime voice failed')), 'Realtime voice failed')
        if (realtimeVoiceSessionErrorAction({ sessionStarted: sessionStartedRef.current }) === 'fallback') {
          sessionFailedRef.current = true
          handleRecoverableFailure(
            'session_error',
            finiteNonNegativeMs(event.timestamp_ms) ?? Date.now()
          )
          socketRef.current?.close(1011, 'session error')
        } else {
          onFatalError?.()
        }
      } else if (event.type === 'barge_in') {
        stopPlayback()
        advancePlaybackGeneration(event.payload?.playback_generation)
      }
    },
    [advancePlaybackGeneration, enqueueAudio, handleRecoverableFailure, onFatalError, startListening, stopPlayback]
  )

  const start = useCallback(async () => {
    clearReconnectTimer()
    const preflight = await getRealtimeVoiceStatus().catch(() => null)
    const unavailableState = realtimeVoiceUnavailableFrontendState(preflight)

    if (unavailableState) {
      setFrontendState(unavailableState)
      onUnavailable?.(unavailableState)

      return
    }
    const initialFrontendState = realtimeVoiceConversationQualityFrontendState(preflight)
    inputFrameMsRef.current = realtimeVoiceInputFrameMs(preflight?.input_frame_ms)
    silenceTimeoutMsRef.current = realtimeVoiceSilenceTimeoutMs(preflight?.silence_timeout_ms)
    speechLevelThresholdRef.current = realtimeVoiceSpeechLevelThreshold(preflight?.speech_level_threshold)
    bargeInMinSpeechMsRef.current = realtimeVoiceBargeInMinSpeechMs(preflight?.barge_in_min_speech_ms)
    preRollMsRef.current = realtimeVoicePreRollMs(preflight?.pre_roll_ms)
    setQualityTargets(realtimeVoiceQualityTargets(preflight))
    const sessionReadyTimeoutMs = realtimeVoiceSessionReadyTimeoutMs(preflight)

    sessionRef.current = sessionId || sessionRef.current
    setCaption(null)
    setFrontendState(initialFrontendState)
    setMetrics({})
    const url = await realtimeVoiceUrl(sessionRef.current)
    const socket = new WebSocket(url)

    sequenceRef.current = 0
    sessionStartedRef.current = false
    sessionFailedRef.current = false
    audioSendChainRef.current = Promise.resolve()
    serverEventChainRef.current = Promise.resolve()
    audioInputGenerationRef.current += 1
    socketRef.current = socket

    let resolveSessionReady: ((ready: boolean) => void) | null = null
    const sessionReady = new Promise<boolean>(resolve => {
      resolveSessionReady = resolve
    })

    socket.onmessage = message => {
      serverEventChainRef.current = queueRealtimeAudioTask(
        serverEventChainRef.current,
        async () => {
          const event = await parseRealtimeVoiceServerMessage(message.data)

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
        },
        error => {
          sessionFailedRef.current = true
          resolveSessionReady?.(false)
          resolveSessionReady = null
          notifyError(error, 'Realtime voice failed')
          onFatalError?.()
        }
      )
    }
    socket.onclose = close => {
      if (socketRef.current !== socket) {
        return
      }
      socketRef.current = null

      const action = realtimeVoiceCloseAction({
        closeCode: close.code,
        enabled: enabledRef.current,
        sessionFailed: sessionFailedRef.current,
        sessionStarted: sessionStartedRef.current
      })
      resolveSessionReady?.(false)
      resolveSessionReady = null
      if (action === 'fallback') {
        handleRecoverableFailure('websocket_closed')
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
      handleRecoverableFailure('websocket_open_failed')
      socket.close()
      if (socketRef.current === socket) {
        socketRef.current = null
      }
      setStatus('idle')
      return
    }

    let sessionReadyTimedOut = false
    const ready = await Promise.race([
      sessionReady,
      new Promise<boolean>(resolve => {
        window.setTimeout(() => {
          sessionReadyTimedOut = true
          resolve(false)
        }, sessionReadyTimeoutMs)
      })
    ])
    if (!ready || socketRef.current !== socket || socket.readyState !== WebSocket.OPEN) {
      socket.close()
      if (socketRef.current === socket) {
        socketRef.current = null
      }
      if (sessionReadyTimedOut) {
        handleRecoverableFailure('session_ready_timeout')
      } else if (!sessionFailedRef.current) {
        handleRecoverableFailure('session_not_ready')
      }
      setStatus('idle')
      return
    }

    setMuted(false)
    try {
      await startListening()
      reconnectAttemptsRef.current = 0
    } catch (error) {
      socket.close(1000, 'microphone unavailable')
      if (socketRef.current === socket) {
        socketRef.current = null
      }
      setStatus('idle')
      throw error
    }
  }, [clearReconnectTimer, handleEvent, handleRecoverableFailure, onFatalError, onUnavailable, sessionId, startListening])

  const end = useCallback(async () => {
    clearReconnectTimer()
    sendEvent('session.closed', { reason: 'client_closed' })
    stopPlayback()
    cleanupInput()
    socketRef.current?.close(1000, 'client closed')
    socketRef.current = null
    sessionStartedRef.current = false
    sessionFailedRef.current = false
    reconnectAttemptsRef.current = 0
    audioSendChainRef.current = Promise.resolve()
    audioInputGenerationRef.current += 1
    setCaption(null)
    setFrontendState(null)
    setQualityTargets(realtimeVoiceQualityTargets(null))
    setMuted(false)
    setStatus('idle')
  }, [cleanupInput, clearReconnectTimer, sendEvent, stopPlayback])

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
    startRef.current = start
  }, [start])

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

  return { caption, end, frontendState, level, metrics, muted, qualityTargets, start, status, stopTurn, toggleMute }
}
