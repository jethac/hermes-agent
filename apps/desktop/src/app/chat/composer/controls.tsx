import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { KbdCombo } from '@/components/ui/kbd'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { AudioLines, Layers3, Loader2, Square, SteeringWheel } from '@/lib/icons'
import { formatCombo } from '@/lib/keybinds/combo'
import { cn } from '@/lib/utils'

import type { ConversationStatus } from './hooks/use-voice-conversation'
import type {
  RealtimeVoiceCaption,
  RealtimeVoiceFrontendState,
  RealtimeVoiceLatencyMetrics,
  RealtimeVoiceQualityTargetsMs
} from './hooks/use-realtime-voice-session'
import { ModelPill } from './model-pill'
import type { ChatBarState, VoiceStatus } from './types'

export const ICON_BTN = 'size-(--composer-control-size) shrink-0 rounded-md'
export const GHOST_ICON_BTN = cn(
  ICON_BTN,
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
)
// Send/voice-conversation primary: solid foreground-on-background circle
// (reads as black-on-white in light mode, white-on-black in dark mode) to
// match the reference composer's high-contrast CTA. Keeps the pill itself
// neutral and lets the action visually dominate the row.
export const PRIMARY_ICON_BTN = cn(
  'size-(--composer-control-primary-size,var(--composer-control-size)) shrink-0 rounded-full p-0',
  'bg-foreground text-background hover:bg-foreground/90',
  'disabled:bg-foreground/30 disabled:text-background disabled:opacity-100'
)

interface ConversationProps {
  active: boolean
  caption?: null | RealtimeVoiceCaption
  frontendState?: null | RealtimeVoiceFrontendState
  level: number
  metrics?: RealtimeVoiceLatencyMetrics
  muted: boolean
  qualityTargets?: RealtimeVoiceQualityTargetsMs
  status: ConversationStatus
  onEnd: () => void
  onStart: () => void
  onStopTurn: () => void
  onToggleMute: () => void
}

export function ComposerControls({
  busy,
  busyAction,
  canSteer,
  canSubmit,
  compactModelPill = false,
  conversation,
  disabled,
  hasComposerPayload,
  state,
  voiceStatus,
  onDictate,
  onSteer
}: {
  busy: boolean
  busyAction: 'queue' | 'stop'
  canSteer: boolean
  canSubmit: boolean
  compactModelPill?: boolean
  conversation: ConversationProps
  disabled: boolean
  hasComposerPayload: boolean
  state: ChatBarState
  voiceStatus: VoiceStatus
  onDictate: () => void
  onSteer: () => void
}) {
  const { t } = useI18n()
  const c = t.composer
  const steerCombo = formatCombo('mod+enter')
  const steerLabel = `${c.steer} (${steerCombo})`

  const steerTip = (
    <span className="inline-flex items-center gap-1.5">
      {c.steer}
      <KbdCombo combo="mod+enter" size="sm" variant="inverted" />
    </span>
  )

  if (conversation.active) {
    return <ConversationPill {...conversation} disabled={disabled} />
  }

  const showVoicePrimary = !busy && !hasComposerPayload

  return (
    <div className="ml-auto flex shrink-0 items-center gap-(--composer-control-gap)">
      <ModelPill compact={compactModelPill} disabled={disabled} model={state.model} />
      {/* While the agent runs and the user is typing, steer takes over the mic's
          slot rather than crowding the row with an extra button. */}
      {canSteer ? (
        <Tip label={steerTip}>
          <Button
            aria-label={steerLabel}
            className={GHOST_ICON_BTN}
            disabled={disabled}
            onClick={onSteer}
            size="icon"
            type="button"
            variant="ghost"
          >
            <SteeringWheel size={14} />
          </Button>
        </Tip>
      ) : (
        <DictationButton disabled={disabled} onToggle={onDictate} state={state.voice} status={voiceStatus} />
      )}
      {showVoicePrimary ? (
        <Tip label={c.startVoice}>
          <Button
            aria-label={c.startVoice}
            className={PRIMARY_ICON_BTN}
            disabled={disabled}
            onClick={() => {
              triggerHaptic('open')
              conversation.onStart()
            }}
            size="icon"
            type="button"
          >
            <AudioLines size={15} />
          </Button>
        </Tip>
      ) : (
        <Tip label={busy ? (busyAction === 'queue' ? c.queueMessage : c.stop) : c.send}>
          <Button
            aria-label={busy ? (busyAction === 'queue' ? c.queueMessage : c.stop) : c.send}
            className={PRIMARY_ICON_BTN}
            disabled={disabled || !canSubmit}
            type="submit"
          >
            {busy ? (
              busyAction === 'queue' ? (
                <Layers3 size={14} />
              ) : (
                <span className="block size-2.5 rounded-[0.1875rem] bg-current" />
              )
            ) : (
              <Codicon name="arrow-up" size="0.875rem" />
            )}
          </Button>
        </Tip>
      )}
    </div>
  )
}

function ConversationPill({
  caption,
  disabled,
  frontendState,
  level,
  metrics,
  muted,
  qualityTargets,
  onEnd,
  onStopTurn,
  onToggleMute,
  status
}: ConversationProps & { disabled: boolean }) {
  const { t } = useI18n()
  const c = t.composer
  const speaking = status === 'speaking'
  const listening = status === 'listening' && !muted
  const quality = realtimeVoiceQuality(metrics, qualityTargets)

  const label =
    status === 'speaking'
      ? c.speaking
      : status === 'transcribing'
        ? c.transcribing
        : status === 'thinking'
          ? c.thinking
          : muted
            ? c.muted
            : c.listening

  return (
    <div className="ml-auto flex shrink-0 items-center gap-(--composer-control-gap)">
      <Tip label={muted ? c.unmuteMic : c.muteMic}>
        <Button
          aria-label={muted ? c.unmuteMic : c.muteMic}
          aria-pressed={muted}
          className={cn(GHOST_ICON_BTN, 'p-0', muted && 'bg-muted text-muted-foreground')}
          disabled={disabled}
          onClick={() => {
            triggerHaptic('selection')
            onToggleMute()
          }}
          size="icon"
          type="button"
          variant="ghost"
        >
          <Codicon name={muted ? 'mic-off' : 'mic'} size="1rem" />
        </Button>
      </Tip>
      {listening && (
        <Button
          aria-label={c.stopListening}
          className="h-(--composer-control-size) shrink-0 gap-1.5 rounded-full px-2.5 text-xs text-muted-foreground hover:bg-accent hover:text-foreground"
          disabled={disabled}
          onClick={() => {
            triggerHaptic('submit')
            onStopTurn()
          }}
          title={c.stopListening}
          type="button"
          variant="ghost"
        >
          <Square className="fill-current" size={11} />
          <span>{c.stopShort}</span>
        </Button>
      )}
      {caption?.text && <RealtimeVoiceCaptionPill caption={caption} />}
      {frontendState && <RealtimeVoiceFrontendStatePill state={frontendState} />}
      {quality && <RealtimeVoiceQualityPill quality={quality} />}
      <Button
        aria-label={c.endConversation}
        className="h-(--composer-control-size) gap-1.5 rounded-full bg-primary px-3 text-xs font-medium text-primary-foreground hover:bg-primary/90"
        disabled={disabled}
        onClick={() => {
          triggerHaptic('close')
          onEnd()
        }}
        title={c.endConversation}
        type="button"
      >
        <ConversationIndicator level={level} listening={listening} speaking={speaking} />
        <span>{c.endShort}</span>
      </Button>
      <span className="sr-only" role="status">
        {label}
      </span>
    </div>
  )
}

function RealtimeVoiceCaptionPill({ caption }: { caption: RealtimeVoiceCaption }) {
  return (
    <Tip label={caption.text}>
      <span
        aria-label={`Realtime voice ${caption.speaker} caption: ${caption.text}`}
        className={cn(
          'hidden h-(--composer-control-size) max-w-[min(22rem,34vw)] shrink min-w-0 items-center rounded-full border border-border/55 bg-muted/45 px-2.5 text-xs text-muted-foreground sm:inline-flex',
          !caption.final && 'italic text-muted-foreground/75'
        )}
      >
        <span className="truncate">{caption.text}</span>
      </span>
    </Tip>
  )
}

function RealtimeVoiceFrontendStatePill({ state }: { state: RealtimeVoiceFrontendState }) {
  const label = state.status === 'fallback' ? 'Fallback' : 'Degraded'
  const reason = state.reason ? state.reason.replace(/_/g, ' ') : label

  return (
    <Tip label={reason}>
      <span
        aria-label={`Realtime voice ${label.toLowerCase()}: ${reason}`}
        className="hidden h-(--composer-control-size) shrink-0 items-center rounded-full border border-amber-500/30 bg-amber-500/10 px-2 text-[0.65rem] font-medium uppercase text-amber-700 sm:inline-flex dark:text-amber-300"
      >
        {label}
      </span>
    </Tip>
  )
}

export interface RealtimeVoiceQuality {
  primaryMs: number
  state: 'good' | 'slow'
  tooltip: Array<{ label: string; ms: number; targetMs: number }>
}

export function realtimeVoiceQuality(
  metrics?: RealtimeVoiceLatencyMetrics,
  targets?: RealtimeVoiceQualityTargetsMs
): RealtimeVoiceQuality | null {
  const rows = [
    metricRow('ASR', metrics?.audioToPartialTranscriptMs, targets?.audio_to_partial_transcript_ms ?? 300),
    metricRow('Audio', metrics?.finalTranscriptToFirstAudioMs, targets?.final_transcript_to_first_audio_ms ?? 900),
    metricRow('Barge', metrics?.bargeInAckMs, targets?.barge_in_ack_ms ?? 150)
  ].filter((row): row is RealtimeVoiceQuality['tooltip'][number] => row !== null)

  if (!rows.length) {
    return null
  }

  const primary = rows.find(row => row.label === 'Audio') || rows[0]

  return {
    primaryMs: primary.ms,
    state: rows.some(row => row.ms > row.targetMs) ? 'slow' : 'good',
    tooltip: rows
  }
}

function metricRow(label: string, ms: number | undefined, targetMs: number) {
  if (typeof ms !== 'number' || !Number.isFinite(ms) || ms < 0) {
    return null
  }

  return { label, ms, targetMs }
}

function formatLatency(ms: number) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`
}

function RealtimeVoiceQualityPill({ quality }: { quality: RealtimeVoiceQuality }) {
  return (
    <Tip
      label={
        <div className="space-y-1 py-0.5 font-mono font-normal leading-4">
          {quality.tooltip.map(row => (
            <div className="flex items-center justify-between gap-3" key={row.label}>
              <span>{row.label}</span>
              <span>
                {formatLatency(row.ms)} / {formatLatency(row.targetMs)}
              </span>
            </div>
          ))}
        </div>
      }
    >
      <span
        aria-label={`Realtime voice latency ${formatLatency(quality.primaryMs)}`}
        className={cn(
          'inline-flex h-(--composer-control-size) shrink-0 items-center gap-1 rounded-full border px-2 font-mono text-[0.65rem]',
          quality.state === 'good'
            ? 'border-emerald-500/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
            : 'border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-300'
        )}
      >
        <span className="text-[0.58rem] font-semibold opacity-75">RT</span>
        <span>{formatLatency(quality.primaryMs)}</span>
      </span>
    </Tip>
  )
}

function ConversationIndicator({
  level,
  listening,
  speaking
}: {
  level: number
  listening: boolean
  speaking: boolean
}) {
  if (speaking) {
    return <Loader2 className="animate-spin" size={12} />
  }

  const bars = [0.55, 0.85, 1, 0.85, 0.55]
  const normalized = Math.max(0, Math.min(level, 1))

  return (
    <span aria-hidden="true" className="flex h-3 items-center gap-0.5">
      {bars.map((weight, index) => {
        const height = listening ? 0.3 + Math.min(0.7, normalized * weight) : 0.3

        return <span className="w-0.5 rounded-full bg-current" key={index} style={{ height: `${height * 100}%` }} />
      })}
    </span>
  )
}

function DictationButton({
  disabled,
  state,
  status,
  onToggle
}: {
  disabled: boolean
  state: ChatBarState['voice']
  status: VoiceStatus
  onToggle: () => void
}) {
  const { t } = useI18n()
  const c = t.composer
  const active = state.active || status !== 'idle'

  const aria =
    status === 'recording' ? c.stopDictation : status === 'transcribing' ? c.transcribingDictation : c.voiceDictation

  return (
    <Tip label={aria}>
      <Button
        aria-label={aria}
        aria-pressed={active}
        className={cn(
          GHOST_ICON_BTN,
          'p-0',
          'data-[active=true]:bg-accent data-[active=true]:text-foreground',
          status === 'recording' && 'bg-primary/10 text-primary hover:bg-primary/15 hover:text-primary',
          status === 'transcribing' && 'bg-primary/10 text-primary'
        )}
        data-active={active}
        disabled={disabled || !state.enabled || status === 'transcribing'}
        onClick={() => {
          triggerHaptic(active ? 'close' : 'open')
          onToggle()
        }}
        size="icon"
        type="button"
        variant="ghost"
      >
        {status === 'recording' ? (
          <Square className="fill-current" size={11} />
        ) : status === 'transcribing' ? (
          <Loader2 className="animate-spin" size={14} />
        ) : (
          <Codicon name="mic" size="0.875rem" />
        )}
      </Button>
    </Tip>
  )
}
