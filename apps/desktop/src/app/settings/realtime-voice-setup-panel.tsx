import { useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import {
  applyRealtimeVoiceProfile,
  getRealtimeVoiceSetup,
  runRealtimeVoiceSmoke,
  type HermesConfigRecord,
  type RealtimeVoiceProviderSetup,
  type RealtimeVoiceSetupResponse,
  type RealtimeVoiceSmokeResponse
} from '@/hermes'
import { Activity, AlertTriangle, CheckCircle2, Loader2, Mic, Play, RefreshCw, Volume2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'

import { CONTROL_TEXT } from './constants'
import { getNested, setNested } from './helpers'
import { ListRow, Pill, SectionHeading } from './primitives'

const PROVIDER_LABELS: Record<string, string> = {
  openai: 'OpenAI Realtime',
  gemini: 'Gemini Live',
  elevenlabs: 'ElevenLabs bridge',
  deepgram: 'Deepgram bridge',
  cartesia: 'Cartesia bridge'
}

const PROVIDER_MODELS: Record<string, string[]> = {
  openai: ['gpt-realtime-2'],
  gemini: ['gemini-3.1-flash-live-preview', 'gemini-2.5-flash-live-preview'],
  elevenlabs: ['eleven_flash_v2_5', 'eleven_multilingual_v2'],
  deepgram: ['nova-3', 'aura-2-thalia-en'],
  cartesia: ['sonic-3.5']
}

const PROVIDER_VOICES: Record<string, string[]> = {
  openai: ['marin', 'cedar', 'alloy', 'verse'],
  gemini: ['Puck', 'Charon', 'Kore', 'Fenrir', 'Aoede'],
  elevenlabs: [''],
  deepgram: ['aura-2-thalia-en', 'aura-2-asteria-en'],
  cartesia: ['']
}

function statusTone(ok?: boolean | null) {
  return ok ? 'primary' : 'muted'
}

function setupProviderFor(config: HermesConfigRecord): string {
  const provider = String(getNested(config, 'voice.realtime.frontend_provider') ?? '')
  const sttUrl = String(getNested(config, 'voice.realtime.streaming_stt_base_url') ?? '')
  const sttModel = String(getNested(config, 'voice.realtime.streaming_stt_model') ?? '')
  if (provider === 'openai_realtime' || provider === 'openai') return 'openai'
  if (provider === 'gemini_live' || provider === 'gemini') return 'gemini'
  if (sttUrl.includes('8767') || sttModel.includes('scribe')) return 'elevenlabs'
  if (sttUrl.includes('8766') || sttModel.includes('nova')) return 'deepgram'
  return 'openai'
}

function envPill(label: string, present?: boolean) {
  return <Pill tone={statusTone(present)}>{present ? `${label} set` : `${label} missing`}</Pill>
}

function ProviderCard({
  active,
  provider,
  onSelect
}: {
  active: boolean
  provider: RealtimeVoiceProviderSetup
  onSelect: (id: string) => void
}) {
  const implemented = provider.implemented !== false
  return (
    <button
      className={cn(
        'min-w-0 rounded-lg border px-3 py-3 text-left transition',
        active ? 'border-primary/70 bg-primary/8' : 'border-border/70 bg-muted/25 hover:bg-muted/45',
        !implemented && 'opacity-65'
      )}
      disabled={!implemented}
      onClick={() => onSelect(provider.id)}
      type="button"
    >
      <div className="flex min-w-0 items-center justify-between gap-2">
        <div className="min-w-0 truncate text-sm font-medium">{provider.label}</div>
        {active ? <CheckCircle2 className="size-4 shrink-0 text-primary" /> : null}
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        <Pill>{provider.kind === 'native_s2s' ? 'Native S2S' : 'Bridge'}</Pill>
        {provider.implemented === false ? <Pill>Planned</Pill> : envPill(provider.api_key_env || 'API key', provider.api_key_present)}
      </div>
      <div className="mt-2 truncate font-mono text-[0.68rem] text-muted-foreground/70">
        {provider.model || provider.provider || provider.id}
      </div>
    </button>
  )
}

function StatusGrid({ setup }: { setup: RealtimeVoiceSetupResponse | null }) {
  const status = setup?.status
  const sidecar = status?.sidecar
  const caps = sidecar?.health?.capabilities ?? {}
  const quality = status?.conversation_quality ?? {}
  return (
    <div className="grid gap-2 sm:grid-cols-3">
      <div className="rounded-lg border border-border/70 bg-muted/20 px-3 py-2">
        <div className="text-xs font-medium">Availability</div>
        <div className="mt-1 flex flex-wrap gap-1.5">
          <Pill tone={statusTone(status?.enabled)}>Realtime {status?.enabled ? 'on' : 'off'}</Pill>
          <Pill tone={statusTone(status?.available)}>{status?.available ? 'Ready' : status?.unavailable_reason || 'Unavailable'}</Pill>
        </div>
      </div>
      <div className="rounded-lg border border-border/70 bg-muted/20 px-3 py-2">
        <div className="text-xs font-medium">Sidecar</div>
        <div className="mt-1 flex flex-wrap gap-1.5">
          <Pill tone={statusTone(sidecar?.healthy ?? sidecar?.autostart)}>{sidecar?.mode || 'none'}</Pill>
          {sidecar?.autostart ? <Pill tone="primary">Autostart</Pill> : null}
        </div>
      </div>
      <div className="rounded-lg border border-border/70 bg-muted/20 px-3 py-2">
        <div className="text-xs font-medium">Capabilities</div>
        <div className="mt-1 flex flex-wrap gap-1.5">
          <Pill tone={statusTone(Boolean(caps.native_s2s))}>S2S</Pill>
          <Pill tone={statusTone(Boolean(caps.streaming_stt))}>STT</Pill>
          <Pill tone={statusTone(Boolean(caps.tts))}>TTS</Pill>
          <Pill tone={statusTone(Boolean(quality.barge_in))}>Barge-in</Pill>
        </div>
      </div>
    </div>
  )
}

export function RealtimeVoiceSetupPanel({
  config,
  onConfigChange
}: {
  config: HermesConfigRecord
  onConfigChange: (config: HermesConfigRecord) => void
}) {
  const [setup, setSetup] = useState<RealtimeVoiceSetupResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [applying, setApplying] = useState(false)
  const [runningSmoke, setRunningSmoke] = useState(false)
  const [smoke, setSmoke] = useState<RealtimeVoiceSmokeResponse | null>(null)
  const [selectedProvider, setSelectedProvider] = useState(() => setupProviderFor(config))
  const [model, setModel] = useState(String(getNested(config, 'voice.realtime.frontend_model') ?? ''))
  const [voice, setVoice] = useState('')
  const [requireDiscordSmoke, setRequireDiscordSmoke] = useState(true)
  const [requireInboundSmoke, setRequireInboundSmoke] = useState(false)

  const activeProvider = useMemo(() => setupProviderFor(config), [config])
  const providers = setup?.providers ?? []
  const selectedSetup = providers.find(provider => provider.id === selectedProvider)
  const discord = setup?.discord

  async function refresh() {
    setLoading(true)
    try {
      setSetup(await getRealtimeVoiceSetup())
    } catch (err) {
      notifyError(err, 'Realtime voice setup failed to load')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  useEffect(() => {
    setSelectedProvider(activeProvider)
  }, [activeProvider])

  useEffect(() => {
    const nextModel = String(getNested(config, 'voice.realtime.frontend_model') ?? '')
    const nextVoice =
      activeProvider === 'openai'
        ? String(getNested(config, 'voice.realtime.openai_realtime_voice') ?? '')
        : activeProvider === 'gemini'
          ? String(getNested(config, 'voice.realtime.gemini_live_voice') ?? '')
          : ''
    setModel(nextModel)
    setVoice(nextVoice)
  }, [activeProvider, config])

  function patchConfig(key: string, value: unknown) {
    onConfigChange(setNested(config, key, value))
  }

  async function applyProvider() {
    if (selectedProvider === 'cartesia') {
      notify({ kind: 'info', title: 'Cartesia bridge is planned', message: 'The setup UI is ready; the bridge implementation is not wired yet.' })
      return
    }
    setApplying(true)
    try {
      const result = await applyRealtimeVoiceProfile({
        preset: selectedProvider,
        model,
        voice,
        enable_discord: Boolean(getNested(config, 'discord.realtime_voice.enabled')),
        google_search: Boolean(getNested(config, 'voice.realtime.gemini_live_google_search')),
        oracle_tool: Boolean(getNested(config, 'voice.realtime.gemini_live_oracle_tool') ?? true)
      })
      onConfigChange(result.config)
      setSetup(result.setup)
      notify({ kind: 'success', title: `${PROVIDER_LABELS[selectedProvider]} applied`, message: 'Realtime voice settings updated.' })
    } catch (err) {
      notifyError(err, 'Realtime voice preset failed')
    } finally {
      setApplying(false)
    }
  }

  async function runSmoke() {
    setRunningSmoke(true)
    setSmoke(null)
    try {
      const result = await runRealtimeVoiceSmoke({
        require_discord: requireDiscordSmoke,
        require_inbound: requireInboundSmoke,
        wait_seconds: requireInboundSmoke ? 15 : 5
      })
      setSmoke(result)
      notify({
        kind: result.ok ? 'success' : 'error',
        title: result.ok ? 'Realtime voice smoke passed' : 'Realtime voice smoke failed',
        message: result.output_dir || 'Evidence run finished.'
      })
    } catch (err) {
      notifyError(err, 'Realtime voice smoke failed')
    } finally {
      setRunningSmoke(false)
      void refresh()
    }
  }

  return (
    <div className="mb-6 grid gap-4">
      <SectionHeading icon={Mic} meta={loading ? 'Checking' : undefined} title="Realtime Voice Setup" />
      <StatusGrid setup={setup} />

      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        {providers.map(provider => (
          <ProviderCard
            active={selectedProvider === provider.id}
            key={provider.id}
            onSelect={id => setSelectedProvider(id)}
            provider={provider}
          />
        ))}
      </div>

      <div className="grid gap-1 rounded-lg border border-border/70 bg-muted/15 p-3">
        <ListRow
          action={
            <Switch
              checked={Boolean(getNested(config, 'voice.realtime.enabled'))}
              onCheckedChange={checked => patchConfig('voice.realtime.enabled', checked)}
            />
          }
          description="Controls desktop realtime capture and managed sidecar availability."
          title="Enable realtime voice"
        />
        <ListRow
          action={
            <Select onValueChange={setModel} value={model || PROVIDER_MODELS[selectedProvider]?.[0] || ''}>
              <SelectTrigger className={CONTROL_TEXT}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {(PROVIDER_MODELS[selectedProvider] ?? [model]).filter(Boolean).map(option => (
                  <SelectItem key={option} value={option}>
                    {option}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          }
          description={selectedSetup?.kind === 'native_s2s' ? 'Low-latency interface model.' : 'Streaming bridge model label.'}
          title="Model"
        />
        {selectedSetup?.kind === 'native_s2s' ? (
          <ListRow
            action={
              <Select onValueChange={setVoice} value={voice || PROVIDER_VOICES[selectedProvider]?.[0] || ''}>
                <SelectTrigger className={CONTROL_TEXT}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {(PROVIDER_VOICES[selectedProvider] ?? [voice]).filter(Boolean).map(option => (
                    <SelectItem key={option} value={option}>
                      {option}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            }
            description="Provider voice used for native realtime audio."
            title="Voice"
          />
        ) : null}
        <div className="flex flex-wrap justify-end gap-2 pt-2">
          <Button disabled={loading} onClick={() => void refresh()} size="sm" type="button" variant="secondary">
            <RefreshCw />
            Refresh
          </Button>
          <Button disabled={applying || selectedProvider === 'cartesia'} onClick={() => void applyProvider()} size="sm" type="button">
            {applying ? <Loader2 className="animate-spin" /> : <Volume2 />}
            Apply Provider
          </Button>
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <div className="rounded-lg border border-border/70 bg-muted/15 p-3">
          <div className="mb-2 flex items-center gap-2 text-sm font-medium">
            <Activity className="size-4 text-muted-foreground" />
            Sidecar
          </div>
          <div className="grid gap-2 text-xs text-muted-foreground">
            <div className="flex items-center justify-between gap-3">
              <span>URL</span>
              <span className="truncate font-mono">{setup?.status.sidecar?.base_url || 'managed loopback'}</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              <Pill tone={statusTone(setup?.status.sidecar?.healthy ?? setup?.status.sidecar?.autostart)}>
                {setup?.status.sidecar?.healthy ? 'Healthy' : setup?.status.sidecar?.autostart ? 'Autostart' : 'Not healthy'}
              </Pill>
              {setup?.status.sidecar?.externally_managed ? <Pill>External</Pill> : null}
              {setup?.status.sidecar?.health?.frontend?.provider ? (
                <Pill>{String(setup.status.sidecar.health.frontend.provider)}</Pill>
              ) : null}
            </div>
          </div>
        </div>

        <div className="rounded-lg border border-border/70 bg-muted/15 p-3">
          <div className="mb-2 flex items-center gap-2 text-sm font-medium">
            <AlertTriangle className="size-4 text-muted-foreground" />
            Discord /voice join
          </div>
          <div className="grid gap-3">
            <div className="flex flex-wrap gap-1.5">
              <Pill tone={statusTone(discord?.enabled)}>Discord realtime {discord?.enabled ? 'on' : 'off'}</Pill>
              {envPill('Bot token', discord?.bot_token_present)}
              {envPill('Guild', discord?.guild_id_present)}
              {envPill('Voice channel', discord?.voice_channel_id_present || discord?.voice_channel_name_present)}
            </div>
            <ListRow
              action={
                <Switch
                  checked={Boolean(getNested(config, 'discord.realtime_voice.enabled'))}
                  onCheckedChange={checked => patchConfig('discord.realtime_voice.enabled', checked)}
                />
              }
              description="Lets Discord voice-channel joins use the realtime sidecar."
              title="Enable Discord realtime"
            />
            <Input
              className={CONTROL_TEXT}
              onChange={event => patchConfig('discord.realtime_voice.sidecar_base_url', event.target.value)}
              placeholder="http://127.0.0.1:8765"
              value={String(getNested(config, 'discord.realtime_voice.sidecar_base_url') ?? discord?.sidecar_base_url ?? '')}
            />
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-border/70 bg-muted/15 p-3">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div>
            <div className="text-sm font-medium">Live Smoke Test</div>
            <div className="mt-1 text-xs text-muted-foreground">
              Runs the focused realtime evidence collector and writes artifacts under the active Hermes profile.
            </div>
          </div>
          <Button disabled={runningSmoke} onClick={() => void runSmoke()} size="sm" type="button">
            {runningSmoke ? <Loader2 className="animate-spin" /> : <Play />}
            Run Test
          </Button>
        </div>
        <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
          <label className="inline-flex items-center gap-2">
            <Switch checked={requireDiscordSmoke} onCheckedChange={setRequireDiscordSmoke} />
            Require Discord join
          </label>
          <label className="inline-flex items-center gap-2">
            <Switch checked={requireInboundSmoke} onCheckedChange={setRequireInboundSmoke} />
            Require inbound speech
          </label>
        </div>
        {smoke ? (
          <div className="mt-3 rounded-md bg-background/70 p-2 text-xs">
            <div className="flex flex-wrap gap-1.5">
              <Pill tone={statusTone(smoke.ok)}>Exit {smoke.exit_code ?? 'timeout'}</Pill>
              <Pill>{smoke.output_dir}</Pill>
            </div>
            {smoke.result?.issues?.length ? (
              <div className="mt-2 font-mono text-[0.68rem] text-destructive">{smoke.result.issues.join(' | ')}</div>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  )
}
