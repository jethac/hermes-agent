# Voice Ops Competitive Landscape

Date: 2026-07-01

Scope: Hermes voice-related PRs and issues visible in `NousResearch/hermes-agent`, compared with the current `wip/hackathon-voiceops-business-agent` branch and Sakana AI's KAME architecture.

Sources used:

- GitHub PR metadata and bodies via `gh pr view` / `gh search prs`.
- Current branch files including `20260626plan.md`, `agent/realtime_voice_kame.py`, `agent/realtime_voice_oracle.py`, `tests/gateway/test_discord_realtime_voice.py`, and `website/docs/developer-guide/realtime-voice-prd.md`.
- Sakana AI KAME overview: https://pub.sakana.ai/kame/
- KAME paper: https://arxiv.org/pdf/2510.02327

## Executive Summary

The voice work in upstream Hermes is active and fragmented. There are at least four major architecture tracks:

1. Direct Discord realtime voice using OpenAI Realtime.
2. Generic external voice runtime protocols such as `voice_server`.
3. Platform-native voice plugins for Daily/WebRTC, Teams CVI, PSTN calls, and Google Meet.
4. The current branch's KAME-style reflex/interpreter/oracle architecture.

The current branch is differentiated. It is the only work I found that explicitly centers the Sakana KAME idea as a Hermes-native three-tier stack: a low-latency reflex owns turn-taking, short acknowledgements, barge-in, and spoken responsiveness; a Gemma-style direct-audio interpreter adjudicates raw audio plus witness transcript hypotheses; and Hermes' active `/model` remains the backend oracle for durable reasoning, tools, files, memory, spend, calls, and approvals.

Other PRs independently converge on parts of that idea:

- `#36903` Google Meet issue has the closest conceptual match: fast conversation path plus deep background Hermes work.
- `#54462` proposes a platform-neutral realtime orchestration API that returns short immediate responses plus deferred tasks.
- `#55660` PSTN voice-call plugin uses `agent_consult` from realtime calls into Hermes.
- `#49088` Teams CVI describes realtime delegation to Hermes via consult/task tools.
- `#27040` voice_server splits audio runtime from Hermes and reconciles spoken history.

But none of those appear to implement the full KAME-style contract as the primary product architecture: structured reflex routing, explicit no-tool authority for the reflex, raw-audio interpreter promotion before durable/action evidence, sidecar isolation, reflex/interpreter/oracle metadata in prompts/status, and provider-neutral KAME bridge tools.

## What KAME Means Here

Sakana KAME is a tandem speech architecture: a fast S2S frontend starts responding immediately, while a backend text LLM runs asynchronously over growing partial transcripts and injects oracle guidance back into the frontend. The published architecture emphasizes low latency, backend flexibility, and "speak while thinking" behavior.

Hermes should borrow the latency split without copying an STT-first trust model.
The target KAME contract is raw-audio-primary: Moshi/open-S2S/classic-ASR text
is useful witness context for the interpreter, but it is not durable user text,
not a scheduler, and not tool authority until raw-audio-grounded interpreter or
oracle promotion.

The current Hermes branch is an applied agent-system version of that pattern, not a literal trained S2S oracle-token model. Its KAME-like elements are:

- Low-latency reflex/interface model owns live speech, routing, turn-taking, barge-in, and brief spoken responses.
- Gemma-style interpreter receives clipped raw audio plus Moshi/open-S2S/classic-ASR witness hypotheses and decides what wording can be promoted.
- Hermes backend oracle owns files, tools, memory, approvals, durable work, and session history through the existing active `/model`.
- Reflex and witness transcripts are explicitly denied direct tool/function/MCP/file/memory/payment/call authority.
- Reflex escalates via structured routes: `local`, `defer`, `oracle_direct`, `reject_or_clarify`.
- Runtime metadata labels the architecture as `kame_interface_oracle` with separate reflex, interpreter, and oracle evidence fields.
- Provider-neutral sidecar path supports reference/local, OpenAI Realtime, Gemini Live, and text-oracle/TTS fallback.
- Evidence and diagnostics are first-class: smoke reports, live evidence artifacts, doctor/status surfaces, replayable event contracts.

The main difference from Sakana KAME is that this branch does not depend on a trained acoustic-token S2S frontend that consumes oracle tokens mid-generation. It approximates the tandem architecture with a reflex/router, sidecar protocol, and Hermes oracle stream. That is more upstreamable for Hermes because it preserves Hermes' existing permission and agent-loop boundaries.

## PR Matrix

| PR | Status | Author | Area | Relevance | KAME Similarity |
| --- | --- | --- | --- | --- | --- |
| [#42611](https://github.com/NousResearch/hermes-agent/pull/42611) | Open | joelneleber | Discord live voice transcription | Bounds Discord utterance buffers, prevents every transcript from forcing agent turns by default, refreshes inactivity on inbound voice. | Low. Valuable hygiene for Discord voice input, not an architecture. |
| [#21504](https://github.com/NousResearch/hermes-agent/pull/21504) | Open | RationallyPrime | Discord realtime mode | Large MVP: OpenAI Realtime wrapper, continuous Discord `AudioSource`, `/voice realtime`, streamed audio deltas, barge-in clearing, tools disabled by default. | Medium-low. Strong transport/playback reference, but mostly direct realtime provider mode rather than reflex/interpreter/oracle split. |
| [#27650](https://github.com/NousResearch/hermes-agent/pull/27650) | Open | Solvely-Colin | Discord OpenAI Realtime | Compact OpenAI Realtime Discord implementation with managed auth, live Discord smoke, response cancel, tool-call bridge, continuous stream playback. | Medium. Has live proof and tool bridging, but it lets the realtime provider invoke Hermes tools directly rather than preserving an oracle boundary. |
| [#51827](https://github.com/NousResearch/hermes-agent/pull/51827) | Open | yungalgo | Browser/WebRTC voice platform | Daily + Deepgram Flux + Cartesia, in-process gateway plugin, barge-in via `agent.interrupt()`, latency telemetry, persistent TTS socket, `voice_model` slot. | Medium. Good production voice-loop patterns, but it is STT -> agent -> TTS/in-process, not KAME reflex/interpreter/oracle. |
| [#49088](https://github.com/NousResearch/hermes-agent/pull/49088) | Open | ahenawy | Microsoft Teams CVI | Massive Teams voice/video/chat/governance platform. Includes realtime speech-to-speech, video perception, avatar rendering, group gate, realtime delegation via Hermes consult/task. | Medium-high conceptually. Delegation resembles KAME, but scope is Teams-specific and heavy; not a clean provider-neutral KAME substrate. |
| [#27040](https://github.com/NousResearch/hermes-agent/pull/27040) | Open | tmylk | Generic `voice_server` gateway | External voice runtime owns audio, STT, TTS, turn-taking, barge-in; Hermes owns routing, persistence, auth, history reconciliation; streams assistant deltas to runtime. | High on separation of concerns, medium on KAME. It splits runtime/Hermes cleanly but does not describe a frontend model that reasons locally then escalates via oracle routes. |
| [#54462](https://github.com/NousResearch/hermes-agent/pull/54462) | Open | fritzpaz | Realtime orchestration API | Adds low-latency `/v1/realtime/*` API: immediate no-tools talker response plus action to start slower background work. | High conceptually. It mirrors fast/deep routing, but is API/chat orchestration rather than full voice sidecar/audio implementation. |
| [#55660](https://github.com/NousResearch/hermes-agent/pull/55660) | Open | dynamite-bud | PSTN voice-call plugin | Native voice_call platform for Telnyx/Twilio/Plivo/mock. Optional realtime S2S with OpenAI or Gemini, barge-in, `agent_consult`, model hangup. | Medium-high. `agent_consult` is close to oracle delegation, but the PR is carrier/platform-specific and keeps its own plugin-local realtime session. |
| [#36903](https://github.com/NousResearch/hermes-agent/issues/36903) | Open issue | naiv-mira-lane | Google Meet realtime plugin | Gemini Live/OpenAI Realtime provider abstraction, fast conversation path, deep background Hermes/tool path, wake gate, async task lifecycle. | Highest conceptual match outside this branch. It names the same latency split: short live response now, Hermes/tool work asynchronously. Not Discord and not a PR. |
| [#43193](https://github.com/NousResearch/hermes-agent/pull/43193) | Open | Cole719 | Discord voice state intent | Requests Discord voice state intent only when `discord.voice_fx.enabled`. | Low. Operational fix; relevant because voice features should not penalize text-only Discord bots. |
| [#44023](https://github.com/NousResearch/hermes-agent/pull/44023) | Open | Cdddo | Discord VoiceMixer bug | Makes `VoiceMixer` inherit `discord.AudioSource` so Discord accepts it. | Low. Important reliability fix for Discord ambient/ack playback. |
| [#43845](https://github.com/NousResearch/hermes-agent/pull/43845) | Open | liuhao1024 | Desktop auto-TTS | Wires desktop `voice.auto_tts` to read assistant replies aloud after runs. | Low. Useful UX, not realtime conversation. |
| [#53313](https://github.com/NousResearch/hermes-agent/pull/53313) | Open | chriswinig | STT transcript echo | Keeps raw transcript echoes opt-in to avoid duplicate voice replies. | Low. Message hygiene for voice/STT flows. |
| [#36169](https://github.com/NousResearch/hermes-agent/pull/36169) | Open | qstyk-agent001-bot | Telegram voice mode | Makes Telegram `/voice on` produce voice-message copies of replies. | Low. Platform behavior polish. |

## Competitive Tracks

### 1. Direct OpenAI Realtime Discord

PRs: `#21504`, `#27650`

This is the clearest competitor for a Discord voice demo. It is simpler to explain: Discord PCM goes to OpenAI Realtime, OpenAI audio comes back, barge-in cancels, and optionally Hermes tools are bridged.

Strengths:

- Strong live-demo story, especially `#27650`.
- Less architectural complexity.
- Provider-side VAD, response cancel, audio events, and tool calls are already available.
- Managed auth support in `#27650` is valuable.

Weaknesses:

- Tends to collapse the frontend into the agent unless tool scope is heavily constrained.
- Direct provider tool calls can bypass the Hermes permission/oracle model if not carefully mediated.
- Less provider-neutral. The best experience depends on one vendor's realtime stack.
- Weaker fit for Hermes' narrow-waist principle unless kept behind sidecar/provider gates.

Recommendation:

Borrow transport proof, auth handling, event mapping, response cancel, playback details, and smoke scripts. Do not adopt the direct tool-call architecture as the main design. Keep OpenAI Realtime as a reflex/provider candidate behind the KAME interpreter and oracle authority boundary.

### 2. Generic External Voice Runtime

PR: `#27040`

This PR is architecturally important because it frames voice as a bidirectional room protocol, not a tool call. The external runtime owns audio, turn-taking, barge-in, and TTS; Hermes owns sessions, routing, auth, and persistence. It also handles spoken-history reconciliation: what the user actually heard can replace or delete interrupted assistant text.

Strengths:

- Clean audio-runtime/Hermes split.
- Protocol could support Pipecat/LiveKit/WebRTC/telephony.
- Partial assistant streaming starts TTS before full text completion.
- Spoken-history reconciliation is a serious production concern that many PRs miss.

Weaknesses:

- The external runtime is not necessarily a low-latency reasoning frontend. It may just be STT/TTS/transport.
- It adds a new platform protocol surface in core.
- It does not encode KAME routes, interpreter promotion, oracle authority boundaries, or reflex capability flags.

Recommendation:

Treat `#27040` as a strong prior for protocol shape, history reconciliation, and external runtime lifecycle. The current branch should be prepared to explain why the KAME sidecar is not just another `voice_server`, or alternatively how KAME events could map onto a future generic voice runtime protocol.

### 3. Platform-Native Voice Plugins

PRs/issues: `#51827`, `#49088`, `#55660`, `#36903`

These tracks optimize for product reach: Daily browser rooms, Teams calls/video, PSTN phone calls, Google Meet. They generally put voice inside a gateway platform plugin instead of the desktop/Discord sidecar path.

Strengths:

- They target real user surfaces with clear operational needs.
- `#51827` has useful barge-in ordering and latency telemetry.
- `#49088` includes governance, group-call policy, vision, DLP, allowlist, and meeting behavior.
- `#55660` has mature carrier lifecycle/security, mock provider, call persistence, DTMF, and realtime media bridges.
- `#36903` has the closest KAME-like fast/deep routing language and async background task lifecycle.

Weaknesses:

- Each platform builds its own voice loop and provider abstractions.
- Teams/PSTN/Meet work is too platform-specific to be the core Hermes realtime substrate.
- Some introduce broad config/plugin surfaces and dependencies.
- Architecture risks fragmentation unless Hermes defines a common realtime voice contract.

Recommendation:

Use these PRs as evidence that voice is not one Discord feature; it is a product category. The current KAME reflex/interpreter/oracle protocol can be positioned as the shared substrate that platform plugins can target over time.

### 4. Realtime Orchestration Without Audio

PR: `#54462`

This is not a voice PR in the transport sense, but it is directly relevant to the KAME competitive question. It proposes a low-latency API that returns a short no-tools response and a deferred task action, specifically to avoid blocking live clients while Hermes does slow work.

Strengths:

- Clean abstraction for fast/deep split.
- Small-ish PR compared with full voice stacks.
- Platform-neutral and potentially reusable by web/mobile/voice clients.

Weaknesses:

- No microphone, playback, VAD, barge-in, provider audio, or Discord integration.
- Could duplicate concepts already in the KAME reflex/oracle routing if both land separately.

Recommendation:

This is the strongest candidate for convergence. If maintainers resist a Discord-specific KAME substrate, propose extracting the route/task semantics into a shared realtime orchestration layer while leaving audio sidecars/providers at the edges.

## KAME-Style Assessment

### Found Full KAME-Style Approach?

No other PR I found implements the full hybrid KAME-style approach as the central design.

The current branch appears unique in combining:

- Explicit KAME vocabulary and role metadata.
- Reflex/interpreter/oracle split.
- Provider-neutral sidecar.
- Structured reflex routes.
- Reflex and witness-transcript no-tool-authority rule.
- Hermes backend oracle prompt context.
- OpenAI/Gemini reflex/provider integration without granting them direct Hermes authority by default.
- Replayable diagnostics and evidence gates.

### Closest Analogues

1. `#36903` Google Meet issue:
   - Fast conversation path plus deep Hermes/tool background path.
   - Wake gate, immediate acknowledgement, async worker.
   - Closest conceptual match, but not Discord, not merged, and not framed around reflex/interpreter/oracle protocol.

2. `#54462` realtime orchestration API:
   - Immediate talker response plus deferred tasks.
   - Platform-neutral.
   - Lacks audio and sidecar implementation.

3. `#55660` voice_call:
   - Realtime provider can call `agent_consult`.
   - Strong telephony runtime.
   - Plugin-local and carrier-specific.

4. `#49088` Teams CVI:
   - Realtime model delegates to Hermes consult/task.
   - Broadest multimodal product vision.
   - Heavy platform-specific PR with governance/video/avatar scope.

5. `#27040` voice_server:
   - Best external-runtime split.
   - More transport/protocol than KAME reflex/oracle model.

### Self-Critique: Reflex and Oracle Are Not Yet Truly Async

The strongest critique of the current branch is that it has the right model-role boundary, but not yet the full runtime behavior implied by that boundary.

In the desired KAME/Hermes design, the user should keep talking to the reflex while oracle work continues in the background. The reflex should stay live: listen, acknowledge, clarify, update task state, accept new user instructions, cancel stale work, and queue additional oracle tasks without blocking the voice loop. The oracle should run as a bounded background work pool, not as a modal turn that freezes the interface.

Today the branch is closer to a structured handoff than a truly concurrent work scheduler. It can distinguish frontend/reflex work from Hermes/oracle work, but the interaction model still risks becoming:

```text
user speaks -> reflex routes -> oracle runs -> voice session waits too much
```

The target should be:

```text
user keeps speaking -> reflex keeps conversing
                  \-> oracle task 1 runs in background
                  \-> oracle task 2 runs in background
                  \-> oracle task 3 runs in background
                  \-> oracle task 4 runs in background
```

The logical capacity limit should come from the active `/model` runtime and safety policy. For a DGX Spark running Nemotron-3 Super, a plausible first benchmark bound is four concurrent oracle tasks, but the architecture is a bounded work pool sized by the selected runtime. The reflex should expose that limit conversationally: accept work until the pool is full, summarize active work, prioritize or cancel tasks when asked, and avoid pretending that background work is complete before the oracle reports evidence.

This is also where the current branch can learn from `#36903` and `#54462`. Those efforts explicitly model fast/deep routing and background task lifecycle. Our advantage is the stricter reflex/interpreter/oracle authority model: witness text is context for Gemma, promoted evidence is required for durable/action work, and the oracle side becomes a real async task queue with progress, cancellation, and capacity management.

## Strategic Positioning

The strongest argument for the current branch is not "we added Discord voice." Others have also done that. The strongest argument is:

Hermes needs a realtime voice architecture that preserves Hermes' agent boundary. Direct realtime providers are good interfaces, but they should not become the agent. The KAME reflex/interpreter/oracle split lets Hermes get low-latency voice while keeping tools, memory, approvals, spend, calls, and durable work inside Hermes.

Positioning points:

- Direct OpenAI Realtime PRs prove the demo, but KAME proves the product architecture.
- `voice_server` proves runtime separation, but KAME adds the reflex/interpreter/oracle authority model.
- Platform plugins prove demand across Discord, Meet, Teams, Daily, and PSTN, but KAME offers a common substrate.
- The current branch already incorporates lessons from the direct PRs: bounded STT, persistent playback, barge-in cancellation, provider event mapping, OpenAI/Gemini reflex providers, smoke evidence.

## Risks

1. Complexity risk:
   - KAME reflex/interpreter/oracle is harder to explain than "OpenAI Realtime in Discord."
   - Mitigation: lead with a small stack of upstreamable PRs: bounded receiver, persistent mixer, sidecar protocol, provider adapters.

2. Premature architecture risk:
   - Maintainers may prefer `voice_server` or `#54462` as the generic substrate.
   - Mitigation: explicitly map KAME events to those abstractions and show what each lacks.

3. Provider parity risk:
   - Direct realtime PRs may look better in live demos.
   - Mitigation: keep OpenAI Realtime and Gemini Live as KAME reflex/provider candidates and show equal or better live evidence.

4. Core footprint risk:
   - Voice can sprawl into core prompt/tool/config.
   - Mitigation: keep sidecar/provider code at edges, config-gated, opt-in, and avoid adding core model tools.

5. Evidence risk:
   - A sophisticated architecture without live artifacts will lose to simpler live PRs.
   - Mitigation: ship current evidence bundles, latency metrics, Discord loopback smoke, live-probe logs, and degraded-mode reports with the PR.

## Recommended Next Moves

1. Write the PR narrative around the authority boundary:
   - Reflex/provider can speak and route.
   - Hermes oracle can reason and act.
   - Tools/files/memory never move into the voice vendor by accident.

2. Split the upstream path:
   - PR 1: bounded Discord receiver and transcript dispatch opt-in (`#42611` class of fix).
   - PR 2: persistent Discord realtime mixer/playback (`#21504`/`#44023` lessons).
   - PR 3: provider-neutral realtime voice sidecar protocol and reference provider.
   - PR 4: Discord KAME integration with status/doctor/evidence.
   - PR 5: OpenAI/Gemini reflex providers behind the KAME boundary.

3. Make reflex/oracle scheduling truly async:
   - Keep the reflex session live while oracle tasks run.
   - Add a bounded oracle work pool sized by the active `/model` runtime and safety policy, with four concurrent tasks as the first DGX Spark / Nemotron-3 Super benchmark target.
   - Give each oracle task explicit state: queued, running, waiting for approval, completed, failed, cancelled.
   - Let the reflex summarize, reprioritize, and cancel background tasks without blocking speech input.
   - Persist only durable task outcomes and user-visible spoken commitments, not every partial reflex hypothesis.

4. Proactively compare against `#27040` and `#54462`:
   - If maintainers want a generic protocol, propose that KAME uses it as transport but contributes the missing oracle/reflex contract.
   - If maintainers want API-first realtime orchestration, propose sharing route/task types with the KAME reflex.

5. Reuse best competitor pieces:
   - `#27650`: managed auth, live smoke markers, OpenAI response cancel, realtime event handling.
   - `#21504`: continuous Discord playback source and tests.
   - `#51827`: latency telemetry, prewarm, persistent provider sockets, barge-in ordering.
   - `#27040`: spoken-history reconciliation and external-runtime protocol lessons.
   - `#55660`: telephony-grade security/lifecycle and `agent_consult` ergonomics.
   - `#36903`: fast/deep background task UX and wake/addressing gate.

6. Avoid copying problematic patterns:
   - Direct provider access to Hermes tools without oracle mediation.
   - User-facing non-secret env vars for behavior that belongs in `config.yaml`.
   - Platform-specific voice abstractions in core.
   - A single massive PR that mixes voice substrate, product UI, provider code, and unrelated branch work.

## Bottom Line

There is meaningful competition, but most competing PRs solve one of three narrower problems: "make Discord realtime work," "make a specific platform voice-capable," or "define a generic external voice runtime." The current branch's KAME-style approach is broader and more Hermes-native: it treats low-latency voice as an interface layer while preserving Hermes as the durable agent/oracle.

The strongest external threat is not a single PR replacing this branch. It is maintainers choosing a simpler substrate from `#27040` or `#54462` and asking KAME work to adapt to it. The best response is to frame KAME as the missing authority and routing layer that can sit on top of those substrate ideas, while continuing to prove that Discord voice works in practice.
