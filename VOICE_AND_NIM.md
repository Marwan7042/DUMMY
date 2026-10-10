# Dummy's Voice + Big Brain Architecture

The local Qwen 9B is the **always-on orchestrator**: general chat, basic/intermediate
coding, full speed, fully private. The free NVIDIA NIM API is wired in as **a tool
the orchestrator can call** — not a replacement. Voice is the interface.

```
        you (voice, Ctrl+T)
              │
              ▼
   ┌─────────────────────┐    consult_big_brain()      ┌──────────────┐
   │  LOCAL 9B (VRAM)     │ ──────────────────────────► │ GLM-5.3      │
   │  orchestrator       │    delegate_task() 12ms ───► │ (NIM cloud)  │
   │  - chat             │    ◄──── background agent ── │  big brain   │
   │  - basic/inter code │         with user's tools   └──────────────┘
   └─────────────────────┘
```

## The three escalation paths

| Path | Tool | When | Blocking? |
|---|---|---|---|
| **Handle it yourself** | — | chat, single-file work, anything the 9B can do | — |
| **Ask for advice** | `consult_big_brain(question, context)` | hard algorithm, tricky bug, second opinion — local keeps control | yes (seconds) |
| **Delegate the project** | `delegate_task(task)` | agentic / multi-file / large work | **No — 12ms.** Cloud model runs in a background thread with full tool access |
| Check on it | `big_brain_status()` | progress of the background brain | no |
| Fetch the report | `big_brain_result()` | when done — walks the user through it | no |

The orchestrator picks the path (policy is in `system_prompt.txt` §16). You keep
chatting with the fast local model the whole time the big brain works; its tool
calls stream past as dim `🧠 big-brain ⚙️ ...` lines, and Dummy announces
completion aloud and on screen.

### Background agent safety

- **One consent prompt at delegation time** (main thread): authorize file edits
  + non-destructive commands, or decline → it runs **read-only** and returns a
  plan instead of changes. `yolo` mode skips the prompt.
- Inside the task it **never prompts** (thread-local consent on the
  PermissionBroker) — prompting from a background thread would deadlock the CLI.
- Destructive commands (`rm -rf`, `sudo`, `git reset --hard`, ...) are **always
  denied** in background mode; they become pending manual steps in the report.
- It gets its own PTY shell (no contention with the main shell), its own
  guardrails (40-round cap, 15-minute deadline, loop detection), and file
  mutations are serialized with the main loop via `_file_ops_lock` (checkpoints
  never race).
- Everything it does is logged in the task spec (`big_brain_status`).

## Voice (Piper TTS + Whisper STT) — the Jarvis layer

**Hands-free: just say "Jarvis".** An always-listening wake word detector
(openWakeWord, bundled `hey_jarvis_v0.1` ONNX model) runs in its own thread
**pinned to the E-cores (16-27)** — the only always-on load in the stack, so
it lives where nothing else runs. Complete hardware map:

| Silicon | Job |
|---|---|
| RTX 5050 (VRAM) | Qwen 9B LLM |
| P-cores 0-7 | **whisper worker** (8 threads — measured faster than 16 on short clips) |
| P-cores 8-15 | **TTS worker** (persistent, model stays hot) |
| **E-cores 16-27** | **always-on wake word listener** (2.4ms per 80ms chunk) |
| Intel iGPU | BGE embeddings (OpenVINO) |

**Latency-tuned (2026-10-09 audit).** The TTS worker is persistent: the voice
model loads once at boot (~0.8s, paid during the greeting), a sentence costs
**~65ms** of say→ack instead of **~870ms** of per-sentence process spawn +
model reload, and playback runs through a player the worker keeps alive for
gapless sentence-to-sentence audio. Barge-in is a **SIGUSR1** to the worker
(instant kill of its player, no reload). Whisper runs 8 threads on a
dedicated partition (16 threads measured *slower* — oversubscription) and
is prewarmed at boot so the first "Jarvis" doesn't pay the model load.
Mic capture stays I/O-bound on all P-cores. Silence end: 1.2s.

Measured detection: **two-stage**. Stage 1 is the openWakeWord detector at a
LOW threshold (0.15, patience 2) — it fires on any voice, tone, or accent
saying "Jarvis" (or "Hey Jarvis"). Stage 2 verifies the whisper transcript
actually contains the name (accent-tolerant: "Jerviz", "Jarviss", Arabic
"جارفز" all pass; "service"/"nervous"/"harvest" are rejected) before anything
happens — a false trigger costs one silent second, nothing more. A ~1.3s
**pre-roll buffer** carries the audio from *before* the trigger into the
capture, so run-on commands ("Jarvis, what time is it?") never lose the
clipped name, and both orders work ("What time is it, Jarvis?" keeps the
front). If you say only the name, he answers **"Yes, sir?"** and listens
once more for the command.

On detection it **stops Jarvis mid-sentence** (barge-in), records your
command with energy-based endpointing (waits for speech, ends after 1.4s
silence, 25s cap), transcribes with whisper, and the transcript lands in
your prompt — **auto-submitted** by default.

| Command | Effect |
|---|---|
| `/voice wake on/off` | toggle the always-listening ear (persisted, default ON) |
| `/voice autosubmit on/off` | submit wake dictation automatically vs. press Enter yourself |
| `/voice voices` | list installed voices (jarvis-high, jarvis-medium, libritts) |
| `/voice voice en_GB-jarvis-medium` | switch instantly + persist |
| `Ctrl+T` | manual push-to-talk (still there as backup) |

Wake tuning (env): `DUMMY_WAKE_THRESHOLD` (0.15), `DUMMY_WAKE_PATIENCE` (2),
`DUMMY_WAKE_CORES` (16-27), `DUMMY_WAKE_ECHO_GUARD` (1 — suppresses detection
while Jarvis is speaking so he never hears himself; set 0 to allow
interrupting his speech with the wake word), `DUMMY_WAKE_VAD_START/STOP`
(energy floors; real thresholds adapt to ambient noise).

**Known environment trap (fixed 2026-10-09):** Python 3.14 made `forkserver`
the default multiprocessing start method on Linux, and `import openvino`
spawns a multiprocessing worker at import time — whose forkserver re-executes
`__main__` (dummy.py) as a **second hidden session** (duplicate wake listener,
mic stream, whisper/TTS workers, doubled logs). dummy.py now forces
`multiprocessing.set_start_method("fork", force=True)` before any heavy
imports, restoring pre-3.14 behavior; a `sys.argv[0] == "-c"` guard in the
`__main__` block is belt-and-suspenders.

**The voice is J.A.R.V.I.S.** — the community-trained Paul Bettany model
(`jgkawell/jarvis`, HuggingFace; personal-use fan model, not for commercial
products). Benchmark on the P-cores: **1.6–4.9× real-time synthesis, ~1s
speech onset** — comfortably fast for streaming.

Voice preference is saved in `~/.dummy/voice.json` next to the on/off state.
Drop any piper voice (`*.onnx` + `*.onnx.json`) into `~/models/piper_voices/`
and it becomes switchable. `DUMMY_PIPER_VOICE` env still overrides everything.

**TTS is real-time streaming.** Replies are spoken *while the LLM generates
them*: the token stream is piped sentence-by-sentence into the persistent
TTS worker (`piper` python API, raw PCM straight into a persistent
`pw-play --raw`), so playback of each sentence starts before its synthesis
finishes. (The one-shot `piper` CLI pipeline remains as a fallback if the
worker can't start; it carries ~0.9s of spawn+load overhead per sentence.)

- Measured per-sentence speech onset: **~0.8s** after the sentence completes
  in the stream (synthesis runs at ~6x real-time on the P-cores, so speech
  never lags behind the text)
- Code blocks are never spoken; markdown (bold, links, bullets, headers,
  numbered markers) is cleaned per sentence
- **Barge-in**: submitting new input, pressing Ctrl+T, or interrupting
  generation kills the speech instantly — mid-sentence, no orphans
- Run-on text with no punctuation is force-split at word boundaries (~300 chars)

Audio work runs in taskset-pinned subprocesses on **all P-cores (`0-15`)** —
the 9B lives entirely in VRAM, so llama-server barely touches the CPU.

| Command              | Effect                                                    |
|----------------------|-----------------------------------------------------------|
| `/voice`             | status panel                                              |
| `/voice on` / `off`  | spoken replies — **persisted across restarts** (default: ON) |
| `/voice test`        | hear a sample sentence                                    |
| `/voice mic`         | 5-second mic round-trip test                              |
| `Ctrl+T`             | push-to-talk dictation → text lands in your prompt (also stops Jarvis first, so the mic doesn't pick up his own voice) |

Jarvis touches: spoken greeting on launch ("Good evening, sir. All systems
are online."), **replies streamed sentence-by-sentence in real time in the
Paul Bettany voice**, "Delegating to the big brain." on hand-off, and "Big
brain task complete." when the cloud brain finishes. The model's WORDS
match the voice too — system_prompt.txt §17 adds the J.A.R.V.I.S. persona:
composed, precise, addresses you as "sir", dry understated wit, engineering
rigor untouched.

Files: `voice.py` (engine: streaming sentence extractor, TTS pipeline
queue/thread, recording, worker management), `voice_worker.py` (persistent
`faster-whisper small` int8, stdlib `wave` decoding to sidestep a
faster-whisper/av 19 API mismatch, ~1-3s per short clip hot).

Config: `DUMMY_VOICE_CORES` (default `0-15`), `DUMMY_PIPER_VOICE`,
`DUMMY_WHISPER_MODEL` (default `small`), `DUMMY_VOICE_CONFIG`.

## NIM plumbing

- Key discovery: `$DUMMY_NIM_KEY` → `$NVIDIA_API_KEY` → `NVIDIANIM:` line in
  `~/keys.txt`. Never printed.
- `/nim on|off` — force full cloud routing for a while; `/nim model <id>` to
  switch brains (`z-ai/glm-5.3`, `z-ai/glm-5.3-flash`, `deepseek-ai/deepseek-v4.1-flash`, ...).
- **`/nim <question>`** — one-shot direct prompt to the big brain: bypasses
  the local model entirely, streams like a normal reply (thinking panel +
  voice), and the answer is NOT added to the local conversation.
- Three routes, three purposes: `/nim <question>` for a quick isolated
  consult, `/nim on` when the big model should *be* the assistant (same
  conversation, all tools), and the orchestrator's own
  consult_big_brain / delegate_task tools when the local 9B asks for help.
- **Auto-fallback**: if llama-server is unreachable, the next turn goes to NIM
  automatically with a notice.
- Verified: NIM streams `reasoning_content` + tool calls in the exact shape
  `_render_stream` already parses, so the tool loop works unchanged.

## Server script fixes (2026-10-09)

- `qwen3.5-9b-server`: repaired `\#!/bin/bash` → `#!/bin/bash`.
- `qwen3.8-27b-server`: added missing `\` after `-ot` so turbo3/turbo4 KV-cache
  quants + flash-attention actually apply.
