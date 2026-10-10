"""
Voice I/O for Dummy: Piper TTS (real-time streaming) + Whisper STT.

TTS is STREAMING: callers pipe response text through begin_utterance() /
feed() / end_utterance() while the LLM generates it. Completed sentences are
spoken as soon as they exist, pipelined through

    piper --output-raw  ->  ffplay (reads PCM while piper still synthesizes)

so playback starts before synthesis finishes. feed() skips code blocks and
cleans markdown per sentence. flush() is barge-in: it kills whatever is being
said the instant the user speaks or submits new input.

STT: push-to-talk (Ctrl+T) -> pinned pw-record -> persistent faster-whisper
worker.

All audio subprocesses are taskset-pinned (default: all P-cores 0-15; the 9B
model lives entirely in VRAM so llama-server barely uses the CPU).

Config (env):
  DUMMY_VOICE_CORES    cores for all audio work (default "0-15")
  DUMMY_PIPER_BIN      piper executable (default ~/.local/bin/piper)
  DUMMY_PIPER_VOICE    piper voice .onnx (default ~/models/piper_voices/
                       en_US-libritts-high.onnx)
  DUMMY_WHISPER_MODEL  faster-whisper size (default "small")
  DUMMY_VOICE_CONFIG   where the on/off preference is stored
                       (default ~/.dummy/voice.json; TTS starts ON)
"""
import os
import json
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import queue
import wave
import warnings
import difflib
from collections import deque
from pathlib import Path

import numpy as np

VOICE_CORES = os.getenv("DUMMY_VOICE_CORES", "0-15")   # mic capture (I/O-bound)
# Latency-tuned core partitions (measured): 8 whisper threads beat 16 on
# short clips (oversubscription), and TTS/whisper must not fight each other.
TTS_CORES = os.getenv("DUMMY_TTS_CORES", "8-15")
WHISPER_CORES = os.getenv("DUMMY_WHISPER_CORES", "0-7")
PIPER_BIN = (
    os.getenv("DUMMY_PIPER_BIN")
    or shutil.which("piper")
    or os.path.expanduser("~/.local/bin/piper")
)
# All installed piper voices live here; pick with /voice voice <name>.
PIPER_VOICES_DIR = Path(os.getenv(
    "DUMMY_PIPER_VOICES_DIR", os.path.expanduser("~/models/piper_voices")
))
# Preferred order when nothing else is set: the community-trained
# Paul Bettany J.A.R.V.I.S. voice first, then anything else installed.
DEFAULT_VOICE_ORDER = ["en_GB-jarvis-high", "en_GB-jarvis-medium", "en_US-libritts-high"]
MAX_RECORD_SECONDS = 120  # watchdog cap for a forgotten recording
FFPLAY = shutil.which("ffplay")

VOICE_DIR = Path.home() / ".dummy" / "voice"
VOICE_DIR.mkdir(parents=True, exist_ok=True)
VOICE_CONFIG = Path(os.getenv("DUMMY_VOICE_CONFIG", str(Path.home() / ".dummy" / "voice.json")))

# ------------------------------------------------------------- wake word --
# Always-listening "Hey Jarvis" detector (openWakeWord, bundled ONNX model).
# It runs in its own thread pinned to the E-CORES: it is the only always-on
# load in the stack, so it lives where nothing else runs. The dGPU has the
# LLM, the P-cores have TTS/whisper bursts, the E-cores have the wake word.
WAKE_CORES = os.getenv("DUMMY_WAKE_CORES", "16-27")
# Two-stage wake: the detector runs LOW (0.15) to catch any voice/tone saying
# "Jarvis"; whisper then verifies the transcript actually contains the name
# before anything happens. A false trigger just costs a silent second.
WAKE_THRESHOLD = float(os.getenv("DUMMY_WAKE_THRESHOLD", "0.15"))
WAKE_PATIENCE = int(os.getenv("DUMMY_WAKE_PATIENCE", "2"))
WAKE_ECHO_GUARD = os.getenv("DUMMY_WAKE_ECHO_GUARD", "1") != "0"
WAKE_TTS_TAIL = 0.8            # seconds after speech ends before wake resumes
WAKE_MIC_CHUNK = 2560          # bytes per read = 80ms @ 16kHz mono s16
# VAD floors: real thresholds adapt to measured ambient noise (3.5x/1.8x).
WAKE_VAD_START = float(os.getenv("DUMMY_WAKE_VAD_START", "150"))
WAKE_VAD_STOP = float(os.getenv("DUMMY_WAKE_VAD_STOP", "100"))
WAKE_MAX_UTTERANCE = 25.0
WAKE_START_TIMEOUT = 5.0
WAKE_SILENCE_END = 1.2


def _openwakeword_model_path() -> str:
    try:
        import openwakeword
        return str(Path(openwakeword.__file__).parent / "resources" / "models" / "hey_jarvis_v0.1.onnx")
    except Exception:
        return ""


def _parse_cpu_spec(spec: str) -> set:
    """'16-27' or '1,3,5' -> set of ints."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                a, b = part.split("-")
                out.update(range(int(a), int(b) + 1))
            except ValueError:
                pass
        else:
            try:
                out.add(int(part))
            except ValueError:
                pass
    return out


def _tts_interpreter():
    """A python interpreter that can `import piper` (the piper pip package
    lives in the system user-site, not the venv). Cached."""
    global _TTS_INTERP
    if _TTS_INTERP is not None:
        return _TTS_INTERP or None
    candidates = [sys.executable, "/usr/bin/python3", "python3"]
    # Also try the interpreter behind the piper binary itself.
    try:
        first = Path(PIPER_BIN).read_text().splitlines()[0].lstrip("#!").strip() if os.path.isfile(PIPER_BIN) else ""
        if first:
            candidates.append(first)
    except Exception:
        pass
    for cand in candidates:
        try:
            r = subprocess.run([cand, "-c", "import piper"], capture_output=True, timeout=15)
            if r.returncode == 0:
                _TTS_INTERP = cand
                return cand
        except Exception:
            continue
    _TTS_INTERP = ""  # negative cache; None means "not checked yet"
    return None


_TTS_INTERP = None


def list_voices() -> dict:
    """{voice_name: onnx_path} for every installed piper voice."""
    out = {}
    try:
        for onnx in sorted(PIPER_VOICES_DIR.glob("*.onnx")):
            if Path(str(onnx) + ".json").is_file():
                out[onnx.stem] = str(onnx)
    except Exception:
        pass
    return out


def _load_voice_config() -> dict:
    try:
        return json.loads(VOICE_CONFIG.read_text())
    except Exception:
        return {}


def _save_voice_config(updates: dict) -> None:
    data = _load_voice_config()
    data.update(updates)
    try:
        VOICE_CONFIG.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _resolve_voice() -> str:
    """env override > saved preference > default order > anything installed."""
    env = os.getenv("DUMMY_PIPER_VOICE")
    if env and os.path.isfile(env):
        return env
    saved = _load_voice_config().get("voice")
    installed = list_voices()
    if saved and saved in installed:
        return installed[saved]
    for name in DEFAULT_VOICE_ORDER:
        if name in installed:
            return installed[name]
    return next(iter(installed.values()), str(PIPER_VOICES_DIR / "en_GB-jarvis-high.onnx"))


def _voice_sample_rate(voice_path: str) -> int:
    """Native sample rate of a piper voice, from its .onnx.json config."""
    try:
        cfg = json.loads(Path(voice_path + ".json").read_text())
        rate = int(cfg.get("audio", {}).get("sample_rate", 0))
        if rate > 0:
            return rate
    except Exception:
        pass
    return 22050


_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002190-\U000021FF\U00002B00-\U00002BFF\U00002700-\U000027BF]+"
)


def _strip_completed_code_blocks(text: str) -> str:
    return re.sub(r"```.*?```", " ", text, flags=re.S)


def _clean_sentence(text: str) -> str:
    """Markdown/emoji -> clean speech text for ONE already-extracted sentence."""
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*?([^*]+)\*\*?", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"^\s*[#>*\-•]+\s*", "", text)       # bullets / headers / quotes
    text = re.sub(r"^\s*\d+[.)]\s*", "", text)          # numbered list markers
    text = re.sub(r"^\s*step\s+\d+[:.)]?\s*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _force_split(text: str, limit: int = 300) -> list:
    """Word-boundary split for run-on text with no punctuation."""
    out, cur = [], ""
    for word in text.split(" "):
        if cur and len(cur) + len(word) + 1 > limit:
            out.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    if cur:
        out.append(cur)
    return out


_WAKE_ALIASES = {
    "jarvis", "jarvas", "jarvus", "jarviz", "jarviss", "jarvis's", "jarvez",
    "jarvys", "jarwis", "jarvice", "jervis", "jerviz", "jervas", "javis",
    "charvis", "javiz", "chavez", "darvis", "travis", "garvis", "harvest", 
    "harvis", "jardis", "java", "javas", "tardis", "garbage", "drivers", 
    "service", "nervous", "target", "dervish", "jobis", "gervais", "nervus"
}


def _word_is_wake(w: str) -> bool:
    """True if one word sounds like the wake name - tolerates whisper's
    spellings ('Jerviz', 'Jarviss') AND non-English scripts (Arabic 'جارفز'),
    which is how accented speech often comes back from transcription."""
    raw = w.strip()
    if "جارف" in raw:  # Arabic-script 'Jarvis' (jarvez/jarvis phonetics)
        return True
    latin = re.sub(r"[^a-z]", "", raw.lower())
    if not latin:
        return False
    if latin in _WAKE_ALIASES or latin.startswith(("jarv", "jerv")):
        return True
    # Consonant-skeleton match: jarvis -> jrvs, jerviz -> jrvz. Must start
    # with 'j' so words like 'nervous' (nrvs) or 'service' (srvc) can't pass.
    skel = re.sub(r"[aeiouy]", "", latin)
    return skel.startswith("j") and difflib.SequenceMatcher(None, skel, "jrvs").ratio() >= 0.7


def _looks_like_wake(text: str, head_words: int = 6) -> bool:
    """Does the transcript contain something that sounds like 'jarvis' in
    the first few words? Stage-2 verification: the detector runs at a low
    threshold to catch any voice/tone saying the name; this filter decides
    if it was really him."""
    return any(_word_is_wake(w) for w in text.split()[:head_words])


def _strip_wake_words(text: str) -> str:
    """Remove the wake name and addressing. Handles both orders:
    'Jarvis, what time is it' -> 'what time is it'
    'What time is it, Jarvis?' -> 'What time is it' (trailing vocative)."""
    words = text.split()
    for i, w in enumerate(words[:6]):
        if _word_is_wake(w):
            after = words[i + 1:]
            if not after and i > 0:
                # Trailing vocative: the command came BEFORE the name.
                return " ".join(words[:i]).strip().lstrip(" ,.:;-!?")
            return " ".join(after).strip().lstrip(" ,.:;-!?")
    # No wake word found (defensive): strip leading fillers only.
    i = 0
    while i < len(words) and re.sub(r"[^a-z]", "", words[i].lower()) in ("hey", "ok", "okay", "hi", "hello"):
        i += 1
    return " ".join(words[i:]).strip().lstrip(" ,.:;-!?")


class VoiceEngine:
    """Streaming Piper TTS + Whisper STT facade. Safe with missing binaries."""

    def __init__(self):
        self.tts_enabled = bool(_load_voice_config().get("tts_enabled", True))
        self.voice_path = _resolve_voice()
        self.voice_rate = _voice_sample_rate(self.voice_path)
        self._tts_q: "queue.Queue[str]" = queue.Queue()
        self._tts_thread = None
        self._feed_buf = ""
        self._current_procs = None        # (piper, player) currently speaking
        self._current_lock = threading.Lock()
        self._rec_proc = None
        self._rec_path = None
        self._whisper_proc = None
        self._whisper_lock = threading.Lock()
        self._transcribe_lock = threading.Lock()
        # Persistent TTS worker (model stays hot; player lives inside it)
        self._tts_worker = None
        self._tts_worker_lock = threading.Lock()
        self._tts_worker_voice = None
        # Timestamp of the last utterance actually sent to speech (used by
        # the chat loop to know whether Jarvis has spoken during this turn).
        self.last_spoke_at = 0.0
        # Wake word ("Hey Jarvis") always-listening state
        cfg = _load_voice_config()
        self.wake_enabled = bool(cfg.get("wake_enabled", True))
        self.auto_submit = bool(cfg.get("auto_submit", True))
        self._wake_thread = None
        self._wake_stop = threading.Event()
        self._pending_dictation = None
        self._pending_lock = threading.Lock()
        self._tts_tail_until = 0.0
        self._wake_gated = threading.Event()  # set while capturing, to gate detection
        if self.tts_enabled:
            self.toggle_tts(True)  # restores + starts the speaker thread
        if self.wake_enabled:
            self.start_wake()

    # ------------------------------------------------------------- TTS ----
    def toggle_tts(self, state=None) -> bool:
        if state is None:
            self.tts_enabled = not self.tts_enabled
        else:
            self.tts_enabled = bool(state)
        _save_voice_config({"tts_enabled": self.tts_enabled})
        if self.tts_enabled and self._tts_thread is None:
            self._tts_thread = threading.Thread(target=self._tts_loop, daemon=True)
            self._tts_thread.start()
        return self.tts_enabled

    def set_voice(self, name: str) -> str:
        """Switch piper voice at runtime. Returns (voice_name, error)."""
        installed = list_voices()
        if name not in installed:
            return None, f"Voice '{name}' not installed. Available: {', '.join(installed) or 'none'}"
        self.voice_path = installed[name]
        self.voice_rate = _voice_sample_rate(self.voice_path)
        _save_voice_config({"voice": name})
        self.flush()  # stop any in-flight speech from the old voice
        self._kill_tts_worker()  # next utterance respawns with the new model
        return name, None

    # ------------------------------------------------ persistent TTS ------
    def _kill_tts_worker(self) -> None:
        with self._tts_worker_lock:
            if self._tts_worker is not None:
                try:
                    self._tts_worker.kill()
                except Exception:
                    pass
                self._tts_worker = None
            self._tts_worker_voice = None

    def _ensure_tts_worker(self):
        """Spawn the pinned, persistent TTS worker (model preloaded) and wait
        for READY. Lock held through the wait so only one thread ever reads it."""
        with self._tts_worker_lock:
            w = self._tts_worker
            if (w is not None and w.poll() is None
                    and self._tts_worker_voice == self.voice_path):
                return w
            if w is not None:
                try:
                    w.kill()
                except Exception:
                    pass
            interp = _tts_interpreter()
            if interp is None:
                return None
            worker_script = str(Path(__file__).resolve().parent / "tts_worker.py")
            try:
                w = subprocess.Popen(
                    ["taskset", "-c", TTS_CORES, interp, worker_script, self.voice_path],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=None,  # status lines visible once in the CLI
                    text=True, bufsize=1,
                )
            except Exception as exc:
                sys.stderr.write(f"[voice] could not spawn tts worker: {exc}\n")
                self._tts_worker = None
                return None
            self._tts_worker = w
            self._tts_worker_voice = self.voice_path
            # Wait for READY (model load ~0.5-1.5s, paid once per voice).
            line = w.stdout.readline()
            if not line.startswith("READY"):
                sys.stderr.write(f"[voice] tts worker failed to start: {line.strip() or 'no reply'}\n")
                self._kill_tts_worker()
                return None
        return w

    def speak(self, text: str) -> None:
        """Queue a complete utterance (notifications, greetings)."""
        if not self.tts_enabled:
            return
        clean = _strip_completed_code_blocks(text)
        clean = _clean_sentence(clean)
        if len(clean) > 900:
            clean = clean[:900]
        if clean:
            self._tts_q.put(clean)

    # ------------------------------------------------ streaming interface --
    def begin_utterance(self, flush: bool = False) -> None:
        """Start a streamed utterance. flush=True also cuts off current speech."""
        self._feed_buf = ""
        if flush:
            self.flush()

    def feed(self, chunk: str) -> None:
        """Stream raw LLM output in; completed sentences are spoken immediately."""
        if not self.tts_enabled or not chunk:
            return
        self._feed_buf += chunk
        self._drain_sentences()

    def end_utterance(self) -> None:
        """Finish a streamed utterance; speak any remaining buffered text."""
        if not self.tts_enabled:
            return
        self._drain_sentences(final=True)

    def flush(self) -> None:
        """Barge-in: drop all queued/pending speech and stop mid-sentence playback."""
        self._feed_buf = ""
        try:
            while True:
                self._tts_q.get_nowait()
        except queue.Empty:
            pass
        # Worker mode: SIGUSR1 makes the worker kill its player instantly -
        # no pipe ordering issues, no model reload.
        with self._tts_worker_lock:
            w = self._tts_worker
        if w is not None and w.poll() is None:
            try:
                w.send_signal(signal.SIGUSR1)
            except Exception:
                pass
        # Fallback mode: kill the one-shot pipeline processes.
        with self._current_lock:
            procs, self._current_procs = self._current_procs, None
        if procs:
            for proc in procs:
                if proc and proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
        self._tts_tail_until = time.time() + 0.5

    def _drain_sentences(self, final: bool = False) -> None:
        buf = self._feed_buf
        held = ""
        if buf.count("```") % 2 == 1:
            # Inside an open code fence: speak what came before it, hold the rest.
            head, tail = buf.rsplit("```", 1)
            held = "```" + tail if not final else ""
            buf = head
        else:
            buf = _strip_completed_code_blocks(buf)
        # Split after sentence enders (+ following space) or newlines (lists, headers).
        parts = re.split(r"(?<=[.!?…])\s+|\n+", buf)
        leftover = parts.pop() if parts else ""
        to_emit = parts + ([leftover] if final else [])
        for part in to_emit:
            clean = _clean_sentence(part)
            if not clean or not re.search(r"[A-Za-z]", clean):
                continue
            for piece in _force_split(clean):
                self._tts_q.put(piece)
        self._feed_buf = ("" if final else leftover) + held

    # --------------------------------------------------------- TTS engine --
    def _tts_loop(self) -> None:
        while True:
            text = self._tts_q.get()
            self._speak_stream(text)

    def _speak_stream(self, text: str) -> None:
        """Speak one sentence through the persistent TTS worker (model hot,
        zero spawn overhead per sentence). Falls back to the one-shot piper
        CLI pipeline if the worker is unavailable. Runs in the TTS thread."""
        if not os.path.isfile(PIPER_BIN) or not os.path.isfile(self.voice_path):
            return
        try:
            w = self._ensure_tts_worker()
        except Exception as exc:
            sys.stderr.write(f"[voice] tts worker spawn failed: {type(exc).__name__}: {exc}\n")
            w = None
        if w is not None:
            try:
                w.stdin.write(json.dumps({"cmd": "say", "text": text}) + "\n")
                w.stdin.flush()
                ack = w.stdout.readline()
                if not ack.startswith("DONE"):
                    err = ack.strip() or "worker died"
                    sys.stderr.write(f"[voice] tts worker: {err[:200]}\n")
                    if w.poll() is not None:
                        self._kill_tts_worker()
                    # Echo-guard: estimate remaining drain time from the byte
                    # count the worker reports (DONE <bytes>), plus a tail.
                    self._tts_tail_until = time.time() + WAKE_TTS_TAIL
                    return
                # DONE <bytes> -> keep the wake detector deaf while the audio
                # is still draining out of the player, not just while we feed it.
                try:
                    sent = int(ack.split()[1])
                except Exception:
                    sent = 0
                drain = sent / (2.0 * max(self.voice_rate, 1))
                self._tts_tail_until = time.time() + drain + WAKE_TTS_TAIL
                self.last_spoke_at = time.time()
                return
            except Exception as exc:
                sys.stderr.write(f"[voice] tts worker write failed: {type(exc).__name__}: {exc}\n")
                self._kill_tts_worker()
        # -------- fallback: one-shot piper CLI pipeline (always works) ------
        self._speak_cli_fallback(text)

    def _speak_cli_fallback(self, text: str) -> None:
        """One-shot pipeline: piper --output-raw -> stdin player. ~0.9s of
        spawn+load overhead per sentence - only used if the worker can't."""
        piper_proc = None
        play_proc = None
        try:
            piper_proc = subprocess.Popen(
                ["taskset", "-c", TTS_CORES, PIPER_BIN, "-m", self.voice_path, "--output-raw"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, start_new_session=True,
            )
            try:
                piper_proc.stdin.write(text.encode("utf-8"))
                piper_proc.stdin.close()
            except Exception:
                pass

            pw_play = shutil.which("pw-play")
            if pw_play:
                # Native PipeWire player: sends stdin PCM to the default sink.
                player_cmd = ["taskset", "-c", TTS_CORES, pw_play, "--raw",
                              "--rate", str(self.voice_rate), "--channels", "1",
                              "--format", "s16", "-"]
            elif FFPLAY:
                player_cmd = ["taskset", "-c", TTS_CORES, FFPLAY,
                              "-f", "s16le", "-ar", str(self.voice_rate), "-ac", "1",
                              "-nodisp", "-autoexit", "-loglevel", "error", "-i", "-"]
            else:
                # No streaming player at all: whole-file synthesis + any player.
                try:
                    piper_proc.kill()
                except Exception:
                    pass
                out = str(VOICE_DIR / "tts.wav")
                subprocess.run(
                    ["taskset", "-c", TTS_CORES, PIPER_BIN, "-m", self.voice_path, "-f", out],
                    input=text, capture_output=True, text=True, timeout=60,
                )
                for name in ("pw-play", "aplay"):
                    player = shutil.which(name)
                    if player and os.path.isfile(out):
                        subprocess.run(["taskset", "-c", TTS_CORES, player, out], timeout=120)
                        break
                return

            play_proc = subprocess.Popen(
                player_cmd, stdin=piper_proc.stdout,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                start_new_session=True,
            )
            # CRITICAL: close OUR copy of piper's stdout. The player only sees
            # EOF (and exits on its own) once every read end of the pipe is
            # closed - keeping it open here deadlocks playback after the
            # first sentence with raw PCM (which has no duration header).
            piper_proc.stdout.close()

            with self._current_lock:
                self._current_procs = (piper_proc, play_proc)
            try:
                _, play_err = play_proc.communicate(timeout=180)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(play_proc.pid), signal.SIGTERM)
                except Exception:
                    pass
                try:
                    play_proc.communicate(timeout=5)
                except Exception:
                    pass
                play_err = b""
            try:
                piper_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    piper_proc.kill()
                except Exception:
                    pass
            if play_proc.returncode not in (0, None):
                msg = (play_err or b"").decode(errors="replace").strip()
                sys.stderr.write(
                    f"[voice] player exited rc={play_proc.returncode}"
                    + (f": {msg[:200]}" if msg else " (no stderr)") + "\n"
                )
                sys.stderr.flush()
            # Echo guard tail: hold wake detection briefly after speech ends.
            self._tts_tail_until = time.time() + WAKE_TTS_TAIL
        except Exception as exc:
            sys.stderr.write(f"[voice] tts failed: {type(exc).__name__}: {exc}\n")
            sys.stderr.flush()
        finally:
            self.last_spoke_at = time.time()
            with self._current_lock:
                if self._current_procs and piper_proc and self._current_procs[0] is piper_proc:
                    self._current_procs = None
            for proc in (play_proc, piper_proc):
                if proc and proc.poll() is None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                if proc is not None:
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        pass

    # ------------------------------------------------------------- STT ----
    def is_recording(self) -> bool:
        return self._rec_proc is not None and self._rec_proc.poll() is None

    def start_recording(self):
        """Spawn a pinned pw-record. Returns (wav_path, error)."""
        if self.is_recording():
            return None, "Already recording."
        if shutil.which("pw-record") is None:
            return None, "pw-record not found (install pipewire-tools)."
        path = str(VOICE_DIR / f"stt_{int(time.time())}.wav")
        try:
            self._rec_proc = subprocess.Popen(
                ["taskset", "-c", VOICE_CORES, "pw-record",
                 "--rate", "16000", "--channels", "1", "--format", "s16", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            self._rec_proc = None
            return None, f"Could not start pw-record: {exc}"
        self._rec_path = path
        proc, watched_path = self._rec_proc, path
        threading.Timer(MAX_RECORD_SECONDS, self._watchdog_stop, args=(proc, watched_path)).start()
        return path, None

    def _watchdog_stop(self, proc, path) -> None:
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._rec_path == path:
            self._rec_proc, self._rec_path = None, None

    def stop_recording(self):
        """Stop the recorder; SIGINT lets pw-record finalize the WAV header."""
        if not self.is_recording():
            self._rec_proc, self._rec_path = None, None
            return None, "Not recording."
        proc, self._rec_proc = self._rec_proc, None
        path, self._rec_path = self._rec_path, None
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            except Exception:
                pass
        return path, None

    def stop_and_transcribe(self):
        """Stop recording and run whisper. Returns (text, error)."""
        path, err = self.stop_recording()
        if path is None:
            return None, err
        if not os.path.isfile(path) or os.path.getsize(path) < 2000:
            return None, "Recording was empty or too short."
        return self.transcribe(path)

    # --------------------------------------------------- whisper worker ----
    def _ensure_whisper(self):
        """Lazy-start the persistent, pinned transcription worker."""
        with self._whisper_lock:
            if self._whisper_proc is not None and self._whisper_proc.poll() is None:
                return self._whisper_proc
            worker = Path(__file__).resolve().parent / "voice_worker.py"
            print(f"[voice] starting whisper worker (pinned to {WHISPER_CORES})...",
                  file=sys.stderr, flush=True)
            self._whisper_proc = subprocess.Popen(
                ["taskset", "-c", WHISPER_CORES, sys.executable, str(worker)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, bufsize=1,
            )
            return self._whisper_proc

    def _kill_whisper(self) -> None:
        if self._whisper_proc is not None:
            try:
                self._whisper_proc.kill()
            except Exception:
                pass
            self._whisper_proc = None

    def transcribe(self, wav_path: str, timeout: int = 300):
        """Send one WAV path to the worker, return (text, error).

        timeout defaults generously (300s): the first-ever call includes a
        one-time model load (or download); after that it returns in seconds.
        Serialized with a lock: the wake listener and Ctrl+T share this worker.
        """
        with self._transcribe_lock:
            return self._transcribe_locked(wav_path, timeout)

    def _transcribe_locked(self, wav_path: str, timeout: int = 300):
        proc = self._ensure_whisper()
        try:
            proc.stdin.write(wav_path + "\n")
            proc.stdin.flush()
        except Exception as exc:
            self._kill_whisper()
            return None, f"whisper worker unreachable: {exc}"

        result = {}

        def _read():
            result["line"] = proc.stdout.readline()

        reader = threading.Thread(target=_read, daemon=True)
        reader.start()
        reader.join(timeout)
        if reader.is_alive():
            self._kill_whisper()
            return None, "transcription timed out"

        line = (result.get("line") or "").strip()
        if not line:
            self._kill_whisper()
            return None, "whisper worker died"
        if line == "OK" or line.startswith("OK\t"):
            text = line[3:].strip() if line.startswith("OK\t") else ""
            if not text:
                return None, "nothing intelligible was heard"
            return text, None
        if line.startswith("ERR\t"):
            return None, line[4:].strip()
        return None, f"unexpected worker reply: {line[:120]}"

    # --------------------------------------------------- wake word ear ----
    def set_wake(self, state: bool) -> None:
        self.wake_enabled = bool(state)
        _save_voice_config({"wake_enabled": self.wake_enabled})
        if self.wake_enabled:
            self.start_wake()
        else:
            self.stop_wake()

    def set_auto_submit(self, state: bool) -> None:
        self.auto_submit = bool(state)
        _save_voice_config({"auto_submit": self.auto_submit})

    def start_wake(self) -> None:
        if self._wake_thread is not None and self._wake_thread.is_alive():
            return
        self._wake_stop.clear()
        self._wake_thread = threading.Thread(target=self._wake_loop, daemon=True, name="wake-word")
        self._wake_thread.start()

    def stop_wake(self) -> None:
        self._wake_stop.set()

    def pop_pending_dictation(self):
        """Thread-safe: returns (and clears) the last wake-word transcript."""
        with self._pending_lock:
            text = self._pending_dictation
            self._pending_dictation = None
            return text

    def _spawn_mic_stream(self):
        """Continuous 16kHz mono s16 raw mic stream on the wake cores."""
        return subprocess.Popen(
            ["taskset", "-c", WAKE_CORES, "pw-record", "--raw",
             "--rate", "16000", "--channels", "1", "--format", "s16", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=WAKE_MIC_CHUNK,
        )

    def _wake_speaking_guarded(self) -> bool:
        """True while our own TTS is speaking (echo guard) or during manual
        push-to-talk - detection is suppressed then, so Jarvis never hears
        himself. Disable with DUMMY_WAKE_ECHO_GUARD=0."""
        if not WAKE_ECHO_GUARD:
            return False
        return (self._current_procs is not None) or (time.time() < self._tts_tail_until)

    def _wake_loop(self) -> None:
        # Cosmetic: onnxruntime warns about the (intentionally hidden) CUDA
        # provider on every session init - it pollutes the prompt otherwise.
        warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")
        try:
            cores = _parse_cpu_spec(WAKE_CORES)
            if cores:
                os.sched_setaffinity(0, cores)  # pin this thread (and any onnx pool threads it creates) to E-cores
        except Exception:
            pass
        try:
            from openwakeword.model import Model
            model_path = _openwakeword_model_path()
            m = Model(wakeword_model_paths=[model_path])
            key = list(m.models.keys())[0]
        except Exception as exc:
            print(f"[voice] wake word unavailable: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return
        print(f"[voice] wake word listening for 'Jarvis' on cores {WAKE_CORES}", file=sys.stderr, flush=True)

        ambient = deque(maxlen=100)  # rolling ~8s of chunk energies while listening
        preroll = deque(maxlen=16)   # last ~1.3s of audio: prepended on trigger
        while not self._wake_stop.is_set():
            proc = None
            try:
                proc = self._spawn_mic_stream()
                while not self._wake_stop.is_set():
                    chunk = proc.stdout.read(WAKE_MIC_CHUNK)
                    if not chunk or len(chunk) < WAKE_MIC_CHUNK:
                        break  # stream died - fall through to respawn
                    preroll.append(chunk)
                    energy = float(np.mean(np.abs(np.frombuffer(chunk, dtype="<i2"))))
                    ambient.append(energy)
                    # Gate: our own voice, manual recording, or wake disabled
                    if (not self.wake_enabled or self._wake_gated.is_set()
                            or self._wake_speaking_guarded() or self.is_recording()):
                        continue
                    scores = m.predict(
                        np.frombuffer(chunk, dtype="<i2"),
                        patience={key: WAKE_PATIENCE},
                        threshold={key: WAKE_THRESHOLD},
                    )
                    if scores.get(key, 0) >= WAKE_THRESHOLD:
                        m.reset()
                        self.flush()  # barge-in: stop speaking instantly
                        # The detector fires mid-word: carry the pre-trigger
                        # audio (the start of "Jarvis" itself) into the capture
                        # so run-on commands aren't clipped.
                        seed = b"".join(preroll)
                        preroll.clear()
                        print(f"\033[96m🎙  wake candidate - verifying...\033[0m", file=sys.stderr, flush=True)
                        try:
                            amb = float(np.percentile(ambient, 10)) if ambient else 0.0
                            self._wake_capture_and_transcribe(proc.stdout, amb, seed=seed)
                        except Exception as exc:
                            # The ear must never die: log and keep listening.
                            print(f"\033[91m🎙  wake capture failed: {type(exc).__name__}: {exc}\033[0m", file=sys.stderr, flush=True)
                        m.reset()
            except Exception as exc:
                print(f"[voice] wake listener hiccup ({type(exc).__name__}: {exc}) - restarting stream", file=sys.stderr, flush=True)
            finally:
                if proc is not None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            if not self._wake_stop.is_set():
                self._wake_stop.wait(2.0)  # brief backoff, then respawn the stream

    def _wake_capture_and_transcribe(self, stream, ambient: float = 0.0, seed: bytes = b"") -> None:
        """Two-stage wake. Stage 1 (cheap, always-on): the openWakeWord
        detector at a LOW threshold - any voice or tone saying "Jarvis"
        triggers. Stage 2 (here): capture, transcribe, and VERIFY the
        transcript really contains the name before anything happens; a
        false trigger costs one silent second. If the user said only the
        name, answer "Yes, sir?" and listen once more for the command.

        `seed` is the pre-roll audio from just before the trigger (the
        start of the wake word itself), so run-on commands keep the name.

        Exceptions propagate to the caller (the wake loop), which catches
        them so the ear thread can never die. The finally block guarantees
        the detection gate is always re-opened."""
        self._wake_gated.set()
        try:
            pcm = self._capture_speech(stream, ambient=ambient, seed=seed)
            if not pcm:
                print("\033[90m🎙  (no speech after trigger)\033[0m", file=sys.stderr, flush=True)
                return
            text = self._transcribe_pcm(pcm)
            if not text:
                return
            if not _looks_like_wake(text):
                # Low-threshold false positive: discard silently.
                print(f"\033[90m🎙  (not a wake phrase: {text[:60]})\033[0m", file=sys.stderr, flush=True)
                return
            remainder = _strip_wake_words(text)
            if not remainder:
                # He only said the name. Answer like JARVIS, then listen for
                # the actual command (waiting out our own voice first).
                print("\033[96m🎙  summoned - answering 'Yes, sir?' and listening\033[0m", file=sys.stderr, flush=True)
                self.speak("Yes, sir?")
                deadline = time.time() + 8.0
                while time.time() < self._tts_tail_until and time.time() < deadline:
                    time.sleep(0.1)
                pcm2 = self._capture_speech(stream, max_sec=15.0, ambient=ambient)
                if pcm2:
                    text2 = self._transcribe_pcm(pcm2)
                    if text2:
                        remainder = _strip_wake_words(text2)
            if remainder:
                with self._pending_lock:
                    self._pending_dictation = remainder
                print(f"\033[96m🎙  heard: {remainder}\033[0m", file=sys.stderr, flush=True)
            else:
                print("\033[90m🎙  (nothing intelligible)\033[0m", file=sys.stderr, flush=True)
        finally:
            self._wake_gated.clear()

    def _transcribe_pcm(self, pcm: bytes):
        """Persist one PCM utterance and run it through the whisper worker."""
        wav_path = str(VOICE_DIR / f"wake_{int(time.time() * 1000)}.wav")
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm)
        print("\033[90m🎙  transcribing...\033[0m", file=sys.stderr, flush=True)
        text, err = self.transcribe(wav_path)
        if not text:
            print(f"\033[90m🎙  ({err})\033[0m", file=sys.stderr, flush=True)
        return text

    def _capture_speech(self, stream, max_sec: float = WAKE_MAX_UTTERANCE, ambient: float = 0.0,
                        seed: bytes = b""):
        """Energy-based endpointing on the live stream: wait for speech,
        stop after WAKE_SILENCE_END of trailing silence. Thresholds adapt to
        the measured ambient noise (env values act as floors). `seed` is
        pre-roll audio (captured before the trigger) prepended to the result."""
        start_thresh = max(WAKE_VAD_START, ambient * 2.0)
        stop_thresh = max(WAKE_VAD_STOP, ambient * 1.5)
        buf = bytearray(seed)
        speech_started = False
        if seed:
            # The trigger guarantees jarvis-like speech just happened: if the
            # pre-roll carries that energy, count it as the speech start.
            for i in range(0, len(seed) - WAKE_MIC_CHUNK + 1, WAKE_MIC_CHUNK):
                e = float(np.mean(np.abs(np.frombuffer(seed[i:i + WAKE_MIC_CHUNK], dtype="<i2"))))
                if e > start_thresh:
                    speech_started = True
                    break
        start_t = time.time()
        quiet_run = 0
        while time.time() - start_t < max_sec:
            chunk = stream.read(WAKE_MIC_CHUNK)
            if not chunk:
                break
            buf += chunk
            energy = float(np.mean(np.abs(np.frombuffer(chunk, dtype="<i2"))))
            if not speech_started:
                if time.time() - start_t > WAKE_START_TIMEOUT:
                    return None
                if energy > start_thresh:
                    speech_started = True
            else:
                quiet_run = quiet_run + 1 if energy < stop_thresh else 0
                if quiet_run * 0.08 >= WAKE_SILENCE_END:
                    break
        return bytes(buf) if speech_started else None

    # ----------------------------------------------------------- prewarm ----
    def prewarm(self) -> None:
        """Boot-time warmup so the first real interaction is fast: the whisper
        worker (its ~2.4s model load would otherwise land on the first 'Jarvis').
        The TTS worker warms naturally via the boot greeting."""
        def _warm():
            try:
                self._ensure_whisper()
            except Exception as exc:
                sys.stderr.write(f"[voice] whisper prewarm failed: {exc}\n")
        threading.Thread(target=_warm, daemon=True).start()

    # ----------------------------------------------------------- status ----
    def status(self) -> str:
        installed = list_voices()
        current = next(
            (n for n, p in installed.items() if p == self.voice_path),
            os.path.basename(self.voice_path),
        )
        return (
            f"TTS: {'ON' if self.tts_enabled else 'OFF'} (streaming: sentences are spoken WHILE the reply is generated)\n"
            f"Voice: {current} ({self.voice_rate} Hz) - installed: {', '.join(installed) or 'none'}\n"
            f"Switch with /voice voice <name> (e.g. en_GB-jarvis-high)\n"
            f"Wake word: {'ON - just say \"Jarvis\"' if self.wake_enabled and self._wake_thread and self._wake_thread.is_alive() else 'OFF'}"
            f" (two-stage: detector {WAKE_THRESHOLD} -> whisper-verified, cores {WAKE_CORES})\n"
            f"Auto-submit dictation: {'ON' if self.auto_submit else 'OFF'}\n"
            f"Whisper model: {os.getenv('DUMMY_WHISPER_MODEL', 'small')} (int8, CPU)\n"
            f"Pinned cores: TTS {TTS_CORES} | whisper {WHISPER_CORES} | "
            f"mic {VOICE_CORES} | wake word {WAKE_CORES} (E-cores)\n"
            f"Preference: saved in {VOICE_CONFIG}\n"
            f"Push-to-talk: Ctrl+T to record, Ctrl+T again to transcribe\n"
            f"Barge-in: new input or Ctrl+T instantly stops the speech"
        )


voice = VoiceEngine()
