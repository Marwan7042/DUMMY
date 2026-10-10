#!/home/marwan/.venv/bin/python3
"""
Whisper STT worker for Dummy.

Launched by voice.py as a persistent, taskset-pinned subprocess so the model
loads once and stays hot. Line protocol on stdio:

    stdin:  one WAV path per line
    stdout: "OK\t<transcription>" or "ERR\t<message>"
    stderr: free-form status logs (never read by the parent)

Pinning: the parent spawns this process under
    taskset -c $DUMMY_VOICE_CORES
so all CT2 threads stay on the P-cores.

Config (env):
    DUMMY_WHISPER_MODEL  faster-whisper model size (default "small";
                         "base" = faster, "medium" = more accurate)
    DUMMY_VOICE_CORES    core list used to size the thread pool (default
                         "1,3,5,7,9,11,13,15" - odd P-core siblings, so it
                         never collides with llama-server pinned to
                         0,2,4,6,8,10,12,14)
"""
import os
import sys
import wave

import numpy as np

MODEL_SIZE = os.getenv("DUMMY_WHISPER_MODEL", "small")
CORES = os.getenv("DUMMY_VOICE_CORES", os.getenv("DUMMY_WHISPER_CORES", "0-7"))


def _decode_pcm_wav(path: str):
    """Decode any 16-bit PCM WAV (pw-record/piper output) without PyAV.

    Returns a float32 mono array resampled to 16 kHz, or None if the file is
    not a simple PCM WAV (caller then falls back to faster-whisper's
    decoder). This sidesteps av/faster-whisper version mismatches for the
    common case.
    """
    try:
        with wave.open(path, "rb") as w:
            if w.getsampwidth() != 2:
                return None
            rate = w.getframerate()
            channels = w.getnchannels()
            raw = w.readframes(w.getnframes())
    except Exception:
        return None
    if rate < 1 or channels < 1:
        return None
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        n_out = int(audio.shape[0] * 16000 / rate)
        audio = np.interp(
            np.linspace(0, audio.shape[0] - 1, n_out),
            np.arange(audio.shape[0]),
            audio,
        ).astype(np.float32)
    return audio


def _core_count(spec: str) -> int:
    """Number of CPUs in a taskset list like '1,3,5' or '0-15'."""
    n = 0
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                a, b = part.split("-")
                n += abs(int(b) - int(a)) + 1
            except ValueError:
                pass
        else:
            try:
                n += 1
            except ValueError:
                pass
    return max(1, min(n, 16))


def main() -> None:
    # Emit problems as an immediate protocol reply, then die: the parent
    # only reads a line after sending a path, and EOF is handled as "dead".
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        print(f"ERR\tfaster-whisper not installed: {exc}", flush=True)
        return

    threads = _core_count(CORES)
    print(f"[voice_worker] loading whisper '{MODEL_SIZE}' int8 on {threads} threads...", file=sys.stderr, flush=True)
    try:
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=threads)
    except Exception as exc:
        print(f"ERR\tcould not load whisper model: {type(exc).__name__}: {exc}", flush=True)
        return
    print(f"[voice_worker] whisper '{MODEL_SIZE}' int8 ready (pinned to {CORES})", file=sys.stderr, flush=True)

    for line in sys.stdin:
        path = line.strip()
        if not path:
            continue
        if not os.path.isfile(path):
            print(f"ERR\tno such file: {path}", flush=True)
            continue
        try:
            audio_arg = path
            audio = _decode_pcm_wav(path)
            if audio is not None:
                audio_arg = audio
            # vad_filter trims leading/trailing silence - much better for
            # short push-to-talk recordings.
            segments, info = model.transcribe(audio_arg, beam_size=5, vad_filter=True)
            text = " ".join(seg.text for seg in segments).strip()
            print(f"OK\t{text}", flush=True)
        except Exception as exc:
            print(f"ERR\t{type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
