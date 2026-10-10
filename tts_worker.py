#!/usr/bin/env python3
"""
Persistent piper TTS worker for Dummy.

Spawned (taskset-pinned) by voice.py under an interpreter that has `piper`
importable. The voice model loads ONCE and stays hot; playback goes through
a pw-play raw player that this worker owns (respawned only after a barge-in,
never between sentences) - so per-sentence overhead is ~0.

Protocol:
    stdin  (JSON per line): {"cmd": "say", "text": "..."}   one utterance
    stdout (line each):     READY                           model loaded
                            DONE <pcm_bytes>                utterance handed to the player
                            ERROR <message>                 utterance failed
    SIGUSR1:                kill the player instantly (barge-in). The current
                            utterance aborts and DONE is still emitted, so the
                            parent never blocks forever.

The parent uses the DONE byte count to estimate how long the audio will
still be draining (for the wake-word echo guard).
"""
import json
import os
import shutil
import signal
import subprocess
import sys

import numpy as np

PLAYER = shutil.which("pw-play")


def main() -> None:
    if len(sys.argv) < 2:
        print("ERROR no model path given", flush=True)
        return
    model_path = sys.argv[1]

    try:
        from piper import PiperVoice
    except ImportError as exc:
        print(f"ERROR piper not importable: {exc}", flush=True)
        return

    try:
        voice = PiperVoice.load(model_path)
    except Exception as exc:
        print(f"ERROR could not load voice: {type(exc).__name__}: {exc}", flush=True)
        return
    print("READY", flush=True)

    player = None

    def kill_player():
        nonlocal player
        if player is not None and player.poll() is None:
            try:
                player.kill()
            except Exception:
                pass
        player = None

    def handle_usr1(signum, frame):
        kill_player()

    signal.signal(signal.SIGUSR1, handle_usr1)

    def ensure_player(rate: int, channels: int):
        nonlocal player
        if player is None or player.poll() is not None:
            if PLAYER is None:
                raise RuntimeError("pw-play not found")
            player = subprocess.Popen(
                [PLAYER, "--raw", "--rate", str(rate),
                 "--channels", str(channels), "--format", "s16", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return player

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            print("ERROR bad json", flush=True)
            continue

        if msg.get("cmd") == "flush":
            kill_player()
            print("DONE 0", flush=True)
            continue

        text = str(msg.get("text", "")).strip()
        if not text:
            print("DONE 0", flush=True)
            continue

        sent = 0
        try:
            p = None
            for chunk in voice.synthesize(text):
                if p is None:
                    p = ensure_player(chunk.sample_rate, chunk.sample_channels)
                pcm = (np.clip(chunk.audio_float_array, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                try:
                    p.stdin.write(pcm)
                    sent += len(pcm)
                except (BrokenPipeError, ValueError):
                    # Barge-in killed the player mid-utterance: stop synthesizing.
                    break
            if p is not None:
                try:
                    p.stdin.flush()
                except Exception:
                    pass
            print(f"DONE {sent}", flush=True)
        except Exception as exc:
            kill_player()
            print(f"ERROR {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
