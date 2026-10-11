#!/home/marwan/.venv/bin/python3
import os
# ---------------------------------------------------------------------------
# CRITICAL: Python 3.14 made 'forkserver' the default multiprocessing start
# method on Linux. OpenVINO spawns a multiprocessing worker at IMPORT time,
# and the forkserver protocol RE-EXECUTES __main__ (this file) in its helper
# process - which booted a whole second hidden Dummy session (second wake
# listener, second mic stream, duplicate logs). Forcing 'fork' restores the
# pre-3.14 behavior: a plain fork, no re-exec, no second session.
# ---------------------------------------------------------------------------
import multiprocessing
try:
    multiprocessing.set_start_method("fork", force=True)
except Exception:
    pass

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import json
import time
import subprocess
import ast
import operator
import math
import re
import signal
import hashlib
import shutil
import textwrap
import select
import threading
try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None
from pathlib import Path
import pexpect

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import choice
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

EXPAND_THINKING = False  # False = collapsed ("thinking ..."), True = expanded steps

kb = KeyBindings()

@kb.add("c-o")
def _handle_ctrl_o(event):
    global EXPAND_THINKING
    EXPAND_THINKING = not EXPAND_THINKING
    status = "EXPANDED (showing steps)" if EXPAND_THINKING else "COLLAPSED (hidden as 'thinking ...')"
    print(f"\n\033[95m💭 Thinking display: {status}\033[0m")
    event.app.invalidate()

@kb.add("c-t")
def _handle_ctrl_t(event):
    # Push-to-talk: first press starts a (P-core pinned) recording, second
    # press stops it, runs Whisper, and drops the text into the prompt.
    if voice_engine is None:
        print("\n\033[91m🎙 Voice module unavailable.\033[0m")
    elif voice_engine.is_recording():
        print("\n\033[96m🎙 Transcribing...\033[0m")
        text, err = voice_engine.stop_and_transcribe()
        if text:
            event.app.current_buffer.insert_text(text)
        elif err:
            print(f"\n\033[91m🎙 STT error: {err}\033[0m")
    else:
        # Barge-in: stop speaking so the mic doesn't pick up Jarvis' own voice.
        voice_engine.flush()
        _, rec_err = voice_engine.start_recording()
        if rec_err:
            print(f"\n\033[91m🎙 {rec_err}\033[0m")
        else:
            print("\n\033[96m🎙 Recording... press Ctrl+T again to stop & transcribe.\033[0m")
    event.app.invalidate()

# ---------------------------------------------------------------------------
# Interactive CLI input
# ---------------------------------------------------------------------------

PROMPT_HISTORY_FILE = Path.home() / ".dummy_cli_history"

prompt_style = Style.from_dict({
    "prompt": "bold ansicyan",
})

# Commands are completed after typing "/" and can be selected with ↑/↓ and Tab.
COMMAND_COMPLETER = NestedCompleter.from_nested_dict({
    "/help": None,
    "exit": None,
    "quit": None,
    "/clear": None,
    "/reload-prompt": None,
    "/new": None,
    "/reset": None,
    "/plan": {
        "clear": None,
    },
    "/rollback": None,
    "/checkpoints": None,
    "/context": None,
    "/docs": None,
    "/graph": {
        "show-all": None,
        "connections": None,
        "search": None,
        "summary": None,
        "preferences": None,
        "people": None,
        "stats": None,
        "remember": None,
        "export": {
            "json": None,
            "dot": None,
        },
        "help": None,
    },
    "/tasks": None,
    "/think": {
        "on": None,
        "off": None,
        "toggle": None,
        "expand": None,
        "collapse": None,
    },
    "/expand": None,
    "/effort": {
        "low": None,
        "medium": None,
        "high": None,
        "xhigh": None,
    },
    "/fast": None,
    "/tokens": None,
    "/tools": None,
    "/model": None,
    "/voice": {
        "on": None,
        "off": None,
        "test": None,
        "mic": None,
        "voice": None,
        "voices": None,
        "wake": {
            "on": None,
            "off": None,
        },
        "autosubmit": {
            "on": None,
            "off": None,
        },
    },
    "/nim": {
        "on": None,
        "off": None,
        "status": None,
        "model": None,
        "ask": None,
    },
})

prompt_session = PromptSession(
    history=FileHistory(str(PROMPT_HISTORY_FILE)),
    completer=COMMAND_COMPLETER,
    complete_while_typing=True,
    style=prompt_style,
    key_bindings=kb,
    multiline=False,
)

# Ensure assistant module is in path BEFORE any imports
# This must happen before any from assistant.* imports
sys.path.insert(0, os.path.expanduser(os.path.abspath(__file__)))

# Now import brain module for conversation persistence
from brain import brain
from brain import start_session, add_message, get_session, list_sessions, delete_session

# Import graph memory modules
from graph.entity_extractor import create_extractor
from graph.graph_tools import (
    get_related_entities, find_conversations_about, get_entity_timeline,
    find_relationships, get_contextual_summary, search_graph,
    get_graph_summary, get_user_preferences, get_user_connections,
    remember_fact
)
from graph.visualizer import graph_viz

extractor = create_extractor()

venv_site = os.path.expanduser("~/.venv/lib/python3.14/site-packages")
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)

try:
    from openai import OpenAI
except ImportError:
    print("Please install the OpenAI library: pip install openai")
    exit(1)

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# Local llama.cpp OpenAI-compatible API
SERVER_BASE_URL = os.getenv("DUMMY_SERVER_URL", "http://localhost:8080").rstrip("/")
MODEL_ENDPOINT = os.getenv("DUMMY_MODEL_ENDPOINT", SERVER_BASE_URL + "/v1")

# Initialize RAG Engine
try:
    from rag_engine import RAGEngine
    rag = RAGEngine()
except Exception as rag_init_err:
    rag = None
    console.print(f"[bold red]⚠️  RAG engine failed to initialize:[/bold red] {rag_init_err}")
    console.print("[dim]search_documents / index_document / list_documents will report 'not initialized' until this is fixed. Check the traceback above for the real cause (missing package, missing embedding model files, etc.) instead of guessing.[/dim]")
 


import urllib.request
import urllib.parse
from collections import deque, defaultdict

# Connect to the local llama.cpp server
client = OpenAI(base_url=MODEL_ENDPOINT, api_key="sk-no-key-required")

# ---------------------------------------------------------------------------
# Voice I/O (Piper TTS + Whisper STT, pinned to P-cores) - optional
# ---------------------------------------------------------------------------
try:
    from voice import voice as voice_engine
except Exception as voice_import_err:
    voice_engine = None
    console.print(f"[yellow]⚠️  Voice module unavailable: {voice_import_err}[/yellow]")

# ---------------------------------------------------------------------------
# NVIDIA NIM (free cloud API) - hybrid routing + auto-fallback when the
# local llama-server is unreachable.
# ---------------------------------------------------------------------------
NIM_BASE_URL = os.getenv("DUMMY_NIM_URL", "https://integrate.api.nvidia.com/v1")
NIM_DEFAULT_MODEL = os.getenv("DUMMY_NIM_MODEL", "z-ai/glm-5.3")

def _load_nim_api_key():
    """$DUMMY_NIM_KEY or $NVIDIA_API_KEY, else the 'NVIDIANIM:' line in ~/keys.txt."""
    key = os.getenv("DUMMY_NIM_KEY") or os.getenv("NVIDIA_API_KEY")
    if key:
        return key.strip()
    try:
        for line in (Path.home() / "keys.txt").read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("nvidianim") or low.startswith("nvidia_nim"):
                return line.split(":", 1)[1].strip() if ":" in line else line.split(None, 1)[1].strip()
    except Exception:
        pass
    return None

NIM_API_KEY = _load_nim_api_key()
nim_client = OpenAI(base_url=NIM_BASE_URL, api_key=NIM_API_KEY) if NIM_API_KEY else None


SESSION_TITLE_FILE = os.path.expanduser("~/.dummy_session_titles.json")

def _load_session_titles():
    try:
        with open(SESSION_TITLE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_session_titles(titles):
    directory = os.path.dirname(SESSION_TITLE_FILE)
    os.makedirs(directory, exist_ok=True)
    tmp = SESSION_TITLE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(titles, f, ensure_ascii=False, indent=2)
    os.replace(tmp, SESSION_TITLE_FILE)


SESSION_TITLES = _load_session_titles()

def _clean_session_title(raw: str) -> str:
    title = (raw or "").strip()
    title = re.sub(r"```.*?```", "", title, flags=re.S).strip()
    title = re.sub(r"^(title|conversation|name)\s*[:=-]\s*", "", title, flags=re.I).strip()
    title = title.strip('\"\'`')
    title = re.sub(r"\s+", " ", title)
    if "\n" in title:
        title = title.splitlines()[0].strip()
    if len(title) > 72:
        title = title[:69].rsplit(" ", 1)[0] + "..."
    return title or "Untitled conversation"


def _looks_like_session_id(value: str, session_id: str = "") -> bool:
    value = (value or "").strip().strip("\"'`")
    compact = value.replace("-", "")
    return (
        bool(value)
        and (
            value.lower() == str(session_id).lower()
            or (len(compact) >= 8 and len(compact) <= 64
                and all(ch in "0123456789abcdefABCDEF" for ch in compact))
        )
    )


def _generate_session_title(session_id: str, user_text: str, force: bool = False) -> str:
    key = str(session_id)
    if (
        not force
        and key in SESSION_TITLES
        and SESSION_TITLES[key].strip()
        and not _looks_like_session_id(SESSION_TITLES[key], key)
    ):
        return SESSION_TITLES[key]

    prompt = (
        "Generate a concise title for this conversation. "
        "Capture the concrete subject or goal, not the conversational framing. "
        "Use 3-8 words. No quotes, no punctuation at the end, no prefix, no explanation.\n\n"
        f"User's first message:\n{user_text[:4000]}"
    )
    try:
        resp = client.chat.completions.create(
            model="llama.cpp",
            messages=[
                {"role": "system", "content": "You generate short, accurate conversation titles."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            max_tokens=24,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content if resp.choices else ""
        title = _clean_session_title(raw)

        # Never accept a backend/session ID as a visible conversation title.
        if _looks_like_session_id(title, key):
            title = "Untitled conversation"

        if title != "Untitled conversation":
            SESSION_TITLES[key] = title
            _save_session_titles(SESSION_TITLES)

        return title
    except Exception as exc:
        console.print(f"[dim]Could not generate conversation title: {exc}[/dim]")
        return "Untitled conversation"


class ContextManager:
    """Server-aware context budgeting and hierarchical conversation compaction.

    Never assumes a fixed context size. Reads llama.cpp /props and /v1/models,
    reserves space for the requested generation, accounts for tool-schema cost,
    and compacts only older turns while preserving the active tool exchange.
    """
    def __init__(self, base_url: str = SERVER_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.max_context_tokens = self._as_int(os.getenv("DUMMY_FALLBACK_CONTEXT")) or 16384
        self.source = "fallback"
        self.last_refresh = 0.0
        self.refresh_interval = 10.0
        self.safety_fraction = 0.08
        self.min_safety_tokens = 512
        self.tool_overhead_tokens = 256
        self.history_ledger = deque(maxlen=12)

    @staticmethod
    def _as_int(value):
        try:
            value = int(value)
            return value if value > 0 else None
        except (TypeError, ValueError):
            return None

    def refresh(self, force: bool = False) -> int:
        now = time.time()
        if not force and (now - self.last_refresh) < self.refresh_interval:
            return self.max_context_tokens

        candidates = []

        # llama.cpp native properties. Different builds expose slightly
        # different shapes, so inspect several known fields conservatively.
        try:
            with urllib.request.urlopen(self.base_url + "/props", timeout=2) as response:
                props = json.loads(response.read().decode("utf-8", errors="replace"))
                dgs = props.get("default_generation_settings", {}) or {}
                for key in ("n_ctx", "n_ctx_train", "context_size"):
                    candidates.append((f"/props:{key}", self._as_int(dgs.get(key))))
                    candidates.append((f"/props:{key}", self._as_int(props.get(key))))
        except Exception:
            pass

        # If the HTTP metadata endpoint is unavailable, inspect the live
        # llama-server command line. This catches the exact runtime -c/--ctx-size.
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                cmdline_path = os.path.join("/proc", entry, "cmdline")
                try:
                    raw = Path(cmdline_path).read_bytes().decode("utf-8", errors="ignore")
                except Exception:
                    continue
                argv = [x for x in raw.split("\x00") if x]
                if not argv or os.path.basename(argv[0]) not in {"llama-server", "llama-server.exe"}:
                    continue
                for i, arg in enumerate(argv):
                    if arg in {"-c", "--ctx-size", "--context-size"} and i + 1 < len(argv):
                        candidates.append(("/proc/llama-server:" + arg, self._as_int(argv[i + 1])))
                        break
                    if arg.startswith("--ctx-size="):
                        candidates.append(("/proc/llama-server:--ctx-size", self._as_int(arg.split("=", 1)[1])))
                        break
        except Exception:
            pass

        # OpenAI-compatible model metadata may advertise context_length.
        try:
            with urllib.request.urlopen(self.base_url + "/v1/models", timeout=2) as response:
                data = json.loads(response.read().decode("utf-8", errors="replace"))
                for model in data.get("data", []) or []:
                    for key in ("context_length", "max_context_length", "n_ctx"):
                        candidates.append((f"/v1/models:{key}", self._as_int(model.get(key))))
        except Exception:
            pass

        valid = [(src, n) for src, n in candidates if n and n >= 1024]
        if valid:
            # Prefer an explicitly configured llama.cpp runtime context over
            # a training-context advertisement. The first valid /props value
            # wins; otherwise use model metadata.
            proc_valid = [x for x in valid if x[0].startswith('/proc/llama-server:')]
            props_valid = [x for x in valid if x[0].startswith('/props:')]
            src, n_ctx = proc_valid[0] if proc_valid else (props_valid[0] if props_valid else valid[0])
            self.max_context_tokens = n_ctx
            self.source = src

        self.last_refresh = now
        return self.max_context_tokens

    def safety_tokens(self) -> int:
        return max(self.min_safety_tokens, int(self.max_context_tokens * self.safety_fraction))

    @staticmethod
    def estimate_tokens(text) -> int:
        if text is None:
            return 0
        # A conservative English/code estimate. Exact llama.cpp tokenizer
        # accounting is server-side; this only needs to keep requests safely
        # below n_ctx.
        chars = len(str(text))
        return max(1, (chars + 3) // 3)

    def message_tokens(self, message: dict) -> int:
        cost = 8
        for key in ("role", "content", "name", "tool_call_id"):
            if key in message and message[key] is not None:
                cost += self.estimate_tokens(message[key])
        if message.get("tool_calls"):
            cost += self.estimate_tokens(json.dumps(message["tool_calls"], ensure_ascii=False))
        return cost

    def tools_tokens(self, tool_defs) -> int:
        if not tool_defs:
            return 0
        return self.estimate_tokens(json.dumps(tool_defs, ensure_ascii=False)) + self.tool_overhead_tokens

    def estimate_request_tokens(self, msg_list: list, tool_defs=None) -> int:
        return sum(self.message_tokens(m) for m in msg_list) + self.tools_tokens(tool_defs)

    def available_generation_tokens(self, msg_list: list, requested_output: int, tool_defs=None) -> int:
        self.refresh()
        safety = self.safety_tokens()
        prompt = self.estimate_request_tokens(msg_list, tool_defs)
        available = self.max_context_tokens - prompt - safety
        return max(0, min(int(requested_output), available))

    def _compact_text(self, text: str, max_chars: int = 900) -> str:
        text = str(text or "")
        if len(text) <= max_chars:
            return text
        head = max_chars // 2
        tail = max_chars - head
        return text[:head] + "\n…[older context compacted]…\n" + text[-tail:]

    def _summarize_turn_for_ledger(self, turn: list) -> str:
        user_text = next((str(m.get("content", "")) for m in turn if m.get("role") == "user"), "")
        assistant_text = next((str(m.get("content", "")) for m in reversed(turn) if m.get("role") == "assistant"), "")
        tool_names = [m.get("name") for m in turn if m.get("role") == "tool" and m.get("name")]
        bits = [f"User: {self._compact_text(user_text, 420)}"]
        if assistant_text:
            bits.append(f"Assistant: {self._compact_text(assistant_text, 420)}")
        if tool_names:
            bits.append("Tools: " + ", ".join(tool_names[:10]))
        return " | ".join(bits)

    def _split_turns(self, history: list):
        turns, current = [], []
        for msg in history:
            if msg.get("role") == "user" and current:
                turns.append(current)
                current = []
            current.append(msg)
        if current:
            turns.append(current)
        return turns

    def _collapse_old_tool_outputs(self, history: list) -> tuple[list, int]:
        if len(history) < 4:
            return history, 0
        # Preserve the current/active turn intact. Older tool outputs can be
        # safely compacted because the original full output is already logged
        # by truncate_tool_output and the agent can call the tool again.
        turns = self._split_turns(history)
        changed = 0
        for ti, turn in enumerate(turns[:-2]):
            for m in turn:
                if m.get("role") == "tool":
                    content = str(m.get("content") or "")
                    if len(content) > 900:
                        m["content"] = self._compact_text(content, 700)
                        changed += 1
        flattened = [m for turn in turns for m in turn]
        return flattened, changed

    def prepare(self, msg_list: list, requested_output: int, tool_defs=None, *, verbose: bool = False) -> tuple[list, int, int]:
        """Return (request_messages, effective_max_output, estimated_prompt_tokens)."""
        self.refresh()
        work = [dict(m) for m in msg_list]
        if work:
            work[0] = dict(work[0])

        # Attach the compact ledger to the system prompt without polluting the
        # persistent message list. It is only injected when older context was
        # actually dropped.
        base_system = str(work[0].get("content", "")) if work else ""
        if self.history_ledger:
            ledger = "\n\nEARLIER CONTEXT LEDGER (older turns were compacted; do not assume omitted details):\n" + "\n".join(
                f"- {item}" for item in list(self.history_ledger)[-8:]
            )
            if work:
                work[0]["content"] = base_system + ledger

        def fits(candidate):
            prompt = self.estimate_request_tokens(candidate, tool_defs)
            return prompt + self.safety_tokens() + max(128, min(int(requested_output), self.max_context_tokens // 2)) <= self.max_context_tokens

        changed = False

        # Stage 1: compact older tool payloads.
        work, collapsed = self._collapse_old_tool_outputs(work)
        if collapsed:
            changed = True
            if verbose:
                console.print(f"[dim yellow]🧹 Context: compacted {collapsed} older tool outputs.[/dim yellow]")

        # Stage 2: drop oldest complete turns until the request fits. Never
        # split the current user/tool exchange.
        while not fits(work):
            history = work[1:]
            turns = self._split_turns(history)
            if len(turns) <= 1:
                break
            removed = turns.pop(0)
            self.history_ledger.append(self._summarize_turn_for_ledger(removed))
            work = [work[0]] + [m for turn in turns for m in turn]
            changed = True

        # Stage 3: if the single surviving turn is itself huge, aggressively
        # compact only its older tool results; user request and assistant/tool
        # protocol messages remain intact.
        if not fits(work):
            for m in work[1:]:
                if m.get("role") == "tool":
                    m["content"] = self._compact_text(m.get("content", ""), 350)
            changed = True

        if changed and len(msg_list) > 0:
            # Commit the compacted history back to the canonical conversation
            # so the same old turns are not re-compacted on every tool round.
            msg_list[:] = [{"role": "system", "content": base_system}] + work[1:]
            # Rebuild the request system message with the ledger for this call.
            work[0] = dict(msg_list[0])
            if self.history_ledger:
                work[0]["content"] = base_system + "\n\nEARLIER CONTEXT LEDGER (older turns were compacted; do not assume omitted details):\n" + "\n".join(
                    f"- {item}" for item in list(self.history_ledger)[-8:]
                )

        prompt_tokens = self.estimate_request_tokens(work, tool_defs)
        available = max(128, self.max_context_tokens - prompt_tokens - self.safety_tokens())
        effective_output = max(128, min(int(requested_output), available))

        if changed and verbose:
            console.print(
                f"[dim yellow]🧠 Context compacted dynamically: ~{prompt_tokens:,}/{self.max_context_tokens:,} prompt tokens; generation budget {effective_output:,}.[/dim yellow]"
            )
        return work, effective_output, prompt_tokens

    def status(self, msg_list: list, tool_defs=None) -> str:
        self.refresh(force=True)
        prompt = self.estimate_request_tokens(msg_list, tool_defs)
        return (
            f"Server context: {self.max_context_tokens:,} tokens ({self.source})\n"
            f"Estimated current request: {prompt:,} tokens\n"
            f"Safety reserve: {self.safety_tokens():,} tokens\n"
            f"Estimated remaining before output: {max(0, self.max_context_tokens - prompt - self.safety_tokens()):,} tokens"
        )

context_manager = ContextManager()


# Base Directories
DUMMY_DIR = os.path.expanduser("~/.dummy")
TASKS_DIR = os.path.join(DUMMY_DIR, "tasks")
BACKUPS_DIR = os.path.join(DUMMY_DIR, "backups")
LOGS_DIR = os.path.join(DUMMY_DIR, "logs")

for d in [DUMMY_DIR, TASKS_DIR, BACKUPS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# Canonical local paths for repositories acquired through the GitHub tools.
REPO_ALIASES: dict[str, str] = {}

def _register_repo_alias(owner: str, repo: str, local_path: str) -> None:
    local_path = os.path.abspath(local_path)
    REPO_ALIASES[f"{owner}/{repo}"] = local_path
    short = repo
    existing = REPO_ALIASES.get(short)
    if existing is None or os.path.abspath(existing) == local_path:
        REPO_ALIASES[short] = local_path

def _resolve_repo_alias_path(p: str) -> str | None:
    if not p or os.path.isabs(p):
        return None
    normalized = p.replace("\\", "/").lstrip("./")
    parts = normalized.split("/", 1)
    alias = parts[0] if parts else ""
    root = REPO_ALIASES.get(alias)
    if not root:
        return None
    tail = parts[1] if len(parts) > 1 else ""
    return os.path.abspath(os.path.join(root, tail)) if tail else os.path.abspath(root)

def resolve_path(p: str) -> str:
    if not p:
        return ""
    original = p
    p = p.strip("'\"`")
    p = p.replace('\\/', '/').replace('\\', '/')

    alias_candidate = _resolve_repo_alias_path(p)
    if alias_candidate is not None:
        return alias_candidate

    # The PTY shell maintains its own cwd. Consult it for relative paths that
    # are not found in Python's process cwd. This keeps clone/shell state and
    # filesystem tools consistent without globally changing Python's cwd.
    if not os.path.isabs(p) and 'persistent_shell' in globals():
        try:
            shell_cwd = persistent_shell.get_cwd()
            shell_candidate = os.path.abspath(os.path.join(shell_cwd, p))
            if os.path.exists(shell_candidate):
                return shell_candidate
        except Exception:
            pass

    rewritten = False
    if p.startswith("/Downloads"):
        p = "/home/marwan" + p
        rewritten = True
    elif p.startswith("/Documents"):
        p = "/home/marwan" + p
        rewritten = True
    elif p.startswith("/models"):
        p = "/home/marwan" + p
        rewritten = True
    elif p.startswith("/assistant"):
        p = "/home/marwan" + p
        rewritten = True
    expanded = os.path.abspath(os.path.expanduser(p))
    if not os.path.exists(expanded):
        alt = os.path.join("/home/marwan", p.lstrip("/"))
        if os.path.exists(alt):
            if alt != expanded:
                console.print(f"[dim yellow]⚠️  resolve_path: '{original}' not found; falling back to '{alt}'[/dim yellow]")
            return alt
    if rewritten and expanded != os.path.abspath(os.path.expanduser(original)):
        console.print(f"[dim yellow]⚠️  resolve_path: rewrote '{original}' -> '{expanded}'[/dim yellow]")
    return expanded


# =====================================================================
# 1. Output Truncation & Summarization Engine
# =====================================================================
def truncate_tool_output(output: str, tool_name: str = "tool", max_chars: int | None = None, max_lines: int | None = None) -> str:
    """Log full tool output and return a context-sized excerpt.

    The caller may pass explicit limits, but normal operation derives a
    conservative limit from the live llama.cpp context rather than a fixed
fixed-token assumption.
    """
    if not output:
        return output

    context_manager.refresh()
    if max_chars is None:
        # Roughly 1–2% of context, with practical bounds. This prevents a
        # single shell result from dominating the conversation.
        max_chars = max(1800, min(14000, context_manager.max_context_tokens * 3))
    if max_lines is None:
        max_lines = max(60, min(300, max_chars // 70))

    lines = output.splitlines()
    if len(output) <= max_chars and len(lines) <= max_lines:
        return output

    ts = int(time.time() * 1000)
    log_file = os.path.join(LOGS_DIR, f"{tool_name}_{ts}.log")
    try:
        with open(log_file, "w", encoding="utf-8", errors="ignore") as f:
            f.write(output)
    except Exception:
        pass

    if len(lines) > max_lines:
        head_count = max(1, max_lines // 3)
        tail_count = max_lines - head_count
        head = "\n".join(lines[:head_count])
        tail = "\n".join(lines[-tail_count:])
        omitted = len(lines) - max_lines
        return (
            f"{head}\n\n"
            f"[… OUTPUT TRUNCATED: {omitted} lines omitted for context budget …]\n"
            f"[… Full output: {log_file} …]\n\n"
            f"{tail}"
        )

    head_chars = max_chars // 3
    tail_chars = max_chars - head_chars
    return (
        f"{output[:head_chars]}\n\n"
        f"[… OUTPUT TRUNCATED: {len(output) - max_chars} characters omitted for context budget …]\n"
        f"[… Full output: {log_file} …]\n\n"
        f"{output[-tail_chars:]}"
    )


# =====================================================================
# 3. URL Fetching Tool
# =====================================================================
def fetch_url(url: str) -> str:
    """Fetch content from any URL and return the response body.
    
    Args:
        url: The URL to fetch (GitHub raw files, web pages, API endpoints, etc.)
    
    Returns:
        URL content with status info
    
    Example:
        fetch_url("https://raw.githubusercontent.com/user/repo/main/src/main.cpp")
    """
    import urllib.request
    import urllib.error
    
    # Security: Only allow http and https to prevent file:// attacks
    if not (url.startswith("http://") or url.startswith("https://")):
        return f"Error: Invalid URL scheme. Only http:// and https:// are permitted for safety. URL: {url}"

    try:
        # Add User-Agent to avoid some server blocks
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            status = response.status
            content_type = response.headers.get("Content-Type", "unknown")
            encoding = response.headers.get("Content-Encoding", "utf-8")
            
            # Read content
            content = response.read().decode("utf-8", errors="replace")
            
            return f"URL: {url}\n" \
                   f"Status: {status}\n" \
                   f"Content-Type: {content_type}\n" \
                   f"Encoding: {encoding}\n" \
                   f"Size: {len(content)} bytes\n" \
                   f"\n--- CONTENT ---\n" \
                   f"{content}"
    except urllib.error.HTTPError as e:
        return f"HTTP Error: {e.code} - {e.reason}\nURL: {url}"
    except urllib.error.URLError as e:
        return f"URL Error: {e.reason}\nURL: {url}"
    except Exception as e:
        return f"Fetch failed: {e}\nURL: {url}"


# =====================================================================
# 2. Git Checkpointing & Transactional Rollbacks
# =====================================================================
class CheckpointManager:
    """Git-backed, content-addressed checkpoints for file mutations.

    Crucially, this never stages or commits the user's entire working tree.
    For files inside a Git repository it stores the exact pre-mutation bytes
    as Git objects plus metadata. Rollback restores only the files that Dummy
    actually checkpointed, preserving unrelated user work. Outside Git, it
    falls back to the previous snapshot mechanism.
    """
    def __init__(self, backup_dir: str = BACKUPS_DIR):
        self.backup_dir = backup_dir
        self.checkpoints = []
        self.counter = 0

    def _git_root(self, path: str) -> str | None:
        directory = path if os.path.isdir(path) else os.path.dirname(path)
        try:
            res = subprocess.run(
                ["git", "-C", directory, "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=3
            )
            if res.returncode == 0:
                return os.path.realpath(res.stdout.strip())
        except Exception:
            pass
        return None

    def _git_blob(self, path: str) -> str | None:
        try:
            with open(path, "rb") as f:
                proc = subprocess.run(
                    ["git", "hash-object", "-w", "--stdin"],
                    input=f.read(), capture_output=True, timeout=5
                )
            if proc.returncode == 0:
                return proc.stdout.decode().strip()
        except Exception:
            pass
        return None

    def create_checkpoint(self, filepath: str = "", description: str = "") -> dict | None:
        if not filepath:
            return None

        abs_path = os.path.abspath(resolve_path(filepath))
        self.counter += 1
        record = {
            "id": f"cp-{int(time.time())}-{self.counter}",
            "target": abs_path,
            "desc": description,
            "time": time.time(),
            "exists": os.path.exists(abs_path),
            "mode": None,
            "git_root": None,
            "git_rel": None,
            "blob": None,
            "fallback_backup": None,
        }

        if os.path.exists(abs_path):
            try:
                record["mode"] = os.stat(abs_path).st_mode & 0o7777
            except OSError:
                pass

        root = self._git_root(abs_path)
        if root and os.path.isfile(abs_path):
            blob = self._git_blob(abs_path)
            if blob:
                record["git_root"] = root
                record["git_rel"] = os.path.relpath(abs_path, root)
                record["blob"] = blob
                self.checkpoints.append(record)
                return record

        # Non-Git or non-regular-file fallback.
        if os.path.isfile(abs_path):
            try:
                bpath = os.path.join(self.backup_dir, f"{record['id']}_{os.path.basename(abs_path)}")
                shutil.copy2(abs_path, bpath)
                record["fallback_backup"] = bpath
            except Exception as e:
                console.print(f"[dim red]Warning: could not create fallback checkpoint for '{abs_path}': {e}[/dim red]")

        self.checkpoints.append(record)
        return record

    def _restore_git_blob(self, cp: dict) -> bool:
        root = cp.get("git_root")
        blob = cp.get("blob")
        target = cp.get("target")
        if not (root and blob and target):
            return False
        try:
            proc = subprocess.run(
                ["git", "-C", root, "cat-file", "blob", blob],
                capture_output=True, timeout=5
            )
            if proc.returncode != 0:
                return False
            with open(target, "wb") as f:
                f.write(proc.stdout)
            if cp.get("mode") is not None:
                os.chmod(target, cp["mode"])
            return True
        except Exception:
            return False

    def rollback(self, steps: int = 1) -> str:
        if not self.checkpoints:
            return "No previous checkpoints available to roll back."

        steps = max(1, int(steps))
        actions = []
        for _ in range(min(steps, len(self.checkpoints))):
            cp = self.checkpoints.pop()
            target = cp.get("target")
            desc = cp.get("desc", "mutation")

            if cp.get("blob"):
                if self._restore_git_blob(cp):
                    actions.append(f"Restored '{target}' from Git object {cp['blob'][:12]} ({desc}).")
                    continue

            backup = cp.get("fallback_backup")
            if backup and os.path.isfile(backup) and target:
                try:
                    shutil.copy2(backup, target)
                    if cp.get("mode") is not None:
                        os.chmod(target, cp["mode"])
                    actions.append(f"Restored '{target}' from fallback checkpoint ({desc}).")
                    continue
                except Exception as e:
                    actions.append(f"Error restoring '{target}': {e}")
                    continue

            if not cp.get("exists") and target and os.path.exists(target):
                try:
                    os.remove(target)
                    actions.append(f"Removed file created by Dummy: '{target}' ({desc}).")
                except Exception as e:
                    actions.append(f"Error removing created file '{target}': {e}")
            else:
                actions.append(f"Checkpoint for '{target}' could not be restored automatically.")

        return "\n".join(actions)

    def list_checkpoints(self, limit: int = 20) -> str:
        rows = self.checkpoints[-max(1, int(limit)):]
        if not rows:
            return "No active checkpoints."
        lines = ["Active Dummy checkpoints (newest last):"]
        for cp in rows:
            backend = f"git:{cp['blob'][:12]}" if cp.get("blob") else "snapshot"
            lines.append(f"- {cp['id']} | {backend} | {cp['target']} | {cp.get('desc','mutation')}")
        return "\n".join(lines)

checkpoint_mgr = CheckpointManager()

def rollback(steps: int = 1) -> str:
    return checkpoint_mgr.rollback(steps=steps)


# =====================================================================
# 3. File Mutation & Structured Diff Engine
# =====================================================================
def write_file(filepath: str, content: str) -> str:
    """Overwrites or creates a file safely with transactional snapshotting and validation."""
    try:
        filepath = resolve_path(filepath)
        
        if not permission_broker.check_mutation(filepath, "write_file"):
            return f"Permission denied: Modification of '{filepath}' was aborted by user."

        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Validate python syntax if target is .py
        if filepath.endswith(".py"):
            try:
                ast.parse(content)
            except SyntaxError as syn_err:
                return (
                    f"Syntax Validation Error: The provided content has a SyntaxError at line {syn_err.lineno}: {syn_err.msg}.\n"
                    f"File '{filepath}' was NOT written. Please correct the syntax and try again."
                )

        checkpoint_mgr.create_checkpoint(filepath, description=f"write_file({os.path.basename(filepath)})")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        line_count = len(content.splitlines())
        return f"Successfully wrote {len(content)} characters ({line_count} lines) to '{filepath}'."
    except Exception as e:
        return f"Error writing to file '{filepath}': {str(e)}"

def apply_diff(filepath: str, search_block: str, replace_block: str) -> str:
    """Aider-style search/replace block patch engine with multi-level tolerance:
    1. Exact match
    2. Whitespace-tolerant match (trailing whitespace stripped)
    3. Indentation-normalized block matching (leading indent shifts)
    Includes syntax validation before writing to disk."""
    try:
        filepath = resolve_path(filepath)
        if not os.path.isfile(filepath):
            return f"Error: File not found: '{filepath}'"

        if not permission_broker.check_mutation(filepath, "apply_diff"):
            return f"Permission denied: Modification of '{filepath}' was aborted by user."

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        new_content = None
        s_cnt = len(search_block.splitlines())
        r_cnt = len(replace_block.splitlines())

        # Level 1: Exact match
        if search_block in content:
            count = content.count(search_block)
            if count > 1:
                return f"Error: search_block matches {count} locations in '{filepath}'. Please provide more surrounding lines for unique context."
            new_content = content.replace(search_block, replace_block, 1)

        if new_content is None:
            # Level 2: Line-by-line whitespace-tolerant match (strip trailing spaces)
            content_lines = content.splitlines(keepends=True)
            search_lines = search_block.splitlines(keepends=True)

            s_stripped = [l.rstrip() for l in search_lines]
            c_stripped = [l.rstrip() for l in content_lines]

            n_s = len(s_stripped)
            matches = []
            for i in range(len(c_stripped) - n_s + 1):
                if c_stripped[i:i + n_s] == s_stripped:
                    matches.append(i)

            if len(matches) == 1:
                idx = matches[0]
                r_lines = replace_block.splitlines(keepends=True)
                if r_lines and not r_lines[-1].endswith('\n') and (idx + n_s < len(content_lines) or content.endswith('\n')):
                    r_lines[-1] = r_lines[-1] + '\n'
                new_lines = content_lines[:idx] + r_lines + content_lines[idx + n_s:]
                new_content = "".join(new_lines)
            elif len(matches) > 1:
                return f"Error: search_block matches {len(matches)} locations in '{filepath}'. Please provide more surrounding lines."

        if new_content is None:
            # Level 3: Normalized indentation matching (handles shifted 2 vs 4 spaces)
            content_lines = content.splitlines(keepends=True)
            search_lines = search_block.splitlines(keepends=True)

            s_dedent = textwrap.dedent("".join(search_lines)).splitlines()
            s_dedent_stripped = [l.rstrip() for l in s_dedent]
            n_s = len(search_lines)

            matches = []
            for i in range(len(content_lines) - n_s + 1):
                candidate = "".join(content_lines[i:i + n_s])
                c_dedent = textwrap.dedent(candidate).splitlines()
                if [l.rstrip() for l in c_dedent] == s_dedent_stripped:
                    matches.append(i)

            if len(matches) == 1:
                idx = matches[0]
                # Determine original indentation of target block
                orig_lead = re.match(r"^\s*", content_lines[idx]).group(0)
                # Re-indent replace block to match target indentation
                r_dedent = textwrap.dedent(replace_block)
                r_indented = textwrap.indent(r_dedent, orig_lead)
                r_lines = r_indented.splitlines(keepends=True)
                if r_lines and not r_lines[-1].endswith('\n') and (idx + n_s < len(content_lines) or content.endswith('\n')):
                    r_lines[-1] = r_lines[-1] + '\n'
                new_lines = content_lines[:idx] + r_lines + content_lines[idx + n_s:]
                new_content = "".join(new_lines)
            elif len(matches) > 1:
                return f"Error: search_block matches {len(matches)} locations (indentation-normalized) in '{filepath}'. Please provide more surrounding lines."

        if new_content is None:
            return f"Error: search_block not found in '{filepath}'. Ensure search_block matches actual file content."

        # Pre-write syntax validation
        if filepath.endswith(".py"):
            try:
                ast.parse(new_content)
            except SyntaxError as syn_err:
                return (
                    f"Syntax Validation Error: Applying this diff would introduce a Python SyntaxError at line {syn_err.lineno}: {syn_err.msg}.\n"
                    f"Patch was aborted to prevent corrupting '{filepath}'. Please check your replacement block indentation/syntax."
                )

        # Take checkpoint before writing
        checkpoint_mgr.create_checkpoint(filepath, description=f"apply_diff({os.path.basename(filepath)})")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)

        return f"Successfully applied diff to '{filepath}' (-{s_cnt} lines, +{r_cnt} lines)."
    except Exception as e:
        return f"Error applying diff to '{filepath}': {str(e)}"


# =====================================================================
# 4. Stateful Shell & Persistent PTY Execution
# =====================================================================
class PersistentShell:
    """Maintains a persistent PTY bash session preserving cwd, environment variables,
    virtualenvs, and background processes across tool calls."""
    PROMPT_MARKER = "___DUMMY_SHELL_PROMPT___"

    def __init__(self):
        self.child = None
        self._spawn()

    def _spawn(self):
        if self.child and self.child.isalive():
            try:
                self.child.close()
            except Exception:
                pass
        # Set large terminal width (1000 cols) to prevent hard linebreaks/wrapping at 80 cols
        self.child = pexpect.spawn(
            "/bin/bash", ["--norc", "--noprofile"],
            encoding="utf-8", echo=False, timeout=30,
            dimensions=(100, 1000)
        )
        try:
            self.child.setwinsize(100, 1000)
        except Exception:
            pass

        self.child.sendline(f'export PS1="{self.PROMPT_MARKER}"')
        self.child.expect(re.escape(self.PROMPT_MARKER))

        # Sensible environment defaults for workshop tools
        self.child.sendline('export TERM=dumb')
        self.child.expect(re.escape(self.PROMPT_MARKER))
        self.child.sendline('export PATH="/home/marwan/.venv/bin:/usr/local/cuda/bin:$PATH"')
        self.child.expect(re.escape(self.PROMPT_MARKER))

        # Start in home directory
        self.child.sendline("cd ~")
        self.child.expect(re.escape(self.PROMPT_MARKER))

    def is_alive(self) -> bool:
        return self.child is not None and self.child.isalive()

    def get_cwd(self) -> str:
        code, out = self.run("pwd", timeout=3)
        return out.strip() if code == 0 else os.getcwd()

    def run(self, command: str, timeout: int = 30) -> tuple[int, str]:
        if not self.is_alive():
            self._spawn()

        clean_cmd = command.strip()

        # Handle background execution if command ends with '&'
        if clean_cmd.endswith("&"):
            self.child.sendline(f"{clean_cmd}")
            time.sleep(0.3)
            try:
                self.child.expect(re.escape(self.PROMPT_MARKER), timeout=3)
                out = self.child.before.strip()
                return 0, f"[Started in background]\n{out}"
            except Exception:
                return 0, f"Command launched in background: {clean_cmd}"

        delim = f"__EC_{int(time.time() * 1000)}__"
        # Compound command to emit exit code and prompt cleanly
        payload = f'{clean_cmd} ; echo "{delim}:$?"'
        self.child.sendline(payload)

        try:
            self.child.expect(re.escape(self.PROMPT_MARKER), timeout=timeout)
            raw = self.child.before
            m = re.search(rf"{delim}:(\d+)", raw)
            exit_code = int(m.group(1)) if m else 0
            out = re.sub(rf"\r?\n?{delim}:\d+", "", raw)
            out = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", out)
            out = out.replace("\r\n", "\n").strip()
            return exit_code, out
        except pexpect.TIMEOUT:
            self.child.sendintr()
            time.sleep(0.3)
            try:
                self.child.expect(re.escape(self.PROMPT_MARKER), timeout=3)
            except Exception:
                self._spawn()
            return -1, f"Command timed out after {timeout} seconds and was interrupted via SIGINT."
        except KeyboardInterrupt:
            # Ctrl+C while a shell command is running: forward SIGINT to the
            # CHILD process, not the whole Dummy process, and resync the
            # prompt - instead of letting the interrupt propagate up and
            # kill the entire CLI.
            self.child.sendintr()
            time.sleep(0.3)
            try:
                self.child.expect(re.escape(self.PROMPT_MARKER), timeout=3)
            except Exception:
                self._spawn()
            return -1, "Command was interrupted by user (Ctrl+C)."


persistent_shell = PersistentShell()


# =====================================================================
# 5. Permission Scaffolding & Human-in-the-Loop Broker
# =====================================================================
class PermissionBroker:
    """Classifies actions into Read-Only, Mutating, and Destructive commands,
    enforcing human confirmation for high-risk operations."""
    DESTRUCTIVE_PATTERN = re.compile(
        r"\b(rm\s+(-[a-zA-Z]*[rf]|--recursive|--force)|rmdir|sudo\b|dd\s+|mkfs\b|chmod\s+(-R|--recursive|\d{3,4})|chown\s+(-R|--recursive)|shutdown\b|reboot\b|poweroff\b|kill\s+-9|killall\b|pkill\b|git\s+reset\s+--hard|git\s+clean\s+-[a-zA-Z]*f|truncate\b|> /dev/|find\s+.*\s+-delete)",
        re.IGNORECASE
    )

    SENSITIVE_PREFIXES = [
        "/etc", "/boot", "/bin", "/sbin", "/usr", "/lib",
        os.path.expanduser("~/.bashrc"),
        os.path.expanduser("~/.profile"),
        os.path.expanduser("~/.bash_profile"),
        os.path.expanduser("~/.ssh"),
    ]

    SENSITIVE_READ_PATHS = [
        os.path.expanduser("~/.ssh"),
        os.path.expanduser("~/keys.txt"),
        os.path.expanduser("~/.env"),
        os.path.expanduser("~/.bash_history"),
    ]

    SAFE_COMMANDS = {
        "ls", "pwd", "git status", "grep", "cat", "whoami", "date", "echo",
        "df", "du", "top", "ps", "hostname", "uname"
    }

    def __init__(self):
        self.mode = "standard"  # 'standard' (ask destructive/sensitive), 'paranoid' (ask all), 'yolo' (auto-approve)
        # Per-thread consent for delegated background ("big brain") agents:
        # set by the background thread itself at startup. Threads with a
        # value other than None never prompt (prompting from a background
        # thread would deadlock the CLI); they obey the recorded consent.
        self._bg_consent = threading.local()

    def set_bg_consent(self, allowed: bool) -> None:
        """Mark the calling thread as a background agent with recorded consent."""
        self._bg_consent.value = bool(allowed)

    def _bg_consent_state(self):
        return getattr(self._bg_consent, "value", None)

    def check_command(self, command: str) -> bool:
        """Returns True if command execution is permitted.
        Uses an allowlist for safe commands; everything else prompts.
        """
        bg = self._bg_consent_state()
        if bg is not None:
            # Background agent: never prompt. Destructive commands are always
            # denied (report them as a pending manual step instead).
            if self.DESTRUCTIVE_PATTERN.search(command):
                return False
            return bg

        if self.mode == "yolo":
            return True

        # Check if the command starts with a safe utility.
        # Simple check: is the first word in the safe list?
        cmd_base = command.strip().split()[0].lower() if command.strip() else ""
        if cmd_base in self.SAFE_COMMANDS and not self.DESTRUCTIVE_PATTERN.search(command):
            return True

        if self.mode == "paranoid" or self.DESTRUCTIVE_PATTERN.search(command):
            console.print()
            console.print(Panel(
                f"[bold red]⚠️  DESTRUCTIVE / HIGH-RISK COMMAND DETECTED[/bold red]\n\n"
                f"[yellow]{command}[/yellow]\n\n"
                f"[dim]Execution requires human confirmation.[/dim]",
                title="[bold red]Security Guard[/bold red]",
                border_style="red"
            ))
            try:
                ans = console.input("[bold red]Authorize execution? [y/N]: [/bold red]").strip().lower()
                return ans in ["y", "yes"]
            except (KeyboardInterrupt, EOFError):
                return False

        # For standard mode, prompt if it's not in the safe list
        console.print()
        console.print(Panel(
            f"[bold yellow]⚠️  UNRECOGNIZED COMMAND DETECTED[/bold yellow]\n\n"
            f"[yellow]{command}[/yellow]\n\n"
            f"[dim]This command is not on the safe allowlist. Confirm to proceed.[/dim]",
            title="[bold yellow]Security Guard[/bold yellow]",
            border_style="yellow"
        ))
        try:
            ans = console.input("[bold yellow]Authorize execution? [y/N]: [/bold yellow]").strip().lower()
            return ans in ["y", "yes"]
        except (KeyboardInterrupt, EOFError):
            return False

        return True

    def check_mutation(self, filepath: str, action: str = "write_file") -> bool:
        """Returns True if file modification is permitted."""
        bg = self._bg_consent_state()
        if bg is not None:
            # Background agent: never prompt. Only plain file edits inside a
            # consented task are allowed; sensitive system paths always denied.
            if not bg:
                return False
            abs_path = os.path.abspath(filepath)
            for p in self.SENSITIVE_PREFIXES:
                if abs_path == p or (os.path.isdir(p) and abs_path.startswith(p + "/")):
                    return False
            return True

        if self.mode == "yolo":
            return True
 
        abs_path = os.path.abspath(filepath)
        is_sensitive = False
        for p in self.SENSITIVE_PREFIXES:
            if abs_path == p or (os.path.isdir(p) and abs_path.startswith(p + "/")):
                is_sensitive = True
                break
 
        # Standard and paranoid modes now BOTH confirm every file mutation.
        # Previously only destructive/sensitive-path writes required
        # confirmation in standard mode, which let the model silently edit
        # arbitrary files with zero human sign-off.
        if is_sensitive:
            label = "SENSITIVE SYSTEM PATH"
        elif self.mode == "paranoid":
            label = "PARANOID MODE"
        else:
            label = "FILE MUTATION"
 
        console.print()
        console.print(Panel(
            f"[bold red]⚠️  {label}: CONFIRMATION REQUIRED[/bold red]\n\n"
            f"Action: [yellow]{action}[/yellow]\n"
            f"Target: [cyan]{abs_path}[/cyan]\n\n"
            f"[dim]Modification requires human confirmation.[/dim]",
            title="[bold red]Security Guard[/bold red]",
            border_style="red"
        ))
        try:
            ans = console.input("[bold red]Authorize file modification? [y/N]: [/bold red]").strip().lower()
            return ans in ["y", "yes"]
        except (KeyboardInterrupt, EOFError):
            return False

    def is_sensitive_path(self, filepath: str) -> bool:
        """True if filepath is (or lives inside) one of the sensitive read
        paths (.ssh, keys.txt, .env, shell history). assert_readable() denies
        these across all reading tools."""
        try:
            abs_path = os.path.abspath(os.path.expanduser(filepath))
        except Exception:
            return False
        for p in self.SENSITIVE_READ_PATHS:
            if abs_path == p or (os.path.isdir(p) and abs_path.startswith(p + os.sep)):
                return True
        return False

    def notify_mutation(self, tool_name: str, details: str):
        """Displays visual notice for mutating actions."""
        console.print(f"[bold yellow]📝 Mutation Action:[/bold yellow] [cyan]{tool_name}[/cyan] ({details})")

permission_broker = PermissionBroker()


# =====================================================================
# 6. Loop Detection & Execution Guardrails
# =====================================================================
class GuardrailManager:
    """Detects repeated failures, cycles, thrashing, and runaway tool use.

    The first repetition produces a corrective signal to the model; only a
    persistent loop aborts the turn. This is much less brittle than simply
    stopping after three identical calls.
    """
    def __init__(self):
        self.max_rounds = 64
        self.max_tool_calls = 128
        self.reset_turn()

    def reset_turn(self):
        self.rounds = 0
        self.tool_calls = 0
        self.events = deque(maxlen=24)
        self.signature_results = defaultdict(list)
        self.consecutive_failures = 0

    @staticmethod
    def signature(tool_name: str, args: dict) -> str:
        payload = json.dumps(args or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(f"{tool_name}:{payload}".encode()).hexdigest()

    def check_round(self) -> tuple[bool, str]:
        self.rounds += 1
        if self.rounds > self.max_rounds:
            return True, f"Execution guardrail: exceeded {self.max_rounds} tool rounds in one user turn."
        if self.tool_calls >= self.max_tool_calls:
            return True, f"Execution guardrail: exceeded {self.max_tool_calls} tool calls in one user turn."
        return False, ""

    def record(self, tool_name: str, args: dict, result: str, is_error: bool) -> tuple[bool, str]:
        self.tool_calls += 1
        sig = self.signature(tool_name, args)
        out_hash = hashlib.sha256(str(result).encode()).hexdigest()
        self.events.append((sig, out_hash, is_error))
        self.signature_results[sig].append((out_hash, is_error))
        self.consecutive_failures = self.consecutive_failures + 1 if is_error else 0

        # Exact same call + same result: first occurrence is normal, second is
        # suspicious, third is an actual loop.
        same = self.signature_results[sig]
        if len(same) >= 2 and same[-1][0] == same[-2][0]:
            repeats = 1
            for i in range(len(same) - 3, -1, -1):
                if same[i][0] == same[-1][0]:
                    repeats += 1
                else:
                    break
            if repeats >= 3:
                return True, (
                    f"Loop detected: '{tool_name}' was invoked with the same arguments and produced the same result {repeats} times. "
                    "Stop repeating it; diagnose the blocker or use a materially different strategy."
                )
            return False, (
                f"Repeated tool result detected for '{tool_name}'. Do not blindly repeat the same call. "
                "Use the result to change strategy, arguments, or target."
            )

        # Detect short A-B-A-B cycles, even when the arguments differ.
        if len(self.events) >= 6:
            recent = list(self.events)[-6:]
            sigs = [e[0] for e in recent]
            if sigs[0] == sigs[2] == sigs[4] and sigs[1] == sigs[3] == sigs[5] and sigs[0] != sigs[1]:
                return True, (
                    "Loop detected: the agent is cycling between the same two tool actions. "
                    "Break the cycle and choose a different diagnostic or implementation path."
                )

        if self.consecutive_failures >= 4:
            return True, (
                f"Execution guardrail: {self.consecutive_failures} consecutive tool failures. "
                "Stop issuing more mutations and explain or re-plan the blocker."
            )

        if is_error:
            return False, (
                f"Tool '{tool_name}' failed. Inspect the exact error above before trying again; "
                "change the command/arguments rather than repeating an identical call."
            )

        return False, ""

guardrail = GuardrailManager()


# =====================================================================
# 7. Explicit Sub-Task Planner / State Machine
# =====================================================================
class PlanManager:
    """Maintains a persistent high-level task plan that stays locked in context."""
    def __init__(self):
        self.plan_items = []  # list of {"title": str, "status": "pending"|"in_progress"|"completed"}

    def update_plan(self, tasks: list) -> str:
        """Accepts a list of task strings or dicts, parses checkboxes, and updates state."""
        parsed = []
        for item in tasks:
            if isinstance(item, dict):
                title = item.get("title") or item.get("task", "")
                status = item.get("status", "pending")
                parsed.append({"title": title, "status": status})
            elif isinstance(item, str):
                s = item.strip()
                status = "pending"
                if s.startswith("[x]") or s.startswith("[X]") or s.startswith("✓"):
                    status = "completed"
                    s = re.sub(r"^(\[x\]|\[X\]|✓)\s*", "", s)
                elif s.startswith("[/]") or s.startswith("[*]") or s.startswith("▶"):
                    status = "in_progress"
                    s = re.sub(r"^(\[/\]|\[\*\]|▶)\s*", "", s)
                elif s.startswith("[ ]"):
                    status = "pending"
                    s = re.sub(r"^\[ \]\s*", "", s)
                parsed.append({"title": s, "status": status})

        self.plan_items = parsed
        self.display_plan()
        return self.view_plan()

    def view_plan(self) -> str:
        if not self.plan_items:
            return "No active task plan defined. Call update_plan(tasks=[...]) to initialize one."
        lines = ["Current Task Plan:"]
        for idx, it in enumerate(self.plan_items, 1):
            box = "[x]" if it["status"] == "completed" else ("[/]" if it["status"] == "in_progress" else "[ ]")
            lines.append(f"{idx}. {box} {it['title']}")
        return "\n".join(lines)

    def display_plan(self):
        if not self.plan_items:
            return
        table = Table(title="🎯 Current Execution Plan", border_style="cyan")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Status", justify="center")
        table.add_column("Task Objective", style="bold")

        for idx, it in enumerate(self.plan_items, 1):
            st = it["status"]
            if st == "completed":
                status_str = "[bold green]✓ Done[/bold green]"
                title_str = f"[dim strike]{it['title']}[/dim strike]"
            elif st == "in_progress":
                status_str = "[bold yellow]▶ Active[/bold yellow]"
                title_str = f"[bold yellow]{it['title']}[/bold yellow]"
            else:
                status_str = "[dim]□ Pending[/dim]"
                title_str = it["title"]
            table.add_row(str(idx), status_str, title_str)
        console.print(table)

    def get_context_summary(self) -> str:
        if not self.plan_items:
            return ""
        return f"\n\nCURRENT HIGH-LEVEL TASK PLAN (Locked in context):\n{self.view_plan()}\n"

plan_mgr = PlanManager()

def update_plan(tasks: list) -> str:
    return plan_mgr.update_plan(tasks)

def view_plan() -> str:
    return plan_mgr.view_plan()


# =====================================================================
# Core Tools
# =====================================================================
def assert_readable(filepath: str) -> None:
    """Global check to block access to sensitive paths across all reading tools."""
    if permission_broker.is_sensitive_path(filepath):
        raise PermissionError(f"Permission denied: Access to sensitive path '{filepath}' is restricted.")

def read_file(filepath: str) -> str:
    try:
        filepath = resolve_path(filepath)
        assert_readable(filepath)
        if not os.path.isfile(filepath):
            return f"Error: File not found: {filepath}"

        # Automatic RAG interception for PDF files
        if filepath.lower().endswith(".pdf"):
            if not rag:
                return "Error: RAG engine is unavailable for PDF parsing."
            res = rag.index_pdf(filepath)
            fname = os.path.basename(filepath)
            overview_matches = rag.search("overview introduction abstract summary", top_k=2, doc_filter=fname)
            overview_text = "\n\n".join([f"[Page {r['page']}]: {r['text']}" for r in overview_matches]) if overview_matches else ""
            return (
                f"[Auto-Indexed PDF: '{fname}' ({res.get('pages', 0)} pages, {res.get('chunks', 0)} chunks)]\n"
                f"Top Introductory Excerpts:\n{overview_text}\n\n"
                f"To look up specific topics or sections, use the 'search_documents' tool with a query."
            )

        # Truncate giant text/code files to preserve context
        size = os.path.getsize(filepath)
        if size > 30000:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                snippet = f.read(6000)
            return f"(File is large: {size} bytes. Showing first 6000 chars):\n{snippet}\n\n...(use read_file_lines to inspect specific ranges)"

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def list_directory(path: str) -> str:
    try:
        path = resolve_path(path)
        return json.dumps(os.listdir(path))
    except Exception as e:
        return f"Error listing directory: {str(e)}"

class TaskManager:
    """Manages explicit asynchronous background tasks."""
    def __init__(self):
        self.tasks = {}
        self.counter = 1

    def run(self, command: str, wait_sec: int = 5) -> str:
        task_id = f"task-{self.counter}"
        self.counter += 1
        log_path = os.path.join(TASKS_DIR, f"{task_id}.log")
        try:
            log_file = open(log_path, "w")
            p = subprocess.Popen(
                command,
                shell=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=os.setsid
            )
            self.tasks[task_id] = {
                "p": p,
                "cmd": command,
                "log": log_path,
                "start": time.time(),
                "file": log_file
            }
        except Exception as e:
            return f"Error launching command: {str(e)}"

        try:
            p.wait(timeout=wait_sec)
            try:
                log_file.close()
            except Exception:
                pass
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                out = f.read().strip()
            return truncate_tool_output(out if out else "Command completed successfully (no output).", tool_name="task")
        except subprocess.TimeoutExpired:
            return (
                f"Command is long-running and was automatically sent to the background as [{task_id}] (PID: {p.pid}).\n"
                f"Log file: {log_path}\n"
                f"Use manage_task(action='status', task_id='{task_id}') to check progress, or manage_task(action='kill', task_id='{task_id}') to terminate."
            )

    def manage(self, action: str, task_id: str = "") -> str:
        action = action.lower().strip()
        if action == "list":
            if not self.tasks:
                return "No background tasks currently tracked."
            lines = ["Tracked background tasks:"]
            for tid, info in self.tasks.items():
                ret = info["p"].poll()
                elapsed = int(time.time() - info["start"])
                status = "RUNNING" if ret is None else f"EXITED({ret})"
                lines.append(f"- {tid}: {status} (PID {info['p'].pid}, elapsed {elapsed}s) -> {info['cmd'][:40]}")
            return "\n".join(lines)

        elif action == "status":
            if not task_id:
                return "Error: task_id is required for action='status'."
            if task_id not in self.tasks:
                return f"Error: Task '{task_id}' not found. Use manage_task(action='list') to view active tasks."
            info = self.tasks[task_id]
            ret = info["p"].poll()
            status = "RUNNING" if ret is None else f"EXITED(code {ret})"
            try:
                with open(info["log"], "r", encoding="utf-8", errors="ignore") as f:
                    log_lines = f.readlines()
                tail = "".join(log_lines[-25:]) if log_lines else "(no output logged yet)"
            except Exception as e:
                tail = f"Error reading log: {e}"
            return f"Task [{task_id}]: Status = {status} (PID {info['p'].pid})\nRecent Log Output:\n{tail}"

        elif action == "kill":
            if not task_id:
                return "Error: task_id is required for action='kill'."
            if task_id not in self.tasks:
                return f"Error: Task '{task_id}' not found."
            info = self.tasks[task_id]
            p = info["p"]
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    time.sleep(0.1)
                    if p.poll() is None:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    return f"Task [{task_id}] (PID {p.pid}) has been terminated."
                except Exception as e:
                    return f"Error killing task {task_id}: {e}"
            return f"Task [{task_id}] has already finished with exit code {p.poll()}."

        else:
            return f"Error: Unknown action '{action}'. Supported actions: 'list', 'status', 'kill'."

task_mgr = TaskManager()

def run_command(command: str, timeout: int = 30) -> str:
    """Execute a bash command in the persistent PTY with a bounded per-call timeout."""
    if re.search(r"\bgit(?:\s+-[^\s]+)*\s+clone\s+https?://github\.com/", command, re.IGNORECASE):
        return (
            "GitHub clone blocked. This tool only executes commands that are already authorized by the current task. "
            "For a read-only GitHub inspection/fetch, use fetch_github_repo or fetch_github_file. "
            "A local checkout requires an explicit user request to clone the repository."
        )
    if not permission_broker.check_command(command):
        return f"Permission denied: Execution of command '{command}' was aborted by user."
    try:
        timeout = max(1, min(int(timeout), 300))
    except (TypeError, ValueError):
        timeout = 30
    exit_code, output = persistent_shell.run(command, timeout=timeout)
    if exit_code != 0:
        res = f"[Exit Code {exit_code}]\n{output}" if output else f"[Exit Code {exit_code}]"
    else:
        res = output if output else "Command completed successfully (no output)."
    try:
        cwd = persistent_shell.get_cwd()
        res = f"[Working directory: {cwd}]\n{res}"
    except Exception:
        pass
    return truncate_tool_output(res, tool_name="bash")

def manage_task(action: str, task_id: str = "") -> str:
    return task_mgr.manage(action=action, task_id=task_id)

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        pass

def _parse_github_url(url: str):
    m = re.match(r"^https?://github\.com/([^/]+)/([^/#?]+)(?:/.*)?$", url.strip())
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    repo = repo.removesuffix('.git')
    return owner, repo


def _github_api_json(url: str):
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "Dummy-Local-Agent/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=12) as response:
        return json.loads(response.read().decode('utf-8', errors='replace'))


def fetch_github_repo(repo_url: str, max_tree_entries: int = 800) -> str:
    """Fetch a GitHub repository read-only through the GitHub API. Never clones."""
    parsed = _parse_github_url(repo_url)
    if not parsed:
        return "Error: expected a GitHub repository URL such as https://github.com/owner/repo"
    owner, repo = parsed
    api_root = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        meta = _github_api_json(api_root)
        default_branch = meta.get("default_branch", "main")
        ref = urllib.parse.quote(default_branch, safe="")
        tree_data = _github_api_json(f"{api_root}/git/trees/{ref}?recursive=1")
        entries = tree_data.get("tree", []) or []
        limit = max(50, min(int(max_tree_entries), 2000))
        tree_lines = [f"{e.get('type','')}: {e.get('path','')}" for e in entries[:limit]]
        if len(entries) > limit:
            tree_lines.append(f"... {len(entries) - limit} more entries omitted")
        readme = ""
        try:
            rd = _github_api_json(f"{api_root}/readme")
            if rd.get("content"):
                import base64
                readme = base64.b64decode(rd["content"]).decode("utf-8", errors="replace")
        except Exception:
            pass
        result = [
            f"Repository: {owner}/{repo}",
            f"URL: https://github.com/{owner}/{repo}",
            "Source: GitHub API (read-only)",
            f"Default branch: {default_branch}",
            f"Description: {meta.get('description') or '(none)'}",
            f"Primary language: {meta.get('language') or '(unknown)'}",
            f"Stars: {meta.get('stargazers_count', 0)} | Forks: {meta.get('forks_count', 0)} | Open issues: {meta.get('open_issues_count', 0)}",
            f"Updated: {meta.get('updated_at') or '(unknown)'}",
            "",
            "Repository tree:",
            *tree_lines,
        ]
        if readme:
            result += ["", "README:", readme[:30000]]
        return truncate_tool_output("\n".join(result), tool_name="fetch_github_repo")
    except Exception as exc:
        return (
            f"Error accessing GitHub repository {owner}/{repo}: {exc}. "
            "No local checkout was created. If this repository is private, GitHub API authentication is required."
        )


def fetch_github_file(repo_url: str, filepath: str, ref: str = "") -> str:
    """Fetch one text file read-only from GitHub. Never clones."""
    parsed = _parse_github_url(repo_url)
    if not parsed:
        return "Error: expected a GitHub repository URL such as https://github.com/owner/repo"
    owner, repo = parsed
    filepath = filepath.strip().lstrip("/")
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{urllib.parse.quote(filepath, safe='/')}"
    if ref:
        url += "?ref=" + urllib.parse.quote(ref, safe="")
    try:
        data = _github_api_json(url)
        if isinstance(data, list):
            return "Error: path is a directory. Use fetch_github_repo to inspect the tree."
        content = data.get("content")
        if not content:
            return f"Error: GitHub did not return text content for {filepath}."
        import base64
        text = base64.b64decode(content).decode("utf-8", errors="replace")
        return truncate_tool_output(
            f"Repository file: {owner}/{repo}/{filepath}\n"
            f"Ref: {ref or 'default branch'}\n"
            "Source: GitHub API (read-only)\n\n"
            f"{text}",
            tool_name="fetch_github_file",
        )
    except Exception as exc:
        return f"Error fetching GitHub file {filepath}: {exc}. No local checkout was created."


def search_web(query: str, max_results: int = 5) -> str:
    # Route explicit GitHub repository URLs to the dedicated repository retriever.
    github_match = re.search(r"https?://github\.com/[^\s]+/[^\s/#?]+", query or "")
    if github_match:
        return fetch_github_repo(github_match.group(0))
    try:
        if 'DDGS' not in globals():
            return "Error: duckduckgo-search library is not installed."
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}\n")
        res = "\n".join(results) if results else "No results found."
        return truncate_tool_output(res, tool_name="search_web")
    except Exception as e:
        return f"Error searching the web: {str(e)}"

def search_documents(query: str, document_filter: str = "") -> str:
    if not rag:
        return "Error: RAG engine not initialized."
    try:
        results = rag.search(query, top_k=4, doc_filter=document_filter if document_filter else None)
        if not results:
            return "No matching document excerpts found in knowledge base."
        formatted = []
        for idx, r in enumerate(results, 1):
            formatted.append(f"[{idx}] {r['filename']} (Page {r['page']}, score: {r['score']}):\n{r['text']}")
        res = "\n\n".join(formatted)
        return truncate_tool_output(res, tool_name="search_documents")
    except Exception as e:
        return f"Error searching documents: {str(e)}"

def index_document(filepath: str) -> str:
    if not rag:
        return "Error: RAG engine not initialized."
    try:
        filepath = resolve_path(filepath)
        res = rag.index_pdf(filepath)
        return json.dumps(res)
    except Exception as e:
        return f"Error indexing document: {str(e)}"

def list_documents() -> str:
    if not rag:
        return "Error: RAG engine not initialized."
    try:
        docs = rag.list_documents()
        if not docs:
            return "No documents indexed yet in knowledge base."
        return json.dumps(docs, indent=2)
    except Exception as e:
        return f"Error listing documents: {str(e)}"

def read_file_lines(filepath: str, start_line: int, end_line: int) -> str:
    try:
        filepath = resolve_path(filepath)
        assert_readable(filepath)
        if not os.path.isfile(filepath):
            return f"Error: File not found: {filepath}"
        start_line = max(1, int(start_line))
        end_line = int(end_line)
        if end_line < start_line:
            return "Error: end_line must be >= start_line"
        if (end_line - start_line) > 250:
            return "Error: Cannot read more than 250 lines at once to preserve tokens. Please specify a smaller range."
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        total = len(lines)
        selected = lines[start_line - 1:end_line]
        numbered = [f"{i + start_line}: {l.rstrip()}" for i, l in enumerate(selected)]
        res = f"(Lines {start_line}-{min(end_line, total)} of {total} in {os.path.basename(filepath)}):\n" + "\n".join(numbered)
        return truncate_tool_output(res, tool_name="read_lines")
    except Exception as e:
        return f"Error reading file lines: {str(e)}"

def code_outline(filepath: str) -> str:
    try:
        filepath = resolve_path(filepath)
        assert_readable(filepath)
        if not os.path.isfile(filepath):
            return f"Error: File not found: {filepath}"
        ext = os.path.splitext(filepath)[1].lower()
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        if ext in [".py"]:
            tree = ast.parse(content)
            outline = []
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    outline.append(f"class {node.name}:")
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            args = [a.arg for a in item.args.args]
                            outline.append(f"    def {item.name}({', '.join(args)}): ...")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [a.arg for a in node.args.args]
                    outline.append(f"def {node.name}({', '.join(args)}): ...")
            res = "\n".join(outline) if outline else "No classes or functions found."
            return truncate_tool_output(res, tool_name="outline")
        elif ext in [".c", ".cpp", ".cu", ".h", ".cuh"]:
            out = []
            for line in content.split("\n"):
                line_s = line.strip()
                if re.match(r"^(struct|class)\s+\w+", line_s):
                    out.append(line_s.split("{")[0].strip())
                elif re.match(r"^(__global__|__device__|__host__|void|int|float|double|auto|static|inline)\s+.*\(.*\)", line_s):
                    sig = line_s.split("{")[0].strip()
                    out.append(sig + ";")
            res = "\n".join(out[:100]) if out else "No C/CUDA signatures found."
            return truncate_tool_output(res, tool_name="outline")
        else:
            lines = content.splitlines()[:50]
            return "\n".join(lines)
    except Exception as e:
        return f"Error extracting code outline: {str(e)}"


# =====================================================================
# Hardened calc() — AST-walking evaluator, no eval()/compile() of
# arbitrary expressions. Only numeric literals, a fixed set of safe
# binary/unary operators, and a whitelist of math functions/constants
# are permitted. Anything else (attribute access, subscripting, name
# lookups outside the whitelist, comprehensions, calls to non-whitelisted
# functions, etc.) is rejected before any evaluation happens.
# =====================================================================
_CALC_ALLOWED_NAMES = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
_CALC_ALLOWED_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})

_CALC_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CALC_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

def _calc_eval_node(node):
    if isinstance(node, ast.Expression):
        return _calc_eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Disallowed constant type: {type(node.value).__name__}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _CALC_BIN_OPS:
            raise ValueError(f"Disallowed binary operator: {op_type.__name__}")
        left = _calc_eval_node(node.left)
        right = _calc_eval_node(node.right)
        return _CALC_BIN_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _CALC_UNARY_OPS:
            raise ValueError(f"Disallowed unary operator: {op_type.__name__}")
        return _CALC_UNARY_OPS[op_type](_calc_eval_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CALC_ALLOWED_NAMES:
            val = _CALC_ALLOWED_NAMES[node.id]
            if callable(val):
                raise ValueError(f"'{node.id}' must be called as a function, not used as a value.")
            return val
        raise ValueError(f"Name '{node.id}' is not permitted in calc.")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct calls to whitelisted functions are permitted.")
        fname = node.func.id
        if fname not in _CALC_ALLOWED_NAMES or not callable(_CALC_ALLOWED_NAMES[fname]):
            raise ValueError(f"Function '{fname}' is not permitted in calc.")
        if node.keywords:
            raise ValueError("Keyword arguments are not permitted in calc.")
        args = [_calc_eval_node(a) for a in node.args]
        return _CALC_ALLOWED_NAMES[fname](*args)
    # Explicitly reject everything else: Attribute, Subscript, Lambda,
    # comprehensions, function defs, imports, string/byte literals used
    # as code, etc.
    raise ValueError(f"Disallowed expression element: {type(node).__name__}")

def calc(expression: str) -> str:
    try:
        expr = expression.strip()
        tree = ast.parse(expr, mode="eval")
        result = _calc_eval_node(tree)
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error calculating expression: division by zero."
    except Exception as e:
        return f"Error calculating expression: {str(e)}"


def search_code(pattern: str, directory: str = ".", file_pattern: str = "") -> str:
    try:
        directory = resolve_path(directory)
        assert_readable(directory)
        if not os.path.isdir(directory):
            return f"Error: Directory not found: {directory}"
        excludes = " ".join([f"--exclude-dir={d}" for d in [".git", "__pycache__", ".venv", "venv", "build", "llama.cpp", "graphify-out", ".gemini", ".dummy"]])
        include_arg = f" --include='{file_pattern}'" if file_pattern else ""
        cmd = f"grep -rnI {excludes}{include_arg} -m 25 {json.dumps(pattern)} {json.dumps(directory)} | head -25"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=8)
        out = res.stdout.strip()
        return truncate_tool_output(out if out else "No matches found.", tool_name="search_code")
    except subprocess.TimeoutExpired:
        return "Error: Code search timed out after 8 seconds."
    except Exception as e:
        return f"Error searching code: {str(e)}"

def graphify_tool(action: str, query: str = "", node_b: str = "") -> str:
    bin_path = "/home/marwan/.venv/bin/graphify"
    if not os.path.isfile(bin_path):
        return "Error: graphify is not installed in /home/marwan/.venv/bin/graphify."
    action = action.lower().strip()
    if action == "explain":
        cmd = f"{bin_path} explain {json.dumps(query)}"
    elif action == "path":
        cmd = f"{bin_path} path {json.dumps(query)} {json.dumps(node_b)}"
    elif action == "query":
        cmd = f"{bin_path} query {json.dumps(query)} --budget 300"
    else:
        return f"Error: Unknown graphify action '{action}'. Use 'explain', 'path', or 'query'."
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        out = res.stdout.strip()
        if res.returncode != 0 and res.stderr:
            out += "\n" + res.stderr.strip()
        return truncate_tool_output(out if out else "No graph data returned.", tool_name="graphify")
    except subprocess.TimeoutExpired:
        return "Error: graphify command timed out after 10 seconds."
    except Exception as e:
        return f"Error querying graphify: {str(e)}"


# =====================================================================
# 9. Big Brain: NVIDIA NIM as a background tool for the local orchestrator
# =====================================================================
# The local 9B model is the always-on orchestrator: general chat and
# basic/intermediate coding. When a task is clearly bigger than the
# orchestrator (agentic / multi-file / large projects), it delegates to a
# much larger cloud model (default: z-ai/glm-5.3 on NVIDIA NIM) that runs
# as a BACKGROUND agent with the user's own tools - so the user keeps
# chatting with the fast local model while the big brain works.
_nim_state = {"model": NIM_DEFAULT_MODEL}
_big_brain_tasks = {}
_big_brain_lock = threading.Lock()
_file_ops_lock = threading.Lock()  # serializes file mutations: main loop vs background agents

BIG_BRAIN_CONSULT_SYSTEM = (
    "You are the big brain: a senior consulting engineer supporting a smaller, fast, "
    "local orchestrator model running on the user's laptop. The orchestrator cannot show "
    "you its conversation or its files, so it includes everything relevant in its question. "
    "Answer with exact code, exact commands, or a crisp step-by-step plan. No pleasantries, "
    "no filler - just the expert answer."
)

BIG_BRAIN_AGENT_ADDENDUM = """

=== BACKGROUND TASK MODE ===
You are currently running as the delegated "big brain" inside a background
worker thread. A smaller local orchestrator model handed you this task and
is keeping the user company while you work. Rules:
- The task description is everything you know; you cannot see the user's
  conversation. Read whatever files you need with your tools first.
- Work fully autonomously, end-to-end: investigate, modify, verify.
- Prefer apply_diff over write_file for existing files.
- Destructive commands (rm -rf, sudo, git reset --hard, ...) are forbidden
  in background mode. If one is genuinely required, skip it and list it as
  a pending manual step in your final report.
- When finished, output a final report containing: what you changed
  (exact file paths), what you verified (commands run + their results),
  and anything left for the user to do.
"""

# Tools the background agent must never touch: session tools are already
# filtered out of the `tools` list; plan/task/checkpoint state is owned by
# the main CLI loop; recursion into the big brain is disabled.
_BG_EXCLUDED_TOOLS = {
    "update_plan", "view_plan", "manage_task", "rollback",
    "delegate_task", "consult_big_brain", "big_brain_status", "big_brain_result",
}

BIG_BRAIN_MAX_ROUNDS = 40
BIG_BRAIN_DEADLINE_SECONDS = 15 * 60


def _nim_model_id() -> str:
    return _nim_state["model"]


def consult_big_brain(question: str, context: str = "") -> str:
    """Blocking expert advice from the cloud brain; the local model keeps control.

    Use when the orchestrator can execute the work itself but needs expert
    input: a hard algorithm, a tricky bug, a design decision, a second opinion.
    """
    if nim_client is None:
        return "Error: NVIDIA NIM unavailable: no API key found. Tell the user to add a 'NVIDIANIM:' line to ~/keys.txt."
    user_msg = f"{question}\n\n--- extra context ---\n{context}" if context else question
    try:
        resp = nim_client.chat.completions.create(
            model=_nim_model_id(),
            messages=[
                {"role": "system", "content": BIG_BRAIN_CONSULT_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=4000,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return "Error: the big brain returned an empty answer (it may have spent its token budget reasoning). Re-ask with a tighter question."
        return truncate_tool_output(text, tool_name="consult_big_brain")
    except Exception as exc:
        return f"Error consulting the big brain: {exc}"


def delegate_task(task: str, context: str = "") -> str:
    """Hand a large, self-contained task to the big brain. Returns IMMEDIATELY:
    the cloud model works in a background thread with full tool access, so the
    user stays with you and can keep chatting. Poll big_brain_status() and
    fetch the outcome with big_brain_result().
    """
    if nim_client is None:
        return "Error: NVIDIA NIM unavailable: no API key found. Tell the user to add a 'NVIDIANIM:' line to ~/keys.txt."

    # One human consent at the delegation boundary (this runs in the main
    # thread, so prompting is safe). Inside the task, the background agent
    # never prompts - it obeys this decision.
    mutation_consent = True
    if permission_broker.mode != "yolo":
        console.print()
        console.print(Panel(
            f"[bold yellow]🧠 BIG BRAIN DELEGATION[/bold yellow]\n\n"
            f"[cyan]{task[:600]}[/cyan]\n\n"
            f"Granting consent lets the cloud model edit files and run "
            f"non-destructive commands autonomously while you keep chatting.\n"
            f"[dim]Decline and it will still work, read-only: it will return a plan instead of changes.[/dim]",
            title="[bold cyan]Delegation Request[/bold cyan]",
            border_style="cyan",
        ))
        try:
            ans = console.input("[bold cyan]Authorize the big brain to make changes for this task? [y/N]: [/bold cyan]").strip().lower()
            mutation_consent = ans in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            mutation_consent = False

    task_id = f"brain-{time.strftime('%H%M%S')}-{len(_big_brain_tasks) + 1}"
    spec = {
        "id": task_id,
        "task": task,
        "context": context or "",
        "consent": mutation_consent,
        "status": "running",
        "started": time.strftime("%H:%M:%S"),
        "finished": None,
        "log": [],
        "files": [],
        "report": "",
        "error": None,
    }
    with _big_brain_lock:
        _big_brain_tasks[task_id] = spec
    threading.Thread(target=_big_brain_agent, args=(spec,), daemon=True).start()

    if voice_engine is not None:
        voice_engine.speak("Delegating to the big brain.")

    consent_note = "with write access" if mutation_consent else "in read-only mode (a plan will be returned instead of changes)"
    return (
        f"Big brain task '{task_id}' started in the background {consent_note}. "
        f"The user can keep chatting with you while it works - tell them that. "
        f"Check progress any time with big_brain_status(); when it finishes, "
        f"fetch the full outcome with big_brain_result() and walk the user through it. "
        f"Do NOT redo this task yourself."
    )


def big_brain_status() -> str:
    """Report the state of background big-brain tasks."""
    with _big_brain_lock:
        if not _big_brain_tasks:
            return "No big brain tasks have been started."
        lines = []
        for spec in _big_brain_tasks.values():
            last = spec["log"][-1][:100] if spec["log"] else "not started yet"
            lines.append(
                f"[{spec['id']}] {spec['status'].upper()} | started {spec['started']}"
                f" | steps: {len(spec['log'])} | last: {last}"
            )
        return "\n".join(lines)


def big_brain_result() -> str:
    """Fetch the report of the most recent big-brain task (waiting or done)."""
    with _big_brain_lock:
        if not _big_brain_tasks:
            return "No big brain tasks have been started."
        spec = list(_big_brain_tasks.values())[-1]
        running = spec["status"] == "running"
    if running:
        return (
            f"Task '{spec['id']}' is still running ({len(spec['log'])} steps so far, "
            f"last action: {spec['log'][-1][:100] if spec['log'] else 'n/a'}). "
            f"Check again in a bit with big_brain_status()."
        )
    head = (
        f"Task '{spec['id']}' - {spec['status'].upper()} ({spec['started']} -> {spec['finished']})\n"
        f"Files touched: {', '.join(spec['files']) or 'none'}\n"
        f"Steps taken: {len(spec['log'])}\n"
    )
    if spec["error"]:
        head += f"Error: {spec['error']}\n"
    body = spec["report"] or "(no report was produced)"
    return truncate_tool_output(head + "\n--- report ---\n" + body, tool_name="big_brain_result")


def _big_brain_agent(spec: dict) -> None:
    """Background runner: the cloud model executes the task with the user's
    tools. Runs entirely in its own thread with its own PTY shell, its own
    guardrails, and non-interactive permission checks."""
    permission_broker.set_bg_consent(spec["consent"])
    bg_shell = PersistentShell()
    try:
        bg_tool_defs = [t for t in tools if t.get("function", {}).get("name") not in _BG_EXCLUDED_TOOLS]
        messages = [
            {"role": "system", "content": load_system_prompt() + BIG_BRAIN_AGENT_ADDENDUM},
            {"role": "user", "content": spec["task"] + (f"\n\n--- extra context from the orchestrator ---\n{spec['context']}" if spec["context"] else "")},
        ]
        last_sig, same_rounds, rounds = None, 0, 0
        deadline = time.time() + BIG_BRAIN_DEADLINE_SECONDS

        while rounds < BIG_BRAIN_MAX_ROUNDS and time.time() < deadline:
            rounds += 1
            try:
                resp = nim_client.chat.completions.create(
                    model=_nim_model_id(),
                    messages=messages,
                    tools=bg_tool_defs,
                    tool_choice="auto",
                    max_tokens=16384,
                )
            except Exception as exc:
                spec["error"] = f"NIM request failed on round {rounds}: {exc}"
                break

            choice = resp.choices[0]
            msg = choice.message
            tool_calls = msg.tool_calls or []
            content = msg.content or ""

            if not tool_calls:
                spec["report"] = content
                break

            messages.append({
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else (tc.function.arguments or {})
                except Exception:
                    fn_args = {}

                sig = hashlib.sha1(json.dumps([fn_name, fn_args], sort_keys=True).encode()).hexdigest()[:12]
                same_rounds = same_rounds + 1 if sig == last_sig else 0
                last_sig = sig

                console.print(f"[dim]🧠 big-brain [{spec['id']}] ⚙️ {fn_name}({str(fn_args)[:110]})[/dim]")
                with _big_brain_lock:
                    spec["log"].append(f"{fn_name}({str(fn_args)[:110]})")

                if fn_name in {"write_file", "apply_diff"} and fn_args.get("filepath"):
                    with _big_brain_lock:
                        if fn_args["filepath"] not in spec["files"]:
                            spec["files"].append(fn_args["filepath"])

                try:
                    if fn_name == "run_command":
                        # Background agents use their own isolated PTY shell.
                        cmd = fn_args.get("command", "")
                        if not permission_broker.check_command(cmd):
                            out = (
                                "Permission denied: destructive commands are forbidden in "
                                "background mode. Skip it and note it as a pending manual step."
                            )
                        else:
                            _, shell_out = bg_shell.run(cmd, timeout=fn_args.get("timeout", 60))
                            out = shell_out
                        out = truncate_tool_output(out if out else "(no output)", tool_name="run_command")
                    elif fn_name in {"write_file", "apply_diff"}:
                        with _file_ops_lock:
                            out, _ = _call_tool(fn_name, fn_args)
                        out = truncate_tool_output(out, tool_name=fn_name)
                    else:
                        out, _ = _call_tool(fn_name, fn_args)
                        out = truncate_tool_output(out, tool_name=fn_name)
                except Exception as exc:
                    out = f"Error executing tool '{fn_name}': {exc}"

                if same_rounds >= 3:
                    out += "\n[EXECUTION GUARDRAIL] You have repeated this exact tool call multiple times in a row. Change your approach or finish the task."
                if same_rounds >= 6:
                    spec["error"] = "Stopped: repeated identical tool calls (agent loop)."
                    break

                messages.append({"role": "tool", "tool_call_id": tc.id, "name": fn_name, "content": out})
            if spec["error"]:
                break
        if not spec["error"] and not spec["report"]:
            if rounds >= BIG_BRAIN_MAX_ROUNDS:
                spec["error"] = f"Ran out of rounds ({BIG_BRAIN_MAX_ROUNDS})."
            elif time.time() >= deadline:
                spec["error"] = "Timed out (15 minutes)."
            else:
                spec["error"] = "Ended without a final report."
    except Exception as exc:
        spec["error"] = f"Background agent crashed: {type(exc).__name__}: {exc}"
    finally:
        permission_broker.set_bg_consent(True)  # clear the thread-local marker
        spec["status"] = "failed" if (spec["error"] and not spec["report"]) else "done"
        spec["finished"] = time.strftime("%H:%M:%S")
        verdict = "failed" if spec["status"] == "failed" else "complete"
        console.print(Panel(
            f"[bold]Big brain task {spec['id']} {verdict}.[/bold] "
            f"{len(spec['log'])} steps, files: {', '.join(spec['files']) or 'none'}."
            + (f"\n[dim]Error: {spec['error']}[/dim]" if spec["error"] else ""),
            title="[bold cyan]🧠 Big Brain[/bold cyan]",
            border_style="cyan" if spec["status"] == "done" else "red",
        ))
        if voice_engine is not None:
            voice_engine.speak("Big brain task complete." if spec["status"] == "done" else "Big brain task failed.")


# =====================================================================
# Tool Definitions for LLM
# =====================================================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "start_session",
            "description": "Start a new conversation session for persistent memory. Returns a session ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_name": {"type": "string", "description": "Optional name for the session (e.g. 'CUDA Debugging')"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_message",
            "description": "Persist a message to an existing session for later recall.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session ID from start_session"},
                    "role": {"type": "string", "description": "'user' or 'assistant'"},
                    "content": {"type": "string", "description": "Message content"}
                },
                "required": ["session_id", "role", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_session",
            "description": "Retrieve a complete conversation session by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session ID to retrieve"}
                },
                "required": ["session_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_sessions",
            "description": "List all saved conversation sessions.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_session",
            "description": "Delete a saved conversation session by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session ID to delete"}
                },
                "required": ["session_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with given content. Use ONLY when the user explicitly requested creating/replacing this file or when file creation is required to complete the current task. For edits, prefer apply_diff. Do NOT create summaries, notes, reports, or documentation files unless the user asked for a file. Keep content reasonably sized; for large edits prefer apply_diff or smaller coherent edits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Target file path"},
                    "content": {"type": "string", "description": "Full file content", "maxLength": 12000}
                },
                "required": ["filepath", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_diff",
            "description": "Aider-style search/replace block patching. Replaces an exact block of existing code with new code. Highly recommended over write_file for edits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Target file path to edit"},
                    "search_block": {"type": "string", "description": "Exact lines of code to find and replace"},
                    "replace_block": {"type": "string", "description": "New replacement code lines"}
                },
                "required": ["filepath", "search_block", "replace_block"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rollback",
            "description": "Revert recent Dummy mutations. Inside Git repositories this restores the exact pre-mutation file bytes from content-addressed Git objects without resetting the user's whole working tree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {"type": "integer", "description": "Number of checkpoints to undo (default 1)"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_plan",
            "description": "Create or update the high-level roadmap/todo list for complex tasks. Statuses: '[x]' completed, '[/]' in_progress, '[ ]' pending.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tasks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of task strings (e.g. ['[x] Inspect signatures', '[/] Refactor kernel', '[ ] Benchmark'])"
                    }
                },
                "required": ["tasks"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "view_plan",
            "description": "View the current high-level execution plan and task completion status.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Executes a shell/bash command in a persistent stateful PTY session. Directory (cwd) and environment variables persist across tool calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "minimum": 1, "maximum": 300, "description": "Maximum seconds to allow this command to run."}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Semantically search indexed local documents and PDFs for facts, numbers, sections, or code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query or keywords"},
                    "document_filter": {"type": "string", "description": "Optional partial filename to search within"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads local files. If given a PDF, automatically indexes it into RAG and returns an overview.",
            "parameters": {
                "type": "object",
                "properties": {"filepath": {"type": "string"}},
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "code_outline",
            "description": "Extracts class/function signatures, CUDA kernels, and structs from a source file (.py, .cu, .cpp, .c, .h) without function bodies, reducing tokens by 90%.",
            "parameters": {
                "type": "object",
                "properties": {"filepath": {"type": "string", "description": "Path to the code file"}},
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_lines",
            "description": "Reads specific line ranges from a file (e.g. lines 40-70) instead of loading the whole file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the file"},
                    "start_line": {"type": "integer", "description": "Starting line (1-indexed)"},
                    "end_line": {"type": "integer", "description": "Ending line (inclusive)"}
                },
                "required": ["filepath", "start_line", "end_line"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Evaluates mathematical and arithmetic expressions for memory bandwidth, hardware latency, and calculations.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Math expression (e.g. '12001 * 128 / 8 / 1000' or '2**16')"}},
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Fast grep search for patterns or symbols across code files, returning file:line without loading full files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "The string or regex to search for"},
                    "directory": {"type": "string", "description": "Directory to search within (default: .)"},
                    "file_pattern": {"type": "string", "description": "Optional glob like '*.py' or '*.cu'"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "graphify_tool",
            "description": "Queries the local Graphify knowledge graph. Actions: 'explain' (describe a concept and neighbors), 'path' (shortest connection between two nodes), or 'query' (traverse graph to answer a question).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["explain", "path", "query"], "description": "Action to run"},
                    "query": {"type": "string", "description": "Concept name for explain, Node A for path, or question for query"},
                    "node_b": {"type": "string", "description": "Node B (only required for 'path' action)"}
                },
                "required": ["action", "query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "index_document",
            "description": "Index a new PDF file into the local vector database so its contents can be searched.",
            "parameters": {
                "type": "object",
                "properties": {"filepath": {"type": "string", "description": "Path to the PDF file"}},
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "Lists all documents currently indexed in the local RAG knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "Lists the files and folders inside a given directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_github_repo",
            "description": "Fetch a GitHub repository read-only through the GitHub API. Use this FIRST whenever the user provides a github.com repository URL or asks to inspect or understand a GitHub repository. Returns repository metadata, recursive file tree, and README when available. NEVER clones or creates a local checkout. Do not use search_web for a specific repository URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string", "description": "GitHub repository URL accessible to the user"},
                    "max_tree_entries": {"type": "integer", "minimum": 50, "maximum": 2000}
                },
                "required": ["repo_url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_github_file",
            "description": "Fetch one text file read-only from a GitHub repository after fetch_github_repo. Use it for important source, headers, build files, tests, or documentation needed for an accurate analysis. NEVER clones the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_url": {"type": "string"},
                    "filepath": {"type": "string", "description": "Repository-relative file path"},
                    "ref": {"type": "string", "description": "Optional branch, tag, or commit"}
                },
                "required": ["repo_url", "filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Searches the internet for information on a given query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "manage_task",
            "description": "Manage asynchronous background tasks and long-running commands. Actions: 'list', 'status', or 'kill'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "status", "kill"], "description": "Action to perform"},
                    "task_id": {"type": "string", "description": "Task identifier (e.g. 'task-1')."}
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "consult_big_brain",
            "description": "Ask a much larger cloud model (the 'big brain') for expert advice, exact code, or a step-by-step plan while YOU keep control and execute locally. Use for hard algorithms, tricky bugs, design decisions, or a second opinion on basic/intermediate tasks. The big brain CANNOT see this conversation or any files, so include ALL relevant context (code, errors, requirements) in your question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "A complete, self-contained question for the expert. Include all code, errors, and requirements it needs."},
                    "context": {"type": "string", "description": "Optional additional context: relevant file contents, prior attempts, constraints."}
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delegate_task",
            "description": "Hand a LARGE, self-contained task to the big brain, which then works autonomously as a background agent with full tool access (it can read/write files, run commands, and verify its own changes). Returns immediately, so the user keeps chatting with you while it works - tell them that. Use for: agentic multi-file refactors, large project work, complex debugging campaigns spanning many steps. Do NOT delegate general chat or tasks you can handle yourself. The task string must be fully self-contained: the big brain cannot see this conversation, but it can read any file with its tools if you give it paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "A complete, self-contained task specification: goal, relevant file paths, constraints, and what 'done' looks like."},
                    "context": {"type": "string", "description": "Optional extra context: what you already tried, what the user said, relevant snippets."}
                },
                "required": ["task"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "big_brain_status",
            "description": "Check the progress of background big-brain tasks (running/finished, steps taken, last action).",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "big_brain_result",
            "description": "Fetch the full report of the most recent big-brain task: what it changed (file paths), what it verified, and remaining manual steps. Use this once the task is done, then walk the user through the outcome.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_related_entities",
            "description": "Find entities related to a given entity up to a certain depth.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Name of the entity to search for"}
                },
                "required": ["entity_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_conversations_about",
            "description": "Find conversation contexts where an entity was mentioned.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Name of the entity"}
                },
                "required": ["entity_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_contextual_summary",
            "description": "Get a comprehensive summary of an entity including properties and connections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Name of the entity"}
                },
                "required": ["entity_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_graph",
            "description": "Search graph nodes by name substring match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "Manually add a significant fact or entity to the graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Name of the entity"},
                    "fact": {"type": "string", "description": "The fact to remember about this entity"}
                },
                "required": ["entity_name", "fact"]
            }
        }
    }
]

# Session lifecycle is CLI-owned, not model-owned. Keeping these out of the
# model tool surface prevents the model from spawning parallel sessions or
# manually duplicating persistence that chat() already handles.
MODEL_HIDDEN_TOOLS = {"start_session", "add_message", "get_session", "list_sessions", "delete_session"}
tools = [
    t for t in tools
    if t.get("function", {}).get("name") not in MODEL_HIDDEN_TOOLS
]

available_functions = {
    "write_file": write_file,
    "apply_diff": apply_diff,
    "rollback": rollback,
    "update_plan": update_plan,
    "view_plan": view_plan,
    "run_command": run_command,
    "search_documents": search_documents,
    "read_file": read_file,
    "read_file_lines": read_file_lines,
    "code_outline": code_outline,
    "calc": calc,
    "search_code": search_code,
    "graphify_tool": graphify_tool,
    "index_document": index_document,
    "list_documents": list_documents,
    "list_directory": list_directory,
    "search_web": search_web,
    "fetch_github_repo": fetch_github_repo,
    "fetch_github_file": fetch_github_file,
    "manage_task": manage_task,
    "fetch_url": fetch_url,
    "start_session": start_session,
    "add_message": add_message,
    "get_session": get_session,
    "list_sessions": list_sessions,
    "delete_session": delete_session,
    "consult_big_brain": consult_big_brain,
    "delegate_task": delegate_task,
    "big_brain_status": big_brain_status,
    "big_brain_result": big_brain_result,
    "get_related_entities": get_related_entities,
    "find_conversations_about": find_conversations_about,
    "get_contextual_summary": get_contextual_summary,
    "search_graph": search_graph,
    "remember_fact": remember_fact,
}
for _hidden_tool in MODEL_HIDDEN_TOOLS:
    available_functions.pop(_hidden_tool, None)

# External system prompt
# Edit this file to change Dummy's behavior without modifying this Python script.
SCRIPT_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = Path(os.getenv("DUMMY_SYSTEM_PROMPT", str(SCRIPT_DIR / "system_prompt.txt"))).expanduser()

def load_system_prompt() -> str:
    try:
        prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise RuntimeError(
            f"System prompt file not found: {SYSTEM_PROMPT_PATH}\n"
            "Create it or set DUMMY_SYSTEM_PROMPT to a valid prompt file."
        )
    if not prompt:
        raise RuntimeError(f"System prompt file is empty: {SYSTEM_PROMPT_PATH}")
    return prompt

SYSTEM_PROMPT = load_system_prompt()


messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

def auto_index_prompt_pdfs(user_input: str):
    """Detect any PDF file paths typed in prompt and auto-index them immediately."""
    if not rag:
        return
    for token in user_input.split():
        clean = token.strip("'\"`,;:()[]")
        if clean.lower().endswith(".pdf"):
            expanded = os.path.abspath(os.path.expanduser(clean))
            if os.path.isfile(expanded):
                fname = os.path.basename(expanded)
                existing = [d["filename"] for d in rag.list_documents()]
                if fname not in existing:
                    console.print(f"[bold yellow]📄 Auto-indexing new PDF:[/bold yellow] {fname}...")
                    res = rag.index_pdf(expanded)
                    if res.get("status") == "success":
                        console.print(f"[bold green]✅ Indexed {fname}:[/bold green] {res['pages']} pages, {res['chunks']} chunks.")

def show_docs_table():
    """Display indexed documents in a clean CLI table."""
    if not rag:
        console.print("[red]RAG engine not available.[/red]")
        return
    docs = rag.list_documents()
    if not docs:
        console.print("[yellow]No documents indexed yet in knowledge base.[/yellow]")
        return
    table = Table(title="Local Knowledge Base (RAG)")
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Filename", style="magenta")
    table.add_column("Pages", justify="right", style="green")
    table.add_column("Chunks", justify="right", style="yellow")
    table.add_column("Indexed At", style="dim")
    for d in docs:
        table.add_row(str(d["id"]), d["filename"], str(d["pages"]), str(d["chunks"]), str(d["date"]))
    console.print(table)

def context_status() -> str:
    return context_manager.status(messages, tools)


def _repair_truncated_json_object(raw: str):
    """Best-effort repair for a model tool-call JSON object truncated mid-generation.

    This is intentionally conservative: it only balances JSON string/bracket state
    and removes an incomplete trailing comma/colon. If repair is not unambiguous,
    return None and let the model recover from a structured tool error.
    """
    candidate = (raw or "").strip()
    if not candidate or candidate[0] not in "[{":
        return None

    stack = []
    in_string = False
    escape = False
    pairs = {"{": "}", "[": "]"}

    for ch in candidate:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in pairs:
            stack.append(pairs[ch])
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None
            stack.pop()

    # A truncated JSON string is the most common llama.cpp failure mode.
    if in_string:
        candidate += '"'

    # Repair only a clearly incomplete tail.
    candidate = re.sub(r',\s*$', '', candidate)
    candidate = re.sub(r':\s*$', ': null', candidate)

    candidate += ''.join(reversed(stack))

    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _extract_tool_arguments(raw: str) -> tuple[dict, str | None]:
    """Parse model-generated tool JSON and attempt a conservative truncation repair."""
    cleaned = (raw or "").strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        value = json.loads(cleaned or "{}")
        if isinstance(value, dict):
            return value, None
        return {}, "Tool arguments must be a JSON object."
    except json.JSONDecodeError as first_error:
        repaired = _repair_truncated_json_object(cleaned)
        if isinstance(repaired, dict):
            return repaired, None
        return {}, f"Invalid JSON tool arguments: {first_error}"


def _canonicalize_tool_call(tc: dict) -> tuple[dict, dict, str | None]:
    """Make a tool call safe to place back into the next OpenAI-compatible request.

    llama.cpp may validate the assistant tool-call arguments when the next turn is
    rendered. Never propagate malformed JSON back into the request history.
    """
    fn = tc.setdefault("function", {})
    raw = fn.get("arguments", "") or "{}"
    args, parse_error = _extract_tool_arguments(raw)
    if parse_error:
        # Preserve only a bounded diagnostic, not the malformed payload itself.
        safe_args = {"_dummy_parse_error": parse_error[:1000]}
    else:
        safe_args = args
    fn["arguments"] = json.dumps(safe_args, ensure_ascii=False, separators=(",", ":"))
    return tc, safe_args, parse_error


def _call_tool(function_name: str, function_args: dict) -> tuple[str, bool]:
    if function_name in MODEL_HIDDEN_TOOLS:
        return f"Error: Tool '{function_name}' is CLI-owned and is not available to the model.", True
    function_to_call = available_functions.get(function_name)
    if not function_to_call:
        return f"Error: Unknown tool '{function_name}'.", True

    try:
        import inspect
        sig = inspect.signature(function_to_call)
        params = sig.parameters
        required = [
            name for name, p in params.items()
            if p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]
        missing = [name for name in required if name not in function_args]
        if missing:
            return f"Error: Missing required tool arguments: {', '.join(missing)}", True

        valid_args = {
            k: v for k, v in function_args.items()
            if k in params and params[k].kind != inspect.Parameter.VAR_POSITIONAL
        }
        result = function_to_call(**valid_args)
        text = str(result)
        return text, text.lstrip().startswith("Error") or text.startswith("Permission denied")
    except Exception as exc:
        return f"Error executing tool '{function_name}': {type(exc).__name__}: {exc}", True


def _render_stream(response):
    tool_calls = []
    full_content = ""
    reasoning_content = ""
    timings = None
    usage = None

    # Real-time voice: sentences of this response are spoken while it streams.
    if voice_engine is not None and voice_engine.tts_enabled:
        voice_engine.begin_utterance()

    def render_output(reasoning: str, content: str):
        clean_content = content
        clean_reasoning = reasoning
        if not clean_reasoning and "<think>" in content:
            before, after = content.split("<think>", 1)
            if "</think>" in after:
                r_part, rem = after.split("</think>", 1)
                clean_reasoning = r_part
                clean_content = before + rem
            else:
                clean_reasoning = after
                clean_content = before

        items = []
        if clean_reasoning.strip():
            if EXPAND_THINKING:
                items.append(Panel(
                    Markdown(clean_reasoning.strip()),
                    title="[bold magenta]💭 Thinking (Ctrl+O to collapse)[/bold magenta]",
                    border_style="dim magenta",
                    expand=False,
                ))
            else:
                words = len(clean_reasoning.strip().split())
                items.append(Panel(
                    f"[dim]thinking ... ({words} words • press Ctrl+O to expand)[/dim]",
                    title="[bold magenta]💭 Thinking[/bold magenta]",
                    border_style="dim magenta",
                    expand=False,
                ))
        elif not clean_content.strip():
            if EXPAND_THINKING:
                items.append(Panel(
                    "[dim]thinking ... (Ctrl+O to collapse)[/dim]",
                    title="[bold magenta]💭 Thinking[/bold magenta]",
                    border_style="dim magenta",
                    expand=False,
                ))
            else:
                items.append(Panel(
                    "[dim]thinking ... (press Ctrl+O to expand)[/dim]",
                    title="[bold magenta]💭 Thinking[/bold magenta]",
                    border_style="dim magenta",
                    expand=False,
                ))

        if clean_content.strip():
            items.append(Markdown(clean_content))
        return Group(*items) if len(items) > 1 else (items[0] if items else Markdown(""))

    fd = sys.stdin.fileno() if (sys.stdin.isatty() and termios is not None and hasattr(termios, "tcgetattr")) else None
    old_settings = termios.tcgetattr(fd) if fd is not None else None
    if fd is not None and tty is not None:
        try:
            tty.setcbreak(fd)
        except Exception:
            fd = None

    with Live(render_output(reasoning_content, full_content), console=console, refresh_per_second=15) as live:
        try:
            for chunk in response:
                # Voice barge-in: a wake-word command arrived mid-turn. Abort
                # the generation; the dictation becomes the next input.
                if voice_engine is not None and voice_engine.voice_interrupt.is_set():
                    console.print("\n[yellow]🎙 Voice command - stopping.[/yellow]")
                    voice_engine.flush()
                    # Drop partial tool calls: executing a half-streamed
                    # write_file/apply_diff could mutate files with truncated
                    # content.
                    return [], full_content, reasoning_content, timings, usage, True

                if fd is not None:
                    try:
                        rlist, _, _ = select.select([sys.stdin], [], [], 0)
                        if rlist:
                            ch = sys.stdin.read(1)
                            if ch in ('\x0f', '\x0F'):  # Ctrl+O
                                global EXPAND_THINKING
                                EXPAND_THINKING = not EXPAND_THINKING
                                live.update(render_output(reasoning_content, full_content))
                    except Exception:
                        pass

                if hasattr(chunk, "usage") and chunk.usage:
                    usage = chunk.usage

                dump = chunk.model_dump()
                if dump.get("timings"):
                    timings = dump["timings"]

                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                reasoning = getattr(delta, "reasoning_content", None)
                if not reasoning:
                    reasoning = dump.get("choices", [{}])[0].get("delta", {}).get("reasoning_content")
                if reasoning:
                    reasoning_content += reasoning
                    live.update(render_output(reasoning_content, full_content))

                if delta.content:
                    full_content += delta.content
                    if voice_engine is not None and voice_engine.tts_enabled:
                        voice_engine.feed(delta.content)
                    live.update(render_output(reasoning_content, full_content))

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        while len(tool_calls) <= tc.index:
                            tool_calls.append({
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            })
                        if tc.id:
                            tool_calls[tc.index]["id"] = tc.id
                        if tc.function.name:
                            tool_calls[tc.index]["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments
        except KeyboardInterrupt:
            console.print("\n[yellow]⏹ Generation interrupted by user.[/yellow]")
            if voice_engine is not None:
                voice_engine.flush()
            return tool_calls, full_content, reasoning_content, timings, usage, True
        except Exception as stream_err:
            console.print(f"[red]Streaming interrupted: {stream_err}[/red]")
            if voice_engine is not None:
                voice_engine.flush()
            return tool_calls, full_content, reasoning_content, timings, usage, True
        finally:
            if voice_engine is not None and voice_engine.tts_enabled:
                voice_engine.end_utterance()
            if fd is not None and old_settings is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except Exception:
                    pass

    return tool_calls, full_content, reasoning_content, timings, usage, False


def _synthesize_guardrail(messages_for_request, enable_thinking: bool, max_tokens: int) -> str:
    params = {
        "model": "llama.cpp",
        "messages": messages_for_request,
        "stream": True,
        "max_tokens": min(max_tokens, 700),
    }
    if not enable_thinking:
        params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    else:
        params["extra_body"] = {"chat_template_kwargs": {"reasoning_effort": "low"}}

    try:
        resp = client.chat.completions.create(**params)
        final = ""
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                final += chunk.choices[0].delta.content
                console.print(chunk.choices[0].delta.content, end="")
        console.print()
        return final.strip()
    except Exception as exc:
        console.print(f"[dim red]Could not synthesize guardrail summary: {exc}[/dim red]")
        return ""


# ---------------------------------------------------------------------------
# Wake-word dictation poller: the always-listening "Hey Jarvis" thread drops
# transcripts into voice_engine; this asyncio task (one per prompt session)
# picks them up, types them into the prompt, and optionally submits them.
# ---------------------------------------------------------------------------
_voice_poller_task = None

def _ensure_voice_poller():
    global _voice_poller_task
    if voice_engine is None:
        return
    try:
        import asyncio
        if _voice_poller_task is not None and not _voice_poller_task.done():
            return

        async def _poll_dictation():
            from prompt_toolkit.application import get_app
            while True:
                await asyncio.sleep(0.15)
                text = voice_engine.pop_pending_dictation()
                if not text:
                    continue
                try:
                    app = get_app()
                    buf = app.current_buffer
                    buf.insert_text(text)
                    if voice_engine.auto_submit:
                        await asyncio.sleep(0.1)  # let the insert render
                        buf.validate_and_handle()
                except Exception:
                    pass

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        _voice_poller_task = loop.create_task(_poll_dictation())
    except Exception as exc:
        console.print(f"[dim red]Wake-word poller failed to start: {exc}[/dim red]")


def chat():
    global SYSTEM_PROMPT
    enable_thinking = True
    reasoning_effort = "low"
    current_max_tokens = 8192
    fast_mode = False
    current_session_id = None
    current_session_name = None
    # NIM hybrid routing: nim_route sends every turn to the free NVIDIA
    # cloud API; force_nim_round is the one-shot fallback when the local
    # llama-server cannot be reached. The active NIM model id lives in
    # module-level _nim_state (shared with the big-brain tools).
    nim_route = False
    force_nim_round = False

    console.print(Panel.fit(
        "[bold yellow]🤖 DUMMY — Autonomous Agentic Workshop Assistant[/bold yellow]\n"
        "[green]Connected to http://localhost:8080[/green] | [cyan]PTY Shell: Active[/cyan] | [magenta]Git Checkpoints: Ready[/magenta]\n"
        "[dim]Shortcuts: /new <name> | /resume | /plan | /rollback | /checkpoints | /context | /safe [standard/paranoid/yolo] | /docs | /tasks | /think | /effort | /fast | /voice | /nim | Ctrl+T=push-to-talk | /tokens | /clear | /reload-prompt | exit[/dim]"
    ))

    context_manager.refresh(force=True)
    console.print(f"[dim]Context detected: {context_manager.max_context_tokens:,} tokens ({context_manager.source}).[/dim]")

    if voice_engine is not None:
        state = "ON (/voice off to mute)" if voice_engine.tts_enabled else "OFF (/voice on)"
        wake = " | 🎙 wake: say 'Jarvis'" if voice_engine.wake_enabled else ""
        console.print(f"[dim]🔊 Voice: {state}{wake} | Ctrl+T = push-to-talk[/dim]")
        if voice_engine.tts_enabled:
            hour = int(time.strftime("%H"))
            greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 18 else "Good evening")
            closers = [
                "All systems are online.",
                "It is good to have you back.",
                "At your service, as always.",
            ]
            closer = closers[int(time.time() / 60) % len(closers)]
            voice_engine.speak(f"{greeting}, sir. {closer}")
        voice_engine.prewarm()  # whisper model loads at boot, not on the first "Jarvis"

    while True:
        try:
            prefixes = []
            if current_session_name:
                prefixes.append(f"[bold blue][SESSION: {current_session_name}][/bold blue]")
            if fast_mode:
                prefixes.append("[bold green][FAST][/bold green]")
            elif enable_thinking:
                disp_effort = "HIGH" if reasoning_effort in ("high", "xhigh") else ("MED" if reasoning_effort == "medium" else "LOW")
                prefixes.append(f"[bold magenta][THINK:{disp_effort}][/bold magenta]")
            # Keep the input prompt clean: show only the current thinking level.
            # Do not expose session labels, Rich markup, or internal prefixes.
            think_label = f"[THINK:{reasoning_effort.upper()}] " if enable_thinking else "[THINK:OFF] "
            plain_prompt = f"{think_label}You > "

            user_input = prompt_session.prompt(
                plain_prompt,
                prompt_continuation=" " * len(plain_prompt),
                pre_run=_ensure_voice_poller,
            ).strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit"]:
            break

        # Barge-in: Jarvis stops talking the moment the user says something.
        if voice_engine is not None and voice_engine.tts_enabled:
            voice_engine.flush()

        cmd = user_input.lower()
        if cmd == "/plan":
            plan_mgr.display_plan()
            continue
        if cmd == "/plan clear":
            plan_mgr.update_plan([])
            console.print("[green]Task plan cleared.[/green]")
            continue
        if cmd.startswith("/rollback"):
            parts = user_input.split()
            steps = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            try:
                res = rollback(steps=steps)
            except Exception as exc:
                res = f"Rollback error: {exc}"
            console.print(Panel(res, title="[bold red]Rollback Triggered[/bold red]", border_style="red"))
            continue
        if cmd == "/checkpoints":
            console.print(Panel(checkpoint_mgr.list_checkpoints(), title="[bold cyan]Dummy Checkpoints[/bold cyan]", border_style="cyan"))
            continue
        if cmd == "/context":
            console.print(Panel(context_manager.status(messages, tools), title="[bold cyan]Context Manager[/bold cyan]", border_style="cyan"))
            continue
        if cmd.startswith("/safe"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1].lower() in ["standard", "paranoid", "yolo"]:
                permission_broker.mode = parts[1].lower()
            else:
                cycle = {"standard": "paranoid", "paranoid": "yolo", "yolo": "standard"}
                permission_broker.mode = cycle.get(permission_broker.mode, "standard")
            console.print(f"🛡️ Permission mode set to [bold cyan]{permission_broker.mode}[/bold cyan].")
            continue
        if cmd.startswith("/graph"):
            args = user_input[6:].strip()
            res = graph_viz.handle_command(args)
            console.print(Panel(res, title="[bold magenta]Knowledge Graph[/bold magenta]", border_style="magenta"))
            continue
        if cmd == "/docs":
            show_docs_table()
            continue
        if cmd == "/tasks":
            console.print(Panel(task_mgr.manage(action="list"), title="[bold yellow]Background Tasks[/bold yellow]", border_style="yellow"))
            continue

        if cmd.startswith("/new"):
            parts = user_input.split(maxsplit=1)
            name = parts[1].strip() if len(parts) > 1 else None
            if not name:
                console.print("[yellow]Usage: /new <session name>[/yellow]")
                continue
            session_id = start_session(name)
            current_session_id = session_id
            current_session_name = name
            console.print(f"[green]Started new session:[/green] {name}")
            continue

        if cmd == "/resume":
            sessions = brain.list_sessions()
            if not sessions:
                console.print("[yellow]No saved sessions found. Use /new <name> to create one.[/yellow]")
                continue
            ordered = sorted(sessions.items(), key=lambda kv: kv[1].get("last_accessed", ""), reverse=True)
            console.print("[bold cyan]=== Saved Sessions ===[/bold cyan]")

            session_choices = []
            for idx, (sid, info) in enumerate(ordered, 1):
                stored_name = info.get("name", "Untitled")
                name = SESSION_TITLES.get(str(sid))
                sess_data = get_session(sid)
                msg_count = len(sess_data.get("messages", [])) if sess_data else 0

                if not name and sess_data:
                    first_user = next(
                        (m.get("content", "") for m in sess_data.get("messages", [])
                         if m.get("role") == "user" and m.get("content")),
                        "",
                    )
                    if first_user:
                        name = _generate_session_title(sid, first_user)

                # Never expose the internal session ID or backend-generated name.
                # If this conversation has no saved title yet, ask the model to
                # generate one from its first user message.
                if (
                    not name
                    or name.strip().lower() in {"untitled", "untitled conversation"}
                    or _looks_like_session_id(name, str(sid))
                ):
                    first_user = next(
                        (m.get("content", "") for m in (sess_data or {}).get("messages", [])
                         if m.get("role") == "user" and m.get("content")),
                        "",
                    )
                    if first_user:
                        name = _generate_session_title(sid, first_user, force=True)

                if (
                    not name
                    or name.strip().lower() in {"untitled", "untitled conversation"}
                    or _looks_like_session_id(name, str(sid))
                ):
                    name = "Unnamed conversation"

                last = info.get("last_accessed", "Unknown")
                label = f"{name}  ({msg_count} msgs, last: {last})"
                session_choices.append((sid, label))

            try:
                selected = choice(
                    message="Resume which conversation? (↑/↓, Enter, Esc to cancel)",
                    options=session_choices,
                    default=session_choices[0][0] if session_choices else None,
                    mouse_support=True,
                )
            except (KeyboardInterrupt, EOFError):
                selected = None

            if selected is None:
                continue

            session_id = selected
            session = brain.get_session(session_id)
            if not session:
                console.print(f"[red]Session data missing on disk for: {session_id}[/red]")
                continue
            messages.clear()
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
            for msg in session.get("messages", []):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "assistant") and content is not None:
                    messages.append({"role": role, "content": content})
            context_manager.history_ledger.clear()
            current_session_id = session_id
            current_session_name = SESSION_TITLES.get(str(session_id)) or session.get("name", "Untitled")
            if not SESSION_TITLES.get(str(session_id)):
                first_user = next(
                    (m.get("content", "") for m in session.get("messages", [])
                     if m.get("role") == "user" and m.get("content")),
                    "",
                )
                if first_user:
                    current_session_name = _generate_session_title(session_id, first_user)
            console.print(f"[green]Resumed: {current_session_name} ({len(session.get('messages', []))} messages)[/green]")
            continue

        if cmd == "/sessions":
            sessions = brain.list_sessions()
            if not sessions:
                console.print("[yellow]No saved sessions found.[/yellow]")
                continue
            
            table = Table(title="Saved Conversations", border_style="cyan")
            table.add_column("Conversation Name", style="bold white")
            table.add_column("Messages", justify="right", style="magenta")
            table.add_column("Last Active", style="dim")
            
            ordered = sorted(sessions.items(), key=lambda kv: kv[1].get("last_accessed", ""), reverse=True)
            for sid, info in ordered:
                name = SESSION_TITLES.get(str(sid)) or info.get("name") or "Unnamed conversation"
                if _looks_like_session_id(name, str(sid)):
                    name = "Unnamed conversation"
                    
                sess_data = get_session(sid)
                msg_count = len(sess_data.get("messages", [])) if sess_data else 0
                table.add_row(name, str(msg_count), info.get("last_accessed", "Unknown"))
                
            console.print(table)
            continue

        if cmd == "/delete":
            sessions = brain.list_sessions()
            if not sessions:
                console.print("[yellow]No saved sessions found to delete.[/yellow]")
                continue
            
            ordered = sorted(sessions.items(), key=lambda kv: kv[1].get("last_accessed", ""), reverse=True)
            session_choices = []
            
            for sid, info in ordered:
                name = SESSION_TITLES.get(str(sid)) or info.get("name") or "Unnamed conversation"
                if _looks_like_session_id(name, str(sid)):
                    name = "Unnamed conversation"
                    
                sess_data = get_session(sid)
                msg_count = len(sess_data.get("messages", [])) if sess_data else 0
                label = f"{name} ({msg_count} msgs)"
                session_choices.append((sid, label))
                
            try:
                selected = choice(
                    message="Select a conversation to DELETE: (↑/↓, Enter, Esc to cancel)",
                    options=session_choices,
                    default=session_choices[0][0] if session_choices else None,
                    mouse_support=True,
                )
            except (KeyboardInterrupt, EOFError):
                selected = None

            if selected is None:
                continue

            # Execute the deletion using the hidden ID
            delete_session(selected)
            if str(selected) in SESSION_TITLES:
                del SESSION_TITLES[str(selected)]
                _save_session_titles(SESSION_TITLES)

            deleted_label = next(
                (label for sid, label in session_choices if sid == selected),
                str(selected),
            )
            console.print(f"[bold red]Deleted session:[/bold red] {deleted_label}")
            
            # If you delete the session you are currently inside, wipe the context clean
            if current_session_id == selected:
                messages.clear()
                messages.append({"role": "system", "content": SYSTEM_PROMPT})
                context_manager.history_ledger.clear()
                current_session_id = None
                current_session_name = None
                console.print("[yellow]Active session was deleted. Started a fresh context.[/yellow]")
            continue

        if cmd in ["/reload-prompt", "/reload_prompt"]:
            try:
                SYSTEM_PROMPT = load_system_prompt()
                messages[0]["content"] = SYSTEM_PROMPT
                console.print(f"[green]System prompt reloaded:[/green] {SYSTEM_PROMPT_PATH}")
            except Exception as exc:
                console.print(f"[red]Could not reload system prompt:[/red] {exc}")
            continue

        if cmd == "/clear":
            messages.clear()
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
            context_manager.history_ledger.clear()
            current_session_id = None
            current_session_name = None
            console.print("[green]Chat history cleared.[/green] [dim](detached from active session)[/dim]")
            continue

        if cmd.startswith("/think") or cmd.startswith("/thinking") or cmd in ["/expand", "/expand-thinking"]:
            parts = user_input.split()
            sub = parts[1].lower() if len(parts) > 1 else ""
            if sub in ["expand", "show", "open"]:
                EXPAND_THINKING = True
                console.print("🧠 Thinking display set to [bold green]EXPANDED[/bold green] (showing detailed steps).")
            elif sub in ["collapse", "hide", "close"]:
                EXPAND_THINKING = False
                console.print("🧠 Thinking display set to [bold yellow]COLLAPSED[/bold yellow] (hidden as 'thinking ...'). Press Ctrl+O to expand.")
            elif sub == "on":
                enable_thinking = True
                current_max_tokens = 2500
                console.print("🧠 Deep Thinking generation: [bold green]ON[/bold green].")
            elif sub == "off":
                enable_thinking = False
                current_max_tokens = 600
                console.print("🧠 Deep Thinking generation: [bold red]OFF[/bold red].")
            elif cmd in ["/expand", "/expand-thinking"]:
                EXPAND_THINKING = not EXPAND_THINKING
                status = "[bold green]EXPANDED[/bold green]" if EXPAND_THINKING else "[bold yellow]COLLAPSED[/bold yellow]"
                console.print(f"🧠 Thinking display: {status}. Press Ctrl+O to toggle anytime.")
            else:
                enable_thinking = not enable_thinking
                current_max_tokens = 2500 if enable_thinking else 600
                console.print(
                    f"🧠 Deep Thinking model generation: [{'bold magenta' if enable_thinking else 'bold green'}]{'ON' if enable_thinking else 'OFF'}[/] "
                    f"(output ceiling: {current_max_tokens}). Press Ctrl+O to toggle thinking steps display."
                )
            continue

        if cmd.startswith("/effort"):
            parts = user_input.split()
            valid = {"low": "low", "med": "medium", "medium": "medium", "high": "xhigh", "xhigh": "xhigh"}
            if len(parts) > 1:
                arg = parts[1].lower()
                if arg in valid:
                    reasoning_effort = valid[arg]
                else:
                    console.print("[yellow]Invalid effort level. Choose: low, med, high.[/yellow]")
                    continue
            else:
                reasoning_effort = {"low": "medium", "medium": "xhigh", "xhigh": "low"}.get(reasoning_effort, "low")
            console.print(f"🎯 Reasoning effort: [bold cyan]{reasoning_effort}[/bold cyan]")
            continue

        if cmd == "/fast":
            fast_mode = not fast_mode
            if fast_mode:
                enable_thinking = False
                current_max_tokens = 300
            else:
                current_max_tokens = 2500
            console.print(f"⚡ Fast mode: [bold]{'ON' if fast_mode else 'OFF'}[/bold].")
            continue

        if cmd.startswith("/tokens"):
            parts = user_input.split()
            if len(parts) > 1:
                try:
                    value = max(128, int(parts[1]))
                    current_max_tokens = value
                    console.print(f"[green]Requested generation ceiling: {current_max_tokens} tokens (dynamically clamped to context).[/green]")
                except ValueError:
                    console.print("[yellow]Usage: /tokens <number>[/yellow]")
            else:
                console.print(f"[yellow]Requested generation ceiling: {current_max_tokens} tokens.[/yellow]")
            continue

        if cmd.startswith("/voice"):
            if voice_engine is None:
                console.print("[red]Voice module failed to load at startup (see the warning above).[/red]")
                continue
            parts = user_input.split()
            sub = parts[1].lower() if len(parts) > 1 else ""
            if sub in ("on", "off"):
                voice_engine.toggle_tts(sub == "on")
                console.print(f"🔊 Spoken replies: [bold]{'ON' if voice_engine.tts_enabled else 'OFF'}[/bold]")
            elif sub == "test":
                voice_engine.toggle_tts(True)
                voice_engine.speak("Voice systems online. Running on the P cores, as requested.")
                console.print("[green]🔊 Speaking a test sentence now (turn your volume up).[/green]")
            elif sub == "wake":
                if len(parts) > 2 and parts[2].lower() in ("on", "off"):
                    voice_engine.set_wake(parts[2].lower() == "on")
                else:
                    voice_engine.set_wake(not voice_engine.wake_enabled)
                state = "ON - just say 'Jarvis'" if voice_engine.wake_enabled else "OFF"
                console.print(f"🎙 Wake word: [bold]{'ON' if voice_engine.wake_enabled else 'OFF'}[/bold] - {state}")
            elif sub == "autosubmit":
                if len(parts) > 2 and parts[2].lower() in ("on", "off"):
                    voice_engine.set_auto_submit(parts[2].lower() == "on")
                else:
                    voice_engine.set_auto_submit(not voice_engine.auto_submit)
                console.print(
                    f"🎙 Auto-submit wake dictation: [bold]{'ON' if voice_engine.auto_submit else 'OFF'}[/bold] "
                    f"[dim]({'submits automatically' if voice_engine.auto_submit else 'presses nothing - you press Enter'})[/dim]"
                )
            elif sub == "voices":
                from voice import list_voices
                installed = list_voices()
                table = Table(title="Installed Piper Voices", border_style="cyan")
                table.add_column("Voice", style="bold white")
                table.add_column("Status")
                for name in installed:
                    is_current = installed[name] == voice_engine.voice_path
                    table.add_row(name, "[green]● active[/green]" if is_current else "")
                console.print(table)
                console.print("[dim]Switch with /voice voice <name> (e.g. /voice voice en_GB-jarvis-medium)[/dim]")
            elif sub in ("voice", "model") and len(parts) > 2:
                name, err = voice_engine.set_voice(parts[2])
                if err:
                    console.print(f"[red]{err}[/red]")
                else:
                    voice_engine.toggle_tts(True)
                    voice_engine.speak(f"Voice switched to {name.replace('-', ' ').replace('en GB', 'English')}. At your service, sir.")
                    console.print(f"[green]🔊 Voice switched to:[/green] {name}")
            elif sub == "mic":
                _, rec_err = voice_engine.start_recording()
                if rec_err:
                    console.print(f"[red]🎙 {rec_err}[/red]")
                else:
                    console.print("[cyan]🎙 Recording 5 seconds - say something...[/cyan]")
                    time.sleep(5)
                    text, stt_err = voice_engine.stop_and_transcribe()
                    if text:
                        console.print(f"[green]🎙 Heard:[/green] {text}")
                    else:
                        console.print(f"[red]🎙 STT error: {stt_err}[/red]")
            else:
                console.print(Panel(voice_engine.status(), title="[bold cyan]Voice I/O[/bold cyan]", border_style="cyan"))
            continue

        if cmd.startswith("/nim"):
            if nim_client is None:
                console.print("[red]NVIDIA NIM unavailable: no API key found ($DUMMY_NIM_KEY, $NVIDIA_API_KEY, or a 'NVIDIANIM:' line in ~/keys.txt).[/red]")
                continue
            parts = user_input.split()
            sub = parts[1].lower() if len(parts) > 1 else ""
            if sub == "on":
                nim_route = True
                console.print(f"[green]☁️ Routing to NVIDIA NIM ({_nim_state['model']}) for subsequent turns.[/green]")
            elif sub == "off":
                nim_route = False
                console.print("[green]🖥️ Back to the local llama-server.[/green]")
            elif sub == "model" and len(parts) > 2:
                _nim_state["model"] = parts[2]
                console.print(f"[green]☁️ NIM model set to {_nim_state['model']}.[/green]")
            elif sub in ("", "status"):
                console.print(Panel(
                    f"API key: found\n"
                    f"Model: {_nim_state['model']}\n"
                    f"Routing: {'NIM (cloud)' if nim_route else 'local llama-server'}\n"
                    f"Endpoint: {NIM_BASE_URL}\n\n"
                    f"[dim]Direct access: /nim <question> fires a one-shot prompt straight at the "
                    f"big brain (answer is NOT added to the local conversation). /nim on routes the "
                    f"whole session to the cloud with all tools; the local model can also call the "
                    f"big brain itself: consult_big_brain (advice), delegate_task (background agent).[/dim]",
                    title="[bold cyan]NVIDIA NIM (free cloud brain)[/bold cyan]", border_style="cyan"))
            else:
                # One-shot direct prompt to the big brain: bypasses the local
                # model entirely, streams like a normal reply, and does NOT
                # enter the local conversation history.
                question = user_input.split(maxsplit=2)[2] if sub == "ask" else user_input.split(maxsplit=1)[1]
                if not question.strip():
                    console.print("[yellow]Usage: /nim <question> (or /nim ask <question>)[/yellow]")
                    continue
                console.print(
                    f"\n[bold cyan]☁️ Big Brain one-shot[/bold cyan] [dim]"
                    f"({_nim_state['model']} - direct, not added to the local conversation)[/dim]"
                )
                try:
                    if voice_engine is not None:
                        voice_engine.voice_interrupt.clear()  # fresh one-shot: re-arm
                    response = nim_client.chat.completions.create(
                        model=_nim_state["model"],
                        messages=[
                            {"role": "system", "content": (
                                "You are the big brain: a large expert cloud model. The user is "
                                "talking to you directly, bypassing their local assistant. Answer "
                                "directly, precisely, and completely - no filler, no pleasantries."
                            )},
                            {"role": "user", "content": question},
                        ],
                        stream=True,
                        max_tokens=8192,
                    )
                    # Same renderer as normal turns: thinking panel, live text,
                    # streamed voice - but the result never enters `messages`.
                    _, answer, _, _, _, interrupted = _render_stream(response)
                    if not interrupted and not answer.strip():
                        console.print("[red]The big brain returned an empty answer (reasoning may have used the token budget) - try rephrasing.[/red]")
                except Exception as exc:
                    console.print(f"[bold red]Big brain error:[/bold red] {exc}")
                continue
            continue

        if cmd.startswith("/index "):
            pdf_path = user_input[7:].strip()
            if rag:
                res = rag.index_pdf(pdf_path)
                console.print(f"Index result: {res}")
            continue

        auto_index_prompt_pdfs(user_input)

        if current_session_id is None:
            auto_name = f"Session {time.strftime('%Y-%m-%d %H:%M')}"
            current_session_id = start_session(auto_name)
            current_session_name = _generate_session_title(current_session_id, user_input)
            console.print(f"[dim]📝 Auto-started session: {current_session_name}[/dim]")

        turn_content = user_input
        if fast_mode:
            turn_content = f"{user_input}\n[Constraint: Answer in under 120 words with high-density bullets or code only. Zero filler.]"

        messages.append({"role": "user", "content": turn_content})
        try:
            add_message(current_session_id, "user", user_input)
            
            # Graph Memory: Extract entities and update graph in background
            def _extract():
                try:
                    extractor.process_message(user_input, role="user", session_id=current_session_id)
                except Exception:
                    pass
            threading.Thread(target=_extract, daemon=True).start()
            
        except Exception as exc:
            console.print(f"[dim red]⚠️ Could not save message to session: {exc}[/dim red]")

        guardrail.reset_turn()
        if voice_engine is not None:
            voice_engine.voice_interrupt.clear()  # fresh turn: re-arm voice barge-in
        backend_label = " [dim]☁️ NIM[/dim]" if (nim_route and nim_client is not None) else ""
        console.print(f"\n[bold yellow]Dummy >[/bold yellow]{backend_label}")

        turn_done = False
        turn_t0 = time.time()
        turn_acked = [False, False]  # long-work acknowledgments, one-shot each
        while not turn_done:
            # Keep the plan current without permanently duplicating it in history.
            active_system_prompt = SYSTEM_PROMPT + plan_mgr.get_context_summary()
            if voice_engine is not None and voice_engine.tts_enabled:
                # The model cannot see or hear - tell it it is being heard.
                # Voice mode: write for the ear. Time: Jarvis knows the hour.
                active_system_prompt += (
                    "\n\n[VOICE MODE ACTIVE - the user hears this reply spoken aloud "
                    "in real time while you write it. Keep spoken text flowing and "
                    "complete; prefer natural sentences over screen-only formatting.]\n"
                    f"[Current time: {time.strftime('%H:%M')}, {time.strftime('%A')}]"
                )
            messages[0]["content"] = active_system_prompt

            # Jarvis-style acknowledgment when the work runs long: one soft
            # line at ~12s (only if he hasn't spoken yet this turn), one at
            # ~45s. The silence in between is him working, not ignoring you.
            if voice_engine is not None and voice_engine.tts_enabled:
                elapsed = time.time() - turn_t0
                if elapsed > 12 and not turn_acked[0]:
                    if voice_engine.last_spoke_at < turn_t0:
                        voice_engine.speak("One moment, sir. This requires a little time.")
                    turn_acked[0] = True
                elif elapsed > 45 and not turn_acked[1]:
                    voice_engine.speak("Still with you, sir. The work is underway.")
                    turn_acked[1] = True

            stop, warning = guardrail.check_round()
            if stop:
                console.print(f"[bold red]🛑 {warning}[/bold red]")
                summary_messages, _, _ = context_manager.prepare(messages, 700, tools, verbose=False)
                final_msg = _synthesize_guardrail(summary_messages, enable_thinking, 700)
                if final_msg:
                    messages.append({"role": "assistant", "content": final_msg})
                    try:
                        add_message(current_session_id, "assistant", final_msg)
                    except Exception:
                        pass
                    if voice_engine is not None:
                        voice_engine.speak(final_msg)
                turn_done = True
                continue

            request_messages, effective_max_tokens, estimated_prompt = context_manager.prepare(
                messages, current_max_tokens, tools, verbose=True
            )

            # Backend selection: manual /nim routing, one-shot fallback after a
            # connection failure, or the local llama-server as before.
            use_nim = (nim_route or force_nim_round) and nim_client is not None
            force_nim_round = False
            if use_nim:
                # Cloud brain. Reasoning is on by default for GLM/DeepSeek on
                # NIM, and tool calls stream in the exact same shape the local
                # llama-server produces, so _render_stream works unchanged.
                nim_params = {
                    "model": _nim_state["model"],
                    "messages": request_messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "max_tokens": min(
                        max(effective_max_tokens, 16384) if tools else effective_max_tokens,
                        16384,
                    ),
                }
                try:
                    response = nim_client.chat.completions.create(**nim_params)
                except Exception as exc:
                    console.print(f"[bold red]Error calling NVIDIA NIM ({_nim_state['model']}):[/bold red] {exc}")
                    turn_done = True
                    continue
            else:
                create_params = {
                    "model": "llama.cpp",
                    "messages": request_messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "max_tokens": max(effective_max_tokens, 16384) if tools else effective_max_tokens,
                }
                if not enable_thinking:
                    create_params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
                else:
                    create_params["extra_body"] = {"chat_template_kwargs": {"reasoning_effort": reasoning_effort}}

                try:
                    response = client.chat.completions.create(**create_params)
                except Exception as exc:
                    if nim_client is not None:
                        console.print(f"[yellow]⚠️ Local llama-server unreachable ({type(exc).__name__}). Falling back to NVIDIA NIM ({_nim_state['model']}) for this turn.[/yellow]")
                        force_nim_round = True
                        continue
                    console.print(f"[bold red]Error connecting to llama-server:[/bold red] {exc}")
                    console.print("[yellow]Check http://localhost:8080/health and the llama-server process.[/yellow]")
                    turn_done = True
                    continue

            tool_calls, full_content, reasoning_content, timings, usage, interrupted = _render_stream(response)

            if timings and timings.get("predicted_n"):
                ctx_display = f"~{estimated_prompt:,}/{context_manager.max_context_tokens:,} estimated prompt"
                if usage and hasattr(usage, "total_tokens"):
                    ctx_display = f"server usage {usage.total_tokens:,}/{context_manager.max_context_tokens:,}"
                console.print(
                    f"[dim]Exact: {timings['predicted_n']} tokens at {timings.get('predicted_per_second', 0):.2f} t/s eval, "
                    f"{timings.get('prompt_per_second', 0):.2f} t/s prompt | {ctx_display}[/dim]"
                )

            if interrupted and not tool_calls:
                turn_done = True
                continue

            if not tool_calls:
                if full_content:
                    messages.append({"role": "assistant", "content": full_content})
                    try:
                        add_message(current_session_id, "assistant", full_content)
                    except Exception as exc:
                        console.print(f"[dim red]⚠️ Could not save assistant message: {exc}[/dim red]")
                    # NOTE: no speak() here - the streaming hook in
                    # _render_stream already spoke this reply sentence by
                    # sentence while it was being generated.
                turn_done = True
                continue

            # Normalize every tool-call argument before putting it into history.
            # This prevents malformed/truncated JSON from poisoning the next request.
            normalized_calls = []
            preparse = {}
            for tc in tool_calls:
                tc, parsed_args, parse_error = _canonicalize_tool_call(tc)
                normalized_calls.append(tc)
                preparse[id(tc)] = (parsed_args, parse_error)
            tool_calls = normalized_calls

            messages.append({
                "role": "assistant",
                "content": full_content if full_content else None,
                "tool_calls": tool_calls,
            })

            abort_loop = False
            recovery_warnings = []
            for tc in tool_calls:
                function_name = tc["function"]["name"]
                function_args, parse_error = preparse[id(tc)]
                if parse_error and "_dummy_parse_error" in function_args:
                    # Make this a normal tool error so the model can repair itself.
                    function_args = {}

                console.print(f"[bold yellow] ⚙️  Executing Tool:[/bold yellow] [cyan]{function_name}[/cyan] {function_args}")

                if parse_error:
                    function_response = f"Error: {parse_error}"
                    is_error = True
                else:
                    if function_name in {"write_file", "apply_diff", "rollback"}:
                        target_path = function_args.get("filepath", "")
                        permission_broker.notify_mutation(function_name, target_path if target_path else str(function_args))
                        with _file_ops_lock:
                            function_response, is_error = _call_tool(function_name, function_args)
                    else:
                        function_response, is_error = _call_tool(function_name, function_args)

                clean_response = truncate_tool_output(function_response, tool_name=function_name)
                should_halt, guard_message = guardrail.record(function_name, function_args, clean_response, is_error)
                if guard_message:
                    recovery_warnings.append(guard_message)
                    clean_response += "\n\n[EXECUTION GUARDRAIL] " + guard_message
                if should_halt:
                    abort_loop = True

                messages.append({
                    "tool_call_id": tc.get("id", ""),
                    "role": "tool",
                    "name": function_name,
                    "content": clean_response,
                })

            if abort_loop:
                guard_text = "\n".join(recovery_warnings) or "Tool-loop guardrail triggered."
                console.print(f"[bold red]🛑 {guard_text}[/bold red]")
                request_messages, _, _ = context_manager.prepare(messages, 700, tools, verbose=False)
                final_msg = _synthesize_guardrail(request_messages, enable_thinking, 700)
                if final_msg:
                    messages.append({"role": "assistant", "content": final_msg})
                    try:
                        add_message(current_session_id, "assistant", final_msg)
                    except Exception:
                        pass
                    if voice_engine is not None:
                        voice_engine.speak(final_msg)
                turn_done = True


if __name__ == "__main__":
    # Insurance: if anything ever re-executes this file via an interpreter
    # ('python -c ...' = multiprocessing spawn/forkserver helper), do NOT
    # start a second assistant session in that helper process.
    if sys.argv and os.path.basename(sys.argv[0]) == "-c":
        sys.exit(0)
    chat()
