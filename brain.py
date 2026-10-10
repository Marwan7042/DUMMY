#!/home/marwan/.venv/bin/python3
"""
Brain module for conversation history and knowledge persistence.
Stores conversations, maintains context, and enables session resume.
No external dependencies - uses pure Python + JSON.

FIXED: metadata["sessions"] is now a dict keyed by session_id, e.g.
    metadata["sessions"][session_id] = {"name": ..., "started": ..., "last_accessed": ...}
Previously every start_session() call overwrote metadata["sessions"] with a
single flat record (no session_id key), which wiped out prior sessions'
metadata and made list_sessions()/add_message()/delete_session() iterate over
{"name": ..., "started": ..., "last_accessed": ...} as if each key were a
session id. That caused AttributeError: 'str' object has no attribute 'get'
in /resume as soon as more than one session existed.
"""
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configuration
BRAIN_DIR = Path.home() / "models" / "brain"
HISTORY_DIR = BRAIN_DIR / "history"
METADATA_FILE = BRAIN_DIR / "metadata.json"

# Ensure directories exist
BRAIN_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


class Brain:
    """Persistent conversation storage and retrieval system."""

    def __init__(self):
        self.history_dir = HISTORY_DIR
        self.metadata_file = METADATA_FILE
        self._init_metadata()

    def _init_metadata(self):
        """Initialize metadata file if it doesn't exist."""
        if not self.metadata_file.exists():
            self.metadata_file.write_text(json.dumps({
                "sessions": {},
                "total_conversations": 0,
                "last_updated": datetime.now().isoformat()
            }, indent=2))

    def _get_session_id(self, session_name: Optional[str] = None) -> str:
        """Generate or retrieve session ID."""
        if session_name:
            return hashlib.md5(session_name.encode()).hexdigest()[:12]
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _load_metadata(self) -> Dict:
        """Load metadata from file."""
        try:
            data = json.loads(self.metadata_file.read_text())
            # Defensive: guarantee shape even if file was hand-edited or corrupted
            if "sessions" not in data or not isinstance(data["sessions"], dict):
                data["sessions"] = {}
            return data
        except Exception:
            return {"sessions": {}, "total_conversations": 0, "last_updated": datetime.now().isoformat()}

    def _save_metadata(self, metadata: Dict):
        """Save metadata to file."""
        self.metadata_file.write_text(json.dumps(metadata, indent=2))

    def start_session(self, session_name: Optional[str] = None) -> str:
        """Start a new conversation session."""
        session_id = self._get_session_id(session_name)
        metadata = self._load_metadata()

        # Create session file
        session_path = self.history_dir / f"{session_id}.json"
        session_data = {
            "id": session_id,
            "name": session_name or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "started": datetime.now().isoformat(),
            "messages": [],
            "context": {}
        }
        session_path.write_text(json.dumps(session_data, indent=2))

        # FIX: key by session_id instead of clobbering the whole sessions dict
        metadata["sessions"][session_id] = {
            "name": session_name or "Untitled",
            "started": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat()
        }
        metadata["total_conversations"] = len(metadata["sessions"])
        metadata["last_updated"] = datetime.now().isoformat()
        self._save_metadata(metadata)

        return session_id

    def add_message(self, session_id: str, role: str, content: str,
                     context: Optional[Dict] = None) -> str:
        """Add a message to an existing session."""
        session_path = self.history_dir / f"{session_id}.json"

        if not session_path.exists():
            raise ValueError(f"Session not found: {session_id}")

        session_data = json.loads(session_path.read_text())

        message = {
            "timestamp": datetime.now().isoformat(),
            "role": role,  # "user" or "assistant"
            "content": content,
            "context": context or {}
        }

        session_data["messages"].append(message)
        session_data["last_accessed"] = datetime.now().isoformat()
        session_path.write_text(json.dumps(session_data, indent=2))

        # FIX: correctly look up / update this session's metadata entry
        metadata = self._load_metadata()
        if session_id in metadata["sessions"]:
            metadata["sessions"][session_id]["last_accessed"] = datetime.now().isoformat()
        else:
            # Session file existed but metadata entry was missing (e.g. legacy
            # data from before this fix) - backfill it instead of silently dropping it.
            metadata["sessions"][session_id] = {
                "name": session_data.get("name", "Untitled"),
                "started": session_data.get("started", datetime.now().isoformat()),
                "last_accessed": datetime.now().isoformat()
            }
        metadata["total_conversations"] = len(metadata["sessions"])
        metadata["last_updated"] = datetime.now().isoformat()
        self._save_metadata(metadata)

        return f"Added message to session {session_id}"

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Retrieve a complete session."""
        session_path = self.history_dir / f"{session_id}.json"
        if not session_path.exists():
            return None
        return json.loads(session_path.read_text())

    def get_session_messages(self, session_id: str,
                              start: int = 0,
                              end: Optional[int] = None) -> List[Dict]:
        """Get messages from a session with pagination."""
        session_data = self.get_session(session_id)
        if not session_data:
            return []
        messages = session_data["messages"]
        end_idx = end if end is not None else len(messages)
        return messages[start:end_idx]

    def search_conversations(self, query: str,
                              limit: int = 5,
                              session_filter: Optional[str] = None) -> List[Dict]:
        """Search conversations using simple keyword matching."""
        sessions = self.list_sessions()  # now a proper {session_id: info} dict
        results = []

        for session_id, session_info in sessions.items():
            if session_filter and session_filter != session_id:
                continue
            session_data = self.get_session(session_id)
            if not session_data:
                continue
            for msg in session_data.get("messages", []):
                content = msg.get("content", "").lower()
                if query.lower() in content:
                    results.append({
                        "session_id": session_id,
                        "session_name": session_info.get("name", "Unknown"),
                        "timestamp": msg.get("timestamp", ""),
                        "role": msg.get("role", ""),
                        "content": msg.get("content", ""),
                        "score": content.count(query.lower())
                    })

        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results[:limit]

    def get_session_metadata(self, session_id: str) -> Optional[Dict]:
        """Get metadata for a single session."""
        metadata = self._load_metadata()
        return metadata["sessions"].get(session_id)

    def list_sessions(self) -> Dict[str, Dict]:
        """List all available sessions, keyed by session_id."""
        metadata = self._load_metadata()
        return metadata["sessions"]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its history."""
        session_path = self.history_dir / f"{session_id}.json"
        if not session_path.exists():
            return False
        session_path.unlink()

        metadata = self._load_metadata()
        if session_id in metadata["sessions"]:
            del metadata["sessions"][session_id]
            metadata["total_conversations"] = len(metadata["sessions"])
            metadata["last_updated"] = datetime.now().isoformat()
            self._save_metadata(metadata)

        return True

    def export_session(self, session_id: str, output_path: str) -> bool:
        """Export a session to a file."""
        session_data = self.get_session(session_id)
        if not session_data:
            return False
        Path(output_path).write_text(json.dumps(session_data, indent=2))
        return True

    def import_session(self, input_path: str) -> Optional[str]:
        """Import a session from a file."""
        try:
            session_data = json.loads(Path(input_path).read_text())
            session_id = session_data.get("id", datetime.now().strftime("%Y%m%d_%H%M%S"))

            session_path = self.history_dir / f"{session_id}.json"
            session_path.write_text(json.dumps(session_data, indent=2))

            metadata = self._load_metadata()
            metadata["sessions"][session_id] = {
                "name": session_data.get("name", "Imported"),
                "started": session_data.get("started", datetime.now().isoformat()),
                "last_accessed": datetime.now().isoformat()
            }
            metadata["total_conversations"] = len(metadata["sessions"])
            metadata["last_updated"] = datetime.now().isoformat()
            self._save_metadata(metadata)

            return session_id
        except Exception as e:
            print(f"Import failed: {e}")
            return None


# Global brain instance
brain = Brain()


# ---------------------------------------------------------------------------
# Module-level wrappers. llama_cli.py does:
#   from assistant.brain import start_session, add_message, get_session,
#                                list_sessions, delete_session
# which needs plain functions at module scope, not just methods on `brain`.
# These just delegate to the global instance above.
# ---------------------------------------------------------------------------
def start_session(session_name: Optional[str] = None) -> str:
    return brain.start_session(session_name)

def add_message(session_id: str, role: str, content: str) -> str:
    return brain.add_message(session_id, role, content)

def get_session(session_id: str) -> Optional[Dict]:
    return brain.get_session(session_id)

def list_sessions() -> Dict[str, Dict]:
    return brain.list_sessions()

def delete_session(session_id: str) -> bool:
    return brain.delete_session(session_id)