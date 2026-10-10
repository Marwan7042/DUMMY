"""
Persistent JSON-based knowledge graph for Dummy's memory system.

Stores entities (people, concepts, preferences) and their relationships
as a directed graph. All data persists locally in ~/models/dummy/graph/data/.

Schema:
    Node: {id, name, type, description, created_at, updated_at, metadata}
    Edge: {id, source, target, type, weight, created_at, context}

No external dependencies — pure Python + JSON.
"""
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import uuid4
from collections import deque

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_NODE_TYPES = [
    "person", "concept", "preference", "project",
    "emotion", "location", "object",
]

VALID_EDGE_TYPES = [
    "knows", "loves", "works_on", "prefers",
    "mentions", "located_in", "related_to",
]

GRAPH_DATA_DIR = Path.home() / "models" / "dummy" / "graph" / "data"
GRAPH_FILE = GRAPH_DATA_DIR / "graph.json"


class GraphStore:
    """Persistent JSON-based knowledge graph for Dummy's memory system.

    Stores entities (people, concepts, preferences) and their relationships
    as a directed graph. All data persists locally in ~/models/dummy/graph/data/.

    Thread-safe: all mutations acquire an internal lock before touching
    the in-memory graph or the backing JSON file.
    """

    def __init__(self):
        GRAPH_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._graph = self._load_graph()
        self._dirty = False
        self._rebuild_indexes()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_graph(self) -> Dict:
        """Load graph from disk, creating an empty one if absent or corrupt."""
        if GRAPH_FILE.exists():
            try:
                data = json.loads(GRAPH_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "nodes" in data and "edges" in data:
                    return data
            except Exception:
                pass
        return {"nodes": {}, "edges": {}, "indexes": {}}

    def _save_graph(self):
        """Atomic write: write to a .tmp file, then rename into place."""
        if not self._dirty:
            return
        tmp = GRAPH_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._graph, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.rename(GRAPH_FILE)
        self._dirty = False

    def _rebuild_indexes(self):
        """Build adjacency index: node_id → list of edge_ids (both directions)."""
        adj: Dict[str, list] = {}
        for eid, edge in self._graph["edges"].items():
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            adj.setdefault(src, []).append(eid)
            adj.setdefault(tgt, []).append(eid)
        self._graph["indexes"] = {"adjacency": adj}

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    def add_node(self, name: str, node_type: str,
                 description: str = "", metadata: Optional[Dict] = None) -> str:
        """Add a node. If one with the same name+type exists, update it instead.

        Returns the node id.
        """
        with self._lock:
            existing = self._find_existing_node(name, node_type)
            if existing:
                nid = existing["id"]
                self._graph["nodes"][nid]["description"] = description or existing.get("description", "")
                self._graph["nodes"][nid]["updated_at"] = datetime.now().isoformat()
                if metadata:
                    self._graph["nodes"][nid].setdefault("metadata", {}).update(metadata)
                self._dirty = True
                self._save_graph()
                return nid

            nid = uuid4().hex[:12]
            now = datetime.now().isoformat()
            self._graph["nodes"][nid] = {
                "id": nid,
                "name": name,
                "type": node_type if node_type in VALID_NODE_TYPES else "concept",
                "description": description,
                "created_at": now,
                "updated_at": now,
                "metadata": metadata or {},
            }
            self._dirty = True
            self._save_graph()
            return nid

    def merge_node(self, name: str, node_type: str,
                   description: str = "", metadata: Optional[Dict] = None) -> str:
        """Find existing node with same name (case-insensitive) and type.

        Updates if found, creates if not. Returns node id.
        Alias for add_node (which already implements merge semantics).
        """
        return self.add_node(name, node_type, description, metadata)

    def update_node(self, node_id: str, **kwargs) -> bool:
        """Update arbitrary properties on an existing node."""
        with self._lock:
            node = self._graph["nodes"].get(node_id)
            if not node:
                return False
            for key, value in kwargs.items():
                if key == "metadata" and isinstance(value, dict):
                    node.setdefault("metadata", {}).update(value)
                else:
                    node[key] = value
            node["updated_at"] = datetime.now().isoformat()
            self._dirty = True
            self._save_graph()
            return True

    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get a node by its id."""
        return self._graph["nodes"].get(node_id)

    def find_nodes(self, name: Optional[str] = None,
                   node_type: Optional[str] = None,
                   limit: int = 20) -> List[Dict]:
        """Search nodes by name (case-insensitive substring) and/or type."""
        results = []
        name_lower = name.lower() if name else None
        for node in self._graph["nodes"].values():
            if name_lower and name_lower not in node["name"].lower():
                continue
            if node_type and node["type"] != node_type:
                continue
            results.append(node)
            if len(results) >= limit:
                break
        return results

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its connected edges."""
        with self._lock:
            if node_id not in self._graph["nodes"]:
                return False
            del self._graph["nodes"][node_id]
            # Remove connected edges
            to_remove = [
                eid for eid, e in self._graph["edges"].items()
                if e["source"] == node_id or e["target"] == node_id
            ]
            for eid in to_remove:
                del self._graph["edges"][eid]
            self._dirty = True
            self._rebuild_indexes()
            self._save_graph()
            return True

    # ------------------------------------------------------------------
    # Edge CRUD
    # ------------------------------------------------------------------

    def add_edge(self, source_id: str, target_id: str, edge_type: str,
                 weight: float = 0.5, context: str = "") -> str:
        """Add a relationship edge between two nodes. Returns edge id."""
        with self._lock:
            eid = uuid4().hex[:12]
            self._graph["edges"][eid] = {
                "id": eid,
                "source": source_id,
                "target": target_id,
                "type": edge_type if edge_type in VALID_EDGE_TYPES else "related_to",
                "weight": max(0.0, min(1.0, weight)),
                "created_at": datetime.now().isoformat(),
                "context": context[:500] if context else "",
            }
            # Update adjacency index
            adj = self._graph["indexes"].get("adjacency", {})
            adj.setdefault(source_id, []).append(eid)
            adj.setdefault(target_id, []).append(eid)
            self._dirty = True
            self._save_graph()
            return eid

    def get_edges(self, node_id: str, direction: str = "both",
                  edge_type: Optional[str] = None) -> List[Dict]:
        """Get edges connected to a node.

        direction: 'out' (source=node), 'in' (target=node), 'both'
        """
        results = []
        adj = self._graph["indexes"].get("adjacency", {})
        for eid in adj.get(node_id, []):
            edge = self._graph["edges"].get(eid)
            if not edge:
                continue
            if direction == "out" and edge["source"] != node_id:
                continue
            if direction == "in" and edge["target"] != node_id:
                continue
            if edge_type and edge["type"] != edge_type:
                continue
            results.append(edge)
        return results

    def update_edge(self, edge_id: str, **kwargs) -> bool:
        """Update edge properties (weight, context, type)."""
        with self._lock:
            edge = self._graph["edges"].get(edge_id)
            if not edge:
                return False
            for key, value in kwargs.items():
                if key == "weight":
                    edge["weight"] = max(0.0, min(1.0, value))
                else:
                    edge[key] = value
            self._dirty = True
            self._save_graph()
            return True

    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge."""
        with self._lock:
            if edge_id not in self._graph["edges"]:
                return False
            del self._graph["edges"][edge_id]
            self._dirty = True
            self._rebuild_indexes()
            self._save_graph()
            return True

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def get_neighbors(self, node_id: str, depth: int = 1,
                      max_results: int = 20) -> List[Dict]:
        """BFS traversal to find connected nodes up to `depth` hops."""
        if node_id not in self._graph["nodes"]:
            return []

        visited = {node_id}
        queue = deque([(node_id, 0)])
        results = []

        while queue and len(results) < max_results:
            current, d = queue.popleft()
            if d > 0:
                node = self._graph["nodes"].get(current)
                if node:
                    results.append({**node, "_depth": d})

            if d < depth:
                for edge in self.get_edges(current):
                    neighbor = edge["target"] if edge["source"] == current else edge["source"]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, d + 1))

        return results[:max_results]

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return graph statistics: counts by node/edge type, totals."""
        nodes_by_type: Dict[str, int] = {}
        for node in self._graph["nodes"].values():
            t = node.get("type", "unknown")
            nodes_by_type[t] = nodes_by_type.get(t, 0) + 1

        edges_by_type: Dict[str, int] = {}
        for edge in self._graph["edges"].values():
            t = edge.get("type", "unknown")
            edges_by_type[t] = edges_by_type.get(t, 0) + 1

        return {
            "total_nodes": len(self._graph["nodes"]),
            "total_edges": len(self._graph["edges"]),
            "nodes_by_type": nodes_by_type,
            "edges_by_type": edges_by_type,
        }

    def export_json(self) -> Dict:
        """Return the full graph as a dict (for serialisation or export)."""
        return {
            "nodes": self._graph["nodes"],
            "edges": self._graph["edges"],
            "exported_at": datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_existing_node(self, name: str, node_type: str) -> Optional[Dict]:
        """Case-insensitive lookup by name + type."""
        name_lower = name.lower()
        for node in self._graph["nodes"].values():
            if node["name"].lower() == name_lower and node["type"] == node_type:
                return node
        return None


# ---------------------------------------------------------------------------
# Global graph store instance
# ---------------------------------------------------------------------------
graph_store = GraphStore()
