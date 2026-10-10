"""
High-level query functions for traversing the knowledge graph.

Used both as LLM tool handlers (exposed to the model via the tool spec)
and as helpers for the /graph CLI commands. Every function returns a
plain string — ready for tool output or console display.
"""
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

from graph.graph_store import graph_store


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_node_by_name(name: str) -> Optional[Dict]:
    """Find a node by name (case-insensitive). Returns the first match or None."""
    results = graph_store.find_nodes(name=name, limit=1)
    return results[0] if results else None


def _node_label(node: Dict) -> str:
    """Short human label for a node."""
    return f"[{node['type']}] {node['name']}"


# ---------------------------------------------------------------------------
# Public query functions — all return str for tool/CLI compatibility
# ---------------------------------------------------------------------------

def get_related_entities(entity_name: str, depth: int = 2,
                         max_results: int = 10) -> str:
    """Find entities related to `entity_name` up to `depth` hops away."""
    try:
        node = _find_node_by_name(entity_name)
        if not node:
            return f"Entity not found: {entity_name}"

        neighbors = graph_store.get_neighbors(node["id"], depth=depth,
                                              max_results=max_results)
        if not neighbors:
            return f"No related entities found for '{entity_name}'."

        lines = [f"Entities related to '{node['name']}' (up to {depth} hops):"]
        for n in neighbors:
            d = n.get("_depth", "?")
            lines.append(f"  • {_node_label(n)}  (depth {d})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error querying related entities: {e}"


def find_conversations_about(entity_name: str) -> str:
    """Find conversation contexts where an entity was mentioned."""
    try:
        node = _find_node_by_name(entity_name)
        if not node:
            return f"Entity not found: {entity_name}"

        edges = graph_store.get_edges(node["id"])
        contexts = []
        for edge in edges:
            ctx = edge.get("context", "").strip()
            if ctx:
                contexts.append({
                    "type": edge["type"],
                    "date": edge.get("created_at", "unknown"),
                    "text": ctx,
                })

        if not contexts:
            return f"No conversation contexts found for '{entity_name}'."

        # Sort by date descending
        contexts.sort(key=lambda c: c["date"], reverse=True)
        lines = [f"Conversations mentioning '{node['name']}':"]
        for c in contexts[:15]:
            date_str = c["date"][:10] if len(c["date"]) >= 10 else c["date"]
            lines.append(f"  [{date_str}] ({c['type']}) {c['text'][:120]}...")
        return "\n".join(lines)
    except Exception as e:
        return f"Error finding conversations: {e}"


def get_entity_timeline(entity_name: str) -> str:
    """Chronological history of an entity's appearances and connections."""
    try:
        node = _find_node_by_name(entity_name)
        if not node:
            return f"Entity not found: {entity_name}"

        lines = [f"Timeline for '{node['name']}' ({node['type']}):"]
        lines.append(f"  Created: {node.get('created_at', 'unknown')}")
        lines.append(f"  Updated: {node.get('updated_at', 'unknown')}")

        edges = graph_store.get_edges(node["id"])
        edges.sort(key=lambda e: e.get("created_at", ""))

        if edges:
            lines.append(f"\n  Connections ({len(edges)} total):")
            for edge in edges:
                other_id = edge["target"] if edge["source"] == node["id"] else edge["source"]
                other = graph_store.get_node(other_id)
                other_name = other["name"] if other else other_id
                date_str = edge.get("created_at", "?")[:10]
                lines.append(f"    [{date_str}] --{edge['type']}--> {other_name}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error building timeline: {e}"


def find_relationships(entity_a: str, entity_b: str) -> str:
    """Find direct and indirect paths between two entities (up to depth 3)."""
    try:
        node_a = _find_node_by_name(entity_a)
        node_b = _find_node_by_name(entity_b)
        if not node_a:
            return f"Entity not found: {entity_a}"
        if not node_b:
            return f"Entity not found: {entity_b}"

        # BFS to find paths from A to B
        target_id = node_b["id"]
        visited = {node_a["id"]}
        queue = deque([(node_a["id"], [node_a["name"]])])
        found_paths = []

        while queue:
            current_id, path = queue.popleft()
            if len(path) > 4:  # depth limit 3 edges = 4 nodes
                continue

            for edge in graph_store.get_edges(current_id):
                neighbor_id = edge["target"] if edge["source"] == current_id else edge["source"]
                neighbor = graph_store.get_node(neighbor_id)
                if not neighbor:
                    continue
                new_path = path + [f"--{edge['type']}-->", neighbor["name"]]

                if neighbor_id == target_id:
                    found_paths.append(new_path)
                elif neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, new_path))

        if not found_paths:
            return f"No relationship path found between '{entity_a}' and '{entity_b}'."

        lines = [f"Relationships between '{node_a['name']}' and '{node_b['name']}':"]
        for p in found_paths[:5]:
            lines.append(f"  {' '.join(p)}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error finding relationships: {e}"


def get_contextual_summary(entity_name: str) -> str:
    """Comprehensive summary of an entity: properties, connections, contexts."""
    try:
        node = _find_node_by_name(entity_name)
        if not node:
            return f"Entity not found: {entity_name}"

        lines = [
            f"=== {node['name']} ===",
            f"Type: {node['type']}",
            f"Description: {node.get('description', 'N/A')}",
            f"Created: {node.get('created_at', '?')}",
            f"Updated: {node.get('updated_at', '?')}",
        ]

        meta = node.get("metadata", {})
        if meta:
            lines.append(f"Metadata: {meta}")

        edges = graph_store.get_edges(node["id"])

        # Group connections by type
        by_type: Dict[str, list] = {}
        for edge in edges:
            other_id = edge["target"] if edge["source"] == node["id"] else edge["source"]
            other = graph_store.get_node(other_id)
            other_name = other["name"] if other else other_id
            direction = "→" if edge["source"] == node["id"] else "←"
            by_type.setdefault(edge["type"], []).append(
                f"{direction} {other_name} (weight: {edge['weight']:.1f})"
            )

        if by_type:
            lines.append(f"\nConnections ({len(edges)} total):")
            for etype, conns in by_type.items():
                lines.append(f"  {etype}:")
                for c in conns[:10]:
                    lines.append(f"    {c}")

        # Conversation contexts
        contexts = [e.get("context", "") for e in edges if e.get("context")]
        if contexts:
            unique_contexts = list(dict.fromkeys(contexts))[:5]
            lines.append(f"\nConversation contexts:")
            for ctx in unique_contexts:
                lines.append(f"  \"{ctx[:100]}...\"")

        return "\n".join(lines)
    except Exception as e:
        return f"Error building summary: {e}"


def search_graph(query: str, limit: int = 10) -> str:
    """Search nodes by name substring match."""
    try:
        results = graph_store.find_nodes(name=query, limit=limit)
        if not results:
            return f"No nodes matching '{query}'."

        lines = [f"Search results for '{query}':"]
        for node in results:
            lines.append(f"  • {_node_label(node)} — {node.get('description', '')[:60]}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error searching graph: {e}"


def get_graph_summary() -> str:
    """High-level graph statistics and most-connected nodes."""
    try:
        stats = graph_store.get_stats()
        lines = [
            "=== Knowledge Graph Summary ===",
            f"Total nodes: {stats['total_nodes']}",
            f"Total edges: {stats['total_edges']}",
        ]

        if stats["nodes_by_type"]:
            lines.append("\nNodes by type:")
            for t, count in sorted(stats["nodes_by_type"].items()):
                lines.append(f"  {t}: {count}")

        if stats["edges_by_type"]:
            lines.append("\nEdges by type:")
            for t, count in sorted(stats["edges_by_type"].items()):
                lines.append(f"  {t}: {count}")

        # Top 10 most-connected nodes
        edge_counts: Dict[str, int] = {}
        for edge in graph_store._graph["edges"].values():
            edge_counts[edge["source"]] = edge_counts.get(edge["source"], 0) + 1
            edge_counts[edge["target"]] = edge_counts.get(edge["target"], 0) + 1

        if edge_counts:
            top = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            lines.append("\nMost connected nodes:")
            for nid, count in top:
                node = graph_store.get_node(nid)
                name = node["name"] if node else nid
                lines.append(f"  {name}: {count} connections")

        return "\n".join(lines)
    except Exception as e:
        return f"Error generating summary: {e}"


def get_user_preferences() -> str:
    """List all preference nodes connected to the user (Marwan)."""
    try:
        user = _find_node_by_name("Marwan")
        if not user:
            return "User node 'Marwan' not found in graph."

        edges = graph_store.get_edges(user["id"], edge_type="prefers")
        if not edges:
            return "No preferences recorded yet."

        lines = ["Your recorded preferences:"]
        for edge in edges:
            pref = graph_store.get_node(edge["target"])
            if pref:
                ctx = edge.get("context", "")[:80]
                lines.append(f"  • {pref['name']} (weight: {edge['weight']:.1f}) — \"{ctx}\"")
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching preferences: {e}"


def get_user_connections() -> str:
    """List all person nodes connected to the user."""
    try:
        user = _find_node_by_name("Marwan")
        if not user:
            return "User node 'Marwan' not found in graph."

        edges = graph_store.get_edges(user["id"], edge_type="knows")
        if not edges:
            return "No connections recorded yet."

        lines = ["Your connections:"]
        for edge in edges:
            person = graph_store.get_node(edge["target"])
            if person:
                desc = person.get("description", "")
                lines.append(f"  • {person['name']} — {desc}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching connections: {e}"


def remember_fact(entity_name: str, fact: str,
                  entity_type: str = "concept") -> str:
    """Manually add a fact/entity to the graph with a 'mentions' edge from user."""
    try:
        user_id = graph_store.merge_node("Marwan", "person", "The user")
        node_id = graph_store.merge_node(entity_name, entity_type, description=fact)
        graph_store.add_edge(
            user_id, node_id, "mentions",
            weight=0.7,
            context=f"User explicitly remembered: {fact}",
        )
        return f"Remembered: '{entity_name}' ({entity_type}) — {fact}"
    except Exception as e:
        return f"Error remembering fact: {e}"
