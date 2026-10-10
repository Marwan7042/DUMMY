"""
CLI visualization and exploration interface for the knowledge graph.

Provides /graph commands for users to explore their conversation
knowledge graph from the terminal. Returns plain strings — the calling
code in dummy.py handles Rich formatting and Panel wrapping.
"""
import json
from pathlib import Path
from typing import Optional

from graph.graph_store import graph_store, GRAPH_DATA_DIR
from graph.graph_tools import (
    get_related_entities,
    get_contextual_summary,
    get_graph_summary,
    search_graph,
    get_user_preferences,
    get_user_connections,
    remember_fact,
)


class GraphVisualizer:
    """CLI visualization and exploration interface for the knowledge graph.

    Dispatches /graph subcommands and renders text output suitable for
    terminal display.
    """

    def handle_command(self, args: str) -> str:
        """Parse and dispatch a /graph subcommand.

        `args` is everything after '/graph ', e.g. 'show-all' or
        'connections Marwan'.
        """
        parts = args.strip().split(maxsplit=1)
        subcmd = parts[0].lower() if parts else "help"
        rest = parts[1].strip() if len(parts) > 1 else ""

        dispatch = {
            "help": lambda: self.show_help(),
            "show-all": lambda: self.show_all(),
            "show": lambda: self.show_all(),
            "connections": lambda: self.show_connections(rest),
            "show-connections": lambda: self.show_connections(rest),
            "search": lambda: self.search(rest),
            "summary": lambda: self.show_summary(rest),
            "preferences": lambda: self.show_preferences(),
            "people": lambda: self.show_people(),
            "stats": lambda: self.show_stats(),
            "remember": lambda: self._handle_remember(rest),
            "export": lambda: self._handle_export(rest),
        }

        handler = dispatch.get(subcmd)
        if handler:
            return handler()
        return f"Unknown /graph subcommand: '{subcmd}'. Type /graph help for usage."

    # ------------------------------------------------------------------
    # Subcommand implementations
    # ------------------------------------------------------------------

    def show_help(self) -> str:
        """List all /graph subcommands with descriptions."""
        return (
            "📊 Knowledge Graph Commands\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  /graph show-all              Show the full graph (nodes + edges)\n"
            "  /graph connections <name>    Show an entity's direct connections\n"
            "  /graph summary <name>        Detailed summary of an entity\n"
            "  /graph search <query>        Search nodes by name\n"
            "  /graph preferences           Show your recorded preferences\n"
            "  /graph people                Show all known people\n"
            "  /graph stats                 Graph statistics\n"
            "  /graph remember <name> <fact> Manually add a fact to the graph\n"
            "  /graph export json           Export full graph to JSON file\n"
            "  /graph export dot            Export graph in Graphviz DOT format\n"
            "  /graph help                  Show this help\n"
        )

    def show_all(self) -> str:
        """ASCII representation of the full graph, grouped by node type."""
        nodes = graph_store._graph["nodes"]
        edges = graph_store._graph["edges"]

        if not nodes:
            return "Graph is empty. Start chatting to build your knowledge graph!"

        # Group nodes by type
        by_type: dict = {}
        for node in nodes.values():
            by_type.setdefault(node["type"], []).append(node)

        lines = [f"🧠 Knowledge Graph ({len(nodes)} nodes, {len(edges)} edges)"]
        lines.append("━" * 50)

        if len(nodes) > 50:
            # Summary mode for large graphs
            lines.append("(Large graph — showing summary)")
            for ntype, type_nodes in sorted(by_type.items()):
                lines.append(f"\n  [{ntype}] ({len(type_nodes)} nodes)")
                for n in type_nodes[:5]:
                    lines.append(f"    • {n['name']}")
                if len(type_nodes) > 5:
                    lines.append(f"    ... and {len(type_nodes) - 5} more")
        else:
            for ntype, type_nodes in sorted(by_type.items()):
                lines.append(f"\n  [{ntype}]")
                for n in type_nodes:
                    edge_count = len(graph_store.get_edges(n["id"]))
                    lines.append(f"    • {n['name']}  ({edge_count} connections)")

        # Show edges
        if edges and len(edges) <= 30:
            lines.append(f"\n  Relationships:")
            for edge in edges.values():
                src = graph_store.get_node(edge["source"])
                tgt = graph_store.get_node(edge["target"])
                src_name = src["name"] if src else edge["source"]
                tgt_name = tgt["name"] if tgt else edge["target"]
                lines.append(f"    {src_name} --{edge['type']}--> {tgt_name}")
        elif edges:
            lines.append(f"\n  ({len(edges)} relationships — use /graph connections <name> to explore)")

        return "\n".join(lines)

    def show_connections(self, entity_name: str) -> str:
        """Show a mini ASCII tree of an entity and its direct connections."""
        if not entity_name:
            return "Usage: /graph connections <entity name>"

        results = graph_store.find_nodes(name=entity_name, limit=1)
        if not results:
            return f"Entity not found: '{entity_name}'"

        node = results[0]
        edges = graph_store.get_edges(node["id"])

        lines = [f"[{node['type']}] {node['name']}"]
        if not edges:
            lines.append("  (no connections)")
            return "\n".join(lines)

        for i, edge in enumerate(edges):
            other_id = edge["target"] if edge["source"] == node["id"] else edge["source"]
            other = graph_store.get_node(other_id)
            other_name = other["name"] if other else other_id
            other_type = other["type"] if other else "?"

            if edge["source"] == node["id"]:
                arrow = f"--{edge['type']}--> [{other_type}] {other_name}"
            else:
                arrow = f"<--{edge['type']}-- [{other_type}] {other_name}"

            connector = "└── " if i == len(edges) - 1 else "├── "
            lines.append(f"  {connector}{arrow}")

        return "\n".join(lines)

    def show_stats(self) -> str:
        """Formatted graph statistics."""
        return get_graph_summary()

    def show_preferences(self) -> str:
        """Show user preferences from the graph."""
        return get_user_preferences()

    def show_people(self) -> str:
        """Show all person-type nodes and their connections."""
        people = graph_store.find_nodes(node_type="person", limit=50)
        if not people:
            return "No people in the graph yet."

        lines = ["👥 People in your knowledge graph:"]
        for person in people:
            edges = graph_store.get_edges(person["id"])
            desc = person.get("description", "")
            lines.append(f"  • {person['name']} — {desc} ({len(edges)} connections)")
        return "\n".join(lines)

    def search(self, query: str) -> str:
        """Search graph nodes by name."""
        if not query:
            return "Usage: /graph search <query>"
        return search_graph(query)

    def show_summary(self, entity_name: str) -> str:
        """Detailed entity summary."""
        if not entity_name:
            return "Usage: /graph summary <entity name>"
        return get_contextual_summary(entity_name)

    def export_json(self) -> str:
        """Export the full graph to a JSON file."""
        data = graph_store.export_json()
        export_path = GRAPH_DATA_DIR / "export.json"
        export_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return f"Graph exported to {export_path}\n({len(data['nodes'])} nodes, {len(data['edges'])} edges)"

    def export_dot(self) -> str:
        """Export graph in Graphviz DOT format."""
        nodes = graph_store._graph["nodes"]
        edges = graph_store._graph["edges"]

        lines = ["digraph KnowledgeGraph {"]
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=box, style=filled, fillcolor=lightyellow];')

        for nid, node in nodes.items():
            label = f"{node['name']}\\n({node['type']})"
            lines.append(f'  "{nid}" [label="{label}"];')

        for edge in edges.values():
            lines.append(
                f'  "{edge["source"]}" -> "{edge["target"]}" '
                f'[label="{edge["type"]}"];'
            )

        lines.append("}")

        export_path = GRAPH_DATA_DIR / "export.dot"
        export_path.write_text("\n".join(lines), encoding="utf-8")
        return f"Graph exported in DOT format to {export_path}\nRender with: dot -Tpng {export_path} -o graph.png"

    # ------------------------------------------------------------------
    # Internal dispatch helpers
    # ------------------------------------------------------------------

    def _handle_remember(self, rest: str) -> str:
        """Parse 'remember <name> <fact>' arguments."""
        parts = rest.split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /graph remember <entity name> <fact description>"
        return remember_fact(parts[0], parts[1])

    def _handle_export(self, fmt: str) -> str:
        """Route to the right export method."""
        fmt = fmt.lower().strip()
        if fmt == "json":
            return self.export_json()
        elif fmt == "dot":
            return self.export_dot()
        return "Usage: /graph export [json|dot]"


# ---------------------------------------------------------------------------
# Global visualizer instance
# ---------------------------------------------------------------------------
graph_viz = GraphVisualizer()
