"""
Entity extraction from conversation messages using pattern matching.

Analyses text to identify named entities (people, places, projects,
preferences, emotions, concepts) and infers relationships between them.
Works without external APIs — pure Python regex + heuristics.

For higher accuracy the LLM can refine results at the orchestration layer,
but this module is self-contained and runs offline.
"""
import re
from datetime import datetime
from typing import Dict, List, Optional


# Common words to EXCLUDE from person-name detection (sentence starters,
# articles, conjunctions, common nouns that happen to be capitalised).
_COMMON_WORDS = frozenset({
    "I", "The", "This", "That", "These", "Those", "My", "Your", "His",
    "Her", "Its", "Our", "Their", "We", "They", "He", "She", "It",
    "What", "Which", "Who", "Where", "When", "Why", "How", "Can",
    "Could", "Would", "Should", "Will", "May", "Might", "Do", "Does",
    "Did", "Is", "Are", "Was", "Were", "Has", "Have", "Had", "But",
    "And", "Or", "So", "Yet", "For", "Nor", "Not", "No", "Yes",
    "Also", "Just", "Very", "Really", "Actually", "Basically",
    "However", "Although", "Because", "Since", "While", "If",
    "Then", "Now", "Here", "There", "Today", "Tomorrow", "Yesterday",
    "Maybe", "Perhaps", "Please", "Thanks", "Thank", "Sure", "OK",
    "Let", "Like", "Well", "Some", "Any", "All", "Each", "Every",
    "Much", "Many", "Most", "More", "Other", "New", "Old", "First",
    "Last", "Next", "Good", "Great", "Nice", "Bad", "Big", "Small",
    "Right", "Need", "Want", "Think", "Know", "See", "Look", "Find",
    "Use", "Try", "Make", "Take", "Get", "Give", "Go", "Come",
    "Tell", "Say", "Ask", "Hey", "Hello", "Hi", "Bye",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday", "January", "February", "March", "April",
    "June", "July", "August", "September", "October", "November",
    "December",
    # Tech words that appear capitalised but aren't names
    "Python", "Java", "Rust", "Linux", "Windows", "MacOS", "Docker",
    "Kubernetes", "React", "Node", "GPU", "CPU", "RAM", "API",
    "HTTP", "HTTPS", "SQL", "HTML", "CSS", "JSON", "YAML", "XML",
    "CUDA", "OpenCV", "TensorFlow", "PyTorch", "Vim", "Git",
})


class EntityExtractor:
    """Extracts entities and relationships from conversation text.

    Uses pattern matching and heuristic rules to identify:
    - Named entities (people, places, projects)
    - User preferences (likes, dislikes)
    - Emotional expressions
    - Conceptual references
    - Relationships between entities

    Works without external APIs — pure Python pattern matching.
    """

    def __init__(self, graph_store):
        self.graph_store = graph_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_entities(self, text: str, role: str = "user") -> Dict[str, List]:
        """Extract entities and relationships from a single message.

        Returns {"entities": [...], "relationships": [...]}.
        """
        entities: List[Dict] = []
        entities.extend(self._extract_persons(text))
        entities.extend(self._extract_preferences(text))
        entities.extend(self._extract_projects(text))
        entities.extend(self._extract_locations(text))
        entities.extend(self._extract_emotions(text))
        entities.extend(self._extract_concepts(text))

        # Deduplicate by (name_lower, type)
        seen = set()
        unique = []
        for ent in entities:
            key = (ent["name"].lower(), ent["type"])
            if key not in seen:
                seen.add(key)
                unique.append(ent)
        entities = unique

        relationships = self._infer_relationships(entities, text)
        return {"entities": entities, "relationships": relationships}

    def process_message(self, text: str, role: str = "user",
                        session_id: str = "") -> Dict:
        """Main entry point: extract, persist to graph, return summary.

        Ensures a 'Marwan' user node exists, then merges every extracted
        entity and relationship into the graph store.
        """
        # Ensure the user node exists
        user_id = self.graph_store.merge_node(
            "Marwan", "person", "The user of this system"
        )

        extraction = self.extract_entities(text, role)
        entities = extraction["entities"]
        relationships = extraction["relationships"]

        new_entities = 0
        new_relationships = 0
        persisted_entities = []

        for ent in entities:
            before = len(self.graph_store._graph["nodes"])
            nid = self.graph_store.merge_node(
                ent["name"], ent["type"],
                description=ent.get("description", ""),
                metadata={"confidence": ent.get("confidence", 0.5),
                          "session": session_id},
            )
            if len(self.graph_store._graph["nodes"]) > before:
                new_entities += 1
            persisted_entities.append({"id": nid, **ent})

        # Build a name→id map for relationship wiring
        name_to_id = {ent["name"].lower(): ent["id"] for ent in persisted_entities}
        name_to_id["marwan"] = user_id

        for rel in relationships:
            src_id = name_to_id.get(rel["source"].lower())
            tgt_id = name_to_id.get(rel["target"].lower())
            if src_id and tgt_id:
                self.graph_store.add_edge(
                    src_id, tgt_id, rel["type"],
                    weight=rel.get("weight", 0.5),
                    context=rel.get("context", text[:300]),
                )
                new_relationships += 1

        return {
            "new_entities": new_entities,
            "new_relationships": new_relationships,
            "entities": [e["name"] for e in entities],
            "relationships": [
                f"{r['source']} --{r['type']}--> {r['target']}"
                for r in relationships
            ],
        }

    # ------------------------------------------------------------------
    # Extraction heuristics
    # ------------------------------------------------------------------

    def _extract_persons(self, text: str) -> List[Dict]:
        """Detect person names via relationship phrases and proper-noun patterns."""
        results = []

        # Pattern 1: "my <relation> <Name>"
        relation_pat = re.compile(
            r"(?:my\s+(?:friend|brother|sister|mom|dad|mother|father|"
            r"wife|husband|colleague|boss|partner|uncle|aunt|cousin)\s+)"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            re.IGNORECASE,
        )
        for m in relation_pat.finditer(text):
            name = m.group(1).strip()
            if name and name not in _COMMON_WORDS:
                results.append({
                    "name": name, "type": "person",
                    "description": f"Mentioned as a personal contact",
                    "confidence": 0.8,
                })

        # Pattern 2: "<Name> is/was/works ..."
        subject_pat = re.compile(
            r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\s+"
            r"(?:is|was|works|lives|said|told|asked|likes|loves)",
        )
        for m in subject_pat.finditer(text):
            name = m.group(1).strip()
            if name not in _COMMON_WORDS and len(name) > 2:
                results.append({
                    "name": name, "type": "person",
                    "description": "Mentioned in conversation",
                    "confidence": 0.5,
                })

        return results

    def _extract_preferences(self, text: str) -> List[Dict]:
        """Detect user preferences: 'I like/love/prefer/enjoy/hate ...'"""
        results = []
        pref_pat = re.compile(
            r"(?:I|i)\s+(?:really\s+)?(?:like|love|prefer|enjoy|hate|dislike|want)\s+"
            r"(.+?)(?:\.|,|$|\band\b)",
            re.IGNORECASE,
        )
        for m in pref_pat.finditer(text):
            pref = m.group(1).strip()
            if pref and len(pref) > 1 and len(pref) < 120:
                # Determine if positive or negative
                verb = re.search(r"(like|love|prefer|enjoy|hate|dislike|want)", m.group(0), re.I)
                sentiment = "negative" if verb and verb.group(1).lower() in ("hate", "dislike") else "positive"
                results.append({
                    "name": pref, "type": "preference",
                    "description": f"{sentiment} preference",
                    "confidence": 0.7,
                })
        return results

    def _extract_projects(self, text: str) -> List[Dict]:
        """Detect project/app references: 'working on X', 'building X', etc."""
        results = []
        proj_pat = re.compile(
            r"(?:working\s+on|building|developing|my\s+project|created|making)\s+"
            r"([A-Za-z][A-Za-z0-9_. -]{1,40})",
            re.IGNORECASE,
        )
        for m in proj_pat.finditer(text):
            name = m.group(1).strip().rstrip(".,;:!?")
            if name and len(name) > 1:
                results.append({
                    "name": name, "type": "project",
                    "description": "Project or thing being built",
                    "confidence": 0.6,
                })
        return results

    def _extract_locations(self, text: str) -> List[Dict]:
        """Detect location references: 'live in X', 'from X', 'based in X'."""
        results = []
        loc_pat = re.compile(
            r"(?:live\s+in|from|based\s+in|located\s+in|moved\s+to|born\s+in)\s+"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        )
        for m in loc_pat.finditer(text):
            place = m.group(1).strip()
            if place and place not in _COMMON_WORDS and len(place) > 2:
                results.append({
                    "name": place, "type": "location",
                    "description": "Location mentioned in conversation",
                    "confidence": 0.6,
                })
        return results

    def _extract_emotions(self, text: str) -> List[Dict]:
        """Detect emotional states: 'I feel X', 'makes me X'."""
        results = []
        emo_pat = re.compile(
            r"(?:I\s+feel|I'm\s+feeling|makes?\s+me|I\s+am)\s+"
            r"(\w+)",
            re.IGNORECASE,
        )
        for m in emo_pat.finditer(text):
            emotion = m.group(1).strip().lower()
            if emotion and len(emotion) > 2 and emotion not in {"the", "like", "that", "this", "a"}:
                results.append({
                    "name": emotion, "type": "emotion",
                    "description": f"Emotional state expressed by user",
                    "confidence": 0.6,
                })
        return results

    def _extract_concepts(self, text: str) -> List[Dict]:
        """Detect technical/domain concepts: quoted terms, 'about X', etc."""
        results = []

        # Quoted terms
        quoted = re.findall(r'"([^"]{2,60})"', text)
        quoted += re.findall(r"'([^']{2,60})'", text)
        for term in quoted:
            term = term.strip()
            if term and len(term) > 2:
                results.append({
                    "name": term, "type": "concept",
                    "description": "Concept mentioned in quotes",
                    "confidence": 0.5,
                })

        # "about <concept>", "regarding <concept>"
        concept_pat = re.compile(
            r"(?:about|regarding|the\s+concept\s+of|interested\s+in|learning\s+about)\s+"
            r"([A-Za-z][A-Za-z0-9 _-]{2,40})",
            re.IGNORECASE,
        )
        for m in concept_pat.finditer(text):
            term = m.group(1).strip().rstrip(".,;:!?")
            if term and len(term) > 2:
                results.append({
                    "name": term, "type": "concept",
                    "description": "Topic of interest",
                    "confidence": 0.4,
                })

        return results

    # ------------------------------------------------------------------
    # Relationship inference
    # ------------------------------------------------------------------

    def _infer_relationships(self, entities: List[Dict],
                             text: str) -> List[Dict]:
        """Given extracted entities, infer relationships to the user and between entities."""
        relationships = []
        context_snippet = text[:300]

        for ent in entities:
            # Every entity gets a "mentions" edge from the user
            relationships.append({
                "source": "Marwan", "target": ent["name"],
                "type": "mentions", "weight": 0.5,
                "context": context_snippet,
            })

            # Specific relationship types based on entity type
            if ent["type"] == "preference":
                relationships.append({
                    "source": "Marwan", "target": ent["name"],
                    "type": "prefers", "weight": 0.7,
                    "context": context_snippet,
                })
            elif ent["type"] == "project":
                relationships.append({
                    "source": "Marwan", "target": ent["name"],
                    "type": "works_on", "weight": 0.7,
                    "context": context_snippet,
                })
            elif ent["type"] == "person":
                relationships.append({
                    "source": "Marwan", "target": ent["name"],
                    "type": "knows", "weight": 0.6,
                    "context": context_snippet,
                })
            elif ent["type"] == "location":
                relationships.append({
                    "source": "Marwan", "target": ent["name"],
                    "type": "located_in", "weight": 0.5,
                    "context": context_snippet,
                })
            elif ent["type"] == "emotion":
                relationships.append({
                    "source": "Marwan", "target": ent["name"],
                    "type": "related_to", "weight": 0.4,
                    "context": context_snippet,
                })

        # Cross-link entities mentioned in the same message
        for i, a in enumerate(entities):
            for b in entities[i + 1:]:
                if a["type"] != b["type"]:
                    relationships.append({
                        "source": a["name"], "target": b["name"],
                        "type": "related_to", "weight": 0.3,
                        "context": context_snippet,
                    })

        return relationships


# ---------------------------------------------------------------------------
# Module-level convenience — import and use without instantiation
# ---------------------------------------------------------------------------
def create_extractor():
    """Create an EntityExtractor wired to the global graph store."""
    from graph.graph_store import graph_store
    return EntityExtractor(graph_store)
