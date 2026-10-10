import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from graph.graph_store import graph_store
from graph.entity_extractor import create_extractor
from graph.graph_tools import get_related_entities, find_conversations_about, get_graph_summary
from graph.visualizer import graph_viz

extractor = create_extractor()

# Simulate a few messages
print("Testing extraction...")
res1 = extractor.process_message("I am Marwan and I love building AI agents.", role="user", session_id="test1")
print(res1)
res2 = extractor.process_message("My project Dummy is getting a graph memory.", role="user", session_id="test1")
print(res2)
res3 = extractor.process_message("Dummy is located in my laptop.", role="user", session_id="test2")
print(res3)

print("\nTesting tools...")
print(get_graph_summary())
print("-" * 20)
print(get_related_entities("Marwan"))
print("-" * 20)
print(find_conversations_about("Dummy"))

print("\nTesting visualizer...")
print(graph_viz.handle_command("show-connections Marwan"))
