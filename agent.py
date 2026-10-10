#!/home/marwan/.venv/bin/python3
import os
import sys
import json
import subprocess
import requests
from .brain import brain
from typing import List, Dict, Any
from rag_engine import RAGEngine

API_URL = "http://127.0.0.1:8080/v1/chat/completions"

SYSTEM_PROMPT = """You are Antigravity Assistant, an intelligent local AI running on the user's Lenovo LOQ laptop (i7-14700HX, RTX 5050).
You have access to tools for searching the user's local document knowledge base (RAG), indexing new PDFs, listing documents, and executing safe bash commands.

Guidelines:
1. When asked about documents, PDFs, or specific technical notes, ALWAYS use the `search_documents` tool first to find relevant excerpts.
2. Cite the document name and page number when answering questions from the knowledge base.
3. If the user mentions a new PDF file to read or learn from, use `index_document` to ingest it into the vector database.
4. Keep your answers concise, accurate, and structured.
"""

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
          "name": "search_documents",
          "description": "Semantically search indexed local documents and PDFs for relevant text, facts, formulas, and sections.",
          "parameters": {
            "type": "object",
            "properties": {
              "query": {
                "type": "string",
                "description": "The search query, concept, or keywords to look for"
              },
              "document_filter": {
                "type": "string",
                "description": "Optional filename or partial filename to restrict search to"
              }
            },
            "required": ["query"]
          }
        }
    },
    {
        "type": "function",
        "function": {
          "name": "index_document",
          "description": "Index a new PDF file into the local vector database so it can be searched.",
          "parameters": {
            "type": "object",
            "properties": {
              "file_path": {
                "type": "string",
                "description": "Absolute or relative path to the PDF file to index"
              }
            },
            "required": ["file_path"]
          }
        }
    },
    {
        "type": "function",
        "function": {
          "name": "list_documents",
          "description": "List all documents currently indexed in the local knowledge base.",
          "parameters": {
            "type": "object",
            "properties": {}
          }
        }
    },
    {
        "type": "function",
        "function": {
          "name": "run_bash_command",
          "description": "Execute a non-interactive bash shell command on the host system.",
          "parameters": {
            "type": "object",
            "properties": {
              "command": {
                "type": "string",
                "description": "The bash command to execute"
              }
            },
            "required": ["command"]
          }
        }
    },
    {
        "type": "function",
        "function": {
          "name": "fetch_url",
          "description": "Fetch content from any URL and return the response body. Useful for reading web pages, GitHub raw files, API responses, etc.",
          "parameters": {
            "type": "object",
            "properties": {
              "url": {
                "type": "string",
                "description": "The URL to fetch (e.g., GitHub raw file URLs, web pages, API endpoints)"
              }
            },
            "required": ["url"]
          }
        }
    },
    {
        "type": "function",
        "function": {
          "name": "start_session",
          "description": "Start a new conversation session. Returns a session ID that can be used to track this conversation.",
          "parameters": {
            "type": "object",
            "properties": {
              "session_name": {
                "type": "string",
                "description": "Optional name for the session (e.g., 'Vecore Analysis', 'CUDA Debugging')"
              }
            },
            "required": []
          }
        }
    },
    {
        "type": "function",
        "function": {
          "name": "add_message",
          "description": "Add a message to the current session. Use this to persist conversation history.",
          "parameters": {
            "type": "object",
            "properties": {
              "session_id": {
                "type": "string",
                "description": "Session ID from start_session"
              },
              "role": {
                "type": "string",
                "description": "Message role: 'user' or 'assistant'"
              },
              "content": {
                "type": "string",
                "description": "Message content"
              }
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
              "session_id": {
                "type": "string",
                "description": "Session ID to retrieve"
              }
            },
            "required": ["session_id"]
          }
        }
    },
    {
        "type": "function",
        "function": {
          "name": "list_sessions",
          "description": "List all available conversation sessions.",
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
          "description": "Delete a conversation session and its history.",
          "parameters": {
            "type": "object",
            "properties": {
              "session_id": {
                "type": "string",
                "description": "Session ID to delete"
              }
            },
            "required": ["session_id"]
          }
        }
    }
]

class LocalAgent:
    def __init__(self, api_url: str = API_URL):
        self.api_url = api_url
        self.rag = RAGEngine()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Dispatch and execute tool calls."""
        if name == "search_documents":
            query = args.get("query", "")
            doc_filter = args.get("document_filter")
            results = self.rag.search(query, top_k=4, doc_filter=doc_filter)
            if not results:
                return "No matching documents found for this query."
            formatted = []
            for idx, r in enumerate(results, 1):
                formatted.append(f"[{idx}] Source: {r['filename']} (Page {r['page']}, Score: {r['score']}):\n{r['text']}")
            return "\n\n".join(formatted)

        elif name == "index_document":
            path = args.get("file_path", "")
            res = self.rag.index_pdf(path)
            return json.dumps(res)

        elif name == "list_documents":
            docs = self.rag.list_documents()
            if not docs:
                return "No documents indexed yet."
            return json.dumps(docs, indent=2)

        elif name == "start_session":
            session_name = args.get("session_name")
            session_id = brain.start_session(session_name)
            return f"Started new session: {session_id}"

        elif name == "add_message":
            session_id = args.get("session_id", "")
            role = args.get("role", "")
            content = args.get("content", "")
            try:
                brain.add_message(session_id, role, content)
                return f"Added message to session {session_id}"
            except Exception as e:
                return f"Error adding message: {e}"

        elif name == "get_session":
            session_id = args.get("session_id", "")
            session = brain.get_session(session_id)
            if not session:
                return f"Session not found: {session_id}"
            return json.dumps(session, indent=2)

        elif name == "delete_session":
            session_id = args.get("session_id", "")
            result = brain.delete_session(session_id)
            return f"Session {'deleted' if result else 'not found'}: {session_id}"

        elif name == "list_sessions":
            sessions = brain.list_sessions()
            return json.dumps(sessions, indent=2)

        elif name == "run_bash_command":
            cmd = args.get("command", "")
            try:
                proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                out = (proc.stdout + proc.stderr).strip()
                return out if out else "(Command completed with no output)"
            except Exception as e:
                return f"Execution error: {e}"

        elif name == "fetch_url":
            url = args.get("url", "")
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                content_type = resp.headers.get("Content-Type", "")
                # Try to detect encoding
                encoding = "utf-8"
                if "charset" in content_type:
                    encoding = content_type.split("charset=")[-1].split(";")[0].strip()
                return f"URL: {url}\nStatus: {resp.status_code}\nContent-Type: {content_type}\nEncoding: {encoding}\n\n--- CONTENT ---\n{resp.text}"
            except requests.exceptions.HTTPError as e:
                return f"HTTP Error: {e}"
            except requests.exceptions.RequestException as e:
                return f"Request failed: {e}"

        return f"Unknown tool: {name}"

    def chat(self, user_input: str) -> str:
        """Send message, handle multi-turn tool calling, return final answer."""
        self.messages.append({"role": "user", "content": user_input})

        while True:
            payload = {
                "messages": self.messages,
                "tools": TOOLS_SPEC,
                "temperature": 0.3,
                "max_tokens": 1024
            }

            try:
                resp = requests.post(self.api_url, json=payload, timeout=120)
                if resp.status_code != 200:
                    return f"Server Error ({resp.status_code}): {resp.text}"
            except requests.exceptions.ConnectionError:
                return "Cannot connect to llama-server at http://127.0.0.1:8080.\nPlease start your server: ~/models/qwen3.8-27b-server"
            except Exception as e:
                return f"Request failed: {e}"

            data = resp.json()
            choice = data["choices"][0]
            msg = choice["message"]
            finish_reason = choice.get("finish_reason")

            # Check if model requested tool call(s)
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                self.messages.append(msg)
                for tool in tool_calls:
                    fn_name = tool["function"]["name"]
                    raw_args = tool["function"].get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args = {}

                    print(f"\033[93m[Tool Call]\033[0m {fn_name}({args})")
                    result_str = self.execute_tool(fn_name, args)

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool["id"],
                        "name": fn_name,
                        "content": result_str
                    })
                # Re-query model with tool output
                continue

            # Model produced final answer
            content = msg.get("content", "")
            self.messages.append({"role": "assistant", "content": content})
            return content

def main():
    agent = LocalAgent()
    print("\033[92m=== Local RAG & Tool Calling Agent ===\033[0m")
    print("Connected to llama-server on http://127.0.0.1:8080")
    print("Type your question, or 'exit' to quit.\n")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\033[94mUser:\033[0m {query}")
        ans = agent.chat(query)
        print(f"\n\033[96mAssistant:\033[0m\n{ans}")
        return

    while True:
        try:
            user_input = input("\033[94mYou > \033[0m").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break
            response = agent.chat(user_input)
            print(f"\n\033[96mAssistant:\033[0m\n{response}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()
