import os
import sqlite3
import subprocess
import numpy as np
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import openvino as ov
from tokenizers import Tokenizer
from pypdf import PdfReader

from pathlib import Path
 
THIS_DIR = Path(__file__).resolve().parent
DB_PATH = str(THIS_DIR / "knowledge.db")
MODEL_DIR = str(THIS_DIR / "models" / "bge-small-en-v1.5")

class IntelIGPUEmbedder:
    """High-throughput vector embedding engine running on Intel UHD iGPU via OpenVINO."""
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = os.path.abspath(os.path.expanduser(model_dir))
        self.core = ov.Core()

        # Prioritize Intel iGPU (GPU.0 / Intel UHD) over dGPU / CPU
        target_device = "CPU"
        for d in self.core.available_devices:
            if d.startswith("GPU") and "Intel" in self.core.get_property(d, "FULL_DEVICE_NAME"):
                target_device = d
                break

        self.device = target_device
        self.device_name = self.core.get_property(self.device, "FULL_DEVICE_NAME")

        onnx_model_path = os.path.join(self.model_dir, "model_optimized.onnx")
        self.compiled = self.core.compile_model(onnx_model_path, self.device)

        self.tokenizer = Tokenizer.from_file(os.path.join(self.model_dir, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)

    def embed(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Compute normalized CLS embeddings for a list of strings on Intel iGPU."""
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded_batch = self.tokenizer.encode_batch(batch_texts)

            max_len = max(len(e.ids) for e in encoded_batch)
            max_len = max(max_len, 8)

            input_ids = np.zeros((len(encoded_batch), max_len), dtype=np.int64)
            attention_mask = np.zeros((len(encoded_batch), max_len), dtype=np.int64)
            token_type_ids = np.zeros((len(encoded_batch), max_len), dtype=np.int64)

            for row_idx, e in enumerate(encoded_batch):
                seq_len = len(e.ids)
                input_ids[row_idx, :seq_len] = e.ids
                attention_mask[row_idx, :seq_len] = e.attention_mask
                token_type_ids[row_idx, :seq_len] = e.type_ids

            outputs = self.compiled([input_ids, attention_mask, token_type_ids])
            last_hidden_state = list(outputs.values())[0]

            cls_vectors = last_hidden_state[:, 0, :]
            norms = np.linalg.norm(cls_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = cls_vectors / norms

            all_embeddings.extend([vec.astype(np.float32) for vec in normalized])

        return all_embeddings

class RAGEngine:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = os.path.abspath(os.path.expanduser(db_path))
        self._init_db()
        # Initialize Intel UHD iGPU embedding engine
        self.embedder = IntelIGPUEmbedder()

    @property
    def device_info(self) -> str:
        return f"{self.embedder.device} ({self.embedder.device_name})"

    @contextmanager
    def _get_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._get_db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE,
                    filepath TEXT,
                    pages INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER,
                    page_num INTEGER,
                    chunk_index INTEGER,
                    text TEXT,
                    embedding BLOB,
                    FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)

    def chunk_text(self, text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
        """Split text into overlapping character windows."""
        text = " ".join(text.split())
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def extract_pdf_pages(self, pdf_path: str) -> List[str]:
        """Extract text per page using native C++ Poppler pinned to E-cores (16-27)."""
        try:
            res = subprocess.run(
                ['taskset', '-c', '16-27', 'pdftotext', pdf_path, '-'],
                capture_output=True, text=True, errors='ignore', timeout=60
            )
            if res.returncode == 0 and res.stdout:
                pages = res.stdout.split('\x0c')
                if pages and not pages[-1].strip():
                    pages.pop()
                if pages:
                    return pages
        except Exception:
            pass

        try:
            reader = PdfReader(pdf_path)
            return [page.extract_text() or "" for page in reader.pages]
        except Exception:
            return []

    def index_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text on E-cores, chunk, embed on Intel iGPU, and store in SQLite."""
        pdf_path = os.path.abspath(os.path.expanduser(pdf_path))
        if not os.path.isfile(pdf_path):
            return {"status": "error", "message": f"File not found: {pdf_path}"}

        filename = os.path.basename(pdf_path)
        pages = self.extract_pdf_pages(pdf_path)
        total_pages = len(pages)

        if not pages or not any(p.strip() for p in pages):
            return {"status": "error", "message": "No readable text found in PDF"}

        all_chunks = []
        metadata = []

        for page_idx, page_text in enumerate(pages):
            page_chunks = self.chunk_text(page_text)
            for chunk_idx, chunk in enumerate(page_chunks):
                all_chunks.append(chunk)
                metadata.append((page_idx + 1, chunk_idx, chunk))

        if not all_chunks:
            return {"status": "error", "message": "No text chunks generated from PDF"}

        # Compute embeddings in batches directly on the Intel UHD iGPU
        embeddings = self.embedder.embed(all_chunks, batch_size=32)

        with self._get_db() as conn:
            conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO documents (filename, filepath, pages) VALUES (?, ?, ?)",
                (filename, pdf_path, total_pages)
            )
            doc_id = cursor.lastrowid

            records = []
            for (page_num, chunk_idx, text), emb in zip(metadata, embeddings):
                emb_bytes = emb.tobytes()
                records.append((doc_id, page_num, chunk_idx, text, emb_bytes))

            cursor.executemany(
                "INSERT INTO chunks (doc_id, page_num, chunk_index, text, embedding) VALUES (?, ?, ?, ?, ?)",
                records
            )

        return {
            "status": "success",
            "filename": filename,
            "pages": total_pages,
            "chunks": len(all_chunks),
            "device": self.device_info
        }

    def search(self, query: str, top_k: int = 4, doc_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find the most semantically relevant chunks for a query."""
        query_emb = self.embedder.embed([query])[0]
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            return []

        with self._get_db() as conn:
            cursor = conn.cursor()
            if doc_filter:
                cursor.execute("""
                    SELECT c.id, c.page_num, c.text, c.embedding, d.filename
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.id
                    WHERE d.filename LIKE ?
                """, (f"%{doc_filter}%",))
            else:
                cursor.execute("""
                    SELECT c.id, c.page_num, c.text, c.embedding, d.filename
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.id
                """)
            rows = cursor.fetchall()

        if not rows:
            return []

        results = []
        for row_id, page_num, text, emb_bytes, filename in rows:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            denom = np.linalg.norm(emb) * query_norm
            sim = float(np.dot(emb, query_emb) / denom) if denom > 0 else 0.0
            results.append({
                "score": round(sim, 4),
                "filename": filename,
                "page": page_num,
                "text": text
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all indexed documents and chunk counts."""
        with self._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT d.id, d.filename, d.pages, d.created_at, COUNT(c.id) as chunk_count
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.doc_id
                GROUP BY d.id
            """)
            rows = cursor.fetchall()
            return [
                {"id": r[0], "filename": r[1], "pages": r[2], "date": r[3], "chunks": r[4]}
                for r in rows
            ]
