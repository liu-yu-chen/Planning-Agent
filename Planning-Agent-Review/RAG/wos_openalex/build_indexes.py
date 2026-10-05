"""Build resumable SQLite BM25 and BGE dense indexes for the OpenAlex corpus."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
CORPUS = HERE / "data" / "corpus.jsonl"
OUT = HERE / "index"
DB_PATH = OUT / "rag.sqlite"
VECTOR_PATH = OUT / "embeddings.f32"
LOCAL_MODEL = HERE.parents[1] / "models" / "bge-large-en-v1.5" / "models" / "BAAI--bge-large-en-v1.5" / "snapshots" / "master"
MODEL = Path(os.environ.get("RAG_EMBEDDING_MODEL", str(LOCAL_MODEL)))
DIMENSION = 1024
CONTENT_TOKENS = 400
OVERLAP = 60


def init_db(db: sqlite3.Connection) -> None:
    db.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT NOT NULL,
            openalex_id TEXT,
            metadata_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_documents_openalex ON documents(openalex_id);
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY,
            doc_id TEXT NOT NULL REFERENCES documents(doc_id),
            chunk_no INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding_row INTEGER NOT NULL UNIQUE,
            UNIQUE(doc_id, chunk_no)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
            doc_id UNINDEXED, chunk_id UNINDEXED, text,
            tokenize='unicode61 remove_diacritics 2'
        );
        CREATE TABLE IF NOT EXISTS build_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    db.commit()


def make_chunks(doc: dict[str, Any], tokenizer: Any) -> list[str]:
    body = "\n\n".join([doc.get("abstract", ""), doc.get("metadata", {}).get("retrieval_labels", "")]).strip()
    if not body:
        body = doc["title"]
    # Calling encode() on long abstracts logs a misleading >512-token warning
    # before we split them. tokenize()+convert() yields the same untruncated IDs.
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(body))
    model_limit = int(getattr(tokenizer, "model_max_length", 512))
    if model_limit > 100_000:
        model_limit = 512
    header = f"Title: {doc['title']}"
    header_ids = tokenizer.encode(header, add_special_tokens=False)
    # Reserve room for the abstract, separators, and the encoder's special tokens.
    header_cap = min(180, max(32, model_limit - 64))
    if len(header_ids) > header_cap:
        header = tokenizer.decode(header_ids[:header_cap], skip_special_tokens=True)
        header_ids = tokenizer.encode(header, add_special_tokens=False)
    content_limit = max(32, min(CONTENT_TOKENS, model_limit - len(header_ids) - 16))
    stride = max(1, content_limit - OVERLAP)
    chunks = []
    for start in range(0, max(1, len(ids)), stride):
        piece_ids = ids[start:start + content_limit]
        piece = tokenizer.decode(piece_ids, skip_special_tokens=True).strip()
        if piece:
            text = f"{header}\n\n{piece}"
            # Tokenization around the title/body boundary can add tokens. Enforce
            # the encoder's actual input limit on the final assembled string.
            while len(tokenizer.tokenize(text)) + 2 > model_limit and len(piece_ids) > 16:
                piece_ids = piece_ids[:-16]
                piece = tokenizer.decode(piece_ids, skip_special_tokens=True).strip()
                text = f"{header}\n\n{piece}"
            if len(tokenizer.tokenize(text)) + 2 > model_limit:
                header = tokenizer.decode(header_ids[:max(16, model_limit // 2)], skip_special_tokens=True)
                text = f"{header}\n\n{piece}"
                while len(tokenizer.tokenize(text)) + 2 > model_limit and len(piece_ids) > 1:
                    piece_ids = piece_ids[:-8]
                    piece = tokenizer.decode(piece_ids, skip_special_tokens=True).strip()
                    text = f"{header}\n\n{piece}"
            chunks.append(text)
        if start + content_limit >= len(ids):
            break
    return chunks or [f"Title: {doc['title']}"]


def batches(lines, size: int):
    group = []
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        group.append((line_no, json.loads(line)))
        if len(group) >= size:
            yield group
            group = []
    if group:
        yield group


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-docs", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--limit", type=int, default=0, help="Optional pilot cap; zero indexes all records")
    args = parser.parse_args()
    if not CORPUS.exists():
        raise FileNotFoundError(f"Run prepare_corpus.py first: {CORPUS}")
    if not MODEL.exists():
        raise FileNotFoundError(f"BGE embedding model not found: {MODEL}")
    OUT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.execute("PRAGMA foreign_keys=ON")
    init_db(db)
    try:
        db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts_check USING fts5(text)")
        db.execute("DROP TABLE fts_check")
    except sqlite3.OperationalError as exc:
        raise RuntimeError("Python SQLite must include FTS5") from exc

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BGE model {MODEL} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, use_fast=True)
    encoder = SentenceTransformer(
        str(MODEL), device=device,
        model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {},
    )
    dimension = int(encoder.get_embedding_dimension())
    if dimension != DIMENSION:
        raise ValueError(f"Expected {DIMENSION}-dimensional BGE embeddings, got {dimension}")

    stored_rows = int(db.execute("SELECT COALESCE(MAX(embedding_row), -1) + 1 FROM chunks").fetchone()[0])
    VECTOR_PATH.touch(exist_ok=True)
    expected_bytes = stored_rows * dimension * 4
    if VECTOR_PATH.stat().st_size != expected_bytes:
        with VECTOR_PATH.open("r+b") as f:
            f.truncate(expected_bytes)

    indexed_docs = int(db.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    print(f"Resuming: {indexed_docs:,} documents, {stored_rows:,} chunks", flush=True)
    processed_now = 0
    last_report = 0
    started_at = time.monotonic()
    progress_file = OUT / "progress.json"
    manifest_path = HERE / "data" / "corpus_manifest.json"
    total_docs = 0
    if manifest_path.exists():
        total_docs = int(json.loads(manifest_path.read_text(encoding="utf-8")).get("record_count", 0))
    total_docs = total_docs or indexed_docs

    def report_progress(stage: str = "embedding_index") -> None:
        elapsed = max(time.monotonic() - started_at, 0.01)
        completed = min(indexed_docs + processed_now, total_docs) if total_docs else indexed_docs + processed_now
        rate = processed_now / elapsed
        remaining = max(0, total_docs - completed) if total_docs else 0
        eta_seconds = remaining / rate if rate > 0 else None

        def duration(value: float | None) -> str:
            if value is None:
                return "estimating"
            seconds = max(0, int(value))
            hours, rem = divmod(seconds, 3600)
            minutes, seconds = divmod(rem, 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        percent = 100.0 * completed / total_docs if total_docs else 0.0
        state = {
            "stage": stage, "completed_documents": completed, "total_documents": total_docs,
            "chunks": stored_rows, "percent": round(percent, 3),
            "documents_per_second": round(rate, 3), "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        progress_file.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"Progress: {completed:,}/{total_docs:,} docs ({percent:.1f}%) | "
            f"{rate:.1f} docs/s | elapsed {duration(elapsed)} | ETA {duration(eta_seconds)} | "
            f"{stored_rows:,} chunks",
            flush=True,
        )
    report_progress(stage="embedding_index_resuming")
    with CORPUS.open(encoding="utf-8") as source:
        for batch in batches(source, args.batch_docs):
            if args.limit and indexed_docs + processed_now >= args.limit:
                break
            docs = []
            for _line_no, doc in batch:
                if args.limit and indexed_docs + processed_now + len(docs) >= args.limit:
                    break
                exists = db.execute("SELECT 1 FROM documents WHERE doc_id=?", (doc["id"],)).fetchone()
                if not exists:
                    docs.append(doc)
            if not docs:
                continue

            expanded = [(doc, make_chunks(doc, tokenizer)) for doc in docs]
            flat_chunks = [(doc, no, text) for doc, texts in expanded for no, text in enumerate(texts)]
            vectors = encoder.encode(
                [item[2] for item in flat_chunks], batch_size=32 if device == "cuda" else 32,
                normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.shape != (len(flat_chunks), dimension):
                raise ValueError(f"Unexpected embedding array shape: {vectors.shape}")

            chunk_rows = []
            fts_rows = []
            first_embedding_row = stored_rows
            next_chunk_id = int(db.execute("SELECT COALESCE(MAX(chunk_id), 0) + 1 FROM chunks").fetchone()[0])
            for i, (doc, chunk_no, text) in enumerate(flat_chunks):
                chunk_id = next_chunk_id + i
                emb_row = stored_rows + i
                chunk_rows.append((chunk_id, doc["id"], chunk_no, text, emb_row))
                fts_rows.append((doc["id"], chunk_id, text))
            with VECTOR_PATH.open("ab") as vf:
                vectors.tofile(vf)
                vf.flush()
                os.fsync(vf.fileno())

            try:
                db.execute("BEGIN")
                db.executemany(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                    [(d["id"], d["title"], d.get("abstract", ""), d.get("metadata", {}).get("openalex_id", ""),
                      json.dumps(d.get("metadata", {}), ensure_ascii=False, separators=(",", ":"))) for d in docs],
                )
                db.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)", chunk_rows)
                db.executemany("INSERT INTO chunk_fts(doc_id, chunk_id, text) VALUES (?, ?, ?)", fts_rows)
                stored_rows += len(flat_chunks)
                processed_now += len(docs)
                db.execute("INSERT OR REPLACE INTO build_state VALUES ('embedding_rows', ?)", (str(stored_rows),))
                db.execute("INSERT OR REPLACE INTO build_state VALUES ('indexed_docs', ?)", (str(indexed_docs + processed_now),))
                db.commit()
            except BaseException:
                db.rollback()
                with VECTOR_PATH.open("r+b") as vf:
                    vf.truncate(first_embedding_row * dimension * 4)
                raise

            if processed_now - last_report >= 5_000:
                report_progress()
                last_report = processed_now

    db.execute("ANALYZE")
    db.commit()
    report_progress(stage="embedding_index_complete")
    db.close()
    manifest = {
        "corpus": "../data/corpus.jsonl", "database": "rag.sqlite",
        "embedding_file": "embeddings.f32", "embedding_model": str(MODEL),
        "embedding_dimension": dimension, "normalization": "L2 normalized",
        "chunk_tokens": CONTENT_TOKENS, "overlap_tokens": OVERLAP,
        "documents_indexed": indexed_docs + processed_now, "chunks_indexed": stored_rows,
        "device": device, "resumable": True,
        "strategies": ["ordinary_hybrid_rag", "openalex_graph_enhanced_rag"],
    }
    (OUT / "index_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
