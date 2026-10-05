"""Compare ordinary hybrid RAG with OpenAlex graph-enhanced RAG."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
import torch


HERE = Path(__file__).resolve().parent
INDEX = HERE / "index"
DB_PATH = INDEX / "rag.sqlite"
VECTOR_PATH = INDEX / "embeddings.f32"
MODEL = Path(os.environ.get(
    "RAG_EMBEDDING_MODEL",
    str(HERE.parents[1] / "models" / "bge-large-en-v1.5" / "models" / "BAAI--bge-large-en-v1.5" / "snapshots" / "master"),
))
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
OLLAMA_URL = os.environ.get("RAG_GENERATION_URL", "http://localhost:11434/api/chat")
GENERATION_MODEL = os.environ.get("RAG_GENERATION_MODEL", "qwen3-4b-thinking-2507")
DIMENSION = 1024
RELATION_WEIGHT = {"author": 1.0, "institution": 0.45, "topic": 0.85, "keyword": 0.65}
_DENSE_ENCODER = None


def fts_search(db: sqlite3.Connection, query: str, limit: int):
    terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
    if not terms or limit <= 0:
        return []
    match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
    try:
        # FTS5's rank column uses BM25 by default and can apply its ranking
        # optimization before joins. Join only the small ranked candidate set;
        # expand it until the exact first `limit` eligible chunks are found.
        # Terms, scoring and the eligibility policy are preserved.
        budget = max(64, limit * 3)
        while True:
            hits = db.execute("""SELECT chunk_id, rank FROM chunk_fts
                WHERE chunk_fts MATCH ? ORDER BY rank LIMIT ?""", (match, budget)).fetchall()
            selected = []
            for chunk_id, score in hits:
                row = db.execute("""SELECT c.chunk_id, c.doc_id, c.text FROM chunks c
                    JOIN citation_eligibility e ON e.doc_id=c.doc_id AND e.eligible=1
                    WHERE c.chunk_id=?""", (int(chunk_id),)).fetchone()
                if row:
                    selected.append((*row, score))
                    if len(selected) == limit:
                        return selected
            if len(hits) < budget:
                return selected
            budget *= 2
    except sqlite3.OperationalError:
        return []


def dense_search(db: sqlite3.Connection, query: str, limit: int):
    global _DENSE_ENCODER
    if not VECTOR_PATH.exists() or not MODEL.exists():
        raise FileNotFoundError("Dense index or local BGE model is missing; run build_indexes.py first.")
    chunk_count = int(db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    if not chunk_count:
        return []
    vectors = np.memmap(VECTOR_PATH, mode="r", dtype=np.float32, shape=(chunk_count, DIMENSION))
    eligible_rows = np.zeros(chunk_count, dtype=np.bool_)
    eligible_rows_db = db.execute("""
        SELECT c.embedding_row FROM chunks c JOIN citation_eligibility e ON e.doc_id=c.doc_id
        WHERE e.eligible=1
    """)
    for (embedding_row,) in eligible_rows_db:
        if 0 <= embedding_row < chunk_count:
            eligible_rows[embedding_row] = True
    if _DENSE_ENCODER is None:
        device = os.environ.get("RAG_EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        _DENSE_ENCODER = SentenceTransformer(str(MODEL), device=device)
    qvec = _DENSE_ENCODER.encode([QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
    best_scores = np.empty(0, dtype=np.float32)
    best_rows = np.empty(0, dtype=np.int64)
    block = 20_000
    for start in range(0, chunk_count, block):
        stop = min(start + block, chunk_count)
        scores = np.asarray(vectors[start:stop] @ qvec, dtype=np.float32)
        scores[~eligible_rows[start:stop]] = -np.inf
        keep = min(limit, len(scores))
        ix = np.argpartition(scores, -keep)[-keep:]
        best_scores = np.concatenate((best_scores, scores[ix]))
        best_rows = np.concatenate((best_rows, ix.astype(np.int64) + start))
        if len(best_rows) > limit:
            top = np.argpartition(best_scores, -limit)[-limit:]
            best_scores, best_rows = best_scores[top], best_rows[top]
    order = np.argsort(best_scores)[::-1]
    selected = best_rows[order]
    scores = best_scores[order]
    row_map = {int(row): (float(score),) for row, score in zip(selected, scores)}
    placeholders = ",".join("?" for _ in selected)
    records = db.execute(
        f"SELECT chunk_id, doc_id, text, embedding_row FROM chunks WHERE embedding_row IN ({placeholders})",
        [int(x) for x in selected],
    ).fetchall()
    by_row = {int(row[3]): row for row in records}
    return [(by_row[int(row)][0], by_row[int(row)][1], by_row[int(row)][2], row_map[int(row)][0])
            for row in selected if int(row) in by_row]


def hybrid(sparse, dense):
    fused: dict[str, dict[str, Any]] = {}
    for rows in (sparse, dense):
        seen = set()
        rank = 0
        for chunk_id, doc_id, text, score in rows:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank += 1
            item = fused.setdefault(doc_id, {"doc_id": doc_id, "chunk_id": chunk_id, "text": text, "rrf": 0.0})
            item["rrf"] += 1.0 / (60 + rank)
    return sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)


def graph_expand(db: sqlite3.Connection, seeds: list[dict[str, Any]], candidate_k: int):
    seed_score = {row["doc_id"]: row["rrf"] for row in seeds[:min(candidate_k, 80)]}
    graph_score: dict[str, float] = defaultdict(float)
    evidence: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    max_topic_degree = 1500
    for seed_id, score in seed_score.items():
        memberships = db.execute(
            "SELECT relation, entity_key, entity_label FROM graph_memberships WHERE doc_id=?", (seed_id,)
        ).fetchall()
        for relation, entity_key, label in memberships:
            degree_row = db.execute(
                "SELECT COUNT(*) FROM graph_memberships WHERE relation=? AND entity_key=?",
                (relation, entity_key),
            ).fetchone()
            degree = int(degree_row[0])
            cap = max_topic_degree if relation in ("topic", "keyword") else 500
            if degree < 2 or degree > cap:
                continue
            peers = db.execute(
            "SELECT gm.doc_id FROM graph_memberships gm JOIN citation_eligibility ce ON ce.doc_id=gm.doc_id "
            "WHERE gm.relation=? AND gm.entity_key=? AND gm.doc_id!=? AND ce.eligible=1 LIMIT 1000",
                (relation, entity_key, seed_id),
            ).fetchall()
            contribution = score * RELATION_WEIGHT.get(relation, 0.5) / math.log2(degree + 1)
            for (peer_id,) in peers:
                graph_score[peer_id] += contribution
                evidence[peer_id][relation].add(label or entity_key)

        cited = db.execute("SELECT ce.target_doc_id FROM citation_edges ce JOIN citation_eligibility e "
                           "ON e.doc_id=ce.target_doc_id WHERE ce.source_doc_id=? AND e.eligible=1 LIMIT 500", (seed_id,))
        for (peer_id,) in cited:
            graph_score[peer_id] += score * 1.35
            evidence[peer_id]["cites"].add(seed_id)
        citing = db.execute("SELECT ce.source_doc_id FROM citation_edges ce JOIN citation_eligibility e "
                            "ON e.doc_id=ce.source_doc_id WHERE ce.target_doc_id=? AND e.eligible=1 LIMIT 500", (seed_id,))
        for (peer_id,) in citing:
            graph_score[peer_id] += score * 1.15
            evidence[peer_id]["cited_by"].add(seed_id)

    if not graph_score:
        return seeds[:candidate_k]
    graph_max = max(graph_score.values())
    base_max = max((row["rrf"] for row in seeds), default=1.0)
    scores = {row["doc_id"]: row["rrf"] for row in seeds}
    payload = {row["doc_id"]: dict(row) for row in seeds}
    candidates = sorted(graph_score, key=graph_score.get, reverse=True)[:candidate_k]
    placeholders = ",".join("?" for _ in candidates)
    docs = db.execute(
        f"SELECT d.doc_id, d.title, d.abstract, d.metadata_json FROM documents d "
        f"JOIN citation_eligibility e ON e.doc_id=d.doc_id AND e.eligible=1 WHERE d.doc_id IN ({placeholders})",
        candidates,
    ).fetchall()
    for doc_id, title, abstract, raw_meta in docs:
        scores[doc_id] = scores.get(doc_id, 0.0) + base_max * 0.8 * graph_score[doc_id] / graph_max
        item = payload.setdefault(doc_id, {"doc_id": doc_id, "chunk_id": None, "text": "", "rrf": 0.0})
        item["title"], item["abstract"], item["metadata"] = title, abstract, json.loads(raw_meta)
        item["graph_evidence"] = {k: sorted(v)[:5] for k, v in evidence[doc_id].items()}
    ordered = sorted(scores, key=scores.get, reverse=True)[:candidate_k]
    return [payload[doc_id] | {"score": scores[doc_id]} for doc_id in ordered if doc_id in payload]


def hydrate(db: sqlite3.Connection, rows: list[dict[str, Any]], top_k: int):
    result = []
    for row in rows:
        paper = db.execute(
            "SELECT d.title, d.abstract, d.metadata_json FROM documents d "
            "JOIN citation_eligibility e ON e.doc_id=d.doc_id AND e.eligible=1 WHERE d.doc_id=?", (row["doc_id"],)
        ).fetchone()
        if not paper:
            continue
        title, abstract, metadata = paper
        item = {**row, "title": title, "abstract": abstract, "metadata": json.loads(metadata)}
        result.append(item)
        if len(result) >= top_k:
            break
    return result


def render_context(results: list[dict[str, Any]], max_chars: int = 18_000) -> str:
    papers = []
    used = 0
    for i, row in enumerate(results, 1):
        meta = row["metadata"]
        doi = meta.get("doi") or row["doc_id"]
        topics = [x.get("name") for x in meta.get("openalex_topics", []) if x.get("name")][:5]
        keywords = [x.get("name") for x in meta.get("openalex_keywords", []) if x.get("name")][:8]
        authorships = meta.get("openalex_authorships", [])
        authors = [x.get("author") for x in authorships if x.get("author")][:8]
        institutions = sorted({
            inst.get("name") for author in authorships for inst in author.get("institutions", [])
            if inst.get("name")
        })[:6]
        graph = row.get("graph_evidence") or {}
        graph_text = "; ".join(f"{kind}: {', '.join(values)}" for kind, values in graph.items() if values)
        block = (
            f"[{i}] DOI: {doi}\nTitle: {row['title']}\nYear: {meta.get('year')}\n"
            f"Journal: {meta.get('journal')}\nAuthors: {', '.join(authors)}\nInstitutions: {', '.join(institutions)}\n"
            f"OpenAlex cited-by count: {meta.get('openalex_cited_by_count')}\n"
            f"OpenAlex topics (discovery metadata, not proof): {', '.join(topics)}\n"
            f"OpenAlex keywords (discovery metadata, not proof): {', '.join(keywords)}\n"
            f"Abstract: {row.get('abstract') or '[No abstract in corpus]'}\n"
        )
        if graph_text:
            block += f"Bibliographic graph links to retrieved papers: {graph_text}\n"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 300:
                papers.append(block[:remaining])
            break
        papers.append(block)
        used += len(block)
    return "\n---\n".join(papers)


def ask_qwen(question: str, context: str, model: str) -> str:
    payload = {
        "model": model, "stream": False,
        "messages": [
            {"role": "system", "content": (
                "Answer using only the supplied literature records. OpenAlex topics and keywords are discovery labels and may be imperfect; "
                "do not treat them as evidence. Bibliographic graph links show citation/coauthor/topic relationships, not proof of a claim. "
                "Cite supporting records by DOI in square brackets. If records do not support an answer, say so. Answer in the language of the question."
            )},
            {"role": "user", "content": f"Question: {question}\n\nRetrieved literature:\n{context}"},
        ],
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    request = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}. Start Ollama and create the model from Modelfile first."
        ) from exc
    return result.get("message", {}).get("content", "")


def retrieve(db: sqlite3.Connection, query: str, candidate_k: int, top_k: int, graph: bool):
    sparse = fts_search(db, query, candidate_k)
    dense = dense_search(db, query, candidate_k)
    seeds = hybrid(sparse, dense)
    ranked = graph_expand(db, seeds, candidate_k) if graph else seeds
    return hydrate(db, ranked, top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="English works best with the current English BGE model")
    parser.add_argument("--strategy", choices=["ordinary", "graph", "both"], default="both")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--model", default=GENERATION_MODEL)
    parser.add_argument("--retrieval-only", action="store_true", help="Skip Qwen generation")
    parser.add_argument("--output", type=Path, help="Append one comparison result as JSONL")
    args = parser.parse_args()
    if not DB_PATH.exists():
        raise SystemExit("Index database missing. Run prepare_corpus.py, build_indexes.py, then build_graph.py.")
    db = sqlite3.connect(DB_PATH)
    modes = ["graph", "ordinary"] if args.strategy == "both" else [args.strategy]
    report: dict[str, Any] = {
        "query": args.query,
        "retrieval_query": args.query,
        "model": None if args.retrieval_only else args.model,
        "context_budget_chars": 18_000,
        "strategies": {},
    }
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='citation_eligibility'").fetchone():
        raise SystemExit("Citation eligibility not ready. Wait for build_all.py to finish indexing, filtering, and graph construction.")
    retrieval_started = time.perf_counter()
    sparse = fts_search(db, args.query, args.candidate_k)
    dense = dense_search(db, args.query, args.candidate_k)
    seeds = hybrid(sparse, dense)
    retrieval_seconds = time.perf_counter() - retrieval_started
    for mode in modes:
        mode_retrieval_started = time.perf_counter()
        ranked = graph_expand(db, seeds, args.candidate_k) if mode == "graph" else seeds
        results = hydrate(db, ranked, args.top_k)
        mode_retrieval_seconds = time.perf_counter() - mode_retrieval_started
        context = render_context(results)
        entry: dict[str, Any] = {
            "results": results,
            "metrics": {
                "shared_hybrid_retrieval_seconds": retrieval_seconds,
                "strategy_rerank_and_hydrate_seconds": mode_retrieval_seconds,
                "total_retrieval_seconds": retrieval_seconds + mode_retrieval_seconds,
                "retrieved_documents": len(results),
                "candidate_documents": len(seeds),
                "context_characters": len(context),
                "graph_expanded_documents": sum(1 for row in results if row.get("graph_evidence")) if mode == "graph" else 0,
            },
        }
        if not args.retrieval_only:
            generation_started = time.perf_counter()
            entry["answer"] = ask_qwen(args.query, context, args.model)
            entry["metrics"]["generation_seconds"] = time.perf_counter() - generation_started
            entry["metrics"]["answer_characters"] = len(entry["answer"])
        report["strategies"][mode] = entry
    db.close()
    report["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    output = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(report, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
