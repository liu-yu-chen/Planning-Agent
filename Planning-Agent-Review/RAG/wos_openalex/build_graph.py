"""Build OpenAlex author, institution, topic, keyword and citation links in SQLite."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
DB = HERE / "index" / "rag.sqlite"
MANIFEST = HERE / "index" / "graph_manifest.json"
BATCH = 20_000


def entity(value: dict, id_key: str = "id", label_key: str = "name"):
    if not isinstance(value, dict):
        return None
    key = (value.get(id_key) or "").strip()
    label = (value.get(label_key) or "").strip()
    return (key or label.casefold(), label or key) if (key or label) else None


def main() -> None:
    if not DB.exists():
        raise SystemExit("Run build_indexes.py before build_graph.py")
    db = sqlite3.connect(DB)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS graph_memberships (
            doc_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            entity_key TEXT NOT NULL,
            entity_label TEXT NOT NULL,
            weight REAL NOT NULL,
            PRIMARY KEY(doc_id, relation, entity_key)
        );
        CREATE INDEX IF NOT EXISTS idx_graph_entity ON graph_memberships(relation, entity_key);
        CREATE INDEX IF NOT EXISTS idx_graph_doc ON graph_memberships(doc_id);
        CREATE TABLE IF NOT EXISTS citation_edges (
            source_doc_id TEXT NOT NULL,
            target_doc_id TEXT NOT NULL,
            PRIMARY KEY(source_doc_id, target_doc_id)
        );
        CREATE INDEX IF NOT EXISTS idx_citation_target ON citation_edges(target_doc_id);
    """)
    total = int(db.execute("SELECT COUNT(*) FROM documents d JOIN citation_eligibility e ON e.doc_id=d.doc_id WHERE e.eligible=1").fetchone()[0])
    work_map = {key: doc for doc, key in db.execute(
        "SELECT d.doc_id, d.openalex_id FROM documents d JOIN citation_eligibility e ON e.doc_id=d.doc_id "
        "WHERE e.eligible=1 AND d.openalex_id IS NOT NULL AND d.openalex_id != ''"
    )}
    # This graph is rebuilt idempotently so a partially interrupted build is safe to rerun.
    db.execute("DELETE FROM graph_memberships")
    db.execute("DELETE FROM citation_edges")
    db.commit()
    cursor = db.execute("SELECT d.doc_id, d.metadata_json FROM documents d "
                         "JOIN citation_eligibility e ON e.doc_id=d.doc_id WHERE e.eligible=1 ORDER BY d.doc_id")
    memberships = []
    citations = []
    processed = 0
    relation_counts = {"author": 0, "institution": 0, "topic": 0, "keyword": 0}
    started_at = time.monotonic()
    last_report = 0

    while True:
        rows = cursor.fetchmany(1000)
        if not rows:
            break
        for doc_id, raw in rows:
            meta = json.loads(raw)
            for author in meta.get("openalex_authorships") or []:
                if not isinstance(author, dict):
                    continue
                item = entity({"id": author.get("author_id"), "name": author.get("author")})
                if item:
                    memberships.append((doc_id, "author", item[0], item[1], 1.0))
                    relation_counts["author"] += 1
                for institution in author.get("institutions") or []:
                    inst_key = (institution.get("ror") or institution.get("id") or institution.get("name") or "").strip()
                    inst_label = (institution.get("name") or inst_key).strip()
                    if inst_key:
                        memberships.append((doc_id, "institution", inst_key, inst_label, 0.65))
                        relation_counts["institution"] += 1
            for topic in meta.get("openalex_topics") or []:
                item = entity(topic)
                if item:
                    memberships.append((doc_id, "topic", item[0], item[1], 1.0))
                    relation_counts["topic"] += 1
            for keyword in meta.get("openalex_keywords") or []:
                item = entity(keyword)
                if item:
                    memberships.append((doc_id, "keyword", item[0], item[1], 0.7))
                    relation_counts["keyword"] += 1
            for referenced_work in meta.get("openalex_referenced_works") or []:
                target = work_map.get(referenced_work)
                if target and target != doc_id:
                    citations.append((doc_id, target))
            processed += 1

        if len(memberships) >= BATCH or len(citations) >= BATCH:
            db.executemany("INSERT OR IGNORE INTO graph_memberships VALUES (?, ?, ?, ?, ?)", memberships)
            db.executemany("INSERT OR IGNORE INTO citation_edges VALUES (?, ?)", citations)
            db.commit()
            memberships.clear()
            citations.clear()
        if processed - last_report >= 10_000:
            elapsed = max(time.monotonic() - started_at, 0.01)
            rate = processed / elapsed
            eta = (total - processed) / rate if rate else None
            hours, rem = divmod(int(eta or 0), 3600)
            minutes, seconds = divmod(rem, 60)
            eta_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}" if eta is not None else "estimating"
            percent = 100 * processed / total if total else 100.0
            print(f"Graph: {processed:,}/{total:,} docs ({percent:.1f}%) | {rate:.1f} docs/s | ETA {eta_text}", flush=True)
            progress = {
                "stage": "graph", "completed_documents": processed, "total_documents": total,
                "percent": round(percent, 3), "documents_per_second": round(rate, 3),
                "eta_seconds": round(eta, 1) if eta is not None else None,
                "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            (HERE / "index" / "progress.json").write_text(
                json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            last_report = processed

    if memberships:
        db.executemany("INSERT OR IGNORE INTO graph_memberships VALUES (?, ?, ?, ?, ?)", memberships)
    if citations:
        db.executemany("INSERT OR IGNORE INTO citation_edges VALUES (?, ?)", citations)
    db.execute("ANALYZE")
    db.commit()

    edge_counts = dict(db.execute("SELECT relation, COUNT(*) FROM graph_memberships GROUP BY relation"))
    entity_counts = dict(db.execute("SELECT relation, COUNT(DISTINCT entity_key) FROM graph_memberships GROUP BY relation"))
    citation_count = db.execute("SELECT COUNT(*) FROM citation_edges").fetchone()[0]
    db.close()
    manifest = {
        "database": "rag.sqlite", "documents": total,
        "membership_edges": edge_counts, "distinct_entities": entity_counts,
        "within_corpus_citation_edges": citation_count,
        "source": "OpenAlex authorships, institutions, topics, keywords and referenced_works",
        "note": "Citation edges are retained only when both works occur in the DOI-filtered corpus.",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (HERE / "index" / "progress.json").write_text(json.dumps({
        "stage": "graph_complete", "completed_documents": total, "total_documents": total,
        "percent": 100.0, "documents_per_second": None, "eta_seconds": 0,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
