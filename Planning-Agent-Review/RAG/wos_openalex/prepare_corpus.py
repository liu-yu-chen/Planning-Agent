"""Stream the DOI-filtered WoS corpus into an OpenAlex-aware RAG corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STORAGE_CONFIG = ROOT / "config" / "corpus_storage.json"
STORAGE = json.loads(STORAGE_CONFIG.read_text(encoding="utf-8")) if STORAGE_CONFIG.exists() else {}
SOURCE = Path(STORAGE.get("source_corpus_directory", str(ROOT / "database"))) / "wos_urban_planning.parquet"
OUT = HERE / "data"
CORPUS = OUT / "corpus.jsonl"
MANIFEST = OUT / "corpus_manifest.json"
BATCH_SIZE = 4096
COLUMNS = [
    "doi", "doi_normalized", "title", "abstract", "publication_year", "year_normalized",
    "journal", "authors", "language", "openalex_language", "openalex_id",
    "openalex_authorships", "openalex_referenced_works", "openalex_cited_by_count",
    "openalex_references_count", "openalex_primary_topic", "openalex_topics",
    "openalex_keywords", "openalex_primary_topic_id", "openalex_primary_topic_name",
    "openalex_subfield_id", "openalex_subfield_name", "openalex_field_id",
    "openalex_field_name", "openalex_domain_id", "openalex_domain_name",
    "open_access", "document_type", "wos_categories", "research_areas",
]
ENGLISH = {"en", "eng", "english"}


def parse_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def to_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUT.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(SOURCE)
    columns = [name for name in COLUMNS if name in parquet.schema_arrow.names]
    counts = {
        "source_rows": 0, "kept_rows": 0, "explicit_non_english_skipped": 0,
        "missing_title_skipped": 0, "openalex_matched": 0,
        "with_abstract": 0, "with_openalex_topics": 0, "with_openalex_keywords": 0,
        "with_authorships": 0, "with_reference_lists": 0,
    }
    seen: set[str] = set()

    with CORPUS.open("w", encoding="utf-8", newline="\n") as output:
        for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=columns):
            for row in batch.to_pylist():
                counts["source_rows"] += 1
                title = clean(row.get("title")) or clean(row.get("openalex_title"))
                abstract = clean(row.get("abstract"))
                if not title:
                    counts["missing_title_skipped"] += 1
                    continue

                languages = {
                    clean(row.get("language")).casefold(),
                    clean(row.get("openalex_language")).casefold(),
                } - {""}
                if languages and not (languages & ENGLISH):
                    counts["explicit_non_english_skipped"] += 1
                    continue

                doi = clean(row.get("doi_normalized") or row.get("doi")).lower()
                year = to_int(row.get("year_normalized") or row.get("publication_year"))
                openalex_id = clean(row.get("openalex_id"))
                if doi:
                    doc_id = doi
                elif openalex_id:
                    doc_id = openalex_id
                else:
                    identity = f"{title.casefold()}|{year or ''}"
                    doc_id = "sha256:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
                if doc_id in seen:
                    continue
                seen.add(doc_id)

                topics_raw = parse_json(row.get("openalex_topics"), [])
                primary_raw = parse_json(row.get("openalex_primary_topic"), {})
                keywords_raw = parse_json(row.get("openalex_keywords"), [])
                authorships_raw = parse_json(row.get("openalex_authorships"), [])
                references = parse_json(row.get("openalex_referenced_works"), [])
                if not isinstance(topics_raw, list):
                    topics_raw = []
                if not isinstance(keywords_raw, list):
                    keywords_raw = []
                if not isinstance(authorships_raw, list):
                    authorships_raw = []
                if not isinstance(references, list):
                    references = []

                topics = []
                for item in topics_raw:
                    if not isinstance(item, dict):
                        continue
                    subfield = item.get("subfield") or {}
                    field = item.get("field") or {}
                    domain = item.get("domain") or {}
                    topics.append({
                        "id": clean(item.get("id")),
                        "name": clean(item.get("display_name")),
                        "score": item.get("score"),
                        "subfield_id": clean(subfield.get("id")),
                        "subfield": clean(subfield.get("display_name")),
                        "field_id": clean(field.get("id")),
                        "field": clean(field.get("display_name")),
                        "domain_id": clean(domain.get("id")),
                        "domain": clean(domain.get("display_name")),
                    })

                keywords = []
                for item in keywords_raw:
                    if isinstance(item, dict):
                        keywords.append({
                            "id": clean(item.get("id")),
                            "name": clean(item.get("display_name")),
                            "score": item.get("score"),
                        })

                authorships = []
                for item in authorships_raw:
                    if not isinstance(item, dict):
                        continue
                    author = item.get("author") or {}
                    institutions = []
                    for inst in item.get("institutions") or []:
                        if isinstance(inst, dict):
                            institutions.append({
                                "id": clean(inst.get("id")),
                                "name": clean(inst.get("display_name")),
                                "ror": clean(inst.get("ror")),
                                "country_code": clean(inst.get("country_code")),
                            })
                    authorships.append({
                        "author_id": clean(author.get("id")),
                        "author": clean(author.get("display_name") or item.get("raw_author_name")),
                        "orcid": clean(author.get("orcid") or item.get("raw_orcid")),
                        "institutions": institutions,
                        "is_corresponding": bool(item.get("is_corresponding")),
                    })

                primary = primary_raw if isinstance(primary_raw, dict) else {}
                primary_subfield = primary.get("subfield") or {}
                primary_field = primary.get("field") or {}
                primary_domain = primary.get("domain") or {}
                metadata = {
                    "doi": doi, "year": year,
                    "journal": clean(row.get("journal")),
                    "authors_wos": clean(row.get("authors")),
                    "language": sorted(languages)[0] if languages else "",
                    "document_type": row.get("document_type"),
                    "openalex_id": openalex_id,
                    "openalex_primary_topic": {
                        "id": clean(primary.get("id")),
                        "name": clean(primary.get("display_name")),
                        "score": primary.get("score"),
                    } if primary else None,
                    "openalex_topics": topics,
                    "openalex_keywords": keywords,
                    "openalex_subfield": clean(primary_subfield.get("display_name") or row.get("openalex_subfield_name")),
                    "openalex_field": clean(primary_field.get("display_name") or row.get("openalex_field_name")),
                    "openalex_domain": clean(primary_domain.get("display_name") or row.get("openalex_domain_name")),
                    "openalex_authorships": authorships,
                    "openalex_referenced_works": [clean(ref) for ref in references if clean(ref)],
                    "openalex_cited_by_count": to_int(row.get("openalex_cited_by_count")),
                    "openalex_references_count": to_int(row.get("openalex_references_count")),
                    "open_access": row.get("open_access"),
                    "wos_categories": row.get("wos_categories") or [],
                    "research_areas": row.get("research_areas") or [],
                }
                topic_names = [topic["name"] for topic in topics if topic["name"]]
                keyword_names = [keyword["name"] for keyword in keywords if keyword["name"]]
                metadata["retrieval_labels"] = "\n".join(
                    part for part in (
                        "OpenAlex topics: " + "; ".join(topic_names) if topic_names else "",
                        "OpenAlex keywords: " + "; ".join(keyword_names) if keyword_names else "",
                    ) if part
                )
                parts = [f"Title: {title}"]
                if abstract:
                    parts.append(f"Abstract: {abstract}")
                if topic_names:
                    parts.append("OpenAlex topics (classification labels; may be imperfect): " + "; ".join(topic_names))
                if keyword_names:
                    parts.append("OpenAlex keywords (machine-assigned labels; may be imperfect): " + "; ".join(keyword_names))
                record = {
                    "id": doc_id, "title": title, "abstract": abstract,
                    "text": "\n\n".join(parts), "metadata": metadata,
                }
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                counts["kept_rows"] += 1
                counts["with_abstract"] += bool(abstract)
                counts["openalex_matched"] += bool(openalex_id)
                counts["with_openalex_topics"] += bool(topics)
                counts["with_openalex_keywords"] += bool(keywords)
                counts["with_authorships"] += bool(authorships)
                counts["with_reference_lists"] += bool(references)

    manifest = {
        "source": str(SOURCE),
        "output": str(CORPUS.relative_to(ROOT)),
        "record_count": counts["kept_rows"],
        "filter": "exclude explicit non-English; retain unknown language and title-only works",
        "retrieval_text": "title + abstract + explicitly labeled OpenAlex topics/keywords",
        "graph_sources": ["OpenAlex authorships", "OpenAlex referenced works", "OpenAlex topics", "OpenAlex keywords", "OpenAlex institutions"],
        "counts": counts,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
