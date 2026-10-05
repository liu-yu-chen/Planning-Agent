"""Deduplicate the prepared corpus by DOI, OpenAlex ID, then normalized title+year."""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CORPUS = DATA / "corpus.jsonl"
MANIFEST = DATA / "corpus_manifest.json"
TEMP = DATA / "corpus.deduplicated.tmp"
BACKUP = DATA / "corpus.pre_dedup.jsonl"
REPORT = DATA / "deduplication_summary.json"
DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)", re.I)
NON_ALNUM = re.compile(r"[^\w]+", re.UNICODE)


def clean_doi(value: object) -> str:
    value = str(value or "").strip().casefold()
    while DOI_PREFIX.match(value):
        value = DOI_PREFIX.sub("", value, count=1)
    return value.rstrip(" .;,)")


def normalized_title(value: object) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(NON_ALNUM.sub(" ", value).split())


def identities(row: dict) -> dict[str, str]:
    meta = row.get("metadata") or {}
    keys = {}
    doi = clean_doi(meta.get("doi"))
    oa_id = str(meta.get("openalex_id") or "").strip().casefold()
    title = normalized_title(row.get("title"))
    year = meta.get("year")
    try:
        year = str(int(year)) if year is not None else ""
    except (TypeError, ValueError):
        year = ""
    if doi:
        keys["doi"] = "doi:" + doi
    if oa_id:
        keys["openalex_id"] = "openalex:" + oa_id
    if title and year:
        keys["title_year"] = "title_year:" + title + "|" + year
    return keys


def main() -> None:
    if not CORPUS.exists():
        raise FileNotFoundError(CORPUS)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    if manifest.get("deduplicated") and manifest.get("deduplication_policy"):
        print(f"Corpus already deduplicated: {manifest.get('record_count'):,} records; skipping rewrite.", flush=True)
        return
    total = int(manifest.get("record_count", 0))
    started = time.monotonic()
    rows = duplicates = 0
    by_key = Counter()
    seen: set[str] = set()
    with CORPUS.open("r", encoding="utf-8") as source, TEMP.open("w", encoding="utf-8", newline="\n") as output:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            keys = identities(row)
            duplicate_type = next((kind for kind in ("doi", "openalex_id", "title_year")
                                   if keys.get(kind) in seen), None)
            # Add all aliases even for a duplicate so later records bridge identities consistently.
            seen.update(keys.values())
            if duplicate_type:
                duplicates += 1
                by_key[duplicate_type] += 1
            else:
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            if rows % 10_000 == 0:
                elapsed = max(time.monotonic() - started, 0.01)
                rate = rows / elapsed
                eta = (total - rows) / rate if total and rate else None
                eta_text = f"{eta / 60:.1f} min" if eta is not None else "estimating"
                print(f"Deduplicating: {rows:,}/{total:,} | removed {duplicates:,} | {rate:,.0f} rows/s | ETA {eta_text}", flush=True)
        output.flush()
        os.fsync(output.fileno())
    if rows != total and total:
        raise RuntimeError(f"Read {rows:,} corpus rows, expected {total:,}; kept temp for inspection.")
    kept = rows - duplicates
    os.replace(CORPUS, BACKUP)
    os.replace(TEMP, CORPUS)
    result = {
        "input_records": rows,
        "unique_records": kept,
        "duplicates_removed": duplicates,
        "duplicates_by_priority": dict(by_key),
        "identity_priority": ["normalized DOI", "OpenAlex work ID", "normalized title + publication year"],
        "fuzzy_title_matching": "not applied; exact normalized title + year is used for deterministic, scalable deduplication",
        "original_corpus_backup": BACKUP.name,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    report_tmp = REPORT.with_suffix(".tmp")
    report_tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(report_tmp, REPORT)
    manifest.update({"record_count": kept, "deduplicated": True,
                     "deduplication_policy": result["identity_priority"],
                     "deduplication_summary": REPORT.name})
    manifest_tmp = MANIFEST.with_suffix(".tmp")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, MANIFEST)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
