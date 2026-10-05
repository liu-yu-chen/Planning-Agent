"""Dry-run and resumable two-stage WoS collector (Starter-compatible fallback)."""
from __future__ import annotations
import argparse, json, logging, os, re, sqlite3, sys, time, hashlib, shutil, random
from datetime import datetime, timezone
from pathlib import Path
import yaml
import pyarrow as pa
import pyarrow.parquet as pq

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from .wos_client import WoSClient
    from .wos_deduplicate import exact_key, fuzzy_title_score
    from .wos_normalize import normalize_document
    from .wos_queries import CORE_QUERY, CROSS_QUERIES, doc_type_filter, query_for_api
    from .wos_schema import BOOL_FIELDS, INT_FIELDS, LIST_FIELDS, OPAQUE_FIELDS, SCHEMA_FIELDS
except ImportError:  # python collection/wos_collector.py
    from wos_client import WoSClient
    from wos_deduplicate import exact_key, fuzzy_title_score
    from wos_normalize import normalize_document
    from wos_queries import CORE_QUERY, CROSS_QUERIES, doc_type_filter, query_for_api
    from wos_schema import BOOL_FIELDS, INT_FIELDS, LIST_FIELDS, OPAQUE_FIELDS, SCHEMA_FIELDS

ROOT = Path(__file__).resolve().parents[1]
CORE_STARTER_PROXY = 'TS=("REPLACE_WITH_APPROVED_SEARCH_CONCEPT")'

def config_load(path: Path = ROOT / "config" / "wos_config.yaml") -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for section in ("paths",):
        for key, value in cfg[section].items():
            cfg[section][key] = str((ROOT / value).resolve())
    return cfg

class RequestLedger:
    """Persist daily API usage before every actual request, including retries."""
    def __init__(self, path: str, cap: int):
        self.path, self.cap = Path(path), cap
        self.day = datetime.now(timezone.utc).date().isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if data.get("utc_day") == self.day:
            self.count = int(data.get("requests", 0))
        else:
            self.count = 0
            # Carry forward same-day calls made by the previous WoS crawler.
            legacy = self.path.parent / "wos_urban_planning_request_ledger.json"
            try:
                old = json.loads(legacy.read_text(encoding="utf-8"))
                if old.get("utc_day") == self.day:
                    self.count = int(old.get("requests", 0))
            except (OSError, json.JSONDecodeError):
                pass
            self._save()
    def _save(self):
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"utc_day": self.day, "requests": self.count}, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
    def reserve(self):
        if self.count >= self.cap:
            raise DailyLimitReached(f"Daily API request allowance exhausted ({self.count}/{self.cap}).")
        self.count += 1
        self._save()

class DailyLimitReached(RuntimeError):
    pass

def _logger(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=path, level=logging.INFO, encoding="utf-8",
                        format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger("wos")

def _atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)

def _query_with_types(query: str, include: list[str]) -> str:
    # Early Access is a publication state of an Article in WoS DT; include it by
    # retaining Article records and normalize the state separately.
    types = [x for x in include if x.casefold() != "early access"]
    return f"({query}) AND {doc_type_filter(types)}" if types else query

def _raw_save(raw_dir: Path, relative: str, data: dict):
    path = raw_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

def _hits(data: dict) -> tuple[list[dict], int | None]:
    meta = data.get("metadata") or {}
    hits = data.get("hits")
    if hits is None:  # Expanded API envelope
        qr = data.get("QueryResult") or {}
        records = ((qr.get("Records") or {}).get("REC")) or []
        if isinstance(records, dict): records = [records]
        return records, int(qr.get("RecordsFound", len(records)))
    return hits, int(meta["total"]) if meta.get("total") is not None else None

def _sample_summary(doc: dict) -> dict:
    source = doc.get("source") or {}
    keywords = doc.get("keywords") or {}
    names = doc.get("names") or {}
    return {
        "title": doc.get("title"),
        "abstract": doc.get("abstract") or (doc.get("summary") or {}).get("abstract"),
        "year": source.get("publishYear") or doc.get("publicationYear"),
        "journal": source.get("sourceTitle") or doc.get("journal"),
        "wos_categories": doc.get("wosCategories") or doc.get("categories"),
        "keywords": (keywords.get("authorKeywords") or []) + (keywords.get("keywordsPlus") or []),
    }

def dry_run(client: WoSClient, cfg: dict, ledger: RequestLedger) -> dict:
    logger = logging.getLogger("wos")
    raw_dir = Path(cfg["paths"]["raw_dir"]) / "dry_run"
    include = cfg["collection"]["document_types"]["include"]
    queries = [CORE_QUERY] + CROSS_QUERIES
    results = []
    core_categories_supported = None
    for spec in queries:
        q = query_for_api(spec, client.api_type)
        q = _query_with_types(q, include)
        samples, total, status, error = [], None, "ok", None
        try:
            data = client.get(q, page=1, limit=50)
            _raw_save(raw_dir, f"{spec.query_id}.json", data)
            hits, total = _hits(data)
            selected = random.sample(hits, min(len(hits), cfg["collection"]["dry_run_samples_per_query"]))
            samples = [_sample_summary(x) for x in selected]
            if spec.stage == "core":
                # The live response may accept an undocumented WC tag while
                # silently interpreting it as topic text; do not call it a
                # category search on Starter, whose official supported-tag list
                # excludes WC/SU. The samples/count remain diagnostic only.
                core_categories_supported = client.api_type != "starter"
                if client.api_type == "starter": status = "unverified_field_tag"
        except Exception as exc:
            status, error = "unsupported_or_error", str(exc)
            if spec.stage == "core": core_categories_supported = False
        result = {"query_id": spec.query_id, "label": spec.label, "stage": spec.stage,
                  "query": q, "status": status, "total_hits": total, "error": error, "samples": samples}
        results.append(result)
        logger.info("dry_run query=%s status=%s total=%s", spec.query_id, status, total)
        print(f"\n[{spec.stage}] {spec.label}: status={status}, total={total if total is not None else 'unavailable'}")
        if error: print(f"  API response: {error}")
        for sample in samples:
            print("  sample: " + json.dumps(sample, ensure_ascii=False))

    fallback = None
    if client.api_type == "starter" and core_categories_supported is False:
        q = _query_with_types(CORE_STARTER_PROXY, include)
        try:
            data = client.get(q, page=1, limit=50)
            _raw_save(raw_dir, "core_starter_proxy.json", data)
            hits, total = _hits(data)
            selected = random.sample(hits, min(len(hits), cfg["collection"]["dry_run_samples_per_query"]))
            fallback = {"query": q, "total_hits": total,
                        "samples": [_sample_summary(x) for x in selected]}
            print(f"\n[core proxy] Starter keyword proxy: total={total}")
            for sample in fallback["samples"]: print("  sample: " + json.dumps(sample, ensure_ascii=False))
        except Exception as exc:
            fallback = {"query": q, "error": str(exc)}
            print(f"\n[core proxy] error: {exc}")
    # Check recent per-year hit counts; Starter pagination is capped, so the
    # formal collector must not start a query-year task over 50k records.
    year_checks = []
    check_specs = list(CROSS_QUERIES)
    if fallback:
        from dataclasses import replace
        check_specs.insert(0, replace(CORE_QUERY, query_id="core_starter_proxy", label="Core keyword proxy",
                                      starter_query=CORE_STARTER_PROXY, expanded_query=CORE_STARTER_PROXY))
    for spec in check_specs:
        base = _query_with_types(query_for_api(spec, client.api_type), include)
        for year in (cfg["collection"]["end_year"], cfg["collection"]["end_year"] - 1):
            query = f"({base}) AND PY={year}"
            try:
                data = client.get(query, page=1, limit=1)
                _raw_save(raw_dir, f"annual/{spec.query_id}_{year}.json", data)
                _, total = _hits(data)
                result = {"query_id": spec.query_id, "year": year, "total_hits": total,
                          "within_starter_page_cap": total is not None and total <= 50000}
            except Exception as exc:
                result = {"query_id": spec.query_id, "year": year, "error": str(exc),
                          "within_starter_page_cap": False}
            year_checks.append(result)
            print(f"[annual check] {spec.query_id} {year}: hits={result.get('total_hits', 'error')}; <=50000={result['within_starter_page_cap']}")
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "api_type": client.api_type,
        "core_category_search_supported": core_categories_supported, "core_starter_proxy": fallback,
        "queries": results, "annual_query_checks": year_checks, "api_requests_today": ledger.count,
        "notes": ["Starter API's published supported query tags omit WC/SU; the WC diagnostic returned results but is treated as unverified and not used for collection.",
                  "Starter responses do not include abstracts, Web of Science categories, research areas, and many Expanded API metadata fields."] if client.api_type == "starter" else [],
    }
    _atomic_json(Path(cfg["paths"]["summary"]).with_name("wos_dry_run_summary.json"), summary)
    return summary

def _init_db(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS papers (paper_id INTEGER PRIMARY KEY, year INTEGER, title_prefix TEXT, row_json TEXT NOT NULL)")
    db.execute("CREATE INDEX IF NOT EXISTS ix_papers_year_prefix ON papers(year,title_prefix)")
    db.execute("CREATE TABLE IF NOT EXISTS aliases (key TEXT PRIMARY KEY, paper_id INTEGER NOT NULL)")
    db.execute("CREATE INDEX IF NOT EXISTS ix_alias_paper ON aliases(paper_id)")
    db.execute("CREATE TABLE IF NOT EXISTS stats (name TEXT PRIMARY KEY, value INTEGER NOT NULL)")
    return db

def _merge_row(old: dict, new: dict) -> dict:
    merged = dict(old)
    for key, value in new.items():
        if key == "source_query":
            merged[key] = sorted(set((old.get(key) or []) + (value or [])))
        elif merged.get(key) in (None, "", []) and value not in (None, "", []):
            merged[key] = value
    return merged

def _store_record(db: sqlite3.Connection, row: dict, fuzzy_threshold: float) -> bool:
    keys = []
    if row.get("wos_id"): keys.append("UT:" + str(row["wos_id"]).strip().casefold())
    if row.get("doi_normalized"): keys.append("DOI:" + row["doi_normalized"])
    if row.get("title_normalized") and row.get("year_normalized"):
        keys.append(f"TY:{row['title_normalized']}:{row['year_normalized']}")
    found = []
    for key in keys:
        hit = db.execute("SELECT paper_id FROM aliases WHERE key=?", (key,)).fetchone()
        if hit and hit[0] not in found: found.append(hit[0])
    if not found and row.get("title_normalized") and row.get("year_normalized"):
        prefix = row["title_normalized"][:6]
        candidates = db.execute("SELECT paper_id,row_json FROM papers WHERE year=? AND title_prefix=? LIMIT 500",
                                (row["year_normalized"], prefix)).fetchall()
        for pid, existing_json in candidates:
            existing = json.loads(existing_json)
            if fuzzy_title_score(row.get("title"), existing.get("title")) >= fuzzy_threshold:
                found.append(pid)
                break
    if found:
        target = found[0]
        old_json = db.execute("SELECT row_json FROM papers WHERE paper_id=?", (target,)).fetchone()[0]
        merged = _merge_row(json.loads(old_json), row)
        for redundant in found[1:]:
            other_json = db.execute("SELECT row_json FROM papers WHERE paper_id=?", (redundant,)).fetchone()
            if other_json:
                merged = _merge_row(merged, json.loads(other_json[0]))
                db.execute("UPDATE aliases SET paper_id=? WHERE paper_id=?", (target, redundant))
                db.execute("DELETE FROM papers WHERE paper_id=?", (redundant,))
        db.execute("UPDATE papers SET row_json=? WHERE paper_id=?", (json.dumps(merged, ensure_ascii=False), target))
        for key in keys: db.execute("INSERT OR REPLACE INTO aliases(key,paper_id) VALUES(?,?)", (key,target))
        return False
    cur = db.execute("INSERT INTO papers(year,title_prefix,row_json) VALUES(?,?,?)",
                     (row.get("year_normalized"), (row.get("title_normalized") or "")[:6], json.dumps(row, ensure_ascii=False)))
    pid = cur.lastrowid
    for key in keys: db.execute("INSERT OR IGNORE INTO aliases(key,paper_id) VALUES(?,?)", (key,pid))
    return True

def _export(db: sqlite3.Connection, parquet_path: Path, jsonl_path: Path, batch_size: int = 1000):
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the previous corpus/schema recoverable before replacing the final paths.
    for target in (parquet_path, jsonl_path):
        legacy = target.with_name(target.stem + "_legacy_20261002" + target.suffix)
        if target.exists() and not legacy.exists():
            shutil.copy2(target, legacy)
    ptmp, jtmp = parquet_path.with_suffix(".parquet.tmp"), jsonl_path.with_suffix(".jsonl.tmp")
    writer = None
    count = 0
    with jtmp.open("w", encoding="utf-8") as jf:
        cursor = db.execute("SELECT row_json FROM papers ORDER BY paper_id")
        while batch := cursor.fetchmany(batch_size):
            records = [json.loads(item[0]) for item in batch]
            for row in records: jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            records = [_arrow_record(row) for row in records]
            table = pa.Table.from_pylist(records, schema=_arrow_schema())
            if writer is None: writer = pq.ParquetWriter(ptmp, _arrow_schema(), compression="zstd")
            writer.write_table(table)
            count += len(records)
    if writer: writer.close()
    else:
        pq.write_table(pa.Table.from_pylist([], schema=_arrow_schema()), ptmp, compression="zstd")
    os.replace(ptmp, parquet_path)
    os.replace(jtmp, jsonl_path)
    return count

def _arrow_schema():
    fields = []
    for name in SCHEMA_FIELDS:
        if name in LIST_FIELDS: typ = pa.list_(pa.string())
        elif name in BOOL_FIELDS: typ = pa.bool_()
        elif name in INT_FIELDS: typ = pa.int64()
        else: typ = pa.string()
        fields.append(pa.field(name, typ))
    return pa.schema(fields)

def _arrow_record(row: dict) -> dict:
    out = {}
    for key in SCHEMA_FIELDS:
        value = row.get(key)
        if value is None:
            out[key] = None
        elif key in LIST_FIELDS:
            vals = value if isinstance(value, list) else [value]
            out[key] = [x if isinstance(x, str) else json.dumps(x, ensure_ascii=False) for x in vals]
        elif key in BOOL_FIELDS:
            out[key] = bool(value) if isinstance(value, bool) else (str(value).casefold() in {"true", "yes", "1"} if value is not None else None)
        elif key in INT_FIELDS:
            try: out[key] = int(value)
            except (TypeError, ValueError): out[key] = None
        elif isinstance(value, (dict, list)) or key in OPAQUE_FIELDS:
            out[key] = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        else:
            out[key] = str(value)
    return out

def _save_checkpoint(path: Path, query_id: str, year: int, page: int, fetched: int, total: int | None,
                     counts: dict):
    try: data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): data = {"version": 1, "tasks": {}, "cumulative": {}}
    data.setdefault("tasks", {})[f"{query_id}:{year}"] = {"page": page, "fetched": fetched, "total": total, "updated_at_utc": datetime.now(timezone.utc).isoformat()}
    data["current_query"], data["current_year"], data["page"] = query_id, year, page
    data["cumulative"] = counts
    _atomic_json(path, data)

def collect(client: WoSClient, cfg: dict, ledger: RequestLedger, dry: dict):
    paths = cfg["paths"]
    checkpoint_path = Path(paths["checkpoint"])
    db_path = Path(paths["staging_dir"]) / "wos_staging.sqlite3"
    raw_root = Path(paths["raw_dir"])
    logger = logging.getLogger("wos")
    db = _init_db(db_path)
    include = cfg["collection"]["document_types"]["include"]
    exclude = {x.casefold() for x in cfg["collection"]["document_types"]["exclude"]}
    specs = []
    # Category-level Core is used only when confirmed by dry-run. Otherwise we
    # collect an explicitly named Starter keyword proxy and report the limitation.
    if dry.get("core_category_search_supported"):
        specs.append(CORE_QUERY)
    elif client.api_type == "starter":
        from dataclasses import replace
        specs.append(replace(CORE_QUERY, query_id="core_starter_proxy", label="Core keyword proxy (Starter limitation)",
                             starter_query=CORE_STARTER_PROXY, expanded_query=CORE_STARTER_PROXY))
    specs.extend(CROSS_QUERIES)
    start, end = cfg["collection"]["start_year"], cfg["collection"]["end_year"]
    counters = {"core_raw_records": 0, "cross_raw_records": 0, "total_raw_records": 0,
                "duplicates_removed": 0, "excluded_document_types": 0, "new_unique_records": 0}
    if checkpoint_path.exists():
        try:
            counters.update(json.loads(checkpoint_path.read_text(encoding="utf-8")).get("cumulative", {}))
        except Exception: pass
    if checkpoint_path.exists():
        legacy_checkpoint = checkpoint_path.with_name("wos_checkpoint_legacy_20261002.json")
        if not legacy_checkpoint.exists() and checkpoint_path.stat().st_size:
            shutil.copy2(checkpoint_path, legacy_checkpoint)
    try:
        for spec in specs:
            base = _query_with_types(query_for_api(spec, client.api_type), include)
            for year in range(end, start - 1, -1):
                task_id = f"{spec.query_id}:{year}"
                try: state = json.loads(checkpoint_path.read_text(encoding="utf-8")).get("tasks", {}).get(task_id, {})
                except Exception: state = {}
                if state.get("total") is not None and state.get("fetched", 0) >= state["total"]:
                    continue
                if state.get("total") is None:
                    query = f"({base}) AND PY={year}"
                    data = client.get(query, page=1, limit=1)
                    _raw_save(raw_root, f"{spec.query_id}/{year}/page_000001.json", data)
                    _, total = _hits(data)
                    if total is None: raise RuntimeError("API response has no total hit count; checkpoint cannot be made safe.")
                    page, fetched = 0, 0
                    _save_checkpoint(checkpoint_path, spec.query_id, year, page, fetched, total, counters)
                else:
                    total, page, fetched = state["total"], state["page"], state["fetched"]
                max_page = (total + cfg["api"]["page_size"] - 1) // cfg["api"]["page_size"]
                for next_page in range(page + 1, max_page + 1):
                    query = f"({base}) AND PY={year}"
                    data = client.get(query, page=next_page, limit=cfg["api"]["page_size"])
                    hits, _ = _hits(data)
                    rel = f"{spec.query_id}/{year}/page_{next_page:06d}.json"
                    _raw_save(raw_root, rel, data)
                    normalized_path = Path(paths["staging_dir"]) / "normalized" / spec.query_id / str(year) / f"page_{next_page:06d}.jsonl"
                    normalized_path.parent.mkdir(parents=True, exist_ok=True)
                    new_count = duplicate_count = excluded_count = 0
                    with normalized_path.open("w", encoding="utf-8") as out:
                        for doc in hits:
                            row = normalize_document(doc, "core" if spec.stage == "core" else spec.label)
                            types = {str(x).strip().casefold() for x in (row.get("document_type") or [])}
                            is_early = row.get("early_access") is True
                            if types & exclude or (types and not (types & {x.casefold() for x in include} or (is_early and "early access" in {x.casefold() for x in include}))):
                                excluded_count += 1
                                continue
                            out.write(json.dumps(row, ensure_ascii=False) + "\n")
                            if _store_record(db, row, cfg["collection"]["fuzzy_title_review_threshold"]): new_count += 1
                            else: duplicate_count += 1
                    db.commit()
                    fetched += len(hits)
                    counters["total_raw_records"] += len(hits)
                    counters["core_raw_records" if spec.stage == "core" else "cross_raw_records"] += len(hits)
                    counters["duplicates_removed"] += duplicate_count
                    counters["excluded_document_types"] += excluded_count
                    counters["new_unique_records"] += new_count
                    _save_checkpoint(checkpoint_path, spec.query_id, year, next_page, fetched, total, counters)
                    logger.info("page saved query=%s year=%s page=%s hits=%s new=%s duplicates=%s",
                                spec.query_id, year, next_page, len(hits), new_count, duplicate_count)
                    if next_page == 1 or next_page % 20 == 0:
                        print(f"{spec.query_id} {year}: page {next_page}/{max_page}, raw {counters['total_raw_records']}, unique+ {counters['new_unique_records']}, API today {ledger.count}/{ledger.cap}")
        count = _export(db, Path(paths["output_parquet"]), Path(paths["output_jsonl"]))
        summary = _make_summary(db, counters, ledger, client, count, dry)
        _atomic_json(Path(paths["summary"]), summary)
        return summary
    except (DailyLimitReached, KeyboardInterrupt) as exc:
        print(f"\nStopping safely: {exc}")
        db.commit()
        count = _export(db, Path(paths["output_parquet"]), Path(paths["output_jsonl"]))
        summary = _make_summary(db, counters, ledger, client, count, dry)
        summary["status"] = "paused_at_checkpoint"
        _atomic_json(Path(paths["summary"]), summary)
        return summary
    finally:
        db.close()

def _make_summary(db, counters: dict, ledger: RequestLedger, client: WoSClient, count: int, dry: dict) -> dict:
    db.row_factory = sqlite3.Row
    fields = ["abstract", "doi", "author_keywords", "affiliations", "citation_count", "year_normalized",
              "document_type_normalized", "wos_categories", "research_areas", "journal", "source_query"]
    freqs = {name: {} for name in fields if name not in {"abstract", "doi", "author_keywords", "affiliations", "citation_count"}}
    present = {name: 0 for name in fields}
    keyword_records = 0
    min_year, max_year = None, None
    for item in db.execute("SELECT row_json FROM papers"):
        r = json.loads(item[0])
        if r.get("author_keywords") or r.get("keywords_plus"):
            keyword_records += 1
        for field in fields:
            value = r.get(field)
            if value not in (None, "", []): present[field] += 1
            if field in freqs:
                vals = value if isinstance(value, list) else ([value] if value not in (None, "") else [])
                for v in vals:
                    k = str(v); freqs[field][k] = freqs[field].get(k, 0) + 1
        y = r.get("year_normalized")
        if y:
            min_year = y if min_year is None else min(min_year, y)
            max_year = y if max_year is None else max(max_year, y)
    def freq(field): return dict(sorted(freqs.get(field, {}).items(), key=lambda kv: (-kv[1], kv[0])))
    return {
        "status": "complete", "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_type": client.api_type, "core_category_search_supported": dry.get("core_category_search_supported"),
        "core_query_raw_records": counters.get("core_raw_records", 0),
        "cross_query_raw_records": counters.get("cross_raw_records", 0),
        "total_raw_records": counters.get("total_raw_records", 0),
        "duplicates_removed": counters.get("duplicates_removed", 0), "unique_papers": count,
        "records_with_abstract": present["abstract"], "records_with_doi": present["doi"],
        "records_with_keywords": keyword_records,
        "records_with_affiliation": present["affiliations"], "records_with_citation_count": present["citation_count"],
        "year_range": [min_year, max_year] if min_year is not None else None,
        "documents_by_year": freq("year_normalized"), "documents_by_document_type": freq("document_type_normalized"),
        "documents_by_wos_category": freq("wos_categories"), "documents_by_research_area": freq("research_areas"),
        "documents_by_journal": freq("journal"), "field_completion_rate": {
            "abstract_completion_rate": round(present["abstract"] / count, 6) if count else 0.0,
            "doi_completion_rate": round(present["doi"] / count, 6) if count else 0.0,
            "keyword_completion_rate": round(keyword_records / count, 6) if count else 0.0,
            "affiliation_completion_rate": round(present["affiliations"] / count, 6) if count else 0.0,
            "citation_count_completion_rate": round(present["citation_count"] / count, 6) if count else 0.0},
        "api_requests_today": ledger.count, "api_daily_cap": ledger.cap,
        "source_query_counts": freq("source_query"),
        "dry_run_summary": str(Path("database/wos_dry_run_summary.json")),
        "limitations": ["On WoS Starter, categories/research areas and abstracts are not exposed; source category core retrieval cannot be claimed. A Starter keyword proxy is used unless an Expanded API dry-run confirms category-field support."] if client.api_type == "starter" else [],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="test API, query totals, and sample records only")
    parser.add_argument("--collect", action="store_true", help="resume full collection after dry-run")
    args = parser.parse_args()
    cfg = config_load()
    logger = _logger(cfg["paths"]["log"])
    ledger = RequestLedger(str(Path(cfg["paths"]["checkpoint"]).with_name("wos_request_ledger.json")), cfg["api"]["max_daily_requests"])
    client = WoSClient(cfg["api"]["type"], cfg["api"]["timeout_seconds"], cfg["api"]["max_retries"],
                       cfg["api"]["requests_per_second"], cfg["api"]["database"], ledger.reserve)
    if args.dry_run:
        result = dry_run(client, cfg, ledger)
        print(f"\nDry run saved. API type={result['api_type']}; requests today={ledger.count}/{ledger.cap}.")
        print("No corpus rows were downloaded by dry-run. Review query samples before starting with --collect.")
    elif args.collect:
        dry_path = Path(cfg["paths"]["summary"]).with_name("wos_dry_run_summary.json")
        if not dry_path.exists(): raise SystemExit("Run --dry-run first; formal collection is gated on dry-run review.")
        dry = json.loads(dry_path.read_text(encoding="utf-8"))
        summary = collect(client, cfg, ledger, dry)
        print(json.dumps({k: summary[k] for k in ("status", "unique_papers", "total_raw_records", "api_requests_today")}, ensure_ascii=False))
    else:
        parser.error("Choose --dry-run or --collect")

if __name__ == "__main__":
    main()

