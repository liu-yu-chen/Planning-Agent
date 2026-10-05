"""Deduplication keys, ordered by WoS UT, DOI, then normalized title + year."""
from __future__ import annotations
from difflib import SequenceMatcher
from .wos_normalize import normalize_doi, text_norm

def exact_key(row: dict) -> str:
    uid = row.get("wos_id")
    if uid:
        return "UT:" + str(uid).strip().casefold()
    doi = normalize_doi(row.get("doi"))
    if doi:
        return "DOI:" + doi
    title, year = text_norm(row.get("title")), row.get("publication_year")
    if title and year:
        return f"TY:{title}:{year}"
    return ""

def fuzzy_title_score(a: str | None, b: str | None) -> float:
    a, b = text_norm(a), text_norm(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()
