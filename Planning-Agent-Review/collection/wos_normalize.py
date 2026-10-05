"""WoS payload normalization. Original API fragments remain alongside standardized columns."""
from __future__ import annotations
import re
import unicodedata
from typing import Any
from .wos_schema import empty_record

def text_norm(value: Any) -> str | None:
    if value is None:
        return None
    s = unicodedata.normalize("NFKC", str(value)).casefold()
    s = re.sub(r"https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"[^\w]+", " ", s, flags=re.UNICODE)
    return " ".join(s.split()) or None

def normalize_doi(value: Any) -> str | None:
    if not value:
        return None
    s = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    s = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", s)
    return s.rstrip(" .;,)") or None

def normalize_author(value: Any) -> str | None:
    return text_norm(value)

def _list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return [x for x in value if x not in (None, "")]
    if isinstance(value, str):
        return [x.strip() for x in re.split(r"\s*;\s*", value) if x.strip()]
    return [value]

def _author_parts(doc: dict) -> tuple[list[str], list[str], list[str], list[str]]:
    names = doc.get("names") or {}
    authors = names.get("authors") or doc.get("authors") or []
    full_names, author_ids, orcids, affiliations = [], [], [], []
    if isinstance(authors, list):
        for a in authors:
            if isinstance(a, str):
                full_names.append(a)
                continue
            if not isinstance(a, dict):
                continue
            full_names.extend(_list(a.get("displayName") or a.get("fullName") or a.get("name")))
            author_ids.extend(_list(a.get("researcherId") or a.get("researcherID") or a.get("authorId")))
            orcids.extend(_list(a.get("orcid")))
            affiliations.extend(_list(a.get("affiliation") or a.get("affiliations")))
    return full_names, author_ids, orcids, affiliations

def normalize_document(doc: dict, source_query: str | list[str]) -> dict:
    row = empty_record()
    full_names, author_ids, orcids, affiliations = _author_parts(doc)
    ids = doc.get("identifiers") or {}
    source = doc.get("source") or {}
    keywords = doc.get("keywords") or {}
    types = _list(doc.get("types") or doc.get("documentType"))
    author_kw = _list(keywords.get("authorKeywords") or keywords.get("author") or doc.get("authorKeywords"))
    plus_kw = _list(keywords.get("keywordsPlus") or keywords.get("plus") or doc.get("keywordsPlus"))
    year = source.get("publishYear") or doc.get("publicationYear") or doc.get("year")
    try:
        year = int(str(year)[:4]) if year else None
    except (ValueError, TypeError):
        year = None
    date = source.get("publishDate") or doc.get("publicationDate")
    citations = doc.get("citations") or []
    source_queries = [source_query] if isinstance(source_query, str) else list(source_query or [])
    title = doc.get("title") or doc.get("titles", {}).get("item", {}).get("title")
    doi = ids.get("doi") or doc.get("doi")
    journal = source.get("sourceTitle") or doc.get("journal")
    row.update({
        "wos_id": doc.get("uid") or doc.get("wos_id") or doc.get("UT"), "doi": doi,
        "title": title, "abstract": doc.get("abstract") or doc.get("summary", {}).get("abstract"),
        "authors": "; ".join(full_names) if full_names else None,
        "author_full_names": full_names, "author_ids": author_ids, "orcid": orcids,
        "affiliations": affiliations, "addresses": _list(doc.get("addresses")),
        "corresponding_author": doc.get("correspondingAuthor"), "publication_year": year,
        "publication_date": date, "journal": journal, "issn": ids.get("issn") or doc.get("issn"),
        "eissn": ids.get("eissn") or doc.get("eissn"), "document_type": types,
        "language": doc.get("language") or doc.get("languages"),
        "author_keywords": author_kw, "keywords_plus": plus_kw,
        "wos_categories": _list(doc.get("wosCategories") or doc.get("categories")),
        "research_areas": _list(doc.get("researchAreas")),
        "citation_count": ((citations[0] or {}).get("count") if citations and isinstance(citations[0], dict) else doc.get("citationCount")),
        "references_count": doc.get("referencesCount"), "publisher": source.get("publisher") or doc.get("publisher"),
        "volume": source.get("volume") or doc.get("volume"), "issue": source.get("issue") or doc.get("issue"),
        "pages": source.get("pages") or doc.get("pages"), "article_number": doc.get("articleNumber"),
        "early_access": bool(doc.get("earlyAccess") or any("early access" in str(t).casefold() for t in types)
                            or any("early access" in str(t).casefold() for t in _list(doc.get("sourceTypes")))),
        "open_access": doc.get("openAccess"), "funding_agencies": _list(doc.get("fundingAgencies")),
        "funding_text": doc.get("fundingText"), "source_query": sorted(set(source_queries)),
        "cited_references": _list(doc.get("citedReferences")),
        "author_affiliation_mapping": doc.get("authorAffiliationMapping"),
        "corresponding_address": doc.get("correspondingAddress"),
        "researcher_id": author_ids, "funding_details": doc.get("fundingDetails"),
        "conference_information": doc.get("conferenceInformation"),
        "subject_categories": _list(doc.get("subjectCategories")),
        "doi_normalized": normalize_doi(doi), "title_normalized": text_norm(title),
        "authors_normalized": [normalize_author(a) for a in full_names if normalize_author(a)],
        "journal_normalized": text_norm(journal), "year_normalized": year,
        "author_keywords_normalized": sorted({text_norm(k) for k in author_kw if text_norm(k)}),
        "document_type_normalized": sorted({text_norm(t) for t in types if text_norm(t)}),
        "wos_categories_normalized": sorted({text_norm(t) for t in row["wos_categories"] if text_norm(t)}),
        "raw_identifiers": ids, "raw_names": doc.get("names"), "raw_source": source,
        "raw_keywords": keywords, "raw_types": types,
    })
    return row
