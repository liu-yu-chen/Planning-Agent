"""Canonical WoS corpus schema and field defaults."""
from __future__ import annotations

SCHEMA_FIELDS = [
    "wos_id", "doi", "title", "abstract", "authors", "author_full_names",
    "author_ids", "orcid", "affiliations", "addresses", "corresponding_author",
    "publication_year", "publication_date", "journal", "issn", "eissn",
    "document_type", "language", "author_keywords", "keywords_plus",
    "wos_categories", "research_areas", "citation_count", "references_count",
    "publisher", "volume", "issue", "pages", "article_number", "early_access",
    "open_access", "funding_agencies", "funding_text", "source_query",
    "cited_references", "author_affiliation_mapping", "corresponding_address",
    "researcher_id", "funding_details", "conference_information", "subject_categories",
    "country", "province", "city", "district", "study_area_raw", "spatial_scale",
    "urban_rural_type", "planning_domain", "primary_topic", "secondary_topics",
    "methodology_primary", "methodology_secondary", "data_source", "data_type",
    "study_period", "sample_size", "research_object", "planning_problem", "policy_type",
    "spatial_method", "statistical_method", "machine_learning_method", "gis_used",
    "remote_sensing_used", "survey_used", "interview_used", "simulation_used",
    "agent_based_model_used", "llm_used", "doi_normalized", "title_normalized",
    "authors_normalized", "journal_normalized", "year_normalized",
    "author_keywords_normalized", "document_type_normalized", "wos_categories_normalized",
    "raw_identifiers", "raw_names", "raw_source", "raw_keywords", "raw_types",
]

LIST_FIELDS = {
    "author_full_names", "author_ids", "orcid", "affiliations", "addresses",
    "author_keywords", "keywords_plus", "wos_categories", "research_areas",
    "funding_agencies", "source_query", "cited_references", "subject_categories",
    "secondary_topics", "raw_types", "author_keywords_normalized", "wos_categories_normalized",
}
BOOL_FIELDS = {"early_access", "open_access", "gis_used", "remote_sensing_used", "survey_used",
               "interview_used", "simulation_used", "agent_based_model_used", "llm_used"}
INT_FIELDS = {"publication_year", "year_normalized", "citation_count", "references_count"}
OPAQUE_FIELDS = {"affiliations", "corresponding_author", "funding_text", "cited_references",
                 "author_affiliation_mapping", "corresponding_address", "researcher_id",
                 "funding_details", "conference_information", "raw_identifiers", "raw_names",
                 "raw_source", "raw_keywords"}

def empty_record() -> dict:
    row = {name: None for name in SCHEMA_FIELDS}
    for name in LIST_FIELDS:
        row[name] = []
    for name in BOOL_FIELDS:
        row[name] = None
    return row
