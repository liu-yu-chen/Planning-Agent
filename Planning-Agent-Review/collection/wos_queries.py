"""Generic query-plan interface. Replace the placeholders with an approved search strategy."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Query:
    query_id: str
    label: str
    starter_query: str
    expanded_query: str
    stage: str

CORE_QUERY = Query("core", "REPLACE_WITH_SCOPE_LABEL", "REPLACE_WITH_APPROVED_QUERY", "REPLACE_WITH_APPROVED_QUERY", "core")
CROSS_QUERIES: list[Query] = []

def query_for_api(query: Query, api_type: str) -> str:
    return query.expanded_query if api_type == "expanded" else query.starter_query

def doc_type_filter(include: list[str]) -> str:
    values = " OR ".join(f'"{x}"' if " " in x else x for x in include)
    return f"DT=({values})"
