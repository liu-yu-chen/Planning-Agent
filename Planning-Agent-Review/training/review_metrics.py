"""Transparent citation and lexical diagnostics; no automatic expert judgement."""
from __future__ import annotations

import math
import re
from collections import Counter

DOI_PATTERN = re.compile(r"\[DOI:\s*", re.I)


def citation_spans(text: str) -> list[tuple[int, int, str]]:
    """Parse balanced citation brackets, including legitimate brackets in old DOIs."""
    result = []
    consumed = 0
    for match in DOI_PATTERN.finditer(text):
        if match.start() < consumed:
            continue
        depth = 1
        for position in range(match.end(), len(text)):
            if text[position] == '[':
                depth += 1
            elif text[position] == ']':
                depth -= 1
                if depth == 0:
                    result.append((match.start(), position + 1, text[match.end():position]))
                    consumed = position + 1
                    break
            if text[position] == '\n':
                break
    return result


def remove_citations(text: str) -> str:
    chunks, start = [], 0
    for left, right, _ in citation_spans(text):
        chunks.append(text[start:left])
        start = right
    chunks.append(text[start:])
    return ''.join(chunks)


def normalize_doi(value: str) -> str:
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value.strip().lower()).rstrip(".,;")


def cited_dois(text: str) -> set[str]:
    return {normalize_doi(d) for _, _, d in citation_spans(text)}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", remove_citations(text).lower())


def rouge_l_f1(answer: str, reference: str) -> float:
    a, b = words(answer), words(reference)
    if not a or not b:
        return 0.0
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b, 1):
            current.append(previous[j-1] + 1 if token == other else max(previous[j], current[-1]))
        previous = current
    common = previous[-1]
    return 2 * common / (len(a) + len(b))


def score_answer(answer: str, reference: str, sources: dict[str, str], expected_dois=None) -> dict:
    allowed = {normalize_doi(d) for d in sources}
    expected = {normalize_doi(d) for d in expected_dois} if expected_dois is not None else cited_dois(reference)
    cited = cited_dois(answer)
    valid = cited & allowed
    precision = len(valid) / len(cited) if cited else None
    recall = len(cited & expected) / len(expected) if expected else None
    citation_f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else 0.0 if expected else None
    evidence = {normalize_doi(k): re.sub(r"\s+", " ", v).strip().lower() for k, v in sources.items()}
    quote_pairs = []
    previous = 0
    spans = citation_spans(answer)
    for left, right, doi in spans:
        segment = answer[previous:left].rstrip()
        quote = re.search(r'["“]([^\n]{20,})["”]\s*$', segment)
        if quote:
            quote_pairs.append((quote.group(1), doi))
        previous = right
    supported = 0
    for quote, doi in quote_pairs:
        exact = re.sub(r"\s+", " ", quote).strip().lower()
        supported += exact in evidence.get(normalize_doi(doi), "")
    tokens = words(answer)
    english_markers = {"the", "and", "of", "to", "in", "for", "with", "from", "these", "this", "is", "are", "not", "study", "paper"}
    return {"citation_precision": precision, "reference_citation_recall": recall,
        "citation_f1": citation_f1, "invalid_cited_dois": sorted(cited - allowed),
        "invalid_citation_count": len(cited - allowed), "source_coverage": len(valid)/len(allowed) if allowed else None,
        "unclosed_citation_count": len(DOI_PATTERN.findall(answer)) - len(spans),
        "reference_requires_citations": bool(expected), "cited_source_count": len(valid),
        "exact_quote_citation_support": supported / len(quote_pairs) if quote_pairs else None,
        "quoted_claims_checked": len(quote_pairs), "unsupported_quoted_claims": len(quote_pairs)-supported,
        "rouge_l_f1": rouge_l_f1(answer, reference) if reference else None,
        "cjk_free": not bool(re.search(r"[\u3400-\u9fff]", answer)),
        "english_marker_check": any(t in english_markers for t in tokens),
        "reasoning_preface_detected": bool(re.search(r"^(?:okay|hmm|let me|i need to|first,? i)", answer.strip(), re.I)),
        "limits": "Citation validity checks source membership; exact-quote support checks verbatim evidence only. ROUGE-L uses lowercase alphanumeric tokens without stemming. These are not semantic faithfulness or expert synthesis scores."}


def aggregate_answers(results: list[dict]) -> dict:
    scores = [r["metrics"] for r in results]
    fields = ("citation_precision", "reference_citation_recall", "citation_f1", "source_coverage",
              "exact_quote_citation_support", "rouge_l_f1", "generation_seconds", "tokens_per_second")
    averages, denominators = {}, {}
    for field in fields:
        values = [s[field] for s in scores if s.get(field) is not None and math.isfinite(s[field])]
        averages[field] = sum(values)/len(values) if values else None
        denominators[field] = len(values)
    return {"questions": len(results), "mean": averages, "metric_sample_counts": denominators,
        "invalid_citation_count": sum(s.get("invalid_citation_count", 0) for s in scores),
        "unclosed_citation_count": sum(s.get("unclosed_citation_count", 0) for s in scores),
        "unsupported_quoted_claims": sum(s.get("unsupported_quoted_claims", 0) for s in scores),
        "quoted_claims_checked": sum(s.get("quoted_claims_checked", 0) for s in scores),
        "generation_limit_rate": sum(bool(s.get("generation_limit_reached")) for s in scores)/len(scores) if scores else None,
        "reasoning_preface_rate": sum(bool(s.get("reasoning_preface_detected")) for s in scores)/len(scores) if scores else None,
        "cjk_free_rate": sum(bool(s.get("cjk_free")) for s in scores)/len(scores) if scores else None,
        "domains": dict(Counter(r.get("domain", "unknown") for r in results)),
        "limits": "Lexical/citation diagnostics and generation speed. No expert relevance, causal validity or semantic entailment labels are inferred."}
