"""Chat UI for the local LlamaIndex + Ollama + planning-corpus assistant."""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama

DIST_DIR = Path(__file__).resolve().parents[1]
if str(DIST_DIR) not in sys.path:
    sys.path.insert(0, str(DIST_DIR))

from urban_planning_agent.core import DEFAULTS, SYSTEM
from urban_planning_agent.live import OpenAlex
from urban_planning_agent.retriever import Retriever

CURRENT_DATE = date.today().isoformat()


def settings_from_cli() -> dict[str, Any]:
    config_path = Path(os.environ.get("PLANNING_REVIEW_CONFIG", "planning-review-local.json"))
    settings = DEFAULTS.copy()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if config_path.is_file():
        settings.update(json.loads(config_path.read_text(encoding="utf-8")))
        configured_database = Path(str(settings.get("database") or ""))
        package_database = config_path.parent / "data" / "rag.sqlite"
        if not configured_database.is_file() and package_database.is_file():
            settings["database"] = str(package_database)
        if settings.get("output_root") == "D:/agent/results":
            settings["output_root"] = str(config_path.parent / "results")
    return settings


def packaged_ollama_api_key() -> str:
    """Read the hidden credential bundled for this user's distributable."""
    key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if key:
        return key
    credential_file = DIST_DIR / ".ollama_credentials.json"
    try:
        value = json.loads(credential_file.read_text(encoding="utf-8"))
        return str(value.get("OLLAMA_API_KEY") or "").strip()
    except (OSError, ValueError, AttributeError):
        return ""


@st.cache_resource(show_spinner=False)
def get_retriever(path: str) -> Retriever:
    return Retriever(path)


def search_text(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    words = [word for word in text.split() if len(word) > 2]
    stop = {"the", "and", "for", "with", "from", "that", "this", "what", "how",
            "can", "does", "are", "was", "were", "about", "into", "between", "their",
            "study", "studies", "review", "research", "please", "explain", "compare"}
    return " ".join(word for word in words if word.lower() not in stop)[:500]


def retrieve_evidence(retriever: Retriever, question: str, top_k: int, candidate_k: int) -> list[dict[str, Any]]:
    query = search_text(question)
    if not query:
        return []
    result = retriever.retrieve(
        {"id": "chat", "question": question, "search_query": query},
        top_k=top_k,
        candidate_k=candidate_k,
    )
    rows: dict[str, dict[str, Any]] = {}
    for mode in ("graph", "ordinary"):
        for paper in result["modes"].get(mode, []):
            doi = paper.get("doi")
            if doi:
                row = rows.setdefault(doi, paper | {"retrieval_modes": []})
                row["retrieval_modes"].append(mode)
    return list(rows.values())


def evidence_context(papers: list[dict[str, Any]], char_budget: int = 10000) -> str:
    blocks = []
    remaining = char_budget
    for index, paper in enumerate(papers, 1):
        title = str(paper.get("title") or "Untitled")[:400]
        abstract = str(paper.get("abstract") or "")[:600]
        authors = paper.get("authors") or []
        if isinstance(authors, str):
            try:
                authors = json.loads(authors)
            except (ValueError, TypeError):
                authors = [authors]
        author_names = []
        for author in authors if isinstance(authors, list) else []:
            if isinstance(author, dict):
                author = author.get("display_name") or author.get("name") or ""
            if author:
                author_names.append(str(author))
        author_text = ", ".join(author_names[:6]) or "Authors unavailable"
        doi = paper.get("doi") or ""
        year = paper.get("year") or "Unknown year"
        journal = paper.get("journal") or "Unknown journal"
        modes = ", ".join(sorted(set(paper.get("retrieval_modes", []))))
        block = (f"[S{index}] {author_text} ({year}). {title}. {journal}. DOI: {doi}. "
                 f"Retrieved by: {modes}. Abstract: {abstract}")
        if len(block) > remaining:
            if not blocks:
                blocks.append(block[:remaining])
            break
        blocks.append(block)
        remaining -= len(block)
    return "\n\n".join(blocks)


def ollama_web_search(query: str, api_key: str, max_results: int = 5) -> list[dict[str, str]]:
    payload = json.dumps({"query": query[:600], "max_results": max_results}).encode("utf-8")
    request = urllib.request.Request(
        "https://ollama.com/api/web_search",
        data=payload,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        data = json.load(response)
    results = []
    for row in data.get("results", [])[:max_results]:
        if row.get("url"):
            results.append({
                "title": str(row.get("title") or row["url"])[:300],
                "url": str(row["url"]),
                "content": str(row.get("content") or "")[:1800],
            })
    return results


def web_evidence_context(results: list[dict[str, str]], char_budget: int = 6000) -> str:
    blocks = []
    remaining = char_budget
    for index, row in enumerate(results, 1):
        block = (f"[W{index}] {row['title']}\nURL: {row['url']}\nSearch retrieved: {CURRENT_DATE}. "
                 f"The search API did not provide a separate publication/update date.\nSnippet: {row['content']}")
        if len(block) > remaining:
            if not blocks:
                blocks.append(block[:remaining])
            break
        blocks.append(block)
        remaining -= len(block)
    return "\n\n".join(blocks)


def openalex_context(papers: list[dict[str, Any]], char_budget: int = 6500) -> str:
    blocks = []
    remaining = char_budget
    for index, paper in enumerate(papers, 1):
        authors = paper.get("authors") or []
        author_text = ", ".join(str(a) for a in authors[:6]) if isinstance(authors, list) else str(authors)
        block = (f"[O{index}] {author_text or 'Authors unavailable'} ({paper.get('year') or 2026}). "
                 f"{paper.get('title') or 'Untitled'}. "
                 f"{paper.get('journal') or 'Unknown journal'}. DOI: {paper.get('doi')}. "
                 f"Abstract: {str(paper.get('abstract') or '')[:480]}")
        if len(block) > remaining:
            if not blocks:
                blocks.append(block[:remaining])
            break
        blocks.append(block)
        remaining -= len(block)
    return "\n\n".join(blocks)


def search_openalex(query: str, api_key: str, key_env: str, limit: int = 10) -> dict[str, Any]:
    previous = os.environ.get(key_env)
    os.environ[key_env] = api_key
    try:
        return OpenAlex(key_env=key_env).search(
            query, from_year=2026, to_year=2026, limit=limit,
            minimum_citations=0, document_types=["article", "review"],
        )
    finally:
        if previous is None:
            os.environ.pop(key_env, None)
        else:
            os.environ[key_env] = previous


def to_llama_messages(history: list[dict[str, str]], system: str) -> list[ChatMessage]:
    messages = [ChatMessage(role=MessageRole.SYSTEM, content=system)]
    for item in history[-4:]:
        role = MessageRole.USER if item["role"] == "user" else MessageRole.ASSISTANT
        messages.append(ChatMessage(role=role, content=item["content"][-1800:]))
    return messages


def main() -> None:
    settings = settings_from_cli()
    st.set_page_config(page_title="Planning Research Chat", page_icon="📚", layout="wide")
    st.markdown("""
    <style>
      .stApp { background: #f5f7fa; }
      section[data-testid="stSidebar"] { background: #122b43; }
      section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #eef6fb !important; }
      section[data-testid="stSidebar"] code { background: #29465f !important; color: #eef6fb !important; }
      section[data-testid="stSidebar"] button { color: #eef6fb !important; background: #29465f !important; }
      .block-container { max-width: 1050px; padding-top: 2rem; }
      [data-testid="stChatMessage"] { border: 1px solid #e5ebf1; border-radius: 16px; }
      [data-testid="stToolbar"], .stDeployButton { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("📚 Planning Research Chat")
    st.caption("本地语料截至 2025 年 · OpenAlex 检索 2026 年文献 · 每轮强制 Ollama 联网搜索")

    with st.sidebar:
        st.subheader("本地研究助手")
        st.caption(f"当前系统日期：{CURRENT_DATE}")
        st.markdown(f"**模型**  \n`{settings.get('model') or '未配置'}`")
        st.markdown(f"**语料库**  \n`{Path(settings.get('database') or '.').name}`")
        st.divider()
        writing_mode = st.selectbox("写作模式", ["正式文献综述", "研究问答"], index=0)
        if writing_mode == "正式文献综述":
            st.caption("生成连贯的综述草稿；本地语料截至 2025 年，OpenAlex 检索 2026 年记录与摘要，不等同于系统综述或全文审读。")
        top_k = st.slider("每种检索方式取文献数", 2, 8, 6)
        candidate_k = st.slider("图谱扩展候选数", 20, 200, 80, step=20)
        include_evidence = st.toggle("显示检索到的证据", value=True)
        openalex_enabled = st.toggle("OpenAlex · 仅 2026 年", value=True)
        api_key = packaged_ollama_api_key()
        st.caption("每轮提问均强制调用 Ollama 联网搜索；API Key 在后台读取，不在界面显示。")
        openalex_key = st.text_input(
            "OPENALEX_API_KEY",
            value=os.environ.get(str(settings.get("openalex_key_env", "OPENALEX_API_KEY")), ""),
            type="password",
            help="OpenAlex 免费账号的 API key。仅用于本轮检索，不会写入配置或分享包。",
        )
        st.markdown("[OpenAlex API Key 申请与配置教程](https://openalex.org/settings/api)")
        st.caption("本地库包含截至 2025 年的记录；OpenAlex 限 2026 年英文 Article/Review 且有 DOI。")
        if st.button("清空当前对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.caption("问题会先整理为英文检索 prompt，再检索本地语料和在线文献；回答语言跟随提问语言。")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("english_prompt"):
                with st.expander("查看本轮使用的英文 prompt"):
                    st.write(message["english_prompt"])
                    if message.get("search_query"):
                        queries = message.get("search_queries") or [message["search_query"]]
                        for index, query in enumerate(queries, 1):
                            st.caption(f"检索分面 {index}：{query}")
            if include_evidence and message.get("sources"):
                with st.expander(f"查看引用来源（{len(message['sources'])}）"):
                    for paper in message["sources"]:
                        st.markdown(f"**{paper.get('title','Untitled')}** ({paper.get('year') or '年份未知'})")
                        st.markdown(f"DOI: [{paper['doi']}](https://doi.org/{paper['doi']})")
                        modes = ", ".join(sorted(set(paper.get("retrieval_modes", []))))
                        st.caption(f"检索路径：{modes} · {paper.get('journal') or '期刊未知'}")
            if message.get("web_sources"):
                with st.expander(f"查看 Ollama 联网来源（{len(message['web_sources'])}）"):
                    for index, row in enumerate(message["web_sources"], 1):
                        st.markdown(f"**[W{index}] [{row['title']}]({row['url']})**")
            if message.get("openalex_sources"):
                with st.expander(f"查看 OpenAlex 2026 年文献（{len(message['openalex_sources'])}）"):
                    for index, paper in enumerate(message["openalex_sources"], 1):
                        st.markdown(f"**[O{index}] {paper.get('title') or 'Untitled'}** (2026)")
                        if paper.get("doi"):
                            st.markdown(f"DOI: [{paper['doi']}](https://doi.org/{paper['doi']})")
            if message.get("web_note"):
                st.caption(message["web_note"])
            if message.get("openalex_note"):
                st.caption(message["openalex_note"])

    prompt = st.chat_input("输入研究主题或问题；综述模式将生成连贯的学术综述草稿。")
    if not prompt:
        if not st.session_state.messages:
            st.info("先写清主题、地区、年份和期望的综述类型；信息不全时，草稿会标明检索范围与证据限制。")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        review_mode = writing_mode == "正式文献综述"
        status = st.status("正在检索文献并组织综述证据…" if review_mode else "正在检索文献并组织回答…", expanded=False)
        try:
            llm = Ollama(
                model=str(settings["model"]),
                base_url=str(settings.get("ollama_url", "http://127.0.0.1:11434")),
                request_timeout=float(settings.get("request_timeout_seconds", 300)),
                context_window=int(settings.get("context_tokens", 8192)),
                temperature=0.2,
                keep_alive="5m",
                thinking=False,
                additional_kwargs={"num_ctx": int(settings.get("context_tokens", 8192)),
                                   "num_predict": 400},
            )
            retrieval_prompt = (
                "Translate the user's research topic into a faithful, concise English prompt. Create three distinct "
                "English literature-search queries that examine complementary aspects of the same question, such as "
                "core phenomenon and outcomes, research designs and measurement, and settings or implementation. "
                "Do not add concepts or geographic scopes the user did not request. Keep each query concise. "
                'Return only a JSON object with string fields "english_prompt" and "search_query" plus an array of three strings named "search_queries". '
                "Do not answer the research question. /no_think\n\nQuestion: " + prompt
            )
            query_response = llm.chat([
                ChatMessage(role=MessageRole.SYSTEM, content=(
                    f"The current system date is {CURRENT_DATE}. Do not treat your training cutoff as today's date. "
                    "Translate research prompts in any language into clear English without changing the request."
                )),
                ChatMessage(role=MessageRole.USER, content=retrieval_prompt),
            ], format="json").message.content or "{}"
            query_plan = json.loads(query_response)
            english_prompt = str(query_plan.get("english_prompt") or "").strip()
            search_query = str(query_plan.get("search_query") or "").strip()
            search_queries = query_plan.get("search_queries")
            if not isinstance(search_queries, list):
                search_queries = [search_query]
            search_queries = [str(q).strip() for q in search_queries if str(q).strip()][:3]
            if not english_prompt or not search_queries:
                raise RuntimeError("Qwen 未能生成英文 prompt 和分面检索式，请缩短问题后重试。")
            if not search_query:
                search_query = "; ".join(search_queries)
            database = str(settings.get("database") or "")
            if not database:
                raise RuntimeError("配置文件尚未指定本地 SQLite 语料库路径。")
            retriever = get_retriever(database)
            papers_by_doi: dict[str, dict[str, Any]] = {}
            for query_index, query in enumerate(search_queries, 1):
                for paper in retrieve_evidence(retriever, query, top_k, candidate_k):
                    doi = str(paper.get("doi") or paper.get("doc_id") or "")
                    if not doi:
                        continue
                    if doi in papers_by_doi:
                        existing = papers_by_doi[doi]
                        existing["retrieval_modes"] = sorted(set(existing.get("retrieval_modes", []) + paper.get("retrieval_modes", [])))
                        existing["query_indices"] = sorted(set(existing.get("query_indices", []) + [query_index]))
                    else:
                        papers_by_doi[doi] = paper | {"query_indices": [query_index]}
            papers = list(papers_by_doi.values())
            local_candidate_count = len(papers)
            if review_mode:
                papers = papers[:8]
            evidence = evidence_context(papers, char_budget=9000 if review_mode else 10000)
            web_results: list[dict[str, str]] = []
            openalex_result: dict[str, Any] = {"papers": []}
            web_note = ""
            openalex_note = ""
            if not api_key.strip():
                raise RuntimeError("本机未找到分享包中的 Ollama API Key，无法按要求完成每轮联网检索。")
            status.update(label="强制检索 Ollama 互联网最新信息…", state="running")
            web_query = (f"{english_prompt}. Find the latest relevant information available as of {CURRENT_DATE}; "
                         "prefer current primary or official sources and include publication/update dates when available. "
                         "Do not treat older background pages as recent developments.")
            try:
                web_results = ollama_web_search(web_query, api_key.strip(), max_results=5)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
                raise RuntimeError(f"本轮强制 Ollama 联网检索失败，因此未生成可能过时的回答。请检查网络或分享包 API key 后重试。\n\n{exc}") from exc
            if not web_results:
                web_note = "Ollama 联网检索已执行，但没有返回可引用网页；回答会明确标注缺少实时来源。"
            if openalex_enabled and openalex_key.strip():
                status.update(label="正在检索 OpenAlex 2026 年文献…", state="running")
                try:
                    openalex_rows: dict[str, dict[str, Any]] = {}
                    for query in search_queries:
                        result = search_openalex(
                            query, openalex_key.strip(),
                            str(settings.get("openalex_key_env", "OPENALEX_API_KEY")),
                            limit=6 if review_mode else 10,
                        )
                        for paper in result.get("papers", []):
                            doi = str(paper.get("doi") or "")
                            if doi:
                                openalex_rows.setdefault(doi, paper)
                    openalex_result = {"papers": list(openalex_rows.values())}
                except Exception as exc:
                    openalex_note = f"OpenAlex 检索不可用，本轮继续使用其他证据。详细信息：{exc}"
            elif openalex_enabled:
                openalex_note = "请在左侧输入自己的 OPENALEX_API_KEY 后启用 OpenAlex 2026 年文献检索。"
            openalex_papers = openalex_result.get("papers", [])
            openalex_candidate_count = len(openalex_papers)
            if review_mode:
                openalex_papers = openalex_papers[:6]
            remote_evidence = web_evidence_context(web_results)
            recent_evidence = openalex_context(openalex_papers, char_budget=5000 if review_mode else 6500)
            if review_mode:
                remote_evidence = web_evidence_context(web_results, char_budget=1200)
            status.update(label=f"本地候选 {local_candidate_count} 篇 · OpenAlex 候选 {openalex_candidate_count} 篇（引用上下文 {len(papers) + len(openalex_papers)} 篇）· 网页 {len(web_results)} 条，正在生成…", state="running")

            review_instructions = (
                "Write only the literature-review section of a larger research paper, in polished academic English. Do not write a standalone review article. "
                "Do not add a review-paper title, abstract, review-methods section, conclusion, evidence matrix, or complete references section. "
                "Use connected scholarly prose and concise thematic subheadings where useful. Discuss evidence scope, disagreements, limitations and defensible gaps within the section. "
                "Build paragraphs around analytical themes represented in the sources, not a numbered sequence of individual papers; cite multiple source IDs inline where evidence permits. "
                "Do not repeat one numerical effect in every section. Keep a reported number attached to its exact source, outcome, setting, and study design; "
                "do not merge unlike measures or present inconsistent ranges as one estimate. Explicitly distinguish land-surface temperature, near-surface air temperature, "
                "urban-rural temperature differences, and thermal-comfort indices when the sources identify them. Do not infer a metric from a sensor or method alone. "
                "Classify evidence by what a study actually tests: direct intervention/effect evidence, observational association, policy/implementation evidence, "
                "or contextual/methodological evidence. Adjacent studies may inform context or methods but cannot support a direct intervention-effect claim. "
                "Describe research gaps only as gaps in this ranked, year-limited retrieval set; never assert that a topic has no studies globally. "
                "The retrieved records are candidates, not a systematic screened sample. State the actual counts and that full texts were not supplied when applicable. "
                "Use only [S] and [O] records as academic citations. [W] web pages are background context and must not appear as scholarly references. "
                "Do not invent metadata or silently resolve contradictions. Before finalizing, check citation IDs, nonduplicated claims, consistent effect sizes, and claims bounded to the evidence."
            )
            review_scope = (f"\n\nReview evidence scope: {local_candidate_count} local and {openalex_candidate_count} OpenAlex ranked candidates were found; "
                           f"the context contains {len(papers)} local and {len(openalex_papers)} OpenAlex source records after context limits. "
                           "the local corpus contains records through 2025, and OpenAlex is filtered to publication year 2026. These are not a systematic search or independently screened corpus. "
                           "Only bibliographic metadata and abstracts are available; no full-text reading has been performed by this chat.")
            language_rule = ("Output this literature-review section in polished academic English only. " if review_mode else
                             "Answer in the same language as the user's original question; do not force a particular language. ")
            system = (SYSTEM + f"\nThe current system date is {CURRENT_DATE}. This date is authoritative for the present; "
                      "your training data may be older. Never describe 2024 or 2025 as the current year based on model memory. "
                      "Every turn has performed Ollama web search for recent information. For time-sensitive claims, rely on its [W] evidence and cite URLs; "
                      "the API gives a retrieval date but not a verified page publication/update date. Do not call a page newly published merely because it was found today. "
                      "If no relevant web result was returned, say current status could not be verified. "
                      + language_rule +
                      + (review_instructions + review_scope if review_mode else "") +
                      "Use the English prompt below to preserve the user's intent. "
                      "Cite local corpus evidence with [S1] markers, 2026 OpenAlex works with [O1] markers, "
                      "and Ollama web search results with [W1] markers. Include DOI for [O] works and URLs for [W] claims. "
                      "Keep all three evidence types distinct. "
                      "If evidence is missing or abstract-level, say so. Do not invent results, study locations, or DOIs. "
                      "Use /no_think to disable extended reasoning.\n\nTranslated English prompt:\n" + english_prompt +
                      "\n\nLocal literature evidence:\n" + (evidence or "No matching records were found in the local corpus.") +
                      "\n\nOpenAlex 2026 literature evidence:\n" + (recent_evidence or "No 2026 OpenAlex records were retrieved.") +
                      "\n\nOllama web search evidence:\n" + (remote_evidence or "No web search results were retrieved."))
            history = [] if review_mode else st.session_state.messages[:-1]
            messages = to_llama_messages(history + [{"role": "user", "content": "Original user question (any language):\n" + prompt +
                "\n\nTranslated English prompt:\n" + english_prompt +
                "\n\nQwen's complementary search queries:\n" + "\n".join(search_queries) + "\n\n/no_think"}], system)
            llm.additional_kwargs["num_predict"] = 2800 if review_mode else 1200
            answer_placeholder = st.empty()
            answer_parts: list[str] = []
            for chunk in llm.stream_chat(messages):
                if chunk.delta:
                    answer_parts.append(chunk.delta)
                    answer_placeholder.markdown("".join(answer_parts) + "▌")
            answer = "".join(answer_parts).strip() or "我没能生成回答，请缩小问题范围后重试。"
            status.update(label="回答完成", state="complete")
            answer_placeholder.markdown(answer)
            with st.expander("查看本轮使用的英文 prompt", expanded=False):
                st.write(english_prompt)
                for index, query in enumerate(search_queries, 1):
                    st.caption(f"检索分面 {index}：{query}")
            if web_note:
                st.info(web_note)
            if openalex_note:
                st.info(openalex_note)
            if include_evidence and papers:
                with st.expander(f"查看引用来源（{len(papers)}）", expanded=False):
                    for index, paper in enumerate(papers, 1):
                        st.markdown(f"**[S{index}] {paper.get('title','Untitled')}** ({paper.get('year') or '年份未知'})")
                        st.markdown(f"DOI: [{paper['doi']}](https://doi.org/{paper['doi']})")
                        modes = ", ".join(sorted(set(paper.get("retrieval_modes", []))))
                        st.caption(f"检索路径：{modes} · {paper.get('journal') or '期刊未知'}")
                        if paper.get("abstract"):
                            st.caption(str(paper["abstract"])[:900])
            if web_results:
                with st.expander(f"查看 Ollama 联网来源（{len(web_results)}）", expanded=False):
                    for index, row in enumerate(web_results, 1):
                        st.markdown(f"**[W{index}] [{row['title']}]({row['url']})**")
                        if row.get("content"):
                            st.caption(row["content"][:900])
            if openalex_papers:
                with st.expander(f"查看 OpenAlex 2026 年文献（{len(openalex_papers)}）", expanded=False):
                    for index, paper in enumerate(openalex_papers, 1):
                        st.markdown(f"**[O{index}] {paper.get('title') or 'Untitled'}** (2026)")
                        if paper.get("doi"):
                            st.markdown(f"DOI: [{paper['doi']}](https://doi.org/{paper['doi']})")
                        if paper.get("abstract"):
                            st.caption(str(paper["abstract"])[:900])
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": papers,
                "openalex_sources": openalex_papers, "web_sources": web_results,
                "english_prompt": english_prompt, "search_query": search_query,
                "search_queries": search_queries, "writing_mode": writing_mode,
                "web_note": web_note, "openalex_note": openalex_note})
        except Exception as exc:
            status.update(label="本轮未完成", state="error")
            st.error(str(exc))
            st.session_state.messages.pop()


if __name__ == "__main__":
    main()
