"""Native Windows desktop client for the literature research assistant."""
from __future__ import annotations

import html
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, Qt, QSettings, Signal
from PySide6.QtGui import QPalette, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit,
    QGroupBox, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSlider, QSpinBox,
    QTextBrowser, QVBoxLayout, QWidget,
)

ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
if getattr(sys, "frozen", False) and not (ROOT / "planning-review-local.json").is_file():
    # Keep the native EXE in its own subfolder so a currently running browser
    # build can remain open while the desktop client is installed alongside it.
    if (ROOT.parent / "planning-review-local.json").is_file():
        ROOT = ROOT.parent
DIST = Path(getattr(sys, "_MEIPASS", ROOT)) / "distribution"
if str(DIST) not in sys.path:
    sys.path.insert(0, str(DIST))


class ResearchWorker(QObject):
    status = Signal(str)
    token = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, prompt: str, settings: dict[str, Any], openalex_key: str,
                 review_mode: bool, top_k: int, candidate_k: int):
        super().__init__()
        self.prompt = prompt
        self.settings = settings
        self.openalex_key = openalex_key
        self.review_mode = review_mode
        self.top_k = top_k
        self.candidate_k = candidate_k

    def run(self) -> None:
        try:
            from llama_index.core.llms import ChatMessage, MessageRole
            from llama_index.llms.ollama import Ollama
            from urban_planning_agent.chat_app import (
                CURRENT_DATE, evidence_context, ollama_web_search,
                openalex_context, packaged_ollama_api_key, retrieve_evidence,
                search_openalex, web_evidence_context,
            )
            from urban_planning_agent.retriever import Retriever

            current_date = CURRENT_DATE
            model = Ollama(
                model=str(self.settings["model"]),
                base_url=str(self.settings.get("ollama_url", "http://127.0.0.1:11434")),
                request_timeout=float(self.settings.get("request_timeout_seconds", 300)),
                context_window=int(self.settings.get("context_tokens", 8192)),
                temperature=0.2, keep_alive="5m", thinking=False,
                additional_kwargs={"num_ctx": int(self.settings.get("context_tokens", 8192)),
                                   "num_predict": 400},
            )
            self.status.emit("Qwen 正在将问题整理为英文检索计划…")
            plan_prompt = (
                "Translate the research question faithfully into concise English and create three complementary "
                "English literature search queries. Return only JSON with english_prompt, search_query and "
                "search_queries (array of three strings). Do not answer. /no_think\n\nQuestion: " + self.prompt
            )
            plan_raw = model.chat([
                ChatMessage(role=MessageRole.SYSTEM, content=f"Today's date is {current_date}. Translate faithfully."),
                ChatMessage(role=MessageRole.USER, content=plan_prompt),
            ], format="json").message.content or "{}"
            import json
            plan = json.loads(plan_raw)
            english = str(plan.get("english_prompt") or "").strip()
            queries = plan.get("search_queries")
            if not isinstance(queries, list):
                queries = [plan.get("search_query", "")]
            queries = [str(q).strip() for q in queries if str(q).strip()][:3]
            if not english or not queries:
                raise RuntimeError("Qwen 未能生成英文检索计划，请缩短问题后重试。")

            database = str(self.settings.get("database") or "")
            if not database or not Path(database).is_file():
                raise RuntimeError(f"未找到本地文献数据库：{database or '(未配置)'}")
            self.status.emit("正在检索本地文献库…")
            # The bundled local corpus ends in 2025. Keep all its records; 2026
            # is the publication-year filter for OpenAlex only.
            retriever = Retriever(database)
            by_doi: dict[str, dict[str, Any]] = {}
            for ix, query in enumerate(queries, 1):
                for paper in retrieve_evidence(retriever, query, self.top_k, self.candidate_k):
                    doi = str(paper.get("doi") or paper.get("doc_id") or "")
                    if doi:
                        if doi in by_doi:
                            by_doi[doi]["retrieval_modes"] = sorted(set(by_doi[doi].get("retrieval_modes", []) + paper.get("retrieval_modes", [])))
                        else:
                            by_doi[doi] = paper | {"query_indices": [ix]}
            local_count = len(by_doi)
            local = list(by_doi.values())[:8 if self.review_mode else 18]

            key = packaged_ollama_api_key()
            if not key:
                raise RuntimeError("未找到分享包内置的 Ollama 联网 API Key，无法执行强制联网检索。")
            self.status.emit("正在强制调用 Ollama 搜索最新互联网信息…")
            web_query = (f"{english}. Find the latest relevant information as of {current_date}; prefer current primary or official sources. "
                         "Include publication/update dates when available. Do not treat older background pages as recent developments.")
            web = ollama_web_search(web_query, key, max_results=5)

            oa: dict[str, dict[str, Any]] = {}
            if self.openalex_key.strip():
                self.status.emit("正在检索 OpenAlex 的 2026 年文献…")
                for query in queries:
                    result = search_openalex(query, self.openalex_key.strip(),
                        str(self.settings.get("openalex_key_env", "OPENALEX_API_KEY")), limit=6 if self.review_mode else 10)
                    for paper in result.get("papers", []):
                        doi = str(paper.get("doi") or "")
                        if doi:
                            oa.setdefault(doi, paper)
            oa_rows = list(oa.values())[:6 if self.review_mode else 18]

            local_ctx = evidence_context(local, char_budget=9000)
            oa_ctx = openalex_context(oa_rows, char_budget=5000)
            web_ctx = web_evidence_context(web, char_budget=1200 if self.review_mode else 6000)
            scope = (f"The retrieval found {local_count} local and {len(oa)} OpenAlex candidates; context contains "
                     f"{len(local)} and {len(oa_rows)} records. The bundled local corpus covers records through 2025; "
                     "OpenAlex is filtered to publication year 2026. "
                     "These are not a systematic search or independently screened corpus. Only metadata and abstracts are available; no full-text reading was performed.")
            review_rules = ""
            if self.review_mode:
                review_rules = (
                    "Write only the literature-review section of a larger research paper, in polished academic English. Do not write a standalone review article. "
                    "Do not add a review-paper title, abstract, review-methods section, conclusion, evidence matrix, or complete references section. "
                    "Use connected scholarly prose and concise thematic subheadings where useful. Synthesize across studies rather than listing papers. "
                    "Discuss evidence scope, disagreements, limitations and defensible gaps within the section. Keep numerical effects tied to exact source, measure, setting and design; distinguish land-surface "
                    "temperature, near-surface air temperature, urban-rural differences and thermal comfort. Separate direct intervention, "
                    "observational, policy and contextual evidence. Gaps apply only to this retrieval set; never claim no studies exist globally. "
                    "State counts and abstract-only limitations in proportionate prose. Cite only supplied [S] and [O] sources inline; [W] are web sources. "
                    "Do not invent metadata or resolve contradictions silently.\n\n" + scope + "\n\n"
                )
            language_instruction = (
                "Output this literature-review section in polished academic English only. " if self.review_mode else
                "Answer in the same language as the user's original question. Do not force a particular language. "
            )
            system = (
                f"The current date is {current_date}; it is authoritative. Never describe 2024 or 2025 as current. "
                "Every turn has performed Ollama web search. For time-sensitive claims rely on [W] evidence and cite URLs. "
                "Search results have retrieval date but no verified publication date. If no relevant web results, say current status is unverified. "
                + language_instruction + review_rules +
                "Cite local records [S1], OpenAlex [O1], and web results [W1] with URLs when they support current claims. "
                "Keep evidence types distinct; do not invent facts.\n\n"
                f"English prompt: {english}\n\nLocal evidence:\n{local_ctx or 'No matching local records.'}\n\n"
                f"OpenAlex 2026 evidence:\n{oa_ctx or 'No OpenAlex records.'}\n\nOllama web results:\n{web_ctx or 'No results; current status unverified.'}"
            )
            messages = [ChatMessage(role=MessageRole.SYSTEM, content=system),
                        ChatMessage(role=MessageRole.USER, content=f"Question: {self.prompt}\nEnglish prompt: {english}\nQueries: {'; '.join(queries)}\n/no_think")]
            model.additional_kwargs["num_predict"] = 2800 if self.review_mode else 1200
            self.status.emit("检索完成，正在撰写…")
            answer_parts: list[str] = []
            for chunk in model.stream_chat(messages):
                if chunk.delta:
                    answer_parts.append(chunk.delta)
                    self.token.emit(chunk.delta)
            answer = "".join(answer_parts).strip()
            self.finished.emit({"answer": answer, "english": english, "queries": queries,
                                "local": local, "web": web, "openalex": oa_rows,
                                "local_count": local_count, "oa_count": len(oa)})
        except Exception as exc:
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from urban_planning_agent.chat_app import settings_from_cli
        self.settings = settings_from_cli()
        self.setWindowTitle("文献综述研究助手")
        self.resize(1180, 800)
        self.setMinimumSize(860, 620)
        self.thread: QThread | None = None
        self.worker: ResearchWorker | None = None
        self.messages: list[tuple[str, str]] = []
        self.appearance_settings = QSettings("LiteratureResearchAgent", "ResearchChat")
        self.live_cursor: QTextCursor | None = None

        root = QWidget()
        row = QHBoxLayout(root)
        row.setContentsMargins(0, 0, 0, 0)
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(280)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(22, 24, 22, 20)
        title = QLabel("文献研究助手")
        title.setObjectName("brand")
        side.addWidget(title)
        side.addSpacing(14)
        side.addWidget(QLabel("写作模式"))
        self.mode = QComboBox()
        self.mode.addItems(["正式文献综述", "研究问答"])
        side.addWidget(self.mode)
        side.addSpacing(8)
        side.addWidget(QLabel("外观"))
        self.theme = QComboBox()
        self.theme.addItems(["跟随系统", "浅色", "深色"])
        side.addWidget(self.theme)
        side.addWidget(QLabel("本地文献范围：2025 年"))
        side.addWidget(QLabel("OpenAlex：仅检索 2026 年"))
        side.addWidget(QLabel("检索深度"))
        self.topk = QSpinBox(); self.topk.setRange(2, 8); self.topk.setValue(6)
        side.addWidget(self.topk)
        self.openalex = QLineEdit()
        self.openalex.setPlaceholderText("可选：粘贴自己的 OpenAlex API key")
        self.openalex.setEchoMode(QLineEdit.EchoMode.Password)
        side.addWidget(self.openalex)
        self.net_status = QLabel("每次提问强制执行 Ollama 联网搜索")
        self.net_status.setWordWrap(True)
        side.addWidget(self.net_status)
        side.addStretch(1)
        clear = QPushButton("清空对话")
        clear.clicked.connect(self.clear_chat)
        side.addWidget(clear)
        row.addWidget(sidebar)

        main = QWidget(); content = QVBoxLayout(main)
        content.setContentsMargins(28, 22, 28, 18)
        self.heading = QLabel("正式文献综述")
        self.heading.setObjectName("heading")
        content.addWidget(self.heading)
        self.feed = QTextBrowser()
        self.feed.setOpenExternalLinks(True)
        self.feed.setObjectName("feed")
        content.addWidget(self.feed, 1)
        self.trace_box = QGroupBox("检索过程与证据依据")
        self.trace_box.setCheckable(True)
        self.trace_box.setChecked(True)
        trace_layout = QVBoxLayout(self.trace_box)
        self.trace_view = QTextBrowser()
        self.trace_view.setMaximumHeight(145)
        trace_layout.addWidget(self.trace_view)
        self.trace_box.toggled.connect(self.trace_view.setVisible)
        content.addWidget(self.trace_box)
        self.status_label = QLabel("就绪")
        content.addWidget(self.status_label)
        compose = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("输入研究主题、地区、时间范围和综述要求…")
        self.input.returnPressed.connect(self.send)
        self.send_button = QPushButton("发送")
        self.send_button.setObjectName("send")
        self.send_button.clicked.connect(self.send)
        compose.addWidget(self.input, 1); compose.addWidget(self.send_button)
        content.addLayout(compose)
        row.addWidget(main, 1)
        self.setCentralWidget(root)
        saved_theme = str(self.appearance_settings.value("theme", "跟随系统"))
        index = self.theme.findText(saved_theme)
        self.theme.setCurrentIndex(index if index >= 0 else 0)
        self.theme.currentTextChanged.connect(self.apply_theme)
        self.apply_theme(self.theme.currentText())
        self.mode.currentIndexChanged.connect(lambda i: self.heading.setText(self.mode.currentText()))
        self.append_message("assistant", "你好！请输入研究问题。我会整理英文检索计划，检索截至 2025 年的本地文献库、2026 年 OpenAlex 文献，并强制搜索 Ollama 互联网信息，再撰写带证据边界的回答。")

    def apply_theme(self, choice: str) -> None:
        self.appearance_settings.setValue("theme", choice)
        if choice == "跟随系统":
            dark = QApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128
        else:
            dark = choice == "深色"
        if dark:
            colors = {"page": "#212121", "panel": "#171717", "surface": "#2f2f2f", "raised": "#3a3a3a",
                      "text": "#ececec", "muted": "#b4b4b4", "border": "#424242", "accent": "#10a37f",
                      "hover": "#19b58c", "user": "#9bbcff", "assistant": "#74d8b5"}
        else:
            colors = {"page": "#f7f7f8", "panel": "#f0f0f0", "surface": "#ffffff", "raised": "#e8e8e8",
                      "text": "#202123", "muted": "#6b6b6b", "border": "#e5e5e5", "accent": "#10a37f",
                      "hover": "#0e8f6f", "user": "#3156a3", "assistant": "#0b8063"}
        c = colors
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background:{c['page']}; color:{c['text']}; font-family:'Microsoft YaHei UI'; font-size:14px; }}
            QWidget#sidebar {{ background:{c['panel']}; color:{c['text']}; border-right:1px solid {c['border']}; }}
            QWidget#sidebar QLabel {{ color:{c['muted']}; }}
            QLabel#brand {{ color:{c['text']}; font-size:22px; font-weight:700; }}
            QLabel#heading {{ font-size:21px; font-weight:650; }}
            QTextBrowser#feed {{ background:{c['page']}; color:{c['text']}; border:0; padding:14px 24px; selection-background-color:{c['accent']}; }}
            QLineEdit, QComboBox, QSpinBox {{ background:{c['surface']}; color:{c['text']}; border:1px solid {c['border']}; border-radius:12px; padding:10px; }}
            QGroupBox {{ color:{c['muted']}; border:1px solid {c['border']}; border-radius:10px; margin-top:8px; padding-top:6px; }}
            QGroupBox::title {{ subcontrol-origin:margin; left:10px; padding:0 5px; }}
            QWidget#sidebar QLineEdit, QWidget#sidebar QComboBox, QWidget#sidebar QSpinBox {{ background:{c['surface']}; color:{c['text']}; border-color:{c['border']}; }}
            QPushButton {{ background:{c['raised']}; color:{c['text']}; border:0; border-radius:9px; padding:10px 16px; }}
            QPushButton#send {{ background:{c['accent']}; color:white; min-width:90px; font-weight:650; }}
            QPushButton:hover {{ background:{c['hover']}; color:white; }}
        """)
        self.feed.document().setDefaultStyleSheet(f"body {{ color:{c['text']}; }} p {{ color:{c['text']}; }} hr {{ border:0; border-top:1px solid {c['border']}; }}")
        self.trace_view.setStyleSheet(f"QTextBrowser {{ background:{c['surface']}; color:{c['text']}; border:0; }}")
        self.feed.clear()
        for role, text in self.messages:
            self.render_message(role, text, c)

    def log_trace(self, text: str) -> None:
        self.status_label.setText(text)
        self.trace_view.append(html.escape(text))
        self.trace_view.moveCursor(QTextCursor.MoveOperation.End)

    def begin_assistant_stream(self) -> None:
        self.messages.append(("assistant", ""))
        dark = self.theme.currentText() == "深色" or (self.theme.currentText() == "跟随系统" and QApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128)
        label_color = "#74d8b5" if dark else "#0b8063"
        self.feed.append(f'<p><span style="color:{label_color};font-weight:700">研究助手</span></p><p style="line-height:1.65">')
        self.live_cursor = QTextCursor(self.feed.document())
        self.live_cursor.movePosition(QTextCursor.MoveOperation.End)

    def append_stream_text(self, text: str) -> None:
        if self.live_cursor is None or not text:
            return
        self.live_cursor.insertText(text)
        self.feed.setTextCursor(self.live_cursor)
        self.feed.ensureCursorVisible()
        if self.messages and self.messages[-1][0] == "assistant":
            role, previous = self.messages[-1]
            self.messages[-1] = (role, previous + text)

    def render_message(self, role: str, text: str, colors: dict[str, str] | None = None) -> None:
        if colors is None:
            dark = self.theme.currentText() == "深色" or (self.theme.currentText() == "跟随系统" and QApplication.palette().color(QPalette.ColorRole.Window).lightness() < 128)
            colors = ({"user": "#9bbcff", "assistant": "#74d8b5"} if dark else {"user": "#3156a3", "assistant": "#0b8063"})
        color = colors["assistant"] if role == "assistant" else colors["user"]
        label = "研究助手" if role == "assistant" else "你"
        safe = html.escape(text).replace("\n", "<br>")
        self.feed.append(f'<p><span style="color:{color};font-weight:700">{label}</span></p><p style="line-height:1.65">{safe}</p><hr>')
        self.feed.moveCursor(QTextCursor.MoveOperation.End)

    def append_message(self, role: str, text: str) -> None:
        self.messages.append((role, text))
        self.render_message(role, text)

    def send(self) -> None:
        prompt = self.input.text().strip()
        if not prompt or self.thread is not None:
            return
        from urban_planning_agent.chat_app import packaged_ollama_api_key
        if not packaged_ollama_api_key():
            QMessageBox.critical(self, "联网搜索配置错误", "未找到内置 Ollama API key；为保证每轮联网，已停止本次回答。")
            return
        self.input.clear()
        self.append_message("user", prompt)
        self.begin_assistant_stream()
        self.trace_view.clear()
        self.trace_box.setChecked(True)
        self.log_trace("已收到问题，正在生成英文检索计划。")
        self.send_button.setEnabled(False)
        self.status_label.setText("正在启动检索…")
        self.thread = QThread(self)
        self.worker = ResearchWorker(prompt, self.settings, self.openalex.text(), self.mode.currentIndex() == 0,
                                     self.topk.value(), 80)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.status.connect(self.log_trace)
        self.worker.token.connect(self.append_stream_text)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup_worker)
        self.thread.start()

    def on_finished(self, result: dict[str, Any]) -> None:
        self.log_trace("Qwen 已生成英文检索计划。")
        self.log_trace("英文检索 prompt：" + result["english"])
        for ix, query in enumerate(result["queries"], 1):
            self.log_trace(f"检索分面 {ix}：{query}")
        self.log_trace(f"本地文献检索完成：{result['local_count']} 篇候选；范围截至 2025 年。")
        self.log_trace(f"OpenAlex 检索完成：{result['oa_count']} 篇 2026 年候选。")
        self.log_trace(f"Ollama 联网搜索完成：{len(result['web'])} 条结果。")
        self.log_trace("已基于候选摘要和网页来源生成回答；来源标识如下。")
        sources = []
        for i, p in enumerate(result["local"], 1):
            sources.append(f"[S{i}] {p.get('title','Untitled')} ({p.get('year','')}) DOI: {p.get('doi','')}")
        for i, p in enumerate(result["openalex"], 1):
            sources.append(f"[O{i}] {p.get('title','Untitled')} DOI: {p.get('doi','')}")
        for i, p in enumerate(result["web"], 1):
            sources.append(f"[W{i}] {p['title']} — {p['url']}")
        if sources:
            for source in sources:
                self.trace_view.append(html.escape(source))
        self.live_cursor = None
        self.status_label.setText("回答完成")
        self.send_button.setEnabled(True)

    def on_failed(self, message: str) -> None:
        self.append_stream_text("\n\n本轮未完成：" + message.split("\n\nTraceback")[0])
        self.live_cursor = None
        self.log_trace("检索或生成失败；已停止输出未完成的答案。")
        self.send_button.setEnabled(True)

    def cleanup_worker(self) -> None:
        if self.worker:
            self.worker.deleteLater()
        if self.thread:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None

    def clear_chat(self) -> None:
        self.messages = []
        self.feed.clear()
        self.append_message("assistant", "新对话已开始。每次提问仍会强制执行 Ollama 联网搜索。")


def main() -> None:
    if "--prepare-corpus" in sys.argv:
        from tkinter import messagebox
        try:
            from urban_planning_agent.bundle import prepare_bundle
            result = prepare_bundle(ROOT / "corpus", ROOT / "data", include_vectors=False)
            messagebox.showinfo("文献综述研究助手", f"文献库已准备完成：\n{result}")
        except Exception as exc:
            messagebox.showerror("文献综述研究助手", f"文献库准备失败：\n\n{exc}")
        return
    config = ROOT / "planning-review-local.json"
    if not config.is_file():
        from tkinter import messagebox
        messagebox.showerror("文献综述研究助手", f"找不到应用配置文件：\n{config}")
        return
    os.environ["PLANNING_REVIEW_CONFIG"] = str(config)
    app = QApplication(sys.argv)
    app.setApplicationName("文献综述研究助手")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
