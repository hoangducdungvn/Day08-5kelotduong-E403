"""Streamlit RAG chatbot for the group project."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import streamlit as st

try:
    from src.task10_generation import generate_with_citation
except ImportError:
    from task10_generation import generate_with_citation


st.set_page_config(page_title="DrugLaw RAG Chatbot", layout="wide")


@dataclass
class ChatTurn:
    role: str
    content: str
    sources: list[dict] | None = None
    retrieval_source: str | None = None


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def append_turn(
    role: str,
    content: str,
    sources: list[dict] | None = None,
    retrieval_source: str | None = None,
) -> None:
    st.session_state.messages.append(
        ChatTurn(
            role=role,
            content=content,
            sources=sources,
            retrieval_source=retrieval_source,
        )
    )


def build_context_from_history(limit: int = 4) -> str:
    turns = st.session_state.messages[-limit:]
    lines = []
    for turn in turns:
        if turn.role == "user":
            lines.append(f"User: {turn.content}")
        elif turn.role == "assistant":
            lines.append(f"Assistant: {turn.content}")
    return "\n".join(lines)


def clean_text(text: str, max_chars: int = 900) -> str:
    """Make retrieved Markdown snippets readable inside Streamlit source cards."""
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text or "")
    text = re.sub(r"\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_`~]", "", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*#{1,6}\s*", ". ", text)
    text = re.sub(r"\s*[-–—]{3,}\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lstrip(".,;:*- ")
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def source_label(source: dict, index: int) -> tuple[str, str, str]:
    metadata = source.get("metadata", {}) or {}
    title = metadata.get("source") or metadata.get("path") or f"Document {index}"
    doc_type = metadata.get("type", "unknown")
    path = metadata.get("path", "")
    return title, doc_type, path


def render_sources(sources: list[dict], *, expanded: bool) -> None:
    with st.expander("Nguồn tài liệu đã dùng", expanded=expanded):
        for index, source in enumerate(sources, 1):
            title, doc_type, path = source_label(source, index)
            score = float(source.get("score", 0.0) or 0.0)
            snippet = clean_text(source.get("content", ""))

            st.markdown(
                f"""
                <div class="source-card">
                  <div class="source-header">
                    <span class="source-index">{index}</span>
                    <span class="source-title">{html.escape(title)}</span>
                    <span class="source-badge">{html.escape(doc_type)}</span>
                  </div>
                  <div class="source-meta">Score: {score:.3f}{' · ' + html.escape(path) if path else ''}</div>
                  <div class="source-snippet">{html.escape(snippet)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def generate_reply(query: str) -> tuple[str, list[dict], str]:
    contextual_query = query
    history_context = build_context_from_history()
    if history_context:
        contextual_query = f"{query}\n\nConversation context:\n{history_context}"

    result = generate_with_citation(contextual_query)
    return result["answer"], result["sources"], result["retrieval_source"]


st.markdown(
    """
    <style>
      .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
      }
      .source-card {
        border: 1px solid #d8e0df;
        border-radius: 8px;
        padding: 12px 14px;
        margin: 10px 0;
        background: #fbfdfc;
      }
      .source-header {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 4px;
      }
      .source-index {
        width: 24px;
        height: 24px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: #0f766e;
        color: white;
        font-size: 13px;
        font-weight: 700;
      }
      .source-title {
        font-weight: 700;
        color: #12312d;
      }
      .source-badge {
        border: 1px solid #9cc6bd;
        border-radius: 999px;
        padding: 2px 8px;
        color: #0f766e;
        font-size: 12px;
        text-transform: uppercase;
      }
      .source-meta {
        color: #58706b;
        font-size: 13px;
        margin-bottom: 8px;
      }
      .source-snippet {
        color: #213532;
        line-height: 1.55;
        white-space: normal;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

init_state()

st.title("Hỏi đáp về pháp luật ma túy và tin tức liên quan")
st.caption("Hỏi đáp về pháp luật ma túy và tin tức liên quan, có citation và nguồn trích dẫn.")

with st.sidebar:
    st.subheader("Điều khiển")
    show_sources = st.toggle("Hiển thị nguồn", value=True)
    show_retrieval = st.toggle("Hiển thị retriever", value=True)
    max_history = st.slider("Số lượt hội thoại giữ lại", 2, 10, 4)
    if st.button("Xóa hội thoại"):
        st.session_state.messages = []
        st.rerun()

for turn in st.session_state.messages:
    with st.chat_message(turn.role):
        st.markdown(turn.content)
        if show_retrieval and turn.role == "assistant" and turn.retrieval_source:
            st.caption(f"Retriever: `{turn.retrieval_source}`")
        if show_sources and turn.role == "assistant" and turn.sources:
            render_sources(turn.sources, expanded=False)

prompt = st.chat_input("Nhập câu hỏi về ma túy, pháp luật hoặc tin tức liên quan...")
if prompt:
    append_turn("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất và tạo câu trả lời..."):
            answer, sources, retrieval_source = generate_reply(prompt)
        st.markdown(answer)
        if show_retrieval:
            st.caption(f"Retriever: `{retrieval_source}`")
        if show_sources and sources:
            render_sources(sources, expanded=False)

    append_turn("assistant", answer, sources=sources, retrieval_source=retrieval_source)

if len(st.session_state.messages) > max_history * 2:
    st.session_state.messages = st.session_state.messages[-max_history * 2 :]