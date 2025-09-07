import os
import json
import textwrap
from datetime import timedelta, timezone
from typing import Any, Dict, List

import streamlit as st
from openai import OpenAI
from google import genai
from google.genai import types
from pinecone import Pinecone


# =========================
# Clients / Indices
# =========================
def get_clients_no_firebase():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    openai_client = OpenAI()

    gemini_client = genai.Client(api_key=gemini_api_key)

    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )

    pc = Pinecone(api_key=pinecone_api_key)
    dense_index = pc.Index("antiqua-dense")
    sparse_index = pc.Index("antiqua-sparse")
    return openai_client, gemini_client, openrouter_client, pc, dense_index, sparse_index


OPENAI, GEMINI, OPENROUTER, PC, DENSE_IDX, SPARSE_IDX = get_clients_no_firebase()
JST = timezone(timedelta(hours=9), "JST")


# =========================
# Helpers (translation, embed, search)
# =========================
def translate_query(query_ja: str, dialogue_history: list, selected_authors: list, selected_works: list) -> str:
    history_str = ""
    if dialogue_history:
        history_str = "PAST CONVERSATION HISTORY:\n"
        for turn in dialogue_history[-10:]:
            role = turn["role"]
            content = turn["content"]
            history_str += f"{role}: {content}\n"
        history_str += "\n"

    filter_context_str = ""
    parts = []
    if selected_authors:
        authors = [name.split("_")[0] for name in selected_authors]
        parts.append(f"Authors: {', '.join(authors)}")
    if selected_works:
        parts.append(f"Works: {', '.join(selected_works)}")
    if parts:
        filter_context_str = "SEARCH SCOPE:\n" + "\n".join(parts)

    full_user_content = f"{history_str}\nCURRENT QUERY TO REFORMULATE:\n{query_ja}\n\n{filter_context_str}".strip()

    system_prompt = textwrap.dedent("""
        You are a highly specialized query reformulator for a vector database focused on Western Classical literature (Ancient Greek & Latin). Your sole purpose is to convert a user's intent into a perfect, self-contained English search query.

        You will receive the user's intent in a structured format containing:
        1.  `PAST CONVERSATION HISTORY`: For context and resolving pronouns.
        2.  `CURRENT QUERY TO REFORMULATE`: The user's latest query, which is your main focus.
        3.  `SEARCH SCOPE`: Use this information (if any) to translate the user's query.

        **Your Four-Step Process:**
        1.  **Analyze Intent:** Fully grasp the user's core question from the `CURRENT QUERY` and surrounding context.
        2.  **Translate to English:** Accurately translate the Japanese query into clear English. This is the most critical first step.
        3.  **Enrich with Context:** Integrate key terms from the `PAST CONVERSATION` and entities from `CONTEXT FROM FILTERS` to make the query specific and self-contained.
        4.  **Format for Search:** Formulate the final query as a concise, unambiguous string, applying domain-specific knowledge (e.g., standard names like "Thucydides", "Nicomachean Ethics"; technical terms like "arete (virtue, excellence)").

        **CRITICAL OUTPUT RULE:**
        - Your response MUST be the final English query string and nothing else.
        - DO NOT include any prefixes, explanations, or conversational text. Just the query.

        ---
        Here are examples of how you must perform.
        ---
        [EXAMPLE 1]
        USER:
        PAST CONVERSATION:
        User: Tell me about Thucydides' account of the plague.
        Assistant: Thucydides describes the Athenian plague in Book 2...

        CURRENT QUERY TO REFORMULATE:
        彼はペリクレスをどこで評価している？

        CONTEXT FROM FILTERS:
        Authors: Thucydides
        Works: History of the Peloponnesian War

        ASSISTANT:
        Where does Thucydides evaluate Pericles in the History of the Peloponnesian War, especially Book 2.65?
        ---
        [EXAMPLE 2]
        USER:
        PAST CONVERSATION:
        User: Summarize Cicero's concept of duty.
        Assistant: In De Officiis, Cicero discusses officium...

        CURRENT QUERY TO REFORMULATE:
        officiumの意味は？

        CONTEXT FROM FILTERS:
        Authors: Cicero
        Works: De Officiis

        ASSISTANT:
        What does Cicero mean by officium (“duty,” “obligation”) in De Officiis, focusing on definitions and examples in Book 1?
        ---
        [EXAMPLE 3]
        USER:
        PAST CONVERSATION:
        User: Explain virtue in Aristotle's ethics.
        Assistant: Aristotle defines aretē in relation to eudaimonia...

        CURRENT QUERY TO REFORMULATE:
        アレテーとエウダイモニアの関係は？

        CONTEXT FROM FILTERS:
        Authors: Aristotle
        Works: Nicomachean Ethics

        ASSISTANT:
        How does Aristotle relate arete (“virtue,” “excellence”) to eudaimonia (“happiness,” “flourishing”) in the Nicomachean Ethics, especially Book I?
        ---
    """)

    final_user_prompt = f"""Please reformulate a search query based on the following information. Your output must be only the final English query string.\n---\n{full_user_content}"""

    res = OPENROUTER.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_user_prompt},
        ],
    )
    return res.choices[0].message.content.strip()


def embed_query(query_en: str) -> List[float]:
    resp = GEMINI.models.embed_content(
        contents=query_en,
        model="gemini-embedding-001",
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return resp.embeddings[0].values


def custom_metadata_filter(selected_authors, selected_works, exclude_toggle):
    filters: Dict[str, Any] = {}
    if exclude_toggle:
        if selected_works:
            filters["work"] = {"$nin": selected_works}
        elif selected_authors:
            filters["author"] = {"$nin": selected_authors}
    else:
        if selected_authors:
            filters["author"] = {"$in": selected_authors}
        if selected_works:
            filters["work"] = {"$in": selected_works}
    return filters if filters else None


def search_pinecone(vec: List[float], query_en: str, k: int = 20, filters=None):
    if filters is None:
        filters = {}
    dense = DENSE_IDX.query(
        namespace="__default__",
        vector=vec,
        top_k=k,
        filter=filters,
        include_metadata=True,
        include_values=False,
    )
    sparse = SPARSE_IDX.search(
        namespace="__default__",
        query={"top_k": k, "inputs": {"text": query_en}, "filter": filters},
        fields=["id", "author", "work", "chunk_text", "text"],
    )
    return dense, sparse


def merge_hits(resp1, resp2):
    def _norm(resp):
        if "result" in resp and "hits" in resp["result"]:
            for h in resp["result"]["hits"]:
                yield {
                    "id": h["_id"],
                    "score": h["_score"],
                    "author": h["fields"].get("author"),
                    "work": h["fields"].get("work"),
                    "chunk_text": h["fields"].get("chunk_text"),
                    "text": h["fields"].get("text"),
                }
        else:
            for h in resp.get("matches", []):
                m = h["metadata"]
                yield {
                    "id": h["id"],
                    "score": h["score"],
                    "author": m.get("author"),
                    "work": m.get("work"),
                    "chunk_text": m.get("chunk_text"),
                    "text": m.get("text"),
                }

    combined = {h["id"]: h for h in _norm(resp1)}
    combined.update({h["id"]: h for h in _norm(resp2)})
    return sorted(combined.values(), key=lambda x: x["score"], reverse=True)

def _author_str(v):
    if isinstance(v, list):
        return ", ".join(v)
    return v or ""

def make_rank_payload(doc: Dict[str, Any]) -> str:
    # 英語のみ・短く：著者／著作を先頭に、次に要約
    author = _author_str(doc.get("author"))
    work = doc.get("work", "")
    summary_en = doc.get("text", "")  # ← index.py では英語要約が "text"
    # トークン節約のため改行は少なめ・記号区切りに
    return f"Author: {author} | Work: {work}\nSummary: {summary_en}"

def build_sources_html_from_rerank(rerank_data: List[Dict[str, Any]], doc_lookup) -> str:
    html = []
    for row in rerank_data:
        rid = str(row["document"].get("id", ""))
        doc = doc_lookup.get(rid, {}) 
        authors_list = doc.get("author", [])
        authors_str = ", ".join(authors_list) if isinstance(authors_list, list) else (authors_list or "")
        html.append(
            f"<details>"
            f"<summary>{(doc.get('id') or rid).replace('_', ' ')} / Score: {float(row['score'])}</summary>"
            f"<p><strong>Author:</strong> {authors_str} &nbsp;&nbsp; <strong>Work:</strong> {doc.get('work','')}</p>"
            f"<p>{doc.get('chunk_text','')}</p>"
            f"<p><strong>【Summary in English】</strong><br/>{doc.get('text','')}</p>"
            f"</details>"
        )
    return "\n".join(html)


def build_sources_html_from_docs(docs: List[Dict[str, Any]]) -> str:
    html = []
    for d in docs:
        authors_list = d.get("author", [])
        authors_str = ", ".join(authors_list) if isinstance(authors_list, list) else (authors_list or "")
        html.append(
            f"<details>"
            f"<summary>{d['id'].replace('_', ' ')} / Score: {float(d['score'])}</summary>"
            f"<p><strong>Author:</strong> {authors_str} &nbsp;&nbsp; <strong>Work:</strong> {d.get('work','')}</p>"
            f"<p>{d.get('chunk_text','')}</p>"
            f"<p><strong>【Summary in English】</strong><br/>{d.get('text','')}</p>"
            f"</details>"
        )
    return "\n".join(html)


# =========================
# Load metadata
# =========================
with open("authors_master.json", "r", encoding="utf-8") as f:
    _data = json.load(f)

authors_data = _data["authors"]
meta_authors = sorted(authors_data.keys())
meta_data: Dict[str, List[str]] = {}
for author, details in authors_data.items():
    works = details.get("works", [])
    meta_data[author] = sorted(works)


# =========================
# Streamlit UI (main only; no sidebar)
# =========================
st.set_page_config(page_title="Antiqua 検索評価", layout="wide")
st.title("🔎 Humanitext Antiqua 検索評価")
st.caption("クエリに基づく候補文脈を2種類表示します（内容のみ）。")

st.subheader("検索条件")
query = st.text_input("クエリ（日本語でOK）", placeholder="例）キケロの officium の定義は？")
authors_sel = st.multiselect("著者フィルタ", options=meta_authors)
candidate_works = sorted({w for a in authors_sel for w in meta_data.get(a, [])})
works_sel = st.multiselect("著作フィルタ（選択著者の作品のみ）", options=candidate_works)
exclude_toggle = st.checkbox("上記の著者／著作を検索から除外", value=False)

# 固定設定（UI非表示）
TOP_K = 20
CTX_N = 10
SHOW_ENQ = False

col_l, col_r = st.columns(2)

if query:
    dialogue_history: List[Dict[str, str]] = [{"role": "user", "content": query}]

    en_query = translate_query(query, dialogue_history, authors_sel, works_sel)
    if SHOW_ENQ:
        st.info(f"English query: {en_query}")

    vec = embed_query(en_query)

    filters = custom_metadata_filter(authors_sel, works_sel, exclude_toggle)
    dense_resp, sparse_resp = search_pinecone(vec, en_query, k=TOP_K, filters=filters)

    processed_dense = merge_hits(dense_resp, {"matches": []})
    processed_hybrid = merge_hits(dense_resp, sparse_resp)
    doc_lookup = { str(d["id"]): d for d in processed_hybrid if d.get("id") }

    docs_for_rerank = []
    for d in processed_hybrid:
        rid = str(d.get("id", ""))  # idは必ず文字列
        if not rid:
            continue
        txt = make_rank_payload(d)
        if not txt:
            continue
        docs_for_rerank.append({"id": rid, "text": txt})

    # ① rank_payload を使うケース（推奨）
    rerank_hybrid = PC.inference.rerank(
        model="pinecone-rerank-v0",
        query=en_query,
        documents=docs_for_rerank,
        rank_fields=["text"],   # ★ 著者・著作＋要約だけを読む
        top_n=CTX_N,
        return_documents=True,
        parameters={"truncate": "END"},
    )

    # 左: 結果A（Dense: リランキングなし、score順上位5）
    with col_l:
        st.markdown("### 結果A")
        if not processed_dense:
            st.warning("該当する文脈が見つかりませんでした。条件を調整してください。")
        else:
            html_left = build_sources_html_from_docs(processed_dense[:CTX_N])
            st.markdown(html_left, unsafe_allow_html=True)

    # 右: 結果B（ハイブリッド: 再ランク上位5）
    with col_r:
        st.markdown("### 結果B")
        if not rerank_hybrid.data:
            st.warning("該当する文脈が見つかりませんでした。条件を調整してください。")
        else:
            html_right = build_sources_html_from_rerank(rerank_hybrid.data, doc_lookup)
            st.markdown(html_right, unsafe_allow_html=True)
else:
    st.info("上の入力欄にクエリを入力して実行してください。")

