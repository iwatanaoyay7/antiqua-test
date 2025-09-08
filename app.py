import os
import textwrap
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

# =========================
# Helpers (translation, embed, search)
# =========================
def translate_query(query_ja: str) -> str:

    full_user_content = f"CURRENT QUERY TO REFORMULATE:\n{query_ja}".strip()

    system_prompt = textwrap.dedent("""
        You are a highly specialized query reformulator for a vector database focused on Western Classical literature (Ancient Greek & Latin). Your sole purpose is to convert a user's intent into a perfect, self-contained English search query.

        You will receive the user's intent in a structured format containing:
        1.  `CURRENT QUERY TO REFORMULATE`: The user's latest query, which is your main focus.

        **Your Four-Step Process:**
        1.  **Analyze Intent:** Fully grasp the user's core question from the `CURRENT QUERY` and surrounding context.
        2.  **Translate to English:** Accurately translate the Japanese query into clear English. This is the most critical first step.
        3.  **Format for Search:** Formulate the final query as a concise, unambiguous string, applying domain-specific knowledge (e.g., standard names like "Thucydides", "Nicomachean Ethics"; technical terms like "arete (virtue, excellence)").

        **CRITICAL OUTPUT RULE:**
        - Your response MUST be the final English query string and nothing else.
        - DO NOT include any prefixes, explanations, or conversational text. Just the query.
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
        html.append(
            f"<details>"
            f"<summary>{(doc.get('id') or rid).replace('_', ' ')} / Score: {float(row['score'])}</summary>"
            f"<p>{doc.get('chunk_text','')}</p>"
            f"<p><strong>【Summary in English】</strong><br/>{doc.get('text','')}</p>"
            f"</details>"
        )
    return "\n".join(html)


def build_sources_html_from_docs(docs: List[Dict[str, Any]]) -> str:
    html = []
    for d in docs:
        html.append(
            f"<details>"
            f"<summary>{d['id'].replace('_', ' ')} / Score: {float(d['score'])}</summary>"
            f"<p>{d.get('chunk_text','')}</p>"
            f"<p><strong>【Summary in English】</strong><br/>{d.get('text','')}</p>"
            f"</details>"
        )
    return "\n".join(html)

# =========================
# Streamlit UI (main only; no sidebar)
# =========================
st.set_page_config(page_title="Antiqua 検索評価", layout="wide")
st.title("🔎 Humanitext Antiqua 検索評価")
st.caption("クエリに基づく候補文脈を2種類表示します（内容のみ）。")

# --- 評価ガイド（expander表示） ---------------------------------------------
def render_evaluation_guide():
    guide_md = textwrap.dedent("""
    # 評価ガイド

    **目的**  
    同じクエリに対する「結果A（左）」と「結果B（右）」の上位候補（Top5）を比べ、どちらがより適切に答えているかを簡潔に判断ください。

    ---  
    ## 何を見るか
    - **Top1 合致度**：1位の文脈がクエリに直撃しているか。  
    - **Top3 合致度**：上位3件のまとまり（的外れの少なさ、重複の少なさ、適切な広がり）。  
    - **Top5 合致度**：上位5件の総合力（関連性の一貫性、不要ノイズの少なさ、順番の妥当性）。  

    各項目ごとに **A / B / 引分 / どちらも不適** の4択で判断します。

    ---  
    ## クエリタイプ
    - **タイプA（著者・著作を明記／複数可）**  
      例：*プラトン『パイドン』の魂の不死*  
      → 理想は、明記した著者・著作（アンカー）が **Top1 に出る**こと。  
      → ただし他著者・他作品でも、クエリ回答に **有用なら評価根拠に含めてよい**（“入っているだけ”で機械的に×にしない）。

    - **タイプB（著者のみ明記／複数可）**  
      例：*キケロの officium の定義*  
      → 明記した著者由来が上位に来るのが望ましいが、有用なら関連他著者も可。

    - **タイプC（著者・著作を明記しない）**  
      例：*アレテーとエウダイモニアの関係*  
      → テーマへの直撃度と順位の適切さを重視（特定作者の前提なし）。

    ---  
    ## 判定のヒント
    - **直撃度**：質問にそのまま答える内容か。  
    - **不要ノイズ**：関係の薄い候補が混じっていないか。  
    - **順序**：より適切なものが上位に来ている。
    - **広がり**：Top3/Top5で重複ばかりにならず、補助的観点が適度に含まれているか。

    ---  
    ## 操作手順
    1. クエリ入力 → **「検索を実行」**を押す（Enterでは送信されません）。  
    2. 左が**結果A**、右が**結果B**。  
    3. 3項目（Top1 / Top3 / Top5）それぞれで **A / B / 引分 / どちらも不適** を選ぶ。  
    4. 次のクエリへ。  
    ※ A/Bがどの方式かは気にしないでください（ブラインド）。

    ---  
    ## 記録（CSV）
    各行：1クエリ  
    - `query_type`（A / B / C）  
    - `query_text`  
    - `top1_winner` / `top3_winner` / `top5_winner`（A / B / 引分 / どちらも不適）  
    最下部に**自由記述コメント**もお願いします（まとめて1つ）。
    """).strip()

    with st.expander("📘 評価ガイド（クリックで開く）", expanded=False):
        st.markdown(guide_md)

# 使いどころ（タイトルの直後など）
render_evaluation_guide()

st.subheader("検索条件")
query = st.text_area("クエリ（日本語でOK）", placeholder="例）キケロの officium の定義は？", height=100)
run_btn = st.button("🔎 検索を実行", type="primary")

# 固定設定（UI非表示）
TOP_K = 20
CTX_N = 5
SHOW_ENQ = False

col_l, col_r = st.columns(2)

if run_btn:
    if not query.strip():
        st.warning("クエリを入力してください。")
        st.stop()
    with st.spinner("検索中…"):
        en_query = translate_query(query)
        if SHOW_ENQ:
            st.info(f"English query: {en_query}")

        vec = embed_query(en_query)

        filters = {}
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
    st.info("上の入力欄にクエリを入力して「検索を実行」を押してください。")

