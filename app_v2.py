import sys
from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import hybrid_graphrag_real_v2 as backend

st.set_page_config(page_title="3D Recon GraphRAG", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:2rem;max-width:1500px}
.hero{border:1px solid rgba(150,150,150,.18);border-radius:18px;padding:22px 24px;margin-bottom:18px;background:rgba(127,127,127,.04)}
.hero-title{font-size:2.2rem;font-weight:760;margin-bottom:8px}
.hero-subtitle{opacity:.72}
.tag{display:inline-block;border:1px solid rgba(150,150,150,.22);border-radius:999px;padding:4px 10px;margin-right:6px;margin-top:8px;font-size:.82rem}
div[data-testid="stMetric"]{border:1px solid rgba(150,150,150,.18);padding:12px 14px;border-radius:14px;background:rgba(127,127,127,.035)}
</style>
""", unsafe_allow_html=True)

if "question" not in st.session_state:
    st.session_state.question = "解释一下VAD对于UDF的作用，以及其他论文是否也针对UDF进行变式运用"
if "result" not in st.session_state:
    st.session_state.result = None

with st.sidebar:
    st.header("Retrieval Settings")
    top_k = st.slider("Vector Top-K", 3, 10, 6)
    st.divider()
    st.subheader("Corpus")
    st.metric("Papers", "6")
    st.metric("Chunks", "530")
    st.subheader("Knowledge Graph")
    st.metric("Canonical entities", "94")
    st.metric("Semantic relations", "143")
    st.caption("Vector retrieval + Neo4j semantic traversal + provenance-grounded synthesis")

st.markdown("""
<div class="hero">
<div class="hero-title">Hybrid GraphRAG for 3D Reconstruction Literature</div>
<div class="hero-subtitle">Research assistant for UDF / surface reconstruction papers</div>
<span class="tag">Neo4j</span><span class="tag">SentenceTransformer</span><span class="tag">Gemini</span><span class="tag">Vector + Graph Retrieval</span><span class="tag">Evidence Provenance</span>
</div>
""", unsafe_allow_html=True)

st.subheader("Try a question")
examples = [
    ("VAD + UDF", "解释一下VAD对于UDF的作用，以及其他论文是否也针对UDF进行变式运用"),
    ("Compare VAD / GeoUDF", "How do VAD and GeoUDF differ in their use of unsigned distance fields?"),
    ("Chamfer Distance", "Which methods in these papers are evaluated using Chamfer Distance?"),
    ("Thin structures", "Which paper focuses on thin-structure reconstruction and how does it differ from standard UDF approaches?"),
]
cols = st.columns(4)
for col, (label, q) in zip(cols, examples):
    if col.button(label, use_container_width=True):
        st.session_state.question = q

question = st.text_area("Research question", key="question", height=110)
ask = st.button("Run Hybrid GraphRAG", type="primary", use_container_width=True)

def run_pipeline(question, top_k):
    with st.status("Running Hybrid GraphRAG...", expanded=True) as status:
        st.write("1/6 Rewriting query...")
        retrieval_query = backend.rewrite_query_for_retrieval(question)

        st.write("2/6 Vector retrieval...")
        vector_hits = backend.vector_retrieve(retrieval_query, top_k)

        st.write("3/6 Graph seed detection...")
        lexical = backend.lexical_entity_seeds(question)
        bridge = backend.chunk_supported_entity_seeds(vector_hits)
        initial = backend.merge_seeds(lexical, bridge)

        st.write("4/6 Graph traversal...")
        first = backend.graph_expand([s["key"] for s in initial], backend.MAX_FIRST_HOP_FACTS)
        secondary = backend.secondary_method_seeds(initial, first)
        second = backend.graph_expand([s["key"] for s in secondary], backend.MAX_SECOND_HOP_FACTS)
        facts = backend.dedupe_graph_facts(first + second)
        facts = facts[: backend.MAX_FIRST_HOP_FACTS + backend.MAX_SECOND_HOP_FACTS]

        st.write("5/6 Recovering provenance...")
        all_seeds = initial + secondary
        introductions = backend.paper_introductions([s["key"] for s in all_seeds])
        graph_chunks = backend.graph_evidence_chunks(facts)

        st.write("6/6 Generating answer...")
        answer = backend.generate_answer(
            question, retrieval_query, vector_hits, all_seeds, facts, introductions, graph_chunks
        )
        status.update(label="Hybrid GraphRAG completed", state="complete", expanded=False)

    return dict(
        retrieval_query=retrieval_query,
        vector_hits=vector_hits,
        all_seeds=all_seeds,
        facts=facts,
        graph_chunks=graph_chunks,
        introductions=introductions,
        answer=answer,
    )

if ask:
    try:
        st.session_state.result = run_pipeline(question, top_k)
    except Exception as e:
        st.error("Request failed. If Gemini returns a temporary 503, retry shortly.")
        st.exception(e)

r = st.session_state.result

if r:
    st.divider()
    st.header("Answer")
    st.markdown(r["answer"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vector chunks", len(r["vector_hits"]))
    c2.metric("Graph seeds", len(r["all_seeds"]))
    c3.metric("Graph facts", len(r["facts"]))
    c4.metric("Provenance chunks", len(r["graph_chunks"]))

    with st.expander("English retrieval query"):
        st.code(r["retrieval_query"], language=None)

    t1, t2, t3, t4 = st.tabs(["Vector Retrieval", "Knowledge Graph", "Evidence", "Pipeline"])

    with t1:
        rows = [
            {"Rank": i, "Paper": h["source_file"], "Page": h["page"], "Score": round(h["score"],4), "Chunk ID": h["chunk_id"]}
            for i, h in enumerate(r["vector_hits"], 1)
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with t2:
        a, b = st.columns([.34, .66])
        with a:
            st.subheader("Seed entities")
            st.dataframe(pd.DataFrame([
                {"Entity": s.get("name"), "Type": s.get("entity_type"), "Source": s.get("seed_source")}
                for s in r["all_seeds"]
            ]), use_container_width=True, hide_index=True)
        with b:
            st.subheader("Semantic graph facts")
            st.dataframe(pd.DataFrame([
                {"Source": f["source"], "Relation": f["relation"], "Target": f["target"], "Evidence": f.get("evidence_count",0)}
                for f in r["facts"]
            ]), use_container_width=True, hide_index=True)

    with t3:
        st.subheader("Vector evidence")
        for i, h in enumerate(r["vector_hits"], 1):
            with st.expander(f"{i}. {h['source_file']} — p.{h['page']} (score {h['score']:.4f})"):
                st.caption(h["chunk_id"])
                st.write(h["text"])
        st.subheader("Graph provenance evidence")
        for i, g in enumerate(r["graph_chunks"], 1):
            with st.expander(f"{i}. {g['source_file']} — p.{g['page']}"):
                st.caption(g["chunk_id"])
                st.write(g["text"])

    with t4:
        st.code("""User Question
↓
Multilingual Query Rewrite
↓
SentenceTransformer Vector Retrieval
↓
Top-K Paper Chunks
↓
Lexical KG Seeds + Vector→Graph Bridge
↓
Neo4j Graph Traversal
↓
Secondary Focal-Method Expansion
↓
Graph Provenance Recovery
↓
Vector + Graph Evidence Fusion
↓
Gemini Grounded Synthesis""", language=None)
        if r["introductions"]:
            st.dataframe(pd.DataFrame(r["introductions"]), use_container_width=True, hide_index=True)
else:
    st.info("Choose a sample question or enter your own query, then run Hybrid GraphRAG.")
