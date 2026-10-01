import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# 1. Make src/ importable
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


# Import the working Hybrid GraphRAG V2 backend.
import hybrid_graphrag_real_v2 as backend


# ============================================================
# 2. Page config
# ============================================================

st.set_page_config(
    page_title="Hybrid GraphRAG for 3D Reconstruction",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# 3. Styling
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        color: #777;
        margin-bottom: 1.2rem;
    }

    .metric-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 8px;
    }

    .small-note {
        font-size: 0.88rem;
        color: #777;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. Header
# ============================================================

st.markdown(
    '<div class="main-title">Hybrid GraphRAG for 3D Reconstruction Literature</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-title">
    Neo4j semantic Knowledge Graph + SentenceTransformer vector retrieval + Gemini grounded synthesis
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5. Sidebar
# ============================================================

with st.sidebar:

    st.header(
        "Retrieval Settings"
    )

    top_k = st.slider(
        "Vector Top-K",
        min_value=3,
        max_value=10,
        value=6,
        step=1,
    )

    st.divider()

    st.markdown(
        "**Corpus**"
    )

    st.write(
        "6 papers / 530 chunks"
    )

    st.markdown(
        "**Knowledge Graph**"
    )

    st.write(
        "94 canonical entities"
    )

    st.write(
        "143 canonical semantic relations"
    )

    st.divider()

    st.markdown(
        """
        <div class="small-note">
        The system uses both vector evidence and semantic graph traversal.
        Graph facts are backed by provenance chunks where available.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 6. Question input
# ============================================================

default_question = (
    "解释一下VAD对于UDF的作用，"
    "以及其他论文是否也针对UDF进行变式运用"
)

question = st.text_area(
    "Ask a research question",
    value=default_question,
    height=100,
)

ask = st.button(
    "Ask Hybrid GraphRAG",
    type="primary",
    use_container_width=True,
)


# ============================================================
# 7. Run pipeline
# ============================================================

if ask:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()


    try:

        # ----------------------------------------------------
        # A. Query rewrite
        # ----------------------------------------------------

        with st.status(
            "Running Hybrid GraphRAG...",
            expanded=True,
        ) as status:

            st.write(
                "1/6 Rewriting query for English retrieval..."
            )

            retrieval_query = (
                backend.rewrite_query_for_retrieval(
                    question
                )
            )


            # ------------------------------------------------
            # B. Vector retrieval
            # ------------------------------------------------

            st.write(
                "2/6 Searching vector index..."
            )

            vector_hits = (
                backend.vector_retrieve(
                    retrieval_query,
                    top_k,
                )
            )


            # ------------------------------------------------
            # C. Initial seeds
            # ------------------------------------------------

            st.write(
                "3/6 Finding graph seed entities..."
            )

            lexical_seeds = (
                backend.lexical_entity_seeds(
                    question
                )
            )

            bridge_seeds = (
                backend.chunk_supported_entity_seeds(
                    vector_hits
                )
            )

            initial_seeds = (
                backend.merge_seeds(
                    lexical_seeds,
                    bridge_seeds,
                )
            )


            # ------------------------------------------------
            # D. First-hop graph traversal
            # ------------------------------------------------

            st.write(
                "4/6 Traversing semantic graph..."
            )

            first_hop_facts = (
                backend.graph_expand(
                    [
                        seed["key"]
                        for seed in initial_seeds
                    ],
                    backend.MAX_FIRST_HOP_FACTS,
                )
            )


            # ------------------------------------------------
            # E. Secondary methods + second hop
            # ------------------------------------------------

            secondary_seeds = (
                backend.secondary_method_seeds(
                    initial_seeds,
                    first_hop_facts,
                )
            )

            second_hop_facts = (
                backend.graph_expand(
                    [
                        seed["key"]
                        for seed in secondary_seeds
                    ],
                    backend.MAX_SECOND_HOP_FACTS,
                )
            )

            graph_facts = (
                backend.dedupe_graph_facts(
                    first_hop_facts
                    + second_hop_facts
                )
            )

            graph_facts = graph_facts[
                :(
                    backend.MAX_FIRST_HOP_FACTS
                    + backend.MAX_SECOND_HOP_FACTS
                )
            ]


            # ------------------------------------------------
            # F. Provenance
            # ------------------------------------------------

            st.write(
                "5/6 Recovering graph provenance..."
            )

            all_seeds = (
                initial_seeds
                + secondary_seeds
            )

            introductions = (
                backend.paper_introductions(
                    [
                        seed["key"]
                        for seed in all_seeds
                    ]
                )
            )

            graph_chunks = (
                backend.graph_evidence_chunks(
                    graph_facts
                )
            )


            # ------------------------------------------------
            # G. Gemini synthesis
            # ------------------------------------------------

            st.write(
                "6/6 Generating grounded answer..."
            )

            answer = (
                backend.generate_answer(
                    question,
                    retrieval_query,
                    vector_hits,
                    all_seeds,
                    graph_facts,
                    introductions,
                    graph_chunks,
                )
            )

            status.update(
                label="Hybrid GraphRAG completed",
                state="complete",
                expanded=False,
            )


        # ====================================================
        # 8. Main answer
        # ====================================================

        st.subheader(
            "Answer"
        )

        st.markdown(
            answer
        )


        # ====================================================
        # 9. Query information
        # ====================================================

        with st.expander(
            "Retrieval query",
            expanded=False,
        ):

            st.code(
                retrieval_query,
                language=None,
            )


        # ====================================================
        # 10. Summary metrics
        # ====================================================

        col1, col2, col3, col4 = st.columns(
            4
        )

        col1.metric(
            "Vector chunks",
            len(
                vector_hits
            ),
        )

        col2.metric(
            "Initial graph seeds",
            len(
                initial_seeds
            ),
        )

        col3.metric(
            "Graph facts",
            len(
                graph_facts
            ),
        )

        col4.metric(
            "Provenance chunks",
            len(
                graph_chunks
            ),
        )


        # ====================================================
        # 11. Tabs
        # ====================================================

        tab_vector, tab_graph, tab_sources, tab_debug = (
            st.tabs(
                [
                    "Vector Retrieval",
                    "Knowledge Graph",
                    "Evidence",
                    "Debug / Pipeline",
                ]
            )
        )


        # ----------------------------------------------------
        # Vector Retrieval tab
        # ----------------------------------------------------

        with tab_vector:

            st.subheader(
                "Top-K Vector Retrieval"
            )

            vector_rows = []

            for index, hit in enumerate(
                vector_hits,
                start=1,
            ):

                vector_rows.append(
                    {
                        "Rank":
                            index,

                        "Paper":
                            hit[
                                "source_file"
                            ],

                        "Page":
                            hit[
                                "page"
                            ],

                        "Score":
                            round(
                                hit[
                                    "score"
                                ],
                                4,
                            ),

                        "Chunk ID":
                            hit[
                                "chunk_id"
                            ],
                    }
                )

            if vector_rows:

                st.dataframe(
                    pd.DataFrame(
                        vector_rows
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.info(
                    "No vector chunks retrieved."
                )


        # ----------------------------------------------------
        # Knowledge Graph tab
        # ----------------------------------------------------

        with tab_graph:

            st.subheader(
                "Graph Seed Entities"
            )

            seed_rows = []

            for seed in all_seeds:

                seed_rows.append(
                    {
                        "Entity":
                            seed.get(
                                "name"
                            ),

                        "Type":
                            seed.get(
                                "entity_type"
                            ),

                        "Seed Source":
                            seed.get(
                                "seed_source"
                            ),
                    }
                )

            if seed_rows:

                st.dataframe(
                    pd.DataFrame(
                        seed_rows
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


            st.subheader(
                "Semantic Graph Facts"
            )

            fact_rows = []

            for fact in graph_facts:

                fact_rows.append(
                    {
                        "Source":
                            fact[
                                "source"
                            ],

                        "Relation":
                            fact[
                                "relation"
                            ],

                        "Target":
                            fact[
                                "target"
                            ],

                        "Evidence Count":
                            fact.get(
                                "evidence_count",
                                0,
                            ),
                    }
                )

            if fact_rows:

                st.dataframe(
                    pd.DataFrame(
                        fact_rows
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.info(
                    "No semantic graph facts retrieved."
                )


        # ----------------------------------------------------
        # Evidence tab
        # ----------------------------------------------------

        with tab_sources:

            st.subheader(
                "Retrieved Paper Evidence"
            )

            st.markdown(
                "#### Vector evidence"
            )

            for index, hit in enumerate(
                vector_hits,
                start=1,
            ):

                title = (
                    f"{index}. "
                    f"{hit['source_file']} "
                    f"— p.{hit['page']} "
                    f"(score {hit['score']:.4f})"
                )

                with st.expander(
                    title
                ):

                    st.caption(
                        hit[
                            "chunk_id"
                        ]
                    )

                    st.write(
                        hit[
                            "text"
                        ]
                    )


            st.markdown(
                "#### Graph provenance evidence"
            )

            for index, item in enumerate(
                graph_chunks,
                start=1,
            ):

                title = (
                    f"{index}. "
                    f"{item['source_file']} "
                    f"— p.{item['page']}"
                )

                with st.expander(
                    title
                ):

                    st.caption(
                        item[
                            "chunk_id"
                        ]
                    )

                    st.write(
                        item[
                            "text"
                        ]
                    )


        # ----------------------------------------------------
        # Debug / pipeline tab
        # ----------------------------------------------------

        with tab_debug:

            st.subheader(
                "Pipeline"
            )

            st.code(
                """
Question
   ↓
Multilingual query rewrite
   ↓
SentenceTransformer vector retrieval
   ↓
Top-K paper chunks
   ↓
Lexical entity seeds + vector-to-graph bridge
   ↓
Neo4j semantic graph traversal
   ↓
Secondary focal-method expansion
   ↓
Graph provenance chunks
   ↓
Vector evidence + graph evidence fusion
   ↓
Gemini grounded synthesis
                """.strip(),
                language=None,
            )

            st.markdown(
                "#### Paper introduction context"
            )

            if introductions:

                st.dataframe(
                    pd.DataFrame(
                        introductions
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.write(
                    "No INTRODUCES context found."
                )


    except Exception as error:

        st.error(
            "The Hybrid GraphRAG request failed."
        )

        st.exception(
            error
        )
