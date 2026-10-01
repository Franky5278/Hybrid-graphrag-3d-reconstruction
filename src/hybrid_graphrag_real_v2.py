import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Environment
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
    GEMINI_API_KEY,
]):
    raise ValueError(
        "Missing Neo4j or Gemini configuration in .env"
    )


# ============================================================
# 2. Configuration
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.5-flash-lite"
VECTOR_INDEX = "paper_chunk_embeddings"

DEFAULT_TOP_K = 6

MAX_LEXICAL_SEEDS = 8
MAX_VECTOR_BRIDGE_SEEDS = 6
MAX_INITIAL_SEEDS = 10

MAX_FIRST_HOP_FACTS = 32
MAX_SECONDARY_METHODS = 6
MAX_SECOND_HOP_FACTS = 28

MAX_GRAPH_EVIDENCE_CHUNKS = 14

SEMANTIC_REL_TYPES = [
    "USES",
    "ADDRESSES",
    "EVALUATES_ON",
    "MEASURED_BY",
    "IMPROVES",
    "COMPARED_WITH",
    "PRODUCES",
    "SOLVES",
]

RELATION_PRIORITY = {
    "USES": 0,
    "PRODUCES": 1,
    "ADDRESSES": 2,
    "SOLVES": 3,
    "EVALUATES_ON": 4,
    "MEASURED_BY": 5,
    "IMPROVES": 6,
    "COMPARED_WITH": 7,
}


# ============================================================
# 3. Clients
# ============================================================

print("Loading embedding model...")

embedder = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model ready.")

gemini = genai.Client(
    api_key=GEMINI_API_KEY
)

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(
        NEO4J_USERNAME,
        NEO4J_PASSWORD,
    ),
)

driver.verify_connectivity()

print("Neo4j connected.")


# ============================================================
# 4. Query rewrite for English embedding index
# ============================================================

def rewrite_query_for_retrieval(question):
    """
    The corpus and embedding index are English-heavy, while the user
    may ask in Chinese. Rewrite only for retrieval; do not answer here.
    """

    prompt = f"""
Rewrite the following user question into ONE concise English
retrieval query for an academic-paper vector search.

Rules:
- Preserve all named methods, acronyms, datasets, and technical terms.
- Translate Chinese into English when needed.
- Do not answer the question.
- Do not add methods or claims that are not in the question.
- Return only the rewritten query, no quotes and no explanation.

Question:
{question}
"""

    try:
        response = gemini.interactions.create(
            model=GEMINI_MODEL,
            input=prompt,
        )

        rewritten = (
            response.output_text
            .strip()
            .strip('"')
            .strip()
        )

        if rewritten:
            return rewritten

    except Exception as error:
        print(
            "Query rewrite warning:",
            error,
        )

    return question


# ============================================================
# 5. Vector retrieval
# ============================================================

def vector_retrieve(
    retrieval_query,
    top_k,
):
    query_embedding = embedder.encode(
        retrieval_query,
        normalize_embeddings=True,
    ).tolist()

    top_k = int(top_k)

    cypher = f"""
    MATCH (c:Chunk)

    SEARCH c IN (
        VECTOR INDEX {VECTOR_INDEX}
        FOR $query_embedding
        LIMIT {top_k}
    )

    SCORE AS score

    RETURN
        c.chunk_id AS chunk_id,
        c.paper_id AS paper_id,
        c.source_file AS source_file,
        c.page AS page,
        c.text AS text,
        score

    ORDER BY score DESC
    """

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        return [
            dict(row)
            for row in session.run(
                cypher,
                query_embedding=query_embedding,
            )
        ]


# ============================================================
# 6. Lexical entity matching in Python
# ============================================================

def load_all_entities():
    cypher = """
    MATCH (e:KnowledgeEntity)

    RETURN
        e.key AS key,
        e.name AS name,
        e.entity_type AS entity_type,
        coalesce(e.aliases, []) AS aliases
    """

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        return [
            dict(row)
            for row in session.run(
                cypher
            )
        ]


ALL_ENTITIES = load_all_entities()


def token_match(
    text,
    candidate,
):
    """
    Match entity names and short acronyms such as UDF, CD, VAD.
    For ASCII tokens, use alphanumeric boundaries to reduce accidental
    substring matches.
    """

    text_cf = text.casefold()
    candidate_cf = candidate.casefold().strip()

    if not candidate_cf:
        return False

    if all(
        ord(ch) < 128
        for ch in candidate_cf
    ):
        pattern = (
            r"(?<![A-Za-z0-9])"
            + re.escape(candidate_cf)
            + r"(?![A-Za-z0-9])"
        )

        return (
            re.search(
                pattern,
                text_cf,
            )
            is not None
        )

    return (
        candidate_cf
        in text_cf
    )


def lexical_entity_seeds(
    question,
):
    matches = []

    for entity in ALL_ENTITIES:
        candidates = [
            entity["name"],
            *entity.get(
                "aliases",
                []
            ),
        ]

        matched_alias = None

        for candidate in sorted(
            set(candidates),
            key=len,
            reverse=True,
        ):
            if token_match(
                question,
                candidate,
            ):
                matched_alias = candidate
                break

        if matched_alias:
            matches.append(
                {
                    "key":
                        entity["key"],

                    "name":
                        entity["name"],

                    "entity_type":
                        entity["entity_type"],

                    "seed_source":
                        "lexical",

                    "matched_text":
                        matched_alias,
                }
            )

    # Prefer longer/more-specific matches.
    matches.sort(
        key=lambda item: len(
            item.get(
                "matched_text",
                ""
            )
        ),
        reverse=True,
    )

    return matches[
        :MAX_LEXICAL_SEEDS
    ]


# ============================================================
# 7. Vector -> graph bridge
# ============================================================

def chunk_supported_entity_seeds(
    vector_hits,
):
    chunk_ids = [
        item["chunk_id"]
        for item in vector_hits
        if item.get(
            "chunk_id"
        )
    ]

    if not chunk_ids:
        return []

    cypher = """
    UNWIND $chunk_ids AS chunk_id

    MATCH
        (e:KnowledgeEntity)
        -[:SUPPORTED_BY]->
        (c:Chunk {chunk_id: chunk_id})

    RETURN
        e.key AS key,
        e.name AS name,
        e.entity_type AS entity_type,
        count(DISTINCT c) AS support_count,
        'vector_bridge' AS seed_source

    ORDER BY
        support_count DESC,
        name

    LIMIT $limit
    """

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        return [
            dict(row)
            for row in session.run(
                cypher,
                chunk_ids=chunk_ids,
                limit=MAX_VECTOR_BRIDGE_SEEDS,
            )
        ]


def merge_seeds(
    lexical_seeds,
    bridge_seeds,
):
    merged = []
    seen = set()

    for seed in (
        lexical_seeds
        + bridge_seeds
    ):
        key = seed.get(
            "key"
        )

        if (
            not key
            or key in seen
        ):
            continue

        seen.add(
            key
        )

        merged.append(
            seed
        )

        if (
            len(merged)
            >= MAX_INITIAL_SEEDS
        ):
            break

    return merged


# ============================================================
# 8. Deduplicate / rank graph facts
# ============================================================

def dedupe_graph_facts(
    facts,
):
    best = {}

    for fact in facts:
        key = (
            fact["source_key"],
            fact["relation"],
            fact["target_key"],
        )

        existing = best.get(
            key
        )

        if (
            existing is None
            or fact.get(
                "evidence_count",
                0
            )
            > existing.get(
                "evidence_count",
                0
            )
        ):
            best[key] = fact

    result = list(
        best.values()
    )

    result.sort(
        key=lambda item: (
            RELATION_PRIORITY.get(
                item["relation"],
                99,
            ),
            -item.get(
                "evidence_count",
                0,
            ),
            item[
                "source"
            ].casefold(),
            item[
                "target"
            ].casefold(),
        )
    )

    return result


# ============================================================
# 9. First-hop graph traversal
# ============================================================

def graph_expand(
    seed_keys,
    limit,
):
    if not seed_keys:
        return []

    cypher = """
    UNWIND $seed_keys AS seed_key

    MATCH
        (seed:KnowledgeEntity {key: seed_key})
        -[r]-
        (neighbor:KnowledgeEntity)

    WHERE
        type(r)
        IN $allowed_rel_types

    RETURN DISTINCT
        startNode(r).key AS source_key,
        startNode(r).name AS source,
        startNode(r).entity_type AS source_type,

        type(r) AS relation,

        endNode(r).key AS target_key,
        endNode(r).name AS target,
        endNode(r).entity_type AS target_type,

        coalesce(
            r.source_files,
            []
        ) AS source_files,

        coalesce(
            r.evidence_pages,
            []
        ) AS evidence_pages,

        coalesce(
            r.evidence_chunk_ids,
            []
        ) AS evidence_chunk_ids,

        coalesce(
            r.evidence_count,
            0
        ) AS evidence_count
    """

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        facts = [
            dict(row)
            for row in session.run(
                cypher,
                seed_keys=seed_keys,
                allowed_rel_types=SEMANTIC_REL_TYPES,
            )
        ]

    facts = dedupe_graph_facts(
        facts
    )

    return facts[
        :limit
    ]


# ============================================================
# 10. Discover focal methods reachable from current facts
# ============================================================

def introduced_method_keys():
    cypher = """
    MATCH
        (:Paper)
        -[:INTRODUCES]->
        (e:KnowledgeEntity)

    RETURN DISTINCT
        e.key AS key
    """

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        return {
            row["key"]
            for row in session.run(
                cypher
            )
        }


FOCAL_METHOD_KEYS = introduced_method_keys()


def secondary_method_seeds(
    initial_seeds,
    first_hop_facts,
):
    existing = {
        seed["key"]
        for seed in initial_seeds
    }

    candidates = []

    for fact in first_hop_facts:
        for key_field, type_field, name_field in [
            (
                "source_key",
                "source_type",
                "source",
            ),
            (
                "target_key",
                "target_type",
                "target",
            ),
        ]:
            key = fact.get(
                key_field
            )

            entity_type = fact.get(
                type_field
            )

            name = fact.get(
                name_field
            )

            if (
                key
                and key
                not in existing
                and key
                in FOCAL_METHOD_KEYS
                and entity_type
                in {
                    "Method",
                    "Model",
                }
            ):
                candidates.append(
                    {
                        "key":
                            key,

                        "name":
                            name,

                        "entity_type":
                            entity_type,

                        "seed_source":
                            "graph_neighbor",
                    }
                )

                existing.add(
                    key
                )

                if (
                    len(candidates)
                    >= MAX_SECONDARY_METHODS
                ):
                    return candidates

    return candidates


# ============================================================
# 11. Paper-introduction context
# ============================================================

def paper_introductions(
    seed_keys,
):
    if not seed_keys:
        return []

    cypher = """
    UNWIND $seed_keys AS seed_key

    MATCH
        (p:Paper)
        -[:INTRODUCES]->
        (e:KnowledgeEntity {key: seed_key})

    RETURN DISTINCT
        p.source_file AS paper,
        e.name AS entity

    ORDER BY
        paper,
        entity
    """

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        return [
            dict(row)
            for row in session.run(
                cypher,
                seed_keys=seed_keys,
            )
        ]


# ============================================================
# 12. Recover graph provenance chunks
# ============================================================

def graph_evidence_chunks(
    graph_facts,
):
    chunk_ids = []
    seen = set()

    for fact in graph_facts:
        for chunk_id in fact.get(
            "evidence_chunk_ids",
            []
        ):
            if (
                not chunk_id
                or chunk_id in seen
            ):
                continue

            seen.add(
                chunk_id
            )

            chunk_ids.append(
                chunk_id
            )

            if (
                len(chunk_ids)
                >= MAX_GRAPH_EVIDENCE_CHUNKS
            ):
                break

        if (
            len(chunk_ids)
            >= MAX_GRAPH_EVIDENCE_CHUNKS
        ):
            break

    if not chunk_ids:
        return []

    cypher = """
    UNWIND $chunk_ids AS chunk_id

    MATCH (
        c:Chunk
        {
            chunk_id:
                chunk_id
        }
    )

    RETURN
        c.chunk_id AS chunk_id,
        c.source_file AS source_file,
        c.page AS page,
        c.text AS text
    """

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        rows = session.run(
            cypher,
            chunk_ids=chunk_ids,
        )

        by_id = {
            row[
                "chunk_id"
            ]:
                dict(
                    row
                )
            for row in rows
        }

    return [
        by_id[
            chunk_id
        ]
        for chunk_id in chunk_ids
        if chunk_id in by_id
    ]


# ============================================================
# 13. Prompt formatting
# ============================================================

def format_vector_evidence(
    vector_hits,
):
    blocks = []

    for index, hit in enumerate(
        vector_hits,
        start=1,
    ):
        blocks.append(
            f"""
[VECTOR {index}]
Paper: {hit["source_file"]}
Page: {hit["page"]}
Score: {hit["score"]:.4f}
Chunk ID: {hit["chunk_id"]}

Text:
{hit["text"]}
""".strip()
        )

    return "\n\n".join(
        blocks
    )


def format_graph_facts(
    graph_facts,
):
    blocks = []

    for index, fact in enumerate(
        graph_facts,
        start=1,
    ):
        blocks.append(
            f"""
[GRAPH {index}]
{fact["source"]} --{fact["relation"]}--> {fact["target"]}
Source files: {fact.get("source_files", [])}
Pages: {fact.get("evidence_pages", [])}
Evidence chunks: {fact.get("evidence_chunk_ids", [])}
""".strip()
        )

    return "\n\n".join(
        blocks
    )


def format_graph_chunks(
    graph_chunks,
):
    blocks = []

    for index, item in enumerate(
        graph_chunks,
        start=1,
    ):
        blocks.append(
            f"""
[GRAPH-EVIDENCE {index}]
Paper: {item["source_file"]}
Page: {item["page"]}
Chunk ID: {item["chunk_id"]}

Text:
{item["text"]}
""".strip()
        )

    return "\n\n".join(
        blocks
    )


def build_answer_prompt(
    question,
    retrieval_query,
    vector_hits,
    all_seeds,
    graph_facts,
    introductions,
    graph_chunks,
):
    seed_text = json.dumps(
        [
            {
                "name":
                    seed.get(
                        "name"
                    ),

                "type":
                    seed.get(
                        "entity_type"
                    ),

                "source":
                    seed.get(
                        "seed_source"
                    ),
            }
            for seed in all_seeds
        ],
        ensure_ascii=False,
        indent=2,
    )

    intro_text = json.dumps(
        introductions,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are answering a technical research question using a
HYBRID GraphRAG evidence set.

Use ONLY the supplied evidence.

Do not use outside knowledge.
Do not invent facts, equations, datasets, results, or relationships.

IMPORTANT SEMANTIC RULE:
A COMPARED_WITH relation only proves that two methods were compared.
It does NOT by itself prove that the compared method is UDF-based,
uses UDF, modifies UDF, or shares the same technical mechanism.

To claim that a method uses or modifies UDF, require direct paper-text
evidence or a USES / PRODUCES relation supported by provenance.

Prefer direct paper text for technical explanations.

If vector and graph evidence disagree, prefer direct paper text and
explicitly state the discrepancy.

If evidence is insufficient for a method, say that the current
retrieved evidence only establishes comparison, not its internal UDF
mechanism.

Cite factual technical statements as:
[Paper filename, p.X]

Do not cite a page unless supplied evidence supports that page.

Answer in the same language as the user's original question.

============================================================
ORIGINAL QUESTION
============================================================

{question}

============================================================
ENGLISH RETRIEVAL QUERY
============================================================

{retrieval_query}

============================================================
GRAPH SEED ENTITIES
============================================================

{seed_text}

============================================================
PAPER INTRODUCTIONS
============================================================

{intro_text}

============================================================
VECTOR EVIDENCE
============================================================

{format_vector_evidence(vector_hits)}

============================================================
SEMANTIC GRAPH FACTS
============================================================

{format_graph_facts(graph_facts)}

============================================================
GRAPH FACT PROVENANCE CHUNKS
============================================================

{format_graph_chunks(graph_chunks)}

============================================================
ANSWER
============================================================
"""


# ============================================================
# 14. Gemini synthesis
# ============================================================

def generate_answer(
    question,
    retrieval_query,
    vector_hits,
    all_seeds,
    graph_facts,
    introductions,
    graph_chunks,
):
    prompt = build_answer_prompt(
        question,
        retrieval_query,
        vector_hits,
        all_seeds,
        graph_facts,
        introductions,
        graph_chunks,
    )

    response = gemini.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
    )

    return response.output_text


# ============================================================
# 15. Console helpers
# ============================================================

def print_vector_hits(
    vector_hits,
):
    print("\n")
    print("=" * 70)
    print("VECTOR RETRIEVAL")
    print("=" * 70)

    for index, hit in enumerate(
        vector_hits,
        start=1,
    ):
        preview = (
            hit["text"]
            .replace(
                "\n",
                " "
            )
            [:180]
        )

        print(
            f"{index}. "
            f"{hit['source_file']} "
            f"| p.{hit['page']} "
            f"| score={hit['score']:.4f}"
        )

        print(
            f"   {preview}..."
        )


def print_seeds(
    title,
    seeds,
):
    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    if not seeds:
        print(
            "No seeds found."
        )
        return

    for index, seed in enumerate(
        seeds,
        start=1,
    ):
        extra = ""

        if seed.get(
            "matched_text"
        ):
            extra = (
                f" | matched="
                f"{seed['matched_text']}"
            )

        print(
            f"{index}. "
            f"{seed['name']} "
            f"[{seed['entity_type']}] "
            f"<- {seed['seed_source']}"
            f"{extra}"
        )


def print_graph_facts(
    graph_facts,
):
    print("\n")
    print("=" * 70)
    print("GRAPH TRAVERSAL")
    print("=" * 70)

    if not graph_facts:
        print(
            "No semantic graph facts found."
        )
        return

    for index, fact in enumerate(
        graph_facts,
        start=1,
    ):
        print(
            f"{index}. "
            f"{fact['source']} "
            f"--{fact['relation']}--> "
            f"{fact['target']}"
        )


# ============================================================
# 16. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid Vector + Semantic Graph RAG "
            "over the 3D reconstruction paper corpus."
        )
    )

    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    args = parser.parse_args()

    question = args.question

    if not question:
        question = input(
            "\nEnter your research question:\n> "
        ).strip()

    if not question:
        print(
            "Question is empty. Exiting."
        )
        return

    print("\n")
    print("=" * 70)
    print("HYBRID GRAPHRAG V2")
    print("=" * 70)

    print(
        f"\nOriginal question:\n"
        f"{question}"
    )

    # --------------------------------------------------------
    # A. Multilingual -> English retrieval rewrite
    # --------------------------------------------------------

    retrieval_query = (
        rewrite_query_for_retrieval(
            question
        )
    )

    print(
        f"\nRetrieval query:\n"
        f"{retrieval_query}"
    )

    # --------------------------------------------------------
    # B. Vector retrieval
    # --------------------------------------------------------

    vector_hits = vector_retrieve(
        retrieval_query,
        args.top_k,
    )

    print_vector_hits(
        vector_hits
    )

    # --------------------------------------------------------
    # C. Initial graph seeds
    # --------------------------------------------------------

    lexical_seeds = (
        lexical_entity_seeds(
            question
        )
    )

    bridge_seeds = (
        chunk_supported_entity_seeds(
            vector_hits
        )
    )

    initial_seeds = merge_seeds(
        lexical_seeds,
        bridge_seeds,
    )

    print_seeds(
        "INITIAL GRAPH SEEDS",
        initial_seeds,
    )

    # --------------------------------------------------------
    # D. First hop
    # --------------------------------------------------------

    first_hop_facts = graph_expand(
        [
            seed["key"]
            for seed in initial_seeds
        ],
        MAX_FIRST_HOP_FACTS,
    )

    # --------------------------------------------------------
    # E. Discover other focal methods connected to seeds
    # --------------------------------------------------------

    secondary_seeds = (
        secondary_method_seeds(
            initial_seeds,
            first_hop_facts,
        )
    )

    print_seeds(
        "SECONDARY METHOD SEEDS",
        secondary_seeds,
    )

    # --------------------------------------------------------
    # F. Second hop from related focal methods
    # --------------------------------------------------------

    second_hop_facts = graph_expand(
        [
            seed["key"]
            for seed in secondary_seeds
        ],
        MAX_SECOND_HOP_FACTS,
    )

    graph_facts = dedupe_graph_facts(
        first_hop_facts
        + second_hop_facts
    )

    graph_facts = graph_facts[
        :(
            MAX_FIRST_HOP_FACTS
            + MAX_SECOND_HOP_FACTS
        )
    ]

    print_graph_facts(
        graph_facts
    )

    # --------------------------------------------------------
    # G. Paper provenance
    # --------------------------------------------------------

    all_seeds = (
        initial_seeds
        + secondary_seeds
    )

    introductions = paper_introductions(
        [
            seed["key"]
            for seed in all_seeds
        ]
    )

    graph_chunks = graph_evidence_chunks(
        graph_facts
    )

    print(
        "\nGraph provenance chunks recovered:",
        len(
            graph_chunks
        ),
    )

    # --------------------------------------------------------
    # H. Gemini grounded synthesis
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("GENERATING GROUNDED ANSWER")
    print("=" * 70)

    answer = generate_answer(
        question,
        retrieval_query,
        vector_hits,
        all_seeds,
        graph_facts,
        introductions,
        graph_chunks,
    )

    print("\n")
    print("=" * 70)
    print("HYBRID GRAPHRAG V2 ANSWER")
    print("=" * 70)

    print(
        answer
    )


if __name__ == "__main__":
    try:
        main()

    finally:
        driver.close()
