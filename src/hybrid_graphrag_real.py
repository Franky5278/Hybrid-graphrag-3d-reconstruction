import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, GEMINI_API_KEY]):
    raise ValueError("Missing Neo4j or Gemini settings in .env")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.5-flash-lite"
VECTOR_INDEX = "paper_chunk_embeddings"

DEFAULT_TOP_K = 5
MAX_GRAPH_SEEDS = 6
MAX_GRAPH_FACTS = 30
MAX_GRAPH_EVIDENCE_CHUNKS = 12

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

print("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("Embedding model ready.")

gemini = genai.Client(api_key=GEMINI_API_KEY)

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
)
driver.verify_connectivity()
print("Neo4j connected.")


def vector_retrieve(question, top_k):
    query_embedding = embedder.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    top_k = int(top_k)

    cypher = f'''
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
    '''

    with driver.session(database=NEO4J_DATABASE) as session:
        return [
            dict(row)
            for row in session.run(
                cypher,
                query_embedding=query_embedding,
            )
        ]


def lexical_entity_seeds(question):
    cypher = '''
    MATCH (e:KnowledgeEntity)

    WHERE
        toLower($question) CONTAINS toLower(e.name)
        OR any(
            alias IN coalesce(e.aliases, [])
            WHERE
                size(alias) >= 4
                AND toLower($question) CONTAINS toLower(alias)
        )

    RETURN DISTINCT
        e.key AS key,
        e.name AS name,
        e.entity_type AS entity_type,
        'lexical' AS seed_source

    LIMIT $limit
    '''

    with driver.session(database=NEO4J_DATABASE) as session:
        return [
            dict(row)
            for row in session.run(
                cypher,
                question=question,
                limit=MAX_GRAPH_SEEDS,
            )
        ]


def chunk_supported_entity_seeds(vector_hits):
    chunk_ids = [
        item["chunk_id"]
        for item in vector_hits
        if item.get("chunk_id")
    ]

    if not chunk_ids:
        return []

    cypher = '''
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

    ORDER BY support_count DESC, name
    LIMIT $limit
    '''

    with driver.session(database=NEO4J_DATABASE) as session:
        return [
            dict(row)
            for row in session.run(
                cypher,
                chunk_ids=chunk_ids,
                limit=MAX_GRAPH_SEEDS,
            )
        ]


def merge_seeds(lexical_seeds, bridge_seeds):
    merged = []
    seen = set()

    for seed in lexical_seeds + bridge_seeds:
        key = seed.get("key")

        if not key or key in seen:
            continue

        seen.add(key)
        merged.append(seed)

        if len(merged) >= MAX_GRAPH_SEEDS:
            break

    return merged


def graph_expand(seeds):
    seed_keys = [
        item["key"]
        for item in seeds
        if item.get("key")
    ]

    if not seed_keys:
        return []

    cypher = '''
    UNWIND $seed_keys AS seed_key

    MATCH
        (seed:KnowledgeEntity {key: seed_key})
        -[r]-
        (neighbor:KnowledgeEntity)

    WHERE type(r) IN $allowed_rel_types

    WITH DISTINCT seed, r, neighbor

    RETURN
        startNode(r).key AS source_key,
        startNode(r).name AS source,
        startNode(r).entity_type AS source_type,
        type(r) AS relation,
        endNode(r).key AS target_key,
        endNode(r).name AS target,
        endNode(r).entity_type AS target_type,
        coalesce(r.source_files, []) AS source_files,
        coalesce(r.evidence_pages, []) AS evidence_pages,
        coalesce(r.evidence_chunk_ids, []) AS evidence_chunk_ids,
        coalesce(r.evidence_count, 0) AS evidence_count

    ORDER BY
        evidence_count DESC,
        source,
        relation,
        target

    LIMIT $limit
    '''

    with driver.session(database=NEO4J_DATABASE) as session:
        return [
            dict(row)
            for row in session.run(
                cypher,
                seed_keys=seed_keys,
                allowed_rel_types=SEMANTIC_REL_TYPES,
                limit=MAX_GRAPH_FACTS,
            )
        ]


def paper_introductions(seeds):
    seed_keys = [
        item["key"]
        for item in seeds
        if item.get("key")
    ]

    if not seed_keys:
        return []

    cypher = '''
    UNWIND $seed_keys AS seed_key

    MATCH
        (p:Paper)
        -[:INTRODUCES]->
        (e:KnowledgeEntity {key: seed_key})

    RETURN DISTINCT
        p.source_file AS paper,
        e.name AS entity

    ORDER BY paper, entity
    '''

    with driver.session(database=NEO4J_DATABASE) as session:
        return [
            dict(row)
            for row in session.run(
                cypher,
                seed_keys=seed_keys,
            )
        ]


def graph_evidence_chunks(graph_facts):
    chunk_ids = []
    seen = set()

    for fact in graph_facts:
        for chunk_id in fact.get("evidence_chunk_ids", []):
            if not chunk_id or chunk_id in seen:
                continue

            seen.add(chunk_id)
            chunk_ids.append(chunk_id)

            if len(chunk_ids) >= MAX_GRAPH_EVIDENCE_CHUNKS:
                break

        if len(chunk_ids) >= MAX_GRAPH_EVIDENCE_CHUNKS:
            break

    if not chunk_ids:
        return []

    cypher = '''
    UNWIND $chunk_ids AS chunk_id

    MATCH (c:Chunk {chunk_id: chunk_id})

    RETURN
        c.chunk_id AS chunk_id,
        c.source_file AS source_file,
        c.page AS page,
        c.text AS text
    '''

    with driver.session(database=NEO4J_DATABASE) as session:
        rows = session.run(
            cypher,
            chunk_ids=chunk_ids,
        )

        by_id = {
            row["chunk_id"]: dict(row)
            for row in rows
        }

    return [
        by_id[chunk_id]
        for chunk_id in chunk_ids
        if chunk_id in by_id
    ]


def format_vector_evidence(vector_hits):
    blocks = []

    for index, hit in enumerate(vector_hits, start=1):
        blocks.append(
            f'''[VECTOR {index}]
Paper: {hit["source_file"]}
Page: {hit["page"]}
Score: {hit["score"]:.4f}
Chunk ID: {hit["chunk_id"]}

Text:
{hit["text"]}'''
        )

    return "\n\n".join(blocks)


def format_graph_facts(graph_facts):
    blocks = []

    for index, fact in enumerate(graph_facts, start=1):
        blocks.append(
            f'''[GRAPH {index}]
{fact["source"]} --{fact["relation"]}--> {fact["target"]}
Source files: {fact.get("source_files", [])}
Pages: {fact.get("evidence_pages", [])}
Evidence chunks: {fact.get("evidence_chunk_ids", [])}'''
        )

    return "\n\n".join(blocks)


def format_graph_chunks(graph_chunks):
    blocks = []

    for index, item in enumerate(graph_chunks, start=1):
        blocks.append(
            f'''[GRAPH-EVIDENCE {index}]
Paper: {item["source_file"]}
Page: {item["page"]}
Chunk ID: {item["chunk_id"]}

Text:
{item["text"]}'''
        )

    return "\n\n".join(blocks)


def build_answer_prompt(
    question,
    vector_hits,
    seeds,
    graph_facts,
    introductions,
    graph_chunks,
):
    seed_text = json.dumps(
        [
            {
                "name": seed.get("name"),
                "type": seed.get("entity_type"),
                "source": seed.get("seed_source"),
            }
            for seed in seeds
        ],
        ensure_ascii=False,
        indent=2,
    )

    intro_text = json.dumps(
        introductions,
        ensure_ascii=False,
        indent=2,
    )

    return f'''
You are answering a technical research question using a HYBRID
GraphRAG evidence set.

Use ONLY the supplied evidence.

Do not use outside knowledge.
Do not invent facts, equations, datasets, results, or relationships.

The evidence comes from:
1. VECTOR EVIDENCE: semantically relevant paper chunks.
2. GRAPH EVIDENCE: semantic relations from Neo4j.
3. GRAPH FACT PROVENANCE: actual paper chunks supporting graph facts.

Rules:
- Prefer direct paper text for technical explanation.
- Use graph facts to connect methods, concepts, tasks, datasets,
  metrics, and comparison methods.
- If vector and graph evidence disagree, prefer direct paper text
  and mention the discrepancy.
- If evidence is insufficient, say so.
- Cite factual technical statements as:
  [Paper filename, p.X]
- Do not cite a page unless supplied evidence supports that page.
- Answer the user's question directly and concisely.

============================================================
QUESTION
============================================================

{question}

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
'''


def generate_answer(
    question,
    vector_hits,
    seeds,
    graph_facts,
    introductions,
    graph_chunks,
):
    prompt = build_answer_prompt(
        question,
        vector_hits,
        seeds,
        graph_facts,
        introductions,
        graph_chunks,
    )

    response = gemini.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
    )

    return response.output_text


def print_vector_hits(vector_hits):
    print("\n" + "=" * 70)
    print("VECTOR RETRIEVAL")
    print("=" * 70)

    for index, hit in enumerate(vector_hits, start=1):
        preview = hit["text"].replace("\n", " ")[:180]

        print(
            f'{index}. {hit["source_file"]} '
            f'| p.{hit["page"]} '
            f'| score={hit["score"]:.4f}'
        )
        print(f"   {preview}...")


def print_seeds(seeds):
    print("\n" + "=" * 70)
    print("GRAPH SEEDS")
    print("=" * 70)

    if not seeds:
        print("No graph seed entities found.")
        return

    for index, seed in enumerate(seeds, start=1):
        print(
            f'{index}. {seed["name"]} '
            f'[{seed["entity_type"]}] '
            f'<- {seed["seed_source"]}'
        )


def print_graph_facts(graph_facts):
    print("\n" + "=" * 70)
    print("GRAPH TRAVERSAL")
    print("=" * 70)

    if not graph_facts:
        print("No semantic graph facts found.")
        return

    for index, fact in enumerate(graph_facts, start=1):
        print(
            f'{index}. {fact["source"]} '
            f'--{fact["relation"]}--> '
            f'{fact["target"]}'
        )


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
        print("Question is empty. Exiting.")
        return

    print("\n" + "=" * 70)
    print("HYBRID GRAPHRAG")
    print("=" * 70)
    print(f"\nQuestion:\n{question}")

    vector_hits = vector_retrieve(
        question,
        args.top_k,
    )
    print_vector_hits(vector_hits)

    lexical_seeds = lexical_entity_seeds(question)
    bridge_seeds = chunk_supported_entity_seeds(
        vector_hits
    )

    seeds = merge_seeds(
        lexical_seeds,
        bridge_seeds,
    )
    print_seeds(seeds)

    graph_facts = graph_expand(seeds)
    introductions = paper_introductions(seeds)
    print_graph_facts(graph_facts)

    graph_chunks = graph_evidence_chunks(
        graph_facts
    )

    print(
        "\nGraph provenance chunks recovered:",
        len(graph_chunks),
    )

    print("\n" + "=" * 70)
    print("GENERATING GROUNDED ANSWER")
    print("=" * 70)

    answer = generate_answer(
        question,
        vector_hits,
        seeds,
        graph_facts,
        introductions,
        graph_chunks,
    )

    print("\n" + "=" * 70)
    print("HYBRID GRAPHRAG ANSWER")
    print("=" * 70)
    print(answer)


if __name__ == "__main__":
    try:
        main()
    finally:
        driver.close()
