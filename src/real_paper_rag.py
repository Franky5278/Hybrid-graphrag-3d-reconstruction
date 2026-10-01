import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# 1. Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(
    PROJECT_ROOT / ".env"
)


NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

NEO4J_DATABASE = os.getenv(
    "NEO4J_DATABASE",
    "neo4j"
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

VECTOR_INDEX_NAME = (
    "paper_chunk_embeddings"
)

TOP_K = 5


# ============================================================
# 2. Validate configuration
# ============================================================

if not NEO4J_URI:
    raise ValueError("NEO4J_URI is missing")

if not NEO4J_USERNAME:
    raise ValueError("NEO4J_USERNAME is missing")

if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD is missing")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing")


# ============================================================
# 3. Connect to Neo4j
# ============================================================

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(
        NEO4J_USERNAME,
        NEO4J_PASSWORD
    )
)

driver.verify_connectivity()

print("Neo4j connected.")


# ============================================================
# 4. Load embedding model
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# 5. Gemini client
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# 6. Research question
# ============================================================

question = (
    "How does VAD use Voronoi geometry "
    "for unsigned distance field reconstruction?"
)


print("\n")
print("=" * 70)
print("QUESTION")
print("=" * 70)

print(question)


# ============================================================
# 7. Convert question into embedding
# ============================================================

query_embedding = embedding_model.encode(
    question,
    normalize_embeddings=True
)


print(
    "\nQuery embedding dimension:",
    len(query_embedding)
)


# ============================================================
# 8. Search real paper chunks
# ============================================================

vector_query = f"""
MATCH (c:Chunk)

SEARCH c IN (
    VECTOR INDEX {VECTOR_INDEX_NAME}
    FOR $query_embedding
    LIMIT {TOP_K}
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


retrieved = []


with driver.session(
    database=NEO4J_DATABASE
) as session:

    result = session.run(
        vector_query,
        query_embedding=query_embedding.tolist()
    )


    for record in result:

        retrieved.append(
            {
                "chunk_id": record["chunk_id"],
                "paper_id": record["paper_id"],
                "source_file": record["source_file"],
                "page": record["page"],
                "text": record["text"],
                "score": record["score"]
            }
        )


# ============================================================
# 9. Print retrieval results
# ============================================================

print("\n")
print("=" * 70)
print("RETRIEVED REAL PAPER CHUNKS")
print("=" * 70)


if not retrieved:

    print(
        "No chunks retrieved."
    )

    driver.close()
    raise SystemExit


for rank, item in enumerate(
    retrieved,
    start=1
):

    print(
        f"\n[{rank}] "
        f"Score: {item['score']:.4f}"
    )

    print(
        "Paper:",
        item["source_file"]
    )

    print(
        "Page:",
        item["page"]
    )

    print(
        "Chunk:",
        item["chunk_id"]
    )

    print("\nText:")

    print(
        item["text"][:1000]
    )

    print(
        "\n" + "-" * 70
    )


# ============================================================
# 10. Build grounded context
# ============================================================

context_blocks = []


for rank, item in enumerate(
    retrieved,
    start=1
):

    block = f"""
SOURCE {rank}

Paper:
{item["source_file"]}

Page:
{item["page"]}

Chunk ID:
{item["chunk_id"]}

Similarity score:
{item["score"]:.4f}

Text:
{item["text"]}
"""

    context_blocks.append(
        block
    )


context = "\n".join(
    context_blocks
)


# ============================================================
# 11. Build RAG prompt
# ============================================================

prompt = f"""
You are answering a technical research question using
retrieved academic-paper evidence.

IMPORTANT RULES:

1. Use only the evidence supplied below.
2. Do not use outside knowledge.
3. Do not invent missing technical details.
4. Distinguish what is explicitly supported from what is
   not supported.
5. Cite supporting evidence using:

   [Paper filename, p. page_number]

6. If multiple papers provide relevant information,
   compare them when appropriate.
7. If the evidence is insufficient, explicitly say so.

============================================================
QUESTION
============================================================

{question}


============================================================
RETRIEVED EVIDENCE
============================================================

{context}


============================================================
ANSWER
============================================================

Provide a concise but technically clear answer.
"""


# ============================================================
# 12. Gemini grounded generation
# ============================================================

response = gemini_client.interactions.create(
    model="gemini-3.5-flash-lite",
    input=prompt
)


# ============================================================
# 13. Final answer
# ============================================================

print("\n")
print("=" * 70)
print("REAL PAPER RAG ANSWER")
print("=" * 70)

print(
    response.output_text
)


# ============================================================
# 14. Source summary
# ============================================================

print("\n")
print("=" * 70)
print("RETRIEVED SOURCES")
print("=" * 70)


for rank, item in enumerate(
    retrieved,
    start=1
):

    print(
        f"{rank}. "
        f"{item['source_file']} "
        f"| page {item['page']} "
        f"| score {item['score']:.4f}"
    )


# ============================================================
# 15. Close Neo4j
# ============================================================

driver.close()


print("\nDone.")