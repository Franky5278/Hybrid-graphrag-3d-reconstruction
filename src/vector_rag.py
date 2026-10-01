import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# 1. Load configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not NEO4J_URI:
    raise ValueError("NEO4J_URI is missing")

if not NEO4J_USERNAME:
    raise ValueError("NEO4J_USERNAME is missing")

if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD is missing")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing")


# ============================================================
# 2. Connect to Neo4j
# ============================================================

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(
        NEO4J_USERNAME,
        NEO4J_PASSWORD
    )
)

driver.verify_connectivity()

print("Connected to Neo4j")


# ============================================================
# 3. Load embedding model
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded")


# ============================================================
# 4. Create Gemini client
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# 5. User question
# ============================================================

question = (
    "How does LHM represent a human using Gaussian primitives?"
)

print("\nQuestion:")
print(question)


# ============================================================
# 6. Embed the question
# ============================================================

query_embedding = embedding_model.encode(
    question,
    normalize_embeddings=True
)


# ============================================================
# 7. Retrieve relevant chunks from Neo4j
# ============================================================

vector_query = """
MATCH (c:Chunk)

SEARCH c IN (
    VECTOR INDEX chunk_embeddings
    FOR $query_embedding
    LIMIT 3
)

SCORE AS score

RETURN
    c.id AS id,
    c.text AS text,
    score

ORDER BY score DESC
"""


retrieved_chunks = []


with driver.session(database=NEO4J_DATABASE) as session:

    result = session.run(
        vector_query,
        query_embedding=query_embedding.tolist()
    )

    for record in result:

        retrieved_chunks.append(
            {
                "id": record["id"],
                "text": record["text"],
                "score": record["score"]
            }
        )


# ============================================================
# 8. Display retrieved context
# ============================================================

print("\nRetrieved context:\n")


for rank, chunk in enumerate(
    retrieved_chunks,
    start=1
):

    print(
        f"{rank}. "
        f"{chunk['id']} "
        f"(score={chunk['score']:.4f})"
    )

    print(
        f"   {chunk['text']}"
    )

    print()


# ============================================================
# 9. Build context for Gemini
# ============================================================

context = "\n".join(
    [
        f"- {chunk['text']}"
        for chunk in retrieved_chunks
    ]
)


prompt = f"""
You are answering a question using retrieved context.

Use only the information contained in the context below.

If the context does not contain enough information,
say that the provided context is insufficient.

Context:
{context}

Question:
{question}

Answer concisely and clearly.
"""


# ============================================================
# 10. Generate grounded answer with Gemini
# ============================================================

interaction = gemini_client.interactions.create(
    model="gemini-3.5-flash-lite",
    input=prompt
)


# ============================================================
# 11. Print final answer
# ============================================================

print("\nRAG Answer:\n")

print(
    interaction.output_text
)


# ============================================================
# 12. Close Neo4j
# ============================================================

driver.close()