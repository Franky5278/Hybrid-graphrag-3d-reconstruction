import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Load configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
database = os.getenv("NEO4J_DATABASE", "neo4j")


if not uri:
    raise ValueError("NEO4J_URI is missing")

if not username:
    raise ValueError("NEO4J_USERNAME is missing")

if not password:
    raise ValueError("NEO4J_PASSWORD is missing")


# ============================================================
# 2. Connect to Neo4j
# ============================================================

driver = GraphDatabase.driver(
    uri,
    auth=(username, password)
)

driver.verify_connectivity()

print("Connected to Neo4j")


# ============================================================
# 3. Load embedding model
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded")


# ============================================================
# 4. User query
# ============================================================

question = (
    "How does LHM represent a human using Gaussian primitives?"
)

print("\nQuestion:")
print(question)


# ============================================================
# 5. Convert question into embedding
# ============================================================

query_embedding = model.encode(
    question,
    normalize_embeddings=True
)

print("\nQuery embedding dimension:")
print(len(query_embedding))


# ============================================================
# 6. Semantic vector search in Neo4j
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


# ============================================================
# 7. Run query
# ============================================================

with driver.session(database=database) as session:

    result = session.run(
        vector_query,
        query_embedding=query_embedding.tolist()
    )

    print("\nTop semantic matches:\n")

    rank = 1

    for record in result:

        print(
            f"{rank}. {record['id']}"
        )

        print(
            f"   Score: {record['score']:.4f}"
        )

        print(
            f"   Text: {record['text']}"
        )

        print()

        rank += 1


# ============================================================
# 8. Close connection
# ============================================================

driver.close()