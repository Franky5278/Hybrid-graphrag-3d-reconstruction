import os
import time
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
# 4. Demo text chunks
# ============================================================

chunks = [
    {
        "id": "chunk_1",
        "text": "LHM uses 3D Gaussian Splatting for human reconstruction."
    },
    {
        "id": "chunk_2",
        "text": "LHM uses SMPL-X as part of the human representation."
    },
    {
        "id": "chunk_3",
        "text": "Neo4j is a graph database that stores nodes and relationships."
    },
    {
        "id": "chunk_4",
        "text": "Vector search retrieves text by comparing semantic embeddings."
    },
]


# ============================================================
# 5. Generate embeddings
# ============================================================

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

print("Embedding shape:", embeddings.shape)


# ============================================================
# 6. Store Chunk nodes in Neo4j
# ============================================================

write_query = """
MERGE (c:Chunk {id: $id})
SET
    c.text = $text,
    c.embedding = $embedding
"""


with driver.session(database=database) as session:

    for chunk, embedding in zip(chunks, embeddings):

        session.run(
            write_query,
            id=chunk["id"],
            text=chunk["text"],
            embedding=embedding.tolist()
        )

        print("Stored:", chunk["id"])


print("All chunks stored")


# ============================================================
# 7. Create Neo4j vector index
# ============================================================

index_query = """
CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS

FOR (c:Chunk)

ON c.embedding

OPTIONS {
    indexConfig: {
        `vector.dimensions`: 384,
        `vector.similarity_function`: 'cosine'
    }
}
"""


with driver.session(database=database) as session:
    session.run(index_query).consume()


print("Vector index creation requested")


# ============================================================
# 8. Wait until vector index becomes ONLINE
# ============================================================

for attempt in range(30):

    with driver.session(database=database) as session:

        result = session.run(
            """
            SHOW VECTOR INDEXES
            YIELD name, state, populationPercent
            WHERE name = 'chunk_embeddings'
            RETURN name, state, populationPercent
            """
        )

        record = result.single()

    if record is None:
        print("Vector index not found yet")

    else:
        print(
            "Index:",
            record["name"],
            "| State:",
            record["state"],
            "| Population:",
            record["populationPercent"]
        )

        if record["state"] == "ONLINE":
            print("Vector index is ONLINE!")
            break

    time.sleep(1)

else:
    print("Vector index is still not ONLINE after waiting.")


# ============================================================
# 9. Close Neo4j connection
# ============================================================

driver.close()