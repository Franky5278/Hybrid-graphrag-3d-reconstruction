import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "paper_chunks.jsonl"
)

load_dotenv(
    PROJECT_ROOT / ".env"
)


# ============================================================
# 2. Configuration
# ============================================================

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

NEO4J_DATABASE = os.getenv(
    "NEO4J_DATABASE",
    "neo4j"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

VECTOR_INDEX_NAME = (
    "paper_chunk_embeddings"
)

EMBEDDING_DIMENSION = 384

# Local embedding batch size
EMBED_BATCH_SIZE = 32

# Number of chunks written to Neo4j per transaction
NEO4J_WRITE_BATCH_SIZE = 50


# ============================================================
# 3. Validate configuration
# ============================================================

if not CHUNKS_FILE.exists():
    raise FileNotFoundError(
        f"Chunk file not found:\n{CHUNKS_FILE}"
    )

if not NEO4J_URI:
    raise ValueError(
        "NEO4J_URI is missing from .env"
    )

if not NEO4J_USERNAME:
    raise ValueError(
        "NEO4J_USERNAME is missing from .env"
    )

if not NEO4J_PASSWORD:
    raise ValueError(
        "NEO4J_PASSWORD is missing from .env"
    )


# ============================================================
# 4. Load all chunks from JSONL
# ============================================================

print("=" * 70)
print("LOADING REAL PAPER CHUNKS")
print("=" * 70)


chunks = []


with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        line = line.strip()

        if not line:
            continue

        chunk = json.loads(
            line
        )

        chunks.append(
            chunk
        )


print(
    f"\nLoaded {len(chunks)} chunks."
)


paper_ids = {
    chunk["paper_id"]
    for chunk in chunks
}


print(
    f"Papers: {len(paper_ids)}"
)


# ============================================================
# 5. Load embedding model
# ============================================================

print("\n")
print("=" * 70)
print("LOADING EMBEDDING MODEL")
print("=" * 70)


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


print(
    "Embedding model loaded."
)


# ============================================================
# 6. Generate embeddings
# ============================================================

print("\n")
print("=" * 70)
print("GENERATING EMBEDDINGS")
print("=" * 70)


texts = [
    chunk["text"]
    for chunk in chunks
]


embeddings = embedding_model.encode(
    texts,
    batch_size=EMBED_BATCH_SIZE,
    normalize_embeddings=True,
    show_progress_bar=True
)


print(
    "\nEmbedding matrix shape:",
    embeddings.shape
)


if embeddings.shape[1] != EMBEDDING_DIMENSION:

    raise ValueError(
        "Unexpected embedding dimension: "
        f"{embeddings.shape[1]}"
    )


# ============================================================
# 7. Attach embedding vector to each chunk
# ============================================================

for chunk, embedding in zip(
    chunks,
    embeddings
):

    chunk["embedding"] = (
        embedding.tolist()
    )


print(
    "Embeddings generated successfully."
)


# ============================================================
# 8. Connect to Neo4j
# ============================================================

print("\n")
print("=" * 70)
print("CONNECTING TO NEO4J")
print("=" * 70)


driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(
        NEO4J_USERNAME,
        NEO4J_PASSWORD
    )
)


driver.verify_connectivity()


print(
    "Connected to Neo4j."
)


# ============================================================
# 9. Delete previous toy Chunk nodes
# ============================================================

print("\n")
print("=" * 70)
print("CLEANING OLD TOY CHUNKS")
print("=" * 70)


with driver.session(
    database=NEO4J_DATABASE
) as session:

    session.run(
        """
        MATCH (c:Chunk)
        DETACH DELETE c
        """
    ).consume()


print(
    "Old Chunk nodes deleted."
)


# ============================================================
# 10. Drop previous toy vector index
# ============================================================

with driver.session(
    database=NEO4J_DATABASE
) as session:

    session.run(
        """
        DROP INDEX chunk_embeddings
        IF EXISTS
        """
    ).consume()


print(
    "Old toy vector index removed."
)


# ============================================================
# 11. Optional:
# Drop real-paper index if it already exists
#
# This makes the script safe to rerun while developing.
# ============================================================

with driver.session(
    database=NEO4J_DATABASE
) as session:

    session.run(
        f"""
        DROP INDEX {VECTOR_INDEX_NAME}
        IF EXISTS
        """
    ).consume()


print(
    "Previous real-paper vector index removed if present."
)


# ============================================================
# 12. Write Paper and Chunk nodes
# ============================================================

print("\n")
print("=" * 70)
print("WRITING REAL PAPER GRAPH")
print("=" * 70)


write_query = """
UNWIND $rows AS row


MERGE (paper:Paper {
    paper_id: row.paper_id
})

SET
    paper.source_file = row.source_file


MERGE (chunk:Chunk {
    chunk_id: row.chunk_id
})

SET
    chunk.paper_id = row.paper_id,
    chunk.source_file = row.source_file,
    chunk.page = row.page,
    chunk.chunk_index = row.chunk_index,
    chunk.token_start = row.token_start,
    chunk.token_end = row.token_end,
    chunk.token_count = row.token_count,
    chunk.text = row.text,
    chunk.embedding = row.embedding


MERGE (paper)-[:HAS_CHUNK]->(chunk)
"""


total_batches = (
    len(chunks)
    + NEO4J_WRITE_BATCH_SIZE
    - 1
) // NEO4J_WRITE_BATCH_SIZE


with driver.session(
    database=NEO4J_DATABASE
) as session:

    for batch_number, start in enumerate(
        range(
            0,
            len(chunks),
            NEO4J_WRITE_BATCH_SIZE
        ),
        start=1
    ):

        batch = chunks[
            start:
            start + NEO4J_WRITE_BATCH_SIZE
        ]


        session.run(
            write_query,
            rows=batch
        ).consume()


        print(
            f"Batch "
            f"{batch_number}/{total_batches} "
            f"written "
            f"({len(batch)} chunks)"
        )


print(
    "\nAll real chunks written."
)


# ============================================================
# 13. Create vector index
# ============================================================

print("\n")
print("=" * 70)
print("CREATING VECTOR INDEX")
print("=" * 70)


create_index_query = f"""
CREATE VECTOR INDEX {VECTOR_INDEX_NAME}
IF NOT EXISTS

FOR (c:Chunk)

ON c.embedding

OPTIONS {{
    indexConfig: {{
        `vector.dimensions`:
            {EMBEDDING_DIMENSION},

        `vector.similarity_function`:
            'cosine'
    }}
}}
"""


with driver.session(
    database=NEO4J_DATABASE
) as session:

    session.run(
        create_index_query
    ).consume()


print(
    "Vector index creation requested."
)


# ============================================================
# 14. Wait for vector index to become ONLINE
# ============================================================

print(
    "\nWaiting for vector index..."
)


for attempt in range(60):

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        result = session.run(
            """
            SHOW VECTOR INDEXES

            YIELD
                name,
                state,
                populationPercent

            WHERE name = $index_name

            RETURN
                name,
                state,
                populationPercent
            """,
            index_name=VECTOR_INDEX_NAME
        )


        record = result.single()


    if record is not None:

        print(
            f"Index: "
            f"{record['name']} "
            f"| State: "
            f"{record['state']} "
            f"| Population: "
            f"{record['populationPercent']}"
        )


        if record["state"] == "ONLINE":

            print(
                "\nVector index is ONLINE!"
            )

            break


    time.sleep(1)


else:

    print(
        "\nWARNING:"
        " Vector index did not become "
        "ONLINE within 60 seconds."
    )


# ============================================================
# 15. Verify Neo4j graph
# ============================================================

print("\n")
print("=" * 70)
print("DATABASE VERIFICATION")
print("=" * 70)


with driver.session(
    database=NEO4J_DATABASE
) as session:

    record = session.run(
        """
        MATCH (p:Paper)-[:HAS_CHUNK]->(c:Chunk)

        RETURN
            count(DISTINCT p) AS papers,
            count(DISTINCT c) AS chunks
        """
    ).single()


print(
    "Paper nodes:",
    record["papers"]
)

print(
    "Chunk nodes:",
    record["chunks"]
)


# ============================================================
# 16. Preview a few stored chunks
# ============================================================

print("\n")
print("=" * 70)
print("DATABASE PREVIEW")
print("=" * 70)


with driver.session(
    database=NEO4J_DATABASE
) as session:

    result = session.run(
        """
        MATCH (p:Paper)-[:HAS_CHUNK]->(c:Chunk)

        RETURN
            p.source_file AS paper,
            c.page AS page,
            c.chunk_id AS chunk_id,
            c.text AS text

        ORDER BY
            paper,
            page,
            chunk_id

        LIMIT 3
        """
    )


    for record in result:

        print("\nPaper:")
        print(
            record["paper"]
        )

        print(
            "Page:",
            record["page"]
        )

        print(
            "Chunk:",
            record["chunk_id"]
        )

        print("\nText:")

        print(
            record["text"][:500]
        )

        print(
            "\n" + "-" * 70
        )


# ============================================================
# 17. Close connection
# ============================================================

driver.close()


print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    "Real paper corpus indexed successfully."
)