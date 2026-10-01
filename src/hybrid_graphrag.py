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
# 4. Gemini client
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


def ask_gemini(prompt):

    interaction = gemini_client.interactions.create(
        model="gemini-3.5-flash-lite",
        input=prompt
    )

    return interaction.output_text.strip()


# ============================================================
# 5. User question
# ============================================================

question = (
    "How does LHM represent a human using Gaussian primitives?"
)

print("\n" + "=" * 70)
print("QUESTION")
print("=" * 70)

print(question)


# ============================================================
# 6. Automatically extract graph entity
# ============================================================

entity_prompt = f"""
Identify the main named entity in the question.

Return ONLY the entity name.

Do not explain anything.

If no named entity is present, return:
NONE

Question:
{question}
"""


entity_name = ask_gemini(
    entity_prompt
)

print("\nDetected entity:")
print(entity_name)


# ============================================================
# 7. VECTOR RETRIEVAL
# ============================================================

query_embedding = embedding_model.encode(
    question,
    normalize_embeddings=True
)


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


vector_chunks = []


with driver.session(
    database=NEO4J_DATABASE
) as session:

    result = session.run(
        vector_query,
        query_embedding=query_embedding.tolist()
    )

    for record in result:

        vector_chunks.append(
            {
                "id": record["id"],
                "text": record["text"],
                "score": record["score"]
            }
        )


print("\n" + "=" * 70)
print("VECTOR RETRIEVAL")
print("=" * 70)


for chunk in vector_chunks:

    print(
        f"{chunk['id']} "
        f"(score={chunk['score']:.4f})"
    )

    print(chunk["text"])
    print()


# ============================================================
# 8. GRAPH RETRIEVAL
# ============================================================

graph_query = """
MATCH (source)-[r]->(target)

WHERE toLower(source.name)
      = toLower($entity_name)

RETURN
    source.name AS source,
    labels(source) AS source_labels,
    type(r) AS relationship,
    target.name AS target,
    labels(target) AS target_labels
"""


graph_facts = []


if entity_name != "NONE":

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        result = session.run(
            graph_query,
            entity_name=entity_name
        )

        for record in result:

            graph_facts.append(
                {
                    "source": record["source"],
                    "relationship": record["relationship"],
                    "target": record["target"],
                    "source_labels": record["source_labels"],
                    "target_labels": record["target_labels"],
                }
            )


print("\n" + "=" * 70)
print("GRAPH RETRIEVAL")
print("=" * 70)


if graph_facts:

    for fact in graph_facts:

        print(
            f"{fact['source']} "
            f"--{fact['relationship']}--> "
            f"{fact['target']}"
        )

else:

    print("No graph facts found.")


# ============================================================
# 9. Build vector context
# ============================================================

vector_context = "\n".join(
    [
        (
            f"[Vector chunk {chunk['id']}, "
            f"score={chunk['score']:.4f}]\n"
            f"{chunk['text']}"
        )
        for chunk in vector_chunks
    ]
)


# ============================================================
# 10. Build graph context
# ============================================================

graph_context = "\n".join(
    [
        (
            f"[Graph fact]\n"
            f"{fact['source']} "
            f"--{fact['relationship']}--> "
            f"{fact['target']}"
        )
        for fact in graph_facts
    ]
)


if not graph_context:
    graph_context = "No graph facts were retrieved."


# ============================================================
# 11. Combine both retrieval channels
# ============================================================

hybrid_prompt = f"""
You are answering a research question using
retrieved evidence.

Use ONLY the evidence provided below.

Do not use outside knowledge.

Do not invent technical details.

If the evidence does not fully answer the question,
clearly state what can be concluded and what remains
unsupported.

========================
VECTOR EVIDENCE
========================

{vector_context}


========================
GRAPH EVIDENCE
========================

{graph_context}


========================
QUESTION
========================

{question}


Give a concise answer based strictly on the evidence.
"""


# ============================================================
# 12. Gemini generation
# ============================================================

answer = ask_gemini(
    hybrid_prompt
)


# ============================================================
# 13. Final output
# ============================================================

print("\n" + "=" * 70)
print("HYBRID GRAPHRAG ANSWER")
print("=" * 70)

print(answer)


# ============================================================
# 14. Close Neo4j
# ============================================================

driver.close()