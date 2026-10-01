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


# ============================================================
# 2. Connect to services
# ============================================================

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

driver.verify_connectivity()

print("Connected to Neo4j")


gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded")


# ============================================================
# 3. Question
# ============================================================

question = (
    "How does LHM represent a human using Gaussian primitives?"
)

print("\n" + "=" * 70)
print("QUESTION")
print("=" * 70)
print(question)


# ============================================================
# Helper: Ask Gemini
# ============================================================

def ask_gemini(prompt):
    interaction = gemini_client.interactions.create(
        model="gemini-3.5-flash-lite",
        input=prompt
    )

    return interaction.output_text


# ============================================================
# 4. METHOD A — LLM ONLY
# ============================================================

llm_only_prompt = f"""
Answer the following question concisely.

Question:
{question}
"""

llm_only_answer = ask_gemini(
    llm_only_prompt
)


print("\n" + "=" * 70)
print("A. LLM ONLY")
print("=" * 70)

print(llm_only_answer)


# ============================================================
# 5. METHOD B — VECTOR RETRIEVAL
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


with driver.session(database=NEO4J_DATABASE) as session:

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


vector_context = "\n".join(
    [
        f"- {chunk['text']}"
        for chunk in vector_chunks
    ]
)


vector_prompt = f"""
Answer the question using only the provided context.

If the context is insufficient, say so.

Context:
{vector_context}

Question:
{question}
"""


vector_answer = ask_gemini(
    vector_prompt
)


print("\n" + "=" * 70)
print("B. VECTOR RAG")
print("=" * 70)

print("\nRetrieved vector context:")

for chunk in vector_chunks:

    print(
        f"- {chunk['id']} "
        f"(score={chunk['score']:.4f})"
    )

    print(
        f"  {chunk['text']}"
    )


print("\nAnswer:")
print(vector_answer)


# ============================================================
# 6. METHOD C — GRAPH RETRIEVAL
# ============================================================

# For this teaching example, we already know
# the question is about the entity "LHM".
#
# Later we will automatically extract this entity
# from the user's natural-language question.

paper_name = "LHM"


graph_query = """
MATCH (paper:Paper)-[r]->(target)

WHERE paper.name = $paper_name

RETURN
    paper.name AS source,
    type(r) AS relationship,
    target.name AS target,
    labels(target) AS labels
"""


graph_facts = []


with driver.session(database=NEO4J_DATABASE) as session:

    result = session.run(
        graph_query,
        paper_name=paper_name
    )

    for record in result:

        graph_facts.append(
            {
                "source": record["source"],
                "relationship": record["relationship"],
                "target": record["target"],
                "labels": record["labels"]
            }
        )


graph_context = "\n".join(
    [
        (
            f"- {fact['source']} "
            f"--{fact['relationship']}--> "
            f"{fact['target']}"
        )
        for fact in graph_facts
    ]
)


graph_prompt = f"""
Answer the question using only the graph facts below.

Do not add information that is not contained in the graph facts.

If the graph facts are insufficient,
say that the graph contains insufficient information.

Graph facts:
{graph_context}

Question:
{question}
"""


graph_answer = ask_gemini(
    graph_prompt
)


print("\n" + "=" * 70)
print("C. GRAPH RETRIEVAL RAG")
print("=" * 70)

print("\nRetrieved graph facts:")

for fact in graph_facts:

    print(
        f"- {fact['source']} "
        f"--{fact['relationship']}--> "
        f"{fact['target']} "
        f"{fact['labels']}"
    )


print("\nAnswer:")
print(graph_answer)


# ============================================================
# 7. Close connection
# ============================================================

driver.close()