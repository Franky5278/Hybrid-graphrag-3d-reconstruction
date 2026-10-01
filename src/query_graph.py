import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


# ============================================================
# 1. Load .env
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


# ============================================================
# 2. Read Neo4j configuration
# ============================================================

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
database = os.getenv("NEO4J_DATABASE", "neo4j")


# ============================================================
# 3. Check configuration
# ============================================================

if not uri:
    raise ValueError("NEO4J_URI is missing")

if not username:
    raise ValueError("NEO4J_USERNAME is missing")

if not password:
    raise ValueError("NEO4J_PASSWORD is missing")


# ============================================================
# 4. Connect to Neo4j
# ============================================================

driver = GraphDatabase.driver(
    uri,
    auth=(username, password)
)


# ============================================================
# 5. Cypher query
# ============================================================

query = """
MATCH (paper:Paper)-[relationship]->(target)
WHERE paper.name = $paper_name

RETURN
    paper.name AS paper,
    type(relationship) AS relationship,
    labels(target) AS target_labels,
    target.name AS target
"""


# ============================================================
# 6. Run query
# ============================================================

with driver.session(database=database) as session:

    result = session.run(
        query,
        paper_name="LHM"
    )

    print("\nRelationships of LHM:\n")

    found = False

    for record in result:

        found = True

        print(
            record["paper"],
            "--",
            record["relationship"],
            "-->",
            record["target"],
            record["target_labels"]
        )

    if not found:
        print("No relationships found.")


# ============================================================
# 7. Close connection
# ============================================================

driver.close()