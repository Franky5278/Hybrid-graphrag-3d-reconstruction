import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


# ============================================================
# 1. Load configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
database = os.getenv("NEO4J_DATABASE", "neo4j")


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
# 3. Create nodes and relationships
# ============================================================

query = """
MERGE (paper:Paper {name: $paper_name})

MERGE (smplx:Method {name: $smplx_name})

MERGE (gs:Method {name: $gs_name})

MERGE (task:Task {name: $task_name})

MERGE (paper)-[:USES]->(smplx)

MERGE (paper)-[:USES]->(gs)

MERGE (paper)-[:ADDRESSES]->(task)

RETURN paper, smplx, gs, task
"""


parameters = {
    "paper_name": "LHM",
    "smplx_name": "SMPL-X",
    "gs_name": "3D Gaussian Splatting",
    "task_name": "Single-Image Human Reconstruction",
}


with driver.session(database=database) as session:

    result = session.run(
        query,
        parameters
    )

    record = result.single()

    print("Knowledge graph created successfully!")

    print("Paper:", record["paper"]["name"])
    print("Method 1:", record["smplx"]["name"])
    print("Method 2:", record["gs"]["name"])
    print("Task:", record["task"]["name"])


driver.close()