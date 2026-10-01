import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


# ----------------------------------------
# 1. 找到项目根目录中的 .env
# ----------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


# ----------------------------------------
# 2. 读取 Neo4j credentials
# ----------------------------------------
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
database = os.getenv("NEO4J_DATABASE", "neo4j")


# ----------------------------------------
# 3. 检查环境变量是否存在
# ----------------------------------------
if not uri:
    raise ValueError("NEO4J_URI is missing from .env")

if not username:
    raise ValueError("NEO4J_USERNAME is missing from .env")

if not password:
    raise ValueError("NEO4J_PASSWORD is missing from .env")


# ----------------------------------------
# 4. 创建 Neo4j Driver
# ----------------------------------------
driver = GraphDatabase.driver(
    uri,
    auth=(username, password)
)


# ----------------------------------------
# 5. 验证连接
# ----------------------------------------
driver.verify_connectivity()

print("Neo4j connection OK")


# ----------------------------------------
# 6. 执行第一条 Cypher
# ----------------------------------------
with driver.session(database=database) as session:

    result = session.run(
        "RETURN 'Hello from Neo4j!' AS message"
    )

    record = result.single()

    print(record["message"])


# ----------------------------------------
# 7. 关闭连接
# ----------------------------------------
driver.close()