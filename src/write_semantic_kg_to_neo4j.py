import json
import os
import re
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KG_FILE = (
    PROJECT_ROOT
    / "data"
    / "kg"
    / "kg_canonical.json"
)

load_dotenv(
    PROJECT_ROOT / ".env"
)


# ============================================================
# 2. Environment
# ============================================================

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

NEO4J_DATABASE = os.getenv(
    "NEO4J_DATABASE",
    "neo4j"
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

if not KG_FILE.exists():
    raise FileNotFoundError(
        f"Canonical KG file not found:\n"
        f"{KG_FILE}"
    )


# ============================================================
# 3. Configuration
# ============================================================

# Rerunning this script should rebuild ONLY the semantic KG layer.
# Existing Paper / Chunk nodes and vector embeddings are preserved.
RESET_EXISTING_SEMANTIC_KG = True


ALLOWED_ENTITY_TYPES = {
    "Method",
    "Concept",
    "Task",
    "Dataset",
    "Metric",
    "Model",
}


ALLOWED_RELATION_TYPES = {
    "INTRODUCES",
    "USES",
    "ADDRESSES",
    "EVALUATES_ON",
    "MEASURED_BY",
    "IMPROVES",
    "COMPARED_WITH",
    "PRODUCES",
    "SOLVES",
}


# ============================================================
# 4. Helpers
# ============================================================

def make_key(
    entity_type,
    name,
):
    """
    Stable unique key for semantic entities.

    Example:
        Method::vad
        Concept::unsigned distance field
    """

    normalized_name = (
        str(name)
        .strip()
        .casefold()
    )

    return (
        f"{entity_type}::"
        f"{normalized_name}"
    )


def normalize_label(label):
    """
    Defensive validation for labels / relationship types.
    We only allow simple Neo4j identifier characters.
    """

    if not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*",
        label,
    ):
        raise ValueError(
            f"Unsafe Neo4j identifier: {label}"
        )

    return label


def evidence_lists(
    evidence,
):
    """
    Neo4j properties cannot contain lists of maps.

    Convert:
        [{"chunk_id": "...", "page": 3}, ...]

    into:
        evidence_chunk_ids = [...]
        evidence_pages = [...]
        evidence_json = "..."
    """

    chunk_ids = []
    pages = []

    seen_chunks = set()
    seen_pages = set()


    for item in evidence or []:
        if not isinstance(
            item,
            dict,
        ):
            continue


        chunk_id = item.get(
            "chunk_id"
        )

        page = item.get(
            "page"
        )


        if (
            chunk_id
            and chunk_id
            not in seen_chunks
        ):
            seen_chunks.add(
                chunk_id
            )

            chunk_ids.append(
                chunk_id
            )


        if (
            page is not None
            and page
            not in seen_pages
        ):
            seen_pages.add(
                page
            )

            pages.append(
                page
            )


    return {
        "chunk_ids":
            chunk_ids,

        "pages":
            pages,

        "json":
            json.dumps(
                evidence or [],
                ensure_ascii=False,
            ),
    }


# ============================================================
# 5. Load canonical KG
# ============================================================

with open(
    KG_FILE,
    "r",
    encoding="utf-8",
) as file:
    kg = json.load(
        file
    )


papers = kg.get(
    "papers",
    []
)

entities = kg.get(
    "entities",
    []
)

relations = kg.get(
    "relations",
    []
)


print("=" * 70)
print("WRITE SEMANTIC KG TO NEO4J")
print("=" * 70)

print(
    f"\nPapers in canonical KG: "
    f"{len(papers)}"
)

print(
    f"Canonical entities: "
    f"{len(entities)}"
)

print(
    f"Canonical relations: "
    f"{len(relations)}"
)


# ============================================================
# 6. Build entity lookup
# ============================================================

entity_lookup = {}


for entity in entities:
    entity_type = entity.get(
        "type"
    )

    name = entity.get(
        "name"
    )


    if (
        entity_type
        not in ALLOWED_ENTITY_TYPES
    ):
        continue


    if not name:
        continue


    key = make_key(
        entity_type,
        name,
    )


    entity[
        "_neo4j_key"
    ] = key


    entity_lookup[
        str(name)
        .strip()
        .casefold()
    ] = entity


    for alias in entity.get(
        "aliases",
        []
    ):
        entity_lookup[
            str(alias)
            .strip()
            .casefold()
        ] = entity


# ============================================================
# 7. Connect to Neo4j
# ============================================================

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(
        NEO4J_USERNAME,
        NEO4J_PASSWORD,
    ),
)


driver.verify_connectivity()


print(
    "\nConnected to Neo4j."
)


# ============================================================
# 8. Reset semantic layer only
# ============================================================

if RESET_EXISTING_SEMANTIC_KG:

    print("\n")
    print("=" * 70)
    print("RESETTING OLD SEMANTIC KG LAYER")
    print("=" * 70)


    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        result = session.run(
            """
            MATCH (e:KnowledgeEntity)

            WITH count(e) AS old_entities

            MATCH (e2:KnowledgeEntity)
            DETACH DELETE e2

            RETURN old_entities
            """
        )


        record = result.single()


    old_entities = (
        record[
            "old_entities"
        ]
        if record
        else 0
    )


    print(
        "Deleted old KnowledgeEntity nodes:",
        old_entities
    )


# ============================================================
# 9. Create uniqueness constraint
# ============================================================

with driver.session(
    database=NEO4J_DATABASE
) as session:

    session.run(
        """
        CREATE CONSTRAINT knowledge_entity_key_unique
        IF NOT EXISTS

        FOR (e:KnowledgeEntity)

        REQUIRE e.key IS UNIQUE
        """
    ).consume()


print(
    "KnowledgeEntity uniqueness constraint ready."
)


# ============================================================
# 10. Write semantic entities
# ============================================================

print("\n")
print("=" * 70)
print("WRITING SEMANTIC ENTITIES")
print("=" * 70)


entities_by_type = defaultdict(
    list
)


for entity in entities:

    entity_type = entity.get(
        "type"
    )

    name = entity.get(
        "name"
    )


    if (
        entity_type
        not in ALLOWED_ENTITY_TYPES
        or not name
    ):
        continue


    evidence = evidence_lists(
        entity.get(
            "evidence",
            []
        )
    )


    row = {
        "key":
            make_key(
                entity_type,
                name,
            ),

        "name":
            name,

        "entity_type":
            entity_type,

        "aliases":
            entity.get(
                "aliases",
                []
            ),

        "source_files":
            entity.get(
                "source_files",
                []
            ),

        "evidence_chunk_ids":
            evidence[
                "chunk_ids"
            ],

        "evidence_pages":
            evidence[
                "pages"
            ],

        "evidence_json":
            evidence[
                "json"
            ],
    }


    entities_by_type[
        entity_type
    ].append(
        row
    )


for entity_type in sorted(
    entities_by_type
):
    label = normalize_label(
        entity_type
    )

    rows = entities_by_type[
        entity_type
    ]


    query = f"""
    UNWIND $rows AS row

    MERGE (
        e:KnowledgeEntity:{label}
        {{
            key: row.key
        }}
    )

    SET
        e.name =
            row.name,

        e.entity_type =
            row.entity_type,

        e.aliases =
            row.aliases,

        e.source_files =
            row.source_files,

        e.evidence_chunk_ids =
            row.evidence_chunk_ids,

        e.evidence_pages =
            row.evidence_pages,

        e.evidence_json =
            row.evidence_json
    """


    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        session.run(
            query,
            rows=rows,
        ).consume()


    print(
        f"{entity_type}: "
        f"{len(rows)} node(s)"
    )


# ============================================================
# 11. Add entity -> Chunk provenance edges
# ============================================================

print("\n")
print("=" * 70)
print("WRITING ENTITY PROVENANCE")
print("=" * 70)


provenance_rows = []


for entity in entities:

    entity_type = entity.get(
        "type"
    )

    name = entity.get(
        "name"
    )


    if (
        entity_type
        not in ALLOWED_ENTITY_TYPES
        or not name
    ):
        continue


    key = make_key(
        entity_type,
        name,
    )


    for evidence in entity.get(
        "evidence",
        []
    ):
        chunk_id = evidence.get(
            "chunk_id"
        )

        page = evidence.get(
            "page"
        )


        if not chunk_id:
            continue


        provenance_rows.append(
            {
                "entity_key":
                    key,

                "chunk_id":
                    chunk_id,

                "page":
                    page,
            }
        )


if provenance_rows:

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        session.run(
            """
            UNWIND $rows AS row

            MATCH (
                e:KnowledgeEntity
                {
                    key:
                        row.entity_key
                }
            )

            MATCH (
                c:Chunk
                {
                    chunk_id:
                        row.chunk_id
                }
            )

            MERGE (
                e
            )-[r:SUPPORTED_BY]->(
                c
            )

            SET
                r.page =
                    row.page
            """,
            rows=provenance_rows,
        ).consume()


print(
    "Entity -> Chunk SUPPORTED_BY edges requested:",
    len(
        provenance_rows
    )
)


# ============================================================
# 12. Ensure every paper profile introduces its focal method
#
# Canonical extraction reported 5 INTRODUCES edges for 6 papers.
# The paper_profiles are therefore used as a reliable repair layer
# so every paper has a focal semantic entry point.
# ============================================================

print("\n")
print("=" * 70)
print("WRITING PAPER -> FOCAL METHOD EDGES")
print("=" * 70)


profile_intro_rows = []


for paper in papers:

    source_file = paper.get(
        "source_file"
    )


    for focal in paper.get(
        "focal_entities",
        []
    ):
        focal_name = focal.get(
            "name"
        )

        focal_type = focal.get(
            "type"
        )


        if (
            not source_file
            or not focal_name
            or focal_type
            not in {
                "Method",
                "Model",
            }
        ):
            continue


        # Resolve canonical entity if possible.
        target_entity = entity_lookup.get(
            str(
                focal_name
            )
            .strip()
            .casefold()
        )


        if target_entity:
            target_type = target_entity[
                "type"
            ]

            target_name = target_entity[
                "name"
            ]

        else:
            target_type = focal_type
            target_name = focal_name


        target_key = make_key(
            target_type,
            target_name,
        )


        profile_intro_rows.append(
            {
                "source_file":
                    source_file,

                "target_key":
                    target_key,
            }
        )


if profile_intro_rows:

    with driver.session(
        database=NEO4J_DATABASE
    ) as session:

        session.run(
            """
            UNWIND $rows AS row

            MATCH (
                p:Paper
                {
                    source_file:
                        row.source_file
                }
            )

            MATCH (
                e:KnowledgeEntity
                {
                    key:
                        row.target_key
                }
            )

            MERGE (
                p
            )-[r:INTRODUCES]->(
                e
            )

            SET
                r.generated_from_profile =
                    true
            """,
            rows=profile_intro_rows,
        ).consume()


print(
    "Paper -> focal INTRODUCES edges requested:",
    len(
        profile_intro_rows
    )
)


# ============================================================
# 13. Write canonical semantic relations
# ============================================================

print("\n")
print("=" * 70)
print("WRITING SEMANTIC RELATIONS")
print("=" * 70)


relations_by_type = defaultdict(
    list
)


for relation in relations:

    relation_type = relation.get(
        "relation"
    )


    if (
        relation_type
        not in ALLOWED_RELATION_TYPES
    ):
        continue


    source_kind = relation.get(
        "source_kind"
    )

    source = relation.get(
        "source"
    )

    target = relation.get(
        "target"
    )


    if not source or not target:
        continue


    target_entity = entity_lookup.get(
        str(
            target
        )
        .strip()
        .casefold()
    )


    if not target_entity:
        print(
            "WARNING: unresolved target:",
            target
        )

        continue


    target_key = target_entity[
        "_neo4j_key"
    ]


    evidence = evidence_lists(
        relation.get(
            "evidence",
            []
        )
    )


    row = {
        "source_kind":
            source_kind,

        "source":
            source,

        "target_key":
            target_key,

        "source_files":
            relation.get(
                "source_files",
                []
            ),

        "evidence_chunk_ids":
            evidence[
                "chunk_ids"
            ],

        "evidence_pages":
            evidence[
                "pages"
            ],

        "evidence_json":
            evidence[
                "json"
            ],

        "evidence_count":
            len(
                evidence[
                    "chunk_ids"
                ]
            ),
    }


    if source_kind == "Entity":

        source_entity = entity_lookup.get(
            str(
                source
            )
            .strip()
            .casefold()
        )


        if not source_entity:

            print(
                "WARNING: unresolved source:",
                source
            )

            continue


        row[
            "source_key"
        ] = source_entity[
            "_neo4j_key"
        ]


    relations_by_type[
        relation_type
    ].append(
        row
    )


for relation_type in sorted(
    relations_by_type
):
    rel_type = normalize_label(
        relation_type
    )

    rows = relations_by_type[
        relation_type
    ]


    # --------------------------------------------------------
    # INTRODUCES:
    # Paper -> KnowledgeEntity
    # --------------------------------------------------------

    if relation_type == "INTRODUCES":

        paper_rows = [
            row
            for row in rows
            if row[
                "source_kind"
            ] == "Paper"
        ]


        if paper_rows:

            query = f"""
            UNWIND $rows AS row

            MATCH (
                p:Paper
                {{
                    source_file:
                        row.source
                }}
            )

            MATCH (
                t:KnowledgeEntity
                {{
                    key:
                        row.target_key
                }}
            )

            MERGE (
                p
            )-[r:{rel_type}]->(
                t
            )

            SET
                r.source_files =
                    row.source_files,

                r.evidence_chunk_ids =
                    row.evidence_chunk_ids,

                r.evidence_pages =
                    row.evidence_pages,

                r.evidence_json =
                    row.evidence_json,

                r.evidence_count =
                    row.evidence_count
            """


            with driver.session(
                database=NEO4J_DATABASE
            ) as session:

                session.run(
                    query,
                    rows=paper_rows,
                ).consume()


            print(
                f"{relation_type}: "
                f"{len(paper_rows)} "
                f"Paper->Entity edge(s)"
            )


    # --------------------------------------------------------
    # All other semantic relations:
    # KnowledgeEntity -> KnowledgeEntity
    # --------------------------------------------------------

    else:

        entity_rows = [
            row
            for row in rows
            if row[
                "source_kind"
            ] == "Entity"
        ]


        if entity_rows:

            query = f"""
            UNWIND $rows AS row

            MATCH (
                s:KnowledgeEntity
                {{
                    key:
                        row.source_key
                }}
            )

            MATCH (
                t:KnowledgeEntity
                {{
                    key:
                        row.target_key
                }}
            )

            MERGE (
                s
            )-[r:{rel_type}]->(
                t
            )

            SET
                r.source_files =
                    row.source_files,

                r.evidence_chunk_ids =
                    row.evidence_chunk_ids,

                r.evidence_pages =
                    row.evidence_pages,

                r.evidence_json =
                    row.evidence_json,

                r.evidence_count =
                    row.evidence_count
            """


            with driver.session(
                database=NEO4J_DATABASE
            ) as session:

                session.run(
                    query,
                    rows=entity_rows,
                ).consume()


            print(
                f"{relation_type}: "
                f"{len(entity_rows)} "
                f"Entity->Entity edge(s)"
            )


# ============================================================
# 14. Verify graph counts
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
        MATCH (
            e:KnowledgeEntity
        )

        RETURN
            count(e)
                AS entities
        """
    ).single()


    entity_count = (
        record[
            "entities"
        ]
        if record
        else 0
    )


    record = session.run(
        """
        MATCH (
            p:Paper
        )-[r:INTRODUCES]->(
            e:KnowledgeEntity
        )

        RETURN
            count(r)
                AS introduces
        """
    ).single()


    introduces_count = (
        record[
            "introduces"
        ]
        if record
        else 0
    )


    record = session.run(
        """
        MATCH (
            a:KnowledgeEntity
        )-[r]->(
            b:KnowledgeEntity
        )

        WHERE type(r)
        IN [
            'USES',
            'ADDRESSES',
            'EVALUATES_ON',
            'MEASURED_BY',
            'IMPROVES',
            'COMPARED_WITH',
            'PRODUCES',
            'SOLVES'
        ]

        RETURN
            count(r)
                AS semantic_relations
        """
    ).single()


    semantic_relation_count = (
        record[
            "semantic_relations"
        ]
        if record
        else 0
    )


    record = session.run(
        """
        MATCH (
            e:KnowledgeEntity
        )-[r:SUPPORTED_BY]->(
            c:Chunk
        )

        RETURN
            count(r)
                AS provenance_edges
        """
    ).single()


    provenance_count = (
        record[
            "provenance_edges"
        ]
        if record
        else 0
    )


print(
    "KnowledgeEntity nodes:",
    entity_count
)

print(
    "Paper -> INTRODUCES edges:",
    introduces_count
)

print(
    "Entity -> Entity semantic edges:",
    semantic_relation_count
)

print(
    "Entity -> Chunk SUPPORTED_BY edges:",
    provenance_count
)


# ============================================================
# 15. Preview focal graph
# ============================================================

print("\n")
print("=" * 70)
print("VAD SEMANTIC GRAPH PREVIEW")
print("=" * 70)


with driver.session(
    database=NEO4J_DATABASE
) as session:

    result = session.run(
        """
        MATCH (
            vad:KnowledgeEntity
            {
                name:
                    'VAD'
            }
        )-[r]->(
            target:KnowledgeEntity
        )

        RETURN
            vad.name
                AS source,

            type(r)
                AS relation,

            target.name
                AS target,

            target.entity_type
                AS target_type

        ORDER BY
            relation,
            target
        """
    )


    preview_rows = [
        dict(record)
        for record in result
    ]


if not preview_rows:

    print(
        "No VAD semantic relations found."
    )

else:

    for row in preview_rows:
        print(
            f"{row['source']} "
            f"--{row['relation']}--> "
            f"{row['target']} "
            f"[{row['target_type']}]"
        )


# ============================================================
# 16. Close
# ============================================================

driver.close()


print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    "Semantic Knowledge Graph written to Neo4j."
)

print(
    "Existing Paper/Chunk/vector-RAG data were preserved."
)
