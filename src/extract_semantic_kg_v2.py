import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "paper_chunks.jsonl"
KG_DIR = PROJECT_ROOT / "data" / "kg"
OUTPUT_FILE = KG_DIR / "kg_extraction_test_v2.jsonl"

load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

MODEL_NAME = "gemini-3.5-flash-lite"
BATCH_SIZE = 8

TEST_MODE = True
TARGET_PAPER_KEYWORD = "VAD"
MAX_BATCHES = 2

ENTITY_TYPES = [
    "Method",
    "Concept",
    "Task",
    "Dataset",
    "Metric",
    "Model",
]

RELATION_TYPES = [
    "INTRODUCES",
    "USES",
    "ADDRESSES",
    "EVALUATES_ON",
    "MEASURED_BY",
    "IMPROVES",
    "COMPARED_WITH",
    "PRODUCES",
    "SOLVES",
    "BASED_ON",
]

if not CHUNKS_FILE.exists():
    raise FileNotFoundError(f"Chunk file not found:\n{CHUNKS_FILE}")

chunks = []

with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            chunks.append(json.loads(line))

print("=" * 70)
print("SEMANTIC KG EXTRACTION V2")
print("=" * 70)
print(f"\nLoaded {len(chunks)} total chunks.")

if TEST_MODE:
    chunks = [
        chunk
        for chunk in chunks
        if TARGET_PAPER_KEYWORD.lower() in chunk["source_file"].lower()
    ]
    print("TEST MODE enabled.")
    print(f"Target paper keyword: {TARGET_PAPER_KEYWORD}")
    print(f"Chunks selected: {len(chunks)}")

client = genai.Client(api_key=GEMINI_API_KEY)


def clean_json_response(text: str) -> str:
    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```"):
        text = text[len("```"):]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def normalize_name(name):
    if not isinstance(name, str):
        return ""
    return " ".join(name.strip().split())


def validate_result(data):
    valid_entities = []
    seen_entities = set()

    for entity in data.get("entities", []):
        name = normalize_name(entity.get("name"))
        entity_type = entity.get("type")

        if not name or entity_type not in ENTITY_TYPES:
            continue

        key = (name.casefold(), entity_type)

        if key in seen_entities:
            continue

        seen_entities.add(key)

        aliases = entity.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []

        cleaned_aliases = []
        for alias in aliases:
            alias = normalize_name(alias)
            if alias and alias.casefold() != name.casefold():
                cleaned_aliases.append(alias)

        evidence = entity.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []

        valid_entities.append(
            {
                "name": name,
                "type": entity_type,
                "aliases": cleaned_aliases,
                "evidence": evidence,
            }
        )

    entity_lookup = {
        entity["name"].casefold(): entity
        for entity in valid_entities
    }

    for entity in valid_entities:
        for alias in entity["aliases"]:
            entity_lookup[alias.casefold()] = entity

    valid_relations = []
    seen_relations = set()

    for relation in data.get("relations", []):
        source = normalize_name(relation.get("source"))
        target = normalize_name(relation.get("target"))
        relation_type = relation.get("type")

        if not source or not target:
            continue

        if relation_type not in RELATION_TYPES:
            continue

        target_entity = entity_lookup.get(target.casefold())
        if target_entity is None:
            continue

        target = target_entity["name"]

        if source == "PAPER":
            source_entity = None
        else:
            source_entity = entity_lookup.get(source.casefold())
            if source_entity is None:
                continue
            source = source_entity["name"]

        if relation_type == "INTRODUCES":
            if source != "PAPER":
                continue
            if target_entity["type"] not in ["Method", "Model"]:
                continue

        method_relations = {
            "USES",
            "ADDRESSES",
            "EVALUATES_ON",
            "MEASURED_BY",
            "IMPROVES",
            "COMPARED_WITH",
            "PRODUCES",
            "SOLVES",
        }

        if relation_type in method_relations:
            if source == "PAPER":
                continue
            if source_entity["type"] not in ["Method", "Model"]:
                continue

        if relation_type == "BASED_ON" and source == "PAPER":
            continue

        relation_key = (
            source.casefold(),
            relation_type,
            target.casefold(),
        )

        if relation_key in seen_relations:
            continue

        seen_relations.add(relation_key)

        evidence = relation.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []

        valid_relations.append(
            {
                "source": source,
                "type": relation_type,
                "target": target,
                "evidence": evidence,
            }
        )

    return {
        "entities": valid_entities,
        "relations": valid_relations,
    }


def build_prompt(batch):
    evidence_blocks = []

    focal_paper = batch[0]["source_file"] if batch else "Unknown"

    for item in batch:
        evidence_blocks.append(
            f"""
CHUNK_ID: {item["chunk_id"]}
PAPER: {item["source_file"]}
PAGE: {item["page"]}

TEXT:
{item["text"]}
"""
        )

    evidence = "\n".join(evidence_blocks)

    return f"""
You are constructing a HIGH-PRECISION semantic knowledge graph
from academic research-paper evidence.

You MUST use only the supplied evidence.

Do NOT use outside knowledge.
Do NOT guess missing relationships.
Do NOT turn mere co-occurrence into a relationship.

Precision is more important than recall.

============================================================
FOCAL PAPER
============================================================

{focal_paper}

The Paper node already exists in Neo4j.

Do NOT create a Paper entity.

The special source string "PAPER" may be used ONLY for:

PAPER --INTRODUCES--> Method/Model

when the focal paper explicitly introduces that named method
or model.

============================================================
ENTITY TYPES
============================================================

Allowed entity types:

{ENTITY_TYPES}

Method:
A named algorithm, technique, framework, optimization method,
reconstruction method, or computational procedure.

Concept:
A technical mathematical, geometric, physical, or scientific
concept.

Task:
A clearly defined research or computational problem.

Dataset:
A named dataset or benchmark.

Metric:
A named evaluation metric.

Model:
A named learned model, architecture, or representation model.

============================================================
ENTITY EXTRACTION RULES
============================================================

1. Extract entities that are technically important to the focal paper.

2. Do NOT extract every background term.

3. Prefer canonical names.

If a method has an explicitly defined acronym, use the acronym
as the canonical entity name and put the full name in aliases.

Example:
{{
  "name": "VAD",
  "type": "Method",
  "aliases": ["Voronoi-Assisted Optimization for Diffusing"]
}}

4. Merge obvious aliases in the same response.

For example, UDF and Unsigned Distance Field should normally be
represented by one entity.

5. Do not create entities for authors, affiliations, universities,
publication venues, section titles, figure numbers, equation numbers,
reference numbers, or generic words such as method, surface, result.

6. If a chunk is mainly bibliography/reference-list text, return no
entities or relations from that chunk.

7. Do not infer internal details of a cited prior method merely because
that prior method is mentioned in related work.

============================================================
RELATIONSHIP TYPES
============================================================

Allowed relationship types:

{RELATION_TYPES}

INTRODUCES
Use ONLY:
PAPER --INTRODUCES--> Method/Model

Never use INTRODUCES for generic concepts such as unsigned distance
field, Voronoi diagram, Poisson equation, or point cloud.

USES
Usually:
Method --USES--> Concept
or
Method --USES--> Method

ADDRESSES
Usually:
Method --ADDRESSES--> Task

EVALUATES_ON
Usually:
Method --EVALUATES_ON--> Dataset

MEASURED_BY
Usually:
Method --MEASURED_BY--> Metric

IMPROVES
Usually:
Method --IMPROVES--> Method

COMPARED_WITH
Usually:
Method --COMPARED_WITH--> Method

PRODUCES
Usually:
Method --PRODUCES--> Concept

SOLVES
Use when the method explicitly solves a mathematical equation,
optimization problem, or named computational problem.

BASED_ON
Use only when the evidence explicitly establishes a technical
dependency or foundation.

Example:
Projection distance field --BASED_ON--> Voronoi diagram

============================================================
RELATIONSHIP QUALITY RULES
============================================================

A relationship must be explicitly supported by the evidence.

Do NOT convert mere co-occurrence into a relationship.

Be especially conservative with related-work sections.

If another method is merely cited as prior work, do not claim that the
focal paper USES or INTRODUCES it.

============================================================
EVIDENCE REQUIREMENT
============================================================

Every entity and relationship MUST contain:
- chunk_id
- page

Use only chunk IDs and page numbers provided in the evidence.

============================================================
OUTPUT FORMAT
============================================================

Return VALID JSON ONLY.

No Markdown.
No explanation before or after the JSON.

Exact structure:

{{
  "entities": [
    {{
      "name": "VAD",
      "type": "Method",
      "aliases": ["Voronoi-Assisted Optimization for Diffusing"],
      "evidence": [
        {{
          "chunk_id": "...",
          "page": 1
        }}
      ]
    }}
  ],
  "relations": [
    {{
      "source": "PAPER",
      "type": "INTRODUCES",
      "target": "VAD",
      "evidence": [
        {{
          "chunk_id": "...",
          "page": 1
        }}
      ]
    }}
  ]
}}

If the evidence does not establish valid entities or relations,
return empty arrays.

============================================================
EVIDENCE
============================================================

{evidence}
"""


def extract_batch(batch, batch_number):
    prompt = build_prompt(batch)

    for attempt in range(1, 4):
        try:
            response = client.interactions.create(
                model=MODEL_NAME,
                input=prompt,
            )

            cleaned = clean_json_response(response.output_text)
            data = json.loads(cleaned)
            return validate_result(data)

        except Exception as error:
            print(
                f"Batch {batch_number} "
                f"attempt {attempt} failed:"
            )
            print(error)

            if attempt < 3:
                time.sleep(2)

    return {
        "entities": [],
        "relations": [],
        "error": True,
    }


KG_DIR.mkdir(parents=True, exist_ok=True)

total_possible_batches = (
    len(chunks) + BATCH_SIZE - 1
) // BATCH_SIZE

if TEST_MODE:
    total_batches = min(
        total_possible_batches,
        MAX_BATCHES,
    )
else:
    total_batches = total_possible_batches

print(f"\nBatch size: {BATCH_SIZE}")
print(f"Batches to process: {total_batches}")

with open(OUTPUT_FILE, "w", encoding="utf-8"):
    pass

total_entities = 0
total_relations = 0

for batch_number in range(1, total_batches + 1):
    start = (batch_number - 1) * BATCH_SIZE
    end = min(start + BATCH_SIZE, len(chunks))
    batch = chunks[start:end]

    print("\n")
    print("=" * 70)
    print(f"BATCH {batch_number}/{total_batches}")
    print("=" * 70)
    print(f"Chunks: {start + 1} - {end}")

    result = extract_batch(
        batch,
        batch_number,
    )

    entities = result.get("entities", [])
    relations = result.get("relations", [])

    total_entities += len(entities)
    total_relations += len(relations)

    output_record = {
        "batch_number": batch_number,
        "source_file": batch[0]["source_file"] if batch else None,
        "chunk_ids": [item["chunk_id"] for item in batch],
        "entities": entities,
        "relations": relations,
    }

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8",
    ) as output:
        output.write(
            json.dumps(
                output_record,
                ensure_ascii=False,
            )
        )
        output.write("\n")

    print(f"Entities extracted: {len(entities)}")
    print(f"Relations extracted: {len(relations)}")

    print("\nEntity preview:")
    for entity in entities[:15]:
        aliases = entity.get("aliases", [])
        alias_text = f" | aliases={aliases}" if aliases else ""
        print(
            f"  [{entity.get('type')}] "
            f"{entity.get('name')}"
            f"{alias_text}"
        )

    print("\nRelationship preview:")
    for relation in relations[:15]:
        print(
            f"  {relation.get('source')} "
            f"--{relation.get('type')}--> "
            f"{relation.get('target')}"
        )

print("\n")
print("=" * 70)
print("KG EXTRACTION SUMMARY")
print("=" * 70)
print("Batches processed:", total_batches)
print("Entities extracted:", total_entities)
print("Relations extracted:", total_relations)
print("\nSaved:")
print(OUTPUT_FILE)
print("\nV2 TEST extraction completed.")
