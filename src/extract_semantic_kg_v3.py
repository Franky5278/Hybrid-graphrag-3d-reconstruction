import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "paper_chunks.jsonl"
KG_DIR = PROJECT_ROOT / "data" / "kg"
OUTPUT_FILE = KG_DIR / "kg_extraction_test_v3.jsonl"

load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")


# ============================================================
# 2. Configuration
# ============================================================

MODEL_NAME = "gemini-3.5-flash-lite"
BATCH_SIZE = 8

TEST_MODE = True
TARGET_PAPER_KEYWORD = "VAD"
MAX_BATCHES = 2

# Test-mode focal method. In the later full-run script,
# this will be discovered automatically per paper.
FOCAL_METHOD_NAMES = {"VAD"}


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


# ============================================================
# 3. Load chunks
# ============================================================

if not CHUNKS_FILE.exists():
    raise FileNotFoundError(f"Chunk file not found:\n{CHUNKS_FILE}")

chunks = []

with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            chunks.append(json.loads(line))

print("=" * 70)
print("SEMANTIC KG EXTRACTION V3")
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
    print(f"Focal method(s): {sorted(FOCAL_METHOD_NAMES)}")
    print(f"Chunks selected: {len(chunks)}")


client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# 4. Helpers
# ============================================================

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


def casefold_set(values):
    return {normalize_name(v).casefold() for v in values if normalize_name(v)}


# ============================================================
# 5. High-precision deterministic validator
# ============================================================

def validate_result(data, focal_method_names):
    """
    V3 policy:
    - Keep only schema-valid relationships.
    - Method-centric relations must originate from the focal method/model.
    - BASED_ON is kept only for concept-level technical dependencies.
    - Prior-work-only method relations are discarded.
    - After relation filtering, orphan entities are pruned.
    """

    focal_cf = casefold_set(focal_method_names)

    # --------------------------------------------------------
    # Validate + deduplicate raw entities
    # --------------------------------------------------------

    entities = []
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

        clean_aliases = []

        for alias in aliases:
            alias = normalize_name(alias)

            if (
                alias
                and alias.casefold() != name.casefold()
                and alias.casefold() not in {
                    a.casefold()
                    for a in clean_aliases
                }
            ):
                clean_aliases.append(alias)

        evidence = entity.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []

        entities.append(
            {
                "name": name,
                "type": entity_type,
                "aliases": clean_aliases,
                "evidence": evidence,
            }
        )

    # --------------------------------------------------------
    # Alias-aware lookup
    # --------------------------------------------------------

    entity_lookup = {}

    for entity in entities:
        entity_lookup[entity["name"].casefold()] = entity

        for alias in entity["aliases"]:
            entity_lookup[alias.casefold()] = entity

    # --------------------------------------------------------
    # Identify canonical focal entities
    # --------------------------------------------------------

    focal_entities = []

    for entity in entities:
        names = {
            entity["name"].casefold(),
            *[
                alias.casefold()
                for alias in entity["aliases"]
            ],
        }

        if names & focal_cf:
            focal_entities.append(entity)

    focal_canonical_cf = {
        entity["name"].casefold()
        for entity in focal_entities
    }

    # --------------------------------------------------------
    # Validate relations
    # --------------------------------------------------------

    valid_relations = []
    seen_relations = set()

    for relation in data.get("relations", []):
        source = normalize_name(relation.get("source"))
        target = normalize_name(relation.get("target"))
        relation_type = relation.get("type")

        if (
            not source
            or not target
            or relation_type not in RELATION_TYPES
        ):
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

        # ----------------------------------------------------
        # PAPER can only INTRODUCE the focal Method / Model
        # ----------------------------------------------------

        if relation_type == "INTRODUCES":
            if source != "PAPER":
                continue

            if target_entity["type"] not in {"Method", "Model"}:
                continue

            target_names_cf = {
                target_entity["name"].casefold(),
                *[
                    alias.casefold()
                    for alias in target_entity["aliases"]
                ],
            }

            if not (target_names_cf & focal_cf):
                continue

        # ----------------------------------------------------
        # Method-centric relations must start from focal method
        # ----------------------------------------------------

        elif relation_type in {
            "USES",
            "ADDRESSES",
            "EVALUATES_ON",
            "MEASURED_BY",
            "IMPROVES",
            "COMPARED_WITH",
            "PRODUCES",
            "SOLVES",
        }:
            if source == "PAPER":
                continue

            if source_entity["type"] not in {"Method", "Model"}:
                continue

            source_names_cf = {
                source_entity["name"].casefold(),
                *[
                    alias.casefold()
                    for alias in source_entity["aliases"]
                ],
            }

            if not (source_names_cf & focal_cf):
                continue

        # ----------------------------------------------------
        # BASED_ON is allowed for core concept-to-concept
        # dependencies, but not prior-method subgraphs.
        # ----------------------------------------------------

        elif relation_type == "BASED_ON":
            if source == "PAPER":
                continue

            if source_entity["type"] != "Concept":
                continue

            if target_entity["type"] != "Concept":
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

    # --------------------------------------------------------
    # Prune orphan entities
    #
    # This removes background entities like alpha-shapes,
    # Ball Pivoting, Power Crust, etc. when they do not
    # participate in a retained focal relation.
    # --------------------------------------------------------

    used_entity_names_cf = set()

    for relation in valid_relations:
        if relation["source"] != "PAPER":
            used_entity_names_cf.add(
                relation["source"].casefold()
            )

        used_entity_names_cf.add(
            relation["target"].casefold()
        )

    # Always keep focal method entity when present
    used_entity_names_cf |= focal_canonical_cf

    pruned_entities = [
        entity
        for entity in entities
        if entity["name"].casefold()
        in used_entity_names_cf
    ]

    return {
        "entities": pruned_entities,
        "relations": valid_relations,
    }


# ============================================================
# 6. Prompt
# ============================================================

def build_prompt(batch):
    focal_paper = (
        batch[0]["source_file"]
        if batch
        else "Unknown"
    )

    evidence_blocks = []

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

    focal_methods = ", ".join(sorted(FOCAL_METHOD_NAMES))

    return f"""
You are extracting a HIGH-PRECISION semantic knowledge graph
for ONE focal academic paper.

Use ONLY the supplied evidence.
Do NOT use outside knowledge.
Do NOT infer relations from mere co-occurrence.
Precision is more important than recall.

============================================================
FOCAL PAPER
============================================================

{focal_paper}

Known focal method/model name(s):

{focal_methods}

The Paper node already exists.
Do NOT create a Paper entity.

============================================================
ALLOWED ENTITY TYPES
============================================================

{ENTITY_TYPES}

Method:
Named algorithm, technique, framework, optimization method,
reconstruction method, or computational procedure.

Concept:
Technical mathematical, geometric, physical, or scientific concept.

Task:
Research/computational problem addressed by the focal method.

Dataset:
Named dataset or benchmark.

Metric:
Named evaluation metric.

Model:
Named learned model, architecture, or representation model.

============================================================
ALLOWED RELATION TYPES
============================================================

{RELATION_TYPES}

============================================================
CRITICAL SCOPE RULE
============================================================

This graph is about the FOCAL METHOD, not the entire bibliography.

Only extract:

A. The focal method/model itself.

B. Entities directly connected to the focal method via:
   USES, ADDRESSES, EVALUATES_ON, MEASURED_BY,
   IMPROVES, COMPARED_WITH, PRODUCES, SOLVES.

C. Core concept-to-concept dependencies that explain the
   focal method, via BASED_ON.

D. PAPER --INTRODUCES--> focal Method/Model.

Do NOT build subgraphs describing unrelated prior methods.

For example, if related work says method X is based on method Y,
do NOT extract X --BASED_ON--> Y unless X or Y is genuinely part
of the focal method's own technical pipeline.

Do NOT extract background methods merely because they are listed.

Do NOT extract authors, institutions, venues, section titles,
figure numbers, equations numbers, or bibliography entries.

============================================================
CANONICALIZATION
============================================================

If the focal method has an acronym, use the acronym as canonical
name and put the full name in aliases.

Use one canonical entity for obvious aliases, e.g.:

UDF
Unsigned Distance Field
unsigned distance field

============================================================
RELATION SEMANTICS
============================================================

INTRODUCES:
PAPER --INTRODUCES--> focal Method/Model only.

USES:
focal Method/Model --USES--> Concept/Method.

ADDRESSES:
focal Method/Model --ADDRESSES--> Task.

EVALUATES_ON:
focal Method/Model --EVALUATES_ON--> Dataset.

MEASURED_BY:
focal Method/Model --MEASURED_BY--> Metric.

IMPROVES:
focal Method/Model --IMPROVES--> Method.

COMPARED_WITH:
focal Method/Model --COMPARED_WITH--> Method.

PRODUCES:
focal Method/Model --PRODUCES--> Concept.

SOLVES:
focal Method/Model --SOLVES--> Concept/Task representing an
explicit mathematical or computational problem.

BASED_ON:
Concept --BASED_ON--> Concept only, when the evidence explicitly
states a technical dependency central to the focal method.

============================================================
EVIDENCE
============================================================

Every entity and every relation MUST include:
- chunk_id
- page

Use only supplied chunk IDs and page numbers.

============================================================
OUTPUT
============================================================

Return VALID JSON ONLY.
No Markdown.
No prose outside JSON.

Exact structure:

{{
  "entities": [
    {{
      "name": "VAD",
      "type": "Method",
      "aliases": [
        "Voronoi-Assisted Optimization for Diffusing"
      ],
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

If no in-scope entities or relations are supported,
return empty arrays.

============================================================
PAPER EVIDENCE
============================================================

{evidence}
"""


# ============================================================
# 7. Gemini call
# ============================================================

def extract_batch(batch, batch_number):
    prompt = build_prompt(batch)

    for attempt in range(1, 4):
        try:
            response = client.interactions.create(
                model=MODEL_NAME,
                input=prompt,
            )

            cleaned = clean_json_response(
                response.output_text
            )

            data = json.loads(cleaned)

            return validate_result(
                data,
                FOCAL_METHOD_NAMES,
            )

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


# ============================================================
# 8. Run test
# ============================================================

KG_DIR.mkdir(parents=True, exist_ok=True)

total_possible_batches = (
    len(chunks) + BATCH_SIZE - 1
) // BATCH_SIZE

total_batches = (
    min(total_possible_batches, MAX_BATCHES)
    if TEST_MODE
    else total_possible_batches
)

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

    record = {
        "batch_number": batch_number,
        "source_file": (
            batch[0]["source_file"]
            if batch
            else None
        ),
        "chunk_ids": [
            item["chunk_id"]
            for item in batch
        ],
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
                record,
                ensure_ascii=False,
            )
        )
        output.write("\n")

    print(f"Entities extracted: {len(entities)}")
    print(f"Relations extracted: {len(relations)}")

    print("\nEntity preview:")
    for entity in entities[:15]:
        aliases = entity.get("aliases", [])
        alias_text = (
            f" | aliases={aliases}"
            if aliases
            else ""
        )

        print(
            f"  [{entity['type']}] "
            f"{entity['name']}"
            f"{alias_text}"
        )

    print("\nRelationship preview:")
    for relation in relations[:15]:
        print(
            f"  {relation['source']} "
            f"--{relation['type']}--> "
            f"{relation['target']}"
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
print("\nV3 high-precision TEST completed.")
