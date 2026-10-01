import json
import os
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "paper_chunks.jsonl"
)

KG_DIR = (
    PROJECT_ROOT
    / "data"
    / "kg"
)

OUTPUT_FILE = (
    KG_DIR
    / "kg_extraction_full_raw.jsonl"
)

PAPER_PROFILE_FILE = (
    KG_DIR
    / "paper_profiles.json"
)


# ============================================================
# 2. Environment
# ============================================================

load_dotenv(
    PROJECT_ROOT / ".env"
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing from .env"
    )


# ============================================================
# 3. Configuration
# ============================================================

MODEL_NAME = "gemini-3.5-flash-lite"

# 12 chunks * ~220 tokens = manageable evidence per request,
# while reducing the number of API calls.
BATCH_SIZE = 12

# Retry each Gemini request if parsing/network/API fails.
MAX_RETRIES = 3

# Delay between successful requests.
# Increase this if your API tier reports rate-limit errors.
REQUEST_DELAY_SECONDS = 1.0

# Number of early chunks used to identify the focal method/model.
PROFILE_CHUNKS = 8


# ============================================================
# 4. KG schema
# ============================================================

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
]


# ============================================================
# 5. Helpers
# ============================================================

def normalize_name(name):
    if not isinstance(name, str):
        return ""

    return " ".join(
        name.strip().split()
    )


def clean_json_response(text):
    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```"):
        text = text[len("```"):]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def call_gemini_json(prompt, label):
    """
    Call Gemini and return parsed JSON.
    Retries failed requests / malformed JSON.
    """

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):
        try:
            response = client.interactions.create(
                model=MODEL_NAME,
                input=prompt,
            )

            cleaned = clean_json_response(
                response.output_text
            )

            return json.loads(
                cleaned
            )

        except Exception as error:
            print(
                f"{label} | attempt "
                f"{attempt}/{MAX_RETRIES} failed:"
            )
            print(error)

            if attempt < MAX_RETRIES:
                time.sleep(
                    2 * attempt
                )

    return None


# ============================================================
# 6. Load chunks
# ============================================================

if not CHUNKS_FILE.exists():
    raise FileNotFoundError(
        f"Chunk file not found:\n"
        f"{CHUNKS_FILE}"
    )


all_chunks = []


with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as file:

    for line in file:
        line = line.strip()

        if not line:
            continue

        all_chunks.append(
            json.loads(line)
        )


# Group chunks by source paper.
papers = defaultdict(list)


for chunk in all_chunks:
    papers[
        chunk["source_file"]
    ].append(chunk)


# Stable ordering.
paper_names = sorted(
    papers.keys()
)


for source_file in paper_names:
    papers[source_file].sort(
        key=lambda item: (
            item["page"],
            item["chunk_index"],
        )
    )


print("=" * 70)
print("FULL SEMANTIC KG EXTRACTION")
print("=" * 70)

print(
    f"\nTotal papers: "
    f"{len(paper_names)}"
)

print(
    f"Total chunks: "
    f"{len(all_chunks)}"
)

print(
    f"Batch size: "
    f"{BATCH_SIZE}"
)


# ============================================================
# 7. Gemini client
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# 8. Paper profile discovery
# ============================================================

def build_profile_prompt(
    source_file,
    paper_chunks
):
    """
    Use early chunks (title/abstract/introduction) to discover
    the focal method/model introduced by this paper.
    """

    selected = paper_chunks[
        :PROFILE_CHUNKS
    ]

    evidence_blocks = []


    for item in selected:
        evidence_blocks.append(
            f"""
PAGE: {item["page"]}
CHUNK_ID: {item["chunk_id"]}

TEXT:
{item["text"]}
"""
        )


    evidence = "\n".join(
        evidence_blocks
    )


    return f"""
Identify the MAIN focal method or model introduced by this
academic paper.

Use ONLY the supplied paper evidence.

Paper filename:
{source_file}

Return VALID JSON ONLY.

Do not return background methods merely cited in related work.

If the paper explicitly defines an acronym, use the acronym as
the canonical name and put the full name in aliases.

If there is no short acronym, use the full proposed method name.

Return exactly:

{{
  "focal_entities": [
    {{
      "name": "canonical method/model name",
      "type": "Method",
      "aliases": [
        "full name if applicable"
      ]
    }}
  ]
}}

Normally there should be ONE focal entity.
Use more than one only if the paper explicitly introduces multiple
peer-level methods/models.

EVIDENCE:

{evidence}
"""


def validate_profile(data):
    result = []

    if not isinstance(
        data,
        dict
    ):
        return result


    for entity in data.get(
        "focal_entities",
        []
    ):
        name = normalize_name(
            entity.get("name")
        )

        entity_type = entity.get(
            "type"
        )

        if (
            not name
            or entity_type not in {
                "Method",
                "Model",
            }
        ):
            continue


        aliases = entity.get(
            "aliases",
            []
        )

        if not isinstance(
            aliases,
            list
        ):
            aliases = []


        cleaned_aliases = []

        for alias in aliases:
            alias = normalize_name(
                alias
            )

            if (
                alias
                and alias.casefold()
                != name.casefold()
            ):
                cleaned_aliases.append(
                    alias
                )


        result.append(
            {
                "name":
                    name,

                "type":
                    entity_type,

                "aliases":
                    cleaned_aliases,
            }
        )


    return result


# Load cached paper profiles when rerunning.
if PAPER_PROFILE_FILE.exists():
    with open(
        PAPER_PROFILE_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        paper_profiles = json.load(
            file
        )
else:
    paper_profiles = {}


print("\n")
print("=" * 70)
print("STEP 1: PAPER PROFILE DISCOVERY")
print("=" * 70)


for paper_index, source_file in enumerate(
    paper_names,
    start=1
):
    if (
        source_file in paper_profiles
        and paper_profiles[
            source_file
        ].get("focal_entities")
    ):
        names = [
            item["name"]
            for item in paper_profiles[
                source_file
            ]["focal_entities"]
        ]

        print(
            f"[{paper_index}/{len(paper_names)}] "
            f"Cached: {source_file}"
        )

        print(
            f"    Focal: {names}"
        )

        continue


    print(
        f"\n[{paper_index}/{len(paper_names)}] "
        f"Discovering focal method:"
    )

    print(
        source_file
    )


    prompt = build_profile_prompt(
        source_file,
        papers[source_file],
    )


    data = call_gemini_json(
        prompt,
        f"PROFILE {paper_index}"
    )


    focal_entities = validate_profile(
        data or {}
    )


    if not focal_entities:
        print(
            "    WARNING: No focal method/model found."
        )

        paper_profiles[
            source_file
        ] = {
            "focal_entities": []
        }

    else:
        print(
            "    Focal:",
            [
                item["name"]
                for item in focal_entities
            ]
        )

        paper_profiles[
            source_file
        ] = {
            "focal_entities":
                focal_entities
        }


    # Save immediately so reruns do not repeat successful calls.
    KG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        PAPER_PROFILE_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            paper_profiles,
            file,
            ensure_ascii=False,
            indent=2,
        )


    time.sleep(
        REQUEST_DELAY_SECONDS
    )


# ============================================================
# 9. Extraction prompt
# ============================================================

def build_extraction_prompt(
    source_file,
    focal_entities,
    batch,
):
    focal_text = json.dumps(
        focal_entities,
        ensure_ascii=False,
    )


    evidence_blocks = []


    for item in batch:
        evidence_blocks.append(
            f"""
CHUNK_ID: {item["chunk_id"]}
PAGE: {item["page"]}

TEXT:
{item["text"]}
"""
        )


    evidence = "\n".join(
        evidence_blocks
    )


    return f"""
Construct a HIGH-PRECISION semantic knowledge graph for ONE
academic research paper.

Use ONLY the supplied evidence.

Do NOT use outside knowledge.
Do NOT infer a relation from mere co-occurrence.
Precision is more important than recall.

============================================================
FOCAL PAPER
============================================================

{source_file}

The Paper node already exists in Neo4j.

Do NOT create a Paper entity.

============================================================
FOCAL METHOD / MODEL
============================================================

The focal method/model for this paper has already been identified:

{focal_text}

Canonicalize mentions such as "our method", "our approach", or a
full method name to the corresponding focal entity ONLY when the
text clearly refers to it.

============================================================
ALLOWED ENTITY TYPES
============================================================

{ENTITY_TYPES}

Method:
Named algorithm, technique, framework, optimization method,
reconstruction method, or computational procedure.

Concept:
Technical mathematical, geometric, scientific, or representation
concept central to the focal method.

Task:
Research/computational problem addressed by the focal method.

Dataset:
Named dataset or benchmark used to evaluate the focal method.

Metric:
Named quantitative evaluation metric.

Model:
Named learned model, architecture, or representation model.

============================================================
ALLOWED RELATION TYPES
============================================================

{RELATION_TYPES}

============================================================
STRICT SCOPE RULE
============================================================

Extract ONLY entities that participate in a supported relationship
involving the focal method/model.

The allowed relationship patterns are:

PAPER --INTRODUCES--> focal Method/Model

focal Method/Model --USES--> Concept/Method/Model

focal Method/Model --ADDRESSES--> Task

focal Method/Model --EVALUATES_ON--> Dataset

focal Method/Model --MEASURED_BY--> Metric

focal Method/Model --IMPROVES--> Method/Model

focal Method/Model --COMPARED_WITH--> Method/Model

focal Method/Model --PRODUCES--> Concept/Model

focal Method/Model --SOLVES--> Concept/Task


Do NOT create subgraphs describing cited prior work.

Do NOT extract a prior method simply because it appears in a
related-work list.

A prior method may be retained ONLY when the focal method is
explicitly compared with it or explicitly improves it.

Do NOT extract bibliography/reference-list content.

Do NOT extract:
- authors
- affiliations
- universities
- venues
- section names
- figure numbers
- equation numbers
- reference numbers
- generic words such as "method", "surface", "result"

============================================================
CANONICALIZATION
============================================================

Prefer singular canonical technical names.

Merge obvious aliases in the same response.

Examples:

UDF / UDFs / unsigned distance field
-> one Concept entity

CD / Chamfer Distance
-> one Metric entity, when the paper defines or uses that alias

Use the supplied focal entity canonical name exactly.

============================================================
EVIDENCE REQUIREMENT
============================================================

Every entity and relation MUST include evidence:
- chunk_id
- page

Use only supplied chunk IDs and pages.

============================================================
OUTPUT FORMAT
============================================================

Return VALID JSON ONLY.

No Markdown.
No prose outside JSON.

{{
  "entities": [
    {{
      "name": "entity canonical name",
      "type": "Concept",
      "aliases": [
        "optional alias"
      ],
      "evidence": [
        {{
          "chunk_id": "...",
          "page": 3
        }}
      ]
    }}
  ],
  "relations": [
    {{
      "source": "focal canonical method name",
      "type": "USES",
      "target": "entity canonical name",
      "evidence": [
        {{
          "chunk_id": "...",
          "page": 3
        }}
      ]
    }}
  ]
}}

When evidence explicitly introduces the focal method, use:

"PAPER" --INTRODUCES--> focal canonical name

If this batch contains no supported focal-method facts, return
empty arrays.

============================================================
EVIDENCE
============================================================

{evidence}
"""


# ============================================================
# 10. Deterministic validation
# ============================================================

def build_focal_lookup(
    focal_entities
):
    lookup = {}

    for entity in focal_entities:
        lookup[
            entity["name"].casefold()
        ] = entity["name"]

        for alias in entity.get(
            "aliases",
            []
        ):
            lookup[
                alias.casefold()
            ] = entity["name"]

    return lookup


def validate_batch_result(
    data,
    focal_entities,
):
    if not isinstance(
        data,
        dict
    ):
        return {
            "entities": [],
            "relations": [],
        }


    focal_lookup = build_focal_lookup(
        focal_entities
    )

    focal_names = set(
        focal_lookup.values()
    )


    # --------------------------------------------------------
    # Validate entities
    # --------------------------------------------------------

    entities = []

    entity_lookup = {}


    # Always make focal entities available to relation validation.
    for focal in focal_entities:
        copy = {
            "name":
                focal["name"],

            "type":
                focal["type"],

            "aliases":
                focal.get(
                    "aliases",
                    []
                ),

            "evidence":
                [],
        }

        entities.append(
            copy
        )

        entity_lookup[
            copy["name"].casefold()
        ] = copy

        for alias in copy[
            "aliases"
        ]:
            entity_lookup[
                alias.casefold()
            ] = copy


    for entity in data.get(
        "entities",
        []
    ):
        name = normalize_name(
            entity.get("name")
        )

        entity_type = entity.get(
            "type"
        )


        if (
            not name
            or entity_type
            not in ENTITY_TYPES
        ):
            continue


        # Canonicalize focal aliases.
        focal_canonical = focal_lookup.get(
            name.casefold()
        )

        if focal_canonical:
            name = focal_canonical


        aliases = entity.get(
            "aliases",
            []
        )

        if not isinstance(
            aliases,
            list
        ):
            aliases = []


        clean_aliases = []

        for alias in aliases:
            alias = normalize_name(
                alias
            )

            if (
                alias
                and alias.casefold()
                != name.casefold()
            ):
                clean_aliases.append(
                    alias
                )


        evidence = entity.get(
            "evidence",
            []
        )

        if not isinstance(
            evidence,
            list
        ):
            evidence = []


        existing = entity_lookup.get(
            name.casefold()
        )


        if existing:
            # Enrich aliases/evidence for existing focal/entity.
            for alias in clean_aliases:
                if (
                    alias.casefold()
                    not in {
                        item.casefold()
                        for item in existing[
                            "aliases"
                        ]
                    }
                ):
                    existing[
                        "aliases"
                    ].append(
                        alias
                    )

            existing[
                "evidence"
            ].extend(
                evidence
            )

            continue


        new_entity = {
            "name":
                name,

            "type":
                entity_type,

            "aliases":
                clean_aliases,

            "evidence":
                evidence,
        }


        entities.append(
            new_entity
        )


        entity_lookup[
            name.casefold()
        ] = new_entity


        for alias in clean_aliases:
            entity_lookup[
                alias.casefold()
            ] = new_entity


    # --------------------------------------------------------
    # Validate relations
    # --------------------------------------------------------

    valid_relations = []

    seen_relations = set()


    for relation in data.get(
        "relations",
        []
    ):
        source = normalize_name(
            relation.get("source")
        )

        target = normalize_name(
            relation.get("target")
        )

        relation_type = relation.get(
            "type"
        )


        if (
            not source
            or not target
            or relation_type
            not in RELATION_TYPES
        ):
            continue


        # Resolve target.
        target_entity = entity_lookup.get(
            target.casefold()
        )

        if target_entity is None:
            continue


        target = target_entity[
            "name"
        ]


        # INTRODUCES is the only PAPER edge.
        if relation_type == "INTRODUCES":
            if source != "PAPER":
                continue

            if target not in focal_names:
                continue

        else:
            # Every other relation must originate from focal entity.
            source_canonical = focal_lookup.get(
                source.casefold()
            )

            if not source_canonical:
                continue

            source = source_canonical


        # Type restrictions by relation.
        if relation_type == "ADDRESSES":
            if target_entity[
                "type"
            ] != "Task":
                continue


        elif relation_type == "EVALUATES_ON":
            if target_entity[
                "type"
            ] != "Dataset":
                continue


        elif relation_type == "MEASURED_BY":
            if target_entity[
                "type"
            ] != "Metric":
                continue


        elif relation_type in {
            "IMPROVES",
            "COMPARED_WITH",
        }:
            if target_entity[
                "type"
            ] not in {
                "Method",
                "Model",
            }:
                continue


        elif relation_type == "USES":
            if target_entity[
                "type"
            ] not in {
                "Concept",
                "Method",
                "Model",
            }:
                continue


        elif relation_type == "PRODUCES":
            if target_entity[
                "type"
            ] not in {
                "Concept",
                "Model",
            }:
                continue


        elif relation_type == "SOLVES":
            if target_entity[
                "type"
            ] not in {
                "Concept",
                "Task",
            }:
                continue


        evidence = relation.get(
            "evidence",
            []
        )

        if not isinstance(
            evidence,
            list
        ):
            evidence = []


        key = (
            source.casefold(),
            relation_type,
            target.casefold(),
        )


        if key in seen_relations:
            continue


        seen_relations.add(
            key
        )


        valid_relations.append(
            {
                "source":
                    source,

                "type":
                    relation_type,

                "target":
                    target,

                "evidence":
                    evidence,
            }
        )


    # --------------------------------------------------------
    # Prune unused non-focal entities
    # --------------------------------------------------------

    used_names = set(
        name.casefold()
        for name in focal_names
    )


    for relation in valid_relations:
        if relation[
            "source"
        ] != "PAPER":
            used_names.add(
                relation[
                    "source"
                ].casefold()
            )

        used_names.add(
            relation[
                "target"
            ].casefold()
        )


    pruned_entities = [
        entity
        for entity in entities
        if entity[
            "name"
        ].casefold()
        in used_names
    ]


    return {
        "entities":
            pruned_entities,

        "relations":
            valid_relations,
    }


# ============================================================
# 11. Resume support
# ============================================================

completed_keys = set()


if OUTPUT_FILE.exists():
    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(
                    line
                )

                completed_keys.add(
                    (
                        item[
                            "source_file"
                        ],
                        item[
                            "batch_number"
                        ],
                    )
                )

            except Exception:
                pass


print("\n")
print("=" * 70)
print("STEP 2: FULL PAPER EXTRACTION")
print("=" * 70)

print(
    f"\nAlready-completed batches "
    f"found: {len(completed_keys)}"
)


# ============================================================
# 12. Full extraction
# ============================================================

total_batches_expected = 0


for source_file in paper_names:
    total_batches_expected += (
        len(
            papers[source_file]
        )
        + BATCH_SIZE
        - 1
    ) // BATCH_SIZE


print(
    f"Expected total batches: "
    f"{total_batches_expected}"
)


global_batch_counter = 0

new_batches_written = 0

new_entity_count = 0

new_relation_count = 0


for paper_index, source_file in enumerate(
    paper_names,
    start=1
):
    focal_entities = paper_profiles.get(
        source_file,
        {}
    ).get(
        "focal_entities",
        []
    )


    print("\n")
    print("#" * 70)

    print(
        f"PAPER "
        f"{paper_index}/{len(paper_names)}"
    )

    print(
        source_file
    )

    print(
        "Focal:",
        [
            item["name"]
            for item in focal_entities
        ]
    )

    print("#" * 70)


    if not focal_entities:
        print(
            "Skipping paper because no focal "
            "method/model profile is available."
        )

        continue


    paper_chunks = papers[
        source_file
    ]


    total_paper_batches = (
        len(paper_chunks)
        + BATCH_SIZE
        - 1
    ) // BATCH_SIZE


    for batch_number in range(
        1,
        total_paper_batches + 1
    ):
        global_batch_counter += 1

        key = (
            source_file,
            batch_number,
        )


        if key in completed_keys:
            print(
                f"Batch "
                f"{batch_number}/"
                f"{total_paper_batches} "
                f"already completed -> skip"
            )

            continue


        start = (
            (batch_number - 1)
            * BATCH_SIZE
        )

        end = min(
            start + BATCH_SIZE,
            len(paper_chunks)
        )


        batch = paper_chunks[
            start:end
        ]


        print(
            f"\nBatch "
            f"{batch_number}/"
            f"{total_paper_batches} "
            f"| chunks "
            f"{start + 1}-{end}"
        )


        prompt = build_extraction_prompt(
            source_file,
            focal_entities,
            batch,
        )


        raw_data = call_gemini_json(
            prompt,
            (
                f"PAPER {paper_index} "
                f"BATCH {batch_number}"
            )
        )


        if raw_data is None:
            print(
                "    FAILED after retries. "
                "Not writing checkpoint; rerun later."
            )

            continue


        validated = validate_batch_result(
            raw_data,
            focal_entities,
        )


        entities = validated[
            "entities"
        ]

        relations = validated[
            "relations"
        ]


        # Remove focal placeholder entity if this batch has
        # no evidence/relations at all.
        if not relations:
            entities = []


        record = {
            "source_file":
                source_file,

            "paper_id":
                batch[0][
                    "paper_id"
                ],

            "batch_number":
                batch_number,

            "chunk_ids": [
                item[
                    "chunk_id"
                ]
                for item in batch
            ],

            "focal_entities":
                focal_entities,

            "entities":
                entities,

            "relations":
                relations,
        }


        KG_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        with open(
            OUTPUT_FILE,
            "a",
            encoding="utf-8"
        ) as output:
            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )

            output.write(
                "\n"
            )


        completed_keys.add(
            key
        )


        new_batches_written += 1

        new_entity_count += len(
            entities
        )

        new_relation_count += len(
            relations
        )


        print(
            f"    Entities: "
            f"{len(entities)}"
        )

        print(
            f"    Relations: "
            f"{len(relations)}"
        )


        for relation in relations[
            :8
        ]:
            print(
                f"      "
                f"{relation['source']} "
                f"--{relation['type']}--> "
                f"{relation['target']}"
            )


        time.sleep(
            REQUEST_DELAY_SECONDS
        )


# ============================================================
# 13. Summary
# ============================================================

print("\n")
print("=" * 70)
print("FULL EXTRACTION RUN SUMMARY")
print("=" * 70)

print(
    "Papers:",
    len(paper_names)
)

print(
    "Corpus chunks:",
    len(all_chunks)
)

print(
    "Expected batches:",
    total_batches_expected
)

print(
    "Completed batch checkpoints:",
    len(completed_keys)
)

print(
    "New batches written this run:",
    new_batches_written
)

print(
    "New entities (before global dedup):",
    new_entity_count
)

print(
    "New relations (before global dedup):",
    new_relation_count
)

print(
    "\nRaw KG output:"
)

print(
    OUTPUT_FILE
)

print(
    "\nPaper profiles:"
)

print(
    PAPER_PROFILE_FILE
)

print(
    "\nIf some batches failed because of rate limits, "
    "run this same script again. Completed batches "
    "will be skipped automatically."
)
