import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KG_DIR = PROJECT_ROOT / "data" / "kg"

RAW_FILE = KG_DIR / "kg_extraction_full_raw.jsonl"
PROFILE_FILE = KG_DIR / "paper_profiles.json"

OUTPUT_FILE = KG_DIR / "kg_canonical.json"
OUTPUT_ENTITIES = KG_DIR / "kg_canonical_entities.jsonl"
OUTPUT_RELATIONS = KG_DIR / "kg_canonical_relations.jsonl"


# ============================================================
# 2. Configuration
# ============================================================

# Keep the semantic graph focused on research concepts/methods,
# rather than implementation plumbing.
DROP_IMPLEMENTATION_TOOLS = True

IMPLEMENTATION_TOOL_NAMES = {
    "pytorch",
    "adam",
}

# Obvious non-entity / citation-like noise.
DROP_NAME_PATTERNS = [
    re.compile(r"\bet al\.?\b", re.IGNORECASE),
]

# Generic dataset-like phrases that are not named datasets.
GENERIC_DATASET_NAMES = {
    "surface reconstruction benchmarks",
    "3d scene",
    "garment",
}

# Generic comparison categories that are not named methods/models.
GENERIC_COMPARISON_NAMES = {
    "poisson-based methods",
}


# ============================================================
# 3. Manual canonicalization map
#
# This intentionally covers only obvious aliases visible in the
# current extraction. It does NOT collapse semantically distinct
# concepts unless they are clear spelling/acronym variants.
# ============================================================

MANUAL_CANONICAL = {
    # Core UDF/SDF concepts
    "udf": "Unsigned Distance Field",
    "udfs": "Unsigned Distance Field",
    "unsigned distance field": "Unsigned Distance Field",
    "unsigned distance fields": "Unsigned Distance Field",

    "sdf": "Signed Distance Field",
    "sdfs": "Signed Distance Field",
    "signed distance field": "Signed Distance Field",
    "signed distance fields": "Signed Distance Field",

    "gwn": "Generalized Winding Number",
    "gwn fields": "Generalized Winding Number",
    "generalized winding number": "Generalized Winding Number",
    "generalized winding numbers": "Generalized Winding Number",
    "generalized winding number fields": "Generalized Winding Number",

    # Geometry concepts
    "voronoi diagram": "Voronoi Diagram",
    "voronoi diagrams": "Voronoi Diagram",
    "projection distance field": "Projection Distance Field",
    "projection distance fields": "Projection Distance Field",
    "bi-directional normal": "Bi-directional Normal",
    "bi-directional normals": "Bi-directional Normal",
    "heat method": "Heat Method",

    # Metrics
    "cd": "Chamfer Distance",
    "chamfer distance": "Chamfer Distance",
    "f-score": "F-Score",
    "f score": "F-Score",
    "hausdorff distance": "Hausdorff Distance",

    # Methods / method aliases
    "capudf": "CAP-UDF",
    "cap-udf": "CAP-UDF",
    "dmu df": "DM-UDF",
    "dmudf": "DM-UDF",
    "dm-udf": "DM-UDF",

    # Datasets / naming variants
    "scannet": "ScanNet",
    "mgn dataset": "MGN",
    "mgn": "MGN",
    "shapenet-car": "ShapeNet-Car",
    "shapenet car": "ShapeNet-Car",
    "shapenet cars": "ShapeNet-Car",
    "shapenet-car": "ShapeNet-Car",

    # Method/concept naming cleanup
    "marching cubes algorithm": "Marching Cubes",
}


# ============================================================
# 4. Manual type overrides for obvious canonical entities
# ============================================================

TYPE_OVERRIDES = {
    "Unsigned Distance Field": "Concept",
    "Signed Distance Field": "Concept",
    "Generalized Winding Number": "Concept",
    "Voronoi Diagram": "Concept",
    "Projection Distance Field": "Concept",
    "Bi-directional Normal": "Concept",
    "Heat Method": "Method",

    "Chamfer Distance": "Metric",
    "F-Score": "Metric",
    "Hausdorff Distance": "Metric",

    "ScanNet": "Dataset",
    "MGN": "Dataset",
    "ShapeNet-Car": "Dataset",

    "VAD": "Method",
    "DCUDF2": "Method",
    "DM-UDF": "Method",
    "MPFs": "Method",
    "GeoUDF": "Method",
    "SuperUDF": "Method",
    "CAP-UDF": "Method",
}


# ============================================================
# 5. Relation type constraints
# ============================================================

ALLOWED_TARGET_TYPES = {
    "INTRODUCES": {"Method", "Model"},
    "USES": {"Concept", "Method", "Model"},
    "ADDRESSES": {"Task"},
    "EVALUATES_ON": {"Dataset"},
    "MEASURED_BY": {"Metric"},
    "IMPROVES": {"Method", "Model"},
    "COMPARED_WITH": {"Method", "Model"},
    "PRODUCES": {"Concept", "Model"},
    "SOLVES": {"Concept", "Task"},
}


# ============================================================
# 6. Helpers
# ============================================================

def normalize_space(text):
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)

    # Normalize Unicode dash variants.
    text = (
        text
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def lookup_key(text):
    text = normalize_space(text)

    return text.casefold()


def canonicalize_name(name):
    name = normalize_space(name)

    if not name:
        return ""

    key = lookup_key(name)

    if key in MANUAL_CANONICAL:
        return MANUAL_CANONICAL[key]

    # Basic singularization only for obvious plural "fields"/"diagrams".
    # Avoid aggressive stemming for research method names.
    return name


def is_noise_name(name):
    name = normalize_space(name)

    if not name:
        return True

    for pattern in DROP_NAME_PATTERNS:
        if pattern.search(name):
            return True

    return False


def evidence_key(item):
    if not isinstance(item, dict):
        return None

    chunk_id = item.get("chunk_id")
    page = item.get("page")

    if chunk_id is None and page is None:
        return None

    return (
        str(chunk_id),
        str(page),
    )


def merge_evidence(target_list, new_items):
    seen = {
        evidence_key(item)
        for item in target_list
        if evidence_key(item) is not None
    }

    for item in new_items or []:
        if not isinstance(item, dict):
            continue

        key = evidence_key(item)

        if key is None:
            continue

        if key in seen:
            continue

        seen.add(key)

        target_list.append(
            {
                "chunk_id": item.get("chunk_id"),
                "page": item.get("page"),
            }
        )


# ============================================================
# 7. Load files
# ============================================================

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Raw KG file not found:\n{RAW_FILE}"
    )

if not PROFILE_FILE.exists():
    raise FileNotFoundError(
        f"Paper profile file not found:\n{PROFILE_FILE}"
    )


records = []

with open(
    RAW_FILE,
    "r",
    encoding="utf-8",
) as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        records.append(
            json.loads(line)
        )


with open(
    PROFILE_FILE,
    "r",
    encoding="utf-8",
) as file:
    paper_profiles = json.load(file)


print("=" * 70)
print("GLOBAL SEMANTIC KG NORMALIZATION")
print("=" * 70)

print(
    f"\nRaw batch records: {len(records)}"
)


# ============================================================
# 8. Collect paper metadata and focal entities
# ============================================================

papers = {}

focal_names = set()


for source_file, profile in paper_profiles.items():
    focal_entities = profile.get(
        "focal_entities",
        []
    )

    normalized_focals = []

    for focal in focal_entities:
        name = canonicalize_name(
            focal.get("name")
        )

        if not name:
            continue

        focal_names.add(
            lookup_key(name)
        )

        aliases = []

        for alias in focal.get(
            "aliases",
            []
        ):
            alias = normalize_space(alias)

            if alias:
                aliases.append(alias)

        normalized_focals.append(
            {
                "name": name,
                "type": focal.get(
                    "type",
                    "Method",
                ),
                "aliases": aliases,
            }
        )

    papers[source_file] = {
        "source_file": source_file,
        "focal_entities": normalized_focals,
    }


# ============================================================
# 9. First pass: build canonical entity pool
# ============================================================

entity_pool = {}

entity_type_votes = defaultdict(
    Counter
)


def ensure_entity(
    raw_name,
    raw_type,
    aliases=None,
    evidence=None,
    source_file=None,
):
    canonical = canonicalize_name(
        raw_name
    )

    if not canonical:
        return None

    if is_noise_name(canonical):
        return None

    key = lookup_key(
        canonical
    )

    if key not in entity_pool:
        entity_pool[key] = {
            "name": canonical,
            "type": None,
            "aliases": [],
            "evidence": [],
            "source_files": [],
        }

    entity = entity_pool[key]

    # Preserve raw names/aliases as searchable aliases.
    candidate_aliases = [
        normalize_space(raw_name)
    ]

    for alias in aliases or []:
        candidate_aliases.append(
            normalize_space(alias)
        )

    known_alias_keys = {
        lookup_key(alias)
        for alias in entity[
            "aliases"
        ]
    }

    for alias in candidate_aliases:
        if not alias:
            continue

        if lookup_key(alias) == key:
            continue

        if lookup_key(alias) in known_alias_keys:
            continue

        entity[
            "aliases"
        ].append(alias)

        known_alias_keys.add(
            lookup_key(alias)
        )

    if raw_type:
        entity_type_votes[
            key
        ][raw_type] += 1

    merge_evidence(
        entity["evidence"],
        evidence or [],
    )

    if (
        source_file
        and source_file
        not in entity[
            "source_files"
        ]
    ):
        entity[
            "source_files"
        ].append(
            source_file
        )

    return entity


for record in records:
    source_file = record.get(
        "source_file"
    )

    for entity in record.get(
        "entities",
        []
    ):
        ensure_entity(
            raw_name=entity.get(
                "name"
            ),
            raw_type=entity.get(
                "type"
            ),
            aliases=entity.get(
                "aliases",
                []
            ),
            evidence=entity.get(
                "evidence",
                []
            ),
            source_file=source_file,
        )


# Ensure focal methods exist even if they were absent in some batches.
for source_file, paper in papers.items():
    for focal in paper[
        "focal_entities"
    ]:
        ensure_entity(
            raw_name=focal[
                "name"
            ],
            raw_type=focal[
                "type"
            ],
            aliases=focal[
                "aliases"
            ],
            evidence=[],
            source_file=source_file,
        )


# ============================================================
# 10. Resolve final entity types
# ============================================================

for key, entity in entity_pool.items():
    canonical_name = entity[
        "name"
    ]

    if canonical_name in TYPE_OVERRIDES:
        entity[
            "type"
        ] = TYPE_OVERRIDES[
            canonical_name
        ]

    else:
        votes = entity_type_votes[
            key
        ]

        if votes:
            entity[
                "type"
            ] = votes.most_common(
                1
            )[0][0]

        else:
            entity[
                "type"
            ] = "Concept"


# ============================================================
# 11. Alias lookup across the whole corpus
# ============================================================

alias_to_key = {}


for key, entity in entity_pool.items():
    alias_to_key[
        key
    ] = key

    for alias in entity[
        "aliases"
    ]:
        alias_to_key[
            lookup_key(alias)
        ] = key


# Add manual alias map to lookup.
for alias, canonical in MANUAL_CANONICAL.items():
    canonical_key = lookup_key(
        canonical
    )

    if canonical_key in entity_pool:
        alias_to_key[
            lookup_key(alias)
        ] = canonical_key


def resolve_entity(name):
    key = lookup_key(
        canonicalize_name(name)
    )

    key = alias_to_key.get(
        key,
        key,
    )

    return entity_pool.get(
        key
    )


# ============================================================
# 12. Second pass: normalize and deduplicate relations
# ============================================================

relation_pool = {}

dropped_reasons = Counter()


def relation_key(
    source_kind,
    source,
    relation_type,
    target,
):
    return (
        source_kind,
        lookup_key(source),
        relation_type,
        lookup_key(target),
    )


for record in records:
    source_file = record.get(
        "source_file"
    )

    for relation in record.get(
        "relations",
        []
    ):
        relation_type = relation.get(
            "type"
        )

        raw_source = normalize_space(
            relation.get(
                "source"
            )
        )

        raw_target = normalize_space(
            relation.get(
                "target"
            )
        )

        if (
            not relation_type
            or not raw_source
            or not raw_target
        ):
            dropped_reasons[
                "missing fields"
            ] += 1

            continue


        target_entity = resolve_entity(
            raw_target
        )

        if target_entity is None:
            dropped_reasons[
                "target entity unresolved"
            ] += 1

            continue


        target_name = target_entity[
            "name"
        ]

        target_type = target_entity[
            "type"
        ]


        # ----------------------------------------------------
        # Remove obvious noise
        # ----------------------------------------------------

        if is_noise_name(
            target_name
        ):
            dropped_reasons[
                "citation-like target"
            ] += 1

            continue


        if (
            relation_type
            == "EVALUATES_ON"
            and lookup_key(
                target_name
            )
            in GENERIC_DATASET_NAMES
        ):
            dropped_reasons[
                "generic dataset phrase"
            ] += 1

            continue


        if (
            relation_type
            == "COMPARED_WITH"
            and lookup_key(
                target_name
            )
            in GENERIC_COMPARISON_NAMES
        ):
            dropped_reasons[
                "generic comparison category"
            ] += 1

            continue


        if (
            DROP_IMPLEMENTATION_TOOLS
            and relation_type
            == "USES"
            and lookup_key(
                target_name
            )
            in IMPLEMENTATION_TOOL_NAMES
        ):
            dropped_reasons[
                "implementation tool"
            ] += 1

            continue


        # ----------------------------------------------------
        # Enforce target-type constraints
        # ----------------------------------------------------

        allowed_types = ALLOWED_TARGET_TYPES.get(
            relation_type
        )

        if (
            allowed_types
            and target_type
            not in allowed_types
        ):
            dropped_reasons[
                (
                    f"type mismatch "
                    f"{relation_type}"
                )
            ] += 1

            continue


        # ----------------------------------------------------
        # Resolve source
        # ----------------------------------------------------

        if raw_source == "PAPER":
            source_kind = "Paper"
            source_name = source_file

            if relation_type != "INTRODUCES":
                dropped_reasons[
                    "non-INTRODUCES PAPER relation"
                ] += 1

                continue

        else:
            source_entity = resolve_entity(
                raw_source
            )

            if source_entity is None:
                dropped_reasons[
                    "source entity unresolved"
                ] += 1

                continue

            source_kind = "Entity"
            source_name = source_entity[
                "name"
            ]


        key = relation_key(
            source_kind,
            source_name,
            relation_type,
            target_name,
        )


        if key not in relation_pool:
            relation_pool[
                key
            ] = {
                "source_kind":
                    source_kind,

                "source":
                    source_name,

                "relation":
                    relation_type,

                "target":
                    target_name,

                "target_type":
                    target_type,

                "evidence":
                    [],

                "source_files":
                    [],
            }


        merged = relation_pool[
            key
        ]


        merge_evidence(
            merged[
                "evidence"
            ],
            relation.get(
                "evidence",
                []
            ),
        )


        if (
            source_file
            and source_file
            not in merged[
                "source_files"
            ]
        ):
            merged[
                "source_files"
            ].append(
                source_file
            )


# ============================================================
# 13. Remove orphan entities after relation cleanup
# ============================================================

used_entity_names = set()


for relation in relation_pool.values():
    if relation[
        "source_kind"
    ] == "Entity":
        used_entity_names.add(
            lookup_key(
                relation[
                    "source"
                ]
            )
        )

    used_entity_names.add(
        lookup_key(
            relation[
                "target"
            ]
        )
    )


# Always preserve focal methods/models.
used_entity_names |= focal_names


canonical_entities = []


for key, entity in entity_pool.items():
    if key not in used_entity_names:
        continue

    entity[
        "aliases"
    ] = sorted(
        set(
            entity[
                "aliases"
            ]
        ),
        key=str.casefold,
    )

    entity[
        "source_files"
    ] = sorted(
        entity[
            "source_files"
        ]
    )

    canonical_entities.append(
        entity
    )


canonical_entities.sort(
    key=lambda item: (
        item[
            "type"
        ],
        item[
            "name"
        ].casefold(),
    )
)


canonical_relations = list(
    relation_pool.values()
)


canonical_relations.sort(
    key=lambda item: (
        item[
            "source"
        ].casefold(),
        item[
            "relation"
        ],
        item[
            "target"
        ].casefold(),
    )
)


# ============================================================
# 14. Statistics
# ============================================================

raw_entity_count = sum(
    len(
        record.get(
            "entities",
            []
        )
    )
    for record in records
)


raw_relation_count = sum(
    len(
        record.get(
            "relations",
            []
        )
    )
    for record in records
)


entity_type_counts = Counter(
    entity[
        "type"
    ]
    for entity in canonical_entities
)


relation_type_counts = Counter(
    relation[
        "relation"
    ]
    for relation in canonical_relations
)


# ============================================================
# 15. Save outputs
# ============================================================

output = {
    "metadata": {
        "raw_batch_records":
            len(records),

        "raw_entity_occurrences":
            raw_entity_count,

        "raw_relation_occurrences":
            raw_relation_count,

        "canonical_entities":
            len(
                canonical_entities
            ),

        "canonical_relations":
            len(
                canonical_relations
            ),

        "entity_type_counts":
            dict(
                entity_type_counts
            ),

        "relation_type_counts":
            dict(
                relation_type_counts
            ),

        "dropped_relation_reasons":
            dict(
                dropped_reasons
            ),
    },

    "papers":
        list(
            papers.values()
        ),

    "entities":
        canonical_entities,

    "relations":
        canonical_relations,
}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        output,
        file,
        ensure_ascii=False,
        indent=2,
    )


with open(
    OUTPUT_ENTITIES,
    "w",
    encoding="utf-8",
) as file:
    for entity in canonical_entities:
        file.write(
            json.dumps(
                entity,
                ensure_ascii=False,
            )
        )

        file.write(
            "\n"
        )


with open(
    OUTPUT_RELATIONS,
    "w",
    encoding="utf-8",
) as file:
    for relation in canonical_relations:
        file.write(
            json.dumps(
                relation,
                ensure_ascii=False,
            )
        )

        file.write(
            "\n"
        )


# ============================================================
# 16. Console summary
# ============================================================

print("\n")
print("=" * 70)
print("NORMALIZATION SUMMARY")
print("=" * 70)

print(
    "Raw entity occurrences:",
    raw_entity_count
)

print(
    "Raw relation occurrences:",
    raw_relation_count
)

print(
    "Canonical entities:",
    len(
        canonical_entities
    )
)

print(
    "Canonical relations:",
    len(
        canonical_relations
    )
)


print("\nEntity types:")

for name, count in sorted(
    entity_type_counts.items()
):
    print(
        f"  {name}: {count}"
    )


print("\nRelation types:")

for name, count in sorted(
    relation_type_counts.items()
):
    print(
        f"  {name}: {count}"
    )


print("\nDropped relation reasons:")

if dropped_reasons:
    for name, count in dropped_reasons.most_common():
        print(
            f"  {name}: {count}"
        )
else:
    print(
        "  None"
    )


print("\n")
print("=" * 70)
print("SAMPLE CANONICAL RELATIONS")
print("=" * 70)


for relation in canonical_relations[
    :30
]:
    source_prefix = (
        "[Paper]"
        if relation[
            "source_kind"
        ] == "Paper"
        else "[Entity]"
    )

    print(
        f"{source_prefix} "
        f"{relation['source']} "
        f"--{relation['relation']}--> "
        f"{relation['target']} "
        f"[{relation['target_type']}]"
    )


print("\nSaved:")
print(OUTPUT_FILE)
print(OUTPUT_ENTITIES)
print(OUTPUT_RELATIONS)
