from pathlib import Path
import json
import re
import unicodedata

from transformers import AutoTokenizer


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

CHUNKS_DIR = (
    PROJECT_ROOT
    / "data"
    / "chunks"
)

OUTPUT_JSONL = (
    CHUNKS_DIR
    / "paper_chunks.jsonl"
)


# ============================================================
# 2. Chunk configuration
# ============================================================

# We use the same tokenizer family as our embedding model.
MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Keep below the embedding model's effective sequence limit.
CHUNK_SIZE = 220

# Consecutive chunks share 40 tokens.
# This reduces information loss near boundaries.
CHUNK_OVERLAP = 40

STEP_SIZE = (
    CHUNK_SIZE
    - CHUNK_OVERLAP
)


# ============================================================
# 3. Load tokenizer
# ============================================================

print("=" * 70)
print("LOADING TOKENIZER")
print("=" * 70)

print(
    f"\nModel: {MODEL_NAME}"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded successfully.")


# ============================================================
# 4. Helper: normalize text
# ============================================================

def normalize_text(text: str) -> str:
    """
    Minimal normalization for retrieval.

    We collapse repeated whitespace but deliberately avoid
    aggressive cleaning because academic papers contain
    equations, symbols and technical notation.
    """

    text = text.replace(
        "\r",
        "\n"
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 5. Helper: make readable paper ID
# ============================================================

def make_paper_id(filename: str) -> str:

    stem = Path(filename).stem

    # Normalize Unicode representation
    stem = unicodedata.normalize(
        "NFKD",
        stem
    )

    # Keep letters/numbers; replace other groups with _
    stem = re.sub(
        r"[^\w]+",
        "_",
        stem,
        flags=re.UNICODE
    )

    stem = stem.strip(
        "_"
    ).lower()

    # Prevent extremely long IDs
    return stem[:80]


# ============================================================
# 6. Helper: parse extracted TXT
# ============================================================

PAGE_PATTERN = re.compile(
    r"={5,}\s*PAGE\s+(\d+)\s*={5,}"
)


def parse_extracted_file(txt_path: Path):

    content = txt_path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    # --------------------------------------------------------
    # Read original PDF filename from metadata
    # --------------------------------------------------------

    source_match = re.search(
        r"^SOURCE_FILE:\s*(.+)$",
        content,
        flags=re.MULTILINE
    )

    if source_match:

        source_file = (
            source_match
            .group(1)
            .strip()
        )

    else:

        source_file = (
            txt_path.stem
            .replace(
                "_extracted",
                ""
            )
            + ".pdf"
        )


    # --------------------------------------------------------
    # Split text using PAGE markers
    # --------------------------------------------------------

    parts = PAGE_PATTERN.split(
        content
    )

    # Format after split:
    #
    # parts[0] = metadata before PAGE 1
    # parts[1] = page number
    # parts[2] = page text
    # parts[3] = next page number
    # parts[4] = next page text
    # ...

    pages = []


    for i in range(
        1,
        len(parts),
        2
    ):

        page_number = int(
            parts[i]
        )

        page_text = (
            parts[i + 1]
            if i + 1 < len(parts)
            else ""
        )

        page_text = normalize_text(
            page_text
        )


        pages.append(
            {
                "page": page_number,
                "text": page_text
            }
        )


    return source_file, pages


# ============================================================
# 7. Helper: split one page into token chunks
# ============================================================

def chunk_page(
    text: str,
    page_number: int,
    paper_id: str,
    source_file: str
):

    if not text:
        return []

    # ========================================================
    # Tokenize while keeping character offsets
    #
    # IMPORTANT:
    # We use token positions only to decide chunk boundaries.
    # We DO NOT decode tokens back into text.
    #
    # Therefore the original paper text is preserved.
    # ========================================================

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False
    )

    token_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    chunks = []

    start_token = 0
    chunk_index = 1


    while start_token < len(token_ids):

        end_token = min(
            start_token + CHUNK_SIZE,
            len(token_ids)
        )


        # ----------------------------------------------------
        # Character range corresponding to this token window
        # ----------------------------------------------------

        char_start = offsets[start_token][0]

        char_end = offsets[end_token - 1][1]


        # ----------------------------------------------------
        # Slice ORIGINAL text
        #
        # Not tokenizer.decode(...)
        # ----------------------------------------------------

        chunk_text = text[
            char_start:char_end
        ].strip()


        chunk_id = (
            f"{paper_id}"
            f"_p{page_number:03d}"
            f"_c{chunk_index:03d}"
        )


        chunks.append(
            {
                "chunk_id": chunk_id,

                "paper_id": paper_id,

                "source_file": source_file,

                "page": page_number,

                "chunk_index": chunk_index,

                "token_start": start_token,

                "token_end": end_token,

                "token_count": (
                    end_token
                    - start_token
                ),

                "char_start": char_start,

                "char_end": char_end,

                "text": chunk_text
            }
        )


        if end_token >= len(token_ids):
            break


        start_token += STEP_SIZE

        chunk_index += 1


    return chunks


# ============================================================
# 8. Find extracted paper files
# ============================================================

txt_files = sorted(
    PROCESSED_DIR.glob(
        "*_extracted.txt"
    )
)


if not txt_files:

    raise FileNotFoundError(
        f"No extracted TXT files found in:\n"
        f"{PROCESSED_DIR}"
    )


CHUNKS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("\n")
print("=" * 70)
print("CHUNKING PAPER COLLECTION")
print("=" * 70)

print(
    f"\nFound {len(txt_files)} extracted papers."
)

print(
    f"Chunk size: {CHUNK_SIZE} tokens"
)

print(
    f"Overlap: {CHUNK_OVERLAP} tokens"
)

print(
    f"Step size: {STEP_SIZE} tokens"
)


# ============================================================
# 9. Process every paper
# ============================================================

all_chunks = []

paper_summaries = []


for paper_number, txt_path in enumerate(
    txt_files,
    start=1
):

    print("\n")
    print("=" * 70)

    print(
        f"[{paper_number}/{len(txt_files)}]"
    )

    print(
        txt_path.name
    )

    print("=" * 70)


    source_file, pages = (
        parse_extracted_file(
            txt_path
        )
    )


    paper_id = make_paper_id(
        source_file
    )


    print(
        f"Source PDF: {source_file}"
    )

    print(
        f"Paper ID: {paper_id}"
    )

    print(
        f"Pages found: {len(pages)}"
    )


    paper_chunks = []


    for page_data in pages:

        page_chunks = chunk_page(
            text=page_data["text"],
            page_number=page_data["page"],
            paper_id=paper_id,
            source_file=source_file
        )


        paper_chunks.extend(
            page_chunks
        )


        print(
            f"Page "
            f"{page_data['page']:03d}: "
            f"{len(page_chunks)} chunk(s)"
        )


    all_chunks.extend(
        paper_chunks
    )


    paper_summaries.append(
        {
            "source_file": source_file,
            "paper_id": paper_id,
            "pages": len(pages),
            "chunks": len(paper_chunks)
        }
    )


    print(
        f"\nPaper chunks: "
        f"{len(paper_chunks)}"
    )


# ============================================================
# 10. Save JSONL
# ============================================================

with open(
    OUTPUT_JSONL,
    "w",
    encoding="utf-8"
) as output_file:

    for chunk in all_chunks:

        output_file.write(
            json.dumps(
                chunk,
                ensure_ascii=False
            )
        )

        output_file.write(
            "\n"
        )


# ============================================================
# 11. Collection summary
# ============================================================

print("\n")
print("=" * 70)
print("CHUNKING SUMMARY")
print("=" * 70)


for summary in paper_summaries:

    print(
        f"\n{summary['source_file']}"
    )

    print(
        f"  Pages : "
        f"{summary['pages']}"
    )

    print(
        f"  Chunks: "
        f"{summary['chunks']}"
    )


print("\n" + "-" * 70)

print(
    "Total papers:",
    len(paper_summaries)
)

print(
    "Total chunks:",
    len(all_chunks)
)

print(
    "Chunk size:",
    CHUNK_SIZE
)

print(
    "Overlap:",
    CHUNK_OVERLAP
)

print(
    "\nSaved:"
)

print(
    OUTPUT_JSONL
)


# ============================================================
# 12. Preview first three chunks
# ============================================================

print("\n")
print("=" * 70)
print("CHUNK PREVIEW")
print("=" * 70)


for chunk in all_chunks[:3]:

    print("\nChunk ID:")
    print(
        chunk["chunk_id"]
    )

    print(
        "Paper:",
        chunk["source_file"]
    )

    print(
        "Page:",
        chunk["page"]
    )

    print(
        "Tokens:",
        chunk["token_count"]
    )

    print("\nText:")

    print(
        chunk["text"][:700]
    )

    print(
        "\n" + "-" * 70
    )