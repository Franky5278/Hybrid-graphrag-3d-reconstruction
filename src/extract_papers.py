from pathlib import Path
import re

from pypdf import PdfReader


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PAPERS_DIR = (
    PROJECT_ROOT
    / "data"
    / "papers"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# 2. Unicode cleaning
# ============================================================

# UTF-16 surrogate range:
#
#   U+D800 ~ U+DFFF
#
# These code points sometimes appear because PDF fonts/glyphs
# are decoded incorrectly by PDF text extraction.
#
# Python can temporarily hold them inside a string, but they
# cannot be directly written as valid UTF-8.
#
# We replace ONLY those invalid surrogate characters with:
#
#   �
#
# This avoids crashing the whole extraction pipeline.

SURROGATE_PATTERN = re.compile(
    r"[\ud800-\udfff]"
)


def clean_unicode_text(text: str):
    """
    Remove invalid Unicode surrogate characters produced
    during PDF text extraction.

    Returns:
        cleaned_text: cleaned string
        invalid_count: number of replaced characters
    """

    invalid_characters = SURROGATE_PATTERN.findall(
        text
    )

    invalid_count = len(
        invalid_characters
    )

    cleaned_text = SURROGATE_PATTERN.sub(
        "�",
        text
    )

    return cleaned_text, invalid_count


# ============================================================
# 3. Check folders
# ============================================================

if not PAPERS_DIR.exists():

    raise FileNotFoundError(
        f"Papers directory not found:\n"
        f"{PAPERS_DIR}"
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. Find all PDF files
# ============================================================

pdf_files = sorted(
    PAPERS_DIR.glob("*.pdf")
)


if not pdf_files:

    raise FileNotFoundError(
        f"No PDF files found in:\n"
        f"{PAPERS_DIR}"
    )


print("=" * 70)
print("PDF COLLECTION")
print("=" * 70)

print(
    f"\nFound {len(pdf_files)} PDF files:\n"
)


for pdf_path in pdf_files:

    print(
        f"- {pdf_path.name}"
    )


# ============================================================
# 5. Collection statistics
# ============================================================

successful_papers = 0
failed_papers = 0

collection_total_pages = 0
collection_non_empty_pages = 0
collection_total_characters = 0
collection_invalid_unicode = 0


# ============================================================
# 6. Process every PDF
# ============================================================

for pdf_index, pdf_path in enumerate(
    pdf_files,
    start=1
):

    print("\n")
    print("=" * 70)

    print(
        f"[{pdf_index}/{len(pdf_files)}] Processing:"
    )

    print(
        pdf_path.name
    )

    print("=" * 70)


    # --------------------------------------------------------
    # 6.1 Open PDF
    # --------------------------------------------------------

    try:

        reader = PdfReader(
            pdf_path
        )

    except Exception as error:

        print(
            "\nERROR: Failed to open PDF"
        )

        print(
            error
        )

        failed_papers += 1

        continue


    total_pages = len(
        reader.pages
    )


    print(
        f"Pages: {total_pages}"
    )


    # --------------------------------------------------------
    # 6.2 Extract pages
    # --------------------------------------------------------

    extracted_pages = []

    paper_invalid_unicode = 0


    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            # ================================================
            # PDF -> raw Python string
            # ================================================

            raw_text = (
                page.extract_text()
                or ""
            )


            # ================================================
            # Unicode cleaning happens HERE
            #
            # page.extract_text()
            #       ↓
            # raw_text
            #       ↓
            # clean_unicode_text()
            #       ↓
            # safe UTF-8 text
            # ================================================

            text, invalid_count = (
                clean_unicode_text(
                    raw_text
                )
            )


            paper_invalid_unicode += (
                invalid_count
            )


            # Remove only leading/trailing whitespace.
            # We deliberately do not aggressively "clean"
            # the academic text at this stage.

            text = text.strip()


            if invalid_count > 0:

                print(
                    f"Page {page_number:03d}: "
                    f"{len(text):5d} characters "
                    f"| replaced "
                    f"{invalid_count} invalid "
                    f"Unicode character(s)"
                )

            else:

                print(
                    f"Page {page_number:03d}: "
                    f"{len(text):5d} characters"
                )


        except Exception as error:

            print(
                f"Page {page_number:03d}: "
                f"EXTRACTION ERROR"
            )

            print(
                f"    {error}"
            )

            text = ""


        extracted_pages.append(
            {
                "page": page_number,
                "text": text
            }
        )


    # --------------------------------------------------------
    # 6.3 Calculate paper statistics
    # --------------------------------------------------------

    non_empty_pages = sum(
        1
        for page_data in extracted_pages
        if page_data["text"]
    )


    total_characters = sum(
        len(page_data["text"])
        for page_data in extracted_pages
    )


    # --------------------------------------------------------
    # 6.4 Build output path
    # --------------------------------------------------------

    output_filename = (
        f"{pdf_path.stem}"
        f"_extracted.txt"
    )


    output_path = (
        OUTPUT_DIR
        / output_filename
    )


    # --------------------------------------------------------
    # 6.5 Save UTF-8 text
    # --------------------------------------------------------

    try:

        with open(
            output_path,
            "w",
            encoding="utf-8",

            # Second safety layer:
            # even if another strange invalid character
            # somehow survives, do not crash the pipeline.
            errors="replace"

        ) as file:

            # -----------------------------------------------
            # Paper-level metadata
            # -----------------------------------------------

            file.write(
                f"SOURCE_FILE: "
                f"{pdf_path.name}\n"
            )

            file.write(
                f"TOTAL_PAGES: "
                f"{total_pages}\n"
            )

            file.write(
                f"NON_EMPTY_PAGES: "
                f"{non_empty_pages}\n"
            )

            file.write(
                f"TOTAL_CHARACTERS: "
                f"{total_characters}\n"
            )

            file.write(
                f"INVALID_UNICODE_REPLACED: "
                f"{paper_invalid_unicode}\n"
            )


            # -----------------------------------------------
            # Page-level text
            # -----------------------------------------------

            for page_data in extracted_pages:

                file.write(
                    "\n\n"
                    "================================"
                    f" PAGE {page_data['page']} "
                    "================================"
                    "\n\n"
                )

                file.write(
                    page_data["text"]
                )


    except Exception as error:

        print(
            "\nERROR: Failed to write output file"
        )

        print(
            error
        )

        failed_papers += 1

        continue


    # --------------------------------------------------------
    # 6.6 Update collection statistics
    # --------------------------------------------------------

    successful_papers += 1

    collection_total_pages += (
        total_pages
    )

    collection_non_empty_pages += (
        non_empty_pages
    )

    collection_total_characters += (
        total_characters
    )

    collection_invalid_unicode += (
        paper_invalid_unicode
    )


    # --------------------------------------------------------
    # 6.7 Paper summary
    # --------------------------------------------------------

    print("\nFinished successfully.")

    print(
        f"Non-empty pages: "
        f"{non_empty_pages}/{total_pages}"
    )

    print(
        f"Characters: "
        f"{total_characters}"
    )

    print(
        f"Invalid Unicode replaced: "
        f"{paper_invalid_unicode}"
    )

    print(
        "Saved:"
    )

    print(
        output_path
    )


# ============================================================
# 7. Collection summary
# ============================================================

print("\n")
print("=" * 70)
print("COLLECTION SUMMARY")
print("=" * 70)

print(
    "PDF files found:",
    len(pdf_files)
)

print(
    "Successfully processed:",
    successful_papers
)

print(
    "Failed:",
    failed_papers
)

print(
    "Total pages:",
    collection_total_pages
)

print(
    "Non-empty pages:",
    collection_non_empty_pages
)

print(
    "Total extracted characters:",
    collection_total_characters
)

print(
    "Invalid Unicode characters replaced:",
    collection_invalid_unicode
)

print(
    "\nProcessed files saved in:"
)

print(
    OUTPUT_DIR
)