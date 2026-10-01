from pathlib import Path

from pypdf import PdfReader


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "papers"
    / "LHM Large Animatable Human Reconstruction Model.pdf"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "lhm_extracted.txt"
)


# ============================================================
# 2. Check PDF
# ============================================================

if not PDF_PATH.exists():
    raise FileNotFoundError(
        f"PDF not found:\n{PDF_PATH}"
    )


print("PDF found:")
print(PDF_PATH)


# ============================================================
# 3. Create output directory
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. Open PDF
# ============================================================

reader = PdfReader(PDF_PATH)

print("\nNumber of PDF pages:")
print(len(reader.pages))


# ============================================================
# 5. Extract page-by-page
# ============================================================

pages = []


for page_number, page in enumerate(
    reader.pages,
    start=1
):

    text = page.extract_text() or ""

    text = text.strip()

    pages.append(
        {
            "page": page_number,
            "text": text
        }
    )

    print(
        f"Page {page_number}: "
        f"{len(text)} characters"
    )


# ============================================================
# 6. Save extracted text
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    for page in pages:

        file.write(
            f"\n\n"
            f"================ PAGE {page['page']} ================\n\n"
        )

        file.write(
            page["text"]
        )


# ============================================================
# 7. Statistics
# ============================================================

total_characters = sum(
    len(page["text"])
    for page in pages
)

non_empty_pages = sum(
    1
    for page in pages
    if page["text"]
)


print("\nExtraction completed!")

print(
    "Non-empty pages:",
    non_empty_pages
)

print(
    "Total extracted characters:",
    total_characters
)

print(
    "\nSaved to:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# 8. Preview first useful text
# ============================================================

print("\n" + "=" * 70)
print("TEXT PREVIEW")
print("=" * 70)


preview = ""

for page in pages:

    if page["text"]:
        preview += page["text"] + "\n"

    if len(preview) >= 2000:
        break


print(
    preview[:2000]
)