from pathlib import Path
import pymupdf

def load_text_file(path: Path) -> list[dict]:
    """Load a text file as a single document."""

    text = path.read_text(
        encoding="utf-8"
    )

    return [
        {
            "text": text,
            "source": path.name,
            "page": None,
        }
    ]


def load_pdf_file(path: Path) -> list[dict]:
    """Extract PDF text while preserving page numbers."""

    document = pymupdf.open(path)

    pages = []

    for page_number, page in enumerate(
        document,
        start=1,
    ):

        text = page.get_text().strip()

        if not text:
            continue

        pages.append(
            {
                "text": text,
                "source": path.name,
                "page": page_number,
            }
        )

    document.close()

    return pages


def load_document(path: Path) -> list[dict]:
    """Load a supported document."""

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return load_text_file(path)

    if suffix == ".pdf":
        return load_pdf_file(path)

    raise ValueError(
        f"Unsupported file type: {suffix}"
    )


def load_documents(directory: Path) -> list[dict]:
    """Load all supported documents."""

    documents = []

    for path in directory.iterdir():

        if path.suffix.lower() not in {
            ".txt",
            ".pdf",
        }:
            continue

        documents.extend(
            load_document(path)
        )

    return documents