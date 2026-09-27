from pathlib import Path

import fitz


def load_text_file(path: Path) -> str:
    """Load text from a .txt file."""
    return path.read_text(encoding="utf-8")


def load_pdf_file(path: Path) -> str:
    """Extract text from a PDF file."""
    document = fitz.open(path)

    pages = []

    for page in document:
        text = page.get_text()
        pages.append(text)

    document.close()

    return "\n".join(pages)


def load_document(path: Path) -> str:
    """Load a supported document."""
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return load_text_file(path)

    if suffix == ".pdf":
        return load_pdf_file(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def load_documents(directory: Path) -> list[dict]:
    """Load all supported documents from a directory."""

    documents = []

    for path in directory.iterdir():

        if path.suffix.lower() not in {".txt", ".pdf"}:
            continue

        text = load_document(path)

        documents.append(
            {
                "text": text,
                "source": path.name,
            }
        )

    return documents