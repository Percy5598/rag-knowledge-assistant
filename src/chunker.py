import re


def clean_text(text: str) -> str:
    """Normalize whitespace."""

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def word_chunker(
    text: str,
    chunk_size: int = 200,
    overlap: int = 40,
) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    text = clean_text(text)

    words = text.split()

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):

        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            break

        chunk = " ".join(chunk_words)

        chunks.append(chunk)

        if start + chunk_size >= len(words):
            break

    return chunks