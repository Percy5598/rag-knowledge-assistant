from pathlib import Path

import faiss
import numpy as np


def create_index(embeddings):
    """Create a FAISS vector index."""

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def search_index(
    index,
    query_embedding,
    top_k: int = 3,
):
    """Search the vector index."""

    query_embedding = np.asarray(
        [query_embedding],
        dtype="float32",
    )

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    return scores[0], indices[0]


def save_index(index, path: Path):
    """Save FAISS index to disk."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(path),
    )


def load_index(path: Path):
    """Load FAISS index from disk."""

    return faiss.read_index(
        str(path)
    )