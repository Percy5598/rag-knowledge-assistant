from src.vector_store import search_index


def retrieve(
    query: str,
    model,
    index,
    chunks: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Retrieve the most relevant chunks."""

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores, indices = search_index(
        index,
        query_embedding,
        top_k=top_k,
    )

    results = []

    for score, index_position in zip(scores, indices):

        if index_position < 0:
            continue

        result = chunks[index_position].copy()

        result["score"] = float(score)

        results.append(result)

    return results