from src.vector_store import search_index
from src.config import SIMILARITY_THRESHOLD

def retrieve(
    query: str,
    model,
    index,
    chunks: list[dict],
    top_k: int = 3,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    """Retrieve relevant chunks above a similarity threshold."""

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

    for score, index_position in zip(
        scores,
        indices,
    ):

        if index_position < 0:
            continue

        if score < similarity_threshold:
            continue

        result = chunks[index_position].copy()

        result["score"] = float(score)

        results.append(result)

    return results