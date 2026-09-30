def recall_at_k(
    retrieved_results,
    relevant_chunk_ids,
):
    """Calculate Recall@K."""

    retrieved_ids = {
        result["chunk_id"]
        for result in retrieved_results
    }

    relevant_ids = set(relevant_chunk_ids)

    if not relevant_ids:
        return 0.0

    found = retrieved_ids.intersection(relevant_ids)

    return len(found) / len(relevant_ids)


def precision_at_k(
    retrieved_results,
    relevant_chunk_ids,
):
    """Calculate Precision@K."""

    if not retrieved_results:
        return 0.0

    relevant_ids = set(relevant_chunk_ids)

    retrieved_relevant = sum(
        1
        for result in retrieved_results
        if result["chunk_id"] in relevant_ids
    )

    return retrieved_relevant / len(retrieved_results)


def reciprocal_rank(
    retrieved_results,
    relevant_chunk_ids,
):
    """Calculate Reciprocal Rank."""

    relevant_ids = set(relevant_chunk_ids)

    for rank, result in enumerate(
        retrieved_results,
        start=1,
    ):
        if result["chunk_id"] in relevant_ids:
            return 1 / rank

    return 0.0