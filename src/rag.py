def format_source(chunk: dict, source_number: int) -> str:
    """Create a readable citation label for a retrieved chunk."""

    source = chunk.get(
        "source",
        "Unknown source",
    )

    page = chunk.get("page")

    if page is not None:
        citation = f"{source}, page {page}"
    else:
        citation = source

    return f"[Source {source_number}: {citation}]"


def create_rag_prompt(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Create a grounded RAG prompt with source information.

    The future LLM should use only the retrieved context
    and cite the relevant source when answering.
    """

    context_parts = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        source_label = format_source(
            chunk,
            i,
        )

        context_parts.append(
            f"{source_label}\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a corporate knowledge assistant.

Answer the user's question using only the provided context.

Rules:
1. Do not invent information.
2. If the answer cannot be found in the context, say:
   "The information was not found in the provided documents."
3. Cite the source that supports your answer.
4. If multiple sources support the answer, cite each relevant source.
5. Keep the answer concise and factual.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    return prompt