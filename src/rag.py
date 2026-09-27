def create_rag_prompt(
    question: str,
    retrieved_chunks: list[dict],
) -> str:
    """Create a grounded RAG prompt."""

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):

        source = chunk.get(
            "source",
            "Unknown source",
        )

        text = chunk["text"]

        context_parts.append(
            f"[Source {i}: {source}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a corporate knowledge assistant.

Answer the user's question using only the provided context.

If the context does not contain enough information to answer
the question, say that the information was not found in the
provided documents.

Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    return prompt