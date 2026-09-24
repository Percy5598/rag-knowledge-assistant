def build_prompt(question, context):
    """
    Build a prompt for an LLM using retrieved document context.

    Parameters
    ----------
    question : str
        The user's question.

    context : str
        Relevant text retrieved from the document collection.

    Returns
    -------
    str
        A prompt containing instructions, context, and the question.
    """

    prompt = f"""
You are a helpful company knowledge assistant.

Answer the question using only the provided context.

If the answer cannot be found in the context,
say that you do not have enough information.

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt


def build_context(chunks):
    """
    Combine retrieved document chunks into a single context string.

    Parameters
    ----------
    chunks : list[str]
        Retrieved document chunks.

    Returns
    -------
    str
        Combined context.
    """

    return "\n\n".join(chunks)


def create_rag_prompt(question, retrieved_chunks):
    """
    Create a complete RAG prompt from a question
    and retrieved document chunks.

    Parameters
    ----------
    question : str
        The user's question.

    retrieved_chunks : list[str]
        Relevant chunks retrieved from the document collection.

    Returns
    -------
    str
        Final prompt ready to be sent to an LLM.
    """

    context = build_context(retrieved_chunks)

    return build_prompt(
        question=question,
        context=context
    )

