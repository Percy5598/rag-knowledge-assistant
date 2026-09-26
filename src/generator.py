"""
Generator will take the prompt, send it to LLM and return the answer.
RAG pipeline

question
    ↓
retrieve()
    ↓
retrieved_chunks
    ↓
create_rag_prompt()
    ↓
prompt
    ↓
generate()
    ↓
answer

This is a temporary generator without LLM
"""
def generate_answer(prompt):
    """
    Generate an answer from a prompt.

    Temporary mock generator used while we do not
    have a real LLM connected.
    """

    if "30 days" in prompt:
        return "Employees receive 30 days of annual leave per calendar year."

    return "I do not have enough information to answer the question."
