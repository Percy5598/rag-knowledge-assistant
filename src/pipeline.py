from src.chunker import word_chunker
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    TOP_K,
)
from src.embeddings import (
    create_embeddings,
    load_embedding_model,
)
from src.rag import create_rag_prompt
from src.retriever import retrieve
from src.vector_store import create_index


def build_knowledge_base(documents: list[dict]):
    """
    Convert documents into chunks and build
    a FAISS vector index.
    """

    chunks = []

    chunk_id = 0

    for document in documents:

        document_chunks = word_chunker(
            document["text"],
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        for chunk in document_chunks:

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "source": document["source"],
                    "page": document["page"],
                }
            )

            chunk_id += 1

    model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = create_embeddings(
        texts,
        model,
    )

    index = create_index(
        embeddings
    )

    return model, index, chunks


def answer_question(
    question: str,
    model,
    index,
    chunks: list[dict],
):
    """Run the complete retrieval + generation pipeline."""

    retrieved_chunks = retrieve(
        query=question,
        model=model,
        index=index,
        chunks=chunks,
        top_k=TOP_K,
    )

    prompt = create_rag_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    return {
        "prompt": prompt,
        "retrieved_chunks": retrieved_chunks,
    }