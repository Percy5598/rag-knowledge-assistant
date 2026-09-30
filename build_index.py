from src.chunker import word_chunker
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    INDEX_PATH,
    RAW_DOCUMENTS_DIR,
    CHUNKS_PATH,
)
from src.embeddings import create_embeddings, load_embedding_model
from src.loader import load_documents
from src.vector_store import create_index, save_chunks, save_index


def main():
    print("Loading documents...")

    documents = load_documents(
        RAW_DOCUMENTS_DIR
    )

    if not documents:
        print("No documents found.")
        return

    print(
        f"Loaded {len(documents)} documents."
    )

    print("Creating chunks...")

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

    print(
        f"Created {len(chunks)} chunks."
    )

    print("Creating embeddings...")

    model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = create_embeddings(
        texts,
        model,
    )

    print("Creating FAISS index...")

    index = create_index(
        embeddings
    )

    print("Saving knowledge base...")

    save_index(
        index,
        INDEX_PATH,
    )

    save_chunks(
        chunks,
        CHUNKS_PATH,
    )

    print()
    print(
        "Knowledge base created successfully."
    )

    print(
        f"Index: {INDEX_PATH}"
    )

    print(
        f"Chunks: {CHUNKS_PATH}"
    )


if __name__ == "__main__":
    main()