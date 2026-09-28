from src.config import (
    INDEX_PATH,
    CHUNKS_PATH,
    TOP_K,
)
from src.embeddings import load_embedding_model
from src.rag import create_rag_prompt
from src.retriever import retrieve
from src.vector_store import (
    load_chunks,
    load_index,
)


def main():

    if not INDEX_PATH.exists():
        print("Knowledge base not found.")
        print("Run:")
        print("python build_index.py")
        return

    print("Loading knowledge base...")

    index = load_index(
        INDEX_PATH
    )

    chunks = load_chunks(
        CHUNKS_PATH
    )

    model = load_embedding_model()

    print(
        f"Loaded {len(chunks)} chunks."
    )

    print()
    print("RAG Knowledge Assistant")
    print("Type 'exit' to quit.")
    print()

    while True:

        question = input("Question: ")

        if question.lower() == "exit":
            break

        results = retrieve(
            query=question,
            model=model,
            index=index,
            chunks=chunks,
            top_k=TOP_K,
        )

        prompt = create_rag_prompt(
            question=question,
            retrieved_chunks=results,
        )

        print()
        print("Retrieved sources:")

        for result in results:

            print(
                f"- {result['source']} "
                f"(score={result['score']:.3f})"
            )

        print()
        print("RAG prompt:")
        print(prompt)

        print()
        print("-" * 60)


if __name__ == "__main__":
    main()