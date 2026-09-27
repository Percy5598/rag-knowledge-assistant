from pathlib import Path

from src.config import RAW_DOCUMENTS_DIR
from src.loader import load_documents
from src.pipeline import (
    answer_question,
    build_knowledge_base,
)


def main():

    print("Loading documents...")

    documents = load_documents(
        RAW_DOCUMENTS_DIR
    )

    if not documents:
        print(
            "No documents found in:"
        )
        print(RAW_DOCUMENTS_DIR)
        return

    print(
        f"Loaded {len(documents)} documents."
    )

    print("Building knowledge base...")

    model, index, chunks = build_knowledge_base(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    print()
    print("RAG Knowledge Assistant")
    print("Type 'exit' to quit.")
    print()

    while True:

        question = input("Question: ")

        if question.lower() == "exit":
            break

        result = answer_question(
            question=question,
            model=model,
            index=index,
            chunks=chunks,
        )

        print()
        print("Retrieved sources:")

        for chunk in result["retrieved_chunks"]:

            print(
                f"- {chunk['source']} "
                f"(score={chunk['score']:.3f})"
            )

        print()
        print("RAG prompt:")
        print(result["prompt"])

        print()
        print("-" * 60)


if __name__ == "__main__":
    main()