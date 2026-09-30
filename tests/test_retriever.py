import numpy as np
import pytest

from src.retriever import retrieve


class FakeModel:

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ):
        return np.array(
            [
                [1.0, 0.0]
            ]
        )


class FakeIndex:

    def search(
        self,
        query_embedding,
        top_k,
    ):
        scores = np.array(
            [
                [0.95, 0.20]
            ],
            dtype="float32",
        )

        indices = np.array(
            [
                [0, 1]
            ],
            dtype="int64",
        )

        return scores, indices


def test_retrieve_returns_most_relevant_chunk():

    chunks = [
        {
            "chunk_id": 0,
            "text": "Annual leave is 30 days.",
            "source": "company_policy.txt",
        },
        {
            "chunk_id": 1,
            "text": "Employees can work remotely.",
            "source": "company_policy.txt",
        },
    ]

    results = retrieve(
        query="How many annual leave days?",
        model=FakeModel(),
        index=FakeIndex(),
        chunks=chunks,
        top_k=2,
        similarity_threshold=0.40,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == 0
    assert results[0]["score"] == pytest.approx(0.95)
    