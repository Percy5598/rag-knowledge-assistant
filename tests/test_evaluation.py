from src.evaluation import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k():
    results = [
        {"chunk_id": 2},
        {"chunk_id": 5},
        {"chunk_id": 7},
    ]

    assert recall_at_k(
        results,
        [5],
    ) == 1.0


def test_precision_at_k():
    results = [
        {"chunk_id": 2},
        {"chunk_id": 5},
        {"chunk_id": 7},
    ]

    assert precision_at_k(
        results,
        [5],
    ) == 1 / 3


def test_reciprocal_rank():
    results = [
        {"chunk_id": 2},
        {"chunk_id": 5},
        {"chunk_id": 7},
    ]

    assert reciprocal_rank(
        results,
        [5],
    ) == 0.5