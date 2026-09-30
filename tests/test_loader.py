from pathlib import Path

from src.loader import load_text_file


def test_load_text_file(tmp_path):
    file_path = tmp_path / "test.txt"

    file_path.write_text(
        "Hello world.",
        encoding="utf-8",
    )

    documents = load_text_file(file_path)

    assert len(documents) == 1
    assert documents[0]["text"] == "Hello world."
    assert documents[0]["source"] == "test.txt"
    assert documents[0]["page"] is None