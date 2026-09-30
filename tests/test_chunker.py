from src.chunker import word_chunker


def test_word_chunker_creates_chunks():
    text = "one two three four five six seven eight"

    chunks = word_chunker(
        text,
        chunk_size=4,
        overlap=1,
    )

    assert len(chunks) > 1
    assert chunks[0] == "one two three four"


def test_word_chunker_overlap():
    text = "one two three four five six"

    chunks = word_chunker(
        text,
        chunk_size=4,
        overlap=1,
    )

    assert chunks[1].startswith("four")


def test_word_chunker_invalid_chunk_size():
    text = "one two three"

    try:
        word_chunker(
            text,
            chunk_size=0,
            overlap=0,
        )
    except ValueError:
        assert True
    else:
        assert False