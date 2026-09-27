from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME


def load_embedding_model():
    """Load the sentence embedding model."""

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def create_embeddings(
    texts: list[str],
    model,
):
    """Convert texts into embedding vectors."""

    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )