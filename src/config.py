from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DOCUMENTS_DIR = PROJECT_ROOT / "data" / "raw" / "documents"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "processed" / "vector_store"

INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 200
CHUNK_OVERLAP = 40

TOP_K = 3
SIMILARITY_THRESHOLD = 0.40