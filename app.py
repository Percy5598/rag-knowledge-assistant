import streamlit as st

from src.config import CHUNKS_PATH, INDEX_PATH, TOP_K
from src.embeddings import load_embedding_model
from src.rag import create_rag_prompt
from src.retriever import retrieve
from src.vector_store import load_chunks, load_index


st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def load_resources():
    model = load_embedding_model()
    index = load_index(INDEX_PATH)
    chunks = load_chunks(CHUNKS_PATH)

    return model, index, chunks


st.title("📚 RAG Knowledge Assistant")

st.write(
    "Ask questions about the company's internal documents."
)

if not INDEX_PATH.exists():
    st.error(
        "Knowledge base not found. "
        "Run `python build_index.py` first."
    )
    st.stop()


model, index, chunks = load_resources()

st.sidebar.header("Knowledge Base")

st.sidebar.write(
    f"Documents: "
    f"{len(set(chunk['source'] for chunk in chunks))}"
)

st.sidebar.write(
    f"Chunks: {len(chunks)}"
)

question = st.text_input(
    "Ask a question",
    placeholder="How many days of annual leave do employees receive?",
)


if question:

    results = retrieve(
        query=question,
        model=model,
        index=index,
        chunks=chunks,
        top_k=TOP_K,
    )

    if not results:

        st.warning(
            "No relevant information was found "
            "in the provided documents."
        )

    else:

        st.subheader("Retrieved Information")

        for result in results:

            with st.expander(
                f"{result['source']} "
                f"(similarity: {result['score']:.3f})"
            ):

                st.write(result["text"])

                if result.get("page"):
                    st.caption(
                        f"Page {result['page']}"
                    )

        prompt = create_rag_prompt(
            question=question,
            retrieved_chunks=results,
        )

        st.subheader("RAG Prompt")

        st.code(
            prompt,
            language="text",
        )

        st.info(
            "Retrieval is working. "
            "A production LLM can be connected to this "
            "prompt for final answer generation."
        )