# RAG Knowledge Assistant

A simple **Retrieval-Augmented Generation (RAG)** knowledge assistant that retrieves relevant information from company documents using semantic search.

## 🚀 Live Demo

**[Try the RAG Knowledge Assistant](https://rag-knowledge-assistant-101.streamlit.app/)**

## 🔍 Features

* PDF and TXT document processing
* Text chunking with overlap
* Sentence Transformer embeddings
* FAISS vector search
* Semantic document retrieval
* Source-aware RAG prompt generation
* Retrieval evaluation using Recall@K, Precision@K and MRR
* Streamlit interface

## 🛠️ Tech Stack

**Python · Sentence Transformers · FAISS · PyMuPDF · NumPy · Streamlit · Pytest**# RAG Knowledge Assistant

## 🧠 Architecture

```text
Documents
   ↓
Text Extraction
   ↓
Chunking
   ↓
Embeddings
   ↓
FAISS Vector Search
   ↓
Relevant Context
   ↓
RAG Prompt
   ↓
Answer Generation
```

## 📌 Purpose

This project was built to understand and implement the core components of a RAG system from the ground up, rather than relying entirely on high-level frameworks.

The architecture is designed so that an LLM can be connected to the generation component later.

## 👨‍💻 Author

**Prashant Shrestha**
