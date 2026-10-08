import os
import functools
from typing import Optional

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@functools.lru_cache(maxsize=1)
def get_embeddings():
    """Cached HuggingFaceEmbeddings instance to prevent repeated 30+ second reload."""
    print("⏳ Initializing embedding model (cached)...")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )


def build_vector_store(transcript: str, in_memory: bool = True):
    """
    Build vector store for transcript chunks.
    Uses in-memory Chroma by default for blazing speed and zero SQLite lock conflicts.
    """
    print("Building vector store...")
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={'chunk_index': i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()

    persist_dir = None if in_memory else CHROMA_DIR

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_dir
    )

    return vector_store


def load_vector_store():
    from langchain_chroma import Chroma
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    return vector_store


def get_retriever(vector_store, k: int = 4):
    return vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={"k": k}
    )
