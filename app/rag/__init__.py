"""轻量 RAG：切块 -> 向量化 -> 向量检索 + BM25 混合召回。"""

from .chunker import Chunk, split_text
from .embedder import HashingEmbedder, OpenAICompatEmbedder, build_embedder
from .loader import load_chunks
from .retriever import Hit, Retriever, build_retriever
from .store import VectorStore

__all__ = [
    "Chunk",
    "split_text",
    "HashingEmbedder",
    "OpenAICompatEmbedder",
    "build_embedder",
    "load_chunks",
    "Hit",
    "Retriever",
    "build_retriever",
    "VectorStore",
]
