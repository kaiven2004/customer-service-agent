from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Settings
from .bm25 import BM25, tokenize
from .chunker import Chunk
from .embedder import Embedder, build_embedder
from .loader import load_chunks
from .store import VectorStore


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float      # 混合分数（用于排序）
    cosine: float     # 向量相似度（用于判断“知识库里到底有没有”）
    bm25: float


class Retriever:
    """向量召回 + BM25 召回，按加权分数融合。

    只用向量时，专有名词（订单号、型号）容易飘；只用 BM25 时，
    “怎么退”和“退款流程”这种不同说法召回不到。混合检索是最便宜的折中。
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        top_k: int = 4,
        min_score: float = 0.18,
        alpha: float = 0.65,
    ):
        self._store = store
        self._embedder = embedder
        self._top_k = top_k
        self._min_score = min_score
        self._alpha = alpha
        self._bm25 = BM25([tokenize(chunk.embed_text()) for chunk in store.records])

    @property
    def chunks(self) -> list[Chunk]:
        return self._store.records

    @property
    def min_score(self) -> float:
        return self._min_score

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        query = query.strip()
        if not query or len(self._store) == 0:
            return []
        limit = top_k or self._top_k
        query_vector = self._embedder.embed([query])[0]
        cosines = self._store.cosine_all(query_vector)
        bm25_scores = self._bm25.scores(tokenize(query))
        max_bm25 = max(bm25_scores) or 1.0

        ranked: list[Hit] = []
        for chunk, cosine, bm25 in zip(
            self._store.records, cosines, bm25_scores
        ):
            hybrid = self._alpha * max(cosine, 0.0) + (1 - self._alpha) * (
                bm25 / max_bm25
            )
            ranked.append(
                Hit(chunk=chunk, score=hybrid, cosine=cosine, bm25=bm25)
            )
        ranked.sort(key=lambda hit: hit.score, reverse=True)
        return ranked[:limit]

    def confident(self, hits: list[Hit]) -> bool:
        """余弦相似度够高，或者关键词命中足够强，都算“知识库里有”。

        纯阈值在短问句上容易误判（“怎么开发票”只和文档共享“发票”两个字），
        所以加一条 BM25 证据作为兜底。
        """
        if not hits:
            return False
        top = hits[0]
        lexical_ok = top.bm25 >= 2.5 and top.cosine >= 0.08
        return top.cosine >= self._min_score or lexical_ok


def build_retriever(
    settings: Settings, force_rebuild: bool = False
) -> Retriever:
    """优先读缓存索引；知识库或 embedding 模型变了就自动重建。"""
    chunks = load_chunks(settings.data_dir / "kb")
    embedder = build_embedder(settings)
    cache: Path = settings.index_path

    if not force_rebuild and cache.exists():
        try:
            store, cached_embedder = VectorStore.load(cache)
            same_embedder = cached_embedder == embedder.name
            same_chunks = [c.chunk_id for c in store.records] == [
                c.chunk_id for c in chunks
            ]
            if same_embedder and same_chunks and len(store):
                return Retriever(
                    store, embedder, settings.top_k, settings.min_score
                )
        except (ValueError, KeyError):
            pass  # 索引损坏就重建，不影响启动

    vectors = embedder.embed([chunk.embed_text() for chunk in chunks])
    store = VectorStore()
    store.add(chunks, vectors)
    store.save(cache, embedder.name)
    return Retriever(store, embedder, settings.top_k, settings.min_score)
