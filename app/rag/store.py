from __future__ import annotations

import json
import math
from pathlib import Path

from .chunker import Chunk


class VectorStore:
    """一个最小的向量库：内存里暴力算余弦相似度，JSON 落盘。

    数据量在几千条以内完全够用；再大就应该换成 FAISS / Qdrant / pgvector。
    """

    def __init__(self) -> None:
        self._records: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def __len__(self) -> int:
        return len(self._records)

    @property
    def dim(self) -> int:
        return len(self._vectors[0]) if self._vectors else 0

    @property
    def records(self) -> list[Chunk]:
        return list(self._records)

    def add(self, records: list[Chunk], vectors: list[list[float]]) -> None:
        if len(records) != len(vectors):
            raise ValueError("records 和 vectors 数量必须一致")
        if vectors and self._vectors and len(vectors[0]) != self.dim:
            raise ValueError("向量维度不一致")
        self._records.extend(records)
        self._vectors.extend(vectors)

    def cosine_all(self, vector: list[float]) -> list[float]:
        if len(vector) != self.dim:
            raise ValueError(f"查询向量维度 {len(vector)} 与索引 {self.dim} 不一致")
        return [self.cosine(vector, stored) for stored in self._vectors]

    def search(self, vector: list[float], top_k: int = 4) -> list[tuple[float, Chunk]]:
        scored = list(zip(self.cosine_all(vector), self._records))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    def save(self, path: Path, embedder_name: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedder": embedder_name,
            "dim": self.dim,
            "items": [
                {"record": record.to_dict(), "vector": vector}
                for record, vector in zip(self._records, self._vectors)
            ],
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> tuple["VectorStore", str]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls()
        store.add(
            [Chunk.from_dict(item["record"]) for item in payload["items"]],
            [item["vector"] for item in payload["items"]],
        )
        return store, payload.get("embedder", "")

    @staticmethod
    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return sum(x * y for x, y in zip(a, b)) / (na * nb)
