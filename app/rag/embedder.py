from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Protocol

from ..config import Settings
from .text import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """把 n-gram 哈希到固定维度，带符号，做 L2 归一化。

    好处：零依赖、确定、可离线跑测试；坏处：没有真正的语义泛化能力。
    生产环境把 EMBEDDING_MODE 换成 openai 即可换成 bge-m3 之类的模型。
    """

    name = "hashing-char-bigram-v1"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        counts = Counter(tokenize(text))
        vector = [0.0] * self.dim
        for feature, count in counts.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector


class OpenAICompatEmbedder:
    """调用 OpenAI 兼容的 embedding 接口（如 SiliconFlow 的 bge-m3）。"""

    def __init__(self, api_key: str, base_url: str, model: str, batch_size: int = 32):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._batch_size = batch_size
        self.name = f"openai:{model}"
        self.dim = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._client.embeddings.create(model=self._model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        if vectors:
            self.dim = len(vectors[0])
        return vectors


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_mode == "openai":
        if not settings.embedding_api_key:
            raise ValueError("EMBEDDING_MODE=openai 时必须配置 EMBEDDING_API_KEY")
        return OpenAICompatEmbedder(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
        )
    return HashingEmbedder()
