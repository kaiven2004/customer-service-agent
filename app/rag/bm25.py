from __future__ import annotations

import math
from collections import Counter

from .text import tokenize


class BM25:
    """占位很短的 BM25 实现，用字符 bigram 处理中文，不需要分词模型。"""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._freqs = [Counter(doc) for doc in corpus]
        self._lengths = [len(doc) for doc in corpus]
        self._avgdl = (sum(self._lengths) / len(self._lengths)) if corpus else 1.0
        doc_freq: Counter[str] = Counter()
        for freq in self._freqs:
            doc_freq.update(freq.keys())
        total = len(corpus)
        self._idf = {
            term: math.log(1 + (total - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        results: list[float] = []
        for freq, length in zip(self._freqs, self._lengths):
            score = 0.0
            for term in query_tokens:
                tf = freq.get(term)
                if not tf:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = tf + self._k1 * (
                    1 - self._b + self._b * length / (self._avgdl or 1.0)
                )
                score += idf * tf * (self._k1 + 1) / denom
            results.append(score)
        return results


__all__ = ["BM25", "tokenize"]
