"""手动检索，用来调试召回效果：python scripts/search_kb.py "退货运费谁出" """

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.rag import build_retriever  # noqa: E402


def main() -> None:
    query = " ".join(sys.argv[1:]) or "退货运费谁承担"
    retriever = build_retriever(Settings.load())
    hits = retriever.search(query)
    print(f"query: {query}  confident={retriever.confident(hits)}")
    for i, hit in enumerate(hits, 1):
        print(
            f"{i}. score={hit.score:.3f} cos={hit.cosine:.3f} bm25={hit.bm25:.2f} "
            f"{hit.chunk.citation} {hit.chunk.heading}"
        )
        print(f"   {hit.chunk.text[:80]}...")


if __name__ == "__main__":
    main()
