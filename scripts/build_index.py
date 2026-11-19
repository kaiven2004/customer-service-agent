"""重建知识库索引：python scripts/build_index.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.rag import build_retriever  # noqa: E402


def main() -> None:
    settings = Settings.load()
    retriever = build_retriever(settings, force_rebuild=True)
    print(f"索引已重建：{len(retriever.chunks)} 个切块 -> {settings.index_path}")
    print(f"embedding: {retriever._embedder.name} ({retriever._embedder.dim} 维)")


if __name__ == "__main__":
    main()
