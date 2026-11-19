from __future__ import annotations

import pytest

from app.rag.chunker import Chunk
from app.rag.embedder import HashingEmbedder
from app.rag.store import VectorStore


def make_chunk(chunk_id: str, text: str) -> Chunk:
    title, _, heading = text.partition("|")
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        title="文档",
        heading=heading or title,
        text=text,
    )


def test_search_ranks_similar_vector_first():
    embedder = HashingEmbedder(dim=128)
    chunks = [
        make_chunk("doc#1", "退货运费由谁承担"),
        make_chunk("doc#2", "发票申请流程"),
        make_chunk("doc#3", "修改收货地址"),
    ]
    store = VectorStore()
    store.add(chunks, embedder.embed([c.embed_text() for c in chunks]))

    query = embedder.embed(["退货的运费谁出"])[0]
    scores = store.search(query, top_k=3)
    assert scores[0][1].chunk_id == "doc#1"
    assert scores[0][0] > scores[1][0]


def test_save_and_load_roundtrip(tmp_path):
    embedder = HashingEmbedder(dim=64)
    store = VectorStore()
    store.add([make_chunk("doc#1", "测试文本")], embedder.embed(["测试文本"]))
    path = tmp_path / "index.json"
    store.save(path, embedder.name)

    loaded, name = VectorStore.load(path)
    assert name == embedder.name
    assert len(loaded) == 1
    assert loaded.dim == 64
    assert loaded.records[0].text == "测试文本"


def test_dimension_mismatch_raises():
    embedder = HashingEmbedder(dim=32)
    store = VectorStore()
    store.add([make_chunk("doc#1", "文本")], embedder.embed(["文本"]))
    with pytest.raises(ValueError):
        store.search([0.1, 0.2], top_k=1)
