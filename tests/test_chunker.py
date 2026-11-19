from __future__ import annotations

from app.rag.chunker import split_text
from app.rag.loader import load_chunks


def test_short_text_is_not_split():
    assert split_text("一句话。", max_chars=50) == ["一句话。"]


def test_long_text_is_split_with_limit():
    text = "这是一句测试文案。" * 60
    pieces = split_text(text, max_chars=100, overlap=20)
    assert len(pieces) > 1
    assert all(len(piece) <= 100 for piece in pieces)
    # 相邻切块有重叠，避免一句话被切断后语义丢失
    assert pieces[0][-10:] in pieces[1]


def test_loader_builds_unique_chunks(settings):
    chunks = load_chunks(settings.data_dir / "kb")
    assert chunks
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert any(chunk.heading == "退货运费由谁承担" for chunk in chunks)
    assert all(chunk.text for chunk in chunks)


def test_embed_text_keeps_title_and_heading(settings):
    chunk = load_chunks(settings.data_dir / "kb")[0]
    assert chunk.title in chunk.embed_text()
    assert chunk.heading in chunk.embed_text()
