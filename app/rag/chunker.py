from __future__ import annotations

from dataclasses import dataclass

SENTENCE_ENDS = "。！？；\n"


@dataclass(frozen=True)
class Chunk:
    """知识库切块。citation 就是回答里引用的 [标题#编号]。"""

    chunk_id: str
    doc_id: str
    title: str
    heading: str
    text: str

    @property
    def citation(self) -> str:
        return f"[{self.title}#{self.chunk_id.rsplit('#', 1)[-1]}]"

    def embed_text(self) -> str:
        """标题和章节名一起参与向量化，避免切块后丢失上下文。"""
        return f"{self.title} / {self.heading}\n{self.text}"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "heading": self.heading,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


def split_text(text: str, max_chars: int = 220, overlap: int = 40) -> list[str]:
    """按字符切分，优先在句号/换行处断开，并保留少量重叠。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind(ch) for ch in SENTENCE_ENDS)
            if cut >= max_chars // 2:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces
