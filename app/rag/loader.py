from __future__ import annotations

import re
from pathlib import Path

from .chunker import Chunk, split_text


def _parse_markdown(text: str, fallback_title: str) -> tuple[str, list[tuple[str, str]]]:
    title = fallback_title
    sections: list[tuple[str, str]] = []
    heading = "概述"
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((heading, body))
        buffer.clear()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("# "):
            title = line[2:].strip() or title
        elif line.startswith("## "):
            flush()
            heading = line[3:].strip() or heading
        else:
            buffer.append(line)
    flush()
    return title, sections


def load_chunks(
    kb_dir: Path, max_chars: int = 220, overlap: int = 40
) -> list[Chunk]:
    """读取 kb 目录下所有 Markdown，按章节切块并编号。"""
    chunks: list[Chunk] = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        doc_id = path.stem
        title, sections = _parse_markdown(
            path.read_text(encoding="utf-8"), fallback_title=doc_id
        )
        index = 0
        for heading, body in sections:
            for piece in split_text(body, max_chars=max_chars, overlap=overlap):
                index += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}#{index}",
                        doc_id=doc_id,
                        title=title,
                        heading=heading,
                        text=piece,
                    )
                )
    return chunks
