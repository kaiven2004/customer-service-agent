from __future__ import annotations

import shutil
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings
from app.rag import build_retriever
from app.tools import ToolContext, build_default_tools

TEST_TODAY = date(2026, 11, 19)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    (data_dir / "kb").mkdir(parents=True)
    shutil.copytree(PROJECT_ROOT / "data" / "kb", data_dir / "kb", dirs_exist_ok=True)
    shutil.copy(PROJECT_ROOT / "data" / "orders.json", data_dir / "orders.json")
    base = Settings.load()
    return replace(
        base,
        data_dir=data_dir,
        index_path=data_dir / "index.json",
        embedding_mode="hashing",
        embedding_api_key="",
        llm_api_key="",
        top_k=4,
        min_score=0.15,
        max_steps=4,
    )


@pytest.fixture
def retriever(settings: Settings):
    return build_retriever(settings)


@pytest.fixture
def registry(settings: Settings, retriever):
    return build_default_tools(settings, retriever, today=TEST_TODAY)


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(session_id="test-session", user_text="")
