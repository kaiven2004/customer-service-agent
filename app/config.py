from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """所有可配置项集中在这里，方便测试时替换。"""

    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_temperature: float
    embedding_mode: str
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    data_dir: Path
    index_path: Path
    top_k: int
    min_score: float
    max_steps: int
    mock_llm: str

    @classmethod
    def load(cls) -> "Settings":
        data_dir = PROJECT_ROOT / "data"
        return cls(
            llm_api_key=os.getenv("LLM_API_KEY", "").strip(),
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com").strip(),
            llm_model=os.getenv("LLM_MODEL", "deepseek-chat").strip(),
            llm_temperature=_float("LLM_TEMPERATURE", 0.2),
            embedding_mode=os.getenv("EMBEDDING_MODE", "hashing").strip().lower(),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY", "").strip(),
            embedding_base_url=os.getenv(
                "EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1"
            ).strip(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3").strip(),
            data_dir=data_dir,
            index_path=data_dir / "index.json",
            top_k=_int("TOP_K", 4),
            min_score=_float("MIN_SCORE", 0.15),
            max_steps=_int("MAX_STEPS", 5),
            mock_llm=os.getenv("MOCK_LLM", "auto").strip().lower(),
        )

    @property
    def use_mock_llm(self) -> bool:
        if self.mock_llm == "on":
            return True
        if self.mock_llm == "off":
            return False
        return not self.llm_api_key
