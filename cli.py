"""终端里直接聊天：python cli.py"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.agent import CustomerServiceAgent  # noqa: E402
from app.config import Settings  # noqa: E402
from app.llm import OpenAICompatLLM  # noqa: E402
from app.mock_llm import RuleBasedLLM  # noqa: E402
from app.rag import build_retriever  # noqa: E402
from app.tools import build_default_tools  # noqa: E402


def main() -> None:
    settings = Settings.load()
    retriever = build_retriever(settings)
    tools = build_default_tools(settings, retriever)
    if settings.use_mock_llm:
        print("[mock 模式] 未配置 LLM_API_KEY，使用规则模拟，仅用于跑通流程。\n")
        llm = RuleBasedLLM()
    else:
        print(f"[{settings.llm_model}] 输入 exit 退出。\n")
        llm = OpenAICompatLLM(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )
    agent = CustomerServiceAgent(llm, tools, max_steps=settings.max_steps)
    session_id = uuid.uuid4().hex[:8]

    while True:
        try:
            text = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            continue
        result = agent.chat(session_id, text)
        print(f"小岚：{result.answer}")
        if result.citations:
            print("引用：" + " ".join(c["citation"] for c in result.citations))
        if result.steps:
            tools_used = " -> ".join(step.tool for step in result.steps)
            print(f"[工具] {tools_used}")
        if result.handoff:
            print("[已转人工]")
        print()


if __name__ == "__main__":
    main()
