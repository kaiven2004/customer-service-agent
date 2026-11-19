"""一条固定对话，用来快速验证链路：python scripts/demo.py"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import CustomerServiceAgent  # noqa: E402
from app.config import Settings  # noqa: E402
from app.llm import OpenAICompatLLM  # noqa: E402
from app.mock_llm import RuleBasedLLM  # noqa: E402
from app.rag import build_retriever  # noqa: E402
from app.tools import build_default_tools  # noqa: E402

QUESTIONS = [
    "你好",
    "退货运费谁承担？",
    "订单 A20261108002 到哪了？",
    "我要退款，订单 A20261105001，因为耳机有杂音",
    "确认",
    "转人工",
]


def main() -> None:
    settings = Settings.load()
    retriever = build_retriever(settings)
    tools = build_default_tools(settings, retriever)
    if settings.use_mock_llm:
        llm = RuleBasedLLM()
        print("== mock 模式（配置 LLM_API_KEY 后可换成真实模型）==\n")
    else:
        llm = OpenAICompatLLM(
            settings.llm_api_key,
            settings.llm_base_url,
            settings.llm_model,
            settings.llm_temperature,
        )
        print(f"== {settings.llm_model} ==\n")

    agent = CustomerServiceAgent(llm, tools, max_steps=settings.max_steps)
    for question in QUESTIONS:
        result = agent.chat("demo", question)
        print(f"用户：{question}")
        print(f"客服：{result.answer}")
        if result.citations:
            print("引用：" + " ".join(c["citation"] for c in result.citations))
        if result.steps:
            print("工具：" + " -> ".join(step.tool for step in result.steps))
        if result.handoff:
            print("状态：已转人工")
        print("-" * 60)


if __name__ == "__main__":
    main()
