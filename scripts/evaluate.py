"""跑一遍评测集：python scripts/evaluate.py

指标都是“工程上够用”的粗粒度指标，不是学术评测：
- 工具召回率：该调的工具有没有调
- 关键词命中：答案有没有覆盖关键事实
- 转人工判断：该转人工的时候有没有转
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import CustomerServiceAgent  # noqa: E402
from app.config import PROJECT_ROOT, Settings  # noqa: E402
from app.llm import OpenAICompatLLM  # noqa: E402
from app.mock_llm import RuleBasedLLM  # noqa: E402
from app.rag import build_retriever  # noqa: E402
from app.tools import build_default_tools  # noqa: E402


def load_cases(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="客服 Agent 评测")
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "eval" / "cases.jsonl"))
    parser.add_argument("--mock", action="store_true", help="强制使用规则模拟模型")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "eval" / "report.json"))
    args = parser.parse_args()

    settings = Settings.load()
    retriever = build_retriever(settings)
    tools = build_default_tools(settings, retriever)

    use_mock = args.mock or settings.use_mock_llm
    if use_mock:
        llm = RuleBasedLLM()
        mode = "mock"
    else:
        llm = OpenAICompatLLM(
            settings.llm_api_key,
            settings.llm_base_url,
            settings.llm_model,
            settings.llm_temperature,
        )
        mode = settings.llm_model

    agent = CustomerServiceAgent(llm, tools, max_steps=settings.max_steps)
    cases = load_cases(Path(args.cases))
    rows: list[dict] = []

    print(f"mode={mode}  cases={len(cases)}\n")
    print(f"{'用例':<22} {'工具':<6} {'关键词':<7} {'转人工':<7} {'结果'}")
    print("-" * 62)

    for case in cases:
        started = time.perf_counter()
        result = agent.chat(f"eval-{case['id']}", case["question"])
        latency = time.perf_counter() - started
        used = [step.tool for step in result.steps]
        expected = case.get("expected_tools", [])
        hit = len(set(expected) & set(used))
        tool_score = hit / len(expected) if expected else 1.0
        keyword_ok = all(
            keyword.lower() in result.answer.lower()
            for keyword in case.get("must_contain", [])
        )
        handoff_ok = result.handoff == case.get("expect_handoff", False)
        passed = tool_score == 1.0 and keyword_ok and handoff_ok
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "answer": result.answer,
                "tools": used,
                "tool_score": tool_score,
                "keyword_ok": keyword_ok,
                "handoff_ok": handoff_ok,
                "handoff": result.handoff,
                "latency_s": round(latency, 2),
                "passed": passed,
            }
        )
        mark = lambda ok: "✓" if ok else "✗"  # noqa: E731
        print(
            f"{case['id']:<22} {mark(tool_score == 1):<6} "
            f"{mark(keyword_ok):<7} {mark(handoff_ok):<7} {mark(passed)}"
        )
        if not passed:
            print(f"  tools={used} answer={result.answer[:80]}")

    total = len(rows)
    summary = {
        "mode": mode,
        "cases": total,
        "pass_rate": round(sum(row["passed"] for row in rows) / total, 3),
        "avg_tool_recall": round(sum(row["tool_score"] for row in rows) / total, 3),
        "keyword_accuracy": round(
            sum(row["keyword_ok"] for row in rows) / total, 3
        ),
        "handoff_accuracy": round(
            sum(row["handoff_ok"] for row in rows) / total, 3
        ),
        "avg_latency_s": round(sum(row["latency_s"] for row in rows) / total, 2),
    }

    print("-" * 62)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "summary": summary,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n明细已写入 {output}")


if __name__ == "__main__":
    main()
