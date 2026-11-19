from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .llm import LLM
from .memory import SessionStore
from .prompts import FINAL_HINT, build_system_prompt
from .tools import ToolContext, ToolRegistry

FALLBACK_ANSWER = "抱歉，这个问题我暂时没法准确回答，已经为您转接人工客服，请稍候。"


@dataclass
class Step:
    tool: str
    arguments: dict
    ok: bool
    preview: str


@dataclass
class AgentResult:
    answer: str
    session_id: str
    citations: list[dict] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    handoff: bool = False

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "session_id": self.session_id,
            "citations": self.citations,
            "steps": [asdict(step) for step in self.steps],
            "handoff": self.handoff,
        }


def _parse_arguments(raw: str) -> dict:
    try:
        value = json.loads(raw or "{}")
        return value if isinstance(value, dict) else {"raw": raw}
    except json.JSONDecodeError:
        return {"raw": raw}


def _preview(data: dict, limit: int = 160) -> str:
    text = json.dumps(data, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "..."


class CustomerServiceAgent:
    """手写的 ReAct 循环：模型决定调工具 -> 执行 -> 结果回填 -> 再决策。

    没有用 LangChain，是为了把“Agent 到底在循环什么”写清楚；
    逻辑等价于 create_tool_calling_agent + AgentExecutor。
    """

    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        max_steps: int = 5,
        system_prompt: str | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.sessions = SessionStore(
            system_prompt or build_system_prompt(), max_turns=6
        )

    def reset(self, session_id: str) -> None:
        self.sessions.reset(session_id)

    def chat(self, session_id: str, text: str) -> AgentResult:
        text = text.strip()
        if not text:
            raise ValueError("消息不能为空")

        memory = self.sessions.get(session_id)
        memory.add_user(text)
        ctx = ToolContext(session_id=session_id, user_text=text)
        citations: dict[str, dict] = {}
        steps: list[Step] = []
        handoff = False
        answer = ""

        for _ in range(self.max_steps):
            response = self.llm.chat(memory.messages(), tools=self.tools.schemas())
            if not response.tool_calls:
                answer = (response.content or "").strip()
                break

            memory.add_tool_calls(response.content, response.tool_calls)
            for call in response.tool_calls:
                result = self.tools.run(call.name, call.arguments, ctx)
                memory.add_tool_result(
                    call.id,
                    call.name,
                    json.dumps(result, ensure_ascii=False),
                )
                ok = bool(result.get("ok"))
                if ok and call.name == "search_kb":
                    for hit in result.get("results", []):
                        citations[hit["citation"]] = {
                            "citation": hit["citation"],
                            "heading": hit["heading"],
                            "doc_id": hit["citation"].split("#")[0].strip("[]"),
                        }
                if ok and call.name == "handoff_to_human":
                    handoff = True
                steps.append(
                    Step(
                        tool=call.name,
                        arguments=_parse_arguments(call.arguments),
                        ok=ok,
                        preview=_preview(result),
                    )
                )
        else:
            # 工具调用次数用完，强制模型收尾一次（不再提供工具）
            final = self.llm.chat(
                [*memory.messages(), {"role": "user", "content": FINAL_HINT}],
                tools=None,
            )
            answer = (final.content or "").strip()

        if not answer:
            answer = FALLBACK_ANSWER
            handoff = True
        memory.add_assistant(answer)
        # 只展示答案里真正引用到的来源；模型没写引用时退回第一条检索结果
        cited = [item for item in citations.values() if item["citation"] in answer]
        return AgentResult(
            answer=answer,
            session_id=session_id,
            citations=cited or list(citations.values())[:1],
            steps=steps,
            handoff=handoff,
        )


def run_turn(
    llm: LLM, tools: ToolRegistry, session_id: str, text: str, max_steps: int = 5
) -> AgentResult:
    """一次性调用，方便脚本和测试使用。"""
    agent = CustomerServiceAgent(llm, tools, max_steps=max_steps)
    return agent.chat(session_id, text)
