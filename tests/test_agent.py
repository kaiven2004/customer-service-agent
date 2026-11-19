from __future__ import annotations

import json

import pytest

from app.agent import CustomerServiceAgent
from app.llm import LLMResponse, ToolCall
from app.memory import SessionMemory


class FakeLLM:
    """脚本化模型：按顺序返回预设响应，用来测试 Agent 循环本身。"""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        self.calls.append([dict(message) for message in messages])
        if not self._responses:
            raise AssertionError("FakeLLM 没有更多预设响应了")
        return self._responses.pop(0)


def tool_call(name: str, arguments: dict, call_id: str = "call_1") -> LLMResponse:
    return LLMResponse(
        tool_calls=[
            ToolCall(
                id=call_id,
                name=name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            )
        ]
    )


def test_agent_searches_kb_and_collects_citations(registry):
    fake = FakeLLM(
        [
            tool_call("search_kb", {"query": "退货运费谁承担"}),
            LLMResponse(content="质量问题由商家承担运费，非质量问题由买家承担。[退换货政策#4]"),
        ]
    )
    agent = CustomerServiceAgent(fake, registry, max_steps=4)
    result = agent.chat("s1", "退货运费谁承担？")

    assert result.answer.startswith("质量问题")
    assert result.steps[0].tool == "search_kb"
    assert result.steps[0].ok is True
    assert result.citations[0]["citation"] == "[退换货政策#4]"
    # 第二轮请求里应该带上 tool 角色的结果
    assert any(message["role"] == "tool" for message in fake.calls[1])


def test_agent_returns_handoff_flag(registry):
    fake = FakeLLM(
        [
            tool_call("handoff_to_human", {"reason": "用户要求人工"}),
            LLMResponse(content="已为您转接人工客服。"),
        ]
    )
    result = CustomerServiceAgent(fake, registry).chat("s2", "转人工")
    assert result.handoff is True
    assert result.steps[0].tool == "handoff_to_human"


def test_agent_stops_after_max_steps(registry):
    fake = FakeLLM(
        [
            tool_call("get_order", {"order_id": "A20261105001"}, "c1"),
            tool_call("get_order", {"order_id": "A20261105001"}, "c2"),
            LLMResponse(content="订单已签收，如需其他帮助请告诉我。"),
        ]
    )
    agent = CustomerServiceAgent(fake, registry, max_steps=2)
    result = agent.chat("s3", "我的订单呢")

    assert len(result.steps) == 2
    assert result.answer == "订单已签收，如需其他帮助请告诉我。"
    # 最后一次收尾调用不再传工具
    assert fake.calls[-1]


def test_unknown_tool_error_goes_back_to_model(registry):
    fake = FakeLLM(
        [
            tool_call("delete_all_orders", {}),
            LLMResponse(content="抱歉，我换个方式帮您查询。"),
        ]
    )
    result = CustomerServiceAgent(fake, registry).chat("s4", "帮我查订单")
    assert result.steps[0].ok is False
    assert "不存在" in result.steps[0].preview
    assert result.answer


def test_empty_message_is_rejected(registry):
    with pytest.raises(ValueError):
        CustomerServiceAgent(FakeLLM([]), registry).chat("s5", "   ")


def test_memory_trim_keeps_tool_pairs():
    memory = SessionMemory("system", max_turns=2)
    memory.add_user("第一轮")
    memory.add_assistant("回答一")
    memory.add_user("第二轮")
    memory.add_tool_calls("", [ToolCall("c1", "get_order", "{}")])
    memory.add_tool_result("c1", "get_order", "{}")
    memory.add_assistant("回答二")
    memory.add_user("第三轮")

    roles = [message["role"] for message in memory.messages()]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert memory.messages()[1]["content"] == "第二轮"
