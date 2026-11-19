"""没有 API key 时的兜底“模型”：用几条规则模拟工具调用。

它只是为了让你在没有 key 的情况下也能把整条链路（工具 -> Agent -> 页面）跑通，
正式演示请配置真实大模型；真实效果取决于 LLM 的函数调用能力。
"""

from __future__ import annotations

import json
import re

from .llm import LLMResponse, ToolCall

ORDER_ID = re.compile(r"[A-Za-z]\d{8,}")
REFUND_WORDS = ("退款", "退钱", "我要退")
LOGISTICS_WORDS = ("物流", "到哪", "哪了", "快递", "到货", "签收")
CONFIRM_WORDS = ("确认", "确定", "同意", "可以", "好的", "没问题")
GREETINGS = ("你好", "您好", "hi", "hello", "在吗", "在么")


class RuleBasedLLM:
    def __init__(self) -> None:
        self._counter = 0

    def _call(self, name: str, arguments: dict) -> LLMResponse:
        self._counter += 1
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id=f"mock_{self._counter}",
                    name=name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                )
            ],
        )

    @staticmethod
    def _history_text(messages: list[dict], limit: int = 6) -> str:
        return "\n".join(
            str(message.get("content", ""))
            for message in messages[-limit:]
            if message.get("role") in {"user", "assistant"}
        )

    @staticmethod
    def _last_order_id(messages: list[dict]) -> str | None:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            match = ORDER_ID.search(str(message.get("content", "")))
            if match:
                return match.group(0).upper()
        return None

    def chat(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> LLMResponse:
        last = messages[-1]
        if last.get("role") == "tool":
            return self._answer_from_tool(str(last.get("content", "")))

        user = str(last.get("content", "")).strip()
        lowered = user.lower()
        order_id = self._last_order_id(messages)
        history = self._history_text(messages)

        if "转人工" in user or "人工客服" in user:
            return self._call("handoff_to_human", {"reason": "用户要求转人工"})

        if order_id and any(word in user for word in REFUND_WORDS + CONFIRM_WORDS):
            reason = "用户申请退款"
            if "因为" in user:
                reason = user.split("因为", 1)[1][:50]
            return self._call(
                "request_refund", {"order_id": order_id, "reason": reason}
            )

        if order_id:
            tool = "get_logistics" if any(
                word in user for word in LOGISTICS_WORDS
            ) else "get_order"
            return self._call(tool, {"order_id": order_id})

        if any(word in lowered for word in GREETINGS) and len(user) <= 12:
            return LLMResponse(
                content="您好，我是小岚商城的客服助手。可以帮您查订单物流、退换货、退款、发票和保修问题。请问有什么可以帮您？"
            )

        if "退款" in history and any(word in user for word in CONFIRM_WORDS):
            return LLMResponse(content="好的，如果还需要其他帮助随时告诉我。")

        return self._call("search_kb", {"query": user})

    def _answer_from_tool(self, content: str) -> LLMResponse:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return LLMResponse(content="抱歉，查询出了点问题，请稍后再试。")

        if not data.get("ok", True):
            return LLMResponse(content=f"抱歉，{data.get('error', '查询失败')}。")

        if "results" in data:
            results = data.get("results") or []
            if not results:
                return self._call(
                    "handoff_to_human", {"reason": "知识库没有检索到答案"}
                )
            top = results[0]
            return LLMResponse(
                content=f"根据知识库：{top['text']} {top['citation']}"
            )

        if "order" in data:
            order = data["order"]
            items = "、".join(item["name"] for item in order["items"])
            return LLMResponse(
                content=(
                    f"订单 {order['order_id']} 当前状态：{order['status']}，"
                    f"商品：{items}，实付 {order['paid']} 元。"
                    f"需要退款或查物流可以继续告诉我。"
                )
            )

        if "traces" in data:
            latest = data["traces"][-1]["desc"]
            return LLMResponse(
                content=(
                    f"订单 {data['order_id']} 由{data['carrier']}配送，"
                    f"运单号 {data['tracking_no']}，最新进度：{latest}。"
                )
            )

        if data.get("status") == "need_confirmation":
            return LLMResponse(content=data["message"] + " 请问确认提交吗？")

        if data.get("status") == "created":
            return LLMResponse(
                content=f"{data['message']} 退款单号 {data['ticket_id']}。"
            )

        if data.get("status") == "handoff":
            return LLMResponse(content=data["message"])

        return LLMResponse(content="好的，已经为您处理。")
