from __future__ import annotations

import json
from dataclasses import replace

from app.tools import ToolContext


def run(registry, name: str, arguments: dict, ctx: ToolContext) -> dict:
    return registry.run(name, json.dumps(arguments, ensure_ascii=False), ctx)


def test_get_order_masks_phone(registry, ctx):
    result = run(registry, "get_order", {"order_id": "A20261105001"}, ctx)
    assert result["ok"] is True
    order = result["order"]
    assert order["status"] == "已签收"
    assert order["phone_masked"] == "138****5678"
    assert order["paid"] == 399.0


def test_unknown_order_returns_structured_error(registry, ctx):
    result = run(registry, "get_order", {"order_id": "A00000000"}, ctx)
    assert result["ok"] is False
    assert "没有找到订单" in result["error"]


def test_logistics_has_traces(registry, ctx):
    result = run(registry, "get_logistics", {"order_id": "A20261108002"}, ctx)
    assert result["ok"] is True
    assert result["status"] == "运输中"
    assert result["traces"]


def test_refund_requires_explicit_confirmation(registry, ctx, settings):
    result = run(
        registry,
        "request_refund",
        {"order_id": "A20261105001", "reason": "耳机有杂音"},
        replace(ctx, user_text="我要退款，耳机有杂音"),
    )
    assert result["ok"] is True
    assert result["status"] == "need_confirmation"
    assert not (settings.data_dir / "refunds.jsonl").exists()


def test_refund_succeeds_after_confirmation(registry, settings):
    ctx = ToolContext("session-1", "我确认退款，麻烦了")
    result = run(
        registry,
        "request_refund",
        {"order_id": "A20261105001", "reason": "耳机有杂音"},
        ctx,
    )
    assert result["ok"] is True
    assert result["status"] == "created"
    assert result["amount"] == 399.0
    ticket = json.loads(
        (settings.data_dir / "refunds.jsonl").read_text(encoding="utf-8").strip()
    )
    assert ticket["order_id"] == "A20261105001"


def test_refund_rejects_expired_order(registry, ctx):
    ctx = replace(ctx, user_text="确认退款")
    result = run(
        registry,
        "request_refund",
        {"order_id": "A20261020003", "reason": "不想要了"},
        ctx,
    )
    assert result["ok"] is False
    assert "超过 7 天" in result["error"]


def test_refund_rejects_duplicate(registry, ctx):
    result = run(
        registry,
        "request_refund",
        {"order_id": "A20261110004", "reason": "重复申请"},
        ctx,
    )
    assert result["ok"] is False
    assert "已经退过款" in result["error"]


def test_handoff_writes_ticket(registry, ctx, settings):
    result = run(
        registry,
        "handoff_to_human",
        {"reason": "用户要求人工"},
        replace(ctx, user_text="转人工"),
    )
    assert result["status"] == "handoff"
    assert (settings.data_dir / "handoffs.jsonl").exists()


def test_unknown_tool_is_handled(registry, ctx):
    result = registry.run("drop_database", "{}", ctx)
    assert result["ok"] is False
    assert "不存在" in result["error"]


def test_invalid_arguments_are_handled(registry, ctx):
    result = registry.run("get_order", "{}", ctx)
    assert result["ok"] is False
    assert "参数不合法" in result["error"]


def test_tool_schema_is_openai_compatible(registry):
    schemas = registry.schemas()
    names = {schema["function"]["name"] for schema in schemas}
    assert {"search_kb", "get_order", "request_refund"} <= names
    for schema in schemas:
        assert schema["type"] == "function"
        assert schema["function"]["parameters"]["type"] == "object"
