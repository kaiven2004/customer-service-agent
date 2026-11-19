from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, Field, ValidationError

from .config import Settings
from .rag import Retriever

logger = logging.getLogger(__name__)

CONFIRM_WORDS = ("确认", "确定", "同意", "没问题", "可以", "好的", "退吧", "申请吧")
CANCEL_WORDS = ("不要", "不用", "取消", "算了", "别退")


class ToolError(Exception):
    """可预期的业务错误，会以结构化结果返回给模型，而不是让程序崩溃。"""


@dataclass
class ToolContext:
    session_id: str
    user_text: str


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel, ToolContext], dict]


class SearchKBArgs(BaseModel):
    query: str = Field(description="检索关键词，保留用户原话里的名词，不要改写成一句话")
    top_k: int = Field(default=4, ge=1, le=8, description="返回条数")


class OrderArgs(BaseModel):
    order_id: str = Field(description="订单号，例如 A20261105001")


class RefundArgs(BaseModel):
    order_id: str = Field(description="要退款的订单号")
    reason: str = Field(description="退款原因，必须来自用户的原话")


class HandoffArgs(BaseModel):
    reason: str = Field(description="转人工的原因，例如知识库无答案、用户要求人工、投诉")


# --------------------------------------------------------------------------
# 模拟订单库
# --------------------------------------------------------------------------

@dataclass
class Order:
    order_id: str
    user: str
    phone: str
    address: str
    status: str
    paid: float
    refund_status: str
    carrier: str
    tracking_no: str
    items: list[dict]
    created_at: str
    delivered_at: str | None
    eta: str | None

    def public(self) -> dict:
        return {
            "order_id": self.order_id,
            "receiver": self.user,
            "phone_masked": mask_phone(self.phone),
            "address": self.address,
            "status": self.status,
            "items": self.items,
            "paid": self.paid,
            "created_at": self.created_at,
            "delivered_at": self.delivered_at,
            "eta": self.eta,
            "carrier": self.carrier,
            "tracking_no": self.tracking_no,
            "refund_status": self.refund_status,
        }


def mask_phone(phone: str) -> str:
    if len(phone) < 7:
        return "***"
    return f"{phone[:3]}****{phone[-4:]}"


class OrderRepository:
    """从 JSON 读订单，签收时间按“今天”倒推，保证演示时数据永远新鲜。"""

    def __init__(self, path: Path, today: date | None = None):
        self.today = today or date.today()
        self._orders: dict[str, Order] = {}
        for raw in json.loads(Path(path).read_text(encoding="utf-8")):
            delivered_days = raw.get("delivered_days_ago")
            delivered = (
                (self.today - timedelta(days=delivered_days)).isoformat()
                if delivered_days is not None
                else None
            )
            created_days = (delivered_days + 3) if delivered_days is not None else 2
            self._orders[raw["order_id"].upper()] = Order(
                order_id=raw["order_id"].upper(),
                user=raw["user"],
                phone=raw["phone"],
                address=raw["address"],
                status=raw["status"],
                paid=float(raw["paid"]),
                refund_status=raw.get("refund_status", "none"),
                carrier=raw.get("carrier", ""),
                tracking_no=raw.get("tracking_no", ""),
                items=raw.get("items", []),
                created_at=(self.today - timedelta(days=created_days)).isoformat(),
                delivered_at=delivered,
                eta=(
                    (self.today + timedelta(days=raw["eta_days"])).isoformat()
                    if raw.get("eta_days") is not None
                    else None
                ),
            )

    def get(self, order_id: str) -> Order | None:
        return self._orders.get(order_id.strip().upper())

    def mark_refunded(self, order_id: str) -> None:
        order = self.get(order_id)
        if order:
            order.refund_status = "refunded"

    def refund_check(self, order: Order) -> tuple[bool, str, int | None]:
        if order.refund_status == "refunded":
            return False, "该订单已经退过款，不能重复申请", None
        if order.delivered_at is None:
            return False, "订单还没有签收，请先拒收，或签收后再申请退货退款", None
        days = (self.today - date.fromisoformat(order.delivered_at)).days
        if days > 7:
            return (
                False,
                f"订单已签收 {days} 天，超过 7 天无理由退货期限，可以改为申请保修维修",
                days,
            )
        return True, "符合 7 天无理由退货条件", days


def _is_confirmed(text: str) -> bool:
    if any(word in text for word in CANCEL_WORDS):
        return False
    return any(word in text for word in CONFIRM_WORDS)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# 工具注册表
# --------------------------------------------------------------------------

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        args_model: type[BaseModel],
        handler: Callable[[BaseModel, ToolContext], dict],
    ) -> None:
        self._tools[name] = Tool(name, description, args_model, handler)

    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.args_model.model_json_schema(),
                },
            }
            for tool in self._tools.values()
        ]

    def run(self, name: str, raw_arguments: str, ctx: ToolContext) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"不存在的工具：{name}"}
        try:
            args = tool.args_model.model_validate_json(raw_arguments or "{}")
            data = tool.handler(args, ctx)
            return {"ok": True, **data}
        except ValidationError as exc:
            return {"ok": False, "error": f"参数不合法：{exc.errors()[0]['msg']}"}
        except ToolError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception:  # noqa: BLE001 - 工具异常不能拖垮整个对话
            logger.exception("tool %s failed", name)
            return {"ok": False, "error": "工具执行失败，请稍后重试或转人工"}

    def names(self) -> list[str]:
        return list(self._tools)


def build_default_tools(
    settings: Settings, retriever: Retriever, today: date | None = None
) -> ToolRegistry:
    repo = OrderRepository(settings.data_dir / "orders.json", today=today)
    refunds_path = settings.data_dir / "refunds.jsonl"
    handoffs_path = settings.data_dir / "handoffs.jsonl"
    registry = ToolRegistry()

    def search_kb(args: SearchKBArgs, _: ToolContext) -> dict:
        hits = retriever.search(args.query, top_k=args.top_k)
        if not retriever.confident(hits):
            return {
                "confident": False,
                "results": [],
                "note": "知识库没有找到可靠答案，不要自行编造，请转人工。",
            }
        return {
            "confident": True,
            "results": [
                {
                    "citation": hit.chunk.citation,
                    "heading": hit.chunk.heading,
                    "text": hit.chunk.text,
                }
                for hit in hits
            ],
        }

    def get_order(args: OrderArgs, _: ToolContext) -> dict:
        order = repo.get(args.order_id)
        if order is None:
            raise ToolError(f"没有找到订单 {args.order_id}，请让用户核对订单号")
        return {"order": order.public()}

    def get_logistics(args: OrderArgs, _: ToolContext) -> dict:
        order = repo.get(args.order_id)
        if order is None:
            raise ToolError(f"没有找到订单 {args.order_id}，请让用户核对订单号")
        if order.status == "运输中":
            traces = [
                {"time": f"{order.created_at} 10:20", "desc": "商家已发货"},
                {"time": f"{order.created_at} 20:05", "desc": f"{order.carrier}已揽收"},
                {"time": f"{repo.today.isoformat()} 08:30", "desc": "运输中，预计次日送达"},
            ]
        else:
            traces = [
                {"time": f"{order.created_at} 10:20", "desc": "商家已发货"},
                {"time": f"{order.delivered_at} 15:40", "desc": "快件已签收"},
            ]
        return {
            "order_id": order.order_id,
            "status": order.status,
            "carrier": order.carrier,
            "tracking_no": order.tracking_no,
            "delivered_at": order.delivered_at,
            "eta": order.eta,
            "traces": traces,
        }

    def request_refund(args: RefundArgs, ctx: ToolContext) -> dict:
        order = repo.get(args.order_id)
        if order is None:
            raise ToolError(f"没有找到订单 {args.order_id}，请让用户核对订单号")
        ok, message, _days = repo.refund_check(order)
        if not ok:
            raise ToolError(message)
        # 程序层面的二次确认，防止模型“擅自”提交退款
        if not _is_confirmed(ctx.user_text):
            return {
                "status": "need_confirmation",
                "message": (
                    f"退款还未提交。请确认：订单 {order.order_id}，"
                    f"实付 {order.paid} 元，原因「{args.reason}」，确认后我再提交。"
                ),
            }
        ticket_id = f"R{repo.today.strftime('%Y%m%d')}{abs(hash(ctx.session_id)) % 10000:04d}"
        _append_jsonl(
            refunds_path,
            {
                "ticket_id": ticket_id,
                "order_id": order.order_id,
                "amount": order.paid,
                "reason": args.reason,
                "session_id": ctx.session_id,
            },
        )
        repo.mark_refunded(order.order_id)
        return {
            "status": "created",
            "ticket_id": ticket_id,
            "order_id": order.order_id,
            "amount": order.paid,
            "message": "退款申请已提交，预计 1-3 个工作日原路退回。",
        }

    def handoff_to_human(args: HandoffArgs, ctx: ToolContext) -> dict:
        ticket_id = f"H{repo.today.strftime('%Y%m%d')}{abs(hash(ctx.session_id)) % 10000:04d}"
        _append_jsonl(
            handoffs_path,
            {
                "ticket_id": ticket_id,
                "reason": args.reason,
                "session_id": ctx.session_id,
                "last_user_message": ctx.user_text,
            },
        )
        return {
            "status": "handoff",
            "ticket_id": ticket_id,
            "message": "已转接人工客服，工作时间 9:00-22:00，请稍候。",
        }

    registry.register(
        "search_kb",
        "检索小岚商城的客服知识库，回答配送、退换货、退款、支付、发票、保修、账号等政策问题。政策类问题必须先查再答。",
        SearchKBArgs,
        search_kb,
    )
    registry.register(
        "get_order",
        "根据订单号查询订单状态、商品、实付金额和收货信息。用户询问自己的订单时必须调用，不要猜测。",
        OrderArgs,
        get_order,
    )
    registry.register(
        "get_logistics",
        "根据订单号查询物流轨迹和预计送达时间。",
        OrderArgs,
        get_logistics,
    )
    registry.register(
        "request_refund",
        "提交退款申请。只能对已签收 7 天内的订单使用，并且必须在用户明确确认订单号和金额后才能调用。",
        RefundArgs,
        request_refund,
    )
    registry.register(
        "handoff_to_human",
        "转人工客服。知识库无答案、工具报错、用户要求人工或投诉时使用。",
        HandoffArgs,
        handoff_to_human,
    )
    return registry
