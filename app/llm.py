from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # 模型返回的 JSON 字符串，执行前再校验


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


class LLM(Protocol):
    """只依赖一个方法，方便用脚本化的 FakeLLM 做单元测试。"""

    def chat(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> LLMResponse: ...


class OpenAICompatLLM:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.2,
    ):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature

    def chat(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        calls = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=call.function.arguments or "{}",
            )
            for call in (message.tool_calls or [])
        ]
        return LLMResponse(
            content=message.content or "", tool_calls=calls, raw=response
        )
