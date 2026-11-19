from __future__ import annotations

import threading

from .llm import ToolCall


class SessionMemory:
    """单个会话的上下文，只保留最近 max_turns 轮，防止 prompt 无限增长。

    注意：工具调用消息必须和它对应的 assistant tool_calls 成对出现，
    所以裁剪时只从 user 消息处切，不能从中间截断。
    """

    def __init__(self, system_prompt: str, max_turns: int = 6):
        self._system = {"role": "system", "content": system_prompt}
        self._messages: list[dict] = []
        self._max_turns = max_turns

    def messages(self) -> list[dict]:
        return [self._system, *self._messages]

    def add_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def add_tool_calls(self, content: str, tool_calls: list[ToolCall]) -> None:
        self._messages.append(
            {
                "role": "assistant",
                "content": content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    }
                    for call in tool_calls
                ],
            }
        )

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> None:
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": content,
            }
        )

    def _trim(self) -> None:
        user_indexes = [
            i for i, message in enumerate(self._messages)
            if message.get("role") == "user"
        ]
        if len(user_indexes) > self._max_turns:
            cut = user_indexes[-self._max_turns]
            del self._messages[:cut]


class SessionStore:
    def __init__(self, system_prompt: str, max_turns: int = 6):
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._sessions: dict[str, SessionMemory] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> SessionMemory:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionMemory(
                    self._system_prompt, self._max_turns
                )
            return self._sessions[session_id]

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)
