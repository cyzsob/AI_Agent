# stream_handler.py — 流式输出工具调用处理（缓冲、追踪、JSON 完整性检测）

import json
import uuid
from typing import Any


def iter_tool_call_pieces(chunk):
    """从 AIMessage / AIMessageChunk 中提取工具调用增量片段 (id, name, args)。

    兼容两种形态：
      - AIMessageChunk.tool_call_chunks：list[dict]（当前 langgraph 版本为 dict）
      - AIMessage.tool_calls：list[dict]
    """
    pieces = getattr(chunk, "tool_call_chunks", None)
    if pieces is not None:
        for tc in pieces:
            if isinstance(tc, dict):
                yield tc.get("id"), tc.get("name"), tc.get("args")
            else:
                yield (
                    getattr(tc, "id", None),
                    getattr(tc, "name", None),
                    getattr(tc, "args", None),
                )
    else:
        for tc in getattr(chunk, "tool_calls", None) or []:
            if isinstance(tc, dict):
                yield tc.get("id"), tc.get("name"), tc.get("args")
            else:
                yield (
                    getattr(tc, "id", None),
                    getattr(tc, "name", None),
                    getattr(tc, "args", None),
                )


class ToolCallTracker:
    """管理流式工具调用的状态追踪：ID 去重、参数累积、JSON 完整性检测、历史收集。

    用法：
        tracker = ToolCallTracker()
        for chunk in agent_stream:
            for tc_id, tc_name, tc_args in iter_tool_call_pieces(chunk):
                tracker.register(tc_id, tc_name)
                tracker.accumulate_args(tc_id, tc_args)
                for event in tracker.flush_complete():
                    yield event  # SSE event
    """

    def __init__(self):
        self._tool_call_ids: set[str] = set()
        self._args_buffer: dict[str, str] = {}
        self._full_args: dict[str, str] = {}
        self._name_buffer: dict[str, str] = {}
        self._tool_messages: list[dict] = []

    def register(self, tc_id: str | None, tc_name: str | None) -> None:
        """记录工具调用 ID 与名称（去重）"""
        if tc_id:
            if tc_id not in self._tool_call_ids:
                self._tool_call_ids.add(tc_id)
            if tc_name:
                self._name_buffer[tc_id] = tc_name

    def accumulate_args(self, tc_id: str | None, tc_args: Any) -> None:
        """累积工具调用参数（按 tool_call_id 键控）"""
        if not tc_id or not tc_args:
            return
        args_str = (
            json.dumps(tc_args, ensure_ascii=False)
            if isinstance(tc_args, dict)
            else tc_args
        )
        self._args_buffer[tc_id] = self._args_buffer.get(tc_id, "") + args_str
        self._full_args[tc_id] = self._full_args.get(tc_id, "") + args_str

    def add_tool_message(self, tc_id: str, content: str) -> None:
        """收集工具返回消息（用于保存到历史）"""
        self._tool_messages.append({
            "role": "tool",
            "content": content if isinstance(content, str) else "",
            "tool_call_id": tc_id,
        })

    def is_registered(self, tc_id: str) -> bool:
        return tc_id in self._tool_call_ids

    def get_tool_name(self, tc_id: str) -> str:
        return self._name_buffer.get(tc_id, "")

    @property
    def tool_call_ids(self) -> set[str]:
        return self._tool_call_ids

    @property
    def tool_messages(self) -> list[dict]:
        return self._tool_messages

    def build_history_tool_calls(self) -> list[dict]:
        """构造历史记录中的 assistant(tool_calls) 部分，保证 tool_calls 与 tool_messages 一一配对"""
        tool_calls = []
        ordered_ids = list(dict.fromkeys(
            list(self._name_buffer.keys()) + list(self._tool_call_ids)
        ))
        for tc_id in ordered_ids:
            if tc_id not in self._tool_call_ids:
                continue
            raw_args = self._full_args.get(tc_id)
            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args)
                except (json.JSONDecodeError, ValueError):
                    parsed_args = {}
            else:
                parsed_args = raw_args or {}
            tool_calls.append({
                "id": tc_id,
                "name": self._name_buffer.get(tc_id, ""),
                "args": parsed_args,
            })
        return tool_calls

    def flush_complete_args_events(self, sse_event_fn) -> list[str]:
        """返回所有已完整接收 JSON 的 TOOL_CALL_ARGS SSE 事件字符串"""
        events = []
        to_delete = []
        for tc_id, accumulated in self._args_buffer.items():
            try:
                json.loads(accumulated)
                if tc_id:
                    events.append(sse_event_fn({
                        "type": "TOOL_CALL_ARGS",
                        "toolCallId": tc_id,
                        "delta": accumulated,
                    }))
                    to_delete.append(tc_id)
            except (json.JSONDecodeError, ValueError):
                pass
        for tc_id in to_delete:
            self._args_buffer.pop(tc_id, None)
        return events

    def flush_remaining_args_events(self, sse_event_fn) -> list[str]:
        """兜底：发送所有未完成的缓冲 args"""
        events = []
        for tc_id, accumulated in self._args_buffer.items():
            if tc_id and accumulated:
                events.append(sse_event_fn({
                    "type": "TOOL_CALL_ARGS",
                    "toolCallId": tc_id,
                    "delta": accumulated,
                }))
        self._args_buffer.clear()
        return events
