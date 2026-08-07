# conversation_history.py — 基于 threadId 的会话历史管理器（内存存储）
#
# 功能：
#   1. 按 threadId 存储对话历史（list[dict]）
#   2. 上下文窗口限制（FIFO，默认最大 20 条消息）
#   3. 提供历史查询、追加、清除接口
#   4. 消息格式为可序列化的 plain dict，便于未来迁移到数据库
#
# MessageRecord 格式：
#   { "role": "user" | "assistant" | "tool",
#     "content": str,
#     "tool_calls": [{"id", "name", "args"}] | None,   # 仅 assistant 有工具调用时
#     "tool_call_id": str | None }                      # 仅 tool 消息时

import json

MAX_HISTORY_LENGTH = 20

# 内存存储：dict[threadId, list[dict]]
_sessions: dict[str, list[dict]] = {}


def _parse_tool_calls(records: list[dict]) -> None:
    """原地修正 tool_calls 中 args 为字符串的情况（兼容旧数据）"""
    for r in records:
        if r["role"] == "assistant" and r.get("tool_calls"):
            for tc in r["tool_calls"]:
                if isinstance(tc.get("args"), str):
                    try:
                        tc["args"] = json.loads(tc["args"])
                    except (json.JSONDecodeError, TypeError):
                        tc["args"] = {}


def get_or_create_history(thread_id: str) -> list[dict]:
    """获取指定会话的历史记录，如不存在则创建空历史"""
    if thread_id not in _sessions:
        _sessions[thread_id] = []
    return _sessions[thread_id]


def to_langchain_messages(records: list[dict]) -> list[dict]:
    """将会话历史转为 LangGraph Agent 可接受的 messages 输入格式"""
    # 预先修复历史中可能的字符串类型 args
    _parse_tool_calls(records)
    messages = []
    for r in records:
        msg = {"role": r["role"], "content": r["content"]}
        if r["role"] == "assistant" and r.get("tool_calls") and len(r["tool_calls"]) > 0:
            msg["tool_calls"] = r["tool_calls"]
        if r["role"] == "tool" and r.get("tool_call_id"):
            msg["tool_call_id"] = r["tool_call_id"]
        messages.append(msg)
    return messages


def append_messages(thread_id: str, messages: list[dict]) -> None:
    """追加多条消息到指定会话的历史末尾，并自动裁剪超出上限的旧消息（FIFO）"""
    history = get_or_create_history(thread_id)
    history.extend(messages)

    # 上下文窗口管理：超过上限时丢弃最早的消息
    if len(history) > MAX_HISTORY_LENGTH:
        excess = len(history) - MAX_HISTORY_LENGTH
        _sessions[thread_id] = history[excess:]


def clear_history(thread_id: str) -> None:
    """清除指定会话的历史"""
    _sessions.pop(thread_id, None)


def clear_all_histories() -> None:
    """清除所有会话历史"""
    _sessions.clear()
