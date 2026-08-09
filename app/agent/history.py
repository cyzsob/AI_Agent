# conversation_history.py — 基于 threadId 的会话历史管理器（内存存储）
#
# 功能：
#   1. 按 threadId 存储对话历史（list[dict]）
#   2. 上下文窗口限制（基于 Token 计数的滑动窗口）
#   3. 提供历史查询、追加、清除接口
#   4. 消息格式为可序列化的 plain dict，便于未来迁移到数据库
#
# MessageRecord 格式：
#   { "role": "user" | "assistant" | "tool",
#     "content": str,
#     "tool_calls": [{"id", "name", "args"}] | None,   # 仅 assistant 有工具调用时
#     "tool_call_id": str | None }                      # 仅 tool 消息时

import json

import tiktoken

# DeepSeek V4 上下文 128K，留 20% 余量给系统提示词和回答输出
MAX_HISTORY_TOKENS = 100_000
TOKEN_ENCODING = "cl100k_base"

# 内存存储：dict[threadId, list[dict]]
_sessions: dict[str, list[dict]] = {}

# Tokenizer 惰性加载（模块级单例）
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(TOKEN_ENCODING)
    return _encoder


def _count_message_tokens(msg: dict) -> int:
    """计算单条消息的 token 数（含 tool_calls / tool_call_id 等元数据）"""
    encoder = _get_encoder()
    # 使用 JSON 字符串近似计算，包括元数据字段的开销
    text = json.dumps(msg, ensure_ascii=False, default=str)
    return len(encoder.encode(text))


def _count_total_tokens(messages: list[dict]) -> int:
    """计算消息列表总 token 数"""
    return sum(_count_message_tokens(m) for m in messages)


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
    """追加多条消息到指定会话的历史末尾，并基于 Token 计数裁剪旧消息（滑动窗口）。

    裁剪策略：从最早的消息开始丢弃，直到总 token 数不超过 MAX_HISTORY_TOKENS。
    保证至少保留最近一条 user 消息（避免裁剪后零 user 消息导致 Agent 异常）。
    """
    history = get_or_create_history(thread_id)
    history.extend(messages)

    # Token 计数裁剪：从头部丢弃旧消息，直到满足上限
    while len(history) > 1 and _count_total_tokens(history) > MAX_HISTORY_TOKENS:
        # 如果第二条是 user 消息（常见的 user→assistant 对），
        # 先跳过第二条，尝试丢弃更早的消息（保持至少一组完整对话）
        if len(history) >= 4 and history[1]["role"] == "user":
            history.pop(0)
            continue
        # 保底：确保不会把最后一条 user 消息裁掉
        remaining_users = [m for m in history[1:] if m["role"] == "user"]
        if not remaining_users:
            # 只剩第一条是 user → 保留它，丢弃当前的
            break
        history.pop(0)

    _sessions[thread_id] = history


def clear_history(thread_id: str) -> None:
    """清除指定会话的历史"""
    _sessions.pop(thread_id, None)


def clear_all_histories() -> None:
    """清除所有会话历史"""
    _sessions.clear()
