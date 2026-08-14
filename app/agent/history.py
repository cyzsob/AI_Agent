# conversation_history.py — 基于 threadId 的会话历史管理器（Redis 短期记忆 + 内存兜底）
#
# 功能：
#   1. 按 threadId 存储对话历史（list[dict]），优先写入 Redis（TTL 闲置过期）；
#      Redis 不可用时降级为进程内内存存储（重启即丢，仅保证服务可用）
#   2. 上下文窗口限制（Token 计数滑动窗口 + 条数上限），
#      被裁剪掉的旧消息返回给调用方，用于触发滚动摘要
#   3. 滚动摘要：被裁剪消息由 LLM 文本摘要提取后存入 memory:short:{tid}:summary
#   4. 提供历史查询、追加、清除接口
#
# MessageRecord 格式：
#   { "role": "user" | "assistant" | "tool",
#     "content": str,
#     "tool_calls": [{"id", "name", "args"}] | None,   # 仅 assistant 有工具调用时
#     "tool_call_id": str | None }                      # 仅 tool 消息时

import json

import tiktoken

from app.core.config import SHORT_MEMORY_MAX_MSGS
from app.core.logging import get_logger
from app.memory import redis_store

logger = get_logger()

# DeepSeek V4 上下文 128K，留 20% 余量给系统提示词和回答输出
MAX_HISTORY_TOKENS = 100_000
TOKEN_ENCODING = "cl100k_base"

# 内存兜底存储：dict[threadId, list[dict]]（Redis 不可用时使用）
_sessions: dict[str, list[dict]] = {}
# 内存兜底滚动摘要：dict[threadId, str]
_fallback_summaries: dict[str, str] = {}

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


# ========== 对外接口（异步，Redis 优先，内存兜底） ==========


async def get_or_create_history(thread_id: str) -> list[dict]:
    """获取指定会话的历史记录；不存在时返回空列表。"""
    records = await redis_store.get_messages(thread_id)
    if records is not None:
        return records
    # Redis 不可用 → 内存兜底
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


async def append_messages(thread_id: str, messages: list[dict]) -> list[dict]:
    """追加多条消息到指定会话的历史末尾，并基于 Token 计数 + 条数上限裁剪。

    裁剪策略：
      - 从最早的消息开始丢弃，直到总 token 数不超过 MAX_HISTORY_TOKENS
        （保证至少保留最近一条 user 消息，避免裁剪后零 user 消息导致 Agent 异常）
      - 最多保留 SHORT_MEMORY_MAX_MSGS 条（短期记忆窗口）

    Returns:
        list[dict]: 被裁剪掉的旧消息（供调用方触发滚动摘要）
    """
    history = [*await get_or_create_history(thread_id), *messages]

    trimmed: list[dict] = []

    # Token 计数裁剪：从头部丢弃旧消息，直到满足上限
    while len(history) > 1 and _count_total_tokens(history) > MAX_HISTORY_TOKENS:
        # 如果第二条是 user 消息（常见的 user→assistant 对），
        # 先跳过第二条，尝试丢弃更早的消息（保持至少一组完整对话）
        if len(history) >= 4 and history[1]["role"] == "user":
            trimmed.append(history.pop(0))
            continue
        # 保底：确保不会把最后一条 user 消息裁掉
        remaining_users = [m for m in history[1:] if m["role"] == "user"]
        if not remaining_users:
            # 只剩第一条是 user → 保留它，丢弃当前的
            break
        trimmed.append(history.pop(0))

    # 条数上限裁剪（短期记忆窗口：只保留最近 N 条）
    overflow = len(history) - SHORT_MEMORY_MAX_MSGS
    if overflow > 0:
        trimmed.extend(history[:overflow])
        history = history[overflow:]

    ok = await redis_store.set_messages(thread_id, history)
    if not ok:
        # Redis 不可用 → 写入内存兜底
        _sessions[thread_id] = history
    return trimmed


async def get_summary(thread_id: str) -> str:
    """读取短期记忆滚动摘要（无则返回空串）。"""
    summary = await redis_store.get_summary(thread_id)
    if summary is None:
        return _fallback_summaries.get(thread_id, "")
    return summary


async def update_summary(thread_id: str, trimmed: list[dict]) -> None:
    """将被裁剪的旧消息做 LLM 文本摘要提取，合并进滚动摘要。

    应在异步后台任务中调用（LLM 调用耗时，不阻塞 SSE 响应）。
    """
    if not trimmed:
        return
    delta_text = "\n".join(
        f"{m.get('role')}: {m.get('content')}"
        for m in trimmed
        if m.get("content")
    )
    if not delta_text.strip():
        return

    from app.memory.summarizer import summarize_delta

    old = await get_summary(thread_id)
    summary = await summarize_delta(old, delta_text)
    if not summary:
        return
    ok = await redis_store.set_summary(thread_id, summary)
    if not ok:
        _fallback_summaries[thread_id] = summary
    else:
        logger.info(f"短期记忆滚动摘要已更新: thread={thread_id}, len={len(summary)}")


async def clear_history(thread_id: str) -> None:
    """清除指定会话的历史（Redis + 内存兜底）"""
    await redis_store.delete_thread(thread_id)
    _sessions.pop(thread_id, None)
    _fallback_summaries.pop(thread_id, None)


async def clear_all_histories() -> None:
    """清除所有会话历史（Redis + 内存兜底）"""
    await redis_store.flush_memory()
    _sessions.clear()
    _fallback_summaries.clear()
