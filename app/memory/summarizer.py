# summarizer.py — 文本摘要提取（短期记忆滚动摘要 + 长期记忆持久化）
#
# 复用 app.agent 的全局 ChatOpenAI 模型（DeepSeek），惰性导入避免循环依赖。
# 所有 LLM 调用均包 try/except：失败时降级为原文截断，保证主流程不受影响。

from app.core.logging import get_logger

logger = get_logger()

# 短期记忆滚动摘要：旧摘要 + 新对话片段 → 更新后的摘要
_ROLLING_SUMMARY_PROMPT = (
    "你负责为一段对话维护滚动摘要。结合已有的旧摘要与新增的对话片段，"
    "输出更新后的摘要。要求：\n"
    "1. 保留关键事实（人名、偏好、目标、重要结论、任务进度）\n"
    "2. 丢弃寒暄、过程性描述与无关细节\n"
    "3. 直接用更新后的摘要文本回答，不要任何解释或前后缀\n\n"
    "旧摘要：\n{old}\n\n"
    "新增对话片段：\n{new}\n\n"
    "更新后的摘要："
)

# 长期记忆持久化：LLM 先判断对话是否值得长期记住（重要性门控），
# 只有包含值得跨会话记住的信息时才输出记忆条目，纯寒暄输出 NONE（跳过持久化）
_TURN_SUMMARY_PROMPT = (
    "判断以下一轮对话是否包含值得长期记住的信息，例如用户的事实陈述、"
    "姓名、偏好、承诺、重要结论、任务进度等。\n"
    "要求：\n"
    "1. 若没有任何值得长期记住的信息（例如纯问候、寒暄、闲聊），只回复 NONE\n"
    "2. 否则只输出一条要点式记忆（50 字以内，第三人称），不要任何解释\n\n"
    "对话：\n{text}\n\n"
    "结果："
)


def _get_model():
    """惰性获取全局 LLM（DeepSeek），避免 app.memory 与 app.agent 循环导入。"""
    from app.agent import model
    return model


def _invoke_sync_prompt(model, prompt: str):
    """同步包装 ainvoke：LLM 调用是阻塞网络请求，调用方应放入线程池。"""
    from langchain_core.messages import SystemMessage
    resp = model.invoke([SystemMessage(content=prompt)])
    content = getattr(resp, "content", None)
    return content if isinstance(content, str) else str(content or "")


async def summarize_delta(old_summary: str, delta_text: str) -> str:
    """滚动摘要：旧摘要 + 新片段 → 更新后摘要（失败时用新片段截断兜底）"""
    if not delta_text.strip():
        return (old_summary or "").strip()
    try:
        model = _get_model()
        prompt = _ROLLING_SUMMARY_PROMPT.format(
            old=(old_summary or "（无）").strip(),
            new=delta_text.strip(),
        )
        # LLM 调用为阻塞网络请求，放入线程池避免阻塞事件循环
        import asyncio
        text = await asyncio.to_thread(_invoke_sync_prompt, model, prompt)
        return text.strip()
    except Exception as err:
        logger.warning(f"对话摘要生成失败，使用原文兜底: {err}")
        return delta_text.strip()[:500]


async def summarize_turn(turn_messages: list[dict]) -> str:
    """将一轮对话压缩为长期记忆条目（带重要性门控）。

    Returns:
        记忆文本；无长期记忆价值或生成失败时返回空串（调用方跳过持久化）。
    """
    lines = "\n".join(
        f"{m.get('role')}: {m.get('content')}"
        for m in turn_messages
        if m.get("content")
    )
    if not lines.strip():
        return ""
    try:
        model = _get_model()
        prompt = _TURN_SUMMARY_PROMPT.format(text=lines.strip())
        import asyncio
        text = await asyncio.to_thread(_invoke_sync_prompt, model, prompt)
        text = text.strip()
        # 模型判定无长期记忆价值（NONE / 无 / 没有）
        if not text or text.upper() == "NONE" or text.rstrip("。！! .") in ("无", "没有"):
            return ""
        return text[:200]
    except Exception as err:
        logger.warning(f"长期记忆摘要生成失败，本轮跳过持久化: {err}")
        return ""
