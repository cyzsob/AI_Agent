# reference.py — 多轮对话指代消解（入口预处理）
#
# 背景：完整多 Agent 流程中 worker 无对话上下文、planner 上下文被截断（仅 3 条
# AI 文本、截断 300 字），代词的还原无法依赖链路内部完成。因此在入口（routes 层）
# 对含代词的问题做一次"指代还原"，还原后的 query 再走后续所有环节
# （fast path 检测 / planner / 兜底），一次消解、全局受益。
#
# 触发条件：正则检测到代词（它/这个/那个/其 等）才调用 LLM，普通问题零开销；
# 消解失败（LLM 异常 / 历史信息不足）时原样返回，不影响主流程。

import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import get_logger

logger = get_logger()

# 代词检测：排除"其它"这类误触发（"它"前是"其"、"其"后是"它"均不匹配）
_REFERENCE_PRONOUN_RE = re.compile(
    r"((?<!其)它(?:的|们)?|其(?!它)|该|这个|那个|这些|那些|上述|如上)"
)

# 还原时提供给 LLM 的历史消息条数上限（一轮 = 1 条 user + 1 条 assistant）
_REFERENCE_MAX_HISTORY_MSGS = 10
# 单条历史消息参与还原的最大长度（避免大段工具回答/长文撑爆输入）
_REFERENCE_MAX_MSG_CHARS = 500

REFERENCE_RESOLVER_PROMPT = """你是对话指代消解助手。用户当前问题中包含代词（如"它/这个/那个/其"），代词所指的具体对象通常在前面的对话历史中。

请结合对话历史，将当前问题中的代词替换为具体、明确的名称，只输出改写后的完整问题。

要求：
1. 只改写代词，其余文字保持原样
2. 历史中能明确确定指代对象时，必须替换（例如："它的开源策略" → "DeepSeek 的开源策略"）
3. 历史中没有足够信息确定指代对象时，原样输出当前问题，禁止编造
4. 只输出改写后的一句话，不要任何解释、引号、前缀或 Markdown 围栏"""


def _has_reference(query: str) -> bool:
    """是否含可消解的代词（普通问题返回 False，跳过 LLM 调用）。"""
    return bool(query) and _REFERENCE_PRONOUN_RE.search(query) is not None


def _build_resolver_input(history: list[dict], query: str) -> list:
    """从历史记录中提取最近的 user/assistant 文本对，构造 LLM 输入。"""
    lines = []
    for r in history:
        role = r.get("role")
        content = r.get("content", "")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        if len(content) > _REFERENCE_MAX_MSG_CHARS:
            content = content[:_REFERENCE_MAX_MSG_CHARS] + "…"
        label = "用户" if role == "user" else "助手"
        lines.append(f"{label}: {content}")
    history_txt = "\n".join(lines[-_REFERENCE_MAX_HISTORY_MSGS:])
    return [
        SystemMessage(content=REFERENCE_RESOLVER_PROMPT),
        HumanMessage(content=f"对话历史：\n{history_txt}\n\n当前问题：{query}"),
    ]


async def resolve_reference(query: str, history: list[dict], model=None) -> str:
    """若 query 含代词，结合 history 用 LLM 还原为具体问题；否则原样返回。"""
    if not _has_reference(query):
        return query
    if model is None:
        # 懒加载共享模型实例，避免模块导入期依赖
        from app.agent import model as default_model

        model = default_model
    try:
        response = await model.ainvoke(_build_resolver_input(history, query))
        resolved = getattr(response, "content", None) or ""
        if isinstance(resolved, str):
            resolved = resolved.strip()
            if resolved and resolved != query:
                logger.info(f"指代消解: {query!r} → {resolved!r}")
                return resolved
    except Exception as err:
        logger.warning(f"指代消解异常，使用原问题: {err}")
    return query
