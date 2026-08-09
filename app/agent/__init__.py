# agent/__init__.py — 多 Agent 协同入口

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

# ========== 环境变量校验 ==========
_required_env = {
    "DEEPSEEK_API_KEY": "DeepSeek API 密钥，用于调用 LLM 模型",
    "DB_HOST": "PostgreSQL 数据库主机地址",
    "DB_NAME": "PostgreSQL 数据库名称",
}
_missing = []
for _key, _desc in _required_env.items():
    if not os.getenv(_key):
        _missing.append(f"  {_key}: {_desc}")
if _missing:
    print(f"[Agent] 缺少必要的环境变量，请检查 .env 文件:\n" + "\n".join(_missing))
    sys.exit(1)

from app.core.logging import get_logger; logger = get_logger()

from langchain_openai import ChatOpenAI
from app.rag.retriever import get_hybrid_retriever
from app.agent.tools import create_tools
from app.mcp.client import load_all_mcp_tools
from app.agent.supervisor import build_multi_agent

# ========== 初始化模型 ==========

model = ChatOpenAI(
    model="deepseek-v4-flash",  # DeepSeek V4 Flash 的 API 名称
    temperature=0,
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com/v1",
    # 防止 LLM 在长回答中重复输出相同内容：
    #   - presence_penalty: 惩罚已出现过的 token，降低段落级重复概率
    #   - frequency_penalty: 按出现频次惩罚高频 token，抑制短语/句子重复
    #   - max_tokens: 输出硬上限，防止重复时无限消耗 token
    presence_penalty=0.4,
    frequency_penalty=0.4,
    max_tokens=4096,
    # 关闭思考模式：deepseek-v4 默认开启思考，其工具调用轮次的
    # reasoning_content 必须在后续请求中回传，而 langchain-openai 会丢弃该字段，
    # 导致多 Agent 交接后（历史含其他 worker 的 tool_call）稳定触发 400。
    # 关闭后无 reasoning_content，彻底规避该问题。
    extra_body={"thinking": {"type": "disabled"}},
)

# ========== 混合检索器配置（供 _get_retriever 复用） ==========

RETRIEVER_OPTS = {
    "vectorK": 10,
    "bm25K": 10,
    "finalK": 3,
    "vectorWeight": 0.5,
    "bm25Weight": 0.5,
    # 重排候选从 10 降到 5：CPU 上 CrossEncoder 逐对推理耗时随候选数线性增长，
    # 候选减半可显著缩短检索阶段延迟，RAG 质量损失极小
    "rerankCandidates": 5,
}


async def _get_retriever():
    """返回混合检索器实例。

    命中全局缓存；文档入库后调用 refresh_retriever() 置空缓存，
    下次调用会自动重建（含最新数据的 BM25 索引），无需重启服务。
    """
    start = time.perf_counter()
    retriever = await get_hybrid_retriever(RETRIEVER_OPTS)
    elapsed = (time.perf_counter() - start) * 1000
    logger.debug(f"检索器初始化耗时: {elapsed:.1f}ms")
    return retriever


async def init_agent():
    """异步初始化多 Agent 系统（包含 await 操作）"""
    start = time.perf_counter()
    # ========== 预热混合检索器（尽早暴露配置错误） ==========
    await _get_retriever()

    # ========== 创建工具（本地 + MCP） ==========
    # 1. 加载本地工具（传入 getter，调用时实时解析，数据更新后无需重启）
    local_tools = create_tools(_get_retriever)

    # 2. 加载 MCP 外部工具
    mcp_tools = await load_all_mcp_tools("config/mcp-servers.config.json")

    # 3. 合并（用于调试输出）
    all_tools = list(local_tools) + list(mcp_tools)

    # ========== 按工具域分组（喂给对应 worker） ==========
    # 注意：tools.py 中 search 工具的注册名是 "search_kb"（非 "search_knowledge_base"），
    # 这里同时匹配两个名称以兼容。
    knowledge_tools = [
        t for t in local_tools
        if t.name in {"get_deepseek_info", "search_kb", "search_knowledge_base"}
    ]
    general_tools = [t for t in local_tools if t.name == "get_weather"]
    devops_tools = [t for t in mcp_tools if t.name.startswith("gitee_")]

    # ========== 调试输出 ==========
    logger.info(f"共 {len(all_tools)} 个工具可用（按域分组）:")
    for group, tools in [
        ("knowledge(知识/检索)", knowledge_tools),
        ("devops(Gitee/MCP)", devops_tools),
        ("general(天气/日常)", general_tools),
    ]:
        for t in tools:
            logger.info(f"  [{group}] {t.name}")

    # ========== 构建多 Agent 图（Supervisor 模式） ==========
    agent = await build_multi_agent(model, {
        "knowledge": knowledge_tools,
        "devops": devops_tools,
        "general": general_tools,
    })

    logger.info(f"多 Agent 系统就绪: supervisor + 3 个专业 worker (初始化耗时: {(time.perf_counter() - start) * 1000:.0f}ms)")
    return agent, all_tools
