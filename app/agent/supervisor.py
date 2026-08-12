# multi_agent.py — 多 Agent 协同（Planner 监督者模式：规划-分发-执行-汇总）
#
# 架构：
#   planner（任务规划器：LLM 拆解用户问题为结构化子任务）
#     ├─ 无依赖 → Send 并行分发到 worker_<name>
#     ├─ 有依赖 → executor 串行执行 worker 子图
#     └─ summarizer（汇总器：整合 task_results + 原始问题 → 最终回答）
#   注：每个 worker 节点在自己的子图执行完成后，直接在源头提取最终回答并写入
#       task_results（不再使用独立 collect 节点）。原因：并行 Send 分支在汇聚到
#       join 之前会先合并进共享状态，若由 collect 节点事后读取 state["messages"]，
#       会读到其他 worker 的内容（最近一次复现中 knowledge 结果被 devops 覆盖）。
#
# 与旧版（严格串行 + handoff 交接）的关键差异：
#   1. 每个 worker 只收到自己的私有指令（HumanMessage），看不到原始问题与其他 worker 内容
#   2. 无依赖任务并行执行（LangGraph Send 动态扇出）
#   3. 最终回答由汇总器统一整合，worker 原始文本不再直接透出给用户
#   4. 移除 handoff_to_xxx 交接机制（Planner 承担任务拆分）

import json
import operator
import re
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send

from app.core.logging import get_logger
logger = get_logger()

# ========== 快速通道（Fast Path） ==========
# 对领域明确的请求跳过 planner + summarizer（3 次 LLM 往返 → 1 次），直达对应 worker。
# 命中多个不同领域的规则时无法确定归属，回退到完整多 Agent 流程。

_FAST_PATH_RULES: list[tuple[tuple[str, ...], str]] = [
    # devops（Gitee 操作）
    (("gitee", "仓库", "issue", "pull request", "pr", "组织", "成员", "通知", "star", "关注"), "devops"),
    # knowledge（DeepSeek 公司 / 技术知识库）
    (("deepseek", "模型", "参数", "架构", "性能", "知识库", "知识", "版本", "技术", "v3", "v4"), "knowledge"),
    # general（天气 / 日常）
    (("天气", "温度", "气温", "气候", "湿度", "weather"), "general"),
]

# 纯闲聊：无任何领域词但命中这些关键词 → 直达 general（无需工具）
_CHAT_KEYWORDS = ("你好", "您好", "hello", "hi", "嗨", "在吗", "谢谢", "感谢", "再见", "拜拜", "辛苦了")


def _contains_keyword(query: str, kw: str) -> bool:
    """关键词匹配：中文按子串匹配；ASCII 按字母数字边界匹配。

    - 不使用 \\b（其按 Unicode 词边界处理，CJK 字符也算词字符，
      会导致 "deepseek的开源战略" 这类中英混排匹配失败）
    - 用 ASCII 字母数字边界，避免 hi 命中 this 等误判
    """
    if re.search(r"[\u4e00-\u9fff]", kw):
        return kw in query
    return re.search(rf"(?<![a-zA-Z0-9_]){re.escape(kw)}(?![a-zA-Z0-9_])", query) is not None


def detect_fast_path(query: str) -> str | None:
    """判断是否可走快速通道，返回目标 worker 节点名；无法确定时返回 None。

    规则：
      1. 命中唯一领域 → 直达该 worker
      2. 命中多个领域 → None（交给 planner 拆解）
      3. 未命中领域但属于纯闲聊 → "general"
      4. 其余情况 → None（保守回退，避免误路由导致回答质量下降）
    """
    q = query.lower()
    matched: set[str] = set()
    for keywords, target in _FAST_PATH_RULES:
        if any(_contains_keyword(q, k) for k in keywords):
            matched.add(target)
    if len(matched) > 1:
        return None
    if len(matched) == 1:
        return matched.pop()
    if any(_contains_keyword(q, k) for k in _CHAT_KEYWORDS):
        return "general"
    return None


# ========== 状态定义 ==========


class MultiAgentState(TypedDict):
    """多 Agent 图的共享状态"""

    messages: Annotated[list, add_messages]  # 完整对话历史（planner 上下文 / 汇总参考 / 历史保存）

    # planner 拆解出的任务列表：[{"id","target","instruction","depends_on","summary"}]
    tasks: list[dict]

    # 各 worker 的执行结果：{target: {"ok": bool, "result": str, "summary": str}}
    # 并行分支通过 operator.or_ 合并，天然线程安全
    task_results: Annotated[dict, operator.or_]

    # 本轮原始用户问题（仅汇总器与日志使用）
    original_query: str


# ========== Worker 专业提示词 ==========


def _worker_prompt(node: str) -> str:
    """按节点名返回领域专用提示词。

    与旧版差异：删除 handoff 交接规则，改为"只完成指令内容 + 回答简洁供汇总"。
    """
    prompts = {
        "knowledge": (
            "你是一个知识库问答 Agent，负责回答关于 DeepSeek 公司及相关技术细节的问题。\n\n"
            "规则：\n"
            "1. 先判断问题类型，二选一，禁止同时调用两个工具：\n"
            "   - 问题属于\"公司整体情况\"（背景、公司介绍、开源战略、商业模式、融资、发展历程等宏观问题）→ "
            "只能调用 get_deepseek_info（该工具返回整篇公司手册全文）\n"
            "   - 问题属于\"具体技术细节\"（模型参数、版本特性、性能指标、架构细节、技术原理、与其他模型对比等）→ "
            "只能调用 search_kb\n"
            "   - 两类难以区分时，优先按\"具体技术细节\"处理，只调用 search_kb\n\n"
            "2. 调用次数限制（务必遵守）：\n"
            "   - search_kb 最多调用 1 次，一次查询尽量覆盖问题所有要点，禁止为同一问题发起多个查询\n"
            "   - 调用 get_deepseek_info 后，若返回内容已能回答用户问题，禁止再调用 search_kb\n\n"
            "3. 基于工具返回的内容回答，禁止凭空编造；工具无结果时如实告知用户\n"
            "4. 你的回答会被汇总器整合进最终回复：只输出结论与要点，不要输出"
            "\"让我先…\"\"我现在…\"之类的过程性描述，不要重复表述\n"
            "5. 只完成下方用户指令中要求的内容；指令中没有提到的内容一律不要回答"
        ),
        "devops": (
            "你是一个 DevOps / Gitee 操作 Agent，负责处理所有与 Gitee 平台相关的操作"
            "（仓库、用户、Issue、PR、通知等）。\n\n"
            "规则：\n"
            "1. 任何 Gitee 查询或操作都**必须**调用对应的 gitee_ 开头工具，禁止凭记忆回答\n"
            "2. 基于工具返回的内容回答，只输出结论与要点，不要输出过程性描述，不要重复表述\n"
            "3. 只完成下方用户指令中要求的内容；指令中没有提到的内容一律不要回答"
        ),
        "general": (
            "你是一个通用助手 Agent，负责日常对话、问候和实时天气查询。\n\n"
            "规则：\n"
            "1. 查询实时天气时**必须**调用 get_weather 工具\n"
            "2. 日常闲聊、打招呼直接友好回答，无需调用工具\n"
            "3. 你的回答会被汇总器整合进最终回复：只输出结论与要点，不要输出"
            "\"让我先…\"\"我现在…\"之类的过程性描述，不要重复表述\n"
            "4. 只完成下方用户指令中要求的内容；指令中没有提到的内容一律不要回答"
        ),
    }
    return prompts.get(node, "你是一个专业 Agent，请使用分配给你的工具回答问题。")


# ========== Planner / Summarizer 提示词 ==========

PLANNER_PROMPT = """你是一个任务规划器。根据用户的原始问题与对话历史，将问题拆解为可并行/串行执行的子任务，分发给以下专业 Agent：
{members_txt}

规则：
1. 每个子任务必须能由上述某个 Agent 独立完成；同一 Agent 最多分配 1 个子任务（同领域多个问题合并为一个子任务，指令中列明所有要点）
2. 为每个子任务生成"指令(instruction)"：只包含该任务所需的信息，禁止包含其他任务的内容；若存在代词（它/这个/那个等），在指令中还原成具体对象
3. 若任务 B 必须使用任务 A 的结果，则 B 的 depends_on 填 A 的 id；否则为 null（填 null 的任务将并行执行）
4. 若问题不包含任何专业领域需求（问候/闲聊），生成一个 target=general 的子任务
5. 只输出 JSON，不要任何解释或 Markdown 围栏

输出格式：
{{"tasks": [{{"id": "t1", "target": "general", "instruction": "……", "depends_on": null, "summary": "一句话概括该任务结论"}}]}}"""

SUMMARIZER_PROMPT = """你是一个回复汇总器。用户可能在一个问题中提出多个子需求，以下是各专业 Agent 的独立回答结果。
请以"用户原始问题"为基准，把各结果整合成一段通顺、完整、亲切的最终回复：
1. 每个子需求只回答一次，禁止重复、禁止复述 Agent 的过程性描述
2. 若某结果为空或失败，如实说明该部分未能完成
3. 直接用最终回复内容回答，不要输出 JSON、不要输出"以下是……"之类的元描述"""


# ========== 工具函数 ==========


def _parse_planner_json(text: str) -> dict | None:
    """容错解析 planner 的 JSON 输出：剥离 Markdown 围栏，按括号配平提取首个对象。"""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _latest_user_query(msgs: list) -> str:
    """取消息列表中最后一条 human 消息的文本。"""
    for m in reversed(msgs):
        if getattr(m, "type", None) == "human":
            content = getattr(m, "content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _build_planner_input(msgs: list, latest_query: str) -> list:
    """构造 planner 的 LLM 输入：最近若干条 worker 结论（截断）+ 最新用户问题。

    只喂精简上下文，避免全量历史（含大段工具结果）导致 TTFT 与成本膨胀。
    """
    inputs = []
    ai_texts = [
        m for m in msgs
        if getattr(m, "type", None) == "ai"
        and isinstance(getattr(m, "content", None), str)
        and m.content.strip()
    ]
    for m in ai_texts[-3:]:
        content = m.content.strip()
        if len(content) > 300:
            content = content[:300] + "…"
        inputs.append(AIMessage(content=content))
    inputs.append(HumanMessage(content=latest_query))
    return inputs


def _extract_worker_result(msgs: list) -> str:
    """从消息列表中提取最后一条带文本的 AI 消息（worker 的最终回答）。"""
    for m in reversed(msgs):
        if (
            getattr(m, "type", None) == "ai"
            and isinstance(getattr(m, "content", None), str)
            and m.content.strip()
        ):
            return m.content.strip()
    return ""


# ========== 节点构造 ==========


def _make_planner_node(model, members: dict[str, str]):
    """创建 planner 节点：LLM 将用户问题拆解为结构化子任务（含校验与兜底）。"""

    members_txt = "\n".join(f"- {node}: {desc}" for node, desc in members.items())
    prompt = PLANNER_PROMPT.format(members_txt=members_txt)

    async def planner_node(state: MultiAgentState):
        msgs = state["messages"]
        latest_query = _latest_user_query(msgs)
        if not latest_query:
            latest_query = state.get("original_query", "")

        response = await model.ainvoke(
            [SystemMessage(content=prompt), *_build_planner_input(msgs, latest_query)]
        )
        content = getattr(response, "content", None) or ""
        if not isinstance(content, str):
            content = str(content)

        parsed = _parse_planner_json(content)
        raw_tasks = (parsed or {}).get("tasks", []) if isinstance(parsed, dict) else []

        # 校验：target 必须是活跃成员；同一 target 的任务合并为一个
        merged: dict[str, dict] = {}
        order: list[str] = []
        for t in raw_tasks:
            if not isinstance(t, dict):
                continue
            target = str(t.get("target", "")).strip()
            if target not in members:
                logger.warning(f"planner 输出未知 target: {target}，忽略")
                continue
            instruction = str(t.get("instruction", "")).strip()
            if not instruction:
                continue
            if target in merged:
                merged[target]["instruction"] += f"\n同时：{instruction}"
                continue
            merged[target] = {
                "id": f"task_{target}",
                "target": target,
                "instruction": instruction,
                "depends_on": t.get("depends_on"),
                "summary": str(t.get("summary", "") or ""),
            }
            order.append(target)

        tasks = [merged[target] for target in order]
        if not tasks:
            # 兜底：解析失败 / 无有效任务 → 单任务交给 general
            tasks = [{
                "id": "task_general",
                "target": "general",
                "instruction": latest_query,
                "depends_on": None,
                "summary": "",
            }]
            logger.warning("planner 解析失败或无有效任务，兜底交给 general")

        logger.info(f"Planner 拆解出 {len(tasks)} 个任务: {[t['target'] for t in tasks]}")
        return {"tasks": tasks, "original_query": latest_query}

    return planner_node


def _make_route_dispatch(worker_nodes: set[str]):
    """创建 planner → worker 的条件边：无依赖并行（Send），有依赖串行（executor）。"""

    def route_dispatch(state: MultiAgentState):
        tasks = state.get("tasks", [])
        if not tasks:
            return END
        if any(t.get("depends_on") for t in tasks):
            logger.info(f"任务存在依赖，走串行执行器: {[t['target'] for t in tasks]}")
            return "executor"
        sends = []
        for t in tasks:
            node = f"worker_{t['target']}"
            if node not in worker_nodes:
                logger.warning(f"未知 worker 节点: {node}，跳过")
                continue
            sends.append(Send(node, {"messages": [HumanMessage(content=t["instruction"])]}))
        if not sends:
            return END
        logger.info(f"并行分发 {len(sends)} 个任务: {[t['target'] for t in tasks]}")
        return sends

    return route_dispatch


def _make_executor_node(worker_graphs: dict):
    """创建 executor 节点：按 planner 给出的依赖顺序串行调用 worker 子图。

    说明：该路径下 worker 子图为命令式调用，中间工具事件不流入 SSE
    （仅最终汇总文本输出）；串行场景较少见，正确性优先。
    """

    async def executor_node(state: MultiAgentState):
        tasks = state.get("tasks", [])
        results = {}
        for t in tasks:
            target = t["target"]
            graph = worker_graphs.get(target)
            if graph is None:
                results[target] = {
                    "ok": False,
                    "result": "",
                    "summary": t.get("summary", ""),
                }
                continue
            try:
                result = await graph.ainvoke(
                    {"messages": [HumanMessage(content=t["instruction"])]}
                )
                text = _extract_worker_result(result.get("messages", []))
                results[target] = {
                    "ok": bool(text),
                    "result": text,
                    "summary": t.get("summary", ""),
                }
            except Exception as err:
                logger.error(f"executor 串行执行 {target} 失败: {err}")
                results[target] = {
                    "ok": False,
                    "result": "",
                    "summary": t.get("summary", ""),
                }
        return {"task_results": results}

    return executor_node


async def join_node(state: MultiAgentState):
    """并行/串行汇聚点：不做任何处理，仅保证各分支汇合后进入汇总器。"""
    return {}


def _make_summarizer_node(model):
    """创建 summarizer 节点：以原始问题 + 各 worker 结果整合最终回答。"""

    async def summarizer_node(state: MultiAgentState):
        original_query = state.get("original_query", "")
        results = state.get("task_results", {})

        result_messages = []
        for target, r in results.items():
            if not isinstance(r, dict):
                r = {"ok": True, "result": str(r or "")}
            text = str(r.get("result", "") or "").strip()
            ok = bool(r.get("ok", True))
            if not text or not ok:
                text = f"[{target} 未能完成该部分，请如实告知用户]"
            result_messages.append(AIMessage(content=f"{target} 的结果：{text}"))

        response = await model.ainvoke(
            [
                SystemMessage(content=SUMMARIZER_PROMPT),
                HumanMessage(content=original_query),
                *result_messages,
            ]
        )
        content = getattr(response, "content", None)
        if not isinstance(content, str):
            content = str(content or "")
        # 直接返回模型响应消息本身（保持 message id 稳定，
        # routes.py 按 id 去重，避免 langgraph 对同一条消息重复发射流事件）
        return {"messages": [response]}

    return summarizer_node


# ========== 图装配（主入口） ==========


async def build_multi_agent(model, workers: dict[str, list]):
    """构建 Planner 监督者模式的多 Agent 图。

    Args:
        model: 共享的 ChatOpenAI 模型实例
        workers: {节点名: 工具列表}（已按工具域分组；空列表的节点自动跳过）

    Returns:
        tuple[CompiledStateGraph, dict]: (编译后的多 Agent 图, worker 子图映射)
        worker 子图供 routes 层"快速通道"直接执行（跳过 planner + summarizer）。
    """
    active_workers = {node: tools for node, tools in workers.items() if tools}
    if not active_workers:
        raise ValueError("没有可用的 worker 工具，无法构建多 Agent 图")

    members_desc = {
        node: {
            "knowledge": "知识问答 Agent（DeepSeek 公司介绍与知识库技术细节）",
            "devops": "DevOps Agent（Gitee 平台操作）",
            "general": "通用助手 Agent（日常对话与实时天气）",
        }.get(node, f"{node} Agent")
        for node in active_workers
    }

    # 编译每个 worker 的 ReAct Agent 子图（不再注入 handoff 工具）
    worker_graphs: dict = {}
    for node, tools in active_workers.items():
        worker_graphs[node] = create_react_agent(
            model=model,
            tools=list(tools),
            prompt=_worker_prompt(node),
        )

    graph = StateGraph(MultiAgentState)

    # planner：任务拆解
    graph.add_node("planner", _make_planner_node(model, members_desc))

    # worker 子图（每 worker 一个节点）
    # 子图包一层 try/except：单个 worker 内部工具异常不拖垮整轮。
    # 结果提取直接在该节点源头完成：并行 Send 分支汇聚到 join 前会把所有 worker
    # 的 messages 合并进共享状态，若靠后续 collect 节点读 state["messages"] 会读到
    # 其他 worker 的输出（knowledge 结果曾被 devops 覆盖）。此处各 worker 只提取
    # 自己子图的最终回答，写入 task_results 的独立键，天然隔离。
    worker_nodes: set[str] = set()
    for node in active_workers:
        wname = f"worker_{node}"
        worker_nodes.add(wname)
        subgraph = worker_graphs[node]

        async def _safe_worker(state: MultiAgentState, _sg=subgraph, _name=wname, _target=node):
            try:
                sub_result = await _sg.ainvoke(state)
                text = _extract_worker_result(sub_result.get("messages", []))
                entry = (
                    {"ok": True, "result": text, "summary": ""}
                    if text
                    else {"ok": False, "result": "", "summary": ""}
                )
                return {
                    "messages": sub_result.get("messages", []),
                    "task_results": {_target: entry},
                }
            except Exception as err:
                logger.error(f"worker {_name} 执行失败: {err}")
                return {
                    "messages": [AIMessage(content=f"[{_name} 任务执行失败: {err}]")],
                    "task_results": {_target: {"ok": False, "result": "", "summary": ""}},
                }

        graph.add_node(wname, _safe_worker)
        graph.add_edge(wname, "join")

    # 串行执行器（有依赖场景）
    graph.add_node("executor", _make_executor_node(worker_graphs))
    graph.add_edge("executor", "join")

    # 汇聚点 → 汇总器 → 结束
    graph.add_node("join", join_node)
    graph.add_edge("join", "summarizer")
    graph.add_node("summarizer", _make_summarizer_node(model))
    graph.add_edge("summarizer", END)

    # START → planner → 条件边（并行 Send / 串行 executor / END）
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", _make_route_dispatch(worker_nodes))

    return graph.compile(), worker_graphs
