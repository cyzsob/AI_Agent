# multi_agent.py — 多 Agent 协同（Supervisor 监督者模式）
#
# 架构：
#   supervisor（协调 Agent，只负责路由）
#     ├─ TransferToKnowledge → knowledge（RAG 工具）
#     ├─ TransferToDevops    → devops（MCP 工具）
#     ├─ TransferToGeneral   → general（天气 + 日常对话）
#     └─ finish_workflow     → END
#
# 每个 worker 是独立的 ReAct Agent（create_react_agent），
# 只持有本领域的工具子集 + 领域专用提示词，完成后将控制权交回 supervisor。

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import RemoveMessage, add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from app.core.logging import get_logger
logger = get_logger()

# ========== 状态定义 ==========


class MultiAgentState(TypedDict):
    """多 Agent 图的共享状态：完整的消息列表（自动追加合并）"""

    messages: Annotated[list, add_messages]

    # supervisor 已扫描到的消息位置（消息索引）。
    # 用于"跨域交接"标记的消费：worker 发起 handoff_to_xxx 后，
    # supervisor 据此确定性路由一次，并推进游标，避免旧标记被重复扫描造成死循环。
    handoff_checked_upto: int

    # 本轮已路由过（已处理完自己领域）的成员列表。
    # 防止"查天气 + 列仓库"这类跨域任务中，general 与 devops 互相转交造成 ping-pong 死循环：
    # 当某个 handoff 的目标成员已在列表中时，说明它已处理过、其输出已在对话中，直接结束。
    handled_members: list[str]


# ========== 跨域交接（worker → supervisor 的显式转交请求） ==========
#
# 背景：原实现中 worker 遇到超出自己领域的需求时，只能"用文字说明交回协调员"，
# 而 supervisor 是否继续路由完全依赖 LLM 对这段文字的猜测——当 worker 说出
# "已交由协调员处理" 之类的话时，supervisor 常误判为"请求已满足"而 finish_workflow，
# 导致跨域任务（如"查天气 + 列 Gitee 仓库"）只完成一半。该行为不稳定（复现时好时坏）。
#
# 修复：给每个 worker 提供 handoff_to_xxx 工具。worker 处理不了的部分必须
# **通过调用工具显式发起转交请求**（工具返回固定标记 HANDOFF_REQUEST:<node>），
# supervisor 在下一轮检测到标记后**确定性**路由到对应成员，不再依赖 LLM 猜文本。

HANDOFF_PREFIX = "handoff_to_"          # worker 转交工具名前缀
HANDOFF_MARKER_PREFIX = "HANDOFF_REQUEST:"  # 转交工具返回的固定标记


# ========== Supervisor 路由工具 ==========


def _make_transfer_tools(members: dict[str, str]) -> list:
    """为每个成员生成一个 TransferToXxx 路由工具 + finish_workflow。

    Args:
        members: {节点名: 中文描述}

    Returns:
        list: 路由工具列表（仅用于触发条件边路由，函数体不被真正执行）
    """
    tools = []

    for node, desc in members.items():
        tool_name = f"TransferTo{node.capitalize()}"

        @tool(tool_name)
        async def transfer() -> str:
            """转交任务给对应的专业 Agent。"""
            return "已转交"

        tools.append(transfer)

    @tool
    async def finish_workflow() -> str:
        """用户请求已处理完成，结束本次任务。"""
        return "完成"

    tools.append(finish_workflow)
    return tools


def _make_handoff_tools(members: dict[str, str], self_node: str) -> list:
    """为某个 worker 生成"跨域交接"工具（转交到其他成员）。

    worker 没有路由权限，但当用户请求包含自己无法处理的部分时，
    必须调用本工具显式发起转交请求，由 supervisor 在下一轮确定性路由。

    Args:
        members: {节点名: 中文描述}
        self_node: 当前 worker 节点名（不给自己生成转交工具）

    Returns:
        list: handoff_to_xxx 工具列表（函数体仅返回固定标记，不真正执行）
    """
    tools = []
    for node, desc in members.items():
        if node == self_node:
            continue
        tool_name = f"{HANDOFF_PREFIX}{node}"

        @tool(tool_name)
        async def handoff(_target: str = node) -> str:
            """当前用户请求中有一部分超出你的能力范围，调用本工具将这部分任务转交给协调员，由其安排给对应专业 Agent 处理。"""
            return f"{HANDOFF_MARKER_PREFIX}{_target}"

        tools.append(handoff)
    return tools


# ========== Worker 专业提示词 ==========


def _worker_prompt(node: str) -> str:
    """按节点名返回领域专用提示词"""
    # 所有 worker 共享的交接规则（追加到各领域规则之后）：
    # 强调"已转交 ≠ 已完成"——防止 worker 读到其他 Agent 的"已转交协调员"字样后，
    # 误以为任务已处理而再次转交（导致 ping-pong 死循环或任务悬空）。
    handoff_rules = (
        "\n\n交接规则（重要）：\n"
        "1. 对话历史中其他 Agent 或你自己说过的\u201c已转交\u201d、\u201c交由协调员/其他专业 Agent 处理\u201d"
        "等表述，只表示任务被转交到协调员，**绝不代表任务已完成**\n"
        "2. 如果历史中还没有该领域的工具结果，属于你领域范围的任务**必须由你实际调用工具完成**，"
        "绝对不要再次转交（否则会造成死循环）\n"
        "3. 只有确实不属于任何已有 Agent 领域、且历史中也没有结果的新任务，才允许调用 handoff_to_xxx 转交\n"
        "4. 你的最终回答**只包含你自己领域的内容**；其他 Agent 已经回答并展示给用户的部分"
        "（如已输出的天气结果），**绝对不要在你的回答中重复**"
    )
    prompts = {
        "knowledge": (
            "你是一个知识库问答 Agent，负责回答关于 DeepSeek 公司及相关技术细节的问题。\n\n"
            "规则：\n"
            "1. 用户询问 DeepSeek 公司整体情况（背景、技术突破、开源战略、商业模式、融资等）时，"
            "**必须**调用 get_deepseek_info 工具获取信息\n"
            "2. 用户询问具体技术细节（模型参数、版本特性、性能指标、架构细节等）时，"
            "**必须**调用 search_knowledge_base 工具检索知识库\n"
            "3. 基于工具返回的内容回答，禁止凭空编造；工具无结果时如实告知用户\n"
            "4. **回答必须结构清晰、避免冗余**：每个要点只写一次，不要在同一个回答中"
            "重复已经说过的段落或句子。如果发现自己在重复已输出的内容，立即停止\n"
            "5. 只使用分配给你的工具；如果用户请求中还有属于其他专业领域（如实时天气、Gitee 操作）"
            "的部分，先检查对话历史：若该部分**已有回答结果**则视为已完成、不要转交；"
            "若尚未完成，完成自己的部分后**必须调用对应的 handoff_to_xxx 工具**转交剩余部分，"
            "然后再简短告知用户已转交；禁止只用文字说\u201c交回协调员\u201d却不调用转交工具"
        ) + handoff_rules,
        "devops": (
            "你是一个 DevOps / Gitee 操作 Agent，负责处理所有与 Gitee 平台相关的操作"
            "（仓库、用户、Issue、PR、通知等）。\n\n"
            "规则：\n"
            "1. 任何 Gitee 查询或操作都**必须**调用对应的 gitee_ 开头工具，禁止凭记忆回答\n"
            "2. 只要用户请求中涉及 Gitee 内容，且历史中还没有 gitee_ 工具的结果，"
            "该任务就属于你，**必须由你调用 gitee_ 工具实际完成**\n"
            "3. 只使用分配给你的工具；如果用户请求中还有属于其他专业领域（如知识问答、实时天气）"
            "的部分，先检查对话历史：若该部分**已有回答结果**则视为已完成、不要转交；"
            "若尚未完成，完成自己的部分后**必须调用对应的 handoff_to_xxx 工具**转交剩余部分，"
            "然后再简短告知用户已转交；禁止只用文字说\u201c交回协调员\u201d却不调用转交工具"
        ) + handoff_rules,
        "general": (
            "你是一个通用助手 Agent，负责日常对话、问候和实时天气查询。\n\n"
            "规则：\n"
            "1. 查询实时天气时**必须**调用 get_weather 工具\n"
            "2. 日常闲聊、打招呼直接友好回答，无需调用工具\n"
            "3. 只使用分配给你的工具；如果用户请求中还有属于其他专业领域"
            "（如 DeepSeek 技术细节、Gitee 操作）的部分，先检查对话历史："
            "若该部分**已有回答结果**则视为已完成、不要转交；若尚未完成，"
            "完成自己能力范围内的工作后，**必须调用对应的 handoff_to_xxx 工具**转交剩余部分，"
            "然后再简短告知用户已转交；禁止只用文字说\u201c交回协调员\u201d却不调用转交工具"
        ) + handoff_rules,
    }
    return prompts.get(node, "你是一个专业 Agent，请使用分配给你的工具回答问题。")


# ========== Supervisor 节点 ==========


def _detect_handoff(msgs: list, start: int) -> str | None:
    """在 msgs[start:] 范围内确定性检测 worker 发起的"跨域交接"请求。

    优先匹配 handoff 工具返回的固定标记（ToolMessage content），
    其次匹配 AIMessage 中的 handoff_to_xxx 工具调用名。

    Returns:
        目标节点名（如 "devops"）；未检测到返回 None
    """
    for msg in msgs[start:]:
        msg_type = getattr(msg, "type", None)
        if msg_type == "tool":
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str) and content.startswith(HANDOFF_MARKER_PREFIX):
                target = content[len(HANDOFF_MARKER_PREFIX):].strip()
                if target:
                    return target
        for tc in getattr(msg, "tool_calls", None) or []:
            name = tc.get("name", "")
            if name.startswith(HANDOFF_PREFIX):
                target = name[len(HANDOFF_PREFIX):]
                if target:
                    return target
    return None


def _clean_history_for_handoff(msgs: list, start: int) -> tuple[list, int]:
    """构造"跨域交接前清理历史"所需的消息更新列表。

    worker 发起 handoff_to_xxx 后，其本轮产生的消息会残留在共享 state 中。
    若不清除，被转交方（下一个 worker）会读到这些消息，可能产生两种误判：
      1. 读到 HANDOFF_REQUEST 标记 / "已转交协调员" 的叙述，误以为任务已处理完，
         只输出汇总而不调用自己的领域工具；
      2. 读到其他 worker 的工具调用记录或最终汇总，干扰其对"任务属于谁"的判断
         ——既可能再次转交（ping-pong 死循环），也可能把其他 worker 已回答过的
         内容（如天气结果）原样抄进自己的最终回答（重复回答）。

    本函数在 supervisor 检测到标记、准备路由到目标 worker 之前被调用：
      删除 msgs[start:] 范围内（即上一轮新增）的**所有消息**：
        - 所有 ToolMessage
        - 所有携带 tool_calls 的 assistant 消息
        - 纯文本 assistant 消息（源 worker 的最终汇总也一并删除）
    这样被转交方只能看到"用户请求 + 之前的对话"，看不到源 worker 的工具执行
    细节与已输出过的答案，只能回答自己领域内、历史中还没有结果的部分。

    防死循环不依赖这些残留消息，而是由 supervisor 的确定性机制保证：
      - handled_members：已处理过的成员不会被再次路由
      - handoff_checked_upto 游标：转交标记只会被消费一次

    Returns:
        (updates, remaining_count)
        - updates: 可放入 Command.update["messages"] 的 RemoveMessage 列表
        - remaining_count: 清理后剩余消息数（供 handoff_checked_upto 游标使用，
          保证下一轮扫描从清理后的正确位置开始，不遗漏新消息）
    """
    remove_ids: set[str] = set()
    for msg in msgs[start:]:
        remove_ids.add(msg.id)

    updates = [RemoveMessage(id=mid) for mid in remove_ids]
    return updates, len(msgs) - len(remove_ids)


# ========== Supervisor 输入裁剪 ==========
#
# 背景：supervisor 每轮路由都携带全量会话历史（含 worker 的检索结果，可能数千
# token），而一轮问答中 supervisor 会被调用多次（初始路由 + worker 返回后判断
# 收尾），导致每次 LLM 调用的 TTFT 与成本都随历史膨胀。路由决策真正依赖的只是
# "最新用户请求 + 各 worker 已输出的结论"，因此这里构造精简输入喂给 LLM。
#
# 注意：跨域交接检测基于 handoff_checked_upto 游标在完整 msgs 上确定性扫描
# （见 _detect_handoff），不依赖此处输入；裁剪只影响 LLM 路由的上下文，不改变
# 交接/收尾的确定性机制。


def _build_supervisor_input(msgs: list) -> list:
    """构造 supervisor 的 LLM 输入：最新一条 user 消息 + 最近若干条 worker 文本结论（截断）。

    相比直接传入全量 msgs，能显著减少每次路由调用的 input token，降低 TTFT 与成本。
    """
    inputs = []

    # 1) 最新一条用户请求（路由决策的主体）
    human_msgs = [
        m for m in msgs if getattr(m, "type", None) == "human"
    ]
    if human_msgs:
        last = human_msgs[-1]
        content = getattr(last, "content", None)
        if isinstance(content, str):
            inputs.append(HumanMessage(content=content))
        else:
            inputs.append(last)

    # 2) 最近若干条 worker 文本结论（供 supervisor 判断"所有部分是否已完成"），
    #    每条截断到固定长度，避免大段回答拖慢路由调用
    ai_texts = [
        m for m in msgs
        if getattr(m, "type", None) == "ai"
        and isinstance(getattr(m, "content", None), str)
        and getattr(m, "content", "").strip()
    ]
    for m in ai_texts[-3:]:
        content = m.content.strip()
        if len(content) > 300:
            content = content[:300] + "…"
        inputs.append(AIMessage(content=content))

    return inputs


def _make_supervisor_node(model, members: dict[str, str], tool_to_node: dict[str, str]):
    """创建 supervisor 节点函数（闭包注入 model、成员映射与路由映射）。

    节点逻辑：
      1. **确定性跨域交接检测**：worker 通过 handoff_to_xxx 工具发起转交请求后，
         这里先扫描"上一轮新增消息"中的转交标记，命中则直接 Command(goto=目标) 路由，
         完全不依赖 LLM 对 worker 文本的猜测（这是修复"跨域交接丢失"的关键）。
      2. 无交接请求时，才绑定路由工具调用 LLM → 根据 tool_call 通过
         Command(goto=...) 直接路由到目标 worker 或 END。

    关键点：supervisor 的路由 AIMessage 和对应的 ToolMessage **不写入**共享的
    messages 状态。否则 worker 子图会把 "TransferToXxx 已调用 + 任务已转交"
    误读为"转交已完成、任务属于另一个 Agent"，从而只输出移交声明而不调用自己的
    工具（第二轮对话不再调用工具的根因）。
    """

    transfer_tools = _make_transfer_tools(members)
    members_txt = "\n".join(f"- {node}: {desc}" for node, desc in members.items())

    async def supervisor_node(state: MultiAgentState):
        msgs = state["messages"]
        handled = list(state.get("handled_members", []))

        # 1) 确定性跨域交接检测（只扫"上一轮新增"的消息：
        #    从最后一条 user 消息之后、且上次已扫描位置之后开始）
        last_user_idx = max(
            (i for i, m in enumerate(msgs) if getattr(m, "type", None) == "human"),
            default=-1,
        )
        start = max(last_user_idx + 1, state.get("handoff_checked_upto", 0))
        handoff_target = _detect_handoff(msgs, start)

        if handoff_target:
            if handoff_target not in tool_to_node.values():
                logger.warning(f"未知转交目标: {handoff_target}，忽略")
            elif handoff_target in handled:
                # 目标成员已处理过（其输出已在对话中）→ 再转交只会 ping-pong，
                # 确定性结束本轮
                logger.info(f"跨域交接目标 {handoff_target} 已处理过，结束")
                return Command(
                    goto=END,
                    update={"handoff_checked_upto": len(msgs)},
                )
            else:
                logger.info(f"检测到跨域交接请求 -> {handoff_target}")
                # 路由前清理本轮历史：删除源 worker 本轮产生的全部消息
                # （工具执行、工具调用与最终汇总），只保留用户请求与之前的对话，
                # 避免被转交方读到标记/叙述/已输出内容后误判任务已完成、
                # 再次转交或重复回答其他 worker 已答过的部分
                msg_updates, remaining = _clean_history_for_handoff(msgs, start)
                return Command(
                    goto=handoff_target,
                    update={
                        "messages": msg_updates,
                        "handoff_checked_upto": remaining,
                        "handled_members": handled + [handoff_target],
                    },
                )

        # 2) 无交接请求 → LLM 路由
        sys_msg = SystemMessage(
            content=(
                "你是一个协调员（supervisor），负责把用户请求分发给以下专业成员之一，并管理任务的完成：\n\n"
                f"{members_txt}\n\n"
                "规则：\n"
                "1. 分析用户最新请求，判断最合适的成员，调用对应的 TransferTo 工具转交任务\n"
                "2. 只输出工具调用，绝不输出任何文本内容，也不要自行回答问题\n"
                "3. 成员完成任务后控制权会回到你这里；**只有当用户请求的所有部分都已完成时**"
                "才调用 finish_workflow 结束\n"
                "4. 若最后一个成员明确表示其部分已完成、但用户请求仍包含其他专业领域未完成的部分，"
                "必须将剩余部分转交给对应成员继续处理，禁止直接结束\n"
                "5. 若请求明显不属于任何成员，也调用 finish_workflow"
            )
        )
        response = await model.bind_tools(transfer_tools).ainvoke(
            # 只喂精简输入（最新用户请求 + worker 结论），避免每次路由都携带
            # 全量历史（含大段检索结果）导致 TTFT 与成本膨胀
            [sys_msg, *_build_supervisor_input(msgs)]
        )

        # 只根据第一个工具调用计算目标节点；不再把 supervisor 的 AIMessage 和
        # 合成的 ToolMessage 写入共享 messages（详见函数 docstring）。
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls and tool_calls[0]["name"] != "finish_workflow":
            goto = tool_to_node.get(tool_calls[0]["name"], "general")
        else:
            # 兜底：LLM 未给出有效路由（空响应或直接调用 finish_workflow）。
            # 典型场景：历史中已出现过"你好→问候"的完整问答，用户再次说"你好"时，
            # supervisor 误判"请求已处理"而直接结束，导致整个 run 零输出
            # （客户端只会收到服务端兜底文本"请求处理完成"）。
            # 修复：只要最新一条是尚未被任何成员处理过的 human 消息，就强制交给
            # general（通用助手）回答——它能直接应付问候/闲聊，也能通过
            # handoff_to_xxx 把真正的专业任务确定性转交给对应 worker。
            last_msg = msgs[-1] if msgs else None
            is_fresh_human = (
                last_msg is not None
                and getattr(last_msg, "type", None) == "human"
            )
            if (
                is_fresh_human
                and "general" in tool_to_node.values()
                and "general" not in handled
            ):
                logger.warning("LLM 未输出有效路由，兜底转交 general")
                goto = "general"
            else:
                goto = END

        update = {"handoff_checked_upto": len(msgs)}
        if goto != END and goto not in handled:
            update["handled_members"] = handled + [goto]
        return Command(goto=goto, update=update)

    return supervisor_node


# ========== 图装配（主入口） ==========


async def build_multi_agent(model, workers: dict[str, list]):
    """构建 Supervisor 多 Agent 图。

    Args:
        model: 共享的 ChatOpenAI 模型实例
        workers: {节点名: 工具列表}（已按工具域分组；空列表的节点自动跳过）

    Returns:
        CompiledStateGraph: 编译后的多 Agent 图（与单 Agent 的 astream 接口兼容）
    """
    # 过滤出有工具的 worker（无工具的节点不创建）
    active_workers = {node: tools for node, tools in workers.items() if tools}
    if not active_workers:
        raise ValueError("没有可用的 worker 工具，无法构建多 Agent 图")

    # 成员描述（供 supervisor 提示词使用）
    members_desc = {
        node: {
            "knowledge": "知识问答 Agent（DeepSeek 公司介绍与知识库技术细节）",
            "devops": "DevOps Agent（Gitee 平台操作）",
            "general": "通用助手 Agent（日常对话与实时天气）",
        }.get(node, f"{node} Agent")
        for node in active_workers
    }

    # 路由工具名 → worker 节点名 的显式映射
    tool_to_node = {
        f"TransferTo{node.capitalize()}": node for node in active_workers
    }

    # 构建 supervisor 节点（内部通过 Command(goto=...) 完成路由）
    supervisor_node = _make_supervisor_node(model, members_desc, tool_to_node)

    graph = StateGraph(MultiAgentState)
    graph.add_node("supervisor", supervisor_node)

    # 添加每个 worker（独立 ReAct Agent），完成后回到 supervisor
    # 每个 worker 额外注入 handoff_to_xxx 转交工具（跨域交接请求的显式通道）
    for node, tools in active_workers.items():
        worker_tools = list(tools) + _make_handoff_tools(members_desc, node)
        worker = create_react_agent(
            model=model,
            tools=worker_tools,
            prompt=_worker_prompt(node),
        )
        graph.add_node(node, worker)
        graph.add_edge(node, "supervisor")

    # START → supervisor（后续节点跳转由 supervisor 的 Command.goto 决定）
    graph.add_edge(START, "supervisor")

    return graph.compile()
