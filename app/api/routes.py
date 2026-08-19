# routes.py — AG-UI 协议兼容的 Agent + RAG 服务器端点（逐 token 流式输出）

import asyncio
import os
import json
import uuid
import traceback
from typing import AsyncGenerator

from dotenv import load_dotenv

load_dotenv()

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, SystemMessage, ToolMessage

from app.core.logging import get_logger
from app.api.sse import (
    AgUiEventType,
    sse_event,
    sse_custom,
    SUPERVISOR_NODE,
    SUMMARIZER_NODE,
    is_internal_tool,
    STREAM_DEBUG,
)
from app.services.message import (
    extract_text_and_images,
    describe_images,
    compose_multimodal_message,
)
from app.api.stream import iter_tool_call_pieces, ToolCallTracker
from app.agent.supervisor import detect_fast_path

logger = get_logger()

router = APIRouter()

# Agent reference (injected by main.py at startup)
_agent_ref: list = [None]
# Worker 子图映射 {节点名: CompiledStateGraph}（快速通道直接执行，注入于启动时）
_worker_graphs_ref: list = [None]

# 后台任务集合：持有引用防止 asyncio 任务被 GC 回收
_bg_tasks: set = set()

# 运行中的多 Agent SSE 任务注册表：run_id -> asyncio.Task（供显式中断接口 cancel）
_run_tasks: dict[str, asyncio.Task] = {}

# 工具结果摘要消息的前缀标记：入短期历史供后续轮次引用，但不进长期记忆沉淀
_TOOL_SUMMARY_PREFIX = "【上轮工具结果摘要】"


def _schedule_background(coro) -> None:
    """调度一个后台协程任务（异步持久化/摘要），并持有引用直到完成。"""
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


def _long_term_messages(turn_messages: list[dict]) -> list[dict]:
    """长期记忆沉淀用消息：排除工具结果摘要，只留用户消息与最终回答。

    工具结果摘要（【上轮工具结果摘要】标记）只进短期历史供后续轮次引用，
    不进入长期记忆，避免 gitee 原始列表等工具数据被向量化进 memory_long。
    """
    return [
        m for m in turn_messages
        if not str(m.get("content", "")).startswith(_TOOL_SUMMARY_PREFIX)
    ]


# ========== 阶段状态事件（CUSTOM） ==========
# 在首字（TEXT_MESSAGE_START 之后、TEXT_MESSAGE_CONTENT 之前）的空窗期
# 通过 AG-UI 标准 CUSTOM 事件（name="status"）按执行阶段上报进度，避免用户看到长时间静默。

_STATUS_TEXT = {
    "planner": "正在拆解任务…",
    "worker": "正在检索/处理中…",
    "summarizer": "正在汇总回答…",
}


def _phase_of(ns, node_name: str | None) -> str | None:
    """根据流事件的节点归属判断当前执行阶段；无法识别时返回 None。

    - ns 非空 → worker 子图内部事件（完整流程下的 worker 阶段）
    - 顶层 planner / summarizer 节点 → 对应阶段
    - 顶层 agent 节点 → 快速通道下 worker 子图的 LLM 节点
    """
    if ns:
        return "worker"
    if node_name == "planner":
        return "planner"
    if node_name == SUMMARIZER_NODE:
        return "summarizer"
    if node_name == "agent":
        return "worker"
    return None


# ========== AG-UI 协议端点 ==========


@router.get("/capabilities")
async def capabilities():
    """声明服务器 AG-UI 能力"""
    return {
        "identity": {
            "name": "DeepSeek RAG Agent",
            "type": "rag",
            "description": "基于 DeepSeek 和 PGVector 的知识库问答助手",
            "version": "1.0.0",
        },
        "transport": {
            "streaming": True,
            "websocket": False,
            "httpBinary": False,
            "pushNotifications": False,
            "resumable": False,
        },
        "tools": {"supported": True},
        "output": {"structuredOutput": False},
        "state": {"persistentState": True},
        "reasoning": {"supported": False},
        "multimodal": {
            "input": {
                "text": True,
                "image": True,
                "audio": False,
                "video": False,
                "file": False,
            },
            "output": {
                "text": True,
                "image": False,
                "audio": False,
                "video": False,
                "file": False,
            },
        },
    }


@router.post("/api/chat")
async def chat(request: Request):
    """AG-UI 聊天端点（SSE 事件流）"""

    # 解析失败必须"正常返回"（而非抛异常）：未处理异常由 Starlette 最外层的
    # ServerErrorMiddleware 直接返回 500，该响应不再经过 CORSMiddleware，导致
    # 浏览器收不到 Access-Control-Allow-Origin，把真实错误误报成"跨域请求失败"。
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError) as err:
        return JSONResponse(
            status_code=400,
            content={
                "error": "请求体不是合法 JSON",
                "hint": "请使用 Content-Type: application/json，且键名必须用双引号",
                "detail": str(err),
            },
        )

    thread_id = body.get("threadId")
    run_id = body.get("runId")
    messages = body.get("messages", [])

    # 参数校验
    if not thread_id or not run_id or not messages or len(messages) == 0:
        async def error_gen():
            yield sse_event({
                "type": AgUiEventType.RUN_ERROR,
                "message": "缺少必需字段: threadId, runId, messages (非空数组)",
                "code": "INVALID_INPUT",
            })
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    # 提取最后一条 user 消息（支持多模态 content 数组）
    user_text = ""
    image_parts = []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_text, image_parts = extract_text_and_images(msg.get("content", ""))
            break

    # 多模态：图片 → 文字描述 → 合并到纯文本
    if image_parts:
        descriptions = await describe_images(image_parts, user_text)
        user_message = compose_multimodal_message(user_text, descriptions)
        logger.info(f"收到 {len(image_parts)} 张图片，已生成描述: {len(user_message)} 字符")
    else:
        user_message = user_text

    return StreamingResponse(
        _make_sse_stream(thread_id, run_id, user_message),
        media_type="text/event-stream",
    )


def _make_sse_stream(
    thread_id: str,
    run_id: str,
    user_message: str,
    predefined_tasks: list | None = None,
):
    """多 Agent 流式执行核心（/api/chat 与 /api/runs/{run_id}/resume 共用）。

    predefined_tasks 非空时为"续跑模式"：跳过快速通道与 planner LLM 拆解，
    直接复用未完成任务执行；其余流程（worker 进度、历史保存、记忆沉淀）一致。
    返回事件流生成器。
    """

    async def event_stream() -> AsyncGenerator[str, None]:
        nonlocal user_message  # 闭包内指代消解会改写 user_message（外层参数作用域）
        request_id = str(uuid.uuid4())[:8]
        log = logger.bind(request_id=request_id)

        # 发送 RUN_STARTED
        yield sse_event({
            "type": AgUiEventType.RUN_STARTED,
            "threadId": thread_id,
            "runId": run_id,
        })

        # 注册运行任务（供显式中断接口 cancel），结束/取消时在 finally 清理
        current_task = asyncio.current_task()
        if current_task is not None:
            _run_tasks[run_id] = current_task

        # 中断时用于持久化"原位续生成"现场（残句 + 原消息 id）；图尚未启动时为空串
        message_id: str = ""
        final_content: str = ""

        try:
            if not user_message.strip():
                raise ValueError("未找到用户消息或消息为空")

            # 多轮对话：加载历史 + 构造完整消息
            from app.agent.history import (
                get_or_create_history,
                to_langchain_messages,
                append_messages,
                get_summary,
                update_summary,
            )

            history = await get_or_create_history(thread_id)

            # 指代消解预处理（入口级）：
            # 完整多 Agent 流程中 worker 无历史、planner 上下文被截断，代词的还原无法
            # 依赖链路内部完成。因此在进入 fast path 检测 / planner 之前，对含代词的问题
            # 结合历史做一次 LLM 还原（如"它的开源带来了什么生态成果"→"DeepSeek 的开源
            # 带来了什么生态成果"），还原后的 query 同时用于路由、记忆检索与后续环节。
            # 不含代词或消解失败时原样返回，普通问题零开销。
            try:
                from app.agent.reference import resolve_reference
                resolved = await resolve_reference(user_message, history)
                if resolved != user_message:
                    log.info(f"指代消解: {user_message!r} → {resolved!r}")
                    user_message = resolved
            except Exception as err:
                log.warning(f"指代消解预处理异常，使用原问题: {err}")

            user_message_record = {"role": "user", "content": user_message}
            full_messages = [*to_langchain_messages(history), user_message_record]

            # 上下文记忆注入：长期记忆（语义+全文混合检索 + distance 阈值过滤）+ 短期记忆滚动摘要。
            # 任一环节失败均静默降级，不影响主对话。
            memory_context: list[str] = []
            if os.getenv("MEMORY_LONG_ENABLED", "true").lower() == "true":
                try:
                    from app.memory.retriever import retrieve_memories
                    memories = await retrieve_memories(user_message)
                    if memories:
                        memory_context.append(
                            "以下是与本次对话相关的长期记忆（来自过往对话）：\n- "
                            + "\n- ".join(m["content"] for m in memories)
                        )
                        log.info(f"注入 {len(memories)} 条长期记忆")
                except Exception as err:
                    log.warning(f"长期记忆检索异常，跳过注入: {err}")
            try:
                short_summary = await get_summary(thread_id)
                if short_summary:
                    memory_context.append(f"本会话早期对话摘要：\n{short_summary}")
            except Exception as err:
                log.warning(f"短期记忆摘要读取异常: {err}")
            if memory_context:
                full_messages.insert(0, SystemMessage(content="\n\n".join(memory_context)))

            # 方案一：在 Agent 执行前立即打开文本消息（"打字中"状态），
            # 让用户第一时间看到回复已开始，而不是在 planner/worker 阶段静默等待
            message_id = str(uuid.uuid4())
            yield sse_event({
                "type": AgUiEventType.TEXT_MESSAGE_START,
                "messageId": message_id,
                "role": "assistant",
            })

            # 运行状态初始化：多 Agent 执行进度写入 Redis（短 TTL），
            # SSE 断开/服务重启后可通过 run_id 查询"上次进行到哪一步"
            try:
                from app.memory.redis_store import update_run_state
                await update_run_state(run_id, **{
                    "thread_id": thread_id,
                    "original_query": user_message,
                    "status": "running",
                })
            except Exception as err:
                log.warning(f"运行状态初始化失败: {err}")

            # 方案二：领域明确的请求走快速通道（直达对应 worker，跳过 planner+summarizer，
            # 3 次 LLM 往返降为 1 次）；无法确定的请求走完整多 Agent 流程
            # 续跑模式跳过快速通道：任务拆解已由 run-state 给出，直接走完整流程
            fast_target = None if predefined_tasks else detect_fast_path(user_message)
            if fast_target:
                log.info(f"快速通道: 直达 worker_{fast_target}")
                worker_graph = _worker_graphs_ref[0].get(fast_target)
                if worker_graph is None:
                    raise ValueError(f"快速通道目标不可用: {fast_target}")
                stream = worker_graph.astream(
                    {"messages": full_messages},
                    stream_mode="messages",
                    subgraphs=True,
                )
            else:
                log.info("完整多 Agent 流程: planner → worker → summarizer")
                stream = _agent_ref[0].astream(
                    {
                        "messages": full_messages,
                        "run_id": run_id,
                        "thread_id": thread_id,
                        "predefined_tasks": predefined_tasks or [],
                    },
                    stream_mode="messages",
                    subgraphs=True,
                )

            tracker = ToolCallTracker()
            final_content = ""
            last_phase: str | None = None

            async for event in stream:
                # subgraphs=True 事件结构兼容处理
                if (
                    isinstance(event, tuple)
                    and len(event) == 2
                    and isinstance(event[0], tuple)
                ):
                    ns, (chunk, metadata) = event
                else:
                    ns, chunk, metadata = (), *event
                node_name = metadata.get("langgraph_node") if metadata else None
                chunk_type = type(chunk).__name__ if chunk else None
                if STREAM_DEBUG:
                    log.debug(
                        f"ns={ns}, node={node_name}, type={chunk_type}, "
                        f"has_tool_call_id={getattr(chunk, 'tool_call_id', None) is not None}"
                    )

                # ---- 阶段状态（CUSTOM 事件，阶段切换时只发一次） ----
                phase = _phase_of(ns, node_name)
                if phase and phase != last_phase:
                    last_phase = phase
                    yield sse_custom("status", {"message": _STATUS_TEXT.get(phase, "正在处理中…")})

                # ---- Supervisor 节点：仅内部路由，不透出 ----
                if node_name == SUPERVISOR_NODE:
                    if STREAM_DEBUG:
                        log.debug(f"Supervisor 内部路由消息，跳过: {chunk_type}")
                    continue

                # ---- 工具执行结果（ToolMessage） ----
                if isinstance(chunk, ToolMessage):
                    log.debug(f"ToolMessage, tool_call_id={getattr(chunk, 'tool_call_id', None)}")

                    tool_msg_name = getattr(chunk, "name", "") or ""
                    if is_internal_tool(tool_msg_name):
                        log.debug(f"内部编排工具结果，跳过: {tool_msg_name}")
                        continue

                    if chunk.tool_call_id:
                        tc_id = chunk.tool_call_id
                        log.debug(f"工具返回, tool_call_id: {tc_id}")

                        # 补发 TOOL_CALL_START
                        if not tracker.is_registered(tc_id):
                            tc_name = getattr(chunk, "name", "") or ""
                            tracker.register(tc_id, tc_name)
                            log.info(f"补发 TOOL_CALL_START, toolCallId: {tc_id}, name: {tc_name}")
                            yield sse_event({
                                "type": AgUiEventType.TOOL_CALL_START,
                                "toolCallId": tc_id,
                                "toolCallName": tc_name,
                            })

                        # 收集工具返回消息
                        content_str = chunk.content if isinstance(chunk.content, str) else ""
                        tracker.add_tool_message(tc_id, content_str)

                        log.debug(f">>> TOOL_CALL_RESULT, toolCallId={tc_id}")
                        yield sse_event({
                            "type": AgUiEventType.TOOL_CALL_RESULT,
                            "messageId": str(uuid.uuid4()),
                            "toolCallId": tc_id,
                            "content": content_str,
                        })

                        log.debug(f">>> TOOL_CALL_END, toolCallId={tc_id}")
                        yield sse_event({
                            "type": AgUiEventType.TOOL_CALL_END,
                            "toolCallId": tc_id,
                        })
                    else:
                        log.warning(f"ToolMessage without tool_call_id! content={str(chunk.content)[:100]}")
                    continue

                # ---- Worker / Agent 节点输出（文本 + 工具调用） ----
                if isinstance(chunk, AIMessage):
                    # 工具调用增量
                    for tc_id, tc_name, tc_args in iter_tool_call_pieces(chunk):
                        if is_internal_tool(tc_name or ""):
                            log.debug(f"跳过内部编排工具事件: {tc_name}")
                            continue

                        if STREAM_DEBUG:
                            log.debug(f"ToolChunk id={tc_id}, args={str(tc_args)[:50] if tc_args else None}")

                        # TOOL_CALL_START
                        if tc_id and not tracker.is_registered(tc_id):
                            tracker.register(tc_id, tc_name)
                            log.info(f"调用工具: {tc_name}, toolCallId: {tc_id}")
                            yield sse_event({
                                "type": AgUiEventType.TOOL_CALL_START,
                                "toolCallId": tc_id,
                                "toolCallName": tc_name or "",
                            })

                        # 累积参数并发送已完成的部分
                        tracker.accumulate_args(tc_id, tc_args)
                        for event_str in tracker.flush_complete_args_events(sse_event):
                            yield event_str

                    # 文本内容透出规则：
                    #   - 快速通道：透出 worker 子图的 agent 节点文本（即最终回答）
                    #   - 完整流程：只透出汇总器节点的最终回答；
                    #     worker 子图文本（ns 非空或节点名非 summarizer）隐藏，
                    #     避免"冗余内容太多 / 重复回答"直接暴露给前端
                    chunk_content = chunk.content if isinstance(chunk.content, str) else ""
                    if fast_target:
                        show_text = (not ns) and node_name == "agent"
                    else:
                        show_text = (not ns) and node_name == SUMMARIZER_NODE
                    if chunk_content and not show_text:
                        if STREAM_DEBUG:
                            log.debug(f"隐藏文本, node={node_name}, ns={ns}, len={len(chunk_content)}")
                    elif chunk_content:
                        final_content += chunk_content

                        if STREAM_DEBUG:
                            log.debug(f">>> TEXT_MESSAGE_CONTENT, delta={chunk_content[:50]}")
                        yield sse_event({
                            "type": AgUiEventType.TEXT_MESSAGE_CONTENT,
                            "messageId": message_id,
                            "delta": chunk_content,
                        })

            # ========== 保存本轮消息到历史 ==========
            # 保存 用户消息 + 最终回答 + 工具结果摘要。
            # 工具结果不以 role="tool" 存历史：OpenAI 兼容 API 校验 tool 消息必须
            # 紧跟 assistant 的 tool_call，历史裁剪可能拆对触发 400。改为带标记的
            # assistant 摘要消息，供后续轮次做指代消解与数据级追问（它/其中/这些）。
            turn_messages = [user_message_record]
            if final_content:
                turn_messages.append({
                    "role": "assistant",
                    "content": final_content,
                })

            tool_summaries = []
            for tm in tracker.tool_messages:
                name = tracker.get_tool_name(tm.get("tool_call_id", ""))
                if is_internal_tool(name):
                    continue
                content = str(tm.get("content", "")).strip()
                if not content:
                    continue
                if len(content) > 800:
                    content = content[:800] + "…"
                tool_summaries.append(f"[{name}] {content}")
            if tool_summaries:
                turn_messages.append({
                    "role": "assistant",
                    "content": f"{_TOOL_SUMMARY_PREFIX}\n" + "\n\n".join(tool_summaries[-5:]),
                })

            # 保存本轮消息到历史；返回被裁剪掉的旧消息（用于触发滚动摘要）
            trimmed = await append_messages(thread_id, turn_messages)

            # 异步后台任务（不阻塞 SSE 响应）：
            #   1) 短期记忆：被裁剪消息 → LLM 滚动摘要
            #   2) 长期记忆：本轮对话 → 摘要提取 → 向量化 → 持久化到 PGVector
            _schedule_background(update_summary(thread_id, trimmed))
            if os.getenv("MEMORY_LONG_ENABLED", "true").lower() == "true" and final_content:
                from app.memory.long_term import persist_round
                _schedule_background(persist_round(thread_id, _long_term_messages(turn_messages)))

            # 兜底：发送残余 args
            for event_str in tracker.flush_remaining_args_events(sse_event):
                log.warning(f"兜底发送残余 args")
                yield event_str

            # 兜底文本（TEXT_MESSAGE_START 已提前发送，此处只补内容）
            if not final_content:
                yield sse_event({
                    "type": AgUiEventType.TEXT_MESSAGE_CONTENT,
                    "messageId": message_id,
                    "delta": "抱歉，我暂时没有生成有效的回复，请换个说法再试一次。",
                })

            # TEXT_MESSAGE_END
            yield sse_event({
                "type": AgUiEventType.TEXT_MESSAGE_END,
                "messageId": message_id,
            })

            # 运行状态收尾：标记完成（记录保留到 TTL 过期，供事后查询）
            try:
                from app.memory.redis_store import update_run_state
                await update_run_state(run_id, **{"status": "done"})
            except Exception as err:
                log.warning(f"运行状态收尾失败: {err}")

            # RUN_FINISHED
            yield sse_event({
                "type": AgUiEventType.RUN_FINISHED,
                "threadId": thread_id,
                "runId": run_id,
                "outcome": {"type": "success"},
            })

        except asyncio.CancelledError:
            # 客户端断开 / 显式中断（cancel 接口）/ 服务停止：标记运行中断后正常取消
            # 同时持久化"原位续生成"现场：残句 + 原消息 id + worker 结果（result:* 由
            # supervisor 写入），供 resume 从汇总处继续采样而非重跑整条链路
            try:
                from app.memory.redis_store import update_run_state
                await update_run_state(run_id, **{
                    "status": "interrupted",
                    "error": "SSE 中断（客户端断开或已取消）",
                    "partial_text": final_content,
                    "message_id": message_id,
                })
            except Exception:
                pass
            raise

        except Exception as err:
            log.error(f"Agent 错误: {err}")
            traceback.print_exc()
            try:
                from app.memory.redis_store import update_run_state
                await update_run_state(run_id, **{
                    "status": "interrupted",
                    "error": str(err)[:500],
                    "partial_text": final_content,
                    "message_id": message_id,
                })
            except Exception:
                pass
            yield sse_event({
                "type": AgUiEventType.RUN_ERROR,
                "message": str(err),
                "code": "AGENT_ERROR",
            })

        finally:
            # 清理运行任务注册（正常结束 / 异常 / 取消均执行）
            if current_task is not None:
                _run_tasks.pop(run_id, None)

    return event_stream()


def _make_summarizer_continue_stream(
    thread_id: str,
    run_id: str,
    original_query: str,
    partial_text: str,
    message_id: str,
    results: dict[str, str],
):
    """汇总阶段原位续生成流（DeepSeek 官网式"中断继续"）。

    适用场景：完整多 Agent 流程的 worker 全部完成、在 summarizer 输出最终回答
    阶段被中断。resume 时不再重跑 planner/worker（它们的结果已由 result:* 持久化），
    而是把 worker 结果 + 已生成的部分文本直接喂给模型，让其从中断处继续采样，
    并用与原 run 相同的 messageId 续写同一条消息气泡。
    """
    async def event_stream() -> AsyncGenerator[str, None]:
        # 注册运行任务（供 cancel 接口中断本次续生成）
        current_task = asyncio.current_task()
        if current_task is not None:
            _run_tasks[run_id] = current_task

        final_content = partial_text
        try:
            yield sse_event({
                "type": AgUiEventType.RUN_STARTED,
                "threadId": thread_id,
                "runId": run_id,
            })

            # 初始化新 run-state：继承 thread/原文/worker 结果，供二次中断与 GET 查询
            try:
                from app.memory.redis_store import update_run_state
                await update_run_state(run_id, **{
                    "thread_id": thread_id,
                    "original_query": original_query,
                    "status": "running",
                    "message_id": message_id,
                    **{f"result:{k}": v for k, v in results.items()},
                })
            except Exception as err:
                log.warning(f"续生成运行状态初始化失败: {err}")

            yield sse_custom("status", {"message": _STATUS_TEXT["summarizer"]})

            from app.agent import model
            from app.agent.supervisor import SUMMARIZER_PROMPT
            from langchain_core.messages import (
                AIMessage as _AIMessage,
                HumanMessage as _HumanMessage,
                SystemMessage as _SystemMessage,
            )

            result_messages = []
            for target, text in results.items():
                text = str(text or "").strip()
                if not text:
                    text = f"[{target} 未能完成该部分，请如实告知用户]"
                result_messages.append(_AIMessage(content=f"{target} 的结果：{text}"))

            msgs: list = [
                _SystemMessage(content=SUMMARIZER_PROMPT + (
                    "\n\n【续写要求】对话中最后一条 assistant 消息是本回答已生成但被用户中断的"
                    "部分草稿。请从该草稿的断点处继续书写，直接续写后续内容，"
                    "不要重复已生成的部分，最终给出完整、连贯的回答。"
                )),
                _HumanMessage(content=original_query),
                *result_messages,
            ]
            if partial_text:
                # 尾部 assistant 残句 → OpenAI 兼容 API 从该处继续采样（配合上面的
                # 续写要求，避免模型从头重新生成导致与已有内容重复）
                msgs.append(_AIMessage(content=partial_text))

            async for chunk in model.astream(msgs):
                delta = chunk.content if isinstance(chunk.content, str) else ""
                if not delta:
                    continue
                final_content += delta
                yield sse_event({
                    "type": AgUiEventType.TEXT_MESSAGE_CONTENT,
                    "messageId": message_id,
                    "delta": delta,
                })

            yield sse_event({
                "type": AgUiEventType.TEXT_MESSAGE_END,
                "messageId": message_id,
            })

            # 保存本轮（用户消息 + 完整最终回答）到历史；被中断的那轮此前未落库
            from app.agent.history import (
                get_or_create_history,
                append_messages,
                update_summary,
            )
            await get_or_create_history(thread_id)
            turn_messages = [
                {"role": "user", "content": original_query},
                {"role": "assistant", "content": final_content},
            ]
            trimmed = await append_messages(thread_id, turn_messages)
            _schedule_background(update_summary(thread_id, trimmed))
            if os.getenv("MEMORY_LONG_ENABLED", "true").lower() == "true" and final_content:
                from app.memory.long_term import persist_round
                _schedule_background(persist_round(thread_id, _long_term_messages(turn_messages)))

            # 收尾：标记完成 + RUN_FINISHED
            try:
                from app.memory.redis_store import update_run_state
                await update_run_state(run_id, **{"status": "done"})
            except Exception as err:
                log.warning(f"续生成运行状态收尾失败: {err}")
            yield sse_event({
                "type": AgUiEventType.RUN_FINISHED,
                "threadId": thread_id,
                "runId": run_id,
                "outcome": {"type": "success"},
            })

        except asyncio.CancelledError:
            try:
                from app.memory.redis_store import update_run_state
                await update_run_state(run_id, **{
                    "status": "interrupted",
                    "error": "SSE 中断（客户端断开或已取消）",
                    "partial_text": final_content,
                    "message_id": message_id,
                })
            except Exception:
                pass
            raise

        except Exception as err:
            log.error(f"续生成错误: {err}")
            traceback.print_exc()
            try:
                from app.memory.redis_store import update_run_state
                await update_run_state(run_id, **{
                    "status": "interrupted",
                    "error": str(err)[:500],
                    "partial_text": final_content,
                    "message_id": message_id,
                })
            except Exception:
                pass
            yield sse_event({
                "type": AgUiEventType.RUN_ERROR,
                "message": str(err),
                "code": "AGENT_ERROR",
            })

        finally:
            _run_tasks.pop(run_id, None)

    return event_stream()


# ========== 历史管理端点 ==========


@router.post("/api/chat/clear")
async def chat_clear(request: Request):
    """清除指定会话的历史"""
    from app.agent.history import clear_history

    body = await request.json()
    thread_id = body.get("threadId")
    if not thread_id:
        return {"success": False, "message": "缺少 threadId"}
    await clear_history(thread_id)
    logger.info(f"已清除会话历史: {thread_id}")
    return {"success": True, "threadId": thread_id}


# ========== 会话管理端点 ==========


@router.post("/api/sessions")
async def create_session_endpoint(request: Request):
    """创建会话，返回服务端生成的 threadId。"""
    from app.agent.history import create_session

    title = None
    try:
        body = await request.json()
        title = body.get("title")
    except (json.JSONDecodeError, ValueError):
        pass  # 无请求体 / 非法 JSON 时按默认标题创建
    session = await create_session(title)
    logger.info(f"已创建会话: {session['threadId']}")
    return session


@router.get("/api/sessions/{thread_id}")
async def get_session_endpoint(thread_id: str):
    """获取会话数据（元数据 + 历史消息 + 滚动摘要）。"""
    from app.agent.history import get_session

    session = await get_session(thread_id)
    if session is None:
        return JSONResponse({"error": "session not found"}, status_code=404)
    return session


@router.delete("/api/sessions/{thread_id}")
async def delete_session_endpoint(thread_id: str):
    """删除会话（元数据 + 历史 + 滚动摘要）。"""
    from app.agent.history import delete_session

    await delete_session(thread_id)
    logger.info(f"已删除会话: {thread_id}")
    return {"success": True, "threadId": thread_id}


# ========== 运行状态查询端点 ==========


@router.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """查询一次多 Agent 运行的状态快照（任务拆解 + 各 worker 进度）。"""
    from app.memory.redis_store import get_run_state

    state = await get_run_state(run_id)
    if not state:
        return JSONResponse({"error": "run not found（可能已过期）"}, status_code=404)
    return state


@router.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    """显式中断一次正在进行的多 Agent 运行。

    通过 run_id 找到对应的 SSE 任务并 cancel()，SSE 端收到 CancelledError 后
    会把 run-state 标记为 interrupted。注意：本注册表为进程内映射，
    多 uvicorn worker 部署时仅能中断当前进程内的运行。
    """
    from app.memory.redis_store import get_run_state, update_run_state

    task = _run_tasks.get(run_id)
    if task is not None and not task.done():
        task.cancel()
        logger.info(f"已发送中断请求: {run_id}")
        return {"success": True, "reason": "cancelled", "runId": run_id, "message": "已发送中断请求"}

    # 任务不在注册表（已完成 / 客户端断开后已清理 / 非本进程）：查 run-state 给出准确结果
    state = await get_run_state(run_id)
    if not state:
        return {
            "success": False,
            "reason": "not_found",
            "message": "该 run 不存在（可能已过期）",
            "runId": run_id,
        }
    if state.get("status") == "done":
        return {
            "success": False,
            "reason": "done",
            "message": "该 run 已完成",
            "runId": run_id,
        }

    # 僵尸 running / interrupted：幂等标记中断，避免误导性"该 run 不存在或已结束"
    await update_run_state(run_id, **{
        "status": "interrupted",
        "error": "取消请求：运行任务已不在注册表（可能已结束或非本进程）",
    })
    return {
        "success": True,
        "reason": "cleaned",
        "message": "运行任务已结束，已标记中断",
        "runId": run_id,
    }


@router.post("/api/runs/{run_id}/resume")
async def resume_run(run_id: str):
    """续跑：基于 run-state 中未完成的任务（pending/running/failed）重新执行。

    读取 run-state → 过滤出未完成任务 → 生成新 run_id 复用同一 SSE 流，
    跳过 planner LLM 拆解（省一次调用）。适用于 SSE 中断/异常后的断点续跑。
    """
    from app.memory.redis_store import get_run_state

    state = await get_run_state(run_id)
    if not state:
        return JSONResponse({"error": "run not found（可能已过期）"}, status_code=404)
    if state.get("status") == "done":
        return JSONResponse({"error": "该 run 已完成，无需续跑"}, status_code=400)

    tasks = state.get("tasks") or []
    unfinished = [t for t in tasks if t.get("status", "pending") != "done"]

    thread_id = state.get("thread_id", "")
    original_query = state.get("original_query", "")
    if not thread_id or not original_query:
        return JSONResponse(
            {"error": "run-state 缺少 thread_id / original_query（中断过早，无法续跑）"},
            status_code=400,
        )

    new_run_id = uuid.uuid4().hex
    if unfinished:
        logger.info(f"续跑: {run_id} -> {new_run_id}，未完成任务: {[t['target'] for t in unfinished]}")
        predefined_tasks = unfinished
    else:
        # 任务全部完成但 run 未收尾：判断是否为"汇总阶段中断"——
        # 是则原位续生成（复用 worker 结果 + 残句，模型从中断处继续采样）；
        # 否则退化为完整重跑（如快速通道中断，无 result:* 可复用）。
        results = {
            k.removeprefix("result:"): v
            for k, v in state.items()
            if k.startswith("result:")
        }
        partial_text = state.get("partial_text", "")
        message_id = state.get("message_id", "")
        if results and message_id:
            logger.info(f"续跑(汇总原位续生成): {run_id} -> {new_run_id}")
            return StreamingResponse(
                _make_summarizer_continue_stream(
                    thread_id, new_run_id, original_query,
                    partial_text, message_id, results,
                ),
                media_type="text/event-stream",
            )
        logger.info(f"续跑(完整重跑): {run_id} -> {new_run_id}，无未完成任务但 run 未结束（非汇总中断）")
        predefined_tasks = None

    return StreamingResponse(
        _make_sse_stream(thread_id, new_run_id, original_query, predefined_tasks=predefined_tasks),
        media_type="text/event-stream",
    )


# ========== 文档增量同步端点 ==========


@router.post("/api/admin/sync")
async def admin_sync():
    """增量同步 data/documents/ 目录到向量库，并刷新检索器缓存（含 BM25）。"""
    from app.ingestion.sync import sync_documents
    from app.rag.retriever import refresh_retriever

    def _run():
        result = sync_documents()
        refresh_retriever()
        return result

    try:
        result = await asyncio.to_thread(_run)
        logger.info(f"同步完成: {result}")
        return {"success": True, **result}
    except Exception as err:
        logger.error(f"同步失败: {err}")
        traceback.print_exc()
        return {"success": False, "message": str(err)}


# ========== 健康检查端点 ==========


@router.get("/health")
async def health():
    """返回服务及所有依赖的健康状态"""
    import asyncpg
    from app.rag.retriever import _cached_hybrid_retriever

    status = {"status": "ok", "checks": {}}

    # 1. 数据库连通性
    try:
        db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", ""),
            "database": os.getenv("DB_NAME", "RAG_test"),
        }
        conn = await asyncpg.connect(**db_config)
        await conn.execute("SELECT 1")
        await conn.close()
        status["checks"]["database"] = "ok"
    except Exception as err:
        status["status"] = "degraded"
        status["checks"]["database"] = f"error: {err}"

    # 2. Agent 是否已加载
    status["checks"]["agent"] = "ok" if _agent_ref[0] is not None else "not_initialized"

    # 3. BM25 检索器缓存状态
    status["checks"]["retriever_cache"] = "ready" if _cached_hybrid_retriever is not None else "not_built"

    # 4. Redis 连通性（上下文记忆模块）
    try:
        from app.memory.redis_store import get_redis
        if await get_redis() is not None:
            status["checks"]["redis"] = "ok"
        else:
            status["status"] = "degraded"
            status["checks"]["redis"] = "unavailable"
    except Exception as err:
        status["status"] = "degraded"
        status["checks"]["redis"] = f"error: {err}"

    return status
