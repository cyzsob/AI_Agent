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
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from app.core.logging import get_logger
from app.api.sse import (
    AgUiEventType,
    sse_event,
    SUPERVISOR_NODE,
    TRANSFER_TOOL_NAMES,
    HANDOFF_TOOL_PREFIX,
    is_internal_tool,
    STREAM_DEBUG,
)
from app.services.message import (
    extract_text_and_images,
    describe_images,
    compose_multimodal_message,
)
from app.api.stream import iter_tool_call_pieces, ToolCallTracker

logger = get_logger()

router = APIRouter()

# Agent reference (injected by main.py at startup)
_agent_ref: list = [None]


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

    async def event_stream() -> AsyncGenerator[str, None]:
        request_id = str(uuid.uuid4())[:8]
        log = logger.bind(request_id=request_id)

        # 发送 RUN_STARTED
        yield sse_event({
            "type": AgUiEventType.RUN_STARTED,
            "threadId": thread_id,
            "runId": run_id,
        })

        try:
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
                log.info(f"收到 {len(image_parts)} 张图片，已生成描述: {len(user_message)} 字符")
            else:
                user_message = user_text

            if not user_message.strip():
                raise ValueError("未找到用户消息或消息为空")

            # 多轮对话：加载历史 + 构造完整消息
            from app.agent.history import (
                get_or_create_history,
                to_langchain_messages,
                append_messages,
            )

            user_message_record = {"role": "user", "content": user_message}
            history = get_or_create_history(thread_id)
            full_messages = [*to_langchain_messages(history), user_message_record]

            # Agent 流式执行
            message_id = str(uuid.uuid4())
            text_message_started = False
            tracker = ToolCallTracker()
            final_content = ""

            stream = _agent_ref[0].astream(
                {"messages": full_messages},
                stream_mode="messages",
                subgraphs=True,
            )

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

                # ---- Worker 节点输出（文本 + 工具调用） ----
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

                    # 文本内容
                    chunk_content = chunk.content if isinstance(chunk.content, str) else ""
                    if chunk_content:
                        final_content += chunk_content

                        if not text_message_started:
                            log.debug(f">>> TEXT_MESSAGE_START, messageId={message_id}")
                            yield sse_event({
                                "type": AgUiEventType.TEXT_MESSAGE_START,
                                "messageId": message_id,
                                "role": "assistant",
                            })
                            text_message_started = True

                        if STREAM_DEBUG:
                            log.debug(f">>> TEXT_MESSAGE_CONTENT, delta={chunk_content[:50]}")
                        yield sse_event({
                            "type": AgUiEventType.TEXT_MESSAGE_CONTENT,
                            "messageId": message_id,
                            "delta": chunk_content,
                        })

            # ========== 保存本轮消息到历史 ==========
            turn_messages = []

            # 用户消息
            turn_messages.append(user_message_record)

            # 工具调用记录 + 工具返回结果
            if tracker.tool_call_ids:
                turn_messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tracker.build_history_tool_calls(),
                })
            turn_messages.extend(tracker.tool_messages)

            # 最终回答
            if final_content:
                turn_messages.append({
                    "role": "assistant",
                    "content": final_content,
                })

            append_messages(thread_id, turn_messages)

            # 兜底：发送残余 args
            for event_str in tracker.flush_remaining_args_events(sse_event):
                log.warning(f"兜底发送残余 args")
                yield event_str

            # 兜底文本开始
            if not text_message_started:
                yield sse_event({
                    "type": AgUiEventType.TEXT_MESSAGE_START,
                    "messageId": message_id,
                    "role": "assistant",
                })
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

            # RUN_FINISHED
            yield sse_event({
                "type": AgUiEventType.RUN_FINISHED,
                "threadId": thread_id,
                "runId": run_id,
                "outcome": {"type": "success"},
            })

        except Exception as err:
            log.error(f"Agent 错误: {err}")
            traceback.print_exc()
            yield sse_event({
                "type": AgUiEventType.RUN_ERROR,
                "message": str(err),
                "code": "AGENT_ERROR",
            })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ========== 历史管理端点 ==========


@router.post("/api/chat/clear")
async def chat_clear(request: Request):
    """清除指定会话的历史"""
    from app.agent.history import clear_history

    body = await request.json()
    thread_id = body.get("threadId")
    if not thread_id:
        return {"success": False, "message": "缺少 threadId"}
    clear_history(thread_id)
    logger.info(f"已清除会话历史: {thread_id}")
    return {"success": True, "threadId": thread_id}


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

    return status
