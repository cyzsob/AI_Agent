# server_agui.py — AG-UI 协议兼容的 Agent + RAG 服务器（逐 token 流式输出）

import asyncio
import os
import json
import uuid
import base64
import traceback
from enum import StrEnum
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from agent import init_agent
from mcp_client import disconnect_all_mcp_servers
from conversation_history import (
    get_or_create_history,
    to_langchain_messages,
    append_messages,
    clear_history,
)
from multimodal import describe_image, download_image


# ========== AG-UI EventType 枚举（本地定义，无 Python 等价包） ==========


class AgUiEventType(StrEnum):
    RUN_STARTED = "RUN_STARTED"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_ARGS = "TOOL_CALL_ARGS"
    TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
    TOOL_CALL_END = "TOOL_CALL_END"


# ========== 多 Agent 节点常量 ==========

# supervisor 节点只负责内部路由，其输出不透出给客户端、不进入历史
SUPERVISOR_NODE = "supervisor"

# supervisor 的路由工具名（兜底过滤，防止嵌套子图节点名与预期不一致时泄漏路由事件）
TRANSFER_TOOL_NAMES = {
    "finish_workflow",
    "TransferToKnowledge",
    "TransferToDevops",
    "TransferToGeneral",
}

# worker 的跨域交接工具名前缀（handoff_to_xxx）：属于内部编排信号，
# 不透出给客户端、不进入会话历史
HANDOFF_TOOL_PREFIX = "handoff_to_"


def _is_internal_tool(name: str) -> bool:
    """判断工具名是否为内部编排工具（supervisor 路由 / worker 跨域交接），
    是则不在 SSE 流中透出、不保存到历史。"""
    return name in TRANSFER_TOOL_NAMES or name.startswith(HANDOFF_TOOL_PREFIX)


# ========== 流式调试开关 ==========
# 逐 token 打印到控制台是阻塞式 I/O，会拖慢事件循环与 SSE 发送（Windows 终端
# 尤为明显）。生产环境默认关闭，仅在 STREAM_DEBUG=1 时打印逐 chunk 调试日志。
STREAM_DEBUG = os.getenv("STREAM_DEBUG") == "1"


# ========== Lifespan（管理全局初始化和清理） ==========

_agent = None
_tools = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent, _tools
    print("[Lifespan] 正在初始化 Agent…")
    _agent, _tools = await init_agent()
    yield
    print("[Lifespan] 正在关闭 MCP 连接…")
    await disconnect_all_mcp_servers()
    print("[Lifespan] 关闭完成")


app = FastAPI(title="DeepSeek RAG Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== SSE 辅助函数 ==========


def sse_event(data: dict) -> str:
    """将 dict 格式化为 SSE data 行"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def iter_tool_call_pieces(chunk):
    """从 AIMessage / AIMessageChunk 中提取工具调用增量片段 (id, name, args)。

    兼容两种形态：
      - AIMessageChunk.tool_call_chunks：list[dict]（当前 langgraph 版本为 dict）
      - AIMessage.tool_calls：list[dict]
    """
    pieces = getattr(chunk, "tool_call_chunks", None)
    if pieces is not None:
        for tc in pieces:
            if isinstance(tc, dict):
                yield tc.get("id"), tc.get("name"), tc.get("args")
            else:
                yield (
                    getattr(tc, "id", None),
                    getattr(tc, "name", None),
                    getattr(tc, "args", None),
                )
    else:
        for tc in getattr(chunk, "tool_calls", None) or []:
            if isinstance(tc, dict):
                yield tc.get("id"), tc.get("name"), tc.get("args")
            else:
                yield (
                    getattr(tc, "id", None),
                    getattr(tc, "name", None),
                    getattr(tc, "args", None),
                )


# ========== 多模态消息解析与图片理解（用户上传图片查询） ==========


def _strip_data_uri(value: str) -> str:
    """去除 data URI 前缀（如 data:image/png;base64,xxx → xxx），普通 base64 原样返回"""
    if isinstance(value, str) and value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    return value


def _extract_text_and_images(content) -> tuple[str, list[dict]]:
    """AG-UI 用户消息 content（str 或 content part 数组）→ (文本, 图片列表)。

    图片列表元素：
      - {"bytes": bytes}  内联 base64 图片
      - {"url": str}      http(s) URL 图片

    兼容两种消息格式：
      - 新版（2025-10 生效）：{type:"image", source:{type:"data"|"url", value, mimeType?}}
      - 旧版（BinaryInputContent 兼容）：{type:"image", data:<base64>, mimeType, url?}
    """
    if isinstance(content, str):
        return content, []

    text_parts = []
    image_parts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            text_parts.append(str(part.get("text", "")))
        elif ptype == "image":
            src = part.get("source") or {}
            stype = src.get("type")
            # 新版：source.data（内联 base64）/ source.url（http(s) URL）
            if stype == "data":
                raw = src.get("value")
                if raw:
                    image_parts.append({"bytes": base64.b64decode(_strip_data_uri(raw))})
            elif stype == "url":
                url = src.get("value")
                if url:
                    image_parts.append({"url": url})
            # 旧版兼容：data / url 字段（无 source 包装）
            elif part.get("data"):
                image_parts.append({"bytes": base64.b64decode(_strip_data_uri(part["data"]))})
            elif part.get("url"):
                image_parts.append({"url": part["url"]})
    return "\n".join(text_parts).strip(), image_parts


async def _describe_images(image_parts: list[dict], user_text: str) -> list[str]:
    """并发调用本地视觉模型描述多张图片。

    单张图片失败返回错误提示文本（不中断整体请求），保证文本部分照常回答。
    """
    async def _describe_one(idx: int, part: dict) -> str:
        try:
            if "bytes" in part:
                description = await describe_image(part["bytes"], user_text)
            else:
                image_bytes = await download_image(part["url"])
                description = await describe_image(image_bytes, user_text)
            if description:
                return description
            return f"（图片{idx + 1} 未能生成描述）"
        except Exception as err:
            print(f"[Multimodal] 图片{idx + 1} 理解失败: {err}")
            return f"（图片{idx + 1} 理解失败：{err}）"

    return await asyncio.gather(*[_describe_one(i, p) for i, p in enumerate(image_parts)])


def _compose_multimodal_message(text: str, descriptions: list[str]) -> str:
    """组装最终纯文本 user 消息：用户文字 + 各图片描述（图片→文字化）"""
    parts = []
    if text and text.strip():
        parts.append(text.strip())
    for i, desc in enumerate(descriptions):
        parts.append(f"[图片{i + 1}]: {desc}")
    return "\n\n".join(parts)


# ========== AG-UI 协议端点 ==========


@app.get("/capabilities")
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


@app.post("/api/chat")
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
        # 发送 RUN_STARTED
        yield sse_event({
            "type": AgUiEventType.RUN_STARTED,
            "threadId": thread_id,
            "runId": run_id,
        })

        try:
            # 从 messages 中提取最后一条 user 消息（支持字符串或多模态 content 数组）
            user_text = ""
            image_parts = []
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_text, image_parts = _extract_text_and_images(msg.get("content", ""))
                    break

            # 多模态：并发调用本地视觉模型把图片转成文字描述，再与文本合并
            if image_parts:
                descriptions = await _describe_images(image_parts, user_text)
                user_message = _compose_multimodal_message(user_text, descriptions)
                print(
                    f"[Multimodal] 收到 {len(image_parts)} 张图片，"
                    f"已生成描述: {len(user_message)} 字符"
                )
            else:
                user_message = user_text

            if not user_message.strip():
                raise ValueError("未找到用户消息或消息为空")

            # ========== 多轮对话：加载历史 + 构造完整消息 ==========
            user_message_record = {"role": "user", "content": user_message}
            history = get_or_create_history(thread_id)
            full_messages = [*to_langchain_messages(history), user_message_record]

            # ========== Agent 执行（逐 token 流式输出，多 Agent 图） ==========
            message_id = str(uuid.uuid4())
            text_message_started = False
            tool_call_ids = set()          # 已发送 TOOL_CALL_START 的 tool_call_id
            tool_call_args_buffer = {}     # 流式参数缓冲（按 tool_call_id 键控，发送后清空）
            tool_call_full_args = {}       # 完整参数（按 tool_call_id 键控，供保存历史，不清空）
            tool_call_name_buffer = {}     # 工具名（按 tool_call_id 键控）

            # 用于收集本轮消息（保存到历史）
            final_content = ""
            tool_messages = []

            # subgraphs=True：worker 是 create_react_agent 嵌套子图，必须开启
            # 才会透传子图内部的逐 token 事件（否则父图只收到整条 AIMessage）
            stream = _agent.astream(
                {"messages": full_messages},
                stream_mode="messages",
                subgraphs=True,
            )

            async for event in stream:
                # subgraphs=True 后事件结构为 (ns, (chunk, metadata))，
                # ns 是命名空间元组（父图事件为空 ()，worker 子图为 ("general:...",)）。
                # 兼容兜底旧格式 (chunk, metadata)。
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
                    print(
                        f"[Stream] ns={ns}, node={node_name}, type={chunk_type}, "
                        f"has_tool_call_id={getattr(chunk, 'tool_call_id', None) is not None}"
                    )

                # ---- Supervisor 节点：仅内部路由，不透出事件、不进入历史 ----
                # supervisor 事件 ns 为空且 node 名为 supervisor；其合成的路由
                # ToolMessage 同样带 supervisor 节点名，一并在此过滤。
                if node_name == SUPERVISOR_NODE:
                    # supervisor 输出是流式逐 token 的 AIMessageChunk，每个 token
                    # 都会进此分支，默认不打印（终端 I/O 阻塞事件循环），调试时开
                    if STREAM_DEBUG:
                        print(f"[Supervisor] 内部路由消息，跳过: {chunk_type}")
                    continue

                # ---- 工具执行结果（ToolMessage）：不依赖节点名 ----
                if isinstance(chunk, ToolMessage):
                    print(
                        f"[Tools] chunk instanceof ToolMessage=True, "
                        f"tool_call_id={getattr(chunk, 'tool_call_id', None)}"
                    )

                    # 内部编排工具（跨域交接等）的结果不透出、不进入历史
                    tool_msg_name = getattr(chunk, "name", "") or ""
                    if _is_internal_tool(tool_msg_name):
                        print(f"[Tools] 内部编排工具结果，跳过: {tool_msg_name}")
                        continue

                    if chunk.tool_call_id:
                        tc_id = chunk.tool_call_id
                        print(f"[Tools] 工具返回, tool_call_id: {tc_id}")

                        # 补发 TOOL_CALL_START（如果 Agent 流中没有发出）
                        if tc_id not in tool_call_ids:
                            tool_call_ids.add(tc_id)
                            tc_name = getattr(chunk, "name", "") or ""
                            print(f"[Agent] 补发 TOOL_CALL_START, toolCallId: {tc_id}, name: {tc_name}")
                            yield sse_event({
                                "type": AgUiEventType.TOOL_CALL_START,
                                "toolCallId": tc_id,
                                "toolCallName": tc_name,
                            })

                        # 收集工具返回消息（用于历史）
                        tool_messages.append({
                            "role": "tool",
                            "content": chunk.content if isinstance(chunk.content, str) else "",
                            "tool_call_id": tc_id,
                        })

                        tool_result_message_id = str(uuid.uuid4())
                        print(f"[Event] >>> TOOL_CALL_RESULT, toolCallId={tc_id}")
                        yield sse_event({
                            "type": AgUiEventType.TOOL_CALL_RESULT,
                            "messageId": tool_result_message_id,
                            "toolCallId": tc_id,
                            "content": chunk.content if isinstance(chunk.content, str) else "",
                        })

                        print(f"[Event] >>> TOOL_CALL_END, toolCallId={tc_id}")
                        yield sse_event({
                            "type": AgUiEventType.TOOL_CALL_END,
                            "toolCallId": tc_id,
                        })
                    else:
                        print(
                            f"[WARN] ToolMessage without tool_call_id! "
                            f"content={str(chunk.content)[:100]}"
                        )
                    continue

                # ---- Worker 节点输出（文本 + 工具调用） ----
                # subgraphs=True 后 worker 内部 LLM 是逐 token 的 AIMessageChunk；
                # AIMessageChunk 是 AIMessage 的子类，统一按 AIMessage 处理。
                if isinstance(chunk, AIMessage):
                    # 1) 工具调用信息（增量输出 / 完整输出）
                    tool_call_pieces = list(iter_tool_call_pieces(chunk))
                    if tool_call_pieces:
                        for tc_id, tc_name, tc_args in tool_call_pieces:
                            # 兜底过滤：路由工具 / 跨域交接工具不透出
                            if _is_internal_tool(tc_name or ""):
                                print(f"[Stream] 跳过内部编排工具事件: {tc_name}")
                                continue

                            if STREAM_DEBUG:
                                print(
                                    f"[ToolChunk] id={tc_id}, "
                                    f"args={str(tc_args)[:50] if tc_args else None}"
                                )

                            # TOOL_CALL_START — 只在首次出现时发送
                            if tc_id and tc_id not in tool_call_ids:
                                tool_call_ids.add(tc_id)
                                print(f"[Agent] 调用工具: {tc_name}, toolCallId: {tc_id}")
                                yield sse_event({
                                    "type": AgUiEventType.TOOL_CALL_START,
                                    "toolCallId": tc_id,
                                    "toolCallName": tc_name or "",
                                })

                            # 记录工具名称（按 tool_call_id 键控）
                            if tc_id and tc_name:
                                tool_call_name_buffer[tc_id] = tc_name

                            # TOOL_CALL_ARGS — 缓冲累积 args（按 tool_call_id 键控，
                            # 避免多节点下 index 从 0 重复导致的混叠）
                            if tc_id and tc_args:
                                # dict 为完整参数（非流式），转为 JSON 字符串再缓冲
                                args_str = (
                                    json.dumps(tc_args, ensure_ascii=False)
                                    if isinstance(tc_args, dict)
                                    else tc_args
                                )
                                tool_call_args_buffer[tc_id] = (
                                    tool_call_args_buffer.get(tc_id, "") + args_str
                                )
                                tool_call_full_args[tc_id] = (
                                    tool_call_full_args.get(tc_id, "") + args_str
                                )

                        # 检测已累积完整的参数并一次性发送
                        to_delete = []
                        for tc_id, accumulated_args in tool_call_args_buffer.items():
                            try:
                                json.loads(accumulated_args)
                                if tc_id:
                                    yield sse_event({
                                        "type": AgUiEventType.TOOL_CALL_ARGS,
                                        "toolCallId": tc_id,
                                        "delta": accumulated_args,
                                    })
                                    to_delete.append(tc_id)
                            except (json.JSONDecodeError, ValueError):
                                pass

                        for tc_id in to_delete:
                            tool_call_args_buffer.pop(tc_id, None)

                    # 2) 文本内容（逐 token 送 + 累积用于历史）
                    chunk_content = (
                        chunk.content
                        if isinstance(chunk.content, str)
                        else ""
                    )
                    if chunk_content:
                        final_content += chunk_content

                        if not text_message_started:
                            print(
                                f"[Event] >>> TEXT_MESSAGE_START, messageId={message_id}"
                            )
                            yield sse_event({
                                "type": AgUiEventType.TEXT_MESSAGE_START,
                                "messageId": message_id,
                                "role": "assistant",
                            })
                            text_message_started = True

                        if STREAM_DEBUG:
                            print(
                                f"[Event] >>> TEXT_MESSAGE_CONTENT, "
                                f"messageId={message_id}, delta={chunk_content[:50]}"
                            )
                        yield sse_event({
                            "type": AgUiEventType.TEXT_MESSAGE_CONTENT,
                            "messageId": message_id,
                            "delta": chunk_content,
                        })

            # ========== 保存本轮消息到历史 ==========
            turn_messages = []

            # 1) 用户消息
            turn_messages.append(user_message_record)

            # 2) 工具调用记录（遍历所有已触发的调用，保证与 tool_messages 一一配对）
            #    注意：无参数工具（如 get_deepseek_info）的 args 为 None/{}，
            #    不会被缓冲到 tool_call_full_args；多工具调用时若只遍历
            #    full_args 会漏掉无参调用，导致历史中 assistant(tool_call)
            #    与 tool(response) 不配对（第二轮报 INVALID_REQUEST 400）。
            #    因此必须以 tool_call_ids 为准遍历，args 缺失时用 {} 兜底。
            if tool_call_ids:
                tool_calls = []
                # 保序：tool_call_name_buffer 按键插入序 = 调用首次出现顺序
                ordered_ids = list(dict.fromkeys(
                    list(tool_call_name_buffer.keys()) + list(tool_call_ids)
                ))
                for tc_id in ordered_ids:
                    if tc_id not in tool_call_ids:
                        continue
                    raw_args = tool_call_full_args.get(tc_id)
                    if isinstance(raw_args, str):
                        try:
                            parsed_args = json.loads(raw_args)
                        except (json.JSONDecodeError, ValueError):
                            parsed_args = {}
                    else:
                        parsed_args = raw_args or {}
                    tool_calls.append({
                        "id": tc_id,
                        "name": tool_call_name_buffer.get(tc_id, ""),
                        "args": parsed_args,
                    })
                turn_messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                })

            # 3) 工具返回结果
            turn_messages.extend(tool_messages)

            # 4) 最终回答
            if final_content:
                turn_messages.append({
                    "role": "assistant",
                    "content": final_content,
                })

            append_messages(thread_id, turn_messages)

            # 兜底：确保所有已缓冲的 args 都被发送
            for tc_id, accumulated_args in tool_call_args_buffer.items():
                if tc_id and accumulated_args:
                    print(f"[WARN] 兜底发送残余 args, toolCallId={tc_id}")
                    yield sse_event({
                        "type": AgUiEventType.TOOL_CALL_ARGS,
                        "toolCallId": tc_id,
                        "delta": accumulated_args,
                    })
            tool_call_args_buffer.clear()

            # 如果 Agent 没有产生文本回答，确保发送 TEXT_MESSAGE_START
            if not text_message_started:
                yield sse_event({
                    "type": AgUiEventType.TEXT_MESSAGE_START,
                    "messageId": message_id,
                    "role": "assistant",
                })
                yield sse_event({
                    "type": AgUiEventType.TEXT_MESSAGE_CONTENT,
                    "messageId": message_id,
                    # 兜底文本（正常路径下 supervisor 已兜底路由，几乎不会触发）：
                    # 明确告知"未生成有效回复"，避免误导性的"请求处理完成"
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
            print(f"Agent 错误: {err}")
            traceback.print_exc()
            yield sse_event({
                "type": AgUiEventType.RUN_ERROR,
                "message": str(err),
                "code": "AGENT_ERROR",
            })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ========== 历史管理端点 ==========


@app.post("/api/chat/clear")
async def chat_clear(request: Request):
    """清除指定会话的历史"""
    body = await request.json()
    thread_id = body.get("threadId")
    if not thread_id:
        return {"success": False, "message": "缺少 threadId"}
    clear_history(thread_id)
    print(f"[Session] 已清除会话历史: {thread_id}")
    return {"success": True, "threadId": thread_id}


# ========== 文档增量同步端点 ==========


@app.post("/api/admin/sync")
async def admin_sync():
    """增量同步 data/documents/ 目录到向量库，并刷新检索器缓存（含 BM25）。

    新增/修改 documents/ 下的文档后调用本端点，服务无需重启即可检索到最新数据。
    """
    from ingest import sync_documents
    from hybrid_retriever import refresh_retriever

    def _run():
        result = sync_documents()
        refresh_retriever()
        return result

    try:
        # ingest 为同步脚本，放入线程池避免阻塞事件循环
        result = await asyncio.to_thread(_run)
        print(f"[Sync] 同步完成: {result}")
        return {"success": True, **result}
    except Exception as err:
        print(f"[Sync] 同步失败: {err}")
        traceback.print_exc()
        return {"success": False, "message": str(err)}


# ========== 启动 ==========

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "3001"))
    print(f"AG-UI Agent+RAG 服务器已启动:")
    print(f"  HTTP:      http://localhost:{port}")
    print(f"  SSE 聊天:  POST http://localhost:{port}/api/chat")
    print(f"  能力声明:  GET  http://localhost:{port}/capabilities")
    uvicorn.run(app, host="0.0.0.0", port=port)
