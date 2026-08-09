# sse_events.py — AG-UI 协议事件类型、常量与 SSE 格式化工具

import json
import os
from enum import StrEnum


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


def sse_event(data: dict) -> str:
    """将 dict 格式化为 SSE data 行"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


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


def is_internal_tool(name: str) -> bool:
    """判断工具名是否为内部编排工具（supervisor 路由 / worker 跨域交接），
    是则不在 SSE 流中透出、不保存到历史。"""
    return name in TRANSFER_TOOL_NAMES or name.startswith(HANDOFF_TOOL_PREFIX)


# ========== 流式调试开关 ==========
# 逐 token 调试输出由 logger.debug 控制；STREAM_DEBUG 控制更细粒度的
# LangGraph 内部事件日志（ns/node/chunk_type），仅在需要排查子图事件时开启。
STREAM_DEBUG = os.getenv("STREAM_DEBUG") == "1"
