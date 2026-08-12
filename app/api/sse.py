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
    # AG-UI 标准自定义事件：阶段进度提示（拆解中/处理中/汇总中），
    # 供前端在首字到达前展示"进行中"状态，避免长时间静默
    CUSTOM = "CUSTOM"


def sse_custom(name: str, value: dict) -> str:
    """格式化 AG-UI CUSTOM 自定义事件。

    Args:
        name: 自定义事件名（如 "status"），供前端按名分发
        value: 任意 JSON 可序列化负载
    """
    return sse_event({"type": AgUiEventType.CUSTOM, "name": name, "value": value})


def sse_event(data: dict) -> str:
    """将 dict 格式化为 SSE data 行"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ========== 多 Agent 节点常量 ==========

# supervisor 节点只负责内部路由，其输出不透出给客户端、不进入历史
SUPERVISOR_NODE = "supervisor"

# 汇总器节点：唯一向客户端透出最终回答文本的节点
SUMMARIZER_NODE = "summarizer"

# 内部编排工具名（Planner 模式下已无 TransferTo*/handoff 工具，保留空集合兜底过滤）
TRANSFER_TOOL_NAMES: set[str] = set()


def is_internal_tool(name: str) -> bool:
    """判断工具名是否为内部编排工具，是则不在 SSE 流中透出、不保存到历史。"""
    return name in TRANSFER_TOOL_NAMES


# ========== 流式调试开关 ==========
# 逐 token 调试输出由 logger.debug 控制；STREAM_DEBUG 控制更细粒度的
# LangGraph 内部事件日志（ns/node/chunk_type），仅在需要排查子图事件时开启。
STREAM_DEBUG = os.getenv("STREAM_DEBUG") == "1"
