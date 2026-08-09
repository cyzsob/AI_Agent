# mcp_client.py — MCP 客户端连接管理器
# 连接 MCP Server，发现工具并包装为 LangChain StructuredTool

import os
import json
import time
import asyncio
from contextlib import AsyncExitStack
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, create_model, Field

from app.core.logging import get_logger
logger = get_logger()

# 所有活跃的 MCP 连接（用于清理）
# 每个连接包含: exit_stack, session, name
_active_connections: list[dict] = []


def json_schema_to_pydantic(input_schema: dict) -> type[BaseModel]:
    """将 MCP 的 JSON Schema 参数定义转为 Pydantic model"""
    if not input_schema or "properties" not in input_schema:
        return create_model("EmptyInput")

    properties = input_schema.get("properties", {})
    required_fields = input_schema.get("required", [])

    fields = {}
    for key, prop in properties.items():
        prop_type = prop.get("type", "string")
        description = prop.get("description", "")
        is_required = key in required_fields

        if prop_type == "string":
            py_type = str
        elif prop_type == "number":
            py_type = float
        elif prop_type == "integer":
            py_type = int
        elif prop_type == "boolean":
            py_type = bool
        elif prop_type == "array":
            py_type = list
        elif prop_type == "object":
            py_type = dict
        elif "enum" in prop:
            py_type = str
        else:
            py_type = Any

        if is_required:
            fields[key] = (py_type, Field(description=description))
        else:
            fields[key] = (py_type, Field(default=None, description=description))

    if not fields:
        return create_model("EmptyInput")

    return create_model("ToolInput", **fields)


async def create_tools_from_mcp_server(
    server_config: dict, exit_stack: AsyncExitStack
) -> list:
    """连接单个 MCP Server，返回其暴露的所有 LangChain 工具

    Args:
        server_config: dict with keys: name, command, args, env
        exit_stack: AsyncExitStack 实例，用于管理连接的 context manager

    Returns:
        list[StructuredTool]
    """
    name = server_config["name"]
    command = server_config["command"]
    args = server_config.get("args", [])
    env = server_config.get("env", {})

    logger.info(f"连接服务器: {name}")
    logger.debug(f"命令: {command} {' '.join(args)}")

    merged_env = {**os.environ, **env}

    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=merged_env,
    )

    # 使用 exit_stack 管理 stdio_client 和 ClientSession 的生命周期（含重试）
    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        try:
            start = time.perf_counter()
            read, write = await exit_stack.enter_async_context(stdio_client(server_params))
            session = await exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"{name}: 连接成功 ({elapsed:.0f}ms)")
            break
        except Exception as err:
            last_error = err
            if attempt < max_retries - 1:
                backoff = 2 ** attempt
                logger.warning(f"{name}: 连接失败 (第{attempt + 1}次), {backoff}s 后重试: {err}")
                await asyncio.sleep(backoff)
            else:
                raise RuntimeError(f"{name}: 连接失败（已重试 {max_retries} 次）: {last_error}") from last_error

    # 发现该 Server 暴露的所有工具
    tools_result = await session.list_tools()
    tools = tools_result.tools
    logger.info(f"{name}: 发现 {len(tools)} 个工具")

    # 将每个 MCP Tool 包装为 LangChain StructuredTool
    langchain_tools = []
    for mcp_tool in tools:
        tool_name = f"{name}_{mcp_tool.name}"  # 加前缀防命名冲突

        # 创建 Pydantic schema（注意：Python MCP SDK 使用 input_schema）
        input_schema = mcp_tool.input_schema if hasattr(mcp_tool, "input_schema") else getattr(mcp_tool, "inputSchema", None)
        if input_schema:
            pydantic_model = json_schema_to_pydantic(input_schema)
        else:
            pydantic_model = create_model("EmptyInput")

        # 创建工具调用函数（用工厂函数避免闭包共享循环变量）
        def build_call_tool(srv_name: str, tool_name: str, sess: ClientSession):
            async def call_tool(**kwargs) -> str:
                try:
                    result = await sess.call_tool(
                        name=tool_name,
                        arguments=kwargs,
                    )
                    parts = []
                    for c in result.content:
                        if hasattr(c, "text"):
                            parts.append(c.text)
                        elif hasattr(c, "resource"):
                            parts.append(json.dumps(c.resource))
                        else:
                            parts.append(str(c))
                    return "\n".join(parts)
                except Exception as err:
                    logger.error(f"{srv_name}.{tool_name} 调用失败: {err}")
                    return f"工具调用失败: {err}"
            return call_tool

        wrapped_tool = StructuredTool.from_function(
            name=tool_name,
            description=mcp_tool.description or "",
            args_schema=pydantic_model,
            coroutine=build_call_tool(name, mcp_tool.name, session),
        )

        langchain_tools.append(wrapped_tool)

    # 记录连接
    _active_connections.append({
        "name": name,
        "exit_stack": exit_stack,
    })

    return langchain_tools


async def disconnect_all_mcp_servers() -> None:
    """断开所有 MCP Server 连接"""
    for conn in reversed(_active_connections):
        try:
            await conn["exit_stack"].aclose()
            logger.info(f"断开连接: {conn['name']}")
        except Exception as err:
            logger.error(f"断开失败: {conn['name']} {err}")
    _active_connections.clear()


async def load_all_mcp_tools(config_path: str) -> list:
    """从配置文件加载所有 MCP Server 的工具

    Args:
        config_path: MCP 服务器配置文件路径 (JSON)

    Returns:
        list[StructuredTool]
    """
    if not os.path.exists(config_path):
        logger.warning(f"配置文件不存在: {config_path}，跳过 MCP 工具加载")
        return []

    with open(config_path, "r", encoding="utf-8") as f:
        server_configs = json.load(f)

    if not isinstance(server_configs, list) or len(server_configs) == 0:
        logger.warning("配置文件为空，跳过")
        return []

    # 为所有 MCP 连接创建一个共享的 AsyncExitStack
    # FastAPI lifespan 中传入同一个 exit_stack，保持生命周期一致
    exit_stack = AsyncExitStack()

    all_tools = []
    for config in server_configs:
        try:
            tools = await create_tools_from_mcp_server(config, exit_stack)
            all_tools.extend(tools)
        except Exception as err:
            logger.error(f'加载服务器 "{config["name"]}" 失败: {err}')

    _active_connections.clear()
    _active_connections.append({
        "name": "_shared_exit_stack",
        "exit_stack": exit_stack,
    })

    return all_tools
