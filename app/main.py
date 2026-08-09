# main.py — 应用入口：FastAPI 工厂 + Lifespan + 启动

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.logging import get_logger

logger = get_logger()

# ========== App 工厂 ==========


def create_app() -> FastAPI:
    """构建并返回 FastAPI 应用实例"""
    app = FastAPI(title="DeepSeek RAG Agent", lifespan=_lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 延迟导入路由（避免循环依赖）
    from app.api.routes import router
    app.include_router(router)

    return app


# ========== Lifespan（管理全局初始化和清理） ==========


async def _agent_init():
    """在 lifespan 中延迟初始化 agent，供 routes 注入"""
    from app.agent import init_agent
    from app.mcp.client import disconnect_all_mcp_servers

    logger.info("正在初始化 Agent…")
    agent, tools = await init_agent()

    # 注入到 routes 模块
    from app.api import routes
    routes._agent_ref[0] = agent

    return agent, tools, disconnect_all_mcp_servers


@asynccontextmanager
async def _lifespan(app: FastAPI):
    agent, tools, disconnect_fn = await _agent_init()
    try:
        yield
    finally:
        logger.info("正在关闭 MCP 连接…")
        try:
            await disconnect_fn()
        except Exception as err:
            logger.error(f"MCP 关闭异常: {err}")
        logger.info("关闭完成")


# ========== 启动 ==========

if __name__ == "__main__":
    import uvicorn

    from app.core.config import SERVER_PORT

    app = create_app()
    logger.info("AG-UI Agent+RAG 服务器已启动:")
    logger.info(f"  HTTP:      http://localhost:{SERVER_PORT}")
    logger.info(f"  SSE 聊天:  POST http://localhost:{SERVER_PORT}/api/chat")
    logger.info(f"  能力声明:  GET  http://localhost:{SERVER_PORT}/capabilities")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
