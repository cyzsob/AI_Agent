# core/exceptions.py — 项目自定义异常

class AppError(Exception):
    """应用基础异常"""


class RAGError(AppError):
    """RAG 检索相关异常"""


class AgentError(AppError):
    """Agent 执行异常"""


class ConfigError(AppError):
    """配置错误"""


class MCPConnectionError(AppError):
    """MCP 连接异常"""
