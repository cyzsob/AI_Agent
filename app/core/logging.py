# logger.py — 统一日志配置（基于 loguru，支持结构化日志、文件轮转、请求追踪）

import sys
from pathlib import Path
from loguru import logger

# 移除默认 handler
logger.remove()

# 控制台输出（彩色、开发友好）
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[request_id]}</cyan> | "
        "<level>{message}</level>"
    ),
    level="DEBUG",
    colorize=True,
)

# 文件输出（JSON 结构化，便于后续检索分析）
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger.add(
    LOG_DIR / "app_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {extra[request_id]} | {name}:{function}:{line} | {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
)

# 错误日志单独文件
logger.add(
    LOG_DIR / "error_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {extra[request_id]} | {name}:{function}:{line} | {message}",
    level="ERROR",
    rotation="10 MB",
    retention="30 days",
    encoding="utf-8",
)


def configure_contextual_logger():
    """返回一个已配置了默认 request_id 上下文的 logger，供模块级使用。

    用法：
        from app.core.logging import configure_contextual_logger
        logger = configure_contextual_logger()
        logger.info("something")
    """
    return logger.bind(request_id="-")


# 默认上下文 logger（request_id="-"）
_default_logger = logger.bind(request_id="-")


def get_logger():
    """获取默认 logger 实例。需要携带 request_id 时使用 logger.bind(request_id=xxx)。"""
    return _default_logger
