# core/config.py — 集中配置管理

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 优先从项目根目录加载 .env
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")


def _require_env(key: str, description: str) -> str:
    """读取必需环境变量，缺失则报错退出"""
    value = os.getenv(key)
    if not value:
        sys.exit(f"[Config] 缺少必要的环境变量 {key} ({description})，请检查 .env 文件")
    return value


# ---- LLM ----
DEEPSEEK_API_KEY = _require_env("DEEPSEEK_API_KEY", "DeepSeek API 密钥")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# ---- PostgreSQL / PGVector ----
DB_HOST = _require_env("DB_HOST", "数据库主机地址")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = _require_env("DB_NAME", "数据库名称")

# ---- 嵌入模型 ----
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ---- Redis / 上下文记忆 ----
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SHORT_MEMORY_TTL = int(os.getenv("SHORT_MEMORY_TTL", "3600"))
SHORT_MEMORY_MAX_MSGS = int(os.getenv("SHORT_MEMORY_MAX_MSGS", "20"))
RUN_STATE_TTL = int(os.getenv("RUN_STATE_TTL", "900"))
MEMORY_DISTANCE_THRESHOLD = float(os.getenv("MEMORY_DISTANCE_THRESHOLD", "0.6"))
MEMORY_RETRIEVAL_K = int(os.getenv("MEMORY_RETRIEVAL_K", "5"))
MEMORY_LONG_ENABLED = os.getenv("MEMORY_LONG_ENABLED", "true").lower() == "true"

# ---- 重排模型 ----
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

# ---- 多模态 ----
GLM_API_KEY = os.getenv("GLM_API_KEY", "")

# ---- 服务 ----
SERVER_PORT = int(os.getenv("PORT", "3001"))

# ---- 检索参数 ----
VECTOR_K = int(os.getenv("VECTOR_K", "10"))
BM25_K = int(os.getenv("BM25_K", "10"))
FINAL_K = int(os.getenv("FINAL_K", "3"))
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "5"))
