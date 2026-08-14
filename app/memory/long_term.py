# long_term.py — 长期记忆异步持久化（PGVector memory_long 表）
#
# 流程：LLM 文本摘要提取 → bge-m3 向量化（Ollama）→ 写入 memory_long
#   - sha256(content) 指纹幂等去重（ON CONFLICT DO NOTHING）
#   - 阻塞的 embedding / LLM 调用放入线程池（asyncio.to_thread）
#   - 持久化失败只记日志，绝不阻塞 / 拖垮主对话流程
#
# 共享 DB 辅助函数（_db_config / _with_vector_codec / _get_embeddings）
# 同时供 app.memory.retriever 复用。

import asyncio
import hashlib
from typing import Optional

import asyncpg

from app.core.config import (
    DB_HOST,
    DB_PORT,
    DB_USER,
    DB_PASSWORD,
    DB_NAME,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)
from app.core.logging import get_logger

logger = get_logger()

_embeddings = None

_MEMORY_DDL = """
CREATE TABLE IF NOT EXISTS memory_long (
    id          BIGSERIAL PRIMARY KEY,
    thread_id   TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(1024),
    fingerprint TEXT UNIQUE,
    created_at  TIMESTAMPTZ DEFAULT now()
)
"""

_HNSW_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_memory_long_hnsw
ON memory_long USING hnsw (embedding vector_cosine_ops)
"""


# ========== 共享 DB / 嵌入辅助 ==========


def _db_config() -> dict:
    return {
        "host": DB_HOST,
        "port": DB_PORT,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "database": DB_NAME,
    }


async def _with_vector_codec(conn) -> None:
    """为 asyncpg 注册 pgvector 的 vector 类型 codec（文本格式）。

    asyncpg 原生无法编解码 vector 类型，不注册会报
    "cannot determine type of parameter" / "no codec available"。
    """
    await conn.set_type_codec(
        "vector",
        encoder=lambda v: "[" + ",".join(str(float(x)) for x in v) + "]",
        decoder=lambda s: [float(x) for x in s.strip("[]").split(",")] if s else [],
        format="text",
    )


def _get_embeddings():
    """惰性单例：bge-m3 嵌入模型（Ollama 本地服务）"""
    global _embeddings
    if _embeddings is None:
        from langchain_ollama import OllamaEmbeddings

        _embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
    return _embeddings


# ========== 建表 ==========


async def init_memory_tables() -> None:
    """创建 memory_long 表与 HNSW 索引（幂等），应用启动时调用。"""
    conn = await asyncpg.connect(**_db_config())
    try:
        await conn.execute(_MEMORY_DDL)
        await conn.execute(_HNSW_INDEX_DDL)
        logger.info("长期记忆表就绪: memory_long (pgvector + HNSW 索引)")
    finally:
        await conn.close()


# ========== 持久化 ==========


async def persist_memory(thread_id: str, content: str) -> bool:
    """将一段记忆异步写入 memory_long（sha256 指纹去重）。

    Returns:
        True: 新插入；False: 去重跳过 / 内容为空 / 失败。
    """
    if not content or not content.strip():
        return False

    try:
        embedding = await asyncio.to_thread(_get_embeddings().embed_query, content)
    except Exception as err:
        logger.warning(f"长期记忆向量化失败（跳过持久化）: {err}")
        return False

    fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
    try:
        conn = await asyncpg.connect(**_db_config())
        try:
            await _with_vector_codec(conn)
            result = await conn.execute(
                "INSERT INTO memory_long (thread_id, content, embedding, fingerprint) "
                "VALUES ($1, $2, $3, $4) "
                "ON CONFLICT (fingerprint) DO NOTHING",
                thread_id,
                content,
                embedding,
                fingerprint,
            )
            inserted = "INSERT 0 1" in result
            if inserted:
                logger.info(f"长期记忆已持久化: thread={thread_id}, len={len(content)}")
            return inserted
        finally:
            await conn.close()
    except Exception as err:
        logger.warning(f"长期记忆持久化失败（不影响主流程）: {err}")
        return False


async def persist_round(thread_id: str, turn_messages: list[dict]) -> None:
    """每轮对话结束后的异步任务入口：摘要提取 → 向量化 → 入库。"""
    from app.memory.summarizer import summarize_turn

    if not turn_messages:
        return
    content = await summarize_turn(turn_messages)
    if not content:
        logger.debug(f"本轮无长期记忆价值，跳过持久化: thread={thread_id}")
        return
    await persist_memory(thread_id, content)
