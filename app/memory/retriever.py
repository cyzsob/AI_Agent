# retriever.py — 长期记忆混合检索（语义 + 全文 → RRF 融合 → distance 阈值过滤）
#
# 检索策略：
#   语义路  ：query 经 bge-m3 向量化，与 memory_long.embedding 求余弦距离（<=> 算子），
#             仅保留 distance <= MEMORY_DISTANCE_THRESHOLD 的记忆（保证提取精度）
#   全文路  ：content ILIKE 子串命中（精确关键词召回，中文友好，无需额外扩展）
#   融合    ：RRF（Reciprocal Rank Fusion），两路权重各 0.5
# 任一环节失败均返回空列表，调用方静默降级为"无长期记忆"，不影响主对话。

import asyncio
from typing import Optional

import asyncpg

from app.core.config import (
    MEMORY_DISTANCE_THRESHOLD,
    MEMORY_RETRIEVAL_K,
)
from app.core.logging import get_logger
from app.memory.long_term import _db_config, _with_vector_codec, _get_embeddings

logger = get_logger()

_RRF_K = 60.0


async def retrieve_memories(query: str, k: Optional[int] = None) -> list[dict]:
    """混合检索长期记忆。

    Args:
        query: 当前用户问题（用作语义查询 + 全文子串）
        k: 返回条数，默认 MEMORY_RETRIEVAL_K

    Returns:
        [{ "id", "thread_id", "content", "distance", "score" }]
        按 RRF 融合分降序，最多 k 条；失败返回 []。
    """
    if not query or not query.strip():
        return []

    k = k or MEMORY_RETRIEVAL_K
    threshold = MEMORY_DISTANCE_THRESHOLD
    query = query.strip()

    # ---------- 1. 查询向量化（阻塞调用入线程池） ----------
    try:
        embedding = await asyncio.to_thread(_get_embeddings().embed_query, query)
    except Exception as err:
        logger.warning(f"长期记忆检索失败（查询向量化）: {err}")
        return []

    # ---------- 2. 两路召回 ----------
    try:
        conn = await asyncpg.connect(**_db_config())
        try:
            await _with_vector_codec(conn)
            vec_rows = await conn.fetch(
                "SELECT id, thread_id, content, "
                "(embedding <=> $1::vector) AS distance "
                "FROM memory_long ORDER BY distance ASC LIMIT $2",
                embedding,
                k * 3,
            )
            ft_rows = await conn.fetch(
                "SELECT id, thread_id, content FROM memory_long "
                "WHERE content ILIKE '%' || $1 || '%' LIMIT $2",
                query,
                k,
            )
        finally:
            await conn.close()
    except Exception as err:
        logger.warning(f"长期记忆检索失败（数据库查询）: {err}")
        return []

    # ---------- 3. distance 阈值过滤（语义路） ----------
    semantic = [
        row for row in vec_rows
        if row["distance"] is not None and float(row["distance"]) <= threshold
    ]

    # ---------- 4. RRF 融合 ----------
    score_map: dict = {}
    for i, row in enumerate(semantic):
        entry = score_map.setdefault(row["id"], {"row": row, "score": 0.0})
        entry["score"] += 0.5 / (_RRF_K + i + 1)
    for i, row in enumerate(ft_rows):
        entry = score_map.setdefault(row["id"], {"row": row, "score": 0.0})
        entry["score"] += 0.5 / (_RRF_K + i + 1)

    merged = sorted(score_map.values(), key=lambda x: x["score"], reverse=True)
    result = [
        {
            "id": item["row"]["id"],
            "thread_id": item["row"]["thread_id"],
            "content": item["row"]["content"],
            "distance": item["row"].get("distance"),
            "score": round(item["score"], 6),
        }
        for item in merged[:k]
    ]
    if result:
        logger.debug(f"长期记忆检索命中 {len(result)} 条 (query={query[:30]})")
    return result
