# redis_store.py — Redis 连接 + 短期记忆存取（带故障降级）
#
# 短期记忆在 Redis 中的 Key 规划：
#   memory:short:{thread_id}:messages   (List，元素为 JSON 编码的消息 dict，LTRIM/条数上限控制长度)
#   memory:short:{thread_id}:summary    (String，LLM 对已滑出窗口内容的滚动摘要)
#   memory:run:{run_id}                 (Hash，多 Agent 运行状态快照：planner 任务拆解 +
#                                        各 worker 执行进度，短 TTL，SSE 中断后可查)
#
# 所有 Key 均带 TTL（SHORT_MEMORY_TTL），每次访问刷新过期时间，实现"闲置过期"。
# Redis 不可用时各函数返回 None/False，由调用方（history.py）降级为进程内内存存储。

import json
import time
from typing import Optional

import redis.asyncio as aioredis

from app.core.config import REDIS_URL, RUN_STATE_TTL, SHORT_MEMORY_TTL
from app.core.logging import get_logger

logger = get_logger()

_MSG_KEY = "memory:short:{thread_id}:messages"
_SUMMARY_KEY = "memory:short:{thread_id}:summary"
_META_KEY = "memory:short:{thread_id}:meta"
_RUN_KEY = "memory:run:{run_id}"

_redis: Optional[aioredis.Redis] = None
_redis_down = False       # Redis 故障标记
_last_retry = 0.0         # 上次重试探测时间
_RETRY_INTERVAL = 30.0    # 故障后每隔 30s 重试一次探测
_PROBE_TIMEOUT = 1.0      # 连接探测超时（秒）


def _msg_key(thread_id: str) -> str:
    return _MSG_KEY.format(thread_id=thread_id)


def _summary_key(thread_id: str) -> str:
    return _SUMMARY_KEY.format(thread_id=thread_id)


def _meta_key(thread_id: str) -> str:
    return _META_KEY.format(thread_id=thread_id)


def _run_key(run_id: str) -> str:
    return _RUN_KEY.format(run_id=run_id)


async def get_redis() -> Optional[aioredis.Redis]:
    """惰性初始化 Redis 客户端并做连通性探测；不可用返回 None。

    探测失败后进入"降级窗口"（_RETRY_INTERVAL 内直接返回 None），
    避免 Redis 宕机时每次操作都阻塞等待连接超时。
    """
    global _redis, _redis_down, _last_retry

    now = time.monotonic()
    if _redis_down and now - _last_retry < _RETRY_INTERVAL:
        return None

    if _redis is None:
        _redis = aioredis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=_PROBE_TIMEOUT,
            socket_timeout=2.0,
        )

    try:
        await _redis.ping()
    except Exception as err:
        logger.warning(f"Redis 不可用，短期记忆降级为内存存储: {err}")
        _redis_down = True
        _last_retry = now
        return None

    if _redis_down:
        logger.info("Redis 恢复可用")
    _redis_down = False
    return _redis


async def get_messages(thread_id: str) -> Optional[list[dict]]:
    """读取短期记忆消息列表；Redis 不可用/读取失败返回 None。"""
    r = await get_redis()
    if r is None:
        return None
    try:
        raw = await r.lrange(_msg_key(thread_id), 0, -1)
        return [json.loads(x) for x in raw]
    except Exception as err:
        logger.warning(f"读取短期记忆失败: {err}")
        return None


async def set_messages(thread_id: str, records: list[dict]) -> bool:
    """整表覆盖写入短期记忆（先删后写），并刷新 TTL。"""
    r = await get_redis()
    if r is None:
        return False
    try:
        key = _msg_key(thread_id)
        pipe = r.pipeline()
        pipe.delete(key)
        if records:
            pipe.rpush(key, *[json.dumps(m, ensure_ascii=False) for m in records])
        pipe.expire(key, SHORT_MEMORY_TTL)
        await pipe.execute()
        return True
    except Exception as err:
        logger.warning(f"写入短期记忆失败: {err}")
        return False


async def get_summary(thread_id: str) -> Optional[str]:
    """读取滚动摘要；Redis 不可用/读取失败返回 None。"""
    r = await get_redis()
    if r is None:
        return None
    try:
        return await r.get(_summary_key(thread_id))
    except Exception as err:
        logger.warning(f"读取短期记忆摘要失败: {err}")
        return None


async def set_summary(thread_id: str, summary: str) -> bool:
    """写入滚动摘要（空串则删除），并刷新 TTL。"""
    r = await get_redis()
    if r is None:
        return False
    try:
        key = _summary_key(thread_id)
        pipe = r.pipeline()
        if summary:
            pipe.set(key, summary)
            pipe.expire(key, SHORT_MEMORY_TTL)
        else:
            pipe.delete(key)
        await pipe.execute()
        return True
    except Exception as err:
        logger.warning(f"写入短期记忆摘要失败: {err}")
        return False


async def delete_thread(thread_id: str) -> None:
    """删除指定会话的全部短期记忆 Key。"""
    r = await get_redis()
    if r is None:
        return
    try:
        await r.delete(_msg_key(thread_id), _summary_key(thread_id))
    except Exception as err:
        logger.warning(f"清理短期记忆失败: {err}")


async def get_session_meta(thread_id: str) -> Optional[dict]:
    """读取会话元数据；Redis 不可用/读取失败返回 None。"""
    r = await get_redis()
    if r is None:
        return None
    try:
        raw = await r.get(_meta_key(thread_id))
        return json.loads(raw) if raw else None
    except Exception as err:
        logger.warning(f"读取会话元数据失败: {err}")
        return None


async def set_session_meta(thread_id: str, meta: dict) -> bool:
    """写入会话元数据，并刷新 TTL。"""
    r = await get_redis()
    if r is None:
        return False
    try:
        key = _meta_key(thread_id)
        pipe = r.pipeline()
        pipe.set(key, json.dumps(meta, ensure_ascii=False))
        pipe.expire(key, SHORT_MEMORY_TTL)
        await pipe.execute()
        return True
    except Exception as err:
        logger.warning(f"写入会话元数据失败: {err}")
        return False


async def delete_session_meta(thread_id: str) -> None:
    """删除指定会话的元数据 Key。"""
    r = await get_redis()
    if r is None:
        return
    try:
        await r.delete(_meta_key(thread_id))
    except Exception as err:
        logger.warning(f"清理会话元数据失败: {err}")


async def flush_memory() -> int:
    """清空全部记忆 Key（测试 / 管理用）。返回删除数量，Redis 不可用时返回 0。"""
    r = await get_redis()
    if r is None:
        return 0
    try:
        keys = await r.keys("memory:*")
        if keys:
            return await r.delete(*keys)
        return 0
    except Exception as err:
        logger.warning(f"清空记忆失败: {err}")
        return 0


# ========== 多 Agent 运行状态（Run State） ==========
# Hash 字段规划：
#   thread_id / original_query / status / tasks(JSON 列表，静态计划) / created_at / updated_at
#   task:{target}  各 worker 的实时进度：pending|running|done|failed（HSET 单字段原子更新，
#                  并行 worker 各自写自己的字段，无并发覆盖）
# TTL 短（RUN_STATE_TTL），SSE 断开/服务重启后用于查询"上次进行到哪一步"。


async def update_run_state(run_id: str, **fields) -> bool:
    """写入/更新指定 run 的状态字段（HSET 按字段原子更新），并刷新 TTL。

    tasks 等非字符串值会 JSON 编码存储；Redis 不可用时返回 False，不影响主流程。
    """
    if not run_id:
        return False
    r = await get_redis()
    if r is None:
        return False
    try:
        key = _run_key(run_id)
        pipe = r.pipeline()
        for k, v in fields.items():
            if v is None:
                continue
            pipe.hset(key, k, v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
        pipe.hset(key, "updated_at", str(time.time()))
        pipe.expire(key, RUN_STATE_TTL)
        await pipe.execute()
        return True
    except Exception as err:
        logger.warning(f"写入运行状态失败: {err}")
        return False


async def get_run_state(run_id: str) -> Optional[dict]:
    """读取 run 状态快照；不存在或 Redis 不可用返回 None。

    返回结构：{thread_id, original_query, status, tasks: [{...}, status], updated_at, ...}
    tasks 数组内每项附上实时 status（来自 task:{target} 字段）。
    """
    r = await get_redis()
    if r is None:
        return None
    try:
        raw = await r.hgetall(_run_key(run_id))
        if not raw:
            return None
        data = dict(raw)
        # 合并各 task 的实时进度
        tasks_raw = data.get("tasks")
        tasks = json.loads(tasks_raw) if tasks_raw else []
        for t in tasks:
            t["status"] = data.get(f"task:{t.get('target')}", "pending")
        data["tasks"] = tasks
        for k in list(data):
            if k.startswith("task:"):
                data.pop(k)
        data["updated_at"] = float(data.get("updated_at", 0))
        return data
    except Exception as err:
        logger.warning(f"读取运行状态失败: {err}")
        return None


async def delete_run_state(run_id: str) -> None:
    """删除指定 run 的状态（测试 / 管理用）。"""
    r = await get_redis()
    if r is None:
        return
    try:
        await r.delete(_run_key(run_id))
    except Exception as err:
        logger.warning(f"清理运行状态失败: {err}")
