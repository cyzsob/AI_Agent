# cleanup_run_state.py — 清理 Redis 中的运行状态（run-state）记录
#
# 用途：SSE 流卡死（如模型调用挂起）会留下僵尸 run-state（status 永远停在
#       running、worker 却已全部 done），导致前端"继续"接口拿到 400。此脚本
#       用于手动清理这些残留记录。
#
# 用法：
#   列出当前所有 run-state（不删除）：
#     .venv\Scripts\python.exe -m app.scripts.cleanup_run_state
#   删除指定 run：
#     .venv\Scripts\python.exe -m app.scripts.cleanup_run_state a0cc3b81-7bbf-4f46-844a-a2e4350e8a0e [run_id2 ...]
#   删除全部 run-state：
#     .venv\Scripts\python.exe -m app.scripts.cleanup_run_state --all

import asyncio
import json
import sys

from app.memory.redis_store import (
    delete_run_state,
    get_run_state,
    get_redis,
)


def _brief(state: dict) -> str:
    """把 run-state 压缩成一行摘要，便于输出。"""
    tasks = state.get("tasks") or []
    statuses = [(t.get("target"), t.get("status")) for t in tasks]
    return json.dumps(
        {
            "status": state.get("status"),
            "tasks": statuses,
            "thread_id": state.get("thread_id", ""),
        },
        ensure_ascii=False,
    )


async def main() -> None:
    r = await get_redis()
    if r is None:
        print("Redis 不可用，无法执行清理（检查 REDIS_URL / Redis 服务）")
        sys.exit(1)

    keys = [k for k in await r.keys("memory:run:*") if k.startswith("memory:run:")]
    if not keys:
        print("当前没有任何 run-state 记录")
        return

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--all" in sys.argv:
        targets = [k.removeprefix("memory:run:") for k in keys]
    elif args:
        targets = args
    else:
        # 无参数：只列出不删除
        print(f"发现 {len(keys)} 条 run-state（未删除，可用 run_id 或 --all 指定清理目标）：")
        for key in keys:
            run_id = key.removeprefix("memory:run:")
            state = await get_run_state(run_id)
            print(f"  - {run_id}  {_brief(state) if state else '(读取失败)'}")
        return

    for run_id in targets:
        state = await get_run_state(run_id)
        if state is None:
            print(f"  [跳过] {run_id}：不存在（可能已过期自动清理）")
            continue
        await delete_run_state(run_id)
        print(f"  [已删除] {run_id}  {_brief(state)}")


if __name__ == "__main__":
    asyncio.run(main())
