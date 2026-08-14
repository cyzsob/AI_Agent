"""中断 → 查询 → 续跑 完整链路演练（需要服务已启动，如端口 3002）。

流程：
  1. 发一个必走完整多 Agent 流程的问题（SSE 流式，后台线程接收）
  2. 等待进入 worker 阶段后，POST /api/runs/{run_id}/cancel 显式中断
  3. GET /api/runs/{run_id} 查看中断现场（status=interrupted + 各任务状态）
  4. POST /api/runs/{run_id}/resume 续跑，收集新的 SSE 流与最终回答
  5. 查询新 run 状态（应为 done）

运行：.venv\\Scripts\\python.exe tests/drill_interrupt_resume.py [端口]
"""

import json
import sys
import threading
import time

import requests

PORT = sys.argv[1] if len(sys.argv) > 1 else "3002"
BASE = f"http://localhost:{PORT}"
THREAD_ID = f"drill-{int(time.time())}"
RUN_ID = f"run-{int(time.time())}"

# 避开 fast-path 关键词，强制走完整多 Agent 流程（planner → worker → summarizer）
QUERY = "帮我整理一下最近的学习计划，再分析一下当前的开源社区整体趋势，给出几点建议"


def collect_stream(url: str, events: list, payload: dict | None = None):
    """在独立线程里消费 SSE 流，把 data: 解析成 dict 追加到 events。"""
    try:
        if payload is not None:
            resp = requests.post(url, json=payload, stream=True, timeout=(10, 180))
        else:
            resp = requests.post(url, stream=True, timeout=(10, 180))
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                try:
                    events.append(json.loads(line[5:].strip()))
                except json.JSONDecodeError:
                    pass
    except Exception as err:
        events.append({"type": "CLIENT_EXCEPTION", "error": str(err)})


def dump_state(tag: str, state: dict):
    print(f"\n== {tag} ==")
    tasks = state.get("tasks", [])
    meta = {k: v for k, v in state.items() if k != "tasks"}
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    if tasks:
        print("tasks:", [(t.get("target"), t.get("status")) for t in tasks])


def wait_until_worker_phase(run_id: str, timeout: float = 150.0) -> dict | None:
    """轮询 run-state，直到出现任务拆解且至少一个 worker 为 running 状态。

    返回该时刻的 run-state 快照；超时返回最后一次响应（可能是 404 / 错误）。
    用于替代固定 sleep：冷启动（Ollama 首次加载 bge-m3 等）时预图阶段耗时
    可能远超 6s，轮询可自适应等待真正进入 worker 阶段再中断。
    """
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            resp = requests.get(f"{BASE}/api/runs/{run_id}", timeout=5)
            if resp.status_code == 200:
                state = resp.json()
                last = state
                tasks = state.get("tasks") or []
                if any(t.get("status") == "running" for t in tasks):
                    return state
        except Exception as err:
            last = {"error": str(err)}
        time.sleep(1)
    return last


def main() -> None:
    events: list[dict] = []
    t = threading.Thread(
        target=collect_stream,
        args=(f"{BASE}/api/chat", events),
        kwargs={"payload": {
            "threadId": THREAD_ID,
            "runId": RUN_ID,
            "messages": [{"role": "user", "content": QUERY}],
        }},
        daemon=True,
    )
    t.start()

    # 1) 轮询等待 planner 拆解完成、进入 worker 阶段后再中断
    state = wait_until_worker_phase(RUN_ID)
    print("== 轮询结果 ==")
    dump_state("中断前 run-state（worker 阶段）", state)
    if not state or not any(
        t.get("status") == "running" for t in (state.get("tasks") or [])
    ):
        print("!! 未在超时时间内观察到 running 状态的 worker，继续尝试 cancel")

    # 2) 显式中断
    cancel = requests.post(f"{BASE}/api/runs/{RUN_ID}/cancel", timeout=5).json()
    print("\n== cancel ==", cancel)

    t.join(timeout=30)
    print("\n== 原 SSE 流收到的事件 ==")
    for e in events:
        tp = e.get("type")
        if tp in ("TEXT_MESSAGE_CONTENT", "CUSTOM"):
            continue
        print(f"{tp}: {json.dumps(e, ensure_ascii=False)[:160]}")

    # 3) 查询中断现场
    try:
        dump_state("中断后 run-state", requests.get(f"{BASE}/api/runs/{RUN_ID}", timeout=5).json())
    except Exception as err:
        print(f"中断后查询失败: {err}")

    # 4) 续跑
    events2: list[dict] = []
    t2 = threading.Thread(target=collect_stream, args=(f"{BASE}/api/runs/{RUN_ID}/resume", events2))
    t2.start()
    t2.join(timeout=180)

    final_text = ""
    new_run_id = None
    print("\n== 续跑 SSE 事件摘要 ==")
    for e in events2:
        tp = e.get("type")
        if tp == "TEXT_MESSAGE_CONTENT":
            final_text += e.get("delta", "")
            continue
        if tp == "RUN_STARTED":
            new_run_id = e.get("runId")
        if tp in ("TEXT_MESSAGE_END", "RUN_FINISHED", "RUN_ERROR", "CLIENT_EXCEPTION"):
            print(f"{tp}: {json.dumps(e, ensure_ascii=False)[:160]}")
    print(f"\n== 续跑最终回答（前 300 字）==\n{final_text[:300]}")
    print(f"\n== 新 runId ==", new_run_id)

    # 5) 查询新 run 状态（应 done）
    if new_run_id:
        try:
            dump_state("续跑后 run-state", requests.get(f"{BASE}/api/runs/{new_run_id}", timeout=5).json())
        except Exception as err:
            print(f"续跑后查询失败: {err}")

    print("\n===== 演练结束 =====")


if __name__ == "__main__":
    main()
