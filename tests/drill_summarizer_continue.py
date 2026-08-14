"""汇总阶段"原位续生成"链路演练（需要服务已启动，如端口 3002）。

验证目标（对应 routes._make_summarizer_continue_stream）：
  在 summarizer 输出最终回答的过程中中断，resume 应：
    1. 不重跑 planner/worker（续跑流中无 TOOL_CALL_* 事件、状态事件只有"正在汇总回答…"）
    2. 复用原 run 的 messageId 继续输出（TEXT_MESSAGE_CONTENT 的 messageId 与首轮一致）
    3. 最终 run-state 为 done

运行：.venv\\Scripts\\python.exe tests/drill_summarizer_continue.py [端口]
"""

import json
import sys
import threading
import time

import requests

PORT = sys.argv[1] if len(sys.argv) > 1 else "3002"
BASE = f"http://localhost:{PORT}"
THREAD_ID = f"drill-sum-{int(time.time())}"
RUN_ID = f"run-sum-{int(time.time())}"

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


def wait_until_partial_text(events: list, min_chars: int = 30, timeout: float = 180.0) -> int:
    """轮询首轮 SSE 事件，直到累计 TEXT_MESSAGE_CONTENT 正文达到 min_chars。

    partial_text 只在中断时才写入 run-state，正常流式期间查不到，因此只能
    从已收集的 SSE 事件里判断 summarizer 是否已在输出。返回已累计字数。
    """
    deadline = time.time() + timeout
    seen_start = False
    while time.time() < deadline:
        total = 0
        for e in events:
            if e.get("type") == "RUN_STARTED":
                seen_start = True
            if e.get("type") == "TEXT_MESSAGE_CONTENT":
                total += len(e.get("delta", ""))
            if e.get("type") == "RUN_FINISHED":
                return -1  # 已跑完，无法中断（测试无效）
        if seen_start and total >= min_chars:
            return total
        time.sleep(0.3)
    return total


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

    # 1) 轮询首轮事件，直到 summarizer 已输出部分正文，再中断
    total_chars = wait_until_partial_text(events)
    print(f"== 轮询结果：首轮已流式输出 {total_chars} 字 ==")
    if total_chars < 0:
        print("!! run 已 RUN_FINISHED，无法中断（测试无效）")
    elif total_chars < 30:
        print("!! 未在超时时间内等到 summarizer 输出正文（可能出错），测试无效")
    else:
        text_first = "".join(
            e.get("delta", "") for e in events if e.get("type") == "TEXT_MESSAGE_CONTENT"
        )
        print(f"中断前残句: {text_first[:80]!r}...")

    # 原 SSE 流的 TEXT_MESSAGE_START messageId（续跑应复用）
    orig_msg_id = None
    for e in events:
        if e.get("type") == "TEXT_MESSAGE_START":
            orig_msg_id = e.get("messageId")
    print("\n== 首轮 messageId ==", orig_msg_id)

    # 2) 显式中断
    cancel = requests.post(f"{BASE}/api/runs/{RUN_ID}/cancel", timeout=5).json()
    print("\n== cancel ==", cancel)

    t.join(timeout=30)

    # 3) 查询中断现场：应有 partial_text + message_id + result:*
    try:
        st = requests.get(f"{BASE}/api/runs/{RUN_ID}", timeout=5).json()
        print("\n== 中断后 run-state 关键字段 ==")
        meta = {k: v for k, v in st.items() if k != "tasks" and not k.startswith("result:")}
        meta["partial_text_len"] = len(st.get("partial_text", ""))
        meta["partial_text_head"] = st.get("partial_text", "")[:60]
        print(json.dumps(meta, ensure_ascii=False, default=str)[:500])
        print("result:* 字段:", [k for k in st if k.startswith("result:")])
    except Exception as err:
        print(f"中断后查询失败: {err}")

    # 4) 续跑：应无工具调用、复用同一 messageId、继续输出
    events2: list[dict] = []
    t2 = threading.Thread(target=collect_stream, args=(f"{BASE}/api/runs/{RUN_ID}/resume", events2))
    t2.start()
    t2.join(timeout=180)

    final_text = ""
    resume_msg_id = None
    tool_events = []
    statuses = []
    new_run_id = None
    finished = False
    print("\n== 续跑 SSE 事件摘要 ==")
    for e in events2:
        tp = e.get("type")
        if tp == "TEXT_MESSAGE_CONTENT":
            resume_msg_id = e.get("messageId", resume_msg_id)
            final_text += e.get("delta", "")
        elif tp == "TEXT_MESSAGE_START":
            resume_msg_id = e.get("messageId", resume_msg_id)
        elif tp in ("TOOL_CALL_START", "TOOL_CALL_RESULT", "TOOL_CALL_END"):
            tool_events.append((tp, e.get("toolCallName") or e.get("toolCallId") or ""))
        elif tp == "CUSTOM":
            statuses.append(e.get("value", {}).get("message"))
        elif tp == "RUN_STARTED":
            new_run_id = e.get("runId")
        elif tp == "RUN_FINISHED":
            finished = True
        if tp in ("TEXT_MESSAGE_END", "RUN_FINISHED", "RUN_ERROR", "CLIENT_EXCEPTION"):
            print(f"{tp}: {json.dumps(e, ensure_ascii=False)[:160]}")

    print("续跑状态事件:", statuses)
    print("续跑中的工具调用事件:", tool_events if tool_events else "（无 —— 未重跑 worker，符合预期）")
    print(f"续跑 messageId == 首轮 messageId: {resume_msg_id == orig_msg_id} "
          f"({resume_msg_id} vs {orig_msg_id})")
    print(f"续跑新增 {len(final_text)} 字: {final_text[:120]!r}...")
    print(f"RUN_FINISHED: {finished}, 新 runId: {new_run_id}")

    # 5) 查询新 run 状态（应 done）
    if new_run_id:
        try:
            st2 = requests.get(f"{BASE}/api/runs/{new_run_id}", timeout=5).json()
            print("\n== 续跑后 run-state ==")
            print("status:", st2.get("status"),
                  "| partial_text_len:", len(st2.get("partial_text", "")),
                  "| result:*:", [k for k in st2 if k.startswith("result:")])
        except Exception as err:
            print(f"续跑后查询失败: {err}")

    print("\n===== 演练结束 =====")


if __name__ == "__main__":
    main()
