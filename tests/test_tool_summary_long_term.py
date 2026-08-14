"""验证：工具结果摘要只进短期历史、不进长期记忆（修复回归测试）。

场景（两轮，模拟真实对话）：
  turn1: 用户请求触发工具调用（gitee 仓库列表）→
         assistant 最终回答 + 【上轮工具结果摘要】标记消息
  turn2: 后续指代问题（"其中有多少个项目和鸿蒙有关"）

验证点：
  1. 单元：_long_term_messages 过滤逻辑（必然执行，无外部依赖）
  2. 集成：persist_round(过滤后) → memory_long 入库内容不含工具数据
           （依赖 PG + Ollama；服务不可达自动 SKIP）
  3. 集成：append_messages 短期历史保留工具摘要，供后续轮次引用
           （依赖 Redis；不可达自动 SKIP）

运行（项目根目录下）：
  python tests/test_tool_summary_long_term.py
"""

import asyncio
import pathlib
import sys
import uuid

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api.routes import _TOOL_SUMMARY_PREFIX, _long_term_messages


# ========== 测试数据 ==========

_TOOL_DATA = (
    "[gitee_list_user_repos] {\"repos\": ["
    "{\"name\": \"common_api\", \"desc\": \"鸿蒙常用的移动设备能力\"}, "
    "{\"name\": \"Napi_ArkTs\", \"desc\": \"\"}, "
    "{\"name\": \"Flutter_HomeManager\", \"desc\": \"\"}]}"
)


def build_turn_messages() -> list[dict]:
    """按 routes.py 保存历史的构造方式生成一轮含工具调用的消息。"""
    return [
        {"role": "user", "content": "帮我查询gitee仓库的项目列表"},
        {
            "role": "assistant",
            "content": "共查询到3个仓库，其中 common_api 为鸿蒙常用、Napi_ArkTs 与 ArkTS 相关。",
        },
        {
            "role": "assistant",
            "content": f"{_TOOL_SUMMARY_PREFIX}\n{_TOOL_DATA}",
        },
    ]


# ========== 1. 单元：过滤逻辑（无外部依赖） ==========


def test_long_term_messages_filters_tool_summary() -> None:
    turn_messages = build_turn_messages()
    filtered = _long_term_messages(turn_messages)

    assert len(filtered) == 2, f"应过滤掉工具摘要消息，实际 {len(filtered)} 条"
    assert all(
        not str(m.get("content", "")).startswith(_TOOL_SUMMARY_PREFIX)
        for m in filtered
    ), "过滤结果中不应含工具摘要标记消息"
    assert [m["role"] for m in filtered] == ["user", "assistant"]
    assert filtered[0] == turn_messages[0]
    assert filtered[1] == turn_messages[1]
    print("PASS  单元: 工具摘要被过滤，仅保留 user + assistant 最终回答")


def test_long_term_messages_keeps_normal_assistant() -> None:
    normal = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮你？"},
    ]
    assert _long_term_messages(normal) == normal, "普通 user/assistant 消息不应被误删"
    print("PASS  单元: 普通消息不被误删")


# ========== 2. 集成：长期记忆真实入库（PG + Ollama，不可达则 SKIP） ==========


async def verify_long_term_clean(thread_id: str) -> None:
    try:
        import asyncpg

        from app.memory import long_term
        from app.memory import summarizer as summarizer_mod
    except Exception as err:
        print(f"SKIP  集成: 依赖导入失败 ({err})")
        return

    # 捕获 persist_round 实际收到的输入；替换 summarize_turn 为固定文本，
    # 使入库链路确定性（不依赖 DeepSeek 摘要结果），且仍走真实 DB 写入。
    captured: dict = {}
    original = summarizer_mod.summarize_turn

    async def fake_summarize_turn(messages):
        captured["messages"] = messages
        return "用户询问 Gitee 仓库列表，共 3 个仓库，其中 common_api 与鸿蒙相关。"

    summarizer_mod.summarize_turn = fake_summarize_turn
    try:
        filtered = _long_term_messages(build_turn_messages())
        await long_term.persist_round(thread_id, filtered)

        # 2.1 persist_round 收到的输入不含工具数据
        received = captured.get("messages", [])
        contents = [str(m.get("content", "")) for m in received]
        assert not any(c.startswith(_TOOL_SUMMARY_PREFIX) for c in contents), \
            "persist_round 不应收到工具摘要消息"
        assert not any("gitee_list_user_repos" in c for c in contents), \
            "persist_round 不应收到工具名/原始数据"
        print(f"PASS  集成: persist_round 收到 {len(received)} 条消息（仅 user+assistant）")

        # 2.2 memory_long 实际入库内容干净
        try:
            conn = await asyncpg.connect(**long_term._db_config())
        except Exception as err:
            print(f"SKIP  集成: PostgreSQL 不可达 ({err})")
            return
        try:
            rows = await conn.fetch(
                "SELECT content FROM memory_long WHERE thread_id=$1 ORDER BY id DESC",
                thread_id,
            )
        finally:
            await conn.close()

        if not rows:
            print("WARN  集成: memory_long 无新增记录（Ollama 嵌入不可用？），跳过 DB 断言")
            return
        for row in rows:
            content = str(row["content"])
            # 泄漏特征：工具摘要标记前缀、工具名（原始 JSON 的签名）
            assert _TOOL_SUMMARY_PREFIX not in content, f"长期记忆含工具摘要标记: {content[:80]}"
            assert "gitee_list_user_repos" not in content, f"长期记忆含工具名/原始数据: {content[:80]}"
        print(f"PASS  集成: memory_long 入库 {len(rows)} 条，均不含工具摘要/工具数据")
    finally:
        summarizer_mod.summarize_turn = original


# ========== 3. 集成：短期历史保留工具摘要（Redis，不可达则 SKIP） ==========


async def verify_short_term_keeps(thread_id: str) -> None:
    try:
        from app.agent.history import append_messages, get_or_create_history
        from app.memory import redis_store
    except Exception as err:
        print(f"SKIP  集成: 依赖导入失败 ({err})")
        return

    try:
        await append_messages(thread_id, build_turn_messages())
        history = await get_or_create_history(thread_id)
    except Exception as err:
        print(f"SKIP  集成: Redis 不可达 ({err})")
        return
    finally:
        try:
            await redis_store.delete_thread(thread_id)
        except Exception:
            pass

    assert len(history) == 3, f"短期历史应含 3 条（含工具摘要），实际 {len(history)} 条"
    assert any(
        str(m.get("content", "")).startswith(_TOOL_SUMMARY_PREFIX) for m in history
    ), "短期历史应保留工具摘要消息（供后续轮次引用）"
    print("PASS  集成: 短期历史保留工具摘要（后续指代问题可引用）")


# ========== 清理 ==========


async def cleanup(thread_id: str) -> None:
    try:
        import asyncpg

        from app.memory import long_term

        conn = await asyncpg.connect(**long_term._db_config())
        try:
            await conn.execute("DELETE FROM memory_long WHERE thread_id=$1", thread_id)
        finally:
            await conn.close()
    except Exception:
        pass


def main() -> None:
    print("=" * 60)
    print("验证: 工具结果摘要只进短期历史、不进长期记忆")
    print("=" * 60)

    failures = 0

    def run(name: str, fn) -> None:
        nonlocal failures
        try:
            fn()
        except Exception as err:
            failures += 1
            print(f"FAIL  {name}: {err}")

    # 单元（无依赖，必然执行）
    run("单元: 过滤工具摘要", test_long_term_messages_filters_tool_summary)
    run("单元: 不误删普通消息", test_long_term_messages_keeps_normal_assistant)

    # 集成（依赖外部服务，按环境可达性执行）
    thread_id = f"test-tool-summary-{uuid.uuid4().hex[:8]}"
    try:
        run("集成: 长期记忆入库", lambda: asyncio.run(verify_long_term_clean(thread_id)))
        run("集成: 短期历史保留", lambda: asyncio.run(verify_short_term_keeps(thread_id)))
    finally:
        asyncio.run(cleanup(thread_id))
        print(f"已清理测试数据: {thread_id}")

    print("=" * 60)
    if failures:
        print(f"存在失败 ({failures} 项)")
        sys.exit(1)
    print("全部通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
