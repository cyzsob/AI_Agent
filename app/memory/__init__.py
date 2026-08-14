# memory/__init__.py — 上下文记忆模块
#
# 分层设计：
#   短期记忆  → Redis（热数据，TTL 自动过期；不可用时由 history.py 降级为内存）
#   长期记忆  → PGVector memory_long 表（冷数据，异步持久化 + 语义/全文混合检索）
