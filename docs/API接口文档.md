# DeepSeek RAG Agent — API 接口文档

> 面向前端开发者的对接文档。本文档覆盖本项目暴露的全部 HTTP 端点，重点讲解 **SSE 流式聊天**（`POST /api/chat`）的协议格式、事件顺序与前端解析方式。

---

## 1. 基本信息

| 项 | 值 |
|---|---|
| 服务地址 | `http://localhost:3001`（由 `.env` 中 `PORT` 配置，默认 `3001`） |
| 协议 | HTTP / SSE（`text/event-stream`） |
| 数据格式 | 请求/响应均为 `application/json`；流式响应为 AG-UI 事件 |
| 跨域 | 已启用 CORS，`allow_origins=["*"]`，浏览器可直接调用 |
| 认证 | 无鉴权（本地服务） |

---

## 2. 端点总览

| 方法 | 路径 | 说明 | 响应类型 |
|---|---|---|---|
| `POST` | `/api/chat` | 流式对话（核心接口） | SSE 事件流 |
| `GET` | `/capabilities` | AG-UI 能力声明 | JSON |
| `POST` | `/api/chat/clear` | 清除指定会话历史 | JSON |
| `GET` | `/api/runs/{run_id}` | 查询一次运行的状态快照 | JSON |
| `POST` | `/api/runs/{run_id}/cancel` | 中断一次正在运行的对话 | JSON |
| `POST` | `/api/runs/{run_id}/resume` | 断点续跑未完成的任务 | SSE 事件流 |
| `POST` | `/api/admin/sync` | 增量同步文档到向量库 | JSON |
| `GET` | `/health` | 服务与依赖健康检查 | JSON |

---

## 3. 核心接口：`POST /api/chat`

### 3.1 请求

```
POST /api/chat
Content-Type: application/json
```

**请求体：**

```json
{
  "threadId": "会话ID，字符串，前端自行生成（如 uuid），同一会话复用同一值",
  "runId": "运行ID，字符串，前端自行生成（如 uuid），每次发送新消息生成新值",
  "messages": [
    {
      "role": "user",
      "content": "用户消息内容"
    }
  ]
}
```

**字段说明：**

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `threadId` | string | ✅ | 会话标识。同一会话内保持不变，用于历史记忆的存取 |
| `runId` | string | ✅ | 单次运行标识。每次调用传新的 uuid，用于中断/续跑/状态查询 |
| `messages` | array | ✅ | 消息数组，取其中**最后一条 `role="user"` 消息**作为本次提问。非空数组 |

**多模态（图片）请求**：`content` 支持 OpenAI 兼容的 content 数组格式，`image_url` 支持 base64 data URI：

```json
{
  "threadId": "t-1",
  "runId": "r-100",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "这张图里有什么？" },
        { "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } }
      ]
    }
  ]
}
```

> 后端会先把图片交给视觉模型生成文字描述，再合并文本一起交给 Agent，最终答案仍以纯文本 SSE 事件返回。

### 3.2 响应格式

- `Content-Type: text/event-stream`
- 每行数据以 `data: ` 前缀开始，`\n\n` 结尾，**没有 `event:` 字段**，事件类型在 JSON 的 `type` 字段中。
- 前端需要用 `fetch` + `ReadableStream` 逐行解析（`EventSource` 只能发 GET，不适用）。

### 3.3 SSE 事件类型总表

| `type` | 含义 | 关键字段 |
|---|---|---|
| `RUN_STARTED` | 运行开始 | `threadId`, `runId` |
| `CUSTOM` | 阶段进度提示 | `name="status"`, `value.message`（拆解中/处理中/汇总中） |
| `TEXT_MESSAGE_START` | 开始输出文本 | `messageId`, `role="assistant"` |
| `TOOL_CALL_START` | 工具开始调用 | `toolCallId`, `toolCallName` |
| `TOOL_CALL_ARGS` | 工具参数增量 | `toolCallId`, `delta`（**累积的 JSON 参数字符串片段**） |
| `TOOL_CALL_RESULT` | 工具返回结果 | `messageId`, `toolCallId`, `content` |
| `TOOL_CALL_END` | 工具调用结束 | `toolCallId` |
| `TEXT_MESSAGE_CONTENT` | 回答文本增量 | `messageId`, `delta` |
| `TEXT_MESSAGE_END` | 回答输出结束 | `messageId` |
| `RUN_FINISHED` | 运行成功结束 | `threadId`, `runId`, `outcome.type="success"` |
| `RUN_ERROR` | 运行出错 | `message`, `code` |

### 3.4 典型事件顺序

```
RUN_STARTED
TEXT_MESSAGE_START
CUSTOM(name=status, message="正在拆解任务…")   ← 可能多次，随阶段切换
TOOL_CALL_START / TOOL_CALL_ARGS / TOOL_CALL_RESULT / TOOL_CALL_END   ← 可能 0~N 轮
TEXT_MESSAGE_CONTENT  × N                     ← 逐字/逐段增量
TEXT_MESSAGE_END
RUN_FINISHED          （或 RUN_ERROR 结束）
```

**事件流示例：**

```
data: {"type": "RUN_STARTED", "threadId": "t-1", "runId": "r-100"}

data: {"type": "TEXT_MESSAGE_START", "messageId": "m-1", "role": "assistant"}

data: {"type": "CUSTOM", "name": "status", "value": {"message": "正在检索/处理中…"}}

data: {"type": "TOOL_CALL_START", "toolCallId": "call_1", "toolCallName": "web_search"}

data: {"type": "TOOL_CALL_ARGS", "toolCallId": "call_1", "delta": "{\"query\": \"DeepSeek\"}"}

data: {"type": "TOOL_CALL_RESULT", "messageId": "tm-1", "toolCallId": "call_1", "content": "检索结果文本..."}

data: {"type": "TOOL_CALL_END", "toolCallId": "call_1"}

data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m-1", "delta": "DeepSeek 是"}

data: {"type": "TEXT_MESSAGE_CONTENT", "messageId": "m-1", "delta": "一家 AI 公司。"}

data: {"type": "TEXT_MESSAGE_END", "messageId": "m-1"}

data: {"type": "RUN_FINISHED", "threadId": "t-1", "runId": "r-100", "outcome": {"type": "success"}}
```

### 3.5 错误情况

| 场景 | 返回 |
|---|---|
| 请求体不是合法 JSON | HTTP 400，JSON：`{"error": "请求体不是合法 JSON", "hint": "...", "detail": "..."}` |
| 缺少 `threadId` / `runId` / `messages` 为空 | HTTP 200，SSE 流中一条 `RUN_ERROR`（`code: "INVALID_INPUT"`） |
| 运行过程出错 | HTTP 200，SSE 流中一条 `RUN_ERROR`（`code: "AGENT_ERROR"`） |

> ⚠️ 参数校验失败和运行错误都通过 **SSE 流内 `RUN_ERROR` 事件**返回（HTTP 状态仍为 200），前端必须解析到 `RUN_ERROR` 时按错误处理，不能只看 HTTP 状态码。

### 3.6 前端接入示例（TypeScript）

```typescript
interface ChatRequest {
  threadId: string;
  runId: string;
  messages: Array<{ role: string; content: unknown }>;
}

interface SSEEvent {
  type: string;
  [key: string]: unknown;
}

/** 发送消息并消费 SSE 事件流 */
export async function chatStream(
  body: ChatRequest,
  handlers: {
    onStatus?: (msg: string) => void;            // 阶段进度
    onToolCallStart?: (id: string, name: string) => void;
    onToolResult?: (content: string) => void;
    onDelta?: (delta: string) => void;           // 回答文本增量
    onDone?: () => void;
    onError?: (message: string, code: string) => void;
  },
  signal?: AbortSignal,
): Promise<void> {
  const resp = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!resp.ok || !resp.body) {
    // 仅当请求体非法 JSON 时才可能是 HTTP 400，其余错误都在流内
    const err = await resp.json().catch(() => null);
    handlers.onError?.(err?.error ?? `HTTP ${resp.status}`, "HTTP_ERROR");
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE 以空行分隔每条事件
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? ""; // 最后一段可能不完整，留到下次

    for (const block of blocks) {
      const dataLine = block
        .split("\n")
        .find((l) => l.startsWith("data: "));
      if (!dataLine) continue;

      const event = JSON.parse(dataLine.slice(6)) as SSEEvent;
      switch (event.type) {
        case "CUSTOM":
          if (event.name === "status") {
            handlers.onStatus?.((event.value as { message: string }).message);
          }
          break;
        case "TOOL_CALL_START":
          handlers.onToolCallStart?.(
            String(event.toolCallId),
            String(event.toolCallName),
          );
          break;
        case "TOOL_CALL_RESULT":
          handlers.onToolResult?.(String(event.content ?? ""));
          break;
        case "TEXT_MESSAGE_CONTENT":
          handlers.onDelta?.(String(event.delta ?? ""));
          break;
        case "RUN_FINISHED":
          handlers.onDone?.();
          break;
        case "RUN_ERROR":
          handlers.onError?.(String(event.message), String(event.code));
          return;
      }
    }
  }
}
```

**使用示例：**

```typescript
await chatStream(
  {
    threadId: "t-1",           // 会话内保持不变
    runId: crypto.randomUUID(), // 每次提问新生成
    messages: [{ role: "user", content: "你好，帮我介绍一下 DeepSeek" }],
  },
  {
    onStatus: (msg) => setStatus(msg),          // "正在拆解任务…"
    onDelta: (d) => appendAnswer(d),             // 逐段追加回答
    onDone: () => setLoading(false),
    onError: (msg) => alert(`出错了：${msg}`),
  },
  abortController.signal,                         // 支持取消
);
```

---

## 4. `GET /capabilities`

能力声明，可用于前端做功能开关。

```
GET /capabilities
```

**响应示例：**

```json
{
  "identity": {
    "name": "DeepSeek RAG Agent",
    "type": "rag",
    "description": "基于 DeepSeek 和 PGVector 的知识库问答助手",
    "version": "1.0.0"
  },
  "transport": { "streaming": true, "websocket": false, "httpBinary": false, "pushNotifications": false, "resumable": false },
  "tools": { "supported": true },
  "output": { "structuredOutput": false },
  "state": { "persistentState": true },
  "reasoning": { "supported": false },
  "multimodal": {
    "input": { "text": true, "image": true, "audio": false, "video": false, "file": false },
    "output": { "text": true, "image": false, "audio": false, "video": false, "file": false }
  }
}
```

---

## 5. `POST /api/chat/clear`

清除指定会话的短期历史与滚动摘要（下次同 `threadId` 对话将从空白开始）。

```
POST /api/chat/clear
Content-Type: application/json
```

**请求体：**

```json
{ "threadId": "t-1" }
```

**响应：**

```json
{ "success": true, "threadId": "t-1" }
```

| 场景 | 返回 |
|---|---|
| 成功 | `{"success": true, "threadId": "..."}` |
| 缺少 `threadId` | `{"success": false, "message": "缺少 threadId"}` |

---

## 6. `GET /api/runs/{run_id}`

查询一次多 Agent 运行的状态快照（任务拆解 + 各 worker 进度）。SSE 中断/刷新后可用来恢复界面状态。

```
GET /api/runs/r-100
```

**响应示例：**

```json
{
  "thread_id": "t-1",
  "original_query": "DeepSeek 的开源带来了什么生态成果",
  "status": "running",
  "tasks": [
    { "target": "web_search", "args": "...", "status": "done" },
    { "target": "doc_qa", "args": "...", "status": "running" }
  ],
  "created_at": "...",
  "updated_at": 1720000000.0
}
```

| 字段 | 说明 |
|---|---|
| `status` | `running` / `done` / `interrupted` |
| `tasks[].status` | `pending` / `running` / `done` / `failed` |

| 场景 | 返回 |
|---|---|
| 成功 | 200，状态 JSON |
| 不存在或已过期（TTL 默认 900s） | 404，`{"error": "run not found（可能已过期）"}` |

---

## 7. `POST /api/runs/{run_id}/cancel`

显式中断正在进行的对话。仅能中断**当前服务进程内**仍在运行的 SSE 任务。

```
POST /api/runs/{run_id}/cancel
```

**响应：**

```json
{ "success": true, "runId": "r-100", "message": "已发送中断请求" }
```

| 场景 | 返回 |
|---|---|
| 中断成功 | `{"success": true, ...}` |
| 该 run 不存在或已结束 | `{"success": false, "message": "该 run 不存在或已结束", "runId": "..."}` |

> 中断后，对应 SSE 流会收到 `RUN_ERROR`（`code: "AGENT_ERROR"`，message 为 "SSE 中断..."）或流直接结束，前端需自行处理"已取消"的 UI 状态。

---

## 8. `POST /api/runs/{run_id}/resume`

断点续跑：基于 run-state 中**未完成任务**（pending/running/failed）重新执行，生成**新的 runId**，返回 SSE 流。

```
POST /api/runs/{run_id}/resume
（无请求体）
```

**响应：** 与 `/api/chat` 相同的 SSE 事件流，但事件中的 `runId` 是**新生成的**。

| 场景 | 返回 |
|---|---|
| 续跑成功 | 200，SSE 事件流（新 runId） |
| run 不存在/已过期 | 404，`{"error": "run not found（可能已过期）"}` |
| run 已完成 | 400，`{"error": "该 run 已完成，无需续跑"}` |
| 没有未完成任务 | 400，`{"error": "没有未完成的任务..."}` |
| 缺少必要字段 | 400，`{"error": "run-state 缺少 thread_id / original_query"}` |

---

## 9. `POST /api/admin/sync`

增量同步 `data/documents/` 目录下的文档到向量库，并刷新检索器缓存（含 BM25）。用于知识库更新后手动触发。

```
POST /api/admin/sync
```

**响应：**

```json
{
  "success": true,
  "added": 3,
  "skipped": 1,
  "failed": 0
}
```

失败时：`{"success": false, "message": "错误信息"}`

---

## 10. `GET /health`

服务与全部依赖的健康检查。

```
GET /health
```

**响应示例：**

```json
{
  "status": "ok",
  "checks": {
    "database": "ok",
    "agent": "ok",
    "retriever_cache": "ready",
    "redis": "ok"
  }
}
```

| 字段 | 取值 |
|---|---|
| `status` | `ok`（全部正常）/ `degraded`（部分依赖异常） |
| `checks.database` | `ok` 或 `error: ...` |
| `checks.agent` | `ok` / `not_initialized` |
| `checks.retriever_cache` | `ready` / `not_built` |
| `checks.redis` | `ok` / `unavailable` / `error: ...` |

---

## 11. 前端对接注意事项

1. **threadId / runId 的生命周期**
   - `threadId`：一个会话一个，保持不变；会话级历史、短期记忆、长期记忆都以它关联。
   - `runId`：每次提问生成新的 uuid；`cancel` / `resume` / 状态查询都以它为索引。
2. **不要用 `EventSource`**：`/api/chat` 是 POST，EventSource 只支持 GET，必须用 `fetch` + 流式读取。
3. **错误在流内**：`RUN_ERROR` 事件（HTTP 200）要按错误处理；只有请求体非法 JSON 才是 HTTP 400。
4. **回答文本的拼接**：同一 `messageId` 的多个 `TEXT_MESSAGE_CONTENT.delta` 按到达顺序直接拼接即可。
5. **工具调用展示（可选）**：`TOOL_CALL_START → TOOL_CALL_ARGS（参数可能分多次发，需累积）→ TOOL_CALL_RESULT → TOOL_CALL_END`，可用于展示"正在检索/正在搜索"的过程；`TOOL_CALL_ARGS.delta` 是累积的 JSON 片段，可能需要自行 `JSON.parse`。
6. **阶段进度**：`CUSTOM`（`name="status"`）事件在回答首字之前到达，建议用于"正在处理…"的 loading 文案，避免用户看到静默。
7. **取消**：中断 SSE 可以直接 `AbortController.abort()`，或调用 `POST /api/runs/{runId}/cancel`（后者会把服务端 run-state 标记为 interrupted）。
