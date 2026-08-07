# DeepSeek RAG Agent — 多 Agent 协同知识库问答系统（支持 MCP 外部工具）

基于 **DeepSeek V4 Flash + LangGraph（Supervisor 多 Agent 模式）+ PGVector + BM25** 的混合检索 RAG 系统，通过 **AG-UI 协议** 提供流式服务，并通过 **MCP（Model Context Protocol）** 接入外部工具。

---

## 目录

- [项目结构](#项目结构)
- [多 Agent 协同架构](#多-agent-协同架构supervisor-监督者模式)
- [MCP 外部工具集成](#mcp-外部工具集成model-context-protocol)
- [回答问题的完整流程](#回答问题的完整流程)
- [各模块说明](#各模块说明)
- [启动方式](#启动方式)
- [MCP 配置指南](#mcp-配置指南)

---

## 项目结构

```
├── app/                        # Python 源码（主程序与各功能模块）
│   ├── server_agui.py          # AG-UI 协议 HTTP+SSE 服务器（主入口）
│   ├── agent.py                # 多 Agent 入口（加载工具、按域分组、构建 Supervisor 图）
│   ├── multi_agent.py          # 多 Agent 协同（Supervisor 监督者模式）
│   ├── tools.py                # 本地 Agent 工具定义（get_deepseek_info / search / weather）
│   ├── conversation_history.py # 多轮对话历史管理器（基于 threadId 的内存存储）
│   ├── mcp_client.py           # MCP 客户端连接管理器（核心）
│   ├── hybrid_retriever.py     # 混合检索器（向量 + BM25 + RRF 融合）
│   ├── reranker.py             # Cross-Encoder 重排器（sentence-transformers）
│   ├── query.py                # 命令行 RAG 查询脚本（改写 -> 检索 -> 生成）
│   ├── ingest.py               # 数据入库脚本（扫描目录 + 解析 TXT/PDF/DOCX + 分块 + 向量化 + 写入 PG）
│   ├── parse_pdf.py            # PDF 混合解析（PyMuPDF 文本提取 + 页面渲染 + RapidOCR 识别）
│   └── diagnose.py             # 诊断脚本（对比三种检索方式效果）
├── config/
│   └── mcp-servers.config.json # MCP 服务器配置文件
├── data/
│   └── documents/              # 知识库源文档（TXT / DOCX / PDF）
├── docs/                       # 设计与个人资料文档（关键功能实现 / 面试题等）
├── logs/                       # 运行日志（error.txt / server_test.log）
├── mcp-servers/                # 本地 MCP Server 示例（calculator，独立子项目）
├── .env                        # 环境变量（DeepSeek API Key + PG 数据库连接）
├── pyproject.toml              # 项目依赖管理
└── README.md
```

---

## 多 Agent 协同架构（Supervisor 监督者模式）

系统采用 **Supervisor 监督者模式**：一个协调 Agent（supervisor）负责把用户请求路由给最合适的专业 worker Agent，每个 worker 只持有本领域的工具子集与专业提示词，完成后将控制权交回 supervisor。

```
用户消息
  │
  ▼
┌────────────────────────────────────────────────────────┐
│  supervisor（协调 Agent，只负责路由）                     │
│    TransferToKnowledge / TransferToDevops /            │
│    TransferToGeneral / finish_workflow                 │
└──────────────┬──────────────────┬──────────────────────┘
               │                  │
               ▼                  ▼
   ┌────────────┐   ┌────────────┐   ┌──────────────┐
   │ knowledge  │   │  devops    │   │  general     │
   │ 知识/检索   │   │ Gitee 操作  │   │ 天气 + 日常   │
   │ RAG 工具   │   │ MCP 工具    │   │ get_weather  │
   └──────┬─────┘   └──────┬─────┘   └──────┬───────┘
          └───────┬────────┘───────┬────────┘
                  ▼                ▼
             supervisor（再路由 / 结束 → END）
```

| Worker | 工具 | 职责 |
|--------|------|------|
| `knowledge` | `get_deepseek_info`、`search_knowledge_base` | DeepSeek 公司介绍与知识库技术细节问答 |
| `devops` | `gitee_*`（MCP 动态发现） | Gitee 平台操作（仓库 / Issue / PR / 通知等） |
| `general` | `get_weather` | 实时天气查询与日常对话 |

**设计要点**：
- supervisor 只输出路由工具调用（不输出文本、不自行回答），其内部路由事件不透出给客户端、不进入对话历史
- 每个 worker 是独立 ReAct Agent，提示词可按领域深度定制，工具集精简后误调概率更低
- 未配置 MCP Server 时 `devops` 节点自动跳过，supervisor 成员列表同步剔除
- 多轮对话历史只保存 worker 的交互，下轮由 supervisor 基于完整历史重新路由，无需持久化内部状态

---

## MCP 外部工具集成（Model Context Protocol）

本系统支持通过 **MCP（Model Context Protocol）** 标准协议接入第三方外部工具。MCP 是一种开放的协议，允许 LLM Agent 以统一的方式发现和调用外部工具，无论这些工具的源码托管在何处（GitHub、Gitee、私有仓库等）。

### 架构概览

```
┌────────────────────────────────────────────────────────────┐
│                    agent.py                                 │
│    create_react_agent({ model, tools: [...本地工具, ...MCP 工具] })   │
└────────────┬───────────────────────────────────────────────┘
             │ 调用工具
             ▼
┌──────────────────────┐     ┌──────────────────────────────┐
│    tools.py          │     │    mcp_client.py              │
│                      │     │                              │
│ get_deepseek_info  ◄─┤     │ Client.connect(transport)    │
│ search_knowledge   ◄─┤     │ client.listTools()           │
│ get_weather        ◄─┤     │ client.callTool()            │
└──────────────────────┘     └──────────┬───────────────────┘
                                        │ stdio JSON-RPC 2.0
                                        ▼
                       ┌─────────────────────────────────────┐
                       │   子进程 MCP Server                  │
                       │                                     │
                       │ npx @gitee/mcp-gitee                │
                       │ npx @modelcontextprotocol/server-*  │
                       │ python -m mcp_server_time           │
                       │ (任何 MCP 协议兼容的可执行程序)       │
                       └─────────────────────────────────────┘
```

### 工作原理

| 步骤 | 说明 |
|------|------|
| ① 配置 | 在 [`mcp-servers.config.json`](file:///d:/AiAgent/my-rag-demo/config/mcp-servers.config.json) 中声明 MCP Server 的启动命令和参数 |
| ② 发现 | 服务启动时，[`mcp_client.py`](file:///d:/AiAgent/my-rag-demo/app/mcp_client.py) 连接到每个 MCP Server 并通过 `client.listTools()` 发现其暴露的所有工具 |
| ③ 包装 | 将每个 MCP Tool 自动包装为 LangChain `StructuredTool`，加前缀（如 `gitee_`）防命名冲突 |
| ④ 注册 | 合并到 Agent 的 `tools` 数组中，动态构建系统提示词，让 LLM 知道所有可用工具 |
| ⑤ 调用 | Agent 在 ReAct 循环中决定调用 MCP 工具时，通过 `client.callTool()` 经 stdio 发送 JSON-RPC 请求 |
| ⑥ 返回 | MCP Server 返回统一格式的 `{ content: [{ type, text }] }`，Agent 整合生成最终回答 |

### 接入方式

MCP 工具不绑定平台。任何遵循 MCP 协议的可执行程序都可以接入：

| 来源 | 示例 | 启动方式 |
|------|------|----------|
| npm 包（官方） | `@modelcontextprotocol/server-github` | `npx -y @modelcontextprotocol/server-github` |
| npm 包（Gitee 官方） | `@gitee/mcp-gitee` | `npx -y @gitee/mcp-gitee@latest` |
| Python 包 | `mcp-server-time` | `python -m mcp_server_time --local-timezone=Asia/Shanghai` |
| Docker 镜像 | `mcp/some-server` | `docker run -i --rm mcp/some-server` |

### 当前已接入的 MCP 工具

| 前缀 | 服务器 | 工具 | 功能 | 来源 |
|------|--------|------|------|------|
| `gitee_` | gitee | 启动时动态发现（`listTools`） | 仓库 / Issue / PR 等 Gitee 操作 | `@gitee/mcp-gitee`（npm 包） |

---

## 回答问题的完整流程

### 一、总览

```
客户端                     服务器                       Agent                       检索层                     数据库            MCP Server
  │                        │                           │                            │                        │               │
  │──── POST /api/chat ────▶│                           │                            │                        │               │
  │     (AG-UI 协议)       │                           │                            │                        │               │
  │                        │── RUN_STARTED ────────────▶│                            │                        │               │
  │                        │    (SSE 事件)              │                            │                        │               │
  │                        │                           │                            │                        │               │
  │                        │── 加载 threadId 历史 ──────│                            │                        │               │
  │                        │    (conversation-history)│                            │                        │               │
  │                        │                           │                            │                        │               │
  │                        │──── agent.astream() ─────▶│                            │                        │               │
  │                        │   (历史 + 新消息)          │                            │                        │               │
  │                        │                           │── 分析问题, 决定调用工具     │                        │               │
  │                        │                           │                            │                        │               │
  │                        │                           │── (可选) get_deepseek_info──▶                        │               │
  │                        │                           │    (读取本地文件)            │                        │               │
  │                        │                           │                            │                        │               │
  │                        │                           │── (可选) search_knowledge   │                        │               │
  │                        │                           │     _base(query) ──────────▶│                        │               │
  │                        │                           │                            │                        │               │
  │                        │                           │                            │── 向量检索 ────────────▶│               │
  │                        │                           │                            │  (PGVector, bge-m3)   │ PostgreSQL    │
  │                        │                           │                            │                        │  documents 表 │
  │                        │                           │                            │── BM25 检索 ◀──────────│               │
  │                        │                           │                            │  (rank_bm25, 单字分词)│               │
  │                        │                           │                            │                        │               │
  │                        │                           │── (可选) MCP 工具调用 ──────│                        │               │
│                        │                           │                            │                        │               │
│                        │                           │── gitee_* (MCP 工具) ───────▶──────────────────────────────────────▶│
│                        │                           │    (SSE: TOOL_CALL_START)  │                        │               │
│                        │                           │                            │                        │               │── stdio
│                        │                           │◀──── 返回工具结果 ──────────│                        │◀──────────────│  JSON-RPC
  │                        │                           │    (SSE: TOOL_CALL_RESULT) │                        │               │
  │                        │                           │                            │                        │               │
  │                        │                           │── LLM 整合信息, 生成回答    │                        │               │
  │                        │◀── 逐 token SSE 流 ───────│                            │                        │               │
  │                        │   (TEXT_MESSAGE_CONTENT)  │                            │                        │               │
  │                        │                           │                            │                        │               │
  │◀── 流式渲染回答 ──────│                           │                            │                        │               │
```

### 二、逐步骤详解

#### 第 1 步：客户端发起请求

客户端通过 **AG-UI 协议** 发送 `POST /api/chat` 请求，Body 格式为：

```json
{
  "threadId": "会话ID",
  "runId": "本次运行ID",
  "messages": [{ "role": "user", "content": "DeepSeek-V4 的参数规模是多少？" }]
}
```

服务端接收位置：[server_agui.py#L50-L73](file:///d:/AiAgent/my-rag-demo/app/server_agui.py#L50-L73)

#### 第 2 步：服务器建立 SSE 连接

[server_agui.py#L75-L78](file:///d:/AiAgent/my-rag-demo/app/server_agui.py#L75-L78) — 设置 `text/event-stream` 响应头，立即发送 `RUN_STARTED` 事件，告知客户端准备接收流式数据。

#### 第 3 步：提取用户消息 + 加载对话历史

[server_agui.py#L86-L113](file:///d:/AiAgent/my-rag-demo/app/server_agui.py#L86-L113)

1. 从 `messages` 数组中从后往前查找最后一条 `role === "user"` 的消息内容
2. 以 `threadId` 为键，从 [`conversation_history.py`](file:///d:/AiAgent/my-rag-demo/conversation_history.py) 中**加载该会话的历史记录**
3. 构造完整消息数组：`[...历史消息, 新user消息]`，传递给 Agent

如果是首次对话，历史为空；多轮对话时 Agent 能看到前几轮的对话内容、工具调用和回答。

#### 第 4 步：Agent 执行 —— ReAct 推理循环

[server_agui.py#L128-L132](file:///d:/AiAgent/my-rag-demo/app/server_agui.py#L128-L132) — 调用 `agent.astream({ messages }, stream_mode="messages")`（传入包含历史记录的完整消息数组），启动 LangGraph 的 `create_react_agent`。

Agent 的核心逻辑在 [agent.py](file:///d:/AiAgent/my-rag-demo/agent.py)：

- **LLM**：DeepSeek V4 Flash（`temperature=0`，精确模式）
- **工具来源**：本地工具（`tools.py`）+ MCP 外部工具（`mcp_client.py`）自动合并
- **系统提示词**：动态构建，列出所有可用工具的名称和描述

**ReAct 循环的工作方式**（可能迭代 1~N 轮）：

| 步骤 | 动作 | 说明 |
| ---- | ---- | ---- |
| ① | LLM 分析问题 | 判断问题类型：公司级 / 技术细节 / 计算 / 简单问题 |
| ② | 决定调用工具 | 如需信息，输出 `tool_call`（工具名 + 参数），可调用本地工具或 MCP 工具 |
| ③ | 执行工具 | 系统执行工具函数（本地函数或 MCP JSON-RPC 调用），返回结果 |
| ④ | LLM 整合回答 | 把工具返回的结果 + 自身知识整合成最终答案 |

简单问题可直接回答，无需调用工具。

#### 第 5 步：工具调用（可选）

**本地工具**定义在 [tools.py#L21-L75](file:///d:/AiAgent/my-rag-demo/app/tools.py#L21-L75)：

| 工具 | 输入 | 功能 |
| ---- | ---- | ---- |
| `get_deepseek_info` | 无参 | 读取 `data/documents/deepseek介绍手册.txt` 全文 |
| `search_knowledge_base` | `{ query: string }` | 混合检索知识库 |
| `get_weather` | `{ city: string }` | 查询指定城市实时天气 |

**MCP 外部工具**由 [mcp_client.py](file:///d:/AiAgent/my-rag-demo/app/mcp_client.py) 在启动时动态发现并注册，自动出现在 Agent 的工具列表中：

| 工具（带前缀） | 参数 | 功能 |
| -------------- | ---- | ---- |
| `gitee_*`（动态发现） | 视具体工具而定 | Gitee 仓库 / Issue / PR 等操作 |

工具调用事件会通过 SSE 实时推送给客户端，事件序列：

```
TOOL_CALL_START  →  TOOL_CALL_ARGS  →  TOOL_CALL_RESULT  →  TOOL_CALL_END
```

#### 第 6 步：混合检索（核心）

[hybrid_retriever.py#L88-L114](file:///d:/AiAgent/my-rag-demo/app/hybrid_retriever.py#L88-L114) — `HybridRetriever._getRelevantDocuments(query)` 并行执行两路检索：

```
                        ┌──────────────────┐
         query ────────▶│    HybridRetriever │
                        │                    │
            ┌───────────┤  并行执行两路检索   ├───────────┐
            │           │                    │           │
            ▼           └────────────────────┘           ▼
   ╔═══════════════════════╗          ╔════════════════════════╗
   ║   向量检索 (语义)      ║          ║   BM25 检索 (关键词)    ║
   ║                       ║          ║                        ║
   ║ PGVectorStore         ║          ║ rank_bm25 内存索引      ║
   ║ bge-m3 嵌入模型       ║          ║ 单字分词（中文逐字）     ║
   ║ similarity_search     ║          ║ BM25 评分               ║
   ║ top-6                 ║          ║ top-6                   ║
   ╚══════════════╤════════╝          ╚════════════╤═══════════╝
                  │                                │
                  └──────────────┬─────────────────┘
                                 │
                                 ▼
                    ╔═════════════════════════╗
                    ║   RRF 融合              ║
                    ║                         ║
                    ║   score = Σ w / (k + r) ║
                    ║   向量权重 0.5           ║
                    ║   BM25 权重 0.5         ║
                    ║   最终返回 top-3        ║
                    ╚════════════╤════════════╝
                                 │
                                 ▼
                     返回 3 个文档片段
```

**RRF（Reciprocal Rank Fusion）融合公式** [`hybrid_retriever.py#L57-L84`](file:///d:/AiAgent/my-rag-demo/app/hybrid_retriever.py#L57-L84)：

```
score(doc) = vectorWeight / (k + rank_vector) + bm25Weight / (k + rank_bm25)
```

- 默认 `k=60`，`vectorWeight=0.5`，`bm25Weight=0.5`
- 一个文档同时在两路检索中出现时，融合得分会叠加，排名更高
- 向量检索擅长语义匹配，BM25 擅长关键词精确匹配，RRF 融合取两者之长

#### 第 7 步：LLM 生成最终回答

Agent 拿到工具返回的信息后（或无需工具直接回答），由 DeepSeek V4 Flash 整合信息，生成自然语言回答。

#### 第 8 步：SSE 流式推送回客户端

[server_agui.py#L134-L264](file:///d:/AiAgent/my-rag-demo/app/server_agui.py#L134-L264) — 服务器将 Agent 的逐 token 输出映射为 AG-UI 事件流：

| SSE 事件 | 触发时机 | 作用 |
| ---------------------- | ------------------- | ------------------------------ |
| `RUN_STARTED` | 请求开始 | 告知客户端开始处理 |
| `TOOL_CALL_START` | Agent 决定调用工具 | 告知客户端"即将调用某个工具" |
| `TOOL_CALL_ARGS` | 工具参数逐步生成 | 推送完整的工具调用参数（JSON） |
| `TEXT_MESSAGE_START` | 首个文本 token 到达 | 标记一段文本回复的开始 |
| `TEXT_MESSAGE_CONTENT` | 每生成一个 token | 流式推送文本片段（每次增量） |
| `TOOL_CALL_RESULT` | 工具执行完毕 | 推送工具返回的结果内容 |
| `TOOL_CALL_END` | 工具生命周期结束 | 标记工具调用已完成 |
| `TEXT_MESSAGE_END` | 文本回复结束 | 标记文本回复完成 |
| `RUN_FINISHED` | 全部处理结束 | 告知客户端处理成功结束 |
| `RUN_ERROR` | 发生错误 | 携带错误信息 |

**多轮对话的事件流差异**：第 2 轮及之后的请求，Agent 的输入中会包含前几轮的对话历史（用户消息 + AI 回答 + 工具调用记录），因此 Agent 能够理解指代（如"它"、"那个"）并支持追问。

运行结束后，服务器会自动将本轮产生的消息保存到 [`conversation_history.py`](file:///d:/AiAgent/my-rag-demo/conversation_history.py) 中，供下一轮使用。

---

### 三、完整路径示例

> **用户问**："DeepSeek-V4 采用了什么架构？"

```
① 客户端 → POST /api/chat { messages: [{ role: "user", content: "..." }] }
② 服务器 → 发送 RUN_STARTED → 启动 agent.astream()
③ Agent  → LLM 分析问题，判断属于"技术细节" → 决定调用 search_knowledge_base
④ 服务器 → 发送 TOOL_CALL_START / TOOL_CALL_ARGS
⑤ 工具   → search_knowledge_base("DeepSeek-V4 架构")
         → 混合检索：向量检索(PGVector) + BM25(rank_bm25) → RRF 融合 → top-3 片段
⑥ 服务器 → 发送 TOOL_CALL_RESULT / TOOL_CALL_END
⑦ Agent  → LLM 整合检索结果，生成回答
⑧ 服务器 → 逐 token 发送 TEXT_MESSAGE_START → CONTENT → END
⑨ 服务器 → 发送 RUN_FINISHED
⑩ 客户端 → 流式渲染完整回答
```

> **用户问**："帮我在 Gitee 上搜索 changelink 的开源项目"

```
① 客户端 → POST /api/chat { messages: [{ role: "user", content: "..." }] }
② 服务器 → 发送 RUN_STARTED → 启动 agent.astream()
③ Agent  → LLM 分析问题，判断需要外部工具 → 决定调用 gitee 搜索工具
④ 服务器 → 发送 TOOL_CALL_START (toolCallName: "gitee_*")
⑤ mcp_client.py → 通过 stdio 发送 JSON-RPC 请求给子进程 MCP Server
         → MCP Server 执行搜索 → 返回仓库列表
⑥ 服务器 → 发送 TOOL_CALL_RESULT / TOOL_CALL_END
⑦ Agent  → LLM 整合搜索结果，生成回答
⑧ 服务器 → 逐 token 发送 TEXT_MESSAGE_START → CONTENT → END
⑨ 服务器 → 发送 RUN_FINISHED
⑩ 客户端 → 流式渲染最终回答
```

---

### 四、关键设计要点

1. **Agent 自主决策** — Agent 根据问题内容自主决定调用哪个工具（本地或 MCP）、是否多轮调用，无需预定义路由
2. **动态工具发现** — MCP 工具在启动时通过 `client.listTools()` 自动发现，新增工具只需修改配置文件，无需改代码
3. **混合同步检索** — 向量检索（语义匹配）和 BM25 检索（关键词精确匹配）并行执行，RRF 融合取长补短
4. **流式输出** — 所有文本和工具调用事件都通过 SSE 实时推送，客户端可逐 token 渲染
5. **本地嵌入** — 向量化由本地 Ollama 服务上的 `bge-m3` 模型完成，不依赖外部 API
6. **MCP 协议标准化** — 所有外部工具通过统一协议接入，无论语言（Node.js/Python/Go）或托管平台（GitHub/Gitee）
7. **多轮对话支持** — 基于 `threadId` 的会话历史管理，Agent 可记忆多轮上下文，支持追问和指代消解

---

## 各模块说明

### server_agui.py — AG-UI 协议服务器

FastAPI HTTP+SSE 服务器，端口 3001：

| 端点 | 说明 |
| ------------------- | ------------------------------------------- |
| `GET /capabilities` | 声明 AG-UI 协议能力 |
| `POST /api/chat` | 核心聊天端点，接收 AG-UI 请求，SSE 流式响应 |
| `POST /api/chat/clear` | 清除指定 `threadId` 的会话历史（用于重置对话） |

### agent.py — 多 Agent 入口（Supervisor 模式）

入口脚本：加载本地 + MCP 工具，按工具域分组，调用 [multi_agent.py](file:///d:/AiAgent/my-rag-demo/multi_agent.py) 的 `build_multi_agent()` 构建 Supervisor 多 Agent 图。底层 LLM 为 DeepSeek（`deepseek-chat`）。

Agent 启动时执行：
1. 加载本地工具（[tools.py](file:///d:/AiAgent/my-rag-demo/app/tools.py)）
2. 通过 [mcp_client.py](file:///d:/AiAgent/my-rag-demo/app/mcp_client.py) 从 [mcp-servers.config.json](file:///d:/AiAgent/my-rag-demo/config/mcp-servers.config.json) 加载所有 MCP 外部工具
3. 按工具域分组：knowledge（RAG 工具）/ devops（`gitee_*` MCP 工具）/ general（天气 + 日常）
4. 构建 Supervisor 图：supervisor 负责路由，各 worker 为独立 ReAct Agent

### multi_agent.py — 多 Agent 协同（Supervisor 监督者模式）

| 组成部分 | 说明 |
|----------|------|
| `MultiAgentState` | 图共享状态（`messages`，`add_messages` 自动合并） |
| `_make_transfer_tools` | supervisor 的路由工具（`TransferToXxx` / `finish_workflow`），仅用于触发条件边 |
| `_make_supervisor_node` | supervisor 节点：绑定路由工具，只输出工具调用、不输出文本 |
| `_make_route_after_supervisor` | 条件边：根据 supervisor 的 tool_call 名称路由到 worker 或 END |
| `build_multi_agent` | 主入口：装配 supervisor + 各 worker（无工具的域自动跳过） |

### conversation_history.py — 多轮对话历史管理器

管理基于 `threadId` 的会话历史，支持多轮对话上下文记忆：

| 函数 | 说明 |
| ---- | ---- |
| `get_or_create_history(threadId)` | 获取或创建会话历史（`MessageRecord[]`） |
| `to_langchain_messages(records)` | 将历史记录转为 Agent 可接受的输入格式 |
| `append_messages(threadId, messages)` | 追加本轮产生的消息到历史 |
| `clear_history(threadId)` | 清除指定会话的历史 |
| `clear_all_histories()` | 清除所有会话历史 |

**消息格式**：`{ role: "user"|"assistant"|"tool", content, tool_calls?, tool_call_id? }`，为可序列化的 plain object。
**上下文窗口**：默认保留最近 20 条消息（约 10 轮对话），超出时自动丢弃最早的消息 (FIFO)。

### tools.py — 本地 Agent 工具

| 工具 | 输入 | 功能 |
| ----------------------- | ------------------- | -------------------------------- |
| `get_deepseek_info` | 无参 | 读取 `deepseek介绍手册.txt` 全文 |
| `search_knowledge_base` | `{ query: string }` | 混合检索知识库 |
| `get_weather` | `{ city: string }` | 查询指定城市实时天气（OpenWeatherMap） |

### mcp_client.py — MCP 客户端连接管理器

核心 MCP 集成模块，提供三个导出函数：

| 函数 | 说明 |
| ---- | ---- |
| `create_tools_from_mcp_server(server_config)` | 连接单个 MCP Server，发现工具并包装为 LangChain StructuredTool |
| `load_all_mcp_tools(config_path)` | 从 JSON 配置文件加载所有 MCP Server 的工具 |
| `disconnect_all_mcp_servers()` | 断开所有 MCP Server 连接（进程退出时清理） |

内部将 MCP Tool 的 JSON Schema 参数定义自动转换为 Pydantic model，适配 LangChain 的工具接口。

### mcp-servers.config.json — MCP 服务器配置

声明要接入的 MCP Server 列表，每个条目包含：

| 字段 | 说明 | 示例 |
| ---- | ---- | ---- |
| `name` | 服务器名称（用作工具名前缀） | `"gitee"` |
| `command` | 启动命令 | `"node"` / `"npx"` / `"python"` |
| `args` | 命令参数数组 | `["-y", "@gitee/mcp-gitee@latest"]` |
| `env` | 额外环境变量 | `{"GITEE_ACCESS_TOKEN": "xxx"}` |

### hybrid_retriever.py — 混合检索器

- **向量检索**：通过 `langchain-postgres` 的 `PGVector` 连接 PostgreSQL，使用 `similarity_search` 进行语义搜索
- **BM25 检索**：自定义 `BM25Retriever`，使用 `rank_bm25` 库在内存中构建索引，中文采用单字粒度分词
- **RRF 融合**：按排名倒数加权融合两路结果

### ingest.py — 数据入库（增量同步）

扫描 `data/documents/` 目录下的 `.txt/.pdf/.docx` 文件，以文件内容 sha256 为指纹做**幂等增量同步**：新增文件 → 解析（TXT 直接读取、PDF 调用 `parse_pdf.py` 混合解析、DOCX 用 `python-docx` 提取文本）→ 分块 → 嵌入入库；修改文件 → 删除旧 chunks 后重新入库，库中只保留最新版本。使用 `RecursiveCharacterTextSplitter` 分块（200 字符/块，重叠 20 字符），`OllamaEmbeddings` + `bge-m3` 向量化后写入 PostgreSQL。支持 `--rebuild`（全量重建）与 `--prune`（清理已删除文档）。服务运行时可通过 `POST /api/admin/sync` 触发同步并自动刷新检索器缓存，无需重启。

### parse_pdf.py — PDF 混合解析

用 PyMuPDF 提取可选择文本 + 渲染页面为图片，再用 RapidOCR 识别图片文字，最后按 `MIN_TEXT_LENGTH`（80 字符）阈值智能合并两路结果。文字型 PDF 走文本提取，扫描件（图片型）PDF 走 OCR。

### query.py — 命令行查询

传统 RAG 三段式：Query Rewrite → 混合检索 → LLM 生成。

### diagnose.py — 诊断脚本

对比纯向量检索、纯 BM25 检索、混合检索三种策略的效果。

---

## 启动方式

```bash
# 1. 创建并激活虚拟环境（推荐），安装主项目依赖
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -e .

# 2. 配置环境变量（.env 文件）
#    需要设置 DEEPSEEK_API_KEY / DB_HOST / DB_PORT / DB_USER / DB_PASSWORD / DB_NAME
#    MCP Server 的额外环境变量也在此文件配置

# 3. 确保 Ollama 服务运行并加载 bge-m3 模型
#    ollama pull bge-m3
#    ollama serve

# 4. 注入知识库数据（幂等增量同步，自动解析 data/documents/ 下的 TXT/PDF/DOCX）
#    首次运行全量入库；之后新增/修改/删除文档后再次运行即可增量同步
python app/ingest.py
#    全量重建（清空向量库与注册表）：python app/ingest.py --rebuild
#    清理已删除文档：python app/ingest.py --prune

# 5. 启动服务（会自动连接配置的 MCP Server）
python app/server_agui.py

# 6. 命令行测试查询
python app/query.py

# 7. 测试 MCP 工具调用（替换为实际地址；具体工具以启动时动态发现为准）
curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"threadId":"test","runId":"1","messages":[{"role":"user","content":"帮我在 Gitee 上搜索 changelink 的开源项目"}]}'

# 8. 测试多轮对话（使用相同的 threadId）
#     第1轮
curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"threadId":"multi-test","runId":"1","messages":[{"role":"user","content":"DeepSeek 是什么公司？"}]}'

#     第2轮（Agent 能理解"它"指 DeepSeek）
curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"threadId":"multi-test","runId":"2","messages":[{"role":"user","content":"它的创始团队有哪些人？"}]}'

#     第3轮（追问细节）
curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"threadId":"multi-test","runId":"3","messages":[{"role":"user","content":"详细讲讲它的开源战略"}]}'

# 9. 清除会话历史（如需重置）
curl -X POST http://localhost:3001/api/chat/clear \
  -H "Content-Type: application/json" \
  -d '{"threadId":"multi-test"}'
```

### 验证 MCP 是否接入成功

启动后观察控制台日志：

```
[MCP] 连接服务器: gitee
[MCP]   命令: npx -y @gitee/mcp-gitee@latest
[MCP] gitee: 连接成功
[MCP] gitee: 发现 N 个工具
```

发送涉及 Gitee 的问题后，SSE 响应中会出现完整的工具调用生命周期：

```
TOOL_CALL_START  →  TOOL_CALL_ARGS  →  TOOL_CALL_RESULT  →  TOOL_CALL_END
```

---

## MCP 配置指南

### 增加一个新的 MCP 工具

在 [mcp-servers.config.json](file:///d:/AiAgent/my-rag-demo/config/mcp-servers.config.json) 中添加配置项：

```json
[
  {
    "name": "github",
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
    "env": {
      "GITHUB_PERSONAL_ACCESS_TOKEN": "你的 GitHub Token"
    }
  },
  {
    "name": "gitee",
    "command": "npx",
    "args": ["-y", "@gitee/mcp-gitee@latest"],
    "env": {
      "GITEE_API_BASE": "https://gitee.com/api/v5",
      "GITEE_ACCESS_TOKEN": "你的 Gitee 令牌"
    }
  }
]
```

重启服务后，Agent 会自动发现并注册新工具。

### 实用 MCP Server 推荐

| 包名 | 功能 | 启动命令 |
| ---- | ---- | ---- |
| `@modelcontextprotocol/server-github` | GitHub 仓库/Issue/PR 管理 | `npx -y @modelcontextprotocol/server-github` |
| `@gitee/mcp-gitee` | Gitee 仓库/Issue/PR 管理 | `npx -y @gitee/mcp-gitee@latest` |
| `@modelcontextprotocol/server-filesystem` | 本地文件系统读写 | `npx -y @modelcontextprotocol/server-filesystem <目录>` |
| `@modelcontextprotocol/server-puppeteer` | 浏览器自动化截图 | `npx -y @modelcontextprotocol/server-puppeteer` |
| `@modelcontextprotocol/server-brave-search` | 网页搜索 | `npx -y @modelcontextprotocol/server-brave-search` |
| `mcp-server-time` | 获取当前时间（Python） | `python -m mcp_server_time --local-timezone=Asia/Shanghai` |
