# DeepSeek RAG Agent — 多 Agent 协同知识库问答系统

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/langgraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

基于 **DeepSeek V4 Flash + LangGraph Multi-Agent（Supervisor 模式）+ PGVector + BM25 + Cross-Encoder 重排** 的混合检索 RAG 系统。通过 **AG-UI 协议** 提供 SSE 流式服务，并通过 **MCP（Model Context Protocol）** 标准协议接入外部工具。

---

## 功能特性

| 特性 | 说明 |
|------|------|
| **多 Agent 协同** | Supervisor 监督者模式：1 个协调 Agent + 3 个专业 Worker（知识/DevOps/通用），按域分派任务 |
| **四层检索流水线** | 向量检索（bge-m3）+ BM25（中文逐字）+ RRF 融合 + Cross-Encoder 重排（bge-reranker） |
| **AG-UI 流式输出** | 完整的 SSE 事件生命周期：RUN_STARTED → TOOL_CALL → TEXT_MESSAGE（逐 token）→ RUN_FINISHED |
| **MCP 外部工具** | 通过 Model Context Protocol 接入第三方工具（如 Gitee），配置文件驱动、零代码扩展 |
| **多轮对话** | 基于 threadId 的对话历史，tiktoken 精确 Token 计数（100K 上限滑动窗口），支持追问和指代消解 |
| **多模态理解** | 用户上传图片 → 智谱 GLM-4V 生成文字描述 → 进入 RAG 流程 |
| **文档增量入库** | sha256 指纹幂等同步，支持 TXT/PDF/DOCX；PDF 混合解析（PyMuPDF + RapidOCR） |
| **结构化日志** | loguru 日志系统，文件轮转 + request_id 追踪 + 错误分离 |
| **健康检查** | `GET /health` 端点：数据库连通性 + Agent 状态 + 检索器缓存状态 |
| **运行时热更新** | `POST /api/admin/sync` 增量同步文档库，无需重启服务 |

---

## 项目结构

```
my-rag-demo/
├── app/                              # 应用源码（企业级分层）
│   ├── main.py                       # 启动入口：FastAPI App 工厂 + Lifespan
│   ├── core/                         # 基础设施层
│   │   ├── config.py                 #   集中配置 + 环境变量校验
│   │   ├── logging.py                #   结构化日志（loguru）
│   │   └── exceptions.py             #   自定义异常类
│   ├── api/                          # Web 接入层
│   │   ├── routes.py                 #   全部端点（/api/chat, /health, /capabilities...）
│   │   ├── sse.py                    #   AG-UI SSE 事件协议
│   │   └── stream.py                 #   流式工具调用追踪器
│   ├── agent/                        # Agent 业务层
│   │   ├── __init__.py               #   init_agent() 入口 + 工具注入
│   │   ├── supervisor.py             #   Supervisor 多 Agent 图构建
│   │   ├── history.py                #   多轮对话历史（Token 计数裁剪）
│   │   └── tools.py                  #   本地工具定义
│   ├── rag/                          # 检索层
│   │   ├── retriever.py              #   混合检索器（向量 + BM25 + RRF + 重排）
│   │   └── reranker.py               #   Cross-Encoder 重排器
│   ├── ingestion/                    # 数据入库层
│   │   ├── sync.py                   #   增量同步入口
│   │   └── pdf_parser.py             #   PDF 混合解析（PyMuPDF + RapidOCR）
│   ├── mcp/                          # MCP 外部工具层
│   │   └── client.py                 #   MCP 客户端连接 + 工具发现 + 重试
│   ├── services/                     # 通用服务层
│   │   ├── multimodal.py             #   多模态视觉理解
│   │   └── message.py                #   多模态消息解析
│   └── scripts/                      # 独立 CLI 工具
│       ├── diagnose.py               #   检索诊断
│       └── query.py                  #   命令行 RAG 查询
├── config/
│   └── mcp-servers.config.json       # MCP 服务器配置
├── data/documents/                   # 知识库源文档
├── docs/                             # 技术文档
├── logs/                             # 运行日志
├── .env                              # 环境变量
└── pyproject.toml                    # 项目依赖
```

---

## 快速开始

### 1. 环境准备

```bash
# Python 3.11+
python --version

# PostgreSQL 14+ 并安装 pgvector 扩展
# 创建数据库后执行: CREATE EXTENSION IF NOT EXISTS vector;

# Ollama（本地嵌入模型）
ollama pull bge-m3
ollama serve
```

### 2. 安装依赖

```bash
git clone <your-repo-url>
cd my-rag-demo

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -e .
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env`，填写必要的配置：

```bash
# LLM
DEEPSEEK_API_KEY=sk-your-key          # DeepSeek API 密钥（必填）
DEEPSEEK_MODEL=deepseek-v4-flash      # 模型名称
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 数据库
DB_HOST=localhost                     # PostgreSQL 主机（必填）
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your-password
DB_NAME=RAG_test                      # 数据库名（必填）

# 嵌入模型
EMBEDDING_MODEL=bge-m3
OLLAMA_BASE_URL=http://localhost:11434

# 多模态（可选，图片理解功能）
GLM_API_KEY=your-zhipu-api-key

# 服务端口
PORT=3001
```

### 4. 注入知识库数据

将文档放入 `data/documents/` 目录，然后运行：

```bash
# 增量同步（幂等，自动跳过未变化的文件）
python app/ingestion/sync.py

# 全量重建
python app/ingestion/sync.py --rebuild

# 清理已删除文档
python app/ingestion/sync.py --prune
```

### 5. 启动服务

```bash
python -m app.main
```

服务默认运行在 `http://localhost:3001`。

---

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/capabilities` | AG-UI 协议能力声明 |
| `POST` | `/api/chat` | 核心聊天端点（SSE 流式响应） |
| `POST` | `/api/chat/clear` | 清除指定会话历史 |
| `POST` | `/api/admin/sync` | 增量同步文档库 + 热刷新检索器 |
| `GET` | `/health` | 健康检查（数据库 / Agent / 检索器） |

### 聊天请求格式

```bash
curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "threadId": "my-session",
    "runId": "run-001",
    "messages": [
      {"role": "user", "content": "DeepSeek-V4 采用了什么架构？"}
    ]
  }'
```

响应为 SSE 事件流，事件类型包括：

```
RUN_STARTED → TOOL_CALL_START → TOOL_CALL_ARGS → TOOL_CALL_RESULT → TOOL_CALL_END
     → TEXT_MESSAGE_START → TEXT_MESSAGE_CONTENT (逐 token) → TEXT_MESSAGE_END
     → RUN_FINISHED
```

### 多轮对话

使用相同的 `threadId` 即可：

```bash
# 第 1 轮
curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"threadId":"s1","runId":"1","messages":[{"role":"user","content":"DeepSeek 是什么公司？"}]}'

# 第 2 轮（Agent 能理解"它"指 DeepSeek）
curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"threadId":"s1","runId":"2","messages":[{"role":"user","content":"它的创始团队有哪些人？"}]}'
```

---

## 多 Agent 架构

```
用户消息
  │
  ▼
┌──────────────────────────────────────────┐
│  Supervisor（协调 Agent，只负责路由分发）    │
│  TransferToKnowledge / TransferToDevops / │
│  TransferToGeneral / finish_workflow      │
└──────┬───────────────┬──────────────┬─────┘
       ▼               ▼              ▼
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │knowledge │  │  devops  │  │ general  │
  │ 知识检索  │  │ Gitee工具 │  │ 天气闲聊  │
  │ RAG 工具  │  │ MCP 工具  │  │  天气API  │
  └────┬─────┘  └────┬─────┘  └────┬─────┘
       └──────┬──────┘──────┬──────┘
              ▼              ▼
         Supervisor（再路由 / END）
```

| Worker | 持有工具 | 职责 |
|--------|---------|------|
| `knowledge` | `get_deepseek_info`, `search_kb` | DeepSeek 公司介绍 + 知识库技术问答 |
| `devops` | `gitee_*`（MCP 动态发现） | Gitee 仓库/Issue/PR 操作 |
| `general` | `get_weather` | 实时天气 + 日常对话 |

**设计要点**：
- Supervisor 的输出不透出给客户端、不进入对话历史（纯内部路由）
- 跨域任务通过 `handoff_to_xxx` 工具确定性交接，避免 LLM 误判
- 未配置 MCP 时 `devops` 节点自动跳过

---

## 混合检索流水线

```
用户查询
  │
  ├──→ 向量检索 (PGVector + bge-m3, 1024维)    Top-10
  │
  ├──→ BM25 检索 (rank_bm25, 中文逐字分词)     Top-10
  │
  └──→ RRF 融合 (Reciprocal Rank Fusion)       Top-10
       │
       └──→ Cross-Encoder 重排 (bge-reranker)   Top-3
```

**RRF 融合公式**：`score = 0.5/(60+rank_vec) + 0.5/(60+rank_bm25)`

向量检索擅长**语义匹配**，BM25 擅长**关键词精确匹配**，RRF 融合取两者之长，Cross-Encoder 做最终精排。

---

## MCP 外部工具集成

在 `config/mcp-servers.config.json` 中声明工具：

```json
[
  {
    "name": "gitee",
    "command": "npx",
    "args": ["-y", "@gitee/mcp-gitee@latest"],
    "env": {
      "GITEE_ACCESS_TOKEN": "你的令牌"
    }
  }
]
```

**工作流程**：启动时连接子进程 → `list_tools()` 自动发现 → JSON Schema → Pydantic → LangChain StructuredTool → 注册到 Agent

新增工具只需修改配置文件，重启服务即可，零代码改动。MCP 连接失败自动重试 3 次（指数退避）。

---

## 日志与监控

```bash
# 查看结构化日志
tail -f logs/app_$(date +%Y-%m-%d).log

# 查看错误日志
tail -f logs/error_$(date +%Y-%m-%d).log

# 健康检查
curl http://localhost:3001/health
# {"status":"ok","checks":{"database":"ok","agent":"ok","retriever_cache":"ready"}}

# 检索诊断
python -m app.scripts.diagnose
```

所有请求自动携带 `request_id`，方便日志串联排查。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| LLM | DeepSeek V4 Flash（OpenAI 兼容 API） |
| Agent 框架 | LangGraph 0.2+ (StateGraph + create_react_agent) |
| 嵌入模型 | bge-m3（Ollama 本地运行，1024 维） |
| 重排模型 | BAAI/bge-reranker-base（Cross-Encoder） |
| 向量数据库 | PostgreSQL + pgvector |
| BM25 | rank_bm25（纯 Python，内存索引） |
| 后端 | FastAPI + Uvicorn + SSE |
| MCP 协议 | mcp Python SDK（stdio JSON-RPC） |
| 多模态 | 智谱 GLM-4V（图片 → 文字） |
| PDF 解析 | PyMuPDF + RapidOCR（onnxruntime） |
| 日志 | loguru（结构化 + 文件轮转） |
| Token 计数 | tiktoken（cl100k_base） |

---

## 更多文档

- [关键功能实现详解](file:///d:/AiAgent/my-rag-demo/docs/关键功能的实现.md)

---

## License

MIT
