# query.py — 命令行 RAG 查询（改写 → 检索 → 生成）

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from hybrid_retriever import get_hybrid_retriever


async def main():
    # ---------- 1. 初始化混合检索器 ----------
    retriever = await get_hybrid_retriever({
        "vectorK": 10,
        "bm25K": 10,
        "finalK": 3,
        "vectorWeight": 0.5,
        "bm25Weight": 0.5,
        "rerankCandidates": 5,
    })

    # ---------- 2. 初始化 DeepSeek 模型 ----------
    model = ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com/v1",
        presence_penalty=0.4,
        frequency_penalty=0.4,
        max_tokens=4096,
    )

    # ---------- 3. Query Rewrite 链 ----------
    rewrite_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """你是一个查询改写助手。你的任务是将用户的问题改写为更适合知识库检索的简洁查询。

规则：
- 去除口语化表达、语气词、重复词
- 将代词（它、这个、那个、它们等）替换为具体名词
- 保持核心语义不变
- 如果问题本身已经是简洁的查询，保留原样
- 只输出改写后的查询，不要任何多余内容""",
        ),
        ("human", "改写以下查询：\n{input}"),
    ])

    rewrite_chain = rewrite_prompt | model | StrOutputParser()

    # ---------- 4. 构建 RAG 提示模板 ----------
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """你是一个智能助手，请根据以下上下文信息回答用户的问题。
如果上下文信息不足以回答问题，请如实告知，不要编造答案。

上下文信息：
{context}""",
        ),
        ("human", "{input}"),
    ])

    # ---------- 5. 构建 RAG 链（改写 → 检索 → 生成） ----------

    async def rewrite_step(user_input: str):
        rewritten = await rewrite_chain.ainvoke({"input": user_input})
        print(f'[Query Rewrite] 原始: "{user_input}"')
        print(f'[Query Rewrite] 改写: "{rewritten}"')
        return {"original": user_input, "rewritten": rewritten}

    async def retrieve_step(data: dict):
        rewritten = data["rewritten"]
        docs = await retriever._get_relevant_documents(rewritten)
        return {
            "context": "\n\n".join(doc.page_content for doc in docs),
            "input": data["original"],
        }

    chain = RunnableLambda(rewrite_step) | RunnableLambda(retrieve_step) | prompt | model | StrOutputParser()

    # ---------- 6. 执行查询 ----------
    user_question = "deepseek的开源战略是什么"
    print(f"问题: {user_question}")

    try:
        answer = await chain.ainvoke(user_question)
        print(f"回答: {answer}")
    except Exception as err:
        print(f"❌ 查询失败: {err}")


if __name__ == "__main__":
    asyncio.run(main())
