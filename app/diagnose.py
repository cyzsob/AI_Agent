# diagnose.py — 诊断检索问题（向量 + BM25 + 混合）

import os
import re
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

import asyncpg
import numpy as np
from rank_bm25 import BM25Okapi

from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from hybrid_retriever import get_hybrid_retriever, tokenize_for_bm25


def tokenize_for_lunr(text: str) -> str:
    """兼容旧版 lunr 风格的分词：中文按字分词"""
    return (
        re.sub(r"([\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff])", r" \1 ", text)
        .replace(r"\s+", " ")
        .strip()
        .lower()
    )


async def main():
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "RAG_test")

    # ---------- 1. 数据库连接 ----------
    conn = await asyncpg.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        database=db_name,
    )

    # ---------- 2. 查看所有 chunk（数据源与 hybrid_retriever.py 一致） ----------
    result = await conn.fetch(
        "SELECT e.id, LEFT(e.document, 80) AS content_preview, e.cmetadata AS metadata "
        "FROM langchain_pg_embedding e "
        "JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
        "WHERE c.name = 'documents' "
        "ORDER BY e.id"
    )
    print("========== 数据库中所有 chunk ==========")
    for i, row in enumerate(result):
        row_dict = dict(row)
        print(f"\n[Chunk {i + 1}] id={row_dict['id'][:8]}...")
        print(f"  内容: {row_dict['content_preview']}...")
        print(f"  元数据: {row_dict['metadata']}")

    # ---------- 3. 构建 BM25 索引（用于单独诊断） ----------
    all_docs = await conn.fetch(
        "SELECT e.id, e.document AS content, e.cmetadata AS metadata "
        "FROM langchain_pg_embedding e "
        "JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
        "WHERE c.name = 'documents' "
        "ORDER BY e.id"
    )
    rows_list = [dict(row) for row in all_docs]

    doc_map = {}
    tokenized_corpus = []
    for row in rows_list:
        row_id = str(row["id"])
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        doc_map[row_id] = {
            "pageContent": row["content"],
            "metadata": metadata,
        }
        tokenized_corpus.append(tokenize_for_lunr(row["content"]).split())

    await conn.close()

    bm25_index = BM25Okapi(tokenized_corpus)

    # ---------- 4. 测试检索（三种方式对比） ----------
    queries = ["deepseek的开源战略是什么", "DeepSeek公司"]

    for query in queries:
        print(f'\n========== 检索: "{query}" ==========')

        # 4a. 纯向量检索
        print("--- [向量检索] ---")
        connection_string = (
            f"postgresql+psycopg://{db_user}:{db_password}"
            f"@{db_host}:{db_port}/{db_name}"
        )

        embeddings = OllamaEmbeddings(
            model="bge-m3",
            base_url="http://localhost:11434",
        )

        vector_store = PGVector(
            embeddings=embeddings,
            collection_name="documents",
            connection=connection_string,
            use_jsonb=True,
        )

        try:
            vec_results = vector_store.similarity_search(query, k=5)
            if not vec_results:
                print("  未检索到结果")
            else:
                for i, doc in enumerate(vec_results):
                    print(f"  [{i + 1}] {doc.page_content[:100]}...")
        except Exception as err:
            print(f"  检索失败: {err}")

        # 4b. 纯 BM25 检索
        print("--- [BM25 检索] ---")
        tokenized_query = tokenize_for_lunr(query).split()
        scores = bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1]

        bm25_results_found = False
        count = 0
        for idx in top_indices:
            if scores[idx] <= 0 or count >= 5:
                break
            doc_id = list(doc_map.keys())[idx]
            doc = doc_map[doc_id]
            print(
                f"  [{count + 1}] (BM25: {scores[idx]:.4f}) "
                f"{doc['pageContent'][:100]}..."
            )
            bm25_results_found = True
            count += 1
        if not bm25_results_found:
            print("  未检索到结果")

        # 4c. 混合检索 + Rerank
        print("--- [混合检索 (向量+BM25 → Rerank)] ---")
        try:
            hybrid_ret = await get_hybrid_retriever({
                "vectorK": 10,
                "bm25K": 10,
                "finalK": 3,
                "vectorWeight": 0.5,
                "bm25Weight": 0.5,
                "rerankCandidates": 10,
                "forceRebuild": True,
            })
            hybrid_docs = await hybrid_ret._get_relevant_documents(query)
            if not hybrid_docs:
                print("  未检索到结果")
            else:
                for i, doc in enumerate(hybrid_docs):
                    rrf_score = doc.metadata.get("rrfScore")
                    rerank_score = doc.metadata.get("rerankScore")
                    if rerank_score is not None:
                        score_str = (
                            f"(rrf: {rrf_score:.4f}, rerank: {rerank_score:.4f})"
                        )
                    else:
                        score_str = f"(rrf: {rrf_score:.4f})"
                    print(f"  [{i + 1}] {score_str} {doc.page_content[:80]}...")
        except Exception as err:
            print(f"  混合检索失败: {err}")

        # 4d. 纯 RRF（关闭重排，对照验证重排增益）
        print("--- [混合检索 (纯 RRF，无 Rerank)] ---")
        try:
            rrf_only_ret = await get_hybrid_retriever({
                "vectorK": 10,
                "bm25K": 10,
                "finalK": 3,
                "vectorWeight": 0.5,
                "bm25Weight": 0.5,
                "enableRerank": False,
                "forceRebuild": True,
            })
            rrf_only_docs = await rrf_only_ret._get_relevant_documents(query)
            if not rrf_only_docs:
                print("  未检索到结果")
            else:
                for i, doc in enumerate(rrf_only_docs):
                    rrf_score = doc.metadata.get("rrfScore")
                    print(f"  [{i + 1}] (rrf: {rrf_score:.4f}) {doc.page_content[:80]}...")
        except Exception as err:
            print(f"  纯 RRF 检索失败: {err}")


if __name__ == "__main__":
    asyncio.run(main())
