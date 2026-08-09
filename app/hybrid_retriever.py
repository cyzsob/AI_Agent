# hybrid_retriever.py — BM25 + 向量混合检索

import os
import re
import hashlib
from typing import Optional

import asyncio
import asyncpg
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings
from reranker import CrossEncoderReranker

# ========== 中文分词辅助 ==========


def tokenize_for_bm25(text: str) -> str:
    """在中文汉字之间插入空格，使得 BM25 可以将每个汉字作为独立 token 索引。
    英文/数字保持原样，按空格分词。"""
    text = re.sub(
        r"([\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff])",
        r" \1 ",
        text,
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


# ========== BM25 检索器 (基于 rank_bm25) ==========


class BM25Retriever(BaseRetriever):
    """使用 rank_bm25 的 BM25 检索器"""

    def __init__(self, bm25_index: BM25Okapi, doc_map: dict, k: int = 6):
        super().__init__()
        self._bm25 = bm25_index
        self._doc_map = doc_map
        self._k = k

    async def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        processed_query = tokenize_for_bm25(query)
        tokenized_query = processed_query.split()

        scores = self._bm25.get_scores(tokenized_query)

        # 获取 top-k 得分的索引
        top_indices = np.argsort(scores)[::-1][: self._k]

        docs = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            # doc_map keys are database row IDs (strings)
            doc_id = list(self._doc_map.keys())[idx]
            doc_data = self._doc_map[doc_id]
            docs.append(
                Document(
                    page_content=doc_data["pageContent"],
                    metadata={**doc_data["metadata"], "bm25Score": float(scores[idx])},
                )
            )

        return docs


# ========== RRF 融合 ==========


def _doc_key(doc: Document) -> str:
    """统一的文档去重键：基于内容哈希。

    向量检索返回的 doc 无 dbId（metadata 仅为入库时的 cmetadata），
    BM25 检索返回的 doc 有 dbId，同一 chunk 在两路的键必须一致才能正确去重，
    因此统一使用内容哈希而非 dbId / 内容前缀。"""
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()


def rrf_merge(
    vector_docs: list[Document],
    bm25_docs: list[Document],
    final_k: int = 3,
    rrf_k: int = 60,
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5,
) -> list[Document]:
    """Reciprocal Rank Fusion — 将多路检索结果按排名加权融合"""
    score_map: dict[str, dict] = {}

    # 向量结果：按排名打分
    for i, doc in enumerate(vector_docs):
        doc_id = _doc_key(doc)
        if doc_id not in score_map:
            score_map[doc_id] = {"doc": doc, "score": 0.0}
        score_map[doc_id]["score"] += vector_weight / (rrf_k + i + 1)

    # BM25 结果：按排名打分
    for i, doc in enumerate(bm25_docs):
        doc_id = _doc_key(doc)
        if doc_id not in score_map:
            score_map[doc_id] = {"doc": doc, "score": 0.0}
        score_map[doc_id]["score"] += bm25_weight / (rrf_k + i + 1)

    # 按融合得分降序排列，并把融合分写回 doc 元数据（便于诊断）
    merged = sorted(score_map.values(), key=lambda x: x["score"], reverse=True)

    for item in merged:
        item["doc"].metadata["rrfScore"] = item["score"]

    return [item["doc"] for item in merged[:final_k]]


# ========== 查询向量缓存 ==========
#
# 每次检索都要对查询调用本地 Ollama 嵌入模型（bge-m3，CPU 上单次数百 ms），
# 相同/相似的查询（如多轮追问、多个用户问同一问题）会反复触发该调用。
# 这里对"规范化后的查询文本 → 向量"做进程内缓存：命中时跳过嵌入调用，
# 直接用缓存向量做 PG 相似度检索，大幅缩短检索阶段延迟。

_QUERY_EMBEDDING_CACHE: dict[str, list[float]] = {}
_QUERY_EMBEDDING_CACHE_SIZE = 512


# ========== 混合检索器 ==========


class HybridRetriever(BaseRetriever):
    """混合检索器：向量 + BM25 → RRF 融合 → (可选) Cross-Encoder 重排"""

    def __init__(
        self,
        vector_store,
        bm25_retriever: BM25Retriever,
        final_k: int = 3,
        rrf_k: int = 60,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        vector_k: int = 6,
        reranker: Optional[CrossEncoderReranker] = None,
        rerank_candidates: int = 10,
        enable_rerank: bool = True,
    ):
        super().__init__()
        self._vector_store = vector_store
        self._vector_k = vector_k
        self._bm25_retriever = bm25_retriever
        self._final_k = final_k
        self._rrf_k = rrf_k
        self._vector_weight = vector_weight
        self._bm25_weight = bm25_weight
        self._reranker = reranker
        self._rerank_candidates = rerank_candidates
        self._enable_rerank = enable_rerank

    def _embed_query_cached(self, query: str) -> list[float]:
        """查询向量化（带进程内缓存，命中时省去 Ollama 嵌入调用）。

        缓存键为规范化后的查询文本；容量超限时整体清空（简单淘汰，
        避免缓存无限膨胀，且向量维度远大于键长，内存可控）。
        """
        cache_key = query.strip().lower()
        cached = _QUERY_EMBEDDING_CACHE.get(cache_key)
        if cached is not None:
            return cached
        vector = self._vector_store.embeddings.embed_query(query)
        if len(_QUERY_EMBEDDING_CACHE) >= _QUERY_EMBEDDING_CACHE_SIZE:
            _QUERY_EMBEDDING_CACHE.clear()
        _QUERY_EMBEDDING_CACHE[cache_key] = vector
        return vector

    async def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        # 并行执行向量检索（sync→线程池，含查询向量缓存）和 BM25 检索（async）
        def _vector_search():
            embedding = self._embed_query_cached(query)
            return self._vector_store.similarity_search_by_vector(
                embedding, k=self._vector_k
            )

        vector_docs, bm25_docs = await asyncio.gather(
            asyncio.to_thread(_vector_search),
            self._bm25_retriever._get_relevant_documents(query),
        )

        # RRF 融合 → 候选池（不直接截断到 final_k，先放大候选供重排）
        candidates = rrf_merge(
            vector_docs,
            bm25_docs,
            final_k=self._rerank_candidates,
            rrf_k=self._rrf_k,
            vector_weight=self._vector_weight,
            bm25_weight=self._bm25_weight,
        )

        # Cross-Encoder 重排（可选）→ 最终 Top-K
        if self._enable_rerank and self._reranker is not None and candidates:
            return await self._reranker.rerank(query, candidates, top_k=self._final_k)

        return candidates[: self._final_k]


# ========== 工厂函数 ==========

_cached_hybrid_retriever: Optional[HybridRetriever] = None


def refresh_retriever() -> None:
    """使缓存的混合检索器失效，下次检索时自动重建（含 BM25 索引）。

    文档增量入库（ingest.py / POST /api/admin/sync）后调用，
    让运行中的服务无需重启即可检索到最新数据。
    """
    global _cached_hybrid_retriever
    _cached_hybrid_retriever = None


async def get_hybrid_retriever(options: Optional[dict] = None) -> HybridRetriever:
    """获取混合检索器实例（带缓存，避免重复构建 BM25 索引）

    Args:
        options: dict with keys:
            vectorK (int): 向量检索取 top-K，默认 6
            bm25K (int): BM25 检索取 top-K，默认 6
            finalK (int): 融合+重排后最终返回 top-K，默认 3
            vectorWeight (float): 向量检索权重，默认 0.5
            bm25Weight (float): BM25 检索权重，默认 0.5
            rerankModel (str): sentence-transformers 重排模型名，默认 "BAAI/bge-reranker-base"
            rerankCandidates (int): RRF 融合后交给重排器的候选池大小，默认 5
            enableRerank (bool): 是否启用重排，默认 True
            forceRebuild (bool): 强制重建缓存，默认 False
    """
    global _cached_hybrid_retriever

    opts = options or {}
    vector_k = opts.get("vectorK", 6)
    bm25_k = opts.get("bm25K", 6)
    final_k = opts.get("finalK", 3)
    vector_weight = opts.get("vectorWeight", 0.5)
    bm25_weight = opts.get("bm25Weight", 0.5)
    rerank_model = opts.get("rerankModel", "BAAI/bge-reranker-base")
    rerank_candidates = opts.get("rerankCandidates", 5)
    enable_rerank = opts.get("enableRerank", True)
    force_rebuild = opts.get("forceRebuild", False)

    if _cached_hybrid_retriever is not None and not force_rebuild:
        return _cached_hybrid_retriever

    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")
    db_name = os.getenv("DB_NAME", "RAG_test")

    connection_string = (
        f"postgresql+psycopg://{db_user}:{db_password}"
        f"@{db_host}:{db_port}/{db_name}"
    )

    # ---------- 1. 初始化向量存储 ----------
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

    # ---------- 2. 从 DB 加载全部文档，构建 rank_bm25 索引 ----------
    conn = await asyncpg.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        database=db_name,
    )

    try:
        # 数据源与 ingest.py 写入一致：从 langchain_postgres 的 PGVector 表读取，
        # 而不是旧版遗留的 documents 表
        rows = await conn.fetch(
            "SELECT e.id, e.document AS content, e.cmetadata AS metadata "
            "FROM langchain_pg_embedding e "
            "JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
            "WHERE c.name = 'documents' "
            "ORDER BY e.id"
        )
    finally:
        await conn.close()

    rows_list = [dict(row) for row in rows]
    if not rows_list:
        raise RuntimeError("数据库中没有文档，请先运行 ingest.py 注入数据。")

    doc_map = {}
    tokenized_corpus = []
    for row in rows_list:
        row_id = str(row["id"])
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            import json
            metadata = json.loads(metadata)
        doc_map[row_id] = {
            "pageContent": row["content"],
            "metadata": {**metadata, "dbId": row_id},
        }
        tokenized_corpus.append(tokenize_for_bm25(row["content"]).split())

    # 构建 BM25 索引
    bm25_index = BM25Okapi(tokenized_corpus)

    # ---------- 3. 创建 BM25 检索器 ----------
    bm25_retriever = BM25Retriever(
        bm25_index=bm25_index,
        doc_map=doc_map,
        k=bm25_k,
    )

    # ---------- 4. 创建混合检索器 ----------
    reranker = CrossEncoderReranker(model_name=rerank_model)

    _cached_hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        final_k=final_k,
        rrf_k=60,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        vector_k=vector_k,
        reranker=reranker,
        rerank_candidates=rerank_candidates,
        enable_rerank=enable_rerank,
    )

    return _cached_hybrid_retriever
