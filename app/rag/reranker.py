# reranker.py — Cross-Encoder 重排器（sentence-transformers）

import asyncio
from typing import Optional

from langchain_core.documents import Document
from app.core.logging import get_logger

logger = get_logger()


class CrossEncoderReranker:
    """基于 sentence-transformers CrossEncoder 的重排器

    使用 BAAI/bge-reranker-base 等交叉编码器模型，对"查询 ↔ 候选文档"逐对打分，
    弥补 bi-encoder（如 bge-m3 向量检索）无法细粒度比较候选片段的不足。
    模型惰性加载，首次调用时从 HuggingFace 下载权重。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        max_length: int = 512,
        batch_size: int = 16,
        device: Optional[str] = None,
    ):
        self._model_name = model_name
        self._max_length = max_length
        self._batch_size = batch_size
        self._device = device
        self._model = None  # 惰性加载

    def _load_model(self):
        """首次调用时加载模型（首次需下载权重，可能较慢）"""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self._model_name,
                max_length=self._max_length,
                device=self._device,
            )

    def _predict(self, query: str, docs: list[Document]) -> list[float]:
        self._load_model()
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]

    async def rerank(
        self,
        query: str,
        docs: list[Document],
        top_k: Optional[int] = None,
    ) -> list[Document]:
        """对候选文档按重排分数降序返回，分数写入 doc.metadata["rerankScore"]。

        模型加载或推理失败（如首次下载失败）时打印警告并按原顺序返回，
        保证 RAG 主流程不被重排拖垮（调用方负责截断）。
        """
        if not docs:
            return []

        try:
            scores = await asyncio.to_thread(self._predict, query, docs)
        except Exception as err:
            logger.warning(f"重排失败，按原顺序返回候选: {err}")
            return docs

        ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)

        ordered: list[Document] = []
        for score, doc in ranked:
            doc.metadata["rerankScore"] = score
            ordered.append(doc)

        if top_k is not None and top_k > 0:
            ordered = ordered[:top_k]
        return ordered
