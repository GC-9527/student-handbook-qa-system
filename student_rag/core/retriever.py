"""
检索器

整合Embedding编码、向量检索、ReRanker重排序的完整检索流程。
"""

import time
from typing import List, Optional
import numpy as np

from student_rag.core.embeddings import EmbeddingModel, EmbeddingFactory
from student_rag.core.vector_store import VectorStore
from student_rag.core.reranker import BaseReranker, RerankerFactory
from student_rag.models.schemas import DocumentChunk, RetrievedChunk, RetrievalResult
from student_rag.utils.logger import logger, log_retrieval_result
from student_rag.config import settings


class VectorRetriever:
    """
    向量检索器

    提供完整的检索流程：
    1. 查询编码（Embedding）
    2. 向量检索（Vector Search）
    3. 重排序（ReRanker）
    4. 结果组装

    支持对比模式：返回重排序前后的结果对比
    """

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None,
        reranker: Optional[BaseReranker] = None,
        embedding_provider: str = None,
        reranker_provider: str = None
    ):
        if embedding_model is None:
            provider = embedding_provider or settings.embedding.provider
            self.embedding_model = EmbeddingFactory.create(provider)
        else:
            self.embedding_model = embedding_model

        self.vector_store = vector_store

        if reranker is None and settings.retrieval.reranker_enabled:
            self.reranker = RerankerFactory.create(reranker_provider)
        else:
            self.reranker = reranker

        self.config = settings.retrieval

        reranker_name = self.reranker.name if self.reranker else "disabled"
        logger.info(
            f"VectorRetriever初始化: "
            f"provider={settings.embedding.provider}, "
            f"top_k={self.config.top_k}, "
            f"reranker={reranker_name}"
        )

    def build_index(self, chunks: List[DocumentChunk]):
        """
        构建向量索引

        步骤：
        1. 编码所有chunks
        2. 创建向量存储
        3. 添加到索引
        """
        logger.info(f"开始构建索引: {len(chunks)}个chunks")

        # 编码
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(
            texts,
            show_progress=True,
            normalize=True  # 归一化，用于余弦相似度
        )

        # 创建向量存储
        dimension = self.embedding_model.dimension
        self.vector_store = VectorStore(
            dimension=dimension,
            index_type=self.config.distance_metric
        )

        # 添加到索引
        self.vector_store.add(chunks, embeddings)

        logger.info(f"索引构建完成: dimension={dimension}, vectors={len(chunks)}")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        enable_rerank: bool = None,
        return_before_rerank: bool = False
    ) -> RetrievalResult:
        """
        执行检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            enable_rerank: 是否启用重排序
            return_before_rerank: 是否同时返回重排序前的结果（用于对比）

        Returns:
            RetrievalResult对象
        """
        start_time = time.time()

        top_k = top_k or self.config.top_k
        enable_rerank = enable_rerank if enable_rerank is not None else self.config.reranker_enabled

        # 步骤1: 编码查询
        query_embedding = self.embedding_model.encode_queries(query)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # 步骤2: 向量检索
        initial_k = top_k * 3 if enable_rerank else top_k  # 如果重排序，多检索一些
        chunks, scores = self.vector_store.search(query_embedding, top_k=initial_k)

        if not chunks:
            return RetrievalResult(
                query=query,
                chunks=[],
                total_found=0,
                retrieval_method="vector"
            )

        # 组装初始结果
        initial_results = [
            RetrievedChunk(
                **chunk.model_dump(),
                similarity_score=score,
                rerank_score=None,
                retrieval_method="vector",
                rank=i + 1
            )
            for i, (chunk, score) in enumerate(zip(chunks, scores))
        ]

        before_rerank = initial_results.copy() if return_before_rerank else None

        # 步骤3: 重排序
        final_results = initial_results
        if enable_rerank and self.reranker and len(initial_results) > 1:
            final_results = self._rerank(query, initial_results, top_k)

        retrieval_time = (time.time() - start_time) * 1000

        # 计算重排序改善率
        rerank_improvement = None
        if before_rerank and final_results:
            rerank_improvement = self._calculate_improvement(
                before_rerank[:top_k],
                final_results
            )

        # 记录结果
        log_retrieval_result(query, final_results, "retrieval")
        logger.info(f"检索完成: {len(final_results)}个结果, 耗时{retrieval_time:.1f}ms")

        return RetrievalResult(
            query=query,
            chunks=final_results[:top_k],
            total_found=len(final_results),
            retrieval_method="vector+rerank" if enable_rerank else "vector",
            before_rerank=before_rerank,
            rerank_improvement=rerank_improvement
        )

    def _rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int
    ) -> List[RetrievedChunk]:
        """
        重排序

        使用ReRanker模型对候选结果进行精排。
        """
        if not self.reranker:
            return candidates

        logger.debug(f"开始重排序: {len(candidates)}个候选")

        # 准备数据
        passages = [chunk.content for chunk in candidates]

        # 重排序
        rerank_scores = self.reranker.rerank(query, passages)

        # 更新分数并排序
        for chunk, score in zip(candidates, rerank_scores):
            chunk.rerank_score = score

        # 按重排序分数排序
        sorted_results = sorted(
            candidates,
            key=lambda x: x.rerank_score if x.rerank_score is not None else 0,
            reverse=True
        )

        # 更新排名
        for i, chunk in enumerate(sorted_results, 1):
            chunk.rank = i

        logger.debug(f"重排序完成: 最高分={max(rerank_scores):.4f}")

        return sorted_results

    def _calculate_improvement(
        self,
        before: List[RetrievedChunk],
        after: List[RetrievedChunk]
    ) -> float:
        """
        计算重排序改善率

        简单指标：检查重排序后前3个结果中有多少是原来的前5个
        """
        before_top5_ids = {c.id for c in before[:5]}
        after_top3_ids = {c.id for c in after[:3]}

        improved = len(after_top3_ids & before_top5_ids)
        return improved / 3.0

    def save_index(self, path: Optional[str] = None):
        """保存索引"""
        if self.vector_store:
            self.vector_store.save(path)

    def load_index(self, path: Optional[str] = None):
        """加载索引"""
        if self.vector_store is None:
            # 先加载配置获取dimension
            import json
            from pathlib import Path

            path = path or settings.storage.index_path
            with open(Path(path) / "config.json", "r") as f:
                config = json.load(f)

            self.vector_store = VectorStore(
                dimension=config["dimension"],
                index_type=config["index_type"]
            )

        self.vector_store.load(path)

    def get_index_stats(self) -> dict:
        """获取索引统计信息"""
        if self.vector_store:
            return self.vector_store.get_stats()
        return {}
