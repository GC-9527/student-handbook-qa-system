"""
向量存储

基于FAISS的向量索引实现，支持：
- 高效相似度检索
- 增量添加向量
- 索引持久化
"""

import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

import faiss

from student_rag.models.schemas import DocumentChunk
from student_rag.utils.logger import logger
from student_rag.config import settings


class VectorStore:
    """
    向量存储

    封装FAISS索引，提供：
    1. 向量添加和删除
    2. 相似度检索
    3. 索引持久化
    4. 元数据管理
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "cosine",
        index_path: Optional[Path] = None
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.index_path = index_path or settings.storage.index_path

        self._index = None
        self._chunks: List[DocumentChunk] = []
        self._id_to_index: Dict[str, int] = {}

        logger.info(f"VectorStore初始化: dimension={dimension}, index_type={index_type}")

    def _create_index(self) -> faiss.Index:
        """创建FAISS索引"""
        if self.index_type == "cosine":
            # 余弦相似度：使用归一化后的内积
            # IndexFlatIP = Inner Product，对于归一化向量等价于余弦相似度
            index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "l2":
            # L2距离
            index = faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")

        logger.info(f"创建索引: {self.index_type}, dimension={self.dimension}")
        return index

    @property
    def index(self) -> faiss.Index:
        """获取或创建索引"""
        if self._index is None:
            self._index = self._create_index()
        return self._index

    def add(
        self,
        chunks: List[DocumentChunk],
        embeddings: np.ndarray
    ):
        """
        添加向量和对应的chunks

        Args:
            chunks: DocumentChunk列表
            embeddings: 对应的向量，shape为 (n_chunks, dimension)
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"chunks数量({len(chunks)})与embeddings数量({embeddings.shape[0]})不匹配")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"向量维度({embeddings.shape[1]})与索引维度({self.dimension})不匹配")

        # 确保是float32（FAISS要求）
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # 添加到索引
        start_idx = len(self._chunks)
        self.index.add(embeddings)

        # 保存chunks和映射
        for i, chunk in enumerate(chunks):
            self._chunks.append(chunk)
            self._id_to_index[chunk.id] = start_idx + i

        logger.info(f"添加{len(chunks)}个向量，索引总数: {len(self._chunks)}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = None
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """
        检索最相似的chunks

        Args:
            query_embedding: 查询向量
            top_k: 返回结果数量
            threshold: 相似度阈值，低于此值的结果会被过滤

        Returns:
            (chunks列表, 相似度分数列表)
        """
        if len(self._chunks) == 0:
            return [], []

        # 确保是float32和2D数组
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # 检索
        scores, indices = self.index.search(query_embedding, min(top_k, len(self._chunks)))

        # 处理结果
        results = []
        result_scores = []

        threshold = threshold or settings.retrieval.similarity_threshold

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS返回-1表示没有更多结果
                continue

            # 对于余弦相似度，分数范围是[-1, 1]，我们取大于阈值的
            # 对于L2距离，分数越小越相似
            if self.index_type == "cosine":
                if score < threshold:
                    continue
            else:
                # L2距离，可以设置最大距离阈值
                pass

            chunk = self._chunks[idx]
            results.append(chunk)
            result_scores.append(float(score))

        return results, result_scores

    def delete(self, chunk_id: str) -> bool:
        """
        删除指定chunk

        注意：FAISS不支持直接删除，这里使用标记删除策略
        """
        if chunk_id not in self._id_to_index:
            return False

        # 标记为已删除（实际删除需要重建索引）
        idx = self._id_to_index[chunk_id]
        self._chunks[idx] = None
        del self._id_to_index[chunk_id]

        logger.info(f"删除chunk: {chunk_id}")
        return True

    def save(self, path: Optional[Path] = None):
        """
        保存索引到磁盘

        保存内容：
        1. FAISS索引文件
        2. Chunks元数据
        3. ID映射
        """
        path = path or self.index_path
        path.mkdir(parents=True, exist_ok=True)

        # 保存FAISS索引
        faiss.write_index(self.index, str(path / "index.faiss"))

        # 保存chunks（过滤掉已删除的）
        valid_chunks = [c for c in self._chunks if c is not None]
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(valid_chunks, f)

        # 保存配置
        config = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "total_vectors": len(valid_chunks),
        }
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info(f"索引已保存: {path}, 共{len(valid_chunks)}个向量")

    def load(self, path: Optional[Path] = None):
        """从磁盘加载索引"""
        path = path or self.index_path

        if not path.exists():
            raise FileNotFoundError(f"索引目录不存在: {path}")

        # 加载配置
        with open(path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        self.dimension = config["dimension"]
        self.index_type = config["index_type"]

        # 加载FAISS索引
        self._index = faiss.read_index(str(path / "index.faiss"))

        # 加载chunks
        with open(path / "chunks.pkl", "rb") as f:
            self._chunks = pickle.load(f)

        # 重建ID映射
        self._id_to_index = {chunk.id: i for i, chunk in enumerate(self._chunks)}

        logger.info(f"索引已加载: {path}, 共{len(self._chunks)}个向量")

    def clear(self):
        """清空索引"""
        self._index = self._create_index()
        self._chunks = []
        self._id_to_index = {}
        logger.info("索引已清空")

    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        valid_chunks = [c for c in self._chunks if c is not None]
        return {
            "total_vectors": len(self._chunks),
            "valid_vectors": len(valid_chunks),
            "deleted_vectors": len(self._chunks) - len(valid_chunks),
            "dimension": self.dimension,
            "index_type": self.index_type,
        }
