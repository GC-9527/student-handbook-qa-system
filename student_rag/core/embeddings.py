"""
Embedding模型封装

提供统一的Embedding接口，支持多种模型和本地/远程部署。
"""

import json
import numpy as np
import requests
from typing import List, Union, Optional
from pathlib import Path
from abc import ABC, abstractmethod

from sentence_transformers import SentenceTransformer

from student_rag.config import settings
from student_rag.utils.logger import logger


class BaseEmbeddingModel(ABC):
    """Embedding模型基类"""

    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """编码文本为向量"""
        pass

    @abstractmethod
    def encode_queries(self, queries: Union[str, List[str]]) -> np.ndarray:
        """编码查询文本"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """获取向量维度"""
        pass


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """
    Sentence-Transformers Embedding封装

    使用sentence-transformers库，支持：
    - BAAI/bge-large-zh-v1.5 (推荐，中文优化)
    - BAAI/bge-base-zh-v1.5 (轻量级)
    - 其他sentence-transformers兼容模型
    """

    def __init__(self, model_name: str = None, device: str = None):
        self.config = settings.embedding
        self.model_name = model_name or self.config.model_name
        self.device = device or self.config.device

        self._model = None
        self._dimension = None

        logger.info(f"SentenceTransformerEmbedding初始化: {self.model_name}, device={self.device}")

    def _load_model(self) -> SentenceTransformer:
        """懒加载模型"""
        if self._model is None:
            logger.info(f"加载Embedding模型: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"模型加载完成: dimension={self._dimension}")
        return self._model

    @property
    def dimension(self) -> int:
        """获取向量维度"""
        if self._dimension is None:
            _ = self._load_model()
        return self._dimension

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = None,
        show_progress: bool = False,
        normalize: bool = None
    ) -> np.ndarray:
        """将文本编码为向量"""
        model = self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or self.config.batch_size
        normalize = normalize if normalize is not None else self.config.normalize_embeddings

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )

        return embeddings

    def encode_queries(self, queries: Union[str, List[str]]) -> np.ndarray:
        """编码查询文本，BGE模型添加指令前缀"""
        if isinstance(queries, str):
            queries = [queries]

        if "bge" in self.model_name.lower():
            queries = [f"为这个句子生成表示以用于检索相关文章：{q}" for q in queries]

        return self.encode(queries)


class OllamaEmbedding(BaseEmbeddingModel):
    """
    Ollama Embedding封装

    使用Ollama服务提供的embedding功能，支持本地部署。
    推荐模型: mxbai-embed-large (1536维)
    """

    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        timeout: int = 60
    ):
        self.config = settings.embedding
        self.llm_config = settings.llm

        self.model = model or self.config.ollama_embedding_model
        self.base_url = base_url or self.llm_config.ollama_base_url
        self.timeout = timeout

        self._dimension = None

        logger.info(f"OllamaEmbedding初始化: model={self.model}, base_url={self.base_url}")

    def _get_dimension(self) -> int:
        """获取模型维度，需要先调用API获取"""
        if self._dimension is None:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": "test"},
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                self._dimension = len(result.get("embedding", []))
                logger.info(f"Ollama Embedding维度: {self._dimension}")
            except Exception as e:
                logger.warning(f"获取Ollama Embedding维度失败: {e}")
                self._dimension = 1024  # 默认值
        return self._dimension

    @property
    def dimension(self) -> int:
        """获取向量维度"""
        return self._get_dimension()

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = None,
        show_progress: bool = False,
        normalize: bool = None
    ) -> np.ndarray:
        """将文本编码为向量"""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                embedding = result.get("embedding", [])

                if normalize:
                    embedding = self._normalize(embedding)

                embeddings.append(embedding)

            except Exception as e:
                logger.error(f"Ollama Embedding请求失败: {e}")
                embedding = [0.0] * self.dimension
                embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)

    def encode_queries(self, queries: Union[str, List[str]]) -> np.ndarray:
        """编码查询文本，Ollama不需要指令前缀"""
        return self.encode(queries)

    def _normalize(self, embedding: List[float]) -> List[float]:
        """L2归一化"""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return [x / norm for x in embedding]
        return embedding


class EmbeddingFactory:
    """Embedding模型工厂"""

    @staticmethod
    def create(provider: str = None) -> BaseEmbeddingModel:
        """
        创建Embedding模型实例

        Args:
            provider: 提供商类型 (sentence_transformers/ollama)

        Returns:
            Embedding模型实例
        """
        provider = provider or settings.embedding.provider

        if provider == "sentence_transformers":
            return SentenceTransformerEmbedding()
        elif provider == "ollama":
            return OllamaEmbedding()
        else:
            logger.warning(f"未知的Embedding提供商: {provider}, 使用默认的sentence_transformers")
            return SentenceTransformerEmbedding()


# 向后兼容：保持原有的EmbeddingModel类名
EmbeddingModel = SentenceTransformerEmbedding


class EmbeddingCache:
    """
    Embedding缓存

    避免重复计算相同文本的embedding，提高性能。
    """

    def __init__(self, cache_dir: Union[str, Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache = {}

    def _get_cache_key(self, text: str, model_name: str) -> str:
        """生成缓存键"""
        import hashlib
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, model_name: str) -> np.ndarray:
        """从缓存获取embedding"""
        key = self._get_cache_key(text, model_name)

        # 先查内存缓存
        if key in self._memory_cache:
            return self._memory_cache[key]

        # 再查磁盘缓存
        cache_file = self.cache_dir / f"{key}.npy"
        if cache_file.exists():
            embedding = np.load(cache_file)
            self._memory_cache[key] = embedding
            return embedding

        return None

    def set(self, text: str, model_name: str, embedding: np.ndarray):
        """设置缓存"""
        key = self._get_cache_key(text, model_name)

        # 内存缓存
        self._memory_cache[key] = embedding

        # 磁盘缓存
        cache_file = self.cache_dir / f"{key}.npy"
        np.save(cache_file, embedding)

    def clear(self):
        """清空缓存"""
        self._memory_cache.clear()
        for f in self.cache_dir.glob("*.npy"):
            f.unlink()
