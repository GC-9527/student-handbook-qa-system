"""
ReRanker重排序模块

提供检索结果的精排功能，显著提升检索质量。

支持的ReRanker方案：
1. FlashRank - 推荐（免费、轻量、无需API Key）
2. Cohere Rerank API - 需要API Key
3. BGE-Reranker - 本地部署（需要下载模型）
"""

from typing import List, Optional, Dict, Any
import numpy as np
from abc import ABC, abstractmethod

from student_rag.config import settings
from student_rag.utils.logger import logger

SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

COHERE_AVAILABLE = False
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    pass

FLASHRANK_AVAILABLE = False
try:
    from flashrank import Ranker as FlashRanker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    pass


class BaseReranker(ABC):
    """ReRanker基类"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        passages: List[str],
        top_k: int = None
    ) -> List[float]:
        """对文档进行重排序，返回相关性分数"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """返回ReRanker名称"""
        pass


class CohereReranker(BaseReranker):
    """
    Cohere Reranker

    使用Cohere的Rerank API进行重排序。
    - 免费额度：每月1,000次调用
    - 支持多语言（包括中文）
    - 效果稳定可靠
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "rerank-multilingual-v3.0",
        base_url: str = None
    ):
        self.config = settings.retrieval
        self.api_key = api_key or getattr(settings.retrieval, 'cohere_api_key', None)
        self.model = model or "rerank-multilingual-v3.0"
        self.base_url = base_url

        if not COHERE_AVAILABLE:
            raise ImportError("请安装cohere: pip install cohere")

        if not self.api_key:
            logger.warning("Cohere API Key未设置，将无法使用Cohere Reranker")

        self._client = None
        logger.info(f"CohereReranker初始化: model={self.model}")

    @property
    def name(self) -> str:
        return f"Cohere-{self.model}"

    def _get_client(self):
        """获取或创建Cohere客户端"""
        if self._client is None and self.api_key:
            self._client = cohere.Client(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def rerank(
        self,
        query: str,
        passages: List[str],
        top_k: int = None
    ) -> List[float]:
        """使用Cohere Rerank API进行重排序"""
        if not self._get_client():
            raise ValueError("Cohere API Key未设置或无效")

        top_k = top_k or self.config.reranker_top_k

        try:
            response = self._get_client().rerank(
                query=query,
                documents=passages,
                model=self.model,
                top_n=top_k,
                return_documents=False
            )

            scores = [0.0] * len(passages)
            for result in response.results:
                scores[result.index] = result.relevance_score

            logger.debug(f"Cohere Rerank完成: {len(passages)}个文档")
            return scores

        except Exception as e:
            logger.error(f"Cohere Rerank失败: {e}")
            raise


class BGEReranker(BaseReranker):
    """BGE Reranker - 本地部署"""

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        max_length: int = 512
    ):
        self.config = settings.retrieval
        self.model_name = model_name or self.config.reranker_model
        self.device = device or settings.embedding.device
        self.max_length = max_length

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("请安装sentence-transformers: pip install sentence-transformers")

        self._model = None
        logger.info(f"BGEReranker初始化: {self.model_name}, device={self.device}")

    @property
    def name(self) -> str:
        return self.model_name

    def _load_model(self) -> CrossEncoder:
        """懒加载模型"""
        if self._model is None:
            logger.info(f"加载BGE-Reranker模型: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=self.max_length,
                trust_remote_code=True
            )
            logger.info("BGE-Reranker模型加载完成")
        return self._model

    def rerank(
        self,
        query: str,
        passages: List[str],
        top_k: int = None
    ) -> List[float]:
        """使用BGE-Reranker进行重排序"""
        model = self._load_model()
        top_k = top_k or self.config.reranker_top_k

        pairs = [[query, passage] for passage in passages]
        scores = model.predict(pairs, show_progress_bar=False)

        if isinstance(scores, np.ndarray):
            scores = scores.tolist()

        logger.debug(f"BGE-Rerank完成: {len(passages)}个文档")
        return scores[:top_k] if top_k else scores


class FlashRankReranker(BaseReranker):
    """
    FlashRank Reranker

    使用FlashRank进行轻量级重排序。
    - 完全免费，无需API Key
    - 轻量级(~4MB)，CPU即可运行
    - 支持多语言

    模型选项：
    - ms-marco-MiniLM-L-12-v2: 英文，轻量快速
    - cross-encoder-ms-marco-MiniLM-L-12-v2: 英文
    - cross-encoder-ms-marco-MultiBERT-L-12: 多语言
    """

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2"
    ):
        self.config = settings.retrieval
        self.model_name = model_name

        if not FLASHRANK_AVAILABLE:
            raise ImportError("请安装flashrank: pip install flashrank")

        self._ranker = None
        logger.info(f"FlashRankReranker初始化: {self.model_name}")

    @property
    def name(self) -> str:
        return f"FlashRank-{self.model_name}"

    def _load_ranker(self):
        """懒加载模型"""
        if self._ranker is None:
            logger.info(f"加载FlashRank模型: {self.model_name}")
            self._ranker = FlashRanker(model_name=self.model_name)
            logger.info("FlashRank模型加载完成")
        return self._ranker

    def rerank(
        self,
        query: str,
        passages: List[str],
        top_k: int = None
    ) -> List[float]:
        """使用FlashRank进行重排序"""
        ranker = self._load_ranker()
        top_k = top_k or self.config.reranker_top_k

        pairs = []
        for idx, passage in enumerate(passages):
            pairs.append({"id": idx, "text": passage})

        request = RerankRequest(query=query, passages=pairs)
        results = ranker.rerank(request)

        scores = [0.0] * len(passages)
        for result in results:
            doc_id = result.get("id", result.get("doc_id", 0))
            if doc_id < len(passages):
                scores[doc_id] = result["score"]

        reranked_results = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)[:top_k]
        scores = [score for score, _ in reranked_results]

        logger.debug(f"FlashRank完成: {len(passages)}个文档")
        return scores


class RerankerFactory:
    """ReRanker工厂"""

    @staticmethod
    def create(provider: str = None) -> Optional[BaseReranker]:
        """创建ReRanker实例"""
        provider = provider or getattr(settings.retrieval, 'reranker_provider', 'flashrank')

        if provider == "flashrank":
            try:
                return FlashRankReranker()
            except Exception as e:
                logger.warning(f"FlashRank Reranker创建失败: {e}")
                return None
        elif provider == "cohere":
            try:
                return CohereReranker()
            except (ValueError, ImportError) as e:
                logger.warning(f"Cohere Reranker创建失败: {e}")
                return None
        elif provider == "bge":
            try:
                return BGEReranker()
            except Exception as e:
                logger.warning(f"BGE Reranker创建失败: {e}")
                return None
        else:
            logger.warning(f"未知的ReRanker提供商: {provider}")
            return None


class RerankerEvaluator:
    """ReRanker效果评估器"""

    def __init__(self, reranker: BaseReranker = None):
        if reranker is None:
            reranker = RerankerFactory.create()
        self.reranker = reranker

    def compare_rankings(
        self,
        query: str,
        initial_ranking: List[Dict[str, Any]],
        top_k: int = 5
    ) -> dict:
        """对比重排序前后的结果"""
        passages = [item['text'] for item in initial_ranking]
        initial_scores = [item['score'] for item in initial_ranking]

        if self.reranker:
            rerank_scores = self.reranker.rerank(query, passages)
        else:
            rerank_scores = initial_scores

        comparison = []
        for i, (init_score, rerank_score) in enumerate(zip(initial_scores, rerank_scores)):
            comparison.append({
                'index': i,
                'text_preview': passages[i][:100] + "...",
                'initial_score': float(init_score),
                'rerank_score': float(rerank_score),
                'score_change': float(rerank_score - init_score),
                'rank_change': 0
            })

        initial_ranks = list(range(len(initial_ranking)))
        rerank_ranks = sorted(range(len(rerank_scores)), key=lambda i: rerank_scores[i], reverse=True)
        rank_mapping = {old_rank: new_rank for new_rank, old_rank in enumerate(rerank_ranks)}

        for item in comparison:
            old_rank = item['index']
            new_rank = rank_mapping[old_rank]
            item['rank_change'] = old_rank - new_rank

        top_k_initial = set(initial_ranks[:top_k])
        top_k_rerank = set(rerank_ranks[:top_k])

        return {
            'query': query,
            'top_k': top_k,
            'total_documents': len(passages),
            'overlap_at_top_k': len(top_k_initial & top_k_rerank),
            'overlap_ratio': len(top_k_initial & top_k_rerank) / top_k,
            'average_score_change': sum(c['score_change'] for c in comparison) / len(comparison),
            'reranker_name': self.reranker.name if self.reranker else "None",
            'comparison_details': comparison[:top_k * 2]
        }

    def evaluate_batch(self, test_cases: List[dict]) -> dict:
        """批量评估ReRanker效果"""
        results = [self.compare_rankings(case['query'], case['initial_ranking'], top_k=5) for case in test_cases]

        avg_overlap = sum(r['overlap_ratio'] for r in results) / len(results)
        avg_score_change = sum(r['average_score_change'] for r in results) / len(results)

        return {
            'total_queries': len(test_cases),
            'average_overlap_at_top5': avg_overlap,
            'average_score_change': avg_score_change,
            'individual_results': results
        }


# 向后兼容
Reranker = BGEReranker
