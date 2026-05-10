"""
RAG引擎

整合文档处理、向量检索、LLM生成的完整RAG流程。
提供引用溯源功能，确保答案可追溯。
"""

import time
from typing import List, Optional

from student_rag.core.document_processor import DocumentProcessor
from student_rag.core.retriever import VectorRetriever
from student_rag.core.embeddings import EmbeddingFactory
from student_rag.core.reranker import RerankerFactory
from student_rag.core.llm import LLMFactory, build_rag_prompt, format_citations
from student_rag.core.vector_store import VectorStore
from student_rag.models.schemas import (
    DocumentChunk,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    RetrievalResult
)
from student_rag.utils.logger import logger
from student_rag.config import settings


class RAGEngine:
    """
    RAG引擎

    提供端到端的RAG问答能力：
    1. 文档索引：PDF → Chunks → Vectors
    2. 检索：Query → Embedding → Vector Search → ReRank
    3. 生成：Context + Query → LLM → Answer with Citations

    特点：
    - 完整的引用溯源（页码、章节）
    - 检索结果对比（重排序前后）
    - 模块化设计，组件可替换
    """

    def __init__(
        self,
        document_processor: DocumentProcessor = None,
        retriever: VectorRetriever = None,
        llm=None
    ):
        self.document_processor = document_processor or DocumentProcessor()
        self.retriever = retriever
        self.llm = llm

        self._initialized = False

        logger.info("RAGEngine初始化完成")

    def initialize(
        self,
        embedding_model: str = None,
        reranker_model: str = None
    ):
        """
        初始化引擎组件

        可以传入自定义模型，否则使用配置中的默认模型。
        """
        # 初始化Embedding模型（通过工厂创建，支持ollama/sentence_transformers）
        embedding = EmbeddingFactory.create()

        # 初始化ReRanker（通过工厂创建，支持cohere/bge）
        reranker = RerankerFactory.create()

        # 初始化检索器
        self.retriever = VectorRetriever(
            embedding_model=embedding,
            reranker=reranker
        )

        # 初始化LLM
        self.llm = LLMFactory.create()

        self._initialized = True
        logger.info("RAGEngine组件初始化完成")

    def index_document(
        self,
        pdf_path: str,
        document_id: str = None
    ) -> List[DocumentChunk]:
        """
        索引单个文档

        流程：
        1. 解析PDF → Chunks
        2. 编码Chunks → Embeddings
        3. 构建向量索引
        """
        if not self._initialized:
            self.initialize()

        logger.info(f"开始索引文档: {pdf_path}")

        # 步骤1: 文档处理
        chunks = self.document_processor.process_pdf(pdf_path, document_id)

        # 步骤2: 构建索引
        self.retriever.build_index(chunks)

        logger.info(f"文档索引完成: {len(chunks)}个chunks")
        return chunks

    def index_documents(
        self,
        pdf_paths: List[str]
    ) -> List[DocumentChunk]:
        """批量索引多个文档"""
        if not self._initialized:
            self.initialize()

        logger.info(f"开始批量索引: {len(pdf_paths)}个文档")

        # 处理所有文档
        all_chunks = self.document_processor.process_multiple_pdfs(pdf_paths)

        # 构建索引
        self.retriever.build_index(all_chunks)

        logger.info(f"批量索引完成: {len(all_chunks)}个chunks")
        return all_chunks

    def query(
        self,
        query: str,
        top_k: int = None,
        enable_rerank: bool = None,
        return_sources: bool = True
    ) -> QueryResponse:
        """
        执行RAG查询

        流程：
        1. 检索相关chunks
        2. 构建prompt（包含引用信息）
        3. 调用LLM生成答案
        4. 组装响应（答案 + 引用 + 来源）
        """
        if not self._initialized:
            raise RuntimeError("引擎未初始化，请先调用initialize()或index_document()")

        start_time = time.time()

        logger.info(f"开始查询: {query}")

        # 步骤1: 检索
        retrieval_start = time.time()
        retrieval_result = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            enable_rerank=enable_rerank,
            return_before_rerank=True  # 用于对比
        )
        retrieval_time = (time.time() - retrieval_start) * 1000

        if not retrieval_result.chunks:
            return QueryResponse(
                query=query,
                answer="抱歉，未找到与问题相关的资料。",
                citations=[],
                sources=[],
                retrieval_time_ms=retrieval_time,
                generation_time_ms=0
            )

        # 步骤2: 构建Prompt
        contexts = [chunk.content for chunk in retrieval_result.chunks]
        citations = format_citations(retrieval_result.chunks)
        prompt = build_rag_prompt(query, contexts, citations)

        # 步骤3: 生成答案
        generation_start = time.time()
        answer = self.llm.generate(prompt)
        generation_time = (time.time() - generation_start) * 1000

        # 步骤4: 组装响应
        response = QueryResponse(
            query=query,
            answer=answer,
            citations=citations[:len(retrieval_result.chunks)],
            sources=retrieval_result.chunks if return_sources else [],
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time
        )

        total_time = (time.time() - start_time) * 1000
        logger.info(
            f"查询完成: 检索{retrieval_time:.1f}ms, 生成{generation_time:.1f}ms, "
            f"总计{total_time:.1f}ms"
        )

        return response

    def query_with_comparison(
        self,
        query: str,
        top_k: int = 5
    ) -> dict:
        """
        执行对比查询（展示重排序效果）

        返回重排序前后的结果对比。
        """
        if not self._initialized:
            raise RuntimeError("引擎未初始化")

        # 检索，返回重排序前的结果
        retrieval_result = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            enable_rerank=True,
            return_before_rerank=True
        )

        # 构建对比报告
        before = retrieval_result.before_rerank or []
        after = retrieval_result.chunks

        comparison = {
            "query": query,
            "before_rerank": [
                {
                    "rank": i + 1,
                    "score": chunk.similarity_score,
                    "content_preview": chunk.content[:100] + "...",
                    "citation": chunk.metadata.get_citation()
                }
                for i, chunk in enumerate(before[:top_k])
            ],
            "after_rerank": [
                {
                    "rank": chunk.rank,
                    "embedding_score": chunk.similarity_score,
                    "rerank_score": chunk.rerank_score,
                    "content_preview": chunk.content[:100] + "...",
                    "citation": chunk.metadata.get_citation()
                }
                for chunk in after
            ],
            "improvement": retrieval_result.rerank_improvement
        }

        return comparison

    def save_index(self, path: str = None):
        """保存索引"""
        if self.retriever:
            self.retriever.save_index(path)
            logger.info(f"索引已保存: {path}")

    def load_index(self, path: str = None):
        """加载索引"""
        if not self._initialized:
            self.initialize()

        if self.retriever:
            self.retriever.load_index(path)
            logger.info(f"索引已加载: {path}")

    def get_stats(self) -> dict:
        """获取引擎统计信息"""
        stats = {
            "initialized": self._initialized,
            "document_processor": self.document_processor.get_chunk_statistics(
                []  # 需要传入实际chunks
            ) if hasattr(self.document_processor, 'get_chunk_statistics') else {},
        }

        if self.retriever:
            stats["retriever"] = self.retriever.get_index_stats()

        return stats


class RAGPipeline:
    """
    RAG流水线（简化版）

    提供一步到位的RAG功能，适合简单场景。
    """

    def __init__(self):
        self.engine = RAGEngine()

    def build_index(self, pdf_path: str):
        """构建索引"""
        return self.engine.index_document(pdf_path)

    def ask(self, question: str) -> str:
        """提问"""
        response = self.engine.query(question)
        return response.answer

    def ask_with_sources(self, question: str) -> QueryResponse:
        """提问并返回来源"""
        return self.engine.query(question, return_sources=True)
