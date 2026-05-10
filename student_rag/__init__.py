"""
学生手册问答助手 - 企业级RAG系统

基于PDF文档的检索增强生成(RAG)问答系统，专为制度类文档优化。
支持文档切分、向量检索、ReRanker重排序、引用溯源等企业级功能。
"""

__version__ = "1.0.0"
__author__ = "Enterprise RAG Team"

from student_rag.core.rag_engine import RAGEngine
from student_rag.core.document_processor import DocumentProcessor
from student_rag.core.retriever import VectorRetriever

__all__ = [
    "RAGEngine",
    "DocumentProcessor",
    "VectorRetriever",
]
