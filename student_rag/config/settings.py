"""
配置管理模块

集中管理所有配置项，支持环境变量覆盖和配置文件加载。
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent


class DocumentConfig(BaseSettings):
    """文档处理配置"""
    model_config = SettingsConfigDict(env_prefix="DOC_")

    # Chunk切分策略
    chunk_size: int = Field(default=800, description="Chunk大小(tokens)")
    chunk_overlap: int = Field(default=160, description="Chunk重叠大小(tokens)")
    chunk_strategy: str = Field(default="semantic", description="切分策略: semantic/recursive")

    # 制度文档优化参数
    preserve_hierarchy: bool = Field(default=True, description="保留文档层级结构")
    min_chunk_size: int = Field(default=100, description="最小Chunk大小")
    max_chunk_size: int = Field(default=1500, description="最大Chunk大小")

    # PDF解析
    extract_tables: bool = Field(default=True, description="提取表格")
    extract_images: bool = Field(default=False, description="提取图片")
    ocr_enabled: bool = Field(default=False, description="启用OCR")


class EmbeddingConfig(BaseSettings):
    """Embedding模型配置"""
    model_config = SettingsConfigDict(env_prefix="EMB_")

    # 提供商: sentence_transformers / ollama
    provider: str = Field(default="ollama", description="Embedding提供商")

    # Sentence-Transformers模型
    model_name: str = Field(
        default="BAAI/bge-large-zh-v1.5",
        description="Sentence-Transformers模型名称"
    )
    device: str = Field(default="cpu", description="运行设备: cpu/cuda")
    normalize_embeddings: bool = Field(default=True, description="归一化向量")

    # Ollama Embedding配置
    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Ollama Embedding模型名称"
    )

    # 批处理
    batch_size: int = Field(default=32, description="批处理大小")
    max_seq_length: int = Field(default=512, description="最大序列长度")


class RetrievalConfig(BaseSettings):
    """检索配置"""
    model_config = SettingsConfigDict(env_prefix="RET_")

    # 向量检索
    top_k: int = Field(default=10, description="初始检索数量")
    similarity_threshold: float = Field(default=0.5, description="相似度阈值")
    distance_metric: str = Field(default="cosine", description="距离度量: cosine/l2/ip")

    # ReRanker
    reranker_enabled: bool = Field(default=True, description="启用ReRanker")
    reranker_provider: str = Field(default="flashrank", description="ReRanker提供商: flashrank/cohere/bge")
    reranker_model: str = Field(
        default="ms-marco-MiniLM-L-12-v2",
        description="ReRanker模型"
    )
    reranker_top_k: int = Field(default=5, description="重排序后保留数量")
    reranker_threshold: float = Field(default=0.3, description="重排序分数阈值")

    # Cohere API配置
    cohere_api_key: Optional[str] = Field(default=None, description="Cohere API密钥")


class LLMConfig(BaseSettings):
    """LLM配置"""
    model_config = SettingsConfigDict(env_prefix="LLM_")

    # 提供商
    provider: str = Field(default="ollama", description="提供商: openai/ollama/azure")

    # OpenAI配置
    api_key: Optional[str] = Field(default=None, description="OpenAI API密钥")
    base_url: Optional[str] = Field(default=None, description="OpenAI API基础URL")

    # Ollama配置
    ollama_model: str = Field(default="Lusizo/qwen2.5-7b-instruct-1m:latest", description="Ollama模型名称")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama服务地址")

    # 模型选择（通用）
    model_name: str = Field(default="gemma4:e4b", description="模型名称")

    # 生成参数
    temperature: float = Field(default=0.3, description="温度参数")
    max_tokens: int = Field(default=2000, description="最大生成token数")
    top_p: float = Field(default=0.9, description="Top-p采样")


class StorageConfig(BaseSettings):
    """存储配置"""
    model_config = SettingsConfigDict(env_prefix="STORAGE_")

    # 向量存储
    vector_store_type: str = Field(default="faiss", description="向量存储类型: faiss/chroma")
    index_path: Path = Field(
        default=PROJECT_ROOT / "data" / "index",
        description="索引存储路径"
    )
    persist_index: bool = Field(default=True, description="持久化索引")

    # 文档存储
    document_path: Path = Field(
        default=PROJECT_ROOT / "data" / "pdfs",
        description="PDF文档路径"
    )


class Settings(BaseSettings):
    """全局配置"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # 子配置
    document: DocumentConfig = DocumentConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()
    storage: StorageConfig = StorageConfig()

    # 应用配置
    debug: bool = Field(default=False, description="调试模式")
    log_level: str = Field(default="INFO", description="日志级别")

    # 溯源配置
    citation_enabled: bool = Field(default=True, description="启用引用溯源")
    citation_format: str = Field(default="footnote", description="引用格式: footnote/inline")


# 全局配置实例
settings = Settings()
