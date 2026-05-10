"""
文档处理器

整合PDF解析、Chunk切分、语义重叠等功能，提供统一的文档处理接口。
"""

from pathlib import Path
from typing import List, Optional
import uuid

from student_rag.core.pdf_parser import PDFParser, SemanticChunker
from student_rag.models.schemas import Document, DocumentChunk
from student_rag.utils.logger import logger
from student_rag.config import settings


class DocumentProcessor:
    """
    文档处理器

    提供从原始PDF到可索引Chunks的完整处理流程：
    1. PDF解析（文本、表格、结构）
    2. Chunk切分（语义边界）
    3. 重叠策略（保持上下文）
    4. 元数据增强（溯源信息）

    Chunk策略说明（适合制度类文档）：
    =================================

    1. Chunk大小选择：512-1024 tokens
       - 制度文档通常有完整的"条件-流程-结果"逻辑单元
       - 过小的chunk会割裂制度条文的完整性
       - 过大的chunk会引入噪声，降低检索精度

    2. 重叠策略：20%（约100-200 tokens）
       - 前向重叠：保留前一个chunk的最后一句
       - 目的：保持跨chunk的上下文连贯性
       - 特别重要：制度文档中条款经常相互引用

    3. 切分边界：优先段落边界，其次句子边界
       - 段落是制度文档的自然语义单元
       - 避免在句子中间切分，保持语义完整

    4. 层级保留：章节标题继承
       - 每个chunk保留所属章节标题
       - 目的：回答时能准确定位到制度章节
    """

    def __init__(self):
        self.pdf_parser = PDFParser()
        self.chunker = SemanticChunker()
        self.config = settings.document

        logger.info(
            f"DocumentProcessor初始化完成: "
            f"chunk_size={self.config.chunk_size}, "
            f"overlap={self.config.chunk_overlap}"
        )

    def process_pdf(
        self,
        pdf_path: str | Path,
        document_id: Optional[str] = None,
        enable_overlap: bool = True
    ) -> List[DocumentChunk]:
        """
        处理PDF文档，生成Chunks

        Args:
            pdf_path: PDF文件路径
            document_id: 文档ID，自动生成if None
            enable_overlap: 是否启用重叠策略

        Returns:
            处理后的DocumentChunk列表
        """
        pdf_path = Path(pdf_path)
        document_id = document_id or str(uuid.uuid4())

        logger.info(f"开始处理文档: {pdf_path.name}")

        # 步骤1: PDF解析
        chunks = self.pdf_parser.parse_to_chunks(pdf_path, document_id)
        logger.info(f"步骤1完成 - PDF解析: 生成{len(chunks)}个初始chunks")

        # 步骤2: 应用重叠策略
        if enable_overlap and len(chunks) > 1:
            overlap_ratio = self.config.chunk_overlap / self.config.chunk_size
            chunks = self.chunker.create_chunks_with_overlap(chunks, overlap_ratio)
            logger.info(f"步骤2完成 - 重叠策略: 应用{overlap_ratio:.0%}重叠")

        # 步骤3: 过滤和验证
        chunks = self._filter_chunks(chunks)
        logger.info(f"步骤3完成 - 过滤验证: 保留{len(chunks)}个有效chunks")

        # 步骤4: 统计信息
        total_tokens = sum(c.token_count for c in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0

        logger.info(
            f"文档处理完成: {pdf_path.name}\n"
            f"  - 总Chunks: {len(chunks)}\n"
            f"  - 总Tokens: {total_tokens}\n"
            f"  - 平均Chunk大小: {avg_tokens:.0f} tokens"
        )

        return chunks

    def process_multiple_pdfs(
        self,
        pdf_paths: List[str | Path],
        enable_overlap: bool = True
    ) -> List[DocumentChunk]:
        """批量处理多个PDF文档"""
        all_chunks = []

        for pdf_path in pdf_paths:
            try:
                chunks = self.process_pdf(pdf_path, enable_overlap=enable_overlap)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"处理文档失败 {pdf_path}: {e}")
                continue

        logger.info(f"批量处理完成: 共{len(pdf_paths)}个文档，{len(all_chunks)}个chunks")
        return all_chunks

    def _filter_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        过滤和验证chunks

        - 移除过短的chunks（可能是噪声）
        - 移除重复的chunks
        """
        filtered = []
        seen_contents = set()

        for chunk in chunks:
            # 过滤太短的内容
            if chunk.token_count < self.config.min_chunk_size:
                continue

            # 过滤重复内容（基于前100个字符）
            content_hash = chunk.content[:100].strip()
            if content_hash in seen_contents:
                continue
            seen_contents.add(content_hash)

            filtered.append(chunk)

        return filtered

    def get_chunk_statistics(self, chunks: List[DocumentChunk]) -> dict:
        """获取Chunk统计信息"""
        if not chunks:
            return {}

        token_counts = [c.token_count for c in chunks]
        char_counts = [c.char_count for c in chunks]

        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "total_chars": sum(char_counts),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "avg_chars": sum(char_counts) / len(char_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "documents": len(set(c.metadata.document_id for c in chunks)),
        }
