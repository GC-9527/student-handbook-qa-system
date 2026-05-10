"""
PDF解析器

提供高质量的PDF文本提取，支持页码、表格、章节结构识别。
使用pdfplumber和pymupdf组合，兼顾准确性和速度。
"""

import uuid
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import pdfplumber
import fitz  # pymupdf

from student_rag.models.schemas import Document, DocumentChunk, ChunkMetadata, DocumentType
from student_rag.utils.text_utils import clean_text, extract_hierarchy, count_tokens
from student_rag.utils.logger import logger
from student_rag.config import settings


@dataclass
class PageContent:
    """页面内容"""
    page_number: int
    text: str
    tables: List[List[List[str]]]
    metadata: Dict[str, Any]


class PDFParser:
    """
    PDF解析器

    功能：
    1. 提取文本内容，保留页码信息
    2. 识别表格结构
    3. 检测章节标题
    4. 清理页眉页脚噪声
    """

    def __init__(self):
        self.document_config = settings.document

    def parse(self, pdf_path: str | Path, document_id: Optional[str] = None) -> Document:
        """
        解析PDF文档

        Args:
            pdf_path: PDF文件路径
            document_id: 文档ID，如果不提供则自动生成

        Returns:
            Document对象，包含所有页面内容和元数据
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        document_id = document_id or str(uuid.uuid4())
        logger.info(f"开始解析PDF: {pdf_path.name} (ID: {document_id})")

        # 使用pdfplumber提取内容
        pages = self._extract_pages(pdf_path)

        # 使用pymupdf获取额外元数据
        metadata = self._extract_metadata(pdf_path)

        # 构建Document对象
        document = Document(
            id=document_id,
            name=pdf_path.stem,
            type=DocumentType.PDF,
            path=str(pdf_path),
            total_pages=len(pages),
            metadata=metadata
        )

        logger.info(f"PDF解析完成: {pdf_path.name}, 共{len(pages)}页")
        return document

    def _extract_pages(self, pdf_path: Path) -> List[PageContent]:
        """提取所有页面内容"""
        pages = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # 提取文本
                    text = page.extract_text() or ""
                    text = clean_text(text)

                    # 提取表格
                    tables = []
                    if self.document_config.extract_tables:
                        tables = page.extract_tables() or []

                    # 页面元数据
                    page_metadata = {
                        "width": page.width,
                        "height": page.height,
                        "has_tables": len(tables) > 0,
                    }

                    pages.append(PageContent(
                        page_number=page_num,
                        text=text,
                        tables=tables,
                        metadata=page_metadata
                    ))

                except Exception as e:
                    logger.warning(f"解析第{page_num}页时出错: {e}")
                    pages.append(PageContent(
                        page_number=page_num,
                        text="",
                        tables=[],
                        metadata={"error": str(e)}
                    ))

        return pages

    def _extract_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """使用pymupdf提取文档元数据"""
        try:
            doc = fitz.open(pdf_path)
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "page_count": len(doc),
            }
            doc.close()
            return metadata
        except Exception as e:
            logger.warning(f"提取元数据失败: {e}")
            return {}

    def parse_to_chunks(self, pdf_path: str | Path, document_id: Optional[str] = None) -> List[DocumentChunk]:
        """
        解析PDF并直接生成Chunks

        这是主要的入口方法，将PDF解析和Chunk切分整合在一起。
        """
        # 解析PDF
        pages = self._extract_pages(Path(pdf_path))
        document_id = document_id or str(uuid.uuid4())
        document_name = Path(pdf_path).stem

        # 生成Chunks
        chunks = []
        chunk_index = 0

        for page in pages:
            if not page.text.strip():
                continue

            # 提取层级信息
            chapter_title, section_title = extract_hierarchy(page.text)

            # 按段落切分页面
            paragraphs = self._split_page_to_paragraphs(page.text)

            for para_idx, paragraph in enumerate(paragraphs):
                # 如果段落太长，需要进一步切分
                sub_chunks = self._split_paragraph_to_chunks(
                    paragraph,
                    page.page_number,
                    para_idx,
                    document_id,
                    document_name,
                    chapter_title,
                    section_title
                )

                for chunk_content in sub_chunks:
                    chunk = self._create_chunk(
                        content=chunk_content,
                        document_id=document_id,
                        document_name=document_name,
                        page_number=page.page_number,
                        paragraph_number=para_idx,
                        chunk_index=chunk_index,
                        chapter_title=chapter_title,
                        section_title=section_title
                    )
                    chunks.append(chunk)
                    chunk_index += 1

        # 更新total_chunks
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)

        logger.info(f"生成Chunks完成: 共{len(chunks)}个chunks")
        return chunks

    def _split_page_to_paragraphs(self, text: str) -> List[str]:
        """将页面文本切分为段落"""
        # 按空行切分
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # 合并过短的段落（可能是标题）
        merged = []
        current = ""
        for para in paragraphs:
            if count_tokens(para) < 20 and current:
                # 可能是标题，合并到下一个段落
                current = current + "\n" + para
            else:
                if current:
                    merged.append(current)
                current = para

        if current:
            merged.append(current)

        return merged

    def _split_paragraph_to_chunks(
        self,
        paragraph: str,
        page_number: int,
        paragraph_number: int,
        document_id: str,
        document_name: str,
        chapter_title: str,
        section_title: str
    ) -> List[str]:
        """
        将段落切分为合适大小的chunks

        策略：
        1. 如果段落本身在目标大小范围内，直接作为一个chunk
        2. 如果段落过长，按句子边界切分
        3. 保持制度条文的完整性
        """
        tokens = count_tokens(paragraph)

        # 如果段落大小合适，直接返回
        if tokens <= self.document_config.chunk_size:
            return [paragraph]

        # 如果段落过长，按句子切分
        sentences = re.split(r'([。！？；])', paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 合并句子成chunks
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for i in range(0, len(sentences), 2):  # 步长2，跳过标点
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            full_sentence = sentence + punctuation

            sentence_tokens = count_tokens(full_sentence)

            if current_tokens + sentence_tokens > self.document_config.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = full_sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += full_sentence
                current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _create_chunk(
        self,
        content: str,
        document_id: str,
        document_name: str,
        page_number: int,
        paragraph_number: int,
        chunk_index: int,
        chapter_title: str = "",
        section_title: str = ""
    ) -> DocumentChunk:
        """创建DocumentChunk对象"""
        chunk_id = f"{document_id}_{page_number}_{chunk_index}"

        metadata = ChunkMetadata(
            document_id=document_id,
            document_name=document_name,
            document_type=DocumentType.PDF,
            page_number=page_number,
            paragraph_number=paragraph_number,
            start_pos=0,
            end_pos=len(content),
            chapter_title=chapter_title or None,
            section_title=section_title or None,
            hierarchy_level=0,
            chunk_index=chunk_index,
            total_chunks=1
        )

        return DocumentChunk(
            id=chunk_id,
            content=content,
            metadata=metadata,
            token_count=count_tokens(content),
            char_count=len(content)
        )


class SemanticChunker:
    """
    语义Chunk切分器

    专为制度类文档优化的切分策略：
    1. 保留章节结构
    2. 按语义边界切分（段落 > 句子）
    3. 智能重叠策略
    """

    def __init__(self):
        self.config = settings.document

    def create_chunks_with_overlap(
        self,
        chunks: List[DocumentChunk],
        overlap_ratio: float = 0.2
    ) -> List[DocumentChunk]:
        """
        为chunks添加重叠内容

        策略：
        - 每个chunk包含前一个chunk的最后20%内容
        - 保留章节标题在重叠区域
        """
        if len(chunks) <= 1:
            return chunks

        overlap_tokens = int(self.config.chunk_size * overlap_ratio)
        result = []

        for i, chunk in enumerate(chunks):
            new_chunk = chunk.model_copy()

            if i > 0:
                # 从前一个chunk获取重叠内容
                prev_chunk = chunks[i - 1]
                overlap_text = self._get_overlap_text(
                    prev_chunk.content,
                    overlap_tokens,
                    from_end=True
                )

                if overlap_text:
                    # 添加章节标题上下文
                    context = ""
                    if prev_chunk.metadata.chapter_title:
                        context = f"【上文来自：{prev_chunk.metadata.chapter_title}】\n"

                    new_chunk.content = context + overlap_text + "\n\n" + chunk.content
                    new_chunk.metadata.overlap_prev = True

            if i < len(chunks) - 1:
                new_chunk.metadata.overlap_next = True

            # 更新统计
            new_chunk.token_count = count_tokens(new_chunk.content)
            new_chunk.char_count = len(new_chunk.content)

            result.append(new_chunk)

        return result

    def _get_overlap_text(self, text: str, max_tokens: int, from_end: bool = True) -> str:
        """从文本中提取指定token数的重叠内容"""
        tokens = count_tokens(text)

        if tokens <= max_tokens:
            return text

        # 简单策略：按字符比例估算
        char_ratio = max_tokens / tokens
        char_count = int(len(text) * char_ratio)

        if from_end:
            # 从末尾取
            overlap = text[-char_count:]
            # 确保从句子边界开始
            sentence_starts = [m.start() for m in re.finditer(r'[。！？；]\s*', overlap)]
            if sentence_starts:
                first_sentence_end = min(sentence_starts)
                overlap = overlap[first_sentence_end + 1:]
            return overlap.strip()
        else:
            # 从开头取
            overlap = text[:char_count]
            # 确保在句子边界结束
            sentence_ends = [m.end() for m in re.finditer(r'[。！？；]', overlap)]
            if sentence_ends:
                last_sentence_end = max(sentence_ends)
                overlap = overlap[:last_sentence_end]
            return overlap.strip()
