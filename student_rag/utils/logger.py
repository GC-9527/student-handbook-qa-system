"""
日志工具

统一的日志管理，支持结构化日志和不同级别的日志记录。
"""

import sys
from pathlib import Path
from loguru import logger as _logger

from student_rag.config import settings

# 移除默认处理器
_logger.remove()

# 添加控制台处理器
_logger.add(
    sys.stdout,
    level=settings.log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
    colorize=True
)

# 添加文件处理器（按天轮转）
log_path = Path(__file__).parent.parent.parent / "logs"
log_path.mkdir(exist_ok=True)

_logger.add(
    log_path / "student_rag_{time:YYYY-MM-DD}.log",
    rotation="00:00",  # 每天午夜轮转
    retention="30 days",  # 保留30天
    level="DEBUG",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

# 导出logger
logger = _logger


def log_retrieval_result(query: str, chunks: list, stage: str = "retrieval"):
    """记录检索结果"""
    logger.info(f"[{stage}] Query: {query}")
    logger.info(f"[{stage}] Retrieved {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:5], 1):  # 只记录前5个
        score = getattr(chunk, 'similarity_score', 'N/A')
        rerank_score = getattr(chunk, 'rerank_score', None)
        citation = chunk.metadata.get_citation()
        if rerank_score:
            logger.debug(f"  [{i}] Score: {score:.4f} → {rerank_score:.4f} | {citation}")
        else:
            logger.debug(f"  [{i}] Score: {score:.4f} | {citation}")
