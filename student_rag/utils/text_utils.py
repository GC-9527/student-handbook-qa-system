"""
文本处理工具

提供文本切分、清理、统计等功能。
"""

import re
from typing import List, Tuple
import tiktoken


def get_tokenizer(model_name: str = "gpt-3.5-turbo") -> tiktoken.Encoding:
    """获取tokenizer"""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """计算文本的token数量"""
    if not text:
        return 0
    tokenizer = get_tokenizer(model_name)
    return len(tokenizer.encode(text))


def clean_text(text: str) -> str:
    """
    清理文本

    - 移除多余的空白字符
    - 统一换行符
    - 移除页眉页脚常见的噪声
    """
    if not text:
        return ""

    # 统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 移除多余的空白行（超过2个连续换行符）
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 移除行首行尾的多余空格
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # 移除多余的空格（超过2个连续空格）
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def split_by_sentences(text: str) -> List[str]:
    """
    按句子切分文本

    支持中文和英文句子边界识别。
    """
    if not text:
        return []

    # 中文句子结束符
    chinese_endings = r'[。！？；]'
    # 英文句子结束符（后跟空格或大写）
    english_endings = r'[.!?;](?=\s+[A-Z])'

    # 合并模式
    pattern = f'({chinese_endings}|{english_endings})'

    # 切分
    parts = re.split(pattern, text)

    # 合并句子及其结束符
    sentences = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and re.match(pattern, parts[i + 1]):
            sentences.append(parts[i] + parts[i + 1])
            i += 2
        else:
            if parts[i].strip():
                sentences.append(parts[i])
            i += 1

    return [s.strip() for s in sentences if s.strip()]


def split_by_paragraphs(text: str) -> List[str]:
    """按段落切分文本"""
    if not text:
        return []

    # 按两个及以上换行符切分
    paragraphs = re.split(r'\n{2,}', text)
    return [p.strip() for p in paragraphs if p.strip()]


def extract_hierarchy(text: str) -> Tuple[str, str]:
    """
    提取文本中的层级标题

    识别常见的标题格式：
    - 第X章/第X节
    - 数字编号（1. 1.1 1.1.1）
    - 中文编号（一、（一））
    """
    lines = text.split('\n')
    chapter_title = None
    section_title = None

    # 章节标题模式
    chapter_patterns = [
        r'^第[一二三四五六七八九十\d]+章',  # 第一章
        r'^第[一二三四五六七八九十\d]+节',  # 第一节
        r'^\d+\.\s+',  # 1. 标题
    ]

    # 小节标题模式
    section_patterns = [
        r'^[（(][一二三四五六七八九十\d]+[）)]',  # （一）
        r'^\d+\.\d+\.?\s+',  # 1.1 标题
    ]

    for line in lines[:5]:  # 检查前5行
        line = line.strip()
        if not line:
            continue

        # 检查章节标题
        for pattern in chapter_patterns:
            if re.match(pattern, line):
                chapter_title = line
                break

        # 检查小节标题
        for pattern in section_patterns:
            if re.match(pattern, line):
                section_title = line
                break

    return chapter_title or "", section_title or ""


def truncate_text(text: str, max_tokens: int, model_name: str = "gpt-3.5-turbo") -> str:
    """截断文本到指定token数"""
    if not text:
        return ""

    tokenizer = get_tokenizer(model_name)
    tokens = tokenizer.encode(text)

    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens)


def merge_small_chunks(chunks: List[str], min_tokens: int, max_tokens: int) -> List[str]:
    """
    合并过小的chunks

    将小于min_tokens的chunk与相邻chunk合并，直到达到min_tokens。
    """
    if not chunks:
        return []

    merged = []
    current = ""

    for chunk in chunks:
        chunk_tokens = count_tokens(chunk)
        current_tokens = count_tokens(current)

        if chunk_tokens < min_tokens and current_tokens < min_tokens:
            # 合并小chunks
            current = current + "\n\n" + chunk if current else chunk
            if count_tokens(current) >= max_tokens:
                merged.append(current)
                current = ""
        else:
            if current:
                merged.append(current)
            current = chunk

    if current:
        merged.append(current)

    return merged


def calculate_overlap(text1: str, text2: str) -> float:
    """
    计算两个文本的重叠度

    返回重叠字符数占较短文本的比例。
    """
    if not text1 or not text2:
        return 0.0

    # 简单的字符集重叠计算
    set1 = set(text1)
    set2 = set(text2)

    intersection = set1 & set2
    union = set1 | set2

    if not union:
        return 0.0

    return len(intersection) / len(union)
