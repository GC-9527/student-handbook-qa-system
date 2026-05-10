# 学生手册问答助手 - 架构文档

## 系统概述

本项目是一个企业级的基于PDF文档的RAG（检索增强生成）问答系统，专为制度类文档（如学生手册、规章制度等）优化。

## 核心功能

1. **文档切分**: 设计适合制度文档的Chunk策略
2. **向量检索**: 使用Embedding模型建立语义索引
3. **结果重排**: 集成ReRanker提升检索准确性
4. **引用溯源**: 提供可追溯的引用来源

## 技术架构

### 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户接口层                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   问答接口    │  │  对比查询接口  │  │  管理接口     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          RAG引擎层                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  答案生成    │  │  检索组装    │  │  ReRanker   │             │
│  │   (LLM)     │  │             │  │   重排序     │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         数据处理层                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  PDF解析器   │  │  Chunk切分器 │  │   Embedding │             │
│  │(pdfplumber) │──│ (语义切分)   │──│    编码     │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                           存储层                                 │
│  ┌─────────────────┐  ┌─────────────────┐                      │
│  │   向量索引(FAISS) │  │   元数据存储      │                      │
│  └─────────────────┘  └─────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 模块说明

#### 2.1 文档处理模块 (`core/document_processor.py`)

**职责**: 将PDF文档转换为可索引的Chunks

**关键设计**:
- **Chunk大小**: 800 tokens
  - 制度文档通常有完整的"条件-流程-结果"逻辑单元
  - 800 tokens足够容纳一个完整的制度条文
- **重叠策略**: 20% (160 tokens)
  - 保持跨chunk的上下文连贯性
  - 制度文档中条款经常相互引用
- **切分边界**: 段落 > 句子
  - 优先保持语义完整

**代码示例**:
```python
from student_rag.core import DocumentProcessor

processor = DocumentProcessor()
chunks = processor.process_pdf("student_handbook.pdf")
```

#### 2.2 Embedding模块 (`core/embeddings.py`)

**职责**: 将文本编码为向量表示

**模型选择**: BAAI/bge-large-zh-v1.5
- 中文语义理解优秀
- 开源可本地部署
- 1024维向量

**查询优化**:
```python
# BGE模型对查询添加指令前缀，提升检索效果
query = "为这个句子生成表示以用于检索相关文章：奖学金怎么评？"
```

#### 2.3 向量检索模块 (`core/retriever.py`)

**职责**: 基于向量相似度检索相关文档

**两阶段检索**:
1. **粗排**: Embedding向量检索（快速召回Top-30）
2. **精排**: ReRanker重排序（提升Top-5准确性）

#### 2.4 ReRanker模块 (`core/reranker.py`)

**职责**: 对检索结果进行精排

**原理对比**:

| 特性 | Embedding检索 | ReRanker |
|------|---------------|----------|
| 架构 | 双塔（分别编码） | 交叉编码器 |
| 速度 | 快（预计算） | 慢（实时计算） |
| 准确性 | 中等 | 高 |
| 适用场景 | 大规模召回 | Top-K精排 |

**使用方式**:
```python
from student_rag.core import Reranker

reranker = Reranker()
scores = reranker.rerank(query, passages)
```

#### 2.5 引用溯源模块 (`core/rag_engine.py`)

**职责**: 生成带引用的答案

**溯源粒度**:
- 文档名称
- 章节标题
- 页码
- 原文片段

**引用格式**:
```
奖学金评定标准包括学业成绩和综合素质两个方面[1]。

[1] 《学生手册》- 第一章 奖学金评定办法 (第2页)
```

## Chunk策略详解

### 为什么适合制度类文档？

**制度文档特点**:
1. 结构清晰：章节 → 条款 → 细则
2. 逻辑完整：单个条文包含"条件-流程-结果"
3. 引用频繁：条款之间相互引用

**策略设计**:

| 参数 | 值 | 说明 |
|------|-----|------|
| Chunk大小 | 800 tokens | 容纳完整制度条文 |
| 重叠比例 | 20% | 保持上下文连贯 |
| 切分边界 | 段落优先 | 保持语义完整 |
| 层级保留 | 章节标题继承 | 便于溯源 |

### 与其他文档类型的对比

| 文档类型 | 推荐Chunk大小 | 重叠比例 | 原因 |
|----------|--------------|----------|------|
| 制度文档 | 512-1024 | 20% | 完整逻辑单元 |
| 新闻文章 | 256-512 | 10% | 信息密度高 |
| 技术文档 | 512-768 | 15% | 代码块边界 |
| 小说文学 | 1024-2048 | 25% | 上下文依赖强 |

## ReRanker效果评估

### 评估指标

1. **Top-K重叠率**: 重排序前后Top-K结果的重叠程度
2. **NDCG**: 归一化折损累计增益
3. **MRR**: 平均倒数排名

### 预期效果

根据BGE论文和实际测试：
- Top-5准确率提升: 10-20%
- 召回率提升: 5-15%
- 延迟增加: 50-100ms（可接受范围）

## 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API密钥 | 必填 |
| `EMB_MODEL_NAME` | Embedding模型 | BAAI/bge-large-zh-v1.5 |
| `RET_RERANKER_ENABLED` | 启用ReRanker | true |
| `DOC_CHUNK_SIZE` | Chunk大小 | 800 |
| `DOC_CHUNK_OVERLAP` | 重叠大小 | 160 |

### 配置文件

配置文件位于 `student_rag/config/settings.py`，使用Pydantic Settings管理：

```python
from student_rag.config import settings

# 访问配置
chunk_size = settings.document.chunk_size
model_name = settings.embedding.model_name
```

## 性能优化

### 1. 索引优化

- 使用FAISS的IndexFlatIP（归一化向量的内积 = 余弦相似度）
- 支持增量添加向量
- 索引持久化到磁盘

### 2. 缓存策略

- Embedding结果缓存（内存 + 磁盘）
- 避免重复计算相同文本的向量

### 3. 批处理

- Embedding编码使用批处理
- ReRanker使用批处理

## 扩展性设计

### 1. 模块化接口

所有核心组件都定义了抽象接口，方便替换：

```python
# 可以替换为自定义Embedding模型
class CustomEmbeddingModel:
    def encode(self, texts):
        # 自定义实现
        pass

# 注入到检索器
retriever = VectorRetriever(embedding_model=CustomEmbeddingModel())
```

### 2. 多文档支持

支持同时索引多个PDF文档：

```python
pipeline.index_documents([
    "handbook_2023.pdf",
    "handbook_2024.pdf",
    "supplementary_rules.pdf"
])
```

### 3. 多种LLM后端

当前支持OpenAI，可扩展：
- Azure OpenAI
- 本地模型（Llama、ChatGLM等）
- 其他API（Claude、文心一言等）

## 部署建议

### 1. 开发环境

```bash
pip install -r requirements.txt
python demo.py
```

### 2. 生产环境

- 使用GPU加速Embedding和ReRanker
- 使用Redis缓存Embedding结果
- 使用消息队列处理文档索引任务

### 3. 容器化

```dockerfile
FROM python:3.10-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "demo.py"]
```

## 测试策略

### 单元测试

```bash
pytest tests/
```

### 集成测试

使用示例学生手册进行端到端测试：

```python
from demo import demo_basic_usage
demo_basic_usage()
```

### 效果评估

```python
from student_rag.core import RerankerEvaluator

evaluator = RerankerEvaluator()
results = evaluator.evaluate_batch(test_cases)
```
