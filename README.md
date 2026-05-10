# 学生手册问答助手 - 企业级RAG系统

基于PDF文档的检索增强生成(RAG)问答系统，专为制度类文档优化。支持文档切分、向量检索、ReRanker重排序、引用溯源等企业级功能。

## 功能特性

### 1. 智能文档切分
- **Chunk大小**: 800 tokens（适合制度文档完整表达）
- **重叠策略**: 20%（保持上下文连贯性）
- **语义边界**: 按段落/句子切分，保持语义完整

### 2. 向量检索
- **Embedding模型**: BAAI/bge-large-zh-v1.5（中文语义理解优秀）
- **向量数据库**: FAISS（高效相似度检索）
- **相似度度量**: 余弦相似度

### 3. ReRanker重排序
- **两阶段检索**: Embedding粗排 + ReRanker精排
- **模型**: BGE-Reranker（交叉编码器）
- **效果提升**: Top-5准确率提升10-20%

### 4. 引用溯源
- **溯源粒度**: 文档 → 章节 → 页码 → 原文片段
- **引用格式**: 脚注式标注（如[1]、[2]）
- **可追溯性**: 支持验证答案来源

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，设置 OPENAI_API_KEY
```

### 运行演示

```bash
python demo.py
```

演示菜单：
1. 系统架构说明
2. Chunk切分策略说明
3. 基础问答演示
4. ReRanker重排序对比
5. 引用溯源演示

## 使用示例

### 基础用法

```python
from student_rag.core import RAGPipeline

# 创建流水线
pipeline = RAGPipeline()

# 构建索引
pipeline.build_index("student_handbook.pdf")

# 提问
answer = pipeline.ask("奖学金怎么评？")
print(answer)
```

### 带引用的问答

```python
# 获取答案和来源
response = pipeline.ask_with_sources("缓考流程是什么？")

print("答案:", response.answer)
print("\n引用来源:")
for i, citation in enumerate(response.citations, 1):
    print(f"[{i}] {citation}")
```

### ReRanker对比

```python
# 对比重排序前后的结果
comparison = pipeline.engine.query_with_comparison(
    "奖学金评定的成绩要求是什么？",
    top_k=5
)

print("重排序前:", comparison['before_rerank'])
print("重排序后:", comparison['after_rerank'])
print("改善率:", comparison['improvement'])
```

## 项目结构

```
student_rag/
├── student_rag/          # 主代码目录
│   ├── core/             # 核心模块
│   │   ├── document_processor.py   # 文档处理
│   │   ├── embeddings.py           # Embedding模型
│   │   ├── vector_store.py         # 向量存储
│   │   ├── retriever.py            # 检索器
│   │   ├── reranker.py             # ReRanker
│   │   ├── llm.py                  # LLM封装
│   │   └── rag_engine.py           # RAG引擎
│   ├── models/           # 数据模型
│   │   └── schemas.py    # Pydantic模型
│   ├── config/           # 配置管理
│   │   └── settings.py   # 配置类
│   └── utils/            # 工具函数
│       ├── logger.py     # 日志工具
│       └── text_utils.py # 文本处理
├── data/                 # 数据目录
│   ├── pdfs/            # PDF文档
│   └── index/           # 向量索引
├── tests/               # 测试目录
├── create_sample_handbook.py  # 创建示例PDF
├── demo.py              # 演示脚本
├── requirements.txt     # 依赖列表
└── README.md           # 本文件
```

## Chunk策略说明

### 为什么这样设计？

**制度文档特点**:
- 结构清晰：章节 → 条款 → 细则
- 逻辑完整：单个条文通常包含"条件-流程-结果"
- 引用频繁：条款之间相互引用

**策略参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| Chunk大小 | 800 tokens | 容纳完整制度条文 |
| 重叠比例 | 20% (160 tokens) | 保持上下文连贯 |
| 切分边界 | 段落优先 | 保持语义完整 |
| 层级保留 | 章节标题继承 | 便于溯源 |

### 与其他文档类型的对比

| 文档类型 | 推荐Chunk大小 | 重叠比例 | 原因 |
|----------|--------------|----------|------|
| 制度文档 | 512-1024 | 20% | 完整逻辑单元 |
| 新闻文章 | 256-512 | 10% | 信息密度高 |
| 技术文档 | 512-768 | 15% | 代码块边界 |
| 小说文学 | 1024-2048 | 25% | 上下文依赖强 |

## ReRanker原理

### 为什么需要ReRanker？

**Embedding检索**（双塔架构）:
- 优点：速度快，可预计算文档向量
- 缺点：查询和文档无交互，可能错过深层语义关联

**ReRanker**（交叉编码器）:
- 优点：查询和文档一起输入，捕捉细粒度交互
- 缺点：需要实时计算，速度较慢

**组合策略**（两阶段检索）:
1. Embedding检索：快速召回Top-30候选
2. ReRanker精排：提升Top-5准确性

### 效果对比

```
查询: 奖学金评定的成绩要求是什么？

重排序前 (Embedding检索):
  排名1: 分数=0.8234 - 第三条 评定条件...
  排名2: 分数=0.8012 - 第五条 评定程序...
  排名3: 分数=0.7856 - 第四条 成绩要求...  <-- 实际最相关

重排序后 (ReRanker精排):
  排名1: ReRank=0.9123 - 第四条 成绩要求...  <-- 提升到第1
  排名2: ReRank=0.8567 - 第三条 评定条件...
  排名3: ReRank=0.8234 - 第五条 评定程序...
```

## 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API密钥 | 必填 |
| `EMB_MODEL_NAME` | Embedding模型 | BAAI/bge-large-zh-v1.5 |
| `EMB_DEVICE` | 运行设备 | cpu |
| `RET_RERANKER_ENABLED` | 启用ReRanker | true |
| `RET_RERANKER_MODEL` | ReRanker模型 | BAAI/bge-reranker-large |
| `DOC_CHUNK_SIZE` | Chunk大小 | 800 |
| `DOC_CHUNK_OVERLAP` | 重叠大小 | 160 |
| `DEBUG` | 调试模式 | false |

### 配置文件

配置文件位于 `student_rag/config/settings.py`，使用Pydantic Settings管理。

```python
from student_rag.config import settings

# 访问配置
chunk_size = settings.document.chunk_size
model_name = settings.embedding.model_name
```

## 技术栈

- **Python**: 3.10+
- **PDF解析**: pdfplumber, pymupdf
- **Embedding**: sentence-transformers (BGE-large-zh)
- **向量检索**: FAISS
- **ReRanker**: BGE-Reranker
- **LLM**: OpenAI API
- **配置**: Pydantic Settings
- **日志**: loguru

## 进阶使用

### 自定义Embedding模型

```python
from student_rag.core import VectorRetriever, EmbeddingModel

class CustomEmbeddingModel(EmbeddingModel):
    def encode(self, texts, **kwargs):
        # 自定义编码逻辑
        pass

retriever = VectorRetriever(embedding_model=CustomEmbeddingModel())
```

### 批量处理多个文档

```python
pipeline.index_documents([
    "handbook_2023.pdf",
    "handbook_2024.pdf",
    "supplementary_rules.pdf"
])
```

### 保存和加载索引

```python
# 保存索引
pipeline.engine.save_index("path/to/index")

# 加载索引
pipeline.engine.load_index("path/to/index")
```

## 架构文档

详细架构说明请参见 [ARCHITECTURE.md](ARCHITECTURE.md)。

## 许可证

MIT License
