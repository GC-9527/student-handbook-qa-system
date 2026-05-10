"""
RAG系统演示脚本
演示学生手册问答助手的完整功能：
1. 文档处理
2. 索引构建
3. 检索查询
4. 重排序对比
5. 引用溯源
"""

import os
import sys
from pathlib import Path

# 添加项目到路径
sys.path.insert(0, str(Path(__file__).parent))

from student_rag.core import RAGPipeline, RerankerEvaluator
from student_rag.core.reranker import print_rerank_comparison
from student_rag.utils.logger import logger


def demo_basic_usage():
    """基础使用演示"""
    print("\n" + "="*80)
    print("演示1: 基础使用")
    print("="*80)

    # 创建RAG流水线
    pipeline = RAGPipeline()

    # 检查示例PDF是否存在
    pdf_path = "data/pdfs/sample_student_handbook.pdf"
    if not Path(pdf_path).exists():
        print(f"示例PDF不存在: {pdf_path}")
        print("请先运行: python create_sample_handbook.py")
        return

    # 构建索引
    print("\n步骤1: 构建索引...")
    try:
        chunks = pipeline.build_index(pdf_path)
        print(f"索引构建完成: {len(chunks)}个chunks")
    except Exception as e:
        print(f"索引构建失败: {e}")
        print("提示: 需要设置OPENAI_API_KEY环境变量才能使用完整功能")
        return

    # 提问
    questions = [
        "奖学金怎么评？",
        "缓考流程是什么？",
        "转专业需要什么条件？",
        "考试违纪有什么后果？",
    ]

    print("\n步骤2: 问答演示")
    for question in questions:
        print(f"\n问题: {question}")
        try:
            response = pipeline.ask_with_sources(question)
            print(f"答案: {response.answer[:200]}...")
            print(f"引用: {len(response.citations)}个来源")
            print(f"耗时: 检索{response.retrieval_time_ms:.0f}ms, 生成{response.generation_time_ms:.0f}ms")
        except Exception as e:
            print(f"查询失败: {e}")


def demo_rerank_comparison():
    """重排序对比演示"""
    print("\n" + "="*80)
    print("演示2: ReRanker重排序效果对比")
    print("="*80)

    pipeline = RAGPipeline()

    pdf_path = "data/pdfs/sample_student_handbook.pdf"
    if not Path(pdf_path).exists():
        print(f"示例PDF不存在: {pdf_path}")
        return

    # 确保已索引
    if not pipeline.engine._initialized:
        print("\n构建索引...")
        try:
            pipeline.build_index(pdf_path)
        except Exception as e:
            print(f"索引构建失败: {e}")
            return

    # 对比查询
    query = "奖学金评定的成绩要求是什么？"
    print(f"\n查询: {query}")

    try:
        comparison = pipeline.engine.query_with_comparison(query, top_k=5)

        print("\n" + "-"*80)
        print("重排序前 (Embedding检索):")
        print("-"*80)
        for item in comparison['before_rerank']:
            print(f"  排名{item['rank']}: 分数={item['score']:.4f}")
            print(f"    内容: {item['content_preview'][:60]}...")
            print(f"    来源: {item['citation']}")

        print("\n" + "-"*80)
        print("重排序后 (ReRanker精排):")
        print("-"*80)
        for item in comparison['after_rerank']:
            print(f"  排名{item['rank']}: Embedding={item['embedding_score']:.4f}, "
                  f"ReRank={item['rerank_score']:.4f}")
            print(f"    内容: {item['content_preview'][:60]}...")
            print(f"    来源: {item['citation']}")

        print("\n" + "="*80)
        print(f"改善率: {comparison['improvement']:.1%}")
        print("="*80)

    except Exception as e:
        print(f"对比查询失败: {e}")


def demo_citation_traceability():
    """引用溯源演示"""
    print("\n" + "="*80)
    print("演示3: 引用溯源功能")
    print("="*80)

    pipeline = RAGPipeline()

    pdf_path = "data/pdfs/sample_student_handbook.pdf"
    if not Path(pdf_path).exists():
        print(f"示例PDF不存在: {pdf_path}")
        return

    if not pipeline.engine._initialized:
        print("\n构建索引...")
        try:
            pipeline.build_index(pdf_path)
        except Exception as e:
            print(f"索引构建失败: {e}")
            return

    query = "申请国家奖学金需要什么条件？"
    print(f"\n查询: {query}")

    try:
        response = pipeline.ask_with_sources(query)

        print("\n" + "-"*80)
        print("生成的答案:")
        print("-"*80)
        print(response.answer)

        print("\n" + "-"*80)
        print("引用来源:")
        print("-"*80)
        for i, (citation, source) in enumerate(zip(response.citations, response.sources), 1):
            print(f"\n[{i}] {citation}")
            print(f"    内容片段: {source.content[:150]}...")
            print(f"    页码: 第{source.metadata.page_number}页")
            if source.metadata.chapter_title:
                print(f"    章节: {source.metadata.chapter_title}")

    except Exception as e:
        print(f"查询失败: {e}")


def demo_chunk_strategy():
    """Chunk策略说明"""
    print("\n" + "="*80)
    print("演示4: Chunk切分策略说明")
    print("="*80)

    from student_rag.core import DocumentProcessor
    from student_rag.config import settings

    print("\n当前配置:")
    print(f"  Chunk大小: {settings.document.chunk_size} tokens")
    print(f"  重叠大小: {settings.document.chunk_overlap} tokens")
    print(f"  重叠比例: {settings.document.chunk_overlap/settings.document.chunk_size:.0%}")

    pdf_path = "data/pdfs/sample_student_handbook.pdf"
    if not Path(pdf_path).exists():
        print(f"\n示例PDF不存在: {pdf_path}")
        return

    print(f"\n处理文档: {pdf_path}")

    processor = DocumentProcessor()
    chunks = processor.process_pdf(pdf_path, enable_overlap=True)

    stats = processor.get_chunk_statistics(chunks)

    print("\n处理结果统计:")
    print(f"  总Chunks: {stats['total_chunks']}")
    print(f"  总Tokens: {stats['total_tokens']}")
    print(f"  平均Chunk大小: {stats['avg_tokens']:.0f} tokens")
    print(f"  最小Chunk: {stats['min_tokens']} tokens")
    print(f"  最大Chunk: {stats['max_tokens']} tokens")

    print("\n示例Chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n--- Chunk {i} ---")
        print(f"  ID: {chunk.id}")
        print(f"  页码: 第{chunk.metadata.page_number}页")
        print(f"  Token数: {chunk.token_count}")
        print(f"  章节: {chunk.metadata.chapter_title or 'N/A'}")
        print(f"  内容预览: {chunk.content[:100]}...")


def print_system_architecture():
    """打印系统架构说明"""
    print("\n" + "="*80)
    print("学生手册问答助手 - 系统架构")
    print("="*80)

    architecture = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                           学生手册问答助手 (RAG系统)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         用户接口层                                    │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│  │  │   问答接口    │  │  对比查询接口  │  │  管理接口     │              │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         RAG引擎层                                     │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │                      答案生成 (LLM)                           │  │   │
│  │  │  - 引用格式化输出                                             │  │   │
│  │  │  - 溯源信息整合                                               │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │                    检索结果组装                                │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │  ┌─────────────────────┐    ┌─────────────────────┐               │   │
│  │  │   ReRanker重排序     │◄───│    向量检索          │               │   │
│  │  │  (交叉编码器精排)     │    │  (Embedding相似度)   │               │   │
│  │  └─────────────────────┘    └─────────────────────┘               │   │
│  │                                                         │         │   │
│  └─────────────────────────────────────────────────────────┼─────────┘   │
│                                                            │              │
│                                                            ▼              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         数据处理层                                    │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │   │
│  │  │   PDF解析器      │──►│   Chunk切分器    │──►│   Embedding编码  │     │   │
│  │  │  (pdfplumber)   │  │  (语义边界切分)   │  │  (BGE-large-zh) │     │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘     │   │
│  │                                                                             │
│  │  Chunk策略:                                                                 │
│  │  - 大小: 800 tokens (适合制度文档完整表达)                                    │
│  │  - 重叠: 160 tokens (20%，保持上下文连贯)                                     │
│  │  - 边界: 段落 > 句子 (优先保持语义完整)                                       │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         存储层                                        │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                          │   │
│  │  │   向量索引(FAISS) │  │   元数据存储      │                          │   │
│  │  │  - 余弦相似度    │  │  - Chunk内容     │                          │   │
│  │  │  - 高效检索      │  │  - 页码/章节信息  │                          │   │
│  │  └─────────────────┘  └─────────────────┘                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

核心技术特点:
1. 文档切分: 语义边界 + 20%重叠，专为制度文档优化
2. 向量检索: BGE-large-zh-v1.5，中文语义理解优秀
3. 重排序: BGE-Reranker，两阶段检索提升准确性
4. 引用溯源: 页码 + 章节 + 原文片段，可追溯验证
"""
    print(architecture)


def main():
    """主函数"""
    print("\n" + "="*80)
    print("学生手册问答助手 - RAG系统演示")
    print("="*80)

    # 检查环境
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  警告: 未设置OPENAI_API_KEY环境变量")
        print("   部分功能（LLM生成）将不可用")
        print("   设置方法: export OPENAI_API_KEY='your-api-key'")

    # 创建示例PDF
    pdf_path = Path("data/pdfs/sample_student_handbook.pdf")
    if not pdf_path.exists():
        print("\n创建示例学生手册...")
        try:
            from create_sample_handbook import create_sample_handbook
            create_sample_handbook()
        except ImportError:
            print("   需要安装reportlab: pip install reportlab")
            return

    # 显示菜单
    while True:
        print("\n" + "="*80)
        print("请选择演示项目:")
        print("  1. 系统架构说明")
        print("  2. Chunk切分策略说明")
        print("  3. 基础问答演示")
        print("  4. ReRanker重排序对比")
        print("  5. 引用溯源演示")
        print("  0. 退出")
        print("="*80)

        choice = input("\n输入选项 (0-5): ").strip()

        if choice == "0":
            print("\n感谢使用，再见！")
            break
        elif choice == "1":
            print_system_architecture()
        elif choice == "2":
            demo_chunk_strategy()
        elif choice == "3":
            demo_basic_usage()
        elif choice == "4":
            demo_rerank_comparison()
        elif choice == "5":
            demo_citation_traceability()
        else:
            print("\n无效选项，请重新输入")


if __name__ == "__main__":
    main()
