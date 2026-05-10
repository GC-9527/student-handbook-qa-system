"""
Ollama连接测试脚本

用于验证Ollama服务和大模型是否正常工作。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from student_rag.core.llm import LLMFactory
from student_rag.utils.logger import logger


def test_ollama_connection():
    """测试Ollama连接"""
    print("\n" + "="*60)
    print("Ollama连接测试")
    print("="*60)

    # 创建Ollama LLM实例
    try:
        llm = LLMFactory.create(provider="ollama")
        print(f"\n[OK] Ollama LLM创建成功")
        print(f"  模型: {llm.model}")
        print(f"  地址: {llm.base_url}")
    except Exception as e:
        print(f"\n[FAIL] Ollama LLM创建失败: {e}")
        return False

    # 测试列出可用模型
    print("\n" + "-"*60)
    print("可用模型列表:")
    print("-"*60)
    models = llm.list_models()
    if models:
        for m in models:
            name = m.get("name", "Unknown")
            size = m.get("size", 0)
            size_gb = size / (1024**3) if size else 0
            print(f"  - {name} ({size_gb:.1f} GB)")
    else:
        print("  无法获取模型列表（可能Ollama服务未正常运行）")

    # 测试简单的对话
    print("\n" + "-"*60)
    print("对话测试:")
    print("-"*60)
    test_prompt = "请用一句话介绍自己"

    print(f"\n问题: {test_prompt}")
    print("回答: ", end="", flush=True)

    try:
        answer = llm.generate(test_prompt, temperature=0.7, max_tokens=200)
        print(answer)
        print(f"\n[OK] 对话测试成功")
        return True
    except ConnectionError as e:
        print(f"\n[ERROR] 连接错误: {e}")
        print("\n请确保:")
        print("  1. Ollama已安装并正在运行")
        print("  2. 模型 gemma4:e4b 已下载 (如果没有，运行: ollama pull gemma4:e4b)")
        print("  3. Ollama服务地址正确 (默认: http://localhost:11434)")
        return False
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        return False


def test_ollama_embedding():
    """测试Ollama Embedding"""
    print("\n" + "-"*60)
    print("Ollama Embedding测试:")
    print("-"*60)

    from student_rag.core.embeddings import OllamaEmbedding

    try:
        embed = OllamaEmbedding()
        print(f"  模型: {embed.model}")
        print(f"  地址: {embed.base_url}")

        # 测试编码
        test_text = "这是一个测试文本"
        print(f"\n  测试文本: {test_text}")

        result = embed.encode(test_text)
        print(f"  向量维度: {len(result[0])}")
        print(f"\n[OK] Ollama Embedding测试成功")
        return True
    except Exception as e:
        print(f"\n[FAIL] Ollama Embedding测试失败: {e}")
        print("\n请确保:")
        print("  1. Ollama版本支持embedding API (需要v0.1.38+)")
        print("  2. 已安装embedding模型: ollama pull mxbai-embed-large")
        return False


def test_with_rag():
    """使用RAG系统测试Ollama"""
    print("\n" + "="*60)
    print("RAG + Ollama 集成测试")
    print("="*60)

    # 检查示例PDF
    pdf_path = Path("data/pdfs/sample_student_handbook.pdf")
    if not pdf_path.exists():
        print(f"\n示例PDF不存在，跳过RAG测试")
        print("请先运行: python create_sample_handbook.py")
        return False

    from student_rag.core import RAGPipeline

    print(f"\n使用Embedding模型: Ollama (mxbai-embed-large)")
    print(f"使用LLM模型: Lusizo/qwen2.5-7b-instruct-1m:latest")
    print("构建索引...")

    try:
        pipeline = RAGPipeline()
        chunks = pipeline.build_index(pdf_path)
        print(f"[OK] 索引构建成功: {len(chunks)} chunks")

        # 提问
        question = "奖学金怎么评？"
        print(f"\n问题: {question}")

        response = pipeline.ask_with_sources(question)
        print(f"\n回答:\n{response.answer}")
        print(f"\n来源数量: {len(response.citations)}")
        print(f"检索耗时: {response.retrieval_time_ms:.0f}ms")
        print(f"生成耗时: {response.generation_time_ms:.0f}ms")

        return True

    except Exception as e:
        print(f"\n[FAIL] RAG测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 先测试Ollama连接
    llm_success = test_ollama_connection()

    if llm_success:
        # 测试Ollama Embedding
        print("\n" + "="*60)
        embed_success = test_ollama_embedding()

        if embed_success:
            # 再测试RAG集成
            print("\n" + "="*60)
            test_with_rag()
        else:
            print("\n请先解决Ollama Embedding问题")
    else:
        print("\n请先解决Ollama连接问题，再进行后续测试")
