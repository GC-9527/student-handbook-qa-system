"""
LLM封装模块

提供统一的LLM调用接口，支持多种后端（OpenAI、Azure、Ollama本地模型等）。
"""

import time
import json
from typing import List, Optional, Dict, Any, Iterator
from abc import ABC, abstractmethod
import requests

from openai import OpenAI

from student_rag.config import settings
from student_rag.utils.logger import logger


class BaseLLM(ABC):
    """LLM基类"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """生成文本"""
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """对话模式"""
        pass

    def generate_stream(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> Iterator[str]:
        """流式生成文本"""
        raise NotImplementedError("当前模型不支持流式输出")


class OpenAILLM(BaseLLM):
    """OpenAI LLM封装"""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None
    ):
        self.config = settings.llm
        self.api_key = api_key or self.config.api_key
        self.base_url = base_url or self.config.base_url
        self.model = model or self.config.model_name

        if not self.api_key:
            raise ValueError("OpenAI API密钥未设置")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        logger.info(f"OpenAILLM初始化: model={self.model}")

    def generate(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """生成文本"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature, max_tokens)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """对话模式"""
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=self.config.top_p
            )

            elapsed = (time.time() - start_time) * 1000
            content = response.choices[0].message.content

            logger.debug(f"LLM生成完成: {len(content)}字符, 耗时{elapsed:.1f}ms")

            return content

        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise


class OllamaLLM(BaseLLM):
    """
    Ollama本地LLM封装

    支持本地部署的大模型，如 gemma4:e4b、llama3、qwen 等。
    Ollama 默认地址: http://localhost:11434
    """

    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        system_prompt: str = None,
        timeout: int = 300
    ):
        self.config = settings.llm
        self.model = model or self.config.ollama_model
        self.base_url = base_url or self.config.ollama_base_url
        self.system_prompt = system_prompt
        self.timeout = timeout

        logger.info(f"OllamaLLM初始化: model={self.model}, base_url={self.base_url}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        stream: bool = False
    ):
        """
        Ollama对话模式

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大token数
            stream: 是否使用流式输出

        Returns:
            生成的文本内容（非流式）或生成器（流式）
        """
        if stream:
            return self._chat_stream(messages, temperature, max_tokens)
        else:
            return self._chat_non_stream(messages, temperature, max_tokens)

    def _chat_non_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """非流式对话"""
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        start_time = time.time()

        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }

            if self.system_prompt:
                payload["system"] = self.system_prompt

            url = f"{self.base_url}/api/chat"
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                stream=False
            )

            response.raise_for_status()
            result = response.json()
            content = result.get("message", {}).get("content", "")

            elapsed = (time.time() - start_time) * 1000
            logger.debug(f"Ollama生成完成: {len(content)}字符, 耗时{elapsed:.1f}ms")

            return content

        except requests.exceptions.Timeout:
            logger.error(f"Ollama请求超时: {self.timeout}秒")
            raise TimeoutError(f"Ollama模型响应超时，请检查模型是否正在运行")
        except requests.exceptions.ConnectionError:
            logger.error(f"Ollama连接失败: {self.base_url}")
            raise ConnectionError(
                f"无法连接到Ollama服务 ({self.base_url})。"
                f"请确保Ollama正在运行，可以尝试: ollama run {self.model}"
            )
        except Exception as e:
            logger.error(f"Ollama调用失败: {e}")
            raise

    def _chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ) -> Iterator[str]:
        """流式对话"""
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        start_time = time.time()

        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }

            if self.system_prompt:
                payload["system"] = self.system_prompt

            url = f"{self.base_url}/api/chat"
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                stream=True
            )

            response.raise_for_status()

            full_content = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data:
                        content = data["message"].get("content", "")
                        full_content += content
                        yield content

            elapsed = (time.time() - start_time) * 1000
            logger.debug(f"Ollama流式生成完成: {len(full_content)}字符, 耗时{elapsed:.1f}ms")

        except requests.exceptions.Timeout:
            logger.error(f"Ollama请求超时: {self.timeout}秒")
            raise TimeoutError(f"Ollama模型响应超时，请检查模型是否正在运行")
        except requests.exceptions.ConnectionError:
            logger.error(f"Ollama连接失败: {self.base_url}")
            raise ConnectionError(
                f"无法连接到Ollama服务 ({self.base_url})。"
                f"请确保Ollama正在运行，可以尝试: ollama run {self.model}"
            )
        except Exception as e:
            logger.error(f"Ollama调用失败: {e}")
            raise

    def generate(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """生成文本"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature, max_tokens)

    def generate_stream(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> Iterator[str]:
        """流式生成文本"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature, max_tokens, stream=True)

    def list_models(self) -> List[Dict[str, Any]]:
        """
        列出Ollama可用的模型

        Returns:
            模型列表
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            result = response.json()
            return result.get("models", [])
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []

    def pull_model(self, model: str = None) -> bool:
        """
        拉取模型

        Args:
            model: 模型名称，不指定则使用当前配置的模型

        Returns:
            是否成功
        """
        model = model or self.model

        try:
            url = f"{self.base_url}/api/pull"
            response = requests.post(
                url,
                json={"name": model},
                timeout=3600  # 1小时超时
            )
            response.raise_for_status()
            logger.info(f"模型 {model} 拉取成功")
            return True
        except Exception as e:
            logger.error(f"模型拉取失败: {e}")
            return False


class LLMFactory:
    """LLM工厂"""

    @staticmethod
    def create(provider: str = None, **kwargs) -> BaseLLM:
        """
        创建LLM实例

        Args:
            provider: 提供商类型 (openai/ollama/azure)
            **kwargs: 额外的配置参数

        Returns:
            LLM实例
        """
        provider = provider or settings.llm.provider

        if provider == "openai":
            return OpenAILLM(**kwargs)
        elif provider == "ollama":
            return OllamaLLM(**kwargs)
        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")


# RAG提示词模板
RAG_PROMPT_TEMPLATE = """你是一个专业的学生手册问答助手。请基于以下提供的参考资料回答问题。

## 参考资料
{context}

## 用户问题
{question}

## 回答要求
1. 基于参考资料回答，不要添加参考资料中没有的信息
2. 如果参考资料不足以回答问题，请明确说明"根据现有资料无法回答"
3. 回答要准确、简洁、有条理
4. 在回答中引用参考资料时，请使用脚注格式标记来源，如[1]、[2]

## 引用格式示例
奖学金评定标准包括学业成绩和综合素质两个方面[1]。申请流程为：学生申请→学院审核→学校审批[2]。

请用中文回答："""

CITATION_PROMPT_TEMPLATE = """请为以下回答添加引用标记。

## 参考资料
{context}

## 原始回答
{answer}

## 要求
1. 在回答中适当位置添加引用标记[1]、[2]等
2. 每个引用标记对应参考资料中的编号
3. 确保引用准确，不要添加不存在的引用
4. 返回完整的、带有引用标记的回答

请直接返回答复："""


def build_rag_prompt(
    question: str,
    contexts: List[str],
    citations: List[str] = None
) -> str:
    """
    构建RAG提示词

    Args:
        question: 用户问题
        contexts: 检索到的上下文列表
        citations: 引用信息列表

    Returns:
        完整的提示词
    """
    # 构建上下文部分
    context_parts = []
    for i, ctx in enumerate(contexts, 1):
        citation = citations[i-1] if citations and i <= len(citations) else ""
        context_parts.append(f"[{i}] {citation}\n{ctx}\n")

    context = "\n".join(context_parts)

    return RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )


def format_citations(chunks: List[Any]) -> List[str]:
    """
    格式化引用信息

    从chunks中提取引用信息，格式化为：
    《文档名》- 章节标题 (第X页)
    """
    citations = []
    for chunk in chunks:
        if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'get_citation'):
            citations.append(chunk.metadata.get_citation())
        else:
            citations.append("")
    return citations
