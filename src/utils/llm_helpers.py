# src/utils/llm_helpers.py

from openai import OpenAI
from src.utils.config import API_KEY, API_BASE, DEFAULT_MODEL

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
)

def call_llm_api(messages,max_tokens = 1000,temperature = 0.7):
    """
    调用 LLM API 并返回响应内容。

    参数:
    messages (list): 要发送到 API 的消息列表。
    max_tokens (int): 最大生成的令牌数。
    temperature (float): 控制生成文本的随机性。
    
    返回:
    str: LLM 返回的响应内容。
    """
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()
