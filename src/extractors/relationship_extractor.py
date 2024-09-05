# src/extractors/relationship_extractor.py
from src.utils.llm_helpers import call_llm_api
from src.utils.message_templates import get_entities_and_relationships_template
from src.utils.parser import parse_entities, parse_relationships
import pandas as pd

def extract_entities_and_relationships(chunk):
    """
    使用 LLM 从 Chunk 对象中提取实体和关系。
    
    参数:
    chunk (Chunk): 包含文本块的 Chunk 对象。
    
    返回:
    tuple: 包含实体和关系的 DataFrame。
    """
    # 获取消息模板，使用 chunk 的 content 属性作为输入
    messages = get_entities_and_relationships_template(chunk.content)
    
    # 调用 LLM API 并获取内容
    content = call_llm_api(messages, max_tokens=1500, temperature=0.7)
    
    if not content:
        return pd.DataFrame(), pd.DataFrame()  # 返回空的 DataFrame
    
    # 提取实体和关系
    df_entities = parse_entities(content)
    df_relationships = parse_relationships(content, df_entities)
    
    return df_entities, df_relationships