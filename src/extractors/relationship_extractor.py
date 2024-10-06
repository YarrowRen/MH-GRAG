# src/extractors/relationship_extractor.py
import pandas as pd
from typing import Tuple

from src.utils.llm_helpers import call_llm_api
from src.utils.message_templates import get_entities_and_relationships_template
from src.utils.parser import parse_entities, parse_relationships
import pandas as pd

def extract_entities_and_relationships(chunk) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

import pandas as pd

def extract_entities_and_relationships_from_df(df: pd.DataFrame, text_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    使用 LLM 从 DataFrame 中的每一行文本提取实体和关系。
    
    参数:
    df (DataFrame): 包含文本数据的 DataFrame 对象。
    text_column (str): DataFrame 中包含文本块的列的列名。
    
    返回:
    tuple: 包含实体和关系的 DataFrame。
    """
    # 用于存储所有实体和关系
    all_entities = []
    all_relationships = []
    
    # 遍历 DataFrame 的每一行
    for _, row in df.iterrows():
        # 获取消息模板，使用 DataFrame 中指定列的文本作为输入
        messages = get_entities_and_relationships_template(row[text_column])
        
        # 调用 LLM API 并获取内容
        content = call_llm_api(messages, max_tokens=1500, temperature=0.7)
        
        if not content:
            continue  # 如果没有内容，则跳过此行
        
        # 提取实体和关系
        df_entities = parse_entities(content)
        df_relationships = parse_relationships(content, df_entities)
        
        # 添加到结果列表
        all_entities.append(df_entities)
        all_relationships.append(df_relationships)
    
    # 将所有实体和关系列表合并为一个 DataFrame
    df_all_entities = pd.concat(all_entities, ignore_index=True)
    df_all_relationships = pd.concat(all_relationships, ignore_index=True)
    
    return df_all_entities, df_all_relationships
