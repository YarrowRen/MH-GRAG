# src/extractors/summarize_element_instances.py
import pandas as pd
from tqdm import tqdm
from src.utils.llm_helpers import call_llm_api
from src.utils.message_templates import get_entity_summary_template, get_relationship_summary_template

def summarize_element_instances(entities_df, relationships_df):
    """
    使用 LLM 为每个实体和关系生成简短的扩写，并存储在新的列中。
    
    参数:
    entities_df (DataFrame): 包含实体的 DataFrame。
    relationships_df (DataFrame): 包含关系的 DataFrame。
    
    返回:
    tuple: 包含更新后的实体和关系 DataFrame 的元组。
    """
    # 为每个实体生成扩写，并显示进度条
    entity_summaries = []
    for _, entity in tqdm(entities_df.iterrows(), total=entities_df.shape[0], desc="Processing Entities"):
        # 调用 message_templates 中的方法来生成实体消息模板
        messages = get_entity_summary_template(entity['entity_name'], entity['entity_type'], entity['description'])
        summary = call_llm_api(messages)
        entity_summaries.append(summary)
    
    entities_df['summary'] = entity_summaries

    # 为每个关系生成扩写，并显示进度条
    relationship_summaries = []
    for _, rel in tqdm(relationships_df.iterrows(), total=relationships_df.shape[0], desc="Processing Relationships"):
        # 调用 message_templates 中的方法来生成关系消息模板
        messages = get_relationship_summary_template(rel['source_entity'], rel['target_entity'], rel['relationship_description'])
        summary = call_llm_api(messages)
        relationship_summaries.append(summary)
    
    relationships_df['summary'] = relationship_summaries

    return entities_df, relationships_df