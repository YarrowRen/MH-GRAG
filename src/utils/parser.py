# src/utils/parser.py
import re
import json
import pandas as pd
import uuid

def parse_entities(content):
    """
    解析实体信息并返回包含实体及其唯一 ID 的 DataFrame。
    """
    entity_pattern = re.compile(r'\{\s*"entity_name":\s*".*?",\s*"entity_type":\s*".*?",\s*"description":\s*".*?"\s*\}', re.DOTALL)
    entities = []
    for match in entity_pattern.finditer(content):
        entity_str = match.group(0)
        try:
            entity = json.loads(entity_str)
            # 生成唯一的 entity_id
            entity['entity_id'] = str(uuid.uuid4())
            entities.append(entity)
        except json.JSONDecodeError:
            print("Error decoding entity JSON")

    # 将实体列表转化为 DataFrame
    if entities:
        df_entities = pd.DataFrame(entities)
    else:
        df_entities = pd.DataFrame(columns=['entity_name', 'entity_type', 'description', 'entity_id'])  # 如果没有实体，返回空的 DataFrame
    
    return df_entities


def parse_relationships(content, df_entities):
    """
    解析关系信息并返回包含映射的关系 DataFrame，包括 source_entity_id 和 target_entity_id。
    """
    relationship_pattern = re.compile(r'\{\s*"source_entity":\s*".*?",\s*"target_entity":\s*".*?",\s*"relationship_type":\s*".*?",\s*"relationship_description":\s*".*?"\s*\}', re.DOTALL)
    relationships = []
    for match in relationship_pattern.finditer(content):
        relationship_str = match.group(0)
        try:
            relationship = json.loads(relationship_str)
            relationships.append(relationship)
        except json.JSONDecodeError:
            print("Error decoding relationship JSON")

    # 将关系列表转化为 DataFrame
    if relationships:
        df_relationships = pd.DataFrame(relationships)
        # 根据 entity_name 在 df_entities 中查找 entity_id，并添加到 df_relationships
        df_relationships = df_relationships.merge(df_entities[['entity_name', 'entity_id']], how='left', left_on='source_entity', right_on='entity_name').rename(columns={'entity_id': 'source_entity_id'}).drop(columns=['entity_name'])
        df_relationships = df_relationships.merge(df_entities[['entity_name', 'entity_id']], how='left', left_on='target_entity', right_on='entity_name').rename(columns={'entity_id': 'target_entity_id'}).drop(columns=['entity_name'])
    else:
        df_relationships = pd.DataFrame(columns=['source_entity', 'target_entity', 'relationship_type', 'relationship_description', 'source_entity_id', 'target_entity_id'])  # 如果没有关系，返回空的 DataFrame
    
    return df_relationships