import re
import json
import uuid
import pandas as pd

def parse_entities(content):
    """
    解析实体信息并返回包含实体及其唯一 ID 的 DataFrame。
    新的实体解析方法支持最新的 Prompt 格式。
    """
    # 匹配实体的 JSON 格式：{"name": ..., "type": ..., "description": ...}
    entity_pattern = re.compile(r'\{\s*"name":\s*".*?",\s*"type":\s*".*?",\s*"description":\s*".*?"\s*\}', re.DOTALL)
    entities = []
    
    for match in entity_pattern.finditer(content):
        entity_str = match.group(0)
        try:
            # 将每个匹配的实体解析为字典
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
        df_entities = pd.DataFrame(columns=['name', 'type', 'description', 'entity_id'])  # 如果没有实体，返回空的 DataFrame
    
    return df_entities


def parse_relationships(content, df_entities):
    """
    解析关系信息并返回包含映射的关系 DataFrame，包括 source_entity_id 和 target_entity_id。
    新的关系解析方法支持最新的 Prompt 格式。
    """
    # 匹配关系的 JSON 格式：{"source": ..., "target": ..., "relationship": ..., "relationship_strength": ...}
    relationship_pattern = re.compile(r'\{\s*"source":\s*".*?",\s*"target":\s*".*?",\s*"relationship":\s*".*?",\s*"relationship_strength":\s*\d+\s*\}', re.DOTALL)
    relationships = []
    
    for match in relationship_pattern.finditer(content):
        relationship_str = match.group(0)
        try:
            # 将每个匹配的关系解析为字典
            relationship = json.loads(relationship_str)
            relationships.append(relationship)
        except json.JSONDecodeError:
            print("Error decoding relationship JSON")
    
    # 将关系列表转化为 DataFrame
    if relationships:
        df_relationships = pd.DataFrame(relationships)
        # 根据 source 和 target 在 df_entities 中查找对应的 entity_id，并添加到 df_relationships
        df_relationships = df_relationships.merge(df_entities[['name', 'entity_id']], how='left', left_on='source', right_on='name').rename(columns={'entity_id': 'source_entity_id'}).drop(columns=['name'])
        df_relationships = df_relationships.merge(df_entities[['name', 'entity_id']], how='left', left_on='target', right_on='name').rename(columns={'entity_id': 'target_entity_id'}).drop(columns=['name'])
    else:
        df_relationships = pd.DataFrame(columns=['source', 'target', 'relationship', 'relationship_strength', 'source_entity_id', 'target_entity_id'])  # 如果没有关系，返回空的 DataFrame
    
    return df_relationships