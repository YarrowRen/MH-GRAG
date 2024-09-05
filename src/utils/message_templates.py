# src/utils/message_templates.py

def get_entities_and_relationships_template(text_chunk):
    """
    返回用于 LLM 调用的消息模板。
    """
    return [
        {"role": "system", "content": "You are an expert in extracting entities and relationships from text. Always respond in English."},
        {"role": "user", "content": f"""
        Extract the named entities from the following text, only using the specified entity types: 
        Person, Organization, Location, Event, Product, Concept, Work of Art, Miscellaneous. 

        Provide the results in a list of tuples where each tuple contains (entity_name, entity_type, description). 
        Make sure that entity_type and description are in English.

        Text: "{text_chunk}"
        """},
        {"role": "user", "content": f"""
        After extracting the entities, extract the relationships between these entities, only use the entities' entity_name which you have design before, 
              only using the specified relationship types: 
        Part-of, Located-in, Affiliated-with, Related-to, Works-for, Produced-by, Happened-at, Uses, Ownership, Family-Relation, Colleague-Relation, Friendship, Mentor-Protege, Romantic-Relation, Professional-Relation, Historical-Relation.

        For each relationship, provide the result as a tuple with the following structure:
        (source_entity, target_entity, relationship_type, relationship_description).

        - `relationship_type` should be one of the specified relationship types above.
        - `relationship_description` should be a concise description of the relationship.

        Examples:
        - (John Doe, Microsoft, Works-for, "John Doe works for Microsoft.")
        - (Paris, France, Located-in, "Paris is located in France.")
        - (Steve Jobs, Apple, Produced-by, "Steve Jobs co-founded Apple.")
        """},
        {"role": "user", "content": """
        Please provide the results in a well-formatted, complete JSON structure with two keys: 'entities' and 'relationships'.
        Ensure the JSON output is valid, and do not truncate the JSON. 
        """}
    ]


def get_entity_summary_template(entity_name, entity_type, description):
    """
    返回生成实体摘要的 LLM 消息模板。
    
    参数:
    entity_name (str): 实体名称。
    entity_type (str): 实体类型。
    description (str): 实体的描述。
    
    返回:
    list: 生成摘要的消息列表。
    """
    return [
        {"role": "system", "content": "You are an expert in expanding brief information into a single concise sentence without adding any assumptions."},
        {"role": "user", "content": f"Expand the following entity information into a single concise sentence without adding any assumptions:\nEntity: {entity_name} ({entity_type})\nDescription: {description}"}
    ]

def get_relationship_summary_template(source_entity, target_entity, relationship_description):
    """
    返回生成关系摘要的 LLM 消息模板。
    
    参数:
    source_entity (str): 源实体名称。
    target_entity (str): 目标实体名称。
    relationship_description (str): 关系的描述。
    
    返回:
    list: 生成摘要的消息列表。
    """
    return [
        {"role": "system", "content": "You are an expert in expanding brief information into a single concise sentence without adding any assumptions."},
        {"role": "user", "content": f"Expand the following relationship into a single concise sentence without adding any assumptions:\nRelationship: {source_entity} -> {target_entity}\nDescription: {relationship_description}"}
    ]


def get_community_report_template(entity_names, descriptions):
    """
    返回生成社区报告的 LLM 消息模板。
    
    参数:
    entity_names (list): 实体名称的列表。
    descriptions (list): 实体描述的列表。
    
    返回:
    list: 用于生成社区报告的消息列表。
    """
    # 构造模型输入的初始内容
    messages = [
        {"role": "system", "content": "你是一位报告总结专家，擅长分析和总结不同实体的信息。你的任务是根据给定的实体名称和描述，生成一个详细的社区报告，突出每个实体的关键特征并解释它们之间可能的关联。"},
        {"role": "user", "content": "以下是社区中的实体及其描述，请生成一个全面的报告。"}
    ]
    
    # 将实体名称和描述添加到用户消息中
    for name, desc in zip(entity_names, descriptions):
        messages.append({"role": "user", "content": f"实体: {name}\n描述: {desc}\n"})
    
    return messages
