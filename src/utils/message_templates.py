# src/utils/message_templates.py

def get_entities_and_relationships_template(text_chunk):
    """
    返回用于 LLM 调用的消息模板
    """
    ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT = f"""
    -Goal-
    Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

    -Steps-
    1. Identify all entities. For each identified entity, extract the following information:
    - entity_name: Name of the entity, capitalized
    - entity_type: One of the following types: [Person, Organization, Location, Event, Product, Concept, Time]
    - entity_description: Comprehensive description of the entity's attributes and activities

    Format each entity output as a JSON entry with the following format:

    {{"name": <entity name>, "type": <type>, "description": <entity description>}}

    2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
    For each pair of related entities, extract the following information:
    - source_entity: name of the source entity, as identified in step 1
    - target_entity: name of the target entity, as identified in step 1
    - relationship_description: explanation as to why you think the source entity and the target entity are related to each other
    - relationship_strength: an integer score between 1 to 10, indicating strength of the relationship between the source entity and target entity

    Format each relationship as a JSON entry with the following format:

    {{"source": <source_entity>, "target": <target_entity>, "relationship": <relationship_description>, "relationship_strength": <relationship_strength>}}

    3. Return output in English as a single list of all JSON entities and relationships identified in steps 1 and 2.

    ######################
    -Examples-
    ######################
    Example 1:
    Text:
    The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.

    ######################
    Example 2:
    Text:
    TechGlobal's (TG) stock skyrocketed in its opening day on the Global Exchange Thursday. But IPO experts warn that the semiconductor corporation's debut on the public markets isn't indicative of how other newly listed companies may perform.

    TechGlobal, a formerly public company, was taken private by Vision Holdings in 2014. The well-established chip designer says it powers 85% of premium smartphones.
    ######################
    Output:
    [
    {{"name": "TECHGLOBAL", "type": "ORGANIZATION", "description": "TechGlobal is a stock now listed on the Global Exchange which powers 85% of premium smartphones"}},
    {{"name": "VISION HOLDINGS", "type": "ORGANIZATION", "description": "Vision Holdings is a firm that previously owned TechGlobal"}},
    {{"source": "TECHGLOBAL", "target": "VISION HOLDINGS", "relationship": "Vision Holdings formerly owned TechGlobal from 2014 until present", "relationship_strength": 5}}
    ]

    ######################
    Example 3:
    Text:
    Five Aurelians jailed for 8 years in Firuzabad and widely regarded as hostages are on their way home to Aurelia.

    The swap orchestrated by Quintara was finalized when $8bn of Firuzi funds were transferred to financial institutions in Krohaara, the capital of Quintara.

    The exchange initiated in Firuzabad's capital, Tiruzia, led to the four men and one woman, who are also Firuzi nationals, boarding a chartered flight to Krohaara.

    They were welcomed by senior Aurelian officials and are now on their way to Aurelia's capital, Cashion.

    The Aurelians include 39-year-old businessman Samuel Namara, who has been held in Tiruzia's Alhamia Prison, as well as journalist Durke Bataglani, 59, and environmentalist Meggie Tazbah, 53, who also holds Bratinas nationality.
    ######################
    Output:
    [
    {{"name": "FIRUZABAD", "type": "GEO", "description": "Firuzabad held Aurelians as hostages"}},
    {{"name": "AURELIA", "type": "GEO", "description": "Country seeking to release hostages"}},
    {{"name": "QUINTARA", "type": "GEO", "description": "Country that negotiated a swap of money in exchange for hostages"}},
    {{"name": "TIRUZIA", "type": "GEO", "description": "Capital of Firuzabad where the Aurelians were being held"}},
    {{"name": "KROHAARA", "type": "GEO", "description": "Capital city in Quintara"}},
    {{"name": "CASHION", "type": "GEO", "description": "Capital city in Aurelia"}},
    {{"name": "SAMUEL NAMARA", "type": "PERSON", "description": "Aurelian who spent time in Tiruzia's Alhamia Prison"}},
    {{"name": "ALHAMIA PRISON", "type": "GEO", "description": "Prison in Tiruzia"}},
    {{"name": "DURKE BATAGLANI", "type": "PERSON", "description": "Aurelian journalist who was held hostage"}},
    {{"name": "MEGGIE TAZBAH", "type": "PERSON", "description": "Bratinas national and environmentalist who was held hostage"}},
    {{"source": "FIRUZABAD", "target": "AURELIA", "relationship": "Firuzabad negotiated a hostage exchange with Aurelia", "relationship_strength": 2}},
    {{"source": "QUINTARA", "target": "AURELIA", "relationship": "Quintara brokered the hostage exchange between Firuzabad and Aurelia", "relationship_strength": 2}},
    {{"source": "QUINTARA", "target": "FIRUZABAD", "relationship": "Quintara brokered the hostage exchange between Firuzabad and Aurelia", "relationship_strength": 2}},
    {{"source": "SAMUEL NAMARA", "target": "ALHAMIA PRISON", "relationship": "Samuel Namara was a prisoner at Alhamia prison", "relationship_strength": 8}},
    {{"source": "SAMUEL NAMARA", "target": "MEGGIE TAZBAH", "relationship": "Samuel Namara and Meggie Tazbah were exchanged in the same hostage release", "relationship_strength": 2}},
    {{"source": "SAMUEL NAMARA", "target": "DURKE BATAGLANI", "relationship": "Samuel Namara and Durke Bataglani were exchanged in the same hostage release", "relationship_strength": 2}},
    {{"source": "MEGGIE TAZBAH", "target": "DURKE BATAGLANI", "relationship": "Meggie Tazbah and Durke Bataglani were exchanged in the same hostage release", "relationship_strength": 2}},
    {{"source": "SAMUEL NAMARA", "target": "FIRUZABAD", "relationship": "Samuel Namara was a hostage in Firuzabad", "relationship_strength": 2}},
    {{"source": "MEGGIE TAZBAH", "target": "FIRUZABAD", "relationship": "Meggie Tazbah was a hostage in Firuzabad", "relationship_strength": 2}},
    {{"source": "DURKE BATAGLANI", "target": "FIRUZABAD", "relationship": "Durke Bataglani was a hostage in Firuzabad", "relationship_strength": 2}}
    ]
    Output:
    [
      {{"name": "CENTRAL INSTITUTION", "type": "ORGANIZATION", "description": "The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday"}},
      {{"name": "MARTIN SMITH", "type": "PERSON", "description": "Martin Smith is the chair of the Central Institution"}},
      {{"name": "MARKET STRATEGY COMMITTEE", "type": "ORGANIZATION", "description": "The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply"}},
      {{"source": "MARTIN SMITH", "target": "CENTRAL INSTITUTION", "relationship": "Martin Smith is the Chair of the Central Institution and will answer questions at a press conference", "relationship_strength": 9}}
    ]

    -Real Data-
    ######################
    entity_types: [Person, Organization, Location, Event, Product, Concept, Time]
    text: "{text_chunk}"
    ######################
    output:
    """

    return [
        {"role": "system", "content": "You are an expert in extracting entities and relationships from text. Always respond in English."},
        {"role": "user", "content": ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT}
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
