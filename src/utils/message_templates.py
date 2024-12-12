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



def generate_community_report_template(core_entity_id, df_entities, related_relationships):
    """
    Generates a community report message based on the given parameters.
    
    Parameters:
    core_entity_id (int): The ID of the core entity.
    df_entities (DataFrame): DataFrame containing all entities.
    related_relationships (DataFrame): DataFrame containing relationships related to the core entity.
    
    Returns:
    str: A message formatted for generating a community report.
    """
    # Extract core entity information
    core_entity = df_entities[df_entities['entity_id'] == core_entity_id].iloc[0]
    core_entity_text = f"Core_Entity\n\nid,entity,description\n{core_entity['entity_id']},{core_entity['entity_name']},{core_entity['description']}\n"
    
    # Extract related entities information
    related_entity_ids = set(related_relationships['source_entity_id']).union(set(related_relationships['target_entity_id']))
    related_entity_ids.discard(core_entity_id)
    related_entities = df_entities[df_entities['entity_id'].isin(related_entity_ids)]
    related_entities_text = "Related_Entities\n\nid,entity,description\n" + "\n".join(
        f"{row['entity_id']},{row['entity_name']},{row['description']}" for _, row in related_entities.iterrows()
    )
    
    # Extract relationships information
    relationships_text = "Relationships\n\nid,source,target,description\n" + "\n".join(
        f"{idx},{row['source_entity']},{row['target_entity']},{row['relationship_description']}"
        for idx, row in related_relationships.iterrows()
    )
    
    # Combine all parts to form input_text
    input_text = f"{core_entity_text}\n{related_entities_text}\n\n{relationships_text}"
    
    message_content = f"""
    You are a community analyst.

    # Goal
    Write a comprehensive assessment report of a community as a community analyst. The content of this report includes an overview of the community's key entities and relationships.

    # Report Structure
    The report should include the following sections:
    - TITLE: community's name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.
    - SUMMARY: An executive summary of the community's overall structure, how its entities are related to each other, and significant points associated with its entities.
    - DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.

    Return output as a well-formed JSON-formatted string with the following format. Don't use any unnecessary escape sequences. The output should be a single JSON object that can be parsed by json.loads.
        {{
            "title": "<report_title>",
            "summary": "<executive_summary>",
            "findings": [{{"summary":"<insight_1_summary>", "explanation": "<insight_1_explanation>"}}, {{"summary":"<insight_2_summary>", "explanation": "<insight_2_explanation>"}}]
        }}

    # Grounding Rules
    After each paragraph, add data record reference if the content of the paragraph was derived from one or more data records. Reference is in the format of [records: <record_source> (<record_id_list>, ...<record_source> (<record_id_list>)]. If there are more than 10 data records, show the top 10 most relevant records.
    Each paragraph should contain multiple sentences of explanation and concrete examples with specific named entities. All paragraphs must have these references at the start and end. Use "NONE" if there are no related roles or records. Everything should be in English.

    Example paragraph with references added:
    This is a paragraph of the output text [records: Entities (1, 2, 3), Claims (2, 5), Relationships (10, 12)]

    # Example Input
    -----------
    Text:

    Entities

    id,entity,description
    5,ABILA CITY PARK,Abila City Park is the location of the POK rally

    Related_Entities

    id,entity,description
    6,POK,POK is an organization holding a rally in Abila City Park
    7,CENTRAL BULLETIN,Central Bulletin is a media outlet covering the POK rally

    Relationships

    id,source,target,description
    37,ABILA CITY PARK,POK RALLY,Abila City Park is the location of the POK rally
    38,ABILA CITY PARK,POK,POK is holding a rally in Abila City Park
    39,ABILA CITY PARK,POKRALLY,The POKRally is taking place at Abila City Park
    40,ABILA CITY PARK,CENTRAL BULLETIN,Central Bulletin is reporting on the POK rally taking place in Abila City Park

    Output:
    {{
        "title": "Abila City Park and POK Rally",
        "summary": "The community revolves around the Abila City Park, which is the location of the POK rally. The park has relationships with POK, POKRALLY, and Central Bulletin, all of which are associated with the rally event.",
        "findings": [
            {{
                "summary": "Abila City Park as the central location",
                "explanation": "Abila City Park is the central entity in this community, serving as the location for the POK rally. This park is the common link between all other entities, suggesting its significance in the community. The park's association with the rally could potentially lead to issues such as public disorder or conflict, depending on the nature of the rally and the reactions it provokes. [records: Entities (id: 5), Relationships (id: 37, 38, 39, 40)]"
            }},
            {{
                "summary": "POK's role in the community",
                "explanation": "POK is another key entity in this community, being the organizer of the rally at Abila City Park. The nature of POK and its rally could be a potential source of threat, depending on their objectives and the reactions they provoke. The relationship between POK and the park is crucial in understanding the dynamics of this community. [records: Relationships (id: 38)]"
            }},
            {{
                "summary": "POKRALLY as a significant event",
                "explanation": "The POKRALLY is a significant event taking place at Abila City Park. This event is a key factor in the community's dynamics and could be a potential source of threat, depending on the nature of the rally and the reactions it provokes. The relationship between the rally and the park is crucial in understanding the dynamics of this community. [records: Relationships (id: 39)]"
            }},
            {{
                "summary": "Role of Central Bulletin",
                "explanation": "Central Bulletin is reporting on the POK rally taking place in Abila City Park. This suggests that the event has attracted media attention, which could amplify its impact on the community. The role of Central Bulletin could be significant in shaping public perception of the event and the entities involved. [records: Relationships (id: 40)]"
            }}
        ]
    }}

    # Real Data

    Use the following text for your answer. Do not make anything up in your answer.

    {input_text}

    Output:"""

    message = [
        {"role": "user", "content": message_content}
    ]  
    return message

def generate_query_template(question, report):
    """
    根据问题和社区报告生成 Prompt，包含每个 finding 的 summary 和 explanation，
    并按照优化的结构进行组织。
    """
    findings = "\n".join([
        f"- {finding['id']}: {finding['summary']}\n  Explanation: {finding.get('explanation', 'No explanation provided.')}"
        for finding in report.get("findings", [])
    ])

    message_content = f"""
    ---Role---
    You are a helpful assistant responding to questions about data in the report provided.

    ---Goal---
    Generate a response to the following question based on the provided report:

    Question: {question}

    ---Instructions---
    - Use the data provided in the report below as the primary context for generating the response.
    - If you don't know the answer or if the input report does not contain sufficient information, respond with: "Information not found in the report."
    - Provide the `id` of the findings used to generate your response.
    - The response should be JSON formatted as follows:
      {{
          "answer": <string>,
          "used_findings": [<list of finding ids>]
      }}

    ---Context---
    Below is the community report data you have access to:

    Title: {report['title']}
    Summary: {report['summary']}

    Findings:
    {findings}

    output: 
    """

    message = [
        {"role": "user", "content": message_content}
    ]  
    return message

def generate_baseline_query_template(question, corpus):
    """
    根据问题和相关语料 (corpus) 生成 Prompt，适用于普通的 RAG 模型。
    """
    # 将 DataFrame 的 'id' 和 'chunk' 列转换为格式化的上下文字符串
    context = "\n".join(
        corpus.apply(lambda row: f"- {row['id']}: {row['chunk']}", axis=1)
    )
    
    message_content = f"""
    ---Role---
    You are a helpful assistant that answers user questions based on the provided corpus.

    ---Goal---
    Generate a response to the following question using the information in the provided corpus:

    Question: {question}

    ---Instructions---
    - Use only the information from the provided corpus to generate your response.
    - If you don't know the answer or if the corpus does not contain sufficient information, respond with: "Information not found in the report."
    - Provide the `id` of the documents (chunks) used to generate your response.
    - The response should be JSON formatted as follows:
      {{
          "answer": <string>,
          "used_documents": [<list of document ids>]
      }}

    ---Context---
    Below is the corpus you have access to:

    {context}

    output:
    """

    message = [
        {"role": "user", "content": message_content}
    ]
    return message


def generate_multiple_choice_query_template(question, multiple_choice, report):
    """
    根据问题和社区报告生成 Prompt，包含每个 finding 的 summary 和 explanation，
    并按照优化的结构进行组织。
    """
    findings = "\n".join([
        f"- {finding['id']}: {finding['summary']}\n  Explanation: {finding.get('explanation', 'No explanation provided.')}"
        for finding in report.get("findings", [])
    ])
    options_string = ", ".join([f"{key}: {value}" for key, value in multiple_choice.items()])

    message_content = f"""
    ---Role---
    You are a helpful assistant responding to questions about data in the report provided.

    ---Goal---
    Generate a response to the following multiple-choice question based on the provided report:

    Question: {question}
    Output your answer by only choosing one from the following choices: {options_string}

    ---Instructions---
    - Use the data provided in the report below as the primary context for generating the response.
    - If you don't know the answer or if the input report does not contain sufficient information, respond with: "Information not found in the report."
    - Provide the `id` of the findings used to generate your response.
    - Do not output any explanations.
    - Output your "answer" by only choosing one from the choices
    - The response should be JSON formatted as follows:
      {{
          "answer": <string>,
          "used_findings": [<list of finding ids>]
      }}

    ---Context---
    Below is the community report data you have access to:

    Title: {report['title']}
    Summary: {report['summary']}

    Findings:
    {findings}

    output: 
    """

    message = [
        {"role": "user", "content": message_content}
    ]  
    return message


def generate_multi_view_question_template(entity_id, entity_name, entity_data, multi_head_report, question_count=5):
    """
    Generate a prompt for summarizing multi-head clustering reports
    and creating high-quality questions and answers with multiple perspectives.

    Parameters:
    - entity_id (str): The ID of the core entity.
    - entity_name (str): The name of the core entity.
    - entity_data (dict): The detailed information about the entity.
    - multi_head_report (str): The multi-head clustering report.
    - question_count (int): Number of questions to generate (default: 5).

    Returns:
    - str: The generated prompt.
    """
    message_content = f"""
    ---Role---
    You are an expert data analyst capable of extracting insights from clustering reports.

    Your task is to:
    1. Analyze the report to identify unique insights or information that can be derived about the core entity (ID: {entity_id}).
    2. Based on these unique insights, generate {question_count} specific questions related to the core entity and provide their corresponding answers using the provided report.
    3. Ensure each question is generated with multiple perspectives:
        - If the report provides sufficient perspectives, generate 4 distinct perspectives for each question.
        - If 4 perspectives are not feasible, generate 3 high-quality perspectives.
        - If only 2 perspectives are possible, ensure both are distinct and insightful.
        - If no multi-perspective view is possible, generate a single high-quality perspective.
    4. Rate each question on a scale of 0-10 based on its quality and the diversity of its perspectives. Focus on clarity, relevance, and originality.
    5. Format the output as JSON.

    ---Input Details---

    1. **Core Entity Information**:
    - Entity ID: {entity_id}
    - Entity Name: {entity_name}
    - Entity Data: {entity_data}

    2. **Report**:
    - Report
        {multi_head_report}

    ---Goal---

    Step 1: Analyze and summarize the report, specifically focusing on the core entity's roles, relationships, patterns, and unique insights.

    Step 2: Generate {question_count} questions about the core entity based on the unique insights from the report. Ensure the questions:
    - Focus on meaningful patterns, relationships, or attributes revealed in the report.
    - Are highly relevant to the core entity's attributes or context.

    Step 3: Provide concise answers to each generated question using the provided clustering report. For each question, summarize the answer from multiple perspectives (up to 4) based on available information.

    Step 4: Rate each question's quality and perspective diversity on a scale of 0-10. Include the score in the output.

    Step 5: Format the output as a JSON object with the following structure:

    ---Output Format---
    Provide your response in the following JSON format:
    ```json
    {{
    "questions_and_answers": [
        {{
        "question": "<Question 1>",
        "score": <Quality Score (0-10)>,
        "answers": [
            {{
            "answer": "<Answer 1>",
            "view": "<View 1>"
            }},
            {{
            "answer": "<Answer 2>",
            "view": "<View 2>"
            }},
            ...
        ]
        }},
        {{
        "question": "<Question 2>",
        "score": <Quality Score (0-10)>,
        "answers": [
            {{
            "answer": "<Answer 1>",
            "view": "<View 1>"
            }},
            {{
            "answer": "<Answer 2>",
            "view": "<View 2>"
            }},
            ...
        ]
        }},
        ...
    ]
    }}
    '''

    ---Example Output---
    ```json
    {{
    "questions_and_answers": [
        {{
        "question": "How does the distribution of charging stations affect EV adoption rates in urban and rural areas?",
        "score": 9,
        "answers": [
            {{
            "answer": "Urban areas have a higher density of EV charging stations, which directly correlates with greater EV adoption rates.",
            "view": "Urban infrastructure advantage"
            }},
            {{
            "answer": "Rural areas face challenges due to sparse charging infrastructure, limiting EV adoption despite government incentives.",
            "view": "Rural infrastructure disparity"
            }},
            {{
            "answer": "The uneven distribution creates a geographical gap, highlighting the need for policy-driven equal access to EV charging facilities.",
            "view": "Policy and access gap"
            }}
        ]
        }},
        {{
        "question": "What role do government incentives play in the adoption of EVs?",
        "score": 8,
        "answers": [
            {{
            "answer": "Tax subsidies reduce the initial cost barrier for consumers, driving higher EV purchases in incentivized regions.",
            "view": "Economic incentives"
            }},
            {{
            "answer": "Policy measures indirectly promote EV-related infrastructure, creating a supportive environment for EV growth.",
            "view": "Infrastructure development"
            }},
            {{
            "answer": "Comparative data show a stark contrast in adoption rates between incentivized and non-incentivized regions.",
            "view": "Global adoption trends"
            }},
            {{
            "answer": "Sustainability goals drive governments to introduce aggressive policies to reduce carbon emissions.",
            "view": "Environmental perspective"
            }}
        ]
        }}
    ]
    }}
    ```

    Please proceed to generate the response in the specified format.

    Output: 
    """
    message = [
        {"role": "user", "content": message_content}
    ]  
    return message
