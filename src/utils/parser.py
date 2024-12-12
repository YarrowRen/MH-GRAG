import re
import json
import uuid
import pandas as pd
import os

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



def parse_gen_qa_llm_answer(entity_id, response, export_path=None):
    """
    Parse the LLM returned JSON formatted answer and extract the content,
    and optionally append the questions, answers, and scores to a CSV file.

    Parameters:
        entity_id (str): The ID of the entity associated with the response.
        response (str): LLM returned JSON string.
        export_path (str, optional): Path to save or append the extracted QA pairs to a CSV file.

    Returns:
        dict: Parsed answer and used_documents.
    """
    # Clean the string, remove external ```json ``` wrapping
    cleaned_answer = response.strip("```").replace("json\n", "")
    
    # Convert the string to Python dictionary
    parsed_answer = json.loads(cleaned_answer)
    
    # Extract the questions and answers
    print("\nQuestions, Answers, and Scores:")
    questions_and_answers = parsed_answer.get("questions_and_answers", [])
    qa_list = []

    for i, qa in enumerate(questions_and_answers, start=1):
        question = qa.get("question", "No question provided")
        score = qa.get("score", "No score provided")
        answers = qa.get("answers", [])
        
        # Consolidate answers and views into a list of dictionaries
        answers_list = [{answer_obj.get("view", "No view provided"): answer_obj.get("answer", "No answer provided")} for answer_obj in answers]
        
        # Print question, score, and answers
        print(f"Q{i}: {question} (Score: {score})")
        for j, answer_dict in enumerate(answers_list, start=1):
            for view, answer in answer_dict.items():
                print(f"A{i}.{j}: {answer} (View: {view})")
        print()
        
        qa_list.append({
            "entity_id": entity_id,
            "question": question,
            "score": score,
            "answers_list": answers_list
        })

    # If export_path is provided, append QA pairs to the CSV file
    if export_path and qa_list:
        df = pd.DataFrame(qa_list)
        
        # Check if the file exists, and append if it does, otherwise create it
        if os.path.exists(export_path):
            df.to_csv(export_path, mode='a', header=False, index=False, encoding="utf-8")
        else:
            df.to_csv(export_path, mode='w', index=False, encoding="utf-8")
        
        print(f"Questions and answers saved to {export_path}")

    return parsed_answer