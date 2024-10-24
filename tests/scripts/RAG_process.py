import pandas as pd
import json
import sys
import os
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.embedding_utils import load_df_from_csv_with_embedding, find_k_nearest_entities
from src.utils.config import OUTPUT_PATH
from src.report.community_report import generate_and_merge_reports
from src.utils.message_templates import generate_query_template
from src.utils.llm_helpers import call_llm_api
from datasets import load_dataset

# Step 1: Load Data
logger.info("Loading data...")
entities_with_embeddings = load_df_from_csv_with_embedding("/Users/boyuren/Documents/multi_head_graph_rag/MH-GRAG-V1/export/multihop_rag/final_entities_with_embeddings_by_chatgpt4o.csv", ['entity_embedding'])
entities_with_clusters = pd.read_csv("/Users/boyuren/Documents/multi_head_graph_rag/MH-GRAG-V1/export/multihop_rag/entities_cluster_result_Oct24th.csv")
relationships = pd.read_csv("/Users/boyuren/Documents/multi_head_graph_rag/MH-GRAG-V1/export/multihop_rag/relationships_with_SAMENAME_Oct24th.csv")
logger.info("Data loaded successfully.")

# Step 2: Load Queries from Multihop-RAG
logger.info("Loading queries from Multihop-RAG...")
ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")
queries = pd.DataFrame(ds["train"])
queries = queries[queries['question_type'].isin(['inference_query', 'null_query'])].head(150)
logger.info("Queries loaded successfully.")

# Step 3: Process Each Query
results = []
results_file_path = "/Users/boyuren/Documents/multi_head_graph_rag/MH-GRAG-V1/export/rag_test/query_results.csv"

# Check if results file exists and load existing data
if os.path.exists(results_file_path):
    results_df = pd.read_csv(results_file_path)
    results = results_df.to_dict('records')
else:
    results_df = pd.DataFrame()

for idx, row in queries.iterrows():
    query_id = idx + 1
    query = row['query']
    logger.info(f"Processing query {idx + 1}/{len(queries)}: {query}")

    try:
        # Step 4: Find Nearest Entities
        logger.info("Finding the nearest entity...")
        k = 1
        nearest_entities = find_k_nearest_entities(query, entities_with_embeddings, k=k)
        entity_id = nearest_entities["entity_id"].iloc[0]
        entity = nearest_entities["entity_name"].iloc[0]
        description = nearest_entities["description"].iloc[0]
        logger.info(f"Nearest entity found: {entity_id} ({entity})")

        # Step 5: Generate and Merge Reports
        logger.info("Generating and merging community reports...")
        merge_reports = generate_and_merge_reports(entity_id, entities_with_clusters, relationships, output_file=f'/Users/boyuren/Documents/multi_head_graph_rag/MH-GRAG-V1/export/rag_test/merged_community_report_{query_id}.json')
        logger.info("Community reports generated and merged successfully.")

        # Step 6: Generate Query Template
        logger.info("Generating query template...")
        query_template = generate_query_template(query, merge_reports)
        logger.info("Query template generated.")

        # Step 7: Call LLM API
        logger.info("Calling LLM API...")
        answer = call_llm_api(query_template)
        logger.info("LLM API call completed.")

        # Step 8: Parse LLM Answer
        def parse_llm_answer(answer):
            """
            Parse the LLM returned JSON formatted answer and extract the content.
            
            Parameters:
                answer (str): LLM returned JSON string.

            Returns:
                dict: Parsed answer and used findings.
            """
            try:
                # Clean the string, remove external ```json ``` wrapping
                cleaned_answer = answer.strip("```").replace("json\n", "")
                
                # Convert the string to Python dictionary
                parsed_answer = json.loads(cleaned_answer)
                
                # Extract answer and used findings
                answer_text = parsed_answer.get("answer", "No answer found.")
                used_findings = parsed_answer.get("used_findings", [])

                return {"answer": answer_text, "used_findings": used_findings}
            
            except json.JSONDecodeError as e:
                logger.error(f"JSONDecodeError: {e}")
                return None

        logger.info("Parsing LLM answer...")
        parsed_result = parse_llm_answer(answer)

        if parsed_result:
            logger.info(f"Answer: {parsed_result['answer']}")
            logger.info(f"Used Findings: {parsed_result['used_findings']}")
            results.append({
                "query_id": query_id,
                "query": query,
                "nearest_entity_id": entity_id,
                "entity": entity,
                "description": description,
                "answer": parsed_result['answer'],
                "used_findings": parsed_result['used_findings']
            })
        else:
            logger.error("Failed to parse the LLM answer.")
            results.append({
                "query_id": query_id,
                "query": query,
                "error": "Failed to parse the LLM answer."
            })

    except Exception as e:
        logger.error(f"Error processing query {query_id}: {e}")
        results.append({
            "query_id": query_id,
            "query": query,
            "error": str(e)
        })

    # Step 9: Save Results to CSV After Each Iteration
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file_path, index=False)
    logger.info(f"Results saved successfully after processing query {idx + 1}/{len(queries)}.")
