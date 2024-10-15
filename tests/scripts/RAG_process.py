import pandas as pd
import json
import sys
import os
import logging

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

# Step 1: Load Data
logger.info("Loading data...")
entities_with_embeddings = load_df_from_csv_with_embedding("/Users/boyuren/Documents/multi_head_graph_rag/MH-GRAG-V1/export/multihop_rag/final_entities_with_embeddings_by_chatgpt4o.csv", ['entity_embedding'])
entities_with_clusters = pd.read_csv("/Users/boyuren/Documents/multi_head_graph_rag/MH-GRAG-V1/export/multihop_rag/filtered_entities_cluster_reults.csv")
relationships = pd.read_csv("/Users/boyuren/Documents/multi_head_graph_rag/MH-GRAG-V1/export/multihop_rag/filtered_relationships_with_duplicates.csv")
logger.info("Data loaded successfully.")

# Step 2: Get Query Embedding
query = "Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch, and is accused by prosecutors of committing fraud for personal gain?"
logger.info(f"Query: {query}")

# Step 3: Find Nearest Entities
logger.info("Finding the nearest entity...")
k = 1
nearest_entities = find_k_nearest_entities(query, entities_with_embeddings, k=k)
entity_id = nearest_entities["entity_id"].iloc[0]
logger.info(f"Nearest entity found: {entity_id}")

# Step 4: Generate and Merge Reports
logger.info("Generating and merging community reports...")
merge_reports = generate_and_merge_reports(entity_id, entities_with_clusters, relationships, output_file=f'/Users/boyuren/Documents/multi_head_graph_rag/MH-GRAG-V1/export/rag_test/merged_community_report_{entity_id}.json')
logger.info("Community reports generated and merged successfully.")

# Step 5: Generate Query Template
logger.info("Generating query template...")
query_template = generate_query_template(query, merge_reports)
logger.info("Query template generated.")

# Step 6: Call LLM API
logger.info("Calling LLM API...")
answer = call_llm_api(query_template)
logger.info("LLM API call completed.")

# Step 7: Parse LLM Answer
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
else:
    logger.error("Failed to parse the LLM answer.")