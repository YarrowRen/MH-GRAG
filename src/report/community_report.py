
from src.utils.llm_helpers import call_llm_api
from src.utils.message_templates import get_community_report_template
from src.gnn_clustering.utils import get_all_related_relationships_within_cluster
from src.utils.message_templates import generate_community_report_template
import pandas as pd
from tqdm import tqdm
import json
from collections import deque

# 用于生成社区报告
def generate_community_report(community_group):
    # 获取该社区的所有实体名称和描述
    entity_names = community_group['entity_name'].tolist()
    descriptions = community_group['description'].tolist()
    
    messages = get_community_report_template(entity_names, descriptions)
    
    # 调用 LLM API 并获取内容
    content = call_llm_api(messages, max_tokens=500, temperature=0.7)
    
    # 返回生成的报告
    return content



def generate_reports_for_all_communities(entities_with_embeddings):
    """
    接收包含实体和 node2vec 嵌入的 DataFrame，将其按社区分组，并为每个社区生成报告。
    最终返回一个包含每个社区的 community_id 和生成报告的 DataFrame。
    
    参数:
    entities_with_embeddings (pd.DataFrame): 包含实体、描述、embedding、node2vec 嵌入和社区 ID 的 DataFrame。
    
    返回:
    pd.DataFrame: 包含每个社区的 community_id 和生成报告的 DataFrame。
    """
    community_reports = []

    # 按 community_id 分组
    for community_id, group in tqdm(entities_with_embeddings.groupby('community_id'), desc="Generating Community Reports"):
        # 获取每个社区的实体报告
        report = generate_community_report(group)

        # 将每个社区的报告以字典形式保存
        community_reports.append({
            'community_id': community_id,
            'community_report': report
        })

    # 将社区报告转换为 DataFrame
    community_reports_df = pd.DataFrame(community_reports)
    
    return community_reports_df

def generate_reports_for_all_sub_communities(entities_with_embeddings):
    """
    接收包含实体和 node2vec 嵌入的 DataFrame，将其按社区与子社区分组，并为每个子社区生成报告。
    最终返回一个包含每个子社区的 community_id 和生成报告的 DataFrame。
    
    参数:
    entities_with_embeddings (pd.DataFrame): 包含实体、描述、embedding、node2vec 嵌入和社区 ID以及子社区 ID 的 DataFrame。
    
    返回:
    pd.DataFrame: 包含每个子社区的 community_id , sub_community_id 和生成报告的 DataFrame。
    """
    sub_community_reports = []

    # 按 community_id 和 sub_community_id 分组，显示进度条
    for (community_id, sub_community_id), group in tqdm(entities_with_embeddings.groupby(['community_id', 'sub_community_id']), desc="Generating Sub-Community Reports"):
        # 获取每个子社区的实体报告
        report = generate_community_report(group)

        # 将每个子社区的报告以字典形式保存
        sub_community_reports.append({
            'community_id': community_id,
            'sub_community_id': sub_community_id, 
            'community_report': report
        })

    # 将子社区报告转换为 DataFrame
    sub_community_reports_df = pd.DataFrame(sub_community_reports)
    
    return sub_community_reports_df


def generate_and_merge_reports(entity_id, entities_with_clusters, relationships, output_file='export/rag_test/merged_community_report.json'):
    """
    根据指定的 entity_id 生成社区报告，并将多个聚类结果合并为一个 JSON 文件。

    参数：
        entity_id (int): 要查询的实体 ID。
        entities_with_clusters (DataFrame): 包含实体及其聚类信息的数据框。
        relationships (DataFrame): 关系数据框。
        output_file (str): 合并后的 JSON 文件路径（默认路径为 'export/rag_test/merged_community_report.json'）。
    """
    cluster_columns = [
        "initial_leiden_cluster", "merge_leiden_cluster", 
        "multihead_leiden_cluster_head_1", "multihead_leiden_cluster_head_2", 
        "multihead_leiden_cluster_head_3"
    ]

    all_responses = []  # 存储所有 response
    longest_response = None  # 记录 summary 最长的 response

    # 遍历每个 cluster_column，生成消息并获取响应
    for cluster_column in cluster_columns:
        related_relationships = get_all_related_relationships_within_cluster(
            entities_with_clusters, relationships, entity_id, cluster_column
        )

        # 如果 related_relationships 为空，则跳过此循环
        if related_relationships.empty:
            print(f"No related relationships found for entity_id {entity_id} in cluster '{cluster_column}'. Skipping...")
            continue

        message = generate_community_report_template(
            entity_id, 
            entities_with_clusters[['entity_name', 'entity_type', 'description', 'entity_id', 'corpus_id']],
            related_relationships
        )

        response = call_llm_api(message, max_tokens=2000)  # 调用 LLM API 获取响应

        # 将字符串类型的 response 转换为字典
        response_dict = json.loads(response)
        all_responses.append(response_dict)

        # 检查当前 response 是否是 summary 最长的 response
        if not longest_response or len(response_dict["summary"]) > len(longest_response["summary"]):
            longest_response = response_dict

    # 初始化合并后的结果
    merged_response = {
        "title": longest_response["title"],
        "summary": longest_response["summary"],
        "findings": []
    }

    # 合并所有 findings，并为每个 finding 添加索引
    finding_index = 1  # 初始化索引
    for response_dict in all_responses:
        for finding in response_dict["findings"]:
            # 为每个 finding 添加索引
            finding["id"] = finding_index
            merged_response["findings"].append(finding)
            finding_index += 1

    # 将合并后的结果保存为 JSON 文件
    with open(output_file, 'w') as file:
        json.dump(merged_response, file, ensure_ascii=False, indent=2)

    print(f"Merged report saved to {output_file}")
    return merged_response