
from src.utils.llm_helpers import call_llm_api
from src.utils.message_templates import get_community_report_template
import pandas as pd
from tqdm import tqdm

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
