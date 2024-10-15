import torch
import pandas as pd
from collections import deque
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import adjusted_mutual_info_score

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dense_adj(edge_index, device='cpu'):
    adj = to_dense_adj(edge_index)[0].to(device)
    return adj

# 计算多个头之间的互信息
def calculate_ami_between_heads(cluster_labels):
    num_heads = len(cluster_labels)
    ami_matrix = [[0] * num_heads for _ in range(num_heads)]
    
    # 计算每两个头之间的 AMI
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            ami = adjusted_mutual_info_score(cluster_labels[i], cluster_labels[j])
            ami_matrix[i][j] = ami
            ami_matrix[j][i] = ami

    return ami_matrix

# 将社区划分转换为节点标签
def communities_to_labels(communities, num_nodes):
    labels = [-1] * num_nodes
    for community_id, community in enumerate(communities):
        for node in community:
            labels[node] = community_id
    return labels


def get_all_related_relationships_within_cluster(df_entities, df_relationships, entity_id, cluster_column):
    """
    获取给定 entity_id 所在簇内的所有相关关系。
    """
    # 检查给定的 entity_id 是否存在于 df_entities 中
    entity_row = df_entities.loc[df_entities['entity_id'] == entity_id]

    if entity_row.empty:
        return pd.DataFrame()  # 返回空的 DataFrame 以避免后续代码崩溃

    # 尝试获取 cluster_label，若不存在则返回空 DataFrame
    cluster_label = entity_row[cluster_column].values[0]

    # 获取属于同一个簇的所有实体 ID
    entities_in_cluster = df_entities[df_entities[cluster_column] == cluster_label]['entity_id'].tolist()

    # 用于存储访问过的节点
    visited = set()
    # 使用队列进行广度优先搜索
    queue = deque([entity_id])

    # 进行广度优先搜索以找到所有与 entity_id 相关联的节点，限定在同一个簇内
    while queue:
        current_entity = queue.popleft()
        if current_entity not in visited:
            visited.add(current_entity)
            # 找到所有与当前节点直接相连的节点，且目标节点在相同簇内
            neighbors = df_relationships[
                ((df_relationships['source_entity_id'] == current_entity) & (df_relationships['target_entity_id'].isin(entities_in_cluster))) |
                ((df_relationships['target_entity_id'] == current_entity) & (df_relationships['source_entity_id'].isin(entities_in_cluster)))
            ]
            # 添加未访问的相邻节点到队列中
            for _, row in neighbors.iterrows():
                if row['source_entity_id'] not in visited:
                    queue.append(row['source_entity_id'])
                if row['target_entity_id'] not in visited:
                    queue.append(row['target_entity_id'])

    # 使用找到的所有相关节点过滤关系，并排除 relationship_description 为 "SAME_NAME" 的关系
    related_relationships = df_relationships[
        (df_relationships['source_entity_id'].isin(visited)) & 
        (df_relationships['target_entity_id'].isin(visited)) &
        (df_relationships['relationship_description'] != "SAME_NAME")
    ]

    return related_relationships