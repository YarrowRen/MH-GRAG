import torch
from torch_geometric.datasets import Planetoid

def load_data(dataset_name='Cora', root='/tmp/Cora'):
    dataset = Planetoid(root=root, name=dataset_name)
    data = dataset[0]
    return data

import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np

def load_custom_data(entities_df, relationships_df):
    # Step 1: 创建节点映射（从 entity_id 到节点索引）
    entity_ids = entities_df['entity_id'].tolist()
    id_to_index = {entity_id: idx for idx, entity_id in enumerate(entity_ids)}
    num_nodes = len(entity_ids)

    # Step 2: 处理节点特征（如果有嵌入则使用嵌入，否则可以使用其他特征）
    if 'entity_embedding' in entities_df.columns:
        # 假设 entity_embedding 列中存储的是列表或 numpy 数组
        embeddings = np.vstack(entities_df['entity_embedding'].values)
    else:
        # 如果没有嵌入，可以使用 one-hot 编码或其他方法
        embeddings = np.eye(num_nodes)

    # 转换为 torch 张量
    x = torch.tensor(embeddings, dtype=torch.float)

    # Step 3: 创建边索引
    # 使用 source_entity_id 和 target_entity_id 列
    source_ids = relationships_df['source_entity_id'].tolist()
    target_ids = relationships_df['target_entity_id'].tolist()

    # 将 entity_id 转换为节点索引
    source_indices = [id_to_index[sid] for sid in source_ids]
    target_indices = [id_to_index[tid] for tid in target_ids]

    # 创建边索引张量
    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)

    # 如果关系是无向的，可以添加反向边
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Step 4: 处理边特征（如果需要）
    # 如果有关系嵌入，可以处理 edge_attr
    if 'relationship_embedding' in relationships_df.columns:
        edge_embeddings = np.vstack(relationships_df['relationship_embedding'].values)
        # 由于我们添加了反向边，需要将边特征也复制一遍
        edge_attr = torch.tensor(np.concatenate([edge_embeddings, edge_embeddings], axis=0), dtype=torch.float)
    else:
        edge_attr = None

    # Step 5: 创建 Data 对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data