import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

# 创建原始图，节点为entity_id，边为source_entity_id和target_entity_id
def create_graph_from_df(entities_df, relationships_df):
    entity_id_to_index = {entity_id: idx for idx, entity_id in enumerate(entities_df['entity_id'])}
    source_indices = relationships_df['source_entity_id'].map(entity_id_to_index).values
    target_indices = relationships_df['target_entity_id'].map(entity_id_to_index).values
    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
    graph_data = Data(edge_index=edge_index)  # 创建PyTorch Geometric图数据对象
    return graph_data, entity_id_to_index

# 基于embedding生成K近邻图，并为每条边分配固定的权重
def generate_knn_graph_from_embeddings(embedding, k=10, fixed_weight=1.0):
    embedding = embedding.cpu().numpy()  # 确保在CPU上进行操作
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(embedding)
    distances, indices = nbrs.kneighbors(embedding)

    num_nodes = embedding.shape[0]
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # 为K近邻图的每条边分配固定的权重
    for i in range(num_nodes):
        for j in range(1, k):  # 跳过第一个点自己本身
            adjacency_matrix[i, indices[i][j]] = fixed_weight  # 设置固定权重

    return adjacency_matrix

# 将K近邻生成的邻接矩阵转换为PyTorch Geometric的edge_index和edge_weight
def knn_to_edge_index(knn_graph):
    sources, targets = knn_graph.nonzero()  # 获取非零元素索引
    weights = knn_graph[sources, targets]   # 获取对应的权重
    edge_index_knn = torch.tensor([sources, targets], dtype=torch.long)
    edge_weight_knn = torch.tensor(weights, dtype=torch.float)
    return edge_index_knn, edge_weight_knn

# 原始图和每个头的K近邻图分别融合
def combine_graphs_separately(original_graph, knn_graphs):
    combined_graphs = []
    edge_index_original = original_graph.edge_index
    edge_weight_original = torch.ones(original_graph.edge_index.shape[1])  # 固定权重为1

    for idx, knn_graph in enumerate(knn_graphs):
        # 将K近邻图转换为edge_index和edge_weight格式
        edge_index_knn, edge_weight_knn = knn_to_edge_index(knn_graph)

        # 分别将原始图与每个头的K近邻图融合
        edge_index_combined = torch.cat([edge_index_original, edge_index_knn], dim=1)
        edge_weight_combined = torch.cat([edge_weight_original, edge_weight_knn], dim=0)

        print(f"Head {idx}: Combined edge index shape: {edge_index_combined.shape}")
        print(f"Head {idx}: Combined edge weights shape: {edge_weight_combined.shape}")
        
        # 存储融合后的图
        combined_graphs.append((edge_index_combined, edge_weight_combined))
    
    return combined_graphs
