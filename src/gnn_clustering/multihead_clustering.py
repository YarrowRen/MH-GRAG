import torch
import pandas as pd
from src.gnn_clustering.train import train_model_multi_head
from src.gnn_clustering.evaluate import (
    get_embeddings,
    get_embeddings_list,
    leiden_embeddings_clustering,
    leiden_clustering
)
from src.gnn_clustering.model import get_multi_head_model, get_model
from src.gnn_clustering.utils import get_dense_adj, communities_to_labels, calculate_ami_between_heads

# 获取初始 Leiden 聚类结果
def get_initial_leiden_results(data, device, leiden_K=3):
    initial_leiden_results = []

    # 使用Leiden算法进行聚类，获取群数和模块度
    initial_communities_leiden, initial_modularity_leiden = leiden_clustering(data)
    initial_leiden_results.append((initial_communities_leiden, initial_modularity_leiden))

    # 初始模型和基于嵌入的 Leiden 聚类
    single_head_model = get_model(data, device=device)
    initial_embeddings = get_embeddings(single_head_model, data, device=device)

    return initial_leiden_results

# # 获取多头 Leiden 聚类结果
# def get_multihead_leiden_results(data, device, num_heads=3, learning_rate=0.01, leiden_K=3):
#     multi_head_leiden_results = []

#     # 初始模型和优化器
#     multi_head_model = get_multi_head_model(data=data, device=device, num_heads=num_heads)
#     optimizer = torch.optim.Adam(multi_head_model.parameters(), lr=learning_rate)

#     # 获取密集邻接矩阵
#     adj = get_dense_adj(data.edge_index, device=device)

#     # 模型训练
#     trained_model = train_model_multi_head(multi_head_model, data, adj, optimizer, num_heads)
#     embeddings_list_after_training = get_embeddings_list(trained_model, data, device)

#     # 存储每个头的聚类结果
#     clustering_results = []
#     modularities_after_training = []

#     # 基于训练后嵌入再次调用封装好的Leiden聚类方法
#     for i, post_training_embeddings in enumerate(embeddings_list_after_training):
#         leiden_communities_after_training, leiden_modularity_after_training = leiden_embeddings_clustering(post_training_embeddings, K=leiden_K)
#         modularities_after_training.append(leiden_modularity_after_training)
#         clustering_results.append(leiden_communities_after_training)

#     multi_head_leiden_results.append((clustering_results, modularities_after_training))

#     return multi_head_leiden_results









import torch
import numpy as np
import igraph as ig
import leidenalg
from sklearn.metrics import normalized_mutual_info_score
from src.utils.graph_utils import create_graph_from_df, combine_graphs_separately

# 将PyTorch Geometric格式的图数据转换为igraph格式
def convert_to_igraph(edge_index):
    num_nodes = int(edge_index.max().item() + 1)
    edges = edge_index.t().tolist()  # 转换为边的列表
    graph = ig.Graph(edges=edges, directed=False)
    graph.add_vertices(num_nodes - len(graph.vs))  # 添加缺失的顶点
    return graph

# 使用Leiden算法执行社区检测
def run_leiden(graph, edge_weights=None):
    partition = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition, weights=edge_weights)
    return partition


def get_clustering_results(combined_graphs):
    partitions = []
    
    # 对每个头的图执行Leiden算法
    for idx, (edge_index, edge_weights) in enumerate(combined_graphs):
        
        # 转换为 igraph 格式
        graph = convert_to_igraph(edge_index)
        
        # 使用Leiden算法进行社区划分
        partition = run_leiden(graph, edge_weights=edge_weights)
        
        # 存储每个头的聚类结果
        partitions.append(partition)

    # 只返回聚类结果的集合（包含原始图和每个头的聚类结果）
    return partitions
