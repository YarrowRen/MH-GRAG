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

# 获取多头 Leiden 聚类结果
def get_multihead_leiden_results(data, device, num_heads=3, learning_rate=0.01, leiden_K=3):
    multi_head_leiden_results = []

    # 初始模型和优化器
    multi_head_model = get_multi_head_model(data=data, device=device, num_heads=num_heads)
    optimizer = torch.optim.Adam(multi_head_model.parameters(), lr=learning_rate)

    # 获取密集邻接矩阵
    adj = get_dense_adj(data.edge_index, device=device)

    # 模型训练
    trained_model = train_model_multi_head(multi_head_model, data, adj, optimizer, num_heads)
    embeddings_list_after_training = get_embeddings_list(trained_model, data, device)

    # 存储每个头的聚类结果
    clustering_results = []
    modularities_after_training = []

    # 基于训练后嵌入再次调用封装好的Leiden聚类方法
    for i, post_training_embeddings in enumerate(embeddings_list_after_training):
        leiden_communities_after_training, leiden_modularity_after_training = leiden_embeddings_clustering(post_training_embeddings, K=leiden_K)
        modularities_after_training.append(leiden_modularity_after_training)
        clustering_results.append(leiden_communities_after_training)

    multi_head_leiden_results.append((clustering_results, modularities_after_training))

    return multi_head_leiden_results