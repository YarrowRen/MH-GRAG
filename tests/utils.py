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

# 测试模型性能的方法
def test_model_performance(data, device, num_tests):
    reports = []

    for report_id in range(1, num_tests + 1):
        # 使用Leiden算法进行聚类，获取簇数和模块度
        initial_communities_leiden, initial_modularity_leiden = leiden_clustering(data)
        initial_num_clusters_leiden = len(initial_communities_leiden)

        # 初始模型和优化器
        num_heads = 3
        multi_head_model = get_multi_head_model(data=data, device=device, num_heads=num_heads)
        single_head_model = get_model(data, device=device)
        optimizer = torch.optim.Adam(multi_head_model.parameters(), lr=0.01)

        # 获取密集邻接矩阵
        adj = get_dense_adj(data.edge_index, device=device)

        # 初始嵌入和基于嵌入的 Leiden 聚类
        initial_embeddings = get_embeddings(single_head_model, data, device=device)
        leiden_communities_before_training, leiden_modularity_before_training = leiden_embeddings_clustering(initial_embeddings, K=3)
        leiden_num_clusters_before_training = len(leiden_communities_before_training)

        # 模型训练
        trained_model = train_model_multi_head(multi_head_model, data, adj, optimizer, num_heads)
        embeddings_list_after_training = get_embeddings_list(trained_model, data, device)

        # 存储每个头的聚类结果
        clustering_results = []
        modularities_after_training = []
        num_clusters_after_training = []

        # 基于训练后嵌入再次调用封装好的Leiden聚类方法
        for i, post_training_embeddings in enumerate(embeddings_list_after_training):
            leiden_communities_after_training, leiden_modularity_after_training = leiden_embeddings_clustering(post_training_embeddings, K=3)
            modularities_after_training.append(leiden_modularity_after_training)
            num_clusters_after_training.append(len(leiden_communities_after_training))

            # 将社区划分转换为节点标签
            predicted_labels = communities_to_labels(leiden_communities_after_training, data.num_nodes)
            clustering_results.append(predicted_labels)

        # 计算多个头之间的 AMI
        ami_matrix = calculate_ami_between_heads(clustering_results)

        # 计算平均值
        avg_modularity_after_training = sum(modularities_after_training) / num_heads
        avg_num_clusters_after_training = sum(num_clusters_after_training) / num_heads
        avg_ami = sum(sum(row) for row in ami_matrix) / (num_heads * (num_heads - 1))

        # 计算提升百分比
        improvement_over_initial_leiden = ((avg_modularity_after_training) / initial_modularity_leiden) * 100
        improvement_over_initial_embedding_leiden = ((avg_modularity_after_training ) / leiden_modularity_before_training) * 100

        # 构建报告数据
        report_data = {
            'id': report_id,
            'nodes': data.num_nodes,
            'edges': data.num_edges,
            'initial_modularity': initial_modularity_leiden,
            'initial_clusters': initial_num_clusters_leiden,
            'initial_modularity_embedding': leiden_modularity_before_training,
            'initial_clusters_embedding': leiden_num_clusters_before_training,
            'avg_modularity_after_training': avg_modularity_after_training,
            'avg_clusters_after_training': avg_num_clusters_after_training,
            'avg_ami': avg_ami,
            'improvement_leiden (%)': improvement_over_initial_leiden,
            'improvement_embedding_leiden (%)': improvement_over_initial_embedding_leiden
        }

        # 添加每个头的聚类信息
        for i in range(num_heads):
            report_data[f'head_{i + 1}_modularity'] = modularities_after_training[i]
            report_data[f'head_{i + 1}_num_clusters'] = num_clusters_after_training[i]

        # 添加AMI矩阵信息（只保留上三角部分，避免重复）
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                report_data[f'head_{i + 1}_ami_with_head_{j + 1}'] = ami_matrix[i][j]

        reports.append(report_data)

    return pd.DataFrame(reports)
