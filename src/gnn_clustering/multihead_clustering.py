import igraph as ig
import leidenalg
from src.gnn_clustering.evaluate import (
    get_embeddings,
    leiden_clustering
)
from src.gnn_clustering.model import get_model

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
