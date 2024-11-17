import torch
import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
import igraph as ig
import leidenalg
import random
from torch_geometric.utils import to_networkx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def get_embeddings(model, data, device='cpu'):
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x.to(device), data.edge_index.to(device))
    return embeddings.cpu().numpy()

def get_embeddings_list(model, data, device='cpu'):
    # 测试模型并获取嵌入
    model.eval()
    with torch.no_grad():
        embeddings_list = model(data.x, data.edge_index)
    return embeddings_list

def kmeans_clustering(embeddings, n_clusters=7):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings)
    return kmeans.labels_

def leiden_clustering(data):
    edge_index = data.edge_index.cpu().numpy()
    edges = list(zip(edge_index[0], edge_index[1]))
    G_ig = ig.Graph(edges=edges, directed=False)
    partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)
    communities = [list(community) for community in partition]
    modularity = partition.modularity
    return communities, modularity

def random_clustering(num_nodes, n_clusters=7):
    random_labels = [random.randint(0, n_clusters - 1) for _ in range(num_nodes)]
    communities = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(random_labels):
        communities[label].append(idx)
    return communities

def compute_modularity(data, communities):
    G_nx = to_networkx(data, to_undirected=True)
    modularity = nx.algorithms.community.modularity(G_nx, communities)
    return modularity

def format_communities(labels, n_clusters):
    communities = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(labels):
        communities[label].append(idx)
    return communities

# 封装的leiden_embeddings_clustering方法
def leiden_embeddings_clustering(embeddings, K=3):
    """
    进行基于嵌入的Leiden聚类，返回社区划分结果和模块度。

    参数：
    embeddings (np.ndarray): 节点的嵌入向量。
    K (int): K近邻的数量，默认为3。

    返回：
    communities (leidenalg.VertexPartition): 社区划分结果。
    modularity (float): Leiden算法的模块度。
    """

    # 判断 embeddings 是否为张量
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    # 计算相似度矩阵
    similarity_matrix = cosine_similarity(embeddings)
    
    # 使用K近邻找到邻居节点
    nbrs = NearestNeighbors(n_neighbors=K, metric='cosine').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    
    # 构建边和权重
    edges = []
    weights = []
    num_nodes = embeddings.shape[0]
    for i in range(num_nodes):
        for j_idx in range(1, K):  # 从 1 开始，避免自环
            j = indices[i][j_idx]
            if i != j:
                edges.append((i, j))
                weights.append(similarity_matrix[i][j])
    
    # 使用 NetworkX 构建图
    G = nx.Graph()
    for edge, weight in zip(edges, weights):
        G.add_edge(edge[0], edge[1], weight=weight)
    
    # 将 NetworkX 图转换为 igraph 图
    G_ig = ig.Graph.TupleList(G.edges(data=True), weights=True, directed=False)
    
    # 使用Leiden算法进行社区划分
    partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)

    # 返回社区划分结果和模块度
    return partition, partition.modularity
