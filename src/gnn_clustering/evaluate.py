import torch
import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
import igraph as ig
import leidenalg
import random
from torch_geometric.utils import to_networkx

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
