import torch
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