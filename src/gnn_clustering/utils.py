import torch
from torch_geometric.utils import to_dense_adj

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dense_adj(edge_index, device='cpu'):
    adj = to_dense_adj(edge_index)[0].to(device)
    return adj
