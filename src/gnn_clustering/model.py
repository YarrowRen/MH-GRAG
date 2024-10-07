import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# 定义多头 GNN 模型
class MultiHeadGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(MultiHeadGNN, self).__init__()
        self.num_heads = num_heads
        self.gnns = torch.nn.ModuleList([
            GNN(in_channels, hidden_channels, out_channels) for _ in range(num_heads)
        ])

    def forward(self, x, edge_index):
        embeddings_list = []
        for gnn in self.gnns:
            embeddings = gnn(x, edge_index)
            embeddings_list.append(embeddings)
        return embeddings_list


def get_model(data, hidden_channels=16, out_channels=16, device='cpu'):
    model = GNN(data.num_features, hidden_channels, out_channels).to(device)
    return model


def get_multi_head_model(data, hidden_channels=16, out_channels=16, device='cpu', num_heads=3):
    model = MultiHeadGNN(data.num_features, hidden_channels, out_channels, num_heads=num_heads).to(device)
    return model