import torch
import torch.nn.functional as F

def modularity_loss(embeddings, adj, reg_lambda=1e-3):
    # 对嵌入向量进行归一化
    embeddings = F.normalize(embeddings, p=2, dim=1)
    # 计算节点之间的嵌入相似度
    sim = torch.mm(embeddings, embeddings.t())
    # 计算度数和总边数
    degrees = adj.sum(dim=1)
    m = adj.sum()
    # 期望的边数
    expected = torch.outer(degrees, degrees) / m
    # 模块度矩阵
    B = adj - expected
    # 计算模块度
    modularity = (sim * B).sum() / m
    # 添加正则化项
    reg = reg_lambda * (embeddings.norm(dim=1) ** 2).sum()
    # 总损失
    loss = - modularity + reg
    return loss
