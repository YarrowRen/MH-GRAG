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

# 定义模块度损失函数，包含正交正则化
def modularity_loss_multi_head(embeddings_list, adj, num_heads, reg_lambda=1e-3, orth_lambda=1.0):
    losses = []
    for i in range(num_heads):
        embeddings = embeddings_list[i]
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
        losses.append(loss)
    
    # 计算正交正则化项
    orth_loss = 0
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            # 计算两个头的嵌入矩阵的内积
            # embeddings_list[i] 和 embeddings_list[j] 的形状为 [num_nodes, embedding_dim]
            # 先对嵌入进行归一化
            h_i = F.normalize(embeddings_list[i], p=2, dim=1)
            h_j = F.normalize(embeddings_list[j], p=2, dim=1)
            # 计算内积矩阵
            inner_product = torch.mm(h_i.t(), h_j)
            # 计算 Frobenius 范数的平方
            orth_loss += torch.norm(inner_product, p='fro') ** 2

    # 总损失
    total_loss = sum(losses) + orth_lambda * orth_loss
    return total_loss