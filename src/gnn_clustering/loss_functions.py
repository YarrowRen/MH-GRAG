import torch
import torch.nn.functional as F

def modularity_loss(embeddings, adj, reg_lambda=1e-3):
    """
    计算模块度损失函数，该函数用于评估图的聚类质量。首先对节点的嵌入向量进行归一化，
    然后计算节点之间的嵌入相似度。接着，计算图中每个节点的度数和总边数，并基于此计算期望的边数。
    通过邻接矩阵减去期望的边数矩阵得到模块度矩阵。模块度通过嵌入相似度与模块度矩阵的乘积求和并归一化得到。
    最后，添加一个正则化项以防止过拟合，最终返回总损失值。
    
    参数:
    embeddings (Tensor): 节点的嵌入向量。
    adj (Tensor): 邻接矩阵。
    reg_lambda (float): 正则化项的权重。

    返回:
    loss (float): 模块度损失值。
    """
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim = torch.mm(embeddings, embeddings.t())
    degrees = adj.sum(dim=1)
    m = adj.sum()
    expected = torch.outer(degrees, degrees) / m
    B = adj - expected
    modularity = (sim * B).sum() / m
    reg = reg_lambda * (embeddings.norm(dim=1) ** 2).sum()
    loss = - modularity + reg
    return loss

def modularity_loss_multi_head(embeddings_list, adj, num_heads, reg_lambda=1e-3, orth_lambda=1.0):
    """
    该函数定义了一个多头模块度损失函数，包含正交正则化。首先，对每个头的嵌入向量进行归一化，
    然后计算节点之间的嵌入相似度。接着，计算图中每个节点的度数和总边数，并基于此计算期望的边数。
    通过邻接矩阵减去期望的边数矩阵得到模块度矩阵。模块度通过嵌入相似度与模块度矩阵的乘积求和并归一化得到。
    添加一个正则化项以防止过拟合。对于正交正则化项，计算不同头之间的嵌入矩阵的内积，并计算其 Frobenius 范数的平方。
    最终返回总损失值。
    
    参数:
    embeddings_list (list of Tensor): 每个头的节点嵌入向量列表。
    adj (Tensor): 邻接矩阵。
    num_heads (int): 头的数量。
    reg_lambda (float): 正则化项的权重。
    orth_lambda (float): 正交正则化项的权重。

    返回:
    total_loss (float): 总损失值。
    """
    losses = []
    for i in range(num_heads):
        embeddings = embeddings_list[i]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim = torch.mm(embeddings, embeddings.t())
        degrees = adj.sum(dim=1)
        m = adj.sum()
        expected = torch.outer(degrees, degrees) / m
        B = adj - expected
        modularity = (sim * B).sum() / m
        reg = reg_lambda * (embeddings.norm(dim=1) ** 2).sum()
        loss = - modularity + reg
        losses.append(loss)
    
    orth_loss = 0
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            h_i = F.normalize(embeddings_list[i], p=2, dim=1)
            h_j = F.normalize(embeddings_list[j], p=2, dim=1)
            inner_product = torch.mm(h_i.t(), h_j)
            orth_loss += torch.norm(inner_product, p='fro') ** 2

    total_loss = sum(losses) + orth_lambda * orth_loss
    return total_loss