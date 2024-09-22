import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.nn import GCNConv
from collections.abc import Set
from random import shuffle
from copy import deepcopy
from typing import List, TypeVar

from collections.abc import Set
from math import exp
from random import choices, shuffle
from typing import TypeVar, List

from copy import deepcopy
import networkx as nx
from networkx import Graph

from sklearn.metrics import adjusted_mutual_info_score

from .quality_functions import QualityFunction
from .utils import DataKeys as Keys
from .utils import Partition, argmax, freeze, node_total, preprocess_graph

T = TypeVar("T")

class GCNLeiden(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, θ=0.3, γ=0.05, λ=0.01):
        super(GCNLeiden, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)
        self.num_heads = num_heads
        self.θ = θ
        self.γ = γ
        self.λ = λ
        self.partitions = []
    
    def forward(self, x, edge_index, G, 𝓗, weight='weight'):  # 添加weight参数，默认值为'weight'
        # 确保图中的所有边都有权重属性
        for u, v in G.edges:
            if weight not in G[u][v]:
                G[u][v][weight] = 1  # 为缺少的边权重添加默认值

        # 生成节点嵌入
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        
        # 初始化每个头的Leiden分区
        partitions = [Partition.singleton_partition(G) for _ in range(self.num_heads)]
        # 调用 multi_head_leiden_with_gnn，传递weight参数
        self.partitions = self.multi_head_leiden_with_gnn(G, 𝓗, partitions, x, weight=weight)
        
        return x
    
    def multi_head_leiden_with_gnn(self, G, 𝓗, partitions, x, weight):  # 添加weight参数
        """
        基于节点嵌入的多头Leiden算法，结合GNN生成的嵌入进行节点移动。
        
        参数
        ----------
        G : Graph
            要处理的图/网络。
        𝓗 : QualityFunction[T]
            要优化的质量函数。
        partitions : List[Partition[T]]
            当前正在优化的分区列表。
        x : Tensor
            GNN生成的节点嵌入。
        weight : str
            边的权重属性。
        
        返回
        -------
        List[Partition[T]]
            更新后的分区列表。
        """
        Q = list(G.nodes)
        shuffle(Q)
        
        visited_nodes = [set() for _ in range(self.num_heads)]
        while True:
            if not Q:
                break
            v = Q.pop(0)
            for i in range(self.num_heads):
                if v in visited_nodes[i]:
                    continue
                visited_nodes[i].add(v)
                
                # 计算节点v移动到目标社区的嵌入相似度和互信息惩罚
                def mutual_info_penalty(target_community, head_index):
                    penalty = 0
                    for j in range(self.num_heads):
                        if j != head_index:
                            penalty += mutual_info_single_node(partitions[head_index], partitions[j], v, target_community)
                    return penalty
                
                def embedding_similarity(node, target_community):
                    # 根据嵌入计算节点v与目标社区的相似度
                    target_embedding = x[list(target_community)].mean(dim=0)
                    node_embedding = x[node]
                    return torch.cosine_similarity(node_embedding, target_embedding, dim=0)
                
                # 寻找节点v在第i个头中的最佳社区
                adj_communities = list(partitions[i].adjacent_communities(v)) + [set()]
                (Cₘ, 𝛥𝓗, _) = argmax(
                    lambda C: 𝓗.delta(partitions[i], v, C) + embedding_similarity(v, C) - self.λ * mutual_info_penalty(C, i),
                    adj_communities
                )
                
                if 𝛥𝓗 > 0:
                    partitions[i].move_node(v, Cₘ)
                    N = {u for u in G[v] if u not in Cₘ and u not in visited_nodes[i]}
                    Q.extend(N - set(Q))
        
        return partitions
    
    def loss_fn(self, G, 𝓗):
        """
        损失函数，结合模块度损失和其他任务损失（如节点分类）。
        """
        # 计算基于Leiden算法的模块度损失
        modularity_loss = 0
        for partition in self.partitions:
            modularity_loss += 𝓗.compute(partition)
        
        # 其他任务损失可以通过分类或回归任务添加
        # 此处只是返回模块度损失作为示例
        return -modularity_loss

def mutual_info(partition1: Partition[T], partition2: Partition[T]) -> float:
    """
    Calculate the mutual information between two partitions using scikit-learn's adjusted_mutual_info_score.
    
    Parameters
    ----------
    partition1 : Partition[T]
        The first partition of nodes into communities.
    partition2 : Partition[T]
        The second partition of nodes into communities.
    
    Returns
    -------
    float
        The adjusted mutual information score between the two partitions.
    """
    # 获取每个节点对应的社区标签
    labels1 = []
    labels2 = []
    
    # 将每个节点映射到其对应的社区标签
    node_to_label1 = {node: i for i, community in enumerate(partition1.communities) for node in community}
    node_to_label2 = {node: i for i, community in enumerate(partition2.communities) for node in community}
    
    # 对每个节点的标签进行排序，使得它们按照相同的顺序进行比较
    for node in partition1.G.nodes():
        labels1.append(node_to_label1[node])
        labels2.append(node_to_label2[node])
    # 使用 scikit-learn 计算调整后的互信息分数
    return adjusted_mutual_info_score(labels1, labels2)

def mutual_info_single_node(
    partition1: Partition[T],
    partition2: Partition[T],
    node: T,
    target_community: Set[T]
) -> float:
    """
    计算将节点移动到目标社区后，与另一个分区的互信息变化。
    
    参数
    ----------
    partition1 : Partition[T]
        节点正在移动的分区。
    partition2 : Partition[T]
        用于比较的另一个分区。
    node : T
        正在移动的节点。
    target_community : Set[T]
        partition1中的目标社区。
        
    返回
    -------
    float
        互信息的变化。
    """
    # 由于互信息是全局度量，精确计算单个节点移动的影响较为复杂。
    # 这里近似地通过在节点移动后重新计算互信息来估计变化量。
    # 注意：这在计算上可能会比较耗时，需要优化。
    
    # 创建partition1的副本并移动节点
    partition1_temp = deepcopy(partition1)
    partition1_temp.move_node(node, target_community)
    
    # 计算移动后partition1_temp与partition2之间的互信息
    return mutual_info(partition1_temp, partition2) - mutual_info(partition1, partition2)
