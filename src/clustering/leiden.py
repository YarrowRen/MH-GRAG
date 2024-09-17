"""
Implementation of the Leiden algorithm for community detection.
"""

from collections.abc import Set
from math import exp
from random import choices, shuffle
from typing import TypeVar

import networkx as nx
from networkx import Graph
from typing import Union

from sklearn.metrics import adjusted_mutual_info_score

from .quality_functions import QualityFunction
from .utils import DataKeys as Keys
from .utils import Partition, argmax, freeze, node_total, preprocess_graph

T = TypeVar("T")


def move_nodes_fast(G: Graph, 𝓟: Partition[T], 𝓗: QualityFunction[T], other_partitions: list[Partition[T]], λ: float) -> Partition[T]:
    """
    通过快速的局部节点移动来改善社区划分的质量，同时考虑与其他头的互信息正则化以鼓励多样性。
    
    参数
    ----------
    G : Graph
        要处理的图/网络。
    𝓟 : Partition[T]
        当前正在优化的社区划分。
    𝓗 : QualityFunction[T]
        要优化的质量函数。
    other_partitions : list[Partition[T]]
        其他头的社区划分，用于互信息正则化。
    λ : float
        互信息正则化的权重。
    
    返回
    -------
    Partition[T]
        节点移动后的更新划分。
    """
    
    # 将所有节点随机排列，开始进行节点移动
    Q = list(G.nodes)
    shuffle(Q)

    # 创建一个集合用于记录已访问的节点
    visited_nodes = set()

    # 循环直到所有节点都被处理
    while True:
        # 从队列中弹出一个节点v，并将其标记为已访问
        v = Q.pop(0)
        visited_nodes.add(v)

        # 定义计算互信息惩罚的函数，目标是惩罚过于相似的社区划分
        def mutual_info_penalty(target_community):
            penalty = 0
            # 遍历其他头的社区划分，累加互信息惩罚
            for other_partition in other_partitions:
                penalty += mutual_info(𝓟, other_partition)
            return penalty

        # 计算v节点移动到每个相邻社区的质量提升，并结合互信息惩罚
        (Cₘ, 𝛥𝓗, _) = argmax(
            lambda C: 𝓗.delta(𝓟, v, C) - λ * mutual_info_penalty(C),
            [*𝓟.adjacent_communities(v), set()]  # 邻接社区和空集作为候选目标
        )

        # 如果移动v节点可以提升质量（𝛥𝓗 > 0），则进行移动
        if 𝛥𝓗 > 0:
            𝓟.move_node(v, Cₘ)  # 将节点v移动到目标社区Cₘ
            
            # 找出与v相连但尚未访问的节点，将其加入队列进行处理
            N = {u for u in G[v] if u not in Cₘ and u not in visited_nodes}
            Q.extend(N - set(Q))  # 确保新节点不会重复加入队列

        # 如果队列Q为空，表示所有节点都已处理完毕，返回最终的社区划分
        if len(Q) == 0:
            return 𝓟




def refine_partition(G: Graph, 𝓟: Partition[T], 𝓗: QualityFunction[T], θ: float, γ: float) -> Partition[T]:
    """通过重复合并的方式细化所有社区，从单节点划分开始。"""
    
    # 将每个节点分配到它自己的社区（初始划分为单节点社区）
    𝓟ᵣ: Partition[T] = Partition.singleton_partition(G, Keys.WEIGHT)

    # 遍历所有社区
    for C in 𝓟:
        # 细化当前社区
        𝓟ᵣ = merge_nodes_subset(G, 𝓟ᵣ, 𝓗, θ, γ, C)

    return 𝓟ᵣ



def merge_nodes_subset(G: Graph, 𝓟: Partition[T], 𝓗: QualityFunction[T], θ: float, γ: float, S: Set[T]) -> Partition[T]:
    """将子集 S 中的节点合并为一个或多个集合，以细化划分 𝓟。"""
    
    # 计算子集 S 的节点总权重
    size_s = node_total(G, S)

    # 选择切割代价较高的节点集合 R
    R = {
        v for v in S
          if nx.cut_size(G, [v], S - {v}, weight=Keys.WEIGHT) >= γ * node_total(G, v) * (size_s - node_total(G, v))
    }  # fmt: skip

    for v in R:
        # 如果 v 是单节点社区，即尚未被合并的节点
        if len(𝓟.node_community(v)) == 1:
            # 只考虑连接良好的社区
            𝓣 = freeze([
                C for C in 𝓟
                  if C <= S and nx.cut_size(G, C, S - C, weight=Keys.WEIGHT) >= γ * float(node_total(G, C) * (size_s - node_total(G, C)))
            ])  # fmt: skip

            # 随机选择一个社区将 v 放入
            # 使用 Python 的 random.choices 进行加权随机选择

            # 列出 𝓣 中的社区以及将 v 移动到该社区的质量函数提升（𝛥𝓗）
            # 只考虑移动 v 到质量函数不变或提升的社区
            communities = [(C, 𝛥𝓗) for (C, 𝛥𝓗) in ((C, 𝓗.delta(𝓟, v, C)) for C in 𝓣) if 𝛥𝓗 >= 0]
            # 根据 𝛥𝓗 值计算随机选择的权重
            weights = [exp(𝛥𝓗 / θ) for (C, 𝛥𝓗) in communities]

            # 最终选择新社区
            # 使用 [0][0] 提取社区，因为 choices 返回的是包含单个 (C, 𝛥𝓗) 元组的列表
            Cₙ = choices(communities, weights=weights, k=1)[0][0]

            # 将 v 移动到选中的社区
            𝓟.move_node(v, Cₙ)

    return 𝓟



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


def multi_head_leiden_with_mutual_info(
    G: Graph, 𝓗: QualityFunction[T], num_heads: int, λ: float, θ: float = 0.3, γ: float = 0.05, weight: Union[str, None] = None
) -> list[Partition[T]]:
    """
    使用多头机制和互信息正则化进行Leiden算法的社区发现。

    参数
    ----------
    G : Graph
        要处理的图/网络。
    𝓗 : QualityFunction[T]
        优化的质量函数。
    num_heads: int
        生成不同社区划分的头的数量。
    λ : float
        互信息正则化在总损失中的权重。
    θ : float, 可选
        Leiden算法的θ参数，默认为0.3。
    γ : float, 可选
        Leiden算法的γ参数，默认为0.05。
    weight: str | None
        图中边的权重属性，默认为None。

    返回
    -------
    list[Partition[T]]
        应用了互信息正则化的不同头生成的社区划分列表。
    """
    partitions = []
    for i in range(num_heads):
        # 传递之前生成的头作为 `other_partitions`
        other_partitions = partitions[:i]  # 每个头传入之前生成的头
        
        # 调用Leiden算法，同时引入互信息正则化
        𝓟ₚ = None
        G_current = preprocess_graph(G, weight)

        if len(partitions) > 0:
            𝓟 = Partition.from_partition(G_current, partitions[i-1], Keys.WEIGHT)
        else:
            𝓟 = Partition.singleton_partition(G_current, Keys.WEIGHT)

        while True:
            # 进行局部节点移动，传入其他头的社区划分和正则化参数λ
            𝓟 = move_nodes_fast(G_current, 𝓟, 𝓗, other_partitions=other_partitions, λ=λ)

            # 检查终止条件：若社区仅由单个节点组成或划分结果收敛
            if len(𝓟) == G_current.order() or 𝓟 == 𝓟ₚ:
                partitions.append(𝓟.flatten())
                break

            𝓟ₚ = 𝓟

            # 使用θ和γ参数对局部划分进行细化
            𝓟ᵣ = refine_partition(G_current, 𝓟, 𝓗, θ, γ)

            # 创建基于当前划分的聚合图
            G_current = 𝓟ᵣ.aggregate_graph()

            # 将原始社区划分映射到聚合图
            partitions_dict: dict[int, set[T]] = {id: set() for id in range(len(𝓟))}
            for v_agg, nodes in G_current.nodes(data=Keys.NODES):
                community_id = 𝓟._node_part[next(iter(nodes))]
                partitions_dict[community_id] = partitions_dict[community_id].union({v_agg})

            # 将划分结果转换为列表形式
            partitions_l: list[set[T]] = list(partitions_dict.values())
            𝓟 = Partition.from_partition(G_current, partitions_l, Keys.WEIGHT)

    return partitions

