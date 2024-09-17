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


def move_nodes_fast_parallel(
    G: Graph,
    partitions: List[Partition[T]],
    𝓗: QualityFunction[T],
    λ: float
) -> List[Partition[T]]:
    """
    对多个分区（头）同时执行快速局部节点移动，考虑互信息正则化。
    
    参数
    ----------
    G : Graph
        要处理的图/网络。
    partitions : List[Partition[T]]
        当前正在优化的分区列表。
    𝓗 : QualityFunction[T]
        要优化的质量函数。
    λ : float
        互信息正则化的权重。
        
    返回
    -------
    List[Partition[T]]
        节点移动后的更新分区列表。
    """
    Q = list(G.nodes)
    shuffle(Q)
    
    num_heads = len(partitions)
    visited_nodes = [set() for _ in range(num_heads)]
    while True:
        if not Q:
            break
        v = Q.pop(0)
        for i in range(num_heads):
            if v in visited_nodes[i]:
                continue
            visited_nodes[i].add(v)
            
            # 计算节点v移动到目标社区的互信息惩罚
            def mutual_info_penalty(target_community, head_index):
                penalty = 0
                for j in range(num_heads):
                    if j != head_index:
                        penalty += mutual_info_single_node(
                            partitions[head_index],
                            partitions[j],
                            v,
                            target_community
                        )
                return penalty
            
            # 寻找节点v在第i个头中的最佳社区
            adj_communities = list(partitions[i].adjacent_communities(v)) + [set()]
            (Cₘ, 𝛥𝓗, _) = argmax(
                lambda C: 𝓗.delta(partitions[i], v, C) - λ * mutual_info_penalty(C, i),
                adj_communities
            )
            
            if 𝛥𝓗 > 0:
                partitions[i].move_node(v, Cₘ)
                N = {u for u in G[v] if u not in Cₘ and u not in visited_nodes[i]}
                Q.extend(N - set(Q))
    return partitions



def refine_partition(G: Graph, 𝓟: Partition[T], 𝓗: QualityFunction[T], θ: float, γ: float) -> Partition[T]:
    """Refine all communities by merging repeatedly, starting from a singleton partition."""
    # Assign each node to its own community
    𝓟ᵣ: Partition[T] = Partition.singleton_partition(G, Keys.WEIGHT)

    # Visit all communities
    for C in 𝓟:
        # refine community
        𝓟ᵣ = merge_nodes_subset(G, 𝓟ᵣ, 𝓗, θ, γ, C)

    return 𝓟ᵣ


def merge_nodes_subset(G: Graph, 𝓟: Partition[T], 𝓗: QualityFunction[T], θ: float, γ: float, S: Set[T]) -> Partition[T]:
    """Merge the nodes in the subset S into one or more sets to refine the partition 𝓟."""
    size_s = node_total(G, S)

    R = {
        v for v in S
          if nx.cut_size(G, [v], S - {v}, weight=Keys.WEIGHT) >= γ * node_total(G, v) * (size_s - node_total(G, v))
    }  # fmt: skip

    for v in R:
        # If v is in a singleton community, i.e. is a node that has not yet been merged
        if len(𝓟.node_community(v)) == 1:
            # Consider only well-connected communities
            𝓣 = freeze([
                C for C in 𝓟
                  if C <= S and nx.cut_size(G, C, S - C, weight=Keys.WEIGHT) >= γ * float(node_total(G, C) * (size_s - node_total(G, C)))
            ])  # fmt: skip

            # Now, choose a random community to put v into
            # We use python's random.choices for the weighted choice, as this is easiest.

            # Have a list of pairs of communities in 𝓣 together with the improvement (𝛥𝓗) of moving v to the community
            # Only consider communities for which the quality function doesn't degrade, if v is moved there
            communities = [(C, 𝛥𝓗) for (C, 𝛥𝓗) in ((C, 𝓗.delta(𝓟, v, C)) for C in 𝓣) if 𝛥𝓗 >= 0]
            # Calculate the weights for the random choice using the 𝛥𝓗 values
            weights = [exp(𝛥𝓗 / θ) for (C, 𝛥𝓗) in communities]

            # Finally, choose the new community
            # Use [0][0] to extract the community, since choices returns a list, containing a single (C, 𝛥𝓗) tuple
            Cₙ = choices(communities, weights=weights, k=1)[0][0]

            # And move v there
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



import pandas as pd
import os

def calculate_and_store_metrics(partitions, 𝓗, filename="metrics.csv"):
    """
    计算所有头的互信息和质量函数值，并将结果追加存储到CSV文件中。
    
    参数
    ----------
    partitions : List[Partition[T]]
        多个头的社区划分。
    𝓗 : QualityFunction[T]
        质量函数实例，用于计算社区划分的质量。
    filename : str
        存储结果的CSV文件名，默认是 'metrics.csv'。
    """
    num_heads = len(partitions)
    results = []

    # 计算每个头的质量函数
    for i in range(num_heads):
        quality = 𝓗(partitions[i])  # 直接调用𝓗来计算Modularity
        result_row = {'head': i, 'quality_function': quality, 'mutual_info': None}
        results.append(result_row)
    
    # 计算头之间的互信息
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            mi = mutual_info(partitions[i], partitions[j])
            result_row = {'head': f'{i}-{j}', 'quality_function': None, 'mutual_info': mi}
            results.append(result_row)

    # 将结果转换为DataFrame
    df = pd.DataFrame(results)

    # 检查文件是否已经存在，决定是写入新文件还是追加
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)  # 写入新文件
    else:
        df.to_csv(filename, mode='a', header=False, index=False)  # 追加写入
    print(f"Results appended to {filename}")




def multi_head_leiden_with_mutual_info_parallel(
    G: Graph, 𝓗: QualityFunction[T], num_heads: int, λ: float, θ: float = 0.3, γ: float = 0.05, weight: str | None = None
) -> List[Partition[T]]:
    """
    使用多个头和互信息正则化并行执行Leiden算法。
    
    参数
    ----------
    G : Graph
        要处理的图/网络。
    𝓗 : QualityFunction[T]
        要优化的质量函数。
    num_heads: int
        要生成的不同分区的头数。
    λ : float
        互信息在总损失中的权重。
    θ : float, optional
        Leiden方法的θ参数，默认值为0.3。
    γ : float, optional
        Leiden方法的γ参数，默认值为0.05。
    weight: str | None
        图中的边权重属性，默认为None。
    
    返回
    -------
    List[Partition[T]]
        生成的不同头的分区列表，应用了互信息正则化。
    """
    # 预处理图
    G = preprocess_graph(G, weight)
    
    # 初始化所有头的分区
    partitions = [Partition.singleton_partition(G, Keys.WEIGHT) for _ in range(num_heads)]
    
    # 记录之前的分区以检查收敛性
    previous_partitions = [None for _ in range(num_heads)]
    
    while True:
        # 并行执行所有分区的节点移动
        partitions = move_nodes_fast_parallel(G, partitions, 𝓗, λ)
        
        # 计算并存储当前头的互信息和质量函数到CSV
        calculate_and_store_metrics(partitions, 𝓗, "leiden_parallel_log.csv")  # 调用合并后的方法

        # 检查所有头是否收敛
        converged = True
        for i in range(num_heads):
            if len(partitions[i]) == G.order() or partitions[i] == previous_partitions[i]:
                continue
            else:
                converged = False
                break
        
        if converged:
            # 返回展开后的分区
            return [partition.flatten() for partition in partitions]
        
        # 记录当前分区
        previous_partitions = [deepcopy(partition) for partition in partitions]
        
        # 精炼所有分区并聚合图
        refined_partitions = []
        for partition in partitions:
            refined_partition = refine_partition(G, partition, 𝓗, θ, γ)
            refined_partitions.append(refined_partition)
        
        # 基于第一个精炼分区聚合图（假设所有分区具有相同的节点集）
        G = refined_partitions[0].aggregate_graph()
        
        # 将分区提升到新的聚合图
        new_partitions = []
        for partition in refined_partitions:
            partitions_dict = {id: set() for id in range(len(partition))}
            for v_agg, nodes in G.nodes(data=Keys.NODES):
                community_id = partition._node_part[next(iter(nodes))]
                partitions_dict[community_id].add(v_agg)
            partitions_l = list(partitions_dict.values())
            new_partition = Partition.from_partition(G, partitions_l, Keys.WEIGHT)
            new_partitions.append(new_partition)
        partitions = new_partitions


