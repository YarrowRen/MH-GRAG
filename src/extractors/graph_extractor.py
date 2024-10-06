import pandas as pd
import networkx as nx
import numpy as np
import igraph as ig
from leidenalg import find_partition
import leidenalg
import matplotlib.pyplot as plt
from tqdm import tqdm
from node2vec import Node2Vec
import os
from typing import Tuple, Dict

class GraphExtractor:
    def __init__(self, embedding_size: int = 1024):
        """
        初始化 GraphExtractor 类。
        
        参数:
        embedding_size (int): 嵌入向量的大小，默认值为1024。
        """
        self.embedding_size = embedding_size
        self.G = nx.Graph()

    def build_graph(self, df_entities: pd.DataFrame, df_relationships: pd.DataFrame) -> nx.Graph:
        """
        根据实体和关系构建图。

        参数:
        df_entities (pd.DataFrame): 包含实体及其嵌入的 DataFrame。
        df_relationships (pd.DataFrame): 包含关系的 DataFrame。
        
        返回:
        networkx.Graph: 构建的图对象。
        """
        # 添加节点
        for _, row in tqdm(df_entities.iterrows(), total=df_entities.shape[0], desc="Adding nodes"):
            self.G.add_node(row['entity_id'])

        # 添加边
        for _, row in tqdm(df_relationships.iterrows(), total=df_relationships.shape[0], desc="Adding edges"):
            if row['source_entity_id'] in self.G.nodes() and row['target_entity_id'] in self.G.nodes():
                self.G.add_edge(row['source_entity_id'], row['target_entity_id'], relationship_type=row['relationship_type'])
        
        return self.G

    def perform_community_detection(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        执行社区检测，并将社区 ID 添加到图的节点中。
        
        返回:
        tuple: 包含社区 ID 和子社区 ID 的映射。
        """
        # 将 networkx 图转换为 igraph
        node_map = {node: idx for idx, node in enumerate(self.G.nodes())}
        edges = [(node_map[u], node_map[v]) for u, v in self.G.edges()]
        G_ig = ig.Graph(edges, directed=False)

        # 使用 Leiden 算法进行社区检测
        partition = find_partition(G_ig, partition_type=leidenalg.ModularityVertexPartition)
        community_ids = partition.membership

        # 分配 community_id
        for node, idx in node_map.items():
            self.G.nodes[node]['community_id'] = community_ids[idx]

        # 进一步对每个社区内部的子社区进行划分
        self.perform_sub_community_detection(community_ids)

        return {node: self.G.nodes[node]['community_id'] for node in self.G.nodes()}, \
               {node: self.G.nodes[node]['sub_community_id'] for node in self.G.nodes()}

    def perform_sub_community_detection(self, community_ids: list):
        """
        对每个社区内部执行基于图结构的子社区划分。
        
        参数:
        community_ids (list): 社区 ID 列表。
        """
        for community_id in tqdm(set(community_ids), desc="Clustering sub-communities"):
            subgraph_nodes = [node for node in self.G.nodes() if self.G.nodes[node]['community_id'] == community_id]
            
            if len(subgraph_nodes) <= 1:
                for node in subgraph_nodes:
                    self.G.nodes[node]['sub_community_id'] = 0
                continue
            
            subgraph = self.G.subgraph(subgraph_nodes)
            if subgraph.number_of_edges() == 0:
                for node in subgraph.nodes():
                    self.G.nodes[node]['sub_community_id'] = 0
                continue
            
            subgraph_edges = [(u, v) for u, v in subgraph.edges()]
            subgraph_ig = ig.Graph.TupleList(subgraph_edges, directed=False)
            sub_partition = find_partition(subgraph_ig, partition_type=leidenalg.ModularityVertexPartition)
            
            subgraph_node_list = list(subgraph.nodes())
            sub_community_ids = sub_partition.membership
            for idx, node in enumerate(subgraph_node_list):
                self.G.nodes[node]['sub_community_id'] = sub_community_ids[idx]

    def visualize_communities(self, title: str = "Community Visualization", sub_community: bool = False):
        """
        可视化社区检测结果。
        
        参数:
        title (str): 图的标题。
        sub_community (bool): 是否可视化子社区。默认 False 表示社区检测。
        """
        attribute = 'sub_community_id' if sub_community else 'community_id'
        colors = {attr_value: plt.cm.tab20(i) for i, attr_value in enumerate(set(nx.get_node_attributes(self.G, attribute).values()))}
        node_colors = [colors[self.G.nodes[node][attribute]] for node in self.G.nodes()]

        plt.figure(figsize=(12, 12))
        nx.draw_networkx(self.G, pos=nx.spring_layout(self.G, seed=42), node_color=node_colors, with_labels=False, node_size=50)
        plt.title(title)
        plt.show()

    def generate_node2vec_embeddings(self, dimensions: int = 128) -> Dict[str, np.ndarray]:
        """
        生成 Node2Vec 嵌入，并将其添加到图的节点中。
        
        参数:
        dimensions (int): Node2Vec 嵌入的维度，默认为128。
        
        返回:
        dict: 节点的 Node2Vec 嵌入向量字典。
        """
        node2vec_model = Node2Vec(self.G, dimensions=dimensions, walk_length=30, num_walks=200, workers=4)
        model = node2vec_model.fit(window=10, min_count=1)
        entity_embeddings = {str(node): model.wv[str(node)] for node in self.G.nodes()}
        return entity_embeddings

    def add_embeddings_to_dataframe(self, df_entities: pd.DataFrame, community_mapping: Dict[str, int], sub_community_mapping: Dict[str, int], node2vec_embeddings: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        将社区 ID、子社区 ID 和 Node2Vec 嵌入添加到 DataFrame。
        
        参数:
        df_entities (pd.DataFrame): 实体的 DataFrame。
        community_mapping (dict): 社区 ID 映射。
        sub_community_mapping (dict): 子社区 ID 映射。
        node2vec_embeddings (dict): Node2Vec 嵌入。
        
        返回:
        pd.DataFrame: 更新后的 DataFrame。
        """
        df_entities['community_id'] = df_entities['entity_id'].map(community_mapping)
        df_entities['sub_community_id'] = df_entities['entity_id'].map(sub_community_mapping)
        df_entities['node2vec_embedding'] = df_entities['entity_id'].map(node2vec_embeddings)
        return df_entities

    def save_dataframe(self, df: pd.DataFrame, output_path: str, filename: str):
        """
        将 DataFrame 保存为 CSV 文件。
        
        参数:
        df (pd.DataFrame): 要保存的 DataFrame。
        output_path (str): 保存路径。
        filename (str): 文件名。
        """
        file_path = os.path.join(output_path, filename)
        df.to_csv(file_path, index=False)
        print(f"DataFrame saved to {file_path}")
