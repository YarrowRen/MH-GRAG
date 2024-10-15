import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.utils.embedding_generator import EmbeddingGenerator

def save_df_with_embedding_as_csv(df, embedding_columns, save_path):
    """
    将 DataFrame 中的多个 embedding 列转换为 JSON 字符串格式，并保存为 CSV 文件。
    
    参数:
    df (pd.DataFrame): 包含 embedding 的 DataFrame。
    embedding_columns (list): embedding 列的名称列表。
    save_path (str): 保存 CSV 文件的路径。
    
    返回:
    None
    """
    # 遍历所有 embedding 列，并确保每一列都被正确转换为 JSON 字符串
    for col in embedding_columns:
        df[col] = df[col].apply(
            lambda x: json.dumps(x.tolist()) if isinstance(x, np.ndarray) else None
        )
    
    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(save_path, index=False)
    print(f"File successfully saved in {save_path}.")


def load_df_from_csv_with_embedding(file_path, embedding_columns):
    """
    从 CSV 文件读取 DataFrame 并解析多个 embedding 列，将其从 JSON 格式恢复为嵌入向量。
    
    参数:
    file_path (str): CSV 文件路径。
    embedding_columns (list): embedding 列的名称列表。
    
    返回:
    pd.DataFrame: 解析后的 DataFrame。
    """
    # 从 CSV 文件读取 DataFrame
    df = pd.read_csv(file_path)
    
    # 遍历所有 embedding 列，并解析 JSON 字符串回 numpy 数组
    for col in embedding_columns:
        df[col] = df[col].apply(
            lambda x: np.array(json.loads(x)) if pd.notnull(x) else None
        )
    
    return df


# 计算查询嵌入和实体嵌入之间的余弦相似度
def find_k_nearest_entities(query, entities_with_embedding, embedding_column="entity_embedding", k=3):
    # 初始化 EmbeddingGenerator 类
    embedding_generator = EmbeddingGenerator()

    query_embedding= embedding_generator.get_query_embedding(query)
    # 过滤掉嵌入为空的情况
    valid_entities = entities_with_embedding.dropna(subset=[embedding_column])
    # 确保所有嵌入向量的维度一致
    entity_embeddings = np.array([emb for emb in valid_entities[embedding_column].values if len(emb) == len(query_embedding)])
    similarities = cosine_similarity([query_embedding], entity_embeddings)[0]
    valid_entities = valid_entities.loc[valid_entities[embedding_column].apply(lambda x: len(x) == len(query_embedding))]
    valid_entities['similarity'] = similarities
    nearest_entities = valid_entities.sort_values(by='similarity', ascending=False).head(k)
    return nearest_entities