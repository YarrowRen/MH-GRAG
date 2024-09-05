import pandas as pd
import json
import numpy as np

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
