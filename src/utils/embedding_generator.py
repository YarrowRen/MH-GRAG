from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
from src.utils.config import EMBEDDING_MODEL, DEFAULT_EMBEDDING_COLUMN_NAME

class EmbeddingGenerator:
    def __init__(self, model_name = EMBEDDING_MODEL, column_name = DEFAULT_EMBEDDING_COLUMN_NAME):
        """
        初始化 EmbeddingGenerator 类。
        
        参数:
        model_name (str): 要加载的模型名称，例如 'intfloat/multilingual-e5-large'。
        column_name (str): 要存储生成嵌入的 DataFrame 列名。
        """
        self.model_name = model_name
        self.column_name = column_name
        # 加载指定的模型
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, df, text_column):
        """
        生成 DataFrame 的嵌入，并将其添加到指定列中。
        
        参数:
        df (DataFrame): 要生成嵌入的 DataFrame。
        text_column (str): 包含要进行嵌入计算的文本的列名。
        
        返回:
        DataFrame: 包含嵌入列的更新后的 DataFrame。
        """
        # 添加进度条
        tqdm.pandas(desc=f"Generating embeddings for {text_column}")
        
        # 对指定列进行嵌入计算
        df[self.column_name] = df[text_column].progress_apply(
            lambda x: self.model.encode(x) if pd.notnull(x) else None
        )
        return df

    def generate_relationship_embeddings(self, df, description_column, source_column, target_column):
        """
        为关系生成嵌入，将 source 和 target 实体及其描述组合成文本来生成嵌入。
        
        参数:
        df (DataFrame): 要生成嵌入的 DataFrame。
        description_column (str): 包含关系描述的列名。
        source_column (str): 源实体的列名。
        target_column (str): 目标实体的列名。
        
        返回:
        DataFrame: 包含嵌入列的更新后的 DataFrame。
        """
        # 添加进度条
        tqdm.pandas(desc=f"Generating relationship embeddings")
        
        # 对指定列进行嵌入计算，将 description、source、target 拼接为输入
        df[self.column_name] = df.progress_apply(
            lambda x: self.model.encode(f"{x[description_column]} between {x[source_column]} and {x[target_column]}")
            if pd.notnull(x[description_column]) else None,
            axis=1
        )
        return df
