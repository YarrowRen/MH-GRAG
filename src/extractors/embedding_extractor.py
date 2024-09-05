import os
from src.utils.embedding_generator import EmbeddingGenerator
from src.utils.embedding_utils import save_df_with_embedding_as_csv, load_df_from_csv_with_embedding
from src.utils.utils import generate_random_string

def generate_and_save_embeddings(df_entities, df_relationships, output_path):
    """
    接收 df_entities 和 df_relationships，并生成 embedding，保存为 CSV 文件。
    
    参数:
    df_entities (pd.DataFrame): 包含实体的 DataFrame。
    df_relationships (pd.DataFrame): 包含关系的 DataFrame。
    output_path (str): 输出 CSV 文件的路径。
    
    返回:
    tuple: 加载后的实体和关系 DataFrame 对象。
    """
    # 初始化 EmbeddingGenerator 类
    entity_embedding_generator = EmbeddingGenerator(model_name='intfloat/multilingual-e5-large', column_name='entity_embedding')
    
    # 为实体生成嵌入
    print("Generating embeddings for entities...")
    df_entities_with_embeddings = entity_embedding_generator.generate_embeddings(df_entities, 'description')
    
    # 初始化 EmbeddingGenerator 类
    relationship_embedding_generator = EmbeddingGenerator(model_name='intfloat/multilingual-e5-large', column_name='relationship_embedding')
    
    # 为关系生成嵌入
    print("Generating embeddings for relationships...")
    df_relationships_with_embeddings = relationship_embedding_generator.generate_relationship_embeddings(
        df_relationships, 'relationship_description', 'source_entity', 'target_entity'
    )
    
    # 生成随机文件名
    entity_filename = f"entities_{generate_random_string(8)}.csv"
    relationship_filename = f"relationships_{generate_random_string(8)}.csv"
    
    # 拼接完整路径
    entity_file_path = os.path.join(output_path, entity_filename)
    relationship_file_path = os.path.join(output_path, relationship_filename)
    
    # 保存带有嵌入的 DataFrame 为 CSV 文件
    save_df_with_embedding_as_csv(df_entities_with_embeddings, ['entity_embedding'], entity_file_path)
    save_df_with_embedding_as_csv(df_relationships_with_embeddings, ['relationship_embedding'], relationship_file_path)
    
    # 输出文本表示成功保存
    print(f"Entities embedding saved to {entity_file_path}")
    print(f"Relationships embedding saved to {relationship_file_path}")
    
    # 读取并返回保存的 CSV 文件
    loaded_entities_df = load_df_from_csv_with_embedding(entity_file_path, ['entity_embedding'])
    loaded_relationships_df = load_df_from_csv_with_embedding(relationship_file_path, ['relationship_embedding'])
    
    return loaded_entities_df, loaded_relationships_df


def generate_community_report_embeddings(community_reports_df, output_path):
    """
    接收 community_reports_df，并生成 embedding，保存为 CSV 文件。
    
    参数:
    community_reports_df (pd.DataFrame): 包含社区报告的df。
    output_path (str): 输出 CSV 文件的路径。
    
    返回:
    DataFrame: 加载后的社区报告 DataFrame 对象。
    """
    # 初始化 EmbeddingGenerator 类
    community_reports_embedding_generator = EmbeddingGenerator(model_name='intfloat/multilingual-e5-large', column_name='community_report_embedding')
    
    # 为实体生成嵌入
    print("Generating embeddings for community reports...")
    community_reports_df_with_embeddings = community_reports_embedding_generator.generate_embeddings(community_reports_df, 'community_report')
    
    # 生成随机文件名
    community_report_filename = f"community_report_{generate_random_string(8)}.csv"
    
    # 拼接完整路径
    community_report_file_path = os.path.join(output_path, community_report_filename)
    
    # 保存带有嵌入的 DataFrame 为 CSV 文件
    save_df_with_embedding_as_csv(community_reports_df_with_embeddings, ['community_report_embedding'], community_report_file_path)
    
    # 输出文本表示成功保存
    print(f"community_report_with_embedding saved to {community_report_file_path}")
    
    # 读取并返回保存的 CSV 文件
    loaded_community_report_embedding = load_df_from_csv_with_embedding(community_report_file_path, ['community_report_embedding'])
    
    return loaded_community_report_embedding