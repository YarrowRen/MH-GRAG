# src/extractors/chunk_extractor.py

import re
from src.domain.Chunk import Chunk  # 导入 Chunk 类

def split_text_into_chunks(text, chunk_size):
    """
    将文本分割为指定大小的块，并确保最后不足一个 chunk_size 的文本与前一个块合并。
    
    参数:
    text (str): 需要分割的文本。
    chunk_size (int): 每个块的字符数。
    embeddings (Optional[list]): 一个可选的列表，包含与每个文本块对应的嵌入，默认为 None。
    
    返回:
    list: 由 Chunk 对象组成的列表。
    """
    # 通过正则表达式来处理连续的空白符
    text = re.sub(r'\s+', ' ', text)
    
    # 将文本按指定大小切分
    raw_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    # 如果最后一个块的大小小于 chunk_size，则将其与前一个块合并
    if len(raw_chunks) > 1 and len(raw_chunks[-1]) < chunk_size:
        raw_chunks[-2] += raw_chunks[-1]
        raw_chunks.pop()  # 删除最后一个已合并的块
    
    # 将文本块转换为 Chunk 对象，并为每个 Chunk 赋予相应的 embedding
    chunks = [Chunk(content=raw_chunks[i], chunk_id=i) for i in range(len(raw_chunks))]
    
    return chunks
