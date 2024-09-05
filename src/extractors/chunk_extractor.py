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

def split_text_into_chunks_smart(text, chunk_size):
    """
    使用智能切割，将中文文本按句子分割并确保块大小接近指定大小，避免句子被截断。
    
    参数:
    text (str): 需要分割的文本。
    chunk_size (int): 每个块的字符数。
    
    返回:
    list: 由 Chunk 对象组成的列表。
    """
    # 通过正则表达式处理连续的空白符
    text = re.sub(r'\s+', ' ', text)
    
    # 中文的句子标点符号：句号、问号、叹号、分号、顿号等
    sentence_endings = r'([。！？；])'
    
    # 使用正则表达式进行句子分割
    sentences = re.split(sentence_endings, text)
    
    # 合并标点符号与句子，避免标点符号被独立切割
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # 如果当前块的大小加上下一句的大小小于设定的 chunk_size，则将句子加到当前块
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence
        else:
            # 否则，将当前块存储，并开始新的块
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    # 添加最后一块
    if current_chunk:
        chunks.append(current_chunk.strip())

    # 如果最后一个块的大小小于 chunk_size，则将其与前一个块合并
    if len(chunks) > 1 and len(chunks[-1]) < chunk_size // 2:
        chunks[-2] += chunks[-1]
        chunks.pop()  # 删除最后一个已合并的块

    # 将文本块转换为 Chunk 对象
    chunk_objects = [Chunk(content=chunks[i], chunk_id=i) for i in range(len(chunks))]

    return chunk_objects