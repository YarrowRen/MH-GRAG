# src/domain/chunk.py

class Chunk:
    def __init__(self, content, chunk_id, embedding = None):
        """
        初始化 Chunk 对象。
        
        参数:
        content (str): 文本块的内容。
        chunk_id (int): 该块的 ID。
        embedding : 该块的embedding
        """
        self.content = content
        self.chunk_id = chunk_id
        self.embedding = embedding 
    def __repr__(self):
        return f"Chunk(id={self.chunk_id}, content={self.content[:20]}...)"
