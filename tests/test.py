# from src.extractors.chunk_extractor import split_text_into_chunks
from src.extractors.chunk_extractor import split_text_into_chunks


with open('data/novel.txt', 'r', encoding='utf-8') as file:
    source_document = file.read()
chunk_size = 600  # 可根据需要调整大小
chunks = split_text_into_chunks(source_document, chunk_size)

for idx, chunk in enumerate(chunks):
    print(f"Chunk {idx + 1}:\n{chunk}\n")