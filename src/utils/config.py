# src/utils/config.py
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# OpenAI API 配置
API_KEY = os.getenv('API_KEY')
API_BASE = os.getenv('BASE_URL')
DEFAULT_MODEL = 'ep-20240824204846-vl99t'  # 默认模型名称

# Embedding 模型配置
EMBEDDING_MODEL = 'intfloat/multilingual-e5-large'
DEFAULT_EMBEDDING_COLUMN_NAME = 'embedding'

# 默认输出路径
OUTPUT_PATH = 'export'

# NEO4J 配置
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PWD = os.getenv('NEO4J_PWD')
RANDOM_PARAM = 0.1