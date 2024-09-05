# 🌟 Multi-Head GraphRAG 

## 🛠️ 待优化：
- 📜 **chunk_extractor**: 调整chunk划分逻辑，尽量保持句子完整或段落完整
- 📝 **summarize_extractor**, **community report**: 优化LLM总结模版，和community report生成模版，使结构和语言统一
- 🔧 **Graph**与**RAG**查询部分封装，抽象参数结构
- 🧠 **Multi-Head**部分结构以及查询逻辑
- 📊 采用**dataframe**完成整个data flow还是将各种类别的实体都抽象为**POJO**
- 🔄 **Entity & Relationship**同类型合并

## 🙋 我们在做什么？

### 1. 📑 **Document -> Chunk**
通过 `chunk_extractor` 模块，将原始文本进行切分处理。根据预设的chunk大小，调整切分逻辑以尽量保持句子或段落的完整性，确保后续处理中的语义连贯性和准确性。

### 2. 🧩 **Chunk -> Entity & Relationship**
利用 `relationship_extractor` 模块结合LLM，从生成的文本分片中提取出对应的实体和关系信息。

### 3. 🧑‍💻 **Entity & Relationship -> Summary**
使用LLM对提取出的实体和关系进行总结，生成对应的简要描述。通过 `summarize_element_instances` 模块，分别为每个实体和关系创建摘要。

### 4. 🔍 **Summary -> Embedding**
通过 `embedding_extractor` 模块，利用预训练的嵌入模型（如 `multilingual-e5-large`）生成实体和关系的嵌入表示。该步骤将摘要数据转化为数值化的向量表示，并将生成的嵌入结果保存到指定的文件路径中，以便后续进行查询、相似性分析或模型训练。

### 5. 🌐 **Embedding -> Graph database (Community)**
在生成的嵌入基础上，使用 `graph_extractor` 模块构建图结构。通过构建图网络，执行社区检测，将实体和关系映射到不同的社区和子社区中。同时生成Node2Vec嵌入表示，用于进一步分析节点的特性和社区的结构。最终，社区信息和Node2Vec嵌入被合并到节点数据中。

### 6. 📝 **Community -> Community Report & Sub Community Report**
通过 `community_report` 模块生成每个社区和子社区的报告。使用LLM生成的摘要与嵌入相结合，输出每个社区的详细报告，并根据这些报告生成对应的嵌入表示。

### 7. 🚀 **Upload to Neo4j**

- 实体数据
- 关系数据
- 社区报告
- 子社区报告

将这些数据组织上传到Neo4j图数据库中，通过 `graph2neo4j` 模块插入实体、关系、社区和子社区数据，以实现图数据库中的完整数据表示和查询。

### 8. 🔍 **Graph & RAG Query Integration**
未完待续...