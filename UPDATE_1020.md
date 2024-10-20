# 最近工作

## 1. 修改训练后聚类方式

在最开始的设计中，经过多头GNN训练后使用得到的多组embedding，利用KMeans进行聚类，存在几个明显问题：
- 聚类后模块度下降明显，虽然相比初始的KMeans聚类有所提升，但明显低于初始状态下的Leiden
- 聚类时需要指定聚类簇数
- 互信息和模块度效果受参数影响十分严重

现在修改为使用训练后的Embedding生成K近邻图，并和初始图生成融合加权图，然后使用融合图进行Leiden聚类
- 各个头以及多头平均的模块度均明显优于原始Leiden结果
- 聚类簇数依赖于Leiden自动确定，不再需要超参数调节
- 互信息和模块度都优化明显

| id  | nodes | edges | initial_modularity | initial_clusters | initial_modularity_embedding | initial_clusters_embedding | avg_modularity_after_training | avg_clusters_after_training | avg_ami  | improvement_leiden (%) | improvement_embedding_leiden (%) | head_1_modularity | head_1_num_clusters | head_2_modularity | head_2_num_clusters | head_3_modularity | head_3_num_clusters | head_1_ami_with_head_2 | head_1_ami_with_head_3 | head_2_ami_with_head_3 |
| --- | ----- | ----- | ------------------ | ---------------- | --------------------------- | -------------------------- | ---------------------------- | --------------------------- | -------- | --------------------- | ------------------------------ | ----------------- | ------------------- | ----------------- | ------------------- | ----------------- | ------------------- | ---------------------- | ---------------------- | ---------------------- |
| 0   | 23570 | 52072 | 0.879755            | 7336             | 0.921488                     | 243                        | 0.936635                     | 282.000000                  | 0.067246 | 106.465507             | 101.643817                      | 0.899519           | 272                 | 0.951673           | 273                 | 0.958713           | 301                 | 0.055935               | 0.064235               | 0.081568               |
| 1   | 23570 | 52072 | 0.878740            | 7317             | 0.900986                     | 237                        | 0.940408                     | 276.333333                  | 0.073374 | 107.017731             | 104.375408                      | 0.930744           | 292                 | 0.947606           | 267                 | 0.942874           | 270                 | 0.076190               | 0.071583               | 0.072349               |
| 2   | 23570 | 52072 | 0.879584            | 7330             | 0.914862                     | 253                        | 0.948053                     | 343.666667                  | 0.085420 | 107.784256             | 103.627903                      | 0.925607           | 310                 | 0.969013           | 467                 | 0.949538           | 254                 | 0.089869               | 0.069830               | 0.096560               |


## 2. 完成MH-GraphRAG从检索至答案生成全流程

```c
2024-10-15 20:05:22,149 - INFO - PyTorch version 2.1.0 available.
2024-10-15 20:05:22,149 - INFO - TensorFlow version 2.17.0 available.
2024-10-15 20:05:22,810 - INFO - Loading data...
2024-10-15 20:05:33,749 - INFO - Data loaded successfully.
2024-10-15 20:05:33,749 - INFO - Query: Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch, and is accused by prosecutors of committing fraud for personal gain?
2024-10-15 20:05:33,749 - INFO - Finding the nearest entity...
2024-10-15 20:05:33,973 - INFO - Use pytorch device_name: mps
2024-10-15 20:05:33,974 - INFO - Load pretrained SentenceTransformer: intfloat/multilingual-e5-large
2024-10-15 20:05:56,809 - INFO - HTTP Request: POST https://api-2.xi-ai.cn/v1/chat/completions "HTTP/1.1 200 OK"
2024-10-15 20:06:17,681 - INFO - HTTP Request: POST https://api-2.xi-ai.cn/v1/chat/completions "HTTP/1.1 200 OK"
2024-10-15 20:06:33,381 - INFO - HTTP Request: POST https://api-2.xi-ai.cn/v1/chat/completions "HTTP/1.1 200 OK"
2024-10-15 20:06:42,125 - INFO - HTTP Request: POST https://api-2.xi-ai.cn/v1/chat/completions "HTTP/1.1 200 OK"
2024-10-15 20:07:07,336 - INFO - HTTP Request: POST https://api-2.xi-ai.cn/v1/chat/completions "HTTP/1.1 200 OK"
Merged report saved to /Users/boyuren/Documents/multi_head_graph_rag/MH-GRAG-V1/export/rag_test/merged_community_report_13127.json
2024-10-15 20:07:07,340 - INFO - Community reports generated and merged successfully.
2024-10-15 20:07:07,340 - INFO - Generating query template...
2024-10-15 20:07:07,340 - INFO - Query template generated.
2024-10-15 20:07:07,340 - INFO - Calling LLM API...
2024-10-15 20:07:10,765 - INFO - HTTP Request: POST https://api-2.xi-ai.cn/v1/chat/completions "HTTP/1.1 200 OK"
2024-10-15 20:07:10,768 - INFO - LLM API call completed.
2024-10-15 20:07:10,768 - INFO - Parsing LLM answer...
2024-10-15 20:07:10,768 - INFO - Answer: The individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges is Sam Bankman-Fried, who is accused by prosecutors of committing fraud for personal gain.
2024-10-15 20:07:10,768 - INFO - Used Findings: [1, 6, 8, 14]
```


---


```json
{
  "title": "Bankman-Fried and the Crypto Industry Trial",
  "summary": "This community centers around Bankman-Fried, a key figure in the cryptocurrency sector currently undergoing scrutiny in a fraud trial. The relationships between Bankman-Fried, the crypto industry, the legal entities involved, and the trial itself illustrate the interconnected dynamics at play in this significant legal and economic event.",
  "findings": [
    {
      "summary": "Bankman-Fried as the central figure",
      "explanation": "Bankman-Fried is the key entity in this community, serving as the central figure in the ongoing fraud trial. His actions and decisions have drawn significant attention and scrutiny, highlighting the broader implications for the cryptocurrency sector. The trial aims to shed light on his intent and actions connected to alleged fraudulent transactions, which could have lasting effects on the credibility of the crypto market. [records: Entities (id: 13127), Relationships (id: 9819, 9822)]",
      "id": 1
    },
    {
      "summary": "Impact of the fraud trial on the crypto industry",
      "explanation": "The fraud trial involving Bankman-Fried is of critical importance to the crypto industry, as it addresses issues that affect the sector's stability and credibility. The trial serves as a focal point for discussions around regulatory practices and ethical standards within the industry, which has been plagued by scandals and bankruptcies. The outcome could reshape public trust and investor confidence in cryptocurrencies. [records: Relationships (id: 9823)]",
      "id": 2
    },
    {
      "summary": "Role of Christopher Lavigne and Withers",
      "explanation": "Christopher Lavigne is a key player in this community, serving as a litigation partner at Withers, where he co-chairs the cryptocurrency practice. His legal expertise is pivotal in navigating the complexities of the fraud trial, providing valuable insights and commentary that could influence public perception and legal precedents related to cryptocurrency litigation. The relationship between Lavigne and Withers emphasizes the importance of specialized legal representation in high-stakes cases like this one. [records: Entities (id: 13128, 13129), Relationships (id: 9820, 9824)]",
      "id": 3
    },
    {
      "summary": "The Southern District of New York's role",
      "explanation": "The Southern District of New York is the federal court overseeing the fraud trial against Bankman-Fried. This court is known for handling high-profile cases, which adds a layer of significance to the proceedings. The court's decisions will not only impact Bankman-Fried but could also set important legal precedents for future cases involving cryptocurrency and fraud, influencing the regulatory landscape. [records: Entities (id: 13130), Relationships (id: 9821)]",
      "id": 4
    },
    {
      "summary": "Interconnections between fraud and the crypto sector",
      "explanation": "The fraud trial is closely linked to the broader issues affecting the crypto industry. Allegations against Bankman-Fried highlight significant concerns regarding transparency, governance, and ethical conduct within the sector. As the trial progresses, the discussions surrounding it may lead to reforms or regulatory changes aimed at preventing similar issues in the future, emphasizing the trial's potential for wider implications. [records: Relationships (id: 9822, 9823)]",
      "id": 5
    },
    {
      "summary": "Sam Bankman-Fried as a central figure",
      "explanation": "Sam Bankman-Fried is a key entity in this community, defined by his role as the founder of FTX and current defendant in the fraud trial. His actions are under scrutiny, particularly allegations of misusing customer funds and engaging in fraudulent activities connected to both FTX and Alameda. This scrutiny stems from the collapse of FTX, which has had lasting effects on the cryptocurrency industry, raising questions about regulatory oversight and ethical practices. Bankman-Fried's legal troubles have not only affected his personal reputation but have also impacted investors and customers connected to FTX. [records: Entities (13127), Relationships (483, 2966, 9819)]",
      "id": 6
    },
    {
      "summary": "Alameda's connection to FTX and financial practices",
      "explanation": "Alameda Research serves as a significant entity within this community, being closely associated with FTX and implicated in several financial controversies. The trading firm has been characterized by its risky financial maneuvers, including allegations of using customer deposits to cover losses and maintain liquidity. Alameda's association with FTX is crucial as it highlights how intertwined the two entities are, raising concerns about their operational practices and the potential for conflicts of interest. The relationship between Bankman-Fried and Alameda underscores the complexities in managing customer funds and operational integrity within the cryptocurrency sector. [records: Entities (639, 646, 13143), Relationships (483, 2966, 9812)]",
      "id": 7
    },
    {
      "summary": "The ongoing fraud trial's significance",
      "explanation": "The trial of Sam Bankman-Fried stands as a pivotal event in this community, addressing serious allegations of fraud, money laundering, and conspiracy. This legal proceeding is not only significant for determining the fate of Bankman-Fried but also has broader implications for the cryptocurrency industry, which has faced heightened scrutiny following the collapse of FTX. Observers are closely watching the trial for insights into potential regulatory changes and the future of digital currencies. The jury's role in this trial is critical as their verdict could set precedents for how similar cases are handled in the future, affecting investor confidence in the cryptocurrency markets. [records: Entities (573), Relationships (418, 1351, 6339)]",
      "id": 8
    },
    {
      "summary": "Josh Naftalis' insights into the trial",
      "explanation": "Josh Naftalis, a former federal prosecutor, has contributed valuable insights regarding the trial of Sam Bankman-Fried. His understanding of the legal implications surrounding the case highlights the complexities involved in prosecuting financial crimes, particularly in the rapidly evolving cryptocurrency landscape. Naftalis' perspectives can provide clarity on the potential outcomes of the trial and the broader implications for investors and the market as a whole. His role underscores the importance of legal expertise in navigating the challenges presented by financial fraud allegations in the digital asset space. [records: Entities (571), Relationships (418)]",
      "id": 9
    },
    {
      "summary": "Impact of the Bahamas on the case",
      "explanation": "The Bahamas plays a crucial role in the narrative surrounding Sam Bankman-Fried and FTX, as it is where the exchange was based and where Bankman-Fried was arrested. This jurisdiction is significant in understanding the legal framework and regulatory environment that FTX operated within. The interactions between Bankman-Fried and Bahamian officials could influence perceptions of compliance and ethical practices within the cryptocurrency industry. The geographical context adds another layer of complexity to the trial, as it raises questions about international regulatory standards and the challenges of prosecuting financial crimes across borders. [records: Entities (649, 6187), Relationships (486, 4737)]",
      "id": 10
    }
  ]
}
```


## 论文写作

