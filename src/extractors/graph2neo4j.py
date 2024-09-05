from neo4j import GraphDatabase
from src.utils.config import NEO4J_PWD, NEO4J_URI, NEO4J_USERNAME

class GraphToNeo4j:
    def __init__(self):
        """
        初始化 Neo4j 数据库连接
        参数:
        - uri: Neo4j 地址 (例如 "bolt://localhost:7687")
        - username: Neo4j 用户名
        - password: Neo4j 密码
        """
        self.uri = NEO4J_URI
        self.username=NEO4J_USERNAME
        self.password=NEO4J_PWD
        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def close(self):
        """
        关闭 Neo4j 数据库连接
        """
        self.driver.close()

    def clear_database(self):
        """
        清空数据库中的所有节点和关系
        """
        with self.driver.session() as session:
            session.write_transaction(self._clear_database)
    
    @staticmethod
    def _clear_database(tx):
        tx.run("MATCH (n) DETACH DELETE n")

    def insert_entities(self, entities_df):
        """
        插入实体到 Neo4j
        参数:
        - entities_df: 包含实体的 DataFrame
        """
        with self.driver.session() as session:
            for index, row in entities_df.iterrows():
                session.write_transaction(
                    self._create_entity, 
                    row['entity_id'], 
                    row['entity_name'], 
                    row['entity_type'], 
                    row['description'], 
                    row['embedding'], 
                    row['node2vec_embedding'], 
                    row['summary'], 
                    row['community_id'], 
                    row['sub_community_id']
                )

    @staticmethod
    def _create_entity(tx, entity_id, entity_name, entity_type, description, embedding, node2vec_embedding, summary, community_id, sub_community_id):
        query = (
            f"MERGE (e:{entity_type} {{entity_id: $entity_id}}) "  # 使用 entity_type 作为标签
            "SET e.entity_name = $entity_name, "
            "e.description = $description, "
            "e.embedding = $embedding, "
            "e.node2vec_embedding = $node2vec_embedding, "
            "e.summary = $summary, "
            "e.community_id = $community_id, "
            "e.sub_community_id = $sub_community_id "
        )
        tx.run(query, entity_id=entity_id, entity_name=entity_name, description=description, embedding=embedding, node2vec_embedding=node2vec_embedding, summary=summary, community_id=community_id, sub_community_id=sub_community_id)

    def insert_relationships(self, relationships_df):
        """
        插入关系到 Neo4j
        参数:
        - relationships_df: 包含关系的 DataFrame
        """
        with self.driver.session() as session:
            for index, row in relationships_df.iterrows():
                session.write_transaction(
                    self._create_relationship, 
                    row['relationship_id'], 
                    row['source_entity_id'], 
                    row['target_entity_id'], 
                    row['relationship_type'], 
                    row['relationship_description'], 
                    row['embedding'], 
                    row['summary']
                )

    @staticmethod
    def _create_relationship(tx, relationship_id, source_entity_id, target_entity_id, relationship_type, relationship_description, embedding, summary):
        relationship_type = relationship_type.replace("-", "_")
        query = (
            "MATCH (a {entity_id: $source_entity_id}), (b {entity_id: $target_entity_id}) "
            f"MERGE (a)-[r:{relationship_type} {{relationship_id: $relationship_id}}]->(b) "  # 使用 relationship_type 作为关系类型
            "SET r.relationship_description = $relationship_description, "
            "r.embedding = $embedding, "
            "r.summary = $summary "
        )
        tx.run(query, relationship_id=relationship_id, source_entity_id=source_entity_id, target_entity_id=target_entity_id, relationship_description=relationship_description, embedding=embedding, summary=summary)

    # 插入社区的函数
    def insert_communities(self, communities_df):
        """
        插入社区节点到 Neo4j
        参数:
        - communities_df: 包含社区的 DataFrame
        """
        with self.driver.session() as session:
            for index, row in communities_df.iterrows():
                session.write_transaction(
                    self._create_community, 
                    row['community_id'], 
                    row['community_report'], 
                    row['community_report_embedding']
                )

    @staticmethod
    def _create_community(tx, community_id, community_report, community_report_embedding):
        query = (
            "MERGE (c:Community {community_id: $community_id}) "
            "SET c.community_report = $community_report, "
            "c.community_report_embedding = $community_report_embedding "
        )
        tx.run(query, community_id=community_id, community_report=community_report, community_report_embedding=community_report_embedding)

    # 插入子社区的函数
    def insert_sub_communities(self, sub_communities_df):
        """
        插入子社区节点并与社区节点关联到 Neo4j
        参数:
        - sub_communities_df: 包含子社区的 DataFrame
        """
        with self.driver.session() as session:
            for index, row in sub_communities_df.iterrows():
                session.write_transaction(
                    self._create_sub_community, 
                    row['community_id'], 
                    row['sub_community_id'], 
                    row['community_report'], 
                    row['community_report_embedding']
                )

    @staticmethod
    def _create_sub_community(tx, community_id, sub_community_id, community_report, community_report_embedding):
        query = (
            "MERGE (sc:SubCommunity {community_id: $community_id, sub_community_id: $sub_community_id}) "
            "SET sc.community_report = $community_report, "
            "sc.community_report_embedding = $community_report_embedding "
            "WITH sc "
            "MATCH (c:Community {community_id: $community_id}) "
            "MERGE (sc)-[:AS_SUB_BELONG_TO]->(c) "
        )
        tx.run(query, community_id=community_id, sub_community_id=sub_community_id, community_report=community_report, community_report_embedding=community_report_embedding)

