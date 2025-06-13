from collections import defaultdict
from ..datamodel.graph import Graph
class SubGraphExtractor():
    def __init__(self, graph_level):
        self.graph_level = graph_level
    def __get_relationship_instances(self, entities, graph):
        # get relationship
        # 第一优先级：网络内关系（即所选实体之间的关系）
        selected_entity_ids = [str(entity.id) for entity in entities]
        for relationship in graph.relationship_instances:
            relationship.attributes = {}
        in_network_relationships = [
            relationship
            for relationship in graph.relationship_instances
            if relationship.source_entity_uuid in selected_entity_ids
               and relationship.target_entity_uuid in selected_entity_ids
        ]
        # 第二优先级 - 网络外关系（即所选实体与不在所选实体中的其他实体之间的关系）
        source_relationships = [
            relationship
            for relationship in graph.relationship_instances
            if relationship.source_entity_uuid in selected_entity_ids
               and relationship.target_entity_uuid not in selected_entity_ids
        ]
        target_relationships = [
            relationship
            for relationship in graph.relationship_instances
            if relationship.target_entity_uuid in selected_entity_ids
               and relationship.source_entity_uuid not in selected_entity_ids
        ]
        out_network_relationships = source_relationships + target_relationships

        # 在网络外关系中，优先考虑相互关系（即与多个选定实体共享的网络外实体的关系）
        out_network_source_ids = [
            relationship.source_entity_uuid
            for relationship in out_network_relationships
            if relationship.source_entity_uuid not in selected_entity_ids
        ]
        out_network_target_ids = [
            relationship.target_entity_uuid
            for relationship in out_network_relationships
            if relationship.target_entity_uuid not in selected_entity_ids
        ]
        out_network_entity_ids = list(
            set(out_network_source_ids + out_network_target_ids)
        )
        out_network_entity_links = defaultdict(int)
        for entity_name in out_network_entity_ids:
            targets = [
                relationship.target_entity_uuid
                for relationship in out_network_relationships
                if relationship.source_entity_uuid == entity_name
            ]
            sources = [
                relationship.source_entity_uuid
                for relationship in out_network_relationships
                if relationship.target_entity_uuid == entity_name
            ]
            out_network_entity_links[entity_name] = len(set(targets + sources))
        # 按链接数量和rank_attributes对网络外关系进行排序
        for rel in out_network_relationships:
            rel.attributes["links"] = (
                out_network_entity_links[rel.source_entity_uuid]
                if rel.source_entity_uuid in out_network_entity_links
                else out_network_entity_links[rel.target_entity_uuid]
            )
        # 先按 attributes[links] 排序，然后按 ranking_attributes 排序

        # for out_network_relationship in out_network_relationships:
        #     del out_network_relationship.attr
        #       ibutes["links"]
        relationships = in_network_relationships + out_network_relationships
        return relationships

    def __get_community_reports(self, entities, graph):
        community_matches = {}
        for entity in entities:
            # 计算社区所包含选中实体的数量
            if entity.community_ids:
                for community_id in entity.community_ids:
                    community_matches[community_id] = (
                            community_matches.get(community_id, 0) + 1
                    )

        community_reports_id_dict = {
            community.id: community for community in graph.community_reports
        }

        # 防止部分社区没有报告
        select_communities = [
            community_reports_id_dict.get(community_id)
            for community_id in community_matches
            if community_id in community_reports_id_dict
        ]

        for community in select_communities:
            if community.attributes is None:
                community.attributes = {}
            community.attributes["matches"] = community_matches[community.id]
        for community in select_communities:
            community.level = int(community.level)
        if self.graph_level != None:
            temp_communities = []
            for community in select_communities:
                if community.level == self.graph_level:
                    temp_communities.append(community)
            return temp_communities
        return select_communities

    def __get_out_graph_entities(self,in_graph_entities, graph):

        mapped_entities_id = [entity.id for entity in in_graph_entities]
        out_network_entities = set()
        for mapped_entity in in_graph_entities:
            for relationship in graph.relationship_instances:
                if relationship.source_entity_uuid == mapped_entity.id and relationship.target_entity_uuid not in mapped_entities_id:
                    out_network_entities.add(relationship.target_entity_uuid)
                if relationship.target_entity_uuid == mapped_entity.id and relationship.source_entity_uuid not in mapped_entities_id:
                    out_network_entities.add(relationship.source_entity_uuid)
        return [entity for entity in graph.entities if entity.id in out_network_entities]


    def extract_by_entities(self, entities, graph):
        out_entities = self.__get_out_graph_entities(entities, graph)
        relationship_instances = self.__get_relationship_instances(entities, graph)
        community_reports = self.__get_community_reports(entities, graph)
        subgraph = Graph(entities=entities+out_entities, relationship_instances=relationship_instances, community_reports=community_reports)
        return subgraph
