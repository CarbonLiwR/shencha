import json
import numpy as np
import pandas as pd
from ..datamodel.graph import Graph, Entity, Relationship, Community

class GraphLoader():
    def __init__(self):
        self.entity_map = {
            'uuid': 'id', 'name': 'name', 'type': 'type', 'attributes': 'attributes',
            'embeddings': 'attributes_embedding', 'sources': 'source_ids', 'communities': 'community_ids'
        }
        self.relationship_map = {
            'uuid': 'id', 'source_entity_uuid': 'source_entity_uuid', 'target_entity_uuid': 'target_entity_uuid',
            'type': 'type', 'name': 'name', 'attributes': 'attributes', "source": "triple_source"
        }
        self.community_report_map = {
            'uuid': 'id', 'title': 'title', 'level': 'level', 'content': 'content',
            'rating': 'rating', 'attributes': 'attributes'
        }
    def __read_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        return index

    def __parse(self, data, **kwargs):
        return [
            {new_key: item.get(old_key, kwargs["map"][old_key]) for old_key, new_key in kwargs["map"].items()}
            for item in json.loads(data)
        ]

    def __parse_by_map(self, content):
        entities = self.__parse(content['entities'], map=self.entity_map)
        relationships = self.__parse(content['relationships'], map=self.relationship_map)
        community_reports = self.__parse(content['communities'], map=self.community_report_map)
        return entities, relationships, community_reports

    def __load_entities(self, df: pd.DataFrame):
        dataclass_list = []
        try:
            for _, row in df.iterrows():
                dataclass_list.append(Entity(
                    id=row.get('id', ''),
                    type=row.get('type', ''),
                    name=row.get('name', ''),
                    community_ids=row.get('community_ids', ''),
                    attributes=json.loads(row['attributes'].replace("'", '"')) if isinstance(row['attributes'],
                                                                                             str) else row.get(
                        'attributes', ''),
                    attributes_embedding=json.loads(row['attributes_embedding']) if isinstance(
                        row['attributes_embedding'], str) else row.get('attributes_embedding', []),
                ))
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")

        return dataclass_list

    def __load_relationships(self, df: pd.DataFrame):
        dataclass_list = []
        for _, row in df.iterrows():
            dataclass_list.append(Relationship(
                id=row.get('id', ''),
                source_entity_uuid=row.get('source_entity_uuid', ''),
                target_entity_uuid=row.get('target_entity_uuid', ''),
                type=row.get('type', ''),
                name=row.get('name', ''),
                attributes=row.get('attributes', ''),
                triple_source=row.get('triple_source', '')
            ))
        return dataclass_list

    def __load_community_reports(self, df: pd.DataFrame):
        dataclass_list = []
        for _, row in df.iterrows():
            dataclass_list.append(Community(
                id=row.get('id', ''),
                title=row.get('title', ''),
                level=row.get('level', ''),
                entity_ids=row.get('entity_ids', []),
                rating=row.get('rating', ''),
                content=row.get('content', '')
            ))
        return dataclass_list

    def __init_graph_data(self, entities, relationships, community_reports):
        entities = self.__load_entities(df=pd.DataFrame(entities))
        relationships = self.__load_relationships(df=pd.DataFrame(relationships))
        community_reports = self.__load_community_reports(df=pd.DataFrame(community_reports))
        graph = Graph(entities, relationships, community_reports)
        return graph

    def load_graph(self, file_path):
        content = self.__read_file(file_path)
        entities, relationships, community_reports = self.__parse_by_map(content)
        graph = self.__init_graph_data(entities, relationships, community_reports)
        return graph