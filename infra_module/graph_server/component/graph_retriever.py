import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .modelcall import EmbeddingModel, ChatModel
from .graph_extractor import SubGraphExtractor
from .ranker import FieldRanker
from .context_builder import GraphContextBuilder

class GraphRetriever():
    def __init__(self,extract_model: ChatModel,
                 embedding_model: EmbeddingModel,
                 sub_graph_extractor: SubGraphExtractor,
                 ranker: FieldRanker,
                 context_builder: GraphContextBuilder):
        self.extract_model: ChatModel = extract_model
        self.embedding_model = embedding_model
        self.sub_graph_extractor: SubGraphExtractor = sub_graph_extractor
        self.ranker: FieldRanker = ranker
        self.context_builder: GraphContextBuilder = context_builder

    def __extract_entities_by_llm(self, user_input):
        entities = []

        extracted_res = self.extract_model.generate(user_input)
        for entity in extracted_res.entities:
            entities.append(entity.content)
        return entities

    def __retrieve_entity(self, queries, database):
        if not queries:
            return []

        entities_list = []

        for entity_name, extracted_embed in queries.items():
            similarities = []

            for entity in database:
                entity_embed = np.array(entity.attributes_embedding)
                if entity_embed.ndim == 1:
                    entity_embed = entity_embed.reshape(1, -1)

                similarity = cosine_similarity(extracted_embed, entity_embed)[0][0]
                similarities.append((entity, similarity))

            entities_list.extend(similarities)

        # ==================
        entities_list.sort(key=lambda x: x[1], reverse=True)
        top_k_entities = [entity for entity, similarity in entities_list[:10]]

        # 去重
        unique_top_k_entities = list({entity.id: entity for entity in top_k_entities}.values())

        return unique_top_k_entities

    def __extract_subgraph_from_graph(self, retrieved_entities, graph):
        subgraph = self.sub_graph_extractor.extract_by_entities(retrieved_entities, graph)
        return subgraph

    def __build_context(self, subgraph):
        context_str, context_df = self.context_builder.build_context(subgraph)
        return context_str, context_df

    def retrieve(self, search_query, graph):

        # graph = self.graph_importer.import_(graph)
        entities = self.__extract_entities_by_llm(search_query)

        entity_embedding = {}
        for entity in entities:
            embedding = np.array(self.embedding_model.generate(entity))
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)

            entity_embedding[entity] = embedding
        retrieved_entities = self.__retrieve_entity(entity_embedding, graph.entities)
        subgraph = self.__extract_subgraph_from_graph(retrieved_entities, graph)
        context_str, context_df = self.__build_context(subgraph)
        return context_str