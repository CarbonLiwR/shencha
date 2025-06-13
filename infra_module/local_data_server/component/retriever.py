from typing import Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ..datamodel.data import TextData
from ..component.modelcall import LocalEmbeddingModel, OpenaiEmbeddingModel


class TextDataSemanticRetriever():
    def __init__(self, embedding_model: Union[LocalEmbeddingModel | OpenaiEmbeddingModel]):
        self.embedding_model: Union[LocalEmbeddingModel | OpenaiEmbeddingModel] =  embedding_model

    def retrieve_from_database(self, query: str, database: TextData) -> str:
        db_data = database.text_blocks
        query_embed = self.embedding_model.generate(query)
        query_vector = np.array(query_embed, dtype=np.float32).reshape(1, -1)
        for emb in db_data:
            stored_emb = np.array(emb.embedding, dtype=np.float32).reshape(1, -1)
            emb.similarity = cosine_similarity(query_vector, stored_emb)[0][0]
        sorted_embeddings = sorted(db_data, key=lambda x: x.similarity, reverse=True)
        result = ""
        for emb in sorted_embeddings[:3]:
            result += emb.content
        return result
