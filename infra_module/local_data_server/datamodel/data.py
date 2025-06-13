from pydantic import BaseModel

class Embedding(BaseModel):
    vector: str | list

class TextBlock(BaseModel):
    similarity: float | None = None
    embedding: Embedding | list | None = None
    content: str

class TextData(BaseModel):
    text_blocks: list[TextBlock]