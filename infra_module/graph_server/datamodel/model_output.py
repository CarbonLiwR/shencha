
from pydantic import BaseModel


class Entity(BaseModel):
    content: str

class EntityExtractedRes(BaseModel):
    entities: list[Entity]