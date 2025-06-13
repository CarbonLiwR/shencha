from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, List, Dict



@dataclass
class Entity():
    id: str
    name: str
    type: Optional[str] = None
    attributes_embedding: Optional[List[float]] = None
    community_ids: Optional[List[str]] = None
    rank: Optional[int] = 1
    attributes: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        name_key: str = "name",
        type_key: str = "type",
        community_key: str = "community",
        rank_key: str = "degree",
        attributes_key: str = "attributes",
    ) -> "Entity":
        """Create a new entity from the dict data."""
        return Entity(
            id=d[id_key],
            name=d[name_key],
            type=d.get(type_key),
            community_ids=d.get(community_key),
            rank=d.get(rank_key, 1),
            attributes=d.get(attributes_key),
        )

@dataclass
class Relationship():
    """A relationship between two entities. This is a generic relationship, and can be used to represent any type of relationship between any two entities."""
    id: str
    source_entity_uuid: str
    target_entity_uuid: str
    type: Optional[str] = None
    name: Optional[str] = None
    weight: Optional[float] = 1.0
    attributes: Optional[Dict[str, Any]] = None
    triple_source: Optional[str] = None
    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        id_key: str = "id",
        type_key: str = "type",
        source_key: str = "source",
        target_key: str = "target",
        weight_key: str = "weight",
        attributes_key: str = "attributes",
        triple_source_key: str = "triple_source",
    ) -> "Relationship":
        """Create a new relationship from the dict data."""
        return Relationship(
            id=d[id_key],
            type=d.get(type_key),
            source_entity_uuid=d[source_key],
            target_entity_uuid=d[target_key],
            weight=d.get(weight_key, 1.0),
            attributes=d.get(attributes_key),
            triple_source=d.get(triple_source_key)
        )

@dataclass
class Community():
    """A protocol for a community in the system."""
    id: str
    title: str = ""
    level: str = ""
    content: str = ""
    entity_ids: list[str] | None = None
    rating: float | None = None
    attributes: dict[str, Any] | None = None

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        full_content_key: str = "full_content",
        title_key: str = "title",
        level_key: str = "level",
        entities_key: str = "entity_ids",
        attributes_key: str = "attributes"
    ) -> "Community":
        """Create a new community from the dict data."""
        return Community(
            id=d[id_key],
            title=d[title_key],
            content=d[full_content_key],
            level=d[level_key],
            entity_ids=d.get(entities_key),
            attributes=d.get(attributes_key),
        )

class Graph():
    def __init__(self, entities, relationship_instances, community_reports):
        self.entities = entities
        self.relationship_instances = relationship_instances
        self.community_reports = community_reports