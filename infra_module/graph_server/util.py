import tiktoken

def num_tokens(text: str) -> int:
    token_encoder = tiktoken.get_encoding("cl100k_base")
    """返回给定文本中的标记数"""
    return len(token_encoder.encode(text=text))

def remove_unrelated_attributes(attributes: dict) -> dict:
        attributes = {
            key: value for key, value in attributes.items()
            if key != "index" and (isinstance(value, str) and value.lower() != "unknown")
        }
        return attributes


def get_entity_information_by_id(entities, given_id):
    for entity in entities:
        if str(entity.id) == given_id:
            entity_information = f"{entity.name},{remove_unrelated_attributes(entity.attributes)}"
            return entity_information