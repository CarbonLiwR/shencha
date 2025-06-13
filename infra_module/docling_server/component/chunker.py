
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from transformers import AutoTokenizer

class Chunker():
    def __init__(self,model_name,max_tokens):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.chunker = HybridChunker(
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        max_tokens=max_tokens)

    def chunk(self, text):
        chunk_res = []
        for result in text:
            chunk_iter = self.chunker.chunk(dl_doc=result.document)

            for chunk in chunk_iter:
                chunk_res.append(chunk.text)

        return chunk_res