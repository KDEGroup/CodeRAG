from pathlib import Path
from typing import List, Optional, Sequence

from loguru import logger
from coderag.retrieve.common import BaseElementIndexer, get_key_from_element
import torch
import torch.nn.functional as F
from coderag.static_analysis import VariableNode, FunctionNode, render_node


class DenseElementIndexer(BaseElementIndexer):
    """
    Dense vector-based indexer using the transformers library.
    """
    def __init__(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = model.device
        self.keys: List[str] = []
        self.embeddings: torch.Tensor = torch.empty(0)
        self.meta: List[VariableNode | FunctionNode] = []
        self.empty: bool = True

    def _encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of texts into dense embeddings using the model.
        """
        batch_embedding_results = []
        batch_size = 256
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                # (batch_size, emb_dim)
                outputs = self.model(**inputs)
                batch_embedding_results.append(outputs)
        # (len(texts), emb_dim)
        return torch.cat(batch_embedding_results, 0)

    def fit(self, elements: Sequence[FunctionNode | VariableNode]) -> None:
        if len(elements) == 0:
            self.empty = True
            return
        self.empty = False
        """
        Build the dense vector index from a list of elements.
        """
        self.keys = [get_key_from_element(elem) for elem in elements]
        self.meta = list(elements)
        if len(self.keys) > 0:
            self.embeddings = self._encode(self.keys)
        logger.info(f"Dense indexer built with {len(self.keys)} elements.")

    def search(self, query: str, k: int = 5) -> List[FunctionNode | VariableNode]:
        """
        Search for the top-k most similar elements to the query.
        """
        if self.empty:
            return []
        # (1, emb_dim)
        query_embedding = self._encode([query])
        similarities = F.cosine_similarity(query_embedding, self.embeddings)
        k = min(k, similarities.size(0))
        topk_scores, topk_indices = torch.topk(similarities, k)
        return [self.meta[i] for i in topk_indices]

