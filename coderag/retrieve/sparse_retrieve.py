from typing import Any, Callable, List, Optional, Sequence
import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from coderag.static_analysis import CodeElement
from coderag.retrieve.common import BaseElementIndexer, get_key_from_element, replace_punctuation_with_space


class TfidfElementIndexer(BaseElementIndexer):
    """
    TF-IDF-based indexer using scikit-learn's TfidfVectorizer.
    """
    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer = TfidfVectorizer()
        self.keys: List[str] = []
        self.vectors: Any = None
        self.meta: List[CodeElement] = []
        self.empty = True

    def fit(self, elements: Sequence[CodeElement]) -> None:
        if len(elements) == 0:
            self.empty = True
            return
        self.empty = False

        self.keys = [get_key_from_element(elem) for elem in elements]
        self.meta = list(elements)
        self.vectors = self.vectorizer.fit_transform(self.keys)
        logger.info(f"TF-IDF indexer built with {len(self.keys)} elements.")

    def search(self, query: str, k: int = 5) -> List[CodeElement]:
        if self.empty:
            return []
        query = replace_punctuation_with_space(query)
        query_vec = self.vectorizer.transform([query])
        scores: np.ndarray = cosine_similarity(query_vec, self.vectors).flatten()
        top_indices: np.ndarray = np.argsort(scores)[::-1][:k]
        return [self.meta[i] for i in top_indices]


class BM25ElementIndexer(BaseElementIndexer):
    """
    BM25-based indexer using rank_bm25.
    """
    def __init__(self, tokenizer: Optional[Callable[[str], List[str]]] = None) -> None:
        self.keys: List[str] = []
        self.meta: List[CodeElement] = []
        self.tokenized_keys: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenizer = tokenizer if tokenizer is not None else lambda text: text.split()
        self.empty = True

    def fit(self, elements: Sequence[CodeElement]) -> None:
        if len(elements) == 0:
            self.empty = True
            return
        self.empty = False
        self.keys = [get_key_from_element(elem) for elem in elements]
        self.meta = list(elements)
        self.tokenized_keys = [self.tokenizer(doc) for doc in self.keys]
        self.bm25 = BM25Okapi(self.tokenized_keys)
        logger.info(f"BM25 indexer built with {len(self.keys)} elements.")

    def search(self, query: str, k: int = 5) -> List[CodeElement]:
        if self.empty:
            return []
        query = replace_punctuation_with_space(query)
        tokenized_query = self.tokenizer(query)
        assert self.bm25 is not None
        scores: np.ndarray = self.bm25.get_scores(tokenized_query)
        top_indices: np.ndarray = np.argsort(scores)[::-1][:k]
        return [self.meta[i] for i in top_indices]
