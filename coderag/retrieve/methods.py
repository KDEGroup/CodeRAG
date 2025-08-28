from typing import Callable, List, Tuple, Optional, Sequence
from loguru import logger
from coderag.retrieve.dense_retrieve import DenseElementIndexer
from coderag.retrieve.sparse_retrieve import TfidfElementIndexer, BM25ElementIndexer
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModel
from coderag.retrieve.common import CodeRetriever, CodeElement, descend_single_subfolder, render_node
from pathlib import Path

def nodes_to_snippets(nodes: Sequence[CodeElement], repo_path: Path) -> list[str]:
    retrieved_code_snippets: list[str] = []
    real_root_path = descend_single_subfolder(repo_path)
    for node in nodes:
        node.file_path = node.file_path.resolve().relative_to(real_root_path.resolve())
        retrieved_code_snippets.append(render_node(node))
    return retrieved_code_snippets
    

class DenseRetriever:
    def __init__(
        self,
        model_name: str
    ):
        logger.info(f"using model {model_name} for dense retrieving ...")
        device = "cuda"
        logger.info(f"using device {device}")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"loading done. ready for dense retrieval.")

    def retrieve_code_snippets_dense(
        self,
        repo_path: Path, 
        query: str, 
        k_func: int, 
        k_var: int, 
        exclude_path: Path | None, 
    ) -> list[str]:
        """
        Retrieve code snippets using dense retrieval.
        """
        retriever = CodeRetriever(
            repo_path, 
            exclude_path=exclude_path, 
            func_indexer=DenseElementIndexer(
                tokenizer=self.tokenizer,
                model=self.model
            ),
            var_indexer=DenseElementIndexer(
                tokenizer=self.tokenizer,
                model=self.model
            )
        )
        retrieved_nodes = retriever.retrieve(
            query=query,
            k_func=k_func,
            k_var=k_var
        )
        return nodes_to_snippets(retrieved_nodes, repo_path=repo_path)


def retrieve_code_snippets_sparse(
    repo_path: Path, 
    query: str, 
    k_func: int, 
    k_var: int, 
    method: str, 
    exclude_path: Path | None, 
    tokenizer: Optional[Callable[[str], List[str]]] = None
) -> list[str]:
    """
    Retrieve code snippets using sparse retrieval (TF-IDF or BM25).
    """
    if method not in ["tfidf", "bm25"]:
        raise ValueError("Unsupported sparse retrieval method. Please use 'tfidf' or 'bm25'.")

    logger.info(f"Using sparse retrieval with method: {method}")
    if method == "tfidf":
        func_indexer = TfidfElementIndexer()
        var_indexer = TfidfElementIndexer()
    elif method == "bm25":
        func_indexer = BM25ElementIndexer(tokenizer)
        var_indexer = BM25ElementIndexer(tokenizer)
    else:
        raise RuntimeError()

    retriever = CodeRetriever(
        repo_path, 
        exclude_path=exclude_path, 
        func_indexer=func_indexer,
        var_indexer=var_indexer
    )
    retrieved_nodes = retriever.retrieve(query, k_func=k_func, k_var=k_var)
    return nodes_to_snippets(retrieved_nodes, repo_path=repo_path)