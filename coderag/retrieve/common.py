from typing import List, Sequence
from abc import ABC, abstractmethod
import string
from loguru import logger
from pydantic import BaseModel
from coderag.static_analysis import CodeElement, FunctionNode, VariableNode, render_node, parse_repository
from pathlib import Path


def replace_punctuation_with_space(origin: str) -> str:
    """
    Replace all punctuation in a string with spaces.
    """
    trans_table = str.maketrans({p: " " for p in string.punctuation})
    return origin.translate(trans_table)


def get_key_from_element(element: CodeElement) -> str:
    """
    Generate a key string from an Element object.
    """
    return render_node(element, use_path=False)


class BaseElementIndexer(ABC):
    """
    Abstract base class for element indexers.
    """
    @abstractmethod
    def fit(self, elements: Sequence[CodeElement]) -> None:
        """
        Build the index from a list of elements.
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[CodeElement]:
        """
        Search for the most similar elements to the query.
        """
        pass




class CodeRetriever:
    """
    Code snippet retriever:
      - Parses all .py files in a repository to extract function and variable nodes.
      - Builds separate indices for functions and variables based on the specified method.
      - Retrieves the most similar nodes for a given query.
    """
    def __init__(
        self, 
        repo_path: Path, 
        exclude_path: Path | None, 
        func_indexer: BaseElementIndexer,
        var_indexer: BaseElementIndexer
    ) -> None:
        logger.info(f"Initializing CodeRetriever for repository: {repo_path}")
        self.nodes: List[CodeElement] = parse_repository(repo_path=repo_path)
        if exclude_path is not None:
            self.nodes = [node for node in self.nodes if node.file_path.resolve() != exclude_path.resolve()]
        self.func_indexer: BaseElementIndexer = func_indexer
        self.var_indexer: BaseElementIndexer = var_indexer

        func_elements: List[FunctionNode] = [elem for elem in self.nodes if isinstance(elem, FunctionNode)]
        var_elements: List[VariableNode] = [elem for elem in self.nodes if isinstance(elem, VariableNode)]

        self.func_indexer.fit(func_elements)
        self.var_indexer.fit(var_elements)

    def retrieve(
        self, query: str, k_func: int = 5, k_var: int = 5
    ) -> List[CodeElement]:
        """
        Retrieve the most similar function and variable nodes for a query.
        """
        func_results = self.func_indexer.search(query, k=k_func)
        var_results = self.var_indexer.search(query, k=k_var)
        return func_results + var_results



class RetrieveResultItem(BaseModel):
    dense: list[str] | None
    sparse: list[str] | None
    dataflow: list[str] | None


class RetrieveResult(BaseModel):
    data_list: list[RetrieveResultItem]


def descend_single_subfolder(path: Path) -> Path:
    '''
    some benchmark has structure like: repo_name/repo_version/somethings in this repo
    so the real root path of such a repo is repo_version dir
    '''
    current = path.resolve()

    while True:
        if not current.exists():
            raise FileNotFoundError(f"Path does not exist: {current}")
        
        if not current.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {current}")

        entries = list(current.iterdir())

        if not entries:
            raise RuntimeError(f"Empty directory: {current}")

        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]

        if len(dirs) == 1 and not files:
            current = dirs[0]
        else:
            break

    return current
