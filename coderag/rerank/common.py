from dataclasses import dataclass, field
from pydantic import BaseModel
from coderag.benchmark.common import BenchmarkItem


@dataclass
class ResultBlock:
    '''
    rerank: list[ResultBlock] -> list[ResultBlock]
    '''
    block_idx: int = -1
    code_snippet: str = ""
    score: float = 0.0


class RerankResultSaving(BaseModel):
    data_list: list[list[ResultBlock]]


def build_rerank_query(benchmark_item: BenchmarkItem) -> str:
    lines = benchmark_item.code_context.splitlines()
    query = '\n'.join(lines[-30:])
    query = f"# {"/".join(benchmark_item.deduped_path_list)}\n{query}"
    return query

def retrieve_block_filter(block: str) -> bool:
    return len(block.split()) < 2048