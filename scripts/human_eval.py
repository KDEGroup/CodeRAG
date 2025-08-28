from loguru import logger
from coderag.config import settings
from coderag.rerank.common import RerankResultSaving
from coderag.benchmark import load_benchmark
from coderag.retrieve.common import RetrieveResult
from operator import itemgetter
from pathlib import Path
import json

sample_n = settings.sample_n
benchmark = load_benchmark()
with open(settings.rerank.human_eval.use_rerank_file, "r") as f:
    rerank_result = RerankResultSaving.model_validate_json(f.read())
    logger.info(f"use reranked result in {settings.build_prompt.use_rerank_file}")
if sample_n is not None:
    benchmark.data_list = benchmark.data_list[:sample_n]
    rerank_result.data_list = rerank_result.data_list[:sample_n]

result = []
for benchmark_item, rerank_result_item in zip(benchmark.data_list, rerank_result.data_list):
    query = benchmark_item.code_context
    docs = [it.code_snippet for it in rerank_result_item]
    result.append({
        "query": query,
        "docs": docs,
    })
save_path = Path(settings.rerank.human_eval.output_file)
save_path.parent.mkdir(parents=True, exist_ok=True)
with open(save_path, "w") as f:
    json.dump(result, f, indent=4)