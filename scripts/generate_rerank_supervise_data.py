from coderag.rerank.supervise.common import SuperviseTrainingData, SuperviseTrainingDataItem
from coderag.benchmark.common import BenchmarkItem
from coderag.rerank.supervise.generate_data import generate_data_for_item
from coderag.benchmark import load_benchmark
from coderag.config import settings
from coderag.retrieve.common import RetrieveResult
from coderag.inference.limited_async_openai import get_client
from operator import itemgetter
from coderag.rerank.common import build_rerank_query, retrieve_block_filter
from loguru import logger
import asyncio

import json
# load data

def main():
    benchmark = load_benchmark()
    with open(settings.rerank.distill.use_retrieve_file, "r") as f:
        retrieve_result = RetrieveResult.model_validate_json(f.read())
    if settings.rerank.retrieve_data_indices_path is not None:
        with open(settings.rerank.retrieve_data_indices_path, "r") as f:
            data_indices: list[int] = json.load(f)
        retrieve_result.data_list = list(itemgetter(*data_indices)(retrieve_result.data_list))
    if settings.sample_n is not None:
        benchmark.data_list = benchmark.data_list[:settings.sample_n]
        retrieve_result.data_list = retrieve_result.data_list[:settings.sample_n]

    open_ai_client = get_client(
        api_key=settings.rerank.distill.teacher_model_api_key,
        base_url=settings.rerank.distill.teacher_model_api_url,
        max_concurrent_requests=10240,
        timeout=600
    )

    async def generate_all() -> SuperviseTrainingData:
        semaphore = asyncio.Semaphore(64) 
        result = SuperviseTrainingData(data_list=[])
        all_tasks = []
        count = 0
        total = len(benchmark.data_list)

        for benchmark_item, retrieve_item in zip(benchmark.data_list, retrieve_result.data_list, strict=True):
            async def generate_data_from_a_benchmark_item(benchmark_item: BenchmarkItem, all_docs: list[str]):
                async with semaphore:
                    query = build_rerank_query(benchmark_item)
                    item_result = await generate_data_for_item(
                        query, 
                        all_docs=all_docs,
                        client=open_ai_client,
                        model_name=settings.rerank.distill.teacher_model_name,
                        each_kind_num=settings.rerank.distill.each_kind_num,
                        select_nums=settings.rerank.distill.choice_num_list
                    )
                    nonlocal count
                    nonlocal total
                    count += 1
                    logger.info(f"{count} / {total} generate {len(item_result)} training data from task {benchmark_item.task_name}")
                    result.data_list.extend(item_result)
            all_code_blocks = (retrieve_item.dense or []) + (retrieve_item.sparse or []) + (retrieve_item.dataflow or [])
            all_code_blocks = [it for it in all_code_blocks if retrieve_block_filter(it)]
            all_tasks.append(generate_data_from_a_benchmark_item(benchmark_item, all_code_blocks))
        await asyncio.gather(*all_tasks)
        return result

    result = asyncio.run(generate_all())
    settings.rerank.distill.train_data_output_file.parent.mkdir(exist_ok=True)
    with open(settings.rerank.distill.train_data_output_file, "w") as f:
        f.write(result.model_dump_json(indent=4))

if __name__ == "__main__":
    main()