import asyncio
from typing import List

from coderag.inference.limited_async_openai import get_client
from coderag.rerank.common import ResultBlock
from loguru import logger
from coderag.config import settings
from coderag.rerank.rerank_llm import RerankerByPrompt
from coderag.retrieve.common import RetrieveResult
from coderag.benchmark.methods import load_benchmark
from coderag.config import settings
from coderag.rerank.common import RerankResultSaving, build_rerank_query, retrieve_block_filter
from coderag.rerank.local_ranker import LocalVllmModelRanker
from transformers import AutoTokenizer
import traceback
import json
from operator import itemgetter


def main():
    benchmark = load_benchmark()
    with open(settings.rerank.use_retrieve_file, "r") as f:
        retrieve_result = RetrieveResult.model_validate_json(f.read())
    
    if settings.rerank.retrieve_data_indices_path is not None:
        with open(settings.rerank.retrieve_data_indices_path) as f:
            indices = json.load(f)
        retrieve_result.data_list = itemgetter(*indices)(retrieve_result.data_list)

    sample_n = settings.sample_n
    if sample_n is not None:
        benchmark.data_list = benchmark.data_list[:sample_n]
        retrieve_result.data_list = retrieve_result.data_list[:sample_n]


    result_blocks: List[List[ResultBlock]] = []
    for item in retrieve_result.data_list:
        sparse_items, dense_items, dataflow_item = item.sparse, item.dense, item.dataflow
        if not settings.rerank.use_dense:
            dense_items = []
        if not settings.rerank.use_sparse:
            sparse_items = []
        if not settings.rerank.use_dataflow:
            dataflow_item = []
        all_prompts = (sparse_items or []) + (dense_items or []) + (dataflow_item or [])
        all_prompts = list(set(all_prompts))
        all_prompts = [it for it in all_prompts if retrieve_block_filter(it)]
        result_blocks.append([
            ResultBlock(
                block_idx=idx,
                code_snippet=it,
            ) for idx, it in enumerate(all_prompts)
        ])
        
    # Reranking process
    logger.info("Reranking...")

    if settings.rerank.method == "api" or not settings.rerank.enable:
        openai_client = get_client(
            api_key=settings.rerank.rerank_api_key,
            base_url=settings.rerank.rerank_api_url,
            max_concurrent_requests=1024,
            timeout=600.0,
            retries=2,
        )

        reranker = RerankerByPrompt(
            model_name=settings.rerank.rerank_model,
            client=openai_client,
            each_call_retry_n=5,
            think=settings.rerank.think
        )
        semaphore = asyncio.Semaphore(128)
        async def rerank_all(
        ) -> List[List[ResultBlock]]:
            """
            Rerank all recalled blocks with retry mechanism and simplified logging using Loguru.

            Returns:
                List of reranked ResultBlock lists; returns empty list at the corresponding position if an error occurs.
            """
            tasks = []
            count = 0
            total = len(result_blocks)

            async def rerank_task(recall_blocks, query, task_name):
                nonlocal count
                ret = []
                try:
                    if settings.rerank.enable:
                        async with semaphore:
                            ret = await reranker.rerank(
                                candidate_holder=recall_blocks,
                                last_holder=ResultBlock(code_snippet=query),
                                topK=min(settings.rerank.top_k, len(recall_blocks))
                            )
                    else:
                        ret = recall_blocks
                except Exception as e:
                    logger.error(
                        f"Rerank failed for task {task_name}, returning empty list."
                    )
                    traceback.print_exc()
                    return []

                # Success branch: update count and log info
                count += 1
                logger.info(
                    f"Reranking completed successfully for repo {task_name} ({count}/{total}). Query snippet: {query[:50]}"
                )
                return ret

            for idx, benchmark_item in enumerate(benchmark.data_list):
                recall_blocks = result_blocks[idx]
                query = build_rerank_query(benchmark_item)
                tasks.append(rerank_task(recall_blocks, query, benchmark_item.task_name))

            results = await asyncio.gather(*tasks)
            return results
        
        rerank_result: List[List[ResultBlock]] = asyncio.run(rerank_all())
        saving_result = RerankResultSaving(data_list=rerank_result)
        settings.rerank.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(settings.rerank.output_file, "w") as f:
            f.write(saving_result.model_dump_json(indent=4))
    else: # local reranking
        model_id = settings.rerank.rerank_model
        batch_size = 128
        total = len(benchmark.data_list)
        count = 0
        total_batch = total // batch_size
        if total % batch_size != 0:
            total_batch += 1

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        ranker = LocalVllmModelRanker(
            model_name_or_path=model_id,
            tokenizer=tokenizer,
            retry_n=5,
            think=settings.rerank.think
        )
        rerank_result: List[List[ResultBlock]] = []
        for i in range(0, total, batch_size):
            count += 1
            logger.info(f"start batch {count}/{total_batch}")
            end_idx = i + batch_size
            batch_benchmark_item = benchmark.data_list[i: end_idx]
            batch_recall_blocks = result_blocks[i: end_idx]
            query_block_pairs: list[tuple[str, list[ResultBlock]]] = []
            for benchmark_item, recall_blocks in zip(batch_benchmark_item, batch_recall_blocks):
                query = build_rerank_query(benchmark_item)
                query_block_pairs.append((query, recall_blocks))
            if settings.rerank.sort_method == "bubble":
                batch_rank_result = ranker.rank_batch_bubble(
                    query_candidate_pairs=query_block_pairs,
                    top_k=settings.rerank.top_k,
                    bubble_window=settings.rerank.bubble_window,
                    bubble_step=settings.rerank.bubble_step
                )
            else:
                batch_rank_result = ranker.rank_batch_heap(
                    query_candidate_pairs=query_block_pairs,
                    top_k=settings.rerank.top_k,
                    heap_child_n=settings.rerank.heap_child_n
                )
                
            assert len(batch_benchmark_item) == len(batch_rank_result)
            for idx, (rank_result, benchmark_item) in enumerate(zip(batch_rank_result, batch_benchmark_item, strict=True)):
                if isinstance(rank_result, list):
                    rerank_result.append(rank_result)
                else:
                    logger.warning(f"task: {benchmark_item.task_name} failed due to {rank_result}. result will be []")
                    rerank_result.append([])
                
                
        saving_result = RerankResultSaving(data_list=rerank_result)
        settings.rerank.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(settings.rerank.output_file, "w") as f:
            f.write(saving_result.model_dump_json(indent=4))

            

            


if __name__ == "__main__":
    main()