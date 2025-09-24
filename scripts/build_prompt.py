from loguru import logger
from coderag.config import settings
from coderag.rerank.common import RerankResultSaving
from coderag.benchmark import load_benchmark
from coderag.build_prompt.merge_retrieval import get_tokenizer, merge_retrieval
from coderag.retrieve.common import RetrieveResult
from operator import itemgetter
import json

def main():
    sample_n = settings.sample_n
    benchmark = load_benchmark()
    if sample_n is not None:
        benchmark.data_list = benchmark.data_list[:sample_n]
    rerank_result = None
    if settings.build_prompt.use_retrieval:
        with open(settings.build_prompt.use_rerank_file, "r") as f:
            rerank_result = RerankResultSaving.model_validate_json(f.read())
            logger.info(f"use reranked result in {settings.build_prompt.use_rerank_file}")
        rerank_result.data_list = rerank_result.data_list[:sample_n]
    else:
        logger.info(f"without retrieval. zero shot.")

    retrieve_result = None
    if settings.build_prompt.use_retrieve_file is not None:
        with open(settings.build_prompt.use_retrieve_file, "r") as f:
            retrieve_result = RetrieveResult.model_validate_json(f.read())
        if settings.build_prompt.retrieve_data_indices_path is not None:
            with open(settings.build_prompt.retrieve_data_indices_path, "r") as f:
                retrieve_data_indices: list[int] = json.load(f)
            retrieve_result.data_list = itemgetter(*retrieve_data_indices)(retrieve_result.data_list)



    result_prompt: list[str] = []

    total = len(benchmark.data_list)
    tokenizer = get_tokenizer(settings.build_prompt.tokenizer_path_or_name)

    for i in range(total):
        benchmark_item = benchmark.data_list[i]
        retrieve_prompts: list[str] = []
        if rerank_result is not None: # use retrieval. other wise use dataflow > dense > sparse
            recall_blocks = rerank_result.data_list[i]
            if settings.build_prompt.use_rerank_k is not None:
                recall_blocks = recall_blocks[:settings.build_prompt.use_rerank_k]
            for item in recall_blocks:
                retrieve_prompts.append(f"{item.code_snippet.replace("'''", '"""')}") # comment
            if len(recall_blocks) == 0 and retrieve_result is not None:
                logger.info(f"task {benchmark_item.task_name} rerank list is [], so use retrieval result")
                retrieve_item = retrieve_result.data_list[i]
                dataflow_dense_sparse = (retrieve_item.dataflow or []) + (retrieve_item.dense or []) + (retrieve_item.sparse or [])
                for item in dataflow_dense_sparse:
                    retrieve_prompts.append(f"{item.replace("'''", '"""')}") # comment
                

        source_code_prefix = f"# {"/".join(benchmark_item.deduped_path_list)}"

        user_prompt, retrieval_truncated, source_code_truncated = merge_retrieval(
            retrieval_infos=retrieve_prompts,
            source_code=benchmark_item.code_context,
            source_code_prefix=source_code_prefix,
            tokenizer=tokenizer,
        )
        result_prompt.append(user_prompt)
        logger.debug(
            f"Building prompt {i + 1}/{total} done, task: {benchmark_item.task_name}, result starts with: {user_prompt[:30]}"
        )

    settings.build_prompt.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.build_prompt.output_file, "w") as f:
        json.dump(result_prompt, f, indent=4)

    
if __name__ == "__main__":
    main()