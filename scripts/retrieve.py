from pathlib import Path
from typing import List, Tuple
from loguru import logger
from coderag.config import settings
import json
from coderag.benchmark.methods import load_benchmark
from coderag.retrieve.dataflow_retrieve import DataflowRetriever
from coderag.retrieve.methods import DenseRetriever, retrieve_code_snippets_sparse
from coderag.retrieve.common import RetrieveResultItem, RetrieveResult
from coderag.retrieve.common import CodeElement
from coderag.build_prompt.merge_retrieval import get_tokenizer, merge_retrieval


def main():
    retrieve_settings = settings.retrieve

    logger.info(f"loading query file {retrieve_settings.use_query_file}")
    with open(retrieve_settings.use_query_file, "r") as f:
        querys: List[str] = json.load(f)

    logger.info("Retrieving...")

    benchmark = load_benchmark()
    sample_n = settings.sample_n
    if sample_n is not None:
        benchmark.data_list = benchmark.data_list[:sample_n]
        querys = querys[:sample_n]
    benchmark_count = len(benchmark.data_list)


    # dense retrieval
    dense_retrieval: list[list[str]] = []
    if retrieve_settings.dense.enable:
        dense_retriever = DenseRetriever(
            model_name=settings.retrieve.dense.emb_model,
        )
        logger.info("Using dense retrieval...")
        benchmark_count = len(benchmark.data_list)
        count = 1
        for query, benchmark_item in zip(querys, benchmark.data_list):
            repo_path = benchmark.get_repo(benchmark_item.repo_name).repo_path
            result = dense_retriever.retrieve_code_snippets_dense(
                repo_path=repo_path,
                query=query,
                k_func=5,
                k_var=5,
                exclude_path=benchmark_item.file_path,
            )
            dense_retrieval.append(result)
            logger.debug(
                f"Dense retrieving {count}/{benchmark_count} done, repo: {benchmark_item.repo_name}, query start with: {query[:30]}"
            )
            count += 1
    # sparse retrieval
    sparse_retrieval: list[list[str]]= []
    if retrieve_settings.sparse.enable:
        logger.info("Using sparse retrieval...")
        count = 0
        for query, benchmark_item in zip(querys, benchmark.data_list):
            repo_path = benchmark.get_repo(benchmark_item.repo_name).repo_path
            result = retrieve_code_snippets_sparse(
                repo_path=repo_path,
                query=query,
                k_func=5,
                k_var=5,
                method=settings.retrieve.sparse.method,
                exclude_path=benchmark_item.file_path,
            )
            sparse_retrieval.append(result)
            logger.debug(
                f"Sparse retrieving {count}/{benchmark_count} done, repo: {benchmark_item.repo_name}, query: {query}"
            )
            count += 1

    # dataflow retrieval
    dataflow_retrieval: List[List[str]]= []
    if retrieve_settings.dataflow.enable:
        tokenizer = get_tokenizer(settings.build_prompt.tokenizer_path_or_name)

        logger.info("Using dataflow retrieval...")
        dataflow_retriever = DataflowRetriever(
            projs_dir=settings.benchmark.repos_path,
            cache_dir=settings.retrieve.dataflow.graph_cache_dir,
            use_cache=settings.retrieve.dataflow.graph_use_cache,
        )
        count = 0
        for benchmark_item in benchmark.data_list:
            repo_path = benchmark.get_repo(benchmark_item.repo_name).repo_path
            result = dataflow_retriever.retrieve(
                project_name=benchmark_item.repo_name,
                fpath=benchmark_item.file_path,
                source_code=benchmark_item.code_context,
                calc_truncated=lambda x: merge_retrieval(
                    retrieval_infos=x,
                    source_code_prefix=f"# {"/".join(benchmark_item.deduped_path_list)}",
                    source_code=benchmark_item.code_context,
                    tokenizer=tokenizer,
                )[1],
            )
            dataflow_retrieval.append(result)
            logger.debug(
                f"Dataflow retrieving {count}/{benchmark_count} done, task name: {benchmark_item.task_name}, query starts with: {benchmark_item.code_context[:30]}"
            )
            count += 1
    # save results
    result = RetrieveResult(
        data_list=[]
    )
    for i in range(len(querys)):
        if settings.retrieve.sparse.enable:
            sparse_it = sparse_retrieval[i]
        else:
            sparse_it = None
        if settings.retrieve.dense.enable:
            dense_it = dense_retrieval[i]
        else:
            dense_it = None
        if settings.retrieve.dataflow.enable:
            dataflow_it = dataflow_retrieval[i]
        else:
            dataflow_it = None
        result.data_list.append(
            RetrieveResultItem(
                sparse=sparse_it,
                dense=dense_it,
                dataflow=dataflow_it
            )
        )
    retrieve_settings.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(retrieve_settings.output_file, "w") as f:
        f.write(result.model_dump_json(indent=4))

if __name__ == "__main__":
    main()