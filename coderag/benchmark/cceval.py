from pathlib import Path
from loguru import logger
from coderag.benchmark.common import load_jsonl, Benchmark, BenchmarkItem, CodeLocation, iterate_repos, CodeRepo
from typing import List

def loadCceval(benchmark_path: Path, repos_path: Path) -> Benchmark:
    json_objs: list = load_jsonl(benchmark_path)
    
    benchmark_data: List[BenchmarkItem] = []
    pkg_counter = {}
    for obj in json_objs:
        metadata = obj["metadata"]
        pkg = metadata["repository"]
        assert isinstance(pkg, str)
        fpath = metadata["file"]
        assert isinstance(fpath, str)
        input_content = obj["prompt"]
        assert isinstance(input_content, str)
        ground_truth = obj["groundtruth"]
        assert isinstance(ground_truth, str)

        if pkg not in pkg_counter:
            pkg_counter[pkg] = 0
        task_id = f"{pkg}/{pkg_counter[pkg]}"
        pkg_counter[pkg] += 1

        fpath_tuple = Path(fpath).parts
        context_start_lineno = int(metadata["context_start_lineno"])
        line_no = int(metadata["context_start_lineno"])
        
        benchmark_data.append(BenchmarkItem(
            code_context=input_content,
            code_context_location=CodeLocation(
                start_line_no = context_start_lineno,
                end_line_no = line_no - 1,
                path_list = list(fpath_tuple)
            ),
            file_path=repos_path.joinpath(pkg / Path(fpath)),
            deduped_path_list=list(fpath_tuple),
            ground_truth=ground_truth,
            task_name=task_id,
            repo_name=pkg
        ))
    logger.info(f"Benchmark length: {len(benchmark_data)}")
    
    repos: list[CodeRepo] = []
    for sub_dir in repos_path.iterdir():
        if sub_dir.is_dir() and sub_dir.name not in ["__pycache__", ".git", ".idea"]:
            repos.append(CodeRepo(
                repo_name=sub_dir.name,
                repo_path=sub_dir,
            ))
    logger.info(f"Repos length: {len(repos)}")
    
    return Benchmark(
        data_list=benchmark_data,
        repos=repos
    )