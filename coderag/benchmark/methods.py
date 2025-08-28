from operator import itemgetter
from loguru import logger
from coderag.config import settings
from coderag.benchmark.common import Benchmark
from coderag.benchmark.recceval import load_ReccEval
from coderag.benchmark.cceval import loadCceval
import json


def load_benchmark() -> Benchmark:
    benchmark = None
    match settings.benchmark.name:
        case "recceval":
            benchmark = load_ReccEval(
                repos_path=settings.benchmark.repos_path,
                benchmark_path=settings.benchmark.meta_data_path,
            )
        case "cceval":
            benchmark = loadCceval(
                repos_path=settings.benchmark.repos_path,
                benchmark_path=settings.benchmark.meta_data_path,
            )
    assert benchmark is not None
    if settings.benchmark.data_indices_path is not None:
        with open(settings.benchmark.data_indices_path, "r") as f:
            indices: list[int] = json.load(f)
        benchmark.data_list = itemgetter(*indices)(benchmark.data_list)
        logger.info(f"use indices file {settings.benchmark.data_indices_path} length={len(indices)}")
    return benchmark