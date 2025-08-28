from loguru import logger
from typing import List
from coderag.build_query.build_query import build_query_by_logits_local, build_query_by_last_k_lines
from coderag.config import settings
from coderag.benchmark.methods import load_benchmark
from coderag.config import settings
import json

def main():
    benchmark = load_benchmark()
    logger.info("Building query...")
    if settings.query.method == "logits":
        logger.info("Using logits method")
        querys: List[str] = build_query_by_logits_local(
            source_codes=[it.code_context for it in benchmark.data_list],
            token_num=5,
            top_k=1,
            chunk_size=3
        )
    elif settings.query.method == "last_k":
        logger.info(f"Using last_k method last_k={settings.query.lask_k}")
        querys: List[str] = build_query_by_last_k_lines(
            source_codes=[it.code_context for it in benchmark.data_list],
            k=settings.query.lask_k,
        )
    else:
        raise ValueError(f"Unknown method: {settings.query.method}")
    settings.query.output_file.parent.mkdir(exist_ok=True, parents=True)
    with open(settings.query.output_file, "w") as f:
        json.dump(querys, f, indent=4)
    logger.info(f"result saved in {settings.query.output_file}")

if __name__ == "__main__":
    main()