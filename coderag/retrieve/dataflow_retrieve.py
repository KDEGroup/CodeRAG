from os import mkdir
from pathlib import Path
from typing import Callable
from coderag.static_analysis.data_flow.preprocess import projectParser, generate_context_graph
from coderag.static_analysis.data_flow.generator import Generator as DataflowGenerator
from loguru import logger

class DataflowRetriever:
    def __init__(
        self, 
        projs_dir: Path, # path to benchmark raw data,
        cache_dir: Path, # path to cache dir
        use_cache: bool, # whether to use cache
    ):
        assert projs_dir.exists(), f"Cannot find {projs_dir}"
        if use_cache:
            logger.info(f"Using cache dir {cache_dir}")
            assert cache_dir.exists(), f"Cannot find {cache_dir}"

            self.dataflow_generator = DataflowGenerator(
                proj_dir=projs_dir,
                info_dir=cache_dir,
            )
            return
        assert (not cache_dir.exists()) or not any(cache_dir.iterdir()), f"Cache dir {cache_dir} is not empty"
        logger.info(f"Generating context graph for {projs_dir}, saving to {cache_dir}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        pkgs = [p.name for p in projs_dir.iterdir() if p.is_dir()]
        generate_context_graph(
            pkg_list=pkgs,
            ds_repo_dir=projs_dir,
            ds_graph_dir=cache_dir,
        )

        self.dataflow_generator = DataflowGenerator(
            proj_dir=projs_dir,
            info_dir=cache_dir,
        )
    def retrieve(
        self, 
        project_name: str,  # project name of this benchmark item
        fpath: Path, # path to the file to be completed
        source_code: str, # context code
        calc_truncated: Callable[[list[str]], bool], # whether to truncate the prompt
    ) -> list[str]:
        return self.dataflow_generator.retrieve_prompt_list(
            project=project_name,
            fpath=str(fpath),
            source_code=source_code,
            calc_truncated=calc_truncated
        )
