from pathlib import Path
from typing import Literal
from pydantic import BaseModel



def init_log():
    pass

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class Benchmark(BaseModel):
    name: Literal["recceval", "cceval"]
    repos_path: Path
    meta_data_path: Path
    data_indices_path: Path | None = None

class RetrieveSparse(BaseModel):
    method: str
    enable: bool
    var_topk: int
    func_topk: int

class RetrieveDense(BaseModel):
    emb_model: str
    enable: bool
    var_topk: int
    func_topk: int 

class RetrieveDataflow(BaseModel):
    enable: bool
    graph_use_cache: bool
    graph_cache_dir: Path


class Query(BaseModel):
    model_name_or_path: str
    model_max_token_n: int
    logits_token_n: int
    chunk_size: int
    output_file: Path
    method: Literal["logits", "last_k"]
    lask_k: int

class Retrieve(BaseModel):
    use_query_file: Path
    output_file: Path
    sparse: RetrieveSparse
    dense: RetrieveDense
    dataflow: RetrieveDataflow

class Distill(BaseModel):
    use_retrieve_file: Path
    retrieve_data_indices_path: Path
    train_data_output_file: Path
    choice_num_list: list[int]
    each_kind_num: int
    teacher_model_name: str
    teacher_model_api_url: str
    teacher_model_api_key: str


    training_base_model_path_or_name: str
    use_training_data_path: Path
    checkpoint_save_path: Path


class HumanEval(BaseModel):
    use_rerank_file: Path
    rerank_data_indices_path: Path
    output_file: Path

class Rerank(BaseModel):
    think: bool
    use_retrieve_file: Path
    output_file: Path
    enable: bool
    method: Literal["api", "local"]
    sort_method: Literal["heap", "bubble"]
    bubble_window: int
    bubble_step: int
    heap_child_n: int
    rerank_api_url: str
    rerank_model: str
    rerank_api_key: str
    top_k: int
    retrieve_data_indices_path: Path | None = None
    use_dense: bool
    use_sparse: bool
    use_dataflow: bool

    distill: Distill
    human_eval: HumanEval


class BuildPrompt(BaseModel):
    use_retrieval: bool
    use_rerank_file: Path
    use_retrieve_file: Path | None = None
    retrieve_data_indices_path: Path | None = None
    output_file: Path
    max_token_n: int
    tokenizer_path_or_name: str
    use_rerank_k: int | None


class Inference(BaseModel):
    use_prompt_file: Path
    output_file: Path
    api_url: str
    api_key: str
    model: str
    max_tokens: int

class Evaluation(BaseModel):
    inference_data_indices_path: Path | None = None
    use_inference_file: Path
    output_file: Path

class Settings(BaseSettings):
    benchmark: Benchmark
    query: Query
    retrieve: Retrieve
    rerank: Rerank
    build_prompt: BuildPrompt
    inference: Inference
    evaluation: Evaluation

    sample_n: int | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"

    model_config = SettingsConfigDict(toml_file='config/config.toml')

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (TomlConfigSettingsSource(settings_cls),)

settings = Settings()
init_log()