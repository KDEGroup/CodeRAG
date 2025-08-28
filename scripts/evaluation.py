from typing import List
from loguru import logger
from coderag.evaluation.metrics import compute_EM, compute_ES, compute_identifier_match
from coderag.config import settings
from coderag.benchmark import load_benchmark
from pydantic import BaseModel
from operator import itemgetter
import json

def main():
    inference_result: list[str] = []
    with open(settings.evaluation.use_inference_file, "r") as f:
        inference_result = json.load(f)

    benchmark = load_benchmark()

    if settings.evaluation.inference_data_indices_path is not None:
        with open(settings.evaluation.inference_data_indices_path, "r") as f:
            data_indices: list[int] = json.load(f)
        inference_result = list(itemgetter(*data_indices)(inference_result))

    sample_n = settings.sample_n
    benchmark.data_list = benchmark.data_list[:sample_n]
    inference_result = inference_result[:sample_n]


    class EvaluationItem(BaseModel):
        task_name: str
        idx: int
        ES: float
        EM: float
        identifier_EM: int
        identifier_F1: float

    class EvaluationResult(BaseModel):
        count: int
        ES: float
        EM: float
        identifier_EM: float
        identifier_F1: float
        details: list[EvaluationItem]


    total = len(benchmark.data_list)

    result = EvaluationResult(
        count=total,
        EM=0.0,
        ES=0.0,
        identifier_F1=0.0,
        identifier_EM=0.0,
        details=[]
    )



    for idx, (benchmark_it, generation) in enumerate(zip(benchmark.data_list, inference_result, strict=True)):
        gt = benchmark_it.ground_truth
        ES = compute_ES(target=gt, prediction=generation)
        EM = compute_EM(target=gt, prediction=generation)
        identifier_EM, identifier_F1 = compute_identifier_match(
            prediction=generation,
            target=gt
        )
        result.ES += ES
        result.EM += EM
        result.identifier_F1 += identifier_F1
        result.identifier_EM += identifier_EM
        result.details.append(EvaluationItem(
            task_name=benchmark_it.task_name,
            idx=idx,
            ES=ES,
            EM=EM,
            identifier_EM=identifier_EM,
            identifier_F1=identifier_F1
        ))
    result.EM /= total
    result.ES /= total
    result.identifier_F1 /= total
    result.identifier_EM /= total
    logger.info("Evaluation completed.")
        


    logger.info(f"Batch count: {result.count}")
    logger.info(f"Batch ES: {result.ES}")
    logger.info(f"Batch EM: {result.EM}")
    logger.info(f"Batch identifier EM: {result.identifier_EM}")
    logger.info(f"Batch identifier F1: {result.identifier_F1}")
    settings.evaluation.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.evaluation.output_file, "w") as f:
        json_result = result.model_dump_json(indent=4)
        f.write(json_result)
    logger.info("Evaluation completed and results saved.")

if __name__ == "__main__":
    main()