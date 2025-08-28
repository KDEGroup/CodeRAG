import asyncio
from typing import Sequence, List, Callable
from coderag.inference.tgi import ask_llm_tgi
from coderag.config import settings
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.t5 import T5ForConditionalGeneration
from loguru import logger
import torch
import torch.nn.functional as F

def build_query_by_last_k_lines(
    source_codes: Sequence[str],
    k: int
) -> List[str]:
    """
    Build queries from the provided code by selecting the last k lines.

    Args:
        source_codes: A sequence of code strings.
        k: Number of lines to select from the end of each code string.

    Returns:
        A list of strings, each containing the last k lines of the corresponding source code.
    """
    return ['\n'.join(src_code.splitlines()[-k:]) for src_code in source_codes]

def process_one_file_sync(
    source_code: str,
    top_k: int,
    chunk_size: int,
    get_scores: Callable[[list[str]], list[float]]
):
    """
    Process a source code string by splitting it into chunks and scoring their relevance
    based on their importance with respect to the final chunk.

    Args:
        source_code: The complete source code to process.
        top_k: The number of most relevant chunks to retain.
        chunk_size: Number of lines per chunk.
        get_scores: A callable that takes a list of prompts and returns a list of scores.

    Returns:
        A string containing the concatenation of the top_k chunks and the final chunk.
    """
    lines = source_code.splitlines()
    chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
    chunks = ["\n".join(chunk_lines) for chunk_lines in chunks]

    if len(chunks) <= top_k + 1:
        return source_code  # Keep all chunks if the total number is small enough

    last_chunk = chunks[-1]
    prompts: list[str] = []
    
    # Create prompts by appending the last chunk to each of the previous chunks
    for i, chunk in enumerate(chunks[:-1]):
        prompt = chunk + "\n" + last_chunk
        prompts.append(prompt)

    scores = get_scores(prompts)
    scores = [(it, i) for i, it in enumerate(scores)]

    # Select the top_k chunks
    scores.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [chunks[i] for i in sorted(i for _, i in scores[:top_k])]    
    final_chunks = top_chunks + [chunks[-1]]  # Always keep the last chunk
    return "\n".join(final_chunks)





def build_query_by_logits_local(
    source_codes: Sequence[str],
    token_num: int,
    top_k: int,
    chunk_size: int
) -> list[str]:
    logger.info(f"loading model {settings.query.model_name_or_path} for building query")
    device = "cuda"
    logger.info(f"use device {device} to build query")
    tokenizer = AutoTokenizer.from_pretrained(settings.query.model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = T5ForConditionalGeneration.from_pretrained(settings.query.model_name_or_path).to(device)
    
    @torch.no_grad()
    def get_scores(prompts: List[str]) -> List[float]:
        inputs = tokenizer.batch_encode_plus(
            prompts,
            add_special_tokens=True,
            return_tensors="pt",
            padding="longest",
            truncation=True
        )
        inputs = inputs.to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=token_num,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=False,
            pad_token_id = tokenizer.eos_token_id
        )
        logits_probability_sum = torch.zeros(size=(len(prompts),), device=device) # (bs,)
        for _, logits in enumerate(outputs.scores): # logits (bs, vacab_size)
            softmax_logits = torch.softmax(logits, dim=1) # (bs, token_n)
            max_softmax_logits, _max_indices = torch.max(softmax_logits, dim=1) # (bs, )
            logits_probability_sum += max_softmax_logits
        return logits_probability_sum.tolist()
    
    def get_scores_batch(prompts: list[str], batch_size: int) -> list[float]:
        result: list[float] = []
        for i in range(0, len(prompts), batch_size):
            batch_result = get_scores(prompts[i:i+batch_size])
            result.extend(batch_result)
        return result
    total = len(source_codes)
    result: list[str] = []
    for idx, src in enumerate(source_codes):
        result_it = process_one_file_sync(
            source_code=src,
            top_k=top_k,
            chunk_size=chunk_size,
            get_scores=lambda x: get_scores_batch(x, batch_size=4)
        )
        result.append(result_it)
        logger.debug(f"finished building query for task {idx+1}/{total}")
    return result
            

