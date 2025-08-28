from .common import SuperviseTrainingDataItem
from coderag.rerank.comparator.common import build_message, CHARACTERS
from openai import AsyncClient
from typing import Sequence
from loguru import logger
import asyncio
import random
import re

# load data

_rng = random.Random(42)
_pattern = re.compile(r"<answer>\[(.*?)\]</answer>")
async def generate_data_for_item(
    query: str,
    all_docs: list[str],
    client: AsyncClient,
    model_name: str,
    each_kind_num: int = 3,
    select_nums: list[int] = [2, 3, 4, 5, 6, 7]
) -> list[SuperviseTrainingDataItem]:
    result: list[SuperviseTrainingDataItem] = []
    docs_group_list: list[tuple[str, ...]] = []
    all_tasks = []

    async def process_one(
        docs: Sequence[str],
    ):
        answer_count: dict[int, int] = {}
        messages = build_message(
            query=query,
            docs=docs,
            think=not "Qwen3" in model_name
        )
        for i in range(5):
            res = await client.chat.completions.create(
                messages=messages,
                model=model_name,
                temperature=0.8,
                max_tokens=10000,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False} 
                } if "Qwen3" in model_name else {}
            )
            response_text = res.choices[0].message.content
            assert response_text is not None
            matches = _pattern.findall(response_text)
            if len(matches) < 1:
                logger.warning(f"no answer tag in response")
                continue
            match_str: str = matches[-1]
            if len(match_str) != 1:
                logger.warning(f"the last match is {match_str} whose len is not 1")
                continue
            choice = match_str
            try:
                choice_idx = CHARACTERS.index(choice)
            except ValueError as e:
                logger.warning(f"the answer choice {choice} is out of characters")
                continue
            if choice_idx not in answer_count:
                answer_count[choice_idx] = 1
            else:
                answer_count[choice_idx] += 1
        
        found_key = None
        found_value = None
        for key, value in answer_count.items():
            if value >= 4:
                found_key = key
                found_value = value
                break
        if found_key is not None:
            logger.info(f"generate a data because of voting of {found_value}/5")
            result.append(SuperviseTrainingDataItem(
                query=query,
                candidates=list(docs),
                option=found_key,
                messages=list(messages),
                expected=f"<answer>[{CHARACTERS[found_key]}]</answer>"
            ))
        else:
            logger.info(f"not a good data because of choices: {answer_count}")

    for i in range(each_kind_num):
        for select_num in select_nums:
            if select_num > len(all_docs):
                continue
            docs = _rng.sample(all_docs, select_num)
            docs_group_list.append(tuple(docs))
    docs_group_list = list(set(docs_group_list))
    for docs in docs_group_list:
        all_tasks.append(process_one(docs))
    await asyncio.gather(*all_tasks)
    return result
    
            