from coderag.rerank.common import ResultBlock
from coderag.rerank.sorter.bubble import WindowedBubbleSorter
from typing import List
from openai import OpenAI, AsyncOpenAI
from typing import List, Tuple
import re
import numpy as np
from loguru import logger
from coderag.config import settings



reasoning_system = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
ordinary_system = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant provides the user with the answer enclosed within <answer> </answer> tags, i.e., <answer> answer here </answer>."
reasoning_user = """
Code Search Query:
{target_query}

Candidate Code Snippets:
{candidate_blocks}
Task:

As a professional software developer, please rank the N={num} candidate code segments based on their relevance to the provided code query. The query is a code snippet of which the next statement needs to be completed, and the segments are contextual pieces that might help in completing the code.
A segment is stated by a candidate index. 

Requirements:
1. Score each candidate segment based on its relevance to the query (1-10 scale, with 1 being the lowest and 10 being the highest);
2. When ranking, consider how well the context provided by each segment aids in completing the code, including keyword relevance, logical flow, and completeness of the information;
3. Make sure the result contains all the segment index and its ranking score and ensure that no index is repeated multiple times;
4. Must list all candidate snippets, with no additions or omissions.

Strict Output Format:

<think>  
Relevance reasoning ...
</think>  
<answer>[Rank1](Score1) > [Rank2](Score2) > ... > [RankN](ScoreN)</answer>

Example:
If there are 5 candidates and you assign them scores of 6, 1, 3, 5, and 9 respectively, your answer should look like this: <think> reasoning process here </think> <answer>[5](9) [1](6) [4](5) [3](3) [2](1)</answer>.
If there are 4 candidates and you assign them scores of 9, 4, 7, and 10 respectively, your answer should look like this: <think> reasoning process here </think> <answer>[4](10) [1](9) [3](7) [2](4)</answer>.
If there are 8 candidates and you assign them scores of 7, 8, 4, 1, 10, 5, 3 and 10 respectively, your answer should look like this: <think> reasoning process here </think> <answer>[5](10) [8](10) [2](8) [1](7) [6](5) [3](4) [7](3) [4](1)</answer>.
"""

ordinary_user = """
Code Search Query:
{target_query}

Candidate Code Snippets:
{candidate_blocks}
Task:

As a professional software developer, please rank the N={num} candidate code segments based on their relevance to the provided code query. The query is a code snippet of which the next statement needs to be completed, and the segments are contextual pieces that might help in completing the code.
A segment is stated by a candidate index. 

Requirements:
1. Score each candidate segment based on its relevance to the query (1-10 scale, with 1 being the lowest and 10 being the highest);
2. When ranking, consider how well the context provided by each segment aids in completing the code, including keyword relevance, logical flow, and completeness of the information;
3. Make sure the result contains all the segment index and its ranking score and ensure that no index is repeated multiple times;
4. Must list all candidate snippets, with no additions or omissions.

Strict Output Format:

<answer>[Rank1](Score1) [Rank2](Score2)  ...  [RankN](ScoreN)</answer>

Examples:
If there are 5 candidates and you assign them scores of 6, 1, 3, 5, and 9 respectively, your answer should look like this: <answer>[5](9) [1](6) [4](5) [3](3) [2](1)</answer>.
If there are 4 candidates and you assign them scores of 9, 4, 7, and 10 respectively, your answer should look like this: <answer>[4](10) [1](9) [3](7) [2](4)</answer>.
If there are 8 candidates and you assign them scores of 7, 8, 4, 1, 10, 5, 3 and 10 respectively, your answer should look like this: <answer>[5](10) [8](10) [2](8) [1](7) [6](5) [3](4) [7](3) [4](1)</answer>.
"""

def feed_into_prompt(
    prompt_template: str,
    query: str,
    candidates: list[str],
    ):
    num = len(candidates)
    tmp_string = ""
    for idx in range(len(candidates)):
        tmp_string += f"[{idx+1}]:\n ```\n{candidates[idx]}\n```\n"
    query_string = f"```\n{query}\n```\n"
    prompt = prompt_template.format(
        candidate_blocks = tmp_string, 
        target_query = query_string,
        num = num,
    )

    return prompt


class OpenAiListwiseComparator:
    def __init__(
        self, 
        model_name: str,
        client: AsyncOpenAI,
        retry_n: int,
        think: bool
    ):
        self.model_name = model_name
        self.client = client
        self.retry_n = retry_n
        self.think = think
        self.qwen3 = model_name.startswith("Qwen3")
        if think:
            self.system_prompt = reasoning_system
            self.user_prompt_template = reasoning_user
        else:
            self.system_prompt = ordinary_system
            self.user_prompt_template = ordinary_user
        if self.qwen3:
            self.system_prompt = ordinary_system
            self.user_prompt_template = ordinary_user

    async def chat(self, messages):
        response = await self.client.chat.completions.create(
            model=settings.rerank.rerank_model,
            messages=messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.think}} if self.qwen3 else None,
        )
        return response.choices[0].message.content


    def create_prompt(
        self,
        query: str,
        candidates: List[str]
        ):
        
        user_prompt = feed_into_prompt(
            prompt_template=self.user_prompt_template,
            query=query,
            candidates=candidates,
        )
        messages = [{
            'role': 'system', 
            'content': self.system_prompt
        }, {
            'role': 'user', 
            'content': user_prompt
        }]
        return messages
        

    def parse_llm_result(
        self,
        response: str,
        candidate_n
    ) -> List[Tuple[int, int]]: # list[(idx, score)]
        # Extract content within <answer> tags
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if not answer_match:
            logger.warning("no pattern matched in response")
            raise ValueError(f"no match where response={response}")
        
        content = answer_match.group(1).strip()
        items = content.split()
        
        parsed_items = []
        pattern = re.compile(r'\[(\d+)\]\((\d+)\)')
        
        for item in items:
            item = item.strip()
            if not item:
                continue
            
            match = pattern.match(item)
            if match:
                num1 = int(match.group(1))
                num2 = int(match.group(2))
                parsed_items.append((num1, num2))
        
        # Check if the number of parsed pairs matches the number of candidates
        if len(parsed_items) != candidate_n:
            logger.warning(f"input num is {candidate_n} but output num is {len(parsed_items)}")
            raise ValueError(f"answer pairs number is incorrect.")

        expected_nums = [i + 1 for i in range(candidate_n)]
        parsed_nums = [idx for idx, score in parsed_items]

        # Validate that all expected indices are present in the parsed result
        if len(expected_nums) != len(parsed_nums) or set(expected_nums) != set(parsed_nums):
            logger.warning(f"expected_nums is {expected_nums} but parsed_nums is {parsed_nums}")
            raise ValueError(f"answer format is incorrect")
            
        return parsed_items
        
    async def sort_k(
        self,
        query: str,
        candidates: list[str],
    ) -> list[tuple[int, str]]: # original_idx, candidate
        messages = self.create_prompt(query=query, candidates=candidates)
        idx_score_pair = None  # Initialize idx_score_pair to avoid unbound error
        for i in range(self.retry_n):
            try:
                res = await self.chat(messages) 
                assert res is not None
                idx_score_pair = self.parse_llm_result(res, len(candidates))
                idx_score_pair.sort(key=lambda pair: pair[1])
                break
            except ValueError as e:
                logger.warning(f"Rerank failed in {i+1}/{self.retry_n}")
        if idx_score_pair is None:
            logger.warning(f"Reranking failed after {self.retry_n} tries.")
            raise ValueError(f"Rerank error")
        return [(idx - 1, candidates[idx - 1]) for idx, _score in idx_score_pair]
