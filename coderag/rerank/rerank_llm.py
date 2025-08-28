from coderag.rerank.common import ResultBlock
from coderag.rerank.sorter.bubble import WindowedBubbleSorter
from typing import List
from openai import OpenAI, AsyncOpenAI
from typing import List, Tuple
import re
import numpy as np
from loguru import logger
from coderag.config import settings
from coderag.rerank.comparator.openai_listwise import OpenAiListwiseComparator



class RerankerByPrompt:
    def __init__(
        self, 
        model_name: str,
        client: AsyncOpenAI,
        each_call_retry_n: int,
        think: bool
    ):
        self.comparator = OpenAiListwiseComparator(
            model_name=model_name,
            client=client,
            retry_n=each_call_retry_n,
            think=think
        )

    async def rerank(
        self,
        candidate_holder: List[ResultBlock],
        last_holder: ResultBlock,
        topK: int,
    ) -> List[ResultBlock]:
        sorter = WindowedBubbleSorter(
            items=candidate_holder,
            window_size=6,
            move_step=3,
            top_k=topK
        )
        for need_sorted in sorter:
            rank_result = await self.comparator.sort_k(
                query=last_holder.code_snippet,
                candidates=[it.code_snippet for it in need_sorted]
            )
            rank_block_result = [candidate_holder[idx] for idx, content in rank_result]
            sorter.submit_sorted(rank_block_result)
        return sorter.get_result()[:topK]
    
