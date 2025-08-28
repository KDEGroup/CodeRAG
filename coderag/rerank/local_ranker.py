from coderag.rerank.comparator.local_setwise_logits import SetWiseLikelihoodLocalScorer
from coderag.rerank.comparator.vllm_setvise_logits import SetWiseLikelihoodVllmScorer
from coderag.rerank.sorter.heap import IterativeHeapSorter
from coderag.rerank.sorter.bubble import WindowedBubbleSorter
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from coderag.rerank.common import ResultBlock
from loguru import logger
from typing import TypeVar

T = TypeVar("T")

def scores_sorter_submit(
    sorter: IterativeHeapSorter[T] | WindowedBubbleSorter[T], 
    score_element_pairs: list[tuple[float, T]]
):
    if isinstance(sorter, WindowedBubbleSorter):
        sorted_element = [element for _score, element in sorted(score_element_pairs, key=lambda x: x[0], reverse=True)]
        sorter.submit_sorted(sorted_element)
    else:
        scores = [score for score, _element in score_element_pairs]
        max_score_idx = 0
        max_score_val = scores[0]
        for idx, score in enumerate(scores):
            if score > max_score_val:
                max_score_val = score
                max_score_idx = idx
        sorter.submit_selection(max_score_idx)

def get_top_k_from_sorter(
    sorter: IterativeHeapSorter[T] | WindowedBubbleSorter[T], 
    k: int
) -> list[T]:
    if isinstance(sorter, WindowedBubbleSorter):
        return sorter.get_result()[:k]
    elif isinstance(sorter, IterativeHeapSorter):
        return list(reversed(sorter.get_result()[-k:]))
    else:
        raise NotImplementedError()


class LocalModelRanker:
    def __init__(
        self,
        model_name_or_path: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        retry_n = 3
    ):
        self.scorer = SetWiseLikelihoodLocalScorer(
            model_name_or_path=model_name_or_path,
            model=model,
            tokenizer=tokenizer,
            think=False # think=False to get a list of meaningful scores otherwise only the best is meaningful.
        )
        self.retry_n = retry_n

    def rank_single(
        self,
        query: str,
        sorter: WindowedBubbleSorter | IterativeHeapSorter,
        top_k: int
    ) -> list[ResultBlock]:
        for need_sorted in sorter:
            scores = None
            for i in range(self.retry_n):
                query_str = query
                candidates_str = [it.code_snippet for it in need_sorted]
                scores = self.scorer.get_scores_batch(
                    query_doc_pairs=[(query_str, candidates_str)],
                    batch_size=1,
                )[0]
                if isinstance(scores, list):
                    break
                else:
                    logger.warning(f"scorer failed in {i+1}/{self.retry_n} because: {scores}")
            if isinstance(scores, list):
                pairs = [(score, candidate) for candidate, score in zip(need_sorted, scores, strict=True)]
                scores_sorter_submit(sorter, pairs)
            else:
                raise ValueError(f"sort faild after {self.retry_n} nums")
        return get_top_k_from_sorter(sorter=sorter, k=top_k)
        
        
    def rank_bubble(
        self,
        query: str,
        candidates: list[ResultBlock],
        top_k: int,
        bubble_window: int,
        bubble_step: int,
    ) -> list[ResultBlock]:
        sorter = WindowedBubbleSorter(
            candidates,
            window_size=bubble_window,
            move_step=bubble_step,
            top_k=min(top_k, len(candidates))
        )
        return self.rank_single(
            query=query,
            sorter=sorter,
            top_k=top_k
        )
    
    def rank_heap(
        self,
        query: str,
        candidates: list[ResultBlock],
        top_k: int,
        heap_child_n: int
    ) -> list[ResultBlock]:
        sorter = IterativeHeapSorter(
            candidates,
            m_arity=heap_child_n,
            top_k=top_k
        )
        return self.rank_single(
            query=query,
            sorter=sorter,
            top_k=top_k
        )

    
    def rank_batch(
        self,
        query_candidate_pairs: list[tuple[str, list[ResultBlock]]],
        top_k: int,
        window_size: int = 6,
        step_size: int = 3,
    ) -> list[list[ResultBlock] | Exception]:
        sorters: list[WindowedBubbleSorter] = []
        unfinished_indices: set[int] = set()
        try_times: list[int] = [0] * len(query_candidate_pairs)
        result: list[list[ResultBlock] | Exception] = [[]] * len(query_candidate_pairs)
        for idx, (query, candidate_list) in enumerate(query_candidate_pairs):
            unfinished_indices.add(idx)
            sorters.append(WindowedBubbleSorter(
                candidate_list,
                window_size=window_size,
                move_step=step_size,
                top_k=min(top_k, len(candidate_list))
            ))
        
        while len(unfinished_indices) > 0:
            batch_indices: list[int] = []
            batch_need_sorted: list[list[ResultBlock]] = []
            batch_input: list[tuple[str, list[str]]] = []
            for idx in unfinished_indices.copy():
                sorter = sorters[idx]
                query, _candidates = query_candidate_pairs[idx]
                try:
                    need_sorted: list[ResultBlock] = next(sorter) # may failed

                    batch_indices.append(idx) # keep order
                    need_sorted_str = [it.code_snippet for it in need_sorted]
                    batch_input.append((query, need_sorted_str))
                    batch_need_sorted.append(need_sorted)
                except StopIteration:
                    unfinished_indices.remove(idx) # finished, so save result
                    result[idx] = sorter.get_result()

            # inference
            batch_scores = self.scorer.get_scores_batch(
                query_doc_pairs=batch_input,
                batch_size=128
            )
            for global_idx, scores, need_sorted in zip(batch_indices, batch_scores, batch_need_sorted, strict=True):
                if isinstance(scores, list): # correct
                    assert len(scores) == len(need_sorted)
                    # update sorter
                    score_blocks = list(zip(scores, need_sorted, strict=True))
                    score_blocks.sort(key=lambda x: x[0], reverse=True) # the higher the score is, the best
                    sorted_blocks = [block for _score, block in score_blocks]
                    sorter = sorters[global_idx]
                    sorter.submit_sorted(sorted_blocks)
                else:
                    e = scores
                    logger.warning(f"subsort failed due to {e}")
                    try_n = try_times[global_idx]
                    if try_n == 3:
                        result[global_idx] = ValueError(f"subsort failed after {self.retry_n} times")
                    else:
                        try_times[global_idx] += 1
        return result


                    

            
        
        
class LocalVllmModelRanker:
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        retry_n = 2,
        think: bool = False
    ):
        self.scorer = SetWiseLikelihoodVllmScorer(
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            think=think # think=False to get a list of meaningful scores otherwise only the best is meaningful.
        )
        self.retry_n = retry_n
        
    def rank_batch(
        self,
        queries: list[str],
        top_k: int,
        sorters: list[WindowedBubbleSorter] | list[IterativeHeapSorter]
    ) -> list[list[ResultBlock] | Exception]:
        total = len(queries)
        assert len(sorters) == total
        unfinished_indices: set[int] = set()
        try_times: list[int] = [0] * total
        current_subsort: list[list[ResultBlock]] = [[]] * total
        result: list[list[ResultBlock] | Exception] = [[]] * total
        for idx in range(total):
            unfinished_indices.add(idx)
        
        while len(unfinished_indices) > 0:
            batch_indices: list[int] = []
            batch_need_sorted: list[list[ResultBlock]] = []
            batch_input: list[tuple[str, list[str]]] = []
            for idx in unfinished_indices.copy():
                sorter = sorters[idx]
                query = queries[idx]
                try:
                    need_sorted: list[ResultBlock] = next(sorter) if try_times[idx] == 0 else current_subsort[idx] # may failed
                    current_subsort[idx] = need_sorted

                    batch_indices.append(idx) # keep order
                    need_sorted_str = [it.code_snippet for it in need_sorted]
                    batch_input.append((query, need_sorted_str))
                    batch_need_sorted.append(need_sorted)
                except StopIteration:
                    unfinished_indices.remove(idx) # finished, so save result
                    result[idx] = get_top_k_from_sorter(sorter, k=top_k)

            # inference
            batch_scores = self.scorer.get_scores_batch(
                query_doc_pairs=batch_input,
                batch_size=128
            )
            for global_idx, scores, need_sorted in zip(batch_indices, batch_scores, batch_need_sorted, strict=True):
                if isinstance(scores, list): # correct
                    try_times[global_idx] = 0
                    assert len(scores) == len(need_sorted)
                    # update sorter
                    score_blocks = list(zip(scores, need_sorted, strict=True))
                    sorter = sorters[global_idx]
                    scores_sorter_submit(sorter, score_blocks)
                else:
                    e = scores
                    logger.warning(f"subsort failed due to {e}")
                    try_n = try_times[global_idx]
                    if try_n == self.retry_n:
                        result[global_idx] = ValueError(f"subsort failed after {self.retry_n} times")
                        unfinished_indices.remove(global_idx)
                    else:
                        try_times[global_idx] += 1
        return result

    
    def rank_batch_bubble(
        self,
        query_candidate_pairs: list[tuple[str, list[ResultBlock]]],
        top_k: int,
        bubble_window: int = 6,
        bubble_step: int = 3,
    ) -> list[list[ResultBlock] | Exception]:
        sorters: list[WindowedBubbleSorter] = []
        queries: list[str] = []
        for query, candidate_list in query_candidate_pairs:
            queries.append(query)
            sorters.append(WindowedBubbleSorter(
                candidate_list,
                window_size=bubble_window,
                move_step=bubble_step,
                top_k=min(top_k, len(candidate_list))
            ))
        return self.rank_batch(
            queries=queries,
            top_k=top_k,
            sorters=sorters
        )

    def rank_batch_heap(
        self,
        query_candidate_pairs: list[tuple[str, list[ResultBlock]]],
        top_k: int,
        heap_child_n: int = 5,
    ) -> list[list[ResultBlock] | Exception]:
        sorters: list[IterativeHeapSorter] = []
        queries: list[str] = []
        for query, candidate_list in query_candidate_pairs:
            queries.append(query)
            sorters.append(IterativeHeapSorter(
                candidate_list,
                m_arity=heap_child_n,
                top_k=min(top_k, len(candidate_list))
            ))
        return self.rank_batch(
            queries=queries,
            top_k=top_k,
            sorters=sorters
        )