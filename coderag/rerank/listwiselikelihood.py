import re
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from .prompt import reasoning_system, ordinary_system, user_set_wise, reasoning_user_set_wise
import torch
import re
import torch

CHARACTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
                "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
                "W"]  # "Passage X" and "Passage Y" will be tokenized into 3 tokens, so we dont use for now

def build_prompt(template: str, query: str, docs: list[str]) -> str:
    assert len(docs) <= len(CHARACTERS)
    documents: list[str] = []
    for idx, doc in enumerate(docs):
        documents.append(f"Passage {CHARACTERS[idx]}:\n{doc}")
    return template.format(
        query=query,
        documents="\n".join(documents)
    )
        
class AnswerTagError(Exception):
    """
    Raised when the <answer> tags are missing or contain invalid content.
    """
    pass




class ListWiseLikelihoodLocalScorer:
    model: PreTrainedModel

    def __init__(
        self,
        model_name_or_path: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer
    ) -> None:
        """
        Initialize the scorer with a given language model and tokenizer.

        Args:
            model_name_or_path: Model identifier or path (not used in the body but kept for compatibility).
            model: A HuggingFace PreTrainedModel.
            tokenizer: A tokenizer corresponding to the model.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"  # Required for generation tasks with left-padded inputs

    def get_scores_batch(
        self,
        query_doc_pairs: list[tuple[str, list[str]]],
        batch_size: int,
        think: bool = False
    ) -> list[list[float]]:
        """
        Compute likelihood scores for a batch of (query, document list) pairs.

        Args:
            query_doc_pairs: A list of tuples, each containing a query and a list of documents.
            batch_size: Number of samples to process in one batch.
            think: Whether to use a more reasoning-focused prompt template.

        Returns:
            A list of lists, each inner list contains the score for each document option.
        """
        from torch.nn.functional import log_softmax, softmax

        all_scores: list[list[float]] = []

        for batch_start in range(0, len(query_doc_pairs), batch_size):
            batch = query_doc_pairs[batch_start: batch_start + batch_size]

            system_prompt = reasoning_system if think else ordinary_system
            user_template = reasoning_user_set_wise if think else user_set_wise
            prompts: list[str] = []

            # Construct prompts for each query-document set
            for query, docs in batch:
                assert len(docs) <= len(CHARACTERS)
                user_prompt = build_prompt(user_template, query, docs)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(prompt)

            # Tokenize prompts and move to model device
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            input_ids = inputs["input_ids"]
            input_length = input_ids.shape[1]

            # Generate responses and get token-level scores
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=512,
                    do_sample=False
                )

            sequences = outputs.sequences[:, input_length:]  # generated part only
            scores = outputs.scores  # token-wise scores

            for seq, score_seq, (query, docs) in zip(sequences, zip(*scores), query_doc_pairs):
                decoded = self.tokenizer.decode(seq, skip_special_tokens=True)

                match = re.search(r"<answer>\[(.*?)\]</answer>", decoded)
                if not match:
                    all_scores.append([float('-inf')] * len(CHARACTERS))
                    continue

                answer_text = match.group(1).strip()
                prefix_text = decoded.split("<answer>[")[0] + "<answer>["
                try:
                    prefix_token_len = len(self.tokenizer(prefix_text, add_special_tokens=False)["input_ids"])
                    answer_token_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
                except Exception:
                    all_scores.append([float('-inf')] * len(CHARACTERS))
                    continue

                if not answer_token_ids or prefix_token_len + len(answer_token_ids) > len(seq):
                    all_scores.append([float('-inf')] * len(CHARACTERS))
                    continue

                # Get softmax of first token logits at the answer start position
                answer_start_pos = prefix_token_len
                first_token_logits = score_seq[answer_start_pos]
                softmax_logits = softmax(first_token_logits, dim=-1)

                # Extract scores for each character choice
                option_scores = []
                for ch in CHARACTERS[:len(docs)]:
                    token_id = self.tokenizer(ch, add_special_tokens=False)["input_ids"][0]
                    option_scores.append(softmax_logits[token_id].item())

                all_scores.append(option_scores)

        return all_scores


        

from typing import TypeVar

T = TypeVar('T')
