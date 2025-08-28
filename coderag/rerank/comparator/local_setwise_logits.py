from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
import re
import torch
from .common import reasoning_system, ordinary_system, think_user_code_set_wise, user_code_set_wise, AnswerFormatError, AnswerTagError, ThinkTagError, CHARACTERS, build_prompt




class SetWiseLikelihoodLocalScorer:
    model: PreTrainedModel

    def __init__(
        self,
        model_name_or_path: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        think: bool = False
    ) -> None:
        """
        Initializes the scorer for set-wise likelihood estimation.

        Args:
            model_name_or_path: Name or path of the model, used for special logic (e.g., Qwen3).
            model: HuggingFace PreTrainedModel instance for generation.
            tokenizer: Corresponding tokenizer for the model.
            think: Whether to use reasoning-augmented prompts and post-processing.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.model_name_or_path = model_name_or_path
        self.system_prompt = reasoning_system if think else ordinary_system
        self.user_template = think_user_code_set_wise if think else user_code_set_wise
        self.think = think

        # Special handling for Qwen3 models — override prompt templates
        if "Qwen3" in model_name_or_path:
            self.system_prompt = ordinary_system
            self.user_template = user_code_set_wise

    def get_scores_batch(
        self,
        query_doc_pairs: list[tuple[str, list[str]]],
        batch_size: int
    ) -> list[list[float] | AnswerFormatError]:
        """
        Computes likelihood scores for each document in a set-wise manner.

        Args:
            query_doc_pairs: List of (query, documents) pairs to score.
            batch_size: Number of pairs to process per batch.

        Returns:
            A list of lists of float scores, or errors such as AnswerTagError / ThinkTagError.
        """
        from torch.nn.functional import softmax

        all_scores: list[list[float] | AnswerFormatError] = []

        for batch_start in range(0, len(query_doc_pairs), batch_size):
            batch = query_doc_pairs[batch_start: batch_start + batch_size]
            prompts: list[str] = []

            for query, docs in batch:
                assert len(docs) <= len(CHARACTERS)
                user_prompt = build_prompt(self.user_template, query, docs)
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                # Qwen3 requires special chat template behavior (e.g., enabling "thinking")
                if "Qwen3" in self.model_name_or_path:
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=self.think
                    )
                else:
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                assert isinstance(prompt, str)
                prompts.append(prompt)

            # Tokenize prompts and transfer to model's device
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            input_ids = inputs["input_ids"]
            input_length = input_ids.shape[1]

            # Generate responses with token-level scores
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=0
                )

            sequences = outputs.sequences[:, input_length:]
            scores = outputs.scores
            assert scores is not None

            for i, (seq, (query, docs)) in enumerate(zip(sequences, batch, strict=True)):
                # Extract token-level logits for this specific sample
                score_seq = [step[i] for step in scores]  # (gen_len, vocab_size)

                decoded = self.tokenizer.decode(seq, skip_special_tokens=True)

                # If reasoning is enabled, we expect exactly one <think>...</think> tag
                if self.think:
                    think_matches = re.findall(r"<think>(.*?)</think>", decoded, re.DOTALL)
                    if len(think_matches) != 1:
                        all_scores.append(ThinkTagError(num=len(think_matches)))
                        continue

                # We expect exactly one <answer>[...]</answer> span
                answer_matches = re.findall(r"<answer>\[(.*?)\]</answer>", decoded)
                if len(answer_matches) != 1:
                    all_scores.append(AnswerTagError(num=len(answer_matches), content=""))
                    continue

                match = answer_matches[0]
                answer_text = match
                prefix_text = decoded.split("<answer>[")[0] + "<answer>["
                prefix_token_len = len(self.tokenizer(prefix_text, add_special_tokens=False)["input_ids"])
                answer_token_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]

                if not answer_token_ids or prefix_token_len + len(answer_token_ids) > len(seq):
                    all_scores.append(AnswerTagError(num=1, content=match))
                    continue

                # Compute likelihood using the first token's logits in the answer span
                answer_start_pos = prefix_token_len
                first_token_logits = score_seq[answer_start_pos]
                softmax_logits = softmax(first_token_logits, dim=-1)

                option_scores = []
                for ch in CHARACTERS[:len(docs)]:
                    token_id = self.tokenizer(ch, add_special_tokens=False)["input_ids"][0]
                    option_scores.append(softmax_logits[token_id].item())

                all_scores.append(option_scores)

        return all_scores


        

