from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
import re
import math
from .common import reasoning_system, ordinary_system, think_user_code_set_wise, user_code_set_wise, AnswerFormatError, AnswerTagError, ThinkTagError, CHARACTERS, build_prompt
from loguru import logger
from vllm import LLM, SamplingParams

class SetWiseLikelihoodVllmScorer:
    model: PreTrainedModel
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        think: bool = False
    ) -> None:
        logger.info("vllm init")

        self.llm = LLM(
            model=model_name_or_path,
            tensor_parallel_size=4,
        )
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.model_name_or_path = model_name_or_path
        self.system_prompt = reasoning_system if think else ordinary_system
        self.user_template = think_user_code_set_wise if think else user_code_set_wise
        self.think = think
        if "Qwen3" in model_name_or_path:  # use hard switch for qwen3
            self.system_prompt = ordinary_system
            self.user_template = user_code_set_wise
    

    def get_scores_batch(
        self,
        query_doc_pairs: list[tuple[str, list[str]]],
        batch_size: int
    ) -> list[list[float] | AnswerFormatError]:

        all_scores: list[list[float] | AnswerFormatError] = []

        for batch_start in range(0, len(query_doc_pairs), batch_size):
            batch = query_doc_pairs[batch_start: batch_start + batch_size]
            prompts: list[str] = []

            for query, docs in batch:
                assert len(docs) <= len(CHARACTERS)
                assert len(docs) <= 20
                user_prompt = build_prompt(self.user_template, query, docs)
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                if "Qwen3" in self.model_name_or_path:
                    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.think)
                else:
                    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                assert isinstance(prompt, str)
                prompts.append(prompt)


            sampling_params = SamplingParams(
                logprobs=20,
                max_tokens=2048,
                top_p=1,
                temperature=0.1,
            )
            outputs = self.llm.generate(
                prompts,
                sampling_params
            )

            for output, (query, docs) in zip(outputs, batch):
                decoded = output.outputs[0].text
                if self.think:
                    think_matches = re.findall(r"<think>(.*?)</think>", decoded, re.DOTALL)
                    if len(think_matches) != 1:
                        logger.error(decoded)
                        all_scores.append(ThinkTagError(num=len(think_matches)))
                        continue
                answer_matches = re.findall(r"<answer>\[(.*?)\]</answer>", decoded)
                if len(answer_matches) != 1:
                    all_scores.append(AnswerTagError(num=len(answer_matches), content=""))
                    continue
                match = answer_matches[0]

                answer_text = match
                answer_token_id = self.tokenizer.encode(answer_text)
                if len(answer_token_id) != 1:
                    all_scores.append(AnswerTagError(num=len(answer_matches), content=answer_text))
                    continue

                prefix_text = decoded.split("<answer>[")[0] + "<answer>["
                prefix_token_len = len(self.tokenizer(prefix_text, add_special_tokens=False)["input_ids"])

                # Find the position of the first answer token in the scores
                answer_start_pos = prefix_token_len  # relative to the generated part
                logprobs = output.outputs[0].logprobs
                assert logprobs is not None
                if answer_start_pos >= len(logprobs):
                    all_scores.append(AnswerTagError(num=0, content="logprobs too short"))
                    continue

                first_token_logits = logprobs[answer_start_pos]

                option_scores = []
                for ch in CHARACTERS[:len(docs)]:
                    token_id = self.tokenizer(ch, add_special_tokens=False)["input_ids"][0]
                    if token_id in first_token_logits:
                        item = first_token_logits[token_id]
                        option_scores.append(math.exp(item.logprob))
                    else:
                        logger.warning(f"option {ch} is not in the top 20, set to 0")
                        option_scores.append(0)
                        

                all_scores.append(option_scores)

        return all_scores

