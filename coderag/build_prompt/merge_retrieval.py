from transformers.models.auto.tokenization_auto import AutoTokenizer
import tiktoken
from coderag.config import settings


def get_tokenizer(model_name: str):
    """
    Return a tokenizer compatible with the given model:
    - Use tiktoken for OpenAI models
    - Use AutoTokenizer from transformers for other models
    """
    # For OpenAI API models
    if "gpt-3.5" in model_name or "gpt-4" in model_name:
        return tiktoken.encoding_for_model(model_name)
    # For vLLM / transformers models
    else:
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def count_tokens(text: str, tokenizer, is_transformers=True):
    """
    General token counting logic
    - transformers tokenizer.encode() returns a list of token ids
    - tiktoken behaves the same way
    """
    if is_transformers:
        return len(tokenizer.encode(text, add_special_tokens=False))
    else:
        return len(tokenizer.encode(text))

def truncate_content_to_fit_token_limit(prefix: str,
                                        content: str,
                                        tokenizer,
                                        max_token_n: int,
                                        is_transformers: bool = True) -> str:
    """
    Truncate the content so that the total number of tokens in prefix + content does not exceed max_token_n.

    Parameters:
    - prefix: a prefix string that must not be truncated
    - content: the main body content that can be truncated
    - tokenizer: a tokenizer from either transformers or tiktoken
    - max_token_n: the maximum number of allowed tokens
    - is_transformers: indicates the tokenizer type (True means transformers)

    Returns:
    - The concatenated string with the content truncated if necessary
    """
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False) if is_transformers else tokenizer.encode(text)

    prefix_tokens = encode(prefix)
    content_tokens = encode(content)
    
    remaining_tokens = max_token_n - len(prefix_tokens)
    if remaining_tokens <= 0:
        return prefix[:100] + "..."  # fallback: prefix is too long

    truncated_content_tokens = content_tokens[-remaining_tokens:]
    truncated_content = tokenizer.decode(truncated_content_tokens)
    
    return prefix + truncated_content


def truncate_and_concatenate(
    source_code: str,
    prefix_text: str,
    suffix_text: str,
    tokenizer,
    max_input_tokens: int
) -> tuple[str, bool, bool]:
    """
    Truncate `prefix_text` and `source_code` to fit within `max_input_tokens`, then concatenate them.
    
    - `suffix_text` is **never truncated**.
    - `prefix_text` is wrapped in triple quotes and truncated from the **right**.
    - `source_code` is truncated from the **left**.
    - Final structure: "'''\n{prefix_text}\n'''\n{suffix_text}{source_code}"

    Example:
        >>> truncate_and_concatenate(
        >>>     source_code="print('hello')\\n" * 200,
        >>>     prefix_text="This function prints logs.",
        >>>     suffix_text="# utils/logger.py",
        >>>     tokenizer=tokenizer,
        >>>     max_input_tokens=1024,
        >>> )

    Args:
        source_code (str): The main code snippet.
        prefix_text (str): Informational or contextual text to prepend.
        suffix_text (str): Extra information to add after prefix. Will not be truncated.
        tokenizer: A tokenizer with `encode` and `decode` methods (e.g., from HuggingFace).
        max_input_tokens (int): Total token budget.

    Returns:
        full_input_str (str): Concatenated and truncated result.
        source_truncated (bool): Whether `source_code` was truncated.
        prefix_truncated (bool): Whether `prefix_text` was truncated.
    """
    source_truncated = False
    prefix_truncated = False

    # Structure: '''<prefix>'''\n<suffix><source>
    prefix_block = prefix_text
    
    suffix_tokens = tokenizer.encode(suffix_text, add_special_tokens=False)
    prefix_tokens = tokenizer.encode(prefix_block, add_special_tokens=False)
    source_tokens = tokenizer.encode(source_code, add_special_tokens=False)

    suffix_len = len(suffix_tokens)
    prefix_len = len(prefix_tokens)
    source_len = len(source_tokens)

    # Available budget for prefix and source (suffix must fit)
    available = max_input_tokens - suffix_len
    if available < 0:
        raise ValueError("Suffix is too long to fit into the input length.")

    # Case 1: source_code fits, try truncate prefix
    if source_len <= 0.5 * max_input_tokens:
        max_prefix_len = available - source_len
        if prefix_len > max_prefix_len:
            prefix_tokens = prefix_tokens[:max_prefix_len]
            prefix_truncated = True
    # Case 2: prefix fits, truncate source_code
    elif prefix_len <= 0.5 * max_input_tokens:
        max_source_len = available - prefix_len
        if source_len > max_source_len:
            source_tokens = source_tokens[-max_source_len:]
            source_truncated = True
    # Case 3: both too long, split budget 50/50
    else:
        half_available = available // 2
        prefix_tokens = prefix_tokens[:half_available]
        source_tokens = source_tokens[-(available - len(prefix_tokens)):]
        prefix_truncated = True
        source_truncated = True

    final_tokens = prefix_tokens + suffix_tokens + source_tokens
    full_input_str = tokenizer.decode(final_tokens, skip_special_tokens=True)

    return full_input_str, source_truncated, prefix_truncated

def merge_retrieval(
    retrieval_infos: list[str],
    source_code_prefix: str,
    source_code: str,
    tokenizer,
) -> tuple[str, bool, bool]:
    """
    Merge retrieval information with source code.

    Args:
        retrieval_infos (list[str]): List of retrieval information.
        source_code_prefix (str): Prefix for the source code.
        source_code (str): The main source code.

    Returns:
        tuple: Merged string, whether the retrival was truncated, and whether the source code was truncated.
    """
    before_prompts: list[str] = []
    if len(retrieval_infos) > 0:
        before_prompts.append("'''")


    before_context: list[str] = []
    if len(retrieval_infos) > 0:
        before_context.append("'''")
    before_context.append(source_code_prefix)

    prefix_prompt = "\n".join(before_prompts) + "\n" + "\n".join(retrieval_infos) # may be too long so that it needs to be truncated from the right
    suffix_prompt = "\n" + "\n".join(before_context) + "\n"
    result = truncate_and_concatenate(
        source_code=source_code,
        prefix_text=prefix_prompt,
        suffix_text=suffix_prompt,
        tokenizer=tokenizer,
        max_input_tokens=settings.build_prompt.max_token_n,
    )
    return result[0], result[2], result[1]