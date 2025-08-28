from dataclasses import dataclass
from typing import Iterable, Sequence, TypedDict
from openai.types.chat import ChatCompletionMessageParam


reasoning_system = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. \
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., \
<think> reasoning process here </think> <answer> answer here </answer>."

ordinary_system = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. \
The assistant provides the user with the answer enclosed within <answer> </answer> tags, i.e., <answer> answer here </answer>."

think_user_code_set_wise = """
Which of the retrieved code snippets is most helpful for completing the following code snippet?
the code snippet to be completed:
{query}

the retrieved code snippet(s):
{documents}

After completing the reasoning process, please provide only the label of the most helpful retrieved code snippet, enclosed in square brackets, within the answer tags. \
For example, if the code snippet C is the most helpful, the answer should be: <answer>[C]</answer>.
"""


user_code_set_wise = """
Which of the retrieved code snippets is most helpful for completing the following code snippet?
the code snippet to be completed:
{query}

the retrieved code snippet(s):
{documents}

Please provide only the label of the most helpful retrieved code snippet, enclosed in square brackets, within the answer tags. \
For example, if the code snippet C is the most helpful, the answer should be: <answer>[C]</answer>.
"""



CHARACTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
                "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
                "W"]  # "Passage X" and "Passage Y" will be tokenized into 3 tokens, so we dont use for now

def build_prompt(template: str, query: str, docs: Sequence[str]) -> str:
    assert len(docs) <= len(CHARACTERS)
    documents: list[str] = []
    for idx, doc in enumerate(docs):
        documents.append(f"[{CHARACTERS[idx]}]\n{doc}")
    return template.format(
        query=query,
        documents="\n".join(documents)
    )


def build_message(query: str, docs: Sequence[str], think: bool) -> list[dict]:
    if think:
        return [
            {"role": "system", "content": reasoning_system},
            {"role": "user", "content": build_prompt(template=think_user_code_set_wise, query=query, docs=docs)}
        ]
    else:
        return [
            {"role": "system", "content": ordinary_system},
            {"role": "user", "content": build_prompt(template=user_code_set_wise, query=query, docs=docs)}
        ]

        
class AnswerFormatError:
    """
    Raised when the llm's response has a wrong format.
    """
    pass

@dataclass
class ThinkTagError(AnswerFormatError):
    num: int

@dataclass
class AnswerTagError(AnswerFormatError):
    """
    Raised when the <answer> tags are missing or contain invalid content.
    """
    num: int
    content: str