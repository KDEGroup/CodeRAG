# CodeRAG
Source code for EMNLP 2025 paper "CodeRAG: Finding Relevant and Necessary Knowledge for Retrieval-Augmented Repository-Level Code Completion"
## set up environment
- install uv from https://docs.astral.sh/uv/
- run
```bash
uv sync
```
- then activate virtural python environment
```bash
source .venv/bin/activate
```
## run scripts
edit config/config.toml then use python run py files sequencely. e.g. python scripts/build_query.py
- scripts/build_query.py
- scripts/retrieve.py
- scripts/rerank.py
- scripts/build_prompt.py
- script/inference.py
    - you can use your own code to do this step. take an array[string] json then return an array[string] json
- script/evaluation.py