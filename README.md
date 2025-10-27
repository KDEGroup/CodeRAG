# CodeRAG

Source code for our **EMNLP 2025** paper: **"CodeRAG: Finding Relevant and Necessary Knowledge for Retrieval-Augmented Repository-Level Code Completion"** [[arXiv]](https://arxiv.org/abs/2509.16112).

---

## 📦 Environment Setup

### 1. **Install [uv](https://docs.astral.sh/uv/)**
### 2. **Synchronize dependencies**
   ```bash
   uv sync
   ```
### 3. **Activate the virtual environment**
   ```bash
   source .venv/bin/activate
   ```

---

## 🚀 Usage

Before running scripts, download benchmarks (recceval and cceval) and edit the configuration file:

```bash
config/config.toml
```

Then execute the Python scripts **sequentially**:

### 1. Build Query
```bash
python scripts/build_query.py
```
- Generates query strings from the benchmark dataset.

### 2. Retrieve Relevant Code Blocks
```bash
python scripts/retrieve.py
```
- Retrieves top-k relevant code blocks using the configured retriever.

### 3. Rerank Retrieved Code Blocks
```bash
python scripts/rerank.py
```
- Reranks retrieved code blocks based on their estimated importance.

### 4. Build Prompts for Generator
```bash
python scripts/build_prompt.py
```
- Constructs prompts from retrieved code blocks for the code completion generator.

### 5. Run Inference
```bash
python scripts/inference.py
```
- Feeds prompts to the generator model.
- You can replace this step with your own inference code.  
  **Input:** JSON file containing an array of strings  
  **Output:** JSON file containing an array of generated completions.

### 6. Evaluate Results
```bash
python scripts/evaluation.py
```
- Evaluates code completion performance using inference results.

---

## 📄 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@inproceedings{coderag2025,
  title={CodeRAG: Finding Relevant and Necessary Knowledge for Retrieval-Augmented Repository-Level Code Completion},
  author={Sheng Zhang, Yifan Ding, Shuquan Lian, Shun Song, Hui Li},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2025}
}
```

---

## 📧 Contact

For questions, please open an issue or contact **dingyf@stu.xmu.edu.cn**.
