from pydantic import BaseModel
from pathlib import Path
import json
import os
from loguru import logger

class CodeLocation(BaseModel):
    path_list: list[str] # e.g. repo_a/folder_a/folder_b/c.py
    start_line_no: int
    end_line_no: int

class BenchmarkItem(BaseModel):
    repo_name: str
    task_name: str
    code_context: str
    file_path: Path
    code_context_location: CodeLocation
    deduped_path_list: list[str]
    ground_truth: str

class CodeRepo(BaseModel):
    repo_name: str
    repo_path: Path

class Benchmark(BaseModel):
    data_list: list[BenchmarkItem]
    repos: list[CodeRepo]

    def get_repo(self, repo_name: str) -> CodeRepo:
        for repo in self.repos:
            if repo.repo_name == repo_name:
                return repo
        raise ValueError(f"Cannot find repo {repo_name}")

def load_jsonl(fname: Path) -> list:
    with open(fname, 'r', encoding='utf8') as f:
        lines = []
        for line in f:
            lines.append(json.loads(line))
        return lines

def iterate_repos(repo_base_path: Path) -> dict[str, list[tuple[list[str], str]]]:
    result = {}
    for subdir in Path(repo_base_path).iterdir():
        result[subdir.name] = iterate_repository(repo_base_path / subdir.name)
    return result
    
def iterate_repository(repo_path: Path) -> list[tuple[list[str], str]]:
    parent_path = Path(repo_path).parent
    result = []

    # Traverse the directory and its subdirectories
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):  # Process only .py files
                # Get the relative path of the file
                full_path = Path(os.path.join(root, file))
                
                # Open the file and read its content
                try:
                    with open(full_path, 'r', encoding="utf-8") as f:
                        content = f.read()
                        # Add the file path and content to the result
                        result.append((list(full_path.relative_to(parent_path).parts), content))
                except UnicodeDecodeError as e:
                    logger.warning(f"Failed to open file {full_path}: {e}")
                except IOError as e:
                    logger.warning(f"Failed to open file {full_path}: {e}")

    return result
