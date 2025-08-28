from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

def split_retrieved_docs(
    raw_docs: List[str],
    chunk_size: int = 1024,
    chunk_overlap: int = 50
) -> List[Document]:
    """
    Split raw retrieved documents (with file path and content) into smaller chunks and add metadata.

    Parameters:
        raw_docs: List[str], each item is formatted as "# path/to/file.py\\n<file content>"
        chunk_size: int, maximum number of characters per chunk
        chunk_overlap: int, number of overlapping characters between adjacent chunks

    Returns:
        List[Document], each containing page_content and metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    split_documents = []

    for raw_doc in raw_docs:
        # Extract path and content
        lines = raw_doc.strip().splitlines()
        if not lines or not lines[0].startswith("# "):
            continue  # Skip invalid formats
        filepath = lines[0][2:].strip()
        content = "\n".join(lines[1:])

        # Split the text
        chunks = splitter.split_text(content)

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": filepath,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            )
            split_documents.append(doc)

    return split_documents

def documents_to_prompt_blocks_syntax_preserving(documents: List[Document]) -> List[str]:
    """
    Convert a list of Document objects into prompt blocks that preserve code syntax,
    e.g., retain the `# path/to/file.py` header, suitable for use as context in LLM inputs.

    Each returned string is formatted as:
    # path/to/file.py [i/N]
    <chunk content>
    """
    prompt_blocks = []

    for doc in documents:
        metadata = doc.metadata
        content = doc.page_content.strip()

        source = metadata.get("source", "unknown")
        chunk_id = metadata.get("chunk_id", -1)
        total_chunks = metadata.get("total_chunks", -1)

        # Header format
        if total_chunks > 1:
            header = f"# {source} [{chunk_id + 1}/{total_chunks}]"
        else:
            header = f"# {source}"

        prompt_block = f"{header}\n{content}"
        prompt_blocks.append(prompt_block)

    return prompt_blocks

def split_big_retrieval(retrieval: str) -> list[str]:
    docs = split_retrieved_docs([retrieval])
    return documents_to_prompt_blocks_syntax_preserving(docs)
