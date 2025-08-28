from typing import Any, Dict, Optional
import httpx
from asyncio import Semaphore

semaphore = Semaphore(10)
timeout = httpx.Timeout(300)
async def ask_llm_tgi(
    prompt: str,
    api_url: str,
    parameters: Optional[Dict[str, Any]] = None
) -> Any:
    payload = {
        "inputs": prompt,
        "parameters": parameters or {}
        
    }
    async with semaphore:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(api_url, json=payload)
            return response.json()

