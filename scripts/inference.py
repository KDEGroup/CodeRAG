import asyncio
from typing import List

from loguru import logger
from coderag.config import settings
from openai import AsyncOpenAI
from coderag.inference.limited_async_openai import get_client
import json

def main():
    with open(settings.inference.use_prompt_file, "r") as f:
        prompt_list = json.load(f)

    if settings.sample_n is not None:
        prompt_list = prompt_list[:settings.sample_n]
    inference_client = AsyncOpenAI(
        api_key=settings.inference.api_key,
        base_url=settings.inference.api_url,
    )
    inference_client = get_client(
        api_key=settings.inference.api_key,
        base_url=settings.inference.api_url,
        max_concurrent_requests=4096,
        timeout=600,
        retries=3,
    )

    semaphore = asyncio.Semaphore(512) 
    async def inference_all() -> List[str]:
        error_count = 0
        error_ids = []
        async def inference_task(prompt: str, idx: int) -> str:
            nonlocal error_count
            nonlocal error_ids
            task_id = f"Task {idx + 1}/{len(prompt_list)}"
            try:
                # Log *before* trying to acquire the semaphore
                logger.debug(f"{task_id}: Waiting for semaphore...")
                async with semaphore:
                    # Log *after* acquiring the semaphore
                    logger.debug(f"{task_id}: Acquired semaphore. Starting API call...")

                    # Timestamp before the potentially long call
                    start_time = asyncio.get_event_loop().time()

                    if "Qwen2.5-Coder" in settings.inference.model:
                        prompt = f"<|fim_prefix|>{prompt}<|fim_suffix|><|fim_middle|>"
                    res = await inference_client.completions.create(
                        model=settings.inference.model,
                        prompt=prompt,
                        max_tokens=settings.inference.max_tokens,
                        temperature=0
                        # You might want to add/test a client-side timeout
                        # timeout=60.0 # e.g., 60 seconds
                    )
                    result_text = res.choices[0].text

                    # Timestamp after the call returns
                    end_time = asyncio.get_event_loop().time()
                    duration = end_time - start_time
                    logger.debug(f"{task_id}: API call finished in {duration:.2f} seconds.")

                    log_message = (
                        f"{task_id}: Processing result. "
                        f"Prompt starts: '{prompt[:50] if isinstance(prompt, str) else prompt[1]["content"][:50]}...', "
                        f"Result starts: '{result_text[:50]}...'"
                    )
                    logger.debug(log_message)
                    return result_text

            # Catch more specific exceptions if possible (e.g., openai.APIError, TimeoutError)
            except asyncio.TimeoutError:
                logger.error(f"{task_id}: API call timed out.")
                error_count += 1
                error_ids.append(idx)
                return ""
            except Exception as e:
                error_count += 1
                error_ids.append(idx)
                logger.error(f"{task_id}: Error during inference for prompt '{prompt[:50]}...': {e}")
                return ""
            finally:
                # Optional: Log when a task fully completes (after semaphore release)
                logger.debug(f"{task_id}: Task complete.")

        tasks = []
        for idx, prompt in enumerate(prompt_list):
            tasks.append(inference_task(prompt, idx))

        results = await asyncio.gather(*tasks)
        logger.info(f"Total errors: {error_count}")
        logger.info(f"Error IDs: {error_ids}")
        return results

    inference_result = asyncio.run(inference_all())
    logger.debug("after inference all")
    settings.inference.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.inference.output_file, "w") as f:
        json.dump(inference_result, f, indent=4)
    logger.info("Inference completed and results saved.")

if __name__ == "__main__":
    main()