from openai import AsyncOpenAI
import httpx

def get_client(
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    max_concurrent_requests: int = 5,
    timeout: float = 10.0,
    retries: int = 2,
) -> AsyncOpenAI:
    limits = httpx.Limits(
        max_connections = max_concurrent_requests,
        max_keepalive_connections = max_concurrent_requests,
    )
    http_client = httpx.AsyncClient(
        timeout = httpx.Timeout(timeout),
        limits = limits,
        transport = httpx.AsyncHTTPTransport(retries=retries),
    )
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client
    )