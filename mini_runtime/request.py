from dataclasses import dataclass
import asyncio

@dataclass
class Request:
    request_id: int
    prompt: str
    submit_time: float
    future: asyncio.Future
    prompt_tokens: int
    max_new_tokens: int 
    
    start_time: float | None = None
    finish_time: float | None = None
    ttft: float | None = None
    tpot: float | None = None