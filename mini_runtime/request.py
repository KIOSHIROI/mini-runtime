from dataclasses import dataclass
import asyncio
from .kv_cache import BlockTable
@dataclass
class Request:
    request_id: int
    prompt: str
    submit_time: float
    future: asyncio.Future
    prompt_tokens: int
    max_new_tokens: int 

    generated_tokens: int = 0
    prefill_done: bool = False
    first_token_time: float | None = None
    start_time: float | None = None
    finish_time: float | None = None
    ttft: float | None = None
    tpot: float | None = None
    
    block_table: BlockTable | None = None 
    