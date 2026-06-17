from dataclasses import dataclass, field
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

    _last_token: int = 0
    _generated_token_ids: list = field(default_factory=list) # 函数默认参数求值陷阱
    generated_tokens: int = 0
    prefill_done: bool = False
    first_token_time: float | None = None
    start_time: float | None = None
    finish_time: float | None = None
    ttft: float | None = None
    tpot: float | None = None
    
    block_table: BlockTable | None = None 
    