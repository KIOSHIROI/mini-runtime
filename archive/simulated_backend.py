import asyncio
from .request import Request

class SimulatedLLMBackend:
    def __init__(
        self,
        prefill_time_per_token: float = 0.01,
        decode_time_per_token: float = 0.05,
    ):
        self.prefill_time_per_token = prefill_time_per_token
        self.decode_time_per_token = decode_time_per_token
    
    async def generate_batch(self, batch: list[Request]):
        max_prompt_tokens = max(r.prompt_tokens for r in batch)
        max_new_tokens = max(r.max_new_tokens for r in batch)
        
        prefill_time = max_prompt_tokens * self.prefill_time_per_token 
        await asyncio.sleep(prefill_time)
        
        first_token_time = asyncio.get_running_loop().time()
        
        decode_time = max_new_tokens * self.decode_time_per_token 
        await asyncio.sleep(decode_time)
        
        finish_time = asyncio.get_running_loop().time()
        
        return first_token_time, finish_time 
        