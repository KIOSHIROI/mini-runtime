import asyncio
import torch
from asyncio import Queue
from .request import Request
from .metrics import Metrics
from .kv_cache import KVCacheManager, BlockTable
from .prefix_cache import PrefixCache
from .config import BLOCK_SIZE, NUM_BLOCKS, MAX_TOKENS_PER_PREFILL_CHUNK, MAX_TOKENS_PER_PREFILL_STEP
from .backends.native_backend import NativeBackend, PrefillInput, BatchDecodeInput
from .profiler import EngineProfiler, MemoryProfiler

class Engine:
    def __init__(
        self,
        backend = None,
        max_batch_size: int = 4,
        request_timeout: float = 30.0,
        num_blocks: int = NUM_BLOCKS,
        block_size: int = BLOCK_SIZE,
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.backend = backend
        
        self.waiting_queue = Queue()
        self.prefilling_requests = []   # 持有 blocks 但 prefill 未完成 
        self.running_requests = []      # prefill 完成，正在 decode
        self.max_batch_size = max_batch_size
        self.request_timeout = request_timeout

        self.next_request_id = 0
        self.engine_task = None
        self.metrics = Metrics()
        self.block_size = block_size

        self.kv_manager = KVCacheManager(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=backend.model.config.num_layers,
            num_kv_heads=backend.model.config.num_kv_heads,
            head_dim=backend.model.config.head_dim,
            device=device,
            dtype=backend.model.embed_tokens.weight.dtype,
        )

        self.backend.kv_manager = self.kv_manager
        self.prefix_cache = PrefixCache(block_size=block_size)
        
        self.engine_profiler = EngineProfiler()
        self.memory_profiler = MemoryProfiler()
        
    async def start(self):
        self.engine_task = asyncio.create_task(self.scheduler_loop())
    
    async def scheduler_loop(self):
        ep = self.engine_profiler
        mp = self.memory_profiler
        while True:
            try:
                ep.step_start()
                
                n = await self.admit_requests()
                ep.admit_done(n)
                
                mp.record("before_prefill", snap = mp.snapshot(self.backend.model, self.kv_manager, str(self.backend.device)))
                p_tokens, p_reqs = await self.prefill_step()
                mp.record("after_prefill", mp.snapshot(self.backend.model, self.kv_manager, str(self.backend.device)))
                ep.prefill_done(p_tokens, p_reqs)
                
                d_tokens, d_reqs = await self.decode_one_step()
                mp.record("after_decode", mp.snapshot(self.backend.model, self.kv_manager, str(self.backend.device)))
                ep.decode_done(d_tokens, d_reqs)
                
                ep.step_end(
                    waiting=self.waiting_queue.qsize(),
                    prefilling=len(self.prefilling_requests),
                    running=len(self.running_requests),
                    budget_used=p_tokens,
                    budget_total=MAX_TOKENS_PER_PREFILL_STEP
                )
            except torch.OutOfMemoryError as e:
                import sys
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"\n    [OOM @ step {ep.step_no}] allocated={allocated:.1f}GB reserved={reserved:.1f}GB: {e}", file=sys.stderr)
                self._fail_all_requests("OOM")
                break
            except Exception as e:
                self._fail_all_requests(f"error: {e}")
                break

    def _fail_all_requests(self, error: str):
        """scheduler loop 崩溃时，resolve 所有未完成的请求。"""
        for r in self.prefilling_requests:
            if r.block_table:
                self.kv_manager.free(r.block_table)
            if not r.future.done():
                r.future.set_result({"request_id": r.request_id, "error": error})
                self.metrics.oom += 1
        for r in self.running_requests:
            if r.block_table:
                self.kv_manager.free(r.block_table)
            if not r.future.done():
                r.future.set_result({"request_id": r.request_id, "error": error})
                self.metrics.oom += 1
        while not self.waiting_queue.empty():
            try:
                r = self.waiting_queue.get_nowait()
                if not r.future.done():
                    r.future.set_result({"request_id": r.request_id, "error": error})
                    self.metrics.oom += 1
                self.waiting_queue.task_done()
            except asyncio.QueueEmpty:
                break
        self.prefilling_requests.clear()
        self.running_requests.clear()
        self.kv_manager.release_all()
    async def admit_requests(self) -> int:
        admitted = 0
        while len(self.running_requests) + len(self.prefilling_requests) < self.max_batch_size:
            try:
                request: Request = self.waiting_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            request.start_time = asyncio.get_running_loop().time()
            match_result = self.prefix_cache.match(request.token_ids)
            matched_blocks = match_result['matched_blocks']
            num_matched_tokens = match_result['num_matched_tokens']
            matched_offset = match_result['matched_offset']
            num_matched_blocks = len(matched_blocks)

            block_table = BlockTable(self.block_size)
            block_table.set_offset(matched_offset)
            # 1. 复用 matched blocks (运行请求引用 +1)
            for bid in matched_blocks:
                self.kv_manager.inc_ref(bid)
                block_table.append_block(bid)
            # 2. 为 remaining 分配新 block; OOM 时 evict prefix cache 重试
            if match_result['remaining_tokens']:
                can_allocate = self.kv_manager.allocate(block_table, len(request.token_ids))
                while not can_allocate:
                    evicted = self.prefix_cache.evict()
                    if evicted is None:
                        break
                    for bid in evicted:
                        self.kv_manager.dec_ref(bid)
                    can_allocate = self.kv_manager.allocate(block_table, len(request.token_ids))
                if not can_allocate:
                    # 回滚已复用的 matched blocks
                    for bid in matched_blocks:
                        self.kv_manager.dec_ref(bid)
                    self.waiting_queue.task_done()
                    self.metrics.oom += 1
                    break

            request.block_table = block_table
            request.num_matched_tokens = num_matched_tokens
            request.matched_offset = matched_offset
            request.num_matched_blocks = num_matched_blocks
            request.match_result = match_result
            request.prefill_progress = num_matched_tokens  # 初始化 prefill_progress 为已复用的 token 数
            self.prefilling_requests.append(request)
            self.metrics.max_running_requests = max(
                self.metrics.max_running_requests, len(self.running_requests)
            )
            
            admitted += 1
            self.waiting_queue.task_done()
        return admitted
    
    async def prefill_step(self):
        if not self.prefilling_requests:
            return 0, 0
        self.metrics.prefill_batches += 1
        budget = MAX_TOKENS_PER_PREFILL_STEP
        max_chunk = MAX_TOKENS_PER_PREFILL_CHUNK
        migrated = [] # 本轮完成的 prefill 的请求
        
        step_tokens = 0
        step_reqs = 0
        for r in self.prefilling_requests:
            if budget <= 0:
                break
            
            prompt_len = len(r.token_ids)
            start = r.prefill_progress
            
            if start < prompt_len:
                chunk_len = min(budget, max_chunk, prompt_len - start)
                end = start + chunk_len
            else:
                # 全缓存命中
                chunk_len = 0
                end = prompt_len 
            is_last = (end == prompt_len)
            
            inp = PrefillInput(
                request_id=r.request_id,
                token_ids=r.token_ids,
                block_ids=r.block_table.block_ids,
                block_offset=r.block_table.offset,
                skip_tokens=start,
                num_cached_blocks=r.num_matched_blocks,
                chunk_start=start,
                chunk_end=end,
                is_last_chunk=is_last
            )
            
            result = self.backend.prefill(inp)  # 同步调用，避免 asyncio.to_thread 显存泄漏
            
            if is_last:
                # 插入 prefix cache
                new_cache_blocks = self.prefix_cache.insert(
                    r.token_ids, list(r.block_table.block_ids), r.match_result)
                for bid in new_cache_blocks:
                    self.kv_manager.inc_ref(bid)
                
                r._last_token = result
                r._generated_token_ids.append(result)
                r.generated_tokens = 1
                r.first_token_time = asyncio.get_running_loop().time()
                self.engine_profiler.request_first_token(r.request_id)
                r.prefill_done = True
                migrated.append(r)
            
            r.prefill_progress = end
            budget -= chunk_len
            step_tokens += chunk_len
            step_reqs += 1
        
        # 迁移完成的请求
        for r in migrated:
            self.prefilling_requests.remove(r)
            self.running_requests.append(r)
    
        return step_tokens, step_reqs
            
    async def decode_one_step(self):
        step_tokens = 0
        
        if not self.running_requests:
            await asyncio.sleep(0.001)
            return 0, 0

        step_reqs = 0
        active = len(self.running_requests) 
               
        
        self.metrics.decode_steps += 1
        
        self.metrics.total_active_requests  += active  
          
        now = asyncio.get_running_loop().time()
                
        finished = []
        oom_requests = []

        for r in self.running_requests:
            total = len(r.token_ids) + r.generated_tokens
            if total > r.block_table.capacity:
                success = self.kv_manager.allocate(r.block_table, total)
                while not success:
                    evicted = self.prefix_cache.evict()
                    if evicted is None:
                        break
                    for bid in evicted:
                        self.kv_manager.dec_ref(bid)
                    success = self.kv_manager.allocate(r.block_table, total)
                if not success:
                    oom_requests.append(r)

        for r in oom_requests:
            self.running_requests.remove(r)
            self.kv_manager.free(r.block_table)
            self.metrics.oom += 1
            if not r.future.done():
                r.future.set_result({
                    "request_id": r.request_id,
                    "error": "OOM",
                })
            else:
                self.metrics.cancelled += 1

        batched = [BatchDecodeInput(
            request_id=r.request_id,
            token_id=r._last_token,
            block_ids=r.block_table.block_ids,
            block_offset=r.matched_offset,
        ) for r in self.running_requests]
        
        next_tokens = self.backend.batch_decode(batched)
        
        for r, next_token in zip(self.running_requests, next_tokens):                
            if r.first_token_time is None:
                r.first_token_time = now
                            
            r.generated_tokens += 1
            step_tokens += 1
            
            if next_token is not None:
                r._generated_token_ids.append(next_token)
            r._last_token = next_token

            if r.generated_tokens >= r.max_new_tokens or r._last_token is None:
                finished.append(r)
        
        step_reqs = len(self.running_requests)
        
        for r in finished:
            self.finish_request(r, now)
            self.running_requests.remove(r)
        
        return step_tokens, step_reqs
    
    def finish_request(self, request: Request, finish_time: float):
        if request.block_table:
            self.kv_manager.free(request.block_table)
        self.backend.release(request.request_id)
        if request.future.done():
            self.metrics.cancelled += 1
            return

        queue_wait = request.start_time - request.submit_time
        service_time = finish_time - request.start_time
        ttft = (request.first_token_time - request.submit_time
                if request.first_token_time is not None else None)
        total = finish_time - request.submit_time
        tpot = (
            (finish_time - request.first_token_time) / request.generated_tokens
            if request.generated_tokens and request.first_token_time is not None else 0
        )

        request.future.set_result({
            "request_id": request.request_id,
            "ttft": ttft,
            "tpot": tpot,
            "total": total,
            "generated_tokens": request.generated_tokens,
            "queue_wait": queue_wait,
            "service_time": service_time,
            "output": self.backend.tokenizer.decode(request._generated_token_ids),
        })
        self.engine_profiler.request_finish(request.request_id, request.generated_tokens)
        self.metrics.success += 1
        if ttft is not None:
            self.metrics.total_ttft += ttft
        self.metrics.total_tpot += tpot
        self.metrics.total_latency += total
        self.metrics.total_output_tokens += request.generated_tokens
        self.metrics.total_queue_wait += queue_wait
        self.metrics.total_service_time += service_time
        
    async def submit(self, prompt: str, max_new_tokens: int = 128):
        """_summary_
        用户调用该函数提交一个生成请求
        
        """
        loop = asyncio.get_running_loop()
        submit_time = loop.time()  # 在 tokenization 之前记录，TTFT 才能反映用户真实感知

        message = [{"role": "user", "content": prompt}]
        chat_text = self.backend.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.backend.tokenizer.encode(chat_text)
        future = loop.create_future()

        request = Request(
            request_id = self.next_request_id,
            prompt = prompt,
            token_ids = input_ids,
            max_new_tokens = max_new_tokens,
            submit_time = submit_time,
            future = future,
        )
        
        self.engine_profiler.request_start(request.request_id)
        
        self.next_request_id += 1
        
        await self.waiting_queue.put(request)
        self.metrics.submitted += 1
        
        try:
            return await asyncio.wait_for(future, timeout=self.request_timeout)
        except asyncio.TimeoutError:
            future.cancel()
            self.metrics.timeout += 1
            return {
                "request_id": request.request_id,
                "error": "timeout",
            }
            
    def snapshot_metrics(self):
        success = self.metrics.success 
        decode_steps = self.metrics.decode_steps
        
        kv_snapshot = self.kv_manager.snapshot()
        
        return {
            "submitted": self.metrics.submitted,
            "success": self.metrics.success,
            "timeout": self.metrics.timeout,
            "cancelled": self.metrics.cancelled,
            "oom": self.metrics.oom,
            "prefill_batches": self.metrics.prefill_batches,
            "decode_steps": self.metrics.decode_steps,
            "avg_active_requests": (
                self.metrics.total_active_requests / decode_steps
                if decode_steps else 0
            ),
            "max_running_requests": self.metrics.max_running_requests,
            "avg_queue_wait": (
                self.metrics.total_queue_wait / success
                if success else 0
            ),
            "avg_service_time": (
                self.metrics.total_service_time / success
                if success else 0
            ),
            "avg_latency": (
                self.metrics.total_latency / success
                if success else 0
            ),
            "avg_ttft": (
                self.metrics.total_ttft / success
                if success else 0
            ),
            "avg_tpot": (
                self.metrics.total_tpot / success
                if success else 0            ),
            "output_tokens_per_sec": (
                self.metrics.total_output_tokens / self.metrics.total_service_time
                if self.metrics.total_service_time > 0 else 0
            ),
            "kv_cache": kv_snapshot,
        }
    async def shutdown(self):
        # 取消等待队列中的请求
        while not self.waiting_queue.empty():
            try:
                request = self.waiting_queue.get_nowait()
                if not request.future.done():
                    request.future.set_result({
                        "request_id": request.request_id,
                        "error": "cancelled",
                    })
                    self.metrics.cancelled += 1
                self.waiting_queue.task_done()
            except asyncio.QueueEmpty:
                break

        # 释放正在运行请求的资源
        for request in self.prefilling_requests:
            if request.block_table:
                self.kv_manager.free(request.block_table)
            if not request.future.done():
                request.future.set_result({
                    "request_id": request.request_id,
                    "error": "cancelled",
                })
                self.metrics.cancelled += 1
                
        for request in self.running_requests:
            if request.block_table:
                self.kv_manager.free(request.block_table)
            if not request.future.done():
                request.future.set_result({
                    "request_id": request.request_id,
                    "error": "cancelled",
                })
                self.metrics.cancelled += 1

        self.prefilling_requests.clear()
        self.running_requests.clear()

        # 强制释放 prefix cache 引用的所有 block tensor
        self.kv_manager.release_all()

        if self.engine_task:
            self.engine_task.cancel()
            await asyncio.gather(self.engine_task, return_exceptions=True)
            