import asyncio 
from asyncio import Queue
from dataclasses import dataclass 
import csv
# batch prefill -> first token -> batch decode -> result
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
    
@dataclass
class Metrics:
    submitted: int = 0
    success: int = 0
    timeout: int = 0
    cancelled: int = 0
    batches: int = 0
    total_batch_size: int = 0
    total_queue_wait: float = 0.0
    total_service_time: float = 0.0
    total_latency: float = 0.0
    total_ttft: float = 0.0
    total_tpot: float = 0.0
    total_output_tokens: int = 0
    
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
        
class MiniRuntime:
    def __init__(
        self,
        max_batch_size: int = 4,
        batch_timeout: float = 0.5,
        request_timeout: float = 1.0,
        num_workers: int = 3,
        backend: SimulatedLLMBackend = SimulatedLLMBackend(),
     ):
        self.request_queue = asyncio.Queue()
        self.batch_queue = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.request_timeout = request_timeout
        self.num_workers = num_workers
        self.scheduler_task = None
        self.worker_tasks = []  
        self.next_request_id = 0
        self.metrics = Metrics()
        self.backend =backend 
    
    def snapshot_metrics(self):
        success = self.metrics.success 
        batches = self.metrics.batches 
        
        return {
            "submitted": self.metrics.submitted,
            "success": self.metrics.success,
            "timeout": self.metrics.timeout,
            "cancelled": self.metrics.cancelled,
            "batches": self.metrics.batches,
            "avg_batch_size": (
                self.metrics.total_batch_size / batches
                if batches else 0
            ),
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
        }
    
    async def start(self):
        self.scheduler_task = asyncio.create_task(self.scheduler())
        self.worker_tasks = [
            asyncio.create_task(self.worker(i))
            for i in range(self.num_workers)
        ]
    
    async def submit(
        self,
        prompt: str,
        prompt_tokens: int = 32,
        max_new_tokens: int = 16,
    ):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request_id = self.next_request_id

        self.next_request_id += 1
        
        request = Request(
            request_id=request_id,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            submit_time=loop.time(),
            future=future
        )
        
        await self.request_queue.put(request)
        self.metrics.submitted += 1
        print(f"submit request-{request_id}, "
        f"queue size={self.request_queue.qsize()}")
        
        try:
            return await asyncio.wait_for(future, timeout=self.request_timeout)
        except asyncio.TimeoutError:
            future.cancel()
            self.metrics.timeout += 1
            return {
                'request_id': request_id,
                'status': 'timeout'
            }
        
    async def scheduler(self):
        while True:
            batch = []
            
            try:
                while len(batch) < self.max_batch_size:
                    request = await asyncio.wait_for(self.request_queue.get(), timeout=self.batch_timeout)
                    self.request_queue.task_done()
                    batch.append(request)
            except asyncio.TimeoutError:
                pass

            if batch:
                self.metrics.batches += 1
                self.metrics.total_batch_size += len(batch)
                await self.batch_queue.put(batch)
                
    async def worker(self, worker_id: int):
        loop = asyncio.get_running_loop()
        while True:
            batch = await self.batch_queue.get()
            print(f"worker-{worker_id} "
                f"processing batch "
                f"size={len(batch)}")
            
            start = loop.time()
            
            for req in batch:
                req.start_time = start
                
            first_token_time, finish = await self.backend.generate_batch(batch)
            
            finish = loop.time()
            
            for request in batch:
                request.finish_time = finish
            
                queue_wait = (
                    request.start_time
                    - request.submit_time
                )
                service_time = (
                    request.finish_time
                    - request.start_time
                )
                total_latency = (
                    request.finish_time
                    - request.submit_time
                )
                ttft = first_token_time - request.submit_time
                tpot = (
                    (finish - first_token_time) / request.max_new_tokens
                    if request.max_new_tokens else 0
                )
                if not request.future.done():
                    request.future.set_result({
                        "request_id": request.request_id,
                        "output": f"response to {request.prompt}",
                        "wait": queue_wait,
                        "service": service_time,
                        "total": total_latency,
                        "ttft": ttft,
                        "tpot": tpot,
                    })
                    self.metrics.success += 1
                    self.metrics.total_queue_wait += queue_wait
                    self.metrics.total_service_time += service_time
                    self.metrics.total_latency += total_latency
                    self.metrics.total_ttft += ttft
                    self.metrics.total_tpot += tpot
                    self.metrics.total_output_tokens += request.max_new_tokens
                else:
                    self.metrics.cancelled += 1
            self.batch_queue.task_done()
        
    async def shutdown(self):
        await self.request_queue.join()
        await self.batch_queue.join()
        
        if self.scheduler_task:
            self.scheduler_task.cancel()

        for task in self.worker_tasks:
            task.cancel()
        
        await asyncio.gather(
            self.scheduler_task,
            *self.worker_tasks,
            return_exceptions=True
        )

def make_workload(kind: str, num_requests: int):
    if kind == "spso":
        return [
            {"prompt_tokens": 16, "max_new_tokens": 16}
            for _ in range(num_requests)
        ]
    if kind == "lpso":
        return [
            {"prompt_tokens": 256, "max_new_tokens": 16}
            for _ in range(num_requests)
        ]
    if kind == "splo":
        return [
            {"prompt_tokens": 16, "max_new_tokens": 128}
            for _ in range(num_requests)
        ]
    if kind == "mixed":
        return [
            {"prompt_tokens": 16, "max_new_tokens": 16}
            if i % 2 == 0
            else {"prompt_tokens": 256, "max_new_tokens": 128}
            for i in range(num_requests)
        ]
    
    raise ValueError(f"unknown workload kind: {kind}")

async def run_benchmark(
    num_requests: int,
    concurrency: int,
    max_batch_size: int,
    num_workers: int,
    request_timeout: float,
    workload_kind: str,
):
    runtime = MiniRuntime(
        max_batch_size = max_batch_size,
        num_workers = num_workers,
        request_timeout = request_timeout,
    )
    
    await runtime.start()
    
    sem = asyncio.Semaphore(concurrency)
    
    workload = make_workload(workload_kind, num_requests)
    
    async def one_request(i: int):
        item = workload[i]
        async with sem:
            return await runtime.submit(
                f"request-{i}",
                prompt_tokens=item["prompt_tokens"],
                max_new_tokens=item["max_new_tokens"],
            )
            
    start = asyncio.get_running_loop().time()
    
    tasks = [
        asyncio.create_task(one_request(i))
        for i in range(num_requests)
    ]
    
    results = await asyncio.gather(*tasks)

    end = asyncio.get_running_loop().time()
    
    await runtime.shutdown()
    
    metrics = runtime.snapshot_metrics() 
    metrics['workload'] = workload_kind
    metrics["max_batch_size"] = max_batch_size
    metrics["num_workers"] = num_workers
    metrics["concurrency"] = concurrency
    metrics['duration'] = end - start
    metrics['throughput_rps'] = num_requests / (end - start) if (end - start) > 0 else 0
    
    return results, metrics

def write_metrics_csv(path: str, rows: list[dict]):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        
async def main():
    all_metrics = []
    # n_r, concur, bs, n_w, to
    configs = [
        (20, 10, 4, 3, 30.0, "spso"),
        (20, 10, 4, 3, 30.0, "lpso"),
        (20, 10, 4, 3, 30.0, "splo"),
        (20, 10, 4, 3, 30.0, "mixed"),
    ]
    for config in configs:
        _, metrics = await run_benchmark(
            num_requests=config[0],
            concurrency=config[1],
            max_batch_size=config[2],
            num_workers=config[3],
            request_timeout=config[4],
            workload_kind=config[5],
        )
        all_metrics.append(metrics)

    for metrics in all_metrics:
        print(metrics)
       
    write_metrics_csv("benchmark_results.csv", all_metrics) 
        


if __name__ == "__main__":
    asyncio.run(main())
    
        