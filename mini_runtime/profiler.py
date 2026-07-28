from dataclasses import dataclass
import time
@dataclass 
class RequestTrace:
    request_id: int
    submit_time: float
    first_token_time: float | None = None
    finish_time: float | None = None
    generated_tokens: int = 0

@dataclass
class StepTrace:
    step_no: int
    admit_ms: float = 0.0
    admit_count: int = 0
    prefill_ms: float = 0.0 
    prefill_tokens: int = 0
    prefill_reqs: int = 0
    decode_ms: float = 0.0
    decode_tokens: int = 0
    decode_reqs: int = 0
    step_total_ms: float = 0.0

class EngineProfiler:
    """每个 scheduler step 记录一次 StepRecord, 每个请求记录一次 RequestRecord"""
    def __init__(self):
        self.requests: dict[int, RequestTrace] = {}
        self.steps: list[StepTrace] = []
        self._step_start: float = 0.0
        self._phase_start: float = 0.0
        self.step_no: int = 0
    
    def step_start(self):
        """scheduler_loop 迭代开始"""
        rt = StepTrace(step_no=self.step_no)
        self.steps.append(rt)
        self._step_start = time.perf_counter()
        self._phase_start = self._step_start
        

    
    def admit_done(self, n: int):
        """admit requests 结束, n = 本轮 admit 了几个"""
        now = time.perf_counter()
        admit_time = (now - self._phase_start) * 1000
        self.steps[-1].admit_count = n
        self.steps[-1].admit_ms = admit_time
        self._phase_start = now
        
    
    def prefill_done(self, tokens: int, reqs: int):
        """prefill_step 结束, tokens = 本轮 prefill 了多少 token, reqs = 本轮 prefill 了多少请求"""
        now = time.perf_counter()
        prefill_time = (now - self._phase_start) * 1000
        self.steps[-1].prefill_tokens = tokens
        self.steps[-1].prefill_reqs = reqs
        self.steps[-1].prefill_ms = prefill_time
        self._phase_start = now
    
    def decode_done(self, tokens: int, reqs: int):
        """decode_one_step 结束"""
        now = time.perf_counter()
        decode_time = (now - self._phase_start) * 1000
        self.steps[-1].decode_tokens = tokens
        self.steps[-1].decode_reqs = reqs
        self.steps[-1].decode_ms = decode_time
        self._phase_start = now
        
    def request_start(self, request_id: int):
        """request 开始处理"""
        rs = RequestTrace(request_id=request_id, submit_time=time.perf_counter())
        
        self.requests[request_id] = rs
    
    def request_first_token(self, request_id: int):
        """request 第一个 token 生成完成"""
        rs = self.requests[request_id]
        rs.first_token_time = time.perf_counter()
    
    def request_finish(self, request_id: int, gen_tokens: int):
        """request 完成"""
        rs = self.requests[request_id]
        rs.finish_time = time.perf_counter()
        rs.generated_tokens = gen_tokens
    
    def step_end(self):
        """scheduler_loop 迭代结束, 汇总本 step"""
        self.steps[self.step_no].step_total_ms = (time.perf_counter() - self._step_start) * 1000
        self.step_no += 1
    
    def summary(self) -> str:
        """打印汇总统计"""
        print(
        f"""
        === Engine Profiler Summary ===
        Steps: {len(self.steps)}
        Total time: {sum(step.step_total_ms for step in self.steps)} ms
        
        Per-step averages:
            admit:\t{sum(step.admit_ms for step in self.steps) / len(self.steps) if self.steps else 0:.2f} ms
            prefill:\t{sum(step.prefill_ms for step in self.steps) / len(self.steps) if self.steps else 0:.2f} ms
            decode:\t{sum(step.decode_ms for step in self.steps) / len(self.steps) if self.steps else 0:.2f} ms
        
        Request stats:
            completed:\t{len([r for r in self.requests.values() if r.finish_time is not None])}
            avg TTFT:\t{sum(r.first_token_time - r.submit_time for r in self.requests.values() if r.first_token_time is not None) * 1000 / len([r for r in self.requests.values() if r.first_token_time is not None]) if [r for r in self.requests.values() if r.first_token_time is not None] else 0:.2f} ms
            avg TPOT:\t{sum((r.finish_time - r.first_token_time) / r.generated_tokens for r in self.requests.values() if r.finish_time is not None and r.generated_tokens > 0) * 1000 / len([r for r in self.requests.values() if r.finish_time is not None and r.generated_tokens > 0]) if [r for r in self.requests.values() if r.finish_time is not None and r.generated_tokens > 0] else 0:.2f} ms
            avg total:\t{sum(r.finish_time - r.submit_time for r in self.requests.values() if r.finish_time is not None) * 1000 / len([r for r in self.requests.values() if r.finish_time is not None]) if [r for r in self.requests.values() if r.finish_time is not None] else 0:.2f} ms
        --------------------------------
        """
        )