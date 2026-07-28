from dataclasses import dataclass
import time
import torch
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
    
    # scheuler profile
    waiting_depth: int = 0      # admit 前等待队列深度
    prefilling_count: int = 0   # admit后 prefilling 请求数
    running_count: int = 0      # admit后 running 请求数
    budget_pct: float = 0.0     # prefill budget 利用率（%）

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
    
    def step_end(self, waiting: int, prefilling: int, running: int, budget_used: int, budget_total: int):
        """scheduler_loop 迭代结束, 汇总本 step"""
        self.steps[self.step_no].step_total_ms = (time.perf_counter() - self._step_start) * 1000
        self.steps[self.step_no].waiting_depth = waiting
        self.steps[self.step_no].prefilling_count = prefilling
        self.steps[self.step_no].running_count = running
        self.steps[self.step_no].budget_pct = (budget_used / budget_total) * 100 if budget_total > 0 else 0
        self.step_no += 1
    
    def summary(self):
        """打印汇总统计。"""
        if not self.steps:
            return

        s = self.steps
        n = len(s)
        total_ms = sum(step.step_total_ms for step in s)

        def avg(values):
            return sum(values) / n

        # request stats
        completed = [r for r in self.requests.values() if r.finish_time is not None]
        c = len(completed)
        ttft_vals = [(r.first_token_time - r.submit_time) * 1000
                     for r in completed if r.first_token_time is not None]
        tpot_vals = [((r.finish_time - r.first_token_time) / r.generated_tokens) * 1000
                     for r in completed if r.finish_time is not None and r.generated_tokens > 0]
        total_vals = [(r.finish_time - r.submit_time) * 1000
                      for r in completed if r.finish_time is not None]

        lines = []
        W1, W2 = 22, 10  # label width, value width

        def row(label, value, unit=""):
            lines.append(f"  {label:<{W1}} {value:>{W2}.2f} {unit}".rstrip())

        lines.append("=== Engine Profile ===")
        lines.append(f"  {'Steps:':<{W1}} {n:>{W2}}")
        lines.append(f"  {'Total time:':<{W1}} {total_ms:>{W2}.2f} ms")

        lines.append("")
        lines.append("  Per-step averages:")
        row("admit:",     avg([st.admit_ms for st in s]),     "ms")
        row("prefill:",   avg([st.prefill_ms for st in s]),   "ms")
        row("decode:",    avg([st.decode_ms for st in s]),    "ms")

        lines.append("")
        lines.append("  Request stats:")
        lines.append(f"  {'completed:':<{W1}} {c:>{W2}}")
        row("avg TTFT:",   sum(ttft_vals) / len(ttft_vals) if ttft_vals else 0,  "ms")
        row("avg TPOT:",   sum(tpot_vals) / len(tpot_vals) if tpot_vals else 0,  "ms")
        row("avg total:",  sum(total_vals) / len(total_vals) if total_vals else 0, "ms")

        lines.append("")
        lines.append("  Scheduler stats:")
        row("avg waiting:",    avg([st.waiting_depth for st in s]))
        row("avg prefilling:", avg([st.prefilling_count for st in s]))
        row("avg running:",    avg([st.running_count for st in s]))
        row("avg budget used:", avg([st.budget_pct for st in s]), "%")

        lines.append("----------------------\n")
        print("\n".join(lines))
        
class ModuleProfiler:
    """累积模块级别的耗时统计, 用于回答 'forward pass 里时间花在哪'."""

    def __init__(self):
        self._timings: dict[str, float] = {}  # name -> total_ms
        self._counts: dict[str, int] = {}     # name -> call count

    def record(self, name: str, elapsed_ms: float):
        """累计一次计时。若 name 含 "/", 同时向上累加父级。"""
        self._timings[name] = self._timings.get(name, 0.0) + elapsed_ms
        self._counts[name] = self._counts.get(name, 0) + 1
        # 向上累加父级
        parent = name.rsplit("/", 1)[0] if "/" in name else None
        if parent:
            self._timings[parent] = self._timings.get(parent, 0.0) + elapsed_ms
            self._counts[parent] = self._counts.get(parent, 0) + 1

    def elapsed(self, start: float) -> float:
        """返回从 start 到现在的 ms。"""
        return (time.perf_counter() - start) * 1000

    def summary(self):
        """打印层次化时间分解, 含全局占比和子项占比。"""
        total = sum(t for n, t in self._timings.items() if "/" not in n)
        if total == 0:
            return

        lines = ["=== Module Profile ==="]

        # 找出所有顶层项 (不含 "/")
        top = sorted(n for n in self._timings if "/" not in n)

        for name in top:
            t = self._timings[name]
            pct = (t / total * 100) if total > 0 else 0.0
            cnt = self._counts.get(name, 0)
            has_children = any(n.startswith(name + "/") for n in self._timings)
            if has_children:
                lines.append(f"  {name + ':':20s} {t:8.2f} ms ({pct:5.1f}%)")
            else:
                lines.append(f"  {name + ':':20s} {t:8.2f} ms ({pct:5.1f}%)  [{cnt} calls]")

            # 子项 (name/xxx)
            prefix = name + "/"
            children = sorted(n for n in self._timings if n.startswith(prefix))
            for child in children:
                ct = self._timings[child]
                cpct = (ct / t * 100) if t > 0 else 0.0
                ccnt = self._counts.get(child, 0)
                label = "  └ " + child[len(prefix):]
                lines.append(f"    {label:20s} {ct:8.2f} ms ({cpct:5.1f}%)  [{ccnt} calls]")

        lines.append(f"  {'TOTAL:':20s} {total:8.2f} ms")
        lines.append("----------------------\n")
        print("\n".join(lines))

class MemoryProfiler:
    """内存快照追踪：模型权重、KV cache、PyTorch 显存。"""

    def __init__(self):
        self._snapshots: list[tuple[str, dict]] = []

    def snapshot(self, model, kv_manager, device: str) -> dict:
        """返回当前内存快照 (MB)。"""
        snap: dict[str, float] = {}

        # 模型权重 (fp32 = 4 bytes per param)
        total_params = sum(p.numel() for p in model.parameters())
        snap["weights"] = total_params * 4 / (1024 * 1024)

        # KV cache pool
        pool = kv_manager.pool
        used_blocks_count = sum(1 for b in kv_manager.blocks if not b.is_free)
        free_blocks_count = sum(1 for b in kv_manager.blocks if b.is_free)
        # 每个 block 的 tensor 大小: 2(k+v) * num_kv_heads * block_size * head_dim * 4(bytes)
        block_bytes = (2 * pool.num_kv_heads * pool.block_size * pool.head_dim * 4)
        block_mb = block_bytes / (1024 * 1024)
        snap["kv_used"] = used_blocks_count * pool.num_layers * block_mb
        snap["kv_free"] = free_blocks_count * pool.num_layers * block_mb

        # PyTorch allocator (only meaningful on CUDA)
        if device == "cuda":
            snap["torch_allocated"] = torch.cuda.memory_allocated() / (1024 * 1024)
            snap["torch_reserved"] = torch.cuda.memory_reserved() / (1024 * 1024)

        return snap

    def record(self, phase: str, snap: dict):
        """记录一个时刻的快照，phase 如 'before_prefill'。"""
        self._snapshots.append((phase, snap))

    def summary(self):
        """打印内存概况：总量分解 + KV cache 使用峰值。"""
        if not self._snapshots:
            return

        # 取最后一张快照作为稳定状态
        _, last = self._snapshots[-1]

        # KV cache 峰值
        kv_used_values = [s["kv_used"] for _, s in self._snapshots]
        kv_used_peak = max(kv_used_values)
        kv_total = last.get("kv_used", 0) + last.get("kv_free", 0)

        lines = ["=== Memory Profile ==="]
        lines.append(f"  weights:            {last.get('weights', 0):8.2f} MB")
        lines.append(f"  KV cache (total):   {kv_total:8.2f} MB")
        lines.append(f"    peak used:        {kv_used_peak:8.2f} MB")
        lines.append(f"    current used:     {last.get('kv_used', 0):8.2f} MB")
        lines.append(f"    current free:     {last.get('kv_free', 0):8.2f} MB")
        if kv_total > 0:
            lines.append(f"    utilization:      {kv_used_peak / kv_total * 100:8.1f}%")
        if "torch_allocated" in last:
            lines.append(f"  torch allocated:    {last['torch_allocated']:8.2f} MB")
            lines.append(f"  torch reserved:     {last['torch_reserved']:8.2f} MB")
        lines.append("----------------------\n")
        print("\n".join(lines))