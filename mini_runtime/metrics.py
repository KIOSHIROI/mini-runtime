from dataclasses import dataclass

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