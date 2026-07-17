from dataclasses import dataclass, field
import asyncio
from .kv_cache import BlockTable

@dataclass
class Request:
    # ========== 输入（调用时传入） ==========
    request_id: int
    prompt: str
    token_ids: list[int]           # 完整的 token 序列（prompt），用于 prefix matching
    max_new_tokens: int
    submit_time: float
    future: asyncio.Future

    # ========== 状态（运行时变化） ==========
    block_table: BlockTable | None = None

    # Prefix Cache 相关
    matched_blocks: list[int] = field(default_factory=list)  # 从 prefix cache 复用的 block
    matched_tokens: int = 0                                     # 复用的 token 数量

    # 生成相关
    generated_tokens: int = 0
    prefill_done: bool = False
    _last_token: int = 0
    _generated_token_ids: list[int] = field(default_factory=list)

    # 时间相关
    start_time: float | None = None
    first_token_time: float | None = None

    # ========== 输出（请求完成后填充） ==========
    finish_time: float | None = None
    ttft: float | None = None
    tpot: float | None = None

    @property
    def prompt_tokens(self) -> int:
        """向后兼容，废弃字段"""
        return len(self.token_ids)

    @property
    def total_tokens(self) -> int:
        """当前总 token 数量（prompt + 已生成）"""
        return len(self.token_ids) + self.generated_tokens