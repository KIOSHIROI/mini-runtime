# Part 4 教学安排：CUDA Kernel 实现（Syllabus）

> 本文件是 Part 4（CUDA Kernels）的**实现与教学一体化安排**。与 Part 1–3
> 不同，本部分不是"先写教材再讲实现"，而是**边实现边写教材**：每个 kernel
> 遵循统一教学单元，实现代码直接进入 `mini_runtime/cuda_kernels/`，
> 正确性与性能由测试和基准脚本验证。

## 0. 教学理念

> **教学方式（本部分采用"教练-学员"模式）**：教练（Deep Code）负责理论讲解、
> 设计引导、代码 review 与验证标准；**所有 kernel 代码由学员（你）亲手编写**。
> 教练不直接写 kernel 代码。每个 kernel 完成后，实现经验沉淀进教材章节。

每个 kernel 的教学单元（对应教材章节的 §2–§5）：

1. **理论推导**（教练讲解 + 教材 §2）：从硬件约束推导 kernel 设计（访存模式、分块、归约）；
2. **Naive 实现 → 优化演进**（你编写）：先正确后高效，每步可运行、可对比；
3. **正确性验证**：与 PyTorch reference 对照（max abs error < 1e-4）；
4. **性能验证**：kernel 时间 vs torch reference（带宽利用率 / FLOPs）；
5. **集成**：替换 mini-runtime 模型中的对应算子（RMSNorm/RoPE/SDPA）。

**铁律**：每个 kernel 提交时必须附带测试与基准，禁止"只写代码不验证"。

## 1. 实现顺序（调整说明）

| 顺序 | Kernel | 教学核心 | 集成目标 | 为何在此位置 |
|------|--------|---------|---------|-------------|
| 0 | 编译基础设施 | cpp_extension、build 集成 | `cuda_kernels/` 目录 | 一切的前提 |
| 1 | RMSNorm | 逐元素 + 行归约 | `model/rms_norm.py` | 最简、真实算子 |
| 2 | RoPE | 位置索引 + 旋转 | `model/rotary.py` | 索引模式 |
| 3 | GEMM | shared memory tiling | 独立 benchmark | FlashAttention 的前置 |
| 4 | FlashAttention | 分块 + 在线 softmax | `model/attention.py` | 本部分顶点 |
| 5 | LayerNorm | 两遍归约 vs Welford | 对照 | RMSNorm 的对照 |
| 6 | CUDA Graph | 捕获/重放 | `backend/native.py` | 框架级收尾 |

**调整理由**：用户原目录为 GEMM→LayerNorm→RMSNorm→RoPE→FlashAttention→
CUDA Graph。调整为 RMSNorm/RoPE 先行：它们是 mini-runtime 模型真实算子、
实现风险最低，能最快建立"kernel 骨架 + 验证流水线"；GEMM 的 tiling 模式
是 FlashAttention 的直接前置（顺序调整后 GEMM 在 FlashAttention 之前）；
LayerNorm 作为 RMSNorm 对照收尾；CUDA Graph 是框架级执行优化（非算子）。

## 2. 目录结构

```
mini_runtime/cuda_kernels/
├── __init__.py            # 延迟加载扩展（import 时编译一次）
├── setup.py               # torch.utils.cpp_extension 配置（或 pyproject 集成）
├── src/
│   ├── common.cuh         # 共享工具：计时、错误检查、grid-stride 宏
│   ├── rms_norm.cu        # ch17 对应
│   ├── rope.cu            # ch19 对应
│   ├── gemm.cu            # ch16 对应
│   ├── flash_attn.cu      # ch20 对应
│   └── layer_norm.cu      # ch18 对应
├── tests/
│   ├── test_rms_norm.py   # 正确性对照
│   ├── test_rope.py
│   ├── test_gemm.py
│   ├── test_flash_attn.py
│   └── test_layer_norm.py
└── benchmarks/
    ├── bench_rms_norm.py  # 性能对照
    ├── bench_gemm.py
    └── bench_flash_attn.py
```

## 3. 每个 kernel 的交付物（DoD）

- [ ] CUDA kernel 源码（naive + 优化至少两版）
- [ ] 正确性测试：与 PyTorch reference max-abs-error 达标
- [ ] 性能基准：与 torch reference 的时间/带宽对比
- [ ] 集成（如适用）：替换 mini-runtime 对应算子 + 全链路测试通过
- [ ] 教材章节（7 段结构）更新

## 4. 环境

| 项 | 值 |
|----|----|
| GPU | NVIDIA A800 80GB PCIe（sm_80） |
| torch | 2.12.1+cu130 |
| nvcc | CUDA 12.4（用户态安装：`~/cuda-12.4/bin/nvcc`，见下文） |
| 编译方式 | `export CUDA_HOME=$HOME/cuda-12.4` + `TORCH_CUDA_ARCH_LIST=8.0` |

### 4.1 环境搭建（方案 A：用户态 CUDA Toolkit，无 root）

```bash
# 1. 下载 CUDA 12.4.1 runfile（约 3GB，只装 toolkit 不装驱动）
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run

# 2. 用户态安装到 ~/cuda-12.4（--toolkit 只装编译工具链）
sh cuda_12.4.1_550.54.15_linux.run --toolkit --silent --toolkitpath=$HOME/cuda-12.4

# 3. 验证
$HOME/cuda-12.4/bin/nvcc --version   # 应显示 release 12.4

# 4. 写入 ~/.bashrc（或每次编译前 export）
export CUDA_HOME=$HOME/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
```

> 说明：CUDA 12.x 的 nvcc 支持 `-std=c++20`（torch 2.12 默认要求），
> 因此装好后可直接使用标准 `torch.utils.cpp_extension` 流程
> （`.cu` 中可正常 include torch 头），无需"分离编译"的规避方案。

## 5. 验证基线（对照 torch reference 的预期）

| Kernel | 正确性判据 | 性能目标（A800） |
|--------|-----------|-----------------|
| RMSNorm | max\|err\| < 1e-4 | ≥ torch 原生（带宽受限） |
| RoPE | max\|err\| < 1e-4 | ≥ torch 实现 |
| GEMM | max\|err\| < 1e-3（fp32） | 与 cuBLAS 差距记录在案（不追求超越） |
| FlashAttention | max\|err\| < 1e-3 | 相对 eager SDPA 的时间收益 |
| LayerNorm | max\|err\| < 1e-4 | ≥ torch 原生 |
| CUDA Graph | 输出一致 | decode 步 launch 开销显著下降 |
