"""教学示例 01：add1 —— 你的第一个 CUDA kernel。

对应教材：Part 4 第 18 章（RMSNorm）之前的热身单元。
教学目的（Phase 0，load_inline 学习阶段）：
  1. kernel 是什么（__global__）
  2. launch 是什么（<<<blocks, threads>>>）
  3. thread/block 索引怎么算
  4. data_ptr() 如何把 PyTorch Tensor 暴露给 CUDA

运行：
    source .venv/bin/activate
    CUDA_HOME=$HOME/cuda-12.4 python tutorials/add1_demo.py
"""

from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>

// ---- device 代码：GPU 上执行的 kernel ----
// 每个线程处理一个元素。GPU 会创建 blocks*threads 个线程。
__global__ void add1_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 全局线程编号
    if (i < n) {                                     // 越界保护：n 不一定整除 threads
        y[i] = x[i] + 1.0f;
    }
}

// ---- host 代码：CPU 上执行，Python 的调用入口 ----
torch::Tensor add1(torch::Tensor x) {
    auto y = torch::empty_like(x);                   // 分配输出（不初始化）
    int n = x.numel();
    int threads = 256;                               // 每 block 256 线程（常见值）
    int blocks = (n + threads - 1) / threads;        // ceil(n/threads)：向上取整
    add1_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n); // data_ptr 取裸指针
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add1", &add1);                            // 注册给 Python
}
"""

ext = load_inline(name="add1_demo", cpp_sources="", cuda_sources=cuda_src, verbose=False)

import torch

x = torch.randn(10, device="cuda")
y = ext.add1(x)
torch.cuda.synchronize()                  # 等 GPU 执行完（验证前必须）
print("max(y - x) =", (y - x).max().item())   # 应输出 1.0
assert torch.allclose(y, x + 1, atol=1e-6)
print("✓ add1 通过")
