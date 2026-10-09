```
## 我建议八月份正式建立一个「面试知识树」

不用无限扩张，就围绕目标岗位。

**C++ / CS 基础**

C++ 对象模型、内存管理、RAII、智能指针、STL、虚函数、多态、模板、并发；OS 重点补进程线程、虚拟内存、锁。

**GPU / CUDA**

SIMT → Warp → SM → Memory Hierarchy → Coalescing → Shared Memory → Bank Conflict → Occupancy → Stream → Event → CUDA Graph → Kernel Optimization。

**LLM Inference**

Transformer → KV Cache → Prefill/Decode → MHA/MQA/GQA → RoPE → FlashAttention → Quantization → Continuous Batching → PagedAttention → Scheduling。

**Serving / vLLM**

请求生命周期 → Scheduler → Block Manager → KV Cache Management → Chunked Prefill → Prefix Caching → TP → PP → Benchmark → TTFT/TPOT/ITL/Throughput。

**Distributed / 进阶**

NCCL/HCCL → Collective Communication → AllReduce/AllGather/ReduceScatter → TP Communication → EP → MoE → DeepEP。

最后这一层不用现在 All-in，前面真正站稳以后再扩。

每天再保留 LeetCode。

这已经够你学很久了。
```
