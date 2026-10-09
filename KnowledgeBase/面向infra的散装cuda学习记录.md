# 前言
这里是参考b站 傅莘莘(傅猪猪) 的课程，课程名字叫做"面向ai infra的cuda零基础入门/面向ai-infra的Cuda零基础教程"。从课程内容来看，主要是在实践中使用cuda去写一些算子，介绍一些相关概念。这里整理一些作者的代码，写一些自己的理解。
同时也记录一些自己学到的零散内容，防止遗忘。
## 资料
[CUDA教程](bilibili.com/video/BV1iYeozdE5j/?spm_id_from=333.1387.0.0)    |    up主 **山哥学ai**，讲的很好
[CUDATutorial](https://cuda.keter.top/) 
Flash Attention学习过程_bilibili    |    up主 **[比飞鸟贵重的多_HKL](https://space.bilibili.com/218427631/?spm_id_from=333.788.upinfo.detail.click)**，讲的很好
Flash Attention为什么那么快？原理讲解_bilibili
神经网络-量化与部署，进阶之路迟早要越过的大山_bilibili
一心二用的Online Softmax_知乎
[TinyInfiniTensor的cuda作业](https://github.com/InfiniTensor/TinyInfiniTensor)
# 个人笔记
## 面向ai infra的cuda记录
### 共享显存的理解
**共享显存是什么？**

| 存储器           | 位置        | 速度                     | 容量     | 作用                |
| ------------- | --------- | ---------------------- | ------ | ----------------- |
| Global Memory | 板卡        | 慢                      | 大（GB级） | 存放输入输出数据          |
| Shared Memory | SM（流多处理器） | 快（比 Global 快 100 倍以上！） | 小（KB级） | 一个 block 内的线程共享数据 |

**为什么用共享显存？**
1. **数据复用**：如果多个线程要访问相同数据，先加载到共享内存，大家都从这里读
2. **合并访问优化**：从 Global 读的时候用合并方式，存到 Shared，再随意访问
3. **线程间通信**：同一个 block 内的线程可以通过共享内存交换数据
**基本用法：**
```cpp
__global__ void kernel(...) {
    __shared__ float s_data[256];  // 声明共享内存
    // 1. 把 Global Memory 数据读入 Shared Memory
    int tid = threadIdx.x;
    if (tid < 256) {
        s_data[tid] = global_data[tid];
    }
    __syncthreads();  // 【重要】等待所有线程都写完
    // 2. 使用 Shared Memory
    float val = s_data[...];
    __syncthreads();  // 用完后，再次同步（可选，视情况而定）
}
```
**`__syncthreads()` 是做什么的？**
**作用：一个块内的所有线程都等在这里，直到大家都执行到这一行。**
**为什么需要？**
```
假设没有 __syncthreads()：
- 线程 0 写完 s_data[0] 就立即去读
- 线程 1 还没写完 s_data[1]
- 线程 0 读到了错误的数据（数据竞争）

有了 __syncthreads()：
- 所有线程写完后，一起继续往下执行
- 保证数据都准备好了
```
**关键点总结：**
1. `__shared__` → 声明共享内存
2. `__syncthreads()` → 同步，防止数据竞争
3. 共享内存**每个 block 独立**，不同 block 之间不共享
4. 共享内存**很小**，要省着用（比如用 tiling）
### 存储体冲突（Bank Conflict）
**什么是存储体？** 共享显存被分成多个**存储体（Bank）**，通常是 32 个（和 warp 大小一样）。
```
Bank 0: s_data[0], s_data[32], s_data[64], ...
Bank 1: s_data[1], s_data[33], s_data[65], ...
Bank 2: s_data[2], s_data[34], s_data[66], ...
...
Bank 31: s_data[31], s_data[63], s_data[95], ...
```
**为什么这样分？** 为了**并行访问**：每个时钟周期，每个 bank 可以独立读/写一个数据！
**什么是存储体冲突？** 当同一个 warp 内的多个线程，同时访问同一个 bank 的不同位置时，就会发生冲突！
```
冲突的情况（假设线程 0-31 都读 s_data[0]）：
Bank 0: [s_data[0] s_data[32] s_data[64] ...]
         ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
         线程0 线程1 线程2 ... 线程31 都想读这个 bank！

结果：串行化，性能下降！
```
**画图理解：**
```
【无冲突的情况】
Bank 0: s_data[0] <- 线程0
Bank 1: s_data[1] <- 线程1
Bank 2: s_data[2] <- 线程2
...
Bank 31: s_data[31] <- 线程31
✅ 完美并行！一个时钟周期完成！

【有冲突的情况】
Bank 0: s_data[0] <- 线程0、线程1、线程2...
Bank 1: s_data[1] <- (空闲)
Bank 2: s_data[2] <- (空闲)
...
❌ 串行化！需要多个时钟周期！
```
**常见的冲突模式和解决方法：**
如果有位移冲突或者跨步冲突，就用padding或者移位的方式打乱行间对齐，让同一列元素不再命中同一个bank。
**经典例子：转置矩阵时的存储体冲突！**
共享内存分 32 个 bank，每个 bank 每周期服务一个地址 `float tile[BDIMY][BDIMX]` 在内存中按行连续存放，同一行相邻元素的 bank 号就是 index % 32：
```
tile[0][0] → bank 0
tile[0][1] → bank 1
...
tile[0][31] → bank 31
tile[ty][tx] → bank (ty * BDIMX + tx) % 32   ←    BDIMX = 32
```
转置时从 tile 读出是按列读：`tile[icol][irow]`，即同一 warp 的相邻线程读同一列的不同行：
```
tid=0 读 tile[0][0] → bank 0
tid=1 读 tile[1][0] → bank (1 * 32 + 0) % 32 = bank 0
tid=2 读 tile[2][0] → bank (2 * 32 + 0) % 32 = bank 0
...
32 个线程全访问 bank 0，串行化成 32 次访问，这就是 32-way bank conflict。
```
Pad 怎么消掉的呢?给列维度加上1就行：
```cpp
__shared__ float tile[BDIMY][BDIMX + 1];  // +1 就是 pad
```
加了 1 之后，每行多了一个元素，下一行的起始地址偏移了 33 而不是 32：
```
tile[0][0] → bank 0
tile[1][0] → bank (1 * 33 + 0) % 32 = bank 1
tile[2][0] → bank (2 * 33 + 0) % 32 = bank 2
  ...
```
同一列不同行现在落在不同 bank 了，冲突消失。代价就是每行浪费 4 字节（1 个 float），对于 16×32 的 tile 来说微不足道。
>所以核心就一句话：pad 打乱了行间对齐，让同列元素不再命中同一个 bank。

### 归约的寄存器实现
`__shfl_down_sync()`是 warp 内的寄存器直传指令，不需要共享内存，warp 内 32 个线程可以直接互相读取对方寄存器里的值。
**基本用法**
```cpp
float val = __shfl_down_sync(mask, var, offset);
```
- mask：哪些线程参与，0xFFFFFFFF 表示 warp 内 32 个线程全部参与
- var：要读取的变量
- offset：从哪个线程读——读的是 当前线程号 + offset 那个线程的 var
**用归约求和举例**
```cpp
// 假设 warp 内 32 个线程，每个线程有一个 val
val += __shfl_down_sync(0xFFFFFFFF, val, 16);
val += __shfl_down_sync(0xFFFFFFFF, val, 8);
val += __shfl_down_sync(0xFFFFFFFF, val, 4);
val += __shfl_down_sync(0xFFFFFFFF, val, 2);
val += __shfl_down_sync(0xFFFFFFFF, val, 1);
```
一步步看（用 8 个线程简化，offset 走 4→2→1）：
```
初始:    t0=a  t1=b  t2=c  t3=d  t4=e  t5=f  t6=g  t7=h

offset=4:
  t0 读 t4 的值: val = a + e
  t1 读 t5 的值: val = b + f
  t2 读 t6 的值: val = c + g
  t3 读 t7 的值: val = d + h
  t4~t7: 超出范围，值无意义（但不影响）

offset=2:
  t0 读 t2 的值: val = (a+e) + (c+g)
  t1 读 t3 的值: val = (b+f) + (d+h)
  t2~t7: 无意义

offset=1:
  t0 读 t1 的值: val = (a+e+c+g) + (b+f+d+h) = 总和
```
最终 t0 拿到了全 warp 的求和结果，5 步完成，全程在寄存器里操作，零共享内存开销，零同步开销（warp 内天然同步）。
**对比共享内存归约**

|               | 共享内存归约             | warp shuffle 归约 |
| ------------- | ------------------ | --------------- |
| 存储位置          | shared memory      | 寄存器             |
| 同步            | 需要 __syncthreads() | 不需要（warp 内天然同步） |
| bank conflict | 可能有                | 没有              |
| 速度            | 快                  | 更快              |

warp 内用 shuffle 归约，warp 间用共享内存归约。这是标准做法。因为warp 内用 shuffle 最快，跨 warp 还得靠共享内存通信。

### 协作组 Cooperative Groups
可以理解为 CUDA 对“线程协作范围”的显式抽象。以前很多协作默认发生在整个 block 或一个 warp 内，现在可以先把线程组织成某个 `group`，再在这个 `group` 内做同步、shuffle、reduce 等 collective 操作。
**先理解一下原本的问题** 以前最常见的同步原语是：
```cpp
__syncthreads();
```
它的语义很硬：同步范围就是**整个 block**。但现实里很多算法真正需要的协作范围其实更小，比如：
- warp 内同步
- 一个 16-thread tile 内同步
- 一个 block 被切成多个小组后，每组各自协作
如果还是强行用整个 block 同步，就会有几个问题：
- 同步范围过大，不必要地让更多线程等待
- 代码里把 warp/block 假设写死，抽象性差
- 辅助函数如果隐含要求“必须整块线程一起进来”，更容易写出难维护甚至死锁的代码
例如以前常见写法里，经常把协作范围硬编码在代码里：
```cpp
__syncthreads();

#define WARP_SIZE 32
val = __shfl_down_sync(0xffffffff, val, offset);
```
**核心理解**
Cooperative Groups 的核心不是“提供一个更花哨的同步 API”，而是把“哪些线程要一起协作”这件事显式写进代码。
也就是先定义 group，再在 group 内协作：
```cpp
group.sync();
```
所以它的价值不只是替代 `__syncthreads()`，而是让线程协作域变成代码里的一个一等概念。
一般常用的 group 有：

| group                   | 含义                                           |
| ----------------------- | -------------------------------------------- |
| `thread_block`          | 当前整个 block                                   |
| `thread_block_tile<32>` | 一个大小为 32 的 tile，很多时候对应一个 warp                |
| `tiled_partition`       | 把一个更大的 group 切成固定大小的小 tile                   |
| `grid_group`            | 整个 grid，但要求 kernel 以 cooperative launch 方式启动 |

最常见的用法：
```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

cg::thread_block block = cg::this_thread_block();
block.sync();  // 等价于 __syncthreads()

auto tile = cg::tiled_partition<16>(block);  // 把 block 切成多个 16-thread tile
tile.sync();  // 只同步当前 tile 内的线程
```
这里有两个理解上的细节：
- `tiled_partition<16>(block)` 不是“随便取前 16 个线程”，而是把整个 block 按规则切成多个 16-thread 小组，每个线程只属于自己的那个 tile。
- `thread_block_tile<32>` 常常可以当作 warp 级协作来理解，但概念上它首先是“32 个线程组成的 tile”，不要完全和硬件 warp 混为一谈。
**一般在什么场景下用**
- block 内再切小组，做分块归约、scan、局部交换
- 写可复用的 device 函数时，把 group 作为参数传进去，显式声明“哪些线程必须一起参与”
- 需要 warp/tile 级 collective 操作，而不想把 warp 大小、掩码、同步语义硬编码在实现里
所以从 infra 视角可以把它记成一句话：
> Cooperative Groups = 把线程协作域显式化，让同步和通信发生在正确的那组线程里，而不是默认粗暴地作用于整个 block。

这里只是先记核心理解，后面如果写 block 内分组归约、warp/tile 级 scan、或者 grid 级同步，再回头看会更顺。
### 线程块簇 Thread Block Clusters
可以把 cluster 理解成“多个 block 组成的协作小队”。它位于 `block` 和 `grid` 之间，是 Hopper 这一代 CUDA 新增的一层执行/协作层级。
层次结构可以记成：
```text
Grid（整个 kernel）
  └─ Cluster（多个 block 的组）
      └─ Block（多个线程）
          └─ Warp（32 个线程）
              └─ Thread（单个线程）
```
从 infra 视角，最核心的一句是：以前 block 往往被看成彼此独立；有了 cluster 之后，可以让多个 block 组成一个更大的协作单元。
**先理解没有 cluster 时的默认世界**
传统 CUDA 里，最自然的协作单位是 block：
- 一个 block 内的线程可以 `__syncthreads()`
- 一个 block 内的线程可以共享 `shared memory`
- block 和 block 之间默认是相互独立的
所以传统写法通常是：
```cpp
kernel<<<100, 256>>>();  // 100 个 block，默认彼此独立
```
这不是说 block 之间绝对不能配合，而是说它们没有天然的、便宜的、结构化的协作机制。跨 block 协作往往要借助：
- global memory
- atomic
- 多个 kernel 分阶段执行
例如一个 block 写数据，另一个 block 读数据，常见套路是：
```cpp
// Block 0 写
global_buffer[idx] = data;
__threadfence();
atomicAdd(&flag, 1);

// Block 1 读（需要等待）
while (atomicAdd(&flag, 0) < expected) { /* spin */ }
int value = global_buffer[idx];
```
这种方式当然能工作，但问题也很明显：
- 通信要绕 global memory
- 需要额外的原子操作和可见性控制
- 代码比较绕，不像 block 内协作那样自然
**为什么需要 cluster**
cluster 的目标就是：让多个 block 不再只是“松散地属于同一个 grid”，而是组成一个更紧密的协作单元。
可以把它想成：
- `block` 是一个班组
- `cluster` 是多个班组组成的小队
- `grid` 是整次 kernel 启动的全部班组
于是 cluster 这一层带来了两个重要变化：
1. 协作范围从单个 block 扩大到多个 block  
2. 后续可以在 cluster 这一层做同步和数据共享
所以 cluster 是一个“中间层”。它比 block 大，能做更多协作；又比整个 grid 小，限制更明确，也更容易高效实现。
**怎么声明和使用**
一个 kernel 可以声明 cluster 大小，例如每个 cluster 包含 4 个 block：
```cpp
__global__ void __cluster_dims__(4, 1, 1)
cluster_kernel(...) {
    // ...
}
```
这表示这个 kernel 的 block 会按 `4 x 1 x 1` 的 cluster 组织。直觉上可以理解成：
- grid 里仍然有很多 block
- 但这些 block 不再只是平铺
- 而是每 4 个 block 组成一个 cluster
如果总共有 100 个 block，cluster size 是 4，那么可以粗略理解成：
- 一共有 25 个 cluster
- 每个 cluster 里有 4 个 block
这里先记直觉就够了。真正写代码时，还要注意 grid/block/cluster 配置是否满足硬件和 runtime 的要求。
**cluster 和 Cooperative Groups 的关系**
前面说 Cooperative Groups 是“把线程协作域显式化”。cluster 正好就是其中一个更大的协作域。
在 kernel 里可以这样拿到当前 cluster：
```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void __cluster_dims__(4, 1, 1)
cluster_kernel(...) {
    auto cluster = cg::this_cluster();
    cluster.sync();  // 同步当前 cluster 内所有 block
}
```
所以可以把它们的关系记成：
- Cooperative Groups 提供了“group”这套抽象
- cluster 是其中一种更大的 group
- `cg::this_cluster()` 让你显式拿到“当前多个 block 组成的协作组”
**它和 DSM (分布式共享内存)的关系**
- cluster 解决的是“多个 block 作为一组协作”
- DSM 解决的是“这组 block 之间的数据怎么共享”
所以 DSM 是 cluster 上面的进一步能力，但你现在先把 cluster 理解成“block 之上的协作层级”就够了。
### CUDA Stream
**先说一下场景：** 传统的单任务执行时：
```
时间轴：
───────────────────────────────────────────────────────────►
CPU:  [准备数据A]                                             [处理结果]        [空闲]
                  ↓                                     ↑
GPU:                       [传输A]  [计算A]  [传回A]     [空闲GPU！]
                   └──-──串行执行─────┘

问题1：GPU传输数据时，计算单元空闲 ❌
问题2：GPU计算时，PCIe通道空闲 ❌
问题3：CPU等待GPU时，不能准备下一批数据 ❌
```
而最理想的执行情况是：
```
时间轴：
─────────────────────────────────────────────────►
Stream 0:  [传A0] [算A0] [回A0]
Stream 1:             [传A1]  [算A1] [回A1]
Stream 2:                        [传A2] [算A2] [回A2]
Stream 3:                                   [传A3] [算A3] [回A3]
           └──────────重叠！并发！──────────┘

优势：
✓ 传输和计算重叠（硬件并发）
✓ 总时间 ≈ max(传输, 计算) 而非 sum(传输, 计算)
✓ GPU和PCIe都充分利用
```
**核心概念理解** 可以把 `stream` 理解成 GPU 上的一条**命令队列**。CPU 负责把任务提交到 stream 里，比如：
- kernel launch
- `cudaMemcpyAsync`
- `cudaMemsetAsync`
- event record
GPU 会按 stream 中的顺序去执行这些命令。从 infra 视角，最值得先记住的一句话是：`grid/block/thread` 描述的是一个 kernel 内部线程如何组织；`stream` 描述的是多个 GPU 操作在时间上如何排队和依赖。
对于 stream 最核心的两条语义是：
1. 同一个 stream 内，操作严格按顺序执行  
2. 不同 stream 之间，默认没有顺序关系，可以并发执行
**为什么需要 stream**
如果只用一条默认队列，所有操作都只能顺序排队：
```cpp
kernelA<<<...>>>();
kernelB<<<...>>>();
cudaMemcpy(...);
```
这样写当然简单，但很多原本可以并行的工作被串起来了。有了多个 stream 之后，就可以把独立任务拆到不同队列里：
- 一个 stream 跑 kernel A
- 一个 stream 跑 kernel B
- 一个 stream 负责搬数据
这样 GPU 就有机会把 copy 和 compute 重叠起来，或者让多个互不依赖的 kernel 部分并发。
所以 stream 的本质价值是：表达依赖关系；暴露并行性；让 copy / compute / 多个 kernel 尽量 overlap
**默认 stream 怎么理解**
最常见的 kernel launch 写法是：
```cpp
my_kernel<<<grid, block>>>();
```
如果没有显式指定 stream，这个操作通常就会进入默认 stream。所以要记住：
- 不写 stream，不代表没有 stream
- 只是你把工作提交到了默认那条队列里
默认 stream 有一些历史语义差异，比如 legacy default stream、per-thread default stream。但当前阶段先不用深挖，先把它当成“默认使用的那条命令队列”就够了。
**kernel launch 为什么说是异步的**
CUDA 里经常说 kernel launch 是异步的，这里的“异步”是相对于 CPU 来说的。
```cpp
my_kernel<<<grid, block>>>();
printf("CPU 继续执行\\n");
```
它的意思不是 kernel 不执行，而是：
- CPU 把 kernel 提交到 stream
- 提交成功后 CPU 立刻往下走
- GPU 之后再按队列顺序慢慢执行
所以经常会看到主机端结尾补一句：
```cpp
cudaDeviceSynchronize();
```
这是因为如果不同步，CPU 可能已经结束甚至程序退出了，GPU 还在跑。
**stream 内顺序执行**
例如：
```cpp
cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream1);
kernelA<<<grid, block, 0, stream1>>>();
kernelB<<<grid, block, 0, stream1>>>();
```
在 `stream1` 里，这三个操作一定按顺序发生：
1. `cudaMemcpyAsync`
2. `kernelA`
3. `kernelB`
所以单个 stream 可以近似理解成一条串行队列。
**不同 stream 之间默认无依赖**
```cpp
kernelA<<<grid, block, 0, stream1>>>();
kernelB<<<grid, block, 0, stream2>>>();
```
这两个 kernel 提交到了不同 stream，所以默认没有先后约束。它们可能：
- 真并行
- 部分重叠
- 最后还是串行
关键点在于：不同 stream 只是给了 GPU 并发的机会，不保证一定并发。真正能不能并发，还要看：
- SM 资源够不够
- kernel 占用了多少寄存器和 shared memory
- copy engine 是否空闲
- kernel 本身是不是太大，已经把机器吃满了
**最常见的用途**
- 用 `cudaMemcpyAsync` 和 kernel overlap，做 copy-compute overlap
- 把多个互不依赖的任务放到不同 stream，提高吞吐
- 一条 stream 表达一条任务链，避免手动加很多同步
- 用 event 在不同 stream 之间建立依赖
**同步怎么理解**
stream 学习里最关键的一点，是把“异步提交”和“显式同步”一起看。
常见同步方式：
1. `cudaDeviceSynchronize()`  
等设备上所有之前提交的工作都做完。最粗暴，也最容易把并行性打碎。
2. `cudaStreamSynchronize(stream)`  
只等某一条 stream 做完。比 device 级同步更细粒度。
3. `cudaEvent_t`  
最灵活。可以在一个 stream 上打点，再让另一个 stream 等它。
例如：
```cpp
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event, 0);
```
这表示：
- `stream1` 先跑
- 跑到 `event` 这里时打一个完成标记
- `stream2` 等这个标记出现后再继续
它本质上是在不同命令队列之间建立依赖关系。
**一个很常见的误区**
很多初学者会写出这种代码：
```cpp
kernelA<<<grid, block, 0, stream1>>>();
cudaDeviceSynchronize();
kernelB<<<grid, block, 0, stream2>>>();
cudaDeviceSynchronize();
```
虽然代码里写了两个 stream，但每一步都全局同步，GPU 的流水线和重叠机会基本被打没了。
所以 stream 的重点不是“创建多个 stream”，而是：尽量只在真正需要的时候同步。
**一个最小使用例子**
```cpp
cudaStream_t s1, s2;
// 创建stream
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);
// 在stream中执行操作
kernelA<<<grid, block, 0, s1>>>();
kernelB<<<grid, block, 0, s2>>>();

cudaStreamSynchronize(s1);
cudaStreamSynchronize(s2);

cudaStreamDestroy(s1);
cudaStreamDestroy(s2);
```
这段代码表达的是：
- 创建两条独立的 stream
- `kernelA` 进入 `s1`
- `kernelB` 进入 `s2`
- 它们之间默认无依赖
- 最后分别等待完成
从 infra 视角看，`stream` 最核心的几点是：它本质上是 GPU 的命令队列，同一条 stream 内的操作严格有序，不同 stream 之间默认没有依赖，因此有机会并发但不保证一定并发；同时 kernel launch 对 CPU 来说默认是异步的，所以如果随手加 `cudaDeviceSynchronize()`，很容易把原本可以重叠的流水线打碎。后面学 `cudaMemcpyAsync`、pinned memory、overlap、event timing、PyTorch current stream 和 CUDA Graph capture 时，都会不断碰到 stream，所以这层执行模型值得尽早建立稳。
**异步操作 vs 同步操作**
这里的“同步/异步”首先要明确：**通常是相对于 CPU(host) 来说的**。
**同步**（以 `cudaMemcpy` 为代表）：
```cuda
cudaMemcpy(...);  // CPU等待传输完成才继续
```
- CPU阻塞，等待操作完成
- 安全、简单，但慢
**异步**（以 `cudaMemcpyAsync`、kernel launch 为代表）：
```cuda
cudaMemcpyAsync(..., stream);  // CPU立即返回
```
- CPU立即返回，操作在后台执行
- 高效，但需要正确管理依赖
不过这里有个很重要的细节：`Async` 这个后缀不等于“任何情况下都一定完全异步”。NVIDIA 官方文档明确说过，这个名字有点误导，实际行为还会受到内存类型和参数的影响。尤其是 host 和 device 之间拷贝时，如果 host 端用的是普通 pageable memory，runtime 可能先把数据搬到临时的 pinned buffer，再发起 DMA，这期间可能发生额外同步，导致 `cudaMemcpyAsync` 看起来没有真正异步。
**Pinned Memory（页锁定内存）**

| 内存类型                     | 普通内存 (malloc) | Pinned Memory (cudaMallocHost) |
| ------------------------ | ------------- | ------------------------------ |
| 物理位置                     | 可被OS换页到磁盘     | 锁定在RAM                         |
| GPU直接访问                  | ❌ 需要中间缓冲      | ✅ DMA直接访问                      |
| 传输带宽                     | 一般更低且更不稳定     | 一般更高，也更稳定                      |
| **`cudaMemcpyAsync` 效果** | 可能退化，未必真异步    | 更容易实现真正异步和 overlap             |

**关键点**：如果想稳定实现 host-device 异步传输，或者想让数据传输和 kernel 计算真正重叠，主机端缓冲区最好使用 Pinned Memory。
```cuda
// 用普通 pageable host memory
float *h_data = (float*)malloc(size);
cudaMemcpyAsync(d_data, h_data, size, ..., stream);
// 代码通常不会报错，但 runtime 可能先做 staging，
// 导致它在 host 侧或 stream 语义上没有你想象中那么“异步”

// 用 pinned host memory
float *h_data;
cudaMallocHost(&h_data, size);  // Pinned Memory
cudaMemcpyAsync(d_data, h_data, size, ..., stream);
// 这种情况下更容易走 DMA，真正发挥 Async + stream overlap 的价值
```
---
**计算与传输重叠的原理**
需要了解硬件基础：
```
┌──────────────────────────────────────┐
│                                  GPU芯片                                     │
│  ┌─────────────┐       ┌──────────────┐      │
│  │       DMA引擎        │       │      SM计算单元        │      │
│  │         (传输)           │       │          (计算)            │      │
│  └─────────────┘       └──────────────┘      │
│                 ↕                                     ↕                         │
│            PCIe总线                          GPU显存                   │
└──────────────────────────────────────┘
关键：DMA引擎 和 计算单元 是 独立硬件！
```
**重叠条件**：
1. 使用不同的stream ✓
2. 操作使用不同硬件资源 ✓
3. 数据无依赖关系 ✓
**时间线对比**：
```
单Stream（串行）：
───────────────────────────────────────────────────►
  [H2D_0] [Kernel_0] [D2H_0] [H2D_1] [Kernel_1] [D2H_1]
  └────────────── 总时间 = 6单位 ─────────────┘

多Stream（并发）：
───────────────────────────────────────────────────►
  [H2D_0][Kernel_0][D2H_0]
     [H2D_1][Kernel_1][D2H_1]
        [H2D_2][Kernel_2][D2H_2]
  └───── 总时间 = 3-4单位 ─────┘

加速比 = 6 / 3.5 ≈ 1.7x
```
### CUDA Graph
先说一下Stream的局限性：重复地管理开销。虽然stream实现了overlap，但是每次执行时kernel launch都需要CPU-GPU交互；Driver需要逐个解析、验证、调度任务；对于固定的计算图，结构每次都一样，却要重复提交。
```cpp
// 深度学习训练的典型场景：重复1000次iteration
for (int iter = 0; iter < 1000; iter++) {
    cudaMemcpyAsync(..., stream);  // ← CPU又提交一次任务（几微秒级）
    kernel<<<..., stream>>>();     // ← kernel launch（几微秒到十几微秒级）
    cudaMemcpyAsync(..., stream);  // ← 又是一次提交（几微秒级）
}
// 如果每轮都有很多这种小操作，累计起来会变成明显的 CPU 调度开销
```
**那cuda graph是干什么的？**
CUDA Graph 可以理解成：**把一整串固定的 GPU 工作流程提前录下来，后面一次性重放**。
这里的“工作流程”可以包括：kernel launch；`cudaMemcpyAsync`；`cudaMemsetAsync`；event / wait……
所以它不是只录一个 kernel，而是录“这一整轮 GPU 要干的事，以及它们之间的依赖关系”。
比如：
```cpp
// 1. 捕获：记录你要的"套餐内容"
cudaStreamBeginCapture(stream, ...);
    kernel1<<<..., stream>>>();              // 这些操作提交到同一条 stream 上并被记录
    cudaMemcpyAsync(..., stream);
    kernel2<<<..., stream>>>();
cudaStreamEndCapture(stream, &graph);

// 2. 实例化：餐厅优化你的套餐（比如提前备菜）
cudaGraphInstantiate(&instance, graph, ...);

// 3. 重放：每天一键下单，自动配送
for (int i = 0; i < 1000; i++) {
    cudaGraphLaunch(instance, stream);  // 一次性提交整个图！
}
```
**为什么会有 CUDA Graph？**
因为 GPU 算得很快，但 CPU 一次次发命令也要花时间。普通 CUDA 执行时，CPU 往往在做这样的事：
```cpp
kernel1<<<...>>>();
kernel2<<<...>>>();
cudaMemcpyAsync(...);
kernel3<<<...>>>();
```
如果每个 kernel 都很小，那 GPU 真正计算的时间可能没多少，反而是 CPU 不断调用 runtime、不断 launch kernel 的开销变得很明显。这个问题在小 batch、decode、多小算子的场景里尤其常见。
所以 CUDA Graph 的核心目标不是“让某个 kernel 算得更快”，而是：
- 减少 CPU launch overhead
- 减少重复的调度和依赖管理开销
- 让 GPU 更容易被持续喂饱
可以把普通模式和 graph 模式这样对比：
```text
普通模式：
CPU 每次都手动发 // 每次演出都临场一个个叫演员上台
A -> B -> C -> D

CUDA Graph 模式：
先把 A -> B -> C -> D 录成一张图 // 先排好节目单
后面每次直接一键重放整张图 // 后面直接整场开演
```
**Graph 里到底装了什么？**
Graph 本质上是一张“有依赖关系的操作图”：
- 节点（node）= 一次 GPU 操作
- 边（edge）= 哪个操作必须等哪个操作完成
例如：
```text
H2D copy -> kernel1 -> kernel2 -> D2H copy
```
这就是一张很简单的 graph。
所以后面如果看到 `graph capture`、`graph replay` 这些词，不要想复杂，它们本质上就是：
- capture：把流程录下来
- replay：把录好的流程重放一遍
**它一般怎么工作？** 可以先记成三步：
1. capture / build：先把一串 GPU 操作录成图  
2. instantiate：把这张图变成一个可执行对象  
3. replay / launch：之后反复执行这张图
这个过程的直觉是：
- 第一次更像“搭模板”
- 后面每次都像“套模板”
所以 CUDA Graph 通常不是第一次就赚，而是**重复执行很多次之后开始值回票价**。
**为什么它和 stream 关系很深？**
前面学 stream 时说过，stream 是 GPU 的命令队列。平时我们是不断往 stream 里塞命令；而 CUDA Graph 经常是先在 stream 上把这一串命令“录下来”，以后再整包提交。
这也是为什么很多资料里会讲 `stream capture`。它的意思不是换了一套完全不同的 CUDA，而是：代码表面上还是像平时一样往 stream 里提交工作，但底层不是立刻按普通方式执行，而是在录一张 graph。
**什么时候它最有价值？**
CUDA Graph 最适合这种场景：
- 每一轮执行流程基本固定
- 每次 kernel 的拓扑关系差不多
- kernel 数量多，而且很多都比较小
- CPU launch overhead 已经明显
典型场景：LLM decode；固定 shape 的推理；bucket 化之后形状比较稳定的推理服务；一轮 step 结构几乎不变的 workload
**什么时候不太适合？**
如果每次执行都很不一样，CUDA Graph 就不那么舒服了。例如：
- shape 经常变化
- 控制流经常变化
- 每轮执行拓扑都不一样
- 中间有很多 CPU 参与决策


## leetGPU刷题记录
有些题实在过于简单，不如直接去[LeetGPU官网](https://leetgpu.com/challengesv)查看，这里只记录一些有代表性的进行学习和分析。
### Host 端代码
在做题的时候，主函数是一个host侧的代码，负责调度GPU使用。这里用第一题举例。
```cpp
// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
```
`extern "C"` 的作用是：**告诉 C++ 编译器不要改函数名**。
LeetGPU 这类平台会从外部找一个叫 `solve` 的函数。如果不用 `extern "C"`，C++ 编译器可能把函数名改成类似 `_Z5solvePKfS0_Pfi` 这种mangled name，平台就找不到。所以它主要是为了 **OJ/测试平台能正确调用你的函数**。
而`const float* A, const float* B, float* C, int N`的意思是，注释已经说了：A、B、C 已经在 GPU 显存里所以这里不需要：
```cpp
cudaMalloc  
cudaMemcpy
```
平台已经准备好了输入输出显存。
```
A  // GPU上的输入数组A
B  // GPU上的输入数组B
C  // GPU上的输出数组C
N  // 数组长度
```
`const float* A` 和 `const float* B` 表示kernel只读它们。`float* C` 表示要把结果写进去。
`int threadsPerBlock = 256;`这里设置：每个block开256个thread，也就是每个block负责256个元素。
为什么常用 256？
因为 GPU 的线程调度基本单位是 **warp**，一个warp通常是 32 个线程。
```
256 = 8 * 32
```
所以 256 是比较常见的经验值，既不会太小浪费调度资源，也不会太大导致资源压力太高。
`int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;`是在算：需要多少个 block 才能覆盖 N 个元素。本质是向上取整：N // threadsPerBlock
但是整数除法会向下取整，所以用这个写法：
```
(N + threadsPerBlock - 1) / threadsPerBlock
```
如果：
```
N = 1000
threadsPerBlock = 256
```
那么：
```
blocksPerGrid = (1000 + 256 - 1) / 256              
			   = 1255 / 256
	           = 4
```
总线程数：
```
4 * 256 = 1024
```
多出来 24 个线程，所以使用的时候需要在kernel里写`if (i < N)`防止越界访问。
`vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);`是 CUDA kernel 启动语法。普通 C++ 函数调用是：
```
func(args);
```
CUDA kernel 调用是：
```
kernel<<<grid_size, block_size>>>(args);
```
这里：
```
vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
```
意思是：启动 blocksPerGrid个block，每个 block有threadsPerBlock个thread，所有线程一起执行vector_add。
比如 `N = 1000`：
```
blocksPerGrid = 4
threadsPerBlock = 256
总线程数 = 1024
```
每个线程都会进入：
```
__global__ void vector_add(...)
```
然后根据自己的：
```
int i = blockDim.x * blockIdx.x + threadIdx.x;
```
找到自己负责的元素。
`cudaDeviceSynchronize();`的原因在于CUDA kernel启动默认是**异步的**。也就是说CPU执行到：
```
vector_add<<<...>>>(...);
```
之后，不会等GPU算完，CPU会继续往下走。所以这里写：
```
cudaDeviceSynchronize();
```
意思是：CPU等待 GPU把前面的kernel执行完
在 OJ 里通常需要同步，确保 `solve` 返回时，`C` 已经写好了。
### Vector Addition
#### 题目
````col
```col-md
flexGrow=1
===
Write a GPU program that performs element-wise addition of two vectors containing 32-bit floating point numbers. The program should take two input vectors of equal length and produce a single output vector containing their sum.
**Implementation Requirements**
- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The final result must be stored in vector `C`
```
```col-md
flexGrow=1
===
编写一个GPU程序，用于执行两个包含32位浮点数的向量的按元素相加操作。该程序应接受两个长度相等的输入向量，并生成一个包含它们之和的输出向量。
**实施要求**
不允许使用外部库
solve函数的签名必须保持不变
最终结果必须存储在向量C中
```
````
#### Example
```
Input:  A = [1.0, 2.0, 3.0, 4.0]
        B = [5.0, 6.0, 7.0, 8.0]
Output: C = [6.0, 8.0, 10.0, 12.0]
```
#### 解法
##### 一维 Naive
先看代码：
```cpp
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```
---
先从 CPU 写法理解。如果是 CPU，应该：
```cpp
for (int i = 0; i < N; i++) {
    C[i] = A[i] + B[i];
}
```
但是GPU的思路完全不同。GPU 会说："我有很多线程，我们一起干！"。于是变成：**每个线程负责一个i**。

| 变量            | 含义                                               |
| ------------- | ------------------------------------------------ |
| `threadIdx.x` | 当前线程在 block 里的编号（0, 1, 2..., blockDim.x-1）       |
| `blockIdx.x`  | 当前 block 在 grid 里的编号（0, 1, 2..., blockPerGrid-1） |
| `blockDim.x`  | 每个 block 有多少线程                                   |
假设`blockDim.x = 4`，`blocksPerGrid = 3`，那么就可以简化为下述分配：

| Block 0     | Block 1     | Block 2       |
| ----------- | ----------- | ------------- |
| `i=0,1,2,3` | `i=4,5,6,7` | `i=8,9,10,11` |
此时的`int i = blockIdx.x * blockDim.x + threadIdx.x;`可以理解为，第几个block × 每个block多少线程 + 当前线程编号，赋值之后每个线程拿到全局唯一编号，知道自己该处理哪份数据。而`if (i < N)` 是必须的，因为总线程数会 ≥ N，最后一个 block 可能有多余线程需要跳过，我们只需要数量少于 N 的线程来工作即可，所以要限制 i 的大小。设置好之后，就会分配固定数量的线程来执行`C[i] = A[i] + B[i]`的操作。
> 用一句chatgpt给我的话，上述方式算的id，本质不是为了在意这些线程的编号，而是为了分配好哪些线程该处理哪些数据。线程编号只是一个中映射方式。
##### 一维 Grid-Stride Loops
为什么用 Grid-Stride？普通的写法：
```cpp
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}
```
这种写法要求 grid 和 block 的配置刚好能覆盖 N，灵活性差。
而Grid-Stride 写法：
```cpp
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (; i < N; i += blockDim.x * gridDim.x) {
        C[i] = A[i] + B[i];
    }
}
```
**举个具体例子理解！** 假设：
N = 16 总共有 16 个元素； blockDim.x = 4（每个 block 4 个线程）; gridDim.x = 2（2 个 block）； 总线程数 = 2 * 4 = 8
**一对一写法**：线程不够！只启动了 8 个线程，但有 16 个元素，处理不完！此时grid-stride loop就可以让每个线程处理多个元素！
```
线程 0 处理：0, 8
线程 1 处理：1, 9
线程 2 处理：2, 10
线程 3 处理：3, 11
线程 4 处理：4, 12
线程 5 处理：5, 13
线程 6 处理：6, 14
线程 7 处理：7, 15
gridDim.x * blockDim.x = 8，这就是 stride（跨步）
	图示：
	元素： 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
	       ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
	       ↑  ↑ ↑ ↑  ↑ ↑ ↑  ↑ ↑  ↑ ↑ ↑  ↑ ↑ ↑ ↑
	       │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
	线程： 0 1 2 3 4 5 6 7 0 1  2   3  4  5   6  7
	       \__________/  \__________/
	        第一次循环            第二次循环
```
所以用 Grid-Stride 的方式，可以启动更少的线程做更多的事，减少调度开销。且同一个 warp 的线程访问的地址依然是连续的。
### Matrix Addition
#### Example
```
Input:  A = [[1.0, 2.0],
             [3.0, 4.0]]
        B = [[5.0, 6.0],
             [7.0, 8.0]]
Output: C = [[6.0, 8.0],
             [10.0, 12.0]]
```
#### 解法
顺带趁火打劫看一下矩阵加，类似于向量加中的一维Naive和一维Grid-Stride loops，代码都可以原封不动的拿来用，只需要把N改成N\*N即可。
```cpp
// 一维 Naive
__global__ void add_naive(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <  N * N) {
        C[i] = A[i] + B[i];
    }
}
// 一维 Grid-Stride loops
__global__ void add_grid_stride_scalar(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for ( ; i < N * N; i += gridDim.x * blockDim.x;) {
        C[i] = A[i] + B[i];
    }
}
```
##### 二维 Grid-Stride
```cpp
// 二维 Grid-Stride loops
__global__ void matrix_add(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int r = row; r < N; r += gridDim.y * blockDim.y) {
        for (int c = col; c < N; c += gridDim.x * blockDim.x) {
            C[r * N + c] = A[r * N + c] + B[r * N + c];
        }
    }
}
```
对于图像处理、矩阵分块的情况，用这种二维跨步的方式来做很直观；后续计算如果改动二维矩阵布局的时候，二维映射能减少索引计算复杂度；但实际上对于内层循环的`c`可以保持内存访问的连续性，对于外层的`r`就会让线程跳跃整行，此时访问变成了大跨步，而行主序的存储，跨行访问并不破坏连续性，所以这种方式性能与一维Grid-Stride大致持平。而多数高性能基础算子库内部依然采用一维Grid+Stride+向量化，因为指令开销最小。
### Matrix Multiplication
这里注意一下当前这个case中，$A$的维度是$M\times K$，B的维度是$K \times N$，输出C的维度$M\times N$。有个讲的非常好的博客：[siboehm的博客](https://siboehm.com/articles/22/CUDA-MMM)，对矩阵乘做了很多深入浅出的优化。然后再补一句，gemm指的是：general matrix multiplication，这是BLAS命名规则，一个线性代数标准库的运算接口。
#### Naive
```cpp
__global__ void sgemm_naive(int M, int N, int K, const float* A, const float* B,
                                                float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < K) {
        float tmp = 0.0f;
        for (int n = 0; n < N; n++) {
            tmp += A[row * N + n] * B[n * K + col];
        }
        C[row * K + col] = tmp;
    }
}
```
直接手画一下图就知道这里的row、col和k是怎么运转的。
#### Global Memory Coalescing
```cpp
#define BLOCK_SIZE 32
__global__ void sgemm_globalMemoryCoalescing(int M, int N, int K, const float* A, const float* B,
                                                float* C) {
    int row = blockIdx.y * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    int col = blockIdx.x * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);
    if (row < M && col < K) {
        float tmp = 0.0f;
        for (int n = 0; n < N; n++) {
            tmp += A[row * N + n] * B[n * K + col];
        }
        C[row * K + col] = tmp;
    }
}
```
#### Shared Memory Cache-Blocking
```cpp
__global__ void sgemm_sharedMemoryCacheBlocking(int M, int K, int N, const float* A, const float* B,
                                                float* C) {
    const int BM = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;

    int col = threadIdx.x;
    int row = threadIdx.y;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A = &A[blockIdx.y * BM * K];
    B = &B[blockIdx.x * BN];
    C = &C[blockIdx.y * BM * N + blockIdx.x * BN];

    float sum = 0.0f;
    for (int k = 0; k < K; k += BK) {
        As[row * BK + col] = A[row * K + col];
        Bs[row * BN + col] = B[row * N + col];
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int i = 0; i < BK; i++) {
            sum += As[row * BK + i] * Bs[i * BN + col];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```


## 一些实践中看起来不那么重要的废话

### ncu 和 nsys 的区别
**ncu** (NVIDIA Nsight Compute) 和 **nsys** (NVIDIA Nsight Systems) 是两个性能分析工具：

| 工具       | 关注层次      | 用途                                                          |
| -------- | --------- | ----------------------------------------------------------- |
| **ncu**  | Kernel 级别 | 分析单个 kernel 的性能细节（指令吞吐量、shared memory bank conflict、寄存器使用等） |
| **nsys** | 系统级别      | 看整体流程（kernel 启动顺序、CPU/GPU 交互、内存拷贝开销等）                       |

**类比理解：**
- **nsys** - 像城市交通摄像头，看整个城市的车流；找"哪个 kernel 慢"
- **ncu** - 像汽车发动机分析仪，细看引擎内部工作；找"这个 kernel 为什么慢"
**使用场景：**
```
用 nsys 先看：
├─ kernel A 启动是不是太慢？
├─ CPU 和 GPU 有没有空闲等待？
├─ cudaMemcpy 是不是占了太多时间？
└─ 发现 kernel B 占用时间最长
↓ 再用 ncu 细看 kernel B：
├─ 寄存器用了多少？有没有溢出？
├─ shared memory 有没有 bank conflict？
├─ L2 cache 命中率怎么样？
└─ 每个 warp 的指令吞吐量够不够？
```
使用方式呢，就是在生成可执行文件后，用指令来生成nsys和ncu的报告。
```bash
nsys profile --trace=cuda,nvtx,osrt -o 报告名 ./可执行文件名
ncu -o 报告名 ./可执行文件名
# 一般这两个报告名直接和可执行文件同名即可，因为报告文件后缀不一样
```
