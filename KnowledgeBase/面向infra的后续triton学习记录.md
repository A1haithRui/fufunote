# 前言
还是从 傅莘莘(傅猪猪) 的课程之后继续学习triton，同时也自己找了资料进行了解。triton本质上就是python的一个库，有着固定的模版和代码，从官网手册来看语法并不难，重点在于要分配block进行操作。这里整理一些代码，写一些自己的散装理解。
## 资料
[Triton视频](https://www.youtube.com/watch?v=FtmnriHLbAg&list=PL_NMDNzkCbLR69QafQUa6yTx-T-VPIoaZ)    |    Youtube上一个老哥讲的，看起来很权威
[官方手册](https://openai.com/index/triton/)    |    非常直观
[CUDATutorial](https://cuda.keter.top/)    |    居然忘了手上有这么个东西，除了讲cuda，也讲到一些triton的东西。
- 突然发现infiniTensor的triton炼气术讲的非常好，非常非常细，前面看的迷糊的，在这里全看懂了。
[infiniTensor-Triton炼气术](https://www.bilibili.com/video/BV1td5y6CErE?vd_source=6703d58dba285e1204c6c03c7e4c32ef)    |    不愧是科班的老师，真的回到了本科上课的感觉，第一课讲triton；后面两课讲九齿。
[Triton-Puzzles](https://github.com/gpu-mode/Triton-Puzzles)    |    在ipynb上练手，已经简单写了几道题，保存在了本地
[九齿文档](https://gxtctab8no8.feishu.cn/wiki/RSZSwpc9siOEXrkUoVyc2lPtnX3)    | 九齿的官方教程
# 个人笔记
## 模版代码
### Kernel 的写法
大部分Triton Kernel都遵循以下模版：
```python
import torch
import triton
import triton.language as tl

@triton.jit # 装饰器，just-in-time，即时编译，运行时让GPU编译，而非提前编译好
# 如果不加这个装饰器的话，就是一个普通的python函数，没法上GPU
def my_kernel(
    input_ptr,          # 输入数据指针
    output_ptr,         # 输出数据指针
    N,                  # 总元素数量
    BLOCK_SIZE: tl.constexpr,  # 编译期常量，每个 Block 处理的元素数
):
    # 1. 获取当前 Block 的 ID
    pid = tl.program_id(axis=0)

    # 2. 计算当前 Block 负责的偏移量范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 3. 生成越界保护掩码
    mask = offsets < N

    # 4. 从全局内存加载数据
    x = tl.load(input_ptr + offsets, mask=mask)

    # 5. 执行计算（核心逻辑）
    y = your_func(x)

    # 6. 将结果写回全局内存
    tl.store(output_ptr + offsets, y, mask=mask)
```
### Host 端的封装
Kernel不能直接被调用，需要一个python函数来进行封装：
```python
def my_operator(x: torch.Tensor) -> torch.Tensor:
    # 确保输入在 GPU 上 并且传入的内存连续
    assert x.is_cuda and x.is_contiguous()
	
    # 预分配输出内存
    output = torch.empty_like(x)
    n_elements = x.numel() # .numel()可以获取这个张量的参数，本质就是把所有维度数相乘

    # 配置执行参数
    BLOCK_SIZE = 256  # 可选的块大小，还可以写512，1024……
    # triton.cdiv向上取整，计算需要多少个Block能够覆盖所有元素
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )  
    # grid是一个元组，表示启动1D、2D或者3D的网格，这里使用(x,)确保它是一个元组

    # 启动 Kernel
    my_kernel[grid](x, output, N, BLOCK_SIZE=BLOCK_SIZE)

    return output
```
关于这里的参数量和分配准则，可以这样理解：
```
输入张量大小 n_elements = 5120, BLOCK_SIZE = 1024 的情况:
|<-1024->|<-1024->|<-1024->|<-1024->|<-1024->
[Block 0][Block 1][Block 2][Block 3][Block 4]
```
### Autotune 自动调优
官网手册里写到了自动调优，实际项目中，`BLOCK_SIZE`等参数需要根据问题的规模来动态选择，所以可以通过`@triton.autotune`装饰器：
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4), # 每个Block分配的Warp的数量，CUDA中学过，一个Warp中有32个线程
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
    ],
    key=['N'],
)

@triton.jit
def my_kernel_autotuned(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Kernel 内容同上
    ...
```
### 算子注册和基准测试
然后还有算子注册和基准测试，这个后续遇到了再补进来。
## 面向ai infra的cuda记录
### DRAM vs SRAM
先理解一下这两个概念，DRAM (Dynamic Random-Access Memory) 就是GPU上的**全局内存**，容量大（RTX 4060 有 7GB），速度慢，访问延迟高，而代码里的 input_ptr 和 output_ptr 指向的就是 DRAM。
SRAM（Static Random-Access Memory）就是GPU上的**共享内存**，容量小（RTX 4060 每个 SM 只有 100KB），速度极快（比 DRAM 快 100 倍以上），访问延迟低。所以一般要把数据加载到SRAM里算。
这个网页里有一张图挺直观：[指路](https://cuda.keter.top/flash_attn/flash_attn_v1_part1)
**那为什么要写这么复杂？**
```
CPU 发起请求
    ↓
GPU 核函数启动
    ↓
DRAM (慢、大) ←─ 只能批量加载，不能一个个读
    ↓ (通过 tl.load 一次性加载一整块)
SRAM (快、小) ←─ 在这里做计算
    ↓ (通过 tl.store 写回)
DRAM (慢、大)
```
>**类比理解**
>**DRAM** = 你的书架（书多，但找书慢）
>**SRAM** = 你的书桌（空间小，但拿取方便）
 写代码的复杂逻辑就像：
>1. 先从书架上把一摞书搬到书桌上（批量加载）
>2. 在书桌上看书、做笔记（在 SRAM 里计算）
>3. 看完了把书放回书架（批量写回）
 不直接在书架上看书（不直接在 DRAM 里计算）是因为太慢了
## 用triton写算子
为什么要写算子？
### Softmax 算子
下面这份代码是把youtube上那位老哥的注释翻译了过来。
```python
@triton.jit 
def _softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,    # 移动到下一行时需要跳过的元素数量
    n_rows, n_cols,                         # 矩阵维度
    BLOCK_SIZE: tl.constexpr,               # 大于 n_cols 的最小 2 的幂
    num_stages: tl.constexpr,
): 
    # 程序处理的起始行号由 pid (program id) 决定
    row_start = tl.program_id(0) 
    # 获取并行程序的总数，用它来决定 for 循环中每次处理完一行后要步进多少
    row_step = tl.num_programs(0) 
        # 每个程序处理的行按 row_step 间隔
        # (例如：如果有 4 个程序，程序 0 处理行 0,4,8...)
    
    # tl.arange() 返回值数组，而 tl.range() 作为迭代器使用
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # Triton 不会真正顺序执行 for 循环的每次迭代，而是可以用 num_stages 
        # 让不同的迭代同时工作。当然，只有在迭代之间互不依赖时才能这样做
        
        # stride 表示指针移动一行需要增加的量（按元素数计，不是按字节计）
        row_start_ptr = input_ptr + row_idx * input_row_stride
            # 对于形状是 [n_rows, n_cols] 的连续二维张量，input_row_stride 一般是 n_cols，
            # 因为从当前行跳到下一行，要跨过整整一行的元素
            # 列方向上相邻元素的步长通常才是 1
            # 如果传入的是非连续张量视图，这个 stride 就更不能写死

        # 把一行数据加载到 SRAM，使用 mask 是因为如果 n_cols 不是 2 的幂，BLOCK_SIZE 会大于 n_cols
        col_offsets = tl.arange(0, BLOCK_SIZE) # 每一行可以放到一个 block 里
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=float('-inf')) 
            # 被 mask 掉的索引位置用 -inf 填充，因为这个值不会影响 softmax 计算

        # 减去最大值以保证数值稳定性
        row_minus_max = row - tl.max(row, axis=0)
            # 减去最大值后，所有无效的 -inf 仍然是 -inf
        # 注意：Triton 中的指数运算很快但是近似的；稍后我们会学一个更快的替代方案
        numerator = tl.exp(row_minus_max)
            # 所有 -inf 值变成 0，因为 exp(-inf)=0
        denominator = tl.sum(numerator, axis=0)
            # 所有无效的 0 值会被求和，但因为是 0 所以不影响结果
        softmax_output = numerator / denominator
            # 所有无效的 0 是 0/sum，所以仍然是 0

        # 把输出写回 DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)
            # 使用 mask 只存储前 n_cols 个有效值
```
#### **num_stages是干嘛的？** 
这个是一个高级优化技术，类似于 CPU 的指令流水线。当循环迭代之间**互不依赖**时可以使用，softmax 刚好就是行与行之间独立；`num_stages` 可以理解成预取/流水的深度，通常更大意味着更积极地重叠加载和计算，但资源占用也会更高，不一定越大越快。
```
num_stages = 1（默认，顺序执行）：
时间线：
加载行 0 → 计算行 0 → 存储行 0 → 加载行 4 → 计算行 4 → 存储行 4
└─────── 一次迭代 ────────┘    └──────── 一次迭代 ────────┘

num_stages = 3（流水线）：
时间线：
加载行 0 → 计算行 0 → 存储行 0
        加载行 4 → 计算行 4 → 存储行 4
                加载行 8 → 计算行 8 → 存储行 8
        └─────── 重叠执行，提高利用率 ────────┘
```
不过一般教程基本会写的简单一些，直接一行一行走。
#### tl.range() 和普通 range 的区别
```python
# 普通 Python range（CPU 上的，在 kernel 里用不了）
for i in range(0, 10, 2):
    print(i)  # 输出: 0, 2, 4, 6, 8
# Triton 的 tl.range（GPU 上的，在 kernel 里用）
for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
    ...
```
这四个参数分别表示：
```python
tl.range(start, stop, step, num_stages=...)
           ↓      ↓     ↓       ↓
         起始   结束  步长   流水线深度
```
在这份代码中：
```python
tl.range(
    row_start,    # 起始行 = pid (当前程序 id)
    n_rows,       # 结束行 = 总行数
    row_step,     # 步长 = 程序总数
    num_stages=num_stages  # 流水线深度
)
```
**举个例子**
假设：n_rows = 100 行，启动 4 个程序 (num_programs = 4)
```python
程序 0 (pid=0):
tl.range(0, 100, 4) → 行 0, 4, 8, 12, ..., 96, 100?
                                          ↓
                                   不超过 n_rows=100
程序 1 (pid=1):
tl.range(1, 100, 4) → 行 1, 5, 9, 13, ..., 97
程序 2 (pid=2):
tl.range(2, 100, 4) → 行 2, 6, 10, 14, ..., 98
程序 3 (pid=3):
tl.range(3, 100, 4) → 行 3, 7, 11, 15, ..., 99
```
最终，所有行都被处理了：0, 1, 2, 3, 4, 5, ..., 99。没有重复，没有遗漏。
#### 核心逻辑
所以核心逻辑就是以下部分（代码没变，只是把繁琐的注释给删了）。这下看起来就比较直观了。
```python
for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
	row_start_ptr = input_ptr + row_idx * input_row_stride
	col_offsets = tl.arange(0, BLOCK_SIZE) 
	input_ptrs = row_start_ptr + col_offsets
	mask = col_offsets < n_cols
	row = tl.load(input_ptrs, mask=mask, other=float('-inf')) 
	
	row_minus_max = row - tl.max(row, axis=0)
	numerator = tl.exp(row_minus_max)
	denominator = tl.sum(numerator, axis=0)
	softmax_output = numerator / denominator
	
	output_row_start_ptr = output_ptr + row_idx * output_row_stride
	tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)
```
不过这个老哥的写法确实很复杂，CUDATutorial上有个比较简单的写法[指路](https://cuda.keter.top/triton/triton_softmax/)
### Matmul 算子
还是把youtube老哥的代码搬过来参考，注释翻译成中文：
```python
# `triton.jit` 装饰的函数可以通过 `triton.autotune` 装饰器来自动调优，它需要：
#   1) 一个 `triton.Config` 对象列表，定义不同的元参数和编译选项配置
#   2) 一个自动调优键（key），其值发生变化时会触发重新评估所有配置；意味着
#      只要 M、N 或 K 随新输入发生变化，Triton 就会重新检查哪个配置最好
@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr, 
    M, N, K, 
    stride_a_M, stride_a_K, 
    stride_b_K, stride_b_N, 
    stride_c_M, stride_c_N, 
    # meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, 
):
    """
    首先我们需要将每个程序 id 映射到它要计算的 C 矩阵的块。
    看一个例子，其中
    M = N = K = 8，
    BLOCK_SIZE_M = BLOCK_SIZE_K = BLOCK_SIZE_N = 2
    一个朴素的实现可能会像这样分配 PID：
    [0,   1,  2,  3]
    [4,   5,  6,  7]
    [8,   9, 10, 11]
    [12, 13, 14, 15]
    这里每个 PID 对应 C 矩阵的一个 2x2 块（C 大小是 8x8）。
    为了计算其中一个块（比如 PID=0），我们需要 A 和 B 的哪些部分呢？
    让我们看看矩阵乘是如何工作的，用 x 标记我们需要用到的 A 和 B 的块，
    来创建对应于 PID=0 的 C 矩阵块：
        A           @       B           =       C
    [x, x, x, x]        [x, _, _, _]        [0, _, _, _]
    [_, _, _, _]        [x, _, _, _]        [_, _, _, _]
    [_, _, _, _]        [x, _, _, _]        [_, _, _, _]
    [_, _, _, _]        [x, _, _, _]        [_, _, _, _]
    所以为了创建对应 PID=0 的 C 的 2x2 块，我们需要 A 顶部四行的四个 2x2 块和 B 前四列的四个 2x2 块。
    注意我们不必同时把这 8 个块全部加载到 SRAM，而可以分别迭代 A 的列和 B 的行，
    在对应的两个块之间做迷你矩阵乘，并在过程中把结果累加起来。
        A           @       B
    [--------->]        [ | , _, _, _]
    [_, _, _, _]        [ | , _, _, _]
    [_, _, _, _]        [ | , _, _, _]
    [_, _, _, _]        [\|/, _, _, _]
    如果我们按上述实现这个算法，它可以工作。
    但是它不会像 PyTorch 的方法那么快，PyTorch 实现了更巧妙的东西。
    要知道为什么，我们需要考虑 SRAM 的使用。
    注意 PID 0 到 3 都会用到 A 的同一行块，同时要记住一个 SM 上可以有多个程序
    共享同一块 SRAM。
    这意味着不必各自独立加载 A 的那一行块（导致一堆重复），一旦一个 PID 加载了该行上的 A 块，其他 3 个 PID 就可以直接复用它！

    幸运的是我们不必显式告诉 Triton 这样做；每次一个 PID 运行 tl.load() 时，
    它首先会自动检查该数据是否已存在于 SRAM 中——因为同一个 SM 上的其他 PID 可能先加载了。
    虽然我们不必显式告诉 Triton 这样做，但我们应当仔细考虑如何通过操控 PID 的顺序来帮 Triton 最好地利用这个能力。
    重要的是，PID 是按顺序分配给 SM 的！！
    重申一下，PID 的顺序决定了 C 的哪些块会共享 SRAM！
    再看一下 PID 0 到 3，具体看它们需要加载 A 和 B 的哪些块：
    PID = 0
    [x, x, x, x]        [x, _, _, _]
    [_, _, _, _]        [x, _, _, _]
    [_, _, _, _]        [x, _, _, _]
    [_, _, _, _]        [x, _, _, _]
    PID = 1
    [x, x, x, x]        [_, x, _, _]
    [_, _, _, _]        [_, x, _, _]
    [_, _, _, _]        [_, x, _, _]
    [_, _, _, _]        [_, x, _, _]
    PID = 2
    [x, x, x, x]        [_, _, x, _]
    [_, _, _, _]        [_, _, x, _]
    [_, _, _, _]        [_, _, x, _]
    [_, _, _, _]        [_, _, x, _]
    PID = 3
    [x, x, x, x]        [_, _, _, x]
    [_, _, _, _]        [_, _, _, x]
    [_, _, _, _]        [_, _, _, x]
    [_, _, _, _]        [_, _, _, x]
    注意虽然它们都可以共享 A 的第一行块，从而避免加载其他三行块，
    但它们实际上会加载 B 的每一列块。
    我们可以做得更好吗？
    我们可以用更少的 A 和 B 总块数来获得同样多的 PID（因此同样多的 C 块）吗？
    目前使用这种我们称为“行优先顺序”的方法，我们正在加载：
        (1 行 A 块) + (4 列 B 块) = 总共 5 行/列块加载到 SRAM

    那如果不把 PID 0-3 放到同一个 SM，而是把 PID 0、1、4、5 放到同一个 SM 呢？
    看看 PID 4 和 5 需要加载什么：
    PID = 4
    [_, _, _, _]        [x, _, _, _]
    [x, x, x, x]        [x, _, _, _]
    [_, _, _, _]        [x, _, _, _]
    [_, _, _, _]        [x, _, _, _]
    PID = 5
    [_, _, _, _]        [_, x, _, _]
    [x, x, x, x]        [_, x, _, _]
    [_, _, _, _]        [_, x, _, _]
    [_, _, _, _]        [_, x, _, _]
    现在在这个假设的新安排下，我们只需要加载：
        (2 行 A 块) + (2 列 B 块) = 总共 4 行/列块加载到 SRAM
    而我们仍然获得同样多的 C 块作为输出！
    这种策略称为“group-major ordering（组优先顺序）”。
    在这个小例子中效果看起来不是很大，但随着块数增加，它在避免
    把 A 和 B 块重复加载到不同 SM 这件事上会越来越有效。

    但要记住，Triton 是根据 PID 顺序把块加载到 SM 的，这意味着
    虽然我们希望 PID 0、1、4、5 共享同一块 SRAM，但实际上 PID 3 和 4
    很可能会打乱这一点。
    那么我们如何确保对应 PID 4 和 5 的 C 块与 0 和 1 加载到同一个 SM 呢？
    我们实际上必须重新排序 PID，也就是把它们重新分配给不同的 C 块。
    记住我们的启动网格是 1 维的（像之前见过的所有启动网格那样），意味着
    它由只有一个元素的元组定义。
    一旦进入核函数，我们的任务就是把这个 1 维 PID 列表变成我们想要的形状。
    再次强调，这里的关键点是数值更接近的 PID 更有可能
    落在同一个 SM，意味着虽然我们说过 0、1、4、5 最好共享 SRAM，
    实际上根据我们之前的“朴素”启动网格，0、1、2、3 会被分到一起。
    所以我们需要做的是把 2 和 3 移走，让它们对应之前分配给 4 和 5 的 C 块。
    与其解释，不如看看这个新的视觉顺序：
    [0,  2,  4,  6]
    [1,  3,  5,  7]
    [8, 10, 12, 14]
    [9, 11, 13, 15] 
    现在，0 到 3 对应组优先顺序！注意在这个例子中我们可以把它可视化成
    将 PID 分成由虚线标出的“组”
    [0,  2, |  4,  6]
    [1,  3, |  5,  7]
    --------|--------
    [8, 10, | 12, 14]
    [9, 11, | 13, 15] 
    这些组的大小由我们的 "GROUP_SIZE" 元参数定义。
    要实现 PID 的这种重排序，我们需要做一些技术上简单但
    惊人地难以跟踪的数学变换。
    """
    # 我们从 1 维启动网格开始，然后把它变成有复杂“组级”顺序的 2 维网格
    PID = tl.program_id(axis=0) 
    # 定义组的大小
    num_PID_along_M = tl.cdiv(M, BLOCK_SIZE_M) # 沿 M 维度的块数
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N) # 沿 N 维度的块数
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    # 算出当前 PID 属于哪个组
    group_id = PID // num_PID_in_group 
    # 告诉我们当前组从哪一行开始
    first_PID_in_group_along_M = group_id * GROUP_SIZE 
    # 这通常等于 GROUP_SIZE；特殊情况是当我们在张量边缘且
    # 其维度不能被 GROUP_SIZE 整除时 # TODO 这对吗？
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE) 
    # 这是将 PID 实际映射到组优先顺序的主要部分
    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
        # (PID % num_PID_in_group) 将当前程序 id 放到组的上下文中
        # (first_PID_in_group_along_m + ...) 将 PID 移到正确的组中
        # (... % group_size_adj) 去掉列分量，让我们落到正确的行
    PID_N = (PID % num_PID_in_group) // group_size_adj
        # (... // group_size_adj) 去掉行分量，让我们落到正确的列
    
    # PID 噩梦已结束，现在可以进入你更熟悉的核函数代码了。

    # 让我们为输入矩阵的第一组块创建指针向量
    offsets_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)
    # 之前课程中加载到 SRAM 的块是向量；这里它们是矩阵
    a_offsets = offsets_M[:, None] * stride_a_M + offsets_K[None, :] * stride_a_K
    b_offsets = offsets_K[:, None] * stride_b_K + offsets_N[None, :] * stride_b_N
    """
    [:, None] 把 [m1,m2,m3] 变成 [[m1],[m2],[m3]] 
    [None, :] 把 [n1,n2,n3] 变成 [[n1,n2,n3]]
    把它们结合起来得到矩阵
    [[m1n1, m1n2, m1n3],
     [m2n1, m2n2, m2n3],
     [m3n1, m3n2, m3n3]] 
    """

    # 输入张量是 fp16，但我们累加到 fp32 的块中以获得更高精度（之后会恢复）
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32) # 完整的 C 是 (M, N)
        # 要了解为什么累加有效，请查看 `./block_wise_matmul.png`
        
    # 我们将沿 A 和 B 的 K 维度迭代，来计算 C 矩阵的一个块
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 沿 K 的越界项需要用 mask 排除
        mask = offsets_K < K - k * BLOCK_SIZE_K
            # k * BLOCK_SIZE_K 是 offsets_k 的当前起始索引。
            # 所以这实际上只在 k 距离 K 在 BLOCK_SIZE_K 之内时激活
            # 意味着在循环的最后一次迭代时触发，并且只当 K 不是 BLOCK_SIZE_K 的倍数时
        
        # 现在我们加载 A 和 B 矩阵的块。如果同一个组中的多个块在同一个 SM 上，
        # 它们可以共享这些加载的值，从而减少从 DRAM 加载的昂贵次数
        a = tl.load(a_ptr + a_offsets, mask=mask[None, :], other=0.0) # 形状 (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptr + b_offsets, mask=mask[:, None], other=0.0) # 形状 (BLOCK_SIZE_K, BLOCK_SIZE_N)
            # 用 0.0 填充任何被 mask 掉的部分，因为它们对下一步求和没有影响

        # 我们沿 K 维度累加
        accumulator = tl.dot(a, b, acc=accumulator)
            # Triton 的操作符号有点奇怪；这实际上是一个迷你矩阵乘而不是点乘
            #   形状 (BLOCK_SIZE_M, BLOCK_SIZE_K) @ (BLOCK_SIZE_K, BLOCK_SIZE_N) = (BLOCK_SIZE_M, BLOCK_SIZE_N)
            # `acc` 告诉 Triton 把矩阵乘的输出直接写到 accumulator，这样比
            #   accumulator += tl.dot(a, b) 更高效

        # 把指针沿 K 向前移动到下一个块
        a_offsets += BLOCK_SIZE_K * stride_a_K
        b_offsets += BLOCK_SIZE_K * stride_b_K
        """
        PID=0 时累加移动的可视化
            A           @       B
        [--------->]        [ | , _, _, _]
        [_, _, _, _]        [ | , _, _, _]
        [_, _, _, _]        [ | , _, _, _]
        [_, _, _, _]        [\|/, _, _, _]
        """

    # 现在我们把数据类型改回期望的输出类型
    accumulator = accumulator.to(tl.float16)

    # 用 mask 写回输出矩阵 C 的块
    c_offsets = stride_c_M * offsets_M[:, None] + stride_c_N * offsets_N[None, :]
    c_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N) # 注意这个 2D mask
    tl.store(c_ptr + c_offsets, accumulator, mask=c_mask) # 形状 (BLOCK_SIZE_M, BLOCK_SIZE_N)

```
这位老哥的写法深入浅出，很详细，然后惊奇的发现某博主卖的课里讲到的代码竟然和老哥基本一致，就拿来当核心实现逻辑的参考版本吧。
```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```
如果把这份 matmul kernel 当成标准模板去背，我觉得更适合背“骨架流程”而不是逐行死记：
1. **先用 `pid` 定位当前 program 负责的 `C` 的哪一个 tile**
2. **根据 `pid_m`、`pid_n` 算出这个 tile 在 `M/N` 维度上的偏移**
3. **构造 `A` 和 `B` 当前 tile 的指针矩阵**
4. **创建 `fp32 accumulator`，沿着 `K` 维一块一块循环**
5. **每轮做 `tl.load -> tl.dot -> accumulate`，然后把 `a_ptrs/b_ptrs` 推进到下一个 `K` 块**
6. **循环结束后把 accumulator cast 成输出 dtype**
7. **最后构造 `C` 的输出指针，带 mask 写回**
一句话版就是：
> Triton matmul 的核心套路 = 定位 `C` 的输出 tile，构造 `A/B` 的 tile 指针，沿 `K` 维分块加载并做迷你矩阵乘累加，最后带 mask 写回输出。

## leetGPU刷题记录
看教程看不懂其实可以直接来刷题。
### Vector Addition
还是拿经典向量加来看，如果是cuda怎么做？
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) {
    C[idx] = A[idx] + B[idx]
}
```
写cuda代码时候需要考虑单个线程的分配，手动管理Grid、Block，然后控制每一个Thread的行为，例如线程读取哪块内存？计算什么元素？何时做同步等等。更高级一些还需要考虑管理共享内存，warp寄存器等来合理利用GPU的性能。
对于Triton，提供了一种这种的方案，**分块编程范式**，不需要管理单个线程，直接以Tile为粒度描述计算。
```python
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0) # 获取当前Program的ID，axis相当于cuda中的.x
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE) # 获取Tile内的所有偏移，相当于cuda的idx

    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = a + b

    tl.store(c_ptr + offsets, c, mask=mask)
```
看起来我们似乎像是在写标量计算，但是`tl.arange`生成了一个向量化的偏移数组，`tl.load`一次性加载了一整块数据，Triton编译器会自动将这块的计算映射到底层线程上。
