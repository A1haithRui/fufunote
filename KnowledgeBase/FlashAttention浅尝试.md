# 前言
硬着头皮学习一下FlashAttention。
## 参考资料
[知乎-AI Infra面试常考—FlashAttention系列](https://zhuanlan.zhihu.com/p/2015196808893192187)
[CUDATutorial-FlashAttention专题](https://cuda.keter.top/flash_attn/)
# 个人笔记
### FlashAttention-1
先从注意力的计算开始思考，标准的缩放点积注意力（Scaled Dot-Product Attention）公式为：$Attention(Q, K, V) = \operatorname{Softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$。
在硬件上面，通常分为四步来进行：
1. 计算分数：$S=Q\cdot K^T \in \mathbb{R}^{N\times N}$，N为序列长度。
2. 缩放与掩码：$S_{scaled}=\frac{S}{\sqrt{d_k}}$，k为$\frac{d_{model}}{n_{head}}$
3. 计算权重：$P = softmax(S_{scaled}) \in \mathbb{R}^{N\times N}$
4. 加权求和：$O=PV\in \mathbb{R}^{N\times d}$
标准实现的情况下，S和P的维度是$N\times N$，当序列长度很大时，计算过程的中间矩阵会占用大量显存，且频繁在 GPU 高速显存（HBM）和计算单元（SRAM）之间来回读写，导致计算单元大部分时间在等待数据传输。FlashAttention是一种计算过程的处理思想，来加速注意力的计算过程。
而这种处理方式，可以总结为**分块（Tiling）** 和 **动态重计算**。前者将大矩阵拆分成小块，使其能放入较小的SRAM上进行快速计算；后者在一个GPU Kernel中完成算有计算，不再写回中间矩阵$S$和$P$。
![标准Attention](https://pica.zhimg.com/v2-be96e852156ae3e909266817690bd610_1440w.jpg)
> 这张图就是一个标准多头Attention计算，对于一个Q要和多个K计算注意力分数，点积得到S，然后计算指数得到A，期间还要计算l作为softmax分母，最后结合起来计算输出O
#### Online Softmax
可以看出softmax计算是最复杂的，它一共经历了三个版本，Naive中只算**指数和然后相除**，exp容易导致数值溢出和精度下降；safe版本在**计算指数和的过程中会让当前值减去最大值**来防止溢出同时保证实现softmax的效果，但是求最大值会多一次遍历；所以这里提出了online版本，用增量更新的方式来计算。
Online Softmax讲的最透彻的是b站up主 **[比飞鸟贵重的多_HKL](https://space.bilibili.com/218427631/?spm_id_from=333.788.upinfo.detail.click)**，推导了计算流程。看一遍就懂了。
![online-softmax](https://pic1.zhimg.com/v2-eb4116e0626b2e6dbaf222ac944c2a96_1440w.jpg)
这样我们可以在不知道全局最大值的情况下，计算和存储中间结果，并且将巨大的Attention矩阵切分成小块放入SRAM中，减少访问HBM并提升计算速度。
然后这里重新定义一下变量，因为原论文和代码里用的是另一套定义模式：
- $m^{(i)}$：当前行到第$i$步时的最大值（用于数值稳定性）；
- $l^{(i)}$：当前行到第$i$步时的指数累加和（归一化因子）；
- $O^{(i)}$：当前行到第$i$步时的部分注意力输出结果。
这三个变量与online Softmax极度相关，储存在全局变量中。FlashAttention第一版就是按照`batch`和`Heads`维度做并行，每个`block`计算一个`[N, head_dim]`的`Q`,` K`, `V`矩阵的attention结果。至于计算过程中分块的处理过程，可以看up主“比飞鸟贵重的多_HKL”的示意图。
#### 分块处理
![知乎图片|800](https://pic4.zhimg.com/v2-7cf0b098340052f08770133ce8dbc9fd_1440w.jpg)
不过这里还是得看伪代码才能理解，我个人把这个流程整理一下。注意下面的r代表行，c代表列。
首先，`Q`, `K`, `V`矩阵保存在HBM中，同时SRAM的尺寸为`M`。然后设置两个块的`BLOCK_SIZE`分别为$B_c=cdiv(\frac{M}{4d})$，$B_r = min(cdiv(\frac{M}{4d}, d))$。*这里的cdiv是triton的向上取整，`Q`, `K`, `V`的维度是$N\times d$。那为什么要除以4d捏？我盲猜是因为每个分块要存`Q`，`K`，`V`和`S`四个小块，所以要给它们除以4。*
然后在HBM中初始化三个变量，就是上面我们定义的那三个：$\mathbf{O}=(0)_{N\times d}$，$\mathcal{l}=(0)_N$，$m=(-\infty)_{N}$。
接下来开始切分`Q`矩阵，存到了$B_r$，把它分成多少块呢？$T_r=cdiv(\frac{N}{B_r})$块即可，这样Q矩阵就变成了$[Q_1, Q_2, ..., Q_{T_r}]$，每一块的维度为$B_r\times d$。
同时切分`K`和`V`矩阵，存到了$B_c$上，切分成$T_c=cdiv(\frac{N}{B_c})$块，这样K矩阵变成了$[K_1, K_2, ..., K_{T_c}]$，V矩阵就变成了$[V_1, V_2, ..., V_{T_c}]$，每一块的维度为$B_c\times d$。
然后再来切分和存储这三个变量，他们几个都存到$B_r$上，且都切分为$B_r$块。得到$[\mathbf{O}_{i},...,\mathbf{O}_{T_r}]$，$[\mathcal{l}_i,...,\mathcal{l}_{T_r}$，$[m_1,...,m_{T_r}]$。
存好之后就可以开始迭代处理了：
外层循环，在列上，j从1到$T_c$，把$\mathbf{K}_j$和$\mathbf{V}_j$从HBM加载到片上SRAM。
	然后内层循环，在行上，i从1到$T_r$，把$\mathbf{Q}_i$，$\mathbf{O}_i$，$\mathcal{l}_i$，$m_i$从HBM加载到SRAM。
	在片上，先计算$\mathbf{S}_{ij}=\mathbf{Q}_i \mathbf{K}^T_j$。*$\mathbf{S}_{ij}$的维度为${B_r\times B_c}$。*
	然后在片上，同时计算这三个值：
		$\widetilde{m}_{ij}=\text{rowmax}(\mathbf{S}_{ij})$。 *$\widetilde{m}_{ij}$的维度为$B_r$。*
		$\widetilde{\mathbf{P}}_{ij}=e^{\mathbf{S}_{ij}-\widetilde{m}_{ij}}$。 *$\widetilde{\mathbf{P}}_{ij}$的维度为$B_r\times B_c$，（pointwise）。*
		$\tilde{\mathcal{l}}_{ij}=\text{rowsum}(\widetilde{\mathbf{P}}_{ij})$。 *$\tilde{\mathcal{l}}_{ij}$的维度为$B_r$。*
	然后在片上，去下一行计算新的这三个值：
		$m_{i}^{new}=\text{max}(m_i, \widetilde{m}_{ij})$。 *$\widetilde{m}_{i}^{new}$的维度为$B_r$。*
		$\mathcal{l}_{i}^{new}=e^{m_i-m_i^{new}}\mathcal{l}_i+e^{\widetilde{m}_{ij}-m_i^{new}} \tilde{\mathcal{l}}_{ij}$。 *$\widetilde{\mathcal{l}}_{i}^{new}$的维度为$B_r$。*
	接下来就要计算矩阵$\mathbf{O}$并且往HBM回写：
		$O_i = diag(\mathcal{l}_{i}^{new})^{-1} (diag(\mathcal{l}_i) e^{m_i-m_i^{new}} \mathbf{O}_i + e^{\widetilde{m}_{ij}-m_i^{new}} \widetilde{\mathbf{P}}_{ij} \mathbf{V})$
	最后把$\mathcal{l}_{i}^{new}$赋值给$\mathcal{l}_i$，把$m_i^{new}$赋值给$m_i$，并写回HBM。
	结束内层循环
结束外层循环。
> 终于理解了一遍了，逻辑很清晰，主要是按照online softmax的思路来计算的，只是注意Q和中间变量存在行上，K和V存在列上。
> CUDATutorial里举了一个N=4，d=2的例子，可以移步去参考。
![知乎](https://pic4.zhimg.com/v2-81a47d03d1fca6dd3636824aa3f23529_1440w.jpg)
#### FLOPs怎么算？
首先，每个分块`[Qi, Kj, Vj, S]`占用的共享内存大小为`tile_size = Bc * d`。因为`S`是得分矩阵，要和`V`矩阵算输出`O`，所以存在共享内存中减少读写次数。
然后，`B_c`和`B_r`的值根据共享内存大小来定，始终要保持`tile_size * 4 < M`。
`Q`矩阵的维度为`[B_r, d]`，要循环`T_r = N / B_r`次。
`K`和`V`矩阵的维度为`[B_c, d]`，要循环`T_c = N / B_c`次。
循环总次数为`T_r * T_c = cdiv(N / B_r) * cdiv(N / B_c)`。因此总的FLOPs为$O(\frac{N^2}{B_cB_r} B_r B_c d)=O(N^2)d$
#### 代码
这里搞了一版torch的写法，去掉了注释，原本详细代码在仓库[how-to-optim-algorigthm-in-cuda](https://github.com/BBuf/how-to-optim-algorithm-in-cuda)的triton目录中。这个代码几乎完全是按照上述思路来写的，一点歧义也没有。
```python
import torch

N, d = 1024, 64
Q_mat, K_mat, V_mat = torch.rand((N, d)), torch.rand((N, d)), torch.rand((N, d))

def standard_softmax_attention(Q, K, V):
    expected_softmax = torch.softmax(Q @ K.T, dim=1)
    expected_attention = expected_softmax @ V
    return expected_attention

def flash_attnetion_v1(Q, K, V, B_r=64, B_c=768):
    N, d = Q.shape
    O = torch.zeros((N, d))
    l = torch.zeros((N, 1))
    m = torch.full((N, 1), -torch.inf)

    for j in range(0, N, B_c):
        Kj = K[j : j + B_c, :]
        Vj = V[j : j + B_c, :]

        for i in range(0, N, B_r):
            Oi = O[i : i + B_r, :]
            Qi = Q[i : i + B_r, :]
            mi = m[i : i + B_r, :]
            li = l[i : i + B_r, :]

            Sij = Qi @ Kj.T
            mij_hat = torch.max(Sij, dim=1).values[:, None]
            pij_hat = torch.exp(Sij - mij_hat)
            lij_hat = torch.sum(pij_hat, dim=1)[:, None]

            mi_new = torch.max(torch.column_stack([mi, mij_hat]), dim=1).values[:, None]
            li_new = torch.exp(mi - mi_new) * li + torch.exp(mij_hat - mi_new) * lij_hat

            O_i = (
                (li * torch.exp(mi - mi_new) * Oi)
                + (torch.exp(mij_hat - mi_new) * pij_hat) @ Vj
            ) / li_new

            m[i : i + B_r, :] = mi_new
            l[i : i + B_r, :] = li_new
            O[i : i + B_r, :] = O_i
    return O

flash_attention_out = flash_attnetion_v1(Q_mat, K_mat, V_mat)
expected_attention = standard_softmax_attention(Q_mat, K_mat, V_mat)
print(torch.allclose(flash_attention_out, expected_attention))

```
### FlashAttention-2
如果说第一版提出了分块和重计算，那么第二版主要优化了**并行维度**，**算法开销**和**任务划分**这三块：
#### 减少非矩阵乘法(Non-matmul)开销
想想，GPU最擅长做的是矩阵乘法，而做指数运算、缩放等操作相对较慢。所以第二版调整了online softmax的计算逻辑。第一版中，每一步循环都要对输出$O$进行重缩放，第二版则将缩放操作移到了循环结束的位置，减少循环中大量不必要的计算。
- 缩放？在第一版的内循环中，每次计算出一块注意力后，都要对历史输出$O^{(i-1)}$进行一次繁琐的缩放，需要乘以旧的局部指数和$l^{(i-1)}$，再除以新的局部指数和$l^{(i)}$。第二版就是把**乘以$l^{(i-1)}$** 和 **除以$l^{(i)}$** 的中间归一化步骤给优化掉。
- 如何去掉？在第一版中，外循环是K和V，内循环是Q和O。因为O在循环中会被不断从HBM中读出和写入，为了保证写入HBM的数据不会因为数值过大而溢出，且满足Attention的数学计算方式，所以每次都把O做了归一化后才写回HBM；第二版中，外循环改成Q和O，内循环改成K和V。此时，当前处理的Q和O会一直停留在超快的SRAM和寄存器中，直到遍历完所有的K和V才写回显存。既然中间结果不会写回显存，我们完全可以在寄存器里维护一个未归一化的中间累加值，等所有循环跑完，最后再做一次除法进行归一化即可。
- 数学上怎么推导？先定义之前代码里的m和l。全局最大值：$m^{(i)}=\max{(m^{(i-1)}, m^{(i)})}$，全局分母(指数和)：$l^{(i)}=l^{(i-1)} e^{m^{(i-1)}-m^{(i)}}+e^{S_i-m^{(i)}}$。
	第一版中，更新方式为：
$$
O^{(i)} = \frac{1}{l^{(i)}}(l^{(i-1)}\cdot e^{m^{(i-1)}-m^{(i)}}\cdot O^{(i-1)}+e^{S_i-m^{(i)}}\cdot V_i)
$$
	每次更新中，GPU都必须执行一次除法操作，除法在GPU中很消耗始终周期。
	在第二版中，我们不再维护归一化的$O^{(i)}$，而是引入一个未归一化的中间变量$\widetilde{O}^{(i)}$ ，定义其为$\widetilde{O}^{(i)}=l^{(i)}O^{(i)}$）(意思就是没除以新的$l$，在这里乘回来了)，那么将这个中间变量带入回去， 第一版的更新公式就变成了：
$$
\widetilde{O}^{(i)} = e^{m^{(i-1)}-m^{(i)}}\cdot \widetilde{O}^{(i)}+e^{S_i-m^{(i)}}\cdot V_i 
$$
（注意，本质就是给上面的公式左右乘了l^{(i)}）
2. 这样操作，原本O的更新只需要加法和乘法，去掉了每轮对l的乘除。当遍历完所有的K和V之后（假设有T块），我们手里拿到了最终的未归一化结果$\widetilde{O}^{(T)}$和最终的分母$l^{(T)}$，此时，只需要做唯一的除法即可：
$$
O=\frac{\widetilde{O}^{(T)}}{l^{(T)}}
$$
流程如图：
![知乎图片](https://pic2.zhimg.com/v2-643e6aa90d08bfcac2ebbea61192a9f1_1440w.jpg)
#### 代码
```python
import torch

N, d = 1024, 64

Q_mat, K_mat, V_mat = torch.rand((N, d)), torch.rand((N, d)), torch.rand((N, d))


def standard_softmax_attention(Q, K, V):
    expected_softmax = torch.softmax(Q @ K.T, dim=1)
    expected_attention = expected_softmax @ V
    return expected_attention


def flash_attention_v2(Q, K, V, B_r=64, B_c=768):
    N, d = Q.shape
    O = torch.zeros((N, d))
    l = torch.zeros((N, 1))
    m = torch.full((N, 1), -torch.inf)

    for i in range(0, N, B_r):
        Qi = Q[i : i + B_r, :]
        for j in range(0, N, B_c):
            Kj = K[j : j + B_c, :]
            Vj = V[j : j + B_c, :]

            Sij = Qi @ Kj.T
            mi_new = torch.max(
                torch.column_stack(
                    [m[i : i + B_r], torch.max(Sij, dim=1).values[:, None]]
                ),
                dim=1,
            ).values[:, None]
            Pij_hat = torch.torch.exp(Sij - mi_new)
            l[i : i + B_r] = (
                torch.exp(m[i : i + B_r] - mi_new) * l[i : i + B_r]
                + torch.sum(Pij_hat, dim=1)[:, None]
            )
            O[i : i + B_r] = (
                torch.exp(m[i : i + B_r] - mi_new) * O[i : i + B_r] + Pij_hat @ Vj
            )
            m[i : i + B_r] = mi_new
    O = O / l
    return O


flash_attention_out = flash_attention_v2(Q_mat, K_mat, V_mat)
expected_attention = standard_softmax_attention(Q_mat, K_mat, V_mat)
print(torch.allclose(flash_attention_out, expected_attention))

```
在GPU内部，32个线程组成一个Warp，第二版重新设计了Warp之间的分工，减少了它们通过共享显存交换数据的次数，从而降低了通信延迟。划分情况大概如下图：
![知乎图片](https://pic3.zhimg.com/v2-32f00a3f4a6e7c4fec197ce51b4b224a_1440w.jpg)

### 以上内容整理的太细节了，仅做推导参考
