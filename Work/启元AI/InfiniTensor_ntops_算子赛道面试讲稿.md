# InfiniTensor ntops 算子开发赛道面试讲稿

> 目标：面试中用 3 到 8 分钟讲清楚这段比赛经历。重点不是把每行代码背下来，而是讲清楚“我做了什么、为什么难、怎么解决、学到了什么”。

## 1. 一句话项目介绍

我参加了 InfiniTensor 启元 AI 算子开发竞赛，做的是 ntops 仓库里的 PyTorch 算子补全。这个项目的核心是用 ninetoothed 这个基于 Triton 的算子 DSL，实现若干 PyTorch 算子的 kernel、torch wrapper 和测试，并在 NVIDIA、沐曦、天数、摩尔等异构 GPU 环境上做验证。

## 2. 背景：ntops、ninetoothed 和 Triton

ntops 可以理解成一个“用九齿实现常见 PyTorch 算子”的库。每个算子通常有两条路径：

- PyTorch reference：直接调用 `torch.xxx`，用于对齐语义和测试正确性。
- ninetoothed implementation：用九齿 DSL 写 kernel，再通过 torch wrapper 暴露成类似 PyTorch 的接口。

ninetoothed 本身是一个建立在 Triton 之上的更高层 DSL。它把 kernel 拆成两层：

- `arrangement`：描述数据如何被切成 block、展平、广播、换维等。
- `application`：描述一个 block 内部每个元素或者每组元素要做什么计算。

和直接写 Triton 相比：

- Triton 更底层，开发者直接写 `tl.arange`、`tl.load`、`tl.store`、mask、pointer arithmetic，控制 program id、block size、num warps 等。
- ninetoothed 更抽象，常见逐元素算子会更简洁，比如 `abs`、`frac`、`silu` 这类可以直接在 application 里表达数学公式。
- 代价是 ninetoothed 的 arrangement 原语比较有限。它适合规则的 elementwise、reduction、broadcast、tile 类模式；但遇到 `roll`、`column_stack`、`meshgrid` 这种数据重排布，不能简单写 split/concat/gather，需要绕到 output-driven gather 或手动指针加载。


## 3. T1-1-4：数据重排布类算子

这一题实现的算子包括：

- `roll`
- `column_stack`
- `mode`
- `meshgrid`
- `cartesian_prod`

这组算子的难点是，它们很多不是“对每个元素做数学运算”，而是“根据输出位置去找输入位置”。比如：

- `roll` 本质上是循环位移。
- `column_stack` 是把多个输入按列拼起来。
- `meshgrid` 是把多个 1D tensor 扩展成网格。
- `cartesian_prod` 可以看成 meshgrid 后再 reshape 成二维组合表。
- `mode` 要在某个维度上找众数，并处理 PyTorch 的 tie-breaking 规则。

九齿的 arrangement 层没有 `concat`、`split`、`gather` 这类高级数据移动原语，所以最后采用的核心思路是 output-driven gather：

> 不是从输入出发去 scatter 到输出，而是让每个 output element 根据自己的 offset 计算它应该读取哪个 input element，然后通过手工 pointer arithmetic 和 `ntl.load` 读出来。

这个方案的优势是：

- 避免依赖九齿缺失的 concat/split 原语。
- 和 GPU 并行模型比较贴合，每个输出位置独立计算来源。
- 对数据重排布算子比较通用。

可以重点讲一个例子：

### roll 的讲法

`torch.roll(x, shifts, dims)` 的语义是沿某些维度循环移动。实现时可以对每个输出位置反推输入位置：

```text
input_index = (output_index - shift) mod dim_size
```

如果是多维 tensor，就把线性 offset 还原成多维坐标，对指定维度做这个模运算，再重新转换成输入线性地址。这样不需要真的 split 和 concat。

### mode 的讲法

`mode` 更像 reduction。对每个 row 或指定维度，要找出现次数最多的值，同时还要对齐 PyTorch 的平局规则。我们最后处理的是：

- 对每个候选值统计出现次数。
- 如果 count 更大就更新答案。
- 如果 count 相同，则按 PyTorch 规则处理 value/index。

还遇到一个跨后端问题：某些 Triton 后端对 rank broadcast 比 NVIDIA 更严格。NVIDIA 上能接受的 `[1]` 到 `[1, 1]` 隐式广播，在 MetaX 后端会报 rank mismatch。修复方式是让 output tile 的 rank 和 application 里生成的值保持一致，比如对 shape 为 `(N, 1)` 的输出先 `squeeze(1)` 再 `tile((1,))`。

这段可以体现一个经验：

> 跨后端测试不是走形式。同一段 DSL 生成的 Triton IR，在 NVIDIA、MetaX、MUSA 等后端上的容忍度不同，shape rank、dtype、mask broadcast 这类细节都可能变成真实 bug。

## 4. T1-1-9：计算、索引更新和复杂语义算子

这一题实现的算子包括：

- `frac`
- `scatter_add`
- `multilabel_margin_loss`
- `fractional_max_pool2d`
- `fractional_max_pool3d`

这组题比 T1-1-4 更混合，既有简单 elementwise，也有 atomic update、reduction、loss 语义和 pooling。

### frac

`frac(x)` 的语义是取小数部分，可以写成：

```text
x - trunc(x)
```

这个是最典型的九齿适配场景：arrangement 用通用 elementwise tile，application 直接写数学表达式。

面试里讲它主要是为了说明：

> 简单逐元素算子在九齿里非常自然，几乎就是把 PyTorch 语义翻译成 DSL 表达式。

### scatter_add

`scatter_add` 的难点是多个输入元素可能写到同一个输出位置，所以不能普通 store，需要 atomic add。

核心思路：

- 在 wrapper 层根据 `dim`、`index` 和 shape 计算或展开目标位置。
- kernel 中每个 lane 负责一个 `src` 元素。
- 对目标地址执行 `ntl.atomic_add(out_ptr + flat_index, src)`。

这里有一个重要经验：

> 带副作用的 kernel 要非常小心 autotune。因为 autotune warmup/benchmark 可能多次运行同一个 kernel，如果它反复对同一块 output 做 atomic add，结果会被累加污染。解决方式是固定 block size，并限制 `max_num_configs=1`，避免多个候选配置重复执行副作用 kernel。

这个点面试很值得讲，因为它不是普通语法问题，而是 benchmark/autotune 和有副作用算子之间的语义冲突。

### multilabel_margin_loss

这是这组里最有技术含量的一个。PyTorch 的 `multilabel_margin_loss` 对每个样本，会根据 target 里的正类集合，计算正类和负类之间的 margin loss。

朴素做法需要动态索引，比如 `input[target[j]]`。但九齿 application 层不方便直接做动态 gather。最后采用了 offsets-based 的 O(C²) 方法：

- 对每个 batch row，tile 出一整行 class 向量。
- 用 `input.offsets(1)` 得到当前 lane 对应的 class id。
- 先扫描 target，构造 positive mask。
- 再对每个正类，利用 `where(class_id == target_j, input, 0)` 取出正类分数。
- 对所有负类计算 `max(0, 1 - pos_value + neg_value)` 并求和。
- reduction 为 `none` 时输出每个样本的 loss；`sum/mean` 时再用一个 reduce_sum kernel 做归约。

这段可以简化成面试话术：

> 这个算子的关键是把动态索引改写成向量匹配。原来想写 input[target[j]]，但九齿里动态下标不好表达；我改成让每个 lane 知道自己的 class_id，然后用 `class_id == target_j` 的 mask 从整行里选出目标类别。这样虽然复杂度是 O(C²)，但在比赛测试规模下可行，而且是纯 kernel 路径。

还可以补充 reduce_sum 的坑：

> 做 reduction kernel 时发现 0-d tensor 在九齿生成代码里不一定有 pointer 变量，所以如果需要 `.data_ptr()` 做 atomic add，结果 tensor 用 1D size-1 更稳。

### fractional_max_pool2d/3d

这类算子的语义比较复杂，难点不只是 max pooling，而是 fractional pooling 的窗口起点由随机相位和 output size 决定。要对齐 PyTorch，需要搞清楚 `_random_samples` 不是每个窗口一个随机数，而是每个 batch/channel/dim 的随机相位。

可以这样讲：

> 对 fractional_max_pool，我主要先通过固定样例、连续递增输入和 `return_indices=True` 去反推 PyTorch 的窗口生成规则，然后再按输出位置计算对应输入窗口，做局部 max 和 index 返回。这类算子的重点是先把框架语义摸准，否则 kernel 写出来也不一定和 PyTorch 对齐。

如果面试官追问实现是否完全纯九齿，需要如实回答：

> 不同算子的九齿纯度不同。`frac`、`scatter_add`、`multilabel_margin_loss` 是比较明确的 kernel 实现；pooling 这类我会重点说明实现路径、测试覆盖和已知限制。比赛里我更关注 PyTorch 语义对齐、可验证性和多后端通过。

## 5. 测试与验证

这段可以短讲，但很重要：

> 每个算子我都写了 PyTorch reference 对照测试。基本方式是随机生成输入，用 torch 原生算子算 reference，再用 ntops.torch 的实现算 actual，比较结果、shape、dtype、device 和异常行为。最后不仅跑了新增算子的 targeted tests，也在 NVIDIA 上跑了全仓库 pytest。

验证结果可以这么说：

- T1-1-4 新增算子 targeted tests：89 passed。
- T1-1-4 全仓库：929 passed, 88 skipped。
- T1-1-9 新增算子 targeted tests：31 passed。
- T1-1-9 全仓库：871 passed, 88 skipped。
- 另外在沐曦、天数、摩尔等平台上做了 targeted tests，并把截图补到 PR 描述里。

不要把数字背得太死，但最好记住一个量级：两个赛题新增测试全部通过，全仓库也过了。

## 6. 我遇到的典型问题

面试官通常喜欢问“遇到了什么困难”。可以准备这几个：

1. 九齿 DSL 表达能力边界

   数据重排布类算子缺少 split/concat/gather 这类直接原语，所以不能照着 PyTorch 语义机械翻译。解决思路是 output-driven gather，用输出 offset 反推输入地址。

2. PyTorch 语义细节

   `mode` 的平局规则、`fractional_max_pool` 的随机相位、`scatter_add` 的重复 index 累加，都是容易和 PyTorch 对不齐的地方。

3. Autotune 和副作用 kernel 冲突

   `scatter_add` 这种 atomic update kernel 如果被 autotune 多次 warmup，会污染输出。因此需要固定配置或避免在候选配置间复用被修改的 output。

4. 跨后端差异

   NVIDIA 能过不代表国产 GPU 后端能过。曾经遇到过 rank broadcast 在某后端更严格的问题，也遇到过 MUSA/MetaX 环境里的动态库、torch backend alias、网络拉代码等工程问题。

5. 环境构建复杂

   InfiniCore/ntops/ninetoothed/Triton/PyTorch/CUDA 或国产后端版本之间有强依赖。比赛中不仅是写 kernel，还要处理实际运行环境。

## 7. 面试中的 3 分钟版本

如果面试时间紧，可以直接背这个版本：

> 我最近参加了 InfiniTensor 的算子开发竞赛，做的是 ntops 仓库里 PyTorch 算子的九齿实现。九齿 ninetoothed 是一个基于 Triton 的更高层 DSL，它把 kernel 拆成 arrangement 和 application 两层：arrangement 负责 tile、flatten、broadcast 这类数据排布，application 负责 block 内计算。相比直接写 Triton，它更抽象、更适合快速写 elementwise/reduction 算子，但在复杂数据重排布上表达能力有限。
>
> 我做了两个赛题。T1-1-4 是 roll、column_stack、meshgrid、cartesian_prod、mode 这类数据重排布算子。难点是九齿没有 split、concat、gather 这些直接原语，所以我采用 output-driven gather 的方式：每个输出位置根据自己的 offset 反推出应该读取哪个输入位置，再用指针算术和 `ntl.load` 完成读取。比如 roll 就是对指定维度做 `(out_index - shift) mod size`，meshgrid/cartesian_prod 则是从输出线性坐标反推每个输入向量的索引。
>
> T1-1-9 更偏计算和索引更新，包括 frac、scatter_add、multilabel_margin_loss、fractional_max_pool。frac 是标准 elementwise；scatter_add 用 `ntl.atomic_add` 处理重复 index；multilabel_margin_loss 的关键是把动态索引 `input[target[j]]` 改写成基于 `input.offsets(1)` 的向量 mask 匹配，从而在九齿里实现 loss 计算和 reduction。这里还踩过一个坑：atomic 这种有副作用的 kernel 不能让 autotune 多次 warmup 污染输出，所以需要固定 block size 和配置数。
>
> 最后我给这些算子补了 torch wrapper 和 pytest，用 PyTorch reference 做对照，并在 NVIDIA 上跑了全仓库测试，也在沐曦、天数、摩尔等国产 GPU 环境上做了 targeted tests。这个项目让我对 PyTorch 算子语义、Triton/DSL 编程模型、以及异构 GPU 后端差异都有了更具体的认识。

## 8. 面试追问备答

### Q1：为什么不用直接写 Triton？

可以回答：

> 这个比赛要求是在 ntops/ninetoothed 框架下实现算子，所以主要用九齿 DSL。直接写 Triton 会更灵活，尤其是复杂 pointer arithmetic 和自定义调度；但九齿的目标是提供更高层抽象，让算子实现可以被统一组织、AOT 编译和接入 InfiniCore。我的工作重点是在这个 DSL 的约束下尽量表达 PyTorch 语义。

### Q2：ninetoothed 的优缺点？

优点：

- 对 elementwise、broadcast、reduction 这类规则算子很简洁。
- arrangement/application 分层让代码结构清楚。
- 可以复用框架的 premake、dtype/shape specialization 和测试接入。

缺点：

- 数据重排布表达能力有限。
- 动态索引、动态 shape、复杂 gather/scatter 不如 Triton 直接。
- 不同后端生成代码的兼容性需要额外验证。

### Q3：你觉得最难的算子是哪一个？

建议答 `multilabel_margin_loss` 或 `mode`。

`multilabel_margin_loss` 的亮点在于：

- PyTorch 语义比较复杂。
- 需要处理 target 中正类集合。
- 动态 gather 被改写成 offsets + mask 的向量化实现。
- 还需要额外 reduction kernel 支持 `sum/mean`。

`mode` 的亮点在于：

- 要处理众数统计和平局规则。
- 是 reduction + tie-breaking。
- 暴露了跨 Triton 后端 rank broadcast 差异。

### Q4：这个项目和普通调 API 有什么区别？

可以回答：

> 普通调 API 是直接使用 PyTorch 算子；这个项目是把 PyTorch 算子语义拆开，重新映射到 GPU kernel DSL 中。我需要考虑每个 output element 对应哪个输入、block 内怎么并行、mask 怎么写、重复写怎么 atomic、reduction 怎么做、以及最终怎么用 pytest 和多设备验证和 PyTorch 对齐。

### Q5：如果给你更多时间，你会怎么优化？

可以回答：

> 我会做三件事。第一，把部分 O(C²) 或 output-driven gather 的实现进一步优化成更适合 cache 和 coalescing 的访问模式。第二，更系统地整理九齿 DSL 的能力边界，沉淀一些 reusable pattern，比如 gather、scatter、row reduction。第三，完善性能 benchmark，不只看 correctness，也比较不同 block size、后端、shape 下的性能。

## 9. 简历写法

可以放在项目经历里：

**InfiniTensor 启元 AI 算子开发竞赛 | ninetoothed / Triton / PyTorch / 异构 GPU**

- 基于 ninetoothed/Triton DSL 实现 InfiniTensor `ntops` 两组 PyTorch 算子，覆盖数据重排布、atomic scatter、reduction、loss、pooling 等类型，并向上游提交 PR。
- 对齐 PyTorch 算子语义，编写 torch wrapper、kernel premake 与 pytest，对 `mode` 平局规则、`scatter_add` 原子累加、`multilabel_margin_loss` 向量化实现等边界进行验证。
- 在 NVIDIA、沐曦、天数、摩尔等异构 GPU 环境完成测试，解决 Triton 后端广播差异、MUSA/MetaX 运行时依赖与部署问题。

## 10. 面试表达提醒

- 不要说“我只是让 AI 帮我写的”。可以说“我在实现过程中借助 AI 做方案讨论和 debug，但我负责理解赛题、跑通环境、验证结果、提交 PR 和复盘实现路径”。
- 不要把所有算子都讲太细。优先讲 `roll`、`mode`、`scatter_add`、`multilabel_margin_loss`。
- 被问到不确定细节时，别硬编。可以说“这个细节我需要回代码确认，但整体思路是……”
- 强调自己学到的是“算子语义到 kernel 表达的映射”，这比背代码更重要。
