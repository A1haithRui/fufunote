# 2026 春季启元人工智能大赛九齿算子开发中期报告

## 一、小组信息

- 小组名称：A1haitham
- 小组成员：狄凡瑞
- 联系方式：2829107824\@qq.com
- 参赛赛道：九齿算子开发赛道
- 报告时间：2026/05/31

## 二、所选赛题

|  赛题名称  |     算子1     |          算子2           | 算子3  |          算子4          |          算子5          |
| :----: | :---------: | :--------------------: | :--: | :-------------------: | :-------------------: |
| T1-1-4 |    roll     |      column_stack      | mode |       meshgrid        |    cartesian_prod     |
| T1-1-9 | scatter_add | multilabel_margin_loss | frac | fractional_max_pool2d | fractional_max_pool3d |

## 三、总体进度

截至目前，已完成九齿算子开发仓库的准备工作，并初步梳理了 `ntops` 仓库中算子实现、PyTorch 风格封装与测试代码之间的关系。当前主要进展如下：

- 已 fork 并 clone `InfiniTensor/ntops` 仓库，建立个人开发仓库。
- 已按赛题分别创建开发分支与 worktree：
  - `2026-spring-DiFanrui-T1-1-4`
  - `2026-spring-DiFanrui-T1-1-9`
- 已使用 `uv` 配置本地 Python 开发环境，并以 editable 方式安装本地 `ntops`。
- 已跑通现有算子的 pytest 测试，例如 `abs`、`silu` 等，确认本地九齿算子编译与执行链路可用。
- 已阅读并分析现有算子结构，明确 `src/ntops/kernels`、`src/ntops/torch` 与 `tests` 三部分的职责：
  - `kernels` 中实现九齿算子的排布与计算逻辑；
  - `torch` 中提供 PyTorch 风格的调用封装；
  - `tests` 中通过 PyTorch 原生算子进行结果对拍。

## 四、各赛题进度与完成情况

### 1. T1-1-4

本赛题包含 `roll`、`column_stack`、`mode`、`meshgrid`、`cartesian_prod` 五个算子。目前已完成该赛题的开发分支与环境准备，并对算子类型进行了初步拆解。

当前进展：

- 已确认 `roll` 算子的基本语义为张量环形平移，核心实现可抽象为输出位置到输入位置的索引映射。
- 已初步理解 `roll` 中 `shifts`、`dims`、负数 shift、多维滚动以及 `dims=None` 时 flatten 后滚动的行为。
- 已确认 `meshgrid`、`cartesian_prod` 属于索引生成/组合类算子，需要重点处理输出形状与索引映射。
- 已确认 `column_stack` 属于拼接类算子，需要处理一维输入升维与多输入列方向堆叠。
- 已确认 `mode` 属于规约类算子，除返回 values 外还需要返回 indices，后续需要重点关注 PyTorch 对相同众数值的索引选择规则。

当前完成情况：

| 算子 | 当前状态 | 后续计划 |
| :--: | :-- | :-- |
| roll | 已完成语义调研与实现思路梳理 | 优先实现九齿 kernel、torch wrapper 与 pytest 对拍 |
| column_stack | 已完成算子类型初步分析 | 在 roll 之后实现，重点处理多输入与输出布局 |
| mode | 已完成难点识别 | 后续单独处理规约逻辑与 index 返回规则 |
| meshgrid | 已完成算子类型初步分析 | 结合 indexing 参数设计测试样例 |
| cartesian_prod | 已完成算子类型初步分析 | 结合输出组合规模设计实现与测试 |

### 2. T1-1-9

本赛题包含 `scatter_add`、`multilabel_margin_loss`、`frac`、`fractional_max_pool2d`、`fractional_max_pool3d` 五个算子。目前已完成该赛题的开发分支与环境准备，并完成初步难度评估。

当前进展：

- 已为 T1-1-9 创建独立 worktree 和开发分支，便于与 T1-1-4 并行开发。
- 已初步判断 `frac` 接近逐元素算子，可优先参考现有 `abs`、`floor`/数值类实现思路进行开发。
- 已识别 `scatter_add` 涉及按 index 写入并累加，可能需要处理写冲突与规约/原子累加相关问题。
- 已识别 `fractional_max_pool2d` 与 `fractional_max_pool3d` 涉及池化窗口、随机采样序列与返回 indices，复杂度较高，适合放在后续阶段集中处理。

当前完成情况：

| 算子 | 当前状态 | 后续计划 |
| :--: | :-- | :-- |
| scatter_add | 已完成难点识别 | 后续研究 index 写入与累加策略 |
| multilabel_margin_loss | 已完成初步归类 | 后续对齐 PyTorch 公式与边界行为 |
| frac | 已完成优先级判断 | 作为 T1-1-9 的首个实现目标 |
| fractional_max_pool2d | 已完成难点识别 | 后续参考现有 pooling 类算子实现 |
| fractional_max_pool3d | 已完成难点识别 | 在 2D 版本基础上扩展 |

## 五、测试与验证情况

目前已完成 `ntops` 仓库内部测试链路验证。现有测试的基本方式为：随机生成输入样例，分别调用九齿实现封装后的 `ntops.torch` 接口与 PyTorch 原生接口，并比较两者输出是否一致。

已验证内容：

- `uv` 环境可正常安装本地 `ntops`。
- `pytest` 可正常运行仓库已有测试。
- 现有算子如 `abs`、`silu` 的测试已通过，说明本地九齿算子编译与执行链路基本可用。

后续每实现一个新算子，将同步补充对应测试文件，例如：

- `tests/test_roll.py`
- `tests/test_frac.py`
- `tests/test_meshgrid.py`

测试将覆盖常见形状、不同 dtype、参数边界、负数维度、空参数或特殊参数等情况，并与 PyTorch 原生实现对拍。

## 六、InfiniCore 接入验证情况

根据赛题文档，九齿算子实现后需要接入 InfiniCore Python 接口并运行对应测试。目前已尝试搭建 InfiniCore 验证环境，已完成部分依赖定位与问题排查：

- 已确认 InfiniCore 需要按顺序安装底层 C 库、C++ 库与 Python 包。
- 已安装并更新 `xmake` 至可用版本。
- 已定位 CUDA 路径、Python 开发头文件、九齿 AOT 生成等环境依赖问题。
- 在运行九齿 AOT 生成时遇到 WSL 内存不足导致的 OOM 问题，当前暂缓完整 InfiniCore 编译验证。

下一阶段将优先完成 `ntops` 中目标算子的正确性实现与本地 pytest 测试；在至少一个目标算子通过 ntops 测试后，再继续处理 InfiniCore 接入验证环境。

## 七、当前问题与后续计划

当前主要问题：

- 目标算子尚未完成正式实现，目前处于环境准备、结构理解与实现方案拆解阶段。
- InfiniCore 完整编译验证受到本地 WSL 内存限制影响，需要后续通过降低并行度、增加 swap 或更换机器环境解决。
- 部分算子涉及复杂索引、规约、写冲突或池化采样逻辑，需要逐个对齐 PyTorch 行为。

后续计划：

1. 优先在 T1-1-4 分支实现 `roll`，补齐 kernel、torch wrapper、导出接口与 pytest。
2. 在 T1-1-9 分支优先实现 `frac`，作为逐元素类算子的快速闭环。
3. 对每个完成的算子补充覆盖常见参数与边界情况的对拍测试。
4. 在 ntops 测试通过后，继续恢复 InfiniCore 接入验证流程。
5. 根据实现进度，逐步推进 `meshgrid`、`column_stack`、`cartesian_prod`、`scatter_add` 等算子。

## 八、参考资料

- 2026 春季启元人工智能大赛赛题文档
- 九齿算子开发文档
- `InfiniTensor/ntops` 仓库现有算子实现与测试
- PyTorch 官方算子行为与接口文档
