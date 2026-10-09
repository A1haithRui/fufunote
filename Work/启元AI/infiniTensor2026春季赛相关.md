# 赛题文档
[2026 春季启元人工智能大赛赛题](https://icn8oz7pxzoz.feishu.cn/wiki/C8ehwUdHViw8Iek4tQmcGPPpnFf)
[九齿文档](https://gxtctab8no8.feishu.cn/wiki/RSZSwpc9siOEXrkUoVyc2lPtnX3)
## 训练赛道
具体一下，训练框架，Generator 抽象选题的链接为：[https://gxtctab8no8.feishu.cn/docx/UkcYdCTLgoYg9YxabfFc631hnzf](https://gxtctab8no8.feishu.cn/docx/UkcYdCTLgoYg9YxabfFc631hnzf)
### 训练系统模块（2-2-X系列）
总奖金：￥15000（奖金额度为**含税**金额）
- **优胜规则：**
    - 具体规则可见对应的[赛题文档](https://gxtctab8no8.feishu.cn/wiki/GNQZww29riHNBlk0iAOcTFBXnDT#share-T8UvdLLYko5p06xfc0UcSr70nwb)。
- **评分细则：**
    - 具体规则可见对应的[赛题文档](https://gxtctab8no8.feishu.cn/wiki/GNQZww29riHNBlk0iAOcTFBXnDT#share-T8UvdLLYko5p06xfc0UcSr70nwb)。
- **提交方式：**
    - 分支命名需命名为：`2026-spring-<GitHub ID>-<赛题号>`，例如 ID 为 ABC 的选手选择并提交 T2-2-1 赛题，则该分支需命名为：`2026-spring-ABC-T2-2-1`
    - PR 命名规范：`[2026春季][赛题号] GitHub ID`
    - PR 描述中需附上：
        - 该赛题在每个实现的平台上的输出或测试结果的截图
        - 赛题报告
        - 署名的 `HONOR_CODE.md` 和 其中提及的 `REFERENCE.md`
    - 其他要求会在之后进行补充与细化
## 训练赛题内容：
一、题目背景
在深度学习训练与推理框架中，随机数生成器（Generator）是支撑参数初始化、随机采样、Dropout、噪声注入等功能的重要基础设施。当前框架中相关能力尚不完善，缺少统一的 Generator 抽象、后端实现以及全局随机种子管理机制，导致随机相关算子的行为与主流框架存在差距，也难以满足后续功能扩展需求。
当前框架在随机数相关能力上仍不完善，主要表现为：
-  缺少统一的 Generator 抽象，随机数状态、种子设置、状态获取与恢复等能力未形成标准接口； 
-  CPU 与 CUDA 后端缺少统一的 Generator 实现，不同设备上的随机行为缺乏一致的调用方式； 
-  缺少全局默认 Generator 机制，随机算子在未显式传入 generator 时无法方便地使用当前设备的默认随机源； 
-  缺少统一的全局随机种子控制入口，导致参数初始化、Dropout 等随机算子在多次运行间难以稳定复现； 
-  随机数相关能力与主流框架（如 PyTorch）存在差距，不利于后续算子扩展、模型对齐与测试验证。
为补齐这一基础能力，本题目要求参赛者参考主流深度学习框架的设计思路，完成一套可扩展的随机数生成器基础设施，实现 CPU 与 CUDA 后端的 Generator，并支持基于 Generator 的随机数生成与种子控制能力。

二、题目目标
参赛者需要围绕框架内的随机数生成体系，完成以下几个方面的工作：
1.  设计并实现统一的 Generator 抽象接口，支持随机数状态管理、种子设置、状态获取与恢复等基础能力。 
2.  分别实现 CPU 与 CUDA 后端对应的 GeneratorImpl，使不同设备上的随机数生成具备统一的调用方式。 
3.  建立全局随机种子控制机制，支持通过统一入口固定随机种子，并使随机相关算子的行为可复现。 
4.  改造随机数相关算子，使其在未显式传入 generator 参数时，能够自动使用当前设备上的默认全局 Generator。 
5.  验证实现结果与主流框架在基本行为上的一致性，包括随机性、可复现性以及跨设备下的接口统一性。

三、任务拆解
1. Generator 抽象与状态管理设计
在框架中设计统一的 Generator 抽象层，用于屏蔽不同设备后端的随机数实现差异。该抽象应支持：
-  设置随机种子，如 ManualSeed() / SetCurrentSeed()； 
-  获取当前种子或初始种子，如 Seed() / InitialSeed()； 
-  获取当前随机数状态，如 GetState()； 
-  恢复随机数状态，如 SetState(state)； 
-  查询所属设备类型； 
-  提供必要的线程安全或状态访问保护机制。 
实现时需注意区分 seed 与 state：seed 只用于初始化随机序列，state 表示随机序列当前推进到的位置。随机算子每次使用 Generator 后，应推进其内部状态，避免重复生成相同随机序列。
推荐设计思路：
-  提供用户侧可持有的 Generator 句柄类； 
-  底层以 GeneratorImpl 作为多态实现基类； 
-  CPU / CUDA 分别派生对应实现； 
-  公共接口不暴露 std::mt19937、curand、Philox 等后端细节； 
-  为后续其他平台 Generator 接入预留扩展空间。 
2. CPU / CUDA 后端 Generator 实现
基于现有设备后端，分别实现：
- CPUGeneratorImpl
- CUDAGeneratorImpl
两者应保持统一接口语义，但内部状态组织方式可以不同，且 不要求 CPU 与 CUDA 在相同 seed 下生成逐元素一致的随机结果。
实现时需重点关注：
-  CPU 与 CUDA 随机数状态如何组织与保存； 
-  CUDA 侧应按设备维度维护独立 Generator； 
- GetState() / SetState() 的状态序列化、反序列化与格式校验；(state 是否来自同类型 Generator) 
-  多次随机调用后，随机序列是否连续推进； 
-  同一 seed、同一设备、同一调用顺序下结果是否可复现； 
-  显式传入 Generator 与使用默认 Generator 时，行为是否符合预期。 
CUDA 后端建议在状态中考虑 seed、offset、counter 等信息，避免不同 kernel 调用之间随机序列重叠。此部分需要保证功能正确、语义清晰、状态可管理、接口一致，为后续随机算子和分布式扩展提供基础。
3. 默认 Generator 与统一随机种子入口
建立框架级默认 Generator 管理机制，支持两种使用方式：
-  用户显式传入某个 Generator； 
-  用户不传入 Generator，系统自动使用当前设备上的默认 Generator。 
该机制应包括：
-  CPU 默认全局 Generator； 
-  各 CUDA 设备对应的默认 Generator； 
-  获取默认 Generator 的统一入口； 
-  统一的全局随机种子设置入口； 
-  设置全局随机种子时，同步更新默认 Generator 的初始状态。 
要求：
-  多次获取同一设备默认 Generator 时，应返回同一随机状态来源； 
-  不同 CUDA 设备的默认 Generator 应相互独立； 
-  用户显式传入 Generator 时，必须使用用户指定 Generator； 
-  未传入 Generator 时，才使用当前设备默认 Generator； 
-  设置统一 seed 后，参数初始化、Dropout、随机采样等行为应可稳定复现。 
这一部分是本题的核心验收点。随机数系统最终服务于训练可复现性，因此必须通过统一入口稳定控制随机行为。
4. 随机相关算子接入改造
改造框架中已有的随机相关使用，使其接入 Generator 机制。建议至少覆盖两类场景：
-  初始化类随机算子，如 uniform、normal、kaiming 等； 
-  训练过程中的随机算子，如 dropout、rand、randn等。 
改造后的算子应满足：
-  支持显式传入 Generator； 
-  未传入 Generator 时，自动使用当前设备默认 Generator； 
-  随机结果受统一 seed 入口控制； 
-  多次调用会推进 Generator 状态； 
-  同一 seed、同一调用顺序下结果可复现。 
不要求一次性改造所有随机算子，但应至少完成一个初始化类算子和一个训练期随机算子，以证明 Generator 机制能够贯通框架层与算子层。
5. 测试与对齐验证
为验证实现结果，需要补充系统化测试。测试重点应放在“语义是否正确”和“是否可复现”，而不是与 PyTorch 逐元素数值一致。
建议至少覆盖以下内容：
（1）接口一致性测试
验证 CPU / CUDA Generator 是否支持统一接口，包括：
-  seed 设置与获取； 
-  state 获取与恢复； 
-  设备类型查询； 
-  默认 Generator 获取； 
-  显式 Generator 与默认 Generator 两条调用路径。 
（2）种子可复现测试
验证统一随机种子入口是否生效，包括：
-  同一 seed 下，多次运行参数初始化结果一致； 
-  同一 seed 下，多次运行 Dropout mask 一致； 
-  不同 seed 下结果应发生变化； 
-  同一 seed、同一调用顺序下结果应一致。 
（3）状态恢复测试
验证 GetState() / SetState() 是否真正恢复随机序列，包括：
-  保存 state； 
-  继续生成一段随机结果； 
-  恢复 state； 
-  再次生成随机结果； 
-  验证恢复后的结果与原序列对齐。 
CPU 后端必须覆盖该测试；CUDA 后端若已实现，也应覆盖。
（4）默认 Generator 行为测试
验证随机算子是否正确使用默认 Generator，包括：
-  不传 Generator 时，是否使用当前设备默认 Generator； 
-  CPU tensor 是否使用 CPU 默认 Generator； 
-  CUDA tensor 是否使用对应 CUDA 设备默认 Generator； 
-  显式传入 Generator 时，是否不会误用默认 Generator。 
（5）主流框架语义对齐验证
可选取典型场景与 PyTorch 进行语义对比，包括：
-  手动设置 seed 后结果可复现； 
-  Dropout 等随机运算在同 seed 下可复现； 
-  随机张量生成接口支持显式 Generator 与默认 Generator； 
-  state 保存与恢复语义一致。 
需要明确：本项目不要求底层随机算法与 PyTorch 逐 bit 一致，也不要求 CPU 与 CUDA 随机结果彼此一致。验收重点是接口语义一致、可复现逻辑一致、默认 Generator 行为一致。

四、提交报告要求
除代码外，参赛者应提交以下内容作为技术报告：
1. 功能正确性验证
需要提供上述 Generator 基础设施功能验证结果
2. 对齐性与行为分析报告
需要说明本项目实现与主流框架的对齐情况。建议从以下角度展开：
-  参考了哪些主流框架设计； 
-  当前实现与 PyTorch 在接口语义上的对应关系； 
-  哪些行为做到基本一致； 
-  哪些行为暂未完全对齐，以及原因分析； 
-  当前实现范围与后续可扩展方向。 
此部分重点不是要求一切细节完全一致，而是要求留档明确设计思路，能够清晰说明：
-  自己实现了什么； 
-  为什么这样设计； 
-  与主流框架相比的设计异同及原因； 
-  后续演进路径是什么。 
3. 测试与可复现性说明
需要提供完整测试脚本与说明，保证 reviewer 可在相同环境下复现主要结果。

五、验收要求
验收要求
项目需满足以下基本要求：
- 代码以 PR 形式提交，结构清晰，具备基本可 review 性； 
- 完成统一 Generator 抽象设计； 
- 实现 CPU Generator，CUDA Generator 建议实现； 
- 建立默认 Generator 管理机制； 
- 提供统一的全局随机种子设置入口； 
- 支持默认 Generator 与显式 Generator 两种使用方式； 
- 支持 Generator 状态的获取与恢复（get_state / set_state）； 
- 至少改造一类初始化算子和一类框架内随机数生成调用，使其接入 Generator； 
- 提供测试或运行日志，验证随机行为具备基本可复现性（同 seed 一致、不同 seed 不同、状态恢复有效）。 
加分项
在满足验收要求的基础上，具备以下内容可作为加分项：
- 设计清晰，接口与实现分层合理，便于后续扩展
- 测试覆盖充分，包含 seed、state、默认/显式 generator、跨设备等关键场景； 
- 调用处改造完全，接口风格统一； 
- 与 PyTorch 的接口语义和行为分析完整，报告质量高
- PR 经过完整 review 流程，达到可合入标准。 
## 推理赛道2
这道服务优化的题目链接为：
[题目链接](https://gxtctab8no8.feishu.cn/wiki/YN34wVFKBiv3gHksOwRcNEOsnB6)
### 服务能力优化（2-1-2）
测试模型：8B模型、70B模型
目标：多并发、长文本服务场景中，输出吞吐和总吞吐量提升；小规模性能无明显下降，有提升可加分。

测试矩阵：

| 模型  | 并发数      | 输入长度（tokens）  | 输出长度（tokens）    |
| --- | -------- | ------------- | --------------- |
| 8B  | 1, 4     | 32, 256, 4096 | 256, 1024, 4096 |
| 8B  | 16, 64   | 32, 256       | 256, 1024, 4096 |
| 70B | 1, 4, 16 | 32, 256       | 256, 1024, 2048 |

参考服务指令
```Python
python python/infinilm/server/inference_server.py --device nvidia --model=/<path-to>/9g_8b_thinking_llama/ --temperature 1.0 --top-p 0.8 --top-k 1 --port 8102 --tp 1 --max-new-tokens 4096 --num-blocks 6144 --max-batch-size <max_con> --enable-graph --enable-paged-attn --attn flash-attn --ignore-eos
```
参考测试脚本
```Bash
# 客户端
unset http_proxy https_proxy all_proxy ALL_PROXY  
batch_size_list=(1 ) # (1 4 16 64)
random_input_len_list=(32 256 4096 ) # (32 256 4096)
random_output_len_list=(256 1024 2048 4096 ) # ( 256 1024 2048 4096 )

# 外层循环：遍历 max-batch_size_list
for batch_size in "${batch_size_list[@]}"; do
    # 中层循环：遍历 random-input-len
    for random_input_len in "${random_input_len_list[@]}"; do
        # 内层循环：遍历 random-output-len
        for random_output_len in "${random_output_len_list[@]}"; do
            sleep 1
            seed=$(date +%s)
            
            # 压力测试参数
            num_prompts=$(($batch_size * 1))
            request_rate=inf  # 或用 1000000.0 但推荐 inf 表示爆发模式

            # 定义输出文件名变量
            output_file="infinilm_client_A100_model=9g_8b_thinking_llama_batch_size=${batch_size}_random_input_len=${random_input_len}_random_output_len=${random_output_len}_seed=${seed}.json"

            # 打印当前组合，同时输出到屏幕并追加到独立文件
            echo "Running with batch_size=${batch_size}, random_input_len=${random_input_len}, random_output_len=${random_output_len}" | tee -a "$output_file"

            # 执行基准测试命令，输出追加到同一个独立文件
            vllm bench serve \
                --backend openai-chat \
                --model 9g_8b_thinking \
                --tokenizer /data-aisoft/mechdancer/models/9g_8b_thinking_llama \
                --endpoint /v1/chat/completions \
                --request-rate "$request_rate" \
                --port 8102 \
                --seed "$seed" \
                --num-prompts "$num_prompts" \
                --max-concurrency "$batch_size" \
                --random-input-len "$random_input_len" \
                --extra-body "{\"max_tokens\": ${random_output_len}}" >> "$output_file" 2>&1
            
        done
    done
done
```
## 算子开发赛道
### **九齿算子开发（1-1-X系列）**
九齿算子开发系列共提供 50 个算子，共打包成 **10 个赛题**。选手可自由选择赛题数量，但每个赛题报名有上限。
- 赛题先到先得：
    - T1-1-1 至 T1-1-3 为每题最多 10 人
    - T1-1-4 至 T1-1-7 为每题最多 15 人
    - T1-1-8 至 T1-1-10 为每题最多 20 人
- 每位选手最多同时只能报名 6 道赛题
- **优胜规则：** 赢者通吃，即每个算子赛题仅有一支队伍获奖。评优规则为：
    - 保证正确性（必须）
    - 均正确的情况下，根据提交时间评定，提交时间最早的队伍胜出
    - **若有任何疑惑、不确定的界定范围或特殊情况，请及时咨询赛题组**
- **评分细则：**
    - GPU 的实现在 **英伟达 + 至少 2 种指定国产 GPU 平台（天数、沐曦和摩尔）**上分别参与性能评分，英伟达平台满分为 40 分，每种国产平台满分为 30 分，总分 100 分
    - 每个平台上性能最优者获得该平台该算子性能的全部分值，其余选手按与第一名的性能比例获取对应比例的得分
    - 九齿 ntops 不直接提供性能测试接口，选手可自行编写对比测试，评判时，评委会使用隐藏测例进行性能评估
    - 九齿实现的算子在性能评定时会关闭自动调优。详情可见 [FAQ-Q2](https://gxtctab8no8.feishu.cn/wiki/V5hkwjLmJiuE0EkFvmjcKfW4nwg#share-E1WOdJrfSoxvMXxKwxPcf5Ignue)
    - 使用九齿开发和性能评定时将以最新的 master 分支为准，因此请持续关注和更新九齿
    - **赛题综合得分计算**：同赛题中的所有算子得分同权累加。
合理制定策略，覆盖更多的算子可以增大赢取奖金的机会！
- **提交方式：**九齿算子实现需向 [ntops](https://github.com/InfiniTensor/ntops) 仓库提交 PR。
    - 分支命名需命名为：`2026-spring-<GitHub ID>-<赛题号>`，例如 ID 为 ABC 的选手选择并提交 T1-1-1 赛题，则该分支需命名为：`2026-spring-ABC-T1-1-1`
    - PR 命名规范：`[2026春季][赛题号] GitHub ID`
    - PR 描述中需附上：
        - 该赛题在每个实现的平台上的测试结果截图。
        - 署名的 `HONOR_CODE.md` 和 其中提及的 `REFERENCE.md`
[九齿文档](https://gxtctab8no8.feishu.cn/wiki/RSZSwpc9siOEXrkUoVyc2lPtnX3)
关于使用九齿完成算子并接入 InfiniCore 的流程：
1. 本地拉取九齿算子库（[ntops](https://github.com/InfiniTensor/ntops)），并安装 `pip install -e .`
2. 在 ntops 中完成算子的实现；
3. 九齿算子接入 InfiniCore Python 接口可以参考：[InfiniCore/python/infinicore/nn/functional/silu.py at main · InfiniTensor/InfiniCore](https://github.com/InfiniTensor/InfiniCore/blob/main/python/infinicore/nn/functional/silu.py)
4. 在 InfiniCore 中运行对应算子的 Python 级测试（uncomment 掉 infiniCore 的算子调用部分），若能通过，则证明接入与实现正确。

# 个人选题

|  赛题名称  |     算子1     |          算子2           | 算子3  |          算子4          |          算子5          |
| :----: | :---------: | :--------------------: | :--: | :-------------------: | :-------------------: |
| T1-1-4 |    roll     |      column_stack      | mode |       meshgrid        |    cartesian_prod     |
| T1-1-9 | scatter_add | multilabel_margin_loss | frac | fractional_max_pool2d | fractional_max_pool3d |
测试指令
```bash
# 1-1-4
python -m pytest -sv tests/test_cartesian_prod.py tests/test_column_stack.py tests/test_meshgrid.py tests/test_mode.py tests/test_roll.py
# 1-1-9
python -m pytest -sv tests/test_frac.py tests/test_scatter_add.py tests/test_multilabel_margin_loss.py tests/test_fractional_max_pool.py
```
不同设备的处理方式：
```bash
# 进入实例后先检查，暂时不要安装或覆盖任何包：
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("path:", torch.__file__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

try:
    import triton
    print("triton:", triton.__version__, triton.__file__)
except Exception as e:
    print("triton unavailable:", repr(e))
PY

python -m pip list | grep -Ei \
'torch|triton|musa|maca|metax|iluvatar|corex|ninetoothed'

# 如果厂商 PyTorch 和 Triton都存在，再创建一个继承系统包的环境，避免覆盖厂商栈
cd /data
python -m venv ntops-env --system-site-packages
source ntops-env/bin/activate

python -m pip install pytest pytest-cov
python -m pip install ninetoothed==0.25.0 --no-deps
```
拉取1-1-4的PR：
```
git clone https://github.com/InfiniTensor/ntops.git ntops-pr88
cd ntops-pr88
git fetch origin pull/88/head:pr-88
git switch pr-88
python -m pip install -e . --no-deps
```
测试 T1-1-4：
```
python -m pytest -sv tests/test_cartesian_prod.py tests/test_column_stack.py tests/test_meshgrid.py tests/test_mode.py tests/test_roll.py
```
然后测试 T1-1-9：
```
cd /data
git clone https://github.com/InfiniTensor/ntops.git ntops-pr89
cd ntops-pr89
git fetch origin pull/89/head:pr-89
git switch pr-89
python -m pip install -e . --no-deps

python -m pytest -sv tests/test_frac.py tests/test_scatter_add.py tests/test_multilabel_margin_loss.py tests/test_fractional_max_pool.py
```
查看设备信息：
```bash
python - <<'PY'
import torch, triton
from importlib.metadata import version

print("Device:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("Triton:", triton.__version__)
print("NineToothed:", version("ninetoothed"))
print("CUDA available:", torch.cuda.is_available())
PY

printf "Commit: "; git rev-parse HEAD
```