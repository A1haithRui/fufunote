基本上现在用conda管理虚拟环境还是有些吃力的。看到开源社区很多都在用uv，试了一下特别方便，甚至可以一个代码仓库用指定的一套环境，都不会把缓存带到全局……不过uv更倾向于现代python的轻便快捷工作流，一扯到底层环境，像cuda、gcc这些，还得靠老演员conda管理。现在圈子貌似更倾向于底层环境用micromamba，完全兼容conda的依赖且速度比conda快n倍~
所以现在就用uv来管python库，mamba来管系统库，就行。
## micromamba
### 安装：
macOS / Linux
```shell
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```
Windows PowerShell：
```powershell
Invoke-Expression ((Invoke-WebRequest -Uri https://micro.mamba.pm/install.ps1 -UseBasicParsing).Content)
```
安装脚本会把 micromamba 放到本地路径，并询问是否初始化 shell；初始化后才能方便地 `micromamba activate`。官方说明里也强调，micromamba 是静态链接的 C++ 可执行文件，不需要 `base` 环境，也不会自带默认 Python。
> 至于这套初始化，还是很方便，不会再有那个很笨重的base，只需要在使用的时候进入env即可。
### 使用
安装好后`~`目录下就有一个`micromamba`的文件夹。当然也可以自己指定位置：`export MAMBA_ROOT_PREFIX=~/micromamba`。这个目录本质上就类似于以前的：
```
~/anaconda3
~/miniconda3
```
里面也会有：
```
envs/      # 虚拟环境
pkgs/      # 包缓存
conda-meta/
```
所以它其实还是那套 conda-forge/conda package 生态，只不过是超轻量 C++ 实现。
使用也很简单，把之前那套conda换成micromamba就行。
> 这里插一嘴，micromamba和mamba还不太一样，mamba虽然也是轻量化了conda，但是本质上还是依赖已有的conda环境，而micromamba完全独立，所以别用mamba！！！
```bash
micromamba create -n llm python=3.11 cuda-toolkit
micromamba activate llm

uv pip install torch transformers vllm # 进入虚拟环境就可以用uv管理
```
## uv
uv给我的直观印象是，不用在全局配环境，一个项目带一套uv来管理就行。以前conda创建全局环境，然后在项目中激活；uv只需要在项目里创建一个`.venv`文件夹就行。
这里直接放一个手册去参考：[uv-zh-cn](https://hellowac.github.io/uv-zh-cn/pip/environments/)。
### 安装
macOS / Linux
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
wget -qO- https://astral.sh/uv/install.sh | sh # 如果没有curl就用wget
```
Windows PowerShell：
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
## 使用
快速安装pytorch和Triton。
这里是uv最爽的一步，因为uv会自动处理cuda版本匹配。
先在你的项目目录下创建一个uv环境，然后激活它：
```bash
uv venv --python=3.14
source .venv/bin/activate
```
之后再终端前面会出现(项目名)的这种格式。这表明我们成功激活环境。
之后安装Pytorch，uv有一个专门的参数，它非常智能，能够自动检测系统的CUDA版本并安装对应的Pytorch。
```bash
uv pip install torch torchvision torchaudio --torch-backend=auto
```
这一步实际上非常快，因为uv是并发下载的。安装好pytorch之后，那么再安装Triton。因为Triton已经发布在PyPI上了，直接装就行。
```
uv pip install triton
```