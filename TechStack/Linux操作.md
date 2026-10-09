# 常规操作
## zsh
对于个人来说，不管是租卡，还是自己的服务器，一般到手之后先通过oh-my-zsh来装zsh。直接套一下下面指令：
```bash
apt update && apt install -y zsh git curl wget # 安装zsh
sh -c "$(wget -O- https://gitee.com/mirrors/oh-my-zsh/raw/master/tools/install.sh)"  # 安装oh-my-zsh, 输入y
# 自动补全 + 语法高亮
git clone https://gitee.com/mirrors/zsh-autosuggestions.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions 
git clone https://gitee.com/mirrors/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
# 启用插件
sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting extract)/' ~/.zshrc
# 生效
source ~/.zshrc
## 然后是p10k的主题, 可选
git clone --depth=1 https://gitee.com/chenxi-mao/powerlevel10k.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/themes/powerlevel10k
sed -i 's/ZSH_THEME="robbyrussell"/ZSH_THEME="powerlevel10k\/powerlevel10k"/' ~/.zshrc
source ~/.zshrc
```
## Neovim
不管是主机的linux系统还是wsl，如果终端链接的话，用neovim写代码还是很方便，如果用apt来安装，版本又很旧，所以还是快速去github上clone最新版的代码，然后编译安装即可。
```bash
# 1. 卸载 apt 版 nvim，如果没有就不管  
sudo apt remove neovim  
sudo apt autoremove
# 2. 安装编译依赖  
sudo apt update  
sudo apt install -y ninja-build gettext cmake unzip curl build-essential git
# 3. 拉源码  
cd ~  # 选择一个文件夹，可以放在~，也可以单独创建位置 
git clone https://github.com/neovim/neovim.git  
cd neovim
# 4. checkout稳定版，然后编译安装
git checkout stable
make CMAKE_BUILD_TYPE=RelWithDebInfo  # 编译
sudo make install # 安装
# 5. 安装完检查
nvim --version  
which nvim
# 正常输出应该是/usr/local/bin/nvim

### 如果是mac就很省事：
brew update
brew install neovim
nvim --version
which nvim
```
然后是lazyvim和自己的配置，快速装一个lazyvim可以省去安装麻烦插件的流程，而自己的配置在vim相关部分。
```bash
# lazyvim这里注意一下路径，一般从git上clone到.config/的nvim文件夹下，就不用自己创建nvim文件夹了
cd ~/.config
git clone https://github.com/LazyVim/starter nvim
rm -rf nvim/.git
# 然后启动就自动安装了
```
然后再把init.lua和lua文件夹下的其他文件替换为自己的配置文件即可。目前的配置保存在个人github代码库`https://github.com/DiFanrui/nvim-config-by-DiFanrui.git`，可以直接clone到nvim文件夹(直接删掉旧的nvim再覆盖就行，lazyvim都不用装)
```bash
git clone https://github.com/DiFanrui/nvim-config-by-DiFanrui.git nvim
# 因为这个仓库是私密的，所以用ssh更方便
git clone git@github.com:DiFanrui/nvim-config-by-DiFanrui.git
```
## python环境相关
一般机子如果默认带miniconda3，那按理来说是自动在base环境的。可以直接用conda，如果刚才安装zsh后`conda init`不生效，那就加一个`conda init zsh`，这样就可以默认在终端激活base环境。
## 大模型环境
### 版本
Autodl：a800；python3.12；cuda12.8；torch2.8.0；
有卡启动后可以直接通过`python -c "import torch; python(torch.cuda.is_available())"`来验证cuda和pytorch版本。
### Flash-attn
如果直接用pip安装，哪怕有ninja，编译也会很慢(曾经tiny-cuda-nn带来的教训)。所以这里直接下载轮子来安装。[https://github.com/Dao-AILab/flash-attention/releases](https://github.com/Dao-AILab/flash-attention/releases)
查看cxx11abi是否为True，通过`import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)`的输出。
然后根据我的环境，下载`flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64`。
上传后，通过指令`pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`
### VeRL
参考博客为：[知乎-flash attention安装教程](https://zhuanlan.zhihu.com/p/655077866)
```Bash
git clone https://github.com/volcengine/verl.git
# 速度慢的话可切换镜像源, 不过一般直接本地clone好再上传
# git clone https://mirror.ghproxy.com/https://github.com/volcengine/verl.git
cd verl

pip install -e .
pip install -e .[vllm] # 如果在zsh里, 方括号会被吞, 使用pip install -e ".[vllm]"
```

# 其它
### 切换cuda的toolkit
不管是切换哪个版本,首先需要把对应的版本下载到服务器上，一般是`usr/local/`的路径下。
然后在bashrc中修改一下：
```bash
nano ~/.bashrc
```
添加以下语句：
```txt
export -p |gerp cuda 找到需要修改的地方进行替换

export CUDA_HOME="/usr/local/cuda-xx.x"
export LD_LIBRARY_PATH=":/usr/local/cuda-xx.x/lib64"
export PATH="/usr/local/cuda-xx.x/bin:/home/iim321/anaconda3/condabin:/usr/local/cuda-xx.x/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"

```
### 添加环境变量
```
env
```
上述指令可以直接查看`~/.bashrc`的内容，也就是我们的环境变量。
如果要修改环境变量，一般采用两种方法：
1. 在文件中修改：
```bash
nano ~/.bashrc
```
在该文件末尾添加：
```bash
export VARIABLE_NAME=value
```
退出后保存文件，运行下述命令：
```bash
source ~/.bashrc
```
2. 在命令行临时加入（关闭终端后失效）：
使用 `export` 命令：
```
export VARIABLE_NAME=value
```
运行`echo`命令查看变量值：
```bash
echo $VARIABLE_NAME
```
如果想在训练的时候指定gpu，可以直接在运行指令前面加上CUDA_VISIBLE_DEVICES=gpu_number
### **通用快捷键**
- **删除当前行的内容**：
    - **`Ctrl + U`**：从当前光标位置删除到行首的内容。
    - **`Ctrl + K`**：从当前光标位置删除到行尾的内容。
    - **`Ctrl + A`**：将光标移动到行首，然后按 **`Ctrl + K`** 可以删除整行内容。
    - **`Ctrl + E`**：将光标移动到行尾，然后按 **`Ctrl + U`** 可以删除整行内容。
## 脚本处理文件数量超额
不管是python还是shell，如果某批次打开的文件/图片数量超过了终端默认的限额（1024），则需要在终端加上一条指令：
```shell
ulimit -n 2048 # 自己定义处理文件数量上线
```
每个终端都需要执行一次，否则就添加进环境变量。
## python运行时需要把文件添加到工作区
在终端输入：
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```
