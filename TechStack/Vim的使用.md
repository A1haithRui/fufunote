这里的内容都是通过博客：[Vim操作命令大全](https://blog.csdn.net/weixin_37657720/article/details/80645991)学到的。
# 1. 关于Vim

可以通过运行vimtutor查看手册，如果是中文环境，那就用vimtutor zh来查看。
## 1.1 Vim的模式
- 正常模式：用快捷键来使用命令的模式。要么按快捷键，要么按`:`进入命令行输入命令。
- 插入模式：可以正常输入文本的模式。正常模式下按i、a、o可以进入。(i:当前位置插入；a:下一个字符处插入；o:新一行插入)
- 可视模式：可以选择文本的模式。正常模式下按v，V，ctrl+v进入。（v:移动光标可以选择文本；V:移动光标选择行；ctrl+v:移动光标选择块）
- 替换模式：正常模式下按R进入。
# 2. 启动Vim
正常情况下直接vim [filename]即可，别的高端操作就不记录了。需要再查。
# 3. 文件操作
:e [filename]    关闭当前编辑的文件，创建并打开新文件(如果已经有新的文件，就从行尾部开始编辑)。如果当前文件没保存，vi会警告。
:e! [filename]    强制关闭当前修改的文件，编辑新的文件。
:e 数字 [filename]    从第n行开始编辑[filename]。
:q    退出，如果当前文件没保存，会警告。
:q!    强制退出，不保存
:w    保存当前修改
:wq    保存当前修改并退出 ，也可以用ZZ。
:saveas [newfilename]    另存为
:Vex    垂直分割一个窗口来浏览文件系统，可以用:q来退出。
......
# 4. 光标的移动
## 4.1 基本移动
以下移动都是在normal模式下。
- h或退格    左移一个字符；
- l或空格    右移一个字符；****
- j    下移一行；
- k    上移一行；
- gj    移动到一段内的下一行；
- gk    移动到一段内的上一行；
- - gg    到文件头部。
- G    到文件尾部。
- +或Enter    把光标移至下一行第一个非空白字符。
- -    把光标移至上一行第一个非空白字符。
- w    前移一个单词，光标停在下一个单词开头；
- W    移动下一个单词开头，但忽略一些标点；
- e    前移一个单词，光标停在下一个单词末尾；
- E    移动到下一个单词末尾，如果词尾有标点，则移动到标点；
- b    后移一个单词，光标停在上一个单词开头；
- B    移动到上一个单词开头，忽略一些标点；
- ge    后移一个单词，光标停在上一个单词末尾；
- gE    同 ge ，不过‘单词’包含单词相邻的标点。
- (    前移1句。
- )    后移1句。
- {    前移1段。
- }    后移1段。
- fc    把光标移到同一行的下一个c字符处
- Fc    把光标移到同一行的上一个c字符处
- tc    把光标移到同一行的下一个c字符前
- Tc    把光标移到同一行的上一个c字符后
- ;    配合f & t使用，重复一次
- ,    配合f & t使用，反向重复一次
上面的操作都可以配合n使用，比如在正常模式(下面会讲到)下输入3h， 则光标向左移动3个字符。
## 4.2 翻屏
- ctrl+f    下翻一屏。
- ctrl+b    上翻一屏。
- ctrl+d    下翻半屏。
- ctrl+u    上翻半屏。
- ctrl+e    向下滚动一行。
- ctrl+y    向上滚动一行。
- n%    到文件n%的位置。
- zz    将当前行移动到屏幕中央。
- zt    将当前行移动到屏幕顶端。
- zb    将当前行移动到屏幕底端。
## 4.3 标记

目前感觉用不上，就先不记了。
# 5. 插入文本
## 5.1 基本插入

比较重要的是：
- i    在光标前插入
>一个小技巧：按8，再按i，进入插入模式，输入=， 按esc进入命令模式，就会出现8个=。这在插入分割线时非常有用，如30i + esc就插入了36个+组成的分割线。
- I    在当前行第一个非空字符前插入
- gI    在当前行第一列插入
- a    在光标后插入
- A    在当前行最后插入
- o    在下面新建一行插入
- O    在上面新建一行插入
## 5.2 改写插入

......
# 6. 剪切，复制和寄存器
## 6.1 剪切和复制粘贴

- [n]x    剪切光标右边n个字符，相当于d[n]
- [n]X    剪切光标左边n个字符，相当于d[n]h

- y    复制在可视模式下选中的文本
- yy或者Y    复制整行文本
- y[n]w    复制n个词
- y[n]l    复制光标右边n个字符
- y[n]h    复制光标左边n个字符

- p    在光标之后粘贴
- P    在光标之前粘贴

- d[n]l    删除（剪切）光标右边n个字符
- d[n]h    删除（剪切）光标左边n个字符
- d0    删除（剪切）当前位置到行首的内容
- [n]dd    删除（剪切）n行
## 6.2 文本对象
不常用
## 6.3 寄存器
不常用
但是记住打开指令是：
- :reg
# 7. 撤销和反撤销

有个比较关键的操作，就是撤销和反撤销，可以用：
u    撤销上一步操作。    
Ctrl + r    重做（反撤销）上一步操作。

---
# 8. 查找和批量替换
## 查找
如果想要查找，可以输入`/`，然后紧接想要查找的单词，并敲回车。然后就可以通过`n`和`N`来上下翻找匹配项。
这是提高效率的“黑科技”。如果想看代码中某个变量在其他地方是怎么用的，**不需要手动敲 `/` 再打字**：
- **`*` (星号)**：直接搜索光标下的**当前单词**，并向下跳转。
- **`#` (井号)**：直接搜索光标下的**当前单词**，并向上跳转。
## 替换
如果想要替换某个字符串，需要输入指令。
```vim
:%s/旧词/新词/g
```
`%`代表用于当前整个文件，如果不加就只换当前行；`s`表示substitute(替换)；`g`代表全局，即找到所有的的匹配项。
如果想带替换确认，就把`g`换成`gc`。
# 个人使用习惯

## 快速跳转光标和前后搜索

如果在普通模式下要快速跳转到当前行末尾，就用`$`，如果要快速跳转到当前行首，就用`^`。
如果向后搜索，就用`f`然后接上要搜索的字符，向前的话就用`F`。

## obsidian里的配置文件

如果安装了插件`vimrc support`，有一个比较关键的设置，就是要把系统剪切板和软件内置寄存器的物理隔离统一起来，这样在别的地方复制的东西就可以直接粘贴。我在设置里把配置文件的路径放到了`Plugin/.obsidian.vimrc`里，只有两行：
```bash
set clipboard=unnamed # 取消隔离剪切板
imap jk <Esc>  # jk神教
```
第二行是用来加入jk神教的，感觉很多人都喜欢用jk来快速跳出插入模式，确实比Esc方便一些。就改了键位。

# Neovim安装和Lazyvim的配置
## Neovim安装
当前windows系统，直接去官网下载msi镜像文件，然后安装。目前安装到默认位置，就是`C:\Program Files\Neovim\`。
## 基础配置文件内容
这里安装好的就是空白的neovim，然后自己手动diy了，配置文件一般默认在`%LOCALAPPDATA%\nvim`，即`C:\Users\自己的用户名\AppData\Local\nvim`。
直接在下面创建`init.lua`，然后开始自己编辑。因为后续考虑装lazyvim，所以这里记录一套自己的通用配置，装好lazyvim后把自己的配置补进去即可。
```lua
-- init.lua内容
require("config.options")
require("config.keymaps")
require("config.autocmds")
```
这里索引了三个路径，分别保存重要配置、快捷键映射和自动执行的指令。已经写过一些详细注释了。目前保存在个人的github仓库里：`https://github.com/DiFanrui/nvim-config-by-DiFanrui.git`
## Lazyvim安装和配置
首先把仓库克隆到本地对应路径：
```bash
git clone https://github.com/LazyVim/starter $env:LOCALAPPDATA\nvim
```
然后删掉它的 git 历史（避免后面提交混乱）：
```powershell
Remove-Item $env:LOCALAPPDATA\nvim\.git -Recurse -Force # windows系统用Remove-Item，别的系统就不多说了
```
然后系统就会自动安装，对应的文件夹格式就不写了。

# 使用过程中的碎碎念
## 续行注释和取消
原生vim是启动了续行注释的，上一行如果在注释中的话，那么不管是什么方式做换行都会在下一行对应位置开启注释。没必要关掉这个功能，如果换行后想跳出注释，直接Ctrl + u即可。
## 行Wrap
Lazyvim默认关闭了wrap，所以不管在options里怎么设置都会被覆盖。这里如果要开启wrap，手动set一下吧。
```bash
# 设置wrap
set wrap
# 关闭wrap
set nowrap
```
## 背景主题和不透明度
先说不透明度，如果在windows系统中用，那么不透明度取决于windows终端的不透明度，我的配置文件在：`C:\Users\28291\AppData\Local\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json`里，可以直接在defaults下找到opacity。
然后是背景主题，在plugins下直接创建colorsscheme.lua的文件即可配置，目前写的配置文件为：
```lua
return {
  {
    "catppuccin/nvim",
    name = "catppuccin",
    priority = 1000,
    opts = {
      transparent_background = true, -- 终端透明（很帅）
      integrations = {
        nvimtree = true,
      },
      custom_highlights = function(colors)
        return {
          -- NvimTree 透明
          NvimTreeNormal = { bg = "NONE" },
          NvimTreeNormalNC = { bg = "NONE" },
          NvimTreeWinSeparator = { bg = "NONE" },
          -- 其他可能需要透明的高亮组
          NormalFloat = { bg = "NONE" },
          FloatBorder = { bg = "NONE" },
          TelescopeNormal = { bg = "NONE" },
          WhichKeyNormal = { bg = "NONE" },
        }
      end,
    },
  },
  {
    "LazyVim/LazyVim",
    opts = {
      colorscheme = "catppuccin-mocha",
    },
  },
}

```
# Neovim 的探索
## 关于python的debug
### 插件
Python 调试不是 pyright 负责的。pyright 主要做 LSP，也就是补全、跳转、诊断和类型检查；真正断点调试一般用 `nvim-dap + debugpy`。
LazyVim 里可以直接用 extras 开启调试能力：
```vim
:LazyExtras
```
在里面搜索到之后，按`x`启用：
```text
dap.core
lang.python
```
启用后同步或重启 Neovim：
```vim
:Lazy sync
```
然后用 Mason 安装 Python 调试器：
```vim
:MasonInstall debugpy
```
### 大致使用方法
常用调试快捷键：
```text
<leader>db    打/取消断点
<leader>dc    开始运行 / 继续运行
<leader>dO    Step Over，执行当前行但不进入函数
<leader>di    Step Into，进入当前调用的函数
<leader>do    Step Out，从当前函数跳出
<leader>dt    结束调试
<leader>du    打开/关闭 DAP UI
<leader>dr    打开/关闭调试 REPL
<leader>de    evaluate 当前表达式或选中内容
<leader>da    带命令行参数运行
```
一个基本流程：先打开 `.py` 文件，在想停住的行按 `<leader>db` 打断点，然后按 `<leader>dc` 开始调试。程序停住之后，可以用 `<leader>dO` 单步跳过、`<leader>di` 进入函数、`<leader>do` 跳出函数。
如果项目使用虚拟环境，先在 Python 文件里执行：
```text
<leader>cv
```
选择对应的 `.venv`，再开始调试。这样 debugpy 会使用正确的 Python 环境，不容易出现找不到依赖的问题。
注意：调试器运行的是磁盘上的文件，不是 Neovim buffer 里尚未保存的内容。所以调试或直接运行之前，最好先 `:w` 或 `:update` 保存一下。之前遇到过代码逻辑没问题，但是输出结果不对，最后发现就是改完没保存。
