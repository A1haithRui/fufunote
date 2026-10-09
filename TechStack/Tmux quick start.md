![../../assets/Pasted image 20250531104129.png]
这篇博客是从阮一峰的博客[ruanyifeng](https://www.ruanyifeng.com/blog/2019/10/tmux.html)抄袭的。
## 会话和进程
使用命令行的典型方法是打开一个终端窗口（以下简称“窗口”）并在其中输入命令。 **用户和计算机之间的这种临时交互称为“会话”**。
会话的一个重要特征是窗口和在其中启动的进程是[链接在一起](https://www.ruanyifeng.com/blog/2016/02/linux-daemon.html)的。当窗口打开时，会话开始；当窗口关闭时，会话结束，会话内的进程也将被终止，无论它们是否运行完毕。
一个典型的例子是[SSH登录](https://www.ruanyifeng.com/blog/2011/12/ssh_remote_login.html)到远程计算机并打开一个远程窗口来执行命令。这时，网络突然断开，再次登录时，找不到上次执行的命令。因为上一个SSH会话已经终止，其中的进程也随之消失了。
为了解决这个问题，会话和窗口可以“解绑”：当窗口关闭时，会话不会终止，而是继续运行。以后需要时，可以将该会话“绑定”到其他窗口。
## Tmux的作用
**Tmux是一个用于“解绑”会话和窗口的工具，将它们完全分离。**
> (1) 它允许在单个窗口中访问多个会话。这对于同时运行多个命令行程序很有用。
> (2) 它可以让一个新窗口“访问”一个现有的会话。
> (3) 它允许每个会话有多个连接窗口，因此多人可以实时共享会话。
> (4) 它还支持任意垂直和水平拆分窗口。

类似的终端复用器包括GNU Screen。Tmux在功能上与之相似，但更易于使用且功能更强大。
# 基本用法
### 安装
Tmux通常需要您自己安装。
```bash
# Ubuntu或Debian
$ sudo apt-get install tmux
# CentOS或Fedora
$ sudo yum install tmux
# Mac
$ brew install tmux
```
### 启动和退出
安装完成后，输入`tmux`命令进入Tmux窗口。
```bash
$ tmux
```
上面的命令将启动Tmux窗口，底部有一个状态栏。状态栏的左侧是窗口信息（编号和名称），右侧是系统信息。
![](https://cdn.beekka.com/blogimg/asset/201910/bg2019102006.png)
按`Ctrl+d`或显式输入`exit`命令退出Tmux窗口。
```bash
$ exit
```
### 前缀键
Tmux窗口有大量的快捷键。所有快捷键都必须由前缀键触发。默认的前缀键是`Ctrl+b`，即在按下`Ctrl+b`后，快捷键才会生效。例如，帮助命令的快捷键是`Ctrl+b ?`。其用法是在Tmux窗口中先按`Ctrl+b`，然后再按`?`来显示帮助信息。然后，按ESC键或`q`键退出帮助。
# 3. 会话管理

### 3.1 新建会话
启动的第一个Tmux窗口编号为`0`，第二个窗口编号为`1`，依此类推。与这些窗口对应的会话是会话0、会话1。
使用数字来区分会话不直观。更好的方法是为会话命名。
```bash
$ tmux new -s <session-name>
```
上面的命令创建一个具有指定名称的新会话。
### 3.2 分离会话
在Tmux窗口中，按`Ctrl+b d`或输入`tmux detach`命令将当前会话与窗口分离。
```bash
$ tmux detach
```
执行上述命令后，将退出当前的Tmux窗口，但会话及其内部的进程仍将在后台运行。`tmux ls`命令可以查看所有当前的Tmux会话。
```bash
$ tmux ls
# 或
$ tmux list-session
```
### 3.3 附加会话
`tmux attach`命令用于重新附加到现有会话。
```bash
# 使用会话编号
$ tmux attach -t 0
# 使用会话名称
$ tmux attach -t <session-name>
```
### 3.4 终止会话
`tmux kill-session`命令用于终止会话。
```bash
# 使用会话编号
$ tmux kill-session -t 0
# 使用会话名称
$ tmux kill-session -t <session-name>
```
### 3.5 切换会话
`tmux switch`命令用于切换会话。
```bash
# 使用会话编号
$ tmux switch -t 0
# 使用会话名称
$ tmux switch -t <session-name>
```
### 3.6 重命名会话
`tmux rename-session`命令用于重命名会话。
```bash
$ tmux rename-session -t 0 <new-name>
```
上面的命令重命名会话0。
### 3.7 会话快捷键
以下是一些与会话相关的快捷键。
- `Ctrl+b d`: 分离当前会话。
- `Ctrl+b s`: 列出所有会话。
- `Ctrl+b $`: 重命名当前会话。
# 4. 最小操作流程
总之，以下是Tmux的最小操作流程。
1.  创建一个新会话 `tmux new -s my_session`。
2.  在Tmux窗口中运行所需的程序。
3.  按快捷键`Ctrl+b d`分离会话。
4.  下次使用时，重新连接到会话 `tmux attach-session -t my_session`。
# 5. 窗格操作
Tmux可以将一个窗口分成多个窗格，每个窗格运行不同的命令。以下命令都在Tmux窗口中执行。
### 5.1 拆分窗格
`tmux split-window`命令用于拆分窗格。
```bash
# 上下拆分两个窗格
$ tmux split-window

# 左右拆分两个窗格
$ tmux split-window -h
```
![](https://cdn.beekka.com/blogimg/asset/201910/bg2019102007.jpg)
### 5.2 移动光标
`tmux select-pane`命令用于移动光标位置。
```bash
# 将光标切换到上方窗格
$ tmux select-pane -U
# 将光标切换到下方窗格
$ tmux select-pane -D
# 将光标切换到左侧窗格
$ tmux select-pane -L
# 将光标切换到右侧窗格
$ tmux select-pane -R
```
### 5.3 交换窗格位置
`tmux swap-pane`命令用于交换窗格位置。
```bash
# 将当前窗格向上移动
$ tmux swap-pane -U
# 将当前窗格向下移动
$ tmux swap-pane -D
```
### 5.4 窗格快捷键
以下是一些窗格操作快捷键。
- `Ctrl+b %`: 左右拆分两个窗格。
- `Ctrl+b "`: 上下拆分两个窗格。
- `Ctrl+b <arrow key>`: 将光标切换到另一个窗格。`<arrow key>`指的是要切换到的窗格的方向键，例如，要切换到下方窗格，请按向下箭头键`↓`。
- `Ctrl+b ;`: 将光标切换到上一个窗格。
- `Ctrl+b o`: 将光标切换到下一个窗格。
- `Ctrl+b {`: 将当前窗格与上一个窗格交换位置。
- `Ctrl+b }`: 将当前窗格与下一个窗格交换位置。
- `Ctrl+b Ctrl+o`: 所有窗格向前移动一个位置，第一个窗格变为最后一个窗格。
- `Ctrl+b Alt+o`: 所有窗格向后移动一个位置，最后一个窗格变为第一个窗格。
- `Ctrl+b x`: 关闭当前窗格。
- `Ctrl+b !`: 将当前窗格拆分为一个独立的窗口。
- `Ctrl+b z`: 当前窗格全屏显示，再次使用将恢复其原始大小。
- `Ctrl+b Ctrl+<arrow key>`: 按箭头键的方向调整窗格大小。
- `Ctrl+b q`: 显示窗格编号。
# 6. 窗口管理
除了将一个窗口划分为多个窗格之外，Tmux还允许创建多个窗口。
### 6.1 新建窗口
`tmux new-window`命令用于创建新窗口。
```bash
$ tmux new-window

# 创建一个具有指定名称的新窗口
$ tmux new-window -n <window-name>
```
### 6.2 切换窗口
`tmux select-window`命令用于切换窗口。
```bash
# 切换到具有指定编号的窗口
$ tmux select-window -t <window-number>
# 切换到具有指定名称的窗口
$ tmux select-window -t <window-name>
```
### 6.3 重命名窗口
`tmux rename-window`命令用于命名（或重命名）当前窗口。
```bash
$ tmux rename-window <new-name>
```
### 6.4 窗口快捷键
以下是一些窗口操作快捷键。
- `Ctrl+b c`: 创建一个新窗口，状态栏将显示多个窗口的信息。
- `Ctrl+b p`: 切换到上一个窗口（按状态栏上的顺序）。
- `Ctrl+b n`: 切换到下一个窗口。
- `Ctrl+b <number>`: 切换到具有指定编号的窗口，其中`<number>`是状态栏上的窗口编号。
- `Ctrl+b w`: 从列表中选择一个窗口。
- `Ctrl+b ,`: 重命名窗口。
# 7. 其他命令
以下是一些其他命令。
```bash
# 列出所有快捷键及其对应的Tmux命令
$ tmux list-keys
# 列出所有Tmux命令及其参数
$ tmux list-commands
# 列出有关所有当前Tmux会话的信息
$ tmux info
# 重新加载当前的Tmux配置
$ tmux source-file ~/.tmux.conf
```