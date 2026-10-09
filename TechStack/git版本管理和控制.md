# Windows上git的http速度很慢的解决方法
在 windows上用clash代理，但是在终端使用git clone时依然很慢甚至报错，这里貌似不能配置proxy，而是需要禁用proxy，通过以下指令：
```powershell
git config --global --unset http.proxy
git config --global --unset https.proxy
```
# github的ssh配置
常用的几台设备最好用ssh配好，这样就不会遇到http被卡连接的情况了。

第一次配置实际上记住下面几条命令就够了：

```bash
# 1. 生成 SSH Key
ssh-keygen -t ed25519 -C "你的GitHub邮箱"

# 2. 查看公钥
cat ~/.ssh/id_ed25519.pub

# 3. 把上面的公钥添加到 GitHub
# GitHub → Settings → SSH and GPG keys → New SSH key

# 4. 测试
ssh -T git@github.com

# 5. SSH clone
git clone git@github.com:user/repo.git
```

整个认证关系可以简单理解成：

```text
你的电脑
│
├── 私钥
│   ~/.ssh/id_ed25519
│   （只保存在本机）
│
└──────── SSH 身份验证 ────────→ GitHub
                                  │
                                  └── 保存你的公钥
                                      id_ed25519.pub
```

**核心原则：GitHub 保存公钥，本机保存私钥。私钥永远不要上传 GitHub、网盘或发给别人。**


# github

clone别人代码：
```bash
git clone URL(网址)

# 例如
git clone https://github.com/cvlab-epfl/wide-depth-range-pose.git
```
如果通过 `git clone <远程仓库URL>` 下载一个仓库，Git 会**自动帮你完成关联**。
- 克隆后，本地仓库会默认关联到克隆的远程仓库，且远程仓库的别名是 `origin`。
- 可以直接使用 `git push`/`git pull`，无需手动关联。
# git
## git常规使用

如果项目有`.git`文件，先清掉：
```bash
sudo rm -rf .git
```
然后手动在github自己的网页创建一个repo，复制url。
在项目中初始化自己的仓库：
```bash
git init
```
然后添加远程仓库:
```bash
git remote add origin "自己仓库的url"

# 如果文件夹不安全
git config --global --add  “文件路径”
```
然后添加文件到暂存区：
```bash
git add .
```
添加文件版本控制：
```bash
git commit -m "Initial commit"
```
推到远程仓库：
用户名就是自己的git的username，密码只能用git的token。先点击头像，然后进入setting，最下面有个developer setting。这里找到personal access tokens，复制即可。
然后把代码推送到远程仓库：
```
git branch -M main
git push -u origin main
```
## git常用指令


## 关于.gitignore

创建好仓库之后，第一件事直接创建.gitignore，然后写上要忽略的文件，添加.gitignore到缓存区。

```Plain
git add .ignore
```

此时应该已经满足忽略条件了，实在不行就提交一次，看看有没有效果。如果有效果，那么后续就不需要再添加和提交.gitignore，只需要每次修改这个文件并保存就行。

# 多人协作管理

## 基本原则

- `main`：稳定分支，必须是可以运行且复现的
- `branch_a`, `branch_b`：假设存在的两个个人分支 
- 不在`main`上直接开发。
- 每次开工前都pull一下`main`分支
## 个人分支使用流程

1. 从`main`创建个人分支
```bash
git checkout main  # 先进入到main分支
git pull origin main  # 拉取main分支
git checkout -b branch_a  # 进入到branch_a分支
```

2. 在个人分支上开发和提交
```bash
git add .  # 添加缓冲区
git commit -m "message"
git push origin branch_a  # 相当于是把本次修改推送到自己的分支上
```

3. 保持和`main`分支同步
```bash
git pull origin main  # 拉去main的时候会检查，无冲突会自动合并；有冲突需要手动解决冲突
git add .
git commit -m "resolve merge conflict"
```

## 合并`main`分支

有一个人专门负责维护`main`，在本地做合并。
```bash
git checkout main
git pull origin main
git merge branch_a
git push origin main
```

如果遇到冲突，遵循以下原则：
- 不同文件 -> 不会冲突
- 同一文件不同位置 -> 自动合并
- 同一文件同一位置 -> 手动编辑解决，vscode会提示：
```
- Accept Current
- Accept Incoming
- Accept Both
- 或手动编辑
```