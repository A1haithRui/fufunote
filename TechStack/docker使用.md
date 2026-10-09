## docker
**docker服务管理**

```bash
# 启动Docker服务（系统级）
sudo systemctl start docker

# 停止Docker服务（所有容器会停止）
sudo systemctl stop docker

# 重启Docker服务（配置变更后使用）
sudo systemctl restart docker

# 查看Docker服务状态
sudo systemctl status docker
```

**docker生命周期管理**

```bash
# 启动已存在的容器（如甲方提供的容器）
docker start [filename]
# 优雅停止容器（发送SIGTERM信号）
docker stop [filename]
# 强制立即停止容器（相当于断电）
docker kill [filename]
# 删除正在使用的容器
docker rm [filename]
# 重启容器（不改变配置）
docker restart [filename]
```

**对于容器**

```bash
# 显示所有容器
docker ps -a
# 删除旧容器
docker rm [container_name]
# 进入容器内部
...
```

一般来讲容器是非常轻量化的，根目录大概为：
```
/bin
/usr
/lib
/etc
/tmp
/home
```

这里想详细记录一下，当我有一个镜像的时候，我需要用这个镜像创建容器，创建通过`docker run ......`的指令。这个时候docker-desktop里会显示：

![[./assets/Pasted image 20251118172336.png]]

这个容器正在运行中，我在vscode里也许要安装docker和dev-containers:

![[./assets/Pasted image 20251118172431.png]]

运行之后就可以通过右键container的名字用shell或者新的vscode窗口进入docker里面进行调试了：

![[./assets/Pasted image 20251118172532.png]]


**对于镜像**

```bash
# 查看所有镜像（确认要删除的镜像名称和ID）
docker images
# 删除单个镜像（通过IMAGE ID或REPOSITORY:TAG）
docker rmi [filename]
# 或
docker rmi [is]  # 替换为实际IMAGE ID前几位即可
# 强制删除（当镜像被容器占用时）
docker rmi -f [filename]
```

**批量清理**

```bash
# 删除所有未被使用的镜像（安全）
docker image prune
# 强制删除所有未被容器使用的镜像
docker image prune -a
# 核弹级清理（删除所有镜像、容器、网络）
docker system prune -a
```

**打包docker**

```bash
# 创建docker
docker build -t [env_name]:latest .
# 打包本地文件方式
docker save -o [filename].tar [fileanme]:latest
# 对方导入
docker load -i [filename].tar
```

## docker-desktop

参考这篇博客：https://blog.csdn.net/smallfish1983/article/details/139861314

之前联网一直连不上，很烦，kimi给了解决方案，很有用。

---
代理配置问题：如果使用 VPN 或代理，Docker Desktop **不会自动继承系统代理**，需要手动配置。
**解决方案**：
```bash
# 在 Docker Desktop 中配置代理
1. 打开 Docker Desktop → Settings → Resources → Proxies
2. 勾选 "Manual proxy configuration"
3. 填写您的代理地址（注意：都是 HTTP）：

   Web Server (HTTP): http://127.0.0.1:7890
   Secure Web Server (HTTPS): http://127.0.0.1:7890  # 注意不是 https://
   
4. 点击 "Apply & Restart" 重启 Docker
```
---
本质上和proxy一样，就是添加一个流量转发。