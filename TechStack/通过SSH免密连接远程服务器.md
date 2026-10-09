这次换主机，需要重新连接一下服务器，之前只记得大体流程，细节忘得差不多了，重新整理一下。
首先，生成公钥密钥，指定名称。在选择密码的时候直接`enter`跳过。
```
ssh-keygen -t rsa -b 2048 -f [自定义文件名]
```
其中：`-t rsa`:指定密钥类型为RSA。`-b 2048`：指定密钥长度为2048位（推荐的安全长度）。`-f [file]`：指定生成的密钥文件名称。
假如我的文件名为focalors，那么指令为：
```
ssh-keygen -t rsa -b 2048 -f id_rsa_focalors
```
这样`~/.ssh`目录下生成两个文件：
- `id_rsa_focalors`（私钥文件）
- `id_rsa_focalors.pub`（公钥文件）
接下来想办法先登录到服务器，然后把自己的目录下有一个`./ssh`文件夹，把这两个文件挪到该文件夹下。然后双击`authorized_keys`文件打开，双击`id_rsa_focalors.pub`文件打开，把`id_rsa_focalors.pub`里面的内容粘贴到`authorized_keys`里面。保存。
然后就可以通过ssh指令连接了。
```
ssh -i ~/.ssh/id_rsa_focalors xxx@xx.xx.xx.xx
```
如果是mobaxterm，点击新建sessions，正常输入端口和用户信息后，勾选`Use private key`，选中我们的`id_rsa_focalors`文件即可。
