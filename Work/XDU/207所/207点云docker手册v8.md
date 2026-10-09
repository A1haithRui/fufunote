**版本 v8 更新说明**
*   离群点剔除接口 (`/api/outlier-removal`) 已更新，现在每次只处理单个文件。
*   离群点剔除接口的返回结果中新增了整个场景的边界框（Bbox）统计信息。
*   离群点剔除接口在服务端会打印详细的格式化报告，方便调试。

---

### 1. 加载 Docker 镜像

在tar包保存的位置打开一个终端，然后运行以下命令。请注意使用最新的 `v8` 版本tar包。

```bash
docker load -i pointcloud-app-0818v8.tar
```

### 2. 启动 Docker 容器

该命令会以后台模式启动服务。

```bash
docker run -d --rm -p 5000:5000 \
  -v /path/to/your/data:/app/data \
  --name point-cloud-service \
  point-cloud-app-0818v8
```
*   `-d`: 后台运行
*   `--rm`: 容器停止后自动删除
*   `-p 5000:5000`: 将主机的5000端口映射到容器的5000端口
*   `-v /path/to/your/data:/app/data`: 将您的本地数据目录挂载到容器的 `/app/data` 目录
*   `--name point-cloud-service`: 为容器命名，方便管理

### 3. 查看实时日志

如果需要查看应用程序的实时输出（包括详细的处理报告），可以新开一个终端来查看容器日志。

```bash
docker logs -f point-cloud-service
```

---

### 4. API 接口调用示例

#### 步骤 1: 离群点去除 (已更新)

调用此接口对单个 `.las` 文件进行离群点剔除。

```bash
curl -X POST -H "Content-Type: application/json" \
-d '{
  "input_path": "/app/data/raw/patch1/43.las",
  "output_path": "/app/data/outlier_removaled/patch1/43.las"
}' \
http://127.0.0.1:5000/api/outlier-removal
```

**成功返回结果示例:**
```json
{
    "output_path": "data/outlier_removaled/patch1/43.las",
    "stats": {
        "scene_bbox": {
            "area": 25710.1021,
            "center": [
                558244.87,
                4221064.42,
                59.54
            ],
            "height": 25.30,
            "length": 113.43,
            "volume": 206589.16,
            "width": 71.96,
            "xy_perimeter": 370.79
        },
        "剩余点数": 246636,
        "参数": {
            "标准差倍数": 3.0,
            "邻点数": 124,
            "采样数": 5000
        },
        "移除点数": 2915,
        "耗时": "0.56秒",
        "自动参数": true
    },
    "status": "success"
}
```

#### 步骤 2: 变化检测

```bash
curl -X POST -H "Content-Type: application/json" \
-d '{
  "source_file": "/app/data/outlier_removaled/patch1/43.las",
  "target_file": "/app/data/outlier_removaled/patch2/43.las",
  "bbox_json_path": "/app/data/result/43/bboxes.json"
}' \
http://127.0.0.1:5000/api/change-detection
```

#### 步骤 3: 评估所有bbox

```bash
curl -X POST -H "Content-Type: application/json" \
-d '{
  "source_file": "/app/data/outlier_removaled/patch1/43.las",
  "target_file": "/app/data/outlier_removaled/patch2/43.las",
  "bbox_json_path": "/app/data/result/43/bboxes.json",
  "evaluation_output_path": "/app/data/result/43/evaluation.json"
}' \
http://127.0.0.1:5000/api/evaluate
```

---

### 注意事项

1.  **路径说明**: 所有API调用中的路径都必须是 **容器内路径** (以 `/app/data` 开头)。
2.  **自动参数**: 系统会自动使用经过优化的参数，无需手动指定。
3.  **目录自动创建**: 系统会自动创建所有指定的输出目录。
