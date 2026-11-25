# NERT-YOLO

面向森林火灾与烟雾检测的 PyTorch 实现，包含 DMO 重参数化瓶颈、C2LS 特征聚合模块和 NFEH 无 NMS 检测头。支持训练、验证、图片与视频推理、结构融合、模型导出和性能计时。

## 安装

使用 Python 3.10–3.12，在项目根目录执行：

```bash
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS 激活环境：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

GPU 环境应先安装与驱动匹配的 PyTorch 2.5.1 和 torchvision 0.20.1，再安装其余依赖。Ultralytics 固定为 8.3.0。

## 数据集

修改 `configs/data.example.yaml`，将 `path` 设置为数据集根目录的绝对路径。类别编号需与标注一致，示例为 `0: fire`、`1: smoke`。

```text
fire_smoke/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
```

每张图片对应一个同名 `.txt` 标注文件。每行格式为：

```text
class_id center_x center_y width height
```

坐标按图像宽高归一化至 0–1。负样本对应空标注文件。

## 训练

```bash
python run.py train --data configs/data.example.yaml --device 0
```

CPU 运行：

```bash
python run.py train --data configs/data.example.yaml --device cpu --batch 2
```

从检查点恢复训练：

```bash
python run.py train --model runs/train/nert/weights/last.pt --data configs/data.example.yaml --device 0 --resume
```

`configs/train.yaml` 定义优化器、学习率、输入尺寸、批量大小和增强设置。默认训练 200 轮，输入 640×640，使用 SGD。训练输出包含检查点、配置和指标记录。

## 验证与推理

```bash
python run.py val --weights runs/train/nert/weights/best.pt --data configs/data.example.yaml --split test --device 0
python run.py predict --weights runs/train/nert/weights/best.pt --source images/ --device 0 --conf 0.25 --max-det 300
python run.py predict --weights runs/train/nert/weights/best.pt --source video.mp4 --device 0
```

验证默认置信度阈值为 0.001，展示推理默认阈值为 0.25。推理 `max-det` 控制输出数量；训练标签分配的 Top-K 分别为 10 和 1。

## 结构融合与导出

```bash
python run.py deploy --weights runs/train/nert/weights/best.pt --output nert-deploy.pt
python run.py predict --weights nert-deploy.pt --source images/ --device 0
python run.py export --weights nert-deploy.pt --format torchscript --device cpu
python run.py export --weights nert-deploy.pt --format onnx --device cpu
```

DMO 融合使用 BatchNorm 的运行统计，并保留空间卷积与点卷积之间的 ReLU。融合后的检查点用于推理，恢复训练使用未融合的 `last.pt`。NFEH 推理仅计算 one-to-one 分支。

TorchScript 使用固定输入尺寸。ONNX 导出另需安装 `onnx`、`onnxslim` 和 `onnxruntime`，部署后端需支持动态局部聚合涉及的算子。

## 计时

```bash
python run.py benchmark --weights nert-deploy.pt --device cuda:0 --batch 1 --warmup 20 --iterations 100
```

工具先融合 DMO 和普通 Conv-BN，随后输出设备、精度、输入形状、预热次数、延迟与吞吐率。计时包含模型前向、解码和 Top-K，不包含读取图片、缩放、数据传输及最终阈值过滤。

报告中的操作数统计覆盖卷积和局部加权求和，乘加计为两次操作；BN、激活、softmax、解码及 Top-K 不计入该操作数。

## 模型配置

`configs/nert.yaml` 定义网络连接、通道和重复次数。默认 Backbone 与 Neck 膨胀率均为 1、1、2、3，DMO 分支数为 3。C2LS 使用 G=4、Ks=3、一个 LSBlock，FFN 扩展比为 2；局部聚合使用原生 PyTorch unfold 和 softmax 归一化权重。

本仓库通道缩放采用 YOLO11n 基准。以上配置定义此实现的具体设置；训练权重与数据集需自行准备，检测指标由实际训练与评估产生。

组件消融使用固定的通道宽度和重复次数：

```bash
python tools/make_variants.py
python run.py train --model configs/baseline.yaml --data configs/data.example.yaml --name baseline --device 0
python run.py train --model configs/dmo.yaml --data configs/data.example.yaml --name dmo --device 0
python run.py train --model configs/c2ls_only.yaml --data configs/data.example.yaml --name c2ls_only --device 0
python run.py train --model configs/dmo_c2ls.yaml --data configs/data.example.yaml --name dmo_c2ls --device 0
```

## 文件结构

```text
NERT_YOLO_Source/
  nert/
    __init__.py
    modules.py
    model.py
  configs/
    nert.yaml
    train.yaml
    data.example.yaml
    baseline.yaml
    dmo.yaml
    c2ls_only.yaml
    dmo_c2ls.yaml
  tools/
    make_variants.py
  run.py
  requirements.txt
  README.md
  LICENSE
```

## 许可与依赖

采用 AGPL-3.0-only 许可，详见 `LICENSE`。依赖 [Ultralytics v8.3.0](https://github.com/ultralytics/ultralytics/tree/v8.3.0)，使用其模型解析、检测训练、双标签分配损失和评估流程。模块注册在模型构建期间生效，不修改安装目录中的上游代码。
