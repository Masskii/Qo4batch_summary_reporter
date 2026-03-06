# 4-Batch CNN 后处理工具使用指南

## 概述

本工具用于批量处理和可视化 Quadric EPU 模型的推理输出结果，支持多种深度学习模型的后处理和可视化。

## 目录结构

```
4batch-example/
├── batch_postprocess_all.py      # 批量后处理主脚本
├── atss_postprocess_visualize.py # 单模型ATSS后处理脚本
├── postprocess_results/          # 输出结果目录
│   ├── summary_report.html       # 汇总报告（打开这个！）
│   ├── summary_report.md         # Markdown报告
│   ├── atss_r50/                 # ATSS R50模型结果
│   ├── ffnet40S/                 # FFNet 40S模型结果
│   └── ...                       # 其他模型结果
└── [各模型目录]/
    ├── *.parameters.json         # 模型参数配置
    ├── *.tensor*.bin             # 输出张量二进制文件
    └── ...
```

## 快速开始

### 1. 批量处理所有模型

```bash
# 进入4batch-example目录
cd E:\test-Models\4_batch_cnn_tool\4batch-example

# 运行批量后处理脚本
python batch_postprocess_all.py --base-dir . --output ./postprocess_results
```

### 2. 处理指定模型

```bash
# 只处理ATSS R50和FFNet 40S
python batch_postprocess_all.py --base-dir . --output ./postprocess_results --models atss_r50 ffnet40S
```

### 3. 单模型详细处理

```bash
# 进入特定模型目录
cd atss_r50

# 运行ATSS专用后处理脚本（特征图可视化）
python atss_postprocess_visualize.py --mode feature --output ./output_vis

# 当有检测结果时（检测框可视化）
python atss_postprocess_visualize.py --mode detection \
    --detections detections.json \
    --output ./output_vis
```

## 命令行参数说明

### batch_postprocess_all.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--base-dir` | 模型输出根目录 | `../4batch-example` |
| `--output` | 结果输出目录 | `./postprocess_results` |
| `--batch-size` | 批处理大小 | `4` |
| `--models` | 指定要处理的模型名称（空格分隔） | 全部模型 |

### atss_postprocess_visualize.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 可视化模式: `feature` 或 `detection` | `feature` |
| `--input-dir` | 输入目录（tensor文件位置） | `.` |
| `--image-dir` | 原始图片目录 | `../../4batch_input/image` |
| `--detections` | 检测结果JSON文件（detection模式） | `None` |
| `--output` | 输出目录 | `./output_vis` |
| `--conf-threshold` | 检测置信度阈值 | `0.5` |

## 输出文件说明

### 每个模型的输出结构

```
postprocess_results/
└── [模型名称]/
    └── output_[序号]_[张量名称]/
        ├── mean.png           # 所有通道的均值可视化
        ├── grid.png           # 多通道网格视图（前16个通道）
        ├── batch_0_mean.png   # 第1张图片的特征图
        ├── batch_1_mean.png   # 第2张图片的特征图
        ├── batch_2_mean.png   # 第3张图片的特征图
        └── batch_3_mean.png   # 第4张图片的特征图
```

### 可视化文件说明

| 文件 | 说明 |
|------|------|
| `mean.png` | 所有特征通道的平均值，显示整体激活情况 |
| `grid.png` | 4x4网格显示前16个通道，便于对比不同通道的特征 |
| `batch_X_mean.png` | 单张图片的特征图，对应batch中的每一张输入图 |

## 支持的模型类型

| 模型类型 | 模型名称 | 输出说明 |
|----------|----------|----------|
| **目标检测** | atss_r50, atss_r101 | FPN多尺度特征图 (P2/P4/P6/P8/P10) |
| **目标检测** | paa_r50, paa_r101 | FPN多尺度特征图 |
| **目标检测** | autoassign_r50 | FPN多尺度特征图 |
| **目标检测** | lad_r50, lad_r101 | FPN多尺度特征图 |
| **语义分割** | ffnet40S/54S/78S/86S/150S | 分割特征图 |
| **姿态估计** | pose_resnet_50 | 姿态特征图 |
| **目标检测** | yolov8n, yolov5n_seg | YOLO特征图 |
| **人脸检测** | Mediapipe_face_* | 人脸检测特征 |

## 查看结果

### 方法1: 打开HTML汇总报告

```bash
# Windows
start postprocess_results\summary_report.html

# Mac/Linux
open postprocess_results/summary_report.html
```

HTML报告包含：
- 所有模型的处理状态
- 输出张量的形状和数据范围
- 每个模型结果的快速链接

### 方法2: 直接查看图片

进入特定模型的输出目录查看PNG图片：

```
postprocess_results/
└── atss_r50/
    └── output_0_p2/
        ├── mean.png     # 用图片查看器打开
        ├── grid.png
        └── batch_0_mean.png
```

## 检测框可视化（未来支持）

当您有检测结果时，可以使用以下方式可视化：

### 1. 准备检测结果JSON

```json
{
  "images": [
    {"image_id": 0, "file_name": "000000000057.jpg"},
    {"image_id": 1, "file_name": "000000000552.jpg"}
  ],
  "detections": [
    {
      "image_id": 0,
      "bbox": [x1, y1, x2, y2],
      "score": 0.95,
      "category_id": 0
    }
  ]
}
```

### 2. 运行可视化

```bash
python atss_postprocess_visualize.py \
    --mode detection \
    --detections your_detections.json \
    --output ./output_vis
```

## 常见问题

### Q1: 脚本报错 "No module named 'PIL'"
**A:** 需要安装依赖：
```bash
pip install Pillow numpy opencv-python matplotlib
```

### Q2: 某些模型没有生成输出
**A:** 检查该模型目录是否存在 `*parameters.json` 文件，脚本依赖此文件来了解输出结构。

### Q3: 生成的图片是全黑或全白
**A:** 这可能是因为该特征图的所有值都相同（常数），这是正常的某些层输出。

### Q4: 如何理解特征图可视化？
**A:**
- **亮色区域** = 高激活值，模型在这些区域"注意"到特征
- **暗色区域** = 低激活值，背景或不重要区域
- **不同通道** = 不同类型的特征（边缘、纹理、形状等）

### Q5: 能否可视化更多通道？
**A:** 可以修改脚本中的 `grid_size` 参数，默认显示16个通道（4x4网格）。

## 输出数据示例

### ATSS R50 FPN输出

| 层级 | 形状 | 分辨率 | 说明 |
|------|------|--------|------|
| P2 | [4, 256, 100, 100] | 100x100 | 高分辨率特征 |
| P4 | [4, 256, 13, 13] | 13x13 | 低分辨率特征 |
| P6 | [4, 256, 25, 25] | 25x25 | 中等分辨率特征 |
| P8 | [4, 256, 50, 50] | 50x50 | 中等分辨率特征 |
| P10 | [4, 256, 7, 7] | 7x7 | 全局特征 |

### 特征图数据范围

每个模型的输出数据范围不同，例如：
- **ATSS R50**: range=[-2.30, 2.15]
- **FFNet 40S**: range=[-5.20, 8.30]
- **YOLOv8n**: range=[-3.50, 4.80]

这些值在可视化时会自动归一化到0-255范围。

## 技术细节

### 二进制文件读取

脚本读取EPU输出的int32二进制文件，并根据fracbits参数进行去量化：

```python
float_value = int_value / (2 ** fracbits)
```

### 特征图归一化

可视化时自动归一化：
```python
normalized = (value - min) / (max - min) * 255
```

## 更新日志

- **v1.0** (2025-03-04)
  - 支持批量处理17种模型
  - 自动识别模型类型
  - 生成HTML和Markdown汇总报告
  - 特征图可视化（均值、网格、batch分离）

## 联系方式

如有问题，请查看：
- 原始预处理脚本: `../4batch_input/preprocess_to_bin_batch.py`
- 模型配置文件: `../4batch_input/json/`
