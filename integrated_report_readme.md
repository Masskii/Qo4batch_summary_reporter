# Integrated Report Generator 使用指南

## 概述

`integrated_report.html` 是4-Batch CNN后处理工具生成的综合报告，包含所有模型的性能指标、输出张量信息和可视化结果。

## 快速开始

### Windows 用户

双击运行 `generate_integrated_report.bat` 文件：

```cmd
generate_integrated_report.bat
```

### Linux/Mac 用户

首先赋予执行权限，然后运行脚本：

```bash
chmod +x generate_integrated_report.sh
./generate_integrated_report.sh
```

## 完整工作流程

生成 `integrated_report.html` 需要以下步骤：

### 1. 准备输入数据

确保以下目录结构存在：

```
4batch-example/
├── [模型目录]/
│   ├── *.qo                    # 模型输出文件
│   ├── *.parameters.json       # 模型参数文件
│   ├── profile_core[0-3].json  # 性能数据文件（可选）
│   └── postprocess/            # 后处理输出目录
│       ├── output_[0-3]_p1/    # 输出张量可视化
│       ├── output_[0-3]_p2/
│       ├── ...
│       └── detection_boxes_*.jpg  # 检测结果图片
├── 4batch_postprocess_all.py   # 后处理脚本
├── generate_integrated_report.py  # 报告生成脚本
└── ...
```

### 2. 运行后处理（如果需要）

如果还没有后处理数据，运行：

```bash
# Linux/Mac
python 4batch_postprocess_all.py \
    --base-dir . \
    --image-dir ../4batch_input/image \
    --heads-dir ./heads \
    --batch-size 4

# Windows
python 4batch_postprocess_all.py --base-dir . --image-dir ..\..\4batch_input\image --heads-dir .\heads
```

### 3. 生成综合报告

```bash
# 使用默认设置
python generate_integrated_report.py

# 指定输出目录和文件名
python generate_integrated_report.py -d /path/to/results -o my_report.html

# 只生成HTML，不生成markdown文件
python generate_integrated_report.py --no-md
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-d, --directory` | 包含模型结果的目录 | 当前目录 `.` |
| `-o, --output` | 输出HTML文件名 | `integrated_report.html` |
| `--no-md` | 跳过生成markdown文件 | 否 |

## 环境变量

脚本支持以下环境变量配置：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BASE_DIR` | 基础目录 | `.` |
| `IMAGE_DIR` | 输入图像目录 | `../4batch_input/image` |
| `HEADS_DIR` | 检测头模型目录 | `./heads` |
| `OUTPUT_FILE` | 输出文件名 | `integrated_report.html` |
| `BATCH_SIZE` | 批处理大小 | `4` |

## 报告内容

生成的 `integrated_report.html` 包含以下内容：

### 1. 性能摘要（Performance Summary）

- **性能指标图表**：使用 ECharts 生成的柱状图，显示各模型的周期数
- **分类筛选**：可按模型类别（Detection、Segmentation等）筛选
- **数据表格**：详细的性能数据表
- **Excel导出**：可导出性能数据到Excel文件

### 2. 模型结果（Model Results）

- **模型卡片**：每个模型的详细信息
  - 模型名称和架构
  - 类别标签
  - 输出张量列表及形状
  - 检测/分割结果图片（如适用）

## 输入文件说明

### profile_core 文件

性能数据从 `profile_core[0-3].json` 文件中读取，包含以下指标：

- `TotalCycles`：总周期数
- `ExtBytes.LOAD`：外部加载字节数
- `ExtBytes.STORE`：外部存储字节数
- `ExecCycles.MAC`：MAC执行周期
- `ExecCycles.COMPUTE`：计算执行周期
- `StallCycles.MEU`：MEU停顿周期

### 后处理输出

后处理输出应包含：

1. **输出张量文件夹**：`output_[batch_index]_[output_name]/`
2. **可视化图像**：每个张量的可视化结果
3. **检测结果**：目标检测模型的边界框可视化图像

## 故障排查

### 问题：报告生成失败

**解决方案**：
1. 检查Python版本（需要3.7+）
2. 确认所有必需的目录存在
3. 验证模型目录中有有效的输出数据

### 问题：没有性能数据

**解决方案**：
- 确保 `profile_core[0-3].json` 文件存在于模型目录中
- 检查JSON文件格式是否正确

### 问题：图表不显示

**解决方案**：
- 检查网络连接（需要加载CDN上的ECharts库）
- 如果离线使用，需要下载ECharts到本地

## 示例输出

运行脚本后，你会看到类似以下的输出：

```
[STEP 1/6] Checking Python environment
─────────────────────────────────────────────────────────────────────
✓ Python found

[STEP 2/6] Checking required directories
─────────────────────────────────────────────────────────────────────
✓ Base directory: .
✓ Found 20 model directories

[STEP 3/6] Checking for existing postprocess data
─────────────────────────────────────────────────────────────────────
✓ Postprocess data found

[STEP 4/6] Checking for performance data
─────────────────────────────────────────────────────────────────────
✓ Performance data found

[STEP 5/6] Generating integrated report
─────────────────────────────────────────────────────────────────────
Scanning directory for model results...
Found 20 model directories
Generated performance_summary.md
Generated integrated_report.html
  - Total models: 20
  - Models with performance data: 18
  - Total output tensors: 76

[STEP 6/6] Report generated
─────────────────────────────────────────────────────────────────────

╔════════════════════════════════════════════════════════════════╗
║   Report Generation Complete!                                 ║
╚════════════════════════════════════════════════════════════════╝

Report location: /path/to/integrated_report.html
```

## 自动化脚本说明

### generate_integrated_report.sh（Linux/Mac）

- 自动检测Python环境
- 检查并提示运行后处理（如果需要）
- 生成综合报告
- 可选：自动在浏览器中打开报告

### generate_integrated_report.bat（Windows）

功能与Linux版本相同，适配Windows命令行环境。

## 高级用法

### 只生成特定模型的报告

```bash
# 只处理特定模型目录
python generate_integrated_report.py -d ./atss_r50
```

### 自定义输出位置

```bash
# 导出到指定位置
python generate_integrated_report.py -o /path/to/reports/my_report.html
```

### 批量生成报告

```bash
# 为多个子目录生成报告
for dir in */; do
    python generate_integrated_report.py -d "$dir" -o "${dir%/}_report.html"
done
```

## 联系支持

如有问题或建议，请联系4-Batch CNN工具团队。

---

**最后更新**：2026-03-06
