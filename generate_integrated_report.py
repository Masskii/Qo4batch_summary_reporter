#!/usr/bin/env python3
"""
Integrated Report Generator for 4-Batch CNN Post-processing Results
Scans local directories and generates integrated_report.html
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, NamedTuple
import sys
import io


class PerformanceData(NamedTuple):
    """Performance data for a model"""
    cycles: int
    ext_load: int
    ext_store: int
    mac: int
    meu_stall: int
    compute: int


class ModelInfo:
    """Store information about a model"""
    def __init__(self, name: str):
        self.name = name
        self.category = "other"
        self.architecture = "Unknown"
        self.outputs = []  # List of (name, shape, status)
        self.detection_images = []  # List of (filename, object_count)
        self.performance: Optional[PerformanceData] = None


def get_model_category(name: str) -> str:
    """Determine model category from name"""
    name_lower = name.lower()
    if name_lower.startswith("mediapipe"):
        return "mediapipe"
    elif any(x in name_lower for x in ["atss", "lad", "paa", "autoassign", "centernet"]):
        return "detection"
    elif "ffnet" in name_lower:
        return "segmentation"
    elif "solo" in name_lower:
        return "instance_segmentation"
    elif "yolo" in name_lower:
        return "yolo"
    elif "pose" in name_lower:
        return "pose"
    return "other"


def get_model_architecture(name: str) -> str:
    """Get model architecture name"""
    name_upper = name.upper()
    if "ATSS" in name_upper:
        return "ATSS"
    elif "LAD" in name_upper:
        return "LAD"
    elif "PAA" in name_upper:
        return "PAA"
    elif "AUTOASSIGN" in name_upper:
        return "AutoAssign"
    elif "CENTERNET" in name_upper:
        return "CenterNet"
    elif "FFNET" in name_upper:
        return "FFNet"
    elif "SOLO" in name_upper:
        return "SOLO"
    elif "YOLO" in name_upper:
        return "YOLO"
    elif "POSE" in name_upper:
        return "PoseResNet"
    elif name_upper.startswith("MEDIAPIPE"):
        return "MediaPipe"
    return "Unknown"


def load_performance_from_profile_core(model_dir: str, model_name: str) -> Optional[PerformanceData]:
    """Load performance data from profile_core*.json files in a model directory"""
    cycles = 0
    ext_load = 0
    ext_store = 0
    mac = 0
    meu_stall = 0
    compute = 0

    # Find all profile_core files
    profile_files = []
    for i in range(4):
        file_path = os.path.join(model_dir, f"profile_core{i}.json")
        if os.path.exists(file_path):
            profile_files.append(file_path)

    if not profile_files:
        return None

    # Aggregate data from all profile files
    for profile_file in profile_files:
        try:
            with open(profile_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        # Find the "default" entry which contains totals
        for entry in data:
            if entry.get("name") == "default":
                entry_data = entry.get("data", {})

                # Extract metrics
                cycles += entry_data.get("TotalCycles", 0)

                ext_bytes = entry_data.get("ExtBytes", {})
                ext_load += ext_bytes.get("LOAD", 0)
                ext_store += ext_bytes.get("STORE", 0)

                exec_cycles = entry_data.get("ExecCycles", {})
                mac += exec_cycles.get("MAC", 0)
                compute += exec_cycles.get("COMPUTE", 0)

                stall_cycles = entry_data.get("StallCycles", {})
                meu_stall += stall_cycles.get("MEU", 0)

                break

    # Average across all cores
    count = len(profile_files)
    if count > 1:
        cycles //= count
        ext_load //= count
        ext_store //= count
        mac //= count
        meu_stall //= count
        compute //= count

    return PerformanceData(cycles, ext_load, ext_store, mac, meu_stall, compute)


def load_performance_data(base_path: str) -> Dict[str, PerformanceData]:
    """Load performance data from performance_summary.md or generate from profile_core files"""
    perf_file = os.path.join(base_path, "performance_summary.md")
    data = {}

    # First, try to load from existing performance_summary.md
    if os.path.exists(perf_file):
        with open(perf_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        in_table = False
        for line in lines:
            if '| Model |' in line or '| Model Name' in line:
                in_table = True
                continue
            if in_table:
                if line.startswith('|---') or line.startswith('| ='):
                    continue
                if not line.startswith('|'):
                    break
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 7:
                    try:
                        model_name = parts[0]
                        cycles = int(parts[1].replace(',', '').replace('`', ''))
                        ext_load = int(parts[2].replace(',', '').replace('`', ''))
                        ext_store = int(parts[3].replace(',', '').replace('`', ''))
                        mac = int(parts[4].replace(',', '').replace('`', ''))
                        meu_stall = int(parts[5].replace(',', '').replace('`', ''))
                        compute = int(parts[6].replace(',', '').replace('`', ''))
                        data[model_name] = PerformanceData(cycles, ext_load, ext_store, mac, meu_stall, compute)
                    except (ValueError, IndexError):
                        pass

        if data:
            return data

    # If no performance_summary.md or it's empty, scan profile_core files
    print("Scanning profile_core files for performance data...")
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if not os.path.isdir(item_path):
            continue

        if item.startswith('.') or item in ['__pycache__', '.claude']:
            continue

        perf = load_performance_from_profile_core(item_path, item)
        if perf and perf.cycles > 0:
            data[item] = perf

    return data


def load_summary_report_data(base_path: str) -> Dict[str, ModelInfo]:
    """Load model data from summary_report.md"""
    summary_file = os.path.join(base_path, "summary_report.md")
    models = {}

    if not os.path.exists(summary_file):
        print(f"Warning: {summary_file} not found")
        return models

    with open(summary_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse model sections
    model_pattern = r'###\s+\*\*(.+?)\*\*\s*\n\*\*(.+?)\*\*'
    for match in re.finditer(model_pattern, content):
        model_name = match.group(1).strip()
        model_info = ModelInfo(model_name)
        model_info.category = get_model_category(model_name)
        model_info.architecture = get_model_architecture(model_name)
        models[model_name] = model_info

    return models


def load_shape_from_parameters(model_name: str, base_path: str) -> Dict[str, List[int]]:
    """Load output shapes from model's parameters.json in parent directory"""
    parent_dir = os.path.dirname(base_path)
    shapes = {}

    # Find parameters.json file in the model directory (e.g., ../atss_r101/*.parameters.json)
    model_dir = os.path.join(parent_dir, model_name)

    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            if file.endswith(".parameters.json"):
                param_file = os.path.join(model_dir, file)
                try:
                    with open(param_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    for param in data.get("parameters", []):
                        if param.get("io_type") == "output":
                            param_list = param.get("parameter", [])
                            if param_list:
                                output_name = param_list[0]  # e.g., "p2"
                                shape = param.get("shape", [])
                                shapes[output_name] = shape
                except (json.JSONDecodeError, IOError):
                    pass
                break  # Found the parameters file, stop searching

    return shapes


def scan_model_directory(base_path: str) -> List[ModelInfo]:
    """Scan directory for model results"""
    perf_data = load_performance_data(base_path)
    summary_data = load_summary_report_data(base_path)

    models = {}

    # First, load from summary_report data
    for name, info in summary_data.items():
        models[name] = info

    # Then scan directories
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if not os.path.isdir(item_path):
            continue

        if item.startswith('.') or item in ['__pycache__', '.claude']:
            continue

        if item not in models:
            models[item] = ModelInfo(item)

        model = models[item]
        model.category = get_model_category(item)
        model.architecture = get_model_architecture(item)

        # Load shapes from parent directory
        shape_map = load_shape_from_parameters(item, base_path)

        # Check if postprocess directory exists
        postprocess_path = os.path.join(item_path, "postprocess")
        scan_path = postprocess_path if os.path.exists(postprocess_path) else item_path

        # Get output folders
        output_dirs = sorted([d for d in os.listdir(scan_path) if d.startswith("output_")])
        for output_dir in output_dirs:
            match = re.search(r'output_\d+_(p\d+)', output_dir)
            if match:
                level = match.group(1)
                # Get shape from parameters.json if available
                shape = shape_map.get(level, [])
                shape_str = str(shape) if shape else "unknown"
                model.outputs.append((level, shape_str, "success"))

        # Get detection images
        for file in sorted(os.listdir(scan_path)):
            if file.startswith("detection_boxes_") and file.endswith((".jpg", ".png")):
                # Extract object count from filename or folder
                model.detection_images.append((file, 0))  # Default to 0

        # Add performance data
        if item in perf_data:
            model.performance = perf_data[item]

    return sorted(models.values(), key=lambda m: m.name.lower())


def generate_html(models: List[ModelInfo], base_path: str, output_file: str = "integrated_report.html"):
    """Generate the integrated HTML report"""

    total_models = len([m for m in models if m.outputs or m.detection_images])
    models_with_perf = len([m for m in models if m.performance])
    total_outputs = sum(len(m.outputs) for m in models)

    # Build chart data grouped by category
    category_colors = {
        'detection': '#FF5722',
        'segmentation': '#2196F3',
        'instance_segmentation': '#00BCD4',
        'pose': '#9C27B0',
        'mediapipe': '#E91E63',
        'yolo': '#FFC107',
        'other': '#757575'
    }

    # Group models by category
    category_data = {}
    for model in models:
        if model.performance:
            cat = model.category
            if cat not in category_data:
                category_data[cat] = {'models': [], 'cycles': [], 'color': category_colors.get(cat, '#757575')}
            category_data[cat]['models'].append(model.name)
            category_data[cat]['cycles'].append(model.performance.cycles)

    # Sort categories and models within each category by cycles
    for cat in category_data:
        sorted_pairs = sorted(zip(category_data[cat]['models'], category_data[cat]['cycles']), key=lambda x: x[1])
        category_data[cat]['models'] = [p[0] for p in sorted_pairs]
        category_data[cat]['cycles'] = [p[1] for p in sorted_pairs]

    # Build chart data for "All" view (sorted by cycles)
    all_models = []
    all_cycles = []
    all_ext_load = []
    all_ext_store = []
    all_mac = []
    all_meu_stall = []
    all_compute = []
    all_colors = []
    color_palette = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F',
        '#BB8FCE', '#85C1E2', '#F8B88B', '#52B788', '#FF6F61', '#6B5B95',
        '#88B04B', '#F7CAC9', '#92A8D1', '#955251', '#B565A7', '#009B77',
        '#DD4124', '#D65076', '#45B8AC', '#EFC050', '#5B5EA6', '#9B2335'
    ]

    for model in models:
        if model.performance:
            all_models.append(model.name)
            all_cycles.append(model.performance.cycles)
            all_ext_load.append(model.performance.ext_load)
            all_ext_store.append(model.performance.ext_store)
            all_mac.append(model.performance.mac)
            all_meu_stall.append(model.performance.meu_stall)
            all_compute.append(model.performance.compute)
            all_colors.append(category_colors.get(model.category, '#757575'))

    # Generate model cards HTML
    model_cards_html = ""
    for model in models:
        if not model.outputs and not model.detection_images:
            continue

        category_class = model.category
        tag_class = f"tag-{model.category}"

        # Build outputs HTML
        outputs_html = ""
        for output_name, shape, status in model.outputs:
            outputs_html += f'''
                    <div class="output-item">
                        <strong>{output_name}</strong> - {shape}
                        <span class="success">{status}</span>
                    </div>'''

        # Build detection gallery HTML
        gallery_html = ""
        if model.detection_images:
            gallery_html = '<div class="detection-gallery"><h4>Detection Results:</h4>'
            for img_file, obj_count in model.detection_images[:4]:  # Max 4 images
                img_path = f"{model.name}/postprocess/{img_file}"
                gallery_html += f'<div class="img-container"><img src="{img_path}" alt="{img_file}" loading="lazy"><span>{img_file}</span></div>'
            gallery_html += '</div>'

        model_cards_html += f'''
            <div class="model-card {category_class}">
                <h3>{model.name} <span class="tag {tag_class}">{model.category.capitalize()}</span></h3>
                <p><em>{model.architecture}</em></p>
                <div class="output-list">{outputs_html}
                </div>{gallery_html}
            </div>'''

    # Generate category tabs HTML
    category_tabs_html = '<button class="cat-tab active" data-category="all">All Models</button>'
    chart_containers_html = '<div class="chart-container" id="chart-all"></div>'

    for cat in sorted(category_data.keys()):
        cat_name = cat.replace('_', ' ').capitalize()
        category_tabs_html += f'<button class="cat-tab" data-category="{cat}">{cat_name}</button>'
        chart_containers_html += f'<div class="chart-container" id="chart-{cat}" style="display:none;"></div>'

    # Generate table rows
    table_rows = ""
    for model in models:
        if model.performance:
            cycles_str = f"{model.performance.cycles:,}"
            table_rows += f'                    <tr><td>{model.name}</td><td>{cycles_str}</td></tr>\n'

    # Generate HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Integrated Report - 4-Batch CNN Results</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.4.3/echarts.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f5f5;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            padding: 30px;
            background: white;
            border-radius: 16px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        .header h1 {{
            color: #333;
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}
        .header p {{
            color: #666;
            font-size: 1.1rem;
        }}
        .nav-tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .nav-tab {{
            padding: 15px 30px;
            background: white;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .nav-tab:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }}
        .nav-tab.active {{
            background: #4CAF50;
            color: white;
        }}
        .section {{
            display: none;
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        .section.active {{
            display: block;
        }}
        .section-title {{
            font-size: 1.8rem;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #4CAF50;
        }}
        .performance-overview {{
            background: #f5f7fa;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
        }}
        .performance-overview p {{
            font-size: 1.1rem;
            color: #333;
        }}
        .category-tabs {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .category-label {{
            font-weight: 600;
            color: #333;
            font-size: 1rem;
        }}
        .cat-tab {{
            padding: 10px 20px;
            background: #f0f0f0;
            border: 2px solid transparent;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.95rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }}
        .cat-tab:hover {{
            background: #e0e0e0;
        }}
        .cat-tab.active {{
            background: #4CAF50;
            color: white;
            border-color: #4CAF50;
        }}
        .chart-container {{
            width: 100%;
            height: 500px;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }}
        .export-btn {{
            padding: 12px 24px;
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }}
        .export-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(17, 153, 142, 0.4);
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .data-table th {{
            background: #4CAF50;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        .data-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        .data-table tr:hover {{
            background: #f9f9f9;
        }}
        .summary-box {{
            background: #4CAF50;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
        }}
        .summary-box h2 {{
            color: white;
            margin-bottom: 15px;
        }}
        .summary-box p {{
            color: white;
            font-size: 1.1rem;
            margin: 5px 0;
        }}
        .model-card {{
            background: #f8f9fa;
            padding: 20px;
            margin: 15px 0;
            border-radius: 12px;
            border-left: 5px solid #4CAF50;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .model-card.detection {{ border-left-color: #FF5722; }}
        .model-card.segmentation {{ border-left-color: #2196F3; }}
        .model-card.pose {{ border-left-color: #9C27B0; }}
        .model-card.instance_segmentation {{ border-left-color: #00BCD4; }}
        .model-card.mediapipe {{ border-left-color: #E91E63; }}
        .model-card.yolo {{ border-left-color: #FFC107; }}
        .model-card h3 {{
            margin-top: 0;
            color: #333;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .tag {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            color: white;
        }}
        .tag-detection {{ background: #FF5722; }}
        .tag-segmentation {{ background: #2196F3; }}
        .tag-pose {{ background: #9C27B0; }}
        .tag-instance_segmentation {{ background: #00BCD4; }}
        .tag-mediapipe {{ background: #E91E63; }}
        .tag-yolo {{ background: #FFC107; }}
        .output-list {{
            margin: 15px 0;
        }}
        .output-item {{
            padding: 10px;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .output-item:last-child {{
            border-bottom: none;
        }}
        .success {{ color: #4CAF50; font-weight: 600; }}
        .error {{ color: #f44336; font-weight: 600; }}
        .detection-gallery {{
            margin: 15px 0;
        }}
        .detection-gallery h4 {{
            color: #555;
            margin-bottom: 10px;
        }}
        .img-container {{
            display: inline-block;
            margin: 5px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            transition: all 0.3s ease;
        }}
        .img-container:hover {{
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        }}
        .img-container img {{
            max-width: 200px;
            display: block;
        }}
        .img-container span {{
            font-size: 12px;
            text-align: center;
            display: block;
            padding: 8px;
            background: #f5f5f5;
        }}
        footer {{
            text-align: center;
            padding: 30px;
            color: #666;
            margin-top: 30px;
        }}
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8rem;
            }}
            .section {{
                padding: 20px;
            }}
            .chart-container {{
                height: 350px;
            }}
            .nav-tabs {{
                flex-direction: column;
            }}
            .nav-tab {{
                width: 100%;
            }}
            .data-table {{
                font-size: 0.9rem;
            }}
            .data-table th, .data-table td {{
                padding: 10px 8px;
            }}
            .img-container img {{
                max-width: 150px;
            }}
        }}
        @media (max-width: 480px) {{
            body {{
                padding: 10px;
            }}
            .header {{
                padding: 20px;
            }}
            .header h1 {{
                font-size: 1.5rem;
            }}
            .chart-container {{
                height: 280px;
            }}
            .model-card {{
                padding: 15px;
            }}
            .img-container img {{
                max-width: 120px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Integrated Report</h1>
            <p>4-Batch CNN Post-processing Results & Performance Analysis</p>
        </div>

        <div class="nav-tabs">
            <button class="nav-tab active" onclick="showSection('performance')">Performance Summary</button>
            <button class="nav-tab" onclick="showSection('summary')">Model Results</button>
        </div>

        <!-- Performance Section -->
        <div id="performance" class="section active">
            <h2 class="section-title">Performance Metrics - Cycles Analysis</h2>

            <div class="performance-overview">
                <p><strong>Total Models Analyzed:</strong> {models_with_perf}</p>
                <button class="export-btn" onclick="exportToExcel()">
                    Export to Excel
                </button>
            </div>

            <div class="category-tabs">
                <span class="category-label">Filter by category:</span>
                {category_tabs_html}
            </div>

            {chart_containers_html}

            <table class="data-table" id="cyclesTable">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Cycles</th>
                    </tr>
                </thead>
                <tbody>
{table_rows}                </tbody>
            </table>
        </div>

        <!-- Summary Section -->
        <div id="summary" class="section">
            <h2 class="section-title">4-Batch CNN Post-processing Results</h2>

            <div class="summary-box">
                <h2>Summary</h2>
                <p><strong>Total Models Processed:</strong> {total_models}</p>
                <p><strong>Successful:</strong> {total_models} <span class="success">OK</span></p>
                <p><strong>Failed:</strong> 0 <span class="error">FAIL</span></p>
                <p><strong>Total Output Tensors:</strong> {total_outputs}</p>
            </div>

            <h3 style="color: #555; margin: 30px 0 20px; font-size: 1.5rem;">Model Results</h3>
{model_cards_html}        </div>

        <footer>
            <p>Generated by 4-batch CNN Post-processing Tool</p>
            <p style="font-size: 0.9rem; margin-top: 10px;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>

    <script>
        // Tab switching
        function showSection(sectionId) {{
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            document.getElementById(sectionId).classList.add('active');
            event.target.classList.add('active');

            if (sectionId === 'performance' && window.myChart) {{
                setTimeout(() => window.myChart.resize(), 100);
            }}
        }}

        // Chart data grouped by category
        const categoryData = {json.dumps(category_data)};
        const allChartData = {{
            models: {json.dumps(all_models)},
            cycles: {json.dumps(all_cycles)},
            extLoad: {json.dumps(all_ext_load)},
            extStore: {json.dumps(all_ext_store)},
            mac: {json.dumps(all_mac)},
            meuStall: {json.dumps(all_meu_stall)},
            compute: {json.dumps(all_compute)},
            colors: {json.dumps(all_colors)}
        }};

        const charts = {{}};

        // Initialize all charts
        function initCharts() {{
            if (typeof echarts === 'undefined') {{
                console.error('ECharts library not loaded');
                return;
            }}

            // Initialize "All" chart
            createChart('all', allChartData.models, allChartData.cycles, allChartData.colors);

            // Initialize category charts
            for (const [cat, data] of Object.entries(categoryData)) {{
                createChart(cat, data.models, data.cycles, [data.color]);
            }}
        }}

        function createChart(chartId, models, cycles, colors) {{
            const chartDom = document.getElementById('chart-' + chartId);
            if (!chartDom) {{
                console.error('Chart container not found:', 'chart-' + chartId);
                return;
            }}

            const chart = echarts.init(chartDom);

            const option = {{
                grid: {{
                    left: '5%',
                    right: '5%',
                    bottom: '15%',
                    top: '5%',
                    containLabel: true
                }},
                xAxis: {{
                    type: 'category',
                    data: models,
                    axisLabel: {{
                        color: '#2E7D32',
                        interval: 0,
                        rotate: 45,
                        fontSize: 10
                    }},
                    axisLine: {{
                        lineStyle: {{
                            color: '#2E7D32'
                        }}
                    }},
                    axisTick: {{
                        lineStyle: {{
                            color: '#2E7D32'
                        }}
                    }},
                    splitLine: {{
                        show: false
                    }}
                }},
                yAxis: {{
                    type: 'value',
                    name: 'cycles',
                    nameTextStyle: {{
                        color: '#2E7D32',
                        fontSize: 14,
                        padding: [0, 0, 0, -20]
                    }},
                    axisLabel: {{
                        color: '#2E7D32',
                        fontSize: 11
                    }},
                    axisLine: {{
                        lineStyle: {{
                            color: '#2E7D32'
                        }}
                    }},
                    axisTick: {{
                        lineStyle: {{
                            color: '#2E7D32'
                        }}
                    }},
                    splitLine: {{
                        show: false
                    }}
                }},
                series: [{{
                    type: 'bar',
                    data: cycles,
                    itemStyle: {{
                        color: function(params) {{
                            return colors[params.dataIndex % colors.length];
                        }},
                        borderRadius: [4, 4, 0, 0]
                    }},
                    barWidth: '70%',
                    emphasis: {{
                        itemStyle: {{
                            shadowBlur: 10,
                            shadowOffsetX: 0,
                            shadowColor: 'rgba(0, 0, 0, 0.5)'
                        }}
                    }}
                }}],
                tooltip: {{
                    trigger: 'axis',
                    axisPointer: {{
                        type: 'shadow'
                    }},
                    formatter: function(params) {{
                        return params[0].name + '<br/>' +
                               'cycles: ' + params[0].value.toLocaleString();
                    }}
                }}
            }};

            chart.setOption(option);
            charts[chartId] = chart;
        }}

        // Category tab switching
        document.addEventListener('click', function(e) {{
            if (e.target.classList.contains('cat-tab')) {{
                const category = e.target.getAttribute('data-category');

                // Update tab styles
                document.querySelectorAll('.cat-tab').forEach(tab => tab.classList.remove('active'));
                e.target.classList.add('active');

                // Show/hide charts
                document.querySelectorAll('.chart-container').forEach(container => {{
                    container.style.display = 'none';
                }});
                const activeChart = document.getElementById('chart-' + category);
                if (activeChart) {{
                    activeChart.style.display = 'block';
                    // Resize the chart after showing
                    if (charts[category]) {{
                        setTimeout(() => charts[category].resize(), 100);
                    }}
                }}
            }}
        }});

        // Export to Excel - Export all performance metrics like Performance Summary.md
        function exportToExcel() {{
            // Get active category
            const activeTab = document.querySelector('.cat-tab.active');
            const category = activeTab ? activeTab.getAttribute('data-category') : 'all';

            let models, cycles, extLoad, extStore, mac, meuStall, compute;
            if (category === 'all') {{
                models = allChartData.models;
                cycles = allChartData.cycles;
                extLoad = allChartData.extLoad;
                extStore = allChartData.extStore;
                mac = allChartData.mac;
                meuStall = allChartData.meuStall;
                compute = allChartData.compute;
            }} else {{
                // For category views, we need to filter the allChartData by category
                const catModels = categoryData[category].models;
                const modelIndexMap = {{}};
                allChartData.models.forEach((m, i) => modelIndexMap[m] = i);

                models = catModels;
                cycles = catModels.map(m => allChartData.cycles[modelIndexMap[m]]);
                extLoad = catModels.map(m => allChartData.extLoad[modelIndexMap[m]]);
                extStore = catModels.map(m => allChartData.extStore[modelIndexMap[m]]);
                mac = catModels.map(m => allChartData.mac[modelIndexMap[m]]);
                meuStall = catModels.map(m => allChartData.meuStall[modelIndexMap[m]]);
                compute = catModels.map(m => allChartData.compute[modelIndexMap[m]]);
            }}

            // Create data array with all performance metrics (matching Performance Summary.md format)
            const data = models.map((model, index) => ({{
                'Model': model,
                'Cycles': cycles[index],
                'Ext Load': extLoad[index],
                'Ext Store': extStore[index],
                'MAC': mac[index],
                'MEU Stall': meuStall[index],
                'Compute': compute[index]
            }}));

            const ws = XLSX.utils.json_to_sheet(data);
            const wb = XLSX.utils.book_new();
            XLSX.utils.book_append_sheet(wb, ws, 'Performance Summary');

            const date = new Date();
            const timestamp = date.toISOString().slice(0, 10);
            XLSX.writeFile(wb, `performance_summary_` + timestamp + '.xlsx');
        }}

        // Window resize handler
        window.addEventListener('resize', function() {{
            for (const [id, chart] of Object.entries(charts)) {{
                chart.resize();
            }}
        }});

        document.addEventListener('DOMContentLoaded', initCharts);
    </script>
</body>
</html>'''

    # Write to file
    full_path = os.path.join(base_path, output_file)
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Generated {output_file}")
    print(f"  - Total models: {total_models}")
    print(f"  - Models with performance data: {models_with_perf}")
    print(f"  - Total output tensors: {total_outputs}")
    print(f"  - Output: {full_path}")
    return full_path


def generate_performance_summary(models: List[ModelInfo], base_path: str, output_file: str = "performance_summary.md"):
    """Generate performance_summary.md from model performance data"""

    lines = [
        "# Performance Summary\n",
        "## 4-Batch CNN Models Performance Metrics\n",
        "| Model | Cycles | Ext Load | Ext Store | MAC | MEU Stall | Compute |",
        "|-------|--------|----------|-----------|-----|-----------|---------|"
    ]

    # Sort by cycles (ascending)
    models_with_perf = [m for m in models if m.performance]
    sorted_models = sorted(models_with_perf, key=lambda x: x.performance.cycles)

    for model in sorted_models:
        p = model.performance
        lines.append(
            f"| {model.name} | `{p.cycles:,}` | `{p.ext_load:,}` | "
            f"`{p.ext_store:,}` | `{p.mac:,}` | `{p.meu_stall:,}` | "
            f"`{p.compute:,}` |"
        )

    # Write to file
    full_path = os.path.join(base_path, output_file)
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Generated {output_file}")
    print(f"  - Total models: {len(models_with_perf)}")
    print(f"  - Output: {full_path}")
    return full_path


def generate_markdown_summary(models: List[ModelInfo], base_path: str, output_file: str = "summary_report.md"):
    """Generate the markdown summary report"""

    total_models = len([m for m in models if m.outputs or m.detection_images])
    total_outputs = sum(len(m.outputs) for m in models)

    lines = [
        "# 4-Batch CNN Post-processing Results\n",
        "## Summary\n",
        f"- **Total Models Processed:** {total_models}",
        f"- **Successful:** {total_models}",
        f"- **Failed:** 0",
        f"- **Total Output Tensors:** {total_outputs}\n",
        "## Model Results\n",
    ]

    # Type icons for markdown
    type_icons = {
        "detection": "🎯",
        "segmentation": "🖼️",
        "instance_segmentation": "📦",
        "yolo": "📦",
        "mediapipe": "📦",
        "pose": "🧍",
        "other": "📦"
    }

    for model in models:
        if not model.outputs and not model.detection_images:
            continue

        icon = type_icons.get(model.category, "📦")
        lines.append(f"### {icon} {model.name}")
        lines.append(f"*{model.architecture} - {model.category}*\n")
        lines.append("| Output | Shape | Status |")
        lines.append("|--------|-------|--------|")

        for output_name, shape, status in model.outputs:
            # Format shape for display
            if isinstance(shape, str):
                shape_str = shape
            elif isinstance(shape, list):
                shape_str = str(shape)
            else:
                shape_str = "unknown"
            status_symbol = "✓" if status == "success" else "✗"
            lines.append(f"| {output_name} | {shape_str} | {status_symbol} |")

        lines.append("")

    # Write to file
    full_path = os.path.join(base_path, output_file)
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Generated {output_file}")
    print(f"  - Total models: {total_models}")
    print(f"  - Total output tensors: {total_outputs}")
    print(f"  - Output: {full_path}")
    return full_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate integrated_report.html and summary_report.md by scanning local directories'
    )
    parser.add_argument(
        '-d', '--directory',
        default='.',
        help='Directory containing model results (default: current directory)'
    )
    parser.add_argument(
        '-o', '--output',
        default='integrated_report.html',
        help='Output HTML filename (default: integrated_report.html)'
    )
    parser.add_argument(
        '--no-md',
        action='store_true',
        help='Skip generating summary_report.md'
    )

    args = parser.parse_args()

    base_path = os.path.abspath(args.directory)

    if not os.path.exists(base_path):
        print(f"Error: Directory '{base_path}' does not exist")
        return 1

    print("Scanning directory for model results...")
    print(f"Base path: {base_path}")

    models = scan_model_directory(base_path)

    if not models:
        print("Warning: No model results found")
        return 1

    print(f"Found {len(models)} model directories")

    # Generate performance_summary.md first
    print()
    perf_md_path = generate_performance_summary(models, base_path)

    # Generate HTML report
    output_path = generate_html(models, base_path, args.output)

    # Generate markdown summary
    if not args.no_md:
        print()
        md_path = generate_markdown_summary(models, base_path)

    return 0


if __name__ == "__main__":
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    exit(main())
