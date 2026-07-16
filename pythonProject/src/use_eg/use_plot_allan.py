# src/use_eg/plot_allan_to_output.py
"""
加载艾伦方差结果文件，绘制并保存：
1. 分组图（两个子图：电场和磁场）
2. 所有通道在一张图上的单图（带自定义标签 Ex, Ey, Bx, By, Bz）
"""
import os
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
from pythonProject.src.plot.plot_allan import AllanPlotter  # 确保路径正确

def main():
    # 文件路径
    allan_file = Path(r"F:\Anylysis_project\pythonProject\output\Ground\Ground.h5.allan_result.h5")
    output_dir = Path(r"F:\Anylysis_project\pythonProject\output\Ground")

    if not allan_file.exists():
        print(f"错误：艾伦方差结果文件不存在: {allan_file}")
        return

    plotter = AllanPlotter(str(allan_file))
    labels = ["Ex", "Ey", "Bx", "By", "Bz"]

    # --- 1. 保存分组图（与PSD风格一致）---
    groups = [
        {"indices": [0, 1], "labels": labels[:2], "title": "Electric Channels"},
        {"indices": [2, 3, 4], "labels": labels[2:], "title": "Magnetic Channels"}
    ]
    fig_group, axes = plotter.plot_groups(groups, linewidth=1.0, figsize=(14, 6))
    base_name = os.path.splitext(os.path.basename(allan_file))[0]
    group_image = os.path.join(output_dir, f"{base_name}_allan_variance_grouped.png" )  # 文件名改为 variance
    fig_group.savefig(group_image, dpi=300, bbox_inches='tight')
    plt.close(fig_group)
    print(f"分组图已保存: {group_image}")



if __name__ == "__main__":
    main()