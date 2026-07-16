import sys
from pathlib import Path
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# 将项目根目录添加到 Python 路径，便于绝对导入
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pythonProject.src.plot.plot_psd import PSDPlotter


def select_file(title="选择文件", filetypes=[("HDF5 files", "*.h5")]):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path


def select_directory(title="选择输出目录"):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return path


def main():
    # 1. 选择 PSD 结果文件
    psd_file = select_file("选择 PSD 结果文件（*.h5）", [("HDF5 files", "*.h5")])
    if not psd_file:
        print("未选择输入文件，退出。")
        return

    # 2. 选择输出目录
    output_dir = select_directory("选择输出目录")
    if not output_dir:
        output_dir = os.path.dirname(psd_file)
        print(f"未选择输出目录，使用输入文件所在目录: {output_dir}")

    # 自动生成输出文件名
    base_name = os.path.splitext(os.path.basename(psd_file))[0]  # 例如 "Ground_psd"
    output_path = os.path.join(output_dir, f"{base_name}_plot.png")

    # 3. 创建绘图器
    plotter = PSDPlotter(psd_file)

    # 定义分组（假设通道顺序为 Ex, Ey, Bx, By, Bz）
    groups = [
        {"indices": [0, 1], "labels": ["Ex", "Ey"], "title": "Electric Channels (Ex, Ey)"},
        {"indices": [2, 3, 4], "labels": ["Bx", "By", "Bz"], "title": "Magnetic Channels (Bx, By, Bz)"}
    ]

    # 绘制分组图像
    fig, axes = plotter.plot_groups(
        groups,
        linewidth=0.5,
        figsize=(6, 9)
    )

    # 添加总标题
    fig.suptitle("Power Spectral Density", fontsize=14)
    plt.tight_layout()

    # 保存图像
    plotter.save_figure(output_path, dpi=300)
    print(f"图像已保存至: {output_path}")

    # 可选：显示图像
    plt.show()


if __name__ == "__main__":
    main()
