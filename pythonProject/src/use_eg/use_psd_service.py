#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功率谱密度计算 - 交互式执行脚本
- 选择预处理后的 HDF5 文件（应包含 TSData/data 和 times）
- 选择 CSV 配置文件（包含采样率、FFT窗口长度等）
- 计算每个通道的 PSD（Welch 法，50% 重叠，汉明窗）
- 保存为 HDF5，并自动复制输入文件的属性（如 channel_names）
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from pythonProject.configs.csv_config_manger import ConfigManager
from pythonProject.src.services.psd_service import PSDCalculator


def main():
    root = tk.Tk()
    root.withdraw()

    # 1. 选择输入 HDF5 文件
    h5_path = filedialog.askopenfilename(
        title="选择预处理后的 HDF5 数据文件",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    if not h5_path:
        print("未选择输入文件，退出。")
        return

    # 2. 选择 CSV 配置文件
    cfg_path = filedialog.askopenfilename(
        title="选择 CSV 配置文件",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not cfg_path:
        print("未选择配置文件，退出。")
        return

    # 3. 选择输出目录（可选）
    output_dir = filedialog.askdirectory(title="选择输出目录")
    if not output_dir:
        output_dir = os.path.dirname(h5_path)
        print(f"未选择输出目录，使用输入文件所在目录: {output_dir}")

    # 自动生成输出文件名：原文件名 + "_psd.h5"
    base_name = os.path.splitext(os.path.basename(h5_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_psd.h5")

    # 4. 加载配置
    cfg_mgr = ConfigManager.from_csv(cfg_path)
    # 存储配置文件路径（以备扩展，如相对路径解析）
    cfg_mgr._config_path = cfg_path

    # 5. 创建 PSD 计算器并执行
    try:
        calculator = PSDCalculator(h5_path, cfg_mgr)
        result = calculator.compute_psd(output_path)
        print("PSD 计算完成")
        print(f"频率点数: {len(result['freqs'])}")
        print(f"PSD 形状: {result['psd'].shape}")
        print(f"结果已保存至: {output_path}")
        messagebox.showinfo("完成", f"PSD 计算完成！\n输出文件：{output_path}")
    except Exception as e:
        messagebox.showerror("错误", f"PSD 计算失败：{str(e)}")
        raise


if __name__ == "__main__":
    main()