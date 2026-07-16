#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制 Phoenix MT 数据合并后的 HDF5 时间序列图
功能：
    1. 弹窗选择 merged_24k.h5 和 merged_150.h5 文件
    2. 读取数据（支持存储在根目录或 /24k、/150 组下）
    3. 分别绘制时间序列图，并保存到用户指定的文件夹
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
import h5py
import numpy as np
import matplotlib.pyplot as plt


def select_h5_file(title: str) -> str:
    """弹窗选择 HDF5 文件，返回路径；若取消返回空字符串"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path


def select_save_folder(title: str) -> str:
    """弹窗选择保存图片的文件夹，返回路径；若取消返回空字符串"""
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder


def read_time_series(h5_path: str, group_path: str = None) -> tuple:
    """
    从 HDF5 文件中读取时间序列数据和采样率。
    支持两种结构：
        1. 数据存储在根目录的 'data' 数据集，采样率在根属性
        2. 数据存储在指定组下（如 '/24k/data'），采样率在同组属性
    参数：
        h5_path: HDF5 文件路径
        group_path: 组名（如 '/24k'），若为 None 则尝试根目录
    返回：
        (data, sample_rate) 元组
    """
    with h5py.File(h5_path, 'r') as hf:
        if group_path and group_path in hf:
            grp = hf[group_path]
            data = grp['data'][:]
            sr = grp.attrs.get('sample_rate_hz', None)
            if sr is None:
                raise KeyError(f"组 {group_path} 中没有 'sample_rate_hz' 属性")
        else:
            # 尝试根目录
            if 'data' in hf:
                data = hf['data'][:]
                sr = hf.attrs.get('sample_rate_hz', None)
            else:
                raise KeyError("文件中未找到 'data' 数据集，且未指定有效组")
        return data, sr


def plot_and_save(data: np.ndarray, sample_rate: float, title: str,
                  save_path: str, max_points: int = 100000):
    # 设置中文字体（避免警告）
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    total_points = len(data)
    if total_points > max_points:
        indices = np.linspace(0, total_points - 1, max_points, dtype=int)
        plot_data = data[indices]
        time = indices / sample_rate
        note = f" (降采样至 {max_points} 点，原始 {total_points} 点)"
    else:
        plot_data = data
        time = np.arange(total_points) / sample_rate
        note = ""

    plt.figure(figsize=(12, 5))
    plt.plot(time, plot_data, linewidth=0.5, color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Counts")
    plt.title(f"{title}{note}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"已保存图片：{save_path}")


def main():
    # 1. 选择两个 HDF5 文件
    file_24k = select_h5_file("请选择 merged_24k.h5 文件")
    if not file_24k:
        print("未选择 24kHz 文件，程序退出。")
        return

    file_150 = select_h5_file("请选择 merged_150.h5 文件")
    if not file_150:
        print("未选择 150Hz 文件，程序退出。")
        return

    # 2. 选择保存图片的文件夹
    save_folder = select_save_folder("请选择图片保存文件夹")
    if not save_folder:
        print("未选择保存文件夹，程序退出。")
        return

    # 3. 读取数据（根据合并脚本保存的结构：24k数据在 /24k 组，150数据在 /150 组）
    try:
        data_24k, sr_24k = read_time_series(file_24k, group_path='/24k')
        print(f"24kHz 数据读取成功：{len(data_24k)} 点，采样率 {sr_24k} Hz")
    except Exception as e:
        messagebox.showerror("错误", f"读取 24kHz 文件失败：{e}")
        return

    try:
        data_150, sr_150 = read_time_series(file_150, group_path='/150')
        print(f"150Hz 数据读取成功：{len(data_150)} 点，采样率 {sr_150} Hz")
    except Exception as e:
        messagebox.showerror("错误", f"读取 150Hz 文件失败：{e}")
        return

    # 4. 生成保存路径
    base_name_24k = os.path.splitext(os.path.basename(file_24k))[0]
    base_name_150 = os.path.splitext(os.path.basename(file_150))[0]
    save_path_24k = os.path.join(save_folder, f"{base_name_24k}_timeseries.png")
    save_path_150 = os.path.join(save_folder, f"{base_name_150}_timeseries.png")

    # 5. 绘图并保存
    plot_and_save(data_24k, sr_24k, f"24kHz Time Series ({base_name_24k})",
                  save_path_24k, max_points=100000)
    plot_and_save(data_150, sr_150, f"150Hz Time Series ({base_name_150})",
                  save_path_150, max_points=100000)

    messagebox.showinfo("完成", f"两张图片已保存至：\n{save_folder}")


if __name__ == "__main__":
    main()