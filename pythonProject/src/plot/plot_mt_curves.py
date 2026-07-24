#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制 MT 视电阻率与相位曲线（XY/YX 模式）
输入：由 compute_mt_response.py 生成的 HDF5 文件
输出：PNG 或 PDF 格式的图片，并可显示于屏幕
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation, AutoMinorLocator
import sys
import os

# 设置绘图风格（可自定义）
plt.style.use('seaborn-v0_8-darkgrid')  # 若没有 seaborn，可改回 'default'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 2.0


def format_freq_axis(ax):
    """辅助函数：设置频率轴为对数刻度，带次刻度"""
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis='x', which='minor', bottom=True)


def plot_mt_curves(h5_path, output_path=None, show_phase_limits=True, freq_high_to_low=True):
    """
    绘制 MT 曲线
    参数:
        freq_high_to_low: bool, 若 True，横坐标从高频到低频（左侧高频），默认 True
    """
    with h5py.File(h5_path, 'r') as f:
        freqs = f['frequency'][:]
        rho_xy = f['rho_xy'][:]
        phase_xy = f['phase_xy'][:]
        rho_yx = f['rho_yx'][:]
        phase_yx = f['phase_yx'][:]

    # 过滤无效值
    valid = np.isfinite(rho_xy) & np.isfinite(rho_yx) & \
            np.isfinite(phase_xy) & np.isfinite(phase_yx)
    freqs = freqs[valid]
    rho_xy = rho_xy[valid]
    phase_xy = phase_xy[valid]
    rho_yx = rho_yx[valid]
    phase_yx = phase_yx[valid]

    if len(freqs) == 0:
        raise ValueError("所有数据点均为无效值")

    # 若需要高频到低频，则反转所有数组
    if freq_high_to_low:
        freqs = freqs[::-1]
        rho_xy = rho_xy[::-1]
        phase_xy = phase_xy[::-1]
        rho_yx = rho_yx[::-1]
        phase_yx = phase_yx[::-1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 视电阻率
    ax1.loglog(freqs, rho_xy, 'bo-', markersize=1, linewidth=0.5, label=r'$\rho_{xy}$')
    ax1.loglog(freqs, rho_yx, 'rs-', markersize=1, linewidth=0.5, label=r'$\rho_{yx}$')
    ax1.set_ylabel(r'Apparent Resistivity ($\Omega \cdot$m)', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='both', direction='in', top=True, right=True)

    # 相位
    ax2.semilogx(freqs, phase_xy, 'bo-', markersize=1, linewidth=0.5, label=r'$\phi_{xy}$')
    ax2.semilogx(freqs, phase_yx, 'rs-', markersize=1, linewidth=0.5, label=r'$\phi_{yx}$')
    ax2.set_xlabel('Frequency (Hz)', fontsize=14)
    ax2.set_ylabel('Phase (degrees)', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='both', direction='in', top=True, right=True)

    if show_phase_limits:
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.6)
        ax2.axhline(90, color='gray', linestyle=':', alpha=0.4)
        ax2.axhline(-90, color='gray', linestyle=':', alpha=0.4)

    fig.suptitle(f'MT Response Curves (freq high->low)', fontsize=16)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {output_path}")
    else:
        plt.show()


# ---------- 命令行 / 交互式入口 ----------
if __name__ == "__main__":
    # 若命令行给了文件路径，则直接使用
    if len(sys.argv) > 1:
        h5_file = sys.argv[1]
    else:
        # 弹出文件选择对话框（如果 tkinter 可用）
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            h5_file = filedialog.askopenfilename(
                title="选择 MT 响应 HDF5 文件",
                filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
            )
            if not h5_file:
                print("未选择文件，退出。")
                sys.exit()
        except ImportError:
            print("未安装 tkinter，请通过命令行参数指定文件路径")
            sys.exit(1)

    # 定义输出图片路径（可选）
    # 若希望自动保存，可取消注释下面两行
    base = os.path.splitext(h5_file)[0]
    out_png = base + "_curves.png"

    # 绘图（若想保存，将 output_path 设为路径字符串）
    plot_mt_curves(h5_file, output_path=None, show_phase_limits=True)