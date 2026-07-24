#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MT 视电阻率与相位计算脚本
输入：滤波后的 HDF5 文件（包含 TSData 数据集及 sample_rate 属性）
输出：HDF5 文件，包含频率、视电阻率（XY/YX）、相位（XY/YX）、阻抗张量元素
"""

import h5py
import numpy as np
from datetime import datetime
import sys
import os
from mt_service import PostProcessor


def compute_rho_phase_from_impedance(freqs, Z):
    """
    由复数阻抗计算视电阻率和相位
    参数:
        freqs : ndarray, 频率 (Hz)
        Z : ndarray, 复数阻抗 (Ω)
    返回:
        rho : ndarray, 视电阻率 (Ω·m)
        phase : ndarray, 相位 (度)
    """
    omega = 2.0 * np.pi * freqs
    mu0 = 4.0 * np.pi * 1e-7
    # 避免 omega=0 时除零
    omega_safe = np.where(omega == 0, 1e-30, omega)
    rho = (np.abs(Z) ** 2) / (mu0 * omega_safe)
    phase = np.degrees(np.arctan2(np.imag(Z), np.real(Z)))
    return rho, phase


def compute_mt_response(
    h5_path,
    nfft=1024,
    overlap=0.5,
    ex_idx=0,
    ey_idx=1,
    hx_idx=2,
    hy_idx=3,
    output_path=None
):
    """
    主函数：读取 HDF5，计算阻抗，保存结果

    参数:
        h5_path      : 输入 HDF5 文件路径
        nfft         : FFT 段长度（建议 2 的幂）
        overlap      : 重叠比例 (0~1)
        ex_idx, ey_idx, hx_idx, hy_idx : 通道列索引（默认 [Ex, Ey, Hx, Hy]）
        output_path  : 输出文件路径（若为 None 则自动生成）
    """
    # ---------- 1. 读取数据 ----------
    with h5py.File(h5_path, 'r') as f:
        if 'TSData' not in f:
            raise KeyError("HDF5 中缺少 'TSData' 数据集")
        data = f['TSData'][:]
        data[:, hx_idx:hy_idx + 1] *= 1e-9  # 将 nT 转为 T

        # 读取采样率（必须）
        try:
            sample_rate = f.attrs['sample_rate']
        except KeyError:
            raise ValueError("HDF5 属性中缺少 'sample_rate'，无法继续")

        # 可选读取通道名称（仅供打印信息）
        ch_names = f.attrs.get('channel_names', None)
        if ch_names is not None:
            print(f"通道名称: {ch_names}")

    n_samples, n_ch = data.shape
    if n_ch < 4:
        raise ValueError(f"数据只有 {n_ch} 个通道，需要至少 4 个 (Ex, Ey, Hx, Hy)")

    print(f"数据形状: {data.shape}, 采样率: {sample_rate} Hz")
    print(f"假设通道索引: Ex={ex_idx}, Ey={ey_idx}, Hx={hx_idx}, Hy={hy_idx}")

    # ---------- 2. 使用 PostProcessor 计算传递函数 ----------
    proc = PostProcessor(sample_rate)

    # 去尖峰（可根据需要开启，此处建议执行）
    print("正在进行去尖峰处理...")
    data_clean = proc.remove_spikes(data, method='mad', threshold=5.0, window_len=51)

    print(f"计算传递函数，nfft={nfft}, overlap={overlap}")
    # 以 Hx 为参考 -> 得到 Ex/Hx 和 Ey/Hx
    freqs, H_ref_hx, _ = proc.robust_estimate(
        data_clean, nfft=nfft, overlap=overlap, ref_channel=hx_idx
    )
    # 以 Hy 为参考 -> 得到 Ex/Hy 和 Ey/Hy
    freqs, H_ref_hy, _ = proc.robust_estimate(
        data_clean, nfft=nfft, overlap=overlap, ref_channel=hy_idx
    )

    # 组合阻抗张量元素
    Zxy = H_ref_hy[ex_idx, :]   # Ex / Hy
    Zyx = H_ref_hx[ey_idx, :]   # Ey / Hx
    # 可选计算 Zxx 和 Zyy（用于质量评估）
    Zxx = H_ref_hx[ex_idx, :]   # Ex / Hx
    Zyy = H_ref_hy[ey_idx, :]   # Ey / Hy

    # ---------- 3. 计算视电阻率和相位 ----------
    rho_xy, phase_xy = compute_rho_phase_from_impedance(freqs, Zxy)
    rho_yx, phase_yx = compute_rho_phase_from_impedance(freqs, Zyx)

    # ---------- 4. 保存结果 HDF5 ----------
    if output_path is None:
        base, ext = os.path.splitext(h5_path)
        output_path = f"{base}_response.h5"

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('frequency', data=freqs)
        f.create_dataset('rho_xy', data=rho_xy)
        f.create_dataset('phase_xy', data=phase_xy)
        f.create_dataset('rho_yx', data=rho_yx)
        f.create_dataset('phase_yx', data=phase_yx)
        # 同时保存复数阻抗（可直接存储为 complex128）
        f.create_dataset('impedance_xy', data=Zxy)
        f.create_dataset('impedance_yx', data=Zyx)
        # 可选保存 Zxx, Zyy
        f.create_dataset('impedance_xx', data=Zxx)
        f.create_dataset('impedance_yy', data=Zyy)

        # 添加属性元数据
        f.attrs['sample_rate'] = sample_rate
        f.attrs['nfft'] = nfft
        f.attrs['overlap'] = overlap
        f.attrs['processing_date'] = datetime.now().isoformat()
        f.attrs['input_file'] = os.path.basename(h5_path)
        f.attrs['units'] = 'rho: Ohm.m, phase: degrees, impedance: (V/m)/T'
        f.attrs['channel_indices'] = f"Ex={ex_idx}, Ey={ey_idx}, Hx={hx_idx}, Hy={hy_idx}"

    print(f"✅ 结果已保存至: {output_path}")
    print(f"   频率点数: {len(freqs)}")
    print(f"   频率范围: {freqs[0]:.3f} ~ {freqs[-1]:.3f} Hz")


# ---------- 命令行 / 交互式入口 ----------
if __name__ == "__main__":
    # 如果命令行提供了输入文件路径
    if len(sys.argv) > 1:
        h5_file = sys.argv[1]
    else:
        # 否则弹出文件选择对话框（需要 tkinter）
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            h5_file = filedialog.askopenfilename(
                title="选择滤波后的 HDF5 文件",
                filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
            )
            if not h5_file:
                print("未选择文件，退出。")
                sys.exit()
        except ImportError:
            print("未安装 tkinter，请通过命令行参数指定文件路径")
            sys.exit(1)

    # 可在此修改默认参数（nfft, overlap, 通道索引）
    compute_mt_response(
        h5_path=h5_file,
        nfft=65536,          # FFT 长度，根据数据长度调整（一般取 2^n）
        overlap=0.5,        # 重叠率
        ex_idx=0,           # 若你的通道顺序不同，请修改这 4 个索引
        ey_idx=1,
        hx_idx=2,
        hy_idx=3
    )
