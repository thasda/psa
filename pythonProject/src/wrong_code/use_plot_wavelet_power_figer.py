"""
实际数据小波变换绘图脚本
功能：
1. 通过对话框选择已保存的 CWT 结果文件（cwt_results.h5）和相干性结果文件（可选）
2. 生成并保存可视化图像：能量谱图（固定频率范围 0.05–500 Hz）
"""
import os
import tkinter as tk
from tkinter import filedialog
import h5py
import numpy as np
from pythonProject.src.wrong_code.plot_wavelet_power_figer import WaveletPlotter


def select_hdf5_file(title="选择 HDF5 文件"):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=[("HDF5 files", "*.h5")])
    root.destroy()
    if not file_path:
        exit()
    return file_path


def select_output_dir():
    root = tk.Tk()
    root.withdraw()
    dir_path = filedialog.askdirectory(title="选择输出目录")
    root.destroy()
    if not dir_path:
        dir_path = os.getcwd()
    return dir_path


def load_and_fix_cwt(plotter, file_path, freq_range=(0.05, 500)):
    """
    统一加载 CWT 结果：
    - 若 load_cwt_results 直接得到 3D 数组，则保持
    - 若只加载到 2D（分组格式），则手动合并所有 cwt_* 通道
    最终确保 plotter.cwt_coeffs 为 (n_freq, n_time, n_channels)
    并且按 freq_range 裁剪数据。
    """
    # 先尝试标准加载
    plotter.load_cwt_results(file_path)

    # 如果加载后是 2D，说明是分组格式，需要手动合并
    if plotter.cwt_coeffs is not None and plotter.cwt_coeffs.ndim == 2:
        print("检测到分组格式，正在合并所有通道...")
        with h5py.File(file_path, 'r') as f:
            groups = sorted([k for k in f.keys() if k.startswith('cwt_')])
            coeffs_list = []
            names = []
            freqs_common = None
            for key in groups:
                grp = f[key]
                coeffs_ch = grp['coeffs'][:]          # (n_freq, n_time)
                if freqs_common is None:
                    freqs_common = grp['frequencies'][:]
                coeffs_list.append(coeffs_ch)
                names.append(key.replace('cwt_', ''))
            plotter.cwt_coeffs = np.stack(coeffs_list, axis=2)  # (n_freq, n_time, n_ch)
            plotter.frequencies = freqs_common
            plotter.channel_names = names
            # 时间轴沿用已加载的
            print(f"✅ 合并完成，形状: {plotter.cwt_coeffs.shape}")

    # 裁剪频率范围
    if plotter.frequencies is not None:
        freq_mask = (plotter.frequencies >= freq_range[0]) & (plotter.frequencies <= freq_range[1])
        plotter.cwt_coeffs = plotter.cwt_coeffs[freq_mask, :, :]
        plotter.frequencies = plotter.frequencies[freq_mask]
        print(f"✅ 频率已裁剪至 {freq_range[0]}-{freq_range[1]} Hz")


def main():
    print("=" * 60)
    print("小波变换绘图工具")
    print("=" * 60)

    # 选择 CWT 结果文件
    cwt_file = select_hdf5_file("选择 CWT 结果文件（cwt_results.h5）")
    print(f"CWT 文件: {cwt_file}")

    # 选择输出目录
    output_dir = select_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 初始化绘图器并进行智能加载与频率裁剪
    plotter = WaveletPlotter()
    load_and_fix_cwt(plotter, cwt_file, freq_range=(0.05, 500))

    # 绘制能量谱图
    base_name = os.path.splitext(os.path.basename(cwt_file))[0]
    energy_fig = os.path.join(output_dir, f"{base_name}_energy_spectra.png")
    plotter.plot_energy_spectra(save_path=energy_fig, show=False)
    print(f"能量谱图已保存: {energy_fig}")

    # 若需要相干性，可在此处添加类似逻辑
    print("所有图像生成完成！")


if __name__ == "__main__":
    main()