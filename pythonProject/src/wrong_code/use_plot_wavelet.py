import matplotlib
matplotlib.use('TkAgg')

import tkinter as tk
from tkinter import filedialog
import os
from plot_wavelet import WaveletPlotter


def select_file(title="选择文件", filetypes=[("HDF5 files", "*.h5")]):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path


def main():
    print("=" * 60)
    print("小波变换绘图工具")
    print("=" * 60)

    # 选择 CWT 结果文件
    cwt_file = select_file("选择 CWT 结果文件（*.h5）", [("HDF5 files", "*.h5")])
    if not cwt_file:
        print("未选择文件，退出。")
        return
    print(f"CWT 文件: {cwt_file}")

    # 选择输出目录
    output_dir = filedialog.askdirectory(title="选择输出目录")
    if not output_dir:
        output_dir = os.path.dirname(cwt_file)
        print(f"未选择输出目录，使用文件所在目录: {output_dir}")
    else:
        print(f"输出目录: {output_dir}")

    # 初始化绘图器并加载数据
    plotter = WaveletPlotter()
    plotter.load_cwt_results(cwt_file)

    # 自动生成输出文件名
    base_name = os.path.splitext(os.path.basename(cwt_file))[0]
    save_path = os.path.join(output_dir, f"{base_name}_energy_spectra.png")

    # 绘制能量谱（可根据需要调整参数）
    plotter.plot_energy_spectra(
        save_path=save_path,
        channels=None,          # 绘制所有通道
        show_cbar=True,
        db_range=(-20, 20),
        decimate_time=10,       # 时间降采样提升绘图速度
        decimate_freq=1,
        show=True               # 显示图形
    )

    print("全部完成！")


if __name__ == "__main__":
    main()