import os
import matplotlib
matplotlib.use('TkAgg')
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, List, Tuple


def plot_compare_ground_underground(
    input_file1: str,
    input_file2: str,
    output_fig: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    dpi: int = 300
) -> None:
    """
    对比 ground 和 underground 两个 HDF5 文件的时间序列数据，
    分别绘制五个通道的子图。

    Parameters
    ----------
    ground_path : str
        ground 数据的 HDF5 文件路径
    underground_path : str
        underground 数据的 HDF5 文件路径
    output_fig : str, optional
        保存图片的路径（如 'compare.png'），不提供则显示
    figsize : tuple, optional
        图形尺寸，默认 (12, 10)
    start_time : str, optional
        截取显示的时间范围起始，格式 'YYYY-MM-DD HH:MM:SS.ffffff'
    end_time : str, optional
        截取显示的时间范围结束
    dpi : int, optional
        图像分辨率，默认 150

    Notes
    -----
    通道映射：
        ground:  Ey  -> underground: ch1
        ground:  Bx  -> underground: ch2
        ground:  Bz  -> underground: ch0
        ground:  By  -> underground: ch4
        ground:  Ex  -> underground: ch3
    """
    # 1. 读取 ground 数据
    with h5py.File(input_file1, 'r') as fg:
        times_g = fg['times'][:]
        ts_g = fg['TSData'][:]
        ch_names_g = fg.attrs.get('channel_names')
        if ch_names_g is not None:
            if isinstance(ch_names_g, (list, np.ndarray)):
                ch_names_g = [name.decode('utf-8') if isinstance(name, bytes) else str(name) for name in ch_names_g]
            else:
                ch_names_g = [str(ch_names_g)]
        else:
            ch_names_g = [f'ch{i}' for i in range(ts_g.shape[1])]

    # 2. 读取 underground 数据
    with h5py.File(input_file2, 'r') as fu:
        times_u = fu['times'][:]
        ts_u = fu['TSData'][:]
        ch_names_u = fu.attrs.get('channel_names')
        if ch_names_u is not None:
            if isinstance(ch_names_u, (list, np.ndarray)):
                ch_names_u = [name.decode('utf-8') if isinstance(name, bytes) else str(name) for name in ch_names_u]
            else:
                ch_names_u = [str(ch_names_u)]
        else:
            ch_names_u = [f'ch{i}' for i in range(ts_u.shape[1])]

    # 3. 转换时间字符串为 datetime 对象（便于绘图和截取）
    def parse_times(times_arr):
        if times_arr.dtype.kind == 'S':
            return [datetime.fromisoformat(t.decode('utf-8')) for t in times_arr]
        elif times_arr.dtype.kind == 'U':
            return [datetime.fromisoformat(str(t)) for t in times_arr]
        else:
            return [datetime.fromisoformat(str(t)) for t in times_arr]

    times_g_dt = parse_times(times_g)
    times_u_dt = parse_times(times_u)

    # 可选：检查时间是否一致
    if len(times_g_dt) != len(times_u_dt):
        print("警告：ground 和 underground 数据点数不同，将使用较短长度进行对齐")
        min_len = min(len(times_g_dt), len(times_u_dt))
        times_g_dt = times_g_dt[:min_len]
        times_u_dt = times_u_dt[:min_len]
        ts_g = ts_g[:min_len, :]
        ts_u = ts_u[:min_len, :]
    else:
        # 检查时间戳是否一致（允许微小误差）
        for i, (tg, tu) in enumerate(zip(times_g_dt, times_u_dt)):
            if abs((tg - tu).total_seconds()) > 1e-6:
                print(f"警告：第 {i} 个时间点不一致: {tg} vs {tu}")
                break

    # 4. 根据时间范围截取（如果提供了 start_time / end_time）
    if start_time is not None:
        start_dt = datetime.fromisoformat(start_time)
        # 找到第一个 >= start_dt 的索引
        indices = [i for i, t in enumerate(times_g_dt) if t >= start_dt]
        if indices:
            start_idx = indices[0]
        else:
            start_idx = 0
    else:
        start_idx = 0

    if end_time is not None:
        end_dt = datetime.fromisoformat(end_time)
        indices = [i for i, t in enumerate(times_g_dt) if t <= end_dt]
        if indices:
            end_idx = indices[-1] + 1
        else:
            end_idx = len(times_g_dt)
    else:
        end_idx = len(times_g_dt)

    times_g_dt = times_g_dt[start_idx:end_idx]
    times_u_dt = times_u_dt[start_idx:end_idx]
    ts_g = ts_g[start_idx:end_idx, :]
    ts_u = ts_u[start_idx:end_idx, :]

    if len(times_g_dt) == 0:
        print("错误：截取后没有数据")
        return

    # 5. 映射通道
    # ground: 需要查找 'Ey','Bx','Bz','By','Ex' 在 ch_names_g 中的索引
    ground_targets = ['Ex', 'Ey', 'Hx', 'Hz', 'Hy']   # ['Ex', 'Ey', 'Hx', 'Hz', 'Hy'] ['ch0', 'ch1', 'ch2', 'ch3', 'ch4']
    underground_targets = ['Ex', 'Ey', 'Hx', 'Hz', 'Hy']  # ['Ex', 'Ey', 'Hx', 'Hz', 'Hy'] ['ch0', 'ch1', 'ch2', 'ch3', 'ch4']

    g_indices = []
    for name in ground_targets:
        try:
            idx = ch_names_g.index(name)
        except ValueError:
            # 如果找不到，尝试大小写不敏感
            lower_names = [n.lower() for n in ch_names_g]
            try:
                idx = lower_names.index(name.lower())
            except ValueError:
                raise ValueError(f"ground 文件中找不到通道 '{name}'，可用通道: {ch_names_g}")
        g_indices.append(idx)

    u_indices = []
    for name in underground_targets:
        try:
            idx = ch_names_u.index(name)
        except ValueError:
            lower_names = [n.lower() for n in ch_names_u]
            try:
                idx = lower_names.index(name.lower())
            except ValueError:
                raise ValueError(f"underground 文件中找不到通道 '{name}'，可用通道: {ch_names_u}")
        u_indices.append(idx)

    # 提取数据
    data_g = ts_g[:, g_indices]  # shape (n_samples, 5)
    data_u = ts_u[:, u_indices]  # shape (n_samples, 5)

    # 6. 绘图
    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True, dpi=dpi)
    if len(axes) == 1:
        axes = [axes]

    for i in range(5):
        ax = axes[i]
        # 主坐标轴：ground 数据（左侧）
        ax.plot(times_g_dt, data_g[:, i], 'k-', linewidth=0.8, label='ground')
        ax.set_ylabel(f'{ground_targets[i]} (ground)', fontsize=10, color='k')
        ax.tick_params(axis='y', labelcolor='k')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.margins(0.05)

        # 次坐标轴：underground 数据（右侧）
        ax2 = ax.twinx()
        ax2.plot(times_u_dt, data_u[:, i], 'b-', linewidth=0.8, label='underground')
        ax2.set_ylabel(f'{underground_targets[i]} (underground)', fontsize=10, color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        # 可调节刻度范围，这里自动根据数据调整
        ax2.margins(0.05)

        # 合并图例（两条曲线的图例放在一起）
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    axes[-1].set_xlabel('Time')
    fig.suptitle('Ground vs Underground Time Series Comparison', fontsize=14)

    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_fig:
        plt.savefig(output_fig, dpi=dpi, bbox_inches='tight')
        print(f"图片已保存至 {output_fig}")
    else:
        plt.show()



import tkinter as tk
from tkinter import filedialog


def select_file(title="选择文件", filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]):
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


# ==================== 使用示例 ====================
if __name__ == '__main__':

    input_file1 = select_file("选择预处理后的ground HDF5 数据文件", [("HDF5 files", "*.h5")])
    input_file2 = select_file("选择预处理后的underground HDF5 数据文件", [("HDF5 files", "*.h5")])
    if not input_file1 and input_file2:
        print("未正确选择输入文件，退出。")
        exit()

    # input_file1 = r"E:\my_data\数据文件\ground20221105_02_00-10_process.h5"
    # input_file2 = r"E:\my_data\数据文件\underground20221105_02_00-10_process.h5"

    # input_file1 = r"E:\my_data\数据文件\ground20221105_02_00-10.h5"
    # input_file2 = r"E:\my_data\数据文件\underground20221105_02_00-10.h5"

    # # 绘制全部数据
    plot_compare_ground_underground(input_file1, input_file2, output_fig="ts_process_fileter_compare_subset.png")       #记得修改名字

    # # 或者只绘制特定时间段（例如前30秒）
    # plot_compare_ground_underground(
    #     input_file1,
    #    input_file2,
    #     start_time='2022-11-05 02:00:10.000000',
    #     end_time='2022-11-05 02:00:10.100000',
    #     output_fig='compare_subset.png'
    # )
