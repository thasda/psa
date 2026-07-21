"""
plot_h5_tsdata.py

绘制单个 HDF5 文件的时间序列图，所有信息均从文件内读取。
- 自动识别通道名称（从属性或数据集）
- 自动解析时间轴（ISO 字符串或数值）
- 支持交互式选择通道、时间范围截取
- 自动降采样（数据点过多时）
- 输出高分辨率 PNG 图片
"""

import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from pathlib import Path


# ==================== 辅助交互函数 ====================
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


def ask_yes_no(question):
    root = tk.Tk()
    root.withdraw()
    ans = messagebox.askyesno("确认", question)
    root.destroy()
    return ans


def ask_string(prompt, initial=""):
    root = tk.Tk()
    root.withdraw()
    val = simpledialog.askstring("输入参数", prompt, initialvalue=initial)
    root.destroy()
    return val


def ask_channel_selection(channel_names: List[str]) -> List[int]:
    """交互式选择要绘制的通道索引（或输入 'all' 全部）"""
    if not channel_names:
        return []
    prompt = "可用通道：\n" + "\n".join([f"{i}: {name}" for i, name in enumerate(channel_names)])
    prompt += "\n\n请输入要绘制的通道索引（逗号分隔），或输入 'all' 绘制全部："
    user_input = ask_string("选择通道", initial="all")
    if user_input is None:
        return []
    user_input = user_input.strip().lower()
    if user_input == 'all':
        return list(range(len(channel_names)))
    indices = []
    for part in user_input.split(','):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 0 <= idx < len(channel_names):
                indices.append(idx)
            else:
                print(f"警告：索引 {idx} 超出范围，忽略。")
        else:
            # 尝试按名称匹配
            matches = [i for i, name in enumerate(channel_names) if name.lower() == part.lower()]
            if matches:
                indices.extend(matches)
            else:
                print(f"警告：无法识别 '{part}'，忽略。")
    indices = sorted(set(indices))
    if not indices:
        print("未选择有效通道，将绘制全部通道。")
        return list(range(len(channel_names)))
    return indices


# ==================== 核心绘图函数 ====================
def plot_single_file(
    input_path: str,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    channels: Optional[List[int]] = None,
    dpi: int = 300,
    show: bool = True
) -> None:
    """
    从 HDF5 文件读取数据并绘制时间序列。

    Parameters
    ----------
    input_path : str
        HDF5 文件路径，必须包含 TSData 和 times 数据集。
    output_path : str, optional
        输出图片路径，不提供则自动生成。
    figsize : tuple
        图形尺寸。
    start_time, end_time : datetime, optional
        截取的时间范围（仅当 times 为 datetime 类型时有效）。
    channels : list of int, optional
        要绘制的通道索引，为 None 时绘制全部。
    dpi : int
        输出图像分辨率。
    show : bool
        是否显示图形窗口。
    """
    # ---------- 读取 HDF5 ----------
    with h5py.File(input_path, 'r') as f:
        # 数据集
        if 'TSData' in f:
            tsdata = f['TSData']
        elif 'data' in f:
            tsdata = f['data']
        else:
            raise KeyError("未找到 TSData 或 data 数据集")

        total_points = tsdata.shape[0]
        n_channels = tsdata.shape[1] if len(tsdata.shape) == 2 else 1
        print(f"数据点数: {total_points}, 通道数: {n_channels}")

        # 通道名称
        channel_names = None
        if 'channel_names' in f.attrs:
            ch_attr = f.attrs['channel_names']
            if isinstance(ch_attr, (list, tuple, np.ndarray)):
                channel_names = [c.decode() if isinstance(c, bytes) else str(c) for c in ch_attr]
            else:
                channel_names = [ch_attr.decode() if isinstance(ch_attr, bytes) else str(ch_attr)]
        elif 'channel_names' in f:
            ch_data = f['channel_names'][:]
            if ch_data.ndim == 1:
                channel_names = [c.decode() if isinstance(c, bytes) else str(c) for c in ch_data]
            else:
                channel_names = [str(c) for c in ch_data.flatten()]
        else:
            channel_names = [f'ch{i}' for i in range(n_channels)]

        # 确保长度匹配
        if len(channel_names) != n_channels:
            if len(channel_names) < n_channels:
                channel_names += [f'ch{i}' for i in range(len(channel_names), n_channels)]
            else:
                channel_names = channel_names[:n_channels]

        # ---------- 时间轴解析 ----------
        if 'times' in f:
            times_raw = f['times'][:]
            if times_raw.dtype.kind in 'SU':
                # ISO 格式字符串
                time_list = []
                for t in times_raw:
                    if isinstance(t, bytes):
                        t = t.decode()
                    try:
                        dt = datetime.fromisoformat(t.replace(' ', 'T'))
                    except:
                        dt = None
                    time_list.append(dt)
                # 检查是否全部成功
                if all(t is not None for t in time_list):
                    time = np.array(time_list)
                    print("时间轴：ISO 字符串已解析为 datetime")
                else:
                    # 有失败，保留字符串（但后续无法进行 datetime 运算）
                    time = np.array(times_raw, dtype='U')
                    print("警告：部分时间解析失败，保留原始字符串")
            else:
                # 数值型（相对秒数）
                time = times_raw
                print("时间轴：数值型（相对时间）")
        else:
            # 无 times，生成相对索引
            time = np.arange(total_points)
            print("警告：未找到 times 数据集，使用索引作为时间轴")

        # 如果 time 是列表，转为 numpy 数组以便索引
        if not isinstance(time, np.ndarray):
            time = np.array(time)

        # ---------- 确定绘制的通道 ----------
        if channels is None:
            plot_indices = list(range(n_channels))
        else:
            plot_indices = [i for i in channels if 0 <= i < n_channels]
            if not plot_indices:
                print("未选择有效通道，将绘制全部。")
                plot_indices = list(range(n_channels))

        # ---------- 降采样（数据点过多时） ----------
        max_points = 20000
        step = max(1, total_points // max_points)
        if step > 1:
            print(f"降采样步长: {step} (原始点数 {total_points})")
        indices = np.arange(0, total_points, step)

        # 对数据和时轴同时降采样
        data = tsdata[indices, :] if n_channels > 1 else tsdata[indices].reshape(-1, 1)
        time = time[indices]   # 关键：同步索引

        # 只保留选定通道
        data = data[:, plot_indices]
        selected_names = [channel_names[i] for i in plot_indices]

        # ---------- 时间范围截取（仅当 time 为 datetime 且用户提供了范围） ----------
        if start_time is not None and end_time is not None:
            if isinstance(time[0], datetime):
                start_idx = np.searchsorted(time, start_time)
                end_idx = np.searchsorted(time, end_time, side='right')
                if start_idx < end_idx:
                    time = time[start_idx:end_idx]
                    data = data[start_idx:end_idx, :]
                    print(f"截取时间范围: {start_time} ~ {end_time}，有效点数 {len(time)}")
                else:
                    print("警告：时间范围无效，使用全部数据")
            else:
                print("警告：时间轴不是 datetime 类型，忽略时间范围截取")

        # 如果 data 为空，报错
        if data.shape[0] == 0:
            raise ValueError("截取后没有数据点，请检查时间范围或文件内容。")

        print(f"最终绘图数据点数: {data.shape[0]}")

    # ---------- 绘图 ----------
    n_plots = data.shape[1]
    if n_plots == 0:
        raise ValueError("没有可绘制的通道")

    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True, dpi=dpi)
    if n_plots == 1:
        axes = [axes]

    for i, (idx, name) in enumerate(zip(plot_indices, selected_names)):
        ax = axes[i]
        ax.plot(time, data[:, i], color='black', linewidth=0.8, label=name)
        ax.set_ylabel(name, fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.margins(0.05)

    # 设置 x 轴格式
    if len(time) > 0 and isinstance(time[0], datetime):
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        axes[-1].set_xlabel('Time', fontsize=12)
    else:
        axes[-1].set_xlabel('Time (samples or seconds)', fontsize=12)

    fig.suptitle(f'Time Series: {Path(input_path).stem}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # ---------- 保存 ----------
    if output_path is None:
        base = Path(input_path).stem
        output_dir = Path(input_path).parent
        output_path = output_dir / f"{base}_timeseries_plot.png"
    else:
        output_path = Path(output_path)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"图片已保存至: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ==================== 主程序交互 ====================
if __name__ == '__main__':
    print("===== 单文件时间序列绘图工具 (纯HDF5) =====")

    # 1. 选择输入文件
    input_file = select_file("选择 HDF5 数据文件", [("HDF5 files", "*.h5")])
    if not input_file:
        print("未选择输入文件，退出。")
        exit()

    # 2. 读取通道名以便选择
    with h5py.File(input_file, 'r') as f:
        if 'channel_names' in f.attrs:
            ch_attr = f.attrs['channel_names']
            if isinstance(ch_attr, (list, tuple, np.ndarray)):
                ch_names = [c.decode() if isinstance(c, bytes) else str(c) for c in ch_attr]
            else:
                ch_names = [ch_attr.decode() if isinstance(ch_attr, bytes) else str(ch_attr)]
        elif 'channel_names' in f:
            ch_data = f['channel_names'][:]
            ch_names = [c.decode() if isinstance(c, bytes) else str(c) for c in ch_data.flatten()]
        else:
            nch = f['TSData'].shape[1] if 'TSData' in f else f['data'].shape[1]
            ch_names = [f'ch{i}' for i in range(nch)]

    # 3. 选择通道
    selected_indices = ask_channel_selection(ch_names)
    if selected_indices is None or len(selected_indices) == 0:
        selected_indices = list(range(len(ch_names)))

    # 4. 选择时间范围（可选）
    use_time_range = ask_yes_no("是否截取特定时间范围？（仅当时间轴为 datetime 时有效）")
    start_dt = None
    end_dt = None
    if use_time_range:
        start_str = ask_string("请输入起始时间（格式：YYYY-MM-DD HH:MM:SS）")
        if start_str:
            try:
                start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
            except:
                print("起始时间格式错误，将忽略。")
        end_str = ask_string("请输入结束时间（格式：YYYY-MM-DD HH:MM:SS）")
        if end_str:
            try:
                end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
            except:
                print("结束时间格式错误，将忽略。")

    # 5. 选择输出目录
    output_dir = select_directory("选择输出目录")
    if not output_dir:
        output_dir = os.path.dirname(input_file)
        print(f"使用输入文件所在目录作为输出目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(output_dir, f"{base_name}_timeseries_plot.png")

    # 6. 绘图
    plot_single_file(
        input_path=input_file,
        output_path=output_path,
        figsize=(12, max(3, len(selected_indices) * 2)),
        start_time=start_dt,
        end_time=end_dt,
        channels=selected_indices,
        dpi=300,
        show=True
    )