import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pythonProject.configs.configmanager import ConfigManager   # 请确保路径正确

# ------------------------------
# 辅助 GUI 函数（沿用原脚本）
# ------------------------------
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

def ask_float(prompt, initial=1000.0):
    root = tk.Tk()
    root.withdraw()
    val = simpledialog.askfloat("输入参数", prompt, initialvalue=initial)
    root.destroy()
    return val

def ask_string(prompt, initial=""):
    root = tk.Tk()
    root.withdraw()
    val = simpledialog.askstring("输入参数", prompt, initialvalue=initial)
    root.destroy()
    return val

# ------------------------------
# 核心绘图函数（方案二：时间窗口截取）
# ------------------------------
def plot_cropped_time_series(
    input_file,              # HDF5 文件路径
    output_dir=None,         # 输出目录，默认为输入文件所在目录
    start_time=None,         # 截取起始时间（datetime 对象或字符串 "YYYY-MM-DD HH:MM:SS"）
    end_time=None,           # 截取结束时间（datetime 对象或字符串）
    sample_rate=None,        # 采样率（Hz），若为 None 则尝试从配置文件或手动输入
    cfg_path=None,           # 配置文件路径（可选），用于获取采样率和全局起始时间
    use_gui=True,            # 是否使用 GUI 交互（若为 False，则必须提供所有必需参数）
    max_display_points=10000 # 绘图最大点数（自动降采样）
):
    """
    根据指定的时间窗口截取 HDF5 数据并绘制时间序列图。

    参数：
        input_file : str
            预处理后的 HDF5 文件路径。
        output_dir : str, optional
            输出图像保存目录，默认与 input_file 同目录。
        start_time : datetime or str, optional
            截取窗口的起始时间。若为字符串，格式为 "YYYY-MM-DD HH:MM:SS"。
            若不提供，且 use_gui=True，则会弹出对话框询问。
        end_time : datetime or str, optional
            截取窗口的结束时间。格式同 start_time。
        sample_rate : float, optional
            采样率（Hz）。若不提供，将尝试从配置文件或手动输入获取。
        cfg_path : str, optional
            配置文件路径（.cfg 或 .csv），用于读取采样率和全局起始时间。
            若不提供且 use_gui=True，会询问是否选择配置文件。
        use_gui : bool, default True
            是否启用 GUI 交互（文件选择、参数输入对话框）。若为 False，则必须提供
            start_time, end_time, sample_rate（或通过 cfg_path 获取）。
        max_display_points : int, default 10000
            绘图前降采样的目标点数，以保持图像清晰并避免过密。

    返回：
        output_path : str
            保存的图像文件路径。
    """
    # ---------- 1. 输入文件检查 ----------
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # ---------- 2. 输出目录 ----------
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_path = os.path.join(output_dir, f"{base_name}_cropped_timeseries.png")

    # ---------- 3. 获取采样率与全局起始时间 ----------
    global_start_time = None   # 整个文件的起始时间（用于将用户输入的时间转换为样本索引）

    # 如果提供了配置文件，优先读取
    if cfg_path is not None and os.path.isfile(cfg_path):
        ext = os.path.splitext(cfg_path)[1].lower()
        try:
            if ext == '.cfg':
                cfg_mgr = ConfigManager.from_aether_cfg(cfg_path)
            elif ext == '.csv':
                cfg_mgr = ConfigManager.from_csv(cfg_path)
            else:
                raise ValueError(f"不支持的配置文件格式: {ext}")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            cfg_mgr = None

        if cfg_mgr is not None:
            if sample_rate is None:
                sample_rate = cfg_mgr.get("采样率 (Hz)")
            # 获取全局起始时间（字符串）
            g_start_str = cfg_mgr.get("开始时间") or cfg_mgr.get("起始时间") or cfg_mgr.get("Start time")
            if g_start_str:
                for fmt in ["%Y %m %d %H %M %S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y%m%d %H%M%S"]:
                    try:
                        global_start_time = datetime.strptime(g_start_str, fmt)
                        break
                    except:
                        continue
                if global_start_time is None:
                    print(f"无法解析配置文件中的起始时间: {g_start_str}，将忽略。")

    # 如果 GUI 模式开启且某些参数缺失，则交互式获取
    if use_gui:
        # 如果没有采样率，则询问
        if sample_rate is None:
            sample_rate = ask_float("请输入采样率 (Hz):", initial=1000.0)
            if sample_rate is None:
                raise ValueError("未提供采样率，无法继续。")

        # 如果没有全局起始时间，询问是否使用绝对时间轴
        if global_start_time is None:
            use_abs = ask_yes_no("文件是否使用绝对时间轴？如需截取时间窗口，请提供全局起始时间。")
            if use_abs:
                g_start_str = ask_string("请输入整个文件的起始时间（格式：YYYY-MM-DD HH:MM:SS）",
                                        "2022-11-06 13:21:00")
                try:
                    global_start_time = datetime.strptime(g_start_str, "%Y-%m-%d %H:%M:%S")
                except:
                    print("全局起始时间格式错误，将无法使用绝对时间截取。")
                    global_start_time = None

        # 获取截取窗口的起始和结束时间
        if start_time is None:
            start_str = ask_string("请输入截取起始时间（格式：YYYY-MM-DD HH:MM:SS）",
                                  "2022-11-06 13:21:00")
            try:
                start_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
            except:
                print("起始时间格式错误，将使用全范围。")
                start_time = None
        else:
            if isinstance(start_time, str):
                start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

        if end_time is None and start_time is not None:
            end_str = ask_string("请输入截取结束时间（格式：YYYY-MM-DD HH:MM:SS）",
                                "2022-11-06 13:25:00")
            try:
                end_time = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
            except:
                print("结束时间格式错误，将使用全范围。")
                end_time = None
        elif isinstance(end_time, str):
            end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

    else:  # 非 GUI 模式，必须提供必要参数
        if sample_rate is None:
            raise ValueError("非 GUI 模式下必须提供 sample_rate 或 cfg_path。")
        if global_start_time is None:
            print("警告：未提供全局起始时间，将尝试从 HDF5 的 times 数据集读取绝对时间。")
        if start_time is None or end_time is None:
            raise ValueError("非 GUI 模式下必须提供 start_time 和 end_time。")

    # ---------- 4. 打开 HDF5 文件并截取数据 ----------
    with h5py.File(input_file, 'r') as f:
        # 查找数据集
        if 'TSData' in f:
            tsdata = f['TSData']
        elif 'data' in f:
            tsdata = f['data']
        elif 'E1' in f:
            tsdata = f['E1']
        elif 'E2' in f:
            tsdata = f['E2']
        else:
            raise KeyError("未找到 TSData、data、E1 或 E2 数据集")

        total_points = tsdata.shape[0]
        n_channels = tsdata.shape[1] if len(tsdata.shape) == 2 else 1
        print(f"总数据点: {total_points}, 通道数: {n_channels}")

        # ---------- 获取通道名称 ----------
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
        if len(channel_names) != n_channels:
            if len(channel_names) < n_channels:
                channel_names += [f'ch{i}' for i in range(len(channel_names), n_channels)]
            else:
                channel_names = channel_names[:n_channels]

        # ---------- 确定截取范围（样本索引） ----------
        start_idx = 0
        end_idx = total_points

        # 优先使用绝对时间截取（如果有全局起始时间和用户指定的起止时间）
        if global_start_time is not None and start_time is not None and end_time is not None:
            # 将用户时间转为相对于全局起始时间的秒数
            start_sec = (start_time - global_start_time).total_seconds()
            end_sec = (end_time - global_start_time).total_seconds()
            start_idx = int(start_sec * sample_rate)
            end_idx = int(end_sec * sample_rate)
            # 边界检查
            start_idx = max(0, min(start_idx, total_points - 1))
            end_idx = max(start_idx + 1, min(end_idx, total_points))
            print(f"时间窗口转换为样本索引: [{start_idx}, {end_idx})")
        else:
            # 回退：尝试从 times 数据集中查找
            if 'times' in f:
                # 读取完整时间轴（可能很大，但可以切片）
                times_full = f['times'][:]
                # 假设 times_full 是字符串或数字，尝试解析
                # 这里简化处理，如果 times 是字符串时间，需全部读入比较（可能很慢）
                # 实际使用中建议提前知道时间轴类型
                print("无法使用绝对时间截取，将显示全范围。")
            else:
                print("没有足够信息截取特定时间窗口，将显示全范围。")

        # 读取截取后的数据
        if n_channels > 1:
            data_full = tsdata[start_idx:end_idx, :]
        else:
            data_full = tsdata[start_idx:end_idx].reshape(-1, 1)
        cropped_points = data_full.shape[0]
        print(f"截取后数据点: {cropped_points}")

        # ---------- 降采样 ----------
        step = max(1, cropped_points // max_display_points)
        indices = np.arange(0, cropped_points, step)
        data = data_full[indices, :]

        # ---------- 生成时间轴 ----------
        if global_start_time is not None:
            # 绝对时间轴（基于截取后的起始）
            time_seconds = (start_idx + indices) / sample_rate
            time = np.array([global_start_time + timedelta(seconds=ts) for ts in time_seconds])
        elif 'times' in f:
            # 如果存在 times 数据集，读取并切片
            times_full = f['times'][start_idx:end_idx]
            time = times_full[indices]   # 假设 times_full 是数值或可索引
            # 若是字符串，需解析，这里不展开
        else:
            # 相对秒数
            time = (start_idx + indices) / sample_rate

    # ---------- 5. 绘图 ----------
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, max(3, n_channels * 2)), sharex=True)
    fig.suptitle(f'Cropped Time Series: {base_name}', fontsize=10)
    if n_channels == 1:
        axes = [axes]

    for i in range(n_channels):
        axes[i].plot(time, data[:, i], linewidth=0.5, color='black')
        axes[i].set_ylabel(channel_names[i], fontsize=8)
        axes[i].grid(True, linestyle='--', alpha=0.6)

    # 设置 x 轴格式
    if time.size > 0 and isinstance(time[0], (datetime, np.datetime64)):
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        axes[-1].set_xlabel('Time', fontsize=12)
    else:
        axes[-1].set_xlabel('Time (seconds)', fontsize=12)

    plt.tight_layout()

    # ---------- 6. 保存图像 ----------
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图像已保存至：{output_path}")
    plt.show()
    return output_path

# ------------------------------
# 使用示例（交互式）
# ------------------------------
if __name__ == "__main__":
    # 交互式运行：选择文件、设置参数
    input_h5 = select_file("选择预处理后的 HDF5 数据文件", [("HDF5 files", "*.h5")])
    if not input_h5:
        print("未选择文件，退出。")
        exit()
    # 调用函数（use_gui=True 会弹出对话框）
    plot_cropped_time_series(input_h5, use_gui=True)