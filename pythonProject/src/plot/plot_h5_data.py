import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from datetime import datetime, timedelta
from pythonProject.configs.configmanager import ConfigManager


# ------------------------------
# 辅助函数：选择文件、目录
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
# 1. 选择输入 HDF5 文件
# ------------------------------
input_file = select_file("选择预处理后的 HDF5 数据文件", [("HDF5 files", "*.h5")])
if not input_file:
    print("未选择输入文件，退出。")
    exit()

# 自动生成基础名称（用于输出文件名）
base_name = os.path.splitext(os.path.basename(input_file))[0]  # 例如 "Ground_preprocessed"

# ------------------------------
# 2. 选择配置文件（可选，支持 .cfg 和 .csv）
# ------------------------------
use_cfg = ask_yes_no("是否使用配置文件（.cfg 或 .csv）获取采样率和开始时间？")
cfg_path = None
sample_rate = None
start_time_str = None

if use_cfg:
    cfg_path = select_file(
        "选择配置文件",
        filetypes=[
            ("Config files", "*.cfg;*.csv"),
            ("CFG files", "*.cfg"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
    )
    if cfg_path:
        ext = os.path.splitext(cfg_path)[1].lower()
        try:
            if ext == '.cfg':
                cfg_mgr = ConfigManager.from_aether_cfg(cfg_path)
            elif ext == '.csv':
                cfg_mgr = ConfigManager.from_csv(cfg_path)
            else:
                raise ValueError(f"不支持的配置文件格式: {ext}")
        except Exception as e:
            print(f"加载配置文件失败: {e}，将手动输入参数。")
            cfg_mgr = None

        if cfg_mgr is not None:
            sample_rate = cfg_mgr.get("采样率 (Hz)")
            start_time_str = cfg_mgr.get("开始时间") or cfg_mgr.get("起始时间") or cfg_mgr.get("Start time")
            if sample_rate is None:
                print("配置文件中未找到采样率，将手动输入。")
            else:
                print(f"从配置文件读取采样率: {sample_rate} Hz")
    else:
        print("未选择配置文件，将手动输入参数。")

# 如果未能从配置文件获取，则手动输入
if sample_rate is None:
    sample_rate = ask_float("请输入采样率 (Hz):", initial=1000.0)
    if sample_rate is None:
        exit()

# 处理开始时间
start_time = None
if start_time_str:
    try:
        # 尝试多种格式
        for fmt in ["%Y %m %d %H %M %S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y%m%d %H%M%S"]:
            try:
                start_time = datetime.strptime(start_time_str, fmt)
                break
            except:
                continue
        if start_time is None:
            raise ValueError
        print(f"从配置文件读取开始时间: {start_time}")
    except:
        print(f"无法解析开始时间字符串: {start_time_str}，将使用相对时间轴。")
        start_time = None
else:
    use_absolute = ask_yes_no("是否使用绝对时间轴？如要使用，请输入开始时间。")
    if use_absolute:
        start_time_str = ask_string("请输入开始时间（格式：YYYY-MM-DD HH:MM:SS）", "2022-11-06 13:21:00")
        try:
            start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
        except:
            print("时间格式错误，将使用相对时间轴。")
            start_time = None

# ------------------------------
# 3. 选择输出目录
# ------------------------------
output_dir = select_directory("选择输出目录")
if not output_dir:
    output_dir = os.path.dirname(input_file)
    print(f"使用输入文件所在目录作为输出目录: {output_dir}")
os.makedirs(output_dir, exist_ok=True)

# 自动生成输出文件名
output_filename = f"{base_name}_timeseries_plot.png"
output_path = os.path.join(output_dir, output_filename)

# ------------------------------
# 4. 加载 HDF5 数据（自动降采样），并获取通道名
# ------------------------------
with h5py.File(input_file, 'r') as f:
    # 查找数据集（优先 TSData，否则 data）
    if 'TSData' in f:
        tsdata = f['TSData']
    elif 'data' in f:
        tsdata = f['data']
    elif 'E1' in f:
        tsdata = f['E1']
    elif 'E2' in f:
        tsdata = f['E2']
    else:
        raise KeyError("未找到 TSData,time 或 data 数据集")

    total_points = tsdata.shape[0]
    n_channels = tsdata.shape[1] if len(tsdata.shape) == 2 else 1
    print(f"总数据点: {total_points}, 通道数: {n_channels}")

    # ---------- 通道名称获取（直接从 HDF5）----------
    channel_names = None
    # 优先从根属性读取（保存时常用）
    if 'channel_names' in f.attrs:
        ch_attr = f.attrs['channel_names']
        # 属性可能是字符串列表或单个字符串
        if isinstance(ch_attr, (list, tuple, np.ndarray)):
            channel_names = [c.decode() if isinstance(c, bytes) else str(c) for c in ch_attr]
        else:
            channel_names = [ch_attr.decode() if isinstance(ch_attr, bytes) else str(ch_attr)]
        print("从 HDF5 属性读取通道名称。")
    # 其次尝试从数据集读取
    elif 'channel_names' in f:
        ch_data = f['channel_names'][:]
        if ch_data.ndim == 1:
            channel_names = [c.decode() if isinstance(c, bytes) else str(c) for c in ch_data]
        else:
            # 若为二维，取第一列或扁平化处理
            channel_names = [str(c) for c in ch_data.flatten()]
        print("从 HDF5 数据集读取通道名称。")
    else:
        channel_names = [f'ch{i}' for i in range(n_channels)]
        print("未找到通道名称，使用默认命名 ch0, ch1, ...")

    # 确保长度匹配
    if len(channel_names) != n_channels:
        print(f"警告：通道名称数量 ({len(channel_names)}) 与数据通道数 ({n_channels}) 不匹配，将截断或补齐。")
        if len(channel_names) < n_channels:
            channel_names += [f'ch{i}' for i in range(len(channel_names), n_channels)]
        else:
            channel_names = channel_names[:n_channels]

    # ---------- 降采样 ----------
    max_points = 10000
    step = max(1, total_points // max_points)
    print(f"降采样步长: {step}")
    indices = np.arange(0, total_points, step)
    data = tsdata[indices, :] if n_channels > 1 else tsdata[indices].reshape(-1, 1)

    # ---------- 时间轴生成 ----------
    if start_time is not None:
        time_seconds = indices / sample_rate
        time = np.array([start_time + timedelta(seconds=ts) for ts in time_seconds])
        print("时间轴已通过起始时间和采样率生成")
    elif 'times' in f:
        times_data = f['times'][indices]
        if times_data.dtype.kind in 'SU':
            # 字符串时间
            time = []
            for t in times_data:
                if isinstance(t, bytes):
                    t = t.decode()
                # 尝试解析 ISO 格式
                try:
                    dt = datetime.fromisoformat(t.replace(' ', 'T'))
                except:
                    dt = None
                time.append(dt)
            time = np.array(time)
        else:
            time = times_data
        print("时间轴已从 HDF5 的 times 数据集读取")
    else:
        time = indices / sample_rate
        print("警告：无法获取绝对时间，使用相对时间（秒）作为横坐标")

# ------------------------------
# 5. 绘图
# ------------------------------
fig, axes = plt.subplots(n_channels, 1, figsize=(12, max(3, n_channels * 2)), sharex=True)
fig.suptitle(f'Time Series: {base_name}', fontsize=10)

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

# ------------------------------
# 6. 保存图像
# ------------------------------
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图像已保存至：{output_path}")

plt.show()