"""
实际数据小波变换计算脚本（分窗CWT版，模仿MATLAB逻辑）
功能：
1. 通过对话框选择 HDF5 格式的时间序列数据
2. 设置采样率、分析频率范围、分窗参数等
3. 对每个通道进行分窗连续小波变换（CWT），自动时间降采样
4. 保存 CWT 结果为 HDF5 文件（含时间轴和影响锥）
5. 选择两个通道计算小波相干性和相位角，保存结果
"""

import h5py
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from pythonProject.src.wrong_code.wavelet_tranform_service import WaveletTransformTool
from scipy import signal


def explore_hdf5_and_select(file_path):
    """打开 HDF5 文件，让用户选择数据集路径、通道名路径、是否使用 times"""
    root = tk.Tk()
    root.withdraw()
    with h5py.File(file_path, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group: {name}")
        print(f"\nHDF5 文件结构: {file_path}")
        f.visititems(print_structure)
        common_datasets = ['TSData', 'data', 'signal', 'X']
        default_dataset = None
        for ds in common_datasets:
            if ds in f:
                default_dataset = ds
                break
    dataset_path = simpledialog.askstring("选择数据集", f"数据矩阵路径（默认: {default_dataset}）:", initialvalue=default_dataset)
    if dataset_path is None:
        raise ValueError("未指定数据集路径")
    use_ch_names = messagebox.askyesno("通道名", "是否有通道名称数据集？")
    channel_names_path = None
    if use_ch_names:
        channel_names_path = simpledialog.askstring("通道名数据集", "通道名称数据集路径（例如 'channel_names'）:")
        if channel_names_path == "":
            channel_names_path = None
    use_times = False
    with h5py.File(file_path, 'r') as f:
        if 'times' in f:
            use_times = messagebox.askyesno("时间轴", "是否使用 'times' 数据集作为时间轴？\n（选“否”则自动生成等间隔时间轴）")
        else:
            print("未找到 'times' 数据集，将自动生成时间轴。")
    root.destroy()
    return dataset_path, channel_names_path, use_times


def get_user_parameters():
    """获取用户输入：采样率、频率范围、分窗参数等"""
    root = tk.Tk()
    root.withdraw()
    sampling_rate = simpledialog.askfloat("输入参数", "采样率 (Hz):", initialvalue=1000.0)
    if sampling_rate is None:
        exit()
    fmin = simpledialog.askfloat("输入参数", "最小分析频率 (Hz):", initialvalue=0.01)
    if fmin is None:
        fmin = 0.1
    fmax = simpledialog.askfloat("输入参数", "最大分析频率 (Hz):", initialvalue=sampling_rate/2)
    if fmax is None:
        fmax = sampling_rate/2
    n_freqs = simpledialog.askinteger("输入参数", "频率点数:", initialvalue=128)
    if n_freqs is None:
        n_freqs = 32
    calc_coherence = messagebox.askyesno("选择", "是否计算两个通道的小波相干性和相位角？")
    # 分窗参数
    window_len = simpledialog.askinteger("分窗参数", "窗口长度（样本数，建议2的幂如8192）:", initialvalue=65536)
    if window_len is None:
        window_len = 8192
    time_decimate = simpledialog.askinteger("分窗参数", "CWT结果时间降采样因子（每多少点取一个）:", initialvalue=1)
    if time_decimate is None:
        time_decimate = 1
    root.destroy()
    return sampling_rate, fmin, fmax, n_freqs, calc_coherence, window_len, time_decimate


def select_hdf5_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="选择 HDF5 数据文件", filetypes=[("HDF5 files", "*.h5")])
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


def select_channels(tool, prompt="选择通道"):
    root = tk.Tk()
    root.withdraw()
    from tkinter import Listbox, MULTIPLE, END, Button, Toplevel
    def on_select():
        selected = [listbox.get(i) for i in listbox.curselection()]
        result.extend(selected)
        top.destroy()
    top = Toplevel(root)
    top.title(prompt)
    top.geometry("300x400")
    listbox = Listbox(top, selectmode=MULTIPLE)
    listbox.pack(fill="both", expand=True)
    for name in tool.channel_names:
        listbox.insert(END, name)
    btn = Button(top, text="确认", command=on_select)
    btn.pack()
    result = []
    root.wait_window(top)
    root.destroy()
    return result


def load_data_with_auto_transpose(file_path, dataset_path, channel_names_path, sampling_rate, use_times=True):
    """加载数据，自动转置为 (channels, samples)，返回 (data, channel_names, time_axis)"""
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_path][:]
        if channel_names_path and channel_names_path in f:
            raw = f[channel_names_path][:]
            channel_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name) for name in raw]
        else:
            channel_names = None
        time_axis_raw = None
        if use_times and 'times' in f:
            time_axis_raw = f['times'][:]
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim != 2:
        raise ValueError(f"数据维度错误: {data.ndim}")
    # 转置：如果样本数 > 通道数，则转置为 (channels, samples)
    if data.shape[0] > data.shape[1] and data.shape[1] > 1:
        print(f"自动转置: (samples={data.shape[0]}, channels={data.shape[1]}) -> (channels, samples)")
        data = data.T
    n_channels, n_samples = data.shape
    # 生成时间轴
    if not use_times or time_axis_raw is None:
        time_axis = np.arange(n_samples) / sampling_rate
    else:
        # 字符串时间解析（简化版，支持常见格式）
        if time_axis_raw.dtype.kind in ('S', 'U'):
            from datetime import datetime
            time_axis = np.zeros(len(time_axis_raw), dtype=float)
            first_dt = None
            ok = True
            for i, ts in enumerate(time_axis_raw):
                if isinstance(ts, bytes):
                    ts = ts.decode('utf-8')
                dt = None
                for fmt in ['%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S']:
                    try:
                        dt = datetime.strptime(ts, fmt)
                        break
                    except:
                        continue
                if dt is None:
                    print(f"警告：无法解析时间 '{ts}'，使用自动时间轴")
                    ok = False
                    break
                if first_dt is None:
                    first_dt = dt
                    time_axis[i] = 0.0
                else:
                    time_axis[i] = (dt - first_dt).total_seconds()
            if not ok:
                time_axis = np.arange(n_samples) / sampling_rate
        else:
            time_axis = time_axis_raw.astype(float)
            if len(time_axis) != n_samples:
                time_axis = np.arange(n_samples) / sampling_rate
    if channel_names is None:
        channel_names = [f'ch{i}' for i in range(n_channels)]
    elif len(channel_names) != n_channels:
        channel_names = [f'ch{i}' for i in range(n_channels)]
    return data, channel_names, time_axis


def main():
    print("=" * 60)
    print("小波变换计算工具（分窗CWT版）")
    print("=" * 60)

    # 获取用户参数
    sr, fmin, fmax, n_freqs, need_coherence, win_len, time_dec = get_user_parameters()
    freqs = np.linspace(fmin, fmax, n_freqs)

    # 选择输入文件
    input_file = select_hdf5_file()
    print(f"输入文件: {input_file}")

    # 选择数据集路径等
    dataset_path, ch_names_path, use_times = explore_hdf5_and_select(input_file)
    print(f"数据矩阵路径: {dataset_path}")
    print(f"通道名路径: {ch_names_path if ch_names_path else '无'}")
    print(f"使用 times 数据集: {use_times}")

    # 选择输出目录
    output_dir = select_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 加载数据
    data, channel_names, time_axis = load_data_with_auto_transpose(
        input_file, dataset_path, ch_names_path, sr, use_times=use_times
    )
    print(f"加载成功，数据形状: {data.shape} (通道数 x 样本数)")
    print(f"通道名: {channel_names}")
    print(f"时间范围: [{time_axis[0]:.2f}, {time_axis[-1]:.2f}] 秒")

    # 可选：对原始数据额外降采样
    downsample_data = messagebox.askyesno("数据降采样", "是否对原始数据额外降采样？\n（分窗CWT内部已做时间降采样，一般不需要）")
    if downsample_data:
        factor = simpledialog.askinteger("降采样", "降采样因子（例如2,5,10）:", initialvalue=5)
        if factor and factor > 1:
            print(f"正在对原始数据降采样，因子: {factor}")
            data = signal.decimate(data, factor, axis=1, ftype='fir')
            time_axis = time_axis[::factor]
            sr = sr / factor
            print(f"降采样后数据形状: {data.shape}, 新采样率: {sr:.2f} Hz")

    # 初始化小波工具
    tool = WaveletTransformTool(sampling_rate=sr)
    tool.data = data
    tool.channel_names = channel_names

    # 执行分窗CWT
    print("正在进行分窗连续小波变换...")
    print(f"窗口长度: {win_len} 点, 重叠比例: 2/3, 时间降采样因子: {time_dec}")
    tms, freq_out, coi, cfs_all = tool.cwt_windowed(
        window_len=win_len,
        overlap_frac=2/3,
        time_decimate=time_dec,
        wavelet='morl',
        freqs=freqs
    )
    print("CWT 完成。")
    print(f"输出时间点数: {len(tms)}, 频率点数: {len(freq_out)}")

    # 保存 CWT 结果
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    cwt_output = os.path.join(output_dir, f"{base_name}_cwt_results.h5")
    tool.save_cwt_results(cwt_output)
    print(f"CWT 结果已保存至: {cwt_output}")

    # 相干性计算（如果需要）
    coherence_output = None
    if need_coherence and len(tool.channel_names) >= 2:
        print("请选择要分析的两个通道：")
        selected = select_channels(tool, "选择两个通道（Ctrl+单击多选）")
        if len(selected) >= 2:
            ch1, ch2 = selected[0], selected[1]
            print(f"计算相干性: {ch1} vs {ch2}")
            coherence, phase, freqs_coh, scales = tool.wavelet_coherence(
                channel1=ch1, channel2=ch2,
                wavelet='morl', freqs=freqs,
                time_smoothing_window=0.2, scale_smoothing_window=0.2
            )
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            coherence_output = os.path.join(output_dir, f"{base_name}_coherence_results.h5")
            tool.save_coherence_phase(
                output_path=coherence_output,
                channel1=ch1, channel2=ch2,
                coherence=coherence, phase=phase,
                frequencies=freqs_coh, scales=scales,
                wavelet='morl'
            )
            print(f"相干性结果已保存至: {coherence_output}")
        else:
            print("未选择两个通道，跳过相干性计算。")
    elif need_coherence:
        print("通道数不足2，无法计算相干性。")

    print("所有计算完成！")
    print(f"输出文件列表:")
    print(f"  - CWT 结果: {cwt_output}")
    if coherence_output:
        print(f"  - 相干性结果: {coherence_output}")


if __name__ == "__main__":
    main()