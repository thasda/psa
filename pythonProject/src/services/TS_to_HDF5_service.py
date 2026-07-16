import numpy as np
import h5py
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from tkinter.simpledialog import askfloat


def read_ts_file(filepath, chs=8, sample_bytes=4, dtype='<i4'):
    """
    读取单个 TS 二进制文件，返回电压数据矩阵 (chs, n_samples)
    不生成时间轴，因为需要全局拼接
    """
    with open(filepath, 'rb') as fid:
        data_bytes = fid.read()

    bytelen = len(data_bytes)
    total_samples = bytelen // (sample_bytes * chs)

    raw = np.frombuffer(data_bytes, dtype=dtype, count=total_samples * chs)
    raw = raw.reshape((chs, total_samples), order='F')

    # 转换为 mV
    voltage = raw.astype(np.float64) / (2 ** 31) * 5000
    return voltage


def concatenate_ts_files(file_list, fs, chs=8):
    """
    拼接多个 TS 文件的数据，并生成整体时间轴
    :param file_list: 按时间顺序排列的文件路径列表
    :param fs: 采样率 (Hz)
    :param chs: 通道数
    :return: total_voltage (chs, total_samples), total_time (total_samples,)
    """
    voltages = []
    total_samples = 0
    for fpath in file_list:
        v = read_ts_file(fpath, chs=chs)
        voltages.append(v)
        total_samples += v.shape[1]

    # 预分配连续数组
    total_voltage = np.hstack(voltages)  # 横向拼接，保持通道维度一致
    # 生成时间轴 (秒)
    total_time = np.arange(total_samples) / fs
    return total_voltage, total_time


def save_to_hdf5(data, times, output_path, sample_rate, channel_names=None, source_files=None):
    """
    保存拼接后的数据到 HDF5
    :param data: (n_samples, n_channels) 或 (n_channels, n_samples) 这里统一转置为 (n_samples, n_channels)
    :param times: (n_samples,) 时间轴（秒）
    :param output_path: 保存路径
    :param sample_rate: 采样率 (Hz)
    :param channel_names: 通道名称列表
    :param source_files: 源文件路径列表（可选）
    """
    # 确保 data 形状为 (n_samples, n_channels)
    if data.ndim == 2:
        if data.shape[0] < data.shape[1] and data.shape[0] != len(times):
            # 假设输入是 (n_channels, n_samples) 需要转置
            data = data.T
    else:
        raise ValueError("Data must be 2D")

    n_channels = data.shape[1]
    if channel_names is None:
        channel_names = [f'ch{i}' for i in range(n_channels)]

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('TSData', data=data, compression='gzip')
        f.create_dataset('times', data=times, compression='gzip')

        f.attrs['sample_rate'] = sample_rate
        f.attrs['channel_names'] = [name.encode('utf-8') if isinstance(name, str) else name for name in channel_names]
        f.attrs['units'] = 'mV'
        f.attrs['description'] = 'Concatenated voltage data from multiple TS binary files'
        if source_files:
            f.attrs['source_files'] = [sf.encode('utf-8') for sf in source_files]

    print(f"数据已保存至: {output_path}")
    print(f"总样本数: {len(times)}, 时间长度: {times[-1]:.2f} 秒, 采样率: {sample_rate} Hz")
    print(f"数据形状: {data.shape}, 通道数: {n_channels}")


def main():
    root = Tk()
    root.withdraw()

    # 多选文件
    file_paths = askopenfilenames(
        title="请选择需要拼接的 TS 文件 (按正确顺序，或后续自动排序)",
        filetypes=[('TS files', '*.TS*'), ('All Files', '*.*')]
    )
    if not file_paths:
        print("未选择文件，程序退出。")
        return

    # 自动按文件名排序（假设文件名包含时间信息）
    file_list = sorted(file_paths)
    print(f"将按以下顺序拼接文件:\n" + "\n".join(file_list))

    # 输入采样率（必须）
    fs = askfloat("采样率", "请输入采样率 (Hz):", initialvalue=1.0, minvalue=0.1)
    if fs is None or fs <= 0:
        print("采样率无效，程序退出。")
        return

    first_voltage = read_ts_file(file_list[0])
    chs = first_voltage.shape[0]
    print(f"原始通道数：{chs}")

    total_voltage, total_time = concatenate_ts_files(file_list, fs, chs)

    keep_indices = [0, 1, 4, 5, 6]
    total_voltage = total_voltage[keep_indices, :]
    print(f"保留通道后：{chs}个通道（索引{keep_indices}")

    # 生成通道名称（可根据实际通道分配修改）

    if chs == 5:
        channel_names = ['E1', 'E2', 'H1', 'H2', 'H3']

    else:
        channel_names = [f'ch{i+1}' for i in range(chs)]

    # 确定保存路径（以第一个文件所在目录为基础，生成拼接文件名）
    base_dir = os.path.dirname(file_list[0])
    base_name = "concatenated_" + os.path.basename(file_list[0]).split('_')[0] + ".h5"
    output_path = os.path.join(base_dir, base_name)

    save_to_hdf5(total_voltage, total_time, output_path, fs,
                 channel_names=channel_names, source_files=file_list)
    print("拼接完成！")


if __name__ == '__main__':
    main()