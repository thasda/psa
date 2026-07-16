# preprocess.py（修改后版本）

"""封装一个预处理类
1 使用ConfigManager类中提取的参数"校准文件"，即3个cmt文件，并调用lemi417中的LEMi417Calibrator类中的函数来去除仪器响应
2 使用ConfigManager类中提取的参数"电长度 (米)"来进行电磁场校正
3 对HDF5数据（其包含TSData和times）进行线性去趋势和归一化处理
4 快速带阻滤波滤除工频噪声harmonics = [
            [49.0, 51.0],   # 50Hz
            [99.0, 101.0],   # 100Hz
            [149.0, 151.0],  # 150Hz
            [199.0, 201.0],  # 200Hz
            [249.0, 251.0],  # 250Hz
            [299.0, 301.0],  # 300Hz
            [349.0, 351.0],  # 350Hz
            [399.0, 401.0],  # 400Hz
            [449.0, 451.0]   # 450Hz
        ]
5 保存预处理后的数据为HDF5格式
"""
"""
预处理类，对 HDF5 格式的时间序列数据进行：
- 仪器响应去除（调用 LEMi417Calibrator）
- 电磁场校正（电长度）
- 线性去趋势
- 归一化（标准化）
- 快速带阻滤波（工频谐波）
- 保存为新的 HDF5 文件
"""

import numpy as np
import h5py
import scipy.signal as signal
from datetime import datetime
from typing import Optional, List
from pythonProject.src.utils.lemi417 import LEMi417Calibrator
from pathlib import Path

class Preprocessor:
    """
    预处理类，对 HDF5 中的 TSData 和 times 进行处理。
    """

    def __init__(self, hdf5_path: str, config_manager):
        """
        参数:
            hdf5_path: 输入 HDF5 文件路径（应包含 TSData 和 times 数据集）
            config_manager: ConfigManager 实例，用于获取预处理所需参数
        """
        self.hdf5_path = hdf5_path
        self.cfg = config_manager

        # 从配置中提取必要参数
        self.sample_rate = self.cfg.get("采样率 (Hz)")
        self.channel_count = self.cfg.get("通道数量")
        self.channel_indices = self.cfg.get("通道索引")          # 原始通道编号列表
        self.channel_gains = self.cfg.get("通道增益")            # 各通道增益（可能用于后续处理）
        self.electrical_lengths = self.cfg.get("电长度 (米)")    # [Ex_len, Ey_len]
        self.calibration_files = self.cfg.get("校准文件")        # 三个磁通道的 CMT 文件列表

        # 工频谐波频率范围（单位 Hz）
        self.harmonics = [
            [49.0, 51.0],   # 50Hz
            [99.0, 101.0],  # 100Hz
            [149.0, 151.0], # 150Hz
            [199.0, 201.0], # 200Hz
            [249.0, 251.0], # 250Hz
            [299.0, 301.0], # 300Hz
            [349.0, 351.0], # 350Hz
            [399.0, 401.0], # 400Hz
            [449.0, 451.0]  # 450Hz
        ]

        # ---------- 新版多通道校准器初始化 ----------
        cal_files = self.cfg.get("校准文件")
        if cal_files and len(cal_files) >= 3:
            # 获取配置文件所在目录（假设 config_manager 有 config_path 属性）
            if hasattr(config_manager, 'config_path'):
                cfg_dir = Path(config_manager.config_path).parent
            else:
                cfg_dir = Path.cwd()

            # 转换为绝对路径
            abs_cal_files = []
            for f in cal_files:
                p = Path(f)
                if not p.is_absolute():
                    p = cfg_dir / p
                abs_cal_files.append(str(p))

            # 使用单个多通道校准器，一次性加载三个文件
            self.cal_mag = LEMi417Calibrator(
                abs_cal_files,
                channel_mapping={'HX': 0, 'HY': 1, 'HZ': 2}
            )
            # 为兼容旧有调用习惯，保留三个引用（均指向同一实例）
            self.cal_hx = self.cal_mag
            self.cal_hy = self.cal_mag
            self.cal_hz = self.cal_mag
            print("仪器响应校准器初始化完成（多通道统一校准）")
        else:
            print("警告：校准文件不足 3 个，将跳过仪器响应去除步骤")
            self.cal_mag = self.cal_hx = self.cal_hy = self.cal_hz = None

        # 后续计算中使用的中间变量
        self.n_samples = None          # 总采样点数
        self.n_channels = None         # 总通道数
        self.x = None                  # 时间轴（秒），用于去趋势
        self.mean = None               # 各通道均值
        self.std = None                # 各通道标准差
        self.slope = None              # 各通道线性趋势斜率
        self.intercept = None           # 各通道线性趋势截距
        self.filter_states = None       # 用于逐块滤波的状态缓存

    # ----------------------------------------------------------------------
    # 内部辅助方法
    # ----------------------------------------------------------------------
    def _read_metadata(self):
        """读取输入 HDF5 的元数据：总样本数、通道数、时间向量"""
        with h5py.File(self.hdf5_path, 'r') as f:
            ex = f['TSData'][:, 0]
            print(f"原始 EX 电压: 均值={ex.mean():.2e}, 标准差={ex.std():.2e}, 范围=[{ex.min():.2e}, {ex.max():.2e}]")
            self.n_samples = f['TSData'].shape[0]
            self.n_channels = f['TSData'].shape[1]
            # 时间数据：假设存储为 ISO 字符串或数值，我们将其转换为相对于起始时间的秒数
            times_raw = f['times'][:]
            # 简单处理：如果 times_raw 是字符串数组，需转换为 datetime；这里假设已为数值秒数
            if times_raw.dtype.kind in 'US':
                # 字符串情况：尝试转换为 datetime 并计算秒偏移
                # 为简化，直接使用序号作为 x（采样点索引）
                self.x = np.arange(self.n_samples) / self.sample_rate
            else:
                # 数值情况：直接作为秒数偏移
                self.x = times_raw.astype(float)

    def _compute_global_stats(self):
        """
        第一遍扫描数据，计算每个通道的：
        - 均值 mean
        - 标准差 std
        - 线性趋势的斜率 slope 和截距 intercept
        """
        self._read_metadata()
        n = self.n_samples
        nch = self.n_channels

        # 累积变量
        sum_y = np.zeros(nch)
        sum_y2 = np.zeros(nch)
        sum_xy = np.zeros(nch)
        sum_x = 0.0
        sum_x2 = 0.0

        chunk_size = 100000  # 分块大小，可根据内存调整
        with h5py.File(self.hdf5_path, 'r') as f:
            ds = f['TSData']
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                block = ds[start:end, :]                     # (block_size, nch)
                x_block = self.x[start:end]                  # (block_size,)

                # 更新累积量
                sum_y += np.sum(block, axis=0)
                sum_y2 += np.sum(block**2, axis=0)
                # 对于每个通道，计算 x*y 的和
                sum_xy += np.sum(x_block[:, np.newaxis] * block, axis=0)
                sum_x += np.sum(x_block)
                sum_x2 += np.sum(x_block**2)

        # 计算均值和标准差
        self.mean = sum_y / n
        variance = (sum_y2 / n) - self.mean**2
        variance = np.maximum(variance, 0)  # 避免数值误差导致负值
        self.std = np.sqrt(variance)

        # 计算线性趋势系数
        # 公式： slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
        denominator = n * sum_x2 - sum_x**2
        if denominator == 0:
            # 如果时间轴完全不变（不可能发生），则斜率为0
            self.slope = np.zeros(nch)
        else:
            self.slope = (n * sum_xy - sum_x * sum_y) / denominator
        self.intercept = (sum_y - self.slope * sum_x) / n

    def _compute_global_stats_from_array(self, data: np.ndarray):
        """
        基于内存中的 numpy 数组计算各通道的全局统计量。
        """
        n_samples, n_channels = data.shape
        self.n_samples = n_samples
        self.n_channels = n_channels

        # 时间轴：若尚未生成则创建（基于采样率）
        if self.x is None:
            self.x = np.arange(n_samples) / self.sample_rate

        # 累积变量
        sum_y = np.sum(data, axis=0)
        sum_y2 = np.sum(data ** 2, axis=0)
        sum_xy = np.sum(self.x[:, np.newaxis] * data, axis=0)
        sum_x = np.sum(self.x)
        sum_x2 = np.sum(self.x ** 2)

        # 均值与标准差
        self.mean = sum_y / n_samples
        variance = (sum_y2 / n_samples) - self.mean ** 2
        variance = np.maximum(variance, 0)
        self.std = np.sqrt(variance)

        # 线性趋势系数
        denominator = n_samples * sum_x2 - sum_x ** 2
        if denominator == 0:
            self.slope = np.zeros(n_channels)
        else:
            self.slope = (n_samples * sum_xy - sum_x * sum_y) / denominator
        self.intercept = (sum_y - self.slope * sum_x) / n_samples


    # ----------------------------------------------------------------------
    # 各处理步骤
    # ----------------------------------------------------------------------
    def remove_instrument_response(self, data: np.ndarray) -> np.ndarray:
        """
        对整个数据矩阵进行仪器响应去除（全序列频域校准）。
        假设通道顺序为：Ex, Ey, Hx, Hy, Hz（索引 2,3,4 为磁通道）。
        """
        if self.channel_count >= 5 and self.cal_mag is not None:
            mag_indices = [2, 3, 4]  # HX, HY, HZ
            channel_names = ['HX', 'HY', 'HZ']
            for idx, name in zip(mag_indices, channel_names):
                print(f"  正在校准通道 {name}（全序列 FFT）...")
                data[:, idx] = self.cal_mag.calibrate_time_series(
                    data[:, idx], self.sample_rate, channel_name=name
                )
        return data

    def correct_electrical_length(self, data_block: np.ndarray) -> np.ndarray:
        """
        电场校正：将原始电压（mV）除以电长度（米），并转换为 mV/km。
        此方法保留定义，但在主流程中默认被注释。
        """
        if self.channel_count >= 2 and self.electrical_lengths is not None:
            ex_len = self.electrical_lengths[0]
            ey_len = self.electrical_lengths[1]
            factor = 1000.0  # mV/m 转 mV/km 的系数
            if ex_len != 0:
                data_block[:, 0] = data_block[:, 0] * factor / ex_len
            if ey_len != 0:
                data_block[:, 1] = data_block[:, 1] * factor / ey_len
        return data_block

    def detrend(self, data_block: np.ndarray, x_block: np.ndarray, method='linear') -> np.ndarray:
        if method == 'linear':
            trend = self.slope * x_block[:, np.newaxis] + self.intercept
            return data_block - trend
        elif method == 'mean':
            return data_block - self.mean
        else:
            return data_block

    def normalize(self, data_block: np.ndarray) -> np.ndarray:
        """
        归一化：减去全局均值后除以全局标准差（标准化）。
        """
        # 避免除以零
        std_safe = np.where(self.std == 0, 1.0, self.std)
        return (data_block - self.mean) / std_safe

    def notch_filter(self, data_block: np.ndarray) -> np.ndarray:
        """
        应用带阻滤波器滤除工频谐波。
        使用 IIR 陷波滤波器级联，并利用状态传递保证块间连续性。
        """
        if self.filter_states is None:
            self.filter_states = {}          # 每个通道对应多个滤波器状态
        b_list, a_list = self._design_notch_filters()

        filtered = np.zeros_like(data_block)
        for ch in range(data_block.shape[1]):
            # 初始化该通道的状态列表（每个滤波器一个状态）
            if ch not in self.filter_states:
                self.filter_states[ch] = [None] * len(b_list)

            x = data_block[:, ch]
            for i, (b, a) in enumerate(zip(b_list, a_list)):
                zi = self.filter_states[ch][i]
                if zi is None:
                    # 初始条件：假设输入开始前信号平稳
                    zi = signal.lfilter_zi(b, a) * x[0]
                y, zi = signal.lfilter(b, a, x, zi=zi)
                x = y
                self.filter_states[ch][i] = zi
            filtered[:, ch] = x
        return filtered

    def _design_notch_filters(self):
        """
        设计所有陷波滤波器，返回系数列表 (b_list, a_list)
        使用品质因数 Q=30（可调整）
        """
        if hasattr(self, '_notch_coeffs'):
            return self._notch_coeffs

        Q = 30.0
        b_list = []
        a_list = []
        for f_range in self.harmonics:
            f0 = np.mean(f_range)            # 中心频率
            # 设计陷波滤波器
            b, a = signal.iirnotch(f0, Q, self.sample_rate)
            b_list.append(b)
            a_list.append(a)
        self._notch_coeffs = (b_list, a_list)
        return b_list, a_list

    # ----------------------------------------------------------------------
    # 主处理流程（调整后：先全序列仪器校正，再分块处理其余步骤）
    # ----------------------------------------------------------------------
    def process(self, output_path: str, chunk_size: int = 100000, normalize=False):
        """
        执行完整的预处理流程：
        1. 读取全部数据到内存
        2. 仪器响应校正（全序列 FFT）
        3. （可选）电场校正（默认注释）
        4. 基于校正后数据计算全局统计量
        5. 分块进行去趋势、陷波滤波、归一化
        6. 保存为 HDF5 文件
        """
        with h5py.File(self.hdf5_path, 'r') as fin:
            # 读取时间轴
            times_data = fin['times'][:]
            # 读取全部原始数据
            print("正在读取全部原始数据到内存...")
            raw_data = fin['TSData'][:]
            n_samples, n_channels = raw_data.shape
            print(f"数据形状: {raw_data.shape}，占用内存约 {raw_data.nbytes / 1024 ** 2:.2f} MB")

        # ---------- 第一步：仪器响应校正（全序列频域处理）----------
        print("开始仪器响应校正（全序列 FFT）...")
        corrected_data = self.remove_instrument_response(raw_data)

        # （可选）电场校正：如需启用请取消下一行注释
        # corrected_data = self.correct_electrical_length(corrected_data)

        # 释放原始数据内存（可选）
        del raw_data

        # ---------- 第二步：基于校正后数据计算全局统计量 ----------
        print("正在计算校正后数据的全局统计量...")
        self._compute_global_stats_from_array(corrected_data)

        # ---------- 第三步：创建输出文件并分块写入处理结果 ----------
        with h5py.File(output_path, 'w') as fout:
            # 保存时间轴
            fout.create_dataset('times', data=times_data)

            # 创建输出数据集，分块大小确保不超过数据形状
            safe_chunk_size = min(chunk_size, n_samples)
            out_ds = fout.create_dataset(
                'TSData',
                shape=(n_samples, n_channels),
                dtype=corrected_data.dtype,
                chunks=(safe_chunk_size, n_channels),
                compression='gzip'
            )

            self.filter_states = {}
            total_blocks = (n_samples + safe_chunk_size - 1) // safe_chunk_size

            for block_idx, start in enumerate(range(0, n_samples, safe_chunk_size)):
                end = min(start + safe_chunk_size, n_samples)
                print(f"处理块 {block_idx + 1}/{total_blocks} (行 {start}-{end})")

                # 从校正后数据中切片
                block = corrected_data[start:end, :].copy()
                x_block = self.x[start:end]

                # 线性去趋势
                block = self.detrend(block, x_block)
                # 工频陷波滤波
                block = self.notch_filter(block)

                # 标准化（可选）
                if normalize:
                    block = self.normalize(block)

                out_ds[start:end, :] = block

            # 复制原始 HDF5 属性并添加处理标记
            with h5py.File(self.hdf5_path, 'r') as fin:
                for key, value in fin.attrs.items():
                    fout.attrs[key] = value
            fout.attrs['processed'] = True
            fout.attrs['preprocess_date'] = datetime.now().isoformat()

        print(f"预处理完成，结果已保存至: {output_path}")


# ----------------------------------------------------------------------
# 使用示例（弹窗选择文件）
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from pathlib import Path
    from pythonProject.configs.configmanager import ConfigManager

    # 隐藏主窗口
    root = tk.Tk()
    root.withdraw()

    # 选择输入 HDF5 文件
    input_path = filedialog.askopenfilename(
        title="选择输入 HDF5 文件",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    if not input_path:
        print("未选择输入文件，程序退出")
        exit(0)
    input_path = Path(input_path)

    # 选择配置文件
    config_path_str = filedialog.askopenfilename(
        title="选择配置文件 (.cfg)",
        filetypes=[("Config files", "*.cfg"), ("All files", "*.*")]
    )
    if not config_path_str:
        print("未选择配置文件，程序退出")
        exit(0)
    config_path = Path(config_path_str)

    # 加载配置
    cfg_mgr = ConfigManager.from_aether_cfg(str(config_path))

    # 修正校准文件路径为绝对路径
    cal_files = cfg_mgr.get("校准文件")
    if cal_files:
        abs_cal = []
        for f in cal_files:
            p = Path(f)
            if not p.is_absolute():
                p = config_path.parent / p
            abs_cal.append(str(p))
        cfg_mgr.add_param("校准文件", abs_cal)

    # 询问输出文件路径（可选，若取消则自动生成）
    output_path = filedialog.asksaveasfilename(
        title="保存预处理后的 HDF5 文件",
        defaultextension=".h5",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
        initialfile=f"{input_path.stem}_preprocessed.h5"
    )
    if output_path:
        output_path = Path(output_path)
    else:
        # 自动生成输出文件名
        output_path = input_path.with_name(f"{input_path.stem}_preprocessed.h5")
        print(f"未指定输出文件，将自动保存至: {output_path}")

    # 可选：设置分块大小（弹窗输入）
    chunk_size = 200000  # 默认值
    try:
        chunk_size_str = filedialog.askstring("分块大小", "请输入分块大小（样本数，默认200000）:", initialvalue="200000")
        if chunk_size_str:
            chunk_size = int(chunk_size_str)
    except:
        pass

    # 执行预处理
    try:
        pre = Preprocessor(str(input_path), cfg_mgr)
        pre.process(str(output_path), chunk_size=chunk_size)
        messagebox.showinfo("完成", f"预处理完成！\n结果已保存至:\n{output_path}")
    except Exception as e:
        messagebox.showerror("错误", f"预处理失败:\n{str(e)}")
        raise