import os
import h5py
import numpy as np
import pywt
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional, Dict, Any


class CwaveletProcessor:
    """
    完全对照 MATLAB Cwavelet 的连续小波变换处理器
    支持从 HDF5 文件读取数据，执行分窗 CWT，并保存结果
    """

    def __init__(self, sampling_rate: float, window_len: int,
                 time_decimate: int = 200, wavelet: str = 'cmor60-0.8125',
                 parallel: bool = True, n_workers: int = 4):
        """
        参数
        ----------
        sampling_rate : float
            采样频率 (Hz)
        window_len : int
            分窗长度（样本数）
        time_decimate : int
            时间降采样因子（每 time_decimate 个点取一个），默认 200
        wavelet : str
            小波名称，默认 'cmor60-0.8125'（与 MATLAB Morse 小波行为一致）
        parallel : bool
            是否并行处理中间窗口，默认 True
        n_workers : int
            并行线程数，默认 4
        """
        self.sampling_rate = sampling_rate
        self.window_len = window_len
        self.time_decimate = time_decimate
        self.wavelet = wavelet
        self.parallel = parallel
        self.n_workers = n_workers

        # 数据存储
        self.data = None            # 形状 (n_samples, n_channels)
        self.channel_names = None   # 可选
        self.original_times = None  # 原始时间轴（如果文件提供）

        # CWT 结果
        self.tms = None             # 时间轴 (秒)
        self.freq = None            # 频率轴 (Hz)
        self.coi = None             # 影响锥 (n_time, n_ch)
        self.cfs = None             # 小波系数 (n_freq, n_time, n_ch)

    # ------------------------------------------------------------------
    # 1. 读取 HDF5 文件
    # ------------------------------------------------------------------
    def load_hdf5(self, file_path: str,
                  dataset_name: str = 'TSData',
                  times_dataset: str = 'times',
                  channel_names_dataset: Optional[str] = None,
                  auto_transpose: bool = True) -> None:
        """
        从 HDF5 文件加载数据

        参数
        ----------
        file_path : str
            HDF5 文件路径
        dataset_name : str
            信号数据集名称，默认为 'TSData'
        times_dataset : str
            时间轴数据集名称，默认为 'times'（可选）
        channel_names_dataset : str
            通道名称数据集（可选）
        auto_transpose : bool
            是否自动转置，使得形状变为 (samples, channels)。默认 True。
            若数据形状为 (channels, samples) 且 channels < samples 则转置。
        """
        with h5py.File(file_path, 'r') as f:
            # 读取信号数据
            if dataset_name not in f:
                # 尝试常见备选名称
                for cand in ['TSData', 'data', 'signal', 'X']:
                    if cand in f:
                        dataset_name = cand
                        break
                else:
                    raise KeyError(f"未找到数据集 '{dataset_name}'，可用: {list(f.keys())}")
            data = f[dataset_name][:]
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            elif data.ndim != 2:
                raise ValueError(f"数据维度应为 1 或 2，实际为 {data.ndim}")

            # 自动转置 (samples, channels) 为第一维
            if auto_transpose and data.shape[0] < data.shape[1]:
                # 如果通道数多于样本数，假设是 (channels, samples) 需要转置
                print(f"自动转置: 原始形状 {data.shape} -> {data.T.shape}")
                data = data.T
            self.data = data.astype(np.float64)   # 形状 (n_samples, n_channels)

            # 读取通道名称
            if channel_names_dataset and channel_names_dataset in f:
                raw = f[channel_names_dataset][:]
                names = []
                for name in raw:
                    if isinstance(name, bytes):
                        names.append(name.decode('utf-8'))
                    else:
                        names.append(str(name))
                if len(names) == self.data.shape[1]:
                    self.channel_names = names
                else:
                    print(f"通道名数量 ({len(names)}) 与数据通道数 ({self.data.shape[1]}) 不匹配，使用默认名称")
                    self.channel_names = [f'ch{i}' for i in range(self.data.shape[1])]
            else:
                self.channel_names = [f'ch{i}' for i in range(self.data.shape[1])]

            # 读取时间轴（如果存在且可用）
            if times_dataset in f:
                times_raw = f[times_dataset][:]
                if times_raw.dtype.kind in ('S', 'U'):
                    # 字符串时间，尝试转换
                    from datetime import datetime
                    first = None
                    time_sec = np.zeros(len(times_raw), dtype=float)
                    for i, ts in enumerate(times_raw):
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
                            print("无法解析 times，将忽略")
                            self.original_times = None
                            break
                        if first is None:
                            first = dt
                            time_sec[i] = 0.0
                        else:
                            time_sec[i] = (dt - first).total_seconds()
                    else:
                        self.original_times = time_sec
                else:
                    # 数值时间，假设已经是秒
                    self.original_times = times_raw.astype(float)
                # 检查长度是否匹配
                if self.original_times is not None and len(self.original_times) != self.data.shape[0]:
                    print("times 长度与样本数不匹配，将忽略")
                    self.original_times = None

    # ------------------------------------------------------------------
    # 2. 核心 CWT 变换（内部函数）
    # ------------------------------------------------------------------
    def _auto_scales(self, n_samples: int, n_scales: int = 128, min_freq: float = 0.01) -> np.ndarray:
        dt = 1.0 / self.sampling_rate
        fc = self._get_wavelet_center_freq()
        max_freq = self.sampling_rate / 2.0
        min_scale = fc / (max_freq * dt)
        max_scale = fc / (min_freq * dt)
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales)
        return scales

    def _get_wavelet_center_freq(self) -> float:
        """获取小波的中心频率（归一化）"""
        if self.wavelet.startswith('cmor'):
            parts = self.wavelet.split('-')
            if len(parts) == 2:
                return float(parts[1])
            else:
                return 0.8125
        else:
            try:
                return pywt.central_frequency(self.wavelet)
            except:
                return 0.8125

    def _slice_len(self, total: int, step: int, start: int, end: int) -> int:
        """计算 [start, end) 内每隔 step 取一个点的数量"""
        return len(np.arange(start, end, step))

    def _compute_coi(self, data_seg: np.ndarray, scales: np.ndarray, freq: np.ndarray, dt: float) -> np.ndarray:
        """
        模拟 MATLAB cwt 返回的 cone of influence (COI)
        返回形状 (n_samples,) 每个时间点的最大可靠频率 (Hz)
        """
        n_pts = len(data_seg)
        coi_freq = np.full(n_pts, np.inf)
        half_support = np.ceil(scales * np.sqrt(2) / dt).astype(int)
        for idx, (scale, f_center) in enumerate(zip(scales, freq)):
            half = half_support[idx]
            if half >= n_pts:
                coi_freq = np.minimum(coi_freq, f_center)
            else:
                coi_freq[:half] = np.minimum(coi_freq[:half], f_center)
                coi_freq[-half:] = np.minimum(coi_freq[-half:], f_center)
        coi_freq[np.isinf(coi_freq)] = freq[-1]
        return coi_freq

    def compute(self, scales: Optional[np.ndarray] = None) -> None:
        """
        执行连续小波变换（分窗模式）

        参数
        ----------
        scales : np.ndarray, optional
            尺度数组，若为 None 则自动生成
        """
        if self.data is None:
            raise RuntimeError("未加载数据，请先调用 load_hdf5()")
        n_samples, n_ch = self.data.shape
        dt = 1.0 / self.sampling_rate
        WLen = self.window_len
        time_decimate = self.time_decimate

        if n_samples <= WLen:
            raise ValueError(f"数据长度 ({n_samples}) 小于窗口长度 ({WLen})，无法分窗")

        # ---- 窗口划分 ----
        overlap = int(WLen * 2 / 3)
        offset = WLen - overlap
        n_windows = (n_samples - overlap) // offset
        if n_windows == 0:
            raise ValueError("窗口数量为0，请调整窗口长度")
        starts = [i * offset for i in range(n_windows)]
        windows = [self.data[start:start+WLen, :] for start in starts]

        # ---- 确定尺度与频率 ----
        if scales is None:
            scales = self._auto_scales(WLen, min_freq=0.01, n_scales=128)
        scales = np.asarray(scales)
        if scales[0] > scales[-1]:
            scales = scales[::-1]
        fc = self._get_wavelet_center_freq()
        freq = fc / (scales * dt)
        if freq[0] > freq[-1]:
            freq = freq[::-1]
            scales = scales[::-1]
        n_freq = len(freq)

        # ---- 预分配最终结果 ----
        len_first = self._slice_len(WLen, time_decimate, 0, overlap)
        len_mid   = self._slice_len(WLen, time_decimate, offset, overlap)
        len_last  = self._slice_len(WLen, time_decimate, offset, WLen)
        total_pts = len_first + (n_windows - 2) * len_mid + len_last
        cfs_all = np.zeros((n_freq, total_pts, n_ch), dtype=np.complex128)
        coi_all = np.zeros((total_pts, n_ch), dtype=np.float64)

        # ---- 逐通道处理 ----
        for ch in range(n_ch):
            print(f"Processing channel {ch+1}/{n_ch} ...")
            cfs_cells = []
            coi_cells = []

            # 第一个窗口
            win0 = windows[0][:, ch]
            coeffs, _ = pywt.cwt(win0, scales, self.wavelet, sampling_period=dt, method='fft')
            if freq[0] > freq[-1]:
                coeffs = coeffs[::-1, :]
            coi_win = self._compute_coi(win0, scales, freq, dt)
            idx = np.arange(0, overlap, time_decimate)
            cfs_cells.append(coeffs[:, idx])
            coi_cells.append(coi_win[idx])

            # 中间窗口（并行）
            mid_windows = windows[1:-1]
            if self.parallel and mid_windows:
                def process_mid(win_data):
                    coeffs, _ = pywt.cwt(win_data, scales, self.wavelet,
                                         sampling_period=dt, method='fft')
                    if freq[0] > freq[-1]:
                        coeffs = coeffs[::-1, :]
                    coi_win = self._compute_coi(win_data, scales, freq, dt)
                    idx = np.arange(offset, overlap, time_decimate)
                    return coeffs[:, idx], coi_win[idx]

                with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                    futures = [executor.submit(process_mid, win[:, ch]) for win in mid_windows]
                    for future in as_completed(futures):
                        cfs_part, coi_part = future.result()
                        cfs_cells.append(cfs_part)
                        coi_cells.append(coi_part)
            else:
                for win in mid_windows:
                    win_data = win[:, ch]
                    coeffs, _ = pywt.cwt(win_data, scales, self.wavelet,
                                         sampling_period=dt, method='fft')
                    if freq[0] > freq[-1]:
                        coeffs = coeffs[::-1, :]
                    coi_win = self._compute_coi(win_data, scales, freq, dt)
                    idx = np.arange(offset, overlap, time_decimate)
                    cfs_cells.append(coeffs[:, idx])
                    coi_cells.append(coi_win[idx])

            # 最后一个窗口
            win_last = windows[-1][:, ch]
            coeffs, _ = pywt.cwt(win_last, scales, self.wavelet,
                                 sampling_period=dt, method='fft')
            if freq[0] > freq[-1]:
                coeffs = coeffs[::-1, :]
            coi_win = self._compute_coi(win_last, scales, freq, dt)
            idx = np.arange(offset, WLen, time_decimate)
            cfs_cells.append(coeffs[:, idx])
            coi_cells.append(coi_win[idx])

            # 拼接
            cfs_all[:, :, ch] = np.hstack(cfs_cells)
            coi_all[:, ch] = np.hstack(coi_cells)

        # ---- 时间轴 ----
        total_time = n_samples / self.sampling_rate
        nt = coi_all.shape[0]
        tms = np.linspace(total_time / nt, total_time, nt)

        # ---- 保存结果 ----
        self.tms = tms
        self.freq = freq
        self.coi = coi_all
        self.cfs = cfs_all

        # 同时为兼容性，构造一个字典存储每个通道的结果（可选）
        self.cwt_results = {}
        for i, name in enumerate(self.channel_names):
            self.cwt_results[name] = {
                'coeffs': self.cfs[:, :, i],
                'frequencies': self.freq,
                'scales': scales,
                'wavelet': self.wavelet
            }

    # ------------------------------------------------------------------
    # 3. 保存结果到 HDF5
    # ------------------------------------------------------------------
    def save(self, output_dir: str, source_filepath: Optional[str] = None,
             add_suffix: str = '.cwt.result.h5') -> str:
        """
        保存 CWT 结果到 HDF5 文件

        参数
        ----------
        output_dir : str
            输出目录
        source_filepath : str, optional
            原始数据文件路径（用于生成输出文件名）。若未提供，则使用 'cwt_result'
        add_suffix : str
            输出文件名后缀，默认为 '.cwt.result.h5'

        返回
        -------
        out_path : str
            保存的文件路径
        """
        if self.cfs is None:
            raise RuntimeError("尚未执行 CWT 计算，请先调用 compute()")

        # 生成输出文件名
        if source_filepath is not None:
            base = os.path.splitext(os.path.basename(source_filepath))[0]
        else:
            base = "cwt_result"
        out_filename = base + add_suffix
        out_path = os.path.join(output_dir, out_filename)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        with h5py.File(out_path, 'w') as f:
            # 保存主要结果
            f.create_dataset('tms', data=self.tms)
            f.create_dataset('freq', data=self.freq)
            f.create_dataset('coi', data=self.coi)
            f.create_dataset('cfs', data=self.cfs)

            # 保存元数据
            metadata = {
                'sampling_rate': self.sampling_rate,
                'window_len': self.window_len,
                'time_decimate': self.time_decimate,
                'wavelet': self.wavelet,
                'n_channels': self.data.shape[1] if self.data is not None else 0,
                'n_samples': self.data.shape[0] if self.data is not None else 0,
            }
            for key, val in metadata.items():
                f.attrs[key] = val

            # 保存通道名
            if self.channel_names is not None:
                dt_str = h5py.string_dtype(encoding='utf-8')
                f.create_dataset('channel_names', data=self.channel_names, dtype=dt_str)

            # 可选：保存原始时间轴（如果存在）
            if self.original_times is not None:
                f.create_dataset('original_times', data=self.original_times)

        print(f"CWT 结果已保存至: {out_path}")
        return out_path


# ========================= 使用示例 =========================
if __name__ == '__main__':
    # 1. 生成示例数据（模拟 HDF5 文件）
    import tempfile
    fs = 100.0
    t = np.linspace(0, 10, int(10*fs))
    signal = np.column_stack([np.sin(2*np.pi*5*t), np.cos(2*np.pi*10*t)])
    times = t  # 秒为单位的时间轴

    with h5py.File('test_data.h5', 'w') as f:
        f.create_dataset('TSData', data=signal)
        f.create_dataset('times', data=times)
        f.create_dataset('channel_names', data=['chA', 'chB'], dtype=h5py.string_dtype())

    # 2. 使用处理器
    processor = CwaveletProcessor(sampling_rate=fs, window_len=512,
                                  time_decimate=200, parallel=True, n_workers=4)
    processor.load_hdf5('test_data.h5', dataset_name='TSData', times_dataset='times')
    processor.compute()
    processor.save(output_dir='./results', source_filepath='test_data.h5')

    # 清理示例文件
    import os
    os.remove('test_data.h5')