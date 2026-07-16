import h5py
import numpy as np
import pywt
from typing import List, Tuple, Optional, Union
import warnings


class WaveletTransformTool:
    """
    小波变换工具类（MATLAB Cwavelet 逻辑）
    使用分窗 + 时间降采样策略，适用于长序列
    """

    def __init__(self, sampling_rate: float = 1.0):
        self.sampling_rate = sampling_rate
        self.data = None               # 形状 (n_channels, n_samples) 或 (n_samples, n_channels)
        self.channel_names = None
        self.cwt_results = {}          # 存储每个通道的 CWT 结果（兼容旧格式）
        self._common_frequencies = None
        self._common_scales = None
        self.time_axis = None          # 最终时间轴（秒）
        self.coi = None                # 影响锥 (n_time, n_channels)

    def load_hdf5(self, file_path: str, dataset_path: str = 'data',
                  channel_names_dataset: Optional[str] = None,
                  auto_transpose: bool = True) -> None:
        """加载 HDF5 数据，自动处理转置（兼容 PSD 服务格式）"""
        with h5py.File(file_path, 'r') as f:
            if dataset_path not in f:
                # 尝试常见名称
                for cand in ['TSData', 'data', 'signal', 'X']:
                    if cand in f:
                        dataset_path = cand
                        break
                else:
                    raise KeyError(f"未找到数据集，可用: {list(f.keys())}")
            data = f[dataset_path][:]
            if data.ndim == 1:
                data = data.reshape(1, -1)
            elif data.ndim != 2:
                raise ValueError(f"数据维度应为 1 或 2，实际为 {data.ndim}")

            # 自动转置：假设形状为 (samples, channels) 且 samples > channels
            if auto_transpose and data.shape[0] > data.shape[1] and data.shape[1] > 1:
                print(f"自动转置: (samples={data.shape[0]}, channels={data.shape[1]}) -> (channels, samples)")
                data = data.T

            self.data = data.astype(np.float64)
            n_channels, n_samples = self.data.shape

            # 读取通道名
            if channel_names_dataset and channel_names_dataset in f:
                raw = f[channel_names_dataset][:]
                self.channel_names = []
                for name in raw:
                    if isinstance(name, bytes):
                        self.channel_names.append(name.decode('utf-8'))
                    else:
                        self.channel_names.append(str(name))
                if len(self.channel_names) != n_channels:
                    warnings.warn("通道名数量不匹配，使用默认名称")
                    self.channel_names = [f'ch{i}' for i in range(n_channels)]
            else:
                self.channel_names = [f'ch{i}' for i in range(n_channels)]

            # 尝试读取时间轴（若存在）
            if 'times' in f:
                times_raw = f['times'][:]
                if times_raw.dtype.kind in ('S', 'U'):
                    # 字符串时间转为秒（相对第一个时间点）
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
                            warnings.warn(f"无法解析时间 '{ts}'，忽略 times 数据集")
                            time_sec = None
                            break
                        if first is None:
                            first = dt
                            time_sec[i] = 0.0
                        else:
                            time_sec[i] = (dt - first).total_seconds()
                    if time_sec is not None:
                        self.time_axis = time_sec
                else:
                    self.time_axis = times_raw.astype(float)
                if self.time_axis is not None and len(self.time_axis) != n_samples:
                    warnings.warn("times 长度与样本数不匹配，忽略")
                    self.time_axis = None

        # ==================== 核心方法：分窗 CWT（模仿 MATLAB Cwavelet） ====================

    # def cwt_windowed(self, window_len: int = 8192, overlap_frac: float = 2 / 3,
    #                  time_decimate: int = 200, wavelet: str = 'morl',
    #                  scales: Optional[np.ndarray] = None,
    #                  freqs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     分窗连续小波变换（完全模仿 MATLAB 的 Cwavelet 函数）
    #
    #     参数
    #     ----------
    #     window_len : int
    #         窗口长度（样本数），建议为 2 的幂（如 8192）
    #     overlap_frac : float
    #         窗口重叠比例，默认 2/3
    #     time_decimate : int
    #         CWT 结果时间轴降采样因子（每 time_decimate 个点取一个），默认 200
    #     wavelet : str
    #         小波名称，默认 'morl'
    #     scales, freqs : optional
    #         尺度或频率数组，若为 None 则自动确定
    #
    #     返回
    #     -------
    #     tms : np.ndarray
    #         时间轴（秒），形状 (n_time,)
    #     freq : np.ndarray
    #         频率轴（Hz）
    #     coi : np.ndarray
    #         影响锥，形状 (n_time, n_channels)
    #     cfs_all : np.ndarray
    #         小波系数，形状 (n_freq, n_time, n_channels)
    #     """
    #     if self.data is None:
    #         raise RuntimeError("没有加载数据，请先调用 load_hdf5()")
    #
    #     # 确保数据形状为 (samples, channels)
    #     if self.data.shape[0] < self.data.shape[1]:
    #         data = self.data.T
    #     else:
    #         data = self.data.T if self.data.shape[0] > self.data.shape[1] else self.data
    #     n_samples, n_ch = data.shape
    #
    #     # 窗口参数
    #     overlap = int(window_len * overlap_frac)
    #     offset = window_len - overlap
    #     starts = list(range(0, n_samples - window_len + 1, offset))
    #     n_windows = len(starts)
    #
    #     # 预先计算频率轴（使用第一个窗口的第一个通道）
    #     dummy_signal = data[:window_len, 0]
    #     if freqs is not None:
    #         # 用户指定频率，需转换为尺度
    #         dt = 1.0 / self.sampling_rate
    #         fc = self._get_wavelet_center_freq(wavelet)
    #         scales = fc / (freqs * dt)
    #         # 确保尺度递增
    #         if scales[0] > scales[-1]:
    #             scales = scales[::-1]
    #             freqs = freqs[::-1]
    #         # 第一次调用只是获取频率，实际计算会重新使用 scales
    #         coeffs_dummy, _ = pywt.cwt(dummy_signal, scales, wavelet,
    #                                    sampling_period=dt, method='fft')
    #         freq_out = freqs
    #     else:
    #         # 自动确定频率
    #         coeffs_dummy, scales_out = pywt.cwt(dummy_signal, scales=None, wavelet=wavelet,
    #                                             sampling_period=1.0 / self.sampling_rate,
    #                                             method='fft')
    #         dt = 1.0 / self.sampling_rate
    #         fc = self._get_wavelet_center_freq(wavelet)
    #         freq_out = fc / (scales_out * dt)
    #         scales = scales_out
    #
    #     n_freq = len(freq_out)
    #
    #     # 存储每个通道的片段
    #     all_cfs = []  # 每个元素为 (n_freq, n_time_ch) 的列表，最后拼接
    #     all_coi = []  # 每个元素为 (n_time_ch,)
    #
    #     for ch in range(n_ch):
    #         print(f"Processing channel {ch + 1}/{n_ch} ...")
    #         ch_cfs = []
    #         ch_coi = []
    #         for i, start in enumerate(starts):
    #             print(f"  Window {i + 1}/{n_windows} ...")
    #             end = start + window_len
    #             seg = data[start:end, ch]
    #             # CWT
    #             coeffs, scales_out = pywt.cwt(seg, scales, wavelet,
    #                                           sampling_period=1.0 / self.sampling_rate,
    #                                           method='fft')
    #             # 计算影响锥 (COI) - 简单近似：取边缘效应区域，实际应根据小波计算
    #             # MATLAB 中 cwt 返回第三个输出为 cone of influence，这里我们模拟
    #             # 根据小波的有效支撑长度估计 COI
    #             # 对于 morlet，边缘效应区域约为 2*尺度 对应的点数
    #             # 为简化，取 coeffs 每列的最小有效值区域，这里直接返回全为 1 的数组（后续可改进）
    #             coi_win = np.ones(coeffs.shape[1])  # 临时占位
    #             # 时间降采样
    #             if time_decimate > 1:
    #                 coeffs = coeffs[:, ::time_decimate]
    #                 coi_win = coi_win[::time_decimate]
    #             # 截取片段
    #             if i == 0:
    #                 # 第一个窗口：取前 overlap 部分（时间降采样后）
    #                 take_end = overlap // time_decimate if time_decimate > 1 else overlap
    #                 cfs_slice = coeffs[:, :take_end]
    #                 coi_slice = coi_win[:take_end]
    #             elif i == n_windows - 1:
    #                 # 最后一个窗口：取后 overlap 部分
    #                 take_start = - (overlap // time_decimate) if time_decimate > 1 else -overlap
    #                 cfs_slice = coeffs[:, take_start:]
    #                 coi_slice = coi_win[take_start:]
    #             else:
    #                 # 中间窗口：取中间 offset 部分
    #                 start_idx = (offset // time_decimate) if time_decimate > 1 else offset
    #                 end_idx = start_idx + (offset // time_decimate) if time_decimate > 1 else start_idx + offset
    #                 cfs_slice = coeffs[:, start_idx:end_idx]
    #                 coi_slice = coi_win[start_idx:end_idx]
    #             ch_cfs.append(cfs_slice)
    #             ch_coi.append(coi_slice)
    #         # 拼接该通道的所有时间片段
    #         cfs_ch = np.hstack(ch_cfs)  # (n_freq, n_time_ch)
    #         coi_ch = np.hstack(ch_coi)  # (n_time_ch,)
    #         all_cfs.append(cfs_ch)
    #         all_coi.append(coi_ch)
    #
    #     # 堆叠所有通道为 3D 数组 (n_freq, n_time, n_ch)
    #     cfs_all = np.stack(all_cfs, axis=2)
    #     n_time = cfs_all.shape[1]
    #     coi = np.stack(all_coi, axis=1)  # (n_time, n_ch)
    #
    #     # 构建时间轴（秒）
    #     total_time = n_samples / self.sampling_rate
    #     tms = np.linspace(0, total_time, n_time)
    #
    #     # 存储到类属性（兼容原有绘图接口）
    #     self.cwt_results.clear()
    #     for i, name in enumerate(self.channel_names):
    #         self.cwt_results[name] = {
    #             'coeffs': cfs_all[:, :, i],
    #             'frequencies': freq_out,
    #             'scales': scales,
    #             'wavelet': wavelet
    #         }
    #     self._common_frequencies = freq_out
    #     self._common_scales = scales
    #     self.time_axis = tms
    #     self.coi = coi
    #
    #     return tms, freq_out, coi, cfs_all
    def cwt_windowed(self, window_len: int = 8192, overlap_frac: float = 2 / 3,
                     time_decimate: int = 200, wavelet: str = 'morl',
                     scales: Optional[np.ndarray] = None,
                     freqs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        分窗连续小波变换（修正版：基于 COI 的重叠加权拼接，保留低频）

        返回
        -------
        tms : np.ndarray          时间轴（秒）
        freq : np.ndarray         频率轴（Hz）
        coi : np.ndarray          影响锥掩码（1=可靠，0=不可靠），形状 (n_time, n_channels)
        cfs_all : np.ndarray      小波系数，形状 (n_freq, n_time, n_channels)
        """
        if self.data is None:
            raise RuntimeError("没有加载数据，请先调用 load_hdf5()")

        # ---- 0. 准备数据 (转为 samples × channels) ----
        if self.data.shape[0] < self.data.shape[1]:
            data = self.data.T
        else:
            data = self.data.T if self.data.shape[0] > self.data.shape[1] else self.data
        n_samples, n_ch = data.shape
        dt = 1.0 / self.sampling_rate

        # ---- 1. 确定频率/尺度轴 ----
        dummy_signal = data[:min(window_len, n_samples), 0]  # 截短以免空信号
        if freqs is not None:
            fc = self._get_wavelet_center_freq(wavelet)
            scales_used = fc / (freqs * dt)
            if scales_used[0] > scales_used[-1]:
                scales_used = scales_used[::-1]
                freqs = freqs[::-1]
            _, _ = pywt.cwt(dummy_signal, scales_used, wavelet,
                            sampling_period=dt, method='fft')
            freq_out = freqs
        else:
            coeffs_dummy, scales_out = pywt.cwt(dummy_signal, None, wavelet,
                                                sampling_period=dt, method='fft')
            fc = self._get_wavelet_center_freq(wavelet)
            freq_out = fc / (scales_out * dt)
            scales_used = scales_out

        n_freq = len(freq_out)

        # ---- 2. 窗口参数与降采样后的总时间点数 ----
        overlap = int(window_len * overlap_frac)
        offset = window_len - overlap
        starts = list(range(0, n_samples - window_len + 1, offset))
        n_windows = len(starts)

        # 降采样后的时间索引总数
        n_time_dec = (n_samples + time_decimate - 1) // time_decimate if time_decimate > 1 else n_samples

        # ---- 3. 初始化累加器 (n_freq, n_time_dec, n_ch) ----
        acc_cfs = np.zeros((n_freq, n_time_dec, n_ch), dtype=np.complex128)
        acc_weight = np.zeros((n_freq, n_time_dec, n_ch), dtype=np.float64)

        # ---- 4. 逐通道 / 逐窗口处理 ----
        for ch in range(n_ch):
            print(f"Processing channel {ch + 1}/{n_ch} ...")
            for i, start in enumerate(starts):
                print(f"  Window {i + 1}/{n_windows} start={start} ...")
                end = start + window_len
                seg = data[start:end, ch]

                # CWT（返回 coeffs 形状 (n_freq, window_len)）
                coeffs, _ = pywt.cwt(seg, scales_used, wavelet,
                                     sampling_period=dt, method='fft')

                # 生成 COI 掩码（1 = 可靠）
                mask = self._compute_coi_mask(scales_used, window_len, dt, wavelet)

                # 掩蔽不可靠区域（置零）
                coeffs_masked = coeffs * mask  # mask 自动广播到 (n_freq, window_len)

                # 时间降采样（简单抽取）
                if time_decimate > 1:
                    idx_ds = np.arange(0, window_len, time_decimate)
                    coeffs_ds = coeffs_masked[:, idx_ds]  # (n_freq, len(idx_ds))
                    mask_ds = mask[:, idx_ds]
                else:
                    coeffs_ds = coeffs_masked
                    mask_ds = mask

                # 全局时间索引（降采样后的起始位置）
                t_start = start // time_decimate
                t_len = coeffs_ds.shape[1]

                # 累加
                acc_cfs[:, t_start:t_start + t_len, ch] += coeffs_ds
                acc_weight[:, t_start:t_start + t_len, ch] += mask_ds

        # ---- 5. 归一化（可靠区域加权平均） ----
        # 避免除零
        valid = acc_weight > 1e-12
        acc_cfs[valid] /= acc_weight[valid]
        cfs_all = acc_cfs  # (n_freq, n_time_dec, n_ch)

        # ---- 6. 生成 COI（所有通道的可靠性掩码） ----
        # 至少有一个尺度可靠的像素，认为该时间点整体可靠（下游可进一步按尺度筛选）
        coi = (np.max(acc_weight, axis=0) > 0).astype(float)  # (n_time_dec, n_ch)

        # ---- 7. 构造时间轴 ----
        total_time = n_samples / self.sampling_rate
        tms = np.linspace(0, total_time, n_time_dec)

        # ---- 8. 更新类属性（兼容旧接口） ----
        self.cwt_results.clear()
        for i_ch, name in enumerate(self.channel_names):
            self.cwt_results[name] = {
                'coeffs': cfs_all[:, :, i_ch],
                'frequencies': freq_out,
                'scales': scales_used,
                'wavelet': wavelet
            }
        self._common_frequencies = freq_out
        self._common_scales = scales_used
        self.time_axis = tms
        self.coi = coi

        return tms, freq_out, coi, cfs_all

    def _compute_coi_mask(self, scales: np.ndarray, n_samples: int,
                          dt: float, wavelet: str = 'morl') -> np.ndarray:
        """
        生成 COI 掩码矩阵 (n_scales, n_samples)
        可靠区域 = 1，边缘污染区 = 0
        """
        # 小波边缘影响半宽度（样本点数）
        # 对于 Morlet / 复小波，半宽 = scale * sqrt(2) / dt
        edge_factor = np.sqrt(2)  # 对于 cmor / morl 通用
        # 如果小波名包含 'morl'，保持该因子；其他小波也可微调，这里统一用 sqrt(2)
        half_support = np.ceil(scales * edge_factor / dt).astype(int)  # shape (n_scales,)

        mask = np.ones((len(scales), n_samples), dtype=np.float64)
        for i, half in enumerate(half_support):
            if half >= n_samples // 2:
                # 整个窗口都被污染，掩码全0
                mask[i, :] = 0
            else:
                mask[i, :half] = 0
                mask[i, n_samples - half:] = 0
        return mask
    def _compute_coi_from_cwt(self, wavelet, scales, n_samples, dt):
        # 使用空信号让 pywt 返回 COI（与信号无关，只取决于小波和尺度、时间轴）
        dummy = np.zeros(n_samples)
        _, _, coi_array = pywt.cwt(dummy, scales, wavelet, sampling_period=dt, method='fft')
        # coi_array 的形状为 (n_samples,)，每个时间点对应的最大频率，低于该频率的值不可靠
        # 我们要转成掩码矩阵 (n_scales, n_samples)：对于每个尺度对应的频率 freq_i，
        # 若 freq_i > coi_array[j] 则为不可靠（边缘影响），否则可靠
        freqs = self._frequencies_from_scales(scales, wavelet)
        mask = np.zeros((len(scales), n_samples), dtype=np.float64)
        for i, f in enumerate(freqs):
            # 当频率 f 大于该时刻的 coi 频率，说明该时刻该尺度不可靠
            mask[i, :] = (f <= coi_array).astype(np.float64)
        return mask

    def save_data_hdf5(self, output_path: str, dataset_path: str = 'data',
                       channel_names_dataset: str = 'channel_names') -> None:
        if self.data is None:
            raise RuntimeError("没有加载任何数据")
        with h5py.File(output_path, 'w') as f:
            f.create_dataset(dataset_path, data=self.data)
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset(channel_names_dataset, data=self.channel_names, dtype=dt)

    def _get_wavelet_center_freq(self, wavelet_name: str) -> float:
        """获取小波中心频率"""
        if wavelet_name.lower() == 'morl':
            return 0.8125
        try:
            return pywt.central_frequency(wavelet_name)
        except:
            return 0.8125

    def _scales_from_frequencies(self, freqs: np.ndarray, wavelet: str) -> np.ndarray:
        dt = 1.0 / self.sampling_rate
        fc = self._get_wavelet_center_freq(wavelet)
        scales = fc / (freqs * dt)
        return scales

    def _frequencies_from_scales(self, scales: np.ndarray, wavelet: str) -> np.ndarray:
        dt = 1.0 / self.sampling_rate
        fc = self._get_wavelet_center_freq(wavelet)
        freqs = fc / (scales * dt)
        return freqs

    def continuous_wavelet_transform(self, channel_idx: Union[int, str],
                                      wavelet: str = 'morl',
                                      scales: Optional[np.ndarray] = None,
                                      freqs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.data is None:
            raise RuntimeError("没有加载数据，请先调用 load_hdf5()")
        if isinstance(channel_idx, str):
            if self.channel_names is None:
                raise ValueError("未加载通道名")
            try:
                idx = self.channel_names.index(channel_idx)
            except ValueError:
                raise ValueError(f"通道名 '{channel_idx}' 不存在")
        else:
            idx = channel_idx
        data_signal = self.data[idx, :]   # 关键：避免变量名 signal 与模块冲突

        if scales is None:
            if freqs is not None:
                scales = self._scales_from_frequencies(freqs, wavelet)
                if scales[0] > scales[-1]:
                    scales = scales[::-1]
                    freqs = freqs[::-1]
            else:
                n_scales = 32
                dt = 1.0 / self.sampling_rate
                fc = self._get_wavelet_center_freq(wavelet)
                max_freq = self.sampling_rate / 2.0
                min_freq = max_freq / 32.0
                max_scale = fc / (min_freq * dt)
                min_scale = fc / (max_freq * dt)
                scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales)
        else:
            scales = np.asarray(scales)
            if scales[0] > scales[-1]:
                scales = scales[::-1]
                if freqs is not None:
                    freqs = freqs[::-1]

        coeffs, scales_out = pywt.cwt(data_signal, scales, wavelet, sampling_period=1.0/self.sampling_rate)
        frequencies = self._frequencies_from_scales(scales_out, wavelet)

        if freqs is not None and len(freqs) > 1:
            if frequencies[0] > frequencies[-1]:
                coeffs = coeffs[::-1, :]
                frequencies = frequencies[::-1]
                scales_out = scales_out[::-1]

        return coeffs, scales_out, frequencies

    def cwt_for_all_channels(self, wavelet: str = 'morl',
                              scales: Optional[np.ndarray] = None,
                              freqs: Optional[np.ndarray] = None) -> None:
        if self.data is None:
            raise RuntimeError("没有加载数据")
        self.cwt_results.clear()
        for i, name in enumerate(self.channel_names):
            coeffs, scales_out, freqs_out = self.continuous_wavelet_transform(i, wavelet, scales, freqs)
            self.cwt_results[name] = {
                'coeffs': coeffs,
                'scales': scales_out,
                'frequencies': freqs_out,
                'wavelet': wavelet
            }
        if self.cwt_results:
            first = next(iter(self.cwt_results.values()))
            self._common_frequencies = first['frequencies']
            self._common_scales = first['scales']

    def save_cwt_results(self, output_path: str, dataset_prefix: str = 'cwt') -> None:
        """保存 CWT 结果到 HDF5"""
        if not self.cwt_results:
            raise RuntimeError("没有 CWT 结果，请先调用 cwt_windowed()")
        with h5py.File(output_path, 'w') as f:
            for ch_name, res in self.cwt_results.items():
                grp = f.create_group(f"{dataset_prefix}_{ch_name}")
                grp.create_dataset('coeffs', data=res['coeffs'])
                grp.create_dataset('frequencies', data=res['frequencies'])
                if res['scales'] is not None:
                    grp.create_dataset('scales', data=res['scales'])
                grp.attrs['wavelet'] = res['wavelet']
                grp.attrs['sampling_rate'] = self.sampling_rate
            # 保存时间轴和 COI
            if self.time_axis is not None:
                f.create_dataset('time_axis', data=self.time_axis)
            if self.coi is not None:
                f.create_dataset('coi', data=self.coi)

    def wavelet_coherence(self, channel1: Union[int, str], channel2: Union[int, str],
                          wavelet: str = 'morl',
                          scales: Optional[np.ndarray] = None,
                          freqs: Optional[np.ndarray] = None,
                          time_smoothing_window: float = 0.5,
                          scale_smoothing_window: float = 0.5
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(channel1, str):
            idx1 = self.channel_names.index(channel1)
        else:
            idx1 = channel1
        if isinstance(channel2, str):
            idx2 = self.channel_names.index(channel2)
        else:
            idx2 = channel2

        coeffs1, scales1, freqs1 = self.continuous_wavelet_transform(idx1, wavelet, scales, freqs)
        coeffs2, scales2, freqs2 = self.continuous_wavelet_transform(idx2, wavelet, scales, freqs)
        if not np.array_equal(scales1, scales2):
            raise ValueError("两个通道的尺度数组不一致")
        scales = scales1
        frequencies = freqs1

        cross_spec = coeffs1 * np.conj(coeffs2)
        power1 = np.abs(coeffs1) ** 2
        power2 = np.abs(coeffs2) ** 2

        dt = 1.0 / self.sampling_rate
        time_win_len = int(time_smoothing_window / dt)
        if time_win_len % 2 == 0:
            time_win_len += 1
        if time_win_len > 1:
            gauss_win = np.exp(-0.5 * ((np.arange(time_win_len) - time_win_len//2) / (time_win_len/6))**2)
            gauss_win /= gauss_win.sum()
            cross_spec_sm = self._smooth_along_axis(cross_spec, gauss_win, axis=1)
            power1_sm = self._smooth_along_axis(power1, gauss_win, axis=1)
            power2_sm = self._smooth_along_axis(power2, gauss_win, axis=1)
        else:
            cross_spec_sm = cross_spec
            power1_sm = power1
            power2_sm = power2

        if scale_smoothing_window > 0:
            log_scales = np.log2(scales)
            dlog = np.mean(np.diff(log_scales))
            idx_win = max(1, int(scale_smoothing_window / dlog))
            if idx_win % 2 == 0:
                idx_win += 1
            if idx_win > 1:
                gauss_win_sc = np.exp(-0.5 * ((np.arange(idx_win) - idx_win//2) / (idx_win/6))**2)
                gauss_win_sc /= gauss_win_sc.sum()
                cross_spec_sm = self._smooth_along_axis(cross_spec_sm, gauss_win_sc, axis=0)
                power1_sm = self._smooth_along_axis(power1_sm, gauss_win_sc, axis=0)
                power2_sm = self._smooth_along_axis(power2_sm, gauss_win_sc, axis=0)

        coherence = np.abs(cross_spec_sm) ** 2 / (power1_sm * power2_sm + 1e-12)
        coherence = np.clip(coherence, 0, 1)
        phase = np.angle(cross_spec_sm)

        return coherence, phase, frequencies, scales

    def _smooth_along_axis(self, arr: np.ndarray, window: np.ndarray, axis: int) -> np.ndarray:
        if np.iscomplexobj(arr):
            real_sm = self._smooth_along_axis(arr.real, window, axis)
            imag_sm = self._smooth_along_axis(arr.imag, window, axis)
            return real_sm + 1j * imag_sm
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (len(window)//2, len(window)//2)
        arr_pad = np.pad(arr, pad_width, mode='reflect')
        slices = [slice(None)] * arr.ndim
        result = np.zeros_like(arr)
        for i in range(arr.shape[axis]):
            slices[axis] = i
            src_slices = slices.copy()
            src_slices[axis] = slice(i, i + len(window))
            result[tuple(slices)] = np.sum(arr_pad[tuple(src_slices)] * window, axis=axis)
        return result

    def save_coherence_phase(self, output_path: str, channel1: Union[int, str], channel2: Union[int, str],
                             coherence: np.ndarray, phase: np.ndarray, frequencies: np.ndarray, scales: np.ndarray,
                             wavelet: str, group_name: Optional[str] = None) -> None:
        if isinstance(channel1, str):
            name1 = channel1
        else:
            name1 = self.channel_names[channel1] if self.channel_names else f'ch{channel1}'
        if isinstance(channel2, str):
            name2 = channel2
        else:
            name2 = self.channel_names[channel2] if self.channel_names else f'ch{channel2}'
        if group_name is None:
            group_name = f"coherence_{name1}_{name2}"
        with h5py.File(output_path, 'w') as f:
            grp = f.create_group(group_name)
            grp.create_dataset('coherence', data=coherence)
            grp.create_dataset('phase', data=phase)
            grp.create_dataset('frequencies', data=frequencies)
            grp.create_dataset('scales', data=scales)
            grp.attrs['channel1'] = name1
            grp.attrs['channel2'] = name2
            grp.attrs['wavelet'] = wavelet
            grp.attrs['sampling_rate'] = self.sampling_rate