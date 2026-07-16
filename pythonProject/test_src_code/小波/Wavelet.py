"""
小波变换服务类 (WaveletService)
依赖：numpy, h5py, pycwt, concurrent.futures
配置源：ConfigManager（见前文）
"""

import h5py
import numpy as np
from pycwt import cwt, Morlet
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


class WaveletService:
    """
    小波变换核心服务类，支持全局与分窗两种模式。
    配置通过 ConfigManager 实例提供，支持中文/英文键名。
    """

    # 配置键名常量（中文）
    KEY_FS = '采样率 (Hz)'
    KEY_WAVELET = '小波类型'
    KEY_FREQ_MIN = '最小频率 (Hz)'
    KEY_FREQ_MAX = '最大频率 (Hz)'
    KEY_NUM_FREQS = '频率点数'
    KEY_WINDOW_SEC = '窗口长度 (秒)'
    KEY_OVERLAP = '重叠比例'
    KEY_CHANNELS = '通道选择'

    def __init__(self, config):
        """
        Parameters
        ----------
        config : ConfigManager
            配置管理器，需包含采样率、小波类型、频率范围等参数。
        """
        self.config = config
        # 提取小波必需参数（必要时提供默认值）
        self.fs = float(self.config.get(self.KEY_FS))
        self.wavelet_name = self.config.get(self.KEY_WAVELET, 'morlet')
        freq_min = float(self.config.get(self.KEY_FREQ_MIN))
        freq_max = float(self.config.get(self.KEY_FREQ_MAX))
        num_freqs = int(self.config.get(self.KEY_NUM_FREQS))
        self.window_seconds = float(self.config.get(self.KEY_WINDOW_SEC, 60.0))
        self.overlap = float(self.config.get(self.KEY_OVERLAP, 0.5))
        self.channels = self.config.get(self.KEY_CHANNELS, None)  # 可选的通道列表

        # 频率与尺度
        self.freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), num_freqs)


        # 内部状态
        self._data = None      # shape (n_times, n_channels)
        self._time = None
        self._channel_names = []

    def _freq2scale(self, freqs):
        dt = 1.0 / self.fs
        w0 = 6
        scale = (w0 + np.sqrt(2 + w0 ** 2)) / (4 * np.pi * freqs * dt)
        return scale

    def load_data(self, hdf5_path: str, dataset_name='data', time_name='time'):
        """
        从 HDF5 读取时间序列数据（时间 × 通道）。

        Parameters
        ----------
        hdf5_path : str
        dataset_name : str
            主数据集名称，默认 'data'
        time_name : str
            时间轴数据集名称，默认 'time'
        """
        with h5py.File(hdf5_path, 'r') as f:
            self._data = f[dataset_name][:]  # (n_times, n_channels)
            if time_name in f:
                self._time = f[time_name][:]
            else:
                self._time = np.arange(self._data.shape[0]) / self.fs
            if 'channels' in f.attrs:
                self._channel_names = list(f.attrs['channels'])
            else:
                self._channel_names = [f'ch{i}' for i in range(self._data.shape[1])]
        self.n_channels = self._data.shape[1]
        logger.info(f"数据加载完成: {self._data.shape[0]} 采样点, {self.n_channels} 通道")

    def compute_full(self, channels=None, max_workers=None):
        """
        全局小波变换：一次性处理整个信号。

        Parameters
        ----------
        channels : list of int or str, optional
            要计算的通道（索引或名称），None 表示全部。
        max_workers : int, optional
            并行进程数。

        Returns
        -------
        WaveletResult
        """
        if self._data is None:
            raise RuntimeError("请先调用 load_data() 加载数据")
        ch_indices = self._resolve_channels(channels)
        # 多通道并行计算
        results = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_ch = {
                executor.submit(_cwt_channel,
                                self._data[:, idx],
                                self._time,
                                None,
                                self.fs,
                                self.wavelet_name,
                                self.freqs): idx
                for idx in ch_indices
            }
            for future in as_completed(future_to_ch):
                idx = future_to_ch[future]
                coeff, power, phase, coi_mask = future.result()
                results[idx] = (coeff, power, phase, coi_mask)

        # 按通道顺序组装
        coeffs, powers, phases, masks = [], [], [], []
        for idx in sorted(results.keys()):
            c, p, ph, m = results[idx]
            coeffs.append(c)
            powers.append(p)
            phases.append(ph)
            masks.append(m)
        return WaveletResult(
            time=self._time,
            freqs=self.freqs,
            coeffs=np.stack(coeffs, axis=0),
            powers=np.stack(powers, axis=0),
            phases=np.stack(phases, axis=0),
            coi_masks=np.stack(masks, axis=0),
            channel_names=[self._channel_names[i] for i in sorted(results.keys())],
            config=self.config
        )

    def compute_windowed(self, channels=None, max_workers=None,
                         window_seconds=None, overlap=None):
        """
        分窗小波变换：将长信号切分为重叠窗口，分别计算后拼接。
        采用“去重叠边缘”策略避免伪迹。

        Parameters
        ----------
        channels : list, optional
        max_workers : int, optional
        window_seconds : float, optional
            窗口长度（秒），默认使用配置。
        overlap : float, optional
            重叠比例 0~1，默认使用配置。

        Returns
        -------
        WaveletResult
        """
        if self._data is None:
            raise RuntimeError("请先调用 load_data() 加载数据")

        window_sec = window_seconds or self.window_seconds
        overlap_frac = overlap if overlap is not None else self.overlap
        window_len = int(window_sec * self.fs)
        step = int(window_len * (1 - overlap_frac))
        if step <= 0:
            raise ValueError("重叠比例过高，步长必须 > 0")

        ch_indices = self._resolve_channels(channels)

        # 生成所有任务 (通道索引, 窗口起始点)
        tasks = []
        for ch_idx in ch_indices:
            starts = list(range(0, self._data.shape[0] - window_len + 1, step))
            if not starts or starts[-1] + window_len < self._data.shape[0]:
                starts.append(self._data.shape[0] - window_len)  # 确保覆盖末尾
            for start in starts:
                tasks.append((ch_idx, start, start + window_len))

        # 并行处理所有窗口
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_cwt_channel,
                                self._data[task[1]:task[2], task[0]],
                                self._time[task[1]:task[2]],
                                None,
                                self.fs,
                                self.wavelet_name,
                                self.freqs): task
                for task in tasks
            }
            # 按通道收集窗口结果
            win_results = {ch: [] for ch in ch_indices}
            for future in as_completed(futures):
                task = futures[future]
                ch_idx, start, end = task
                coeff, power, phase, coi_mask = future.result()
                win_results[ch_idx].append((start, end, coeff, power, phase, coi_mask))

        # ---------- 拼接各通道窗口（去重叠边缘） ----------
        # 重叠长度（点数）
        overlap_len = window_len - step
        # 切割边长度：一半重叠，窗口两侧各去掉 overlap_len/2
        cut = overlap_len // 2

        final_coeff, final_power, final_phase, final_mask = [], [], [], []
        for ch in ch_indices:
            wins = sorted(win_results[ch], key=lambda x: x[0])
            coeff_cuts, power_cuts, phase_cuts, mask_cuts = [], [], [], []

            for i, (start, end, coeff_win, power_win, phase_win, mask_win) in enumerate(wins):
                # 窗口系数 shape: (n_freqs, n_times_in_win)
                if i == 0 and i == len(wins) - 1:
                    # 仅一个窗口，保留全部
                    coeff_cuts.append(coeff_win)
                    power_cuts.append(power_win)
                    phase_cuts.append(phase_win)
                    mask_cuts.append(mask_win)
                elif i == 0:
                    # 第一个窗口：只切右侧
                    coeff_cuts.append(coeff_win[..., :window_len - cut])
                    power_cuts.append(power_win[..., :window_len - cut])
                    phase_cuts.append(phase_win[..., :window_len - cut])
                    mask_cuts.append(mask_win[..., :window_len - cut])
                elif i == len(wins) - 1:
                    # 最后一个窗口：只切左侧
                    coeff_cuts.append(coeff_win[..., cut:])
                    power_cuts.append(power_win[..., cut:])
                    phase_cuts.append(phase_win[..., cut:])
                    mask_cuts.append(mask_win[..., cut:])
                else:
                    # 中间窗口：切两侧
                    coeff_cuts.append(coeff_win[..., cut:window_len - cut])
                    power_cuts.append(power_win[..., cut:window_len - cut])
                    phase_cuts.append(phase_win[..., cut:window_len - cut])
                    mask_cuts.append(mask_win[..., cut:window_len - cut])

            # 沿时间轴拼接
            coeff_cat = np.concatenate(coeff_cuts, axis=-1)
            power_cat = np.concatenate(power_cuts, axis=-1)
            phase_cat = np.concatenate(phase_cuts, axis=-1)
            mask_cat = np.concatenate(mask_cuts, axis=-1)

            final_coeff.append(coeff_cat)
            final_power.append(power_cat)
            final_phase.append(phase_cat)
            final_mask.append(mask_cat)

        # 构造输出时间轴（拼接后长度可能与原始长度略有差异，但应接近）
        recon_time = self._time[:final_coeff[0].shape[-1]]  # 简单截取，可根据偏移微调
        return WaveletResult(
            time=recon_time,
            freqs=self.freqs,
            coeffs=np.stack(final_coeff, axis=0),
            powers=np.stack(final_power, axis=0),
            phases=np.stack(final_phase, axis=0),
            coi_masks=np.stack(final_mask, axis=0),
            channel_names=[self._channel_names[i] for i in ch_indices],
            config=self.config
        )

    def run_and_save(self, hdf5_input, output_path, prefix='result',
                     mode='full', channels=None, max_workers=4, **kwargs):
        """一站式：加载数据 → 计算 → 保存 HDF5。"""
        # 提取 load_data 需要的参数，避免干扰其他 kwargs
        dataset_name = kwargs.pop('dataset_name', 'data')
        time_name = kwargs.pop('time_name', 'time')
        self.load_data(hdf5_input, dataset_name=dataset_name, time_name=time_name)
        if mode == 'full':
            result = self.compute_full(channels, max_workers)
        elif mode == 'windowed':
            result = self.compute_windowed(channels, max_workers, **kwargs)
        else:
            raise ValueError("mode 必须为 'full' 或 'windowed'")
        return result.to_hdf5(output_path, prefix)

    def _resolve_channels(self, channels):
        """将外部通道标识（索引/名称）转换为索引列表"""
        if channels is None:
            return list(range(self.n_channels))
        if isinstance(channels[0], str):
            name2idx = {name: i for i, name in enumerate(self._channel_names)}
            return [name2idx[ch] for ch in channels]
        return channels


# ================== 模块级计算函数（多进程序列化需要） ==================
def _cwt_channel(data, time, scales, fs, wavelet_name, freqs):
    dt = 1 / fs
    wavelet = Morlet()
    W, sj, freqs_out, coi_time, _, _ = cwt(data, dt, wavelet, freqs=freqs)
    power = np.abs(W) ** 2
    phase = np.angle(W)
    # 构建COI掩码
    N = len(data)
    coi_mask = np.zeros_like(power, dtype=bool)
    for j, s in enumerate(sj):
        edge = s * np.sqrt(2) / fs
        edge_idx = int(edge / dt)
        if edge_idx >= N // 2:
            coi_mask[j, :] = True
        else:
            coi_mask[j, :edge_idx] = True
            coi_mask[j, N - edge_idx:] = True
    return W, power, phase, coi_mask


# ================== 小波结果容器 ==================
class WaveletResult:
    """小波计算结果容器，支持 HDF5 保存。"""

    def __init__(self, time, freqs, coeffs, powers, phases, coi_masks, channel_names, config):
        self.time = time
        self.freqs = freqs
        self.coeffs = coeffs      # (ch, freq, time) complex
        self.powers = powers      # (ch, freq, time) float
        self.phases = phases      # (ch, freq, time) float
        self.coi_masks = coi_masks  # (ch, freq, time) bool
        self.channel_names = channel_names
        self.config = config      # ConfigManager 实例

    def to_hdf5(self, output_path, prefix='wavelet_result'):
        """
        保存为 HDF5 文件，自动创建目录。

        Parameters
        ----------
        output_path : str
            目录路径
        prefix : str
            文件名前缀
        """
        import os
        os.makedirs(output_path, exist_ok=True)
        filename = f"{prefix}_{self.config.get('小波类型', 'cwt')}.h5"
        fullpath = os.path.join(output_path, filename)
        with h5py.File(fullpath, 'w') as f:
            f.create_dataset('time', data=self.time)
            f.create_dataset('freqs', data=self.freqs)
            f.create_dataset('coeffs', data=self.coeffs, compression='gzip')
            f.create_dataset('powers', data=self.powers, compression='gzip')
            f.create_dataset('phases', data=self.phases, compression='gzip')
            f.create_dataset('coi_masks', data=self.coi_masks, compression='gzip')
            f.attrs['channel_names'] = self.channel_names
            # 保存关键配置属性到 HDF5 根属性
            for key in self.config.keys():
                try:
                    val = self.config.get(key)
                    # 只保存简单类型（int/float/str/bool）
                    if isinstance(val, (int, float, str, bool, np.generic)):
                        f.attrs[key] = val
                    elif isinstance(val, list) and all(isinstance(v, (int,float,str,bool)) for v in val):
                        f.attrs[key] = val
                except Exception:
                    pass
        logger.info(f"小波结果已保存至: {fullpath}")
        return fullpath