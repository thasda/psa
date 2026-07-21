# src/utils/allan_service.py
"""
艾伦方差计算器（支持全内存快速计算和分块近似）
参照互功率谱类的设计，支持从 HDF5 读取数据、从配置管理器获取采样率。
"""

import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, Tuple, List, Dict


class AllanDeviationCalculator:
    """
    艾伦方差计算器。

    提供两种计算模式：
        - 'fast': 全内存累积和算法，速度快，适合内存能容纳全部数据的场景。
        - 'chunked': 分块近似算法，内存友好，适合超大文件（结果略异于标准定义）。

    标准艾伦方差公式（对等间隔时间序列）：
        sigma^2(tau) = 0.5 * E[(y_{k+1} - y_k)^2],
        其中 y_k 是长度为 tau 的块内平均值。
    """

    def __init__(self, hdf5_path: Union[str, Path], config_manager):
        """
        参数:
            hdf5_path: 预处理后的 HDF5 文件路径（应包含 TSData 和 times）
            config_manager: ConfigManager 实例，用于获取采样率等参数
        """
        self.hdf5_path = Path(hdf5_path)
        self.cfg_mgr = config_manager

        # 获取采样率（与 preprocess.py 保持完全一致）
        cfg = config_manager.get_config_dict() if hasattr(config_manager, 'get_config_dict') else {}
        self.sample_rate = cfg.get("采样率(Hz)") or cfg.get("采样率 (Hz)")
        if self.sample_rate is None:
            # 若配置中无，尝试从 HDF5 属性读取
            with h5py.File(self.hdf5_path, 'r') as f:
                self.sample_rate = f.attrs.get('sample_rate')
            if self.sample_rate is None:
                raise ValueError("配置中缺少采样率，且 HDF5 属性中也未找到 sample_rate")

        self.sample_rate = float(self.sample_rate)
        self._read_metadata_and_channels()

    def _read_metadata_and_channels(self):
        """读取样本数、通道数、通道名称以及原始属性"""
        with h5py.File(self.hdf5_path, 'r') as f:
            # 查找数据集
            if 'TSData' in f:
                ds = f['TSData']
            elif 'data' in f:
                ds = f['data']
            else:
                raise KeyError("未找到 TSData 或 data 数据集")
            self.n_samples, self.n_channels = ds.shape
            # 读取通道名称
            self.channel_names = self._parse_channel_names(f)
            # 复制所有属性（用于保存时保留）
            self.attrs = dict(f.attrs)

    def _parse_channel_names(self, h5_file) -> List[str]:
        """从 HDF5 文件对象中解析通道名称列表"""
        names = None
        if 'channel_names' in h5_file.attrs:
            ch_attr = h5_file.attrs['channel_names']
            if isinstance(ch_attr, (list, tuple, np.ndarray)):
                names = [c.decode() if isinstance(c, bytes) else str(c) for c in ch_attr]
            else:
                names = [ch_attr.decode() if isinstance(ch_attr, bytes) else str(ch_attr)]
        elif 'channel_names' in h5_file:
            ch_data = h5_file['channel_names'][:]
            if ch_data.ndim == 1:
                names = [c.decode() if isinstance(c, bytes) else str(c) for c in ch_data]
            else:
                names = [str(c) for c in ch_data.flatten()]
        if names is None:
            names = [f'ch{i}' for i in range(self.n_channels)]
        # 确保长度匹配
        if len(names) != self.n_channels:
            if len(names) < self.n_channels:
                names += [f'ch{i}' for i in range(len(names), self.n_channels)]
            else:
                names = names[:self.n_channels]
        return names

    # ----------------------------------------------------------------------
    # 公共接口
    # ----------------------------------------------------------------------
    def compute_allan_variance(
        self,
        taus: Optional[np.ndarray] = None,
        max_tau_factor: float = 0.1,
        output_path: Optional[Union[str, Path]] = None,
        method: str = 'fast',
        chunk_size: Optional[int] = None,
    ) -> Dict:
        """
        计算艾伦方差 sigma^2(tau)。

        参数:
            taus: 指定的 tau 值数组（单位：秒）。若为 None 则自动生成（几何分布）。
            max_tau_factor: 最大 tau 占数据总时长的比例（默认 0.1），用于自动生成 tau。
            output_path: 输出 HDF5 文件路径，若指定则保存结果。
            method: 计算方法，'fast'（全内存累积和，推荐）或 'chunked'（分块近似）。
            chunk_size: 仅在 method='chunked' 时有效，分块大小（样本数）。

        返回:
            dict: 包含 'taus' (秒), 'allan_var' (形状 [n_channels, len(taus)]),
                  'channel_names' 等信息的字典。
        """
        # 参数验证
        if method not in ('fast', 'chunked'):
            raise ValueError("method 必须是 'fast' 或 'chunked'")
        if not (0 < max_tau_factor <= 1):
            raise ValueError("max_tau_factor 必须在 (0, 1] 范围内")
        if method == 'chunked' and (chunk_size is None or chunk_size <= 0):
            raise ValueError("chunked 模式必须指定 chunk_size > 0")

        total_time = self.n_samples / self.sample_rate
        # 生成或验证 taus
        taus = self._prepare_taus(taus, max_tau_factor, total_time)

        if method == 'fast':
            allan_var = self._compute_allan_fast(taus)
        else:
            allan_var = self._compute_allan_chunked(taus, chunk_size)

        result = {
            'taus': taus,
            'allan_var': allan_var,
            'channel_names': self.channel_names,
            'sample_rate': self.sample_rate
        }

        if output_path is not None:
            self._save_results(output_path, result)

        return result

    # ----------------------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------------------
    def _prepare_taus(
        self,
        taus: Optional[np.ndarray],
        max_tau_factor: float,
        total_time: float
    ) -> np.ndarray:
        """生成或验证 tau 数组，返回唯一且排序后的正 tau 值（秒）。"""
        if taus is None:
            max_tau = total_time * max_tau_factor
            min_tau = 1.0 / self.sample_rate
            if max_tau < min_tau:
                max_tau = min_tau
            # 几何分布生成 50 个 tau，然后取整到采样间隔的整数倍
            taus = np.geomspace(min_tau, max_tau, num=50)
            taus = np.unique(np.round(taus * self.sample_rate) / self.sample_rate)
        else:
            taus = np.asarray(taus).flatten()
            if np.any(taus <= 0):
                raise ValueError("所有 tau 必须为正数")
            taus = np.unique(taus)
        return taus

    def _compute_allan_fast(self, taus: np.ndarray) -> np.ndarray:
        """
        使用累积和算法快速计算艾伦方差（全内存）。
        假设数据无 NaN（预处理已去趋势和校准）。
        """
        print("使用 fast 模式（全内存累积和）...")
        with h5py.File(self.hdf5_path, 'r') as f:
            data = f['TSData'][:]  # (n_samples, n_channels)

        n_samples, n_channels = data.shape
        fs = self.sample_rate
        allan_var = np.full((n_channels, len(taus)), np.nan, dtype=np.float64)

        for ch in range(n_channels):
            x = data[:, ch].astype(np.float64)
            # 累积和（用于快速求块均值）
            cumsum = np.cumsum(x)
            for i, tau in enumerate(taus):
                m = int(round(tau * fs))
                if m == 0 or m >= n_samples:
                    continue
                n_groups = n_samples // m
                if n_groups < 2:
                    continue
                n_used = n_groups * m
                # 计算每个块的均值
                # cumsum[km] - cumsum[(k-1)m]
                group_means = (cumsum[m-1:n_used:m] - np.append(0, cumsum[m-1:n_used-m:m])) / m
                diff = group_means[1:] - group_means[:-1]
                var = 0.5 * np.mean(diff ** 2)
                allan_var[ch, i] = var

        return allan_var

    def _compute_allan_chunked(self, taus: np.ndarray, chunk_size: int) -> np.ndarray:
        """
        分块近似计算艾伦方差。
        将数据分为不重叠的块，每块独立计算艾伦方差，然后按 tau 加权平均。
        注意：此方法破坏长程相关性，结果仅作近似。
        """
        print(f"使用 chunked 模式，块大小: {chunk_size} 样本")
        n_samples = self.n_samples
        n_channels = self.n_channels
        fs = self.sample_rate
        n_taus = len(taus)

        sum_weighted_var = np.zeros((n_channels, n_taus))
        weight_sum = np.zeros((n_channels, n_taus))

        with h5py.File(self.hdf5_path, 'r') as f:
            ds = f['TSData']
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                block = ds[start:end, :]  # (block_size, n_channels)
                block_len = block.shape[0]
                if block_len < 2:
                    continue

                for ch in range(n_channels):
                    x = block[:, ch].astype(np.float64)
                    for i, tau in enumerate(taus):
                        m = int(round(tau * fs))
                        if m <= 0 or m >= block_len:
                            continue
                        n_groups = block_len // m
                        if n_groups < 2:
                            continue
                        x_trunc = x[:n_groups * m]
                        groups = x_trunc.reshape(n_groups, m)
                        group_means = np.mean(groups, axis=1)
                        diff = np.diff(group_means)
                        var = 0.5 * np.mean(diff ** 2)
                        w = n_groups - 1
                        sum_weighted_var[ch, i] += var * w
                        weight_sum[ch, i] += w

        allan_var = np.full((n_channels, n_taus), np.nan)
        for ch in range(n_channels):
            for i in range(n_taus):
                if weight_sum[ch, i] > 0:
                    allan_var[ch, i] = sum_weighted_var[ch, i] / weight_sum[ch, i]

        return allan_var

    def _save_results(self, output_path: Union[str, Path], result: Dict):
        """
        保存艾伦方差结果到 HDF5，格式参照互功率谱分析结果。
        """
        with h5py.File(output_path, 'w') as f:
            # 写入全局属性
            f.attrs['analysis_type'] = 'allan_variance'
            f.attrs['sample_rate'] = self.sample_rate
            f.attrs['n_channels'] = self.n_channels
            f.attrs['n_samples'] = self.n_samples
            f.attrs['date'] = datetime.now().isoformat()
            f.attrs['source_file'] = str(self.hdf5_path)
            # 复制原始属性
            for k, v in self.attrs.items():
                if k not in f.attrs:
                    f.attrs[k] = v

            # 保存数据集
            f.create_dataset('taus', data=result['taus'], dtype='float32')
            f.create_dataset('allan_variance', data=result['allan_var'], dtype='float32', compression='gzip')
            # 保存通道名称（作为字符串数组）
            ch_names = np.array(result['channel_names'], dtype='S')
            f.create_dataset('channel_names', data=ch_names)

        print(f"艾伦方差结果已保存至: {output_path}")