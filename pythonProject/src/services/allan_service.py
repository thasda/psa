# src/utils/allan_service.py
"""
艾伦方差计算器（支持全内存累积和快速计算 和 分块近似）
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Union, Tuple


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
        self.cfg = config_manager

        self.sample_rate = self.cfg.get("采样率 (Hz)")
        if self.sample_rate is None:
            raise ValueError("配置中缺少采样率 (Hz)")

        self._read_metadata()

    def _read_metadata(self):
        """读取样本数和通道数"""
        with h5py.File(self.hdf5_path, 'r') as f:
            self.n_samples, self.n_channels = f['TSData'].shape or f['data'].shape

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
    ) -> dict:
        """
        计算艾伦方差 sigma^2(tau)。

        参数:
            taus: 指定的 tau 值数组（单位：秒）。若为 None 则自动生成（几何分布）。
            max_tau_factor: 最大 tau 占数据总时长的比例（默认 0.1），用于自动生成 tau。
            output_path: 输出 HDF5 文件路径，若指定则保存结果。
            method: 计算方法，'fast'（全内存累积和，推荐）或 'chunked'（分块近似）。
            chunk_size: 仅在 method='chunked' 时有效，分块大小（样本数）。

        返回:
            dict: 包含 'taus' (秒) 和 'allan_var' (形状 [n_channels, len(taus)]) 的字典。
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
            result = self._compute_allan_fast(taus)
        else:
            result = self._compute_allan_chunked(taus, chunk_size)

        if output_path is not None:
            self._save_results(output_path, result['taus'], result['allan_var'])

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

    def _compute_allan_fast(self, taus: np.ndarray) -> dict:
        """
        使用累积和算法快速计算艾伦方差（全内存）。
        算法复杂度 O(N * len(taus))，但实际通过累积和避免内层循环。
        """
        print("使用 fast 模式（全内存累积和）...")
        with h5py.File(self.hdf5_path, 'r') as f:
            data = f['TSData'][:]  # (n_samples, n_channels)

        n_samples, n_channels = data.shape
        fs = self.sample_rate
        allan_var = np.full((n_channels, len(taus)), np.nan, dtype=np.float64)

        for ch in range(n_channels):
            x = data[:, ch].astype(np.float64)
            # 检查并处理 NaN
            if np.any(np.isnan(x)):
                print(f"通道 {ch} 包含 NaN，将使用 nanmean，结果可能不准确")
            # 计算累积和（用于快速求区间均值）
            cumsum = np.cumsum(np.nan_to_num(x, nan=0.0))  # NaN 视为 0，但需谨慎
            # 更稳健：先填充 NaN 为 0，但会引入偏差。这里建议用户先清洗数据。
            # 我们仍使用原始数据，在均值计算时用 nanmean，但累积和方式不支持忽略 NaN，
            # 因此若数据有 NaN，此方法结果不可靠。我们会在文档中说明。
            # 实际建议用户预处理数据确保无 NaN。
            # 为简化，这里假设数据无 NaN，若有则回退到逐块循环。
            if np.any(np.isnan(x)):
                print(f"警告：通道 {ch} 包含 NaN，fast 模式可能出错，将回退到循环计算")
                allan_var[ch, :] = self._compute_allan_slow_loop(x, taus, fs)
                continue

            for i, tau in enumerate(taus):
                m = int(round(tau * fs))
                if m == 0:
                    continue
                if m >= n_samples:
                    continue
                # 可以使用的完整块数
                n_groups = n_samples // m
                if n_groups < 2:
                    continue
                # 截断到完整块
                n_used = n_groups * m
                # 计算每个块的均值：利用累积和
                # cumsum 索引：cumsum[km] - cumsum[(k-1)m]
                group_means = (cumsum[m-1:n_used:m] - np.append(0, cumsum[m-1:n_used-m:m])) / m
                # 相邻块均值的差
                diff = group_means[1:] - group_means[:-1]
                var = 0.5 * np.mean(diff ** 2)
                allan_var[ch, i] = var

        return {'taus': taus, 'allan_var': allan_var}

    def _compute_allan_slow_loop(self, x: np.ndarray, taus: np.ndarray, fs: float) -> np.ndarray:
        """回退的逐 tau 循环（处理 NaN 或小数据）"""
        result = np.full(len(taus), np.nan)
        for i, tau in enumerate(taus):
            m = int(round(tau * fs))
            if m <= 0 or m >= len(x):
                continue
            n_groups = len(x) // m
            if n_groups < 2:
                continue
            x_trunc = x[:n_groups * m]
            groups = x_trunc.reshape(n_groups, m)
            group_means = np.nanmean(groups, axis=1)
            if np.any(np.isnan(group_means)):
                continue
            diff = np.diff(group_means)
            var = 0.5 * np.nanmean(diff ** 2)
            result[i] = var
        return result

    def _compute_allan_chunked(self, taus: np.ndarray, chunk_size: int) -> dict:
        """
        分块近似计算艾伦方差。
        将数据分为不重叠的块，每块独立计算艾伦方差，然后按 tau 加权平均（权重 = 该块中可用差分次数）。
        注意：此方法破坏长程相关性，结果仅作近似。
        """
        print(f"使用 chunked 模式，块大小: {chunk_size} 样本")
        n_samples = self.n_samples
        n_channels = self.n_channels
        fs = self.sample_rate

        n_taus = len(taus)
        # 加权累加器：分子（加权方差和）和分母（权重）
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
                    # 去除 NaN 对块的影响：如果块内大量 NaN，可跳过，这里简单处理
                    if np.all(np.isnan(x)):
                        continue
                    for i, tau in enumerate(taus):
                        m = int(round(tau * fs))
                        if m <= 0 or m >= block_len:
                            continue
                        n_groups = block_len // m
                        if n_groups < 2:
                            continue
                        x_trunc = x[:n_groups * m]
                        # 处理 NaN：将 NaN 设为 0 会影响结果，但为了不丢失整个块，使用 nanmean
                        # 更严格：若任何组内全为 NaN，则跳过
                        groups = x_trunc.reshape(n_groups, m)
                        group_means = np.nanmean(groups, axis=1)
                        if np.any(np.isnan(group_means)):
                            continue
                        diff = np.diff(group_means)
                        var = 0.5 * np.nanmean(diff ** 2)
                        # 权重 = 差分数量 (n_groups - 1)
                        w = n_groups - 1
                        sum_weighted_var[ch, i] += var * w
                        weight_sum[ch, i] += w

        # 计算加权平均方差
        allan_var = np.full((n_channels, n_taus), np.nan)
        for ch in range(n_channels):
            for i in range(n_taus):
                if weight_sum[ch, i] > 0:
                    allan_var[ch, i] = sum_weighted_var[ch, i] / weight_sum[ch, i]

        return {'taus': taus, 'allan_var': allan_var}

    def _save_results(self, output_path: Union[str, Path], taus: np.ndarray, allan_var: np.ndarray):
        """保存艾伦方差结果到 HDF5"""
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('taus', data=taus, dtype='float32')
            f.create_dataset('allan_variance', data=allan_var, dtype='float32', compression='gzip')
            f.attrs['sample_rate'] = self.sample_rate
            f.attrs['n_channels'] = self.n_channels
            f.attrs['n_samples'] = self.n_samples
            f.attrs['method'] = 'fast' if allan_var.shape[0] == self.n_channels else 'chunked'
        print(f"艾伦方差结果已保存至: {output_path}")