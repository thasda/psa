"""功率谱密度计算服务
- 加载预处理后的 HDF5 文件（应包含 TSData 或 data 以及 times）
- 使用 ConfigManager 获取参数（采样率、FFT 窗口长度）
- 对每个通道分别计算功率谱密度（Welch 法，50% 重叠，汉明窗）
- 保存结果为 HDF5 文件，包含频率轴和 PSD 数据
"""

import numpy as np
import h5py
from scipy import signal
from pathlib import Path
from typing import Optional, Union


class PSDCalculator:
    """
    功率谱密度计算器
    """

    def __init__(self, hdf5_path: Union[str, Path], config_manager):
        """
        参数:
            hdf5_path: 预处理后的 HDF5 文件路径（应包含 TSData 或 data 以及 times）
            config_manager: ConfigManager 实例，用于获取采样率、FFT窗口长度等参数
        """
        self.hdf5_path = Path(hdf5_path)
        self.cfg = config_manager

        # 从配置中提取必要参数
        self.sample_rate = self.cfg.get("采样率 (Hz)")
        self.nperseg = self.cfg.get("FFT窗口长度")

        if self.sample_rate is None:
            raise ValueError("配置中缺少采样率 (Hz)")
        if self.nperseg is None:
            raise ValueError("配置中缺少 FFT窗口长度")

        # 元数据占位
        self.n_samples = None
        self.n_channels = None
        self.channel_names = None
        self.units = None
        self.dataset_name = None   # 记录是 'TSData' 还是 'data'

        # 读取元数据（自动识别数据集名称）
        self._read_metadata()

    def _read_metadata(self):
        """根据 HDF5 文件中存在的数据集，自动选择读取方法"""
        with h5py.File(self.hdf5_path, 'r') as f:
            if 'TSData' in f:
                self.dataset_name = 'TSData'
                self._read_metadata_tsdata(f)
            elif 'data' in f:
                self.dataset_name = 'data'
                self._read_metadata_data(f)
            else:
                raise KeyError(f"HDF5 文件中未找到 'TSData' 或 'data' 数据集: {self.hdf5_path}")

    def _read_metadata_tsdata(self, f):
        """处理 TSData 数据集的元数据"""
        ds = f['TSData']
        self.n_samples, self.n_channels = ds.shape
        # 采样率优先使用 ConfigManager 中的值，其次从属性读取
        if self.sample_rate is None:
            self.sample_rate = f.attrs.get('sample_rate', ds.attrs.get('sample_rate', None))
        # 通道名称
        raw_names = f.attrs.get('channel_names', None)
        if raw_names is not None:
            self.channel_names = [n.decode() if isinstance(n, bytes) else n for n in raw_names]
        else:
            self.channel_names = [f'ch{i}' for i in range(self.n_channels)]
        self.units = f.attrs.get('units', 'unknown')
        # 可选打印第一个通道信息
        ex_data = ds[:, 0]
        print(f"[TSData] {self.channel_names[0]} ({self.units}): "
              f"均值={ex_data.mean():.2e}, 标准差={ex_data.std():.2e}")

    def _read_metadata_data(self, f):
        """处理 data 数据集的元数据（拼接脚本生成的格式）"""
        ds = f['data']
        self.n_samples, self.n_channels = ds.shape
        if self.sample_rate is None:
            self.sample_rate = f.attrs.get('sample_rate', None)
        raw_names = f.attrs.get('channel_names', None)
        if raw_names is not None:
            self.channel_names = [n.decode() if isinstance(n, bytes) else n for n in raw_names]
        else:
            self.channel_names = [f'ch{i}' for i in range(self.n_channels)]
        self.units = f.attrs.get('units', 'unknown')
        ex_data = ds[:, 0]
        print(f"[data] {self.channel_names[0]} ({self.units}): "
              f"均值={ex_data.mean():.2e}, 标准差={ex_data.std():.2e}")

    def compute_psd(self, output_path: Optional[Union[str, Path]] = None,
                    chunk_size: Optional[int] = None) -> dict:
        """
        计算功率谱密度。

        参数:
            output_path: 输出 HDF5 文件路径。如果为 None，则自动生成（在原文件名后加 _psd）。
            chunk_size: 分块大小（样本数）。如果为 None，则一次性加载全部数据；
                        如果指定，则分块读取并累积 PSD（Bartlett 法，无重叠，与 Welch 略有差异）。
                        注意：为精确实现 Welch 法（50%重叠），建议一次性加载全部数据。

        返回:
            dict: 包含频率数组和 PSD 数组的字典（方便直接使用）。
        """
        if output_path is None:
            output_path = self.hdf5_path.with_name(self.hdf5_path.stem + "_psd.h5")

        # 如果指定了 chunk_size，则采用分块处理（简单平均，无重叠，即 Bartlett 法）
        if chunk_size is not None:
            return self._compute_psd_chunked(output_path, chunk_size)
        else:
            return self._compute_psd_full(output_path)

    def _compute_psd_full(self, output_path: Path) -> dict:
        """一次性加载全部数据，使用 scipy.signal.welch 计算 PSD（Welch 法，50%重叠）"""
        print(f"正在加载数据: {self.hdf5_path}")
        with h5py.File(self.hdf5_path, 'r') as f:
            # 使用自动识别的数据集名称
            data = f[self.dataset_name][:]  # shape: (n_samples, n_channels)
            times = f['times'][:] if 'times' in f else None

        print(f"数据形状: {data.shape}, 采样率: {self.sample_rate} Hz, nperseg: {self.nperseg}")

        # 计算每个通道的 PSD
        freqs = None
        psd_list = []
        for ch in range(data.shape[1]):
            f, Pxx = signal.welch(
                data[:, ch],
                fs=self.sample_rate,
                window='hamming',
                nperseg=self.nperseg,
                noverlap=self.nperseg // 2,
                scaling='density',
                axis=0
            )
            if freqs is None:
                freqs = f
            psd_list.append(Pxx)

        psd_array = np.array(psd_list)  # shape: (n_channels, n_freqs)

        # 保存为 HDF5
        self._save_results(output_path, freqs, psd_array, times)

        return {'freqs': freqs, 'psd': psd_array}

    def _compute_psd_chunked(self, output_path: Path, chunk_size: int) -> dict:
        """
        分块读取数据，每块分别计算 PSD，然后平均（Bartlett 法，无重叠）。
        注意：此方法不实现 50% 重叠，仅作为内存不足时的备选。
        """
        print(f"使用分块处理，块大小: {chunk_size} 样本")
        n_samples = self.n_samples
        n_channels = self.n_channels

        psd_sum = None
        total_blocks = 0
        freqs = None

        with h5py.File(self.hdf5_path, 'r') as f:
            ds = f[self.dataset_name]
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                block = ds[start:end, :]
                print(f"处理块 {start}-{end}")

                # 每个块内部仍使用 Welch 方法（有重叠）
                for ch in range(n_channels):
                    f_ch, Pxx = signal.welch(
                        block[:, ch],
                        fs=self.sample_rate,
                        window='hamming',
                        nperseg=self.nperseg,
                        noverlap=self.nperseg // 2,
                        scaling='density',
                        axis=0
                    )
                    if freqs is None:
                        freqs = f_ch
                        psd_sum = np.zeros((n_channels, len(freqs)))

                    psd_sum[ch, :] += Pxx

                total_blocks += 1

        psd_avg = psd_sum / total_blocks

        # 保存结果
        self._save_results(output_path, freqs, psd_avg)

        return {'freqs': freqs, 'psd': psd_avg}

    def _save_results(self, output_path: Path, freqs: np.ndarray, psd: np.ndarray,
                      times: Optional[np.ndarray] = None):
        """将 PSD 结果保存为 HDF5 文件"""
        print(f"正在保存结果至: {output_path}")
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('freqs', data=freqs, dtype='float32')
            f.create_dataset('psd', data=psd, dtype='float32', compression='gzip')
            if times is not None:
                f.create_dataset('times', data=times)
            # 保存元数据
            f.attrs['sample_rate'] = self.sample_rate
            f.attrs['nperseg'] = self.nperseg
            f.attrs['noverlap'] = self.nperseg // 2
            f.attrs['window'] = 'hamming'
            f.attrs['n_channels'] = self.n_channels
            f.attrs['n_samples'] = self.n_samples
        print("保存完成")