# src/services/pdf_service.py
"""
功率谱概率密度计算器
从预处理时间序列计算多段功率谱，并统计每个频率点的概率密度分布。
支持分块处理、多通道、自定义窗口参数。
"""
import warnings

import numpy as np
import h5py
from scipy import signal
from pathlib import Path
from typing import Optional, Union, List


class PSDProbabilityDensity:
    """
    功率谱概率密度计算器
    输入：预处理后的HDF5文件（包含 TSData 和 times）
    输出：多段功率谱三维数组，以及每个频率点的统计分布（直方图、分位数等）
    """

    def __init__(self, hdf5_path: Union[str, Path], config_manager=None):
        """
        参数:
            hdf5_path: 预处理后的HDF5文件路径（应包含 TSData 和 times）
            config_manager: 可选，ConfigManager实例，用于获取采样率、FFT窗口长度等
        """
        self.hdf5_path = Path(hdf5_path)
        self.cfg = config_manager

        # 从配置中获取采样率（如果提供）
        if self.cfg is not None:
            self.sample_rate = self.cfg.get("采样率 (Hz)")
            self.default_nperseg = self.cfg.get("FFT窗口长度")
        else:
            self.sample_rate = None
            self.default_nperseg = None

        self._read_metadata()

    def _read_metadata(self):
        """读取样本数和通道数"""
        with h5py.File(self.hdf5_path, 'r') as f:
            self.n_samples, self.n_channels = f['TSData'].shape
            if self.sample_rate is None:
                self.sample_rate = f.attrs.get('sample_rate', None)
            if self.sample_rate is None:
                raise ValueError("无法确定采样率，请通过config_manager或文件属性提供")

    def compute_multitaper_psd(self, output_path: Union[str, Path],
                                nperseg: Optional[int] = None,
                                noverlap: Optional[int] = None,
                                window: str = 'hamming',
                                scaling: str = 'density',
                                detrend: str = 'constant',
                                return_onesided: bool = True,
                                chunk_size: Optional[int] = None) -> dict:
        """
        计算多段功率谱（使用 spectrogram），并保存结果。

        参数:
            output_path: 输出HDF5文件路径
            nperseg: 每段长度（样本数），若为None则使用配置中的FFT窗口长度
            noverlap: 重叠样本数，若为None则设为 nperseg//2
            window: 窗函数类型
            scaling: 'density' 或 'spectrum'
            detrend: 去趋势方式
            return_onesided: 是否返回单边谱（实信号）
            chunk_size: 分块大小（样本数），若指定则分块读取数据以节省内存

        返回:
            dict: 包含频率轴和功率谱数组的字典，同时结果已保存至文件
        """
        # 确定参数
        if nperseg is None:
            if self.default_nperseg is not None:
                nperseg = self.default_nperseg
            else:
                raise ValueError("必须指定 nperseg 或提供配置中的FFT窗口长度")
        if noverlap is None:
            noverlap = nperseg // 2

        print(f"参数: fs={self.sample_rate}, nperseg={nperseg}, noverlap={noverlap}")

        # 计算频率轴（用于保存）
        freqs = np.fft.rfftfreq(nperseg, d=1/self.sample_rate)

        # 分块或全内存处理
        if chunk_size is not None:
            return self._compute_chunked(output_path, freqs, nperseg, noverlap,
                                         window, scaling, detrend, return_onesided,
                                         chunk_size)
        else:
            # 全内存模式：一次性加载数据
            print("正在加载数据...")
            with h5py.File(self.hdf5_path, 'r') as f:
                data = f['TSData'][:]  # (n_samples, n_channels)

            # 对每个通道计算 spectrogram
            print("计算多段功率谱...")
            # 对第一个通道，获取时间轴长度
            f, t, Sxx_first = signal.spectrogram(
                data[:, 0],
                fs=self.sample_rate,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling=scaling,
                detrend=detrend,
                return_onesided=return_onesided,
                axis=0
            )
            n_freqs = len(f)
            n_times = len(t)
            n_channels = self.n_channels

            # 预分配数组 (n_times, n_channels, n_freqs)
            Sxx_all = np.zeros((n_times, n_channels, n_freqs), dtype=np.float32)
            Sxx_all[:, 0, :] = Sxx_first.T

            # 处理其余通道
            for ch in range(1, n_channels):
                f, t, Sxx = signal.spectrogram(
                    data[:, ch],
                    fs=self.sample_rate,
                    window=window,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    scaling=scaling,
                    detrend=detrend,
                    return_onesided=return_onesided,
                    axis=0
                )
                Sxx_all[:, ch, :] = Sxx.T

            # 保存多段PSD
            self._save_multitaper(output_path, freqs, t, Sxx_all, noverlap)

            return {'freqs': freqs, 'times': t, 'psd_segments': Sxx_all}

    def _compute_chunked(self, output_path, freqs, nperseg, noverlap,
                         window, scaling, detrend, return_onesided, chunk_size):
        """
        分块计算多段功率谱，并累积结果（平均、直方图等）。
        这里先实现分块累加，但不保存每段PSD（可节省内存），直接计算统计量。
        """
        raise NotImplementedError("分块计算暂未完全实现，建议使用全内存模式或先实现所需功能")

    def _save_multitaper(self, output_path, freqs, times, psd_segments, noverlap):
        """保存多段功率谱到HDF5"""
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('freqs', data=freqs, dtype='float32')
            f.create_dataset('times', data=times, dtype='float32')
            f.create_dataset('psd_segments', data=psd_segments, dtype='float32',
                             compression='gzip', chunks=True)
            f.attrs['sample_rate'] = self.sample_rate
            f.attrs['nperseg'] = (len(freqs) - 1) * 2
            f.attrs['noverlap'] = noverlap
            f.attrs['n_channels'] = self.n_channels
            f.attrs['n_samples'] = self.n_samples
            f.attrs['n_segments'] = psd_segments.shape[0]
        print(f"多段PSD结果已保存至: {output_path}")

    # src/services/pdf_service.py (修正版)

    import numpy as np
    import h5py
    from scipy import signal
    from pathlib import Path
    from typing import Optional, Union, List
    import warnings

    class PSDProbabilityDensity:
        """
        功率谱概率密度计算器 (修正版)
        """

        # ... (__init__, _read_metadata, compute_multitaper_psd, _compute_chunked,
        #      _save_multitaper, compute_quantiles 保持不变，此处省略以节省篇幅)

        def compute_pdf_statistics(self, psd_segments_h5: Union[str, Path],
                                   output_path: Union[str, Path],
                                   n_bins: int = 50,
                                   log_scale: bool = True,
                                   min_power: Optional[float] = None,
                                   max_power: Optional[float] = None):
            """
            从已保存的多段PSD文件中，计算每个频率点的功率直方图（概率密度估计）。

            参数:
                psd_segments_h5: 包含多段PSD的HDF5文件路径（由 compute_multitaper_psd 生成）
                output_path: 输出HDF5文件路径
                n_bins: 直方图分箱数
                log_scale: 是否在功率轴上使用对数坐标（推荐）
                min_power, max_power: 直方图的功率范围（若None则自动根据数据确定）
            """
            # 1. 读取数据
            with h5py.File(psd_segments_h5, 'r') as f:
                freqs = f['freqs'][:]
                psd_segments = f['psd_segments'][:]  # shape (n_times, n_channels, n_freqs)

            n_freqs = len(freqs)
            n_times, n_channels, _ = psd_segments.shape
            print(f"数据维度: 段数={n_times}, 通道数={n_channels}, 频率点数={n_freqs}")

            # 2. 确定功率范围 (bin_edges)
            if min_power is None or max_power is None:
                # 收集所有非NaN的功率值
                all_psd = psd_segments[~np.isnan(psd_segments)]
                if log_scale:
                    all_psd = all_psd[all_psd > 0]  # 对数坐标要求正值
                if len(all_psd) == 0:
                    raise ValueError("未找到有效的正功率值，无法自动确定功率范围。请手动指定 min_power 和 max_power。")

                if min_power is None:
                    min_power = np.percentile(all_psd, 1)
                if max_power is None:
                    max_power = np.percentile(all_psd, 99)

                # 防止范围过窄或无效
                if min_power <= 0 and log_scale:
                    # 若最小值为0或负，取最小正值的百分之一
                    positive_vals = all_psd[all_psd > 0]
                    if len(positive_vals) > 0:
                        min_power = np.percentile(positive_vals, 1)
                    else:
                        min_power = 1e-12
                if min_power == max_power:
                    min_power = min_power / 10
                    max_power = max_power * 10
                    warnings.warn(f"功率范围过窄，已自动扩展为 [{min_power:.3e}, {max_power:.3e}]")

            # 创建 bins
            if log_scale:
                if min_power <= 0:
                    raise ValueError(f"对数坐标下 min_power 必须 > 0，当前值为 {min_power}")
                bins = np.logspace(np.log10(min_power), np.log10(max_power), n_bins + 1)
            else:
                bins = np.linspace(min_power, max_power, n_bins + 1)

            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bin_widths = np.diff(bins)

            # 3. 计算直方图 (hist_counts)
            hist_counts = np.zeros((n_channels, n_freqs, n_bins), dtype=np.uint32)

            # 进度提示
            total_iters = n_channels * n_freqs
            current_iter = 0
            print("正在计算直方图...")
            for ch in range(n_channels):
                for i_f in range(n_freqs):
                    segment_powers = psd_segments[:, ch, i_f]
                    # 去除 NaN
                    segment_powers = segment_powers[~np.isnan(segment_powers)]
                    if len(segment_powers) > 0:
                        hist, _ = np.histogram(segment_powers, bins=bins)
                        hist_counts[ch, i_f, :] = hist

                    current_iter += 1
                    if current_iter % 10000 == 0:
                        print(f"进度: {current_iter}/{total_iters}")

            # 4. 归一化为概率密度 (每个通道/频率点独立)
            print("正在计算概率密度...")
            pdf = np.full(hist_counts.shape, np.nan, dtype=np.float32)

            for ch in range(n_channels):
                for i_f in range(n_freqs):
                    total = hist_counts[ch, i_f, :].sum()
                    if total == 0:
                        continue
                    # 密度 = 计数 / (总计数 * bin宽度)
                    pdf[ch, i_f, :] = hist_counts[ch, i_f, :] / (total * bin_widths)

            # 5. 保存结果
            print(f"保存结果至: {output_path}")
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('freqs', data=freqs, dtype='float32')
                f.create_dataset('bin_centers', data=bin_centers, dtype='float32')
                f.create_dataset('bin_edges', data=bins, dtype='float32')
                f.create_dataset('pdf', data=pdf, dtype='float32', compression='gzip')
                f.create_dataset('hist_counts', data=hist_counts, dtype='uint32', compression='gzip')
                f.attrs['n_bins'] = n_bins
                f.attrs['log_scale'] = log_scale
                f.attrs['min_power'] = min_power
                f.attrs['max_power'] = max_power
                f.attrs['n_channels'] = n_channels
                f.attrs['n_freqs'] = n_freqs
                f.attrs['n_segments'] = n_times

            print("完成！")

        # compute_quantiles 方法也需要类似修正（独立保存，不要放在通道循环内）
        def compute_quantiles(self, psd_segments_h5: Union[str, Path],
                              output_path: Union[str, Path],
                              quantiles: List[float] = [0.1, 0.5, 0.9]):
            """
            从多段PSD文件中计算每个频率点的分位数。
            """
            with h5py.File(psd_segments_h5, 'r') as f:
                freqs = f['freqs'][:]
                psd_segments = f['psd_segments'][:]  # (n_times, n_channels, n_freqs)

            n_freqs = len(freqs)
            n_times, n_channels, _ = psd_segments.shape
            quantiles = np.array(quantiles)
            n_quantiles = len(quantiles)

            quantile_values = np.zeros((n_channels, n_freqs, n_quantiles), dtype=np.float32)

            print("正在计算分位数...")
            for ch in range(n_channels):
                for i_f in range(n_freqs):
                    segment_powers = psd_segments[:, ch, i_f]
                    segment_powers = segment_powers[~np.isnan(segment_powers)]
                    if len(segment_powers) == 0:
                        quantile_values[ch, i_f, :] = np.nan
                    else:
                        quantile_values[ch, i_f, :] = np.percentile(segment_powers, quantiles * 100)

            # 保存 (只保存一次)
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('freqs', data=freqs, dtype='float32')
                f.create_dataset('quantiles', data=quantiles, dtype='float32')
                f.create_dataset('quantile_values', data=quantile_values, dtype='float32', compression='gzip')
                f.attrs['n_channels'] = n_channels
                f.attrs['n_freqs'] = n_freqs
                f.attrs['n_segments'] = n_times
            print(f"功率谱分位数结果已保存至: {output_path}")

    def compute_pdf_statistics(self, psd_segments_h5: Union[str, Path],
                               output_path: Union[str, Path],
                               n_bins: int = 50,
                               log_scale: bool = True,
                               min_power: Optional[float] = None,
                               max_power: Optional[float] = None):
        """
        从已保存的多段PSD文件中，计算每个频率点的功率直方图（概率密度估计）。

        参数:
            psd_segments_h5: 包含多段PSD的HDF5文件路径（由 compute_multitaper_psd 生成）
            output_path: 输出HDF5文件路径
            n_bins: 直方图分箱数
            log_scale: 是否在功率轴上使用对数坐标（推荐）
            min_power, max_power: 直方图的功率范围（若None则自动根据数据确定）
        """
        # 1. 读取数据
        with h5py.File(psd_segments_h5, 'r') as f:
            freqs = f['freqs'][:]
            psd_segments = f['psd_segments'][:]  # shape (n_times, n_channels, n_freqs)

        n_freqs = len(freqs)
        n_times, n_channels, _ = psd_segments.shape
        print(f"数据维度: 段数={n_times}, 通道数={n_channels}, 频率点数={n_freqs}")

        # 2. 确定功率范围 (bin_edges)
        if min_power is None or max_power is None:
            # 收集所有非NaN的功率值
            all_psd = psd_segments[~np.isnan(psd_segments)]
            if log_scale:
                all_psd = all_psd[all_psd > 0]  # 对数坐标要求正值
            if len(all_psd) == 0:
                raise ValueError("未找到有效的正功率值，无法自动确定功率范围。请手动指定 min_power 和 max_power。")

            if min_power is None:
                min_power = np.percentile(all_psd, 1)
            if max_power is None:
                max_power = np.percentile(all_psd, 99)

            # 防止范围过窄或无效
            if min_power <= 0 and log_scale:
                # 若最小值为0或负，取最小正值的百分之一
                positive_vals = all_psd[all_psd > 0]
                if len(positive_vals) > 0:
                    min_power = np.percentile(positive_vals, 1)
                else:
                    min_power = 1e-12
            if min_power == max_power:
                min_power = min_power / 10
                max_power = max_power * 10
                warnings.warn(f"功率范围过窄，已自动扩展为 [{min_power:.3e}, {max_power:.3e}]")

        # 创建 bins
        if log_scale:
            if min_power <= 0:
                raise ValueError(f"对数坐标下 min_power 必须 > 0，当前值为 {min_power}")
            bins = np.logspace(np.log10(min_power), np.log10(max_power), n_bins + 1)
        else:
            bins = np.linspace(min_power, max_power, n_bins + 1)

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_widths = np.diff(bins)

        # 3. 计算直方图 (hist_counts)
        hist_counts = np.zeros((n_channels, n_freqs, n_bins), dtype=np.uint32)

        # 进度提示
        total_iters = n_channels * n_freqs
        current_iter = 0
        print("正在计算直方图...")
        for ch in range(n_channels):
            for i_f in range(n_freqs):
                segment_powers = psd_segments[:, ch, i_f]
                # 去除 NaN
                segment_powers = segment_powers[~np.isnan(segment_powers)]
                if len(segment_powers) > 0:
                    hist, _ = np.histogram(segment_powers, bins=bins)
                    hist_counts[ch, i_f, :] = hist

                current_iter += 1
                if current_iter % 10000 == 0:
                    print(f"进度: {current_iter}/{total_iters}")

        # 4. 归一化为概率密度 (每个通道/频率点独立)
        print("正在计算概率密度...")
        pdf = np.full(hist_counts.shape, np.nan, dtype=np.float32)

        for ch in range(n_channels):
            for i_f in range(n_freqs):
                total = hist_counts[ch, i_f, :].sum()
                if total == 0:
                    continue
                # 密度 = 计数 / (总计数 * bin宽度)
                pdf[ch, i_f, :] = hist_counts[ch, i_f, :] / (total * bin_widths)

        # 5. 保存结果
        print(f"保存结果至: {output_path}")
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('freqs', data=freqs, dtype='float32')
            f.create_dataset('bin_centers', data=bin_centers, dtype='float32')
            f.create_dataset('bin_edges', data=bins, dtype='float32')
            f.create_dataset('pdf', data=pdf, dtype='float32', compression='gzip')
            f.create_dataset('hist_counts', data=hist_counts, dtype='uint32', compression='gzip')
            f.attrs['n_bins'] = n_bins
            f.attrs['log_scale'] = log_scale
            f.attrs['min_power'] = min_power
            f.attrs['max_power'] = max_power
            f.attrs['n_channels'] = n_channels
            f.attrs['n_freqs'] = n_freqs
            f.attrs['n_segments'] = n_times

        print("完成！")

    # compute_quantiles 方法也需要类似修正（独立保存，不要放在通道循环内）
    def compute_quantiles(self, psd_segments_h5: Union[str, Path],
                          output_path: Union[str, Path],
                          quantiles: List[float] = [0.1, 0.5, 0.9]):
        """
        从多段PSD文件中计算每个频率点的分位数。
        """
        with h5py.File(psd_segments_h5, 'r') as f:
            freqs = f['freqs'][:]
            psd_segments = f['psd_segments'][:]  # (n_times, n_channels, n_freqs)

        n_freqs = len(freqs)
        n_times, n_channels, _ = psd_segments.shape
        quantiles = np.array(quantiles)
        n_quantiles = len(quantiles)

        quantile_values = np.zeros((n_channels, n_freqs, n_quantiles), dtype=np.float32)

        print("正在计算分位数...")
        for ch in range(n_channels):
            for i_f in range(n_freqs):
                segment_powers = psd_segments[:, ch, i_f]
                segment_powers = segment_powers[~np.isnan(segment_powers)]
                if len(segment_powers) == 0:
                    quantile_values[ch, i_f, :] = np.nan
                else:
                    quantile_values[ch, i_f, :] = np.percentile(segment_powers, quantiles * 100)

        # 保存 (只保存一次)
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('freqs', data=freqs, dtype='float32')
            f.create_dataset('quantiles', data=quantiles, dtype='float32')
            f.create_dataset('quantile_values', data=quantile_values, dtype='float32', compression='gzip')
            f.attrs['n_channels'] = n_channels
            f.attrs['n_freqs'] = n_freqs
            f.attrs['n_segments'] = n_times
        print(f"功率谱分位数结果已保存至: {output_path}")

if __name__ == "__main__":
    from pythonProject.configs.configmanager import ConfigManager

    # 定义路径
    cfg_path = r"F:\Anylysis_project\pythonProject\tests\test_data\Ground.cfg"
    h5_path = r"F:\Anylysis_project\pythonProject\output\Ground_preprocessed.h5"
    output_dir = Path(r"F:\Anylysis_project\pythonProject\output")

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_mgr = ConfigManager.from_aether_cfg(cfg_path)

    # 创建计算器
    pdf_calc = PSDProbabilityDensity(h5_path, cfg_mgr)

    # 计算多段功率谱（保存到绝对路径）
    multitaper_path = output_dir / "Ground_multitaper_psd.h5"
    pdf_calc.compute_multitaper_psd(multitaper_path)

    # 计算概率密度统计（使用同一个绝对路径）
    pdf_output_path = output_dir / "Ground_psd_pdf.h5"
    pdf_calc.compute_pdf_statistics(
        multitaper_path,          # 输入：多段PSD文件
        pdf_output_path,          # 输出：PDF文件
        n_bins=50,
        log_scale=True
    )

    # 计算分位数
    quantile_path = output_dir / "Ground_psd_quantiles.h5"
    pdf_calc.compute_quantiles(
        multitaper_path,
        quantile_path,
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )