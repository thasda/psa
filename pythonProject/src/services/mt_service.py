import numpy as np
from scipy import signal
from scipy.fft import rfft, irfft
from typing import Optional, Tuple, List


class PostProcessor:
    """
    后处理类：去尖峰 + 稳健估计（频域加权平均）

    Parameters
    ----------
    sample_rate : float
        采样率（Hz）
    """

    def __init__(self, sample_rate: float):
        self.sample_rate = sample_rate

    # ==================== 1. 去尖峰（时域） ====================
    def remove_spikes(self,
                      data: np.ndarray,
                      method: str = 'mad',
                      threshold: float = 5.0,
                      window_len: int = 51,
                      replace_method: str = 'interp') -> np.ndarray:
        """
        检测并去除时间序列中的尖峰（脉冲噪声）。

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_channels)
            输入数据
        method : str
            'mad' : 使用滑动窗口的中位数绝对偏差（MAD）检测；
            'std' : 使用标准差（均值+threshold*std）检测；
            'fixed' : 固定阈值（绝对值超过 threshold 则视为尖峰）
        threshold : float
            阈值倍数（对 MAD/std 有效）
        window_len : int
            滑动窗口长度（必须为奇数）
        replace_method : str
            'interp' : 线性插值替换；
            'median' : 用窗口内中值替换；
            'zero' : 置零

        Returns
        -------
        cleaned : np.ndarray
            处理后的数据
        """
        if window_len % 2 == 0:
            window_len += 1  # 确保为奇数
        half = window_len // 2
        n_samples, n_ch = data.shape
        cleaned = data.copy()
        pad_width = half
        # 对每个通道独立处理
        for ch in range(n_ch):
            x = data[:, ch].astype(float)
            # 边缘填充（镜像）
            x_pad = np.pad(x, pad_width, mode='reflect')
            spike_mask = np.zeros(n_samples, dtype=bool)

            # 检测尖峰
            if method == 'mad':
                # 滑动中位数和 MAD
                med = np.zeros(n_samples)
                mad = np.zeros(n_samples)
                for i in range(n_samples):
                    window = x_pad[i:i + window_len]
                    med[i] = np.median(window)
                    mad[i] = np.median(np.abs(window - med[i]))
                # 避免除零
                mad[mad < 1e-12] = 1.0
                spike_mask = np.abs(x - med) > threshold * mad
            elif method == 'std':
                # 滑动均值与标准差
                mean = np.zeros(n_samples)
                std = np.zeros(n_samples)
                for i in range(n_samples):
                    window = x_pad[i:i + window_len]
                    mean[i] = np.mean(window)
                    std[i] = np.std(window)
                std[std < 1e-12] = 1.0
                spike_mask = np.abs(x - mean) > threshold * std
            elif method == 'fixed':
                spike_mask = np.abs(x) > threshold
            else:
                raise ValueError(f"不支持的 method: {method}")

            # 若没有尖峰则跳过
            if not np.any(spike_mask):
                continue

            # 替换尖峰点
            idx = np.where(spike_mask)[0]
            if replace_method == 'interp':
                # 线性插值：用最近的非尖峰点插值
                good_idx = np.where(~spike_mask)[0]
                if len(good_idx) < 2:
                    # 若好点太少，直接中值填充
                    cleaned[:, ch] = np.median(x)
                    continue
                # 使用 numpy 插值
                cleaned[:, ch] = np.interp(np.arange(n_samples), good_idx, x[good_idx])
            elif replace_method == 'median':
                for i in idx:
                    # 取该点前后窗口的中值（不含自身）
                    start = max(0, i - half)
                    end = min(n_samples, i + half + 1)
                    # 若窗口内全是尖峰，则用全局中值
                    seg = x[start:end]
                    good_seg = seg[~spike_mask[start:end]]
                    if len(good_seg) == 0:
                        cleaned[i, ch] = np.median(x)
                    else:
                        cleaned[i, ch] = np.median(good_seg)
            elif replace_method == 'zero':
                cleaned[idx, ch] = 0.0
            else:
                raise ValueError(f"不支持的 replace_method: {replace_method}")

        return cleaned

    # ==================== 2. 稳健估计（频域加权平均） ====================
    def robust_estimate(self,
                        data: np.ndarray,
                        nfft: int,
                        overlap: float = 0.5,
                        ref_channel: Optional[int] = None,
                        max_iter: int = 10,
                        tol: float = 1e-3,
                        robust_scale: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对时间序列进行分段加窗，计算互功率谱，并采用迭代重加权（IRLS）求取稳健平均频谱
        或传递函数（若提供参考通道）。

        参数
        ----------
        data : np.ndarray, shape (n_samples, n_channels)
            输入数据（已去尖峰/滤波）
        nfft : int
            FFT 点数（段长度）
        overlap : float
            段重叠比例（0~1）
        ref_channel : int or None
            若提供参考通道索引，则计算所有通道相对于该通道的传递函数（H = 互谱 / 自谱）
            若不提供，则返回各通道的功率谱密度（PSD）的稳健平均值
        max_iter : int
            稳健估计最大迭代次数
        tol : float
            权重变化收敛容差
        robust_scale : bool
            是否使用稳健尺度（MAD）更新权重

        Returns
        -------
        freqs : np.ndarray
            频率数组（正频率）
        spec : np.ndarray
            若 ref_channel 为 None，形状 (n_channels, n_freq) 为 PSD 稳健估计；
            否则形状 (n_channels, n_freq) 为复数传递函数 H(f) = 通道 / 参考通道
        weights : np.ndarray
            每个频率的最终权重（用于评估可靠性）
        """
        n_samples, n_ch = data.shape
        # 分段
        noverlap = int(nfft * overlap)
        # 使用 scipy 生成段索引
        segments = []
        step = nfft - noverlap
        for start in range(0, n_samples - nfft + 1, step):
            segments.append(data[start:start + nfft, :])
        if len(segments) < 2:
            raise ValueError("数据长度太短，无法分成至少2段")

        n_seg = len(segments)
        # 加窗
        window = signal.windows.hann(nfft, sym=False)
        # 预计算每个段的傅里叶变换（实信号 -> 正频率）
        spec_seg = []  # 存储每个段的 (n_ch, n_freq) 复数谱
        for seg in segments:
            # 加窗
            seg_win = seg * window[:, np.newaxis]
            # 实 FFT，只取正频率（nfft//2+1 个点）
            fft_vals = rfft(seg_win, axis=0)  # (n_freq, n_ch)
            spec_seg.append(fft_vals.T)  # 转置为 (n_ch, n_freq)
        spec_seg = np.array(spec_seg)  # (n_seg, n_ch, n_freq)
        n_freq = spec_seg.shape[2]

        freqs = np.fft.rfftfreq(nfft, d=1/self.sample_rate)

        if ref_channel is None:
            # ---- 计算各通道的功率谱密度（PSD）稳健平均 ----
            # 每个段、每个通道的 PSD (实部)
            psd_seg = np.abs(spec_seg) ** 2  # (n_seg, n_ch, n_freq)
            # 初始权重：均等
            weights = np.ones((n_seg, n_freq))
            # 迭代重加权
            for it in range(max_iter):
                # 加权平均（沿段维度）
                weighted_sum = np.sum(psd_seg * weights[..., np.newaxis], axis=0)  # (n_ch, n_freq)
                weight_sum = np.sum(weights, axis=0) + 1e-12
                avg_psd = weighted_sum / weight_sum[np.newaxis, :]  # (n_ch, n_freq)
                # 计算残差（每段、每频率的相对偏差）
                residuals = np.abs(psd_seg - avg_psd[np.newaxis, ...]) / (avg_psd[np.newaxis, ...] + 1e-12)
                # 更新权重：使用 Huber 或 Tukey 双权
                if robust_scale:
                    # 用 MAD 做稳健尺度
                    mad_res = np.median(np.abs(residuals - np.median(residuals, axis=0)), axis=0)
                    mad_res[mad_res < 1e-12] = 1.0
                    # Tukey 双权：|r| > 4.685 权重为0
                    c = 4.685
                    r_scaled = residuals / (mad_res[np.newaxis, :, :] + 1e-12)
                    w_new = np.where(np.abs(r_scaled) <= c,
                                     (1 - (r_scaled / c) ** 2) ** 2,
                                     0)
                else:
                    # 简单反比权重
                    w_new = 1.0 / (residuals + 1e-6)
                # 检查收敛
                diff = np.mean(np.abs(w_new - weights))
                weights = w_new
                if diff < tol:
                    break
            # 最终估计
            final_psd = np.sum(psd_seg * weights[..., np.newaxis], axis=0) / (np.sum(weights, axis=0)[np.newaxis, :] + 1e-12)
            return freqs, final_psd, np.mean(weights, axis=0)  # 权重均值作为可靠性指标

        else:
            # ---- 计算相对于参考通道的传递函数（复数） ----
            # 段：自谱 (ref) 和互谱 (ch, ref)
            spec_ref = spec_seg[:, ref_channel, :]  # (n_seg, n_freq)
            # 计算自谱和互谱
            s_ref = np.abs(spec_ref) ** 2  # (n_seg, n_freq)
            # 初始化权重
            weights = np.ones((n_seg, n_freq))
            # 复数传递函数 H = 互谱 / 自谱
            for it in range(max_iter):
                # 加权平均自谱和互谱
                S_ref_avg = np.sum(s_ref * weights, axis=0) / (np.sum(weights, axis=0) + 1e-12)
                # 对每个通道，计算互谱加权平均
                H_avg = np.zeros((n_ch, n_freq), dtype=complex)
                for ch in range(n_ch):
                    # 互谱：spec_ch * conj(spec_ref)
                    cross = spec_seg[:, ch, :] * np.conj(spec_ref)  # (n_seg, n_freq)
                    # 加权平均
                    cross_avg = np.sum(cross * weights, axis=0) / (np.sum(weights, axis=0) + 1e-12)
                    H_avg[ch, :] = cross_avg / (S_ref_avg + 1e-12)
                # 计算残差：每个段的传递函数与 H_avg 的差异（相对幅值）
                residuals = np.zeros((n_seg, n_freq))
                for ch in range(n_ch):
                    H_seg = spec_seg[:, ch, :] * np.conj(spec_ref) / (s_ref + 1e-12)  # (n_seg, n_freq)
                    # 计算相对幅度差
                    res_ch = np.abs(np.abs(H_seg) - np.abs(H_avg[ch, :])) / (np.abs(H_avg[ch, :]) + 1e-12)
                    residuals += res_ch
                residuals /= n_ch  # 平均所有通道
                # 更新权重（同之前）
                if robust_scale:
                    mad_res = np.median(np.abs(residuals - np.median(residuals, axis=0)), axis=0)
                    mad_res[mad_res < 1e-12] = 1.0
                    c = 4.685
                    r_scaled = residuals / (mad_res[np.newaxis, :] + 1e-12)
                    w_new = np.where(np.abs(r_scaled) <= c,
                                     (1 - (r_scaled / c) ** 2) ** 2,
                                     0)
                else:
                    w_new = 1.0 / (residuals + 1e-6)
                diff = np.mean(np.abs(w_new - weights))
                weights = w_new
                if diff < tol:
                    break
            # 最终传递函数
            S_ref_avg = np.sum(s_ref * weights, axis=0) / (np.sum(weights, axis=0) + 1e-12)
            H_final = np.zeros((n_ch, n_freq), dtype=complex)
            for ch in range(n_ch):
                cross = spec_seg[:, ch, :] * np.conj(spec_ref)
                cross_avg = np.sum(cross * weights, axis=0) / (np.sum(weights, axis=0) + 1e-12)
                H_final[ch, :] = cross_avg / (S_ref_avg + 1e-12)
            return freqs, H_final, np.mean(weights, axis=0)

