import numpy as np
import pandas as pd
from scipy.signal import firwin, filtfilt, iirfilter, iirnotch, lfilter
from typing import Union, List, Optional


class SignalFilter:
    """
    信号滤波类，支持 FIR、IIR、自适应滤波器以及工频陷波。
    输入输出均为 pandas.DataFrame，索引为时间。
    """

    def __init__(self, sample_rate: float):
        """
        参数:
            sample_rate: 采样率 (Hz)
        """
        self.sample_rate = sample_rate
        self.nyquist = 0.5 * sample_rate

    # ---------- 工频陷波（简单实现） ----------
    def notch_powerline(self, data: pd.DataFrame, freqs: List[float] = [50, 100, 150, 200, 250, 300, 350, 400, 450],
                        Q: float = 30.0) -> pd.DataFrame:
        """
        级联陷波滤波器滤除指定频率及其谐波。

        参数:
            data: 输入数据 (DataFrame)
            freqs: 需要滤除的频率列表 (Hz)
            Q: 品质因数

        返回:
            滤波后的 DataFrame
        """
        if data.empty:
            return data.copy()

        filtered = data.copy()
        for f0 in freqs:
            if f0 >= self.nyquist:
                continue
            b, a = iirnotch(f0, Q, self.sample_rate)
            # 对每一列应用 filtfilt 以保证零相位
            for col in filtered.columns:
                filtered[col] = filtfilt(b, a, filtered[col].values)
        return filtered

    # ---------- FIR 滤波器 ----------
    def apply_fir(self, data: pd.DataFrame, cutoff: Union[float, List[float]],
                  num_taps: int, btype: str = 'lowpass',
                  window: str = 'hamming') -> pd.DataFrame:
        """
        FIR 滤波器设计 (使用 firwin + filtfilt)

        参数:
            data: 输入数据 (DataFrame)
            cutoff: 截止频率 (Hz)，带通/带阻为 [low, high]
            num_taps: 滤波器阶数（抽头数）
            btype: 'lowpass', 'highpass', 'bandpass', 'bandstop'
            window: 窗函数 (默认 'hamming')

        返回:
            滤波后的 DataFrame
        """
        if data.empty:
            return data.copy()

        # 统一处理 cutoff 为列表
        if isinstance(cutoff, (int, float)):
            cutoff = [cutoff]
        elif not isinstance(cutoff, list):
            raise TypeError("cutoff 必须为数值或列表")

        # 校验长度与 btype 匹配
        if btype in ['lowpass', 'highpass'] and len(cutoff) != 1:
            raise ValueError(f"{btype} 需要 1 个截止频率，得到 {len(cutoff)} 个")
        if btype in ['bandpass', 'bandstop'] and len(cutoff) != 2:
            raise ValueError(f"{btype} 需要 2 个截止频率，得到 {len(cutoff)} 个")

        # 设计滤波器 (传入绝对频率, 指定 fs)
        taps = firwin(num_taps, cutoff, fs=self.sample_rate,
                      window=window, pass_zero=btype)

        # 应用 filtfilt (零相位)
        filtered = data.copy()
        padlen = num_taps - 1
        for col in filtered.columns:
            filtered[col] = filtfilt(taps, 1.0, filtered[col].values, padlen=padlen)
        return filtered

    # ---------- IIR 滤波器 ----------
    def apply_iir(self, data: pd.DataFrame, cutoff: Union[float, List[float]],
                  order: int, btype: str = 'lowpass',
                  iir_type: str = 'butter', ripple: float = 1.0) -> pd.DataFrame:
        """
        IIR 滤波器 (使用 iirfilter + filtfilt)

        参数:
            data: 输入数据 (DataFrame)
            cutoff: 截止频率 (Hz)，带通/带阻为 [low, high]
            order: 滤波器阶数
            btype: 'lowpass', 'highpass', 'bandpass', 'bandstop'
            iir_type: 'butter', 'cheby1', 'cheby2', 'ellip'
            ripple: 通带纹波 (dB) 或阻带衰减 (dB)，取决于 iir_type

        返回:
            滤波后的 DataFrame
        """
        if data.empty:
            return data.copy()

        # 统一处理 cutoff
        if isinstance(cutoff, (int, float)):
            cutoff = [cutoff]
        elif not isinstance(cutoff, list):
            raise TypeError("cutoff 必须为数值或列表")

        # 校验长度
        if btype in ['lowpass', 'highpass'] and len(cutoff) != 1:
            raise ValueError(f"{btype} 需要 1 个截止频率，得到 {len(cutoff)} 个")
        if btype in ['bandpass', 'bandstop'] and len(cutoff) != 2:
            raise ValueError(f"{btype} 需要 2 个截止频率，得到 {len(cutoff)} 个")

        # 归一化频率 (相对于 Nyquist)
        norm_cutoff = [f / self.nyquist for f in cutoff]

        # 设计滤波器系数
        if iir_type == 'butter':
            b, a = iirfilter(order, norm_cutoff, btype=btype, ftype='butter', output='ba')
        elif iir_type == 'cheby1':
            b, a = iirfilter(order, norm_cutoff, rp=ripple, btype=btype, ftype='cheby1', output='ba')
        elif iir_type == 'cheby2':
            b, a = iirfilter(order, norm_cutoff, rs=ripple, btype=btype, ftype='cheby2', output='ba')
        elif iir_type == 'ellip':
            # 使用相同的 ripple 作为通带纹波，阻带衰减设为 ripple*2 (可调整)
            b, a = iirfilter(order, norm_cutoff, rp=ripple, rs=ripple*2, btype=btype, ftype='ellip', output='ba')
        else:
            raise ValueError(f"不支持的 IIR 类型: {iir_type}")

        # 应用 filtfilt
        filtered = data.copy()
        padlen = 3 * (max(len(b), len(a)) - 1)
        for col in filtered.columns:
            filtered[col] = filtfilt(b, a, filtered[col].values, padlen=padlen)
        return filtered

    # ---------- 自适应滤波器（简化版，使用 LMS/NLMS/RLS，针对单通道） ----------
    @staticmethod
    def _lms(x: np.ndarray, filt_len: int, step: float) -> np.ndarray:
        """LMS 自适应滤波器 (单通道)"""
        w = np.zeros(filt_len)
        y = np.zeros_like(x)
        y[:filt_len] = x[:filt_len]
        for n in range(filt_len, len(x)):
            x_n = x[n - filt_len:n][::-1]
            y[n] = np.dot(w, x_n)
            e = x[n] - y[n]  # 以自身为期望（去自相关）
            w += step * e * x_n
        return y

    @staticmethod
    def _nlms(x: np.ndarray, filt_len: int, step: float) -> np.ndarray:
        """NLMS 自适应滤波器"""
        w = np.zeros(filt_len)
        y = np.zeros_like(x)
        y[:filt_len] = x[:filt_len]
        eps = 1e-10
        for n in range(filt_len, len(x)):
            x_n = x[n - filt_len:n][::-1]
            y[n] = np.dot(w, x_n)
            e = x[n] - y[n]
            norm_x2 = np.dot(x_n, x_n) + eps
            w += (step / norm_x2) * e * x_n
        return y

    @staticmethod
    def _rls(x: np.ndarray, filt_len: int, lambda_inv: float) -> np.ndarray:
        """RLS 自适应滤波器 (lambda_inv = 1/λ)"""
        lambd = 1.0 / lambda_inv
        w = np.zeros(filt_len)
        P = (1.0 / lambda_inv) * np.eye(filt_len)
        y = np.zeros_like(x)
        y[:filt_len] = x[:filt_len]
        for n in range(filt_len, len(x)):
            x_n = x[n - filt_len:n][::-1]
            y[n] = np.dot(w, x_n)
            e = x[n] - y[n]
            # 增益向量
            denom = lambd + np.dot(x_n.T, np.dot(P, x_n))
            k = np.dot(P, x_n) / denom
            w += k * e
            # 更新 P
            P = (1.0 / lambd) * (P - np.outer(k, np.dot(x_n.T, P)))
        return y

    def apply_adaptive(self, data: pd.DataFrame, filter_length: int, step_size: float,
                       algorithm: str = 'LMS') -> pd.DataFrame:
        """
        自适应滤波器 (目前仅支持单通道信号，逐列处理)

        参数:
            data: 输入数据 (DataFrame)
            filter_length: 滤波器长度
            step_size: 步长 (对于 RLS 为 1/遗忘因子)
            algorithm: 'LMS', 'NLMS', 'RLS'

        返回:
            滤波后的 DataFrame
        """
        if data.empty:
            return data.copy()

        alg_map = {
            'LMS': self._lms,
            'NLMS': self._nlms,
            'RLS': self._rls
        }
        if algorithm.upper() not in alg_map:
            raise ValueError(f"不支持的自适应算法: {algorithm}")

        func = alg_map[algorithm.upper()]
        filtered = data.copy()
        for col in filtered.columns:
            x = filtered[col].values.astype(float)
            # 填充NaN为0（简单处理）
            x = np.nan_to_num(x)
            filtered[col] = func(x, filter_length, step_size)
        return filtered