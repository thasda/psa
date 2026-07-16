"""
LEMI-417 磁传感器校准模块（多通道支持，全序列频域校准）
解析 .cmt 文本校准文件，为每个磁通道提供仪器响应去除。
"""

import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path


class LEMi417Calibrator:
    """
    LEMI-417 多通道磁校准器。
    一次加载三个 .cmt 文件，分别为 HX, HY, HZ 提供频域校准。
    校准操作在整条时间序列上进行，保证低频分辨率和相位连续性。
    """

    def __init__(self, calibration_files, channel_mapping=None):
        """
        参数:
            calibration_files: list of str, 包含三个校准文件路径的列表，
                               顺序应为 [HX, HY, HZ]。
            channel_mapping: dict, 可选，通道名称到文件索引的映射。
                            若不提供，默认映射为 {'HX':0, 'HY':1, 'HZ':2}。
        """
        if not calibration_files or len(calibration_files) < 3:
            raise ValueError("必须提供三个校准文件（HX, HY, HZ）")

        self.calibration_files = calibration_files
        self.channel_mapping = channel_mapping if channel_mapping else {
            'HX': 0, 'HY': 1, 'HZ': 2
        }

        # 为每个通道存储插值器
        self._interpolators = {}
        for ch_name, idx in self.channel_mapping.items():
            file_path = calibration_files[idx]
            self._interpolators[ch_name] = self._build_interpolator(file_path)

    def _build_interpolator(self, file_path):
        """从 .cmt 文件构建插值器字典"""
        data = np.loadtxt(file_path, comments='#')
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(f"校准文件 {file_path} 应包含三列：频率 灵敏度 相位")

        freqs = data[:, 0]
        sens = data[:, 1]   # mV/nT
        phases = data[:, 2] # 度

        # 按频率排序（保证递增）
        idx = np.argsort(freqs)
        freqs = freqs[idx]
        sens = sens[idx]
        phases = phases[idx]

        # 对数频率插值器
        log_freqs = np.log(freqs)
        log_sens = np.log(sens)

        interp_sens = interp1d(
            log_freqs, log_sens,
            kind='linear',
            bounds_error=False,
            fill_value=(log_sens[0], log_sens[-1])
        )
        interp_phase = interp1d(
            log_freqs, phases,
            kind='linear',
            bounds_error=False,
            fill_value=(phases[0], phases[-1])
        )

        return {
            'freqs': freqs,
            'sens': sens,
            'phases': phases,
            'interp_sens': interp_sens,
            'interp_phase': interp_phase
        }

    def calibrate_time_series(self, data, sample_rate, channel_name=None):
        """
        对整条时间序列进行仪器响应去除（频域一次完成）。

        参数:
            data: 1D numpy array, 原始电压数据（单位：mV）
            sample_rate: float, 采样率（Hz）
            channel_name: str, 通道名称（'HX', 'HY', 'HZ'），用于选择校准文件。

        返回:
            calibrated: 1D numpy array, 校准后的磁场数据（单位：nT）
        """
        if channel_name is None:
            # 若未指定，尝试从 channel_mapping 推断（单文件兼容）
            if len(self._interpolators) == 1:
                channel_name = list(self._interpolators.keys())[0]
            else:
                raise ValueError("多通道校准器必须指定 channel_name")

        interp = self._interpolators.get(channel_name)
        if interp is None:
            raise ValueError(f"未找到通道 '{channel_name}' 的校准器，可用: {list(self._interpolators.keys())}")

        n = len(data)
        # 实数 FFT
        fft_data = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

        # 计算校正因子
        cal_factor = self._compute_factor(freqs, interp)

        # 应用校正
        fft_cal = fft_data * cal_factor

        # 逆变换回时域
        calibrated = np.fft.irfft(fft_cal, n=n)
        return calibrated

    def _compute_factor(self, freqs, interp):
        """根据频率数组和插值器计算复数校正因子"""
        factor = np.ones_like(freqs, dtype=complex)
        nonzero = freqs > 0

        if np.any(nonzero):
            log_f = np.log(freqs[nonzero])
            log_s = interp['interp_sens'](log_f)
            phase_deg = interp['interp_phase'](log_f)

            sens = np.exp(log_s)
            phase_rad = np.deg2rad(phase_deg)
            # 校正：除以灵敏度，并减去相位滞后
            factor[nonzero] = (1.0 / sens) * np.exp(-1j * phase_rad)

        # 零频处理：使用最低频率灵敏度，相位为0
        if len(interp['sens']) > 0:
            factor[0] = 1.0 / interp['sens'][0]
        else:
            factor[0] = 1.0

        return factor

    def get_frequency_response(self, frequencies, channel_name='HX'):
        """
        获取指定频率点的复数频率响应 H(f) = V_out / B_true。
        用于调试或绘图。
        """
        interp = self._interpolators[channel_name]
        freqs = np.asarray(frequencies)
        nonzero = freqs > 0
        log_f = np.log(freqs[nonzero])
        log_s = interp['interp_sens'](log_f)
        sens = np.exp(log_s)
        phase_deg = interp['interp_phase'](log_f)
        phase_rad = np.deg2rad(phase_deg)

        response = np.zeros_like(freqs, dtype=complex)
        response[nonzero] = sens * np.exp(1j * phase_rad)
        if len(interp['sens']) > 0:
            response[0] = interp['sens'][0]
        return response