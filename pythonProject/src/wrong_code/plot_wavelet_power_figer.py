"""
用于绘制小波变换结果
1 读取并加载HDF5格式的小波变换结果
2 绘制小波能量谱，横坐标为Time（HH），纵坐标为频率（Hz，使用对数值），能量使用热力图（从-20dB/Hz到20dB/Hz，颜色从红-白-蓝）
3 多通道图像上下子图结构绘制
4 添加绘制小波相关性和相位图的功能
5 自定义保存图像路径
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from typing import List, Optional, Tuple, Union
import os


class WaveletPlotter:
    def __init__(self):
        # 新版结构
        self.cwt_coeffs = None    # (n_freq, n_time, n_channels)
        self.frequencies = None
        self.time = None
        self.coi = None
        self.channel_names = None
        self.sampling_rate = None

    def load_cwt_results(self, file_path):
        """
        修复：兼容新版 WaveletTransformTool 保存的 CWT 结果
        """
        with h5py.File(file_path, 'r') as f:
            # 新版结构（你现在用的）
            if 'coeffs' in f and 'frequencies' in f and 'time_axis' in f:
                self.cwt_coeffs = f['coeffs'][:]  # (n_freq, n_time, n_channels)
                self.frequencies = f['frequencies'][:]
                self.time = f['time_axis'][:]
                self.coi = f['coi'][:] if 'coi' in f else None
                print(f"✅ 已加载新版 CWT 结果")
                print(f"  系数形状: {self.cwt_coeffs.shape}")
                print(f"  频率数: {len(self.frequencies)}")
                print(f"  时间点: {len(self.time)}")

                # 自动生成通道名
                n_channels = self.cwt_coeffs.shape[2]
                self.channel_names = [f"Ch{i}" for i in range(n_channels)]
                return

            # 旧版结构兼容
            for key in f.keys():
                if key.startswith("cwt_"):
                    grp = f[key]
                    self.cwt_coeffs = grp['coeffs'][:]
                    self.frequencies = grp['frequencies'][:]
                    self.time = f['time_axis'][:] if 'time_axis' in f else np.arange(self.cwt_coeffs.shape[1])
                    self.channel_names = [key.replace("cwt_", "")]
                    return

        raise ValueError("未找到 CWT 结果")

    def _compute_energy_db(self, coeffs, db_range=(-20, 20)):
        energy = np.abs(coeffs) ** 2
        energy = np.maximum(energy, 1e-20)
        energy_db = 10 * np.log10(energy)
        energy_db = np.nan_to_num(energy_db, nan=db_range[0], posinf=db_range[1], neginf=db_range[0])
        energy_db = np.clip(energy_db, db_range[0], db_range[1])
        return energy_db

    def _sort_by_frequency(self, coeffs, freqs):
        """
        按频率升序排列系数和频率轴，确保 imshow 从下到上为低频→高频
        """
        sort_idx = np.argsort(freqs)
        return coeffs[sort_idx], freqs[sort_idx]

    def plot_energy_spectra(self, save_path=None, figsize=(12, 8),
                            channels=None, show_cbar=True,
                            db_range=(-20, 20),
                            decimate_time=1, decimate_freq=1,
                            show=True):
        """
        绘制小波能量谱（优化版）

        Parameters
        ----------
        save_path : str or None
            保存路径，不传则只显示不保存。
        show : bool
            是否弹窗显示图像（默认 True）。
        decimate_time : int
            时间方向步长。
        decimate_freq : int
            频率方向步长。
        """
        if self.cwt_coeffs is None:
            raise RuntimeError("未加载 CWT 结果，请先调用 load_cwt_results()")

        n_freq, n_time, n_channels_total = self.cwt_coeffs.shape

        # 选择通道
        if channels is None:
            channels = list(range(n_channels_total))
        n_channels = len(channels)

        # ---- 关键步骤：按频率升序重排所有数据 ----
        coeffs_all, freqs_all = self._sort_by_frequency(self.cwt_coeffs, self.frequencies)
        # 时间轴不变
        time_axis = self.time

        # 降采样索引
        freq_idx = slice(None, None, decimate_freq)
        time_idx = slice(None, None, decimate_time)
        freqs_plot = freqs_all[freq_idx]
        t_hour_plot = time_axis[time_idx] / 3600.0

        # 预计算所有通道的能量（降采样后）
        energy_db_list = []
        for ch in channels:
            coeffs = coeffs_all[freq_idx, time_idx, ch]
            energy_db_list.append(self._compute_energy_db(coeffs, db_range))

        # 创建子图
        fig, axes = plt.subplots(n_channels, 1, figsize=figsize,
                                 sharex=True, constrained_layout=True)
        if n_channels == 1:
            axes = [axes]

        cmap = "RdBu_r"
        norm = TwoSlopeNorm(vmin=db_range[0], vcenter=0, vmax=db_range[1])

        for idx, (ch_idx, energy_db) in enumerate(zip(channels, energy_db_list)):
            ax = axes[idx]

            # 预防全等数组导致 matplotlib 颜色条鼠标事件溢出
            if energy_db.max() == energy_db.min():
                energy_db[0, 0] += 1e-12

            im = ax.imshow(energy_db.T, aspect='auto', origin='lower',
                           cmap=cmap, norm=norm,
                           extent=[t_hour_plot[0], t_hour_plot[-1],
                                   freqs_plot[0], freqs_plot[-1]])
            im.format_cursor_data = lambda data: ''

            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'Channel: {self.channel_names[ch_idx]}')
            ax.set_yscale('log')
            # 明确设置 y 轴范围，防止对数坐标出界
            ax.set_ylim(freqs_plot[0], freqs_plot[-1])

        axes[-1].set_xlabel('Time (HH)')

        if show_cbar:
            cbar = fig.colorbar(im, ax=axes, pad=0.02, location='right')
            cbar.set_label('dB/Hz')

        plt.suptitle('Wavelet Energy Spectra', fontsize=14)

        # ---- 保存逻辑 ----
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 能量谱图已保存至: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()
