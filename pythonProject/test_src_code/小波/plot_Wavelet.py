import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter


class WaveletPlotter:
    """小波变换结果绘图类，支持功率谱、全局谱和能量谱密度图。"""

    def __init__(self, source=None, hdf5_path=None):
        """通过 WaveletResult 对象或 HDF5 文件路径初始化。"""
        if source is not None:
            self.result = source
        elif hdf5_path is not None:
            self.result = self._load_from_hdf5(hdf5_path)
        else:
            raise ValueError("请提供 source (WaveletResult) 或 hdf5_path")

    def _load_from_hdf5(self, path):
        """从 HDF5 文件加载小波结果。"""
        with h5py.File(path, 'r') as f:
            time = f['time'][:]
            freqs = f['freqs'][:]
            powers = f['powers'][:]
            coi_masks = f['coi_masks'][:]
            try:
                coeffs = f['coeffs'][:]
                phases = f['phases'][:]
            except KeyError:
                coeffs = None
                phases = None
            ch_names = list(f.attrs.get('channel_names', []))

        # 简易数据容器
        class ResultContainer:
            pass

        res = ResultContainer()
        res.time = time
        res.freqs = freqs
        res.powers = powers
        res.coi_masks = coi_masks
        res.coeffs = coeffs
        res.phases = phases
        res.channel_names = ch_names
        return res

    # ---------- 基础绘图方法 ----------
    def plot_power(self, channel=0, ax=None, figsize=(10, 6),
                   cmap='jet', show_coi=True, save_path=None, **kwargs):
        """绘制单通道小波功率谱，COI 以半透明阴影覆盖。"""
        ch_idx = self._resolve_channel(channel)
        power = self.result.powers[ch_idx]
        freqs = self.result.freqs
        time = self.result.time
        mask = self.result.coi_masks[ch_idx] if show_coi else None

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ext = [time[0], time[-1], freqs[0], freqs[-1]]
        im = ax.imshow(power, aspect='auto', origin='lower',
                       extent=ext, cmap=cmap, **kwargs)
        ax.set_yscale('log')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ch_name = self.result.channel_names[ch_idx] if self.result.channel_names else f'ch{channel}'
        ax.set_title(f'Wavelet Power Spectrum - {ch_name}')
        plt.colorbar(im, ax=ax, label='Power')

        if show_coi and mask is not None:
            # 用灰白色半透明覆盖受 COI 影响区域
            masked_power = np.ma.masked_where(mask, np.ones_like(power))
            ax.imshow(masked_power, extent=ext, aspect='auto',
                      origin='lower', cmap=plt.cm.gray, vmin=0, vmax=1, alpha=0.5)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return ax

    def plot_global(self, channel=0, ax=None, save_path=None):
        """绘制全局小波谱（时间平均功率，忽略 COI 区域）。"""
        ch_idx = self._resolve_channel(channel)
        power = self.result.powers[ch_idx]
        mask = self.result.coi_masks[ch_idx]
        power_masked = np.ma.masked_where(mask, power)
        global_power = power_masked.mean(axis=1)

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(global_power, self.result.freqs)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Power')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Global Wavelet Spectrum')
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return ax

    def plot_all_channels_power(self, save_dir=None, prefix='power_', **kwargs):
        """批量绘制所有通道的功率谱。"""
        for ch in range(len(self.result.powers)):
            name = self.result.channel_names[ch] if self.result.channel_names else f'ch{ch}'
            path = os.path.join(save_dir, f"{prefix}{name}.png") if save_dir else None
            self.plot_power(channel=ch, save_path=path, **kwargs)

    # ---------- 能量谱密度绘图（dB/Hz，小时横轴） ----------
    def plot_scalogram(self,
                       channel=0,
                       ax=None,
                       figsize=(12, 6),
                       cmap=None,
                       vmin=None, vmax=None,
                       title='Wavelet Power Spectral Density',
                       time_units='s',
                       show_colorbar=True,
                       show_coi=False,          # 是否叠加 COI 阴影
                       save_path=None,
                       dpi=150):
        """
        绘制小波能量谱（功率谱密度），横轴为小时，纵轴对数频率，颜色 dB/Hz。

        Parameters
        ----------
        channel : int or str
            通道索引或名称。
        ax : matplotlib.axes.Axes, optional
            绘图目标轴，若为 None 则创建新图。
        figsize : tuple
            图像大小。
        cmap : str or Colormap, optional
            颜色映射，默认使用自定义蓝‑灰‑红。
        vmin, vmax : float, optional
            dB/Hz 范围，默认自动取 1% ～ 99% 分位数。
        title : str
            标题。
        time_units : str
            输入时间单位，'s' 表示秒，将转换为小时。
        show_colorbar : bool
            是否显示 colorbar（图例）。
        show_coi : bool
            是否叠加影响锥阴影。
        save_path : str
            保存路径（含文件名），如 '/path/to/figure.png'。
        dpi : int
            图像分辨率。
        """
        # 解析通道
        if isinstance(channel, str):
            try:
                ch_idx = self.result.channel_names.index(channel)
            except ValueError:
                raise ValueError(f"通道 '{channel}' 不存在。")
        else:
            ch_idx = channel

        power = self.result.powers[ch_idx]
        freqs = np.asarray(self.result.freqs)
        time = np.asarray(self.result.time)

        # ---------- 横坐标转换为小时 ----------
        if time_units.lower() == 's':
            time_hours = time / 3600.0
        else:
            time_hours = time

        # ---------- 计算功率谱密度 (dB/Hz) ----------
        delta_f = np.gradient(freqs)                     # 每个频率的等效带宽
        delta_f = np.maximum(delta_f, 1e-12)
        psd = power / delta_f[:, np.newaxis]             # 功率谱密度
        psd_db = 10 * np.log10(psd + 1e-40)              # 转为 dB

        # ---------- 自定义颜色映射（深蓝 → 灰白 → 深红） ----------
        if cmap is None:
            colors = [(0.0, 0.0, 0.6),   # 深蓝
                      (0.85, 0.85, 0.85), # 灰白
                      (0.7, 0.0, 0.0)]   # 深红
            cmap = LinearSegmentedColormap.from_list('BlueWhiteRed', colors, N=256)
        else:
            cmap = plt.get_cmap(cmap)

        # ---------- 自动确定颜色范围 ----------
        if vmin is None or vmax is None:
            flat = psd_db[np.isfinite(psd_db)]
            if len(flat) > 0:
                low, high = np.percentile(flat, 1), np.percentile(flat, 99)
                if vmin is None: vmin = low
                if vmax is None: vmax = high
            else:
                if vmin is None: vmin = -20
                if vmax is None: vmax = 20

        # ---------- 绘图 ----------
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        extent = [time_hours[0], time_hours[-1], freqs[0], freqs[-1]]
        im = ax.imshow(psd_db,
                       aspect='auto',
                       origin='lower',
                       extent=extent,
                       cmap=cmap,
                       vmin=vmin,
                       vmax=vmax)

        ax.set_yscale('log')
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title(title)

        # 横轴格式化为 HH:MM（可选，当前仅显示小时数值）
        # ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x):02d}:{int((x%1)*60):02d}'))

        # COI 阴影（可选）
        if show_coi:
            mask = self.result.coi_masks[ch_idx]
            if mask is not None:
                coi_overlay = np.ma.masked_where(mask, np.ones_like(power))
                ax.imshow(coi_overlay, extent=extent, aspect='auto',
                          origin='lower', cmap=plt.cm.gray, vmin=0, vmax=1, alpha=0.5)

        # colorbar 置于右侧
        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax, label='Power Spectral Density [dB/Hz]')
            cbar.ax.yaxis.set_label_coords(-0.1, 0.5)

        fig.tight_layout()

        # 保存
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Scalogram 已保存至: {save_path}")

        return ax

    # ---------- 辅助方法 ----------
    def _resolve_channel(self, channel):
        if isinstance(channel, str):
            try:
                return self.result.channel_names.index(channel)
            except ValueError:
                raise ValueError(f"通道 {channel} 不存在")
        return channel