# src/visualization/psd_plotter.py
"""
功率谱密度可视化工具
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, List, Dict, Any


class PSDPlotter:
    """PSD 结果绘图器"""

    def __init__(self, psd_h5_path: Union[str, Path]):
        """
        参数:
            psd_h5_path: PSD 计算结果 HDF5 文件路径（应包含 freqs 和 psd 数据集）
        """
        self.psd_h5_path = Path(psd_h5_path)
        self.freqs = None
        self.psd = None
        self.n_channels = None
        self.sample_rate = None
        self._load_data()

    def _load_data(self):
        """加载 PSD 数据"""
        with h5py.File(self.psd_h5_path, 'r') as f:
            self.freqs = f['freqs'][:]
            self.psd = f['psd'][:]          # shape: (n_channels, n_freqs)
            self.n_channels = self.psd.shape[0]
            self.sample_rate = f.attrs.get('sample_rate', None)

    def plot_single(self, channel_idx: int, ax=None, **kwargs):
        """
        绘制单个通道的 PSD

        参数:
            channel_idx: 通道索引
            ax: matplotlib 轴对象，若为 None 则新建
            **kwargs: 传递给 ax.loglog 的参数
        返回:
            ax: matplotlib 轴对象
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(14, 8))

        ax.loglog(self.freqs, self.psd[channel_idx, :], **kwargs)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (dB/Hz)')
        ax.set_title(f'Channel {channel_idx} Power Spectral Density')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        return ax

    def plot_all(self, ax=None, legend: bool = True, **kwargs):
        """
        在同一张图上绘制所有通道的 PSD

        参数:
            ax: matplotlib 轴对象
            legend: 是否显示图例
            **kwargs: 传递给 ax.loglog 的通用参数（如 linestyle, marker 等）
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(14, 6))

        for ch in range(self.n_channels):
            ax.loglog(self.freqs, self.psd[ch, :], label=f'Ch{ch}', **kwargs)

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (dB/Hz)')
        ax.set_title('Power Spectral Density - All Channels')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        if legend:
            ax.legend()
        return ax

    def plot_groups(
        self,
        groups: List[Dict[str, Any]],
        colors: Optional[List[str]] = None,
        linewidth: float = 0.8,
        figsize: tuple = (6, 9),
        **kwargs
    ) -> tuple:
        """
        按通道分组绘制 PSD 曲线到多个子图。

        参数:
            groups: 列表，每个元素为字典，包含：
                    - 'indices': 通道索引列表
                    - 'labels': 对应标签列表
                    - 'title': (可选) 子图标题
            colors: 颜色列表，用于各通道（如果为 None，使用默认 SCI 配色）
            linewidth: 线条宽度
            figsize: 图形大小 (width, height)
            **kwargs: 传递给 ax.loglog 的额外参数

        返回:
            fig, axes: matplotlib 图形和轴对象
        """
        if colors is None:
            # SCI 论文常用颜色（Tableau 10）
            colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]

        n_groups = len(groups)
        fig, axes = plt.subplots(n_groups, 1, figsize=(figsize[1], figsize[0]))  # 垂直排列，交换宽高

        if n_groups == 1:
            axes = [axes]

        for idx, group in enumerate(groups):
            ax = axes[idx]
            indices = group['indices']
            labels = group['labels']
            title = group.get('title', f'Group {idx + 1}')

            for i, (ch_idx, label) in enumerate(zip(indices, labels)):
                color = colors[i % len(colors)]
                # 将 PSD 转换为 dB（假设 self.psd 是功率谱密度）
                psd_db = 10 * np.log10(self.psd[ch_idx, :] + 1e-12)  # 添加微小偏移避免 log(0)
                # 使用 semilogx 绘制，横坐标对数，纵坐标线性（dB 值）
                ax.semilogx(
                    self.freqs,
                    psd_db,
                    color=color,
                    linewidth=linewidth,
                    label=label,
                    **kwargs
                )

            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('PSD (dB/Hz)')  # 保持不变
            ax.set_title(title)
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            ax.legend(loc='best', fontsize='small')

        plt.tight_layout()
        return fig, axes

    def save_figure(self, output_path: Union[str, Path], **kwargs):
        """将当前图形保存到文件（需先调用 plot_* 方法）"""
        plt.savefig(output_path, **kwargs)
        print(f"图形已保存至: {output_path}")