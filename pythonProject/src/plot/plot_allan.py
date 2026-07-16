# src/visualization/allan_plotter.py
"""
艾伦方差可视化工具
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, List, Dict, Any


class AllanPlotter:
    """艾伦方差绘图器"""

    def __init__(self, allan_h5_path: Union[str, Path]):
        """
        参数:
            allan_h5_path: 艾伦方差结果 HDF5 文件路径（应包含 taus 和 allan_variance 数据集）
        """
        self.allan_h5_path = Path(allan_h5_path)
        self.taus = None
        self.allan_var = None          # 方差数组，形状 (n_channels, n_taus)
        self.n_channels = None
        self.sample_rate = None
        self._load_data()

    def _load_data(self):
        """加载艾伦方差数据，兼容新旧数据集名称"""
        with h5py.File(self.allan_h5_path, 'r') as f:
            self.taus = f['taus'][:]
            # 尝试加载新名称 'allan_variance'，若不存在则回退到旧名称 'allan_dev'（兼容旧版本）
            if 'allan_variance' in f:
                self.allan_var = f['allan_variance'][:]
            elif 'allan_dev' in f:
                self.allan_var = f['allan_dev'][:]
            else:
                raise KeyError("HDF5 文件中缺少 'allan_variance' 或 'allan_dev' 数据集")
            self.n_channels = self.allan_var.shape[0]
            self.sample_rate = f.attrs.get('sample_rate', None)

    def plot_single(self, channel_idx: int, ax=None, **kwargs):
        """绘制单个通道的艾伦方差曲线（双对数坐标）"""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        tau = self.taus
        y = self.allan_var[channel_idx, :]
        mask = ~np.isnan(y)
        tau_plot = tau[mask]
        y_plot = y[mask]
        ax.loglog(tau_plot, y_plot, **kwargs)
        ax.set_xlabel('τ (s)')
        ax.set_ylabel('Allan Variance')
        ax.set_title(f'Channel {channel_idx} Allan Variance')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        return ax

    def plot_all(self, ax=None, legend=True, channel_labels=None, **kwargs):
        """
        绘制所有通道的艾伦方差曲线。

        参数:
            channel_labels: list of str, 长度等于通道数，用于自定义图例标签。
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        for ch in range(self.n_channels):
            tau = self.taus
            y = self.allan_var[ch, :]
            mask = ~np.isnan(y)
            if channel_labels is not None and ch < len(channel_labels):
                label = channel_labels[ch]
            else:
                label = kwargs.pop('label', f'Ch{ch}')
            ax.loglog(tau[mask], y[mask], label=label, **kwargs)

        ax.set_xlabel('τ (s)')
        ax.set_ylabel('Allan Variance')
        ax.set_title('Allan Variance - All Channels')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        if legend:
            ax.legend()
        return ax

    def plot_all_and_save(self, output_path: Union[str, Path],
                          save_kwargs: Optional[dict] = None,
                          channel_labels: Optional[List[str]] = None,
                          **plot_kwargs):
        """
        绘制所有通道的艾伦方差并保存为图片。

        参数:
            channel_labels: 传递给 plot_all 的自定义标签列表
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        self.plot_all(ax=ax, channel_labels=channel_labels, **plot_kwargs)
        if save_kwargs is None:
            save_kwargs = {}
        self.save_figure(output_path, **save_kwargs)
        plt.close(fig)

    def plot_groups(self, groups: List[Dict[str, Any]], colors: Optional[List[str]] = None,
                    linewidth: float = 0.8, figsize: tuple = (14, 6), **kwargs) -> tuple:
        """
        按通道分组绘制艾伦方差曲线到多个子图（上下结构）。

        参数:
            groups: 列表，每个元素为字典，包含：
                    - 'indices': 通道索引列表
                    - 'labels': 对应标签列表
                    - 'title': (可选) 子图标题
        """
        if colors is None:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        n_groups = len(groups)
        # 上下结构：n_groups 行，1 列
        fig, axes = plt.subplots(n_groups, 1, figsize=(figsize[0], figsize[1] * n_groups))
        if n_groups == 1:
            axes = [axes]  # 统一为列表形式便于循环
        for idx, group in enumerate(groups):
            ax = axes[idx]
            indices = group['indices']
            labels = group['labels']
            title = group.get('title', f'Group {idx + 1}')
            for i, (ch_idx, label) in enumerate(zip(indices, labels)):
                color = colors[i % len(colors)]
                tau = self.taus
                y = self.allan_var[ch_idx, :]
                mask = ~np.isnan(y)
                ax.loglog(tau[mask], y[mask], color=color, linewidth=linewidth,
                          label=label, **kwargs)
            ax.set_xlabel('τ (s)')
            ax.set_ylabel('Allan Variance')
            ax.set_title(title)
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            ax.legend(loc='best', fontsize='small')
        plt.tight_layout()
        return fig, axes

    def save_figure(self, output_path: Union[str, Path], **kwargs):
        """将当前图形保存到文件（需先调用 plot_* 方法）"""
        plt.savefig(output_path, **kwargs)
        print(f"图形已保存至: {output_path}")