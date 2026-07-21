# src/visualization/allan_plotter.py
"""
艾伦方差分组绘图器
支持单文件分组（电场/磁场上下子图）和双文件对比（第二个文件虚线）
每个通道可独立设置颜色
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, List, Tuple


class AllanGroupPlotter:
    def __init__(self, h5_path: Union[str, Path]):
        self.h5_path = Path(h5_path)
        self.taus = None
        self.allan_var = None
        self.n_channels = None
        self.sample_rate = None
        self.channel_names = None
        self._load_data()

    def _load_data(self):
        with h5py.File(self.h5_path, 'r') as f:
            self.taus = f['taus'][:]
            if 'allan_variance' in f:
                self.allan_var = f['allan_variance'][:]
            elif 'allan_dev' in f:
                self.allan_var = f['allan_dev'][:]
            else:
                raise KeyError("缺少 'allan_variance' 或 'allan_dev'")
            self.n_channels = self.allan_var.shape[0]
            self.sample_rate = f.attrs.get('sample_rate', None)
            if 'channel_names' in f:
                ch_data = f['channel_names'][:]
                self.channel_names = [c.decode() if isinstance(c, bytes) else str(c) for c in ch_data]
            else:
                self.channel_names = [f'Ch{i}' for i in range(self.n_channels)]

    def get_channel_name(self, idx: int) -> str:
        return self.channel_names[idx] if idx < len(self.channel_names) else f'Ch{idx}'

    @staticmethod
    def plot_group(
        plotter: 'AllanGroupPlotter',
        group_indices: List[int],
        group_labels: List[str],
        group_title: str = '',
        ax: Optional[plt.Axes] = None,
        colors: Optional[List[str]] = None,
        linestyle: str = '-',
        linewidth: float = 1.5,
        xscale: str = 'log',
        yscale: str = 'log',
        show_grid: bool = True,
        legend: bool = True,
        **plot_kwargs
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        if colors is None:
            cmap = plt.cm.tab10
            colors = [cmap(i % 10) for i in range(len(group_indices))]
        # 确保颜色数量足够
        if len(colors) < len(group_indices):
            cmap = plt.cm.tab10
            colors = colors + [cmap(i % 10) for i in range(len(colors), len(group_indices))]

        for idx, ch_idx in enumerate(group_indices):
            y = plotter.allan_var[ch_idx, :]
            mask = ~np.isnan(y)
            label = group_labels[idx] if idx < len(group_labels) else plotter.get_channel_name(ch_idx)
            ax.plot(plotter.taus[mask], y[mask],
                    color=colors[idx],
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=label,
                    **plot_kwargs)
        ax.set_xlabel('τ (s)')
        ax.set_ylabel('Allan Variance')
        if group_title:
            ax.set_title(group_title)
        if show_grid:
            ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if legend:
            ax.legend()
        return ax

    @classmethod
    def plot_eh_groups(
        cls,
        plotter: 'AllanGroupPlotter',
        e_indices: List[int],
        h_indices: List[int],
        e_labels: Optional[List[str]] = None,
        h_labels: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 8),
        e_colors: Optional[List[str]] = None,
        h_colors: Optional[List[str]] = None,
        e_linestyle: str = '-',
        h_linestyle: str = '-',
        linewidth: float = 1.5,
        xscale: str = 'log',
        yscale: str = 'log',
        show_grid: bool = True,
        legend: bool = True,
        title: Optional[str] = None,
        **plot_kwargs
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        if e_labels is None:
            e_labels = [plotter.get_channel_name(i) for i in e_indices]
        if h_labels is None:
            h_labels = [plotter.get_channel_name(i) for i in h_indices]
        if e_colors is None:
            cmap = plt.cm.tab10
            e_colors = [cmap(i % 10) for i in range(len(e_indices))]
        if h_colors is None:
            cmap = plt.cm.tab10
            h_colors = [cmap((i+3) % 10) for i in range(len(h_indices))]

        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        cls.plot_group(plotter, e_indices, e_labels, group_title="Electric Field (E)",
                       ax=ax_top, colors=e_colors, linestyle=e_linestyle,
                       linewidth=linewidth, xscale=xscale, yscale=yscale,
                       show_grid=show_grid, legend=legend, **plot_kwargs)

        cls.plot_group(plotter, h_indices, h_labels, group_title="Magnetic Field (H)",
                       ax=ax_bottom, colors=h_colors, linestyle=h_linestyle,
                       linewidth=linewidth, xscale=xscale, yscale=yscale,
                       show_grid=show_grid, legend=legend, **plot_kwargs)

        ax_bottom.set_xlabel('τ (s)')
        if title:
            fig.suptitle(title)
        plt.tight_layout()
        return fig, (ax_top, ax_bottom)

    @classmethod
    def plot_eh_compare(
        cls,
        plotter1: 'AllanGroupPlotter',
        plotter2: 'AllanGroupPlotter',
        e_indices1: List[int],
        h_indices1: List[int],
        e_indices2: List[int],
        h_indices2: List[int],
        e_labels1: Optional[List[str]] = None,
        h_labels1: Optional[List[str]] = None,
        e_labels2: Optional[List[str]] = None,
        h_labels2: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 8),
        e_colors1: Optional[List[str]] = None,
        h_colors1: Optional[List[str]] = None,
        e_colors2: Optional[List[str]] = None,
        h_colors2: Optional[List[str]] = None,
        e_linestyle1: str = '-',
        h_linestyle1: str = '-',
        e_linestyle2: str = '--',
        h_linestyle2: str = '--',
        linewidth: float = 1.5,
        xscale: str = 'log',
        yscale: str = 'log',
        show_grid: bool = True,
        legend: bool = True,
        title: Optional[str] = None,
        **plot_kwargs
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        if e_labels1 is None:
            e_labels1 = [f"{plotter1.get_channel_name(i)} (1)" for i in e_indices1]
        if h_labels1 is None:
            h_labels1 = [f"{plotter1.get_channel_name(i)} (1)" for i in h_indices1]
        if e_labels2 is None:
            e_labels2 = [f"{plotter2.get_channel_name(i)} (2)" for i in e_indices2]
        if h_labels2 is None:
            h_labels2 = [f"{plotter2.get_channel_name(i)} (2)" for i in h_indices2]

        if e_colors1 is None:
            cmap = plt.cm.tab10
            e_colors1 = [cmap(i % 10) for i in range(len(e_indices1))]
        if h_colors1 is None:
            cmap = plt.cm.tab10
            h_colors1 = [cmap((i+3) % 10) for i in range(len(h_indices1))]
        if e_colors2 is None:
            e_colors2 = e_colors1[:len(e_indices2)]
        if h_colors2 is None:
            h_colors2 = h_colors1[:len(h_indices2)]

        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # 电场：文件1实线
        cls.plot_group(plotter1, e_indices1, e_labels1, group_title="",
                       ax=ax_top, colors=e_colors1, linestyle=e_linestyle1,
                       linewidth=linewidth, xscale=xscale, yscale=yscale,
                       show_grid=show_grid, legend=legend, **plot_kwargs)
        # 电场：文件2虚线
        cls.plot_group(plotter2, e_indices2, e_labels2, group_title="",
                       ax=ax_top, colors=e_colors2, linestyle=e_linestyle2,
                       linewidth=linewidth, xscale=xscale, yscale=yscale,
                       show_grid=False, legend=legend, **plot_kwargs)
        ax_top.set_title("Electric Field (E)")

        # 磁场：文件1实线
        cls.plot_group(plotter1, h_indices1, h_labels1, group_title="",
                       ax=ax_bottom, colors=h_colors1, linestyle=h_linestyle1,
                       linewidth=linewidth, xscale=xscale, yscale=yscale,
                       show_grid=show_grid, legend=legend, **plot_kwargs)
        # 磁场：文件2虚线
        cls.plot_group(plotter2, h_indices2, h_labels2, group_title="",
                       ax=ax_bottom, colors=h_colors2, linestyle=h_linestyle2,
                       linewidth=linewidth, xscale=xscale, yscale=yscale,
                       show_grid=False, legend=legend, **plot_kwargs)
        ax_bottom.set_title("Magnetic Field (H)")

        ax_bottom.set_xlabel('τ (s)')
        if title:
            fig.suptitle(title)
        plt.tight_layout()
        return fig, (ax_top, ax_bottom)
