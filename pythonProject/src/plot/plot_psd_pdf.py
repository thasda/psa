"""
绘制功率谱概率密度图（PDF）
输入：由 pdf_service.compute_pdf_statistics 生成的 HDF5 文件
输出：保存为图片，显示每个通道的 PDF 热图
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, List


def plot_psd_pdf(
    pdf_h5_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    channel_names: Optional[List[str]] = None,
    figsize: tuple = (10, 8),
    dpi: int = 300,
    vmin: float = None,
    vmax: float = None,
    cmap: str = 'viridis',
    xscale: str = 'log',
    yscale: str = 'log',
    xlabel: str = 'Frequency (Hz)',
    ylabel: str = 'PSD ((mV/km)²/Hz)',
    save_fig: bool = True,
    show: bool = False
):
    """
    绘制功率谱概率密度图。

    参数:
        pdf_h5_path: 包含 pdf 数据集的 HDF5 文件路径（由 compute_pdf_statistics 生成）
        output_dir: 输出目录（如果为 None，则放在 pdf_h5_path 同级目录下）
        channel_names: 通道名称列表，长度与通道数一致，默认使用 'Ch0', 'Ch1', ...
        figsize: 图形大小
        dpi: 分辨率
        vmin, vmax: 颜色条范围（概率密度）
        cmap: 颜色映射
        xscale: x 轴刻度类型（'log' 或 'linear'）
        yscale: y 轴刻度类型（'log' 或 'linear'）
        xlabel: x 轴标签
        ylabel: y 轴标签
        save_fig: 是否保存图片
        show: 是否显示图形
    """
    pdf_h5_path = Path(pdf_h5_path)
    if output_dir is None:
        output_dir = pdf_h5_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(pdf_h5_path, 'r') as f:
        freqs = f['freqs'][:]               # (n_freqs,)
        bin_centers = f['bin_centers'][:]   # (n_bins,)
        pdf = f['pdf'][:]                   # (n_channels, n_freqs, n_bins)
        log_scale = f.attrs.get('log_scale', True)

    n_channels = pdf.shape[0]
    n_freqs = len(freqs)
    n_bins = len(bin_centers)

    if channel_names is None:
        channel_names = [f'Channel {i}' for i in range(n_channels)]
    else:
        if len(channel_names) != n_channels:
            raise ValueError(f"channel_names 长度 ({len(channel_names)}) 与通道数 ({n_channels}) 不一致")

    # 创建网格边缘（用于 pcolormesh）
    # 对数坐标下边缘使用几何平均，线性坐标下使用线性插值
    if xscale == 'log':
        # 确保频率为正数
        freq_edges = np.concatenate(([freqs[0] / np.sqrt(freqs[1]/freqs[0])],
                                     np.sqrt(freqs[:-1] * freqs[1:]),
                                     [freqs[-1] * np.sqrt(freqs[-1]/freqs[-2])]))
    else:
        freq_edges = np.linspace(freqs[0] - (freqs[1]-freqs[0])/2,
                                 freqs[-1] + (freqs[-1]-freqs[-2])/2,
                                 n_freqs+1)

    if yscale == 'log':
        # 确保功率为正数
        bin_edges = np.concatenate(([bin_centers[0] / np.sqrt(bin_centers[1]/bin_centers[0])],
                                    np.sqrt(bin_centers[:-1] * bin_centers[1:]),
                                    [bin_centers[-1] * np.sqrt(bin_centers[-1]/bin_centers[-2])]))
    else:
        bin_edges = np.linspace(bin_centers[0] - (bin_centers[1]-bin_centers[0])/2,
                                bin_centers[-1] + (bin_centers[-1]-bin_centers[-2])/2,
                                n_bins+1)

    # 创建网格
    X, Y = np.meshgrid(freq_edges, bin_edges)  # X shape (n_bins+1, n_freqs+1)

    # 为每个通道绘图
    for ch in range(n_channels):
        fig, ax = plt.subplots(figsize=figsize)
        # pdf[ch] shape: (n_freqs, n_bins)，转置为 (n_bins, n_freqs) 匹配网格
        pdf_ch = pdf[ch, :, :].T  # (n_bins, n_freqs)

        # pcolormesh 使用 shading='flat' 时，Z 的维度应为 (n_bins, n_freqs)
        mesh = ax.pcolormesh(X, Y, pdf_ch, shading='flat', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'PSD Probability Density - {channel_names[ch]}')
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label('Probability Density')

        if save_fig:
            out_path = output_dir / f"psd_pdf_{channel_names[ch]}.png"
            plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
            print(f"保存图片: {out_path}")
        if show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    # 示例用法
    pdf_file = r"F:\Anylysis_project\pythonProject\output\Ground_psd_pdf.h5"
    plot_psd_pdf(
        pdf_file,
        channel_names=['Ex', 'Ey', 'Bx', 'By', 'Bz'],
        ylabel='PSD ((mV/km)²/Hz)',
        show=True
    )