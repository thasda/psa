# tests/test_plot_psd.py
"""
测试 PSDPlotter 类，使用真实 PSD 结果文件。
需要以下文件存在：
- F:\Anylysis_project\pythonProject\output\1804323_300_preprocessed_psd.h5
"""

import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent  # 当前文件在 tests/ 下，上一级到项目根目录
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import matplotlib
# 使用非交互式后端，避免弹出图形窗口
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

from pythonProject.src.plot.plot_psd import PSDPlotter

# 真实 PSD 文件路径
PSD_FILE = project_root / "output" / "1804323_300_preprocessed_psd.h5"


@pytest.mark.slow
def test_psd_plotter_initialization():
    """测试 PSDPlotter 初始化和数据加载"""
    assert PSD_FILE.exists(), f"PSD 文件不存在: {PSD_FILE}"

    plotter = PSDPlotter(str(PSD_FILE))

    # 检查属性是否正确加载
    assert plotter.freqs is not None, "freqs 未加载"
    assert plotter.psd is not None, "psd 未加载"
    assert plotter.n_channels is not None, "n_channels 未加载"
    assert plotter.sample_rate is not None, "sample_rate 未加载"

    # 验证形状和 sample_rate
    with h5py.File(PSD_FILE, 'r') as f:
        expected_freqs = f['freqs'][:]
        expected_psd = f['psd'][:]
        expected_sample_rate = f.attrs.get('sample_rate', None)

    np.testing.assert_array_equal(plotter.freqs, expected_freqs)
    np.testing.assert_array_equal(plotter.psd, expected_psd)
    assert plotter.n_channels == expected_psd.shape[0]
    assert plotter.sample_rate == expected_sample_rate


@pytest.mark.slow
def test_plot_single():
    """测试 plot_single 方法不抛出异常，并返回正确的轴对象"""
    plotter = PSDPlotter(str(PSD_FILE))

    # 创建一个 figure 和轴
    fig, ax = plt.subplots()
    returned_ax = plotter.plot_single(0, ax=ax, color='blue', linewidth=0.8)

    # 验证返回的轴是同一个对象
    assert returned_ax is ax

    # 验证轴上有数据（至少一条线）
    assert len(ax.lines) == 1

    # 验证轴标签和标题已设置
    assert ax.get_xlabel() == 'Frequency (Hz)'
    assert ax.get_ylabel() == 'PSD (dB/Hz)'
    assert ax.get_title() == 'Channel 0 Power Spectral Density'

    plt.close(fig)


@pytest.mark.slow
def test_plot_all():
    """测试 plot_all 方法"""
    plotter = PSDPlotter(str(PSD_FILE))

    fig, ax = plt.subplots()
    returned_ax = plotter.plot_all(ax=ax)

    assert returned_ax is ax
    # 应绘制所有通道
    assert len(ax.lines) == plotter.n_channels

    # 图例应存在（默认 True）
    assert ax.get_legend() is not None

    plt.close(fig)


@pytest.mark.slow
def test_plot_groups():
    """测试 plot_groups 方法"""
    plotter = PSDPlotter(str(PSD_FILE))

    # 定义分组（假设至少5个通道）
    groups = [
        {"indices": [0, 1], "labels": ["Ex", "Ey"], "title": "Electric Channels"},
        {"indices": [2, 3, 4], "labels": ["Bx", "By", "Bz"], "title": "Magnetic Channels"}
    ]

    fig, axes = plotter.plot_groups(groups)

    # 检查返回的 figure 和 axes
    assert fig is not None
    assert len(axes) == len(groups)

    # 检查每个子图中的线条数
    for i, group in enumerate(groups):
        ax = axes[i]
        assert len(ax.lines) == len(group['indices'])
        assert ax.get_title() == group['title']
        assert ax.get_legend() is not None

    plt.close(fig)


@pytest.mark.slow
@pytest.mark.slow
def test_save_figure(tmp_path):
    """测试 save_figure 方法能否正常保存图片"""
    plotter = PSDPlotter(str(PSD_FILE))

    # 先绘制一张图
    fig, ax = plt.subplots()
    plotter.plot_single(0, ax=ax)

    # 保存到临时文件
    output_path = tmp_path / "test_plot.png"
    plotter.save_figure(str(output_path))

    # 验证文件已生成且大小大于0
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(fig)


@pytest.mark.slow
def test_plot_groups_custom_colors():
    """测试 plot_groups 自定义颜色"""
    plotter = PSDPlotter(str(PSD_FILE))

    groups = [
        {"indices": [0, 1], "labels": ["Ex", "Ey"]}
    ]
    colors = ['#ff0000', '#00ff00']  # 红和绿

    fig, axes = plotter.plot_groups(groups, colors=colors)

    # 检查线条颜色是否正确
    ax = axes[0]
    for i, line in enumerate(ax.lines):
        assert line.get_color() == colors[i]

    plt.close(fig)