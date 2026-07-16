# tests/test_allan_plotter.py
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pythonProject.src.plot.plot_allan import AllanPlotter


@pytest.fixture
def create_sample_allan(tmp_path):
    """创建模拟的艾伦方差 HDF5 文件"""
    file_path = tmp_path / "test_allan.h5"
    n_channels = 5
    n_taus = 50
    taus = np.logspace(-2, 2, n_taus)
    # 生成随机的艾伦标准差数据（白噪声理论值应为 1/sqrt(tau) 量级）
    allan_dev = np.random.randn(n_channels, n_taus) * 0.1 + 1.0 / np.sqrt(taus)

    with h5py.File(file_path, 'w') as f:
        f.create_dataset('taus', data=taus)
        f.create_dataset('allan_dev', data=allan_dev)
        f.attrs['sample_rate'] = 1000
        f.attrs['n_channels'] = n_channels
    return file_path


def test_allan_plotter_initialization(create_sample_allan):
    plotter = AllanPlotter(str(create_sample_allan))
    assert plotter.taus is not None
    assert plotter.allan_dev is not None
    assert plotter.n_channels == 5
    assert plotter.sample_rate == 1000


def test_plot_single(create_sample_allan):
    plotter = AllanPlotter(str(create_sample_allan))
    fig, ax = plt.subplots()
    returned_ax = plotter.plot_single(0, ax=ax, color='blue')
    assert returned_ax is ax
    assert len(ax.lines) == 1
    plt.close(fig)


def test_plot_all(create_sample_allan):
    plotter = AllanPlotter(str(create_sample_allan))
    fig, ax = plt.subplots()
    returned_ax = plotter.plot_all(ax=ax)
    assert len(ax.lines) == plotter.n_channels
    plt.close(fig)


def test_plot_groups(create_sample_allan):
    plotter = AllanPlotter(str(create_sample_allan))
    groups = [
        {"indices": [0, 1], "labels": ["Ex", "Ey"], "title": "Electric"},
        {"indices": [2, 3, 4], "labels": ["Bx", "By", "Bz"], "title": "Magnetic"}
    ]
    fig, axes = plotter.plot_groups(groups)
    assert len(axes) == 2
    assert len(axes[0].lines) == 2
    assert len(axes[1].lines) == 3
    plt.close(fig)


def test_save_figure(create_sample_allan, tmp_path):
    plotter = AllanPlotter(str(create_sample_allan))
    fig, ax = plt.subplots()
    plotter.plot_single(0, ax=ax)
    output = tmp_path / "test.png"
    plotter.save_figure(str(output))
    assert output.exists()
    assert output.stat().st_size > 0
    plt.close(fig)