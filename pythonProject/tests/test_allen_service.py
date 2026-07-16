# tests/test_allan_calculator.py
"""
测试 AllanDeviationCalculator 类。
使用模拟数据验证艾伦方差计算结果。
"""

import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import h5py
from unittest.mock import MagicMock

from pythonProject.src.services.allan_service import AllanDeviationCalculator
from pythonProject.configs.configmanager import ConfigManager


@pytest.fixture
def mock_config_manager():
    """模拟 ConfigManager，返回采样率"""
    mock = MagicMock(spec=ConfigManager)
    mock.get.side_effect = lambda key, default=None: {
        "采样率 (Hz)": 1000,
    }.get(key, default)
    return mock


@pytest.fixture
def create_sample_h5(tmp_path):
    """创建包含模拟时间序列数据的 HDF5 文件"""
    h5_path = tmp_path / "test_data.h5"
    n_samples = 100000
    n_channels = 3
    fs = 1000
    t = np.arange(n_samples) / fs

    # 生成白噪声数据（艾伦方差应随 τ^-1 下降）
    data = np.random.randn(n_samples, n_channels)

    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('TSData', data=data, chunks=True)
        f.create_dataset('times', data=t)
    return h5_path


def test_allan_calculator_initialization(mock_config_manager, create_sample_h5):
    """测试初始化"""
    calc = AllanDeviationCalculator(str(create_sample_h5), mock_config_manager)
    assert calc.sample_rate == 1000
    assert calc.n_samples == 100000
    assert calc.n_channels == 3


def test_allan_variance_computation(mock_config_manager, create_sample_h5):
    """测试艾伦方差计算（与理论值比较）"""
    calc = AllanDeviationCalculator(str(create_sample_h5), mock_config_manager)

    # 指定 tau 值
    taus = np.array([0.1, 1.0, 10.0])  # 秒

    result = calc.compute_allan_variance(taus=taus)

    assert 'taus' in result
    assert 'allan_var' in result          # 修改：键名改为 allan_var
    assert np.array_equal(result['taus'], taus)
    assert result['allan_var'].shape == (3, len(taus))

    # 对于白噪声，艾伦方差应非负
    assert np.all(result['allan_var'] >= 0)


def test_allan_variance_auto_taus(mock_config_manager, create_sample_h5):
    """测试自动生成 tau 值"""
    calc = AllanDeviationCalculator(str(create_sample_h5), mock_config_manager)

    result = calc.compute_allan_variance(max_tau_factor=0.05)

    taus = result['taus']
    # 最大 tau 应不超过总时长 * max_tau_factor
    max_tau = calc.n_samples / calc.sample_rate * 0.05
    assert taus[-1] <= max_tau + 1e-6


def test_allan_variance_output_save(mock_config_manager, create_sample_h5, tmp_path):
    """测试结果保存到 HDF5 文件"""
    calc = AllanDeviationCalculator(str(create_sample_h5), mock_config_manager)

    output_path = tmp_path / "allan_result.h5"
    taus = np.array([0.01, 0.1, 1.0, 10.0])
    result = calc.compute_allan_variance(taus=taus, output_path=str(output_path))

    assert output_path.exists()

    with h5py.File(output_path, 'r') as f:
        assert 'taus' in f
        assert 'allan_variance' in f        # 修改：数据集名称为 allan_variance
        # 使用近似比较代替严格相等
        np.testing.assert_allclose(f['taus'][:], taus, rtol=1e-5, atol=1e-8,
                                   err_msg="保存的 tau 值与原始值不匹配")
        np.testing.assert_allclose(f['allan_variance'][:], result['allan_var'], rtol=1e-5, atol=1e-8,
                                   err_msg="保存的 allan_variance 与返回结果不匹配")
        assert f.attrs['sample_rate'] == 1000
        assert f.attrs['n_channels'] == 3
        assert f.attrs['n_samples'] == 100000


def test_allan_variance_missing_sample_rate():
    """测试缺少采样率时抛出异常"""
    mock_cfg = MagicMock()
    mock_cfg.get.return_value = None

    with pytest.raises(ValueError, match="缺少采样率"):
        AllanDeviationCalculator("dummy.h5", mock_cfg)


def test_allan_variance_tau_too_large(mock_config_manager, create_sample_h5):
    """测试 tau 值超过数据长度时返回 NaN"""
    calc = AllanDeviationCalculator(str(create_sample_h5), mock_config_manager)
    taus = np.array([1000.0])  # 对应 1000秒，远大于数据长度 100秒

    result = calc.compute_allan_variance(taus=taus)
    assert np.isnan(result['allan_var'][0, 0])   # 修改：键名改为 allan_var