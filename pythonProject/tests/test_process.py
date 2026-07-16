# tests/test_process.py
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
TEST_DIR = Path(__file__).parent          # tests 目录
PROJECT_ROOT = TEST_DIR.parent             # 项目根目录
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import h5py
from unittest.mock import MagicMock

from pythonProject.src.utils.preprocess import Preprocessor
from pythonProject.configs.configmanager import ConfigManager


@pytest.fixture
def mock_config_manager():
    mock = MagicMock(spec=ConfigManager)
    config = {
        "采样率 (Hz)": 1000,
        "通道数量": 5,
        "通道索引": [1, 2, 3, 4, 5],
        "通道增益": [1.0, 1.0, 1.0, 1.0, 1.0],
        "电长度 (米)": [50.0, 31.0],
        "校准文件": ["HX.cmt", "HY.cmt", "HZ.cmt"],
    }
    mock.get.side_effect = lambda key, default=None: config.get(key, default)
    return mock


@pytest.fixture
def sample_hdf5_file(tmp_path):
    """创建模拟的 HDF5 输入文件，包含 TSData 和 times"""
    h5_path = tmp_path / "test_input.h5"
    n_samples = 10000
    n_channels = 5
    fs = 1000
    t = np.arange(n_samples) / fs

    data = np.zeros((n_samples, n_channels))
    for ch in range(n_channels):
        data[:, ch] = 2.0 * np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(n_samples)
    trend = 0.01 * t[:, np.newaxis]
    data += trend

    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('TSData', data=data, chunks=True, compression='gzip')
        f.create_dataset('times', data=t)
        f.attrs['sample_rate'] = fs
    return h5_path


@pytest.fixture
def sample_cmt_files(tmp_path):
    paths = []
    for name in ["HX.cmt", "HY.cmt", "HZ.cmt"]:
        p = tmp_path / name
        freqs = np.logspace(0, 3, 20)
        amps = np.ones_like(freqs) * 0.8
        phases = np.zeros_like(freqs)
        np.savetxt(p, np.column_stack((freqs, amps, phases)), fmt='%.6f')
        paths.append(str(p))
    return paths


@pytest.fixture
def preprocessor_with_mocks(tmp_path, sample_hdf5_file, mock_config_manager, sample_cmt_files):
    """返回 Preprocessor 实例，校准器使用真实文件，但后续可 mock 方法"""
    # 更新配置中的校准文件路径
    mock_config_manager.get.side_effect = lambda key, default=None: {
        "采样率 (Hz)": 1000,
        "通道数量": 5,
        "通道索引": [1, 2, 3, 4, 5],
        "通道增益": [1.0, 1.0, 1.0, 1.0, 1.0],
        "电长度 (米)": [50.0, 31.0],
        "校准文件": sample_cmt_files,
    }.get(key, default)

    pre = Preprocessor(str(sample_hdf5_file), mock_config_manager)

    # 将校准方法替换为恒等映射的 mock（避免真实 FFT 计算，并保留调用记录）
    for cal in [pre.cal_hx, pre.cal_hy, pre.cal_hz]:
        if cal:
            cal.calibrate_time_series = MagicMock(side_effect=lambda data, fs, method='fft': data)
    return pre


def test_preprocessor_init(mock_config_manager):
    pre = Preprocessor("dummy.h5", mock_config_manager)
    assert pre.sample_rate == 1000
    assert pre.channel_count == 5
    assert pre.electrical_lengths == [50.0, 31.0]
    assert pre.calibration_files == ["HX.cmt", "HY.cmt", "HZ.cmt"]


def test_compute_global_stats(preprocessor_with_mocks):
    pre = preprocessor_with_mocks
    with h5py.File(pre.hdf5_path, 'r') as f:
        data = f['TSData'][:]
        t = f['times'][:]

    pre._compute_global_stats()

    n = pre.n_samples
    exp_mean = np.mean(data, axis=0)
    exp_std = np.std(data, axis=0, ddof=0)
    sum_x = np.sum(t)
    sum_x2 = np.sum(t**2)
    sum_y = np.sum(data, axis=0)
    sum_xy = np.sum(t[:, np.newaxis] * data, axis=0)
    denom = n * sum_x2 - sum_x**2
    if denom != 0:
        exp_slope = (n * sum_xy - sum_x * sum_y) / denom
    else:
        exp_slope = np.zeros(data.shape[1])
    exp_intercept = (sum_y - exp_slope * sum_x) / n

    np.testing.assert_allclose(pre.mean, exp_mean, rtol=1e-5)
    np.testing.assert_allclose(pre.std, exp_std, rtol=1e-5)
    np.testing.assert_allclose(pre.slope, exp_slope, rtol=1e-5)
    np.testing.assert_allclose(pre.intercept, exp_intercept, rtol=1e-5)


def test_correct_electrical_length(preprocessor_with_mocks):
    pre = preprocessor_with_mocks
    block = np.ones((10, 5))
    corrected = pre.correct_electrical_length(block.copy())
    np.testing.assert_allclose(corrected[:, 0], 1.0 / 50.0)
    np.testing.assert_allclose(corrected[:, 1], 1.0 / 31.0)
    np.testing.assert_allclose(corrected[:, 2:], block[:, 2:])


def test_detrend(preprocessor_with_mocks):
    pre = preprocessor_with_mocks
    pre._compute_global_stats()

    with h5py.File(pre.hdf5_path, 'r') as f:
        data = f['TSData'][:]
        x = f['times'][:]

    detrended = pre.detrend(data, x)

    for ch in range(data.shape[1]):
        slope, _ = np.polyfit(x, detrended[:, ch], 1)
        assert abs(slope) < 1e-6, f"通道{ch}去趋势后斜率{slope}过大"


def test_normalize(preprocessor_with_mocks):
    pre = preprocessor_with_mocks
    pre._compute_global_stats()

    with h5py.File(pre.hdf5_path, 'r') as f:
        data = f['TSData'][:]

    normalized = pre.normalize(data)

    np.testing.assert_allclose(np.mean(normalized, axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.std(normalized, axis=0, ddof=0), 1.0, atol=1e-6)


def test_notch_filter_design(preprocessor_with_mocks):
    pre = preprocessor_with_mocks
    b_list, a_list = pre._design_notch_filters()
    assert len(b_list) == len(pre.harmonics)
    assert len(a_list) == len(pre.harmonics)
    assert all(len(b) == 3 for b in b_list)
    assert all(len(a) == 3 for a in a_list)


def test_notch_filter_removes_harmonic(preprocessor_with_mocks):
    pre = preprocessor_with_mocks
    fs = 1000
    t = np.arange(0, 2, 1/fs)
    f0 = 50.0
    signal = np.sin(2 * np.pi * f0 * t)
    block = signal.reshape(-1, 1)

    pre.sample_rate = fs
    b_list, a_list = pre._design_notch_filters()

    from scipy.signal import filtfilt
    filtered = block.copy()
    for b, a in zip(b_list, a_list):
        filtered = filtfilt(b, a, filtered, axis=0, padlen=150)

    fft_before = np.abs(np.fft.rfft(block[:, 0]))
    fft_after = np.abs(np.fft.rfft(filtered[:, 0]))
    freqs = np.fft.rfftfreq(len(block), 1/fs)
    idx = np.argmin(np.abs(freqs - 50))
    assert fft_after[idx] < 0.1 * fft_before[idx]


def test_remove_instrument_response_called(preprocessor_with_mocks):
    pre = preprocessor_with_mocks
    block = np.random.randn(100, 5)
    for cal in [pre.cal_hx, pre.cal_hy, pre.cal_hz]:
        cal.calibrate_time_series.reset_mock()

    pre.remove_instrument_response(block)

    # HX
    pre.cal_hx.calibrate_time_series.assert_called_once()
    args, kwargs = pre.cal_hx.calibrate_time_series.call_args
    np.testing.assert_array_equal(args[0], block[:, 2])
    assert args[1] == pre.sample_rate
    assert kwargs.get('method') == 'fft'

    # HY
    pre.cal_hy.calibrate_time_series.assert_called_once()
    args, kwargs = pre.cal_hy.calibrate_time_series.call_args
    np.testing.assert_array_equal(args[0], block[:, 3])
    assert args[1] == pre.sample_rate
    assert kwargs.get('method') == 'fft'

    # HZ
    pre.cal_hz.calibrate_time_series.assert_called_once()
    args, kwargs = pre.cal_hz.calibrate_time_series.call_args
    np.testing.assert_array_equal(args[0], block[:, 4])
    assert args[1] == pre.sample_rate
    assert kwargs.get('method') == 'fft'


def test_process_integration(tmp_path, preprocessor_with_mocks):
    pre = preprocessor_with_mocks
    output_path = tmp_path / "output.h5"
    chunk_size = 2000

    pre.process(str(output_path), chunk_size=chunk_size)

    assert output_path.exists()

    with h5py.File(output_path, 'r') as f:
        out_data = f['TSData'][:]

        # 验证均值接近0（放宽容差至0.1）
        np.testing.assert_allclose(np.mean(out_data, axis=0), 0.0, atol=0.1)
        # 验证标准差接近1（放宽容差至0.1）
        np.testing.assert_allclose(np.std(out_data, axis=0, ddof=0), 1.0, atol=0.1)


def test_missing_calibration_files(tmp_path, mock_config_manager, sample_hdf5_file):
    mock_config_manager.get.side_effect = lambda key, default=None: {
        "采样率 (Hz)": 1000,
        "通道数量": 5,
        "通道索引": [1, 2, 3, 4, 5],
        "通道增益": [1.0, 1.0, 1.0, 1.0, 1.0],
        "电长度 (米)": [50.0, 31.0],
        "校准文件": ["HX.cmt"],  # 只有一个文件
    }.get(key, default)

    pre = Preprocessor(str(sample_hdf5_file), mock_config_manager)
    assert pre.cal_hx is None
    assert pre.cal_hy is None
    assert pre.cal_hz is None

    output_path = tmp_path / "output_no_cal.h5"
    pre.process(str(output_path))   # 应正常完成
    assert output_path.exists()