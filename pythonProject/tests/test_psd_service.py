# tests/test_psd_service.py
"""
测试 PSDCalculator 类，使用真实预处理后的 HDF5 数据。
需要以下文件存在：
- F:\Anylysis_project\pythonProject\tests\test_data\1804323_300.cfg
- F:\Anylysis_project\pythonProject\output\1804323_300_preprocessed.h5
"""

import pytest
import h5py
import numpy as np
from pathlib import Path

from pythonProject.configs.configmanager import ConfigManager
from pythonProject.src.services.psd_service import PSDCalculator

# 真实数据路径
DATA_DIR = Path("F:/Anylysis_project/pythonProject/output")
CFG_PATH = Path("F:/Anylysis_project/pythonProject/tests/test_data/Ground.cfg")
H5_PATH = DATA_DIR / "Ground_preprocessed.h5"
OUTPUT_PATH = DATA_DIR / "Ground_preprocessed_psd.h5"


@pytest.mark.slow
def test_psd_calculator_real_data():
    """使用真实预处理数据测试 PSD 计算（默认输出路径）"""
    # 1. 检查必需文件是否存在
    assert CFG_PATH.exists(), f"配置文件不存在: {CFG_PATH}"
    assert H5_PATH.exists(), f"预处理数据文件不存在: {H5_PATH}"

    # 2. 加载配置
    cfg_mgr = ConfigManager.from_aether_cfg(str(CFG_PATH))

    # 3. 创建计算器
    calculator = PSDCalculator(str(H5_PATH), cfg_mgr)

    # 4. 计算 PSD（不指定输出路径，使用默认文件名）
    result = calculator.compute_psd()

    # 5. 验证返回结果
    assert 'freqs' in result, "返回结果缺少 'freqs'"
    assert 'psd' in result, "返回结果缺少 'psd'"
    freqs = result['freqs']
    psd = result['psd']

    assert isinstance(freqs, np.ndarray), "freqs 应为 numpy 数组"
    assert freqs.ndim == 1, "freqs 应为一维"
    assert len(freqs) > 0, "频率数组不应为空"
    assert freqs[0] >= 0, "频率应从 0 或正数开始"

    # 检查 PSD 形状：应为 (通道数, 频率点数)
    with h5py.File(H5_PATH, 'r') as f:
        n_channels = f['TSData'].shape[1]
    assert psd.shape == (n_channels, len(freqs)), f"PSD 形状应为 ({n_channels}, {len(freqs)})"

    # 6. 验证默认输出文件存在
    default_output = H5_PATH.with_name(H5_PATH.stem + "_psd.h5")
    assert default_output.exists(), f"默认输出文件未生成: {default_output}"

    # 7. 验证输出文件内容（考虑 float32 与 float64 精度差异）
    with h5py.File(default_output, 'r') as f:
        assert 'freqs' in f, "输出文件缺少 freqs 数据集"
        assert 'psd' in f, "输出文件缺少 psd 数据集"
        # 使用近似比较代替严格相等
        np.testing.assert_allclose(f['freqs'][:], freqs, rtol=1e-5, atol=1e-8,
                                   err_msg="输出文件的 freqs 与返回结果不一致")
        np.testing.assert_allclose(f['psd'][:], psd, rtol=1e-5, atol=1e-8,
                                   err_msg="输出文件的 psd 与返回结果不一致")
        assert f.attrs['sample_rate'] == calculator.sample_rate, "采样率属性错误"
        assert f.attrs['nperseg'] == calculator.nperseg, "nperseg 属性错误"
        assert f.attrs['window'] == 'hamming', "window 属性错误"

    # 可选：清理生成的文件（默认注释掉，以保留结果供人工检查）
    # default_output.unlink()


@pytest.mark.slow
def test_psd_calculator_with_output_path():
    """测试指定输出路径"""
    assert CFG_PATH.exists()
    assert H5_PATH.exists()

    cfg_mgr = ConfigManager.from_aether_cfg(str(CFG_PATH))
    calculator = PSDCalculator(str(H5_PATH), cfg_mgr)

    import tempfile
    tmp_output = Path(tempfile.gettempdir()) / "test_psd_output.h5"
    result = calculator.compute_psd(str(tmp_output))

    # 验证文件生成
    assert tmp_output.exists(), f"临时输出文件未生成: {tmp_output}"

    # 验证内容（使用近似比较）
    with h5py.File(tmp_output, 'r') as f:
        assert 'freqs' in f
        assert 'psd' in f
        np.testing.assert_allclose(f['freqs'][:], result['freqs'], rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(f['psd'][:], result['psd'], rtol=1e-5, atol=1e-8)

    # 清理临时文件
    tmp_output.unlink()


@pytest.mark.slow
def test_psd_calculator_missing_config_key():
    """测试缺少必要配置参数时的异常"""
    # 模拟缺少 FFT窗口长度 的配置
    from unittest.mock import MagicMock
    mock_cfg = MagicMock()
    mock_cfg.get.side_effect = lambda key, default=None: {
        "采样率 (Hz)": 1000,
        # "FFT窗口长度" 缺失
    }.get(key, default)

    with pytest.raises(ValueError, match="缺少 FFT窗口长度"):
        PSDCalculator(str(H5_PATH), mock_cfg)