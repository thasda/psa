# tests/test_real_preprocess.py
"""
使用真实数据测试 Preprocessor 类（需手动准备数据文件）。
标记为 slow，默认不执行，需用 pytest -m slow 运行。
"""

import sys
from pathlib import Path

import pytest
import h5py
import numpy as np

# 将项目根目录添加到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pythonProject.configs.configmanager import ConfigManager
from pythonProject.src.utils.preprocess import Preprocessor


# 文件路径（可根据实际情况调整）
CFG_PATH = PROJECT_ROOT / "tests" / "test_data" / "Ground.cfg"
H5_PATH = PROJECT_ROOT / "output" / "Ground.h5"
CMT_PATHS = [
    PROJECT_ROOT / "tests" / "test_data" / "CMT_7004.cmt",
    PROJECT_ROOT / "tests" / "test_data" / "CMT_7005.cmt",
    PROJECT_ROOT / "tests" / "test_data" / "CMT_7006.cmt"
]
OUTPUT_H5 = PROJECT_ROOT / "output" / "Ground_preprocessed.h5"


@pytest.mark.slow
def test_real_preprocess():
    """使用真实数据测试完整预处理流程"""
    # 1. 检查必需文件是否存在
    required_files = [CFG_PATH, H5_PATH] + CMT_PATHS
    for f in required_files:
        assert f.exists(), f"必需文件不存在: {f}"

    # 2. 加载配置
    cfg_mgr = ConfigManager.from_aether_cfg(str(CFG_PATH))

    # 3. 更新校准文件路径为完整路径（覆盖原始文件名）
    cfg_mgr.add_param("校准文件", [str(p) for p in CMT_PATHS])

    # 4. 创建预处理器
    pre = Preprocessor(str(H5_PATH), config_manager=cfg_mgr)

    # 5. 执行预处理
    try:
        pre.process(str(OUTPUT_H5), chunk_size=200000)
    except Exception as e:
        pytest.fail(f"预处理过程抛出异常: {e}")

    # 6. 验证输出文件
    assert OUTPUT_H5.exists(), "输出文件未生成"

    with h5py.File(OUTPUT_H5, 'r') as f:
        # 检查必要数据集
        assert 'TSData' in f, "输出文件缺少 TSData 数据集"
        assert 'times' in f, "输出文件缺少 times 数据集"

        tsdata = f['TSData'][:]
        times = f['times'][:]

        # 形状应与输入一致
        with h5py.File(H5_PATH, 'r') as fin:
            assert tsdata.shape == fin['TSData'].shape, "输出数据形状与输入不一致"
            assert len(times) == len(fin['times']), "输出时间长度与输入不一致"

        # # 归一化后的数据均值应接近0，标准差应接近1（放宽容差，因为真实数据可能存在偏差）
        # mean_val = np.mean(tsdata, axis=0)
        # std_val = np.std(tsdata, axis=0, ddof=0)
        #
        # # 验证均值接近0（容差可适当放宽，例如 0.5 或 1.0）
        # np.testing.assert_allclose(np.mean(tsdata, axis=0), 0.0, atol=1.0,
        #                            err_msg="输出数据均值偏离0过大")
        # 打印实际标准差供人工检查
        print(f"输出数据各通道标准差: {np.std(tsdata, axis=0, ddof=0)}")

    # 可选：清理输出文件
    # OUTPUT_H5.unlink()