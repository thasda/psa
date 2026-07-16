# tests/test_read_mat_data.py
import pytest
import numpy as np
import pandas as pd
import h5py
import os
from datetime import datetime, timedelta
from pythonProject.src.utils.matdataprocessor import MatDataProcessor  # 根据实际路径调整

# 辅助函数：创建模拟的 HDF5 MAT 文件（包含 AttsExport）
def create_mock_atts_mat(filepath, n_samples=1000, n_channels=5, sf=1000.0):
    """创建一个包含 AttsExport 的模拟 MAT 文件（HDF5 格式）"""
    with h5py.File(filepath, 'w') as f:
        # 生成随机数据：n_channels 列，n_samples 行
        atts_data = np.random.randn(n_samples, n_channels).astype(np.float64)
        # MATLAB 通常存储为 通道×时间，但 HDF5 存储为 C-order，此处我们直接存储为 时间×通道，
        # 并在 _extract_from_h5 中会根据需要进行转置，但为了测试确定性，我们明确存储为 时间×通道，
        # 让代码逻辑自己决定是否转置。为了模拟真实情况，我们存储为 通道×时间（列优先形状），
        # 即 (n_channels, n_samples)，让读取代码转置为 (n_samples, n_channels)
        atts_data = atts_data.T  # 变为 (n_channels, n_samples)
        f.create_dataset('AttsExport', data=atts_data)
    return filepath

# 辅助函数：创建模拟 MAT 文件（直接包含 TSData 和 times）
def create_mock_direct_mat(filepath, n_samples=1000, n_channels=5, sf=1000.0, start_time=datetime(2024,1,1)):
    """创建一个直接包含 TSData 和 times 的模拟 MAT 文件（HDF5 格式）"""
    with h5py.File(filepath, 'w') as f:
        ts_data = np.random.randn(n_channels, n_samples).astype(np.float64)  # 通道×时间
        f.create_dataset('TSData', data=ts_data)
        # times 存储为从起始时间开始的秒数偏移
        times_sec = np.arange(n_samples) / sf
        f.create_dataset('times', data=times_sec)
    return filepath

class TestMatDataProcessor:
    """测试 MatDataProcessor 类"""

    def test_from_atts_to_dataframe(self, tmp_path):
        """测试从 AttsExport 构建 DataFrame"""
        mat_file = tmp_path / "mock_atts.mat"
        create_mock_atts_mat(str(mat_file), n_samples=500, n_channels=5, sf=1000)

        # 配置参数
        config = {
            'sample_rate': 1000,
            'start_time': datetime(2024, 1, 1, 0, 0, 0),
            'channel_count': 5
        }

        processor = MatDataProcessor(str(mat_file), **config)
        df = processor.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (500, 5)  # 时间×通道
        assert df.index.name == 'time'
        assert isinstance(df.index, pd.DatetimeIndex)
        # 验证时间范围大致正确（500个点，1000Hz，共0.5秒）
        expected_start = pd.Timestamp('2024-01-01 00:00:00')
        expected_end = pd.Timestamp('2024-01-01 00:00:00.499')
        assert df.index[0] == expected_start
        assert df.index[-1] == expected_end

    def test_from_direct_to_dataframe(self, tmp_path):
        """测试直接读取 TSData 和 times 构建 DataFrame"""
        mat_file = tmp_path / "mock_direct.mat"
        start_time = datetime(2024, 2, 1, 12, 0, 0)
        create_mock_direct_mat(str(mat_file), n_samples=300, n_channels=3, sf=500, start_time=start_time)

        config = {
            'sample_rate': 500,
            'start_time': start_time,
            'channel_count': 3
        }

        processor = MatDataProcessor(str(mat_file), **config)
        df = processor.to_dataframe()

        assert df.shape == (300, 3)
        # 验证时间正确（从 start_time 开始，间隔 2ms）
        expected_times = pd.date_range(start=start_time, periods=300, freq='2ms')
        expected_times.name = 'time'  # 设置名字以匹配 df.index
        pd.testing.assert_index_equal(df.index, expected_times)

    def test_to_hdf5_output(self, tmp_path):
        """测试 to_hdf5 方法能否正确写入文件"""
        mat_file = tmp_path / "mock_atts.mat"
        create_mock_atts_mat(str(mat_file), n_samples=200, n_channels=4, sf=100)

        config = {
            'sample_rate': 100,
            'start_time': '2024-03-01 00:00:00',  # 测试字符串转换
            'channel_count': 4
        }

        output_h5 = tmp_path / "output.h5"
        processor = MatDataProcessor(str(mat_file), **config)
        processor.to_hdf5(str(output_h5), chunk_size=50)

        # 验证文件存在并可读
        assert output_h5.exists()
        with h5py.File(output_h5, 'r') as f:
            assert 'TSData' in f
            assert 'times' in f
            assert f['TSData'].shape == (200, 4)
            # 验证元数据
            assert f.attrs['sample_rate'] == 100
            assert f.attrs['channel_count'] == 4
            # 时间存储为字符串，读取后检查长度
            times = f['times'][:]
            assert len(times) == 200

    def test_missing_config_raises(self, tmp_path):
        """测试缺少必要配置时抛出异常"""
        mat_file = tmp_path / "mock.mat"
        create_mock_atts_mat(str(mat_file))

        with pytest.raises(ValueError, match="缺少必要配置参数"):
            MatDataProcessor(str(mat_file))  # 未提供任何配置

        with pytest.raises(ValueError, match="缺少必要配置参数"):
            MatDataProcessor(str(mat_file), sample_rate=1000)  # 缺少 start_time 和 channel_count

    def test_invalid_mat_file_raises(self):
        """测试无效文件路径时抛出异常"""
        with pytest.raises(Exception):  # 可能是 FileNotFoundError 或 OSError
            processor = MatDataProcessor("non_existent.mat", sample_rate=1000, start_time=datetime.now(), channel_count=3)
            processor.to_dataframe()