"""
测试 WaveletService / WaveletResult 的完整功能
可以使用真实数据，也可通过生成的合成信号进行验证。
"""

import os
import tempfile
import shutil
import unittest
import h5py
import numpy as np

# 根据你的项目结构调整导入路径
from pythonProject.test_src_code.小波.Wavelet import WaveletService, WaveletResult, _cwt_channel

# ---------- 简易配置类，模拟 ConfigManager ----------
class MockConfig:
    def __init__(self, params):
        self._params = params
    def get(self, key, default=None):
        return self._params.get(key, default)
    def keys(self):
        return self._params.keys()

# ---------- 生成模拟多通道数据 ----------
def generate_test_data(fs=1000, duration=10, n_channels=3, freq_components=(5, 20, 80)):
    """生成带有已知频率成分的正弦波混合信号，并存入临时 HDF5 文件。"""
    t = np.arange(0, duration, 1/fs)
    data = np.zeros((len(t), n_channels))
    for ch in range(n_channels):
        # 每个通道叠加不同频率和幅值的正弦波
        for f in freq_components:
            amp = 1.0 / (ch+1)   # 不同通道幅值不同
            data[:, ch] += amp * np.sin(2 * np.pi * f * t + np.random.rand()*2*np.pi)
        # 加入微弱噪声
        data[:, ch] += 0.05 * np.random.randn(len(t))

    # 保存为临时 HDF5
    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, "test_data.h5")
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('signal', data=data)
        f.create_dataset('time', data=t)
        f.attrs['channels'] = [f'ch{i}' for i in range(n_channels)]
    return filepath, data, t, tmpdir

# ---------- 配置参数 ----------
def get_test_config(fs=1000, fmin=1, fmax=100, nfreqs=20):
    return MockConfig({
        '采样率 (Hz)': fs,
        '小波类型': 'morlet',
        '最小频率 (Hz)': fmin,
        '最大频率 (Hz)': fmax,
        '频率点数': nfreqs,
        '窗口长度 (秒)': 2.0,
        '重叠比例': 0.5
    })

class TestWaveletService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = None
        cls.config = get_test_config()
        # 生成测试数据
        cls.h5path, cls.data, cls.time, cls.tmpdir = generate_test_data(
            fs=int(cls.config.get('采样率 (Hz)')),
            duration=4, n_channels=2
        )

    @classmethod
    def tearDownClass(cls):
        if cls.tmpdir and os.path.exists(cls.tmpdir):
            shutil.rmtree(cls.tmpdir)

    def setUp(self):
        self.service = WaveletService(self.config)

    # -------- 测试数据加载 --------
    def test_load_data(self):
        self.service.load_data(self.h5path, dataset_name='signal', time_name='time')
        self.assertEqual(self.service._data.shape, self.data.shape)
        self.assertEqual(len(self.service._time), len(self.time))
        self.assertEqual(self.service.n_channels, self.data.shape[1])
        self.assertListEqual(self.service._channel_names, ['ch0', 'ch1'])

    # -------- 测试全局小波计算 --------
    def test_compute_full(self):
        self.service.load_data(self.h5path, dataset_name='signal', time_name='time')
        result = self.service.compute_full(channels=None, max_workers=2)
        self.assertIsInstance(result, WaveletResult)
        n_ch = self.data.shape[1]
        n_freqs = int(self.config.get('频率点数'))
        n_times = len(self.time)
        self.assertEqual(result.coeffs.shape, (n_ch, n_freqs, n_times))
        self.assertEqual(result.powers.shape, (n_ch, n_freqs, n_times))
        self.assertEqual(result.phases.shape, (n_ch, n_freqs, n_times))
        self.assertEqual(result.coi_masks.shape, (n_ch, n_freqs, n_times))
        # 检查 COI 掩码：边缘应为 True，中间大部分为 False
        mid = n_times // 2
        self.assertTrue(result.coi_masks[0, -1, 0])    # 时间起点，低频受影响
        self.assertTrue(result.coi_masks[0, -1, -1])   # 时间终点，低频受影响
        self.assertFalse(result.coi_masks[0, 0, mid])  # 高频中间不受影响（如果频率范围合适）

    # -------- 测试分窗计算 --------
    def test_compute_windowed(self):
        self.service.load_data(self.h5path, dataset_name='signal', time_name='time')
        result = self.service.compute_windowed(channels=None, max_workers=2)
        self.assertIsInstance(result, WaveletResult)
        n_ch = self.data.shape[1]
        n_freqs = int(self.config.get('频率点数'))
        # 拼接后时间长度可能与原始略有差异，但应在合理范围
        self.assertEqual(result.coeffs.shape[0], n_ch)
        self.assertEqual(result.coeffs.shape[1], n_freqs)
        # 长度应略小于原始时间（因为去除了重叠边缘）
        self.assertLessEqual(result.coeffs.shape[2], len(self.time))
        self.assertGreater(result.coeffs.shape[2], len(self.time)*0.5)  # 至少有原始一半以上

    # -------- 测试保存与重新加载 --------
    def test_save_and_reload(self):
        self.service.load_data(self.h5path, dataset_name='signal', time_name='time')
        result = self.service.compute_full(max_workers=1)
        out_dir = tempfile.mkdtemp()
        try:
            saved_path = result.to_hdf5(out_dir, prefix='test')
            self.assertTrue(os.path.exists(saved_path))
            # 重新加载并检查
            with h5py.File(saved_path, 'r') as f:
                self.assertIn('time', f)
                self.assertIn('freqs', f)
                self.assertIn('coeffs', f)
                self.assertIn('powers', f)
                self.assertIn('phases', f)
                self.assertIn('coi_masks', f)
                self.assertListEqual(list(f.attrs['channel_names']), ['ch0', 'ch1'])
                # 验证数据一致性（近似）
                np.testing.assert_array_almost_equal(f['time'][:], result.time)
                np.testing.assert_array_almost_equal(f['freqs'][:], result.freqs)
                np.testing.assert_array_almost_equal(f['powers'][:], result.powers)
        finally:
            shutil.rmtree(out_dir)

    # -------- 测试一站式 run_and_save --------
    def test_run_and_save(self):
        out_dir = tempfile.mkdtemp()
        try:
            result_path = self.service.run_and_save(
                hdf5_input=self.h5path,
                output_path=out_dir,
                prefix='oneclick',
                mode='full',
                max_workers=2,
                dataset_name='signal',
                time_name='time'
            )
            self.assertTrue(os.path.exists(result_path))
            with h5py.File(result_path, 'r') as f:
                self.assertEqual(f['coeffs'].shape[0], self.data.shape[1])  # 通道数一致
        finally:
            shutil.rmtree(out_dir)

    # -------- 测试特定通道选择 --------
    def test_channel_selection(self):
        self.service.load_data(self.h5path, dataset_name='signal', time_name='time')
        result = self.service.compute_full(channels=['ch0'], max_workers=1)
        self.assertEqual(result.coeffs.shape[0], 1)
        self.assertListEqual(result.channel_names, ['ch0'])

    # -------- 测试 _cwt_channel 独立函数 --------
    def test_cwt_channel_function(self):
        data = self.data[:, 0]
        n_freqs = int(self.config.get('频率点数'))
        freqs = np.logspace(np.log10(1), np.log10(100), n_freqs)
        W, power, phase, mask = _cwt_channel(data, self.time, None, 1000, 'morlet', freqs)
        self.assertEqual(W.shape, (n_freqs, len(data)))
        self.assertTrue(np.iscomplexobj(W))
        self.assertEqual(power.shape, W.shape)
        self.assertEqual(phase.shape, W.shape)
        self.assertEqual(mask.shape, W.shape)
        self.assertEqual(mask.dtype, bool)

if __name__ == '__main__':
    unittest.main()