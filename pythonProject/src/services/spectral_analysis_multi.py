"""
spectral_analysis_multi.py

支持多数据源的电磁场频域耦合分析：
- 可同时载入多个 HDF5 文件（预处理后格式）
- 指定任意通道对（可来自不同文件）
- 自动时间对齐（截取公共时间区间）
- 计算互功率谱、相干谱、相位谱
"""

import numpy as np
import h5py
from scipy import signal
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox


# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------

def _parse_channel_names(attrs: dict) -> List[str]:
    """从 HDF5 属性中解析通道名称列表"""
    names = attrs.get('channel_names')
    if names is None:
        for key in ['channels', 'channel_names', 'channelNames']:
            if key in attrs:
                names = attrs[key]
                break
    if names is None:
        raise ValueError("未找到通道名称属性")
    if isinstance(names, (bytes, np.bytes_)):
        names = names.decode()
    if isinstance(names, str):
        names = names.split(',') if ',' in names else [names]
    if isinstance(names, np.ndarray):
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    return list(names)


def _load_hdf5_data(hdf5_path: str) -> Dict:
    """
    加载 HDF5 文件，返回包含数据、通道名、采样率、时间轴和属性的字典。
    """
    with h5py.File(hdf5_path, 'r') as f:
        data = f['TSData'][:]
        attrs = dict(f.attrs)
        # 时间轴（ISO 字符串）
        times_raw = f['times'][:]
        if times_raw.dtype.kind in 'US':
            times = [t.decode() if isinstance(t, bytes) else t for t in times_raw]
        else:
            raise ValueError("times 必须是字符串格式")
        # 采样率
        fs = attrs.get('sample_rate')
        if fs is None:
            fs = attrs.get('采样率(Hz)')
        if fs is None:
            fs = 1.0
        fs = float(fs)
        channel_names = _parse_channel_names(attrs)
    return {
        'data': data,
        'channels': channel_names,
        'fs': fs,
        'times': times,
        'attrs': attrs
    }


def _align_times(sources: List[Dict]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    根据各数据源的时间轴，截取所有源共有的时间区间。
    返回：公共时间轴（datetime对象列表）和各源截取后的数据数组列表。
    """
    # 将所有时间字符串转为 datetime 对象
    dt_sources = []
    for src in sources:
        dt_list = [datetime.fromisoformat(t.replace('Z', '+00:00')) for t in src['times']]
        dt_sources.append(np.array(dt_list))

    # 计算所有源的最大起始时间和最小结束时间（取交集）
    starts = [dt[0] for dt in dt_sources]
    ends = [dt[-1] for dt in dt_sources]
    t_start = max(starts)
    t_end = min(ends)

    if t_start >= t_end:
        raise ValueError("各数据源的时间区间无重叠，无法进行时域对齐")

    aligned_data = []
    for dt, data in zip(dt_sources, [src['data'] for src in sources]):
        # 找到 t_start 和 t_end 在 dt 中的索引
        idx_start = np.searchsorted(dt, t_start)
        idx_end = np.searchsorted(dt, t_end, side='right')  # 包含 t_end 的最后一个点
        if idx_end - idx_start < 2:
            raise ValueError("对齐后数据点过少，请检查时间重叠长度")
        aligned_data.append(data[idx_start:idx_end, :])

    # 返回对齐后的数据（每个源一个数组）和公共时间轴（用第一个源的时间）
    common_times = dt_sources[0][idx_start:idx_end]
    return common_times, aligned_data


# -----------------------------------------------------------------------------
# 核心类：支持多源互谱分析
# -----------------------------------------------------------------------------

class MultiSourceSpectrum:
    """
    多源频谱分析类，支持从多个 HDF5 文件中选取任意通道对进行分析。
    """

    def __init__(self, source_paths: List[str],
                 nperseg: int = 256, noverlap: Optional[int] = None,
                 nfft: Optional[int] = None, window: str = 'hann',
                 detrend: str = 'constant'):
        """
        参数:
            source_paths : 数据源文件路径列表，每个文件需为预处理后 HDF5 格式。
            其他参数同 scipy.signal.csd。
        """
        self.source_paths = source_paths
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap is not None else nperseg // 2
        self.nfft = nfft if nfft is not None else nperseg
        self.window = window
        self.detrend = detrend

        # 加载所有数据源
        self.sources = [_load_hdf5_data(p) for p in source_paths]
        # 统一采样率检查
        fs_set = {src['fs'] for src in self.sources}
        if len(fs_set) != 1:
            raise ValueError(f"所有数据源的采样率必须相同，当前为 {fs_set}")
        self.fs = fs_set.pop()

        # 时间对齐
        self.common_times, self.aligned_data = _align_times(self.sources)
        self.n_samples = self.aligned_data[0].shape[0]

        # 通道名称映射：为每个源添加索引前缀以便区分
        self.channel_map = {}  # 键：'源索引:通道名' 或 '文件名:通道名'
        for i, src in enumerate(self.sources):
            base_name = Path(self.source_paths[i]).stem
            for ch in src['channels']:
                key1 = f"{i}:{ch}"
                key2 = f"{base_name}:{ch}"
                self.channel_map[key1] = (i, ch)
                self.channel_map[key2] = (i, ch)
        # 也保留不带前缀的通道名，但若有重名则只保留第一个（警告）
        for i, src in enumerate(self.sources):
            for ch in src['channels']:
                if ch not in self.channel_map:
                    self.channel_map[ch] = (i, ch)
                else:
                    print(f"警告：通道名 '{ch}' 在多个源中存在，请使用 '源索引:通道名' 明确指定")

    def get_channel_data(self, channel_spec: str) -> np.ndarray:
        """根据通道规格（如 '0:EX' 或 'EX'）返回对齐后的数据列"""
        if channel_spec not in self.channel_map:
            raise ValueError(f"未知通道规格 '{channel_spec}'，可用规格: {list(self.channel_map.keys())}")
        src_idx, ch_name = self.channel_map[channel_spec]
        ch_idx = self.sources[src_idx]['channels'].index(ch_name)
        return self.aligned_data[src_idx][:, ch_idx]

    def compute_csd(self, pairs: List[Tuple[str, str]]) -> Dict:
        """
        计算指定通道对的互功率谱密度。

        pairs: 列表，每个元素为 (通道规格1, 通道规格2)
        返回：字典，键为 'ch1_ch2'，包含频率、互谱、通道信息
        """
        results = {}
        for ch1, ch2 in pairs:
            x = self.get_channel_data(ch1)
            y = self.get_channel_data(ch2)
            f, Pxy = signal.csd(x, y, fs=self.fs, window=self.window,
                                nperseg=self.nperseg, noverlap=self.noverlap,
                                nfft=self.nfft, detrend=self.detrend)
            key = f"{ch1}_{ch2}"
            results[key] = {
                'freq': f,
                'Pxy': Pxy,
                'ch1': ch1,
                'ch2': ch2,
                'source1': self.channel_map[ch1][0],
                'source2': self.channel_map[ch2][0]
            }
        return results

    def compute_coherence(self, pairs: List[Tuple[str, str]]) -> Dict:
        """计算相干谱"""
        results = {}
        for ch1, ch2 in pairs:
            x = self.get_channel_data(ch1)
            y = self.get_channel_data(ch2)
            f, Cxy = signal.coherence(x, y, fs=self.fs, window=self.window,
                                      nperseg=self.nperseg, noverlap=self.noverlap,
                                      nfft=self.nfft, detrend=self.detrend)
            key = f"{ch1}_{ch2}"
            results[key] = {
                'freq': f,
                'coherence': Cxy,
                'ch1': ch1,
                'ch2': ch2
            }
        return results

    def compute_phase(self, pairs: List[Tuple[str, str]]) -> Dict:
        """计算相位谱（由互谱的相位角）"""
        results = {}
        for ch1, ch2 in pairs:
            x = self.get_channel_data(ch1)
            y = self.get_channel_data(ch2)
            f, Pxy = signal.csd(x, y, fs=self.fs, window=self.window,
                                nperseg=self.nperseg, noverlap=self.noverlap,
                                nfft=self.nfft, detrend=self.detrend)
            phase = np.angle(Pxy)
            key = f"{ch1}_{ch2}"
            results[key] = {
                'freq': f,
                'phase': phase,
                'Pxy': Pxy,
                'ch1': ch1,
                'ch2': ch2
            }
        return results

    def save_results(self, output_path: str, results: Dict,
                     analysis_type: str = 'crosspower',
                     global_attrs: Optional[dict] = None):
        """
        将计算结果保存为 HDF5 文件。

        参数:
            output_path : 输出路径
            results     : 由 compute_* 返回的字典
            analysis_type : 'crosspower', 'coherence', 'phase'
            global_attrs : 额外的全局属性
        """
        with h5py.File(output_path, 'w') as f:
            # 写入全局属性
            if global_attrs is not None:
                for k, v in global_attrs.items():
                    f.attrs[k] = v
            f.attrs['analysis_type'] = analysis_type
            f.attrs['fs'] = self.fs
            f.attrs['nperseg'] = self.nperseg
            f.attrs['noverlap'] = self.noverlap
            f.attrs['nfft'] = self.nfft
            f.attrs['window'] = self.window
            f.attrs['detrend'] = self.detrend
            f.attrs['date'] = datetime.now().isoformat()
            # 记录原始文件
            f.attrs['source_files'] = [str(p) for p in self.source_paths]

            for key, res in results.items():
                grp = f.create_group(key)
                grp.attrs['channel1'] = res['ch1']
                grp.attrs['channel2'] = res['ch2']
                grp.create_dataset('freq', data=res['freq'])
                if analysis_type == 'crosspower':
                    grp.create_dataset('Pxy', data=res['Pxy'])
                    grp.create_dataset('real', data=np.real(res['Pxy']))
                    grp.create_dataset('imag', data=np.imag(res['Pxy']))
                elif analysis_type == 'coherence':
                    grp.create_dataset('coherence', data=res['coherence'])
                elif analysis_type == 'phase':
                    grp.create_dataset('phase', data=res['phase'])
                    grp.create_dataset('Pxy', data=res['Pxy'])


# -----------------------------------------------------------------------------
# 便捷函数：一键分析三种谱并分别保存
# -----------------------------------------------------------------------------

def analyze_all_multi(source_paths: List[str],
                      pairs: List[Tuple[str, str]],
                      output_dir: Optional[str] = None,
                      **kwargs):
    """
    对多源数据，同时计算互谱、相干、相位，并保存三个 HDF5 文件。

    参数:
        source_paths : 数据源文件列表
        pairs        : 通道对列表，如 [('0:EX', '1:HX'), ('EX', 'HY')]
        output_dir   : 输出目录（默认源文件所在目录）
        **kwargs     : 传递给 MultiSourceSpectrum 的参数 (nperseg, noverlap, ...)
    """
    mss = MultiSourceSpectrum(source_paths, **kwargs)

    # 准备输出路径
    if output_dir is None:
        output_dir = Path(source_paths[0]).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = "_".join([Path(p).stem for p in source_paths])[:50]  # 避免过长

    # 计算并保存三种谱
    res_csd = mss.compute_csd(pairs)
    mss.save_results(output_dir / f"{base_name}_crosspower.h5", res_csd, 'crosspower')

    res_coh = mss.compute_coherence(pairs)
    mss.save_results(output_dir / f"{base_name}_coherence.h5", res_coh, 'coherence')

    res_phase = mss.compute_phase(pairs)
    mss.save_results(output_dir / f"{base_name}_phase.h5", res_phase, 'phase')

    print(f"分析完成，结果保存在: {output_dir}")


# -----------------------------------------------------------------------------
# 交互式入口（可选择多个文件并指定通道对）
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # 选择多个 HDF5 文件
    file_paths = filedialog.askopenfilenames(
        title="选择一个或多个预处理后的 HDF5 文件",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    if not file_paths:
        print("未选择文件，退出。")
        exit(0)

    # 选择输出目录
    output_dir = filedialog.askdirectory(title="选择输出目录（取消则使用第一个文件所在目录）")
    if not output_dir:
        output_dir = None

    # 这里需要用户手动指定通道对。为简化，示例中使用第一个文件的第一个通道与第二个文件的第一个通道。
    # 实际应用中，用户可修改此部分或从命令行传入。
    print("请根据数据源的通道名称，在代码中设置 pairs 变量。")
    print("可用通道规格示例：'0:EX'（第1个源）、'1:HX'（第2个源）等。")
    # 下面演示自动生成所有跨源组合（仅作示例）
    # 更灵活的方式：用户可在代码中直接定义 pairs = [('0:EX', '1:HX'), ...]
    # 由于交互式无法动态输入，这里自动生成第一个和第二个源的所有通道交叉组合。
    if len(file_paths) >= 2:
        # 加载第一个和第二个源的通道名
        src0 = _load_hdf5_data(file_paths[0])
        src1 = _load_hdf5_data(file_paths[1])
        pairs = [(f"0:{ch0}", f"1:{ch1}") for ch0 in src0['channels'] for ch1 in src1['channels']]
        print(f"自动生成跨源通道对 {len(pairs)} 个")
    else:
        # 单一文件，则生成内部所有两两组合
        src = _load_hdf5_data(file_paths[0])
        chs = src['channels']
        pairs = [(chs[i], chs[j]) for i in range(len(chs)) for j in range(i+1, len(chs))]
        print(f"单源，生成内部通道对 {len(pairs)} 个")

    try:
        analyze_all_multi(list(file_paths), pairs, output_dir,
                          nperseg=256, noverlap=128)
        messagebox.showinfo("完成", f"多源频谱分析完成！\n结果保存在: {output_dir or Path(file_paths[0]).parent}")
    except Exception as e:
        messagebox.showerror("错误", f"分析失败：\n{str(e)}")
        raise