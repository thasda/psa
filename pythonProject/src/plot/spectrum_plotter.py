"""
spectrum_plotter.py

绘图类：读取 HDF5 频谱分析结果文件，支持多种绘图类型及交互式数据标记。
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

# 尝试导入 mplcursors 以实现交互式标记
try:
    import mplcursors
    HAS_MPLCURSORS = True
except ImportError:
    HAS_MPLCURSORS = False


class SpectrumPlotter:
    """
    频谱结果绘图器，支持单图、三合一组合图，以及交互式数据点标记。
    可自动合并同目录下的 crosspower、coherence、phase 文件。
    """

    def __init__(self, h5_path: str):
        """
        参数:
            h5_path: 主 HDF5 文件路径（可以是任意一个，如 crosspower.h5）
        """
        self.base_path = Path(h5_path)
        self.dir = self.base_path.parent
        self.stem = self.base_path.stem.replace('_crosspower', '').replace('_coherence', '').replace('_phase', '')

        self.files = {
            'crosspower': self.dir / f"{self.stem}_crosspower.h5",
            'coherence': self.dir / f"{self.stem}_coherence.h5",
            'phase': self.dir / f"{self.stem}_phase.h5",
        }
        self.available_files = {k: p for k, p in self.files.items() if p.exists()}
        if not self.available_files:
            raise FileNotFoundError(f"未找到任何相关结果文件，请检查前缀: {self.stem}")

        self._data = {}
        self._load_data()
        self._pair_kinds = {}
        for pair in self._data.keys():
            kinds = []
            if 'Pxy' in self._data[pair]:
                kinds.extend(['magnitude', 'phase'])
            if 'coherence' in self._data[pair]:
                kinds.append('coherence')
            if 'phase' in self._data[pair] and 'phase' not in kinds:
                kinds.append('phase')
            self._pair_kinds[pair] = list(set(kinds))
        self.phase_conversion = 'rad'
        if 'phase' in self.available_files:
            with h5py.File(self.available_files['phase'], 'r') as f:
                if 'phase_conversion' in f.attrs:
                    self.phase_conversion = f.attrs['phase_conversion']

    def _load_data(self):
        """加载所有可用文件中的通道对数据"""
        pair_names = []
        for kind in ['crosspower', 'coherence', 'phase']:
            if kind in self.available_files:
                with h5py.File(self.available_files[kind], 'r') as f:
                    pair_names = list(f.keys())
                    break
        if not pair_names:
            raise ValueError("没有可用的文件")

        for pair in pair_names:
            self._data[pair] = {'freq': None, 'Pxy': None, 'coherence': None, 'phase': None}
            for kind, fpath in self.available_files.items():
                with h5py.File(fpath, 'r') as f:
                    if pair not in f:
                        continue
                    grp = f[pair]
                    if 'freq' in grp:
                        self._data[pair]['freq'] = grp['freq'][:]
                    if kind == 'crosspower' and 'Pxy' in grp:
                        self._data[pair]['Pxy'] = grp['Pxy'][:]
                    elif kind == 'coherence' and 'coherence' in grp:
                        self._data[pair]['coherence'] = grp['coherence'][:]
                    elif kind == 'phase' and 'phase' in grp:
                        self._data[pair]['phase'] = grp['phase'][:]
            freq = None
            for key in ['freq']:
                if self._data[pair][key] is not None:
                    freq = self._data[pair][key]
                    break
            if freq is None:
                raise ValueError(f"通道对 {pair} 缺少频率轴")
            for key in ['Pxy', 'coherence', 'phase']:
                if self._data[pair][key] is not None and len(self._data[pair][key]) != len(freq):
                    raise ValueError(f"通道对 {pair} 的 {key} 长度与频率不匹配")
            self._data[pair]['freq'] = freq

    def get_pairs(self) -> List[str]:
        return list(self._data.keys())

    def _convert_phase(self, phase_rad: np.ndarray, freq: np.ndarray, conversion: str) -> np.ndarray:
        """根据转换类型将原始相位（弧度）转换为指定格式"""
        if conversion == 'rad':
            return phase_rad
        elif conversion == 'deg':
            return np.rad2deg(phase_rad)
        elif conversion == 'unwrap_rad':
            return np.unwrap(phase_rad)
        elif conversion == 'unwrap_deg':
            return np.rad2deg(np.unwrap(phase_rad))
        elif conversion == 'time_delay':
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.divide(phase_rad, 2 * np.pi * freq,
                                 out=np.full_like(phase_rad, np.nan),
                                 where=(freq > 1e-12))
        elif conversion == 'group_delay':
            if len(phase_rad) < 3:
                raise ValueError("至少需要3个频率点才能计算群延迟")
            omega = 2 * np.pi * freq
            dphase = np.gradient(phase_rad, omega)
            return -dphase
        else:
            raise ValueError(f"不支持的相位转换: {conversion}")

    def get_available_kinds(self, pair: Optional[str] = None) -> List[str]:
        if pair is not None:
            return self._pair_kinds.get(pair, [])
        common = set(self._pair_kinds.get(self.get_pairs()[0], []))
        for p in self.get_pairs()[1:]:
            common &= set(self._pair_kinds.get(p, []))
        return list(common)

    def _get_data(self, pair: str, kind: str, phase_conversion: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, str]:
        data_dict = self._data[pair]
        freq = data_dict['freq']
        if kind == 'magnitude':
            if data_dict['Pxy'] is not None:
                data = np.abs(data_dict['Pxy'])
                ylabel = '|Pxy|'
            elif data_dict['coherence'] is not None:
                data = data_dict['coherence']
                ylabel = 'Coherence'
            else:
                raise ValueError(f"无法从 {pair} 提取幅度")
        elif kind == 'phase':
            # 获取原始相位（弧度）
            if data_dict['phase'] is not None:
                phase_rad = data_dict['phase']
            elif data_dict['Pxy'] is not None:
                phase_rad = np.angle(data_dict['Pxy'])
            else:
                raise ValueError(f"无法从 {pair} 提取相位")
            # 应用转换
            if phase_conversion is None:
                phase_conversion = self.phase_conversion
            data = self._convert_phase(phase_rad, freq, phase_conversion)
            # 生成合适的 ylabel
            if phase_conversion in ['time_delay', 'group_delay']:
                ylabel = f'Phase ({phase_conversion}) [s]'
            elif phase_conversion in ['deg', 'unwrap_deg']:
                ylabel = 'Phase (deg)'
            else:  # 'rad', 'unwrap_rad'
                ylabel = 'Phase (rad)'
            return freq, data, ylabel
        elif kind == 'coherence':
            if data_dict['coherence'] is not None:
                data = data_dict['coherence']
                ylabel = 'Coherence'
            else:
                raise ValueError(f"通道对 {pair} 无相干数据")
        else:
            raise ValueError(f"未知绘图类型: {kind}")
        return freq, data, ylabel

    # ---------- 交互式标记辅助 ----------
    def _add_interactive_cursor(self, fig, axes, data_series, labels=None):
        """
        为图形添加交互式数据点标记。
        data_series: 列表，每个元素为 (x, y, label) 或 (x, y)
        """
        if not HAS_MPLCURSORS:
            print("提示: 未安装 mplcursors，交互式标记功能不可用。请运行 pip install mplcursors")
            return

        # 如果是单个轴，包装为列表
        if not isinstance(axes, (list, tuple)):
            axes = [axes]
        # 确保 data_series 与 axes 数量匹配
        if len(data_series) != len(axes):
            # 如果 data_series 长度小于 axes，则只对前几个添加
            # 但通常我们传入与子图对应的序列
            pass

        # 收集所有线条对象
        lines = []
        for ax in axes:
            lines.extend(ax.lines)

        # 使用 mplcursors 为所有线条添加光标
        cursor = mplcursors.cursor(lines, hover=True)  # hover=True 表示悬停显示

        @cursor.connect("add")
        def on_add(sel):
            # 获取数据点坐标
            x, y = sel.target
            # 构造标签文本
            label = f"({x:.3g}, {y:.3g})"
            # 如果提供了 labels，可以附加更多信息
            sel.annotation.set_text(label)
            sel.annotation.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        # 如果希望点击固定标注，可以用 cursor.connect("add") 并设置 sel.annotation.draggable(True)
        # 但默认悬停显示，点击可固定（取决于 mplcursors 版本）
        # 这里保持默认行为：悬停显示，点击固定

        return cursor

    # ---------- 单图绘制（增强版） ----------
    def plot(self, pairs, kind: str = 'auto',phase_conversion:Optional[str] = None,
             xscale: str = 'linear', yscale: str = 'linear',
             line_kwargs: Optional[Dict[str, Any]] = None,
             figsize: Tuple[int, int] = (10, 6), title: Optional[str] = None,
             save_path: Optional[str] = None, show: bool = True,
             interactive: bool = True
             ):
        """
        绘制一个或多个通道对的指定类型图谱。

        参数:
            pairs         : 通道对名称（字符串或列表）
            kind          : 绘图类型 'magnitude', 'phase', 'coherence', 'auto'
            xscale        : x轴刻度类型 'linear' 或 'log'
            yscale        : y轴刻度类型 'linear' 或 'log'
            line_kwargs   : 传递给 plt.plot 的参数字典，如 {'color':'red', 'linewidth':2, 'linestyle':'--'}
            figsize       : 图形尺寸
            title         : 总标题
            save_path     : 保存路径
            show          : 是否显示
            interactive   : 是否启用交互式数据点标记
        """
        for i, pair in enumerate(pairs):
            freq, data, ylabel = self._get_data(pair, kind, phase_conversion=phase_conversion)

        if isinstance(pairs, str):
            pairs = [pairs]
        if len(pairs) == 0:
            raise ValueError("至少选择一个通道对")
        if kind == 'auto':
            for k in ['coherence', 'phase', 'magnitude']:
                if k in self.get_available_kinds(pairs[0]):
                    kind = k
                    break
            else:
                raise ValueError(f"通道对 {pairs[0]} 无可用数据")

        # 默认线条样式
        if line_kwargs is None:
            line_kwargs = {'color': 'blue', 'linewidth': 1.5}

        fig, axes = plt.subplots(len(pairs), 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        data_series_for_cursor = []
        for i, pair in enumerate(pairs):
            freq, data, ylabel = self._get_data(pair, kind)
            ax = axes[i]
            ax.plot(freq, data, **line_kwargs)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{pair} - {kind}')
            ax.grid(True, alpha=0.3)
            # 设置坐标轴刻度
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            # 收集数据点用于交互
            data_series_for_cursor.append((freq, data, f'{pair}'))

        if title:
            fig.suptitle(title)

        plt.tight_layout()

        # 交互式标记
        cursor = None
        if interactive and HAS_MPLCURSORS:
            cursor = self._add_interactive_cursor(fig, axes, data_series_for_cursor)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            # 如果不显示，需要关闭图形？但如果交互式，保持打开
            pass

        return fig, axes, cursor

    # ---------- 三合一组合图（增强版） ----------
    def plot_combined(self, pair: str,
                      phase_conversion: Optional[str] = None,
                      coh_threshold: float = 0.8,
                      mask_below_threshold: bool = True,
                      interference_freqs: Optional[List[float]] = None,
                      xscale: str = 'linear',
                      yscale_coh: str = 'linear',
                      yscale_mag: str = 'log',
                      yscale_phase: str = 'linear',
                      line_kwargs_coh: Optional[Dict[str, Any]] = None,
                      line_kwargs_mag: Optional[Dict[str, Any]] = None,
                      line_kwargs_phase: Optional[Dict[str, Any]] = None,
                      figsize: Tuple[int, int] = (12, 10),
                      title: Optional[str] = None,
                      save_path: Optional[str] = None,
                      show: bool = True,
                      interactive: bool = True):
        """
        绘制三合一组合图：上-相干谱，中-互谱幅度，下-相位谱。
        支持坐标轴刻度、线条样式定制及交互标记。

        参数:
            pair                  : 通道对名称
            coh_threshold         : 相干阈值（红色虚线）
            mask_below_threshold  : 是否在幅度和相位图中用灰色填充低相干区域
            interference_freqs    : 干扰频率列表，如 [50, 150]
            xscale                : x轴刻度类型 'linear' 或 'log'
            yscale_coh            : 相干谱 y轴刻度
            yscale_mag            : 幅度谱 y轴刻度（建议 'log'）
            yscale_phase          : 相位谱 y轴刻度
            line_kwargs_*         : 各子图线条样式参数字典
            figsize               : 图形尺寸
            title                 : 总标题
            save_path             : 保存路径
            show                  : 是否显示
            interactive           : 是否启用交互式数据点标记
        """
        # 检查数据
        data = self._data.get(pair)
        if data is None:
            raise ValueError(f"通道对 {pair} 不存在")
        if data['freq'] is None:
            raise ValueError(f"通道对 {pair} 频率轴缺失")
        if data['coherence'] is None:
            raise ValueError(f"通道对 {pair} 缺少相干数据")
        freq = data['freq']

        # 准备数据
        coh = data['coherence']
        if data['Pxy'] is not None:
            mag = np.abs(data['Pxy'])
        else:
            raise ValueError("缺少互功率谱数据")
        # 获取原始相位
        if data['phase'] is not None:
            phase_rad = data['phase']
        elif data['Pxy'] is not None:
            phase_rad = np.angle(data['Pxy'])
        else:
            raise ValueError("缺少相位数据")
        # 应用转换
        if phase_conversion is None:
            phase_conversion = self.phase_conversion
        phase = self._convert_phase(phase_rad, freq, phase_conversion)

        # 默认线条样式
        if line_kwargs_coh is None:
            line_kwargs_coh = {'color': 'blue', 'linewidth': 1.5}
        if line_kwargs_mag is None:
            line_kwargs_mag = {'color': 'green', 'linewidth': 1.5}
        if line_kwargs_phase is None:
            line_kwargs_phase = {'color': 'purple', 'linewidth': 1.5}

        # 创建子图
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 1, hspace=0.1, height_ratios=[1, 1, 1])
        ax_coh = fig.add_subplot(gs[0])
        ax_mag = fig.add_subplot(gs[1], sharex=ax_coh)
        ax_phase = fig.add_subplot(gs[2], sharex=ax_coh)

        # 绘制相干谱
        ax_coh.plot(freq, coh, **line_kwargs_coh)
        ax_coh.axhline(y=coh_threshold, color='red', linestyle='--', alpha=0.7, label=f'阈值 {coh_threshold}')
        ax_coh.set_ylabel('Coherence')
        ax_coh.set_ylim([-0.05, 1.05])
        ax_coh.grid(True, alpha=0.3)
        ax_coh.legend(loc='upper right')
        ax_coh.set_title(f'通道对: {pair}')
        ax_coh.set_xscale(xscale)
        ax_coh.set_yscale(yscale_coh)

        # 绘制幅度谱（对数坐标建议）
        mag_pos = np.maximum(mag, 1e-20)
        ax_mag.plot(freq, mag_pos, **line_kwargs_mag)
        ax_mag.set_ylabel('|Pxy|')
        ax_mag.grid(True, alpha=0.3, which='both')
        ax_mag.set_xscale(xscale)
        ax_mag.set_yscale(yscale_mag)

        # 绘制相位谱
        ax_phase.plot(freq, phase, **line_kwargs_phase)
        ax_phase.set_xlabel('Frequency (Hz)')
        ax_phase.set_ylabel(f'Phase ({phase_conversion})')
        ax_phase.grid(True, alpha=0.3)
        ax_phase.set_xscale(xscale)
        ax_phase.set_yscale(yscale_phase)

        # 低相干区域填充
        if mask_below_threshold:
            below = coh < coh_threshold
            if np.any(below):
                diff = np.diff(np.concatenate(([False], below, [False])))
                start_idx = np.where(diff == 1)[0]
                end_idx = np.where(diff == -1)[0]
                for s, e in zip(start_idx, end_idx):
                    f_start = freq[s]
                    f_end = freq[e-1]
                    ax_mag.axvspan(f_start, f_end, facecolor='gray', alpha=0.2)
                    ax_phase.axvspan(f_start, f_end, facecolor='gray', alpha=0.2)

        # 干扰频率标注
        if interference_freqs is not None:
            for f0 in interference_freqs:
                if freq[0] <= f0 <= freq[-1]:
                    for ax in [ax_coh, ax_mag, ax_phase]:
                        ax.axvline(x=f0, color='orange', linestyle=':', alpha=0.7, linewidth=1)
                    ax_coh.text(f0, 0.95, f'{f0} Hz', color='orange', ha='center', va='top', rotation=45, fontsize=8)

        # 设置 x 轴范围（保持与数据一致）
        ax_phase.set_xlim([freq[0], freq[-1]])

        if title:
            fig.suptitle(title)

        plt.tight_layout()

        # 交互式标记
        cursor = None
        if interactive and HAS_MPLCURSORS:
            # 为每个子图提供数据点
            data_series = [
                (freq, coh, 'Coherence'),
                (freq, mag_pos, '|Pxy|'),
                (freq, phase, 'Phase')
            ]
            cursor = self._add_interactive_cursor(fig, [ax_coh, ax_mag, ax_phase], data_series)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()

        return fig, (ax_coh, ax_mag, ax_phase), cursor

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()