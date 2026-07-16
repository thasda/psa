import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Optional, Tuple
import h5py

def bluewhitered_cmap() -> LinearSegmentedColormap:
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
    return LinearSegmentedColormap.from_list('bluewhitered', colors, N=256)

class CWTPlotter:
    def __init__(self,
                 tms: np.ndarray,
                 freq: np.ndarray,
                 coi: np.ndarray,
                 cfs_all: np.ndarray,
                 inx_ch: List[int],
                 day_hour: str = 'hour',
                 initial_time_str: str = '2024-02-01T12:30:00.000',
                 start_time_str: Optional[str] = None,
                 end_time_str: Optional[str] = None):
        self.tms = tms
        self.freq = freq
        self.coi = coi
        self.cfs_all = cfs_all
        self.inx_ch = inx_ch
        self.day_hour = day_hour
        self.nch = len(inx_ch)

        self.initial_time = datetime.strptime(initial_time_str, "%Y-%m-%dT%H:%M:%S.%f")
        # 修正：直接使用 tms 作为秒偏移
        self.abs_times = np.array([self.initial_time + timedelta(seconds=float(sec)) for sec in tms])

        if start_time_str is None:
            self.start_time = self.initial_time
        else:
            self.start_time = datetime.strptime(start_time_str, "%Y-%m-%dT%H:%M:%S.%f")
        if end_time_str is None:
            self.end_time = self.initial_time + timedelta(seconds=float(tms[-1]))
        else:
            self.end_time = datetime.strptime(end_time_str, "%Y-%m-%dT%H:%M:%S.%f")

        self.cmap = bluewhitered_cmap()

    @classmethod
    def from_h5(cls, h5_path: str,
                inx_ch: List[int],
                day_hour: str = 'hour',
                initial_time_str: str = '2024-02-01T12:30:00.000',
                start_time_str: Optional[str] = None,
                end_time_str: Optional[str] = None) -> 'CWTPlotter':
        """从 .cwt.result.h5 文件加载数据并创建实例"""
        with h5py.File(h5_path, 'r') as f:
            tms = f['tms'][:]
            freq = f['freq'][:]
            coi = f['coi'][:]      # (n_time, n_channels)
            cfs = f['cfs'][:]      # (n_freq, n_time, n_channels)
        return cls(tms, freq, coi, cfs, inx_ch, day_hour,
                   initial_time_str, start_time_str, end_time_str)

    def _get_time_indices_for_hour_mode(self) -> Tuple[int, int]:
        idx1 = np.searchsorted(self.abs_times, self.start_time)
        idx2 = np.searchsorted(self.abs_times, self.end_time) - 1
        if idx1 >= len(self.abs_times) or idx2 < 0 or idx1 > idx2:
            raise ValueError("指定的时间范围超出数据范围或无效")
        return idx1, idx2

    def plot(self, figsize: Tuple[float, float] = (12, 8), dpi: int = 150,
             save_path: Optional[str] = None, show: bool = True) -> None:
        fig, axes = plt.subplots(self.nch, 1, figsize=figsize, sharex=False, dpi=dpi)
        if self.nch == 1:
            axes = [axes]

        if self.day_hour == 'hour':
            i1, i2 = self._get_time_indices_for_hour_mode()
            plot_times = self.abs_times[i1:i2+1]
            x_numeric = mdates.date2num(plot_times)

            for i, ax in enumerate(axes):
                ch_idx = self.inx_ch[i]
                power = np.abs(self.cfs_all[:, i1:i2+1, ch_idx]) ** 2
                db = 10 * np.log10(power + 1e-20)
                vmin, vmax = (-50, 50) if i < 2 else (-30, 30)
                db_clipped = np.clip(db, vmin, vmax)
                print(f"频率范围: {self.freq[0]:.2f} ~ {self.freq[-1]:.2f} Hz")
                print(f"dB 最小值: {db.min():.1f}, 最大值: {db.max():.1f}")
                print(f"10 Hz 对应的行索引: {np.argmin(np.abs(self.freq - 10))}")
                print(
                    f"该行 dB 值范围: {db[np.argmin(np.abs(self.freq - 10)), :].min():.1f} ~ {db[np.argmin(np.abs(self.freq - 10)), :].max():.1f}")
                im = ax.imshow(db_clipped, aspect='auto', origin='lower',
                               extent=[x_numeric[0], x_numeric[-1],
                                       self.freq[0], self.freq[-1]],
                               cmap=self.cmap, vmin=vmin, vmax=vmax)
                ax.set_yscale('log')
                ax.set_ylim(0.001, 500)
                ax.set_yticks([0.01, 0.1, 1, 10, 100, 500])
                ax.yaxis.set_major_formatter(plt.ScalarFormatter())
                ax.set_ylabel('Frequency (Hz)')
                ax.set_xlabel('Time[HH:MM:SS]')
                ax.tick_params(direction='out')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
                fig.autofmt_xdate()
                cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
                cbar.set_label('dB/Hz')
            axes[-1].set_xlim(plot_times[0], plot_times[-1])

        elif self.day_hour == 'day':
            x_numeric = mdates.date2num(self.abs_times)
            for i, ax in enumerate(axes):
                ch_idx = self.inx_ch[i]
                power = np.abs(self.cfs_all[:, :, ch_idx]) ** 2
                db = 10 * np.log10(power + 1e-20)
                vmin, vmax = -50, 50
                db_clipped = np.clip(db, vmin, vmax)

                im = ax.imshow(db_clipped, aspect='auto', origin='lower',
                               extent=[x_numeric[0], x_numeric[-1],
                                       self.freq[0], self.freq[-1]],
                               cmap=self.cmap, vmin=vmin, vmax=vmax)
                ax.set_yscale('log')
                ax.set_ylim(0.01, 100)
                ax.set_yticks([0.01, 0.1, 1, 10, 100])
                ax.yaxis.set_major_formatter(plt.ScalarFormatter())
                ax.set_ylabel('Frequency (Hz)')
                ax.set_xlabel('Time (HH)')
                ax.tick_params(direction='out')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
                fig.autofmt_xdate()
                cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
                cbar.set_label('dB/Hz')
        else:
            raise ValueError("day_hour 必须是 'hour' 或 'day'")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()