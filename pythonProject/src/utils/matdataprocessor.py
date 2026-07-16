import numpy as np
import pandas as pd
import h5py
import scipy.io as sio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pythonProject.configs.configmanager import ConfigManager  # 请根据实际路径调整


class MatDataProcessor:
    """
    MATLAB 时间序列数据处理器（原始数据保留版）。

    设计原则：
        - 不对原始数据做任何滤波、降采样或幅值缩放。
        - 仅负责从 MAT 文件提取数据并转换为 HDF5 格式，保留全部原始信息。
        - 时间轴强制使用配置中的开始时间和采样率生成。
        - 支持通道名称配置，但不修改数据值。

    支持两种 MAT 结构：
        1. 直接包含 TSData 和 times 变量；
        2. 包含 AttsExport 变量，需根据配置参数（通道数、采样率、起始时间）构建 TSData 和 times。
    """

    def __init__(self, mat_path: str, config_manager: Optional[ConfigManager] = None, **config_overrides):
        self.mat_path = mat_path
        self.config = {}

        # 从 ConfigManager 加载配置
        if config_manager is not None:
            cfg_dict = config_manager.get_config_dict()
            self.config['sample_rate'] = cfg_dict.get("采样率 (Hz)")
            self.config['start_time'] = cfg_dict.get("开始时间")
            self.config['channel_count'] = cfg_dict.get("通道数量")
            self.config['channel_indices'] = cfg_dict.get("通道索引")
            self.config['channel_gains'] = cfg_dict.get("通道增益")
            self.config['electrical_lengths'] = cfg_dict.get("电长度 (米)")
            self.config['calibration_files'] = cfg_dict.get("校准文件")
            self.config['channel_names'] = cfg_dict.get("通道名称")

        # 覆盖/补充直接传入的参数
        self.config.update(config_overrides)

        # 必须参数校验
        required = ['sample_rate', 'start_time', 'channel_count']
        missing = [r for r in required if r not in self.config or self.config[r] is None]
        if missing:
            raise ValueError(f"缺少必要配置参数: {missing}")

        # 解析通道名称（仅用于元数据，不修改数据）
        ch_names = self.config.get('channel_names')
        if ch_names is None:
            self.channel_names = None
        elif isinstance(ch_names, str):
            self.channel_names = [name.strip() for name in ch_names.split(',') if name.strip()]
        else:
            self.channel_names = list(ch_names)

        if self.channel_names is not None:
            if len(self.channel_names) != self.config['channel_count']:
                raise ValueError(
                    f"通道名称数量 ({len(self.channel_names)}) 与通道数 ({self.config['channel_count']}) 不符"
                )

        # 将起始时间字符串转为 datetime 对象
        if isinstance(self.config['start_time'], str):
            self.config['start_time'] = pd.to_datetime(self.config['start_time']).to_pydatetime()

    # ----------------------------------------------------------------------
    # 内部数据提取（与之前相同）
    # ----------------------------------------------------------------------
    def _read_mat_raw(self):
        try:
            with h5py.File(self.mat_path, 'r') as f:
                print("使用 HDF5 模式读取 MAT 文件")
                return self._extract_from_h5(f)
        except Exception as e:
            print(f"HDF5 打开失败，尝试 scipy.io.loadmat: {e}")
            mat_data = sio.loadmat(self.mat_path)
            return self._extract_from_dict(mat_data)

    def _extract_from_h5(self, f: h5py.File) -> Dict[str, Any]:
        data = {}
        if 'AttsExport' in f:
            atts = f['AttsExport'][()].T
            data['atts'] = atts
        for name in ['TSData', 'times', 'sf', 'nch']:
            if name in f:
                val = f[name][()]
                if isinstance(val, np.ndarray) and val.ndim == 2 and name == 'TSData':
                    val = val.T
                data[name] = val
        return data

    def _extract_from_dict(self, d: Dict) -> Dict[str, Any]:
        data = {}
        if 'AttsExport' in d:
            atts = d['AttsExport']
            if atts.ndim == 2:
                atts = atts.T
            data['atts'] = atts
        for name in ['TSData', 'times', 'sf', 'nch']:
            if name in d:
                data[name] = d[name]
        return data

    # ----------------------------------------------------------------------
    # 构建 TSData 和 times —— 关键修改：取消所有降采样和滤波
    # ----------------------------------------------------------------------
    def _build_from_atts(self, atts: np.ndarray) -> tuple:
        """
        从 AttsExport 构建 TSData。
        原代码中的 downsample_factor 降采样逻辑已被移除，以保证原始数据完整性。
        """
        dat = atts  # 直接使用原始数据
        sf = self.config['sample_rate']  # 使用原始采样率

        nch = self.config['channel_count']
        ts_data = dat[:, :nch] if dat.shape[1] >= nch else dat

        n_samples = ts_data.shape[0]
        times = pd.date_range(
            start=self.config['start_time'],
            periods=n_samples,
            freq=pd.DateOffset(seconds=1 / sf)
        ).to_pydatetime()
        return ts_data, times

    def _read_direct(self, raw_data: Dict) -> tuple:
        """直接读取 TSData，强制使用配置的开始时间生成时间轴"""
        ts_data = raw_data['TSData']
        n_samples = ts_data.shape[0]
        times = pd.date_range(
            start=self.config['start_time'],
            periods=n_samples,
            freq=pd.DateOffset(seconds=1 / self.config['sample_rate'])
        ).to_pydatetime()
        return ts_data, times

    # ----------------------------------------------------------------------
    # 对外接口
    # ----------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        raw = self._read_mat_raw()
        if 'atts' in raw:
            print("从 AttsExport 构建 TSData 和 times...")
            ts_data, times = self._build_from_atts(raw['atts'])
        elif 'TSData' in raw:
            print("直接读取 TSData 并强制使用配置时间轴...")
            ts_data, times = self._read_direct(raw)
        else:
            raise KeyError("MAT 文件中既无 AttsExport 也无 TSData，无法提取时间序列")

        nch = ts_data.shape[1]
        if self.channel_names is not None:
            columns = self.channel_names[:nch]
        else:
            columns = [f"ch{i}" for i in range(nch)]
        df = pd.DataFrame(ts_data, index=times, columns=columns)
        df.index.name = 'time'
        return df

    def to_hdf5(self, output_path: str, chunk_size: int = 100000, **hdf5_kwargs):
        raw = self._read_mat_raw()
        if 'atts' in raw:
            ts_data, times = self._build_from_atts(raw['atts'])
        elif 'TSData' in raw:
            ts_data, times = self._read_direct(raw)
        else:
            raise KeyError("无法提取时间序列数据")

        n_samples, n_channels = ts_data.shape
        dtype = ts_data.dtype

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with h5py.File(output_path, 'w') as h5:
            ds = h5.create_dataset(
                'TSData',
                shape=(n_samples, n_channels),
                dtype=dtype,
                chunks=(chunk_size, n_channels),
                **hdf5_kwargs
            )
            time_strs = np.array([t.isoformat() for t in times], dtype='S26')
            h5.create_dataset('times', data=time_strs, dtype='S26')

            # 存储通道名称属性
            if self.channel_names is not None:
                h5.attrs['channel_names'] = [name.encode('utf-8') for name in self.channel_names]
            else:
                h5.attrs['channel_names'] = [f"ch{i}".encode() for i in range(n_channels)]

            # 分块写入数据（直接拷贝，无修改）
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                ds[start:end, :] = ts_data[start:end, :]
                print(f"已写入 {end}/{n_samples} 行")

            # 保存其他配置属性
            for key, value in self.config.items():
                if value is not None and isinstance(value, (str, int, float, bool)):
                    h5.attrs[key] = value
        print(f"数据已保存至 {output_path}")

    # ======================================================================
    # 交互式方法（与之前相同，支持 .cfg 和 .csv）
    # ======================================================================
    @classmethod
    def interactive_convert(cls):
        root = tk.Tk()
        root.withdraw()

        # 1. 选择 MAT 文件
        mat_path = filedialog.askopenfilename(
            title="选择 MATLAB (.mat) 数据文件",
            filetypes=[("MAT files", "*.mat"), ("All files", "*.*")]
        )
        if not mat_path:
            print("未选择输入文件，退出。")
            return

        # 2. 选择配置文件（支持 .cfg 和 .csv）
        use_cfg = messagebox.askyesno("配置文件", "是否使用配置文件（.cfg 或 .csv）获取参数？")
        config_manager = None
        config_overrides = {}

        if use_cfg:
            cfg_path = filedialog.askopenfilename(
                title="选择配置文件",
                filetypes=[("Config files", "*.cfg;*.csv"), ("All files", "*.*")]
            )
            if cfg_path:
                ext = os.path.splitext(cfg_path)[1].lower()
                try:
                    if ext == '.cfg':
                        config_manager = ConfigManager.from_aether_cfg(cfg_path)
                    elif ext == '.csv':
                        config_manager = ConfigManager.from_csv(cfg_path)
                    else:
                        raise ValueError(f"不支持的配置文件格式: {ext}")
                    print(f"已加载配置文件: {cfg_path}")
                    cfg_dict = config_manager.get_config_dict()
                    print("配置摘要:")
                    for k in ['采样率 (Hz)', '开始时间', '通道数量', '通道名称']:
                        if k in cfg_dict:
                            print(f"  {k}: {cfg_dict[k]}")
                except Exception as e:
                    messagebox.showerror("错误", f"加载配置文件失败:\n{str(e)}")
                    return
            else:
                print("未选择配置文件，将手动输入参数。")
                use_cfg = False

        if not use_cfg:
            sample_rate = simpledialog.askfloat("输入参数", "采样率 (Hz):", initialvalue=1000.0)
            if sample_rate is None:
                return
            start_time_str = simpledialog.askstring(
                "输入参数", "开始时间 (格式: YYYY-MM-DD HH:MM:SS)",
                initialvalue="2022-11-13 15:00:00"
            )
            if start_time_str is None:
                return
            channel_count = simpledialog.askinteger("输入参数", "通道数量:", initialvalue=5)
            if channel_count is None:
                return
            channel_names_str = simpledialog.askstring(
                "输入参数", "通道名称（逗号分隔，留空使用默认 ch0,ch1...）:"
            )
            config_overrides = {
                'sample_rate': sample_rate,
                'start_time': start_time_str,
                'channel_count': channel_count,
            }
            if channel_names_str:
                config_overrides['channel_names'] = [n.strip() for n in channel_names_str.split(',') if n.strip()]
            config_manager = None

        # 3. 生成输出路径
        base_name = os.path.splitext(os.path.basename(mat_path))[0]
        output_dir = filedialog.askdirectory(title="选择输出目录")
        if not output_dir:
            output_dir = os.path.dirname(mat_path)
        output_path = os.path.join(output_dir, f"{base_name}.h5")

        # 4. 执行转换
        try:
            processor = cls(mat_path, config_manager=config_manager, **config_overrides)
            processor.to_hdf5(output_path)
            messagebox.showinfo("完成", f"转换完成！\n输出文件：{output_path}")
        except Exception as e:
            messagebox.showerror("错误", f"转换失败：{str(e)}")
            raise


if __name__ == "__main__":
    MatDataProcessor.interactive_convert()
