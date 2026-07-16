import numpy as np
import pandas as pd
import h5py
import scipy.io as sio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pythonProject.configs.csv_config_manger import ConfigManager  # 请确保路径正确


class MatDataProcessor:
    """
    MATLAB 时间序列数据处理器（带物理单位转换）。

    功能：
        - 从 CSV 配置文件读取参数（开始时间、采样率、通道数、通道索引、增益、电长度、传感器灵敏度等）。
        - 提取 MAT 文件中的 AttsExport 数据，按通道索引选取，并进行电场/磁场单位转换。
        - 根据开始时间和采样率生成精确的时间轴。
        - 保存为 HDF5 格式，保留全部原始数据（仅做单位转换，不滤波、不降采样）。

    支持两种 MAT 结构：
        1. 直接包含 TSData 和 times 变量（但仍使用配置的开始时间重新生成时间轴）。
        2. 包含 AttsExport 变量（主要使用方式）。
    """

    def __init__(self, mat_path: str, config_manager: Optional[ConfigManager] = None, **config_overrides):
        self.mat_path = mat_path
        self.config = {}

        # 从 ConfigManager 加载配置
        if config_manager is not None:
            cfg_dict = config_manager.get_config_dict()
            # 使用中文键名（与 CSV 列“参数名”一致）
            self.config['sample_rate'] = cfg_dict.get("采样率(Hz)") or cfg_dict.get("采样率 (Hz)")
            self.config['start_time'] = cfg_dict.get("开始时间")
            self.config['channel_count'] = cfg_dict.get("通道数量")
            self.config['channel_indices'] = cfg_dict.get("通道索引")
            self.config['channel_gains'] = cfg_dict.get("通道增益")
            self.config['electrical_lengths'] = cfg_dict.get("电长度（米）") or cfg_dict.get("电长度 (米)")
            self.config['calibration_files'] = cfg_dict.get("校准文件")
            self.config['sensor_sensitivity'] = cfg_dict.get("传感器灵敏度(mV/nT)") or cfg_dict.get("传感器灵敏度")
            self.config['channel_names'] = cfg_dict.get("通道名") or cfg_dict.get("通道名称")

        # 覆盖/补充直接传入的参数
        self.config.update(config_overrides)

        # 必须参数校验
        required = ['sample_rate', 'start_time', 'channel_count', 'channel_indices', 'channel_gains']
        missing = [r for r in required if r not in self.config or self.config[r] is None]
        if missing:
            raise ValueError(f"缺少必要配置参数: {missing}")

        # 解析通道名称（仅用于元数据）
        ch_names = self.config.get('channel_names')
        if ch_names is None:
            self.channel_names = None
        elif isinstance(ch_names, str):
            self.channel_names = [name.strip() for name in ch_names.split(',') if name.strip()]
        else:
            self.channel_names = list(ch_names)

        # 解析通道索引（列表，转换为0基索引）
        indices = self.config['channel_indices']
        if isinstance(indices, str):
            indices = [int(x.strip()) for x in indices.split(',') if x.strip()]
        elif isinstance(indices, (list, tuple)):
            indices = [int(x) for x in indices]
        else:
            indices = [int(indices)]
        self.channel_indices = [i - 1 for i in indices]  # MATLAB 1-based -> Python 0-based

        # 解析通道增益（列表）
        gains = self.config['channel_gains']
        if isinstance(gains, str):
            gains = [float(x.strip()) for x in gains.split(',') if x.strip()]
        elif isinstance(gains, (list, tuple)):
            gains = [float(x) for x in gains]
        else:
            gains = [float(gains)]
        self.channel_gains = gains

        # 解析电长度（仅用于电场通道，通常为2个）
        elens = self.config.get('electrical_lengths')
        if elens is None:
            self.electrical_lengths = None
        elif isinstance(elens, str):
            self.electrical_lengths = [float(x.strip()) for x in elens.split(',') if x.strip()]
        elif isinstance(elens, (list, tuple)):
            self.electrical_lengths = [float(x) for x in elens]
        else:
            self.electrical_lengths = [float(elens)]

        # 传感器灵敏度（用于磁场）
        sens = self.config.get('sensor_sensitivity')
        if sens is None:
            self.sensor_sensitivity = 1.0  # 默认值
        else:
            self.sensor_sensitivity = float(sens)

        # 通道数量校验
        nch = self.config['channel_count']
        if len(self.channel_indices) != nch:
            raise ValueError(f"通道索引数量 ({len(self.channel_indices)}) 与通道数 ({nch}) 不符")
        if len(self.channel_gains) != nch:
            raise ValueError(f"通道增益数量 ({len(self.channel_gains)}) 与通道数 ({nch}) 不符")
        if self.channel_names is not None and len(self.channel_names) != nch:
            raise ValueError(f"通道名称数量 ({len(self.channel_names)}) 与通道数 ({nch}) 不符")

        # 解析开始时间
        start_time_str = self.config['start_time']
        if isinstance(start_time_str, str):
            # 尝试多种格式
            try:
                self.start_time = pd.to_datetime(start_time_str).to_pydatetime()
            except:
                # 尝试 ISO 格式带 T
                self.start_time = datetime.fromisoformat(start_time_str.replace('T', ' '))
        else:
            self.start_time = start_time_str

    # ----------------------------------------------------------------------
    # 内部数据提取
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
            atts = f['AttsExport'][()].T   # 转置为 (样本, 通道)
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
    # 数据修正（单位转换）
    # ----------------------------------------------------------------------
    def _correct_data(self, raw_data: np.ndarray) -> np.ndarray:
        """
        根据配置对原始数据进行物理单位转换。
        假定 raw_data 形状为 (样本数, 原始通道数)，需先按索引提取。
        """
        # 1. 按通道索引提取所需列
        extracted = raw_data[:, self.channel_indices]  # (样本, nch)

        # 2. 逐通道应用修正
        nch = self.config['channel_count']
        corrected = np.empty_like(extracted, dtype=float)

        for i in range(nch):
            col = extracted[:, i]
            gain = self.channel_gains[i]
            # 判断该通道是电场还是磁场（根据名称或位置约定）
            # 通常前两个为 Ex, Ey（电场），后面为 Hx, Hy, Hz（磁场）
            # 也可以根据电长度配置长度来判断：如果 i < len(electrical_lengths) 则视为电场
            if self.electrical_lengths is not None and i < len(self.electrical_lengths):
                # 电场通道：除以 (电长度 * 1e-3 * 增益)
                elen = self.electrical_lengths[i]
                factor = elen * 1e-3 * gain
                if factor != 0:
                    corrected[:, i] = col / factor
                else:
                    corrected[:, i] = col  # 若因子为0则保持原值
                    print(f"警告：通道 {i} 电长度或增益为0，未转换")
            else:
                # 磁场通道：除以 (传感器灵敏度 * 增益)
                factor = self.sensor_sensitivity * gain
                if factor != 0:
                    corrected[:, i] = col / factor
                else:
                    corrected[:, i] = col
                    print(f"警告：通道 {i} 传感器灵敏度或增益为0，未转换")

        return corrected

    # ----------------------------------------------------------------------
    # 构建 TSData 和时间轴（根据开始时间和采样率）
    # ----------------------------------------------------------------------
    def _build_data_and_time(self, raw: Dict) -> tuple:
        """从 raw 中提取原始数据，修正后生成时间轴"""
        # 获取原始数据
        if 'atts' in raw:
            raw_data = raw['atts']
            print("从 AttsExport 提取数据...")
        elif 'TSData' in raw:
            raw_data = raw['TSData']
            print("从 TSData 提取数据...")
        else:
            raise KeyError("MAT 文件中既无 AttsExport 也无 TSData")

        # 修正数据
        ts_data = self._correct_data(raw_data)
        n_samples = ts_data.shape[0]

        # 生成时间轴
        sf = self.config['sample_rate']
        start = self.start_time
        # 使用 pandas 生成 datetime 索引，然后转为 numpy datetime64
        time_index = pd.date_range(start=start, periods=n_samples, freq=pd.DateOffset(seconds=1/sf))
        times = time_index.to_pydatetime()

        return ts_data, times

    # ----------------------------------------------------------------------
    # 对外接口
    # ----------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        raw = self._read_mat_raw()
        ts_data, times = self._build_data_and_time(raw)

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
        ts_data, times = self._build_data_and_time(raw)

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
            # 存储 times 为 ISO 字符串（便于跨平台）
            time_strs = np.array([t.isoformat() for t in times], dtype='S26')
            h5.create_dataset('times', data=time_strs, dtype='S26')

            # 存储通道名称
            if self.channel_names is not None:
                h5.attrs['channel_names'] = [name.encode('utf-8') for name in self.channel_names]
            else:
                h5.attrs['channel_names'] = [f"ch{i}".encode() for i in range(n_channels)]

            # 分块写入数据
            for start in range(0, n_samples, chunk_size):
                end = min(start + chunk_size, n_samples)
                ds[start:end, :] = ts_data[start:end, :]
                print(f"已写入 {end}/{n_samples} 行")

            # 保存其他配置属性（便于追溯）
            for key, value in self.config.items():
                if value is not None and isinstance(value, (str, int, float, bool)):
                    h5.attrs[key] = value
            # 额外保存修正参数
            h5.attrs['sensor_sensitivity'] = self.sensor_sensitivity
            if self.electrical_lengths:
                h5.attrs['electrical_lengths'] = self.electrical_lengths

        print(f"数据已保存至 {output_path}")

    # ======================================================================
    # 交互式方法（需输入开始时间）
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

        # 2. 选择配置文件（必须为 CSV）
        use_cfg = messagebox.askyesno("配置文件", "是否使用 CSV 配置文件获取参数？")
        config_manager = None
        config_overrides = {}

        if use_cfg:
            cfg_path = filedialog.askopenfilename(
                title="选择 CSV 配置文件",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if cfg_path:
                try:
                    config_manager = ConfigManager.from_csv(cfg_path)
                    print(f"已加载配置文件: {cfg_path}")
                    cfg_dict = config_manager.get_config_dict()
                    print("配置摘要:")
                    for k in ['采样率(Hz)', '开始时间', '通道数量', '通道索引', '通道增益', '电长度（米）', '传感器灵敏度(mV/nT)']:
                        if k in cfg_dict:
                            print(f"  {k}: {cfg_dict[k]}")
                except Exception as e:
                    messagebox.showerror("错误", f"加载配置文件失败:\n{str(e)}")
                    return
            else:
                print("未选择配置文件，将手动输入参数。")
                use_cfg = False

        if not use_cfg:
            # 手动输入所有必要参数
            start_time_str = simpledialog.askstring(
                "输入参数", "开始时间 (格式: YYYY-MM-DD HH:MM:SS 或 YYYY-MM-DDTHH:MM:SS.sss):",
                initialvalue="2022-11-05 01:17:05.240"
            )
            if not start_time_str:
                return
            sample_rate = simpledialog.askfloat("输入参数", "采样率 (Hz):", initialvalue=1000.0)
            if sample_rate is None:
                return
            channel_count = simpledialog.askinteger("输入参数", "通道数量:", initialvalue=5)
            if channel_count is None:
                return
            channel_indices_str = simpledialog.askstring(
                "输入参数", "通道索引 (逗号分隔，如 1,2,3,4,5):", initialvalue="1,2,3,4,5"
            )
            if not channel_indices_str:
                return
            channel_gains_str = simpledialog.askstring(
                "输入参数", "通道增益 (逗号分隔):", initialvalue="1,1,1,1,1"
            )
            if not channel_gains_str:
                return
            electrical_lengths_str = simpledialog.askstring(
                "输入参数", "电长度(米) (逗号分隔，仅电场通道):", initialvalue="19.5,38.5"
            )
            sensor_sensitivity = simpledialog.askfloat(
                "输入参数", "传感器灵敏度 (mV/nT):", initialvalue=1.0
            )
            if sensor_sensitivity is None:
                sensor_sensitivity = 1.0
            channel_names_str = simpledialog.askstring(
                "输入参数", "通道名称 (逗号分隔，留空使用默认):"
            )

            config_overrides = {
                'start_time': start_time_str,
                'sample_rate': sample_rate,
                'channel_count': channel_count,
                'channel_indices': channel_indices_str,
                'channel_gains': channel_gains_str,
                'electrical_lengths': electrical_lengths_str,
                'sensor_sensitivity': sensor_sensitivity,
            }
            if channel_names_str:
                config_overrides['channel_names'] = [n.strip() for n in channel_names_str.split(',') if n.strip()]
            config_manager = None

        # 3. 生成输出路径
        base_name = os.path.splitext(os.path.basename(mat_path))[0]
        output_dir = filedialog.askdirectory(title="选择输出目录")
        if not output_dir:
            output_dir = os.path.dirname(mat_path)
        output_path = os.path.join(output_dir, f"{base_name}_corrected.h5")

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