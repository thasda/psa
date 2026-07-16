# preprocess.py（修正属性复制版）

import numpy as np
import h5py
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from pythonProject.configs.csv_config_manger import ConfigManager
from pythonProject.src.utils.lemi417 import LEMi417Calibrator


class Preprocessor:
    """
    预处理类：
    - 输入 HDF5 应包含 TSData 和 times（times 应为 ISO 格式字符串）
    - 对磁通道（Hx, Hy, Hz）进行仪器响应去除（CMT 校准）
    - 一阶差分去趋势
    - 输出 HDF5，times 保持 ISO 字符串格式，并复制原始属性（如 channel_names）
    """

    def __init__(self, hdf5_path: str, config_manager: ConfigManager):
        self.hdf5_path = hdf5_path
        cfg = config_manager.get_config_dict()

        self.sample_rate = cfg.get("采样率(Hz)") or cfg.get("采样率 (Hz)")
        if self.sample_rate is None:
            raise ValueError("配置中缺少采样率")

        self.calibration_files = cfg.get("校准文件")
        if not self.calibration_files or len(self.calibration_files) < 3:
            print("警告：校准文件不足 3 个，将跳过仪器响应去除")
            self.calibrator = None
        else:
            cfg_dir = Path(config_manager._config_path).parent if hasattr(config_manager, '_config_path') else Path.cwd()
            abs_paths = []
            for f in self.calibration_files:
                p = Path(f)
                if not p.is_absolute():
                    p = cfg_dir / p
                abs_paths.append(str(p))
            self.calibrator = LEMi417Calibrator(
                abs_paths,
                channel_mapping={'HX': 0, 'HY': 1, 'HZ': 2}
            )
            print("校准器初始化成功（磁通道）")

    def _read_data_and_attrs(self):
        """读取数据、时间轴和所有属性"""
        with h5py.File(self.hdf5_path, 'r') as f:
            data = f['TSData'][:]
            times_raw = f['times'][:]
            attrs = dict(f.attrs)   # 复制所有属性

        # 解析 times 为 datetime 对象列表
        if times_raw.dtype.kind in 'US':
            times_dt = []
            for t in times_raw:
                if isinstance(t, bytes):
                    t = t.decode()
                times_dt.append(datetime.fromisoformat(t))
        else:
            raise ValueError(
                "输入 HDF5 的 times 必须为 ISO 格式字符串。"
                "请使用 mat_to_hdf5.py 生成的 HDF5 文件。"
            )
        return data, times_dt, attrs

    def _apply_instrument_response(self, data: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return data
        nch = data.shape[1]
        mag_indices = [2, 3, 4] if nch >= 5 else list(range(max(2, nch-3), nch))
        mag_indices = [i for i in mag_indices if i < nch]
        print(f"对磁通道 {mag_indices} 进行仪器响应校正...")
        corrected = data.astype(float).copy()
        for idx, ch_name in zip(mag_indices, ['HX', 'HY', 'HZ']):
            if idx < nch:
                corrected[:, idx] = self.calibrator.calibrate_time_series(
                    corrected[:, idx], self.sample_rate, channel_name=ch_name
                )
        return corrected

    def _first_difference(self, data: np.ndarray, times_dt: list):
        diff_data = data[1:, :] - data[:-1, :]
        diff_times = times_dt[1:]
        return diff_data, diff_times

    def process(self, output_path: str):
        print(f"读取数据: {self.hdf5_path}")
        data, times_dt, attrs = self._read_data_and_attrs()
        print(f"原始形状: {data.shape}，时间点数: {len(times_dt)}")

        # 仪器响应
        data = self._apply_instrument_response(data)

        # 一阶差分
        data, times_dt = self._first_difference(data, times_dt)
        print(f"差分后形状: {data.shape}，时间点数: {len(times_dt)}")

        # 转换为 ISO 字符串数组
        times_iso = np.array([t.isoformat() for t in times_dt], dtype='S26')

        # 保存 HDF5，复制所有属性并添加预处理标记
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('TSData', data=data, compression='gzip')
            f.create_dataset('times', data=times_iso)

            # 复制原始属性（包括 channel_names）
            for k, v in attrs.items():
                f.attrs[k] = v

            # 添加/更新预处理相关属性
            f.attrs['preprocessed'] = True
            f.attrs['method'] = 'instrument_response + first_difference'
            f.attrs['date'] = datetime.now().isoformat()
            f.attrs['sample_rate'] = self.sample_rate

        print(f"预处理完成，保存至: {output_path}")


# ======================================================================
# 交互式执行（不变）
# ======================================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    input_path = filedialog.askopenfilename(
        title="选择输入 HDF5 文件",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    if not input_path:
        print("未选择输入文件，退出。")
        exit(0)

    config_path = filedialog.askopenfilename(
        title="选择 CSV 配置文件",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not config_path:
        print("未选择配置文件，退出。")
        exit(0)

    cfg_mgr = ConfigManager.from_csv(config_path)
    cfg_mgr._config_path = config_path

    output_path = filedialog.asksaveasfilename(
        title="保存预处理后的 HDF5 文件",
        defaultextension=".h5",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
        initialfile="preprocessed.h5"
    )
    if not output_path:
        base = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{base}_preprocessed.h5")
        print(f"未指定输出路径，自动保存至: {output_path}")

    try:
        pre = Preprocessor(input_path, cfg_mgr)
        pre.process(output_path)
        messagebox.showinfo("完成", f"预处理完成！\n输出文件：{output_path}")
    except Exception as e:
        messagebox.showerror("错误", f"预处理失败：\n{str(e)}")
        raise