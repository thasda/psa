import os
import sys
import logging
from tkinter import filedialog
import matplotlib
import numpy as np
import pandas as pd
from typing import Any, Dict
from PIL._tkinter_finder import tk
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QDialogButtonBox, QLabel, QHeaderView, \
    QMainWindow, QApplication, QTableWidget, QTableView, QSizePolicy, QMessageBox, QProgressBar
from matplotlib import ticker
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from DataInterface import DataInterface, ElectroVoyant
from PyQt6.QtCore import QObject, pyqtSignal

from 软件代码.LEMi417 import LEMi417Calibrator
from 软件代码.配置参数_main import DialogWindow
from 软件代码.软件界面 import Ui_MainWindow
from 软件代码.级联滤波 import CascadeFiltering

# 设置 Matplotlib 后端
matplotlib.use('QtAgg')  # 明确使用兼容Qt6的后端
# 验证后端设置
logging.info(f"成功导入 Matplotlib {matplotlib.__version__}，使用后端: {matplotlib.get_backend()}")

# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'  # 设置中文字体为黑体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["figure.dpi"] = 300  # 提高默认DPI
HORIZONTAL = QtCore.Qt.Orientation.Horizontal
DISPLAY_ROLE = QtCore.Qt.ItemDataRole.DisplayRole

# ===================== 工作线程基类 =====================
class WorkerThread(QThread):
    """工作线程基类，所有耗时操作都应该继承此类"""
    progress = pyqtSignal(int)  # 进度信号
    status = pyqtSignal(str)    # 状态信息信号
    finished = pyqtSignal(object)  # 完成信号，传递结果
    error = pyqtSignal(str)     # 错误信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_cancelled = False

    def cancel(self):
        """取消当前操作"""
        self._is_cancelled = True

    def run(self):
        """子类必须实现此方法"""
        raise NotImplementedError


# ===================== 数据读取线程 =====================
class DataReadWorker(WorkerThread):
    """数据读取工作线程 - 包含完整的校准处理功能"""

    def __init__(self, file_path: str, config: ElectroVoyant, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.config = config
        self._is_cancelled = False

    def run(self):
        try:
            self.status.emit("正在读取数据文件...")  # type: ignore
            self.progress.emit(5)  # type: ignore

            # 获取MAT文件所在目录
            mat_file_dir = os.path.dirname(self.file_path)
            print(f"MAT文件所在目录: {mat_file_dir}")

            # 读取MAT数据 - 同时读取TSData和times
            self.status.emit("正在解析MAT文件...")  # type: ignore
            self.progress.emit(10)  # type: ignore

            data_interface = DataInterface(self.file_path, self.config.sample_rate, self.config.start_time)
            data = data_interface.read_mat_file(target_datasets=["TSData", "times"])

            ts_data = data["TSData"]
            mat_times = data["times"]  # 从MAT文件中读取times数据

            self.progress.emit(20)  # type: ignore
            self.status.emit("数据读取完成，开始处理校准...")  # type: ignore

            # ========== 校准过程 ==========
            # 如果有校准文件，加载并应用校准（放在数据维度处理前）
            cal_processor = None
            if hasattr(self.config, 'calibration_files') and self.config.calibration_files:
                self.status.emit("正在加载校准文件...")  # type: ignore
                self.progress.emit(25)  # type: ignore

                print("正在加载校准文件...")

                # 获取校准文件列表
                fnCAL = self.config.calibration_files

                # 如果fnCAL是空格分隔的字符串，转换为列表
                if isinstance(fnCAL, str):
                    fnCAL_list = fnCAL.split()
                else:
                    fnCAL_list = list(fnCAL)

                print(f"校准文件列表: {fnCAL_list}")

                # 检查校准文件是否存在
                missing_files = []
                valid_files = []
                for cal_file in fnCAL_list:
                    cal_file_path = os.path.join(mat_file_dir, cal_file)
                    if os.path.exists(cal_file_path):
                        valid_files.append(cal_file)
                    else:
                        missing_files.append(cal_file)

                if missing_files:
                    print(f"警告: 以下校准文件不存在: {missing_files}")
                    # 由于在线程中不能直接弹出对话框，需要通过信号通知主线程
                    self.status.emit(f"校准文件缺失: {missing_files}，将跳过校准")  # type: ignore
                else:
                    # 所有校准文件都存在，创建校准处理器
                    cal_processor = self.create_calibration_processor(mat_file_dir, fnCAL_list)
                    cal_data_list = cal_processor.load_calibrations()

                    # 获取通道索引，用于确定哪个通道应用哪个校准文件
                    if hasattr(self.config, 'channel_indices'):
                        channel_indices = self.config.channel_indices
                        # 如果是MATLAB的1-based索引，转换为0-based
                        if isinstance(channel_indices, (list, np.ndarray)) and len(channel_indices) > 0:
                            if channel_indices[0] >= 1:
                                channel_indices_py = [idx - 1 for idx in channel_indices]
                            else:
                                channel_indices_py = channel_indices
                        else:
                            # 默认通道顺序
                            channel_indices_py = list(range(min(ts_data.shape[1], 5)))
                    else:
                        # 默认通道顺序
                        channel_indices_py = list(range(min(ts_data.shape[1], 5)))

                    # 应用校准到原始数据
                    if cal_data_list and len(cal_data_list) > 0:
                        self.status.emit("正在应用校准...")  # type: ignore
                        self.progress.emit(30)  # type: ignore

                        print("应用校准到原始数据...")

                        # 定义通道名称映射
                        channel_mapping = {i: ['EX', 'EY', 'HX', 'HY', 'HZ'][i] for i in
                                           range(min(5, ts_data.shape[1]))}

                        # 遍历校准文件并应用到对应的通道
                        # 假设校准文件顺序对应：HX, HY, HZ
                        magnetic_channels = ['HX', 'HY', 'HZ']

                        for cal_idx, cal_data in enumerate(cal_data_list):
                            if cal_data is None:
                                continue

                            # 确定当前校准文件对应的通道
                            if cal_idx < len(magnetic_channels):
                                target_channel = magnetic_channels[cal_idx]

                                # 找到目标通道在数据中的索引
                                channel_idx = None
                                for i, ch_name in channel_mapping.items():
                                    if ch_name == target_channel and i in channel_indices_py:
                                        channel_idx = i
                                        break

                                if channel_idx is not None and channel_idx < ts_data.shape[1]:
                                    # 应用校准
                                    print(f"  对通道 {target_channel} 应用校准 {fnCAL_list[cal_idx]}")

                                    # 使用校准处理器应用校准
                                    calibrated_channel_data = cal_processor.apply_calibration(
                                        ts_data[:, channel_idx],
                                        channel_idx=cal_idx
                                    )

                                    # 更新原始数据
                                    ts_data[:, channel_idx] = calibrated_channel_data
                                else:
                                    print(f"  警告: 找不到校准文件 {fnCAL_list[cal_idx]} 对应的通道 {target_channel}")
                            else:
                                print(f"  警告: 校准文件索引 {cal_idx} 超出预期范围")

            self.progress.emit(40)  # type: ignore
            self.status.emit("校准处理完成，进行数据维度处理...")  # type: ignore

            # ========== 数据维度处理 ==========
            print(f"原始TSData形状: {ts_data.shape}")

            if ts_data.shape[0] == 5 and ts_data.shape[1] > 5:
                ts_data = ts_data.T
                print(f"TSData转置后形状: {ts_data.shape}")
            elif ts_data.shape[1] == 5 and ts_data.shape[0] > 5:
                print(f"TSData已正确排列: {ts_data.shape}")
            else:
                print(f"TSData形状异常: {ts_data.shape}")
                if ts_data.size > 10:
                    if ts_data.shape[0] > ts_data.shape[1]:
                        print(f"假设TSData不需要转置")
                    else:
                        ts_data = ts_data.T
                        print(f"假设TSData需要转置，转置后形状: {ts_data.shape}")
                else:
                    raise ValueError(f"无法处理TSData的形状: {ts_data.shape}")

            self.progress.emit(50)  # type: ignore
            self.status.emit("处理时间数据...")  # type: ignore

            # ========== 检查times数据 ==========
            if mat_times is None or len(mat_times) == 0:
                print("警告: MAT文件中没有times数据，将使用配置的起始时间生成时间序列")
                # 如果MAT文件中没有times，则使用配置的起始时间生成
                start_time = self.config.start_time
                time_points = ts_data.shape[0]
                time_index = pd.date_range(start=start_time, periods=time_points, freq='s')
            else:
                # 将MAT文件中的times转换为datetime索引
                print(f"从MAT文件中读取到times数据，长度: {len(mat_times)}")

                # 检查times的数据类型并转换为datetime
                if isinstance(mat_times[0], str):
                    # 如果是字符串格式，直接解析
                    time_index = pd.to_datetime(mat_times, format='%Y-%m-%dT%H:%M:%S.%f', utc=True)
                elif isinstance(mat_times[0], np.datetime64):
                    # 如果是numpy的datetime64格式
                    time_index = pd.DatetimeIndex(mat_times)
                else:
                    # 尝试其他常见格式
                    try:
                        time_index = pd.to_datetime(mat_times)
                    except Exception as e:
                        print(f"无法解析times数据: {e}")
                        print("将使用配置的起始时间生成时间序列")
                        start_time = self.config.start_time
                        time_points = ts_data.shape[0]
                        time_index = pd.date_range(start=start_time, periods=time_points, freq='s')

            self.progress.emit(60)  # type: ignore
            self.status.emit("创建DataFrame...")  # type: ignore

            # ========== 创建DataFrame ==========
            columns = ['EX', 'EY', 'HX', 'HY', 'HZ']
            # 确保列数与数据通道数匹配
            if len(columns) > ts_data.shape[1]:
                columns = columns[:ts_data.shape[1]]

            df = pd.DataFrame(ts_data, index=time_index, columns=columns).astype('float32')

            print(f"创建DataFrame完成，形状: {df.shape}")
            print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")

            self.progress.emit(70)  # type: ignore
            self.status.emit("根据配置截取时间范围...")  # type: ignore

            # ========== 根据配置的时间范围截取数据 ==========
            if hasattr(self.config, 'start_time') and hasattr(self.config, 'end_time'):
                sTime = pd.to_datetime(self.config.start_time)
                eTime = pd.to_datetime(self.config.end_time)

                print(f"截取时间范围: {sTime} 到 {eTime}")

                # 确保时间在DataFrame的时间范围内
                if sTime >= df.index[0] and eTime <= df.index[-1]:
                    df = df.loc[sTime:eTime]
                    print(f"截取后DataFrame形状: {df.shape}")
                else:
                    print("警告: 指定的时间范围超出数据范围，使用完整数据")
            else:
                print("未找到start_time和end_time配置，使用完整数据")

            self.progress.emit(90)  # type: ignore
            self.status.emit("清理内存...")  # type: ignore

            # ========== 清理内存 ==========
            del ts_data, data
            import gc
            gc.collect()

            self.progress.emit(100)  # type: ignore
            self.status.emit("数据读取完成！")  # type: ignore

            # 返回结果
            result = {
                'df': df,
                'data_interface': data_interface,
                'file_path': self.file_path,
                'cal_processor': cal_processor  # 可选：返回校准处理器
            }
            self.finished.emit(result)  # type: ignore

        except Exception as e:
            import traceback
            error_msg = f"数据读取失败: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)  # type: ignore

    def create_calibration_processor(self, mat_file_dir, fnCAL_list):
        """创建校准处理器（与原始代码完全一致）"""

        # 创建简化的校准处理器
        class SimpleCalibrationProcessor:
            def __init__(self, base_dir, cal_files):
                self.base_dir = base_dir
                self.cal_files = cal_files
                self.cal_data_list = []

            def load_calibrations(self):
                """加载校准文件"""
                self.cal_data_list = []
                for cal_file in self.cal_files:
                    try:
                        cal_file_path = os.path.join(self.base_dir, cal_file)
                        cal_data = self.read_calibration_file(cal_file_path)
                        self.cal_data_list.append(cal_data)
                        print(f"  成功读取校准文件: {cal_file}")
                    except Exception as e:
                        print(f"  读取校准文件失败: {cal_file}, 错误: {e}")
                        self.cal_data_list.append(None)
                return self.cal_data_list

            @staticmethod
            def read_calibration_file(file_path):
                """读取校准文件"""
                print(f"读取校准文件: {file_path}")

                # 检查文件扩展名
                if file_path.endswith('.txt') or file_path.endswith('.cal') or file_path.endswith('.cmt'):
                    # 读取文本格式的校准文件
                    data = []
                    with open(file_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith('#'):
                                continue

                            # 尝试解析为数字
                            try:
                                # 处理可能是空格或制表符分隔的数据
                                parts = line.split()
                                if len(parts) >= 2:
                                    # 假设是频率和幅度/相位数据
                                    freq = float(parts[0])
                                    value = float(parts[1])
                                    data.append([freq, value])
                                elif len(parts) == 1:
                                    # 如果是单个数值（如缩放因子）
                                    data.append(float(parts[0]))
                            except ValueError:
                                print(f"    警告: 无法解析行: {line}")

                    if len(data) == 1:
                        return data[0]  # 返回单个数值
                    else:
                        return np.array(data)  # 返回数组

                elif file_path.endswith('.mat'):
                    # 读取MAT格式的校准文件
                    import scipy.io as sio
                    cal_data = sio.loadmat(file_path)
                    return cal_data

                else:
                    # 尝试按二进制读取
                    try:
                        data = np.fromfile(file_path, dtype=np.float32)
                        return data
                    except:
                        raise ValueError(f"无法识别的校准文件格式: {file_path}")

            def apply_calibration(self, data, channel_idx=0):
                """应用校准到数据"""
                if channel_idx >= len(self.cal_data_list) or self.cal_data_list[channel_idx] is None:
                    return data

                cal_data = self.cal_data_list[channel_idx]

                # 根据校准数据类型应用不同的校准方法
                if np.isscalar(cal_data):
                    # 简单缩放因子
                    print(f"    应用缩放因子: {cal_data}")
                    return data * cal_data
                elif isinstance(cal_data, np.ndarray):
                    if cal_data.ndim == 1:
                        # 一维数组，可能是简单的校准曲线
                        print(f"    应用一维校准曲线，长度: {len(cal_data)}")
                        if len(cal_data) == len(data):
                            return data * cal_data
                        else:
                            print(f"    警告: 校准曲线长度({len(cal_data)})与数据长度({len(data)})不匹配")
                            return data
                    elif cal_data.ndim == 2 and cal_data.shape[1] == 2:
                        # 二维数组，可能是频率响应数据（频率, 响应）
                        print(f"    应用频率响应校准，点数: {cal_data.shape[0]}")
                        # 这里需要实现频域校准
                        return data
                    else:
                        print(f"    警告: 未知的校准数据格式，形状: {cal_data.shape}")
                        return data
                else:
                    print(f"    警告: 未知的校准数据类型: {type(cal_data)}")
                    return data

        return SimpleCalibrationProcessor(mat_file_dir, fnCAL_list)


# ===================== 滤波处理线程 =====================
class FilterWorker(WorkerThread):
    """滤波处理工作线程"""

    def __init__(self, data_interface, filter_type: str, params: Dict, parent=None):
        super().__init__(parent)
        self.data_interface = data_interface
        self.filter_type = filter_type
        self.params = params

    def run(self):
        try:
            self.status.emit(f"正在应用{self.filter_type}滤波...")    # type: ignore
            self.progress.emit(10)  # type: ignore

            if self.filter_type == "FIR":
                cutoff = self.params.get('cutoff')
                num_taps = self.params.get('num_taps')
                btype = self.params.get('btype')

                self.progress.emit(30)  # type: ignore
                filtered_data = self.data_interface.apply_fir_filter(cutoff, num_taps, btype)

            elif self.filter_type == "IIR":
                cutoff = self.params.get('cutoff')
                order = self.params.get('order')
                iir_type = self.params.get('iir_type')
                btype = self.params.get('btype')
                ripple = self.params.get('ripple')

                self.progress.emit(30)  # type: ignore
                filtered_data = self.data_interface.apply_iir_filter(cutoff, order, iir_type, btype, ripple)

            elif self.filter_type == "Notch":
                # 级联带阻滤波
                harmonics = self.params.get('harmonics', [])
                filtered_data = self.data_interface.TSData.copy()

                for i, cutoff in enumerate(harmonics):
                    if self._is_cancelled:
                        self.status.emit("操作已取消")   # type: ignore
                        return

                    progress = 30 + int(60 * (i + 1) / len(harmonics))
                    self.progress.emit(progress)    # type: ignore
                    self.status.emit(f"处理频带 {i + 1}/{len(harmonics)}: {cutoff[0]}-{cutoff[1]}Hz")   # type: ignore

                    self.data_interface.TSData = filtered_data
                    filtered_data = self.data_interface.apply_iir_filter(
                        cutoff, 4, 'butter', 'bandstop', 65
                    )

                    if filtered_data is None:
                        raise Exception(f"频带 {cutoff} 滤波失败")

            elif self.filter_type == "Adaptive":
                algorithm = self.params.get('algorithm')
                filter_length = self.params.get('filter_length')
                step_size = self.params.get('step_size')

                self.progress.emit(30)  # type: ignore
                filtered_data = self.data_interface.apply_adaptive_filter(algorithm, filter_length, step_size)

            else:
                raise Exception(f"未知的滤波类型: {self.filter_type}")

            self.progress.emit(90)  # type: ignore
            self.status.emit("滤波完成")    # type: ignore

            if filtered_data is not None:
                result = {
                    'filtered_data': filtered_data,
                    'filter_type': self.filter_type
                }
                self.finished.emit(result)  # type: ignore
            else:
                self.error.emit("滤波处理返回空数据")    # type: ignore

        except Exception as e:
            self.error.emit(f"滤波处理失败: {str(e)}")    # type: ignore


# ===================== PSD计算线程 =====================
class PSDWorker(WorkerThread):
    """PSD计算工作线程"""

    def __init__(self, data_interface, nperseg: int, use_filtered: bool = True, parent=None):
        super().__init__(parent)
        self.data_interface = data_interface
        self.nperseg = nperseg
        self.use_filtered = use_filtered

    def run(self):
        try:
            self.status.emit("正在计算功率谱密度...")    # type: ignore
            self.progress.emit(20)  # type: ignore

            original_data = self.data_interface.TSData
            if self.use_filtered and hasattr(self.data_interface, 'filtered_data'):
                self.data_interface.TSData = self.data_interface.filtered_data

            self.progress.emit(40)  # type: ignore
            psd_results = self.data_interface.welch_psd(nperseg=self.nperseg)

            self.progress.emit(80)  # type: ignore

            if not psd_results:
                raise Exception("PSD计算返回空结果")

            # 转换为DataFrame
            channels = list(psd_results.keys())
            freqs = psd_results[channels[0]]['freqs']
            psd_data = {}
            for channel in channels:
                psd_data[channel] = psd_results[channel]['psd']

            psd_df = pd.DataFrame(psd_data, index=freqs)
            psd_df.index.name = '频率 (Hz)'

            self.progress.emit(100) # type: ignore
            self.status.emit("PSD计算完成") # type: ignore

            # 恢复原始数据
            self.data_interface.TSData = original_data

            self.finished.emit(psd_df)  # type: ignore

        except Exception as e:
            self.error.emit(f"PSD计算失败: {str(e)}")   # type: ignore


# ===================== Allan方差计算线程 =====================
class AllanWorker(WorkerThread):
    """Allan方差计算工作线程"""

    def __init__(self, data_interface, use_filtered: bool = True, parent=None):
        super().__init__(parent)
        self.data_interface = data_interface
        self.use_filtered = use_filtered

    def run(self):
        try:
            self.status.emit("正在计算Allan方差...")  # type: ignore
            self.progress.emit(10)  # type: ignore

            original_data = self.data_interface.TSData
            if self.use_filtered and hasattr(self.data_interface, 'filtered_data'):
                self.data_interface.TSData = self.data_interface.filtered_data

            max_channels = min(5, self.data_interface.TSData.shape[1])

            allan_data = {}
            total = max_channels

            for i in range(max_channels):
                if self._is_cancelled:
                    self.status.emit("操作已取消")   # type: ignore
                    return

                progress = 20 + int(70 * (i + 1) / total)
                self.progress.emit(progress)    # type: ignore
                self.status.emit(f"处理通道 {i + 1}/{max_channels}...") # type: ignore

                taus, avar, _, _ = self.data_interface.calculate_allan_variance_in(channel=i)
                if taus is not None and avar is not None:
                    channel_name = ['EX', 'EY', 'HX', 'HY', 'HZ'][i] if i < 5 else f'Ch{i}'
                    allan_data[channel_name] = pd.Series(avar, index=taus)

            self.progress.emit(95)  # type: ignore

            if not allan_data:
                raise Exception("没有有效的Allan方差数据")

            allan_df = pd.DataFrame(allan_data)
            allan_df.index.name = '平均时间 τ (s)'

            self.progress.emit(100) # type: ignore
            self.status.emit("Allan方差计算完成") # type: ignore

            self.data_interface.TSData = original_data
            self.finished.emit(allan_df)    # type: ignore

        except Exception as e:
            self.error.emit(f"Allan方差计算失败: {str(e)}")   # type: ignore


# ===================== 小波变换线程 =====================
class WaveletWorker(WorkerThread):
    """小波变换计算线程"""

    def __init__(self, data_interface, channel_idx: int, use_filtered: bool = True, parent=None):
        super().__init__(parent)
        self.data_interface = data_interface
        self.channel_idx = channel_idx
        self.use_filtered = use_filtered

    def run(self):
        try:
            self.status.emit("正在计算小波变换...") # type: ignore
            self.progress.emit(20)  # type: ignore

            original_data = self.data_interface.TSData
            if self.use_filtered and hasattr(self.data_interface, 'filtered_data'):
                self.data_interface.TSData = self.data_interface.filtered_data

            self.progress.emit(40)  # type: ignore

            cwtmatr, frequencies, scales = self.data_interface.wavelet_transform(channel=self.channel_idx)

            if cwtmatr is None or frequencies is None:
                raise Exception("小波变换计算失败")

            self.progress.emit(80)  # type: ignore

            # 构建时间轴
            time_points = self.data_interface.TSData.shape[0]
            time = np.arange(time_points) / self.data_interface.sample_rate

            result = {
                'cwtmatr': cwtmatr,
                'frequencies': frequencies,
                'time': time,
                'scales': scales
            }

            self.progress.emit(100) # type: ignore
            self.status.emit("小波变换计算完成")    # type: ignore

            self.data_interface.TSData = original_data
            self.finished.emit(result)  # type: ignore

        except Exception as e:
            self.error.emit(f"小波变换计算失败: {str(e)}")  # type: ignore


# ===================== 级联滤波线程 =====================
class CascadeFilterWorker(WorkerThread):
    """级联滤波处理线程"""

    def __init__(self, file_path: str, fs_guess: int = 300, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.fs_guess = fs_guess

    def run(self):
        try:
            self.status.emit("开始级联滤波处理...") # type: ignore
            self.progress.emit(10)  # type: ignore

            cf = CascadeFiltering()
            cf.show_hampel_flag = True
            cf.show_fir_flag = True

            self.status.emit("读取数据...") # type: ignore
            self.progress.emit(20)  # type: ignore

            # 直接调用全流程处理
            cf.smooth_kf_pipeline(infile=self.file_path, fs_guess=self.fs_guess)

            self.progress.emit(90)  # type: ignore
            self.status.emit("级联滤波完成")  # type: ignore

            self.finished.emit(cf)  # type: ignore

        except Exception as e:
            self.error.emit(f"级联滤波失败: {str(e)}")    # type: ignore


# 继承QObject以支持信号槽（解决线程安全）
class OutputRedirector(QObject):
    # 定义信号：传递要显示的文本（Qt要求信号必须在QObject子类中定义）
    text_signal = pyqtSignal(str)

    def __init__(self, plain_text_edit):
        super().__init__()
        self.plain_text_edit = plain_text_edit
        # 绑定信号到UI更新方法（信号槽自动保证主线程执行）
        # noinspection PyUnresolvedReferences
        self.text_signal.connect(self._append_text)
        # 记录原始输出流，用于恢复
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def write(self, text):
        """重写write方法：拦截print输出，通过信号传递文本"""
        # 过滤空行（避免print()输出空行）
        stripped_text = text.strip()
        if stripped_text:
            # noinspection PyUnresolvedReferences
            self.text_signal.emit(stripped_text)  # 发送信号（线程安全）

    def flush(self):
        """兼容标准输出的flush方法，必须实现"""
        pass

    def _append_text(self, text):
        """实际更新UI的方法（由信号触发，确保在主线程执行）"""
        self.plain_text_edit.appendPlainText(text)
        # 自动滚动到最新内容（提升体验）
        scroll_bar = self.plain_text_edit.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def restore(self):
        """恢复系统默认输出流（窗口关闭时调用）"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

# =================================================================  弹窗模块 ===========================================

# =====================  配置表格弹窗 =====================


class ConfigTableDialog(QDialog):
    """配置信息展示弹窗（内置QTableWidget）"""

    def __init__(self, config_dict, parent=None):
        super().__init__(parent)
        self.tableWidget = None
        self.config_dict = config_dict
        self.parent = parent
        self.setWindowTitle("配置信息")
        self.setModal(True)
        self.resize(800, 600)
        self.init_ui()
        self.fill_config_data()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        self.tableWidget = QTableWidget()
        main_layout.addWidget(self.tableWidget)

    def fill_config_data(self):
        self.tableWidget.setRowCount(len(self.config_dict))
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(["参数", "值"])
        row = 0
        for key, value in self.config_dict.items():
            item_key = QtWidgets.QTableWidgetItem(key)
            item_value = QtWidgets.QTableWidgetItem(str(value))
            self.tableWidget.setItem(row, 0, item_key)
            self.tableWidget.setItem(row, 1, item_value)
            row += 1
        self.tableWidget.setColumnWidth(0, 200)
        self.tableWidget.setColumnWidth(1, 400)


# ===================== 原始数据展示弹窗（内置QTableView） =====================
class DataTableDialog(QDialog):
    """MAT数据展示弹窗（内置QTableView）"""

    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.tableView = None
        self.table_model = None
        self.df = df  # 接收DataFrame数据
        self.parent = parent
        self.setWindowTitle("数据预览")
        self.setModal(True)
        self.resize(800, 600)  # 更大的弹窗尺寸适配表格
        self.init_ui()
        self.fill_table_data()

    def init_ui(self):
        """初始化弹窗UI：核心是QTableView"""
        main_layout = QVBoxLayout(self)

        # 创建QTableView（替代原主界面的tableView）
        self.tableView = QTableView()
        # 设置表格属性
        self.tableView.setAlternatingRowColors(True)  # 隔行变色，提升可读性
        self.tableView.setSortingEnabled(True)  # 允许点击表头排序
        self.tableView.horizontalHeader().setStretchLastSection(True)  # 最后一列自适应
        main_layout.addWidget(self.tableView)

    def fill_table_data(self):
        """填充数据到QTableView（适配大数据集）"""
        # 大数据集仅显示前1000行预览
        display_df = self.df.head(1000) if len(self.df) > 10000 else self.df

        # 绑定PandasTableModel到QTableView
        self.table_model = PandasTableModel(display_df)
        self.tableView.setModel(self.table_model)

        # 自动调整列宽
        self.tableView.resizeColumnsToContents()
        # 设置列宽策略（拉伸适配）
        header = self.tableView.horizontalHeader()
        for col in range(self.table_model.columnCount()):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)

        # 提示信息
        if len(self.df) > 10000:
            print(f"提示：数据共{len(self.df)}行，仅显示前1000行预览")


# ===================== 通用DataFrame弹窗 =====================

class DataFrameDialog(QDialog):
    """通用的DataFrame表格展示弹窗"""

    def __init__(self, df, title="数据表格", parent=None):
        super().__init__(parent)
        self.tableView = None
        self.table_model = None
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(1000, 600)  # 弹窗初始尺寸

        # 初始化UI
        self.init_ui(df)

    def init_ui(self, df):
        """初始化弹窗UI：使用PandasTableModel展示DataFrame"""
        main_layout = QVBoxLayout(self)

        # 创建表格视图
        self.tableView = QTableView()
        # 设置表格样式（与原逻辑一致）
        self.tableView.setAlternatingRowColors(True)
        self.tableView.setSortingEnabled(True)

        # 绑定PandasTableModel
        self.table_model = PandasTableModel(df)
        self.tableView.setModel(self.table_model)

        # 自适应列宽/行高（与原逻辑一致）
        self.tableView.resizeColumnsToContents()
        self.tableView.resizeRowsToContents()

        main_layout.addWidget(self.tableView)

# ===================== PDF统计结果弹窗=====================


class PSDResultDialog(QDialog):
    """PSD/概率密度统计结果弹窗"""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.tableView = None
        self.table_model = None
        self.setWindowTitle("功率谱概率密度统计结果")
        self.setModal(True)
        self.resize(1000, 600)  # 弹窗初始尺寸

        # 初始化UI
        self.init_ui(model)

    def init_ui(self, model):
        """初始化弹窗UI：包含表格视图"""
        main_layout = QVBoxLayout(self)

        # 创建表格视图
        self.tableView = QTableView()
        self.tableView.setAlternatingRowColors(True)  # 隔行变色
        self.tableView.setSortingEnabled(True)  # 允许排序
        self.tableView.setModel(model)  # 绑定数据模型

        # 自适应列宽/行高
        for col in range(model.columnCount()):
            self.tableView.resizeColumnsToContents()
            self.tableView.resizeRowsToContents()

        main_layout.addWidget(self.tableView)

# ===================== PandasTableModel（适配QTableView） =====================


class PandasTableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent: QtCore.QModelIndex = None) -> int:
        return self._data.shape[0]

    def columnCount(self, parent: QtCore.QModelIndex = None) -> int:
        return self._data.shape[1]

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid():
            if role == QtCore.Qt.ItemDataRole.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = DISPLAY_ROLE) -> Any:
        if orientation == HORIZONTAL and role == DISPLAY_ROLE:
            return self._data.columns[section]
        return None


# ================================================== 主窗口模块 ==========================================================
class NewMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.calibrator = None
        self.selected_cal_files_label = None
        self.channel_mapping_spins = None
        self.wlen = None
        self.toolbar = None
        self.wavelet_results = None
        self.allan_results = None
        self.psd_results = None
        self.raw_time_data = None
        self.view_window_4 = None
        self.view_window_3 = None
        self.view_window = None
        self.view_window_2 = None
        self.psd_pdf_results = None
        self.label = None
        self.canvas = None
        self.figure = None
        self.last_selected_text = None
        self.setupUi(self)

        # 输出重定向
        self.redirector = OutputRedirector(self.plainTextEdit)
        sys.stdout = self.redirector
        sys.stderr = self.redirector

        self.config = None
        self.df = None
        self.create_canvas()
        self.data_interface = None
        self.connections()
        self.axes = None

        # ========== 多线程相关 ==========
        self.current_worker = None  # 当前工作线程
        self.progress_bar = None    # 进度条
        self._init_progress_bar()

    def _init_progress_bar(self):
        """初始化进度条"""
        self.progress_bar = QProgressBar(self.statusBar())
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

    def _start_worker(self, worker: WorkerThread, show_progress: bool = True):
        """启动工作线程"""
        # 如果已有线程在运行，询问是否中断
        if self.current_worker and self.current_worker.isRunning():
            reply = QMessageBox.question(
                self,
                "确认",
                "当前有任务正在运行，是否取消？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.current_worker.cancel()
                self.current_worker.wait()
            else:
                return

        # 连接信号
        worker.progress.connect(self._on_progress)  # type: ignore
        worker.status.connect(self._on_status)  # type: ignore
        worker.finished.connect(self._on_worker_finished)  # type: ignore
        worker.error.connect(self._on_worker_error)  # type: ignore

        # 显示进度条
        if show_progress:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

        # 保存并启动线程
        self.current_worker = worker
        worker.start()

    def _on_progress(self, value: int):
        """进度更新"""
        self.progress_bar.setValue(value)

    def _on_status(self, message: str):
        """状态更新"""
        self.statusBar().showMessage(message)
        print(message)  # 同时输出到控制台

    def _on_worker_finished(self, result):
        """线程完成处理"""
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("完成", 3000)
        self.current_worker = None

        # 处理结果（子类需要重写或使用回调）
        self._handle_worker_result(result)

    def _on_worker_error(self, error_msg: str):
        """线程错误处理"""
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("错误", 3000)
        self.current_worker = None

        QMessageBox.critical(self, "错误", error_msg)
        print(f"错误: {error_msg}")

    def _handle_worker_result(self, result):
        """处理线程结果（子类可重写）"""
        pass

    def connections(self, calibrator=None):
        # 修复QAction信号绑定
        self.action1.triggered.connect(self.quick_read)  # 一键读取
        self.action2.triggered.connect(self.read_mat_file)  # 读取mat时间序列数据
        self.action3.triggered.connect(self.parse)  # 读取cfg参数
        self.action4.triggered.connect(self.open_view_interface_2)  # 打开工程配置窗口
        self.action5.triggered.connect(self.def_cfg)  # 修改参数
        self.action1_2.triggered.connect(self.finite_impulse_response_filter)  # FIR滤波
        self.action2_2.triggered.connect(self.infinite_impulse_response_filter)  # IIR滤波
        self.action3_2.triggered.connect(self.adaptive_filter)  # 自适应滤波
        self.action4_2.triggered.connect(self.notch_filter_butterworth)  # 快速带阻滤波
        self.actionpsd.triggered.connect(self.calculate_psd)  # 计算功率谱密度
        self.action01.triggered.connect(self.plot_power_spectrum)  # 绘制功率谱密度图
        self.actionpdf.triggered.connect(self.calculate_psd_pdf)  # 计算功率谱概率密度
        self.action02.triggered.connect(self.plot_pdf)  # 绘制功率谱概率密度图
        self.actionallen.triggered.connect(self.calculate_allan_variance)  # 计算艾伦方差
        self.action03.triggered.connect(self.plot_allan_variance)  # 绘制艾伦方差图
        self.actionwavelt.triggered.connect(self.calculate_wavelet_transform)  # 计算小波变换
        self.action04.triggered.connect(self.plot_wavelet_transform)  # 绘制小波变换图
        self.actiondegress.triggered.connect(self.apply_detrend)  # 去趋势
        self.actiondegress_2.triggered.connect(self.de_instrumental_response)  # 去仪器响应
        self.actioncalibration.triggered.connect(self.apply_calibration)  # 电磁场校正
        self.actionpca.triggered.connect(self.pca_filter)  # 主成分分析
        self.action05.triggered.connect(self.pca_filter)  # 绘制主成分分析图
        self.action06.triggered.connect(self.debug_cascade_filtering)  # 绘制级联滤波图
        self.action07.triggered.connect(self.update_plot)  # 绘制时间序列图
        self.action08.triggered.connect(self.plot_raw_vs_filtered)  # 绘制时域处理滤波前后对比图
        self.action09.triggered.connect(self.plot_psd_comparison)  # 绘制功率谱滤波前后对比图
        self.action10.triggered.connect(self.plot_allan_comparison)  # 绘制艾伦方差滤波前后对比图
        self.action11.triggered.connect(self.plot_calibration_response)  # 绘制标定频率响应曲线
        self.actionreviewmap.triggered.connect(self.open_view_interface)  # 打开场景窗口
        self.actionjocread.triggered.connect(self.open_view_interface_2)  # 打开工程配置环境

    # ===================== 文件读取模块 =====================

    def def_cfg(self):
        """自定义配置信息 - 打开配置对话框"""
        dialog = DialogWindow(self)  # 创建配置对话框

        # 如果有现有配置，则加载到对话框中
        if hasattr(self, 'config') and self.config:
            # 将ElectroVoyant配置转换为对话框配置字典
            config_dict = {
                "传感器灵敏度 (mV/nT)": self.config.sensor_sensitivity,
                "数据目录": self.config.data_directory,
                "数据文件列表": self.config.data_list_file,
                "采样率 (Hz)": self.config.sample_rate,
                "通道数量": self.config.channel_count,
                "通道索引": self.config.channel_indices,
                "通道增益": self.config.channel_gains,
                "电长度 (米)": self.config.electrical_lengths,
                "开始时间": self.config.start_time,
                "结束时间": self.config.end_time,
                "FFT窗口长度": self.config.fft_window_length,
                "校准文件": self.config.calibration_files
            }
            dialog.set_config_dict(config_dict)
        else:
            # 如果没有现有配置，使用默认配置
            dialog.def_config()

        # 显示对话框
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            # 获取用户配置
            new_config = dialog.get_config_dict()

            # 更新主窗口配置对象
            if not hasattr(self, 'config'):
                # 如果没有config对象，需要创建一个（根据您的实际结构调整）
                # 这里假设您有一个创建配置对象的方法
                pass

            # 更新配置参数
            self.config.sensor_sensitivity = new_config["传感器灵敏度 (mV/nT)"]
            self.config.data_directory = new_config["数据目录"]
            self.config.data_list_file = new_config["数据文件列表"]
            self.config.sample_rate = new_config["采样率 (Hz)"]
            self.config.channel_count = new_config["通道数量"]
            self.config.channel_indices = new_config["通道索引"]
            self.config.channel_gains = new_config["通道增益"]
            self.config.electrical_lengths = new_config["电长度 (米)"]
            self.config.start_time = new_config["开始时间"]
            self.config.end_time = new_config["结束时间"]
            self.config.fft_window_length = new_config["FFT窗口长度"]
            self.config.calibration_files = new_config["校准文件"]

            print("配置已更新:")
            for key, value in new_config.items():
                print(f"  {key}: {value}")
        else:
            print("配置未更改")

    def quick_read(self):
        self.parse()
        if self.config and hasattr(self.config, 'data_directory') and hasattr(self.config, 'data_list_file'):
            file_path = os.path.join(self.config.data_directory, self.config.data_list_file)
            self.read_mat_file(file_path)
        else:
            print("错误：配置文件未成功解析，无法获取数据路径")

    def parse(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "选择配置文件", "", "配置文件 (*.cfg);;所有文件 (*)"
        )
        if not file_path:
            return
        try:
            self.lineEdit.setText(file_path)
            self.config = ElectroVoyant(file_path)
            self.config.parse()
            print("配置文件解析成功:")
            self.update_ui_with_config()
        except Exception as e:
            print(f"配置解析出错: {e}")
            import traceback
            traceback.print_exc()

    def read_mat_file(self, file_path=None):
        """多线程读取MAT文件（包含完整功能）"""
        if not file_path:
            if not hasattr(self, 'config') or not self.config:
                print("请先导入配置文件")
                return
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                None, "选择MAT文件", self.config.data_directory, "MAT文件 (*.mat);;所有文件 (*)"
            )
            if not file_path:
                return

        self.lineEdit.setText(file_path)
        print(f"正在读取文件: {file_path}")

        # 检查校准文件是否存在（这部分在线程外进行，因为需要用户交互）
        mat_file_dir = os.path.dirname(file_path)

        # 检查校准文件
        missing_files = []
        if hasattr(self.config, 'calibration_files') and self.config.calibration_files:
            fnCAL = self.config.calibration_files
            if isinstance(fnCAL, str):
                fnCAL_list = fnCAL.split()
            else:
                fnCAL_list = list(fnCAL)

            for cal_file in fnCAL_list:
                cal_file_path = os.path.join(mat_file_dir, cal_file)
                if not os.path.exists(cal_file_path):
                    missing_files.append(cal_file)

            if missing_files:
                print(f"警告: 以下校准文件不存在: {missing_files}")
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "警告",
                    f"找不到以下校准文件:\n{', '.join(missing_files)}\n是否继续加载数据而不应用校准？",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No
                )
                if reply == QtWidgets.QMessageBox.StandardButton.No:
                    print("用户取消加载")
                    return
                else:
                    print("继续加载数据，不应用校准")
                    # 从配置中移除缺失的校准文件
                    self.config.calibration_files = [f for f in fnCAL_list if f not in missing_files]

        # 创建工作线程
        worker = DataReadWorker(file_path, self.config, self)
        worker.finished.connect(self._on_data_read_finished)  # type: ignore
        worker.error.connect(self._on_data_read_error)  # type: ignore
        worker.progress.connect(self._on_progress)  # type: ignore
        worker.status.connect(self._on_status)  # type: ignore

        self._start_worker(worker)

    def _on_data_read_finished(self, result):
        """数据读取完成处理"""
        self.df = result['df']
        self.data_interface = result['data_interface']

        # 如果有校准处理器，可以保存
        if 'cal_processor' in result and result['cal_processor']:
            self.cal_processor = result['cal_processor']

        print(f"创建DataFrame完成，形状: {self.df.shape}")
        print(f"时间范围: {self.df.index[0]} 到 {self.df.index[-1]}")

        # 弹出数据展示弹窗
        data_dialog = DataTableDialog(self.df, parent=self)
        data_dialog.exec()

        # 更新图表
        if hasattr(self, 'update_plot'):
            self.update_plot()

    def _on_data_read_error(self, error_msg):
        """数据读取错误处理"""
        print(f"读取数据出错: {error_msg}")
        QtWidgets.QMessageBox.critical(self, "错误", f"数据读取失败:\n{error_msg}")

    def create_calibration_processor(self, mat_file_dir, fnCAL_list):
        """创建校准处理器"""

        # 创建简化的校准处理器
        class SimpleCalibrationProcessor:
            def __init__(self, base_dir, cal_files):
                self.base_dir = base_dir
                self.cal_files = cal_files
                self.cal_data_list = []

            def load_calibrations(self):
                """加载校准文件"""
                self.cal_data_list = []
                for cal_file in self.cal_files:
                    try:
                        cal_file_path = os.path.join(self.base_dir, cal_file)
                        cal_data = self.read_calibration_file(cal_file_path)
                        self.cal_data_list.append(cal_data)
                        print(f"  成功读取校准文件: {cal_file}")
                    except Exception as e:
                        print(f"  读取校准文件失败: {cal_file}, 错误: {e}")
                        self.cal_data_list.append(None)
                return self.cal_data_list

            @staticmethod
            def read_calibration_file(file_path):
                """读取校准文件"""
                # 这里需要根据你的校准文件格式来实现
                # 假设是文本文件，包含频率响应数据

                print(f"读取校准文件: {file_path}")

                # 检查文件扩展名
                if file_path.endswith('.txt') or file_path.endswith('.cal') or file_path.endswith('.cmt'):
                    # 读取文本格式的校准文件
                    data = []
                    with open(file_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith('#'):
                                continue

                            # 尝试解析为数字
                            try:
                                # 处理可能是空格或制表符分隔的数据
                                parts = line.split()
                                if len(parts) >= 2:
                                    # 假设是频率和幅度/相位数据
                                    freq = float(parts[0])
                                    value = float(parts[1])
                                    data.append([freq, value])
                                elif len(parts) == 1:
                                    # 如果是单个数值（如缩放因子）
                                    data.append(float(parts[0]))
                            except ValueError:
                                print(f"    警告: 无法解析行: {line}")

                    if len(data) == 1:
                        return data[0]  # 返回单个数值
                    else:
                        return np.array(data)  # 返回数组

                elif file_path.endswith('.mat'):
                    # 读取MAT格式的校准文件
                    import scipy.io as sio
                    cal_data = sio.loadmat(file_path)
                    return cal_data

                else:
                    # 尝试按二进制读取
                    try:
                        data = np.fromfile(file_path, dtype=np.float32)
                        return data
                    except:
                        raise ValueError(f"无法识别的校准文件格式: {file_path}")

            def apply_calibration(self, data, channel_idx=0):
                """应用校准到数据"""
                if channel_idx >= len(self.cal_data_list) or self.cal_data_list[channel_idx] is None:
                    return data

                cal_data = self.cal_data_list[channel_idx]

                # 根据校准数据类型应用不同的校准方法
                if np.isscalar(cal_data):
                    # 简单缩放因子
                    print(f"    应用缩放因子: {cal_data}")
                    return data * cal_data
                elif isinstance(cal_data, np.ndarray):
                    if cal_data.ndim == 1:
                        # 一维数组，可能是简单的校准曲线
                        print(f"    应用一维校准曲线，长度: {len(cal_data)}")
                        # 这里需要根据你的具体格式实现
                        # 简单示例：如果长度匹配，逐点相乘
                        if len(cal_data) == len(data):
                            return data * cal_data
                        else:
                            print(f"    警告: 校准曲线长度({len(cal_data)})与数据长度({len(data)})不匹配")
                            return data
                    elif cal_data.ndim == 2 and cal_data.shape[1] == 2:
                        # 二维数组，可能是频率响应数据（频率, 响应）
                        print(f"    应用频率响应校准，点数: {cal_data.shape[0]}")
                        # 这里需要实现频域校准
                        # 暂时返回原始数据，需要根据你的需求实现
                        return data
                    else:
                        print(f"    警告: 未知的校准数据格式，形状: {cal_data.shape}")
                        return data
                else:
                    print(f"    警告: 未知的校准数据类型: {type(cal_data)}")
                    return data

        return SimpleCalibrationProcessor(mat_file_dir, fnCAL_list)

    def update_ui_with_config(self):
        config_dict = self.config.get_config_dict()
        config_dialog = ConfigTableDialog(config_dict, parent=self)
        config_dialog.exec()

# =========================================== 调用视图模块 ===============================================================

    def open_view_interface(self):
        from 视图界面_main import viewWindow  # 实际使用时取消注释
        self.view_window = viewWindow()
        self.view_window.show()
        pass

    def open_view_interface_2(self):
        """打开工程界面"""
        from 工程文件配置界面_main import eigneerWindow

        self.view_window_2 = eigneerWindow()

        self.view_window_2.show()

# ============================================ 预处理模块 ================================================================

    def apply_detrend(self):
        """应用去趋势处理"""
        if not hasattr(self, 'df'):
            print("请先加载数据")
            return

        try:
            print("正在应用去趋势处理...")

            # 获取通道索引
            if hasattr(self.config, 'channel_indices'):
                inxCH = self.config.channel_indices

                if isinstance(inxCH, (list, np.ndarray)) and len(inxCH) > 0:
                    if inxCH[0] >= 1:
                        inxCH_py = [idx - 1 for idx in inxCH]
                    else:
                        inxCH_py = inxCH
                else:
                    inxCH_py = list(range(self.df.shape[1]))
            else:
                # 默认所有通道
                inxCH_py = list(range(self.df.shape[1]))

            # 创建或使用现有的data_interface
            if not hasattr(self, 'data_interface'):
                sample_rate = self.config.sample_rate if hasattr(self.config, 'sample_rate') else 1
                self.data_interface = DataInterface("", sample_rate)

            # 设置数据（保持为DataFrame）
            self.data_interface.TSData = self.df

            print(f"使用通道索引: {inxCH_py}")
            print(f"数据形状: {self.df.shape}")

            # 调用prewhitening方法进行去趋势处理
            # 注意：FirstDiff=False, RemoveMean=True
            detrended_data = self.data_interface.prewhitening(
                inxCH=inxCH_py,
                FirstDiff=False,  # 设置为False
                RemoveMean=True  # 去除均值
            )

            if detrended_data is None:
                print("去趋势处理失败")
                return

            # 删除最后一条记录
            # 这个操作在preWhitening函数之外进行
            original_length = len(detrended_data)
            if isinstance(detrended_data, pd.DataFrame):
                detrended_data = detrended_data.iloc[:-1, :]
            else:
                detrended_data = detrended_data[:-1, :]

            print(f"删除最后一条记录，长度从 {original_length} 变为 {len(detrended_data)}")

            # 更新时间索引（同样删除最后一个时间点）
            new_index = self.df.index[:-1]

            # 更新DataFrame
            self.df = pd.DataFrame(
                detrended_data,
                index=new_index,
                columns=self.df.columns
            ).astype('float32')

            # 更新filtered_data
            if hasattr(self.data_interface, 'filtered_data'):
                self.data_interface.filtered_data = self.df

            print(f"去趋势处理完成，新数据形状: {self.df.shape}")
            print(f"新时间范围: {self.df.index[0]} 到 {self.df.index[-1]}")

            # 显示处理后的统计信息
            print("\n去趋势后数据统计:")
            for col in self.df.columns:
                print(f"{col}: 均值={self.df[col].mean():.6f}, 标准差={self.df[col].std():.6f}")

            # 如果存在更新图表的方法，则调用
            if hasattr(self, 'update_plot'):
                self.update_plot()

            # 弹出数据展示弹窗
            data_dialog = DataTableDialog(self.df, parent=self)
            data_dialog.exec()

            print("去趋势处理流程完成")

        except Exception as e:
            print(f"去趋势处理出错: {e}")
            import traceback
            traceback.print_exc()

    def apply_calibration(self):
        """应用电磁场矫正"""
        if not hasattr(self, 'df'):
            print("请先加载数据")
            return

        try:
            # 创建对话框
            dialog = QtWidgets.QDialog()
            dialog.setWindowTitle("电磁场矫正参数设置")
            dialog.setMinimumSize(400, 400)

            layout = QtWidgets.QVBoxLayout(dialog)

            # 添加标签页
            tab_widget = QtWidgets.QTabWidget()

            # 自动校正标签页
            auto_tab = QtWidgets.QWidget()
            auto_layout = QtWidgets.QFormLayout(auto_tab)

            # 显示当前配置参数
            config_params = self.config.get_config_dict() if hasattr(self.config, 'get_config_dict') else {}

            auto_layout.addRow(QtWidgets.QLabel("<b>当前配置文件参数:</b>"))

            # 传感器灵敏度
            hsen_label = QtWidgets.QLabel(
                f"传感器灵敏度 (mV/nT): {config_params.get('传感器灵敏度 (mV/nT)', '未设置')}")
            auto_layout.addRow(hsen_label)

            # 电长度
            elen_label = QtWidgets.QLabel(f"电长度 (米): {config_params.get('电长度 (米)', '未设置')}")
            auto_layout.addRow(elen_label)

            # 通道增益
            gain_label = QtWidgets.QLabel(f"通道增益: {config_params.get('通道增益', '未设置')}")
            auto_layout.addRow(gain_label)

            # 通道索引
            inxch_label = QtWidgets.QLabel(f"通道索引: {config_params.get('通道索引', '未设置')}")
            auto_layout.addRow(inxch_label)

            # 通道数量
            nch_label = QtWidgets.QLabel(f"通道数量: {config_params.get('通道数量', '未设置')}")
            auto_layout.addRow(nch_label)

            # 添加使用自动校正的复选框
            use_auto_correction = QtWidgets.QCheckBox("使用配置文件参数进行自动校正")
            use_auto_correction.setChecked(True)
            auto_layout.addRow(use_auto_correction)

            tab_widget.addTab(auto_tab, "自动校正")

            # 手动校正标签页
            manual_tab = QtWidgets.QWidget()
            manual_layout = QtWidgets.QFormLayout(manual_tab)

            manual_layout.addRow(QtWidgets.QLabel("<b>电场通道校正:</b>"))

            # EX通道
            ex_scale_label = QtWidgets.QLabel("EX通道缩放因子:")
            ex_scale_spin = QtWidgets.QDoubleSpinBox()
            ex_scale_spin.setRange(0.001, 1000.0)
            ex_scale_spin.setValue(1.0)
            ex_scale_spin.setDecimals(6)
            ex_scale_spin.setSingleStep(0.1)
            manual_layout.addRow(ex_scale_label, ex_scale_spin)

            ex_offset_label = QtWidgets.QLabel("EX通道偏移量:")
            ex_offset_spin = QtWidgets.QDoubleSpinBox()
            ex_offset_spin.setRange(-1000.0, 1000.0)
            ex_offset_spin.setValue(0.0)
            ex_offset_spin.setDecimals(6)
            ex_offset_spin.setSingleStep(0.1)
            manual_layout.addRow(ex_offset_label, ex_offset_spin)

            # EY通道
            ey_scale_label = QtWidgets.QLabel("EY通道缩放因子:")
            ey_scale_spin = QtWidgets.QDoubleSpinBox()
            ey_scale_spin.setRange(0.001, 1000.0)
            ey_scale_spin.setValue(1.0)
            ey_scale_spin.setDecimals(6)
            ey_scale_spin.setSingleStep(0.1)
            manual_layout.addRow(ey_scale_label, ey_scale_spin)

            ey_offset_label = QtWidgets.QLabel("EY通道偏移量:")
            ey_offset_spin = QtWidgets.QDoubleSpinBox()
            ey_offset_spin.setRange(-1000.0, 1000.0)
            ey_offset_spin.setValue(0.0)
            ey_offset_spin.setDecimals(6)
            ey_offset_spin.setSingleStep(0.1)
            manual_layout.addRow(ey_offset_label, ey_offset_spin)

            manual_layout.addRow(QtWidgets.QLabel("<b>磁场通道校正:</b>"))

            # HX通道
            hx_scale_label = QtWidgets.QLabel("HX通道缩放因子:")
            hx_scale_spin = QtWidgets.QDoubleSpinBox()
            hx_scale_spin.setRange(0.001, 1000.0)
            hx_scale_spin.setValue(1.0)
            hx_scale_spin.setDecimals(6)
            hx_scale_spin.setSingleStep(0.1)
            manual_layout.addRow(hx_scale_label, hx_scale_spin)

            hx_offset_label = QtWidgets.QLabel("HX通道偏移量:")
            hx_offset_spin = QtWidgets.QDoubleSpinBox()
            hx_offset_spin.setRange(-1000.0, 1000.0)
            hx_offset_spin.setValue(0.0)
            hx_offset_spin.setDecimals(6)
            hx_offset_spin.setSingleStep(0.1)
            manual_layout.addRow(hx_offset_label, hx_offset_spin)

            # HY通道
            hy_scale_label = QtWidgets.QLabel("HY通道缩放因子:")
            hy_scale_spin = QtWidgets.QDoubleSpinBox()
            hy_scale_spin.setRange(0.001, 1000.0)
            hy_scale_spin.setValue(1.0)
            hy_scale_spin.setDecimals(6)
            hy_scale_spin.setSingleStep(0.1)
            manual_layout.addRow(hy_scale_label, hy_scale_spin)

            hy_offset_label = QtWidgets.QLabel("HY通道偏移量:")
            hy_offset_spin = QtWidgets.QDoubleSpinBox()
            hy_offset_spin.setRange(-1000.0, 1000.0)
            hy_offset_spin.setValue(0.0)
            hy_offset_spin.setDecimals(6)
            hy_offset_spin.setSingleStep(0.1)
            manual_layout.addRow(hy_offset_label, hy_offset_spin)

            # HZ通道
            hz_scale_label = QtWidgets.QLabel("HZ通道缩放因子:")
            hz_scale_spin = QtWidgets.QDoubleSpinBox()
            hz_scale_spin.setRange(0.001, 1000.0)
            hz_scale_spin.setValue(1.0)
            hz_scale_spin.setDecimals(6)
            hz_scale_spin.setSingleStep(0.1)
            manual_layout.addRow(hz_scale_label, hz_scale_spin)

            hz_offset_label = QtWidgets.QLabel("HZ通道偏移量:")
            hz_offset_spin = QtWidgets.QDoubleSpinBox()
            hz_offset_spin.setRange(-1000.0, 1000.0)
            hz_offset_spin.setValue(0.0)
            hz_offset_spin.setDecimals(6)
            hz_offset_spin.setSingleStep(0.1)
            manual_layout.addRow(hz_offset_label, hz_offset_spin)

            tab_widget.addTab(manual_tab, "手动校正")

            layout.addWidget(tab_widget)

            # 按钮
            button_box = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Ok |
                QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
            layout.addWidget(button_box)

            button_box.accepted.connect(dialog.accept)      # type: ignore
            button_box.rejected.connect(dialog.reject)      # type: ignore

            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                print("开始应用电磁场矫正...")

                # 创建数据副本
                df_corrected = self.df.copy()

                # 获取当前标签页
                current_tab_index = tab_widget.currentIndex()

                if current_tab_index == 0 and use_auto_correction.isChecked():
                    # 使用自动校正（基于配置文件参数）
                    print("使用配置文件参数进行自动校正")

                    # 获取配置参数
                    HSEN = config_params.get('传感器灵敏度 (mV/nT)', 1.0)
                    ELen = config_params.get('电长度 (米)', [1.0, 1.0])
                    Gain = config_params.get('通道增益', [1.0, 1.0, 1.0, 1.0, 1.0])
                    inxCH = config_params.get('通道索引', [1, 2, 3, 4, 5])
                    nch = config_params.get('通道数量', 5)

                    # 确保所有参数都是正确的类型
                    def ensure_float(value):
                        """确保值是浮点数"""
                        if isinstance(value, (list, np.ndarray)):
                            # 如果是列表，尝试取第一个元素
                            if len(value) > 0:
                                return float(value[0])
                            else:
                                return 1.0
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            return 1.0

                    def ensure_float_list(value, min_length):
                        """确保值是浮点数列表"""
                        if not isinstance(value, (list, np.ndarray)):
                            # 如果不是列表，创建列表
                            val = ensure_float(value)
                            return [val] * min_length

                        # 展平嵌套列表
                        flattened = []
                        for item in value:
                            if isinstance(item, (list, np.ndarray)):
                                # 递归展平
                                for subitem in item:
                                    flattened.append(ensure_float(subitem))
                            else:
                                flattened.append(ensure_float(item))

                        # 确保最小长度
                        while len(flattened) < min_length:
                            flattened.append(1.0)

                        return flattened[:min_length]  # 截断到指定长度

                    # 确保参数类型正确
                    HSEN = ensure_float(HSEN)
                    ELen = ensure_float_list(ELen, 2)  # ELen需要至少2个元素
                    Gain = ensure_float_list(Gain, max(5, nch))  # Gain需要足够的长度

                    # 转换为Python索引（0-based）
                    if isinstance(inxCH, (list, np.ndarray)):
                        # 确保inxCH是整数列表
                        inxCH_py = []
                        for idx in inxCH:
                            try:
                                inxCH_py.append(int(float(idx)) - 1)  # 先转浮点再转整数
                            except (ValueError, TypeError):
                                inxCH_py.append(0)
                    else:
                        inxCH_py = list(range(min(nch, 5)))

                    # 确保索引在有效范围内
                    inxCH_py = [max(0, min(idx, len(Gain) - 1)) for idx in inxCH_py]

                    print(f"使用校正参数: HSEN={HSEN}, ELen={ELen}, Gain={Gain}, inxCH_py={inxCH_py}")

                    # 执行自动校正
                    # ##### Electric field correcting #####
                    # EX通道校正
                    if nch >= 1 and 'EX' in df_corrected.columns:
                        ex_idx = int(inxCH_py[0]) if len(inxCH_py) > 0 else 0
                        # 注意：配置文件中的ELen单位已经是米，不需要再乘以1e-3
                        df_corrected['EX'] = df_corrected['EX'] / (ELen[0] * Gain[ex_idx])
                        print(f"EX通道校正: ELen={ELen[0]}, Gain={Gain[ex_idx]}")

                    # EY通道校正
                    if nch >= 2 and 'EY' in df_corrected.columns:
                        ey_idx = int(inxCH_py[1]) if len(inxCH_py) > 1 else 1
                        df_corrected['EY'] = df_corrected['EY'] / (ELen[1] * Gain[ey_idx])
                        print(f"EY通道校正: ELen={ELen[1]}, Gain={Gain[ey_idx]}")

                    # ##### Magnetic field correcting #####
                    if nch > 2 and 'HX' in df_corrected.columns:
                        # HX通道校正
                        hx_idx = int(inxCH_py[2]) if len(inxCH_py) > 2 else 2
                        df_corrected['HX'] = df_corrected['HX'] / (HSEN * Gain[hx_idx])
                        print(f"HX通道校正: HSEN={HSEN}, Gain={Gain[hx_idx]}")

                    if nch > 3 and 'HY' in df_corrected.columns:
                        # HY通道校正
                        hy_idx = int(inxCH_py[3]) if len(inxCH_py) > 3 else 3
                        df_corrected['HY'] = df_corrected['HY'] / (HSEN * Gain[hy_idx])
                        print(f"HY通道校正: HSEN={HSEN}, Gain={Gain[hy_idx]}")

                    if nch > 4 and 'HZ' in df_corrected.columns:
                        # HZ通道校正
                        hz_idx = int(inxCH_py[4]) if len(inxCH_py) > 4 else 4
                        df_corrected['HZ'] = df_corrected['HZ'] / (HSEN * Gain[hz_idx])
                        print(f"HZ通道校正: HSEN={HSEN}, Gain={Gain[hz_idx]}")

                    print("自动校正完成")

                else:
                    # 使用手动校正
                    print("使用手动校正参数")

                    # 应用电场通道校正
                    if 'EX' in df_corrected.columns:
                        ex_scale = ex_scale_spin.value()
                        ex_offset = ex_offset_spin.value()
                        df_corrected['EX'] = df_corrected['EX'] * ex_scale + ex_offset
                        print(f"EX通道校正: 缩放={ex_scale}, 偏移={ex_offset}")

                    if 'EY' in df_corrected.columns:
                        ey_scale = ey_scale_spin.value()
                        ey_offset = ey_offset_spin.value()
                        df_corrected['EY'] = df_corrected['EY'] * ey_scale + ey_offset
                        print(f"EY通道校正: 缩放={ey_scale}, 偏移={ey_offset}")

                    # 应用磁场通道校正
                    if 'HX' in df_corrected.columns:
                        hx_scale = hx_scale_spin.value()
                        hx_offset = hx_offset_spin.value()
                        df_corrected['HX'] = df_corrected['HX'] * hx_scale + hx_offset
                        print(f"HX通道校正: 缩放={hx_scale}, 偏移={hx_offset}")

                    if 'HY' in df_corrected.columns:
                        hy_scale = hy_scale_spin.value()
                        hy_offset = hy_offset_spin.value()
                        df_corrected['HY'] = df_corrected['HY'] * hy_scale + hy_offset
                        print(f"HY通道校正: 缩放={hy_scale}, 偏移={hy_offset}")

                    if 'HZ' in df_corrected.columns:
                        hz_scale = hz_scale_spin.value()
                        hz_offset = hz_offset_spin.value()
                        df_corrected['HZ'] = df_corrected['HZ'] * hz_scale + hz_offset
                        print(f"HZ通道校正: 缩放={hz_scale}, 偏移={hz_offset}")

                    print("手动校正完成")

                # 更新数据
                self.df = df_corrected

                # 显示校正后的统计信息
                print("\n校正后数据统计:")
                print(f"数据形状: {self.df.shape}")
                for col in self.df.columns:
                    print(f"{col}: 均值={self.df[col].mean():.6f}, 标准差={self.df[col].std():.6f}")

                # 更新图表
                if hasattr(self, 'update_plot'):
                    self.update_plot()

                print("电磁场矫正完成")

        except Exception as e:
            print(f"电磁场矫正出错: {e}")
            import traceback
            traceback.print_exc()

    def de_instrumental_response(self):
        """去仪器响应：通过标定文件进行数据校正"""
        if not hasattr(self, 'df') or self.df is None or self.df.empty:
            print("请先加载数据")
            return

        try:
            # ========== 1. 创建标定文件对话框 ==========
            dialog = QtWidgets.QDialog()
            dialog.setWindowTitle("仪器响应校正参数设置")
            dialog.setMinimumSize(500, 400)

            layout = QtWidgets.QVBoxLayout(dialog)

            # 创建标签页
            tab_widget = QtWidgets.QTabWidget()

            # 标签页1: 自动校正（使用配置文件中的标定文件）
            auto_tab = QtWidgets.QWidget()
            auto_layout = QtWidgets.QFormLayout(auto_tab)

            # 显示当前配置参数
            config_params = self.config.get_config_dict() if hasattr(self.config, 'get_config_dict') else {}

            auto_layout.addRow(QtWidgets.QLabel("<b>当前配置文件参数:</b>"))

            # 检查是否有标定文件
            has_calibration = False
            if hasattr(self.config, 'calibration_files'):
                fnCAL = self.config.calibration_files
                if isinstance(fnCAL, str):
                    fnCAL_list = fnCAL.split()
                else:
                    fnCAL_list = list(fnCAL)

                if len(fnCAL_list) > 0:
                    has_calibration = True
                    cal_label = QtWidgets.QLabel(f"标定文件: {', '.join(fnCAL_list[:3])}")
                    if len(fnCAL_list) > 3:
                        cal_label.setText(f"{cal_label.text()} 等共{len(fnCAL_list)}个文件")
                    auto_layout.addRow(cal_label)

            if not has_calibration:
                auto_layout.addRow(QtWidgets.QLabel("未在配置文件中找到标定文件"))

            # 使用自动校正的复选框
            use_auto_calibration = QtWidgets.QCheckBox("使用配置文件中的标定文件进行自动校正")
            use_auto_calibration.setChecked(has_calibration)
            use_auto_calibration.setEnabled(has_calibration)
            auto_layout.addRow(use_auto_calibration)

            tab_widget.addTab(auto_tab, "自动校正")

            # 标签页2: 手动选择标定文件
            manual_tab = QtWidgets.QWidget()
            manual_layout = QtWidgets.QVBoxLayout(manual_tab)

            # 添加选择标定文件的按钮
            select_cal_btn = QtWidgets.QPushButton("选择标定文件 (.cmt)")
            select_cal_btn.clicked.connect(lambda: self._select_calibration_files(dialog))      # type: ignore
            manual_layout.addWidget(select_cal_btn)

            # 显示已选择的标定文件
            self.selected_cal_files_label = QtWidgets.QLabel("未选择标定文件")
            self.selected_cal_files_label.setWordWrap(True)
            manual_layout.addWidget(self.selected_cal_files_label)

            manual_layout.addSpacing(20)

            # 添加通道映射配置
            channel_mapping_group = QtWidgets.QGroupBox("通道与标定文件映射")
            channel_mapping_layout = QtWidgets.QFormLayout(channel_mapping_group)

            # 通道映射说明
            channel_mapping_layout.addRow(QtWidgets.QLabel("<small>请指定每个通道使用的标定文件索引(0-based)</small>"))

            # 通道列表
            channels = ['HX', 'HY', 'HZ']  # 磁场通道需要校准
            self.channel_mapping_spins = {}

            for channel in channels:
                if channel in self.df.columns:
                    spin_box = QtWidgets.QSpinBox()
                    spin_box.setRange(0, 9)
                    spin_box.setValue(0)
                    spin_box.setToolTip(f"{channel}通道使用的标定文件索引")
                    channel_mapping_layout.addRow(f"{channel}通道:", spin_box)
                    self.channel_mapping_spins[channel] = spin_box

            manual_layout.addWidget(channel_mapping_group)

            tab_widget.addTab(manual_tab, "手动校正")

            # 标签页3: 校正参数设置
            param_tab = QtWidgets.QWidget()
            param_layout = QtWidgets.QFormLayout(param_tab)

            # 采样率设置
            sample_rate_label = QtWidgets.QLabel("采样率 (Hz):")
            sample_rate_edit = QtWidgets.QLineEdit(str(getattr(self.config, 'sample_rate', 1000)))
            sample_rate_edit.setValidator(QtGui.QDoubleValidator(0.1, 1000000.0, 2))
            param_layout.addRow(sample_rate_label, sample_rate_edit)

            # 校正方法选择
            method_label = QtWidgets.QLabel("校正方法:")
            method_combo = QtWidgets.QComboBox()
            method_combo.addItems(["FFT频域校正", "滤波器校正", "窄带单频点校正"])
            param_layout.addRow(method_label, method_combo)

            # 输出单位设置
            unit_label = QtWidgets.QLabel("输出单位:")
            unit_combo = QtWidgets.QComboBox()
            unit_combo.addItems(["物理单位 (nT)", "原始单位", "dB"])
            param_layout.addRow(unit_label, unit_combo)

            # 可视化选项
            visualize_cb = QtWidgets.QCheckBox("显示频率响应曲线")
            visualize_cb.setChecked(True)
            param_layout.addRow(visualize_cb)

            tab_widget.addTab(param_tab, "参数设置")

            layout.addWidget(tab_widget)

            # 按钮
            button_box = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Ok |
                QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
            layout.addWidget(button_box)

            button_box.accepted.connect(dialog.accept)      # type: ignore
            button_box.rejected.connect(dialog.reject)      # type: ignore

            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                print("开始应用仪器响应校正...")

                # 获取参数
                sample_rate = float(sample_rate_edit.text())
                method = method_combo.currentText()
                unit = unit_combo.currentText()
                show_visualization = visualize_cb.isChecked()

                # 获取当前标签页
                current_tab_index = tab_widget.currentIndex()

                if current_tab_index == 0 and use_auto_calibration.isChecked():
                    # 自动校正模式
                    print("使用配置文件中的标定文件进行自动校正")

                    # 获取标定文件
                    fnCAL_list = self._get_calibration_files_from_config()

                    if not fnCAL_list:
                        print("错误：配置文件中没有标定文件")
                        return

                    # 获取标定文件目录
                    calib_dir = self._get_calibration_directory()

                    # 创建校正器
                    calibrator = self._create_calibrator_from_files(fnCAL_list, calib_dir)

                    if calibrator is None:
                        print("错误：无法创建校正器")
                        return

                    # 应用校正
                    self._apply_calibration_with_calibrator(
                        calibrator, sample_rate, method, unit, show_visualization
                    )

                else:
                    # 手动校正模式
                    print("使用手动选择的标定文件进行校正")

                    # 获取手动选择的标定文件
                    manual_cal_files = getattr(self, 'manual_cal_files', [])

                    if not manual_cal_files:
                        print("错误：未选择标定文件")
                        return

                    # 获取通道映射
                    channel_mapping = {}
                    for channel, spin_box in self.channel_mapping_spins.items():
                        if channel in self.df.columns:
                            file_idx = spin_box.value()
                            if file_idx < len(manual_cal_files):
                                channel_mapping[channel] = file_idx

                    # 创建校正器
                    calibrator = self._create_calibrator_from_files(
                        manual_cal_files,
                        "",  # 文件已包含完整路径
                        channel_mapping
                    )

                    if calibrator is None:
                        print("错误：无法创建校正器")
                        return

                    # 应用校正
                    self._apply_calibration_with_calibrator(
                        calibrator, sample_rate, method, unit, show_visualization
                    )

                print("仪器响应校正完成")

        except Exception as e:
            print(f"仪器响应校正出错: {e}")
            import traceback
            traceback.print_exc()

    def _select_calibration_files(self, dialog):
        """选择标定文件"""
        file_dialog = QtWidgets.QFileDialog(dialog)
        file_dialog.setWindowTitle("选择标定文件 (.cmt)")
        file_dialog.setNameFilter("标定文件 (*.cmt);;所有文件 (*)")
        file_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFiles)

        if file_dialog.exec():
            files = file_dialog.selectedFiles()
            if files:
                self.manual_cal_files = files
                file_names = [os.path.basename(f) for f in files]
                self.selected_cal_files_label.setText(f"已选择 {len(files)} 个文件:\n" + "\n".join(file_names[:5]))
                if len(files) > 5:
                    self.selected_cal_files_label.setText(
                        self.selected_cal_files_label.text() + f"\n...等共{len(files)}个文件")

    def _get_calibration_files_from_config(self):
        """从配置文件中获取标定文件列表"""
        if not hasattr(self.config, 'calibration_files'):
            return []

        fnCAL = self.config.calibration_files
        if isinstance(fnCAL, str):
            return fnCAL.split()
        else:
            return list(fnCAL)

    def _get_calibration_directory(self):
        """获取标定文件目录"""
        # 尝试从配置文件获取数据目录
        if hasattr(self.config, 'data_directory'):
            return self.config.data_directory
        # 尝试从当前数据文件路径获取
        elif hasattr(self, 'data_interface') and hasattr(self.data_interface, 'file_path'):
            return os.path.dirname(self.data_interface.file_path)
        else:
            return "."

    def _create_calibrator_from_files(self, cal_files, base_dir="", channel_mapping=None):
        """从标定文件创建校正器"""
        try:
            # 检查标定文件是否存在
            missing_files = []
            valid_cal_files = []

            for cal_file in cal_files:
                # 构建完整路径
                if base_dir and not os.path.isabs(cal_file):
                    full_path = os.path.join(base_dir, cal_file)
                else:
                    full_path = cal_file

                if os.path.exists(full_path):
                    valid_cal_files.append(full_path)
                else:
                    missing_files.append(cal_file)

            if missing_files:
                print(f"警告: 以下标定文件不存在: {missing_files}")

                reply = QtWidgets.QMessageBox.question(
                    self,
                    "警告",
                    f"找不到以下标定文件:\n{', '.join(missing_files)}\n是否继续使用找到的标定文件？",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No
                )

                if reply == QtWidgets.QMessageBox.StandardButton.No:
                    return None

            if not valid_cal_files:
                print("错误: 没有找到有效的标定文件")
                return None

            print(f"找到 {len(valid_cal_files)} 个有效的标定文件")

            # 创建校正器
            calibrator = LEMi417Calibrator(valid_cal_files, channel_mapping)

            # 验证校正器
            if calibrator.validate():
                print("校正器创建成功")
                return calibrator
            else:
                print("警告: 校正器验证失败")
                return calibrator

        except Exception as e:
            print(f"创建校正器时出错: {e}")
            return None

    def _apply_calibration_with_calibrator(self, calibrator, sample_rate, method, unit, show_visualization):
        """使用校正器应用校正"""
        global corrected_data
        try:
            # 保存校准器到类属性，以便后续使用
            self.calibrator = calibrator

            # 创建数据副本
            df_corrected = self.df.copy()

            # 获取需要校正的通道
            channels_to_calibrate = calibrator.get_channels_to_calibrate()

            print(f"需要校正的通道: {channels_to_calibrate}")

            # 对每个通道应用校正
            for channel in channels_to_calibrate:
                if channel not in df_corrected.columns:
                    print(f"警告: 通道 {channel} 不在数据中，跳过")
                    continue

                print(f"校正通道 {channel}...")

                # 获取通道数据
                channel_data = df_corrected[channel].values

                # 根据方法选择校正方式
                if method == "FFT频域校正":
                    corrected_data = calibrator.calibrate_time_series(
                        channel_data, sample_rate, method='fft'
                    )
                elif method == "滤波器校正":
                    corrected_data = calibrator.calibrate_time_series(
                        channel_data, sample_rate, method='filter'
                    )
                elif method == "窄带单频点校正":
                    # 弹出对话框让用户输入中心频率
                    center_freq, ok = QtWidgets.QInputDialog.getDouble(
                        self,
                        "中心频率",
                        f"请输入 {channel} 通道的中心频率 (Hz):",
                        1.0,  # 默认值
                        0.001,  # 最小值
                        10000.0,  # 最大值
                        3  # 小数位数
                    )

                    if ok:
                        corrected_data = calibrator.calibrate_narrowband(
                            channel_data, center_freq
                        )
                    else:
                        print(f"用户取消，跳过 {channel} 通道")
                        continue

                # 单位转换
                if unit == "物理单位 (nT)":
                    # 转换为nT单位（假设原始数据是电压，需要转换为磁场强度）
                    # 这里需要根据具体的传感器灵敏度进行转换
                    # 暂时假设校正器已经处理了单位转换
                    pass
                elif unit == "dB":
                    # 转换为dB单位
                    corrected_data = 20 * np.log10(np.abs(corrected_data) + 1e-12)

                # 更新数据
                df_corrected[channel] = corrected_data

                print(f"通道 {channel} 校正完成")

            # 更新数据
            self.df = df_corrected

            # 更新数据接口
            if hasattr(self, 'data_interface'):
                self.data_interface.TSData = self.df
                if hasattr(self.data_interface, 'filtered_data'):
                    self.data_interface.filtered_data = self.df

            # 显示校正后的统计信息
            print("\n校正后数据统计:")
            for channel in channels_to_calibrate:
                if channel in self.df.columns:
                    print(f"{channel}: 均值={self.df[channel].mean():.6f}, 标准差={self.df[channel].std():.6f}")

            # 可视化
            if show_visualization:
                calibrator.plot_calibration_response(self.figure, self.canvas)

            # 弹出数据展示弹窗
            data_dialog = DataTableDialog(self.df, parent=self)
            data_dialog.exec()

            # 更新图表
            if hasattr(self, 'update_plot'):
                self.update_plot()

            print(f"仪器响应校正完成，校正了 {len(channels_to_calibrate)} 个通道")

        except Exception as e:
            print(f"应用校正时出错: {e}")
            import traceback
            traceback.print_exc()

# =============================================== 滤波器 ================================================================
    def finite_impulse_response_filter(self):
        """FIR滤波（多线程）"""
        if not hasattr(self, 'df'):
            print("请先加载数据")
            return

        try:
            # 弹出对话框获取滤波参数
            dialog = QtWidgets.QDialog()
            dialog.setWindowTitle("FIR滤波参数")
            dialog.setMinimumSize(300, 150)

            layout = QtWidgets.QFormLayout(dialog)

            filter_type_label = QtWidgets.QLabel("滤波器类型:")
            filter_type_combo = QtWidgets.QComboBox()
            filter_type_combo.addItems(["低通", "高通", "带通", "带阻"])

            cutoff_label = QtWidgets.QLabel("截止频率 (Hz):")
            cutoff_edit = QtWidgets.QLineEdit("0.1, 10.0")

            num_taps_label = QtWidgets.QLabel("滤波器抽头数:")
            num_taps_spin = QtWidgets.QSpinBox()
            num_taps_spin.setRange(1, 1001)
            num_taps_spin.setValue(65)
            num_taps_spin.setSingleStep(2)

            layout.addRow(filter_type_label, filter_type_combo)
            layout.addRow(cutoff_label, cutoff_edit)
            layout.addRow(num_taps_label, num_taps_spin)

            button_box = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Ok |
                QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
            layout.addRow(button_box)

            button_box.accepted.connect(dialog.accept)  # type: ignore
            button_box.rejected.connect(dialog.reject)  # type: ignore

            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                filter_type = filter_type_combo.currentText()
                cutoff_text = cutoff_edit.text()
                num_taps = num_taps_spin.value()

                try:
                    cutoff = [float(x.strip()) for x in cutoff_text.split(',')]
                except ValueError:
                    print("无效的截止频率值")
                    return

                if num_taps % 2 == 0:
                    num_taps += 1
                    print(f"警告: FIR滤波器抽头数必须为奇数，已调整为 {num_taps}")

                # 根据滤波器类型设置参数
                btype = None
                if filter_type == "低通":
                    btype = 'lowpass'
                    if len(cutoff) != 1:
                        print("低通滤波器需要一个截止频率")
                        return
                elif filter_type == "高通":
                    btype = 'highpass'
                    if len(cutoff) != 1:
                        print("高通滤波器需要一个截止频率")
                        return
                elif filter_type == "带通":
                    btype = 'bandpass'
                    if len(cutoff) != 2:
                        print("带通滤波器需要两个截止频率")
                        return
                elif filter_type == "带阻":
                    btype = 'bandstop'
                    if len(cutoff) != 2:
                        print("带阻滤波器需要两个截止频率")
                        return

                # 设置数据接口
                self.data_interface.TSData = self.df
                self.data_interface.sample_rate = self.config.sample_rate

                # 准备参数
                params = {
                    'cutoff': cutoff,
                    'num_taps': num_taps,
                    'btype': btype
                }

                # 创建工作线程
                worker = FilterWorker(self.data_interface, "FIR", params, self)
                worker.finished.connect(self._on_filter_finished)   # type: ignore
                self._start_worker(worker)

        except Exception as e:
            print(f"FIR滤波处理出错: {e}")

    def infinite_impulse_response_filter(self):
        """IIR滤波（多线程）"""
        if not hasattr(self, 'df'):
            print("请先加载数据")
            return

        try:
            dialog = QtWidgets.QDialog()
            dialog.setWindowTitle("IIR滤波参数")
            dialog.setMinimumSize(300, 200)

            layout = QtWidgets.QFormLayout(dialog)

            filter_type_label = QtWidgets.QLabel("滤波器类型:")
            filter_type_combo = QtWidgets.QComboBox()
            filter_type_combo.addItems(["巴特沃斯", "切比雪夫I型", "切比雪夫II型", "椭圆"])

            response_type_label = QtWidgets.QLabel("响应类型:")
            response_type_combo = QtWidgets.QComboBox()
            response_type_combo.addItems(["低通", "高通", "带通", "带阻"])

            cutoff_label = QtWidgets.QLabel("截止频率 (Hz):")
            cutoff_edit = QtWidgets.QLineEdit("0.1, 10.0")

            order_label = QtWidgets.QLabel("滤波器阶数:")
            order_spin = QtWidgets.QSpinBox()
            order_spin.setRange(1, 10)
            order_spin.setValue(4)

            ripple_label = QtWidgets.QLabel("波纹 (dB):")
            ripple_spin = QtWidgets.QDoubleSpinBox()
            ripple_spin.setRange(0.1, 20.0)
            ripple_spin.setValue(3.0)
            ripple_spin.setSingleStep(0.1)

            layout.addRow(filter_type_label, filter_type_combo)
            layout.addRow(response_type_label, response_type_combo)
            layout.addRow(cutoff_label, cutoff_edit)
            layout.addRow(order_label, order_spin)
            layout.addRow(ripple_label, ripple_spin)

            button_box = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Ok |
                QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
            layout.addRow(button_box)

            button_box.accepted.connect(dialog.accept)  # type: ignore
            button_box.rejected.connect(dialog.reject)  # type: ignore

            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                filter_type = filter_type_combo.currentText()
                response_type = response_type_combo.currentText()
                cutoff_text = cutoff_edit.text()
                order = order_spin.value()
                ripple = ripple_spin.value()

                try:
                    cutoff = [float(x.strip()) for x in cutoff_text.split(',')]
                except ValueError:
                    print("无效的截止频率值")
                    return

                # 映射参数
                iir_type = None
                if filter_type == "巴特沃斯":
                    iir_type = 'butter'
                elif filter_type == "切比雪夫I型":
                    iir_type = 'cheby1'
                elif filter_type == "切比雪夫II型":
                    iir_type = 'cheby2'
                elif filter_type == "椭圆":
                    iir_type = 'ellip'

                btype = None
                if response_type == "低通":
                    btype = 'low'
                    if len(cutoff) != 1:
                        print("低通滤波器需要一个截止频率")
                        return
                elif response_type == "高通":
                    btype = 'high'
                    if len(cutoff) != 1:
                        print("高通滤波器需要一个截止频率")
                        return
                elif response_type == "带通":
                    btype = 'bandpass'
                    if len(cutoff) != 2:
                        print("带通滤波器需要两个截止频率")
                        return
                elif response_type == "带阻":
                    btype = 'bandstop'
                    if len(cutoff) != 2:
                        print("带阻滤波器需要两个截止频率")
                        return

                # 设置数据接口
                self.data_interface.TSData = self.df
                self.data_interface.sample_rate = self.config.sample_rate

                # 准备参数
                params = {
                    'cutoff': cutoff,
                    'order': order,
                    'iir_type': iir_type,
                    'btype': btype,
                    'ripple': ripple
                }

                # 创建工作线程
                worker = FilterWorker(self.data_interface, "IIR", params, self)
                worker.finished.connect(self._on_filter_finished)   # type: ignore
                self._start_worker(worker)

        except Exception as e:
            print(f"IIR滤波处理出错: {e}")

    def notch_filter_butterworth(self):
        """快速带阻滤波（多线程）"""
        if not hasattr(self, 'df') or self.df is None:
            print("请先加载数据")
            return

        # 设置数据接口
        self.data_interface.TSData = self.df.copy()
        self.data_interface.sample_rate = self.config.sample_rate

        # 定义谐波频带
        harmonics = [
            [49.0, 51.0],   # 50Hz
            [99.0, 101.0],   # 100Hz
            [149.0, 151.0],  # 150Hz
            [199.0, 201.0],  # 200Hz
            [249.0, 251.0],  # 250Hz
            [299.0, 301.0],  # 300Hz
            [349.0, 351.0],  # 350Hz
            [399.0, 401.0],  # 400Hz
            [449.0, 451.0]   # 450Hz
        ]

        # 准备参数
        params = {'harmonics': harmonics}

        # 创建工作线程
        worker = FilterWorker(self.data_interface, "Notch", params, self)
        worker.finished.connect(self._on_filter_finished)   # type: ignore
        self._start_worker(worker)

    def adaptive_filter(self):
        """自适应滤波（多线程）"""
        if not hasattr(self, 'df'):
            print("请先加载数据")
            return

        try:
            dialog = QtWidgets.QDialog()
            dialog.setWindowTitle("自适应滤波参数")
            dialog.setMinimumSize(300, 150)

            layout = QtWidgets.QFormLayout(dialog)

            algorithm_label = QtWidgets.QLabel("算法:")
            algorithm_combo = QtWidgets.QComboBox()
            algorithm_combo.addItems(["LMS", "NLMS", "RLS"])

            filter_length_label = QtWidgets.QLabel("滤波器长度:")
            filter_length_spin = QtWidgets.QSpinBox()
            filter_length_spin.setRange(2, 1000)
            filter_length_spin.setValue(64)

            step_size_label = QtWidgets.QLabel("步长:")
            step_size_edit = QtWidgets.QLineEdit("0.01")

            layout.addRow(algorithm_label, algorithm_combo)
            layout.addRow(filter_length_label, filter_length_spin)
            layout.addRow(step_size_label, step_size_edit)

            button_box = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Ok |
                QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
            layout.addRow(button_box)

            button_box.accepted.connect(dialog.accept)  # type: ignore
            button_box.rejected.connect(dialog.reject)  # type: ignore

            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                algorithm = algorithm_combo.currentText()
                filter_length = filter_length_spin.value()

                try:
                    step_size = float(step_size_edit.text())
                except ValueError:
                    print("无效的步长值")
                    return

                # 设置数据接口
                self.data_interface.TSData = self.df
                self.data_interface.sample_rate = self.config.sample_rate

                # 准备参数
                params = {
                    'algorithm': algorithm,
                    'filter_length': filter_length,
                    'step_size': step_size
                }

                # 创建工作线程
                worker = FilterWorker(self.data_interface, "Adaptive", params, self)
                worker.finished.connect(self._on_filter_finished)   # type: ignore
                self._start_worker(worker)

        except Exception as e:
            print(f"自适应滤波处理出错: {e}")

    def _on_filter_finished(self, result):
        """滤波完成处理"""
        filtered_data = result['filtered_data']
        filter_type = result['filter_type']

        # 更新数据
        self.data_interface.filtered_data = filtered_data
        self.raw_time_data = self.df
        self.df = filtered_data

        print(f"{filter_type}滤波处理完成")

        # 更新图表
        self.update_plot()

        # 弹出数据展示弹窗
        data_dialog = DataTableDialog(self.df, parent=self)
        data_dialog.exec()

    def debug_cascade_filtering(self):
        """级联滤波调试（多线程）"""
        print("===== 步骤1：选择测试文件 =====")
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        file_path = filedialog.askopenfilename(
            title="选择Excel测试数据文件",
            filetypes=[("Excel文件", "*.xlsx *.xls"), ("所有文件", "*.*")],
            initialdir=r"E:\gui\软件代码"
        )

        if not file_path:
            print("❌ 用户取消了文件选择")
            return

        print(f"✅ 已选择文件：{file_path}")

        # 创建工作线程
        worker = CascadeFilterWorker(file_path, 300, self)
        worker.finished.connect(self._on_cascade_filter_finished)   # type: ignore
        self._start_worker(worker)

    def _on_cascade_filter_finished(self, cf):
        """级联滤波完成处理"""
        print("✅ 级联滤波调试通过！")


# ============================================= 计算区域内置调用函数 =======================================================
    def display_pca_results(self, pca_results):
        """显示PCA分析结果，优化表格展示"""
        # 创建结果展示对话框
        result_dialog = QtWidgets.QDialog(self)
        result_dialog.setWindowTitle("PCA分析结果")
        result_dialog.setMinimumSize(800, 600)

        # 使用标签页展示不同结果
        tab_widget = QtWidgets.QTabWidget()
        layout = QtWidgets.QVBoxLayout(result_dialog)
        layout.addWidget(tab_widget)

        # 1. 主成分载荷矩阵
        components_df = pd.DataFrame(
            pca_results['components'],
            columns=self.df.columns,
            index=[f'主成分{i + 1}' for i in range(pca_results['components'].shape[0])]
        )
        components_table = QtWidgets.QTableView()
        components_model = PandasTableModel(components_df)
        components_table.setModel(components_model)
        components_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tab_widget.addTab(components_table, "主成分载荷矩阵")

        # 2. 解释方差比例
        explained_df = pd.DataFrame({
            '解释方差比例': pca_results['explained_variance_ratio'],
            '累积解释方差比例': np.cumsum(pca_results['explained_variance_ratio'])
        }, index=[f'主成分{i + 1}' for i in range(len(pca_results['explained_variance_ratio']))])
        explained_table = QtWidgets.QTableView()
        explained_model = PandasTableModel(explained_df)
        explained_table.setModel(explained_model)
        explained_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tab_widget.addTab(explained_table, "解释方差比例")

        # 3. 转换后的数据（前100行）
        transformed_data = pca_results['transformed_data']
        transformed_df = pd.DataFrame(
            transformed_data[:100],  # 只显示前100行，避免数据量过大
            columns=[f'主成分{i + 1}' for i in range(transformed_data.shape[1])]
        )
        transformed_table = QtWidgets.QTableView()
        transformed_model = PandasTableModel(transformed_df)
        transformed_table.setModel(transformed_model)
        transformed_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        tab_widget.addTab(transformed_table, "PCA转换后的数据（前100行）")

        result_dialog.exec()

    @staticmethod
    def _get_channel_names(max_channels=None):
        """获取标准通道名称列表"""
        base_names = ['EX', 'EY', 'HX', 'HY', 'HZ']
        if max_channels is None:
            return base_names
        return [base_names[i] if i < len(base_names) else f'通道{i + 1}'
                for i in range(max_channels)]

    def _select_channel_dialog(self, channel_count):
        """
        显示通道选择对话框

        返回:
            int: 选中的通道索引(0-based)，取消则返回None
        """
        channel_names = self._get_channel_names(channel_count)
        channel_options = [f"{i + 1}. {name}" for i, name in enumerate(channel_names)]

        dialog = QDialog()
        dialog.setWindowTitle("选择通道")
        dialog.setMinimumSize(250, 200)

        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("请选择要分析的通道:"))

        list_widget = QListWidget()
        list_widget.addItems(channel_options)
        list_widget.setCurrentRow(0)
        layout.addWidget(list_widget)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        # 忽略类型检查警告
        button_box.accepted.connect(dialog.accept)  # type: ignore
        button_box.rejected.connect(dialog.reject)  # type: ignore
        layout.addWidget(button_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_text = list_widget.currentItem().text()
            try:
                return int(selected_text.split('.')[0]) - 1
            except (ValueError, IndexError):
                print(f"无效的通道选择: {selected_text}", "warning")
        return None

    def _display_dataframe(self, df, title="数据表格", log_prefix=""):
        if df is None or df.empty:
            print(f"{log_prefix}数据为空，无法显示{title}", "error")
            return False

        try:
            df_dialog = DataFrameDialog(df, title=title, parent=self)
            df_dialog.exec()

            print(f"{log_prefix}{title}已显示，形状: {df.shape}")
            return True
        except Exception as e:
            print(f"{log_prefix}显示{title}时出错: {str(e)}", "critical")
            return False

    def _validate_data_interface(self):
        """验证数据接口是否存在且包含有效数据"""
        if not hasattr(self, 'data_interface') or self.data_interface.TSData is None:
            print("没有可用数据", "warning")
            return False
        return True

# ===========================================计算类 ====================================================================

    def pca_filter(self):
        """主成分分析:用于数据降维，提取主要特征"""
        if not hasattr(self, 'df') or self.df is None or self.df.empty:
            print("请先加载数据")
            return

        try:
            # 弹出对话框获取PCA参数
            dialog = QtWidgets.QDialog()
            dialog.setWindowTitle("主成分分析参数")
            dialog.setMinimumSize(350, 200)  # 增大对话框尺寸
            dialog.setModal(True)  # 设置为模态对话框

            layout = QtWidgets.QVBoxLayout(dialog)  # 使用垂直布局更灵活

            # 参数设置分组
            params_group = QtWidgets.QGroupBox("分析参数")
            params_layout = QtWidgets.QFormLayout()
            params_group.setLayout(params_layout)

            # 主成分数量选择
            n_components_label = QtWidgets.QLabel("主成分数量:")
            n_components_spin = QtWidgets.QSpinBox()
            max_components = min(10, self.df.shape[1])  # 最多10个或数据列数
            n_components_spin.setRange(1, max_components)
            n_components_spin.setValue(min(3, max_components))  # 默认3个
            params_layout.addRow(n_components_label, n_components_spin)

            # 是否进行数据标准化
            scale_label = QtWidgets.QLabel("数据标准化:")
            scale_check = QtWidgets.QCheckBox()
            scale_check.setChecked(True)  # 默认标准化
            scale_check.setToolTip("对数据进行均值为0、方差为1的标准化处理")
            params_layout.addRow(scale_label, scale_check)

            # 高级选项：是否显示载荷热力图
            plot_heatmap_label = QtWidgets.QLabel("显示载荷热力图:")
            plot_heatmap_check = QtWidgets.QCheckBox()
            plot_heatmap_check.setChecked(True)
            params_layout.addRow(plot_heatmap_label, plot_heatmap_check)

            layout.addWidget(params_group)

            # 按钮区域
            button_box = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.StandardButton.Ok |
                QtWidgets.QDialogButtonBox.StandardButton.Cancel
            )
            layout.addWidget(button_box)

            button_box.accepted.connect(dialog.accept)  # type: ignore
            button_box.rejected.connect(dialog.reject)  # type: ignore

            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                n_components = n_components_spin.value()
                scale_data = scale_check.isChecked()
                plot_heatmap = plot_heatmap_check.isChecked()

                print(f"执行主成分分析: 主成分数量={n_components}, 标准化={scale_data}")

                # 调用数据接口进行PCA分析
                self.data_interface.TSData = self.df
                original_ts_data = self.data_interface.TSData
                self.data_interface.TSData = self.data_interface.filtered_data
                pca_results = self.data_interface.perform_pca(
                    n_components=n_components,
                    scale=scale_data
                )

                if pca_results:
                    # 显示结果
                    self.display_pca_results(pca_results)
                    # 可视化结果
                    self.plot_pca_results(pca_results, plot_heatmap)
                    # 恢复为原始数据
                    self.data_interface.TSData = original_ts_data
        except Exception as e:
            print(f"主成分分析出错: {e}")

    def calculate_psd_pdf(self):
        """计算功率谱概率密度（考虑多个记录段）"""
        global freq, original_ts_data
        if not self._validate_data_interface():
            return False

        try:
            print("开始计算功率谱概率密度（多个记录段）...")

            # 获取原始数据
            original_ts_data = self.data_interface.TSData
            filtered_data = self.data_interface.filtered_data

            # 设置数据为滤波后的数据
            self.data_interface.TSData = filtered_data

            # 获取采样率
            sampling_rate = getattr(self.data_interface, 'sampling_rate', 1000)

            # 设置参数
            segment_length = self.config.fft_window_length  # 记录段长度（点数）
            step = segment_length // 2  # 步长（50%重叠）

            self.psd_pdf_results = {}

            # 对每个通道单独处理
            for channel in filtered_data.columns:
                # 获取通道数据
                channel_data = filtered_data[channel].values

                # 检查数据长度
                if len(channel_data) < segment_length:
                    print(f"通道 {channel} 数据长度不足，跳过")
                    continue

                # 将数据分成多个记录段（滑动窗口）
                segments = []
                n_samples = len(channel_data)

                for start in range(0, n_samples - segment_length + 1, step):
                    segment = channel_data[start:start + segment_length]
                    segments.append(segment)

                n_segments = len(segments)

                if n_segments == 0:
                    print(f"通道 {channel} 无法分成记录段，跳过")
                    continue

                # 对每个记录段计算PSD
                all_psd_segments = []

                for i, segment in enumerate(segments):
                    # 使用Welch方法计算当前记录段的PSD
                    # 这里简化处理，使用FFT计算功率谱
                    from scipy.signal import welch

                    freq, psd = welch(
                        segment,
                        fs=sampling_rate,
                        nperseg=segment_length,
                        noverlap=segment_length // 2,
                        scaling='density'
                    )

                    # 转换为dB
                    psd_dB = 10 * np.log10(np.maximum(psd, 1e-20))
                    all_psd_segments.append(psd_dB)

                # 转换为数组：形状为 (n_segments, n_freq)
                all_psd_segments = np.array(all_psd_segments)
                n_freq = len(freq)

                # 定义PSD的dB范围：-200到-50 dB，窗长1 dB
                psd_bins = np.arange(-200, -49, 1)  # -200到-50，步长1 dB
                n_bins = len(psd_bins) - 1

                # 初始化概率密度矩阵：频率 × PSD区间
                pdf_matrix = np.zeros((n_freq, n_bins))

                # 对每个频率点，统计PSD值落在各个区间的记录段个数
                for i in range(n_freq):
                    # 获取当前频率点所有记录段的PSD值
                    freq_psd_values = all_psd_segments[:, i]

                    # 使用直方图统计落在各个区间的个数
                    counts, _ = np.histogram(freq_psd_values, bins=psd_bins)

                    # 计算概率密度：个数 / 总记录段数
                    if n_segments > 0:
                        pdf_matrix[i, :] = counts / n_segments

                # 计算PSD区间中心值
                psd_bin_centers = (psd_bins[:-1] + psd_bins[1:]) / 2

                # 计算平均PSD（所有记录段的平均）
                mean_psd_dB = np.mean(all_psd_segments, axis=0)

                # 存储结果
                self.psd_pdf_results[channel] = {
                    'frequencies': freq,
                    'psd_dB': mean_psd_dB,  # 平均PSD
                    'psd_bins': psd_bins,
                    'psd_bin_centers': psd_bin_centers,
                    'pdf_matrix': pdf_matrix,
                    'n_segments': n_segments,
                    'mean_psd_dB': np.mean(mean_psd_dB),
                    'std_psd_dB': np.std(mean_psd_dB),
                    'all_psd_segments': all_psd_segments,  # 可选：保存所有记录段的PSD

                    # 新增：用于绘图的简化数据
                    'bin_centers': psd_bin_centers,  # 复用psd_bin_centers
                    'pdf': np.mean(pdf_matrix, axis=0) if pdf_matrix.ndim > 1 else pdf_matrix,  # 平均PDF
                    'bin_edges': psd_bins  # 复用psd_bins作为bin_edges
                }

                print(f"通道 {channel}: {n_segments} 个记录段，频率点数: {n_freq}")

            # 恢复原始数据
            self.data_interface.TSData = original_ts_data

            if not self.psd_pdf_results:
                print("所有通道的PSD概率密度计算失败", "error")
                return False

            print(f"功率谱概率密度计算完成，有效通道: {list(self.psd_pdf_results.keys())}")
            return True

        except Exception as e:
            print(f"计算功率谱概率密度时出错: {str(e)}", "critical")
            import traceback
            traceback.print_exc()

            # 确保恢复原始数据
            if 'original_ts_data' in locals():
                self.data_interface.TSData = original_ts_data
            return False

    def calculate_psd(self):
        """计算PSD（多线程）"""
        if not hasattr(self, 'data_interface') or self.data_interface.TSData is None:
            print("没有可用数据")
            return

        # 创建工作线程
        worker = PSDWorker(self.data_interface, self.config.fft_window_length, True, self)
        worker.finished.connect(self._on_psd_finished)  # type: ignore
        self._start_worker(worker)

    def _on_psd_finished(self, psd_df):
        """PSD计算完成处理"""
        self.psd_results = psd_df
        self._display_dataframe(psd_df, "PSD分析结果", "PSD计算: ")

    def calculate_allan_variance(self):
        """计算Allan方差（多线程）"""
        if not hasattr(self, 'data_interface') or self.data_interface.TSData is None:
            print("没有可用数据")
            return

        # 创建工作线程
        worker = AllanWorker(self.data_interface, True, self)
        worker.finished.connect(self._on_allan_finished)    # type: ignore
        self._start_worker(worker)

    def _on_allan_finished(self, allan_df):
        """Allan方差计算完成处理"""
        self.allan_results = allan_df
        self._display_dataframe(allan_df, "Allan方差分析结果", "Allan方差计算: ")

    def calculate_wavelet_transform(self):
        """计算小波变换（多线程）"""
        if not hasattr(self, 'data_interface') or self.data_interface.TSData is None:
            print("没有可用数据")
            return

        data = self.data_interface.TSData
        channel_count = data.shape[1] if data.ndim > 1 else 0

        if channel_count == 0:
            print("数据列数为0，无法进行小波变换")
            return

        # 获取用户选择的通道
        channel_idx = self._select_channel_dialog(channel_count)
        if channel_idx is None:
            print("用户取消了操作")
            return

        channel_name = self._get_channel_names(channel_count)[channel_idx]
        print(f"开始计算通道 {channel_name} 的小波变换...")

        # 创建工作线程
        worker = WaveletWorker(self.data_interface, channel_idx, True, self)
        worker.finished.connect(lambda result: self._on_wavelet_finished(result, channel_name)) # type: ignore
        self._start_worker(worker)

    def _on_wavelet_finished(self, result, channel_name):
        """小波变换完成处理"""
        self.wavelet_results = result

        # 创建DataFrame
        cwtmatr = result['cwtmatr']
        frequencies = result['frequencies']
        time = result['time']

        # 限制显示的时间点
        max_time_points = 100
        display_time = time[:max_time_points]
        display_cwt = np.abs(cwtmatr[:, :max_time_points])

        wavelet_df = pd.DataFrame(
            data=display_cwt,
            index=frequencies,
            columns=[f"t={t:.2f}s" for t in display_time]
        )
        wavelet_df.index.name = "频率 (Hz)"

        title = f"通道 {channel_name} 的小波变换结果"
        self._display_dataframe(wavelet_df, title, "小波变换计算: ")

# ============================================== 可视化模块 ===========================================================

    def export_figure(self):
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出图像", "", "PDF (*.pdf);;SVG (*.svg);;PNG (*.png);;所有文件 (*)"
        )
        if not file_path:
            return
        dpi = 300  # 可让用户选择
        format = file_path.split('.')[-1]
        self.figure.savefig(file_path, dpi=dpi, format=format, bbox_inches='tight')

    def update_plot(self):
        """更新当前数据的绘图显示"""
        # ========== 画布判空保护 ==========
        if self.figure is None:
            print("错误：画布未初始化，无法绘制数据")
            return
        if not hasattr(self, 'df') or self.df is None:
            print("没有数据可绘制")
            return

        try:
            # 清除之前的绘图
            self.figure.clear()

            # ========== 自适应画布大小 ==========
            # 获取画布实时像素尺寸（Qt组件的width/height返回像素值）
            canvas_pixel_width = self.canvas.width()
            canvas_pixel_height = self.canvas.height()

            # 转换为英寸（与create_canvas中设置的DPI=100保持一致）
            fig_width = canvas_pixel_width / 100
            fig_height = canvas_pixel_height / 100

            # 设置图表尺寸为画布自适应大小
            self.figure.set_size_inches(fig_width, fig_height)

            # 创建5行1列的子图布局，共享x轴
            self.axes = self.figure.subplots(5, 1, sharex=True)
            self.figure.subplots_adjust(hspace=0.3)  # 增加子图间距

            # 将axes数组扁平化为一维数组，确保可以正确访问每个Axes对象
            self.axes = self.axes.ravel()

            # 设置字体大小和线条粗细
            title_font_size = 10
            label_font_size = 6
            tick_font_size = 8
            legend_font_size = 6
            line_width = 0.6

            # 绘制每个通道的数据
            for i, column in enumerate(self.df.columns):
                if i < len(self.axes):
                    ax = self.axes[i]
                    ax.plot(self.df.index, self.df[column], label=column, linewidth=line_width)

                    # 设置标题
                    ax.set_title(f'{column} time series ', fontsize=title_font_size)

                    # 设置纵坐标单位
                    if column in ['EX', 'EY']:
                        ax.set_ylabel("nV/m", fontsize=label_font_size)
                    else:
                        ax.set_ylabel("nT", fontsize=label_font_size)

                    # 设置刻度字体
                    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)

                    # 设置图例
                    ax.legend(fontsize=legend_font_size)

                    # 设置网格线
                    ax.grid(True, linewidth=0.1)

            # 设置x轴标签
            if hasattr(self.figure, 'supxlabel'):
                self.figure.supxlabel('time', fontsize=label_font_size)
            else:
                self.axes[-1].set_xlabel('time', fontsize=label_font_size)

            # 自动调整布局（适配自适应尺寸）
            self.figure.tight_layout()

            # 重绘画布
            self.canvas.draw()

            print("数据图已更新（自适应画布大小）")

        except Exception as e:
            print(f"更新数据图时出错: {e}")

    def create_canvas(self):
        """创建Matplotlib画布（兼容QWidget布局）"""
        try:
            # 主绘图区域
            groupBox_width = 700
            groupBox_height = 490
            toolbar_height = 30  # 工具栏固定高度
            margin = 10  # 边距

            # 计算画布尺寸（减去边距和工具栏高度）
            canvas_width = groupBox_width - 2 * margin
            canvas_height = groupBox_height - toolbar_height - 2 * margin

            # 设置Figure尺寸（英寸单位，需除以DPI）
            fig_width = canvas_width / 100
            fig_height = canvas_height / 100

            self.figure = Figure(figsize=(fig_width, fig_height), dpi=100)
            self.canvas = FigureCanvas(self.figure)

            # 添加导航工具栏并设置固定尺寸
            self.toolbar = NavigationToolbar(self.canvas, self.groupBox)
            self.toolbar.setFixedSize(canvas_width, toolbar_height)

            # 创建垂直布局（确保QWidget有布局才能添加组件）
            if not self.groupBox.layout():
                self.groupBox.setLayout(QVBoxLayout())

            # 通过布局添加组件（替代直接addWidget）
            self.groupBox.layout().addWidget(self.canvas)
            self.groupBox.layout().addWidget(self.toolbar)

            print("固定尺寸绘图画布初始化成功")
        except Exception as e:
            print(f"警告: 绘图画布初始化失败: {e}")

    def plot_calibration_response(self):
        """绘制标定频率响应曲线"""
        if self.calibrator is None:
            print("错误：请先进行仪器响应校正以创建校准器")
            QtWidgets.QMessageBox.warning(
                self,
                "警告",
                "请先进行仪器响应校正以创建校准器\n\n请先执行: 预处理 → 去仪器响应"
            )
            return

        try:
            print("正在绘制标定频率响应曲线...")
            self.calibrator.plot_calibration_response(self.figure, self.canvas)
        except Exception as e:
            print(f"绘制标定频率响应曲线时出错: {e}")
            import traceback
            traceback.print_exc()

    def plot_power_spectrum(self):
        """绘制功率谱密度图"""
        if not hasattr(self, 'data_interface') or self.data_interface.TSData is None:
            print("没有数据可计算PSD")
            return

        try:
            # 核心配置：设置mathtext字体为STIX（含完整数学符号）
            plt.rcParams["mathtext.fontset"] = "stix"  # 使用STIX字体渲染数学符号
            plt.rcParams["font.family"] = ["STIXGeneral", "DejaVu Sans", "Arial"]
            plt.rcParams["axes.unicode_minus"] = False  # 启用Unicode减

            # 清除之前的绘图
            self.figure.clear()

            # 计算PSD
            original_ts_data = self.data_interface.TSData
            self.data_interface.TSData = self.data_interface.filtered_data
            psd_results = self.data_interface.welch_psd(nperseg=self.config.fft_window_length)
            self.data_interface.TSData = original_ts_data

            # 创建子图
            self.axes = self.figure.add_subplot(111)

            # 绘制每个通道的PSD（转换为dB单位）
            for channel, result in psd_results.items():
                # 将线性PSD转换为分贝单位
                psd_dB = 10 * np.log10(result['psd'])
                # 使用semilogx（x轴对数，y轴线性，因为dB已经是"对数"值了）
                self.axes.semilogx(result['freqs'], psd_dB, label=channel, linewidth=0.3)

            self.axes.set_title("Power Spectral Density Plot")
            self.axes.set_xlabel("frequency (Hz)")
            self.axes.set_ylabel("power spectral density (dB/Hz)")
            self.axes.legend()
            self.axes.grid(True)

            # 自动调整布局
            self.figure.tight_layout()

            # 重绘画布
            self.canvas.draw()

            print("功率谱密度图已绘制")

        except Exception as e:
            print(f"绘制功率谱密度图时出错: {e}")

    def plot_pdf(self):
        """绘制功率谱概率密度图（分贝表示）"""
        if not hasattr(self, 'psd_pdf_results') or not self.psd_pdf_results:
            print("没有可用的功率谱概率密度数据，请先计算", "error")
            return

        try:
            # 核心配置：设置mathtext字体为STIX（含完整数学符号）
            plt.rcParams["mathtext.fontset"] = "stix"  # 使用STIX字体渲染数学符号
            plt.rcParams["font.family"] = ["STIXGeneral", "DejaVu Sans", "Arial"]
            plt.rcParams["axes.unicode_minus"] = False  # 启用Unicode减

            # 清除现有图表
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            import matplotlib.colors as mcolors

            # 正确的颜色列表格式（无嵌套）
            colors = [
                mcolors.to_rgba((237 / 255, 173 / 255, 197 / 255)),
                mcolors.to_rgba((206 / 255, 170 / 255, 208 / 255)),
                mcolors.to_rgba((149 / 255, 132 / 255, 193 / 255)),
                mcolors.to_rgba((108 / 255, 190 / 255, 195 / 255)),
                mcolors.to_rgba((170 / 255, 215 / 255, 200 / 255))
            ]
            markers = ['-', '--', '-.', ':', '-']
            channel_names = list(self.psd_pdf_results.keys())

            # 收集所有通道的有效x轴数据范围
            all_valid_centers = []
            all_x_ranges = []  # 存储每个通道的有效x范围

            for i, channel in enumerate(channel_names):
                data = self.psd_pdf_results[channel]
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]

                # 检查数据结构是否匹配
                if 'bin_centers' not in data or 'pdf' not in data:
                    print(f"通道 {channel} 的数据结构不完整，跳过绘图")
                    continue

                # 绘制概率密度曲线（使用分贝值）
                bin_centers = data['bin_centers']
                pdf = data['pdf']

                # 过滤掉概率密度为零或接近零的点（避免对数坐标问题）
                # 使用相对阈值而不是绝对阈值
                max_pdf = np.max(pdf)
                if max_pdf > 0:
                    valid_mask = pdf > max_pdf * 0.01  # 保留大于最大概率1%的点
                else:
                    valid_mask = pdf > 0

                if np.sum(valid_mask) < 2:  # 至少需要两个点才能绘制曲线
                    print(f"通道 {channel} 有效数据点不足，跳过绘图")
                    continue

                # 记录有效数据范围
                if np.sum(valid_mask) > 0:
                    valid_centers = bin_centers[valid_mask]
                    all_valid_centers.extend(valid_centers)
                    all_x_ranges.append([np.min(valid_centers), np.max(valid_centers)])

                ax.plot(bin_centers[valid_mask], pdf[valid_mask], marker,
                        color=color, label=f' {channel}', linewidth=1.5)

            if not all_valid_centers:
                print("没有有效数据可绘制", "error")
                return

            # 设置图表属性
            ax.set_title('Power Spectral Density Probability Density Distribution', fontsize=12)
            ax.set_xlabel('Power Spectral Density (dB)', fontsize=10)
            ax.set_ylabel('Probability Density', fontsize=10)

            # 动态设置x轴范围
            if all_valid_centers:
                # 计算所有有效数据的总体范围
                overall_min = np.min(all_valid_centers)
                overall_max = np.max(all_valid_centers)

                # 计算数据范围
                data_range = overall_max - overall_min

                # 如果数据范围很窄，扩展显示范围
                if data_range < 20:  # 如果数据范围小于20dB，认为是"很窄"
                    # 扩展范围，至少显示30dB的范围
                    extension = max(15, 30 - data_range) / 2
                    x_min = overall_min - extension
                    x_max = overall_max + extension

                    # 检查是否有bin_edges边界信息，确保不超出计算范围
                    if channel_names and 'bin_edges' in self.psd_pdf_results[channel_names[0]]:
                        bin_edges = self.psd_pdf_results[channel_names[0]]['bin_edges']
                        x_min = max(bin_edges[0], x_min)
                        x_max = min(bin_edges[-1], x_max)

                    # 如果扩展后仍然很窄，则使用更大的扩展
                    if (x_max - x_min) < 30:
                        center = (x_min + x_max) / 2
                        x_min = center - 15
                        x_max = center + 15

                        # 再次检查边界
                        if 'bin_edges' in self.psd_pdf_results[channel_names[0]]:
                            bin_edges = self.psd_pdf_results[channel_names[0]]['bin_edges']
                            x_min = max(bin_edges[0], x_min)
                            x_max = min(bin_edges[-1], x_max)
                else:
                    # 数据范围已经足够大，直接使用数据范围并稍微扩展
                    margin = data_range * 0.05  # 5%的边距
                    x_min = overall_min - margin
                    x_max = overall_max + margin

                    # 检查边界
                    if channel_names and 'bin_edges' in self.psd_pdf_results[channel_names[0]]:
                        bin_edges = self.psd_pdf_results[channel_names[0]]['bin_edges']
                        x_min = max(bin_edges[0], x_min)
                        x_max = min(bin_edges[-1], x_max)

                ax.set_xlim(x_min, x_max)

                print(f"设置的x轴范围: [{x_min:.2f}, {x_max:.2f}]，数据范围: [{overall_min:.2f}, {overall_max:.2f}]")

            # 动态设置y轴为对数坐标
            # 收集所有概率密度值
            all_pdf_values = []
            for channel in channel_names:
                if channel in self.psd_pdf_results:
                    data = self.psd_pdf_results[channel]
                    if 'pdf' in data:
                        all_pdf_values.extend(data['pdf'][data['pdf'] > 0])

            if all_pdf_values:
                pdf_min = np.min(all_pdf_values)
                pdf_max = np.max(all_pdf_values)

                # 如果概率密度值跨越多个数量级，使用对数坐标
                if pdf_max / pdf_min > 100:  # 跨越2个数量级以上
                    ax.set_yscale('log')
                    ax.set_ylabel('Probability Density (log scale)', fontsize=10)

                    # 设置合适的y轴范围（避免对数坐标下的负值问题）
                    y_min = pdf_min * 0.5
                    y_max = pdf_max * 2
                    ax.set_ylim(y_min, y_max)
                else:
                    # 线性坐标下稍微扩展y轴范围
                    margin = (pdf_max - pdf_min) * 0.1
                    ax.set_ylim(max(0, pdf_min - margin), pdf_max + margin)

            ax.grid(True, linestyle='--', alpha=0.7)

            # 添加图例
            if channel_names:
                ax.legend(fontsize=8, loc='best', framealpha=0.7)

            # 设置刻度标签大小
            ax.tick_params(axis='both', which='major', labelsize=8)

            # 调整布局并刷新画布
            self.figure.tight_layout()
            self.canvas.draw()

            print("功率谱概率密度图绘制完成")

        except Exception as e:
            print(f"绘制功率谱概率密度图时出错: {str(e)}", "critical")
            import traceback
            traceback.print_exc()

    def plot_allan_variance(self):
        """绘制Allan方差图"""
        if not hasattr(self, 'data_interface') or self.data_interface.TSData is None:
            print("没有数据可进行Allan方差分析")
            return

        try:
            plt.rcParams["mathtext.fontset"] = "stix"
            plt.rcParams["font.family"] = ["STIXGeneral", "DejaVu Sans", "Arial"]
            plt.rcParams["axes.unicode_minus"] = False

            self.figure.clear()
            self.axes = self.figure.add_subplot(111)

            # 通道名称和单位映射
            channel_config = {
                'EX': {'name': 'EX', 'unit': 'mV', 'color': '#1f77b4'},
                'EY': {'name': 'EY', 'unit': 'mV', 'color': '#ff7f0e'},
                'HX': {'name': 'HX', 'unit': 'nT', 'color': '#2ca02c'},
                'HY': {'name': 'HY', 'unit': 'nT', 'color': '#d62728'},
                'HZ': {'name': 'HZ', 'unit': 'nT', 'color': '#9467bd'}
            }

            # 保存原始数据，使用滤波数据计算
            original_ts_data = self.data_interface.TSData

            # 检查是否有滤波数据，如果没有使用原始数据
            if hasattr(self.data_interface, 'filtered_data') and self.data_interface.filtered_data is not None:
                self.data_interface.TSData = self.data_interface.filtered_data
            else:
                print("使用原始数据进行Allan方差分析")

            # 获取实际通道名称
            max_channels = min(5, self.data_interface.TSData.shape[1])
            if isinstance(self.data_interface.TSData, pd.DataFrame):
                actual_channel_names = self.data_interface.TSData.columns.tolist()[:max_channels]
            else:
                actual_channel_names = [f'Ch{i}' for i in range(max_channels)]

            # 计算并绘制每个通道的Allan方差
            for channel_idx in range(max_channels):
                time_taus, avar, _, _ = self.data_interface.calculate_allan_variance_in(
                    channel=channel_idx
                )

                if time_taus is not None and avar is not None and len(time_taus) > 0:
                    # 确定通道名称和单位
                    actual_name = actual_channel_names[channel_idx]

                    # 尝试匹配通道配置
                    channel_key = None
                    for key in channel_config:
                        if key in actual_name.upper():
                            channel_key = key
                            break

                    if channel_key and channel_key in channel_config:
                        config = channel_config[channel_key]
                        label = f"{config['name']} ({config['unit']})"
                        color = config['color']
                        # 单位转换：根据单位调整显示
                        if config['unit'] == 'mV':
                            # 电场数据：单位转换为(mV)²
                            display_avar = avar  # 单位：(mV)²
                        elif config['unit'] == 'nT':
                            # 磁场数据：单位转换为(nT)²
                            display_avar = avar  # 单位：(nT)²
                        else:
                            display_avar = avar
                            label = f"{actual_name}"
                    else:
                        # 未知通道类型
                        label = f"{actual_name}"
                        color = f'C{channel_idx}'
                        display_avar = avar

                    self.axes.loglog(time_taus, display_avar, 'o-',
                                     color=color,
                                     markersize=4,
                                     label=label,
                                     linewidth=1.5)

            # 设置图表属性
            self.axes.set_title("Allan Variance Analysis of Time Series Data", fontsize=12)
            self.axes.set_xlabel("Averaging Time τ (s)", fontsize=10)

            # 根据数据通道类型设置纵坐标标签
            # 如果同时有电场和磁场，标注通用单位
            has_electric = any('EX' in name.upper() or 'EY' in name.upper() for name in actual_channel_names)
            has_magnetic = any(
                'HX' in name.upper() or 'HY' in name.upper() or 'HZ' in name.upper() for name in actual_channel_names)

            if has_electric and not has_magnetic:
                ylabel = "Allan Variance σ²(τ) (（mV/km）²)"
            elif has_magnetic and not has_electric:
                ylabel = "Allan Variance σ²(τ) (nT²)"
            else:
                ylabel = "Allan Variance σ²(τ)"

            self.axes.set_ylabel(ylabel, fontsize=10)

            self.axes.legend(fontsize=9, loc='best', framealpha=0.8)
            self.axes.grid(True, which='both', linestyle='--', alpha=0.7)

            # 恢复原始数据
            self.data_interface.TSData = original_ts_data

            self.figure.tight_layout()
            self.canvas.draw()

            print("Allan方差图已绘制完成")

        except Exception as e:
            print(f"绘制Allan方差图时出错: {e}")
            import traceback
            traceback.print_exc()

    def plot_wavelet_transform(self):
        """绘制小波变换图（修复对数坐标空白+支持多选通道+自适应布局）"""
        if not hasattr(self, 'data_interface') or self.data_interface.TSData is None:
            print("没有数据可进行小波变换")
            return

        # 使用滤波后的数据计算
        original_ts_data = self.data_interface.TSData
        self.data_interface.TSData = self.data_interface.filtered_data

        try:
            # ========== 1. 修复数学符号字体问题（避免负号/减号显示异常） ==========
            plt.rcParams["mathtext.fontset"] = "stix"
            plt.rcParams["font.family"] = ["STIXGeneral", "DejaVu Sans", "Arial"]
            plt.rcParams["axes.unicode_minus"] = False

            # ========== 2. 数据有效性检查 ==========
            data = self.data_interface.TSData
            if data is None:
                print("数据为空")
                return

            channel_count = data.shape[1] if data.ndim > 1 else 0
            if channel_count == 0:
                print(f"数据列数为0，无法进行小波变换")
                return

            # 通道名称映射
            channel_names = ['EX', 'EY', 'HX', 'HY', 'HZ']
            channel_options = [f"{i + 1}. {channel_names[i] if i < len(channel_names) else f'通道{i + 1}'}"
                               for i in range(channel_count)]

            # ========== 3. 改造对话框：支持多选通道 ==========
            dialog = QDialog()
            dialog.setWindowTitle("选择通道（可多选）")
            dialog.setMinimumSize(300, 250)

            layout = QVBoxLayout(dialog)
            layout.addWidget(QLabel("请选择要进行小波变换的通道（按住Ctrl/Shift可多选）:"))

            list_widget = QListWidget()
            # 修复：PyQt6中使用 ExtendedSelection 替代 MultiExtended
            list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
            for option in channel_options:
                list_widget.addItem(option)
            list_widget.setCurrentRow(0)  # 默认选中第一个通道
            layout.addWidget(list_widget)

            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                          QDialogButtonBox.StandardButton.Cancel)
            button_box.accepted.connect(dialog.accept)  # type: ignore
            button_box.rejected.connect(dialog.reject)  # type: ignore
            layout.addWidget(button_box)

            # 显示对话框并获取选中结果
            result = dialog.exec()
            if result != QDialog.DialogCode.Accepted:
                print("用户取消了操作")
                return

            # 获取所有选中的通道索引
            selected_items = list_widget.selectedItems()
            if not selected_items:
                print("未选择任何有效通道")
                return
            selected_channel_idxs = []
            for item in selected_items:
                try:
                    idx = int(item.text().split('.')[0]) - 1
                    if 0 <= idx < channel_count:
                        selected_channel_idxs.append(idx)
                except (ValueError, IndexError):
                    continue
            if not selected_channel_idxs:
                print("未获取到有效通道索引")
                return

            # ========== 4. 清空画布，准备自适应布局 ==========
            self.figure.clear()
            self.figure.set_tight_layout(True)  # 开启画布自动紧凑布局
            # 设置画布大小策略为自适应
            self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            # ========== 5. 动态创建子图布局（根据选中通道数自适应） ==========
            n_channels = len(selected_channel_idxs)
            # 布局策略：≤3个通道用单列，>3个通道用2列，自动适配行数
            n_cols = 2 if n_channels > 3 else 1
            n_rows = (n_channels + n_cols - 1) // n_cols  # 向上取整计算行数

            # 遍历选中的通道，逐个绘制
            for plot_idx, channel_idx in enumerate(selected_channel_idxs, 1):
                # 创建子图（1-based索引）
                self.axes = self.figure.add_subplot(n_rows, n_cols, plot_idx)

                # ========== 6. 计算小波变换并修复数据（核心解决空白问题） ==========
                cwtmatr, frequencies, scales = self.data_interface.wavelet_transform(channel=channel_idx)
                if cwtmatr is None or frequencies is None or cwtmatr.shape[1] == 0:
                    print(f"通道 {channel_idx + 1} 小波变换结果无效，跳过绘制")
                    self.axes.set_title(f"通道 {channel_names[channel_idx]}: 数据无效")
                    continue

                # 计算时间轴
                time_points = data.shape[0]
                time = np.arange(time_points) / self.data_interface.sample_rate

                # ========== 关键修复：处理对数坐标不支持的0/负数 ==========
                # 1. 替换时间轴中的0/负数为极小正数（避免对数坐标无法显示）
                time = np.where(time <= 1e-8, 1e-8, time)
                # 2. 替换频率轴中的0/负数为极小正数
                frequencies = np.where(frequencies <= 1e-8, 1e-8, frequencies)
                # 3. 确保cwtmatr为正数（取绝对值，避免绘图异常）
                cwt_abs = np.abs(cwtmatr)

                # ========== 7. 绘制热力图（优化对数坐标显示） ==========
                im = self.axes.pcolormesh(time, frequencies, cwt_abs,
                                          shading='auto', cmap='jet',
                                          rasterized=True)  # 优化渲染性能

                # ========== 8. 设置对数坐标+优化刻度（解决空白/刻度稀疏问题） ==========
                self.axes.set_xscale('log')
                self.axes.set_yscale('log')
                # 设置坐标范围为数据实际范围（避免空白区域）
                self.axes.set_xlim(time.min(), time.max())
                self.axes.set_ylim(frequencies.min(), frequencies.max())
                # 优化对数坐标刻度显示（避免刻度过于密集/稀疏）
                self.axes.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
                self.axes.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
                self.axes.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs='auto'))
                self.axes.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs='auto'))

                # ========== 9. 设置标题/标签（自适应布局） ==========
                self.axes.set_title(f"Wavelet Transform: {channel_names[channel_idx]}", fontsize=10)
                self.axes.set_xlabel("Time (s)", fontsize=8)
                self.axes.set_ylabel("Frequency (Hz)", fontsize=8)
                self.axes.tick_params(labelsize=7)  # 缩小刻度标签，适配多子图

                # ========== 10. 为每个子图添加独立颜色条（自适应大小） ==========
                cbar = self.figure.colorbar(im, ax=self.axes, label='Amplitude', shrink=0.8)
                cbar.ax.tick_params(labelsize=7)
                cbar.set_label('Amplitude', fontsize=8)

            # ========== 11. 自适应画布布局（核心：避免子图重叠/空白） ==========
            self.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,
                                        wspace=0.3, hspace=0.4)  # 调整子图间距
            self.canvas.draw()  # 重绘画布
            self.canvas.update()  # 更新画布显示

            print(f"成功绘制 {n_channels} 个通道的小波变换图")

        except Exception as e:
            print(f"绘制小波变换图时出错: {e}")
            import traceback
            traceback.print_exc()  # 打印详细异常栈，便于调试
        finally:
            # 恢复为原始数据
            self.data_interface.TSData = original_ts_data

    def plot_raw_vs_filtered(self):
        """绘制原始数据与滤波后数据的对比图（一列五行细长型，仅用两种颜色）"""
        # 检查数据是否存在
        if not hasattr(self, 'raw_time_data') or self.raw_time_data is None or self.raw_time_data.empty:
            print("请先加载原始数据")
            return

        if not hasattr(self.data_interface, 'filtered_data') or self.data_interface.filtered_data is None:
            print("请先进行滤波处理，生成滤波后的数据")
            return

        try:
            # 清除现有图表
            self.figure.clear()

            # 获取通道数量和名称
            channels = self.raw_time_data.columns.tolist()
            num_channels = len(channels)

            # 强制1列布局（行数=通道数，确保一列五行）
            rows = num_channels
            cols = 1

            # # 设置细长型画布尺寸（宽6，高15，适配垂直布局）
            # self.figure.set_size_inches(6, 15)

            # 获取时间轴数据
            sample_rate = getattr(self.config, 'sample_rate', 1.0)
            time = np.arange(len(self.raw_time_data)) / sample_rate

            # 定义两种颜色：原始数据用蓝色，滤波后用红色（可根据需要调整）
            raw_color = '#1f77b4'  # 蓝色（原始数据）
            filtered_color = '#d62728'  # 红色（滤波后数据）

            # 为每个通道创建子图（垂直排列）
            for i, channel in enumerate(channels, 1):
                ax = self.figure.add_subplot(rows, cols, i)

                # 绘制原始数据：实线+原始数据颜色
                ax.plot(time, self.raw_time_data[channel], linestyle='-', color=raw_color,
                        alpha=0.8, linewidth=0.3, label='raw_data')

                # 绘制滤波后数据：虚线+滤波后数据颜色
                ax.plot(time, self.data_interface.filtered_data[channel], linestyle='-', color=filtered_color,
                        alpha=0.8, linewidth=0.3, label='filtered_data')

                # 设置子图属性
                ax.set_title(f'channel {channel}', fontsize=9)
                ax.set_xlabel('time (s)', fontsize=8)
                if channels in ['EX', 'EY']:
                    ax.set_ylabel("nV/m", fontsize=8)
                else:
                    ax.set_ylabel("nT", fontsize=8)
                ax.tick_params(axis='both', labelsize=7)
                ax.legend(fontsize=7)  # 每个子图保留图例，清晰区分两种数据
                ax.grid(True, linestyle='--', alpha=0.5)

            # 调整布局
            self.figure.suptitle('raw_data vs filtered_data', fontsize=12)
            self.figure.tight_layout(rect=[0, 0, 1, 0.98])  # 预留标题空间
            self.canvas.draw()

            print("原始数据与滤波后数据对比图（两种颜色）绘制完成")

        except Exception as e:
            print(f"绘制数据对比图时出错: {e}")

    def plot_pca_results(self, pca_results, plot_heatmap=True):
        """可视化PCA分析结果，适配固定尺寸画布"""
        # 清除现有图表
        self.figure.clear()

        # 根据画布尺寸计算布局（画布实际尺寸：680x440像素 @100DPI → 6.8x4.4英寸）
        rows = 2 if plot_heatmap else 1
        cols = 1

        # 1. 绘制解释方差比例图
        ax1 = self.figure.add_subplot(rows, cols, 1)
        explained = pca_results['explained_variance_ratio']
        cumulative = np.cumsum(explained)
        n = len(explained)

        # 绘制条形图和累积曲线
        bars = ax1.bar(range(1, n + 1), explained, alpha=0.7,
                       color='#4CAF50', label='Single-factor variance')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(range(1, n + 1), cumulative, 'r-', marker='o',
                      linewidth=1.5, markersize=4, label='Cumulative Explained Variance')

        # 添加数值标签（适配小画布）
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{explained[i]:.1%}', ha='center', va='bottom', fontsize=7)

        # 优化坐标轴（适配小尺寸）
        ax1.set_xlabel('Number of principal components', fontsize=9)
        ax1.set_ylabel('Single-factor variance', fontsize=9)
        ax1_twin.set_ylabel('Cumulative Explained Variance', fontsize=9, color='red')
        ax1.set_title('Proportion of variance explained by principal components', fontsize=10, pad=5)

        # 限制y轴范围
        ax1.set_ylim(0, min(1.1, max(explained) * 1.2))
        ax1_twin.set_ylim(0, 1.1)

        # 精简x轴刻度
        if n > 8:  # 画布较窄，减少刻度数量
            step = max(1, n // 8)
            ax1.set_xticks(range(1, n + 1, step))
        else:
            ax1.set_xticks(range(1, n + 1))
        ax1.tick_params(axis='x', labelsize=8)
        ax1.tick_params(axis='y', labelsize=8)
        ax1_twin.tick_params(axis='y', labelsize=8, colors='red')

        # 添加参考线
        ax1.axhline(y=0.8, color='gray', linestyle='--', alpha=0.7)
        ax1.text(0.02, 0.82, '80% threshold', transform=ax1.transAxes,
                 ha='left', va='bottom', color='gray', fontsize=7)

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best',
                   fontsize=7, frameon=True, framealpha=0.9)

        ax1.grid(True, alpha=0.3, linestyle='--')

        # 2. 绘制载荷热力图（适配小画布）
        if plot_heatmap and pca_results['components'].size > 0:
            ax2 = self.figure.add_subplot(rows, cols, 2)
            components = pca_results['components']

            # 绘制热力图（调整 aspect 适应宽度）
            im = ax2.imshow(components, cmap='coolwarm', aspect='auto')

            # 紧凑的颜色条
            cbar = self.figure.colorbar(im, ax=ax2, shrink=0.8)
            cbar.set_label('Load value', fontsize=7)
            cbar.ax.tick_params(labelsize=6)

            # 精简标签（避免溢出）
            ax2.set_xticks(range(len(self.df.columns)))
            ax2.set_xticklabels(self.df.columns, rotation=45, ha='right',
                                fontsize=6, rotation_mode='anchor')
            ax2.set_yticks(range(components.shape[0]))
            ax2.set_yticklabels([f'PC{i + 1}' for i in range(components.shape[0])],
                                fontsize=7)
            ax2.set_title('Principal Component Load Heat Map', fontsize=10, pad=5)
            ax2.tick_params(axis='both', which='major', pad=2)  # 减少刻度与标签间距

        # 关键：适配固定画布的布局调整
        self.figure.tight_layout()
        if plot_heatmap:
            # 垂直间距根据画布高度调整（总高度4.4英寸）
            self.figure.subplots_adjust(hspace=0.45, top=0.92, bottom=0.1, left=0.12, right=0.92)
        else:
            self.figure.subplots_adjust(top=0.92, bottom=0.15, left=0.12, right=0.92)

        self.canvas.draw()

    def plot_psd_comparison(self):
        """绘制原始数据与滤波后数据的功率谱密度对比图（一列多行细长型，仅用两种颜色，自适应坐标轴范围）"""
        # 数据有效性检查
        if not hasattr(self, 'data_interface'):
            print("数据接口不存在")
            return
        if self.data_interface.TSData is None:
            print("没有原始数据可计算PSD")
            return
        if not hasattr(self.data_interface, 'filtered_data') or self.data_interface.filtered_data is None:
            print("没有滤波后的数据，无法生成对比图")
            return

        try:
            # 保持字体配置
            plt.rcParams["mathtext.fontset"] = "stix"
            plt.rcParams["font.family"] = ["STIXGeneral", "DejaVu Sans", "Arial"]
            plt.rcParams["axes.unicode_minus"] = False

            # 清除现有图表
            self.figure.clear()

            # 计算原始和滤波后PSD
            raw_psd = self.data_interface.welch_psd(nperseg=self.config.fft_window_length)
            original_ts_data = self.data_interface.TSData
            self.data_interface.TSData = self.data_interface.filtered_data
            filtered_psd = self.data_interface.welch_psd(nperseg=self.config.fft_window_length)
            self.data_interface.TSData = original_ts_data

            # 获取通道列表和数量
            channels = list(raw_psd.keys())
            num_channels = len(channels)
            if num_channels == 0:
                print("未检测到任何通道的PSD数据")
                return

            # 强制1列布局（行数=通道数，细长型）
            rows = num_channels
            cols = 1

            # # 设置细长型画布尺寸（宽6，高15，适配垂直布局）
            # self.figure.set_size_inches(6, 15)

            # 定义两种固定颜色：原始数据用蓝色，滤波后用红色
            raw_color = '#1f77b4'  # 蓝色（原始PSD）
            filtered_color = '#d62728'  # 红色（滤波后PSD）

            # 为每个通道创建子图（垂直排列）
            for i, channel in enumerate(channels, 1):
                # 创建当前通道的子图
                ax = self.figure.add_subplot(rows, cols, i)

                # 确保数据是numpy数组
                # 原始PSD数据
                raw_freqs = np.asarray(raw_psd[channel]['freqs'])
                raw_vals = np.asarray(raw_psd[channel]['psd'])

                # 滤波后PSD数据
                filt_freqs = np.asarray(filtered_psd[channel]['freqs'])
                filt_vals = np.asarray(filtered_psd[channel]['psd'])

                # ============= 关键修复：正确创建掩码 =============
                # 创建布尔掩码（一次创建，多次使用）
                raw_mask = raw_vals > 0
                filt_mask = filt_vals > 0

                # 使用掩码过滤数据
                raw_freqs_valid = raw_freqs[raw_mask]
                raw_vals_valid = raw_vals[raw_mask]

                filt_freqs_valid = filt_freqs[filt_mask]
                filt_vals_valid = filt_vals[filt_mask]
                # ================================================

                # ============= 转换为dB单位 =============
                # 如果仍有0值，设置最小正值
                MIN_VALUE = 1e-30

                # 处理原始数据
                raw_vals_safe = np.copy(raw_vals_valid)
                # 确保没有0或负值
                raw_vals_safe[raw_vals_safe <= 0] = MIN_VALUE
                raw_vals_dB = 10 * np.log10(raw_vals_safe)

                # 处理滤波后数据
                filt_vals_safe = np.copy(filt_vals_valid)
                filt_vals_safe[filt_vals_safe <= 0] = MIN_VALUE
                filt_vals_dB = 10 * np.log10(filt_vals_safe)

                # ============= 绘制曲线 =============
                # 原始数据（实线）
                ax.semilogx(
                    raw_freqs_valid, raw_vals_dB,
                    linestyle='-', color=raw_color, alpha=0.9, linewidth=0.3,
                    label='Raw PSD'
                )

                # 滤波后数据（虚线）
                ax.semilogx(
                    filt_freqs_valid, filt_vals_dB,
                    linestyle='--', color=filtered_color, alpha=0.9, linewidth=0.3,
                    label='Filtered PSD'
                )

                # ============= 设置坐标轴范围 =============
                # 合并所有dB值
                all_dB = []
                if len(raw_vals_dB) > 0:
                    all_dB.extend(raw_vals_dB)
                if len(filt_vals_dB) > 0:
                    all_dB.extend(filt_vals_dB)

                if len(all_dB) > 0:
                    all_dB_array = np.array(all_dB)
                    # 计算数据范围
                    dB_min = np.min(all_dB_array)
                    dB_max = np.max(all_dB_array)
                    dB_range = dB_max - dB_min

                    # 设置合理的范围
                    if dB_range > 0:
                        ax.set_ylim(dB_min - 0.1 * dB_range, dB_max + 0.1 * dB_range)
                    else:
                        ax.set_ylim(dB_min - 10, dB_max + 10)
                else:
                    # 默认范围
                    ax.set_ylim(-100, 0)

                # 设置x轴范围
                all_freqs = []
                if len(raw_freqs_valid) > 0:
                    all_freqs.extend(raw_freqs_valid)
                if len(filt_freqs_valid) > 0:
                    all_freqs.extend(filt_freqs_valid)

                if len(all_freqs) > 0:
                    all_freqs_array = np.array(all_freqs)
                    freq_min = np.min(all_freqs_array[all_freqs_array > 0])
                    freq_max = np.max(all_freqs_array)

                    if 0 < freq_min < freq_max:
                        # 对数坐标范围设置
                        log_min = np.log10(freq_min)
                        log_max = np.log10(freq_max)
                        log_range = log_max - log_min

                        ax.set_xlim(
                            10 ** (log_min - 0.05 * log_range),
                            10 ** (log_max + 0.05 * log_range)
                        )

                # ============= 设置子图属性 =============
                ax.set_title(f'Channel {channel}', fontsize=10)
                ax.set_xlabel("Frequency (Hz)", fontsize=8)
                ax.set_ylabel("PSD (dB/Hz)", fontsize=8)
                ax.legend(loc='best', fontsize=7)
                ax.tick_params(axis='both', labelsize=7)
                ax.grid(True, linestyle='--', alpha=0.5)

            # 设置总标题
            self.figure.suptitle('Raw vs Filtered Power Spectral Density', fontsize=12)
            self.figure.tight_layout(rect=[0, 0, 1, 0.98])
            self.canvas.draw()

            print("功率谱密度对比图已绘制（dB刻度）")

        except Exception as e:
            print(f"绘制PSD对比图时出错: {e}")

    def plot_allan_comparison(self):
        """
        绘制原始数据与滤波后数据的Allan偏差对比图
        风格：一列多行细长型布局，仅用两种固定颜色，自适应坐标轴范围（排除极端值）
        """
        if not hasattr(self, 'data_interface'):
            print("数据接口不存在", "error")
            return
        if self.data_interface.TSData is None:
            print("没有原始数据可计算Allan偏差", "error")
            return
        if not hasattr(self.data_interface, 'filtered_data') or self.data_interface.filtered_data is None:
            print("没有滤波后的数据，无法生成Allan偏差对比图", "error")
            return

        try:
            # 全局字体配置（和PSD图保持一致）
            plt.rcParams["mathtext.fontset"] = "stix"
            plt.rcParams["font.family"] = ["STIXGeneral", "DejaVu Sans", "Arial"]
            plt.rcParams["axes.unicode_minus"] = False

            self.figure.clear()

            if not self._validate_data_interface():
                return

            # 计算原始/滤波后数据的Allan偏差
            # 保存原始数据引用，避免覆盖
            original_ts_data = self.data_interface.TSData
            max_channels = min(5, self.data_interface.TSData.shape[1])
            channel_names = self._get_channel_names(max_channels)

            # 计算原始数据的Allan偏差
            print("开始计算原始数据的Allan偏差...")
            raw_allan_data = {}
            for i in range(max_channels):
                taus, adev, _, _ = self.data_interface.calculate_allan_deviation(channel=i)
                # 过滤无效数据（确保taus和adev非空且长度一致）
                if taus is not None and adev is not None and len(taus) == len(adev):
                    raw_allan_data[channel_names[i]] = {
                        'taus': np.array(taus),
                        'adev': np.array(adev)
                    }

            # 计算滤波后数据的Allan偏差
            print("开始计算滤波后数据的Allan偏差...")
            self.data_interface.TSData = self.data_interface.filtered_data
            filtered_allan_data = {}
            for i in range(max_channels):
                taus, adev, _, _ = self.data_interface.calculate_allan_deviation(channel=i)
                if taus is not None and adev is not None and len(taus) == len(adev):
                    filtered_allan_data[channel_names[i]] = {
                        'taus': np.array(taus),
                        'adev': np.array(adev)
                    }

            # 恢复原始数据（避免影响后续操作）
            self.data_interface.TSData = original_ts_data

            # 筛选有效对比通道
            valid_channels = [
                ch for ch in channel_names
                if ch in raw_allan_data and ch in filtered_allan_data
            ]
            if not valid_channels:
                print("没有有效的Allan偏差对比数据", "error")
                return
            num_channels = len(valid_channels)

            # 绘图布局与样式配置
            # 固定1列多行布局（细长型）
            rows = num_channels
            cols = 1

            # 固定配色（和PSD图保持一致）
            raw_color = '#1f77b4'  # 蓝色：原始数据
            filtered_color = '#d62728'  # 红色：滤波后数据

            # 逐通道绘制子图
            for idx, channel in enumerate(valid_channels, 1):
                # 创建子图
                ax = self.figure.add_subplot(rows, cols, idx)

                # 数据预处理（过滤非正数，避免对数坐标报错）
                # 原始Allan数据
                raw_taus = raw_allan_data[channel]['taus']
                raw_adev = raw_allan_data[channel]['adev']
                raw_mask = (raw_adev > 0) & (raw_taus > 0)  # 过滤非正数
                raw_taus_valid = raw_taus[raw_mask]
                raw_adev_valid = raw_adev[raw_mask]

                # 滤波后Allan数据
                filt_taus = filtered_allan_data[channel]['taus']
                filt_adev = filtered_allan_data[channel]['adev']
                filt_mask = (filt_adev > 0) & (filt_taus > 0)
                filt_taus_valid = filt_taus[filt_mask]
                filt_adev_valid = filt_adev[filt_mask]

                # 绘制Allan偏差曲线（log-log坐标）
                # 原始数据曲线
                ax.loglog(
                    raw_taus_valid, raw_adev_valid,
                    linestyle='-', color=raw_color, alpha=0.8, linewidth=1,
                    label='raw_allan'
                )

                # 滤波后数据曲线
                ax.loglog(
                    filt_taus_valid, filt_adev_valid,
                    linestyle='-', color=filtered_color, alpha=0.8, linewidth=1,
                    label='filtered_allan'
                )

                # 自适应坐标轴范围（排除1%极端值，留余量）
                # 纵轴（Allan偏差）：对数坐标，排除1%/99%分位数
                all_adev = np.concatenate([raw_adev_valid, filt_adev_valid]) if len(raw_adev_valid) and len(
                    filt_adev_valid) else None
                if all_adev is not None and len(all_adev) > 0:
                    adev_min = np.percentile(all_adev, 1)
                    adev_max = np.percentile(all_adev, 99)
                    ax.set_ylim(adev_min * 0.9, adev_max * 1.1)
                else:
                    print(f"警告：通道{channel}无有效Allan偏差值，使用默认纵轴范围", "warning")

                # 横轴（平均时间）：对数坐标，取实际范围±5%余量
                all_taus = np.concatenate([raw_taus_valid, filt_taus_valid]) if len(raw_taus_valid) and len(
                    filt_taus_valid) else None
                if all_taus is not None and len(all_taus) > 0:
                    taus_min = np.min(all_taus)
                    taus_max = np.max(all_taus)
                    ax.set_xlim(taus_min * 0.95, taus_max * 1.05)
                else:
                    print(f"警告：通道{channel}无有效平均时间值，使用默认横轴范围", "warning")

                ax.set_title(f'channel {channel}', fontsize=9)
                ax.set_xlabel("averaging time (s)", fontsize=8)
                ax.set_ylabel("Allan deviation", fontsize=8)
                ax.legend(loc='upper right', fontsize=7)
                ax.tick_params(axis='both', labelsize=7)  # 小刻度字号
                ax.grid(True, linestyle='--', alpha=0.5)  # 虚线网格

            self.figure.suptitle('raw_allan vs filtered_allan', fontsize=12)
            self.figure.tight_layout(rect=[0, 0, 1, 0.98])  # 预留总标题空间
            self.canvas.draw()

            print("Allan偏差对比图已绘制完成（一列多行细长型，自适应坐标轴范围）")

        except Exception as e:
            print(f"绘制Allan偏差对比图时发生错误: {str(e)}", "critical")

# ================================================ 存储模块 =============================================================

# =============================================== 其他内置函数 ===========================================================
    def closeEvent(self, event):
        self.redirector.restore()
        event.accept()


if __name__ == "__main__":
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    window = NewMainWindow()
    window.show()
    sys.exit(app.exec())
