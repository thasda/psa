import sys
from PyQt6.QtWidgets import QMainWindow
from pythonProject.src.interface.软件界面 import Ui_MainWindow
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from pythonProject.src.services import psd_service, allan_service
from pythonProject.src.utils import matdataprocessor, lemi417, preprocess, HDF5_seek


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