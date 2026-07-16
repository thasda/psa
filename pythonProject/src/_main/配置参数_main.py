import sys
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import QDateTime

from pythonProject.src.interface.配置参数 import Ui_Dialog


class DialogWindow(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent=None, encoding="utf-8"):
        super().__init__(parent)
        self.setupUi(self)

        self.encoding = encoding

        # 设置默认时间
        current_time = QDateTime.currentDateTime()
        self.dateTimeEdit.setDateTime(current_time)
        self.dateTimeEdit_2.setDateTime(current_time.addSecs(3600))
        # 连接时间变化信号，确保结束时间不早于开始时间
        self.dateTimeEdit.dateTimeChanged.connect(self.validate_times)
        self.dateTimeEdit_2.dateTimeChanged.connect(self.validate_times)

    def validate_times(self):
        """验证时间设置是否合理"""
        start_time = self.dateTimeEdit.dateTime()
        end_time = self.dateTimeEdit_2.dateTime()

        if end_time <= start_time:
            # 如果结束时间不晚于开始时间，自动调整
            self.dateTimeEdit_2.setDateTime(start_time.addSecs(3600))

    def _parse_int_list(self, text):
        """解析整数列表"""
        items = self._parse_list(text)
        result = []
        for item in items:
            try:
                # 先尝试转为浮点，再转整数，处理 "1.0" 这种情况
                result.append(int(float(item)))
            except (ValueError, TypeError):
                try:
                    result.append(int(item))
                except (ValueError, TypeError):
                    result.append(1)  # 默认值
        return result

    def get_config_dict(self):
        """获取配置字典"""
        return {
            "传感器灵敏度 (mV/nT)": self.doubleSpinBox.value(),
            "数据目录": self.lineEdit.text(),
            "数据文件列表": self.lineEdit_2.text(),
            "采样率 (Hz)": self.doubleSpinBox_2.value(),
            "通道数量": self.spinBox.value(),
            "通道索引": self._parse_int_list(self.lineEdit_3.text()),  # 使用新的整数解析方法
            "通道增益": self._parse_float_list(self.lineEdit_4.text()),
            "电长度 (米)": self._parse_float_list(self.lineEdit_5.text()),
            "开始时间": self.dateTimeEdit.dateTime().toString("yyyy-MM-dd HH:mm:ss"),
            "结束时间": self.dateTimeEdit_2.dateTime().toString("yyyy-MM-dd HH:mm:ss"),
            "FFT窗口长度": self.spinBox_3.value() if hasattr(self, 'spinBox_3') else self.spinBox_2.value(),
            "校准文件": self._parse_list(self.lineEdit_6.text())
        }

    def def_config(self):
        """设置默认配置 - 与主窗口参数完全对应"""
        default_config = {
            "传感器灵敏度 (mV/nT)": 1,
            "数据目录": "./data",
            "数据文件列表": "data_list.txt",
            "采样率 (Hz)": 1,
            "通道数量": 5,
            "通道索引": [1, 2, 3, 4, 5],
            "通道增益": [1.0, 1.0, 1.0, 1.0, 1.0],
            "电长度 (米)": [10.0, 10.0],
            # 修复时间格式
            "开始时间": QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss"),
            "结束时间": QDateTime.currentDateTime().addSecs(3600).toString("yyyy-MM-dd HH:mm:ss"),
            "FFT窗口长度": 1024,
            "校准文件": ["cal1.cmt", "cal2.cmt", "cal3.cmt"]
        }

        # 应用默认配置到UI
        self.set_config_dict(default_config)
        return default_config

    def set_config_dict(self, config_dict):
        """设置配置到UI控件"""
        if not config_dict:
            return

        # 传感器灵敏度
        if "传感器灵敏度 (mV/nT)" in config_dict:
            self.doubleSpinBox.setValue(config_dict["传感器灵敏度 (mV/nT)"])

        # 数据目录
        if "数据目录" in config_dict:
            self.lineEdit.setText(config_dict["数据目录"])

        # 数据文件列表
        if "数据文件列表" in config_dict:
            self.lineEdit_2.setText(config_dict["数据文件列表"])

        # 采样率
        if "采样率 (Hz)" in config_dict:
            self.doubleSpinBox_2.setValue(config_dict["采样率 (Hz)"])

        # 通道数量
        if "通道数量" in config_dict:
            self.spinBox.setValue(config_dict["通道数量"])

        # 通道索引
        if "通道索引" in config_dict:
            if isinstance(config_dict["通道索引"], list):
                self.lineEdit_3.setText(", ".join(map(str, config_dict["通道索引"])))
            else:
                self.lineEdit_3.setText(str(config_dict["通道索引"]))

        # 通道增益
        if "通道增益" in config_dict:
            if isinstance(config_dict["通道增益"], list):
                self.lineEdit_4.setText(", ".join(map(str, config_dict["通道增益"])))
            else:
                self.lineEdit_4.setText(str(config_dict["通道增益"]))

        # 电长度
        if "电长度 (米)" in config_dict:
            if isinstance(config_dict["电长度 (米)"], list):
                self.lineEdit_5.setText(", ".join(map(str, config_dict["电长度 (米)"])))
            else:
                self.lineEdit_5.setText(str(config_dict["电长度 (米)"]))

        # 开始时间 - 修复解析格式
        if "开始时间" in config_dict:
            if isinstance(config_dict["开始时间"], str):
                # 尝试多种格式解析时间
                time = None
                formats_to_try = [
                    "yyyy-MM-dd HH:mm:ss",
                    "yyyy-MM-dd hh:mm:ss",
                    "yyyy/MM/dd HH:mm:ss",
                    "yyyy/MM/dd hh:mm:ss"
                ]

                for fmt in formats_to_try:
                    time = QDateTime.fromString(config_dict["开始时间"], fmt)
                    if time.isValid():
                        break

                if time and time.isValid():
                    self.dateTimeEdit.setDateTime(time)
                else:
                    # 如果无法解析，使用当前时间
                    print(f"警告: 无法解析开始时间: {config_dict['开始时间']}")
                    self.dateTimeEdit.setDateTime(QDateTime.currentDateTime())

        # 结束时间 - 修复解析格式
        if "结束时间" in config_dict:
            if isinstance(config_dict["结束时间"], str):
                # 尝试多种格式解析时间
                time = None
                formats_to_try = [
                    "yyyy-MM-dd HH:mm:ss",
                    "yyyy-MM-dd hh:mm:ss",
                    "yyyy/MM/dd HH:mm:ss",
                    "yyyy/MM/dd hh:mm:ss"
                ]

                for fmt in formats_to_try:
                    time = QDateTime.fromString(config_dict["结束时间"], fmt)
                    if time.isValid():
                        break

                if time and time.isValid():
                    self.dateTimeEdit_2.setDateTime(time)
                else:
                    # 如果无法解析，使用当前时间+1小时
                    print(f"警告: 无法解析结束时间: {config_dict['结束时间']}")
                    self.dateTimeEdit_2.setDateTime(QDateTime.currentDateTime().addSecs(3600))

        # FFT窗口长度
        if "FFT窗口长度" in config_dict:
            # 检查是哪个SpinBox
            if hasattr(self, 'spinBox_3'):
                self.spinBox_3.setValue(config_dict["FFT窗口长度"])
            else:
                self.spinBox_2.setValue(config_dict["FFT窗口长度"])

        # 校准文件
        if "校准文件" in config_dict:
            if isinstance(config_dict["校准文件"], list):
                self.lineEdit_6.setText(", ".join(map(str, config_dict["校准文件"])))
            else:
                self.lineEdit_6.setText(str(config_dict["校准文件"]))

    def _parse_list(self, text):
        """解析逗号或空格分隔的列表"""
        if not text:
            return []
        return [item.strip() for item in text.replace(',', ' ').split() if item.strip()]

    def _parse_float_list(self, text):
        """解析浮点数列表"""
        items = self._parse_list(text)
        try:
            return [float(item) for item in items]
        except ValueError:
            return items


# 使用示例
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    # 创建对话框
    dialog = DialogWindow()

    # 设置默认配置
    dialog.def_config()

    # 显示对话框
    if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
        config = dialog.get_config_dict()
        print("配置信息:", config)
    else:
        print("用户取消")

    sys.exit(app.exec())
