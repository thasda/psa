import json
import os
import traceback
from datetime import datetime
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import Qt, QDateTime
from PyQt6.QtWidgets import QFileDialog
from pythonProject.src.interface.工程文件配置界面 import Ui_widget


class eigneerWindow(QtWidgets.QMainWindow, Ui_widget):
    def __init__(self):
        super().__init__()
        # 关键：设置无边框和透明背景
        self.drag_start_window = None
        self.drag_start_global = None
        self.status_bar = self.statusBar()
        self.project_data = None
        self.current_project_path = None
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)  # 去掉默认窗口边框
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)  # 背景透明
        self.setupUi(self)
        self.connections()

        # 固定窗口大小（禁止调整）
        self.setFixedSize(self.size())

    def connections(self):
        # 连接动作
        self.pushButton_4.clicked.connect(self.minimize_window)  # 最小化窗口
        self.pushButton_3.clicked.connect(self.close_window)  # 关闭窗口
        self.pushButton.clicked.connect(self._browse_data_dir)  # 选择数据文件路径
        self.pushButton_2.clicked.connect(self.save_project_file)  # 保存工程文件
        self.pushButton_5.clicked.connect(self.read_project_file)  # 打开工程文件
        self.pushButton_6.clicked.connect(self.save_project_as)  # 另存为新的工程文件

    def close_window(self):
        self.close()

    def minimize_window(self):
        """窗口最小化函数"""
        self.showMinimized()

    def read_project_file(self):
        """打开工程文件"""
        project_file, _ = QFileDialog.getOpenFileName(
            self, "选择工程文件", "", "工程文件 (*.dataProj *.json);;所有文件 (*)"
        )
        if not project_file:
            return

        try:
            with open(project_file, 'r', encoding='utf-8') as f:
                self.project_data = json.load(f)

            # 更新界面
            self._sync_project_to_ui()
            self.status_bar.showMessage(f"已打开工程：{os.path.basename(project_file)}")
            self.current_project_path = project_file

        except json.JSONDecodeError:
            self.status_bar.showMessage("打开工程失败：文件格式错误")

        except Exception as e:
            self.status_bar.showMessage(f"打开工程失败：{str(e)}")
            traceback.print_exc()

    def save_project_file(self):
        """保存工程文件（若已存在则覆盖，否则调用另存为）"""
        if not self.current_project_path:
            self.save_project_as()
            return

        try:
            project_data = self._collect_project_info(self.current_project_path)

            with open(self.current_project_path, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, ensure_ascii=False, indent=4)

            self.project_data = project_data
            self.status_bar.showMessage(f"工程已保存：{os.path.basename(self.current_project_path)}")

        except Exception as e:
            self.status_bar.showMessage(f"保存工程失败：{str(e)}")
            traceback.print_exc()

    def save_project_as(self):
        """另存为新工程文件"""
        default_dir = os.path.dirname(self.current_project_path) if self.current_project_path else ""
        project_file, _ = QFileDialog.getSaveFileName(
            self, "保存工程文件", default_dir, "工程文件 (*.dataProj);;JSON文件 (*.json);;所有文件 (*)"
        )
        if not project_file:
            return

        # 确保文件后缀
        if not project_file.endswith(('.dataProj', '.json')):
            project_file += '.dataProj'

        self.current_project_path = project_file
        self.save_project_file()  # 调用保存逻辑

    def _collect_project_info(self, project_path):
        """收集工程信息（从界面到数据）"""
        project_dir = os.path.dirname(project_path)

        # 基础信息
        project_info = {
            "project_name": self.lineEdit_3.text(),
            "create_time": self.project_data["create_time"] if self.project_data else datetime.now().strftime(
                "%Y-%m-%dT%H:%M:%S"),
            "last_modified": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "description": self.plainTextEdit.toPlainText(),
            "version": "1.0"
        }

        # 配置信息
        config_info = {
            "data_directory": self._get_relative_path(self.lineEdit.text(), project_dir),
            "data_list_file": self.lineEdit_2.text(),
            "sample_rate": self.doubleSpinBox.value(),
            "measurement_scenarios": self.plainTextEdit_2.toPlainText(),
            "start_time": self.dateTimeEdit.dateTime().toString("yyyy-MM-dd HH:mm:ss"),
            "end_time": self.dateTimeEdit_3.dateTime().toString("yyyy-MM-dd HH:mm:ss")
        }

        return {
            **project_info,
            "config": config_info
        }

    def _sync_project_to_ui(self):
        """将工程数据同步到界面"""
        if not self.project_data:
            return

        # 工程信息
        self.lineEdit_3.setText(self.project_data.get("project_name", ""))
        self.dateTimeEdit_4.setDateTime(QDateTime.fromString(self.project_data.get("create_time", ""),
                                                             "yyyy-MM-ddTHH:mm:ss"))
        self.dateTimeEdit_5.setDateTime(QDateTime.fromString(self.project_data.get("last_modified", ""),
                                                             "yyyy-MM-ddTHH:mm:ss"))
        self.plainTextEdit.setPlainText(self.project_data.get("description", ""))

        # 配置参数
        config = self.project_data.get("config", {})
        self.lineEdit.setText(config.get("data_directory", ""))
        self.lineEdit_2.setText(config.get("data_list_file", ""))
        self.plainTextEdit_2.setPlainText(config.get("measurement_scenarios", ""))
        self.doubleSpinBox.setValue(float(config.get("sample_rate", 1000.0)))

        # 起始时间
        start_time = config.get("start_time")
        if start_time:
            self.dateTimeEdit.setDateTime(QDateTime.fromString(start_time, "yyyy-MM-dd HH:mm:ss"))

        end_time = config.get("end_time")
        if end_time:
            self.dateTimeEdit_3.setDateTime(QDateTime.fromString(end_time, "yyyy-MM-dd HH:mm:ss"))

    def _browse_data_dir(self):
        """浏览选择数据目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择数据目录", self.lineEdit.text() or os.getcwd()
        )
        if dir_path:
            self.lineEdit.setText(dir_path)

    @staticmethod
    def _get_relative_path(target_path, base_dir):
        """获取相对路径，增加调试信息"""
        if not target_path or not base_dir:
            return target_path

        try:
            # 打印原始路径用于调试
            print(f"原始 target_path: {target_path}")
            print(f"原始 base_dir: {base_dir}")

            # 规范化路径（添加尾部斜杠，确保路径格式一致）
            if not base_dir.endswith(os.path.sep):
                base_dir += os.path.sep

            # 转换为绝对路径
            target_abs = os.path.abspath(target_path)
            base_abs = os.path.abspath(base_dir)

            # 打印规范化后的路径
            print(f"规范化 target_abs: {target_abs}")
            print(f"规范化 base_abs: {base_abs}")

            # 计算相对路径
            relative_path = os.path.relpath(target_abs, base_abs)

            # 打印计算结果
            print(f"计算结果: {relative_path}")

            return relative_path
        except ValueError as e:
            print(f"路径错误: {e}")
            return target_path  # 跨盘符或其他错误时返回绝对路径
        except Exception as e:
            print(f"未知错误: {e}")
            return target_path

    def mousePressEvent(self, event):
        # 记录鼠标按下时的全局位置和窗口当前位置
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_global = event.globalPosition().toPoint()
            self.drag_start_window = self.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        # 只处理左键拖动，并且确保已经记录了拖动起点
        if (event.buttons() & Qt.MouseButton.LeftButton and
                self.drag_start_global is not None and
                self.drag_start_window is not None):
            # 计算鼠标位移增量
            current_global = event.globalPosition().toPoint()
            delta = current_global - self.drag_start_global

            # 根据位移增量计算新的窗口位置
            new_position = self.drag_start_window + delta

            # 边界检查（可选，防止窗口被拖出屏幕）
            screen_geometry = QtWidgets.QApplication.primaryScreen().geometry()
            new_position.setX(max(0, min(new_position.x(), screen_geometry.width() - self.width())))
            new_position.setY(max(0, min(new_position.y(), screen_geometry.height() - self.height())))

            # 移动窗口
            self.move(new_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        # 清理拖动状态
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_global = None
            self.drag_start_window = None
            event.accept()


# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     main_window = eigneerWindow()
#     main_window.show()
#
#     sys.exit(app.exec())
