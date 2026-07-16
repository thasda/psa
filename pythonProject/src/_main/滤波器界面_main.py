import sys
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import Qt
from pythonProject.src.interface.滤波器界面 import Ui_Form


class filterWindow(QtWidgets.QMainWindow, Ui_Form):
    def __init__(self):
        super().__init__()
        self.drag_start_window = None
        self.drag_start_global = None
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)  # 去掉默认窗口边框
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)  # 背景透明
        self.setupUi(self)
        self.connections()

    def connections(self):
        # 连接动作
        self.pushButton_2.clicked.connect(self.minimize_window)  # 最小化窗口
        self.pushButton.clicked.connect(self.close_window)  # 关闭窗口

    def close_window(self):
        self.close()

    def minimize_window(self):
        """窗口最小化函数"""
        self.showMinimized()

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



