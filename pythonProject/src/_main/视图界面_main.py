import sys
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt
from pythonProject.src.interface.视图界面 import Ui_widget  # 假设你的UI文件已正确生成


class viewWindow(QtWidgets.QMainWindow, Ui_widget):
    def __init__(self):
        super().__init__()
        self.drag_start_window = None
        self.drag_start_global = None
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)  # 去掉默认窗口边框
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)  # 背景透明
        self.setupUi(self)
        # 固定窗口大小（禁止调整）
        self.setFixedSize(self.size())
        # 初始化缩放相关参数
        self.current_scale = 1.0  # 初始缩放比例
        self.min_scale = 0.5  # 最小缩放比例（50%）
        self.max_scale = 2.0  # 最大缩放比例（200%）
        self.connections()

    def connections(self):
        # 连接动作
        self.pushButton.clicked.connect(self.minimize_window)  # 最小化窗口
        self.pushButton_2.clicked.connect(self.close_window)  # 关闭窗口
        self.radioButton.clicked.connect(self.do_something_for_option1)  # 勾选场景
        self.radioButton_2.clicked.connect(self.do_something_for_option2)  # 勾选场景
        # 连接垂直滑块与缩放函数
        self.verticalSlider.setMinimum(int(self.min_scale * 50))  # 最小值50（对应50%）
        self.verticalSlider.setMaximum(int(self.max_scale * 200))  # 最大值200（对应200%）
        self.verticalSlider.setValue(int(self.current_scale * 100))  # 初始值100（对应100%）
        self.verticalSlider.valueChanged.connect(self.on_slider_changed)  # 滑块变化时触发缩放

    def close_window(self):
        self.close()

    def do_something_for_option1(self):
        self.show_shiyitu(r"F:\Anylysis_project\pythonProject\logs\figure\场景示意图\地上测量场景模型.png")
        self.textEdit.setText(r"F:\Anylysis_project\pythonProject\logs\figure\场景示意图\地上测量场景模型")

    def do_something_for_option2(self):
        self.show_shiyitu(r"F:\Anylysis_project\pythonProject\logs\figure\场景示意图\地下测量场景模型")
        self.textEdit.setText(r"F:\Anylysis_project\pythonProject\logs\figure\场景示意图\地下测量场景模型")

    def show_shiyitu(self, image_file):
        # 显示图像
        scene = QtWidgets.QGraphicsScene()
        pixmap = QtGui.QPixmap(image_file)
        if pixmap.isNull():
            # 处理图片加载失败的情况
            scene.addText(f"无法加载图片: {image_file}")
        else:
            scene.addPixmap(pixmap)
        self.graphicsView.setScene(scene)
        # 应用当前缩放比例
        self.graphicsView.fitInView(
            scene.sceneRect(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio
        )

    def on_slider_changed(self, value):
        """处理垂直滑块变化，实现图片缩放"""
        # 将滑块值（50-200）转换为缩放比例（0.5-2.0）
        self.current_scale = value / 100.0

        # 获取当前场景
        scene = self.graphicsView.scene()
        if scene:
            # 缩放视图（保持 aspect ratio）
            self.graphicsView.resetTransform()  # 重置之前的变换
            self.graphicsView.scale(self.current_scale, self.current_scale)  # 应用新缩放
            # 确保图片居中显示
            self.graphicsView.centerOn(scene.sceneRect().center())

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

            # 边界检查（防止窗口被拖出屏幕）
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


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    view_window = viewWindow()
    view_window.show()
    sys.exit(app.exec())
