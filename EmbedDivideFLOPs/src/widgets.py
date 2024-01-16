from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph


class LetterboxLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setBrush(QtGui.QColor(0, 0, 0))
        painter.drawRect(self.rect())

        if self.pixmap() is None:
            return super().paintEvent(event)

        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        w, h = self.width() - 2, self.height() - 2
        pw, ph = self.pixmap().width(), self.pixmap().height()
        if pw == 0 or ph == 0:
            painter.end()
            return super().paintEvent(event)
        label_aspect = w / h
        pixmap_aspect = pw / ph
        if label_aspect > pixmap_aspect:
            scale = h / ph
            offset = int((w - pw * scale) / 2)
            painter.drawPixmap(offset, 0, self.pixmap().scaledToHeight(h))
        else:
            scale = w / pw
            offset = int((h - ph * scale) / 2)
            painter.drawPixmap(0, offset, self.pixmap().scaledToWidth(w))
        painter.end()


class ProgressModal(QtWidgets.QWidget):
    label: QtWidgets.QLabel
    progress: QtWidgets.QProgressBar

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint
        )
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setObjectName("ModalProgress")

        layout = QtWidgets.QVBoxLayout()
        self.label = QtWidgets.QLabel("Loading...")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)

        layout.addWidget(self.label)
        layout.addWidget(self.progress)
        sub_widget = QtWidgets.QWidget()
        sub_widget.setLayout(layout)
        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().addWidget(sub_widget)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setContentsMargins(0, 0, 0, 0)

    def showEvent(self, a0: QtGui.QShowEvent) -> None:
        screen_resolution = QtWidgets.QApplication.desktop().screenGeometry()
        screen_width = screen_resolution.width()
        self.setMinimumWidth(int(screen_width * 2 / 3))
        screen_size = QtWidgets.QApplication.desktop().screenGeometry()
        widget_size = self.size()
        self.move(
            int((screen_size.width() - widget_size.width()) / 2),
            int((screen_size.height() - widget_size.height()) / 2),
        )
        return super().showEvent(a0)


class WrapperWidget(QtWidgets.QWidget):
    btn_warmup: QtWidgets.QPushButton
    btn_start: QtWidgets.QPushButton
    btn_quit: QtWidgets.QPushButton
    plot_widget: pyqtgraph.PlotWidget
    label_model_flop: QtWidgets.QLabel
    label_model_flops: QtWidgets.QLabel
    working_status: QtWidgets.QLabel
    working_progress: QtWidgets.QProgressBar

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setObjectName("WrapperWidget")
        self.init_ui()
        return

    def init_ui(self) -> None:
        screen_resolution = QtWidgets.QApplication.desktop().screenGeometry()
        screen_width = screen_resolution.width()

        layout = QtWidgets.QHBoxLayout()
        layout_menu = QtWidgets.QVBoxLayout()
        self.btn_warmup = QtWidgets.QPushButton("GPU Warmup")
        self.btn_start = QtWidgets.QPushButton("Test Start")
        self.btn_quit = QtWidgets.QPushButton("Quit")
        self.btn_warmup.setMinimumWidth(int(screen_width * 2 / 5))
        self.btn_warmup.setObjectName("btnWarmup")
        self.btn_start.setObjectName("btnStart")
        self.btn_quit.setObjectName("btnQuit")

        layout_stats = QtWidgets.QFormLayout()
        self.label_model_flop = QtWidgets.QLabel("-")
        self.label_model_flops = QtWidgets.QLabel("-")

        layout_viewer = QtWidgets.QVBoxLayout()
        self.plot_widget = pyqtgraph.PlotWidget(self)
        self.working_status = QtWidgets.QLabel("-")
        self.working_progress = QtWidgets.QProgressBar(self)

        self.working_status.setAlignment(QtCore.Qt.AlignCenter)

        def line_item() -> QtWidgets.QFrame:
            line = QtWidgets.QFrame()
            line.setFrameShape(QtWidgets.QFrame.HLine)
            line.setFrameShadow(QtWidgets.QFrame.Sunken)
            return line

        # layout_stats.addRow("Model FLOP", self.label_model_flop)
        layout_stats.addRow("Average FLOPs", self.label_model_flops)

        layout_menu.addWidget(self.btn_warmup)
        layout_menu.addWidget(self.btn_start)
        layout_menu.addWidget(line_item())
        layout_menu.addLayout(layout_stats)
        layout_menu.addWidget(line_item())
        layout_menu.addStretch()
        layout_menu.addWidget(self.btn_quit)

        layout_viewer.addWidget(self.plot_widget)
        layout_viewer.addWidget(self.working_status)
        layout_viewer.addWidget(self.working_progress)

        layout.addLayout(layout_menu)
        layout.addLayout(layout_viewer)
        self.setLayout(layout)
        return

    def showEvent(self, a0: QtGui.QShowEvent) -> None:
        super().showEvent(a0)
