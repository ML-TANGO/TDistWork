import os
import time
from pathlib import Path

import pyqtgraph
from PyQt5 import QtWidgets, QtGui, QtCore

from src.widgets import WrapperWidget, ProgressModal
from src.stylesheet import QSS
from src.model_thread import ModelThread

os.chdir(os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("DISPLAY", "") == "":
    os.environ.__setitem__("DISPLAY", ":0")


class MainWindow(QtWidgets.QMainWindow):
    progress_modal: ProgressModal
    wrapper_widget: WrapperWidget

    __warmup_flops_arr: list
    __warmup_flops_data: list
    __warmup_loop_time: float
    __infer_flops_arr: list
    __infer_flops_data: list
    __infer_loop_time: float

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hanulsoft Test App")
        self.model_thread = ModelThread(self)
        self.__calc_flops_arr = []
        self.__calc_section_flops_arr = []

        self.progress_modal = ProgressModal()
        self.wrapper_widget = WrapperWidget(self)
        self.setCentralWidget(self.wrapper_widget)

        QtCore.QMetaObject.connectSlotsByName(self)

    @QtCore.pyqtSlot()
    def on_ModelThread_modelReady(self):
        QtCore.QTimer.singleShot(0, self.progress_modal.close)
        self.wrapper_widget.setEnabled(True)
        self.wrapper_widget.btn_start.setEnabled(False)
        self.wrapper_widget.label_model_flop.setText(
            f"{self.model_thread.flop / 10**9:.4f} GFLOPs"
        )
        font = self.wrapper_widget.label_model_flop.font()
        font.setPointSize(16)
        axis = pyqtgraph.AxisItem(orientation="bottom")
        axis.setLabel(text="Time (s)", units=None, unitPrefix=None)
        self.wrapper_widget.plot_widget.getPlotItem().setAxisItems({"bottom": axis})
        self.wrapper_widget.plot_widget.setLabel("left", "TFLOPs")
        self.wrapper_widget.plot_widget.showGrid(x=True, y=True)
        self.__curve = self.wrapper_widget.plot_widget.plot(
            [0.0], symbol="o", symbolSize=5, symbolBrush=("r")
        )
        self.__calc_flops_arr = []
        self.__calc_section_flops_arr = []

    @QtCore.pyqtSlot(float)
    def on_ModelThread_modelWarmup(self, gflops):
        self.__warmup_flops_arr.append(gflops / 10**3)
        if self.__warmup_loop_time == 0:
            self.__warmup_loop_time = time.time()
        if time.time() - self.__warmup_loop_time > 1:
            self.__warmup_flops_data.append(
                sum(self.__warmup_flops_arr) / len(self.__warmup_flops_arr)
            )
            self.__curve.setData([0] + self.__warmup_flops_data)
            self.__warmup_loop_time = time.time()
            self.__warmup_flops_arr = []
            self.wrapper_widget.working_progress.setValue(
                int(len(self.__warmup_flops_data) / 10 * 100)
            )
        if len(self.__warmup_flops_data) >= 10:
            self.model_thread.warmup_stop()
        return

    @QtCore.pyqtSlot()
    def on_ModelThread_modelWarmupDone(self):
        print("Warmup Done")
        self.wrapper_widget.btn_start.setEnabled(True)
        return

    @QtCore.pyqtSlot(float)
    def on_ModelThread_modelInference(self, gflops):
        self.__infer_flops_arr.append(gflops / 10**3)
        if self.__infer_loop_time == 0:
            self.__infer_loop_time = time.time()
        if time.time() - self.__infer_loop_time > 1:
            self.__infer_flops_data.append(
                sum(self.__infer_flops_arr) / len(self.__infer_flops_arr)
            )
            self.__curve.setData(self.__infer_flops_data)
            self.__infer_loop_time = time.time()
            self.__infer_flops_arr = []
            self.wrapper_widget.working_progress.setValue(
                int(len(self.__infer_flops_data) / 60 * 100)
            )
        if len(self.__infer_flops_data) >= 60:
            self.model_thread.inference_stop()
        aver_flops = sum(self.__infer_flops_data[1:]) / (
            len(self.__infer_flops_data) - 1
        )
        self.wrapper_widget.label_model_flops.setText(f"{aver_flops:.4f} TFLOPs")

    @QtCore.pyqtSlot()
    def on_ModelThread_modelInferenceDone(self):
        self.wrapper_widget.btn_start.setEnabled(True)
        self.wrapper_widget.working_progress.setValue(100)

    @QtCore.pyqtSlot(int, int)
    def on_ModelThread_modelStatus(self, work, total):
        self.wrapper_widget.model_workig_status.setText(f"{work: 5}/{total}")
        self.wrapper_widget.working_progress.setValue(int(work * 100 / total))

    @QtCore.pyqtSlot()
    def on_btnWarmup_clicked(self):
        print("Warmup button clicked")
        self.model_thread.start_model_warmup()
        # self.wrapper_widget.plot_widget.setYRange(min=0)
        self.wrapper_widget.plot_widget.setXRange(min=1, max=10)
        self.wrapper_widget.btn_start.setEnabled(False)
        self.wrapper_widget.btn_warmup.setEnabled(False)
        self.__warmup_flops_arr = []
        self.__warmup_flops_data = []
        self.__warmup_loop_time = 0

    @QtCore.pyqtSlot()
    def on_btnStart_clicked(self):
        print("Start button clicked")
        # self.wrapper_widget.plot_widget.setYRange(min=0)
        self.wrapper_widget.plot_widget.setXRange(min=1, max=60)
        self.model_thread.start_model_inference()
        self.wrapper_widget.btn_start.setEnabled(False)
        self.wrapper_widget.working_progress.setValue(0)
        self.__infer_flops_arr = []
        self.__infer_flops_data = []
        self.__infer_loop_time = 0

    @QtCore.pyqtSlot()
    def on_btnQuit_clicked(self):
        print("Quit button clicked")
        if (
            QtWidgets.QMessageBox.question(
                self,
                "Quit",
                "Are you sure to quit?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            == QtWidgets.QMessageBox.No
        ):
            return
        if self.model_thread.isRunning():
            self.model_thread.quit()
            self.model_thread.wait()
        self.close()

    def showEvent(self, a0: QtGui.QShowEvent) -> None:
        def start_app():
            self.wrapper_widget.setEnabled(False)
            self.progress_modal.show()
            self.model_thread.start_model_init()

        QtCore.QTimer.singleShot(0, start_app)
        return super().showEvent(a0)


if __name__ == "__main__":
    workspace = Path(__file__).parent
    os.chdir(workspace)
    app = QtWidgets.QApplication([])
    screen_resolution = app.desktop().screenGeometry()
    screen_size = QtCore.QSize(screen_resolution.width(), screen_resolution.height())

    splash_pix = QtGui.QPixmap("splash.jpg").scaled(
        screen_size,
        QtCore.Qt.KeepAspectRatio,
    )
    splash = QtWidgets.QSplashScreen(splash_pix)
    splash.show()
    app.processEvents()

    window = MainWindow()
    splash.finish(window)
    window.showFullScreen()

    screen_width = min(screen_resolution.width(), screen_resolution.height())
    text_size = screen_width * 0.04
    app.setStyleSheet(
        QSS
        + "* {"
        + f"font-size: {int(text_size)}px;"
        + "} QPushButton, QLabel, QWidget {"
        + f"padding: {int(text_size * 0.2)}px {int(text_size * 0.5)}px;"
        + "} QLabel {"
        + f"font-size: {int(text_size * 0.6)}px;"
        + "}"
    )

    app.exec_()
