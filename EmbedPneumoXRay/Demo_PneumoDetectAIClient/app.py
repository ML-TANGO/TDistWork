import sys
import socket
import argparse
import random
import pickle
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QPushButton
from PyQt5.QtWidgets import QListWidget, QListWidgetItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtCore import pyqtSignal as Signal, pyqtSlot as Slot


import torch

from typing import List

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

APP_DIR = Path(__file__).parent
WORK_DIR = Path(__file__).parent.parent
sys.path.append(str(APP_DIR))
sys.path.append(str(WORK_DIR))

from _ModelThread import ModelThread


def print_versions():
    """버전 스트링 출력"""
    import pkg_resources

    print("== VERION ======================")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyQt5: {pkg_resources.get_distribution('PyQt5').version}")
    print("=================================")


class ClientThread(QThread):
    __ip: str
    __port: int
    __socket: socket.socket

    errorLog = Signal(str)
    recvData = Signal(bytes)

    def __init__(self, ip: str, port: int = 8000):
        """클라이언트 스레드"""
        super().__init__()
        self.__ip = ip
        self.__port = port

    def run(self):
        """서버 접속"""
        while True:
            try:
                self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.__socket.connect((self.__ip, self.__port))
                print(f"connected to {self.__ip}:{self.__port}")
                break
            except Exception as e:
                print(f"connect error: {e}")
                self.sleep(1)
        return

    def send(self, data: bytes) -> None:
        """데이터 전송"""
        # self.start()
        # self.wait()
        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__socket.connect((self.__ip, self.__port))
        self.__socket.sendall(data)
        print(f"send data: ({len(data)} bytes) {data[:100]} Complete")

    def recv(self) -> bytes:
        """데이터 수신"""
        received_data = b""
        while True:
            data = self.__socket.recv(4096)
            if not data:
                break
            received_data += data
            if len(data) < 4096:
                break
        return received_data

    def close(self):
        """소켓 닫기"""
        self.__socket.close()


class AppMainWindow(QMainWindow):
    __data_count: int = 16
    __data: List[dict] = []
    __list_width: int = 700
    __server_ip: str = ""
    __server_port: int = 8000
    __client: ClientThread = None
    __model: ModelThread = None

    def __init__(self) -> None:
        """AI 연산 서버용 메인 윈도우"""
        super().__init__()
        self.__model = ModelThread()
        self._init_data()  # 데이터 설정
        self._init_ui()  # UI 설정

    def _init_data(self) -> None:
        """데이터 설정"""
        data_path = Path(__file__).parent.parent / "data"
        data: List[dict] = []
        for data_path in data_path.glob("*/*.jpeg"):
            sub_data = {
                "path": str(data_path.resolve()),
                "label": data_path.parent.name,
                "name": data_path.name,
            }
            data.append(sub_data)
        random.shuffle(data)
        self.__data = data[: self.__data_count]

    def _init_ui(self) -> None:
        """UI 설정"""
        self.setWindowTitle("Demo PneumoDetect AI Client")
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #333;
                color: #fff;
                font-size: 28px;
            }
            QListWidget {
                background-color: #000;
                color: #fff;
                font-size: 28px;
            }
            QListWidget::item {
                padding: 10px 5px;
            }
            QPushButton {
                font-size: 34px;
                background-color: #121212;
                color: #fff;
                padding: 10px 30px;
            }
            QLabel {
                font-size: 30px;
                color: #fff;
            }
            """
        )
        central_widget = QWidget(self)
        central_layout = QHBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)

        # 리스트 위젯 초기화
        listWidget = QListWidget(self)
        listWidget.itemClicked.connect(self.on_list_item_clicked)
        listWidget.setSpacing(5)
        listWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        listWidget.setMinimumWidth(self.__list_width)
        for data in self.__data:
            item_str = f'[{data.get("label", "None")}] {data.get("name", "File")}'
            item = QListWidgetItem()
            item.setData(0, item_str)
            item.setData(1, data["path"])
            item.setData(3, data["name"])
            item.setData(4, data["label"])
            listWidget.addItem(item)
        central_layout.addWidget(listWidget)

        # 이미지 정보 위젯 초기화
        screen_size = QApplication.primaryScreen().size()
        image_info_widget = QWidget(self)
        image_info_layout = QVBoxLayout(image_info_widget)
        image_info_layout.setSpacing(15)
        image_info_layout.setContentsMargins(5, 15, 5, 5)
        self.image_label = QLabel("", image_info_widget)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMaximumWidth(screen_size.width() - self.__list_width - 10)
        self.image_label.setMaximumHeight(screen_size.height() - 300)
        self.image_label.setScaledContents(True)
        self.image_path_label = QLabel("", image_info_widget)
        result_layout = QHBoxLayout()
        self.send_data_button = QPushButton("서버에 전송", image_info_widget)
        self.send_data_button.clicked.connect(self.on_send_data_button_clicked)
        self.send_data_button.hide()
        self.server_result_label = QLabel("", image_info_widget)
        self.server_error_label = QLabel("", image_info_widget)
        result_layout.addWidget(self.send_data_button, 0)
        result_layout.addWidget(self.server_result_label, 1)
        image_info_layout.addLayout(result_layout, 0)
        image_info_layout.addWidget(self.image_label, 1)
        image_info_layout.addWidget(self.image_path_label, 0)
        image_info_layout.addStretch(1)
        image_info_layout.addWidget(self.server_error_label, 0)
        central_layout.addWidget(image_info_widget, 1)

        self.setCentralWidget(central_widget)
        return

    def on_list_item_clicked(self, item: QListWidgetItem):
        """리스트 아이템 클릭 이벤트"""
        path = item.data(1)
        print(f"clicked: {path}")

        screen_size = QApplication.primaryScreen().size()
        pixmap = QPixmap(path)
        size = screen_size.width() - 550, screen_size.height()
        pixmap = pixmap.scaled(*size, aspectRatioMode=Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        self.image_path_label.setText(path)
        self.server_result_label.setText(
            f"[{self.__server_ip}:{self.__server_port}] 전송 대기 중"
        )

        self.send_data_button.show()

    def on_send_data_button_clicked(self):
        """서버에 데이터 전송"""
        self.server_result_label.setText("텐서 연산 중")
        path = self.image_path_label.text()
        self.server_result_label.update()
        result = self.__model(path, False)
        serialized_data = pickle.dumps(result.cpu())
        print(f"send tensor: ({len(serialized_data)} bytes) {serialized_data[:100]}")
        self.server_result_label.setText("텐서 전송 중")
        self.__client.send(serialized_data)
        self.server_result_label.setText("텐서 전송 완료")
        self.on_recv_data(self.__client.recv())
        self.__client.close()

    def on_recv_data(self, data: bytes) -> None:
        """서버에서 데이터 수신"""
        print(f"recv tensor: ({len(data)} bytes) {data[:100]}")
        result = pickle.loads(data)
        print(f"result: {result.shape} ({result})")
        result_txt = "정상" if result[0] == 0 else "폐렴"
        self.server_result_label.setText(f"결과: {result_txt}")

    def connect_server(self, ip: str, port: int = 8000) -> None:
        """AI 연산 서버 접속"""
        if ip == "":
            ip = socket.gethostbyname(socket.gethostname())
        if self.__client:
            self.__client.close()
        self.__client = ClientThread(ip, port)
        self.__client.errorLog.connect(self.log)
        self.__client.recvData.connect(self.on_recv_data)
        self.__server_ip = ip
        self.__server_port = port

    def log(self, msg: str) -> None:
        """로그 출력"""
        self.server_error_label.setText(msg)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.3.5")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    """메인 함수"""
    args = arg_parse()
    app = QApplication([])
    splash = QSplashScreen(QPixmap(str(APP_DIR / "splash.jpg")))
    splash.show()
    main_window = AppMainWindow()
    splash.finish(main_window)
    main_window.connect_server(args.ip, args.port)
    main_window.showFullScreen()
    sys.exit(app.exec_())


if __name__ == "__main__":
    print_versions()
    main()
