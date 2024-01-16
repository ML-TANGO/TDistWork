import sys
import socket
import threading
import pickle
from pathlib import Path
from typing import Dict, Any

from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWidgets import QWidget, QGroupBox, QLabel, QTextEdit, QPushButton
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout
from PySide6.QtWidgets import QStyleFactory
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt, QThread, Signal, QSize

import torch
import yaml

sys.path.append(Path(__file__).parent.parent.as_posix())

ROOT_DIR = Path(__file__).parent.parent

from core import SshClientThread
from core import IpCheckerThread

_FONT_SIZE = 18


class Config:
    __config_path: Path = Path(__file__).parent.parent / "config.yml"
    __config: Dict[str, Any] = {}

    @classmethod
    def init(cls):
        """설정 파일 로드"""
        with open(cls.__config_path, "r") as f:
            cls.__config = yaml.load(f, Loader=yaml.FullLoader)

    @property
    def ip_list(self) -> list:
        """접속 가능한 IP 목록"""
        return self.__config.get("ip_list", [])

    @property
    def username(self) -> str:
        """SSH 접속 계정"""
        return self.__config.get("username", "hanul")

    @property
    def password(self) -> str:
        """SSH 접속 비밀번호"""
        return self.__config.get("password", "hanul")

    @property
    def display(self) -> str:
        """SSH 실행 디스플레이"""
        return self.__config.get("display", ":0")

    @property
    def token(self) -> str:
        """Gitlab 토큰"""
        return self.__config.get("token", "")

    @property
    def repo(self) -> str:
        """Gitlab 레포지토리"""
        return self.__config.get("repository", "hanulsoft/ETRI-IITP-Medical-Demo")

    @property
    def server_ip(self) -> str:
        """AI 연산 서버 IP"""
        return socket.gethostbyname(socket.gethostname())

    @property
    def server_port(self) -> int:
        """AI 연산 서버 포트"""
        return self.__config.get("server_port", 8000)


class ServerThread(QThread):
    __server_socket: socket.socket = None
    __model: torch.nn.Module = None

    serverLog = Signal(str)

    def __init__(self) -> None:
        """AI 연산 서버"""
        super().__init__()

    def run(self):
        """AI 연산 서버 시작"""
        Config.init()
        weight_path = (
            Path(__file__).parent.parent / "model/ckpt_densenet201_partial_2.pt"
        )
        self.__model = torch.load(weight_path)
        self.__server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__server_socket.bind((Config().server_ip, Config().server_port))
        self.__server_socket.listen(5)
        self.serverLog.emit(f"서버 시작: {Config().server_ip}:{Config().server_port}")
        try:
            while True:
                client, addr = self.__server_socket.accept()
                thread = threading.Thread(
                    target=self.accept_client, args=(client, addr)
                )
                thread.daemon = True
                thread.start()
        except KeyboardInterrupt:
            self.serverLog.emit("서버 종료 명령 수신")
        finally:
            self.stop_server()

    def stop_server(self) -> None:
        """AI 연산 서버 종료"""
        self.__server_socket.close()
        self.serverLog.emit(f"서버 종료: {Config().server_ip}:{Config().server_port}")

    def accept_client(self, client: socket.socket, address: str) -> None:
        """클라이언트 접속 처리"""
        self.serverLog.emit(f"클라이언트 접속: {address}")
        # client.settimeout(3.0)
        data_len = len(pickle.dumps(torch.rand(1, 64, 64, 64)))
        try:
            received_data = b""
            while True:
                data = client.recv(4096)
                if not data:
                    break
                received_data += data
                if len(received_data) >= data_len:
                    break
        except socket.timeout:
            self.serverLog.emit(f"클라이언트 접속 대기 시간 초과: {address}")
            client.close()
            return
        except Exception as e:
            self.serverLog.emit(f"클라이언트 접속 오류: {address} {e}")
            client.close()
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor: torch.Tensor = pickle.loads(received_data)
        self.serverLog.emit(f"데이터 수신: {len(received_data):,} byte ({tensor.shape})")
        tensor = tensor.to(device)
        result = self.__model(tensor).argmax(dim=1).cpu()
        result = pickle.dumps(result)
        client.sendall(result)
        self.serverLog.emit(f"데이터 송신: {len(result)} byte")
        client.close()


class AppMainWindow(QMainWindow):
    __ip_checker_thread: Dict[str, IpCheckerThread] = {}
    __ip_ssh_client_thread: Dict[str, SshClientThread] = {}
    __ip_status_label: Dict[str, QLabel] = {}
    __start_client_button: Dict[str, QLabel] = {}
    __server: ServerThread = None

    def __init__(self) -> None:
        """AI 연산 서버용 메인 윈도우"""
        super().__init__()
        Config.init()
        self._setup_ui()
        self.__server = ServerThread()
        self.__server.serverLog.connect(self.log)
        self.__server.start()

    def _setup_ui(self):
        """UI 초기화"""
        self.setWindowTitle("Demo: PneumoDetect AI Server")
        self.setStyle(QStyleFactory.create("Fusion"))
        self.setStyleSheet(f"font-size: {_FONT_SIZE}px;")
        self.setMinimumSize(800, 600)

        main_layout = QVBoxLayout()
        main_widget = QWidget(self)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        device_status_group = QGroupBox("Device Status")
        device_status_layout = QHBoxLayout()
        for ip in Config().ip_list:
            style_sheet = f"border-radius: {int(_FONT_SIZE/2)}px;"
            style_sheet += "background-color: red;"
            ip_status_label = QLabel()
            ip_label = QLabel(ip)
            ip_status_label.setStyleSheet(style_sheet)
            ip_status_label.setFixedSize(_FONT_SIZE, _FONT_SIZE)

            self.__ip_status_label[ip] = ip_status_label
            device_status_layout.addWidget(ip_status_label, 0)
            device_status_layout.addWidget(ip_label, 1)

            ip_checker = IpCheckerThread(ip)
            ip_checker.connected.connect(self.on_ip_connected)
            ip_checker.disconnected.connect(self.on_ip_disconnected)
            ip_checker.start()
            self.__ip_checker_thread[ip] = ip_checker

            ssh_client = SshClientThread(ip, Config().username, Config().password)
            self.__ip_ssh_client_thread[ip] = ssh_client

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        policy = Qt.ScrollBarPolicy
        self.log_area.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.log_area.setVerticalScrollBarPolicy(policy.ScrollBarAlwaysOn)
        self.log_area.setHorizontalScrollBarPolicy(policy.ScrollBarAlwaysOff)
        main_layout.addWidget(self.log_area, 1)

        device_status_group.setLayout(device_status_layout)
        main_layout.addWidget(device_status_group)

    def log(self, text: str) -> None:
        """로그 출력"""
        self.log_area.append(text)

    def on_ip_connected(self, ip: str) -> None:
        """장비 접속 이벤트 핸들러"""
        style_sheet = f"border-radius: {int(_FONT_SIZE/2)}px;"
        style_sheet += "background-color: green;"
        self.__ip_status_label[ip].setStyleSheet(style_sheet)
        self.__ip_ssh_client_thread[ip].connect()

        repo_path = "/home/hanul/ETRI-IITP-Medical-Demo"
        ssh = self.__ip_ssh_client_thread[ip]
        if ssh.exist_file(repo_path):
            print(f"{ip} repo exist ETRI-IITP-Medical-Demo")
            ssh.command(f"cd {repo_path}")
            ssh.command("git pull")
            print(f"{ip} pull ETRI-IITP-Medical-Demo")
        else:
            print(f"{ip} repo not exist ETRI-IITP-Medical-Demo")
            print(f"{ip} clone ETRI-IITP-Medical-Demo")
            cmd = "git clone --depth 1"
            ssh.command(f"{cmd} https://{Config().token}@{Config().repo}")
            ssh.command(f"cd {repo_path}")
        ssh.command(f"export DISPLAY={Config().display}")
        print(f"{ip} run ETRI-IITP-Medical-Demo")
        ssh.command(
            "python3 Demo_PneumoDetectAIClient/app.py"
            f" --ip {Config().server_ip}"
            f" --port {Config().server_port}"
        )

    def on_ip_disconnected(self, ip: str) -> None:
        """장비 접속 해제 이벤트 핸들러"""
        style_sheet = f"border-radius: {int(_FONT_SIZE/2)}px;"
        style_sheet += "background-color: red;"
        self.__ip_status_label[ip].setStyleSheet(style_sheet)
        self.__ip_ssh_client_thread[ip].close()

    def closeEvent(self, event) -> None:
        for ip in self.__ip_checker_thread:
            self.__ip_checker_thread[ip].close()
        for ip in self.__ip_ssh_client_thread:
            self.__ip_ssh_client_thread[ip].close()
        self._stop_server()
        return super().closeEvent(event)


def main():
    app = QApplication([])
    app_icon = QIcon()
    icon_pixmap_16 = QIcon((ROOT_DIR / "icon.png").as_posix()).pixmap(16, 16)
    icon_pixmap_32 = QIcon((ROOT_DIR / "icon.png").as_posix()).pixmap(32, 32)
    icon_pixmap_48 = QIcon((ROOT_DIR / "icon.png").as_posix()).pixmap(48, 48)
    icon_pixmap_256 = QIcon((ROOT_DIR / "icon.png").as_posix()).pixmap(256, 256)
    app_icon.addPixmap(icon_pixmap_16)
    app_icon.addPixmap(icon_pixmap_32)
    app_icon.addPixmap(icon_pixmap_48)
    app_icon.addPixmap(icon_pixmap_256)
    app.setWindowIcon(app_icon)
    window = AppMainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
