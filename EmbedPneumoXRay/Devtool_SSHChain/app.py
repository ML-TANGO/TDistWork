import sys
from pathlib import Path
from typing import Tuple

from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QWidget, QGroupBox
from PySide6.QtWidgets import QLabel, QTextEdit, QLineEdit
from PySide6.QtWidgets import QToolButton
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout
from PySide6.QtWidgets import QStyleFactory, QStyle
from PySide6.QtCore import Qt, Slot
from PySide6.QtCore import QTimer

import yaml
from ansi2html import Ansi2HTMLConverter

sys.path.append(Path(__file__).parent.parent.as_posix())
from core import SshClientThread
from core import IpCheckerThread


def print_versions():
    """버전 스트링 출력"""
    import pkg_resources

    print("== VERION ======================")
    print(f"Python: {sys.version}")
    print(f"PySide6: {pkg_resources.get_distribution('PySide6').version}")
    print(f"PyYAML: {pkg_resources.get_distribution('PyYAML').version}")
    print(f"Paramiko: {pkg_resources.get_distribution('paramiko').version}")
    print(f"ansi2html: {pkg_resources.get_distribution('ansi2html').version}")
    print("=================================")


class SshClientTextArea(QTextEdit):
    __scroll_lock: bool = True
    __prev_output: str = ""
    __ssh_thread: SshClientThread
    __update_timer: QTimer
    __ansi_converter: Ansi2HTMLConverter

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.setContextMenuPolicy(Qt.NoContextMenu)
        self.setAcceptRichText(False)
        self.setUndoRedoEnabled(False)
        self.setStyleSheet("background-color: black; color: white;")

        self.__update_timer = QTimer()
        self.__update_timer.setInterval(100)
        self.__update_timer.timeout.connect(self.on_update_output)
        self.__update_timer.start()

        self.__ansi_converter = Ansi2HTMLConverter()

    def setSshTread(self, ssh_thread: SshClientThread):
        """SSH 스레드 설정"""
        self.__ssh_thread = ssh_thread

    def setScrollLock(self, lock: bool):
        """스크롤 락 설정"""
        self.__scroll_lock = lock

    @Slot()
    def on_update_output(self):
        if not self.__ssh_thread:
            return
        self.__ssh_thread._update_output()
        output = self.__ssh_thread.output
        if self.__prev_output == output:
            return
        self.__prev_output = output
        html = self.__ansi_converter.convert(output)
        prev_scroll_pos = self.verticalScrollBar().value()
        self.setHtml(html)
        if self.__scroll_lock:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        else:
            self.verticalScrollBar().setValue(prev_scroll_pos)


class AppMainWindow(QMainWindow):
    __config_path: Path = Path(__file__).parent.parent / "config.yml"
    __config: dict
    __ip_checkers: dict = {}
    __ssh_clients: dict = {}
    __device_widgets: dict = {}

    def __init__(self):
        super().__init__()
        self.__config = self._load_config()
        self._init_ui()

    def _load_config(self):
        """설정 파일 로드"""
        with open(self.__config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def _init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("SSH Chain Monitor")
        self.setMinimumSize(800, 600)
        self.setStyle(QStyleFactory.create("Fusion"))

        main_layout = QVBoxLayout()
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        device_layout = QHBoxLayout()
        for ip in self.ip_list:
            # 장비 접속 체크 스레드
            ip_checker = IpCheckerThread(ip)
            ip_checker.connected.connect(lambda ip: self.on_connected(ip))
            ip_checker.disconnected.connect(lambda ip: self.on_disconnected(ip))
            ip_checker.start()

            # 장비 SSH 접속 스레드
            ssh_clients = SshClientThread(ip, self.username, self.password)
            device_widget, device_items = self._init_device_widget(ip)
            device_layout.addWidget(device_widget, 1)
            device_items["output"].setSshTread(ssh_clients)

            self.__ssh_clients[ip] = ssh_clients
            self.__device_widgets[ip] = device_items
            self.__ip_checkers[ip] = ip_checker
        main_layout.addLayout(device_layout)

        cmd_layout = QHBoxLayout()
        self.cmd_input = QLineEdit()
        self.cmd_input.returnPressed.connect(self.run_command)
        self.cmd_input.setPlaceholderText("SSH Command Input")
        cmd_layout.addWidget(self.cmd_input)
        send_button = QToolButton()
        send_button.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        send_button.setText("Send File")
        send_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        send_button.clicked.connect(self.send_file)
        cmd_layout.addWidget(send_button)
        main_layout.addLayout(cmd_layout)

    def _init_device_widget(self, ip: str) -> Tuple[QGroupBox, dict]:
        """장비 위젯 및 서브위젯 출력"""
        device_items = {}
        device_layout = QVBoxLayout()
        device_widget = QGroupBox(ip)
        device_widget.setLayout(device_layout)

        status_layout = QHBoxLayout()
        label = QLabel("SSH Status:")
        status = QLabel("Disconnected")
        device_items["status"] = status
        lock_button = QToolButton()
        lock_button.setCheckable(True)
        lock_button.setChecked(True)
        lock_button.setIcon(self.style().standardIcon(QStyle.SP_ArrowDown))
        lock_button.toggled.connect(
            lambda checked: device_items["output"].setScrollLock(checked)
        )
        status_layout.addWidget(label)
        status_layout.addWidget(status, 1)
        status_layout.addWidget(lock_button)
        device_layout.addLayout(status_layout)

        output = SshClientTextArea()
        device_items["output"] = output
        device_layout.addWidget(output, 1)

        return device_widget, device_items

    def on_connected(self, ip: str) -> None:
        """장비 접속시 호출"""
        self.__device_widgets[ip]["status"].setText("Connected")
        if not self.__ssh_clients[ip].connected:
            self.__ssh_clients[ip].connect()
        print(f"{ip} connected")

    def on_disconnected(self, ip: str) -> None:
        """장비 접속 해제시 호출"""
        self.__device_widgets[ip]["status"].setText("Disconnected")
        self.__ssh_clients[ip].close()
        print(f"{ip} disconnected")

    def run_command(self) -> None:
        """SSH 명령어 실행"""
        for ip in self.ip_list:
            if not self.__ssh_clients[ip].connected:
                continue
            self.__ssh_clients[ip].command(self.cmd_input.text())
        self.cmd_input.setText("")

    def send_file(self) -> None:
        """파일 전송"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Send File")
        if not file_path:
            return
        for ip in self.ip_list:
            if not self.__ssh_clients[ip].connected:
                continue
            self.__ssh_clients[ip].send_file(file_path)

    @property
    def ip_list(self) -> list:
        return self.__config.get("ip_list", [])

    @property
    def username(self) -> str:
        return self.__config.get("username", "hanul")

    @property
    def password(self) -> str:
        return self.__config.get("password", "hanul")


def main():
    app = QApplication(sys.argv)
    window = AppMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    print_versions()
    main()
