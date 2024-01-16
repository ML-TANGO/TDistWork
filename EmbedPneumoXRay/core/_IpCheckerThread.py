import socket

try:
    from PySide6.QtCore import QThread, Signal
except ImportError:
    from PyQt5.QtCore import QThread, pyqtSignal as Signal


class IpCheckerThread(QThread):
    __ip: str
    __port: int
    __running: bool
    __connection: bool

    connected = Signal(str)
    disconnected = Signal(str)

    def __init__(self, ip: str, port: int = 22):
        """설졍한 IP와 포트가 접속 가능한지 확인하는 스레드"""
        super().__init__()
        self.__ip = ip
        self.__port = port
        self.__running = False

    def start(self):
        """IP 체크 스레드 시작"""
        self.__running = True
        self.__connection = False
        super().start(QThread.LowPriority)

    def stop(self):
        """IP 체크 스레드 종료"""
        self.__running = False
        self.wait()
        self.terminate()

    def close(self):
        """IP 체크 스레드 종료"""
        self.stop()

    def run(self):
        """IP 체크 스레드 실행"""
        while self.__running:
            connection = False
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                sock.connect((self.__ip, self.__port))
                sock.close()
                connection = True
            except socket.error:
                connection = False
            if self.__connection != connection:
                if connection:
                    self.connected.emit(self.__ip)
                else:
                    self.disconnected.emit(self.__ip)
                self.__connection = connection
            self.sleep(1)

    @property
    def ip(self) -> str:
        """IP"""
        return self.__ip

    @property
    def connection(self) -> bool:
        """IP 접속 여부"""
        return self.__connection
