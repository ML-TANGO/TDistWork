import re
from pathlib import Path

from PySide6.QtCore import QThread
from paramiko import SSHClient, AutoAddPolicy, Channel


class SshClientThread(QThread):
    __ip: str
    __username: str
    __password: str
    __client: SSHClient
    __channel: Channel
    __output: str

    def __init__(self, ip: str, username: str, password: str):
        """SSH 접속을 위한 스레드"""
        super().__init__()
        self.__ip = ip
        self.__username = username
        self.__password = password
        self.__output = ""
        self.__channel = None
        self.__client = SSHClient()
        self.__client.load_system_host_keys()
        self.__client.set_missing_host_key_policy(AutoAddPolicy())

    def connect(self):
        """SSH 접속"""
        if self.__client.get_transport():
            return
        self.__client.connect(self.ip, username=self.username, password=self.password)
        self.__channel = self.__client.invoke_shell()
        self.__output = ""
        self._update_output()
        print(f"{self.ip} connected SSH")

    def command(self, command: str) -> str:
        """SSH 명령어 실행"""
        self.__channel.send(command + "\n")
        self._update_output()

    def send_file(self, local_path: str) -> None:
        """파일 전송"""
        if not self.__client.get_transport():
            print(f"{self.__ip} SSH not connected")
            return
        local_path = Path(local_path).resolve().as_posix()
        filename = Path(local_path).name
        host_path = self.__client.exec_command("pwd")[1].read().decode("utf-8").strip()
        host_path = (Path(host_path) / filename).as_posix()
        sftp = self.__client.open_sftp()
        print(f"send {local_path} to {host_path}")
        sftp.put(local_path, host_path)
        sftp.close()

    def exist_file(self, host_path: str) -> bool:
        """파일 존재 여부 확인"""
        if not self.__client.get_transport():
            print(f"{self.__ip} SSH not connected")
            return False
        host_path = Path(host_path).as_posix()
        sftp = self.__client.open_sftp()
        try:
            sftp.stat(host_path)
        except FileNotFoundError:
            return False
        return True

    def close(self):
        """SSH 접속 해제"""
        if not self.__client.get_transport():
            return
        self.__client.close()
        self.__channel = None
        self.__output = ""
        print(f"{self.ip} disconnected SSH")

    def _update_output(self) -> None:
        """SSH 출력 업데이트"""
        if not self.__channel:
            return
        while self.__channel.recv_ready():
            self.__output += self.__channel.recv(1024).decode("utf-8")
        self.__output = re.sub("\r", "", self.__output)
        if len(self.__output.split("\n")) > 1000:
            self.__output = "\n".join(self.__output.split("\n")[-1000:])
        return

    @property
    def connected(self) -> bool:
        return self.__client.get_transport() is not None

    @property
    def ip(self) -> str:
        return self.__ip

    @property
    def username(self) -> str:
        return self.__username

    @property
    def password(self) -> str:
        return self.__password

    @property
    def output(self) -> str:
        self._update_output()
        return self.__output
