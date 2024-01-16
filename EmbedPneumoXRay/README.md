# Jetson 통합 시연 프로그램

12월 6일 IITP에서 진행한 흉부 X레이 진단 시연 프로그램 및 개발도구입니다.
Jetson 클아이언트와 Windows 서버로 구성되어 있습니다.
각 클라이언트는 서버를 통해 git을 이용한 동기화를 수행합니다.

## 개발환경 구성

서버 개발환경(가상 개발환경 구성):

```ps1
python -m venv _pyenv
_pyenv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

클라이언트 개발환경:

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-pyqt5

# 파이토치 및 토치비전을 별도로 설치해야 합니다
```

## Devtool: SSH Chain

개발 중 여러 대의 Jetson에 동시에 명령을 내리기 위한 Python GUI 프로그램.

`/config.yml` 설정:

- `ip_list:`의 항목에 따라 클라이언트에 접속합니다.
- `username:`과 `password:`는 클라이언트의 로그인 정보입니다.

## Demo: Pneumo Detect AI Server

흉부 X레이로부터 연산한 1차 먼볼루전 레이어 결과를 수신하고 나머지 연산을 수행하여 결과를 클라이언트에 전송하는 서버 프로그램.

`/config.yml` 설정:

- `ip_list:`의 항목에 따라 클라이언트에 접속합니다.
- `username:`과 `password:`는 클라이언트의 로그인 정보입니다.
- `token:`은 클라이언트가 사용할 git 토큰입니다.
- `repository:`는 클라이언트가 사용할 git 저장소입니다.

서버는 클라이언트를 인지하면 클라이언트 장치가 자동으로 git pull을 수행하여 소스코드를 업데이트 하고
클라이언트 프로그램을 실행하도록 합니다.

서버를 종료하면 원격으로 실행한 클라이언트도 종료됩니다.

## Demo: Pneumo Detect AI Client

선택한 흉부 X레이로부터 연산한 1차 먼볼루전 레이어 결과를 전송하는 GUI 프로그램.
해당 프로그램은 `data/`의 흉부 X레이 이미지를 무작위로 20개를 추출하여 GUI에 선택 가능한 메뉴로 표출합니다.

## VNC 설정

`DISPLAY=:0`이 활성화 되어있는 상태에서 vino-server를 실행합니다.

```bash
gsettings set org.gnome.Vino prompt-enabled false
gsettings set org.gnome.Vino require-encryption false
DISPLAY=:0 /usr/lib/vino/vino-server
```
