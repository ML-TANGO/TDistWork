# [Jetson] 입력연산 시험평가 프로그램

1 초당 처리하는 부동소수점 연산량(FLOPs) 평균을 계산하기 위한 응용프로그램

<!-- @import "[TOC]" {cmd="toc" depthFrom=2 depthTo=3 orderedList=true} -->

<div style="background: #88888830;padding: 15px 5px;">
<!-- code_chunk_output -->

1. [응용프로그램 정보](#응용프로그램-정보)
    1. [실행 환경: 하드웨어](#실행-환경-하드웨어)
    2. [실행 환경: OS 및 종속성](#실행-환경-os-및-종속성)
    3. [실행 방법](#실행-방법)
2. [소스코드 정보](#소스코드-정보)
    1. [언어 및 개발종속성](#언어-및-개발종속성)
    2. [빌드 및 실행](#빌드-및-실행)
3. [유틸리티](#유틸리티)
    1. [시험 파일 리스트 생성](#시험-파일-리스트-생성)
    2. [FLOPs 계산](#flops-계산)
4. [시작 프로그램 등록](#시작-프로그램-등록)

<!-- /code_chunk_output -->
</div>

## 응용프로그램 정보

### 실행 환경: 하드웨어

- NVIDIA Jetson Orin Nano (8GB)
- 저장공간: 32GB 이상의 SD 카드 혹은 NVMe SSD
- 7W 제한 모드 및 MAXN 모드에서 동작 가능
- 처음 실행시 시험 데이터셋 다운로드를 위해 외부네트워크가 필요함

### 실행 환경: OS 및 종속성

- Ubuntu 20.04 LTS 기반 L4T(Linux for Tegra) R35.4.1
- JetPack 5.1.2
  - CUDA 11.4
  - TensorRT 8.0.1
  - OpenCV 4.5.4

```bash
# L4T 버전 확인
$ dpkg-query --show nvidia-l4t-core
nvidia-l4t-core 35.4.1-20230801124926
```

## 실행

### kaggle API Key 등록

[Kaggle](https://www.kaggle.com/)에서 API Key를 발급받아 `~/.kaggle/kaggle.json`에 저장한다.

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 종속성 설치

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip

cd EmbedPneumoXRay
pip install -U pip
python -m venv --system-site-packages _pyenv
source _pyenv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 소스코드 정보

### 언어 및 개발종속성

- Python 3.8+
- PyTorch 2.0.0
- ONNX 1.10.1
- pyqtgraph 0.13.3
- PyQt5 5.15.4

