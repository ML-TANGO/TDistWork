from pathlib import Path
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal as Signal

from PIL import Image
import numpy as np

import torch
from torchvision import transforms

from model import DenseNet, densenet201

WORK_DIR = Path(__file__).parent.parent


class ModelThread(QThread):
    __using_origin: bool = False
    __model_origin: DenseNet
    __model_partial: torch.nn.Module
    __image: np.ndarray
    __result: torch.Tensor
    __device: str = "cuda:0"

    modelResult = Signal(torch.Tensor)

    def __init__(self) -> None:
        """모델 스레드"""
        super().__init__()
        self.__result = (np.array([]), np.array([]))
        self.__image = None
        self.__model_origin = None
        self.__model_partial = None
        self.__device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._init_model_partial()
        # self.start(WORK_DIR / "data/NORMAL/IM-0001-0001.jpeg")

    def __call__(self, img: str, use_origin: bool = False) -> torch.Tensor:
        """모델 연산"""
        self.start(img, use_origin)
        self.wait()
        return self.get_result()

    def _init_model_origin(self) -> None:
        """원본 모델 초기화"""
        if not self.__model_origin is None:
            return
        weight_path = str(WORK_DIR / "model/ckpt_densenet201.pt")
        load_info = torch.load(weight_path, map_location=self.__device)
        model_state_dict = load_info["model_state_dict"]
        print(f"load model on {self.__device}")
        print(f"model path: {weight_path}")
        self.__model_origin = densenet201(pretrained=True, num_classes=2)
        self.__model_origin.to(self.__device)
        self.__model_origin.load_state_dict(model_state_dict, strict=False)
        self.__model_origin.eval()
        print("model loaded")

    def _init_model_partial(self) -> None:
        """분할 모델 초기화"""
        if not self.__model_partial is None:
            return
        weight_path = str(WORK_DIR / "model/ckpt_densenet201_partial_1.pt")
        weight_path_2 = str(WORK_DIR / "model/ckpt_densenet201_partial_2.pt")

        if Path(weight_path).exists():
            # 사전 분할한 학습모델 로드
            self.__model_partial = torch.load(weight_path)
            self.__model_partial.to(self.__device)
            self.__model_partial.eval()
            # 모델 디버그
            # self.__model_partial2 = torch.load(weight_path_2)
            # self.__model_partial2.to(self.__device)
            # self.__model_partial2.eval()
            print("partial model loaded")
            return

        self._init_model_origin()
        first_layers = []  # Input Layer ~ Max Pool Layer
        second_layers = []  # Max Pool Layer ~ Output Layer
        first_layer_names = []
        for name, module in self.__model_origin.features.named_modules():
            if name == "":
                continue
            first_layer_names.append(name)
            if isinstance(module, torch.nn.MaxPool2d):
                break
        for name, module in self.__model_origin.features.named_children():
            if name in first_layer_names:
                first_layers.append(module)
            else:
                second_layers.append(module)
        print(f"first layers: {first_layer_names}")
        second_layers += [
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(start_dim=1),
            self.__model_origin.classifier,
        ]
        self.__model_partial = torch.nn.Sequential(*first_layers)
        self.__model_partial.to(self.__device)
        self.__model_partial.eval()
        self.__model_partial2 = torch.nn.Sequential(*second_layers)
        self.__model_partial2.to(self.__device)
        self.__model_partial2.eval()
        torch.save(self.__model_partial, weight_path)
        torch.save(self.__model_partial2, weight_path_2)
        return

    def start(self, image_path: str, using_origin: bool = False):
        """스레드 시작"""
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        with Image.open(image_path) as image:
            image = image.convert("L")
            self.__image = transform(image).unsqueeze(0)
        self.__using_origin = using_origin
        super().start()

    def run(self):
        """모델 연산"""
        if self.__image is None:
            return
        if self.__using_origin:
            self._init_model_origin()
            with torch.no_grad():
                self.__image = self.__image.to(self.__device)
                result = self.__model_origin(self.__image)
                self.__result = result.argmax(dim=1).cpu()
                print(f"result: {self.__result} ({result})")
            self.modelResult.emit(self.__result)
        else:
            self._init_model_partial()
            with torch.no_grad():
                self.__image = self.__image.to(self.__device)
                result = self.__model_partial(self.__image)
                self.__result = result.cpu()
                print(f"result: {self.__result.shape}")
                # 모델 디버그
                # result = self.__model_partial2(result)
                # self.__result = result.argmax(dim=1).cpu()
                # print(f"result: {self.__result} ({result})")
            self.modelResult.emit(self.__result)

    def get_result(self) -> torch.Tensor:
        """결과 반환"""
        return self.__result
