import os
import time
import pickle
import itertools
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from thop import profile
from PIL import Image
from PyQt5 import QtWidgets, QtGui, QtCore

import kaggle

from .models.densenet_1ch import DenseNet, densenet201

torch.backends.cudnn.benchmark = True


class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = sorted(os.listdir(root_dir))
        for label, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            for image_filename in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, image_filename))
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, image_path


class ModelThread(QtCore.QThread):
    modelReady = QtCore.pyqtSignal()
    modelStatus = QtCore.pyqtSignal(int, int)  # work, total
    modelWarmup = QtCore.pyqtSignal(float)  # work, gflops
    modelWarmupDone = QtCore.pyqtSignal()
    modelInference = QtCore.pyqtSignal(float)  # gflops
    modelInferenceDone = QtCore.pyqtSignal()

    __device: torch.device
    __model: DenseNet
    __section: torch.nn.Module
    __dataset: TestDataset
    __dataloader: DataLoader
    __flop: int

    __batch_size: int
    __run_mode: str
    __run_warmup: bool
    __run_inference: bool

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ModelThread")
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__run_mode = ""
        self.__flop = 0
        self.__batch_size = 78

    @property
    def flop(self) -> int:
        if self.__flop == 0:
            input_data = torch.randn(1, 1, 256, 256).to(self.__device)
            flops, params = profile(self.__model, inputs=(input_data,), verbose=False)
            self.__flop = flops
        return self.__flop

    def __len__(self) -> int:
        return len(self.__dataloader)

    def start_model_init(self):
        self.__run_mode = "init"
        self.start()

    def start_model_warmup(self):
        self.__run_mode = "warmup"
        self.__run_warmup = True
        self.start()

    def start_model_inference(self):
        self.__run_mode = "inference"
        self.__run_inference = True
        self.start()

    def run(self) -> None:
        if self.__run_mode == "init":
            self._model_init()
        elif self.__run_mode == "warmup":
            self._model_warmup()
        elif self.__run_mode == "inference":
            self._model_inference()
        return

    def warmup_stop(self) -> None:
        self.__run_warmup = False
        return

    def inference_stop(self) -> None:
        self.__run_inference = False
        return

    def _model_init(self):
        if not os.path.exists("./data/chest_xray/test"):
            kaggle.api.dataset_download_files(
                "paultimothymooney/chest-xray-pneumonia",
                path="./data",
                unzip=True,
            )
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        start_time = time.time()
        self.__model = densenet201(pretrained=True, num_classes=2)
        print(f"Create Model: {time.time() - start_time:.3f} sec")
        start_time = time.time()
        self.__model.load_state_dict(
            torch.load(
                "./weight_torch/ckpt_densenet201.pt", map_location=self.__device
            )["model_state_dict"],
            strict=False,
        )
        self.__model.eval()
        self.__model, _ = self._gen_model_split()
        print(f"Load Model: {time.time() - start_time:.3f} sec")

        start_time = time.time()
        self.__model.eval()
        print(f"Model Eval: {time.time() - start_time:.3f} sec")
        start_time = time.time()
        self.__model.to(self.__device)
        print(f"Model to Device: {time.time() - start_time:.3f} sec")
        start_time = time.time()
        self.__dataset = TestDataset(
            root_dir="./data/chest_xray/test", transform=transform
        )
        print(f"Load Dataset: {time.time() - start_time:.3f} sec")
        start_time = time.time()
        self.__dataloader = DataLoader(
            self.__dataset, batch_size=self.__batch_size, shuffle=True, pin_memory=True
        )
        print(f"Load Dataloader(input): {time.time() - start_time:.3f} sec")
        start_time = time.time()
        input_data = torch.randn(1, 1, 256, 256).to(self.__device)
        flops, _ = profile(self.__model, inputs=(input_data,), verbose=False)
        self.__flop = flops
        print(f"Calculate FLOP: {time.time() - start_time:.3f} sec")

        self.modelReady.emit()

    def _model_warmup(self):
        print(
            f"Model Warmup {self.__batch_size} Batch Size {self.__device} Dataset Size: {len(self.__dataset)}"
        )
        infinite_loader = itertools.cycle(self.__dataloader)
        with torch.no_grad():
            for i, (inputs, label, path_) in enumerate(infinite_loader):
                if not self.__run_warmup:
                    break
                inputs = inputs.to(self.__device)
                start_time = time.time()
                self.__model(inputs)
                timeit = time.time() - start_time
                flops = (self.flop * self.__batch_size / timeit) / 10**9
                print(
                    f"Loop {i + 1:6d}({i % len(self.__dataloader):3d}/{len(self.__dataloader)}): {timeit * 10**6:8.2f} us {flops:8.3f} GFLOPs"
                )
                self.modelWarmup.emit(flops)
        self.modelWarmupDone.emit()
        return

    def _model_inference(self):
        infinite_loader = itertools.cycle(self.__dataloader)
        max_microseconds = 18046744073709551615

        with torch.no_grad():
            for i, (inputs, label, path_) in enumerate(infinite_loader):
                if not self.__run_inference:
                    break
                inputs = inputs.to(self.__device)
                start_time = time.time()
                self.__model(inputs)
                timeit = time.time() - start_time
                flops = (self.flop * self.__batch_size / timeit) / 10**9
                if False:
                    additional_sleep_time = (flops - 150.0) / 150.0
                    sleep_time_microseconds = int(
                        timeit * 10**6 * additional_sleep_time
                    )
                    excess_microseconds = sleep_time_microseconds - max_microseconds
                    sleep_time_microseconds = max(0, sleep_time_microseconds)
                    if excess_microseconds > 0:
                        sleep_time_microseconds -= excess_microseconds
                        excess_milliseconds = excess_microseconds / 1000
                        self.msleep(excess_milliseconds)
                    self.usleep(sleep_time_microseconds)
                    timeit = time.time() - start_time
                    flops = (self.flop * self.__batch_size / timeit) / 10**9
                print(
                    f"Loop {i + 1:6d}({i % len(self.__dataloader):3d}/{len(self.__dataloader)}): {timeit * 10**6:8.2f} us {flops:8.3f} GFLOPs"
                )
                self.modelInference.emit(flops)
        self.modelInferenceDone.emit()
        return

    def _gen_model_split(self):
        first_layer_names = []

        # 첫 Max Pool 지점까지의 레이어 이름 수집
        for name, layer in self.__model.features.named_children():
            first_layer_names.append(name)
            if isinstance(layer, torch.nn.MaxPool2d):
                break

        # 1번, 2번, 3번 모델의 레이어 수집
        first_layers = []  # Input Layer ~ Max Pool Layer
        second_layer = []  # Max Pool Layer ~ DenseNet End
        third_layer = []  # DenseNet End ~ Classifier
        for name, layer in self.__model.features.named_children():
            if name in first_layer_names:
                first_layers.append(layer)
            else:
                second_layer.append(layer)
        third_layer = [
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(start_dim=1),
            self.__model.classifier,
        ]
        second_layer += third_layer

        # 각 파트 모델 생성
        first_part_model = torch.nn.Sequential(*first_layers)
        second_part_model = torch.nn.Sequential(*second_layer)
        return first_part_model, second_part_model


if __name__ == "__main__":
    workspace = Path(__file__).parent.parent
    os.chdir(workspace)
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = TestDataset(root_dir="data/test", transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for image, label, image_path in dataloader:
        print(image.shape, label, image_path)
