from abc import ABC, abstractmethod
from typing import Type, final

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, SVHN, VisionDataset


class AbstractDataModule(ABC):
    @abstractmethod
    def get_dataset_class(self) -> Type[VisionDataset]:
        pass

    @abstractmethod
    def get_data_dir(self) -> str:
        pass

    @abstractmethod
    def _prepare_data(self):
        pass

    @abstractmethod
    def _setup(self) -> tuple[VisionDataset, VisionDataset]:
        pass

    @abstractmethod
    def _get_batch_size(self) -> int:
        pass

    def __init__(self) -> None:
        self.data_dir = self.get_data_dir()
        self.dataset_class = self.get_dataset_class()
        self.batch_size = self._get_batch_size()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self._prepare_data()
        self.train, self.val = self._setup()

    def get_data(self, phase: str):
        match phase:
            case "train":
                return DataLoader(
                    self.train,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=8,
                )
            case "val" | _:
                return DataLoader(
                    self.val, batch_size=self.batch_size, num_workers=8
                )


@final
class MNISTDataModule(AbstractDataModule):
    def _get_batch_size(self) -> int:
        return 10000

    def get_dataset_class(self) -> Type[VisionDataset]:
        return MNIST

    def get_data_dir(self) -> str:
        return "./data/mnist"

    def _prepare_data(self):
        # download / one time processing
        self.dataset_class(self.data_dir, train=True, download=True)
        self.dataset_class(self.data_dir, train=False, download=True)

    def _setup(self):
        return self.dataset_class(
            self.data_dir, train=True, transform=self.transform
        ), self.dataset_class(
            self.data_dir, train=False, transform=self.transform
        )


@final
class SVHNDataModule(AbstractDataModule):
    def _get_batch_size(self) -> int:
        return 2048

    def get_dataset_class(self) -> Type[VisionDataset]:
        return SVHN

    def get_data_dir(self) -> str:
        return "./data/svhn"

    def _prepare_data(self):
        # download / one time processing
        self.dataset_class(self.data_dir, split="train", download=True)
        self.dataset_class(self.data_dir, split="test", download=True)

    def _setup(self):
        return self.dataset_class(
            self.data_dir, split="train", transform=self.transform
        ), self.dataset_class(
            self.data_dir, split="test", transform=self.transform
        )
