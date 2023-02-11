from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule:
    def __init__(
        self, data_dir: str = "./data/mnist", batch_size: int = 10000
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        self._prepare_data()
        self.train, self.val = self._setup()

    def _prepare_data(self):
        # download / one time processing
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=False)

    def _setup(self):
        return MNIST(
            self.data_dir, train=True, transform=self.transform
        ), MNIST(self.data_dir, train=False, transform=self.transform)

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
