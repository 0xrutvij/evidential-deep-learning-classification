from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import Callable, Optional, final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from data import MNISTDataModule

LossFunction = Callable[
    [torch.Tensor, torch.Tensor, int, int, int, str],
    torch.Tensor,
]


@dataclass(kw_only=True)
class Metric:
    phases: list = field(default_factory=list)
    epochs: list = field(default_factory=list)


@dataclass(kw_only=True)
class Loss(Metric):
    losses: list = field(default_factory=list)

    def update(self, loss: float, phase: str, epoch: int):
        self.losses.append(loss)
        self.phases.append(phase)
        self.epochs.append(epoch)


@dataclass(kw_only=True)
class Accuracy(Metric):
    accuracies: list = field(default_factory=list)

    def update(self, accuracy: float, phase: str, epoch: int):
        self.accuracies.append(accuracy)
        self.phases.append(phase)
        self.epochs.append(epoch)


@dataclass
class Metrics:
    accuracy: Accuracy = field(default_factory=Accuracy)
    losses: Loss = field(default_factory=Loss)


@dataclass(kw_only=True)
class ClassificationModelTrainer:
    model: nn.Module
    dataloaders: MNISTDataModule
    num_classes: int
    optimizer: Optimizer
    scheduler: Optional[_LRScheduler]
    device: str = "cpu"
    num_epochs: int = 25

    def __post_init__(self):
        self.metrics = Metrics()
        self.best_model_wts = deepcopy(self.model.state_dict())
        self.best_acc = 0.0
        self.epoch_loss = 0.0
        self.epoch_acc = 0.0

    @abstractmethod
    def _get_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, epoch: int
    ) -> torch.Tensor:
        pass

    def _update_metrics(self, phase: str, epoch: int, **metric_kwargs):
        self.metrics.losses.update(self.epoch_loss, phase, epoch)
        self.metrics.accuracy.update(self.epoch_acc, phase, epoch)

    def _unified_loop(self, phase: str, epoch: int):
        inputs: torch.Tensor
        labels: torch.Tensor
        preds: torch.Tensor
        outputs: torch.Tensor

        running_loss = 0.0
        running_corrects = 0.0
        dataloader = self.dataloaders.get_data(phase)

        match phase:
            case "train":
                print("Training...")
                self.model.train()
            case "val" | _:
                print("Validating...")
                self.model.eval()

        for _, (inputs, labels) in enumerate(dataloader):
            inputs.to(self.device)
            labels.to(self.device)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, dim=1)
                loss = self._get_loss(outputs, labels, epoch)

            if phase == "train":
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()

        if self.scheduler is not None and phase == "train":
            self.scheduler.step()

        self.epoch_loss = running_loss / len(dataloader.dataset)
        self.epoch_acc = running_corrects / len(dataloader.dataset)
        self._update_metrics(phase, epoch)
        print(
            f"{phase.capitalize()} loss: "
            f"{self.epoch_loss:.4f} acc: {self.epoch_acc:.4f}"
        )

        if phase == "val" and self.epoch_acc > self.best_acc:
            self.best_acc = self.epoch_acc
            self.best_model_wts = deepcopy(self.model.state_dict())

    def train(self, num_epochs: Optional[int] = None):
        if num_epochs is not None:
            self.num_epochs = num_epochs

        start = time()

        for epoch in tqdm(range(self.num_epochs)):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print("-" * 25)
            self._unified_loop("train", epoch)
            self._unified_loop("val", epoch)
            print()

        time_elapsed = time() - start

        print(
            f"Training Complete in {time_elapsed//60}m {time_elapsed%60:.0f}s"
        )
        print(f"Best val Acc: {self.best_acc:.4f}")

        self.model.load_state_dict(self.best_model_wts)
        return self.model, self.metrics


@final
@dataclass(kw_only=True)
class CrossEntropyModelTrainer(ClassificationModelTrainer):
    criterion = field(default_factory=nn.CrossEntropyLoss)

    def _get_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, epoch: int
    ) -> torch.Tensor:
        return self.criterion(outputs, labels)


@final
@dataclass(kw_only=True)
class EvidentialTrainer(ClassificationModelTrainer):
    criterion: LossFunction

    def _get_loss(
        self, outputs: torch.Tensor, labels: torch.Tensor, epoch: int
    ) -> torch.Tensor:
        y: torch.Tensor
        preds: torch.Tensor

        y = F.one_hot(labels, self.num_classes)
        y = y.to(self.device)
        preds = torch.max(outputs, dim=1)

        loss = self.criterion(
            outputs, y.float(), epoch, self.num_classes, 10, self.device
        )

        matches = torch.eq(preds, labels).float().reshape(-1, 1)
        acc = torch.mean(matches)
        evidence = F.relu(outputs)
        alpha = evidence + 1
        u = self.num_classes / torch.sum(alpha, dim=1, keepdim=True)

        total_evidence = torch.sum(evidence, 1, keepdim=True)
        mean_evidence = torch.mean(total_evidence)

        _ = (acc, u, mean_evidence)
        # TODO: read paper and address the following lines properly
        return loss
