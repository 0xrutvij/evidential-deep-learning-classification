from typing import Optional, Type

import matplotlib.pyplot as plt
import torch

from data import AbstractDataModule, MNISTDataModule, SVHNDataModule
from lenet import LeNet
from losses import EdlLosses, LossFunction
from tester import RotatingImageClassifier, SingleImageTester
from trainer import (
    ClassificationModelTrainer,
    CrossEntropyModelTrainer,
    EvidentialTrainer,
)


def show_example(dataset_name: str = "mnist"):
    match dataset_name:
        case "mnist":
            examples = MNISTDataModule().get_data("val")
            data, targets = next(iter(examples))
            _ = plt.figure()
            for i in range(6):
                plt.subplot(2, 3, i + 1)
                plt.tight_layout()
                plt.imshow(data[i][0], cmap="gray", interpolation="none")
                plt.title(f"Ground Truth: {targets[i]}")
                plt.xticks([])
                plt.yticks([])
            plt.savefig(f"./images/examples_{dataset_name}.jpg")
        case "svhn":
            examples = SVHNDataModule().get_data("val")
            data, targets = next(iter(examples))
            _ = plt.figure()
            for i in range(6):
                plt.subplot(2, 3, i + 1)
                plt.tight_layout()
                plt.imshow(data[i].permute(1, 2, 0), interpolation="none")
                plt.title(f"Ground Truth: {targets[i]}")
                plt.xticks([])
                plt.yticks([])
            plt.savefig(f"./images/examples_{dataset_name}.jpg")


def train(
    dataset_name: Optional[str] = None,
    model_type: Optional[str] = None,
    epochs: int = 25,
    use_dropout: bool = True,
    device: str = "cpu",
):
    num_epochs = epochs
    num_classes = 10

    trainer: ClassificationModelTrainer
    criterion: LossFunction | torch.nn.CrossEntropyLoss
    model_name: str
    dataset: Type[AbstractDataModule]

    match dataset_name:
        case "svhn":
            dataset = SVHNDataModule
            channels = 3
        case "mnist" | _:
            dataset = MNISTDataModule
            channels = 1

    model = LeNet(dropout=use_dropout, channels=channels)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=5e-3
    )

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1
    )

    match model_type:
        case "digamma":
            criterion = EdlLosses.edl_digamma_loss
            model_name = "uncertainty_digamma"

        case "log":
            criterion = EdlLosses.edl_log_loss
            model_name = "uncertainty_log"
        case "mse":
            criterion = EdlLosses.edl_mse_loss
            model_name = "uncertainty_mse"
        case _:
            criterion = torch.nn.CrossEntropyLoss()
            model_name = ""

    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        trainer = CrossEntropyModelTrainer(
            model=model,
            dataloaders=dataset(),
            num_classes=num_classes,
            optimizer=optimizer,
            scheduler=exp_lr_scheduler,
            device=device,
            num_epochs=num_epochs,
            criterion=criterion,
        )

    else:
        trainer = EvidentialTrainer(
            model=model,
            dataloaders=dataset(),
            num_classes=num_classes,
            optimizer=optimizer,
            scheduler=exp_lr_scheduler,
            device=device,
            num_epochs=num_epochs,
            criterion=criterion,
        )

    save_file = (
        f"./models/model_{dataset_name}"
        f"{'_' if model_name else ''}{model_name}.pt"
    )

    model, metrics = trainer.train(num_epochs=num_epochs)

    state = {
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    torch.save(state, save_file)
    print(f"Saved {save_file}")


def test(
    dataset_name: Optional[str] = None,
    model_type: Optional[str] = None,
    use_dropout: bool = True,
    device: str = "cpu",
    show_single_img_comp: bool = False,
):

    num_classes = 10
    uncertainty = True
    model_name: str
    imshape: tuple[int, ...]

    match dataset_name:
        case "svhn":
            classif_img, _ = SVHNDataModule().val[26]
            channels = 3
            imshape = (32, 32, 3)
            threshold = 0.3
            ex_name = "4"
        case "mnist" | _:
            classif_img, _ = MNISTDataModule().val[5]
            channels = 1
            imshape = (28, 28)
            threshold = 0.5
            ex_name = "1"

    match model_type:
        case "digamma":
            model_name = "uncertainty_digamma"
            threshold -= 0.1
        case "log":
            model_name = "uncertainty_log"
            threshold -= 0.1
        case "mse":
            model_name = "uncertainty_mse"
            threshold -= 0.1
        case _:
            model_name = ""
            uncertainty = False

    model = LeNet(dropout=use_dropout, channels=channels)
    model.to(device)

    optimzer = torch.optim.Adam(model.parameters())

    load_file = (
        f"./models/model_{dataset_name}"
        f"{'_' if model_name else ''}{model_name}.pt"
    )

    save_file = (
        f"./results/rotate_{dataset_name}"
        f"{'_' if model_name else ''}{model_name}.jpg"
    )

    checkpoint = torch.load(load_file)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimzer.load_state_dict(checkpoint["optimizer_state_dict"])

    model.eval()

    ric = RotatingImageClassifier(
        model,
        uncertainty=uncertainty,
        device=device,
        num_classes=num_classes,
        threshold=threshold,
    )

    ric.classify(
        img=classif_img,
        filename=save_file,
        imshape=imshape,
        ex_name=ex_name,
        title_data=f"{model_name if model_name else 'standard'}",
    )

    if show_single_img_comp:
        sit = SingleImageTester(model=model)
        sit.test(
            "./data/one.jpg",
            uncertainty=uncertainty,
            device=device,
            num_classes=num_classes,
        )

        sit.test(
            "./data/yoda.jpg",
            uncertainty=uncertainty,
            device=device,
            num_classes=num_classes,
        )
