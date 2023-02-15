from typing import Optional

import matplotlib.pyplot as plt
import torch

from data import MNISTDataModule
from lenet import LeNet
from losses import EdlLosses
from trainer import (
    ClassificationModelTrainer,
    CrossEntropyModelTrainer,
    EvidentialTrainer,
)


def show_example():
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
    plt.savefig("./images/examples.jpg")


def train(
    model_type: Optional[str] = None,
    epochs: int = 25,
    use_dropout: bool = True,
    device: str = "cpu",
):
    num_epochs = epochs
    num_classes = 10

    model = LeNet(dropout=use_dropout)
    model.to(device)

    optimzer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-3)

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimzer, step_size=7, gamma=0.1
    )

    trainer: ClassificationModelTrainer
    model_name: str

    match model_type:
        case "digamma":
            trainer = EvidentialTrainer(
                model=model,
                dataloaders=MNISTDataModule(),
                num_classes=num_classes,
                optimizer=optimzer,
                scheduler=exp_lr_scheduler,
                device=device,
                num_epochs=num_epochs,
                criterion=EdlLosses.edl_digamma_loss,
            )
            model_name = "uncertainty_digamma"
        case "log":
            trainer = EvidentialTrainer(
                model=model,
                dataloaders=MNISTDataModule(),
                num_classes=num_classes,
                optimizer=optimzer,
                scheduler=exp_lr_scheduler,
                device=device,
                num_epochs=num_epochs,
                criterion=EdlLosses.edl_log_loss,
            )
            model_name = "uncertainty_log"
        case "mse":
            trainer = EvidentialTrainer(
                model=model,
                dataloaders=MNISTDataModule(),
                num_classes=num_classes,
                optimizer=optimzer,
                scheduler=exp_lr_scheduler,
                device=device,
                num_epochs=num_epochs,
                criterion=EdlLosses.edl_mse_loss,
            )
            model_name = "uncertainty_mse"
        case _:
            trainer = CrossEntropyModelTrainer(
                model=model,
                dataloaders=MNISTDataModule(),
                num_classes=num_classes,
                optimizer=optimzer,
                scheduler=exp_lr_scheduler,
                device=device,
                num_epochs=num_epochs,
            )
            model_name = ""

    save_file = f"./results/model_{model_name}.pt"

    model, metrics = trainer.train(num_epochs=num_epochs)

    state = {
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimzer.state_dict(),
    }

    torch.save(state, save_file)
    print(f"Saved {save_file}")
