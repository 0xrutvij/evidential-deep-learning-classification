from pathlib import Path
from re import S
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

from utils import rotate_img


class SingleImageTester:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def test(
        self,
        img_path: str,
        uncertainty: bool = False,
        device: str = "cpu",
        num_classes: int = 10,
    ):

        img_tensor: torch.Tensor
        output: torch.Tensor
        preds: torch.Tensor
        uncertainty_val: Optional[torch.Tensor] = None

        img = Image.open(img_path).convert("L")
        transform = transforms.Compose(
            [transforms.Resize((28, 28)), transforms.ToTensor()]
        )
        img_tensor = transform(img)
        img_tensor.unsqueeze_(0)
        img_variable: torch.Tensor = Variable(img_tensor)
        img_variable = img_variable.to(device)

        output = self.model(img_variable)
        _, preds = torch.max(output, 1)

        if uncertainty:
            evidence = F.relu(output)
            alpha = evidence + 1
            uncertainty_val = num_classes / torch.sum(
                alpha, dim=1, keepdim=True
            )
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)

            output = output.flatten()
            prob = prob.flatten()
            preds = preds.flatten()

        else:
            prob = F.softmax(output, dim=1)

        output = output.flatten()
        prob = prob.flatten()
        preds = preds.flatten()

        output_msg = f"""
        Predict:     {preds[0]}
        Probs:       {prob}
        """

        output_msg = (
            output_msg
            if uncertainty_val is None
            else output_msg + "\nUncertainty: {uncertainty}"
        )

        labels = torch.arange(10)
        _ = plt.figure(figsize=[6.2, 5])
        fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 3]})

        title = f"Classified as {preds[0]}"

        if uncertainty_val is not None:
            title += f", Uncertainty: {uncertainty_val.item()}"

        plt.title(title)
        axs[0].set_title("One")
        axs[0].imshow(img, cmap="gray")
        axs[0].axis("off")

        axs[1].bar(labels, prob.detach().numpy(), width=0.5)
        axs[1].set_xlim([0, 9])
        axs[1].set_ylim([0, 1])
        axs[1].set_xticks(labels)
        axs[1].set_xlabel("Classes")
        axs[1].set_ylabel("Classification Probability")

        fig.tight_layout()

        plt.savefig(f"./results/{Path(img_path).stem}")


class RotatingImageClassifier:
    def __init__(
        self,
        model: nn.Module,
        uncertainty: bool = False,
        threshold: float = 0.5,
        device: str = "cpu",
        num_classes: int = 10,
    ) -> None:

        self.model = model
        self.uncertainty = uncertainty
        self.threshold = threshold
        self.device = device
        self.num_classes = num_classes
        self.mdeg = 180
        self.ndeg = self.mdeg // 10 + 1

    def classify(self):
        ldeg, lp, lu, classification = [], [], [], []

        scores = np.zeros((1, num_classes))
        rimgs = np.zeros((28, 28 * ndeg))
        for i, deg in enumerate(np.linspace(0, mdeg, ndeg)):
            nimg = rotate_img()
