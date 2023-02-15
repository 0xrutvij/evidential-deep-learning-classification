from pathlib import Path
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

    def classify(self, img: torch.Tensor, filename: str):
        output: torch.Tensor
        preds: torch.Tensor

        ldeg, lp, lu, classifications = [], [], [], []

        scores = np.zeros((1, self.num_classes))
        rimgs = np.zeros((28, 28 * self.ndeg))
        for i, deg in enumerate(np.linspace(0, self.mdeg, self.ndeg)):
            nimg = rotate_img(img.numpy()[0], deg).reshape(28, 28)
            nimg = nimg.clip(min=0, max=1)

            start, stop = i * 28, (i + 1) * 28
            rimgs[:, start:stop] = nimg
            transform = transforms.ToTensor()
            img_tensor = transform(nimg)
            img_tensor.unsqueeze_(0)
            img_variable: torch.Tensor = Variable(img_tensor)
            img_variable = img_variable.to(self.device)

            output = self.model(img_variable)
            _, preds = torch.max(output, dim=1)

            if self.uncertainty:
                evidence = F.relu(output)
                alpha = evidence + 1
                alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
                uncertainty_val = self.num_classes / alpha_sum
                prob = alpha / alpha_sum
                lu.append(uncertainty_val.mean())

            else:
                prob = F.softmax(output, dim=1)

            output = output.flatten()
            prob = prob.flatten()
            preds = preds.flatten()

            classifications.append(preds[0].item())
            probs_np: np.ndarray = prob.detach().numpy()
            scores += probs_np >= self.threshold
            ldeg.append(deg)
            lp.append(prob.tolist())

        labels: np.ndarray = np.arange(10)[scores[0].astype(bool)]
        lp = np.array(lp)[:, labels]  # type: ignore
        c = ["black", "blue", "red", "brown", "purple", "cyan"]
        marker = ["s", "^", "o"] * 2
        labels = labels.tolist()
        fig = plt.figure(figsize=[6.2, 5])
        fig, axs = plt.subplot(3, gridspec_kw={"height_ratios": [4, 1, 12]})

        for i in range(len(labels)):
            axs[2].plot(
                ldeg, lp[:, i], marker=marker[i], c=c[i]  # type: ignore
            )

        if self.uncertainty:
            labels += ["uncertainty"]
            axs[2].plot(ldeg, lu, marker="<", c="red")

        print(classifications)

        axs[0].set_title('Rotated "1" Digit Classifications')
        axs[0].imshow(1 - rimgs, cmap="gray")
        axs[0].axis("off")
        plt.pause(0.001)

        axs[1].table(cellText=[classifications], bbox=[0, 1.2, 1, 1])
        axs[1].axis("off")

        axs[2].legend(labels)
        axs[2].set_xlim([0, self.mdeg])
        axs[2].set_ylim([0, 1])
        axs[2].set_xlabel("Rotation Degree")
        axs[2].set_ylabel("Classification Probability")

        plt.savefig(filename)
