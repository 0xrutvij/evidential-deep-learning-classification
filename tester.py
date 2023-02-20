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
            title += f", Uncertainty: {uncertainty_val.detach().cpu().item()}"

        plt.title(title)
        axs[0].set_title("One")
        axs[0].imshow(img, cmap="gray")
        axs[0].axis("off")

        axs[1].bar(labels, prob.detach().cpu().numpy(), width=0.5)
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

    def classify(
        self,
        img: torch.Tensor,
        filename: str,
        ex_name: str,
        imshape: tuple[int, ...],
        title_data: str,
    ):
        output: torch.Tensor
        preds: torch.Tensor

        degree_ls, prob_ls, uncertainty_ls, classifications = [], [], [], []
        scores = np.zeros((1, self.num_classes))
        imdim = len(imshape)

        if imdim == 2:
            rimgs = np.zeros((imshape[0], imshape[1] * self.ndeg))
            img_np = img.numpy()[0]
        elif imdim == 3:
            rimgs = np.zeros((imshape[0], imshape[1] * self.ndeg, imshape[2]))
            img_np = img.permute(1, 2, 0).numpy()

        transform = transforms.ToTensor()

        for i, deg in enumerate(np.linspace(0, self.mdeg, self.ndeg)):
            nimg = (
                rotate_img(img_np, deg, imshape=imshape)
                .reshape(*imshape)
                .clip(min=0, max=1)
            )

            start, stop = i * imshape[0], (i + 1) * imshape[0]

            if imdim == 2:
                rimgs[:, start:stop] = nimg
                img_tensor = transform(nimg).unsqueeze_(0).to(self.device)
            else:
                rimgs[:, start:stop, :] = nimg
                img_tensor = transform(nimg).unsqueeze_(0).to(self.device)

            output = self.model(img_tensor)
            _, preds = torch.max(output, dim=1)

            if self.uncertainty:
                evidence = F.relu(output)
                alpha = evidence + 1
                alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
                uncertainty_val = self.num_classes / alpha_sum
                prob = alpha / alpha_sum
                uncertainty_ls.append(uncertainty_val.mean().detach().cpu())

            else:
                prob = F.softmax(output, dim=1)

            output = output.flatten()
            prob = prob.flatten()
            preds = preds.flatten()

            classifications.append(preds[0].item())
            probs_np: np.ndarray = prob.detach().cpu().numpy()
            scores += probs_np >= self.threshold
            degree_ls.append(deg)
            prob_ls.append(prob.tolist())

        labels: np.ndarray = np.arange(10)[scores[0].astype(bool)]
        prob_ls = np.array(prob_ls)[:, labels]  # type: ignore
        c = ["black", "blue", "red", "brown", "purple", "cyan"]
        marker = ["s", "^", "o"] * 2
        labels = labels.tolist()
        fig, axs = plt.subplots(
            3, gridspec_kw={"height_ratios": [4, 1, 12]}, figsize=[6.2, 5]
        )

        for i in range(len(labels)):
            axs[2].plot(
                degree_ls,
                prob_ls[:, i],  # type: ignore
                marker=marker[i],
                c=c[i],
            )

        if self.uncertainty:
            labels += ["uncertainty"]
            axs[2].plot(degree_ls, uncertainty_ls, marker="<", c="red")

        print(classifications)

        empty_lst = []
        empty_lst.append(classifications)
        axs[1].table(cellText=empty_lst, bbox=[0, 1.2, 1, 1])
        axs[1].axis("off")

        axs[2].legend(labels)
        axs[2].set_xlim([0, self.mdeg])
        axs[2].set_ylim([0, 1])
        axs[2].set_xlabel("Rotation Degree")
        axs[2].set_ylabel("Classification Probability")
        axs[0].set_title(
            f"Rotated {ex_name} Digit Classifications: {title_data}"
        )
        axs[0].axis("off")
        axs[0].imshow(1 - rimgs, cmap="gray")

        plt.show()

        fig.savefig(filename)
