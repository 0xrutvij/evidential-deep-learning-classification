from typing import Callable

import torch
import torch.nn.functional as F

LossFunction = Callable[
    [torch.Tensor, torch.Tensor, int, int, int, str],
    torch.Tensor,
]


class Evidence:
    @staticmethod
    def relu(y: torch.Tensor) -> torch.Tensor:
        return F.relu(y)

    @staticmethod
    def exp(y: torch.Tensor) -> torch.Tensor:
        return torch.exp(torch.clamp(y, -10, 10))

    @staticmethod
    def softplus(y: torch.Tensor) -> torch.Tensor:
        return F.softplus(y)


class EdlLosses:
    @staticmethod
    def _kl_divergence(
        alpha: torch.Tensor, num_classes: int, device: str = "cpu"
    ) -> torch.Tensor:
        ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )

        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=1, keepdim=True)
        )
        kl = first_term + second_term
        return kl

    @staticmethod
    def _loglikelihood_loss(
        y: torch.Tensor, alpha: torch.Tensor, device: str = "cpu"
    ):
        y = y.to(device)
        alpha = alpha.to(device)
        alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum(
            (y - (alpha / alpha_sum)) ** 2, dim=1, keepdim=True
        )
        loglikelihood_var = torch.sum(
            alpha
            * (alpha_sum - alpha)
            / (alpha_sum * alpha_sum * (alpha_sum + 1)),
            dim=1,
            keepdim=True,
        )
        loglikelihood = loglikelihood_err + loglikelihood_var
        return loglikelihood

    @classmethod
    def _mse_loss(
        cls,
        y: torch.Tensor,
        alpha: torch.Tensor,
        epoch_num: int,
        num_classes: int,
        annealing_step: int,
        device: str = "cpu",
    ):
        y = y.to(device)
        alpha = alpha.to(device)
        loglikelihood = cls._loglikelihood_loss(y, alpha, device=device)

        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
        )

        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * cls._kl_divergence(
            kl_alpha, num_classes, device=device
        )
        return loglikelihood + kl_div

    @classmethod
    def _edl_loss(
        cls,
        func: Callable,
        y: torch.Tensor,
        alpha: torch.Tensor,
        epoch_num: int,
        num_classes: int,
        annealing_step: int,
        device: str = "cpu",
    ):
        y = y.to(device)
        alpha = alpha.to(device)
        alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
        A = torch.sum(y * (func(alpha_sum) - func(alpha)), dim=1, keepdim=True)

        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
        )

        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * cls._kl_divergence(
            kl_alpha, num_classes, device=device
        )
        return A + kl_div

    @classmethod
    def edl_mse_loss(
        cls,
        output: torch.Tensor,
        target: torch.Tensor,
        epoch_num: int,
        num_classes: int,
        annealing_step: int,
        device: str = "cpu",
    ):
        evidence = Evidence.relu(output)
        alpha = evidence + 1
        loss = torch.mean(
            cls._mse_loss(
                target,
                alpha,
                epoch_num,
                num_classes,
                annealing_step,
                device=device,
            )
        )
        return loss

    @classmethod
    def edl_log_loss(
        cls,
        output: torch.Tensor,
        target: torch.Tensor,
        epoch_num: int,
        num_classes: int,
        annealing_step: int,
        device: str = "cpu",
    ):
        evidence = Evidence.relu(output)
        alpha = evidence + 1
        loss = torch.mean(
            cls._edl_loss(
                torch.log,
                target,
                alpha,
                epoch_num,
                num_classes,
                annealing_step,
                device=device,
            )
        )
        return loss

    @classmethod
    def edl_digamma_loss(
        cls,
        output: torch.Tensor,
        target: torch.Tensor,
        epoch_num: int,
        num_classes: int,
        annealing_step: int,
        device: str = "cpu",
    ):
        evidence = Evidence.relu(output)
        alpha = evidence + 1
        loss = torch.mean(
            cls._edl_loss(
                torch.digamma,
                target,
                alpha,
                epoch_num,
                num_classes,
                annealing_step,
                device=device,
            )
        )
        return loss
