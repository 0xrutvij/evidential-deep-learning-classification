from typing import Callable

import torch

LossFunction = Callable[
    [torch.Tensor, torch.Tensor, int, int, int, str],
    torch.Tensor,
]
