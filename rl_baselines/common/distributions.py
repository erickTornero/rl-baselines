from torch.distributions import ExponentialFamily
from torch.distributions.utils import broadcast_all
import torch
import torch.nn as nn
import numpy as np
from numbers import Number
from typing import Tuple
class TrainableNormalLogVar(nn.Module):
    def __init__(self, ndim: int) -> None:
        super().__init__()
        self.max_logvar = torch.nn.Parameter(
            torch.tensor(
                1 * np.ones([ndim]),
                dtype=torch.float32,
                requires_grad=True
            )
        )
        self.min_logvar = torch.nn.Parameter(
            torch.tensor(
                -1 * np.ones([ndim]), 
                dtype=torch.float32, 
                requires_grad=True
            )
        )
    
    def __call__(self, example_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if example_tensor.ndim > self.min_logvar.ndim:
            batch_size = example_tensor.shape[0]
            min_logvar = self.min_logvar.unsqueeze(0).repeat(batch_size, 1)
            max_logvar = self.max_logvar.unsqueeze(0).repeat(batch_size, 1)
        else:
            min_logvar = self.min_logvar
            max_logvar = self.max_logvar
        return (min_logvar, max_logvar)

class NormalLogVar(ExponentialFamily):
    def __init__(
        self,
        loc: torch.Tensor,
        logvar: torch.Tensor,
        max_logvar: torch.Tensor,
        min_logvar: torch.Tensor,
        validate_args=None
    ):
        self.loc, self.logvar, self.max_logvar, self.min_logvar = broadcast_all(
            loc,
            logvar,
            max_logvar,
            min_logvar
        )
        if isinstance(loc, Number) and isinstance(logvar, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def softplus_raw(self, x: torch.Tensor):
        # Performs the elementwise softplus on the input
        # softplus(x) = 1/B * log(1+exp(B*x))
        #with torch.no_grad():
        B = torch.tensor(1, dtype=torch.float)
        return (torch.log(1 + torch.exp(x.mul_(B)))).div_(B)
    
    def log_loss(self, value: torch.Tensor) -> torch.Tensor:
        logvar = self.clamp_logvar(self.logvar)
        inv_var = torch.exp(-logvar)
        A = torch.mean(torch.sum(((self.loc - value)** 2) * (inv_var), dim=1), dim=0)
        B = torch.mean(torch.sum(logvar, dim=1), dim=0)
        return A + B

    def clamp_logvar(self, logvar: torch.Tensor) -> torch.Tensor:
        ulogvar = self.max_logvar - self.softplus_raw(self.max_logvar - logvar)
        ulogvar = self.min_logvar + self.softplus_raw(ulogvar - self.min_logvar)
        return ulogvar

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            logvar = self.clamp_logvar(self.logvar)
            std = torch.sqrt(torch.exp(logvar))
            return torch.normal(mean=self.loc.expand(shape), std=std.expand(shape))