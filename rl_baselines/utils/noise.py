from __future__ import annotations
import torch
from typing import Optional, Union
import math

class OrnsteinUhlenbeck:
    def __init__(
        self,
        delta: float,
        sigma: float,
        ou_a: float,
        ou_mu: float,
        seed: int,
        size: int,
        device: Union[str, torch.DeviceObjType]='cpu',
        dtype=torch.float32
    ):
        self.delta = delta
        self.sigma = sigma
        self.ou_a = ou_a
        self.ou_mu = ou_mu
        self.seed = seed
        self.size = size
        self.device = device
        self.dtype = dtype
        self._sqrt_delta_sigma = math.sqrt(self.delta) * self.sigma
    
    def _brownian_motion_log_returns(self):
        return torch.normal(
            torch.zeros(self.size, device=self.device, dtype=self.dtype),
            torch.ones(self.size, device=self.device, dtype=self.dtype) * self._sqrt_delta_sigma
        )

    def _ornstein_uhlenbeck_level(self, prev_ou_level: torch.Tensor):
        drift = self.ou_a * (self.ou_mu - prev_ou_level) * self.delta
        randomness = self._brownian_motion_log_returns()
        return prev_ou_level + drift + randomness
    
    def init(self, initial_level: Optional[torch.Tensor]=None) -> torch.Tensor:
        if initial_level is None:
            initial_level = torch.rand(self.size, dtype=self.dtype, device=self.device) - 0.5
        self._level = self._ornstein_uhlenbeck_level(initial_level)
        return self._level
    
    def __call__(self) -> torch.Tensor:
        if not hasattr(self, "_level"):
            return self.init()
        self._level = torch.clip(self._ornstein_uhlenbeck_level(self._level), -1.0, 1.0)
        return self._level
    
    def get_ornstein_noise_length(self, length) -> torch.Tensor:
        orns_noise = []
        for _ in range(length):
            level = self()
            orns_noise.append(level)
        return torch.vstack(orns_noise)
    
    def to(self, device: Optional[Union[str, torch.DeviceObjType]]=None, dtype: Optional[torch.dtype]=None) -> OrnsteinUhlenbeck:
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ornstein = OrnsteinUhlenbeck(0.6, 0.2, ou_a=1, ou_mu=0.0, seed=20, size=2, dtype=torch.float32, device=torch.device('cuda'))
    noise = ornstein.get_ornstein_noise_length(1000)
    import pdb;pdb.set_trace()
    plt.plot(noise.cpu().numpy())
    plt.ylim(-1, 1)
    plt.show()
