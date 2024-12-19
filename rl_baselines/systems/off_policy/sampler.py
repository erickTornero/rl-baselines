from __future__ import annotations
import torch
from torch.distributions import Categorical
from torchrl.data import TensorSpec, OneHotDiscreteTensorSpec
from typing import Callable, Optional, Union
class CategoricalSampler:
    def __init__(
        self, 
        action_spec: TensorSpec, 
        return_onehot: bool=False
    ) -> None:
        self.action_spec = action_spec
        self._return_onehot = return_onehot

    def __call__(self, action_probs: torch.Tensor) -> torch.Tensor:
        action = Categorical(action_probs).sample()
        if self._return_onehot and isinstance(self.action_spec, OneHotDiscreteTensorSpec):
            action = torch.nn.functional.one_hot(action, num_classes=self.action_spec.space.n)
        # uncomment following line to allow gymenv environment
        #action = torch.nn.functional.one_hot(action, num_classes=2)
        return action

class ContinuousExplorationDDPGSampler:
    """
        add noise to an action in continuous setting
    """
    def __init__(
        self,
        noise_process: Callable[[], torch.Tensor],
    ):
        self.noise_process = noise_process

    def __call__(self, action_mean: torch.Tensor) -> torch.Tensor:
        noise = self.noise_process()
        return action_mean + noise

    def reset_noise(self):
        self.noise_process.init()


class ContinuousExplorationTD3Sampler:
    """
        add noise to an action in continuous setting
    """
    def __init__(
        self,
        noise_process: Callable[[Optional[int]], torch.Tensor],
    ):
        self.noise_process = noise_process

    def __call__(self, action_mean: torch.Tensor) -> torch.Tensor:
        batch_size = action_mean.shape[0] if action_mean.ndim == 2 else None
        noise = self.noise_process(batch_size)
        return action_mean + noise

    def reset_noise(self):
        self.noise_process.init()


class ActionContinuousClamper:
    def __init__(
        self,
        action_spec: TensorSpec,
    ):
        self.action_spec = action_spec

    def __call__(self, action):
        return torch.clamp(action, self.action_spec.low, self.action_spec.high)

    def to(self, device: Optional[Union[torch.DeviceObjType, str]]=None) -> ActionContinuousClamper:
        self.action_spec = self.action_spec.to(device)