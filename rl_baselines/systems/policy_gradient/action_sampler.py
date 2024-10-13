import torch
from torch.distributions import Categorical, Normal
from torchrl.data import TensorSpec, OneHotDiscreteTensorSpec

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

class ContinuousSampler:
    def __init__(self, action_spec: TensorSpec) -> None:
        self.action_spec = action_spec
    def __call__(
        self,
        mean_action: torch.Tensor,
        std_action: torch.Tensor
    ) -> torch.Tensor:
        sampler = Normal(mean_action, std_action)
        action = sampler.sample()
        # cliping
        action = torch.clip(
            action,
            min=self.action_spec.space.low.to(action.device),
            max=self.action_spec.space.high.to(action.device)
        )

        return action

class ContinuousLogVarSampler:
    def __init__(self, action_spec: TensorSpec) -> None:
        self.action_spec = action_spec
    def __call__(
        self,
        mean_action: torch.Tensor,
        logvar_action: torch.Tensor,
        max_logvar: torch.Tensor,
        min_logvar: torch.Tensor,
    ) -> torch.Tensor:
        from rl_baselines.common.distributions import NormalLogVar
        sampler = NormalLogVar(mean_action, logvar_action, max_logvar, min_logvar)
        action = sampler.sample()
        # cliping
        action = torch.clip(
            action,
            min=self.action_spec.space.low.to(action.device),
            max=self.action_spec.space.high.to(action.device)
        )

        return action