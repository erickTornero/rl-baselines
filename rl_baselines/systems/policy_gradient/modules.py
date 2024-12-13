import torch
from torch import nn

class TargetEstimator(nn.Module):
    def __init__(
        self,
        value_network: nn.Module,
        gamma: float,
    ):
        super().__init__()
        self.value_network = value_network
        self.gamma = gamma

    def __call__(
        self,
        reward: torch.Tensor,
        next_observation: torch.Tensor
    ) -> torch.Tensor:
        target = reward + self.gamma * self.value_network(next_observation)
        return target

class AdvantageEstimator(TargetEstimator):
    def __init__(
        self, 
        value_network: nn.Module,
        gamma: float,
    ):
        super().__init__(value_network, gamma)

    def __call__(
        self,
        reward: torch.Tensor,
        observation: torch.Tensor,
        next_observation: torch.Tensor
    ) -> torch.Tensor:
        target = super().__call__(reward, next_observation)
        state_value = self.value_network(observation)
        return target  - state_value

class TDError(TargetEstimator):
    def __init__(
        self, 
        value_network: nn.Module,
        gamma: float,
    ):
        super().__init__(value_network, gamma)

    def __call__(
        self,
        reward: torch.Tensor,
        observation: torch.Tensor,
        next_observation: torch.Tensor
    ) -> torch.Tensor:
        target = super().__call__(reward, next_observation)
        state_value = self.value_network(observation)
        return target  - state_value