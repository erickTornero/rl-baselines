import torch
from typing import Tuple, Union
from torch import nn
class ReinforceLoss:
    def __call__(
        self,
        action_probs: torch.Tensor, 
        action: torch.Tensor, 
        G: torch.Tensor
    ) -> torch.Tensor:
        J = torch.gather(action_probs, 0, action)
        J = -torch.log(J) * G
        return J.mean()

class ReinforceWithBaselineLoss:
    def __call__(
        self,
        action_probs: torch.Tensor, # must be with gradients
        baseline_value: torch.Tensor, # must be with gradients
        action: torch.Tensor,
        delta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reinforce_loss = ReinforceLoss()(
            action_probs,
            action,
            delta,
        )

        baseline_loss = -delta * baseline_value
        baseline_loss = baseline_loss.mean()

        return (reinforce_loss, baseline_loss)

class ReinforceWithActorCriticLoss:
    def __call__(
        self,
        action_probs: torch.Tensor, # must be with gradients
        state_value: torch.Tensor, # must be with gradients
        action: torch.Tensor,
        delta: torch.Tensor,
        gamma_cumm: Union[torch.Tensor, float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reinforce_loss = ReinforceLoss()(
            action_probs,
            action,
            delta,
        )
        reinforce_loss *= gamma_cumm

        critic_loss = -delta * state_value
        critic_loss = critic_loss.mean()

        return (reinforce_loss, critic_loss)

class BellmanDelta:
    def __init__(self, state_value_network: nn.Module, gamma: float) -> None:
        self.state_value_network = state_value_network
        self.gamma = gamma

    def __call__(
        self,
        observation: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:

        value_expected = reward + self.gamma * self.state_value_network(next_observation) * (1 - done.to(dtype=torch.float32))
        delta = value_expected - self.state_value_network(observation)
        return delta